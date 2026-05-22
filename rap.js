// rap.js — RAP mesoanalysis HTTP server (CAPE, CIN, 0-6km bulk shear, 0-1km SRH, Fixed SigTor)
//
// Pipeline:
//   1. Every 10 min, fetch the latest available RAP awp13f00.grib2 from NOMADS,
//      using the filter_rap.pl subregion filter to keep the download ~15 MB.
//   2. Spawn worker_threads (./rap-worker.js) — one per (shortName, level)
//      combo — that shell out to `grib_get_data` (eccodes CLI) and parse the
//      lat/lon/value rows. Up to 8 workers run in parallel across the
//      8-vCPU Railway instance.
//   3. Compute derived fields:
//      - shear:  0-6km bulk shear from |V500mb - V10m|
//      - srh:    0-1km storm-relative helicity (HLCY, heightAboveGroundLayer level=1000)
//      - sigtor: simplified STP = (CAPE/1500)×(0-1km SRH/150)×(BWD/20), each ∈[0,4]
//   4. Cache fields in memory keyed by run time; serve via NDJSON endpoints.
//
// NOTE: deviates from the user's NDJSON spec by emitting a leading
//   {type:"meta", ...} line. Without it iOS can't size a single MTLTexture
//   upload — the meta tells it (nx, ny, lat/lon bbox, valid_time).
//
// Endpoints:
//   GET /rap/cape   → NDJSON: meta line + {lat,lon,value} rows + {type:"end"}
//   GET /rap/cin    → same
//   GET /rap/shear  → same
//   GET /rap/srh    → same
//   GET /rap/sigtor → same
//   GET /healthz    → "ok"
//
// Deployment: Railway runtime image needs the eccodes binary on PATH. On a
// Debian-based image:    apt-get install -y libeccodes-tools
// Run with:              node rap.js

'use strict';

const http  = require('http');
const https = require('https');
const fs    = require('fs');
const path  = require('path');
const os    = require('os');
const { Worker } = require('worker_threads');

const PORT        = parseInt(process.env.PORT || '3000', 10);
const NUM_WORKERS = 8;  // pool size; current refresh uses 7 tasks
const REFRESH_MS  = 10 * 60 * 1000;
const TMP_DIR     = path.join(os.tmpdir(), 'rap-cache');
fs.mkdirSync(TMP_DIR, { recursive: true });

// Sector definitions. The `id` matches the iOS RAPSector.id values; the bbox
// is passed straight into the NOMADS filter_rap.pl subregion params so each
// sector downloads only the GRIB2 bytes inside its window — small sectors
// shrink the per-sector fetch + parse dramatically.
// Sector IDs and bboxes match the real NESDIS GOES-East ABI sub-sector
// definitions used by NOAA STAR (https://www.star.nesdis.noaa.gov/GOES/).
// The iOS client sends ?sector=<id> from the same list so RAP data is
// clipped to the exact bbox of the satellite imagery displayed alongside it.
const SECTORS = {
    "CONUS": { name: "CONUS",                latMin: 23.5, latMax: 50.0, lonMin: -125.0, lonMax: -66.5 },
    "ne":    { name: "Northeast",            latMin: 36.0, latMax: 50.0, lonMin: -82.0,  lonMax: -65.0 },
    "eus":   { name: "U.S. Atlantic Coast",  latMin: 28.0, latMax: 47.0, lonMin: -85.0,  lonMax: -65.0 },
    "se":    { name: "Southeast",            latMin: 26.0, latMax: 38.0, lonMin: -94.0,  lonMax: -75.0 },
    "mex":   { name: "Mexico / Gulf",        latMin: 14.0, latMax: 33.0, lonMin: -118.0, lonMax: -86.0 },
    "car":   { name: "Caribbean",            latMin: 10.0, latMax: 25.0, lonMin: -90.0,  lonMax: -60.0 },
    "cgl":   { name: "Central Great Lakes",  latMin: 38.0, latMax: 50.0, lonMin: -94.0,  lonMax: -76.0 },
    "umv":   { name: "Upper Mississippi",    latMin: 36.0, latMax: 49.0, lonMin: -99.5,  lonMax: -82.5 },
    "smv":   { name: "Southern Mississippi", latMin: 28.0, latMax: 41.0, lonMin: -98.0,  lonMax: -82.0 },
    "sp":    { name: "Southern Plains",      latMin: 26.5, latMax: 39.5, lonMin: -107.0, lonMax: -91.0 },
    "nr":    { name: "Northern Rockies",     latMin: 38.0, latMax: 51.0, lonMin: -120.0, lonMax: -100.0 },
    "sr":    { name: "Southern Rockies",     latMin: 27.0, latMax: 42.0, lonMin: -116.0, lonMax: -96.0 },
    "pnw":   { name: "Pacific Northwest",    latMin: 40.0, latMax: 52.0, lonMin: -131.0, lonMax: -111.0 },
    "psw":   { name: "Pacific Southwest",    latMin: 28.0, latMax: 42.0, lonMin: -126.0, lonMax: -106.0 },
};

function newCacheEntry() {
    return {
        validTime: null,
        meta: null,
        cape: null, cin: null, shear: null,
        srh: null, sigtor: null,
        loading: false,
        lastError: null,
        lastAttempt: null,
        refreshCount: 0,
    };
}
// One cache per sector. Populated lazily on first request.
const sectorCache = {};
// Tracks most-recently-requested sectors so the background refresh loop
// keeps them warm. CONUS is always primed.
const activeSectors = new Set(["CONUS"]);

function pad(n) { return String(n).padStart(2, '0'); }

// Main fields: CAPE, CIN, 500mb+10m winds, 500mb ABSV.
// HLCY is intentionally excluded here — NOMADS level filters (lev_surface,
// lev_500_mb, lev_10_m_above_ground) only pass messages whose typeOfLevel
// matches one of those level selectors.  HLCY uses typeOfLevel=heightAboveGroundLayer
// which matches NONE of them, so it would be silently dropped.  It is fetched
// separately via hlcyURLForHour() with no level filter.
function rapURLForHour(date, sector) {
    const yyyy = date.getUTCFullYear();
    const ymd  = `${yyyy}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}`;
    const hh   = pad(date.getUTCHours());
    const params = new URLSearchParams({
        file: `rap.t${hh}z.awp130pgrbf00.grib2`,
        lev_surface: 'on',
        lev_500_mb: 'on',
        lev_10_m_above_ground: 'on',
        var_CAPE: 'on',
        var_CIN: 'on',
        var_UGRD: 'on',
        var_VGRD: 'on',
        subregion: '',
        leftlon:   String(sector.lonMin),
        rightlon:  String(sector.lonMax),
        toplat:    String(sector.latMax),
        bottomlat: String(sector.latMin),
        dir: `/rap.${ymd}`,
    });
    return `https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl?${params.toString()}`;
}

// Separate small download for HLCY only — no level filter so NOMADS passes
// all heightAboveGroundLayer HLCY messages (0-1km and 0-3km).
function hlcyURLForHour(date, sector) {
    const yyyy = date.getUTCFullYear();
    const ymd  = `${yyyy}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}`;
    const hh   = pad(date.getUTCHours());
    const params = new URLSearchParams({
        file: `rap.t${hh}z.awp130pgrbf00.grib2`,
        var_HLCY: 'on',
        subregion: '',
        leftlon:   String(sector.lonMin),
        rightlon:  String(sector.lonMax),
        toplat:    String(sector.latMax),
        bottomlat: String(sector.latMin),
        dir: `/rap.${ymd}`,
    });
    return `https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl?${params.toString()}`;
}

function downloadToFile(url, destPath, redirects = 0) {
    return new Promise((resolve, reject) => {
        if (redirects > 5) return reject(new Error('too many redirects'));
        https.get(url, res => {
            if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                res.resume();
                downloadToFile(res.headers.location, destPath, redirects + 1).then(resolve, reject);
                return;
            }
            if (res.statusCode !== 200) {
                res.resume();
                reject(new Error(`HTTP ${res.statusCode} from ${url}`));
                return;
            }
            const file = fs.createWriteStream(destPath);
            res.pipe(file);
            file.on('finish', () => file.close(() => resolve()));
            file.on('error', reject);
        }).on('error', reject);
    });
}

function runExtraction(task) {
    return new Promise((resolve, reject) => {
        const w = new Worker(path.join(__dirname, 'rap-worker.js'), { workerData: task });
        w.on('message', msg => {
            if (msg.error) reject(new Error(msg.error));
            else resolve(msg);
        });
        w.on('error', reject);
        w.on('exit', code => { if (code !== 0) reject(new Error(`worker exit ${code}`)); });
    });
}

function inferGridMeta(points) {
    if (!points || !points.length) return null;
    let lat_min = Infinity, lat_max = -Infinity;
    let lon_min = Infinity, lon_max = -Infinity;
    const lats = new Set();
    const lons = new Set();
    for (const p of points) {
        if (p.lat < lat_min) lat_min = p.lat;
        if (p.lat > lat_max) lat_max = p.lat;
        if (p.lon < lon_min) lon_min = p.lon;
        if (p.lon > lon_max) lon_max = p.lon;
        // Bin to nearest 0.125° (multiply by 8) so the set count approximates
        // the native RAP-130 grid spacing (~0.13°). Using 0.01° bins produced
        // nx/ny ~13× too large (22M cells), causing iOS to OOM and freeze.
        lats.add(Math.round(p.lat * 8));
        lons.add(Math.round(p.lon * 8));
    }
    return { nx: lons.size, ny: lats.size, lat_min, lat_max, lon_min, lon_max };
}

async function refreshSector(sectorId) {
    const sector = SECTORS[sectorId];
    if (!sector) return;
    if (!sectorCache[sectorId]) sectorCache[sectorId] = newCacheEntry();
    const cache = sectorCache[sectorId];
    if (cache.loading) return;
    cache.loading = true;
    cache.lastAttempt = new Date().toISOString();
    cache.refreshCount++;
    try {
        // Try the most recent 4 hours in descending order — NOMADS publishes
        // ~30-45 min after the run hour; the current hour may not yet exist.
        // Two downloads per run: main (CAPE/CIN/winds/ABSV) + hlcy (SRH only).
        const now = new Date();
        now.setUTCMinutes(0, 0, 0);
        let downloaded = null;
        for (let i = 0; i < 4; i++) {
            const t      = new Date(now.getTime() - i * 3600 * 1000);
            const stamp  = t.toISOString().replace(/[:.]/g, '_');
            const mainDest = path.join(TMP_DIR, `rap_${sectorId}_${stamp}.grib2`);
            const hlcyDest = path.join(TMP_DIR, `rap_hlcy_${sectorId}_${stamp}.grib2`);
            try {
                // Download both files concurrently; skip if already cached on disk.
                await Promise.all([
                    (fs.existsSync(mainDest) && fs.statSync(mainDest).size > 1000)
                        ? Promise.resolve()
                        : downloadToFile(rapURLForHour(t, sector), mainDest),
                    (fs.existsSync(hlcyDest) && fs.statSync(hlcyDest).size > 1000)
                        ? Promise.resolve()
                        : downloadToFile(hlcyURLForHour(t, sector), hlcyDest),
                ]);
                if (fs.statSync(mainDest).size > 1000) {
                    downloaded = { date: t, path: mainDest, hlcyPath: hlcyDest };
                    break;
                }
            } catch (e) {
                console.warn(`[rap:${sectorId}] fetch ${t.toISOString()} failed: ${e.message}`);
            }
        }
        if (!downloaded) {
            console.warn(`[rap:${sectorId}] no run available yet`);
            return;
        }

        // 7 parallel field extractions split across two GRIB2 files:
        //   mainPath  — CAPE, CIN, winds  (lev_surface/500mb/10m filters)
        //   hlcyPath  — HLCY only         (no level filter; needed because
        //               heightAboveGroundLayer messages are silently dropped when
        //               any lev_* filter is active in the NOMADS request)
        const mainPath = downloaded.path;
        const hlcyPath = downloaded.hlcyPath;
        const tasks = [
            { gribPath: mainPath, tag: 'cape',  candidates: [
                { shortName: 'cape', typeOfLevel: 'surface', level: 0 },
            ]},
            { gribPath: mainPath, tag: 'cin',   candidates: [
                { shortName: 'cin',  typeOfLevel: 'surface', level: 0 },
            ]},
            { gribPath: mainPath, tag: 'u500',  candidates: [
                { shortName: 'u',    typeOfLevel: 'isobaricInhPa', level: 500 },
                { shortName: 'UGRD', typeOfLevel: 'isobaricInhPa', level: 500 },
            ]},
            { gribPath: mainPath, tag: 'v500',  candidates: [
                { shortName: 'v',    typeOfLevel: 'isobaricInhPa', level: 500 },
                { shortName: 'VGRD', typeOfLevel: 'isobaricInhPa', level: 500 },
            ]},
            { gribPath: mainPath, tag: 'u10',   candidates: [
                { shortName: '10u' },
                { shortName: 'u',    typeOfLevel: 'heightAboveGround', level: 10 },
                { shortName: 'UGRD', typeOfLevel: 'heightAboveGround', level: 10 },
            ]},
            { gribPath: mainPath, tag: 'v10',   candidates: [
                { shortName: '10v' },
                { shortName: 'v',    typeOfLevel: 'heightAboveGround', level: 10 },
                { shortName: 'VGRD', typeOfLevel: 'heightAboveGround', level: 10 },
            ]},
            // 0-1km SRH read from the dedicated HLCY-only file.
            { gribPath: hlcyPath, tag: 'srh1',  candidates: [
                { shortName: 'hlcy', typeOfLevel: 'heightAboveGroundLayer', level: 1000 },
            ]},
        ];
        const results = await Promise.all(tasks.map(t =>
            runExtraction({ gribPath: t.gribPath, tag: t.tag, candidates: t.candidates })));
        const byTag = {};
        results.forEach((r, i) => {
            byTag[tasks[i].tag] = r.points || [];
            if ((r.points || []).length === 0) {
                console.warn(`[rap:${sectorId}] tag=${tasks[i].tag} 0 points; tried=${JSON.stringify(r.triedCandidates)} debug=${(r.debug || '').slice(0, 600)}`);
            } else if (r.matched) {
                console.log(`[rap:${sectorId}] tag=${tasks[i].tag} matched=${JSON.stringify(r.matched)} points=${r.points.length}`);
            }
        });

        // Compute 0-6km bulk shear from V500mb - V10m magnitude.
        // eccodes preserves grid scan order across messages from the same GRIB2,
        // so index-aligned subtraction is safe.
        const u500 = byTag.u500, v500 = byTag.v500;
        const u10  = byTag.u10,  v10  = byTag.v10;
        const nShear = Math.min(u500.length, v500.length, u10.length, v10.length);
        const shearPts = new Array(nShear);
        for (let i = 0; i < nShear; i++) {
            const du = u500[i].value - u10[i].value;
            const dv = v500[i].value - v10[i].value;
            shearPts[i] = { lat: u500[i].lat, lon: u500[i].lon, value: Math.sqrt(du * du + dv * dv) };
        }

        // SRH: 0-1km storm-relative helicity, m²/s² (raw from eccodes, no scaling).
        const srhPts = byTag.srh1;

        // Significant Tornado Parameter (simplified):
        //   STP = clamp(CAPE/1500,0,4) × clamp(0-1km SRH/150,0,4) × clamp(BWD/20,0,4)
        const capePts = byTag.cape;
        const srh1Pts = byTag.srh1;
        const nSig = Math.min(capePts.length, srh1Pts.length, shearPts.length);
        const sigtorPts = new Array(nSig);
        for (let i = 0; i < nSig; i++) {
            const capeFac = Math.min(Math.max(capePts[i].value / 1500, 0), 4);
            const srhFac  = Math.min(Math.max(srh1Pts[i].value / 150,  0), 4);
            const bwdFac  = Math.min(Math.max(shearPts[i].value / 20,  0), 4);
            sigtorPts[i]  = { lat: capePts[i].lat, lon: capePts[i].lon, value: capeFac * srhFac * bwdFac };
        }

        const meta = inferGridMeta(byTag.cape);
        cache.validTime = downloaded.date.toISOString();
        cache.meta      = meta;
        cache.cape      = byTag.cape;
        cache.cin       = byTag.cin;
        cache.shear     = shearPts;
        cache.srh       = srhPts;
        cache.sigtor    = sigtorPts;
        console.log(`[rap:${sectorId}] cache refreshed validTime=${cache.validTime} nx=${meta && meta.nx} ny=${meta && meta.ny} cape=${cache.cape.length} cin=${cache.cin.length} shear=${cache.shear.length} srh=${cache.srh.length} sigtor=${cache.sigtor.length}`);
        cache.lastError = null;
    } catch (e) {
        cache.lastError = e.message;
        console.error(`[rap:${sectorId}] refresh failed: ${e.message}`);
    } finally {
        cache.loading = false;
    }
}

// Background loop — refreshes every sector that has been requested at least
// once. CONUS is primed at startup.
async function refreshAllActive() {
    for (const sectorId of activeSectors) {
        await refreshSector(sectorId);
    }
}

function streamParam(res, param, sectorId) {
    if (!SECTORS[sectorId]) sectorId = "CONUS";
    activeSectors.add(sectorId);
    const cache = sectorCache[sectorId];
    if (!cache || !cache.meta || !cache[param]) {
        // Kick off a fetch async; tell the client we're not ready yet so it can
        // poll. On first request for a new sector this takes ~30-60s.
        if (!cache || !cache.loading) refreshSector(sectorId);
        const c = sectorCache[sectorId] || {};
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            error: 'rap not ready',
            sector: sectorId,
            loading: !!c.loading,
            lastAttempt: c.lastAttempt || null,
            lastError: c.lastError || null,
            refreshCount: c.refreshCount || 0,
        }));
        return;
    }
    const data = cache[param];
    const meta = cache.meta;
    res.writeHead(200, {
        'Content-Type': 'application/x-ndjson',
        'Cache-Control': 'no-store',
    });
    res.write(JSON.stringify({
        type: 'meta', param, sector: sectorId,
        nx: meta.nx, ny: meta.ny,
        lat_min: meta.lat_min, lat_max: meta.lat_max,
        lon_min: meta.lon_min, lon_max: meta.lon_max,
        valid_time: cache.validTime,
    }) + '\n');
    for (let i = 0; i < data.length; i++) {
        res.write(JSON.stringify(data[i]) + '\n');
    }
    res.write(JSON.stringify({ type: 'end' }) + '\n');
    res.end();
}

// Public handler — mountable into another http server (e.g. server.js).
// Returns true if the request was handled, false otherwise (so the parent
// router can fall through to its own routes).
//
// Recognized paths (with or without a `/rap` prefix):
//   /rap/cape    /cape
//   /rap/cin     /cin
//   /rap/shear   /shear
//   /rap/srh     /srh
//   /rap/sigtor  /sigtor
function handle(req, res) {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    let p = url.pathname;
    if (p.startsWith('/rap')) p = p.slice(4) || '/';
    const sectorId = url.searchParams.get('sector') || 'CONUS';
    if (p === '/cape')    { streamParam(res, 'cape',    sectorId); return true; }
    if (p === '/cin')     { streamParam(res, 'cin',     sectorId); return true; }
    if (p === '/shear')   { streamParam(res, 'shear',   sectorId); return true; }
    if (p === '/srh')     { streamParam(res, 'srh',    sectorId); return true; }
    if (p === '/sigtor')  { streamParam(res, 'sigtor', sectorId); return true; }
    if (p === '/status') {
        // Per-sector status. Defaults to the one in ?sector= or all-sectors summary.
        const onlyId = url.searchParams.get('sector');
        const ids = onlyId ? [onlyId] : Object.keys(sectorCache);
        const sectors = {};
        for (const id of ids) {
            const c = sectorCache[id] || {};
            sectors[id] = {
                ready: !!(c.meta && c.cape),
                validTime: c.validTime || null,
                loading: !!c.loading,
                lastAttempt: c.lastAttempt || null,
                lastError: c.lastError || null,
                refreshCount: c.refreshCount || 0,
                counts: {
                    cape:   c.cape   ? c.cape.length   : 0,
                    cin:    c.cin    ? c.cin.length    : 0,
                    shear:  c.shear  ? c.shear.length  : 0,
                    srh:    c.srh    ? c.srh.length    : 0,
                    sigtor: c.sigtor ? c.sigtor.length : 0,
                },
            };
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            activeSectors: Array.from(activeSectors),
            availableSectors: Object.keys(SECTORS),
            sectors,
        }));
        return true;
    }
    if (p === '/sectors') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(SECTORS));
        return true;
    }
    return false;
}

// Kick off the refresh cycle as soon as this module is required.
refreshSector("CONUS");
const refreshTimer = setInterval(refreshAllActive, REFRESH_MS);
refreshTimer.unref && refreshTimer.unref();
console.log(`[rap] module loaded (workers=${NUM_WORKERS}, sectors=${Object.keys(SECTORS).length})`);

module.exports = { handle, refresh: refreshAllActive, refreshSector };

// Stand-alone mode — only runs when invoked directly (`node rap.js`), not
// when required from server.js.
if (require.main === module) {
    const server = http.createServer((req, res) => {
        if (handle(req, res)) return;
        if (new URL(req.url, `http://${req.headers.host}`).pathname === '/healthz') {
            res.writeHead(200); res.end('ok'); return;
        }
        res.writeHead(404); res.end('not found');
    });
    server.listen(PORT, () => console.log(`[rap] standalone listening on ${PORT}`));
}
