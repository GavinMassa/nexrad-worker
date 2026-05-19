// rap.js — RAP mesoanalysis HTTP server (CAPE, CIN, 0-6km bulk shear)
//
// Pipeline:
//   1. Every 10 min, fetch the latest available RAP awp13f00.grib2 from NOMADS,
//      using the filter_rap.pl subregion filter to keep the download ~15 MB.
//   2. Spawn worker_threads (./rap-worker.js) — one per (shortName, level)
//      combo — that shell out to `grib_get_data` (eccodes CLI) and parse the
//      lat/lon/value rows. Up to 8 workers run in parallel across the
//      8-vCPU Railway instance.
//   3. Compute 0-6km bulk shear from |V500mb - V10m|.
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
const NUM_WORKERS = 8;  // pool size; current refresh uses 6 of them per cycle
const REFRESH_MS  = 10 * 60 * 1000;
const TMP_DIR     = path.join(os.tmpdir(), 'rap-cache');
fs.mkdirSync(TMP_DIR, { recursive: true });

const SUBREGION = { leftlon: -135, rightlon: -60, toplat: 55, bottomlat: 20 };

const cache = {
    validTime: null,    // ISO8601
    meta: null,         // { nx, ny, lat_min, lat_max, lon_min, lon_max }
    cape: null,         // [{lat,lon,value}]
    cin: null,
    shear: null,
    loading: false,
};

function pad(n) { return String(n).padStart(2, '0'); }

function rapURLForHour(date) {
    const yyyy = date.getUTCFullYear();
    const ymd  = `${yyyy}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}`;
    const hh   = pad(date.getUTCHours());
    const params = new URLSearchParams({
        file: `rap.t${hh}z.awp13f00.grib2`,
        lev_surface: 'on',
        lev_500_mb: 'on',
        lev_10_m_above_ground: 'on',
        var_CAPE: 'on',
        var_CIN: 'on',
        var_UGRD: 'on',
        var_VGRD: 'on',
        subregion: '',
        leftlon:   String(SUBREGION.leftlon),
        rightlon:  String(SUBREGION.rightlon),
        toplat:    String(SUBREGION.toplat),
        bottomlat: String(SUBREGION.bottomlat),
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
        // bin to nearest 0.01° for set counting (RAP is ~0.13°)
        lats.add(Math.round(p.lat * 100));
        lons.add(Math.round(p.lon * 100));
    }
    return { nx: lons.size, ny: lats.size, lat_min, lat_max, lon_min, lon_max };
}

async function refresh() {
    if (cache.loading) return;
    cache.loading = true;
    try {
        // Try the most recent 4 hours in descending order — NOMADS publishes
        // ~30-45 min after the run hour; the current hour may not yet exist.
        const now = new Date();
        now.setUTCMinutes(0, 0, 0);
        let downloaded = null;
        for (let i = 0; i < 4; i++) {
            const t = new Date(now.getTime() - i * 3600 * 1000);
            const url = rapURLForHour(t);
            const stamp = t.toISOString().replace(/[:.]/g, '_');
            const dest = path.join(TMP_DIR, `rap_${stamp}.grib2`);
            try {
                if (!fs.existsSync(dest) || fs.statSync(dest).size < 1000) {
                    await downloadToFile(url, dest);
                }
                if (fs.statSync(dest).size > 1000) {
                    downloaded = { date: t, path: dest };
                    break;
                }
            } catch (e) {
                console.warn(`[rap] fetch ${t.toISOString()} failed: ${e.message}`);
            }
        }
        if (!downloaded) {
            console.warn('[rap] no run available yet');
            return;
        }

        // 6 parallel field extractions (CAPE sfc, CIN sfc, U/V at 500mb, U/V at 10m).
        // Spec called for 8 workers — pool size is 8 but this refresh uses 6.
        const tasks = [
            { tag: 'cape',  shortName: 'cape', typeOfLevel: 'surface',           level: 0   },
            { tag: 'cin',   shortName: 'cin',  typeOfLevel: 'surface',           level: 0   },
            { tag: 'u500',  shortName: 'u',    typeOfLevel: 'isobaricInhPa',     level: 500 },
            { tag: 'v500',  shortName: 'v',    typeOfLevel: 'isobaricInhPa',     level: 500 },
            { tag: 'u10',   shortName: 'u',    typeOfLevel: 'heightAboveGround', level: 10  },
            { tag: 'v10',   shortName: 'v',    typeOfLevel: 'heightAboveGround', level: 10  },
        ];
        const results = await Promise.all(tasks.map(t =>
            runExtraction({ gribPath: downloaded.path, ...t })));
        const byTag = {};
        results.forEach((r, i) => { byTag[tasks[i].tag] = r.points || []; });

        // Compute 0-6km bulk shear from V500mb - V10m magnitude.
        // eccodes preserves grid scan order across messages from the same GRIB2,
        // so index-aligned subtraction is safe.
        const u500 = byTag.u500, v500 = byTag.v500;
        const u10  = byTag.u10,  v10  = byTag.v10;
        const n = Math.min(u500.length, v500.length, u10.length, v10.length);
        const shearPts = new Array(n);
        for (let i = 0; i < n; i++) {
            const du = u500[i].value - u10[i].value;
            const dv = v500[i].value - v10[i].value;
            shearPts[i] = { lat: u500[i].lat, lon: u500[i].lon, value: Math.sqrt(du * du + dv * dv) };
        }

        const meta = inferGridMeta(byTag.cape);
        cache.validTime = downloaded.date.toISOString();
        cache.meta      = meta;
        cache.cape      = byTag.cape;
        cache.cin       = byTag.cin;
        cache.shear     = shearPts;
        console.log(`[rap] cache refreshed validTime=${cache.validTime} nx=${meta && meta.nx} ny=${meta && meta.ny} cape=${cache.cape.length} cin=${cache.cin.length} shear=${cache.shear.length}`);
    } catch (e) {
        console.error(`[rap] refresh failed: ${e.message}`);
    } finally {
        cache.loading = false;
    }
}

function streamParam(res, param) {
    const data = cache[param];
    const meta = cache.meta;
    if (!data || !meta) {
        res.writeHead(503, { 'Content-Type': 'text/plain' });
        res.end('rap not ready');
        return;
    }
    res.writeHead(200, {
        'Content-Type': 'application/x-ndjson',
        'Cache-Control': 'no-store',
    });
    res.write(JSON.stringify({
        type: 'meta', param,
        nx: meta.nx, ny: meta.ny,
        lat_min: meta.lat_min, lat_max: meta.lat_max,
        lon_min: meta.lon_min, lon_max: meta.lon_max,
        valid_time: cache.validTime,
    }) + '\n');
    // Bulk-stream rows. For ~260k points each JSON.stringify is fine; total ~10 MB.
    for (let i = 0; i < data.length; i++) {
        res.write(JSON.stringify(data[i]) + '\n');
    }
    res.write(JSON.stringify({ type: 'end' }) + '\n');
    res.end();
}

const server = http.createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (url.pathname === '/rap/cape')  return streamParam(res, 'cape');
    if (url.pathname === '/rap/cin')   return streamParam(res, 'cin');
    if (url.pathname === '/rap/shear') return streamParam(res, 'shear');
    if (url.pathname === '/healthz')   { res.writeHead(200); return res.end('ok'); }
    res.writeHead(404); res.end('not found');
});

server.listen(PORT, () => {
    console.log(`[rap] listening on ${PORT} (workers=${NUM_WORKERS})`);
    refresh();
    setInterval(refresh, REFRESH_MS);
});
