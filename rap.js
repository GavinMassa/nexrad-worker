// rap.js — RAP mesoanalysis HTTP server (wgrib2 backend)
//
// Pipeline (per 10-min refresh cycle):
//   1. HEAD-check both NOMADS GRIB2 URLs; skip cycle if Last-Modified unchanged.
//   2. Concurrently download two filtered GRIB2s from NOMADS filter_rap.pl:
//        main  — CAPE, CIN, T2m, Td2m, UGRD/VGRD at 500mb / 10m / 0-6km layer
//        hlcy  — HLCY only (no lev_* filter; NOMADS silently drops
//                heightAboveGroundLayer messages when any lev_* is active)
//   3. Single wgrib2 pass per file extracts all needed fields into flat
//      Float32 .bin files (-if/-fi conditional blocks, one file scan each).
//   4. Derive in the main thread:
//        shear = 0-6km BWD; falls back to 500mb–10m if layer missing
//        lcl   = 125 × (T2m_K – Td2m_K)  meters AGL  [Bolton 1980]
//        stp   = (CAPE/1500) × lcl_term × (SRH/150) × min(1.5, BWD/10.288)
//                [Thompson et al. 2003 fixed-layer STP]
//   5. Serve GET /rap/all — binary response: five concatenated Float32 CONUS
//      grids (cape, cin, shear, srh, stp), metadata in X-Meso-Meta header.
//
// Grid: awp130pgrbf00.grib2 — CONUS Lambert Conformal, NX=451 × NY=337 row-major.
//
// Endpoints:
//   GET /rap/all    → octet-stream body: [cape][cin][shear][srh][stp] Float32
//                     X-Meso-Meta: JSON {nx,ny,bbox,valid_time,params,...}
//   GET /rap/status → JSON diagnostics
//   GET /healthz    → "ok"   (handled by server.js, not this module)
//
// Deployment: Dockerfile must have `wgrib2` on PATH:
//   apt-get install -y wgrib2
// libeccodes-tools / grib_get_data are no longer required.

'use strict';

const http  = require('http');
const https = require('https');
const fs    = require('fs');
const path  = require('path');
const os    = require('os');
const zlib  = require('zlib');
const { spawn } = require('child_process');

// ── Constants ────────────────────────────────────────────────────────────────

const PORT       = parseInt(process.env.PORT || '3000', 10);
const REFRESH_MS = 10 * 60 * 1000;
const TMP_DIR    = path.join(os.tmpdir(), 'rap-cache');
fs.mkdirSync(TMP_DIR, { recursive: true });

// Fixed CONUS grid for awp130pgrbf00.grib2 (RAP 13km Lambert Conformal).
// 337 rows × 451 cols, row-major scan.
const NX = 451;
const NY = 337;

// Approximate corner lat/lons for the awp130 CONUS Lambert Conformal domain.
// Used by iOS to position the Metal overlay.  Refined from operational RAP
// grid spec; adjust if pixel offsets look wrong.
const LAT_MIN = 21.14;
const LAT_MAX = 47.84;
const LON_MIN = -134.09;
const LON_MAX = -60.92;

// SIDECAR_URL env var: set to http://<internal-hostname>:4000
// Internal hostname visible in Railway → sidecar service → Settings → Networking
const SIDECAR_URL = process.env.SIDECAR_URL || null;

// ── Blend response cache ─────────────────────────────────────────────────────
// Sidecar produces a new blend once per hour. Cache the response body so
// repeated iOS requests (e.g. app foreground, map pan/zoom) each serve
// instantly from memory instead of fetching 14 MB from the sidecar.
// Cache stores compressed bytes — compression runs once per sidecar cycle,
// every subsequent request within the TTL pays zero gzip cost.
let blendCache = { raw: null, compressed: null, metaHeader: null, cacheKey: null, fetchedAt: 0 };
// No fixed TTL — the cache is invalidated by valid_time change (cheap meta
// check), not a timer. The sidecar rewrites every ~10 min (mesonet re-blend)
// or ~60 min (full cycle). MAX_STALE_MS is only a safety net: if the meta
// endpoint is unreachable we serve the cached copy rather than refetch 14 MB on
// every request, but never for longer than this so a stuck meta can't freeze
// output indefinitely.
const MAX_STALE_MS = 75 * 60 * 1000;   // 75 min — longer than a full cycle

// Archived hour blend cache — immutable once fetched (past cycles never change).
// Grows to at most ARCHIVE_HOURS (6) entries × ~13 MB compressed each ≈ 80 MB.
// key: "YYYYMMDDHH" → { compressed: Buffer, metaHeader: string }
const hourBlendCache = new Map();

async function getBlendAll() {
    // ── Step 1: cheap check against sidecar meta.json ───────────────────────
    // meta.json is a tiny JSON file (~200 bytes) written atomically every time
    // the sidecar produces new output. We key on `generated_at` (wall-clock
    // write time), which changes on EVERY write — including the ~10-min mesonet
    // re-blends that keep `valid_time` fixed at the model cycle hour. Fall back
    // to valid_time for older sidecars that don't emit generated_at yet.
    // Checked first (< 5 ms) before deciding whether to fetch the 14 MB binary.
    let latestKey = null;
    try {
        const metaResp = await fetch(`${SIDECAR_URL}/blend/meta`, {
            signal: AbortSignal.timeout(5_000),
        });
        if (metaResp.ok) {
            const metaJson = await metaResp.json();
            latestKey = metaJson.generated_at || metaJson.valid_time || null;
        }
    } catch {
        // meta fetch failed — handled below (serve cache if we have one)
    }

    // ── Step 2: serve cache when safe ───────────────────────────────────────
    // Only consider the cache fresh enough if it's younger than MAX_STALE_MS.
    const cacheAge = blendCache.compressed ? (Date.now() - blendCache.fetchedAt) : Infinity;
    if (blendCache.compressed && cacheAge < MAX_STALE_MS) {
        if (latestKey && latestKey === blendCache.cacheKey) {
            return blendCache;   // confirmed unchanged — instant, zero sidecar cost
        }
        if (!latestKey) {
            // Meta unreachable (transient sidecar hiccup): serve the cached copy
            // rather than refetch 14 MB on every request. Bounded by MAX_STALE_MS.
            return blendCache;
        }
        // else: generated_at changed → fall through and fetch fresh data.
    }

    // ── Step 3: cache miss / changed / too stale — fetch full binary ────────
    const t0 = Date.now();
    const resp = await fetch(`${SIDECAR_URL}/blend/all`, {
        signal: AbortSignal.timeout(120_000),
    });
    if (!resp.ok) throw new Error(`sidecar ${resp.status}`);
    const metaHeader = resp.headers.get('x-meso-meta') || '{}';
    const raw = Buffer.from(await resp.arrayBuffer());
    console.log(`[blend] fetched from sidecar: ${Date.now() - t0}ms  raw=${(raw.length / 1024 / 1024).toFixed(1)}MB`);
    const compressed = await new Promise((resolve, reject) =>
        zlib.gzip(raw, { level: 1 }, (err, r) => err ? reject(err) : resolve(r))
    );
    console.log(`[blend] compressed: ${(compressed.length / 1024 / 1024).toFixed(1)}MB  ratio=${(raw.length / compressed.length).toFixed(1)}x  took=${Date.now() - t0}ms`);

    // Prefer the key from the meta check; fall back to the X-Meso-Meta header.
    let newKey = latestKey;
    if (!newKey) {
        try {
            const h = JSON.parse(metaHeader);
            newKey = h.generated_at || h.valid_time || null;
        } catch {}
    }

    blendCache = { raw, compressed, metaHeader, cacheKey: newKey, fetchedAt: Date.now() };
    return blendCache;
}

// ── In-memory state ──────────────────────────────────────────────────────────

const state = {
    validTime:         null,
    lastModifiedMain:  null,   // Last-Modified from NOMADS main file HEAD
    lastModifiedHlcy:  null,   // Last-Modified from NOMADS hlcy file HEAD
    cape:  null,               // Float32Array, NX*NY
    cin:   null,               // Float32Array, NX*NY
    shear: null,               // Float32Array, NX*NY  (0-6km BWD, m/s)
    srh:   null,               // Float32Array, NX*NY  (0-1km SRH, m²/s²)
    stp:   null,               // Float32Array, NX*NY  (Fixed SigTor, dimensionless)
    loading:      false,
    lastError:    null,
    lastAttempt:  null,
    refreshCount: 0,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function pad(n) { return String(n).padStart(2, '0'); }

/** Resolve the HTTP redirect chain and return the final URL string. */
async function resolveRedirects(url, depth = 0) {
    if (depth > 5) throw new Error('too many redirects');
    return new Promise((resolve, reject) => {
        const req = https.request(url, { method: 'HEAD' }, res => {
            res.resume();
            if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                resolveRedirects(res.headers.location, depth + 1).then(resolve, reject);
            } else {
                resolve({ statusCode: res.statusCode, headers: res.headers, url });
            }
        });
        req.setTimeout(15_000, () => { req.destroy(new Error('HEAD timeout')); });
        req.on('error', reject).end();
    });
}

/**
 * HEAD-check a NOMADS URL.
 * Returns the Last-Modified string if present, or null.
 * Returns 'GONE' if NOMADS returns 404/500 (run not available yet).
 */
async function headCheck(url) {
    try {
        const r = await resolveRedirects(url);
        if (r.statusCode === 404 || r.statusCode >= 500) return 'GONE';
        return r.headers['last-modified'] || null;
    } catch {
        return null;
    }
}

/** Download URL to destPath, following redirects. */
function downloadToFile(url, destPath, depth = 0) {
    return new Promise((resolve, reject) => {
        if (depth > 5) return reject(new Error('too many redirects'));
        https.get(url, res => {
            if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                res.resume();
                downloadToFile(res.headers.location, destPath, depth + 1).then(resolve, reject);
                return;
            }
            if (res.statusCode !== 200) {
                res.resume();
                return reject(new Error(`HTTP ${res.statusCode} from ${url}`));
            }
            const file = fs.createWriteStream(destPath);
            res.pipe(file);
            file.on('finish', () => file.close(resolve));
            file.on('error', reject);
        }).on('error', reject);
    });
}

/**
 * Run wgrib2 on gribPath, extracting each field to its own .bin file.
 *
 * fields: Array of { tag: string, match: string }
 *   match is a regex substring tested against wgrib2's inventory line, e.g.
 *   ':CAPE:surface:'
 *
 * Builds the command:
 *   wgrib2 <gribPath> \
 *     -if '<match1>' -no_header -bin <tag1.bin> -fi \
 *     -if '<match2>' -no_header -bin <tag2.bin> -fi \
 *     ...
 *
 * wgrib2 scans the file once; each -if/-fi block fires only for matching
 * records.  stdout (the verbose inventory) is discarded; only stderr is
 * captured for error reporting.
 *
 * Returns { [tag]: Float32Array | null }
 */
function runWgrib2(gribPath, fields, stamp) {
    const binPaths = {};
    // -order we:ns  — reorder output north-to-south before writing binary.
    // RAP GRIB2 (awp130p) stores data south-to-north (row 0 = lat_min).
    // The iOS Metal renderer expects row 0 = lat_max (northernmost), so we
    // fix the scan order here rather than flipping on the client.
    const args = [gribPath, '-order', 'we:ns'];
    for (const f of fields) {
        const bp = path.join(TMP_DIR, `${f.tag}_${stamp}.bin`);
        binPaths[f.tag] = bp;
        args.push('-if', f.match, '-no_header', '-bin', bp, '-fi');
    }

    return new Promise((resolve, reject) => {
        // Ignore stdout (wgrib2 inventory) to avoid buffering 150k+ lines.
        const proc = spawn('wgrib2', args, { stdio: ['ignore', 'ignore', 'pipe'] });
        let stderr = '';
        proc.stderr.on('data', d => { stderr += d.toString(); });
        proc.on('error', err => reject(new Error(`wgrib2 spawn failed: ${err.message}`)));
        proc.on('close', code => {
            if (code !== 0) {
                return reject(new Error(`wgrib2 exit ${code}: ${stderr.slice(0, 800)}`));
            }
            const result = {};
            for (const f of fields) {
                const bp = binPaths[f.tag];
                if (fs.existsSync(bp) && fs.statSync(bp).size >= 4) {
                    const buf = fs.readFileSync(bp);
                    // Slice into a clean ArrayBuffer (avoids shared-pool alignment issues).
                    const ab  = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
                    result[f.tag] = new Float32Array(ab);
                } else {
                    result[f.tag] = null;
                    console.warn(`[rap] wgrib2: no data for tag=${f.tag} match="${f.match}" file=${path.basename(gribPath)}`);
                    if (stderr) console.warn(`[rap] wgrib2 stderr: ${stderr.slice(0, 400)}`);
                }
                // Clean up immediately — these are intermediate files.
                try { fs.unlinkSync(binPaths[f.tag]); } catch {}
            }
            resolve(result);
        });
    });
}

/**
 * Return the full wgrib2 inventory for a file as a string.
 * Only use on small files (a handful of records) — stdout is buffered in memory.
 */
function getInventory(gribPath) {
    return new Promise((resolve) => {
        const proc = spawn('wgrib2', [gribPath], { stdio: ['ignore', 'pipe', 'ignore'] });
        let out = '';
        proc.stdout.on('data', d => { out += d.toString(); });
        proc.on('error', () => resolve('(spawn failed)'));
        proc.on('close', () => resolve(out.trim() || '(empty)'));
    });
}

/** Delete files older than maxAgeMs from TMP_DIR. */
function cleanOldFiles(maxAgeMs = 3 * 3600 * 1000) {
    try {
        const now = Date.now();
        for (const name of fs.readdirSync(TMP_DIR)) {
            const full = path.join(TMP_DIR, name);
            try {
                if (now - fs.statSync(full).mtimeMs > maxAgeMs) fs.unlinkSync(full);
            } catch {}
        }
    } catch {}
}

// ── STP derivation ───────────────────────────────────────────────────────────

/**
 * Bolton (1980) LCL height approximation.
 * t2m and td2m in Kelvin (difference is identical in Celsius).
 * Returns LCL height in meters AGL.
 */
function lclHeight(t2m, td2m) {
    return 125.0 * (t2m - td2m);
}

/**
 * Thompson et al. (2003) fixed-layer STP.
 *   capeTerm  = mlcape / 1500
 *   lclTerm   = clamp((2000 – lclM) / 1000, 0, 1)
 *   srhTerm   = srh1km / 150
 *   shearTerm = min(1.5, bwd6ms / 10.288)   [20 kt = 10.288 m/s]
 */
function computeSTP(mlcape, lclM, srh1km, bwd6ms) {
    const capeTerm  = mlcape / 1500;
    const lclTerm   = Math.min(1.0, Math.max(0.0, (2000 - lclM) / 1000));
    const srhTerm   = srh1km / 150;
    const shearTerm = Math.min(1.5, bwd6ms / 10.288);
    return Math.max(0, capeTerm * lclTerm * srhTerm * shearTerm);
}

// ── NOMADS URLs ──────────────────────────────────────────────────────────────
//
// Two downloads per cycle (same reason as before):
//   main — CAPE/CIN/winds/TMP/DPT with lev_* filters
//   hlcy — HLCY only, NO lev_* filter.
//
// NOMADS silently drops heightAboveGroundLayer messages (which HLCY uses)
// whenever any lev_* filter is active, regardless of which file is requested.
//
// awp130pgrbf00.grib2 is the only RAP product served through filter_rap.pl.
// No subregion params — full CONUS download gives the fixed NX=451/NY=337 grid.

function rapURLForHour(date) {
    const yyyy = date.getUTCFullYear();
    const ymd  = `${yyyy}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}`;
    const hh   = pad(date.getUTCHours());
    const params = new URLSearchParams({
        file: `rap.t${hh}z.awp130pgrbf00.grib2`,
        lev_surface:            'on',
        lev_500_mb:             'on',
        lev_10_m_above_ground:  'on',
        lev_2_m_above_ground:   'on',
        var_CAPE: 'on',
        var_CIN:  'on',
        var_UGRD: 'on',
        var_VGRD: 'on',
        var_TMP:  'on',
        var_DPT:  'on',
        dir: `/rap.${ymd}`,
    });
    return `https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl?${params}`;
}

function hlcyURLForHour(date) {
    const yyyy = date.getUTCFullYear();
    const ymd  = `${yyyy}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}`;
    const hh   = pad(date.getUTCHours());
    const params = new URLSearchParams({
        file: `rap.t${hh}z.awp130pgrbf00.grib2`,
        var_HLCY: 'on',
        dir: `/rap.${ymd}`,
    });
    return `https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl?${params}`;
}

// ── Refresh pipeline ─────────────────────────────────────────────────────────

async function refresh() {
    if (state.loading) return;
    state.loading     = true;
    state.lastAttempt = new Date().toISOString();
    state.refreshCount++;

    try {
        // ── 1. Find the latest available run (try 4 hours back) ──────────────
        const now = new Date();
        now.setUTCMinutes(0, 0, 0);

        let downloaded = null;
        for (let i = 0; i < 4; i++) {
            const t     = new Date(now.getTime() - i * 3600_000);
            const stamp = `${t.getUTCFullYear()}${pad(t.getUTCMonth()+1)}${pad(t.getUTCDate())}_${pad(t.getUTCHours())}`;
            const mainURL  = rapURLForHour(t);
            const hlcyURL  = hlcyURLForHour(t);
            const mainDest = path.join(TMP_DIR, `rap_main_${stamp}.grib2`);
            const hlcyDest = path.join(TMP_DIR, `rap_hlcy_${stamp}.grib2`);

            // ── 2. HEAD-check both files before downloading ──────────────────
            const [mainMod, hlcyMod] = await Promise.all([
                headCheck(mainURL),
                headCheck(hlcyURL),
            ]);

            if (mainMod === 'GONE') {
                // This run doesn't exist yet; try an earlier one.
                continue;
            }

            // Skip full cycle if both files are unchanged and we have data.
            const mainUnchanged = mainMod && mainMod === state.lastModifiedMain;
            const hlcyUnchanged = hlcyMod && hlcyMod === state.lastModifiedHlcy;
            if (mainUnchanged && hlcyUnchanged && state.cape) {
                console.log(`[rap] NOMADS files unchanged (${stamp}), skipping download`);
                return;
            }

            try {
                await Promise.all([
                    (mainUnchanged && fs.existsSync(mainDest) && fs.statSync(mainDest).size > 1000)
                        ? Promise.resolve()
                        : downloadToFile(mainURL, mainDest),
                    (hlcyUnchanged && fs.existsSync(hlcyDest) && fs.statSync(hlcyDest).size > 1000)
                        ? Promise.resolve()
                        : downloadToFile(hlcyURL, hlcyDest),
                ]);

                const mainSz = fs.existsSync(mainDest) ? fs.statSync(mainDest).size : 0;
                if (mainSz > 1000) {
                    if (mainMod) state.lastModifiedMain = mainMod;
                    if (hlcyMod) state.lastModifiedHlcy = hlcyMod;
                    downloaded = { date: t, stamp, mainDest, hlcyDest };
                    break;
                }
            } catch (e) {
                console.warn(`[rap] fetch ${stamp} failed: ${e.message}`);
            }
        }

        if (!downloaded) {
            console.warn('[rap] no run available yet');
            return;
        }

        const { stamp, mainDest, hlcyDest } = downloaded;
        console.log(`[rap] extracting run ${stamp}…`);

        // ── 3. Single wgrib2 pass per file ───────────────────────────────────
        //
        // Main fields (from mainDest):
        //   CAPE, CIN  — surface parcel (awp130p stores them at typeOfLevel=surface)
        //   UGRD, VGRD — 500mb isobaric  (upper wind for shear fallback)
        //   UGRD, VGRD — 10m AGL         (surface wind)
        //   TMP, DPT   — 2m AGL          (Bolton LCL for STP)
        //   UGRD, VGRD — 6000-0m AGL layer (0-6km BWD; absent in awp130p,
        //                 fallback to 500mb–10m fires automatically)
        //
        // HLCY fields (from hlcyDest):
        //   HLCY       — 0-1km SRH

        const mainFields = [
            { tag: 'cape', match: ':CAPE:surface:'                },
            { tag: 'cin',  match: ':CIN:surface:'                 },
            { tag: 'u500', match: ':UGRD:500 mb:'                },
            { tag: 'v500', match: ':VGRD:500 mb:'                },
            { tag: 'u10',  match: ':UGRD:10 m above ground:'     },
            { tag: 'v10',  match: ':VGRD:10 m above ground:'     },
            { tag: 't2m',  match: ':TMP:2 m above ground:'       },
            { tag: 'td2m', match: ':DPT:2 m above ground:'       },
            // 0-6km layer winds — absent from awp130p; gracefully
            // absent if the bin file comes back null (fallback fires below).
            { tag: 'u6k',  match: ':UGRD:6000-0 m above ground:' },
            { tag: 'v6k',  match: ':VGRD:6000-0 m above ground:' },
        ];
        const hlcyFields = [
            // wgrib2 -if uses POSIX ERE, so .* works.
            // Match 0-1km SRH regardless of whether wgrib2 writes the layer as
            // "0-1000 m above ground" or "1000-0 m above ground".
            { tag: 'srh1', match: ':HLCY:.*1000' },
            // Fallback: grab 0-3km SRH if 0-1km is absent from this file.
            { tag: 'srh3', match: ':HLCY:.*3000' },
        ];

        // Log HLCY file size + full inventory (small file; only 1-2 records).
        // This appears in Railway logs and shows the exact wgrib2 level strings
        // so we can verify the match patterns are correct.
        const hlcySz = fs.existsSync(hlcyDest) ? fs.statSync(hlcyDest).size : 0;
        console.log(`[rap] HLCY file size: ${hlcySz} bytes`);
        if (hlcySz > 0) {
            const hlcyInv = await getInventory(hlcyDest);
            console.log(`[rap] HLCY inventory:\n${hlcyInv}`);
        }

        // Run both extractions concurrently (each is one file scan).
        const [mainData, hlcyData] = await Promise.all([
            runWgrib2(mainDest, mainFields, stamp),
            runWgrib2(hlcyDest, hlcyFields, stamp),
        ]);
        const d = { ...mainData, ...hlcyData };

        // Validate required fields.
        const required = ['cape', 'u500', 'v500', 'u10', 'v10'];
        for (const tag of required) {
            if (!d[tag] || d[tag].length < NX * NY) {
                throw new Error(`Required field missing or undersized: tag=${tag} ` +
                                `got=${d[tag] ? d[tag].length : 0} expected=${NX * NY}`);
            }
        }
        // HLCY: prefer 0-1km, fall back to 0-3km, hard-fail if neither present.
        if (d.srh1 && d.srh1.length >= NX * NY) {
            console.log('[rap] using 0-1km SRH');
        } else if (d.srh3 && d.srh3.length >= NX * NY) {
            console.log('[rap] 0-1km SRH absent — falling back to 0-3km SRH');
            d.srh1 = d.srh3;
        } else {
            throw new Error(`HLCY missing: srh1=${d.srh1?.length ?? 0} srh3=${d.srh3?.length ?? 0} expected=${NX * NY}`);
        }

        const n = NX * NY;

        // ── 4a. 0-6km BWD shear (fallback to 500mb–10m) ─────────────────────
        const shear = new Float32Array(n);
        if (d.u6k && d.u6k.length === n) {
            console.log('[rap] using 0-6km AGL layer winds for shear');
            for (let i = 0; i < n; i++) {
                const du = d.u6k[i] - d.u10[i];
                const dv = d.v6k[i] - d.v10[i];
                shear[i] = Math.sqrt(du * du + dv * dv);
            }
        } else {
            console.log('[rap] 0-6km layer absent → falling back to 500mb–10m bulk shear');
            for (let i = 0; i < n; i++) {
                const du = d.u500[i] - d.u10[i];
                const dv = d.v500[i] - d.v10[i];
                shear[i] = Math.sqrt(du * du + dv * dv);
            }
        }

        // ── 4b. Fixed-layer STP (Thompson 2003) ─────────────────────────────
        //   LCL: Bolton (1980)  lcl_m = 125 × (T2m_K – Td2m_K)
        //   STP = (CAPE/1500) × clamp((2000−lcl)/1000, 0,1) × (SRH/150)
        //         × min(1.5, BWD/10.288)
        const hasTd = d.t2m && d.td2m && d.t2m.length === n && d.td2m.length === n;
        if (!hasTd) console.warn('[rap] T2m/Td2m missing — LCL defaulting to 1000 m AGL');

        const stp = new Float32Array(n);
        const cape  = d.cape;
        const srh1  = d.srh1;
        for (let i = 0; i < n; i++) {
            const lcl = hasTd ? lclHeight(d.t2m[i], d.td2m[i]) : 1000.0;
            stp[i] = computeSTP(cape[i], lcl, srh1[i], shear[i]);
        }

        // ── 5. Commit to state ───────────────────────────────────────────────
        state.validTime = downloaded.date.toISOString();
        state.cape  = cape;
        state.cin   = d.cin || new Float32Array(n);  // CIN may be absent
        state.shear = shear;
        state.srh   = srh1;
        state.stp   = stp;
        state.lastError = null;

        // Quick sanity log — max STP in first 10k cells.
        let stpMax = 0;
        const check = Math.min(10000, n);
        for (let i = 0; i < check; i++) if (stp[i] > stpMax) stpMax = stp[i];
        console.log(`[rap] refreshed validTime=${state.validTime} n=${n} ` +
                    `cape0=${cape[0].toFixed(1)} stpMax(sample)=${stpMax.toFixed(2)}`);

        // Clean GRIB files older than 3 h.
        cleanOldFiles(3 * 3600_000);

    } catch (e) {
        state.lastError = e.message;
        console.error(`[rap] refresh failed: ${e.message}`);
    } finally {
        state.loading = false;
    }
}

// ── HTTP handler ─────────────────────────────────────────────────────────────
//
// Mounted by server.js via: rap.handle(req, res)
// Strips leading '/rap' so internally routes are /all, /status, etc.

async function handle(req, res) {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    let p = url.pathname;
    if (p.startsWith('/rap')) p = p.slice(4) || '/';

    // ── GET /rap/all ─────────────────────────────────────────────────────────
    // Binary response: five consecutive Float32 grids (NX*NY values each).
    // Grid order: cape, cin, shear, srh, stp.
    // Metadata in X-Meso-Meta response header (JSON).
    //
    // iOS reads the header for dims/bbox, then slices the body into 5 typed
    // arrays of (nx * ny) Float32 values each.
    if (p === '/all') {
        if (!state.cape) {
            if (!state.loading) refresh();
            res.writeHead(503, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                error: 'rap not ready',
                loading: state.loading,
                lastAttempt: state.lastAttempt,
                lastError:   state.lastError,
                refreshCount: state.refreshCount,
            }));
            return true;
        }

        const PARAMS  = ['cape', 'cin', 'shear', 'srh', 'stp'];
        const BPP     = NX * NY * 4;   // bytes per param
        const meta    = JSON.stringify({
            nx: NX, ny: NY,
            lat_min: LAT_MIN, lat_max: LAT_MAX,
            lon_min: LON_MIN, lon_max: LON_MAX,
            valid_time:     state.validTime,
            params:         PARAMS,
            bytes_per_param: BPP,
        });

        res.writeHead(200, {
            'Content-Type':                  'application/octet-stream',
            'X-Meso-Meta':                   meta,
            'Access-Control-Allow-Origin':   '*',
            'Access-Control-Expose-Headers': 'X-Meso-Meta',
            'Cache-Control':                 'no-store',
            'Content-Length':                String(PARAMS.length * BPP),
        });
        res.end(Buffer.concat([
            Buffer.from(state.cape.buffer),
            Buffer.from(state.cin.buffer),
            Buffer.from(state.shear.buffer),
            Buffer.from(state.srh.buffer),
            Buffer.from(state.stp.buffer),
        ]));
        return true;
    }

    // ── GET /rap/status ──────────────────────────────────────────────────────
    if (p === '/status') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            ready:        !!state.cape,
            validTime:    state.validTime,
            loading:      state.loading,
            lastAttempt:  state.lastAttempt,
            lastError:    state.lastError,
            refreshCount: state.refreshCount,
            lastModifiedMain: state.lastModifiedMain,
            lastModifiedHlcy: state.lastModifiedHlcy,
            gridPoints:   state.cape ? state.cape.length : 0,
            params: state.cape ? {
                cape:  state.cape.length,
                cin:   state.cin.length,
                shear: state.shear.length,
                srh:   state.srh.length,
                stp:   state.stp.length,
            } : null,
        }));
        return true;
    }

    // ── GET /rap/blend/meta ──────────────────────────────────────────────────
    // Tiny JSON passthrough — used by getBlendAll() for cheap revalidation.
    // Returns the sidecar's meta.json: {valid_time, nx, ny, ...} (~200 bytes).
    if (p === '/blend/meta') {
        if (!SIDECAR_URL) {
            res.writeHead(503, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'SIDECAR_URL not set' }));
            return true;
        }
        try {
            const r = await fetch(`${SIDECAR_URL}/blend/meta`, {
                signal: AbortSignal.timeout(5_000),
            });
            const body = await r.text();
            res.writeHead(r.status, {
                'Content-Type':                'application/json',
                'Cache-Control':               'no-cache',
                'Access-Control-Allow-Origin': '*',
            });
            return res.end(body);
        } catch (err) {
            res.writeHead(503);
            return res.end(JSON.stringify({ error: err.message }));
        }
    }

    // ── GET /rap/blend/history ───────────────────────────────────────────────
    // Returns history.json from the sidecar — list of available archived hours.
    if (p === '/blend/history') {
        if (!SIDECAR_URL) {
            res.writeHead(503, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'SIDECAR_URL env var not set' }));
            return true;
        }
        try {
            const r = await fetch(`${SIDECAR_URL}/blend/history`);
            const body = await r.text();
            res.writeHead(r.status, {
                'Content-Type':                'application/json',
                'Cache-Control':               'no-cache',
                'Access-Control-Allow-Origin': '*',
            });
            return res.end(body);
        } catch (err) {
            res.writeHead(503, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ error: err.message }));
        }
    }

    // ── GET /rap/blend/:hour ─────────────────────────────────────────────────
    // Serves an archived blend cycle for a specific hour (YYYYMMDDHH).
    // Immutable once fetched — cached forever in hourBlendCache (no TTL).
    // Must appear BEFORE /blend/all so the regex never matches "all".
    const m = p.match(/^\/blend\/(\d{10})$/);
    if (m) {
        const hour = m[1];
        if (!SIDECAR_URL) {
            res.writeHead(503, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'SIDECAR_URL env var not set' }));
            return true;
        }
        try {
            // Serve from in-process cache when available (immutable — past hours never change).
            if (hourBlendCache.has(hour)) {
                const cached = hourBlendCache.get(hour);
                res.writeHead(200, {
                    'Content-Type':                  'application/octet-stream',
                    'Content-Encoding':              'gzip',
                    'Content-Length':                String(cached.compressed.length),
                    'X-Meso-Meta':                   cached.metaHeader,
                    'Access-Control-Allow-Origin':   '*',
                    'Access-Control-Expose-Headers': 'X-Meso-Meta',
                    'Cache-Control':                 'public, max-age=86400',
                });
                return res.end(cached.compressed);
            }
            // Cache miss — fetch from sidecar, compress, cache.
            const r = await fetch(`${SIDECAR_URL}/blend/${hour}`, {
                signal: AbortSignal.timeout(60_000),
            });
            if (!r.ok) {
                res.writeHead(r.status);
                return res.end(`Hour ${hour} not available`);
            }
            const metaHeader = r.headers.get('x-meso-meta') || '{}';
            const raw = Buffer.from(await r.arrayBuffer());
            const compressed = await new Promise((resolve, reject) =>
                zlib.gzip(raw, { level: 1 }, (err, result) =>
                    err ? reject(err) : resolve(result))
            );
            hourBlendCache.set(hour, { compressed, metaHeader });
            console.log(`[blend] cached hour ${hour}: ${(compressed.length / 1024 / 1024).toFixed(1)}MB`);
            res.writeHead(200, {
                'Content-Type':                  'application/octet-stream',
                'Content-Encoding':              'gzip',
                'Content-Length':                String(compressed.length),
                'X-Meso-Meta':                   metaHeader,
                'Access-Control-Allow-Origin':   '*',
                'Access-Control-Expose-Headers': 'X-Meso-Meta',
                'Cache-Control':                 'public, max-age=86400',
            });
            return res.end(compressed);
        } catch (err) {
            console.error(`[blend] hour ${hour} error:`, err.message);
            if (!res.headersSent) { res.writeHead(503); res.end(err.message); }
            return true;
        }
    }

    // ── GET /rap/blend/all ───────────────────────────────────────────────────
    // Proxy to the Python sidecar's /blend/all with in-process gzip cache.
    // See getBlendAll() above — compress once, serve many.
    if (p === '/blend/all') {
        if (!SIDECAR_URL) {
            res.writeHead(503, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'SIDECAR_URL env var not set' }));
            return true;
        }

        // getBlendAll() returns cached compressed bytes when available (cache hit = ~5ms).
        // On a cache miss it fetches from sidecar, compresses once, caches, then returns.
        try {
            const cache = await getBlendAll();
            res.writeHead(200, {
                'Content-Type':                  'application/octet-stream',
                'Content-Encoding':              'gzip',
                'Content-Length':                String(cache.compressed.length),
                'X-Meso-Meta':                   cache.metaHeader,
                'Access-Control-Allow-Origin':   '*',
                'Access-Control-Expose-Headers': 'X-Meso-Meta',
                'Cache-Control':                 'no-store',
            });
            res.end(cache.compressed);
        } catch (e) {
            if (!res.headersSent) {
                res.writeHead(502, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: `sidecar unreachable: ${e.message}` }));
            }
        }
        return true;
    }

    return false;
}

// ── Startup ──────────────────────────────────────────────────────────────────

refresh();
const refreshTimer = setInterval(refresh, REFRESH_MS);
if (refreshTimer.unref) refreshTimer.unref();
console.log('[rap] module loaded (wgrib2 backend, CONUS full-grid, NX=451 NY=337)');

module.exports = { handle, refresh };

// Stand-alone mode (`node rap.js` for local testing).
if (require.main === module) {
    const server = http.createServer((req, res) => {
        if (handle(req, res)) return;
        const p = new URL(req.url, 'http://localhost').pathname;
        if (p === '/healthz') { res.writeHead(200); res.end('ok'); return; }
        res.writeHead(404); res.end('not found');
    });
    server.listen(PORT, () => console.log(`[rap] standalone on ${PORT}`));
}
