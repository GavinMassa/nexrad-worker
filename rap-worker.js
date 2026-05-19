// rap-worker.js — eccodes-backed field extractor (worker_threads).
//
// Spawned by rap.js per logical field. Accepts a list of `candidates` (each
// {shortName, typeOfLevel?, level?}) and tries them in order, returning the
// first that yields >0 grid points. This handles eccodes-table variations
// where, e.g., 10m winds are sometimes shortName=`10u`/`10v` rather than
// `u`/`v` with typeOfLevel=heightAboveGround.
//
// On exhaustion, runs `grib_ls -p shortName,typeOfLevel,level,paramId` against
// the file and returns the listing in `debug` so the caller can see what's
// actually inside the GRIB2.

'use strict';

const { parentPort, workerData } = require('worker_threads');
const { spawn, spawnSync } = require('child_process');

const { gribPath, candidates, tag } = workerData;

function buildWhere(c) {
    const parts = [`shortName=${c.shortName}`];
    if (c.typeOfLevel != null) parts.push(`typeOfLevel=${c.typeOfLevel}`);
    if (c.level != null)       parts.push(`level=${c.level}`);
    return parts.join(',');
}

function tryCandidate(c) {
    return new Promise(resolve => {
        const where = buildWhere(c);
        const proc = spawn('grib_get_data', ['-w', where, '-F', '%.4f', gribPath]);
        let stdout = '', stderr = '';
        proc.stdout.on('data', b => { stdout += b.toString(); });
        proc.stderr.on('data', b => { stderr += b.toString(); });
        proc.on('error', err => resolve({ points: [], stderr: `spawn failed: ${err.message}` }));
        proc.on('close', code => {
            if (code !== 0) { resolve({ points: [], stderr }); return; }
            const points = [];
            const lines = stdout.split('\n');
            let headerSkipped = false;
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line) continue;
                if (!headerSkipped) { headerSkipped = true; continue; }
                const parts = line.split(/\s+/);
                if (parts.length < 3) continue;
                const lat = parseFloat(parts[0]);
                let lon  = parseFloat(parts[1]);
                const value = parseFloat(parts[2]);
                if (Number.isNaN(lat) || Number.isNaN(lon) || Number.isNaN(value)) continue;
                if (lon > 180) lon -= 360;
                points.push({ lat, lon, value });
            }
            resolve({ points, stderr });
        });
    });
}

(async () => {
    for (const c of candidates) {
        const r = await tryCandidate(c);
        if (r.points.length > 0) {
            parentPort.postMessage({ points: r.points, matched: c, tag });
            return;
        }
    }
    // All candidates exhausted with 0 points. Capture what's actually in the file.
    let debug = '';
    try {
        const ls = spawnSync('grib_ls', ['-p', 'shortName,typeOfLevel,level,paramId', gribPath],
                             { encoding: 'utf8' });
        debug = (ls.stdout || '') + (ls.stderr || '');
    } catch (e) {
        debug = `grib_ls failed: ${e.message}`;
    }
    parentPort.postMessage({
        points: [],
        tag,
        triedCandidates: candidates,
        debug: debug.slice(0, 4000),  // keep payload reasonable
    });
})();
