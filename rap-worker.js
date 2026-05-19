// rap-worker.js — eccodes-backed field extractor (worker_threads).
//
// Spawned by rap.js for each (shortName, typeOfLevel, level) tuple. Shells out
// to `grib_get_data` (eccodes CLI) with a `-w` filter that selects only the
// matching GRIB2 message(s), then parses the "Latitude Longitude Value" rows
// from stdout. Posts {points: [{lat, lon, value}, ...]} back to the main thread.
//
// On Railway / Debian, install with:  apt-get install -y libeccodes-tools

'use strict';

const { parentPort, workerData } = require('worker_threads');
const { spawn } = require('child_process');

const { gribPath, shortName, typeOfLevel, level } = workerData;
const where = `shortName=${shortName},typeOfLevel=${typeOfLevel},level=${level}`;
const args  = ['-w', where, '-F', '%.4f', gribPath];

const proc = spawn('grib_get_data', args);
let stdout = '';
let stderr = '';
proc.stdout.on('data', chunk => { stdout += chunk.toString(); });
proc.stderr.on('data', chunk => { stderr += chunk.toString(); });
proc.on('error', err => {
    parentPort.postMessage({ error: `spawn grib_get_data failed: ${err.message}` });
});
proc.on('close', code => {
    if (code !== 0) {
        parentPort.postMessage({ error: `grib_get_data exit ${code}: ${stderr.trim()}` });
        return;
    }
    const points = [];
    const lines = stdout.split('\n');
    // First non-empty line is the header "Latitude, Longitude, Value" — skip it.
    let headerSkipped = false;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        if (!headerSkipped) { headerSkipped = true; continue; }
        const parts = line.split(/\s+/);
        if (parts.length < 3) continue;
        const lat = parseFloat(parts[0]);
        let lon   = parseFloat(parts[1]);
        const value = parseFloat(parts[2]);
        if (Number.isNaN(lat) || Number.isNaN(lon) || Number.isNaN(value)) continue;
        if (lon > 180) lon -= 360;  // normalize to -180..180
        points.push({ lat, lon, value });
    }
    parentPort.postMessage({ points });
});
