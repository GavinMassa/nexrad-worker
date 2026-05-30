const http  = require('http');
const zlib  = require('zlib');
const { URL } = require('url');
const seekBzip = require('seek-bzip');
let sharp;
try {
  sharp = require('sharp');
  console.log('[sharp] loaded successfully');
} catch (e) {
  console.error('[sharp] failed to load — tile endpoint will return 503:', e.message);
}
const fs    = require('fs');
const rap   = require('./rap');

const PORT = process.env.PORT || 3000;

process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err.stack || err);
});
process.on('unhandledRejection', (err) => {
  console.error('Unhandled rejection:', err.stack || err);
});

// ============================================================
// IN-MEMORY CACHE
// ============================================================

const cache = new Map();
const CACHE_MAX = 50;
const SCAN_TTL = 5 * 60 * 1000;
const VOL_TTL = 30 * 60 * 1000;

function cacheGet(key) {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expires) { cache.delete(key); return null; }
  return entry.value;
}

function cacheSet(key, value, ttl) {
  if (cache.size >= CACHE_MAX) {
    const oldest = cache.keys().next().value;
    cache.delete(oldest);
  }
  cache.set(key, { value, expires: Date.now() + ttl });
}

// ── Sidecar proxy — satellite images ────────────────────────────────────────
// Node fetches satellite JPEGs from the Python sidecar over localhost HTTP.
// Results are cached in-process for 4.5 min (just under the 5-min update interval).
// When the JPEG changes, all tile cache entries for that product are evicted.
const SIDECAR_URL = process.env.SIDECAR_URL || 'http://localhost:4000';
const SAT_JPEG_TTL_MS = 4.5 * 60 * 1000;
const satJpegCache = {
  geocolor: { buf: null, expires: 0 },
  visible:  { buf: null, expires: 0 },
};

async function fetchSatJpeg(product) {
  const entry = satJpegCache[product];
  const now = Date.now();
  if (entry.buf && now < entry.expires) return entry.buf;

  let resp;
  try {
    resp = await fetch(`${SIDECAR_URL}/satellite/${product}`,
                      { signal: AbortSignal.timeout(15000) });
  } catch (e) {
    console.error(`[satellite] fetch ${product} from sidecar failed:`, e.message);
    return entry.buf;   // return stale buffer rather than failing tile render
  }
  if (!resp.ok) {
    console.warn(`[satellite] sidecar returned ${resp.status} for ${product}`);
    return entry.buf;
  }
  const buf = Buffer.from(await resp.arrayBuffer());

  // If JPEG content changed, evict all tile cache entries for this product.
  if (entry.buf && !buf.equals(entry.buf)) {
    for (const k of tileCache.keys()) {
      if (k.startsWith(product + '/')) tileCache.delete(k);
    }
    console.log(`[satellite] ${product} updated — tile cache cleared`);
  }
  entry.buf = buf;
  entry.expires = now + SAT_JPEG_TTL_MS;
  return buf;
}

// ── Satellite tile cache ─────────────────────────────────────────────────────
// Keyed by "{product}/{z}/{x}/{y}". Stores rendered PNG Buffer.
// Invalidated when the source JPEG file changes (mtime check on each miss).
const tileCache = new Map();
const TILE_CACHE_MAX = 5000;   // z=8 CONUS needs ~800 tiles; 5000 avoids eviction mid-session
// TILE_SIZE = 256 shared with NEXRAD renderer below — declared once there.

// GOES CONUS bbox — must match sidecar/satellite.py constants exactly.
const SAT_LAT_MIN =  14.568;
const SAT_LAT_MAX =  53.297;
const SAT_LON_MIN = -135.038;
const SAT_LON_MAX =  -59.975;


// ============================================================
// BZIP2 DECODER (seek-bzip npm package)
// ============================================================

function bzip2Decode(input) {
  return new Uint8Array(seekBzip.decode(Buffer.from(input)));
}

// ============================================================
// ZLIB HELPERS (Node.js built-in)
// ============================================================

function zlibDecompress(data) {
  try {
    return new Uint8Array(zlib.inflateSync(Buffer.from(data)));
  } catch {
    return data;
  }
}

function deflateCompress(data) {
  return new Uint8Array(zlib.deflateSync(Buffer.from(data)));
}

// ============================================================
// NEXRAD LEVEL II PARSER
// ============================================================

const STATION_COORDS = {
  ABR:[45.456,-98.413],ABX:[35.15,-106.824],AKQ:[36.984,-77.008],AMA:[35.233,-101.709],
  AMX:[25.611,-80.413],APX:[44.906,-84.72],ARX:[43.823,-91.191],ATX:[48.195,-122.496],
  BBX:[39.496,-121.632],BGM:[42.2,-75.985],BHX:[40.499,-124.292],BIS:[46.771,-100.76],
  BLX:[45.854,-108.607],BMX:[33.172,-86.77],BOX:[41.956,-71.137],BRO:[25.916,-97.419],
  BUF:[42.949,-78.737],BYX:[24.597,-81.703],CAE:[33.949,-81.119],CBW:[46.039,-67.806],
  CBX:[43.49,-116.236],CCX:[40.923,-78.004],CLE:[41.413,-81.86],CLX:[32.656,-81.042],
  CRP:[27.784,-97.511],CXX:[44.511,-73.166],CYS:[41.152,-104.806],DAX:[38.501,-121.678],
  DDC:[37.761,-99.969],DFX:[29.273,-100.281],DGX:[32.28,-89.984],DIX:[39.947,-74.411],
  DLH:[46.837,-92.21],DMX:[41.731,-93.723],DOX:[38.826,-75.44],DTX:[42.7,-83.472],
  DVN:[41.612,-90.581],DYX:[32.538,-99.254],EAX:[38.81,-94.264],EMX:[31.894,-110.63],
  ENX:[42.586,-74.064],EOX:[31.46,-85.459],EPZ:[31.873,-106.698],ESX:[35.701,-114.891],
  EVX:[30.564,-85.922],EWX:[29.704,-98.029],EYX:[35.098,-117.561],FCX:[37.024,-80.274],
  FDR:[34.362,-98.976],FDX:[34.635,-103.63],FFC:[33.364,-84.566],FSD:[43.588,-96.729],
  FSX:[34.574,-111.198],FTG:[39.787,-104.546],FWS:[32.573,-97.303],GGW:[48.206,-106.625],
  GJX:[39.062,-108.214],GLD:[39.367,-101.7],GRB:[44.499,-88.111],GRK:[30.722,-97.383],
  GRR:[42.894,-85.545],GSP:[34.883,-82.22],GWX:[33.897,-88.329],GYX:[43.891,-70.257],
  HDX:[33.077,-106.12],HGX:[29.472,-95.079],HNX:[36.314,-119.632],HPX:[36.737,-87.285],
  HTX:[34.931,-86.083],ICT:[37.655,-97.443],ICX:[37.591,-112.862],ILN:[39.42,-83.822],
  ILX:[40.151,-89.337],IND:[39.708,-86.28],INX:[36.175,-95.565],IWA:[33.289,-111.67],
  IWX:[41.359,-85.7],JAX:[30.485,-81.702],JGX:[32.675,-83.351],JKL:[37.591,-83.313],
  JUA:[18.116,-66.078],LBB:[33.654,-101.814],LCH:[30.125,-93.216],LGX:[47.117,-124.107],
  LIX:[30.337,-89.826],LNX:[41.958,-100.576],LOT:[41.605,-88.085],LRX:[40.74,-116.803],
  LSX:[38.699,-90.683],LTX:[33.989,-78.429],LVX:[37.975,-85.944],LWX:[38.975,-77.478],
  LZK:[34.836,-92.262],MAF:[31.943,-102.189],MAX:[42.081,-122.717],MBX:[48.393,-100.864],
  MHX:[34.776,-76.876],MKX:[42.968,-88.551],MLB:[28.113,-80.654],MOB:[30.68,-88.24],
  MPX:[44.849,-93.566],MQT:[46.531,-87.548],MRX:[36.169,-83.402],MSX:[47.041,-113.986],
  MTX:[41.263,-112.448],MUX:[37.155,-121.898],MVX:[47.528,-97.326],MXX:[32.537,-85.79],
  NKX:[32.919,-117.042],NQA:[35.345,-89.873],OAX:[41.32,-96.367],OHX:[36.247,-86.563],
  OKX:[40.866,-72.864],OTX:[47.681,-117.627],PAH:[37.068,-88.772],PBZ:[40.532,-80.218],
  PDT:[45.691,-118.853],POE:[34.892,-92.975],PUX:[38.46,-104.181],RAX:[35.665,-78.49],
  RGX:[39.754,-119.462],RIW:[43.066,-108.477],RLX:[38.311,-81.723],RTX:[45.715,-122.965],
  SFX:[43.106,-112.686],SGF:[37.235,-93.4],SHV:[32.451,-93.841],SJT:[31.371,-100.492],
  SOX:[33.818,-117.636],SRX:[35.29,-94.362],TBW:[27.706,-82.402],TFX:[47.46,-111.385],
  TLH:[30.398,-84.329],TLX:[35.333,-97.278],TWX:[38.997,-96.232],TYX:[43.756,-75.68],
  UDX:[44.125,-102.83],UEX:[40.321,-98.442],VAX:[30.89,-83.002],VBX:[34.838,-120.397],
  VNX:[36.741,-98.128],VTX:[34.412,-119.179],VWX:[41.4,-87.725],YUX:[32.495,-114.656],
};

function decompressBlock(compressed) {
  if (compressed[0] === 0x42 && compressed[1] === 0x5A) {
    return bzip2Decode(compressed);
  }
  return zlibDecompress(compressed);
}

function parseMessage31(data, offset, length) {
  if (length < 60) return null;
  const view = new DataView(data.buffer, data.byteOffset + offset, Math.min(length, data.byteLength - offset));

  const azimuth = view.getFloat32(12, false);
  const elevationNumber = view.getUint8(22);
  const elevation = view.getFloat32(24, false);
  const dataBlockCount = view.getUint16(30, false);

  const moments = [];
  for (let i = 0; i < dataBlockCount; i++) {
    const ptrOff = 32 + i * 4;
    if (ptrOff + 4 > length) break;
    const blockPtr = view.getUint32(ptrOff, false);
    if (blockPtr === 0 || blockPtr + 28 >= length) continue;

    const bOff = data.byteOffset + offset + blockPtr;
    if (bOff + 28 > data.buffer.byteLength) continue;
    const bv = new DataView(data.buffer, bOff);
    const blockType = String.fromCharCode(bv.getUint8(0));
    if (blockType !== 'D') continue;

    const nameBytes = new Uint8Array(data.buffer, bOff + 1, 3);
    const name = String.fromCharCode(...nameBytes).trim();
    const numGates = bv.getUint16(8, false);
    const firstGateRange = bv.getUint16(10, false);
    const gateSizeMeters = bv.getUint16(12, false);
    const wordSize = bv.getUint8(19);
    const scale = bv.getFloat32(20, false);
    const mOffset = bv.getFloat32(24, false);

    const bytesPerGate = wordSize === 16 ? 2 : 1;
    const dataLen = numGates * bytesPerGate;
    if (bOff + 28 + dataLen > data.buffer.byteLength) continue;

    const gateData = new Uint8Array(data.buffer, bOff + 28, dataLen);
    moments.push({ name, numGates, firstGateRange, gateSizeMeters, scale, offset: mOffset, wordSize, data: new Uint8Array(gateData) });
  }

  return { azimuth, elevation, elevationNumber, moments };
}

// Walks the entire volume file emitting one message-31 record at a time via callback.
// Never accumulates records — each record becomes GC-able after the callback returns.
// Callback may return false to stop iteration early.
async function iterateMessage31Records(buffer, callback) {
  const data = new Uint8Array(buffer);
  const view = new DataView(buffer);
  let pos = 24;
  while (pos + 4 < data.length) {
    const blockSize = view.getInt32(pos, false);
    pos += 4;
    if (blockSize === 0) break;
    const absSize = Math.abs(blockSize);
    if (pos + absSize > data.length) break;

    const compressed = data.slice(pos, pos + absSize);
    pos += absSize;

    let decompressed;
    try { decompressed = decompressBlock(compressed); } catch { continue; }

    let mPos = 0;
    while (mPos + 28 <= decompressed.length) {
      const ctmEnd = mPos + 12;
      if (ctmEnd + 16 > decompressed.length) break;
      const hv = new DataView(decompressed.buffer, decompressed.byteOffset + ctmEnd, 16);
      const msgSize = hv.getUint16(0, false) * 2;
      const msgType = hv.getUint8(3);

      if (msgType === 31) {
        const start = ctmEnd + 16;
        const len = msgSize > 16 ? msgSize - 16 : decompressed.length - start;
        const rec = parseMessage31(decompressed, start, Math.min(len, decompressed.length - start));
        if (rec && rec.moments.length > 0) {
          const cont = await callback(rec);
          if (cont === false) return;
        }
      }

      const total = 12 + Math.max(msgSize, 16);
      mPos += total;
      if (mPos <= ctmEnd) mPos = ctmEnd + 18;
    }
  }
}

function parseMessagesFromBlock(data) {
  const records = [];
  let pos = 0;
  while (pos + 28 <= data.length) {
    const ctmEnd = pos + 12;
    if (ctmEnd + 16 > data.length) break;
    const hv = new DataView(data.buffer, data.byteOffset + ctmEnd, 16);
    const msgSize = hv.getUint16(0, false) * 2;
    const msgType = hv.getUint8(3);

    if (msgType === 31) {
      const start = ctmEnd + 16;
      const len = msgSize > 16 ? msgSize - 16 : data.length - start;
      const rec = parseMessage31(data, start, Math.min(len, data.length - start));
      if (rec && rec.moments.length > 0) records.push(rec);
    }

    const total = 12 + Math.max(msgSize, 16);
    pos += total;
    if (pos <= ctmEnd) pos = ctmEnd + 18;
  }
  return records;
}

function parseLevel2(buffer) {
  const data = new Uint8Array(buffer);
  const view = new DataView(buffer);

  const decoder = new TextDecoder('ascii');
  const stationId = decoder.decode(new Uint8Array(buffer, 20, 4)).trim().replace(/^K/, '');
  const coords = STATION_COORDS[stationId] || [0, 0];

  const allRecords = [];
  let pos = 24;

  while (pos + 4 < data.length) {
    const blockSize = view.getInt32(pos, false);
    pos += 4;
    if (blockSize === 0) break;
    const absSize = Math.abs(blockSize);
    if (pos + absSize > data.length) break;

    const compressed = data.slice(pos, pos + absSize);
    pos += absSize;

    let decompressed;
    try {
      decompressed = decompressBlock(compressed);
    } catch { continue; }

    const records = parseMessagesFromBlock(decompressed);
    allRecords.push(...records);
  }

  const sweepMap = new Map();
  for (const rec of allRecords) {
    for (const m of rec.moments) {
      const key = `${rec.elevationNumber}:${m.name}`;
      if (!sweepMap.has(key)) sweepMap.set(key, { records: [], moment: m, elevation: rec.elevation });
      sweepMap.get(key).records.push(rec);
    }
  }

  const sweeps = [];
  for (const [, group] of sweepMap) {
    const { moment, records, elevation } = group;
    const radials = records.map(rec => {
      const m = rec.moments.find(x => x.name === moment.name);
      let gates;
      if (m.wordSize === 16) {
        const n = (m.data.length >> 1);
        gates = new Uint16Array(n);
        for (let i = 0; i < n; i++) gates[i] = (m.data[i*2] << 8) | m.data[i*2 + 1];
      } else {
        gates = new Uint8Array(m.data); // copy
      }
      return { azimuth: rec.azimuth, gates };
    }).sort((a, b) => a.azimuth - b.azimuth);

    sweeps.push({
      product: moment.name, elevation, numGates: moment.numGates,
      firstGateRange: moment.firstGateRange, gateSizeMeters: moment.gateSizeMeters,
      scale: moment.scale, offset: moment.offset, wordSize: moment.wordSize,
      radials, stationLat: coords[0], stationLon: coords[1],
    });
  }
  return { sweeps };
}

// ============================================================
// COLOR TABLES
// ============================================================

// Gradient tables: [minVal, maxVal, r1,g1,b1,a1, r2,g2,b2,a2]
const REF_GRAD = [
  [-30,-20, 116,78,173,0, 147,141,117,255],
  [-20,-10, 150,145,83,255, 210,212,180,255],
  [-10,10,  204,207,180,255, 65,91,158,255],
  [10,18,   67,97,162,255, 106,208,228,255],
  [18,22,   111,214,232,255, 53,213,91,255],
  [22,35,   17,213,24,255, 9,94,9,255],
  [35,40,   29,104,9,255, 234,210,4,255],
  [40,50,   255,226,0,255, 255,128,0,255],
  [50,60,   255,0,0,255, 113,0,0,255],
  [60,65,   255,255,255,255, 255,146,255,255],
  [65,70,   255,117,255,255, 225,11,227,255],
  [70,75,   178,0,255,255, 99,0,214,255],
  [75,85,   5,236,240,255, 1,32,32,255],
  [85,95,   1,32,32,255, 1,32,32,255],
];
// VEL .pal converted from MPH to m/s (÷2.237)
const VEL_GRAD = [
  [-53.6,-25.9, 0,0,255,255, 71,240,240,255],
  [-25.9,-22.4, 71,240,240,255, 82,247,89,255],
  [-22.4,-17.9, 82,247,89,255, 0,255,0,255],
  [-17.9,-4.5,  0,255,0,255, 16,96,16,255],
  [-4.5,-0.01,  16,96,16,255, 112,128,112,255],
  [-0.01,0.01,  144,128,144,255, 144,128,144,255],
  [0.01,4.5,    144,128,144,255, 112,0,0,255],
  [4.5,17.9,    112,0,0,255, 255,0,0,255],
  [17.9,22.4,   255,0,0,255, 255,55,26,255],
  [22.4,25.9,   255,55,26,255, 254,154,39,255],
  [25.9,31.3,   254,154,39,255, 255,255,0,255],
  [31.3,53.6,   255,255,0,255, 164,89,68,255],
];
const SW_GRAD = [
  [0,2,   100,100,100,180, 100,100,100,180],
  [2,4,   0,200,255,200, 0,200,255,200],
  [4,6,   0,255,0,220, 0,255,0,220],
  [6,8,   255,255,0,230, 255,255,0,230],
  [8,10,  255,150,0,240, 255,150,0,240],
  [10,15, 255,0,0,245, 255,0,0,245],
  [15,20, 200,0,200,250, 200,0,200,250],
  [20,40, 255,255,255,255, 255,255,255,255],
];
const RHO_GRAD = [
  [0.00,0.45, 15,15,140,255, 15,15,140,255],
  [0.45,0.60, 15,15,140,255, 10,10,190,255],
  [0.60,0.75, 10,10,190,255, 120,120,255,255],
  [0.75,0.80, 120,120,255,255, 95,245,100,255],
  [0.80,0.85, 95,245,100,255, 135,215,10,255],
  [0.85,0.90, 135,215,10,255, 255,255,0,255],
  [0.90,0.95, 255,255,0,255, 255,140,0,255],
  [0.95,0.97, 255,140,0,255, 225,3,0,255],
  [0.97,0.99, 225,3,0,255, 139,30,77,255],
  [0.99,1.00, 139,30,77,255, 255,180,215,255],
  [1.00,1.05, 255,180,215,255, 164,54,150,255],
];

function getColor(product, value) {
  const table = product === 'VEL' ? VEL_GRAD : product === 'SW' ? SW_GRAD : product === 'RHO' ? RHO_GRAD : REF_GRAD;
  for (const entry of table) {
    const [min, max] = entry;
    if (value >= min && value < max) {
      const t = (max === min) ? 0 : (value - min) / (max - min);
      const r1 = entry[2], g1 = entry[3], b1 = entry[4], a1 = entry[5];
      const r2 = entry[6], g2 = entry[7], b2 = entry[8], a2 = entry[9];
      return [
        Math.round(r1 + (r2 - r1) * t),
        Math.round(g1 + (g2 - g1) * t),
        Math.round(b1 + (b2 - b1) * t),
        Math.round(a1 + (a2 - a1) * t),
      ];
    }
  }
  return [0, 0, 0, 0];
}

// ============================================================
// TILE RENDERER
// ============================================================

const DEG2RAD = Math.PI / 180;
const EARTH_R = 6371000;
const TILE_SIZE = 256;

function tileToLatLon(z, x, y) {
  const n = 2 ** z;
  return {
    minLon: (x / n) * 360 - 180,
    maxLon: ((x + 1) / n) * 360 - 180,
    maxLat: Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) / DEG2RAD,
    minLat: Math.atan(Math.sinh(Math.PI * (1 - 2 * (y + 1) / n))) / DEG2RAD,
  };
}

function haversine(lat1, lon1, lat2, lon2) {
  const dLat = (lat2 - lat1) * DEG2RAD, dLon = (lon2 - lon1) * DEG2RAD;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * DEG2RAD) * Math.cos(lat2 * DEG2RAD) * Math.sin(dLon / 2) ** 2;
  return EARTH_R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function bearing(lat1, lon1, lat2, lon2) {
  const dLon = (lon2 - lon1) * DEG2RAD;
  const y = Math.sin(dLon) * Math.cos(lat2 * DEG2RAD);
  const x = Math.cos(lat1 * DEG2RAD) * Math.sin(lat2 * DEG2RAD) -
            Math.sin(lat1 * DEG2RAD) * Math.cos(lat2 * DEG2RAD) * Math.cos(dLon);
  return (Math.atan2(y, x) / DEG2RAD + 360) % 360;
}

function renderTileData(sweep, z, tileX, tileY) {
  const { minLat, maxLat, minLon, maxLon } = tileToLatLon(z, tileX, tileY);
  const rgba = new Uint8Array(TILE_SIZE * TILE_SIZE * 4);

  const maxRange = sweep.firstGateRange + sweep.numGates * sweep.gateSizeMeters;
  const numRadials = sweep.radials.length;
  if (numRadials === 0) return rgba;

  const azimuths = sweep.radials.map(r => r.azimuth);

  for (let py = 0; py < TILE_SIZE; py++) {
    const lat = maxLat - (py / TILE_SIZE) * (maxLat - minLat);
    for (let px = 0; px < TILE_SIZE; px++) {
      const lon = minLon + (px / TILE_SIZE) * (maxLon - minLon);

      const dist = haversine(sweep.stationLat, sweep.stationLon, lat, lon);
      if (dist < sweep.firstGateRange || dist > maxRange) continue;

      const az = bearing(sweep.stationLat, sweep.stationLon, lat, lon);

      let lo = 0, hi = numRadials - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; azimuths[mid] < az ? lo = mid + 1 : hi = mid; }
      let best = lo;
      let bestDiff = Math.min(Math.abs(azimuths[lo] - az), 360 - Math.abs(azimuths[lo] - az));
      for (const idx of [lo - 1, lo + 1, 0, numRadials - 1]) {
        if (idx >= 0 && idx < numRadials) {
          const d = Math.min(Math.abs(azimuths[idx] - az), 360 - Math.abs(azimuths[idx] - az));
          if (d < bestDiff) { bestDiff = d; best = idx; }
        }
      }

      const radial = sweep.radials[best];
      const gateIdx = Math.floor((dist - sweep.firstGateRange) / sweep.gateSizeMeters);
      if (gateIdx < 0 || gateIdx >= radial.gates.length) continue;

      const raw = radial.gates[gateIdx];
      if (raw <= 1) continue;

      const phys = (raw - sweep.offset) / sweep.scale;
      const [r, g, b, a] = getColor(sweep.product, phys);
      if (a === 0) continue;

      const idx = (py * TILE_SIZE + px) * 4;
      rgba[idx] = r; rgba[idx + 1] = g; rgba[idx + 2] = b; rgba[idx + 3] = a;
    }
  }
  return rgba;
}

// ============================================================
// PNG ENCODER
// ============================================================

function crc32(buf) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let j = 0; j < 8; j++) crc = (crc >>> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function pngChunk(type, data) {
  const typeBytes = Buffer.from(type, 'ascii');
  const chunk = Buffer.alloc(4 + 4 + data.length + 4);
  chunk.writeUInt32BE(data.length, 0);
  typeBytes.copy(chunk, 4);
  Buffer.from(data).copy(chunk, 8);
  const crcBuf = Buffer.concat([typeBytes, Buffer.from(data)]);
  chunk.writeUInt32BE(crc32(crcBuf), 8 + data.length);
  return chunk;
}

function encodePNG(width, height, rgba) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = 6;

  const raw = Buffer.alloc(height * (1 + width * 4));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + width * 4)] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + y * width * 4, width * 4).copy(raw, y * (1 + width * 4) + 1);
  }
  const compressed = deflateCompress(raw);

  const ihdrC = pngChunk('IHDR', ihdr);
  const idatC = pngChunk('IDAT', compressed);
  const iendC = pngChunk('IEND', Buffer.alloc(0));

  return Buffer.concat([sig, ihdrC, idatC, iendC]);
}

// ============================================================
// S3 FETCH (Unidata public bucket)
// ============================================================

const S3_BASE = 'https://unidata-nexrad-level2.s3.amazonaws.com';

async function listScans(station, date) {
  const cacheKey = `scans:${station}:${date}`;
  const cached = cacheGet(cacheKey);
  if (cached) return cached;

  const [year, month, day] = date.split('-');
  const prefix = `${year}/${month}/${day}/${station}/`;
  const resp = await fetch(`${S3_BASE}?list-type=2&prefix=${encodeURIComponent(prefix)}&delimiter=/`);
  if (!resp.ok) return [];

  const xml = await resp.text();
  const scans = [];
  const re = /<Key>([^<]+)<\/Key>\s*<LastModified>[^<]+<\/LastModified>\s*(?:<[A-Za-z]+>[^<]*<\/[A-Za-z]+>\s*)*<Size>(\d+)<\/Size>/g;
  let m;
  while ((m = re.exec(xml))) {
    const key = m[1], size = parseInt(m[2]);
    if (key.includes('MDM') || size < 100000) continue;
    const tsMatch = key.match(/(\d{8})_(\d{6})/);
    if (!tsMatch) continue;
    const d = tsMatch[1], t = tsMatch[2];
    const iso = `${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6,8)}T${t.slice(0,2)}:${t.slice(2,4)}:${t.slice(4,6)}Z`;
    scans.push({ key, time: iso, station, size });
  }

  if (scans.length > 0) {
    const isToday = date === new Date().toISOString().slice(0, 10);
    cacheSet(cacheKey, scans, isToday ? SCAN_TTL : 60 * 60 * 1000);
  }
  return scans;
}

async function fetchVolumeFile(station, time) {
  let isoTime = time.includes('T') ? time :
    time.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3T$4:$5:$6Z');
  const date = isoTime.slice(0, 10);
  const scans = await listScans(station, date);
  if (scans.length === 0) return null;

  const target = new Date(isoTime).getTime();
  const closest = scans.reduce((best, s) =>
    Math.abs(new Date(s.time).getTime() - target) < Math.abs(new Date(best.time).getTime() - target) ? s : best
  );

  const volCacheKey = `vol:${closest.key}`;
  const cachedVol = cacheGet(volCacheKey);
  if (cachedVol) return cachedVol;

  const resp = await fetch(`${S3_BASE}/${closest.key}`);
  if (!resp.ok) return null;
  const data = await resp.arrayBuffer();

  cacheSet(volCacheKey, data, VOL_TTL);
  return data;
}

// ============================================================
// HTTP SERVER
// ============================================================

function sendJson(res, data, status = 200) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  });
  res.end(JSON.stringify(data));
}

function sendPng(res, png) {
  res.writeHead(200, {
    'Content-Type': 'image/png',
    'Cache-Control': 'public, max-age=300',
    'Access-Control-Allow-Origin': '*',
  });
  res.end(png);
}

// Web Mercator tile → geographic bounds (WGS-84 degrees)
function tileToBBox(x, y, z) {
  const n = Math.pow(2, z);
  const lonMin = x / n * 360 - 180;
  const lonMax = (x + 1) / n * 360 - 180;
  const latMax = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) * 180 / Math.PI;
  const latMin = Math.atan(Math.sinh(Math.PI * (1 - 2 * (y + 1) / n))) * 180 / Math.PI;
  return { lonMin, lonMax, latMin, latMax };
}

// Render one XYZ tile from the source satellite JPEG.
// Returns a PNG Buffer or null if the tile is fully outside the image bbox.
async function renderSatTile(product, z, x, y) {
  if (!sharp) return null;

  const cacheKey = `${product}/${z}/${x}/${y}`;
  if (tileCache.has(cacheKey)) return tileCache.get(cacheKey);

  const tb = tileToBBox(x, y, z);

  // Fully outside satellite coverage — return transparent PNG.
  if (tb.lonMax <= SAT_LON_MIN || tb.lonMin >= SAT_LON_MAX ||
      tb.latMax <= SAT_LAT_MIN || tb.latMin >= SAT_LAT_MAX) {
    const transparent = await sharp({
      create: { width: TILE_SIZE, height: TILE_SIZE, channels: 4,
                background: { r: 0, g: 0, b: 0, alpha: 0 } }
    }).png().toBuffer();
    return transparent;
  }

  // Fetch cached JPEG buffer — sharp reads directly from Buffer, no full decode needed.
  const jpegBuf = await fetchSatJpeg(product);
  if (!jpegBuf) return null;

  // Get source dimensions without full decode (fast header read).
  const meta = await sharp(jpegBuf).metadata();
  const srcW = meta.width, srcH = meta.height;

  function mercY(latDeg) {
    const r = latDeg * Math.PI / 180;
    return Math.log(Math.tan(Math.PI / 4 + r / 2));
  }
  const yMercSatMin = mercY(SAT_LAT_MIN);
  const yMercSatMax = mercY(SAT_LAT_MAX);

  function lonToPixel(lon) {
    return (lon - SAT_LON_MIN) / (SAT_LON_MAX - SAT_LON_MIN) * srcW;
  }
  function latToPixel(lat) {
    return (yMercSatMax - mercY(lat)) / (yMercSatMax - yMercSatMin) * srcH;
  }

  const px0 = lonToPixel(Math.max(tb.lonMin, SAT_LON_MIN));
  const px1 = lonToPixel(Math.min(tb.lonMax, SAT_LON_MAX));
  const py0 = latToPixel(Math.min(tb.latMax, SAT_LAT_MAX));
  const py1 = latToPixel(Math.max(tb.latMin, SAT_LAT_MIN));

  const cropX = Math.max(0, Math.floor(Math.min(px0, px1)));
  const cropY = Math.max(0, Math.floor(Math.min(py0, py1)));
  const cropW = Math.min(srcW - cropX, Math.ceil(Math.max(px0, px1)) - cropX);
  const cropH = Math.min(srcH - cropY, Math.ceil(Math.max(py0, py1)) - cropY);

  if (cropW <= 0 || cropH <= 0) return null;

  const tileScaleX = TILE_SIZE / (lonToPixel(tb.lonMax) - lonToPixel(tb.lonMin));
  const tileScaleY = TILE_SIZE / (latToPixel(tb.latMin) - latToPixel(tb.latMax));
  const dstX = Math.round((cropX - lonToPixel(tb.lonMin)) * tileScaleX);
  const dstY = Math.round((cropY - latToPixel(tb.latMax)) * tileScaleY);
  const dstW = Math.max(1, Math.round(cropW * tileScaleX));
  const dstH = Math.max(1, Math.round(cropH * tileScaleY));

  const clampedDstX = Math.max(0, dstX);
  const clampedDstY = Math.max(0, dstY);
  const maxW = TILE_SIZE - clampedDstX;
  const maxH = TILE_SIZE - clampedDstY;
  const clampedDstW = Math.max(1, Math.min(dstW, maxW));
  const clampedDstH = Math.max(1, Math.min(dstH, maxH));

  let pngBuf;
  // Fast path: crop fills the whole canvas — skip composite, one pipeline call.
  if (clampedDstX === 0 && clampedDstY === 0 &&
      clampedDstW >= TILE_SIZE && clampedDstH >= TILE_SIZE) {
    pngBuf = await sharp(jpegBuf)
      .extract({ left: cropX, top: cropY, width: cropW, height: cropH })
      .resize(TILE_SIZE, TILE_SIZE, { fit: 'fill', kernel: 'lanczos3' })
      .png({ compressionLevel: 6 })
      .toBuffer();
  } else {
    // Edge tile: crop sits at an offset — composite onto transparent canvas.
    const cropped = await sharp(jpegBuf)
      .extract({ left: cropX, top: cropY, width: cropW, height: cropH })
      .resize(clampedDstW, clampedDstH, { fit: 'fill', kernel: 'lanczos3' })
      .toBuffer();
    pngBuf = await sharp({
      create: { width: TILE_SIZE, height: TILE_SIZE, channels: 4,
                background: { r: 0, g: 0, b: 0, alpha: 0 } }
    })
    .composite([{ input: cropped, left: clampedDstX, top: clampedDstY }])
    .png({ compressionLevel: 6 })
    .toBuffer();
  }

  if (tileCache.size >= TILE_CACHE_MAX) {
    tileCache.delete(tileCache.keys().next().value);
  }
  tileCache.set(cacheKey, pngBuf);
  return pngBuf;
}

async function serveSatellite(res, product) {
  const buf = await fetchSatJpeg(product);
  if (!buf) {
    res.writeHead(503, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: `satellite_${product} not ready yet` }));
    return;
  }
  res.writeHead(200, {
    'Content-Type':                'image/jpeg',
    'Content-Length':              buf.length,
    'Cache-Control':               'public, max-age=270',
    'Access-Control-Allow-Origin': '*',
    'X-Satellite-Product':         product,
  });
  res.end(buf);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const path = url.pathname;

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    return res.end();
  }

  // Mounted RAP mesoanalysis router — handles /rap/all, /rap/status, /rap/blend/*.
  // rap.handle() is async; must be awaited so the response is actually sent
  // before server.js falls through to other routes.
  if (path.startsWith('/rap/')) {
    const handled = await rap.handle(req, res);
    if (handled) return;
  }

  try {
    // GET /satellite/:product/tiles/:z/:x/:y.png
    // XYZ tile slices of the pre-reprojected satellite JPEG.
    // product: "geocolor" or "visible"
    // z: zoom level (4–12 recommended; outside coverage returns transparent)
    const satTileMatch = path.match(/^\/satellite\/(geocolor|visible)\/tiles\/(\d+)\/(\d+)\/(\d+)\.png$/i);
    if (satTileMatch) {
      const product = satTileMatch[1].toLowerCase();
      const z = parseInt(satTileMatch[2]), x = parseInt(satTileMatch[3]), y = parseInt(satTileMatch[4]);
      if (z < 1 || z > 14 || isNaN(x) || isNaN(y)) {
        res.writeHead(400); return res.end('Bad tile coordinates');
      }
      try {
        const tile = await renderSatTile(product, z, x, y);
        if (!tile) { res.writeHead(204); return res.end(); }
        res.writeHead(200, {
          'Content-Type':                'image/png',
          'Content-Length':              tile.length,
          'Cache-Control':               'public, max-age=270',
          'Access-Control-Allow-Origin': '*',
        });
        return res.end(tile);
      } catch (err) {
        console.error('[tile] error:', err.message);
        res.writeHead(500); return res.end('Tile error');
      }
    }

    // GET /satellite/geocolor — pre-reprojected GOES-19 GeoColor JPEG
    if (path === '/satellite/geocolor') {
      return serveSatellite(res, 'geocolor');
    }

    // GET /satellite/visible — pre-reprojected GOES-19 Band 02 visible JPEG
    if (path === '/satellite/visible') {
      return serveSatellite(res, 'visible');
    }

    // GET /satellite/meta — metadata (timestamps, bbox, dimensions)
    if (path === '/satellite/meta') {
      try {
        const resp = await fetch(`${SIDECAR_URL}/satellite/meta`,
                                 { signal: AbortSignal.timeout(5000) });
        if (!resp.ok) throw new Error(`sidecar ${resp.status}`);
        const meta = await resp.text();
        res.writeHead(200, {
          'Content-Type':                'application/json',
          'Cache-Control':               'public, max-age=60',
          'Access-Control-Allow-Origin': '*',
        });
        return res.end(meta);
      } catch {
        res.writeHead(503, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ error: 'satellite_meta not ready yet' }));
      }
    }

    // GET /api/scans/:station?date=YYYY-MM-DD
    let m = path.match(/^\/api\/scans\/([A-Z]{3,4})$/i);
    if (m) {
      const station = m[1].toUpperCase();
      const date = url.searchParams.get('date') || new Date().toISOString().slice(0, 10);
      return sendJson(res, { station, date, scans: await listScans(station, date) });
    }

    // GET /api/scans/:station/latest
    m = path.match(/^\/api\/scans\/([A-Z]{3,4})\/latest$/i);
    if (m) {
      const station = m[1].toUpperCase();
      const today = new Date().toISOString().slice(0, 10);
      let scans = await listScans(station, today);
      if (scans.length === 0) {
        const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
        scans = await listScans(station, yesterday);
      }
      return scans.length > 0
        ? sendJson(res, scans[scans.length - 1])
        : sendJson(res, { error: 'No scans found' }, 404);
    }

    // GET /api/products/:station?time=...
    m = path.match(/^\/api\/products\/([A-Z]{3,4})$/i);
    if (m) {
      const station = m[1].toUpperCase();
      const time = url.searchParams.get('time');
      if (!time) return sendJson(res, { error: 'time parameter required' }, 400);
      const vol = await fetchVolumeFile(station, time);
      if (!vol) return sendJson(res, { error: 'No data' }, 404);
      const parsed = parseLevel2(vol);
      const products = [...new Set(parsed.sweeps.map(s => s.product))];
      const elevations = [...new Set(parsed.sweeps.map(s => s.elevation.toFixed(1)))].sort((a, b) => a - b);
      return sendJson(res, { station, time, products, elevations });
    }

    // GET /tiles/:station/:product/:z/:x/:y.png?time=...&elevation=0.5
    m = path.match(/^\/tiles\/([A-Z]{3,4})\/([A-Z0-9]+)\/(\d+)\/(\d+)\/(\d+)\.png$/i);
    if (m) {
      const station = m[1].toUpperCase();
      const product = m[2].toUpperCase();
      const z = parseInt(m[3]), x = parseInt(m[4]), y = parseInt(m[5]);
      let time = url.searchParams.get('time');
      const elevation = parseFloat(url.searchParams.get('elevation') || '0.5');

      if (!time || time === 'latest') {
        const today = new Date().toISOString().slice(0, 10);
        let scans = await listScans(station, today);
        if (scans.length === 0) {
          const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
          scans = await listScans(station, yesterday);
        }
        if (scans.length === 0) return sendJson(res, { error: 'No scans found' }, 404);
        time = scans[scans.length - 1].time;
      }

      const vol = await fetchVolumeFile(station, time);
      if (!vol) return sendJson(res, { error: 'No data' }, 404);

      const parsed = parseLevel2(vol);
      const sweeps = parsed.sweeps.filter(s => s.product === product);
      if (sweeps.length === 0) return sendJson(res, { error: `No ${product} data` }, 404);

      const sweep = sweeps.reduce((a, b) =>
        Math.abs(a.elevation - elevation) <= Math.abs(b.elevation - elevation) ? a : b
      );

      const rgba = renderTileData(sweep, z, x, y);
      const png = encodePNG(TILE_SIZE, TILE_SIZE, rgba);
      return sendPng(res, png);
    }

    // GET /volume/:station/:product.json?time=...
    // Streams NDJSON. One radial per line. Server memory stays flat: one record at a time.
    m = path.match(/^\/volume\/([A-Z]{3,4})\/([A-Z0-9]+)\.json$/i);
    if (m) {
      const station = m[1].toUpperCase();
      const product = m[2].toUpperCase();
      let time = url.searchParams.get('time');

      if (!time || time === 'latest') {
        const today = new Date().toISOString().slice(0, 10);
        let scans = await listScans(station, today);
        if (scans.length === 0) {
          const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
          scans = await listScans(station, yesterday);
        }
        if (scans.length === 0) return sendJson(res, { error: 'No scans found' }, 404);
        time = scans[scans.length - 1].time;
      }

      let vol = await fetchVolumeFile(station, time);
      if (!vol) return sendJson(res, { error: 'No data' }, 404);

      const stationKey = station.replace(/^K/, '');
      const coords = STATION_COORDS[stationKey] || [0, 0];

      const acceptEncoding = req.headers['accept-encoding'] || '';
      const useGzip = acceptEncoding.includes('gzip');

      const headers = {
        'Content-Type': 'application/x-ndjson',
        'Cache-Control': 'public, max-age=120',
        'Access-Control-Allow-Origin': '*',
      };
      if (useGzip) headers['Content-Encoding'] = 'gzip';
      res.writeHead(200, headers);

      const gzip = useGzip ? zlib.createGzip({ level: 6 }) : null;
      if (useGzip) gzip.pipe(res);
      const sink = useGzip ? gzip : res;

      const write = (chunk) => new Promise(resolve => {
        if (sink.write(chunk)) resolve();
        else sink.once('drain', resolve);
      });

      const MAX_GATES = 1000;
      let targetElevNum = -1;
      let metaWritten = false;

      try {
        await iterateMessage31Records(vol, async (rec) => {
          // Lock onto the first elevation that contains the requested product
          if (targetElevNum === -1) {
            if (rec.moments.find(mo => mo.name === product)) {
              targetElevNum = rec.elevationNumber;
            } else {
              return; // keep scanning
            }
          }
          if (rec.elevationNumber !== targetElevNum) {
            return false; // past our sweep — stop iteration
          }

          const moment = rec.moments.find(mo => mo.name === product);
          if (!moment) return;

          if (!metaWritten) {
            await write(JSON.stringify({
              type: 'meta',
              site: station,
              product,
              timestamp: time,
              elevation: rec.elevation,
              station_lat: coords[0],
              station_lon: coords[1],
              gate_size_m: moment.gateSizeMeters,
              first_gate_m: moment.firstGateRange,
              scale: moment.scale,
              offset: moment.offset,
              num_gates: Math.min(moment.numGates, MAX_GATES),
            }) + '\n');
            metaWritten = true;
          }

          const numGates = Math.min(moment.numGates, MAX_GATES);
          const gates = new Array(numGates);
          if (moment.wordSize === 16) {
            for (let i = 0; i < numGates; i++) {
              gates[i] = (moment.data[i * 2] << 8) | moment.data[i * 2 + 1];
            }
          } else {
            for (let i = 0; i < numGates; i++) gates[i] = moment.data[i];
          }

          await write(JSON.stringify({ type: 'radial', azimuth: rec.azimuth, gates }) + '\n');
        });

        await write(JSON.stringify({ type: 'end' }) + '\n');
      } catch (err) {
        console.error('[/volume] stream error:', err.stack || err);
      } finally {
        vol = null;
        if (useGzip) gzip.end();
        else res.end();
      }
      return;
    }

    // GET /image/:station/:product.png?time=...&elevation=0.5&size=2048
    m = path.match(/^\/image\/([A-Z]{3,4})\/([A-Z0-9]+)\.png$/i);
    if (m) {
      const station = m[1].toUpperCase();
      const product = m[2].toUpperCase();
      const imgSize = Math.min(parseInt(url.searchParams.get('size') || '2048'), 4096);
      let time = url.searchParams.get('time');
      const elevation = parseFloat(url.searchParams.get('elevation') || '0.5');

      if (!time || time === 'latest') {
        const today = new Date().toISOString().slice(0, 10);
        let scans = await listScans(station, today);
        if (scans.length === 0) {
          const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
          scans = await listScans(station, yesterday);
        }
        if (scans.length === 0) return sendJson(res, { error: 'No scans found' }, 404);
        time = scans[scans.length - 1].time;
      }

      const vol = await fetchVolumeFile(station, time);
      if (!vol) return sendJson(res, { error: 'No data' }, 404);

      const parsed = parseLevel2(vol);
      const sweeps = parsed.sweeps.filter(s => s.product === product);
      if (sweeps.length === 0) return sendJson(res, { error: `No ${product} data` }, 404);

      const sweep = sweeps.reduce((a, b) =>
        Math.abs(a.elevation - elevation) <= Math.abs(b.elevation - elevation) ? a : b
      );

      const maxRange = sweep.firstGateRange + sweep.numGates * sweep.gateSizeMeters;
      const rangeDeg = (maxRange / EARTH_R) / DEG2RAD;
      const lonScale = rangeDeg / Math.cos(sweep.stationLat * DEG2RAD);

      const minLat = sweep.stationLat - rangeDeg;
      const maxLat = sweep.stationLat + rangeDeg;
      const minLon = sweep.stationLon - lonScale;
      const maxLon = sweep.stationLon + lonScale;

      const rgba = new Uint8Array(imgSize * imgSize * 4);
      const numRadials = sweep.radials.length;
      if (numRadials > 0) {
        const azimuths = sweep.radials.map(r => r.azimuth);
        for (let py = 0; py < imgSize; py++) {
          const lat = maxLat - (py / imgSize) * (maxLat - minLat);
          for (let px = 0; px < imgSize; px++) {
            const lon = minLon + (px / imgSize) * (maxLon - minLon);
            const dist = haversine(sweep.stationLat, sweep.stationLon, lat, lon);
            if (dist < sweep.firstGateRange || dist > maxRange) continue;
            const az = bearing(sweep.stationLat, sweep.stationLon, lat, lon);
            let lo = 0, hi = numRadials - 1;
            while (lo < hi) { const mid = (lo + hi) >> 1; azimuths[mid] < az ? lo = mid + 1 : hi = mid; }
            let best = lo;
            let bestDiff = Math.min(Math.abs(azimuths[lo] - az), 360 - Math.abs(azimuths[lo] - az));
            for (const idx of [lo - 1, lo + 1, 0, numRadials - 1]) {
              if (idx >= 0 && idx < numRadials) {
                const d = Math.min(Math.abs(azimuths[idx] - az), 360 - Math.abs(azimuths[idx] - az));
                if (d < bestDiff) { bestDiff = d; best = idx; }
              }
            }
            const radial = sweep.radials[best];
            const gateIdx = Math.floor((dist - sweep.firstGateRange) / sweep.gateSizeMeters);
            if (gateIdx < 0 || gateIdx >= radial.gates.length) continue;
            const raw = radial.gates[gateIdx];
            if (raw <= 1) continue;
            const phys = (raw - sweep.offset) / sweep.scale;
            const [r, g, b, a] = getColor(sweep.product, phys);
            if (a === 0) continue;
            const idx = (py * imgSize + px) * 4;
            rgba[idx] = r; rgba[idx + 1] = g; rgba[idx + 2] = b; rgba[idx + 3] = a;
          }
        }
      }

      const png = encodePNG(imgSize, imgSize, rgba);
      res.writeHead(200, {
        'Content-Type': 'image/png',
        'Cache-Control': 'public, max-age=120',
        'Access-Control-Allow-Origin': '*',
        'X-Radar-Bbox': `${minLat},${minLon},${maxLat},${maxLon}`,
      });
      return res.end(png);
    }

    // GET /debug/parse/:station — test download + parse without tile rendering
    m = path.match(/^\/debug\/parse\/([A-Z]{3,4})$/i);
    if (m) {
      const station = m[1].toUpperCase();
      const t0 = Date.now();
      try {
        const today = new Date().toISOString().slice(0, 10);
        let scans = await listScans(station, today);
        const listMs = Date.now() - t0;
        if (scans.length === 0) return sendJson(res, { error: 'No scans', listMs });

        const latest = scans[scans.length - 1];
        const t1 = Date.now();
        const vol = await fetchVolumeFile(station, latest.time);
        const fetchMs = Date.now() - t1;
        if (!vol) return sendJson(res, { error: 'Fetch failed', listMs, fetchMs });

        const t2 = Date.now();
        const parsed = parseLevel2(vol);
        const parseMs = Date.now() - t2;

        const products = [...new Set(parsed.sweeps.map(s => s.product))];
        return sendJson(res, {
          station, scan: latest.time, fileSize: vol.byteLength,
          sweeps: parsed.sweeps.length, products,
          timing: { listMs, fetchMs, parseMs, totalMs: Date.now() - t0 }
        });
      } catch (err) {
        return sendJson(res, { error: err.message, stack: err.stack, timing: { totalMs: Date.now() - t0 } }, 500);
      }
    }

    // GET / — health check
    if (path === '/' || path === '/health') {
      return sendJson(res, { status: 'ok', endpoints: [
        'GET /api/scans/{STATION}?date=YYYY-MM-DD',
        'GET /api/scans/{STATION}/latest',
        'GET /api/products/{STATION}?time=YYYYMMDD_HHMMSS',
        'GET /tiles/{STATION}/{PRODUCT}/{z}/{x}/{y}.png?time=YYYYMMDD_HHMMSS&elevation=0.5',
        'GET /debug/parse/{STATION}',
        'GET /satellite/geocolor',
        'GET /satellite/visible',
        'GET /satellite/meta',
        'GET /satellite/:product/tiles/:z/:x/:y.png',
      ]});
    }

    sendJson(res, { error: 'Not found' }, 404);
  } catch (err) {
    console.error('Request error:', err.stack || err);
    sendJson(res, { error: err.message, stack: err.stack }, 500);
  }
});

server.timeout = 120000;
server.listen(PORT, () => {
  console.log(`NEXRAD Level II server listening on port ${PORT}`);
});

// Pre-fetch both satellite JPEGs into cache at startup so the first tile
// request hits the in-memory buffer rather than triggering a sidecar fetch.
async function prefetchSatelliteImages() {
  for (const product of ['geocolor', 'visible']) {
    try {
      await fetchSatJpeg(product);
      console.log(`[satellite] pre-fetched ${product}`);
    } catch (e) {
      console.warn(`[satellite] pre-fetch failed for ${product}:`, e.message);
    }
  }
}
prefetchSatelliteImages();
