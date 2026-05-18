const http = require('http');
const zlib = require('zlib');
const { URL } = require('url');
const seekBzip = require('seek-bzip');

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
      const gates = [];
      if (m.wordSize === 16) {
        for (let i = 0; i < m.data.length - 1; i += 2) gates.push((m.data[i] << 8) | m.data[i + 1]);
      } else {
        for (let i = 0; i < m.data.length; i++) gates.push(m.data[i]);
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

  try {
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

    // GET /volume/:station/:product.json?time=...&elevation=0.5
    m = path.match(/^\/volume\/([A-Z]{3,4})\/([A-Z0-9]+)\.json$/i);
    if (m) {
      const station = m[1].toUpperCase();
      const product = m[2].toUpperCase();
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

      const body = JSON.stringify({
        site: station,
        product: sweep.product,
        timestamp: time,
        elevation: sweep.elevation,
        station_lat: sweep.stationLat,
        station_lon: sweep.stationLon,
        gate_size_m: sweep.gateSizeMeters,
        first_gate_m: sweep.firstGateRange,
        scale: sweep.scale,
        offset: sweep.offset,
        num_gates: sweep.numGates,
        radials: sweep.radials,
      });

      const acceptEncoding = req.headers['accept-encoding'] || '';
      if (acceptEncoding.includes('gzip')) {
        const compressed = zlib.gzipSync(body);
        res.writeHead(200, {
          'Content-Type': 'application/json',
          'Content-Encoding': 'gzip',
          'Cache-Control': 'public, max-age=120',
          'Access-Control-Allow-Origin': '*',
        });
        return res.end(compressed);
      }
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, max-age=120',
        'Access-Control-Allow-Origin': '*',
      });
      return res.end(body);
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
