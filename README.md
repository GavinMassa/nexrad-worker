# NEXRAD Level II Radar Server

Node.js server that fetches, parses, and serves NEXRAD Level II radar data as map tiles. Supports both live and archived data from the Unidata S3 archive.

## API Endpoints

### List available scans
```
GET /api/scans/{STATION}?date=YYYY-MM-DD
```

### Get latest scan
```
GET /api/scans/{STATION}/latest
```

### Get available products
```
GET /api/products/{STATION}?time=YYYYMMDD_HHMMSS
```

### Get map tile
```
GET /tiles/{STATION}/{PRODUCT}/{z}/{x}/{y}.png?time=YYYYMMDD_HHMMSS&elevation=0.5
```

- **STATION**: 3-4 letter NEXRAD station ID (e.g., `KTLX`, `TLX`)
- **PRODUCT**: `REF`, `VEL`, `SW`, `ZDR`, `PHI`, or `RHO`
- **time**: `YYYYMMDD_HHMMSS` or ISO 8601, or `latest` for most recent scan
- **elevation**: Antenna elevation angle in degrees (default: 0.5)

## Deploy on Railway

1. Push this repo to GitHub
2. Create a new Railway project from the GitHub repo
3. Railway auto-detects Node.js and runs `npm start`
4. No environment variables needed — data comes from public S3

## Local Development

```bash
node server.js
# Server starts on http://localhost:3000
```

## Data Source

Unidata NEXRAD Level II archive on AWS: `s3://unidata-nexrad-level2/`
- Public, no authentication required
- Data available from 1991 to present
- Near-real-time data typically available within minutes
