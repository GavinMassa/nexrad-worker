import asyncio, json, logging, os
from pathlib import Path
from aiohttp import web

log = logging.getLogger(__name__)
OUT_DIR = Path('/app/sidecar-out')
PORT = int(os.environ.get('SIDECAR_PORT', '4000'))

async def handle_all(request):
    meta_path = OUT_DIR / 'meta.json'
    if not meta_path.exists():
        return web.Response(status=503, text='sidecar not ready')

    try:
        meta = json.loads(meta_path.read_text())
        params = meta['params']

        # Concatenate all param .bin files in order.
        # Guard against an atomic rename race (writer.py writes .tmp → final):
        # if read_bytes() throws (e.g. file briefly missing mid-rename), return
        # 503 rather than crashing — the next request will succeed.
        chunks = []
        for param in params:
            bin_path = OUT_DIR / f'{param}.bin'
            if not bin_path.exists():
                log.warning(f'handle_all: missing bin for param={param}')
                return web.Response(status=503, text=f'missing bin: {param}')
            try:
                chunks.append(bin_path.read_bytes())
            except OSError as e:
                log.warning(f'handle_all: read failed for param={param}: {e}')
                return web.Response(status=503, text=f'bin read error: {param}')

        body = b''.join(chunks)
        meta['bytes_per_param'] = meta['nx'] * meta['ny'] * 4
        return web.Response(
            body=body,
            content_type='application/octet-stream',
            headers={
                'X-Meso-Meta':                   json.dumps(meta),
                'Access-Control-Expose-Headers': 'X-Meso-Meta',
                'Cache-Control':                 'no-store',
            },
        )
    except Exception as e:
        log.error(f'handle_all error: {e}', exc_info=True)
        return web.Response(status=500, text=str(e))

async def handle_status(request):
    meta_path = OUT_DIR / 'meta.json'
    if not meta_path.exists():
        return web.json_response({'ready': False})
    try:
        meta = json.loads(meta_path.read_text())
        return web.json_response({'ready': True, **meta})
    except Exception as e:
        return web.json_response({'ready': False, 'error': str(e)})

async def handle_health(request):
    """Lightweight liveness probe — always returns 200 immediately.
    Keeps Railway from killing the container during long IDW/blend cycles."""
    return web.Response(text='ok')

async def handle_satellite_product(request):
    """Serve a pre-reprojected GOES-19 satellite JPEG written by satellite_worker().
    product: 'geocolor' or 'visible'"""
    product = request.match_info['product']
    if product not in ('geocolor', 'visible'):
        return web.Response(status=400, text='unknown product; use geocolor or visible')
    fpath = OUT_DIR / f'satellite_{product}.jpg'
    if not fpath.exists():
        return web.Response(status=503, text=f'satellite_{product}.jpg not ready yet')
    try:
        data = fpath.read_bytes()
        return web.Response(
            body=data,
            content_type='image/jpeg',
            headers={'Cache-Control': 'public, max-age=270'},
        )
    except Exception as e:
        log.error(f'handle_satellite_product error: {e}', exc_info=True)
        return web.Response(status=500, text=str(e))

async def handle_satellite_meta(request):
    """Serve satellite_meta.json written by satellite_worker()."""
    fpath = OUT_DIR / 'satellite_meta.json'
    if not fpath.exists():
        return web.Response(status=503, text='satellite_meta.json not ready yet')
    try:
        data = fpath.read_text()
        return web.Response(
            text=data,
            content_type='application/json',
            headers={'Cache-Control': 'public, max-age=60'},
        )
    except Exception as e:
        log.error(f'handle_satellite_meta error: {e}', exc_info=True)
        return web.Response(status=500, text=str(e))

async def start_server():
    app = web.Application()
    app.router.add_get('/blend/all',           handle_all)
    app.router.add_get('/blend/status',        handle_status)
    app.router.add_get('/health',              handle_health)
    app.router.add_get('/healthz',             handle_health)
    # /satellite/meta must be registered before /satellite/{product} so the
    # literal path wins over the parameterised route in aiohttp's router.
    app.router.add_get('/satellite/meta',      handle_satellite_meta)
    app.router.add_get('/satellite/{product}', handle_satellite_product)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    log.info(f'Sidecar HTTP server listening on port {PORT}')
    return runner
