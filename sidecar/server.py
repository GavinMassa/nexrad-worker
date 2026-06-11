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

async def handle_blend_meta(request: web.Request) -> web.Response:
    """Lightweight metadata endpoint for Node revalidation.
    Returns meta.json (~200 bytes) so rap.js can check valid_time
    before deciding whether to fetch the full 14MB blend binary."""
    path = OUT_DIR / 'meta.json'
    if not path.exists():
        raise web.HTTPServiceUnavailable(reason='meta.json not ready')
    return web.Response(
        text=path.read_text(),
        content_type='application/json',
        headers={
            'Cache-Control':               'no-cache',
            'Access-Control-Allow-Origin': '*',
        },
    )

async def handle_blend_history(request: web.Request) -> web.Response:
    """Return history.json — list of available archived hour keys."""
    path = OUT_DIR / 'history.json'
    if not path.exists():
        raise web.HTTPServiceUnavailable(reason='history.json not ready')
    return web.Response(
        text=path.read_text(),
        content_type='application/json',
        headers={'Cache-Control':               'no-cache',
                 'Access-Control-Allow-Origin': '*'},
    )

async def handle_blend_hour(request: web.Request) -> web.Response:
    """Return archived blend binary for a specific hour (YYYYMMDDHH)."""
    hour = request.match_info['hour']
    if not (len(hour) == 10 and hour.isdigit()):
        raise web.HTTPBadRequest(reason='Invalid hour format (expected YYYYMMDDHH)')
    cycle_dir = OUT_DIR / hour
    meta_path = cycle_dir / 'meta.json'
    if not cycle_dir.exists() or not meta_path.exists():
        raise web.HTTPNotFound(reason=f'Hour {hour} not in archive')
    meta = json.loads(meta_path.read_text())
    params = meta.get('params', [])
    bpp = meta['nx'] * meta['ny'] * 4
    chunks = []
    for param in params:
        bin_path = cycle_dir / f'{param}.bin'
        if not bin_path.exists():
            raise web.HTTPInternalServerError(
                reason=f'Missing {param}.bin for hour {hour}')
        chunks.append(bin_path.read_bytes())
    body = b''.join(chunks)
    meta['bytes_per_param'] = bpp
    log.info(f'handle_blend_hour: serving hour={hour} params={len(params)} '
             f'body={len(body)//1024}KB')
    return web.Response(
        body=body,
        content_type='application/octet-stream',
        headers={
            'X-Meso-Meta':                   json.dumps(meta),
            'Access-Control-Expose-Headers': 'X-Meso-Meta',
            'Cache-Control':                 'public, max-age=86400',
            'Access-Control-Allow-Origin':   '*',
        },
    )

async def handle_health(request):
    """Lightweight liveness probe — always returns 200 immediately.
    Keeps Railway from killing the container during long IDW/blend cycles."""
    return web.Response(text='ok')

async def handle_satellite_product(request):
    product = request.match_info.get('product', '')
    if product not in ('geocolor', 'visible'):
        raise web.HTTPNotFound()
    path = OUT_DIR / f'satellite_{product}.jpg'
    if not path.exists():
        raise web.HTTPServiceUnavailable()
    return web.FileResponse(path, headers={
        'Content-Type':                'image/jpeg',
        'Cache-Control':               'public, max-age=270',
        'Access-Control-Allow-Origin': '*',
    })

async def handle_satellite_meta(request):
    path = OUT_DIR / 'satellite_meta.json'
    if not path.exists():
        raise web.HTTPServiceUnavailable()
    return web.Response(text=path.read_text(), content_type='application/json',
                        headers={'Cache-Control':               'public, max-age=60',
                                 'Access-Control-Allow-Origin': '*'})

async def start_server():
    app = web.Application()
    # Literal routes before parameterised so aiohttp never ambiguates them.
    app.router.add_get('/blend/history',       handle_blend_history)
    app.router.add_get('/blend/all',           handle_all)
    app.router.add_get('/blend/meta',          handle_blend_meta)
    app.router.add_get('/blend/status',        handle_status)
    # Parameterised route last — matches any /blend/<10-digit-hour>
    app.router.add_get('/blend/{hour}',        handle_blend_hour)
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
