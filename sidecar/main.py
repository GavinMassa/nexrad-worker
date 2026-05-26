import asyncio, logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from scipy.ndimage import zoom
from fetch_rtma import fetch_rtma
from fetch_rap import fetch_rap
from mesonet import fetch_mesonet_obs, inject_observations
from blend import blend as do_blend
from writer import write_output

logging.basicConfig(level=logging.INFO, format='[sidecar] %(message)s')
log = logging.getLogger(__name__)

# Shared thread pool for CPU-bound work (IDW analysis + blend derivation).
# max_workers=2 — numpy-heavy and memory-intensive; more workers would
# contend on the same physical cores and RAM.
_thread_pool = ThreadPoolExecutor(max_workers=2)

# Must match DOWNSAMPLE_FACTOR in writer.py.  The IDW runs on the same
# 399×586 grid that ultimately gets written to disk — no information is
# lost because the obs network spacing (30-50 km) is coarser than this.
_IDW_FACTOR = 0.25


def _inject_downsampled(rtma_full: dict, stations: list) -> dict:
    """
    Downsample RTMA to 0.25×, run IDW obs injection on the small grid,
    upsample the correction back to full resolution, and apply.

    Memory budget:
      Full grid:  1597×2345 × 8800 stations × 4 bytes per chunk → OOM
      Small grid: 399×586   × 8800 stations × 4 bytes (full matrix) ≈ 8 MB ✓

    inject_observations() in mesonet.py is unchanged — it just receives a
    smaller dict and returns a corrected version of it.
    """
    t2m_orig  = rtma_full['t2m']
    td2m_orig = rtma_full['td2m']
    ny_full, nx_full = t2m_orig.shape

    # Downsample all RTMA fields needed by inject_observations
    small = {
        't2m':  zoom(t2m_orig,          _IDW_FACTOR, order=1),
        'td2m': zoom(td2m_orig,         _IDW_FACTOR, order=1),
        'u10':  zoom(rtma_full['u10'],  _IDW_FACTOR, order=1),
        'v10':  zoom(rtma_full['v10'],  _IDW_FACTOR, order=1),
        'lats': zoom(rtma_full['lats'], _IDW_FACTOR, order=1),
        'lons': zoom(rtma_full['lons'], _IDW_FACTOR, order=1),
    }
    log.info(f'[mesonet] IDW on downsampled grid '
             f'{small["t2m"].shape[0]}×{small["t2m"].shape[1]} '
             f'(full={ny_full}×{nx_full})')

    # Run IDW on the small grid — mesonet.py compute_innovation_field now
    # allocates the full distance matrix without chunking, which is safe here
    corrected_small = inject_observations(small, stations)

    # Upsample the T/Td delta back to full resolution
    UP = 1.0 / _IDW_FACTOR
    delta_t  = zoom(corrected_small['t2m']  - small['t2m'],  UP, order=1)
    delta_td = zoom(corrected_small['td2m'] - small['td2m'], UP, order=1)

    # Crop to exact original shape — zoom rounding can produce ±1 pixel
    delta_t  = delta_t [:ny_full, :nx_full]
    delta_td = delta_td[:ny_full, :nx_full]

    corrected_full = dict(rtma_full)    # shallow copy — lats/lons/u10/v10 pass through
    corrected_full['t2m']  = (t2m_orig  + delta_t ).astype(np.float32)
    corrected_full['td2m'] = (td2m_orig + delta_td).astype(np.float32)
    return corrected_full


async def run_cycle():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    log.info(f'Starting cycle for {now.strftime("%Y-%m-%d %H:00Z")}')
    try:
        # Fetch RTMA, RAP, and mesonet obs concurrently — all three are
        # independent network requests; running in parallel saves ~20s per cycle.
        rtma_task    = asyncio.create_task(fetch_rtma(now))
        rap_task     = asyncio.create_task(fetch_rap(now))
        mesonet_task = asyncio.create_task(fetch_mesonet_obs())
        rtma, rap, stations = await asyncio.gather(rtma_task, rap_task, mesonet_task)

        if rtma is None:
            log.warning('RTMA fetch returned None — skipping cycle')
            return
        if rap is None:
            log.warning('RAP fetch returned None — writing RTMA-only output')
            write_output(rtma, now)
            return

        loop = asyncio.get_running_loop()

        # Inject mesonet obs via downsampled IDW — see _inject_downsampled above.
        # Runs in thread pool so the event loop stays free to serve /blend/all.
        # If injection raises for any reason, cycle continues with raw RTMA.
        if stations:
            try:
                rtma = await loop.run_in_executor(
                    _thread_pool, _inject_downsampled, rtma, stations
                )
                log.info('[mesonet] injection complete')
            except Exception as e:
                log.warning(f'[mesonet] injection failed: {e} — using raw RTMA',
                            exc_info=True)
        else:
            log.warning('[mesonet] no obs available this cycle — using raw RTMA')

        # Blend derivation is CPU-heavy (gradient fields, RAP interpolation).
        # Off-loaded to thread pool so the event loop stays responsive.
        blended = await loop.run_in_executor(_thread_pool, do_blend, rtma, rap)
        write_output(blended, now)
        log.info('Cycle complete')
    except Exception as e:
        log.error(f'Cycle failed: {e}', exc_info=True)


async def scheduler():
    await run_cycle()  # run immediately on startup
    while True:
        now = datetime.now(timezone.utc)
        # Next :28 past the hour
        next_run = now.replace(minute=28, second=0, microsecond=0)
        if now >= next_run:
            # timedelta addition handles hour=23 → next day correctly.
            next_run = (next_run + timedelta(hours=1)).replace(minute=28, second=0, microsecond=0)
        wait = (next_run - now).total_seconds()
        log.info(f'Next cycle in {wait:.0f}s at {next_run.strftime("%H:%M")}Z')
        await asyncio.sleep(wait)
        await run_cycle()


async def main():
    from server import start_server
    runner = await start_server()
    try:
        await scheduler()
    finally:
        _thread_pool.shutdown(wait=False)
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
