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
# max_workers=2 — both tasks are numpy-heavy and memory-intensive; more than
# 2 concurrent workers would contend on the same physical cores and RAM.
_thread_pool = ThreadPoolExecutor(max_workers=2)

# Must match DOWNSAMPLE_FACTOR in writer.py so the IDW runs on the same
# resolution grid that ultimately gets written to disk.
_IDW_FACTOR = 0.25


def _inject_downsampled(rtma: dict, stations: list) -> dict:
    """
    Run mesonet IDW on a 0.25× downsampled RTMA grid, then upsample the
    T/Td correction back to full resolution and apply it.

    Why: the chunk-based IDW on the full 1597×2345 grid allocates
    ~50 rows × 2345 cols × N_stations × 4 bytes per chunk.  At 8000+
    stations that's ~3-4 GB per chunk — the process OOMs before finishing
    a single row-slice.  The observation network spacing is 30-50 km, so
    there is no information in the IDW correction at 2.5 km resolution;
    running at 10 km (0.25× = ~400×586) is physically equivalent and
    reduces the distance matrix 16× to ~240 MB peak.

    Returns a new rtma dict with corrected full-resolution t2m / td2m.
    All other keys (lats, lons, u10, v10) are carried over unchanged.
    """
    t2m_orig  = rtma['t2m']    # (1597, 2345) float32
    td2m_orig = rtma['td2m']
    ny, nx    = t2m_orig.shape

    # Downsample all fields needed by inject_observations
    lats_small = zoom(rtma['lats'], _IDW_FACTOR, order=1)
    lons_small = zoom(rtma['lons'], _IDW_FACTOR, order=1)
    t2m_small  = zoom(t2m_orig,    _IDW_FACTOR, order=1)
    td2m_small = zoom(td2m_orig,   _IDW_FACTOR, order=1)

    rtma_small = {
        'lats': lats_small,
        'lons': lons_small,
        't2m':  t2m_small,
        'td2m': td2m_small,
        'u10':  zoom(rtma['u10'], _IDW_FACTOR, order=1),
        'v10':  zoom(rtma['v10'], _IDW_FACTOR, order=1),
    }

    log.info(f'[mesonet] IDW running on downsampled grid '
             f'{t2m_small.shape[0]}×{t2m_small.shape[1]} '
             f'(full={ny}×{nx})')

    # Run injection on the small grid — mesonet.py is unchanged
    rtma_small_corrected = inject_observations(rtma_small, stations)

    # Compute delta on small grid, upsample to full resolution
    upsample      = 1.0 / _IDW_FACTOR
    delta_t_full  = zoom(rtma_small_corrected['t2m']  - t2m_small,  upsample, order=1)
    delta_td_full = zoom(rtma_small_corrected['td2m'] - td2m_small, upsample, order=1)

    # Crop to exact original shape — zoom rounds and may produce ny±1 / nx±1
    delta_t_full  = delta_t_full [:ny, :nx]
    delta_td_full = delta_td_full[:ny, :nx]

    corrected = dict(rtma)    # shallow copy — carries lats, lons, u10, v10 through
    corrected['t2m']  = (t2m_orig  + delta_t_full ).astype(np.float32)
    corrected['td2m'] = (td2m_orig + delta_td_full).astype(np.float32)
    return corrected


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

        # Inject mesonet obs via downsampled IDW (see _inject_downsampled above).
        # Runs in thread pool so the event loop stays free to serve /blend/all.
        if stations:
            try:
                rtma = await loop.run_in_executor(
                    _thread_pool, _inject_downsampled, rtma, stations
                )
            except Exception as e:
                log.warning(f'[mesonet] injection failed: {e} — '
                            f'continuing with raw RTMA', exc_info=True)
        else:
            log.warning('[mesonet] no obs available this cycle — using raw RTMA')

        # Blend derivation is also CPU-heavy (gradient fields, RAP interpolation).
        # Off-loaded to the same thread pool for the same reason.
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
