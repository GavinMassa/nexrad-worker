import asyncio, logging, json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from scipy.ndimage import zoom as ndimage_zoom
from fetch_rtma import fetch_rtma
from fetch_rap import fetch_rap
from mesonet import fetch_mesonet_obs, spatial_thin, compute_correction
from blend import blend as do_blend
from writer import write_output, OUT_DIR

logging.basicConfig(level=logging.INFO, format='[sidecar] %(message)s')
log = logging.getLogger(__name__)

# Shared thread pool for CPU-bound work (IDW + blend derivation).
# max_workers=2 — both tasks are numpy-heavy; more workers contend on the
# same cores and RAM without benefit.
_thread_pool = ThreadPoolExecutor(max_workers=2)
_cycle_lock  = asyncio.Lock()

# Mesonet correction files written by mesonet_worker(), read by run_cycle().
# Stored in the same OUT_DIR (/app/sidecar-out) so they persist across restarts.
MESONET_DT_PATH   = OUT_DIR / 'mesonet_delta_t.bin'
MESONET_DTD_PATH  = OUT_DIR / 'mesonet_delta_td.bin'
MESONET_META_PATH = OUT_DIR / 'mesonet_meta.json'
MESONET_MAX_AGE_S = 90 * 60   # discard corrections older than 90 minutes


# ── Mesonet background worker ─────────────────────────────────────────────────

async def mesonet_worker():
    """
    Fetch IEM obs every 10 minutes, thin, run IDW on the 399×586 blend grid,
    write correction files atomically. Completely independent of run_cycle() —
    never blocks it and never crashes the process (all exceptions caught).

    Correction files are read by run_cycle() and applied in <1 s via a simple
    array read + upsample, with no IDW running inline with the blend pipeline.
    """
    FACTOR = 0.25

    while True:
        try:
            # Need meta.json to know the blend output grid dimensions and bbox.
            # Wait (non-blocking) until the first blend cycle has completed.
            meta_path = OUT_DIR / 'meta.json'
            if not meta_path.exists():
                log.info('[mesonet] waiting for first blend cycle to complete...')
                await asyncio.sleep(60)
                continue

            meta     = json.loads(meta_path.read_text())
            ny_small = meta['ny']
            nx_small = meta['nx']

            stations = await fetch_mesonet_obs()
            if not stations:
                log.warning('[mesonet] no obs — skipping correction update')
                await asyncio.sleep(600)
                continue

            # Restrict to the RTMA domain bbox from meta.json
            lat_min = meta['lat_min'];  lat_max = meta['lat_max']
            lon_min = meta['lon_min'];  lon_max = meta['lon_max']
            stations_domain = [
                s for s in stations
                if lat_min <= s['lat'] <= lat_max
                and lon_min <= s['lon'] <= lon_max
            ]
            log.info(f'[mesonet] {len(stations_domain)}/{len(stations)} in domain')

            # Spatial thinning — reduces 8000+ raw stations to ~500-800
            thinned = spatial_thin(stations_domain)
            if len(thinned) < 10:
                log.warning('[mesonet] too few stations after thinning — skipping')
                await asyncio.sleep(600)
                continue

            # Reconstruct the blend output lat/lon grid from bbox + dimensions.
            # Row 0 = lat_max (N→S, matching writer.py flip convention).
            lats_1d = np.linspace(meta['lat_max'], meta['lat_min'], ny_small,
                                  dtype=np.float32)
            lons_1d = np.linspace(meta['lon_min'], meta['lon_max'], nx_small,
                                  dtype=np.float32)
            grid_lons, grid_lats = np.meshgrid(lons_1d, lats_1d)

            # RTMA reference grids (t2m / td2m at 399×586) are written by
            # run_cycle() after each blend. Wait if not available yet.
            t2m_path  = OUT_DIR / 'rtma_t2m_small.bin'
            td2m_path = OUT_DIR / 'rtma_td2m_small.bin'
            if not t2m_path.exists() or not td2m_path.exists():
                log.info('[mesonet] waiting for RTMA reference grids...')
                await asyncio.sleep(60)
                continue

            grid_t2m  = np.frombuffer(t2m_path.read_bytes(),
                                      dtype=np.float32).reshape(ny_small, nx_small)
            grid_td2m = np.frombuffer(td2m_path.read_bytes(),
                                      dtype=np.float32).reshape(ny_small, nx_small)

            # IDW is CPU-bound but fast with the thinned network (~600 stations)
            loop = asyncio.get_running_loop()
            delta_t, delta_td = await loop.run_in_executor(
                _thread_pool, compute_correction,
                thinned, grid_lats, grid_lons, grid_t2m, grid_td2m,
            )

            # Write correction files atomically — run_cycle() may read at any time
            for fpath, arr in [(MESONET_DT_PATH, delta_t), (MESONET_DTD_PATH, delta_td)]:
                tmp = fpath.parent / (fpath.name + '.tmp')
                arr.tofile(str(tmp))
                tmp.replace(fpath)

            meta_corr = {
                'nx':         nx_small,
                'ny':         ny_small,
                'n_stations': len(thinned),
                'updated':    datetime.now(timezone.utc).isoformat(),
            }
            tmp_m = MESONET_META_PATH.parent / (MESONET_META_PATH.name + '.tmp')
            tmp_m.write_text(json.dumps(meta_corr))
            tmp_m.replace(MESONET_META_PATH)
            log.info(f'[mesonet] correction written: {len(thinned)} stations, '
                     f'{ny_small}×{nx_small} grid')

        except Exception as e:
            log.error(f'[mesonet] worker error: {e}', exc_info=True)

        await asyncio.sleep(600)   # 10-minute interval


# ── Hourly blend cycle ────────────────────────────────────────────────────────

async def run_cycle():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    log.info(f'Starting cycle for {now.strftime("%Y-%m-%d %H:00Z")}')
    if _cycle_lock.locked():
        log.warning('Previous cycle still running — skipping')
        return
    async with _cycle_lock:
        try:
            rtma_task = asyncio.create_task(fetch_rtma(now))
            rap_task  = asyncio.create_task(fetch_rap(now))
            rtma, rap = await asyncio.gather(rtma_task, rap_task)

            if rtma is None:
                log.warning('RTMA fetch returned None — skipping cycle')
                return
            if rap is None:
                log.warning('RAP fetch returned None — writing RTMA-only output')
                write_output(rtma, now)
                return

            FACTOR = 0.25
            loop   = asyncio.get_running_loop()

            # Save downsampled RTMA T/Td as the background reference for
            # mesonet_worker(). Atomic write so the worker never reads partial data.
            def _save_rtma_ref(rtma_full):
                t_small  = ndimage_zoom(rtma_full['t2m'],  FACTOR, order=1).astype(np.float32)
                td_small = ndimage_zoom(rtma_full['td2m'], FACTOR, order=1).astype(np.float32)
                for fpath, arr in [(OUT_DIR / 'rtma_t2m_small.bin',  t_small),
                                   (OUT_DIR / 'rtma_td2m_small.bin', td_small)]:
                    tmp = fpath.parent / (fpath.name + '.tmp')
                    arr.tofile(str(tmp))
                    tmp.replace(fpath)
                log.info(f'[mesonet] saved RTMA reference grid {t_small.shape}')

            await loop.run_in_executor(_thread_pool, _save_rtma_ref, rtma)

            # Apply the most recent mesonet correction if available and fresh.
            # This is fast (<1 s): read two small .bin files + two zoom calls.
            # The correction was computed by mesonet_worker() in the background.
            if (MESONET_META_PATH.exists()
                    and MESONET_DT_PATH.exists()
                    and MESONET_DTD_PATH.exists()):
                try:
                    corr_meta = json.loads(MESONET_META_PATH.read_text())
                    updated   = datetime.fromisoformat(corr_meta['updated'])
                    age_s     = (datetime.now(timezone.utc) - updated).total_seconds()
                    if age_s < MESONET_MAX_AGE_S:
                        ny_c = corr_meta['ny'];  nx_c = corr_meta['nx']
                        dt   = np.frombuffer(MESONET_DT_PATH.read_bytes(),
                                             dtype=np.float32).reshape(ny_c, nx_c)
                        dtd  = np.frombuffer(MESONET_DTD_PATH.read_bytes(),
                                             dtype=np.float32).reshape(ny_c, nx_c)
                        # Upsample correction to full RTMA resolution, crop ±1px
                        UP = 1.0 / FACTOR
                        ny_full, nx_full = rtma['t2m'].shape
                        dt_full  = ndimage_zoom(dt,  UP, order=1)[:ny_full, :nx_full]
                        dtd_full = ndimage_zoom(dtd, UP, order=1)[:ny_full, :nx_full]
                        rtma = dict(rtma)
                        rtma['t2m']  = np.clip(
                            rtma['t2m']  + dt_full,  200.0, 340.0
                        ).astype(np.float32)
                        rtma['td2m'] = np.clip(
                            rtma['td2m'] + dtd_full, 200.0, 320.0
                        ).astype(np.float32)
                        log.info(f'[mesonet] applied correction '
                                 f'(age={age_s/60:.0f} min, '
                                 f'stations={corr_meta["n_stations"]})')
                    else:
                        log.warning(f'[mesonet] correction stale '
                                    f'({age_s/60:.0f} min) — skipping')
                except Exception as e:
                    log.warning(f'[mesonet] correction read failed: {e} '
                                f'— using raw RTMA')

            # Blend derivation is CPU-heavy; off-loaded to thread pool.
            blended = await loop.run_in_executor(_thread_pool, do_blend, rtma, rap)
            write_output(blended, now)
            log.info('Cycle complete')
        except Exception as e:
            log.error(f'Cycle failed: {e}', exc_info=True)


# ── Scheduler ─────────────────────────────────────────────────────────────────

async def scheduler():
    await run_cycle()   # run immediately on startup
    while True:
        now      = datetime.now(timezone.utc)
        next_run = now.replace(minute=28, second=0, microsecond=0)
        if now >= next_run:
            # timedelta addition handles hour=23 → next day correctly.
            next_run = (next_run + timedelta(hours=1)).replace(
                minute=28, second=0, microsecond=0
            )
        wait = (next_run - now).total_seconds()
        log.info(f'Next cycle in {wait:.0f}s at {next_run.strftime("%H:%M")}Z')
        await asyncio.sleep(wait)
        await run_cycle()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    from server import start_server
    runner = await start_server()
    try:
        # scheduler() and mesonet_worker() run concurrently on the same event loop.
        # They share _thread_pool but never block each other — scheduler holds
        # _cycle_lock during blend; mesonet_worker uses separate file paths.
        await asyncio.gather(
            scheduler(),
            mesonet_worker(),
        )
    finally:
        _thread_pool.shutdown(wait=False)
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
