import asyncio, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
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

        # Inject mesonet obs into RTMA surface fields before blending.
        # Runs in thread pool so the event loop stays free to serve /blend/all
        # requests while the IDW analysis (CPU + memory bound) is in progress.
        if stations:
            try:
                rtma = await loop.run_in_executor(
                    _thread_pool, inject_observations, rtma, stations
                )
            except Exception as e:
                log.warning(f'[mesonet] inject_observations raised: {e} — '
                            f'continuing with raw RTMA', exc_info=True)
        else:
            log.warning('[mesonet] no obs available this cycle — using raw RTMA')

        # Blend derivation is also CPU-heavy (gradient fields, interpolation).
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
