import asyncio, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from fetch_rtma import fetch_rtma
from fetch_rap import fetch_rap
from blend import blend as do_blend
from writer import write_output

logging.basicConfig(level=logging.INFO, format='[sidecar] %(message)s')
log = logging.getLogger(__name__)

# Shared thread pool for CPU-bound work (IDW analysis + blend derivation).
# max_workers=2 — numpy-heavy and memory-intensive; more workers would
# contend on the same physical cores and RAM.
_thread_pool = ThreadPoolExecutor(max_workers=2)
_cycle_lock  = asyncio.Lock()


async def run_cycle():
    if _cycle_lock.locked():
        log.warning('Previous cycle still running — skipping this tick')
        return
    async with _cycle_lock:
        await _run_cycle_inner()


async def _run_cycle_inner():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    log.info(f'Starting cycle for {now.strftime("%Y-%m-%d %H:00Z")}')
    try:
        # Fetch RTMA and RAP concurrently — independent downloads.
        # mesonet disabled — causes OOM due to cycle overlap on Railway restart.
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

        loop = asyncio.get_running_loop()

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
