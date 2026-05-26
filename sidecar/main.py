import asyncio, logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from scipy.ndimage import zoom
from fetch_rtma import fetch_rtma
from fetch_rap import fetch_rap
from blend import blend as do_blend
from writer import write_output

logging.basicConfig(level=logging.INFO, format='[sidecar] %(message)s')
log = logging.getLogger(__name__)

# Shared thread pool for CPU-bound work (blend derivation).
# max_workers=2 — numpy-heavy and memory-intensive.
_thread_pool = ThreadPoolExecutor(max_workers=2)


async def run_cycle():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    log.info(f'Starting cycle for {now.strftime("%Y-%m-%d %H:00Z")}')
    try:
        # Fetch RTMA and RAP concurrently — independent downloads.
        # mesonet removed — crashes sidecar due to memory pressure on the
        # full 1597×2345 grid; needs redesign to operate at 400×586 from the start.
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

        # Mesonet obs injection disabled — not production-stable yet.
        # The IDW distance matrix on any chunk of the full grid exceeds
        # Railway hobby container memory regardless of row-chunking strategy.
        # Re-enable once mesonet module is redesigned to work on the
        # downsampled 400×586 grid natively.
        #
        # if stations:
        #     try:
        #         rtma = await loop.run_in_executor(
        #             _thread_pool, _inject_downsampled, rtma, stations
        #         )
        #     except Exception as e:
        #         log.warning(f'[mesonet] injection failed: {e} — using raw RTMA',
        #                     exc_info=True)
        # else:
        #     log.warning('[mesonet] no obs available this cycle — using raw RTMA')

        loop = asyncio.get_running_loop()

        # Blend derivation is CPU-heavy (gradient fields, RAP interpolation).
        # Off-loaded to thread pool so the event loop stays free.
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
