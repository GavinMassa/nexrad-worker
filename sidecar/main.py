import asyncio, logging
from datetime import datetime, timezone
from fetch_rtma import fetch_rtma
from writer import write_output

logging.basicConfig(level=logging.INFO, format='[sidecar] %(message)s')
log = logging.getLogger(__name__)

async def run_cycle():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    log.info(f'Starting cycle for {now.strftime("%Y-%m-%d %H:00Z")}')
    try:
        rtma = await fetch_rtma(now)
        if rtma is None:
            log.warning('RTMA fetch returned None — skipping cycle')
            return
        write_output(rtma, now)
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
            next_run = next_run.replace(hour=next_run.hour + 1)
        wait = (next_run - now).total_seconds()
        log.info(f'Next cycle in {wait:.0f}s at {next_run.strftime("%H:%M")}Z')
        await asyncio.sleep(wait)
        await run_cycle()

if __name__ == '__main__':
    asyncio.run(scheduler())
