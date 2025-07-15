import threading
import asyncio
from dask.distributed.scheduler import Scheduler
from queue import Queue


async def scheduler(q: Queue):
    sched = await Scheduler()
    q.put(sched.address)
    await sched.finished()

def start_scheduler():
    """
    Launch the Dask scheduler in a background daemon thread.
    Returns the Thread object in case you want to join() later.
    """
    q = Queue()
    def _runner():
        asyncio.run(scheduler(q))

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t, q.get()