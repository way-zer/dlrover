import asyncio
import multiprocessing
import threading
import time
import traceback

import ray


class STATE:
    VAR: str = "INIT"


@ray.remote
def work():
    print(threading.current_thread().name, traceback.format_stack())


@ray.remote(concurrency_groups={"main": 1})
class Worker:
    def __init__(self, master: "Master"):
        self.master = master
        print("INIT",threading.current_thread().name, traceback.format_stack())

    @ray.method(concurrency_group="main")
    def run(self):
        print("Starting worker")
        print("RUN", threading.current_thread().name, traceback.format_stack())
        print(f"hello {ray.get(self.master.get_name.remote())}")
        ray.get(work.remote())
        time.sleep(2)
        print("END worker")

    def other(self):
        print("OTHER", threading.current_thread().name)
        time.sleep(2)


@ray.remote
class Master:
    async def start(self):
        worker = Worker.remote(ray.get_runtime_context().current_actor)
        job = worker.run.remote()
        print("Master started")
        await worker.other.remote()
        await job

    async def get_name(self):
        return "World"


def main():
    master = Master.remote()
    ray.get(master.start.remote())


if __name__ == "__main__":
    main()
