import asyncio
import threading

import ray


class Actor:
    def __init__(self):
        print("Actor initialized.")
        print(threading.current_thread().name)  # default thread

    def method_sync(self):
        print("This is a synchronous method.")
        print(threading.current_thread().name)  # default thread

    async def method_async(self):
        print("This is an asynchronous method.")
        print(threading.current_thread().name)  # default thread
        print(asyncio.get_running_loop())
        print("All threads:")
        for thread in threading.enumerate():
            print(thread.name, "is_main:", thread == threading.main_thread())

        asyncio.wait


if __name__ == "__main__":
    actor = ray.remote(Actor).remote()
    ref1 = actor.method_sync.remote()  # This will run in the main thread
    ref2 = actor.method_async.remote()  # This will run in the main thread
    ray.get([ref1, ref2])  # Wait for both methods to complete
