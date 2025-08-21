import ray


class Actor:
    async def debug(self):
        from remote_pdb import RemotePdb

        print("Debugging actor...")
        RemotePdb("127.0.0.1", 4444).set_trace()


if __name__ == "__main__":
    actor = ray.remote(Actor).remote()
    ref = actor.debug.remote()
    ray.get(ref)
