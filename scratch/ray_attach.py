import traceback

import ray


def start_debug():
    from remote_pdb import RemotePdb

    print("Debugging actor...")
    RemotePdb("127.0.0.1", 60001).set_trace()


import traceback

with open("debug.log") as file:
    traceback.print_exception(e, file=file)

if __name__ == "__main__":
    ray.init("auto")
    ref = ray.get_actor("ELASTIC_16-0_8-0").__ray_call__.remote(start_debug)
    print("Debugging started, waiting for connection...")
    ray.get(ref)
