import runpy
import sys


def main():
    sys.argv = ["test_ray_async.py", "xxx", "xxx"]
    runpy.run_path("local-bak/scratch/test_ray_async.py", run_name="__main__")


import torch.distributed.rpc
