# queue_sweep.py

import time
import queue
import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
import random


def worker(gpu_id: int, task_queue):
    """Worker loop: bind to GPU, pull jobs from queue, do dummy work."""
    torch.cuda.set_device(gpu_id)
    print(f"[Worker {gpu_id}] started, device: {torch.cuda.get_device_name(gpu_id)}")
    while True:
        try:
            run_id = task_queue.get_nowait()
        except queue.Empty:
            print(f"[Worker {gpu_id}] no more runs, exiting.")
            break

        print(f"[Worker {gpu_id}] starting run {run_id}")
        # --- your real training logic here ---
        # Dummy compute: matrix multiply and sleep
        x = torch.randn(500, 500, device=gpu_id)
        y = x @ x.T
        time.sleep(random.random())
        print(f"[Worker {gpu_id}] finished run {run_id}, norm={y.norm().item():.2f}")


def main():
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeError("No CUDA GPUs detected.")

    # Prepare a list of 20 “runs” (replace with your actual configs)
    runs = list(range(20))

    # Create a shared queue and enqueue all runs
    manager = Manager()
    task_queue = manager.Queue()
    for r in runs:
        task_queue.put(r)

    # Launch one process per GPU
    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu in range(ngpus):
        p = mp.Process(target=worker, args=(gpu, task_queue))
        p.start()
        processes.append(p)

    # Wait for all to complete
    for p in processes:
        p.join()

    print("All runs completed.")


if __name__ == "__main__":
    main()
