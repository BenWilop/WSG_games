import queue
import torch as t
import torch.multiprocessing as mp
from multiprocessing import Manager
from dataclasses import dataclass
from typing import Callable, List, Any, Tuple


@dataclass
class Job:
    """
    Represents a job to be executed, containing the function and its arguments.
    """

    function: Callable  # needs to accept device as the first argument!
    args: Tuple[Any, ...]


def _gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
):
    """
    Generic worker function that runs on a specific GPU.
    It fetches jobs from the task_queue and executes them.
    """
    t.cuda.set_device(gpu_id)
    print(f"[Worker {gpu_id}] on {t.cuda.get_device_name(gpu_id)}")
    device = str(t.device(f"cuda:{gpu_id}"))

    while True:
        try:
            job: Job = task_queue.get_nowait()
            print(f"[Worker {gpu_id}] starting job: {job.function.__name__}")
            job.args = (
                device,
            ) + job.args  # Prepend device to args for the job function
            job.function(*job.args)
            print(f"[Worker {gpu_id}] finished job: {job.function.__name__}")
            t.cuda.empty_cache()
        except queue.Empty:
            print(f"[Worker {gpu_id}] no more tasks, exiting.")
            break
        except Exception as e:
            print(f"[Worker {gpu_id}] encountered an error: {e}")
            t.cuda.empty_cache()
            break


class ParallelGpuExecutor:
    """
    Manages a pool of GPU workers to execute a list of jobs in parallel.
    """

    def __init__(self, ngpus: int | None = None):
        if ngpus:
            self.ngpus = ngpus
        else:
            self.ngpus = t.cuda.device_count()
        if self.ngpus == 0:
            raise RuntimeError("No CUDA GPUs detected.")
        print(f"Found {self.ngpus} GPUs.")
        mp.set_start_method("spawn", force=True)
        self.manager = Manager()
        self.task_queue = self.manager.Queue()

    def submit_jobs(self, jobs: List[Job]):
        """
        Submits a list of jobs to the task queue and starts worker processes.
        """
        if not jobs:
            print("No jobs to submit.")
            return

        for job in jobs:
            self.task_queue.put(job)

        processes = []
        for gpu_id in range(self.ngpus):
            p = mp.Process(
                target=_gpu_worker,
                args=(
                    gpu_id,
                    self.task_queue,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("All submitted jobs completed.")
