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

    function: Callable
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
            print(
                f"[Worker {gpu_id}] starting job: {job.function.__name__} with args (first 5): {str(job.args)[:100]}..."
            )
            # Pass the device to the job function if it's designed to accept it
            # This requires the target function to potentially handle a 'device' kwarg
            # or the device to be part of job.args if explicitly needed by the function
            # For this refactor, we'll assume the function might need the device
            # and can be designed to accept it as the first argument or a keyword argument.
            # A common pattern is to pass it as a keyword argument if the function supports it.
            # Or, as in the original code, make device part of the arguments.
            # For simplicity, let's modify the job function to accept gpu_id or device as an arg.
            job.args = (
                gpu_id,
            ) + job.args  # Prepend gpu_id to args for the job function
            job.function(*job.args)
            print(f"[Worker {gpu_id}] finished job: {job.function.__name__}")
            t.cuda.empty_cache()
        except queue.Empty:
            print(f"[Worker {gpu_id}] no more tasks, exiting.")
            break
        except Exception as e:
            print(f"[Worker {gpu_id}] encountered an error: {e}")
            # Optionally, put the job back in the queue or log it for retrial
            t.cuda.empty_cache()  # Ensure cache is cleared even on error
            break  # Exit worker on error to prevent loops, or implement more robust error handling


class ParallelGpuExecutor:
    """
    Manages a pool of GPU workers to execute a list of jobs in parallel.
    """

    def __init__(self):
        self.ngpus = t.cuda.device_count()
        if self.ngpus == 0:
            raise RuntimeError("No CUDA GPUs detected.")
        print(f"Found {self.ngpus} GPUs.")
        # Using 'spawn' is generally safer for CUDA with multiprocessing
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
