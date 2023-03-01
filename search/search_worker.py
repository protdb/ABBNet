import os
import time
import uuid

import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import logging
from logger_utils.logger_utils import task_logger_configure, listener_configure, listener_process
from model_processor.model_initialization import load_shared_model
from search.search_task import SearchTask


def search_task(
        shared_model,
        queue,
        log_queue,
        task_id,
        pdb_path,
        chain,
        e_value_trash,
        search_mode,
        page_size,
        hierarchy_max_depth,
        limit):
    task_logger_configure(log_queue)
    logger = logging.getLogger('task_logger')
    try:
        task = SearchTask(task_id=task_id,
                          pdb_file=pdb_path,
                          chain=chain,
                          e_value_trash=e_value_trash,
                          search_mode=search_mode,
                          page_size=page_size,
                          hierarchy_max_depth=hierarchy_max_depth,
                          limit=limit
                          )
        task.set_model(shared_model)
        task.set_output_queue(queue)
        logger.info(msg=f"Task {task_id} started")
        task.search()
        logger.info(msg=f"Task {task_id} successfully completed")
    except Exception as e:
        logger.error(msg=f"Error: {str(e)}")
        raise
    finally:
        pass
        # if task is not None:
        #     task.clear_task_ws()


class SearchWorker(object):
    def __init__(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.processes = {}
        self.logger_queue = mp.Queue(-1)
        listener = mp.Process(target=listener_process,
                              args=(self.logger_queue, listener_configure))
        listener.start()
        task_logger_configure(self.logger_queue)
        self.logger = logging.getLogger()
        self.model = load_shared_model()
        self.output_queue = Queue(maxsize=-1)
        self.logger.info('Worker initialize')

    def create_task(self,
                    pdb_path,
                    chain,
                    e_value_trash='auto',
                    search_mode=0,
                    page_size=50,
                    hierarchy_max_depth=2,
                    limit=None,
                    task_id=None,
                    ):
        if not task_id:
            task_id = str(uuid.uuid4())
        process = mp.Process(target=search_task,
                             args=(self.model,
                                   self.output_queue,
                                   self.logger_queue,
                                   task_id,
                                   pdb_path,
                                   chain,
                                   e_value_trash,
                                   search_mode,
                                   page_size,
                                   hierarchy_max_depth,
                                   limit))
        self.processes.update({task_id: process})

        return task_id

    def start_task(self, task_id):
        if task_id not in self.processes:
            raise IndexError(f'Task {task_id} not found')
        process = self.processes[task_id]
        process.start()
        self.clear_task_buffer()

    def wait(self):
        self.clear_task_buffer()
        for task_id in self.processes:
            process = self.processes[task_id]
            time.sleep(1e-2)
            process.join()

    def terminate_task(self, task_id):
        self.clear_task_buffer()
        if task_id not in self.processes:
            return
        process = self.processes[task_id]
        process.terminate()

    def stop_worker(self):
        self.logger.info("Worker terminated")
        for task_id in list(self.processes.keys()):
            self.terminate_task(task_id)

    def is_alive(self, task_id):
        if task_id not in self.processes:
            return False
        return self.processes[task_id].is_alive()

    def clear_task_buffer(self):
        for task_id in list(self.processes.keys()):
            process = self.processes[task_id]
            if not process.is_alive():
                del self.processes[task_id]

    def read_queue(self):
        q_item = self.output_queue.get()
        return q_item



