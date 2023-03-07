import os

from worker_framework import Runner, TaskTransfer
from worker_framework.tools.workdir import CreateDirMode
from worker_framework.tools.fileprocessor import FileProcessor
from worker_framework.message_managers import HttpPollMsgManager

from result_callback import get_result_callback
from abbconfig import ABBNetConfig
from get_chain_id import get_chain_id

from fast_search.search_engine import FastSearch


class ABBNetRunner(Runner):
    cfg = ABBNetConfig()
    create_dir_mode = CreateDirMode.not_required
    auto_store_file = False
    msg_manager_class = HttpPollMsgManager

    def handler(self, task: TaskTransfer) -> TaskTransfer:
        print(f"handling {task.id}")
        file_name = f"{task.apfid}_{task.id}.pdb"
        FileProcessor(
            task.url,
            self.cfg.local_cache_dir,
            target_filename=file_name,
            preprocess=False
        ).store_file()
        file_path = os.path.join(self.cfg.local_cache_dir, file_name)
        if task.search_parameters.e_value is None or task.search_parameters.e_value <= 0:
            e_value = 'auto'
        else:
            e_value = task.search_parameters.e_value
        if not task.search_parameters.chain:
            chain = get_chain_id(file_path)
        else:
            chain = task.search_parameters.chain
        search_engine = FastSearch(
            task_id=task.id,
            pdb_file=file_path,
            chain=chain,
            e_value_trash=e_value,
            page_size=10
        )
        search_engine.run_search(get_result_callback(self.cfg, task))
        return task


if __name__ == "__main__":
    ABBNetRunner().run()
