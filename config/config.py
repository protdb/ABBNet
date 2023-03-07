import os.path
import shutil
from pathlib import Path
from utils.utils import singleton


@singleton
class BaseConfig(object):
    pdb_dir = '/pdb'
    working_dir = '/data'
    samples_dir = 'samples'
    tmp_dir = '_tmp'
    train_dir = 'train'
    val_dir = 'val'
    model_dir = 'model'
    model_name = 'abbnet'
    logging = True
    n_sample_workers = 4
    min_chain_len = 16
    max_chain_len = 2048
    batch_size = 8
    num_workers = 8
    train_epochs = 16
    eval_models_every = 2
    log_result = 'file'
    log_dir = 'log'
    metric_log_file = 'metric.log'
    worker_log_file = 'samples.log'
    debug_dir = 'debug'
    samples_size = 50
    augmentation = {'rotate': True, 'translate': True}
    augmentation_factor = 0.4
    train_file = '/home/dp/Downloads/download_pdb_seq (2).txt'
    stride_folder = "/data/stride"
    finetune_samples_file = 'finetune_samples.pkl'
    finetune_split_factor = 0.7
    finetune_set_names = ['train_set_idx', 'test_set_idx']

    def get_pdb_dir(self):
        assert os.path.exists(str(self.pdb_dir))
        return Path(self.pdb_dir)

    def get_sample_size(self):
        return self.samples_size

    def make_working_dir(self):
        working_dir = Path(self.working_dir)
        working_dir.mkdir(exist_ok=True)
        samples_dir = working_dir / self.samples_dir
        samples_dir.mkdir(exist_ok=True)
        train_dir = samples_dir / self.train_dir
        val_dir = samples_dir / self.val_dir
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        model_dir = working_dir / self.model_dir
        model_dir.mkdir(exist_ok=True)
        log_dir = working_dir / self.log_dir
        log_dir.mkdir(exist_ok=True)
        tmp_dir = samples_dir / self.tmp_dir
        if os.path.exists(tmp_dir):
            shutil.rmtree(str(tmp_dir))
        tmp_dir.mkdir(exist_ok=True)

    def get_samples_dir(self):
        working_dir = Path(self.working_dir)
        samples_dir = working_dir / self.samples_dir
        train_dir = samples_dir / self.train_dir
        val_dir = samples_dir / self.val_dir
        tmp_dir = samples_dir / self.tmp_dir
        return train_dir, val_dir, tmp_dir

    def get_model_path(self):
        working_dir = Path(self.working_dir)
        model_dir = working_dir / self.model_dir
        model_path = model_dir / self.model_name
        return model_path

    def get_train_file_path(self):
        return self.train_file

    def get_metric_log_file(self):
        working_dir = Path(self.working_dir)
        log_dir = working_dir / self.log_dir
        log_path = log_dir / self.metric_log_file
        return log_path

    def get_debug_dir(self):
        features_dir = Path(self.working_dir)
        debug_dir = features_dir / self.debug_dir
        debug_dir.mkdir(exist_ok=True)
        return debug_dir

    def get_log_dir(self):
        working_dir = Path(self.working_dir)
        log_dir = working_dir / self.log_dir
        log_dir.mkdir(exist_ok=True)
        return log_dir


@singleton
class RunnerConfig(object):
    run_mode = 'train'


@singleton
class ModelParams(object):
    alphabet_size = 21
    sequence_emb_dim = 21
    num_rbf = 16
    neighbours_agg = 32
    num_positional_embeddings = 16
    dihedral_embed_dim = 16
    rbf_cutoff_lower = 0.0
    rbf_cutoff_upper = 24.0
    hidden_emb_nodes = 64
    n_gvp_encoder_layers = 3
    drop_rate = 0.3
    rnn_dim = 32
    negative_from_anchor_factor = 0.3
    pad = 2048
    debug = False


class DBConfig(object):
    def __init__(self):
        config = BaseConfig()
        self.saml_db_folder = 'saml_db'
        self.saml_db_file = 'saml_db.data'
        self.blast_db_folder = 'saml_blast'
        self.blast_db_name = 'abbnet_blast'
        self.preprocessed_dir = 'preprocessed_pdb'
        self.e_value_trash = 0.001
        self.min_saml_thresh_ = 16
        self.pdb_dir = config.get_pdb_dir()
        config.make_working_dir()
        working_dir = Path(config.working_dir)
        self.preprocessed_dir = working_dir / self.preprocessed_dir
        self.preprocessed_dir.mkdir(exist_ok=True)
        saml_db_folder = working_dir / self.saml_db_folder
        saml_db_folder.mkdir(exist_ok=True)
        self.blast_db_folder = saml_db_folder / self.blast_db_folder
        self.blast_db_folder.mkdir(exist_ok=True)
        self.saml_db_file_path = saml_db_folder / self.saml_db_file
        _, _, self.tmp_dir = config.get_samples_dir()
        log_dir =  config.get_log_dir()
        self.logging_file = log_dir / 'db_log.log'

    def get_db_path(self):
        return self.saml_db_file_path

    def get_fa_path(self):
        return self.blast_db_folder / f'{str(self.saml_db_file_path.stem)}.fa'

    def get_blast_folder(self):
        return self.blast_db_folder

    def get_blast_db_name(self):
        return self.blast_db_folder / self.blast_db_name


class VisConfig(object):
    def __init__(self):
        config = BaseConfig()
        self.vis_folder = 'vis_data'
        self.blast_results_file = 'blast_out'
        config.make_working_dir()
        working_dir = Path(config.working_dir)
        self.vis_folder = working_dir / self.vis_folder
        self.vis_folder.mkdir(exist_ok=True)
        self.blast_results_file = self.vis_folder / self.blast_results_file

    def get_blast_out_path(self):
        return self.blast_results_file


class SearchConfig(object):
    def __init__(self):
        self.base_config = BaseConfig()
        self.base_config.make_working_dir()
        self.upload_result = False
        self.upload_source = False
        self.upload_extract = False
        self.upload_dir = 'search_results'
        self.copy_extracted_folder = 'extracted'
        self.hierarchy_max_depth = 2
        self.e_scale_factor = {1: 1.0, 0.9: 0.1, 0.8: 0.0001}
        working_dir = Path(self.base_config.working_dir)
        self.upload_dir = working_dir / self.upload_dir
        self.upload_dir.mkdir(exist_ok=True)
        self.copy_extracted_folder = None if not self.upload_extract else self.copy_extracted_folder

    def get_pdb_dir(self):
        return self.base_config.get_pdb_dir()

    def get_upload_dir(self):
        return self.upload_dir

    def get_extracted_copy_folder(self):
        return self.copy_extracted_folder


class LogConfig(object):
    def __init__(self):
        self.log_format = "%(levelname)s  %(asctime)s  %(message)s"
        self.base_config = BaseConfig()
        self.logging_dir = Path(self.base_config.get_log_dir())
        self.log_file = 'search.log'
        self.max_bytes = 1024
        self.backupCount = 5

    def get_loging_file(self):
        if os.path.exists(self.logging_dir) and self.log_file:
            return self.logging_dir / self.log_file
        return None


