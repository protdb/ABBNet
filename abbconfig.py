from worker_framework.configuration import DefaultConfig, StrEnvParam, BoolEnvParam


class ABBNetConfig(DefaultConfig):
    rmq_upload_uri_string = StrEnvParam("amqp://uploader:Uploader@localhost:5672/data_uploads")
    rmq_upload_exchange = StrEnvParam("uploads")
    local_pdb_database = StrEnvParam("/pdb")
    local_cache_dir = StrEnvParam("/ramdisk")
    create_db_on_startup = BoolEnvParam(False)