import glob
import os.path


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


def get_files(dir_name, ext='*.*'):
    assert os.path.exists(dir_name)
    files = glob.glob(f'{dir_name}/{ext}')
    return files
