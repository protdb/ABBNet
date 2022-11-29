import os.path
from pathlib import Path

from model_processor.data_processor import ModelProcessor
from utils.utils import get_files

TEST_DIR = Path('/home/dp/Data/SAML/test')
sample_1 = TEST_DIR / 'pair1'
sample_2 = TEST_DIR / 'pair2'
sample_3 = TEST_DIR / 'pair3'
sample_4 = TEST_DIR / 'pdb'


def load_samples(sample):
    assert os.path.exists(sample)
    files = get_files(sample, ext='*.pdb')
    return files


def process_pair(pair_files):
    chain = 'A'
    for file in pair_files:
        result = processor.process_folder(file, chain)
        print(f"file: {result['file']}")
        print(f"sequence: {result['sequence']}")
        print(f"SAML: {result['alphabet']}")
        print()


def process_dir(sample_dir):
    chain = 'A'
    files = load_samples(sample_dir)
    for file in files:
        processor.process_folder(file, chain)


if __name__ == '__main__':
    processor = ModelProcessor()
    samples = load_samples(sample_3)
    process_pair(samples)

