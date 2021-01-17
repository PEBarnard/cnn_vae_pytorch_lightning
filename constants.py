from os.path import realpath, dirname, join
import torch

BASE_DIR = join(dirname(realpath(__file__)))
CONFIG_DIR = join(BASE_DIR, 'config')
RESULTS_DIR = join(BASE_DIR, 'results')
ARTEFACT_DIR = join(BASE_DIR, 'artefact')

USING_GPU = torch.cuda.is_available()