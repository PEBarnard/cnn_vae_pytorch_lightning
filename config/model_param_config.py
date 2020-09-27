import yaml
from os.path import join


def model_config(dir_name, file):
    path = join(dir_name, file)
    with open(path, 'rt') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
