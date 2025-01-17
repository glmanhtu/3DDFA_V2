import os

import yaml

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_abs_path(*path):
    return os.path.join(root_path, *path)


class LandmarkDetectorConfig:

    def __init__(self):
        self.config = None

    def load_config_file(self, file_path):
        self.config = yaml.load(open(file_path), Loader=yaml.SafeLoader)

    def update_abs_path(self, key):
        current_path = self.config[key]
        self.config[key] = get_abs_path(*current_path.split("/"))

    @staticmethod
    def default_config(name="mb1_120x120.yml"):
        config_file = get_abs_path("configs", name)
        config = LandmarkDetectorConfig()
        config.load_config_file(config_file)
        config.update_abs_path("checkpoint_fp")
        config.update_abs_path("bfm_fp")
        return config
