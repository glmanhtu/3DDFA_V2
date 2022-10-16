import os

import yaml

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class LandmarkDetectorConfig:

    def __init__(self):
        self.config = None

    def load_config_file(self, file_path):
        self.config = yaml.load(open(file_path), Loader=yaml.SafeLoader)

    @staticmethod
    def default_config(name="mb1_120x120.yml"):
        config_file = os.path.join(root_path, "config", name)
        config = LandmarkDetectorConfig()
        config.load_config_file(config_file)
        return config
