import os
from typing import Any

import yaml


def load_config() -> Any:
    path = os.getenv('CONFIG_PATH', 'config.yml')
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # Baked images set WEIGHTS_DIR to the in-image weight root; mounted deployments
    # leave it unset and fall back to the ~/.cache path in config.yml. Overriding by
    # env keeps one config.yml working for both.
    weights_dir = os.getenv('WEIGHTS_DIR')
    if weights_dir:
        config.setdefault('storage', {})['weights_dir'] = weights_dir

    # storage paths may be absolute, ~-relative (weight caches live under the
    # user's home), or relative to config.yml
    filedir = os.path.dirname(os.path.abspath(path))
    for key, value in config.get('storage', {}).items():
        value = os.path.expanduser(value)
        if not os.path.isabs(value):
            value = os.path.join(filedir, value)
        config['storage'][key] = value
    return config


config = load_config()
