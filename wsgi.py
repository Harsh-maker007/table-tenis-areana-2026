import os

from src.config import load_config
from src.web import create_app


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


VIDEO_PATH = os.getenv("VIDEO_PATH", "")
CAMERA_INDEX = _env_int("CAMERA_INDEX", 0)

config = load_config()
app = create_app(VIDEO_PATH, CAMERA_INDEX, config)
