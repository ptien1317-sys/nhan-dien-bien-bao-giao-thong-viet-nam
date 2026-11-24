"""Doc configs/its_runtime.yaml (dung chung voi nhanh C++)."""
from pathlib import Path
import yaml

DEFAULT_PATH = Path("configs/its_runtime.yaml")


def load_runtime_config(path=DEFAULT_PATH) -> dict:
    """Tra ve dict long nhau. Nem FileNotFoundError neu thieu file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay file cau hinh: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
