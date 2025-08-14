#!/usr/bin/env python3
import yaml
import subprocess
from pathlib import Path

CONFIG_PATH = Path(__file__).parents[1] / "config" / "settings.yaml"

def load_settings(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    settings = load_settings(CONFIG_PATH)
    dataset_id = settings['dataset']['id']
    data_root = Path(settings['data_root']).expanduser()
    raw_dir = data_root / "raw" / settings['dataset']['name']

    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Downloading {settings['dataset']['name']} ({dataset_id}) to {raw_dir} ...\n")
    try:
        subprocess.run(
            ["openneuro-py", "download", "--dataset", dataset_id, "--target-dir", str(raw_dir.as_posix())],
            check=True
        )
        print("\n✅ Download complete.")
    except subprocess.CalledProcessError as e:
        print("\n❌ Download failed. Make sure `openneuro-py` is installed and you have internet access.")
        raise e

if __name__ == "__main__":
    main()
