from pathlib import Path

import pandas as pd


def download_data(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_id = '1AP_wjJU649-gTGtaMxrp8GEx-Ru3fhuH'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    df = pd.read_csv(url)
    df.to_csv(path, index=False)


if __name__ == '__main__':
    download_data(Path('artifacts/raw/dataset.csv'))
