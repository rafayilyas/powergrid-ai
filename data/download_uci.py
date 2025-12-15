import requests
import zipfile
from pathlib import Path
import io

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
OUT = Path('data/raw/household_power_consumption.txt')


def download_and_extract(url: str = URL, out: Path = OUT):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print('Dataset already present at', out)
        return out
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    # list of files; look for the household file
    for name in z.namelist():
        if 'household_power_consumption.txt' in name:
            z.extract(name, out.parent)
            # move to expected name
            extracted = out.parent / name
            extracted.rename(out)
            print('Extracted to', out)
            return out
    raise RuntimeError('Expected file not found in zip')


if __name__ == '__main__':
    print('Downloading UCI dataset...')
    print(download_and_extract())
