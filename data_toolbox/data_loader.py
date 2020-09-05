from zipfile import ZipFile
from tqdm import tqdm
import requests
import pickle
import os


class DataLoader:
    @property
    def data_dir(self):
        base_dir = os.getenv('APPDATA') or os.getenv('HOME') or '.'
        data_dir = os.path.join(base_dir, '.cache', 'data-toolbox')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def download_file(self, url, name):
        file_path = os.path.join(self.data_dir, name)
        if os.path.exists(file_path):
            return file_path
        with open(file_path, 'wb') as f:
            print(f'[+] Downloading {url}...')
            with requests.get(url, stream=True) as r:
                total = int(r.headers.get('content-length', 0))
                pb = tqdm(total=total, unit='iB', unit_scale=True)
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pb.update(len(chunk))
                pb.close()
        return file_path

    def open(self, name): raise Exception("NotImplemented")


class ZippedDataLoader(DataLoader):

    def __init__(self, file_name, zip_url):
        self.zip_path = self.download_file(zip_url, file_name)
        self.zip_ref = ZipFile(self.zip_path)

    def open(self, asset_name):
        return self.zip_ref.open(asset_name, mode='r')


class PickledDataLoader(DataLoader):

    def __init__(self, file_name):
        self.pkl_path = os.path.join(
            self.data_dir,
            file_name
        )
        if not os.path.exists(self.pkl_path):
            with open(self.pkl_path, 'wb') as f:
                pickle.dump({}, f)

    def open(self, asset_name):
        with open(self.pkl_path, 'rb') as f:
            return pickle.load(f).get(asset_name)

    def save(self, asset_name, value):
        with open(self.pkl_path, 'rb') as f:
            d = pickle.load(f)
            d[asset_name] = value
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(d, f)
