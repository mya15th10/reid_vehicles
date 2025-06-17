import os
import zipfile
import requests
from tqdm import tqdm
os.environ['KAGGLE_USERNAME'] = 'mya15th10'
os.environ['KAGGLE_KEY'] = '6424c2e0d5e3d4301049dd521ad2f3eb'
from kaggle.api.kaggle_api_extended import KaggleApi


def download_file(url, destination):
    """
    Download files in small parts to avoid memory errors
    """

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() #Check http error

        #Get the total size of file
        total_size = int(response.headers.get('content-length', 0))
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,

        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                if chunk:  # Filter the connection parts
                    size = file.write(chunk)
                    bar.update(size)
        return True
    except Exception as e:
        print(f"Lỗi khi tải file: {e}")
        return False
def download_veri_dataset(target_dir="./raw/VeRi"):
    """
    Download dataset Veri-776 from Kaggle
    
    Args:
        target_dir: Folder to save dataset
    """

    #Create folder 
    os.makedirs(target_dir,exist_ok=True)


    #Link to file zip
    zip_path = os.path.join(target_dir, "veri-dataset.zip")

    #initize kaggle api 
    api = KaggleApi()
    api.authenticate()

    print("Loading dataset VeRi-776 from Kaggle")

    #Download dataset
    api.dataset_download_files(
        "abhyudaya12/veri-vehicle-re-identification-dataset",
        path = target_dir,
        unzip=True, #direct extract
        quiet=False
    )

    print(f"Loaded dataset in {target_dir}")

    #Check structure of dataset
    check_dataset_structure(target_dir)


def check_dataset_structure(dataset_dir):
    """
    check structure of VeRi-776 dataset
    """

    expected_dirs = ['image_train', 'image_query', 'image_test']
    expected_files = ['name_train.txt', 'name_query.txt', 'name_test.txt', 'vehicle_info.txt']

    #Cjeck folder
    for dir_name in expected_dirs:
        dir_path = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(dir_path):
            image_count = len([f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')])
            print(f"{dir_name}: {image_count}")
        else:
            print(f"Not find folder {dir_name}")
    
    #Check file
    for file_name in expected_files:
        file_path = os.path.join(dataset_dir, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"{file_name}: {line_count}")
        else:
            print("Not find file {file_name}")

    
if __name__ == "__main__":
    download_veri_dataset()