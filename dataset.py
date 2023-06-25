import os
import wget
import zipfile

class Downloader:
    def __init__(self):
        pass
    
    def download_celeb_a(root):
        # download celebA dataset
        data_root = root
        base_url = "https://graal.ift.ulaval.ca/public/celeba/"

        if not os.path.exists(data_root):
            os.mkdir(data_root)
        
        file = "img_align_celeba.zip"

        dataset_folder = data_root
        os.makedirs(dataset_folder, exist_ok=True)

        url = f"{base_url}/{file}"
        if not os.path.exists(os.path.join(dataset_folder, file)):
            print(f"Downloading {file}...")
            wget.download(url, out=os.path.join(dataset_folder, file))
        if os.path.exists(os.path.join(dataset_folder, "img_align_celeba")):
            print("Dataset already downloaded, start training...")
            return
        with zipfile.ZipFile(os.path.join(dataset_folder, file), 'r') as ziphandler:
            print(f"Extracting {file}...")
            ziphandler.extractall(dataset_folder)
            print("Done!")
            print("Start training...")

        
    


    pass
