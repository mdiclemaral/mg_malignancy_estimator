from google.cloud import storage
import os


def upload_images(bucket_name: str, folders: list):
    """
    folders: list of str containing paths to folders
    """
    client: storage.Client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    for folder in folders:
        bucket_folder = folder.split("/")[-1]
        for fn in os.listdir(folder):
            blob = bucket.blob(f"{bucket_folder}/{fn}")
            blob.upload_from_filename(os.path.join(folder, fn))
            
            
            
if __name__ == "__main__":
    upload_images("mogram-dev", ["data/malignant", "data/benign"])
