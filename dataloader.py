import json
import os
import cv2
import numpy as np
from tqdm import tqdm

from google.cloud import storage
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
import utils.data as data

from typing import List, Tuple, Union, Dict, Any
from random import sample

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/maraldiclemaral/Documents/vitamu/repositories/mogram-storage-viewer.json'


def read_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)


class DataLoader:
    def __init__(self, config: dict, num_samples: Union[int, Tuple[int, int]] = None):
        self.config = config
        self.num_samples = num_samples
        
        self.sample_count: int = 0
        self.malignant_count: int = 0
        self.benign_count: int = 0
        
        self.client: storage.Client = storage.Client()
        
        self.buckets = {e["bucket_name"]: self.client.get_bucket(e["bucket_name"]) for e in self.config["buckets"]}
        self.meta_blobs = {e["bucket_name"]: e["meta"] for e in self.config["buckets"]}
        self.set_blobs = {e["bucket_name"]: e["set"] for e in self.config["buckets"]}
        self.label_blobs = {e["bucket_name"]: e["labels"] for e in self.config["buckets"]}
    
        # will take a lot of time if we were to work with multiple buckets so updating to 1.
        self.labels = self.process_labels(self.fetch_labels(
            self.config["buckets"][0]["bucket_name"]
            ))
    
    def fetch_metadata(self, bucket_name: str) -> Dict[str, Any]:
        """ Fetch metadata from bucket """
        bucket = self.buckets[bucket_name]
        meta_blob = bucket.blob(self.meta_blobs[bucket_name])
        return json.loads(meta_blob.download_as_string())
    
    def fetch_set(self, bucket_name: str, set_type: str) -> List[str]:
        """ Fetch labels from bucket """
        bucket = self.buckets[bucket_name]
        set_blob = bucket.blob(self.set_blobs[bucket_name])
        return json.loads(set_blob.download_as_string())[set_type]
    
    def fetch_labels(self, bucket_name: str) -> List[Dict[str, Any]]:
        """ Fetch labels from bucket """
        bucket = self.buckets[bucket_name]
        labels_blob = bucket.blob(self.label_blobs[bucket_name])
        return json.loads(labels_blob.download_as_string())
    

    def process_labels(self, labels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ Process labels to return a dict """
        labels_dict = {}
        for label in labels:
            labels_dict.update({label["externalId"]: label["tasks"][0]})
        return labels_dict
    
    def extract_coordinates(self, external_id):  # -> List[Tuple[int, int]]
        coordinates = []
        birads = []
        objects = self.labels[external_id]["objects"]
        for obj in objects:

            if "segmentation" in obj.keys():

                for zone, classificatinos in zip(obj["segmentation"]["zones"], obj['classifications']):
                    birads.append(max(classificatinos['answer']))
                    coordinates.append(zone["region"])
                


        return coordinates, birads

    def draw_polygon_from_coordinates(self, regions, image_shape):
        masks = []
        for region in regions:
            region = [r[::-1] for r in region]
            mask = polygon2mask(image_shape, np.array(region))
            masks.append(mask)
        merged_mask = np.logical_or.reduce(masks)
        # merged_mask = np.stack([merged_mask] * 3, axis=-1)

        return merged_mask, len(masks)

    def expand_bounding_box(self, box, k, max_dims):
        x, y, w, h = box
        x_new = max(0, x - k)
        y_new = max(0, y - k)
        w_new = min(max_dims[1], x + w + k) - x_new
        h_new = min(max_dims[0], y + h + k ) - y_new
        return (x_new, y_new, w_new, h_new)

    def extract_rois(self, mask, k=20):
        # Check if mask is boolean and convert to uint8 binary image
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        
        # Ensure the mask is single-channelc 
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            # Assuming all channels are the same, we take one
            mask = mask[:, :, 0]
        
        # Convert to binary format required by findContours
        ret, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Image dimensions
        max_dims = mask.shape[:2]

        # List of bounding boxes
        bounding_boxes = []

        # Process each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            expanded_box = self.expand_bounding_box((x, y, w, h), k, max_dims)

            bounding_boxes.append(expanded_box)

        return bounding_boxes, contours

    def load_image(self, bucket: storage.Bucket, path: str) -> np.ndarray:
        blob = bucket.blob(path)
        img = cv2.imdecode(np.frombuffer(blob.download_as_string(), np.uint8), cv2.IMREAD_COLOR)

        return img
    
    def check_mass_type(self, values):

        for value in values:
            if int(value) in [0, 3, 4, 5, 6]:
                return "malignant"
        return "benign"
                



    def __call__(self, 
                 num_samples: Union[int, Tuple[int, int]]) -> Tuple[List[np.ndarray], 
                                                                    List[np.ndarray],
                                                                    List[Dict[str, Any]]]:
        """
        num_samples: int or Tuple[int, int] -> if tuple: malignant and benign samples are set
        returns: (list of images, masks, metadata)
        """
        self.num_samples = num_samples

        images, masks, metadata, new_meta, all_contours, all_rois =  {}, {}, {}, {}, {}, {}
        print("Loading data...")
        
        if isinstance(num_samples, tuple):
            m, b = num_samples
            
        for i, (bucket_name, bucket) in enumerate(self.buckets.items()):

            meta = self.fetch_metadata(bucket_name)

            set_type = self.config["buckets"][i]["set_type"]
            filenames = self.fetch_set(bucket_name, set_type)

            if self.num_samples == 0:
                self.num_samples = len(filenames)

            if isinstance(self.num_samples, int):
                pbar = tqdm(total=num_samples)
            elif isinstance(self.num_samples, tuple) and len(num_samples)== 2:
                pbar = tqdm(total=(num_samples[0] + num_samples[1]))
            else:
                print("Invalid sample size.")
                exit()

            for meta_fn in meta.keys():

                if bucket_name == "mogram-data-1":
                    fn = meta_fn.split("/")[-1]
                    load_fn = fn
                    
                    if not meta_fn in filenames:
                        continue
                else:
                    fn = meta_fn.split("/")[-1]
                    load_fn = fn.split("-")[-1] 

                    if not fn in filenames:
                        continue
                
                mask_regions, birads_scores = self.extract_coordinates(fn)
                if len(mask_regions) == 0:
                    continue

                if isinstance(self.num_samples, int):
                    if self.sample_count >= self.num_samples:
                        break 
                else:
                    if m <= 0 and b <= 0:
                        break

                    mass_type = self.check_mass_type(birads_scores)

                    if mass_type == "malignant":
                        if m > 0:
                            m -= 1
                        else: 
                            continue
                    elif mass_type == "benign":
                        if b > 0:
                            b -= 1
                        else: 
                            continue
                    else: 
                        print("One of the birads scores in the dataset is not valid..")
                        continue



                image = self.load_image(bucket, f"processed/{load_fn}")

                mask, len_masks = self.draw_polygon_from_coordinates(mask_regions, image.shape[:2])

                rois, contours = self.extract_rois(mask)


                if len(rois) != len(birads_scores):
                    # print(f"Error: Mismatch between number of rois and birads scores for {meta_fn}")
                    # print("len birads:  ", len(birads_scores))
                    # print("birads:  ", birads_scores)
                    # print("len rois:    ", len(rois))
                    # print("len contours:    ", len(contours))
                    # print("len regions: ", len(mask_regions))
                    # print("len masks: ", len_masks )
                    continue
                result = {"rois": rois, "contours": [], "birads": birads_scores}

                images[meta_fn]= image
                masks[meta_fn] = mask
                metadata[meta_fn] = meta[meta_fn]
                all_contours[meta_fn] = contours
                all_rois[meta_fn] = rois


                if meta_fn not in new_meta:
                    new_meta[meta_fn] = result
                else:
                    new_meta[meta_fn].update(result)
                self.sample_count += 1

                pbar.update(1)
                pbar.refresh()

        return images, masks, metadata

# def draw_mask_on_image(image, mask, title="Mask"):
#     # Ensure the mask is in single channel and np.uint8 format
#     if mask.dtype != np.uint8:
#         mask = (mask > 0).astype(np.uint8) * 255
#     if mask.ndim == 3:
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     if mask.shape[:2] != image.shape[:2]:
#         mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
#     colored_mask = np.zeros_like(image)
#     colored_mask[mask == 255] = [0, 255, 0]  # Green color
#     alpha = 0.5  # Transparency factor.
#     result= cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
#     cv2.putText(result, title, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10, cv2.LINE_AA)
#     return result



if __name__ == "__main__":
    config = read_config("config.json")
    loader = DataLoader(config)
    images, masks, metadata = loader((1))
    # print(metadata)

    # os.makedirs(output_dir, exist_ok=True)
    
    # for idx, key in enumerate(metadata.keys()):
    #     print(f"Processing image {key}")
    #     combined_image = combine_images(images[key], masks[key], contours[key], rois[key])
        
    #     output_path = os.path.join(output_dir, f"output_{idx}.png")
    #     success = cv2.imwrite(output_path, combined_image)
        
    #     if success:
    #         print(f"Saved image {idx} to {output_path}")
    #     else:
    #         print(f"Failed to save image {idx} to {output_path}")
        
    #     if idx >= 4: 
    #         break