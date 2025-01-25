import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import gc


from typing import Dict, List

import utils.model as models
import utils.data as data
from dataloader import DataLoader, read_config

from utils.mass import MassSegmentation
from utils.roi import ROIextraction
from utils.processor import remove_redundant_area, preprocess, inference, postprocess
from utils.malignancy.estimator import Estimator
from utils.process import resize_array, remove_array_padding
from utils.dicom import dicom_manager
from utils.cloud import upload_image

from tqdm import tqdm

from copy import deepcopy

import pickle

def save_mask_dict(mask_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(mask_dict, f)

def load_mask_dict(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
    

def mask2binary(mask: np.ndarray, threshold: float = 0.5) -> None:
    # Apply threshold to create binary mask
    binary_mask = (mask >= threshold).astype(np.uint8)  # True/False -> 1/0 -> uint8
    return binary_mask

def log_debug_list(debug_list, log_file='debug_log.txt'):
    with open(log_file, 'a') as f:
        for debug_item in list(debug_list):
            f.write(f"{debug_item}\n")

if __name__ == "__main__":
    if not os.path.exists("output"):
        os.makedirs("output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/mass.pt"
    roi_path = "models/roi.pt"

    config = read_config("config.json")
    dataloader = DataLoader(config=config, num_samples=1)
    
    # load model
    roi: ROIextraction = models.load_roi(models.load_checkpoint_bytes(roi_path), device)
    model: MassSegmentation = models.load_model(models.load_checkpoint_bytes(model_path), device)
    model_name = "xgboost"
    malignancy_estimator = Estimator(model_name=model_name)
    
    # read data
    data_folders = ["data/malignant", "data/benign"]
    # ddd = {"R-MLO": 0, "L-MLO": 1, "R-CC": 2, "L-CC": 3}

    # images, masks, meta = dataloader(3)
    # print(meta)
    urls = []  # 327
    debug_list = []

    # processed_samples = set()
    # itr = 0
    # num_prev_processed_samples = 0
    # sample_itr = 0

    for i in tqdm(range(1, 100), desc="Processing"):
        itr_processed_samples = {f"{i}-L_MLO", f"{i}-R_MLO", f"{i }-L_CC", f"{i }-R_CC"}
        
        paths = {"R-MLO": f"/mnt/mogram001/pacs/data/P{i:06}-1.dcm", 
                "L-MLO": f"/mnt/mogram001/pacs/data/P{i:06}-2.dcm", 
                "R-CC": f"/mnt/mogram001/pacs/data/P{i :06}-3.dcm", 
                "L-CC": f"/mnt/mogram001/pacs/data/P{i  :06}-4.dcm"}
        views, pixel_arrays_in_use, meta = dicom_manager.load(paths)

        views = {k.replace("-", "_"): v for k, v in views.items()}

        # extract roi
        pixel_arrays, boundaries = remove_redundant_area(roi, views)
        
        # preprocess
        model_inputs = preprocess(pixel_arrays) # LMLO, RMLO, LCC, RCC
        model_outputs = inference(model, model_inputs, views.keys())
        
        # extract features from mass output,

        mask_dict = model_outputs  # ["mass_segment"]

        image_dict = {key: (model_inputs[i].squeeze().permute(1,2,0).numpy()* 255.0).astype(np.uint8) for i, key in enumerate(views.keys())}
        mask_dict = {key: mask2binary(mask_dict[key]) for key in mask_dict.keys()}

        # save_mask_dict(mask_dict=mask_dict, filename="output/saved_mask.pkl")
        # loaded_mask_dict = load_mask_dict(filename="output/saved_mask.pkl")
        
        # print("inference comparison", np.all([np.array_equal(mask_dict[k], loaded_mask_dict[k]) for k in mask_dict.keys()]))
        
        probability_masks, temp_debug_list = malignancy_estimator(i, deepcopy(image_dict), deepcopy(mask_dict))
        log_debug_list(temp_debug_list)
        # debug_list.extend(temp_debug_list)


        # processed_samples.update(itr_processed_samples)
        
        # model_outputs["mass_segment"] = probability_masks
        # model_outputs = probability_masks        
        # original_pixel_arrays, output_pixel_arrays = postprocess(i, views, 
        #                                                         model_outputs, 
        #                                                         boundaries, views.keys())
        # for key in views.keys():
        #     cv2.imwrite(f"output/{i}_{key}.png", output_pixel_arrays[key])
        
        
        # concat_mask = np.concatenate(
        #     [
        #         np.concatenate(
        #             [output_pixel_arrays["R_MLO"], output_pixel_arrays["L_MLO"]],
        #             axis=1,
        #         ),
        #         np.concatenate(
        #             [output_pixel_arrays["R_CC"], output_pixel_arrays["L_CC"]],
        #             axis=1,
        #         ),
        #     ],
        #     axis=0,
        # )
        # url = upload_image(f"P{i :06}.jpeg", concat_mask)
        # urls.append(url)
        # if url is None:
        #     print("none geldi")

        # except Exception as e:
        #     print(f"Error at index {i}: {e}")
    #     sample_itr +=1
    # processed_samples = processed_samples.union(itr_processed_samples)
    # if num_prev_processed_samples >= len(processed_samples):
    #     print("ITR: ", itr)
    #     print("num prev: ", num_prev_processed_samples)
    #     print("processed samples: ", len(processed_samples))
    #     print("FINISH")
    #     break
    # num_prev_processed_samples = len(processed_samples)
    # log_debug_list(debug_list)
    # itr += i


        pixel_arrays.clear()
        boundaries.clear()
        model_inputs = None
        model_outputs.clear()
        image_dict.clear()
        mask_dict.clear()
        del pixel_arrays, boundaries, model_inputs, model_outputs, image_dict, mask_dict
        gc.collect()

            
    # import json   
    # with open("urls.json", "w+") as f:
    #     f.write(json.dumps(urls, indent=4))
        
        