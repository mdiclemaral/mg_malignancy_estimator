import os
import numpy as np
import pickle 
import cv2

import os
import os.path

from .data_preparer import DataPreparer  
from .feature_extractor import FeatureExtractor
from .xg_classifier import XG

from .mlp import MLP
import joblib
class Estimator:
    def __init__(self, model_name="xgboost") -> None:
        self.feature_extractor = FeatureExtractor()

        path = os.path.join(os.getcwd(), "models/")

        if model_name == "mlp":
            self.model = MLP()
            model_path = path + "mlp_model.pkl"
        elif model_name == "xgboost":
            self.model = XG()
            model_path = path + "xgboost_model.joblib"
            # model_path = "/home/maral/mass_malignancy/models/xgboost_model.pkl"
        
        # with open(model_path, 'rb') as f:
        #     self.weights = pickle.load(f)
        self.weights = joblib.load(model_path)

    def __call__(self, i, image_dict, mask_dict):
        debug_list= []
        data_preparer = DataPreparer()
        df = data_preparer(image_dict, mask_dict)

        if len(df.keys()) == 0:  # 0 mass detected
            return mask_dict, debug_list
        
        idd = i
        features = self.feature_extractor(df)
        results, y_proba = self.model.test_model(idd, self.weights, features)
        mass_lists = {"L_MLO": [], "R_MLO":[], "L_CC":[], "R_CC":[]}
        main_key_print = list(results.keys())[0].split("-")[0]
        # print( f"{i}_{main_key_print}:  {y_proba}")
        debug_list.append(f"{i}_{main_key_print}:  {y_proba}")
        for key in results.keys():
            
            main_key = key.split("-")[0]
            # print(key)
            # print(main_key, results[key])
            mass_lists[main_key].append({"probability": results[key][1], "contour": df[key]["contour"]})
            
        for key in mask_dict.keys():
            mask = mask_dict[key]
            for result in mass_lists[key]:
                contour = result["contour"]
                probability = result["probability"]
                mask = cv2.fillPoly(mask.astype(np.float32), [contour], probability)
                
            mask_dict[key] = mask
                
        return mask_dict, debug_list

        