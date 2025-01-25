import cv2
import numpy as np
from radiomics import featureextractor
from SimpleITK import GetImageFromArray
import pandas as pd


import pickle
import logging
# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
# ... or set level for specific class
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

class FeatureExtractor:
    def __init__(self):
        self.extractor = featureextractor.RadiomicsFeatureExtractor()

    def preprocess(self, img, mask):
        # Ensure mask is binary and of type uint8
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) 

        # Resize mask to match image dimensions if necessary
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convert image and mask to SimpleITK images
        img = img[: , : , 0]

        img_itk = GetImageFromArray(img)
        mask_itk = GetImageFromArray(mask)


        # Ensure both have the same dimensions
        if img_itk.GetDimension() != mask_itk.GetDimension():
            raise ValueError(f"Image and mask dimensions do not match: {img_itk.GetDimension()} vs {mask_itk.GetDimension()}")

        return img_itk, mask_itk

    def __call__(self, dataset):
        with open('utils/malignancy/list_of_features.pkl', 'rb') as f:  # TODO: change path here
            listoffeatures_ = pickle.load(f)  # list of strings representing features

        # Filtered list
        listofextractedfeatures = [item for item in listoffeatures_ if item not in ['external_id', 'label']]
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()  # TODO: instead just enable the features we need from pkl

        extracted_dataset = []

        for i, sample_id in enumerate(dataset.keys()):
            sample = dataset[sample_id]
            img = sample['cropped_image']
            mask = sample['cropped_mask']

            img, mask = self.preprocess(img, mask)

            try:
                # Execute feature extraction
                extracted_features = extractor.execute(img, mask)

                temp = {key: value for key, value in extracted_features.items() if "diagnostics" not in key}
                temp["sample_id"] = sample_id

                extracted_dataset.append(temp)
    

            except ValueError as e:
                # Skip the sample that caused the error
                print(f"Skipping sample {sample_id} due to error: {e}")

        df = pd.DataFrame.from_dict(extracted_dataset)
        listofextractedfeatures.append("sample_id")
        # print("features w/o filtering", df.columns )
        # print("features w/o filtering", len(df.columns))
        # print("features", df[listoffeatures_] )

        return df[listofextractedfeatures]