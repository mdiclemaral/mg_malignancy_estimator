import cv2

class DataPreparer:
    """
    Description:
    Prepares patchwise dataset from the given dataset
    
    """  
    def expand_bounding_box(self, box, k, max_dims):
        x, y, w, h = box
        x_new = max(0, x)
        y_new = max(0, y)
        w_new = min(max_dims[1], x + w) - x_new
        h_new = min(max_dims[0], y + h) - y_new
        return (x_new, y_new, w_new, h_new)
        
    def extract_rois(self, binary_mask, k=20):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        return contours, bounding_boxes
    
    def crop_image_and_mask(self, image, mask, roi):
        x, y, w, h = roi
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]

        return cropped_image, cropped_mask
    
    def process_dataset(self, image_dict, mask_dict):
        dataset = {}
        for sample_id in image_dict.keys():
            contour_list, roi_list = self.extract_rois(mask_dict[sample_id])
            sample_image = image_dict[sample_id]
            sample_mask = mask_dict[sample_id]
            try:
                for i, roi in enumerate(roi_list):
                    # Crop the image and mask based on the ROI
                    cropped_image, cropped_mask = self.crop_image_and_mask(sample_image, sample_mask, roi)
                    # Append the data to the dataset
                    dataset[f'{sample_id}-{i}'] = {
                        'cropped_image': cropped_image,
                        'cropped_mask': cropped_mask,
                        'roi': roi,
                        'contour': contour_list[i]
                    }
            except:
                dataset[f"{sample_id}"] = None
                print(f"Error: An error occurred while processing sample {sample_id}")

        return dataset


    def __call__(self, image_dict, mask_dict):
        try:
            dataset = self.process_dataset(image_dict, mask_dict)
            return dataset

        except Exception as e:
            print(f"Error: An error occurred during data preparation: {e}")

if __name__ == "__main__":
    data_preparer = DataPreparer()
    dataset = data_preparer()