import os
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from pydicom import dcmread, dcmwrite
from pydicom.dataset import FileDataset
from pydicom.uid import UID


# Parameters
UID_PREFIX = os.getenv("UID_PREFIX", "1.2.840.532.233")
UID_SUFFIX = os.getenv("UID_SUFFIX", "0.0")
SEGMENTA_VERSION_NAME = os.getenv("SEGMENTA_VERSION_NAME", "Seg 0.1.0")

ASSET_DIR = Path(__file__).parents[1] / "assets"
DATA_DIR = Path(__file__).parents[2] / "data"


# Others
LOG_COLOR = "cyan"


# DICOM Manager
class DicomManager:
    """
    Manager responsible for handling DICOM files and metadata.
    Default attributes for generating Secondary Capture images are defined.
    """

    defaults = {
        "SOPClassUID": "ReferencedSOPClassUID",
        "SOPInstanceUID": "ReferencedSOPInstanceUID",
        "PatientName": None,
        "PatientID": None,
        "PatientBirthDate": None,
        "PatientAge": None,
        "PatientSex": None,
        "StudyDate": None,
        "StudyTime": None,
        "AccessionNumber": None,
        "ReferringPhysicianName": None,
        "StudyID": None,
        "StudyInstanceUID": None,
        "StudyDescription": None,
        "Modality": None,
        "Manufacturer": None,
        "InstitutionName": None,
        "ManufacturerModelName": None,
        "DeviceSerialNumber": None,
        "SoftwareVersions": None,
    }

    def __init__(self) -> None:
        """
        Initialize the DICOM manager.
        """
        pass

    @property
    def keys(self) -> list[str]:
        """
        MG view keys.

        Returns
        -------
        keys : list[str]
            The keys for the MG views. (["R-MLO", "L-MLO", "R-CC", "L-CC"])
        """
        return ["R-MLO", "L-MLO", "R-CC", "L-CC"]

    def initialize(self) -> None:
        """
        Initialize the DICOM manager with default pixel arrays.
        These pixel arrays are used when the DICOM files are not available.
        """
        with ZipFile(str(ASSET_DIR / "default.zip"), "r") as file:
            file.extractall(DATA_DIR)

        self.pixel_arrays: dict[str, np.ndarray] = {}

        paths = ["R-MLO.dcm", "L-MLO.dcm", "R-CC.dcm", "L-CC.dcm"]

        for path in paths:
            dataset = dcmread(str(DATA_DIR / path), force=True)
            view = path.split(".")[0]
            self.pixel_arrays[view] = self.get_pixel_array(dataset)

    def load(self, study: dict[str, str]) -> tuple[
        dict[str, np.ndarray],
        dict[str, bool],
        dict[str, dict[str, str | UID | int | None]],
    ]:
        """
        Load a study from the given paths.

        Parameters
        ----------
        study : dict[str, str]
            The study containing the paths for the DICOM files.

        Returns
        -------
        pixel_arrays : dict[str, np.ndarray]
            The pixel arrays for the study.

        pixel_arrays_in_use : dict[str, bool]
            The pixel arrays in use.

        meta : dict[str, dict[str, str | UID | int | None]]
            The metadata for the study.
        """
        pixel_arrays = {}
        pixel_arrays_in_use = {key: False for key in self.keys}
        meta = {}

        for key in self.keys:
            if key not in study:
                pixel_arrays[key] = self.pixel_arrays[key].copy()
                continue

            path = study[key]
            dataset = self.load_dataset(path)
            if dataset is None:
                pixel_arrays[key] = self.pixel_arrays[key].copy()
                continue

            pixel_array = self.get_pixel_array(dataset)
            if pixel_array is None:
                pixel_arrays[key] = self.pixel_arrays[key].copy()
                continue
            pixel_arrays[key] = pixel_array
            pixel_arrays_in_use[key] = True

            meta[key] = self.get_meta(dataset)

        return pixel_arrays, pixel_arrays_in_use, meta

    def load_dataset(self, path: str) -> FileDataset | None:
        """
        Load dataset from the given path.
        After loading the dataset, the file is removed.

        Parameters
        ----------
        path : str
            The path to the DICOM file.

        Returns
        -------
        dataset : FileDataset | None
            The loaded dataset.
        """
        try:
            dataset = dcmread(path, force=True)
        except BaseException as e:
            dataset = None
        finally:
            pass
        return dataset

    def get_pixel_array(self, dataset: FileDataset | None) -> np.ndarray | None:
        """
        Get the pixel array from the dataset.

        Parameters
        ----------
        dataset : FileDataset | None
            The dataset to get the pixel array from.

        Returns
        -------
        pixel_array : np.ndarray | None
            The pixel array from the dataset.
        """
        if dataset is None:
            return None

        pixel_array = deepcopy(dataset.pixel_array).astype(np.float32)

        try:
            window_center = float(dataset.WindowCenter)
        except:
            window_center = float(dataset.WindowCenter[0])

        try:
            window_width = float(dataset.WindowWidth)
        except BaseException as e:
            window_width = float(dataset.WindowWidth[0])

        pixel_array = np.clip(
            (pixel_array - window_center + ((window_width - 1) / 2) + 0.5) / window_width, 0.0, 1.0
        )

        if "PhotometricInterpretation" in dataset and dataset.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = 1.0 - pixel_array

        if len(pixel_array.shape) == 2:
            pixel_array = np.stack([pixel_array] * 3, axis=-1)

        return pixel_array

    def get_meta(self, dataset: FileDataset) -> dict[str, str | UID | int | None]:
        """
        Extract metadata from the dataset.

        Parameters
        ----------
        dataset : FileDataset
            The dataset to extract metadata from.

        Returns
        -------
        meta : dict[str, str | UID | int | None]
            The extracted metadata.
        """
        meta = {}
        for source_field, target_field in self.defaults.items():
            if source_field in dataset:
                if target_field is not None:
                    meta[target_field] = getattr(dataset, source_field)
                else:
                    meta[source_field] = getattr(dataset, source_field)

        return meta

    def write_dataset(self, dataset: FileDataset | None) -> str | None:
        """
        Write the dataset to a generated path.

        Parameters
        ----------
        dataset : FileDataset | None
            The dataset to write.

        Returns
        -------
        path : str | None
            The path to the written dataset.
        """
        if dataset is None:
            return None

        try:
            path = str(DATA_DIR / f"{dataset.SOPInstanceUID}.dcm")
            dcmwrite(path, dataset, write_like_original=False)
            return path
        except BaseException as e:
            return None


dicom_manager = DicomManager()