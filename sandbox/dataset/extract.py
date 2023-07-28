import cv2
import json
import math
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import re

from pathlib import Path
from PIL import Image


# Path
DATA_DIR = "raw"

# Regex
LABEL_IMG_PATTERN = '.*\.(JPG|png)'
LABEL_MASK_PATTERN = '.*\.json'
MSI_IMG_PATTERN = '.*\.png'

cwd = Path.cwd()
data_folder = Path(cwd, DATA_DIR)
reg_data_folder = Path(cwd, 'reg-data')


class Sample:

    def __init__(self, data_folder, sample_info):
        self.data_folder = data_folder
        self.sample_id = None
        self.sample_path = None
        self.references = None
        self.data = None
        self._init_sample_info(sample_info)

    def _init_sample_info(self, sample_info):
        self.sample_id = sample_info.get('id')
        self.sample_path = Path(self.data_folder, self.sample_id)
        self.references = sample_info.get('ref_runs')
        self.data = sample_info.get('data_runs')

    def __repr__(self):
        return f"Sample(id={self.sample_id})"
    
    # Data Images
    def get_data_paths(self, with_cap=True):
        """Return list of key-value pairs of label image and run image paths.
        
        Return: [
            {
                'label_img_path': Path,
                'label_mask_path': Path,
                'run_img_paths': [Path, Path, ...]
            },
        ]
        """
        
        labelled_folder = self.data.get('capillary').get('included') if with_cap else self.data.get('capillary').get('excluded')
        possible_label_folder = ['Definitive segmentation', 'Definitive segmentations']
        labelled_img_path = None
        for folder in possible_label_folder:
            labelled_img_path = Path(self.sample_path, 'Labelling data', folder, labelled_folder)
            if labelled_img_path.exists():
                break
        
        if labelled_img_path is None:
            raise Exception(f"Labelled image folder not found for sample: {self.sample_id}")

        paths = []
        run_group = self.data.get('runs').keys()
        for group in run_group:
            group_path = Path(labelled_img_path, group)
            label_img_path = [x for x in group_path.iterdir() if re.match(LABEL_IMG_PATTERN, x.name)][0]
            label_mask_path = [x for x in group_path.iterdir() if re.match(LABEL_MASK_PATTERN, x.name)][0]

            run_ids = self.data.get('runs').get(group)
            for run_id in run_ids:
                run_path = Path(data_folder, self.sample_id, run_id)
                run_img_paths = [x for x in run_path.iterdir() if re.match(MSI_IMG_PATTERN, x.name)]

                paths.append({
                    'label_img_path': label_img_path,
                    'label_mask_path': label_mask_path,
                    'run_img_paths': sorted(run_img_paths)
                })
        
        return paths

    # Reference Images
    def get_reference_paths(self):
        """Return list of key-value pairs of reference image paths.
        
        Return: {
            'orient': Path,
            'white': Path,
            'dark': Path
        }
        """
        
        paths = {}
        ref_imgs = self.references.keys()
        for ref_img in ref_imgs:
            run_ids = self.references.get(ref_img)
            run_paths = [Path(data_folder, self.sample_id, run_id) for run_id in run_ids]
            run_img_paths = [x for x in run_paths[0].iterdir() if re.match(MSI_IMG_PATTERN, x.name)]

            paths[ref_img] = sorted(run_img_paths)
        
        return paths
    
    # Reference Images
    @property
    def orient_ref_path(self):
        return self.get_reference_paths().get('orient')

    @property
    def white_ref_path(self):
        return self.get_reference_paths().get('white')

    @property
    def dark_ref_path(self):
        return self.get_reference_paths().get('dark')