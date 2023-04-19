import cv2
import json
import numpy as np
import re
import tqdm

from enum import Enum
from pathlib import Path
from PIL import Image

LABEL_IMG_PATTERN = '.*\.JPG'
LABEL_MASK_PATTERN = '.*\.json'
MSI_IMG_PATTERN = '.*\.png'


class Tissue(Enum):
    BLOOD = 1
    BLOOD_VESSEL = 2
    ARACHNOID = 3
    TUMOUR_CORE = 4
    CORTICAL_SURFACE = 5
    TUMOUR_MARGINS = 6
    WHITE_MATTER = 7
    DURA_MATTER = 8

TISSUE_DICT = {
    'Blood': Tissue.BLOOD,
    'Blood vessel': Tissue.BLOOD_VESSEL,
    'Arachnoid': Tissue.ARACHNOID,
    'Tumour Core': Tissue.TUMOUR_CORE,
    'Cortical Surface': Tissue.CORTICAL_SURFACE,
    'Tumour Margins': Tissue.TUMOUR_MARGINS,
    'White Matter': Tissue.WHITE_MATTER,
    'Dura': Tissue.DURA_MATTER
}

cwd = Path.cwd()
data_folder = Path(cwd, 'dataset-sample')
reg_label_folder = Path(cwd, 'reg-labels')


# Load image
def load_image(image_path):
    return np.asarray(Image.open(image_path).convert('L'))

# Load label mask
def load_label_mask(label_mask_path, label_img_shape):
    labels = json.load(label_mask_path.open())
    labels = labels.get('shapes')

    label_mask = np.zeros(label_img_shape, dtype=np.uint8)
    for label in labels:
        if label.get('shape_type') == 'linestrip':
            points = label.get('points')
            curr_point = points[0]
            for point in points[1:]:
                cv2.line(
                    label_mask,
                    (int(curr_point[0]), int(curr_point[1])),
                    (int(point[0]), int(point[1])),
                    TISSUE_DICT[label.get('label')].value,
                    6
                )
                curr_point = point
        elif label.get('shape_type') == 'polygon':
            points = np.array(label.get('points'), dtype=np.int32)
            cv2.fillPoly(
                label_mask,
                [points],
                TISSUE_DICT[label.get('label')].value
            )
        else:
            raise ValueError(f'Unknown shape type: {label.get("shape_type")}')
    
    return label_mask


class SampleLoader:

    def __init__(self, data_folder, sample_info):
        self.data_folder = data_folder
        self.id = None
        self.sample_path = None
        self.references = None
        self.data = None
        self._init_sample_info(sample_info)

    def _init_sample_info(self, sample_info):
        self.id = sample_info.get('id')
        self.sample_path = Path(self.data_folder, self.id)
        self.references = sample_info.get('ref_runs')
        self.data = sample_info.get('data_runs')
        self.transform = sample_info.get('transform')

    def __repr__(self):
        return f"Sample(id={self.id})"
    
    # Data Images
    def get_data_paths(self, with_cap=True):
        """Return list of key-value pairs of label image and run image paths.
        
        Return: [
            {
                'run_group': str,
                'label_img_path': Path,
                'label_mask_path': Path,
                'runs': [
                    {
                        'run_id': str,
                        'run_img_paths': [Path]
                    }
                ]
            }
        ]
        """
        
        labelled_folder = self.data.get('capillary').get('included') if with_cap else self.data.get('capillary').get('excluded')
        labelled_img_path = Path(self.sample_path, 'Labelling data', 'Definitive segmentation', labelled_folder)

        paths = []
        run_group = self.data.get('runs').keys()
        for group in run_group:
            group_path = Path(labelled_img_path, group)
            label_img_path = [x for x in group_path.iterdir() if re.match(LABEL_IMG_PATTERN, x.name)][0]
            label_mask_path = [x for x in group_path.iterdir() if re.match(LABEL_MASK_PATTERN, x.name)][0]

            group_info = {
                'group_id': group,
                'label_img_path': label_img_path,
                'label_mask_path': label_mask_path,
                'runs': []
            }

            run_ids = self.data.get('runs').get(group)
            for run_id in run_ids:
                run_path = Path(data_folder, self.id, run_id)
                run_img_paths = [x for x in run_path.iterdir() if re.match(MSI_IMG_PATTERN, x.name)]

                group_info['runs'].append({
                    'run_id': run_id,
                    'run_img_paths': sorted(run_img_paths)
                })
            
            paths.append(group_info)
        
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
            run_paths = [Path(data_folder, self.id, run_id) for run_id in run_ids]
            if run_paths == []:
                paths[ref_img] = None
                continue
            run_img_paths = [x for x in run_paths[0].iterdir() if re.match(MSI_IMG_PATTERN, x.name)]

            if run_img_paths != []:
                paths[ref_img] = run_img_paths[0]
            else:
                paths[ref_img] = None
        
        return paths
    
    # Image Properties
    @property
    def orient_ref_path(self):
        return self.get_reference_paths().get('orient')

    @property
    def white_ref_path(self):
        return self.get_reference_paths().get('white')

    @property
    def dark_ref_path(self):
        return self.get_reference_paths().get('dark')


class RunLoader:

    def __init__(
            self,
            sample_id,
            run_group,
            run_id,
            label_img_path,
            label_mask_path,
            run_img_paths,
            transform=None,
            save_path=None
        ):
        self.sample_id = sample_id
        self.run_group = run_group
        self.run_id = run_id
        self.save_path = save_path
        self._init_data_paths(label_img_path, label_mask_path, run_img_paths, transform)
    

    def _init_data_paths(self, label_img_path, label_mask_path, run_img_paths, transform):
        self.label_img = load_image(label_img_path)
        self.label_mask = load_label_mask(label_mask_path, self.label_img.shape)
        self.run_imgs = [load_image(run_img_path) for run_img_path in run_img_paths]

        # Known image transformation
        # Flip
        img_flip = transform.get('flip')
        self.label_img = cv2.flip(self.label_img, 0) if 'v' in img_flip else self.label_img
        self.label_mask = cv2.flip(self.label_mask, 0) if 'v' in img_flip else self.label_mask
        self.label_img = cv2.flip(self.label_img, 1) if 'h' in img_flip else self.label_img
        self.label_mask = cv2.flip(self.label_mask, 1) if 'h' in img_flip else self.label_mask
        # Rotate
        img_rotate = transform.get('rotate')
        if img_rotate == 90:
            self.label_img = cv2.rotate(self.label_img, cv2.ROTATE_90_CLOCKWISE)
            self.label_mask = cv2.rotate(self.label_mask, cv2.ROTATE_90_CLOCKWISE)
        if img_rotate == 180:
            self.label_img = cv2.rotate(self.label_img, cv2.ROTATE_180)
            self.label_mask = cv2.rotate(self.label_mask, cv2.ROTATE_180)
        if img_rotate == -90:
            self.label_img = cv2.rotate(self.label_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.label_mask = cv2.rotate(self.label_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    
    def adjust_reflectance(self, white_ref_imgs, dark_ref_imgs):
        # Calculate reflectance
        res = []
        for i in range(len(self.run_imgs)):
            img = self.run_imgs[i]
            white_val = np.average(white_ref_imgs[i])
            dark_val = np.average(dark_ref_imgs[i])
            adj_img = (img - dark_val) / (white_val - dark_val)
            res.append(adj_img)
        
        return res

    def register_label(self):
        label_img = self.label_img.copy()
        label_mask = self.label_mask.copy()

        sift = cv2.SIFT_create()

        # Find keypoints
        try:
            max_good = -1
            best_matches = None
            best_kp_1, best_kp_2 = None, None
            h, w = self.run_imgs[0].shape
            for run_img in tqdm.tqdm(
                self.run_imgs,
                desc=f'Registering Label: {self.sample_id}-{self.run_id}',
                position=3,
                leave=False
            ):
                # Find keypoints
                kp_1, desc_1 = sift.detectAndCompute(label_img, None)
                kp_2, desc_2 = sift.detectAndCompute(run_img, None)

                # # brute force feature matching
                # bf = cv2.BFMatcher()
                # matches = bf.knnMatch(desc_1, desc_2, k=2)

                # flann feature matching
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks = 50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(desc_1, desc_2, k=2)

                # store all the good matches as per Lowe's ratio test.
                good = []
                for m, n in matches:
                    if m.distance < 0.75*n.distance:
                        good.append(m)
                
                if len(good) > max_good:
                    max_good = len(good)
                    best_matches = good
                    best_kp_1 = kp_1
                    best_kp_2 = kp_2

            MIN_MATCH_COUNT = 30
            if max_good >= MIN_MATCH_COUNT:
                src_pts = np.float32([ best_kp_1[m.queryIdx].pt for m in best_matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ best_kp_2[m.trainIdx].pt for m in best_matches ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
            else:
                return 'Not enough matches are found - {len(best_matches)} < {MIN_MATCH_COUNT}'
            
            # Transform label image
            self.label_img = cv2.warpPerspective(label_img, M, (w, h))
            # self.label_mask = cv2.warpPerspective(label_mask, M, (w, h))
            return None
        except cv2.error as e:
            return e
    
    def save(self):
        if self.save_path is None:
            return

        # Save label image
        label_img_path = Path(self.save_path, self.sample_id, 'img', f'{self.run_id}.png')
        label_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(label_img_path), self.label_img)

        # label_mask_path = Path(self.save_path, self.sample_id, 'label', f'{self.run_id}.png')
        # label_mask_path.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(str(label_mask_path), self.label_mask)


if __name__ == '__main__':
    sample_infos = json.load(Path(data_folder, 'hs-info.json').open())
    samples = [SampleLoader(data_folder, sample_info) for sample_info in sample_infos.get('data')]
    samples = samples[1:]

    run_logs = ''
    for sample in tqdm.tqdm(samples, desc='Samples', position=0):
        run_groups = sample.get_data_paths()
        for run_group in tqdm.tqdm(run_groups, desc='Run Groups', position=1, leave=False):
            for run_id in tqdm.tqdm(run_group.get('runs'), desc='Runs', position=2, leave=False):
                run_data = RunLoader(
                    sample.id,
                    run_group.get('group_id'),
                    run_id.get('run_id'),
                    run_group.get('label_img_path'),
                    run_group.get('label_mask_path'),
                    run_id.get('run_img_paths'),
                    transform=sample.transform,
                    save_path=reg_label_folder
                )
                # run_data.adjust_reflectance(
                #     load_image(sample.white_ref_path),
                #     load_image(sample.dark_ref_path)
                # )
                reg = run_data.register_label()
                if reg is not None:
                    run_logs += f'Error at {sample.id}-{run_id.get("run_id")}\n'
                    continue
                run_data.save()
    
    print('These runs failed to register:')
    print(run_logs)