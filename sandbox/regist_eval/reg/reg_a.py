import cv2
import numpy as np


PREPROCESSING_METHODS = {
    "eq_hist": cv2.equalizeHist,
    "clahe": cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8)).apply,
    "sharpen": cv2.filter2D,
}


# Feature matching params
K = 2
LOWES_RATIO = 0.8


class RegA:
    def __init__(
        self, preprocessing_methods=["eq_hist"], extraction_method="sift", matching_method="flann"
    ):
        self._init_methods(preprocessing_methods, extraction_method, matching_method)

    def _init_methods(self, preprocessing_methods, extraction_method, matching_method):
        self.preprocessing_methods = preprocessing_methods

        if extraction_method == "sift":
            self.extractor = cv2.SIFT_create()
        elif extraction_method == "surf":
            self.extractor = cv2.SURF_create()
        elif extraction_method == "orb":
            self.extractor = cv2.ORB_create()
        elif extraction_method == "akaze":
            self.extractor = cv2.AKAZE_create()
        else:
            raise ValueError("Invalid extraction method")

        if matching_method == "flann":
            FLANN_INDEX_KDTREE = 1
            TREE_SIZE = 5
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=TREE_SIZE)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif matching_method == "bf":
            self.matcher = cv2.BFMatcher()
        else:
            raise ValueError("Invalid matching method")

    def preprocess(self, img):
        for method in self.preprocessing_methods:
            if method == "sharpen":
                img = PREPROCESSING_METHODS[method](
                    img, -1, np.array([[0, -1, 0], [-1, 9, -1], [0, -1, 0]])
                )
            else:
                img = PREPROCESSING_METHODS[method](img)
        return img

    def detect_keypoint(self, img):
        kp, desc = self.extractor.detectAndCompute(img, None)
        return kp, desc

    def register(self, img_a, img_b):
        img_a_cp = img_a.copy()
        img_b_cp = img_b.copy()
        img_a_cp = self.preprocess(img_a_cp)
        img_b_cp = self.preprocess(img_b_cp)

        kp_a, desc_a = self.detect_keypoint(img_a_cp)
        kp_b, desc_b = self.detect_keypoint(img_b_cp)

        matches = self.matcher.knnMatch(desc_a, desc_b, k=K)

        # Filter matches by lowe's ratio
        good_matches = []
        for m, n in matches:
            if m.distance < LOWES_RATIO * n.distance:
                good_matches.append(m)

        # Filter matches by homography
        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
            return None

        h, w = img_b.shape
        return cv2.warpPerspective(img_a, M, (w, h))
