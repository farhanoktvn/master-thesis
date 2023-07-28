import cv2
import numpy as np
import pandas as pd

from pathlib import Path

from utils import load_bw_img, load_color_img


WAVELENGTH_AMOUNT = 29
CHANNEL_WAVELENGTH = {
    "b": (460, 540),
    "g": (540, 620),
    "r": (620, 700),
}


class DataLoader:
    def __init__(self, root_dir, metadata_dir):
        self.root_dir = Path(root_dir)
        self.metadata = pd.read_hdf(metadata_dir)  # read h5 file
        self.samples = self._init_samples(self.root_dir)

    def _init_samples(self, root_dir):
        sample_ids = self.metadata["sample_id"].unique()
        samples = list()
        for sample_id in sample_ids:
            samples.append(
                Sample(
                    sample_id,
                    self.root_dir,
                    self.metadata.loc[self.metadata["sample_id"] == sample_id],
                )
            )
        return samples

    def __repr__(self):
        return f"DataLoader(root_dir={self.root_dir}, num_samples={len(self.samples)})"


class Sample:
    def __init__(self, sample_id, root_dir, sample_df):
        self.sample_id = sample_id
        self.root_dir = root_dir
        self._init_data(sample_df)

    def _init_data(self, sample_df):
        run_ids = sample_df["run_id"].unique()

        runs = list()
        orients = list()
        whites = list()
        darks = list()

        for run_id in run_ids:
            # remove run_id with run_group equal to orient white or dark
            run_group = sample_df.loc[sample_df["run_id"] == run_id]["run_group"].tolist()[0]
            if run_group == "orient":
                orients.append(
                    Run(
                        self.sample_id,
                        run_id,
                        self.root_dir,
                        sample_df.loc[sample_df["run_id"] == run_id],
                    )
                )
            elif run_group == "white":
                whites.append(
                    Run(
                        self.sample_id,
                        run_id,
                        self.root_dir,
                        sample_df.loc[sample_df["run_id"] == run_id],
                    )
                )
            elif run_group == "dark":
                darks.append(
                    Run(
                        self.sample_id,
                        run_id,
                        self.root_dir,
                        sample_df.loc[sample_df["run_id"] == run_id],
                    )
                )
            else:
                runs.append(
                    Run(
                        self.sample_id,
                        run_id,
                        self.root_dir,
                        sample_df.loc[sample_df["run_id"] == run_id],
                    )
                )
        self.runs = runs
        self.orients = orients
        self.whites = whites
        self.darks = darks

    def __repr__(self) -> str:
        return f"Sample({self.sampl})"


class Run:
    def __init__(self, sample_id, run_id, root_dir, run_df):
        self.sample_id = sample_id
        self.run_id = run_id
        self.root_dir = root_dir
        self._init_run(run_df)

    def _init_run(self, run_df):
        if run_df.iloc[0]["label_img_path"] != "nan":
            self.label_img_path = self.root_dir / run_df.iloc[0]["label_img_path"]
            self.label_mask_path = self.root_dir / run_df.iloc[0]["label_mask_path"]
            self.flip = run_df.iloc[0]["flip"]
            self.rotate = int(float(run_df.iloc[0]["rotate"]))

        run_img_paths = run_df["run_img_paths"].tolist()[0]
        run_img_paths = [self.root_dir / img_path for img_path in run_img_paths]
        run_img_paths_len = len(run_img_paths)

        self.collection_count = run_img_paths_len // WAVELENGTH_AMOUNT
        self.collection = [list() for _ in range(self.collection_count)]
        for i in range(run_img_paths_len):
            collection_idx = i % self.collection_count
            self.collection[collection_idx].append(run_img_paths[i])

    def __repr__(self) -> str:
        return f"Run(sample={self.sample_id}, run={self.run_id})"

    def get_label_image(self, channel=""):
        # to_bw: convert to bw image
        # channel: bw, r, g, b or empty string

        # Check whether the label is color or bw
        if self.label_img_path.suffix.lower == "png":  # bw: taken from spectral image
            image = load_bw_img(self.label_img_path)
        else:  # color
            image = load_color_img(self.label_img_path)
            try:
                if channel == "bw":
                    image = image.convert("L")
                elif channel == "r":
                    image = image.getchannel("R")
                elif channel == "g":
                    image = image.getchannel("G")
                elif channel == "b":
                    image = image.getchannel("B")
                else:
                    pass
            except ValueError:
                print(f"Cannot get channel {channel} from {self.label_img_path.name}")
                print("Use the bw image instead")
                image = image.convert("L")

        # Convert to numpy array
        image = np.asarray(image)

        # Flip and rotate
        if self.flip == "h":
            image = cv2.flip(image, 1)
        elif self.flip == "v":
            image = cv2.flip(image, 0)
        elif self.flip == "hv":
            image = cv2.flip(image, -1)
        elif self.flip != "":
            pass

        # Rotate
        if self.rotate == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotate == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.rotate != 0:
            pass

        return image

    def get_spectral_images(self, collection_idx=0, channel="", with_path=False):
        collection = self.collection[collection_idx]

        # Filter by channel
        if channel != "":
            new_collection = list()
            wavelength_range = CHANNEL_WAVELENGTH[channel]
            for nm in range(wavelength_range[0], wavelength_range[1], 10):
                try:
                    new_collection.append([p for p in collection if str(nm) in p.name][0])
                except IndexError as e:
                    raise Exception(f"Cannot find image with wavelength {nm} in {collection[0]}")
            collection = new_collection

        # Load images
        spectral_images = list()
        spectral_images_paths = list()
        for image_path in collection:
            image = load_bw_img(image_path)
            image = np.asarray(image)
            spectral_images.append(image)
            if with_path:
                spectral_images_paths.append(image_path)

        if with_path:
            return spectral_images, spectral_images_paths

        return spectral_images

    def get_spectral_image(self, wavelength=560, collection_idx=0):
        collection = self.collection[collection_idx]
        image_path = [p for p in collection if str(wavelength) in p.name][0]
        image = load_bw_img(image_path)
        image = np.asarray(image)
        return image
