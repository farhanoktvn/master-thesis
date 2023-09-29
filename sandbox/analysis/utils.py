import torch.nn.functional as F
import PIL.Image as Image
import torch
import numpy as np

from PIL import Image


def load_bw_img(img_path):
    img = Image.open(img_path).convert("L")
    return img


def load_color_img(img_path):
    img = Image.open(img_path)
    return img
