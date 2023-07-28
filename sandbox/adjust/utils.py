from PIL import Image


def load_bw_img(img_path):
    img = Image.open(img_path)
    img = img.convert("L")
    return img


def load_color_img(img_path):
    img = Image.open(img_path)
    return img
