import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from streamlit_image_comparison import image_comparison

from metrics import gmsd, ssim, psnr

# set page config
st.set_page_config(page_title="Image-Comparison Example", layout="centered")

# open images

titles = [
    "HS017 - Run01",
    "HS004 - Run10",
    "HS001 - Run06",
]


# Deformable Registration
reg_paths = [
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/moving.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/demons.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/bspline.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/morph.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/disp.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/moving.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/demons.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/bspline.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/morph.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/disp.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/moving.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/demons.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/bspline.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/morph.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/disp.png"),
]

tgt_paths = [
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr2/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/fixed.png"),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/curr4/fixed.png"),
]

# Stack Registration

for i in range(len(reg_paths)):
    img_path_a = reg_paths[i]
    img_path_b = tgt_paths[i]
    img_a = Image.open(img_path_a).convert("L")
    img_b = Image.open(img_path_b).convert("L")
    img_a = np.asarray(img_a)
    img_b = np.asarray(img_b)

    if i % 5 == 0:
        st.header(titles[i // 5])

    # render image-comparison
    image_comparison(
        img1=img_a,
        img2=img_b,
    )

    st.caption(f"GMSD: {gmsd(img_a, img_b):.3f}")
    st.caption(f"SSIM: {ssim(img_a, img_b):.3f}")
    st.caption(f"PSNR: {psnr(img_a, img_b):.3f}")
