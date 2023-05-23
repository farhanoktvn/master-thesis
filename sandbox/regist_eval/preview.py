import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from streamlit_image_comparison import image_comparison

# set page config
st.set_page_config(page_title="Image-Comparison Example", layout="centered")

# open images

reg_paths = [
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/1/reg_a.png"),
    Path(
        "/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/1/SISLK_Result/Registered_Sequence/frame_0.png"
    ),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/2/reg_a.png"),
    Path(
        "/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/2/SISLK_Result/Registered_Sequence/frame_0.png"
    ),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/3/reg_a.png"),
    Path(
        "/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/3/SISLK_Result/Registered_Sequence/frame_0.png"
    ),
]

tgt_paths = [
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/1/target.png"),
    Path(
        "/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/1/SISLK_Result/Registered_Sequence/frame_1.png"
    ),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/2/target.png"),
    Path(
        "/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/2/SISLK_Result/Registered_Sequence/frame_1.png"
    ),
    Path("/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/3/target.png"),
    Path(
        "/Users/farhanoktavian/imperial/thesis/sandbox/regist_eval/images/3/SISLK_Result/Registered_Sequence/frame_1.png"
    ),
]

for i in range(len(reg_paths)):
    img_path_a = reg_paths[i]
    img_path_b = tgt_paths[i]
    img_a = Image.open(img_path_a).convert("L")
    img_b = Image.open(img_path_b).convert("L")
    img_a = np.asarray(img_a)
    img_b = np.asarray(img_b)

    # render image-comparison
    image_comparison(
        img1=img_a,
        img2=img_b,
    )
