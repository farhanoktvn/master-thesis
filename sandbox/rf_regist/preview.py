import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from streamlit_image_comparison import image_comparison

from metrics import gmsd, ssim, psnr

# set page config
st.set_page_config(page_title="Image-Comparison Example", layout="centered")

# open images


SAMPLES = [
    ("HS005", "Run07"),
    ("HS005", "Run07"),
    ("HS010", "Run05"),
]

titles = []
reg_paths = []
tgt_paths = []

for SAMPLE_ID, RUN_ID in SAMPLES:
    titles.append(f"{SAMPLE_ID} - {RUN_ID}")

    reg_paths.append(
        Path(
            f"/Users/farhanoktavian/imperial/thesis/sandbox/rf_regist/images/{SAMPLE_ID}/{RUN_ID}-mi.png"
        )
    )
    reg_paths.append(
        Path(
            f"/Users/farhanoktavian/imperial/thesis/sandbox/rf_regist/images/{SAMPLE_ID}/{RUN_ID}-bspl.png"
        )
    )
    # reg_paths.append(
    #     Path(
    #         f"/Users/farhanoktavian/imperial/thesis/sandbox/rf_regist/images/{SAMPLE_ID}/{RUN_ID}-disp.png"
    #     )
    # )
    reg_paths.append(
        Path(
            f"/Users/farhanoktavian/imperial/thesis/sandbox/rf_regist/images/{SAMPLE_ID}/{RUN_ID}-rf.png"
        )
    )

    tgt_path = Path(
        f"/Users/farhanoktavian/imperial/thesis/sandbox/rf_regist/images/{SAMPLE_ID}/{RUN_ID}-raw.png"
    )
    # tgt_path_list = [tgt_path] * 4
    tgt_path_list = [tgt_path] * 3
    tgt_paths.extend(tgt_path_list)


# Stack Registration
for i in range(len(reg_paths)):
    img_path_a = reg_paths[i]
    img_path_b = tgt_paths[i]
    img_a = Image.open(img_path_a).convert("L")
    img_b = Image.open(img_path_b).convert("L")
    img_a = np.asarray(img_a)
    img_b = np.asarray(img_b)

    if i % 3 == 0:
        st.header(titles[i // 3])

    # render image-comparison
    image_comparison(
        img1=img_a,
        img2=img_b,
    )

    st.caption(f"GMSD: {gmsd(img_a, img_b):.3f}")
    st.caption(f"SSIM: {ssim(img_a, img_b):.3f}")
    st.caption(f"PSNR: {psnr(img_a, img_b):.3f}")
