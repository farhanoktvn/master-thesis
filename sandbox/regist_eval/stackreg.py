import cv2
import time
from multiprocessing import Pool
from pathlib import Path

from loader.dataloader import DataLoader
from reg.reg_d import RegD

DATA_DIR = "/Users/farhanoktavian/imperial/thesis/sandbox/dataset/raw"
METADATA_DIR = "/Users/farhanoktavian/imperial/thesis/sandbox/dataset/data.h5"
STACKREG_DIR = Path("/Users/farhanoktavian/imperial/thesis/sandbox/dataset/stackreg3")

run_loader = DataLoader(DATA_DIR, METADATA_DIR)


sample = run_loader.samples[3]
run = sample.runs[6]
print(f"{sample.sample_id} - {run.run_id}")


target_wavelength = 600

reg = RegD()


def reg_and_write(img, img_path, tgt_img_ln):
    img = reg.register(img, tgt_img_ln)
    save_path = STACKREG_DIR / sample.sample_id / run.run_id
    save_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path / img_path.name), img)


print("Start")
start_t = time.time()

for c in range(run.collection_count):
    print(f"Collection {c}/{run.collection_count}")
    imgs, img_paths = run.get_spectral_images(collection_idx=c, with_path=True)
    tgt_img = run.get_spectral_image(wavelength=target_wavelength, collection_idx=c)
    tgt_imgs = [tgt_img for _ in range(len(imgs))]

    pool_arg = zip(imgs, img_paths, tgt_imgs)

    with Pool() as p:
        p.starmap(reg_and_write, pool_arg)

end_t = time.time()
print("Done")
print(f"Time taken: {end_t - start_t}s")
