
## Gist
Using in-vivo and ex-vivo brain tumour samples. Discrimination based on spectral ratio oh hemoglobin. Ex-vivo generate higher hemoglobin ratios


## Data
- 400—1000 nm
- 7 in-vivo and 14 ex-vivo
- gliblastoma grade IV, metastatic secondary breast cancer, meningioma grade I and II, and astrocytoma (glioma) grade II.
- near-square images with 826 channels
- R545/R560 band for hemoglobin


## Preprocessing
- Noise reduction and band removal
- Dark and white reference image
- 400—440nm and 909—1000nm were removed


## Learning Process
- K-means clustering, result in two clusters in almost all images
- Spectral analysis based on the tissue absorbance value


## Evaluation
- Higher ratio values in ex-vivo data
- Normal tissue has the highest ratio followed by tumour then blood vessels