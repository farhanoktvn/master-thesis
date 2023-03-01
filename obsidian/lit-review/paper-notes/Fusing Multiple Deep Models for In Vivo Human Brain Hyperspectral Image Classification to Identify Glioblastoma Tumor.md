
## Gist
Using a fusion of DNN models. Data from 16 patients. 96,69% for four-class segmentation and 96.34% for GBM tunmor identification.


## Data
- [[HELICoiD project — a new use of hyperspectral imaging for brain cancer detection in real-time during neurosurgical operations]]


## Preprocessing
- Image calibration, dark and white
- Hyperspectral signal identification by minimum error (HySIME) filter
- Dimensionality reduction 400—440nm and 902—1000nm are removed
- Data normalisation


## Learning Process
- Combining spectral and spectral-spatial
- Combined machine learning model
- Spectral phsor analysis and SMOTE, followed by 1D and 2D CNN (FMDM)


## Evaluation
- Highest accuracy on all test, compared to 1D-NN and 2D-NN
- Highest accuracy on tumor tissue segmentation than HRnet+OCR, CnCnn, DPResnet and DBDA. Input is the stacked phasor representation and spectral features of the HSI
- Methods: accuracy, precision, recall