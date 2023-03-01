
## Gist
HSI channel reduction and preselection to optimise performance vs. accuracy trade off. Done using deep learning. Collab w/ Carl Zeiss


## Data
- 468—787 nm
- Acquired using Carl-Zeiss camera
- 5 patients, underwent glioma resection


## Preprocessing
- Clustering by spectral angle mapper (SAM)
- Improve signal-to-noise ratio using SLIC algorithm
- Produced SLIC tiles as input to the neural network


## Learning Process
- random forest, SVM with radial basis function, and multi-layer perceptron
- CNN


## Evaluation
- RF: 86%, SVM: 91%, One layer MLP: 92%
- CNN: Highest is 89% with 12 channels compression