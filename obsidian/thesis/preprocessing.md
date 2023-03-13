
- [[preprocessing#Adjustment by Reference|Adjustment by Reference]]
- [[preprocessing#Dimensionality Reduction|Dimensionality Reduction]]
- [[preprocessing#Normalisation|Normalisation]]

## Adjustment by Reference
### Reflectance
$$I_{\text{ref}} = \frac{I_\text{raw} - I_\text{dark}}{I_\text{white} - I_\text{dark}}$$
### Absorbance
$$I_{\text{abs}} = -\log \frac{I_\text{raw}}{I_\text{ref}}$$

## Dimensionality Reduction
### PCA
1. Center the matrix by subtracting with mean spectral vector
2. Compute the covariance matrix $$\bar{X}\bar{X}^T$$
3. Decompose the covariance matrix into its components
4. Sort eigenvalues and eigenvectors in descending order
5. The first K vectors are used to approximate the image

### HySIME
- HySime algorithm estimates the signal and the noise correlation matrices and then selects the subset of eigenvalues that best represents the signal subspace in the minimum mean squared error sense

## Normalisation