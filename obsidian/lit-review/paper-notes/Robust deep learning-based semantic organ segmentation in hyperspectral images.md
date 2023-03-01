
## Gist
Segmentation of HSI images from pigs with 19 classes. Unprocessed HSI data has its advantage over RGb of camera-processed data.


## Data
- 500—1000nm
- 506 HSI images from 20 pigs
- For each orga, 32—405 images were acquired from 5—20 individuals

### Image classification
- Pixel spectrum is wavelength of a pixel
- Superpixel is 32x32xC iamge
- Patch is 64x64xC image
- Image is full size image


## Preprocessing
- Adjusted using dark and white reference cube
- l1-norm is applied to each pixel spectrum


## Learning Process
- Image types based classification
- Image are pased into U-Net, while spectrum is passed into a separate DL network
- Network outputs are aggregated at the end


## Evaluation
- Image type produced greater evaluation score than other image input type
- HSI performs better than RGB and TPI (tissue parameter images; includes StO2, perfusion, and hemoglobin)
- Method: dive similarity coefficient, average sufrace distancem and normalised surface dice