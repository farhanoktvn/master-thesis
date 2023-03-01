
## Gist
- CNN and SVM for classification. 80% overall accuracy


## Data
- Based on the [[HELICoiD project — a new use of hyperspectral imaging for brain cancer detection in real-time during neurosurgical operations]]


## Preprocessing
- 400—440nm and 700—826nm removed
- Ranging 440—902
- Use a band-selection algorithm, ant-colony optimisation


## Learning Process
- Use a combined 3D then 3D CNN
- softmax layer at the end
- Also uses linear SVM classifier
- Another is hybrid CNN then SVM clasifier


## Evaluation
- 3D-2D model perform the best with 80% accuracy
- However, AUC is highest when CNN is combined with SVM
- Methods: accuracy and ROC