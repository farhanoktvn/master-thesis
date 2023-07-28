import cv2
import numpy as np
import SimpleITK as sitk


class RegC:
    def __init__(self):
        pass

    def register(self, img_a, img_b):
        img_a_cp = img_a.copy()
        img_a_cp = img_a_cp.astype(np.float32)
        img_b_cp = img_b.copy()
        img_b_cp = img_b_cp.astype(np.float32)

        moving = sitk.GetImageFromArray(img_a_cp)
        fixed = sitk.GetImageFromArray(img_b_cp)

        initialTx = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.AffineTransform(fixed.GetDimension())
        )

        R = sitk.ImageRegistrationMethod()

        displacementField = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
        displacementField.CopyInformation(fixed)
        displacementTx = sitk.DisplacementFieldTransform(displacementField)
        del displacementField
        displacementTx.SetSmoothingGaussianOnUpdate(
            varianceForUpdateField=0.0, varianceForTotalField=1.5
        )

        R.SetMovingInitialTransform(initialTx)
        R.SetInitialTransform(displacementTx, inPlace=True)

        R.SetMetricAsANTSNeighborhoodCorrelation(4)
        R.MetricUseFixedImageGradientFilterOff()

        R.SetShrinkFactorsPerLevel([3, 2, 1])
        R.SetSmoothingSigmasPerLevel([2, 1, 1])

        R.SetOptimizerScalesFromPhysicalShift()
        R.SetOptimizerAsGradientDescent(
            learningRate=1,
            numberOfIterations=300,
            estimateLearningRate=R.EachIteration,
        )

        R.Execute(fixed, moving)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(displacementTx)

        out = resampler.Execute(moving)
        out = sitk.GetArrayFromImage(out)
        out = out.astype(np.uint8)

        h, w = img_b.shape
        # return cv2.warpPerspective(img_a, M, (w, h))
        # return cv2.warpPerspective(img_a, M, (w, h)), M
        # return M, w, h
        return out
