import cv2
import numpy as np
import SimpleITK as sitk


class RegD:
    def __init__(self):
        pass

    def register(self, img_a, img_b):
        img_a_cp = img_a.copy()
        img_a_eq = cv2.equalizeHist(img_a_cp)
        img_a_cp = img_a_cp.astype(np.float32)
        img_a_eq = img_a_eq.astype(np.float32)

        img_b_cp = img_b.copy()
        img_b_eq = cv2.equalizeHist(img_b_cp)
        # img_b_cp = img_b_cp.astype(np.float32)
        img_b_eq = img_b_eq.astype(np.float32)

        moving = sitk.GetImageFromArray(img_a_cp)
        moving_eq = sitk.GetImageFromArray(img_a_eq)
        # fixed = sitk.GetImageFromArray(img_b_cp)
        fixed_eq = sitk.GetImageFromArray(img_b_eq)

        transformDomainMeshSize = [8] * moving_eq.GetDimension()
        tx = sitk.BSplineTransformInitializer(fixed_eq, transformDomainMeshSize)

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()

        R.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=50,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7,
        )
        R.SetInitialTransform(tx, True)
        R.SetInterpolator(sitk.sitkLinear)

        outTx = R.Execute(fixed_eq, moving_eq)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_eq)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)

        out = resampler.Execute(moving)
        out = sitk.GetArrayFromImage(out)
        out = out.astype(np.uint8)

        h, w = img_b.shape
        # return cv2.warpPerspective(img_a, M, (w, h))
        # return cv2.warpPerspective(img_a, M, (w, h)), M
        # return M, w, h
        return out
