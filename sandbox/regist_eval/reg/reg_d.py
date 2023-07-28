import cv2
import numpy as np
import SimpleITK as sitk


class RegD:
    def __init__(self):
        pass

    def register(self, img_a, img_b):
        img_a_cp = img_a_cp.astype(np.float32)
        img_b_cp = img_b_cp.astype(np.float32)

        moving = sitk.GetImageFromArray(img_a_cp)
        fixed = sitk.GetImageFromArray(img_b_cp)

        transformDomainMeshSize = [8] * moving.GetDimension()
        tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

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

        outTx = R.Execute(fixed, moving)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)

        out = resampler.Execute(moving2)
        out = sitk.GetArrayFromImage(out)
        out = out.astype(np.uint8)

        h, w = img_b.shape
        # return cv2.warpPerspective(img_a, M, (w, h))
        # return cv2.warpPerspective(img_a, M, (w, h)), M
        # return M, w, h
        return out
