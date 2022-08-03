import SimpleITK as sitk
import os


def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

ct_folder_path = r"C:\Users\Mendes\Desktop\ProjectoDistorcaoMRI\CT_1_vazio"

mri_folder_path = r"C:\Users\Mendes\Desktop\ProjectoDistorcaoMRI\MRI Protoc SRS"

fixed_series_reader = sitk.ImageSeriesReader()
fixed_names = fixed_series_reader.GetGDCMSeriesFileNames(ct_folder_path)
fixed_series_reader.SetFileNames(fixed_names)
fixed_image = fixed_series_reader.Execute()
fixed = sitk.Cast(fixed_image, sitk.sitkFloat32)

moving_series_reader = sitk.ImageSeriesReader()
moving_names = moving_series_reader.GetGDCMSeriesFileNames(mri_folder_path)
moving_series_reader.SetFileNames(moving_names)
moving_image = moving_series_reader.Execute()
moving = sitk.Cast(moving_image, sitk.sitkFloat32)

R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.001, 200)
R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

outTx = R.Execute(fixed, moving)

print("-------")
print(outTx)
print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
print(f" Iteration: {R.GetOptimizerIteration()}")
print(f" Metric value: {R.GetMetricValue()}")

sitk.WriteTransform(outTx, "rigid_transform.h5")

if "SITK_NOSHOW" not in os.environ:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    sitk.Show(cimg, "ImageRegistration1 Composition")