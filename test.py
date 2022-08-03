import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

ct_folder_path = r"C:\Users\Mendes\Desktop\ProjectoDistorcaoMRI\CT_1_vazio"

## Read folder with series
ct_reader = sitk.ImageSeriesReader()
ct_file_names = ct_reader.GetGDCMSeriesFileNames(ct_folder_path)
ct_reader.SetFileNames(ct_file_names)
ct_reader.MetaDataDictionaryArrayUpdateOn()
ct_volume = ct_reader.Execute()

imgs_list_idx = [64, 39, 14, -11, -36, -61, -86, -111]

slab_z_index = imgs_list_idx[0]

## Select image with z-index
for i in range(0, ct_volume.GetSize()[2]):
    if int(ct_reader.GetMetaData(i, '0020|1041')) == slab_z_index:
        img_idx = i

img_slice = ct_volume[:,:,img_idx]

## Get info
window_center = int(ct_reader.GetMetaData(img_idx, '0028|1050'))
window_width = int(ct_reader.GetMetaData(img_idx, '0028|1051'))
img_height = img_slice.GetHeight()
img_width = img_slice.GetWidth()
pixel_spacing = img_slice.GetSpacing()
origin = img_slice.GetOrigin()

## Intensity Window
img_slice = sitk.IntensityWindowing(img_slice, window_center - window_width, window_center + window_width)

## Remove the table from image (if hardtop)
table_height = ct_reader.GetMetaData(img_idx, '0018|1130')
table_thickness = 79
table_cut = int(round(img_width/2, 0) + int(table_height) - table_thickness)

## Create a mask for the table
mask = sitk.Image([img_height, img_width], sitk.sitkUInt16, 1)
mask.SetSpacing(pixel_spacing)
mask.SetOrigin(origin)
for i in range(0, img_width):
    for j in range(0, table_cut):
        mask.SetPixel(i, j, 1) 

## Remove the table
table_masked_img = sitk.Mask(img_slice, maskImage=mask, outsideValue=0, maskingValue=0)
table_masked_img_255 = sitk.Cast(sitk.RescaleIntensity(table_masked_img), sitk.sitkUInt8)

#image = sitk.InvertIntensity(sitk.Abs(img_slice))

## Erosion and dilation
image_comp = sitk.BinaryErode(sitk.BinaryDilate(img_slice, [3, 1]), [3, 1])

## Compute the connected components
components = sitk.ConnectedComponent(image_comp, table_masked_img_255, True)

## Fill the holes inside the components
fill_holes = sitk.BinaryFillhole(components)

## Mask the image to get only the body
masked_img = sitk.Mask(img_slice, maskImage=fill_holes)


###### Detection of the holes inside the body #######
## Image threshold
thresh = sitk.MaximumEntropyThreshold(masked_img)

## Morphological operations (Opening + Closing)
clean_thresh = sitk.BinaryOpeningByReconstruction(thresh, [1, 1, 1])
clean_thresh = sitk.BinaryClosingByReconstruction(clean_thresh, [1, 1, 1])

## Compute the gradients (holes have a high gradient)
feature_img = sitk.GradientMagnitudeRecursiveGaussian(clean_thresh, sigma=0.8)

## Level set segmentation based on shape - Needs a distance map
distance = sitk.IsoContourDistance(clean_thresh, levelSetValue=0, farValue=25)
shape_detector = sitk.ShapeDetectionLevelSet(distance, feature_img, curvatureScaling=5, numberOfIterations=2000)
shape_thresh = shape_detector>0

comp = sitk.ConnectedComponent(shape_thresh, True)

## Fill the holes inside the components
filled = sitk.GrayscaleFillhole(comp)

filter = sitk.LabelShapeStatisticsImageFilter()
filter.Execute(filled)

# relabelMap = {}

relabelMap =  {i : 0 for i in filter.GetLabels() if filter.GetRoundness(i) < 0.9}

print(relabelMap)

output = sitk.ChangeLabel(filled, changeMap=relabelMap)

## visualize the results
fig, ax = plt.subplots(1, 2)
ax[0].imshow(sitk.GetArrayFromImage(masked_img), cmap='gray')
ax[0].contour(sitk.GetArrayFromImage(fill_holes), colors='green')
ax[1].imshow(sitk.GetArrayFromImage(masked_img), cmap='gray')
ax[1].contour(sitk.GetArrayFromImage(fill_holes), colors='green')
ax[1].contour(sitk.GetArrayFromImage(sitk.Subtract(output, fill_holes)), colors='red')
plt.tight_layout()
plt.show()