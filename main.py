import SimpleITK as sitk
from matplotlib import colors
import matplotlib.pyplot as plt
from tools import CTBodyDetector, RandoHolesDetector, RandoHolesDetectorCV

ct_folder_path = r"C:\Users\Mendes\Desktop\18479762"

## Read folder with series
ct_reader = sitk.ImageSeriesReader()
ct_file_names = ct_reader.GetGDCMSeriesFileNames(ct_folder_path)
ct_reader.SetFileNames(ct_file_names)
ct_reader.MetaDataDictionaryArrayUpdateOn()
ct_volume = ct_reader.Execute()

imgs_list_idx = [120, 64, 39, 14, -11, -36, -61, -86, -111]

slab_z_index = imgs_list_idx[0]

## Select image with z-index
for i in range(0, ct_volume.GetSize()[2]):
    if int(float(ct_reader.GetMetaData(i, '0020|1041'))) == slab_z_index:
        img_idx = i

## Get info
window_center = int(ct_reader.GetMetaData(img_idx, '0028|1050'))
window_width = int(ct_reader.GetMetaData(img_idx, '0028|1051'))
table_height = int(float(ct_reader.GetMetaData(img_idx, '0018|1130').strip()))

img_slice = ct_volume[:,:,img_idx]
## Intensity Window for viewing purposes
img_slice_view = sitk.IntensityWindowing(img_slice, window_center - window_width, window_center + window_width)

# CT Body Detection
body_detector = CTBodyDetector()
body_detector.setTableHeight(table_height)
body_cnt = body_detector.execute(img_slice)
masked_image = body_detector.getBodyMask()

# CT Holes Detector
holes_detector = RandoHolesDetector()
holes = holes_detector.execute(masked_image)

# holes_detector = RandoHolesDetectorCV()
# holes = holes_detector.execute(masked_image)

## visualize the results
fig, ax = plt.subplots(1, 1)
ax.imshow(sitk.GetArrayFromImage(img_slice_view), cmap='gray')
ax.contour(sitk.GetArrayFromImage(body_cnt), colors='green')
ax.contour(sitk.GetArrayFromImage(holes), colors='red')
#ax.imshow(sitk.GetArrayFromImage(holes), cmap='gray')
plt.tight_layout()
plt.show()
