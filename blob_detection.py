import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2

ct_folder_path = r"C:\Users\Mendes\Desktop\ProjectoDistorcaoMRI\CT_1_vazio"

## Read folder with series
ct_reader = sitk.ImageSeriesReader()
ct_file_names = ct_reader.GetGDCMSeriesFileNames(ct_folder_path)
ct_reader.SetFileNames(ct_file_names)
ct_reader.MetaDataDictionaryArrayUpdateOn()
ct_volume = ct_reader.Execute()

slab_z_index = 64

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

## Binary Threshold
img_thresh = sitk.BinaryThreshold(img_slice, lowerThreshold=-875.6, upperThreshold=-571.6, insideValue=1, outsideValue=0)

## Remove the table from image (if hardtop)
table_height = ct_reader.GetMetaData(img_idx, '0018|1130')
table_thickness = 72.3
table_cut = int(round(img_width/2, 0) + int(table_height) - table_thickness)

## Create a mask for the table
mask = sitk.Image([img_height, img_width], sitk.sitkUInt16, 1)
mask.SetSpacing(pixel_spacing)
mask.SetOrigin(origin)
for i in range(0, img_width):
    for j in range(0, table_cut):
        mask.SetPixel(i, j, 1) 

masked_img = sitk.Mask(img_slice, maskImage=mask, outsideValue=0, maskingValue=0)

## Convert to 8bit image [0, 255]
masked_img_255 = sitk.Cast(sitk.RescaleIntensity(masked_img), sitk.sitkUInt8)
img_255 = sitk.Cast(sitk.RescaleIntensity(img_slice), sitk.sitkUInt8)

## Find the contours for body detection
contours, hierarchy = cv2.findContours(sitk.GetArrayFromImage(masked_img_255), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

## Blob detection - Filtering parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 25
params.maxArea = 100
params.thresholdStep = 1
params.minThreshold = 10
params.maxThreshold = 100
params.filterByCircularity = True
params.minCircularity = 0.65
params.filterByInertia = True
params.minInertiaRatio = 0.65
params.filterByConvexity = True
params.minConvexity = 0.65
params.minDistBetweenBlobs = 8
params.minRepeatability = 1

## Create a blob detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

## Detect blobs
keypoints = detector.detect(sitk.GetArrayFromImage(masked_img_255))

## Blank image to draw the keypoints
blank = sitk.Image([img_height, img_width], sitk.sitkUInt8, 1)

## Draw keypoints on image and display z-index
im_with_keypoints = cv2.drawKeypoints(sitk.GetArrayFromImage(img_255), keypoints, sitk.GetArrayFromImage(blank), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.putText(im_with_keypoints, 'z=' + str(slab_z_index) + 'mm', (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

## Draw body contour (largest)
if len(contours) != 0:
    cnt = max(contours, key = cv2.contourArea)
    cv2.drawContours(im_with_keypoints, [cnt], 0, (0, 255, 0), 1)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(im_with_keypoints,(x, y),(x + w, y + h),(0, 0, 255), 1)

## Display
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

fig, axes = plt.subplots()
axes.imshow(im_with_keypoints, cmap='gray')
plt.tight_layout()
plt.show()