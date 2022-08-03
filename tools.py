import SimpleITK as sitk
import cv2


class CTBodyDetector:
    """ 
    Body detector for CT images. 
    --------------
    The detection is based on morphological operations (erosion + dilation + fill holes).
    To remove the bed, table heigh must be specified before the execute method.
    It returns a sitk image with the same size and height as the original.

    Steps:
    -------------
        * Binary Threshold image > 0
        * Dilation + Erosion 
        * Binary fill hole
        * Get connected components
        * Compute Gradient Magnitude Gaussian
        * Filter labels by kurtosis
    """
    def __init__(self)  -> None:
        self.image = None
        self.fill_holes = None
        self.table_height = None
        self.table_thickness = 79
        self.table_cut = None
        self.cnt = None

    def setTableHeight(self, table_height) -> None:
        self.table_height = table_height

    def getTableMask(self) -> sitk.Image:
        ## Calculate how much we need to cut the image to remove the table
        self.table_cut = int(round(self.image.GetWidth()/2, 0) + int(self.table_height) - self.table_thickness)

        ## Create a mask for the table
        mask = sitk.Image([self.image.GetHeight(), self.image.GetWidth()], sitk.sitkUInt16, 1)
        mask.SetSpacing(self.image.GetSpacing())
        mask.SetOrigin(self.image.GetOrigin())
        for i in range(0, self.image.GetWidth()):
            for j in range(0, self.table_cut):
                mask.SetPixel(i, j, 1) 

        return sitk.Mask(self.image, maskImage=mask, outsideValue=0, maskingValue=0)

    def execute(self, image: sitk.Image) -> sitk.Image:
        self.image = image
        
        masked_image = self.getTableMask()

        ## Create Binary image: Positive Pixel values
        img_slice_thresh = masked_image > 0

        ## Erosion and dilation
        image_comp = sitk.BinaryErode(sitk.BinaryDilate(img_slice_thresh, [3, 1]), [3, 1])

        ## Fill the holes inside
        filled = sitk.BinaryFillhole(image_comp)

        ## Compute the connected components
        self.cnt = sitk.ConnectedComponent(filled, True)

        return self.cnt

    def getBodyMask(self) -> sitk.Image:
        return sitk.Mask(self.image, maskImage=self.cnt)


class RandoHolesDetector:
    def __init__(self) -> None:
        self.image = None

    def execute(self, image: sitk.Image) -> sitk.Image:
        self.image = image

        inverted = sitk.InvertIntensity(self.image)

        ## Image threshold
        #thresh = inverted > 800
        thresh = sitk.OtsuMultipleThresholds(self.image, 3, valleyEmphasis=True, numberOfHistogramBins=256)

        comp = sitk.ConnectedComponent(thresh, True)

        filter = sitk.LabelShapeStatisticsImageFilter()
        filter.Execute(comp)

        relabelMap =  {i : 0 for i in filter.GetLabels() if filter.GetRoundness(i) < 0.86 if filter.GetElongation(i) > 0.9}

        output = sitk.ChangeLabel(comp, changeMap=relabelMap)
        return output

class RandoHolesDetectorCV:
    def __init__(self) -> None:
        self.image = None
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.minArea = 25
        self.params.maxArea = 100
        self.params.thresholdStep = 1
        self.params.minThreshold = 10
        self.params.maxThreshold = 100
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.65
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.65
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.65
        self.params.minDistBetweenBlobs = 8
        self.params.minRepeatability = 1
        ## Create a blob detector with the parameters
        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def execute(self, image: sitk.Image) -> sitk.Image:
        self.image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)
        ## Detect blobs
        keypoints = self.detector.detect(sitk.GetArrayFromImage(self.image))

        ## Blank image to draw the keypoints
        blank = sitk.Image([self.image.GetHeight(), self.image.GetWidth()], sitk.sitkUInt8, 1)

        ## Draw keypoints on image and display z-index
        im_with_keypoints = cv2.drawKeypoints(sitk.GetArrayFromImage(blank), keypoints, sitk.GetArrayFromImage(blank), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return sitk.GetImageFromArray(im_with_keypoints)