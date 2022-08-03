import SimpleITK as sitk


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

        ## Fill the holes inside the components
        filled = sitk.BinaryFillhole(image_comp)

        ## Compute the connected components
        body_cnt = sitk.ConnectedComponent(filled, True)

        return body_cnt

    def getBodyMask(self) -> sitk.Image:
        return sitk.Mask(self.image, maskImage=self.components)


class RandoHolesDetector:
    def __init__(self) -> None:
        self.image = None

    def execute(self, image: sitk.Image) -> sitk.Image:
        self.image = image

        ## Image threshold
        thresh = sitk.MaximumEntropyThreshold(self.image)

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

        relabelMap =  {i : 0 for i in filter.GetLabels() if filter.GetRoundness(i) < 0.9}

        output = sitk.ChangeLabel(filled, changeMap=relabelMap)
        return output