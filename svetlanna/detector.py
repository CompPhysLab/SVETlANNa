import torch
from torch import nn
from svetlanna import SimulationParameters
from .elements import Element


class Detector(Element):
    # TODO: Must an Element be a parent class?
    """
    Object that plays a role of a physical detector in an optical system:
    transforms incident field to intensities for further image analysis
    """
    def __init__(
            self,
            simulation_parameters: SimulationParameters,
            func='intensity'
    ):
        super().__init__(simulation_parameters)
        # TODO: add some normalization for the output tensor of intensities? or not?
        self.func = func

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """
        Method that returns the image obtained from the incident field by a detector
        (in the simplest case the image on a detector is an intensities image)
        ...

        Parameters
        ----------
        input_field : torch.tensor()
            A tensor of an incident field on a detector.

        Returns
        -------
        torch.tensor()
            The field after propagating through the aperture
        """
        detector_output = None
        # TODO: add some normalization for intensities? what is with units?
        if self.func == 'intensity':
            detector_output = input_field.abs().pow(2)  # field absolute values squared
        return detector_output


class DetectorProcessorClf(nn.Module):
    """
    The necessary layer to solve a classification task. Must be placed after a detector.
    This layer process an image from the detector and calculates probabilities of belonging to classes.
    """
    def __init__(self, num_classes: int, segmented_detector=None, segmentation_type='strips'):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes in a classification task.
        segmented_detector : torch.Tensor
            A tensor of the same shape as detector, where
            each pixel in the mask is marked by a class number from 0 to self.num_classes
        segmentation_type : str
            If `segmented_detector` is not defined, that parameter defines one of the methods to markup detector:
            1) 'strips' – vertical stripes zones symmetrically arranged relative to the detector center
            2)
        """
        super().__init__()
        self.num_classes = num_classes
        self.segmented_detector = segmented_detector  # markup of a detector by classes zones
        if segmented_detector is not None:  # if a detector segmentation is not defined
            self.segmented_detector = self.segmented_detector.int()
            # TODO: weights could be custom!
            self.segments_weights = self.weight_segments()
        self.segmentation_type = segmentation_type

    def detector_segmentation(self, detector_shape: torch.Size) -> torch.Tensor:
        """
        Function that markups a detector area by classes zones.
        ...

        Parameters
        ----------
        detector_shape : torch.Size
            Shape of a detector.

        Returns
        -------
        detector_markup : torch.Tensor(dtype=torch.int32)
            A tensor of the same shape as detector, where
            1) each pixel in the mask is marked by a class number from 0 to self.num_classes;
            2) if pixel is marked as -1 it is not belonging to any class during a computation of probabilities;
            3) each class zone can be highlighted as torch.where(detector_markup == ind_class, 1, 0).
        """
        detector_y, detector_x = detector_shape
        detector_markup = (-1) * torch.ones(size=detector_shape, dtype=torch.int32)

        if self.segmentation_type == 'strips':
            # segments are vertical strips, symmetrically arranged relative to the detector center!
            # TODO: gaps between strips? check if possible etc.
            if self.num_classes % 2 == 0:  # even number of classes
                central_class = 0  # no central class, classes are symmetrically arranged
                if detector_x % 2 == 0:  # even number of detector "pixels" in x-direction
                    # Strips: |..111222|333444..|
                    x_center_left_ind = int(detector_x // 2)
                    x_center_right_ind = x_center_left_ind
                    strip_width = int(detector_x // self.num_classes)
                else:  # odd number of detector "pixels" in x-direction
                    # Strips: |.111222|.|333444.|
                    x_center_left_ind = int(detector_x // 2)
                    x_center_right_ind = x_center_left_ind + 1
                    strip_width = int((detector_x - 1) // self.num_classes)

            else:  # odd number of classes
                central_class = 1  # there is a central strip
                strip_width = int(detector_x // self.num_classes)
                if detector_x % 2 == 0:  # even number of detector "pixels" in x-direction
                    if strip_width % 2 == 0:  # can symmetrically arrange a central class strip
                        # Strips: |..111122|223333..|
                        x_center_left_ind = int(detector_x // 2 - strip_width // 2)
                        x_center_right_ind = int(detector_x // 2 + strip_width // 2)
                    else:  # should make a center strip of even width
                        # Strips: |.11122|22333.|
                        center_strip_width = strip_width + 1  # becomes even!
                        x_center_left_ind = int(detector_x // 2 - center_strip_width // 2)
                        x_center_right_ind = int(detector_x // 2 + center_strip_width // 2)
                        # update width for other strips except the center one
                        strip_width = int(x_center_left_ind // (self.num_classes // 2))
                else:  # odd number of detector "pixels" in x-direction
                    if strip_width % 2 == 0:  # should make a center strip of odd width for symmetry
                        # Strips: |11112|2|23333|
                        center_strip_width = strip_width + 1  # becomes odd!
                        x_center_left_ind = int(detector_x // 2 - center_strip_width // 2)
                        x_center_right_ind = int(detector_x // 2 + 1 + center_strip_width // 2)
                        # update width for other strips except the center one
                        strip_width = int(x_center_left_ind // (self.num_classes // 2))
                    else:  # can symmetrically arrange a central class strip
                        # Strips: |1112|2|2333|
                        x_center_left_ind = int(detector_x // 2 - strip_width // 2)
                        x_center_right_ind = int(detector_x // 2 + 1 + strip_width // 2)
                # mask for the central class
                ind_central_class = int(self.num_classes // 2)
                detector_markup[:, x_center_left_ind:x_center_right_ind] = ind_central_class

            # fill masks from the detector center (like apertures for each class)
            # from the center to left
            for ind in range(self.num_classes // 2):  # left half of the detector
                ind_class = int(self.num_classes // 2 - 1 - ind)
                ind_left_border = x_center_left_ind - strip_width * (ind + 1)
                ind_right_border = x_center_left_ind - strip_width * ind
                assert torch.all(-1 == detector_markup[:, ind_left_border:ind_right_border]).item()
                detector_markup[:, ind_left_border:ind_right_border] = ind_class
            # from the center to right
            for ind in range(self.num_classes // 2):  # right half of the detector
                ind_class = int(ind + self.num_classes // 2 + central_class)
                ind_left_border = x_center_right_ind + strip_width * ind
                ind_right_border = x_center_right_ind + strip_width * (ind + 1)
                assert torch.all(-1 == detector_markup[:, ind_left_border:ind_right_border]).item()
                detector_markup[:, ind_left_border:ind_right_border] = ind_class

        return detector_markup

    def weight_segments(self) -> torch.Tensor:
        """
        Calculates weights for segments if segments having different areas.
        Comment: weight_i * area_i = const
        ...

        Returns
        -------
        torch.Tensor
            A tensor of weights for further calculation of integrals.
            shape=(1, self.num_classes)
        """
        classes_areas = torch.zeros(size=(1, self.num_classes))
        for ind_class in range(self.num_classes):
            classes_areas[0, ind_class] = torch.where(ind_class == self.segmented_detector, 1, 0).sum().item()
        min_class_area = classes_areas.min().item()
        return min_class_area / classes_areas

    def forward(self, detector_data: torch.Tensor) -> torch.Tensor:
        """
        Calculates probabilities of belonging to classes by detector image.
        ...

        Parameters
        ----------
        detector_data : torch.Tensor
            A tensor that represents an image on a detector.

        Returns
        -------
        torch.Tensor
            A tensor of probabilities of element belonging to classes for further calculation of loss.
            shape=(1, self.num_classes)
        """
        if self.segmented_detector is None:  # there is no predefined segments of a detector for classes
            # TODO: must we make it in __init__? But we need a detector (detector_data) shape for it!
            self.segmented_detector = self.detector_segmentation(detector_data.shape)
            self.segments_weights = self.weight_segments()

        integrals_by_classes = torch.zeros(size=(1, self.num_classes))
        # TODO: what to do with multiple wavelengths?
        for ind_class in range(self.num_classes):
            mask_class = torch.where(ind_class == self.segmented_detector, 1, 0)
            integrals_by_classes[0, ind_class] = (
                    detector_data * mask_class
            ).sum().item()

        integrals_by_classes = integrals_by_classes * self.segments_weights
        # TODO: maybe some function like SoftMax? but integrals can be large!
        return integrals_by_classes / integrals_by_classes.sum().item()

