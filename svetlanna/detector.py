import torch
from torch import nn
from svetlanna import SimulationParameters
from svetlanna.elements import Element
from svetlanna.wavefront import Wavefront


class Detector(Element):
    """
    Object that plays a role of a physical detector in an optical system:
        (1) func='intensity'
            transforms incident field to intensities for further image analysis
        (2) ...
    """
    def __init__(
            self,
            simulation_parameters: SimulationParameters,
            func='intensity'
    ):
        """
        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters for a further optical network.
        func : str
            A parameter that defines a function that will be applied to an incident field
            to obtain a detector image.
            (1) func='intensity' – detector returns intensities
            (2) ...
        """
        super().__init__(simulation_parameters)
        # TODO: add some normalization for the output tensor of intensities? or not?
        self.func = func

    def forward(self, input_field: Wavefront) -> torch.Tensor:
        """
        Method that returns the image obtained from the incident field by a detector
        using self.func.
        in the simplest case the image on a detector is an intensities image)
        ...

        Parameters
        ----------
        input_field : Wavefront
            A tensor (Wavefront) of an incident field on a detector.

        Returns
        -------
        detector_output : torch.Tensor
            The image on a detector (according to self.func).
        """
        detector_output = None

        if self.func == 'intensity':
            # TODO: add some normalization for intensities? what is with units?
            detector_output = torch.Tensor(
                input_field.abs().pow(2)
            )  # field absolute values squared

        return detector_output


class DetectorProcessorClf(nn.Module):
    """
    The necessary layer to solve a classification task. Must be placed after a detector.
    This layer process an image from the detector and calculates probabilities of belonging to classes.
    """
    def __init__(
            self,
            num_classes: int,
            simulation_parameters: SimulationParameters,
            segmented_detector: torch.Tensor | None = None,
            segments_weights: torch.Tensor | None = None,
            segments_zone_size: torch.Size | None = None,
            segmentation_type: str = 'strips',
            device: str | torch.device = torch.get_default_device(),
    ):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes in a classification task.
        simulation_parameters : SimulationParameters
            Simulation parameters for a further optical network.
        segmented_detector : torch.Tensor | None
            A tensor of the same shape as detector, where
            each pixel in the mask is marked by a class number from 0 to self.num_classes
        segments_weights : torch.Tensor | None
            Weights for each class segment. The factor by which the detector integral over the zone is multiplied.
        segments_zone_size : torch.Size | None
            A size of a zone (square in a middle of a detector), where segments will be placed.
            If None - match the simulation parameters.
        segmentation_type : str
            If `segmented_detector` is not defined, that parameter defines one of the methods to markup detector:
            1) 'strips' – vertical stripes zones symmetrically arranged relative to the detector center
            2) ...
        device : str | torch.device
            Device, where network training will be conducted.
        """
        super().__init__()
        self.num_classes = num_classes
        self.simulation_parameters = simulation_parameters  # only to get sizes - devices mustn't match

        self.__device = device

        self.segments_zone_size = segments_zone_size
        self.segmentation_type = segmentation_type

        if segmented_detector is not None:  # if a detector segmentation is not defined
            self.segmented_detector = segmented_detector.int()  # markup of a detector by classes zones
        else:  # detector is not segmented
            self.segmentation_type = segmentation_type
            if segments_zone_size is None:
                sim_params_size = self.simulation_parameters.axes_size(axs=('H', 'W'))  # [H, W]
                # make a detector segmentation according to self.segmentation_type
                self.segmented_detector = self.detector_segmentation(sim_params_size)
            else:
                self.segmented_detector = self.detector_segmentation(
                    torch.Size(self.segments_zone_size)
                )

        # TODO: extend self.segmented_detector size if it is not match with SimulationParameters?

        # calculate weights for segments
        if segments_weights is not None:
            self.segments_weights = segments_weights
        else:
            # TODO: weights could be custom?
            self.segments_weights = self.weight_segments()

        # move tensors to device
        self.segmented_detector = self.segmented_detector.to(self.__device)
        self.segments_weights = self.segments_weights.to(self.__device)

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

        # add padding to match simulation parameters Wavefront shape
        sim_params_size = self.simulation_parameters.axes_size(axs=('H', 'W'))
        if not sim_params_size == detector_shape:
            y_nodes, x_nodes = sim_params_size  # goal size
            y_mask, x_mask = detector_shape  # current size
            # params for padding
            pad_top = int((y_nodes - y_mask) / 2)
            pad_bottom = y_nodes - pad_top - y_mask
            pad_left = int((x_nodes - x_mask) / 2)
            pad_right = x_nodes - pad_left - x_mask
            # add padding of -1
            detector_markup = nn.functional.pad(
                input=detector_markup,
                pad=(pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=-1
            )

        # if the detector size matches with sim params
        assert detector_markup.size() == sim_params_size

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
        # TODO: solve the problem with dimensions...
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
        integrals_by_classes = torch.zeros(size=(1, self.num_classes))
        # TODO: what to do with multiple wavelengths?
        for ind_class in range(self.num_classes):
            # `mask_class` will be on the same device as `self.segmented_detector`!
            mask_class = torch.where(ind_class == self.segmented_detector, 1, 0)
            integrals_by_classes[0, ind_class] = (
                    detector_data * mask_class
            ).sum().item()

        integrals_by_classes = integrals_by_classes * self.segments_weights
        # TODO: maybe some function like SoftMax? but integrals can be large!
        return integrals_by_classes / integrals_by_classes.sum().item()

    def batch_zone_integral(self, batch_detector_data: torch.Tensor, ind_class: int) -> torch.Tensor:
        """
        Returns an integral (sum) of a detector data over a selected zone (`ind_class`).
        ...

        Parameters
        ----------
        batch_detector_data : torch.Tensor
            A batch of images from a detector.
        ind_class : int
            Index of a class.

        Returns
        -------
        torch.Tensor
            Sum of intensities over the selected zone for a batch.
            Sze of a tensor = [batch_size]
        """
        mask_class = torch.where(ind_class == self.segmented_detector, 1, 0)
        # sum by two last dimensions!
        class_integral = (batch_detector_data * mask_class).sum(dim=(-2, -1))
        # Comment: class_integral.size() = [batch_size, dim_0, dim_1...] (all dimensions except 'W' and 'H')

        if len(class_integral.size()) > 1:
            # TODO: how to process other dimensions? user must define by himself?
            return class_integral.sum(
                dim=tuple(
                    range(1, len(class_integral.size()))  # all dimensions except batch_size dimension
                )
            )  # return.size() = [batch_size]
        else:  # no other dimensions except ['W', 'H'] for each item in the batch
            return class_integral

    def batch_forward(self, batch_detector_data: torch.Tensor) -> torch.Tensor:
        """
        Calculates probabilities of belonging to classes for a batch of detector images.
        ...

        Parameters
        ----------
        batch_detector_data : torch.Tensor
            A batch of images from a detector.
            shape=(batch_size, ... 'H', 'W').

        Returns
        -------
        torch.Tensor
            A tensor of probabilities of element belonging to classes for further calculation of loss.
            shape=(batch_size, self.num_classes)
        """
        # TODO: make `.forward()` universal for a batch and for a single wavefront!
        # TODO: use simulation parameters to understand if there is a batch dimension?
        # if not batch_detector_data.device == self.__device:
        #     # TODO: it does not work!
        #     raise ValueError('A data batch and DetectorProcessorClf must be on the same device!')

        batch_size = batch_detector_data.size()[0]  # batch size is a 0'th dimension!

        integrals_by_classes = torch.zeros(size=(batch_size, self.num_classes)).to(self.__device)
        for ind_class in range(self.num_classes):
            integrals_by_classes[:, ind_class] = (
                    self.batch_zone_integral(batch_detector_data, ind_class) *
                    self.segments_weights[0, ind_class]
            )

        return integrals_by_classes / torch.unsqueeze(integrals_by_classes.sum(dim=1), 1)

    def to(self, device: str | torch.device | int) -> 'DetectorProcessorClf':
        if self.__device == torch.device(device):
            return self

        return DetectorProcessorClf(
            num_classes=self.num_classes,
            simulation_parameters=self.simulation_parameters,
            segmented_detector=self.segmented_detector,
            device=device
        )

    @property
    def device(self) -> str | torch.device | int:
        return self.__device
