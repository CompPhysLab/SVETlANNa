import warnings

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as functional

from svetlanna import SimulationParameters
from svetlanna import Wavefront
from svetlanna import elements


class DatasetOfWavefronts(Dataset):

    def __init__(
            self,
            init_ds: Dataset,
            transformations: transforms.Compose,
            sim_params: SimulationParameters,
    ):
        """
        Parameters
        ----------
        init_ds : torch.utils.data.Dataset
            An initial dataset (of images and labels).
        transformations : transforms.Compose
            A sequence of transforms that will be applied to dataset elements (images).
        sim_params : SimulationParameters
            Simulation parameters for a further optical network.
        """
        self.init_ds = init_ds
        self.transformations = transformations

        self.sim_params = sim_params  # to check if all transforms results in right shape
        self.check_transformations()  # print warnings if necessary

    def check_transformations(self):
        """
        Checks if transformations transforms an image to a right-shaped Wavefront.
        """
        random_image = functional.to_pil_image(torch.rand(size=(5, 5)))  # random image
        wavefront = self.transformations(random_image)

        # check type
        if not isinstance(wavefront, Wavefront):
            warnings.warn(
                message='An output object is not of the Wavefront type!'
            )

        # compare nodes number of the resulted Wavefront (last two dimensions) with simulation parameters
        sim_nodes_shape = torch.Size([self.sim_params.y_nodes, self.sim_params.x_nodes])  # [W, H]
        if not wavefront.size()[-2:] == sim_nodes_shape:
            warnings.warn(
                message='A shape of a resulted Wavefront does not match with SimulationParameters!'
            )

    def __len__(self):
        return len(self.init_ds)

    def __getitem__(self, ind: int) -> tuple:
        """
        Parameters
        ----------
        ind : int
            Index of element to return.

        Returns
        -------
        tuple
            An element of dataset: tuple(Wavefront, class)
            A size of a wavefront must be in a correspondence with simulation parameters!
        """
        raw_image, label = self.init_ds[ind]
        # apply transformations
        wavefront_image = self.transformations(raw_image)

        return wavefront_image, label


class WavefrontsDatasetSimple(Dataset):
    """
    Dataset of wavefronts for a classification task for an optical network.
        Each raw image is encoded in the amplitude and/or phase.
    """

    def __init__(
            self,
            images_ds: Dataset,
            image_transforms_comp: transforms.Compose,
            sim_params: SimulationParameters,
    ):
        """
        Parameters
        ----------
        images_ds : torch.utils.data.Dataset
            A dataset of raw images and classes labels.
        image_transforms_comp : transforms.Compose
            A sequence of transforms that will be applied to an image before its convertation to an SLM mask.
        sim_params : SimulationParameters
            Simulation parameters for a further optical network.
        """
        # TODO: add a parameter for choosing what to use to encode an image: use only amplitude/phase or both?
        self.images_ds = images_ds
        self.image_transforms_comp = image_transforms_comp

        self.sim_params = sim_params

    def __len__(self):
        return len(self.images_ds)

    def __getitem__(self, ind: int) -> tuple:
        """
        Parameters
        ----------
        ind : int
            Index of element to return.

        Returns
        -------
        tuple
            An element of dataset: tuple(wavefront tensor, class)
            A size of a wavefront is in correspondence with simulation parameters!
        """
        raw_image, label = self.images_ds[ind]
        # apply transforms
        transformed_image = self.image_transforms_comp(raw_image)
        transformed_image_size = transformed_image.size()[-2:]  # [H, W]

        # we need to resize an image to match simulation parameters (layers dimensions)
        y_nodes, x_nodes = self.sim_params.y_nodes, self.sim_params.x_nodes
        # check (last two dimensions) if transformations result in a proper size
        if not transformed_image_size == torch.Size([y_nodes, x_nodes]):
            # add padding if transformed_image is not match with sim_params!
            pad_top = int((y_nodes - transformed_image_size[0]) / 2)
            pad_bottom = y_nodes - pad_top - transformed_image_size[0]
            pad_left = int((x_nodes - transformed_image_size[1]) / 2)
            pad_right = x_nodes - pad_left - transformed_image_size[1]

            padding = transforms.Pad(
                padding=(pad_left, pad_top, pad_right, pad_bottom),  # [left, top, right, bottom]
                fill=0,
            )
            transformed_image = padding(transformed_image)

        # secondly, we must create a wavefront based on the image
        max_val = transformed_image.max()
        min_val = transformed_image.min()
        normalized_image = (transformed_image - min_val) / (max_val - min_val)  # values from 0 to 1

        # TODO: use only amplitude/phase or both?
        phases = normalized_image * torch.pi
        amplitudes = normalized_image

        wavefront_image = Wavefront(amplitudes * torch.exp(1j * phases))

        return wavefront_image, label


class WavefrontsDatasetWithSLM(Dataset):
    """
    Dataset of wavefronts for a classification task for an optical network.
        Each raw image is used as a mask for SLM, that illuminated by a some beam field.
        A resulted wavefront will be an input tensor for an optical network.
    """

    def __init__(
            self,
            images_ds: Dataset,
            image_transforms_comp: transforms.Compose,
            sim_params: SimulationParameters,
            beam_field: torch.Tensor,
            system_before_slm: list,
            slm_levels: int = 256
    ):
        """
        Parameters
        ----------
        images_ds : torch.utils.data.Dataset
            A dataset of raw images and classes labels.
        image_transforms_comp : transforms.Compose
            A sequence of transforms that will be applied to an image before its convertation to an SLM mask.
        sim_params : SimulationParameters
            Simulation parameters for a further optical network.
        beam_field : torch.Tensor
            A field of a beam (result of Beam.forward) that is used for an images wavefronts generation.
        system_before_slm : list(Element)
            A list of Elements between a beam and an SLM. The beam field is going through them before the SLM.
        slm_levels : int
            Number of phase quantization levels for the SLM, by default 256
        """
        self.images_ds = images_ds
        self.image_transforms_comp = image_transforms_comp

        self.sim_params = sim_params

        self.beam_field = beam_field
        # TODO: maybe we can extract simulation parameters from every element?
        self.system_before_slm = system_before_slm

        self.slm_levels = slm_levels

    def __len__(self):
        return len(self.images_ds)

    def __getitem__(self, ind: int) -> tuple:
        """
        Parameters
        ----------
        ind : int
            Index of element to return.

        Returns
        -------
        tuple
            An element of dataset: tuple(wavefront tensor, class)
            A size of a wavefront is in correspondence with simulation parameters!
        """
        raw_image, label = self.images_ds[ind]
        # apply transforms
        transformed_image = self.image_transforms_comp(raw_image)
        transformed_image_size = transformed_image.size()[-2:]  # [H, W]

        # we need to resize an image to match simulation parameters (layers dimensions)
        y_nodes, x_nodes = self.sim_params.y_nodes, self.sim_params.x_nodes
        if not transformed_image_size == torch.Size([y_nodes, x_nodes]):
            # check (last two dimensions) if we already resized an image by applying self.image_transforms_comp
            resize = transforms.Resize(
                size=(y_nodes, x_nodes),
                interpolation=InterpolationMode.NEAREST,  # <- interpolation function?
            )  # by default applies to last two dimensions!
            transformed_image = resize(transformed_image)

        # secondly, we must somehow transform an image to a wavefront
        output_field = self.beam_field
        for element in self.system_before_slm:
            output_field = element.forward(input_field=output_field)

        # use an image as a mask for SLM
        # TODO: make it possible to use a mask of any values (add normalization by levels within an SLM)
        mask = (transformed_image * (self.slm_levels - 1)).to(torch.int32)
        image_based_slm = elements.SpatialLightModulator(
            simulation_parameters=self.sim_params,
            mask=mask,
            number_of_levels=self.slm_levels
        )

        wavefront_image = image_based_slm.forward(output_field)

        return wavefront_image, label
