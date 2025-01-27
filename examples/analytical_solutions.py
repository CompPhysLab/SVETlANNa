import numpy as np
import scipy as sp
import torch

from scipy.special import jv
from tqdm import tqdm


class RectangleFresnel:
    """A class describing the analytical solution for the problem of free
      propagation after planar wave passes a rectangular aperture
    """
    def __init__(
        self,
        distance: float,
        x_size: float,
        y_size: float,
        x_nodes: int,
        y_nodes: int,
        width: float,
        height: float,
        wavelength: torch.Tensor | float
    ):
        """Constructor method

        Parameters
        ----------
        distance : float
            Distance between the square aperture and the screen
        x_size : float
            System size along the axis ox
        y_size : float
            System size along the axis oy
        x_nodes : int
            Number of computational nodes along the axis ox
        y_nodes : int
            Number of computational nodes along the axis oy
        width : float
            Width of the aperture
        height : float
            Height of the aperture
        wavelength : torch.Tensor | float
            Wavelength of the incident planar wave
        """

        self.distance = distance
        self.x_size = x_size
        self.y_size = y_size
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.width = width
        self.height = height
        self.wavelength = wavelength

    def field(self) -> np.ndarray:
        """Method that describes the intensity profile on the screen

        Returns
        -------
        np.ndarray
            2d Intensity profile
        """
        x_linear = np.linspace(-self.x_size / 2, self.x_size / 2, self.x_nodes)
        y_linear = np.linspace(-self.y_size / 2, self.y_size / 2, self.y_nodes)
        x_grid, y_grid = np.meshgrid(x_linear, y_linear)

        if type(self.wavelength) is float:
            wave_number = 2 * np.pi / self.wavelength
        else:
            wave_number = 2 * np.pi / self.wavelength[..., None, None]
            x_grid = x_grid[None, :]
            y_grid = y_grid[None, :]

        psi1 = -np.sqrt(wave_number/(np.pi*self.distance))*(self.width/2
                                                            + x_grid)
        psi2 = np.sqrt(wave_number/(np.pi*self.distance))*(self.width / 2
                                                           - x_grid)
        eta1 = -np.sqrt(wave_number/(np.pi*self.distance))*(self.height / 2
                                                            + y_grid)
        eta2 = np.sqrt(wave_number/(np.pi*self.distance))*(self.height / 2
                                                           - y_grid)

        s_psi1, c_psi1 = sp.special.fresnel(psi1)
        s_psi2, c_psi2 = sp.special.fresnel(psi2)
        s_eta1, c_eta1 = sp.special.fresnel(eta1)
        s_eta2, c_eta2 = sp.special.fresnel(eta2)

        field = np.exp(1j * wave_number * self.distance) * (1 / 2j) * (
            (c_psi2 - c_psi1) + 1j * (s_psi2 - s_psi1)
        ) * (
            (c_eta2 - c_eta1) + 1j * (s_eta2 - s_eta1)
        )

        # intensity = (1/4)*(np.power((c_psi2 - c_psi1), 2) +
        #                    np.power((s_psi2 - s_psi1), 2))*(
        #                        np.power((c_eta2 - c_eta1), 2) + np.power(
        #                            (s_eta2 - s_eta1), 2))

        return field

    def intensity(self) -> np.ndarray:
        return np.abs(self.field()) ** 2


class CircleFresnel:
    """A class describing the analytical solution for the problem of free
      propagation after planar wave passes a circular aperture aperture
    """
    def __init__(
        self,
        distance: float,
        x_size: float,
        y_size: float,
        x_nodes: int,
        y_nodes: int,
        radius: float,
        wavelength: torch.Tensor | float,
        summation_number: int = 50
    ):

        self.distance = distance
        self.x_size = x_size
        self.y_size = y_size
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.radius = radius
        self.summation_number = summation_number
        self.wavelength = wavelength

    def field(self) -> np.ndarray:

        x_linear = np.linspace(-self.x_size / 2, self.x_size / 2, self.x_nodes)
        y_linear = np.linspace(-self.y_size / 2, self.y_size / 2, self.y_nodes)
        x_grid, y_grid = np.meshgrid(x_linear, y_linear)

        if type(self.wavelength) is float:
            wave_number = 2 * np.pi / self.wavelength
        else:
            wave_number = 2 * np.pi / self.wavelength[..., None, None]
            x_grid = x_grid[None, :]
            y_grid = y_grid[None, :]

        radius = np.sqrt(x_grid**2 + y_grid**2)

        series = np.zeros_like(x_grid, dtype=np.complex128)

        for n in tqdm(range(self.summation_number)):
            series += ((
                -1j * radius / (self.radius)
            ) ** n) * jv(
                n, 2 * np.pi * self.radius * radius / (self.wavelength * self.distance)  # noqa: E501
            )

        self.field = np.exp(1j * wave_number * self.distance) * (
            1 - np.exp(
                1j * np.pi * radius**2 / (self.wavelength * self.distance)
            ) * np.exp(
                1j * np.pi * self.radius**2 / (self.wavelength * self.distance)
            ) * series
        )

        return self.field

    def intensity(self) -> np.ndarray:
        x_linear = np.linspace(-self.x_size / 2, self.x_size / 2, self.x_nodes)
        y_linear = np.linspace(-self.y_size / 2, self.y_size / 2, self.y_nodes)
        x_grid, y_grid = np.meshgrid(x_linear, y_linear)

        if type(self.wavelength) is float:
            wave_number = 2 * np.pi / self.wavelength
        else:
            wave_number = 2 * np.pi / self.wavelength[..., None, None]
            x_grid = x_grid[None, :]
            y_grid = y_grid[None, :]

        radius = np.sqrt(x_grid**2 + y_grid**2)

        intensity = 1 / (1 + np.exp((radius / self.radius)**2))**2 * (
            1 + jv(0, 2 * np.pi * self.radius * radius / (self.wavelength * self.distance))**2 - 2*np.cos(
                np.pi * self.radius**2/(self.wavelength * self.distance) + np.pi * radius**2 / (self.distance*self.wavelength)
            ) * jv(0, 2 * np.pi * self.radius * radius / (self.wavelength * self.distance))
        )
        return intensity
