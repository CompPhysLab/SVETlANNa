import torch
from .simulation_parameters import SimulationParameters
from typing import Any, Self


class NumericalMesh:
    """Stores values related to numerical mesh"""
    def __init__(self, simulation_parameters: SimulationParameters):
        self.x_linspace = torch.linspace(
            start=-simulation_parameters.x_size/2,
            end=simulation_parameters.x_size/2,
            steps=simulation_parameters.x_nodes
        )
        self.y_linspace = torch.linspace(
            start=-simulation_parameters.y_size/2,
            end=simulation_parameters.y_size/2,
            steps=simulation_parameters.y_nodes
        )
        self.x_grid, self.y_grid = torch.meshgrid(
            self.x_linspace, self.y_linspace, indexing='xy'
        )


class Wavefront(torch.Tensor):
    """Class that represents wavefront"""
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        # see https://github.com/albanD/subclass_zoo/blob/ec47458346c2a1cfcd5e676926a4bbc6709ff62e/base_tensor.py   # noqa: E501
        data = torch.as_tensor(data)
        return super(cls, Wavefront).__new__(cls, data)

    @property
    def intensity(self) -> torch.Tensor:
        """Calculates intensity of the wavefront

        Returns
        -------
        torch.Tensor
            intensity
        """
        return torch.abs(torch.Tensor(self)) ** 2

    @property
    def phase(self) -> torch.Tensor:
        """Calculates phase of the wavefront

        Returns
        -------
        torch.Tensor
            phase from $0$ to $2\\pi$
        """
        res = torch.angle(torch.Tensor(self))
        res[res < 0] += 2 * torch.pi
        return res

    @classmethod
    def plane_wave(
        cls,
        simulation_parameters: SimulationParameters,
        distance: float = 0.,
        wave_direction: Any = None,
        initial_phase: float = 0.
    ) -> Self:
        """Generate wavefront of the plane wave

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            simulation parameters
        distance : float, optional
            free wave propagation distance, by default 0.
        wave_direction : Any, optional
            three component tensor-like vector with x,y,z coordinates.
            The resulting field propagates along the vector, by default
            the wave propagates along z direction.
        initial_phase : float, optional
            additional phase to the resulting field, by default 0.

        Returns
        -------
        Wavefront
            plane wave field.
        """
        # by default the wave propagates along z direction
        if wave_direction is None:
            wave_direction = [0., 0., 1.]

        numerical_mesh = NumericalMesh(
            simulation_parameters=simulation_parameters
        )

        wave_direction = torch.tensor(wave_direction, dtype=torch.float32)
        if wave_direction.shape != torch.Size([3]):
            raise ValueError("wave_direction should contain exactly three components")  # noqa: E501
        wave_direction = wave_direction / torch.norm(wave_direction)

        wave_number = 2 * torch.pi / simulation_parameters.wavelength

        field = cls(torch.exp(
            1j * (wave_direction[0] * wave_number * numerical_mesh.x_grid +
                  wave_direction[1] * wave_number * numerical_mesh.y_grid +
                  wave_direction[2] * wave_number * distance +
                  initial_phase)
        ))

        return field

    @classmethod
    def gaussian_beam(
        cls,
        simulation_parameters: SimulationParameters,
        waist_radius: float,
        distance: float = 0.,
    ) -> Self:
        """Generates the Gaussian beam.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            simulation parameters
        waist_radius : float
            Waist radius of the beam
        distance : float, optional
            free wave propagation distance, by default 0.

        Returns
        -------
        Wavefront
            Beam field in the plane oXY propagated over the distance
        """

        numerical_mesh = NumericalMesh(
            simulation_parameters=simulation_parameters
        )

        wave_number = 2 * torch.pi / simulation_parameters.wavelength

        rayleigh_range = torch.pi * (waist_radius**2) / simulation_parameters.wavelength    # noqa: E501

        radial_distance_squared = numerical_mesh.x_grid**2 + numerical_mesh.y_grid**2   # noqa: E501

        hyperbolic_relation = waist_radius * (1 + (distance / rayleigh_range)**2)**(1/2)    # noqa: E501

        inverse_radius_of_curvature = distance / (distance**2 + rayleigh_range**2)  # noqa: E501

        # Gouy phase
        gouy_phase = torch.arctan(torch.tensor(distance / rayleigh_range))

        field = torch.exp(-1j * wave_number * radial_distance_squared * inverse_radius_of_curvature / 2)    # noqa: E501
        field *= torch.exp(torch.tensor(-1j * wave_number * distance))
        field *= torch.exp(1j * gouy_phase)
        field *= torch.exp(-radial_distance_squared / (hyperbolic_relation)**2)
        field *= waist_radius / hyperbolic_relation

        field = cls(field)

        return field

    @classmethod
    def spherical_wave(
        cls,
        simulation_parameters: SimulationParameters,
        distance: float,
        initial_phase: float = 0.
    ) -> Self:
        """Generate wavefront of the spherical wave

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            simulation parameters
        distance : float
            distance between the source and the oXY plane.
        initial_phase : float, optional
            additional phase to the resulting field, by default 0.

        Returns
        -------
        Wavefront
            Beam field
        """
        numerical_mesh = NumericalMesh(
            simulation_parameters=simulation_parameters
        )

        wave_number = 2 * torch.pi / simulation_parameters.wavelength

        radius = torch.sqrt(
            numerical_mesh.x_grid**2 + numerical_mesh.y_grid**2 + distance**2  # noqa: E501
        )

        field = 1 / radius * torch.exp(
            1j * (wave_number * radius + initial_phase)
        )

        field = cls(field)

        return field
