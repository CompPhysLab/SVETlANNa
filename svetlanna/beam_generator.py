import torch
from .simulation_parameters import SimulationParameters


class Beam:

    """A class describing light beams
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        amplitude: float = 1.
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        amplitude : float, optional
            Amplitude of the field from the source, by default 1.
        """

        self.simulation_parameters = simulation_parameters

        self._x_size = simulation_parameters.x_size
        self._y_size = simulation_parameters.y_size
        self._x_nodes = simulation_parameters.x_nodes
        self._y_nodes = simulation_parameters.y_nodes
        self._wavelength = simulation_parameters.wavelength

        self.amplitude = amplitude

        self._x_linspace = torch.linspace(
            -self._x_size/2, self._x_size/2, self._x_nodes
        )
        self._y_linspace = torch.linspace(
            -self._y_size/2, self._y_size/2, self._y_nodes
        )
        self._x_grid, self._y_grid = torch.meshgrid(
            self._x_linspace, self._y_linspace, indexing='xy'
        )


# TODO: check docstring
class GaussianBeam(Beam):

    """A class that describes the Gaussian Beam

    Returns
    -------
    _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        amplitude: float = 1
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        amplitude : float, optional
            Amplitude of the field from the source, by default 1
        """

        super().__init__(simulation_parameters, amplitude)

    def forward(
        self,
        distance: float,
        waist_radius: float,
        refractive_index: float = 1.
    ) -> torch.Tensor:
        """Method that describes propagation of the Gaussian beam

        Parameters
        ----------
        distance : float
            Field propagation distance
        waist_radius : float
            Waist radius of the beam
        refractive_index : float, optional
            Refractive index of the medium, by default 1.

        Returns
        -------
        torch.Tensor
            Beam field in the plane oXY propagated over the distance
        """

        wave_number = 2 * torch.pi * refractive_index / self._wavelength

        rayleigh_range = torch.pi * (waist_radius**2) * refractive_index / (
            self._wavelength)

        radial_distance_squared = torch.pow(self._x_grid, 2) + torch.pow(
            self._y_grid, 2)

        hyperbolic_relation = waist_radius * (1 + (
            distance / rayleigh_range)**2)**(1/2)

        radius_of_curvature = distance * (1 + (rayleigh_range / distance)**2)

        # Gouy phase
        gouy_phase = torch.arctan(torch.tensor(distance / rayleigh_range))

        field = self.amplitude * (waist_radius / hyperbolic_relation) * (
            torch.exp(-radial_distance_squared / (hyperbolic_relation)**2) * (
                torch.exp(-1j * (wave_number * distance + wave_number * (
                    radial_distance_squared) / (2 * radius_of_curvature) - (
                        gouy_phase))))
        )
        return field


# TODO: check docstring
class PlaneWave(Beam):
    """A class that describes the plane wave

    Parameters
    ----------
    Beam : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        amplitude: float = 1
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        amplitude : float, optional
            Amplitude of the field from the source, by default 1
        """

        super().__init__(simulation_parameters, amplitude)

    def forward(
        self,
        distance: float,
        wave_vector: torch.Tensor,
        initial_phase: float = 0.
    ) -> torch.Tensor:
        """Method that describes propagation of the plane wave

        Parameters
        ----------
        distance : float
            Field propagation distance
        wave_vector : torch.Tensor
            Wave vector of the planar wave
        initial_phase : float, optional
            Initial phase of the planar wave, by default 0.

        Returns
        -------
        torch.Tensor
            Field in the plane oXY propagated over the distance
        """

        field = self.amplitude * torch.exp(
            1j * (wave_vector[0] * self._x_grid +
                  wave_vector[1] * self._y_grid +
                  wave_vector[2] * distance +
                  initial_phase)
        )

        return field


# TODO: check docstring
class SphericalWave(Beam):
    """A class that describes spherical wave

    Parameters
    ----------
    Beam : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        amplitude: float = 1
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        amplitude : float, optional
            Amplitude of the field from the source, by default 1
        """

        super().__init__(simulation_parameters, amplitude)

    def forward(
        self,
        distance: float,
        wave_vector: torch.Tensor,
        initial_phase: float = 0.
    ) -> torch.Tensor:
        """Method that describes propagation of the spherical wave

        Parameters
        ----------
        distance : float
            Field propagation distance
        wave_vector : torch.Tensor
            Wave vector of the spherical wave
        initial_phase : float, optional
            Initial phase of the spherical wave, by default 0.

        Returns
        -------
        torch.Tensor
            Field in the plane oXY propagated over the distance
        """

        radius = torch.sqrt(
            torch.pow(self._x_grid, 2) + torch.pow(self._y_grid, 2) + distance**2  # noqa: E501
        )

        field = self.amplitude / radius * torch.exp(
            1j * (wave_vector[0] * self._x_grid +
                  wave_vector[1] * self._y_grid +
                  wave_vector[2] * distance +
                  initial_phase)
        )

        return field
