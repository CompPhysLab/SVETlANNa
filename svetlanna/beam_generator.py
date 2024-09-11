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

        """Constructor class

        :param simulation_parameters: Class exemplar, that describes optical
        system
        :type simulation_parameters: SystemParameters
        :param amplitude: Amplitude of the field from the source, default=1
        :type amplitude: float
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


class GaussianBeam(Beam):

    """A class describing gaussian beam

    :param Beam: parent class
    :type Beam: class
    """
    # TODO: create docstring

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        amplitude: float = 1
    ):

        """Constructor method

        :param simulation_parameters: _description_
        :type simulation_parameters: SimulationParameters
        :param amplitude: _description_, defaults to 1
        :type amplitude: float, optional
        """

        super().__init__(simulation_parameters, amplitude)

    def forward(
        self,
        distance: float,
        waist_radius: float,
        refractive_index: float = 1.
    ) -> torch.Tensor:

        """method describes Gaussian beam

        :param distance: Distance over which the beam propagates
        :type distance: float
        :param waist_radius: Waist radius of the beam
        :type waist_radius: float
        :param refractive_index: Refractive index of the medium, defaults to 1
        :type refractive_index: float
        :return: Beam field in the plane oXY propagated over the distance
        :rtype: torch.Tensor
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


# TODO: create docstring
class PlaneWave(Beam):

    """A class describing plane wave

    :param Beam: _description_
    :type Beam: _type_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        amplitude: float = 1
    ):

        """Constructor method

        :param simulation_parameters: _description_
        :type simulation_parameters: SimulationParameters
        :param amplitude: _description_, defaults to 1
        :type amplitude: float, optional
        """

        super().__init__(simulation_parameters, amplitude)

    def forward(
        self,
        distance: float,
        wave_vector: torch.Tensor,
        initial_phase: float = 0.
    ) -> torch.Tensor:

        """Method generates the planar wave

        :param distance: Distance over which the beam propagates
        :type distance: float
        :param wave_vector: wave vector of the planar wave
        :type wave_vector: torch.Tensor
        :param initial_phase: initial phase of the planar wave, defaults to 0
        :type initial_phase: float, optional
        :return: Field in the plane oXY propagated over the distance
        :rtype: torch.Tensor
        """

        field = self.amplitude * torch.exp(
            1j * (wave_vector[0] * self._x_grid +
                  wave_vector[1] * self._y_grid +
                  wave_vector[2] * distance +
                  initial_phase)
        )

        return field


# TODO: create docstrings
class SphericalWave(Beam):
    """_summary_

    :param Beam: _description_
    :type Beam: _type_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        amplitude: float = 1
    ):

        """Constructor method

        :param simulation_parameters: _description_
        :type simulation_parameters: SimulationParameters
        :param amplitude: _description_, defaults to 1
        :type amplitude: float, optional
        """

        super().__init__(simulation_parameters, amplitude)

    def forward(
        self,
        distance: float,
        wave_vector: torch.Tensor,
        initial_phase: float = 0.
    ) -> torch.Tensor:

        """Method that generates the spherical wave

        :param wave_vector: wave vector of the spherical wave
        :type wave_vector: torch.Tensor
        :param initial_phase: initial phase of the spherical wave, defaults
                              to 0
        :type initial_phase: float, optional
        :return: Field in the plane oXY propagated over the distance
        :rtype: torch.Tensor
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
