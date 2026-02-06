import torch
from enum import Enum
from .simulation_parameters import SimulationParameters
from typing import Any, Self, Iterable, cast, TYPE_CHECKING


class PolarizationComponent(Enum):
    """Represents the polarization components of a wavefront

    Parameters
    ----------
    Enum : _type_
        _description_
    """
    X = "x"
    Y = "y"
    ALL = "all"


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
    def max_intensity(self) -> float:
        """Calculates maximum intensity of the wavefront

        Returns
        -------
        float
            maximum intensity
        """
        return self.intensity.max().item()

    @property
    def phase(self) -> torch.Tensor:
        """Calculates phase of the wavefront

        Returns
        -------
        torch.Tensor
            phase from $-\\pi$ to $\\pi$
        """
        # HOTFIX: problem with phase of -0. in visualization
        res = torch.angle(torch.Tensor(self) + 0.0)
        return res

    def fwhm(self, simulation_parameters: SimulationParameters) -> tuple[float, float]:
        """Calculates full width at half maximum of the wavefront

        Returns
        -------
        tuple[float, float]
            full width at half maximum along x and y axes
        """

        x_step = torch.diff(simulation_parameters.axes.x)[0].item()
        y_step = torch.diff(simulation_parameters.axes.y)[0].item()

        max_intensity = self.max_intensity
        half_max_intensity = max_intensity / 2

        indices = torch.nonzero(self.intensity >= half_max_intensity)

        min_y, min_x = torch.min(indices, dim=0)[0]
        max_y, max_x = torch.max(indices, dim=0)[0]

        fwhm_x = (max_x - min_x) * x_step
        fwhm_y = (max_y - min_y) * y_step

        return fwhm_x.item(), fwhm_y.item()

    @overload
    @classmethod
    def plane_wave(
        cls,
        simulation_parameters: SimulationParameters,
        distance: float = 0.0,
        wave_direction: Any = None,
        initial_phase: float = 0.0,
    ) -> Self:
        """Generate wavefront of the plane wave

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            simulation parameters
        distance : float, optional
            free wave propagation distance, by default 0.
        wave_direction : Union[Sequence[float], torch.Tensor, None], optional
            direction of wave propagation, by default None
        initial_phase : float, optional
            initial phase of the wave, by default 0.
        stokes_vector : Union[Sequence[float], torch.Tensor, None], optional
            Stokes vector defining the polarization state, by default None

        Returns
        -------
        Self
            Wavefront of the generated plane wave
        """

        # by default the wave propagates along z direction
        if wave_direction is None:
            wave_direction = [0.0, 0.0, 1.0]

        wave_direction = torch.tensor(
            wave_direction, device=simulation_parameters.device
        )
        if wave_direction.shape != torch.Size([3]):
            raise ValueError("wave_direction should contain exactly three components")

        wave_direction = wave_direction / torch.norm(
            wave_direction + 0.0  # if wave_direction is Int, cast to Float
        )

        # Cast axes
        sim_params = simulation_parameters
        wavelength = sim_params.cast(sim_params.wavelength, "wavelength")
        x = sim_params.cast(sim_params.x, "x")
        y = sim_params.cast(sim_params.y, "y")

        wave_number = 2 * torch.pi / wavelength
        kxx = wave_direction[0] * wave_number * x
        kyy = wave_direction[1] * wave_number * y
        kzz = wave_direction[2] * wave_number * distance

        field = torch.exp(1j * (kxx + kyy + kzz + initial_phase))

        return cls(field)

    @overload
    @classmethod
    def gaussian_beam(
        cls,
        simulation_parameters: SimulationParameters,
        waist_radius: float,
        distance: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> Self:
        """Generates the Gaussian beam.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            simulation parameters
        waist_radius : float
            waist radius of the beam
        stokes_vector : Union[Sequence[float], torch.Tensor], optional
            Stokes vector defining the polarization state, by default None
        distance : float, optional
            free wave propagation distance, by default 0.
        dx : float, optional
            horizontal position of the beam center, by default 0.
        dy : float, optional
            vertical position of the beam center, by default 0.

        Returns
        -------
        Wavefront
            Gaussian beam field in the plane oXY propagated over the distance
        """

        # Cast axes
        sim_params = simulation_parameters
        wavelength = sim_params.cast(sim_params.wavelength, "wavelength")
        x = sim_params.cast(sim_params.x, "x")
        y = sim_params.cast(sim_params.y, "y")

        wave_number = 2 * torch.pi / wavelength

        rayleigh_range = torch.pi * waist_radius**2 / wavelength

        radial_distance_squared = (x - dx) ** 2 + (y - dy) ** 2

        hyperbolic_relation = waist_radius * torch.sqrt(
            1 + (distance / rayleigh_range) ** 2
        )

        inverse_radius_of_curvature = distance / (distance**2 + rayleigh_range**2)

        gouy_phase = torch.arctan(distance / rayleigh_range)

        field = waist_radius / hyperbolic_relation
        field = field * torch.exp(-radial_distance_squared / hyperbolic_relation**2)
        field = field * torch.exp(1j * wave_number * distance)
        field = field * torch.exp(1j * wave_number * radial_distance_squared * inverse_radius_of_curvature / 2)  # fmt: skip
        field = field * torch.exp(-1j * gouy_phase)

        return cls(field)

    @classmethod
    def spherical_wave(
        cls,
        simulation_parameters: SimulationParameters,
        distance: float,
        initial_phase: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> Self:
        """Generate wavefront of the spherical wave

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            simulation parameters
        stokes_vector: Union[Sequence[float], torch.Tensor], optional
            Stokes vector defining the polarization state, by default None
        distance : float, optional
            distance between the source and the oXY plane, by default 0.
        initial_phase : float, optional
            additional phase to the resulting field, by default 0.
        dx : float, optional
            horizontal position of the spherical wave center, by default 0.
        dy : float, optional
            vertical position of the spherical wave center, by default 0.

        Returns
        -------
        Wavefront
            Gaussian beam field in the plane oXY propagated over the distance
        """
        # Cast axes
        sim_params = simulation_parameters
        wavelength = sim_params.cast(sim_params.wavelength, "wavelength")
        x = sim_params.cast(sim_params.x, "x")
        y = sim_params.cast(sim_params.y, "y")

        wave_number = 2 * torch.pi / wavelength
        radius = torch.sqrt((x - dx) ** 2 + (y - dy) ** 2 + distance**2)

        phase = wave_number * radius
        field = torch.exp(1j * (phase + initial_phase)) / radius

        return cls(field)

    # === methods below are added for typing only ===

    if TYPE_CHECKING:

        def __mul__(self, other: Any) -> Self: ...

        def __rmul__(self, other: Any) -> Self: ...

        def __add__(self, other: Any) -> Self: ...

        def __radd__(self, other: Any) -> Self: ...

        def __truediv__(self, other: Any) -> Self: ...

        def __rtruediv__(self, other: Any) -> Self: ...


DEFAULT_LAST_AXES_NAMES = (
    # 'pol',
    # 'wavelength',
    "y",
    "x",
)


def mul(
    wf: Wavefront,
    b: Any,
    b_axis: str | Iterable[str],
    sim_params: SimulationParameters | None = None,
) -> Wavefront:
    """Multiplication of the wavefront and tensor.

    Parameters
    ----------
    wf : Wavefront
        wavefront
    b : Any
        tensor
    b_axis : str | Iterable[str]
        tensor's axis name
    sim_params : SimulationParameters | None, optional
        simulation parameters, by default None

    Returns
    -------
    Wavefront
        product result
    """

    # if b is not a tensor, use default mul operation
    if not isinstance(b, torch.Tensor):
        return wf * b

    if sim_params is None:
        wf_axes = DEFAULT_LAST_AXES_NAMES
    else:
        wf_axes = sim_params.axes.names

    from .axes_math import tensor_dot

    res, _ = tensor_dot(wf, b, wf_axes, b_axis, preserve_a_axis=True)
    return cast(Wavefront, res)
