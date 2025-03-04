import torch
from .simulation_parameters import SimulationParameters
from typing import Any, Self, Iterable, cast, TYPE_CHECKING
from .axes_math import tensor_dot, cast_tensor


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
            phase from $0$ to $2\\pi$
        """
        # HOTFIX: problem with phase of -0. in visualization
        res = torch.angle(torch.Tensor(self) + 0.0)
        # res[res < 0] += 2 * torch.pi
        return res

    def fwhm(
        self,
        simulation_parameters: SimulationParameters
    ) -> tuple[float, float]:
        """Calculates full width at half maximum of the wavefront

        Returns
        -------
        tuple[float, float]
            full width at half maximum along x and y axes
        """

        x_step = torch.diff(simulation_parameters.axes.W)[0].item()
        y_step = torch.diff(simulation_parameters.axes.H)[0].item()

        max_intensity = self.max_intensity
        half_max_intensity = max_intensity / 2

        indices = torch.nonzero(self.intensity >= half_max_intensity)

        min_y, min_x = torch.min(indices, dim=0)[0]
        max_y, max_x = torch.max(indices, dim=0)[0]

        fwhm_x = (max_x - min_x) * x_step
        fwhm_y = (max_y - min_y) * y_step

        return fwhm_x.item(), fwhm_y.item()

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

        wave_direction = torch.tensor(
            wave_direction,
            dtype=torch.float32,
            device=simulation_parameters.device
        )
        if wave_direction.shape != torch.Size([3]):
            raise ValueError(
                "wave_direction should contain exactly three components"
            )
        wave_direction = wave_direction / torch.norm(wave_direction)

        wave_number = 2 * torch.pi / simulation_parameters.axes.wavelength
        x = simulation_parameters.axes.W[None, :]
        y = simulation_parameters.axes.H[:, None]

        kxx, axes = tensor_dot(wave_number, x, 'wavelength', ('H', 'W'))
        kyy, _ = tensor_dot(wave_number, y, 'wavelength', ('H', 'W'))
        kzz = wave_number[..., None, None] * distance
        field = torch.exp(
            1j * (
                wave_direction[0] * kxx +
                wave_direction[1] * kyy +
                wave_direction[2] * kzz +
                initial_phase
            )
        )
        field = torch.exp(1j * wave_direction[0] * kxx)
        field = field * torch.exp(1j * wave_direction[1] * kyy)
        field = field * torch.exp(1j * wave_direction[2] * kzz + initial_phase)

        return cls(cast_tensor(field, axes, simulation_parameters.axes.names))

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

        wave_number = 2 * torch.pi / simulation_parameters.axes.wavelength

        rayleigh_range = torch.pi * (waist_radius**2) / simulation_parameters.axes.wavelength    # noqa: E501

        x = simulation_parameters.axes.W[None, :]
        y = simulation_parameters.axes.H[:, None]
        radial_distance_squared = x**2 + y**2

        hyperbolic_relation = waist_radius * (1 + (distance / rayleigh_range)**2)**(1/2)    # noqa: E501

        inverse_radius_of_curvature = distance / (distance**2 + rayleigh_range**2)  # noqa: E501

        # Gouy phase
        gouy_phase = torch.arctan(distance / rayleigh_range)

        phase1, axes1 = tensor_dot(
            a=1j * wave_number * inverse_radius_of_curvature / 2,
            b=radial_distance_squared,
            a_axis='wavelength',
            b_axis=('H', 'W')
        )

        field = torch.exp(phase1)
        field, _ = tensor_dot(
            a=field,
            b=torch.exp(1j * wave_number * distance),
            a_axis=axes1, b_axis='wavelength', preserve_a_axis=True
        )
        field, _ = tensor_dot(
            a=field,
            b=torch.exp(-1j * gouy_phase),
            a_axis=axes1, b_axis='wavelength', preserve_a_axis=True
        )
        phase2, axes2 = tensor_dot(
            a=-1/(hyperbolic_relation)**2,
            b=radial_distance_squared,
            a_axis='wavelength',
            b_axis=('H', 'W')
        )
        field, axes = tensor_dot(
            a=field,
            b=torch.exp(phase2),
            a_axis=axes1,
            b_axis=axes2,
            preserve_a_axis=True
        )
        field, _ = tensor_dot(
            a=field,
            b=waist_radius / hyperbolic_relation,
            a_axis=axes,
            b_axis='wavelength',
            preserve_a_axis=True
        )

        return cls(
            cast_tensor(field, axes, simulation_parameters.axes.names)
        )

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
        wave_number = 2 * torch.pi / simulation_parameters.axes.wavelength

        x = simulation_parameters.axes.W[None, :]
        y = simulation_parameters.axes.H[:, None]

        radius = torch.sqrt(
            (x**2 + y**2) + distance**2
        )

        phase, axes = tensor_dot(
            a=wave_number,
            b=radius,
            a_axis='wavelength',
            b_axis=('H', 'W')
        )
        field, _ = tensor_dot(
            a=torch.exp(1j * (phase + initial_phase)),
            b=1 / radius,
            a_axis=axes,
            b_axis=('H', 'W'),
            preserve_a_axis=True
        )

        return cls(cast_tensor(field, axes, simulation_parameters.axes.names))

    # === methods below are added for typing only ===

    if TYPE_CHECKING:
        def __mul__(self, other: Any) -> Self:
            ...

        def __rmul__(self, other: Any) -> Self:
            ...

        def __add__(self, other: Any) -> Self:
            ...

        def __radd__(self, other: Any) -> Self:
            ...

        def __truediv__(self, other: Any) -> Self:
            ...

        def __rtruediv__(self, other: Any) -> Self:
            ...


DEFAULT_LAST_AXES_NAMES = (
    # 'pol',
    # 'wavelength',
    'H',
    'W'
)


def mul(
    wf: Wavefront,
    b: Any,
    b_axis: str | Iterable[str],
    sim_params: SimulationParameters | None = None
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

    res, _ = tensor_dot(wf, b, wf_axes, b_axis, preserve_a_axis=True)
    return cast(Wavefront, res)
