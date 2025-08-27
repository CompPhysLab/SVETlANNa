import torch
from enum import Enum
from .simulation_parameters import SimulationParameters
from typing import Any, Self, Iterable, Tuple, cast, TYPE_CHECKING, overload, Union, Sequence, Literal  # noqa: E501
from .axes_math import tensor_dot, cast_tensor


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

    @property
    def axes_names(self) -> tuple[str, ...]:
        """Returns the names of the axes of the wavefront

        Returns
        -------
        tuple[str, ...]
            Names of the axes
        """
        if hasattr(self, 'axes'):
            return self.axes.names
        else:
            return DEFAULT_LAST_AXES_NAMES

    @overload
    def get_polarization_components(self, component: int) -> torch.Tensor:
        """Returns the polarization component of the wavefront

        Parameters
        ----------
        component : int
            Index of the polarization component

        Returns
        -------
        torch.Tensor
            Polarization component
        """
        ...

    @overload
    def get_polarization_components(
        self,
        component: Literal['x', 'y', 'all'] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Returns the polarization component of the wavefront

        Parameters
        ----------
        component : Literal['x', 'y', 'all'], optional
            The polarization component to retrieve

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            The requested polarization component(s)
        """
        ...

    def get_polarization_components(self, component=None):
        """Returns the polarization component of the wavefront

        Parameters
        ----------
        component : Literal['x', 'y', 'all'], optional
            The polarization component to retrieve, by default None

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            The requested polarization component(s)
        """

        if not hasattr(self, 'axes'):
            raise ValueError("Axes are not defined")

        if "pol" not in self.axes_names:
            raise ValueError("Polarization axes is not defined")

        pol_index = self.axes_names.index("pol")
        all_components = torch.unbind(self, dim=pol_index)

        if component is None or component == "all":
            return all_components
        elif isinstance(component, int):
            return all_components[component]
        elif component == "x":
            return all_components[0]
        elif component == "y":
            return all_components[1]
        else:
            raise ValueError(f"Unknown component: {component}")

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

    @overload
    @classmethod
    def plane_wave(
        cls,
        simulation_parameters: SimulationParameters,
        distance: float = 0.,
        wave_direction: Union[Sequence[float], torch.Tensor, None] = None,
        initial_phase: float = 0.
    ) -> Self:
        """Generate linear polarized plane wave"""
        ...

    @overload
    @classmethod
    def plane_wave(
        cls,
        simulation_parameters: SimulationParameters,
        distance: float = 0.,
        wave_direction: Union[Sequence[float], torch.Tensor, None] = None,
        stokes_vector: Union[Sequence[float], torch.Tensor] = ...
    ) -> Self:
        """Generate polarized plane wave defined by Stokes vector"""
        ...

    @classmethod
    def plane_wave(
        cls,
        simulation_parameters: SimulationParameters,
        distance: float = 0.,
        wave_direction: Union[Sequence[float], torch.Tensor, None] = None,
        initial_phase: float = 0.,
        stokes_vector: Union[Sequence[float], torch.Tensor, None] = None
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
            wave_direction = [0., 0., 1.]

        _wave_direction = cls._validate_and_normalize_wave_direction(
            wave_direction, simulation_parameters
        )

        wave_number = 2 * torch.pi / simulation_parameters.axes.wavelength
        x = simulation_parameters.axes.W[None, :]
        y = simulation_parameters.axes.H[:, None]

        kxx, axes = tensor_dot(wave_number, x, 'wavelength', ('H', 'W'))
        kyy, _ = tensor_dot(wave_number, y, 'wavelength', ('H', 'W'))
        kzz = wave_number[..., None, None] * distance

        field = torch.exp(1j * _wave_direction[0] * kxx)
        field = field * torch.exp(1j * _wave_direction[1] * kyy)
        field = field * torch.exp(1j * _wave_direction[2] * kzz + initial_phase)    # noqa: E501

        return cls._define_polarization(
            field, axes, stokes_vector, simulation_parameters
        )

    @overload
    @classmethod
    def gaussian_beam(
        cls,
        simulation_parameters: SimulationParameters,
        waist_radius: float,
        distance: float = 0.,
        dx: float = 0.,
        dy: float = 0.,
    ) -> Self:
        "Generate linear polarized Gaussian beam"
        ...

    @overload
    @classmethod
    def gaussian_beam(
        cls,
        simulation_parameters: SimulationParameters,
        waist_radius: float,
        stokes_vector: Union[Sequence[float], torch.Tensor] = None,
        distance: float = 0.,
        dx: float = 0.,
        dy: float = 0.,
    ) -> Self:
        "Generate polarized Gaussian beam defined by Stokes vector"
        ...

    @classmethod
    def gaussian_beam(
        cls,
        simulation_parameters: SimulationParameters,
        waist_radius: float,
        stokes_vector: Union[Sequence[float], torch.Tensor] = None,
        distance: float = 0.,
        dx: float = 0.,
        dy: float = 0.,
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

        wave_number = 2 * torch.pi / simulation_parameters.axes.wavelength

        rayleigh_range = torch.pi * (waist_radius**2) / simulation_parameters.axes.wavelength    # noqa: E501

        x = simulation_parameters.axes.W[None, :] - dx
        y = simulation_parameters.axes.H[:, None] - dy
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

        return cls._define_polarization(
            field, axes, stokes_vector, simulation_parameters
        )

    @overload
    @classmethod
    def spherical_wave(
        cls,
        simulation_parameters: SimulationParameters,
        distance: float = 0.,
        initial_phase: float = 0.,
        dx: float = 0.,
        dy: float = 0.,
    ) -> Self:
        """Generate linear polarized spherical wave"""

    @overload
    @classmethod
    def spherical_wave(
        cls,
        simulation_parameters: SimulationParameters,
        stokes_vector: Union[Sequence[float], torch.Tensor] = None,
        distance: float = 0.,
        dx: float = 0.,
        dy: float = 0.,
    ) -> Self:
        """Generate polarized spherical wave defined by Stokes vector"""

    @classmethod
    def spherical_wave(
        cls,
        simulation_parameters: SimulationParameters,
        stokes_vector: Union[Sequence[float], torch.Tensor] = None,
        distance: float = 0.,
        initial_phase: float = 0.,
        dx: float = 0.,
        dy: float = 0.,
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
        wave_number = 2 * torch.pi / simulation_parameters.axes.wavelength

        x = simulation_parameters.axes.W[None, :] - dx
        y = simulation_parameters.axes.H[:, None] - dy

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

        return cls._define_polarization(
            field, axes, stokes_vector, simulation_parameters
        )

    @classmethod
    def _validate_and_normalize_wave_direction(
        cls,
        wave_direction: Union[Sequence[float], torch.Tensor, None],
        simulation_parameters: SimulationParameters
    ) -> torch.Tensor:
        """
        Validates and normalizes wave_direction parameter.

        Parameters
        ----------
        wave_direction : Union[Sequence[float], torch.Tensor, None]
            Wave direction vector
        simulation_parameters : SimulationParameters
            Simulation parameters for device information

        Returns
        -------
        torch.Tensor
            Normalized wave direction tensor

        Raises
        ------
        ValueError
            If wave_direction doesn't have exactly 3 components
        TypeError
            If wave_direction is not a supported type
        """
        if isinstance(wave_direction, (list, tuple)):
            if len(wave_direction) != 3:
                raise ValueError(
                    "wave_direction must have 3 components," +
                    f" got {len(wave_direction)}"
                )
            wave_direction = torch.tensor(
                wave_direction,
                dtype=torch.float32,
                device=simulation_parameters.device
            )
        elif isinstance(wave_direction, torch.Tensor):
            if wave_direction.numel() != 3:
                raise ValueError(
                    "wave_direction tensor must have 3 elements," +
                    f" got {wave_direction.numel()}"
                )
            wave_direction = wave_direction.to(
                dtype=torch.float32,
                device=simulation_parameters.device
            ).flatten()
        else:
            try:
                wave_direction = torch.tensor(
                    wave_direction,
                    dtype=torch.float32,
                    device=simulation_parameters.device
                ).flatten()
                if wave_direction.numel() != 3:
                    raise ValueError(
                        "wave_direction must have 3 components," +
                        f" got {wave_direction.numel()}"
                    )
            except Exception as e:
                raise TypeError(
                    "wave_direction must be list, tuple, or tensor" +
                    " with 3 elements"
                ) from e

        # Normalize the direction vector
        wave_direction = wave_direction / torch.norm(wave_direction)
        return wave_direction

    @classmethod
    def _validate_and_normalize_stokes_vector(
        cls,
        stokes_vector: Union[Sequence[float], torch.Tensor, None],
        simulation_parameters: SimulationParameters
    ) -> torch.Tensor:
        """
        Validates and normalizes stokes_vector parameter.

        Parameters
        ----------
        stokes_vector : Union[Sequence[float], torch.Tensor, None]
            Stokes vector for polarization
        simulation_parameters : SimulationParameters
            Simulation parameters for device information

        Returns
        -------
        torch.Tensor
            Normalized stokes vector tensor

        Raises
        ------
        ValueError
            If stokes_vector doesn't have exactly 4 components or is
            physically invalid
        TypeError
            If stokes_vector is not a supported type
        """
        if isinstance(stokes_vector, (list, tuple)):
            if len(stokes_vector) != 4:
                raise ValueError(
                    "stokes_vector must have 4 components," +
                    f" got {len(stokes_vector)}"
                )
            stokes_vector = torch.tensor(
                stokes_vector,
                dtype=torch.float32,
                device=simulation_parameters.device
            )
        elif isinstance(stokes_vector, torch.Tensor):
            if stokes_vector.numel() != 4:
                raise ValueError(
                    "stokes_vector tensor must have 4 elements," +
                    f" got {stokes_vector.numel()}"
                )
            stokes_vector = stokes_vector.to(
                dtype=torch.float32,
                device=simulation_parameters.device
            ).flatten()
        else:
            try:
                stokes_vector = torch.tensor(
                    stokes_vector,
                    dtype=torch.float32,
                    device=simulation_parameters.device
                ).flatten()
                if stokes_vector.numel() != 4:
                    raise ValueError(
                        "stokes_vector must have 4 components," +
                        f" got {stokes_vector.numel()}"
                    )
            except Exception as e:
                raise TypeError(
                    "stokes_vector must be list, tuple, or tensor" +
                    " with 4 elements"
                ) from e

        if (stokes_vector[0] > 0.0):
            normalized_stokes_vector = stokes_vector / stokes_vector[0]
        else:
            raise ValueError("Invalid first component of the Stokes vector")

        return normalized_stokes_vector

    @classmethod
    def _define_polarization(
        cls,
        field: torch.Tensor,
        axes: tuple[str, ...],
        stokes_vector: Union[Sequence[float], torch.Tensor] | None,
        simulation_parameters: SimulationParameters
    ) -> Self:
        """Defines the polarization state of the wavefront.

        Parameters
        ----------
        field : torch.Tensor
            The electric field tensor of the wavefront
        axes : tuple[str, ...]
            The axes of the wavefront tensor
        stokes_vector : Union[Sequence[float], torch.Tensor] | None
            The Stokes vector defining the polarization state, by default None.
        simulation_parameters : SimulationParameters
            The simulation parameters for the wavefront.

        Returns
        -------
        Self
            The wavefront with the defined polarization state.
        """

        if stokes_vector is None:

            wavefront = cls(
                cast_tensor(field, axes, simulation_parameters.axes.names)
            )

        else:
            _stokes_vector = cls._validate_and_normalize_stokes_vector(
                stokes_vector, simulation_parameters
            )
            s0, s1, s2, s3 = _stokes_vector

            ex_amplitude = torch.sqrt((s0 + s1) / 2)
            ey_amplitude = torch.sqrt((s0 - s1) / 2)
            phase_diff = torch.atan2(s3, s2)

            amplitude_vec = torch.tensor([
                ex_amplitude * torch.exp(1j * phase_diff), ey_amplitude
            ])

            field, axes = tensor_dot(amplitude_vec, field, "pol", axes)
            wavefront = cls(
                cast_tensor(field, axes, simulation_parameters.axes.names)
            )

        wavefront.axes = simulation_parameters.axes
        return wavefront

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
