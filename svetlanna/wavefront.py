import torch
from .simulation_parameters import SimulationParameters
from typing import Any, Self, Iterable, cast, TYPE_CHECKING
from scipy.special import hermite

class Wavefront(torch.Tensor):
    """Class that represents wavefront.
    It is a subclass of `torch.Tensor` with additional properties and methods for wavefront analysis and generation.
    """

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        # see https://github.com/albanD/subclass_zoo/blob/ec47458346c2a1cfcd5e676926a4bbc6709ff62e/base_tensor.py   # noqa: E501
        data = torch.as_tensor(data)
        return super(cls, Wavefront).__new__(cls, data)

    @property
    def intensity(self) -> torch.Tensor:
        """Intensity of the wavefront.

        Returns
        -------
        torch.Tensor
            Intensity ($|E|^2$).
        """
        return torch.abs(torch.Tensor(self)) ** 2

    @property
    def max_intensity(self) -> float:
        """Maximum intensity of the wavefront.

        Returns
        -------
        float
            Maximum intensity value.
        """
        return self.intensity.max().item()

    @property
    def phase(self) -> torch.Tensor:
        r"""Phase of the wavefront.

        Returns
        -------
        torch.Tensor
            Phase angle in the range $[-\pi, \pi]$.
        """
        # HOTFIX: problem with phase of -0. in visualization
        res = torch.angle(torch.Tensor(self) + 0.0)
        return res

    def fwhm(self, simulation_parameters: SimulationParameters) -> tuple[float, float]:
        """Full width at half maximum (FWHM) of the wavefront intensity.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.

        Returns
        -------
        tuple[float, float]
            FWHM along x and y axes.
        """

        x_step = torch.diff(simulation_parameters.x)[0].item()
        y_step = torch.diff(simulation_parameters.y)[0].item()

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
        distance: float = 0.0,
        wave_direction: Any = None,
        initial_phase: float = 0.0,
    ) -> Self:
        r"""Create a plane wave wavefront defind by the formula
        $$
        E(x, y) = \exp\left( i \left( k_x x + k_y y + k_z z + \phi_0 \right) \right)
        $$

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        distance : float, optional
            Free wave propagation distance $z$, by default 0.
        wave_direction : Any, optional
            Three component tensor-like vector with ($d_x$, $d_y$, $d_z$) coordinates,
            so $\vec{k} = k \frac{\vec{d}}{||\vec{d}||}$
            The resulting field propagates along the vector, by default
            the wave propagates along z direction.
        initial_phase : float, optional
            Additional phase offset ($\phi_0$), by default 0.

        Returns
        -------
        Wavefront
            Plane wave field.
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

    @classmethod
    def gaussian_beam(
        cls,
        simulation_parameters: SimulationParameters,
        waist_radius: float,
        distance: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> Self:
        r"""Generates the Gaussian beam wavefront defined by the formula
        $$
        E(x, y) = \frac{w_0}{w(z)} \exp\left( -\frac{(x - d_x)^2 + (y - d_y)^2}{w(z)^2} \right) \exp\left( i \left( k z + k\frac{(x - d_x)^2 + (y - d_y)^2}{2 R(z)} - \zeta(z) \right) \right)
        $$
        where $w(z) = w_0 \sqrt{1 + \left( \frac{z}{z_R} \right)^2}$,
        $R(z) = z \left( 1 + \left( \frac{z_R}{z} \right)^2 \right)$,
        $\zeta(z) = \arctan\left( \frac{z}{z_R} \right)$,
        and $z_R = \frac{\pi w_0^2}{\lambda}$ is the Rayleigh range.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        waist_radius : float
            Beam waist radius ($w_0$).
        distance : float, optional
            Free wave propagation distance $z$, by default 0.
        dx : float, optional
            Horizontal offset of the beam center ($d_x$), by default 0.
        dy : float, optional
            Vertical offset of the beam center ($d_y$), by default 0.

        Returns
        -------
        Wavefront
            Gaussian beam field in the oXY plane.
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
        r"""Generate wavefront of the spherical wave
        $$
        E(x, y) = \frac{1}{r} \exp\left( i \left( k r + \phi_0 \right) \right)
        $$
        where $r = \sqrt{(x - d_x)^2 + (y - d_y)^2 + z^2}$ is the distance from the point source to the point $(x, y)$ in the oXY plane.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        distance : float
            Distance from the point source to the oXY plane ($z$).
        initial_phase : float, optional
            Phase offset at the source ($\phi_0$), by default 0.
        dx : float, optional
            Horizontal position of the point source ($d_x$), by default 0.
        dy : float, optional
            Vertical position of the point source ($d_y$), by default 0.

        Returns
        -------
        Wavefront
            Spherical wave field in the oXY plane.
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

    @classmethod
def hermite_gauss(
    cls,
    simulation_parameters: SimulationParameters,
    waist_radius: float,
    distance: float = 0.0,
    dx: float = 0.0,
    dy: float = 0.0,
    m: int = 0,
    n: int = 0,
    amplitude: float = 1.0,
) -> Self:
    r"""Generate Hermite-Gaussian mode wavefront.

    The field is defined (in the waist plane) as:
    $$
    E(x, y) = A \, H_m\left(\frac{\sqrt{2}(x-d_x)}{w_0}\right)
                H_n\left(\frac{\sqrt{2}(y-d_y)}{w_0}\right)
                \exp\left(-\frac{(x-d_x)^2+(y-d_y)^2}{w_0^2}\right)
    $$
    For propagation distance $z \neq 0$, the beam parameters evolve:
    $w(z)=w_0\sqrt{1+(z/z_R)^2}$, $R(z)=z\left(1+(z_R/z)^2\right)$,
    and the Gouy phase becomes $(m+n+1)\arctan(z/z_R)$.

    Parameters
    ----------
    simulation_parameters : SimulationParameters
        Simulation parameters.
    waist_radius : float
        Beam waist radius ($w_0$).
    distance : float, optional
        Propagation distance $z$ from the waist, by default 0.
    dx, dy : float, optional
        Beam center offset, by default 0.
    m, n : int, optional
        Mode indices (x and y directions), by default 0 (fundamental mode).
    amplitude : float, optional
        Peak amplitude, by default 1.0.

    Returns
    -------
    Wavefront
        Hermite-Gaussian field.
    """
    sim_params = simulation_parameters
    wavelength = sim_params.cast(sim_params.wavelength, "wavelength")
    x = sim_params.cast(sim_params.x, "x")   # expected 2D tensor
    y = sim_params.cast(sim_params.y, "y")   # expected 2D tensor

    # Shift coordinates
    X = x - dx
    Y = y - dy

    wave_number = 2 * torch.pi / wavelength
    z_R = torch.pi * waist_radius**2 / wavelength  # Rayleigh range

    # Polynomial Hermite using recurrence (PyTorch implementation)
    def hermite_poly(n: int, xi: torch.Tensor) -> torch.Tensor:
        """Evaluate Hermite polynomial H_n(xi) using recurrence."""
        if n == 0:
            return torch.ones_like(xi)
        if n == 1:
            return 2 * xi
        H_prev2 = torch.ones_like(xi)
        H_prev1 = 2 * xi
        for k in range(2, n + 1):
            H_curr = 2 * xi * H_prev1 - 2 * (k - 1) * H_prev2
            H_prev2, H_prev1 = H_prev1, H_curr
        return H_prev1

    # Beam parameters at distance z
    if distance == 0:
        # Waist plane
        w = waist_radius
        R = torch.tensor(float('inf'))  # infinite radius
        gouy = torch.tensor(0.0)
    else:
        w = waist_radius * torch.sqrt(1 + (distance / z_R) ** 2)
        R = distance * (1 + (z_R / distance) ** 2)
        gouy = (m + n + 1) * torch.arctan(distance / z_R)

    # Scaled coordinates
    xi_x = torch.sqrt(torch.tensor(2.0)) * X / w
    xi_y = torch.sqrt(torch.tensor(2.0)) * Y / w

    Hm = hermite_poly(m, xi_x)
    Hn = hermite_poly(n, xi_y)


    # Transverse profile
    field = Hm * Hn * torch.exp(-(X**2 + Y**2) / w**2)

    if distance != 0:
        # Add phase terms: propagation phase, curvature, and Gouy phase
        field = field * torch.exp(1j * wave_number * distance)
        field = field * torch.exp(1j * wave_number * (X**2 + Y**2) / (2 * R))
        field = field * torch.exp(-1j * gouy)

    return cls(field)
    # === methods below are added for typing only ===

    if TYPE_CHECKING:

        def __mul__(self, other: Any) -> Self: ...

        def __rmul__(self, other: Any) -> Self: ...

        def __add__(self, other: Any) -> Self: ...

        def __radd__(self, other: Any) -> Self: ...

        def __truediv__(self, other: Any) -> Self: ...

        def __rtruediv__(self, other: Any) -> Self: ...

        def to(self, *args, **kwargs) -> Self: ...


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
        wf_axes = sim_params.axis_names

    from .axes_math import tensor_dot

    res, _ = tensor_dot(wf, b, wf_axes, b_axis, preserve_a_axis=True)
    return cast(Wavefront, res)