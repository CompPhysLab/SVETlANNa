from typing import Literal, Iterable, Tuple
import torch
import torch.nn.functional as F
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront
from warnings import warn
from ..specs import PrettyReprRepr, ParameterSpecs
from ..visualization import ElementHTML, jinja_env


class FreeSpace(Element):
    """A class that describes a propagation of the field in free space
    between two optical elements
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        distance: OptimizableFloat,
        method: Literal['fresnel-TF', 'fresnel-IR', 'ASM', 'zpASM', 'RSC', 'zpRSC']
        method: Literal["fresnel", "AS"],
    ):
        # TODO: rewrite docstrings
        """Free space element.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            An instance describing the optical system's simulation parameters.
        distance : float
            The distance of the free space propagation.
        method : Literal['fresnel', 'AS', 'zpAS']
            Method describing propagation in free space
                (1) 'AS' - angular spectrum method,
                (2) 'fresnel' - fresnel approximation,
                (3) 'zpAS' - angular spectrum method with zero padding
                (4) 'RSC' - Rayleigh-Sommerfeld Convolution method
        """
        super().__init__(simulation_parameters)

        self.distance = self.process_parameter("distance", distance)
        self.method = self.process_parameter("method", method)

        # params extracted from SimulationParameters
        self._device = self.simulation_parameters.device

        self._x_index = self.simulation_parameters.index("x")
        self._y_index = self.simulation_parameters.index("y")

        x = self.simulation_parameters.x
        y = self.simulation_parameters.y

        x_nodes = x.shape[0]
        y_nodes = y.shape[0]

        # Compute spatial grid spacing
        dx = (x[1] - x[0]) if x_nodes > 1 else 1.0
        dy = (y[1] - y[0]) if y_nodes > 1 else 1.0

        # Compute wave vectors
        kx = 2 * torch.pi * torch.fft.fftfreq(x_nodes, dx, device=device)
        ky = 2 * torch.pi * torch.fft.fftfreq(y_nodes, dy, device=device)

        # Compute wave vectors grids
        _kx = self.simulation_parameters.cast(kx, "x")
        _ky = self.simulation_parameters.cast(ky, "y")

        # Calculate (kx^2+ky^2) / k^2 relation
        # 1) Calculate wave vector
        wave_number = self.simulation_parameters.cast(
            2 * torch.pi / self.simulation_parameters.wavelength, "wavelength"
        )

        # 2) Calculate (kx^2+ky^2) tensor
        kx2ky2 = _kx**2 + _ky**2

        # 3) Calculate (kx^2+ky^2) / k^2
        relation = kx2ky2 / wave_number**2

        # TODO: Remove legacy filter
        use_legacy_filter = False

        # Legacy low pass filter, (kx^2+ky^2) / k^2 <= 1
        # The filter removes contribution of evanescent waves
        if use_legacy_filter:
            # TODO: Shouldn't the 88'th string be here?
            condition = relation <= 1  # calculate the low pass filter condition  # noqa
            condition = condition.to(_kx)  # cast bool to float

            # Registering Buffer for _low_pass_filter
            self._low_pass_filter: torch.Tensor | int = self.make_buffer(
                "_low_pass_filter", condition
            )
        else:
            self._low_pass_filter = 1

        # Reshape wave vector for further calculations

        # Registering Buffer for _wave_number
        self._wave_number = self.make_buffer("_wave_number", wave_number)

        # Calculate kz
        if use_legacy_filter:
            # kz = sqrt(k^2 - (kx^2 + ky^2)), if (kx^2 + ky^2) / k^2 <= 1
            #    or
            # kz = |k| otherwise
            wave_number_z = torch.sqrt(
                self._wave_number**2 - self._low_pass_filter * kx2ky2
            )
        else:
            # kz = sqrt(k^2 - (kx^2 + ky^2))
            wave_number_z = torch.sqrt(
                self._wave_number**2 - kx2ky2 + 0j
            )  # 0j is required to convert argument to complex

        # Registering Buffer for _wave_number_z
        self._wave_number_z = self.make_buffer("_wave_number_z", wave_number_z)

        # Calculate kz taylored, used by Fresnel approximation
        wave_number_z_eff_fresnel = -0.5 * kx2ky2 / self._wave_number

        # Registering Buffer for _wave_number_z_eff_fresnel
        self._wave_number_z_eff_fresnel = self.make_buffer(
            "_wave_number_z_eff_fresnel", wave_number_z_eff_fresnel
        )

        # Warnings for fulfilling the method criteria
        # See (9.32), (9.36) in
        # Fourier Optics and Computational Imaging (2nd ed)
        # by Kedar Khare, Mansi Butola and Sunaina Rajor
        Lx = torch.abs(x[-1] - x[0])
        Ly = torch.abs(y[-1] - y[0])
        if method == "AS":
            kx_max = torch.max(torch.abs(_kx))
            ky_max = torch.max(torch.abs(_ky))
            x_condition = kx_max >= wave_number / torch.sqrt(
                1 + (2 * distance / Lx) ** 2
            )
            y_condition = ky_max >= wave_number / torch.sqrt(
                1 + (2 * distance / Ly) ** 2
            )

            if not torch.all(x_condition):
                warn(
                    "Aliasing problems may occur in the AS method. "
                    "Consider reducing the distance "
                    "or increasing the Nx*dx product."
                )
            if not torch.all(y_condition):
                warn(
                    "Aliasing problems may occur in the AS method. "
                    "Consider reducing the distance "
                    "or increasing the Ny*dy product."
                )

        if method == "fresnel":
            diagonal_squared = Lx**2 + Ly**2
            condition = distance**3 > wave_number / 8 * (diagonal_squared) ** 2

            if not torch.all(condition):
                warn(
                    "The paraxial (near-axis) optics condition "
                    "required for the Fresnel method is not satisfied. "
                    "Consider increasing the distance "
                    "or decreasing the screen size."
                )

    def impulse_response_angular_spectrum(self) -> torch.Tensor:
        """Creates the impulse response function for angular spectrum method
        (zpASM and ASM)

        Returns
        -------
        torch.Tensor
            2d impulse response function for angular spectrum method
        """
        return torch.exp(
            (1j * self.distance) * self._wave_number_z
        )

    def impulse_response_fresnel(self) -> torch.Tensor:
        """Creates the impulse response function for fresnel approximation

        Returns
        -------
        torch.Tensor
            2d impulse response function for fresnel approximation
        """

        # Fourier image of impulse response function
        # 0 if k^2 < (kx^2 + ky^2) [if use_legacy_filter]
        return (
            self._low_pass_filter
            * torch.exp((1j * self.distance) * self._wave_number_z_eff_fresnel)
            * torch.exp((1j * self.distance) * self._wave_number)
        )

    def _impulse_response(self) -> torch.Tensor:
        """Calculates the impulse response function based on selected method

        Returns
        -------
        torch.Tensor
            The impulse response function
        """

        if self.method == "AS":
            return self.impulse_response_angular_spectrum()

        elif self.method == "fresnel":
            return self.impulse_response_fresnel()

            raise ValueError("Unknown forward propagation method")

    # TODO: ask for tol parameter, maybe move it to init?
    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """Calculates the field after propagating in the free space

        Parameters
        ----------
        input_field : Wavefront
            Field before propagation in free space

        Returns
        -------
        Wavefront
            Field after propagation in free space

        Raises
        ------
        ValueError
            Occurs when a non-existent direct distribution method is chosen
        """

        input_field_fft = torch.fft.fft2(
            incident_wavefront, dim=(self._y_index, self._x_index)
        )

        impulse_response_fft = self._impulse_response()

        # Fourier image of output field
        output_field_fft = input_field_fft * impulse_response_fft

        output_field = torch.fft.ifft2(
            output_field_fft, dim=(self._y_index, self._x_index)
        )

        return output_wavefront

    def reverse(self, transmission_wavefront: Wavefront) -> Wavefront:
        # TODO: Check the description...
        """Calculate the field after it propagates in the free space
        in the backward direction.

        Parameters
        ----------
        transmission_field : Wavefront
            Field to be propagated in the backward direction

        Returns
        -------
        Wavefront
            Propagated in the backward direction field
        """

        transmission_field_fft = torch.fft.fft2(
            transmission_wavefront, dim=(self._y_index, self._x_index)
        )

        impulse_response_fft = self._impulse_response().conj()

        # Fourier image of output field
        incident_field_fft = transmission_field_fft * impulse_response_fft

        incident_field = torch.fft.ifft2(
            incident_field_fft, dim=(self._y_index, self._x_index)
        )

        return incident_field

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [
            ParameterSpecs(
                "distance",
                [
                    PrettyReprRepr(self.distance),
                ],
            )
        ]

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template("widget_free_space.html.jinja").render(
            index=index, name=name, subelements=subelements
        )
