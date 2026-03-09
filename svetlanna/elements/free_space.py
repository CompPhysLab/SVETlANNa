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

import matplotlib.pyplot as plt


class FreeSpace(Element):
    """A class that describes a propagation of the wavefront in free space
    between two optical elements
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        distance: OptimizableFloat,
        method: Literal["ASM", "zpASM", "RSC", "zpRSC"],
    ):
        # TODO: rewrite docstrings

        super().__init__(simulation_parameters)

        self.distance = self.process_parameter("distance", distance)
        self.method = self.process_parameter("method", method)

        self._x_index = self.simulation_parameters.index("x")
        self._y_index = self.simulation_parameters.index("y")

        x = self.simulation_parameters.x
        y = self.simulation_parameters.y

        self._x = self.make_buffer("_x", x)
        self._y = self.make_buffer("_y", y)

        x_nodes = x.shape[0]
        y_nodes = y.shape[0]

        # Compute spatial grid spacing
        dx = (x[1] - x[0]) if x_nodes > 1 else torch.tensor([1.0])
        dy = (y[1] - y[0]) if y_nodes > 1 else torch.tensor([1.0])

        self._dx = self.make_buffer("_dx", dx)
        self._dy = self.make_buffer("_dy", dy)

        # Calculate wave number
        wave_number = self.simulation_parameters.cast(
            2 * torch.pi / self.simulation_parameters.wavelength, "wavelength"
        )

        # Registering Buffer for _wave_number
        self._wave_number = self.make_buffer("_wave_number", wave_number)

        if self.method == "ASM":
            _nodes = (
                self._y.shape[0],
                self._x.shape[0],
            )

            kx2ky2, kx, ky = self._calculate_kx2ky2(
                simulation_parameters=self.simulation_parameters,
                sampling_intervals=(self._dy, self._dx),
                nodes=_nodes,
                device=self.simulation_parameters.device,
            )

            self._kx2ky2 = self.make_buffer("_kx2ky2", kx2ky2)

            # Warnings for fulfilling the method criteria
            # See (9.32), (9.36) in
            # Fourier Optics and Computational Imaging (2nd ed)
            # by Kedar Khare, Mansi Butola and Sunaina Rajor
            Lx = torch.abs(self._x[-1] - self._x[0])
            Ly = torch.abs(self._y[-1] - self._y[0])

            self._Lx = self.make_buffer("_Lx", Lx)
            self._Ly = self.make_buffer("_Ly", Ly)

            kx_max = torch.max(torch.abs(kx))
            ky_max = torch.max(torch.abs(ky))

            x_condition = kx_max >= self._wave_number / torch.sqrt(
                1 + (2 * self.distance / Lx) ** 2
            )
            y_condition = ky_max >= self._wave_number / torch.sqrt(
                1 + (2 * self.distance / Ly) ** 2
            )

            if not torch.all(x_condition):
                warn(
                    "In the ASM aliasing problems may occur. "
                    "Consider reducing the distance "
                    "or increasing the Nx*dx product or using zpAS method."
                )
            if not torch.all(y_condition):
                warn(
                    "In the ASM aliasing problems may occur. "
                    "Consider reducing the distance "
                    "or increasing the Ny*dy product or using zpAS method."
                )

            kz_squared = torch.pow(self._wave_number.abs(), 2) - self._kx2ky2

            kz_squared_cond = kz_squared >= 0

            # Calculate kz
            wave_number_z_non_negative = torch.sqrt(
                kz_squared * kz_squared_cond + 0j
            )  # 0j is required to convert argument to complex

            # take into account evanescent waves
            wave_number_z_negative = 1j * torch.sqrt(
                -kz_squared * ~kz_squared_cond + 0j
            )

            wave_number_z = wave_number_z_non_negative + wave_number_z_negative

            # Registering Buffer for _wave_number_z
            self._wave_number_z = self.make_buffer("_wave_number_z", wave_number_z)

        if self.method == "zpRSC":

            distances_x = self._calculate_critical_distances(
                nodes=x_nodes,
                total_paddings=x_nodes,
                wavelength=self.simulation_parameters.wavelength,
                d=self._dx,
            )

            distances_y = self._calculate_critical_distances(
                nodes=y_nodes,
                total_paddings=y_nodes,
                wavelength=self.simulation_parameters.wavelength,
                d=self._dy,
            )

            x_condition = self.distance >= distances_x
            y_condition = self.distance >= distances_y

            if not torch.all(x_condition):
                warn(
                    "In zpRSC method propagation distance is not large enough in the x direction."  # noqa: E501
                    "It is preferable to use the zpASM or ASM method."
                )

            if not torch.all(y_condition):
                warn(
                    "In zpRSC method propagation distance is not large enough in the y direction."  # noqa: E501
                    "It is preferable to use the zpASM or ASM method."
                )

            ndim = max(-self._x_index, -self._y_index)

            padding_order = [0] * (2 * ndim)

            self._x_paddings = x_nodes // 2 if x_nodes % 2 == 0 else (x_nodes // 2 + 1)
            self._y_paddings = y_nodes // 2 if y_nodes % 2 == 0 else (y_nodes // 2 + 1)

            padding_order[-(2 * self._x_index + 1)] = self._x_paddings
            padding_order[-(2 * self._x_index + 2)] = self._x_paddings
            padding_order[-(2 * self._y_index + 1)] = self._y_paddings
            padding_order[-(2 * self._y_index + 2)] = self._y_paddings

            self._padding_order = padding_order

            self._nodes = (y_nodes, x_nodes)

            self._size = (self._dy * self._nodes[0], self._dx * self._nodes[1])

        if self.method == "RSC":

            distances_x = self._calculate_critical_distances(
                nodes=x_nodes,
                total_paddings=0,
                wavelength=self.simulation_parameters.wavelength,
                d=self._dx,
            )

            distances_y = self._calculate_critical_distances(
                nodes=y_nodes,
                total_paddings=0,
                wavelength=self.simulation_parameters.wavelength,
                d=self._dy,
            )

            x_condition = self.distance >= distances_x
            y_condition = self.distance >= distances_y

            if not torch.all(x_condition):
                warn(
                    "In RSC method propagation distance is not large enough in the x direction."  # noqa: E501
                    "It is preferable to use the zpASM or ASM method."
                )

            if not torch.all(y_condition):
                warn(
                    "In RSC method propagation distance is not large enough in the y direction."  # noqa: E501
                    "It is preferable to use the zpASM or ASM method."
                )

            self._nodes = (y_nodes, x_nodes)

            Lx = torch.abs(self._x[-1] - self._x[0])
            Ly = torch.abs(self._y[-1] - self._y[0])

            self._size = (Ly, Lx)

        if self.method == "zpASM":
            _nodes = (
                self._y.shape[0],
                self._x.shape[0],
            )

            distances_x = self._calculate_critical_distances(
                nodes=_nodes[1],
                total_paddings=_nodes[1],
                wavelength=self.simulation_parameters.wavelength,
                d=self._dx,
            )

            distances_y = self._calculate_critical_distances(
                nodes=_nodes[0],
                total_paddings=_nodes[0],
                wavelength=self.simulation_parameters.wavelength,
                d=self._dy,
            )

            x_condition = self.distance <= distances_x
            y_condition = self.distance <= distances_y

            if not torch.all(x_condition):
                warn(
                    "In the zpASM method high-frequency noise may occur in the x direction. "
                    "Consider reducing the distance or using the zpRSC/RSC method."
                )

            if not torch.all(y_condition):
                warn(
                    "In the zpASM method high-frequency noise may occur in the y direction. "
                    "Consider reducing the distance or using the zpRSC/RSC method."
                )

    @staticmethod
    def _calculate_kx2ky2(
        simulation_parameters: SimulationParameters,
        sampling_intervals: Tuple[torch.Tensor, torch.Tensor],
        nodes: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        y_nodes, x_nodes = nodes
        dy, dx = sampling_intervals

        # Compute wave vectors
        kx = (
            2
            * torch.pi
            * torch.fft.fftshift(torch.fft.fftfreq(x_nodes, dx, device=device))
        )
        ky = (
            2
            * torch.pi
            * torch.fft.fftshift(torch.fft.fftfreq(y_nodes, dy, device=device))
        )

        # Compute wave vectors grids
        _kx = simulation_parameters.cast(kx, "x", shape_check=False)
        _ky = simulation_parameters.cast(ky, "y", shape_check=False)

        kx2ky2 = _kx**2 + _ky**2

        return kx2ky2, _kx, _ky

    @staticmethod
    def _calculate_paddings(
        wavelength: torch.Tensor,
        distance: OptimizableFloat,
        sampling_interval: torch.Tensor,
    ) -> int:

        total_num_of_paddings = torch.max(
            distance
            * wavelength
            / (
                2
                * sampling_interval**2
                * torch.sqrt(1 - (wavelength / (2 * sampling_interval)) ** 2)
            )
        )

        num_of_paddings_rounded = torch.ceil(total_num_of_paddings / 2).to(torch.int)
        return int(num_of_paddings_rounded.item())

    @staticmethod
    def _impulse_response_angular_spectrum(
        distance: OptimizableFloat, wave_number_z: torch.Tensor
    ) -> torch.Tensor:

        return torch.exp(1j * distance * wave_number_z)

    @staticmethod
    def _transfer_function_rsc(
        simulation_parameters: SimulationParameters,
        wave_number: torch.Tensor,
        nodes: Tuple[int, int],
        size: Tuple[torch.Tensor, torch.Tensor],
        distance: OptimizableFloat,
    ):
        y_nodes, x_nodes = nodes

        Ly, Lx = size

        x = torch.linspace(
            -Lx / 2, Lx / 2, x_nodes, device=simulation_parameters.device
        )
        y = torch.linspace(
            -Ly / 2, Ly / 2, y_nodes, device=simulation_parameters.device
        )

        _x = simulation_parameters.cast(x, "x", shape_check=False)
        _y = simulation_parameters.cast(y, "y", shape_check=False)

        r = torch.sqrt(_x**2 + _y**2 + distance**2)

        kr = wave_number * r

        e_ikr = torch.exp(1j * kr)

        spherical_wave = e_ikr / (2 * torch.pi * r)

        paraxial_factor = distance / r

        spherical_wave_factored = spherical_wave * paraxial_factor

        first_factor = spherical_wave_factored * -1j * wave_number

        second_factor = spherical_wave_factored * (1 / r)

        transfer_function = first_factor + second_factor

        return transfer_function

    @staticmethod
    def _calculate_critical_distances(
        nodes: int, total_paddings: int, wavelength: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        distances = (
            (nodes + total_paddings)
            * torch.sqrt(1 - (wavelength / (2 * d)) ** 2)
            * (d**2)
            / wavelength
        )

        return distances

    def _propagate_wavefront(self, wavefront: Wavefront, method: str) -> Wavefront:
        if method == "zpASM":

            # calculate paddings for zero-padding AS method
            x_paddings = self._calculate_paddings(
                wavelength=self.simulation_parameters.wavelength,
                distance=self.distance,
                sampling_interval=self._dx,
            )
            y_paddings = self._calculate_paddings(
                wavelength=self.simulation_parameters.wavelength,
                distance=self.distance,
                sampling_interval=self._dy,
            )

            ndim = max(-self._x_index, -self._y_index)

            padding_order = [0] * (2 * ndim)

            padding_order[-(2 * self._x_index + 1)] = x_paddings
            padding_order[-(2 * self._x_index + 2)] = x_paddings
            padding_order[-(2 * self._y_index + 1)] = y_paddings
            padding_order[-(2 * self._y_index + 2)] = y_paddings

            # add paddings
            wavefront_padded = F.pad(wavefront, padding_order, mode="constant", value=0)

            # spectrum with zero frequency components in the center
            wavefront_padded_fft = torch.fft.fftshift(
                torch.fft.fft2(wavefront_padded, dim=(self._y_index, self._x_index))
            )

            _nodes = (
                self.simulation_parameters.y.shape[0] + 2 * y_paddings,
                self.simulation_parameters.x.shape[0] + 2 * x_paddings,
            )

            kx2ky2, _, _ = self._calculate_kx2ky2(
                simulation_parameters=self.simulation_parameters,
                sampling_intervals=(self._dy, self._dx),
                nodes=_nodes,
                device=self.simulation_parameters.device,
            )

            kz_squared = torch.pow(self._wave_number.abs(), 2) - kx2ky2

            kz_squared_cond = kz_squared >= 0

            # Calculate kz
            wave_number_z_non_negative = torch.sqrt(
                kz_squared * kz_squared_cond + 0j
            )  # 0j is required to convert argument to complex

            # take into account evanescent waves
            wave_number_z_negative = 1j * torch.sqrt(
                -kz_squared * ~kz_squared_cond + 0j
            )

            wave_number_z = wave_number_z_non_negative + wave_number_z_negative

            impulse_response_fft_padded = self._impulse_response_angular_spectrum(
                distance=self.distance, wave_number_z=wave_number_z
            )

            output_wavefront_padded_fft = (
                wavefront_padded_fft * impulse_response_fft_padded
            )

            output_wavefront_padded = torch.fft.ifft2(
                output_wavefront_padded_fft, dim=(self._y_index, self._x_index)
            )

            # remove added paddings
            y_start = y_paddings
            y_end = output_wavefront_padded.shape[self._y_index] - y_paddings
            x_start = x_paddings
            x_end = output_wavefront_padded.shape[self._x_index] - x_paddings

            slices = [slice(None)] * output_wavefront_padded.ndim
            slices[self._y_index] = slice(y_start, y_end)
            slices[self._x_index] = slice(x_start, x_end)

            # Apply slices to remove paddings
            output_wavefront = output_wavefront_padded[tuple(slices)]

        elif method == "ASM":

            wavefront_fft = torch.fft.fftshift(
                torch.fft.fft2(wavefront, dim=(self._y_index, self._x_index))
            )

            impulse_response_fft = self._impulse_response_angular_spectrum(
                distance=self.distance, wave_number_z=self._wave_number_z
            )

            output_wavefront_fft = wavefront_fft * impulse_response_fft

            output_wavefront = torch.fft.ifft2(
                output_wavefront_fft, dim=(self._y_index, self._x_index)
            )

        elif method == "zpRSC":

            # add paddings
            wavefront_padded = F.pad(
                wavefront, self._padding_order, mode="constant", value=0
            )

            wavefront_padded_fft = torch.fft.fft2(
                wavefront_padded, dim=(self._y_index, self._x_index)
            )
            print("sim params device:", self.simulation_parameters.device)
            transfer_function = (
                self._transfer_function_rsc(
                    simulation_parameters=self.simulation_parameters,
                    wave_number=self._wave_number,
                    nodes=self._nodes,
                    size=self._size,
                    distance=self.distance,
                )
                * self._dx
                * self._dy
            )

            transfer_function_padded = F.pad(
                transfer_function, self._padding_order, mode="constant", value=0
            )

            # transfer_function_padded = transfer_function

            impulse_response_fft_padded = torch.fft.fft2(
                transfer_function_padded, dim=(self._y_index, self._x_index)
            )

            output_wavefront_padded_fft = (
                wavefront_padded_fft * impulse_response_fft_padded
            )

            output_wavefront_padded = torch.fft.fftshift(
                torch.fft.ifft2(
                    output_wavefront_padded_fft, dim=(self._y_index, self._x_index)
                ),
                dim=(self._y_index, self._x_index),
            )

            y_start = self._y_paddings
            y_end = output_wavefront_padded.shape[self._y_index] - self._y_paddings
            x_start = self._x_paddings
            x_end = output_wavefront_padded.shape[self._x_index] - self._x_paddings

            slices = [slice(None)] * output_wavefront_padded.ndim
            slices[self._y_index] = slice(y_start, y_end)
            slices[self._x_index] = slice(x_start, x_end)

            # Apply slices to remove paddings
            output_wavefront = output_wavefront_padded[tuple(slices)]

        elif method == "RSC":

            wavefront_fft = torch.fft.fft2(
                wavefront, dim=(self._y_index, self._x_index)
            )

            transfer_function = (
                self._transfer_function_rsc(
                    simulation_parameters=self.simulation_parameters,
                    wave_number=self._wave_number,
                    nodes=self._nodes,
                    size=self._size,
                    distance=self.distance,
                )
                * self._dx
                * self._dy
            )

            impulse_response_fft = torch.fft.fft2(
                transfer_function, dim=(self._y_index, self._x_index)
            )

            output_wavefront_fft = wavefront_fft * impulse_response_fft

            output_wavefront = torch.fft.fftshift(
                torch.fft.ifft2(
                    output_wavefront_fft, dim=(self._y_index, self._x_index)
                ),
                dim=(self._y_index, self._x_index),
            )

        else:

            raise ValueError("Unknown forward propagation method")

        return output_wavefront

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """Calculates the wavefront after propagating in the free space

        Parameters
        ----------
        incident_wavefront : Wavefront
            Wavefront before propagation in free space

        Returns
        -------
        Wavefront
            Wavefront after propagation in free space

        Raises
        ------
        ValueError
            Occurs when a non-existent direct distribution method is chosen
        """

        output_wavefront = self._propagate_wavefront(
            wavefront=incident_wavefront, method=self.method
        )

        return output_wavefront

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
