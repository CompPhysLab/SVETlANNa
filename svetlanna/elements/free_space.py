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


X_MESSAGE_ASM = (
    "In the ASM aliasing problems may occur. "
    "Consider reducing the distance "
    "or increasing the Nx*dx product or using zpAS method."
)

Y_MESSAGE_ASM = (
    "In the ASM aliasing problems may occur. "
    "Consider reducing the distance "
    "or increasing the Ny*dy product or using zpAS method."
)

X_MESSAGE_zpASM = (
    "In the zpASM method high-frequency noise may occur along the x direction. "
    "Consider reducing the distance or using the zpRSC/RSC method."
)

Y_MESSAGE_zpASM = (
    "In the zpASM method high-frequency noise may occur along the y direction. "
    "Consider reducing the distance or using the zpRSC/RSC method."
)

X_MESSAGE_zpRSC = (
    "In zpRSC method propagation distance is not large enough in the x direction."
    "It is preferable to use the zpASM or ASM method."
)

Y_MESSAGE_zpRSC = (
    "In zpRSC method propagation distance is not large enough in the y direction."
    "It is preferable to use the zpASM or ASM method."
)

X_MESSAGE_RSC = (
    "In RSC method propagation distance is not large enough in the x direction."
    "It is preferable to use the zpASM or ASM method."
)

Y_MESSAGE_RSC = (
    "In RSC method propagation distance is not large enough in the y direction."
    "It is preferable to use the zpASM or ASM method."
)


class FreeSpace(Element):
    """A class that describes a propagation of the wavefront in free space
    between two optical elements
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        distance: OptimizableFloat,
        method: Literal["ASM", "zpASM", "RSC", "zpRSC"],
        total_paddings_x: int | None = None,
        total_paddings_y: int | None = None,
    ):
        # TODO: rewrite docstrings

        super().__init__(simulation_parameters)

        self.distance = self.process_parameter("distance", distance)
        self.method = self.process_parameter("method", method)

        self._total_paddings_x = self.process_parameter(
            "_total_paddings_x", total_paddings_x
        )
        self._total_paddings_y = self.process_parameter(
            "_total_paddings_y", total_paddings_y
        )

        self._x_index = self.simulation_parameters.index("x")
        self._y_index = self.simulation_parameters.index("y")

        x = self.simulation_parameters.x
        y = self.simulation_parameters.y

        self._x = self.make_buffer("_x", x)
        self._y = self.make_buffer("_y", y)

        x_nodes = x.shape[0]
        y_nodes = y.shape[0]

        self._x_nodes = x_nodes
        self._y_nodes = y_nodes

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
            self._init_asm()
        elif self.method == "zpASM":
            self._init_zpasm()
        elif self.method == "RSC":
            self._init_rsc()
        elif self.method == "zpRSC":
            self._init_zprsc()

        self._propagation_methods = {
            "zpASM": self._propagate_by_zpasm,
            "ASM": self._propagate_by_asm,
            "zpRSC": self._propagate_by_zprsc,
            "RSC": self._propagate_by_rsc,
        }

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
        kx = 2 * torch.pi * torch.fft.fftfreq(x_nodes, dx, device=device)
        ky = 2 * torch.pi * torch.fft.fftfreq(y_nodes, dy, device=device)

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

    @staticmethod
    def _calculate_wave_number_z(
        wave_number: torch.Tensor, kx2ky2: torch.Tensor
    ) -> torch.Tensor:

        # other way
        wave_number_z = torch.sqrt(wave_number**2 - kx2ky2 + 0j)

        return wave_number_z

    @staticmethod
    def _validate_conditions(
        x_condition: torch.Tensor,
        y_condition: torch.Tensor,
        x_message: str,
        y_message: str,
    ) -> None:

        if not torch.all(x_condition):
            warn(x_message)
        if not torch.all(y_condition):
            warn(y_message)

    def _init_asm(self) -> None:

        kx2ky2, kx, ky = self._calculate_kx2ky2(
            simulation_parameters=self.simulation_parameters,
            sampling_intervals=(self._dy, self._dx),
            nodes=(self._y_nodes, self._x_nodes),
            device=self.simulation_parameters.device,
        )

        self._kx2ky2 = self.make_buffer("_kx2ky2", kx2ky2)
        self._kx = self.make_buffer("_kx", kx)
        self._ky = self.make_buffer("_ky", ky)

        # Warnings for fulfilling the method criteria
        # See (9.32), (9.36) in
        # Fourier Optics and Computational Imaging (2nd ed)
        # by Kedar Khare, Mansi Butola and Sunaina Rajor
        Lx = torch.abs(self._x[-1] - self._x[0])
        Ly = torch.abs(self._y[-1] - self._y[0])

        self._Lx = self.make_buffer("_Lx", Lx)
        self._Ly = self.make_buffer("_Ly", Ly)

        kx_max = torch.max(torch.abs(self._kx))
        ky_max = torch.max(torch.abs(self._ky))

        self._kx_max = self.make_buffer("_kx_max", kx_max)
        self._ky_max = self.make_buffer("_ky_max", ky_max)

        x_condition = self._kx_max >= self._wave_number / torch.sqrt(
            1 + (2 * self.distance / self._Lx) ** 2
        )
        y_condition = self._ky_max >= self._wave_number / torch.sqrt(
            1 + (2 * self.distance / self._Ly) ** 2
        )

        self._x_condition = self.make_buffer("_x_condition", x_condition)
        self._y_condition = self.make_buffer("_y_condition", y_condition)

        self._validate_conditions(
            x_condition=self._x_condition,
            y_condition=self._y_condition,
            x_message=X_MESSAGE_ASM,
            y_message=Y_MESSAGE_ASM,
        )

        wave_number_z = self._calculate_wave_number_z(
            wave_number=self._wave_number, kx2ky2=self._kx2ky2
        )

        # Registering Buffer for _wave_number_z
        self._wave_number_z = self.make_buffer("_wave_number_z", wave_number_z)

    def _init_zpasm(self) -> None:

        distances_x = self._calculate_critical_distances(
            nodes=self._x_nodes,
            total_paddings=self._x_nodes,
            wavelength=self.simulation_parameters.wavelength,
            d=self._dx,
        )

        distances_y = self._calculate_critical_distances(
            nodes=self._y_nodes,
            total_paddings=self._y_nodes,
            wavelength=self.simulation_parameters.wavelength,
            d=self._dy,
        )

        self._distances_x = self.make_buffer("_distances_x", distances_x)
        self._distances_y = self.make_buffer("_distances_y", distances_y)

        x_condition = self.distance <= distances_x
        y_condition = self.distance <= distances_y

        self._x_condition = self.make_buffer("_x_condition", x_condition)
        self._y_condition = self.make_buffer("_y_condition", y_condition)

        self._validate_conditions(
            x_condition=self._x_condition,
            y_condition=self._y_condition,
            x_message=X_MESSAGE_zpASM,
            y_message=Y_MESSAGE_zpASM,
        )

        if self._total_paddings_x is not None:
            self._x_paddings = (
                self._total_paddings_x // 2
                if self._total_paddings_x % 2 == 0
                else (self._total_paddings_x // 2 + 1)
            )
        else:
            self._x_paddings = (
                self._x_nodes // 2
                if self._x_nodes % 2 == 0
                else (self._x_nodes // 2 + 1)
            )

        if self._total_paddings_y is not None:
            self._y_paddings = (
                self._total_paddings_y // 2
                if self._total_paddings_y % 2 == 0
                else (self._total_paddings_y // 2 + 1)
            )
        else:
            self._y_paddings = (
                self._y_nodes // 2
                if self._y_nodes % 2 == 0
                else (self._y_nodes // 2 + 1)
            )

        ndim = max(-self._x_index, -self._y_index)

        self._padding_order = [0] * (2 * ndim)

        self._padding_order[-(2 * self._x_index + 1)] = self._x_paddings
        self._padding_order[-(2 * self._x_index + 2)] = self._x_paddings
        self._padding_order[-(2 * self._y_index + 1)] = self._y_paddings
        self._padding_order[-(2 * self._y_index + 2)] = self._y_paddings

        _upscaled_nodes = (
            self._y_nodes + 2 * self._y_paddings,
            self._x_nodes + 2 * self._x_paddings,
        )

        kx2ky2, _, _ = self._calculate_kx2ky2(
            simulation_parameters=self.simulation_parameters,
            sampling_intervals=(self._dy, self._dx),
            nodes=_upscaled_nodes,
            device=self.simulation_parameters.device,
        )
        self._kx2ky2 = self.make_buffer("_kx2ky2", kx2ky2)

        wave_number_z = self._calculate_wave_number_z(
            wave_number=self._wave_number, kx2ky2=self._kx2ky2
        )
        self._wave_number_z = self.make_buffer("_wave_number_z", wave_number_z)

    def _init_rsc(self) -> None:

        distances_x = self._calculate_critical_distances(
            nodes=self._x_nodes,
            total_paddings=0,
            wavelength=self.simulation_parameters.wavelength,
            d=self._dx,
        )

        distances_y = self._calculate_critical_distances(
            nodes=self._y_nodes,
            total_paddings=0,
            wavelength=self.simulation_parameters.wavelength,
            d=self._dy,
        )

        self._distances_x = self.make_buffer("_distances_x", distances_x)
        self._distances_y = self.make_buffer("_distances_y", distances_y)

        x_condition = self.distance >= self._distances_x
        y_condition = self.distance >= self._distances_y

        self._x_condition = self.make_buffer("_x_condition", x_condition)
        self._y_condition = self.make_buffer("_y_condition", y_condition)

        self._validate_conditions(
            x_condition=self._x_condition,
            y_condition=self._y_condition,
            x_message=X_MESSAGE_RSC,
            y_message=Y_MESSAGE_RSC,
        )

        Lx = torch.abs(self._x[-1] - self._x[0])
        Ly = torch.abs(self._y[-1] - self._y[0])

        self._Lx = self.make_buffer("_Lx", Lx)
        self._Ly = self.make_buffer("_Ly", Ly)

    def _init_zprsc(self) -> None:

        distances_x = self._calculate_critical_distances(
            nodes=self._x_nodes,
            total_paddings=self._x_nodes,
            wavelength=self.simulation_parameters.wavelength,
            d=self._dx,
        )

        distances_y = self._calculate_critical_distances(
            nodes=self._y_nodes,
            total_paddings=self._y_nodes,
            wavelength=self.simulation_parameters.wavelength,
            d=self._dy,
        )

        self._distances_x = self.make_buffer("_distances_x", distances_x)
        self._distances_y = self.make_buffer("_distances_y", distances_y)

        x_condition = self.distance >= self._distances_x
        y_condition = self.distance >= self._distances_y

        self._x_condition = self.make_buffer("_x_condition", x_condition)
        self._y_condition = self.make_buffer("_y_condition", y_condition)

        self._validate_conditions(
            x_condition=self._x_condition,
            y_condition=self._y_condition,
            x_message=X_MESSAGE_zpRSC,
            y_message=Y_MESSAGE_zpRSC,
        )

        ndim = max(-self._x_index, -self._y_index)

        self._padding_order = [0] * (2 * ndim)

        if self._total_paddings_x is not None:
            self._x_paddings = (
                self._total_paddings_x // 2
                if self._total_paddings_x % 2 == 0
                else (self._total_paddings_x // 2 + 1)
            )
        else:
            self._x_paddings = (
                self._x_nodes // 2
                if self._x_nodes % 2 == 0
                else (self._x_nodes // 2 + 1)
            )

        if self._total_paddings_y is not None:
            self._y_paddings = (
                self._total_paddings_y // 2
                if self._total_paddings_y % 2 == 0
                else (self._total_paddings_y // 2 + 1)
            )
        else:
            self._y_paddings = (
                self._y_nodes // 2
                if self._y_nodes % 2 == 0
                else (self._y_nodes // 2 + 1)
            )

        self._padding_order[-(2 * self._x_index + 1)] = self._x_paddings
        self._padding_order[-(2 * self._x_index + 2)] = self._x_paddings
        self._padding_order[-(2 * self._y_index + 1)] = self._y_paddings
        self._padding_order[-(2 * self._y_index + 2)] = self._y_paddings

    def _propagate_by_asm(self, wavefront: Wavefront) -> Wavefront:

        wavefront_fft = torch.fft.fft2(wavefront, dim=(self._y_index, self._x_index))

        impulse_response_fft = self._impulse_response_angular_spectrum(
            distance=self.distance, wave_number_z=self._wave_number_z
        )

        output_wavefront_fft = wavefront_fft * impulse_response_fft

        output_wavefront = torch.fft.ifft2(
            output_wavefront_fft, dim=(self._y_index, self._x_index)
        )

        return Wavefront(output_wavefront)

    def _propagate_by_zpasm(self, wavefront: Wavefront) -> Wavefront:

        # add paddings
        wavefront_padded = Wavefront(
            F.pad(wavefront, self._padding_order, mode="constant", value=0)
        )

        output_wavefront_padded = self._propagate_by_asm(wavefront=wavefront_padded)

        # remove added paddings
        y_start = self._y_paddings
        y_end = output_wavefront_padded.shape[self._y_index] - self._y_paddings
        x_start = self._x_paddings
        x_end = output_wavefront_padded.shape[self._x_index] - self._x_paddings

        slices = [slice(None)] * output_wavefront_padded.ndim
        slices[self._y_index] = slice(y_start, y_end)
        slices[self._x_index] = slice(x_start, x_end)

        # apply slices to remove paddings
        output_wavefront = output_wavefront_padded[tuple(slices)]

        return Wavefront(output_wavefront)

    def _propagate_by_rsc(self, wavefront: Wavefront) -> Wavefront:

        wavefront_fft = torch.fft.fft2(wavefront, dim=(self._y_index, self._x_index))

        transfer_function = (
            self._transfer_function_rsc(
                simulation_parameters=self.simulation_parameters,
                wave_number=self._wave_number,
                nodes=(self._y_nodes, self._x_nodes),
                size=(self._Ly, self._Lx),
                distance=self.distance,
            )
            * self._dx
            * self._dy
        )

        impulse_response_fft = torch.fft.fft2(
            transfer_function, dim=(self._y_index, self._x_index)
        )

        output_wavefront_fft = wavefront_fft * impulse_response_fft

        output_wavefront = torch.fft.ifftshift(
            torch.fft.ifft2(output_wavefront_fft, dim=(self._y_index, self._x_index))
        )
        return Wavefront(output_wavefront)

    def _propagate_by_zprsc(self, wavefront: Wavefront) -> Wavefront:

        # add paddings
        wavefront_padded = F.pad(
            wavefront, self._padding_order, mode="constant", value=0
        )

        wavefront_padded_fft = torch.fft.fft2(
            wavefront_padded, dim=(self._y_index, self._x_index)
        )

        x_nodes_padded = wavefront_padded_fft.shape[self._x_index]
        y_nodes_padded = wavefront_padded_fft.shape[self._y_index]
        _size = (self._dy * y_nodes_padded, self._dx * x_nodes_padded)

        transfer_function_padded = (
            self._transfer_function_rsc(
                simulation_parameters=self.simulation_parameters,
                wave_number=self._wave_number,
                nodes=(y_nodes_padded, x_nodes_padded),
                size=_size,
                distance=self.distance,
            )
            * self._dx
            * self._dy
        )

        impulse_response_fft_padded = torch.fft.fft2(
            transfer_function_padded, dim=(self._y_index, self._x_index)
        )

        output_wavefront_padded_fft = wavefront_padded_fft * impulse_response_fft_padded

        output_wavefront_padded = torch.fft.ifftshift(
            torch.fft.ifft2(
                output_wavefront_padded_fft, dim=(self._y_index, self._x_index)
            )
        )

        y_start = self._y_paddings
        y_end = output_wavefront_padded.shape[self._y_index] - self._y_paddings
        x_start = self._x_paddings
        x_end = output_wavefront_padded.shape[self._x_index] - self._x_paddings

        slices = [slice(None)] * output_wavefront_padded.ndim
        slices[self._y_index] = slice(y_start, y_end)
        slices[self._x_index] = slice(x_start, x_end)

        # apply slices to remove paddings
        output_wavefront = output_wavefront_padded[tuple(slices)]

        return Wavefront(output_wavefront)

    def _propagate_wavefront(self, wavefront: Wavefront, method: str) -> Wavefront:

        propagation_function = self._propagation_methods.get(method)

        if propagation_function is None:
            raise ValueError(f"Unknown forward propagation method: {method}")

        return propagation_function(wavefront)

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
