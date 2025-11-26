from typing import Literal, Iterable, Tuple
import torch
import torch.nn.functional as F
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront
from ..axes_math import tensor_dot
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

        self.distance = self.process_parameter('distance', distance)
        self.method = self.process_parameter('method', method)

        # params extracted from SimulationParameters
        self._device = self.simulation_parameters.device

        self._w_index = self.simulation_parameters.axes.index('W')
        self._h_index = self.simulation_parameters.axes.index('H')

        _x_linear = self.simulation_parameters.axes.W
        _y_linear = self.simulation_parameters.axes.H

        self._x_linear = self.make_buffer('_x_linear', _x_linear)
        self._y_linear = self.make_buffer('_y_linear', _y_linear)

        self._x_nodes = self._x_linear.shape[0]
        self._y_nodes = self._y_linear.shape[0]

        # Compute spatial grid spacing
        _sampling_interval_x = (self._x_linear[1] - self._x_linear[0])
        _sampling_interval_y = (self._y_linear[1] - self._y_linear[0])

        self._sampling_interval_x = self.make_buffer(
            '_sampling_interval_x', _sampling_interval_x
        )

        self._sampling_interval_y = self.make_buffer(
            '_sampling_interval_y', _sampling_interval_y
        )

        # Calculate wave vector of shape ('wavelength') or ()
        _k = 2 * torch.pi / self.simulation_parameters.axes.wavelength
        self._k = self.make_buffer('_k', _k)

        # Reshape wave vector for further calculations
        wave_number = self._k[..., None, None]  # shape: ('wavelength', 1, 1) or (1, 1)  # noqa
        # Registering Buffer for _wave_number
        self._wave_number = self.make_buffer(
            '_wave_number', wave_number
        )

        _Lx = torch.abs(self._x_linear[-1] - self._x_linear[0])
        _Ly = torch.abs(self._y_linear[-1] - self._y_linear[0])

        self._Lx = self.make_buffer('_Lx', _Lx)
        self._Ly = self.make_buffer('_Ly', _Ly)

    def _calculate_kx2ky2(
        self,
        sampling_interval: Tuple[torch.Tensor, torch.Tensor],
        nodes: Tuple[int, int],
        device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        _x_nodes, _y_nodes = nodes
        _dx, _dy = sampling_interval

        # Compute wave vectors
        kx_linear = 2 * torch.pi * torch.fft.fftshift(
            torch.fft.fftfreq(
                _x_nodes, _dx, device=device
            )
        )
        ky_linear = 2 * torch.pi * torch.fft.fftshift(
            torch.fft.fftfreq(
                _y_nodes, _dy, device=device
            )
        )

        # Compute wave vectors grids
        kx_grid = kx_linear[None, :]  # shape: (1, 'W')
        ky_grid = ky_linear[:, None]  # shape: ('H', 1)

        # Calculate (kx^2+ky^2) tensor
        kx2ky2 = kx_grid ** 2 + ky_grid ** 2  # shape: ('H', 'W')
        return kx2ky2, kx_linear, ky_linear

    def _calculate_paddings(
        self,
        sampling_interval: torch.Tensor
    ) -> int:
        _wavelength = self.simulation_parameters.axes.wavelength
        num_of_paddings = torch.max(
            self.distance * _wavelength / (
                2 * sampling_interval ** 2 * torch.sqrt(
                    1 - (_wavelength / (2 * sampling_interval))**2
                )
            )
        )
        num_of_paddings_rounded = torch.ceil(num_of_paddings / 2).to(torch.int)
        return int(num_of_paddings_rounded.item())

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
        return torch.exp(
            (1j * self.distance) * self._wave_number_z_eff_fresnel
        ) * torch.exp(
            (1j * self.distance) * self._wave_number
        )

    def transfer_function_rsc(
        self, x_nodes: int, y_nodes: int
    ) -> tuple[torch.Tensor, tuple[str, ...]]:

        x_linear = torch.linspace(
            -self._Lx / 2, self._Lx / 2, x_nodes
        )
        y_linear = torch.linspace(
            -self._Ly / 2, self._Ly / 2, y_nodes
        )

        # Compute wave vectors grids
        x_grid = x_linear[None, :]  # shape: (1, 'W')
        y_grid = y_linear[:, None]  # shape: ('H', 1)

        r = torch.sqrt(
            x_grid ** 2 + y_grid ** 2 + self.distance ** 2 + 1e-5
        )  # shape: ('H', 'W')

        # Calculate (kx^2+ky^2) / k^2 relation
        kr, kr_axes = tensor_dot(
            a=self._k,
            b=r,
            a_axis='wavelength',
            b_axis=('H', 'W')
        )  # shape: ('wavelength', 'H', 'W') or ('H', 'W') depending on k shape # noqa: E501

        e_ikr = torch.exp(1j * kr)

        spherical_wave, sw_axes = tensor_dot(
            a=e_ikr,
            b=1 / (2 * torch.pi * r),
            a_axis=kr_axes,
            b_axis=('H', 'W'),
            preserve_a_axis=True
        )

        paraxial_factor = self.distance / r

        spherical_wave_factored, sw_f_axes = tensor_dot(
            a=spherical_wave,
            b=paraxial_factor,
            a_axis=sw_axes,
            b_axis=('H', 'W'),
            preserve_a_axis=True
        )

        first_factor = spherical_wave_factored * -1j * self._wave_number

        second_factor, ax = tensor_dot(
            a=spherical_wave_factored,
            b=1 / r,
            a_axis=sw_f_axes,
            b_axis=('H', 'W'),
            preserve_a_axis=True
        )

        transfer_function = first_factor + second_factor

        return transfer_function, ax

    def _propagate_wavefront(
        self,
        wavefront: Wavefront,
        method: str
    ) -> Wavefront:
        if method == 'zpASM':

            # calculate paddings for zero-padding AS method
            x_paddings = self._calculate_paddings(
                sampling_interval=self._sampling_interval_x
            )
            y_paddings = self._calculate_paddings(
                sampling_interval=self._sampling_interval_y
            )

            if x_paddings > self._x_nodes / 2:
                warn(
                    (
                        'In zpASM number of paddings in x direction '
                        + f'({str(x_paddings)})'
                        + ' exceeds the number of nodes. It is preferable to use the RSC method'    # noqa: E501
                    )
                )

            if y_paddings > self._y_nodes / 2:
                warn(
                    (
                        'In zpASM number of paddings in y direction '
                        + f'({str(y_paddings)})'
                        + ' exceeds the number of nodes. It is preferable to use the RSC method'    # noqa: E501
                    )
                )

            padding_order = [0] * (2 * wavefront.ndim)

            padding_order[-(2 * self._w_index + 1)] = x_paddings
            padding_order[-(2 * self._w_index + 2)] = x_paddings
            padding_order[-(2 * self._h_index + 1)] = y_paddings
            padding_order[-(2 * self._h_index + 2)] = y_paddings

            # add paddings
            wavefront_padded = F.pad(
                wavefront, padding_order, mode='constant', value=0
            )    # noqa: E501

            # spectrum with zero frequency components in the center
            wavefront_padded_fft = torch.fft.fftshift(
                torch.fft.fft2(
                    wavefront_padded,
                    dim=(self._h_index, self._w_index)
                )
            )

            # Calculate (kx^2+ky^2) tensor
            kx2ky2, _, _ = self._calculate_kx2ky2(
                sampling_interval=(self._sampling_interval_x, self._sampling_interval_y),   # noqa: E501
                nodes=(self._x_nodes + 2 * x_paddings, self._y_nodes + 2 * y_paddings),    # noqa: E501
                device=self._device
            )   # shape: ('H', 'W')

            # Calculate (kx^2+ky^2) / k^2 relation
            _, relation_axes = tensor_dot(
                a=1 / (self._k ** 2),
                b=kx2ky2,
                a_axis='wavelength',
                b_axis=('H', 'W')
            )  # shape: ('wavelength', 'H', 'W') or ('H', 'W') depending on k shape # noqa: E501

            kx2ky2_cond = self._wave_number ** 2 - kx2ky2 >= 0
            # Calculate kz
            wave_number_z_non_negative = torch.sqrt(
                self._wave_number ** 2 - kx2ky2 * kx2ky2_cond + 0j
            )  # 0j is required to convert argument to complex

            # take into account evanescent waves
            wave_number_z_negative = 1j * torch.sqrt(
                kx2ky2 * ~kx2ky2_cond + 0j - self._wave_number ** 2
            )

            wave_number_z = wave_number_z_non_negative + wave_number_z_negative
            # Registering Buffer for _wave_number_z
            self._wave_number_z = self.make_buffer(
                '_wave_number_z', wave_number_z
            )

            self.impulse_response_fft = self.impulse_response_angular_spectrum()
            self.transfer_function = torch.fft.ifft2(self.impulse_response_fft)

            output_wavefront_padded_fft, _ = tensor_dot(
                a=wavefront_padded_fft,  # example shape: (5, 'wavelength', 1, 'H', 'W')    # noqa: E501
                b=self.impulse_response_fft,  # example shape: ('wavelength', 'H', 'W')  # noqa: E501
                a_axis=self.simulation_parameters.axes.names,
                b_axis=relation_axes,
                preserve_a_axis=True  # check that the output has the input shape   # noqa: E501
            )  # example output shape: (5, 'wavelength', 1, 'H', 'W')

            output_wavefront_padded = torch.fft.ifft2(
                output_wavefront_padded_fft,
                dim=(self._h_index, self._w_index)
            )

            # remove added paddings
            h_start = y_paddings
            h_end = output_wavefront_padded.shape[self._h_index] - y_paddings
            w_start = x_paddings
            w_end = output_wavefront_padded.shape[self._w_index] - x_paddings

            slices = [slice(None)] * output_wavefront_padded.ndim
            slices[self._h_index] = slice(h_start, h_end)
            slices[self._w_index] = slice(w_start, w_end)

            # Apply slices to remove paddings
            output_wavefront = output_wavefront_padded[tuple(slices)]

        elif method == 'ASM':

            wavefront_fft = torch.fft.fftshift(
                torch.fft.fft2(
                    wavefront,
                    dim=(self._h_index, self._w_index)
                )
            )

            # Calculate (kx^2+ky^2) tensor
            kx2ky2, kx_linear, ky_linear = self._calculate_kx2ky2(
                sampling_interval=(self._sampling_interval_x, self._sampling_interval_y),   # noqa: E501
                nodes=(self._x_nodes, self._y_nodes),
                device=self._device
            )   # shape: ('H', 'W')

            kx_max = torch.max(torch.abs(kx_linear))
            ky_max = torch.max(torch.abs(ky_linear))
            x_condition = kx_max >= self._k / torch.sqrt(1 + (2*self.distance / self._Lx)**2)    # noqa: E501
            y_condition = ky_max >= self._k / torch.sqrt(1 + (2*self.distance / self._Ly)**2)    # noqa: E501

            if not torch.all(x_condition):
                warn(
                    'In the ASM aliasing problems may occur. '
                    'Consider reducing the distance '
                    'or increasing the Nx*dx product or using zpAS method.'
                )
            if not torch.all(y_condition):
                warn(
                    'In the ASM aliasing problems may occur. '
                    'Consider reducing the distance '
                    'or increasing the Ny*dy product or using zpAS method.'
                )

            # Calculate (kx^2+ky^2) / k^2 relation
            _, relation_axes = tensor_dot(
                a=1 / (self._k ** 2),
                b=kx2ky2,
                a_axis='wavelength',
                b_axis=('H', 'W')
            )  # shape: ('wavelength', 'H', 'W') or ('H', 'W') depending on k shape # noqa: E501

            kx2ky2_cond = self._wave_number ** 2 - kx2ky2 >= 0
            # Calculate kz
            wave_number_z_non_negative = torch.sqrt(
                self._wave_number ** 2 - kx2ky2 * kx2ky2_cond + 0j
            )  # 0j is required to convert argument to complex

            # take into account evanescent waves
            wave_number_z_negative = 1j * torch.sqrt(
                kx2ky2 * ~kx2ky2_cond + 0j - self._wave_number ** 2
            )

            wave_number_z = wave_number_z_non_negative + wave_number_z_negative

            # Calculate kz
            wave_number_z = torch.sqrt(
                self._wave_number ** 2 - kx2ky2 + 0j
            )  # 0j is required to convert argument to complex

            # Registering Buffer for _wave_number_z
            self._wave_number_z = self.make_buffer(
                '_wave_number_z', wave_number_z
            )

            self.impulse_response_fft = self.impulse_response_angular_spectrum()
            self.transfer_function = torch.fft.ifft2(self.impulse_response_fft)

            output_wavefront_fft, _ = tensor_dot(
                a=wavefront_fft,  # example shape: (5, 'wavelength', 1, 'H', 'W')   # noqa: E501
                b=self.impulse_response_fft,  # example shape: ('wavelength', 'H', 'W')  # noqa: E501
                a_axis=self.simulation_parameters.axes.names,
                b_axis=relation_axes,
                preserve_a_axis=True  # check that the output has the input shape   # noqa: E501
            )  # example output shape: (5, 'wavelength', 1, 'H', 'W')

            output_wavefront = torch.fft.ifft2(
                output_wavefront_fft,
                dim=(self._h_index, self._w_index)
            )

        elif method == 'fresnel-IR':
            wavefront_fft = torch.fft.fft2(
                wavefront,
                dim=(self._h_index, self._w_index)
            )
            # Calculate (kx^2+ky^2) tensor
            kx2ky2, kx_linear, ky_linear = self._calculate_kx2ky2(
                sampling_interval=(self._sampling_interval_x, self._sampling_interval_y),   # noqa: E501
                nodes=(self._x_nodes, self._y_nodes),
                device=self._device
            )   # shape: ('H', 'W')

            # Calculate kz taylored, used by Fresnel approximation
            wave_number_z_eff_fresnel = - 0.5 * kx2ky2 / self._wave_number

            # Registering Buffer for _wave_number_z_eff_fresnel
            self._wave_number_z_eff_fresnel = self.make_buffer(
                '_wave_number_z_eff_fresnel', wave_number_z_eff_fresnel
            )

            diagonal_squared = self._Lx**2 + self._Ly**2
            condition = self.distance**3 > self._k / 8 * (diagonal_squared)**2

            if not torch.all(condition):
                warn(
                    'The paraxial (near-axis) optics condition '
                    'required for the Fresnel method is not satisfied. '
                    'Consider increasing the distance '
                    'or decreasing the screen size.'
                )

            # Calculate (kx^2+ky^2) / k^2 relation
            _, relation_axes = tensor_dot(
                a=1 / (self._k ** 2),
                b=kx2ky2,
                a_axis='wavelength',
                b_axis=('H', 'W')
            )  # shape: ('wavelength', 'H', 'W') or ('H', 'W') depending on k shape # noqa: E501

            impulse_response_fft = self.impulse_response_fresnel()

            output_wavefront_fft, _ = tensor_dot(
                a=wavefront_fft,  # example shape: (5, 'wavelength', 1, 'H', 'W')   # noqa: E501
                b=impulse_response_fft,  # example shape: ('wavelength', 'H', 'W')  # noqa: E501
                a_axis=self.simulation_parameters.axes.names,
                b_axis=relation_axes,
                preserve_a_axis=True  # check that the output has the input shape   # noqa: E501
            )  # example output shape: (5, 'wavelength', 1, 'H', 'W')

            output_wavefront = torch.fft.ifft2(
                output_wavefront_fft,
                dim=(self._h_index, self._w_index)
            )
        elif method == 'zpRSC':
            # calculate paddings for zero-padding AS method
            x_paddings_asm = self._calculate_paddings(
                sampling_interval=self._sampling_interval_x
            )
            y_paddings_asm = self._calculate_paddings(
                sampling_interval=self._sampling_interval_y
            )

            if x_paddings_asm < self._x_nodes / 2:
                warn(
                    (
                        'In zpRSC method optimal number of paddings in x direction '    # noqa: E501
                        + f'({str(x_paddings_asm)})'
                        + ' is less than the number of added nodes in the zpRSC method. It is preferable to use the zpASM method.'    # noqa: E501
                    )
                )

            if y_paddings_asm < self._y_nodes / 2:
                warn(
                    (
                        'In zpRSC method optimal number of paddings in y direction '    # noqa: E501
                        + f'({str(y_paddings_asm)})'
                        + ' is less than the number of added nodes in the zpRSC method. It is preferable to use the zpASM method.'    # noqa: E501
                    )
                )

            x_paddings = (int(self._x_nodes / 2) + 1) * 2

            y_paddings = (int(self._y_nodes / 2) + 1) * 2

            distance_x = (self._x_nodes + 2 * x_paddings) * torch.sqrt(
                1 - (self.simulation_parameters.axes.wavelength / (2 * self._sampling_interval_x)) ** 2   # noqa: E501
            ) * (self._sampling_interval_x ** 2) / self.simulation_parameters.axes.wavelength   # noqa: E501

            distance_y = (self._y_nodes + 2 * y_paddings) * torch.sqrt(
                1 - (self.simulation_parameters.axes.wavelength / (2 * self._sampling_interval_y)) ** 2   # noqa: E501
            ) * (self._sampling_interval_y ** 2) / self.simulation_parameters.axes.wavelength   # noqa: E501

            x_condition = self.distance < distance_x
            y_condition = self.distance < distance_y

            if not torch.all(x_condition):
                warn(
                    'In zpRSC method propagation distance is not large enough in the x direction.'  # noqa: E501
                    'It is preferable to use the zpASM or ASM method.'
                )

            if not torch.all(y_condition):
                warn(
                    'In zpRSC method propagation distance is not large enough in the y direction.'  # noqa: E501
                    'It is preferable to use the zpASM or ASM method.'
                )

            padding_order = [0] * (2 * wavefront.ndim)

            padding_order[-(2 * self._w_index + 1)] = x_paddings
            padding_order[-(2 * self._w_index + 2)] = x_paddings
            padding_order[-(2 * self._h_index + 1)] = y_paddings
            padding_order[-(2 * self._h_index + 2)] = y_paddings

            # add paddings
            wavefront_padded = F.pad(
                wavefront, padding_order, mode='constant', value=0
            )

            wavefront_padded_fft = torch.fft.fftshift(torch.fft.fft2(
                wavefront_padded,
                dim=(self._h_index, self._w_index)
            ))

            self.transfer_function, tf_axes = self.transfer_function_rsc(
                x_nodes=wavefront_padded.shape[self._w_index],
                y_nodes=wavefront_padded.shape[self._h_index]
            )

            self.impulse_response_fft = torch.fft.fftshift(torch.fft.fft2(
                self.transfer_function,
                dim=(self._h_index, self._w_index)
            ))

            output_wavefront_padded_fft, _ = tensor_dot(
                a=wavefront_padded_fft,  # example shape: (5, 'wavelength', 1, 'H', 'W')
                b=self.impulse_response_fft,  # example shape: ('wavelength', 'H', 'W')
                a_axis=self.simulation_parameters.axes.names,
                b_axis=tf_axes,
                preserve_a_axis=True  # check that the output has the input shape
            )  # example output shape: (5, 'wavelength', 1, 'H', 'W')

            output_wavefront_padded = torch.fft.ifftshift(torch.fft.ifft2(
                output_wavefront_padded_fft,
                dim=(self._h_index, self._w_index)
            ))

            h_start = y_paddings
            h_end = output_wavefront_padded.shape[self._h_index] - y_paddings
            w_start = x_paddings
            w_end = output_wavefront_padded.shape[self._w_index] - x_paddings

            slices = [slice(None)] * output_wavefront_padded.ndim
            slices[self._h_index] = slice(h_start, h_end)
            slices[self._w_index] = slice(w_start, w_end)

            # Apply slices to remove paddings
            output_wavefront = output_wavefront_padded[tuple(slices)]

        elif method == 'RSC':

            distance_x = self._x_nodes * torch.sqrt(
                1 - (self.simulation_parameters.axes.wavelength / (2 * self._sampling_interval_x)) ** 2   # noqa: E501
            ) * (self._sampling_interval_x ** 2) / self.simulation_parameters.axes.wavelength   # noqa: E501

            distance_y = self._y_nodes * torch.sqrt(
                1 - (self.simulation_parameters.axes.wavelength / (2 * self._sampling_interval_y)) ** 2   # noqa: E501
            ) * (self._sampling_interval_y ** 2) / self.simulation_parameters.axes.wavelength   # noqa: E501

            x_condition = self.distance < distance_x
            y_condition = self.distance < distance_y

            if not torch.all(x_condition):
                warn(
                    'In RSC method propagation distance is not large enough in the x direction.'  # noqa: E501
                    'It is preferable to use the zpASM or ASM method.'
                )

            if not torch.all(y_condition):
                warn(
                    'In RSC method propagation distance is not large enough in the y direction.'  # noqa: E501
                    'It is preferable to use the zpASM or ASM method.'
                )

            wavefront_fft = torch.fft.fftshift(
                torch.fft.fft2(
                    wavefront,
                    dim=(self._h_index, self._w_index)
                )
            )

            self.transfer_function, tf_axes = self.transfer_function_rsc(
                x_nodes=self._x_nodes,
                y_nodes=self._y_nodes
            )

            self.impulse_response_fft = torch.fft.fftshift(
                torch.fft.fft2(
                    self.transfer_function,
                    dim=(self._h_index, self._w_index)
                )
            )

            output_wavefront_fft, _ = tensor_dot(
                a=wavefront_fft,  # example shape: (5, 'wavelength', 1, 'H', 'W')
                b=self.impulse_response_fft,  # example shape: ('wavelength', 'H', 'W')
                a_axis=self.simulation_parameters.axes.names,
                b_axis=tf_axes,
                preserve_a_axis=True  # check that the output has the input shape
            )  # example output shape: (5, 'wavelength', 1, 'H', 'W')

            output_wavefront = torch.fft.fftshift(torch.fft.ifft2(
                    output_wavefront_fft,
                    dim=(self._h_index, self._w_index)
                ))

            norm = (2 * torch.pi) ** 2 * self._x_nodes * self._y_nodes

            output_wavefront = output_wavefront / norm

        else:

            raise ValueError("Unknown forward propagation method")

        return output_wavefront

    def _propagate_wavefront_inverse(
        self,
        wavefront: Wavefront,
        method: str
    ) -> Wavefront:
        if method == 'zpASM':
            # calculate paddings for zero-padding AS method
            x_paddings = self._calculate_paddings(
                sampling_interval=self._sampling_interval_x
            )
            y_paddings = self._calculate_paddings(
                sampling_interval=self._sampling_interval_y
            )

            if x_paddings > self._x_nodes / 2:
                warn(
                    (
                        'In zpASM number of paddings in x direction '
                        + f'({str(x_paddings)})'
                        + ' exceeds the number of nodes. It is preferable to use the RSC method'    # noqa: E501
                    )
                )

            if y_paddings > self._y_nodes / 2:
                warn(
                    (
                        'In zpASM number of paddings in y direction '
                        + f'({str(y_paddings)})'
                        + ' exceeds the number of nodes. It is preferable to use the RSC method'    # noqa: E501
                    )
                )

            padding_order = [0] * (2 * wavefront.ndim)

            padding_order[-(2 * self._w_index + 1)] = x_paddings
            padding_order[-(2 * self._w_index + 2)] = x_paddings
            padding_order[-(2 * self._h_index + 1)] = y_paddings
            padding_order[-(2 * self._h_index + 2)] = y_paddings

            # add paddings
            wavefront_padded = F.pad(
                wavefront, padding_order, mode='constant', value=0
            )    # noqa: E501

            # spectrum with zero frequency components in the center
            wavefront_padded_fft = torch.fft.fftshift(
                torch.fft.fft2(
                    wavefront_padded,
                    dim=(self._h_index, self._w_index)
                )
            )

            # Calculate (kx^2+ky^2) tensor
            kx2ky2, _, _ = self._calculate_kx2ky2(
                sampling_interval=(self._sampling_interval_x, self._sampling_interval_y),   # noqa: E501
                nodes=(self._x_nodes + 2 * x_paddings, self._y_nodes + 2 * y_paddings),    # noqa: E501
                device=self._device
            )   # shape: ('H', 'W')

            # Calculate (kx^2+ky^2) / k^2 relation
            _, relation_axes = tensor_dot(
                a=1 / (self._k ** 2),
                b=kx2ky2,
                a_axis='wavelength',
                b_axis=('H', 'W')
            )  # shape: ('wavelength', 'H', 'W') or ('H', 'W') depending on k shape # noqa: E501

            kx2ky2_cond = self._wave_number ** 2 - kx2ky2 >= 0
            # Calculate kz
            wave_number_z_non_negative = torch.sqrt(
                self._wave_number ** 2 - kx2ky2 * kx2ky2_cond + 0j
            )  # 0j is required to convert argument to complex

            # take into account evanescent waves
            wave_number_z_negative = 1j * torch.sqrt(
                kx2ky2 * ~kx2ky2_cond + 0j - self._wave_number ** 2
            )

            wave_number_z = wave_number_z_non_negative + wave_number_z_negative
            # Registering Buffer for _wave_number_z
            self._wave_number_z = self.make_buffer(
                '_wave_number_z', wave_number_z
            )

            self.impulse_response_fft_inverse = self.impulse_response_fft.conj()
            self.transfer_function = torch.fft.ifft2(self.impulse_response_fft_inverse)

            output_wavefront_padded_fft, _ = tensor_dot(
                a=wavefront_padded_fft,  # example shape: (5, 'wavelength', 1, 'H', 'W')    # noqa: E501
                b=self.impulse_response_fft_inverse,  # example shape: ('wavelength', 'H', 'W')  # noqa: E501
                a_axis=self.simulation_parameters.axes.names,
                b_axis=relation_axes,
                preserve_a_axis=True  # check that the output has the input shape   # noqa: E501
            )  # example output shape: (5, 'wavelength', 1, 'H', 'W')

            output_wavefront_padded = torch.fft.ifft2(
                output_wavefront_padded_fft,
                dim=(self._h_index, self._w_index)
            )

            # remove added paddings
            h_start = y_paddings
            h_end = output_wavefront_padded.shape[self._h_index] - y_paddings
            w_start = x_paddings
            w_end = output_wavefront_padded.shape[self._w_index] - x_paddings

            slices = [slice(None)] * output_wavefront_padded.ndim
            slices[self._h_index] = slice(h_start, h_end)
            slices[self._w_index] = slice(w_start, w_end)

            # Apply slices to remove paddings
            output_wavefront = output_wavefront_padded[tuple(slices)]

        return output_wavefront


    def forward(
        self,
        incident_wavefront: Wavefront
    ) -> Wavefront:
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

        output_wavefront = self._propagate_wavefront(
            wavefront=incident_wavefront,
            method=self.method
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

        output_wavefront = self._propagate_wavefront_inverse(
            wavefront=transmission_wavefront,
            method=self.method
        )

        return output_wavefront

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [
            ParameterSpecs(
                'distance', [
                    PrettyReprRepr(self.distance),
                ]
            )
        ]

    @staticmethod
    def _widget_html_(
        index: int,
        name: str,
        element_type: str | None,
        subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template('widget_free_space.html.jinja').render(
            index=index, name=name, subelements=subelements
        )
