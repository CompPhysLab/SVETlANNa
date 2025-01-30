from typing import Iterable, Literal
from .elements import Element
from .simulation_parameters import SimulationParameters
from torch import nn
from torch import Tensor
import torch
import base64
import io
from .visualization.widgets import LinearOpticalSetupWidget
from .visualization.widgets import LinearOpticalSetupStepwiseForwardWidget


StepwisePlotTypes = Literal['A'] | Literal['I'] | Literal['phase'] | Literal['Re'] | Literal['Im'] 


class LinearOpticalSetup:
    """
    A linear optical network composed of Element's
    """
    def __init__(self, elements: Iterable[Element]) -> None:
        """
        Parameters
        ----------
        elements : Iterable[Element]
            A set of optical elements which make up a setup.
        """
        self.elements = elements
        self.net = nn.Sequential(*elements)  # torch network

        if all((hasattr(el, 'reverse') for el in self.elements)):

            class ReverseNet(nn.Module):
                def forward(self, Ein: Tensor) -> Tensor:
                    for el in reversed(list(elements)):
                        Ein = el.reverse(Ein)
                    return Ein

            self._reverse_net = ReverseNet()
        else:
            self._reverse_net = None

    def forward(self, input_wavefront: Tensor) -> Tensor:
        """
        A forward function for a network assembled from elements.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        torch.Tensor
            A wavefront after the last element of the network (output of the network).
        """
        return self.net(input_wavefront)

    def stepwise_forward(self, input_wavefront: Tensor):
        """
        Function that consistently applies forward method of each element to an input wavefront.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        str
            A string that represents a scheme of a propagation through a setup.
        list(torch.Tensor)
            A list of an input wavefront evolution during a propagation through a setup.
        """
        this_wavefront = input_wavefront
        # list of wavefronts while propagation of an initial wavefront through the system
        steps_wavefront = [this_wavefront]  # input wavefront is a zeroth step

        optical_scheme = ''  # string that represents a linear optical setup (schematic)

        self.net.eval()
        for ind_element, element in enumerate(self.net):
            # for visualization in a console
            element_name = type(element).__name__
            optical_scheme += f'-({ind_element})-> [{ind_element + 1}. {element_name}] '
            # TODO: Replace len(...) with something for Iterable?
            if ind_element == len(self.net) - 1:
                optical_scheme += f'-({ind_element + 1})->'
            # element forward
            this_wavefront = element.forward(this_wavefront)
            steps_wavefront.append(this_wavefront)  # add a wavefront to list of steps

        return optical_scheme, steps_wavefront

    def reverse(self, Ein: Tensor) -> Tensor:
        if self._reverse_net is not None:
            return self._reverse_net(Ein)
        raise TypeError('Reverse propagation is impossible. All elements should have reverse method.')

    def show(self, **settings) -> LinearOpticalSetupWidget:
        widget = LinearOpticalSetupWidget()
        elements = []
        for index, element in enumerate(self.elements):
            elements.append({
                'index': index,
                'type': element.__class__.__name__,
                'specs_html': element._repr_html_()
            })

        new_settings = {}
        for name in widget.settings.keys():
            if name in settings:
                new_settings[name] = settings[name]
            else:
                new_settings[name] = widget.settings[name]

        widget.settings = new_settings
        widget.elements = elements
        return widget

    def show_stepwise_forward(
        self,
        input_wavefront: Tensor,
        simulation_parameters: SimulationParameters,
        types_to_plot: tuple[StepwisePlotTypes, ...] = ('I', 'phase'),
        **settings
    ) -> LinearOpticalSetupStepwiseForwardWidget:
        widget = LinearOpticalSetupStepwiseForwardWidget()
        elements = []
        for index, element in enumerate(self.elements):
            elements.append({
                'index': index,
                'type': element.__class__.__name__,
                'specs_html': element._repr_html_()
            })

        new_settings = {}
        for name in widget.settings.keys():
            if name in settings:
                new_settings[name] = settings[name]
            else:
                new_settings[name] = widget.settings[name]

        widget.settings = new_settings
        widget.elements = elements

        import matplotlib.pyplot as plt

        with torch.no_grad():
            _, stepwise_wavefront = self.stepwise_forward(
                input_wavefront=input_wavefront
            )

        wavefront_images = []
        for wavefront in stepwise_wavefront:
            stream = io.BytesIO()

            width = simulation_parameters.axes.W.cpu()
            height = simulation_parameters.axes.H.cpu()

            N_plots = len(types_to_plot)

            width_to_height = (
                width.max() - width.min()
            ) / (
                height.max() - height.min()
            )

            figure, ax = plt.subplots(
                    1, N_plots,
                    figsize=(2+3*N_plots*width_to_height, 3),
                    dpi=120
                )

            for i, plot_type in enumerate(types_to_plot):
                axes = ax[i] if N_plots != 1 else ax
                if plot_type == 'A':
                    axes.pcolorfast(
                        width,
                        height,
                        wavefront.abs().cpu().numpy()
                    )
                    axes.set_title('Amplitude')

                elif plot_type == 'I':
                    axes.pcolorfast(
                        width,
                        height,
                        (wavefront.abs()**2).cpu().numpy()
                    )
                    axes.set_title('Intensity')

                elif plot_type == 'phase':
                    axes.pcolorfast(
                        width,
                        height,
                        wavefront.angle().cpu().numpy(),
                        vmin=-torch.pi,
                        vmax=torch.pi,
                    )
                    axes.set_title('Phase')

                elif plot_type == 'Re':
                    axes.pcolorfast(
                        width,
                        height,
                        wavefront.real.cpu().numpy(),
                    )
                    axes.set_title('Real part')

                elif plot_type == 'Re':
                    axes.pcolorfast(
                        width,
                        height,
                        wavefront.imag.cpu().numpy(),
                    )
                    axes.set_title('Imaginary part')

                axes.set_aspect('equal')

            plt.tight_layout()
            figure.savefig(stream)
            plt.close(figure)

            wavefront_images.append(
                base64.b64encode(stream.getvalue()).decode()
            )

        widget.wavefront_images = wavefront_images

        return widget
