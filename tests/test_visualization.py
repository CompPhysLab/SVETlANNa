from svetlanna.visualization import LinearOpticalSetupStepwiseForwardWidget
from svetlanna.visualization import LinearOpticalSetupWidget
import svetlanna
import torch


def test_html_element():
    sim_params = svetlanna.SimulationParameters(
        {'W': torch.tensor([0]), 'H': torch.tensor([0]), 'wavelength': 1}
    )
    element = svetlanna.elements.FreeSpace(sim_params, distance=1, method='AS')

    assert element._repr_html_()


def test_widget_setup():
    sim_params = svetlanna.SimulationParameters(
        {
            'W': torch.linspace(-1, 1, 10),
            'H': torch.linspace(-1, 1, 10),
            'wavelength': 1
        }
    )
    setup = svetlanna.LinearOpticalSetup(
        [svetlanna.elements.Aperture(sim_params, mask=torch.rand(10, 10))]
    )

    assert isinstance(setup.show(), LinearOpticalSetupWidget)
    assert isinstance(
        setup.show_stepwise_forward(
            svetlanna.Wavefront(torch.full((10, 10), 1.)),
            sim_params
        ),
        LinearOpticalSetupStepwiseForwardWidget
    )
