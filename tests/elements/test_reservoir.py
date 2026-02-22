from svetlanna.elements import SimpleReservoir, DiffractiveLayer
from svetlanna import SimulationParameters, Wavefront
import torch


def test_queue():
    sim_params = SimulationParameters(
        {
            "x": torch.tensor([0]),
            "y": torch.tensor([0]),
            "wavelength": 1.0,
        }
    )
    reservoir = SimpleReservoir(
        sim_params,
        nonlinear_element=DiffractiveLayer(sim_params, mask=torch.tensor([[0.0]])),
        delay_element=DiffractiveLayer(sim_params, mask=torch.tensor([[0.0]])),
        delay=2,
        feedback_gain=1,
        input_gain=1,
    )

    # Empty queue yields no feedback.
    assert reservoir.pop_feedback_queue() is None

    wf1 = Wavefront.plane_wave(sim_params)
    wf2 = Wavefront.plane_wave(sim_params)

    reservoir.append_feedback_queue(wf1)
    # Still below delay length.
    assert reservoir.pop_feedback_queue() is None

    reservoir.append_feedback_queue(wf2)
    # Queue length reaches delay, first element is returned.
    assert reservoir.pop_feedback_queue() is wf1
    # Back below delay length.
    assert reservoir.pop_feedback_queue() is None

    # Validate queue reset.
    assert len(reservoir.feedback_queue) > 0
    reservoir.drop_feedback_queue()
    assert len(reservoir.feedback_queue) == 0


def test_forward(sim_params_simple: SimulationParameters):
    # Elements are placeholders; only queueing matters here.
    nonlinear_element = DiffractiveLayer(
        sim_params_simple, mask=torch.zeros(sim_params_simple.axis_sizes(("y", "x")))
    )
    delay_element = DiffractiveLayer(
        sim_params_simple, mask=torch.zeros(sim_params_simple.axis_sizes(("y", "x")))
    )
    feedback_gain = 0.8
    input_gain = 0.6
    delay = 5

    reservoir = SimpleReservoir(
        sim_params_simple,
        nonlinear_element=nonlinear_element,
        delay_element=delay_element,
        delay=delay,
        feedback_gain=feedback_gain,
        input_gain=input_gain,
    )

    wf = Wavefront.plane_wave(sim_params_simple)

    for i in range(delay):
        # Queue grows until it reaches the delay length.
        assert len(reservoir.feedback_queue) == i
        wf_out = reservoir(wf)
        assert len(reservoir.feedback_queue) == i + 1

        wf_out_expected = nonlinear_element(input_gain * wf)
        assert torch.allclose(wf_out, wf_out_expected)

    wf_out = reservoir(wf)
    # Queue does not exceed the delay length.
    assert len(reservoir.feedback_queue) == delay

    # Delay line contributes after it is filled.
    wf_out_not_expected = nonlinear_element(input_gain * wf)
    assert not torch.allclose(wf_out, wf_out_not_expected)

    # First contribution from the delay line.
    wf_out_expected = nonlinear_element(
        input_gain * wf + feedback_gain * nonlinear_element(input_gain * wf)
    )
    assert torch.allclose(wf_out, wf_out_expected)


def test_device(device_simple: str):
    """Test reservoir on different devices."""

    sim_params = SimulationParameters(
        x=torch.tensor([0]), y=torch.tensor([0]), wavelength=1.0
    )
    wavefront = Wavefront.plane_wave(sim_params).to(device=device_simple)

    assert sim_params.device == torch.get_default_device()
    reservoir = SimpleReservoir(
        sim_params,
        nonlinear_element=DiffractiveLayer(sim_params, mask=torch.tensor([[0.0]])),
        delay_element=DiffractiveLayer(sim_params, mask=torch.tensor([[0.0]])),
        delay=2,
        feedback_gain=1,
        input_gain=1,
    )
    reservoir.to(device=device_simple)

    assert reservoir(wavefront).device.type == device_simple

    # Simulation parameters on device
    sim_params.to(device=device_simple)

    assert sim_params.device.type == device_simple
    reservoir = SimpleReservoir(
        sim_params,
        nonlinear_element=DiffractiveLayer(
            sim_params, mask=torch.tensor([[0.0]]).to(device=device_simple)
        ),
        delay_element=DiffractiveLayer(
            sim_params, mask=torch.tensor([[0.0]]).to(device=device_simple)
        ),
        delay=2,
        feedback_gain=1,
        input_gain=1,
    )

    assert reservoir(wavefront).device.type == device_simple


def test_to_specs():
    """Stupid test to increase code coverage."""
    sim_params = SimulationParameters(
        x=torch.linspace(-10, 10, 20), y=torch.linspace(-10, 10, 20), wavelength=1.0
    )

    reservoir = SimpleReservoir(
        sim_params,
        nonlinear_element=DiffractiveLayer(sim_params, mask=torch.tensor([[0.0]])),
        delay_element=DiffractiveLayer(sim_params, mask=torch.tensor([[0.0]])),
        delay=2,
        feedback_gain=1,
        input_gain=1,
    )

    assert reservoir.to_specs()
    assert isinstance(reservoir._widget_html_(0, "", None, []), str)
