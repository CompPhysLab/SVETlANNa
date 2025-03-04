from svetlanna.elements import SimpleReservoir, DiffractiveLayer
from svetlanna import SimulationParameters, Wavefront
import torch


def test_queue():
    sim_params = SimulationParameters(
        {
            'W': torch.tensor([0]),
            'H': torch.tensor([0]),
            'wavelength': 1.
        }
    )
    reservoir = SimpleReservoir(
        sim_params,
        nonlinear_element=DiffractiveLayer(
            sim_params, mask=torch.tensor([[0.]])
        ),
        delay_element=DiffractiveLayer(
            sim_params, mask=torch.tensor([[0.]])
        ),
        delay=2,
        feedback_gain=1,
        input_gain=1
    )

    # feedback queue is empty
    # delay is 2, but no element in queue,
    # therefore pop_feedback_queue returns None
    assert reservoir.pop_feedback_queue() is None

    wf1 = Wavefront.plane_wave(sim_params)
    wf2 = Wavefront.plane_wave(sim_params)

    reservoir.append_feedback_queue(wf1)
    # feedback queue (wf1,)
    # delay is 2, but only 1 element in queue,
    # therefore pop_feedback_queue returns None
    assert reservoir.pop_feedback_queue() is None

    reservoir.append_feedback_queue(wf2)
    # delay line (wf1, wf2)
    # delay is 2, there are 2 elements in queue,
    # therefore pop_feedback_queue returns element
    assert reservoir.pop_feedback_queue() is wf1
    # delay line (wf2,)
    # delay is 2, but only 1 element in queue
    assert reservoir.pop_feedback_queue() is None

    # test drop_feedback_queue
    assert len(reservoir.feedback_queue) > 0
    reservoir.drop_feedback_queue()
    assert len(reservoir.feedback_queue) == 0


def test_forward():
    sim_params = SimulationParameters(
        {
            'W': torch.tensor([0]),
            'H': torch.tensor([0]),
            'wavelength': 1.
        }
    )

    nonlinear_element = DiffractiveLayer(
        sim_params, mask=torch.tensor([[0.]])
    )
    delay_element = DiffractiveLayer(
        sim_params, mask=torch.tensor([[0.]])
    )
    feedback_gain = 0.8
    input_gain = 0.6
    delay = 5

    reservoir = SimpleReservoir(
        sim_params,
        nonlinear_element=nonlinear_element,
        delay_element=delay_element,
        delay=delay,
        feedback_gain=feedback_gain,
        input_gain=input_gain
    )

    wf = Wavefront.plane_wave(sim_params)

    for i in range(delay):
        # test if feedback queue grows
        assert len(reservoir.feedback_queue) == i
        wf_out = reservoir(wf)
        assert len(reservoir.feedback_queue) == i + 1

        wf_out_expected = nonlinear_element(input_gain * wf)
        assert torch.allclose(wf_out, wf_out_expected)

    wf_out = reservoir(wf)
    # test if feedback queue does not grows over delay size
    assert len(reservoir.feedback_queue) == delay

    # test if delay line now play some role in the output
    wf_out_not_expected = nonlinear_element(input_gain * wf)
    assert not torch.allclose(wf_out, wf_out_not_expected)

    # hard coded very first delay line related contribution
    wf_out_expected = nonlinear_element(
        input_gain * wf + feedback_gain * nonlinear_element(
            input_gain * wf
        )
    )
    assert torch.allclose(wf_out, wf_out_expected)
