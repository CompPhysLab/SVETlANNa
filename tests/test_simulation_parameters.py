from svetlanna.simulation_parameters import Axes, AxisNotFound
from svetlanna import SimulationParameters
import pytest
import torch


def test_axes():
    # Test required axes are actually required
    with pytest.raises(ValueError):
        Axes({})
        SimulationParameters(
            {
                "W": torch.linspace(-1, 1, 10),
            }
        )
    with pytest.raises(ValueError):
        Axes(
            {
                "W": torch.linspace(-1, 1, 10),
                "H": torch.linspace(-1, 1, 10),
            }
        )
    Axes(
        {
            "W": torch.linspace(-1, 1, 10),
            "H": torch.linspace(-1, 1, 10),
            "wavelength": torch.tensor(312),
        }
    )

    # Test with wrong H and W axis shape
    with pytest.raises(ValueError):
        Axes(
            {
                "W": torch.tensor([[10.0]]),  # wrong shape
                "H": torch.linspace(-1, 1, 10),
                "wavelength": torch.tensor(312),
            }
        )
    with pytest.raises(ValueError):
        Axes(
            {
                "W": torch.linspace(-1, 1, 10),
                "H": torch.tensor([[10.0]]),  # wrong shape
                "wavelength": torch.tensor(312),
            }
        )

    # Test with wrong additional axes shape
    with pytest.raises(ValueError):
        Axes(
            {
                "W": torch.linspace(-1, 1, 10),
                "H": torch.linspace(-1, 1, 10),
                "wavelength": torch.tensor(312),
                "pol": torch.tensor([[1.2, 3.4]]),  # wrong shape
            }
        )

    w_axis = torch.linspace(-1, 1, 10)
    pol_axis = torch.tensor([1.0, 0.0])
    axes = Axes(
        {
            "W": w_axis,
            "H": torch.linspace(-1, 1, 10),
            "wavelength": torch.tensor(312),
            "pol": pol_axis,
        }
    )

    # Test names of non-scalar axes
    assert axes.names == ("pol", "H", "W")

    # Test indices
    assert axes.index("pol") == -3
    assert axes.index("H") == -2
    assert axes.index("W") == -1
    with pytest.raises(AxisNotFound):
        axes.index("wavelength")  # scalar axis
    with pytest.raises(AxisNotFound):
        axes.index("t")  # axis does not exists

    # Test __getattribute__ for named axes
    assert axes.W is w_axis
    assert axes.pol is pol_axis

    # Test __setattr__ failure for axes
    with pytest.warns(UserWarning):
        axes.W = pol_axis
    assert axes.W is w_axis

    # Test __getitem__
    assert axes["W"] is w_axis
    assert axes["pol"] is pol_axis
    assert axes["wavelength"] == torch.tensor(312)
    with pytest.raises(AxisNotFound):
        axes["t"]  # axis does not exists

    # Test disabled __setitem__
    with pytest.raises(RuntimeError):
        axes["W"] = w_axis
    with pytest.raises(RuntimeError):
        axes["pol"] = pol_axis
    with pytest.raises(RuntimeError):
        axes["t"] = 123

    # Test __dir__
    assert set(dir(axes)) == {"H", "W", "pol", "wavelength"}


def test_simulation_parameters():
    w_axis = torch.linspace(-1, 2, 13)
    h_axis = torch.linspace(-12, -3, 25)
    pol_axis = torch.tensor([1.0, 0.0])
    sim_params = SimulationParameters(
        {"W": w_axis, "H": h_axis, "wavelength": 123.0, "pol": pol_axis, "t": 0.0}
    )

    # Test __getitem__
    assert sim_params["W"] is w_axis
    assert sim_params["pol"] is pol_axis
    assert sim_params["t"] == 0
    assert sim_params["wavelength"] == 123

    # Test meshgrid
    meshgrid_W, meshgrid_H = sim_params.meshgrid("W", "H")
    assert torch.allclose(meshgrid_W, w_axis[None, ...])
    assert torch.allclose(meshgrid_H, h_axis[..., None])

    meshgrid_W1, meshgrid_W2 = sim_params.meshgrid("W", "W")
    assert torch.allclose(meshgrid_W1, w_axis[None, ...])
    assert torch.allclose(meshgrid_W2, w_axis[..., None])

    meshgrid_H, meshgrid_wl = sim_params.meshgrid("H", "wavelength")
    assert torch.allclose(meshgrid_H, h_axis[None, ...])
    assert torch.allclose(meshgrid_wl, torch.tensor(123.0)[None])

    # Test axes_size
    assert sim_params.axes_size(("W",)) == torch.Size((13,))
    assert sim_params.axes_size(("wavelength", "H")) == torch.Size((1, 25))
    assert sim_params.axes_size(("H",)) == torch.Size((25,))

    with pytest.warns(UserWarning):
        # non existing axis
        assert sim_params.axes_size(("a", "H")) == torch.Size((0, 25))


@pytest.fixture(
    scope="function",
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda is not available"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="mps is not available"
            ),
        ),
    ],
)
def default_device(request):
    # Set the default device
    old_default_device = torch.get_default_device()
    torch.set_default_device(request.param)
    yield torch.get_default_device()
    torch.set_default_device(old_default_device)


def test_device(default_device: torch.device):
    w_axis = torch.linspace(-1, 2, 13, device="cpu")
    h_axis = torch.linspace(-12, -3, 25)

    if default_device.type != "cpu":
        with pytest.raises(ValueError):
            SimulationParameters(
                {
                    "W": w_axis,
                    "H": h_axis.to(default_device),
                    "wavelength": 123.0,
                }
            )

    # Test if in the following case the axis tensor is located on the device
    sim_params = SimulationParameters(
        {  # type: ignore
            "W": [1.0, 2.0, 3.0],
            "H": [1.0, 2.0, 3.0],
            "wavelength": 123.0,
        }
    )
    assert sim_params.axes.W.device == default_device

    # Test to() method
    transferred_sim_params = sim_params.to(default_device)
    assert transferred_sim_params is sim_params

    # Test to('cpu')
    transferred_sim_params = sim_params.to("cpu")
    assert transferred_sim_params.device.type == "cpu"  # type: ignore
    for axis_name in sim_params.axes.names:
        assert transferred_sim_params.axes[axis_name].device.type == "cpu"
    # And back
    transferred_sim_params = transferred_sim_params.to(default_device)
    assert transferred_sim_params.device == default_device
    for axis_name in sim_params.axes.names:
        assert transferred_sim_params.axes[axis_name].device == default_device


def test_axes_scalar_names():
    axes = Axes(
        {
            "W": torch.linspace(-1, 1, 10),
            "H": torch.linspace(-1, 1, 8),
            "wavelength": torch.tensor(0.5),
            "time": torch.tensor(1.0),
            "pol": torch.tensor([1.0, 0.0]),
        }
    )

    assert "wavelength" in axes.scalar_names
    assert "time" in axes.scalar_names
    assert "W" not in axes.scalar_names
    assert "H" not in axes.scalar_names
    assert "pol" not in axes.scalar_names
    assert len(axes.scalar_names) == 2


def test_axes_shapes():
    axes = Axes(
        {
            "W": torch.linspace(-1, 1, 10),
            "H": torch.linspace(-1, 1, 8),
            "wavelength": torch.linspace(0.4, 0.6, 5),
        }
    )

    # shapes should match negative indexing
    assert axes.shapes[-1] == 10  # W
    assert axes.shapes[-2] == 8  # H
    assert axes.shapes[-3] == 5  # wavelength


def test_axes_ensure_order():
    axes = Axes(
        {
            "W": torch.linspace(-1, 1, 10),
            "H": torch.linspace(-1, 1, 8),
            "wavelength": torch.linspace(0.4, 0.6, 5),
        }
    )

    t = torch.rand(2, 5, 8, 10)  # batch, wavelength, H, W

    # No change when already in correct order
    t_same = axes.ensure_order(t, "H", "W")
    assert t_same.shape == torch.Size([2, 5, 8, 10])

    # Swap H and W
    t_swapped = axes.ensure_order(t, "W", "H")
    assert t_swapped.shape == torch.Size([2, 5, 10, 8])

    # Move wavelength to end
    t_wl_end = axes.ensure_order(t, "wavelength")
    assert t_wl_end.shape == torch.Size([2, 8, 10, 5])

    # Error on non-existent axis
    with pytest.raises(AxisNotFound):
        axes.ensure_order(t, "nonexistent")


def test_frozen_by_default():
    sp = SimulationParameters(
        W=torch.linspace(-1, 1, 10),
        H=torch.linspace(-1, 1, 10),
        wavelength=torch.tensor(0.5),
    )

    assert sp.frozen is True

    with pytest.raises(RuntimeError):
        sp.pol = torch.tensor([1.0, 0.0])


def test_unfreeze():
    sp = SimulationParameters(
        W=torch.linspace(-1, 1, 10),
        H=torch.linspace(-1, 1, 10),
        wavelength=torch.tensor(0.5),
    )

    assert sp.frozen is True
    sp.unfreeze()
    assert sp.frozen is False

    sp.pol = torch.tensor([1.0, 0.0])
    assert "pol" in sp.axis_names


def test_cast():
    sp = SimulationParameters(
        W=torch.linspace(-1, 1, 10),
        H=torch.linspace(-1, 1, 8),
        wavelength=torch.linspace(0.4, 0.6, 5),
    )

    # Cast H, W tensor to full axes shape
    t = torch.rand(8, 10)
    casted = sp.cast(t, "H", "W")
    assert casted.shape == torch.Size([1, 8, 10])

    # Cast wavelength tensor
    t_wl = torch.rand(5)
    casted_wl = sp.cast(t_wl, "wavelength")
    assert casted_wl.shape == torch.Size([5, 1, 1])

    # Error on shape mismatch
    with pytest.raises(ValueError):
        sp.cast(torch.rand(7, 10), "H", "W")


def test_cast_with_scalar_axis():
    sp = SimulationParameters(
        W=torch.linspace(-1, 1, 10),
        H=torch.linspace(-1, 1, 8),
        wavelength=torch.tensor(0.5),
    )

    t = torch.rand(8, 10)
    casted = sp.cast(t, "H", "W")
    assert casted.shape == torch.Size([8, 10])  # no change, wavelength is scalar


def test_reorder():
    sp = SimulationParameters(
        W=torch.linspace(-1, 1, 10),
        H=torch.linspace(-1, 1, 8),
        wavelength=torch.linspace(0.4, 0.6, 5),
    )

    t = torch.rand(2, 5, 8, 10)  # batch, wavelength, H, W
    reordered = sp.reorder(t, "W", "H")
    assert reordered.shape == torch.Size([2, 5, 10, 8])


def test_to_inplace():
    sp = SimulationParameters(
        W=torch.linspace(-1, 1, 10),
        H=torch.linspace(-1, 1, 8),
        wavelength=torch.tensor(0.5),
    )

    original_id = id(sp)
    result = sp.to("cpu")

    assert result is sp
    assert id(result) == original_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_device_canonical():
    sp = SimulationParameters(
        W=torch.linspace(-1, 1, 10),
        H=torch.linspace(-1, 1, 8),
        wavelength=torch.tensor(0.5),
    )

    sp.to("cuda")
    assert sp.device == torch.device("cuda:0")
    assert sp.axes.W.device == sp.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_element_to_transfers_sim_params():
    from svetlanna.elements import ThinLens

    sp = SimulationParameters(
        W=torch.linspace(-1, 1, 100),
        H=torch.linspace(-1, 1, 100),
        wavelength=torch.tensor(0.5e-6),
    )

    lens = ThinLens(sp, focal_length=0.1)
    assert sp.device.type == "cpu"

    lens.to("cuda")
    assert sp.device.type == "cuda"
    assert sp.axes.W.device.type == "cuda"
