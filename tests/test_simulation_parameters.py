from svetlanna.simulation_parameters import AxisNotFound
from svetlanna import SimulationParameters
import pytest
import torch


def test_axes():
    # Test required axes are actually required
    with pytest.raises(ValueError):
        SimulationParameters(
            {
                "x": torch.linspace(-1, 1, 10),
            }
        )
    with pytest.raises(ValueError):
        SimulationParameters(
            {
                "x": torch.linspace(-1, 1, 10),
                "y": torch.linspace(-1, 1, 10),
            }
        )
    SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 10),
            "y": torch.linspace(-1, 1, 10),
            "wavelength": torch.tensor(312),
        }
    )

    # Test with wrong y and x axis shape
    with pytest.raises(ValueError):
        SimulationParameters(
            {
                "x": torch.tensor([[10.0]]),  # wrong shape
                "y": torch.linspace(-1, 1, 10),
                "wavelength": torch.tensor(312),
            }
        )
    with pytest.raises(ValueError):
        SimulationParameters(
            {
                "x": torch.linspace(-1, 1, 10),
                "y": torch.tensor([[10.0]]),  # wrong shape
                "wavelength": torch.tensor(312),
            }
        )

    # Test with wrong additional axes shape
    with pytest.raises(ValueError):
        SimulationParameters(
            {
                "x": torch.linspace(-1, 1, 10),
                "y": torch.linspace(-1, 1, 10),
                "wavelength": torch.tensor(312),
                "pol": torch.tensor([[1.2, 3.4]]),  # wrong shape
            }
        )

    w_axis = torch.linspace(-1, 1, 10)
    pol_axis = torch.tensor([1.0, 0.0])
    axes = SimulationParameters(
        {
            "x": w_axis,
            "y": torch.linspace(-1, 1, 10),
            "wavelength": torch.tensor(312),
            "pol": pol_axis,
        }
    )

    # Test names of non-scalar axes
    assert axes.names == ("pol", "y", "x")

    # Test indices
    assert axes.index("pol") == -3
    assert axes.index("y") == -2
    assert axes.index("x") == -1
    with pytest.raises(AxisNotFound):
        axes.index("wavelength")  # scalar axis
    with pytest.raises(AxisNotFound):
        axes.index("t")  # axis does not exists

    # Test __getattribute__ for named axes
    assert axes.x is w_axis
    assert axes.pol is pol_axis

    # Test __setattr__ failure for axes
    with pytest.warns(UserWarning):
        axes.x = pol_axis
    assert axes.x is w_axis

    # Test __getitem__
    assert axes["x"] is w_axis
    assert axes["pol"] is pol_axis
    assert axes["wavelength"] == torch.tensor(312)
    with pytest.raises(AxisNotFound):
        axes["t"]  # axis does not exists

    # Test disabled __setitem__
    with pytest.raises(RuntimeError):
        axes["x"] = w_axis
    with pytest.raises(RuntimeError):
        axes["pol"] = pol_axis
    with pytest.raises(RuntimeError):
        axes["t"] = 123

    # Test __dir__
    assert set(dir(axes)).issuperset({"y", "x", "pol", "wavelength"})


def test_simulation_parameters():
    x_axis = torch.linspace(-1, 2, 13)
    y_axis = torch.linspace(-12, -3, 25)
    pol_axis = torch.tensor([1.0, 0.0])
    sim_params = SimulationParameters(
        {"x": x_axis, "y": y_axis, "wavelength": 123.0, "pol": pol_axis, "t": 0.0}
    )

    # Test __getitem__
    assert sim_params["x"] is x_axis
    assert sim_params["pol"] is pol_axis
    assert sim_params["t"] == 0
    assert sim_params["wavelength"] == 123

    # Test meshgrid
    meshgrid_X, meshgrid_Y = sim_params.meshgrid("x", "y")
    assert torch.allclose(meshgrid_X, x_axis[None, ...])
    assert torch.allclose(meshgrid_Y, y_axis[..., None])

    meshgrid_X1, meshgrid_X2 = sim_params.meshgrid("x", "x")
    assert torch.allclose(meshgrid_X1, x_axis[None, ...])
    assert torch.allclose(meshgrid_X2, x_axis[..., None])

    meshgrid_Y, meshgrid_wl = sim_params.meshgrid("y", "wavelength")
    assert torch.allclose(meshgrid_Y, y_axis[None, ...])
    assert torch.allclose(meshgrid_wl, torch.tensor(123.0)[None])

    # Test axes_size
    assert sim_params.axes_size(("x",)) == torch.Size((13,))
    assert sim_params.axes_size(("wavelength", "y")) == torch.Size((1, 25))
    assert sim_params.axes_size(("y",)) == torch.Size((25,))

    with pytest.warns(UserWarning):
        # non existing axis
        assert sim_params.axes_size(("a", "y")) == torch.Size((0, 25))


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
    x_axis = torch.linspace(-1, 2, 13, device="cpu")
    y_axis = torch.linspace(-12, -3, 25)

    if default_device.type != "cpu":
        with pytest.raises(ValueError):
            SimulationParameters(
                {
                    "x": x_axis,
                    "y": y_axis.to(default_device),
                    "wavelength": 123.0,
                }
            )

    # Test if in the following case the axis tensor is located on the device
    sim_params = SimulationParameters(
        {  # type: ignore
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "wavelength": 123.0,
        }
    )
    assert sim_params.axes.x.device == default_device

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


def test_axes_names_scalar():
    axes = SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 10),
            "y": torch.linspace(-1, 1, 8),
            "wavelength": torch.tensor(0.5),
            "time": torch.tensor(1.0),
            "pol": torch.tensor([1.0, 0.0]),
        }
    )

    assert "wavelength" in axes.names_scalar
    assert "time" in axes.names_scalar
    assert "x" not in axes.names_scalar
    assert "y" not in axes.names_scalar
    assert "pol" not in axes.names_scalar
    assert len(axes.names_scalar) == 2


def test_axes_ensure_order():
    axes = SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 10),
            "y": torch.linspace(-1, 1, 8),
            "wavelength": torch.linspace(0.4, 0.6, 5),
        }
    )

    t = torch.rand(2, 5, 8, 10)  # batch, wavelength, H, W

    # No change when already in correct order
    t_same = axes.reorder(t, "y", "x")
    assert t_same.shape == torch.Size([2, 5, 8, 10])

    # Swap y and x
    t_swapped = axes.reorder(t, "x", "y")
    assert t_swapped.shape == torch.Size([2, 5, 10, 8])

    # Move wavelength to end
    t_wl_end = axes.reorder(t, "wavelength")
    assert t_wl_end.shape == torch.Size([2, 8, 10, 5])

    # Error on non-existent axis
    with pytest.raises(AxisNotFound):
        axes.reorder(t, "nonexistent")


def test_cast():
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 8),
        wavelength=torch.linspace(0.4, 0.6, 5),
    )

    # Cast y, x tensor to full axes shape
    t = torch.rand(8, 10)
    casted = sp.cast(t, "y", "x")
    assert casted.shape == torch.Size([1, 8, 10])

    # Cast wavelength tensor
    t_wl = torch.rand(5)
    casted_wl = sp.cast(t_wl, "wavelength")
    assert casted_wl.shape == torch.Size([5, 1, 1])

    # Error on shape mismatch
    with pytest.raises(ValueError):
        sp.cast(torch.rand(7, 10), "y", "x")


def test_cast_with_scalar_axis():
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 8),
        wavelength=torch.tensor(0.5),
    )

    t = torch.rand(8, 10)
    casted = sp.cast(t, "y", "x")
    assert casted.shape == torch.Size([8, 10])  # no change, wavelength is scalar


def test_reorder():
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 8),
        wavelength=torch.linspace(0.4, 0.6, 5),
    )

    t = torch.rand(2, 5, 8, 10)  # batch, wavelength, y, x
    reordered = sp.reorder(t, "x", "y")
    assert reordered.shape == torch.Size([2, 5, 10, 8])


def test_to_inplace():
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 8),
        wavelength=torch.tensor(0.5),
    )

    original_id = id(sp)
    result = sp.to("cpu")

    assert result is sp
    assert id(result) == original_id


@pytest.mark.parametrize(
    ("device",),
    [
        pytest.param("cpu"),
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
def test_to_device_canonical(device):
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 8),
        wavelength=torch.tensor(0.5),
    )

    sp.to(device)
    assert sp.device.type == device
    assert sp.axes.x.device.type == device


@pytest.mark.parametrize(
    ("device",),
    [
        pytest.param("cpu"),
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
def test_element_to_transfers_sim_params(device):
    from svetlanna.elements import ThinLens

    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 100),
        y=torch.linspace(-1, 1, 100),
        wavelength=torch.tensor(0.5e-6),
    )

    lens = ThinLens(sp, focal_length=0.1)
    assert sp.device.type == "cpu"

    lens.to(device)
    assert sp.device.type == device
    assert sp.axes.x.device.type == device
