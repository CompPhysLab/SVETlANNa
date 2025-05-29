from svetlanna.simulation_parameters import Axes, AxisNotFound
from svetlanna import SimulationParameters
import pytest
import torch


def test_axes():
    """
    Tests the Axes class for correct axis handling and validation."""
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
    """
    Tests the SimulationParameters class functionality.

        This tests the __getitem__ method, meshgrid generation, and axes size retrieval
        of the SimulationParameters class with various parameters. It also checks for
        expected warnings when accessing non-existent axes.

        Args:
            None

        Returns:
            None
    """
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
    """
    Provides a fixture for setting the default PyTorch device.

        This fixture iterates through 'cpu', 'cuda' (if available), and 'mps' (if available)
        as parameters, temporarily setting the default device to each one within the scope of a test function.
        It yields the current default device and then restores the original default device after the test completes.

        Args:
            request: The pytest request object providing access to fixture parameters.

        Returns:
            str: The currently set default PyTorch device (e.g., 'cpu', 'cuda', or 'mps').
    """
    # Set the default device
    old_default_device = torch.get_default_device()
    torch.set_default_device(request.param)
    yield torch.get_default_device()
    torch.set_default_device(old_default_device)


def test_device(default_device: torch.device):
    """
    Tests device placement and transfer for SimulationParameters.

        This method verifies that the SimulationParameters class correctly handles
        device placement of axis tensors, raises errors when appropriate, and
        that the `to()` method functions as expected for transferring data between devices.

        Args:
            default_device: The default device to use for testing.

        Returns:
            None
    """
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
