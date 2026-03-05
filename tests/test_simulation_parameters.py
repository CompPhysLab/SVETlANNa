from svetlanna.simulation_parameters import AxisNotFound
from svetlanna import SimulationParameters, Wavefront
from svetlanna.elements import Element
import pytest
import torch
from itertools import chain


class ElementToTest(Element):
    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        return incident_wavefront


def test_simulation_parameters_init():
    # === OLD API TESTS ===

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

    # Wrong x/y axis shapes.
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

    # Test with non-str name:
    with pytest.raises(TypeError):
        SimulationParameters(
            {
                "x": torch.linspace(-1, 1, 10),
                "y": torch.linspace(-1, 1, 10),
                "wavelength": torch.tensor(312),
                ("t",): 123,  # type: ignore
            }
        )

    # Working case
    SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 10),
            "y": torch.linspace(-1, 1, 10),
            "wavelength": torch.tensor(312),
        }
    )

    # === NEW API TESTS ===

    # Test required axes are actually required
    with pytest.raises(ValueError):
        SimulationParameters(  # type: ignore
            x=torch.linspace(-1, 1, 10),
        )
    with pytest.raises(ValueError):
        SimulationParameters(  # type: ignore
            x=torch.linspace(-1, 1, 10),
            y=torch.linspace(-1, 1, 10),
        )

    # Test with wrong y and x axis shape
    with pytest.raises(ValueError):
        SimulationParameters(
            x=torch.tensor([[10.0]]),  # wrong shape
            y=torch.linspace(-1, 1, 10),
            wavelength=torch.tensor(312),
        )
    with pytest.raises(ValueError):
        SimulationParameters(
            x=torch.linspace(-1, 1, 10),
            y=torch.tensor([[10.0]]),  # wrong shape
            wavelength=torch.tensor(312),
        )

    # Test with wrong additional axes shape
    with pytest.raises(ValueError):
        SimulationParameters(
            x=torch.linspace(-1, 1, 10),
            y=torch.linspace(-1, 1, 10),
            wavelength=torch.tensor(312),
            pol=torch.tensor([[1.2, 3.4]]),  # wrong shape
        )

    # Working case
    SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 10),
        wavelength=torch.tensor(312),
    )

    # Test with both API types
    with pytest.raises(ValueError):
        SimulationParameters(  # type: ignore
            {
                "x": torch.linspace(-1, 1, 10),
                "wavelength": torch.tensor(312),
            },
            y=torch.linspace(-1, 1, 10),
        )


def test_simulation_parameters_from_ranges():
    x_range = (-1, 2)
    x_points = 13
    y_range = (-12, -3)
    y_points = 25

    x = torch.linspace(x_range[0], x_range[1], x_points)
    y = torch.linspace(y_range[0], y_range[1], y_points)
    wavelength = 123.0

    sim_paras = SimulationParameters.from_ranges(
        x_range=x_range,
        x_points=x_points,
        y_range=y_range,
        y_points=y_points,
        wavelength=wavelength,
    )

    assert torch.allclose(sim_paras.x, x)
    assert torch.allclose(sim_paras.y, y)
    assert torch.allclose(sim_paras.wavelength, torch.tensor(wavelength))


def test_simulation_parameters_from_dict():

    x = torch.linspace(-1, 2, 13)
    y = torch.linspace(-12, -3, 25)
    wavelength = 123.0

    sim_paras = SimulationParameters.from_dict(
        {"x": x, "y": y, "wavelength": wavelength}
    )

    assert torch.allclose(sim_paras.x, x)
    assert torch.allclose(sim_paras.y, y)
    assert torch.allclose(sim_paras.wavelength, torch.tensor(wavelength))


def test_simulation_parameters_equal(device_simple: str):
    sim_params1 = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 10),
        wavelength=torch.tensor(312),
    )

    sim_params2 = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 10),
        wavelength=torch.tensor(312),
    )

    assert sim_params1.equal(sim_params1)
    assert sim_params1.equal(sim_params2)

    sim_params2.to(device_simple)
    # Comparing instances on different devices raises `RuntimeError`
    # because `torch.equal` requires the same device.
    if sim_params1.device != sim_params2.device:
        with pytest.raises(RuntimeError):
            assert sim_params1.equal(sim_params2)

    # Non-equal axes.
    sim_params2 = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 20),
        wavelength=torch.tensor(312),
    )
    assert not sim_params1.equal(sim_params2)

    # Extra axes.
    sim_params2 = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 20),
        wavelength=torch.tensor(312),
        pol=torch.tensor([1.0, 0.0]),
    )

    assert not sim_params1.equal(sim_params2)


def test_simulation_parameters_clone(device_simple: str):
    sim_params1 = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 10),
        wavelength=torch.tensor(312),
    )
    sim_params2 = sim_params1.clone()

    assert sim_params1 is not sim_params2
    assert sim_params1.equal(sim_params2)

    # Clone stays on the same device.
    sim_params1.to(device_simple)
    sim_params2 = sim_params1.clone()

    assert sim_params1.device.type == device_simple
    assert sim_params1.equal(sim_params2)


def test_simulation_parameters_axes():

    x_axis = torch.linspace(-1, 1, 10)
    pol_axis = torch.tensor([1.0, 0.0])
    sim_params = SimulationParameters(
        {
            "x": x_axis,
            "y": torch.linspace(-1, 1, 10),
            "wavelength": torch.tensor(312),
            "pol": pol_axis,
        }
    )

    # Test names of non-scalar axes
    assert sim_params.axis_names == ("pol", "y", "x")
    assert sim_params._axis_names_scalar == ("wavelength",)

    # Test indices
    assert sim_params.index("pol") == -3
    assert sim_params.index("y") == -2
    assert sim_params.index("x") == -1

    with pytest.raises(AxisNotFound):
        sim_params.index("wavelength")  # scalar axis

    with pytest.raises(AxisNotFound):
        sim_params.index("t")  # axis does not exists

    # Test __getattribute__ for named axes
    assert sim_params.x is x_axis
    assert sim_params.pol is pol_axis

    # Test __setattr__ warnings for axes
    with pytest.warns(UserWarning):
        sim_params.x = pol_axis
    # Test that axis was not changed
    assert sim_params.x is x_axis

    # Test __getitem__
    assert sim_params["x"] is x_axis
    assert sim_params["pol"] is pol_axis
    assert sim_params["wavelength"] == torch.tensor(312)
    with pytest.raises(AxisNotFound):
        sim_params["t"]  # axis does not exists

    # Test disabled __setitem__
    with pytest.raises(RuntimeError):
        sim_params["x"] = x_axis
    with pytest.raises(RuntimeError):
        sim_params["pol"] = pol_axis
    with pytest.raises(RuntimeError):
        sim_params["t"] = 123

    # Test __dir__
    assert set(dir(sim_params)).issuperset({"y", "x", "pol", "wavelength"})

    # Test __contains__
    assert "x" in sim_params
    assert "t" not in sim_params


def test_simulation_parameters_meshgrid():
    x_axis = torch.linspace(-1, 2, 13)
    y_axis = torch.linspace(-12, -3, 25)
    pol_axis = torch.tensor([1.0, 0.0])
    sim_params = SimulationParameters(
        {"x": x_axis, "y": y_axis, "wavelength": 123.0, "pol": pol_axis, "t": 0.0}
    )

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

    meshgrid_wl, meshgrid_X = sim_params.meshgrid("wavelength", "x")
    assert torch.allclose(meshgrid_wl, torch.tensor(123.0)[None])
    assert torch.allclose(meshgrid_X, x_axis[..., None])

    meshgrid_wl1, meshgrid_wl2 = sim_params.meshgrid("wavelength", "wavelength")
    assert torch.allclose(meshgrid_wl1, torch.tensor(123.0)[None])
    assert torch.allclose(meshgrid_wl2, torch.tensor(123.0)[None])

    # Test missing axis:
    with pytest.raises(AxisNotFound):
        sim_params.meshgrid("x", "z")


def test_simulation_parameters_axes_size():
    Nx = 13
    Ny = 25
    Npol = 2
    x_axis = torch.linspace(-1, 2, Nx)
    y_axis = torch.linspace(-12, -3, Ny)
    pol_axis = torch.tensor([1.0, 0.0])
    sim_params = SimulationParameters(
        {"x": x_axis, "y": y_axis, "wavelength": 123.0, "pol": pol_axis, "t": 0.0}
    )

    # Test axes_size
    assert sim_params.axis_sizes(("x",)) == torch.Size((Nx,))
    assert sim_params.axis_sizes(("wavelength", "y")) == torch.Size((1, Ny))
    assert sim_params.axis_sizes(("y",)) == torch.Size((Ny,))

    # Test axes_size with no arguments
    # Order is important here!
    assert sim_params.axis_sizes() == torch.Size((Npol, Ny, Nx))

    with pytest.raises(AxisNotFound):
        # non existing axis
        assert sim_params.axis_sizes(("a", "y")) == torch.Size((0, Ny))


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
        # Test error if tensors on diffrent devices
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
        {
            "x": [1.0, 2.0, 3.0],  # type: ignore
            "y": [1.0, 2.0, 3.0],  # type: ignore
            "wavelength": 123.0,
        }
    )
    assert sim_params.x.device == default_device

    # Test to() method
    transferred_sim_params = sim_params.to(default_device)
    assert transferred_sim_params is sim_params

    # Test to('cpu')
    transferred_sim_params = sim_params.to("cpu")
    assert transferred_sim_params.device.type == "cpu"  # type: ignore

    for axis_name in chain(sim_params.axis_names, sim_params._axis_names_scalar):
        assert transferred_sim_params[axis_name].device.type == "cpu"

    # And back
    transferred_sim_params = transferred_sim_params.to(default_device)
    assert transferred_sim_params.device == default_device

    for axis_name in chain(sim_params.axis_names, sim_params._axis_names_scalar):
        assert transferred_sim_params[axis_name].device == default_device


def test_axes_names_scalar():
    sp = SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 10),
            "y": torch.linspace(-1, 1, 8),
            "wavelength": torch.tensor(0.5),
            "time": torch.tensor(1.0),
            "pol": torch.tensor([1.0, 0.0]),
        }
    )

    assert "wavelength" in sp._axis_names_scalar
    assert "time" in sp._axis_names_scalar
    assert "x" not in sp._axis_names_scalar
    assert "y" not in sp._axis_names_scalar
    assert "pol" not in sp._axis_names_scalar
    assert len(sp._axis_names_scalar) == 2


def test_cast():
    Nx = 10
    Ny = 8
    Nwl = 5
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, Nx),
        y=torch.linspace(-1, 1, Ny),
        wavelength=torch.linspace(0.4, 0.6, Nwl),
        t=2.0,
    )

    # Cast x tensor to full axes shape
    t = torch.rand(Nx)
    casted = sp.cast(t, "x")
    assert casted.shape == torch.Size([1, 1, Nx])
    assert torch.allclose(casted, t[None, None, ...])

    # Cast y tensor to full axes shape
    t = torch.rand(Ny)
    casted = sp.cast(t, "y")
    assert casted.shape == torch.Size([1, Ny, 1])
    assert torch.allclose(casted, t[None, ..., None])

    # Cast y, x tensor to full axes shape
    t = torch.rand(Ny, Nx)
    casted = sp.cast(t, "y", "x")
    assert casted.shape == torch.Size([1, Ny, Nx])
    assert torch.allclose(casted, t[None, ...])

    # Cast y, x tensor to full axes shape
    t = torch.rand(Nx, Ny)
    casted = sp.cast(t, "x", "y")
    assert casted.shape == torch.Size([1, Ny, Nx])
    assert torch.allclose(casted, t.T[None, ...])

    # Cast wavelength tensor
    t = torch.rand(Nwl)
    casted = sp.cast(t, "wavelength")
    assert casted.shape == torch.Size([Nwl, 1, 1])
    assert torch.allclose(casted, t[..., None, None])

    # Cast tensor with additional scalar axis
    t = torch.rand(Nx)
    casted = sp.cast(t, "t", "x")
    assert casted.shape == torch.Size([1, 1, Nx])
    assert torch.allclose(casted, t[None, None, ...])
    casted = sp.cast(t, "x", "t")
    assert casted.shape == torch.Size([1, 1, Nx])
    assert torch.allclose(casted, t[None, None, ...])

    # Cast scalar
    t = torch.tensor(123)
    casted = sp.cast(t, "t")
    assert casted.shape == torch.Size([1, 1, 1])
    assert torch.allclose(casted, t[None, None, None])

    # Error on shape mismatch
    with pytest.raises(ValueError):
        sp.cast(torch.rand(Ny + 1, Nx), "y", "x")

    # Error on additional axis in the tensor
    with pytest.raises(ValueError):
        sp.cast(torch.rand(10, Ny, Nx), "y", "x")
    with pytest.raises(ValueError):
        sp.cast(torch.rand(Ny, Nx, 10), "y", "x")

    # Error on absence of axis in the tensor
    with pytest.raises(ValueError):
        sp.cast(torch.rand(Ny, Nx), "wavelength", "y", "x")
    with pytest.raises(ValueError):
        sp.cast(torch.rand(Ny, Nx), "y", "wavelength", "x")

    # Error on unknown axis
    with pytest.raises(AxisNotFound):
        sp.cast(torch.rand(Nx, Ny), "y", "m")


def test_cast_with_scalar_axis():
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 8),
        wavelength=torch.tensor(0.5),
    )

    t = torch.rand(8, 10)
    casted = sp.cast(t, "y", "x")
    assert casted.shape == torch.Size([8, 10])  # no change, wavelength is scalar


def test_to_device_canonical(device_simple: str):
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 8),
        wavelength=torch.tensor(0.5),
    )

    result = sp.to(device_simple)

    assert result is sp

    assert sp.device.type == device_simple
    assert sp.x.device.type == device_simple


def test_element_to_transfers_sim_params(device_simple: str):

    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 100),
        y=torch.linspace(-1, 1, 100),
        wavelength=torch.tensor(0.5e-6),
    )

    element = ElementToTest(sp)
    element_sp = element.simulation_parameters
    assert element_sp is not sp
    assert element_sp.equal(sp)

    element.to(device_simple)
    assert element_sp is element.simulation_parameters
    assert element_sp.device.type == device_simple

    sp.to(device_simple)
    assert element_sp is not sp
    assert element_sp.equal(sp)


def test_clear_cache():
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 100),
        y=torch.linspace(-1, 1, 100),
        wavelength=torch.tensor(0.5e-6),
    )

    # Test axis_sizes cache
    assert len(sp._cache_axis_sizes) == 0
    sp.axis_sizes(("x", "y"))
    assert len(sp._cache_axis_sizes) == 1
    sp.axis_sizes(("x", "y"))
    assert len(sp._cache_axis_sizes) == 1  # cache hit, no new entry

    sp._clear_caches()
    assert len(sp._cache_axis_sizes) == 0
    sp.axis_sizes(("x", "y"))
    assert len(sp._cache_axis_sizes) == 1

    # Test _cast_info cache
    sp._clear_caches()
    assert len(sp._cache_cast_info) == 0
    t = torch.rand(100)
    sp.cast(t, "x")
    assert len(sp._cache_cast_info) == 1
    sp.cast(t, "x")
    assert len(sp._cache_cast_info) == 1  # cache hit, no new entry

    sp._clear_caches()
    assert len(sp._cache_cast_info) == 0
    sp.cast(t, "x")
    assert len(sp._cache_cast_info) == 1


def test_repr():
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 8),
        wavelength=torch.tensor(0.5),
        pol=torch.tensor([1.0, 0.0]),
    )

    repr_str = repr(sp)
    assert "SimulationParameters" in repr_str
    assert "x=(10,)" in repr_str
    assert "y=(8,)" in repr_str
    assert "wavelength=0.5" in repr_str
    assert "pol=(2,)" in repr_str


def test_legacy():
    x = torch.linspace(-1, 1, 10)
    y = torch.linspace(-1, 1, 8)
    sp = SimulationParameters(
        x=x,
        y=y,
        wavelength=torch.tensor(0.5),
    )

    # Test legacy axes object
    with pytest.warns(DeprecationWarning):
        assert sp is sp.axes

    # Test legacy names property
    with pytest.warns(DeprecationWarning):
        assert sp.axis_names is sp.names

    # Test legacy axes_size method
    with pytest.warns(DeprecationWarning):
        # is used, not ==, because of caching
        assert sp.axis_sizes(("x", "y")) is sp.axes_size(("x", "y"))

    # Test legacy axis names (W, H)
    with pytest.warns(DeprecationWarning):
        assert sp.W is sp.x
    with pytest.warns(DeprecationWarning):
        assert sp.H is sp.y
    with pytest.warns(DeprecationWarning):
        assert sp["W"] is sp["x"]
    with pytest.warns(DeprecationWarning):
        assert sp["H"] is sp["y"]
    with pytest.warns(DeprecationWarning):
        assert torch.allclose(sp.meshgrid("x", "y")[0], sp.meshgrid("W", "H")[0])
    with pytest.warns(DeprecationWarning):
        assert torch.allclose(sp.meshgrid("x", "y")[1], sp.meshgrid("W", "H")[1])
    with pytest.warns(DeprecationWarning):
        assert sp.axis_sizes(("x",)) == sp.axis_sizes(("W",))
    with pytest.warns(DeprecationWarning):
        assert sp.index("x") == sp.index("W")

    # Test legacy initialization
    with pytest.warns(DeprecationWarning):
        sp_old = SimulationParameters(
            {
                "W": x,
                "H": y,
                "wavelength": torch.tensor(0.5),
            }
        )
    assert torch.allclose(sp_old.x, x)
    assert torch.allclose(sp_old.y, y)


def test_delattr_blocked():
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 10),
        wavelength=1.0,
    )
    with pytest.raises(AttributeError, match="read-only"):
        del sp.x
    with pytest.raises(AttributeError, match="read-only"):
        del sp.wavelength
    # Non-axis attributes can still be deleted
    sp._custom_attr = 42
    del sp._custom_attr


def test_reserved_axis_names():
    with pytest.raises(ValueError, match="conflicts with nn.Module"):
        SimulationParameters(
            x=torch.linspace(-1, 1, 10),
            y=torch.linspace(-1, 1, 10),
            wavelength=1.0,
            training=torch.tensor([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="cannot start with underscore"):
        SimulationParameters(
            x=torch.linspace(-1, 1, 10),
            y=torch.linspace(-1, 1, 10),
            wavelength=1.0,
            _hidden=torch.tensor([1.0]),
        )


def test_equal_checks_axis_order():
    sp1 = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        wavelength=torch.linspace(0.4, 0.6, 5),
        y=torch.linspace(-1, 1, 8),
    )
    sp2 = SimulationParameters(
        wavelength=torch.linspace(0.4, 0.6, 5),
        x=torch.linspace(-1, 1, 10),
        y=torch.linspace(-1, 1, 8),
    )
    # Same values but different axis ordering → not equal
    assert not sp1.equal(sp2)

    # Same ordering → equal
    sp3 = sp1.clone()
    assert sp1.equal(sp3)


def test_clone_preserves_axis_order():
    sp = SimulationParameters(
        x=torch.linspace(-1, 1, 10),
        wavelength=torch.linspace(0.4, 0.6, 5),
        y=torch.linspace(-1, 1, 8),
    )
    sp2 = sp.clone()
    assert sp.axis_names == sp2.axis_names
    assert sp._axis_names_scalar == sp2._axis_names_scalar
