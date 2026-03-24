import torch
import pytest
import numpy as np
import sympy as sp  # type: ignore
from typing import Callable
from svetlanna import Wavefront, SimulationParameters

import scipy.special
from svetlanna.wavefront import hermite_poly


def _hermite_gauss_symbolic():
    x, y, w0, wavelength, z, dx, dy = sp.symbols("x y w0 wavelength z dx dy", real=True)
    m, n = sp.symbols("m n", integer=True, nonnegative=True)

    k = 2 * sp.pi / wavelength
    zR = sp.pi * w0**2 / wavelength

    w = w0 * sp.sqrt(1 + (z / zR) ** 2)
    R = z * (1 + (zR / z) ** 2)
    zeta = (1 + n + m) * sp.atan(z / zR)

    r2 = (x - dx) ** 2 + (y - dy) ** 2

    Hm = sp.hermite(m, sp.sqrt(2) * (x - dx) / w)
    Hn = sp.hermite(n, sp.sqrt(2) * (y - dy) / w)

    E = (
        (w0 / w)
        * Hn
        * Hm
        * sp.exp(-r2 / w**2)
        * sp.exp(sp.I * (k * z + k * r2 / (2 * R) - zeta))
    )
    E = sp.simplify(E)

    return sp.lambdify(
        (x, y, w0, wavelength, z, dx, dy, m, n),
        E,
        modules=["numpy", {"hermite": scipy.special.eval_hermite}],
    )


hermite_gauss_analytical: Callable[..., np.ndarray] = _hermite_gauss_symbolic()


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_hermite_polynomial(n: int):
    """
    Test that the first few Hermite polynomials are correct
    """
    x = torch.linspace(-3, 3, 200)

    torch.testing.assert_close(
        hermite_poly(n, x),
        torch.from_numpy(scipy.special.hermite(n)(x.numpy()).astype(np.float32)),
    )


@pytest.mark.parametrize("distance", [0.1, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("waist_radius", [0.3, 0.7])
def test_zero_orders(
    distance: float, waist_radius: float, sim_params_simple: SimulationParameters
):
    """
    Test that HG00 is the same as Gaussian beam
    """

    wf_hg00 = Wavefront.hermite_gauss(
        sim_params_simple, waist_radius=waist_radius, distance=distance, m=0, n=0
    )
    wf_gauss = Wavefront.gaussian_beam(
        sim_params_simple, waist_radius=waist_radius, distance=distance
    )

    torch.testing.assert_close(wf_hg00, wf_gauss)


@pytest.mark.parametrize("distance", [0.1, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("waist_radius", [0.3, 0.7])
@pytest.mark.parametrize("m,n", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)])
@pytest.mark.parametrize("dx", (1.0, 3))
@pytest.mark.parametrize("dy", (1.0, -3))
@pytest.mark.parametrize("wavelength", (0.5, torch.tensor([0.5, 1.0])))
def test_hermite_gauss(distance, waist_radius, m, n, dx, dy, wavelength):
    """
    Test that the Hermite-Gauss modes match the numerical implementation for a range of parameters
    """
    # Params
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-3, 3, 150),
            "y": torch.linspace(-3, 3, 150),
            "wavelength": wavelength,
        }
    )

    # Grid for wavefront calc
    x = sim_params.cast(sim_params.x, "x")
    y = sim_params.cast(sim_params.y, "y")
    wl = sim_params.cast(sim_params.wavelength, "wavelength")

    E_svet = Wavefront.hermite_gauss(
        sim_params,
        waist_radius=waist_radius,
        distance=distance,
        dx=dx,
        dy=dy,
        m=m,
        n=n,
    )
    E_num = hermite_gauss_analytical(
        x.numpy(), y.numpy(), waist_radius, wl.numpy(), distance, dx, dy, m, n
    ).astype(np.complex64)

    torch.testing.assert_close(
        E_svet,
        torch.from_numpy(E_num),
    )

    I_svet = np.abs(E_svet.numpy()) ** 2
    I_num = np.abs(E_num) ** 2

    I_svet = I_svet / I_svet.max()
    I_num = I_num / I_num.max()

    # Quality metrics
    mse = np.mean((I_svet - I_num) ** 2)
    correlation = np.corrcoef(I_svet.flatten(), I_num.flatten())[0, 1]

    assert (
        correlation >= 1.0 - 1e-6
    ), f"Low correlation: {correlation:.4f} for HG{m}{n}, z={distance}"
    assert mse < 1e-7, f"HIGH MSE: {mse:.4f} for HG{m}{n}, z={distance}"


def test_hermite_gauss_orthogonality():
    """
    Modes Orthogonality check
    """
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-5, 5, 300),
            "y": torch.linspace(-5, 5, 300),
            "wavelength": 1.0,
        }
    )

    waist_radius = 1.0
    modes = [(0, 0), (1, 0), (0, 1), (1, 1)]

    overlap_matrix = np.zeros((len(modes), len(modes)))

    for i, (m1, n1) in enumerate(modes):
        wf1 = Wavefront.hermite_gauss(sim_params, waist_radius=waist_radius, m=m1, n=n1)
        norm1 = torch.sqrt(
            torch.sum(wf1.intensity)
            * (sim_params.x[1] - sim_params.x[0])
            * (sim_params.y[1] - sim_params.y[0])
        )

        for j, (m2, n2) in enumerate(modes):
            wf2 = Wavefront.hermite_gauss(
                sim_params, waist_radius=waist_radius, m=m2, n=n2
            )
            norm2 = torch.sqrt(
                torch.sum(wf2.intensity)
                * (sim_params.x[1] - sim_params.x[0])
                * (sim_params.y[1] - sim_params.y[0])
            )

            # Overlap integral
            overlap = (
                torch.sum(torch.conj(wf1) * wf2)
                * (sim_params.x[1] - sim_params.x[0])
                * (sim_params.y[1] - sim_params.y[0])
            )

            overlap_normalized = torch.abs(overlap) / (norm1 * norm2)
            overlap_matrix[i, j] = overlap_normalized.item()

    for i in range(len(modes)):
        assert (
            abs(overlap_matrix[i, i] - 1.0) < 0.1
        ), f"Diag {i}: {overlap_matrix[i, i]}"
        for j in range(len(modes)):
            if i != j:
                assert (
                    overlap_matrix[i, j] < 0.1
                ), f"Non-diag {i},{j}: {overlap_matrix[i, j]}"


@pytest.mark.parametrize(
    "m,n,expected_nodes_x,expected_nodes_y",
    [
        (0, 0, 0, 0),
        (1, 0, 1, 0),
        (0, 1, 0, 1),
        (2, 0, 2, 0),
        (0, 2, 0, 2),
        (1, 1, 1, 1),
    ],
)
def test_hermite_gauss_nodes(m, n, expected_nodes_x, expected_nodes_y):
    """
    Number of nodes check
    """
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-5, 5, 500),
            "y": torch.linspace(-5, 5, 500),
            "wavelength": 1.0,
        }
    )

    waist_radius = 1.0

    wf = Wavefront.hermite_gauss(sim_params, waist_radius=waist_radius, m=m, n=n)

    intensity = wf.intensity.numpy()

    # Central sections
    cx, cy = intensity.shape[0] // 2, intensity.shape[1] // 2
    horizontal = intensity[cx, :]
    vertical = intensity[:, cy]

    horizontal = horizontal / horizontal.max()
    vertical = vertical / vertical.max()

    threshold = 0.05

    def count_nodes(line):
        # Areas lower than threshold
        below = line < threshold
        # Finding changes
        changes = np.where(np.diff(below.astype(int)) != 0)[0]
        # Number of nodes = number of transitions down
        nodes = 0
        for i in changes:
            if below[i + 1] and not below[i]:
                nodes += 1
        return nodes

    nodes_x = count_nodes(horizontal)
    nodes_y = count_nodes(vertical)

    print(f"\nHG{m}{n}: Nodes expected: ({expected_nodes_x}, {expected_nodes_y})")
    print(f"Recieved nodes: ({nodes_x}, {nodes_y})")

    # check with a margin (sometimes nodes can merge)
    assert (
        nodes_x >= expected_nodes_x - 1
    ), f"X: expected ~{expected_nodes_x}, recieved {nodes_x}"
    assert (
        nodes_x <= expected_nodes_x + 1
    ), f"X: expected ~{expected_nodes_x}, recieved {nodes_x}"
    assert (
        nodes_y >= expected_nodes_y - 1
    ), f"Y: expected ~{expected_nodes_y}, recieved {nodes_y}"
    assert (
        nodes_y <= expected_nodes_y + 1
    ), f"Y: expected ~{expected_nodes_y}, recieved {nodes_y}"
