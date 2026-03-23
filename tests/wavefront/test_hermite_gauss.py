import torch
import pytest
import numpy as np
from scipy.special import hermite, factorial
from svetlanna import Wavefront, SimulationParameters


def hermite_gauss_numerical(x, y, w0, wavelength, z, m, n, dx=0, dy=0):
    """
    Analytic realisation of Hermite-Gauss modes using scipy
    """
    k = 2 * np.pi / wavelength
    zR = np.pi * w0**2 / wavelength

    # Coordinates
    X = x - dx
    Y = y - dy

    if z == 0:
        # Calculation at waist point
        xi_x = np.sqrt(2) * X / w0
        xi_y = np.sqrt(2) * Y / w0

        Hx = hermite(n)(xi_x)
        Hy = hermite(m)(xi_y)

        # Norm
        norm = np.sqrt(2 / (2**n * factorial(n) * np.pi)) * np.sqrt(
            2 / (2**m * factorial(m) * np.pi)
        )

        E = norm / w0 * Hx * Hy * np.exp(-(X**2 + Y**2) / w0**2)

    else:
        # Calculating for propagation case
        w = w0 * np.sqrt(1 + (z / zR) ** 2)
        R = z * (1 + (zR / z) ** 2)
        gouy = (m + n + 1) * np.arctan(z / zR)

        xi_x = np.sqrt(2) * X / w
        xi_y = np.sqrt(2) * Y / w

        Hx = hermite(n)(xi_x)
        Hy = hermite(m)(xi_y)

        norm = np.sqrt(2 / (2**n * factorial(n) * np.pi)) * np.sqrt(
            2 / (2**m * factorial(m) * np.pi)
        )

        E = (
            (norm * w0 / w)
            * Hx
            * Hy
            * np.exp(-(X**2 + Y**2) / w**2)
            * np.exp(1j * (k * z + k * (X**2 + Y**2) / (2 * R) - gouy))
        )

    return E


@pytest.mark.parametrize("distance", [0.1, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("waist_radius", [0.3, 0.7])
@pytest.mark.parametrize("m,n", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)])
def test_hermite_gauss_vs_numerical(distance, waist_radius, m, n):
    """
    SVETlANNa realisation
    """
    # Params
    wavelength = 0.5
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-3, 3, 150),
            "y": torch.linspace(-3, 3, 150),
            "wavelength": wavelength,
        }
    )

    # SVETlANNa wavefront
    wf_svetlanna = Wavefront.hermite_gauss(
        sim_params, waist_radius=waist_radius, distance=distance, dx=0, dy=0, m=m, n=n
    )

    # Grid for wavefront calc
    x_np = sim_params.x.numpy()
    y_np = sim_params.y.numpy()
    X, Y = np.meshgrid(x_np, y_np, indexing="ij")

    # Numeric field
    E_num = hermite_gauss_numerical(X, Y, waist_radius, wavelength, distance, m, n)

    # SVETlANNa wavefront to numpy
    E_svet = wf_svetlanna.detach().cpu().numpy()

    I_svet = np.abs(E_svet) ** 2
    I_num = np.abs(E_num) ** 2

    I_svet = I_svet / I_svet.max()
    I_num = I_num / I_num.max()

    # Quality metrics
    mse = np.mean((I_svet - I_num) ** 2)
    correlation = np.corrcoef(I_svet.flatten(), I_num.flatten())[0, 1]

    assert (
        correlation > 0.98
    ), f"Low correlation: {correlation:.4f} for HG{m}{n}, z={distance}"
    assert mse < 0.01, f"HIGH MSE: {mse:.4f} for HG{m}{n}, z={distance}"


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
