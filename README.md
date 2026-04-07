[![en](https://img.shields.io/badge/lang-EN-blue.svg)](https://github.com/CompPhysLab/SVETlANNa/blob/main/README.md)
[![ru](https://img.shields.io/badge/lang-RU-green.svg)](https://github.com/CompPhysLab/SVETlANNa/blob/main/README.ru.md)

# SVETlANNa

SVETlANNa is an open-source Python library for simulating free-space optical setups and neuromorphic systems such as Diffractive Neural Networks.
It is built on the PyTorch framework, leveraging key features such as tensor-based computations and automatic differentiation.

At its core, SVETlANNa relies on Fourier optics and includes several optical elements such as free space, apertures, phase masks, thin lenses, and Spatial Light Modulators (SLMs).

The library name combines the Russian word "svet" ("light" in English) and the abbreviation ANN (artificial neural network).
At the same time, the full word sounds like the Slavic female name Svetlana.

## Documentation
Documentation for SVETlANNa is available at [compphyslab.github.io/SVETlANNa.docs](https://compphyslab.github.io/SVETlANNa.docs/).

There is also a supporting GitHub repository, [SVETlANNa.docs](https://github.com/CompPhysLab/SVETlANNa.docs), containing numerous application examples in Jupyter notebook format.


## Quick Start
```python
import torch
import svetlanna as sv
from svetlanna.units import ureg

# define the computational grid and the wavelength of light
sim_params = sv.SimulationParameters(
    x=torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 256),
    y=torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 256),
    wavelength=10 * ureg.um,
)

setup = sv.LinearOpticalSetup([
    sv.elements.RoundAperture(sim_params, radius=2 * ureg.mm),
    sv.elements.FreeSpace(sim_params, distance=10 * ureg.cm, method='ASM'),
])

incident_field = sv.Wavefront.plane_wave(sim_params)
output_field = setup(incident_field)

```

## Features

- Free-space propagation solvers, including the Angular spectrum method (ASM) and the Rayleigh-Sommerfeld convolution (RSC). See [this work](https://doi.org/10.1364/OL.393111) for a definition of the methods and a comparison between them.
- Support for solving classical diffractive optical element and SLM optimization problems with the Gerchberg-Saxton and Hybrid Input-Output algorithms.
- Flexible API that supports custom elements.
- Native GPU acceleration for all computations.
- Visualization tools.


## Possible applications

- Modeling and optimization of optical systems and optical beams propagating in free space.
- Calculation of phase-mask, Diffractive Optical Element (DOE), and SLM parameters for both classical optical systems and neuromorphic optical computers.
- Modeling and optimization of Optical Neural Network and Diffractive Optical Neural Network parameters for different tasks.

## Installation

You can install SVETlANNa from PyPI using pip:
```bash
pip install svetlanna
```
You should install PyTorch separately, following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/), to ensure compatibility with your system and desired features (e.g., CUDA support).

## Examples

Result of training a feed-forward optical neural network for the MNIST classification task: the image of the digit "8" is passed through a stack of 10 phase plates with adjusted phase masks. Selected detector regions correspond to different classes of digits. The predicted class is identified by the detector region with the maximum optical intensity.

Examples of visualization of optical setups and optical fields:

<img src="./pics/visualization.png" alt="drawing" width="400"/>

Example of a five-layer Diffractive Optical Neural Network trained to recognize numbers from the MNIST database:

<img src="./pics/MNIST example 1.png" alt="drawing" width="400"/>

<img src="./pics/MNIST example 2.png" alt="drawing" width="400"/>

<img src="./pics/MNIST example 3.png" alt="drawing" width="400"/>

# Contributing

Contributions are always welcome!
See `CONTRIBUTING.md` for ways to get started.

# Acknowledgements

The initial work on this repository was supported by the [Foundation for Assistance to Small Innovative Enterprises](https://en.fasie.ru/).

# Authors

- [@aashcher](https://github.com/aashcher)
- [@alexeykokhanovskiy](https://github.com/alexeykokhanovskiy)
- [@Den4S](https://github.com/Den4S)
- [@djiboshin](https://github.com/djiboshin)
- [@Nevermind013](https://github.com/Nevermind013)

# License

[Mozilla Public License Version 2.0](https://www.mozilla.org/en-US/MPL/2.0/)
