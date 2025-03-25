
Release is planned for May 2025.

# SVETlANNa

SVETlANNa is an open-source Python library for simulation of free-space optical set-ups and neuromorphic systems such as Diffractive Neural Networks. It is primarily built on the PyTorch framework, leveraging key features such as tensor-based computations and efficient parallel processing. At its core, SvetlANNa relies on the Fourier optics, supporting multiple propagation models, including the Angular spectrum method and the Fresnel approximation.

There is a supporting github project containing numerous application examples in the Jupyter notebok format. This project will be opened upon the release.

The name of the library is composed of the Russian word "svet", which is the light in English and the abbreviation ANN - artificial neural network, and simultaneously this word sounds like a Russian female name Svetlana.
## Abbreviations

NN - Neural Network

ANN - Artificial Neural Network

ONN - Optical Neural Network

DONN - Diffractive Optical Neural Network

DOE - Diffractive Optical Element

SLM - Spatial Light Modulator
## Features

- forward propagation models include the Angular spectrum method and the Fresnel approximation
- possibility to solve the classical DOE/SLM optimization problem with the Gerchberg-Saxton and hybrid inpu-output algorithms
- support for custom elements and optimization methods
- support for various free-space ONN architectures including feed-forward NN, autoencoders, and recurrent NN
- cross platform
- full GPU aceleration
- companion repository with numerous .ipynb examples
- custom logging, project management, and analysis tools
- tests for the whole functionality


# Installation, Usage and Examples

## Running From Source

1. Create a virtual environment
2. Install the PyTorch (it is up to the user to choose a version)
```bash
  pip install torch
```
3. Istall the Poetry, or check that its version is 2.0.0 or greater
```bash
  pip install poetry
```
4. In the library folder execute the command
```bash
  poetry install
```

## Installation From PIP

Not yet ...
## Running Tests

To run tests, run the following command

```bash
  pytest
```

## Documentation

[Documentation]()


## Examples

Result of training the feed-forward optical neural network for the MNIST classification task: The image of the figure "8" is passed through a stack of 10 phase plates with adjusted phase masks. Selected regions of the detector correspond to different classes of figures. The class of the figure is identified by the detector region that measures the maximum optical intensity.

Examples of visualzation of optical set-ups and optical fields:

<img src="./pics/visualization.png" alt="drawing" width="400"/>

Example a of five-layer DONN trained to recognize numbers from the MNIST database:

<img src="./pics/MNIST example 1.png" alt="drawing" width="400"/>

<img src="./pics/MNIST example 2.png" alt="drawing" width="400"/>

<img src="./pics/MNIST example 3.png" alt="drawing" width="400"/>


# Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


# Acknowledgements

The work on this repository was initiated within the grant by the [Foundation for Assistance to Small Innovative Enterprises](https://en.fasie.ru/)
# Authors

- [@aashcher](https://github.com/aashcher)
- [@alexeykokhanovskiy](https://github.com/alexeykokhanovskiy)
- [@Den4S](https://github.com/Den4S)
- [@djiboshin](https://github.com/djiboshin)
- [@Nevermind013](https://github.com/Nevermind013)

# License

[Mozilla Public License Version 2.0](https://www.mozilla.org/en-US/MPL/2.0/)
