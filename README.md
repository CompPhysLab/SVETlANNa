[![en](https://img.shields.io/badge/lang-EN-blue.svg)](https://github.com/CompPhysLab/SVETlANNa/blob/main/README.md)
[![ru](https://img.shields.io/badge/lang-RU-green.svg)](https://github.com/CompPhysLab/SVETlANNa/blob/main/README.ru.md)

# SVETlANNa

SVETlANNa is an open-source Python library for simulation of free-space optical set-ups and neuromorphic systems such as Diffractive Neural Networks. It is primarily built on the PyTorch framework, leveraging key features such as tensor-based computations and efficient parallel processing. At its core, SvetlANNa relies on the Fourier optics, supporting multiple propagation models, including the Angular spectrum method and the Fresnel approximation.

There is a supporting github project [SVETlANNa.docs](https://github.com/CompPhysLab/SVETlANNa.docs) containing numerous application examples in the Jupyter notebook format.

The name of the library is composed of the Russian word "svet", which is the "light" in English and the abbreviation ANN standing for an artificial neural network, and simultaneously the whole word sounds like a Russian female name Svetlana.

## Abbreviations

NN - Neural Network

ANN - Artificial Neural Network

ONN - Optical Neural Network

DONN - Diffractive Optical Neural Network

DOE - Diffractive Optical Element

SLM - Spatial Light Modulator

## Features

- based on the [PyTorch](https://pytorch.org/)
- forward propagation models include the Angular spectrum method and the Fresnel approximation
- possibility to solve the classical DOE/SLM optimization problem with the Gerchberg-Saxton and hybrid input-output algorithms
- support for custom elements and optimization methods
- support for various free-space ONN architectures including feed-forward NN, autoencoders, and recurrent NN
- cross platform
- full GPU aceleration
- companion repository with numerous .ipynb examples
- custom logging, project management, analysis, and visualization tools
- tests for the whole functionality

## Possible applications

- modeling and optimization of optical systems and optical beams propagating in free space;
- calculation of parameters of phase masks, DOEs and SLMs for both classical optical systems and neuromorphic optical computers;
- modeling and optimization of parameters of ONN and DONN for problems of all-optical classification and prediction.

# Installation, Usage and Examples

For usage there are two options, either run from the source or install from pip.

## Installation From PyPI

```bash
pip install svetlanna
```

## Running From Source

We reccomend first to install [git](https://git-scm.com) in your system.

1. Clone the repository:
```bash
  git clone https://github.com/CompPhysLab/SVETlANNa.git
```
2. Move to the project dyrectory:
```bash
  cd SVETlANNa
```
3. Create a virtual environment for Python 3.11 (see [venv documentation](https://docs.python.org/3/library/venv.html)) and activate it.
4. Istall the Poetry (version 2.0.0 or greater):
```bash
  pip install poetry
```
5. Install all depencies:
```bash
  poetry install --all-extras
```
6. Install the PyTorch (check the instructions at the official [cite](https://pytorch.org/get-started/locally/)). Choose options appropriate to your system. For running the library on a GPU you need to have an appropriate graphic card and a PyTorch version which supports this card.
```bash
  pip install torch
```

## Running Tests

The tests can be executed when the library is running from source only. PIP istallation does not include tests.

NOTE: running tests requires at least 4 Gb of free memory.

To run tests, run the following command (within the activated virtual environment in the library folder)
```bash
  pytest
```

## Documentation

[Documentation](https://compphyslab.github.io/SVETlANNa/)

## Running examples with library installation

Examples of various scenarios and setups are provided in the repository [SVETlANNa.docs](https://github.com/CompPhysLab/SVETlANNa.docs).

To run the example locally follow the instructions:
1. Clone the repository (you should have an installed [git](https://git-scm.com)):
```bash
  git clone https://github.com/CompPhysLab/SVETlANNa.docs.git
```
2. Go to the project directory:
```bash
  cd SVETlANNa.docs
```
3.Create a virtual environment for Python 3.11 (see [venv documentation](https://docs.python.org/3/library/venv.html)) and activate it. All subsequent actions should be done within the activated environment.
4. Install the main library:
```bash
  pip install svetlanna
```
5. Install the PyTorch (check the instructions at the official [cite](https://pytorch.org/get-started/locally/)). Choose options appropriate to your system. For running the library on a GPU you need to have an appropriate graphic card and a PyTorch version which supports this card.
6. If you do not have the Jupyter notebook in your system, install it:
```bash
  pip install notebook
```
7. Install additional dependencies:
```bash
  pip install reservoirpy matplotlib tqdm requests av scikit-image py-cpuinfo gputil
```
8. Run the Jupyter notebook:
```bash
  jupyter notebook
```
and open the page in your browser (see “Or copy and paste one of these URLs:” in a terminal window)
9. Go to the folder SVETlANNa.docs. Examples are grouped into subject subfolders in form of .ipynb files. To run them you need to open a file in the Jupyter notebook and run all code cells (key Shift+Enter).

## Running exaples with SVETlANNa installed from source

1. Clone the repository (you should have an installed [git](https://git-scm.com)):
```bash
	git clone https://github.com/CompPhysLab/SVETlANNa.docs.git 
```
All subsequent actions should be done within the activated environment, which was used to install the SVETlANNa library from source.
2. If you do not have the Jupyter notebook in your system, install it:
```bash
  pip install notebook
```
3. Install additional dependencies:
```bash
  pip install reservoirpy matplotlib tqdm requests av scikit-image py-cpuinfo gputil
```
4. Run the Jupyter notebook:
```bash
  jupyter notebook
```
and open the page in your browser (see “Or copy and paste one of these URLs:” in a terminal window)
5. Go to the folder SVETlANNa.docs. Examples are grouped into subject subfolders in form of .ipynb files. To run them you need to open a file in the Jupyter notebook and run all code cells (key Shift+Enter).

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
