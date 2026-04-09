# Contributing to SVETlANNa

The SVETlANNa project welcomes contributions from the community!
Whether you're fixing a bug, adding a new feature, or improving documentation, your help is appreciated.

## Asking for help

If you have any questions or need assistance, please don't hesitate to [open an issue](https://github.com/CompPhysLab/SVETlANNa/issues/new/choose) or reach out to the maintainers.

## Reporting Issues

If you encounter any bugs, issues, or concerns, please report them by [opening an issue](https://github.com/CompPhysLab/SVETlANNa/issues/new/choose).
When reporting an issue, please provide as much detail as possible, including steps to reproduce the issue, expected behavior, and any relevant error messages or logs.

## Suggesting Features

You can suggest new features in two ways:
1. By [opening an issue](https://github.com/CompPhysLab/SVETlANNa/issues/new/choose) and describing the feature you'd like to see.
Describe the feature you are requesting and its use case. If you have a simple code example that illustrates the feature, please include it in the issue description.
It is very helpful to include a code implementation if you want to add more physics-related functionality.
2. By [opening a pull request](https://github.com/CompPhysLab/SVETlANNa/pulls).
You should first fork the repository, make your changes in a separate branch, and then open a pull request to merge your changes into the main branch.
When opening a pull request, please provide a clear description of the changes you made and the motivation behind them.
The rules for contributing code are as follows:
    - DO NOT commit any binary files, Jupyter notebooks, images, or any files that are not directly related to the codebase.
    - It is recommended to use the development environment tools (Black, Flake8, Mypy) to ensure code quality and consistency (see the [development environment](#development-environment) section below).
    - It is recommended that all code be tested with pytest and have good test coverage.

In both cases, the maintainers can reject or ask for changes to your contribution if it does not align with the project's goals.



# Development Environment

This repository uses several tools to ensure code quality and consistency, including Black for code formatting, pytest for testing, and Poetry for dependency management.

## Setup

[Poetry](https://github.com/python-poetry/poetry) is used to manage dependencies and virtual environments.
To set up the development environment, follow these steps:

1. [Install Poetry](https://python-poetry.org/docs/#installation) globally on your machine.
2. Inside the repository, run:
```bash
poetry install --all-extras
```
This installs the dependencies and sets up the virtual environment.


## Code Formatting and Linting

We use [Black](https://github.com/psf/black) for code formatting.
To run Black on specific files or directories, use:
```bash
poetry run black <file_or_directory>
```

We use [Flake8](https://github.com/PyCQA/flake8) as a code linter.
To run Flake8 on specific files or directories, use:
```bash
poetry run flake8 <file_or_directory>
```

## Type Checking

To run [Mypy](https://github.com/python/mypy) for type checking, use:
```bash
poetry run mypy <file_or_directory>
```

## Tests

Pytest is used for testing:

```bash
poetry run pytest
```

To generate coverage reports, use:
```bash
poetry run pytest --cov=svetlanna --cov-report=html
```

## VS Code

Extensions for Black, Flake8, and Mypy are available.
Please install them so you can see inline hints and diagnostics during development (see `.vscode/extensions.json` file).
- The Black extension allows you to run formatting automatically on save.
- The Flake8 and Mypy extensions report errors during development, so you can fix them on the spot.

## Docs

We use `numpy` style for docstrings.



# Committing

Before committing, run:
```bash
poetry run black svetlanna
poetry run mypy svetlanna
poetry run flake8 svetlanna
poetry run pytest
```
