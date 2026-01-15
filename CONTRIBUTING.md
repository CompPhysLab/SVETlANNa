This repo uses several tools to ensure code quality and consistency, including Black for code formatting, pytest for testing, and Sphinx for documentation.

# Development Environment

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
- The Flake8 and Mypy extensions lint errors during development, so you can fix them on the spot.

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