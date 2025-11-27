This repo uses several tools to ensure code quality and consistency, including Black for code formatting, pytest for testing, and Sphinx for documentation.

# Development Environment Setup

Poetry is used to manage dependencies and virtual environments. To set up the development environment, follow these steps:

1. Install Poetry
2. Inside the repository, run
```bash
poetry install --all-extras
```
to install the dependencies and set up the virtual environment.


## Code Formatting and Linting

We use `black` for code formatting.
To run black on specific files or directories, use the following command:
```bash
poetry run black <file_or_directory>
```

We use flake8 as a code linter.
To run flake8 on specific files or directories, use the following command:
```bash
poetry run flake8 <file_or_directory>
```

## Type Checking

To run mypy for type checking, use the following command:
```bash
poetry run mypy <file_or_directory>
```

## Tests

Pytest is used for testing:

```bash
poetry run pytest
```

## Docs

We use `numpy` style for docstrings.



# Committing

```bash
poetry run black svetlanna
poetry run mypy svetlanna
poetry run flake8 svetlanna
poetry run pytest
```