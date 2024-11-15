from typing import Iterable, Any, Generator, TextIO, Generic, TypeVar
from abc import ABCMeta, abstractmethod
from io import BufferedWriter
from contextlib import contextmanager
from pathlib import Path
from numpy.typing import ArrayLike
import numpy as np


class ParameterSaveContext:
    """Generates different context managers that can be used
    to write a parameter data to output stream or file.
    """
    def __init__(
        self,
        parameter_name: str,
        directory: Path,
        stream: TextIO
    ):
        """
        Parameters
        ----------
        parameter_name : str
            the human-readable name for the parameter
        directory : str
            the directory where the generated file will be saved, if any
        stream :  TextIO
            stream where the generated text will be written, if any
        """
        self.parameter_name = parameter_name
        self._directory = directory
        self._generated_files: list[Path] = []  # paths of all generated files
        self._stream = stream

    def get_new_filepath(self, extension: str) -> Path:
        """Create a new filepath for a specific extension.
        The generated filename of a specific extension will have a unique name
        ending with `_<n>.<extension>`, where `<n>` is auto-incrementing index.

        Parameters
        ----------
        extension : str
            filename extension

        Returns
        -------
        Path
            relative path to the file
        """
        suffix = '.' + extension

        total_files = len(
            list(
                filter(
                    lambda f: f.suffix == suffix,
                    self._generated_files
                    )
                )
            )
        file_name = self.parameter_name + f'_{total_files}'

        return Path(self._directory,  file_name).with_suffix(suffix)

    @contextmanager
    def file(self, filepath: Path) -> Generator[BufferedWriter, Any, None]:
        """Context manager for the output file

        Parameters
        ----------
        filepath : Path
            filepath

        Yields
        ------
        Generator[BufferedWriter, Any, None]
            Buffer
        """
        with open(filepath, mode='wb') as file:
            yield file
        self._generated_files.append(filepath)

    @contextmanager
    def stdout(self) -> Generator[TextIO, Any, None]:
        """Context manager for the output stream

        Yields
        ------
        Generator[TextIO, Any, None]
            Buffer
        """
        yield self._stream


ParameterSaveContext_ = TypeVar(
    'ParameterSaveContext_',
    bound=ParameterSaveContext
)


class Representation(Generic[ParameterSaveContext_]):
    """Base class for a parameter representation"""
    ...


class MarkdownRepresentation(
    Representation[ParameterSaveContext_],
    metaclass=ABCMeta
):
    """Representation that can be exported to markdown file"""
    @abstractmethod
    def to_markdown(self, context: ParameterSaveContext_):
        ...


class StrRepresentation(
    Representation[ParameterSaveContext_],
    metaclass=ABCMeta
):
    """Representation that can be exported in the text format"""
    @abstractmethod
    def to_str(self, context: ParameterSaveContext_):
        ...


class ImageRepr(StrRepresentation, MarkdownRepresentation):
    """Representation of the parameter as an image.
    Image generation is based on the `matplotlib` package.
    """
    def __init__(
        self,
        value: Any,
        mpl_kwargs: dict[str, Any] | None = None,
        format: str = 'png',
        show_image: bool = True
    ):
        """
        Parameters
        ----------
        value : Any
            The image data. See `matplotlib.pyplot.imshow` docs.
        mpl_kwargs : dict[str, Any] | None, optional
            kwargs, that will be passed to `matplotlib.pyplot.imshow`,
            by default None
        format : str, optional
            the image format, by default 'png'
        """
        super().__init__()
        self.value = value
        self.format = format
        self.mpl_kwargs = mpl_kwargs if mpl_kwargs is not None else {}
        self.show_image = show_image

    def _draw_image(self, context: ParameterSaveContext, filepath: Path):
        """Draw an image into the file"""
        import matplotlib.pyplot as plt

        with context.file(filepath=filepath) as f:
            figure, ax = plt.subplots()
            ax.imshow(self.value, **self.mpl_kwargs)
            figure.savefig(f)
            plt.close(figure)

    def to_str(self, context: ParameterSaveContext):
        filepath = context.get_new_filepath(extension=self.format)

        self._draw_image(context=context, filepath=filepath)

        with context.stdout() as f:
            f.write(f'The image is saved to {filepath}\n')

    def to_markdown(self, context: ParameterSaveContext):
        filepath = context.get_new_filepath(extension=self.format)

        self._draw_image(context=context, filepath=filepath)

        with context.stdout() as f:
            f.write(f'The image is saved to `{filepath}`:\n')
            if self.show_image:
                f.write(f'![{context.parameter_name}]({filepath})')


class ReprRepr(StrRepresentation, MarkdownRepresentation):
    """Representation of the parameter as a plain text.
    The `__repr__` method is used to generate the text. 
    """
    def __init__(self, value: Any):
        """
        Parameters
        ----------
        value : Any
            object with defined `__repr__` method that will be used
            to generate plain text.
        """
        super().__init__()
        self.value = value

    def to_str(self, context: ParameterSaveContext):
        with context.stdout() as f:
            f.write(f'{repr(self.value)}\n')

    def to_markdown(self, context: ParameterSaveContext):
        with context.stdout() as f:
            f.write(f'```\n{repr(self.value)}\n```\n')


class NpyFileRepr(StrRepresentation):
    """Representation of the parameter as a `.npy` file.
    """
    def __init__(self, value: ArrayLike):
        """
        Parameters
        ----------
        value : ArrayLike
            parameter data.
        """
        super().__init__()
        self.value = value

    def _save_to_file(self, context: ParameterSaveContext, filepath: Path):
        with context.file(filepath=filepath) as f:
            np.save(f, self.value)

    def to_str(self, context: ParameterSaveContext):
        filepath = context.get_new_filepath(extension='npy')

        self._save_to_file(context, filepath)

        with context.stdout() as f:
            f.write(f'The numpy array is saved to {filepath}\n')

    def to_markdown(self, context: ParameterSaveContext):
        filepath = context.get_new_filepath(extension='npy')

        self._save_to_file(context, filepath)

        with context.stdout() as f:
            f.write(f'The numpy array is saved to `{filepath}`\n')


class ParameterSpecs:
    """Container with all representations for the parameter.
    """
    def __init__(
        self,
        parameter_name: str,
        representations: Iterable[Representation]
    ) -> None:
        """
        Parameters
        ----------
        name : str
            the parameter's name.
        representations : Iterable[ParameterRepr]
            all representations of the parameter.
        """
        self.parameter_name = parameter_name
        self.representations = representations
