from typing import Iterable, Any, Generator, TextIO, Generic, TypeVar, Literal
from typing import Protocol
from abc import ABCMeta, abstractmethod
from io import BufferedWriter, BytesIO
from contextlib import contextmanager
from pathlib import Path
from PIL import Image
from numpy.typing import ArrayLike
import numpy as np
import torch
import numbers
import base64


class ParameterSaveContext:
    """Generates different context managers that can be used
    to write a parameter data to output stream or file.
    """
    def __init__(
        self,
        parameter_name: str,
        directory: Path
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

        # calculate total number of created files with the same extension
        total_files = len(
            list(
                filter(
                    lambda f: f.suffix == suffix,
                    self._generated_files
                    )
                )
            )

        # name of the new file ending with `_<n>.<extension>`
        file_name = self.parameter_name + f'_{total_files}'

        # save filepath of the file
        filepath = Path(self._directory,  file_name).with_suffix(suffix)
        self._generated_files.append(filepath)

        # create a new folder for the file if there is none
        Path.mkdir(self._directory, parents=True, exist_ok=True)

        return filepath

    def rel_filepath(self, filepath: Path) -> Path:
        """Get relative to specs file filepath

        Parameters
        ----------
        filepath : Path
            absolute path

        Returns
        -------
        Path
            relative path
        """
        return filepath.relative_to(self._directory.parent)

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
    def to_markdown(self, out: TextIO, context: ParameterSaveContext_) -> None:
        """Write the parameter related data to be shown in a markdown file.
        The text should be written to the `out` stream.

        Parameters
        ----------
        out : TextIO
            output text stream
        context : ParameterSaveContext_
            the parameter save context
        """


class StrRepresentation(
    Representation[ParameterSaveContext_],
    metaclass=ABCMeta
):
    """Representation that can be exported in the text format"""
    @abstractmethod
    def to_str(self, out: TextIO, context: ParameterSaveContext_) -> None:
        """Write the parameter related data to be shown as a plain text.
        The text should be written to the `out` stream.

        Parameters
        ----------
        out : TextIO
            output text stream
        context : ParameterSaveContext_
            the parameter save context
        """


class HTMLRepresentation(
    Representation[ParameterSaveContext_],
    metaclass=ABCMeta
):
    """Representation that can be exported to the HTML"""
    @abstractmethod
    def to_html(self, out: TextIO, context: ParameterSaveContext_) -> None:
        """Write the parameter related data to be shown in a HTML file.
        The text should be written to the `out` stream.

        Parameters
        ----------
        out : TextIO
            output text stream
        context : ParameterSaveContext_
            the parameter save context
        """


class ImageRepr(StrRepresentation, MarkdownRepresentation, HTMLRepresentation):
    """Representation of the parameter as an image.
    Image generation is based on the `pillow` package.
    """
    def __init__(
        self,
        value: Any,
        mode: Literal['1', 'L', 'LA', 'I', 'P', 'RGB', 'RGBA'] = 'L',
        format: str = 'png',
        show_image: bool = True
    ):
        """
        Parameters
        ----------
        value : Any
            The image data. See `matplotlib.pyplot.imshow` docs.
        mode : Literal['1', 'L', 'LA', 'I', 'P', 'RGB', 'RGBA'], optional
            the mode of the image, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes.
            By default `L`
        format : str, optional
            the image format, by default 'png'
        """
        super().__init__()
        self.value = value
        self.format = format
        self.mode = mode
        self.show_image = show_image

    def draw_image(
        self,
        context: ParameterSaveContext,
        filepath: Path
    ) -> Image.Image:
        """Draw image into the file, using `pillow` package.

        Parameters
        ----------
        context : ParameterSaveContext
            the parameter save context
        filepath : Path
            path to the image file to be created
        """

        with context.file(filepath=filepath) as f:
            image = Image.fromarray(self.value, mode=self.mode)
            image.save(f, format=self.format)

        return image

    def to_str(self, out: TextIO, context: ParameterSaveContext):
        filepath = context.get_new_filepath(extension=self.format)

        self.draw_image(context=context, filepath=filepath)

        out.write(f'The image is saved to {context.rel_filepath(filepath)}\n')

    def to_markdown(self, out: TextIO, context: ParameterSaveContext):
        filepath = context.get_new_filepath(extension=self.format)
        rel_filepath = context.rel_filepath(filepath)

        self.draw_image(context=context, filepath=filepath)

        out.write(f'The image is saved to `{rel_filepath}`\n')
        if self.show_image:
            out.write(f'\n![{context.parameter_name}]({rel_filepath})\n\n')

    def to_html(self, out: TextIO, context: ParameterSaveContext):

        image = Image.fromarray(self.value, mode=self.mode)

        if self.show_image:
            buffer = BytesIO()
            image.save(buffer, format=self.format)
            # encode image using base64
            encoded_image = base64.b64encode(buffer.getvalue()).decode()

            # src for <img/> HTML element
            img_src = f"data:image/{self.format};base64, {encoded_image}"

            out.write(
                f'\n<img class="spec-img" src="{img_src}"/>\n'
            )


class ReprRepr(StrRepresentation, MarkdownRepresentation, HTMLRepresentation):
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

    def to_str(self, out: TextIO, context: ParameterSaveContext):
        out.write(f'{repr(self.value)}\n')

    def to_markdown(self, out: TextIO, context: ParameterSaveContext):
        out.write(f'```\n{repr(self.value)}\n```\n')

    def to_html(self, out: TextIO, context: Any):
        self.to_str(out, context)


class NpyFileRepr(StrRepresentation, MarkdownRepresentation):
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

    def save_to_file(self, context: ParameterSaveContext, filepath: Path):
        """Save the parameter related data to `npy` file.

        Parameters
        ----------
        context : ParameterSaveContext
            the parameter save context
        filepath : Path
            path to the file to be created
        """
        with context.file(filepath=filepath) as f:
            np.save(f, self.value)

    def to_str(self, out: TextIO, context: ParameterSaveContext):
        filepath = context.get_new_filepath(extension='npy')
        rel_filepath = context.rel_filepath(filepath)

        self.save_to_file(context, filepath)

        out.write(f'The numpy array is saved to {rel_filepath}\n')

    def to_markdown(self, out: TextIO, context: ParameterSaveContext):
        filepath = context.get_new_filepath(extension='npy')
        rel_filepath = context.rel_filepath(filepath)

        self.save_to_file(context, filepath)

        out.write(f'The numpy array is saved to `{rel_filepath}`\n')


class PrettyReprRepr(ReprRepr, HTMLRepresentation):
    """Same as ReprRepr but with better handling of
    Parameters and BoundedParameter"""
    def __init__(
        self,
        value: Any,
        units: str | None = None
    ):
        """Representation of the parameter as a plain text.
        The `__repr__` method is used to generate the
        text if the `value` is not `torch.Tensor` or `Parameter`.

        Parameters
        ----------
        value : Any
            object to generate plain text of.
        units : str | None, optional
            units of the value, by default None
        """
        super().__init__(value)
        self.units = units

    def _repr(self) -> str:
        units_suffix = '' if self.units is None else f' [{self.units}]'
        class_name: str = self.value.__class__.__name__

        if isinstance(self.value, torch.Tensor):
            shape = self.value.shape

            # If the value is scalar, it can be directly printed out
            if len(shape) == 0:

                try:
                    from svetlanna import ConstrainedParameter

                    # Print minimum and maximum values for ConstrainedParameter
                    if isinstance(self.value, ConstrainedParameter):

                        min_val = self.value.min_value.item()
                        max_val = self.value.max_value.item()

                        s = f'{class_name}\n'
                        s += f'  ┏ min value {min_val}{units_suffix}\n'
                        s += f'  ┗ max value {max_val}{units_suffix}\n'
                        return s + f'{self.value.item()}{units_suffix}'

                except ImportError:
                    pass

                return f'{class_name}\n{self.value.item()}{units_suffix}'

            # Print shape of the tensor
            shape_str = "x".join(map(str, shape))
            return f'{class_name} of size ({shape_str}){units_suffix}'

        # If the value is number, it can be directly printed out
        if isinstance(self.value, numbers.Number):
            return f'{self.value}{units_suffix}'

        return repr(self.value)

    def to_str(self, out: TextIO, context: ParameterSaveContext):
        out.write(f'{self._repr()}\n')

    def to_markdown(self, out: TextIO, context: ParameterSaveContext):
        out.write(f'```\n{self._repr()}\n```\n')

    def to_html(self, out: TextIO, context: ParameterSaveContext):
        self.to_str(out, context)


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


class SubelementSpecs:
    """Container for named subelement
    """
    def __init__(
        self,
        subelement_type: str,
        subelement: 'Specsable'
    ):
        """
        Parameters
        ----------
        subelement_type : str
            human-readable type of the subelement.
        subelement : Specsable
            the subelement.
        """
        self.subelement_type = subelement_type
        self.subelement = subelement


class Specsable(Protocol):
    """Represents any specsable object"""
    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        ...
