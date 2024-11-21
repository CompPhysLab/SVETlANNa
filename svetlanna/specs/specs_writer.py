from typing import Iterable, TextIO, Protocol, TypeVar, Generic
from .specs import ParameterSpecs
from .specs import ParameterSaveContext, Representation
from .specs import StrRepresentation, MarkdownRepresentation
from pathlib import Path
import itertools
from dataclasses import dataclass
from io import StringIO


_HTML_STYLE = """<style>
.svetlanna-specs-container {
  display: block;
}
.svetlanna-specs-details {
  padding-left: 2rem;
}
.svetlanna-specs-parameter-name {
    margin-top: 0.5rem;
}
.svetlanna-specs-representation {
  margin-bottom: 0.5rem;
  padding-left: 2rem;
}
.svetlanna-specs-container > details {
  border: 1px solid gray;
  font-family: monospace;
}
.svetlanna-specs-container > details > summary {
  background-color: #cff1f0;
  padding: 0.5rem;
  color: black;
}
.svetlanna-specs-representation > pre {
  white-space: pre-wrap;
}
/*
.svetlanna-specs-container > details > summary:before {
  content: "> ";
}
.svetlanna-specs-container > details[open] > summary:before {
  content: "- ";
}
*/
</style>
"""


class Specsable(Protocol):
    def to_specs(self) -> Iterable['ParameterSpecs']:
        ...


_T = TypeVar('_T')


@dataclass(frozen=True, slots=True)
class _IndexedObject(Generic[_T]):
    value: _T
    index: int


@dataclass
class _WriterContext:
    parameter_name: _IndexedObject[str]
    representation: _IndexedObject[Representation]
    context: ParameterSaveContext


def _context_generator(
    element: Specsable,
    element_index: int,
    directory: str | Path
):
    specs_directory = Path(
        directory, f'{element_index}_{element.__class__.__name__}'
    )
    Path.mkdir(specs_directory, parents=True, exist_ok=True)

    parameter_representations = {}

    for spec in element.to_specs():
        parameter_name = spec.parameter_name
        representations = spec.representations

        if parameter_name in parameter_representations:
            parameter_representations[parameter_name].append(representations)
        else:
            parameter_representations[parameter_name] = [representations]

    for parameter_index, (parameter_name, representations) in enumerate(parameter_representations.items()):

        context = ParameterSaveContext(
            parameter_name=parameter_name,
            directory=specs_directory
        )

        for representation_index, representation in enumerate(itertools.chain(*representations)):
            yield _WriterContext(
                parameter_name=_IndexedObject(
                    parameter_name, parameter_index
                ),
                representation=_IndexedObject(
                    representation, representation_index
                ),
                context=context
            )


def write_specs_to_str(
    element: Specsable,
    element_index: int,
    directory: str | Path,
    stream: TextIO,
):
    writer_context_generator = _context_generator(
        element,
        element_index,
        directory,
    )

    # create and write header for the element
    element_name = element.__class__.__name__
    indexed_name = f"({element_index}) {element_name}"

    if element_index == 0:
        element_header = ''
    else:
        element_header = '\n'

    element_header += f'{indexed_name}\n'
    stream.write(element_header)

    # loop over representations
    for writer_context in writer_context_generator:

        # create header for parameter specs
        if writer_context.parameter_name.index == 0:
            specs_header = ''
        else:
            specs_header = '\n'
        specs_header += f'    {writer_context.parameter_name.value}\n'

        # write header for parameter only in the beginning of representations
        if writer_context.representation.index == 0:
            stream.write(specs_header)

        representation = writer_context.representation.value

        if isinstance(representation, StrRepresentation):
            # write separator between two representations
            if writer_context.representation.index != 0:
                stream.write('\n\n')

            _stream = StringIO('')

            representation.to_str(
                out=_stream,
                context=writer_context.context
            )

            s = _stream.getvalue()
            # add spaces at the beginning of each line
            new_line_prefix = ' ' * 8
            stream.write(
                new_line_prefix + new_line_prefix.join(
                    s.splitlines(keepends=True)
                )
            )


def write_specs_to_markdown(
    element: Specsable,
    element_index: int,
    directory: str | Path,
    stream: TextIO,
):
    writer_context_generator = _context_generator(
        element,
        element_index,
        directory,
    )

    # create and write header for the element
    element_name = element.__class__.__name__
    indexed_name = f"({element_index}) {element_name}"

    element_header = '' if element_index == 0 else '\n'
    element_header += f"# {indexed_name}\n"

    stream.write(element_header)

    for writer_context in writer_context_generator:

        # create header for parameter specs
        if writer_context.parameter_name.index == 0:
            specs_header = ''
        else:
            specs_header = '\n'
        specs_header += f'**{writer_context.parameter_name.value}**\n'

        # write header for parameter only in the beginning of representations
        if writer_context.representation.index == 0:
            stream.write(specs_header)

        representation = writer_context.representation.value

        if isinstance(representation, MarkdownRepresentation):
            # write separator between two representations
            if writer_context.representation.index != 0:
                stream.write('\n\n')

            representation.to_markdown(
                out=stream,
                context=writer_context.context
            )


def write_specs_to_html(
    element: Specsable,
    element_index: int,
    directory: str | Path,
    stream: TextIO,
):

    writer_context_generator = _context_generator(
        element,
        element_index,
        directory,
    )

    element_name = element.__class__.__name__
    indexed_name = f"({element_index}) {element_name}"

    s = f"""
    {_HTML_STYLE}
    <div class="svetlanna-specs-container">
    <details>
    <summary>{indexed_name}</summary>
    <div class="svetlanna-specs-details">
    <div>
    """

    for writer_context in writer_context_generator:

        specs_header = f"""
        <div class="svetlanna-specs-parameter-name">
            <b>{writer_context.parameter_name.value}</b>
        </div>
        """

        # write header for parameter only in the beginning of representations
        if writer_context.representation.index == 0:
            s += specs_header

        representation = writer_context.representation.value

        if isinstance(representation, StrRepresentation):
            _stream = StringIO('')

            representation.to_str(
                out=_stream,
                context=writer_context.context
            )

            s += f"""
            <div class="svetlanna-specs-representation">
<pre>{_stream.getvalue()}</pre>
            </div>
            """

    s += '</div></div></details></div>'
    stream.write(s)


def write_specs(
    *iterables: Specsable,
    filename: str = 'specs.txt',
    directory: str | Path = '',
):
    Path.mkdir(Path(directory), parents=True, exist_ok=True)
    path = Path(directory, filename)

    with open(path, 'w') as file:
        if filename.endswith('.txt'):
            for elemennt_index, element in enumerate(iterables):
                write_specs_to_str(
                    element=element,
                    element_index=elemennt_index,
                    directory=directory,
                    stream=file
                )
        elif filename.endswith('.md'):
            for elemennt_index, element in enumerate(iterables):
                write_specs_to_markdown(
                    element=element,
                    element_index=elemennt_index,
                    directory=directory,
                    stream=file
                )
        else:
            raise ValueError(
                "Unknown file extension. Filename should end with '.md' or '.txt'."
            )
