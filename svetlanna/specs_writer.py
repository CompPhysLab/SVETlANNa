from typing import Iterable, TextIO
from .specs import ParameterSpecs
from .specs import ParameterSaveContext, Representation
from .specs import StrRepresentation, MarkdownRepresentation
from pathlib import Path
import itertools
from dataclasses import dataclass
from typing import Protocol


class Specsable(Protocol):
    def to_specs(self) -> Iterable[ParameterSpecs]:
        ...


@dataclass
class _WriterContext:
    element: Specsable
    element_index: int
    parameter_name: str
    parameter_index: int
    representation: Representation
    representation_index: int
    stream: TextIO
    context: ParameterSaveContext


def _context_generator(
    elements: Iterable[Specsable],
    directory: str | Path,
    stream: TextIO,
):
    for element_index, element in enumerate(elements):
        specs_directory = Path(
            directory, f'{element_index}_{element.__class__.__name__}'
        )
        Path.mkdir(specs_directory, parents=True, exist_ok=True)

        names_representations = {}

        for spec in element.to_specs():
            parameter_name = spec.parameter_name
            representations = spec.representations

            if parameter_name in names_representations:
                names_representations[parameter_name].append(representations)
            else:
                names_representations[parameter_name] = [representations]

        for parameter_index, (parameter_name, representations) in enumerate(names_representations.items()):

            context = ParameterSaveContext(
                parameter_name=parameter_name,
                directory=specs_directory,
                stream=stream
            )

            for representation_index, representation in enumerate(itertools.chain(*representations)):
                yield _WriterContext(
                    element=element,
                    element_index=element_index,
                    parameter_name=parameter_name,
                    parameter_index=parameter_index,
                    representation=representation,
                    representation_index=representation_index,
                    stream=stream,
                    context=context
                )


def write_specs_to_str(
    elements: Iterable[Specsable],
    directory: str | Path,
    stream: TextIO,
    line_width: int = 80
):
    element_index = -1
    parameter_index = -1

    for writer_context in _context_generator(elements, directory, stream):

        element_name = writer_context.element.__class__.__name__
        indexed_name = f"({writer_context.element_index}) {element_name}"

        if writer_context.element_index == 0:
            element_header = ''
        else:
            element_header = '\n'

        element_header += f'┏{"━" * (line_width-2)}┓\n'
        element_header += f'┃{" " * (line_width-2)}┃\n'
        element_header += f'┃{indexed_name:^{(line_width-2)}s}┃\n'
        element_header += f'┃{" " * (line_width-2)}┃\n'
        element_header += f'┗{"━" * (line_width-2)}┛\n'

        # write header only in the beginning of the element
        if element_index != writer_context.element_index:
            stream.write(element_header)
            element_index = writer_context.element_index

        if writer_context.parameter_index == 0:
            specs_header = ''
        else:
            specs_header = '\n'

        parameter_name = writer_context.parameter_name
        specs_header += f'┌{"─" * (line_width//2-2)}┐\n'
        specs_header += f'│{parameter_name:^{(line_width//2-2)}s}│\n'
        specs_header += f'└{"─" * (line_width//2-2)}┘\n'

        # write header for parameter only in the beginning of the element
        if parameter_index != writer_context.parameter_index:
            stream.write(specs_header)
            parameter_index = writer_context.parameter_index

        if isinstance(writer_context.representation, StrRepresentation):
            if writer_context.representation_index != 0:
                stream.write(f'{"┄" * (line_width//2)}\n')
            writer_context.representation.to_str(
                context=writer_context.context
            )


def write_specs_to_markdown(
    elements: Iterable[Specsable],
    directory: str | Path,
    stream: TextIO,
):
    element_index = -1
    parameter_index = -1

    for writer_context in _context_generator(elements, directory, stream):

        element_name = writer_context.element.__class__.__name__
        indexed_name = f"({writer_context.element_index}) {element_name}"
        element_header = f"# {indexed_name}\n"

        # write header only in the beginning of the element
        if element_index != writer_context.element_index:
            stream.write(element_header)
            element_index = writer_context.element_index

        parameter_name = writer_context.parameter_name
        specs_header = f'**{parameter_name}**\n\n'

        # write header for parameter only in the beginning of the element
        if parameter_index != writer_context.parameter_index:
            stream.write(specs_header)
            parameter_index = writer_context.parameter_index

        if isinstance(writer_context.representation, MarkdownRepresentation):

            if writer_context.representation_index != 0:
                stream.write('\n\n')

            writer_context.representation.to_markdown(
                context=writer_context.context
            )


def write_specs(
    *iterables: Specsable,
    filename: str = 'specs.txt',
    directory: str | Path = '',
):
    Path.mkdir(Path(directory), parents=True, exist_ok=True)
    path = Path(directory, filename)

    with open(path, 'w') as file:
        if filename.endswith('.txt'):
            write_specs_to_str(
                elements=iterables,
                directory=directory,
                stream=file
            )
        elif filename.endswith('.md'):
            write_specs_to_markdown(
                elements=iterables,
                directory=directory,
                stream=file
            )
        else:
            raise ValueError(
                "Unknown file extension. Filename should end with '.md' or '.txt'."
            )
