from typing import Iterable, TextIO, TypeVar, Generic, Generator, TypeAlias
from .specs import SubelementSpecs, Specsable
from .specs import ParameterSaveContext, Representation
from .specs import StrRepresentation, MarkdownRepresentation
from .specs import HTMLRepresentation
from pathlib import Path
import itertools
from dataclasses import dataclass, field
from io import StringIO


_T = TypeVar('_T')


@dataclass(frozen=True, slots=True)
class _IndexedObject(Generic[_T]):
    value: _T
    index: int


@dataclass
class _WriterContext:
    """Storage for additional info within ParameterSaveContext"""
    parameter_name: _IndexedObject[str]
    representation: _IndexedObject[Representation]
    context: ParameterSaveContext


_WriterContextGenerator: TypeAlias = Generator[_WriterContext, None, None]


def context_generator(
    element: Specsable,
    element_index: int,
    directory: str | Path,
    subelements: list[SubelementSpecs]
) -> _WriterContextGenerator:
    """Generate _WriterContext for the element

    Parameters
    ----------
    element : Specsable
        Element
    element_index : int
        Index of the element. It is used to create
        unique directory for element specs.
    directory : str | Path
        Directory where element directory is created

    Yields
    ------
    _WriterContext
        context
    """
    specs_directory = Path(
        directory, f'{element_index}_{element.__class__.__name__}'
    )

    # sort all iterators based on parameter name
    repr_iterators: dict[str, list[Iterable[Representation]]] = {}

    for spec in element.to_specs():

        if isinstance(spec, SubelementSpecs):
            subelements.append(spec)
            continue

        parameter_name = spec.parameter_name
        representations = spec.representations

        if parameter_name in repr_iterators:
            repr_iterators[parameter_name].append(representations)
        else:
            repr_iterators[parameter_name] = [representations]

    # create representations iterator for each parameter
    parameter_representations = {
        name: itertools.chain(*iters) for name, iters in repr_iterators.items()
    }

    for parameter_index, (parameter_name, representations) in enumerate(parameter_representations.items()):

        # create context for parameter
        context = ParameterSaveContext(
            parameter_name=parameter_name,
            directory=specs_directory
        )

        for representation_index, representation in enumerate(representations):
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
    writer_context_generator: _WriterContextGenerator,
    stream: TextIO,
):

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
                stream.write('\n')

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
    writer_context_generator: _WriterContextGenerator,
    stream: TextIO,
):

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
                stream.write('\n')

            representation.to_markdown(
                out=stream,
                context=writer_context.context
            )


def write_specs_to_html(
    element: Specsable,
    element_index: int,
    writer_context_generator: _WriterContextGenerator,
    stream: TextIO,
):

    s = '<div style="font-family:monospace;">'

    for writer_context in writer_context_generator:

        specs_header = f"""
        <div style="margin-top:0.5rem;">
            <b>{writer_context.parameter_name.value}</b>
        </div>
        """

        # write header for parameter only in the beginning of representations
        if writer_context.representation.index == 0:
            s += specs_header

        representation = writer_context.representation.value

        if isinstance(representation, HTMLRepresentation):
            _stream = StringIO('')

            representation.to_html(
                out=_stream,
                context=writer_context.context
            )

            s += f"""
            <div style="margin-bottom: 0.5rem;padding-left: 2rem;">
<pre style="white-space:pre-wrap;">{_stream.getvalue()}</pre>
            </div>
            """
    s += "</div>"
    stream.write(s)


@dataclass(frozen=True, slots=True)
class _ElementInTree:
    element: Specsable
    element_index: int
    children: list['_ElementInTree'] = field(default_factory=list)
    subelement_type: str | None = None

    def create_copy(
        self, subelement_type: str | None
    ) -> '_ElementInTree':
        return _ElementInTree(
            element=self.element,
            element_index=self.element_index,
            children=self.children,
            subelement_type=subelement_type
        )


class _ElementsIterator:
    def __init__(self, *iterables: Specsable, directory: str | Path) -> None:
        self.iterables = tuple(iterables)
        self.directory = directory
        self._iterated: dict[int, _ElementInTree] = {}
        self._tree: list[_ElementInTree] | None = None

    def __iter__(
        self
    ) -> Generator[tuple[int, Specsable, _WriterContextGenerator], None, None]:

        def f(
            specsables: Iterable[Specsable | SubelementSpecs],
            parent_children: list[_ElementInTree]
        ):

            for element in specsables:

                if isinstance(element, SubelementSpecs):
                    element_name = element.subelement_type
                    element = element.subelement
                else:
                    element_name = None

                # The element specs should be written only once
                if element_in_tree := self._iterated.get(id(element)):
                    # The copy of the element in the tree is created
                    new_element_in_tree = element_in_tree.create_copy(
                        subelement_type=element_name
                    )
                    parent_children.append(new_element_in_tree)
                    continue

                subelements: list[SubelementSpecs] = []
                index = len(self._iterated)
                writer_context_generator = context_generator(
                    element, index, self.directory, subelements
                )  # Subelements list is appended inside the generator

                yield index, element, writer_context_generator

                # Create a new tree element
                element_in_tree = _ElementInTree(
                    element,
                    index,
                    subelement_type=element_name
                )
                self._iterated[id(element)] = element_in_tree
                parent_children.append(element_in_tree)

                # Repeat the process for all subelements of the tree element
                yield from f(subelements, element_in_tree.children)

        self._iterated = {}
        self._tree = []
        yield from f(self.iterables, self._tree)

    @property
    def tree(self) -> list[_ElementInTree]:
        """Get a tree of all elements iterated

        Returns
        -------
        list[_ElementInTree]
            Elements tree.
        """
        if self._tree is None:
            # Iterate to build a tree if not already exists
            for _, _, i in self:
                for _ in i:
                    pass
            return self.tree
        return self._tree


def write_elements_tree_to_str(
    tree: list[_ElementInTree],
    stream: TextIO,
):
    stream.write('\n\nTree:\n')

    def _write_element(tree_level: int, element: _ElementInTree):
        stream.write(' ' * (8 * tree_level))
        element_name = element.element.__class__.__name__
        indexed_name = f'({element.element_index}) {element_name}'

        if element.subelement_type is not None:
            stream.write(f'[{element.subelement_type}] ')
        stream.write(f'{indexed_name}\n')

        for subelement in element.children:
            _write_element(tree_level + 1, subelement)

    for element in tree:
        _write_element(0, element)


def write_elements_tree_to_markdown(
    tree: list[_ElementInTree],
    stream: TextIO,
):
    stream.write('\n\n# Tree:\n')

    def _write_element(tree_level: int, element: _ElementInTree):
        stream.write(' ' * (4 * tree_level) + '* ')
        element_name = element.element.__class__.__name__
        indexed_name = f'`({element.element_index}) {element_name}`'

        if element.subelement_type is not None:
            stream.write(f'[{element.subelement_type}] ')
        stream.write(f'{indexed_name}\n')

        for subelement in element.children:
            _write_element(tree_level + 1, subelement)

    for element in tree:
        _write_element(0, element)


def write_specs(
    *iterables: Specsable,
    filename: str = 'specs.txt',
    directory: str | Path = 'specs',
):
    Path.mkdir(Path(directory), parents=True, exist_ok=True)
    path = Path(directory, filename)

    elements = _ElementsIterator(*iterables, directory=directory)

    with open(path, 'w') as file:
        if filename.endswith('.txt'):
            for elemennt_index, element, writer_context_generator in elements:
                write_specs_to_str(
                    element=element,
                    element_index=elemennt_index,
                    writer_context_generator=writer_context_generator,
                    stream=file
                )
            write_elements_tree_to_str(
                tree=elements.tree,
                stream=file
            )
        elif filename.endswith('.md'):
            for elemennt_index, element, writer_context_generator in elements:
                write_specs_to_markdown(
                    element=element,
                    element_index=elemennt_index,
                    writer_context_generator=writer_context_generator,
                    stream=file
                )
            write_elements_tree_to_markdown(
                tree=elements.tree,
                stream=file
            )
        else:
            raise ValueError(
                "Unknown file extension. ' \
                'Filename should end with '.md' or '.txt'."
            )

    return elements
