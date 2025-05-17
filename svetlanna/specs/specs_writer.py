from typing import Iterable, TextIO, TypeVar, Generic, Generator, TypeAlias
from .specs import SubelementSpecs, Specsable
from .specs import ParameterSaveContext, Representation
from .specs import StrRepresentation, MarkdownRepresentation
from .specs import HTMLRepresentation
from pathlib import Path
import itertools
from dataclasses import dataclass, field
from io import StringIO


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _IndexedObject(Generic[_T]):
    """
    Represents an object with a value and its corresponding index.

        This class is designed to hold data paired with an index, useful for
        situations where tracking the original position of an item is important,
        such as during sorting or reordering operations.

        Attributes:
            value: The data associated with this object.
            index: The original index or position of the value.
    """

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
    subelements: list[SubelementSpecs],
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
    specs_directory = Path(directory, f"{element_index}_{element.__class__.__name__}")

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

    for parameter_index, (parameter_name, representations) in enumerate(
        parameter_representations.items()
    ):

        # create context for parameter
        context = ParameterSaveContext(
            parameter_name=parameter_name, directory=specs_directory
        )

        for representation_index, representation in enumerate(representations):
            yield _WriterContext(
                parameter_name=_IndexedObject(parameter_name, parameter_index),
                representation=_IndexedObject(representation, representation_index),
                context=context,
            )


def write_specs_to_str(
    element: Specsable,
    element_index: int,
    writer_context_generator: _WriterContextGenerator,
    stream: TextIO,
):
    """
    Writes the specifications of an element to a stream.

        Args:
            element: The element whose specifications are to be written.
            element_index: The index of the element in a list of elements.
            writer_context_generator: A generator that yields writer contexts for different representations.
            stream: The output stream to write the specifications to.

        Returns:
            None
    """

    # create and write header for the element
    element_name = element.__class__.__name__
    indexed_name = f"({element_index}) {element_name}"

    if element_index == 0:
        element_header = ""
    else:
        element_header = "\n"

    element_header += f"{indexed_name}\n"
    stream.write(element_header)

    # loop over representations
    for writer_context in writer_context_generator:

        # create header for parameter specs
        if writer_context.parameter_name.index == 0:
            specs_header = ""
        else:
            specs_header = "\n"
        specs_header += f"    {writer_context.parameter_name.value}\n"

        # write header for parameter only in the beginning of representations
        if writer_context.representation.index == 0:
            stream.write(specs_header)

        representation = writer_context.representation.value

        if isinstance(representation, StrRepresentation):
            # write separator between two representations
            if writer_context.representation.index != 0:
                stream.write("\n")

            _stream = StringIO("")

            representation.to_str(out=_stream, context=writer_context.context)

            s = _stream.getvalue()
            # add spaces at the beginning of each line
            new_line_prefix = " " * 8
            stream.write(
                new_line_prefix + new_line_prefix.join(s.splitlines(keepends=True))
            )


def write_specs_to_markdown(
    element: Specsable,
    element_index: int,
    writer_context_generator: _WriterContextGenerator,
    stream: TextIO,
):
    """
    Writes specifications for an element to a markdown stream.

        Args:
            element: The element whose specs are being written.
            element_index: The index of the element in a list of elements.
            writer_context_generator: A generator that yields writer contexts
                containing parameter and representation information.
            stream: The output stream to write markdown to.

        Returns:
            None
    """

    # create and write header for the element
    element_name = element.__class__.__name__
    indexed_name = f"({element_index}) {element_name}"

    element_header = "" if element_index == 0 else "\n"
    element_header += f"# {indexed_name}\n"

    stream.write(element_header)

    for writer_context in writer_context_generator:

        # create header for parameter specs
        if writer_context.parameter_name.index == 0:
            specs_header = ""
        else:
            specs_header = "\n"
        specs_header += f"**{writer_context.parameter_name.value}**\n"

        # write header for parameter only in the beginning of representations
        if writer_context.representation.index == 0:
            stream.write(specs_header)

        representation = writer_context.representation.value

        if isinstance(representation, MarkdownRepresentation):
            # write separator between two representations
            if writer_context.representation.index != 0:
                stream.write("\n")

            representation.to_markdown(out=stream, context=writer_context.context)


def write_specs_to_html(
    element: Specsable,
    element_index: int,
    writer_context_generator: _WriterContextGenerator,
    stream: TextIO,
):
    """
    Writes specifications to an HTML stream.

        Iterates through writer contexts and writes HTML representations of specs,
        including parameter headers where appropriate.

        Args:
            element: The element whose specs are being written.
            element_index: The index of the element.
            writer_context_generator: A generator yielding writer contexts for each representation.
            stream: The output stream to write the HTML to.

        Returns:
            None
    """

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
            _stream = StringIO("")

            representation.to_html(out=_stream, context=writer_context.context)

            s += f"""
            <div style="margin-bottom: 0.5rem;padding-left: 2rem;">
<pre style="white-space:pre-wrap;">{_stream.getvalue()}</pre>
            </div>
            """
    s += "</div>"
    stream.write(s)


@dataclass(frozen=True, slots=True)
class _ElementInTree:
    """
    Represents an element within a tree structure.

        This class holds information about an element, its index, children, and
        potential subelement type. It provides functionality to create copies of the element.

        Attributes:
            element: The actual element being represented.
            element_index: The index of the element in its parent's list of children.
            children: A list of child _ElementInTree objects.
            subelement_type:  The type of subelement, if applicable.

        Methods:
            create_copy(subelement_type): Creates a copy of the current element.
    """

    element: Specsable
    element_index: int
    children: list["_ElementInTree"] = field(default_factory=list)
    subelement_type: str | None = None

    def create_copy(self, subelement_type: str | None) -> "_ElementInTree":
        return _ElementInTree(
            element=self.element,
            element_index=self.element_index,
            children=self.children,
            subelement_type=subelement_type,
        )


class _ElementsIterator:
    """
    Iterates through a collection of iterables, yielding elements with their index and writer context.

    This class provides an iterator that handles multiple iterables, subelements, and ensures
    each element is written only once while building a tree structure of the iterated elements.
    """

    def __init__(self, *iterables: Specsable, directory: str | Path) -> None:
        """
        Initializes a new instance of the class.

            Args:
                *iterables: The iterables to process. Can be multiple.
                directory: The directory associated with the iterables.

            Returns:
                None
        """
        self.iterables = tuple(iterables)
        self.directory = directory
        self._iterated: dict[int, _ElementInTree] = {}
        self._tree: list[_ElementInTree] | None = None

    def __iter__(
        self,
    ) -> Generator[tuple[int, Specsable, _WriterContextGenerator], None, None]:
        """
        Iterates through the specsables and yields them with their index and writer context.

            This method traverses a collection of specsables, handling subelements and ensuring
            each element is written only once. It maintains an internal state to track iterated
            elements and builds a tree structure as it iterates.

            Args:
                None

            Returns:
                Generator[tuple[int, Specsable, _WriterContextGenerator]]: A generator that yields tuples
                containing the index of the element, the element itself (Specsable or SubelementSpecs),
                and a writer context generator for the element.
        """

        def f(
            specsables: Iterable[Specsable | SubelementSpecs],
            parent_children: list[_ElementInTree],
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
                    element, index, subelement_type=element_name
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
    """
    Writes the elements tree to a string stream.

        Args:
            tree: The list of root elements in the tree.
            stream: The output stream to write to.

        Returns:
            None
    """
    stream.write("\n\nTree:\n")

    def _write_element(tree_level: int, element: _ElementInTree):
        stream.write(" " * (8 * tree_level))
        element_name = element.element.__class__.__name__
        indexed_name = f"({element.element_index}) {element_name}"

        if element.subelement_type is not None:
            stream.write(f"[{element.subelement_type}] ")
        stream.write(f"{indexed_name}\n")

        for subelement in element.children:
            _write_element(tree_level + 1, subelement)

    for element in tree:
        _write_element(0, element)


def write_elements_tree_to_markdown(
    tree: list[_ElementInTree],
    stream: TextIO,
):
    """
    Writes the elements tree to a markdown formatted string.

        Args:
            tree: The list of root elements in the tree.
            stream: A file-like object to write the markdown output to.

        Returns:
            None
    """
    stream.write("\n\n# Tree:\n")

    def _write_element(tree_level: int, element: _ElementInTree):
        stream.write(" " * (4 * tree_level) + "* ")
        element_name = element.element.__class__.__name__
        indexed_name = f"`({element.element_index}) {element_name}`"

        if element.subelement_type is not None:
            stream.write(f"[{element.subelement_type}] ")
        stream.write(f"{indexed_name}\n")

        for subelement in element.children:
            _write_element(tree_level + 1, subelement)

    for element in tree:
        _write_element(0, element)


def write_specs(
    *iterables: Specsable,
    filename: str = "specs.txt",
    directory: str | Path = "specs",
):
    """
    Writes specifications from iterables to a file.

        Creates a directory if it doesn't exist and writes the specs to either a
        text or markdown file based on the filename extension.

        Args:
            *iterables: One or more iterable objects containing specification data.
            filename: The name of the output file (e.g., 'specs.txt' or 'specs.md').
            directory: The directory to write the file to. Defaults to 'specs'.

        Returns:
            _ElementsIterator: An iterator object representing the written elements and their tree structure.
    """
    Path.mkdir(Path(directory), parents=True, exist_ok=True)
    path = Path(directory, filename)

    elements = _ElementsIterator(*iterables, directory=directory)

    with open(path, "w") as file:
        if filename.endswith(".txt"):
            for elemennt_index, element, writer_context_generator in elements:
                write_specs_to_str(
                    element=element,
                    element_index=elemennt_index,
                    writer_context_generator=writer_context_generator,
                    stream=file,
                )
            write_elements_tree_to_str(tree=elements.tree, stream=file)
        elif filename.endswith(".md"):
            for elemennt_index, element, writer_context_generator in elements:
                write_specs_to_markdown(
                    element=element,
                    element_index=elemennt_index,
                    writer_context_generator=writer_context_generator,
                    stream=file,
                )
            write_elements_tree_to_markdown(tree=elements.tree, stream=file)
        else:
            raise ValueError(
                "Unknown file extension. ' \
                'Filename should end with '.md' or '.txt'."
            )

    return elements
