from typing import Iterable
import torch
from svetlanna.elements import Element
from svetlanna import SimulationParameters
from svetlanna.specs import ParameterSpecs, ReprRepr, SubelementSpecs
from svetlanna.specs.specs_writer import context_generator
from svetlanna.specs.specs_writer import write_specs_to_str
from svetlanna.specs.specs_writer import write_specs_to_markdown
from svetlanna.specs.specs_writer import write_specs_to_html
from svetlanna.specs.specs_writer import write_specs
from svetlanna.specs.specs_writer import _ElementInTree, _ElementsIterator
from svetlanna.specs.specs_writer import write_elements_tree_to_str
from svetlanna.specs.specs_writer import write_elements_tree_to_markdown
from svetlanna.wavefront import Wavefront
from io import StringIO
from pathlib import Path
import pytest


class SpecsTestElement(Element):
    """
    Tests a set of parameter specifications or subelement specifications."""

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        test_specs: Iterable[ParameterSpecs | SubelementSpecs],
    ) -> None:
        """
        Initializes a new instance of the class.

            Args:
                simulation_parameters: The simulation parameters to use.
                test_specs: An iterable of parameter specifications or subelement specifications
                    to be tested.

            Returns:
                None
        """
        super().__init__(simulation_parameters)
        self.test_specs = test_specs

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """
        Passes the wavefront to the next layer.

          This method simply calls the `forward` method of the parent class,
          effectively passing the incident wavefront along for further processing.

          Args:
            incident_wavefront: The input wavefront representing the current state
              of the wave propagation.

          Returns:
            Wavefront: The output wavefront after being processed by the next layer.
        """
        return super().forward(incident_wavefront)

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        """
        Returns the test specifications.

            Args:
                None

            Returns:
                Iterable[ParameterSpecs | SubelementSpecs]: An iterable of parameter or subelement specifications.
        """
        return self.test_specs


def test_context_generator(tmp_path):
    """
    Tests the context generator with a sample SpecsTestElement.

        This test verifies that the context generator produces contexts with the
        correct parameter names, representations, and indices. It also tests
        the output of writing specs to string, markdown, and HTML formats.

        Args:
            tmp_path: A temporary path for testing purposes.

        Returns:
            None
    """
    simulation_parameters = SimulationParameters(
        axes={"W": torch.tensor([0]), "H": torch.tensor([0]), "wavelength": 1}
    )

    repr1 = ReprRepr(1.0)
    repr2 = ReprRepr(2.0)
    repr3 = ReprRepr(3.0)
    repr4 = ReprRepr(4.0)
    subelement = SpecsTestElement(simulation_parameters, [])

    element = SpecsTestElement(
        simulation_parameters=simulation_parameters,
        test_specs=[
            ParameterSpecs("test1", [repr1, repr2]),
            ParameterSpecs(
                "test2",
                [
                    repr3,
                ],
            ),
            ParameterSpecs(
                "test2",  # test for the parameter spec with the same name
                [
                    repr4,
                ],
            ),
            SubelementSpecs("test_type", subelement),
        ],
    )

    subelements: list[SubelementSpecs] = []
    contexts = list(context_generator(element, 0, tmp_path, subelements))

    # === test contexts ===
    # test subelements
    assert len(subelements) == 1
    assert subelements[0].subelement is subelement

    # test parameter_name attribute
    assert contexts[0].parameter_name.value == "test1"
    assert contexts[1].parameter_name.value == "test1"
    assert contexts[2].parameter_name.value == "test2"
    assert contexts[3].parameter_name.value == "test2"

    assert contexts[0].parameter_name.index == 0
    assert contexts[1].parameter_name.index == 0
    assert contexts[2].parameter_name.index == 1
    assert contexts[3].parameter_name.index == 1

    # test representation attribute
    assert repr1 is contexts[0].representation.value
    assert repr2 is contexts[1].representation.value
    assert repr3 is contexts[2].representation.value
    assert repr4 is contexts[3].representation.value

    # === test to_str ===
    test_stream = StringIO()
    writer_context_generator = context_generator(element, 0, tmp_path, [])
    write_specs_to_str(element, 0, writer_context_generator, test_stream)
    assert test_stream.getvalue()

    # test for another header
    test_stream = StringIO()
    writer_context_generator = context_generator(element, 1, tmp_path, [])
    write_specs_to_str(element, 1, writer_context_generator, test_stream)
    assert test_stream.getvalue()

    # === test to_markdown ===
    test_stream = StringIO()
    writer_context_generator = context_generator(element, 0, tmp_path, [])
    write_specs_to_markdown(element, 0, writer_context_generator, test_stream)
    assert test_stream.getvalue()

    # === test to_html ===
    test_stream = StringIO()
    writer_context_generator = context_generator(element, 0, tmp_path, [])
    write_specs_to_html(element, 0, writer_context_generator, test_stream)
    assert test_stream.getvalue()


def test_ElementInTree():
    """
    Tests the creation and copying of an ElementInTree object.

        This test verifies that creating a copy of _ElementInTree correctly
        shares references to immutable attributes (element, element_index, children)
        while creating a new instance for mutable attributes (subelement_type).

        Parameters:
            None

        Returns:
            None
    """
    simulation_parameters = SimulationParameters(
        axes={"W": torch.tensor([0]), "H": torch.tensor([0]), "wavelength": 1}
    )
    element = SpecsTestElement(
        simulation_parameters=simulation_parameters, test_specs=[]
    )

    tree_element = _ElementInTree(element, 123, [], "test_1")
    tree_element_copy = tree_element.create_copy("test_2")

    assert tree_element_copy.element is tree_element.element
    assert tree_element_copy.element_index is tree_element.element_index
    assert tree_element_copy.children is tree_element.children
    assert tree_element_copy.subelement_type != tree_element.subelement_type
    assert tree_element_copy.subelement_type == "test_2"


def test_ElementsIterator():
    """
    Tests the functionality of the ElementsIterator class.

        This test case creates a complex element structure with nested subelements and
        verifies that the iterator correctly traverses this structure, yielding each
        element in the expected order. It also checks if the tree is saved and rebuilt
        correctly during multiple iterations.

        Args:
            None

        Returns:
            None
    """
    simulation_parameters = SimulationParameters(
        axes={"W": torch.tensor([0]), "H": torch.tensor([0]), "wavelength": 1}
    )

    repr1 = ReprRepr(1.0)
    subelement1 = SpecsTestElement(simulation_parameters, [])
    subelement2 = SpecsTestElement(simulation_parameters, [])
    subelement3 = SpecsTestElement(
        simulation_parameters,
        [
            SubelementSpecs("subelement1", subelement1),
            SubelementSpecs("subelement2", subelement2),
        ],
    )

    element = SpecsTestElement(
        simulation_parameters=simulation_parameters,
        test_specs=[
            ParameterSpecs(
                "test1",
                [
                    repr1,
                ],
            ),
            SubelementSpecs("subelement1_copy", subelement1),
            SubelementSpecs("subelement3", subelement3),
        ],
    )

    elements = _ElementsIterator(element, directory="")

    # Test iterator output
    iterated_indices = []
    iterated_elements = []
    for el_index, el, wcg in elements:
        # Iteration over _WriterContextGenerator is required!
        # Otherwise subelemnts are not iterated
        for _ in wcg:
            pass

        iterated_indices.append(el_index)
        iterated_elements.append(el)

    assert iterated_indices == list(range(4))
    assert iterated_elements == [element, subelement1, subelement3, subelement2]

    # Test if the tree is saved in the iterator
    tree = elements.tree
    assert tree is elements.tree

    # Test second run over the iterator
    second_iterated_indices = []
    second_iterated_elements = []
    for el_index, el, wcg in elements:
        # Iteration over _WriterContextGenerator is required!
        # Otherwise subelemnts are not iterated
        for _ in wcg:
            pass

        second_iterated_indices.append(el_index)
        second_iterated_elements.append(el)

    assert second_iterated_indices == iterated_indices
    assert second_iterated_elements == iterated_elements
    # Test if the tree has been rebuild
    assert elements.tree is not tree
    assert elements.tree == tree

    # Test if the tree can be generated automatically
    new_elements = _ElementsIterator(element, directory="")
    assert new_elements.tree is not tree
    assert new_elements.tree == tree
    assert new_elements.tree is new_elements.tree


def test_write_tree(tmp_path):
    """
    Tests writing the elements tree to both string and markdown formats.

        This test creates a nested structure of simulation elements and then
        verifies that `write_elements_tree_to_str` and `write_elements_tree_to_markdown`
        can successfully write this tree to a string stream and produce non-empty output,
        respectively.

        Args:
            tmp_path:  A temporary path (not directly used in the test but required by the fixture).

        Returns:
            None
    """
    simulation_parameters = SimulationParameters(
        axes={"W": torch.tensor([0]), "H": torch.tensor([0]), "wavelength": 1}
    )

    repr1 = ReprRepr(1.0)
    subelement1 = SpecsTestElement(simulation_parameters, [])
    subelement2 = SpecsTestElement(simulation_parameters, [])
    subelement3 = SpecsTestElement(
        simulation_parameters,
        [
            SubelementSpecs("subelement1", subelement1),
            SubelementSpecs("subelement2", subelement2),
        ],
    )

    element = SpecsTestElement(
        simulation_parameters=simulation_parameters,
        test_specs=[
            ParameterSpecs(
                "test1",
                [
                    repr1,
                ],
            ),
            SubelementSpecs("subelement1_copy", subelement1),
            SubelementSpecs("subelement3", subelement3),
        ],
    )

    elements = _ElementsIterator(element, directory="")

    # === test str ===
    stream = StringIO("")
    write_elements_tree_to_str(elements.tree, stream)
    assert stream.getvalue()

    # === test md ===
    stream = StringIO("")
    write_elements_tree_to_markdown(elements.tree, stream)
    assert stream.getvalue()


def test_write_specs(tmp_path):
    """
    Tests the write_specs function with different file formats."""
    simulation_parameters = SimulationParameters(
        axes={"W": torch.tensor([0]), "H": torch.tensor([0]), "wavelength": 1}
    )

    repr1 = ReprRepr(1.0)
    subelement1 = SpecsTestElement(simulation_parameters, [])
    subelement2 = SpecsTestElement(simulation_parameters, [])
    subelement3 = SpecsTestElement(
        simulation_parameters,
        [
            SubelementSpecs("subelement1", subelement1),
            SubelementSpecs("subelement2", subelement2),
        ],
    )

    element = SpecsTestElement(
        simulation_parameters=simulation_parameters,
        test_specs=[
            ParameterSpecs(
                "test1",
                [
                    repr1,
                ],
            ),
            SubelementSpecs("subelement1_copy", subelement1),
            SubelementSpecs("subelement3", subelement3),
        ],
    )

    # === test txt ===
    write_specs(element, filename="test_specs.txt", directory=tmp_path)
    assert Path.exists(tmp_path / "test_specs.txt")

    # === test md ===
    write_specs(element, filename="test_specs.md", directory=tmp_path)
    assert Path.exists(tmp_path / "test_specs.md")

    # === test unknown format ===
    with pytest.raises(ValueError):
        write_specs(element, filename="test_specs.test", directory=tmp_path)
