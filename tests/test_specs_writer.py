from typing import Iterable
import torch
from svetlanna.elements import Element
from svetlanna import SimulationParameters
from svetlanna.specs import ParameterSpecs, ReprRepr
from svetlanna.specs.specs_writer import _context_generator
from svetlanna.specs.specs_writer import write_specs_to_str
from svetlanna.specs.specs_writer import write_specs_to_markdown
from svetlanna.specs.specs_writer import write_specs_to_html
from svetlanna.specs.specs_writer import write_specs
from svetlanna.wavefront import Wavefront
from io import StringIO
from pathlib import Path
import pytest


class SpecsTestElement(Element):

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        test_specs: Iterable[ParameterSpecs]
    ) -> None:
        super().__init__(simulation_parameters)
        self.test_specs = test_specs

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        return super().forward(incident_wavefront)

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return self.test_specs


def test_context_generator(tmp_path):
    simulation_parameters = SimulationParameters(
        axes={'W': torch.tensor([0]), 'H': torch.tensor([0]), 'wavelength': 1}
    )

    repr1 = ReprRepr(1.)
    repr2 = ReprRepr(2.)
    repr3 = ReprRepr(3.)
    repr4 = ReprRepr(4.)

    element = SpecsTestElement(
        simulation_parameters=simulation_parameters,
        test_specs=[
            ParameterSpecs(
                'test1',
                [
                    repr1,
                    repr2
                ]
            ),
            ParameterSpecs(
                'test2',
                [
                    repr3,
                ]
            ),
            ParameterSpecs(
                'test2',  # test for the parameter spec with the same name
                [
                    repr4,
                ]
            )
        ]
    )

    contexts = list(_context_generator(element, 0, tmp_path))

    # === test contexts ===
    # test parameter_name attribute
    assert contexts[0].parameter_name.value == 'test1'
    assert contexts[1].parameter_name.value == 'test1'
    assert contexts[2].parameter_name.value == 'test2'
    assert contexts[3].parameter_name.value == 'test2'

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
    write_specs_to_str(element, 0, tmp_path, test_stream)
    assert test_stream.getvalue()

    # test for another header
    test_stream = StringIO()
    write_specs_to_str(element, 1, tmp_path, test_stream)
    assert test_stream.getvalue()

    # === test to_markdown ===
    test_stream = StringIO()
    write_specs_to_markdown(element, 0, tmp_path, test_stream)
    assert test_stream.getvalue()

    # === test to_html ===
    test_stream = StringIO()
    write_specs_to_html(element, 0, tmp_path, test_stream)
    assert test_stream.getvalue()


def test_write_specs(tmp_path):
    simulation_parameters = SimulationParameters(
        axes={'W': torch.tensor([0]), 'H': torch.tensor([0]), 'wavelength': 1}
    )

    repr1 = ReprRepr(1.)

    element = SpecsTestElement(
        simulation_parameters=simulation_parameters,
        test_specs=[
            ParameterSpecs(
                'test1',
                [
                    repr1,
                ]
            )
        ]
    )

    # === test txt ===
    write_specs(element, filename='test_specs.txt', directory=tmp_path)
    assert Path.exists(tmp_path / 'test_specs.txt')

    # === test md ===
    write_specs(element, filename='test_specs.md', directory=tmp_path)
    assert Path.exists(tmp_path / 'test_specs.md')

    # === test unknown format ===
    with pytest.raises(ValueError):
        write_specs(element, filename='test_specs.test', directory=tmp_path)
