from svetlanna.specs import ParameterSaveContext, ParameterSpecs
from svetlanna.specs import SubelementSpecs
from svetlanna.specs import ImageRepr, ReprRepr, NpyFileRepr, PrettyReprRepr
from svetlanna import ConstrainedParameter
from pathlib import Path
from io import StringIO
import random
import numpy as np
import pytest
from PIL import Image
import torch
import builtins

###############################################################################
#                       ParameterSaveContext tests                            #
###############################################################################


def test_save_context_get_new_filepath(tmp_path):
    context = ParameterSaveContext(
        parameter_name='test',
        directory=tmp_path,
    )

    # test filename
    path = context.get_new_filepath("testext")
    assert Path(tmp_path, 'test_0.testext') == path

    path = context.get_new_filepath("testext")
    assert Path(tmp_path, 'test_1.testext') == path


def test_save_context_file(tmp_path):
    context = ParameterSaveContext(
        parameter_name='test',
        directory=tmp_path,
    )

    # create a new file and write test text
    text = str(random.random())
    path = context.get_new_filepath("testext")
    with context.file(path) as file:
        file.write(text.encode())

    # check if the test text is written into the file
    with open(path, 'rb') as file:
        assert file.readline() == text.encode()

    # check if the new file will have another name, but same folder
    new_path = context.get_new_filepath("testext")
    assert new_path != path
    assert new_path.parent == path.parent


def test_save_context_rel_filepath(tmp_path):
    contexts = ParameterSaveContext(
        parameter_name='test',
        directory=tmp_path,
    )

    path = contexts.get_new_filepath("testext")

    assert tmp_path == path.parent
    assert Path(path.parent.name, path.name) == contexts.rel_filepath(path)


###############################################################################
#                             ImageRepr tests                                 #
###############################################################################


@pytest.mark.usefixtures('tmp_path')
@pytest.mark.parametrize(
    'mode', ('1', 'L', 'LA', 'I', 'P', 'RGB', 'RGBA')
)
def test_image_repr_draw_image(tmp_path, mode):
    context = ParameterSaveContext(
        parameter_name='test',
        directory=tmp_path,
    )

    # TODO: mode-based test
    image_to_draw = np.array([[1]])

    repr = ImageRepr(
        value=image_to_draw,
        mode=mode
    )

    # draw image to the path
    path = context.get_new_filepath("png")
    image = repr.draw_image(context, path)

    # open the image and compare it with the drawn one
    assert np.all(np.array(Image.open(path)) == np.array(image))


def test_image_repr_to(tmp_path):
    context = ParameterSaveContext(
        parameter_name='test',
        directory=tmp_path,
    )

    repr = ImageRepr(
        value=np.array([[0.5]])
    )

    # test for all possible exports
    test_out = StringIO()
    repr.to_str(test_out, context)
    assert test_out.getvalue()

    test_out = StringIO()
    repr.to_markdown(test_out, context)
    assert test_out.getvalue()

    test_out = StringIO()
    repr.to_html(test_out, context)
    assert test_out.getvalue()


###############################################################################
#                              ReprRepr tests                                 #
###############################################################################


def test_repr_repr_to(tmp_path):
    context = ParameterSaveContext(
        parameter_name='test',
        directory=tmp_path,
    )

    repr = ReprRepr(
        value=np.array([[0.5]])
    )

    # test for all possible exports
    test_out = StringIO()
    repr.to_str(test_out, context)
    assert test_out.getvalue()

    test_out = StringIO()
    repr.to_markdown(test_out, context)
    assert test_out.getvalue()

    test_out = StringIO()
    repr.to_html(test_out, context)
    assert test_out.getvalue()


###############################################################################
#                          PrettyReprRepr tests                               #
###############################################################################


@pytest.mark.usefixtures('tmp_path')
@pytest.mark.parametrize(
    'value', (
        np.random.rand(2, 2),
        torch.rand(2, 2),
        torch.tensor(random.random()),
        random.random(),
        random.randint(0, 10),
        ConstrainedParameter(10, 0, 20)
    )
)
def test_pretty_repr_repr_to(tmp_path, value, monkeypatch):
    context = ParameterSaveContext(
        parameter_name='test',
        directory=tmp_path,
    )

    repr = PrettyReprRepr(
        value=value
    )

    # test for all possible exports
    test_out = StringIO()
    repr.to_str(test_out, context)
    assert test_out.getvalue()

    test_out = StringIO()
    repr.to_markdown(test_out, context)
    assert test_out.getvalue()

    test_out = StringIO()
    repr.to_html(test_out, context)
    assert test_out.getvalue()

    # Test processing of svetlanna absence
    if isinstance(value, torch.Tensor) and len(value.shape) == 0:

        # monkeypatching import statement
        original_import = builtins.__import__

        def import_with_no_svetlanna(name, *args, **kwargs):
            if name == "svetlanna":
                raise ImportError
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', import_with_no_svetlanna)

        # Test if default string is written to the buffer
        test_out = StringIO()
        repr.to_str(test_out, context)
        class_name = value.__class__.__name__
        assert test_out.getvalue() == f'{class_name}\n{value.item()}\n'


###############################################################################
#                            NpyFileRepr tests                                #
###############################################################################


@pytest.mark.usefixtures('tmp_path')
@pytest.mark.parametrize(
    'value', (
        np.random.rand(10, 10),
        random.random(),
        random.randint(0, 10),
    )
)
def test_npy_file_repr_save_to_file(tmp_path, value):
    context = ParameterSaveContext(
        parameter_name='test',
        directory=tmp_path,
    )

    repr = NpyFileRepr(
        value=value
    )

    # save the value to a new file
    path = context.get_new_filepath("png")
    repr.save_to_file(context, path)

    # read a value from the file and compare it with the saved one
    assert np.all(np.load(path) == value)


def test_npy_file_repr_to(tmp_path):
    context = ParameterSaveContext(
        parameter_name='test',
        directory=tmp_path,
    )

    repr = NpyFileRepr(
        value=np.array([[0.5]])
    )

    # test for all possible exports
    test_out = StringIO()
    repr.to_str(test_out, context)
    assert test_out.getvalue()

    test_out = StringIO()
    repr.to_markdown(test_out, context)
    assert test_out.getvalue()


###############################################################################
#                           ParameterSpecs tests                              #
###############################################################################


def test_parameter_specs():
    representations = (
        ReprRepr(123),
        ReprRepr(321),
    )

    specs = ParameterSpecs(
        parameter_name='test',
        representations=representations
    )

    assert specs.representations == representations


###############################################################################
#                           SubelementSpecs tests                             #
###############################################################################


def test_subelement_specs():
    specs = [
        ParameterSpecs('test', [])
    ]

    class Subelement:
        def to_specs(self):
            return specs

    subelement = Subelement()
    subelement_specs = SubelementSpecs(
        'test_type',
        subelement
    )

    assert subelement_specs.subelement is subelement
