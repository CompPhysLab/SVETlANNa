from svetlanna.specs import ParameterSaveContext
from pathlib import Path
import sys
import random


def test_save_context_get_new_filepath(tmpdir):
    contexts = ParameterSaveContext(
        parameter_name='test',
        directory=tmpdir,
    )

    # test filename
    path = contexts.get_new_filepath("testext")
    assert Path(tmpdir, 'test_0.testext') == path

    path = contexts.get_new_filepath("testext")
    assert Path(tmpdir, 'test_1.testext') == path


def test_save_context_file(tmpdir):
    contexts = ParameterSaveContext(
        parameter_name='test',
        directory=tmpdir,
    )

    # create a new file and write test text
    text = str(random.random())
    path = contexts.get_new_filepath("testext")
    with contexts.file(path) as file:
        file.write(text.encode())

    # check if the test text is written into the file
    with open(path, 'rb') as file:
        assert file.readline() == text.encode()

    # check if the new file will have another name, but same folder
    new_path = contexts.get_new_filepath("testext")
    assert new_path != path
    assert new_path.parent == path.parent
