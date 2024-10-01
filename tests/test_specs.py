from svetlanna.specs import SaveContexts
from pathlib import Path
import random


def test_save_context_get_new_filepath(tmpdir):
    contexts = SaveContexts(
        parameter_name='test',
        directory=tmpdir
    )

    # test filename
    path = contexts.get_new_filepath("testext")
    assert Path(tmpdir, 'test_0.testext') == path

    contexts._generated_files.append(path)
    path = contexts.get_new_filepath("testext")
    assert Path(tmpdir, 'test_1.testext') == path


def test_save_context_file(tmpdir):
    contexts = SaveContexts(
        parameter_name='test',
        directory=tmpdir
    )

    # create a new file and write test text
    text = str(random.random())
    path = contexts.get_new_filepath("testext")
    with contexts.file("testext") as file:
        file.write(text.encode())

    # check if the test text is written into the file
    with open(path, 'rb') as file:
        assert file.readline() == text.encode()

    # check if the new file will have another name, but same folder
    new_path = contexts.get_new_filepath("testext")
    assert new_path != path
    assert new_path.parent == path.parent
