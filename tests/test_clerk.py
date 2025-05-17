from svetlanna import Clerk
from svetlanna.clerk import ClerkMode, CHECKPOINT_FILENAME_PATTERN
import pytest
import torch


def test_init(tmp_path):
    """
    Tests the Clerk initialization and experiment directory validation.

        Args:
            tmp_path: A temporary path to be used for testing.

        Returns:
            None
    """
    # Test the experiment directory
    clerk = Clerk(tmp_path)
    assert clerk.experiment_directory == tmp_path

    # Test if the experiment directory is not a directory case
    new_path = tmp_path / "test"
    assert not new_path.exists()
    with open(new_path, "w"):
        pass

    with pytest.raises(ValueError):
        clerk = Clerk(new_path)


def test_make_experiment_dir(tmp_path):
    """
    Tests the _make_experiment_dir method to ensure it creates a directory.

        Args:
            tmp_path: A temporary path for testing purposes.

        Returns:
            None
    """
    new_path = tmp_path / "test"
    clerk = Clerk(new_path)

    assert not new_path.exists()
    clerk._make_experiment_dir()
    assert new_path.exists()


def test_get_log_stream(tmp_path):
    """
    Tests the _get_log_stream method."""
    clerk = Clerk(tmp_path)

    tag = "123"
    with clerk._get_log_stream(tag) as stream:
        # Test if the file was created
        assert (tmp_path / (tag + ".jsonl")).exists()

    # Test if the stream is not closed after the context is closed
    assert not stream.closed

    # Test if the same stream is used for the same tag
    with clerk._get_log_stream(tag) as stream2:
        assert stream is stream2

    # Test if the same stream is not used for the different tag
    other_tag = "312"
    assert tag != other_tag
    with clerk._get_log_stream(other_tag) as stream3:
        assert stream is not stream3


def test_get_log_stream_mode(tmp_path):
    """
    Tests that the log stream mode is 'w' for new runs and 'a' for resumed runs.

        Args:
            tmp_path: A temporary path to use for the Clerk instance.

        Returns:
            None
    """
    clerk = Clerk(tmp_path)

    tag = "123"
    # Test if the stream mode is 'w' for 'new_run' mode
    # By default 'new_run' mode is used
    with clerk:
        with clerk._get_log_stream(tag) as stream:

            assert clerk._mode == ClerkMode.new_run
            assert stream.mode == "w"

    # Test if the stream mode is 'a' for 'resume' mode
    # The clerk.begin() should be used to set 'resume' mode
    with clerk.begin(resume=True):
        with clerk._get_log_stream(tag) as stream:

            assert clerk._mode == ClerkMode.resume
            assert stream.mode == "a"


def test_get_log_stream_flushed(tmp_path):
    """
    Tests the behavior of log stream flushing within a context manager.

        This test verifies that the flush method is called on the underlying stream
        only when explicitly requested via the `flush` parameter to
        `_get_log_stream`. It also checks that flush isn't called if the context
        manager exits normally without requesting a flush.

        Args:
            tmp_path: A temporary path for Clerk initialization.

        Returns:
            None
    """
    # TODO: refactoring
    clerk = Clerk(tmp_path)
    tag = "123"

    with clerk._get_log_stream(tag) as stream:
        pass

    # Test if flush does not called after context is closed
    is_flushed = False

    def monkey_flush():
        nonlocal is_flushed
        is_flushed = True

    stream.flush = monkey_flush

    with clerk._get_log_stream(tag) as stream:
        pass
    assert not is_flushed

    # Test if flush is called after context is closed if flush is true
    with clerk._get_log_stream(tag, flush=True) as stream:
        pass
    assert is_flushed


def test_conditions(tmp_path):
    """
    Tests the saving and loading of experiment conditions.

        This method creates a Clerk instance, saves a dictionary of conditions to
        a JSON file within an experiment directory, and then verifies that the
        directory and file are created correctly. It also loads the conditions
        from the saved file and asserts that they match the original conditions.

        Args:
            tmp_path: A temporary path for creating the experiment directory.

        Returns:
            None
    """
    experiment_dir = tmp_path / "experiment"
    clerk = Clerk(experiment_dir)

    conditions = {
        "test1": 123,
        "test2": [123, 10.0, "a"],
        "test3": {"t": "e", "s": "t"},
    }
    clerk.save_conditions(conditions)

    # Test if the folder and the file are created
    assert experiment_dir.exists()
    assert (experiment_dir / "conditions.json").exists()

    # Test if when loaded, the conditions are the same
    new_clerk = Clerk(experiment_dir)
    loaded_conditions = new_clerk.load_conditions()

    assert loaded_conditions is not conditions
    assert loaded_conditions == conditions


def test_logs(tmp_path):
    """
    Tests the log writing and loading functionality of the Clerk.

        This tests checks that logs cannot be written before a context is active,
        that files are created/appended to correctly, and that loaded messages match
        the original messages in both regular and resume modes.

        Args:
            tmp_path: A temporary path for creating log files.

        Returns:
            None
    """
    clerk = Clerk(tmp_path)

    tag = "test"
    messages = [{"a": 123, "b": 321.0}, {"a": 321, "b": 5423}]

    # Test if log can't be written before the clerk is used in any context
    with pytest.raises(RuntimeError):
        for message in messages:
            clerk.write_log(tag, message)

    # Test if log file does not exist
    assert not (tmp_path / (tag + ".jsonl")).exists()

    # Write the logs
    with clerk:
        for message in messages:
            clerk.write_log(tag, message)

    # Test if log file is created
    assert (tmp_path / (tag + ".jsonl")).exists()

    # Test if when loaded, the messages are the same
    loaded_messages = list(clerk.load_logs(tag))
    assert loaded_messages is not messages
    assert loaded_messages == messages

    # Test if in resume mode the logs are appended in existing file
    tag2 = "test2"
    assert not (tmp_path / (tag2 + ".jsonl")).exists()

    with clerk.begin(resume=True):
        for message in messages:
            clerk.write_log(tag2, message)

    assert (tmp_path / (tag2 + ".jsonl")).exists()

    loaded_messages = clerk.load_logs(tag2)
    for i, message in enumerate(loaded_messages):
        assert message == messages[i % len(messages)]

    # Test if when not in resume mode the logs are written in empty file
    with clerk.begin(resume=False):
        for message in messages:
            clerk.write_log(tag2, message)

    loaded_messages = list(clerk.load_logs(tag2))
    assert loaded_messages is not messages
    assert loaded_messages == messages


def test_logs_pandas(tmp_path):
    """
    Tests loading logs to a Pandas DataFrame.

        This test writes log messages with a specific tag, then loads them into a
        Pandas DataFrame and verifies that the loaded data matches the original
        messages.

        Args:
            tmp_path: A temporary path for storing log files.

        Returns:
            None
    """
    clerk = Clerk(tmp_path)

    tag = "test"
    messages = [{"a": 123, "b": 321.0}, {"a": 321, "b": 5423}]

    with clerk:
        for message in messages:
            clerk.write_log(tag, message)

    df = clerk.load_logs_to_pandas(tag)
    # Test if when loaded, the messages are the same
    assert messages == [message.to_dict() for _, message in df.iterrows()]


def test_checkpoints(tmp_path):
    """
    Tests the Clerk's checkpointing functionality.

        This tests various scenarios including writing checkpoints, loading them,
        cleaning up old checkpoints, and handling metadata and targets.

        Args:
            tmp_path: A temporary path for storing checkpoint files.

        Returns:
            None
    """
    clerk = Clerk(tmp_path)
    checkpoints_filepath = tmp_path / "checkpoints.txt"

    # Test if checkpoint can't be written before
    # the clerk is used in any context
    with pytest.raises(RuntimeError):
        clerk.write_checkpoint()

    assert not checkpoints_filepath.exists()

    # Test if the clean_checkpoints method even works
    clerk.clean_checkpoints()

    # Write checkpoint with metadata and no targets
    with clerk:
        for i in range(11):
            clerk.write_checkpoint(metadata={"i": i})

    assert checkpoints_filepath.exists()

    # Test the content of checkpoints.txt file
    clerk = Clerk(tmp_path)
    first_run_checkpoint_filenames: list[str] = []
    with open(checkpoints_filepath) as file:
        for i, line in enumerate(file.readlines()):
            checkpoint_filename = f"{i}.pt"
            first_run_checkpoint_filenames.append(checkpoint_filename)

            assert line == checkpoint_filename + "\n"

            assert (tmp_path / checkpoint_filename).exists()

            metadata = clerk.load_checkpoint(i)
            assert metadata == {"i": i}
            same_metadata = clerk.load_checkpoint(checkpoint_filename)
            assert same_metadata == metadata

    # New context of the clerk
    with clerk:
        for i in range(3):
            clerk.write_checkpoint()

    # Test if the metadata is None
    for i in range(3):
        assert clerk.load_checkpoint(i) is None

    # Test the content of checkpoints.txt file has been changed
    second_run_checkpoint_filenames = []
    with open(checkpoints_filepath) as file:
        assert len(file.readlines()) == 3
        for i in range(3):
            second_run_checkpoint_filenames.append(f"{i}.pt")

    # Test clean_checkpoints
    clerk.clean_checkpoints()
    for checkpoint_filename in first_run_checkpoint_filenames:
        if checkpoint_filename in second_run_checkpoint_filenames:
            assert (tmp_path / checkpoint_filename).exists()
        else:
            assert not (tmp_path / checkpoint_filename).exists()

    # Test clean_checkpoints if checkpoints.txt does not exist
    checkpoints_filepath.unlink()
    clerk.clean_checkpoints()
    for checkpoint_filename in first_run_checkpoint_filenames:
        assert not (tmp_path / checkpoint_filename).exists()

    class ObjectWithState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.test_parameter: torch.Tensor
            self.register_buffer("test_parameter", torch.tensor(0.0))

    clerk = Clerk(tmp_path)
    object1 = ObjectWithState()
    object2 = ObjectWithState()
    clerk.set_checkpoint_targets({"1": object1, "2": object2})

    # Write checkpoint with targets
    with clerk:
        for i in range(6):
            object2.test_parameter += 2
            clerk.write_checkpoint()

    # Test load_checkpoint
    for i in range(6):
        clerk.load_checkpoint(i)
        assert object1.test_parameter.item() == 0
        assert object2.test_parameter.item() == 2.0 + 2.0 * i

    # Test load_checkpoint for specific target
    object1.test_parameter = torch.tensor(123)
    object2.test_parameter = torch.tensor(321)
    object3 = ObjectWithState()
    for i in range(6):
        clerk.load_checkpoint(i, targets={"2": object3})
        # Test if object1 and object2 does not change
        assert object1.test_parameter.item() == 123
        assert object2.test_parameter.item() == 321
        # Test if object3 changes
        assert object3.test_parameter.item() == 2 + 2 * i

    # Test if more checkpoints has been saved when resume mode
    clerk = Clerk(tmp_path)
    clerk.set_checkpoint_targets({"1": object1, "2": object2})
    assert object1.test_parameter.item() != 0
    assert object2.test_parameter.item() != 16

    with clerk.begin(resume=True):
        for i in range(2):
            object2.test_parameter += 2
            clerk.write_checkpoint()

    with open(checkpoints_filepath) as file:
        assert len(file.readlines()) == 8

    # Test if last checkpoint was automatically loaded in resume mode
    assert object1.test_parameter.item() == 0
    assert object2.test_parameter.item() == 16

    # Test if resume_load_last_checkpoint can be turned off
    clerk = Clerk(tmp_path)
    clerk.set_checkpoint_targets({"1": object1, "2": object2})
    object1.test_parameter = torch.tensor(123)
    object2.test_parameter = torch.tensor(321)

    with clerk.begin(resume=True, resume_load_last_checkpoint=False):
        for i in range(6):
            object2.test_parameter += 2
            clerk.write_checkpoint()

    # Test if last checkpoint was not automatically loaded
    assert object1.test_parameter.item() == 123
    assert object2.test_parameter.item() == 321 + 6 * 2


def test_context(tmp_path):
    """
    Tests the Clerk context manager functionality.

        This tests nested context usage, directory creation, and automatic stream closing.
        It also verifies exception handling when detaching a stream within a clerk context.

        Args:
            tmp_path: A temporary path for testing purposes.

        Returns:
            None
    """
    new_path = tmp_path / "test"
    clerk = Clerk(new_path)

    assert not new_path.exists()

    with pytest.raises(RuntimeError):
        with clerk:
            with clerk:
                pass

    # Test if the folder has been made
    assert new_path.exists()

    # Test if all streams are closed automatically
    with clerk._get_log_stream("test", flush=False) as stream:
        assert not stream.closed

        with clerk:
            pass

        assert stream.closed

    with pytest.raises(ExceptionGroup):
        with clerk._get_log_stream("test", flush=False) as stream:
            with clerk:
                stream.detach()


def test_backup_checkpoint(tmp_path):
    """
    Tests the backup checkpoint functionality of the Clerk class.

        This test verifies that a backup checkpoint is created when an exception
        occurs during a clerk session with autosave enabled, and that it can be
        loaded and cleaned up correctly. It also tests the handling of exceptions
        during the preparation of checkpoint data.

        Args:
            tmp_path: A temporary path for creating test files.

        Returns:
            None
    """
    clerk = Clerk(tmp_path)
    checkpoints_filepath = tmp_path / "checkpoints.txt"

    class SpecificException(Exception):
        pass

    try:
        with clerk.begin(autosave_checkpoint=True):
            clerk.write_checkpoint()
            raise SpecificException
    except SpecificException:
        pass

    # Test if the backup checkpoint is not in 'checkpoints.txt'
    with open(checkpoints_filepath) as file:
        assert file.readlines() == ["0.pt\n"]

    backup_checkpoints: list[str] = []
    # Find backup checkpoint files
    for file in tmp_path.iterdir():
        if file.name.endswith(".pt"):
            if not CHECKPOINT_FILENAME_PATTERN.match(file.name):
                backup_checkpoints.append(file.name)

    assert len(backup_checkpoints) == 1

    # Test metadata
    metadata = clerk.load_checkpoint(backup_checkpoints[0])
    assert isinstance(metadata, dict)
    assert "time" in metadata
    assert "description" in metadata

    # Test clean_backup_checkpoints method
    assert (tmp_path / backup_checkpoints[0]).exists()
    clerk.clean_backup_checkpoints()
    assert not (tmp_path / backup_checkpoints[0]).exists()

    # Test if exception group raised when error during
    # backup checkpoint saving
    class PrepareCheckpointDataException(Exception):
        pass

    def destroyed_prepare_checkpoint_data(*args, **kwargs):
        raise PrepareCheckpointDataException

    with pytest.raises(ExceptionGroup) as exc_info:
        with clerk.begin(autosave_checkpoint=True):
            clerk._prepare_checkpoint_data = destroyed_prepare_checkpoint_data
            raise SpecificException

    assert exc_info.group_contains(PrepareCheckpointDataException)
