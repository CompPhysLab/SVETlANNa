import torch
from pathlib import Path
import json
import re
from contextlib import contextmanager

from typing import Protocol, Any, Self, TYPE_CHECKING, Generator
from typing import Generic, TypeVar
from enum import StrEnum
from io import TextIOWrapper

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd


class StatefulTorchClass(Protocol):
    """A protocol for PyTorch objects that have a state dictionary,
    such as modules and optimizers."""

    def state_dict(self, *args, **kwargs) -> dict[str, Any]: ...

    def load_state_dict(self, *args, **kwargs) -> Any: ...


class ClerkMode(StrEnum):
    """An internal class used to distinguish clerk modes.

    The `new_run` mode indicates that all log files should be rewritten
    along with all checkpoints. In contrast, the `resume` mode
    means that new logs will be appended to existing log files,
    and checkpoint indices will continue to increment.

    """

    new_run = "new_run"
    resume = "resume"


CHECKPOINT_FILENAME_SUFFIX = ".pt"
CHECKPOINT_FILENAME_PATTERN = re.compile(f"^\\d+\\{CHECKPOINT_FILENAME_SUFFIX}$")

ConditionsType = TypeVar("ConditionsType")


class Clerk(Generic[ConditionsType]):
    def __init__(
        self,
        experiment_directory: str,
    ) -> None:
        """A lightweight alternative to TensorBoard and other logging frameworks
        for tracking the training process, storing experiment metadata,
        and handling checkpoints.


        The Clerk is not a new concept but a minimal implementation included
        in the framework to start training models without any dependencies.

        Parameters
        ----------
        experiment_directory : str
            The directory where experiment data will be stored.
        """
        self.experiment_directory = Path(experiment_directory)

        if self.experiment_directory.exists():
            if not self.experiment_directory.is_dir():
                raise ValueError("Experiment directory should be a directory!")

        # Paths of used files
        self._path_conditions = self.experiment_directory / "conditions.json"
        self._path_checkpoints = self.experiment_directory / "checkpoints.txt"

        # Initial states and settings of the clerk
        self._checkpoint_targets = {}
        self._mode = ClerkMode.new_run
        self._resume_load_last_checkpoint = True
        self._in_use = False
        self._last_checkpoint_index = -1
        self._log_streams: dict[str, TextIOWrapper] = {}

    def _make_experiment_dir(self) -> None:
        """Create the experiment directory if it does not exist."""
        self.experiment_directory.mkdir(exist_ok=True)

    def _path_log(self, tag: str) -> Path:
        """Return the path for a file containing logs with a specific tag.

        Parameters
        ----------
        tag : str
            The tag associated with the logs.

        Returns
        -------
        Path
            The path to the file containing logs with the specified tag.
        """
        return self.experiment_directory / f"{tag}.jsonl"

    def _path_checkpoint(self, index: str | int) -> Path:
        """Return the path for a checkpoint with a specific index.

        Parameters
        ----------
        index : str | int
            The index of the checkpoint.

        Returns
        -------
        Path
            The path to the checkpoint file.
        """
        if isinstance(index, int):
            filename = f"{index}.pt"
        else:
            filename = index

        return self.experiment_directory / filename

    @contextmanager
    def _get_log_stream(self, tag: str, flush: bool = False) -> Generator[TextIOWrapper, None, None]:
        """Yield a stream for a specific tag, where logs with the tag will be written.
        The stream automatically flushes when closing the context.

        Parameters
        ----------
        tag : str
            The logging tag.
        flush: bool
            If flush is true, the underlying stream is forcibly flushed.

        Returns
        -------
        TextIOWrapper
            A writable stream.
        """
        # Check if the stream with a specific tag is open and return it
        if tag in self._log_streams:
            stream = self._log_streams[tag]
        else:
            # Open a new stream
            path = self._path_log(tag)
            if self._mode == ClerkMode.new_run:
                mode = "w"  # Open and clean the file before using
            else:
                mode = "a"  # Open the file
            stream = open(path, mode)
            logger.debug(
                f"A new stream ({stream}) for logs "
                f"with tag '{tag}' has been opened."
            )

            # Save the stream
            self._log_streams[tag] = stream

        try:
            yield stream
        finally:
            # Flush the stream
            if flush:
                stream.flush()

    def save_conditions(self, conditions: ConditionsType):
        """Save the experiment conditions. The conditions are stored
        in the experiment directory in the file `conditions.json`.

            Parameters
            ----------
            conditions : Any
                Any data that can be serialized into JSON. For example,
                a `dict` with string keys.
        """
        # Create the experiment directory if it doesn't exist
        self._make_experiment_dir()

        # Write the conditions
        with open(self._path_conditions, "w") as file:
            json.dump(conditions, file)

    def load_conditions(self) -> ConditionsType:
        """Read and return the experiment conditions.

        Returns
        -------
        Any
            The experiment conditions.
        """
        with open(self._path_conditions, "r") as file:
            return json.load(file)

    def set_checkpoint_targets(self, targets: dict[str, StatefulTorchClass]):
        """Set the targets with state dict to be saved in the checkpoint.
        The same targets will be automatically used when loading from the checkpoint.

        Parameters
        ----------
        targets : dict[str, StatefulTorchClass]
            Targets with unique keys. The same keys should be used when
            loading the checkpoint.
        """
        self._checkpoint_targets = targets

    def begin(
        self, resume: bool = False, resume_load_last_checkpoint: bool = True
    ) -> Self:
        """Configure the clerk for a new context.

        Parameters
        ----------
        resume : bool, optional
            If True, logs will continue to append to the files if they already
            exist, and the number of checkpoints will continue to grow.
            By default, False.
        resume_load_last_checkpoint : bool, optional
            If True, the very last checkpoint (if available) will be used to load
            checkpoint targets' states before entering the context. By default, True.
            This mechanism works only if `resume=True`. The last checkpoint is
            identified in `checkpoints.txt` and has the largest index.

        Returns
        -------
        Self
            The clerk.
        """
        # set the mode
        if resume:
            self._mode = ClerkMode.resume
        else:
            self._mode = ClerkMode.new_run

        # set resume_load_last_checkpoint setting
        self._resume_load_last_checkpoint = resume_load_last_checkpoint

        return self

    def _check_in_use(self):
        """Raise an error if the clerk is not used within a context."""
        if not self._in_use:
            raise RuntimeError(
                "The clerk is not in any context! "
                'Use the clerk with a "with" statement.'
            )

    def write_log(self, tag: str, data: dict, flush: bool = False):
        """Write new data to a log file with a specific tag.
        The data should be a dictionary that can be serialized into JSON.

        Parameters
        ----------
        tag : str
            The tag for the data. The name of the log file will be
            based on this tag: `<tag>.jsonl`.
        data : dict
            The data to be logged. It should be serializable into JSON.
        flush: bool
            If flush is true, the underlying stream is forcibly flushed.
        """
        # Check if the clerk is currently in use
        self._check_in_use()

        # Prepare the data
        data_str = json.dumps(data) + "\n"
        # Get a stream to write to
        with self._get_log_stream(tag, flush) as stream:
            # Write the data
            stream.write(data_str)

    def write_checkpoint(self, metadata: object | None = None):
        """Write the states of selected targets into a new checkpoint file.

        Parameters
        ----------
        metadata : object | None, optional
            The metadata for the new checkpoint, by default None.
        """
        # Check if the clerk is currently in use
        self._check_in_use()

        # Create a dictionary to be saved
        data = {}
        for key, instance in self._checkpoint_targets.items():
            data[key] = instance.state_dict()

        if metadata is not None:
            data["checkpoint_metadata"] = metadata

        index = self._last_checkpoint_index + 1

        logger.debug(f"Starting to save the checkpoint with index {index}...")
        path = self._path_checkpoint(index)
        torch.save(data, path)

        with open(self._path_checkpoints, "a") as file:
            file.write(path.name + "\n")

        logger.debug(f"Checkpoint with index {index} was successfully saved")
        self._last_checkpoint_index = index

    def load_checkpoint(
        self, index: str | int, targets: dict[str, StatefulTorchClass] | None = None
    ) -> object | None:
        """Load the checkpoint with a specific index and apply state dicts to
        checkpoint targets. If the targets are not provided, the checkpoint
        targets are obtained from the clerk settings.

        If the index is integer, that the checkpoint `<index>.pt` is used.
        Otherwise, index is used as a filename.


        Parameters
        ----------
        index : str | int
            The checkpoint index.
        targets : dict[str, StatefulTorchClass] | None, optional
            Targets with unique keys, by default None.
            See the `set_checkpoint_targets` method.

        Returns
        -------
        object | None
            The checkpoint metadata, if it exists.
        """

        path = self._path_checkpoint(index)
        data = torch.load(path)

        metadata = data.get("checkpoint_metadata")

        if targets is None:
            targets = self._checkpoint_targets

        for key, instance in targets.items():
            instance.load_state_dict(data[key])

        logger.info(
            f"Checkpoint with index {index} has been loaded. "
            "Keys of the targets that have been loaded from the state dict: "
            f'{" ".join(targets.keys())}'
        )

        return metadata

    def load_logs(self, tag: str) -> Generator[dict, None, None]:
        """Load logs from the file specified by the tag.

        Parameters
        ----------
        tag : str
            The tag for the data. The name of the log file will be
            read is `<tag>.jsonl`.

        Yields
        ------
        Generator[dict, None, None]
            Log data with the specific tag.
        """
        path = self._path_log(tag)
        with open(path, "r") as file:
            while line := file.readline():
                yield json.loads(line)

    def load_logs_to_pandas(self, tag: str) -> "pd.DataFrame":
        """Load logs from the file specified by the tag and return them as a pandas
        DataFrame.

        Parameters
        ----------
        tag : str
            The tag for the data. The name of the log file will be
            read is `<tag>.jsonl`.

        Returns
        -------
        pd.DataFrame
            Log data with the specific tag.
        """
        import pandas as pd

        path = self._path_log(tag)
        return pd.read_json(path, lines=True)

    def _checkpoint_filename_correctness(self, filename: str) -> bool:
        """Check if the checkpoint filename is correct."""
        return bool(re.match(CHECKPOINT_FILENAME_PATTERN, filename))

    def clean_checkpoints(self):
        """Remove checkpoints that are not listed in `checkpoints.txt`.
        If `checkpoints.txt` does not exist, then remove all .pt files.
        """

        if self._path_checkpoints.exists():
            # If checkpoints file exists, read all checkpoint files names
            with open(self._path_checkpoints, "r") as file:
                all_checkpoints_filenames = file.read().split("\n")

            for file in self.experiment_directory.iterdir():
                filename = file.name
                if self._checkpoint_filename_correctness(filename):
                    if filename not in all_checkpoints_filenames:
                        # if the file name is not listed in checkpoints file
                        file.unlink()
        else:
            # Remove all files that matches the pattern
            for file in self.experiment_directory.iterdir():
                filename = file.name
                if self._checkpoint_filename_correctness(filename):
                    file.unlink()

    def __enter__(self):
        # Check if the clerk is not in use in other context
        if self._in_use:
            raise RuntimeError("The clerk is already is used in some other context!")
        self._in_use = True

        # Create the experiment directory if it doesn't exist
        self._make_experiment_dir()

        if self._mode == ClerkMode.resume:
            # If the clerk in resume mode
            if self._path_checkpoints.exists():
                # if `checkpoints.txt` exists find an index of the last checkpoint
                with open(self._path_checkpoints, "r") as file:
                    index = -1
                    while line := file.readline():
                        line = line.removesuffix("\n")
                        if self._checkpoint_filename_correctness(line):
                            new_index = int(
                                line.removesuffix(CHECKPOINT_FILENAME_SUFFIX)
                            )
                            if new_index > index:
                                index = new_index

                    self._last_checkpoint_index = index
                    logger.debug(
                        f"Index {self._last_checkpoint_index} of the last "
                        "checkpoint has been retrieved from checkpoints.txt."
                    )

            if self._last_checkpoint_index >= 0:
                # If last checkpoint index is found
                if self._resume_load_last_checkpoint:
                    # If resume_load_last_checkpoint is set to True
                    self.load_checkpoint(index=self._last_checkpoint_index)

        else:
            # Remove already existing `checkpoints.txt`
            self._path_checkpoints.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc_value, traceback):
        """Resets the clerk's internal state and closes all log streams when
        exiting the context. This method ensures that any resources are
        properly released and that the clerk is ready for a new context.
        """
        self._in_use = False
        self._last_checkpoint_index = -1
        self._mode = ClerkMode.new_run
        self._resume_load_last_checkpoint = True

        exceptions = []
        for stream in self._log_streams.values():
            try:
                stream.close()
                logger.debug(f"The stream ({stream}) for logs has been closed")
            except Exception as e:
                exceptions.append(e)
        self._log_streams = {}

        if exceptions:
            raise ExceptionGroup(
                "Exceptions occurred during log file closing", exceptions
            )
