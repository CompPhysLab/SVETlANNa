from typing import Protocol
from svetlanna import Wavefront


class LinearOpticalSetupLike(Protocol):
    """
    Protocol for objects that behave like linear optical setups.

    This protocol provides flexibility when defining optical setups: any callable
    object (or composition of callables) is valid as long as it accepts a
    [Wavefront][svetlanna.Wavefront] and returns a [Wavefront][svetlanna.Wavefront].

    It generalizes [`LinearOpticalSetup`][svetlanna.LinearOpticalSetup].
    """

    def __call__(self, __input_wavefront: Wavefront) -> Wavefront: ...
