from svetlanna.specs import ParameterSpecs, SubelementSpecs, PrettyReprRepr, Specsable
from svetlanna import Wavefront
from collections import deque
from typing import TYPE_CHECKING, Iterable, cast
from svetlanna.visualization import ElementHTML, jinja_env
import torch
from .base import LinearOpticalSetupLike


class SimpleReservoir(torch.nn.Module):
    def __init__(
        self,
        nonlinear_element: LinearOpticalSetupLike,
        delay_element: LinearOpticalSetupLike,
        feedback_gain: float,
        input_gain: float,
        delay: int,
    ) -> None:
        r"""Reservoir network.
        The main idea is explained in [the work](https://doi.org/10.1364/OE.20.022783).
        The governing formula is:
        $$
        x_\text{out}[i] = F_\text{NL}(\beta x_\text{in}[i] + \alpha F_\text{D}(x_\text{out}[i-\tau]))
        $$
        where $F_\text{NL}$ is the nonlinear element, $F_\text{D}$ is the delay element,
        $\alpha$ is the feedback_gain, $\beta$ is the input_gain,
        $\tau$ is the delay in samples.
        The user should match the delay in samples with the actual
        light propagation time in $F_\text{D}$.

        Parameters
        ----------
        nonlinear_element : LinearOpticalSetupLike
            The nonlinear element the light passes through.
        delay_element : LinearOpticalSetupLike
            The delay line element.
        feedback_gain : float
            The feedback (delay line) gain $\alpha$.
        input_gain : float
            The input gain $\beta$
        delay : int
            The delay time, measured in samples,
            that the light spends in the delay line.

        Examples
        --------
        ```python
        import svetlanna as sv

        ...

        reservoir = sv.networks.SimpleReservoir(
            nonlinear_element=nonlinear_element,
            delay_element=delay_element,
            delay=2,
            feedback_gain=feedback_gain,
            input_gain=input_gain,
        )

        for input_wavefront in input_wavefront_sequence:
            output = reservoir(input_wavefront)

        # clear the delay line before the next sequence or batch
        reservoir.drop_feedback_queue()
        ```
        """
        super().__init__()

        self.nonlinear_element = nonlinear_element
        self.delay_element = delay_element

        self.feedback_gain = feedback_gain
        self.input_gain = input_gain
        self.delay = delay

        # create FIFO queue for delay line
        self.feedback_queue: deque[Wavefront] = deque(maxlen=self.delay)

    def append_feedback_queue(self, field: Wavefront):
        """Append a new wavefront to the feedback queue.

        Parameters
        ----------
        field : Wavefront
            The new wavefront to be added to the end of the queue.
        """
        self.feedback_queue.append(field)

    def pop_feedback_queue(self) -> None | Wavefront:
        """Retrieve and remove the first element from the feedback queue
        if available.

        Returns
        -------
        None | Wavefront
            The first wavefront in the queue if the queue is not empty;
            otherwise, None.
        """
        if len(self.feedback_queue) < self.delay:
            return None
        return self.feedback_queue.popleft()

    def drop_feedback_queue(self) -> None:
        """Clear all elements from the feedback queue."""
        self.feedback_queue.clear()

    def forward(self, input_wavefront: Wavefront) -> Wavefront:
        # get an element from feedback line queue
        delayed = self.pop_feedback_queue()

        if delayed is not None:
            delay_output = self.feedback_gain * self.delay_element(delayed)
            output = self.nonlinear_element(
                input_wavefront * self.input_gain + delay_output
            )
        else:
            # if the delay line is empty
            output = self.nonlinear_element(input_wavefront * self.input_gain)

        # add output to the delay line
        self.append_feedback_queue(output)
        return output

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        specs: list[ParameterSpecs | SubelementSpecs] = [
            ParameterSpecs("feedback_gain", (PrettyReprRepr(self.feedback_gain),)),
            ParameterSpecs("input_gain", (PrettyReprRepr(self.input_gain),)),
            ParameterSpecs("delay", (PrettyReprRepr(self.delay),)),
        ]

        if hasattr(self.nonlinear_element, "to_specs"):
            nonlinear_element = cast(Specsable, self.nonlinear_element)
            specs.append(SubelementSpecs("Nonlinear element", nonlinear_element))
        if hasattr(self.delay_element, "to_specs"):
            delay_element = cast(Specsable, self.delay_element)
            specs.append(SubelementSpecs("Delay element", delay_element))
        return specs

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template("widget_reservoir.html.jinja").render(
            index=index, name=name, subelements=subelements
        )

    if TYPE_CHECKING:

        def __call__(self, incident_wavefront: Wavefront) -> Wavefront: ...
