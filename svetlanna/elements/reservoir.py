from ..parameters import OptimizableFloat
from ..simulation_parameters import SimulationParameters
from ..wavefront import Wavefront
from .element import Element
from collections import deque


class SimpleReservoir(Element):
    """Reservoir element."""
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        nonlinear_element: Element,
        delay_element: Element,
        feedback_gain: OptimizableFloat,
        input_gain: OptimizableFloat,
        delay: int
    ) -> None:
        """Reservoir element.
        The main idea is explained in https://doi.org/10.1364/OE.20.022783.
        The formula is following
        $$
        x_{out}[i] = F_{NL}(\beta x_{in}[i] + \alpha F_{D}(x_{out}[i-\tau]))
        $$
        where $F_{NL}$ is nonlinear element, $F_{D}$ is delay element,
        $\alpha$ is feedback_gain, $\beta$ is input_gain, $\tau$ is delay.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            An instance describing the optical system's simulation parameters.
        nonlinear_element : Element | LinearOpticalSetup
            Nonlinear element the light goes through.
        delay_element : Element | LinearOpticalSetup
            Delay line element.
        feedback_gain : OptimizableFloat
            The feedback (delay line) gain $\alpha$.
        input_gain : OptimizableFloat
            The input gain $\beta$
        delay : int
            The time in samples light goes through delay line.
        """
        super().__init__(simulation_parameters)

        self.nonlinear_element = nonlinear_element
        self.delay_element = delay_element

        self.feedback_gain = self.process_parameter(
            'feedback_gain', feedback_gain
        )
        self.input_gain = self.process_parameter(
            'input_gain', input_gain
        )
        self.delay = self.process_parameter(
            'delay', delay
        )

        # create FIFI queue for delay line
        self.feedback_queue: deque[Wavefront] = deque(maxlen=self.delay)

    def append_feedback_queue(self, field: Wavefront):
        """Append feedback line queue with wavefront (add element to the end).

        Parameters
        ----------
        field : Wavefront
            New wavefront.
        """
        self.feedback_queue.append(field)

    def pop_feedback_queue(self) -> None | Wavefront:
        """Get the first element from feedback line queue if possible.

        Parameters
        ----------
        field : Wavefront
            New wavefront.
        """
        if len(self.feedback_queue) < self.delay:
            return None
        return self.feedback_queue.popleft()

    def drop_feedback_queue(self) -> None:
        """Clear the feedback line queue.
        """
        self.feedback_queue.clear()

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        # get an element from feedback line queue
        delayed = self.pop_feedback_queue()

        if delayed is not None:
            delay_output = self.feedback_gain * self.delay_element(delayed)
            output = self.nonlinear_element(
                incident_wavefront * self.input_gain + delay_output
            )
        else:
            # if the delay line is empty
            output = self.nonlinear_element(
                incident_wavefront * self.input_gain
            )

        # add output to the delay line
        self.append_feedback_queue(output)
        return output
