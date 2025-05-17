from svetlanna.specs import ParameterSpecs, SubelementSpecs, PrettyReprRepr
from ..parameters import OptimizableFloat
from ..simulation_parameters import SimulationParameters
from ..wavefront import Wavefront
from .element import Element
from collections import deque
from typing import TYPE_CHECKING, Iterable, Union
from ..visualization import ElementHTML, jinja_env

if TYPE_CHECKING:
    from svetlanna.setup import LinearOpticalSetup


class SimpleReservoir(Element):
    """Reservoir element."""

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        nonlinear_element: Union[Element, "LinearOpticalSetup"],
        delay_element: Union[Element, "LinearOpticalSetup"],
        feedback_gain: OptimizableFloat,
        input_gain: OptimizableFloat,
        delay: int,
    ) -> None:
        """Reservoir element.
        The main idea is explained in https://doi.org/10.1364/OE.20.022783.
        The governing formula is:
        $$
        x_{out}[i] = F_{NL}(\beta x_{in}[i] + \alpha F_{D}(x_{out}[i-\tau]))
        $$
        where $F_{NL}$ is the nonlinear element, $F_{D}$ is the delay element,
        $\alpha$ is the feedback_gain, $\beta$ is the input_gain,
        $\tau$ is the delay in samples.
        The user should match the delay in samples with the actual
        light propagation time in $F_{D}$.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            An instance describing the optical system's simulation parameters.
        nonlinear_element : Element | LinearOpticalSetup
            The nonlinear element the light passes through.
        delay_element : Element | LinearOpticalSetup
            The delay line element.
        feedback_gain : OptimizableFloat
            The feedback (delay line) gain ($\alpha$).
        input_gain : OptimizableFloat
            The input gain $\beta$
        delay : int
            The delay time, measured in samples,
            that the light spends in the delay line.
        """
        super().__init__(simulation_parameters)

        self.nonlinear_element = nonlinear_element
        self.delay_element = delay_element

        self.feedback_gain = self.process_parameter("feedback_gain", feedback_gain)
        self.input_gain = self.process_parameter("input_gain", input_gain)
        self.delay = self.process_parameter("delay", delay)

        # create FIFI queue for delay line
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

        Parameters
        ----------
        field : Wavefront
            The first wavefront in the queue, or None if the queue is empty
            or not full yet.
        """
        if len(self.feedback_queue) < self.delay:
            return None
        return self.feedback_queue.popleft()

    def drop_feedback_queue(self) -> None:
        """Clear all elements from the feedback queue."""
        self.feedback_queue.clear()

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """
        Processes an incoming wavefront through a feedback loop.

            This method simulates the propagation of a wavefront through a system
            with a delay line and nonlinear elements. It retrieves a delayed element
            from a queue, combines it with the current input, processes the result
            through a nonlinear function, and adds the output back to the delay line.

            Args:
                incident_wavefront: The incoming wavefront signal.

            Returns:
                Wavefront: The processed output wavefront after applying feedback and
                           nonlinear transformations.
        """
        # get an element from feedback line queue
        delayed = self.pop_feedback_queue()

        if delayed is not None:
            delay_output = self.feedback_gain * self.delay_element(delayed)
            output = self.nonlinear_element(
                incident_wavefront * self.input_gain + delay_output
            )
        else:
            # if the delay line is empty
            output = self.nonlinear_element(incident_wavefront * self.input_gain)

        # add output to the delay line
        self.append_feedback_queue(output)
        return output

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        """
        Returns an iterable of parameter and subelement specifications.

            Args:
                None

            Returns:
                Iterable[ParameterSpecs | SubelementSpecs]: An iterable containing
                ParameterSpecs for 'feedback_gain', 'input_gain', and 'delay', as well
                as SubelementSpecs for 'nonlinear_element' and 'delay_element'.
        """
        return (
            ParameterSpecs("feedback_gain", (PrettyReprRepr(self.feedback_gain),)),
            ParameterSpecs("input_gain", (PrettyReprRepr(self.input_gain),)),
            ParameterSpecs("delay", (PrettyReprRepr(self.delay),)),
            SubelementSpecs("Nonlinear element", self.nonlinear_element),
            SubelementSpecs("Delay element", self.delay_element),
        )

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        """
        Renders the HTML for a widget using a Jinja2 template.

          Args:
              index: The index of the widget.
              name: The name of the widget.
              element_type: The type of element (e.g., 'input', 'select'). Can be None.
              subelements: A list of ElementHTML objects representing the sub-elements
                  of the widget.

          Returns:
              str: The rendered HTML string for the widget.
        """
        return jinja_env.get_template("widget_reservoir.html.jinja").render(
            index=index, name=name, subelements=subelements
        )
