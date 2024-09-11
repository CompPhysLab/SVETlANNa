from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationParameters:

    """A class which describes characteristic sizes of the system
    """

    x_size: float
    y_size: float
    x_nodes: int
    y_nodes: int
    wavelength: float

    # def __init__(
    #     self,
    #     x_size: float,
    #     y_size: float,
    #     x_nodes: int,
    #     y_nodes: int,
    #     wavelength: float
    # ):

    #     """Constructor method

    #     :param x_size: System size along the axis ox
    #     :type x_size: float
    #     :param y_size: System size along the axis oy
    #     :type y_size: float
    #     :param x_nodes: Number of computational nodes along the axis ox
    #     :type x_nodes: int
    #     :param y_nodes: Number of computational nodes along the axis oy
    #     :type y_nodes: int
    #     :param wavelength: wavelength of the source
    #     :type wavelength: float
    #     """

    #     self._x_size = x_size
    #     self._y_size = y_size
    #     self._x_nodes = x_nodes
    #     self._y_nodes = y_nodes
    #     self._wavelength = wavelength
