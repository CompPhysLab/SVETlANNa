import numpy as np
import scipy as sp


class SquareFresnel:
    """A class describing the analytical solution for the problem of free
      propagation after planar wave passes a square aperture
    """
    def __init__(
        self,
        distance: float,
        x_size: float,
        y_size: float,
        x_nodes: int,
        y_nodes: int,
        square_size: float,
        wavelength: float
    ):
        """Constructor method

        Parameters
        ----------
        distance : float
            Distance between the square aperture and the screen
        x_size : float
            System size along the axis ox
        y_size : float
            System size along the axis oy
        x_nodes : int
            Number of computational nodes along the axis ox
        y_nodes : int
            Number of computational nodes along the axis oy
        square_size : float
            Square aperture side size
        wavelength : float
            Wavelength of the incident planar wave
        """

        self.distance: float = distance
        self.x_size: float = x_size
        self.y_size: float = y_size
        self.x_nodes: int = x_nodes
        self.y_nodes: int = y_nodes
        self.square_size: float = square_size
        self.wavelength: float = wavelength

    def intensity(self) -> np.ndarray:
        """Method that describes the intensity profile on the screen

        Returns
        -------
        np.ndarray
            2d Intensity profile
        """

        wave_number = 2*np.pi/self.wavelength
        x_linear = np.linspace(-self.x_size / 2, self.x_size / 2, self.x_nodes)
        y_linear = np.linspace(-self.y_size / 2, self.y_size / 2, self.y_nodes)
        x_grid, y_grid = np.meshgrid(x_linear, y_linear)

        psi1 = -np.sqrt(wave_number/(np.pi*self.distance))*(self.square_size/2
                                                            + x_grid)
        psi2 = np.sqrt(wave_number/(np.pi*self.distance))*(self.square_size/2
                                                           - x_grid)
        eta1 = -np.sqrt(wave_number/(np.pi*self.distance))*(self.square_size/2
                                                            + y_grid)
        eta2 = np.sqrt(wave_number/(np.pi*self.distance))*(self.square_size/2
                                                           - y_grid)

        s_psi1, c_psi1 = sp.special.fresnel(psi1)
        s_psi2, c_psi2 = sp.special.fresnel(psi2)
        s_eta1, c_eta1 = sp.special.fresnel(eta1)
        s_eta2, c_eta2 = sp.special.fresnel(eta2)

        intensity = (1/4)*(np.power((c_psi2 - c_psi1), 2) +
                           np.power((s_psi2 - s_psi1), 2))*(
                               np.power((c_eta2 - c_eta1), 2) + np.power(
                                   (s_eta2 - s_eta1), 2))

        return intensity


class CircleFresnel:
    """A class describing the analytical solution for the problem of free
      propagation after planar wave passes a circular aperture aperture
    """
    def __init__(
        self,
        distance: float,
        x_size: float,
        y_size: float,
        x_nodes: int,
        y_nodes: int,
        radius_aperture: float,
        wavelength: float
    ):
        """Constructor method

        Parameters
        ----------
        distance : float
            Distance between the circular aperture and the screen
        x_size : float
            System size along the axis ox
        y_size : float
            System size along the axis oy
        x_nodes : int
            Number of computational nodes along the axis ox
        y_nodes : int
            Number of computational nodes along the axis oy
        radius_aperture : float
            Radius of the circular aperture
        wavelength : float
            wavelength of the incident plane wave
        """
        self.distance: float = distance
        self.x_size: float = x_size
        self.y_size: float = y_size
        self.x_nodes: int = x_nodes
        self.y_nodes: int = y_nodes
        self.radius_aperture: float = radius_aperture
        self.wavelength: float = wavelength

    def intensity(self, amplitude_initial: np.ndarray) -> np.ndarray:
        """Method that describes the 2d intensity profile on the screen

        :param amplitude_initial: 2d-amplitude of the incident wave
        :type amplitude_initial: np.ndarray
        :return: 2d intensity profile on the screen
        :rtype: np.ndarray
        """

        x_linear = np.linspace(-self.x_size / 2, self.x_size / 2, self.x_nodes)
        y_linear = np.linspace(-self.y_size / 2, self.y_size / 2, self.y_nodes)
        x_grid, y_grid = np.meshgrid(x_linear, y_linear)

        radius_squared = np.power(x_grid, 2) + np.power(y_grid, 2)
        radius = np.sqrt(radius_squared)

        wave_number = 2*np.pi/self.wavelength

        def func_cos(x, y):
            return x*np.cos(np.pi*x**2/(self.wavelength*self.distance))*(
                sp.special.j0(2*np.pi*x*y/(self.wavelength*self.distance)))

        def func_sin(x, y):
            return x*np.sin(np.pi*x**2/(self.wavelength*self.distance))*(
                sp.special.j0(2*np.pi*x*y/(self.wavelength*self.distance)))

        field_after_aperture = np.zeros((self.y_nodes, self.x_nodes))

        for i in range(self.x_nodes):
            for j in range(self.y_nodes):

                integral_cos = sp.integrate.quad(func_cos, 0.,
                                                 self.radius_aperture,
                                                 args=(radius[i, j],),
                                                 limit=1000)[0]

                integral_sin = sp.integrate.quad(func_sin, 0.,
                                                 self.radius_aperture,
                                                 args=(radius[i, j],),
                                                 limit=1000)[0]
                field_after_aperture[i, j] = amplitude_initial[i, j]*(
                    wave_number)/(1j*self.distance)*np.exp(1j*(
                        wave_number*self.distance +
                        wave_number*radius[i, j]**2/(2*self.distance)))*(
                            integral_cos + 1j*integral_sin)

        intensity = np.abs(field_after_aperture)**2

        return intensity

    def intensity_1d(self, amplitude_initial: np.ndarray) -> np.ndarray:
        """Method describing 1d-intensity distribution on the screen as a
#         function of distance

        Parameters
        ----------
        amplitude_initial : np.ndarray
            Method describing 1d-intensity distribution on the screen as a
#         function of distance

        Returns
        -------
        np.ndarray
            1d intensity profile on the screen
        """

        wave_number = 2*np.pi/self.wavelength
        x_linear = np.linspace(-self.x_size / 2, self.x_size / 2, self.x_nodes)
        y_linear = np.linspace(-self.y_size / 2, self.y_size / 2, self.y_nodes)
        x_grid, y_grid = np.meshgrid(x_linear, y_linear)

        radius_squared = np.power(x_grid, 2) + np.power(y_grid, 2)
        radius = np.sqrt(radius_squared)[int(self.y_nodes/2)]
        amplitude_initial = amplitude_initial[int(self.y_nodes/2)]

        def func_cos(x, y):
            return x*np.cos(np.pi*x**2/(self.wavelength*self.distance))*(
                sp.special.j0(2*np.pi*x*y/(self.wavelength*self.distance)))

        def func_sin(x, y):
            return x*np.sin(np.pi*x**2/(self.wavelength*self.distance))*(
                sp.special.j0(2*np.pi*x*y/(self.wavelength*self.distance)))

        field_after_aperture = np.array([])

        for i in range(self.x_nodes):
            integral_cos = sp.integrate.quad(func_cos, 0, self.radius_aperture,
                                             args=(radius[i],), limit=200)[0]
            integral_sin = sp.integrate.quad(func_sin, 0, self.radius_aperture,
                                             args=(radius[i],), limit=200)[0]

            field = amplitude_initial[i]*wave_number/(1j*self.distance)*np.exp(
                1j*(wave_number*self.distance + wave_number*radius[i]**2/(
                    2*self.distance)))*(integral_cos + 1j*integral_sin)

            field_after_aperture = np.append(field_after_aperture, field)

        intensity = np.abs(field_after_aperture)**2

        return intensity


# доработать
class CircleFraunhofer:
    def __init__(self, x_size, y_size, x_nodes, y_nodes, distance, wavelength,
                 diameter):
        """_summary_

        :param x_size: _description_
        :type x_size: _type_
        :param y_size: _description_
        :type y_size: _type_
        :param x_nodes: _description_
        :type x_nodes: _type_
        :param y_nodes: _description_
        :type y_nodes: _type_
        :param distance: _description_
        :type distance: _type_
        :param wavelength: _description_
        :type wavelength: _type_
        :param diameter: _description_
        :type diameter: _type_
        """

        self.x_size: float = x_size
        self.y_size: float = y_size
        self.x_nodes: int = x_nodes
        self.y_nodes: int = y_nodes
        self.distance: float = distance
        self.wavelength: float = wavelength
        self.diameter: float = diameter

    def intensity(self):
        wave_number = 2 * np.pi / self.wavelength
        x = np.linspace(-self.x_size / 2, self.x_size / 2, self.x_nodes)
        y = np.linspace(-self.y_size / 2, self.y_size / 2, self.y_nodes)
        xv, yv = np.meshgrid(x, y)

        # intensity_theory = np.zeros((self.Nx, self.Ny))

        radius = np.sqrt(np.power(xv, 2) + np.power(yv, 2))

        intensity = np.power(wave_number*self.diameter**2/(8*self.distance), 2
                             )*np.power(
                                 2*sp.special.jv(1, wave_number*self.diameter *
                                                 radius/(2*self.distance))/(
                                             wave_number*self.diameter*radius/(
                                                 2*self.distance)), 2)
        return intensity


class RectangularFraunhofer:
    def __init__(self, distance, x_size, y_size, x_nodes, y_nodes,
                 rect_size_x, rect_size_y, wavelength):
        self.distance: float = distance
        self.x_size: float = x_size
        self.y_size: float = y_size
        self.x_nodes: int = x_nodes
        self.y_nodes: int = y_nodes
        self.rect_size_x: float = rect_size_x
        self.rect_size_y: float = rect_size_y
        self.wavelength: float = wavelength

    def intensity(self):
        x = np.linspace(-self.x_size / 2, self.x_size / 2, self.x_nodes)
        y = np.linspace(-self.y_size / 2, self.y_size / 2, self.y_nodes)
        xv, yv = np.meshgrid(x, y)

        ax = self.x_size*xv/(self.wavelength*self.distance)
        ay = self.y_size*yv/(self.wavelength*self.distance)

        intensity = (self.x_size*self.y_size/(self.wavelength*self.distance)
                     )**2*(np.sin(ax)/ax)**2*(np.sin(ay)/ay)**2
        return intensity
