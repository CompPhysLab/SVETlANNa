from enum import Enum

# SI prefixes
_T = 1e12
_G = 1e9
_M = 1e6
_k = 1e3
_d = 1e-1
_c = 1e-2
_m = 1e-3
_u = 1e-6
_n = 1e-9
_p = 1e-12
_f = 1e-15
_a = 1e-18


class ureg(Enum):
    """Unit registry for SI-prefixed length, time, and frequency units.

    A simple unit registry supporting SI prefixes (T, G, M, k, m, u, n, p, f, a)
    for length (m), time (s), and frequency (Hz) units. Supports basic arithmetic
    operations with scalars.

    Warning
    -------
    Units are multiplicative factors only; they carry no information about the
    physical quantity.
    Keep eye on the units you use to ensure consistency across calculations.

    Warning
    -------
    Round-off errors may occur when using very large and very small units due to floating-point precision limits.

    Examples
    --------
    ```python
    from svetlanna.units import ureg

    wavelength = 500 * ureg.nm  # 5e-7
    x = torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10)
    y = torch.linspace(-5, 5, 10) * ureg.mm

    print(f'λ={wavelength / ureg.um:.3f} μm')  # >>> λ=0.500 μm
    ```

    Attributes
    ----------
    Gm, Mm, km, m, dm, cm, mm, um, nm, pm : float
        Length units (gigameters to picometers).
    Gs, Ms, ks, s, ds, cs, ms, us, ns, ps, fs, as_ : float
        Time units (gigaseconds to attoseconds).
    THz, GHz, MHz, kHz, Hz, dHz, cHz, mHz, uHz, nHz, pHz : float
        Frequency units (terahertz to picohertz).
    """

    Gm = _G
    Mm = _M
    km = _k
    m = 1
    dm = _d
    cm = _c
    mm = _m
    um = _u
    nm = _n
    pm = _p

    Gs = _G
    Ms = _M
    ks = _k
    s = 1
    ds = _d
    cs = _c
    ms = _m
    us = _u
    ns = _n
    ps = _p
    fs = _f
    as_ = _a

    THz = _T
    GHz = _G
    MHz = _M
    kHz = _k
    Hz = 1
    dHz = _d
    cHz = _c
    mHz = _m
    uHz = _u
    nHz = _n
    pHz = _p

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __pow__(self, other):
        return self.value**other

    def __array__(self, dtype=None, copy=None):
        import numpy

        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")
        return numpy.array(self.value, dtype=dtype)
