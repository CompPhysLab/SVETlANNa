from enum import Enum

# SI prefixes
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
    """Unit registry.
    To use it one should multiply variable by the units:
    .. code-block:: python

        var = 10
        assert var * ureg.mm == 10*1e-2
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
        return self.value ** other

    def __array__(self, dtype=None, copy=None):
        import numpy
        if copy is False:
            raise ValueError(
                "`copy=False` isn't supported. A copy is always created."
            )
        return numpy.array(self.value, dtype=dtype)
