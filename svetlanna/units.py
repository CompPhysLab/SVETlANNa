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
        """
        Multiplies the value of this object by another number.

          Args:
            other: The number to multiply this object's value by.

          Returns:
            float: The result of multiplying the object's value by the other number.
        """
        return self.value * other

    def __rmul__(self, other):
        """
        Returns the result of multiplying 'other' by the value.

          This method enables multiplication with this object on either side
          (e.g., `2 * MyObject` or `MyObject * 2`). It leverages Python's
          multiplication operator to achieve this.

          Args:
            other: The value to multiply by the object's value.

          Returns:
            The result of the multiplication.
        """
        return other * self.value

    def __truediv__(self, other):
        """
        Divides the value of this object by another.

          Args:
            other: The number to divide this object's value by.

          Returns:
            float: The result of dividing this object's value by the given number.
        """
        return self.value / other

    def __rtruediv__(self, other):
        """
        Divides another number by the value of this instance.

          Args:
            other: The number to be divided by the instance's value.

          Returns:
            float: The result of dividing `other` by the instance's `value`.
        """
        return other / self.value

    def __pow__(self, other):
        """
        Calculates the power of this value.

          Args:
            other: The exponent to raise the value to.

          Returns:
            float: The result of raising the value to the power of 'other'.
        """
        return self.value**other

    def __array__(self, dtype=None, copy=None):
        """
        Returns an array representation of the value.

            Args:
                dtype: The desired data type of the returned array.
                copy: Whether to allocate a copy of the underlying data.

            Returns:
                numpy.ndarray: A NumPy array containing the values.  A copy is always created,
                               so attempting to set `copy=False` will raise a ValueError.
        """
        import numpy

        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")
        return numpy.array(self.value, dtype=dtype)
