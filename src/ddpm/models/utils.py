
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def divisible_by(numer, denom):
    return (numer % denom) == 0


def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)
