from typing import Any


def squeeze_tuple(t: Any) -> Any:
    if isinstance(t, tuple):
        if len(t) == 1:
            return t[0]  # Return the single element
        return t  # Return the tuple unchanged
    raise TypeError("Input is not a tuple")
