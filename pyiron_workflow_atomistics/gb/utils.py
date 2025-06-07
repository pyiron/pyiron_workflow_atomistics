import pyiron_workflow as pwf


def axis_to_index(axis):
    """
    Convert an axis identifier to a numeric index.

    Parameters
    ----------
    axis : str or int
        - If str, must be one of the keys in the mapping {"a": 0, "b": 1, "c": 2}.
        - If int, it is returned as-is.

    Returns
    -------
    int
        The numeric index corresponding to the given axis.

    Raises
    ------
    ValueError
        If axis is a string not in the mapping.
    TypeError
        If axis is neither str nor int.
    """
    mapping = {"a": 0, "b": 1, "c": 2}

    if isinstance(axis, str):
        try:
            return mapping[axis]
        except KeyError:
            raise ValueError(
                f"Invalid axis string '{axis}'. "
                f"Expected one of {list(mapping.keys())}."
            )
    elif isinstance(axis, int):
        return axis
    else:
        raise TypeError(f"Axis must be either str or int, not {type(axis).__name__}.")
