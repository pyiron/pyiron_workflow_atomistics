Writing your own Engine
========================

To write your own Engine, you need to subclass the `Engine` class and implement the `get_calculate_fn` method.

The `get_calculate_fn` method should return a tuple of a function and a dictionary of kwargs.

The function should be a function that takes a structure and returns a dictionary of results.

The dictionary of kwargs should be a dictionary of kwargs that will be passed to the function.

The function and dictionary of kwargs will be used to calculate the structure.