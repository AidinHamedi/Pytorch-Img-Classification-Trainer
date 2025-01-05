# Main >>>
def filter_by_types(data, allowed_types):
    """
    Filters a dictionary to only include values of the specified types.

    Args:
        data: The dictionary to filter.
        allowed_types: A list or tuple of allowed types (e.g., [int, float, str]).

    Returns:
        A new dictionary containing only the allowed types.
    """

    def filter_value(value):
        if isinstance(value, tuple(allowed_types)):
            if isinstance(value, (list, tuple)):
                return [filter_value(item) for item in value]
            elif isinstance(value, dict):
                return {key: filter_value(val) for key, val in value.items()}
            elif isinstance(value, (set, frozenset)):
                return {filter_value(item) for item in value}
            return value
        return None

    return {k: filter_value(v) for k, v in data.items() if filter_value(v) is not None}
