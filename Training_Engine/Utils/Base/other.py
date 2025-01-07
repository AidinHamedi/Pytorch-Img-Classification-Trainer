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


def format_seconds(seconds: int) -> str:
    """
    Converts a given number of seconds into a human-readable time string.

    Parameters:
        seconds (int): The total number of seconds.

    Returns:
        str: A string representing the time in the format of hs ms s,
             where h, m, and s are hours, minutes, and seconds respectively.
             Only includes units with non-zero values.
    """
    hours = seconds // 3600
    remaining = seconds % 3600
    minutes = remaining // 60
    seconds = round(remaining % 60)

    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours}h")
    if minutes > 0:
        time_parts.append(f"{minutes}m")
    if seconds > 0:
        time_parts.append(f"{seconds}s")

    if time_parts == []:
        return "0s"

    return " ".join(time_parts)
