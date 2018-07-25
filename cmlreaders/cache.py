"""Global handling of cache configuration."""

enabled = True


def enable():
    """Globally enable caching for readers that are configured to support it.

    """
    global enabled
    enabled = True


def disable(clear: bool = False):
    """Globally disable caching for all readers.

    Parameters
    ----------
    clear
        When True, clear all cached data.

    """
    global enabled
    enabled = False
    if clear:
        clear_all()


def clear_all():
    """Clear all cached results."""
    raise NotImplementedError
