from datetime import datetime, timezone

VERSION_FORMAT = "%Y-%m-%dT%H.%M.%S.%fZ"


def generate_timestamp() -> str:
    """Generate the timestamp to be used by versioning.

    Returns:
        String representation of the current timestamp.

    """
    current_ts = datetime.now(tz=timezone.utc).strftime(VERSION_FORMAT)
    return current_ts[:-4] + current_ts[-1:]  # Don't keep microseconds
