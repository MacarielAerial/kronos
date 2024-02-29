from kronos.nodes.utils_data_interfaces import generate_timestamp


def test_generate_timestamp_format() -> None:
    timestamp = generate_timestamp()
    # Example format: 2024-01-01T00.00.00Z
    assert timestamp.endswith("Z"), "Timestamp must end with 'Z'"
    assert (
        len(timestamp) == 24
    ), "Timestamp format length should match YYYY-MM-DDTHH.MM.SSZ"
