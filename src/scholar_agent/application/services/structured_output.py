import json
import re


def parse_items(
    raw_output: str, first_key: str, second_key: str
) -> tuple[tuple[str, str], ...]:
    """Parse a model-generated JSON array containing two string fields."""
    start_index = raw_output.find("[")
    end_index = raw_output.rfind("]")
    if start_index == -1 or end_index == -1 or end_index < start_index:
        normalized_output = raw_output.strip().removeprefix("```json").removeprefix("```")
        normalized_output = normalized_output.removesuffix("```").strip()
    else:
        normalized_output = raw_output[start_index : end_index + 1]

    # Clean up trailing commas inside arrays/objects
    normalized_output = re.sub(r",(\s*[\]}])", r"\1", normalized_output)

    try:
        payload = json.loads(normalized_output)
    except json.JSONDecodeError as error:
        raise ValueError("The local model did not return valid JSON.") from error
    if not isinstance(payload, list):
        raise ValueError("The local model must return a JSON array.")

    parsed_items: list[tuple[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each generated item must be a JSON object.")
        first_value = item.get(first_key)
        second_value = item.get(second_key)
        if not isinstance(first_value, str) or not isinstance(second_value, str):
            raise ValueError("Generated item fields must be strings.")
        parsed_items.append((first_value.strip(), second_value.strip()))
    return tuple(parsed_items)
