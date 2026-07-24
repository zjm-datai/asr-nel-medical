from __future__ import annotations


EMPTY = "<empty>"


def substitution_cost(reference_character: str, hypothesis_character: str) -> int:
    if reference_character == hypothesis_character:
        return 0
    reference_is_content = reference_character.isalnum() or "\u3400" <= reference_character <= "\u9fff"
    hypothesis_is_content = hypothesis_character.isalnum() or "\u3400" <= hypothesis_character <= "\u9fff"
    return 2 if reference_is_content != hypothesis_is_content else 1


def aligned_entity_error_span(
    reference: str,
    hypothesis: str,
    start: int,
    end: int,
) -> tuple[str, float]:
    """Map a reference entity span to its character-level ASR alignment."""
    if not 0 <= start < end <= len(reference):
        raise ValueError(f"invalid reference span: {start}:{end}")

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distance = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(columns):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, columns):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + substitution_cost(reference[i - 1], hypothesis[j - 1]),
            )

    operations = []
    i = len(reference)
    j = len(hypothesis)
    while i or j:
        if (
            i
            and j
            and reference[i - 1] == hypothesis[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            operations.append(("equal", i - 1, j - 1, i - 1))
            i -= 1
            j -= 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + substitution_cost(
            reference[i - 1], hypothesis[j - 1]
        ):
            operations.append(("replace", i - 1, j - 1, i - 1))
            i -= 1
            j -= 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            operations.append(("delete", i - 1, None, i - 1))
            i -= 1
        else:
            operations.append(("insert", None, j - 1, i))
            j -= 1
    operations.reverse()

    exact = True
    hypothesis_positions = []
    mapped_reference_characters = 0
    for operation, reference_index, hypothesis_index, reference_cursor in operations:
        if reference_index is not None and start <= reference_index < end:
            exact &= operation == "equal"
            if hypothesis_index is not None:
                hypothesis_positions.append(hypothesis_index)
                mapped_reference_characters += 1
        elif operation == "insert" and start < reference_cursor < end:
            exact = False
            hypothesis_positions.append(hypothesis_index)

    if exact:
        return EMPTY, 1.0
    if not hypothesis_positions:
        return EMPTY, 0.0
    left = min(hypothesis_positions)
    right = max(hypothesis_positions) + 1
    text = hypothesis[left:right].strip()
    if not text:
        return EMPTY, 0.0
    confidence = mapped_reference_characters / max(1, end - start)
    return text, round(confidence, 6)
