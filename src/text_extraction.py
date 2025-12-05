from typing import List, Dict, Any, Tuple

def paddle_to_word_list(ocr_result) -> List[Dict[str, Any]]:
    """
    Flatten PaddleOCR result into a list of words with centers and confidence.

    Each item: {
        "text": str,
        "conf": float,
        "cx": float,
        "cy": float,
        "box": list  # 4 corner points
    }
    """
    words = []
    if not ocr_result:
        return words

    for line in ocr_result:
        for box, (txt, score) in line:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cx = float(sum(xs) / 4.0)
            cy = float(sum(ys) / 4.0)
            words.append(
                {
                    "text": txt,
                    "conf": float(score),
                    "cx": cx,
                    "cy": cy,
                    "box": box,
                }
            )
    return words

def group_words_into_lines(
    words: List[Dict[str, Any]], y_thresh: float = 25.0
) -> List[List[Dict[str, Any]]]:
    """
    Group words into lines using y-coordinate proximity.
    """
    if not words:
        return []

    # sort by y then x
    words_sorted = sorted(words, key=lambda w: (w["cy"], w["cx"]))
    lines: List[List[Dict[str, Any]]] = []
    current_line: List[Dict[str, Any]] = [words_sorted[0]]
    current_y = words_sorted[0]["cy"]

    for w in words_sorted[1:]:
        if abs(w["cy"] - current_y) <= y_thresh:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
            current_y = w["cy"]

    lines.append(current_line)
    return lines

def line_to_text(line_words: List[Dict[str, Any]]) -> str:
    """
    Convert a list of word dicts into a single text line.
    """
    line_words = sorted(line_words, key=lambda w: w["cx"])
    return " ".join(w["text"] for w in line_words)

def extract_target_line(
    ocr_result, pattern: str = "_1_"
) -> Tuple[str | None, float, List[Dict[str, Any]]]:
    """
    Find the text line that contains a given pattern.

    Returns:
        (line_text or None, avg_confidence, line_words)
    """
    words = paddle_to_word_list(ocr_result)
    lines = group_words_into_lines(words)

    best_line = None
    best_conf = 0.0
    best_words: List[Dict[str, Any]] = []

    for lw in lines:
        text = line_to_text(lw)
        if pattern in text:
            avg_conf = float(sum(w["conf"] for w in lw) / len(lw))
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_line = text
                best_words = lw

    return best_line, best_conf, best_words
