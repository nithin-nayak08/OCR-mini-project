import easyocr

# Basic reader; parameters will be tuned in readtext
ocr_reader = easyocr.Reader(["en"], gpu=False)

def run_ocr(image):
    """
    Run EasyOCR on a preprocessed image.

    Returns result wrapped in a Paddle-like structure:
    [[(box, (text, score)), ...]]
    """
    result = ocr_reader.readtext(
        image,
        detail=1,          # box, text, score
        paragraph=False,
        low_text=0.3,      # keep low-score detections
        text_threshold=0.4 # more sensitive than default
    )
    wrapped = [[(box, (text, score)) for (box, text, score) in result]]
    return wrapped
