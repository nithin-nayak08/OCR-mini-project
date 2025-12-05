import cv2
import numpy as np

def load_image_bytes(file_bytes) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV BGR image."""
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    return img

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img_gray: np.ndarray) -> np.ndarray:
    """Reduce noise in grayscale image."""
    return cv2.fastNlMeansDenoising(img_gray, h=15)

def binarize(img_gray: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding to handle uneven lighting."""
    return cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )

def resize_for_ocr(img: np.ndarray, target_height: int = 800) -> np.ndarray:
    """Upscale small images to improve OCR accuracy."""
    h, w = img.shape[:2]
    if h >= target_height:
        return img
    scale = target_height / h
    new_size = (int(w * scale), target_height)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

def preprocess_for_ocr(file_bytes) -> np.ndarray:
    """
    Decode -> resize -> grayscale -> denoise -> CLAHE -> binarize.
    """
    img = load_image_bytes(file_bytes)
    img = resize_for_ocr(img)

    gray = to_grayscale(img)
    gray = denoise(gray)

    # Contrast enhancement for faint prints
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    bin_img = binarize(gray)
    return bin_img
