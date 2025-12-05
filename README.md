## Shipping Label OCR – Mini Project

This project is a mini OCR system that reads shipping label / waybill images and extracts the full text line that contains the digit `1` (for example a tracking‑like string such as `163233702292313922_1_lWV`). The focus is on robust extraction of that target line even when labels are noisy or partially degraded.

### Tech stack

- Python  
- Streamlit for the web UI  
- EasyOCR as the OCR engine (open‑source)  
- OpenCV and NumPy for image preprocessing

### Features

- Upload label images (JPG / JPEG / PNG) from the browser  
- Image preprocessing: resize, grayscale, denoise, contrast enhancement, binarization  
- OCR using EasyOCR with bounding boxes and confidence scores  
- Post‑processing to group words into lines and extract the line containing `1`  
- Debug view that prints all detected text lines to help analyse failures

### Project structure

```text
.
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── src/
│   ├── preprocessing.py   # Image preprocessing pipeline
│   ├── ocr_engine.py      # EasyOCR wrapper
│   ├── text_extraction.py # Line grouping and target-line extraction
│   └── utils.py           # Helper utilities (if needed)
└── tests/
    └── __init__.py        # Placeholder for unit tests
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/OCR-mini-project.git
cd OCR-mini-project
```

2. Install dependencies (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

### Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

Workflow:

1. Upload a shipping label/waybill image.  
2. Click **Run OCR**.  
3. The app shows:
   - All detected text lines (debug section)  
   - The extracted line that contains the digit `1` and an estimated confidence score  
   - Raw OCR JSON output, which can be saved as part of the assessment results

### Approach (short)

- **Preprocessing:** resize small images, convert to grayscale, denoise, apply CLAHE for contrast, then adaptive thresholding to create a clean binary image for OCR.
- **OCR:** run EasyOCR on the preprocessed image to obtain bounding boxes, recognized text, and per‑word confidence scores.
- **Text extraction:** convert words to line groups based on y‑coordinate proximity, sort by x‑position, join into full lines, and select the line whose text contains the pattern `"1"`.
For Testing click on https://drive.google.com/drive/folders/1Psu8m0ZiHzMpxa8-4mKg5wsbverHIBIX?usp=drive_link for OCR images
  


