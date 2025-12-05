import streamlit as st
from src.preprocessing import preprocess_for_ocr
from src.ocr_engine import run_ocr
from src.text_extraction import (
    extract_target_line,
    paddle_to_word_list,
    group_words_into_lines,
    line_to_text,
)

st.title("Shipping Label OCR – Target Line Extractor")

uploaded_file = st.file_uploader(
    "Upload shipping label / waybill image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    st.image(img_bytes, caption="Uploaded image", use_column_width=True)

    if st.button("Run OCR"):
        with st.spinner("Processing..."):
            # 1) Preprocess image
            pre_img = preprocess_for_ocr(img_bytes)

            # 2) Run OCR
            ocr_result = run_ocr(pre_img)

            # 3) Debug: show all detected lines
            st.subheader("Debug: all detected lines")
            words = paddle_to_word_list(ocr_result)
            lines = group_words_into_lines(words)
            for i, lw in enumerate(lines):
                st.write(f"Line {i}: {line_to_text(lw)}")

            # 4) Extract line containing digit '1'
            #    (relaxed pattern; earlier it was '_1_')
            target_line, conf, line_words = extract_target_line(
                ocr_result,
                pattern="_1"
            )

        st.subheader("Extracted line containing '1'")
        if target_line:
            st.write(f"**Line:** {target_line}")
            st.write(f"**Estimated confidence:** {conf:.3f}")
        else:
            st.error("No line containing '1' was found in this image.")

        st.subheader("Raw OCR JSON")
        st.json(ocr_result)
