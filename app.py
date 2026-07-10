
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crack_detector.keras")

model = load_model()

def highlight_cracks(image):

    # Convert PIL image to OpenCV format
    img = np.array(image)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    # Detect edges
    edges = cv2.Canny(
        blur,
        50,
        150
    )

    # Create red highlight layer
    highlighted = img.copy()

    highlighted[edges > 0] = [255, 0, 0]

    return highlighted

def create_pdf(result, confidence, severity):

    buffer = BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(
        80,
        750,
        "AI Structural Health Inspection Report"
    )

    pdf.setFont("Helvetica", 12)

    pdf.drawString(
        80,
        700,
        f"Date: {datetime.now().strftime('%d-%m-%Y')}"
    )

    pdf.drawString(
        80,
        670,
        f"Prediction: {result}"
    )

    pdf.drawString(
        80,
        640,
        f"Confidence: {confidence:.2f}%"
    )

    pdf.drawString(
        80,
        610,
        f"Risk Level: {severity}"
    )

    pdf.drawString(
        80,
        570,
        "Recommendation:"
    )

    if result == "Crack":
        recommendation = "Structural inspection recommended."
    else:
        recommendation = "Continue regular monitoring."

    pdf.drawString(
        80,
        540,
        recommendation
    )

    pdf.save()

    buffer.seek(0)

    return buffer

class_names = ["No Crack", "Crack"]

st.set_page_config(page_title="AI Structural Health Monitoring")

st.title("🏗️ AI Structural Health Monitoring")
st.write("Upload a concrete or building image to detect cracks.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    highlighted_image = highlight_cracks(image)

    st.subheader("🔍 AI Crack Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            highlighted_image,
            caption="AI Highlighted Crack Areas",
            use_container_width=True
        )
