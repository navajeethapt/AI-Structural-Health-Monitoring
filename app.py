
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
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((128,128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array,0)

    prediction = model.predict(img_array)

    score = tf.nn.softmax(prediction[0])

    confidence = float(np.max(score))*100

    result = class_names[np.argmax(score)]

        st.subheader("Structural Analysis Result")

    if result == "Crack":
        st.error("🔴 Crack Detected")
    else:
        st.success("🟢 No Crack Detected")

    st.subheader("Confidence")
    st.write(f"{confidence:.2f}%")

    if confidence < 70:
        severity = "🟢 LOW"
    elif confidence < 90:
        severity = "🟡 MEDIUM"
    else:
        severity = "🔴 HIGH"

    st.subheader("Risk Assessment")

    if confidence < 70:
        st.info("🟢 LOW RISK")
    elif confidence < 90:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK")

    st.write(
        "Recommendation: "
        + (
            "Monitor periodically."
            if confidence < 70
            else "Schedule structural inspection."
            if confidence < 90
            else "Immediate structural inspection required."
        )
    )

    st.subheader("Severity")
    st.write(severity)

    if severity == "🟢 LOW":
        st.info("Recommendation: Monitor periodically.")

    elif severity == "🟡 MEDIUM":
        st.warning("Recommendation: Inspect within 30 days.")

    else:
        st.error("Recommendation: Immediate structural inspection required.")

    pdf_file = create_pdf(
        result,
        confidence,
        severity
    )

    st.download_button(
        label="📄 Download Inspection Report",
        data=pdf_file,
        file_name="Structural_Inspection_Report.pdf",
        mime="application/pdf"
    )
