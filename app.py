
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Structural Health Monitoring",
    page_icon="🏗️",
    layout="wide"
)

# -----------------------------
# Load AI Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crack_detector.keras")

model = load_model()

class_names = ["No Crack", "Crack"]

# -----------------------------
# Crack Highlight Function
# -----------------------------
def highlight_cracks(image):

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    highlighted = img.copy()

    highlighted[edges > 0] = [255, 0, 0]

    return highlighted

# -----------------------------
# PDF Generator
# -----------------------------
def create_pdf(result, confidence, severity):

    buffer = BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(60, 760, "AI Structural Health Inspection Report")

    pdf.setFont("Helvetica", 12)

    pdf.drawString(60, 720, f"Date: {datetime.now().strftime('%d-%m-%Y')}")

    pdf.drawString(60, 690, f"Prediction: {result}")

    pdf.drawString(60, 660, f"Confidence: {confidence:.2f}%")

    pdf.drawString(60, 630, f"Risk Level: {severity}")

    if result == "Crack":
        recommendation = "Immediate structural inspection is recommended."
    else:
        recommendation = "No visible crack detected. Continue routine monitoring."

    pdf.drawString(60, 600, "Recommendation:")

    pdf.drawString(60, 570, recommendation)

    pdf.save()

    buffer.seek(0)

    return buffer

# -----------------------------
# App UI
# -----------------------------
st.title("🏗️ AI Structural Health Monitoring")

st.write(
    "Upload a concrete or building image to detect structural cracks."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
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

    img = image.resize((128, 128))

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    score = tf.nn.softmax(prediction[0])

    confidence = float(np.max(score)) * 100

    result = class_names[np.argmax(score)]

    st.divider()

    st.subheader("🧠 AI Analysis")

    if result == "Crack":
        st.error("🔴 Crack Detected")
    else:
        st.success("🟢 No Crack Detected")

    st.metric("Confidence", f"{confidence:.2f}%")

    if confidence < 70:
        severity = "🟢 LOW"
    elif confidence < 90:
        severity = "🟡 MEDIUM"
    else:
        severity = "🔴 HIGH"

    st.metric("Risk Level", severity)

    if severity == "🟢 LOW":
        st.info("Recommendation: Monitor periodically.")

    elif severity == "🟡 MEDIUM":
        st.warning("Recommendation: Schedule a structural inspection.")

    else:
        st.error("Recommendation: Immediate structural inspection required.")

    pdf = create_pdf(
        result,
        confidence,
        severity
    )

    st.download_button(
        "📄 Download Inspection Report",
        pdf,
        file_name="Structural_Inspection_Report.pdf",
        mime="application/pdf"
    )
