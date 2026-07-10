
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("crack_detector.keras")

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

    st.subheader("Prediction")
    st.success(result)

    st.subheader("Confidence")
    st.write(f"{confidence:.2f}%")

    if confidence < 70:
        severity = "🟢 LOW"
    elif confidence < 90:
        severity = "🟡 MEDIUM"
    else:
        severity = "🔴 HIGH"

    st.subheader("Severity")
    st.write(severity)

    if severity=="🟢 LOW":
        st.info("Recommendation: Monitor periodically.")
    elif severity=="🟡 MEDIUM":
        st.warning("Recommendation: Inspect within 30 days.")
    else:
        st.error("Recommendation: Immediate structural inspection required.")
