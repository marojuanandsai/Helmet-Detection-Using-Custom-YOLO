import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load trained YOLO model
model = YOLO("best.pt")

# Page configuration
st.set_page_config(
    page_title="Helmet Detection",
    page_icon="🪖",
    layout="centered"
)

st.title("🪖 Helmet Detection System")
st.write("Upload an image to detect helmets and motorcycles.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, caption="Input Image", use_container_width=True)

    # Detection
    if st.button("🔍 Detect"):

        with st.spinner("Detecting..."):

            results = model.predict(
                source=image,
                conf=0.25
            )

        # Display result
        result_image = results[0].plot()

        st.subheader("Detection Result")
        st.image(
            result_image,
            caption="Detected Objects",
            use_container_width=True
        )

        st.success("Detection completed successfully! ✅")