import cv2
import numpy as np
from datetime import datetime
import streamlit as st
from io import BytesIO
import os

# Use Streamlit's file uploader to allow image selection only once
uploaded_file = st.sidebar.file_uploader("Choose a JPG image", type="jpg")

# Updated BGR grayscale values for mapping as a NumPy array
bgr_values = np.array([
    (198, 198, 198),
    (161, 161, 161),
    (125, 125, 125),
    (100, 100, 100),
    (80, 80, 80),
    (60, 60, 60),
    (40, 40, 40),
    (20, 20, 20),
    (10, 10, 10),
    (5, 5, 5),
    (0, 0, 0)
], dtype=np.uint8)

def main():
    # Only proceed if an image has been uploaded
    if uploaded_file is not None:
        # Load the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Extract the original file name without the extension
        original_name = os.path.splitext(uploaded_file.name)[0]

        # User-defined adjustments
        h = st.sidebar.slider("Hue", min_value=0, max_value=360, value=30, step=1)
        s = st.sidebar.slider("Saturation", min_value=0, max_value=255, value=30, step=10)
        s_fine = st.sidebar.slider("Saturation_fine", min_value=0, max_value=9, value=0, step=1)
        v_inc = st.sidebar.slider("Value Multiplier", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        scale_percent = st.sidebar.slider("Scale Percentage", min_value=10, max_value=200, value=50, step=10)

        # Resize image only if necessary
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        # Convert the image to HSV for bulk hue, saturation, and value adjustments
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 0] = h  # Apply user-selected hue
        hsv_image[..., 1] = (s + s_fine)  # Apply user-selected saturation
        hsv_image[..., 2] = np.clip(hsv_image[..., 2] * v_inc, 0, 255).astype(np.uint8)  # Adjust value intensity

        # Convert adjusted HSV back to BGR
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # Vectorized approach: Reshape the image and calculate distances to the grayscale BGR values
        reshaped_image = adjusted_image.reshape(-1, 3)
        diff = reshaped_image[:, None, :] - bgr_values  # Broadcasting for difference calculation
        distances = np.linalg.norm(diff, axis=2)  # Euclidean distance for closest match
        closest_indices = np.argmin(distances, axis=1)  # Find indices of closest match

        # Map the image pixels to the closest grayscale BGR value in `bgr_values`
        coerced_image = bgr_values[closest_indices].reshape(adjusted_image.shape)

        # Convert final image to RGB for Streamlit display
        result_x = cv2.cvtColor(coerced_image, cv2.COLOR_BGR2RGB)
        st.image(result_x, caption="Final Image", use_column_width=False, output_format="JPEG")

        # Save option with download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{original_name}_updated_{timestamp}.jpg"

        # Convert image to in-memory buffer for download
        is_success, buffer = cv2.imencode(".jpg", coerced_image)
        if is_success:
            st.download_button(
                label="Download Final Image",
                data=BytesIO(buffer),
                file_name=output_filename,
                mime="image/jpeg"
            )
    else:
        st.info("Please upload a JPG image to proceed.")

if __name__ == "__main__":
    main()
