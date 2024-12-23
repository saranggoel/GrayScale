import cv2
import numpy as np
from datetime import datetime
import streamlit as st
from io import BytesIO
import os
from sklearn.cluster import KMeans

# Use Streamlit's file uploader to allow image selection only once
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

def main():
    if uploaded_file is not None:
        # Load the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        original_name = os.path.splitext(uploaded_file.name)[0]

        # User-defined adjustments
        k = st.sidebar.slider("Number of clusters (k)", min_value=1, max_value=100, value=10, step=1)
        scale_percent = st.sidebar.slider("Scale Percentage", min_value=10, max_value=200, value=100, step=10)

        # Resize image
        new_dims = (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100))
        image = cv2.resize(image, new_dims, interpolation=cv2.INTER_AREA)

        # Convert to grayscale and flatten
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixel_values = gray_image.flatten()

        # KMeans clustering for pixel values
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(pixel_values.reshape(-1, 1))
        centroids = np.sort(kmeans.cluster_centers_.flatten())

        # Combine centroids with additional values
        final_values = np.sort(np.concatenate((centroids, [0, 255])))

        # Vectorized mapping to closest values
        diff_matrix = np.abs(gray_image[:, :, None] - final_values)
        coerced_image = final_values[np.argmin(diff_matrix, axis=2)]

        # Convert to BGR for display and output
        coerced_bgr_image = cv2.cvtColor(coerced_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        result_x = cv2.cvtColor(coerced_bgr_image, cv2.COLOR_BGR2RGB)
        st.image(result_x, caption="Final Image", use_column_width=False, output_format="JPEG")

        # Save option with download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{original_name}_k{k}_{timestamp}.jpg"
        is_success, buffer = cv2.imencode(".jpg", coerced_bgr_image)
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
