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
        n = st.sidebar.slider("Number of values (N)", min_value=1, max_value=256, value=25, step=1)
        p = st.sidebar.slider("Clusters for selected values (p)", min_value=1, max_value=100, value=5, step=1)
        k = st.sidebar.slider("Number of clusters (k)", min_value=1, max_value=100, value=10, step=1)
        scale_percent = st.sidebar.slider("Scale Percentage", min_value=10, max_value=200, value=100, step=10)
        sort_order = st.sidebar.radio("Sort Pixel Values", ("Increasing", "Decreasing"))

        # Additional sliders for the new selection logic
        start_value = st.sidebar.slider("Start Value", min_value=0, max_value=255, value=50, step=1)
        skip_d = st.sidebar.slider("Skip (d)", min_value=1, max_value=50, value=5, step=1)

        # Option to turn borders on or off
        draw_borders = st.sidebar.checkbox("Draw Black Borders Around Groups", value=True)

        # Sliders to reduce clutter
        min_contour_area = st.sidebar.slider("Minimum Contour Area", min_value=100, max_value=5000, value=100, step=100)
        border_thickness = st.sidebar.slider("Border Thickness", min_value=1, max_value=10, value=2, step=1)

        p = min(p, n)

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

        # Adjust top N value selection logic
        unique_pixels, counts = np.unique(pixel_values, return_counts=True)
        sorted_indices = np.argsort(-counts if sort_order == "Decreasing" else counts)

        # Select values starting from start_value and skipping every 'd' values
        start_index = np.searchsorted(unique_pixels[sorted_indices], start_value, side="left")
        selected_indices = sorted_indices[start_index::skip_d][:n]
        selected_values = unique_pixels[selected_indices]

        # Ensure 'selected_values' has exactly 'n' elements
        if len(selected_values) < n:
            st.warning("Not enough values in range to select N. Adjust Start Value or Skip (d).")
            selected_values = selected_values[:n]

        # KMeans clustering on selected values
        selected_kmeans = KMeans(n_clusters=p, random_state=42, n_init='auto')
        selected_kmeans.fit(selected_values.reshape(-1, 1))
        selected_centroids = np.sort(selected_kmeans.cluster_centers_.flatten())

        # Combine centroids and additional values
        final_values = np.sort(np.concatenate((centroids, [0, 255], selected_centroids)))
        print(final_values)

        # Vectorized mapping to closest values
        diff_matrix = np.abs(gray_image[:, :, None] - final_values)
        coerced_image = final_values[np.argmin(diff_matrix, axis=2)]

        # Convert coerced image to BGR for display and output
        coerced_bgr_image = cv2.cvtColor(coerced_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Draw black borders if the checkbox is checked
        if draw_borders:
            for value in final_values:
                # Find where the current final value is
                mask = coerced_image == value
                # Get the contours of these regions
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours by area based on the user-defined minimum contour area
                contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

                # Draw black borders around the contours with adjustable border thickness
                for contour in contours:
                    cv2.drawContours(coerced_bgr_image, [contour], -1, (0, 0, 0), border_thickness)

        # Convert BGR to RGB for Streamlit display
        result_x = cv2.cvtColor(coerced_bgr_image, cv2.COLOR_BGR2RGB)
        st.image(result_x, caption="Final Image", use_column_width=False, output_format="JPEG")

        # Save option with download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{original_name}_a{start_value}_d{skip_d}_n{n}_p{p}_k{k}_order{sort_order}_a{min_contour_area}_t{border_thickness}_{timestamp}.jpg"
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
