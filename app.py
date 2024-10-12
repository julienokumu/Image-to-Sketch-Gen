# Import libraries
import cv2 # For image processing
import numpy as np # For numerical operations
import streamlit as st # For building the web app

# Define the function to convert image to sketch
def convert_to_sketch(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_image = cv2.bitwise_not(gray_image)

    # Apply Gaussian Blur to the inverted image
    blurred_image = cv2.GaussianBlur(inverted_image, (55,55), sigmaX = 0, sigmaY = 0)

    # Invert the blurred image
    inverted_blur = cv2.bitwise_not(blurred_image)

    # Create the sketch by dividing the grayscale image by the inverted blur
    sketch = cv2.divide(gray_image, inverted_blur, scale = 256.0)

    return sketch

# Setting up the streamlit interface 
# Title
st.title("Image to Sketch Gen | CodewithJulien")

# File uploader widget to upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Check if file is uploaded
if uploaded_file is not None:
    # Read the uploaded image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Create a container to hold the images
    container = st.container()

    with container:

        # Create two columns to display the images
        col1, col2 = st.columns(2)

        # Display the original image
        with col1:
            st.image(image, caption='Original Image', width=210)

        # Convert the image to sketch
        sketch = convert_to_sketch(image)

        # Display the sketch
        with col2:
            st.image(sketch, caption='Sketch Image', width=210)


    # Download button for the sketch
    st.download_button(
        label="Download Sketch",
        data=cv2.imencode('.png', sketch)[1].tobytes(),
        file_name='sketch.png',
        mime='image/png'
    )
