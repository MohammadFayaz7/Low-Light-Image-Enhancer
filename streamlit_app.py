import streamlit as st
from lowlight_test import lowlight
from PIL import Image

# Streamlit page configuration
st.set_page_config(page_title="Low-Light Image Enhancer", page_icon="🌙", layout="wide")

# Header and description
st.title("✨ Low-Light Image Enhancement Using CNN")
st.markdown("""
This app enhances low-light images using a pre-trained **Zero-DCE** model. 
Upload your low-light image, and the model will process it to improve brightness and detail.
""")

# Create a sidebar with some options
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. **Upload an Image**: Select a low-light image to enhance.
2. **Enhance the Image**: Press the **Enhance Image** button to process the image.
3. **Download**: Once the image is processed, you can download the enhanced version.
""")

# Upload the image
uploaded_file = st.file_uploader("Choose a low-light image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the original uploaded image
    image = Image.open(uploaded_file)

    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)

    # Display the original image in the first column
    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", width=400)

    # Enhance the image
    if st.button("Enhance Image"):
        with st.spinner("Enhancing image... Please wait."):
            enhanced_image = lowlight(uploaded_file)  # Backend function call
            
            # Display the enhanced image in the second column
            with col2:
                st.subheader("Enhanced Image")
                st.image(enhanced_image, caption="Enhanced Image", width=400)

                # Save the enhanced image and provide a download link below the enhanced image
                enhanced_image_path = "enhanced_image.jpg"
                enhanced_image.save(enhanced_image_path)
                with open(enhanced_image_path, "rb") as file:
                    st.download_button(
                        label="Download Enhanced Image",
                        data=file,
                        file_name=enhanced_image_path,
                        mime="image/jpeg"
                    )
    else:
        st.warning("Click on the 'Enhance Image' button to process your image.")
else:
    st.info("Please upload an image to get started.")
    