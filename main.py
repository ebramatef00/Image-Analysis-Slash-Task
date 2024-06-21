import streamlit as st
import cv2
import numpy as np
from image_Analysis import analyse_image

    
def main():
    st.title("Welcome to analysing image Program")

    uploaded_image=st.file_uploader("Upload your Image",type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
          st.image(uploaded_image)
          file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
          opencv_image = cv2.imdecode(file_bytes, 1)
          component_names = analyse_image(opencv_image)
          if st.button("Analyse Image"):
                    # Display the results
                    st.write("Detected components:")
                    for component in component_names:
                        st.write(f"- {component}")



if __name__ == "__main__":
    main()