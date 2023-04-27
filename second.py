import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from streamlit_option_menu import option_menu
# from streamlet_lottie import st_lottie


st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")
#
# def load_lottieurl(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_coding = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_8ql1aq7w.json")
# Use local CSS
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ----header section----
with st.container():
    st.header("image classification")
    st.title("group project")
    st.write("sem4 group project adding random text to see if it works")
    # st

# ----first section----

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("problem statement")
        st.write("##")
        st.write(
            """  
         The intent is to develop a machine learning and deep learning model that can accurately classify cotton plant leaf images as healthy or diseased, potentially aiding in early detection and prevention of crop losses.  
            """)
        st.write("more text")

    with right_column:
       
        img = Image.open(r"D:\FILES\SUBJECTS\SEM-4\TUTORIAL\output1.png")

        # display image in streamlit app
        st.image(img, caption="Your local image", width=500)

        

#arjuns code 1


with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header(" Image Augmentation")
        st.write("##")
        st.write(""" In the case of a leaf image, flipping the image horizontally can create a mirror image, which can be useful in training models to recognize the leaf from different angles. Cropping the image can remove unwanted background or foreground, and focus on the leaf's key features. Blurring the image can reduce the effect of noise and make the leaf features more prominent. Saturating the image can increase the color intensity of the leaf, while contrasting the image can enhance the edges of the leaf's features. Applying gamma correction can adjust the brightness and contrast of the image and make it more visually appealing.By combining these image augmentation techniques, we can create a more extensive and diverse dataset that can help improve the accuracy of machine learning models used to recognize leaves.""")

        st.write("more text")

    with right_column:
       
        img = Image.open(r"C:\Users\amite\Downloads\WhatsApp Image 2023-04-27 at 12.52.54.jpg")

        # display image in streamlit app
        st.image(img, caption="Your local image", width=500)


#arjuns code 2
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Dataset Description ")
        st.write("##")
        st.write(
            """ In the case of a leaf image, flipping the image horizontally can create a mirror image, which can be useful in training models to recognize the leaf from different angles. Cropping the image can remove unwanted background or foreground, and focus on the leaf's key features. Blurring the image can reduce the effect of noise and make the leaf features more prominent. Saturating the image can increase the color intensity of the leaf, while contrasting the image can enhance the edges of the leaf's features. Applying gamma correction can adjust the brightness and contrast of the image and make it more visually appealing.By combining these image augmentation techniques, we can create a more extensive and diverse dataset that can help improve the accuracy of machine learning models used to recognize leaves.""")

        st.write("more text")

    with right_column:
        img = Image.open(r"D:\FILES\SUBJECTS\SEM-4\TUTORIAL\output2.png")

        # display image in streamlit app
        st.image(img, caption="Your local image", use_column_width=True)

