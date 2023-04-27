
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from streamlit_option_menu import option_menu

import os
print(os.getcwd())
# switch pages with buttons on navigation bar

import subprocess

if st.button('EDA'):
    subprocess.run(["streamlit", "run", "second.py"])
import streamlit as st


#
# # Define the button
# button_html = '''
#     <button onclick="window.location.href='D:\\FILES\\SUBJECTS\\SEM-4\\jfdjfdn\\root\\second.py'">
#         Click me to go to other file
#     </button>
# '''
#
# # Display the button
# st.components.v1.html(button_html)







import subprocess
if not os.path.isfile('model_1.h5'):
    subprocess.run(['curl --output model_1.h5 "https://github.com/AMITESH30/WEB_deploy/blob/main/model.h5?raw=true"'], shell=True)

# Define the CNN model
model = tf.keras.models.load_model("model_1.h5" ,compile=False)

# Define the image classification labels
labels = ['The leaf is diseased cotton leaf', 'The leaf is diseased cotton plant', 'The leaf is fresh cotton leaf', 'The leaf is fresh cotton plant']
#
# with st.sidebar:
#     selected = option_menu(
#         menu_title = None,
#         options = ["Home", "About", "Contact"],
#         icons = ["🏠", "📖", "📞"],
#         menu_icon = "☰",
#         default_index=0,
#         orientation= "horizontal"
#
#     )
# if selected == "Home":
#     st.title("Home")
#
# elif selected == "About":
#     st.title("About")
#
# elif selected == "Contact":
#     st.title("Contact")



# Create a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    return image


# Create the Streamlit app
st.title('Image Classification')
st.write('Upload an image and the model will predict its classification.')

# Create the upload button
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

# Make a prediction if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and make a prediction
    image = preprocess_image(image)
    prediction = model.predict(image)
    label = labels[np.argmax(prediction)]

    # Display the edge-detected image side by side with the original image
    img = np.array(image[0]*255, dtype=np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)
    st.image([img, img_edges], caption=['Original Image', 'Canny Edge Detection'], width=300)

    # display the gray scale of the image side by side with the original image
    img = np.array(image[0]*255, dtype=np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    st.image([img, img_gray], caption=['Original Image', 'Gray Scale'], width=300)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Sobel edge detection to grayscale image
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Normalize edges
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Display images side-by-side
    st.image([img, sobel_edges], caption=['Original Image','Sobel Edges'], width=300)



    # Display the prediction
    st.write('Prediction: ', label)
