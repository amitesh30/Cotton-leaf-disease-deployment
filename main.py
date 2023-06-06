
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from streamlit_option_menu import option_menu

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")
# switch pages with buttons on navigation bar

# config = st.secrets["config"]


tab1, tab2, tab3 = st.tabs(["About us", "Our Project", "MODEL"])

with tab1:
    img = Image.open(r"./Images/Untitled design.png")
    st.image(img,use_column_width=True)



with tab2:

    # ----header section----
    with st.container():
        st.header("INTRODUCTION")
        st.title("DATASET-Cotton Disease Detection")
        st.markdown(" - Cotton is an important crop worldwide and leaf diseases can have a significant impact on yield and quality. Visual inspection of leaves is time-consuming and subject to human error, which highlights the need for automated systems that can accurately classify cotton plant leaf images.")
        st.markdown(" - Machine learning techniques have shown promise in image analysis and classification tasks, and can potentially assist in the detection and diagnosis of cotton leaf diseases.")
        st.markdown(" - The motivation for this project is to develop a machine learning model that can accurately classify cotton plant leaf images into healthy and diseased categories, which can potentially improve the accuracy of cotton plant disease detection and diagnosis.")
        st.markdown(" - The background of the project involves exploring various image processing techniques, feature extraction methods, and classification models to identify the best approach for cotton plant leaf image classification ")

    # ----first section----

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.title("OBJECTIVES & AIMS: ",)
            st.write("##")
            st.subheader("Problem Statement")
            st.write(
                """  
             The intent is to develop a machine learning and deep learning model that can accurately classify cotton plant leaf images as healthy or diseased, potentially aiding in early detection and prevention of crop losses.  
                """)
            st.write("##")

            st.subheader("OBJECTIVES & AIMS: ")
            st.markdown(" - To investigate the effectiveness of supervised machine learning for cotton plant leaf disease detection and classification using image analysis. The intent is to develop a machine learning and deep learning model that can accurately classify cotton plant leaf images as healthy or diseased, potentially aiding in early detection and prevention of crop losses. ")
            st.markdown(" - To develop a model that can accurately classify cotton plant leaf images into healthy and diseased categories, using image analysis and classification techniques. The project also aims to compare and identify the best approach for cotton plant leaf image classification, which can potentially assist in the prevention of crop losses and improve the yield and quality of cotton crops. ")



        with right_column:
            img = Image.open(r"./Images/output1.png")
            st.write("##")
            st.write("##")
            st.write("##")

            # display image in streamlit app
            st.image(img, caption="Your local image", width=500)

    # ----second section----
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.title("Dataset Description ")
            st.write("##")
            st.write(
                """ The Cotton Disease Dataset is a collection of images of diseased and healthy cotton plants. The dataset is segregated into 4 parts-diseased cotton leaf , diseased cotton plant, fresh cotton leaf, fresh cotton plant. The dataset is a valuable resource for researchers and practitioners in the field of plant pathology, as it provides a large and diverse set of images for the development and evaluation of disease diagnosis and classification systems.""")
            # st.write("more text")

        with right_column:
            img = Image.open(r"./Images/output2.png")

            # display image in streamlit app
            st.image(img, caption="Dataset Distribution ", width=500)


    # ----third section----



    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.title(" Image Augmentation")
            st.write("##")
            st.markdown("- In the case of a leaf image, flipping the image horizontally can create a mirror image, which can be useful in training models to recognize the leaf from different angles.")
            st.markdown("- Cropping the image can remove unwanted background or foreground, and focus on the leaf's key features. ")
            st.markdown("- Blurring the image can reduce the effect of noise and make the leaf features more prominent. "   )
            st.markdown("- Saturating the image can increase the color intensity of the leaf, while contrasting the image can enhance the edges of the leaf's features.")
            st.markdown("- Applying gamma correction can adjust the brightness and contrast of the image and make it more visually appealing.By combining these image augmentation techniques, we can create a more extensive and diverse dataset that can help improve the accuracy of machine learning models used to recognize leaves.")


        with right_column:
            img = Image.open(r"./Images/dataaug.jpg")
            st.write("##")
            st.write("##")
            st.write("##")
            # display image in streamlit app
            st.image(img, width=900)
            # ----fourth section----
    with st.container():
        st.write("---")
        st.title("Image Processing Techniques")
        left_column,middle_column, right_column = st.columns(3)

        with left_column:
            st.markdown("- thresholded-> binary image")
            img = Image.open(r"./Images/1st.jpg")
            st.image(img, width=400)
        with middle_column:
            st.markdown("- gray scale->thresholding ")
            img = Image.open(r"./Images/2nd.jpg")
            st.image(img, width=400)
        with right_column:
            st.markdown("- hsv->thresholding")
            img = Image.open(r"./Images/3rd.jpg")
            st.image(img, width=400)

# ----fifth section----
    with st.container():
        st.write("---")
        st.title("Machine Learning Models results")
        left_column, right_column = st.columns(2)

        with left_column:
            st.subheader("ALGORITHMS USED:")
            st.markdown("- KNN algorithm for cotton plant leaf disease classification uses a non-parametric approach to make predictions based on the K nearest neighbors of a sample.")
            st.markdown("- SVM algorithm in cotton plant leaf disease classification separates data points using hyperplanes and maximizes the margin between different classes.")
            st.markdown("- Decision tree algorithm for cotton plant leaf disease classification makes decisions by recursively splitting the data into smaller subsets based on the most informative feature.")
            st.markdown("- Decision tree + AdaBoost algorithm improves the accuracy of decision trees by weighting the importance of misclassified samples.")
            st.markdown("- Random forest algorithm for cotton plant leaf disease classification combines multiple decision trees to reduce overfitting and improve generalization performance.")
            st.markdown("- ResNet, a deep learning algorithm used in cotton plant leaf disease classification, uses residual connections to overcome the problem of vanishing gradients and improve the training of very deep neural networks.")

        with right_column:
            img = Image.open(r"./Images/plot 4.jpg")
            st.image(img,caption="Machine Learning Models", use_column_width=True)



            # ----fifth section----
    with st.container():
        st.write("---")
        st.title("Project Poster")
        img = Image.open(r"./Images/Slide1.PNG")
        st.image(img, use_column_width=True)
















with tab3:
    st.write("MODEL")

    # Define the CNN model
    model = tf.keras.models.load_model(r"resnet152V2_model.h5")

    # Define the image classification labels
    labels = ['The leaf is diseased cotton leaf', 'The leaf is diseased cotton plant', 'The leaf is fresh cotton leaf',
              'The leaf is fresh cotton plant']


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
        st.image(image, caption='Uploaded Image', width =900)

        # Preprocess the image and make a prediction
        image = preprocess_image(image)
        prediction = model.predict(image)
        st.write(np.argmax(prediction))
        label = labels[np.argmax(prediction)]

        # Display the edge-detected image side by side with the original image
        img = np.array(image[0] * 255, dtype=np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 200)
        st.image([img, img_edges], caption=['Original Image', 'Canny Edge Detection'], width=300)

        # display the gray scale of the image side by side with the original image
        img = np.array(image[0] * 255, dtype=np.uint8)
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
        st.image([img, sobel_edges], caption=['Original Image', 'Sobel Edges'], width=300)

        # Display the prediction
        # st.write('Prediction: ', label)
        st.write('<span style="font-size:50px">','Prediction:', '</span>','<span style="font-size:35px">' + label + '</span>', unsafe_allow_html=True)

