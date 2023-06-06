# Streamlit-leaf-detection-gui

# Cotton Leaf Disease Detection with Streamlit

This is a project to detect cotton leaf diseases using machine learning and Streamlit. The app takes an image of a cotton leaf as input and predicts whether the leaf is healthy or diseased, and if it is diseased, which disease it is most likely to be.

## Setup

To run the app, you will need to install the required dependencies. You can do this using pip and the provided requirements.txt file:

```
pip install -r requirements.txt
```

## Usage

To start the app, run the following command:

```
streamlit run app.py
```

This will start a local Streamlit server and open the app in your default browser. You can then upload an image of a cotton leaf using the file uploader, and the app will display the prediction.

## Model

The model used for this project is a convolutional neural network (CNN) that was trained on a dataset of cotton leaf images. The dataset contains images of healthy cotton leaves as well as leaves with four common diseases: Alternaria leaf spot, angular leaf spot, gray mold, and powdery mildew.

The model was trained using TensorFlow and Keras, and achieved an accuracy of over 90% on the validation set.

## Website

You can try out the cotton leaf disease detection app by visiting our website:

- Link: [https://amitesh30-cotton-leaf-disease-deployment-main-kcsx2k.streamlit.app/](https://amitesh30-cotton-leaf-disease-deployment-main-kcsx2k.streamlit.app/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
