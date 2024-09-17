import streamlit as st  #for creating app interface
import tensorflow as tf  #for using dl functionalitites
import numpy as np  #numerical opr
from PIL import Image #importing image module from PIL lib. for image processing

#function to load the pre-trained model and cache it for opt performance
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('flower_model_trained.hdf5')
    return model

#function to predict the class of the input image using the loaded model
def predict_class(image, model):
    image = tf.cast(image, tf.float32) #converting image to float32 datatype
    image = tf.image.resize(image, [180,180])  #resizing the input image to match the models input shape
    image = np.expand_dims(image, axis=0)  #adding extra dimension to match the models input requirements
    prediction = model.predict(image)
    return prediction

model = load_model() #calling the func to load the pretrained model
st.title("Flower Classifier") #title of the streamlit app

#creating a file uploader component for uploading images of type png or jpg
file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])

if file is None:
    st.text("Waiting for upload...")

else:
    slot = st.empty()
    #indicating the inference process is ongoing
    slot.text("Running inference...")

    test_image = Image.open(file)

    st.image(test_image, caption="Input Image", width=400)  #displaying the img uplaoded by the user with suitable caption and given width'

    pred = predict_class(np.asarray(test_image), model)

    class_names = ['daisy', 'dandelion', 'tulip', 'sunflower', 'rose'] #defining class names for diff flower types

    result = class_names[np.argmax(pred)] #determine the predicted class by selecting the one w highest probability(we will get enu index)

    output = 'The image is a ' + result

    slot.text('Done')

    st.success(output)

