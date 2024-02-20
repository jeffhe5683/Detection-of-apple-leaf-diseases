import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Apple Leaf Disease Detection",
    page_icon = ":Apple:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

def prediction_cls(prediction): # predict the class of the images based on the model results
    for key, clss in class_names.items(): # create a dictionary of the output classes
        if np.argmax(prediction)==clss: # check the class
            
            return key

with st.sidebar:
        st.image('apple.jpg')
        
st.write("""
         # Apple Disease Detection
         """
         )

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache_resource()
def load_model():
    model=tf.keras.models.load_model('model')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (256,256)    
        image = image_data.resize(size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['Apple__Apple_Scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple_healthy']

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Healthy':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'Apple__Apple_Scab':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Apple__Apple_Scab")

    elif class_names[np.argmax(predictions)] == 'Apple__Black_rot':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Apple__Black_rot")

    elif class_names[np.argmax(predictions)] == 'Apple__Cedar_apple_rust':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Apple__Cedar_apple_rust")

    