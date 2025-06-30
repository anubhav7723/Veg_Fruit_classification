import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

MODEL_PATH = r'C:\Users\anubh\OneDrive\Desktop\CNN-Projects\Veg_Fruit_classification\Training\Image_classify.keras'
st.header('Image Classification Model')
model = load_model(MODEL_PATH)

data_cat =  ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

IMAGE_SIZE = 180
img = st.text_input('enter image name','Apple.jpg')

image = tf.keras.utils.load_img(img,target_size= (IMAGE_SIZE,IMAGE_SIZE))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

st.image(img,width = 200)
st.write('Vegitable/Fruit in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))