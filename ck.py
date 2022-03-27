from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import streamlit as st
st.title("Healthy Meat for a Healthy You!")
st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)
st.write('Choose your meat type')
from PIL import Image
image = Image.open('C:/Users/ptljk/chick.jpg')

st.image(image,width=200)

if st.button('Chicken'):
     st.write('Start Capturing or Upload Image')
else:
     st.write('Choose an Image')
img_width, img_height = 224, 224
train_data_dir = 'C:/Users/ptljk/chicken_train'
validation_data_dir = 'C:/Users/ptljk/chicken_test'
nb_train_samples =22
nb_validation_samples = 22
epochs = 10
batch_size = 16
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer='rmsprop',
metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
model.save('model_saved.h5')
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import keras
import tensorflow
from keras import backend as K
from keras.models import model_from_config, Sequential
from keras.models import load_model

model = keras.models.load_model('C:/Users/ptljk/model_saved.h5')


MAX_FRAMES = 2
import cv2 as cv2


run = st.button("Click to render server camera")

if run:
    capture = cv2.VideoCapture(0)
    img_display = st.empty()
    for i in range(MAX_FRAMES):
        ret, img = capture.read()
        img_display.image(img, channels='BGR')
    capture.release()
    st.markdown("Render complete")
    cv2.imwrite("uf.1.jpg",img) #save image

ii = st.file_uploader("Choose an image...", type=".jpg")
if ii is not None:
    nn=ii.name
    image = load_img('C:/Users/ptljk/testfile/'+str(nn), target_size=(227, 227))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1,227,227,3)
    label = model.predict(img)
    st.write("Predicted Class (1 - Fresh , 0- Unfresh): ", (label[0][0]))
    st.title("Freshness Report")
    st.write("Chosen Meat: Chicken")
    image = Image.open('C:/Users/ptljk/chick.jpg')
    st.image(image,width=200)
    st.write('Color')
    if ((label[0][0])<.40):
        im = Image.open('C:/Users/ptljk/40p.jpg')
        st.image(im,width=200)
    elif (.41<(label[0][0])<.70):
        im = Image.open('C:/Users/ptljk/70p.jpg')
        st.image(im,width=200)
    elif (.71<(label[0][0])<1):
        im = Image.open('C:/Users/ptljk/90p.jpg')
        st.image(im,width=200)

    st.write('Freshness')
    if ((label[0][0])<.30):
        im = Image.open('C:/Users/ptljk/30p.jpg')
        st.image(im,width=200)
    elif (.30<(label[0][0])<.50):
        im = Image.open('C:/Users/ptljk/40p.jpg')
        st.image(im,width=200)
    elif (.50<(label[0][0])<.70):
        im = Image.open('C:/Users/ptljk/70p.jpg')
        st.image(im,width=200)
    elif (.70<(label[0][0])<.80):
        im = Image.open('C:/Users/ptljk/80p.jpg')
        st.image(im,width=200)
    elif (.80<(label[0][0])<1):
        im = Image.open('C:/Users/ptljk/90p.jpg')
        st.image(im,width=200)
    st.write("Decision ")
    if ((label[0][0])>.50):
        st.write("Fresh Meat")
        im = Image.open('C:/Users/ptljk/wb.jpg')
        st.image(im,width=200)
    elif ((label[0][0])<.50):
        st.write("Old Meat")
        im = Image.open('C:/Users/ptljk/harm.jpg')
        st.image(im,width=200)
        
    st.write('Thanks for choosing our application, Find your complementary Recepie to Healthify You!')
    im = Image.open('C:/Users/ptljk/r1.jpeg')
    st.image(im)
