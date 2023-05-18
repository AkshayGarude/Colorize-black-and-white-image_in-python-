#!/usr/bin/env python
# coding: utf-8


# import the necessary packages
import numpy as np
import cv2
import streamlit as st
from PIL import Image



def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # load our serialized black and white colorizer model and cluster
    # center points from disk

    prototxt = r"C:\Users\MyVampire\Desktop\Project DL\Colorizer-master\models\models_colorization_deploy_v2.prototxt"
    model = r"C:\Users\MyVampire\Desktop\Project DL\Colorizer-master\models\colorization_release_v2.caffemodel"
    points = r"C:\Users\MyVampire\Desktop\Project DL\Colorizer-master\models\pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab") # The layer with this ID is responsible for predicting the 'a' and 'b' channels of the Lab color space.
    conv8 = net.getLayerId("conv8_313_rh") #retrieves the layer ID of the layer named "conv8_313_rh" from the network (net).
    pts = pts.transpose().reshape(2, 313, 1, 1) #This line performs some transformations on the pts array. The pts array represents the cluster center points used for colorization
    net.getLayer(class8).blobs = [pts.astype("float32")] # assigns the modified pts array as the weights of the layer with ID class8,
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")] #This line assigns a constant value to the weights of the layer with ID conv8. It creates an array filled with the value 2.606, with dimensions (1, 313).


    # scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
    scaled = img.astype("float32") / 255.0 #(img) to float32 and scales the pixel values by dividing them by 255.0. 
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)# RGB image (scaled) to the LAB color space using the OpenCV function 


    # resize the Lab image to 224x224 (the dimensions that colorization network accepts), split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    # Return the colorized images
    return colorized

##########################################################################################################
    
st.write("""
          # Colorize your Black and white image
          """
          )

st.write("This is an webapp to turn Colorize your B&W images.")
st.write("Hii Wlecome to the Webpage ")

file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)
    
    st.text("Your original image")
    st.image(image, use_column_width=True)
    
    st.text("Your colorized image")
    color = colorizer(img)
    
    st.image(color, use_column_width=True)
    
    print("done!")





