import streamlit as st
# import numpy as np
# import cv2 as cv2 
from PIL import Image

st.title("GANs Turing Test")
path = "https://github.com/Reuben27/Smart-Farming-Streamlit/raw/main/images/"
images = ["F1.png", "R1.jpg", "F2.png", "R2.jpg", "F3.png", "R3.jpg"]
done1 = False
done2 = False
done3 = False
agree1_a = False
agree2_a = False
agree3_a = False
agree1_b = False
agree2_b = False
agree3_b = False

if(done1 == False):
  col_1, col_2 = st.columns(2)
  if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
  with col_1:
    st.subheader("Image 1")
    st.image(path + images[0])
    agree1_a = st.checkbox('This' , key = 1)
  with col_2:
    st.subheader("Image 2")
    st.image(path + images[1])
    agree1_b = st.checkbox('This', key = 2)

if(agree1_a or agree1_b):
  done1 = True

if(done1 == True and done2 == False):
  col_1, col_2 = st.columns(2)
  if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
  with col_1:
    st.subheader("Image 1")
    st.image(path + images[0])
    agree2_a = st.checkbox('This', key = 3)
  with col_2:
    st.subheader("Image 2")
    st.image(path + images[1])
    agree2_b= st.checkbox('This', key = 4)

if(agree2_a or agree2_b):
  done2 = True

if(done1 == True and done2 == True and done3 == False):
  col_1, col_2 = st.columns(2)
  if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
  with col_1:
    st.subheader("Image 1")
    st.image(path + images[0])
    agree3_a = st.checkbox('This', key = 5)
  with col_2:
    st.subheader("Image 2")
    st.image(path + images[1])
    agree3_b = st.checkbox('This', key = 6)

if(agree3_a or agree3_b):
  done3 = True

if(done1 and done2 and done3):
  st.header("All Donee!")
  st.header(str(int(int(agree1_b) + int(agree2_b) + int(agree3_b))) + "/3")