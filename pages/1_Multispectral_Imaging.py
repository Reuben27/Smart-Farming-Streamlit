import streamlit as st
import numpy as np
import cv2 as cv2 
from PIL import Image
from ultralytics import YOLO
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt

# Load a model
model = YOLO(".\\pages\\best.pt") 
# Paths to input band files
red_band_path = ".\\images\\red_band.TIF"
green_band_path = ".\\images\\green_band.TIF"
nir_band_path = ".\\images\\nir_band.TIF"

def object_detection(image):
  results = model([image])  # return a list of Results objects

  i = 0
  # Process results list
  for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    name = "result_" + str(i) + ".jpg"
    result.save(filename=name)  # save to disk
    modified_image = Image.open(name)
    modified_img_array = np.array(modified_image)
    return modified_img_array

def reconstruction():
  # Open each band
  with rasterio.open(red_band_path) as red_src:
    red_band = red_src.read(1)

  with rasterio.open(green_band_path) as green_src:
    green_band = green_src.read(1)

  with rasterio.open(nir_band_path) as nir_src:
    nir_band = nir_src.read(1)

  # Create a new 3-band image
  combined_image = np.zeros((red_band.shape[0], red_band.shape[1], 3), dtype=np.uint8)

  # Normalize bands and assign to RGB channels
  combined_image[:, :, 0] = np.uint8((red_band / np.max(red_band)) * 255)
  combined_image[:, :, 1] = np.uint8((green_band / np.max(green_band)) * 255)
  combined_image[:, :, 2] = np.uint8((nir_band / np.max(nir_band)) * 255)

  # image = Image.open(combined_image)
  # # image = uploaded_file.read()
  # new_new = np.array(image)
  st.image(combined_image, clamp=True, channels='BGR')

  # Display the combined image
  # plt.imshow(combined_image)
  # plt.axis('off')
  # plt.title('Combined Image')
  # plt.show()


st.title("Multispectral Imagery")
uploaded_files = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "JPG"], accept_multiple_files=True)
print("Hey")

Object_Detection = st.button("Do Object Detection")

Reconstruction = st.button("Do reconstruction")

if Reconstruction:
  reconstruction()

if Object_Detection:
  if uploaded_files is not None:
    for uploaded_file in uploaded_files:
      print(uploaded_file)
      image = Image.open(uploaded_file)
      # image = uploaded_file.read()
      original_img_array = np.array(image)
      # modified_img_array = np.array(image)

      st.subheader("Original Image")
      # st.image(original_img_array, clamp=True, channels='BGR')
      modified_img_array = object_detection(image)

      st.subheader("Output Image")
      # st.image(modified_img_array, clamp=True, channels='BGR')

  