import streamlit as st
import numpy as np
import cv2 as cv2 
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

esrgn_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(esrgn_path)

###### Image Augmentation Options ######
def rotation(img_array, degree):   
  rows, cols, temp = img_array.shape
  M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
  dst = cv2.warpAffine(img_array,M,(cols,rows))
  st.image(dst)

def translation(img_array,x,y):   
  rows, cols, temp = img_array.shape
  M = np.float32([[1,0,x*img_array.shape[0]],[0,1,-y*img_array.shape[1]]])
  dst = cv2.warpAffine(img_array,M,(cols,rows))
  st.image(dst)

def blurring(img_array,BlurAmount):  
  rows, cols, temp = img_array.shape
  Blurred = cv2.blur(img_array,(BlurAmount,BlurAmount))
  st.image(Blurred)

def brighter(img,factor):
  if(factor <= 100):
    factor = factor - 100
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  v = cv2.add(v,factor)
  v[v > 255] = 255
  v[v < 0] = 0
  final_hsv = cv2.merge((h, s, v))
  img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
  #enhancer = ImageEnhance.Brightness(image)
  #im_output = enhancer.enhance(factor)
  st.image(img)

def flip(img_array):   
  rows, cols, temp = img_array.shape
  flip=cv2.flip(img_array,1)
  st.image(flip)

def preprocessing(img_array):
	imageSize = (tf.convert_to_tensor(img_array.shape[:-1]) // 4) * 4
	cropped_image = tf.image.crop_to_bounding_box(img_array, 0, 0, imageSize[0], imageSize[1])
	preprocessed_image = tf.cast(cropped_image, tf.float32)
	return tf.expand_dims(preprocessed_image, 0)

def srgan(img_array):
  preprocessed_image = preprocessing(img_array) # Preprocess the LR Image
  new_image = model(preprocessed_image) # Runs the model
  return np.array(tf.squeeze(new_image) / 255.0)

def stitch(images):
  imgs = []
  # Set smaller desired width and height for resizing
  new_width = 400  # Adjust this as needed
  new_height = 300  # Adjust this as needed
  for img_array in images:
    img = cv2.resize(img_array, (new_width, new_height))
    imgs.append(img)

  stitcher = cv2.createStitcher() if hasattr(cv2, 'createStitcher') else cv2.Stitcher_create()
  (status, output) = stitcher.stitch(imgs)
  if status == cv2.Stitcher_OK:
    st.image(output)
  else:
    st.write("Stitching ain't successful.")

def all_changes(img_array, degree, x, y, brighter, BlurAmount, flip, upscale):
  rows,cols, temp = img_array.shape
  #Rotation
  if (degree != None):
    rows, cols, temp = img_array.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    img_array = cv2.warpAffine(img_array,M,(cols,rows))

  #Translation
  if (x != None and y != None):
    rows, cols, temp = img_array.shape
    M = np.float32([[1,0,x*img_array.shape[0]],[0,1,-y*img_array.shape[1]]])
    img_array = cv2.warpAffine(img_array,M,(cols,rows))

  #Brightness
  if (brighter != None):
    if(brighter <= 100):
      brighter = brighter - 100
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,brighter)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img_array = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

  #Blurred
  if (BlurAmount != None):
    img_array = cv2.blur(img_array,(BlurAmount,BlurAmount))

  #Flip
  if (flip == "Yes"):
    rows,cols, temp = img_array.shape
    img_array=cv2.flip(img_array,1)

  #Upscale
  if (upscale == "Yes"):
    img_array=srgan(img_array)

  st.image(img_array, clamp=True, channels='BGR')

rotating_degree = None 
x_direction = None 
y_direction = None
blur = None 

st.title("Smart Farming Using Aerial Imagery")
st.text("Our project explores the transformative potential of smart farming, driven by drone-based aerial imagery, in the context of cotton crop cultivation. Smart farming uses modern information and communication technologies to enhance crop quality and quantity while optimizing labour. By leveraging data from drone imagery, this approach offers real-time insights, enabling precise interventions to monitor crop health, identify high-growth areas, and combat pests and weeds.")
st.text("Keywords: Smart Farming, Image Stitching, Image Superresolution, Cotton Crops")
st.header("Image Augmentation Options")

uploaded_files = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
stitch_or_not = 'No'

if uploaded_files is not None:
  if(len(uploaded_files) > 1):
    st.subheader("Stitch Images")
    stitch_or_not = st.radio("Do you want to stitch the images together?", ('Yes', 'No'), 1)

  st.subheader("Upscale Images")
  upscale_or_not = st.radio("Do you want to upscale the images?", ('Yes', 'No'), 1)

  st.subheader("Rotate Images")
  rotating_degree = st.slider('Select the degree upto which you want to rotate the image', -20, 20, step = 1, value = 0)

  st.subheader("Flip Images")
  flip_or_not = st.radio("Do you want to flip the images?", ('Yes', 'No'), 1)

  st.subheader("Brighten Images")
  brightness_factor = st.slider("Select the brightness factor", 0.75, 1.25, step = 0.05, value = 1.00)

  st.subheader("Translate Images")
  col_x, col_y = st.columns(2)
  with col_x:
    x_direction = st.slider("Select the translation along x direction", -0.2, 0.2, step = 0.02, value = 0.0)
  with col_y:
    y_direction = st.slider("Select the translation along y direction", -0.2, 0.2, step = 0.02, value = 0.0)

  Augment_Image = st.button("Augment Image")

  if Augment_Image:
    if(stitch_or_not == 'Yes'):
      st.subheader("Stitched Images")
      images = []
      for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        original_img_array = np.array(image)
        images.append(original_img_array)
      stitch(images)
    
    for uploaded_file in uploaded_files:
      image = Image.open(uploaded_file)
      original_img_array = np.array(image)
      modified_img_array = np.array(image)
      col_1, col_2 = st.columns(2)
      with col_1:
        st.subheader("Original Image")
        st.image(original_img_array, clamp=True, channels='BGR')
      with col_2:
        st.subheader("Augmentated Image")
        all_changes(modified_img_array, rotating_degree, x_direction, y_direction, int(brightness_factor*100), blur, flip_or_not, upscale_or_not)