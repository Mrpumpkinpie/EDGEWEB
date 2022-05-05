
#Import:

import sys
import streamlit as st
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from PIL import Image, ImageOps

#Functions:

def order_points(pts):
  rect = np.zeros((4, 2), dtype='float32')

  s = pts.sum(axis=1)

  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  diff = np.diff(pts, axis=1)

  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]

  return rect


def transform_picture(img, pts):
  rect = order_points(pts)
  tl, tr, br, bl = rect

  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))

  dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")

  m = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(img, m, (maxWidth, maxHeight))

  return warped

def get_plate(image):
  img = imutils.resize(image, width=350)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
  plt.imshow(bfilter, cmap='gray')
  edged = cv2.Canny(bfilter, 30, 200) #Edge detection
  boarder = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
  st.write("Edge Detection:")
  st.image(boarder)
  keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(keypoints)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] 

  location = None
  for contour in contours:
      approx = cv2.approxPolyDP(contour, 10, True)
      if len(approx) == 4:
          location = approx
          break
  if location is not None:
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    new_imageshow = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    new_new_image = transform_picture(rgb, location.reshape(4, 2))
    return new_new_image
  else:
    st.title("No plate found")

def read_plate(cropped_image,lan):
  reader = easyocr.Reader([lan])
  result = reader.readtext(cropped_image)
  for item in result:
    st.write(item[-2],end=" ")

  return result

#UI Setup:
CURRENT_THEME = "dark"
IS_DARK_THEME = True
st.set_page_config(page_title='GetPlate', page_icon="○")
st.set_option('deprecation.showfileUploaderEncoding', False) # disable deprecation error
with open("app.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#Customize
st.title("GetPlate")
st.markdown("Creator:")
st.markdown("63110116 Thanin Katanyutapant")
st.markdown("63110117 Intouch Wangtakoondee")
st.markdown("63110118 Tulatorn Prakitjanuruk")
imgupload = st.file_uploader("Please upload the picture", type=["png", "jpg", "jpeg"])
if imgupload is not None:
  image = Image.open(imgupload)
  st.image(image)
  img = Image.open(imgupload).convert('RGB')
  conimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  langu = st.radio(
  "Language of the plate:",
  ('Eng', 'Thai'))
  if langu == 'Eng':
    lan = "en"
    st.write("Selected Langugage: English")
  else:
    lan = "th"
    st.write("Selected Langugage: Thai")
  if st.button('Get Plate NOW!'):
    plate = get_plate(conimg)
    if plate is not None:
      st.write("License Plate & Readings:")
      replate = imutils.resize(plate, width=350)
      st.image(replate)
      reading = read_plate(plate,lan)
