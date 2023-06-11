from fastai.vision.all import (
    load_learner,
    PILImage,
)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_webrtc import webrtc_streamer
from pathlib import Path
import urllib.request
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import mediapipe as mp
from PIL import Image
import torch
import cv2

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mpPose.Pose()

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)
cap.set(3,224)
cap.set(4,224)
output_placeholder = st.empty()
pred_placeholder = st.empty()
MODEL_URL = "https://huggingface.co/datasets/haruhash/fruit-classification/blob/main/fruit.pkl"
urllib.request.urlretrieve(MODEL_URL, "fruit.pkl")
learn_inf = load_learner('fruit.pkl')

def predict(learn, img):
    pred, pred_idx, pred_prob = learn.predict(img)
    if pred == '00':
        return "00", pred_prob[pred_idx]*100
    elif pred == '01':
        return "01", pred_prob[pred_idx]*100
    elif pred == '02':
        return "02", pred_prob[pred_idx]*100
    elif pred == '03':
        return "03", pred_prob[pred_idx]*100
    elif pred == '04':
        return "04", pred_prob[pred_idx]*100
    elif pred == '05':
        return "05", pred_prob[pred_idx]*100

def segment():
    global output_image,mp_model,bg_image
    # For static images:
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192) # gray
    MASK_COLOR = (255, 255, 255) # white
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # Generate solid color images for showing the output selfie segmentation mask.
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        output_image = np.where(condition, fg_image, bg_image)
        cv2.imwrite('/tmp/selfie_segmentation_output' + str(idx) + '.png', output_image)

    # For webcam input:
    BG_COLOR = (192, 192, 192) # gray
    cap = cv2.VideoCapture(0)
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:
      bg_image = None
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image,(55,55),0)
        if bg_image is None:
          bg_image = np.zeros(image.shape, dtype=np.uint8)
          bg_image[:] = BG_COLOR
          
        output_image = np.where(condition, image, bg_image)

        mp_model = pose.process(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(output_image)
        result = predict(learn_inf, output_image)
        tolerance = 0.01

        if abs(result[1] - 79.0080) < tolerance:
            output_placeholder.error("Please stay on camera.")
        else:
            if result[0] == "00":
                output_placeholder.success(f"Apple {result[1]:.02f}%")
            if result[0] == "01":
                output_placeholder.success(f"Banana {result[1]:.02f}%")
            if result[0] == "02":
                output_placeholder.success(f"Mango {result[1]:.02f}%")
            if result[0] == "03":
                output_placeholder.success(f"Orange {result[1]:.02f}%")
            if result[0] == "04":
                output_placeholder.success(f"Watermelon {result[1]:.02f}%")
            if result[0] == "05":
                output_placeholder.warning(f"Others")

def main():
    st.sidebar.header("Fruit Classification")
    st.title("Fruit Classification")
    while True:
        segment()
        
if __name__ == '__main__':
    main()
