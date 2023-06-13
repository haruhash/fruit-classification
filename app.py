from fastai.vision.all import (
    load_learner,
    PILImage,
)
import urllib.request
import glob
import streamlit as st
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
    
imdb_url = 'https://www.kasetprice.com/ราคา/มะม่วง/วันนี้'
imdb_response = requests.get(imdb_url)
imdb_soup = BeautifulSoup(imdb_response.text, 'html.parser')
group = imdb_soup.find_all('div', {'class': 'price-list-cost'})

MODEL_URL = "https://huggingface.co/spaces/haruhash/fruit-classification/resolve/main/fruit.pkl"
urllib.request.urlretrieve(MODEL_URL, "fruit.pkl")
learn_inf = load_learner('fruit.pkl')

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

def callback():
    st.session_state.button_clicked = True

def get_image_from_upload():
    uploaded_file = st.file_uploader("Upload Picture",type=['png','jpeg', 'jpg'])
    if uploaded_file is not None:
        st.image(PILImage.create((uploaded_file)))
        return PILImage.create((uploaded_file))
    return None

def take_a_picture():
    picture = st.camera_input("Take a picture")
    if picture:
        return PILImage.create((picture)) 
    return None
        
def predictMango(learn, img):
    pred, pred_idx, pred_prob = learn.predict(img)
    if pred == '00':
        mnType = "Extra"
    elif pred == '01':
        mnType = "Grade A"
    elif pred == '02':
        mnType = "Grade B"
    else:
        mnType = "r"
    if mnType == "r":
        st.error(f"This is a rotten mango with the probability of {pred_prob[pred_idx]*100:.02f}%")
    else:
        st.success(f"This is {mnType} mango with the probability of {pred_prob[pred_idx]*100:.02f}%")
    return int(pred)

def predict(learn, img):
    pred, pred_idx, pred_prob = learn.predict(img)
    if pred == '00':
        frType = "Apple"
    elif pred == '01':
        frType = "Banana"
    elif pred == '02':
        frType = "Mango"
        MODEL_URL = "https://huggingface.co/spaces/haruhash/fruit-classification/resolve/main/mangograde.pkl"
        urllib.request.urlretrieve(MODEL_URL, "mangograde.pkl")
        learn_inf = load_learner('mangograde.pkl')
        tp = predictMango(learn_inf, img)
    elif pred=='03':
        frType = "Orange"
    elif pred=='04':
        frType = "Watermelon"
    else:
        st.error(f"Others with the probability of {pred_prob[pred_idx]*100:.02f}%")
        return None
    st.success(f"This is {frType} with the probability of {pred_prob[pred_idx]*100:.02f}%")
    if pred == '02' and tp != 3:
        mList = ["เขียวเสวย", "แก้ว", "โชคอนันต์(สุก)", "โชคอนันต์(ดิบ)", "น้ำดอกไม้(แก่)", "น้ำดอกไม้(สุก)", "น้ำดอกไม้(อ่อน)", "ฟ้าลั่น", "มันเดือนเก้า", "อกร่อง(ดิบ)", "อกร่อง(สุก)", "อื่นๆ"]
        seldat = st.selectbox('Type of mango', mList)
        confirm = st.button('Confirm')
        if confirm:
            i = 0
            for mStr in mList:
                if seldat == mStr:
                    nStr = group[i].string.strip()
                    if nStr == '-':
                        st.error(f"No price data")
                    else:
                        st.success(f"Price : {((4 - tp) / 2 * np.float_(nStr)):.02f}/kg")
                    break
                i = i + 1

def main():
    st.title('Mango Classification Model')
    datasrc = st.sidebar.selectbox("Type of picture input", ["Upload", "Camera"])
    if datasrc == "Upload": 
        image = get_image_from_upload()
    else:
        image = take_a_picture()
    result = st.button(label = 'Classify', on_click = callback)
    if result or st.session_state.button_clicked:
        predict(learn_inf, image)        
    
if __name__ == '__main__':
    main()
