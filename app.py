from fastai.vision.all import (
    load_learner,
    PILImage,
)
import urllib.request
import glob
import streamlit as st
    
MODEL_URL = "https://huggingface.co/spaces/haruhash/fruit-classification/resolve/main/fruit.pkl"
urllib.request.urlretrieve(MODEL_URL, "fruit.pkl")
learn_inf = load_learner('fruit.pkl')

def get_image_from_upload():
    uploaded_file = st.file_uploader("Upload Fruit Pictures Files",type=['png','jpeg', 'jpg'])
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
        mnType = "Class 1"
    elif pred == '02':
        mnType = "Class 2"
    else:
        mnType = "Class 3"
    st.success(f"This is {mnType} mango with the probability of {pred_prob[pred_idx]*100:.02f}%")

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
        predictMango(learn_inf, img)
    elif pred=='03':
        frType = "Orange"
    elif pred=='04':
        frType = "Watermelon"
    else:
        st.error(f"Others with the probability of {pred_prob[pred_idx]*100:.02f}%")
        return None
    st.success(f"This is {frType} with the probability of {pred_prob[pred_idx]*100:.02f}%")
def main():
    st.title('Fruit Classification Model')
    datasrc = st.sidebar.radio("Select input source.", ["Uploaded file", "Take a picture"])
    if datasrc == "Uploaded file": 
        image = get_image_from_upload()
    else:
        image = take_a_picture()
    result = st.button('Classify')
    if result:
        predict(learn_inf, image)        
    
if __name__ == '__main__':
    main()
