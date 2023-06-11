import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import torch

n = torch.load(map_location = 'cpu')

def main():
	pickle_in = open('fruit.pkl', 'rb')
	classifier = pickle.load(pickle_in)

	st.sidebar.header("Fruit Classification")
	st.title("Fruit Classification")
	uploaded_file = st.file_uploader("Choose a fruit picture file", type=["png", "jpg", "jpeg"])

	submit = st.button('Predict')

	if submit:
		image = Image.open(uploaded_file)
		img_array = np.asarray(image)
		img_reshape = img_array[np.newaxis, ...]
		prediction = classifier.predict(img_reshape)
		if prediction == 0:
			st.write("")
			st.success("Apple Picture")
		elif prediction == 1:
			st.write("")
			st.success("Banana Picture")
		elif prediction == 2:
			st.write("")
			st.success("Mango Picture")
		elif prediction == 3:
			st.write("")
			st.success("Orange Picture")
		elif prediction == 4:
			st.write("")
			st.success("Watermelon Picture")
		else:
			st.write("")
			st.error("Can't detect what this fruit is...")
if __name__ == '__main__':
	main()
