import streamlit as st
import cv2
from laplacian_pyr import gaussian_pyr, laplacian_pyr, laplacian_display
import numpy as np

tiger = cv2.imread('./images/tigerc.png')
tiger = cv2.cvtColor(tiger, cv2.COLOR_BGR2RGB) / 255.0

st.title('Pyramids and Wavelets')
st.header('Image Resampling')
st.subheader('Interpolation')
st.subheader('Decimation')
st.header('Pyramids')
st.subheader('Gaussian Pyramid')
user_upload = st.file_uploader('Upload Image', ['png', 'jpg', 'jpeg'], False, )
if user_upload is not None:
    ...
st.image(gaussian_pyr(tiger, 5))
st.subheader('Laplacian Pyramid')
st.image(np.clip(laplacian_display(laplacian_pyr(tiger, 5), 5), 0, 1))
st.header('Wavelets')



