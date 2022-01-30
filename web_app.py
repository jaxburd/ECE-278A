import streamlit as st
import cv2
from laplacian_pyr import gaussian_pyr, laplacian_pyr, laplacian_display
import numpy as np

tiger = cv2.imread('./images/tigerc.png')
tiger = cv2.cvtColor(tiger, cv2.COLOR_BGR2RGB) / 255.0

st.title('Pyramids and Wavelets')
st.header('Image Resampling')
st.write('Image resampling is an important and common procedure in image processing that allows \
    images to be represented at different scales for various purposes. Image resampling can be \
    divided into two techniques depending on whether the user want to approximate a higher \
    resolution image (Interpolation) or to reduce the resulution of the image (Decimation)')

st.subheader('Interpolation')
st.write('The main idea of interpolation is to approximate a continuous representation of the \
    image using the existing samples, and then use that continuous approximation to resample at a \
    higher rate. There are several different types of continuous approximators that can be used for \
    this purpose depending on the desired quality and computational efficiency needed for the application. \
    The approximation and sampling steps of image interpolation are typically combined and implemented in the \
    form of a modified convolution:')
st.image('./images/figures/interp_equation.png', width=400)
st.image('./images/figures/interpolation.png', caption='Interpolation Techniques')
st.text('TODO: Examples, Discussion about different interpolation kernels')
st.subheader('Decimation')
st.text('TODO: ')
st.header('Pyramids')
st.text('TODO: More Information, Image blending demo/example')
st.subheader('Gaussian Pyramid')
user_upload = st.file_uploader('Upload Image', ['png', 'jpg', 'jpeg'], False, )
if user_upload is not None:
    ...
gauss_levels = st.slider('Pyramid Levels', 1, 6, 3, key=1)
st.image(gaussian_pyr(tiger, gauss_levels-1))
st.subheader('Laplacian Pyramid')
laplace_levels = st.slider('Pyramid Levels', 1, 6, 3, key=2)
st.image(np.clip(laplacian_display(laplacian_pyr(tiger, laplace_levels), laplace_levels), 0, 1))
st.header('Wavelets')



