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
    divided into two techniques depending on whether the user wants to approximate a higher \
    resolution image (Interpolation) or to reduce the resulution of the image (Decimation)')

st.subheader('Interpolation')
st.write('The main idea of interpolation is to approximate a continuous representation of the \
    image using the existing samples, and then use that continuous approximation to resample at a \
    higher rate. There are several different types of continuous approximators that can be used for \
    this purpose depending on the desired quality and computational efficiency needed for the application. \
    The approximation and sampling steps of image interpolation are typically combined and implemented in the \
    form of a modified convolution:')
st.image('./images/figures/interp_equation.png', width=400)
st.write('Where r is the scalar upsampling rate, f() is the original image, and h() is the interpolation kernel')
st.image('./images/figures/interpolation.png', caption='Interpolation Techniques')
st.write('Common Interpolation Kernels')
st.write('Bilinear: Bilinear interpolation is generates a peiceswise linear representation of the source image \
    to resample. It is a computationally efficient algorithm that only requires a 2x2 convolution kernel. \
    The main drawback of this method is that the approximation is not continuously differentiable which can result \
    in unappealing creasing in the rescaled image.')
st.write('Bicubic: Bicubic interpolation is the result of fitting cubic splines to the image data and results \
    in a smoother, more visually appealing result when compared with bilinear interpolation. \
    The main drawback of this technique is that the interpolation convolution requires a 4x4 kernel \
    making it more computationally expensive than Bilinear interpolation')
st.image('./images/figures/interpolation_compare.png', caption='Visual comparison of interpolation kernels on a sample distribution')
st.subheader('Decimation')
st.write('Decimation is the process of reducing a resolution by first blurring the image to avoid aliasing and then only keeping \
    every rth sample where integer r is the decimation rate. In the case that r is not an integer, the image can fist be interpolated \
    by integer factor L, then decimated by integer factor M such that r = L/M. Decimation can also be implemented in the form of a \
    modified convolution with form:')
st.image('./images/figures/decimation_equation.png', width=400)
st.write('Where r is the decimation rate, f() is the original image, and h() is the smoothing kernel')
st.write('Much like interpolation, there are several kernel options to choose from, each with different frequency cutoff properties.')
st.header('Pyramids')
st.write('Image pyramids are multiscale respresentations of images that are very useful for computer vision tasks \
    such as coarse-to-fine image search algorithms and multiscale pattern recognition/feature tracking.')
st.subheader('Gaussian Pyramid')
st.write('A Gaussian pyramid is constructed by successively decimating an image by a factor of 2 until \
    the desired number of levels is reached. Despite what the name suggests, the decimation blurring kernel used is typically in the \
    form of a binomial distribution. The name orginates from the fact that repeated convolutions of \
    the binomial kernel converges to a gaussian rather than the use of a gaussian kernel.')
user_upload = st.file_uploader('Upload Image', ['png', 'jpg', 'jpeg'], False, )
if user_upload is not None:
    ...
gauss_levels = st.slider('Pyramid Levels', 1, 6, 3, key=1)
st.image(gaussian_pyr(tiger, gauss_levels-1))
st.subheader('Laplacian Pyramid')
st.write('A Laplacian pyramid can be computed from a corresponding Gaussian pyramid by interpolating each level other than the original \
    by a factor of 2, then taking the difference between the gaussian pyramid and the interpolated results. The result of the difference \
    between the low pass Gaussian images and \"lower\" pass interpolated images is a bandpass representation of the original image \
    at different frequency bands. The name Laplacian pyramid comes from the idea that the levels of the pyramid are approximately the \
    same as convolving the original images with a Laplacian of a Gaussian kernel')
laplace_levels = st.slider('Pyramid Levels', 1, 6, 3, key=2)
st.image(np.clip(laplacian_display(laplacian_pyr(tiger, laplace_levels), laplace_levels), 0, 1), caption='Note: \
    Though it is sometimes not pictured in the Laplacian Pyramid, the smallest level of the Gaussian pyramid is needed to be able to \
    reconstruct the original image')
st.header('Wavelets')



