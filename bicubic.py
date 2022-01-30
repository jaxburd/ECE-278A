import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def bicubic_rescale(img, factor, a=-0.5):
    H = int(img.shape[0] * factor)
    W = int(img.shape[1] * factor)
    channels = img.shape[2]
    # pad input image with repeated edge pixels
    padded = np.zeros((img.shape[0] + 4, img.shape[1] + 4, channels), dtype=np.float32)
    padded[2:img.shape[0]+2, 2:img.shape[1]+2, :] = img
    padded[0:2, :, :] = padded[2:4, :, :]
    padded[:, 0:2 ,:] = padded[:, 2:4, :]
    # construct array that will store the output image
    output = np.zeros((H, W, channels), dtype=np.float32)

    for c in range(channels):
        for i in range(H):
            for j in range(W):
                ux = j / factor + 2.0
                uy = i / factor + 2.0
                dx = ux - math.floor(ux)
                dy = uy - math.floor(uy)
                ux = int(ux)
                uy = int(uy)
                # image patch
                im_mat = padded[(uy-2):(uy+2), (ux-2):(ux+2), c].transpose()
                # bicubic rows kernel
                mat_r = np.array([[_h(1+dx, a), _h(dx, a), _h(1-dx, a), _h(2-dx, a)]])
                # bicubic columns kernel
                mat_c = np.array([[_h(1+dy, a)], [_h(dy, a)], [_h(1-dy, a)], [_h(2-dy, a)]])
                # pixel value computation
                output[i, j, c] = np.dot(np.dot(mat_r, im_mat), mat_c)

    return output


# bicubic kernel generating function
def _h(x, a):
    if (abs(x) >= 0) & (abs(x) <= 1):
        return (a+2)*(abs(x)**3)-(a+3)*(abs(x)**2)+1
        
    elif (abs(x) > 1) & (abs(x) <= 2):
        return a*(abs(x)**3)-(5*a)*(abs(x)**2)+(8*a)*abs(x)-4*a
    return 0

