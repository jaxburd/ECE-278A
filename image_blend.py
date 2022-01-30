from laplacian_pyr import gaussian_pyr, laplacian_pyr, laplacian_reconstruct
import cv2
import matplotlib.pyplot as plt
import numpy as np

def blend(filename, levels):
    mask = cv2.imread('./blend/mask/' + filename) / 255.0
    imageA = cv2.imread('./blend/imageA/' + filename) / 255.0
    imageB = cv2.imread('./blend/imageB/' + filename) / 255.0

    mask_pyr = gaussian_pyr(mask, levels)
    imageA_pyr = laplacian_pyr(imageA, levels)
    imageB_pyr = laplacian_pyr(imageB, levels)

    new_pyr = (imageA_pyr * mask_pyr) + (imageB_pyr * (1 - mask_pyr))
    return laplacian_reconstruct(new_pyr, levels)

result = blend('fruitBlend.png', 5)
result = cv2.cvtColor(np.clip(result * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.show()
