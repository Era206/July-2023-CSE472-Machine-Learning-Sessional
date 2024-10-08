import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_rank_approximation(A, k):
    U, s, Vt = np.linalg.svd(A)   
    Uk = U[:, :k]
    Sk = np.diag(s[:k])
    Vk = Vt[:k, :]   
    newA = np.dot(Uk, np.dot(Sk, Vk))
    return newA

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ratio = gray.shape[0] / gray.shape[1]
new_width = 950
new_height = int(new_width * ratio)
 
k = [1, 5, 10, 20, 30, 50, 60, 70, 100, 200, 400, 900]
plotCount = len(k)

fig, axs = plt.subplots(3, 4, figsize=(12, 10))

for i in range(3):
    for j in range(4):
        index = i * 4 + j
        if index < plotCount:
            approxImage = low_rank_approximation(gray, k[index])
            axs[i, j].imshow(approxImage, cmap='gray')
            axs[i, j].set_title(f'n_components= {k[index]}')

plt.tight_layout()
#save the plot
plt.savefig('imageReconstruction.png')
plt.show()



