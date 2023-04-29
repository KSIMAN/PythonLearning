from scipy.spatial import distance
from skimage import io
import imageio
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from sympy import  Line2D
import matplotlib.pyplot as plt

from skimage.segmentation import slic, mark_boundaries
import numpy as np
from sklearn.cluster import KMeans


#image = io.imread(input("Путь до изображения: "))

image = imageio.imread("photos/2.jpg")[:,:,:3]

print(image.shape)

image_gray = rgb2gray(image)


segments = slic(image_gray, start_label=0, n_segments=200, compactness=20)
segments_ids = np.unique(segments)
print(segments_ids)

# centers
centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
print(centers)
vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
plt.imshow(mark_boundaries(image_gray, segments))
plt.scatter(centers[:, 1], centers[:, 0], c='y')

for i in range(bneighbors.shape[1]):
    y0, x0 = centers[bneighbors[0, i]]
    y1, x1 = centers[bneighbors[1, i]]

    l = Line2D([x0, x1], [y0, y1], alpha=0.5)
    ax.add_line(l)

blobs_log = blob_log(image_gray, max_sigma=20, num_sigma=10, threshold=.05)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_title('Laplacian of Gaussian')
ax.imshow(image)
c_stars = 0
for blob in blobs_log:
    y, x, r = blob
    if r > 2:
        continue
    ax.add_patch(plt.Circle((x, y), r, color='purple', linewidth=2, fill=False))
    c_stars += 1
print("Количество звёзд: " + str(c_stars))
ax.set_axis_off()
plt.tight_layout()
plt.show()