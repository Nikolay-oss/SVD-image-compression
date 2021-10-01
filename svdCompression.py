import numpy as np
from numpy import byte, mat, eye, uint8, uint16
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


fig_n = 1

def svd_compress(img_array, n):
	U, S, V = np.linalg.svd(img_array, full_matrices=False)
	S = np.diag(S[:n])
	img_svd = U[:, :n] @ S @ V[:n, :]
	return img_svd

def gray_svd_format(img_array, n):
	img_array = np.mean(img_array, 2)
	U, S, V = np.linalg.svd(img_array, full_matrices=False)
	S = np.diag(S[:n])
	img_svd = U[:, :n] @ S @ V[:n, :]
	return img_svd

def split_RGB_channels(img):
	img1 = img[:,:,0]
	img2 = img[:,:,1]
	img3 = img[:,:,2]
	return img1, img2, img3

def show_image(img, cmap=None):
	global fig_n
	plt.figure(fig_n)
	plt.axis('off')
	plt.imshow(img, cmap)
	fig_n += 1

def svd_example(path, n=30):
	img = mpimg.imread(path)
	show_image(img) # origin image
	img1, img2, img3 = split_RGB_channels(img)
	show_image(img1, 'Reds')
	show_image(img2, 'Greens')
	show_image(img3, 'Blues')

	img_svd1 = svd_compress(img1, n)
	print(img_svd1.shape)
	img_svd2 = svd_compress(img2, n)
	print(img_svd2.shape)
	img_svd3 = svd_compress(img3, n)
	print(img_svd3.shape)

	img_svd = np.arange(img.size).reshape(img.shape[0], img.shape[1], img.shape[2])

	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			img_svd[i, j, 0] = (img_svd1[i, j])
			img_svd[i, j, 1] = (img_svd2[i, j])
			img_svd[i, j, 2] = (img_svd3[i, j])

	show_image(img_svd)
	img_svd_gray = gray_svd_format(img, n)
	show_image(img_svd_gray, 'gray')

	# plt.imsave('img.jpg', img_svd)

	plt.show()

def main():
	svd_example('sources/2.jpg', 5)

if __name__ == "__main__":
	main()
