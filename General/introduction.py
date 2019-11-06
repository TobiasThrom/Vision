import numpy as np 
import cv2
import matplotlib.pyplot as plt

def main():
    e1 = cv2.getTickCount()

    #Load an color image in grayscale
    #img = cv2.imread('LaneDetection\input\solidWhiteCurve.jpg', 0)
    #plt.imshow(img, cmap='gray')
    #plt.show()
    create_random_color_image()

    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    print("The code was executed in ",time,"seconds.")


def create_random_bw_image(x=540, y=960):
    rand_image = np.zeros((x,y), np.uint8)
    for i in range(rand_image.shape[0]):
        for j in range(rand_image.shape[1]):
            rand_image[i, j] = np.random.randint(0, 255)
    plt.imshow(rand_image)
    plt.show()
    return rand_image

def create_random_color_image(x=540, y=960):
    rand_image = np.zeros((x,y,3), np.uint8)
    for i in range(rand_image.shape[0]):
        for j in range(rand_image.shape[1]):
            B_random = np.random.randint(0, 255)
            G_random = np.random.randint(0, 255)
            R_random = np.random.randint(0, 255)
            rand_image[i, j] = [B_random, G_random, R_random]
    plt.imshow(rand_image)
    plt.show()
    return rand_image


if __name__ == "__main__":
    main()
