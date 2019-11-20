import numpy as np 
import cv2


def main():
    e1 = cv2.getTickCount()

    #Load an color image in grayscale
    #img = cv2.imread('LaneDetection\input\solidWhiteCurve.jpg', 0)
    #plt.imshow(img, cmap='gray')
    #plt.show()
    #create_random_color_image()
    corner_detection()

    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    print("The code was executed in ",time,"seconds.")


def create_random_bw_image(x=540, y=960):
    rand_image = np.zeros((x,y), np.uint8)
    for i in range(rand_image.shape[0]):
        for j in range(rand_image.shape[1]):
            rand_image[i, j] = np.random.randint(0, 255)
    cv2.imshow('image',rand_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return rand_image

def create_random_color_image(x=540, y=960):
    rand_image = np.zeros((x,y,3), np.uint8)
    for i in range(rand_image.shape[0]):
        for j in range(rand_image.shape[1]):
            B_random = np.random.randint(0, 255)
            G_random = np.random.randint(0, 255)
            R_random = np.random.randint(0, 255)
            rand_image[i, j] = [B_random, G_random, R_random]
    cv2.imshow('image',rand_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return rand_image

def corner_detection():
    img = cv2.imread('General\images\chessboard.jpeg', cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray2 = np.float32(gray)

    #Harris Corner Detection
    dst = cv2.cornerHarris(gray2,2,5,0.04)
    dst = cv2.dilate(dst,None)
    img[dst>0.1*dst.max()]=[0,0,255]

    #Shi-Tomasi Corner Detector
    corners = cv2.goodFeaturesToTrack(gray,25,0.05,20)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)


    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('General\output\output-chessboard.jpeg', img)




if __name__ == "__main__":
    main()
