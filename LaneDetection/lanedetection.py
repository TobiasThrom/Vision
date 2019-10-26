import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import plot,show
import numpy as np
from numpy import arange,array,ones,linalg
import math
import cv2
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines_hough = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines_hough)

    return line_img


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    imshape = img.shape

    vertices = np.array([[(50, imshape[0]), (450, 320), (500, 320), (900, imshape[0])]], dtype=np.int32)
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    left_x1_max = 200
    left_y1_max = imshape[0]
    left_x2_max = 450
    left_y2_max = 320

    right_x1_max = 500
    right_y1_max = 320
    right_x2_max = 870
    right_y2_max = imshape[0]


    for line in lines:
        for x1, y1, x2, y2 in line:

            x_values = float(x2 - x1)
            y_values = float(y2 - y1)
            slope = float(y_values / x_values)
            if slope < 0:
                # Ignore obviously invalid lines
                if slope > -.5 or slope < -.8:
                    continue

                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)

            else:
                # Ignore obviously invalid lines
                if slope < .5 or slope > .8:
                    continue
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)

    right_x.append(right_x1_max)
    right_x.append(right_x2_max)
    right_y.append(right_y1_max)
    right_y.append(right_y2_max)

    left_x.append(left_x1_max)
    left_x.append(left_x2_max)
    left_y.append(left_y1_max)
    left_y.append(left_y2_max)

    x_right = np.asarray(right_x)
    y_right = np.asarray(right_y)
    A_right = array([x_right, ones(len(x_right))])
    w_right = linalg.lstsq(A_right.T, y_right)[0]  # obtaining the parameters
    line_right = w_right[0] * x_right + w_right[1]  # regression line

    x_left = np.asarray(left_x)
    y_left = np.asarray(left_y)
    A_left = array([x_left, ones(len(x_left))])
    w_left = linalg.lstsq(A_left.T, y_left)[0]  # obtaining the parameters
    line_left = w_left[0] * x_left + w_left[1]  # regression line

    x1_right = min(x_right)
    x2_right = int((imshape[0] - w_right[1])/w_right[0])
    y1_right = int(min(line_right))
    y2_right = imshape[0]

    x1_left = int((imshape[0] - w_left[1])/w_left[0])
    x2_left =max(x_left)
    y1_left = imshape[0]
    y2_left = int(min(line_left))


    cv2.line(img, (x1_right,y1_right),(x2_right,y2_right), color, thickness)
    cv2.line(img, (x2_left, y2_left), (x1_left, y1_left), color, thickness)


def weighted_img(img, initial_img, alpha=.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def get_slope(x1, y1, x2, y2):
    return float(float(y2 - y1) / float(x2 - x1))

def process_image(image):

    imshape = image.shape
    #transform the image to greyscake
    gray = grayscale(image)
    #apply a gaussian blur to smooth the image for the canny edge detection filter
    blur_gray = gaussian_blur(gray, 5)
    #apply the canny edge detection filter
    canny_blur = canny(blur_gray, 100, 200)
    plt.imshow(canny_blur)
    plt.show()
    #only a certain field of the image is relevant for the next step
    vertices = np.array([[(50, imshape[0]), (450, 320), (500, 320), (900, imshape[0])]], dtype=np.int32)
    region_masked = region_of_interest(canny_blur, vertices)
    plt.imshow(region_masked)
    plt.show()
    #appy hough transformation
    hough_picture = hough_lines(canny_blur, 2, np.pi / 180, 20, 50, 30)
    
    result = weighted_img(hough_picture, image)
    plt.imshow(hough_picture)
    plt.show()
    return result


def main():
    images = os.listdir('LaneDetection/input/')
    for img_file in images:
        print('Loading an image')
        image = mpimg.imread("LaneDetection/input/" + img_file)

        print('Processing an image')
        processed_image = process_image(image)
        plt.imshow( processed_image)
        plt.show()

        print('Saving an image \n')
        mpimg.imsave('LaneDetection/output/lines-' + img_file, processed_image)

if __name__== "__main__":
  main()