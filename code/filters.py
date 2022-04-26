import cv2 as cv
import numpy as np
import math 
GRADIENT_THRESH = (60, 150)
S_CHANNEL_THRESH = (180, 255)
L_CHANNEL_THRESH = (180, 255)
B_CHANNEL_THRESH = (150, 200)
L2_CHANNEL_THRESH = (225, 255)
####    Functions
def seperate_hls(rgb_img):
    hls = cv.cvtColor(rgb_img, cv.COLOR_RGB2HLS)
    h,l,s =  cv.split(hls)
    return h, l, s

def seperate_lab(rgb_img):
    lab = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
    l,a,b= cv.split(lab)
    return l, a, b

def seperate_luv(rgb_img):
    luv = cv.cvtColor(rgb_img, cv.COLOR_BGR2Luv)
    l,u,v = cv.split(luv)
    return l, u, v

def binary_threshold_lab_luv(rgb_img, bthresh, lthresh):
    l, a, b = seperate_lab(rgb_img)
    l2, u, v = seperate_luv(rgb_img)
    binary = np.zeros_like(l)
    binary[
        ((b > bthresh[0]) & (b <= bthresh[1])) |
        ((l2 > lthresh[0]) & (l2 <= lthresh[1]))
    ] = 1
    return binary

def binary_threshold_hls(rgb_img, sthresh, lthresh):
    h, l, s = seperate_hls(rgb_img)
    binary = np.zeros_like(h)
    binary[
        ((s > sthresh[0]) & (s <= sthresh[1])) &
        ((l > lthresh[0]) & (l <= lthresh[1]))
    ] = 1
    return binary

def gradient_threshold(channel, thresh):
    # Take the derivative in x
    sobelx = cv.Sobel(channel, cv.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold gradient channel
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

def filter_colors(image):
	# Filter white pixels
	white_threshold = 220 
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv.inRange(image, lower_white, upper_white)
	white_image = cv.bitwise_and(image, image, mask=white_mask)
	# Filter yellow pixels
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	lower_yellow = np.array([10,90,80])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv.bitwise_and(image, image, mask=yellow_mask)
	# Combine the two above images
	image2 = cv.addWeighted(white_image, 1., yellow_image, 1., 0.)
	return image2