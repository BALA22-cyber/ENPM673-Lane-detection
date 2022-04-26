import numpy as np
import cv2 as cv


def hough_transform(canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2, make_copy=True):

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(img, (x1, y1), (x2, y2), color, thickness)
    
    return img

def separate_lines(lines, img):
    img_shape = img.shape[0:2]
    
    middle_x = img_shape[1] / 2
    
    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1 
            if dx == 0:
                #Discarding line since we can't gradient is undefined at this dx
                continue
            dy = y2 - y1
            
            # Similarly, if the y value remains constant as x increases, discard line
            if dy == 0:
                continue
            
            slope = dy / dx
            epsilon = 0.3
            if abs(slope) <= epsilon:
                continue
            
            if slope < 0 and x1 < middle_x and x2 < middle_x:
                # Lane should also be within the left hand side of region of interest
                left_lane_lines.append([[x1, y1, x2, y2]])
            elif x1 >= middle_x and x2 >= middle_x:
                # Lane should also be within the right hand side of region of interest
                right_lane_lines.append([[x1, y1, x2, y2]])
    
    return left_lane_lines, right_lane_lines

def color_lanes(img, left_lane_lines, right_lane_lines, left_lane_color=[0,0,255], right_lane_color=[0,255,0]):
    left_colored_img = draw_lines(img, left_lane_lines, color=left_lane_color, make_copy=True)
    right_colored_img = draw_lines(left_colored_img, right_lane_lines, color=right_lane_color, make_copy=False)
    
    return right_colored_img