
import cv2 as cv
import numpy as np
import math 

from filters import *

pre_left_x = None
pre_right_y = None
video = cv.VideoCapture("/home/kb2205/Desktop/ENPM 673/PROJECT 2/balase22_project2/input/challenge.mp4")

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Predict Turn.avi', fourcc, 40.0, (1280,720))
while True:
    isTrue,frame = video.read()
    kframe = frame.copy()
    Fframe = frame.copy()
    filter = filter_colors(frame)
   
    gray = cv.cvtColor(filter , cv.COLOR_BGR2GRAY)
    poly = gray.copy()
    internal_poly = np.array( [[[0,720],[200,700],[600,400],[780,420],[1240,710],[1280,720],[1280,0],[0,0]]],dtype=np.int32)
    cv.fillPoly( poly , internal_poly, (0,0,0) )
    blur =  cv.GaussianBlur(poly,(3,3),cv.BORDER_DEFAULT)
    retval,threshed = cv.threshold(blur,150,250,cv.THRESH_BINARY)
    cv.imshow("blur",blur)
    cv.imshow("threshed",threshed)

    lines = cv.HoughLinesP(threshed,1,np.pi/180,threshold=50,minLineLength=1,maxLineGap=2)
    
    # Iterate over points
    for points in lines:
        x1,y1,x2,y2=points[0]
        cv.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

    
    lane_test_undist = Fframe
    s_binary = binary_threshold_lab_luv(lane_test_undist, B_CHANNEL_THRESH, L2_CHANNEL_THRESH)

    # Gradient threshold on S channel
    h, l, s = seperate_hls(lane_test_undist)
    sxbinary = gradient_threshold(s, GRADIENT_THRESH)

    # Combine two binary images to view their contribution in green and red
    color_binary = np.dstack((sxbinary, s_binary, np.zeros_like(s_binary))) * 255
    s_binary = np.uint8(s_binary*255)
    # cv.imshow("undistorted",lane_test_undist)
    # cv.imshow('B/L Channel', s_binary)
    # cv.imshow("Gradient Threshold S/L-Channel Binary",np.uint8(sxbinary*255))
    # cv.imshow("combined Binary",color_binary)
    
    #####     Warping
    h,w = frame.shape[:2]
    cornerpoints = [ (600,424),(750,424),(1100,660),(240,660)]
    p1 = np.float32([[(0,0),(200,0),(200,200),(0,200)]])
    p2=np.float32(cornerpoints)
    h1, w1,s = frame.shape 
    H = cv.getPerspectiveTransform(p2,p1)
    # H =cv.findHomography(p2,p1)
    Hinv = cv.getPerspectiveTransform(p1,p2)
    cva = color_binary + filter 
    cv.imshow("cva",cva)
   
    cvag = cv.cvtColor(filter,cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # clahee = clahe.apply(cvag)
    warped = cv.warpPerspective(cvag,H,(200,200))
    warped = cv.GaussianBlur(warped,(3,3),cv.BORDER_DEFAULT)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    warped = cv.dilate(warped,kernel)
    warped = cv.erode(warped,kernel)
    retval,threshed_warp = cv.threshold(warped,150,255,cv.THRESH_BINARY)
    
    # new = linefit(threshed_warp)
    left_x, left_y = np.where(threshed_warp[:, :100] == 255)
    coeffs = np.polyfit(left_x, left_y, 2)
    left_x = np.arange(70, 200, 1)
    left_y = np.polyval(coeffs, left_x)
    
    if pre_left_x is None: pre_left_x = left_y
    pre_left_x = 0.75*pre_left_x + 0.25*left_y
    
    left_pts = np.int0(np.c_[pre_left_x, left_x])
    
    right_x, right_y = np.where(threshed_warp[:, 130:] == 255)
    coeffs = np.polyfit(right_x, right_y, 3)
    right_x = np.arange(70, 200, 1)
    right_y = np.polyval(coeffs, left_x)
    
    if pre_right_y is None: pre_right_y = right_y
    pre_right_y = 0.75*pre_right_y + 0.25*right_y
    
    right_pts = np.int0(np.c_[130 + pre_right_y, right_x])
    
    new = np.uint8(np.zeros((200, 200, 3))*255)
    cv.polylines(new, [left_pts], False, [0, 0, 255], 4)
    cv.polylines(new, [right_pts], False, [0, 0, 255], 4)
    
    pts = np.r_[left_pts, np.flipud(right_pts)]
    cv.fillPoly(new, [pts], [0, 255, 0])
    cv.imshow("polyfit",new)
    lanes1 = cv.warpPerspective(new, Hinv, (w, h), flags = cv.INTER_LINEAR) 
    overlap = np.uint8(0.5*frame.copy() + 0.3*lanes1)
    
    for points in lines:
        x1,y1,x2,y2=points[0]
        cv.line(overlap,(x1,y1),(x2,y2),(0,255,0),2)

    cv.imshow("final",overlap)
    cv.imshow("Birds eye",threshed_warp)
    cv.imshow("polyfit",new)
    out.write(overlap)
    if cv.waitKey(25) & 0xFF==ord("k"):
        break
video.release()
out.release()
cv.destroyAllWindows()