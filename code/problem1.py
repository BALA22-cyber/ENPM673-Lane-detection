import cv2 as cv
import numpy as np

def hist_equalize(image):
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = np.uint8(cdf/cdf.max()*255)
    equalized = cdf_normalized[image]
    return equalized

## Here k is the kernel size
def adaptive_Hist_Equalize(image, k = 2):
    image = image.copy()
    h, w = image.shape
    sh, sw = h//k, w//k
    for i in range(0, h, sh):
        for j in range(0, w, sw):
            image[i:i+sh, j:j+sw] = hist_equalize(image[i:i+sh, j:j+sw])
    return image

video = cv.VideoCapture("/home/kb2205/Desktop/ENPM 673/PROJECT 2/balase22_project2/input/problem1.avi")
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('HistogramEqualization.avi', fourcc, 1, (1224,370))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out2 = cv.VideoWriter('AdaptiveEqualization.avi', fourcc, 1, (1224,370))

while True:
    isTrue, frame = video.read()
    
    ### Split into BGR for equalization

    b, g, r = cv.split(frame)
    
    equalizedb = hist_equalize(b)
    adp_equalizedb = adaptive_Hist_Equalize(b)
    
    equalizedg = hist_equalize(g)
    adp_equalizedg = adaptive_Hist_Equalize(g)
    
    equalizedr = hist_equalize(r)
    adp_equalizedr = adaptive_Hist_Equalize(r)

    ### HSV scale
    merged1 = cv.merge([equalizedb ,equalizedg, equalizedr])
    merged2 = cv.merge([adp_equalizedb, adp_equalizedg, adp_equalizedr])

    cv.imshow('Input', frame)
    cv.imshow('Histogram Equalization', merged1)
    cv.imshow('Adaptive Histogram Equalization', merged2)
    out.write(merged1)
    out2.write(merged2)
    if cv.waitKey(300) & 0xFF==ord("k"):
        break
video.release()
out.release()
out2.release()
cv.destroyAllWindows()

