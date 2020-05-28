#Script to cause erosion a gray scale image
import numpy as np #To import numpy library
import cv2 #To import cv2 module
image = cv2.imread('myimage.jpg',0) #opening gray scale image
#To return matrix of ones
matrix = np.ones((10,10),np.int8) #matrix of size 10 and data type int8
erosion = cv2.erode(image,matrix,iterations = 2)#To erode the image by 2 iterating twice
cv2.imshow('myimage.jpg',erosion) #To display the image
key=cv2.waitKey(10000) & 0xFF #Display duration = 10 seconds & Mask for 64-bit systems
if key==27: #Press Escape key to close the image window
    cv2.destroyAllWindows()
elif key==ord('q'): #Press 'q' key to quit the image window
    cv2.destroyAllWindows()
elif key==ord('e'):#Press 'e' key to exit the image window
    cv2.destroyAllWindows()
elif key==ord('x'):#Press 'x' key to cancel the image window
    cv2.destroyAllWindows()
cv2.destroyAllWindows() #To destroy windows anyway
