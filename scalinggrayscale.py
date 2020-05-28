#Script to scale a gray scale image
import cv2 #To import cv2 module
image = cv2.imread('myimage.jpg',0) #opening gray scale image
scale = cv2.resize(image,(300,300), interpolation = cv2.INTER_AREA) #scale down the image to 300x300
cv2.imshow('myimage.jpg',scale) #To display the image
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
