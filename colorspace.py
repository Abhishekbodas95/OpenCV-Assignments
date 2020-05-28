#Script to convert an colored image to Hue, Saturation and Value(HSV)
import cv2 #To import cv2 module
image = cv2.imread('myimage.jpg',1) #To read the image
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#To convert image into HSV
cv2.imshow('myimage.jpg',hsv)
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
