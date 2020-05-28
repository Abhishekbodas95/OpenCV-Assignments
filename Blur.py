#Script to blur a image in grayscale
import cv2 #To import cv2 module
image = cv2.imread('myimage.jpg',0) #opening image in gray scale
blur = cv2.blur(image,(20,20)) #To blur the image
cv2.imshow('myimage.jpg',blur)
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
