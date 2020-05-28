#Script to convert an colored image to gray scale
import cv2 #To import cv2 module
image = cv2.imread('Myimage.jpg') #opening colored image
convert = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting to gray scale
cv2.imshow('Myimage.jpg',convert) #displaying image
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
