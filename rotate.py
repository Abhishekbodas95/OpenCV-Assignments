#Script to rotate a gray scale image
import cv2 #To import cv2 module
image = cv2.imread('myimage.jpg',0) #opening image in gray scale
row,column=image.shape #Returns a tuple of width and height
print(row,column) #To get the height and width of the image
rotate=cv2.getRotationMatrix2D((542,462),130,1) #Rotating the matrix and specifying center coordinates,angle and scale factor
out = cv2.warpAffine(image,rotate,(column,row)) #size of output image
cv2.imshow('myimage.jpg',out) #To display the image
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
