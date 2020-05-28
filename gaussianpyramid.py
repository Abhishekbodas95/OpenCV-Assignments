#Script to display gaussian pyramid of an image
import cv2
import numpy as np
image = cv2.imread('myimage.jpg')
copy = image.copy()
space=int(image.shape[0])*int(image.shape[1])
#Taking image dimensions
row=copy.shape[0]
column=copy.shape[1]
channel=copy.shape[2]
newimage = np.zeros((row,column,channel), dtype=np.uint8) #zero array image
merge = np.concatenate((copy,newimage), axis=1) #merging the original image with a darken image of same size
height = 0
width = image.shape[1]
for i in range(0,5):

    lowerimage = cv2.pyrDown(image) #lowering image resolution
    space += int(lowerimage.shape[0])*int(lowerimage.shape[1])
    merge[height: height + lowerimage.shape[0], width: width + lowerimage.shape[1]]=lowerimage #appending size
    height += lowerimage.shape[0]
    image = lowerimage
    i+=1
print("space requirement for the pyramid is:",space)
cv2.imshow("result", merge)
size=int(merge.size/3)
print("size of smallest rectangular image is:",size)
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
