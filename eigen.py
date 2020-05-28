#Script to display edges of image using eigen values
import cv2
import glob #To import glob module
import numpy as np 
from matplotlib import pyplot as plt

eigenhist=[]
print("Eigen")
for eachimage in glob.glob('ST2main/*.jpg'): #Location of unziped image folder
    print(eachimage)
    colorimage = cv2.imread(eachimage) 
    colorimage=cv2.blur(colorimage,(5,5))  
    
    bluex = cv2.Sobel(colorimage[:,:,0],cv2.CV_64F,1,0,ksize=3) #To compute image gradients x,y of each colored channels
    bluey = cv2.Sobel(colorimage[:,:,0],cv2.CV_64F,0,1,ksize=3)
    greenx = cv2.Sobel(colorimage[:,:,1],cv2.CV_64F,1,0,ksize=3)
    greeny = cv2.Sobel(colorimage[:,:,1],cv2.CV_64F,0,1,ksize=3)
    redx = cv2.Sobel(colorimage[:,:,2],cv2.CV_64F,1,0,ksize=3)
    redy = cv2.Sobel(colorimage[:,:,2],cv2.CV_64F,0,1,ksize=3)
    
    a = np.square(bluex) + np.square(greenx) + np.square(redx)                  #Matrix S= |a b|
    b = np.multiply(bluex,bluey) + np.multiply(greenx,greeny) + np.multiply(redx,redy)   # |b c|
    c = np.square(bluey) + np.square(greeny) + np.square(redy)
    
    elambda=0.5*((a+c) + np.sqrt(np.square(a+c)-4*(np.multiply(a,c)-np.square(b)))) #Formula for lambda 
    emagnitude = np.sqrt(elambda)
    cv2.imshow("eigen edges",(np.uint8(emagnitude)))
    
    x = np.divide(-b,a-elambda,where=a - elambda != 0)    #x=(-b/a-lambda)
    e1 = np.divide(x,np.sqrt(np.square(x)+1))  
    e2 = np.divide(1,np.sqrt(np.square(x)+1))   
    eangle = cv2.phase(e1,e2,angleInDegrees=True) #Direction in degrees
    eangle = np.rint(np.divide(eangle,5))
    
    ehist,ebin = np.histogram(eangle,36,[0,36])  #36bins
    eigenhist.append(ehist)
    plt.plot(ehist)
    plt.text(0,0,"Eigen histograms", bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()

def hintersection(h1,h2): #histogram comparison function by calculating intersection
    hmin = np.sum(np.minimum(h1, h2))
    hmax = np.sum(np.maximum(h1, h2))
    return float(hmin/hmax)

def hchisquared(h1,h2): #histogram comparison function by calculating chi square
    hchi=0
    for i in range(0,len(h1)):
        if (h1[i]+h2[i])>5:
            hchi+=(((h1[i]- h2[i])**2)/float(h1[i]+h2[i]))
    return hchi

#Using above functions to compare image pairs
intmatrix = np.zeros((99,99),dtype='uint8')
for i1 in range(0,99):
    for i2 in range(0,99):
        intmatrix[i1][i2] = (1-(hintersection(eigenhist[i1],eigenhist[i2])))*255        
plt.imshow(intmatrix, cmap='hot')
plt.colorbar()
plt.text(0,0,"Intersection", bbox=dict(facecolor='yellow', alpha=0.5)) #To display text
plt.show()

chimatrix = np.zeros((99,99), dtype='float64')
for i1 in range(0,99):
    for i2 in range(0,99):
        chimatrix[i1][i2]=hchisquared(eigenhist[i1],eigenhist[i2])                               
plt.imshow(chimatrix,cmap='hot')
plt.colorbar()
plt.text(0,0,"Chi squared", bbox=dict(facecolor='yellow', alpha=0.5)) #To display text
plt.show() 
#Display in sequential order, Close each window for the next window,press q to exit

    