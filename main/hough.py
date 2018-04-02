
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from PIL import Image

coin = cv2.imread('22.jpg') # read the input image
coin_resize = cv2.resize(coin, (200, 200)) # resize the input image to 300x300 size
coin_resize2 = cv2.cvtColor(coin_resize, cv2.COLOR_BGR2RGB)


#coin_resize = cv2.resize(coin, None, fx=0.1, fy=0.1, interpolation =cv2.INTER_AREA )
coin_HSV = cv2.cvtColor(coin_resize2, cv2.COLOR_BGR2HSV) # convert to HSV color
coin_HSV = cv2.cvtColor(coin_HSV, cv2.COLOR_BGR2RGB)
coin_S = coin_HSV[:,:,1] # extract only saturatuon S

coin_blur = cv2.medianBlur(coin_S, 5) # blur
coin_blur = cv2.medianBlur(coin_blur, 5) # blur

ignore, coin_Otsu = cv2.threshold(coin_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # convert to Otsu
#plt.subplot(121),plt.imshow(coin_Otsu)
#plt.show()
coin_laplacian = cv2.Laplacian(coin_Otsu,cv2.CV_64F) # convert to Laplacian

image = coin_laplacian.astype(np.uint8)
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1,20, param1 = 600, param2 = 30, minRadius = 0, maxRadius =0 )

print (circles)

circles = np.uint16(np.around(circles))
print("\n")
circles2=circles[0,:,2] # splits the array to only extract the last column of each row (which is the radius)
print("The list of radius of given circles are: ", circles2)


pts = np.array([[159,101],[230,102],[230,164],[159,164]])
rect = cv2.boundingRect(pts)
x,y,w,h = rect
cropped = coin_resize[y:y+h, x:x+w].copy()
pts = pts - pts.min(axis=0)
mask = np.zeros(cropped.shape[:2], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
dst = cv2.bitwise_and(cropped, cropped, mask=mask)
bg = np.ones_like(cropped, np.uint8)*255
cv2.bitwise_not(bg,bg, mask=mask)
dst2 = bg+ dst
cv2.imwrite("cropped.png", cropped)


for i in circles[0,:]:
    
    h=cv2.circle(coin_laplacian,(i[0],i[1]),i[2],(0,255,0),2) # draw the outer circle
    cv2.circle(coin_laplacian,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

nickel = 0
penny = 0
quarter = 0
dime = 0

for i in circles2[:]:
	if (i>=14):
		quarter=quarter+1

for i in circles2[:]:
    if (i>=10 and i<= 11):
        dime=dime+1

for i in circles2[:]:
    if (i==13):
        nickel=nickel+1

for i in circles2[:]:
    if (i>=11 and i<= 12):
        penny=penny+1

# print statements for each coin type
print("\n")
print("Number of Quarters:", quarter)
print("\n")
print("Number of Dimes:", dime)
print("\n")
print("Number of Nickels:", nickel)
print("\n")
print("Number of Pennies:", penny)
print("\n")
print("\n")
print("Total Number of Coins:", penny + nickel + dime + quarter)
print("\n")

# Find total amount of money 
nickel_amount = .05
penny_amount = .01
quarter_amount = .25
dime_amount = .10

total_amount_USA = (nickel*nickel_amount) + (penny*penny_amount) + (quarter*quarter_amount) + (dime*dime_amount)
print("The total amount in USA currency: $",total_amount_USA)

coin_resize2 = cv2.cvtColor(coin_resize, cv2.COLOR_BGR2RGB)



plt.subplot(121),plt.imshow(coin_resize2)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(image)
plt.title('Hough Transformation'), plt.xticks([]), plt.yticks([])
plt.show()

