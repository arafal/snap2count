#tensorflow imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf


from django.shortcuts import render
from django.http import HttpResponse
from main.forms import ImageUploadForm
from main.models import ImageModel
# from main.hough import importImage 



import sys, types, os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from PIL import Image

import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings


def home(request):
    return render(request,'main/home.html')
    
def contact(request):
    return render(request,'main/contact.html')

def about(request):
    return render(request,'main/about.html')

def upload(request):
    return render(request,'main/upload.html')

#tensorflow functions
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def handle_uploaded_file(f):
    with open('tmp.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def upload_pic(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST , request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['image'])
            # data = request.FILES['image'] # or self.files['image'] in your form
            # path = default_storage.save('tmp.jpg', ContentFile(data.read()))
            image = cv2.imread('tmp.jpg') # read the input image
            # # image = os.path.join(settings.MEDIA_ROOT, path)
            # print("IMAGE TYPE!!!!!!!!!!!!!!!! " + str(type(image)))


   #          m = ImageModel()
   #          m.model = form.cleaned_data['image']
			
			# #Image to be used for hough transformation
   #          image = form.cleaned_data['image']

            #coin = cv2.imread('many.jpg') # read the input image
            pts = np.array([[0,0],[0,3000],[3000,0],[3000,3000]])
            rect = cv2.boundingRect(pts)
            x,y,w,h = rect
            cropped = image[y:y+h, x:x+w].copy()
            coin_resize = cv2.resize(cropped, (300, 300)) # resize the input image to 300x300 size
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

            #print (circles)

            circles = np.uint16(np.around(circles))
            print("\n")
            circles2=circles[0,:,2] # splits the array to only extract the last column of each row (which is the radius)
            #print("The list of radius of given circles are: ", circles2)


            # preparing to crop coins to pass to neural network

            #storing all radii of coins in array called radius
            radius=[]
            j=0
            i=0
            for i in circles[0,:,2]:
                    radius.append(circles[0,j,2])
                    j=j+1

            #storing all x values of coins in array called xvalue
            xvalue=[]
            xx=0
            i=0
            for i in circles[0,:,0]:
                    xvalue.append(circles[0,xx,0])
                    xx=xx+1


            #storing all y values of coins in array called yvalue
            yvalue=[]
            yy=0
            i=0
            for i in circles[0,:,1]:
                    yvalue.append(circles[0,yy,1])
                    yy=yy+1


            #radius1=circles[0,0,2] 
            #radius2=circles[0,1,2] 
            #radius3=circles[0,2,2] 
            #radius4=circles[0,3,2] 

            #x1=circles[0,0,0] 
            #x2=circles[0,1,0] 
            #x3=circles[0,2,0]
            #x4=circles[0,3,0] 

            #y1=circles[0,0,1] 
            #y2=circles[0,1,1] 
            #y3=circles[0,2,1] 
            #y4=circles[0,3,1]  


            #finding the crop values by doing these operations:
            #(x, y+radius) (x, y-radius) (x+radius, y) (x-radius, y)


            #cropping the coins from the input image and making them seperate images
            tmp=0
            i=0
            x=0
            r=0
            y=0
            
            if (not os.path.isdir("TensorFlowInputs")):
                os.mkdir("TensorFlowInputs")

            for i in circles[0,:,1]:

                    #finding the x,y,r values for each specific coin
                    x=xvalue[tmp]
                    y=yvalue[tmp]
                    r=radius[tmp]

                    pts = np.array([[x,y-r],[x-r+1,y],[x,y+r],[x+r+2,y]])
                    rect = cv2.boundingRect(pts)
                    x,y,w,h = rect
                    cropped = coin_resize[y:y+h, x:x+w].copy()
                    pts = pts - pts.min(axis=0)
                    mask = np.zeros(cropped.shape[:2], np.uint8)
                    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
                    bg = np.ones_like(cropped, np.uint8)*255
                    cv2.bitwise_not(bg,bg, mask=mask)
                    dst2 = bg + dst

                    #write each coin to an image and store in folder called "TensorFlowInputs"
                    #so that we can pass these images into the neural network
                    cv2.imwrite(os.path.join("TensorFlowInputs", str(tmp)) + ".jpg", cropped)
                    # cv2.imwrite("TensorFlowInputs/coin" + str(tmp) + ".jpg", cropped)
                    tmp=tmp+1


            i=0
            for i in circles[0,:]:
                
                h=cv2.circle(coin_laplacian,(i[0],i[1]),i[2],(0,255,0),2) # draw the outer circle
                cv2.circle(coin_laplacian,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

# nickel = 0
# penny = 0
# quarter = 0
# dime = 0

# for i in circles2[:]:
# 	if (i>=14):
# 		quarter=quarter+1

# for i in circles2[:]:
#     if (i>=10 and i<= 11):
#         dime=dime+1

# for i in circles2[:]:
#     if (i==13):
#         nickel=nickel+1

# for i in circles2[:]:
#     if (i>=11 and i<= 12):
#         penny=penny+1

# # print statements for each coin type
# print("\n")
# print("Number of Quarters:", quarter)
# print("\n")
# print("Number of Dimes:", dime)
# print("\n")
# print("Number of Nickels:", nickel)
# print("\n")
# print("Number of Pennies:", penny)
# print("\n")
# print("\n")
# print("Total Number of Coins:", penny + nickel + dime + quarter)
# print("\n")

# # Find total amount of money 
# nickel_amount = .05
# penny_amount = .01
# quarter_amount = .25
# dime_amount = .10

# total_amount_USA = (nickel*nickel_amount) + (penny*penny_amount) + (quarter*quarter_amount) + (dime*dime_amount)
# print("The total amount in USA currency: $",total_amount_USA)

# coin_resize2 = cv2.cvtColor(coin_resize, cv2.COLOR_BGR2RGB)

 

# plt.subplot(121),plt.imshow(coin_resize2)
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(image)
# plt.title('Hough Transformation'), plt.xticks([]), plt.yticks([])
# plt.show()



##Katheryne's Hough Python code
#-----------------------------------------------------------------------------------------------

##Tensorflow
            pennyCount = 0
            nickelCount = 0
            dimeCount = 0
            quarterCount = 0

            model_file = "tf_files/retrained_graph.pb"
            label_file = "tf_files/retrained_labels.txt"
            input_height = 299
            input_width = 299
            input_mean = 0
            input_std = 255
            input_layer = "Mul"
            output_layer = "final_result"

            graph = load_graph(model_file)

            for filename in os.listdir('TensorFlowInputs'):
                file_name = os.path.join('TensorFlowInputs', filename)

                t = read_tensor_from_image_file(file_name,
                                        input_height=input_height,
                                        input_width=input_width,
                                        input_mean=input_mean,
                                        input_std=input_std)

                input_name = "import/" + input_layer
                output_name = "import/" + output_layer
                input_operation = graph.get_operation_by_name(input_name);
                output_operation = graph.get_operation_by_name(output_name);

                with tf.Session(graph=graph) as sess:
                    start = time.time()
                    results = sess.run(output_operation.outputs[0],
                                    {input_operation.outputs[0]: t})
                    end=time.time()
                results = np.squeeze(results)

                top_k = results.argsort()[-5:][::-1]
                labels = load_labels(label_file)

                if labels[0] == "penny":
                    pennyCount += 1
                if labels[0] == "nickel":
                    nickelCount += 1
                if labels[0] == "dimes":
                    dimeCount += 1
                if labels[0] == "quarter":
                    quarterCount += 1

                # delete image after using it
                os.remove(file_name)

            print(str(pennyCount) + " pennys\n" + str(nickelCount)+ " nickels\n" + str(dimeCount)+ " dimes\n" + str(quarterCount)+ " quarters\n")
            total = pennyCount + (nickelCount * 5) + (dimeCount * 10) + (quarterCount * 25)

            totalDollars = 0
            if total > 100:
                totalDollars = int(total/100)
    
            totalChange = total % 100
            print("Total amount: " + str(totalDollars) + "." + str(totalChange))





	#TODO: Import python 

            data = form.cleaned_data
            # m.save()
            return render(request,'main/success.html',{'data':data})
            return HttpResponseForbidden('allowed only via POST')
