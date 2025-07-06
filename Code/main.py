# Import Necessary Libraries
import numpy as np
import cv2
import pickle

# Paths
video_path=".\Image And Videos\\Car Parking.mp4"
file_path=".\Parking Position"

# Reading Video
vid = cv2.VideoCapture(video_path)

# Opening our file which contains coordinates
with open(file_path, 'rb') as f:
    poslist = pickle.load(f)

width, height = (250-205), (168-150)

# This Function will predict empty space in the parking 
def check_parking_space(processed_img):
    space_counter=0
    for pos in poslist:
        x,y=pos
        
        cropped_img=processed_img[y:y+height,x:x+width]
        
        # cv2.imshow(str(x*y),cropped_img)
        
        # it will count the number of non-zero elements in the array
        count=cv2.countNonZero(cropped_img)
        
        # cvzone.putTextRect(img,str(count),(x,y+height-2),scale=0.7,thickness=1,offset=0,colorR=(0,0,0),colorT=(255,255,255))
        
        if count<80:
            color=(0,255,0)
            thickness=2
            space_counter+=1
        else:
            color=(0,0,255)
            thickness=1

        cv2.rectangle(img, pos, ((pos[0]+width),(pos[1]+height)), color=color,thickness=thickness)
    cv2.rectangle(img,(455,10),(825,40),(255,0,255),cv2.FILLED)
    cv2.putText(img,"Free Space : {} out of {}".format(space_counter,len(poslist)),(475,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

while True:
    
    # It will loop the video
    if vid.get(cv2.CAP_PROP_POS_FRAMES) == vid.get(cv2.CAP_PROP_FRAME_COUNT):
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = vid.read()
    
    # Resizing the video frames
    img = cv2.resize(img, (1279, 696))

    # Converting Image into gray scale image to detect edges
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # blurring the image will help to detect edges more clearly    
    blur_img=cv2.GaussianBlur(gray_img,ksize=(3,3),sigmaX=1) 
    
    # Converting The Blurred Image into Binary Image
    threshold_img=cv2.adaptiveThreshold(blur_img,
                                        maxValue=255, 
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        thresholdType=cv2.THRESH_BINARY_INV, 
                                        blockSize=25,
                                        C=16)
    
    # Reducing The Noise From Image By Using medianBlur()
    median_img=cv2.medianBlur(threshold_img, ksize=5)
    
    # Dilating the image
    kernel=np.zeros((3,3),np.uint8)
    dilated_img=cv2.dilate(median_img,kernel=kernel,iterations=1)

    # Calling the function which will show empty space in video
    check_parking_space(dilated_img)

    cv2.imshow("Original Video", img)
    # cv2.imshow("Gray Scale Video", gray_img)
    # cv2.imshow("Blurred Video", blur_img)
    # cv2.imshow("Threshold Video", threshold_img)
    # cv2.imshow("Median Video", median_img)
    # cv2.imshow("Dilatted Video", dilated_img)
    
    # This will hold the image on screen until 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
