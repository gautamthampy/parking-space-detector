# Import necessary Libraries
import cv2
import pickle

# Defining Image Path and File Path
img_path="../Image And Videos/Car Parking.png"
file_path="../Parking Position"

width,height=(250-205),(168-150)

# Functions To Put Rectangular Box For Parking Space
def mouseclick(events,x,y,flags, param): # flags, params are used for handling extra parameter coming from setMOuseCallback 
    
    # This 'if' block used to append coordinates in poslist when mouse's left button is clicked which gives the initial  coordinates to make rectangular box.
    if events==cv2.EVENT_LBUTTONDOWN:
        poslist.append((x,y))
    
    # This 'if' block used to pop the rectangular box coordinated when mouse's right button is clicked.
    if events==cv2.EVENT_RBUTTONDOWN:
            for i,pos in  enumerate(poslist):
                 x1,y1=pos
                 if x1<x<x1+width and y1<y<y1+height:
                      poslist.pop(i)
    
    # All The Coordinates will be saved in a file 
    with open (file_path,'wb') as f:
         pickle.dump(poslist,f)

# Using try-except block to handle error while opening a file
try:
    with open(file_path,'rb') as f:
        poslist=pickle.load(f)
except:
     poslist=[]          


# This code block will generate rectangular box according to the coordinates and showing it on image
while True:
    img=cv2.imread(img_path) # reading the image
    img=cv2.resize(img,(1279,696)) # resizing the image according to the requirements 

    # This will generate rectangular box
    for pos in poslist:
        cv2.rectangle(img,pos,((pos[0]+width),(pos[1]+height)),(0,0,255),2)
    
    # Showing the image with rectangular boxes
    cv2.imshow("Parking",img)
    
    cv2.setMouseCallback("Parking",mouseclick)

    # image will remain on screen until 'q' key is not pressed
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
