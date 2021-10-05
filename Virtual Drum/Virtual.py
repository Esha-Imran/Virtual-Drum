#--------------------------------------------------------------------------Imports-------------------------------------------------------------------#

import cv2
import numpy as np
import glob
from playsound import playsound
from tkinter import *

#--------------------------------------------------------------------------Configure Stick Class-------------------------------------------------------------------#


class ConfigureStick:
    def __init__(self):
        self.useload = True

    @staticmethod
    def nothing(x):
            pass

    def finding_color(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        cv2.namedWindow("Trackbars")
 
        cv2.createTrackbar("L - H", "Trackbars", 0, 179, ConfigureStick.nothing)
        cv2.createTrackbar("L - S", "Trackbars", 0, 255, ConfigureStick.nothing)
        cv2.createTrackbar("L - V", "Trackbars", 0, 255, ConfigureStick.nothing)
        cv2.createTrackbar("U - H", "Trackbars", 179, 179, ConfigureStick.nothing)
        cv2.createTrackbar("U - S", "Trackbars", 255, 255, ConfigureStick.nothing)
        cv2.createTrackbar("U - V", "Trackbars", 255, 255, ConfigureStick.nothing)
        while True:

            ret, frame = self.cap.read()
            if not ret:  # frame captures without errors...
                break
            frame = cv2.flip(frame, 1)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            # set the lower and upper range according to the value selected by the trackbar.
            lower_range = np.array([l_h, l_s, l_v])
            upper_range = np.array([u_h, u_s, u_v])

            # filter and get the binary mask, where white represents your target color.
            mask = cv2.inRange(hsv, lower_range, upper_range)

            # optionally you can also show the real part of the target color
            res = cv2.bitwise_and(frame, frame, mask=mask)

            mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # stack all frames and show it
            stacked = np.hstack((mask_3, frame, res))
            cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))

            key = cv2.waitKey(1)
            if key == 27:
                thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
                print(thearray)
                break

            if key == ord('s'):
                thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
                print(thearray)

                # Also save this array as drum.npy
                np.save('media/m12/drumval', thearray)
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def processing(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # If true then load color range from memory
        if self.useload:
            drumval = np.load('media/m12/drumval.npy')

        # kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)

        while (1):

            # Take each frame and flip it
            ret, frame = self.cap.read()
            if not ret:  # frame captures without errors...
                break
            frame = cv2.flip(frame, 1)

            # Convert BGR to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # If you're reading from memory then load the upper and lower ranges from there
            if self.useload:
                lower_range = drumval[0]
                upper_range = drumval[1]

            # Otherwise define your own custom values for upper and lower range.
            else:
                lower_range = np.array([150, 70, 173])
                upper_range = np.array([179, 189, 255])

            mask = cv2.inRange(hsv, lower_range, upper_range)

            # perform the morphological operations to get rid of the noise
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            res = cv2.bitwise_and(frame, frame, mask=mask)

            mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # stack all frames and show it
            stacked = np.hstack((mask_3, frame, res))
            cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.8, fy=0.8))

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def track(self):

        # If true then load color range from memory
        if self.useload:
            penval = np.load('media/m12/drumval.npy')

        self.cap = cv2.VideoCapture(0)

        # kernel for morphological operations
        kernel = np.ones((5,5),np.uint8)

        # set the window to autosize so we can view this full screen.
        cv2.namedWindow('image', cv2.WINDOW_FULLSCREEN)

        # this threshold is used to filter noise, the contour area must be bigger than this to qualify as an actual contour.
        noiseth = 500

        while(1):
            
            # Take each frame and flip it
            _, frame = self.cap.read()
            frame = cv2.flip( frame, 1 )

            # Convert BGR to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # If you're reading from memory then load the upper and lower ranges from there
            if self.useload:
                    lower_range = penval[0]
                    upper_range = penval[1]
                    
            # Otherwise define your own custom values for upper and lower range.
            else:             
               lower_range  =  np.array([150,70,173])
               upper_range =  np.array([179,189,255])
            
            mask = cv2.inRange(hsv, lower_range, upper_range)
            
            # perform the morphological operations to get rid of the noise
            mask = cv2.erode(mask,kernel,iterations = 1)
            mask = cv2.dilate(mask,kernel,iterations = 2)
            
            # detect contour.
            contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            # Make sure there was a contour present and also its size was bigger than some threshold.
            if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
                
                # grab the biggest contour
                c = max(contours, key = cv2.contourArea)
                
                # Draw a bounding box around it.
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)        

            cv2.imshow('image',frame)
            
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        self.cap.release()

#--------------------------------------------------------------------------Virtual Drum Class-------------------------------------------------------------------#


class VirtualDrum:
    def __init__(self):
        self.height = 480 
        self.width = 640
        self.useload = True
        self.drums =[] 
        self.rings = []

        for filename in glob.glob('media/M12/drumsounds/ringsound/*.wav'): 
            self.rings.append(filename)
        #print(ringsl)    

        for filename in glob.glob('media/M12/drumsounds/drums/*.wav'): 
            self.drums.append(filename)
        #print(drumsl)  

    @staticmethod
    def inside_roi(x,y,listd):
        found = False
        for idx, it in enumerate(listd):
            if y > it[0] and y < it[1] and x > it[2] and x < it[3]:
                found = True
                break

        return found,idx

    def application(self):
        
        if self.useload:
            drumval = np.load('media/M12/drumval.npy')

        # These are the list of ROIs for the drums and the rings.
        d1 = [[368, 480, 182, 273] , [368, 480, 348 ,441], [348, 480, 2 , 117] ,  [343, 480, 513, 640]]
        r1 = [ [80 ,163 ,2 ,80] , [2 ,50 ,144 ,231]   ,  [0 ,49 ,385 ,486] ,  [88 ,156 ,564 ,640]]

        # Variable which is True when drumstick is in cotact with an ROI
        touching = False

        kernel = np.ones((5,5),np.uint8)
        drum2left =  cv2.imread('media/M12/drumpics/drumcanvas.png')
        resized_drum = cv2.resize(drum2left, (self.width, self.height))

        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow('image2', cv2.WINDOW_FULLSCREEN)

        gray_img = cv2.cvtColor(resized_drum,cv2.COLOR_BGR2GRAY) 
        _ , mask_background = cv2.threshold(gray_img, 247, 255, cv2.THRESH_BINARY_INV)

        while(True):
            
            ret, frame = self.cap.read()
            frame = cv2.flip( frame, 1 ) 
            if ret:
                
                # Convert BGR to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # If you're reading from memory then load the upper and lower ranges from there
                if self.useload:
                        lower_range = drumval[0]
                        upper_range = drumval[1]

                # Otherwise define your own custom values for upper and lower range.
                else:             
                   lower_range  = np.array([150,70,173])
                   upper_range = np.array([179,189,255])

                mask = cv2.inRange(hsv, lower_range, upper_range)

                # perform the morphological operations to get rid of the noise
                mask = cv2.erode(mask,kernel,iterations = 1)
                mask = cv2.dilate(mask,kernel,iterations = 2)
                
                contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                
                # Make sure there was a contour present and also its size was bigger than some threshold.
                if len(contours) > 0 and cv2.contourArea(max(contours, key = cv2.contourArea)) > 450:
                    
                    # Grab the biggest contour
                    cnt = max(contours, key = cv2.contourArea)
                    x,y,w,h = cv2.boundingRect(cnt)
                    
                    # get the midpoint of the drum
                    midx = int(x+(w/2))
                    midy = int(y +(h/2))
                    
                    #Optionally You can track the drums
                    #cv2.circle(frame,(midx,midy), 5, (0,255,0), -1)
                    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    
                    # check if midpoints are inside the drums ROI
                    found,index   = VirtualDrum.inside_roi(midx,midy,d1)
                    
                    # if the point lied inside thedrum roi then play the sound of that instrument using the index
                    if found:
                        # Only play if toucing was previously false
                        if touching == False:
                            playsound(self.drums[index], False)
                            touching = True

                    else:
                        # if the mid point was not inside drum roi then check if they are inside the rings ROI
                        found,index   = VirtualDrum.inside_roi(midx,midy,r1)
                        if found:
                            
                            if touching == False:
                                playsound(self.rings[index],False)
                                touching = True
                                
                        else:
                            touching = False

         
                img_bg = cv2.bitwise_and(frame,frame,mask = cv2.bitwise_not(mask_background))
                img_fg = cv2.bitwise_and(resized_drum,resized_drum,mask =mask_background)
                combined = cv2.add(img_bg,img_fg)
              

            cv2.imshow('image2',combined)
            k = cv2.waitKey(1) 
            if k == ord('q'):
                break
                        
        cv2.destroyAllWindows()
        self.cap.release()     

#--------------------------------------------------------------------------Objects Instantiated-------------------------------------------------------------------#

obj1= ConfigureStick()
obj2 = VirtualDrum()

#--------------------------------------------------------------------------Tkinter GUI----------------------------------------------------------------------------#


root=Tk()
root.geometry("1280x700+40+0")
root.title("Virtual Drum")

#--------------------------------------------------------------------------GUI Media----------------------------------------------------------------------------#

img1 = PhotoImage(file="media/logo.png")
img2 = PhotoImage(file="media/logo-text.png")
img3 = PhotoImage(file='media/configure.png')
img4 = PhotoImage(file='media/run.png')
img5 = PhotoImage(file='media/track.png')

#--------------------------------------------------------------------------GUI Frames and Labels----------------------------------------------------------------------------#

window_frame = Frame(root, width=1280, height=700,bg="#101010")
window_frame.place(x=0, y=0)

main_logo_image = Label(window_frame,image=img1,bg='#101010')
main_logo_image.place(x=500,y=10)

logo_text_image = Label(window_frame,image=img2,bg='#101010')
logo_text_image.place(x=200,y=220)

#--------------------------------------------------------------------------GUI Buttons----------------------------------------------------------------------------#

configure_button = Button(window_frame,image=img3, bd=0, command=obj1.finding_color,bg="#101010")
configure_button.place(x=200, y=400)

application_button = Button(window_frame,image=img4, bd=0, command=obj2.application,bg="#101010")
application_button.place(x=640, y=400)

track_button = Button(window_frame,image=img5, bd=0, command=obj1.track,bg="#101010")
track_button.place(x=400, y=550)

root.mainloop()
