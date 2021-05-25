import cv2
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog
import speech_recognition as sr
import pyaudio
import random
import time
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *
from yolov3.yolov4 import Create_Yolo
from gtts import gTTS
import os
import playsound
from PIL import Image
import pyttsx3
import matplotlib.pyplot as plt
import speech_recognition as sr

# Defining CreateWidgets() function to create necessary tkinter widgets
def CreateWidgets():
    
    second_frame.feedlabel = Label(second_frame, bg="#641E16", fg="white", text="try saying 'What can you do?'", font=('Comic Sans MS',20))
    second_frame.feedlabel.grid(row=1, column=1, padx=10, pady=10)
    
    second_frame.speak = Button(second_frame, text="Speak", command=Process, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=20)
    second_frame.speak.grid(row=2, column=1, padx=10, pady=10)
    
    second_frame.status = Label(second_frame, bg="#641E16", fg="white", text="", font=('Comic Sans MS',20))
    second_frame.status.grid(row=3, column=1, padx=10, pady=10)
    
    second_frame.va = Label(second_frame, bg="#641E16", fg="white", text="", font=('Comic Sans MS',20))
    second_frame.va.grid(row=4, column=1, padx=10, pady=10)
    
    
    
def speak(output):
        #initiating the speech engine
        engine = pyttsx3.init()
        #speaking the desired output 
        engine.say(output)
        second_frame.va.configure(text = "VA : " + output)
        engine.runAndWait() 
        
def speech_recog():
    #recognizer class
    r=sr.Recognizer()
    #Specifing the microphone to be activated
    mic = sr.Microphone()

    #listening to the user 
    with mic as s:
        audio = r.record(s,duration=5)
        print("recorded")
        r.adjust_for_ambient_noise(s)

    #Converting the audio to text  
    try:
        """I use google engine to convert the speech 
         text but you may use other engines such as 
         sphinx,IBM speech to text etc."""
        speech = r.recognize_google(audio)
        #self.Input.config(text=speech)
        second_frame.status.configure(text = "you : " + speech)
        print(speech)
        return speech

    except sr.UnknownValueError:
        #calling the text to speech function
        speak("please try again,couldnt identify")

    except sr.WaitTimeoutError as e:
        speak("please try again") 
        
def ShowFeed():
    # Capturing frame by frame
    ret, frame = second_frame.cap.read()

    if ret:
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)

        # Displaying date and time on the feed
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))

        # Changing the frame color from BGR to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Creating an image memory from the above frame exporting array interface
        videoImg = Image.fromarray(cv2image)

        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image = videoImg)

        # Configuring the label to display the frame
        second_frame.cameraLabel.configure(image=imgtk)

        # Keeping a reference
        second_frame.cameraLabel.imgtk = imgtk

        # Calling the function after 10 milliseconds
        second_frame.cameraLabel.after(10, ShowFeed)
    else:
        # Configuring the label to display the frame
        second_frame.cameraLabel.configure(image='')

def destBrowse():
    # Presenting user with a pop-up for directory selection. initialdir argument is optional
    # Retrieving the user-input destination directory and storing it in destinationDirectory
    # Setting the initialdir argument is optional. SET IT TO YOUR DIRECTORY PATH
    destDirectory = filedialog.askdirectory(initialdir="YOUR DIRECTORY PATH")

    # Displaying the directory in the directory textbox
    destPath.set(destDirectory)
    
def Capture():
    # Storing the date in the mentioned format in the image_name variable
    image_name = datetime.now().strftime('%d-%m-%Y %H-%M-%S')

    # If the user has selected the destination directory, then get the directory and save it in image_path
    if destPath.get() != '':
        image_path = destPath.get()
    # If the user has not selected any destination directory, then set the image_path to default directory
    else:
        image_path = "YOUR DEFAULT DIRECTORY PATH"

    # Concatenating the image_path with image_name and with .jpg extension and saving it in imgName variable
    imgName = image_path + '/' + image_name + ".jpg"

    # Capturing the frame
    ret, frame = second_frame.cap.read()

    # Displaying date and time on the frame
    cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (430,460), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))

    # Writing the image with the captured frame. Function returns a Boolean Value which is stored in success variable
    success = cv2.imwrite(imgName, frame)

    if success :
        messagebox.showinfo("SUCCESS", "IMAGE CAPTURED AND SAVED IN " + imgName)
    
    return imgName

 # Defining StopCAM() to stop WEBCAM Preview
def StopCAM():
    # Stopping the camera using release() method of cv2.VideoCapture()
    second_frame.cap.release()
    global cam 
    cam = 0

    # Configuring the CAMBTN to display accordingly
    second_frame.CAMBTN.config(text="START CAMERA", command=StartCAM)

    # Displaying text message in the camera label
    second_frame.cameraLabel.config(text="CAMERA IS OFF", font=('Comic Sans MS',40))
    
    second_frame.CAMBTN.grid_forget()
    second_frame.cameraLabel.grid_forget()
    second_frame.saveLocationEntry.grid_forget()
    second_frame.browseButton.grid_forget()
    second_frame.captureBTN.grid_forget()

def StartCAM():
    second_frame.cameraLabel = Label(second_frame, bg="#641E16", borderwidth=3, relief="groove")
    second_frame.cameraLabel.grid(row=5, column=1, padx=10, pady=10, columnspan=2)
    

    second_frame.saveLocationEntry = Entry(second_frame, width=55, textvariable=destPath)
    second_frame.saveLocationEntry.grid(row=7, column=1, padx=10, pady=10)
    

    second_frame.browseButton = Button(second_frame, width=10, text="BROWSE", command=destBrowse)
    second_frame.browseButton.grid(row=7, column=2, padx=10, pady=10)
    

    second_frame.captureBTN = Button(second_frame, text="CAPTURE", command=Capture, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=20)
    second_frame.captureBTN.grid(row=8, column=1, padx=10, pady=10)
    

    second_frame.CAMBTN = Button(second_frame, text="STOP CAMERA", command=StopCAM, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=13)
    second_frame.CAMBTN.grid(row=8, column=2)
    # Creating object of class VideoCapture with webcam index
    second_frame.cap = cv2.VideoCapture(0)
    global cam
    cam = 1

    # Setting width and height
    width_1, height_1 = 640, 480
    second_frame.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    second_frame.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)

    # Configuring the CAMBTN to display accordingly
    second_frame.CAMBTN.config(text="STOP CAMERA", command=StopCAM)

    # Removing text message from the camera label
    second_frame.cameraLabel.config(text="")

    # Calling the ShowFeed() Function
    ShowFeed()

# def on_closing():
#     second_frame.cap.release()
#     second_frame.destroy()




def horizon_3(b):
    im = Image.open(b)
    im.save('input.jpg')
    h,w= im.size  
    print(h,w)
    left1 = 0
    top1 = 0
    right1 = h/3
    bottom1 = w
    im1 = im.crop((left1, top1, right1, bottom1))
    left2 = h/3
    top2 = 0
    right2 = 2*h/3
    bottom2 = w
    im2 = im.crop((left2, top2, right2, bottom2))
    left3 = 2*h/3
    top3 = 0
    right3 = h
    bottom3 = w
    im3 = im.crop((left3, top3, right3, bottom3))
    im1.save("test_1.jpg")
    im2.save("test_2.jpg")
    im3.save("test_3.jpg")
    return h/3, 2*h/3




def model_1(h_add_1, h_add_2):
    speak("Please speak")
    print("Please talk:")
    

    r=sr.Recognizer()
    #Specifing the microphone to be activated
    mic = sr.Microphone()

    #listening to the user 
    with mic as s:
        audio = r.record(s,duration=5)
        print("recorded")
        r.adjust_for_ambient_noise(s)

    #Converting the audio to text  
    try:
        """I use google engine to convert the speech 
         text but you may use other engines such as 
         sphinx,IBM speech to text etc."""
       	a = r.recognize_google(audio)
        #self.Input.config(text=speech)
        
        print(a)
    except sr.UnknownValueError:
        #calling the text to speech function
        speak("please try again,couldnt identify")
        return 0

    except sr.WaitTimeoutError as e:
        speak("please try again")
        return 0

    #with sr.Microphone() as source:
    #    audio_data=r.record(source,duration=5)
    #    print("recognizing")
    #    a=r.recognize_google(audio_data)
    #    print("Text: "+a)
    from nltk.tokenize import word_tokenize 
    import sys
    c=word_tokenize(a) 
    f = open("model_data/dataset_names.txt", "r")
    things = []

    for x in f:
        thing = str(x)
        thing = thing.replace("\n","")
        things.append(thing)

    print("Things:", things)
    output=[]
    for i in range(len(things)):
        for j in range(len(c)):
            if(c[j].lower()==things[i].lower()):
                output.append(things[i])

    for i in range(len(c)):
        if(c[i] == 'mobile' or c[i] == 'phone'):
            output.append("Mobile_phone")
        if(c[i] == 'alarm'):
            output.append("Alarm_clock")
        if(c[i] == 'backpack'):
            output.append("back_pack")
    print(output)


    import cv2 

    image_path = "input.jpg"
    image_path_1   = "test_1.jpg"
    image_path_2   = "test_2.jpg"
    image_path_3   = "test_3.jpg"
    video_path   = "./test.mp4"

    import shutil

    original = str(os.getcwd()) + image_path
    target = '../Model_2/TensorFlow-2.x-YOLOv3/' + image_path

    shutil.copyfile(original, target)

    input_labels = output
    position = ""
    if(len(input_labels)> 0):
        yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
        yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights
        image, coords, img_h, img_w, outputs = detect_image(yolo, image_path, "./IMAGES/0.jpg", input_labels, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
        image_1, coords_1, img_h_1, img_w_1, outputs_1 = detect_image(yolo, image_path_1, "./IMAGES/1.jpg", input_labels, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
        image_2, coords_2, img_h_2, img_w_2, outputs_2 = detect_image(yolo, image_path_2, "./IMAGES/2.jpg", input_labels, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
        image_3, coords_3, img_h_3, img_w_3, outputs_3 = detect_image(yolo, image_path_3, "./IMAGES/3.jpg", input_labels, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
        #image, coords,img_h,img_w, outputs = detect_image(yolo, image_path, "./IMAGES/plate_1_detect.jpg", input_labels, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
        #detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
        #detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
        found = 0

        if(len(coords) > 0):

            num_objs = len(coords)
            count = 0
            f = open("detections.txt","a")
            g = open("positions.txt", "a")
            for coordinates in coords:
                x1 = coordinates[0]
                x2 = coordinates[1]
                y1 = coordinates[2]
                y2 = coordinates[3]
                obj = outputs[0]
                
                position = str(outputs[count]) + " at "
                
                if(y2 <= img_h/2 or (y1 <= img_h/4 and y2 <= 3*img_h/4)):
                	position = position + "Top"
                elif (y1>img_h/4 and y2<3*img_h/4):
                	position = position + "Middle"
                else:
                	position = position + "Bottom"

                if(x2<=img_w/2 or (x1 <= img_h/4 and x2<= 3*img_w/4)):
                	position = position + " and left of image"
                elif(x1<img_w/4 and x2<3*img_w/4):
                	position = position + " and Center of image"
                else:
                	position = position + " and right of image"

                '''
				if(y2 <= img_h/2 or (y1 <= img_h/4 and y2 <= 3*img_h/4)):
					position = position + "Top"

				elif(y1>img_h/4 and y2<3*img_h/4):
					position = position + "Middle"
				
				else:
					position = position + "Bottom"
				

				if(x2<=img_w/2 or (x1 <= img_h/4 and x2<= 3*img_w/4)):
					position = position + " and Left of image"
				
				elif(x1<img_w/4 and x2<3*img_w/4):
					position = position + " and Center of image"
				
				else:
					position = position + " and Right of image"
				'''
                f.write(outputs[count] +  " " + str(x1) + " " + str(x2) + " " + str(y1) + " " + str(y2) + "\n")
                g.write(position + "\n")
                count = count + 1

            f.close()
            g.close()
            found = 1


        if(len(coords_1) > 0 and found == 0):

            num_objs = len(coords_1)
            count = 0
            f = open("detections.txt","a")
            g = open("positions.txt", "a")
            for coordinates in coords_1:
                x1 = coordinates[0]
                x2 = coordinates[1]
                y1 = coordinates[2]
                y2 = coordinates[3]
                obj = outputs_1[0]
                f.write(outputs_1[count] +  " " + str(x1) + " " + str(x2) + " " + str(y1) + " " + str(y2) + "\n")
                g.write(outputs_1[count] + " at left of image\n")
                count = count + 1

            f.close()
            g.close()
        if(len(coords_2) > 0 and found == 0):

            num_objs = len(coords_2)
            count = 0
        
            f = open("detections.txt","a")
            g = open("positions.txt", "a")
            for coordinates in coords_2:
                x1 = h_add_1 + coordinates[0]
                x2 = h_add_1 + coordinates[1]
                y1 = coordinates[2]
                y2 = coordinates[3]
                obj = outputs_2[0]
                f.write(outputs_2[count] +  " " + str(x1) + " " + str(x2) + " " + str(y1) + " " + str(y2) + "\n")
                g.write(outputs_2[count] + " at centre of image\n")
                count = count + 1

            f.close()
            g.close()
        
        if(len(coords_3) > 0 and found == 0):

            num_objs = len(coords_3)
            count = 0

            f = open("detections.txt","a")
            g = open("positions.txt", "a")
            for coordinates in coords_3:
                x1 = h_add_2 + coordinates[0]
                x2 = h_add_2 + coordinates[1]
                y1 = coordinates[2]
                y2 = coordinates[3]
                obj = outputs_3[0]
                

                f.write(outputs_3[count] +  " " + str(x1) + " " + str(x2) + " " + str(y1) + " " + str(y2) + "\n")
                g.write(outputs_3[count] + " at right of image\n")
                count = count + 1

            f.close()
            g.close()

        if(len(coords_1)==0 and len(coords_2)==0 and len(coords_3)==0 and len(coords) == 0):
            position = position + "No such items found"
        #detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)
    else:
        position = position + "No such items found"


    if(position == "No such items found"):
        myobj = gTTS(text=position, lang='en', slow=False)			
        myobj.save("welcome.mp3")
        image = Image.open(image_path)
        image.show()
        playsound.playsound('welcome.mp3', True)
        return 0
    else:
    	return 1



def draw_box(coords, label, image_path, output_path):

    image      = cv2.imread(image_path)
    if image is None:
    	print("exit")
    	return
    image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = image.shape
    bbox_color = (255,0,0)
    bbox_thick = int(0.6 * (image_h + image_w) / 1000)
    if bbox_thick < 1: bbox_thick = 1
    fontScale = 0.75 * bbox_thick

    x1 = int(float(coords[0]))
    x2 = int(float(coords[1]))
    y1 = int(float(coords[2]))
    y2 = int(float(coords[3]))
    top_left = (x1,y1)
    bot_right = (x2,y2)
    Text_colors = (255,255,255)
    cv2.rectangle(image, top_left, bot_right, bbox_color, bbox_thick*2)

    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                          fontScale, thickness=bbox_thick)
    # put filled text rectangle
    cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

    # put text above rectangle
    cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
    cv2.imwrite(output_path, image)



def model_2():

	f = open("model_data/dataset_names.txt", "r")
	things = []

	for x in f:
		thing = str(x)
		thing = thing.replace("\n","")
		things.append(thing)


	import cv2 


	image_path   = "input.jpg"
	video_path   = "./test.mp4"

	input_labels = things
	position = ""

	items = []

	output_path = "IMAGES/plate_1_detect.jpg"
	with open("../../TensorFlow-2.x-YOLOv3/detections.txt","r") as f:
		for obj in f:
			temp = obj.split()
			items.append(temp)

	if(len(input_labels)> 0):
		yolo = Load_Yolo_model()
		image, coords,img_h,img_w, outputs = detect_image(yolo, image_path, output_path, input_labels, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
	#detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
	#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

		print("No of objects: " + str(len(coords)) + "\n\n\n")

		num = len(coords)
		print("Num: ",num)
		if(len(coords) > 0):
			num_objs = len(coords)
			count = 0
			#im = Image.open(image_path)
			#im.show()

			engine = pyttsx3.init()
			rate = engine.getProperty('rate')   # getting details of current speaking rate
			engine.setProperty('rate', 150) 
			if(len(items) > 0):
				for item in items:
					IOU = []
					for coordinates in coords:

						x1 = coordinates[0]
						x2 = coordinates[1]
						y1 = coordinates[2]
						y2 = coordinates[3]
						obj = outputs[0]
						name = item[0]
						x1_ = float(item[1])
						x2_ = float(item[2])
						y1_ = float(item[3])
						y2_ = float(item[4])
						#if(x1_ > x1 and x2_ < x2):
						#	if(y1_ > y1 and y2_ < y2):
						#		position = str(name) + " on " + outputs[count]
						#	else:
						#		position = str(name) + " beside " + outputs[count]
						#else:
						#	if(y1_ > y1 and y2_ < y2):
						#		position = str(name) + " beside " + outputs[count]
						#print(position + "\n")
						#engine.say(position)
						#engine.runAndWait()

						xA = max(x1, x1_)
						yA = max(y1, y1_)
						xB = min(x2, x2_)
						yB = min(y2, y2_)
						# compute the area of intersection rectangle
						interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
						# compute the area of both the prediction and ground-truth
						# rectangles
						boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
						boxBArea = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
						# compute the intersection over union by taking the intersection
						# area and dividing it by the sum of prediction + ground-truth
						# areas - the interesection area
						iou = interArea / float(boxAArea + boxBArea - interArea)
						IOU.append(iou)
						count = count + 1

					max_iou = 0
					for i in range(len(IOU)):
						print(IOU[i], outputs[i])
						if IOU[i] >= max_iou:
							max_iou = IOU[i]
							object_closest = outputs[i]
							coords_2 = coords[i]
							print(coords_2)
					draw_box(item[1:],name,"input.jpg", "output.jpg")
					draw_box(coords_2,object_closest,"output.jpg", "output2.jpg")
					if(max_iou < 0.5):
						position = str(item[0]) + " closest to " + object_closest
					else:
						position = str(item[0]) + " on " + object_closest
					print(position)
					im = Image.open("output2.jpg")
					im.show()
					print("Hello")
					speak(position)
					print("Bye")
					#engine.runAndWait()
		#playsound.playsound('welcome.mp3', True)	
			else:
				position = position + "No such items found"
	#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)
		else:
			position = position + "No such items found"


	if(position == "No such items found"):
		if(num == 0):
			f = open("../../TensorFlow-2.x-YOLOv3/positions.txt","r")
			text = f.readlines()
			im = Image.open(image_path)
			im.show()
			for line in text:
			    num = num + 1
			    pos = line.strip()
			    print(pos)
			    myobj = gTTS(text=pos, lang='en', slow=False)			
			    myobj.save("pos" + str(num) + ".mp3")
			    playsound.playsound("pos" + str(num) + ".mp3", True)
			    os.remove("pos" + str(num) + ".mp3")
			f.close()
		else:
			myobj = gTTS(text=position, lang='en', slow=False)			
			myobj.save("welcome.mp3")
			image = cv2.imread(image_path)
			#Splaysound.playsound('welcome.mp3', False)
			cv2.imshow('image',image)
			cv2.waitKey(0)








def greet():
       #greets the user with a random phrase from A
        A=["Hi,nice to meet you","hello","Nice to meet you","hey,nice to meet you","good to meet you!"]
        b=random.choice(A)
        speak(b)
        
def tell_time():
    localtime = time.asctime(time.localtime(time.time()))
    a = localtime[11:16]
    speak(a)

def tell_day():
    localtime = time.asctime(time.localtime(time.time()))
    day = localtime[0:3]
    if day == "Sun":
        speak("it's sunday")
    if day == "Mon":
        speak("it's monday")
    if day == "Tue":
        speak("it's tuesday")
    if day == "Wed":
        speak("it's wednesday")
    if day == "Thu":
        speak("it's thursday")
    if day == "Fri":
        sspeak("it's friday")
    if day == "Sat":
        speak("it's saturday")

def tell_month():
    localtime = time.asctime(time.localtime(time.time()))
    m_onth = localtime[4:7]
    if m_onth == "Jan":
        speak("it's january")
    if m_onth == "Feb":
        speak("it's february")
    if m_onth == "Mar":
        speak("it's march")
    if m_onth == "Apr":
        speak("it's april")
    if m_onth == "May":
        speak("it's may")
    if m_onth == "Jun":
        speak("it's june")
    if m_onth == "Jul":
        speak("it's july")
    if m_onth == "Aug":
        speak("it's august")
    if m_onth == "Sep":
        speak("it's september")
    if m_onth == "Oct":
        speak("it's october")
    if m_onth == "Nov":
        speak("it's november")
    if m_onth == "Dec":
        speak("it's december")

def shut():
    #bids the user goodbye and quits
    A=random.choice(["bye", "good bye", "take care bye"])
    speak(A)
    exit()
        
def functions():
        speak("here is a list of what i can do")
        messagebox.showinfo("VBAE functions", "1.Try saying 'Hi','Hello'" +
                            "\n2.Try asking 'What day is this?'" +
                            "\n3.Try asking 'What month is it?'" +
                            "\n4.Try asking 'What time is it?'" +
                            "\n5.Try asking 'Find my (Bottle) ?"+
                            "\n6.To close say 'Bye' or 'Sleep' or 'See you later'")   
def on_closing():
        if(cam):
            second_frame.cap.release()
        root.destroy()
        
def Process():
    #second_frame.status.configure(text="listening")
    second_frame.va.configure(text = "")
    second_frame.status.configure(text = "")
    speech=str(speech_recog())
    A=["hi","hello","hey","hai","hey dream""hi dream","hello dream"]
    B=["what day is it","what day is today","what day is this"]
    C=["what month is it","what month is this"]
    D=["what time is it","what is the time","time please",]
    E=["bye","bye dream","shutdown","quit"]

    if speech=="what can you do":
        functions()  
        
    elif speech[0:4]=="find":
        #StartCAM()
        searching()
    elif speech in A:
        greet()

    elif speech =="who are you":
        speak("I'm your personal assistant")

    elif speech in B:
        tell_day()

    elif speech in C:
        tell_month()

    elif speech in D:
        tell_time()

    elif speech in E:
        shut()
    else:
        speak("I am sorry couldn't perform the task you specified")

def searching():
	filename = askopenfilename()
	
	if filename == "":
		speak("Image not selected")
		return

	import shutil

	#target = r'C:/Users/Harshith/Downloads/YOLO_dataset/testTensorFlow-2.x-YOLOv3/input.jpg'
	#target = str(os.getcwd()) + 'input.jpg'
	#shutil.copyfile(filename, target)

	h_add_1, h_add_2 = horizon_3(filename)
	success = model_1(h_add_1, h_add_2)
	#get_ipython().run_line_magic('cd', '..')
	#get_ipython().run_line_magic('cd', 'Model_2/TensorFlow-2.x-YOLOv3')
	print("ugduagfgd: ",success)
	if(success == 1):
		#os.chdir('C:/Users/Harshith/Downloads/YOLO_dataset/test/Model_2/TensorFlow-2.x-YOLOv3')
		os.chdir('../Model_2/TensorFlow-2.x-YOLOv3')
		model_2()
		
		#os.chdir('C:/Users/Harshith/Downloads/YOLO_dataset/test/TensorFlow-2.x-YOLOv3')
		os.chdir('../../TensorFlow-2.x-YOLOv3')
	if os.path.exists("detections.txt"):
		os.remove("detections.txt")
		os.remove("positions.txt")

# Creating object of tk class
root = tk.Tk()
root.title("Voice Assistant")

destPath = StringVar()
cam = 0

root.geometry("780x670")
root.resizable(True,True)
root.configure(background = "#641E16")

# Create A Main Frame
main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)

# Create A Canvas
my_canvas = Canvas(main_frame)
my_canvas.configure(background = "#641E16")
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

# Add A Scrollbar To The Canvas
my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)

# Configure The Canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))

# Create ANOTHER Frame INSIDE the Canvas
second_frame = Frame(my_canvas)
second_frame.configure(background = "#641E16")

# Add that New frame To a Window In The Canvas
my_canvas.create_window((0,0), window=second_frame, anchor="nw")


CreateWidgets()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()




path = r"C:/Users/Harshith/Downloads/YOLO_dataset/test/TensorFlow-2.x-YOLOv3/t1.jpg"





