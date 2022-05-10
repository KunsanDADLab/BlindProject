import RPi.GPIO as GPIO
import time
import cv2
import datetime
import pygame

pygame.init()
GPIO.setmode(GPIO.BOARD)



pins = (18,19,21)


def led(pins, color, t):
    RGBs = (
        (1,0,0),
        (0,1,0),
        (0,0,1)
    )

    GPIO.setup(pins[0], GPIO.OUT)
    GPIO.setup(pins[1], GPIO.OUT)
    GPIO.setup(pins[2], GPIO.OUT)

    GPIO.output(pins[0], RGBs[color][0])
    GPIO.output(pins[1], RGBs[color][1])
    GPIO.output(pins[2], RGBs[color][2])

    time.sleep(t)
    
    GPIO.cleanup(pins)


"""
led(pins, 0, 3)
led(pins, 1, 3)
led(pins,2,3)
"""

GPIO.setup(3,GPIO.IN)

a=0


while True:
    if GPIO.input(3)==0:
        print("button")
        capture=cv2.VideoCapture(0)
        width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)#width=int(capture.get(3))
        height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)#height=int(capture.get(4))
        while(capture.isOpened):            
            ret,frame=capture.read()
            frame=cv2.resize(frame,dsize=(640,480),interpolation=cv2.INTER_AREA)
            if ret==False:
                break
            cv2.imshow("VideoFrame",frame)
            cv2.waitKey(1)
            #now=datetime.datetime.now().strftime("%d_%H-%M-%S")
            #key=cv2.waitKey(33)
            
            if GPIO.input(3)==0:
                print("capture")
                cv2.IMREAD_UNCHANGED
                cv2.imwrite("/home/pi/captureig/"+str(a)+".png",frame)
                cv2.imshow('capture'+str(a),frame)
                print("capture success")
                a=a+1
                #f=open("/home/pi/test.txt","r")
                #line=f.readlines()
                #for i in line:
                    #led(pins,int(i),1)
                
        capture.release()
        cv2.destroyAllWindows()
        print("video stop")

