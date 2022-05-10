import socket
import sys
import cv2
import numpy
import base64
#import datetime
import time

class ClientSocket:
    def __init__(self, ip, port):
        self.TCP_SERVER_IP = ip # ip we'll use
        self.TCP_SERVER_PORT = port # port we'll use
        self.connectCount = 0 # the num of connected clients
        self.connectServer()

    def connectServer(self):
        try:
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT)) # connect to Server
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0
            self.sendImages()
        except Exception as e:
            print(e)
            self.connectCount += 1
            if self.connectCount == 10: # max client count = 10
                print(u'Connect fail %d times. exit program'%(self.connectCount))
                sys.exit()
            print(u'%d times try to connect with server'%(self.connectCount))
            self.connectServer()

    def sendImages(self):
        cnt = 0
        capture = cv2.VideoCapture(0) # param : video's path
        #capture.set(cv2.CAP_PROP_FRAME_WIDTH)
        #capture.set(cv2.CAP_PROP_FRAME_HEIGHT)
        '''
        while True:
            ret, image = capture.read()
            image = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imshow('Image', image)
    
            if cv2.waitKey(30) > 0:
                break

        cv2.destroyAllWindows()
        '''
        
        try:
            
            while capture.isOpened():
                ret, frame = capture.read()
                resize_frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)

                #now = time.localtime()
                #stime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                result, imgencode = cv2.imencode('.jpg', resize_frame, encode_param)
                data = numpy.array(imgencode)
                stringData = base64.b64encode(data) # encode to base64
                length = str(len(stringData))
                
                self.sock.sendall(length.encode('utf-8').ljust(64))
                self.sock.send(stringData) # send to server encoded StringData as base64
                #self.sock.send(stime.encode('utf-8').ljust(64))
                
                
                
        except Exception as e:
            print(e)
            self.sock.close()
            time.sleep(1)
            self.connectServer()
            self.sendImages()

def main():
    TCP_IP = '202.31.147.213'
    TCP_PORT = 8963
    client = ClientSocket(TCP_IP, TCP_PORT)

if __name__ == "__main__":
    main()