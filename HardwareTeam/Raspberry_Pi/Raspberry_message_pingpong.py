import RPi.GPIO as GPIO
import socket
import sys
import cv2
import numpy
import base64
import time

class ClientSocket:
    def __init__(self, ip, port):
        self.TCP_SERVER_IP = ip # ip we'll use
        self.TCP_SERVER_PORT = port # port we'll use
        self.connectCount = 0 # the num of connected clients
        self.connectServer()

    def connectServer(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Original :=> self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT)) # connect to Server
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0
            self.sendImages()
        except Exception as e:
            print(e)
            self.connectCount += 1
            # if self.connectCount == 10: # max client count = 10
            #     print(u'Connect fail %d times. exit program'%(self.connectCount))
            #     sys.exit()
            print(u'%d times try to connect with server'%(self.connectCount))
            self.connectServer()

    def sendImages(self):
        receiveHandler = True
        capture=cv2.VideoCapture(0)
        width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)#width=int(capture.get(3))
        height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)#height=int(capture.get(4))

        while(capture.isOpened):
            ret, frame=capture.read()
            frame=cv2.resize(frame,dsize=(640,480),interpolation=cv2.INTER_AREA)
            if ret==False:
                break
            cv2.imshow("VideoFrame",frame)
            cv2.waitKey(1)
            
            if GPIO.input(3)==0:
                try:
                    if receiveHandler:
                        ret, frame = capture.read()
                        resize_frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)

                        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                        result, imgencode = cv2.imencode('.jpg', resize_frame, encode_param)
                        data = numpy.array(imgencode)
                        stringData = base64.b64encode(data) # encode to base64
                        length = str(len(stringData))
                        
                        self.sock.sendall(length.encode('utf-8').ljust(64))
                        self.sock.send(stringData) # send to server encoded StringData as base64
                        #self.sock.send(stime.encode('utf-8').ljust(64))
                        
                        # #CASE 2
                        # receiveHandler = False
                        
                        #CASE 1
                        processResult = self.recvall(self.sock, 64).decode('utf-8')
                        print(processResult)
                        #use 'if' for playing sounds mother fyckerssss
                        
                except Exception as e:
                    print(e)
                    self.sock.close()
                    time.sleep(1)
                    self.connectServer()
                    self.sendImages()
                    
            if GPIO.input(5)==0:
                print("button")
            
            # #CASE 2
            # if receiveHandler == False:
            #     processResult = self.recvall(self.sock, 64).decode('utf-8')
            #     print(processResult)
            #     receiveHandler = True
                
        capture.release()
        cv2.destroyAllWindows()
        print("video stop")

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

def main():
    TCP_IP = '202.31.147.213'
    TCP_PORT = 8963
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(3,GPIO.IN)
    GPIO.setup(5,GPIO.IN)
    
    client = ClientSocket(TCP_IP, TCP_PORT)

if __name__ == "__main__":
    main()