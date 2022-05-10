import socket
import threading
import numpy 
import cv2
# import datetime
# import time
import base64

class ServerSocket:
    
    def __init__(self, ip, port): # ip와 port를 매개변수로 받음
        self.TCP_IP = ip # ip설정
        self.TCP_PORT = port # port 설정
        self.socketOpen() # 소켓 오픈
        self.receiveThread = threading.Thread(target=self.receiveImages) # 이미지를 받는 스레드
        self.receiveThread.start() # 스레드 시작

    def socketClose(self): # 소켓을 닫음
        self.sock.close()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is close')

    def socketOpen(self): # 소켓을 엶
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.sock.bind((self.TCP_IP, self.TCP_PORT)) # 서버 바인드
        self.sock.listen(1) # 클라이언트는 일단 1명으로 설정
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is open')
        self.conn, self.addr = self.sock.accept()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is connected with client')

    def receiveImages(self): # 이미지를 받는 스레드

        try:
            while True:
                # TCP socket으로 이미지를 송수신할 때 가장 중요한 것은 클라이언트에서 서버로 해당 이미지 데이터의 크기를 같이 보내는 것이다. 
                # TCP socket을 사용해서 한 번에 보낼 수 있는 데이터의 크기는 제한되어 있으므로 이미지 데이터를 string으로 변환해서 보낼 때 이 크기가 얼마나 큰 지가 중요하다. 
                # 따라서, 이미지의 크기를 먼저 받고 그 크기만큼만 socket에서 데이터를 받아서 다시 이미지 데이터의 형태로 변환해야 한다.
                
                length = self.recvall(self.conn, 64) # 이미지 크기를 스트링으로 변환된 것을 받아옴
                length1 = length.decode('utf-8') # utf-8로 해독함
                stringData = self.recvall(self.conn, int(length1)) # base64로 인코딩된 stringData를 받음
                #stime = self.recvall(self.conn, 64)
                #print('send time: ' + stime.decode('utf-8'))
                # now = time.localtime()
                # print('receive time: ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
                
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8) 
                # 바이너리 파일을 읽어옴 (바이너리 파일로 변환한다가 맞을듯)
                # pram1 : base64로 인코딩된 stringData를 다시 string형태로 디코딩
                # pram2 : 부호없는 정수(양수) 0 ~ 255 = 2^8
                
                decimg = cv2.imdecode(data, 1) # 바이너리로 변환된 파일을 디코딩
                cv2.imshow("image", decimg) # 이미지 보여주기
                cv2.waitKey(1) # 0.001초 대기, ms단위임 1000이면 1초, 근데 0이면 무한대기
        except Exception as e:
            print(e)
            self.socketClose()
            cv2.destroyAllWindows()
            self.socketOpen()
            self.receiveThread = threading.Thread(target=self.receiveImages)
            self.receiveThread.start()

    def recvall(self, sock, count):
        buf = b''
        while count: # count가 있으면
            newbuf = sock.recv(count) # 크기를 받아옴
            if not newbuf: return None # 크기가 없으면 종료
            buf += newbuf # 버퍼에 크기 추가
            count -= len(newbuf) # count에서 버퍼길이를 빼줌
        return buf # 버퍼 리턴

def main():
    server = ServerSocket('202.31.147.213', 8963)

if __name__ == "__main__":
    main()