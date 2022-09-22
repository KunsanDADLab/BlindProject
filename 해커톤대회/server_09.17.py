# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import socket
import threading

class ServerSocket:
    def __init__(self, ip, port): # ip와 port를 매개변수로 받음
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.socketOpen() # 소켓 오픈
        ##self.receiveThread = threading.Thread(target=self.receiveImages) # 이미지를 받는 스레드
        ##self.receiveThread.start() # 스레드 시작

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

    def get_bytes_stream(self,sock,length):
        buffer=b''
        try:
            remain=length
            while True:
                data=sock.recv(remain)
                buffer+=data
                if len(buffer)==length:
                    break
                elif len(buffer)<length:
                    remain=length-len(buffer)
        except Exception as e:
            print(e)
        return buffer[:length]

    def receiveImages(self): # 이미지를 받는 스레드
        try:
            # TCP socket으로 이미지를 송수신할 때 가장 중요한 것은 클라이언트에서 서버로 해당 이미지 데이터의 크기를 같이 보내는 것이다. 
            # TCP socket을 사용해서 한 번에 보낼 수 있는 데이터의 크기는 제한되어 있으므로 이미지 데이터를 string으로 변환해서 보낼 때 이 크기가 얼마나 큰 지가 중요하다. 
            # 따라서, 이미지의 크기를 먼저 받고 그 크기만큼만 socket에서 데이터를 받아서 다시 이미지 데이터의 형태로 변환해야 한다.
            
            #결과 보내기
            len_bytes_string=bytearray(self.conn.recv(1024))[2:]
            print(len_bytes_string)
            len_bytes=len_bytes_string.decode("utf-8")
            length=int(len_bytes)

            img_bytes=self.get_bytes_stream(self.conn,length)
            img_path="./fromjava.jpg"
            with open(img_path,"wb") as writer:
                writer.write(img_bytes)
            print("파일 생성")
                
        except Exception as e:
            print(e)
            self.socketClose()
            self.socketOpen()
            ##self.receiveImages()
            ##self.receiveThread = threading.Thread(target=self.receiveImages) # 이미지를 받는 스레드
            ##self.receiveThread.start() # 스레드 시작

    def sendResult(self):
        try:
            #머신러닝 파일에서 저장된 파일 불러서 결과 안드로이드로 보내기
            f=open("./results.txt",'r')
            msg=f.read()
            print("자바에 보낸 결과 "+msg)
            #msg="1*2&3*4"
            data=msg.encode('utf-8')
            length=len(data)
            self.conn.sendall(length.to_bytes(20,byteorder='little'))
            self.conn.sendall(data)
        except Exception as e:
            print(e)
            self.socketClose()
            self.socketOpen()
            ##self.receiveThread = threading.Thread(target=self.receiveImages) # 이미지를 받는 스레드
            ##self.receiveThread.start() # 스레드 시작
        

    def recvall(self, sock, count):
        buf = b''
        while count: # count가 있으면
            newbuf = sock.recv(count) # 크기를 받아옴
            if not newbuf: return None # 크기가 없으면 종료
            buf += newbuf # 버퍼에 크기 추가
            count -= len(newbuf) # count에서 버퍼길이를 빼줌
        return buf # 버퍼 리턴


@torch.no_grad()
def run(
        weights=ROOT / 'best_n_22.09.01.pt',  # model.pt path(s)
        source=ROOT / 'fromjava.jpg',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images => BOOL:TRUE
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) #TYE BOOL
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    sc = ServerSocket("172.17.0.2", 8963)

    ##receiveThread = threading.Thread(target=sc.receiveImages) # 이미지를 받는 스레드
    ##receiveThread.start() # 스레드 시작
    while(True):
        print("receive start")
        sc.receiveImages()
        print("receive end")
        
        print("analyze start")
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                #reset result file
                results='0'
                f=open("./results.txt","w") #results 파일 0으로 초기화
                f.write(results)
                f.close()
                
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                print("problem start")
                if len(det):
                    print("problem issue")
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    print("problem end")
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    #juhyun editing
                    emergency = "scooter", "motorcycle", "bicycle", "car", "bus"
                    warining = "tree_trunk", "bollard_stainless", "bollard_marble", "bollard_rubber", "bench", "person", "grating"
                    #, "fire_hydrant", "bollard_u"
                    results='0'
                    #predict
                    for *xyxy, conf, cls in reversed(det):
                        label=names[int(cls)] #탐지된 객체의 이름을 가져옴
                        #객체의 좌표를 구함 / coordinate : 좌표
                        coordinate = xyxy
                        coordinate = coordinate[0], 640 - coordinate[1], coordinate[2], 640 - coordinate[3] #1사분면 좌표를 기준으로 작성 #현재 들어오는 사진들이 640
                        print("ALL : ",label, coordinate)
                        if coordinate[3] < 245: #245픽셀을 기준으로 아래에 있는 객체들을 탐지
                            if label in emergency:
                                print("Emergency!! : ", label, coordinate)
                                if results != '0':
                                    results += "&" + "E*" + str(label)
                                else: results = "E*" + str(label)
                            elif label in warining:
                                if (coordinate[0] + coordinate[2]) / 2 < 320: #X좌표를 기준으로 왼쪽 함수 f(x)에 대입을 할건지 오른쪽 함수 g(x)에 대입을 할건지 구분
                                    if coordinate[3] < (lambda x : ( 337 / 213 ) * x )(coordinate[2]): #lambda : f(x) / 진로방해 O
                                        print("Left Warning!! : ", label, coordinate)
                                        if results != '0':
                                            results += "&" + "W*" + str(label)
                                        else: results = "W*" + str(label)
                                else:
                                    if coordinate[3] < (lambda x : ( - 337 / 214 ) * x + ( 107840 / 107 ))(coordinate[0]): #lambda : g(x) / 진로방해 O
                                        print("Right Warining : ",label, coordinate)
                                        if results != '0':
                                            results += "&" + "W*" + str(label)
                                        else: results = "W*" + str(label)
                    print("analyzing end")
                    #saving results
                    results=str(results)
                    if results != '' :
                        f=open("./results.txt","w")
                        f.write(results)
                        f.close()
                    print(results)
                    

                    sc.sendResult()
                    print("send done")
                else :
                    results='0'
                    f=open("./results.txt","w") #results 파일 0으로 초기화
                    f.write(results)
                    f.close()
                    sc.sendResult()
                    

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best_n_22.09.01.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'fromjava.jpg', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)