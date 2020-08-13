import time
import threading
import cv2
import paho.mqtt.client as mqtt
import numpy as np
import pycuda.autoinit
from utils.yolo_classes import get_cls_dict
from utils.yolo import TrtYOLO
from utils.visualization import BBoxVisualization

class TrtThread(threading.Thread):

    def __init__(self, model, conf_th):
        threading.Thread.__init__(self)
        self.model = model
        self.conf_th = conf_th
        self.trt_yolo = None   # to be created when run
        self.running = False

    def run(self):
        global image
        global cnbr
        print('TrtThread: loading the TRT YOLO engine...')
        self.trt_yolo = TrtYOLO(self.model, INPUT_HW)
        print('TrtThread: start running...waiting for images to analyze')
        self.running = True
        while self.running:
            if image is not None:
                print('Found image from camera: ', cnbr)
                img = image
                image = None
                tic = time.time()
                boxes, confs, clss = self.trt_yolo.detect(img, self.conf_th)
                toc = time.time()
                print('Detection time: ', (toc - tic))
                (H, W) = img.shape[:2]
                cls_dict = get_cls_dict(80)
                vis = BBoxVisualization(cls_dict)
                img = vis.draw_bboxes(img, boxes, confs, clss)

                # prepare & send image via mqtt
                client_thr = mqtt.Mosquitto()
                client_thr.connect("192.168.5.20", 1883, 60)
                client_thr.loop_start()
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                img_str = cv2.imencode('.jpg', img)[1].tostring()
                client_thr.publish("trt_yolo/result", img_str, 1, False)
                client_thr.loop_stop()
                client_thr.disconnect()
        del self.trt_yolo
        print('TrtThread: stopped...')

    def stop(self):
        self.running = False
        self.join()

def initmqtt(mqttThreadEvent):

    def on_connect(client, userdata, flags, rc):
        print("Connected to broker:", rc)
        client.subscribe("trt_yolo/detect/#", 0)

    def on_subscribe(client, userdata, mid, granted_qos):
        print('Subscribed to topics:', userdata, mid, granted_qos)


    def on_message(client, userdata, msg):
        global image
        global cnbr
        global th_abort
        #print (msg.payload)
        #print (len(msg.payload))
        if(len(msg.payload)>1000):
            cnbr = msg.topic.split('/')[2]
            # convert string of image data to uint8
            nparr = np.fromstring(msg.payload, np.uint8)
            # decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if msg.payload.decode('utf-8') == 'stop':
            th_abort = True

    client = mqtt.Mosquitto()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_subscribe = on_subscribe
    resp = client.connect("192.168.5.20", 1883, 60)
    client.loop_start()
    while th_abort == False:
        time.sleep(1)
    client.loop_stop()
    client.disconnect()
    print ("mqttThread terminated")
    exit()

INPUT_HW = (416, 416)
image = None
th_abort = False
cnbr = 0

mqttThreadEvent = threading.Event()
mqttThread = threading.Thread(
        target=initmqtt,
        args=(mqttThreadEvent,)
        )
mqttThread.start()

trt_thread = TrtThread('yolov4-tiny-416', conf_th=0.3)
trt_thread.start()  # start the child thread
while th_abort == False:
    time.sleep(1)
trt_thread.stop()   # stop the child thread

