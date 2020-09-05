import time
import threading
import cv2
import paho.mqtt.client as mqtt
import numpy as np
import json
import pycuda.driver as cuda
import pycuda.autoinit

from utils.yolo_classes import get_cls_dict
from utils.yolo_with_plugins import TrtYOLO
from utils.visualization import BBoxVisualization

MQTT_SERVER = '192.168.5.20'
MQTT_DETECT_TOPIC = 'trt_yolo/detect/+'
MQTT_RESULT_TOPIC_BASE = 'trt_yolo/result'

INPUT_HW = (288, 288)
YOLOV4_MODEL = 'yolov4-288'
CONFIDENCE_THRESHOLD = 0.83

# https://interviewbubble.com/typeerror-object-of-type-float32-is-not-json-serializable/
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class TrtThread(threading.Thread):

    def __init__(self, model, conf_th):
        threading.Thread.__init__(self)
        self.model = model
        self.conf_th = conf_th
        self.trt_yolo = None   # to be created when run
        self.cuda_ctx = None  # to be created when run
        self.running = False

    def run(self):
        global image

        print('TrtThread: loading the TRT YOLO engine...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_yolo = TrtYOLO(self.model, INPUT_HW)
        print('TrtThread: start running...waiting for images to analyze')
        self.running = True
        while self.running:
            if image is not None:
                #print('Found image with msg_id: ', msg_id)
                img = image
                image = None

                tic = time.time()
                boxes, confs, clss = self.trt_yolo.detect(img, self.conf_th)
                toc = time.time()
                inference_time = toc - tic

                (H, W) = img.shape[:2]
                cls_dict = get_cls_dict(80)
                vis = BBoxVisualization(cls_dict)
                img = vis.draw_bboxes(img, boxes, confs, clss)

                # Prepare and send result via MQTT
                classes_topic = MQTT_RESULT_TOPIC_BASE + '/' + msg_id + '/classes'
                image_topic = MQTT_RESULT_TOPIC_BASE + '/' + msg_id + '/image'

                class_names = list(map(lambda x: cls_dict.get(x), clss))
                class_score_tuples = list(zip(class_names, confs))

                detections = []
                for class_name, score in class_score_tuples:
                    score = round(score, 2)
                    detections.append({'class': class_name, 'score': score})

                print('Processed msg_id {0:15s} in {1:.2f}s, classes: {2}'
                        .format(msg_id, inference_time, detections))

                classes_payload = json.dumps(detections, cls=NumpyEncoder)

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                image_payload = cv2.imencode('.jpg', img)[1].tostring()

                client_thr = mqtt.Mosquitto()
                client_thr.connect(MQTT_SERVER, 1883, 60)
                client_thr.loop_start()

                client_thr.publish(image_topic, image_payload, 1, False)
                client_thr.publish(classes_topic, classes_payload, 1, False)

                client_thr.loop_stop()
                client_thr.disconnect()
        del self.trt_yolo
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.running = False
        self.join()

def initmqtt(mqttThreadEvent):

    def on_connect(client, userdata, flags, rc):
        print("Connected to broker:", rc)
        client.subscribe(MQTT_DETECT_TOPIC, 0)

    def on_subscribe(client, userdata, mid, granted_qos):
        print('Subscribed to topics:', userdata, mid, granted_qos)


    def on_message(client, userdata, msg):
        global image
        global msg_id
        global th_abort
        #print (msg.payload)
        #print (len(msg.payload))
        if(len(msg.payload)>1000):
            msg_id = msg.topic.split('/')[2]
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
    resp = client.connect(MQTT_SERVER, 1883, 60)
    client.loop_start()
    while th_abort == False:
        time.sleep(1)
    client.loop_stop()
    client.disconnect()
    print ("mqttThread terminated")
    exit()

image = None
th_abort = False
msg_id = None

mqttThreadEvent = threading.Event()
mqttThread = threading.Thread(
        target=initmqtt,
        args=(mqttThreadEvent,)
        )
mqttThread.start()

cuda.init()  # init pycuda driver

trt_thread = TrtThread(YOLOV4_MODEL, conf_th=CONFIDENCE_THRESHOLD)
trt_thread.start()  # start the child thread
while th_abort == False:
    time.sleep(1)
trt_thread.stop()   # stop the child thread

