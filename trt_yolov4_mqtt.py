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
MQTT_DETECT_TOPIC = 'trt_yolo/detect/+/+'
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

class TrtThread(threading.Thread, threading.Condition):

    def __init__(self, model, condition, client, conf_th):
        threading.Thread.__init__(self)
        self.model = model
        self.conf_th = conf_th
        self.condition = condition
        self.client = client
        self.trt_yolo = None   # to be created when run
        self.cuda_ctx = None  # to be created when run

    def run(self):
        global image
        global th_abort

        print('TrtThread: loading the TRT YOLO engine...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_yolo = TrtYOLO(self.model, INPUT_HW)

        print('TrtThread: start running...waiting for images to analyze')

        while not th_abort:
            with self.condition:
                self.condition.wait()
            if image is not None:
                with self.condition:
                    img = image
                    image = None
                    _cam_id = cam_id
                    _msg_id = msg_id
                    self.condition.notify()

                tic = time.time()
                boxes, confs, clss = self.trt_yolo.detect(img, self.conf_th)
                toc = time.time()
                inference_time = toc - tic

                cls_dict = get_cls_dict(80)
                vis = BBoxVisualization(cls_dict)
                img = vis.draw_bboxes(img, boxes, confs, clss)

                # Prepare and send result via MQTT
                classes_topic = MQTT_RESULT_TOPIC_BASE + '/' + _cam_id + '/' + _msg_id + '/classes'
                image_topic = MQTT_RESULT_TOPIC_BASE + '/' + _cam_id + '/' + _msg_id + '/image'

                class_names = list(map(lambda x: cls_dict.get(x), clss))
                class_score_tuples = list(zip(class_names, confs))

                detections = []
                for class_name, score in class_score_tuples:
                    score = round(score, 2)
                    detections.append({'class': class_name, 'score': score})

                print('Processed msg_id {1:15s} from {0:7s} in {2:.2f}s, result: {3}'
                        .format(_cam_id, _msg_id, inference_time, detections))

                classes_payload = json.dumps(detections, cls=NumpyEncoder)

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                image_payload = cv2.imencode('.jpg', img)[1].tostring()

                self.client.publish(image_topic, image_payload, 1, False)
                self.client.publish(classes_topic, classes_payload, 1, False)

        self.cuda_ctx.pop()
        del self.trt_yolo
        del self.cuda_ctx
        print('TrtThread: stopped...')
        exit()


class MqttThread(threading.Thread, threading.Condition):

    def __init__(self, condition, client):
        threading.Thread.__init__(self)
        self.condition = condition
        self.client = client
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        resp = self.client.connect(MQTT_SERVER, 1883, 60)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected to broker:", rc)
        client.subscribe(MQTT_DETECT_TOPIC, 0)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print('Subscribed to topics:', userdata, mid, granted_qos)

    def on_message(self, client, userdata, msg):
        #print('got msg ' + msg.topic)
        global image
        global msg_id
        global cam_id
        global th_abort

        if (len(msg.payload) > 1000):
            with self.condition:
                # get id's from topic
                cam_id = msg.topic.split('/')[2]
                msg_id = msg.topic.split('/')[3]
                # convert string of image data to uint8
                nparr = np.fromstring(msg.payload, np.uint8)
                # decode image
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.condition.notify()
        if msg.payload.decode('utf-8') == 'stop':
            th_abort = True

    def run(self):
        global th_abort
        while not th_abort:
            time.sleep(1)
        self.client.loop_stop()
        self.client.disconnect()
        print ("MqttThread: stopped...")
        with self.condition:
            self.condition.notifyAll()
        exit()


th_abort = False
image = None
msg_id = None
cam_id = None

cuda.init()  # init pycuda driver
client = mqtt.Mosquitto()

condition = threading.Condition()
mqtt_thread = MqttThread(condition, client,)
mqtt_thread.start()
trt_thread = TrtThread(YOLOV4_MODEL, condition, client, CONFIDENCE_THRESHOLD,)
trt_thread.start()

