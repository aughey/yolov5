import sys
import argparse
import time
import math
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.stdin_reader import LoadStdin
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import pymongo
from datetime import datetime,timezone
import queue
from threading import Thread
import uuid
from elasticsearch import Elasticsearch
import requests
import json


http_server = "http://127.0.0.1:5000"

#mongo_client = pymongo.MongoClient("mongodb://" + "127.0.0.1" + ":27017/")
#db = mongo_client["yolov5"]

database_queue = queue.Queue()

def save(table,data):
    database_queue.put([table,data])

def database_writer():
    print("database writer starting")
    while True:
        table,data = database_queue.get()
        headers = {'Content-type': 'application/octet-stream'}
        try:
            if data is None:
                break

            if 'frame' in data:
                #print("Encoding image")
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                frame = cv2.imencode('.jpg', data['frame'], encode_param)[1].tobytes()
                requests.post(http_server + "/set/frame", data = frame, headers = headers)
                requests.post(http_server + "/set/" + data['uuid'], data = frame, headers = headers)
                del data['frame']

            #print("Posting to database " + table)
            requests.post(http_server + "/message/" + table, data = json.dumps(data), headers=headers)
        except:
            print("Caught exception",sys.exc_info()[0])

    print("Done with database writer")

databasethread = Thread(target=database_writer, daemon=True)
print("Starting database thread")
databasethread.start()

class DistMetric:
    def __init__(self,xy,kind):
        self.xy = xy
        self.kind = kind

    def SameKind(self,other):
        return self.kind == other.kind
    
    def closeTo(self,other):
        if not self.SameKind(other):
            return False
        dx = self.xy[0] - other.xy[0]
        dy = self.xy[1] - other.xy[1]
        dist = 640 * math.sqrt(dx*dx+dy*dy)
        return dist < 10 # 10 pixel closeness

class StaticObj:
    def __init__(self,metric,data,t):
        self.data = data;
        self.last_seen = t
        self.first_seen = t
        self.static = False
        self.metric = metric
        self.last_write = None
    
    def needsWriting(self,t):
        if self.last_write is None or t > self.last_write + 10:
            self.last_write = t
            return True
        else:
            return False
    
    def isStatic(self):
        return self.static
    
    def age(self,t):
        return t - self.first_seen
    
    def tooOld(self,t):
        return (t - self.last_seen) > 5
    
    def oldEnough(self,t):
        return self.age(t) > 5
    
    def closeToMetric(self,metric):
        return metric.closeTo(self.metric)
    
    def Refresh(self,t):
        self.last_seen = t
    
    def Promote(self,t):
        if not self.isStatic() and self.oldEnough(t):
            self.static = True
            return True
        else:
            return False
    

class StaticCache:
    def __init__(self):
        self.static_objects = []
        self.close_dist = 10
        self.static_time = 5
    
    def stats(self):
        total = len(self.static_objects)
        static = len(list(filter(lambda v: v.isStatic(), self.static_objects)))
        return "Total count: " + str(total) + ", static: " + str(static)
    
    def Add(self,metric,data,t):
        for obj in self.static_objects:
            if obj.closeToMetric(metric):
                obj.Refresh(t)
                return obj
        # add it
        obj = StaticObj(metric,data,t)

        self.static_objects.append(obj)
        return obj

    def Promote(self,t):
        promoted = []
        for obj in self.static_objects:
            if obj.Promote(t):
                promoted.append(obj)
        
        toremove = []
        for obj in self.static_objects:
            if obj.tooOld(t):
                toremove.append(obj)
        
        for obj in toremove:
            self.static_objects.remove(obj)
            
        return promoted
          
static_objects = StaticCache()


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    elif source == "stdin":
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStdin(img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        starttime = time.time()
        thistime = datetime.now(timezone.utc).isoformat()

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                had_detection = False
                frame_id = str(uuid.uuid4())

                # Write results
                #interesting_labels = ['truck','car','person','dog','bicycle','motorbike','cat','backpack','handbag']
                original_image = im0.copy()

                detections = []

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    label = f'{names[int(cls)]} {conf:.2f}'
                    line = (cls, label, *xywh)  # label format
                    
                    labelname = f'{names[int(cls)]}'
                    
                    # area = (640*xywh[2]) * (640*xywh[3])
                    # if area < 20*20:
                    #     continue
                    
                    detections.append( {
                        'label': labelname,
                        'conf': conf.item(),
                        'xywh': xywh,
                    })

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    

                    print(line)

                #static_objects.Promote(starttime)
                #print("Static object stats: " + static_objects.stats())

                if len(detections) > 0:
                    save('yolov5',{
                        'proc_time':t2 - t1,
                        'uuid':frame_id,
                        'time': thistime,
                        'detections': detections,
                        'frame':original_image
                    })


            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
