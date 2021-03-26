import sys
from utils.datasets import letterbox
import numpy as np
import queue
from threading import Thread

class LoadStdin:  # for inference
    def __init__(self, img_size=640, stride=32):
        # p = str(Path(path).absolute())  # os-agnostic absolute path
        # if '*' in p:
        #     files = sorted(glob.glob(p, recursive=True))  # glob
        # elif os.path.isdir(p):
        #     files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        # elif os.path.isfile(p):
        #     files = [p]  # files
        # else:
        #     raise Exception(f'ERROR: {p} does not exist')

        # images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        # videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        # ni, nv = len(images), len(videos)

        self.mode = 'stream'

        self.img_size = img_size
        self.stride = stride

        self.source = sys.stdin.buffer

        # Reading in 640x360 3 color
        self.width=640
        self.height=360
        self.depth = 3
        self.queue = queue.Queue()

        thread = Thread(target=self.update, daemon=True)
        print("Starting reader thread")
        thread.start()

        # self.files = images + videos
        # self.nf = ni + nv  # number of files
        # self.video_flag = [False] * ni + [True] * nv
        # self.mode = 'image'
        # if any(videos):
        #     self.new_video(videos[0])  # new video
        # else:
        #     self.cap = None
        # assert self.nf > 0, f'No images or videos found in {p}. ' \
        #                     f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def update(self):
        # Reading in 640x360 3 color
        try:
            while True:
                toread = self.width*self.height*self.depth
                data = bytearray(0)
                while toread > 0:
                    thisdata = self.source.read(toread)
                    if len(thisdata) <= 0:
                        raise Exception("stop")
                    data.extend(thisdata)
                    toread -= len(thisdata)
                self.queue.put(data)
        except:
            print("Done with reader thread")
            self.queue.put(None)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # if self.count == self.nf:
        #     raise StopIteration        return self.lastframe

        data = None

        # Strip pending data in the queue
        while not self.queue.empty():
            data = self.queue.get()
            if data is None:
                raise StopIteration

        skip = 0

        if data is not None:
            skip -= 1

        while data is None or skip > 0:
            data = self.queue.get()
            if data is None:
                raise StopIteration
            skip -= 1
        
        img0 = np.frombuffer(data, dtype=np.uint8)
        img0 = img0.reshape((self.height,self.width,self.depth))

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return 'stdin', img, img0, None

  
    def __len__(self):
        return 0