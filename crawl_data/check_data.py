import cv2
from tqdm import tqdm
import os
import threading
from PIL import Image
import time

def check_data(root_path, max_thread):
    file_path_list = os.listdir(root_path)
    workers = []
    p = 0

    for i, file_path in enumerate(file_path_list):
        workers.append(check_worker(os.path.join(root_path, file_path)))
    
    for i, worker in enumerate(workers):
        if i < max_thread:
            worker.start()
        else:
            p = min(i, max_thread)
            break

    while True:
        done_flag = True
        for worker in workers:
            if worker.flag is False:    # This worker has not compeleted job.
                done_flag = False
            elif worker.flag is True:   # This worker has completed job and has not been detected.
                worker.flag = None     
                if p < len(workers):
                    workers[p].start()
                    p = p + 1
        if done_flag is True:
            break
        time.sleep(0.1)
    print('done')

class check_worker(threading.Thread):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.flag = False
    
    def run(self):
        img_list = sorted(os.listdir(self.file_path))
        for img_name in img_list:
            flag = is_valid_image(os.path.join(self.file_path, img_name))
            if flag is not True:
                print('Broken: ' + os.path.join(self.file_path, img_name))
                os.remove(os.path.join(self.file_path, img_name))
        self.flag = True
        print(self.file_path + ' completed!')

def is_valid_image(path):
    try:
        bValid = True
        fileObj = open(path, 'rb')
        buf = fileObj.read()
        if not buf.startswith(b'\xff\xd8'):
            bValid = False
        elif buf[6:10] in (b'JFIF', b'Exif'):
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                bValid = False
        else:
            try:
                Image.open(fileObj).verify()
            except Exception as e:
                bValid = False
    except Exception as e:
        return False

    try:
        img = Image.open(path)
    except Exception as e:
        return False

    img = cv2.imread(path)
    if img is None:
        return False

    return bValid

if __name__ == '__main__':
    check_data('/opt/data/private/zli/data/Surf_Online', max_thread = 32)

