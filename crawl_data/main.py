# coding:utf-8
# Collect the data of the required categories under multi-thread way.

import threading
import time
import os
import pdb
import shutil

from collect_data import Collect_Given_Keyword

class Manage_workers():
    def __init__(self, keywords, max_number, worker_number, save_path):
        self.keywords = keywords # keywords should be a list.
        if type(max_number) != int:
            raise Exception('{max_number} should be an interger.')
        self.max_number = max_number    # The maximum downloaded images for each keyword.
        self.worker_number = worker_number  # The number of threads for loading data.
        self.save_path = save_path # The path for saving all the data.
        self.worker_list = []   # The list to save the running workers
        self.job_index = 0  # Indicate the next job index waiting for being done.

        if self.worker_number > len(self.keywords):
            self.worker_number = len(self.keywords)

        for i, keyword in enumerate(self.keywords):
            if os.path.exists(os.path.join(self.save_path, 'category' + str(i))):
                shutil.rmtree(os.path.join(self.save_path, 'category' + str(i)))
            os.makedirs(os.path.join(self.save_path, 'category' + str(i)))
                

    def main(self):
        # Initialize worker_number workers.
        for i in range(self.worker_number):
            self.worker_list.append(one_worker(self.keywords[self.job_index], self.max_number, os.path.join(self.save_path, 'category' + str(self.job_index)), self.job_index))
            print('Start job' + str(self.job_index))
            self.job_index += 1
        for worker in self.worker_list:
            worker.start()

        #while self.job_index < len(self.keywords):
        while len(self.worker_list) != 0:
            time.sleep(0.1)
            # Check the status of every worker.
            for i, worker in enumerate(self.worker_list):
                if worker.status == False:
                    del self.worker_list[i]
                    print('Close a job')
                    if self.job_index < len(self.keywords):
                        self.worker_list.append(one_worker(self.keywords[self.job_index], self.max_number, os.path.join(self.save_path, 'category' + str(self.job_index)), self.job_index))
                        self.job_index += 1
                        self.worker_list[-1].start()
                        print('Start job ' + str(self.job_index))
                    break
                    
        

class one_worker(threading.Thread):
    def __init__(self, keyword, max_num, save_path, job_index):
        super().__init__()
        self.status = True  # False indicates this thread has been completed.
        self.job = Collect_Given_Keyword(keyword, max_num, save_path, job_index)

    def run(self):
        self.job.download_images()
        self.job.Check_img()
        self.status = False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Crawl data from the Internet.')
    parser.add_argument('--target_path', type = str, default = './data',
                    help='The path to save collected data.')
    parser.add_argument('--num_per_class', type = int, default = 3000,
                    help = 'The number of collected images per class.')
    
    args = parser.parse_args()

    from category import Category_list
    print('Category number: ' + str(len(Category_list)))
    a = Manage_workers(Category_list, max_number = args.num_per_class, worker_number = 32, save_path = args.target_path)
    a.main()
   

