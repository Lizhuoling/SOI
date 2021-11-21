# coding:utf-8
# Collect the data of given keywords.

import re
import sys
import urllib
import pdb
import requests
import os
import cv2

class Collect_Given_Keyword():
    '''Collect images online with the given keyword and save the collected images to the specified path.
    The images are downloaded from Baidu Image.'''
    def __init__(self, keyword, max_num, save_path, job_index,
            headers = {"User-Agent" : "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;"},
            timeout = 3):
        self.keyword = keyword
        self.max_num = max_num      # The maximum number of images downloaded.
        self.save_path = save_path  # The path of saving the downloaded images.
        self.headers = headers
        self.timeout = timeout      # The maximum time of downloading an image.
        self.img_cnt = 0            # The number of images having been downloaded.
        self.job_index = job_index

    def get_onepage_urls(self, this_page_url):
        '''Return all the urls of the found images in this page and the url of the next page.'''
        if not this_page_url:
            print('This has been the last page.')
            return [], '' 
        try:
            html = requests.get(this_page_url, headers = self.headers)
            html.encoding = 'utf-8'
            html = html.text
        except Exception as e:
            print(e)
            pic_urls = []
            fanye_url = ''
            return pic_urls, fanye_url
        pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
        fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
        fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
        return pic_urls, fanye_url

    def download_onepage_pic(self, pic_urls):
        """Download all the images in this page given the URL."""
        for i, pic_url in enumerate(pic_urls):
            try:
                pic = requests.get(pic_url, timeout = self.timeout)
                img_name = 'category' + str(self.job_index) + '_' + str(self.img_cnt) + '.jpg'
                self.img_cnt += 1
                string = os.path.join(self.save_path, img_name)
                with open(string, 'wb') as f:
                    f.write(pic.content)
                    print('Download image %s successfully: %s' % (str(self.img_cnt), str(pic_url)))
            except Exception as e:
                print('Fail to download image %s: %s' % (str(self.img_cnt), str(pic_url)))
                print(e)
                continue

    def download_images(self):
        init_url = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word=' 
        init_url += urllib.parse.quote(self.keyword, safe='/')
        pic_urls, next_page_url = self.get_onepage_urls(init_url)
        Page_cnt = 0
        while True:
            Page_cnt += 1
            print('Download the ' + str(Page_cnt) + ' page.')
            self.download_onepage_pic(pic_urls)
            pic_urls, next_page_url = self.get_onepage_urls(next_page_url)
            if next_page_url == '' and pic_urls == []:
                break
            elif self.img_cnt > self.max_num:
                break
        print('done!')

    def Check_img(self):
        '''Check the quality of the downloaded images and remove the broken images.'''
        img_list = os.listdir(self.save_path)
        print('Check image quality.')
        for i, img_name in enumerate(img_list):
            quality_flag = self.is_valid_image(os.path.join(self.save_path, img_name))
            if quality_flag is not True:
                os.remove(os.path.join(self.save_path, img_name))
        print('done!')

    def is_valid_image(self, path):
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
                    #print(e)
        except Exception as e:
            return False
        return bValid

            

if __name__ == '__main__':
    collector = Collect_Given_Keyword('tree', 100, '../test')
    collector.download_images()
    collector.Check_img()
