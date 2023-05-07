import pytesseract
import cv2
from pathlib import Path
import numpy as np
import csv
import random
import os
from os import walk


files = []
path = 'C:\\Users\\admin\\Desktop\\TCC\\extracao_de_texto\\imagens_extracao\\'
for (dirpath, dirnames, filenames) in walk(path):
  files.extend(filenames)
  break
t = len(files)
for i in range(t):
    kernel = np.ones((1,1), np.uint8)
    file_path = 'C:\\Users\\admin\\Desktop\\TCC\\extracao_de_texto\\imagens_extracao\\'+files[i]+''
    file_name = Path(file_path).stem
    img = cv2.imread('C:\\Users\\admin\\Desktop\\TCC\\extracao_de_texto\\imagens_extracao\\'+files[i]+'')
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.dilate(img, kernel, iterations = 1)

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\admin\\Desktop\\tesseract\\tesseract.exe'

    txt = pytesseract.image_to_string(img,lang = 'eng')
    txt = txt[:-1]
    txt = txt.replace('\n',' ')

    with open('data-set.csv', 'a', newline='') as csvfile:
        fieldnames = ['id','title', 'text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'id': random.randint(1,10000), 'title': file_name, 'text': txt, 'label':'TRUE'})
    