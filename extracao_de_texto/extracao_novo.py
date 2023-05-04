import pytesseract
import cv2
from pathlib import Path
import numpy as np
import csv

kernel = np.ones((1,1), np.uint8)
file_path = 'C:\\Users\\admin\\Desktop\\TCC\\extracao_de_texto\\imagens_extracao\\image_2.jpg'
file_name = Path(file_path).stem
img = cv2.imread('C:\\Users\\admin\\Desktop\\TCC\\extracao_de_texto\\imagens_extracao\\image_2.jpg')
img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)
img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
img = cv2.dilate(img, kernel, iterations = 1)
img = cv2.medianBlur(img, 3)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\admin\\Desktop\\tesseract\\tesseract.exe'
txt = pytesseract.image_to_string(img,lang = 'eng')
txt = txt[:-1]
txt = txt.replace('\n',' ')

with open('names.csv', 'w', newline='') as csvfile:
    fieldnames = ['imagem_titulo', 'texto', 'veracidade']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'imagem_titulo': file_name, 'texto': txt, 'veracidade':'0'})
   