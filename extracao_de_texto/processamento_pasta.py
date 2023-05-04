from PIL import Image
import cv2 as OpenCv
from pytesseract import pytesseract
import os

#Define path to tessaract.exe
path_to_tesseract = r'C:\\Users\\admin\\Desktop\\tesseract\\tesseract.exe'

#Define path to images folder
path_to_images = r'C:\\Users\\admin\\Desktop\\TCC\\extracao_de_texto\\imagens_extracao\\'

#Point tessaract_cmd to tessaract.exe
pytesseract.tesseract_cmd = path_to_tesseract

#Get the file names in the directory
for root, dirs, file_names in os.walk(path_to_images):
    #Iterate over each file name in the folder
    for file_name in file_names:
        #Open image with PIL
        img = Image.open(path_to_images + file_name)

        #Extract text from image
        text = pytesseract.image_to_string(img)

        print(text) 