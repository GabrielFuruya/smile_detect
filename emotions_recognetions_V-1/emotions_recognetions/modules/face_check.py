import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path,makedirs
from os.path import isfile, join    
from modules.utils import from_base64    
import os

class LoadImagem : 
    def __init__(self):
        pass
        
    
    
    def image_gray(self,imagem):
        imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        return imagem_gray
         
         
        
    def classificador_faces(self,image_gray):
        classificador = cv2.CascadeClassifier("./resources/modelo/Classificadorface/haarcascade_frontalface_default.xml")
        faces = classificador.detectMultiScale(image_gray, 1.3, 5)
        
        return faces
            
        
    def cut_faces(self,faces,imagem_anotada):
        faces_list = []
        for (x,y,w,h) in faces:
            cv2.rectangle(imagem_anotada, (x,y), (x+w, y+h), (255, 255, 0) ,2)
            imagem_roi = imagem_anotada[y:y+h, x:x+w]
            faces_list.append(cv2.cvtColor(imagem_roi, cv2.COLOR_RGB2BGR))
        return faces_list
            
            

    def padronizar_imagem(self,imagem_caminho):
        for  i, j in enumerate(imagem_caminho):
            cv2.imwrite("./convert/image_preproces/"+str(i)+".jpg", j) 
        
        
        lis = []
        for i in os.listdir("./convert/image_preproces/"):
            imagem = cv2.imread("./convert/image_preproces/"+i,cv2.IMREAD_GRAYSCALE)
            imagem = cv2.resize(imagem, (64, 64), interpolation=cv2.INTER_LANCZOS4)
            lis.append(imagem)
        return lis
           

    

    def save_to_faces_pattern(self,list_faces_treino):
        face_imagem = 0
        for arq in list_faces_treino:
            face_imagem +=1
       
            cv2.imwrite("./convert/image_pattern/" + str(face_imagem) + ".jpg",arq)
        return
        
    def run(self, base64):
        imagem_b = from_base64(base64)
        print('base64 TO img')
        imagem_c = self.image_gray(imagem_b)
        print('Img TO gray')
        imagem_a = self.classificador_faces(imagem_c)
        print('Img TO Classifier Face')
        imagem_d = self.cut_faces(imagem_a,imagem_b)
        print('Img TO Cut Faces')
        imagem_e = self.padronizar_imagem(imagem_d)
        print('Img TO Padronize')
        imagem_f = self.save_to_faces_pattern(imagem_e)
        print('Save Faces')
      

            
            
        

