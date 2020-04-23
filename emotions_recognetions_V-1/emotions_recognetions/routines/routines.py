from modules.face_check import LoadImagem
from modules.predict import Verification
import os
import shutil
import datetime
from modules.db import Database
class Classifie:
    def __init__(self):  
        pass
    
    def run(self,data,file_name):
        if os.path.exists("/home/gabriel/example-api/convert/image_preproces/"):
            shutil.rmtree("/home/gabriel/example-api/convert/image_preproces/")
    
        paste_pre = "/home/gabriel/example-api/convert/image_preproces/"
        if not os.path.exists(paste_pre):
            os.mkdir(paste_pre)
        
        if os.path.exists("/home/gabriel/example-api/convert/image_pattern/"):
            shutil.rmtree("/home/gabriel/example-api/convert/image_pattern/")

        paste_patte = '/home/gabriel/example-api/convert/image_pattern/'
        if not os.path.exists(paste_patte): 
            os.mkdir(paste_patte)

        model_smile = {}
        date = datetime.datetime.now()
        LoadImagem().run(data)
        # img_path = '/home/gabriel/example-api/resources/Others/Imagens/image_pattern/1.jpg'
        for j, i in enumerate(os.listdir('/home/gabriel/example-api/convert/image_pattern')):
            print('Load ML model')
            print('pegeueiimg', i)
            model_smile[i], array = Verification().smile('/home/gabriel/example-api/convert/image_pattern/'+i)
            Database().insert_db(file_name,date.__str__(),"face %s"%(j), str(array),model_smile[i])
        

        #smile_no_smile =  model_smile.smile(img_path)
        print ('AQUI É SILE' , model_smile)
        return model_smile  