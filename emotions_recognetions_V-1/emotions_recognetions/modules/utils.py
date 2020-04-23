import base64
import numpy as np
import cv2



def from_base64(base64_data):
    # print("aqui base64",base64_data)
    nparr = base64.b64decode(base64_data)
    # print("aqui npar1",nparr)
    nparr = np.fromstring(nparr,np.uint8)
        
    # print("aqui npar2",nparr)
    fr_ = cv2.imdecode(nparr, 1)
    #print("aqui fr", fr_)
    return fr_