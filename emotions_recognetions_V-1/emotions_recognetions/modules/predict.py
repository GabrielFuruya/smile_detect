import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os

class Verification:
    def __init__(self):
        #if not os.path.exists('/home/gabriel/example-api/resources/modelo/Machine_learning/weights.hdf5'):
        print('Start Create Model')
        self.model = Sequential()
        self.model.add(Conv2D(32, (3,3), input_shape = (64,64,1), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(32, (3,3), input_shape = (64,64,1), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
                                                                                                                                                                                                                                                                                                                                    
        self.model.add(Dense(1, activation = 'sigmoid'))

        print('Create Model')
        self.model.load_weights('/home/gabriel/example-api/resources/modelo/Machine_learning/weights.hdf5')
        print('Load Model')

    def smile(self, img_path):
        print('Load Image')
        print(img_path)
        img = image.load_img(img_path, target_size = (64,64) ,color_mode="grayscale")
        x = image.img_to_array(img)
        x2 = np.expand_dims(x, axis=0)
        print('Before Predict')
        pred = self.model.predict(x2)
        print('Predict Model')
        print(pred)
        if pred[0][0] == 1:
            result = "SMILE"
        else:   
            result = "NO-SMILE"
        print(result)


        return result, x
                
            

    