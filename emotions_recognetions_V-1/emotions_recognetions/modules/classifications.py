
import os 
from glob import glob 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from shutil import copyfile,rmtree
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard




smile_txt = open("/home/gabriel/example-api/resources/Others/SMILE_list.txt", "r")
nosmile_txt = open("/home/gabriel/example-api/resources/Others/NON-SMILE_list.txt", "r")

all_images = glob('/home/gabriel/example-api/resources/Others/lfwcrop_grey/faces/*.pgm')

lista_smile = smile_txt.readlines()
lista_nosmile = nosmile_txt.readlines()

smile = []
nosmile = []
path = "/home/gabriel/example-api/resources/Others/lfwcrop_grey/faces"
for line in lista_smile:
        # remover a quebra de linha '/n' e extensão
        c = line[:-5]
        
        c = c+".pgm"
        
        if path + c in all_images:
            
        
            smile.append(c)
        
for line in lista_nosmile:
        # remover a quebra de linha '/n' e extensão
        c = line[:-5]
        
        c = c+".pgm"
        
        if path + c in all_images:
            
        
            nosmile.append(c)


ns = len(smile)
nns = len(nosmile)
print("Smile: %s, No-Smile: %s"  %(ns,nns ))


# ##  Smile train, teste, val 



# Smile train, teste, val 
n_train = int (ns - (ns * 0.3))
n_test = int (ns * 0.20)
n_val = int (ns * 0.10)

total = n_test + n_train + n_val

print("SMILE -->N Train: %s N test: %s, N val: %s" %(n_train, n_test, n_val))
print(total)




train_smile = smile[:n_train] #70%
test_smile = smile[n_train:n_train+n_test] #20%
val_smile = smile[n_train+n_test:] #10%

print("SMILE -->Train %s, Teste %s, Val %s" %(len(train_smile),len(test_smile), len(val_smile) ))


# ##  No-Smile train, teste, val 



# Smile train, teste, val 
n_train = int (nns - (nns * 0.3))
n_test = int (nns * 0.20)
n_val = int (nns * 0.10)

total = n_test + n_train + n_val

print("NOSMILE -->N Train: %s N test: %s, N val: %s" %(n_train, n_test, n_val))
print(total)




train_nosmile = nosmile[:n_train] #70%
test_nosmile = nosmile[n_train:n_train+n_test] #20%
val_nosmile = nosmile[n_train+n_test:] #10%

print("NOSMILE -->Train %s, Teste %s, Val %s" %(len(train_nosmile),len(test_nosmile), len(val_nosmile) ))


# # Cria pastas de Treinamento e Testes




try:  
    rmtree("/home/gabriel/example-api/resources/Others/lfwcrop_grey/data")
    rmtree("/home/gabriel/example-api/resources/Others/lfwcrop_grey/data2")  
    
except:
    pass
    

finally:
    

    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data") 
    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data2") # ira receber as imagens geradas pelo keras

    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/train")
    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/train/smile")
    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/train/nosmile")

    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/test")
    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/test/smile")
    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/test/nosmile")

    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/val")
    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/val/smile")
    os.mkdir("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/val/nosmile")


# # Copiando arquivos para as pasta de treinamento e teste
# 




for i in train_smile: 
    #print(i)
    copyfile("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/faces/"+i, "/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/train/smile/"+i[:-4]+".jpg")
    
for i in train_nosmile: 
    #print(i)
    copyfile("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/faces/"+i, "/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/train/nosmile/"+i[:-4]+".jpg")
    
    
for i in test_smile: 
    #print(i)
    copyfile("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/faces/"+i, "/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/test/smile/"+i[:-4]+".jpg")
    
for i in test_nosmile: 
    #print(i)
    copyfile("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/faces/"+i, "/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/test/nosmile/"+i[:-4]+".jpg")
    
for i in val_smile: 
    #print(i)
    copyfile("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/faces/"+i, "/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/val/smile/"+i[:-4]+".jpg")
    
for i in val_nosmile: 
    #print(i)
    copyfile("/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/faces/"+i, "/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/val/nosmile/"+i[:-4]+".jpg")


# # Gerando banco de dados de trainamento, teste e de validação
# 
# 



from keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,   # Altera de forma randômica as imagens horizontalmente (10% da largura total)
    height_shift_range = 0.1,  # Altera de forma randômica as imagens verticalmente (10% da altura total)
    zoom_range = 0.1,
    #horizontal_flip = True    # De forma randômica inverte imagens horizontalmente
    )
datagen_test = ImageDataGenerator(
    rescale = 1./255    
    )
    
    
datagen_val = ImageDataGenerator(
    rescale = 1./255
    )




data_train = datagen_train.flow_from_directory(
    '/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/train/',
    target_size=(64,64),
    batch_size=32,
    class_mode="binary",
    color_mode="grayscale",
    save_to_dir="/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data2",
    save_format='png',
    save_prefix='aug'
    
)

data_test = datagen_train.flow_from_directory(
    '/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/test/',
    target_size=(64,64),
    #batch_size=32,
    class_mode="binary",
    color_mode="grayscale"
)

data_val = datagen_train.flow_from_directory(
    '/home/gabriel/projeto/emotions_recognetions/lfwcrop_grey/data/val/',
    target_size=(64,64),
    #batch_size=32,
    class_mode="binary",
    color_mode="grayscale"
)


# ## Imprimindo imagens do banco de treinamento




x,y = data_train.next()

#fig = plt.figure()
#fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(1, 7):
    plt.subplot(2,3 ,i)
    image = x[i]     
    plt.imshow(image.reshape(64, 64), cmap=plt.get_cmap('gray'))
plt.show()




x[0].shape


# # Modelo CNN 



model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (64,64,1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), input_shape = (64,64,1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation = 'sigmoid'))


model.summary()




model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




# Salvando os melhores pesos
check = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
# Para de treinar se tiver overfitting
early = EarlyStopping(monitor='val_loss',patience=2)
# TensorBoard
tensorboard = TensorBoard(log_dir='./logs')

model.fit_generator(data_train, steps_per_epoch=840, epochs=10, validation_steps=122, validation_data=data_val,
                    use_multiprocessing=True,
                    callbacks=[check, early, tensorboard]                  
                   
                   )




#model.load_weights('./weights1.hdf5')


# # Testado modelo com dados de testes



score = model.evaluate_generator(data_test,steps= 239, verbose=1)

acc = score[1] * 100
loss = score[0]
print("Dados de teste --> Accuracy: %.2f %% Loss: %.3f" %(acc,loss))




# plot the training loss and accuracy

# H = model.history

# N = 5
# plt.figure(figsize=(20,10))
# plt.subplot(1,2,1)

# plt.style.use("ggplot")
# #plt.figure(figsize=(10,10))#
# #plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")#
# #plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")#
# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
# plt.title("Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower right")
# plt.savefig("plot.png")


# plt.subplot(1,2,2)

# #plt.figure(figsize=(10,10))
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.title("Loss")
# plt.xlabel("Epoch ")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="upper right")
# plt.savefig("plot1.png")

#plt.show()


# salvando modelo (estrutura)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)





x,y = data_test.next()

pred = model.predict(x)

if pred[0] == 1:
    result = "SMILE"
    #print (result)
else:
    result = "NO-SMILE"
    #print (result)

image = x[0]     
plt.imshow(image.reshape(64, 64), cmap=plt.get_cmap('gray'))
plt.text(0,60, result, fontsize=24,color='red')
plt.show()









