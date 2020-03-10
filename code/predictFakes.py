
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,Adadelta,Adam
import numpy as np
from keras.applications import VGG16
from keras.applications  import inception_v3
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import os,sys
from PIL import Image
import json
import imageio as imio
import csv,cv2

test_lable_file = 'd:/deepfake-detection-challenge/test_videos/metadata.json'
cropedTestImgs = 'd:/deepfake-detection-challenge/cropdTestImgs'


listTestVid  = os.listdir('d:/deepfake-detection-challenge/test_videos')
listTestImgs = os.listdir(cropedTestImgs)

#print(listTestVid)
#print('******************** /n')
#print(listTestImgs)

baseImgNames=[]
for j,imgs in enumerate(listTestImgs):
    baseImgNames.append(imgs[:-4])

count=0
baseVidName=[]
for i,vids in enumerate(listTestVid):
     baseVidName.append(vids[:-4])

#     if baseVidName in  baseImgNames:
#         count+=1
#         print(baseVidName)

print('total no. of missing test images  = ' + str(count))
size = 229
def load_test_data():
    
    listDirs = os.listdir(cropedTestImgs)
    ALL_IMAGES=[]
    IMAGES = []
    #y = np.zeros(len(listDirs)-1)
    #y_indx=0
    for imgs in  listDirs:

        try :

            img = imio.imread(cropedTestImgs + '/' + str(imgs)) 
            im = Image.fromarray(img)
            #size = (229,229)

            print('abut ot do  resize on new_image  ...')
            new_image = cv2.resize(np.array(im),(size,size))
            #new_image = np.array(im.resize(size, Image.BICUBIC))

            print('done resize, new_image size = ' + str(new_image.shape))

            IMAGES.append(np.expand_dims(new_image,axis=0))   

        except  Exception as inst:
            print("Error occured at doing marker stuff of image file -  ")  
            continue
        
    IMAGES = np.concatenate(IMAGES,axis=0) 

    print('in load test data function and outputs shape is = ' + str(IMAGES.shape))

    #Xtest = IMAGES.astype('float') / 255.0
    
    print('in load_data() -- shape of X_rtest = '+ str(IMAGES.shape))

    return IMAGES  #,y_test


img_dimens=229
def predict_fakes(Xtest):

    

    # img_input = Input(shape=(img_dimens,img_dimens,3))
    # num_classes=1

    batch_size=24

    # x = Conv2D(32,(3, 3), activation='relu', padding='same',name='block1_conv1')(img_input)
    # x= MaxPooling2D((2,2), strides=(2,2), name='block1_pool1')(x)
    
    # x = Conv2D(32,(3,3), activation='relu', padding='same',name='block1_conv2')(x)
    # x= MaxPooling2D((2,2), strides=(2,2), name='block1_pool2')(x)
    
    # x = Conv2D(64,(3,3), activation='relu', padding='same',name='block2_conv1')(x)
    # x= MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool1')(x)
    
    # x = Conv2D(64,(3,3), activation='relu', padding='same',name='block2_conv2')(x)
    # x= MaxPooling2D((2, 2), strides=(2, 2), name='block2_poo2')(x)    
    
    # x=Flatten()(x)
    # x=Dense( 128,activation='relu')(x)

    # x = Dropout(0.5)(x) 
    # x=Dense(num_classes,activation='sigmoid')(x)

    # model = Model(img_input, x, name='first_CNN')    
    # 
    # 
    # # Block 1

    num_classes=1
    img_width=229
    img_height=229

    model = keras.models.Sequential()

    model.add(inception_v3.InceptionV3(include_top=False,input_shape=(img_width,img_height,3),pooling='max'))

    model.add(Dense(1024 ,activation='relu'))
  
    model.add(Dense(128,activation='relu'))
   
    model.add(Dense(num_classes,activation='sigmoid'))     

    Xtest=Xtest.reshape(Xtest.shape[0],img_dimens,img_dimens,3)
    #Xtest=Xtest.astype('float') / 255.0
    Xtest = Xtest / 255.
    print('Xtest shape =' + str(Xtest.shape))

    model.load_weights("C:/Users/MaaD/Documents/Downloads/fulVidDeepFakeCode/best_model_firstInceptv3_imageNetWeights_adamOptimize_70_30_split_epochs_10_batch_24_2Dense_1024_128_dropout05.hdf5")
    te_pr = model.predict(Xtest, batch_size=batch_size, verbose=1)

    return te_pr


if __name__ == "__main__":
 
    X_test=load_test_data()
    probs=predict_fakes(X_test); 
    #print(probs)
    fp = open('./submission_inceptv3_properVers1.csv','w')
    listTestVid = os.listdir(cropedTestImgs)
    
    ## test
    #probs= 0.1*(np.ones((295)))
    #print(probs)

    reals =  np.where(probs >= 0.5,1,0)  

    #reals = (int) (probs == True) 


    #sumReals = np.sum(reals)
    #print(sumReals) 
    print('******************  Probabiities and claassifications for test videos are : ******************************* \n \n')

    for i in range( len(listTestVid) ): # reals.shape[0]):
        basenameVid = listTestVid[i][:-4]
        #if basenameVid in baseImgNames:
        #    indx = baseImgNames.index(basenameVid) 


        print(' Video  -- ' + str(listTestVid[i]) + ' : \n')

        print(reals[i])
        #print('\n')

        print(probs[i])
        #print('\n')       

        
    #i=0
        rawStr=""
        if  basenameVid in  baseImgNames:
            rawStr = listTestVid[i]  + ','  + str(reals[i][0])  #ndx][0])
        #else:
        #    ranLable = np.random.randint(2) #, size=1)
        #    rawStr = listTestVid[i]  + ','  + str(ranLable)    
        fp.write(rawStr + '\n')

    fp.close()
    # with open('./submission1.csv', mode='a+') as employee_file:
    #     employee_writer = csv.writer(employee_file, delimiter='/') #, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     rawStr = listTestVid[i]  + ','  + str(reals[i])

    #     employee_writer.writerow(rawStr)
    #     i+=1 

    # with open('employee_birthday.txt') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         if line_count == 0:
    #             print(f'Column names are {", ".join(row)}')
    #             line_count += 1
    #         else:
    #             print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
    #             line_count += 1
    #     print(f'Processed {line_count} lines.')



    shite = 0   