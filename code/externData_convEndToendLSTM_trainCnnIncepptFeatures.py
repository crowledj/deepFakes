import numpy as np
#from tensorflow import keras
import keras
#from keras.models import Sequential
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input,LSTM,Reshape,TimeDistributed,Concatenate
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,Adadelta,Adam
import scipy
from keras.applications  import inception_v3
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
#from keras.sequences import LSTM
import sklearn
from scipy.io import loadmat,savemat    
import sys,json
from sklearn.preprocessing import MinMaxScaler
sys.path.append('../Documents/Downloads/')
from sklearn.metrics import average_precision_score,classification_report
from tensorflow import set_random_seed
set_random_seed(711)
import os,sys,re,time
from collections import defaultdict
import imageio 
from PIL import Image
import cv2
from sklearn import utils
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import h5py
import hdf5storage

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#from tensorflow.python.client import device_lib
#print('device  => ' + str(device_lib.list_local_devices()))

train_path ='d:/deepfake-detection-challenge/train_sample_videos'
test_path ='d:/deepfake-detection-challenge/test_videos'
train_lable_file = 'd:/deepfake-detection-challenge/metadata.json'
cropedImgDir= 'd:/deepfake-detection-challenge/cropdTestImgs'
cropedTrainAllImgDir = 'd:/deepfake-detection-challenge/testingIMgsinSeqForConvDelp/'
extraRealDatSet12_Dir = 'd:/deepfake-detection-challenge/newFramesImgsSavesForextraRealdata_frmdownload12Ofsetkaggle/'

## Addition of new data outside the Competition's dataset to upscale the minority (real vids) class.
## source is ''YouTubeFaces ' video -> images D
extern_data = 'D:\\deepfake-detection-challenge\\aligned_images_DB\\YouTubeFacesDB_imgsAlignd/'
#start timer ...
start = time.time()

json1_file = open(train_lable_file)
json1_str = json1_file.read()
json1_data = json.loads(json1_str)

# total number of smaples (videos) after 18 seemimgly corrupt vids from the full 400 dataset are excluded.
# tot no. of REAl labeled vids = 77 ( = 76 when one defective is removed, therefore tot FAKe lables = 306)
# thus , need to add a further 230 real videos to roperly balance the dataset.
#num_data_smaples = 455
sequence_len = 9
#num_real_lables = 150

## helper functions :
def find_substring(substring, string):
    """
    Returns list of indices where substring begins in string
    >>> find_substring(’me’, "The cat says meow, meow")
    [13, 19]
    """
    indices = []
    index = -1 # Begin at -1 so index + 1 is 0
    while True:
        # Find next index of substring, by starting search from index + 1
        index = string.find(substring, index + 1)
        if index == -1:
            break # All occurrences have been found
        indices.append(index)
 
    return indices 

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key) 

def delpGueraModel():

    num_lstm_units = 2048 
    num_classes = 2
    global sequence_len

    img_width=299
    img_height=299
    img_input_1 = Input(shape=(sequence_len,img_width,img_height,3))
    num_features_per_Spatialframe_model = 2048

    feature_out_1 = TimeDistributed(inception_v3.InceptionV3(include_top=False,pooling='max',weights='imagenet'))(img_input_1)
    z = (LSTM(num_lstm_units,input_shape=(sequence_len,num_features_per_Spatialframe_model),return_sequences=False, recurrent_dropout = 0.5, dropout = 0.5))(feature_out_1)
    x = (Dense(512,activation='relu'))(z)
    x = (Dropout(0.5))(x)
    out = (Dense(num_classes,activation='softmax'))(x)
        
    model = Model(img_input_1, out)

    return model    

vidsDict= defaultdict(list)
vidImgList = sorted_alphanumeric(os.listdir(cropedTrainAllImgDir))   
train_vids_data = os.listdir(train_path)

print('orig vidImgList :')
print(str(vidImgList))

## extra kaggle dataset 12 REAL data : 
xTraKaggle12_vidImgList = sorted_alphanumeric(os.listdir(extraRealDatSet12_Dir))

filename = 'D:\deepfake-detection-challenge\LSTmFeatureInceptpreprocessTryFixeddataOrderin/properVideoDictTrackingRedoingThirdTime_JUSTDEMEANIMAGEWithInceptPreProc_RGB40VidsWith10FrameGapAswellFeaturesForLSTM'
# start timer :
fullProcesVidTimeT1 = time.time()
#only use all - cutoff to try balance the dataset 
cutoff = 1
for indx,uniq_vids in enumerate(train_vids_data[:-cutoff]):        
    baseVidname  = uniq_vids[:-4]
    
    print('doing video ' + str(indx) + ' named =>' + uniq_vids)

    ## dont include 'currupted videos' in data set at all..
    if baseVidname == 'abofeumbvv' or baseVidname == 'adhsbajydo' or baseVidname == 'andaxzscny'  or baseVidname == 'atvmxvwyns'  or baseVidname == 'avvdgsennp' or baseVidname == 'axwgcsyphv' or baseVidname == 'bbvgxeczei' or baseVidname == 'bqkdbcqjvb' \
        or baseVidname == 'cdyakrxkia' or baseVidname == 'cwqlvzefpg' or baseVidname == 'cycacemkmt' or baseVidname == 'czmqpxrqoh' or baseVidname == 'dhoqofwoxa' or baseVidname == 'djvutyvaio' or baseVidname == 'dkhlttuvmx' or baseVidname == 'dqnyszdong' \
        or baseVidname == 'eoewqcpbgt' or baseVidname == 'esyhwdfnxs' or baseVidname == 'dbzpcjntve' :
            print('skipping key => ' + str(baseVidname))
            continue

    vidsDict[baseVidname] = []
    for j,vidImgName in enumerate(vidImgList):
        
        if find_substring(baseVidname,vidImgName):
            #print('got past outer find_subStr check - adding image ' + str(vidImgName) + ' for video ' + str(uniq_vids))
            tmp = j
            while tmp < len(vidImgList) -1 and find_substring(baseVidname,vidImgList[tmp+1]) :
                vidsDict[baseVidname].append(vidImgList[tmp])
                tmp+=1
            break    
        else:
            continue

xtraRealkag12Vids = sorted_alphanumeric(os.listdir('d:/realsONly_dfdc_train_part_12'))
xTraKagvidsDict = {}
for indx,uniq_vids in enumerate(xtraRealkag12Vids):        
    baseVidname  = uniq_vids[:-4]
    
    print('doing EXTRA Kagle 12  video ' + str(indx) + ' named => ' + uniq_vids)
    xTraKagvidsDict[baseVidname] = []
    for j,vidImgName in enumerate(xTraKaggle12_vidImgList):
        
        if find_substring(baseVidname,vidImgName):
            #print('got past outer find_subStr check - adding image ' + str(vidImgName) + ' for video ' + str(uniq_vids))
            tmp = j
            while tmp < len(xTraKaggle12_vidImgList) -1 and find_substring(baseVidname,xTraKaggle12_vidImgList[tmp+1]) :
                xTraKagvidsDict[baseVidname].append(xTraKaggle12_vidImgList[tmp])
                tmp+=1
            break    
        else:
            #xTraKagvidsDict.pop(baseVidname,No                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ne)
            continue
            
    for i in xTraKagvidsDict.copy():
        if not xTraKagvidsDict[i]:
            xTraKagvidsDict.pop(i)

print(' len of kag 12 extra nDataDict after cleaning it : ' + str(len(xTraKagvidsDict)))
print('xTraKagvidsDict : ')
print(xTraKagvidsDict) 

lable_file = 'd:/deepfake-detection-challenge/dfdc_train_part_34\metadata.json'

json1_file = open(lable_file)
json1_str = json1_file.read()
json3_data = json.loads(json1_str)     

xtraRealkag34Vids = sorted_alphanumeric(os.listdir('d:/deepfake-detection-challenge\dfdc_train_part_34'))
xTraKaggle34_vidImgList = sorted_alphanumeric(os.listdir('d:/deepfake-detection-challenge\\dfdc_train_part_34_RealSeqImgs'))
xTraKag34vidsDict = {}
for indx,uniq_vids in enumerate(xtraRealkag34Vids[:645]):        

    if json3_data[uniq_vids]['label'] == 'FAKE':
        continue

    baseVidname  = uniq_vids[:-4]
    print('In set 34 procesing a real video no. ' + str(indx) + ' named => ' + uniq_vids)
    xTraKag34vidsDict[baseVidname] = []
    for j,vidImgName in enumerate(xTraKaggle34_vidImgList):
        
        if find_substring(baseVidname,vidImgName):
            
            tmp = j
            while tmp < len(xTraKaggle34_vidImgList) -1 and find_substring(baseVidname,xTraKaggle34_vidImgList[tmp+1]) :
                print('in while got past outer find_subStr check - adding image ' + str(vidImgName) + ' for video ' + str(uniq_vids))
                xTraKag34vidsDict[baseVidname].append(xTraKaggle34_vidImgList[tmp])
                tmp+=1
            break    
        else:
            #xTraKagvidsDict.pop(baseVidname,None)
            continue
            
    for i in xTraKag34vidsDict.copy():
        if not xTraKag34vidsDict[i]:
            xTraKag34vidsDict.pop(i)

print(' len of kag 12 extra nDataDict after cleaning it : ' + str(len(xTraKag34vidsDict)))
print('xTraKag34vidsDict : ')
print(xTraKag34vidsDict) 


## add two 'new' datasets of reals vid images to inti dict. :
vidsDict.update(xTraKagvidsDict)
vidsDict.update(xTraKag34vidsDict)
vidsImgsDict = vidsDict 
num_data_smaples = len(vidsImgsDict)

size = (299,299)
allVidsFeatures = OrderedDict() 
featurPerVideoDict = {}
img_dict = {}
countVids=0          

y= np.zeros(num_data_smaples)
full_counter = 0
countrealVids = 0                                                                   
for keys in vidsImgsDict.keys():
    featureMat = []    
    countVids+=1
    last_key = 'aagfhgtpmv'
    lstINdex=0 

    print(' vid key = ' + keys)
    key = keys + '.mp4'
    if key in json1_data.keys():   

        if json1_data[key]['label'] == 'FAKE':
            y[countrealVids] = 1
        else:
            y[countrealVids] = 0 

    countrealVids+=1     
    print(' @ key = ' + str(keys) + '  countrealVids = ' + str(countrealVids))   
    for indx,imgs in enumerate(vidsImgsDict[keys]):

        # must decide on a better convention here :   

        if countrealVids >= 506:
            img = imageio.imread( 'd:/deepfake-detection-challenge/dfdc_train_part_34_RealSeqImgs/' + str(imgs))
        else:    
            img = imageio.imread( 'd:/deepfake-detection-challenge/combOldnNewSeqDataDelpG/' + str(imgs))

        im = Image.fromarray(img)      
        input_img = np.array(im.resize(size, Image.BICUBIC))
        ## Change pre-processing to subtract the 3 channel means instead, prior to inputting to Inception 

        # mean_x_chanel = np.mean(input_img,axis=0)
        # mean_yx_chanel = np.mean(mean_x_chanel,axis=0)
        # input_img  = input_img - mean_yx_chanel

        ## usually a preprocessing tchnique prior to running inception but not in Delp's paper
        #input_img = inception_v3.preprocess_input(input_img.astype(np.float32))
        input_img = input_img.astype(np.float32)
        featureMat.append(np.expand_dims(np.array(input_img),axis=0))
        lstINdex = indx

    #print('about to do appendoing of all its feature data into dict @ key = ' + str(keys) + ' ...')
    if featureMat:
        featurPerVideoDict[keys] = np.array(featureMat)
        allImgsPerVidMat = np.concatenate(featureMat,axis=0)
        last_key = keys

    else:
        print('Encountered an empty sequnce of images for video => ' + str(keys))
        featurPerVideoDict[keys] = np.random.random_sample(featurPerVideoDict[last_key].shape)
        allImgsPerVidMat = np.concatenate(featurPerVideoDict[last_key],axis=0)

    allVidsFeatures[keys] = np.concatenate(np.expand_dims(allImgsPerVidMat,axis=0) ,axis=0)  

allData = []
## cut or pad the feature matrices @ 20 elements in :   
regurlySpacedVidsONlyDict = []
final_data=[]
properConcatdLSTMFetaurDataDict = allVidsFeatures
key_counter = 0
frame_cutoff = sequence_len
#unused parameter for now
#vid_frame_offset = 120
for indx in allVidsFeatures.keys():

    if properConcatdLSTMFetaurDataDict[indx].shape[0] != 0 and properConcatdLSTMFetaurDataDict[indx].shape[0] < frame_cutoff :
        firstBlock = properConcatdLSTMFetaurDataDict[indx][:properConcatdLSTMFetaurDataDict[indx].shape[0],:]  
        arr_new = np.expand_dims(properConcatdLSTMFetaurDataDict[indx][properConcatdLSTMFetaurDataDict[indx].shape[0]-1],axis=0) 
        tiled_arr = np.repeat(arr_new,(frame_cutoff - properConcatdLSTMFetaurDataDict[indx].shape[0]),axis=0)
        padded_arr = np.concatenate((firstBlock ,tiled_arr),axis=0)
        regurlySpacedVidsONlyDict.append(padded_arr)

    elif properConcatdLSTMFetaurDataDict[indx].shape[0] == 0 :
        regurlySpacedVidsONlyDict.append(np.zeros((frame_cutoff,2048)))

    else:    
        regurlySpacedVidsONlyDict.append(properConcatdLSTMFetaurDataDict[indx][:frame_cutoff,:])

    final_data.append(np.expand_dims(np.array(regurlySpacedVidsONlyDict[key_counter]),axis=0))  
    key_counter+=1 
    print('Processed video no.' + str(key_counter))


final_data=np.concatenate(final_data,axis=0)

#print('Number of real videos - minority class = ' + str( y_indx ))
posIndxs = np.argwhere(y == 1)
print('posIndxs = ' + str(posIndxs))
frst100posIndxs = posIndxs[:1]

print('prior to remove y label vect. = ' + str(y))
print('prior to remove finalData.shape = ' + str(final_data.shape))
print('prior to remove y.shape = ' + str(y.shape))

## remove 103 +ive/fake sample to balance out dataset - commented out for now

#final_data = np.delete(final_data,frst100posIndxs,axis=0)
#y = np.delete(y,frst100posIndxs,axis=0)

print('post remove y label vect. = ' + str(y))
num_data_smaples = y.shape[0]

print('Post removing indxs finalData.shape = ' + str(final_data.shape))
print('Post removing indxs y.shape = ' + str(y.shape))

## save final data (with cutting & padding of separte video frames to have a standard of 'sequnce_len' sequences) to .mat  files :
# split this dict into 4 parts as it's quite large :

## svae data to .mat  files :
# split = 13
# chunk = len(allVidsFeatures) // split
# print('splitting up dicts into ' + str(split) + '  parts ... i.e into chunks of size (no.vids/keys) = ' + str(chunk) + ' for ' +  str(num_data_smaples) + ' videos ....') 

# Write - save final data just prior to training 
#final_data_witMeanAllchannels_seq_lenofDelDict = {}
#final_data_witMeanAllchannels_seq_lenofDelDict[u'allVidsFeatures_noPreproces_balancesDaatwithExtYouTubeFaces_maxScali_finalDataPostFrameSeq_cut_pading'] = final_data
#hdf5storage.write(final_data_witMeanAllchannels_seq_lenofDelDict,'d:\deepfake-detection-challenge\dataSavdFrmdelpGueraprog/final_data_noPreproces_balancesDaatwithExtYouTubeFaces_maxScaling_seqLenOf_PostPad_cutForSameSeqLens' + str(sequence_len) + '.mat',final_data_witMeanAllchannels_seq_lenofDelDict,do_compression=True)
#scipy.io.savemat('d:/deepfake-detection-challenge/firstExternyoutubedataFiles/allVidsFeatures_InceptPreprocesMethod_AllRealdataKag12ndExtYouTubeFaces_finalData3.mat',final_data_witMeanAllchannels_seq_lenofDelDict,do_compression=True)

#print('  Shape of final_data = ' + str(final_data.shape))
#print('  Shape of regurlySpacedVidsONlyDict = ' + str(regurlySpacedVidsONlyDict_np.shape))
#print(' regurlySpacedVidsONlyDict is of type ' + str(type(regurlySpacedVidsONlyDict_np)))

#Xtrain,Xval,Xtest = final_data[:round(0.6*final_data.shape[0]),:,:,:,:], final_data[round(0.6*final_data.shape[0]): \
#    round(0.8*final_data.shape[0]),:,:,:,:], final_data[round(0.8*final_data.shape[0]):,:,:,:,:]
#Xtrain,Xval = train_test_split(final)
#Xtrain, Xtest, ytrain, ytest = train_test_split(final_data, y, test_size=0.4, random_state=42)

regurlySpacedVidsONlyDict_np = np.array(regurlySpacedVidsONlyDict)

idx = np.arange(final_data.shape[0])
idx_train,idx_test = train_test_split(idx,test_size=0.15,random_state=42)
idx_train,idx_val = train_test_split(idx_train,test_size=0.15,random_state=42)

Xtrain = final_data[idx_train,:,:,:,:]
Ytrain = y[idx_train]
Xval = final_data[idx_val,:,:,:,:]
Yval = y[idx_val]
Xtest = final_data[idx_test,:,:,:,:]
Ytest = y[idx_test] 

#max_Xtrain = 1.0 #np.max(Xtrain)
#,max_Xval,max_Xtest =  np.max(Xtrain),np.max(Xval),np.max(Xtest)  # 1.0,1.0,1.0
    
Xtrain,Xval,Xtest  = Xtrain / 255. , Xval / 255. , Xtest / 255. 
Xtrain,Xval,Xtest  = Xtrain.reshape(-1,Xtrain.shape[1],Xtrain.shape[2],Xtrain.shape[3],Xtrain.shape[4]) ,Xval.reshape(-1,\
    Xval.shape[1],Xval.shape[2],Xval.shape[3],Xval.shape[4]), Xtest.reshape(-1,Xtest.shape[1],Xtest.shape[2],Xtest.shape[3],Xtest.shape[4])

## TEST : try resmaple data t up sample the real dataset  :

# list_values = [ v for v in dict.values(final_data) ]
# finlDataArr_np = np.array(list_values)
##                        comment out resmapling method for now :
# X_balnce,y_balance, idxs_balance = balanced_sample_maker(regurlySpacedVidsONlyDict_np, y, sample_size= 400 - y_indx, random_seed=None)

# print('Post the resampling balancing func. -- indxs are =  ' + str(idxs_balance) ) # + ' ia  real -- minority class index ')
# print('Post the resampling balancing func. -- new X_bal arrays shape = ' + str(X_balnce.shape))  
# print('Post the resampling balancing func. -- new label vec. y_bal arrays shape = ' + str(y_balance.shape) ) 
# print('Post the resampling balancing func. -- new label vec. y values are : ' + str(y_balance)) 

print('dim of Xtrain = ' + str(Xtrain.shape))
print('dim of Xval = ' + str(Xval.shape))
print('dim of Xtest = ' + str(Xtest.shape))
#Ytrain, Yval,Ytest = y[:round(0.6*y.shape[0])], y[round(0.6*y.shape[0]):round(0.8*y.shape[0])], y[round(0.8 *y.shape[0]):] 
print('dim of Ytrain = ' + str(Ytrain.shape))
print('dim of Yval = ' + str(Yval.shape))
print('dim of Ytest = ' + str(Ytest.shape))

print(' Ytrain values = ' + str(Ytrain))
print(' Ytest values = ' + str(Ytest))
print(' Yval values = ' + str(Yval))

num_fakes = np.sum(y)
print('no. of Positive (fakes) samples in the dataset = ' + str(num_fakes))
print('no. of Negative (reals) samples in the dataset = ' + str(num_data_smaples - num_fakes))

print('num. positive examples in Training set = ' + str(np.sum(Ytrain)))
print('num. negative examples in Training set = ' + str(len(Ytrain) - np.sum(Ytrain)))
print('num. positive examples in Validation set = ' + str(np.sum(Yval)))
print('num. negative examples in Validation set = ' + str(len(Yval) - np.sum(Yval)))
print('num. positive examples in Test set = ' + str(np.sum(Ytest)))
print('num. negative examples in Test set = ' + str(len(Ytest) - np.sum(Ytest)))

## transform your class labels to categorical types for Nueral net :
Ytrain =  np.concatenate((Ytrain.reshape((-1,1)),1.0 - Ytrain.reshape((-1,1))),axis=1)  
Yval   =  np.concatenate((Yval.reshape((-1,1)),1.0 -   Yval.reshape((-1,1))),axis=1) 
Ytest  =  np.concatenate((Ytest.reshape((-1,1)),1.0 -  Ytest.reshape((-1,1))),axis=1)  
sampl_arr = np.ones((Xtrain.shape[0],299*299*3,sequence_len))

##  write test data separat;ey out too file : (switch off for now as data is a bit large to save into one .mat file)
# allData = {}
# allData['Xtrain'] = Xtrain
# allData['Ytrain'] = Ytrain

# allData['Xval'] = Xval
# allData['Ytrain'] = Yval

# allData['Xtest'] = Xtest
# allData['Ytest'] = Ytest

# scipy.io.savemat('d:\deepfake-detection-challenge\dataSavdFrmdelpGueraprog/trainValntestDatawithExtDatIncludedBalncd300SmplesNegnPosnoPreprocBut255Normize.mat',allData,do_compression=True)

if __name__ == "__main__":
    
    #incept_layer = incept_local_notop()
    ml_model = delpGueraModel()
    #num_classes=1
    num_lstm_units = 2048 
    weights_used = 0
    learn_rate = 0.00001
    batch_size = 2
    dropout = 1
    epoch_num = 15

    print(ml_model.summary())   
    ## change to Adam at some pt.
    adam = Adam(lr=learn_rate, beta_1=0.9,beta_2=0.999,decay=0.000001)
    ml_model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])
    mcp = ModelCheckpoint(filepath='./ExtUtubDtaDelpGEnd2end_inceptV3_numfrmsPerSeq_9lstmUntsBalncd300SmplesNegnPos_With255NormlzePreProc' + str(num_lstm_units) + '_learn_rate_' + str(learn_rate) + \
        '_dropot_usd_' + str(dropout) + '_weight_usd_' + str(weights_used) + '_batch_size_' + str(batch_size)  + '_epoch_num_' + str(epoch_num) + \
            '_.hdf5', verbose=1,monitor='val_loss',save_best_only=True)
    
    history = ml_model.fit(Xtrain, Ytrain,
                        shuffle=True,
                        batch_size=batch_size, 
                        epochs=epoch_num,
                        verbose=1, callbacks=[mcp],
                        validation_data=(Xval,Yval))

    result = ml_model.evaluate(Xtest,Ytest,batch_size=batch_size)

    print('result on test set is = ')
    print(result)  

    print('\n Actual test set Labels => \n ')
    print(Ytest)    

    preds_on_tstSet = ml_model.predict(Xtest)             

    print('probability predictions on test set is = ')
    print(preds_on_tstSet)           

    print('model precisiont is = ')
    precsion_recall_avg = average_precision_score(Ytest,y_score)
    print(precsion_recall_avg)

    classReport = classification_report(Ytest,y_score)
    print('mclassification report is = ')
    print(classReport)
