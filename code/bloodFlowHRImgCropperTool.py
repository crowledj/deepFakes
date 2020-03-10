import sys
sys.path.append('c:/Users/MaaD/AppData/Local/Temp/facial_landmarks.py')
import numpy as np
import os
from numpy import genfromtxt
import scipy
import scipy.misc
from scipy.io import savemat,loadmat
from PIL import Image
import PIL
import json
import cv2
import dlib
import face_recognition
import imageio,time
import matplotlib.pyplot as plt
from peakutils.peak import indexes
from scipy.signal import butter,filtfilt

dlib.DLIB_USE_CUDA=False

train_path ='d:/deepfake-detection-challenge/train_sample_videos'
test_path ='d:/deepfake-detection-challenge/test_videos'
train_lable_file = 'd:/deepfake-detection-challenge/metadata.json'
cropedImgDir= 'd:/deepfake-detection-challenge/cropdTestImgs'
cropedTrainAllImgDir = 'd:\\deepfake-detection-challenge\\cropdTrainFullVidImgDirsAllFrames'

json1_file = open(train_lable_file)
json1_str = json1_file.read()
json1_data = json.loads(json1_str)      

## Define a low pass filter to emulate the 'hmrBandpassFilter' used in the matlab code
def butter_bandpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff)#, btype='low', analog=False)
    print(' in plain butter_bandpass  and returning from this func. ...')
    return b, a

def butter_lowpass_filter(data, cutoff,fs, order=3):
    b, a = butter_bandpass(cutoff ,fs, order=order)
    #y = lfilter(b, a, data)
    ##Kang prefers the look of this filtered signal to pauls so use this filter
    print(' in inner  plain butter_LOWpass  and doing filtfilt from this func. ... shape of data = ' + str(data.shape))
    y = filtfilt(b, a, data)
    print(' in inner  plain butter_LOWpass  and PAST & returning from this func. ...')
    return y

# class Video:
#     def __init__(self, path):
#         self.path = path
#         self.container = imageio.get_reader(path, 'ffmpeg')
#         self.length = self.container.count_frames()
#         self.fps = self.container.get_meta_data()['fps']
    
#     def init_head(self):
#         self.container.set_image_index(0)
    
#     def next_frame(self):
#         self.container.get_next_data()
    
#     def get(self, key):
#         return self.container.get_data(key)
    
#     def __call__(self, key):
#         return self.get(key)
    
#     def __len__(self):
#         return self.length

def shuffle_split_data(X, y):
    split = np.random.rand(X.shape[0]) < 0.7

    X_Train = X[split,:,:,:]
    y_Train = y[split]
    X_Test =  X[~split,:,:,:]
    y_Test = y[~split]

    return X_Train, y_Train, X_Test, y_Test

save_interval = 2
margin = 0.2
img_resize=229
size = 229 


ALL_IMAGES=[]
IMAGES = []
y = []
allImgsDict = {}

countVid=1
counter = 0
def processVid(vid):

    y_img = []
    try:

        video_capture = cv2.VideoCapture(os.path.join(train_path + '/', str(vid)))
        name = vid
        #global countVid
        global counter 
        
        print(' BEGIN to process Video : ' + str(vid)  + ' has frame rate  and label = ' + str(json1_data[name]['label']))
        #countVid+=1

        # Initialize variables
        face_locations = []
        
        #os.mkdir(cropedTrainAllImgDir + '/' + name[:-4])
        greenChanel_sig = []
        while video_capture.isOpened(): 
            
            #print('back in for at iteration -- ' + str(i))
            #try :
                        
            counter += 1
            #print(' processing  frame no ' + str(counter)  + '... :)')
            ret, frame = video_capture.read()

            if counter >= 295 :
                break

            #if counter % 20 != 0:
            #s    continue

            print(' counter = ' + str(counter))
            # Grab a single frame of video
            #vidCaptureandFrameFlipT0 = time.time()

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            #vidCaptureandFrameFlipT1 = time.time()

            #totalVidCapturenFrameFlip = vidCaptureandFrameFlipT1 - vidCaptureandFrameFlipT0
            #print('vidCaptureandFrameFlip of image took -- ' + str(totalVidCapturenFrameFlip) + '  milliseconds')

            #faceRecogT0 = time.time()

            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)

            #faceRecogT1 = time.time()

            #faceRecogtime = faceRecogT1 - faceRecogT0
            #print('faceRecog time of image took -- ' + str(faceRecogtime) + '  milliseconds')


            if face_locations is not None:
            # Display the results
                for face_position in face_locations:
                    # Draw a box around the face

                    #croppingLoopT0 = time.time()

                    offset = round(margin * (face_position[2] - face_position[0]))
                    y0 = max(face_position[0] + offset, 0)
                    x1 = min(face_position[1] - offset, rgb_frame.shape[1])
                    y1 = min(face_position[2] - offset, rgb_frame.shape[0])
                    x0 = max(face_position[3] + offset, 0)
                    face = rgb_frame[y0:y1,x0:x1]

                    #print('dim of face = ' + str(face.shape ))

                    green_channelVals = np.mean(np.mean(face,axis=1),axis=0)[1]

                    #print('shape of green cqhnnel signal = ' + str(green_channelVals.shape))

                    #print('.. and some values from this green cqhnnel signal = ' + str(green_channelVals))

                    greenChanel_sig.append(green_channelVals)

                    #croppingLoopT1 = time.time(),

                    # cv2.imshow('full_frame',frame)

                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     continue

                    # cv2.imshow('blah',face)

                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     continue

                    #totalCrop = croppingLoopT1 - croppingLoopT0
                    #print('cropping of image took -- ' + str(totalCrop) + '  milliseconds')

                    #cv2.rectangle(face, (left, top), (right, bottom), (0, 0, 255), 2)

                    #imgResizernAppendT0 = time.time()

                    inp = cv2.resize(face,(size,size))
                    #IMAGES.append(np.expand_dims(inp,axis=0))   

                    # imgResizernAppendT1 = time.time()

                    # imgResizenAppend = imgResizernAppendT1 - imgResizernAppendT0
                    # print('resizing and appending of image took -- ' + str(imgResizenAppend) + '  milliseconds')

                    #print('shape of inp to go into IMAGES datastruct = ' + str(inp.shape))

                    #saveDir = cropedTrainAllImgDir 

                    # imageWriteT0 = time.time()
                    #imageio.imwrite(saveDir + '/' + name + '_' + str(counter) + '.jpg', face)
                    
                    # imageWriteT1 = time.time()

                    # total = imageWriteT1 - imageWriteT0
                    # print('saving of image took -- ' + str(total) + '  milliseconds')
            else:
                print('no face found in frame ' + str(counter))
                continue

        #print(' length of green signal  = ' + str(greenChanel_sig))

        greenChanel_sig_np = np.array(greenChanel_sig) #.flatten()

        print('video => ' + str(name) + ' green channel signal : ')
        print(greenChanel_sig_np)

        bestSnR_FfullSignal = greenChanel_sig_np

        print('It looks like ... ')

        F_rate = 27.27
        avgLength = 30

        heartRate_sig = butter_lowpass_filter(bestSnR_FfullSignal, 2.0, F_rate, order=4)
        print('got past  OUTER low_pass filter and doing convolution -- hr signal pre-smooth is ...')
            
        N=25
        smoothed_sig  = np.convolve(heartRate_sig, np.ones((N,))/N)[(N-1):]    

        print(' DID convolution & about to do indexes func. ... heartRate_sig = ' + str(heartRate_sig))

        print('got past  convolution -- -- hr signal post-smooth is ...')

        plt.plot(smoothed_sig)
        plt.show()
        
        indices = indexes(np.array(smoothed_sig), thres=0.0001, min_dist=0)

        print(' Found peaks with indexes func.  & finding diff idir na peaks ... indices = ' + str(indices))
            
        peak_index_vect=indices[1:]
    
        hitv=np.zeros((len(peak_index_vect)-1))
        #hitv1=np.zeros((len(peak_index_vect)-1))

        for j in range(0,len(peak_index_vect)-1):
            hitv[j] = peak_index_vect[j+1]-peak_index_vect[j]       

        ## Mean of time dist between two peaks = HR:
        #avg_peakDist=np.mean(hitv)
        
        hitv=hitv[0:len(peak_index_vect)-2]     

        print(' getting mean of diff idir na peaks  Function... hitv non short = ' + str(hitv))
    
        ## Mean of time dist between two peaks = HR:
        avg_peakDist=np.mean(hitv) #_short)       

        print(' finally calculating the acg. HR rate from this ...')     
        
        avg_HR=round((60.0*F_rate)/avg_peakDist)  
        #hr_list.append(avg_HR)
        #global_HR=hr_list[-1]
        
        #count=count+1
        #print('After')
        #print(count)
        #print('seconds .....')
        
        print('**************************************************')
        print('****************  avg. exact HR = ***************')
        print((60.0*F_rate)/avg_peakDist)
        print('**************************************************')
        
        print('**************************************************')
        print('**************** Rounded AVG. HR = ***************')
        print(avg_HR)
        print('**************************************************') 


        plt.plot(smoothed_sig)
        plt.show()


            # if json1_data[name]['label'] == 'REAL':
            #     y_img.append(0)
            # else:
            #     y_img.append(1)     
        
    except Exception as inst:
    
        print("Exceptopn occured at doing videos image file  no  ...")
        print(inst)    

    return IMAGES,y_img

if __name__ == "__main__":

    listVids = os.listdir(train_path) 
    filenames = []
    save_jump=4
    for k,vids in enumerate(listVids[26:35]):

        print('processing video => ' + str(vids))

        fullProcesVidFuncT0 = time.time()    
        images,y_ = processVid(vids)


    # hr_sig = [105.02342811, 104.51985454, 104.04725294, 103.63904823, 103.32584402,103.12974804, 103.05787319, 103.09784576, 103.21765702, 103.37044842,
    # 103.50300851, 103.56577201, 103.52209163, 103.3551499 , 103.07165458,
    # 102.70210786, 102.29780477, 101.92480492, 101.655118  , 101.555522,
    # 101.67498964, 102.03260099, 102.60877491, 103.34311192, 104.14145812,
    # 104.89257661, 105.49139773, 105.86261324, 105.9773269 , 105.85758634,
    # 105.5679631 , 105.19773946, 104.83979648, 104.57222915, 104.44655659,
    # 104.48353064, 104.67537012, 104.99233224, 105.39150995, 105.82599705,
    # 106.25286062, 106.63879365, 106.96290072, 107.21666718, 107.40166792,
    # 107.52592677, 107.59997979, 107.63359144, 107.63375396, 107.60417376,
    # 107.54605508, 107.45972593, 107.34651435, 107.21025434, 107.05792629,
    # 106.89925968, 106.7455294 , 106.60803786, 106.49679701, 106.41977835,
    # 106.38286388, 106.39034837, 106.44559864, 106.55138541, 106.70952119,
    # 106.9197021 , 107.17774776, 107.47368344, 107.79029427, 108.10287758,
    # 108.38084947, 108.5915142 , 108.70562997, 108.70353906, 108.57998707,
    # 108.34584989, 108.02598191, 107.65381833, 107.26442254, 106.8879211,
    # 106.54482665, 106.2439856 , 105.9831172 , 105.75132682, 105.53267673,
    # 105.30988939, 105.06746837, 104.79383757, 104.48240811, 104.13170292,
    # 103.74478448, 103.32829434, 102.8914739 , 102.44555803, 102.00378459,
    # 101.58183786, 101.19794813, 100.87151101, 100.61951076, 100.45138006,
    # 100.36446302, 100.34264928, 100.35938161, 100.38400243, 100.3889149,
    # 100.35512547, 100.27495319, 100.15202283,  99.9994147 ,  99.8369591,
    #  99.68841855,  99.57895757,  99.5329854 ,  99.57224369,  99.71395496,
    #  99.96897359, 100.34012315, 100.82107036, 101.39598754, 102.03998143,
    # 102.72011393, 103.39691322, 104.02640002, 104.56269786, 104.96123191,
    # 105.18232764, 105.19480565, 104.97919818, 104.53057717, 103.86135265,
    # 103.00418754, 102.01414704, 100.96804263,  99.95883512,  99.08444531,
    #  98.43266535,  98.06573325,  98.00861098,  98.24408331,  98.71603784,
    #  99.34023997, 100.0199981 , 100.66287234, 101.19460413, 101.56778729,
    # 101.7647245 , 101.79545632, 101.69261979, 101.5046622 , 101.28835204,
    # 101.10092257, 100.99194808, 100.99535016, 101.12257272, 101.35854436,
    # 101.66219454, 101.97273468, 102.22150457, 102.34714458, 102.31006549,
    # 102.10181787, 101.74650304, 101.29413468, 100.80849127, 100.35336752,
    #  99.98092096,  99.72448608,  99.59654589,  99.59107353,  99.68850631,
    #  99.8613669 , 100.07897637, 100.31053461, 100.52672042, 100.7006372,
    # 100.80923737, 100.83613117, 100.77585585, 100.63850855, 100.45272053,
    # 100.26484019, 100.13305462, 100.11671918, 100.26289595, 100.5935034,
    # 101.09696823, 101.72738377, 102.41202496, 103.06549464, 103.60686995,
    # 103.97559995, 104.14249398, 104.11355374, 103.92630418, 103.64021933,
    # 103.32411616, 103.04351618, 102.85019801, 102.7751861 , 102.8257027,
    # 102.9861188 , 103.22247581, 103.48967398, 103.7400461 , 103.93184002,
    # 104.03612044, 104.04078125, 103.95085163, 103.78516988, 103.57055946,
    # 103.335342  , 103.103987  , 102.89402707, 102.71548202, 102.57230242,
    # 102.4649311 , 102.39299628, 102.35732684, 102.36081002, 102.40799081,
    # 102.50365379, 102.6508626 , 102.84900071, 103.09226079, 103.36884732,
    # 103.66101695, 103.94605047, 104.1982468 , 104.3918912 , 104.50484802,
    # 104.52213539, 104.43874885, 104.26114679, 104.00708669, 103.70381765,
    # 103.38495647, 103.08662904, 102.84350996, 102.68517041, 102.63278397,
    # 102.69605016, 102.87042247, 103.13529884, 103.4543297 , 103.7789207,
    # 104.05511232, 104.23263471, 104.27389457, 104.16062146, 103.8967664,
    # 103.50735245, 103.03383938, 102.52710112, 102.03940947, 101.6168814,
    # 101.29363435, 101.08843054, 101.00401693, 101.02885961, 101.14065402,
    # 101.31088618, 101.50975036, 101.71077152, 101.89445031, 102.05020541,
    # 102.17602529, 102.27568716, 102.35406174, 102.41169056, 102.44029865,
    # 102.42097881, 102.32624347, 102.12597444, 101.79585123, 101.32571088,
    # 100.72503432, 100.0236032 ,  99.26712808,  98.50955161,  97.80469794,
    #  97.19946176,  96.72942189,  96.41668796,  96.26942618,  96.28262899,
    #  96.43989169,  96.71603958,  97.08040754,  97.5004576 ,  97.94528462,
    #  98.38849153,  98.81002307,  99.19687031,  99.54294141,  99.84853872,
    # 100.11965895, 100.36697735, 100.60425805, 100.84615079]

    # plt.plot(hr_sig)
    # plt.show()

    # F_rate = 27.0

    # indices = indexes(np.array(hr_sig ), thres=0.0001, min_dist=0)

    # print(' Found peaks with indexes func.  & finding diff idir na peaks ... indices = ' + str(indices))
        
    # peak_index_vect=indices[1:]
    
    # hitv=np.zeros((len(peak_index_vect)-1))
    # #hitv1=np.zeros((len(peak_index_vect)-1))

    # for j in range(0,len(peak_index_vect)-1):
    #     hitv[j] = peak_index_vect[j+1]-peak_index_vect[j]       

    # ## Mean of time dist between two peaks = HR:
    # #avg_peakDist=np.mean(hitv)
    
    # hitv_short=hitv[0:len(peak_index_vect)-2]    

    # print(' getting mean of diff idir na peaks  Function... hitv non short = ' + str(hitv))

    # ## Mean of time dist between two peaks = HR:
    # avg_peakDist=np.mean(hitv) #_short)       

    # print(' finally calculating the acg. HR rate from this ...')     
    
    # avg_HR=round((60.0*F_rate)/avg_peakDist)  
    # #hr_list.append(avg_HR)
    # #global_HR=hr_list[-1]
    
    # #count=count+1
    # #print('After')
    # #print(count)
    # #print('seconds .....')
    
    # print('**************************************************')
    # print('****************  avg. exact HR = ***************')
    # print((60.0*F_rate)/avg_peakDist)
    # print('**************************************************')
    
    # print('**************************************************')
    # print('**************** Rounded AVG. HR = ***************')
    # print(avg_HR)
    # print('**************************************************') 




    #     fullProcesVidFuncT1 = time.time()

    #     Fulltotal = fullProcesVidFuncT1 - fullProcesVidFuncT0
    #     print(' ********************    saving of image took -- ' + str(Fulltotal) + '  milliseconds ****************')

    #     ALL_IMAGES=np.concatenate(images,axis=0)
    #     #print('Size of ALL_IMAGES = ' + str(ALL_IMAGES.shape))
    #     #print('before appppending y again ' + str(len(y_)))
    #     y.append(y_)
    #     #print('Size of labels vector y  = ' + str(len(y)))
        
    #     if k % save_jump == 0 and k*300 >= counter :
        
    #         ImgsFilename2 = cropedTrainAllImgDir + '/' + 'mainThreadsConcatdEveryImgsFrom_kVal_' + str(k) + '_' + vids[:-4]  + '.mat'
    #         print(' Saving Data to accumlated .mat file : ' + str(ImgsFilename2) + '  ... :)')
    #         everyImgsDict_part = {}
    #         prev_seg = k - counter
    #         #clamp the value at zero
    #         if prev_seg < 0:
    #             prev_seg = 0

    #         everyImgsDict_part['concatdImgsomeVids']  = ALL_IMAGES[prev_seg:,:]
    #         everyImgsDict_part['concatdImgsSomeVid_labels'] = y
    #         counter = 0
            
    #         scipy.io.savemat(ImgsFilename2,everyImgsDict_part,do_compression=True)

    # allImgsDict['concatdImgsEveryVidFrames']  = ALL_IMAGES
    # allImgsDict['concatdImgsAllVid_fixed_labels'] = y
    # ImgsFilename = cropedTrainAllImgDir + '/' + 'concatdFixed_labels_EveryVidsData.mat'
    # scipy.io.savemat(ImgsFilename,allImgsDict,do_compression=True)

    # print('completed processing all videos ... !')
