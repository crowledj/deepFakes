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

dlib.DLIB_USE_CUDA=True

train_path ='d:/deepfake-detection-challenge/dfdc_train_part_45/dfdc_train_part_45'
test_path ='d:/deepfake-detection-challenge/test_videos'
train_lable_file = 'd:/deepfake-detection-challenge/dfdc_train_part_45/dfdc_train_part_45/metadata.json'
#cropedImgDir= 'd:/deepfake-detection-challenge/cropdTestImgs'
cropedTrainAllImgDir = 'd:/deepfake-detection-challenge/dfdc_train_part_45_sepaarateTestSetSeqImgs'

json1_file = open(train_lable_file)
json1_str = json1_file.read()
json1_data = json.loads(json1_str)      

class Video:
    def __init__(self, path):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']
    
    def init_head(self):
        self.container.set_image_index(0)
    
    def next_frame(self):
        self.container.get_next_data()
    
    def get(self, key):
        return self.container.get_data(key)
    
    def __call__(self, key):
        return self.get(key)
    
    def __len__(self):
        return self.length

def shuffle_split_data(X, y):
    split = np.random.rand(X.shape[0]) < 0.7

    X_Train = X[split,:,:,:]
    y_Train = y[split]
    X_Test =  X[~split,:,:,:]
    y_Test = y[~split]

    return X_Train, y_Train, X_Test, y_Test

save_interval = 20
margin = 0.2
img_resize=229
size = 229 

ALL_IMAGES=[]
IMAGES = []
y = []
allImgsDict = {}

countVid=1
def processVid(vid):

    try:

        video_capture = Video(os.path.join(train_path + '/', str(vid)))
        name = vid
        global countVid
        #print(' BEGIN to process Video : ' + str(vid)  + ' has frame rate ' + str(video_capture.fps)  + ' and label = ' + str(json1_data[name]['label']))
        countVid+=1

        # Initialize variables
        face_locations = []
        y_img=[]
        #os.mkdir(cropedTrainAllImgDir)
 
    except Exception as inst:
        
        print("Error occured at doing marker stuff of image file  ...")
        print(inst)    

    counter = 0
    for i in range(0,video_capture.__len__(),save_interval):
        
        #print('back in for at iteration -- ' + str(i))
        try :
            
            counter += 1
            #print(' processing  frame no ' + str(counter)  + '... :)')
            # Grab a single frame of video
            frame = video_capture.get(i)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces in the current frame of video
            face_positions = face_recognition.face_locations(rgb_frame)
            if face_positions is not None:

                # Display the results
                for face_position in face_positions:
                    # Draw a box around the face
                    offset = round(margin * (face_position[2] - face_position[0]))
                    y0 = max(face_position[0] - offset, 0)
                    x1 = min(face_position[1] + offset, rgb_frame.shape[1])
                    y1 = min(face_position[2] + offset, rgb_frame.shape[0])
                    x0 = max(face_position[3] - offset, 0)
                    face = rgb_frame[y0:y1,x0:x1]

                    inp = cv2.resize(face,(size,size))
                    IMAGES.append(np.expand_dims(inp,axis=0))   
                    imageio.imwrite(cropedTrainAllImgDir + '/' + name[:-4] + '_' + str(counter) + '.jpg', face)
            else:
                print('no face found in frame ' + str(counter))
                continue   
        
        except Exception as inst:
        
            print("Error occured at doing video's image file  no " + str(counter) + ' ...')
            print(inst)    
            continue

    return 0 #IMAGES,y_img

if __name__ == "__main__":

    listVids = os.listdir(train_path)
    filenames = []
    countRealVids = 0
    for k,vids in enumerate(listVids):

        countRealVids +=1
        print('doing Real sequence no. ' + str(countRealVids))
        processVid(vids)
        print('Did video no. ' + str(k) + ' which is => ' + str(vids)) 

        # if json1_data[vids]['label'] == 'REAL':
            
        #     countRealVids +=1
        #     print('doing Real sequence no. ' + str(countRealVids))
        #     processVid(vids)
        #     print('Did video no. ' + str(k) + ' which is => ' + str(vids))
           
        #     if countRealVids == 5:
        #         break

        # else:

        #     print('Doing a fake video -- skipping -- ' + str(vids))


#        ALL_IMAGES=np.concatenate(images,axis=0)
    #     print('Size of ALL_IMAGES = ' + str(ALL_IMAGES.shape))
    #     print('before appppending y again ' + str(len(y_)))
    #     y.append(y_)
    #     print('Size of labels vector y  = ' + str(len(y)))
        
    #     if k % 1000 == 0 :
            
    #         ImgsFilename2 = cropedTrainAllImgDir + '/' + 'concatdEveryImgsFrom_' + vids[:-4]  + '.mat'
    #         print(' Saving Data to accumlated .mat file : ' + str(ImgsFilename2) + '  ... :)')
    #         everyImgsDict_part = {}
    #         everyImgsDict_part['concatdImgsomeVids']  = ALL_IMAGES
    #         everyImgsDict_part['concatdImgsSomeVid_labels'] = y
            
    #         scipy.io.savemat(ImgsFilename2,everyImgsDict_part)

    # allImgsDict['concatdImgsEveryVidFrames']  = ALL_IMAGES
    # allImgsDict['concatdImgsAllVid_fixed_labels'] = y
    # ImgsFilename = cropedTrainAllImgDir + '/' + 'concatdFixed_labels_EveryVidsData.mat'
    # scipy.io.savemat(ImgsFilename,allImgsDict)

    print('completed processing all videos ... !')
