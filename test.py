from keras.initializers import glorot_uniform
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
from keras.utils import CustomObjectScope
from keras.preprocessing.image import  img_to_array

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# image folder
folder_path = 'dataset/test/'

# path to model
model_path = 'model/sewer_weight_9.h5'
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model(model_path)

# dimensions of images
img_width, img_height = 224, 224


i = 0
images = []
for img in os.listdir(folder_path):
   img1 = image.load_img(os.path.join(folder_path, img), target_size=(img_width, img_height))
   img2 = img_to_array(img1)
   img2 = np.expand_dims(img2, axis=0)
   classes = model.predict(img2)[0]
   idxs = np.argsort(classes)[::-1][:1]

   classname = ['IN', 'JOINT,FAULTY', 'PIPE,BROKEN', 'OUT', 'DEBRIS,SILTY', 'JOINT,OPEN',
                'HORIZONTAL,CRACK', 'VERTICAL,CRACK', 'LATERAL,PROTRUDING']

   out = cv2.imread(os.path.join(folder_path, img))

   for (i, j) in enumerate(idxs):
       label = "{}:{:.2f}%".format(classname[idxs[i]], classes[idxs[i]] * 100)
       cv2.putText(out, label, (10, (i * 30) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

   cv2.imwrite("visualization/sd/%s"%img,out)
