from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def eachFile(filepath):
	pathDir = os.listdir(filepath)
	out = []
	for allDir in pathDir:
		child = allDir
		out.append(child)
	return out


NUM_CLASSES = 10
TRAIN_PATH = 'dataset/train/'
TEST_PATH = 'dataset/test/'

FC_NUMS = 4096

FREEZE_LAYERS = 17

IMAGE_SIZE = 224

base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(4096, activation='relu')(x)
prediction = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=prediction)

for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(directory=TRAIN_PATH,
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE), classes=eachFile(TRAIN_PATH))
filepath = 'model/sewer_weight.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

history_ft = model.fit_generator(train_generator, epochs=10, callbacks=[checkpoint])
model.save('model/sewer_weight_final.h5')