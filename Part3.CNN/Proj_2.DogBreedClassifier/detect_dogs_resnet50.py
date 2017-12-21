import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import prepare_data

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(prepare_data.path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return [((prediction <= 268) & (prediction >= 151)), prediction]


human_files_short = prepare_data.human_files[:100]
dog_files_short = prepare_data.train_files[:100]

# in human files
pos_idx = {}
total_cnt = 0
for idx, img_path in enumerate(human_files_short):
    pos, prediction = dog_detector(img_path)
    if pos == True:
        pos_idx[idx] = prediction
    total_cnt += 1
print('{}% is detected in humans.'.format(len(pos_idx) / total_cnt * 100))

# in dog files
pos_idx.clear()
total_cnt = 0
for idx, img_path in enumerate(dog_files_short):
    pos, prediction = dog_detector(img_path)
    if pos == True:
        pos_idx[idx] = img_path
    total_cnt += 1
print('{}% is detected in dogs.'.format(len(pos_idx) / total_cnt * 100))
