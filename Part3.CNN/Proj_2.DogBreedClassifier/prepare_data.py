from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
import numpy as np
from glob import glob
import pickle
from tqdm import tqdm


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def save_preprocessed_data(path, dict_data):
    with open(path, 'wb') as f:
        pickle.dump(dict_data, f)


def load_preprocessed_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path)
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


working_dir = './udacity_DLNF/Part3.CNN/Proj_2.DogBreedClassifier'
data_filename = working_dir + '/preprocessed.dat'
import os.path
exist_data = os.path.isfile(data_filename)
_data = {}

if exist_data == False:
    """
    ###################################################
    # Import Dog Dataset
    ###################################################
    # Download the dog dataset from 
    # https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
    # Unzip the folder and place it in the repo, 
    # at location path/to/dog-project/dogImages.
    """
    dog_img_path = working_dir + '/dogImages'

    # load train, test, and validation datasets
    train_files, train_targets = load_dataset(dog_img_path + '/train')
    valid_files, valid_targets = load_dataset(dog_img_path + '/valid')
    test_files, test_targets = load_dataset(dog_img_path + '/test')

    # load list of dog names
    dog_names = [item[20:-1]
                 for item in sorted(glob(dog_img_path + '/train/*/'))]
    _data['dog_names'] = dog_names

    # pre-process the data for Keras
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # rescale [0, 255] --> [0, 1]
    train_tensors = paths_to_tensor(train_files).astype('float32') / 255
    valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
    test_tensors = paths_to_tensor(test_files).astype('float32') / 255

    _data['train'] = [train_files, train_targets, train_tensors]
    _data['valid'] = [valid_files, valid_targets, valid_tensors]
    _data['test'] = [test_files, test_targets, test_tensors]

    """
    ###################################################
    # Import Human Dataset
    ###################################################
    # Download the human dataset from
    # https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
    # Unzip the folder and place it in the repo, 
    # at location path/to/dog-project/lfw. 
    # If you are using a Windows machine, 
    # you are encouraged to use 7zip to extract the folder.
    """
    import random
    random.seed(8675309)

    # load filenames in shuffled human dataset
    human_files = np.array(glob(working_dir + '/lfw/*/*'))
    random.shuffle(human_files)
    _data['human_files'] = human_files

    """
    ###################################################
    """
    # save preprocessed data
    print('save preprocessed data ...\n')
    save_preprocessed_data(data_filename, _data)
else:
    print('load preprocessed data ...\n')
    _data = load_preprocessed_data(data_filename)

    train_files, train_targets, train_tensors = _data['train']
    valid_files, valid_targets, valid_tensors = _data['valid']
    test_files, test_targets, test_tensors = _data['test']
    dog_names = _data['dog_names']
    human_files = _data['human_files']

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' %
      len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.\n' % len(test_files))
print('There are %d total human images.\n' % len(human_files))
