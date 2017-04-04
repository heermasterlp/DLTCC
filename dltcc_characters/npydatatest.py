import numpy as np
import os

from PIL import Image, ImageDraw

image_files = "/Users/liupeng/Documents/dl2tcc/DataSet/SourceImages/Kai_Images_150_150_200_train"

npy_file = "/Users/liupeng/Documents/dl2tcc/DataSet/DataSetFiles/TrainSet/Kai_150_150_200_train.npy"


def test():
    img_files = os.listdir(image_files)

    npy_data = np.load(npy_file)

    if img_files is None:
        print("image is none!")
        return
    print(img_files)
    img_list = []
    img_dist_list = []
    for img_item in img_files:
        print(img_item)
        img = Image.open(os.path.join(image_files, img_item))

        input_bitmap = np.array(img.convert("1"))

        input_array = input_bitmap.reshape(input_bitmap.shape[0] * input_bitmap.shape[1])
        img_dist_list.append({img_item:input_array})
        img_list.append(input_array)
    img_list = np.array(img_list)

    print(img_list.shape)
    print(npy_data.shape)
    print(img_dist_list)

    print(np.array_equal(img_list, npy_data))


test()







