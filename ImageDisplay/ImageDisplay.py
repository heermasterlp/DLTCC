from __future__ import absolute_import

from PIL import Image
import numpy as np


# Show images
def show_image(img_array):
    if img_array is None:
        print("image array should not none!")
        return
    img_reshape = img_array.reshape((250, 250))
    image = Image.fromarray(np.uint8(img_reshape) * 255)
    image.show()


# Save images
def save_image(img_array, path):
    if img_array is None:
        print("image array should not none!")
        return
    img_reshape = img_array.reshape((250, 250))
    image = Image.fromarray(np.uint8(img_reshape) * 255)
    image.save(path)


def test():
    npy_files = "../../DataSet/DataSetFiles/TrainSet/Qigong_250_250_400_train.npy"
    img_data = np.load(npy_files)

    show_image(img_data[0])

    npy_files = "../../DataSet/DataSetFiles/TrainSet/Kai_250_250_400_train.npy"
    img_data = np.load(npy_files)

    show_image(img_data[0])


def test1():
    # i = Image.open("/Users/liupeng/Documents/dl2tcc/DataSet/SourceImages/Kai_Images_250_250_400_train/二.jpg")
    # i.show()

    npy_files = "../../DataSet/DataSetFiles/TestSet/Qigong_250_250_40_test.npy"
    img_data = np.load(npy_files)
    show_image(img_data[10])

if __name__ == "__main__":
    test1()