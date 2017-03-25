from __future__ import absolute_import

from PIL import Image
import numpy as np

SIZE = 50
IMAGE_WIDTH = 40
IMAGE_HEIGHT = 200

# Show images
def show_bitmap(img_array):
    if img_array is None:
        print("image array should not none!")
        return
    img_array = np.array(img_array)
    img_reshape = img_array.reshape((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = Image.fromarray(np.uint8(img_reshape) * 255)
    image.show()


def show_graymap(img_array):
    if img_array is None:
        print("Graymap should not none!")
        return
    img_array = np.array(img_array)
    img_reshape = img_array.reshape((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = Image.fromarray(np.float32(img_reshape) * 255)
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
    img_data = np.load(npy_files)

    show_bitmap(img_data[0])

    npy_files = "../../DataSet/DataSetFiles/TrainSet/Kai_250_250_400_train.npy"


    show_bitmap(img_data[0])


def test1():
    # i = Image.open("/Users/liupeng/Documents/dl2tcc/DataSet/SourceImages/Kai_Images_250_250_400_train/二.jpg")
    # i.show()

    npy_files = "../../DataSet/DataSetFiles/TrainSet/Qigong_250_250_400_train.npy"
    img_data = np.load(npy_files)
    # show_bitmap(img_data[10])
    # print(int(img_data[10]))
    print(img_data.shape)

if __name__ == "__main__":
    test1()