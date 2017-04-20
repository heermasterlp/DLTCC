import numpy as np
import os
from PIL import Image

'''
    Get numpy array data set from  source images and target images.

    Generate two .npy data set files.
'''

Kai_250_250_400_train = "../../DataSet/SourceImages/kai_128_128_200_train/"
Kai_npy_250_250_400_train = "../../DataSet/DataSetFiles/TrainSet/kai_128_128_200_train.npy"
Qigong_250_250_400_train = "../../DataSet/SourceImages/Qigong_128_128_200_train/"
Qigong_npy_250_250_400_train = "../../DataSet/DataSetFiles/TrainSet/Qigong_128_128_200_train.npy"


Kai_250_250_40_test = "../../DataSet/SourceImages/kai_128_128_20_test/"
Kai_npy_250_250_40_test = "../../DataSet/DataSetFiles/TestSet/kai_128_128_20_test.npy"
Qigong_250_250_40_test = "../../DataSet/SourceImages/Qigong_128_128_20_test/"
Qigong_npy_250_250_40_test = "../../DataSet/DataSetFiles/TestSet/Qigong_128_128_20_test.npy"


# Generate the npy file
def generate(src_dict, target_file):
    files = os.listdir(src_dict)
    if files is None:
        return

    print("File number:", len(files))

    # Convert file to numpy array data with npy formatting.
    with open(target_file, "wb") as tar_file:
        bitmap_list = []

        index = 0
        # read file and get data
        for file in files:
            if ".jpg" not in file:
                continue

            fl = Image.open(os.path.join(src_dict, file))
            fl_bitmap = np.array(fl.convert("1"))

            fl_array = fl_bitmap.reshape(fl_bitmap.shape[0] * fl_bitmap.shape[1])

            if len(fl_array) == 0:
                continue

            bitmap_list.append(fl_array)

            print("index:", index)
            print(fl_array.shape)
            index += 1

        font_bitmaps = np.array(bitmap_list)

        print("font_bitmaps length:", len(font_bitmaps))
        np.save(tar_file, font_bitmaps)


def main():
    # generate(SRC_IMAGES, "kai_image_120.npy")
    # generate(TAR_IMAGES, "qigong_image_120.npy")

    # kai train data set
    # generate(Kai_train_images, "Kai_train_120.npy")

    # qigong train data set
    # generate(Qigong_train_images, "Qigong_train_120.npy")

    # kai test data set
    # generate(Kai_test_images, "Kai_test_20.npy")

    # qigong test data set
    # generate(Qigong_test_images, "Qigong_test_20.npy")

    # generate kai 50x50 200 training data set and npy files
    # generate(Kai_train_50_50_200, Kai_train_50_50_200_npy)

    # generate kai 50x50 40 testing data set and npy files
    # generate(Kai_test_50_50_40, Kai_test_50_50_40_npy)

    # generate qi gong 50x50 200 training data set and npy files
    # generate(Qigong_train_50_50_200, Qigong_train_50_50_200_npy)

    # generate qi gong 50x50 40 testing data set and npy file.
    # generate(Qigong_test_50_50_40, Qigong_test_50_50_40_npy)
    generate(Kai_250_250_400_train, Kai_npy_250_250_400_train)
    generate(Qigong_250_250_400_train, Qigong_npy_250_250_400_train)
    generate(Kai_250_250_40_test, Kai_npy_250_250_40_test)
    generate(Qigong_250_250_40_test, Qigong_npy_250_250_40_test)



if __name__ == "__main__":
    main()