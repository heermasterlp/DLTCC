

# Characters dictionary of Chinese.
CHARACTER_DICT_FILE = "../character_dict/chinese_characters.txt"

# Fonts Files
KAI_FONTS_DICT = "../ttf_files/fangzhengkaisc.TTF"
QIDONG_FONTS_DICT = "../ttf_files/qigongscfont.TTF"

# ImageDisplay files storage path
KAI_TRAIN_IMAGES = "../../DataSet/SourceImages"
KAI_TEST_IMAGES = "../../DataSet/SourceImages"
QIGONG_TRAIN_IMAGES = "../../DataSet/SourceImages"
QIGONG_TEST_IMAGES = "../../DataSet/SourceImages"

# npy files storage path
KAI_TRAIN_NPY = "../../DataSet/DataSetFiles/TrainSet"
KAI_TEST_NPY = "../../DataSet/DataSetFiles/TestSet"
QIGONG_TRAIN_NPY = "../../DataSet/DataSetFiles/TrainSet"
QIGONG_TEST_NPY = "../../DataSet/DataSetFiles/TestSet"

'''
    Generate the train and test data sets with npy formatting based the condition.
    input: image_shape(width, height),train_dataset_size, test_dataset_size
'''


def generate_train_test_dataset_npy_files(image_shape, train_size, test_size):

    if train_size == 0 or test_size == 0:
        print("Data sets size should not equals 0. train_size={0}, test_size={1}".format(train_size, test_size))

    # images file path

    # npy file path

    # Generate training and testing data sets
    img_width, img_height = image_shape

    assert img_width == img_height == 0

    # Generate the images files

