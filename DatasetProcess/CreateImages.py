from PIL import Image, ImageDraw, ImageFont
import os


'''
    Generate the ImageDisplay SourceImages of Kai script and Qigong script and save to gray scale images.
'''

ORIANGL_HEIGHT = 32
ORIANGL_WIDTH = 32
CHARACTER_SIZE = 32

# Characters dictionary of Chinese.
CHARACTER_DICT_FILE = "../character_dict/chinese_characters.txt"
Chars_train_dict = "../character_dict/char_train.txt"
Chars_test_dict = "../character_dict/char_test.txt"

# Fonts Files
KAI_FONTS_DICT = "../ttf_files/fangzhengkaisc.TTF"
QIDONG_FONTS_DICT = "../ttf_files/qigongscfont.TTF"

# ImageDisplay dataset files.
# KAI_IMAGE_FILES = "../../DataSet/SourceImages/Kai_Images_50_50/"
# QIDONG_IMAGE_FILES = "../../DataSet/SourceImages/Qigong_Images_50_50/"

Kai_128_128_200_train = "../../DataSet/SourceImages/Kai_128_128_train/"
Kai_128_128_20_test = "../../DataSet/SourceImages/Kai_128_128_20_test/"
Qigong_128_128_200_train = "../../DataSet/SourceImages/Qigong_256_256_200_train/"
Qigong_128_128_20_test = "../../DataSet/SourceImages/Qigong_256_256_20_test/"

data_dir = "../../DataSet/SourceImages/data_32_32/"


class DataSetGenerate:

    def __init__(self):
        pass

    @staticmethod
    def generate():
        index = 0

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        # Generate images based on the font dict and characters dict.
        with open(CHARACTER_DICT_FILE, mode="r") as input_files:
            contents = input_files.readlines()

            # font file
            kai_font = ImageFont.truetype(KAI_FONTS_DICT, CHARACTER_SIZE)
            qigong_font = ImageFont.truetype(QIDONG_FONTS_DICT, CHARACTER_SIZE)

            for line in contents:
                # print("char: ", line)
                # font images
                character = line.strip()

                # create character images
                rgb_img = Image.new("RGB", (ORIANGL_WIDTH*2, ORIANGL_HEIGHT))
                gray_img = Image.new("1", (ORIANGL_WIDTH*2, ORIANGL_HEIGHT))

                rgb_draw = ImageDraw.Draw(rgb_img)
                # gray_draw = ImageDraw.Draw(gray_img)

                try:
                    rgb_draw.text((0, 0), character, font=kai_font)
                    rgb_draw.text((ORIANGL_WIDTH, 0), character, font=qigong_font)

                except:
                    print("Exception:")

                # convert RGB to gray scale images
                for x in range(rgb_img.width):
                    for y in range(rgb_img.height):
                        rgb_value = rgb_img.getpixel((x, y))
                        if rgb_value == (0, 0, 0):
                            gray_img.putpixel((x, y), 1)
                        else:
                            gray_img.putpixel((x, y), 0)
                # Save the gray scale images.
                gray_img.save(os.path.join(data_dir, "%d_%04d.jpg" % (0, index)))

                rgb_img.close()
                gray_img.close()
                # exit()

                # count
                index += 1
                # if index == 10:
                #     exit()
                print("index:", index)


if __name__ == "__main__":
    DataSetGenerate.generate()



