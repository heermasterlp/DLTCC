from PIL import Image, ImageDraw, ImageFont

import numpy as np



# test image font library
def test_read_character_dict_files():
    character_dict = "../character_dict/chinese_characters.txt"
    with open(character_dict, mode='r') as read_file:
        for line in read_file.readlines():
            # print(line.strip())
            character = line.strip()

            # draw the font images
            img_file = character + ".jpg"

def test_rgb_to_grayscale():

    img = Image.new('RGB', (800, 800))
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("../ttf_files/fangzhengkaisc.TTF", 800)

    draw.text((0, 0), "启功", font=font)

    print(img.width)
    print(img.height)
    print(img.mode)
    for x in range(img.width):
        for y in range(img.height):
            if img.getpixel((x, y)) == (0, 0, 0):
                print(0)

    img.show()


def test_numpy_save():
    ay = np.array([1,0,0,0,0,1,1])
    np.save('file', ay)
    by = np.array([0,0,0,0,0,0,0])
    np.save("file", by)


def test_read_bitmap_and_save_to_array():
    src = "../DataSet/Kai_Images/一.jpg"

    img = Image.open(src, "r")

    print("img shape:", (img.width, img.height))

    bitmap_list = []

    for x in range(img.width):
        for y in range(img.height):
            bitmap_list.append(int(img.getpixel((x, y))))

    print(bitmap_list)


def test_numpy_data_set():
    data = np.load("Kai_test_50_50_40_npy.npy")
    if data is None:
        print("Data is none!")
        return

    print(data[0])
    # for i in data[0]:
    #     if i != 0:
    #         print(i)

    print(data.shape)




if __name__ == "__main__":
    # test_read_character_dict_files()
    # test_rgb_to_grayscale()
    # test_numpy_save()
    # test_read_bitmap_and_save_to_array()
    test_numpy_data_set()