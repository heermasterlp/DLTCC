1. 创建原始数据集，主要是楷书和启功字体的图片。图片的格式是二值图，大小初步设置问 800x800。
字库中的汉字总共： 6722 个
生成的楷书的图片数： 6722个
生成的启功字体的图片数：6722个

2. The size of data set images, 100?, 200?, 300?

对比多种不同的方法：
    (1) image的尺寸， 100－150-200-250-.....，是不是包含的信息越丰富，效果越好？
    (2) 模型的结构，是不是模型越复杂，效果越好？
    (3) 模型的设计，设计一个好的模型？

Exact the data of gray scale images(0 or 1) and save to npy files.

Use .npy file to save the bitmap data.


3. Load the data set

data.train.xs = [5000, 200 x 200]
data.train.ys = [5000, 200 x 200]

data.test.xs = [1700, 200 x 200]
data.test.ys = [1700, 200 x 200]



4. Build the models

convolutional layer   filter 确保 还是 50 x 50





5. Train the models



6. evaluate the results.



