from __future__ import absolute_import

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk
import numpy as np
import matplotlib as plt
plt.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf

import dltcc_models
import os
import input_data

# 200x200 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_150_150_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_150_150_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_150_150_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_150_150_20_test.npy"

# validation size
VALIDATION_SIZE = 50

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}
model_path = "../../checkpoints/models_150_200"
checkpoint_path = "../../checkpoints/checkpoints_150_150"

# threshold
THEROSHOLD = 0.6

input_list = []
target_list = []
image_index = 0

top = tk.Tk()
top.geometry('1200x800') #window size
top.title('Comparision')#window title
var_c = tk.StringVar()#stroe the value of covered
var_l = tk.StringVar()#store the value of the lack points
var_m = tk.StringVar()#store the value of the much points
var_total = tk.StringVar()
var_o = tk.StringVar()
var_p = tk.StringVar()

var_accuracy = tk.StringVar()
E_Accuracy = tk.Entry(top, textvariable=var_accuracy, borderwidth=3, highlightcolor='red', state='disabled')

# label_C = tk.Label(top, text='C:')
# label_L = tk.Label(top, text='L：')
# label_M = tk.Label(top, text='M：')
label_O = tk.Label(top, text='Input Image')
label_P = tk.Label(top, text='Predict Image')
label_true = tk.Label(top, text='True Image')
label_compare = tk.Label(top, text='Compare Result')

# label_CR = tk.Label(top, text='Result')
label_accuracy = tk.Label(top, text='Accuracy:')
# label_O_blackpoints = tk.Label(top, text='Bw(o):')
# label_P_blackpoints = tk.Label(top, text='Bw(p):')

# xx.grid(row = 0, column = 2, columnspan = 2, rowspan = 2, sticky = W+E+N+S, padx = 5, pady = 5)
# E_C = tk.Entry(top, textvariable=var_c, borderwidth=3, highlightcolor='red', state='disabled')
# E_L = tk.Entry(top, textvariable=var_l, borderwidth=3, highlightcolor='red', state='disabled')
# E_M = tk.Entry(top, textvariable=var_m, borderwidth=3, highlightcolor='red', state='disabled')
#
# E_Total = tk.Entry(top, textvariable =var_total, borderwidth=3, highlightcolor='red', state='disabled')
# E_O = tk.Entry(top, textvariable=var_o, borderwidth=3, highlightcolor='red', state='disabled')
# E_P = tk.Entry(top, textvariable=var_p, borderwidth=3, highlightcolor='red', state='disabled')

canvas_input = tk.Canvas(top, width=300, height=300, bg='white')
canvas_predict = tk.Canvas(top, width=300, height=300, bg='white')
canvas_true = tk.Canvas(top, width=300, height=300, bg='white')
canvas_compare = tk.Canvas(top, width=300, height=300, bg='white')


label_O.grid(row=0, column=1, sticky=tk.W+tk.E)
label_P.grid(row=0, column=2, sticky=tk.W+tk.E)
label_compare.grid(row=2, column=1, sticky=tk.W+tk.E)
label_true.grid(row=2, column=2, sticky=tk.W+tk.E)

canvas_input.grid(row=1, column=1)
canvas_predict.grid(row=1, column=2)
canvas_compare.grid(row=3, column=1)
canvas_true.grid(row=3, column=2)


img_TK_Input = None
img_TK_P = None
img_P = None
img_Input = None
img_TK_True = None
img_True = None

fig = Figure(figsize=(3, 3), dpi=100)
fig.canvas = FigureCanvasTkAgg(fig, master=top)
fig.canvas.show()
fig.canvas.get_tk_widget().grid(row=3, column=0)

# Data set
data_set = input_data.read_data_sets(train_dir, validation_size=40)

# place variable
x = tf.placeholder(tf.float32, shape=[None, 150 * 150], name="x")
phase_train = tf.placeholder(tf.bool, name='phase_train')

# Build models
dltcc_obj = dltcc_models.Dltcc()
dltcc_obj.build(x, phase_train)

# Saver
saver = tf.train.Saver()

# output probability
sess = tf.Session()
# Reload the well-trained models
ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("The checkpoint models found!")
else:
    print("The checkpoint models not found!")


def btnHelloClicked():
    var_c.set("entry")


def binary(X):
    W, H = X.shape
    for w in range(W):
        for h in range(H):
            if X[w, h] < 128:
                X[w, h] = 0
            else:
                X[w, h] = 1
    return X


def comparion(X, Y):
    x = []
    y = []
    x_l = []
    y_l = []
    x_m = []
    y_m = []
    np.set_printoptions(threshold=1e6)
    try:
        # X = X.convert('L')
        # Y = Y.convert('L')
        # X = (np.asarray(X)/255).astype(dtype = np.uint8)
        # print(X)
        X = (np.asarray(X)).astype(dtype=np.uint8)
        X = binary(X)
        # print(X)

        # Y = (np.asarray(Y)/255).astype(dtype = np.uint8)
        Y = (np.asarray(Y)).astype(dtype=np.uint8)
        Y = binary(Y)
        # print(Y)
        np.set_printoptions(threshold=1e6)

        # print(X)
        W, H = X.shape
        # print(W,H)
        total_points = Y.sum()
    except Exception as e:
        return
    correct_count = 0
    less = 0
    much = 0
    O_black_p = 0
    p_black_p = 0
    for w in range(W):
        for h in range(H):
            if X[w, h] == 1:
                p_black_p += 1
            if Y[w, h] == 1:
                O_black_p += 1
    for w in range(W):
        for h in range(H):
            if (X[w, h] == Y[w, h]) and(X[w, h] == 1):
                correct_count += 1
                # print(w,h)
                x.append(h)
                y.append(w)
            else:
                if (Y[w, h] == 1) and (X[w, h] == 0):
                    less += 1
                    x_l.append(h)
                    y_l.append(w)
                if(Y[w, h] == 0) and (X[w, h] == 1):
                    much += 1
                    x_m.append(h)
                    y_m.append(w)

    var_c.set(float(correct_count/total_points))
    print(correct_count)
    var_l.set(less)
    var_m.set(much)
    var_total.set(total_points)
    var_o.set(O_black_p)
    var_p.set(p_black_p)
    drawPic(x, y, x_l, y_l, x_m, y_m)


def compare():
    comparion(img_P, img_Input)


def open_o_image(canvas_input, canvas_true):
    filename = askopenfilename(initialdir='../')
    if filename == "":
        return
    try:

        truefilename = filename.replace("Kai", "Qigong")

        global img_TK_Input
        global img_Input
        global img_TK_True
        global img_True

        image = Image.open(filename, mode='r')
        w, h = image.size
        img_Input = image
        img_TK_Input = ImageTk.PhotoImage(image)
        # print(img)

        # True image
        true_image = Image.open(truefilename, mode='r')
        t_w, t_h = true_image.size
        img_TK_True = ImageTk.PhotoImage(true_image)
        img_True = true_image

        canvas_input.create_image(w, h, image=img_TK_Input)
        canvas_true.create_image(t_w, t_h, image=img_TK_True)
    except Exception as e:
        return


def predict(canvas):
    if img_Input is None or img_True is None:
        print("input or true should not none")
    global img_TK_P

    input_array = img_Input.reshape(img_Input.shape[0] * img_Input.shape[1])
    input_array = np.reshape(input_array, [-1, img_Input.shape[0] * img_Input.shape[1]])

    if input_array is None:
        print("Input should not none!")

    # prediction shape: [batch_size, width * height]
    prediction = sess.run(dltcc_obj.y_prob, feed_dict={x: input_array, phase_train: False})

    if prediction is None:
        return
    img_pred = []
    for index in range(prediction.shape[1]):
        if prediction[0][index] >= THEROSHOLD:
            img_pred.append(1)
        else:
            img_pred.append(0)

    predict_array = np.array(img_pred)
    print(predict_array)

    predict_reshape = predict_array.reshape((img_Input.shape[0], img_Input.shape[1]))
    predict_image = Image.fromarray(np.uint8(predict_reshape) * 255)

    img_TK_P = ImageTk.PhotoImage(predict_image)
    pred_w = img_Input.shape[0]
    pred_h = img_Input.shape[1]
    canvas.create_image(pred_h, pred_w, image=img_TK_P)

    # calculate accuracy
    img_true_array = np.array(img_True.convert("1"))
    img_true_array = np.reshape(img_true_array, 22500)
    diff = np.abs(np.subtract(img_true_array, predict_array))
    sum = np.sum(diff)
    accuracy = 1 - sum / 22500
    var_accuracy.set(accuracy)


def open_p_image(canvas):
    filename = askopenfilename(initialdir='./')
    if filename == "":
        return
    try:
        image = Image.open(filename, mode='r')
        w, h = image.size
        global img_TK_P
        global img_P
        img_P = image
        img_TK_P = ImageTk.PhotoImage(image)
        canvas.create_image(w, h, image=img_TK_P)
    except Exception as e:
        return


def drawPic(x, y, x_l, y_l, x_m, y_m):
    fig.clf()
    sub_fig = fig.add_subplot(111)
    sub_fig.invert_yaxis()
    # sub_fig.axis('off')
    # sub_fig.invert_xaxis()
    sub_fig.scatter(x, y, s=3, color='g')
    sub_fig.scatter(x_m, y_m, s=3, color='b')
    sub_fig.scatter(x_l, y_l, s=3, color='m')
    # sub_fig.plot(x_m, y_m, color = 'b')
    fig.canvas.show()


# def open(canvas_input, canvas_true):
#     global input_list
#     global target_list
#     input_list = []
#     target_list = []
#
#     file_directory = askdirectory()
#     if file_directory is None:
#         return
#     target_directory = file_directory.replace('Kai', 'Qigong')
#     # print(file_directory)
#     files = os.listdir(file_directory)
#     for fl in files:
#         if '.jpg' in fl:
#             input_list.append(os.path.join(file_directory, fl))
#             target_list.append(os.path.join(target_directory, fl))
#     print(input_list)
#
#     # show the first image of input and target
#     try:
#         global image_index
#         input_image = input_list[image_index]
#         target_image = target_list[image_index]
#
#         global img_TK_Input
#         global img_Input
#         global img_TK_True
#         global img_True
#
#         image = Image.open(input_image, mode='r')
#         w, h = image.size
#         img_Input = image
#         img_TK_Input = ImageTk.PhotoImage(image)
#         # print(img)
#
#         # True image
#         true_image = Image.open(target_image, mode='r')
#         t_w, t_h = true_image.size
#         img_TK_True = ImageTk.PhotoImage(true_image)
#         img_True = true_image
#
#         canvas_input.create_image(w, h, image=img_TK_Input)
#         canvas_true.create_image(t_w, t_h, image=img_TK_True)
#     except Exception as e:
#         return

def open(canvas_input, canvas_true):

    input_filename = askopenfilename()
    if input_filename is None:
        return

    target_filename = input_filename.replace('Kai', 'Qigong')
    input_array = np.load(input_filename)
    target_array = np.load(target_filename)

    if input_array is None or target_array is None:
        return

    # show the first image of input and target
    try:
        global image_index
        input_image = input_array[image_index]
        target_image = target_array[image_index]

        global img_TK_Input
        global img_Input
        global img_TK_True
        global img_True

        input_img = np.array(input_image)
        img_reshape = input_img.reshape((150, 150))
        image = Image.fromarray(np.uint8(img_reshape) * 255)
        w, h = image.size
        img_Input = img_reshape
        img_TK_Input = ImageTk.PhotoImage(image)
        # print(img)

        # Target image
        target_img = np.array(target_image)
        target_reshape = target_img.reshape((150, 150))
        true_image = Image.fromarray(np.uint8(target_reshape) * 255)
        t_w, t_h = true_image.size
        img_TK_True = ImageTk.PhotoImage(true_image)
        img_True = target_reshape

        canvas_input.create_image(w, h, image=img_TK_Input)
        canvas_true.create_image(t_w, t_h, image=img_TK_True)
    except Exception as e:
        return


def next(canvas_input, canvas_true):
    global image_index
    image_index += 1
    image_index = image_index % len(input_list)
    try:
        input_image = input_list[image_index]
        target_image = target_list[image_index]

        global img_TK_Input
        global img_Input
        global img_TK_True
        global img_True

        image = Image.open(input_image, mode='r')
        w, h = image.size
        img_Input = image
        img_TK_Input = ImageTk.PhotoImage(image)
        # print(img)

        # True image
        true_image = Image.open(target_image, mode='r')
        t_w, t_h = true_image.size
        img_TK_True = ImageTk.PhotoImage(true_image)
        img_True = true_image

        canvas_input.create_image(w, h, image=img_TK_Input)
        canvas_true.create_image(t_w, t_h, image=img_TK_True)
    except Exception as e:
        return


def previous(canvas_input, canvas_true):
    global image_index
    image_index -= 1
    if image_index < 0:
        image_index += len(input_list)

    try:
        input_image = input_list[image_index]
        target_image = target_list[image_index]

        global img_TK_Input
        global img_Input
        global img_TK_True
        global img_True

        image = Image.open(input_image, mode='r')
        w, h = image.size
        img_Input = image
        img_TK_Input = ImageTk.PhotoImage(image)
        # print(img)

        # True image
        true_image = Image.open(target_image, mode='r')
        t_w, t_h = true_image.size
        img_TK_True = ImageTk.PhotoImage(true_image)
        img_True = true_image

        canvas_input.create_image(w, h, image=img_TK_Input)
        canvas_true.create_image(t_w, t_h, image=img_TK_True)
    except Exception as e:
        return


# btn_O = tk.Button(top, text="Open", command=lambda: open_o_image(canvas_input, canvas_true))
btn_O = tk.Button(top, text="Open", command=lambda: open(canvas_input, canvas_true))
btn_P = tk.Button(top, text="Predict", command=lambda: predict(canvas_predict))
btn_Q = tk.Button(top, text="Quit", command=lambda: top.quit())
btn_pre = tk.Button(top, text='Previous', command=lambda: previous(canvas_input, canvas_true))
btn_next = tk.Button(top, text='next', command=lambda: next(canvas_input, canvas_true))

# btn_C = tk.Button(top, text="Compare", command=compare)
btn_O.grid(row=1, column=0, sticky=tk.N+tk.W+tk.E)
btn_P.grid(row=1, column=0, sticky=tk.W+tk.E)
# label_accuracy.grid(row=1, column=0, sticky=tk.S+tk.W+tk.E)
E_Accuracy.grid(row=1, column=0, sticky=tk.S+tk.W+tk.E)

btn_pre.grid(row=2, column=0, sticky=tk.N+tk.W+tk.E)
btn_next.grid(row=2, column=0, sticky=tk.W+tk.E)
btn_Q.grid(row=2, column=0, sticky=tk.S+tk.W+tk.E)
# btn_C.grid(row=3, column=1, sticky=tk.S+tk.W+tk.E)
while True:
    try:
        top.mainloop()
        break
    except UnicodeDecodeError:
        pass
