import tkinter as tk
from tkinter.ttk import Progressbar

from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk
import numpy as np
import matplotlib as plt
plt.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf
import os

IMG_WIDTH = 200
IMG_HEIGHT = 40

# model saver
model_dir = "../../checkpoints/models_200_200_4_17_heng"

#threshold
THEROSHOLD = 0.1

input_list = []
input_filename_list = []
target_list = []
Image_index = 0
input_dir = None
target_dir = None

predict_result = None

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X_dim = IMG_HEIGHT*IMG_WIDTH
h_dim = 200
z_dim = X_dim

""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])


D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

# Saver
saver = tf.train.Saver()

# output probability
sess = tf.Session()
# Reload the well-trained models
ckpt = tf.train.get_checkpoint_state(model_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("The checkpoint models found!")
else:
    print("The checkpoint models not found!")

root = tk.Tk()
root.geometry('900x800')
root.title('DL2TCC')

img_TK_Input = None
img_TK_Predict = None
img_Predict = None
img_Input = None
img_TK_True = None
img_True = None

var_filelist = tk.StringVar(value=input_filename_list)
var_threshold = tk.StringVar(value=THEROSHOLD)

var_accuracy = tk.StringVar()
E_Accuracy = tk.Entry(root, textvariable=var_accuracy, borderwidth=3, highlightcolor='red', state='disabled')

label_Input = tk.Label(root, text="Input Image")
label_Predict = tk.Label(root, text="Predict Image")
label_True = tk.Label(root, text="True Image")
label_Compare = tk.Label(root, text="Compare Image")

label_Accuracy = tk.Label(root, text="Accuracy:")

canvas_input = tk.Canvas(root, width=300, height=300, bg='white')
canvas_predict = tk.Canvas(root, width=300, height=300, bg='white')
canvas_true = tk.Canvas(root, width=300, height=300, bg='white')
canvas_compare = tk.Canvas(root, width=300, height=300, bg='white')

btn_Open = tk.Button(root, text="Open", command=lambda: open(canvas_input, canvas_true))
btn_Predict = tk.Button(root, text="Predict", command=lambda: predict(canvas_predict, canvas_compare))

lbox = tk.Listbox(root, listvariable=var_filelist, height=10)

theroshold_bar = tk.Scale(root, variable=var_threshold, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.0001,
                          digits=3)

label_Input.grid(row=0, column=2, sticky=tk.W+tk.E)
label_Predict.grid(row=0, column=3, sticky=tk.W+tk.E)
label_Compare.grid(row=2, column=2, sticky=tk.W+tk.E)
label_True.grid(row=2, column=3, sticky=tk.W+tk.E)

canvas_input.grid(row=1, column=2)
canvas_predict.grid(row=1, column=3)
canvas_compare.grid(row=3, column=2)
canvas_true.grid(row=3, column=3)

btn_Open.grid(row=1, column=0, sticky=tk.N+tk.W+tk.E)
btn_Predict.grid(row=1, column=0, sticky=tk.W+tk.E)
# label_accuracy.grid(row=1, column=0, sticky=tk.S+tk.W+tk.E)
E_Accuracy.grid(row=1, column=0, sticky=tk.S+tk.W+tk.E)

lbox.grid(row=3, column=0, padx=5, pady=0, sticky=tk.W+tk.E)

theroshold_bar.grid(row=4, column=0, sticky=tk.W+tk.E)

fig = Figure(figsize=(3, 3), dpi=26)
fig.canvas = FigureCanvasTkAgg(fig, master=root)
fig.canvas.show()
fig.canvas.get_tk_widget().grid(row=3, column=2)


def open(canvas_input, canvas_true):
    global input_list
    global target_list
    global input_filename_list
    global input_dir
    global target_dir

    # clear
    input_list = []
    target_list = []
    input_filename_list = []

    input_directory = askdirectory()
    if input_directory == "":
        return

    var_filelist.set(input_filename_list)

    target_directory = input_directory.replace('Kai', 'Qigong')
    input_dir = input_directory
    target_dir = target_directory
    # input images
    files = os.listdir(input_directory)
    for fl in files:
        if '.jpg' in fl:
            input_list.append(os.path.join(input_directory, fl))
            target_list.append(os.path.join(target_directory, fl))
            input_filename_list.append(fl)

    var_filelist.set(input_filename_list)
    try:
        global Image_index
        input_img = input_list[Image_index]
        target_img = target_list[Image_index]

        global img_TK_Input
        global img_Input

        global img_TK_True
        global img_True

        # input image
        image = Image.open(input_img, mode='r')
        w, h = image.size
        img_Input = image
        img_TK_Input = ImageTk.PhotoImage(image)

        # true image
        image = Image.open(target_img, mode='r')
        t_w, t_h = image.size
        img_True = image
        img_TK_True = ImageTk.PhotoImage(image)

        canvas_input.create_image(w, h, image=img_TK_Input)
        canvas_true.create_image(t_w, t_h, image=img_TK_True)
    except Exception as e:
        return


def update(event):
    widget = event.widget
    selection = widget.curselection()
    select_item = widget.get(selection[0])
    input_img = os.path.join(input_dir, select_item)
    target_img = os.path.join(target_dir, select_item)

    try:
        global Image_index

        index = 0
        for i in input_filename_list:
            if select_item in i:
                break
            index += 1
        Image_index = index
        print(Image_index)

        global img_TK_Input
        global img_Input

        global img_TK_True
        global img_True

        # input image
        image = Image.open(input_img, mode='r')
        w, h = image.size
        img_Input = image
        img_TK_Input = ImageTk.PhotoImage(image)

        # true image
        image = Image.open(target_img, mode='r')
        t_w, t_h = image.size
        img_True = image
        img_TK_True = ImageTk.PhotoImage(image)

        canvas_input.create_image(w, h, image=img_TK_Input)
        canvas_true.create_image(t_w, t_h, image=img_TK_True)
    except Exception as e:
        return

lbox.bind("<Double-Button-1>", func=update)


def predict(canvas, canvas_compare):
    if img_Input is None or img_True is None:
        print("input or true should not none")
    global img_TK_Predict
    global img_Predict

    global THEROSHOLD

    THEROSHOLD = float(var_threshold.get())

    input_bitmap = np.array(img_Input.convert("1"))

    input_array = input_bitmap.reshape(input_bitmap.shape[0] * input_bitmap.shape[1])
    input_array = np.reshape(input_array, [-1, input_bitmap.shape[0] * input_bitmap.shape[1]])

    if input_array is None:
        print("Input should not none!")

    # prediction shape: [batch_size, width * height]
    # prediction = sess.run(dltcc_obj.y_prob, feed_dict={x: input_array, phase_train: False})
    prediction = sess.run(G_sample, feed_dict={Z: input_array})

    if prediction is None:
        return

    pred_arr = np.array(prediction)

    img_pred = []
    for index in range(pred_arr.shape[1]):
        if pred_arr[0][index] >= THEROSHOLD:
            img_pred.append(1)
        else:
            img_pred.append(0)

    predict_array = np.array(img_pred)

    predict_reshape = predict_array.reshape((input_bitmap.shape[0], input_bitmap.shape[1]))
    predict_image = Image.fromarray(np.uint8(predict_reshape) * 255)

    img_TK_Predict = ImageTk.PhotoImage(predict_image)
    pred_w = input_bitmap.shape[0]
    pred_h = input_bitmap.shape[1]
    canvas.create_image(pred_h, pred_w, image=img_TK_Predict)

    # calculate accuracy
    img_true_array = np.array(img_True.convert("1"))
    img_true_array = np.reshape(img_true_array, 8000)
    diff = np.abs(np.subtract(img_true_array, predict_array))
    sum = np.sum(diff)
    accuracy = 1 - sum / 8000
    var_accuracy.set(accuracy)
    img_true_array = np.reshape(img_true_array, (40, 200))

    rgbArray = np.zeros((40, 200, 3), 'uint8')
    rgbArray[..., 0] = predict_reshape * 256
    rgbArray[..., 1] = img_true_array * 256
    compare_img = Image.fromarray(rgbArray, 'RGB')
    compare_TK_img = ImageTk.PhotoImage(compare_img)
    canvas_compare.create_image(pred_h, pred_w, image=compare_TK_img)


def filter(ay, threshold):
    result = []
    for it in range(ay.shape[0]):
        if ay[it] >= threshold:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

def comparion(X, Y):
    x = []
    y = []
    x_l = []
    y_l = []
    x_m = []
    y_m = []
    np.set_printoptions(threshold=1e6)
    W, H = X.shape
    # print(W,H)
    total_points = Y.sum()
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
    drawPic(x, y, x_l, y_l, x_m, y_m)


def drawPic(x, y, x_l, y_l, x_m, y_m):
    fig.clf()
    sub_fig = fig.add_subplot(111)
    sub_fig.invert_yaxis()
    # sub_fig.axis('off')
    # sub_fig.invert_xaxis()
    sub_fig.scatter(x, y, s=1, color='g')
    sub_fig.scatter(x_m, y_m, s=1, color='b')
    sub_fig.scatter(x_l, y_l, s=1, color='m')
    # sub_fig.plot(x_m, y_m, color = 'b')
    fig.canvas.show()


while True:
    try:
        root.mainloop()
        break
    except UnicodeDecodeError:
        pass


