import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image,ImageTk
import numpy as np
import matplotlib as plt

plt.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
top = tk.Tk();
# top.geometry('600x400');#window size
top.title('Comparision');#window title
var_c=tk.StringVar();#stroe the value of covered
var_l=tk.StringVar();#store the value of the lack points
var_m=tk.StringVar();#store the value of the much points
var_total=tk.StringVar()
var_o=tk.StringVar()
var_p=tk.StringVar()
label_C=tk.Label(top,text='C:');
label_L=tk.Label(top,text='L：');
label_M=tk.Label(top,text='M：');
label_O=tk.Label(top,text='Original font');
label_P=tk.Label(top,text='Output font');
label_CR=tk.Label(top,text='Result');
label_totalpoints=tk.Label(top,text='Tp:');
label_O_blackpoints=tk.Label(top,text='Bw(o):');
label_P_blackpoints=tk.Label(top,text='Bw(p):');

# xx.grid(row=0, column=2, columnspan=2, rowspan=2, sticky=W+E+N+S, padx=5, pady=5)
E_C=tk.Entry(top,textvariable=var_c,borderwidth=3,highlightcolor = 'red',state='disabled')
E_L=tk.Entry(top,textvariable=var_l,borderwidth=3,highlightcolor = 'red',state='disabled')
E_M=tk.Entry(top,textvariable=var_m,borderwidth=3,highlightcolor = 'red',state='disabled')
E_Total=tk.Entry(top,textvariable=var_total,borderwidth=3,highlightcolor = 'red',state='disabled')
E_O=tk.Entry(top,textvariable=var_o,borderwidth=3,highlightcolor = 'red',state='disabled')
E_P=tk.Entry(top,textvariable=var_p,borderwidth=3,highlightcolor = 'red',state='disabled')
canvas_o=tk.Canvas(top,width=300,height=300,bg='black');
canvas_p=tk.Canvas(top,width=300,height=300,bg='black');
label_O.grid(row=0,column=0,sticky=tk.W+tk.E);
label_P.grid(row=0,column=1,sticky=tk.W+tk.E);
label_CR.grid(row=2,column=0,sticky=tk.W+tk.E);
label_C.grid(row=4,column=0,sticky=tk.W);
label_L.grid(row=5,column=0,sticky=tk.W);
label_M.grid(row=6,column=0,sticky=tk.W);
label_totalpoints.grid(row=4,column=1,sticky=tk.W);
label_O_blackpoints.grid(row=5,column=1,sticky=tk.W);
label_P_blackpoints.grid(row=6,column=1,sticky=tk.W);
E_C.grid(row=4,column=0)
E_L.grid(row=5,column=0)
E_M.grid(row=6,column=0)
E_Total.grid(row=4,column=1,sticky=tk.E)
E_O.grid(row=5,column=1,sticky=tk.E)
E_P.grid(row=6,column=1,sticky=tk.E)
img_TK_O=None;
img_TK_P=None;
img_P=None;
img_O=None;
def btnHelloClicked():
    var_c.set("entry")
def binary(X):
    W,H=X.shape;
    for w in range(W):
        for h in range(H):
            if X[w,h]<128:
                X[w,h]=0;
            else:
                X[w,h]=1;
    return X;
def comparion(X,Y):
    x=[];
    y=[];
    x_l=[];
    y_l=[];
    x_m=[];
    y_m=[];
    np.set_printoptions(threshold=1e6)
    try:
        # X=X.convert('L');
        # Y=Y.convert('L');
        # X=(np.asarray(X)/255).astype(dtype=np.uint8);
        # print(X)
        X=(np.asarray(X)).astype(dtype=np.uint8)
        X=binary(X);
        # print(X);

        # Y=(np.asarray(Y)/255).astype(dtype=np.uint8);
        Y=(np.asarray(Y)).astype(dtype=np.uint8)
        Y=binary(Y);
        # print(Y);
        np.set_printoptions(threshold=1e6)

        # print(X);
        W,H=X.shape;
        # print(W,H)
        total_points=Y.sum();
    except Exception as e:
        return ;
    correct_count=0;
    less=0;
    much=0;
    O_black_p=0;
    p_black_p=0;
    for w in range(W):
        for h in range(H):
            if X[w,h]==1:
                p_black_p+=1;
            if Y[w,h]==1:
                O_black_p+=1;
    for w in range(W):
        for h in range(H):
            if (X[w,h]==Y[w,h]) and(X[w,h]==1):
                correct_count+=1;
                # print(w,h);
                x.append(h);
                y.append(w);
            else:
                if (Y[w,h]==1) and (X[w,h]==0):
                    less+=1;
                    x_l.append(h);
                    y_l.append(w);
                if(Y[w,h]==0) and (X[w,h]==1):
                    much+=1;
                    x_m.append(h);
                    y_m.append(w);

    var_c.set(float(correct_count/total_points))
    print(correct_count);
    var_l.set(less);
    var_m.set(much);
    var_total.set(total_points);
    var_o.set(O_black_p);
    var_p.set(p_black_p);
    drawPic(x,y,x_l,y_l,x_m,y_m);
def compare():
    comparion(img_P,img_O)
def open_o_image(canvas):
    filename=askopenfilename(initialdir='./');
    if filename=="":
        return
    try:
        image=Image.open(filename,mode='r');
        w,h=image.size;
        global img_TK_O;
        global img_O
        img_O=image;
        img_TK_O=ImageTk.PhotoImage(image)
        # print(img)
        canvas.create_image(w,h,image=img_TK_O);
    except Exception as e:
        return
def open_p_image(canvas):
    filename=askopenfilename(initialdir='./');
    if filename=="":
        return
    try:
        image=Image.open(filename,mode='r');
        w,h=image.size;
        global img_TK_P;
        global img_P;
        img_P=image;
        img_TK_P=ImageTk.PhotoImage(image)
        canvas.create_image(w,h,image=img_TK_P);
    except Exception as e:
        return
canvas_o.grid(row=1, column=0)
canvas_p.grid(row=1,column=1)

fig = Figure(figsize=(3,3),dpi=100)
fig.canvas = FigureCanvasTkAgg(fig, master=top)
fig.canvas.show()
fig.canvas.get_tk_widget().grid(row=3, column=0);

def drawPic(x,y,x_l,y_l,x_m,y_m):
    fig.clf();
    sub_fig=fig.add_subplot(111)
    sub_fig.invert_yaxis()
    # sub_fig.axis('off');
    # sub_fig.invert_xaxis()
    sub_fig.scatter(x, y, s=3, color='g')
    sub_fig.scatter(x_m, y_m, s=3, color='b')
    sub_fig.scatter(x_l, y_l, s=3, color='m')
    # sub_fig.plot(x_m, y_m, color='b')
    fig.canvas.show()

btn_O = tk.Button(top, text="Original", command=lambda: open_o_image(canvas_o))
btn_P = tk.Button(top, text="Predict", command=lambda: open_p_image(canvas_p))
btn_C = tk.Button(top, text="Compare", command=compare)
btn_O.grid(row=3,column=1,sticky=tk.N+tk.W+tk.E)
btn_P.grid(row=3,column=1,sticky=tk.W+tk.E)
btn_C.grid(row=3,column=1,sticky=tk.S+tk.W+tk.E)
top.mainloop()
