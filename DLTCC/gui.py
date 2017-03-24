from tkinter import Tk, Label, Button

class MainGui:
    def __init__(self, master):
        self.master = master
        master.title("DL for TCC v0.1")



root = Tk()
my_gui = MainGui(root)
root.mainloop()