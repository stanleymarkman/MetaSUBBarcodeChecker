#Created: Stanley Markman 8/9/18 WCMC
#Provides a GUI for manually validating barcode data associated with the MetaSUB project
import tkinter

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("MetaSUB Barcode Checker")

root = tkinter.Tk()
gui = GUI(root)
root.mainloop()
