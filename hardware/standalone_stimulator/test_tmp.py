#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2022-03-04 09:47:22
# Name       : test_tmp.py
# Version    : V1.0
# Description: .
#========================================
# importing only those functions which
# are needed
from tkinter import Tk, mainloop, TOP
from tkinter.ttk import Button
from tkinter.messagebox import _show
import tkinter as tk
from tkinter.messagebox import showinfo

root = tk.Tk()
choices = ('network one', 'network two', 'network three')
var = tk.StringVar(root)


def refresh():
    # Reset var and delete all old options
    var.set('')
    network_select['menu'].delete(0, 'end')

    # Insert list of new options (tk._setit hooks them up to var)
    new_choices = ('one', 'two', 'three')
    for choice in new_choices:
        network_select['menu'].add_command(label=choice, command=tk._setit(var, choice))
    showinfo(title='test', message='update')
    print(var.get())

network_select = tk.OptionMenu(root, var, *choices)
network_select.grid()

# I made this quick refresh button to demonstrate
tk.Button(root, text='Refresh', command=refresh).grid()

root.mainloop()

#========================================
# import tkinter as tk


# window = tk.Tk()
# window.title('My Window')
# window.geometry('100x100')

# l = tk.Label(window, bg='white', width=20, text='empty')
# l.pack()

# def print_selection():
    # if (var1.get() == 1) & (var2.get() == 0):
        # l.config(text='I love Python ')
    # elif (var1.get() == 0) & (var2.get() == 1):
        # l.config(text='I love C++')
    # elif (var1.get() == 0) & (var2.get() == 0):
        # l.config(text='I do not anything')
    # else:
        # l.config(text='I love both')

# var1 = tk.IntVar()
# var2 = tk.IntVar()
# c1 = tk.Checkbutton(window, text='Python',variable=var1, onvalue=1, offvalue=0, command=print_selection)
# c1.pack()
# c2 = tk.Checkbutton(window, text='C++',variable=var2, onvalue=1, offvalue=0, command=print_selection)
# c2.pack()

# window.mainloop()
#========================================
# creating tkinter window
# root = Tk()

# button = Button(root, text = 'Geeks')
# button.pack(side = TOP, pady = 5)

# # in after method 5000 milliseconds
# # is passed i.e after 5 seconds
# # a message will be prompted
# root.after(5000, lambda : _show('Title', 'Prompting after 5 seconds'))

# # Destroying root window after 6.7 seconds
# root.after(6700, root.destroy)

# mainloop()

#========================================
# from tkinter import *

# root = Tk()
# root.geometry('250x150')

# button1 = Button(text="Left")
# button1.pack(side = LEFT)

# button2 = Button(text="Top")
# button2.pack(side = TOP)

# button3 = Button(text="Right")
# button3.pack(side = RIGHT)
# button4 = Button(text="Bottom")
# button4.pack(side = BOTTOM)

# root.mainloop()
# from tkinter import *

# def donothing():
    # print ('IT WORKED')
# root=Tk()
# root.title(string='LOGIN PAGE')

# frame1=Frame(root)
# frame1.pack(side=TOP,fill=X)

# frame2=Frame(root)
# frame2.pack(side=TOP, fill=X)

# m=Menu(frame1)
# root.config(menu=m)

# submenu=Menu(m)
# m.add_cascade(label='File',menu=submenu)
# submenu.add_command(label='New File', command=donothing)
# submenu.add_command(label='Open', command=donothing)
# submenu.add_separator()
# submenu.add_command(label='Exit', command=frame1.quit)


# editmenu=Menu(m)
# m.add_cascade(label='Edit', menu=editmenu)
# editmenu.add_command(label='Cut',command=donothing)
# editmenu.add_command(label='Copy',command=donothing)
# editmenu.add_command(label='Paste',command=donothing)
# editmenu.add_separator()
# editmenu.add_command(label='Exit', command=frame1.quit)


# # **** ToolBar *******

# toolbar=Frame(frame1,bg='grey')
# toolbar.pack(side=TOP,fill=X)
# btn1=Button(toolbar, text='Print', command=donothing)
# btn2=Button(toolbar, text='Paste', command=donothing)
# btn3=Button(toolbar, text='Cut', command=donothing)
# btn4=Button(toolbar, text='Copy', command=donothing)
# btn1.pack(side=LEFT,padx=2, fill=BOTH, expand=1)
# btn2.pack(side=LEFT,padx=2, fill=BOTH, expand=1)
# btn3.pack(side=LEFT,padx=2, fill=BOTH, expand=1)
# btn4.pack(side=LEFT,padx=2, fill=BOTH, expand=1)

# # ***** LOGIN CREDENTIALS ******
# label=Label(frame2,text='WELCOME TO MY PAGE',fg='red',bg='white')
# label.grid(row=3,column=1)

# label1=Label(frame2,text='Name')
# label2=Label(frame2,text='Password')
# label1.grid(row=4,column=0,sticky=E)
# label2.grid(row=5,column=0,sticky=E)

# entry1=Entry(frame2)
# entry2=Entry(frame2)
# entry1.grid(row=4,column=1)
# entry2.grid(row=5,column=1)

# chk=Checkbutton(frame2,text='KEEP ME LOGGED IN')
# chk.grid(row=6,column=1)

# btn=Button(frame2,text='SUBMIT')
# btn.grid(row=7,column=1)




# # **** StatusBar ******************

# status= Label(root,text='Loading',bd=1,relief=SUNKEN,anchor=W)
# status.pack(side=BOTTOM, fill=X)

# root.mainloop()
#========================================
# import tkinter as tk

# root = tk.Tk()

# b1 = tk.Button(root, text='b1')
# b2 = tk.Button(root, text='b2')
# b1.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)      # pack starts packing widgets on the left
# b2.pack(side=b1.LEFT)      # and keeps packing them to the next place available on the left
# root.mainloop()

#========================================
# import matplotlib
# matplotlib.use('TkAgg')

# from numpy import arange, sin, pi
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure

# import sys
# if sys.version_info[0] < 3:
    # import Tkinter as Tk
# else:
    # import tkinter as Tk


# def destroy(e):
    # sys.exit()

# root = Tk.Tk()
# root.wm_title("Embedding in TK")


# f = Figure(figsize=(5, 4), dpi=100)
# a = f.add_subplot(111)
# t = arange(0.0, 3.0, 0.01)
# s = sin(2*pi*t)

# a.plot(t, s)
# a.set_title('Tk embedding')
# a.set_xlabel('X axis label')
# a.set_ylabel('Y label')


# # a tk.DrawingArea
# canvas = FigureCanvasTkAgg(f, master=root)
# canvas.draw()
# canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

# canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

# button = Tk.Button(master=root, text='Quit', command=sys.exit)
# button.pack(side=Tk.BOTTOM)

# Tk.mainloop()
