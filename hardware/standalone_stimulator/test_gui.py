#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-08-19 20:13:48
# Name       : test_gui.py
# Version    : V1.0
# Description: A minimal example for tkinter based gui
#========================================
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import tkinter as tk
import tkinter.font as tkFont
from tkinter import Label, Entry, Button, END


class MyWindow:
    def __init__(self, window, fontsize=20):
        self.window = window
        self.window_geometry()
        self.fontStyle = tkFont.Font(family="Times New Roman", size=fontsize) # "Lucida Grande"
        self.button_place()
        self.wave_display()

        self.window.mainloop()

    def button_place(self):

        self.ch1_label = Label(self.window, text='CH1', font=self.fontStyle)
        self.ch2_label = Label(self.window, text='CH2', font=self.fontStyle)

        self.freq_label = Label(self.window, text='Frequency (Hz)', font=self.fontStyle)
        self.phase_label = Label(self.window, text='Phase (Degree)', font=self.fontStyle)
        self.amp_label = Label(self.window, text='Voltage (V)', font=self.fontStyle)
        self.offset_label = Label(self.window, text='Offset (V)', font=self.fontStyle)
        self.output_label = Label(self.window, text='Output', font=self.fontStyle)

        self.freq_entry_ch1 = Entry(self.window, font=self.fontStyle)
        self.phase_entry_ch1 = Entry(font=self.fontStyle)
        self.amp_entry_ch1 = Entry(font=self.fontStyle)
        self.offset_entry_ch1 = Entry(font=self.fontStyle)
        # self.output_entry_ch1 = Entry(font=self.fontStyle)

        self.freq_entry_ch2 = Entry(font=self.fontStyle)
        self.phase_entry_ch2 = Entry(font=self.fontStyle)
        self.amp_entry_ch2 = Entry(font=self.fontStyle)
        self.offset_entry_ch2 = Entry(font=self.fontStyle)
        # self.output_entry_ch2 = Entry(font=self.fontStyle)

        self.ch1_label.grid(row=0, column=2)
        self.ch2_label.grid(row=0, column=3)

        self.freq_label.grid(row=1, column=1)
        self.phase_label.grid(row=2, column=1)
        self.amp_label.grid(row=3, column=1)
        self.offset_label.grid(row=4, column=1)
        self.output_label.grid(row=5, column=1)

        self.freq_entry_ch1.grid(row=1, column=2)
        self.phase_entry_ch1.grid(row=2, column=2)
        self.amp_entry_ch1.grid(row=3, column=2)
        self.offset_entry_ch1.grid(row=4, column=2)
        # self.output_entry_ch1.grid(row=4, column=2)

        self.freq_entry_ch2.grid(row=1, column=3)
        self.phase_entry_ch2.grid(row=2, column=3)
        self.amp_entry_ch2.grid(row=3, column=3)
        self.offset_entry_ch2.grid(row=4, column=3)
        # self.output_entry_ch2.grid(row=4, column=3)

    def wave_display(self):

        self.fig, self.ax = plt.subplots(2, 1)
        self.ax[0].set_title('CH1')
        self.ax[0].set_facecolor('xkcd:black')
        self.ax[0].grid(True)
        self.ax[0].set_xlabel('Time/sec')
        self.ax[0].set_ylabel('Voltage/V')

        self.ax[1].set_title('CH2')
        self.ax[1].set_facecolor('xkcd:black')
        self.ax[1].grid(True)
        self.ax[1].set_xlabel('Time/sec')
        self.ax[1].set_ylabel('Voltage/V')

        plt.ion()
        t = np.arange(0.0,3.0,0.01)
        s = np.sin(np.pi*t)
        self.ax[0].plot(t,s)
        self.fig.tight_layout()

        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        plot_widget = canvas.get_tk_widget()

        def update():
            s = np.cos(np.pi*t)
            self.ax[0].plot(t,s, color='xkcd:yellow green')
            self.ax[1].plot(t,s, color='xkcd:sky blue')
            self.fig.tight_layout()
            self.fig.canvas.draw()

        # plot_widget.pack(expand=True, side=LEFT)
        plot_widget.grid(row=0, column=0, rowspan=7)
        tk.Button(self.window, text="CH1", command=lambda: self.signal_out(chn=1)).grid(row=5, column=2,)
        tk.Button(self.window, text="CH2", command=lambda: self.signal_out(chn=2)).grid(row=5, column=3)
        tk.Button(self.window, text="Update Parameters", command=self.sin_update).grid(row=6, column=1, columnspan=3)
        # self.window.mainloop()
    def signal_out(self, chn):
        if chn == 1:
            print('CH1')
        else:
            print('CH2')
        pass


    def sin_update(self):

        self.freq_val_ch1 = float(self.freq_entry_ch1.get())
        self.phase_val_ch1 = float(self.phase_entry_ch1.get())
        self.amp_val_ch1 = float(self.amp_entry_ch1.get())
        self.offset_val_ch1 = float(self.offset_entry_ch1.get())

        self.freq_val_ch2 = float(self.freq_entry_ch2.get())
        self.phase_val_ch2 = float(self.phase_entry_ch2.get())
        self.amp_val_ch2 = float(self.amp_entry_ch2.get())
        self.offset_val_ch2 = float(self.offset_entry_ch2.get())

        t = np.arange(0.0, 1.0, 0.001)
        ch1_signal = self.offset_val_ch1 + self.amp_val_ch1 * np.sin(
            2*np.pi*self.freq_val_ch1*t + self.phase_val_ch1 / 180 * np.pi)
        self.ax[0].clear()
        self.ax[0].plot(t, ch1_signal, color='xkcd:yellow green')

        ch2_signal = self.offset_val_ch2 + self.amp_val_ch2 * np.sin(
            2*np.pi*self.freq_val_ch2*t + self.phase_val_ch2 / 180 * np.pi)
        self.ax[1].clear()
        self.ax[1].plot(t, ch2_signal, color='xkcd:sky blue')

        self.ax[0].set_title('CH1')
        self.ax[0].set_facecolor('xkcd:black')
        self.ax[0].grid(True)
        self.ax[0].set_xlabel('Time/sec')
        self.ax[0].set_ylabel('Voltage/V')
        self.ax[1].set_title('CH2')
        self.ax[1].set_facecolor('xkcd:black')
        self.ax[1].grid(True)
        self.ax[1].set_xlabel('Time/sec')
        self.ax[1].set_ylabel('Voltage/V')

        self.fig.tight_layout()
        self.fig.canvas.draw()

        self.scpi_cmd = {'voltage': [self.amp_val_ch1, self.amp_val_ch2],
                         'frequency': [self.freq_val_ch1, self.freq_val_ch2],
                         'phase': [self.phase_val_ch1, self.phase_val_ch2],
                         'offset': [self.offset_val_ch1, self.offset_val_ch2]}

        # SG().para_set(self.cmd)

    def button_switch(self):
        import tkinter as tk
        from PIL import Image, ImageTk

        def change_pic():
            print(b2['state'])
            if b2["state"] == "normal":
                # b2["text"] = "enable"
                b2["state"] = "active"
                vlabel.configure(image=self.window.photo1)
                print('active')
            else:
                b2["state"] = "normal"
                b2["text"] = "disable"
                vlabel.configure(image=self.window.photo)
                print('normal')

        photo = 'Figures/button1.jpeg'
        photo1 = "Figures/button2.jpeg"
        self.window.photo = ImageTk.PhotoImage(Image.open(photo))
        self.window.photo0 = ImageTk.PhotoImage(Image.open(photo1))

        vlabel=tk.Label(self.window, image=self.window.photo)
        vlabel.pack()

        b2=tk.Button(window,text="Capture",command=change_pic)
        b2.pack()

        self.window.mainloop()
    def window_geometry(self):
        for id_row, row in enumerate([1]*7):
            self.window.grid_rowconfigure(id_row, weight=row)
        for id_col, col in enumerate([5, 2, 2, 2]):
            self.window.grid_columnconfigure(id_col, weight=col)

        screen_width = self.window.winfo_screenwidth()
        screen_height = self. window.winfo_screenheight()
        self.window_width = screen_width * .4
        self.window_height = screen_height * .2
        self.window_start_x = 100
        self.window_start_y = 100
        window.geometry("%dx%d+%d+%d" % (self.window_width, self.window_height,
                                         self.window_start_x,
                                         self.window_start_y))

window = tk.Tk()
mywin = MyWindow(window)
window.title('Hello Python')
mywin.wave_display()
window.mainloop()

