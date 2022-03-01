#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-08-10 20:13:48
# Name       : test_gui.py
# Version    : V1.0
# Description: A minimal example for tkinter based gui
#========================================
import numpy as np
import time

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import tkinter as tk
import tkinter.font as tkFont
from tkinter import Label, Entry, Button, OptionMenu, StringVar, END

from PIL import Image, ImageTk

from signal_generator import SignalGenerator as SG

class tACSWindow():
    def __init__(self, window, fontsize=20):
        self.window = window
        self.window_geometry()
        self.fontStyle = tkFont.Font(family="Times New Roman", size=fontsize) # "Lucida Grande"
        self.button_place()
        self.wave_display()
        try:
            self.dev_connect()
            self.dev_available = True
        except:
            self.dev_available = False
            pass

        self.window.mainloop()

    def button_place(self):
        root = '.'  #  '/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/hardware/standalone_stimulator/'
        self.fig_root = root
        subsps_coef = 2

        im_ch1 = tk.PhotoImage(file=f'{root}/Figures/label_ch1.png').subsample(subsps_coef)
        self.ch1_label = Label(self.window, image=im_ch1, font=self.fontStyle, bg="white")
        self.ch1_label.image = im_ch1

        im_ch2 = tk.PhotoImage(file=f'{root}/Figures/label_ch2.png').subsample(subsps_coef)
        self.ch2_label = Label(self.window, image=im_ch2, font=self.fontStyle, bg="white")
        self.ch2_label.image = im_ch2

        im_amp = tk.PhotoImage(file=f'{root}/Figures/label_amp.png').subsample(subsps_coef)
        self.amp_label = Label(self.window, image=im_amp, font=self.fontStyle, bg="white")
        self.amp_label.image = im_amp

        im_freq = tk.PhotoImage(file=f'{root}/Figures/label_freq.png').subsample(subsps_coef)
        self.freq_label = Label(self.window, image=im_freq, font=self.fontStyle, bg="white")
        self.freq_label.image = im_freq

        im_phase = tk.PhotoImage(file=f'{root}/Figures/label_phase.png').subsample(subsps_coef)
        self.phase_label = Label(self.window, image=im_phase, font=self.fontStyle, bg="white")
        self.phase_label.image = im_phase

        im_offset = tk.PhotoImage(file=f'{root}/Figures/label_offset.png').subsample(subsps_coef)
        self.offset_label = Label(self.window, image=im_offset, font=self.fontStyle, bg="white")
        self.offset_label.image = im_offset

        im_fade = tk.PhotoImage(file=f'{root}/Figures/label_fade_dur.png').subsample(subsps_coef)
        self.fade_label = Label(self.window, image=im_fade, font=self.fontStyle, bg="white")
        self.fade_label.image = im_fade

        im_dur = tk.PhotoImage(file=f'{root}/Figures/label_stim_dur.png').subsample(subsps_coef)
        self.duration_label = Label(self.window, image=im_dur, font=self.fontStyle, bg="white")
        self.duration_label.image = im_dur

        im_output = tk.PhotoImage(file=f'{root}/Figures/label_output.png').subsample(subsps_coef)
        self.output_label = Label(self.window, image=im_output, font=self.fontStyle, bg="white")
        self.output_label.image = im_output

        im_stim_type = tk.PhotoImage(file=f'{root}/Figures/label_stim_type.png').subsample(subsps_coef)
        self.stim_label = Label(self.window, image=im_stim_type, font=self.fontStyle, bg="white")
        self.stim_label.image = im_stim_type

        self.freq_entry_ch1 = Entry(self.window, font=self.fontStyle)
        self.phase_entry_ch1 = Entry(self.window, font=self.fontStyle)
        self.amp_entry_ch1 = Entry(self.window, font=self.fontStyle)
        self.offset_entry_ch1 = Entry(self.window, font=self.fontStyle)
        # self.output_entry_ch1 = Entry(font=self.fontStyle)
        self.fade_entry_ch1 = Entry(self.window, font=self.fontStyle)
        self.duration_entry_ch1 = Entry(self.window, font=self.fontStyle)

        self.freq_entry_ch2 = Entry(self.window, font=self.fontStyle)
        self.phase_entry_ch2 = Entry(self.window, font=self.fontStyle)
        self.amp_entry_ch2 = Entry(self.window, font=self.fontStyle)
        self.offset_entry_ch2 = Entry(self.window, font=self.fontStyle)
        # self.output_entry_ch2 = Entry(self.window, font=self.fontStyle)
        self.fade_entry_ch2 = Entry(self.window, font=self.fontStyle)
        self.duration_entry_ch2 = Entry(self.window, font=self.fontStyle)

        self.ch1_label.grid(row=0, column=2)
        self.ch2_label.grid(row=0, column=3)
        self.output_label.grid(row=8, column=1)
        self.stim_label.grid(row=1, column=1)

        self.comp_list = np.asarray(
            [[self.amp_label, self.freq_label, self.phase_label,
              self.offset_label, self.fade_label, self.duration_label],
             [self.amp_entry_ch1, self.freq_entry_ch1, self.phase_entry_ch1,
              self.offset_entry_ch1, self.fade_entry_ch1,
              self.duration_entry_ch1],
             [self.amp_entry_ch2, self.freq_entry_ch2, self.phase_entry_ch2,
              self.offset_entry_ch2, self.fade_entry_ch2,
              self.duration_entry_ch2]], dtype='object')
        self.comp_list = self.comp_list.T
        row_start = 2
        col_start = 1
        for row in range(self.comp_list.shape[0]):
            for col in range(self.comp_list.shape[1]):
                self.comp_list[row, col].grid(row=row+row_start,
                                              column=col+col_start)
        self.init_checkbox()

    def init_checkbox(self):

        stim_option = ['tACS', 'tDCS', 'tRNS']
        self.click_ch1 = StringVar()
        self.click_ch1.set(stim_option[0])
        self.stim_menu_ch1 = OptionMenu(self.window, self.click_ch1, *stim_option,
                                        command=self.entry_state_update)
        self.stim_menu_ch1.grid(row=1, column=2, ipadx=90)
        self.stim_menu_ch1.configure(font=self.fontStyle)
        self.menu_opt_ch1 = self.window.nametowidget(self.stim_menu_ch1.menuname)
        self.menu_opt_ch1.config(font=self.fontStyle)

        self.click_ch2 = StringVar()
        self.click_ch2.set(stim_option[0])
        self.stim_menu_ch2 = OptionMenu(self.window, self.click_ch2, *stim_option,
                                        command=self.entry_state_update)
        self.stim_menu_ch2.grid(row=1, column=3, ipadx=90)
        self.stim_menu_ch2.configure(font=self.fontStyle)
        self.menu_opt_ch2 = self.window.nametowidget(self.stim_menu_ch2.menuname)
        self.menu_opt_ch2.config(font=self.fontStyle)


        # from tkinter import ttk
        # self.click_ch1 = ttk.Combobox(self.window, values=stim_option,
                                      # postcommand=self.entry_state_update)
        # self.click_ch1.grid(row=1, column=2)
        # self.click_ch1.bind("<<ComboboxSelected>>", self.entry_state_update)

    def entry_state_update(self, event):
        for id_click, click in enumerate([self.click_ch1, self.click_ch2]):
            click_val = click.get()

            if click_val == 'tACS':
                state_list = ['normal'] * 6
            elif click_val == 'tDCS':
                state_list = ['normal', 'disable', 'disable',
                              'disable', 'normal', 'normal']
            elif click_val == 'tRNS':
                state_list = ['normal', 'disable', 'disable',
                              'normal', 'normal', 'normal']

            for id_entry, entry_state in enumerate(state_list):
                self.comp_list[id_entry, id_click+1].config(state=entry_state)


    def dev_connect(self):
        self.sig_gen = SG(dev='/dev/usbtmc1')

    def wave_display(self):

        self.fig, self.ax = plt.subplots(2, 1, facecolor=(1, 1, 1),
                                         figsize=(self.window_width*0.4/100, self.window_height*0.8/100))
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
        plot_widget.grid(row=1, column=0, rowspan=7)

        im_on = f'{self.fig_root}/Figures/button_on.png'
        im_off = f'{self.fig_root}/Figures/button_off.png'
        self.window.fig_on = tk.PhotoImage(file=im_on).subsample(3)
        self.window.fig_off = tk.PhotoImage(file=im_off).subsample(3)

        self.bt_out1 = tk.Button(self.window, image=self.window.fig_off, bg="white", command=lambda: self.signal_out(chn=1))
        self.bt_out1.image = self.window.fig_off
        self.bt_out1.grid(row=8, column=2)
        self.bt_out2 = tk.Button(self.window, image=self.window.fig_off, bg="white", command=lambda: self.signal_out(chn=2))
        self.bt_out2.image = self.window.fig_off
        self.bt_out2.grid(row=8, column=3)

        im_update = "./Figures/button_update.png"
        self.window.fig_update = tk.PhotoImage(file=im_update).subsample(3)
        self.bt_update = tk.Button(self.window, image=self.window.fig_update, command=self.para_update)
        self.bt_update.image = self.window.fig_update
        self.bt_update.grid(row=8, column=0)
        # self.window.mainloop()

    def para_update(self):
        for id_click, click in enumerate([self.click_ch1, self.click_ch2]):
            click_val = click.get()
            if click_val == 'tACS':
                self.sin_update(chn=id_click+1)
            elif click_val == 'tDCS':
                self.dc_update(chn=id_click+1)
            elif click_val == 'tRNS':
                self.noise_update(chn=id_click+1)

    def signal_out(self, chn):
        self.load_entry()
        def state_switch(button, channel):
            if button["state"] == "normal":
                button["state"] = "active"
                if self.dev_available:
                    self.sig_gen.on(chn=channel)
                button.configure(image=self.window.fig_on)
                # vlabel.configure(image=self.window.photo1)
                print('active')
            else:
                button["state"] = "normal"
                if self.dev_available:
                    self.sig_gen.off(chn=channel)
                button.configure(image=self.window.fig_off)
                # vlabel.configure(image=self.window.photo)
                print('normal')

        def amp_adjust(val, chn):
            click_list = [self.click_ch1, self.click_ch2]
            click = click_list[chn-1]
            click_val = click.get()

            if self.dev_available:
                if click_val == 'tACS':
                    self.sig_gen.amp(val, chn=chn)
                elif click_val == 'tDCS':
                    self.sig_gen.para_set({'offset': val}, chn=chn)
                elif click_val == 'tRNS':
                    offset = self.offset_val_ch1 if chn == 1 else self.offset_val_ch2
                    self.sig_gen.para_set({'noise': [val, offset]}, chn=chn)

        def fade(amp, fade_dur, chn, step_per_sec=2, status='start'):
            sleep_dur = 1 / step_per_sec
            step_list = np.linspace(0.002, amp, int(fade_dur*step_per_sec))
            if status == 'start':
                # self.sig_gen.amp(0.002, chn=chn)
                amp_adjust(0.002, chn=chn)
                self.sig_gen.on(chn=chn)
                for stim_val in step_list:
                    amp_adjust(val=stim_val, chn=chn)
                    time.sleep(sleep_dur)
            elif status == 'finish':
                for stim_val in step_list[::-1]:
                    amp_adjust(val=stim_val, chn=chn)
                    # self.sig_gen.amp(stim_val, chn=chn)
                    time.sleep(sleep_dur)
                self.sig_gen.off(chn=chn)
#         photo = 'Figures/button1.jpeg'
        # photo1 = "Figures/button2.jpeg"
        # self.window.photo = ImageTk.PhotoImage(Image.open(photo))
        # self.window.photo0 = ImageTk.PhotoImage(Image.open(photo1))


        if chn == 1:
            amplitude, stim_dur, fade_dur = self.amp_val_ch1, self.duration_val_ch1, self.fade_val_ch1
            button_out = self.bt_out1
        else:
            amplitude, stim_dur, fade_dur = self.amp_val_ch2, self.duration_val_ch2, self.fade_val_ch2
            button_out = self.bt_out2

        if stim_dur == '' and fade_dur == '':
            state_switch(button_out, channel=chn)
        elif fade_dur != '':
            assert stim_dur != '', 'When fade is non-empty, duration also must be non-empty'
            fade(amp=amplitude, chn=chn, status='start', fade_dur=float(fade_dur))
            # start_time = time.time()
            # while (time.time() - start_time) < float(self.duration_val_ch1):
            time.sleep(float(stim_dur))
            fade(amp=amplitude, chn=chn, status='finish', fade_dur=float(fade_dur))
        else:
            self.sig_gen.on(chn=chn)
            time.sleep(float(stim_dur))
            self.sig_gen.off(chn=chn)

    def load_entry(self):
        def customize_float(string_val):
            try:
                return float(string_val)
            except ValueError:
                return None

        self.freq_val_ch1 = customize_float(self.freq_entry_ch1.get())
        self.phase_val_ch1 = customize_float(self.phase_entry_ch1.get())
        self.amp_val_ch1 = customize_float(self.amp_entry_ch1.get())
        self.offset_val_ch1 = customize_float(self.offset_entry_ch1.get())
        self.fade_val_ch1 = self.fade_entry_ch1.get()
        self.duration_val_ch1 = self.duration_entry_ch1.get()

        self.freq_val_ch2 = customize_float(self.freq_entry_ch2.get())
        self.phase_val_ch2 = customize_float(self.phase_entry_ch2.get())
        self.amp_val_ch2 = customize_float(self.amp_entry_ch2.get())
        self.offset_val_ch2 = customize_float(self.offset_entry_ch2.get())
        self.fade_val_ch2 = self.fade_entry_ch1.get()
        self.duration_val_ch2 = self.duration_entry_ch1.get()

    def sin_update(self, chn):

        self.load_entry()
        if chn == 1:
            amp, freq, phase, offset = self.amp_val_ch1, self.freq_val_ch1, \
                self.phase_val_ch1, self.offset_val_ch1
            curve_color = 'xkcd:yellow green'
            title = 'CH1'
        elif chn == 2:
            amp, freq, phase, offset = self.amp_val_ch2, self.freq_val_ch2, \
                self.phase_val_ch2, self.offset_val_ch2
            curve_color = 'xkcd:sky blue'
            title = 'CH2'

        t = np.arange(0.0, 1.0, 0.001)
        ch_signal = offset + amp * np.sin(2*np.pi*freq*t + phase / 180 * np.pi)

        self.ax[chn-1].clear()
        self.ax[chn-1].plot(t, ch_signal, color=curve_color)

        self.ax[chn-1].set_title(title)
        self.ax[chn-1].set_facecolor('xkcd:black')
        self.ax[chn-1].grid(True)
        self.ax[chn-1].set_xlabel('Time/sec')
        self.ax[chn-1].set_ylabel('Voltage/V')
        self.fig.tight_layout()
        self.fig.canvas.draw()

        # self.scpi_cmd_ch = {'voltage': amp,
                            # 'frequency': freq,
                            # 'phase': phase,
        #                     'offset': offset}
        self.scpi_cmd_ch = {'sin': [freq, amp, offset, phase]}

        if self.dev_available:
            self.sig_gen.para_set(self.scpi_cmd_ch, chn=chn)

    def dc_update(self, chn):

        self.load_entry()
        if chn == 1:
            amp = self.amp_val_ch1
            curve_color = 'xkcd:yellow green'
            title = 'CH1'
        elif chn == 2:
            amp = self.amp_val_ch2
            curve_color = 'xkcd:sky blue'
            title = 'CH2'

        t = np.arange(0.0, 1.0, 0.001)
        ch_signal = amp * np.ones(t.shape)

        self.ax[chn-1].clear()
        self.ax[chn-1].plot(t, ch_signal, color=curve_color)
        self.ax[chn-1].set_ylim([-amp-1, amp+1])

        self.ax[chn-1].set_title(title)
        self.ax[chn-1].set_facecolor('xkcd:black')
        self.ax[chn-1].grid(True)
        self.ax[chn-1].set_xlabel('Time/sec')
        self.ax[chn-1].set_ylabel('Voltage/V')
        self.fig.tight_layout()
        self.fig.canvas.draw()

        self.scpi_cmd_ch = {'dc': amp}

        if self.dev_available:
            self.sig_gen.para_set(self.scpi_cmd_ch, chn=chn)

    def noise_update(self, chn):

        self.load_entry()
        if chn == 1:
            amp, offset = self.amp_val_ch1, self.offset_val_ch1
            curve_color = 'xkcd:yellow green'
            title = 'CH1'
        elif chn == 2:
            amp, offset = self.amp_val_ch2, self.offset_val_ch2
            curve_color = 'xkcd:sky blue'
            title = 'CH2'

        t = np.arange(0.0, 1.0, 0.001)
        ch_signal = np.random.normal(loc=offset, scale=amp, size=t.shape)

        self.ax[chn-1].clear()
        self.ax[chn-1].plot(ch_signal, color=curve_color)
        self.ax[chn-1].axhline(y=offset, color='white', linestyle='dashed')

        self.ax[chn-1].set_title(title)
        self.ax[chn-1].set_facecolor('xkcd:black')
        self.ax[chn-1].grid(True)
        self.ax[chn-1].set_xlabel('Time/sec')
        self.ax[chn-1].set_ylabel('Voltage/V')
        self.fig.tight_layout()
        self.fig.canvas.draw()

        self.scpi_cmd_ch = {'noise': [amp, offset]}

        if self.dev_available:
            print(self.scpi_cmd_ch)
            self.sig_gen.para_set(self.scpi_cmd_ch, chn=chn)

    def window_geometry(self):
        for id_row, row in enumerate([1]*7):
            self.window.grid_rowconfigure(id_row, weight=row)
        for id_col, col in enumerate([5, 2, 2, 2]):
            self.window.grid_columnconfigure(id_col, weight=col)

        screen_width = self.window.winfo_screenwidth()
        screen_height = self. window.winfo_screenheight()
        self.window_width = screen_width * .3
        self.window_height = screen_height * .2
        self.window_start_x = 100
        self.window_start_y = 100
        window.geometry("%dx%d+%d+%d" % (self.window_width, self.window_height,
                                         self.window_start_x,
                                         self.window_start_y))
        self.window.configure(bg=self._from_rgb((255, 255, 255)))

    def _from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb

window = tk.Tk()
mywin = tACSWindow(window)
window.title('Hello Python')
mywin.wave_display()
window.mainloop()

