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
        self.dev_available = False
        try:
            self.dev_connect()
        except:
            print('No connected device')
            pass

        self.window.mainloop()

    def button_place(self):
        root = '.'  # '/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/hardware/standalone_stimulator/'
        self.fig_root = root
        subsps_coef = 2

        im_on = f'{self.fig_root}/Figures/button_on.png'
        im_off = f'{self.fig_root}/Figures/button_off.png'
        im_update = "./Figures/button_update.png"
        self.window.fig_on = tk.PhotoImage(file=im_on).subsample(3)
        self.window.fig_off = tk.PhotoImage(file=im_off).subsample(3)
        self.window.fig_update = tk.PhotoImage(file=im_update).subsample(3)

        self.para_widgets_mat = np.asarray(
            [['Device List', [''], None, 'CH1', 'CH2'],
             [(self.window.fig_on, print('111')), None, 'Stimulation Type', ['tACS', 'tDCS', 'tRNS', 'Arb'], ['tACS', 'tDCS', 'tRNS', 'Arb']],
             [None, None, 'Arbitrary Data', 0, 0],
             [None, None, 'Voltage', 1, 1],
             [None, None, 'Frequency', 10, 10],
             [None, None, 'Phase', 0, 0],
             [None, None, 'Offset', 0, 0],
             [None, None, 'Fade In/Out', 5, 5],
             [None, None, 'Stim. Duration', 5, 5],
             [(self.window.fig_update, self.para_update), None, 'Output', (self.window.fig_off, lambda: self.signal_out(chn=1), 'bt_out'), (self.window.fig_off, lambda: self.signal_out(chn=2), 'bt_out')],
             [None, None, 'Timer', '', '']],
            dtype='object')
        self.para_label = {'Device List': 'label_ch1.png',
                           'CH1': 'label_ch1.png',
                           'CH2': 'label_ch2.png',
                           'Arbitrary Data': 'label_stim_type.png',
                           'Stimulation Type': 'label_stim_type.png',
                           'Voltage': 'label_amp.png',
                           'Frequency': 'label_freq.png',
                           'Phase': 'label_phase.png',
                           'Offset': 'label_offset.png',
                           'Fade In/Out': 'label_fade_dur.png',
                           'Stim. Duration': 'label_stim_dur.png',
                           'Output': 'label_output.png',
                           'Timer': 'label_output.png',
                          }
        self.para_widgets_mat_obj = np.empty(self.para_widgets_mat.shape,
                                             dtype='object')

        self.entry_row_start, self.entry_col_start = 2, 3
        row_start, col_start = 0, 0
        self.click_list, self.bt_out = [], []
        for (row, col), grid_val in np.ndenumerate(self.para_widgets_mat):
            if grid_val is None:
                # None value means no button neither entry
                continue
            grid_type = type(grid_val)
            if grid_type == int or grid_type == float:
                tmp_obj = Entry(self.window, font=self.fontStyle)
                tmp_obj.insert(0, str(grid_val))
                tmp_obj.grid(row=row+row_start, column=col+col_start)
            elif grid_type == str:
                if grid_val == '':
                    tmp_obj = Label(self.window)
                else:
                    im_obj = tk.PhotoImage(file=f'{root}/Figures/{self.para_label[grid_val]}').subsample(subsps_coef)
                    tmp_obj = Label(self.window, image=im_obj, font=self.fontStyle, bg="white")
                    tmp_obj.image = im_obj
                tmp_obj.grid(row=row+row_start, column=col+col_start)
            elif grid_type == list:
                tmp_click = StringVar()
                tmp_click.set(grid_val[0])
                tmp_stim_menu = OptionMenu(self.window, tmp_click, *grid_val,
                                           command=self.entry_state_update)
                tmp_stim_menu.grid(row=row+row_start, column=col+col_start,
                                   ipadx=90)
                tmp_stim_menu.configure(font=self.fontStyle)

                tmp_menu_opt = self.window.nametowidget(tmp_stim_menu.menuname)
                tmp_menu_opt.config(font=self.fontStyle)
                tmp_obj = [tmp_stim_menu, tmp_menu_opt]
                if 'tACS' in grid_val:
                    self.click_list.append(tmp_click)
                del tmp_click
            elif grid_type == tuple:
                tmp_obj = Button(self.window, image=grid_val[0], bg="white", command=grid_val[1])
                tmp_obj.image = grid_val[0]
                tmp_obj.grid(row=row+row_start, column=col+col_start)
                if len(grid_val) == 3:
                    self.bt_out.append(tmp_obj)
            else:
                import pdb;pdb.set_trace()
                raise ValueError('Unsupported Variable Value')

            self.para_widgets_mat_obj[row, col] = tmp_obj
            del tmp_obj

        # self.bt_update = tk.Button(self.window, image=self.window.fig_update, command=self.para_update)
        # self.bt_update.image = self.window.fig_update
        # self.bt_update.grid(row=9, column=0)

    def entry_state_update(self, event):
        for id_click, click in enumerate(self.click_list):
            click_val = click.get()

            if click_val == 'tACS':
                state_list = ['disable', 'normal', 'normal', 'normal',
                              'normal', 'normal', 'normal', 'normal']
            elif click_val == 'tDCS':
                state_list = ['disable', 'normal', 'disable', 'disable',
                              'disable', 'normal', 'normal', 'normal']
            elif click_val == 'tRNS':
                state_list = ['disable', 'normal', 'disable', 'disable',
                              'normal', 'normal', 'normal', 'normal']
            elif click_val == 'Arb':
                state_list = ['normal', 'disable', 'disable', 'disable',
                              'disable', 'disable', 'disable', 'normal']
            for id_entry, entry_state in enumerate(state_list):
                self.para_widgets_mat_obj[id_entry+self.entry_row_start, id_click+self.entry_col_start].config(state=entry_state)


    def dev_connect(self):
        self.sig_gen = SG(dev='/dev/usbtmc1')
        self.dev_available = True

    def wave_display(self):

        # self.fig, self.ax = plt.subplots(2, 1, facecolor=(1, 1, 1),
        #                                  figsize=(self.window_width*0.4/100, self.window_height*0.8/100))
        self.fig, self.ax = plt.subplots(2, 1, facecolor=(1, 1, 1))
        self.curve_color_list = ['xkcd:yellow green', 'xkcd:sky blue']
        self.title_list = ['CH1', 'CH2']

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
        plot_widget.grid(row=2, column=0, rowspan=7, columnspan=2, sticky='nsew')

        # self.window.mainloop()


    def ax_plt(self, xs, ys, chn):

        self.ax[chn-1].clear()
        self.ax[chn-1].plot(xs, ys, color=self.curve_color_list[chn-1])
        self.ax[chn-1].set_title(self.title_list[chn-1])
        self.ax[chn-1].set_facecolor('xkcd:black')
        self.ax[chn-1].grid(True)
        self.ax[chn-1].set_xlabel('Time/sec')
        self.ax[chn-1].set_ylabel('Voltage/V')
        self.fig.tight_layout()
        self.fig.canvas.draw()



    def para_update(self):
        for id_click, click in enumerate(self.click_list):
            click_val = click.get()
            if click_val == 'tACS':
                self.sin_update(chn=id_click+1)
            elif click_val == 'tDCS':
                self.dc_update(chn=id_click+1)
            elif click_val == 'tRNS':
                self.noise_update(chn=id_click+1)
            elif click_val == 'Arb':
                self.arb_update(chn=id_click+1)

    def load_entry(self):
        def customize_float(string_val):
            try:
                return float(string_val)
            except ValueError:
                return string_val

        # arb, amp, freq, phase, offset, fade, stim dur
        self.entry_data = np.empty((7, 2), dtype='object')
        for (i, j), _ in np.ndenumerate(self.entry_data):
            self.entry_data[i, j] = customize_float(
                self.para_widgets_mat_obj[i+self.entry_row_start, j+self.entry_col_start].get())


    def sin_update(self, chn):

        self.load_entry()
        amp, freq, phase, offset = self.entry_data[1:5, chn-1]

        t = np.arange(0.0, 1.0, 0.001)
        ch_signal = offset + amp * np.sin(2*np.pi*freq*t + phase / 180 * np.pi)
        self.ax_plt(t, ch_signal, chn)

        self.scpi_cmd_ch = {'sin': [freq, amp, offset, phase]}
        if self.dev_available:
            self.sig_gen.para_set(self.scpi_cmd_ch, chn=chn)


    def dc_update(self, chn):

        self.load_entry()
        amp = self.entry_data[1, chn-1]

        t = np.arange(0.0, 1.0, 0.001)
        ch_signal = amp * np.ones(t.shape)

        self.ax_plt(t, ch_signal, chn)
        self.scpi_cmd_ch = {'dc': amp}

        if self.dev_available:
            self.sig_gen.para_set(self.scpi_cmd_ch, chn=chn)


    def noise_update(self, chn):

        self.load_entry()
        amp, offset = self.entry_data[[1, 4], chn-1]

        t = np.arange(0.0, 1.0, 0.001)
        ch_signal = np.random.normal(loc=offset, scale=amp, size=t.shape)

        self.ax_plt(t, ch_signal, chn)
        self.scpi_cmd_ch = {'noise': [amp, offset]}

        if self.dev_available:
            print(self.scpi_cmd_ch)
            self.sig_gen.para_set(self.scpi_cmd_ch, chn=chn)

    def arb_update(self, chn):
        pass

    def signal_out(self, chn):
        self.load_entry()
        self.para_update()

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
            click_list = self.click_list.copy()
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
                state_switch(button_out, channel=chn)
                if self.dev_available:
                    self.sig_gen.on(chn=chn)
                for stim_val in step_list:
                    amp_adjust(val=stim_val, chn=chn)
                    time.sleep(sleep_dur)
            elif status == 'finish':
                for stim_val in step_list[::-1]:
                    amp_adjust(val=stim_val, chn=chn)
                    # self.sig_gen.amp(stim_val, chn=chn)
                    time.sleep(sleep_dur)

                state_switch(button_out, channel=chn)
                if self.dev_available:
                    self.sig_gen.off(chn=chn)


        def stim_timer(chn, duration):
            print('Timer start')
            if duration > 0:
                duration -= 1
                self.para_widgets_mat_obj[10, chn+self.entry_col_start-1]['text'] = duration
                self.window.after(1000, lambda: stim_timer(chn, duration))
                print(duration)
            else:
                self.window.after(1000, state_switch(self.bt_out[chn-1], channel=chn))


        amp, fade_dur, stim_dur = self.entry_data[[1, -2, -1], chn-1]
        button_out = self.bt_out[chn-1]

        if stim_dur == '' and fade_dur == '':
            state_switch(button_out, channel=chn)
        elif fade_dur != '':
            assert stim_dur != '', 'When fade is non-empty, duration also must be non-empty'
            fade(amp=amp, chn=chn, status='start', fade_dur=float(fade_dur))
            stim_timer(chn=chn, duration=float(stim_dur))
            state_switch(button_out, channel=chn)
            fade(amp=amp, chn=chn, status='finish', fade_dur=float(fade_dur))
            state_switch(button_out, channel=chn)
        else:
            print('-----')
            print(button_out['state'])
            state_switch(button_out, channel=chn)
            print('-----')
            print(button_out['state'])
            stim_timer(chn=chn, duration=float(stim_dur))
            print('Finish stim')
            print('-----')
            print(button_out['state'])
            state_switch(button_out, channel=chn)
            print('-----')
            print(button_out['state'])

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
#         window.geometry("%dx%d+%d+%d" % (self.window_width, self.window_height,
                                         # self.window_start_x,
                                         # self.window_start_y))
        self.window.configure(bg=self._from_rgb((255, 255, 255)))

    def _from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb

window = tk.Tk()
mywin = tACSWindow(window)
window.title('Hello Python')
mywin.wave_display()
window.mainloop()

