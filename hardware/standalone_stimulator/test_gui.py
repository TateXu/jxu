#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-08-10 20:13:48
# Name       : test_gui.py
# Version    : V1.0
# Description: A minimal example for tkinter based gui
#========================================

import sys, os
import numpy as np
import platform
import time

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import tkinter as tk
import tkinter.font as tkFont
from tkinter import (Label, Entry, Button, Checkbutton, OptionMenu,
                     StringVar, messagebox, simpledialog, END)

from PIL import Image, ImageTk

from signal_generator import SignalGenerator as SG

class tACSWindow():
    def __init__(self, window, fontsize=20):

        self.fontStyle = tkFont.Font(family="Arial", size=fontsize) # "Lucida Grande"
        is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
        print(is_conda)
        if is_conda:
            messagebox.showwarning('Warning', 'The font of GUI cannot be rendered by Anaconda Python. For better visulization, please use other version of Python')
        self.window = window
        self.dev_available = False
        self.window_geometry()
        self.button_place()
        self.wave_display()
        self.window.mainloop()

    def button_place(self):
        root = '/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/hardware/standalone_stimulator/'
        self.fig_root = root
        subsps_coef = 2

        im_on = f'{self.fig_root}/Figures/button_on.png'
        im_off = f'{self.fig_root}/Figures/button_off.png'
        im_update = f'{self.fig_root}/Figures/button_update.png'
        self.window.fig_on = tk.PhotoImage(file=im_on).subsample(3)
        self.window.fig_off = tk.PhotoImage(file=im_off).subsample(3)
        self.window.fig_update = tk.PhotoImage(file=im_update).subsample(3)

        self.para_widgets_mat = np.asarray(
            [[{'VISA': self.dev_list}, {'USBTMC': self.dev_list}, None, 'CH1', 'CH2'],
             ['Device List', [''], 'Stimulation Type', ['tACS', 'tDCS', 'tRNS', 'Arb'], ['tACS', 'tDCS', 'tRNS', 'Arb']],
             [(self.window.fig_on, lambda: self.dev_connect()), None, 'Arbitrary Data', 0, 0],
             [None, None, 'Voltage', 1, 1],
             [None, None, 'Frequency', 10, 10],
             [None, None, 'Phase', 0, 0],
             [None, None, 'Offset', 0, 0],
             [None, None, 'Fade In/Out', 5, 5],
             [None, None, 'Stim. Duration', 5, 5],
             [(self.window.fig_update, self.para_update), None, 'Output', (self.window.fig_off, lambda: self.signal_out(chn=1), 'bt_out'), (self.window.fig_off, lambda: self.signal_out(chn=2), 'bt_out')],
             [(self.window.fig_update, self.refresh), None, 'Timer', '', '']],
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
        self.click_list, self.bt_out, self.check_list = [], [], []
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
                    tmp_obj = Label(self.window, font=self.fontStyle)
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
                                   ipadx=90, sticky='ew')
                tmp_stim_menu.configure(font=self.fontStyle)

                tmp_menu_opt = self.window.nametowidget(tmp_stim_menu.menuname)
                tmp_menu_opt.config(font=self.fontStyle)
                tmp_obj = [tmp_stim_menu, tmp_menu_opt]
                if 'tACS' in grid_val:
                    self.click_list.append(tmp_click)
                else:
                    self.dev_click = tmp_click
                del tmp_click
            elif grid_type == tuple:
                tmp_obj = Button(self.window, image=grid_val[0], bg="white", command=grid_val[1])
                tmp_obj.image = grid_val[0]
                tmp_obj.grid(row=row+row_start, column=col+col_start)
                if len(grid_val) == 3:
                    self.bt_out.append(tmp_obj)
            elif grid_type == dict:
                tmp_check = tk.IntVar()
                tmp_obj = Checkbutton(self.window, variable=tmp_check,
                                      text=list(grid_val.keys())[0], onvalue=1,
                                      offvalue=0, bg="white",
                                      command=list(grid_val.values())[0],
                                      font=self.fontStyle)
                tmp_obj.grid(row=row+row_start, column=col+col_start)
                self.check_list.append(tmp_check)
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

    def dev_list(self):
        visa_status = self.check_list[0].get()
        usbtmc_status = self.check_list[1].get()
        self.os_ver = platform.platform()
        if 'Windows' in self.os_ver and usbtmc_status:
            messagebox.showerror(title='Warning', message='USBTMC protocol' +
                                 ' is not available on Windows system. Use ' +
                                 'the default VISA protocol')
            # click yes in msg box, should toggle a button which disable usbtmc
            usbtmc_status = 0

        if visa_status and usbtmc_status:
            messagebox.showwarning(
                title='Warning', message='Please select only one protocol!')
            return None
        elif not visa_status and usbtmc_status:
            from signal_generator import USBTMC
            import os

            tmp = USBTMC(inst=False)
            for i, i_dev in enumerate(tmp.available_port_list()):
                try:
                    _ = os.open(i_dev, os.O_RDWR)
                except OSError:
                    pwd = simpledialog.askstring(title="Test", prompt='Enter your root pwd to change port access')
                    os.system(f'echo {pwd} | sudo -S chmod 666 {i_dev}')

            self.all_devices, self.dev_inst_list = USBTMC(inst=False).dev_list()
            self.protocol = 'USBTMC'
        elif visa_status and not usbtmc_status:
            from signal_generator import VISA
            self.all_devices, self.dev_inst_list = VISA(inst=False).dev_list()
            self.protocol = 'VISA'
        else:
            return None

        self.dev_click.set('')
        menu = self.para_widgets_mat_obj[1, 1][0]['menu']
        menu.delete(0, 'end')
        for choice in self.all_devices:
            menu.add_command(label=choice, command=tk._setit(self.dev_click, choice))

        self.refresh()


    def dev_connect(self):
        click_val = self.dev_click.get()
        ind_ = self.all_devices.index(click_val)

        dev = self.dev_inst_list[ind_]
        if dev is None:
            messagebox.showwarning('Warning', 'No available devices!')
        print(ind_)

        self.sig_gen = SG(dev=dev, protocol=self.protocol)
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
        plot_widget.grid(row=3, column=0, rowspan=6, columnspan=2, sticky='nsew')

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
            self.sig_gen.para_set(self.scpi_cmd_ch, chn=chn)

    def arb_update(self, chn):
        # read file
        import pickle
        self.load_entry()
        arb_data_path = self.entry_data[0, chn-1]
        with open(arb_data_path, 'rb') as f:
            arb_data = pickle.load(f)

        len_data = len(arb_data['data'])
        t = np.linspace(0.0, len_data/arb_data['sps'], len_data)
        self.ax_plt(t, arb_data['data'], chn)

        if self.dev_available:
            self.sig_gen.sps = arb_data['sps']
            self.sig_gen.arb_func(data=arb_data['data'], chn=chn)


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
                    self.sig_gen.amp(value=val, chn=chn)
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
                if self.dev_available:
                    state_switch(button_out, channel=chn)
                for stim_val in step_list:
                    time.sleep(sleep_dur)
                    amp_adjust(val=stim_val, chn=chn)
            elif status == 'finish':
                for stim_val in step_list[::-1]:
                    amp_adjust(val=stim_val, chn=chn)
                    # self.sig_gen.amp(stim_val, chn=chn)
                    time.sleep(sleep_dur)

                print('off output')
                if self.dev_available:
                    state_switch(button_out, channel=chn)


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
            print(float(fade_dur))
            fade(amp=amp, chn=chn, status='start', fade_dur=float(fade_dur))
            stim_timer(chn=chn, duration=float(stim_dur))
            fade(amp=amp, chn=chn, status='finish', fade_dur=float(fade_dur))
        else:
            state_switch(button_out, channel=chn)
            stim_timer(chn=chn, duration=float(stim_dur))
            state_switch(button_out, channel=chn)

    def refresh(self):
        self.para_widgets_mat_obj[1, 1][0].config(width=2)
        self.window_ratio()
        self.window.update()

    def window_ratio(self):
        for id_row, row in enumerate([1]*7):
            self.window.grid_rowconfigure(id_row, weight=row)
        for id_col, col in enumerate([2, 2, 2, 2, 2]):
            self.window.grid_columnconfigure(id_col, weight=col)

    def window_geometry(self):
        self.window_ratio()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self. window.winfo_screenheight()

        ratio = (screen_width*9) / (screen_height*16)
        self.window_width = screen_width * .5 / ratio
        self.window_height = screen_height * .5
        self.window_start_x = screen_width * .2
        self.window_start_y = screen_height * .2
        window.geometry("%dx%d+%d+%d" % (self.window_width, self.window_height,
                                         self.window_start_x,
                                         self.window_start_y))
        self.window.configure(bg=self._from_rgb((255, 255, 255)))

    def _from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb

window = tk.Tk()
mywin = tACSWindow(window)


