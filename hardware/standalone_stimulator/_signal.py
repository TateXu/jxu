from jxu.hardware.waveform_generator import *
from jxu.hardware.waveform_generator import rigoldg1062z
import time
import inspect
from inspect import getcallargs
import numpy as np


def init_sin_func(source=None, function='sin', frequency=50, phase=0, volt=[1, 0]):
    if source == None:
        source = 1
    args = getcallargs(init_sin_func, source, function, frequency, phase, volt)
    chn_root = ':SOUR' + str(source)
    cmd = []
    for i, (label, val) in enumerate(args.items()):
        if label == 'source':
            pass
        elif label == 'volt' and isinstance(val, list):
            cmd.append(chn_root + ':VOLT' + ' ' + str(val[0]))
            cmd.append(chn_root + ':VOLT:OFFS' + ' ' + str(val[1]))
        else:
            cmd.append(chn_root + ':' + label[:4].upper() + ' ' + str(val).upper())
    return cmd


class SignalGenerator(rigoldg1062z.RigolFunctionGenerator):

    def __init__(self, out_chn=1, mode='sin', amp=0.5):
        super().__init__()
        self.out_chn = out_chn


    def status(self):
        for i in range(1, 3):
            self.write(':SOURce' + str(i) + ':APPLy?')
            time.sleep(0.2)
            print('CHN'+ str(i) +':')
            print(self.read())
    def para_set(self, parameter, value, chn=None):
        if parameter == None or value == None:
            raise ValueError("Required input: parameter and value!")
        if chn == None:
            chn = self.out_chn
        data_str = ':SOURce' + str(chn) + ':' + parameter.upper() + ' ' + str(value).upper()
        self.cmd = data_str
        print("CMD is set: " + data_str)

    def query(self, cmd):
        self.write(cmd)
        time.sleep(0.2)
        print(self.read())
        return self.read()

    def on(self, chn=None):
        if chn == None:
            chn = self.out_chn
        self.write(':OUTPut' + str(chn) + ' ON')

    def off(self, chn=None):
        if chn == None:
            chn = self.out_chn
        self.write(':OUTPut' + str(chn) + ' OFF')

    def impedance_check(self, chn=None):
        if chn == None:
            chn = self.out_chn
        self.write(':OUTPut' + str(chn) + ':IMPedance?')


    def sin_func(self, source=None, function='sin', frequency=50, phase=0, volt=[1, 0]):
        self.cmd = init_sin_func(source=source, function=function, frequency=frequency, phase=phase, volt=volt)

    def fade_in_out(self, duration=0.5, tone='flat', vol=1, sps=512, freq=4.0):
        if tone == 'flat':
            amp = vol
        elif tone == 'increase':
            amp = np.linspace(vol*0.3, vol, duration * sps)
        elif tone == 'decrease':
            amp = np.linspace(vol, vol*0.3, duration * sps)
        else:
            raise ValueError("Wrong input of tone!")
        self.sps = sps

        esm = np.arange(duration * sps)
        wf = np.sin(2 * np.pi * esm * freq / sps)
        wf_slice = wf * amp
        wf_int = np.int16((wf_slice+1)/2* 16383)

        return wf_int

    def reset(self):
        self.write('*RST;*CLS;*OPC?')
        time.sleep(2.0)
        self.read()
        print("Reset is done!")


    def arb_func(self, data, chn=None):
        if chn == None:
            chn = self.out_chn
        # self.clear_mem = '
        n_data = len(data)
        self.write(':SOUR' + str(chn) + ':APPL:ARB ' + str(self.sps))
        time.sleep(0.2)
        self.write(':SOUR' + str(chn) + ':DATA:POIN VOLATILE, ' + str(n_data))
        time.sleep(0.2)
        for ind in range(len(data)):
            data_str = ':DATA:VALue VOLATILE,' + str(ind+1) + ', ' + str(data[ind])
            self.write(data_str)
            time.sleep(0.2)
    def exec_func(self):
        self.query('*IDN?')
        if self.cmd == None:
            raise ValueError("No command to run!")
        else:
            if not isinstance(self.cmd, list):
                self.cmd = [self.cmd]
            for single_cmd in self.cmd:
                self.write(single_cmd)
            self.status()


    def amp(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.write(':SOUR' + str(chn) + ':VOLT?' )
            time.sleep(1.0)
            print(self.read())
        else:
            self.write(':SOUR' + str(chn) + ':VOLT ' + str(value))


    def offset(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.write(':SOUR' + str(chn) + ':VOLT:OFFS?' )
            time.sleep(1.0)
            print(self.read())
        else:
            self.write(':SOUR' + str(chn) + ':VOLT:OFFS ' + str(value))


    def frequency(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.write(':SOUR' + str(chn) + ':FREQ?' )
            time.sleep(1.0)
            print(self.read())
        else:
            self.write(':SOUR' + str(chn) + ':FREQ ' + str(value))

    def phase(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.write(':SOUR' + str(chn) + ':PHAS?' )
            time.sleep(1.0)
            print(self.read())
        else:
            self.write(':SOUR' + str(chn) + ':PHAS ' + str(value))

    def mode(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.write(':SOUR' + str(chn) + ':FUNC?' )
            time.sleep(1.0)
            print(self.read())
        else:
            self.write(':SOUR' + str(chn) + ':FUNC ' + value.upper())


    def offset(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.write(':SOUR' + str(chn) + ':VOLT:OFFS?' )
            time.sleep(1.0)
            print(self.read())
        else:
            self.write(':SOUR' + str(chn) + ':VOLT:OFFS ' + str(value))


class SignalMeasurement(rigoldg1062z.RigolOscilloscope):

    def __init__(self, out_chn=1, mode='sin', amp=0.5):
        super().__init__()
        self.out_chn = out_chn

    def query(self, cmd):
        self.write(cmd)
        time.sleep(0.2)
        print(self.read())
        return self.read()

    def reset(self):
        self.write('*RST;*CLS;*OPC?')
        time.sleep(2.0)
        self.read()
        print("Reset is done!")

    def exec_func(self):
        self.query('*IDN?')
        if self.cmd == None:
            raise ValueError("No command to run!")
        else:
            if not isinstance(self.cmd, list):
                self.cmd = [self.cmd]
            for single_cmd in self.cmd:
                self.write(single_cmd)
            self.status()

    def cur_vol(self, chn=None, freq_avg=False):
        if chn == None:
            chn = self.out_chn

        self.write(':MEASure:STATistic:ITEM? CURR,VMIN')
        vmin = self.read()

        self.write(':MEASure:STATistic:ITEM? CURR,VMAX')
        vmax = self.read()

        if freq_avg:
            self.write(':MEASure:STATistic:ITEM? AVER,FREQ')
        else:
            self.write(':MEASure:STATistic:ITEM? CURR,FREQ')

        freq = self.read()


        return np.float32(vmin), np.float32(vmax), np.float32(freq)

    def clear(self):
        self.write(':MEAS:CLE ALL')


        # print(cmd)
    # def output(self)

        # print(cmd)
    # def output(self)

