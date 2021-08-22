import os
import time
import numpy as np
# Read more on http://blog.philippklaus.de/2012/05/rigol-dg1022-arbitrary-waveform-function-generator/
# This file is similar to https://github.com/sbrinkmann/PyOscilloskop/blob/master/src/rigolScope.py

class BaseDriver:
    """Convert the python command into low level I/O command (SCPI)

    Parameters
    ----------

    Methods
    -------


    Attributes
    ----------


    Returns
    -------

    """

    def __init__(self, dev=None):
        if dev is None:
            dev_path_dict = {i: i_dev
                             for i, i_dev in enumerate(self.available_list())}
            print(dev_path_dict)
            id_dev = input('Available devices are listed above and input the \
                         corresponding key number to select the desired \
                         device. Should you have no idea about which one to \
                         choose, enter "?"\n')
            if id_dev == '?':
                id_dev = 0
                pass
            self.dev = dev_path_dict[int(id_dev)]
        else:
            assert os.path.exists(dev), 'Input device path does not exist, \
                please double-check your input of parameter dev'
            self.dev = dev

        self.device_open()
        self.info()

    def device_open(self):
        try:
            # use os.fstat to detect file is opened or not
            # Only read and write access are needed
            self.dev_fd = os.open(self.dev, os.O_RDWR)
            dup_check = [os.fstat(i) == os.fstat(self.dev_fd) for i in range(
                self.dev_fd)]
            if any(dup_check):
                print('The device is already opened, use the first opened fd')
                self.dev_fd = dup_check.index(True)
            print(f'Device is opened with file descriptor {self.dev_fd}')
        except OSError:
            self.port_access()  # what is the current access of that port and what is the current user
            print('run the script with sudo')

    def set_cmd(self, scpi_command):
        # Why use utf8? Find SCPI doc
        # Low level I/O, fd must be returned by os.open()
        assert type(scpi_command) == str, 'SCPI command MUST be written in string'
        os.write(self.dev_fd, scpi_command.encode(encoding='utf8'))

    def query_cmd(self, length=1000):
        return os.read(self.dev_fd, length)

    def port_access(self):
        access = os.stat(self.dev)
        print(oct(access.st_mode))

    def info(self):
        self.set_cmd("*IDN?")
        return self.query_cmd(1000)

    def available_list(self, port_root='/dev'):
        target_fd = os.popen(f'ls {port_root} |grep "usbtmc"')
        port_list = target_fd.readlines()
        # remove white space character '\n' and concatenate with port root
        port_list = [f'{port_root}/{x.strip()}' for x in port_list]
        return port_list

    def reset(self):
        self.set_cmd('*RST;*CLS;*OPC?')
        time.sleep(2.0)
        self.query_cmd()
        print("Reset is done!")

class SignalGenerator(BaseDriver):

    def __init__(self, out_chn=1, mode='sin', amp=0.5):
        super().__init__()
        self.out_chn = out_chn

    def chn_check(self, chn):
        if chn is None:
            chn = self.out_chn
        self.prefix = f':SOUR{chn}'
        return chn

    def status(self):
        for i in range(1, 3):
            self.set_cmd(':SOURce' + str(i) + ':APPLy?')
            time.sleep(0.2)
            print('CHN'+ str(i) +':')
            print(self.query_cmd())

    def para_set(self, para_dict, chn=None):
        self.chn_check(chn)
        special_dict = {'offset': 'VOLT:OFFS'}
        for key, val in para_dict.items():
            if key not in special_dict.keys():
                self.set_cmd(self.prefix + ':' + key[:4].upper() + ' ' + str(val).upper())
            else:
                self.set_cmd(self.prefix + ':' + special_dict[key] + ' ' + str(val).upper())

            time.sleep(0.05)

    def query(self, cmd):
        self.set_cmd(cmd)
        time.sleep(0.2)
        print(self.query_cmd())
        return self.query_cmd()

    def on(self, chn=None):
        self.chn_check(chn)
        self.set_cmd(f':OUTPut{chn} ON')

    def off(self, chn=None):
        self.chn_check(chn)
        self.set_cmd(':OUTPut' + str(chn) + ' OFF')

    def impedance_check(self, chn=None):
        self.chn_check(chn)
        self.set_cmd(':OUTPut' + str(chn) + ':IMPedance?')

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

    def arb_func(self, data, chn=None):
        chn = self.chn_check(chn)
        # self.clear_mem = '
        n_data = len(data)
        self.set_cmd(':SOUR' + str(chn) + ':APPL:ARB ' + str(self.sps))
        time.sleep(0.2)
        self.set_cmd(':SOUR' + str(chn) + ':DATA:POIN VOLATILE, ' + str(n_data))
        time.sleep(0.2)
        for ind in range(len(data)):
            data_str = ':DATA:VALue VOLATILE,' + str(ind+1) + ', ' + str(data[ind])
            self.set_cmd(data_str)
            time.sleep(0.2)

    def exec_func(self):
        self.query('*IDN?')
        if self.cmd == None:
            raise ValueError("No command to run!")
        else:
            if not isinstance(self.cmd, list):
                self.cmd = [self.cmd]
            for single_cmd in self.cmd:
                self.set_cmd(single_cmd)
            self.status()


    def amp(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.set_cmd(':SOUR' + str(chn) + ':VOLT?' )
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':VOLT ' + str(value))


    def offset(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.set_cmd(':SOUR' + str(chn) + ':VOLT:OFFS?' )
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':VOLT:OFFS ' + str(value))


    def frequency(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.set_cmd(':SOUR' + str(chn) + ':FREQ?' )
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':FREQ ' + str(value))

    def phase(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.set_cmd(':SOUR' + str(chn) + ':PHAS?' )
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':PHAS ' + str(value))

    def mode(self, value, query=None, chn=None):
        if chn == None:
            chn = self.out_chn
        if query is not None:
            self.set_cmd(':SOUR' + str(chn) + ':FUNC?' )
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':FUNC ' + value.upper())


class RigolFunctionGenerator(BaseDriver):
    """Class to control a Rigol DG1062Z waveform generator"""
    def __init__(self, device='/dev/usbtmc1'):
        if(device == None):
            listOfDevices = usbtmc.getDeviceList()
            if(len(listOfDevices) == 0):
                raise ValueError("There is no device to access")

            self.device = listOfDevices[0]
        else:
            self.device = device

        self.meas = usbtmc.UsbTmcDriver(self.device)

        self.name = self.meas.getName()
        print(self.name)

    def write(self, command, out=True):
        """Send an arbitrary command directly to the scope"""
        self.meas.write(command, out)

    def read(self):
        """Read an arbitrary amount of data directly from the scope"""
        return self.meas.read()

    def reset(self):
        """Reset the instrument"""
        self.meas.sendReset()



class RigolOscilloscope:
    """Class to control a Rigol DG1062Z waveform generator"""
    def __init__(self, device='/dev/usbtmc2'):
        if(device == None):
            listOfDevices = usbtmc.getDeviceList()
            if(len(listOfDevices) == 1):
                raise ValueError("There is no device to access")

            self.device = listOfDevices[0]
        else:
            self.device = device

        self.meas = usbtmc.UsbTmcDriver(self.device)

        self.name = self.meas.getName()
        print(self.name)

    def write(self, command, out=True):
        """Send an arbitrary command directly to the scope"""
        self.meas.write(command, out)

    def read(self):
        """Read an arbitrary amount of data directly from the scope"""
        return self.meas.read()

    def reset(self):
        """Reset the instrument"""
        self.meas.sendReset()
