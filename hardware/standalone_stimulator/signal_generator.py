import os
import time
import numpy as np
# Read more on http://blog.philippklaus.de/2012/05/rigol-dg1022-arbitrary-waveform-function-generator/
# This file is similar to https://github.com/sbrinkmann/PyOscilloskop/blob/master/src/rigolScope.py

class BaseDriver:
    """Convert the python command into low level I/O command (SCPI)
    On Linux system, the usb device can be accessed as a device node

    Parameters
    ----------
    dev : str | None (default None)
        The location of usbtmc device

    Attributes
    ----------
    dev: str | None (default None)
        The location of usbtmc device
    dev_fd: int
        File descriptor of the device

    Returns
    -------

    """

    def __init__(self, dev=None):
        self.dev_init(dev)
        self.dev_fd = self.device_open()
        self.info(dev_fd=self.dev_fd)

    def dev_init(self, dev):
        """ Initialize the devices based on given location

        Parameters
        ----------
        dev : str | None (default None)
            If None, then list all available USBTMC devices and select one
            If a string is given, then check its correctness

        Returns
        -------

        """

        if dev is None:
            # List all avaiable devices
            dev_path_dict = {i: i_dev
                             for i, i_dev in enumerate(self.available_list())}
            print(dev_path_dict)

            # Retrieve the devices info
            for tmp_dev_id, tmp_dev_loc in dev_path_dict.items():
                tmp_dev_fd = self.device_open(tmp_dev_loc)
                tmp_msg = self.info(dev_fd=tmp_dev_fd)
                print(f'Id: {tmp_dev_id}, Device info: {tmp_msg}')

            dev_id = input('Available devices are listed above and input ' +
                           'the corresponding id number to select the ' +
                           'desired device.\n')
            self.dev = dev_path_dict[int(dev_id)]
        else:
            assert os.path.exists(dev), 'Input device path does not exist, \
                please double-check your input of parameter dev'
            self.dev = dev

    def device_open(self, dev=None):
        """ Open the device node

        Parameters
        ----------
        dev : str | None (default None)
            If None, then list all available USBTMC devices and select one
            If a string is given, then check its correctness

        Attributes
        ----------
        dev: str | None (default None)
            The location of usbtmc device
        dev_fd: int
            File descriptor of the device

        Returns
        -------

        """
        if dev is None:
            dev = self.dev
        try:
            # Only read and write access are needed to open the device
            dev_fd = os.open(dev, os.O_RDWR)
        except OSError:
            print('run the script with sudo')
            # Check the current access of dev port and what is the current user
            self.port_access(dev)
            pwd = input('Please input the password for root accesss:\n')
            # For convenience, give all users with read and write permission,
            # the minimum permission should be 006
            os.system(f'echo {pwd} | sudo -S chmod 666 {dev}')
            dev_fd = os.open(dev, os.O_RDWR)
        finally:
            # Use os.fstat to detect file is opened or not
            dup_check = [os.fstat(i) == os.fstat(dev_fd) for i in range(
                dev_fd)]
            if any(dup_check):
                print('The device is already opened, use the first opened fd')
                dev_fd = dup_check.index(True)
            print(f'Device is opened with file descriptor {dev_fd}')

            return dev_fd

    def set_cmd(self, scpi_command, dev_fd=None):
        # Low level I/O to send data stream to device
        if dev_fd is None:
            dev_fd = self.dev_fd
        assert type(scpi_command) == str, 'SCPI command MUST be written in string'
        os.write(dev_fd, scpi_command.encode(encoding='utf8'))

    def query_cmd(self, length=1000, dev_fd=None):
        # Low level I/O to receive data stream from to device
        if dev_fd is None:
            dev_fd = self.dev_fd
        return os.read(dev_fd, length)

    def port_access(self, dev=None):
        # Check current access of given port/dev/address
        if dev is None:
            self.dev = dev
        access = os.stat(dev)
        print(f'\nThe current access permission is: {oct(access.st_mode)}')

    def info(self, dev_fd=None):
        # Retrieve the device information
        self.set_cmd("*IDN?", dev_fd=dev_fd)
        return self.query_cmd(1000, dev_fd=dev_fd)

    def available_list(self, port_root='/dev'):
        # List all available USBTMC devices that can be found under given root
        target_fd = os.popen(f'ls {port_root} |grep "usbtmc"')
        port_list = target_fd.readlines()
        # remove white space character '\n' and concatenate with port root
        port_list = [f'{port_root}/{x.strip()}' for x in port_list]
        return port_list

    def reset(self):
        # Reset the device
        self.set_cmd('*RST;*CLS;*OPC?')
        time.sleep(2.0)
        self.query_cmd()
        print("Reset is done!")


class SignalGenerator(BaseDriver):
    """Convert the python command into low level I/O command (SCPI)
    On Linux system, the usb device can be accessed as a device node

    Parameters
    ----------
    dev : str | None (default None)
        The location of usbtmc device
    out_chn : 1 | 2 (default 1)
        Output channel to configure
    mode : 'sin' | 'sweep?'
        !!! should be checked one by one and filling in
    amp : float (default 0.5)
        Amplitude of signal in the unit of Volt

    Attributes
    ----------
    dev: str | None (default None)
        The location of usbtmc device
    dev_fd: int
        File descriptor of the device

    Returns
    -------

    """


    def __init__(self, dev='/dev/usbtmc1', out_chn=1, mode='sin', amp=0.5):
        super().__init__(dev=dev)
        self.out_chn = out_chn

    def chn_check(self, chn):
        # To ensure the output channel is not None
        if chn is None:
            chn = self.out_chn
        self.prefix = f':SOUR{chn}'
        return chn

    def status(self):
        # Get current configurations of both channels
        # For sine mode, the parameters are in the order of frequency,
        # amplitude, offset and phase
        for i in [1, 2]:
            self.set_cmd(':SOURce' + str(i) + ':APPLy?')
            time.sleep(0.2)
            print(f'CHN{str(i)}:')
            print(self.query_cmd())

    def para_set(self, para_dict, chn=None):
        # To conveniently configure all parameters in one python command
        chn = self.chn_check(chn)
        special_dict = {'offset': 'VOLT:OFFS',
                        'sin': ['APPL:SIN', '', '', '', ''],
                        'dc': 'APPL:DC 1,1,',
                        'noise': ['APPL:NOIS', '']}
        for key, val in para_dict.items():
            if key not in special_dict.keys():
                self.set_cmd(self.prefix + ':' + key[:4].upper() + ' ' + str(val).upper())
            else:
                if type(special_dict[key]) == list:
                    suffix = ''
                    for i, j in zip(special_dict[key], val):
                        suffix += f'{i} {str(j).upper()},'
                    self.set_cmd(self.prefix + ':' + suffix[:-1])
                    print(self.prefix + ':' + suffix[:-1])
                else:
                    self.set_cmd(self.prefix + ':' + special_dict[key] + ' ' + str(val).upper())
            time.sleep(0.05)

    def query(self, cmd):
        self.set_cmd(cmd)
        time.sleep(0.2)
        print(self.query_cmd())
        return self.query_cmd()

    def on(self, chn=None):
        # Turn on the output channel
        chn = self.chn_check(chn)
        self.set_cmd(f':OUTPut{chn} ON')

    def off(self, chn=None):
        # Turn off the output channel
        chn = self.chn_check(chn)
        self.set_cmd(':OUTPut' + str(chn) + ' OFF')

    def amp(self, value=None, get_cfg=False, chn=None):
        """ Configure the amplitude or get the current configuration

        Parameters
        ----------
        value : float | None (default None)
            Value of amplitude to set
        get_cfg : bool (default False)
            If True, get the current amplitude. If False, set the amplitude
        chn : 1 | 2 (default 1)
            Output channel to configure

        """

        assert value is not None or get_cfg, 'When get_cfg is False, valid \
            value must be given'
        chn = self.chn_check(chn)
        if get_cfg:
            self.set_cmd(':SOUR' + str(chn) + ':VOLT?')
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':VOLT ' + str(value))


    def offset(self, value, query=None, chn=None):

        chn = self.chn_check(chn)
        if query is not None:
            self.set_cmd(':SOUR' + str(chn) + ':VOLT:OFFS?' )
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':VOLT:OFFS ' + str(value))


    def frequency(self, value, query=None, chn=None):

        chn = self.chn_check(chn)
        if query is not None:
            self.set_cmd(':SOUR' + str(chn) + ':FREQ?' )
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':FREQ ' + str(value))

    def phase(self, value, query=None, chn=None):

        chn = self.chn_check(chn)
        if query is not None:
            self.set_cmd(':SOUR' + str(chn) + ':PHAS?' )
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':PHAS ' + str(value))

    def mode(self, value, query=None, chn=None):

        chn = self.chn_check(chn)
        if query is not None:
            self.set_cmd(':SOUR' + str(chn) + ':FUNC?' )
            time.sleep(1.0)
            print(self.query_cmd())
        else:
            self.set_cmd(':SOUR' + str(chn) + ':FUNC ' + value.upper())


    def fade_in_out_arb(self, duration=0.5, tendency='flat', sps=512, vol=1,
                        freq=4.0):
        """ An examplar function to realize the fade in/out of signal via
        generating arbitrary data. This function should be used combining with
        the arb_func() function.

        Parameters
        ----------
        duration : float (default 0.5)
            Duration of the fade in/out signal
        tendency : 'flat' | 'increase' | 'decrease' (default 'flat')
            'flat' indicates without fade in/out. 'increase' indicates fade in,
            i.e., the signal amplitude increases. 'decrease' indicates fade out,
            i.e., the signal amplitude decreases.
        sps : int (default 512)
            Samples per second, which defines the temporal resolution of signal
        vol : int or float (default 1)
            Amplitude/Volume of the sinusoid signal in the unit of Volt
        freq : int or float (default 4.0)
            Frequency of the sinusoid signal in the unit of Hz

        Returns
        -------
        wf_int :
            Fade in/out data stream in the type of int16

        """

        if tendency == 'flat':
            amp = vol
        elif tendency == 'increase':
            amp = np.linspace(vol*0.3, vol, duration * sps)
        elif tendency == 'decrease':
            amp = np.linspace(vol, vol*0.3, duration * sps)
        else:
            raise ValueError("Wrong input of tone!")
        self.sps = sps

        esm = np.arange(duration * sps)
        wf = np.sin(2 * np.pi * esm * freq / sps)
        wf_slice = wf * amp
        wf_int = np.int16((wf_slice+1)/2 * 16383)

        return wf_int

    def arb_func(self, data, chn=None):
        """ An examplar function to realize the fade in/out of signal via
        generating arbitrary data. This function should be used combining with
        the arb_func() function.

        Parameters
        ----------
        duration : float (default 0.5)
            Duration of the fade in/out signal
        tendency : 'flat' | 'increase' | 'decrease' (default 'flat')
            'flat' indicates without fade in/out. 'increase' indicates fade in,
            i.e., the signal amplitude increases. 'decrease' indicates fade out,
            i.e., the signal amplitude decreases.
        sps : int (default 512)
            Samples per second, which defines the temporal resolution of signal
        vol : int or float (default 1)
            Amplitude/Volume of the sinusoid signal in the unit of Volt
        freq : int or float (default 4.0)
            Frequency of the sinusoid signal in the unit of Hz

        Returns
        -------
        wf_int :
            Fade in/out data stream in the type of int16

        """
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


    def impedance_check(self, chn=None):
        self.chn_check(chn)
        self.set_cmd(':OUTPut' + str(chn) + ':IMPedance?')

    def sin_func(self, source=None, function='sin', frequency=50, phase=0, volt=[1, 0]):
        self.cmd = init_sin_func(source=source, function=function, frequency=frequency, phase=phase, volt=volt)


