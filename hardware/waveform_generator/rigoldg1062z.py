from jxu.hardware.waveform_generator import usbtmc

# Read more on http://blog.philippklaus.de/2012/05/rigol-dg1022-arbitrary-waveform-function-generator/
# 
# This file is similar to https://github.com/sbrinkmann/PyOscilloskop/blob/master/src/rigolScope.py

class RigolFunctionGenerator:
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
