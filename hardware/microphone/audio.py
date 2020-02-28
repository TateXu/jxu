from pyo import *

inname, inind = pa_get_input_devices()
outname, outind = pa_get_output_devices()

if inind[inname.index('default')] == outind[outname.index('default')]:
    print(inind[inname.index('default')])
else:
    raise ValueError("Not default! Please manually check")
