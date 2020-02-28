

import time
import psychopy.voicekey as vk
vk.pyo_init(rate=44100, buffersize=32)

# What signaler class to use? Here just the demo signaler:
from psychopy.voicekey.demo_vks import DemoVoiceKeySignal as Signaler

# Create a voice-key to be used:
vpvk = vk.OnsetVoiceKey(
    sec=2.0, baseline=0.1,
    file_out='test.wav')

# Start it recording (and detecting):
vpvk.start()  # non-blocking; don't block when using Builder
time.sleep(2.4)
vpvk.stop()
# Create a voice-key to be used:
vpvk = vk.OnsetVoiceKey(
    file_in='chunk0.wav')
import pdb
pdb.set_trace()
# Start it recording (and detecting):
vpvk.start()  # non-blocking; don't block when using Builder
time.sleep(2.4)
vpvk.stop()
import pdb
pdb.set_trace
