from jxu.data.loader import *
from jxu.audio.audiosignal import *
from auditok import AudioRegion

all_cnt = []
for nr_trial in range(160):
    # name = audio_denoise(audio_loader(trial=nr_trial))
    
    wav_std(audio_loader(trial=nr_trial, std=False), sps=24000)
    region = AudioRegion.load(audio_loader(trial=nr_trial))
    # regions = region.split_and_plot() # or just region.splitp()
    aaa = region.split()
    all_cnt.append(len(list(aaa)))
    del aaa
    # for segment in aaa:
    #     au_seg_s = segment.meta.start
    #     au_seg_e = segment.meta.end
    #     cnt += 1

import pdb
pdb.set_trace()