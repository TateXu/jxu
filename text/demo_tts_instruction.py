


tts = False

if tts:
    from jxu.basiccmd.tts import *
else:
    from jxu.audio.audiosignal import *
loc='/home/jxu/File/Data/NIBS/Stage_one/Audio/Soundeffect/artifact/'

file_name_list = ['eye_horizontal.mp3', 
             'eye_vertical.mp3', 
             'eye_circular.mp3', 
             'eye_blink.mp3', 
             'frown.mp3', 
             'head_horizontal.mp3', 
             'head_vertical.mp3', 
             'arm.mp3', 
             'leg.mp3', 
             'fist.mp3', 
             'wrist.mp3', 
             'upper_body.mp3', 
             'teeth.mp3', 
             'no_respiration.mp3',
             'respiration.mp3']
ssml_list = ['<speak>Please move both eyeballs horizontally. </speak>',
    '<speak>Please move both eyeballs vertically. </speak>',
    '<speak>Please move both eyeballs circularly. </speak>',
    '<speak>Please blink eyes. </speak>',
    '<speak>Please frown and release it repeatedly. </speak>',
    '<speak>Please rotate head horizontally. </speak>',
    '<speak>Please rotate head vertically. </speak>',
    '<speak>Please lift and lower both arms repeatedly. </speak>',
    '<speak>Please lift and lower both legs repeatedly. </speak>',
    '<speak>Please make a fist and let it go repeatedly. </speak>',
    '<speak>Please rotate both wrists repeatedly. </speak>',
    '<speak>Please move your whole upper body. </speak>',
    '<speak>Please clench your teech and let go repeatedly. </speak>',
    '<speak>Please stay still and hold you breath. </speak>',
    '<speak>Please breathe in and out deeply. </speak>']

for ssml, file_name in zip(ssml_list, file_name_list):
    if tts:
        google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
    else:
        mp3_to_wav(loc+file_name[:-4])



"""


# General:


file_name = 'Instruction.mp3'
ssml="<speak>Thank you for participating in our experiment about personalized non-invasive brain stimulation for second language learning. You will be asked to perform the following three tasks. The first is to read out the displayed German sentence while we are recording. The second task is to keep seated and relaxed while either opening or closing your eyes. In the third task, you will hear a German sentence with a missing word, and you will be expected to fill in the blank by speaking out your answer. Now, we will start a practice block to make you more familiar with these tasks. Please feel free to ask the experimenter any questions you might have during the practice block.</speak>"
if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


file_name = 'Instruction_main.mp3'

ssml="<speak>Thank you for participating in our experiment about personalized non-invasive brain stimulation for second language learning. Now, the practice session is finished and please press space key to start the experiment. </speak>"
if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


file_name = 'prac_finish.mp3'
ssml= "<speak>The practice block is finished. Please inform the experimenter, and we will start the experiment..</speak>"

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])

file_name = 'block_start.mp3'
ssml= "<speak>Now, if you are ready to start the next block, please press space key to continue.</speak>"

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])

file_name = 'block_intro_start.mp3'
ssml= "<speak>Now, if you are ready to start thís block, please press space key to continue.</speak>"

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


file_name = 'block_finish.mp3'
ssml= "<speak>This block is finished. Please have some rest and ask the experimenter if you would like a drink or snack. When you are ready to continue the experiment, press the space key.</speak>"
if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


file_name = 'exp_finish.mp3'
ssml= "<speak>This experiment is finished. Please follow the instructions of the experimenter. Thanks for your participation.</speak>"

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])



file_name = 'press_cont.mp3'
ssml= "<speak>Now, please press space key to continue.</speak>"

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])



# Calibration:

file_name = 'cali_intro_1.mp3'
ssml= '<speak>In this block, each trial will start with a melodious beep sound, which is like this.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


file_name = 'cali_intro_2.mp3'
ssml= '<speak>Within each trial, please read out the displayed German sentence after hearing a beep sound with increasing pitch, which is like this.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


file_name = 'cali_intro_3.mp3'
ssml= '<speak>This trial will finish after the beep sound with decreasing pitch, which is like this. </speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])

infiles=['cali_intro_1.wav', 'reminder.wav', 'cali_intro_2.wav',  'C3A_C4A_tone_decrease_1s_new.wav', 'cali_intro_3.wav', 'C4A_C3A_tone_decrease_1s_new.wav','block_intro_start.wav']
outfile='calibration.wav'
wav_concat(infiles, outfile)

# RS:
file_name = 'rs_intro_1.mp3'
ssml='<speak>In this block, please keep seated and relaxed.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])



file_name = 'rs_intro_2.mp3'

ssml= '<speak>You will be asked to either open or close eyes.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


file_name = 'rs_intro_3.mp3'
ssml= '<speak>A beep sound with decreasing pitch will indicate this block is finished, which is like this.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


infiles=['rs_intro_1.wav', 'rs_intro_2.wav', 'rs_intro_3.wav', 'C4A_C3A_tone_decrease_1s_new.wav', 'rs_close.wav']  # rs_close  rs_open

outfile='rs_intro_close.wav'
wav_concat(infiles, outfile)




file_name = 'rs_open.mp3'
ssml= '<speak>Now, please open your eyes and keep relaxed. And blinking eyes is allowed.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])



file_name = 'rs_close.mp3'
ssml= '<speak>Now, please close your eyes and keep relaxed.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])




# Q_A:
file_name = 'qa_intro_1.mp3'
ssml= '<speak>In this block, each trial will start with a melodious beep sound, which is like this.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])



file_name = 'qa_intro_2.mp3'
ssml= '<speak>Within each trial, you will first hear a German sentence with a missing word censored by silence, which sounds like a blank.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


file_name = 'qa_intro_3.mp3'
ssml= '<speak>Subsequently, after hearing a beep sound with increasing pitch, you should fill in the blank with speaking out your answer. The beep sound sounds like this.</speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])




file_name = 'qa_intro_4.mp3'
ssml= '<speak>A beep sound with decreasing pitch will indicate the finish of this trial, which is like this. </speak>'

if tts:
    google_text_to_speech(ssml_string=ssml, audio_location=loc+file_name, speed=0.9, pitch=0.0, lang='en-US')
else:
    mp3_to_wav(loc+file_name[:-4])


infiles=['qa_intro_1.wav', 'reminder.wav', 'qa_intro_2.wav',  '40Hz_new.wav', 'qa_intro_3.wav',  'C3A_C4A_tone_decrease_1s_new.wav', 'qa_intro_4.wav', 'C4A_C3A_tone_decrease_1s_new.wav','block_intro_start.wav']
outfile='q_a_update_assr.wav'
wav_concat(infiles, outfile)

if not tts:
    file_list = ['qa_intro_1.wav', 'reminder.wav', 'qa_intro_2.wav',  '40Hz_new.wav', 'qa_intro_3.wav',  'C3A_C4A_tone_decrease_1s_new.wav', 'qa_intro_4.wav', 'C4A_C3A_tone_decrease_1s_new.wav','block_intro_start.wav']
    import pdb 
    
    pdb.set_trace()
    infiles = [loc + i for i in file_list]
    outfile = loc + 'q_a_update_assr.wav'
    wav_concat(infiles, outfile, sps=24000)



"""
