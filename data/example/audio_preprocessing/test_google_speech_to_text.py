#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-09-07 17:14:51
# Name       : test_google_speech_to_text.py
# Version    : V1.0
# Description: .
#========================================
import io
from google.cloud import speech_v1p1beta1
from google.cloud.speech_v1p1beta1 import enums


def sample_recognize(local_file_path):
    """
    Performs synchronous speech recognition on an audio file

    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
    """

    client = speech_v1p1beta1.SpeechClient()

    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.mp3'

    # The language of the supplied audio
    language_code = "de-DE"
    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 44100

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
    }

    with io.open(local_file_path, "rb") as f:
        content = f.read()
    audio = {"content": content}
    # audio = {"uri": storage_uri}

    response = client.recognize(config, audio)
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]

        print(u"Transcript: {0}\n Confidence: {1}".format(
            alternative.transcript, alternative.confidence))

sample_recognize('/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/NUK/Audio/' +\
                 'Session_01/Exp_data/All/Combined/QA_T_001.wav')

# sample_recognize('/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/NUK/Audio/' +\
                 # 'Session_01/Exp_data/All/Valid_segs/QA_trial_0.wav')
