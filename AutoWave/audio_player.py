from pydub.playback import play
from pydub import AudioSegment
def AudioPlayer(audioname):
    '''
    This function is used for playing the audio 
    '''
    audio=AudioSegment.from_wav(audioname)
    play(audio)