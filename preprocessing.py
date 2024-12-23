import tensorflow as tf
from omegaconf import OmegaConf


def load_config(config_file):
   global config
   config=config_file   

def decode_audio(audio_binary):
    audio,sr=tf.audio.decode_wav(audio_binary)
    audio=tf.squeeze(audio,axis=-1)
    audio=tf.signal.frame(audio,config.preprocessing.duration*config.preprocessing.target_sr,config.preprocessing.duration*config.preprocessing.target_sr,True)
    return audio

def get_spec(noisy,clean):
  noisy_audio_binary=tf.io.read_file(noisy)
  clean_audio_binary=tf.io.read_file(clean)
  noisy_waveform=decode_audio(noisy_audio_binary)
  clean_waveform=decode_audio(clean_audio_binary)
  noisy_waveform=tf.cast(noisy_waveform,tf.float32)
  clean_waveform=tf.cast(clean_waveform,tf.float32)
  
  noisy_spectrogram=tf.signal.stft(
      noisy_waveform,frame_length=config.preprocessing.frame_length,frame_step=config.preprocessing.frame_step)
  noisy_spectrogram=tf.abs(noisy_spectrogram)
  clean_spectrogram=tf.signal.stft(
      clean_waveform,frame_length=config.preprocessing.frame_length,frame_step=config.preprocessing.frame_step)
  clean_spectrogram=tf.abs(clean_spectrogram)
#   return tf.expand_dims(noisy_spectrogram,axis=-1),tf.expand_dims(clean_spectrogram,axis=-1)
  return noisy_spectrogram,clean_spectrogram

if __name__=="__main__":
   load_config()