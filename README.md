# Text and Audio Augmented Reality (TARA) Emotion Detection


## Installation
> git clone https://github.com/eliird/tara
> pip install -e .
> Download the weights from the following [link](https://mega.nz/file/DJox1YgS#RbfBzYmvCisgBjqSFY3MY5gvsEXiU4wBTNNnKuMFoKk)

## Usage
```
from tara.inference import EmotionRecognition
import librosa as lb

# initialize the model
# the model requires the path of the saved_weights
model = EmotionRecognition('./saved_weights/text_audio.pt')

# load the audio data in the library
# sampling rate should be 16000

f_path = './dia0_utt0.mp4'
audio, sr = lb.load(f_path, sr=16000)

# get the text info from speech to text model
text = "something person said in the video" # from speech 2 text model

# detect the model
out = model.detect(audio, text)
out

```

## Things to note
The model is trained on maximum of 3 second audio data and the maximum length of sentence = 95.

The best f1 score of the model is 62% on the `MELD` dataset.

Make sure the audio data is used is less than 3 seconds. If the data is greater than that the code automatically uses the first 3 seconds only.