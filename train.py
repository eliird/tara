# training the combined model
import os
import pickle

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


from tara.model import TextAudioModel
from typing import Any
from tara.data import load_audio_dataloader, load_text_audio_dataloader
from transformers import  Wav2Vec2Processor, AutoTokenizer

from tara.trainer import TrainingArgs, TrainerTextAudio

text_model_path = "distilbert-base-uncased"
audio_model_path = "facebook/wav2vec2-base-960h"

processor = Wav2Vec2Processor.from_pretrained(audio_model_path)
tokenizer = AutoTokenizer.from_pretrained(text_model_path)


ta_model = TextAudioModel(
    audio_model_path,
    text_model_path,
    num_labels=7
)


train, test, val = load_text_audio_dataloader(
    './data',
    processor,
    tokenizer,
    batch_size=16
)

trainer = TrainerTextAudio(
    model= ta_model,
    trainloader=train,
    testloader=val,
    out_dir='./saved_weights/',
    args = TrainingArgs(epochs=20)
)

print("Trainer created")
trainer.train()

with open('./logs/training_stats.pkl', 'wb') as f:
    pickle.dump(trainer.training_stats, f)