import torch
import torch.nn as nn
from transformers import(
    BertForSequenceClassification,
    Wav2Vec2ForSequenceClassification
)


class TextAudioModel(nn.Module):
    def __init__(self, 
                 audio_model_path: str,
                 text_model_path: str,
                 num_labels: int,
                 criterion= nn.CrossEntropyLoss()):
        super(TextAudioModel, self).__init__()
        self.text_model_path = text_model_path
        self.audio_model_path = audio_model_path
        self.num_labels = num_labels        
        
        self.text_model = None
        self.audio_model = None
        
        self.load_model()
    
        self.text_model.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(256)
            )
        
        self.audio_model.classifier = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(256)
            )
        self.fc = nn.Linear(512, num_labels)
        self.criterion = criterion
    
    def forward(self, audio_input, text_input, text_mask, labels=None):
        audio = self.audio_model(audio_input).logits
        text = self.text_model(text_input,
                               token_type_ids=None,
                               attention_mask=text_mask).logits
     
        out = self.fc(torch.cat((audio, text), dim=1))
        
        if labels is None:
            loss = None
        else:
            loss = self.criterion(out, labels)
        
        return {'logits': out, 'loss': loss}
         
        
    def load_model(self):
        if '.pt' in self.text_model_path:
            print("Loading Text model from local repository")
            self.text_model = torch.load(self.text_model_path)
        else:
            print("Loading Text Model from Huggingface")
            self.text_model = BertForSequenceClassification.from_pretrained(
                self.text_model_path,
                num_labels=self.num_labels,
                output_attentions=False, 
                output_hidden_states=False
            )
            
        
        if '.pt' in self.audio_model_path:
            print("Loading Audio model from local repository")
            
            self.audio_model = torch.load(self.audio_model_path)
        else:
            print("Loading Audio model from Huggingface")
            
            self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.audio_model_path,
                num_labels=self.num_labels,
            )
            
            
        