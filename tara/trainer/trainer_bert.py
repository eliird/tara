import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import datetime, time, random
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class TrainingArgs:
    def __init__(self, device='cuda', learning_rate=2e-5, epsilon=1e-8, epochs=4) -> None:
        self.device = device
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epochs = epochs
        self.warmup_steps = 0
        self.seed = 1024

    
class TrainerBERT:
    def __init__(self,
                 model: nn.Module,
                 trainloader: DataLoader,
                 testloader: DataLoader,
                 out_dir: str,
                 args: TrainingArgs) -> None:
        
        self.args = args
        self.num_epochs = self.args.epochs
        self.device = self.args.device
        
        self.train_data = trainloader
        self.val_data = testloader

        self.out_dir = out_dir
        self.best_acc = 0
        
        self.training_stats = []

        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=self.args.lr, eps=self.args.epsilon)
        self.schedular = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps= self.args.warmup_steps, num_training_steps=len(trainloader) * self.args.epochs            
        )
        
        self.fix_seeds(self.args.seed)
    
    def fix_seeds(self, seed_val):
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        
    def validate(self):
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()
        # Tracking variables 
        best_eval_accuracy = 0
        total_eval_loss = 0
        
        preds = []
        labels = []
        
        for _, batch in enumerate(tqdm(self.val_data)):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                output= self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            loss = output.loss
            total_eval_loss += loss.item()
            # Move logits and labels to CPU if we are using GPU
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            preds.extend(np.argmax(logits, axis=1))
            labels.extend(label_ids)
            # total_eval_accuracy += accuracy_score(np.argmax(logits, axis=1), label_ids)
        # Report the final accuracy for this validation run.
        preds, labels = np.array(preds), np.array(labels)
        avg_val_accuracy = f1_score(preds.flatten(), labels.flatten(), average='weighted')# total_eval_accuracy / len(dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(self.val_data)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        if avg_val_accuracy > self.best_acc:
            torch.save(self.model.state_dict(), os.path.join(self.out_dir, 'bert.pt'))
            self.best_acc = avg_val_accuracy
            
        return (avg_val_accuracy, avg_val_loss, validation_time)
    
    def train_epoch(self, epoch_i):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.num_epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        self.model.train()
        for step, batch in enumerate(tqdm(self.train_data)):
        
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)        
            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    
            self.optimizer.step()
            self.schedular.step()
            

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(self.train_data)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))    
        return (avg_train_loss, training_time)
    
    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_time = self.train_epoch(epoch)
            val_acc, val_loss, val_time = self.validate()
            self.training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': train_loss,
                    'Valid. Loss': val_loss,
                    'Valid. Accur.': val_acc,
                    'Training Time': train_time,
                    'Validation Time': val_time
                }
            )