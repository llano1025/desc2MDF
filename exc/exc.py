import torch
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from tqdm import tqdm
from utils.metrics import BertMetrics
import torch.nn.functional as F


class BertClassifier:
    def __init__(self, model_name, model_type, data):
        # Model initiation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.num_label = data.size
        self.label_dict = data.label_dict
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.num_label)
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5, eps=1e-8)
        self.epochs = 5
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=len(data.dataloader_train) * self.epochs)
        self.seed_val = 17
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)

    def _evaluate(self, dataloader_val):
        self.model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in dataloader_val:
            batch = tuple(b.to(self.device) for b in batch)

            inputs = {'input_ids': batch[0].to(self.device),
                      'attention_mask': batch[1].to(self.device),
                      'labels': batch[2].to(self.device),
                      }

            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    def _predict(self, input_ids):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                              num_labels=self.num_label,
                                                              output_attentions=False,
                                                              output_hidden_states=False)
        model.load_state_dict(torch.load(f'exc/checkpoints/{self.model_type}_finetuned_BERT_epoch_{self.epochs}.pth'))
                                         # map_location=torch.device('cpu')
        model.to(self.device)
        self.model.eval()
        predictions = []

        for i, inputs in enumerate(input_ids):
            inputs = inputs.to(self.device)

            with torch.no_grad():
                 outputs = self.model(inputs)

            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_index = torch.argmax(probabilities, dim=-1)
            key = list(self.label_dict)
            predictions.append(key[predicted_class_index])
        print(predictions)
        return predictions

    def _train(self, dataloader_train, dataloader_validation):
        for epoch in tqdm(range(1, self.epochs + 1)):

            self.model.train()
            loss_train_total = 0

            progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
            for batch in progress_bar:
                self.model.zero_grad()

                batch = tuple(b.to(self.device) for b in batch)

                inputs = {'input_ids': batch[0].to(self.device),
                          'attention_mask': batch[1].to(self.device),
                          'labels': batch[2].to(self.device),
                          }
                outputs = self.model(**inputs)

                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            torch.save(self.model.state_dict(), f'exc/checkpoints/{self.model_type}_typefinetuned_BERT_epoch_{epoch}.pth')

            tqdm.write(f'\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

            val_loss, predictions, true_vals = self._evaluate(dataloader_validation)
            score = BertMetrics(predictions, true_vals)
            val_f1 = score.f1_score_func
            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'F1 Score (Weighted): {val_f1}')
