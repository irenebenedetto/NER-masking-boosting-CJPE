'''
python save_embeddings.py   --tokenizer_path roberta-large 
                            --sentence_encoder_path results/roberta-large_last_text_roberta 
                            --ds_train_path trainData/ILDC_sentences.csv
'''

"""
CUDA_VISIBLE_DEVICES=1 python save_embedding_general.py  --ds_train_path ~/data/legal/ILDC_expert_data/ILDC_expert_explain_sentences.csv --sentence_encoder_path /data2/legaleval/roberta-large_last_text --split explain --output_path ~/data/legal/ILDC_expert_data
CUDA_VISIBLE_DEVICES=1 python save_embedding_general.py  --ds_train_path ~/data/legal/ILDC_multi_entities_roberta_explain_sentences_with_entities.csv --sentence_encoder_path /data2/legaleval/roberta-large_last_text-entity_roberta --split explain --output_path ~/data/legal/ILDC_expert_data



"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from argparse import ArgumentParser
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm

class SentenceDataset(Dataset):
    def __init__(self, dataset_path, tokenizer=None, split="train", strategy="first", max_sentences=256):
        self.split = split
        self.data = pd.read_csv(dataset_path)
        self.data = self.data[self.data['split'] == self.split]
        self.data = self.data.reset_index(drop=True)
        self.data = self.data.groupby('doc_index')
        self.group_indexes = list(self.data.groups.keys())
        self.tokenizer = tokenizer
        self.max_len = self.tokenizer.model_max_length
        self.strategy = strategy
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.group_indexes)

    def __getitem__(self, idx):
        sents = self.data.get_group(self.group_indexes[idx])["sentence"].tolist()
        if len(sents) > self.max_sentences:
            if self.strategy == "first":
                sents = sents[:self.max_sentences]
            elif self.strategy == "last":
                sents = sents[-self.max_sentences:]
            else:
                raise ValueError("Strategy not supported")
        tokenized = self.tokenizer(sents, padding="max_length", truncation=True, return_tensors="pt")
        '''
        if len(sents) < self.max_sentences:
            pad_amount = self.max_sentences - len(sents)
            tokenized["input_ids"] = torch.cat([tokenized["input_ids"], torch.zeros(pad_amount, self.max_len, dtype=torch.long)], dim=0)
            tokenized["attention_mask"] = torch.cat([tokenized["attention_mask"], torch.zeros(pad_amount, self.max_len, dtype=torch.long)], dim=0)
        '''
        label = self.data.get_group(self.group_indexes[idx])["label"].tolist()[0]
        return tokenized, label

if __name__ == "__main__":

    parser = ArgumentParser(description='Training script')
    parser.add_argument('--tokenizer_path', 
        help='HF model name', 
        default="roberta-large", 
        required=False, 
        type=str)
    parser.add_argument('--sentence_encoder_path', 
        help='HF model name', 
        default="roberta-large", 
        required=False, 
        type=str)
    parser.add_argument('--ds_train_path', 
        help='File name of the dataset', 
        default="trainData/ILDC_sentences.csv", 
        required=False,
        type=str)
    
    parser.add_argument('--split', 
        help='Split', 
        default="explain", 
        required=False,
        type=str)
    
    parser.add_argument('--output_path', 
        help='Output directory where embedding and labels will be stored. Created if not existing', 
        default="trainData/", 
        required=False,
        type=str)
    
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path        # e.g., 'roberta-base'
    sentence_encoder_path = args.sentence_encoder_path        # e.g., 'roberta-base'
    ds_train_path = args.ds_train_path  # e.g., 'ILDC_single_train_dev.csv'
    split = args.split #e.g, 'explain' or 'train'
    output_path = args.output_path # e.g., 'trainData/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    new_tokens = ["[COURT]", "[JUDGE]", "[PETITIONERS]", "[RESPONDENT]", "[DATE]", "[ORG]", "[GPE]", "[PRECEDENT]", "[CASE_NUMBER]", "[WITNESS]", "[OTHER_PERSON]"]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    data_split_dataset = SentenceDataset(ds_train_path, tokenizer, split=split, strategy="last", max_sentences=256)
    data_split_loader = torch.utils.data.DataLoader(data_split_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    sentence_encoder = AutoModelForSequenceClassification.from_pretrained(sentence_encoder_path, local_files_only=True)
    sentence_encoder.eval()
    sentence_encoder.to(device)

    if "roberta" in ds_train_path:
        tp = "roberta"
    elif "luke" in ds_train_path:
        tp = "luke"
    else:
        tp = "none"

    embeddings = []
    labels = []
    for i, batch in tqdm(enumerate(data_split_loader), total=len(data_split_loader)):
        with torch.no_grad():
            tokenized, label = batch
            input_ids = tokenized["input_ids"].view(-1, 512).to(device)
            attention_mask = tokenized["attention_mask"].view(-1, 512).to(device)
            output = sentence_encoder(input_ids, attention_mask, output_hidden_states=True)
            embeddings.append(output.hidden_states[-1][:, 0, :].cpu().numpy())
            labels.append(label)

            del attention_mask
            del output
            del tokenized
            del batch
            torch.cuda.empty_cache()

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    from pathlib import Path

    Path(output_path).mkdir(parents=True, exist_ok=True)


    np.save(f"{output_path}/embeddings_{tp}_{split}.npy", embeddings)
    np.save(f"{output_path}/labels_{tp}_{split}.npy", labels)

