"""python 3.10
Example:
python train.py --model_path nlpaueb/legal-bert-base-uncased \
    --ds_path /content/NER_TRAIN/NER_TRAIN_PREAMBLE.json \
    --output_folder results \
    --config example_config/bert-config.json \
"""

import json
from code.dataset import LJPEClassificationDataset
from code.utils import compute_metrics
import spacy
import os
from transformers import Trainer, DefaultDataCollator, TrainingArguments
from transformers import AutoModelForSequenceClassification
from argparse import ArgumentParser
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":

    parser = ArgumentParser(description='Training script')
    parser.add_argument('--model_path', 
        help='HF model name', 
        default="roberta-base", 
        required=False, 
        type=str)
    parser.add_argument('--ds_train_path', 
        help='File name of the dataset', 
        default="trainData/ILDC_single_train_dev.csv", 
        required=False,
        type=str)
    parser.add_argument('--ds_valid_path', 
        help='File name of the dataset', 
        default="trainData/ILDC_single_train_dev.csv", 
        required=False, 
        type=str)
    parser.add_argument('--output_folder', 
        help='Output folder', 
        default="results/", 
        required=False, 
        type=str)
    parser.add_argument('--config', 
        help='Parameter config', 
        default="config/example_config.json", 
        required=False, 
        type=str)
    parser.add_argument('--strategy',
        help='Strategy for training',
        default="first",
        choices=["first", "last", "sentences"],
        required=False,
        type=str)
    parser.add_argument('--version',
        help='Strategy for training',
        default="text",
        choices=["text", "text-entity"],
        required=False,
        type=str)
  
    args = parser.parse_args()

    model_path = args.model_path        # e.g., 'roberta-base'
    ds_train_path = args.ds_train_path  # e.g., 'ILDC_single_train_dev.csv'
    ds_valid_path = args.ds_valid_path  # e.g., 'ILDC_single_train_dev.csv'
    output_folder = args.output_folder  # e.g., './results'
    strategy = args.strategy            # e.g., 'first'
    version = args.version              # e.g., 'text'
    config = json.load(open(args.config))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    new_tokens = ["[COURT]", "[JUDGE]", "[PETITIONERS]", "[RESPONDENT]", "[DATE]", "[ORG]", "[GPE]", "[PRECEDENT]", "[CASE_NUMBER]", "[WITNESS]", "[OTHER_PERSON]"]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    
    train_ds = LJPEClassificationDataset(ds_train_path, model_path, split="train", strategy=strategy, version=version, tokenizer=tokenizer)
    dev_ds = LJPEClassificationDataset(ds_valid_path, model_path, split="dev", strategy=strategy, version=version, tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    model.resize_token_embeddings(len(tokenizer))

    if version == "text-entity":
        if "roberta" in ds_train_path:
            tp = "roberta"
        elif "luke" in ds_train_path:
            tp = "luke"
        else:
            tp = "ERROR"
    else:
        tp = "null"


    data_collator = DefaultDataCollator()

    model_name = model_path.split("/")[-1]
    output_folder = f'results/{model_name}_{strategy}_{version}_{tp}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    training_args = TrainingArguments(
        output_dir=output_folder,
        num_train_epochs=config['EPOCHS'],
        learning_rate=config['LR'],
        per_device_train_batch_size=config['BATCH_SIZE'],
        per_device_eval_batch_size=config['BATCH_SIZE'],
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=2,
        metric_for_best_model="f1-macro",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(output_folder)
