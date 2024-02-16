from transformers import (
    AutoModelForTokenClassification,
)
import torch
import numpy as np
from transformers import LukeTokenizer, AutoTokenizer, RobertaTokenizerFast


class NERExplainer:
    def __init__(self, ner_model_path, tokenizer):
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_model_path
        )
        self.ner_model.to('cuda')
        self.ner_model.eval()
        self.tokenizer = tokenizer

    def score_sentence_with_ner(self, text, idx_to_labels):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, verbose=False,  # padding='max_length'
        ).to('cuda')

        with torch.no_grad():
            logits = self.ner_model(**inputs).logits

        predicted_token_class_ids = logits.argmax(-1).squeeze(0).cpu().numpy().tolist()

        n_STATUTE = 0
        n_PROVISION = 0
        n_PRECEDENT = 0

        for predicted_label in predicted_token_class_ids:
            predicted_token_class = idx_to_labels[predicted_label]

            if 'STATUTE' in predicted_token_class:
                n_STATUTE += 1

            if 'PROVISION' in predicted_token_class:
                n_PROVISION += 1

            if 'PRECEDENT' in predicted_token_class:
                n_PRECEDENT += 1

        score = (n_STATUTE + n_PROVISION + n_PRECEDENT) / len(predicted_token_class_ids)
        return score


if __name__ == "__main__":
    model = './NER models/judgment/studio-ousia/luke-large/checkpoint-703'

    original_label_list = [
        "COURT",
        "PETITIONER",
        "RESPONDENT",
        "JUDGE",
        "DATE",
        "ORG",
        "GPE",
        "STATUTE",
        "PROVISION",
        "PRECEDENT",
        "CASE_NUMBER",
        "WITNESS",
        "OTHER_PERSON",
    ]

    labels_list = ["B-" + l for l in original_label_list]
    labels_list += ["I-" + l for l in original_label_list]
    labels_list = sorted(labels_list + ["O"])[::-1]
    labels_to_idx = dict(
        zip(sorted(labels_list)[::-1], range(len(labels_list)))
    )
    idx_to_labels = {v[1]: v[0] for v in labels_to_idx.items()}
    text = "In this reference under Section 66 (1) of the Indian Income-tax Act, 1922, at the instance of the assessee Messrs. Dayabhai & Co. of Barwani, the question posed for our answer is: \"Whether on the facts and in the circumstances of this case, the assessee is entitled to registration under Section 26-A of the Indian Income-tax Act for the assessment year 1956-57?\" 2."

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    ner_expl = NERExplainer(ner_model_path=model, tokenizer=tokenizer)
    score = ner_expl.score_sentence_with_ner(text, idx_to_labels)
    print(f'Score: {score}')