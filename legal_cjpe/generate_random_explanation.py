import warnings
warnings.filterwarnings("ignore")

from architecture.second_level_model import SecondLevelModel
from argparse import ArgumentParser
from ferret.explainers.explanation import Explanation
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
import json
from tqdm import tqdm
import pickle
from copy import deepcopy

from explain.ner_explainer import legal_ner_labels_init, NERExplainer
from explain.utils_explain import ner_boosting, get_most_relevant_sentences_ids
from transformers import AutoModelForTokenClassification, RobertaTokenizerFast
from explain.utils_explain import get_most_relevant_sentences, get_most_relevant_sentences_ids



#python generate_random_explanation.py  --output_score_explanations X_BATCH_output_explanations --output_explanations_discrete X_BATCH_explanations_discrete_batching --output_explanations_relevant_ids X_BATCH_explanations_discrete_ids_batching
#python evalute_ILDC_explanations.py --explanation_dir ~/projects/legaleval-ljpe/X_BATCH_explanations_discrete_batching/random --output_dir PROVA_RANDOM



if __name__ == '__main__':

    parser = ArgumentParser(description='Generate explanations')
 
    parser.add_argument('--input_data_dir', 
        help='Output directory in which explanations will be stored', 
        default='~/data/legal', 
        required=False,
        type=str)
    
    
    parser.add_argument('--output_score_explanations', 
        help='Output dir of output score explanations', 
        default='score_explanations', 
        required=False,
        type=str)
    
    parser.add_argument('--output_explanations_discrete', 
        help='Output dir of output discrete explanations', 
        default='explanations_discrete_batching', 
        required=False,
        type=str)
    
    parser.add_argument('--output_explanations_relevant_ids', 
        help='Output dir of output discrete explanations of relevant ids', 
        default='explanations_discrete_ids_batching', 
        required=False,
        type=str)


    args = parser.parse_args()

    explainer_name = 'random'
    input_data_dir = args.input_data_dir
    output_score_explanations = args.output_score_explanations
    output_explanations_discrete = args.output_explanations_discrete
    output_explanations_relevant_ids = args.output_explanations_relevant_ids



    # Sentences 
    df_sentences = pd.read_csv(os.path.join(input_data_dir,'ILDC_for_explanation_sentences_masking.csv'))
    df_sentences.drop(columns=['Unnamed: 0'], inplace=True)

    with open(os.path.join(input_data_dir, 'ILDC_single_entities_roberta_explain_filenames_mapping.csv'), "r") as fp:
        filenames = json.load(fp)

    

   
    
    explanations = []
    loaded_explanations = False

    print(len(filenames))


    # For each document
    for doc_id in tqdm(range(len(filenames))): 
        # Get sentences
        original_sentences = list(df_sentences.loc[df_sentences['doc_index']==doc_id]['original_sentence'].values)
    
        if len(original_sentences)>256:
            # Truncate to last 256 sentences
            original_sentences = original_sentences[-256:]

        import random
        importances = [random.random() for i in range(10)]
        output = Explanation(
                    None,
                    original_sentences,
                    np.array(importances),
                    explainer_name,
                    'random',
                )

        explanations.append(output)

    print(len(explanations))


    #### SAVE SCORE EXPLANATIONS
    
    score_explanation_dir = os.path.join(output_score_explanations, explainer_name)
    Path(score_explanation_dir).mkdir(parents=True, exist_ok=True)



    filename = os.path.join(score_explanation_dir, f'explanations_{explainer_name}.pickle')

    with open(filename, 'wb') as handle:
        pickle.dump(explanations, handle)

    filename = os.path.join(score_explanation_dir, f'explanations_scores_{explainer_name}.pickle')
    
    with open(filename, 'wb') as handle:
        pickle.dump([explanation.scores for explanation in explanations], handle)


    output_explanations_discrete = os.path.join(output_explanations_discrete, explainer_name)
    Path(output_explanations_discrete).mkdir(parents=True, exist_ok=True)

    output_explanations_relevant_ids = os.path.join(output_explanations_relevant_ids, explainer_name)
    Path(output_explanations_relevant_ids).mkdir(parents=True, exist_ok=True)

    relevant_sentences = []
    
    for (k_type, k_value) in [('perc', 40)]:#, ('perc', 35), ('perc', 30), ('perc', 25), ('perc', 20)]:

        pred_expl_results = []
        cnt = 0
        avg_diff = 0

        for i, explanation in enumerate(explanations):
            doc_id = filenames[i]
            predicted_class = 'random'
            text_explanation = get_most_relevant_sentences(explanation, k_type, k_value)
            pred_expl_results.append([doc_id, predicted_class, text_explanation])
            relevant_sentences.append([doc_id, get_most_relevant_sentences_ids(explanation, k_type, k_value), predicted_class])
            



        df_pred =pd.DataFrame(pred_expl_results, columns=['uid', 'decision', 'explanation'])

        discrete_explanation_filename = os.path.join(output_explanations_discrete, f'predictions_{explainer_name}_{k_type}_{k_value}.csv')
        df_pred.to_csv(discrete_explanation_filename, index=False)

        relevant_sentences_ids_filename = os.path.join(output_explanations_relevant_ids, f'relevant_sentences_ids_{explainer_name}_{k_type}_{k_value}.csv')
        with open(relevant_sentences_ids_filename, 'wb') as handle:
            pickle.dump(relevant_sentences, handle)
    