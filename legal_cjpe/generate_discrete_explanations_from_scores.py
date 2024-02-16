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



def compute_ner_boosting_difference_stats(boosted_explanation, original_explanation, k_type, k_value, cnt, avg_diff):
    from explain.utils_explain import get_most_relevant_sentences_ids
    boosted_sentence_top_ids = get_most_relevant_sentences_ids(boosted_explanation, k_type, k_value)
    # original_explanation = original_explanations[i]
    sentence_top_ids  = get_most_relevant_sentences_ids(original_explanation, k_type, k_value)
    diff =  list(set(boosted_sentence_top_ids) - set(sentence_top_ids))

    if diff!=[]:
        cnt+=1
        avg_diff += len(diff)/ boosted_sentence_top_ids.shape[0]
    return cnt, avg_diff

if __name__ == '__main__':

    parser = ArgumentParser(description='Generate explanations')

    parser.add_argument('--explainer_name', 
        help='Explainer name', 
        default='loo', 
        required=True,
        #choices=['loo', 'gradient', 'gradientXinput'],
        type=str)

    
    parser.add_argument('--input_explanations_dir', 
        help='Input directory in which explanations are stored', 
        default='~/projects/legaleval-ljpe/BATCH_output_explanations/none/loo', 
        required=False,
        type=str)
    
    parser.add_argument('--model_name', 
        help='Model name', 
        default='second_level_train_last_mask_none_4_2_5e_05', 
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
    
    parser.add_argument('--filename_path', 
        help='Filename mapping', 
        default='~/data/legal/ILDC_expert_explain_filenames_mapping.csv', 
        required=False,
        type=str)
    
    parser.add_argument('--masking', 
        help='Choose masking type', 
        default="none", 
        required=False,
        choices=['none', 'roberta'],
        type=str)
    
    parser.add_argument('--input_data_dir', 
        help='Output directory in which explanations will be stored', 
        default='~/data/legal', 
        required=False,
        type=str)
    
    
    

    args = parser.parse_args()

    input_explanations_dir = args.input_explanations_dir

    output_explanations_discrete = args.output_explanations_discrete

    output_explanations_relevant_ids = args.output_explanations_relevant_ids

    explainer_name = args.explainer_name
    model_name = args.model_name
    filename_path = args.filename_path
    masking_type = args.masking
    input_data_dir = args.input_data_dir
    
    
    df_eval_results = []
    relevant_sentences = []

    from explain.utils_explain import get_most_relevant_sentences, get_most_relevant_sentences_ids
    rename_class = {1: 'Accepted', 0: 'Denied'}

    from pathlib import Path

    filename = os.path.join(input_explanations_dir, f'explanations_{model_name}_{explainer_name}.pickle')

    print(filename)

    with open(filename, 'rb') as handle:
        explanations = pickle.load(handle)

    
    print(os.path.join(input_data_dir,f'predicted_classes_masking_{masking_type}.npy'))

    with open(os.path.join(input_data_dir,f'predicted_classes_masking_{masking_type}.npy'), 'rb') as f:
            predicted_classes = np.load(f)
            predicted_classes = list(predicted_classes)


    with open(os.path.join(input_data_dir, 'ILDC_single_entities_roberta_explain_filenames_mapping.csv'), "r") as fp:
        filenames = json.load(fp)



    output_explanations_discrete = os.path.join(output_explanations_discrete, masking_type, explainer_name)
    Path(output_explanations_discrete).mkdir(parents=True, exist_ok=True)

    output_explanations_relevant_ids = os.path.join(output_explanations_relevant_ids, masking_type, explainer_name)
    Path(output_explanations_relevant_ids).mkdir(parents=True, exist_ok=True)


    if '_NER' in explainer_name:

        applied_ner_boosting = True
        explainer_name_original, boosting = explainer_name.split('_NER')
        print(f'Boosted explainer: {explainer_name_original} by {boosting}')
        original_explanations_filename = os.path.join(input_explanations_dir[:-len('_NER'+boosting)], f'explanations_{model_name}_{explainer_name_original}.pickle')
        with open(original_explanations_filename, 'rb') as handle:
            original_explanations = pickle.load(handle)

        boosted_stats = {}
    else:
        applied_ner_boosting = False
        
    model_name_clean = model_name.replace('-', '_')
        
    for (k_type, k_value) in [('perc', 40), ('perc', 35), ('perc', 30), ('perc', 25), ('perc', 20), ('perc', 45), ('perc', 50), ('perc', 100)]:

        pred_expl_results = []
        cnt = 0
        avg_diff = 0
        relevant_sentences = []

        for i, explanation in enumerate(explanations):
            doc_id = filenames[i]
            predicted_class = rename_class[predicted_classes[i]]
            text_explanation = get_most_relevant_sentences(explanation, k_type, k_value)
            pred_expl_results.append([doc_id, predicted_class, text_explanation])
            relevant_sentences.append([doc_id, get_most_relevant_sentences_ids(explanation, k_type, k_value), predicted_class])

            if applied_ner_boosting:
                cnt, avg_diff = compute_ner_boosting_difference_stats(explanation, original_explanations[i], k_type, k_value, cnt, avg_diff)
                
        if applied_ner_boosting and  cnt==0:
            print(f'NO boosting for {k_type} {k_value}: {cnt}')
            boosted_stats[f'{k_type}_{k_value}'] = None
            continue

        if applied_ner_boosting and  cnt!=0:
            print(f'Boosting for {k_type} {k_value}: {cnt}/{len(explanations)}')

        if applied_ner_boosting:
            boosted_stats[f'{k_type}_{k_value}']= {'modified': {cnt/len(explanations)}, 'avg_diff_perc': {avg_diff/len(explanations)}}


        df_pred =pd.DataFrame(pred_expl_results, columns=['uid', 'decision', 'explanation'])
        discrete_explanation_filename = os.path.join(output_explanations_discrete, f'predictions_{model_name_clean}_{explainer_name}_{k_type}_{k_value}.csv')
        df_pred.to_csv(discrete_explanation_filename, index=False)

        relevant_sentences_ids_filename = os.path.join(output_explanations_relevant_ids, f'relevant_sentences_ids_{model_name_clean}_{explainer_name}_{k_type}_{k_value}.csv')
        with open(relevant_sentences_ids_filename, 'wb') as handle:
            pickle.dump(relevant_sentences, handle)
    
    if applied_ner_boosting:
        with open(os.path.join(output_explanations_discrete, f'NERstats_{model_name_clean}_{explainer_name}.pickle'), 'wb') as handle:
            pickle.dump(boosted_stats, handle)
