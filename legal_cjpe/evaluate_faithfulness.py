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


#python evaluate_faithfulness.py --explainer_name loo


if __name__ == '__main__':

    parser = ArgumentParser(description='Evaluate faitfulness')

    parser.add_argument('--explainer_name', 
        help='Explainer name', 
        default='loo', 
        required=True,
        #choices=['loo', 'gradient', 'gradientXinput'],
        type=str)
    
    parser.add_argument('--input_data_dir', 
        help='Output directory in which explanations will be stored', 
        default='~/data/legal', 
        required=False,
        type=str)
    

    parser.add_argument('--masking', 
        help='Choose masking type', 
        default="none", 
        required=False,
        choices=['none', 'roberta'],
        type=str)
    
    parser.add_argument('--input_explanations_relevant_ids', 
        help='Output directory in which explanations will be stored', 
        default='explanations_discrete_ids_batching', 
        required=False,
        type=str)
    

    parser.add_argument('--model_dir', 
        help='Model directory', 
        default="~/models/legal",  
        required=False,
        type=str)
    

    parser.add_argument('--attention_layers', 
        help='Number attention layers', 
        default=4, 
        required=False,
        type=int)
    

    parser.add_argument('--mlp_layers', 
        help='Number mlp layers', 
        default=2, 
        required=False,
        type=int)
    
    parser.add_argument('--output_result', 
        help='Output dir of faithfulness results', 
        default='W_faithfulness_result', 
        required=False,
        type=str)
    
    parser.add_argument('--generate_predictions', 
    help='Flag to apply generate predictions, loaded otherwise', 
        default=False, action="store_true")
    
    
    parser.add_argument('--k_type', 
        help='Type used for defining the discrete explanation', 
        default="perc", 
        required=False,
        choices=['perc', 'k'],
        type=str)
    
    parser.add_argument('--k_value', 
        help='Value used for defining the discrete explanation', 
        default=40, 
        required=False,
        type=int)
    
    

    args = parser.parse_args()

    explainer_name = args.explainer_name
    masking_type = args.masking
    input_data_dir = args.input_data_dir
    generate_predictions = args.generate_predictions

    masking = args.masking
    k_type = args.k_type
    k_value = args.k_value
    output_result = args.output_result
    input_data_dir = args.input_data_dir



    # Model parameters
    attention_layers = args.attention_layers
    mlp_layers = args.mlp_layers

    lr = '5e-05'

    model_dir = args.model_dir
    model_name = f'second_level_train_last_mask-{masking_type}_{attention_layers}_{mlp_layers}_{lr}'
    model_name_clean = model_name.replace('-', '_')

    

    input_explanations_relevant_ids = args.input_explanations_relevant_ids

    relevant_sentences_ids_filename = os.path.join(input_explanations_relevant_ids, masking, explainer_name, f'relevant_sentences_ids_{model_name_clean}_{explainer_name}_{k_type}_{k_value}.csv')

    if explainer_name=='random':
        relevant_sentences_ids_filename = os.path.join(input_explanations_relevant_ids, 'random', explainer_name, f'relevant_sentences_ids_{explainer_name}_{k_type}_{k_value}.csv')


    if os.path.isfile(relevant_sentences_ids_filename)==False:
        print('NOT EXIST')
        print(f'{relevant_sentences_ids_filename} not exist')
        exit(-1)


    # Load embedding

    embd = np.load(os.path.join(input_data_dir, f'embeddings_{masking_type}_explain.npy'), allow_pickle=True)

    # Load classes
    true_classes = np.load(os.path.join(input_data_dir, f'labels_{masking_type}_explain.npy'), allow_pickle=True)
    true_classes = [i.numpy()[0] for i in true_classes]



    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_size = torch.from_numpy(embd[0]).shape[1]
    model = SecondLevelModel(d_model=embedding_size, d_hid=embedding_size, nlayers=attention_layers, mlp_layers=mlp_layers)
    t = torch.load(os.path.join(model_dir, model_name, 'model.pt'))
    model.eval()
    model.load_state_dict(t)
    model.to(device);

    from code.second_level_dataset import LJPESecondLevelClassificationDataset
    from torch.utils.data import DataLoader


    strategy = 'last'
    max_sentences = 256

    data_ds = LJPESecondLevelClassificationDataset(embd, np.array([1]*embd.shape[0]), strategy=strategy, max_sentences=max_sentences)


    ds_dataloader = DataLoader(data_ds, batch_size=56, shuffle=False, num_workers=16, pin_memory=True) 

    # Sentences 
    df_sentences = pd.read_csv(os.path.join(input_data_dir,'ILDC_for_explanation_sentences_masking.csv'))
    df_sentences.drop(columns=['Unnamed: 0'], inplace=True)

    with open(os.path.join(input_data_dir, 'ILDC_single_entities_roberta_explain_filenames_mapping.csv'), "r") as fp:
        filenames = json.load(fp)

    
    if generate_predictions == False:

        if os.path.isfile(os.path.join(input_data_dir,f'predicted_prob_masking_{masking_type}.npy')) == False:
            raise ValueError('Prediction not available')
        
        with open(os.path.join(input_data_dir,f'predicted_prob_masking_{masking_type}.npy'), 'rb') as f:
            output_probs = np.load(f)

        with open(os.path.join(input_data_dir,f'predicted_classes_masking_{masking_type}.npy'), 'rb') as f:
            predicted_classes = np.load(f)
            predicted_classes = list(predicted_classes)
    else:
        predicted_classes = []

   
    with torch.no_grad():
        for batch in ds_dataloader:
            embeddings, attention_masks, labels = batch
            embeddings = embeddings.to(device)
            attention_masks = attention_masks.to(device)

            if generate_predictions:
            
                output = model(embeddings, attention_masks.transpose(1, 0))
                output_probs = torch.cat([1-output, output], dim=1).detach().cpu().numpy()
                predicted_classes.extend((output.squeeze(1)>0.5).int().cpu().detach().numpy())

                # Save probs and predicted classes

                with open(os.path.join(input_data_dir,f'predicted_prob_masking_{masking_type}.npy'), 'wb') as f:
                    np.save(f, output_probs)

                with open(os.path.join(input_data_dir,f'predicted_classes_masking_{masking_type}.npy'), 'wb') as f:
                    np.save(f, np.array(predicted_classes))





    with open(relevant_sentences_ids_filename, 'rb') as handle:
        relevant_ids = pickle.load( handle)

    rename_class_name = {'Accepted':1, 'Denied':0}

    evaluation_results = {}

    compr = []

    with torch.no_grad():
        # For each document
        for doc_id in tqdm(range(len(data_ds))): 
            doc_name, relevant_sentence_id, target_class_name = relevant_ids[doc_id]

            if explainer_name=='random':
                target_class = predicted_classes[doc_id]
            else:
                target_class = rename_class_name[target_class_name]
            
            embeddings_rem = embeddings.clone()
            attention_masks_rem = attention_masks.clone()

            attention_masks_rem[doc_id][relevant_sentence_id] = 0

            # Output probability when removing relevant sentences
            output_rem = model(embeddings_rem.to(device), attention_masks_rem.transpose(1, 0).to(device))

            # Output class probability
            output_probs_remove = torch.cat([1-output_rem, output_rem], dim=1).detach().cpu().numpy()


            compr.append((output_probs[doc_id]-output_probs_remove[doc_id])[target_class])

    print(np.mean(compr))
    evaluation_results['compr'] = compr

    suff = []

    with torch.no_grad():
        # For each document
        for doc_id in tqdm(range(len(data_ds))): 
            doc_name, relevant_sentence_id, target_class_name = relevant_ids[doc_id]

            if explainer_name=='random':
                target_class = predicted_classes[doc_id]
            else:
                target_class = rename_class_name[target_class_name]

            # Array of sentence id, 0 to len(sentences)
            sentences_id = torch.where(attention_masks[doc_id]> 0)[0]

            ids_not_relevant = np.setdiff1d(sentences_id.cpu(), relevant_sentence_id)
            
            embeddings_rem = embeddings.clone()
            attention_masks_rem = attention_masks.clone()

            attention_masks_rem[doc_id][ids_not_relevant] = 0

            # Output probability when removing relevant sentences
            output_rem = model(embeddings_rem.to(device), attention_masks_rem.transpose(1, 0).to(device))

            # Output class probability
            output_probs_remove = torch.cat([1-output_rem, output_rem], dim=1).detach().cpu().numpy()


            suff.append((output_probs[doc_id]-output_probs_remove[doc_id])[target_class])

    np.mean(suff)
    evaluation_results['suff'] = suff


    # NO


    # NO

    compr2 = []

    with torch.no_grad():
        # For each document
        for doc_id in tqdm(range(len(data_ds))): 
            doc_name, relevant_sentence_id, target_class_name = relevant_ids[doc_id]

            if explainer_name=='random':
                target_class = predicted_classes[doc_id]
            else:
                target_class = rename_class_name[target_class_name]


            ids_not_relevant = np.setdiff1d(sentences_id.cpu(), relevant_sentence_id)
            
            embeddings_rem = embeddings.clone()
            embeddings_rem[doc_id][:ids_not_relevant.shape[0]] = embeddings[doc_id][ids_not_relevant]
            embeddings_rem[doc_id][ids_not_relevant.shape[0]:] = torch.zeros((embeddings_rem.shape[1]-ids_not_relevant.shape[0], embeddings_rem.shape[2]))

            attention_masks_rem[doc_id] = torch.zeros(attention_masks_rem[doc_id].shape)
            attention_masks_rem[doc_id][:ids_not_relevant.shape[0]] = 1


            # Output probability when removing relevant sentences
            output_rem = model(embeddings_rem.to(device), attention_masks_rem.transpose(1, 0).to(device))

            # Output class probability
            output_probs_remove = torch.cat([1-output_rem, output_rem], dim=1).detach().cpu().numpy()


            compr2.append((output_probs[doc_id]-output_probs_remove[doc_id])[target_class])

    suff2 = []

    with torch.no_grad():
        # For each document
        for doc_id in tqdm(range(len(data_ds))): 
            doc_name, relevant_sentence_id, target_class_name = relevant_ids[doc_id]

            if explainer_name=='random':
                target_class = predicted_classes[doc_id]
            else:
                target_class = rename_class_name[target_class_name]

                
            embeddings_rem = embeddings.clone()
            embeddings_rem[doc_id][:relevant_sentence_id.shape[0]] = embeddings[doc_id][relevant_sentence_id]
            embeddings_rem[doc_id][relevant_sentence_id.shape[0]:] = torch.zeros((embeddings_rem.shape[1]-relevant_sentence_id.shape[0], embeddings_rem.shape[2]))


            attention_masks_rem[doc_id] = torch.zeros(attention_masks_rem[doc_id].shape)
            attention_masks_rem[doc_id][:relevant_sentence_id.shape[0]] = 1

            # Output probability when removing relevant sentences
            output_rem = model(embeddings_rem.to(device), attention_masks_rem.transpose(1, 0).to(device))

            # Output class probability
            output_probs_remove = torch.cat([1-output_rem, output_rem], dim=1).detach().cpu().numpy()


            suff2.append((output_probs[doc_id]-output_probs_remove[doc_id])[target_class])
 
 
    evaluation_results['compr2'] = compr2
    evaluation_results['suff2'] = suff2

    output_result = os.path.join(output_result, masking_type, explainer_name)
    Path(output_result).mkdir(parents=True, exist_ok=True)


    with open(os.path.join(output_result, f'faithfulness_{model_name_clean}_{explainer_name}_{k_type}_{k_value}.pickle'), 'wb') as handle:
            pickle.dump(evaluation_results, handle)
