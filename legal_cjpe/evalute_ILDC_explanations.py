from argparse import ArgumentParser
from metricmaker import evaluate_explanation_ILDC_metrics
import os

import pandas as pd
from pathlib import Path


# CUDA_VISIBLE_DEVICES=1  python evalute_ILDC_explanations.py --explanation_dir ~/projects/legal/legaleval-ljpe-v2/legaleval-ljpe/output_explanations/none/loo


# CUDA_VISIBLE_DEVICES=1  python evalute_ILDC_explanations.py --explanation_dir ~/projects/legal/legaleval-ljpe-v2/legaleval-ljpe/output_explanations_masked/loo_NER5 --output_dir results_eval_explanations_masked
# CUDA_VISIBLE_DEVICES=1  python evalute_ILDC_explanations.py --explanation_dir ~/projects/legal/legaleval-ljpe-v2/legaleval-ljpe/output_explanations_masked/gradient_NER5 --output_dir results_eval_explanations_masked
# CUDA_VISIBLE_DEVICES=1  python evalute_ILDC_explanations.py --explanation_dir ~/projects/legal/legaleval-ljpe-v2/legaleval-ljpe/output_explanations_masked/gradientXinput_NER5 --output_dir results_eval_explanations_masked

if __name__ == '__main__':

    parser = ArgumentParser(description='Generate explanations')

    parser.add_argument('--explanation_dir', 
        help='Directory with explanations', 
        default='~/projects/legal/legaleval-ljpe-v2/legaleval-ljpe/output_explanations/loo', 
        required=False,
        type=str)


    parser.add_argument('--output_dir', 
        help='Output directory in which explanations will be stored', 
        default="results_eval_explanations", 
        required=False,
        type=str)
    
    args = parser.parse_args()
    explanation_dir = args.explanation_dir
    output_dir = args.output_dir


    import json
    with open('gold_explanations_ranked.json', 'r') as fp:
        gold_explanations_ranked =  json.load(fp)


    print('explanation_dir', explanation_dir)


    considered_combinations = [('perc', 40)] #, ('perc', 35), ('perc', 30), ('perc', 25), ('perc', 20), ('k', 10), ('k', 15), ('k', 20)] #[('perc', 45), ('perc', 50), ('perc', 100)]  #
    
    explanation_files = list(Path(os.path.join(explanation_dir)).glob('*.csv'))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(output_dir)
    print('Evaluating explanations...')
    for e, explanation_file in enumerate(explanation_files):
        print(f"{e+1}/{len(explanation_files)}")
        k_type, k_value = (str(explanation_file).split("_")[-2:])
        # Get type and parameter
        print(explanation_file)
        if (k_type, int(k_value[:-4])) in considered_combinations:
            df_explanation = pd.read_csv(explanation_file)
            doc_ids = list(df_explanation.uid)

            explanations_dict = df_explanation[['uid', 'explanation']].set_index('uid').to_dict()['explanation']

            # Evaluate explanation
            #experiments = [(1,10)] #, (1,1), (2,2), (3,3,), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (1,5), (5,10)]
            exp = (1, 10)
            res = evaluate_explanation_ILDC_metrics(doc_ids, exp[0], exp[1], explanations_dict, gold_explanations_ranked)

            res_df = pd.DataFrame(res).T
            res_df['mean'] = res_df.mean(axis=1)
            explanation_config = os.path.basename(explanation_file)[:-4]
            res_df.to_csv(os.path.join(output_dir, f"{explanation_config}_result.csv"))
    if len(explanation_files)==0:
        print('EMPTY!')