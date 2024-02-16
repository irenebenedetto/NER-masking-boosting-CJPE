

cuda_device=0
masking='roberta'
score_explanations_dir='X_BATCH_output_explanations'
explanations_discrete_dir='X_BATCH_explanations_discrete_batching'
explanations_relevant_ids_dir='X_BATCH_explanations_discrete_ids_batching'
output_evaluation_dir='X_BATCH_results_eval_explanations'

generate_predictions='--generate_predictions'


explainer='gradient'
CUDA_VISIBLE_DEVICES=${cuda_device}  python generate_explanations_batch.py  --explainer_name  ${explainer} --masking ${masking}  --output_score_explanations ${score_explanations_dir}  ${generate_predictions}



generate_explanations=false
generate_discrete_explanations=true
evaluate_explanations=false


for explainer in  gradient #loo gradientXinput
do
    echo ${masking} ${explainer} 'no boosting'

    if ${generate_explanations} ; then
        CUDA_VISIBLE_DEVICES=${cuda_device}  python generate_explanations_batch.py  --explainer_name ${explainer} --masking ${masking}  --output_score_explanations ${score_explanations_dir}
    fi
    if ${generate_discrete_explanations}; then
        CUDA_VISIBLE_DEVICES=${cuda_device}  python generate_discrete_explanations_from_scores.py  --explainer_name ${explainer} --masking ${masking}  --input_explanations_dir ${score_explanations_dir}/${masking}/${explainer} --output_explanations_discrete ${explanations_discrete_dir} --output_explanations_relevant_ids ${explanations_relevant_ids_dir} --model_name second_level_train_last_mask_${masking}_4_2_5e_05
    fi
    if ${evaluate_explanations} ; then
        CUDA_VISIBLE_DEVICES=${cuda_device}  python evalute_ILDC_explanations.py --explanation_dir ~/projects/legaleval-ljpe/${explanations_discrete_dir}/${masking}/${explainer} --output_dir ${output_evaluation_dir}
    fi
done



generate_explanations=false
generate_discrete_explanations=true
evaluate_explanations=false


for explainer in loo gradient gradientXinput
do

    for boosting in 1 2 5 7 3
    do
        echo ${masking} ${explainer} 'boosting'

        if ${generate_explanations} ; then
            CUDA_VISIBLE_DEVICES=${cuda_device}  python generate_explanations_batch.py  --explainer_name ${explainer} --apply_ner_boosting   --boosting_parameter ${boosting} --masking ${masking}  --output_score_explanations ${score_explanations_dir}
        fi
        if ${generate_discrete_explanations}; then
            CUDA_VISIBLE_DEVICES=${cuda_device}  python generate_discrete_explanations_from_scores.py  --explainer_name ${explainer}_NER${boosting} --masking ${masking}  --input_explanations_dir ${score_explanations_dir}/${masking}/${explainer}_NER${boosting} --output_explanations_discrete ${explanations_discrete_dir} --output_explanations_relevant_ids ${explanations_relevant_ids_dir} --model_name second_level_train_last_mask_${masking}_4_2_5e_05
        fi
        if ${evaluate_explanations} ; then
            CUDA_VISIBLE_DEVICES=${cuda_device}  python evalute_ILDC_explanations.py --explanation_dir ~/projects/legaleval-ljpe/${explanations_discrete_dir}/${masking}/${explainer}_NER${boosting} --output_dir ${output_evaluation_dir}
        fi
    done
done
