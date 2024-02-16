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


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate explanations")

    parser.add_argument(
        "--explainer_name",
        help="Explainer name",
        default="loo",
        required=True,
        choices=["loo", "gradient", "gradientXinput"],
        type=str,
    )

    parser.add_argument(
        "--masking",
        help="Choose masking type",
        default="none",
        required=False,
        choices=["none", "roberta"],
        type=str,
    )

    parser.add_argument(
        "--input_data_dir",
        help="Output directory in which explanations will be stored",
        default="~/data/legal",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--model_dir",
        help="Model directory",
        default="~/models/legal",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--attention_layers",
        help="Number attention layers",
        default=4,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--mlp_layers", help="Number mlp layers", default=2, required=False, type=int
    )

    parser.add_argument(
        "--output_score_explanations",
        help="Output dir of output score explanations",
        default="score_explanations",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--generate_predictions",
        help="Flag to apply generate predictions, loaded otherwise",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--ner_model_path",
        help="Path of NER model",
        default="~/models/legal/NER/judgement/studio-ousia/luke-large/checkpoint-703",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--apply_ner_boosting",
        help="Flag to apply NER boosting",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--boosting_parameter",
        help="NER boosting parameter",
        default=5,
        required=False,
        type=int,
    )

    args = parser.parse_args()

    explainer_name = args.explainer_name
    masking_type = args.masking
    input_data_dir = args.input_data_dir
    output_score_explanations = args.output_score_explanations
    generate_predictions = args.generate_predictions
    apply_ner_boosting = args.apply_ner_boosting
    boosting_parameter = args.boosting_parameter
    ner_model_path = args.ner_model_path

    # Load embedding

    embd = np.load(
        os.path.join(input_data_dir, f"embeddings_{masking_type}_explain.npy"),
        allow_pickle=True,
    )

    # Load classes
    true_classes = np.load(
        os.path.join(input_data_dir, f"labels_{masking_type}_explain.npy"),
        allow_pickle=True,
    )
    true_classes = [i.numpy()[0] for i in true_classes]

    # Model parameters
    attention_layers = args.attention_layers
    mlp_layers = args.mlp_layers

    lr = "5e-05"

    model_dir = args.model_dir
    model_name = f"second_level_train_last_mask-{masking_type}_{attention_layers}_{mlp_layers}_{lr}"
    model_name_clean = model_name.replace("-", "_")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_size = torch.from_numpy(embd[0]).shape[1]
    model = SecondLevelModel(
        d_model=embedding_size,
        d_hid=embedding_size,
        nlayers=attention_layers,
        mlp_layers=mlp_layers,
    )
    t = torch.load(os.path.join(model_dir, model_name, "model.pt"))
    model.eval()
    model.load_state_dict(t)
    model.to(device)

    from code.second_level_dataset import LJPESecondLevelClassificationDataset
    from torch.utils.data import DataLoader

    strategy = "last"
    max_sentences = 256

    data_ds = LJPESecondLevelClassificationDataset(
        embd,
        np.array([1] * embd.shape[0]),
        strategy=strategy,
        max_sentences=max_sentences,
    )

    ds_dataloader = DataLoader(
        data_ds, batch_size=56, shuffle=False, num_workers=16, pin_memory=True
    )

    # Sentences
    df_sentences = pd.read_csv(
        os.path.join(input_data_dir, "ILDC_for_explanation_sentences_masking.csv")
    )
    df_sentences.drop(columns=["Unnamed: 0"], inplace=True)

    with open(
        os.path.join(
            input_data_dir, "ILDC_single_entities_roberta_explain_filenames_mapping.csv"
        ),
        "r",
    ) as fp:
        filenames = json.load(fp)

    if generate_predictions == False:
        if (
            os.path.isfile(
                os.path.join(
                    input_data_dir, f"predicted_prob_masking_{masking_type}.npy"
                )
            )
            == False
        ):
            raise ValueError("Prediction not available")

        with open(
            os.path.join(input_data_dir, f"predicted_prob_masking_{masking_type}.npy"),
            "rb",
        ) as f:
            output_probs = np.load(f)

        with open(
            os.path.join(
                input_data_dir, f"predicted_classes_masking_{masking_type}.npy"
            ),
            "rb",
        ) as f:
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
                output_probs = (
                    torch.cat([1 - output, output], dim=1).detach().cpu().numpy()
                )
                predicted_classes.extend(
                    (output.squeeze(1) > 0.5).int().cpu().detach().numpy()
                )

                # Save probs and predicted classes

                with open(
                    os.path.join(
                        input_data_dir, f"predicted_prob_masking_{masking_type}.npy"
                    ),
                    "wb",
                ) as f:
                    np.save(f, output_probs)

                with open(
                    os.path.join(
                        input_data_dir, f"predicted_classes_masking_{masking_type}.npy"
                    ),
                    "wb",
                ) as f:
                    np.save(f, np.array(predicted_classes))

    explanations = []
    loaded_explanations = False

    if apply_ner_boosting:
        # Check if we can load explanations
        score_explanation_dir = os.path.join(
            output_score_explanations, masking_type, explainer_name
        )
        Path(score_explanation_dir).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(
            score_explanation_dir,
            f"explanations_{model_name_clean}_{explainer_name}.pickle",
        )
        if os.path.isfile(filename):
            import pickle

            with open(filename, "rb") as handle:
                explanations = pickle.load(handle)
                loaded_explanations = True

            print(f"Explanations {explainer_name} loaded")
        else:
            raise ValueError

    # We load them in the case of (i) no boosting or (ii) we could not load them
    if apply_ner_boosting is False or loaded_explanations is False:
        print("Boosting", apply_ner_boosting)
        print("Loaded explanations", loaded_explanations)

        if explainer_name == "loo":
            print("Generating loo explanations...")

            with torch.no_grad():
                # For each document
                for doc_id in tqdm(range(len(data_ds))):
                    # Get sentences
                    sentences = list(
                        df_sentences.loc[
                            df_sentences["doc_index"] == doc_id
                        ].sentence.values
                    )

    # We generate them in the case of (i) no boosting or (ii) we could not load them
    if apply_ner_boosting is False or loaded_explanations is False:
        print("Boosting", apply_ner_boosting)
        print("Loaded explanations", loaded_explanations)

        if explainer_name == "loo":
            print("Generating loo explanations...")

            with torch.no_grad():
                # For each document
                for doc_id in tqdm(range(len(data_ds))):
                    # Get sentences
                    sentences = list(
                        df_sentences.loc[
                            df_sentences["doc_index"] == doc_id
                        ].sentence.values
                    )
                    if len(sentences) > 256:
                        # Truncate to last 256 sentences
                        sentences = sentences[-256:]

                    original_sentences = list(
                        df_sentences.loc[df_sentences["doc_index"] == doc_id][
                            "original_sentence"
                        ].values
                    )

                    if len(original_sentences) > 256:
                        # Truncate to last 256 sentences
                        original_sentences = original_sentences[-256:]

                    # Array of sentence id, 0 to len(sentences)
                    sentences_id = torch.where(attention_masks[doc_id] > 0)[0]

                    # Init importance scores
                    importances = []

                    # The target class for the explanation is the predicted one
                    target_class = predicted_classes[doc_id]

                    # For each sentence
                    for sentence_id in sentences_id:
                        embeddings_rem = embeddings.clone()
                        attention_masks_rem = attention_masks.clone()

                        # Set to 0 the attention for a single sentence
                        attention_masks_rem[doc_id][sentence_id] = 0

                        # Output probability when removing a single sentence
                        output_rem = model(
                            embeddings_rem.to(device),
                            attention_masks_rem.transpose(1, 0).to(device),
                        )

                        # Output class probability
                        output_probs_remove = (
                            torch.cat([1 - output_rem, output_rem], dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        importances.append(
                            (output_probs[doc_id] - output_probs_remove[doc_id])[
                                target_class
                            ]
                        )
                    output = Explanation(
                        None,
                        original_sentences,
                        np.array(importances),
                        "loo",
                        target_class,
                    )

                    explanations.append(output)

        elif explainer_name in ["gradient", "gradientXinput"]:
            print(f"Generating {explainer_name} explanations...")

            from explain.sentence_gradient_c import compute_feature_importances

            # For each document

            embeddings_rem = embeddings.clone()
            attention_masks_rem = attention_masks.clone()

            multiply_per_input = False if explainer_name == "gradient" else True

            print("multiply_per_input", multiply_per_input)

            gradient_importances = compute_feature_importances(
                embeddings,
                attention_masks,
                model,
                multiply_by_inputs=multiply_per_input,
                target=list(map(int, predicted_classes)),
            )

            for doc_id in tqdm(range(len(data_ds))):
                # Get sentences
                sentences = list(
                    df_sentences.loc[
                        df_sentences["doc_index"] == doc_id
                    ].sentence.values
                )
                if len(sentences) > 256:
                    # Truncate to last 256 sentences
                    sentences = sentences[-256:]

                original_sentences = list(
                    df_sentences.loc[df_sentences["doc_index"] == doc_id][
                        "original_sentence"
                    ].values
                )

                if len(original_sentences) > 256:
                    # Truncate to last 256 sentences
                    original_sentences = original_sentences[-256:]

                # Array of sentence id, 0 to len(sentences)
                sentences_id = torch.where(attention_masks[doc_id] > 0)[0]
                importances = (
                    gradient_importances[doc_id, sentences_id, :].detach().cpu().numpy()
                )

                # pool over hidden size
                importances = importances.sum(-1)

                target_class = predicted_classes[doc_id]

                explanation = Explanation(
                    None,
                    original_sentences,
                    np.array(importances),
                    explainer_name,
                    target_class,
                )

                explanations.append(explanation)

        else:
            raise ValueError

    if apply_ner_boosting:
        if masking_type == "none":
            idx_to_labels = legal_ner_labels_init()
            text = 'In this reference under Section 66 (1) of the Indian Income-tax Act, 1922, at the instance of the assessee Messrs. Dayabhai & Co. of Barwani, the question posed for our answer is: "Whether on the facts and in the circumstances of this case, the assessee is entitled to registration under Section 26-A of the Indian Income-tax Act for the assessment year 1956-57?" 2.'

            ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
            ner_model.to(device)
            ner_model.eval()
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

            ner_explainer = NERExplainer(ner_model, tokenizer, idx_to_labels)

        else:
            ner_explainer = NERExplainer(None, None, None)

        ner_explanations = []
        # TODO - move to discrete explanation generation!
        print("Generating NER importance scores..")
        with torch.no_grad():
            for i in tqdm(range(len(data_ds))):
                doc_id = i
                sentences = list(
                    df_sentences.loc[
                        df_sentences["doc_index"] == doc_id
                    ].sentence.values
                )

                if len(sentences) > 256:
                    # Truncate to last 256 sentences
                    sentences = sentences[-256:]

                if masking_type == "roberta":
                    original_sentences = list(
                        df_sentences.loc[df_sentences["doc_index"] == doc_id][
                            "original_sentence"
                        ].values
                    )

                    if len(original_sentences) > 256:
                        # Truncate to last 256 sentences
                        original_sentences = original_sentences[-256:]

                if masking_type == "none":
                    exp_ner = ner_explainer.compute_feature_importance(sentences)
                else:
                    exp_ner = ner_explainer.compute_feature_importance_wo_ner_call(
                        sentences
                    )
                    exp_ner.tokens = original_sentences

                ner_explanations.append(exp_ner)

        # Store a copy for later computing statistics of NER boosting
        original_explanations = deepcopy(explanations)

        # Generate boosted explanations
        explanations = ner_boosting(explanations, ner_explanations, boosting_parameter)

        explainer_name_original = explainer_name
        explainer_name = f"{explainer_name}_NER{boosting_parameter}"

    #### SAVE SCORE EXPLANATIONS

    score_explanation_dir = os.path.join(
        output_score_explanations, masking_type, explainer_name
    )
    Path(score_explanation_dir).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(
        score_explanation_dir,
        f"explanations_{model_name_clean}_{explainer_name}.pickle",
    )

    with open(filename, "wb") as handle:
        pickle.dump(explanations, handle)

    filename = os.path.join(
        score_explanation_dir,
        f"explanations_scores_{model_name_clean}_{explainer_name}.pickle",
    )

    with open(filename, "wb") as handle:
        pickle.dump([explanation.scores for explanation in explanations], handle)

    if apply_ner_boosting and loaded_explanations is False:
        # Store also original

        filename = os.path.join(
            score_explanation_dir,
            f"explanations_{model_name_clean}_{explainer_name_original}.pickle",
        )

        with open(filename, "wb") as handle:
            pickle.dump(original_explanations, handle)

        filename = os.path.join(
            score_explanation_dir,
            f"explanations_scores_{model_name_clean}_{explainer_name_original}.pickle",
        )

        with open(filename, "wb") as handle:
            pickle.dump(
                [explanation.scores for explanation in original_explanations], handle
            )
