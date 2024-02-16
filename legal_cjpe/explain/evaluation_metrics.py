import torch
import numpy as np


def classify_instance_in_dev(model, embeddings, attention_masks):
    embeddings = embeddings
    attention_masks = attention_masks
    e = embeddings.unsqueeze(0).repeat(256, 1, 1)
    a = attention_masks.unsqueeze(0).repeat(256, 1)
    output = model(e, a)[0].cpu().detach()
    #prob = output[0].detach().cpu()[0]
    output = torch.cat([output, 1-output], dim=0)
    return output #torch.Tensor([prob, 1-prob])


def evaluate_comprehensiveness(relevant_sentence_id, target, model, embeddings, attention_masks, device):
    embeddings = embeddings.to(device)
    attention_masks = attention_masks.to(device)

    original_output = classify_instance_in_dev(model, embeddings, attention_masks)


    
    attention_masks_remove =attention_masks.clone()
    attention_masks_remove[relevant_sentence_id] = 0

    del attention_masks
    torch.cuda.empty_cache()
    removed_output = classify_instance_in_dev(
            model, embeddings, attention_masks_remove
        )
    
    del attention_masks_remove
    torch.cuda.empty_cache()
    
    return original_output[target]-removed_output[target]


def evaluate_sufficiency(relevant_sentence_id, target, model, embeddings, attention_masks, device):


    not_relevant_sentence_id = [i for i in np.where(attention_masks==1)[0] if i not in relevant_sentence_id]

    
    embeddings = embeddings.to(device)
    attention_masks = attention_masks.to(device)
    attention_masks

    original_output = classify_instance_in_dev(model, embeddings, attention_masks)

    

    attention_masks_remove = attention_masks.clone()
    # Set to zero the not relevant
    attention_masks_remove[not_relevant_sentence_id] = 0

    

    del attention_masks
    torch.cuda.empty_cache()
    keep_output = classify_instance_in_dev(
            model, embeddings, attention_masks_remove
        )
    
    del attention_masks_remove
    torch.cuda.empty_cache()
    
    return original_output[target]-keep_output[target]