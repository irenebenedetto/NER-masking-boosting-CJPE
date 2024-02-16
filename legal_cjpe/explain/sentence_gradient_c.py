from captum.attr import Saliency, InputXGradient
import torch

def compute_feature_importances(
        input_embeds,
        attention_masks,
        model,
        multiply_by_inputs = False,
        target = 1,
    ):


        def func(input_embeds):
            output = model(
                input_embeds, attention_masks.transpose(1, 0)
            )
            output_prob = torch.cat([1-output, output], dim=1)
            return output_prob

        dl = InputXGradient(func) if multiply_by_inputs else Saliency(func)
        
        attr = dl.attribute(input_embeds, target=target)
        return attr
