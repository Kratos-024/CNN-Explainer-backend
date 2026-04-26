import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
from img_file import TRANSFORM, model 

INT_TO_CLASS = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def _get_layer_blocks(mdl):
    # Adjust these names if your model architecture uses different attribute names
    return [
        ("Conv1", mdl.conv1),
        ("Conv2", mdl.conv2),
        ("Conv3", mdl.conv3),
        ("Conv4", mdl.conv4),
    ]

def _activation_stats(tensor: torch.Tensor):
    t = tensor.detach().cpu().float()
    mean = float(t.mean())
    var = float(t.var())
    sparsity = float((t == 0).float().mean() * 100)  
    return mean, var, sparsity

def _activation_distribution(tensor: torch.Tensor, bins=7):
    t = tensor.detach().cpu().float().numpy().flatten()
    counts, edges = np.histogram(t, bins=bins)
    result = []
    for i, count in enumerate(counts):
        label = f"{edges[i]:.1f}"
        result.append({"bin": label, "val": int(count)})
    return result

def _run_forward_and_collect(input_tensor: torch.Tensor, training_mode: bool):
    device = next(model.parameters()).device
    x = input_tensor.to(device)

    if training_mode:
        model.train()
        # FIX: BatchNorm requires batch_size > 1 in train mode.
        # We force BatchNorm layers to eval mode so they use running statistics,
        # while keeping Dropout layers in train mode.
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
    else:
        model.eval()

    blocks = _get_layer_blocks(model)
    layer_stats = []
    last_act = None

    with torch.set_grad_enabled(False):
        for block_name, block in blocks:
            x = block(x)
            mean, var, sparsity = _activation_stats(x)
            layer_stats.append({
                "layer":   block_name,
                "mean":    round(mean, 4),
                "var":     round(var, 4),
                "sparsity": round(sparsity, 2),
            })
            last_act = x

        x = model.gap(x)
        logits = model.classifier(x)

    # Always revert back to eval for safety
    model.eval()

    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    dist = _activation_distribution(last_act)

    return probs, layer_stats, dist

def get_dropout_comparison(image_bytes: bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_tensor = TRANSFORM(img).unsqueeze(0)
    
    # 1. Run without Dropout (Eval Mode)
    probs_no, stats_no, dist_no = _run_forward_and_collect(input_tensor, training_mode=False)
    
    # 2. Run with Dropout (Train Mode - but BN fixed)
    probs_do, stats_do, dist_do = _run_forward_and_collect(input_tensor, training_mode=True)

    pred_no = INT_TO_CLASS[int(np.argmax(probs_no))].capitalize()
    pred_do = INT_TO_CLASS[int(np.argmax(probs_do))].capitalize()

    class_prob_no = [
        {"cls": INT_TO_CLASS[i].capitalize(), "conf": round(p * 100, 1)}
        for i, p in enumerate(probs_no)
    ]
    class_prob_do = [
        {"cls": INT_TO_CLASS[i].capitalize(), "conf": round(p * 100, 1)}
        for i, p in enumerate(probs_do)
    ]

    layer_stats_merged = [
        {
            "layer":  stats_no[i]["layer"],
            "meanNo": stats_no[i]["mean"],
            "meanDo": stats_do[i]["mean"],
            "varNo":  stats_no[i]["var"],
            "varDo":  stats_do[i]["var"],
            "sparNo": stats_no[i]["sparsity"],
            "sparDo": stats_do[i]["sparsity"],
        }
        for i in range(len(stats_no))
    ]

    return {
        "predictionNo":  pred_no,
        "predictionDo":  pred_do,
        "classProbNo":   class_prob_no,
        "classProbDo":   class_prob_do,
        "layerStats":    layer_stats_merged,
        "distNo":        dist_no,
        "distDo":        dist_do,
    }