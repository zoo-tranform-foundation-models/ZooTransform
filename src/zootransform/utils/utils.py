from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import torch.nn.functional as F
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_sequence_likelihoods(
    sequences, model, tokenizer, batch_size=512
):
    all_outputs = []
    likelihoods = []
    dataloader = DataLoader(sequences, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_seqs in tqdm(dataloader):
            target_ids = tokenizer(batch_seqs, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in target_ids.items()}
            outputs = model(**inputs)
            logits = outputs.logits.detach()
            all_outputs.append(logits)

            # Calculate probabilities per position
            batch_probs = F.softmax(logits, dim=-1)
            # Gather the prob for each correct token
            for i in range(len(target_ids["input_ids"])):
                t = target_ids["input_ids"][i][1:-1]
                seq_probs = batch_probs[i][torch.arange(len(t)), t]
                # Sequence (log-)likelihood (sum or mean depending on use-case)
                seq_likelihood = seq_probs.mean().item()
                likelihoods.append(seq_likelihood)
            del inputs, outputs, logits, batch_probs, seq_probs
            torch.cuda.empty_cache()
            gc.collect()
    return all_outputs, likelihoods


