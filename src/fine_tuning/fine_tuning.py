from typing import List
import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from src.model.species_model import SpeciesAwareESM2

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from tqdm.auto import tqdm


class LoraFinetuner:
    """
    Fine-tune species-aware ESM2 using LoRA, optionally aligning to frozen embeddings.
    Args:
        base_model: an initialized SpeciesAwareESM2 object
        r, alpha, dropout: LoRA hyperparameters (default r=8, alpha=16, dropout=0.05)
        target_modules: list of module names (such as ["attention.self.key","attention.self.value"])
        lr: learning rate (default 1e-4)
        batch_size: batch size (default 4)
    """
    def __init__(self, base_model: SpeciesAwareESM2, r=8, alpha=16, dropout=0.05, target_modules=None, lr=1e-4, batch_size=4):
        self.device = base_model.device
        self.tokenizer = base_model.tokenizer
        self.max_length = base_model.max_length
        self.batch_size = batch_size

        if target_modules is None:
            target_modules = ["attention.self.key", "attention.self.value"]

        # Wrap model with LoRA
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.model = get_peft_model(base_model.model, lora_cfg).to(self.device)
        # self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        ) # only optimize the trainable LoRA params
        self.loss_fn = nn.MSELoss()  # Align embeddings to frozen ESM2

    def train(self, species_batch, sequence_batch, frozen_embeddings=None, epochs=3):
        dataset = ProteinDataset(
            species_batch=[f"<sp_{s}>" for s in species_batch],
            sequence_batch=sequence_batch,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar: #TODO - labels would be added in supervised case
                batch = {k: v.to(self.device) for k,v in batch.items()}
                outputs = self.model(**batch)
                embeddings = outputs.last_hidden_state.mean(dim=1) # mean pooling

                # Optional: align to frozen embeddings
                if frozen_embeddings is not None and len(frozen_embeddings) >= embeddings.size(0):
                    target_batch = frozen_embeddings[:embeddings.size(0)].to(self.device)
                    loss = self.loss_fn(embeddings, target_batch)
                    frozen_embeddings = frozen_embeddings[embeddings.size(0):]  # move window
                else:
                    # Self-supervised
                    # loss = embeddings.norm()  #TODO - placeholder for self-supervised loss
                    loss = (embeddings ** 2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if len(loader) > 0:
                avg_loss = epoch_loss / len(loader)
            else:
                avg_loss = float("nan")

            total_loss = avg_loss
            print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

        return float(total_loss)

    @torch.no_grad()
    def embed(self, species_batch, sequence_batch):
        self.model.eval()
        dataset = ProteinDataset(
            species_batch=[f"<sp_{s}>" for s in species_batch],
            sequence_batch=sequence_batch,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        loader = DataLoader(dataset, batch_size=self.batch_size)
        all_embeddings = []
        for batch in loader:
            batch = {k: v.to(self.device) for k,v in batch.items()}
            outputs = self.model(**batch)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0).cpu()

class ProteinDataset(Dataset):
    """
    Dataset for species-aware protein sequences.
    Args:
        species_batch: list of species identifiers
        sequence_batch: list of protein sequences
        tokenizer: tokenizer for the model
        max_length: maximum sequence length (default 1024)
    """

    def __init__(self, species_batch, sequence_batch, tokenizer, max_length=1024):
        self.species_batch = species_batch
        self.sequence_batch = sequence_batch
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequence_batch)

    def __getitem__(self, idx):
        text = f"{self.species_batch[idx]} {self.sequence_batch[idx]}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt")

        return {k: v.squeeze(0) for k,v in enc.items()}

# class LoraESMFinetuner:
#     """
#     Fine-tune the SpeciesAwareESM2 model with LoRA using
#     self-supervised Masked Language Modeling (MLM).
#     """
#
#     def __init__(self, base_model, r=8, alpha=16, dropout=0.05,
#                  target_modules=None, lr=1e-4, batch_size=4,
#                  mlm_probability=0.15):
#         """
#         Args:
#             base_model: an initialized SpeciesAwareESM2 object
#             r, alpha, dropout: LoRA hyperparameters
#             target_modules: list of module names (usually ["q_proj","v_proj"])
#             lr: learning rate
#             batch_size: batch size
#             mlm_probability: fraction of tokens to mask
#         """
#
#         self.device = base_model.device
#         self.tokenizer = base_model.tokenizer
#         self.max_length = base_model.max_length
#         self.batch_size = batch_size
#         # self.mlm_probability = mlm_probability
#
#         # ---- Wrap model with LoRA ----
#         if target_modules is None:
#             target_modules = ["attention.self.key", "attention.self.value"]
#
#         # Freeze base model
#         for param in base_model.model.parameters():
#             param.requires_grad = False
#
#         lora_cfg = LoraConfig(
#             r=r,
#             lora_alpha=alpha,
#             target_modules=target_modules,
#             lora_dropout=dropout,
#             bias="none",
#             task_type="FEATURE_EXTRACTION",  # ESM is a language model
#         )
#         self.model = get_peft_model(base_model.model, lora_cfg).to(self.device)
#
#         self.optimizer = AdamW(self.model.parameters(), lr=lr)
#         self.mse_loss = nn.MSELoss()
#         # self.data_collator = DataCollatorForLanguageModeling(
#         #     tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_probability
#         # )
#         print(f"LoRA finetuner ready on {self.device}")
#
#     def _make_dataset(self, species_batch, sequence_batch, teacher_embeddings):
#         texts = [f"<sp_{s}> {seq}" for s, seq in zip(species_batch, sequence_batch)]
#         encodings = self.tokenizer(
#             texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=self.max_length
#         )
#         teacher_embeddings = torch.tensor(teacher_embeddings, dtype=torch.float32)
#
#         class _ProteinDataset(Dataset):
#             def __init__(self, encodings, teacher_emb):
#                 self.encodings = encodings
#                 self.teacher_emb = teacher_emb
#             def __len__(self):
#                 return self.teacher_emb.shape[0]
#             def __getitem__(self, idx): #override method to get item at index idx
#                 return {k: v[idx] for k, v in self.encodings.items()}, self.teacher_emb[idx]
#
#         return _ProteinDataset(encodings, teacher_embeddings)
#
#     def train(self, species_batch, sequence_batch, teacher_embeddings, epochs=3):
#         dataset = self._make_dataset(species_batch, sequence_batch, teacher_embeddings)
#         loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
#
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0
#             for batch_inputs, teacher_emb in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
#                 batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
#                 teacher_emb = teacher_emb.to(self.device)
#
#                 outputs = self.model(**batch_inputs)
#                 seq_emb = outputs.last_hidden_state[:, 1:teacher_emb.shape[1]+1, :]  # skip species token
#                 loss = self.mse_loss(seq_emb, teacher_emb)
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 total_loss += loss.item()
#
#             print(f"Epoch {epoch+1} completed — avg distillation loss: {total_loss/len(loader):.4f}")
#
#     @torch.no_grad()
#     def embed(self, species_batch, sequence_batch):
#         self.model.eval()
#         texts = [f"<sp_{s}> {seq}" for s, seq in zip(species_batch, sequence_batch)]
#         inputs = self.tokenizer(
#             texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=self.max_length
#         ).to(self.device)
#         outputs = self.model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).cpu().numpy()