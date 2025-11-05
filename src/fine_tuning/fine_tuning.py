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

class LoraESMFinetuner:
    """
    Fine-tune the SpeciesAwareESM2 model with LoRA using
    self-supervised Masked Language Modeling (MLM).
    """

    def __init__(self, base_model, r=8, alpha=16, dropout=0.05,
                 target_modules=None, lr=1e-4, batch_size=4,
                 mlm_probability=0.15):
        """
        Args:
            base_model: an initialized SpeciesAwareESM2 object
            r, alpha, dropout: LoRA hyperparameters
            target_modules: list of module names (usually ["q_proj","v_proj"])
            lr: learning rate
            batch_size: batch size
            mlm_probability: fraction of tokens to mask
        """

        self.device = base_model.device
        self.tokenizer = base_model.tokenizer
        self.max_length = base_model.max_length
        self.batch_size = batch_size
        self.mlm_probability = mlm_probability

        # ---- Wrap model with LoRA ----
        if target_modules is None:
            target_modules=["attention.self.key", "attention.self.value"]

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",  # ESM is a language model
        )
        self.model = get_peft_model(base_model.model, lora_cfg).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_probability
        )
        print(f"LoRA finetuner ready on {self.device}")

    def _make_dataset(self, species_batch, sequence_batch):
        """Tokenize data and build a torch Dataset."""

        texts = [f"<sp_{s}> {seq}" for s, seq in zip(species_batch, sequence_batch)]
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=self.max_length
        )

        class _ProteinDataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __len__(self):
                return len(self.encodings["input_ids"])
            def __getitem__(self, idx): #override method to get item at index idx
                return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

        return _ProteinDataset(encodings)

    def train(self, species_batch, sequence_batch, epochs=3):
        """Main fine-tuning loop."""
        dataset = self._make_dataset(species_batch, sequence_batch)
        loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=self.data_collator
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1} completed — avg MLM loss: {avg_loss:.4f}")

    @torch.no_grad()
    def embed(self, species_batch, sequence_batch):
        """Return mean embeddings from the fine-tuned LoRA model."""
        self.model.eval()
        texts = [f"<sp_{s}> {seq}" for s, seq in zip(species_batch, sequence_batch)]
        tokenized = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length
        ).to(self.device)

        outputs = self.model(**tokenized)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()