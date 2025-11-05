from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from src.zootransform.model.species_model import SpeciesAwareESM2

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
        self.loss_fn = nn.MSELoss()  # Align embeddings to frozen ESM2 #TODO - change to masking (loss = crossentropy_loss)

    def train(self, species_batch, sequence_batch, frozen_embeddings=None, epochs=3):
        dataset = ProteinDataset(
            species_batch=[f"<sp_{s}>" for s in species_batch],
            sequence_batch=sequence_batch,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        #total_loss = float("inf")
        epoch_losses = []

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
                    # Self-supervised / L2 regularization on embeddings
                    # loss = embeddings.norm()
                    loss = (embeddings ** 2).mean() #TODO - change to cross-entropy loss with masking

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if len(loader) > 0:
                avg_loss = epoch_loss / len(loader)
            else:
                avg_loss = float("nan")

            # --- Track loss ---
            # total_loss = avg_loss
            if not torch.isnan(torch.tensor(avg_loss)):
                epoch_losses.append(avg_loss)
            print(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

        if len(epoch_losses) > 0:
            final_loss = float(np.mean(epoch_losses))
        else:
            final_loss = float("inf")

        print(f"Returning final average loss across epochs: {final_loss:.4f}")
        return final_loss

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


class LoraFinetunerMLM:
    """
    Fine-tune SpeciesAwareESM2 using LoRA with Masked Language Modeling (MLM)
    """

    def __init__(self, base_model: SpeciesAwareESM2, r=8, alpha=16, dropout=0.05,
                 target_modules=None, lr=1e-4, batch_size=4, mlm_probability=0.15):

        self.device = base_model.device
        self.tokenizer = base_model.tokenizer
        self.max_length = base_model.max_length
        self.batch_size = batch_size
        self.mlm_probability = mlm_probability

        if target_modules is None:
            target_modules = ["k_proj", "q_proj", "v_proj", "embed_tokens"]

        # Wrap base model with LoRA
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            #task_type=TaskType.TOKEN_CLS #TODO - specify MLM task type
        )
        self.model = get_peft_model(base_model.model, lora_cfg).to(self.device)

        # Only optimize LoRA parameters
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

        # MLM collator :
        # automatically adds the labels tensor for MLM:
        # It copies input_ids to labels and randomly masks tokens according to mlm_probability.
        # So after collating, the batch dictionary will have:
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_probability
        )

        self.loss_fn = nn.CrossEntropyLoss()  # standard MLM loss

    def train(self, species_batch: List[str], sequence_batch: List[str], epochs=3):
        # Prepare dataset
        dataset = ProteinDataset(species_batch, sequence_batch, tokenizer=self.tokenizer,
                                 max_length=self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            collate_fn=self.data_collator)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                # batch contains masked input_ids and labels
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                #logits = outputs.last_hidden_state  # shape: [batch, seq_len, vocab_size]
                logits = outputs.logits  # shape: [batch, seq_len, vocab_size]

                # MLM loss expects [batch*seq_len, vocab_size] and labels [batch*seq_len]
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(loader) if len(loader) > 0 else float("nan")
            print(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

    @torch.no_grad()
    # def embed(self, species_batch: List[str], sequence_batch: List[str]):
    #     """
    #     Compute embeddings (mean-pooled) for sequences without masking
    #     """
    #     self.model.eval()
    #     dataset = ProteinDataset(species_batch, sequence_batch, tokenizer=self.tokenizer,
    #                              max_length=self.max_length)
    #     loader = DataLoader(dataset, batch_size=self.batch_size)
    #
    #     all_embeddings = []
    #     for batch in loader:
    #         batch = {k: v.to(self.device) for k, v in batch.items()}
    #         outputs = self.model(**batch)
    #         embeddings = outputs.last_hidden_state.mean(dim=1)
    #         all_embeddings.append(embeddings)
    #
    #     return torch.cat(all_embeddings, dim=0).cpu()
    @torch.no_grad()
    def embed(self, species_batch: List[str], sequence_batch: List[str]):
        """
        Compute embeddings (mean-pooled) for sequences without masking.
        Works with MLM models by using hidden states.
        """
        self.model.eval()
        dataset = ProteinDataset(species_batch, sequence_batch, tokenizer=self.tokenizer,
                                 max_length=self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        all_embeddings = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Request hidden states
            outputs = self.model(**batch, output_hidden_states=True)

            # Last hidden layer embeddings
            last_hidden = outputs.hidden_states[-1]  # shape: [batch, seq_len, hidden_dim]

            # Mean pooling over sequence length
            embeddings = last_hidden.mean(dim=1)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0).cpu()


if __name__ == "__main__":
    data = pd.DataFrame({
        "species": ["human", "mouse", "ecoli", "human", "mouse", "ecoli", "human", "mouse", "ecoli"],
        "sequence": [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            "MKVSAIAKQRQISFVKSHFSRQLRERLGLIEVQ",
            "MKTVYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            "MKVSAIAKQRQISFVKSHFSRQLRERLGLIEVQ",
            "MKTVYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            "MKVSAIAKQRQISFVKSHFSRQLRERLGLIEVQ",
            "MKTVYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
        ]
    })

    from src.zootransform.fine_tuning.fine_tuning import LoraFinetunerMLM  # new MLM version
    from src.zootransform.model.species_model import SpeciesAwareESM2
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Species-aware model
    species_model = SpeciesAwareESM2(species_list=["human", "mouse", "ecoli"])
    species_model.model.to(device)

    species_batch = data["species"].tolist()
    sequence_batch = data["sequence"].tolist()

    finetuner = LoraFinetunerMLM(
        base_model=species_model,
        r=8,
        alpha=16,
        dropout=0.05,
        target_modules=["attention.self.key", "attention.self.value"],  # LoRA targets
        lr=1e-4,
        batch_size=4,
        mlm_probability=0.15  # fraction of tokens to mask
    )

    finetuner.train(
        species_batch=species_batch,
        sequence_batch=sequence_batch,
        epochs=5
    )

    tuned_embeddings = finetuner.embed(species_batch, sequence_batch)
    print("Tuned embeddings shape:", tuned_embeddings.shape)