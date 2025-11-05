# Core libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Transformers and PEFT
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model

# Data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Utilities
import gc
from tqdm.auto import tqdm

# Set style for prettier plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

print("✓ All libraries imported successfully!")


class SpeciesAwareESM2:
    """
    Wrapper around ESM2 model to handle species-specific tokens.
    Generates embeddings for protein sequences with species context.

    Input:
    - model_name: Name of the pretrained model; default is "facebook/esm2_t6_8M_UR50D"
    - device: Computation device (CPU or GPU); default is auto-detected
    - species_list: List of species names to create special tokens for
    - max_length: Maximum sequence length for tokenization; default is 1024

    Attributes:
    - model: Pretrained ESM2 model
    - tokenizer: Corresponding tokenizer with added species tokens
    - species_tokens: List of special tokens for each species
    - device: Computation device (CPU or GPU)
    - max_length: Maximum sequence length for tokenization

    Methods:
    - prepare_inputs(species, sequence): Prepares tokenized inputs with species token
    - embed(species, sequence): Generates embeddings for a given (species, sequence) pair
    - forward(species_batch, sequence_batch): Forward pass for a batch of (species, sequence) pairs
    - visualize_special_tokens(): Visualizes the special tokens in the tokenizer vocabulary

    """

    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", device=None, species_list=None, max_length=1024):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        self.model_name = model_name
        self.max_length = max_length

        print(f"Loading model: {model_name}")
        # self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Default species tokens if not provided
        if species_list is None:
            species_list = ['Arabidopsis thaliana',
                            'Bos taurus',
                            'Escherichia coli',
                            'Homo sapiens',
                            'Mus musculus',
                            'Oryza sativa',
                            'Rattus norvegicus',
                            'Rhodotorula toruloides',
                            'Saccharolobus solfataricus',
                            'Saccharomyces cerevisiae',
                            'Schizosaccharomyces pombe',
                            'Staphylococcus aureus']

        # e.g. "<sp_human>", "<sp_mouse>", "<sp_ecoli>"
        self.species_tokens = [f"<sp_{s}>" for s in species_list]
        print(f"Adding species tokens: {self.species_tokens}")

        # Add as special tokens
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.species_tokens})
        print(f"Added {num_added} new special tokens")

        # Resize embeddings if tokens were added
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)
            print(f"Resized model embeddings to {len(self.tokenizer)} tokens")

            # Ensure lm_head bias is also resized
            if hasattr(self.model, "lm_head") and self.model.lm_head.bias is not None:
                old_bias = self.model.lm_head.bias.data
                new_bias = torch.zeros(self.model.config.vocab_size, device=old_bias.device)
                new_bias[:old_bias.size(0)] = old_bias
                self.model.lm_head.bias = torch.nn.Parameter(new_bias)

        # Mapping from species name to token
        self.species_to_token = {s: f"<sp_{s}>" for s in species_list}

        print("✓ Model and tokenizer ready!")
        print(f"  Hidden size: {self.model.config.hidden_size}")
        print(f"  Number of layers: {self.model.config.num_hidden_layers}")

    def prepare_inputs(self, species, sequence):
        """
        Prepend the species token to the sequence and tokenize it.
        """
        # check that species is valid
        if species not in self.species_to_token:
            raise ValueError(
                f"Unknown species '{species}'. Valid options: {list(self.species_to_token.keys())}")

        species_token = self.species_to_token[species]
        input_text = species_token + sequence

        tokens = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        return tokens

    @torch.no_grad()
    def embed(self, species, sequence):
        """
        Generate embeddings for a given (species, sequence) pair.
        Returns the last hidden state from the model.
        """
        inputs = self.prepare_inputs(species, sequence)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

    def forward(self, species_batch, sequence_batch):
        """
        Forward pass for a batch of (species, sequence) pairs.
        species_batch: list of species names
        sequence_batch: list of sequences
        """
        texts = [
            self.species_to_token[s] + seq
            for s, seq in zip(species_batch, sequence_batch)
        ]
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        return self.model(**tokens)

    def visualize_special_tokens(self):
        vocab = self.tokenizer.get_vocab()

        # Separate special tokens from amino acid tokens
        special_tokens = {k: v for k,
                          v in vocab.items() if '<' in k or '|' in k}
        amino_acid_tokens = {k: v for k, v in vocab.items(
        ) if k not in special_tokens and len(k) == 1}

        fig, (ax2) = plt.subplots(1, 1, figsize=(14, 5))
        special_names = list(special_tokens.keys())
        special_ids = list(special_tokens.values())
        bars = ax2.barh(special_names, special_ids,
                        edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Token ID', fontsize=11)
        ax2.set_ylabel('Special Token', fontsize=11)
        ax2.set_title('Special Tokens', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        for i, (name, val) in enumerate(zip(special_names, special_ids)):
            ax2.text(val + 0.5, i, str(val), va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

        print("\nToken Types:")
        print(
            f"  Amino acids: {len(amino_acid_tokens)} tokens (standard 20 + variants)")
        print(f"  Special tokens: {len(special_tokens)} tokens")
        print(f"  Total vocabulary: {len(vocab)} tokens")

        print("\nSpecial Tokens:")
        for name, token_id in special_tokens.items():
            descriptions = {
                '<pad>': 'Padding token (fills sequences to same length)',
                '<eos>': 'End of sequence marker',
                '<unk>': 'Unknown token (for invalid amino acids)',
                '<cls>': 'Start of sequence marker',
                '<mask>': 'Mask token (for training)',
            }
            desc = descriptions.get(name, 'Special token')
            print(f"  {name:8s} (ID {token_id:2d}): {desc}")

    @torch.no_grad()
    def embed_with_uncertainty(self, species, sequences, n_mc_draws=20):
        """
        Monte Carlo dropout uncertainty estimation.
        Returns mean embedding and uncertainty estimate.

        Inputs:
        - species: species names
        - sequences: protein sequences
        """
        self.model.train()  # keep dropout active!
        embeddings = []

        for _ in range(n_mc_draws):
            emb = self.model.embed(species, sequences)
            emb_mean = emb.mean(dim=1).cpu().numpy()
            embeddings.append(emb_mean)

        embeddings = np.stack(embeddings, axis=0)
        mean_embedding = embeddings.mean(axis=0)
        uncertainty = embeddings.std(axis=0).mean()

        return mean_embedding, uncertainty

    # def check_stability(self):
    #     mean_unc = []
    #     for n in [5, 10, 20, 50, 100]:
    #         _, unc = self.embed_with_uncertainty(species, sequences, n_samples=n)
    #         mean_unc.append(unc)
    #     plt.plot([5, 10, 20, 50, 100], mean_unc, marker='o')
    #     plt.xlabel("Number of MC samples")
    #     plt.ylabel("Estimated uncertainty")
    #     plt.show()
