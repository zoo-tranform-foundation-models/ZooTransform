# ZooTransform

## Overview
ZooTransform explores how the bias present in the ESM-2 protein language model affects its usefulness when predicting protein fitness across different species.

## Motivation
Recent work suggests that protein language models are biased toward species that are overrepresented in their training data ([preprint](https://www.biorxiv.org/content/10.1101/2024.03.07.584001v1.full.pdf)). Understanding and correcting this bias is key to deploying these models for less-studied organisms.

## Approach
- Fine-tune the ESM-2 model to be explicitly aware of the species by introducing a species-identity token.
- Measure how the added species context changes the correlation between the model’s predictions and deep mutational scanning (DMS) experiments for species that are underrepresented in UniProt.

## Data
We use DMS assays curated by ProteinGym and focus on species other than humans, *E. coli*, *Saccharomyces cerevisiae*, and *Arabidopsis thaliana*.