from peft import LoraConfig, get_peft_model


def print_trainable_parameters(model):
    """Count and display trainable vs total parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_pct = 100 * trainable_params / all_param
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Total params: {all_param:,}")
    print(f"  Trainable: {trainable_pct:.2f}%")


def fine_tune_model(species_model):
    lora_config = LoraConfig(
        r=8,                     # rank of the update matrices
        lora_alpha=32,            # scaling factor
        target_modules=["q_proj", "v_proj"],  # attention projection layers
        lora_dropout=0.1,         # dropout for LoRA layers
        bias="none",              # keep original bias
        task_type="SEQ_CLS"       # depends on task: "SEQ_CLS" / "CAUSAL_LM" etc.
    )

    # Create LoRA-wrapped model
    species_model.ft_model = get_peft_model(species_model.model, lora_config)
    print("✓ LoRA adapters added to model")

    for name, param in species_model.model.model.named_parameters():
        if param.requires_grad:
            print(name)  # should only show LoRA layers

    print("📊 Model Parameters Comparison:")
    print("-" * 60)
    print("Before LoRA (original model):")
    print_trainable_parameters(species_model.model)
    print("\nAfter LoRA (adapted model):")
    print_trainable_parameters(species_model.ft_model)
    print("-" * 60)
    print("\n✓ LoRA adapters applied successfully!")
    # optimizer = AdamW(species_model.model.parameters(), lr=1e-4)

