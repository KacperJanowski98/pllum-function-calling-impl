"""
Utilities for fine-tuning the PLLuM model for function calling.
"""

import json
import logging
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    PeftModel,
    get_peft_model,
)
from unsloth import FastLanguageModel
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check CUDA availability at module import time
if torch.cuda.is_available():
    logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    try:
        # Try to import and use nvidia-ml-py for more detailed GPU info
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(f"GPU memory: Total={info.total / 1e9:.2f}GB, Free={info.free / 1e9:.2f}GB, Used={info.used / 1e9:.2f}GB")
        pynvml.nvmlShutdown()
    except (ImportError, Exception) as e:
        logger.warning(f"Could not get detailed GPU info: {e}")
else:
    logger.warning("CUDA is not available. Training will be very slow on CPU.")


# Custom PyTorch Dataset for function calling
class FunctionCallingDataset(Dataset):
    """Dataset for function calling fine-tuning."""
    
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.num_examples = len(input_ids)
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }


@dataclass
class PLLuMFineTuningConfig:
    """Configuration for fine-tuning PLLuM model."""
    
    # Model configuration
    model_name_or_path: str = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"
    output_dir: str = "models/pllum-function-calling"
    
    # QLoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    max_steps: int = -1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    
    # Dataset parameters
    max_seq_length: int = 1024
    dataset_path: str = "data/translated_dataset.json"
    
    # Tokenization
    padding: str = "max_length"
    pad_to_multiple_of: Optional[int] = 8
    
    # CUDA and device settings
    use_cuda: bool = True  # Default to using CUDA if available
    device_map: str = "auto"  # Let HF decide on device mapping
    

def format_function_calling_prompt(example: Dict) -> str:
    """
    Format a function calling example into a prompt suitable for PLLuM instruction format.
    
    Args:
        example: A dictionary containing 'query', 'tools', and 'answers' fields
        
    Returns:
        Formatted prompt string
    """
    query = example["query"]
    tools = example["tools"]
    answers = example["answers"]
    
    # Handle case where tools and answers are stored as JSON strings
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
            logger.info("Parsed tools from JSON string")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing tools JSON: {e}")
            # Provide a fallback if parsing fails
            tools = [{"name": "error", "description": "Error parsing tools JSON"}]
    
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
            logger.info("Parsed answers from JSON string")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing answers JSON: {e}")
            # Provide a fallback if parsing fails
            answers = [{"name": "error", "arguments": {}}]
    
    # Format tools as a string
    tools_str = json.dumps(tools, indent=2, ensure_ascii=False)
    
    # Format expected answers as a string
    answers_str = json.dumps(answers, indent=2, ensure_ascii=False)
    
    # Create instruction in PLLuM format
    prompt = f"""<|im_start|>user
Poniżej znajduje się zapytanie i lista dostępnych narzędzi.
Proszę wywołać odpowiednie narzędzie, aby odpowiedzieć na zapytanie użytkownika.

Zapytanie: {query}

Dostępne narzędzia:
{tools_str}
<|im_end|>
<|im_start|>assistant
{answers_str}
<|im_end|>"""

    return prompt


def prepare_dataset(
    dataset_path: str, 
    tokenizer, 
    max_length: int = 1024,
    custom_format_func=None
) -> FunctionCallingDataset:
    """
    Prepare the dataset for fine-tuning.
    
    Args:
        dataset_path: Path to the JSON dataset
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        custom_format_func: Optional custom formatting function
        
    Returns:
        A PyTorch Dataset for training
    """
    # Default to format_function_calling_prompt if no custom function is provided
    format_func = custom_format_func or format_function_calling_prompt
    
    # Load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    num_examples = len(data)
    logger.info(f"Loaded {num_examples} examples from {dataset_path}")
    
    # Format the examples
    formatted_examples = [format_func(example) for example in data]
    logger.info(f"Formatted {len(formatted_examples)} examples")
    
    # Tokenize the examples
    tokenized = tokenizer(
        formatted_examples,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Create dataset
    dataset = FunctionCallingDataset(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        tokenized["input_ids"].clone()  # Labels are the same as input_ids for causal LM
    )
    
    logger.info(f"Created dataset with {len(dataset)} examples")
    return dataset


def setup_model_and_tokenizer(config: PLLuMFineTuningConfig):
    """
    Setup the PLLuM model and tokenizer with QLoRA 4-bit quantization using Unsloth.
    
    Args:
        config: Fine-tuning configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Ensure CUDA if required and available
    if config.use_cuda and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        config.use_cuda = False
    
    # Use cuda if available and requested
    device = "cuda" if config.use_cuda and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Log CUDA memory before loading model
    if device == "cuda":
        logger.info(f"CUDA memory before loading model: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    
    # Define quantization config for BitsAndBytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )
    
    # Load PLLuM model and tokenizer using Unsloth's optimized loader
    logger.info(f"Loading model from {config.model_name_or_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name_or_path,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Automatically decide based on GPU availability
        quantization_config=bnb_config if device == "cuda" else None,
        device_map=config.device_map if device == "cuda" else None,
    )
    
    # Log CUDA memory after loading model
    if device == "cuda":
        logger.info(f"CUDA memory after loading model: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    
    # Define target modules for LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Apply LoRA adapters using Unsloth
    # Pass individual parameters instead of a LoraConfig object
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
    )
    
    # Make sure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def train_model(
    model,
    tokenizer,
    train_dataset: FunctionCallingDataset,
    config: PLLuMFineTuningConfig,
) -> PeftModel:
    """
    Train the PLLuM model with the prepared dataset.
    
    Args:
        model: The model to fine-tune
        tokenizer: The tokenizer
        train_dataset: The prepared Dataset for training
        config: Fine-tuning configuration
        
    Returns:
        Fine-tuned model
    """
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=torch.cuda.is_available(),  # Mixed precision training if CUDA is available
        bf16=False,  # Use FP16 instead of BF16
        report_to="tensorboard",
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory if CUDA available
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting model training...")
    
    # Log CUDA memory before training
    if torch.cuda.is_available():
        logger.info(f"CUDA memory before training: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    
    trainer.train()
    
    # Log CUDA memory after training
    if torch.cuda.is_available():
        logger.info(f"CUDA memory after training: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    
    # Save the model
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info(f"Model saved to {config.output_dir}")
    return model


def load_fine_tuned_model(model_path: str, device: str = "auto"):
    """
    Load a fine-tuned PLLuM model with LoRA adapters.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on ("cpu", "cuda", or "auto")
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading fine-tuned model from {model_path} on {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # For the base model, we'll use standard methods instead of Unsloth
    # First load the base model with quantization
    base_model_id = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"  # Base model ID
    
    # Setup BitsAndBytes config for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load the base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config if device == "cuda" else None,
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Now load the LoRA adapter separately
    model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer


def generate_function_call(
    model,
    tokenizer,
    query: str,
    tools: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
):
    """
    Generate a function call using the fine-tuned model.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        query: User query
        tools: Available tools
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Generated function call as a dictionary
    """
    # Handle case where tools might be a string (should be a list of dicts)
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
            logger.info("Parsed tools from JSON string for generation")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing tools JSON for generation: {e}")
            # Provide a fallback if parsing fails
            tools = [{"name": "error", "description": "Error parsing tools JSON"}]
    
    # Format the tools as a JSON string
    tools_str = json.dumps(tools, indent=2, ensure_ascii=False)
    
    # Create prompt
    prompt = f"""<|im_start|>user
Poniżej znajduje się zapytanie i lista dostępnych narzędzi.
Proszę wywołać odpowiednie narzędzie, aby odpowiedzieć na zapytanie użytkownika.

Zapytanie: {query}

Dostępne narzędzia:
{tools_str}
<|im_end|>
<|im_start|>assistant
"""
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the model's device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Extract generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the assistant's response
    assistant_response = generated_text.split("<|im_start|>assistant")[1]
    assistant_response = assistant_response.split("<|im_end|>")[0].strip()
    
    # Try to parse the JSON response
    try:
        function_call = json.loads(assistant_response)
        return function_call
    except json.JSONDecodeError:
        # If parsing fails, return the raw response
        return {"raw_response": assistant_response}


def check_cuda_compatibility():
    """
    Check CUDA compatibility and print detailed information about the setup.
    
    Returns:
        Dictionary with CUDA information
    """
    cuda_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    
    if cuda_info["cuda_available"]:
        # Get details for each CUDA device
        for i in range(cuda_info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            cuda_info["devices"].append({
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1e9,
                "multi_processor_count": props.multi_processor_count,
            })
            
        # Try to get more detailed info with pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            
            cuda_info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
            
            for i in range(cuda_info["device_count"]):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                cuda_info["devices"][i].update({
                    "memory_free_gb": mem_info.free / 1e9,
                    "memory_used_gb": mem_info.used / 1e9,
                    "gpu_utilization": util_info.gpu,
                    "memory_utilization": util_info.memory,
                })
            
            pynvml.nvmlShutdown()
        except (ImportError, Exception) as e:
            cuda_info["pynvml_error"] = str(e)
    
    # Print the CUDA information
    if cuda_info["cuda_available"]:
        logger.info(f"CUDA is available - Version: {cuda_info['cuda_version']}")
        logger.info(f"Found {cuda_info['device_count']} CUDA device(s)")
        
        for i, device in enumerate(cuda_info["devices"]):
            logger.info(f"Device {i}: {device['name']}")
            logger.info(f"  Compute capability: {device['compute_capability']}")
            logger.info(f"  Total memory: {device['total_memory_gb']:.2f} GB")
            
            if "memory_free_gb" in device:
                logger.info(f"  Free memory: {device['memory_free_gb']:.2f} GB")
                logger.info(f"  Used memory: {device['memory_used_gb']:.2f} GB")
                logger.info(f"  GPU utilization: {device['gpu_utilization']}%")
                logger.info(f"  Memory utilization: {device['memory_utilization']}%")
    else:
        logger.warning("CUDA is not available. Training will be very slow on CPU.")
    
    return cuda_info
