#!/usr/bin/env python3
"""
Simplified single-cell test script for C2S-Scale model.

This script loads the model and runs inference on a single cell
for quick testing and debugging purposes.
"""

import argparse
import pandas as pd
import scanpy as sc
import cell2sentence as cs
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import logging
import random
import numpy as np
import torch

# Try to import nvidia-ml-py for GPU memory profiling
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Try to import psutil for CPU RAM profiling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize NVML if available
if NVML_AVAILABLE:
    try:
        pynvml.nvmlInit()
        NVML_INITIALIZED = True
    except Exception as e:
        logger.warning(f"Failed to initialize NVML: {e}")
        NVML_INITIALIZED = False
else:
    NVML_INITIALIZED = False
    if torch.cuda.is_available():
        logger.warning("nvidia-ml-py not available. GPU memory profiling will be limited to PyTorch's memory stats.")

# Set seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data(data_path: str):
    """Load single-cell data from h5ad file."""
    logger.info(f"Loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)
    logger.info(f"Loaded data with shape: {adata.shape}")
    
    if 'organism' not in adata.obs.columns:
        adata.obs['organism'] = 'Homo sapiens'
        logger.info("Set default organism to 'Homo sapiens'")
    
    return adata


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_gpu_memory_info():
    """Get detailed GPU memory information using nvidia-ml-py."""
    memory_info = []
    
    if not torch.cuda.is_available():
        return memory_info
    
    num_gpus = torch.cuda.device_count()
    
    # Try to use nvidia-ml-py for detailed memory info
    use_nvml = NVML_INITIALIZED
    if use_nvml:
        try:
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                total_gb = mem_info.total / (1024**3)
                used_gb = mem_info.used / (1024**3)
                free_gb = mem_info.free / (1024**3)
                used_pct = (mem_info.used / mem_info.total) * 100
                
                memory_info.append({
                    'gpu': i,
                    'total_gb': total_gb,
                    'used_gb': used_gb,
                    'free_gb': free_gb,
                    'used_pct': used_pct,
                    'source': 'nvidia-ml-py'
                })
        except Exception as e:
            logger.warning(f"Error getting NVML memory info: {e}")
            # Fall back to PyTorch
            use_nvml = False
    
    # Fall back to PyTorch memory stats if NVML not available
    if not use_nvml:
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(i) / (1024**3)
            free_gb = total_gb - reserved_gb
            used_pct = (reserved_gb / total_gb) * 100
            
            memory_info.append({
                'gpu': i,
                'total_gb': total_gb,
                'used_gb': reserved_gb,
                'allocated_gb': allocated_gb,
                'reserved_gb': reserved_gb,
                'free_gb': free_gb,
                'used_pct': used_pct,
                'source': 'pytorch'
            })
    
    return memory_info


def get_cpu_ram_info():
    """Get CPU RAM information using psutil."""
    if not PSUTIL_AVAILABLE:
        return None
    
    try:
        # System-wide memory info
        mem = psutil.virtual_memory()
        
        # Process-specific memory info
        process = psutil.Process()
        process_mem = process.memory_info()
        
        return {
            'system_total_gb': mem.total / (1024**3),
            'system_used_gb': mem.used / (1024**3),
            'system_available_gb': mem.available / (1024**3),
            'system_used_pct': mem.percent,
            'process_rss_gb': process_mem.rss / (1024**3),  # Resident Set Size
            'process_vms_gb': process_mem.vms / (1024**3),  # Virtual Memory Size
        }
    except Exception as e:
        logger.warning(f"Error getting CPU RAM info: {e}")
        return None


def log_memory(stage: str):
    """Log GPU and CPU memory information at a specific stage."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Memory Status - {stage}")
    logger.info(f"{'='*60}")
    
    # CPU RAM info
    cpu_ram_info = get_cpu_ram_info()
    if cpu_ram_info:
        logger.info("CPU RAM:")
        logger.info(f"  System Total:    {cpu_ram_info['system_total_gb']:.2f} GB")
        logger.info(f"  System Used:     {cpu_ram_info['system_used_gb']:.2f} GB ({cpu_ram_info['system_used_pct']:.1f}%)")
        logger.info(f"  System Available: {cpu_ram_info['system_available_gb']:.2f} GB")
        logger.info(f"  Process RSS:     {cpu_ram_info['process_rss_gb']:.2f} GB")
        logger.info(f"  Process VMS:     {cpu_ram_info['process_vms_gb']:.2f} GB")
    else:
        logger.info("CPU RAM: psutil not available")
    
    # GPU memory info
    if torch.cuda.is_available():
        memory_info = get_gpu_memory_info()
        
        if memory_info:
            logger.info("")
            logger.info("GPU Memory:")
            for info in memory_info:
                if info['source'] == 'nvidia-ml-py':
                    logger.info(f"  GPU {info['gpu']}:")
                    logger.info(f"    Total:  {info['total_gb']:.2f} GB")
                    logger.info(f"    Used:   {info['used_gb']:.2f} GB ({info['used_pct']:.1f}%)")
                    logger.info(f"    Free:   {info['free_gb']:.2f} GB")
                else:
                    logger.info(f"  GPU {info['gpu']}:")
                    logger.info(f"    Total:    {info['total_gb']:.2f} GB")
                    logger.info(f"    Reserved: {info['reserved_gb']:.2f} GB ({info['used_pct']:.1f}%)")
                    logger.info(f"    Allocated: {info['allocated_gb']:.2f} GB")
                    logger.info(f"    Free:      {info['free_gb']:.2f} GB")
        else:
            logger.info("GPU Memory: No GPUs available")
    else:
        logger.info("GPU Memory: CUDA not available")
    
    logger.info(f"{'='*60}\n")


def load_model(model_id: str, cache_dir: str = None, multi_gpu: bool = False):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    
    if multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Multi-GPU mode: distributing model across {num_gpus} GPUs")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        device = "cuda:0"
        if hasattr(model, 'hf_device_map'):
            logger.info(f"Device map: {model.hf_device_map}")
    else:
        device = get_device()
        logger.info(f"Single device mode: {device}")
        
        # Determine dtype based on device
        # MPS supports float16 but some operations may need float32
        if device == "cuda":
            dtype = torch.float16
        elif device == "mps":
            # Use float32 for better MPS compatibility
            dtype = torch.float32
        else:
            dtype = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        ).to(device)
    
    model.eval()
    logger.info(f"Model loaded successfully on {device}")
    return model, tokenizer, device


def get_first_cell_sentence(adata, annotation_key: str):
    """Convert first cell to cell sentence format."""
    logger.info(f"Converting first cell to sentence format")
    
    adata_copy = adata.copy()
    
    if "counts" not in adata.layers:
        raise ValueError("Counts layer not found. C2S-Scale needs raw counts.")
    
    adata_copy.X = adata.layers["counts"]
    sc.pp.normalize_total(adata_copy)
    sc.pp.log1p(adata_copy, base=10)
    adata_copy.var_names = adata_copy.var["gene_names"]
    adata_copy.obs['cell_type'] = adata_copy.obs[annotation_key]
    
    if 'organism' not in adata_copy.obs.columns:
        adata_copy.obs['organism'] = 'Homo sapiens'
    
    # Convert to arrow dataset
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata_copy, 
        random_state=SEED, 
        sentence_delimiter=' ',
        label_col_names=['organism', 'cell_type']
    )
    
    # Get first cell
    first_cell = arrow_ds[0]
    cell_sentence = first_cell["cell_sentence"]
    organism = first_cell.get("organism", "Homo sapiens")
    true_label = first_cell.get("cell_type", "unknown")
    
    logger.info(f"First cell true label: {true_label}")
    logger.info(f"Cell sentence length: {len(cell_sentence.split())} genes")
    
    return cell_sentence, organism, true_label


def format_prompt(cell_sentence: str, organism: str, n_genes: int) -> str:
    """Format the prompt for a single cell."""
    # Truncate to top N genes
    genes = cell_sentence.split()[:n_genes]
    truncated_sentence = ' '.join(genes)
    
    prompt = f"""The following is a list of {n_genes} gene names ordered by descending expression level in a {organism} cell. Your task is to give the cell type which this cell belongs to based on its gene expression.
Cell sentence: {truncated_sentence}.
The cell type corresponding to these genes is:"""
    
    return prompt


def run_inference(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 20):
    """Run inference on a single prompt."""
    logger.info("Running inference...")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    logger.info(f"Input token count: {inputs['input_ids'].shape[1]}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response


def extract_prediction(response: str) -> str:
    """Extract the predicted cell type from model response."""
    pred = response.split("The cell type corresponding to these genes is:")[-1].strip()
    pred = pred.replace('<ctrl100>', '').replace('<eos>', '').replace('<pad>', '')
    pred = pred.strip().rstrip('.')
    return pred


def main():
    parser = argparse.ArgumentParser(description="Test C2S-Scale on a single cell")
    parser.add_argument("--input", "-i", required=True, help="Input h5ad file path")
    parser.add_argument("--model", "-m", default="vandijklab/C2S-Scale-Gemma-2-27B",
                       help="HuggingFace model ID")
    parser.add_argument("--annotation_key", default="AIFI_L2",
                       help="Key in adata.obs containing true labels")
    parser.add_argument("--n_genes", type=int, default=200,
                       help="Number of genes to use")
    parser.add_argument("--cache_dir", default=None, help="Model cache directory")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                       help="Maximum new tokens to generate")
    parser.add_argument("--multi_gpu", action="store_true",
                       help="Enable multi-GPU mode")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Log device info
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"         Memory: {mem:.1f} GB")
    elif torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available")
    else:
        logger.info("No GPU available, will use CPU")
    
    # Log initial GPU memory state
    log_memory("Initial State (Before Data Loading)")
    
    # Load data
    adata = load_data(args.input)
    
    # Log GPU memory after data loading
    log_memory("After Data Loading (Before Model Loading)")
    
    # Load model
    model, tokenizer, device = load_model(args.model, args.cache_dir, args.multi_gpu)
    
    # Log GPU memory after model loading
    log_memory("After Model Loading")
    
    # Get first cell sentence
    cell_sentence, organism, true_label = get_first_cell_sentence(adata, args.annotation_key)
    
    # Format prompt
    prompt = format_prompt(cell_sentence, organism, args.n_genes)
    logger.info(f"\n{'='*60}\nPROMPT:\n{'='*60}\n{prompt}\n{'='*60}")
    
    # Log GPU memory before inference
    log_memory("Before Inference")
    
    # Run inference
    response = run_inference(model, tokenizer, prompt, device, args.max_new_tokens)
    
    # Log GPU memory after inference
    log_memory("After Inference")
    
    # Extract prediction
    prediction = extract_prediction(response)
    
    # Print results
    logger.info(f"\n{'='*60}\nRESULTS:\n{'='*60}")
    logger.info(f"True label:      {true_label}")
    logger.info(f"Predicted label: {prediction}")
    logger.info(f"Match: {'✓' if true_label.lower() == prediction.lower() else '✗'}")
    logger.info(f"\nFull response:\n{response}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

