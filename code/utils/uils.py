import json
import os
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import huggingface_hub  # type: ignore
import numpy
import torch
import tqdm  # type: ignore
from huggingface_hub import HfFileSystem
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.tokenization_utils import PreTrainedTokenizer  # type: ignore
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast  # type: ignore

from examples.moelora.moelora_config import moeLoRAConfig  # type: ignore

from examples.moelora.moelora import from_pretrained, moeLoRAModel  # type: ignore
from log.logging import reset_logging, get_logger

logger = get_logger(__name__)


def load_model(
        model_name: str,
        device: str,
        dtype: torch.dtype,
        adapters: Optional[Dict[str, str]] = None,
        use_flash_attention_2: bool = False,
        load_moelora: bool = True,
        verbose: bool = False,
        from_safetensors: bool = True,
) -> Tuple[Union[AutoModelForCausalLM, moeLoRAModel], Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Convenience function to load a model with the specified adapters like the X-LoRA config, converting it to moeLoRA if specified.

    Args:
        model_name (`str`):
            Directory or HF model repo ID to load the moeLoRA model from.
        device (`str`):
            Device to load the base model and the moeLoRA model to.
        dtype (`torch.dtype`):
            Datatype for the base model.
        adapters (`list` or `dict`, defaults to None):
            Specify a mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
            Specify the list if the adapters were trainable. Specify this parameter to override use of the trained adapters.
        use_flash_attention_2 (`bool`, *optional*, defaults to False):
            Use FlashAttention v2 for the base model.
        load_moelora (`bool`, *optional*, defaults to True):
            Load the model to moeLoRA.
        verbose (`bool`, *optional*, defaults to False):
            Enable verbose loading.
        from_safetensors (`bool`, *optional*, defaults to True):
            Whether to load the classifier weights from a .pt or .safetensors file.

    Returns:
        Tuple whose elements are respectively:

        model (`AutoModelForCausalLM` or `moeLoRAModel`):
            The model.

        tokenizer (`AutoTokenizer`):
            The tokenizer.
    """
    if not os.path.exists(model_name):
        s = HfFileSystem()
        filenames = [file["name"][len(model_name) + 1:] for file in s.ls(model_name)]  # type: ignore
        
        # Download classifier, conf and get paths
        moelora_classifier = [
            name for name in filenames if "moelora_classifier.safetensors" in name or "moelora_classifier.pt" in name
        ][0]
        moelora_config = [name for name in filenames if "moelora_config.json" in name][0]
        classifier_path = huggingface_hub.hf_hub_download(model_name, moelora_classifier)
        config_path = huggingface_hub.hf_hub_download(model_name, moelora_config)
        
        # The names of the adapters which must be in folders
        adapter_names = [
            file["name"][len(model_name) + 1:]  # type: ignore
            for file in s.ls(model_name)
            if file["type"] == "directory"  # type: ignore
        ]
        if "adapter_config" in adapter_names:
            raise ValueError("Got adapter_config in the adapter names. That should not be there.")
        adapter_paths = {}
        # Effective model ID (path)
        new_model_id = config_path.replace("/moelora_config.json", "")
        if adapters is None:
            subfolders = []
            # Download adapters and confs, taking note of effective subdirs for them and the paths
            for adapter_name in adapter_names:
                adapter_name_path = os.path.join(model_name, adapter_name)
                adapter_filename = [
                    name
                    for name in [file["name"][len(adapter_name_path) + 1:] for file in s.ls(adapter_name_path)]
                    # type: ignore
                    if name.endswith(".safetensors")
                ][0]
                huggingface_hub.hf_hub_download(model_name, adapter_filename, subfolder=adapter_name)
                cfg_filename = [
                    name
                    for name in [file["name"][len(adapter_name_path) + 1:] for file in s.ls(adapter_name_path)]
                    # type: ignore
                    if name == "adapter_config.json"
                ][0]
                huggingface_hub.hf_hub_download(model_name, cfg_filename, subfolder=adapter_name)
                subfolders.append(adapter_name)
                adapter_paths[adapter_name] = os.path.join(model_name, new_model_id)
        else:
            subfolders = None
            adapter_paths = adapters
    else:
        adapter_paths = adapters
        classifier_path = None
        config_path = None
        new_model_id = model_name
        subfolders = None
    
    with open(config_path if config_path is not None else os.path.join(model_name, "moelora_config.json"), "r") as f:
        conf = json.load(f)
        conf["device"] = torch.device(device)
        
        if "adapters" not in conf:
            conf["adapters"] = adapters
        moelora_config = moeLoRAConfig(**conf)
    
    model = AutoModelForCausalLM.from_pretrained(
        moelora_config.base_model_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=dtype,
        use_flash_attention_2=use_flash_attention_2,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        moelora_config.base_model_id,
        trust_remote_code=True,
        device_map=device,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if load_moelora:
        model = from_pretrained(
            load_directory=model_name,
            from_safetensors=from_safetensors,
            model=model,
            verbose=verbose,
            device=device,
            adapters=adapter_paths,
            config_path=config_path,
            classifier_path=classifier_path,
            subfolders=subfolders,
        )
        if verbose:
            logger.info("X-LoRA loaded.")
    
    elif verbose:
        logger.info("No X-LoRA loaded, just base model.")
    
    return model, tokenizer


def load_scalings_log(path: str, verbose: bool = False) -> List[torch.Tensor]:
    """
    Load the scalings log, with awareness to the two types.

    Args:
        path (`str`):
            The path provided to `flush_log_scalings`.
        verbose (`bool`, defaults to `False`)
            Display tqdm.
    """
    
    output: List[torch.Tensor] = []
    if pathlib.Path(f"{path}-mapping.json").exists():
        with open(f"{path}-mapping.json", "r") as f:
            mapping: Dict[str, List[int]] = json.loads(f.read())
        
        mapping_full: Dict[int, torch.Tensor] = {}
        maxindex = -1
        
        if verbose:
            iterator = iter(tqdm.tqdm(mapping.items()))  # type: ignore
        else:
            iterator = iter(mapping.items())
        
        for file, indices in iterator:
            npy_arr = numpy.load(file)
            torch_arr = torch.from_numpy(npy_arr)
            tensors = torch_arr.split(1, dim=0)
            for tensor, index in zip(tensors, indices):
                mapping_full[index] = tensor
                if index > maxindex:
                    maxindex = index
        
        for index in range(maxindex + 1):
            output.append(mapping_full[index])
    
    else:
        npy_arr = numpy.load(path)
        torch_arr = torch.from_numpy(npy_arr)
        output.extend(torch_arr.split(1, dim=0))
    
    return output


def print_model_parameters(model):
    """
        查看模型参数的 requires_grad 情况
        """
    print("Layer Name & Parameters")
    print("----------------------------")
    for name, param in model.named_parameters():
        print(f"{name:50} | Requires_grad: {param.requires_grad}")