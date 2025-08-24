import json
import os
from typing import Dict, Optional, Union

import torch
import tqdm  # type: ignore
from peft.peft_model import PeftModel
from peft.tuners import lora
from safetensors.torch import load_model, load_file  # type: ignore
from transformers import PreTrainedModel

from classifier import InhibitorFlagPayload, moeLoRAClassifier
from config import moeLoRAConfig
from insertion import (
    BaseTunerWrapper,
    PeftModelWrapper,
    moeLoRAConv2dLayer,
    moeLoRAEmbeddingLayer,
    moeLoRALinearLayer,
)
from log.logging import reset_logging, get_logger

logger = get_logger(__name__)


class moeLoRAModel(PeftModel, PeftModelWrapper):
    def __new__(cls):
        raise RuntimeError(
            "moeLoRAModel is a non instantiatable type and can only be created through `add_moelora_to_model`."
        )


def convert_layers_to_moelora(
        base: PeftModel,
        verbose: bool,
        config: moeLoRAConfig,
) -> int:
    """
    Returns the number of swapped layers.
    """
    total_swapped = 0
    for module in base.modules():
        if isinstance(module, lora.Linear):
            new_layer: Union[moeLoRALinearLayer, moeLoRAEmbeddingLayer, moeLoRAConv2dLayer] = moeLoRALinearLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped = total_swapped + 1
        elif isinstance(module, lora.Embedding):
            new_layer = moeLoRAEmbeddingLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped = total_swapped + 1
        elif isinstance(module, lora.Conv2d):
            new_layer = moeLoRAConv2dLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped = total_swapped + 1
    
    if verbose:
        logger.info(
            f"LoRA -> moeLoRA complete: Swapped {total_swapped} LoRA layers (out of {len(list(base.modules()))} modules)."
        )
    
    return total_swapped


def add_moelora_to_model(
        model: PreTrainedModel,
        moelora_config: moeLoRAConfig,
        verbose: bool = False,
        **kwargs,
) -> moeLoRAModel:
    """
    This method converts all LoRA adapters to moeLoRA layers, and it is one of the intended entrypoints
    for use of moeLoRA. All LoRA adapters will be frozen, and the moeLoRAClassifier is initialized.

    Args:
        model (`PreTrainedModel`):
            The model to add the LoRA adapters to. It may be modified in place. If applicable, `use_cache` must be False.
        verbose (`bool`, defaults to `False`):
            Display tqdm, total swapping count.
    Returns:
        model (`moeLoRAModel`):
            The new model.
    """
    
    if hasattr(model.config, "use_cache"):
        assert not model.config.use_cache, "`use_cache` must be False"
    
    use_trainable_adapters = moelora_config.use_trainable_adapters
    subfolders_in_kwargs = "subfolders" in kwargs and kwargs["subfolders"] is not None
    if verbose:
        if subfolders_in_kwargs:
            adapters_items = iter(tqdm.tqdm(zip(moelora_config.adapters.items(), kwargs["subfolders"])))
        else:
            adapters_items = iter(tqdm.tqdm(moelora_config.adapters.items()))
    else:
        if subfolders_in_kwargs:
            adapters_items = iter(zip(moelora_config.adapters.items(), kwargs["subfolders"]))
        else:
            adapters_items = iter(moelora_config.adapters.items())
    first_item = next(adapters_items)
    if subfolders_in_kwargs:
        model_peft = PeftModel.from_pretrained(
            model, first_item[0][1], first_item[0][0], is_trainable=use_trainable_adapters, subfolder=first_item[1]
        )
    else:
        model_peft = PeftModel.from_pretrained(
            model,
            first_item[1],
            first_item[0],  # type: ignore
            is_trainable=use_trainable_adapters,  # type: ignore
            # torch_device='cpu',
        )
    
    if subfolders_in_kwargs:
        for (adapter_name, model_id), subfolder in adapters_items:
            model_peft.load_adapter(model_id, adapter_name, is_trainable=use_trainable_adapters, subfolder=subfolder)
    else:
        for adapter_name, model_id in adapters_items:
            model_peft.load_adapter(model_id, adapter_name, is_trainable=use_trainable_adapters,
                                    # torch_device='cpu'
                                    )  # type: ignore
    
    model_peft.base_model.set_adapter(list(moelora_config.adapters.keys()))
    
    def hook(module, *args, **kwargs) -> None:
        
        args_real = args[0]
        kwargs_real: dict = args[1]
        kwargs_real.update(kwargs)
        # print("+++++++++++++++++++++++hook+++++++++++++++++++++++")
        moelora_classifier: moeLoRAClassifier = model_peft.internal_moelora_classifier  # type: ignore
        
        if "_moelora_classifier_inhibitor_flag" in kwargs_real:
            payload: InhibitorFlagPayload = kwargs_real["_moelora_classifier_inhibitor_flag"]
            
            del kwargs_real["_moelora_classifier_inhibitor_flag"]
            
            model_peft.internal_moelora_scalings = torch.full(  # type: ignore
                (payload.batch_size, payload.seq_len, moelora_classifier.n_layers, moelora_classifier.n_classes),
                payload.override_scaling_pass_value,
            )
            
            return
        
        moelora_scalings = moelora_classifier.forward(
            *args_real,
            **kwargs_real,
        )
        # Set the scalings
        model_peft.internal_moelora_scalings = moelora_scalings
    
    model.register_forward_pre_hook(hook, with_kwargs=True, prepend=True)
    
    model_peft.base_model.eval()
    if not use_trainable_adapters:
        total_frozen = 0
        for name, param in model_peft.base_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False
                total_frozen += 1
        if verbose:
            logger.info(f"Froze {total_frozen} adapters.")
    
    assert isinstance(model_peft.base_model, lora.LoraModel)
    
    total_swapped = convert_layers_to_moelora(
        model_peft,
        verbose,
        moelora_config,
    )
    
    n_classes = len(moelora_config.adapters)
    moelora_classifier = moeLoRAClassifier(model_peft, moelora_config, n_classes, total_swapped)
    
    # Setup the internal state
    base_model_wrapper = BaseTunerWrapper(model_peft.base_model, moelora_classifier)
    model_peft.base_model.forward = base_model_wrapper.forward  # type: ignore[method-assign]
    peft_model_wrapper = PeftModelWrapper(
        model_peft,
        model_peft.save_pretrained,
        moelora_config,
        model_peft.get_nb_trainable_parameters,
        model_peft.generate,
    )
    model_peft.save_pretrained = peft_model_wrapper.save_pretrained  # type: ignore[method-assign]
    model_peft.generate = peft_model_wrapper.generate  # type: ignore
    
    assert not hasattr(model_peft, "set_use_trainable_adapters")
    model_peft.set_use_trainable_adapters = peft_model_wrapper.set_use_trainable_adapters  # type: ignore
    
    assert not hasattr(model_peft, "print_scalings_predictions")
    model_peft.print_scalings_predictions = peft_model_wrapper.print_scalings_predictions  # type: ignore
    
    assert not hasattr(model_peft, "enable_scalings_logging")
    model_peft.enable_scalings_logging = peft_model_wrapper.enable_scalings_logging  # type: ignore
    
    assert not hasattr(model_peft, "disable_scalings_logging")
    model_peft.disable_scalings_logging = peft_model_wrapper.disable_scalings_logging  # type: ignore
    
    assert not hasattr(model_peft, "flush_log_scalings")
    model_peft.flush_log_scalings = peft_model_wrapper.flush_log_scalings  # type: ignore
    
    assert not hasattr(model_peft, "get_scalings_log")
    model_peft.get_scalings_log = peft_model_wrapper.get_scalings_log  # type: ignore
    
    assert not hasattr(model_peft, "set_scaling_pass_value")
    model_peft.set_scaling_pass_value = peft_model_wrapper.set_scaling_pass_value  # type: ignore
    
    assert not hasattr(model_peft, "set_global_scaling_weight")
    model_peft.set_global_scaling_weight = peft_model_wrapper.set_global_scaling_weight  # type: ignore
    
    assert not hasattr(model_peft, "get_latest_scalings")
    model_peft.get_latest_scalings = peft_model_wrapper.get_latest_scalings  # type: ignore
    
    assert not hasattr(model_peft, "get_global_scaling_weight")
    model_peft.get_global_scaling_weight = peft_model_wrapper.get_global_scaling_weight  # type: ignore
    
    assert not hasattr(model_peft, "set_topk_lora")
    model_peft.set_topk_lora = peft_model_wrapper.set_topk_lora  # type: ignore
    
    assert not hasattr(model_peft, "get_topk_lora")
    model_peft.get_topk_lora = peft_model_wrapper.get_topk_lora  # type: ignore
    
    assert not hasattr(model_peft, "clear_scalings_log")
    model_peft.clear_scalings_log = peft_model_wrapper.clear_scalings_log  # type: ignore
    
    model_peft.get_nb_trainable_parameters = peft_model_wrapper.get_nb_trainable_parameters  # type: ignore
    
    model_peft.print_trainable_parameters = peft_model_wrapper.print_trainable_parameters  # type: ignore
    
    # Setup the model internal state
    assert not hasattr(model_peft, "internal_moelora_classifier")
    model_peft.internal_moelora_classifier = moelora_classifier
    
    assert not hasattr(model_peft, "internal_moelora_scalings")
    model_peft.internal_moelora_scalings = None  # type: ignore
    
    return model_peft  # type: ignore


def add_moelora_to_model_for_eval(
        model: PreTrainedModel,
        moelora_config: moeLoRAConfig,
        verbose: bool = False,
        **kwargs,
) -> (PeftModel, moeLoRAClassifier):
    """
    This method converts all LoRA adapters to moeLoRA layers, and it is one of the intended entrypoints
    for use of moeLoRA. All LoRA adapters will be frozen, and the moeLoRAClassifier is initialized.
    Do not perform training, but evaluate and directly set the classifier

    Args:
        model (`PreTrainedModel`):
            The model to add the LoRA adapters to. It may be modified in place. If applicable, `use_cache` must be False.
        verbose (`bool`, defaults to `False`):
            Display tqdm, total swapping count.
    Returns:
        model (`moeLoRAModel`):
            The new model.
    """
    
    if hasattr(model.config, "use_cache"):
        assert not model.config.use_cache, "`use_cache` must be False"
    if verbose:
        adapters_items = iter(tqdm.tqdm(moelora_config.adapters.items()))
    else:
        adapters_items = iter(moelora_config.adapters.items())
    first_item = next(adapters_items)
    
    model_peft = PeftModel.from_pretrained(
        model,
        first_item[1],
        first_item[0],  # type: ignore
        is_trainable=False,  # type: ignore
        # torch_device='cpu',
    )
    
    for adapter_name, model_id in adapters_items:
        model_peft.load_adapter(model_id, adapter_name, is_trainable=False,
                                # torch_device='cpu'
                                )  # type: ignore
    
    model_peft.base_model.set_adapter(list(moelora_config.adapters.keys()))
    
    total_swapped = convert_layers_to_moelora(
        model_peft,
        verbose,
        moelora_config,
    )
    
    n_classes = len(moelora_config.adapters)
    moelora_classifier = moeLoRAClassifier(model_peft, moelora_config, n_classes, total_swapped)
    
    assert not hasattr(model_peft, "internal_moelora_scalings")
    model_peft.internal_moelora_scalings = None  # type: ignore
    
    return model_peft, moelora_classifier  # type: ignore


def from_pretrained(
        model: PreTrainedModel,
        moelora_config: moeLoRAConfig,
        device: str,
        adapters: Optional[Dict[str, str]] = None,
        verbose: bool = False,
        from_safetensors: bool = True,
        **kwargs,
) -> (PeftModel, moeLoRAClassifier):
    """
    Loads a pretrained classifier and potentially adapters from the specified folder while initializing the model. This is the counterpart to `save_pretrained`.
    If trainable adapters was enabled, those saved adapters will be loaded.

    This method is very similar to `add_moelora_to_model`: it converts all LoRA adapters to moeLoRA layers, and it is one of
    the intended entrypoints for use of moeLoRA. All LoRA adapters will be frozen, and the moeLoRAClassifier is initialized.

    Args:
        load_directory (`str`):
            The directory or HF model repo ID to load the weights from.
        model (`PreTrainedModel`):
            The model to add the LoRA adapters to. It may be modified in place. If applicable, `use_cache` must be False.
        device (`str`):
            Device of the model, used to load the classifier.
        adapters (`dict`, *optional*, defaults to None):
            Specify a mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
            Specify the list if the adapters were trainable. Specify this parameter to override use of the trained adapters.
        verbose (`bool`, defaults to `False`):
            Display tqdm, total swapping count.
        from_safetensors (`bool`, *optional*, defaults to True):
            Whether to load the classifier weights from a .pt or .safetensors file.

    Returns:
        model (`moeLoRAModel`):
            The new model.
    """
    
    if adapters is None or moelora_config.use_trainable_adapters:
        adapters_real = moelora_config.adapters
    else:
        assert isinstance(adapters, dict)
        adapters_real = adapters
    moelora_config.adapters = adapters_real
    model_peft, classifier = add_moelora_to_model_for_eval(model, moelora_config, verbose, **kwargs)
    
    classifier_path = moelora_config.classifier_path
    
    if from_safetensors:
        state_dict = load_file(classifier_path)
        classifier.to(device)
    else:
        state_dict = torch.load(
            classifier_path if classifier_path is not None else os.path.join(moelora_config.classifier_path,
                                                                             "moelora_classifier.pt")
        )
    classifier.load_state_dict(state_dict)  # type: ignore
    
    return model_peft, classifier


from transformers.modeling_outputs import (  # type: ignore
    ModelOutput,
)


def use_classifier_obtain_weight(model_peft: PeftModel, classifier: moeLoRAClassifier,
                                 input_ids, mean_scalings_dict, **kwargs):
    kwargs["output_hidden_states"] = True
    kwargs["return_dict"] = True
    
    classifier.set_override_scaling_pass_value(None)
    
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    # scalings = [batch_size, seq_len, n_layers, n_classes]
    model_peft.internal_moelora_scalings = torch.full(  # type: ignore
        (batch_size, seq_len, classifier.n_layers, classifier.n_classes),
        classifier.override_scaling_pass_value,
    )

    result: ModelOutput = model_peft.forward(
        input_ids=input_ids,
        **kwargs,
    )
    hidden_states = result.hidden_states  # type: ignore

    assert hidden_states is not None
    hidden_state = hidden_states[-1]  # Get the last hidden state

    ### Classifier run
    # hidden_state=[batch_size, seq_len, hidden_size]
    for layer in classifier.inner:
        hidden_state = layer.forward(hidden_state)

    logits = classifier.last.forward(hidden_state)

    ### Repeat to make layerwise scalings if the classifier layer does not
    if not classifier.config.layerwise_scalings:
        logits = logits.unsqueeze(2)
        logits = logits.expand(-1, -1, classifier.n_layers, -1)

    ### Classifier run
    scalings = logits.reshape(batch_size, seq_len, classifier.n_layers, classifier.n_classes)
    # scalings = [batch_size, seq_len, n_layers, n_classes]
    # logger.info(
    #     f"batch_size: {batch_size}, seq_len: {seq_len}, n_layers: {classifier.n_layers}, n_classes: {classifier.n_classes}")
    if classifier.config.enable_softmax:
        scalings = classifier.softmax(scalings)

    if classifier.n_predictions_lifetime > 0:
        logger.info(f"Scaling predictions: {scalings}")
        classifier.n_predictions_lifetime = classifier.n_predictions_lifetime - 1

    if classifier.scalings_logging:
        classifier.log_scalings.append(scalings)
    
    mean_scalings = scalings.mean(dim=[0, 1, 2])
    
    # logger.info(f"use_classifier_obtain_weight-----mean_scalings: {mean_scalings}")
    mean_scalings_dict["mean_scalings_list"].append(mean_scalings.tolist())
    # Set the scalings
    model_peft.internal_moelora_scalings = scalings
