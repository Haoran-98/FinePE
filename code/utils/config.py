import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class moeLoRAConfig:
    
    model_type = "moelora"
    
    hidden_size: int
    base_model_id: str
    device: torch.device
    adapters: Dict[str, str]
    classifier_path: str
    global_scaling_weight: float
    enable_softmax: bool = True
    enable_softmax_topk: bool = False
    layerwise_scalings: bool = True  # 是否开启层级门控权重
    moelora_depth: int = 2  # 门控结构隐藏层深度
    moelora_size: int = 256  # 隐藏层大小
    enable_relu_and_dropout: bool = True
    use_bias: bool = True
    moelora_dropout_p: float = 0.2
    use_trainable_adapters: bool = False
    softmax_temperature: float = 0.1
    top_k_lora: Optional[int] = None
    scaling_pass_value: float = 1.0
    global_scaling_weight: float = 1.0
    loss_balance_scale = 2.0
    use_evaluate = True  # 是否评估
    classifier_path = ""
    adapters_scales_dict = {0: 1.0,
                            1: 1.0,
                            2: 1.0,
                            3: 1.0,
                            4: 1.0,
                            5: 1.0}
    
    def __post_init__(self):
        if self.enable_softmax_topk and self.top_k_lora is None:
            warnings.warn("`enable_softmax_topk` enabled `top_k_lora` is not set")
        
        if self.enable_softmax_topk and self.enable_softmax:
            warnings.warn(
                "`enable_softmax_topk` and `enable_softmax` are both enabled. This will result in worse performance."
            )
        
        if self.top_k_lora is not None and self.top_k_lora < 1:
            warnings.warn("`top_k_lora` value must be at least 1.")
