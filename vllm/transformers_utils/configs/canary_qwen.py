# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for NVIDIA Canary-Qwen (SALM) model.

This config handles the NeMo-style config format used by Canary-Qwen models.
"""

from typing import Any

from transformers import PretrainedConfig, Qwen2Config


class CanaryQwenConfig(PretrainedConfig):
    """Configuration for Canary-Qwen Speech Recognition model.

    This model uses NeMo's SALM (Speech-Augmented Language Model) architecture
    with a FastConformer encoder and Qwen3 LLM backbone.
    """

    model_type = "canary_qwen"

    def __init__(
        self,
        # LLM config
        pretrained_llm: str = "Qwen/Qwen3-1.7B",
        # ASR encoder config
        pretrained_asr: str = "nvidia/canary-1b-flash",
        # Audio placeholder token
        audio_locator_tag: str = "<|audioplaceholder|>",
        # Prompt format
        prompt_format: str = "qwen",
        # Perception module config
        perception: dict[str, Any] | None = None,
        # LoRA config (if any)
        lora: dict[str, Any] | None = None,
        # Parameters to prevent from freezing during training
        prevent_freeze_params: list[str] | None = None,
        # Whether to load pretrained weights for submodules
        pretrained_weights: bool = True,
        # Text config (for LLM)
        text_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.pretrained_llm = pretrained_llm
        self.pretrained_asr = pretrained_asr
        self.audio_locator_tag = audio_locator_tag
        self.prompt_format = prompt_format
        self.perception = perception or {}
        self.lora = lora
        self.prevent_freeze_params = prevent_freeze_params or []
        self.pretrained_weights = pretrained_weights

        # Create text config for the LLM backbone
        if text_config is not None:
            self.text_config = Qwen2Config(**text_config)
        else:
            # Default Qwen3-1.7B config (actual config from Qwen/Qwen3-1.7B)
            self.text_config = Qwen2Config(
                vocab_size=151936,
                hidden_size=2048,
                intermediate_size=6144,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                hidden_act="silu",
                max_position_embeddings=32768,
                rms_norm_eps=1e-6,
                tie_word_embeddings=True,
            )

        # Set hidden size for multimodal projections
        self.hidden_size = self.text_config.hidden_size

        # Audio token index (will be set after tokenizer is loaded)
        self.audio_token_index = None

        # Set architectures for vLLM model detection
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["CanaryQwenForConditionalGeneration"]

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "CanaryQwenConfig":
        """Load config from pretrained model.

        This handles both standard HuggingFace configs and NeMo-style configs.
        """
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        # Handle NeMo-style config
        if "pretrained_llm" in config_dict or "pretrained_asr" in config_dict:
            return cls(**config_dict, **kwargs)

        # Handle standard HF config
        return cls(**config_dict, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        output = super().to_dict()
        if hasattr(self, "text_config") and self.text_config is not None:
            output["text_config"] = self.text_config.to_dict()
        return output

