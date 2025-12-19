# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only NVIDIA Canary-Qwen speech recognition model.

This model is a Speech-Augmented Language Model (SALM) that combines:
- FastConformer encoder from nvidia/canary-1b-flash
- Qwen3-1.7B LLM backbone with LoRA adapters
- Audio perception module for speech-to-text

Reference: https://huggingface.co/nvidia/canary-qwen-2.5b

Requires NeMo toolkit to be installed:
    pip install nemo_toolkit[asr]
"""

from collections.abc import Iterable, Mapping
from typing import Annotated, Literal

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import AutoWeightsLoader, WeightsMapper, init_vllm_registered_model, maybe_prefix

logger = init_logger(__name__)

# Default audio placeholder token used by Canary-Qwen
_AUDIO_PLACEHOLDER_TOKEN = "<|audioplaceholder|>"

# Sampling rate expected by the model
_SAMPLING_RATE = 16000

# Maximum audio duration in seconds (from training)
_MAX_AUDIO_DURATION_S = 40.0


class CanaryQwenAudioInputs(TensorSchema):
    """
    Audio input features for Canary-Qwen model.

    Dimensions:
        - b: Batch size
        - t: Time samples in audio waveform
    """

    type: Literal["audio_features"]
    audio_signal: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("b", "t")]
    """Raw audio waveform signal."""

    audio_signal_length: Annotated[torch.Tensor, TensorShape("b")]
    """Length of each audio signal in samples."""

    audio_embed_sizes: Annotated[list[int], TensorShape("b")]
    """Number of audio embedding tokens for each audio in the batch."""


class CanaryQwenAudioEmbedInputs(TensorSchema):
    """Pre-computed audio embeddings input."""

    type: Literal["audio_embeds"]
    audio_embeds: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("b", "t", "d")]
    """Pre-computed audio embeddings."""


class CanaryQwenProcessingInfo(BaseProcessingInfo):
    """Processing information for Canary-Qwen model."""

    _audio_token_id: int | None = None

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        # Ensure the audio placeholder token is in the tokenizer early
        # This is needed because profiling happens before model init
        self._ensure_audio_token_in_tokenizer()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Canary-Qwen supports one audio per prompt
        return {"audio": 1}

    def get_max_audio_tokens(self) -> int:
        """Get maximum number of audio tokens based on max audio duration."""
        # This is an estimate based on typical FastConformer output
        # FastConformer has ~4x subsampling, and modality adapter may add more
        # For 40s audio at 16kHz: 640000 samples -> ~160000 frames after mel
        # After encoder subsampling (~4x): ~40000 -> after adapter: ~10000
        # Conservative estimate for profiling
        return 10000

    def get_max_audio_len(self) -> int:
        """Get maximum audio length in samples."""
        return int(_MAX_AUDIO_DURATION_S * _SAMPLING_RATE)

    def _ensure_audio_token_in_tokenizer(self) -> int:
        """Ensure the audio placeholder token is in the tokenizer.

        Returns the token ID.
        """
        tokenizer = self.get_tokenizer()

        # Check if token already exists using convert_tokens_to_ids
        # (get_vocab() may return stale data due to caching)
        token_id = tokenizer.convert_tokens_to_ids(_AUDIO_PLACEHOLDER_TOKEN)
        if token_id is not None:
            return token_id

        # Token not in vocab - add it as a special token
        # This modifies the tokenizer in place
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [_AUDIO_PLACEHOLDER_TOKEN]}
        )
        if num_added > 0:
            token_id = tokenizer.convert_tokens_to_ids(_AUDIO_PLACEHOLDER_TOKEN)
            logger.info(
                "Added audio placeholder token '%s' to tokenizer (id=%d)",
                _AUDIO_PLACEHOLDER_TOKEN,
                token_id,
            )

        token_id = tokenizer.convert_tokens_to_ids(_AUDIO_PLACEHOLDER_TOKEN)
        if token_id is None:
            raise ValueError(
                f"Failed to add audio placeholder token '{_AUDIO_PLACEHOLDER_TOKEN}' "
                "to tokenizer"
            )
        return token_id

    def get_audio_token_id(self) -> int:
        """Get the audio placeholder token ID."""
        if self._audio_token_id is None:
            self._audio_token_id = self._ensure_audio_token_in_tokenizer()
        return self._audio_token_id


class CanaryQwenMultiModalProcessor(
    BaseMultiModalProcessor[CanaryQwenProcessingInfo]
):
    """Multi-modal processor for Canary-Qwen model."""

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=_SAMPLING_RATE)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_signal=MultiModalFieldConfig.batched("audio"),
            audio_signal_length=MultiModalFieldConfig.batched("audio"),
            audio_embed_sizes=MultiModalFieldConfig.batched("audio"),
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: "MultiModalDataItems",
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """Our processor does NOT apply updates - we just tokenize."""
        # Return False so that the framework applies the PromptReplacement
        # we return from _get_prompt_updates
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        # Get audio token ID - the tokenizer must have <|audioplaceholder|> as a token
        audio_token_id = self.info.get_audio_token_id()

        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio = audios.get(item_idx)
            audio_length = audio.shape[-1]
            # Estimate number of audio tokens based on audio length
            num_audio_tokens = self._estimate_audio_tokens(audio_length)
            # Return token ID sequence
            return [audio_token_id] * num_audio_tokens

        # Use token ID-based matching - requires <|audioplaceholder|> in tokenizer
        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_replacement,
            )
        ]

    def _estimate_audio_tokens(self, audio_length_samples: int) -> int:
        """Estimate the number of audio tokens for a given audio length.

        This matches NeMo's AudioPerceptionModule token calculation.
        For Canary-Qwen with FastConformer (dw_striding):
        - n_fft = 512
        - hop_length = 160 (window_stride=0.01 * sample_rate=16000)
        - subsampling: 3 conv layers with stride=2 each (total 8x)
        - modality_adapter = identity (no additional subsampling)
        """
        seq_len = torch.as_tensor(audio_length_samples)

        # Feature extractor (STFT/mel spectrogram) settings
        n_fft = 512
        hop_length = 160
        stft_pad_amount = n_fft // 2  # center=True default

        # Convolutional subsampling frontend settings (dw_striding)
        left_padding = 1
        right_padding = 1
        kernel_size = 3
        stride = 2
        repeat_num = 3  # 3 conv layers -> 2^3 = 8x subsampling

        # Step 1: Estimate filterbank/mel spectrogram length
        pad_amount = stft_pad_amount * 2
        fbank_len = torch.floor_divide(
            seq_len + pad_amount - n_fft, hop_length
        )

        # Step 2: Estimate conv subsampling length
        # Formula: floor((length + add_pad) / stride) + 1, repeated
        add_pad = left_padding + right_padding - kernel_size
        lengths = fbank_len.float()
        for _ in range(repeat_num):
            lengths = torch.floor((lengths + add_pad) / stride) + 1.0

        return int(lengths.long().item())

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process the input prompt and audio data."""
        # Ensure the audio token is in the tokenizer before encoding
        self.info.get_audio_token_id()

        tokenizer = self.info.get_tokenizer()
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        # Tokenize the prompt
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

        result = BatchFeature(
            dict(input_ids=[prompt_ids]),
            tensor_type="pt",
        )

        if audios:
            # Process audio data
            audio_list = []
            audio_lengths = []
            audio_embed_sizes = []

            for audio in audios:
                if isinstance(audio, torch.Tensor):
                    audio_tensor = audio
                else:
                    audio_tensor = torch.as_tensor(audio)

                # Ensure 1D audio
                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.squeeze()

                audio_list.append(audio_tensor)
                audio_lengths.append(audio_tensor.shape[-1])
                audio_embed_sizes.append(
                    self._estimate_audio_tokens(audio_tensor.shape[-1])
                )

            result["audio_signal"] = audio_list
            result["audio_signal_length"] = torch.tensor(audio_lengths)
            result["audio_embed_sizes"] = torch.tensor(audio_embed_sizes)

        return result


class CanaryQwenDummyInputsBuilder(
    BaseDummyInputsBuilder[CanaryQwenProcessingInfo]
):
    """Dummy inputs builder for profiling Canary-Qwen model."""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_len(),
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return _AUDIO_PLACEHOLDER_TOKEN * num_audios


def _load_nemo_perception_module(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Load NeMo's AudioPerceptionModule.

    This function lazily imports NeMo to avoid dependency issues.
    """
    try:
        from nemo.collections.speechlm2.modules import AudioPerceptionModule
        from omegaconf import DictConfig
    except ImportError as e:
        raise ImportError(
            "NeMo is required for Canary-Qwen model. "
            "Install it with: pip install nemo_toolkit[asr]"
        ) from e

    perception_config = DictConfig(config)
    perception = AudioPerceptionModule(perception_config)
    perception = perception.to(device=device, dtype=dtype)
    perception.eval()

    return perception


@MULTIMODAL_REGISTRY.register_processor(
    CanaryQwenMultiModalProcessor,
    info=CanaryQwenProcessingInfo,
    dummy_inputs=CanaryQwenDummyInputsBuilder,
)
class CanaryQwenForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
):
    """
    NVIDIA Canary-Qwen Speech Recognition Model.

    This is a Speech-Augmented Language Model (SALM) that combines a
    FastConformer audio encoder with a Qwen3-1.7B LLM backbone.
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # Map NeMo weight names to vLLM weight names
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm.": "language_model.",
            "llm.model.": "language_model.model.",
            "embed_tokens.": "language_model.model.embed_tokens.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER_TOKEN
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        # Ensure the audio placeholder token is in the tokenizer
        # This must be done early so all components use the same tokenizer
        self._ensure_audio_token_in_tokenizer(vllm_config)

        # Store perception config from model config if available
        # Will be loaded during weight loading
        self._perception_config = getattr(config, "perception", None)
        self._perception: nn.Module | None = None

        # Initialize the language model (Qwen3)
        # The actual LLM config should be in config.text_config or similar
        llm_config = getattr(config, "text_config", config)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=llm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )

        # Audio placeholder token from config
        self._audio_locator_tag = getattr(
            config, "audio_locator_tag", _AUDIO_PLACEHOLDER_TOKEN
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _ensure_audio_token_in_tokenizer(self, vllm_config: VllmConfig) -> None:
        """Ensure the audio placeholder token is in the tokenizer.

        The base Qwen tokenizer doesn't have <|audioplaceholder|>, but Canary-Qwen
        requires it. This method adds it to the tokenizer if not already present.
        """
        from vllm.tokenizers import cached_tokenizer_from_config

        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        if tokenizer is None:
            return

        vocab = tokenizer.get_vocab()
        if _AUDIO_PLACEHOLDER_TOKEN not in vocab:
            num_added = tokenizer.add_special_tokens(
                {"additional_special_tokens": [_AUDIO_PLACEHOLDER_TOKEN]}
            )
            if num_added > 0:
                logger.info(
                    "Added audio placeholder token '%s' to tokenizer (id=%d)",
                    _AUDIO_PLACEHOLDER_TOKEN,
                    tokenizer.convert_tokens_to_ids(_AUDIO_PLACEHOLDER_TOKEN),
                )

    @property
    def perception(self) -> nn.Module:
        """Lazily initialize perception module."""
        if self._perception is None:
            raise RuntimeError(
                "Perception module not initialized. "
                "Call load_weights() first."
            )
        return self._perception

    def _init_perception_from_config(self, perception_config: dict) -> None:
        """Initialize the perception module from config."""
        device = next(self.language_model.parameters()).device
        dtype = next(self.language_model.parameters()).dtype

        self._perception = _load_nemo_perception_module(
            perception_config,
            device=device,
            dtype=dtype,
        )
        self._perception_config = perception_config

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> CanaryQwenAudioInputs | CanaryQwenAudioEmbedInputs | None:
        """Parse and validate audio input from kwargs."""
        audio_signal = kwargs.pop("audio_signal", None)
        audio_signal_length = kwargs.pop("audio_signal_length", None)
        audio_embed_sizes = kwargs.pop("audio_embed_sizes", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_embeds is not None:
            return CanaryQwenAudioEmbedInputs(
                type="audio_embeds",
                audio_embeds=audio_embeds,
            )

        if audio_signal is None:
            return None

        if not isinstance(audio_signal, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of audio signal. Got type: {type(audio_signal)}"
            )

        # Handle list of tensors (variable length audios)
        if isinstance(audio_signal, list):
            # Pad and stack
            max_len = max(a.shape[-1] for a in audio_signal)
            padded = []
            for audio in audio_signal:
                if audio.shape[-1] < max_len:
                    pad_size = max_len - audio.shape[-1]
                    audio = torch.nn.functional.pad(audio, (0, pad_size))
                padded.append(audio)
            audio_signal = torch.stack(padded, dim=0)

        # Ensure audio_signal_length is a tensor
        if audio_signal_length is None:
            audio_signal_length = torch.tensor(
                [audio_signal.shape[-1]] * audio_signal.shape[0],
                device=audio_signal.device,
            )
        elif not isinstance(audio_signal_length, torch.Tensor):
            audio_signal_length = torch.tensor(audio_signal_length)

        # Compute audio_embed_sizes if not provided
        if audio_embed_sizes is None:
            processor = CanaryQwenMultiModalProcessor(
                CanaryQwenProcessingInfo(self.vllm_config.model_config)
            )
            audio_embed_sizes = [
                processor._estimate_audio_tokens(length.item())
                for length in audio_signal_length
            ]
        elif isinstance(audio_embed_sizes, torch.Tensor):
            audio_embed_sizes = audio_embed_sizes.tolist()

        return CanaryQwenAudioInputs(
            type="audio_features",
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            audio_embed_sizes=audio_embed_sizes,
        )

    def _process_audio_input(
        self,
        audio_input: CanaryQwenAudioInputs | CanaryQwenAudioEmbedInputs,
    ) -> tuple[torch.Tensor, ...]:
        """Process audio input through the perception module."""
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            if isinstance(audio_embeds, torch.Tensor):
                return tuple(audio_embeds)
            return tuple(audio_embeds)

        audio_signal = audio_input["audio_signal"]
        audio_signal_length = audio_input["audio_signal_length"]

        # Move to correct device/dtype
        device = next(self.perception.parameters()).device
        dtype = next(self.perception.parameters()).dtype

        if isinstance(audio_signal, list):
            audio_signal = torch.stack(audio_signal, dim=0)

        audio_signal = audio_signal.to(device=device, dtype=dtype)
        audio_signal_length = audio_signal_length.to(device=device)

        # Run through perception module
        # NeMo's AudioPerceptionModule expects (B, T) audio and returns (B, T', D)
        with torch.no_grad():
            audio_embeds, audio_embed_lens = self.perception(
                input_signal=audio_signal,
                input_signal_length=audio_signal_length,
            )

        # Split by actual embedding lengths
        audio_embed_sizes = audio_input["audio_embed_sizes"]
        result = []
        for i, size in enumerate(audio_embed_sizes):
            # Take only the valid portion of embeddings
            actual_len = min(size, audio_embed_lens[i].item())
            result.append(audio_embeds[i, :actual_len])

        return tuple(result)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        """Compute audio embeddings if audio inputs are present."""
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        audio_features = self._process_audio_input(audio_input)
        return audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        model_output = self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load model weights from checkpoint.

        This handles:
        1. Perception module weights
        2. LLM weights (with LoRA merged)
        3. Token embeddings

        The NeMo checkpoint has this structure:
        - perception.* -> perception module
        - embed_tokens.weight -> language_model.model.embed_tokens.weight
        - llm.base_model.model.model.* -> language_model.model.*
          - .base_layer.weight for layers with LoRA
          - .lora_A.default.weight, .lora_B.default.weight for LoRA
        """
        # Collect weights for different components
        perception_weights = {}
        llm_base_weights = {}
        embed_weights = {}
        lora_a_weights = {}
        lora_b_weights = {}

        # Weight name prefixes in NeMo checkpoint
        perception_prefix = "perception."
        llm_peft_prefix = "llm.base_model.model."  # PEFT wrapper prefix
        llm_prefix = "llm."
        embed_prefix = "embed_tokens."

        weights_list = list(weights)

        for name, tensor in weights_list:
            if name.startswith(perception_prefix):
                # Perception module weights - strip prefix
                perception_weights[name[len(perception_prefix) :]] = tensor
            elif name.startswith(embed_prefix):
                # Embedding weights - will be remapped
                embed_weights[name] = tensor
            elif ".lora_A." in name:
                # LoRA A matrix
                lora_a_weights[name] = tensor
            elif ".lora_B." in name:
                # LoRA B matrix
                lora_b_weights[name] = tensor
            elif name.startswith(llm_peft_prefix):
                # LLM weights with PEFT prefix
                # Remove llm.base_model.model. and map to language_model
                suffix = name[len(llm_peft_prefix) :]
                # Handle base_layer.weight for layers with LoRA
                if ".base_layer.weight" in suffix:
                    suffix = suffix.replace(".base_layer.weight", ".weight")
                new_name = "language_model." + suffix
                llm_base_weights[new_name] = tensor
            elif name.startswith(llm_prefix):
                # Fallback: LLM weights without PEFT prefix
                new_name = "language_model." + name[len(llm_prefix) :]
                llm_base_weights[new_name] = tensor
            else:
                # Unknown weights - try to route to language model
                llm_base_weights[name] = tensor

        # Load perception module
        if perception_weights:
            self._load_perception_weights(perception_weights)

        # Merge LoRA weights into base weights
        llm_base_weights = self._merge_lora_weights(
            llm_base_weights, lora_a_weights, lora_b_weights
        )

        # Load LLM weights using AutoWeightsLoader
        loader = AutoWeightsLoader(self)

        # Combine LLM and embed weights
        combined_weights = []
        for name, tensor in llm_base_weights.items():
            combined_weights.append((name, tensor))
        for name, tensor in embed_weights.items():
            # Map embed_tokens to language_model
            new_name = f"language_model.model.{name}"
            combined_weights.append((new_name, tensor))

        loaded = loader.load_weights(iter(combined_weights))
        return loaded

    def _load_perception_weights(
        self,
        weights: dict[str, torch.Tensor],
    ) -> None:
        """Load perception module weights."""
        # Try to get perception config from the checkpoint or model config
        if self._perception is None:
            # Use config from model if available, otherwise use default
            if self._perception_config is not None:
                perception_config = self._perception_config
                if hasattr(perception_config, "to_dict"):
                    perception_config = perception_config.to_dict()
                elif hasattr(perception_config, "__dict__"):
                    perception_config = dict(perception_config)
            else:
                perception_config = self._get_default_perception_config()

            logger.info("Initializing Canary-Qwen perception module")
            self._init_perception_from_config(perception_config)

        # Load the weights
        missing, unexpected = self._perception.load_state_dict(
            weights, strict=False
        )
        if missing:
            logger.debug(
                "Some perception weights were not loaded (expected for buffers): %s",
                missing[:5] if len(missing) > 5 else missing,
            )

    def _get_default_perception_config(self) -> dict:
        """Get default perception module config for Canary-Qwen.

        This config matches nvidia/canary-qwen-2.5b perception module.
        Values are from the HuggingFace config.json.
        """
        return {
            "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": 128,
                "n_fft": 512,
                "frame_splicing": 1,
                "dither": 1e-05,
                "pad_to": 0,
                "pad_value": 0.0,
                "log": True,
            },
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 128,
                "feat_out": -1,
                "n_layers": 32,
                "d_model": 1024,
                "subsampling": "dw_striding",
                "subsampling_factor": 8,
                "subsampling_conv_channels": 256,
                "causal_downsampling": False,
                "ff_expansion_factor": 4,
                "self_attention_model": "rel_pos",
                "n_heads": 8,
                "att_context_size": [-1, -1],
                "xscaling": False,
                "untie_biases": True,
                "pos_emb_max_len": 5000,
                "conv_kernel_size": 9,
                "conv_norm_type": "batch_norm",
                "dropout": 0.1,
                "dropout_emb": 0.0,
                "dropout_att": 0.1,
                "dropout_pre_encoder": 0.1,
            },
            "modality_adapter": {
                "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                "d_model": 1024,
            },
            "output_dim": 2048,  # Qwen3-1.7B hidden size
        }

    def _merge_lora_weights(
        self,
        base_weights: dict[str, torch.Tensor],
        lora_a_weights: dict[str, torch.Tensor],
        lora_b_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Merge LoRA weights into base weights.

        LoRA applies: W' = W + alpha * (B @ A) / r

        Args:
            base_weights: Dict of base weight name -> tensor (already remapped)
            lora_a_weights: Dict of original LoRA A weight name -> tensor
            lora_b_weights: Dict of original LoRA B weight name -> tensor

        Returns:
            Updated base_weights dict with LoRA merged
        """
        if not lora_a_weights or not lora_b_weights:
            return base_weights

        # Get LoRA config for scaling
        lora_config = getattr(self.config, "lora", None) or {}
        lora_alpha = lora_config.get("lora_alpha", 256)
        lora_r = lora_config.get("r", 128)
        scaling = lora_alpha / lora_r

        # Group LoRA weights by target module
        # Format: llm.base_model.model.model.layers.X.self_attn.q_proj.lora_A.default.weight
        lora_pairs = {}

        for name, tensor in lora_a_weights.items():
            # Extract the target module path
            # Remove llm.base_model.model. prefix and .lora_A.default.weight suffix
            base_name = name.replace("llm.base_model.model.", "")
            base_name = base_name.replace(".lora_A.default.weight", ".weight")
            # Map to vLLM naming: model.layers.X -> model.layers.X
            vllm_name = "language_model." + base_name
            if vllm_name not in lora_pairs:
                lora_pairs[vllm_name] = {}
            lora_pairs[vllm_name]["lora_A"] = tensor

        for name, tensor in lora_b_weights.items():
            base_name = name.replace("llm.base_model.model.", "")
            base_name = base_name.replace(".lora_B.default.weight", ".weight")
            vllm_name = "language_model." + base_name
            if vllm_name not in lora_pairs:
                lora_pairs[vllm_name] = {}
            lora_pairs[vllm_name]["lora_B"] = tensor

        # Merge each LoRA pair into base weights
        merged_count = 0
        for target_name, lora_pair in lora_pairs.items():
            if "lora_A" not in lora_pair or "lora_B" not in lora_pair:
                logger.warning(
                    "Incomplete LoRA pair for %s, skipping merge", target_name
                )
                continue

            if target_name not in base_weights:
                logger.warning(
                    "Base weight %s not found for LoRA merge, skipping", target_name
                )
                continue

            lora_A = lora_pair["lora_A"]
            lora_B = lora_pair["lora_B"]
            base_weight = base_weights[target_name]

            # Compute LoRA contribution: scaling * B @ A
            # lora_A: (r, in_features), lora_B: (out_features, r)
            # Result: (out_features, in_features)
            lora_delta = lora_B @ lora_A

            # Merge with scaling
            merged = base_weight + scaling * lora_delta.to(base_weight.dtype)
            base_weights[target_name] = merged
            merged_count += 1

        logger.info("Merged %d LoRA weight pairs (scaling=%.2f)", merged_count, scaling)
        return base_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix in multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="perception.proj",
            tower_model="perception.encoder",
        )

