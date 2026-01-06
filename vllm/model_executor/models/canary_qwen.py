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
    PromptIndexTargets,
    PromptInsertion,
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

_AUDIO_PLACEHOLDER_TOKEN = "<|audioplaceholder|>"
_SAMPLING_RATE = 16000
_MAX_AUDIO_DURATION_S = 40.0


class CanaryQwenAudioInputs(TensorSchema):
    """Audio input features for Canary-Qwen. Shapes: b=batch, t=time samples."""

    type: Literal["audio_features"]
    audio_signal: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("b", "t")]
    audio_signal_length: Annotated[torch.Tensor, TensorShape("b")]
    audio_embed_sizes: Annotated[list[int], TensorShape("b")]


class CanaryQwenAudioEmbedInputs(TensorSchema):
    """Pre-computed audio embeddings. Shapes: b=batch, t=time, d=dimension."""

    type: Literal["audio_embeds"]
    audio_embeds: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("b", "t", "d")]


class CanaryQwenProcessingInfo(BaseProcessingInfo):
    _audio_token_id: int | None = None

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._ensure_audio_token_in_tokenizer()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_max_audio_tokens(self) -> int:
        return CanaryQwenMultiModalProcessor._estimate_audio_tokens(self.get_max_audio_len())

    def get_max_audio_len(self) -> int:
        return int(_MAX_AUDIO_DURATION_S * _SAMPLING_RATE)

    def _ensure_audio_token_in_tokenizer(self) -> int:
        tokenizer = self.get_tokenizer()

        token_id = tokenizer.convert_tokens_to_ids(_AUDIO_PLACEHOLDER_TOKEN)
        if token_id is not None:
            return token_id

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
        if self._audio_token_id is None:
            self._audio_token_id = self._ensure_audio_token_in_tokenizer()
        return self._audio_token_id


class CanaryQwenMultiModalProcessor(
    BaseMultiModalProcessor[CanaryQwenProcessingInfo]
):
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
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        from vllm.multimodal.processing import PromptUpdateDetails

        audio_token_id = self.info.get_audio_token_id()

        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio = audios.get(item_idx)
            audio_length = audio.shape[-1]
            num_audio_tokens = self._estimate_audio_tokens(audio_length)

            audio_tokens = [audio_token_id] * num_audio_tokens
            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_replacement,
            )
        ]

    @staticmethod
    def _estimate_audio_tokens(audio_length_samples: int) -> int:
        """Estimate audio tokens matching NeMo's FastConformer calculation."""
        seq_len = torch.as_tensor(audio_length_samples)

        n_fft = 512
        hop_length = 160
        stft_pad_amount = n_fft // 2

        left_padding = 1
        right_padding = 1
        kernel_size = 3
        stride = 2
        repeat_num = 3

        pad_amount = stft_pad_amount * 2
        fbank_len = torch.floor_divide(seq_len + pad_amount - n_fft, hop_length)

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
        self.info.get_audio_token_id()

        tokenizer = self.info.get_tokenizer()
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

        result = BatchFeature(
            dict(input_ids=[prompt_ids]),
            tensor_type="pt",
        )

        if audios:
            audio_list = []
            audio_lengths = []
            audio_embed_sizes = []

            for audio in audios:
                audio_tensor = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio)
                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.squeeze()

                audio_list.append(audio_tensor)
                audio_lengths.append(audio_tensor.shape[-1])
                audio_embed_sizes.append(self._estimate_audio_tokens(audio_tensor.shape[-1]))

            result["audio_signal"] = audio_list
            result["audio_signal_length"] = torch.tensor(audio_lengths)
            result["audio_embed_sizes"] = torch.tensor(audio_embed_sizes)

        return result


class CanaryQwenDummyInputsBuilder(
    BaseDummyInputsBuilder[CanaryQwenProcessingInfo]
):
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
        return "Transcribe the following: " + _AUDIO_PLACEHOLDER_TOKEN * num_audios


def _load_nemo_perception_module(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    try:
        from nemo.collections.speechlm2.modules import AudioPerceptionModule
        from omegaconf import DictConfig
    except ImportError as e:
        raise ImportError(
            "NeMo is required for Canary-Qwen model. "
            "Install it with: pip install nemo_toolkit[asr,tts]"
        ) from e

    perception_config = DictConfig(config)
    perception = AudioPerceptionModule(perception_config).to(device=device, dtype=dtype).eval()
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
    """NVIDIA Canary-Qwen-2.5B model."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

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

        self._ensure_audio_token_in_tokenizer(vllm_config)

        llm_config = getattr(config, "text_config", config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=llm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )

        self.perception_config = getattr(config, "perception", None)
        self.perception = _load_nemo_perception_module(
            self.perception_config,
            device=next(self.language_model.parameters()).device,
            dtype=next(self.language_model.parameters()).dtype,
        )

        self._audio_locator_tag = getattr(
            config, "audio_locator_tag", _AUDIO_PLACEHOLDER_TOKEN
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _ensure_audio_token_in_tokenizer(self, vllm_config: VllmConfig) -> None:
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

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> CanaryQwenAudioInputs | CanaryQwenAudioEmbedInputs | None:
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

        if isinstance(audio_signal, list):
            max_len = max(a.shape[-1] for a in audio_signal)
            padded = []
            for audio in audio_signal:
                if audio.shape[-1] < max_len:
                    pad_size = max_len - audio.shape[-1]
                    audio = torch.nn.functional.pad(audio, (0, pad_size))
                padded.append(audio)
            audio_signal = torch.stack(padded, dim=0)

        if audio_signal_length is None:
            audio_signal_length = torch.tensor(
                [audio_signal.shape[-1]] * audio_signal.shape[0],
                device=audio_signal.device,
            )
        elif not isinstance(audio_signal_length, torch.Tensor):
            audio_signal_length = torch.tensor(audio_signal_length)

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
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            return tuple(audio_embeds) if isinstance(audio_embeds, torch.Tensor) else tuple(audio_embeds)

        audio_signal = audio_input["audio_signal"]
        audio_signal_length = audio_input["audio_signal_length"]

        device = next(self.perception.parameters()).device

        if isinstance(audio_signal, list):
            audio_signal = torch.stack(audio_signal, dim=0)

        audio_signal = audio_signal.to(device=device, dtype=torch.float32)
        audio_signal_length = audio_signal_length.to(device=device)

        with torch.no_grad():
            audio_embeds, audio_embed_lens = self.perception(
                input_signal=audio_signal,
                input_signal_length=audio_signal_length,
            )
        result = []
        for i, size in enumerate(audio_embed_lens.cpu()):
            result.append(audio_embeds[i, :size])
        return tuple(result)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        return self._process_audio_input(audio_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Apply token embeddings to input_ids and merge with multimodal embeddings if present."""
        from .utils import _merge_multimodal_embeddings

        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )


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

        llm_output = self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return llm_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        perception_weights = {}
        llm_base_weights = {}
        embed_weights = {}
        lora_a_weights = {}
        lora_b_weights = {}

        perception_prefix = "perception."
        llm_peft_prefix = "llm.base_model.model."
        llm_prefix = "llm."
        embed_prefix = "embed_tokens."

        weights_list = list(weights)

        for name, tensor in weights_list:
            if name.startswith(perception_prefix):
                perception_weights[name[len(perception_prefix):]] = tensor
            elif name.startswith(embed_prefix):
                embed_weights[name] = tensor
            elif ".lora_A." in name:
                lora_a_weights[name] = tensor
            elif ".lora_B." in name:
                lora_b_weights[name] = tensor
            elif name.startswith(llm_peft_prefix):
                suffix = name[len(llm_peft_prefix):]
                if ".base_layer.weight" in suffix:
                    suffix = suffix.replace(".base_layer.weight", ".weight")
                new_name = "language_model." + suffix
                llm_base_weights[new_name] = tensor
            elif name.startswith(llm_prefix):
                new_name = "language_model." + name[len(llm_prefix):]
                llm_base_weights[new_name] = tensor
            else:
                llm_base_weights[name] = tensor

        llm_base_weights = self._merge_lora_weights(
            llm_base_weights, lora_a_weights, lora_b_weights
        )

        self.perception.load_state_dict(perception_weights, strict=True)

        loader = AutoWeightsLoader(self)

        combined_weights = []
        for name, tensor in llm_base_weights.items():
            combined_weights.append((name, tensor))
        for name, tensor in embed_weights.items():
            new_name = f"language_model.model.{name}"
            combined_weights.append((new_name, tensor))

        ans = loader.load_weights(iter(combined_weights)) | {perception_prefix + k for k in self.perception.state_dict().keys()}
        return ans

    def _merge_lora_weights(
        self,
        base_weights: dict[str, torch.Tensor],
        lora_a_weights: dict[str, torch.Tensor],
        lora_b_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not lora_a_weights or not lora_b_weights:
            return base_weights

        lora_config = getattr(self.config, "lora", None) or {}
        lora_alpha = lora_config.get("lora_alpha", 256)
        lora_r = lora_config.get("r", 128)
        scaling = lora_alpha / lora_r

        lora_pairs = {}

        for name, tensor in lora_a_weights.items():
            base_name = name.replace("llm.base_model.model.", "")
            base_name = base_name.replace(".lora_A.default.weight", ".weight")
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

        merged_count = 0
        for target_name, lora_pair in lora_pairs.items():
            if "lora_A" not in lora_pair or "lora_B" not in lora_pair:
                logger.warning("Incomplete LoRA pair for %s, skipping merge", target_name)
                continue

            if target_name not in base_weights:
                logger.warning("Base weight %s not found for LoRA merge, skipping", target_name)
                continue

            orig_dtype = base_weights[target_name].dtype
            lora_A = lora_pair["lora_A"].float()
            lora_B = lora_pair["lora_B"].float()
            base_weight = base_weights[target_name].float()

            lora_delta = lora_B @ lora_A
            merged = base_weight + scaling * lora_delta
            base_weights[target_name] = merged.to(orig_dtype)
            merged_count += 1

        logger.info("Merged %d LoRA weight pairs (scaling=%.2f)", merged_count, scaling)
        return base_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="perception.proj",
            tower_model="perception.encoder",
        )

