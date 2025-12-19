# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVIDIA Canary-Qwen speech recognition model.

This model requires NeMo toolkit to be installed:
    pip install nemo_toolkit[asr]
"""

import json

import numpy as np
import pytest
import pytest_asyncio

from ....conftest import AUDIO_ASSETS, AudioTestAssets, VllmRunner
from ....utils import RemoteOpenAIServer
from ...registry import HF_EXAMPLE_MODELS

MODEL_NAME = "nvidia/canary-qwen-2.5b"

# Audio placeholder token used by Canary-Qwen
AUDIO_PLACEHOLDER = "<|audioplaceholder|>"

# Test prompts for ASR
AUDIO_PROMPTS = AUDIO_ASSETS.prompts(
    {
        "mary_had_lamb": "Transcribe the following:",
        "winning_call": "Transcribe the following:",
    }
)

AudioTuple = tuple[np.ndarray, int]

CHUNKED_PREFILL_KWARGS = {
    "enable_chunked_prefill": True,
    "max_num_seqs": 2,
    "max_num_batched_tokens": 16,
}


def _check_nemo_available():
    """Check if NeMo is available."""
    try:
        import nemo.collections.speechlm2  # noqa: F401

        return True
    except ImportError:
        return False




# Skip all tests if NeMo is not available
pytestmark = pytest.mark.skipif(
    not _check_nemo_available(),
    reason="NeMo toolkit is required for Canary-Qwen model. "
    "Install with: pip install nemo_toolkit[asr,tts]",
)


def _get_prompt_with_audio(question: str) -> str:
    """Create a prompt with audio placeholder for Canary-Qwen.

    Canary-Qwen uses a simple prompt format with the audio placeholder.
    """
    return f"{question} {AUDIO_PLACEHOLDER}"


def run_transcription_test(
    vllm_runner: type[VllmRunner],
    prompts_and_audios: list[tuple[str, list[AudioTuple]]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    **kwargs,
):
    """Run transcription test with Canary-Qwen model."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    # The model's config.json is in NeMo format and doesn't have architectures
    # or model_type. We need to override them to tell vLLM which model/config to use.
    hf_overrides = {
        "architectures": ["CanaryQwenForConditionalGeneration"],
        "model_type": "canary_qwen",
    }

    # The model on HuggingFace has an incorrect tokenizer (audio codec tokenizer)
    # We need to use the LLM's tokenizer. The model will automatically add
    # <|audioplaceholder|> during initialization.
    llm_tokenizer = "Qwen/Qwen3-1.7B"

    with vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": 1},
        hf_overrides=hf_overrides,
        tokenizer_name=llm_tokenizer,
        **kwargs,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [prompt for prompt, _ in prompts_and_audios],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[audios for _, audios in prompts_and_audios],
        )

    # Verify that tokens were generated
    assert all(tokens for tokens, *_ in vllm_outputs), (
        "No tokens generated for some inputs"
    )

    # Return outputs for further inspection if needed
    return vllm_outputs


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_canary_qwen_transcription(
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Test basic ASR transcription with Canary-Qwen."""
    # Use the first audio asset for testing
    audio, sr = audio_assets[0].audio_and_sample_rate

    # Canary-Qwen expects 16kHz audio
    assert sr == 16000, f"Expected 16kHz audio, got {sr}Hz"

    prompt = _get_prompt_with_audio("Transcribe the following:")

    outputs = run_transcription_test(
        vllm_runner,
        [(prompt, [audio])],
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
    )

    # Check that we got some transcription output
    tokens, text, logprobs = outputs[0]
    print(text)
    assert len(tokens) > 0, "No tokens generated"
    assert len(text) > 0, "No text generated"


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "vllm_kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param(CHUNKED_PREFILL_KWARGS, id="chunked_prefill"),
    ],
)
def test_canary_qwen_with_different_configs(
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    vllm_kwargs: dict,
) -> None:
    """Test Canary-Qwen with different vLLM configurations."""
    audio, sr = audio_assets[0].audio_and_sample_rate
    assert sr == 16000, f"Expected 16kHz audio, got {sr}Hz"

    prompt = _get_prompt_with_audio("Transcribe the following:")

    outputs = run_transcription_test(
        vllm_runner,
        [(prompt, [audio])],
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        **vllm_kwargs,
    )

    # Verify output
    tokens, text, logprobs = outputs[0]
    print(text)
    assert len(tokens) > 0, "No tokens generated"


# Online serving tests
@pytest.fixture
def server(audio_assets: AudioTestAssets):
    """Start vLLM server for online serving tests."""
    # The model's config.json is in NeMo format and doesn't have architectures
    # or model_type. We need to override them.
    hf_overrides = {
        "architectures": ["CanaryQwenForConditionalGeneration"],
        "model_type": "canary_qwen",
    }
    # The model on HuggingFace has an incorrect tokenizer (audio codec tokenizer)
    # We need to use the LLM's tokenizer. The model will automatically add
    # <|audioplaceholder|> during initialization.
    llm_tokenizer = "Qwen/Qwen3-1.7B"
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"audio": 1}),
        "--trust-remote-code",
        "--hf-overrides",
        json.dumps(hf_overrides),
        "--tokenizer",
        llm_tokenizer,
    ]

    with RemoteOpenAIServer(
        MODEL_NAME, args, env_dict={"VLLM_AUDIO_FETCH_TIMEOUT": "30"}
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    """Get async client for online serving tests."""
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_canary_qwen_online_serving(
    client, audio_assets: AudioTestAssets
) -> None:
    """Test Canary-Qwen with online serving API."""
    # Use the first audio asset
    audio_asset = audio_assets[0]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_asset.url}},
                {"type": "text", "text": "Transcribe the following:"},
            ],
        }
    ]

    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=128
    )

    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    # Should either complete or hit length limit
    assert choice.finish_reason in ("stop", "length")
    # Should have some content
    assert choice.message.content is not None
    assert len(choice.message.content) > 0

