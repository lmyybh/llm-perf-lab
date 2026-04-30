"""Random synthetic dataset for benchmark requests."""

import random
from typing import Iterable

import typer
from transformers import PreTrainedTokenizerBase

from llmperf.common import (
    ChatCompletionInput,
    ChatCompletionMessage,
    LLMRequest,
    SamplingParams,
)


class RandomDataset:
    """Dataset that generates synthetic requests from random token sequences.

    Attributes:
        name (str): Dataset name used for identification.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to decode sampled
            token ids into message text.
        num_requests (int): Number of requests to generate.
        min_input_tokens (int): Minimum number of prompt tokens per request.
        max_input_tokens (int): Maximum number of prompt tokens per request.
        min_output_tokens (int): Minimum completion token target per request.
        max_output_tokens (int): Maximum completion token target per request.
        seed (int): Random seed used for reproducible generation.
    """

    name: str = "random"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        min_input_tokens: int,
        max_input_tokens: int,
        min_output_tokens: int,
        max_output_tokens: int,
        seed: int = 0,
    ) -> None:
        """Initialize a random synthetic dataset.

        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer used to decode
                sampled token ids.
            num_requests (int): Number of requests to generate.
            min_input_tokens (int): Minimum number of prompt tokens.
            max_input_tokens (int): Maximum number of prompt tokens.
            min_output_tokens (int): Minimum completion token target.
            max_output_tokens (int): Maximum completion token target.
            seed (int): Random seed used for reproducible generation.

        Raises:
            typer.BadParameter: Raised when any range or count is invalid.
        """
        if num_requests <= 0:
            raise typer.BadParameter(
                "random dataset num_requests must be greater than 0"
            )
        if min_input_tokens <= 0 or max_input_tokens <= 0:
            raise typer.BadParameter(
                "random dataset input token bounds must be greater than 0"
            )
        if min_output_tokens <= 0 or max_output_tokens <= 0:
            raise typer.BadParameter(
                "random dataset output token bounds must be greater than 0"
            )
        if min_input_tokens > max_input_tokens:
            raise typer.BadParameter(
                "random dataset min_input_tokens cannot exceed max_input_tokens"
            )
        if min_output_tokens > max_output_tokens:
            raise typer.BadParameter(
                "random dataset min_output_tokens cannot exceed max_output_tokens"
            )

        self.tokenizer = tokenizer
        self.num_requests = num_requests
        self.min_input_tokens = min_input_tokens
        self.max_input_tokens = max_input_tokens
        self.min_output_tokens = min_output_tokens
        self.max_output_tokens = max_output_tokens
        self.seed = seed

    def _count_template_tokens(self) -> int:
        return ChatCompletionInput(
            messages=[ChatCompletionMessage(role="user", content="")]
        ).estimate_prompt_tokens_length(self.tokenizer)

    def _build_candidate_token_ids(self) -> list[int]:
        """Build token ids that are safe to sample for prompt generation.

        Returns:
            list[int]: Sampleable token ids excluding known special ids.

        Raises:
            typer.BadParameter: Raised when the tokenizer vocabulary is empty.
        """
        vocab_size = self.tokenizer.vocab_size
        if vocab_size <= 0:
            raise typer.BadParameter("random dataset tokenizer vocabulary is empty")

        special_ids = {
            token_id
            for token_id in self.tokenizer.all_special_ids
            if 0 <= token_id < vocab_size
        }
        candidate_ids = [
            token_id for token_id in range(vocab_size) if token_id not in special_ids
        ]

        return candidate_ids or list(range(vocab_size))

    def _build_request_text(
        self, rng: random.Random, candidate_token_ids: list[int], num_tokens: int
    ) -> str:
        """Decode a random token sequence into prompt text.

        Args:
            rng (random.Random): Random generator used for reproducible sampling.
            candidate_token_ids (list[int]): Token ids available for sampling.
            num_tokens (int): Number of tokens to sample for the prompt.

        Returns:
            str: Decoded prompt text.
        """
        token_ids = [rng.choice(candidate_token_ids) for _ in range(num_tokens)]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if text:
            return text

        fallback_token = candidate_token_ids[0]
        return self.tokenizer.decode([fallback_token] * num_tokens).strip() or "x"

    def _compute_content_token_budget(
        self, total_input_tokens: int, template_tokens: int
    ) -> int:
        """Compute the prompt content budget after subtracting template tokens.

        Args:
            total_input_tokens (int): Target total prompt token count including
                chat template overhead.
            template_tokens (int): Token overhead of the empty chat template.

        Returns:
            int: Token budget available for random prompt content.

        Raises:
            typer.BadParameter: Raised when the requested total token target is
                smaller than the template overhead.
        """
        content_budget = total_input_tokens - template_tokens
        if content_budget <= 0:
            raise typer.BadParameter(
                "random dataset input token target must exceed chat template overhead"
            )

        return content_budget

    def _estimate_prompt_tokens(self, prompt_text: str) -> int:
        return ChatCompletionInput(
            messages=[ChatCompletionMessage(role="user", content=prompt_text)]
        ).estimate_prompt_tokens_length(self.tokenizer)

    def iter_requests(self) -> Iterable[LLMRequest]:
        """Yield synthetic benchmark requests with random prompt lengths.

        Returns:
            Iterable[LLMRequest]: Generated synthetic requests.
        """
        rng = random.Random(self.seed)
        candidate_token_ids = self._build_candidate_token_ids()
        template_tokens = self._count_template_tokens()

        for _ in range(self.num_requests):
            input_tokens = rng.randint(self.min_input_tokens, self.max_input_tokens)
            output_tokens = rng.randint(self.min_output_tokens, self.max_output_tokens)
            content_tokens = self._compute_content_token_budget(
                total_input_tokens=input_tokens,
                template_tokens=template_tokens,
            )
            prompt_text = self._build_request_text(
                rng=rng,
                candidate_token_ids=candidate_token_ids,
                num_tokens=content_tokens,
            )

            yield LLMRequest(
                input=ChatCompletionInput(
                    messages=[ChatCompletionMessage(role="user", content=prompt_text)]
                ),
                sampling_params=SamplingParams(
                    max_completion_tokens=output_tokens,
                    ignore_eos=True,
                ),
                extra={"prompt_tokens": self._estimate_prompt_tokens(prompt_text)},
            )
