from dataclasses import dataclass
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union, List, Optional, Tuple
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from chronos import ChronosTokenizer, ChronosModel, ChronosConfig

@dataclass
class ChronosPipeline:
    tokenizer: ChronosTokenizer
    model: ChronosModel

    def _prepare_and_validate_context(self, context: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2
        return context

    @torch.no_grad()
    def embed(self, context: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, Any]:
        context_tensor = self._prepare_and_validate_context(context=context)
        token_ids, attention_mask, tokenizer_state = (
            self.tokenizer.context_input_transform(context_tensor)
        )
        embeddings = self.model.encode(
            input_ids=token_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
        ).cpu()
        return embeddings, tokenizer_state

    def predict(self,
                context: Union[torch.Tensor, List[torch.Tensor]],
                prediction_length: Optional[int] = None,
                num_samples: Optional[int] = None,
                temperature: Optional[float] = None,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                limit_prediction_length: bool = True) -> torch.Tensor:
        context_tensor = self._prepare_and_validate_context(context=context)

        if prediction_length is None:
            prediction_length = self.model.config.prediction_length

        if prediction_length > self.model.config.prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {self.model.config.prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        while remaining > 0:
            token_ids, attention_mask, scale = self.tokenizer.context_input_transform(
                context_tensor
            )
            samples = self.model(
                token_ids.to(self.model.device),
                attention_mask.to(self.model.device),
                min(remaining, self.model.config.prediction_length),
                num_samples,
                temperature,
                top_k,
                top_p,
            )
            prediction = self.tokenizer.output_transform(
                samples.to(scale.device), scale
            )

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            context_tensor = torch.cat(
                [context_tensor, prediction.median(dim=1).values], dim=-1
            )

        return torch.cat(predictions, dim=-1)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = AutoConfig.from_pretrained(*args, **kwargs)
        print(config)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        chronos_config = ChronosConfig(**config.chronos_config)

        if chronos_config.model_type == "seq2seq":
            inner_model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
        else:
            assert chronos_config.model_type == "causal"
            inner_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        model = ChronosModel(config=chronos_config, model=inner_model)
        return cls(
            tokenizer=chronos_config.create_tokenizer(),
            model=model,
        )

    def to_ddp(self):
        self.model = DDP(self.model)

