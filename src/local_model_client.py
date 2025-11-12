from typing import Optional, Dict, Any
import os

try:
    import torch
    import transformers
    from transformers import pipeline
except Exception:
    transformers = None
    torch = None


class LocalModelClient:
    def __init__(
        self,
        heavy_model: Optional[str] = None,
        light_model: Optional[str] = None,
        device: Optional[str] = None,
        prefer: Optional[str] = None,
        force_heavy: bool = False,
        offload_folder: Optional[str] = None,
    ) -> None:
        if transformers is None:
            raise RuntimeError("transformers or torch not available for LocalModelClient")

        self.heavy_model = heavy_model or os.getenv("LOCAL_HEAVY_MODEL", "tiiuae/falcon-7b-instruct")
        self.light_model = light_model or os.getenv("LOCAL_LIGHT_MODEL", "distilgpt2")
        self.device = device or os.getenv("LOCAL_DEVICE") or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.prefer = prefer or os.getenv("LOCAL_PREFER", "light")
        self.force_heavy = force_heavy or os.getenv("LOCAL_FORCE_HEAVY", "false").lower() == "true"
        self.offload_folder = offload_folder or os.getenv("LOCAL_OFFLOAD_FOLDER")

        model_name = self._choose_model_name()
        device_arg = 0 if self.device.startswith("cuda") else -1
        self.generator = self._create_generator_for_model(model_name, device_arg)

    def _choose_model_name(self) -> str:
        return self.heavy_model if self.prefer == "heavy" else self.light_model

    def _create_generator_for_model(self, model_name: str, device_arg: int):
        if model_name != self.heavy_model:
            return pipeline("text-generation", model=model_name, device=device_arg)

        gpu_available = torch is not None and torch.cuda.is_available()
        if not gpu_available and not self.force_heavy:
            raise RuntimeError("Heavy model requested but no CUDA GPU detected. Refuse to load heavy model on CPU unless --force-heavy is used.")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(torch.float16 if torch is not None else None),
            )

            tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb,
                device_map="auto",
                offload_folder=self.offload_folder,
            )
            return pipeline("text-generation", model=model, tokenizer=tok, device=device_arg)
        except Exception:
            return pipeline("text-generation", model=model_name, device=device_arg)

    def generate(self, prompt: str, max_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        pipeline_kwargs: Dict[str, Any] = {}
        if "max_new_tokens" in gen_kwargs:
            pipeline_kwargs["max_new_tokens"] = gen_kwargs.pop("max_new_tokens")
        else:
            pipeline_kwargs["max_length"] = gen_kwargs.pop("max_tokens", max_tokens)

        for k in ("temperature", "do_sample", "top_k", "top_p", "num_return_sequences"):
            if k in gen_kwargs:
                pipeline_kwargs[k] = gen_kwargs[k]

        pipeline_kwargs.update(gen_kwargs)

        outputs = self.generator(prompt, **pipeline_kwargs)
        if isinstance(outputs, list) and len(outputs) > 0:
            first = outputs[0]
            text = first.get("generated_text") or first.get("text") or str(first)
            return {"text": text}
        return {"text": str(outputs)}
