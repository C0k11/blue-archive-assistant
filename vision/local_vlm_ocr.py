import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor


try:
    from transformers import AutoModelForVision2Seq  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForVision2Seq = None  # type: ignore


try:
    from transformers import AutoModelForCausalLM  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore


try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None  # type: ignore


@dataclass
class LocalVlmConfig:
    model: str
    models_dir: str
    hf_home: str
    device: str = "cuda"
    max_new_tokens: int = 2048


class LocalVlmOcr:
    def __init__(self, cfg: LocalVlmConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._model = None
        self._processor = None

    def _set_hf_cache(self) -> None:
        os.environ.setdefault("HF_HOME", self.cfg.hf_home)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cfg.hf_home)

    def _resolve_model_path(self) -> Tuple[str, Optional[str]]:
        m = (self.cfg.model or "").strip()
        if not m:
            raise ValueError("model is required")

        p = Path(m)
        if p.exists() and p.is_dir():
            return str(p), None

        models_dir = Path(self.cfg.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        local_dir = models_dir / m.replace("/", "__").replace(":", "__")

        if local_dir.exists() and local_dir.is_dir() and any(local_dir.iterdir()):
            return str(local_dir), m

        if snapshot_download is None:
            raise ImportError("huggingface_hub is required to download models")

        snapshot_download(
            repo_id=m,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return str(local_dir), m

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None and self._processor is not None:
                return

            self._set_hf_cache()
            model_path, _ = self._resolve_model_path()

            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            model = None
            if AutoModelForVision2Seq is not None:
                try:
                    model = AutoModelForVision2Seq.from_pretrained(
                        model_path,
                        torch_dtype="auto",
                        device_map="auto" if self.cfg.device == "cuda" else None,
                        trust_remote_code=True,
                    )
                except Exception:
                    model = None

            if model is None:
                if AutoModelForCausalLM is None:
                    raise ImportError("transformers AutoModelForCausalLM is not available")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto" if self.cfg.device == "cuda" else None,
                    trust_remote_code=True,
                )

            self._model = model.eval()

    def ocr(self, *, image_path: str, prompt: str, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if hasattr(self._processor, "apply_chat_template"):
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = self._processor(text=[text], images=[image], return_tensors="pt")

        if hasattr(self._model, "device"):
            device = getattr(self._model, "device")
            try:
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            except Exception:
                pass

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens) if max_new_tokens else int(self.cfg.max_new_tokens),
                do_sample=False,
            )

        if isinstance(inputs, dict) and "input_ids" in inputs:
            try:
                generated = generated[:, inputs["input_ids"].shape[1] :]
            except Exception:
                pass

        out = self._processor.batch_decode(generated, skip_special_tokens=True)
        text_out = out[0] if out else ""
        return {"raw": text_out}
