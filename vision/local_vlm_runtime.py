import threading
from typing import Optional

from vision.local_vlm_ocr import LocalVlmConfig, LocalVlmOcr


_LOCAL_VLM_LOCK = threading.Lock()
_LOCAL_VLM: Optional[LocalVlmOcr] = None
_LOCAL_VLM_KEY: str = ""


def get_local_vlm(*, model: str, models_dir: str, hf_home: str, device: str) -> LocalVlmOcr:
    global _LOCAL_VLM, _LOCAL_VLM_KEY
    key = f"{model}|{models_dir}|{hf_home}|{device}"
    with _LOCAL_VLM_LOCK:
        if _LOCAL_VLM is not None and _LOCAL_VLM_KEY == key:
            return _LOCAL_VLM

        cfg = LocalVlmConfig(
            model=model,
            models_dir=models_dir,
            hf_home=hf_home,
            device=device,
            max_new_tokens=2048,
        )
        _LOCAL_VLM = LocalVlmOcr(cfg)
        _LOCAL_VLM_KEY = key
        return _LOCAL_VLM
