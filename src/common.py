"""Skupni gradniki: konfiguracija, register orodij, nalaganje modela.

Vse skripte uvazajo od tu, da je vedenje enotno na Mac (mps) in NVIDIA (cuda).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

# LoRA se pripne samo na jezikovni del modela. Negativni pogled naprej izkljuci
# vizualni stolp - Ministral 3 je multimodalen in ucenje vizualnega dela je
# izven obsega tega cevovoda (glej README, razdelek "Slike").
LORA_TARGET_REGEX = (
    r"^(?!.*(?:vision|visual|patch_merger|multi_modal|mmproj|image))"
    r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Register orodij


@dataclass
class ToolRegistry:
    tools: list[dict[str, Any]]

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ToolRegistry":
        with open(ROOT / path if not Path(path).is_absolute() else path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(tools=raw["tools"])

    def groups(self) -> list[str]:
        return sorted({t.get("group", "ungrouped") for t in self.tools})

    def select(self, groups: list[str] | None = None) -> list[dict[str, Any]]:
        """Vrne orodja v OpenAI obliki, brez internih polj (group, _status)."""
        chosen = self.tools if groups is None else [t for t in self.tools if t.get("group") in groups]
        cleaned = []
        for tool in chosen:
            cleaned.append({"type": tool["type"], "function": tool["function"]})
        return cleaned

    def names(self) -> set[str]:
        return {t["function"]["name"] for t in self.tools}

    def spec(self, name: str) -> dict[str, Any] | None:
        for tool in self.tools:
            if tool["function"]["name"] == name:
                return tool["function"]
        return None


# ---------------------------------------------------------------------------
# Strojna oprema


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    # Mistralovi tokenizatorji na Hubu imajo napacen regex za razbijanje besed.
    # Brez tega zastavice je tokenizacija drugacna od tiste, s katero je bil
    # model naucen. Starejsi transformers argumenta ne pozna - takrat gremo brez.
    try:
        tok = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    except (TypeError, ValueError):
        tok = AutoTokenizer.from_pretrained(model_id)

    if tok.pad_token is None:
        # Za maskiranje je vseeno kateri token je pad - izgube na njem ne racunamo.
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: dict[str, Any], *, for_training: bool):
    """Nalozi osnovni model glede na napravo in natancnost iz konfiguracije.

    4bit (QLoRA) zahteva bitsandbytes in torej NVIDIA. Na Apple Silicon
    uporabi bf16 - adapter je majhen, osnovni model ostane zamrznjen.
    """
    import torch
    from transformers import AutoConfig

    mcfg = cfg["model"]
    device = resolve_device(mcfg.get("device", "auto"))
    precision = mcfg.get("precision", "bf16")

    if precision == "4bit" and device != "cuda":
        raise RuntimeError(
            f"precision: 4bit zahteva CUDA (bitsandbytes), zaznana naprava je '{device}'.\n"
            "Na Apple Silicon nastavi precision: bf16 v config.yaml."
        )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.bfloat16)

    kwargs: dict[str, Any] = {"dtype": dtype}
    if precision == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = None

    model = _auto_load(mcfg["id"], kwargs)

    if kwargs["device_map"] is None:
        model = model.to(device)

    adapter = mcfg.get("adapter")
    if adapter:
        from peft import PeftModel

        adapter_path = ROOT / adapter if not Path(adapter).is_absolute() else Path(adapter)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        if not for_training:
            model.eval()

    _ = AutoConfig  # ohranjeno za berljivost uvozov
    return model, device


def _auto_load(model_id: str, kwargs: dict[str, Any]):
    """Ministral 3 je multimodalen, zato razred ni vedno AutoModelForCausalLM.

    Poskusimo multimodalni razred, nato navadnega. Ce oba padeta, uporabnik
    dobi obe napaki naenkrat namesto zavajajoce ene same.
    """
    import transformers

    errors = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model_id, **kwargs)
        except Exception as exc:  # noqa: BLE001 - namerno sirok, oboje porocamo
            errors.append(f"  {cls_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Modela '{model_id}' ni bilo mogoce nalozit z nobenim razredom:\n" + "\n".join(errors)
    )


def resolve_tools(record: dict[str, Any], registry: ToolRegistry) -> list[dict[str, Any]]:
    """Primer lahko navede 'tools' neposredno ali 'tool_groups' (iz registra)."""
    if "tools" in record:
        return record["tools"]
    if "tool_groups" in record:
        return registry.select(record["tool_groups"])
    return registry.select()


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Poenoti obliko tool_calls: arguments kot slovar (ne kot niz).

    Razlicni predlogi za klepet pricakujejo eno ali drugo; slovar je oblika, ki
    jo Mistralovi predlogi serializirajo sami.
    """
    out = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("tool_calls"):
            calls = []
            for call in msg["tool_calls"]:
                call = json.loads(json.dumps(call))  # globoka kopija
                args = call["function"].get("arguments")
                if isinstance(args, str):
                    call["function"]["arguments"] = json.loads(args)
                calls.append(call)
            msg["tool_calls"] = calls
        out.append(msg)
    return out
