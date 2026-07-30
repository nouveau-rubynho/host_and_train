"""Lokalni streznik, zdruzljiv z OpenAI /v1/chat/completions.

    python -m src.serve

Adapter se nalozi POLEG osnovnega modela (peft), brez zlivanja in brez
pretvorbe v GGUF. Zamenjas ga tako, da spremenis model.adapter v config.yaml.

Za produkcijo na NVIDIA glej README, razdelek "vLLM".
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .common import ROOT, ToolRegistry, load_config, load_model, load_tokenizer, normalize_messages

# Ministral 3 izpise klic orodja natanko takole (preverjeno na predlogi modela):
#   [TOOL_CALLS]rpi.gpio_set[ARGS]{"host": "garaza", "pin": 17, "state": "high"}
# Vec klicev je lahko zaporedoma. Zato ne razclenjujemo JSON polja kot celote,
# ampak beremo par (ime, argumenti) za vsakim oznacevalnikom posebej.
TOOL_CALL_RE = re.compile(r"\[TOOL_CALLS\]\s*(?P<name>[\w.\-]+)\s*\[ARGS\]\s*")

# Zakljucni posebni tokeni, ki jih ne zelimo v besedilu odgovora.
TRAILING_SPECIAL = ("</s>", "[/INST]")


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Iz generiranega besedila izlusci klice orodij.

    Vrne (besedilo ali None, seznam klicev). Ce razclenitev ne uspe, besedilo
    vrnemo nedotaknjeno - bolje surov izpis kot tiho pozrt odgovor.
    """
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    plain_parts: list[str] = []
    pos = 0

    while True:
        match = TOOL_CALL_RE.search(text, pos)
        if not match:
            plain_parts.append(text[pos:])
            break

        plain_parts.append(text[pos:match.start()])
        try:
            arguments, end = decoder.raw_decode(text, match.end())
        except json.JSONDecodeError:
            # Odrezan ali pokvarjen JSON - ne ugibaj, vrni surovo besedilo.
            plain_parts.append(text[match.start():])
            break

        calls.append(
            {
                "id": uuid.uuid4().hex[:9],
                "type": "function",
                "function": {
                    "name": match.group("name"),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
        pos = end

    content = "".join(plain_parts)
    for token in TRAILING_SPECIAL:
        content = content.replace(token, "")
    return (content.strip() or None), calls


class Engine:
    """Model, tokenizator in generiranje na enem mestu."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.tokenizer = load_tokenizer(cfg["model"]["id"])
        self.registry = ToolRegistry.load(cfg["data"]["tool_registry"])
        print(f"Nalagam model {cfg['model']['id']} ...")
        self.model, self.device = load_model(cfg, for_training=False)
        adapter = cfg["model"].get("adapter")
        print(f"Naprava: {self.device}   adapter: {adapter or 'brez (osnovni model)'}")

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        import torch

        scfg = self.cfg["serve"]
        max_new_tokens = max_new_tokens or scfg["max_new_tokens"]
        temperature = scfg["temperature"] if temperature is None else temperature

        # return_dict=True: v transformers 5 apply_chat_template sicer vrne
        # BatchEncoding in ne tenzorja, zato generate() pade z AttributeError.
        inputs = self.tokenizer.apply_chat_template(
            normalize_messages(messages),
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # skip_special_tokens=False: [TOOL_CALLS] in [ARGS] STA posebna tokena,
        # in brez njiju klica orodja ni mogoce prepoznati.
        completion = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=False)
        return parse_tool_calls(completion)


# ---------------------------------------------------------------------------
# Sled: vsak pogovor se zapise v ISTI shemi, kot jo bere trenazer.
# To je vir bodocih ucnih podatkov - glej README, "Podatki iz uporabe".


def log_trace(cfg: dict[str, Any], messages, tools, reply_message) -> None:
    if not cfg["serve"].get("log_traces"):
        return
    path = ROOT / cfg["serve"]["trace_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "_source": "serve",
        "_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_model": cfg["model"]["id"],
        "_adapter": cfg["model"].get("adapter"),
        "tools": tools,
        "messages": list(messages) + [reply_message],
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Podmnozica OpenAI /v1/chat/completions + bliznjica 'tool_groups'.

    Razred MORA biti na ravni modula: zaradi `from __future__ import annotations`
    so anotacije nizi, FastAPI pa jih razresuje v globalnem imenskem prostoru.
    Ce je razred lokalen v funkciji, ga ne najde in telo zahtevka razglasi za
    manjkajoc parameter poizvedbe.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_groups: list[str] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    model: str | None = None


def build_app(cfg: dict[str, Any]):
    from fastapi import FastAPI

    app = FastAPI(title="Silent Guardian")
    engine = Engine(cfg)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": cfg["model"]["id"],
            "adapter": cfg["model"].get("adapter"),
            "device": engine.device,
            "tool_groups": engine.registry.groups(),
        }

    @app.get("/v1/tools")
    def list_tools(groups: str | None = None):
        selected = groups.split(",") if groups else None
        return {"tools": engine.registry.select(selected)}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        tools = req.tools
        if tools is None and req.tool_groups is not None:
            tools = engine.registry.select(req.tool_groups)

        content, tool_calls = engine.generate(
            req.messages, tools, req.max_tokens, req.temperature
        )

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls

        log_trace(cfg, req.messages, tools, message)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": cfg["model"]["id"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Lokalni OpenAI-zdruzljivi streznik")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    import uvicorn

    cfg = load_config(args.config)
    host = args.host or cfg["serve"]["host"]
    port = args.port or cfg["serve"]["port"]

    print(f"Streznik: http://{host}:{port}   (dokumentacija: /docs)")
    uvicorn.run(build_app(cfg), host=host, port=port)


if __name__ == "__main__":
    main()
