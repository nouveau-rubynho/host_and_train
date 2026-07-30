"""Gradnja ucne mnozice: predloga za klepet, maskiranje izgube, zbiralnik.

To je jedro projekta. Ce je tukaj napaka, se ucenje ne bo pritozilo - model se
bo tiho ucil narobe. Zato: `python -m src.dataset inspect` in poglej z ocmi.

Maskiranje: izgubo racunamo SAMO na tokenih, ki jih generira asistent
(vkljucno s klici orodij). Sistemski poziv, uporabnikovi vnosi in odgovori
orodij so zamaskirani z -100.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .common import ROOT, ToolRegistry, load_config, load_tokenizer, normalize_messages, resolve_tools

IGNORE_INDEX = -100


class DatasetError(ValueError):
    """Napaka v podatkih, ne v kodi - sporocilo je namenjeno uporabniku."""


# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name}:{lineno} ni veljaven JSON: {exc}") from exc


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    n_trained: int
    n_total: int
    lineno: int

    @property
    def trained_ratio(self) -> float:
        return self.n_trained / max(self.n_total, 1)


def build_example(
    record: dict[str, Any],
    tokenizer,
    registry: ToolRegistry,
    lineno: int = 0,
) -> Example:
    """Zgradi en ucni primer z masko na asistentovih zavojih.

    Postopek: za vsak asistentov zavoj i izrisemo predpono (zavoji do i, z
    generacijskim pozivom) in celoto (zavoji do vkljucno i). Razlika v dolzini
    tokenov je natanko tisto, kar se mora model nauciti generirati.
    """
    messages = normalize_messages(record["messages"])
    tools = resolve_tools(record, registry)

    if not messages:
        raise DatasetError(f"vrstica {lineno}: prazen 'messages'")
    if messages[-1]["role"] != "assistant":
        raise DatasetError(
            f"vrstica {lineno}: zadnji zavoj mora biti 'assistant', je '{messages[-1]['role']}'. "
            "Primer, ki se ne konca z asistentom, nima cesa uciti."
        )

    if messages[0]["role"] == "assistant":
        raise DatasetError(
            f"vrstica {lineno}: pogovor se zacne z 'assistant' - "
            "primer mora imeti pred odgovorom vsaj sistemski poziv ali uporabnikov vnos"
        )

    def render(msgs: list[dict], add_generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            msgs,
            tools=tools or None,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    # Masko racunamo po ZNAKIH, ne po tokenih. Razlog: ce tokeniziramo predpono
    # posebej in celoto posebej, se lahko tokeni na meji zlijejo drugace in
    # dolzina predpone ni veljaven zamik. Znakovni odmiki tega problema nimajo.
    full_text = render(messages, add_generation_prompt=False)

    encoding = tokenizer(
        full_text,
        add_special_tokens=False,  # predloga je BOS ze vstavila
        return_offsets_mapping=True,
    )
    full_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    labels = [IGNORE_INDEX] * len(full_ids)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        prefix_text = render(messages[:i], add_generation_prompt=True)
        upto_text = render(messages[: i + 1], add_generation_prompt=False)

        # Predloga mora biti stabilna po predponah, sicer bi maskirali napacen
        # del besedila. Raje glasno pademo kot da tiho ucimo smeti.
        if not full_text.startswith(prefix_text) or not full_text.startswith(upto_text):
            raise DatasetError(
                f"vrstica {lineno}, zavoj {i}: predloga za klepet ni stabilna po predponah - "
                "maske ni mogoce zanesljivo izracunati. Preveri razlicico transformers "
                "oziroma chat_template modela."
            )

        start_char, end_char = len(prefix_text), len(upto_text)
        for pos, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= tok_start:  # posebni tokeni brez znakovnega razpona
                continue
            if start_char <= tok_start and tok_end <= end_char:
                labels[pos] = full_ids[pos]

    n_trained = sum(1 for x in labels if x != IGNORE_INDEX)
    if n_trained == 0:
        raise DatasetError(f"vrstica {lineno}: maska je prazna - model se ne bi naucil nicesar")

    return Example(
        input_ids=full_ids,
        labels=labels,
        n_trained=n_trained,
        n_total=len(full_ids),
        lineno=lineno,
    )


def build_dataset(cfg: dict[str, Any], tokenizer, registry: ToolRegistry) -> list[Example]:
    """Zgradi vse primere. Predolge ZAVRZE (ne obreze) in to izpise.

    Obrezan klic orodja je pokvarjen ucni primer - model bi se ucil generirati
    nedokoncan JSON. Zato raje izgubimo primer kot da pokvarimo ucenje.
    """
    max_len = cfg["data"]["max_seq_len"]
    examples, dropped = [], []

    for lineno, record in read_jsonl(cfg["data"]["seed_file"]):
        ex = build_example(record, tokenizer, registry, lineno)
        if ex.n_total > max_len:
            dropped.append((lineno, ex.n_total))
            continue
        examples.append(ex)

    if dropped:
        print(f"\n[opozorilo] zavrzenih {len(dropped)} primerov, daljsih od max_seq_len={max_len}:")
        for lineno, n in dropped:
            print(f"  vrstica {lineno}: {n} tokenov")
        print("  -> povecaj data.max_seq_len ali skrajsaj primere\n")

    if not examples:
        raise DatasetError("po filtriranju ni ostal noben primer")
    return examples


# ---------------------------------------------------------------------------


class Collator:
    """Dopolni serijo do najdaljsega primera. Izgube na dopolnilu ne racunamo."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[Example]) -> dict:
        import torch

        width = max(len(b.input_ids) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            pad = width - len(b.input_ids)
            input_ids.append(b.input_ids + [self.pad_token_id] * pad)
            labels.append(b.labels + [IGNORE_INDEX] * pad)
            attention.append([1] * len(b.input_ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# CLI: inspect / stats

GREEN, DIM, RESET = "\033[42m\033[30m", "\033[2m", "\033[0m"


def render_mask(ex: Example, tokenizer) -> str:
    """Izpise primer tako, da je maska VIDNA.

    Zeleno = model se uci generirati. Sivo = samo kontekst.
    """
    out = []
    for tok_id, label in zip(ex.input_ids, ex.labels):
        piece = tokenizer.decode([tok_id])
        out.append(f"{GREEN}{piece}{RESET}" if label != IGNORE_INDEX else f"{DIM}{piece}{RESET}")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gradnja in pregled ucne mnozice")
    ap.add_argument("command", choices=["inspect", "stats"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--index", type=int, default=0, help="kateri primer prikazati (inspect)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg["model"]["id"])
    registry = ToolRegistry.load(cfg["data"]["tool_registry"])
    examples = build_dataset(cfg, tokenizer, registry)

    if args.command == "stats":
        lengths = sorted(e.n_total for e in examples)
        trained = sum(e.n_trained for e in examples)
        total = sum(e.n_total for e in examples)
        print(f"primerov:        {len(examples)}")
        print(f"tokenov skupaj:  {total}")
        print(f"od tega ucenih:  {trained} ({trained / total:.1%})")
        print(f"dolzina min/med/max: {lengths[0]} / {lengths[len(lengths) // 2]} / {lengths[-1]}")
        print(f"max_seq_len:     {cfg['data']['max_seq_len']}")
        return

    ex = examples[args.index]
    print(f"--- primer {args.index} (vrstica {ex.lineno}) ---")
    print(f"{GREEN}zeleno{RESET} = model se uci generirati   {DIM}sivo{RESET} = samo kontekst\n")
    print(render_mask(ex, tokenizer))
    print(f"\n{ex.n_trained}/{ex.n_total} tokenov v izgubi ({ex.trained_ratio:.1%})")


if __name__ == "__main__":
    main()
