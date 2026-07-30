"""Preprost pogovor v terminalu - za hitro preverjanje, ali model klice orodja.

    python -m src.chat --groups rpi
    python -m src.chat --groups rpi --prompt "Prizgi luc v garazi."

Naprave niso prikljucene, zato odgovor orodja vtipkas sam (ali pritisnes Enter
za privzeti {"ok": true}). Tako vidis celotno zanko: klic -> rezultat -> povzetek.
"""

from __future__ import annotations

import argparse
import json

from .common import load_config
from .serve import Engine, log_trace

SYSTEM_PROMPT = (
    "Si Silent Guardian, krmilnik pametnega doma. Kličeš orodja za upravljanje naprav. "
    "Nikoli ne ugibaj vrednosti obveznih parametrov - če manjkajo, vprašaj uporabnika."
)

BOLD, CYAN, YELLOW, RESET = "\033[1m", "\033[36m", "\033[33m", "\033[0m"


def run_turn(engine: Engine, cfg, messages, tools) -> None:
    """Generira, in ce pride klic orodja, pobere rezultat ter nadaljuje."""
    while True:
        content, tool_calls = engine.generate(messages, tools)

        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)

        if content:
            print(f"{CYAN}model:{RESET} {content}")

        if not tool_calls:
            log_trace(cfg, messages[:-1], tools, message)
            return

        for call in tool_calls:
            fn = call["function"]
            print(f"{YELLOW}klic orodja:{RESET} {fn['name']}({fn['arguments']})")
            raw = input("  rezultat (JSON, Enter = {\"ok\": true}): ").strip()
            messages.append(
                {
                    "role": "tool",
                    "name": fn["name"],
                    "tool_call_id": call["id"],
                    "content": raw or json.dumps({"ok": True}),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pogovor z modelom v terminalu")
    ap.add_argument("--config", default=None)
    ap.add_argument("--groups", default=None, help="skupine orodij, npr. 'rpi,pc' (privzeto vse)")
    ap.add_argument("--prompt", default=None, help="enkraten vnos namesto interaktivnega nacina")
    ap.add_argument("--system", default=SYSTEM_PROMPT)
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = Engine(cfg)

    groups = args.groups.split(",") if args.groups else None
    tools = engine.registry.select(groups)
    print(f"\nOrodja v kontekstu ({len(tools)}): {', '.join(t['function']['name'] for t in tools)}\n")

    base = [{"role": "system", "content": args.system}]

    if args.prompt:
        run_turn(engine, cfg, base + [{"role": "user", "content": args.prompt}], tools)
        return

    messages = list(base)
    print(f"{BOLD}Vpisi sporocilo. Prazna vrstica konca pogovor.{RESET}\n")
    while True:
        try:
            user = input(f"{BOLD}ti:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            return
        messages.append({"role": "user", "content": user})
        run_turn(engine, cfg, messages, tools)
        print()


if __name__ == "__main__":
    main()
