"""Preverjanje ucne mnozice PRED ucenjem. Ne potrebuje modela ne interneta.

Zazeni to vsakic, ko dodas primere:

    python -m src.validate

Napake so oznacene z NAPAKA (ucenje ne bo delovalo) ali OPOZORILO (bo delovalo,
a verjetno ni tisto, kar si hotel).
"""

from __future__ import annotations

import argparse
import re
from typing import Any

from .common import ToolRegistry, load_config, resolve_tools
from .dataset import read_jsonl

VALID_ROLES = {"system", "user", "assistant", "tool"}

# Nekatere Mistralove predloge zahtevajo tool_call_id dolzine 9 iz [a-zA-Z0-9].
# Ni univerzalno, zato je to opozorilo in ne napaka - a ceneje je se drzati.
TOOL_CALL_ID_RE = re.compile(r"^[a-zA-Z0-9]{9}$")

JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, msg: str) -> None:
        self.errors.append(f"NAPAKA    {where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"OPOZORILO {where}: {msg}")


def check_arguments(where: str, name: str, args: Any, spec: dict[str, Any], rep: Report) -> None:
    if not isinstance(args, dict):
        rep.error(where, f"'{name}': arguments morajo biti objekt, so {type(args).__name__}")
        return

    params = spec.get("parameters", {})
    props = params.get("properties", {})
    required = params.get("required", [])

    for key in required:
        if key not in args:
            rep.error(where, f"'{name}': manjka obvezni parameter '{key}'")

    for key, value in args.items():
        if key not in props:
            rep.error(where, f"'{name}': neznan parameter '{key}' (ni v shemi orodja)")
            continue
        expected = props[key].get("type")
        py_type = JSON_TYPES.get(expected)
        # bool je podrazred int v Pythonu - loci ju, sicer true prestane kot integer
        if py_type and (not isinstance(value, py_type) or (expected != "boolean" and isinstance(value, bool))):
            rep.error(where, f"'{name}': parameter '{key}' naj bo {expected}, je {type(value).__name__}")
        enum = props[key].get("enum")
        if enum and value not in enum:
            rep.error(where, f"'{name}': '{key}' = {value!r} ni v dovoljenih vrednostih {enum}")


def validate_record(record: dict[str, Any], lineno: int, registry: ToolRegistry, rep: Report) -> None:
    where = f"vrstica {lineno}"

    if "messages" not in record:
        rep.error(where, "manjka polje 'messages'")
        return
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        rep.error(where, "'messages' mora biti neprazen seznam")
        return

    try:
        tools = resolve_tools(record, registry)
    except Exception as exc:  # noqa: BLE001
        rep.error(where, f"orodij ni bilo mogoce razresiti: {exc}")
        return

    offered = {t["function"]["name"] for t in tools}
    if "tool_groups" in record:
        unknown_groups = set(record["tool_groups"]) - set(registry.groups())
        if unknown_groups:
            rep.error(where, f"neznane skupine orodij: {sorted(unknown_groups)}")

    if messages[-1]["role"] != "assistant":
        rep.error(where, f"zadnji zavoj je '{messages[-1]['role']}', mora biti 'assistant'")

    pending: dict[str, str] = {}  # tool_call_id -> ime orodja
    has_assistant_content = False

    for i, msg in enumerate(messages):
        mwhere = f"{where}, zavoj {i}"
        role = msg.get("role")
        if role not in VALID_ROLES:
            rep.error(mwhere, f"neveljavna vloga '{role}'")
            continue

        if role == "system" and i != 0:
            rep.warn(mwhere, "sistemski poziv ni na zacetku pogovora")

        if role == "assistant":
            calls = msg.get("tool_calls")
            if not calls and not msg.get("content"):
                rep.error(mwhere, "asistentov zavoj je prazen (ne content ne tool_calls)")
            if calls:
                for call in calls:
                    cid = call.get("id", "")
                    fn = call.get("function", {})
                    name = fn.get("name")
                    if not TOOL_CALL_ID_RE.match(str(cid)):
                        rep.warn(mwhere, f"tool_call_id '{cid}' ni 9 alfanumericnih znakov")
                    if name not in registry.names():
                        rep.error(mwhere, f"orodje '{name}' ne obstaja v registru")
                        continue
                    if name not in offered:
                        rep.error(
                            mwhere,
                            f"orodje '{name}' ni med ponujenimi v tem primeru - "
                            "model se uci klicati nekaj, cesar ne vidi v kontekstu",
                        )
                    check_arguments(mwhere, name, fn.get("arguments"), registry.spec(name), rep)
                    pending[cid] = name
            else:
                has_assistant_content = True

        elif role == "tool":
            cid = msg.get("tool_call_id")
            if cid not in pending:
                rep.error(mwhere, f"odgovor orodja z id '{cid}' se ne ujema z nobenim klicem pred njim")
            else:
                if msg.get("name") and msg["name"] != pending[cid]:
                    rep.error(mwhere, f"ime '{msg['name']}' se ne ujema s klicem '{pending[cid]}'")
                pending.pop(cid)
            if not msg.get("content"):
                rep.error(mwhere, "odgovor orodja je prazen")

    if pending:
        rep.error(where, f"klici orodij brez odgovora: {sorted(pending)}")
    if not has_assistant_content:
        rep.warn(
            where,
            "primer nima nobenega besedilnega odgovora asistenta - "
            "model se uci samo klicati orodja, ne pa tudi porocati uporabniku",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Preveri ucno mnozico")
    ap.add_argument("--config", default=None)
    ap.add_argument("--file", default=None, help="preglej drugo datoteko namesto data.seed_file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    registry = ToolRegistry.load(cfg["data"]["tool_registry"])
    path = args.file or cfg["data"]["seed_file"]

    rep = Report()
    count = 0
    for lineno, record in read_jsonl(path):
        count += 1
        validate_record(record, lineno, registry, rep)

    print(f"Pregledanih primerov: {count}   ({path})")
    print(f"Skupine orodij v registru: {', '.join(registry.groups())}\n")

    for line in rep.warnings:
        print(line)
    for line in rep.errors:
        print(line)

    if not rep.errors and not rep.warnings:
        print("Vse v redu.")
    elif not rep.errors:
        print(f"\n{len(rep.warnings)} opozoril, 0 napak - ucenje bo delovalo.")
    else:
        print(f"\n{len(rep.errors)} napak, {len(rep.warnings)} opozoril - popravi pred ucenjem.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
