#!/usr/bin/env python3
"""Preveri okolje PRED vsem ostalim.

    python check_env.py

Vecina zacetnih tezav (posebej na RTX 50xx) se pokaze tukaj kot jasna napaka,
namesto kasneje kot nerazumljiv sesutje sredi ucenja.
"""

from __future__ import annotations

import importlib
import platform
import sys

OK, BAD, WARN = "[v]", "[x]", "[!]"


def line(status: str, text: str) -> None:
    print(f"{status} {text}")


def check_package(name: str, minimum: str | None = None, required: bool = True) -> str | None:
    try:
        mod = importlib.import_module(name)
    except ImportError:
        line(BAD if required else WARN, f"{name}: ni namescen")
        return None
    version = getattr(mod, "__version__", "?")
    line(OK, f"{name}: {version}")
    return version


def main() -> int:
    problems = 0

    print(f"Python:   {sys.version.split()[0]}")
    print(f"Platforma: {platform.system()} {platform.machine()}\n")

    if sys.version_info < (3, 10):
        line(BAD, "Python mora biti vsaj 3.10")
        problems += 1

    print("--- paketi ---")
    for pkg in ("torch", "transformers", "peft", "accelerate", "yaml", "fastapi"):
        if check_package(pkg) is None:
            problems += 1
    check_package("bitsandbytes", required=False)

    print("\n--- strojna oprema ---")
    try:
        import torch
    except ImportError:
        line(BAD, "torch ni namescen - nadaljnjih preverb ni mogoce izvesti")
        return 1

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        line(OK, f"CUDA: {name}  (sm_{major}{minor}, {total:.1f} GB VRAM, torch cuda {torch.version.cuda})")

        # Blackwell (RTX 50xx) = sm_120. Stari torch nima jeder zanj in pade
        # z 'no kernel image is available for execution on the device'.
        if major >= 12:
            supported = torch.cuda.get_arch_list()
            if not any(f"sm_{major}{minor}" in a for a in supported):
                line(
                    BAD,
                    f"ta torch NE podpira sm_{major}{minor} (zna: {', '.join(supported)}).\n"
                    "    Namesti: pip install torch --index-url https://download.pytorch.org/whl/cu128",
                )
                problems += 1

        if total < 9:
            line(
                WARN,
                f"{total:.0f} GB VRAM: za ucenje uporabi 3B model, precision: 4bit "
                "in data.max_seq_len <= 2048",
            )

        try:
            import bitsandbytes  # noqa: F401
            x = torch.zeros(1, device="cuda")
            _ = x + 1
            line(OK, "bitsandbytes je na voljo -> precision: 4bit (QLoRA) deluje")
        except ImportError:
            line(WARN, "bitsandbytes ni namescen -> precision: 4bit ne bo delal")
        except Exception as exc:  # noqa: BLE001
            line(BAD, f"CUDA je vidna, a racun ne deluje: {exc}")
            problems += 1

    elif torch.backends.mps.is_available():
        line(OK, "Apple Silicon (mps)")
        line(WARN, "bitsandbytes na mps ne dela -> v config.yaml nastavi precision: bf16")
    else:
        line(WARN, "ne CUDA ne mps - vse bo teklo na CPU in bo zelo pocasno")

    print()
    if problems:
        print(f"{problems} tezav. Popravi jih pred nadaljevanjem.")
        return 1
    print("Okolje je pripravljeno.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
