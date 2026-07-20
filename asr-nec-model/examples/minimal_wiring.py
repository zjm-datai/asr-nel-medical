from __future__ import annotations

from asr_nec_model import EMPTY, EC, SEP


def main() -> None:
    print("Minimal wiring example for ASR NEC model.")
    print(f"Special tokens: {EMPTY}, {SEP}, {EC}")
    print("Next: load whisper weights, build datasets, and call training/inference helpers.")


if __name__ == "__main__":
    main()

