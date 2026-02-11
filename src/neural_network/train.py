"""Entry-point de antrenare (Etapa 5/6).

Repository-ul contine scripturi de antrenare per tip de obiect (ex: `train_chair.py`).
Acest fisier exista ca entrypoint stabil (`train.py`) pentru structura de predare,
fara a schimba implementarea existenta.

Implicit ruleaza antrenarea pentru `chair`.
"""

from __future__ import annotations

from src.neural_network.train_chair import train


if __name__ == "__main__":
    train()
