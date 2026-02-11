"""Script de optimizare / tuning (Etapa 6).

In acest proiect, compararea configuratiilor (arhitecturi MLP) este implementata
in `compare_architectures.py`. Acest fisier exista ca entrypoint stabil pentru
structura de predare (optimize.py), fara a schimba logica existenta.
"""

from __future__ import annotations

from src.neural_network.compare_architectures import main


if __name__ == "__main__":
    main()
