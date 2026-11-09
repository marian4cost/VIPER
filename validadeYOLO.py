"""
Ponto de entrada para validação e benchmark de latência/FPS do modelo YOLOv8 de placas.
"""
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from validadeYOLO import main

if __name__ == "__main__":
    main()
