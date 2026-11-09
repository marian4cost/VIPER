"""
Ponto de entrada para validação e benchmark de latência/FPS do modelo YOLOv8 de placas.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.validadeYOLO import main

if __name__ == "__main__":
    main()
