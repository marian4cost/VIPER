"""
Ponto de entrada para treinamento do modelo YOLOv8 de detecção de placas veiculares com telemetria de hardware.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.trainYOLO import main

if __name__ == "__main__":
    main()
