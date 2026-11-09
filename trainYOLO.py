"""
Ponto de entrada para treinamento do modelo YOLOv8 de detecção de placas veiculares com telemetria de hardware.
"""
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from trainYOLO import main

if __name__ == "__main__":
    main()
