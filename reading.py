"""
Ponto de entrada para testes de OCR e regras heurísticas VIPER de placas Mercosul e padrão antigo.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.reading import main

if __name__ == "__main__":
    main()
