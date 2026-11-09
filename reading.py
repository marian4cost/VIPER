"""
Ponto de entrada para testes de OCR e regras heurísticas VIPER de placas Mercosul e padrão antigo.
"""
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from reading import executar_testes_leitura

if __name__ == "__main__":
    executar_testes_leitura()
