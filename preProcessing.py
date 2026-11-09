"""
Ponto de entrada para testes de pré-processamento de imagens de veículos e placas (warp perspective + Otsu).
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preProcessing import executar_testes_e_gerar_relatorio

if __name__ == "__main__":
    executar_testes_e_gerar_relatorio()
