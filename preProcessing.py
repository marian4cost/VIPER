"""
Ponto de entrada para testes de pré-processamento de imagens de veículos e placas (warp perspective + Otsu).
"""
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preProcessing import executar_testes_e_gerar_relatorio

if __name__ == "__main__":
    executar_testes_e_gerar_relatorio()
