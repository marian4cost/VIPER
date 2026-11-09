"""
Ponto de entrada para geração do relatório executivo e técnico do Projeto VIPER em PDF.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.gerar_pdf_relatorio import criar_relatorio_pdf

if __name__ == "__main__":
    criar_relatorio_pdf()
