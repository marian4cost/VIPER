"""
Ponto de entrada para geração do relatório executivo e técnico do Projeto VIPER em PDF.
"""
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gerar_pdf_relatorio import criar_relatorio_pdf

if __name__ == "__main__":
    criar_relatorio_pdf()
