"""
Ponto de entrada do Projeto VIPER - Pipeline Integrado.
Executa a detecção de veículos, localização de placas, alinhamento geométrico,
binarização e leitura OCR com validação heurística Mercosul/antigo e telemetria de hardware.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.main import executar_pipeline_completo

if __name__ == "__main__":
    executar_pipeline_completo()
