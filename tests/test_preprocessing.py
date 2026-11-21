"""Testes unitários para funções de pré-processamento de imagem do VIPER."""
import numpy as np
import pytest
from src.reading import recortar_roi_suave


def test_recortar_roi_suave_dimensions():
    # Cria uma imagem de teste 100x200
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped = recortar_roi_suave(dummy_img, pct_pad=0.05)
    
    # Com 5% de margem: h=100 -> pad=5 -> y1=5, y2=95 (altura 90)
    # w=200 -> pad=10 -> x1=10, x2=190 (largura 180)
    assert cropped.shape[0] == 90
    assert cropped.shape[1] == 180
    assert cropped.shape[2] == 3
