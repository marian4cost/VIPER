"""Configuração de testes do VIPER com mocks automáticos para módulos pesados ausentes."""
import sys
from unittest.mock import MagicMock

# Módulos pesados opcionais que podem não estar instalados no ambiente de CI/teste básico
OPTIONAL_MODULES = [
    "cv2",
    "easyocr",
    "torch",
    "torchvision",
    "ultralytics",
    "reportlab",
    "reportlab.lib",
    "reportlab.lib.pagesizes",
    "reportlab.lib.styles",
    "reportlab.lib.colors",
    "reportlab.platypus",
]

for mod in OPTIONAL_MODULES:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            mock = MagicMock()
            if mod == "torch":
                mock.cuda.is_available.return_value = False
            sys.modules[mod] = mock
