"""Testes unitários para o módulo de leitura e validação de placas do VIPER."""
import pytest
from src.reading import _validar_string_7char, corrigir_leitura_placa


def test_validar_placa_mercosul_valida():
    placa, padrao = _validar_string_7char("BRA2E19")
    assert placa == "BRA2E19"
    assert padrao == "Mercosul"


def test_validar_placa_antiga_valida():
    placa, padrao = _validar_string_7char("ABC1234")
    assert placa == "ABC1234"
    assert padrao == "Padrão Antigo"


def test_correcao_letras_numeros_mercosul():
    # 'O' na posição numérica deve virar '0', '1' na posição de letra deve virar 'I'
    placa, padrao = corrigir_leitura_placa("BRA2E1O")
    assert placa == "BRA2E10"
    assert padrao == "Mercosul"


def test_correcao_ruido_8_caracteres():
    # Ruído à esquerda (ex: caractere extra no início)
    placa, padrao = corrigir_leitura_placa("XABC1234")
    assert placa == "ABC1234"
    assert padrao == "Padrão Antigo"


def test_placa_invalida_retorna_incerto():
    placa, padrao = corrigir_leitura_placa("INVALIDA123")
    assert padrao == "Padrão incerto"
