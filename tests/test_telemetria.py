"""Testes unitários para o módulo de telemetria de hardware do VIPER."""
import time
from src.telemetria import capturar_desempenho_maquina, formatar_relatorio_hardware, Temporizador


def test_capturar_desempenho_maquina():
    stats = capturar_desempenho_maquina()
    assert "cpu_percent" in stats
    assert "ram_usada_mb" in stats
    assert "ram_total_mb" in stats
    assert "gpu" in stats
    assert stats["ram_total_mb"] > 0


def test_formatar_relatorio_hardware():
    relatorio = formatar_relatorio_hardware()
    assert "HARDWARE & COMPUTAÇÃO" in relatorio
    assert "Uso de CPU Host" in relatorio
    assert "Memória RAM Usada" in relatorio


def test_temporizador():
    timer = Temporizador()
    time.sleep(0.01)
    duracao_ms = timer.parar()
    assert duracao_ms >= 5.0
    assert timer.duracao_s() >= 0.005
