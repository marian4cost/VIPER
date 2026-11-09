import os
import time
import psutil
import torch


def capturar_desempenho_maquina():
    """
    Captura métricas instantâneas de desempenho da máquina:
    - Uso de CPU (%)
    - Memória RAM (Usada em MB, Total em MB e Percentual)
    - GPU / VRAM (VRAM Alocada em MB, Pico de VRAM em MB, Nome da GPU) se disponível.
    """
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    ram_usada_mb = mem.used / (1024 ** 2)
    ram_total_mb = mem.total / (1024 ** 2)
    ram_pct = mem.percent

    gpu_info = {
        "disponivel": False,
        "dispositivo": "CPU",
        "vram_alocada_mb": 0.0,
        "vram_pico_mb": 0.0,
        "vram_reservada_mb": 0.0
    }

    if torch.cuda.is_available():
        gpu_info["disponivel"] = True
        gpu_info["dispositivo"] = torch.cuda.get_device_name(0)
        gpu_info["vram_alocada_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_info["vram_pico_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_info["vram_reservada_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)

    return {
        "cpu_percent": cpu_pct,
        "ram_usada_mb": ram_usada_mb,
        "ram_total_mb": ram_total_mb,
        "ram_percent": ram_pct,
        "gpu": gpu_info
    }


def formatar_relatorio_hardware(hw_stats=None):
    """
    Retorna uma string formatada contendo as métricas de hardware atuais.
    """
    if hw_stats is None:
        hw_stats = capturar_desempenho_maquina()

    gpu = hw_stats["gpu"]
    if gpu["disponivel"]:
        vram_str = (
            f"{gpu['dispositivo']} | Alocada: {gpu['vram_alocada_mb']:.2f} MB "
            f"| Pico: {gpu['vram_pico_mb']:.2f} MB | Reservada: {gpu['vram_reservada_mb']:.2f} MB"
        )
    else:
        vram_str = "N/A (Execução em CPU)"

    linhas = [
        "💻 HARDWARE & COMPUTAÇÃO:",
        f" ├─ Uso de CPU Host: {hw_stats['cpu_percent']:.1f}%",
        f" ├─ Memória RAM Usada: {hw_stats['ram_usada_mb']:.2f} MB / {hw_stats['ram_total_mb']:.2f} MB ({hw_stats['ram_percent']:.1f}%)",
        f" └─ GPU & VRAM: {vram_str}"
    ]
    return "\n".join(linhas)


class Temporizador:
    """Helper de tempo para medir trechos de código com time.perf_counter()."""
    def __init__(self):
        self.inicio = time.perf_counter()
        self.fim = None

    def parar(self):
        self.fim = time.perf_counter()
        return self.duracao_ms()

    def duracao_s(self):
        f = self.fim if self.fim is not None else time.perf_counter()
        return f - self.inicio

    def duracao_ms(self):
        return self.duracao_s() * 1000.0
