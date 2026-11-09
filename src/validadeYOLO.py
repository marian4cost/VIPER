import os
import sys
import time
import torch
from ultralytics import YOLO

# Permite importar o módulo telemetria quando executado da raiz ou da pasta novos_codigos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from telemetria import capturar_desempenho_maquina, formatar_relatorio_hardware, Temporizador


# ================================
# Helper para Log Duplo (Terminal + TXT)
# ================================
class DualLogger:
    def __init__(self, filepath="relatorio_validacao.txt"):
        self.filepath = filepath
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("=== RELATÓRIO OFICIAL DE VALIDAÇÃO E LATÊNCIA ===\n\n")

    def log(self, text):
        print(text)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def encontrar_pesos_validos(root_dir):
    """Localiza o arquivo de pesos best.pt lidando com variações de caminho de execução."""
    candidatos = [
        os.path.join(root_dir, "runs", "detect", "placa_detector", "weights", "best.pt"),
        os.path.join(root_dir, "runs", "detect", "runs", "detect", "placa_detector", "weights", "best.pt"),
        os.path.abspath("runs/detect/placa_detector/weights/best.pt"),
        os.path.abspath("runs/detect/runs/detect/placa_detector/weights/best.pt")
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c
    # Retorna o primeiro caminho como padrão caso ainda não tenha treinado
    return candidatos[0]


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    relatorio_path = os.path.join(root_dir, "relatorios", "validacao_e_metricas", "relatorio_validacao.txt")
    os.makedirs(os.path.dirname(relatorio_path), exist_ok=True)
    logger = DualLogger(relatorio_path)

    timer_global = Temporizador()
    hw_inicio = capturar_desempenho_maquina()

    logger.log("====================================================")
    logger.log(formatar_relatorio_hardware(hw_inicio))
    logger.log("====================================================\n")

    weights_path = encontrar_pesos_validos(root_dir)
    data_yaml = os.path.join(root_dir, "data.yaml")
    imgsz = 640

    logger.log(f"1. CARREGANDO MODELO EM MODO DE AVALIAÇÃO...\n   Pesos: {weights_path}")
    if not os.path.exists(weights_path):
        logger.log(f"⚠️ Alerta: Arquivo de pesos '{weights_path}' não encontrado. Certifique-se de executar o treinamento primeiro.")

    model = YOLO(weights_path if os.path.exists(weights_path) else "yolov8n.pt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Congela o modelo para inferência/avaliação
    model.eval()

    # ================================
    # 2️⃣ Testes de Qualidade (Métricas & Matriz de Confusão)
    # ================================
    logger.log("\n2. EXECUTANDO AVALIAÇÃO NO SPLIT DE TESTE...")
    timer_val = Temporizador()
    
    # O método val() gera plots automáticos como a Matriz de Confusão (confusion_matrix.png)
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        device=device,
        plots=True,
        save_json=False
    )
    tempo_val_s = timer_val.duracao_s()

    # Cálculo do F1-Score
    precision = metrics.box.mp
    recall = metrics.box.mr
    f1_score = (2 * precision * recall) / (precision + recall + 1e-16)

    hw_val = capturar_desempenho_maquina()

    quality_summary = (
        "====================================================\n"
        "=== MÉTRICAS DE QUALIDADE (SPLIT: TEST) ===\n"
        "====================================================\n"
        f" - Precision (Precisão): {precision:.4f}\n"
        f" - Recall (Revocação): {recall:.4f}\n"
        f" - F1-Score: {f1_score:.4f}\n"
        f" - mAP@0.5: {metrics.box.map50:.4f}\n"
        f" - mAP@0.5:0.95: {metrics.box.map:.4f}\n"
        f" - Matriz de Confusão & Gráficos salvos em: {metrics.save_dir}\n"
        f" - Tempo Total da Validação: {tempo_val_s:.2f} s\n"
        "----------------------------------------------------\n"
        f"{formatar_relatorio_hardware(hw_val)}\n"
        "====================================================\n"
    )
    logger.log(quality_summary)

    # ================================
    # 3️⃣ Inference Benchmark (Latência, Warm-up e FPS)
    # ================================
    logger.log("\n3. INICIANDO BENCHMARK DE LATÊNCIA E FPS (BATCH SIZE = 1)...")
    timer_bench = Temporizador()

    # Imagem simulada em tensor PyTorch
    dummy_input = torch.randint(0, 255, (1, 3, imgsz, imgsz), dtype=torch.uint8, device=device)

    # Warm-up das GPUs para estabilizar clocks
    WARMUP_STEPS = 50
    logger.log(f" ├─ Executando Warm-up ({WARMUP_STEPS} iterações)...")
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model.predict(dummy_input, verbose=False, device=device)

    # Medição de Latência com Sincronização Estrita
    NUM_RUNS = 100
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    logger.log(f" ├─ Medindo latências isoladas ao longo de {NUM_RUNS} execuções...")
    
    with torch.no_grad():
        for _ in range(NUM_RUNS):
            res = model.predict(dummy_input, verbose=False, device=device)
            speed = res[0].speed  # Dicionário nativo do Ultralytics em milissegundos
            preprocess_times.append(speed['preprocess'])
            inference_times.append(speed['inference'])
            postprocess_times.append(speed['postprocess'])

    tempo_bench_s = timer_bench.duracao_s()

    # Médias de Tempo (em milissegundos)
    avg_pre = sum(preprocess_times) / len(preprocess_times)
    avg_inf = sum(inference_times) / len(inference_times)
    avg_post = sum(postprocess_times) / len(postprocess_times)
    avg_total_lat = avg_pre + avg_inf + avg_post

    # Cálculo de FPS End-to-End ($FPS = \frac{1000}{t_{total}}$)
    fps_end_to_end = 1000.0 / avg_total_lat if avg_total_lat > 0 else 0

    hw_final = capturar_desempenho_maquina()
    tempo_total_geral_s = timer_global.duracao_s()

    latency_summary = (
        "====================================================\n"
        "=== BENCHMARK DE LATÊNCIA E DESEMPENHO EM TEMPO REAL ===\n"
        "====================================================\n"
        f" - Hardware Utilizado: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
        f" - Configuração: Batch Size = 1 | Img Size = {imgsz}x{imgsz}\n"
        f" - Pré-processamento Médio: {avg_pre:.2f} ms\n"
        f" - Forward Pass (Inferência) Médio: {avg_inf:.2f} ms\n"
        f" - Pós-processamento / NMS Médio: {avg_post:.2f} ms\n"
        f" - Latência Total End-to-End: {avg_total_lat:.2f} ms\n"
        f" - Taxa de Quadros (FPS Real): {fps_end_to_end:.2f} FPS\n"
        f" - Tempo Total do Benchmark: {tempo_bench_s:.2f} s\n"
        f" - Tempo Geral de Execução: {tempo_total_geral_s:.2f} s\n"
        "----------------------------------------------------\n"
        f"{formatar_relatorio_hardware(hw_final)}\n"
        "====================================================\n"
    )
    logger.log(latency_summary)
    logger.log(f"✨ Testes de validação finalizados! Relatório salvo em '{relatorio_path}'.")


if __name__ == "__main__":
    main()
