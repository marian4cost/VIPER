import time
import torch
from ultralytics import YOLO

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


def main():
    logger = DualLogger("relatorio_validacao.txt")
    weights_path = "/home/mariana-costa/Documentos/projects/viper/VIPER/runs/detect/runs/detect/placa_detector/weights/best.pt"
    data_yaml = "data.yaml"
    imgsz = 640

    logger.log("1. CARREGANDO MODELO EM MODO DE AVALIAÇÃO...")
    model = YOLO(weights_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Congela o modelo para inferência/avaliação
    model.eval()

    # ================================
    # 2️⃣ Testes de Qualidade (Métricas & Matriz de Confusão)
    # ================================
    logger.log("\n2. EXECUTANDO AVALIAÇÃO NO SPLIT DE TESTE...")
    
    # O método val() gera plots automáticos como a Matriz de Confusão (confusion_matrix.png)
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        device=device,
        plots=True,
        save_json=False
    )

    # Cálculo do F1-Score
    precision = metrics.box.mp
    recall = metrics.box.mr
    f1_score = (2 * precision * recall) / (precision + recall + 1e-16)

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
        "====================================================\n"
    )
    logger.log(quality_summary)

    # ================================
    # 3️⃣ Inference Benchmark (Latência, Warm-up e FPS)
    # ================================
    logger.log("\n3. INICIANDO BENCHMARK DE LATÊNCIA E FPS (BATCH SIZE = 1)...")

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

    # Médias de Tempo (em milissegundos)
    avg_pre = sum(preprocess_times) / len(preprocess_times)
    avg_inf = sum(inference_times) / len(inference_times)
    avg_post = sum(postprocess_times) / len(postprocess_times)
    avg_total_lat = avg_pre + avg_inf + avg_post

    # Cálculo de FPS End-to-End ($FPS = \frac{1000}{t_{total}}$)
    fps_end_to_end = 1000.0 / avg_total_lat if avg_total_lat > 0 else 0

    latency_summary = (
        "====================================================\n"
        "=== BENCHMARK DE LATÊNCIA E DESEMPENHO EM TEMPO REAL ===\n"
        "====================================================\n"
        f" - Hardware Utilizado: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
        f" - Configuração: Batch Size = 1 | Img Size = {imgsz}x{imgsz}\n"
        f" - Pré-processamento Médio: {avg_pre:.2f} ms\n"
        f" - Forward Pass (Infêrencia) Médio: {avg_inf:.2f} ms\n"
        f" - Pós-processamento / NMS Médio: {avg_post:.2f} ms\n"
        f" - Latência Total End-to-End: {avg_total_lat:.2f} ms\n"
        f" - Taxa de Quadros (FPS Real): {fps_end_to_end:.2f} FPS\n"
        "====================================================\n"
    )
    logger.log(latency_summary)
    logger.log(" Testes de validação finalizados! Relatório salvo em 'relatorio_validacao.txt'.")


if __name__ == "__main__":
    main()