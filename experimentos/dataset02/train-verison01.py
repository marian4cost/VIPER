import os
import time
import torch
import pandas as pd
from ultralytics import YOLO

# ================================
# 0️⃣ Estruturas e Callbacks de Monitoramento
# ================================
class TrainingBenchmarkLogger:
    def __init__(self, txt_file_path="relatorio_apresentacao.txt"):
        self.epoch_start_time = None
        self.batch_start_time = None
        self.batch_times = []
        self.txt_file_path = txt_file_path

    def on_train_epoch_start(self, trainer):
        self.epoch_start_time = time.perf_counter()
        self.batch_times = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer):
        self.batch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer):
        if self.batch_start_time is not None:
            batch_dur = (time.perf_counter() - self.batch_start_time) * 1000
            self.batch_times.append(batch_dur)

    def on_train_epoch_end(self, trainer):
        epoch_dur = time.perf_counter() - self.epoch_start_time
        avg_batch = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0

        peak_vram_str = "CPU (VRAM N/A)"
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_vram_str = f"{peak_vram_mb:.2f} MB"

        log_msg = (
            f"[BENCHMARK ÉPOCA {trainer.epoch + 1}/{trainer.epochs}]\n"
            f" ├─ Duração da Época (t_epoch): {epoch_dur:.2f} s\n"
            f" ├─ Tempo Médio por Batch: {avg_batch:.2f} ms\n"
            f" └─ Pico de Alocação VRAM: {peak_vram_str}\n\n"
        )

        # Imprime no terminal em tempo real
        print(log_msg)

        # Salva continuamente no arquivo de texto
        with open(self.txt_file_path, "a", encoding="utf-8") as f:
            f.write(log_msg)


# ================================
# Função Principal de Execução
# ================================
def main():
    txt_filename = "relatorio_apresentacao.txt"

    # Cria/Limpa o arquivo TXT com o cabeçalho inicial
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO DE AVALIAÇÃO DE TREINAMENTO ===\n\n")

    logger = TrainingBenchmarkLogger(txt_file_path=txt_filename)

    # 1️⃣ Carregar modelo base
    model = YOLO("yolov8n.pt")

    # Registrar Callbacks
    model.add_callback("on_train_epoch_start", logger.on_train_epoch_start)
    model.add_callback("on_train_batch_start", logger.on_train_batch_start)
    model.add_callback("on_train_batch_end", logger.on_train_batch_end)
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)

    # 2️⃣ Determinar Dispositivo de Treino
    DEVICE = "0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # 3️⃣ Treinar modelo
    PROJECT_DIR = "runs/detect"
    EXP_NAME = "placa_detector"
    TOTAL_EPOCHS = 120

    results = model.train(
        data=os.path.abspath("data.yaml"),
        epochs=TOTAL_EPOCHS,
        imgsz=640,
        batch=4,
        device=DEVICE,
        project=PROJECT_DIR,
        name=EXP_NAME,
        exist_ok=True,
        plots=True
    )

    print("\n Treinamento e Benchmark finalizados com sucesso!")

    save_dir = str(model.trainer.save_dir)
    results_csv = os.path.join(save_dir, "results.csv")
    results_png = os.path.join(save_dir, "results.png")

    # 4️⃣ Leitura Automática das Métricas Finais via Pandas (se existir results.csv)
    map50_val = "N/A"
    map50_95_val = "N/A"
    box_loss_val = "N/A"
    cls_loss_val = "N/A"
    dfl_loss_val = "N/A"

    if os.path.exists(results_csv):
        try:
            df = pd.read_csv(results_csv)
            # Remove espaços em branco dos nomes das colunas
            df.columns = df.columns.str.strip()
            
            # Pega os valores da última época gravada
            last_row = df.iloc[-1]
            map50_val = f"{last_row.get('metrics/mAP50(B)', 'N/A'):.4f}"
            map50_95_val = f"{last_row.get('metrics/mAP50-95(B)', 'N/A'):.4f}"
            box_loss_val = f"{last_row.get('train/box_loss', 'N/A'):.4f}"
            cls_loss_val = f"{last_row.get('train/cls_loss', 'N/A'):.4f}"
            dfl_loss_val = f"{last_row.get('train/dfl_loss', 'N/A'):.4f}"
        except Exception as e:
            print(f"Aviso: Não foi possível ler as métricas finais do CSV automaticamente: {e}")

    # 5️⃣ Gravação do Resumo Final Estruturado no TXT
    summary_text = (
        "\n"
        "====================================================\n"
        "=== DADOS PARA O ROTEIRO DE APRESENTAÇÃO ===\n"
        "====================================================\n"
        "1. CONFIGURAÇÃO DO EXPERIMENTO:\n"
        f"   - Arquitetura Usada: YOLOv8 (yolov8n.pt)\n"
        f"   - Dispositivo/GPU: {gpu_name}\n"
        f"   - Épocas Totais: {TOTAL_EPOCHS}\n"
        "   - Batch Size: 4\n"
        "   - Img Size: 640x640\n"
        "   - Propósito do Modelo: Detecção de Placas\n\n"
        "2. ARTEFATOS E GRÁFICOS GERADOS:\n"
        f"   - Histórico Completo (CSV): {results_csv}\n"
        f"   - Curvas de Loss e mAP (PNG): {results_png}\n\n"
        "3. RESUMO DE CONVERGÊNCIA (Extraído da última época):\n"
        f"   - mAP50 Final: {map50_val}\n"
        f"   - mAP50-95 Final: {map50_95_val}\n"
        f"   - Perda de Caixa (box_loss): {box_loss_val}\n"
        f"   - Perda de Classe (cls_loss): {cls_loss_val}\n"
        f"   - Perda de Distribuição (dfl_loss): {dfl_loss_val}\n"
        "====================================================\n"
    )

    print(summary_text)

    with open(txt_filename, "a", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n Relatório completo do roteiro salvo em: {txt_filename}")

    # 6️⃣ Profiler de Eficiência do Pipeline (PyTorch Profiler)
    if torch.cuda.is_available():
        print("\n[INICIANDO PERFILAMENTO DE GPU (PyTorch Profiler)]")

        best_pt_path = os.path.join(save_dir, "weights", "best.pt")
        prof_model = YOLO(best_pt_path) if os.path.exists(best_pt_path) else model
        profiler_logs_dir = os.path.join(save_dir, "profiler_logs")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_logs_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        ) as prof:
            dummy_img = torch.randint(0, 255, (640, 640, 3), dtype=torch.uint8).numpy()
            for _ in range(5):
                prof_model.predict(source=dummy_img, device=DEVICE, verbose=False)
                prof.step()

        print(f" Logs do PyTorch Profiler salvos em '{profiler_logs_dir}'.")
        print(f" Para visualizar no TensorBoard rode: tensorboard --logdir={profiler_logs_dir}")


if __name__ == '__main__':
    main()