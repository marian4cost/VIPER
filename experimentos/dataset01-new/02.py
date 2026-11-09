import os
import time
import shutil
import torch
from ultralytics import YOLO


# ================================
# 1️⃣ Callbacks de Monitoramento por Época
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

        current_epoch = trainer.epoch + 1
        log_msg = (
            f"[BENCHMARK ÉPOCA {current_epoch}/{trainer.epochs}]\n"
            f" ├─ Duração da Época (t_epoch): {epoch_dur:.2f} s\n"
            f" ├─ Tempo Médio por Batch: {avg_batch:.2f} ms\n"
            f" └─ Pico de Alocação VRAM: {peak_vram_str}\n\n"
        )

        print(log_msg)

        with open(self.txt_file_path, "a", encoding="utf-8") as f:
            f.write(log_msg)


# ================================
# 2️⃣ Função Principal de Treinamento
# ================================
def main():
    txt_filename = "relatorio_apresentacao.txt"

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO DE AVALIAÇÃO DE TREINAMENTO ===\n\n")

    # Garante o caminho absoluto no diretório do próprio script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.join(BASE_DIR, "runs", "detect")
    EXP_NAME = "placa_detector"
    TOTAL_EPOCHS = 120

    logger = TrainingBenchmarkLogger(txt_file_path=txt_filename)

    model = YOLO("yolov8n.pt")

    # Registro de callbacks para métricas de desempenho
    model.add_callback("on_train_epoch_start", logger.on_train_epoch_start)
    model.add_callback("on_train_batch_start", logger.on_train_batch_start)
    model.add_callback("on_train_batch_end", logger.on_train_batch_end)
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)

    DEVICE = "0" if torch.cuda.is_available() else "cpu"

    results = model.train(
        data=os.path.abspath("dataset/data.yaml"),
        epochs=TOTAL_EPOCHS,
        imgsz=1280,
        batch=4,
        workers=2,
        device=DEVICE,
        project=PROJECT_DIR,  # Agora aponta para dataset01-new/runs/detect
        name=EXP_NAME,
        exist_ok=True,
        plots=True,
        weight_decay=0.0005,
        degrees=10.0,
        close_mosaic=10,
        cls=1.5
    )

    print("\nTreinamento e Benchmark finalizados com sucesso!")

    # Copia o best.pt original para a raiz com um nome customizado
    original_best = os.path.join(results.save_dir, "weights", "best.pt")
    destination_path = os.path.abspath("modelo_placa_best.pt")

    if os.path.exists(original_best):
        shutil.copy(original_best, destination_path)
        print(f"✅ Novo arquivo salvo com sucesso em: {destination_path}")
    else:
        print("⚠️ O arquivo best.pt original não foi encontrado.")


if __name__ == '__main__':
    main()
