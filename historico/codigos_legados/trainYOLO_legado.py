import os
import time
import psutil
import torch
from ultralytics import YOLO


# ================================
# 1️⃣ Callbacks com Telemetria de Hardware + Desempenho do Modelo
# ================================
class TrainingBenchmarkLogger:
    def __init__(self, txt_file_path="relatorio_apresentacao.txt"):
        self.epoch_start_time = None
        self.batch_start_time = None
        self.batch_times = []
        self.cpu_usages = []
        self.ram_usages = []
        self.batch_losses = []  # Acumulador para corrigir o bug de loss
        self.txt_file_path = txt_file_path

    def on_train_epoch_start(self, trainer):
        self.epoch_start_time = time.perf_counter()
        self.batch_times = []
        self.cpu_usages = []
        self.ram_usages = []
        self.batch_losses = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer):
        self.batch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer):
        if self.batch_start_time is not None:
            batch_dur = (time.perf_counter() - self.batch_start_time) * 1000
            self.batch_times.append(batch_dur)
            self.cpu_usages.append(psutil.cpu_percent())
            self.ram_usages.append(psutil.virtual_memory().used / (1024 ** 2))  # MB

            # Captura de Losses em tempo real a cada batch
            try:
                if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    loss_vals = trainer.loss_items
                    if isinstance(loss_vals, torch.Tensor):
                        loss_vals = loss_vals.detach().cpu().numpy()
                    if len(loss_vals) >= 3:
                        self.batch_losses.append([float(loss_vals[0]), float(loss_vals[1]), float(loss_vals[2])])
            except Exception:
                pass

    def on_train_epoch_end(self, trainer):
        epoch_dur = time.perf_counter() - self.epoch_start_time
        avg_batch = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        avg_cpu = sum(self.cpu_usages) / len(self.cpu_usages) if self.cpu_usages else 0
        peak_ram = max(self.ram_usages) if self.ram_usages else 0

        peak_vram_str = "CPU (VRAM N/A)"
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_vram_str = f"{peak_vram_mb:.2f} MB"

        # Média real das perdas calculada no final da época
        box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0
        if self.batch_losses:
            arr_losses = torch.tensor(self.batch_losses).mean(dim=0).tolist()
            box_loss, cls_loss, dfl_loss = arr_losses[0], arr_losses[1], arr_losses[2]

        # Métricas de validação da época
        map50, map50_95, precision, recall = 0.0, 0.0, 0.0, 0.0
        try:
            if hasattr(trainer, "metrics") and trainer.metrics:
                map50 = trainer.metrics.get("metrics/mAP50(B)", 0.0)
                map50_95 = trainer.metrics.get("metrics/mAP50-95(B)", 0.0)
                precision = trainer.metrics.get("metrics/precision(B)", 0.0)
                recall = trainer.metrics.get("metrics/recall(B)", 0.0)
        except Exception as e:
            print(f"⚠️ [Warning] Não foi possível extrair métricas de validação: {e}")

        current_epoch = trainer.epoch + 1
        log_msg = (
            f"====================================================\n"
            f"[DIAGNÓSTICO COMPLETO ÉPOCA {current_epoch}/{trainer.epochs}]\n"
            f"====================================================\n"
            f"💻 HARDWARE & COMPUTAÇÃO:\n"
            f" ├─ Duração da Época (t_epoch): {epoch_dur:.2f} s\n"
            f" ├─ Tempo Médio por Batch: {avg_batch:.2f} ms\n"
            f" ├─ Uso Médio da CPU Host: {avg_cpu:.1f}%\n"
            f" ├─ Pico de RAM do Sistema: {peak_ram:.2f} MB\n"
            f" └─ Pico de Alocação VRAM: {peak_vram_str}\n"
            f"🎯 APRENDIZADO & DESEMPENHO DO MODELO:\n"
            f" ├─ Box Loss (Ajuste Caixa): {box_loss:.4f}\n"
            f" ├─ Class Loss (Detecção Placa): {cls_loss:.4f}\n"
            f" ├─ DFL Loss (Ajuste Bordas): {dfl_loss:.4f}\n"
            f" ├─ Precision (Validação): {precision:.4f}\n"
            f" ├─ Recall (Validação): {recall:.4f}\n"
            f" ├─ mAP@0.5: {map50:.4f}\n"
            f" └─ mAP@0.5:0.95: {map50_95:.4f}\n"
            f"====================================================\n\n"
        )

        print(log_msg)
        with open(self.txt_file_path, "a", encoding="utf-8") as f:
            f.write(log_msg)


# ================================
# 2️⃣ Função Principal
# ================================
def main():
    txt_filename = "relatorio_apresentacao00.txt"

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO COMPLETO DE TREINAMENTO, EXPORTAÇÃO E VARREDURA CONF ===\n\n")

    PROJECT_DIR = "runs/detect"
    EXP_NAME = "placa_detector"
    TOTAL_EPOCHS = 120
    DATA_YAML = os.path.abspath("data.yaml")

    logger = TrainingBenchmarkLogger(txt_file_path=txt_filename)
    model = YOLO("yolov8n.pt")

    model.add_callback("on_train_epoch_start", logger.on_train_epoch_start)
    model.add_callback("on_train_batch_start", logger.on_train_batch_start)
    model.add_callback("on_train_batch_end", logger.on_train_batch_end)
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)

    DEVICE = "0" if torch.cuda.is_available() else "cpu"

    # --- ETAPA 1: TREINAMENTO DO MODELO ---
    results = model.train(
        data=DATA_YAML,
        epochs=TOTAL_EPOCHS,
        imgsz=1280,
        batch=4,
        device=DEVICE,
        project=PROJECT_DIR,
        name=EXP_NAME,
        exist_ok=True,
        plots=True,
        weight_decay=0.0005,
        degrees=10.0,
        close_mosaic=10,
        cls=1.5
    )

    best_weights_path = os.path.join(PROJECT_DIR, EXP_NAME, "weights", "best.pt")
    best_model = YOLO(best_weights_path)

    # --- ETAPA 2: EXPORTAÇÃO E QUANTIZAÇÃO ---
    print("\n📦 [EXPORTAÇÃO] Exportando modelo para formatos otimizados...")
    onnx_path = best_model.export(format="onnx", imgsz=1280, dynamic=True)
    print(f"✅ [EXPORT] Modelo ONNX salvo em: {onnx_path}")

    try:
        tflite_path = best_model.export(format="tflite", imgsz=640, int8=True, data=DATA_YAML)
        print(f"✅ [QUANTIZAÇÃO INT8] Modelo TFLite Quantizado salvo em: {tflite_path}")
    except Exception as e:
        print(f"⚠️ [QUANTIZAÇÃO Warning] Falha ao exportar TFLite INT8: {e}")

    # --- ETAPA 3: VARREDURA (SWEEP) DE THRESHOLD DE CONFIANÇA & NMS ---
    print("\n🔍 [SWEEP] Iniciando Varredura de Threshold de Confiança (conf: 0.10 a 0.70)...")
    sweep_header = (
        "\n====================================================\n"
        "📊 VARREDURA DE THRESHOLD DE CONFIANÇA & NMS (SWEEP)\n"
        "====================================================\n"
        "Conf_Thresh | Precision | Recall   | mAP@0.5  | mAP@0.5:0.95\n"
        "----------------------------------------------------\n"
    )
    print(sweep_header)
    with open(txt_filename, "a", encoding="utf-8") as f:
        f.write(sweep_header)

    conf_thresholds = [round(c, 2) for c in np.arange(0.10, 0.75, 0.05)]
    for conf_val in conf_thresholds:
        val_results = best_model.val(
            data=DATA_YAML,
            imgsz=1280,
            batch=4,
            conf=conf_val,
            iou=0.6,
            device=DEVICE,
            plots=False,
            verbose=False
        )

        p = val_results.results_dict.get("metrics/precision(B)", 0.0)
        r = val_results.results_dict.get("metrics/recall(B)", 0.0)
        map50 = val_results.results_dict.get("metrics/mAP50(B)", 0.0)
        map50_95 = val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)

        row_str = f"  {conf_val:.2f}     |  {p:.4f}   |  {r:.4f}  |  {map50:.4f}  |  {map50_95:.4f}\n"
        print(row_str, end="")
        with open(txt_filename, "a", encoding="utf-8") as f:
            f.write(row_str)

    print("\n✨ Treinamento, Exportação e Varredura finalizados com sucesso!")


if __name__ == '__main__':
    main()