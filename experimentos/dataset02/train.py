import os
import time
import glob
import cv2
import numpy as np
import torch
import pandas as pd
from ultralytics import YOLO


# ================================
# 1️⃣ Módulo Direto para Gerar Mapa de Calor
# ================================
def generate_and_save_heatmap(trainer, img_path, output_dir, epoch_num):
    """
    Extrai ativações da camada do YOLOv8 e salva o heatmap a cada época.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"epoch_{epoch_num:03d}_heatmap.jpg")

    if not os.path.exists(img_path):
        print(f"⚠️ [CAM Warning] Arquivo de teste não encontrado: {img_path}")
        return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠️ [CAM Warning] Não foi possível ler a imagem (formato inválido): {img_path}")
        return

    h, w, _ = img_bgr.shape
    
    try:
        model = trainer.model
        
        img_resized = cv2.resize(img_bgr, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        activation = {}
        def hook_fn(module, input, output):
            activation['target'] = output.detach()

        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            target_layer = model.model.model[-2]
        elif hasattr(model, 'model'):
            target_layer = model.model[-2]
        else:
            target_layer = list(model.children())[-2]

        handle = target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = model(img_tensor)

        handle.remove()

        if 'target' in activation:
            act = activation['target'][0].cpu().numpy()
            heatmap = np.mean(act, axis=0)
            heatmap = np.maximum(heatmap, 0)
            
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)

            heatmap = cv2.resize(heatmap, (w, h))
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
            cv2.imwrite(out_file, overlay)
            print(f"📸 [CAM Saved] Mapa de calor salvo: {out_file}")
        else:
            print(f"⚠️ Não foi possível capturar ativação na época {epoch_num}")

    except Exception as e:
        print(f"⚠️ Erro ao gerar heatmap na época {epoch_num}: {e}")


# ================================
# 2️⃣ Callbacks de Monitoramento por Época
# ================================
class TrainingBenchmarkLogger:
    def __init__(self, txt_file_path="relatorio_apresentacao.txt", test_image_path=None, cam_out_dir=None):
        self.epoch_start_time = None
        self.batch_start_time = None
        self.batch_times = []
        self.txt_file_path = txt_file_path
        self.test_image_path = test_image_path
        self.cam_out_dir = cam_out_dir

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

        if self.test_image_path and self.cam_out_dir:
            generate_and_save_heatmap(
                trainer=trainer,
                img_path=self.test_image_path,
                output_dir=self.cam_out_dir,
                epoch_num=current_epoch
            )


# ================================
# Função Principal
# ================================
def main():
    txt_filename = "relatorio_apresentacao.txt"

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO DE AVALIAÇÃO DE TREINAMENTO ===\n\n")

    PROJECT_DIR = "runs/detect"
    EXP_NAME = "placa_detector"
    TOTAL_EPOCHS = 120
    
    cam_output_dir = os.path.abspath("cam_outputs")

    # Busca restrita a extensões válidas de imagem
    possible_paths = (
        glob.glob("dataset/val/images/*.jpg") + 
        glob.glob("dataset/val/images/*.png") + 
        glob.glob("**/images/*.jpg", recursive=True) +
        glob.glob("**/images/*.png", recursive=True)
    )
    
    # Filtra mantendo apenas extensões de imagem reais
    valid_images = [
        p for p in possible_paths 
        if p.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if valid_images:
        sample_image_path = os.path.abspath(valid_images[0])
        print(f"📌 Usando a imagem de amostra: {sample_image_path}")
    else:
        sample_image_path = os.path.abspath("sample_test.jpg")
        if not os.path.exists(sample_image_path):
            blank_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.rectangle(blank_img, (200, 200), (400, 300), (255, 255, 255), -1)
            cv2.imwrite(sample_image_path, blank_img)
            print(f"⚠️ Imagem sintética criada para o heatmap: {sample_image_path}")

    logger = TrainingBenchmarkLogger(
        txt_file_path=txt_filename,
        test_image_path=sample_image_path,
        cam_out_dir=cam_output_dir
    )

    model = YOLO("yolov8n.pt")

    model.add_callback("on_train_epoch_start", logger.on_train_epoch_start)
    model.add_callback("on_train_batch_start", logger.on_train_batch_start)
    model.add_callback("on_train_batch_end", logger.on_train_batch_end)
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)

    DEVICE = "0" if torch.cuda.is_available() else "cpu"

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


if __name__ == '__main__':
    main()