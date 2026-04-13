from ultralytics import YOLO
import torch

model = YOLO("yolov8n.pt")

results = model.train(
model.train(
    data="data.yaml",
    epochs=120,
    imgsz=640,
    batch=4,
    device="cpu",
    name="placa_detector"
)
)

print("Treinamento finalizado")