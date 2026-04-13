from ultralytics import YOLO

model = YOLO("runs/detect/placa_detector/weights/best.pt")

metrics = model.val(
    data="data.yaml",
    split="test",   
    imgsz=640
)

print("\n===== RESULTADOS =====")

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

print("Precisão:", metrics.box.mp)
print("Recall:", metrics.box.mr)