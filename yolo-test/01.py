from ultralytics import YOLO
import cv2
import os

# --------------------------------------------------
# CONFIGURAÇÕES
# --------------------------------------------------

# Caminho da imagem (local)
IMAGE_PATH = "b.webp"

# Ou URL direta (descomente se quiser usar URL)
# IMAGE_PATH = "https://ultralytics.com/images/bus.jpg"

OUTPUT_PATH = "resultado_veiculos.jpg"

# Classes de veículos no COCO
VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}

# --------------------------------------------------
# CARREGA MODELO YOLO
# --------------------------------------------------
model = YOLO("yolov8n.pt")  # leve e rápido

# --------------------------------------------------
# LEITURA DA IMAGEM
# --------------------------------------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("Imagem não encontrada ou inválida.")

# --------------------------------------------------
# INFERÊNCIA
# --------------------------------------------------
results = model(img)

# --------------------------------------------------
# DESENHO DAS BORDAS
# --------------------------------------------------
for r in results:
    boxes = r.boxes
    names = r.names

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]

        if label in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Desenha bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Texto
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

# --------------------------------------------------
# SALVA RESULTADO
# --------------------------------------------------
cv2.imwrite(OUTPUT_PATH, img)

print(f"Imagem processada salva em: {OUTPUT_PATH}")