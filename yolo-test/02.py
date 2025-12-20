from ultralytics import YOLO
import cv2

# -----------------------------------------
# CONFIGURAÇÕES
# -----------------------------------------
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# -----------------------------------------
# MODELO
# -----------------------------------------
model = YOLO("yolov8n.pt")

# -----------------------------------------
# CAPTURA DA CÂMERA
# -----------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Não foi possível acessar a câmera.")

# -----------------------------------------
# LOOP DE LEITURA
# -----------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]

            if label in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Detecção de Veículos", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------------------
# FINALIZA
# -----------------------------------------
cap.release()
cv2.destroyAllWindows()