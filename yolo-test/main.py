import cv2
from ultralytics import YOLO

try:
    model = YOLO('yolov8n.pt')
    print("✅ Modelo YOLOv8 carregado com sucesso.")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo YOLO: {e}")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Erro: Não foi possível abrir a webcam (índice 0).")
    exit()

print("🎥 Webcam iniciada. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Fim do stream da webcam.")
        break

    results = model(frame, stream=True, verbose=False)

    for r in results:
        annotated_frame = r.plot()

        cv2.imshow('YOLOv8 Deteccao de Objetos (COCO)', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Aplicação encerrada.")