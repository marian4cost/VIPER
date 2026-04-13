from ultralytics import YOLO
import cv2
import os
from datetime import datetime

model = YOLO("yolov8n.pt")   

show_stream = True

CAR_CLASS = 2

coco_names = model.names

video_source = "https://appnittrans.niteroi.rj.gov.br:8888/000024/last_video.mp4"
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    raise ValueError("Não foi possível abrir a stream de vídeo. Verifique a fonte.")

output_dir = "fotos_carros"
os.makedirs(output_dir, exist_ok=True)

ROI_X_MIN = 0.10
ROI_X_MAX = 0.90
ROI_Y_MIN = 0.55
ROI_Y_MAX = 0.70

MIN_BOX_WIDTH = 120
MIN_BOX_HEIGHT = 70

CROP_MARGIN_X = 0.12
CROP_MARGIN_Y = 0.12

saved_track_ids = set()
photo_count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h_frame, w_frame = frame.shape[:2]
    roi_x1 = int(w_frame * ROI_X_MIN)
    roi_x2 = int(w_frame * ROI_X_MAX)
    roi_y1 = int(h_frame * ROI_Y_MIN)
    roi_y2 = int(h_frame * ROI_Y_MAX)

    results = model.track(
        frame,
        classes=[CAR_CLASS],
        conf=0.40,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            if track_id is not None:
                label = f"{coco_names[cls]} ID:{track_id} ({conf:.2f})"
            else:
                label = f"{coco_names[cls]} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            box_w = x2 - x1
            box_h = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            inside_roi = roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2
            size_ok = box_w >= MIN_BOX_WIDTH and box_h >= MIN_BOX_HEIGHT

            if track_id is not None and track_id not in saved_track_ids:
                if not inside_roi or not size_ok:
                    continue

                margin_x = int(box_w * CROP_MARGIN_X)
                margin_y = int(box_h * CROP_MARGIN_Y)

                x1c, y1c = max(0, x1 - margin_x), max(0, y1 - margin_y)
                x2c, y2c = min(w_frame, x2 + margin_x), min(h_frame, y2 + margin_y)

                if x2c > x1c and y2c > y1c:
                    car_crop = frame[y1c:y2c, x1c:x2c]
                    if car_crop.size > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        photo_name = f"carro_id{track_id}_{timestamp}_{photo_count:04d}.png"
                        photo_path = os.path.join(output_dir, photo_name)
                        cv2.imwrite(photo_path, car_crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        photo_count += 1
                        saved_track_ids.add(track_id)

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 180, 0), 2)
    cv2.putText(
        frame,
        "Zona util",
        (roi_x1, max(20, roi_y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 180, 0),
        2,
    )

    if (show_stream == True):
        cv2.imshow("Carros Detectados", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Detecção concluída! Fotos salvas em: {output_dir}")
