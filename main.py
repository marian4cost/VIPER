from ultralytics import YOLO
import cv2
import numpy as np
import os
from tkinter import Tk, Button, Label, filedialog, Toplevel
from PIL import Image, ImageTk

model = YOLO("yolov8n.pt")

vehicle_classes = [1, 2, 3, 4, 5, 6, 7, 8]  # índices COCO de veículos

def show_image(title, img):
    win = Toplevel()
    win.title(title)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_rgb)

    imgtk = ImageTk.PhotoImage(image=im)

    lbl = Label(win, image=imgtk)
    lbl.image = imgtk
    lbl.pack()

def process_image():
    path = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Arquivos de imagem", "*.jpg *.jpeg *.png")]
    )

    if not path:
        return

    img_original = cv2.imread(path)
    if img_original is None:
        print("❌ Erro ao abrir imagem.")
        return

    results = model.predict(
        source=path,
        classes=vehicle_classes,
        conf=0.4,
        save=False,
        show=False
    )

    if not results or len(results[0].boxes) == 0:
        print("⚠️ Nenhum veículo encontrado.")
        return

    box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1_car, y1_car, x2_car, y2_car = box

    car_w = x2_car - x1_car
    car_h = y2_car - y1_car

    placa_y1 = y1_car + int(car_h * 0.70)
    placa_y2 = y2_car

    placa_x1 = x1_car + int(car_w * 0.35)
    placa_x2 = x1_car + int(car_w * 0.65)

    H, W, _ = img_original.shape

    placa_x1 = max(0, placa_x1)
    placa_y1 = max(0, placa_y1)
    placa_x2 = min(W, placa_x2)
    placa_y2 = min(H, placa_y2)

    placa_recortada = img_original[placa_y1:placa_y2, placa_x1:placa_x2]

    img_draw = img_original.copy()
    cv2.rectangle(img_draw, (x1_car, y1_car), (x2_car, y2_car), (0, 255, 0), 2)
    cv2.rectangle(img_draw, (placa_x1, placa_y1), (placa_x2, placa_y2), (255, 0, 0), 2)

    show_image("Veículo Detectado", img_draw)
    show_image("Placa Estimada (Recorte)", placa_recortada)

    save_path = os.path.join(os.getcwd(), "placa_recortada.jpg")
    cv2.imwrite(save_path, placa_recortada)

    print(f"✅ Recorte da placa salvo em: {save_path}")

root = Tk()
root.title("VIPER")
root.geometry("400x200")

lbl = Label(root, text="🚗 VIPER", font=("Arial", 16))
lbl.pack(pady=20)

btn = Button(root, text="Selecionar Imagem", font=("Arial", 14), command=process_image)
btn.pack(pady=10)

root.mainloop()
