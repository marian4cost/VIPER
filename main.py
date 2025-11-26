# =============================================================
# 🚗 IDENTIFICADOR DE VEÍCULOS + OCR + INTERFACE GRÁFICA
# =============================================================

from ultralytics import YOLO
import cv2
import numpy as np
import os
from tkinter import Tk, Button, Label, filedialog, Toplevel
from PIL import Image, ImageTk
import easyocr   # ← OCR

# =============================================================
# 1️⃣ CARREGAR MODELO YOLO + EASYOCR
# =============================================================
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(["en"])   # Português não melhora OCR de placas

vehicle_classes = [1, 2, 3, 4, 5, 6, 7, 8]


# =============================================================
# Função para exibir imagem em TK
# =============================================================
def show_image(title, img):
    win = Toplevel()
    win.title(title)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im)

    lbl = Label(win, image=imgtk)
    lbl.image = imgtk
    lbl.pack()


# =============================================================
# 2️⃣ FUNÇÃO PRINCIPAL
# =============================================================
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

    # --- Rodar YOLO ---
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

    # --- Estimativa da área da placa ---
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

    # =============================================================
    # 🔧 PRÉ-PROCESSAMENTO PARA OCR
    # =============================================================
    placa = placa_recortada.copy()

    gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    denoise = cv2.bilateralFilter(gray, 9, 75, 75)

    bin_img = cv2.adaptiveThreshold(
        denoise, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    placa_final = cv2.resize(morph, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # =============================================================
    # 🔤 OCR COM EASYOCR
    # =============================================================
    result = reader.readtext(placa_final)

    texto_placa = ""
    for (bbox, text, prob) in result:
        if len(text) >= 5:      # evita lixo
            texto_placa = text.upper()
            break

    print("\n=========================")
    print("📌 TEXTO LIDO DA PLACA:")
    print("➡️", texto_placa if texto_placa else "Não reconhecida")
    print("=========================\n")

    # =============================================================
    # Exibir imagens
    # =============================================================
    img_draw = img_original.copy()
    cv2.rectangle(img_draw, (x1_car, y1_car), (x2_car, y2_car), (0, 255, 0), 2)
    cv2.rectangle(img_draw, (placa_x1, placa_y1), (placa_x2, placa_y2), (255, 0, 0), 2)

    show_image("Veículo Detectado", img_draw)
    show_image("Placa Recortada", placa_recortada)
    show_image("Pré-processada p/ OCR", placa_final)

    # Salva imagem final preprocessada
    save_path = os.path.join(os.getcwd(), "placa_preprocessada.jpg")
    cv2.imwrite(save_path, placa_final)

    # Mostra a placa reconhecida em popup Tk
    popup = Toplevel()
    popup.title("Placa Reconhecida")
    Label(popup, text=f"📘 Placa: {texto_placa}", font=("Arial", 20)).pack(padx=20, pady=20)

    print(f"✅ Pré-processamento concluído e salvo em: {save_path}")


# =============================================================
# 3️⃣ INTERFACE TKINTER
# =============================================================
root = Tk()
root.title("VIPER")
root.geometry("400x200")

lbl = Label(root, text="🚗 VIPER", font=("Arial", 16))
lbl.pack(pady=20)

btn = Button(root, text="Selecionar Imagem", font=("Arial", 14), command=process_image)
btn.pack(pady=10)

root.mainloop()
