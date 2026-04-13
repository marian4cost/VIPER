import cv2
import os
import time
import pandas as pd
from ultralytics import YOLO
import easyocr
import re
import numpy as np
import shutil

DIR_ENTRADA = "fotos_carros"
DIR_PRONTAS = "placas_finalizadas"  
DIR_HISTORICO = "historico_carros"  
PLANILHA = "registro_placas.xlsx"

os.makedirs(DIR_PRONTAS, exist_ok=True)
os.makedirs(DIR_HISTORICO, exist_ok=True)

yolo_placa = YOLO("runs/detect/placa_detector/weights/best.pt")
reader = easyocr.Reader(["en"], gpu=True, verbose=False)

def corrigir_mercosul(texto):
    texto = re.sub(r'[^A-Z0-9]', '', texto.upper())
    if len(texto) != 7: return texto
    l_para_n = {'O':'0','I':'1','J':'1','S':'5','Z':'2','G':'6','B':'8','Q':'9','D':'0'}
    n_para_l = {'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B','9':'Q'}
    c = list(texto)
    for i in [0,1,2,4]: 
        if c[i] in n_para_l: c[i] = n_para_l[c[i]]
    for i in [3,5,6]: 
        if c[i] in l_para_n: c[i] = l_para_n[c[i]]
    return "".join(c)

def preprocess(img):
    h, w = img.shape[:2]
    corte = int(h * 0.22)
    img = img[corte:h, 0:w]
    img = cv2.resize(img, (600, int((h-corte)*(600/w))), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(thresh, -1, kernel)

print("🧠 Motor B: Processador de Placas pronto...")

while True:
    arquivos = [f for f in os.listdir(DIR_ENTRADA) if f.endswith(('.png', '.jpg'))]
    
    if not arquivos:
        time.sleep(2) 
        continue

    for arq in arquivos:
        caminho_img = os.path.join(DIR_ENTRADA, arq)
        img_carro = cv2.imread(caminho_img)
        
        res = yolo_placa(img_carro, verbose=False)
        if len(res[0].boxes) > 0:
            box = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
            crop_placa = img_carro[box[1]:box[3], box[0]:box[2]]
            
            placa_limpa = preprocess(crop_placa)
            nome_placa_salva = f"limpa_{arq}"
            caminho_placa_salva = os.path.join(DIR_PRONTAS, nome_placa_salva)
            cv2.imwrite(caminho_placa_salva, placa_limpa)

            ocr_res = reader.readtext(placa_limpa, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", decoder="wordbeamsearch")
            
            texto_lido = "ERRO_LEITURA"
            confianca = 0
            if ocr_res:
                texto_bruto = ocr_res[0][1].replace(" ","")
                texto_lido = corrigir_mercosul(texto_bruto)
                confianca = ocr_res[0][2]

            dados = {
                'Data_Hora': [time.strftime("%Y-%m-%d %H:%M:%S")],
                'Arquivo_Original': [arq],
                'ID_Veiculo': [arq.split('_')[1]],
                'Texto_Placa': [texto_lido],
                'Confianca': [f"{confianca:.2f}"],
                'Path_Placa_Limpa': [caminho_placa_salva]
            }
            
            df = pd.DataFrame(dados)
            if not os.path.isfile(PLANILHA):
                df.to_excel(PLANILHA, index=False)
            else:
                with pd.ExcelWriter(PLANILHA, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    try:
                        existente = pd.read_excel(PLANILHA)
                        novo_df = pd.concat([existente, df], ignore_index=True)
                        novo_df.to_excel(PLANILHA, index=False)
                    except: df.to_excel(PLANILHA, index=False)

            print(f"✅ Processado: {arq} -> {texto_lido}")

        shutil.move(caminho_img, os.path.join(DIR_HISTORICO, arq))

    time.sleep(1)