import os
import re
import time
import cv2
import numpy as np
import torch
import easyocr
from ultralytics import YOLO

# ==========================================
# CONFIGURAÇÕES DE DIRETÓRIOS E MODELOS
# ==========================================
DIR_ENTRADA_VEICULOS = "fotos_carros"         # Entrada: fotos dos veículos
DIR_SAIDA_PROCESSADAS = "placas_processadas"   # Saída: imagens das placas binarizadas
DIR_SAIDA_DEBUG = "debug_saida_viper"         # Saída: imagens anotadas com a leitura
ARQUIVO_RELATORIO = "relatorio_execucao_main.txt"
ARQUIVO_LOG_VIPER = "acessos_viper.log"

os.makedirs(DIR_ENTRADA_VEICULOS, exist_ok=True)
os.makedirs(DIR_SAIDA_PROCESSADAS, exist_ok=True)
os.makedirs(DIR_SAIDA_DEBUG, exist_ok=True)

# Modelos YOLO
MODELO_VEICULO = "yolov8n.pt"
MODELO_PLACA = "runs/detect/placa_detector/weights/best.pt"  # Ajuste o caminho se necessário

CLASSES_VEICULOS = [2, 3, 5, 7]  # Carro, moto, ônibus, caminhão no COCO Dataset

# Regex Duplo para Placas do Brasil
REGEX_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
REGEX_ANTIGO = re.compile(r"^[A-Z]{3}[0-9]{4}$")


# ==========================================
# 1️⃣ FUNÇÕES DE PRÉ-PROCESSAMENTO GEOMÉTRICO
# ==========================================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Aplica o letterbox mantendo o aspect ratio original com padding."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (dw, dh)


def ordenar_pontos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def encontrar_quadrilatero_placa(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / max(h, 1)
            if 1.8 <= aspect_ratio <= 6.5:
                return approx
    return None


def warp_placa(img, quad):
    pts = quad.reshape(4, 2).astype(np.float32)
    rect = ordenar_pontos(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def preprocessar_crop_para_ocr(crop_placa):
    if crop_placa is None or crop_placa.size == 0:
        return None

    crop_padded = cv2.copyMakeBorder(
        crop_placa, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    quad = encontrar_quadrilatero_placa(crop_padded)
    placa_alinhada = (
        warp_placa(crop_padded, quad) if quad is not None else crop_padded
    )

    h, w = placa_alinhada.shape[:2]
    target_w = 400
    target_h = int(h * (target_w / float(w))) if w > 0 else 100
    resized = cv2.resize(
        placa_alinhada, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def recortar_roi_suave(img, pct_pad=0.03):
    h, w = img.shape[:2]
    pad_h = int(h * pct_pad)
    pad_w = int(w * pct_pad)

    y1, y2 = pad_h, max(pad_h + 1, h - pad_h)
    x1, x2 = pad_w, max(pad_w + 1, w - pad_w)

    return img[y1:y2, x1:x2]


# ==========================================
# 2️⃣ HEURÍSTICA E REGRAS VIPER (CORREÇÃO DE PLACAS)
# ==========================================
def _validar_string_7char(texto):
    letras_para_numeros = {
        "O": "0", "I": "1", "J": "1", "S": "5", "Z": "2",
        "G": "6", "B": "8", "Q": "9", "D": "0", "A": "4",
    }
    numeros_para_letras = {
        "0": "O", "1": "I", "2": "Z", "5": "S",
        "6": "G", "8": "B", "9": "Q", "4": "A",
    }

    # Mercosul (LLLNLNN)
    cand_mercosul = list(texto)
    for i in [0, 1, 2, 4]:
        if cand_mercosul[i] in numeros_para_letras:
            cand_mercosul[i] = numeros_para_letras[cand_mercosul[i]]
    for i in [3, 5, 6]:
        if cand_mercosul[i] in letras_para_numeros:
            cand_mercosul[i] = letras_para_numeros[cand_mercosul[i]]

    str_mercosul = "".join(cand_mercosul)
    if REGEX_MERCOSUL.match(str_mercosul):
        return str_mercosul, "Mercosul"

    # Padrão Antigo (LLLNNNN)
    cand_antigo = list(texto)
    for i in [0, 1, 2]:
        if cand_antigo[i] in numeros_para_letras:
            cand_antigo[i] = numeros_para_letras[cand_antigo[i]]
    for i in [3, 4, 5, 6]:
        if cand_antigo[i] in letras_para_numeros:
            cand_antigo[i] = letras_para_numeros[cand_antigo[i]]

    str_antigo = "".join(cand_antigo)
    if REGEX_ANTIGO.match(str_antigo):
        return str_antigo, "Padrão Antigo"

    return None, None


def corrigir_leitura_placa(texto_sujo):
    texto = re.sub(r"[^A-Z0-9]", "", texto_sujo.upper())

    if len(texto) == 7:
        res, padrao = _validar_string_7char(texto)
        if res:
            return res, padrao

    elif len(texto) == 8:
        cand1 = texto[1:]
        res1, padrao1 = _validar_string_7char(cand1)
        if res1:
            return res1, padrao1

        cand2 = texto[:-1]
        res2, padrao2 = _validar_string_7char(cand2)
        if res2:
            return res2, padrao2

        for i in range(1, 7):
            cand_mid = texto[:i] + texto[i+1:]
            res_mid, padrao_mid = _validar_string_7char(cand_mid)
            if res_mid:
                return res_mid, padrao_mid

    return texto, "Padrão incerto"


# ==========================================
# 3️⃣ FUNÇÃO PRINCIPAL DO PIPELINE COMPLETO
# ==========================================
def executar_pipeline_completo():
    EXTENSOES_VALIDAS = (".png", ".jpg", ".jpeg", ".bmp")
    arquivos = [
        f for f in os.listdir(DIR_ENTRADA_VEICULOS)
        if f.lower().endswith(EXTENSOES_VALIDAS)
    ]

    if not arquivos:
        print(f"⚠️ Nenhuma imagem encontrada em '{DIR_ENTRADA_VEICULOS}'. Adicione fotos de veículos para iniciar.")
        return

    print("🚀 Carregando modelos YOLO e EasyOCR...")
    yolo_veiculo = YOLO(MODELO_VEICULO)
    
    # Valida se o modelo treinado existe; se não, avisa o usuário
    if os.path.exists(MODELO_PLACA):
        yolo_placa = YOLO(MODELO_PLACA)
    else:
        print(f"⚠️ Modelo de placas '{MODELO_PLACA}' não encontrado. Usando 'yolov8n.pt' como fallback.")
        yolo_placa = YOLO("yolov8n.pt")

    gpu_disponivel = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=gpu_disponivel, verbose=False)

    easyocr_params = dict(
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        decoder="wordbeamsearch",
        beamWidth=5,
        mag_ratio=1.5,
        paragraph=False,
        detail=1,
    )

    relatorio_linhas = []
    relatorio_linhas.append("=" * 80 + "\n")
    relatorio_linhas.append("PIPELINE INTEGRADO VIPER - PRÉ-PROCESSAMENTO, IDENTIFICAÇÃO E OCR\n")
    relatorio_linhas.append("=" * 80 + "\n\n")

    tempos_total = []
    tempos_ocr = []
    placas_lidas_com_sucesso = 0

    for idx, arq in enumerate(arquivos, start=1):
        t_inicio_img = time.perf_counter()
        caminho_img = os.path.join(DIR_ENTRADA_VEICULOS, arq)
        img_carro = cv2.imread(caminho_img)

        if img_carro is None:
            relatorio_linhas.append(f"[{idx}/{len(arquivos)}] ❌ {arq}: Erro ao carregar imagem.\n\n")
            continue

        # --- ETAPA 1: DETECÇÃO DO VEÍCULO E DA PLACA (YOLO) ---
        res_veiculo = yolo_veiculo(img_carro, verbose=False, classes=CLASSES_VEICULOS)
        crops_veiculos = []

        if len(res_veiculo[0].boxes) > 0:
            for box_v in res_veiculo[0].boxes.xyxy.cpu().numpy().astype(int):
                crops_veiculos.append(img_carro[box_v[1]:box_v[3], box_v[0]:box_v[2]])

        crops_veiculos.append(img_carro)  # Fallback: a imagem inteira

        placa_binaria = None
        placa_encontrada = False

        for crop_v in crops_veiculos:
            res_placa = yolo_placa(crop_v, verbose=False)

            if len(res_placa[0].boxes) > 0:
                box_p = res_placa[0].boxes.xyxy[0].cpu().numpy().astype(int)
                crop_placa = crop_v[box_p[1]:box_p[3], box_p[0]:box_p[2]]

                placa_binaria = preprocessar_crop_para_ocr(crop_placa)
                if placa_binaria is not None:
                    placa_encontrada = True
                    break

        # Fallback de Contornos
        if not placa_encontrada:
            quad = encontrar_quadrilatero_placa(img_carro)
            if quad is not None:
                crop_direto = warp_placa(img_carro, quad)
                placa_binaria = preprocessar_crop_para_ocr(crop_direto)

        if placa_binaria is None:
            relatorio_linhas.append(f"[{idx}/{len(arquivos)}] ❌ {arq}: Falha na localização/pré-processamento da placa.\n\n")
            continue

        # Salva a imagem da placa binarizada
        caminho_placa_processada = os.path.join(DIR_SAIDA_PROCESSADAS, f"limpa_{arq}")
        cv2.imwrite(caminho_placa_processada, placa_binaria)

        # --- ETAPA 2: LEITURA OCR E REGRAS DE CORREÇÃO ---
        t_ocr_start = time.perf_counter()
        
        # Converte a binarizada de volta para BGR para passar no EasyOCR
        img_for_ocr = cv2.cvtColor(placa_binaria, cv2.COLOR_GRAY2BGR)
        img_crop_ocr = recortar_roi_suave(img_for_ocr, pct_pad=0.03)

        results_ocr = reader.readtext(img_crop_ocr, **easyocr_params)
        t_ocr_end = time.perf_counter()

        tempo_ocr_ms = (t_ocr_end - t_ocr_start) * 1000
        tempos_ocr.append(tempo_ocr_ms)

        texto_final = "NÃO RECONHECIDA"
        tipo_padrao = "Padrão incerto"
        bbox_detectado = None

        for bbox, text, prob in results_ocr:
            if prob > 0.20:
                limpo = text.replace(" ", "").upper()
                texto_corrigido, padrao = corrigir_leitura_placa(limpo)

                if padrao != "Padrão incerto":
                    texto_final = texto_corrigido
                    tipo_padrao = padrao
                    bbox_detectado = bbox
                    placas_lidas_com_sucesso += 1
                    break
                else:
                    texto_final = texto_corrigido
                    tipo_padrao = padrao
                    bbox_detectado = bbox

        # --- ETAPA 3: GERAR IMAGEM ANOTADA (DEBUG) E LOG ---
        img_anotada = img_for_ocr.copy()
        if bbox_detectado is not None:
            (tl, tr, br, bl) = bbox_detectado
            pt_tl = (int(tl[0]), int(tl[1]))
            pt_br = (int(br[0]), int(br[1]))

            cv2.rectangle(img_anotada, pt_tl, pt_br, (0, 255, 0), 2)
            cv2.putText(
                img_anotada,
                f"{texto_final} ({tipo_padrao})",
                (pt_tl[0], max(15, pt_tl[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        caminho_debug = os.path.join(DIR_SAIDA_DEBUG, f"anotado_{arq}")
        cv2.imwrite(caminho_debug, img_anotada)

        # Regrava log de acesso estilo sistema
        payload_log = f"Arquivo: {arq} | Placa: {texto_final} | Tipo: {tipo_padrao} | OCR Time: {tempo_ocr_ms:.2f}ms"
        with open(os.path.join(DIR_SAIDA_DEBUG, ARQUIVO_LOG_VIPER), "a", encoding="utf-8") as log_f:
            log_f.write(payload_log + "\n")

        t_fim_img = time.perf_counter()
        tempo_total_ms = (t_fim_img - t_inicio_img) * 1000
        tempos_total.append(tempo_total_ms)

        # Registro no Relatório Texto
        relatorio_linhas.append(f"[{idx}/{len(arquivos)}] ARQUIVO: {arq}\n")
        relatorio_linhas.append(f"  ├─ Placa Binarizada Salva em: {caminho_placa_processada}\n")
        relatorio_linhas.append(f"  ├─ Placa Identificada: {texto_final} ({tipo_padrao})\n")
        relatorio_linhas.append(f"  └─ Tempo total pipeline: {tempo_total_ms:.2f} ms (OCR: {tempo_ocr_ms:.2f} ms)\n\n")

    # --- ETAPA 4: CONSOLIDAÇÃO GERAL ---
    relatorio_linhas.append("=" * 80 + "\n")
    relatorio_linhas.append("RESUMO DA EXECUÇÃO INTEGRADA\n")
    relatorio_linhas.append("=" * 80 + "\n")
    relatorio_linhas.append(f" - Total de Imagens Processadas: {len(arquivos)}\n")
    relatorio_linhas.append(f" - Placas com Leitura Aprovada: {placas_lidas_com_sucesso} ({(placas_lidas_com_sucesso / len(arquivos)) * 100:.1f}%)\n")
    if tempos_total:
        relatorio_linhas.append(f" - Tempo Médio Total por Imagem: {np.mean(tempos_total):.2f} ms\n")
        relatorio_linhas.append(f" - Tempo Médio Exclusivo de OCR: {np.mean(tempos_ocr):.2f} ms\n")

    texto_final_relatorio = "".join(relatorio_linhas)

    with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as f:
        f.write(texto_final_relatorio)

    print(texto_final_relatorio)
    print(f"📄 Relatório consolidado gerado e salvo em: '{ARQUIVO_RELATORIO}'")


if __name__ == "__main__":
    executar_pipeline_completo()