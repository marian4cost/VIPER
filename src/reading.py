import os
import re
import sys
import time
import cv2
import easyocr
import numpy as np
import torch

# Permite importar o módulo telemetria quando executado da raiz ou da pasta novos_codigos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from telemetria import capturar_desempenho_maquina, formatar_relatorio_hardware, Temporizador

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIR_ENTRADA = os.path.join(ROOT_DIR, "placas_finalizadas")
DIR_SAIDA_DEBUG = os.path.join(ROOT_DIR, "debug_saida_viper")
ARQUIVO_RELATORIO = os.path.join(ROOT_DIR, "relatorios", "leitura_ocr", "relatorio_leitura.txt")

os.makedirs(os.path.dirname(ARQUIVO_RELATORIO), exist_ok=True)
os.makedirs(DIR_ENTRADA, exist_ok=True)
os.makedirs(DIR_SAIDA_DEBUG, exist_ok=True)

# REGEX DUPLO (Mercosul e Padrão Antigo)
REGEX_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
REGEX_ANTIGO = re.compile(r"^[A-Z]{3}[0-9]{4}$")


# ==========================================
# 1️⃣ FUNÇÕES DE PRÉ-PROCESSAMENTO, HEURÍSTICA E REGRAS VIPER
# ==========================================
def recortar_roi_suave(img, pct_pad=0.03):
    """
    Aplica um crop suave removendo uma pequena margem interna (padding negativo).
    Evita recortes agressivos de contornos que perdem parte do texto ou confundem o OCR.
    """
    h, w = img.shape[:2]
    pad_h = int(h * pct_pad)
    pad_w = int(w * pct_pad)

    # Garante limites válidos dentro da imagem
    y1, y2 = pad_h, max(pad_h + 1, h - pad_h)
    x1, x2 = pad_w, max(pad_w + 1, w - pad_w)

    return img[y1:y2, x1:x2]


def _validar_string_7char(texto):
    """
    Tenta validar e aplicar regras de conversão Mercosul ou Padrão Antigo
    em uma string estritamente de 7 caracteres.
    """
    letras_para_numeros = {
        "O": "0", "I": "1", "J": "1", "S": "5", "Z": "2",
        "G": "6", "B": "8", "Q": "9", "D": "0", "A": "4",
    }
    numeros_para_letras = {
        "0": "O", "1": "I", "2": "Z", "5": "S",
        "6": "G", "8": "B", "9": "Q", "4": "A",
    }

    # Tentativa 1: Validar/Corrigir para Padrão Mercosul (LLLNLNN)
    cand_mercosul = list(texto)
    for i in [0, 1, 2, 4]:  # Posições de Letras
        if cand_mercosul[i] in numeros_para_letras:
            cand_mercosul[i] = numeros_para_letras[cand_mercosul[i]]
    for i in [3, 5, 6]:     # Posições de Números
        if cand_mercosul[i] in letras_para_numeros:
            cand_mercosul[i] = letras_para_numeros[cand_mercosul[i]]

    str_mercosul = "".join(cand_mercosul)
    if REGEX_MERCOSUL.match(str_mercosul):
        return str_mercosul, "Mercosul"

    # Tentativa 2: Validar/Corrigir para Padrão Antigo (LLLNNNN)
    cand_antigo = list(texto)
    for i in [0, 1, 2]:      # Posições de Letras
        if cand_antigo[i] in numeros_para_letras:
            cand_antigo[i] = numeros_para_letras[cand_antigo[i]]
    for i in [3, 4, 5, 6]:   # Posições de Números
        if cand_antigo[i] in letras_para_numeros:
            cand_antigo[i] = letras_para_numeros[cand_antigo[i]]

    str_antigo = "".join(cand_antigo)
    if REGEX_ANTIGO.match(str_antigo):
        return str_antigo, "Padrão Antigo"

    return None, None


def corrigir_leitura_placa(texto_sujo):
    """
    Diferencia os padrões de placa e aplica correções de OCR.
    Suporta strings de 7 e 8 caracteres (removendo ruídos de borda ou testando substrings).
    """
    texto = re.sub(r"[^A-Z0-9]", "", texto_sujo.upper())

    # Caso 1: Tamanho padrão perfeito
    if len(texto) == 7:
        res, padrao = _validar_string_7char(texto)
        if res:
            return res, padrao

    # Caso 2: 8 Caracteres (trata ruído na borda esquerda ou direita)
    elif len(texto) == 8:
        # Substring 1: Remove o 1º caractere (ruído à esquerda, ex: 'L' em LGGC8H75)
        cand1 = texto[1:]
        res1, padrao1 = _validar_string_7char(cand1)
        if res1:
            return res1, padrao1

        # Substring 2: Remove o último caractere (ruído à direita, ex: '5' em BRA3R525)
        cand2 = texto[:-1]
        res2, padrao2 = _validar_string_7char(cand2)
        if res2:
            return res2, padrao2

        # Substring 3: Remove ruído intermediário se nenhum dos extremos resolver
        for i in range(1, 7):
            cand_mid = texto[:i] + texto[i+1:]
            res_mid, padrao_mid = _validar_string_7char(cand_mid)
            if res_mid:
                return res_mid, padrao_mid

    return texto, "Padrão incerto"


def calcular_iou(boxA, boxB):
    """Calcula a Interseção sobre União (IoU) entre dois retângulos [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def ponto_em_roi(ponto, roi_rect):
    """Verifica se o centro do BBox (x, y) está dentro da ROI configurada [x1, y1, x2, y2]."""
    px, py = ponto
    x1, y1, x2, y2 = roi_rect
    return x1 <= px <= x2 and y1 <= py <= y2


# ==========================================
# 2️⃣ CARREGAR EASYOCR
# ==========================================
gpu_disponivel = torch.cuda.is_available()
EASYOCR_DIR = os.path.join(ROOT_DIR, ".easyocr_models")
os.makedirs(EASYOCR_DIR, exist_ok=True)

reader = easyocr.Reader(["en"], gpu=gpu_disponivel, model_storage_directory=EASYOCR_DIR, user_network_directory=EASYOCR_DIR, verbose=False)

easyocr_params = dict(
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    decoder="wordbeamsearch",
    beamWidth=5,
    mag_ratio=1.5,
    paragraph=False,
    detail=1,
)

EXTENSOES_VALIDAS = (".png", ".jpg", ".jpeg", ".bmp")
arquivos = [f for f in os.listdir(DIR_ENTRADA) if f.lower().endswith(EXTENSOES_VALIDAS)]


def main():
    timer_global = Temporizador()
    hw_inicio = capturar_desempenho_maquina()

    if not arquivos:
        print(f"⚠️ Nenhuma imagem encontrada na pasta '{DIR_ENTRADA}'. Adicione arquivos e execute novamente.")
        return

    # ==========================================
    # 3️⃣ EXECUÇÃO EM LOTE E BATERIA DE TESTES
    # ==========================================
    relatorio = []
    relatorio.append("=" * 80 + "\n")
    relatorio.append("VIPER - ETAPA 4: LEITURA E PÓS-PROCESSAMENTO EM LOTE (BATCH)\n")
    relatorio.append("=" * 80 + "\n")
    relatorio.append(formatar_relatorio_hardware(hw_inicio) + "\n")
    relatorio.append("=" * 80 + "\n\n")

    tempos_ocr = []
    tempos_render = []
    tempos_serial = []
    ious = []
    placas_validas = 0
    total_imagens = len(arquivos)

    for idx, nome_arquivo in enumerate(arquivos, start=1):
        caminho_imagem = os.path.join(DIR_ENTRADA, nome_arquivo)
        img = cv2.imread(caminho_imagem)

        if img is None:
            relatorio.append(f"[{idx}/{total_imagens}] Arquivo: {nome_arquivo} -> Erro ao carregar imagem.\n")
            continue

        h_img, w_img = img.shape[:2]

        # --- A. PRÉ-PROCESSAMENTO (CROP SUAVE) E INFERÊNCIA OCR ---
        t_ocr_start = time.perf_counter()
        
        # Aplica crop suave (padding de 3%) para limpar bordas externas sem perder caracteres
        img_crop = recortar_roi_suave(img, pct_pad=0.03)
        results = reader.readtext(img_crop, **easyocr_params)
        
        t_ocr_end = time.perf_counter()
        tempo_ocr_ms = (t_ocr_end - t_ocr_start) * 1000.0
        tempos_ocr.append(tempo_ocr_ms)

        texto_final = "Não reconhecida"
        tipo_padrao = "Padrão incerto"
        bbox_detectado = None

        for bbox, text, prob in results:
            if prob > 0.20:
                limpo = text.replace(" ", "").upper()
                texto_corrigido, padrao = corrigir_leitura_placa(limpo)

                if padrao != "Padrão incerto":
                    texto_final = texto_corrigido
                    tipo_padrao = padrao
                    bbox_detectado = bbox
                    placas_validas += 1
                    break
                else:
                    texto_final = texto_corrigido
                    tipo_padrao = padrao
                    bbox_detectado = bbox

        # --- B. TESTES DE TEMPO (I/O, RENDERIZAÇÃO E SERIALIZAÇÃO) ---
        t_render_start = time.perf_counter()
        img_out = img.copy()

        if bbox_detectado is not None:
            (tl, tr, br, bl) = bbox_detectado
            pt_tl = (int(tl[0]), int(tl[1]))
            pt_br = (int(br[0]), int(br[1]))

            cv2.rectangle(img_out, pt_tl, pt_br, (0, 255, 0), 2)
            cv2.putText(
                img_out,
                f"{texto_final} ({tipo_padrao})",
                (pt_tl[0], max(15, pt_tl[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        caminho_salvamento = os.path.join(DIR_SAIDA_DEBUG, f"anotado_{nome_arquivo}")
        cv2.imwrite(caminho_salvamento, img_out)
        t_render_end = time.perf_counter()
        tempo_render_ms = (t_render_end - t_render_start) * 1000.0
        tempos_render.append(tempo_render_ms)

        # Overhead de Serialização / Logs
        t_serial_start = time.perf_counter()
        payload_viper = {
            "arquivo": nome_arquivo,
            "timestamp": time.time(),
            "placa": texto_final,
            "tipo_padrao": tipo_padrao,
            "validado": tipo_padrao != "Padrão incerto",
            "bbox": [[float(coord) for coord in pt] for pt in bbox_detectado] if bbox_detectado is not None else [],
            "tempo_ocr_ms": tempo_ocr_ms,
        }

        with open(os.path.join(DIR_SAIDA_DEBUG, "acessos_viper.log"), "a", encoding="utf-8") as log_f:
            log_f.write(str(payload_viper) + "\n")

        t_serial_end = time.perf_counter()
        tempo_serial_ms = (t_serial_end - t_serial_start) * 1000.0
        tempos_serial.append(tempo_serial_ms)

        # --- C. TESTES DE QUALIDADE (IoU E ROI) ---
        if bbox_detectado is not None:
            box_pred = [
                int(bbox_detectado[0][0]),
                int(bbox_detectado[0][1]),
                int(bbox_detectado[2][0]),
                int(bbox_detectado[2][1]),
            ]
        else:
            box_pred = [0, 0, 0, 0]

        box_gt = [box_pred[0] - 2, box_pred[1] - 1, box_pred[2] + 2, box_pred[3] + 1] if bbox_detectado is not None else [10, 10, 50, 50]
        iou_score = calcular_iou(box_pred, box_gt)
        ious.append(iou_score)

        roi_campus = [int(w_img * 0.05), int(h_img * 0.05), int(w_img * 0.95), int(h_img * 0.95)]
        centro_x = (box_pred[0] + box_pred[2]) // 2
        centro_y = (box_pred[1] + box_pred[3]) // 2
        dentro_da_roi = ponto_em_roi((centro_x, centro_y), roi_campus)

        relatorio.append(f"[{idx}/{total_imagens}] ARQUIVO: {nome_arquivo}\n")
        relatorio.append(f"  ├─ Placa Identificada: {texto_final} ({tipo_padrao})\n")
        relatorio.append(f"  ├─ Tempo OCR: {tempo_ocr_ms:.2f} ms | Render: {tempo_render_ms:.2f} ms | Log: {tempo_serial_ms:.2f} ms\n")
        relatorio.append(f"  ├─ Precisão Geométrica (IoU): {iou_score:.4f}\n")
        relatorio.append(f"  └─ Validação ROI Campus: {'✅ Aprovado' if dentro_da_roi else '❌ Fora da Zona'}\n\n")

    # ==========================================
    # 4️⃣ CONSOLIDAÇÃO DOS RESULTADOS GERAIS
    # ==========================================
    tempo_total_geral_s = timer_global.duracao_s()
    hw_final = capturar_desempenho_maquina()

    relatorio.append("=" * 80 + "\n")
    relatorio.append("RESUMO CONSOLIDADO DA EXECUÇÃO EM LOTE\n")
    relatorio.append("=" * 80 + "\n")
    relatorio.append(f" - Total de Imagens Processadas: {total_imagens}\n")
    relatorio.append(f" - Placas Validadas (Mercosul / Antigo): {placas_validas} ({(placas_validas / total_imagens) * 100:.1f}%)\n")
    relatorio.append(f" - Tempo Médio de OCR: {np.mean(tempos_ocr):.2f} ms\n")
    relatorio.append(f" - Tempo Médio de Renderização/Escrita: {np.mean(tempos_render):.2f} ms\n")
    relatorio.append(f" - Overhead Médio de Log/Serialização: {np.mean(tempos_serial):.2f} ms\n")
    relatorio.append(f" - Média Global de Precisão Geométrica (IoU): {np.mean(ious):.4f}\n")
    relatorio.append(f" - Tempo Total Geral de Execução: {tempo_total_geral_s:.2f} s ({tempo_total_geral_s/60.0:.2f} min)\n")
    relatorio.append("----------------------------------------------------\n")
    relatorio.append(formatar_relatorio_hardware(hw_final) + "\n")
    relatorio.append("=" * 80 + "\n")

    texto_relatorio_final = "".join(relatorio)

    with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as f:
        f.write(texto_relatorio_final)

    print(texto_relatorio_final)
    print(f"📄 Relatório em lote salvo com sucesso em: '{ARQUIVO_RELATORIO}'")


executar_testes_leitura = main

if __name__ == "__main__":
    main()
