import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Permite importar o módulo telemetria quando executado da raiz ou da pasta novos_codigos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from telemetria import capturar_desempenho_maquina, formatar_relatorio_hardware, Temporizador

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIR_ENTRADA = os.path.join(ROOT_DIR, "fotos_carros")
DIR_PRONTAS = os.path.join(ROOT_DIR, "placas_finalizadas")
ARQUIVO_RELATORIO = os.path.join(ROOT_DIR, "relatorio_testes_preprocessamento.txt")

os.makedirs(DIR_ENTRADA, exist_ok=True)
os.makedirs(DIR_PRONTAS, exist_ok=True)

# MODELOS
yolo_veiculo_path = os.path.join(ROOT_DIR, "yolov8n.pt")
yolo_placa_path = os.path.join(ROOT_DIR, "runs", "detect", "placa_detector", "weights", "best.pt")
if not os.path.exists(yolo_placa_path):
    alt_path = os.path.join(ROOT_DIR, "runs", "detect", "runs", "detect", "placa_detector", "weights", "best.pt")
    if os.path.exists(alt_path):
        yolo_placa_path = alt_path

yolo_veiculo = YOLO(yolo_veiculo_path)
yolo_placa = YOLO(yolo_placa_path if os.path.exists(yolo_placa_path) else yolo_veiculo_path)

CLASSES_VEICULOS = [2, 3, 5, 7]


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

    # Margem de segurança
    crop_padded = cv2.copyMakeBorder(
        crop_placa, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Correção de perspectiva (Warp)
    quad = encontrar_quadrilatero_placa(crop_padded)
    placa_alinhada = (
        warp_placa(crop_padded, quad) if quad is not None else crop_padded
    )

    # Redimensionamento mantendo aspect ratio
    h, w = placa_alinhada.shape[:2]
    target_w = 400
    target_h = int(h * (target_w / float(w))) if w > 0 else 100
    resized = cv2.resize(
        placa_alinhada, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )

    # PRÉ-PROCESSAMENTO:
    # 1. Escala de cinza
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 2. Suavização Gaussiana para remover ruídos pequenos do fundo
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Binarização de Otsu (Cálculo automático do threshold ideal)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


# ==============================================================================
# SUÍTE DE TESTES E EXECUÇÃO
# ==============================================================================
def executar_testes_e_gerar_relatorio():
    timer_global = Temporizador()
    hw_inicio = capturar_desempenho_maquina()

    arquivos = [
        f
        for f in os.listdir(DIR_ENTRADA)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not arquivos:
        print(f"⚠️ Nenhuma imagem encontrada na pasta '{DIR_ENTRADA}'.")
        return

    relatorio_linhas = []
    relatorio_linhas.append("=" * 70 + "\n")
    relatorio_linhas.append("RELATÓRIO DE DESEMPENHO E QUALIDADE DE PRÉ-PROCESSAMENTO\n")
    relatorio_linhas.append("=" * 70 + "\n")
    relatorio_linhas.append(formatar_relatorio_hardware(hw_inicio) + "\n")
    relatorio_linhas.append("=" * 70 + "\n\n")

    # 1. TESTE DE TEMPO E GARGALOS (I/O vs CPU)
    timer_etapa1 = Temporizador()
    tempos_io = []
    tempos_cpu = []

    for arq in arquivos:
        caminho_img = os.path.join(DIR_ENTRADA, arq)

        t0 = time.perf_counter()
        img = cv2.imread(caminho_img)
        t1 = time.perf_counter()
        tempos_io.append(t1 - t0)

        if img is None:
            continue

        t2 = time.perf_counter()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lb, r, pad = letterbox(img_rgb, new_shape=(640, 640))
        img_norm = img_lb.astype(np.float32) / 255.0
        t3 = time.perf_counter()
        tempos_cpu.append(t3 - t2)

    duracao_etapa1_s = timer_etapa1.duracao_s()
    media_io_ms = (sum(tempos_io) / len(tempos_io)) * 1000 if tempos_io else 0.0
    media_cpu_ms = (sum(tempos_cpu) / len(tempos_cpu)) * 1000 if tempos_cpu else 0.0

    relatorio_linhas.append("[1] TESTE DE TEMPO POR OPERAÇÃO UNITÁRIA\n")
    relatorio_linhas.append(f" - Imagens testadas: {len(arquivos)}\n")
    relatorio_linhas.append(
        f" - Média de I/O de disco (cv2.imread): {media_io_ms:.2f} ms\n"
    )
    relatorio_linhas.append(
        f" - Média de operações de CPU (RGB + Letterbox + Normalização): {media_cpu_ms:.2f} ms\n"
    )
    relatorio_linhas.append(
        f" - Razão I/O / CPU: {(media_io_ms / media_cpu_ms if media_cpu_ms > 0 else 0):.2f}x\n"
    )
    relatorio_linhas.append(f" - Tempo total da etapa 1: {duracao_etapa1_s:.2f} s\n\n")

    # 2. THROUGHPUT DO DATALOADER
    timer_etapa2 = Temporizador()
    loop_count = 0
    for _ in range(3):
        for arq in arquivos:
            caminho_img = os.path.join(DIR_ENTRADA, arq)
            img = cv2.imread(caminho_img)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _ = letterbox(img_rgb, new_shape=(640, 640))
                loop_count += 1

    tempo_total_tp = timer_etapa2.duracao_s()
    throughput = loop_count / tempo_total_tp if tempo_total_tp > 0 else 0

    relatorio_linhas.append(
        "[2] THROUGHPUT DO PREPROCESSAMENTO (SEM INFERÊNCIA)\n"
    )
    relatorio_linhas.append(f" - Total de iterações: {loop_count}\n")
    relatorio_linhas.append(
        f" - Tempo total gasto: {tempo_total_tp:.2f} segundos\n"
    )
    relatorio_linhas.append(
        f" - Throughput de pré-processamento: {throughput:.2f} img/s\n\n"
    )

    # 3. TESTE DE QUALIDADE E DENSIDADE POR RESOLUÇÃO (IMGSZ)
    timer_etapa3 = Temporizador()
    resolucoes = [(320, 320), (416, 416), (640, 640)]
    relatorio_linhas.append(
        "[3] AVALIAÇÃO DE PRESERVAÇÃO DE RESOLUÇÃO (IMGSZ)\n"
    )

    for arq in arquivos:
        caminho_img = os.path.join(DIR_ENTRADA, arq)
        img = cv2.imread(caminho_img)
        if img is None:
            continue

        h_orig, w_orig = img.shape[:2]
        relatorio_linhas.append(
            f"   • Arquivo: {arq} (Original: {w_orig}x{h_orig})\n"
        )

        for imgsz in resolucoes:
            img_lb, ratio, (dw, dh) = letterbox(img, new_shape=imgsz)
            h_proc, w_proc = img_lb.shape[:2]

            aspect_orig = float(w_orig) / h_orig
            w_util = w_proc - (2 * dw)
            h_util = h_proc - (2 * dh)
            aspect_proc = float(w_util) / h_util if h_util > 0 else 0
            distorcao = abs(aspect_orig - aspect_proc)

            relatorio_linhas.append(
                f"     - Resolução {imgsz[0]}x{imgsz[1]}: Fator de Escala={ratio:.4f} | Distorção Aspect Ratio={distorcao:.4f}\n"
            )
        relatorio_linhas.append("\n")

    duracao_etapa3_s = timer_etapa3.duracao_s()
    relatorio_linhas.append(f" - Tempo total da avaliação de resoluções: {duracao_etapa3_s:.2f} s\n\n")

    # 4. EXECUÇÃO DO PIPELINE DE DETECÇÃO/PRÉ-PROCESSAMENTO
    timer_etapa4 = Temporizador()
    relatorio_linhas.append(
        "[4] RESULTADO DA EXECUÇÃO DO PIPELINE DE DETECÇÃO/PRÉ-PROCESSAMENTO\n"
    )
    placas_processadas = 0

    for arq in arquivos:
        caminho_img = os.path.join(DIR_ENTRADA, arq)
        img_carro = cv2.imread(caminho_img)

        if img_carro is None:
            continue

        res_veiculo = yolo_veiculo(
            img_carro, verbose=False, classes=CLASSES_VEICULOS
        )
        crops_veiculos = []

        if len(res_veiculo[0].boxes) > 0:
            for box_v in res_veiculo[0].boxes.xyxy.cpu().numpy().astype(int):
                crops_veiculos.append(
                    img_carro[box_v[1] : box_v[3], box_v[0] : box_v[2]]
                )

        # Fallback primário: Adiciona a imagem inteira às opções de busca da placa via YOLO
        crops_veiculos.append(img_carro)

        placa_encontrada = False
        for crop_v in crops_veiculos:
            res_placa = yolo_placa(crop_v, verbose=False)

            if len(res_placa[0].boxes) > 0:
                box_p = res_placa[0].boxes.xyxy[0].cpu().numpy().astype(int)
                crop_placa = crop_v[box_p[1] : box_p[3], box_p[0] : box_p[2]]

                placa_limpa = preprocessar_crop_para_ocr(crop_placa)

                if placa_limpa is not None:
                    caminho_saida = os.path.join(DIR_PRONTAS, f"limpa_{arq}")
                    cv2.imwrite(caminho_saida, placa_limpa)
                    placas_processadas += 1
                    relatorio_linhas.append(
                        f"   - ✅ {arq}: Placa recortada e binarizada via YOLO salva em '{caminho_saida}'\n"
                    )
                    placa_encontrada = True
                    break

        # FALLBACK SECUNDÁRIO: Se o YOLO não achar nada no crop ou na imagem inteira
        if not placa_encontrada:
            quad = encontrar_quadrilatero_placa(img_carro)
            if quad is not None:
                crop_direto = warp_placa(img_carro, quad)
                placa_limpa = preprocessar_crop_para_ocr(crop_direto)
                if placa_limpa is not None:
                    caminho_saida = os.path.join(DIR_PRONTAS, f"limpa_{arq}")
                    cv2.imwrite(caminho_saida, placa_limpa)
                    placas_processadas += 1
                    relatorio_linhas.append(
                        f"   - 🔄 {arq}: Placa recuperada via Fallback de Contornos salva em '{caminho_saida}'\n"
                    )
                    placa_encontrada = True

        if not placa_encontrada:
            relatorio_linhas.append(
                f"   - ❌ {arq}: Falha ao localizar a placa por YOLO e por Contornos.\n"
            )

    duracao_etapa4_s = timer_etapa4.duracao_s()
    hw_final = capturar_desempenho_maquina()
    tempo_total_geral_s = timer_global.duracao_s()

    relatorio_linhas.append(
        f"\nTotal de placas finalizadas com sucesso: {placas_processadas}/{len(arquivos)}\n"
    )
    relatorio_linhas.append(f"Tempo da etapa de detecção/binarização: {duracao_etapa4_s:.2f} s\n")
    relatorio_linhas.append(f"Tempo total dos testes de pré-processamento: {tempo_total_geral_s:.2f} s\n\n")
    relatorio_linhas.append("=" * 70 + "\n")
    relatorio_linhas.append(formatar_relatorio_hardware(hw_final) + "\n")
    relatorio_linhas.append("=" * 70 + "\n")

    # GRAVAÇÃO DO RELATÓRIO NO DISCO
    relatorio_texto = "".join(relatorio_linhas)
    with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as f:
        f.write(relatorio_texto)

    print(relatorio_texto)
    print(f"📄 Relatório atualizado salvo em: '{ARQUIVO_RELATORIO}'")


if __name__ == "__main__":
    executar_testes_e_gerar_relatorio()
