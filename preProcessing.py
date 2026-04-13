import cv2
import numpy as np

def localizar_placa(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) 
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    v = np.median(blur)
    low = int(max(0, 0.33 * v))
    high = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    melhor = None
    melhor_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            if w > h * 2:
                if area > melhor_area:
                    melhor_area = area
                    melhor = approx
    return melhor

def warp_plate(img, quad):
    pts = quad.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    rect = np.array([tl, tr, br, bl], dtype="float32")
    w1, w2 = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
    h1, h2 = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
    width, height = int(max(w1, w2)), int(max(h1, h2))
    dst = np.array([[0,0], [width,0], [width,height], [0,height]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (width, height))

def tratar_especifico_mercosul(img_warp):
    h, w = img_warp.shape[:2]
    corte_superior = int(h * 0.22) 
    img_sem_topo = img_warp[corte_superior:h, 0:w]
    largura_alvo = 600 
    proporcao = largura_alvo / float(w)
    altura_alvo = int((h - corte_superior) * proporcao)
    img_grande = cv2.resize(img_sem_topo, (largura_alvo, altura_alvo), interpolation=cv2.INTER_CUBIC)
    
    return img_grande

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(thresh, -1, kernel)

    return sharp

img = cv2.imread("placa_0.jpg")

if img is None:
    print("Erro: Não foi possível carregar 'placa_0.jpg'. Verifique o caminho.")
else:
    img_with_padding = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    quad = localizar_placa(img_with_padding)

    if quad is not None:
        alinhada = warp_plate(img_with_padding, quad)
        
        mercosul_focada = tratar_especifico_mercosul(alinhada)
        final = preprocess(mercosul_focada)

        cv2.imwrite("Placa_Final_Pronta.png", final)
        cv2.imshow("1. Alinhada Original", alinhada)
        cv2.imshow("2. Sem Faixa e Ampliada", mercosul_focada)
        cv2.imshow("3. Resultado para OCR", final)
        print("✅ Processamento concluído com sucesso!")
        cv2.waitKey(0)
    else:
        print("⚠️ Não foi possível localizar os contornos da placa.")
        cv2.imshow("Imagem que falhou", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()