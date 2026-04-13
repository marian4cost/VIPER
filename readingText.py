import cv2
import easyocr
import re

def corrigir_leitura_mercosul(texto_sujo):
    texto = re.sub(r'[^A-Z0-9]', '', texto_sujo.upper())
    
    if len(texto) != 7:
        return texto 

    letras_para_numeros = {
        'O': '0', 'I': '1', 'J': '1', 'S': '5', 'Z': '2', 
        'G': '6', 'B': '8', 'Q': '9', 'D': '0'
    }
    numeros_para_letras = {
        '0': 'O', '1': 'I', '2': 'Z', '5': 'S', 
        '6': 'G', '8': 'B', '9': 'Q'
    }

    lista_char = list(texto)

    indices_letras = [0, 1, 2, 4]
    indices_numeros = [3, 5, 6]

    for i in indices_letras:
        char = lista_char[i]
        if char in numeros_para_letras:
            lista_char[i] = numeros_para_letras[char]
    for i in indices_numeros:
        char = lista_char[i]
        if char in letras_para_numeros:
            lista_char[i] = letras_para_numeros[char]

    return "".join(lista_char)

reader = easyocr.Reader(["en"], gpu=True, verbose=False)

easyocr_params = dict(
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    decoder="wordbeamsearch",
    beamWidth=5,           
    mag_ratio=1.5,         
    paragraph=False,
    detail=1
)

image_path = "Placa_Final_Pronta.png"
img = cv2.imread(image_path)

if img is None:
    print("Erro ao abrir imagem.")
    exit()

results = reader.readtext(img, **easyocr_params)

texto_final = "Não reconhecida"

for (bbox, text, prob) in results:
    if prob > 0.20:
        limpo = text.replace(" ", "").upper()
        print(f"Bruto do OCR: {limpo} (Confiança: {prob:.2f})")
        
        texto_corrigido = corrigir_leitura_mercosul(limpo)
        
        if re.match(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', texto_corrigido):
            texto_final = texto_corrigido
            break 
        else:
            texto_final = f"{texto_corrigido} (Padrão incerto)"

print("\n=========================")
print(f"📌 PLACA IDENTIFICADA: {texto_final}")
print("=========================\n")

for (bbox, text, prob) in results:
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    br = (int(br[0]), int(br[1]))
    cv2.rectangle(img, tl, br, (0, 255, 0), 2)
    cv2.putText(img, texto_final, (tl[0], tl[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Resultado Final OCR", img)
cv2.waitKey(0)
cv2.destroyAllWindows()