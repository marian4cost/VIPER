import os
from pathlib import Path
from ultralytics import YOLO

def auto_label_dataset():
    # 1. Carrega o modelo YOLOv8 padrão (pré-treinado no dataset COCO)
    # Ele já detecta carros, veículos e outros objetos comuns muito bem.
    print("🚀 Carregando modelo pré-treinado...")
    model = YOLO("yolov8n.pt")

    # 2. Mapeamento das pastas
    base_dir = Path(__file__).parent.resolve()
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"

    # Splits que você possui no seu projeto
    splits = ["train", "valid", "test"]

    # Caso sua pasta de validação se chame 'val' em vez de 'valid':
    if not (images_dir / "valid").exists() and (images_dir / "val").exists():
        splits[1] = "val"

    for split in splits:
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split

        if not img_split_dir.exists():
            print(f"⚠️ Pasta de imagem não encontrada: {img_split_dir}, pulando...")
            continue

        # Garante que a pasta de labels destino exista
        lbl_split_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(img_split_dir.glob("*.[jJ][pP][gG]")) + \
                      list(img_split_dir.glob("*.[jJ][pP][eE][gG]")) + \
                      list(img_split_dir.glob("*.[pP][nN][gG]"))

        print(f"\n📂 Processando '{split}': {len(image_files)} imagens encontradas...")

        for img_path in image_files:
            # Roda a inferência com limiar de confiança de 25%
            results = model.predict(source=str(img_path), conf=0.25, verbose=False)
            
            txt_filename = img_path.stem + ".txt"
            txt_path = lbl_split_dir / txt_filename

            lines = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Mapeia tudo para a classe 0 ('placa') no seu data.yaml
                    cls_id = 0  
                    
                    # Converte para formato YOLO normalizado: (x_center, y_center, width, height)
                    xywhn = box.xywhn[0].tolist()
                    x_center, y_center, w, h = xywhn
                    
                    lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

            # Escreve o arquivo .txt (se nenhuma caixa for detectada, cria um arquivo vazio)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

    print("\n✅ Auto-anotação concluída com sucesso para todas as imagens!")

if __name__ == "__main__":
    auto_label_dataset()