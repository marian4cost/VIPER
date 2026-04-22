# 🔍 Sistema de Identificação de Placas Veiculares com YOLOv8

Este projeto consiste em um sistema de ponta a ponta para **Detecção, Rastreamento e Reconhecimento de Caracteres (OCR)** de placas veiculares no padrão Mercosul. O sistema utiliza modelos de **Deep Learning** de última geração para monitorar fluxos de tráfego de forma automatizada.

## 🚀 Funcionalidades

- **Rastreamento em Tempo Real**  
  Identificação e monitoramento de veículos via ByteTrack (YOLOv8).

- **Gatilho por Região de Interesse (ROI)**  
  Captura otimizada de fotos apenas quando o veículo cruza a zona de melhor visibilidade.

- **Pipeline de Visão Computacional**  
  Tratamento de imagem com alinhamento de perspectiva (Warp), binarização de Otsu e filtros de nitidez.

- **OCR com Heurística Mercosul**  
  Reconhecimento de caracteres com correção automática baseada no padrão `LLLNLNN`.

- **Persistência de Dados**  
  Geração automática de registros em planilhas Excel (`.xlsx`) e armazenamento de evidências fotográficas.

## 🏗️ Arquitetura do Sistema

O projeto é dividido em dois motores independentes que operam de forma assíncrona:

### 1. Motor de Captura (`monitoramento_track.py`)

Responsável pela ingestão do vídeo, detecção de veículos e salvamento de recortes de alta qualidade.

- **Entrada:** Arquivo de vídeo (`.mp4`) ou Stream IP  
- **Saída:** Recortes de veículos salvos na pasta `/fotos_carros`

### 2. Motor de Processamento (`main_processador.py`)

Responsável pela inteligência do sistema. Monitora a pasta de entrada e processa cada imagem sequencialmente.

- **Entrada:** Fotos brutas dos veículos  
- **Saída:**  
  - Recorte da placa tratada  
  - Leitura OCR  
  - Registro em planilha  

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.9+  
- **Detecção/Tracking:** YOLOv8 (Ultralytics)  
- **OCR:** EasyOCR  
- **Visão Computacional:** OpenCV, NumPy  
- **Manipulação de Dados:** Pandas, Openpyxl  

## 📋 Pré-requisitos e Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Instale as dependências
```bash
pip install ultralytics easyocr opencv-python pandas openpyxl
```

### 3. Modelos
Certifique-se de ter o arquivo best.pt (modelo treinado de placas) no diretório indicado no código.

🚦 Como Utilizar

1. Inicie o Motor de Captura
```bash
python track.py
```
3. Inicie o Motor de Processamento (em outro terminal)
```bash
python main.py
```

📂 Saídas do Sistema
```bash
registro_placas.xlsx → Planilha com todos os registros
/placas_finalizadas → Imagens das placas após pré-processamento
```

📈 Pipeline de Processamento de Imagem

Para garantir maior precisão no OCR, cada placa passa pelas seguintes etapas:

- Localização
- Detecção da bounding box da placa
- Warp Perspective
- Alinhamento geométrico para visualização frontal
- Remoção de Topo
- Exclusão da faixa azul "Brasil" (padrão Mercosul)
- Binarização
- Conversão para escala de cinza + Threshold de Otsu
- Sharpening
- Aumento de nitidez para destacar os caracteres

## 👨‍💻 Autor
Mariana da Costa Lisboa
