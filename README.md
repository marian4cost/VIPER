# 🐍 VIPER — Veículos e Identificação de Placas com Reconhecimento

Sistema integrado e de alto desempenho para **Detecção, Rastreamento, Pré-processamento Geométrico, Leitura OCR e Telemetria de Hardware** aplicado ao monitoramento veicular e identificação automatizada de placas (padrão **Mercosul** e **padrão antigo brasileiro**).

---

## 🎯 Visão Geral

O **VIPER** combina modelos de Deep Learning de última geração (YOLOv8) com técnicas avançadas de Visão Computacional clássica (Warp Perspective, binarização adaptativa de Otsu, morfologia matemática) e OCR (EasyOCR) suportado por motor heurístico com regex duplo.

Além disso, conta com um módulo de **Telemetria de Hardware em Tempo Real**, que registra consumo de CPU, RAM do sistema, alocação de VRAM de GPU, latência de inferência (pré-processamento, forward pass e pós-processamento/NMS) e taxa de quadros (*FPS real*).

---

## 🏗️ Arquitetura do Pipeline

```
[Imagem / Fluxo de Vídeo]
          │
          ▼
┌────────────────────────┐
│  1. Detecção Veicular  │  YOLOv8 (Classes: carro, moto, ônibus, caminhão)
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│  2. Detecção de Placa  │  YOLOv8 Customizado (placa_detector) + Fallback Geométrico
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│  3. Pré-Processamento  │  Warp Perspective (correção de ângulo) +
│     & Binarização      │  Letterbox + Suavização Gaussiana + Binarização de Otsu
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│  4. Leitura OCR &      │  EasyOCR + Heurística VIPER (correção de ruídos O/0, I/1, etc.)
│     Validação Heurística│  Validação de padrão Mercosul (ABC1D23) e Antigo (ABC1234)
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│  5. Telemetria & Logs  │  Monitoramento de Hardware (CPU, RAM, VRAM, Latência ms, FPS)
│     Persistência       │  Persistência em log estruturado e relatórios executivos
└────────────────────────┘
```

---

## 📁 Estrutura de Pastas do Projeto

```plaintext
viper/
├── src/                               # 🧠 Núcleo de Código Modular do Projeto
│   ├── main.py                        # Pipeline completo integrado com telemetria
│   ├── preProcessing.py               # Módulo de pré-processamento geométrico e Otsu
│   ├── reading.py                     # Módulo de OCR e heurísticas Mercosul/Antigo
│   ├── trainYOLO.py                   # Treinamento YOLOv8 com callbacks de benchmark
│   ├── validadeYOLO.py                # Validação de métricas (mAP, F1) e benchmark FPS
│   ├── telemetria.py                  # Telemetria de CPU, RAM, VRAM e temporização
│   └── gerar_pdf_relatorio.py         # Gerador de relatório executivo em PDF
│
├── relatorios/                        # 📊 Relatórios de Execução e Benchmarks
│   ├── relatorio_viper_projeto.pdf    # Relatório executivo completo em PDF
│   ├── pipeline_main/                 # Logs consolidados de execução do pipeline
│   ├── validacao_e_metricas/          # Métricas de validação, matrizes e latência
│   ├── preprocessamento/              # Relatórios de testes de alinhamento e Otsu
│   ├── leitura_ocr/                   # Relatórios de leitura e validação OCR
│   ├── apresentacao/                  # Relatórios consolidados para apresentações
│   └── treinamento/                   # Relatórios de treinamento de épocas
│
├── historico/                         # 📜 Histórico e Códigos Legados
│   └── codigos_legados/               # Versões anteriores dos scripts (preservação histórica)
│
├── fotos_carros/                      # 🚗 Imagens de teste para validação do sistema
├── placas_processadas/                # 🖼️ Imagens das placas binarizadas após pré-processamento
├── placas_finalizadas/                # 🖼️ Placas prontas para OCR
├── debug_saida_viper/                 # 🔍 Imagens anotadas e acessos_viper.log
├── images/                            # 🗂️ Dataset formatado YOLO (train, valid, test)
├── runs/                              # 📈 Pesos do modelo treinado (placa_detector) e gráficos
│
├── main.py                            # 🚀 Ponto de entrada rápido (Pipeline Integrado)
├── preProcessing.py                   # 🚀 Ponto de entrada rápido (Pré-processamento)
├── reading.py                         # 🚀 Ponto de entrada rápido (Leitura OCR)
├── trainYOLO.py                       # 🚀 Ponto de entrada rápido (Treinamento YOLO)
├── validadeYOLO.py                    # 🚀 Ponto de entrada rápido (Validação e FPS)
├── gerar_pdf_relatorio.py             # 🚀 Ponto de entrada rápido (Gerador de PDF)
│
├── data.yaml                          # Configuração de classes e splits do YOLO
├── requirements.txt                   # Dependências do projeto atualizadas
└── TAREFAS.md                         # Checklist de organização e roadmap
```

---

## ⚡ Instalação e Configuração

### 1. Clonar o Repositório
```bash
git clone https://github.com/marian4cost/VIPER.git
cd VIPER
```

### 2. Criar e Ativar Ambiente Virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### 1. Executar o Pipeline Completo VIPER
Processa as imagens de veículos da pasta `fotos_carros/`, realiza o recorte, aplica pré-processamento com Otsu, faz o OCR, valida os caracteres e gera os logs em `debug_saida_viper/`:
```bash
python main.py
```

### 2. Testar Apenas o Pré-Processamento
Testa o alinhamento de perspectiva (`warp_placa`) e o thresholding de Otsu:
```bash
python preProcessing.py
```

### 3. Testar Apenas a Leitura OCR
Executa a validação de caracteres e as correções heurísticas sobre as placas em `placas_finalizadas/`:
```bash
python reading.py
```

### 4. Validar o Modelo e Medir FPS/Latência
Avalia as métricas de qualidade (Precision, Recall, F1-Score, mAP@0.5) e calcula a taxa real de quadros por segundo (FPS):
```bash
python validadeYOLO.py
```

### 5. Treinar o Detector de Placas
Inicia o treinamento com YOLOv8 e log de hardware/perdas a cada época:
```bash
python trainYOLO.py
```

### 6. Gerar o Relatório Técnico em PDF
Gera o relatório executivo diagramado em PDF com gráficos e tabelas:
```bash
python gerar_pdf_relatorio.py
```

---

## 📊 Heurística VIPER de Correção OCR

Devido a similaridades visuais entre caracteres alfanuméricos em placas brasileiras, o módulo `reading.py` aplica correções contextuais baseadas no padrão posicional da placa:

| Caractere Lido | Posição Esperada (Letra) | Posição Esperada (Número) |
| :---: | :---: | :---: |
| `0` ou `O` | `O` | `0` |
| `1` ou `I` | `I` | `1` |
| `2` ou `Z` | `Z` | `2` |
| `4` ou `A` | `A` | `4` |
| `5` ou `S` | `S` | `5` |
| `8` ou `B` | `B` | `8` |

- **Padrão Mercosul**: `[A-Z]{3}[0-9][A-Z][0-9]{2}` (ex.: `BRA2E19`)
- **Padrão Antigo**: `[A-Z]{3}[0-9]{4}` (ex.: `ABC1234`)

---

## 👩‍💻 Autora
**Mariana da Costa Lisboa**  
Projeto VIPER — Reconhecimento Inteligente de Veículos e Placas
