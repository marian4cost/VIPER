# 🚗 VIPER — Identificação de Placas Veiculares em Imagens Estáticas

O VIPER é um sistema de Reconhecimento Automático de Placas Veiculares (ALPR) que detecta veículos, localiza a placa e realiza a leitura dos caracteres de forma automatizada.
O projeto foi desenvolvido com foco em campi universitários, oferecendo uma solução de baixo custo, eficiente e escalável para segurança e controle de acesso.

✅ 🎯 Objetivo do Projeto

Desenvolver um sistema capaz de:

Detectar veículos em imagens

Localizar automaticamente a placa

Pré-processar a imagem para aumentar legibilidade

Reconhecer os caracteres da placa via OCR

Operar com hardware simples e econômico (ex: Raspberry Pi)

🧩 🚀 Motivação

Campi universitários e pequenos condomínios sofrem com:

Falta de controle de acesso eficiente

Sistemas comerciais caros

Necessidade crescente de segurança patrimonial

O VIPER surge como alternativa viável, acessível e nacional, alinhada às demandas de Smart Cities e IoT.

🧠 🛠️ Tecnologias Utilizadas
Componente	Função
YOLOv8	Detecção do veículo
OpenCV	Pré-processamento da imagem
EasyOCR	Reconhecimento dos caracteres da placa
Tkinter	Interface gráfica
Python	Linguagem principal
Raspberry Pi (futuro)	Execução embarcada e de baixo consumo
🔎 📦 Funcionamento do Sistema

Fluxo resumido:

Upload/entrada da imagem

YOLO detecta o veículo

A região onde a placa deve estar é estimada

A placa é recortada

A imagem passa por pré-processamento:

Grayscale

CLAHE (contraste)

Redução de ruído

Binarização

Morfologia

Redimensionamento

EasyOCR lê os caracteres

O resultado é exibido na interface

💻 🖥️ Interface

O VIPER conta com uma interface simples em Tkinter:

Botão para selecionar imagem

Exibição:

Veículo detectado

Placa recortada

Placa pré-processada

Popup com o texto reconhecido

📥 📌 Instalação
Pré-requisitos

Python 3.10 ou 3.12 (recomendado)

pip atualizado

Passos
# Criar ambiente virtual
python -m venv .venv

# Ativar
# Windows:
.venv\Scripts\activate

# Instalar dependências
pip install ultralytics
pip install opencv-python
pip install easyocr
pip install pillow

▶️ 🏁 Como Executar
python main.py

✅ Resultados Esperados

Baixo custo operacional

Funcionamento em tempo quase real

Alta acessibilidade energética (quando embarcado)

Aplicação distribuída e escalável

🔐 📚 Privacidade e Segurança

O projeto considera:

Criptografia dos dados (fase futura)

Controle de acesso

Respeito à LGPD

Armazenamento mínimo e seguro

🌎 Aplicações Reais

Campi universitários

Estacionamentos

Condomínios

Empresas de pequeno porte

Controle logístico

🔮 Próximos Passos

Implantação embarcada em Raspberry Pi

Banco de placas cadastradas

Envio de dados para nuvem (Render/Railway)

Alarme e monitoramento em tempo real

Suporte a câmera IP

👤 👥 Autores

Marcos Vinicius S. Melo

Mariana C. Lisboa

Rubens de S. Matos Júnior

Alfredo M. Vieira

Instituto Federal de Educação, Ciência e Tecnologia de Sergipe – Campus Lagarto
Projeto desenvolvido para a SNCT.
