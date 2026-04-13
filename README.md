# 📘 Sistema de Identificação Automática de Placas Veiculares  
### YOLOv8 + EasyOCR — Monitoramento de Acesso ao Campus Universitário

Este projeto implementa um sistema completo para **detecção** e **leitura automática** de placas veiculares, com foco na identificação dos veículos que acessam o campus universitário.  
A solução combina visão computacional, deep learning e OCR, utilizando:

- 🟦 **YOLOv8** para Detecção da Placa  
- 🟩 **EasyOCR + Heurísticas Inteligentes** para Leitura e Correção  
- 🟧 Refinamento baseado no **padrão Mercosul**

O projeto foi estruturado em três etapas principais:

1. Treinamento do modelo YOLO  
2. Validação do modelo YOLO  
3. Leitura OCR e pós-processamento da placa  

---

## 🧠 Motivação do Projeto

O controle de acesso ao campus universitário é uma tarefa essencial para **segurança**, **organização** e **gestão de fluxo**.

A automatização da leitura de placas reduz falhas humanas e possibilita:

- Registro automático de entrada/saída  
- Auditoria digital de veículos  
- Integração futura com cancelas e sistemas internos  

A escolha das tecnologias foi motivada por:

### 🔹 Por que YOLOv8?

- Framework **estado da arte** em detecção de objetos  
- Rápido mesmo em CPU  
- Fácil adaptação e treinamento com datasets personalizados  
- Excelente desempenho em regiões pequenas (como placas)

### 🔹 Por que EasyOCR?

- Suporte nativo a caracteres alfanuméricos  
- Alta precisão em imagens de baixo contraste  
- Facilidade de uso e carregamento rápido  
- Permite pós-processamento customizado (heurísticas + regex)

### 🔹 Por que heurísticas para placas Mercosul?

O OCR frequentemente confunde caracteres visualmente semelhantes:

**O ↔ 0**, **I ↔ 1**, **S ↔ 5**, **Q ↔ 9**, etc.

O padrão Mercosul segue o formato:

LLL N L NN (L = letra, N = número)

# 👨‍💻 Autor

Projeto desenvolvido por: Mariana Costa, Marcos Vinicius Souza
📌 Foco: Visão Computacional • Deep Learning • Automação de Segurança
