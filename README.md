# VIPER — Vehicle Identification and Plate Extraction Recognition

Sistema de detecção, rastreamento e reconhecimento automático de placas veiculares, desenvolvido com foco no monitoramento de acesso de veículos em ambientes institucionais, especialmente campus universitários.

O VIPER utiliza técnicas de visão computacional, Deep Learning e Processamento Digital de Imagens (PDI) para identificar veículos, localizar suas placas e reconhecer os caracteres presentes nelas, com suporte aos padrões brasileiros **Mercosul (`ABC1D23`) e Antigo Brasileiro (`ABC1234`)**. O sistema também integra mecanismos de telemetria de hardware e avaliação de desempenho computacional.

---

## Sobre o projeto

O VIPER (Vehicle Identification and Plate Extraction Recognition) é um projeto de pesquisa e desenvolvimento voltado à construção de uma solução de reconhecimento automático de placas veiculares (ALPR — *Automatic License Plate Recognition*).

A proposta consiste em desenvolver um sistema capaz de processar imagens e vídeos, detectar veículos, localizar suas respectivas placas e extrair os caracteres presentes nelas.

O projeto explora a integração de modelos de Deep Learning e técnicas tradicionais de Processamento Digital de Imagens, considerando aspectos como precisão, desempenho computacional, modularidade e possibilidade de implantação em dispositivos de baixo custo.

Além do reconhecimento das placas, o VIPER incorpora mecanismos de monitoramento de recursos computacionais, permitindo investigar o consumo de hardware e o tempo necessário para a execução das diferentes etapas do processamento.

## Objetivos

- Detectar veículos em imagens e vídeos.
- Realizar o rastreamento de veículos durante o monitoramento.
- Localizar placas veiculares dos padrões Mercosul e Antigo Brasileiro.
- Aplicar técnicas de processamento digital de imagens para melhorar a qualidade das regiões de interesse.
- Corrigir distorções geométricas e de perspectiva das placas.
- Reconhecer os caracteres das placas utilizando OCR.
- Aplicar regras de pós-processamento para tratar possíveis erros de reconhecimento.
- Estruturar os registros obtidos para posterior consulta e análise.
- Monitorar o consumo de recursos computacionais durante a execução.
- Avaliar latência, FPS e métricas de desempenho dos modelos.
- Investigar a viabilidade de execução em dispositivos embarcados e de baixo custo.

## Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Linguagem principal do projeto. |
| YOLOv8 | Detecção de veículos e placas veiculares. |
| ByteTrack | Rastreamento de objetos em vídeos. |
| OpenCV | Processamento digital de imagens e vídeos, recorte e transformações geométricas. |
| EasyOCR | Reconhecimento óptico de caracteres. |
| NumPy | Operações e manipulação de matrizes. |
| Pandas | Organização e manipulação de dados. |
| psutil | Monitoramento de recursos de hardware, como CPU e memória RAM. |
| ReportLab | Geração automatizada de relatórios técnicos em PDF. |
| Expressões regulares (Regex) | Validação e tratamento dos padrões alfanuméricos das placas. |

## Arquitetura geral

O VIPER é organizado em etapas independentes de processamento, responsáveis por transformar imagens e vídeos em informações estruturadas sobre os veículos e suas respectivas placas.

O fluxo geral de funcionamento é composto pelas seguintes etapas:

1. **Captura:** obtenção de imagens ou vídeos provenientes de arquivos, câmeras ou fluxos de vídeo.
2. **Identificação de veículos:** detecção dos veículos presentes na cena.
3. **Rastreamento:** acompanhamento dos veículos detectados ao longo dos quadros do vídeo.
4. **Detecção de placas:** localização das regiões correspondentes às placas veiculares.
5. **Extração da região de interesse:** recorte da placa identificada para as próximas etapas.
6. **Pré-processamento:** aplicação de técnicas de processamento digital de imagens para preparar a região da placa.
7. **Reconhecimento (OCR):** extração dos caracteres presentes na placa.
8. **Pós-processamento:** aplicação de regras e heurísticas para validar e corrigir o resultado de acordo com os padrões brasileiros.
9. **Armazenamento:** organização dos resultados para registro e análise posterior.
10. **Telemetria:** monitoramento dos recursos computacionais e do desempenho das etapas de processamento.

Essa divisão permite desenvolver, testar e aprimorar cada etapa individualmente, facilitando a manutenção e a evolução do sistema.

## Módulos e microcódigos do sistema

O VIPER utiliza diferentes componentes de software para executar as etapas do reconhecimento veicular. Cada microcódigo possui uma responsabilidade específica dentro do pipeline, permitindo que as funcionalidades sejam desenvolvidas e avaliadas de maneira independente.

### 1. Identificação e detecção de veículos

**Tecnologias:** YOLOv8 e OpenCV.

Responsável por identificar os veículos presentes nas imagens e nos fluxos de vídeo.

Principais responsabilidades:

- Receber e processar quadros provenientes de vídeos ou câmeras.
- Identificar veículos por meio de modelos de detecção.
- Selecionar as classes de objetos relevantes para o projeto.
- Obter as coordenadas das regiões delimitadoras (*bounding boxes*).
- Aplicar critérios para selecionar veículos dentro da área monitorada.

### 2. Rastreamento de veículos

**Arquivo de referência:** `track.py`

**Tecnologia:** ByteTrack.

Responsável por acompanhar os veículos identificados ao longo dos quadros de um vídeo, permitindo manter uma identificação associada ao objeto durante o monitoramento.

Principais responsabilidades:

- Realizar a detecção de objetos em sequência de quadros.
- Aplicar o algoritmo ByteTrack.
- Acompanhar as trajetórias dos veículos.
- Trabalhar com uma Região de Interesse (ROI).
- Selecionar veículos que atendam aos critérios definidos.
- Extrair imagens dos veículos para processamento posterior.

### 3. Detecção e identificação de placas

**Tecnologia:** YOLOv8 com modelo personalizado.

Responsável por localizar as placas veiculares nas imagens dos veículos selecionados.

Principais responsabilidades:

- Carregar os pesos do modelo treinado para detecção de placas.
- Identificar a localização das placas na imagem.
- Obter as coordenadas das regiões detectadas.
- Extrair a região correspondente à placa.
- Encaminhar a imagem extraída para as etapas de processamento.

### 4. Extração e recorte de imagens

**Tecnologia:** OpenCV.

Responsável por isolar a região da placa identificada pelo detector, produzindo uma imagem específica para o reconhecimento dos caracteres.

Principais responsabilidades:

- Interpretar as coordenadas da detecção.
- Realizar o recorte da região delimitada.
- Verificar as dimensões da imagem extraída.
- Preparar a região de interesse para o pré-processamento.

### 5. Pré-processamento de imagens

**Tecnologias:** OpenCV e NumPy.

Responsável por melhorar as características visuais da imagem da placa antes da aplicação do OCR.

O pré-processamento é uma etapa importante porque fatores como iluminação, contraste, resolução, ruído e inclinação podem interferir na leitura dos caracteres.

Entre as técnicas utilizadas no projeto estão:

- **Redimensionamento da imagem:** ajuste das dimensões da região de interesse.
- **Escala de cinza:** conversão da imagem para uma representação monocromática.
- **CLAHE:** equalização adaptativa do histograma para melhoria do contraste local.
- **Binarização de Otsu:** separação entre regiões de primeiro plano e fundo por meio de limiarização automática.
- **Sharpening:** aplicação de filtros para realce de nitidez.
- **Warp Perspective:** correção geométrica da placa por transformação de perspectiva.
- **Detecção de quadrilátero:** identificação dos vértices da placa por meio do algoritmo `encontrar_quadrilatero_placa`.
- **Retificação da placa:** utilização de `warp_placa`, com as funções `cv2.getPerspectiveTransform` e `cv2.warpPerspective`, para transformar uma placa inclinada em uma representação frontal.
- **Letterbox:** redimensionamento com preservação da proporção original e preenchimento das dimensões restantes, evitando distorções nos caracteres.

As técnicas podem ser avaliadas individualmente ou combinadas, permitindo investigar quais configurações oferecem melhores condições para o reconhecimento.

### 6. Reconhecimento óptico de caracteres (OCR)

**Tecnologia:** EasyOCR.

Responsável por interpretar os caracteres presentes na imagem da placa e transformar as informações visuais em texto.

Principais responsabilidades:

- Receber a imagem pré-processada.
- Executar o reconhecimento óptico de caracteres.
- Restringir o conjunto de caracteres considerados, quando aplicável.
- Obter o texto reconhecido.
- Recuperar os valores de confiança retornados pelo mecanismo de OCR.

### 7. Pós-processamento e validação de placas

**Tecnologias:** Python e expressões regulares (Regex).

Responsável por tratar os resultados obtidos pelo OCR, considerando as características alfanuméricas dos padrões brasileiros de placas veiculares.

O sistema contempla dois formatos:

**Padrão Mercosul**

Formato: `LLLNLNN`

Exemplo: `ABC1D23`

**Padrão Antigo Brasileiro**

Formato: `LLLNNNN`

Exemplo: `ABC1234`

Em que:

- `L` representa uma letra.
- `N` representa um número.

Principais responsabilidades:

- Normalizar o texto retornado pelo OCR.
- Remover caracteres indesejados.
- Identificar o padrão alfanumérico correspondente.
- Validar a sequência por meio de expressões regulares específicas para cada padrão.
- Aplicar regras posicionais para corrigir confusões entre letras e números.
- Identificar resultados que não atendam aos formatos esperados.

Essa etapa busca reduzir erros decorrentes de semelhanças visuais entre caracteres, como `0` e `O`, ou `1` e `I`, respeitando as posições previstas em cada padrão de placa.

### 8. Armazenamento e gerenciamento dos resultados

**Tecnologias:** Pandas e arquivos Excel.

Responsável por organizar os resultados obtidos durante o processamento e manter registros das identificações realizadas.

Principais responsabilidades:

- Registrar as placas reconhecidas.
- Armazenar informações relacionadas ao processamento.
- Organizar os resultados em planilhas.
- Manter o histórico das imagens processadas.
- Facilitar a consulta e a análise posterior dos registros.

### 9. Treinamento e avaliação dos modelos

**Arquivos de referência:** `trainYOLO.py` e `validadeYOLO.py`

**Tecnologia:** Ultralytics YOLOv8.

Responsável pelo treinamento e pela avaliação dos modelos de detecção utilizados pelo VIPER.

Principais responsabilidades:

- Preparar os dados de treinamento.
- Configurar os parâmetros do modelo.
- Executar o treinamento.
- Gerar os pesos do modelo.
- Avaliar o desempenho por meio de métricas de detecção.
- Calcular métricas como precisão, revocação, F1-score e mAP.
- Realizar avaliações de latência e desempenho computacional.
- Disponibilizar os modelos treinados para utilização no pipeline.

O treinamento é uma atividade independente da execução do reconhecimento veicular, embora seus resultados sejam utilizados durante a inferência.

### 10. Telemetria e monitoramento de hardware

**Arquivo de referência:** `telemetria.py`

**Tecnologias:** psutil e ferramentas de monitoramento e medição de desempenho utilizadas pelo projeto.

Responsável por acompanhar o consumo de recursos computacionais durante a execução do sistema, permitindo avaliar seu comportamento em diferentes condições de processamento.

Principais indicadores monitorados:

- **CPU (%):** percentual de utilização do processador.
- **Memória RAM (MB):** consumo e pico de utilização da memória principal.
- **VRAM (MB):** alocação de memória de vídeo, quando disponível e suportada pelo ambiente.
- **Latência (ms):** tempo de execução das etapas de pré-processamento, inferência e OCR.
- **FPS real (*End-to-End*):** quantidade de quadros processados por segundo considerando o fluxo completo do sistema.

A telemetria também está integrada por meio de callbacks nos scripts de treinamento e processamento, permitindo acompanhar indicadores durante as execuções.

Esse módulo é especialmente relevante para a investigação da viabilidade do VIPER em dispositivos embarcados e plataformas de baixo custo, pois permite relacionar a qualidade do processamento com o consumo de recursos computacionais.

### 11. Geração automatizada de relatórios

**Arquivos de referência:** `gerar_pdf_relatorio.py` e `relatorio_viper_projeto.pdf`

**Tecnologia:** ReportLab.

Responsável por gerar relatórios em PDF a partir das informações e resultados organizados pelo projeto.

Principais responsabilidades:

- Consolidar informações técnicas do projeto.
- Organizar resultados e indicadores de desempenho.
- Gerar documentos em formato PDF.
- Disponibilizar um relatório estruturado para consulta e documentação.

O arquivo `relatorio_viper_projeto.pdf` corresponde ao relatório gerado pelo processo automatizado.

## Estrutura do repositório

A estrutura do VIPER é organizada para separar os módulos de processamento, os scripts de execução, os recursos de treinamento e os documentos de pesquisa.

| Arquivo ou diretório | Descrição |
|---|---|
| `src/` | Núcleo de código modular do projeto, contendo os componentes e microcódigos. |
| `main.py` | Ponto de entrada do processamento principal. |
| `track.py` | Script relacionado ao rastreamento de veículos em vídeos. |
| `trainYOLO.py` | Script relacionado ao treinamento do modelo YOLO. |
| `validadeYOLO.py` | Script de avaliação e benchmark dos modelos, incluindo métricas e latência. |
| `telemetria.py` | Módulo de monitoramento de recursos computacionais. |
| `gerar_pdf_relatorio.py` | Script responsável pela geração automatizada de relatórios em PDF. |
| `relatorios/` | Diretório destinado aos relatórios e documentos gerados pelo projeto. |
| `relatorio_viper_projeto.pdf` | Relatório técnico gerado automaticamente. |
| `images/` | Imagens utilizadas na documentação e apresentação do projeto. |
| `runs/detect/` | Diretório associado aos resultados de execução e treinamento do YOLO. |
| `requirements.txt` | Arquivo de dependências do projeto. |

Os microcódigos são organizados conforme suas responsabilidades dentro do diretório `src/`. Para identificar a implementação de cada componente, consulte os respectivos arquivos Python e suas funções.

## Requisitos

Antes de executar o projeto, é necessário possuir:

- Python instalado.
- Git instalado.
- Gerenciador de pacotes pip.
- Ambiente virtual Python recomendado.
- Dependências do projeto instaladas.
- Pesos dos modelos YOLO necessários para as etapas de detecção.

A disponibilidade de GPU depende da configuração do ambiente e dos componentes utilizados. A execução em CPU também deve ser considerada na configuração do OCR e na avaliação do desempenho.

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/marian4cost/VIPER.git
```

### 2. Acessar o diretório

```bash
cd VIPER
```

### 3. Criar um ambiente virtual

```bash
python -m venv .venv
```

### 4. Ativar o ambiente virtual

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux:**

```bash
source .venv/bin/activate
```

### 5. Instalar as dependências

```bash
pip install -r requirements.txt
```

## Modelos de Deep Learning

O VIPER utiliza modelos YOLO para as etapas de detecção de veículos e placas.

Os pesos treinados devem estar disponíveis nos caminhos esperados pelos scripts de execução.

No processamento principal, o modelo de detecção de placas utiliza o seguinte caminho:

```text
runs/detect/placa_detector/weights/best.pt
```

Caso os pesos não estejam presentes, será necessário disponibilizá-los antes da execução.

**Importante:** os pesos dos modelos devem possuir origem documentada. Quando forem utilizados conjuntos de dados ou modelos de terceiros, suas respectivas licenças e condições de uso devem ser respeitadas.

## Execução

O repositório possui scripts relacionados ao processamento principal, rastreamento, treinamento, avaliação e geração de relatórios.

**Processamento principal**

```bash
python main.py
```

**Rastreamento de veículos**

```bash
python track.py
```

**Treinamento do modelo**

```bash
python trainYOLO.py
```

**Avaliação e benchmark**

```bash
python validadeYOLO.py
```

**Geração do relatório em PDF**

```bash
python gerar_pdf_relatorio.py
```

Os scripts podem depender de modelos previamente treinados, arquivos de entrada e configurações específicas. Consulte o código e a documentação correspondente antes da execução. Os comandos acima representam os pontos de entrada dos scripts, não uma garantia de execução sem configuração adicional.

## Pesquisa e desenvolvimento

Além da implementação do sistema, o VIPER envolve atividades de investigação e experimentação relacionadas a:

- Detecção de placas veiculares.
- Comparação de técnicas de reconhecimento óptico de caracteres.
- Pré-processamento e melhoria de imagens.
- Correção geométrica e retificação de perspectiva.
- Validação dos padrões brasileiros de placas.
- Avaliação de métricas de detecção.
- Análise de latência e taxa real de processamento.
- Monitoramento do consumo de CPU, RAM e VRAM.
- Execução em plataformas de hardware com recursos computacionais limitados.
- Integração entre componentes de visão computacional e sistemas de armazenamento.

Os resultados experimentais devem ser apresentados com identificação do conjunto de dados, configuração utilizada, ambiente de execução e métricas correspondentes a cada experimento.

## Autoria

Projeto VIPER — Vehicle Identification and Plate Extraction Recognition.

Desenvolvido no contexto de pesquisa e desenvolvimento em visão computacional, Deep Learning e reconhecimento automático de placas veiculares.
