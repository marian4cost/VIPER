# VIPER — Vehicle Identification and Plate Extraction Recognition

Sistema de detecção, rastreamento e reconhecimento automático de placas veiculares, desenvolvido com foco no monitoramento de acesso de veículos em ambientes institucionais, especialmente campus universitários.

O **VIPER** utiliza técnicas de visão computacional e Deep Learning para identificar veículos, localizar suas placas e realizar o reconhecimento dos caracteres, com suporte ao padrão brasileiro de placas Mercosul.

## Sobre o projeto

O VIPER (*Vehicle Identification and Plate Extraction Recognition*) é um projeto de pesquisa e desenvolvimento voltado à construção de uma solução de reconhecimento automático de placas veiculares (ALPR — *Automatic License Plate Recognition*).

A proposta consiste em desenvolver um sistema capaz de processar imagens e vídeos, detectar veículos, localizar suas respectivas placas e extrair os caracteres presentes nelas.

O projeto busca explorar a integração de modelos de Deep Learning e técnicas tradicionais de Processamento Digital de Imagens (PDI), considerando aspectos como precisão, desempenho computacional, modularidade e possibilidade de implantação em dispositivos de baixo custo.

### Objetivos

- Detectar veículos em imagens e vídeos.
- Realizar o rastreamento de veículos durante o monitoramento.
- Localizar placas veiculares, incluindo o padrão Mercosul.
- Aplicar técnicas de processamento digital de imagens para melhorar a qualidade das regiões de interesse.
- Reconhecer os caracteres das placas utilizando OCR.
- Aplicar regras de pós-processamento para tratar possíveis erros de reconhecimento.
- Estruturar os registros obtidos para posterior consulta e análise.
- Investigar a viabilidade de execução em dispositivos embarcados.

## Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Linguagem principal do projeto. |
| YOLOv8 | Detecção de veículos e placas veiculares. |
| ByteTrack | Rastreamento de objetos em vídeos. |
| OpenCV | Processamento digital de imagens e vídeos. |
| EasyOCR | Reconhecimento óptico de caracteres. |
| NumPy | Operações e manipulação de matrizes. |
| Pandas | Organização e manipulação de dados. |

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
8. **Pós-processamento:** aplicação de regras e heurísticas para adequar o resultado ao padrão de placas brasileiras.
9. **Armazenamento:** organização dos resultados para registro e análise posterior.

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

O pré-processamento é uma etapa importante porque fatores como iluminação, contraste, resolução e ruído podem interferir na leitura dos caracteres.

Entre as técnicas utilizadas no projeto estão:

- Redimensionamento da imagem.
- Conversão para escala de cinza.
- Equalização adaptativa de histograma por meio do CLAHE.
- Binarização pelo método de Otsu.
- Aplicação de filtros para realce de nitidez (*Sharpening*).
- Operações de melhoria da representação visual dos caracteres.

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

**Tecnologia:** Python e expressões regulares.

Responsável por tratar os resultados obtidos pelo OCR, considerando as características alfanuméricas das placas brasileiras.

No caso do padrão Mercosul, a sequência de caracteres segue o formato:

`LLLNLNN`

Em que:
- `L` representa uma letra.
- `N` representa um número.

Principais responsabilidades:

- Normalizar o texto retornado pelo OCR.
- Remover caracteres indesejados.
- Verificar a estrutura alfanumérica da placa.
- Aplicar regras posicionais para corrigir confusões entre letras e números.
- Identificar resultados que não atendam ao padrão esperado.

Essa etapa busca reduzir erros decorrentes de semelhanças visuais entre caracteres, como `0` e `O`, ou `1` e `I`.

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

**Arquivo de referência:** `trainYOLO.py`

**Tecnologia:** Ultralytics YOLOv8.

Responsável pelo treinamento dos modelos de detecção utilizados pelo VIPER.

Principais responsabilidades:

- Preparar os dados de treinamento.
- Configurar os parâmetros do modelo.
- Executar o treinamento.
- Gerar os pesos do modelo.
- Avaliar o desempenho por meio de métricas de detecção.
- Disponibilizar os modelos treinados para utilização no pipeline.

O treinamento é uma atividade independente da execução do reconhecimento veicular, embora seus resultados sejam utilizados durante a inferência.

## Estrutura do repositório

A estrutura completa pode ser consultada diretamente na árvore de diretórios do GitHub.

Entre os principais arquivos e diretórios estão:

| Arquivo ou diretório | Descrição |
|---|---|
| `main.py` | Ponto de entrada do processamento principal. |
| `track.py` | Script relacionado ao rastreamento de veículos em vídeos. |
| `trainYOLO.py` | Script relacionado ao treinamento do modelo YOLO. |
| Microcódigos de identificação | Componentes responsáveis pela detecção e seleção de veículos e placas. |
| Microcódigos de pré-processamento | Componentes responsáveis pelas técnicas de melhoria e preparação das imagens. |
| Microcódigos de OCR e leitura | Componentes responsáveis pelo reconhecimento dos caracteres. |
| Microcódigos de pós-processamento | Componentes responsáveis pelas regras de validação e correção das placas. |
| Microcódigos de armazenamento | Componentes responsáveis pela organização e registro dos resultados. |
| `docs/` | Documentação do projeto. |
| `images/` | Imagens utilizadas na documentação e apresentação do projeto. |
| `runs/detect/` | Diretório associado aos resultados de execução e treinamento do YOLO. |
| `requirements.txt` | Arquivo de dependências do projeto, após a padronização do nome. |

> Os microcódigos são organizados conforme suas responsabilidades no sistema. Para identificar a implementação de cada componente, consulte os respectivos arquivos Python e suas funções.

## Requisitos

Antes de executar o projeto, é necessário possuir:

- Python instalado.
- Git instalado.
- Gerenciador de pacotes `pip`.
- Ambiente virtual Python recomendado.
- Dependências do projeto instaladas.
- Pesos dos modelos YOLO necessários para as etapas de detecção.

A disponibilidade de GPU depende da configuração do ambiente e dos componentes utilizados. A execução em CPU também deve ser considerada na configuração do OCR.

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

```powershell
.venv\Scripts\activate
```

**Linux:**

```bash
source .venv/bin/activate
```

### 5. Instalar as dependências

Após padronizar o arquivo de dependências para `requirements.txt`:

```bash
pip install -r requirements.txt
```

> Caso o arquivo ainda esteja nomeado como `requeriments.txt`, será necessário utilizar o nome existente ou renomeá-lo antes de executar o comando acima.

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

O repositório possui scripts relacionados ao processamento principal, rastreamento e treinamento.

### Processamento principal

```bash
python main.py
```

### Rastreamento de veículos

```bash
python track.py
```

### Treinamento do modelo

```bash
python trainYOLO.py
```

> Os scripts podem depender de modelos previamente treinados, arquivos de entrada e configurações específicas. Consulte o código e a documentação correspondente antes da execução. Os comandos acima representam os pontos de entrada dos scripts, não uma garantia de execução sem configuração adicional.

## Pesquisa e desenvolvimento

Além da implementação do sistema, o VIPER envolve atividades de investigação e experimentação relacionadas a:

- Detecção de placas veiculares.
- Comparação de técnicas de reconhecimento óptico de caracteres.
- Pré-processamento e melhoria de imagens.
- Avaliação de métricas de detecção.
- Análise de desempenho e tempo de processamento.
- Execução em plataformas de hardware com recursos computacionais limitados.
- Integração entre componentes de visão computacional e sistemas de armazenamento.

Os resultados experimentais devem ser apresentados com identificação do conjunto de dados, configuração utilizada, ambiente de execução e métricas correspondentes a cada experimento.
