# Aula Completa: `train.py` — Treinamento do Modelo de Crop Inteligente

> **Objetivo do script:** Treinar uma rede neural que, dada uma imagem original, aprende a prever **onde recortar** (as coordenadas do crop ideal), reproduzindo o padrão de recortes que foram feitos manualmente por um humano.

---

## Sumário

1. [Visão Geral do Problema](#1-visão-geral-do-problema)
2. [Imports — O que cada biblioteca faz](#2-imports--o-que-cada-biblioteca-faz)
3. [Sistema de Logging](#3-sistema-de-logging)
4. [Dispositivo de Execução (CPU vs GPU)](#4-dispositivo-de-execução-cpu-vs-gpu)
5. [CropDataset — O Dataset Customizado](#5-cropdataset--o-dataset-customizado)
6. [Cálculo de Bounding Boxes (Template Matching)](#6-cálculo-de-bounding-boxes-template-matching)
7. [Cache de Bounding Boxes](#7-cache-de-bounding-boxes)
8. [MarginAwareCropModel — A Arquitetura do Modelo](#8-marginawarecropmodel--a-arquitetura-do-modelo)
9. [Funções de Loss (Perda)](#9-funções-de-loss-perda)
10. [Função `train()` — O Loop de Treinamento](#10-função-train--o-loop-de-treinamento)
11. [Ponto de Entrada (`__main__`)](#11-ponto-de-entrada-__main__)
12. [Glossário de Conceitos](#12-glossário-de-conceitos)

---

## 1. Visão Geral do Problema

### O que este projeto resolve?

Imagine que você tem **milhares de imagens** e precisa recortá-las sempre no mesmo estilo — por exemplo, centralizar um produto, remover bordas desnecessárias, ou enquadrar um rosto. Fazer isso manualmente é inviável.

A ideia é:

1. Você recorta **algumas dezenas/centenas de imagens** manualmente (criando pares: original + recortada).
2. O modelo **aprende o padrão** desses recortes.
3. Depois, ele recorta **novas imagens automaticamente**, imitando seu estilo.

### Como o modelo "aprende" onde recortar?

O modelo recebe uma imagem original e precisa prever **4 números**:

```
[x1, y1, x2, y2]
```

Esses 4 números definem um retângulo (bounding box) dentro da imagem:

```
(0,0) ─────────────────────── (1,0)
  │                              │
  │    (x1,y1)──────────┐       │
  │      │   CONTEÚDO   │       │
  │      │   DO CROP    │       │
  │      └──────────(x2,y2)     │
  │                              │
(0,1) ─────────────────────── (1,1)
```

- **(x1, y1)** = canto superior esquerdo do recorte
- **(x2, y2)** = canto inferior direito do recorte
- Todos os valores estão **normalizados entre 0 e 1** (proporção da imagem, não pixels)

**Exemplo concreto:** Se uma imagem tem 1000x800 pixels e o modelo prevê `[0.10, 0.05, 0.90, 0.95]`, isso significa recortar:
- De `x = 100px` até `x = 900px` (largura)
- De `y = 40px` até `y = 760px` (altura)

---

## 2. Imports — O que cada biblioteca faz

```python
import logging          # Sistema de logs (substituindo print)
import os               # Operações de sistema de arquivos
import time             # Medir tempo de execução
import datetime         # Gerar timestamps para nomes de arquivos
import cv2              # OpenCV — leitura e processamento de imagens
import numpy as np      # Operações numéricas com arrays
import torch            # Framework de deep learning (núcleo)
import torch.nn as nn   # Módulos de redes neurais (camadas, ativações)
import torch.optim as optim  # Otimizadores (AdamW, SGD, etc.)
from concurrent.futures import ThreadPoolExecutor  # Paralelismo com threads
from torch.utils.data import Dataset, DataLoader   # Estruturas para alimentar dados ao modelo
from torchvision import models, transforms          # Modelos pré-treinados e transformações de imagem
from sklearn.model_selection import train_test_split # Dividir dados em treino/validação
from tqdm import tqdm   # Barras de progresso no terminal
```

### Por que cada uma?

| Biblioteca | Papel neste script |
|---|---|
| `logging` | Todo output vai para arquivo `.log` E terminal simultaneamente. Nunca usamos `print()`. |
| `cv2` (OpenCV) | Lê imagens do disco, converte cores (BGR→RGB), redimensiona, e faz template matching. |
| `numpy` | Manipula arrays de bounding boxes (os 4 números do crop) de forma eficiente. |
| `torch` | O coração: define tensores, roda operações na GPU, calcula gradientes automaticamente. |
| `torch.nn` | Define as camadas da rede neural (Linear, Dropout, BatchNorm, etc.). |
| `torch.optim` | Contém o otimizador (AdamW) que ajusta os pesos da rede durante o treino. |
| `ThreadPoolExecutor` | Calcula bounding boxes em paralelo usando múltiplas threads (I/O bound). |
| `Dataset` / `DataLoader` | Abstração do PyTorch para carregar dados em batches, com shuffle e paralelismo. |
| `torchvision.models` | Fornece a EfficientNet-B0 pré-treinada no ImageNet. |
| `torchvision.transforms` | Normaliza imagens para o formato que a EfficientNet espera. |
| `train_test_split` | Divide os dados aleatoriamente em 90% treino + 10% validação. |
| `tqdm` | Mostra barras de progresso para acompanhar loops longos. |

---

## 3. Sistema de Logging

```python
def _setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join("logs", f"{script_name}_{timestamp}.log")
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("train")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()     # → Terminal
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_filename, encoding="utf-8")  # → Arquivo
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logging.captureWarnings(True)
    return logger
```

### O que acontece aqui, passo a passo:

1. **Cria a pasta `logs/`** se ela não existir.
2. **Gera um nome de arquivo único** baseado no nome do script + data/hora (ex: `train_2026-03-21_14-30-00.log`). Isso evita sobrescrever logs anteriores.
3. **Define o formato** das mensagens: `2026-03-21 14:30:00 [INFO] Mensagem aqui`.
4. **Cria dois "destinos"** para os logs:
   - **StreamHandler (sh):** Mostra mensagens de nível `INFO` ou superior no terminal.
   - **FileHandler (fh):** Grava **tudo** (inclusive `DEBUG`) no arquivo `.log`.
5. **`if logger.handlers: return logger`** — Evita duplicar handlers se a função for chamada mais de uma vez.
6. **`logging.captureWarnings(True)`** — Captura warnings do Python (ex: DeprecationWarning) e os redireciona para o sistema de logging.

### Por que dois níveis diferentes?

- No **terminal**, você quer ver apenas o essencial (progresso, erros). Nível `INFO`.
- No **arquivo**, você quer **tudo** para poder investigar problemas depois. Nível `DEBUG`.

```python
log: logging.Logger = logging.getLogger("train")
```

Essa linha cria uma referência global ao logger. Note que `_setup_logging()` só é chamada no `if __name__ == "__main__"`, mas o `log` já fica disponível para uso em todo o módulo — ele está "vazio" (sem handlers) até `_setup_logging()` configurá-lo.

---

## 4. Dispositivo de Execução (CPU vs GPU)

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### O que é CUDA?

CUDA é a tecnologia da NVIDIA que permite usar a GPU para cálculos. GPUs são **ordens de magnitude mais rápidas** que CPUs para operações de deep learning, porque processam milhares de operações em paralelo.

- **`torch.cuda.is_available()`** verifica se existe uma GPU NVIDIA com drivers CUDA instalados.
- Se sim → usa `cuda` (GPU). Se não → usa `cpu`.

Treinar na CPU é possível mas **muito mais lento** (pode ser 10-50x mais devagar dependendo do modelo).

---

## 5. CropDataset — O Dataset Customizado

### O que é um Dataset no PyTorch?

O PyTorch exige que seus dados implementem a interface `Dataset`, que tem apenas dois métodos obrigatórios:

- `__len__()` — quantos exemplos existem
- `__getitem__(idx)` — retorna o exemplo na posição `idx`

O DataLoader usa esses métodos para montar batches automaticamente.

### Construtor (`__init__`)

```python
class CropDataset(Dataset):
    def __init__(self, original_paths, cropped_paths, bbox_data, img_size=300):
        self.img_size = img_size
        self.pairs = []
        
        for orig_path, crop_path, bbox_norm in zip(original_paths, cropped_paths, bbox_data):
            try:
                self.pairs.append({
                    'orig_path': orig_path,
                    'crop_path': crop_path,
                    'bbox': bbox_norm.astype(np.float32),
                })
            except Exception as e:
                log.error(f"Erro ao processar {orig_path}: {e}")
                continue
```

**O que acontece:**
1. Recebe três listas paralelas: caminhos das originais, caminhos dos crops, e os bounding boxes já calculados.
2. **Não carrega nenhuma imagem na memória** — apenas armazena os caminhos. Isso é o "carregamento sob demanda" (lazy loading).
3. Para cada trio, cria um dicionário com o caminho e o bbox, e guarda na lista `self.pairs`.
4. Se qualquer erro ocorrer (ex: path inválido), loga o erro e pula para o próximo.

### Por que lazy loading?

Se o dataset tiver 10.000 imagens de 5MB cada, carregar tudo na RAM consumiria **50GB**. Com lazy loading, cada imagem é lida do disco **apenas quando o DataLoader a solicita**, e descartada depois. Isso troca velocidade por memória.

### `__getitem__` — O coração do Dataset

```python
def __getitem__(self, idx):
    pair = self.pairs[idx]
    
    # Carregar imagem original somente quando necessário
    orig = cv2.imread(pair['orig_path'])
    if orig is None:
        dummy_img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        dummy_bbox = np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)
        return dummy_img, dummy_bbox
    
    # BGR → RGB
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    # Redimensionar para tamanho fixo
    orig_resized = cv2.resize(orig_rgb, (self.img_size, self.img_size))
    # Converter para tensor PyTorch
    orig_tensor = transforms.ToTensor()(orig_resized)
    # Normalizar com médias do ImageNet
    orig_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(orig_tensor)
    
    return orig_tensor, pair['bbox']
```

**Pipeline de processamento da imagem, passo a passo:**

#### Passo 1: Leitura com OpenCV
```python
orig = cv2.imread(pair['orig_path'])
```
O OpenCV lê a imagem do disco como um array NumPy com shape `(altura, largura, 3)`. Os 3 canais são **BGR** (Azul, Verde, Vermelho) — diferente do padrão RGB.

#### Passo 2: Fallback de segurança
```python
if orig is None:
    dummy_img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
    dummy_bbox = np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)
    return dummy_img, dummy_bbox
```
Se a imagem não puder ser lida (arquivo corrompido, path errado), retorna uma imagem preta com um bbox padrão (crop quase total) para não travar o treinamento.

#### Passo 3: Conversão BGR → RGB
```python
orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
```
A EfficientNet foi treinada com imagens RGB. Se enviarmos BGR, as cores estariam invertidas e o modelo não funcionaria bem.

#### Passo 4: Redimensionamento
```python
orig_resized = cv2.resize(orig_rgb, (self.img_size, self.img_size))
```
Todas as imagens precisam ter **o mesmo tamanho** para formar batches. A rede espera `360x360` neste projeto. Imagens originais de qualquer resolução são redimensionadas para esse tamanho fixo.

> **Nota:** Isso pode distorcer a proporção (aspect ratio) da imagem. Porém, como os bounding boxes também estão normalizados (0-1), a distorção não afeta a correspondência entre imagem e bbox.

#### Passo 5: Conversão para Tensor
```python
orig_tensor = transforms.ToTensor()(orig_resized)
```
`ToTensor()` faz duas coisas:
1. Converte o array NumPy `(H, W, C)` para tensor PyTorch `(C, H, W)` — mudança de eixos.
2. Converte valores de `[0, 255]` (uint8) para `[0.0, 1.0]` (float32) — normalização de escala.

#### Passo 6: Normalização ImageNet
```python
orig_tensor = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)(orig_tensor)
```

Esses números **não são arbitrários**. São a média e o desvio padrão calculados sobre **todos os ~1.2 milhões de imagens** do dataset ImageNet. A fórmula aplicada em cada canal é:

$$\text{pixel\_normalizado} = \frac{\text{pixel} - \text{média}}{\text{desvio padrão}}$$

**Por que normalizar?** Como a EfficientNet foi pré-treinada no ImageNet com esses valores, os pesos internos da rede "esperam" receber dados nessa escala. Sem essa normalização, os features extraídos pela rede seriam ruidosos e o treinamento seria muito mais difícil.

#### Retorno
```python
return orig_tensor, pair['bbox']
```
Retorna uma tupla:
- **`orig_tensor`** — Tensor `(3, 360, 360)` com a imagem normalizada
- **`pair['bbox']`** — Array `(4,)` com `[x1, y1, x2, y2]` normalizados (o "gabarito")

---

## 6. Cálculo de Bounding Boxes (Template Matching)

### O problema

Temos pares de imagens: a original e a versão recortada. Precisamos descobrir **exatamente onde** a versão recortada se encaixa dentro da original, para criar o "gabarito" (ground truth) do treinamento.

### A solução: Template Matching

```python
def _compute_bbox_for_pair(orig_path, crop_path):
    orig = cv2.imread(orig_path)
    crop = cv2.imread(crop_path)
    if orig is None or crop is None:
        return np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)

    orig_h, orig_w = orig.shape[:2]

    gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_orig, gray_crop, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    x1, y1 = max_loc
    x2, y2 = x1 + crop.shape[1], y1 + crop.shape[0]

    return np.array([
        x1 / orig_w, y1 / orig_h,
        x2 / orig_w, y2 / orig_h
    ], dtype=np.float32)
```

### Como o Template Matching funciona?

Imagine que você tem um quebra-cabeça (imagem original) e uma peça (imagem recortada). O Template Matching **desliza a peça sobre o quebra-cabeça**, posição por posição, calculando em cada ponto **o quão bem a peça se encaixa**.

```
Imagem Original (cinza):                      Crop (cinza):
┌──────────────────────┐                       ┌──────┐
│                      │                       │      │
│      ┌──────┐        │    ← O algoritmo     │ peça │
│      │ AQUI │        │      encontra essa    │      │
│      │ ESTÁ │        │      posição          └──────┘
│      └──────┘        │
│                      │
└──────────────────────┘
```

**Passo a passo:**

1. **Converte para escala de cinza** — Template matching é mais rápido e robusto em tons de cinza.
2. **`cv2.matchTemplate(gray_orig, gray_crop, cv2.TM_CCOEFF_NORMED)`** — Desliza o crop sobre a original, gerando um "mapa de correlação". Cada pixel do mapa indica o quão bem o crop se encaixa naquela posição. O método `TM_CCOEFF_NORMED` retorna valores entre -1 e 1, onde 1 = encaixe perfeito.
3. **`cv2.minMaxLoc(result)`** — Encontra o ponto de máxima correlação. `max_loc` é a coordenada `(x, y)` do canto superior esquerdo onde o encaixe é melhor.
4. **Calcula o canto inferior direito:** `x2 = x1 + largura_do_crop`, `y2 = y1 + altura_do_crop`.
5. **Normaliza** dividindo pela largura/altura da original, resultando em valores entre 0 e 1.

### Exemplo numérico

- Original: 1000x800 pixels
- Crop: 800x600 pixels
- Template matching encontra o crop na posição (100, 50)

```
x1 = 100 / 1000 = 0.10
y1 = 50  / 800  = 0.0625
x2 = (100 + 800) / 1000 = 0.90
y2 = (50 + 600)  / 800  = 0.8125

bbox = [0.10, 0.0625, 0.90, 0.8125]
```

Isso significa: "o crop começa em 10% da largura, 6.25% da altura, e vai até 90% da largura, 81.25% da altura."

---

## 7. Cache de Bounding Boxes

```python
def compute_bounding_boxes(orig_paths, crop_paths, cache_path=None, max_workers=None):
```

O cálculo de bounding boxes via template matching é **computacionalmente caro** (lê duas imagens do disco + faz correlação cruzada para cada par). Se você tem 1000 pares, isso pode levar vários minutos.

### Estratégia de cache

```python
if cache_path is not None and os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if cache.get("orig_paths") == orig_paths and cache.get("crop_paths") == crop_paths:
        return cache.get("bboxes", [])
```

1. Se o arquivo de cache existe, carrega-o.
2. **Verifica se as listas de caminhos são idênticas** — se você adicionou/removeu imagens do dataset, o cache é invalidado automaticamente e tudo é recalculado.
3. Se o cache é válido, retorna os bboxes em milissegundos ao invés de minutos.

### Paralelismo com ThreadPoolExecutor

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for bbox in tqdm(executor.map(_compute_bbox_for_pair, orig_paths, crop_paths), total=len(orig_paths)):
        bbox_list.append(bbox)
```

Como o template matching é **I/O bound** (a maior parte do tempo é gasta lendo imagens do disco), usa-se `ThreadPoolExecutor` para processar múltiplos pares simultaneamente. O `max_workers` padrão é o número de CPUs disponíveis.

> **I/O bound vs CPU bound:** 
> - **I/O bound** = o programa espera mais por disco/rede do que por cálculos → use **threads**.
> - **CPU bound** = o programa gasta mais tempo calculando → use **processos** (multiprocessing).

### Salvamento do cache

Após o cálculo, os resultados são salvos em `models/bbox_cache.pkl` usando `pickle` (serialização Python). Na próxima execução, o cache evita todo o reprocessamento.

---

## 8. MarginAwareCropModel — A Arquitetura do Modelo

### Visão geral da arquitetura

```
Imagem 360x360x3
       │
       ▼
┌─────────────────┐
│  EfficientNet-B0 │  ← Backbone (extrator de features) pré-treinado no ImageNet
│  (features)      │
└────────┬────────┘
         │  Mapa de features: (batch, 1280, 12, 12)
         ▼
┌─────────────────┐
│ AdaptiveAvgPool  │  ← Compacta dimensões espaciais para (1, 1)
│  (1, 1)          │
└────────┬────────┘
         │  Vetor: (batch, 1280)
         ▼
┌─────────────────┐
│   Flatten        │  ← Achata para vetor 1D
└────────┬────────┘
         │  Vetor: (batch, 1280)
         ▼
┌─────────────────┐
│   Regressor      │  ← Cabeça customizada (MLP)
│  1280→512→128→4  │
└────────┬────────┘
         │  Saída: (batch, 4)
         ▼
┌─────────────────┐
│    Sigmoid       │  ← Força saída entre [0, 1]
└────────┬────────┘
         │
         ▼
    [x1, y1, x2, y2]  ← Coordenadas normalizadas do crop
```

### O que é Transfer Learning?

A EfficientNet-B0 foi treinada no ImageNet (1.2 milhão de imagens, 1000 categorias). Ela aprendeu a reconhecer bordas, texturas, formas, objetos — conhecimento visual genérico.

Em vez de treinar uma rede do zero (que exigiria centenas de milhares de imagens), **reutilizamos esse conhecimento**. A EfficientNet funciona como um "extrator de features visuais" e nós adicionamos apenas uma cabeça customizada (o regressor) que converte essas features em coordenadas de crop.

### O Backbone (EfficientNet-B0)

```python
try:
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
except AttributeError:
    backbone = models.efficientnet_b0(pretrained=True)

self.features = backbone.features
```

- **`models.efficientnet_b0(weights=...)`** — Carrega a EfficientNet-B0 com pesos pré-treinados do ImageNet. O `try/except` garante compatibilidade com diferentes versões do PyTorch.
- **`self.features = backbone.features`** — Pega apenas a parte convolucional da rede (sem o classificador final). Essa parte transforma uma imagem `(3, 360, 360)` em um mapa de features `(1280, 12, 12)` — 1280 canais de informação, cada um com resolução 12x12.

### Por que EfficientNet?

A família EfficientNet usa uma técnica chamada **compound scaling** que equilibra profundidade, largura e resolução da rede de forma otimizada. O resultado é um modelo que extrai features de alta qualidade com **menos parâmetros e menos computação** do que alternativas como ResNet ou VGG.

| Modelo | Parâmetros | Top-1 Accuracy (ImageNet) |
|---|---|---|
| ResNet-50 | 25.6M | 76.1% |
| EfficientNet-B0 | 5.3M | 77.1% |

A EfficientNet-B0 é **mais precisa** com **5x menos parâmetros**.

### Adaptive Average Pooling

```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
```

O mapa de features tem shape `(batch, 1280, 12, 12)`. O `AdaptiveAvgPool2d((1, 1))` calcula a **média de cada canal** sobre todas as posições espaciais, resultando em `(batch, 1280, 1, 1)`:

```
Canal 0:                    Canal 0 (após pooling):
┌─────────────┐
│ 0.2  0.5  0.3│            
│ 0.1  0.8  0.4│  → média → [0.38]
│ 0.3  0.6  0.2│
└─────────────┘

(isso acontece para cada um dos 1280 canais)
```

Isso torna o modelo **agnóstico à posição espacial** — ele captura "o que está na imagem" sem se prender a "onde está" nessa resolução intermediária.

### O Regressor (MLP)

```python
self.regressor = nn.Sequential(
    nn.Dropout(0.3),           # ①
    nn.Linear(1280, 512),      # ②
    nn.BatchNorm1d(512),       # ③
    nn.LeakyReLU(0.1),         # ④
    nn.Dropout(0.2),           # ⑤
    nn.Linear(512, 128),       # ⑥
    nn.BatchNorm1d(128),       # ⑦
    nn.LeakyReLU(0.1),         # ⑧
    nn.Linear(128, 4)          # ⑨
)
```

Cada camada explicada:

| # | Camada | O que faz | Por que |
|---|---|---|---|
| ① | `Dropout(0.3)` | Desliga 30% dos neurônios aleatoriamente durante o treino | Previne overfitting — força a rede a não depender de neurônios específicos |
| ② | `Linear(1280, 512)` | Multiplicação matricial: transforma vetor de 1280 dims para 512 | Compressão de informação |
| ③ | `BatchNorm1d(512)` | Normaliza os valores dentro de cada batch | Estabiliza o treinamento e permite learning rates maiores |
| ④ | `LeakyReLU(0.1)` | Função de ativação: $f(x) = \max(0.1x, x)$ | Introduz não-linearidade. "Leaky" permite gradiente pequeno para valores negativos, evitando "neurônios mortos" |
| ⑤ | `Dropout(0.2)` | Desliga 20% dos neurônios | Mais regularização, mas menos agressiva que a camada anterior |
| ⑥ | `Linear(512, 128)` | Comprime de 512 para 128 dimensões | Continua comprimindo a informação |
| ⑦ | `BatchNorm1d(128)` | Normaliza novamente | Estabilidade |
| ⑧ | `LeakyReLU(0.1)` | Ativação não-linear | Mesma função |
| ⑨ | `Linear(128, 4)` | Camada final: 128 → 4 valores | Produz os 4 números do bbox |

### Inicialização inteligente dos biases

```python
nn.init.constant_(self.regressor[-1].bias, 0.0)
self.regressor[-1].bias.data[0] = 0.07  # x1 ~ 7%
self.regressor[-1].bias.data[1] = 0.07  # y1 ~ 7%
self.regressor[-1].bias.data[2] = 0.93  # x2 ~ 93%
self.regressor[-1].bias.data[3] = 0.93  # y2 ~ 93%
```

**Por que isso é importante?**

Sem essa inicialização, a rede começaria prevendo valores aleatórios — talvez `[0.5, 0.5, 0.5, 0.5]` (um retângulo de área zero). Isso geraria losses enormes no início e o treinamento demoraria muito para convergir.

Com a inicialização em `[0.07, 0.07, 0.93, 0.93]`, a rede **já começa prevendo um crop razoável** (margem de ~7% em cada lado). A partir daí, ela só precisa **refinar** essas previsões, o que acelera significativamente a convergência.

### Forward pass

```python
def forward(self, x):
    x = self.features(x)           # EfficientNet extrai features
    x = self.avgpool(x)            # Pooling: (B, 1280, H, W) → (B, 1280, 1, 1)
    x = torch.flatten(x, 1)       # Achata: (B, 1280, 1, 1) → (B, 1280)
    return torch.sigmoid(self.regressor(x))  # Regressor + Sigmoid
```

O `torch.sigmoid()` no final garante que a saída está sempre entre 0 e 1:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Sem o sigmoid, a rede poderia prever valores como -0.3 ou 1.5, que não fazem sentido como coordenadas normalizadas.

---

## 9. Funções de Loss (Perda)

As funções de loss (ou custo) medem **o quão errada** está a previsão do modelo. O otimizador usa esse valor para ajustar os pesos. Quanto menor a loss, melhor o modelo.

### IoU Loss

**IoU (Intersection over Union)** é a métrica padrão para avaliar a qualidade de bounding boxes.

```python
def iou_loss(pred, target):
    pred = pred.clamp(1e-6, 1 - 1e-6)
    target = target.clamp(1e-6, 1 - 1e-6)
    
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    
    union = pred_area + target_area - inter + 1e-6
    iou = inter / union
    return 1 - iou.mean()
```

#### Conceito visual do IoU

```
     Predição          Gabarito           Interseção
   ┌──────────┐      ┌──────────┐      
   │          │      │          │      ┌──────┐
   │   ┌──────┼──┐   │  ┌───┐  │      │      │  ← Área em comum
   │   │  ////│  │   │  │   │  │      │      │
   │   │  ////│  │   │  └───┘  │      └──────┘
   └───┼──────┘  │   │          │
       └─────────┘   └──────────┘

IoU = Área da Interseção / Área da União
```

$$IoU = \frac{\text{Interseção}}{\text{Predição} + \text{Gabarito} - \text{Interseção}}$$

- **IoU = 1.0** → predição perfeita (100% de sobreposição)
- **IoU = 0.0** → nenhuma sobreposição

A loss é `1 - IoU`, então:
- Loss = 0 → perfeito
- Loss = 1 → péssimo

#### Detalhamento do cálculo

```python
# 1. Clamp para evitar divisão por zero ou logaritmos de zero
pred = pred.clamp(1e-6, 1 - 1e-6)

# 2. Calcular a interseção dos dois retângulos
x1 = torch.max(pred[:, 0], target[:, 0])  # Borda esquerda mais à direita
y1 = torch.max(pred[:, 1], target[:, 1])  # Borda superior mais abaixo
x2 = torch.min(pred[:, 2], target[:, 2])  # Borda direita mais à esquerda
y2 = torch.min(pred[:, 3], target[:, 3])  # Borda inferior mais acima

# 3. Área da interseção (clamped para não ser negativa)
inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

# 4. Áreas individuais
pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

# 5. União = A + B - Interseção (+ epsilon para estabilidade numérica)
union = pred_area + target_area - inter + 1e-6

# 6. IoU e loss
iou = inter / union
return 1 - iou.mean()  # Média do batch
```

O `[:, 0]` significa "todos os exemplos do batch, coluna 0". Lembre-se que `pred` tem shape `(batch_size, 4)`.

### Margin-Aware Loss

Esta é a loss **mais importante** do projeto. Enquanto o IoU mede a sobreposição geral, o Margin-Aware Loss foca especificamente nas **margens** (o espaço entre a borda da imagem e o crop).

```python
def margin_aware_loss(pred, target):
```

#### Por que as margens importam?

Considere dois crops com o mesmo IoU:

```
Crop A (margens simétricas):      Crop B (margens assimétricas):
┌────────────────────┐            ┌────────────────────┐
│    ┌──────────┐    │            │┌──────────┐        │
│    │          │    │            ││          │        │
│    │  CONTEÚDO│    │            ││ CONTEÚDO │        │
│    │          │    │            ││          │        │
│    └──────────┘    │            │└──────────┘        │
└────────────────────┘            └────────────────────┘
      CORRETO ✓                      ERRADO ✗
```

Ambos podem ter IoU alto, mas o Crop B tem as margens erradas — o conteúdo está deslocado para a esquerda. O Margin-Aware Loss detecta e penaliza isso.

#### Como funciona

```python
# 1. Calcular as 4 margens (distância de cada borda do bbox até a borda da imagem)
pred_margins = torch.stack([
    pred[:, 0],           # margem esquerda = x1
    pred[:, 1],           # margem superior = y1
    1.0 - pred[:, 2],     # margem direita = 1 - x2
    1.0 - pred[:, 3]      # margem inferior = 1 - y2
], dim=1)
```

Visualização das margens:

```
 0                    1
 ├────── x1 ─────┬───── (1-x2) ────┤
 │   margem      │     margem      │
 │   esquerda    │     direita     │
 │               │                 │
```

```python
# 2. Mesma coisa para o gabarito
target_margins = torch.stack([
    target[:, 0], target[:, 1],
    1.0 - target[:, 2], 1.0 - target[:, 3]
], dim=1)

# 3. Calcular dimensões do conteúdo
content_width = pred[:, 2] - pred[:, 0]   # largura do crop
content_height = pred[:, 3] - pred[:, 1]  # altura do crop

# 4. Erro relativo: diferença das margens normalizada pelo conteúdo
margin_error = torch.abs(pred_margins - target_margins)
margin_error[:, 0] /= (content_width + 1e-6)   # esquerda / largura
margin_error[:, 2] /= (content_width + 1e-6)    # direita / largura
margin_error[:, 1] /= (content_height + 1e-6)   # topo / altura
margin_error[:, 3] /= (content_height + 1e-6)   # baixo / altura
```

#### Por que normalizar pelo conteúdo?

Imagine duas situações:
- **Crop grande** (80% da imagem): Uma margem errada de 2% é pouco perceptível.
- **Crop pequeno** (20% da imagem): Uma margem errada de 2% é **muito** perceptível (pode ser 10% do tamanho do conteúdo!).

A divisão pelo tamanho do conteúdo garante que **erros em crops pequenos sejam penalizados proporcionalmente mais**.

### Combined Loss

```python
def combined_loss(pred, target, alpha=0.5):
    return alpha * iou_loss(pred, target) + (1 - alpha) * margin_aware_loss(pred, target)
```

No treinamento, é usado `alpha=0.45`:

$$\mathcal{L}_{\text{total}} = 0.45 \times \mathcal{L}_{\text{IoU}} + 0.55 \times \mathcal{L}_{\text{margem}}$$

O peso ligeiramente maior na loss de margem (55% vs 45%) reflete a prioridade do projeto: **a precisão das margens é mais importante que a sobreposição bruta**.

---

## 10. Função `train()` — O Loop de Treinamento

### Configurações

```python
IMG_SIZE = 360      # Resolução de entrada da rede
BATCH_SIZE = 16     # Quantas imagens processar de cada vez
NUM_WORKERS = os.cpu_count() or 4  # Threads para carregar dados em paralelo
EPOCHS = 100        # Máximo de passagens pelo dataset inteiro
PATIENCE = 15       # Quantas épocas sem melhora antes de parar
```

#### O que é um Batch?

Em vez de processar uma imagem por vez (lento) ou todas de uma vez (memória insuficiente), o treinamento processa **batches** (lotes) de imagens.

```
Dataset: [img1, img2, img3, ..., img1000]

Batch 1: [img1, img2, ..., img16]    → forward → loss → backward → atualiza pesos
Batch 2: [img17, img18, ..., img32]  → forward → loss → backward → atualiza pesos
...
Batch 63: [img993, ..., img1000]     → forward → loss → backward → atualiza pesos

↑ Isso é 1 ÉPOCA (uma passagem completa pelo dataset)
```

#### O que é uma Época?

Uma **época** é uma passagem completa por **todos** os dados de treino. O modelo vê cada imagem exatamente uma vez por época. Normalmente são necessárias dezenas ou centenas de épocas para o modelo convergir.

### Carregamento e Pareamento dos Dados

```python
orig_files = sorted([os.path.join(orig_dir, f) for f in os.listdir(orig_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
crop_files = sorted([os.path.join(crop_dir, f) for f in os.listdir(crop_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and '_editado' in f.lower()])
```

1. Lista todas as imagens originais em `dataset/origin/`.
2. Lista todas as imagens recortadas em `dataset/cropped/` (que contenham `_editado` no nome).

```python
orig_dict = {os.path.splitext(os.path.basename(f))[0].replace('_editado', ''): f for f in orig_files}
crop_dict = {os.path.splitext(os.path.basename(f))[0].replace('_editado', ''): f for f in crop_files}
common_names = set(orig_dict.keys()) & set(crop_dict.keys())
```

3. Cria dicionários `{nome_base: caminho_completo}` para originais e crops.
4. Encontra os **nomes em comum** (interseção de conjuntos) — garante que só usa pares completos.

**Exemplo:**
- `dataset/origin/foto001.jpg` → chave `"foto001"`
- `dataset/cropped/foto001_editado.jpg` → chave `"foto001"`
- Ambos existem → par válido.

### Divisão Treino/Validação

```python
train_orig, val_orig, train_crop, val_crop, train_bbox, val_bbox = train_test_split(
    orig_paths, crop_paths, bbox_data, test_size=0.1, random_state=42
)
```

- **90% para treino** — o modelo aprende com esses dados.
- **10% para validação** — o modelo é avaliado com dados que **nunca viu** durante o treino.
- **`random_state=42`** — semente fixa para reprodutibilidade. O mesmo split sempre produz os mesmos conjuntos.

### Por que separar treino e validação?

Se avaliarmos o modelo nos mesmos dados em que ele treinou, ele pode ter **decorado** os dados (overfitting) e parecer perfeito, mas falhar em imagens novas. A validação simula o cenário real.

### DataLoader

```python
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True
)
```

| Parâmetro | Valor | Significado |
|---|---|---|
| `batch_size` | 16 | Quantas imagens por batch |
| `shuffle` | True | Embaralha a ordem dos dados a cada época (evita viés de ordenação) |
| `drop_last` | True | Descarta o último batch se for incompleto (evita problemas com BatchNorm) |
| `num_workers` | CPU count | Quantas threads pré-carregam dados em paralelo |
| `pin_memory` | True | Reserva memória "pinada" para transferência CPU→GPU mais rápida |
| `prefetch_factor` | 2 | Pré-carrega 2 batches por worker enquanto a GPU processa o batch atual |
| `persistent_workers` | True | Mantém os workers vivos entre as épocas (evita overhead de criação/destruição de threads) |

### Otimizador AdamW

```python
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
```

- **AdamW** é uma variante do Adam que aplica weight decay (regularização L2) de forma **desacoplada** dos gradientes, resultando em melhor generalização.
- **`lr=3e-4` (0.0003)** — Learning rate: o quão "agressivamente" os pesos são atualizados a cada passo. Muito alto → instabilidade. Muito baixo → demora para convergir.
- **`weight_decay=1e-4`** — Penalidade proporcional ao tamanho dos pesos. Evita que os pesos cresçam demais (regularização).

### Scheduler (Agendador de Learning Rate)

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=6
)
```

Quando a loss de validação **para de melhorar por 6 épocas**, o scheduler **divide a learning rate pela metade** (`factor=0.5`).

```
Época 1-20:  lr = 0.0003    (learning rate inicial)
Época 21-26: lr = 0.0003    (loss estagnou, mas patience = 6)
Época 27:    lr = 0.00015   (redução! optimizer atualiza com passos menores)
...
```

A lógica é: no início, passos grandes encontram a região certa; depois, passos menores refinam o resultado.

### Mixed Precision (Precisão Mista)

```python
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
```

#### O que é Mixed Precision?

Por padrão, todos os cálculos usam **float32** (32 bits por número). Mixed Precision alterna entre:
- **float16** (16 bits) — para o forward pass e cálculos intermediários (mais rápido, menos memória)
- **float32** (32 bits) — para acumulação de gradientes e atualização de pesos (precisão necessária)

**Benefícios:**
- Quase **2x mais rápido** em GPUs com Tensor Cores (RTX 3060 Ti tem!)
- Usa **~metade da memória** de VRAM
- Sem perda perceptível de qualidade

O `GradScaler` é necessário porque gradientes em float16 podem ser pequenos demais (underflow). Ele **escala** os gradientes antes do backward e **des-escala** antes da atualização.

### O Loop de Treinamento (por época)

```python
for epoch in range(EPOCHS):
    # ===== TREINO =====
    model.train()
    train_loss = 0.0
    
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                preds = model(inputs)
                loss = combined_loss(preds, targets, alpha=0.45)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(inputs)
            loss = combined_loss(preds, targets, alpha=0.45)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        train_loss += loss.item()
```

#### Dissecando cada linha:

**`model.train()`** — Ativa o modo de treinamento. Isso habilita Dropout (desligamento aleatório de neurônios) e BatchNorm (usa estatísticas do batch). No modo `eval()`, Dropout é desativado e BatchNorm usa estatísticas acumuladas.

**`inputs.to(DEVICE, non_blocking=True)`** — Move os dados para a GPU. `non_blocking=True` permite que a transferência seja assíncrona (a CPU não espera a GPU terminar de receber os dados).

**`optimizer.zero_grad()`** — Zera os gradientes acumulados. Sem isso, os gradientes do batch anterior se somariam aos novos (comportamento indesejado na maioria dos casos).

**Forward pass com autocast:**
```python
with torch.cuda.amp.autocast():
    preds = model(inputs)           # Modelo prevê [x1, y1, x2, y2]
    loss = combined_loss(preds, targets, alpha=0.45)  # Calcula o erro
```
`autocast()` converte automaticamente operações para float16 quando seguro e mantém float32 quando necessário.

**Backward pass:**
```python
scaler.scale(loss).backward()       # Escala a loss e calcula gradientes
scaler.unscale_(optimizer)          # Des-escala para gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Limita gradientes
scaler.step(optimizer)              # Atualiza pesos
scaler.update()                     # Ajusta fator de escala para próxima iteração
```

**Gradient Clipping:** Se os gradientes forem muito grandes (explosão de gradientes), são cortados para no máximo norma 1.0. Isso evita instabilidade no treinamento:

```
Sem clipping:     gradiente = [100, -200, 50]  → pesos mudam drasticamente → instabilidade
Com clipping(1.0): gradiente = [0.41, -0.82, 0.41]  → atualização controlada
```

### A Validação

```python
model.eval()
val_loss = 0.0
val_iou = 0.0
val_margin_error = 0.0

with torch.no_grad():
    for inputs, targets in val_loader:
        ...
        preds = model(inputs)
        loss = combined_loss(preds, targets, alpha=0.45)
        
        val_loss += loss.item() * inputs.size(0)
```

**`model.eval()`** — Desativa Dropout e BatchNorm em modo de treino. Importante: o modelo se comporta **diferente** nos modos `train()` e `eval()`.

**`torch.no_grad()`** — Desativa o cálculo de gradientes. Na validação, não precisamos de gradientes (não estamos atualizando pesos), então economizamos memória e computação.

**`loss.item() * inputs.size(0)`** — Multiplica o loss médio pelo número de exemplos no batch para acumular corretamente quando os batches têm tamanhos diferentes.

### Métricas calculadas na validação

1. **Val Loss** — Loss combinada (IoU + margens) sobre os dados de validação.
2. **Val IoU** — IoU médio entre predições e gabaritos.
3. **Margin Error** — Erro médio relativo das margens.

### Early Stopping

```python
if avg_val_loss < best_loss:
    best_loss = avg_val_loss
    patience_counter = 0
    torch.save({...}, "models/best_model.pth")
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        log.info(f"Early stopping ativado após {epoch+1} épocas")
        break
```

**Conceito:** O modelo melhora rapidamente no início, mas depois de certo ponto começa a **decorar** os dados de treino. O sinal disso é que a loss de treino continua caindo, mas a **loss de validação para de cair** (ou sobe).

```
Loss de treino:    ──────────────────►   (sempre desce)
Loss de validação: ──────────┐
                             │ ← overfitting começa aqui
                             └───────►  (começa a subir)
                             
                         │ PATIENCE │
                         │ 15 epochs│
                         └──────────┘
                              ↓
                        PARA O TREINO
```

Early stopping monitora a loss de validação e **para o treinamento** depois de 15 épocas sem melhora, evitando overfitting e economizando tempo.

### O que é salvo no modelo

```python
torch.save({
    'model_state_dict': model.state_dict(),  # Todos os pesos da rede
    'val_loss': best_loss,                    # Melhor loss alcançada
    'iou': avg_iou,                           # IoU no momento do salvamento
    'margin_error': avg_margin_err,           # Erro de margem
    'img_size': IMG_SIZE,                     # Tamanho de entrada (para inferência)
    'epoch': epoch + 1                        # Época em que foi salvo
}, "models/best_model.pth")
```

O `state_dict()` contém **apenas os pesos** da rede (não a arquitetura). Para carregar o modelo depois, você precisa:
1. Criar uma instância de `MarginAwareCropModel()`.
2. Carregar o `state_dict` nela com `model.load_state_dict(checkpoint['model_state_dict'])`.

---

## 11. Ponto de Entrada (`__main__`)

```python
if __name__ == "__main__":
    _setup_logging()
    start_time = time.time()
    train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    log.info(f"Tempo total de execução: {elapsed_hms}")
```

1. **`if __name__ == "__main__"`** — Garante que o código só executa quando o script é rodado diretamente (`python train.py`), não quando é importado como módulo.
2. **`_setup_logging()`** — Configura os handlers de log (terminal + arquivo).
3. **Mede o tempo total** do treinamento e exibe no formato HH:MM:SS.

---

## 12. Glossário de Conceitos

| Conceito | Explicação |
|---|---|
| **Backbone** | A parte principal de uma rede neural que extrai features das imagens. Neste caso, a EfficientNet-B0. |
| **Batch** | Um subconjunto dos dados processado de uma vez. Exemplo: 16 imagens de um dataset de 1000. |
| **BatchNorm** | Normaliza os valores dentro de cada batch para ter média 0 e desvio 1. Estabiliza o treinamento. |
| **Bounding Box (bbox)** | Retângulo definido por 4 coordenadas [x1, y1, x2, y2] que delimita uma região de interesse. |
| **Convergência** | Quando o modelo para de melhorar significativamente — a loss estabiliza. |
| **Dropout** | Técnica que desliga neurônios aleatoriamente durante o treino para evitar overfitting. |
| **Early Stopping** | Para o treinamento quando a validação para de melhorar, evitando overfitting. |
| **Época (Epoch)** | Uma passagem completa por todos os dados de treino. |
| **Forward Pass** | O caminho "para frente": dados entram no modelo e uma previsão sai. |
| **Backward Pass** | O caminho "para trás": o erro (loss) é propagado de volta pela rede, calculando gradientes. |
| **Gradiente** | A direção e intensidade em que cada peso precisa ser ajustado para reduzir a loss. |
| **Ground Truth** | O "gabarito" — o valor correto que o modelo deveria prever. |
| **IoU** | Intersection over Union. Mede a sobreposição entre dois retângulos (0 = nenhuma, 1 = perfeita). |
| **Learning Rate** | O tamanho do "passo" que o otimizador dá ao ajustar os pesos. Muito grande → instável. Muito pequeno → lento. |
| **Loss (perda)** | Um número que mede o quão errado o modelo está. O objetivo é minimizá-lo. |
| **Mixed Precision** | Usar float16 e float32 juntos para acelerar o treinamento sem perder qualidade. |
| **Normalização** | Ajustar os valores para uma escala padrão (ex: 0-1, ou média 0 / desvio 1). |
| **Overfitting** | Quando o modelo "decora" os dados de treino e perde a capacidade de generalizar para dados novos. |
| **Regularização** | Técnicas para evitar overfitting: Dropout, weight decay, early stopping, etc. |
| **Sigmoid** | Função que comprime qualquer valor para o intervalo (0, 1). Fórmula: $\sigma(x) = \frac{1}{1+e^{-x}}$ |
| **State Dict** | Dicionário com todos os pesos treinados de uma rede neural. Pode ser salvo e carregado. |
| **Template Matching** | Técnica de visão computacional que encontra uma imagem menor dentro de uma maior por correlação. |
| **Tensor** | Array multidimensional — a unidade básica de dados no PyTorch. Similar a arrays NumPy, mas roda na GPU. |
| **Transfer Learning** | Reutilizar um modelo treinado em outra tarefa como ponto de partida. Economiza tempo e dados. |
| **Weight Decay** | Penalidade que reduz o tamanho dos pesos da rede a cada passo. Forma de regularização. |

---

## Fluxo Completo Resumido

```
1. Carregar caminhos de imagens originais e recortadas
                    │
2. Parear pelo nome (foto001.jpg ↔ foto001_editado.jpg)
                    │
3. Calcular bounding boxes via Template Matching (com cache)
                    │
4. Dividir em 90% treino + 10% validação
                    │
5. Criar DataLoaders (batches de 16 imagens, shuffle, workers paralelos)
                    │
6. Criar modelo EfficientNet-B0 + Regressor (com pesos ImageNet)
                    │
7. Para cada época (até 100):
   │
   ├── TREINO: Forward → Loss → Backward → Atualizar pesos
   │
   ├── VALIDAÇÃO: Forward → Loss → Métricas (sem atualizar pesos)
   │
   ├── Se melhorou → salvar modelo
   │
   └── Se 15 épocas sem melhora → parar (Early Stopping)
                    │
8. Modelo final salvo em models/best_model.pth
```
