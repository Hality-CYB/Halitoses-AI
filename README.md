# IA do Hality

Triagem de indício de halitose a partir de uma foto da língua. Este repositório contém
os modelos de IA do projeto.

| Seção | Modelo | Autor |
|---|---|---|
| [Classificador](#classificador) | "esta língua indica halitose?" | Eduardo Alcaria Lopes |
| [Segmentador](#segmentador) | "onde exatamente está a língua?" | Paulo Augusto |

## Descrição

A pessoa fotografa a própria língua e o sistema devolve uma de quatro respostas:

- **indicio**: indício de halitose, com probabilidade
- **sem_indicio**: sem indício, com probabilidade
- **inconclusivo**: não há base suficiente para responder
- **rejeitado**: a foto não serve, sempre com o motivo ("foto tremida", "não identifiquei uma língua")

O sistema **não diagnostica**. Ele faz triagem e sempre orienta a procurar um dentista.

O sinal vem da saburra, a camada esbranquiçada no dorso da língua. Língua com muita saburra
é mais clara e menos saturada, e a saburra abriga as bactérias que produzem os compostos
sulfurados voláteis (VSC) responsáveis pelo odor.

## Arquitetura

São três modelos, cada um responde uma pergunta:

| Modelo | Pergunta | Treinado com |
|---|---|---|
| Gate | "isto é uma língua?" | 2.331 fotos de língua contra 5.000 fotos do COCO |
| Segmentador (TongueNet) | "onde exatamente está a língua?" | fotos da clínica |
| Classificador | "esta língua indica halitose?" | 306 fotos com nota clínica |

Caminho de uma foto até a resposta:

```
foto
 |
 |- 1. normalização      decodifica (JPEG, PNG, BMP, HEIC), corrige rotação EXIF, limita a 1024 px
 |
 |  as etapas 2 a 7 rodam 5 vezes, sobre variações geométricas da mesma foto
 |- 2. qualidade          muito escura? muito clara? tremida?        -> rejeita
 |- 3. gate               tem uma língua aí?                          -> rejeita
 |- 4. segmentação        TongueNet recorta a língua                  (Segmentador)
 |- 5. sanidade           a máscara tem área e forma plausíveis?      -> rejeita
 |- 6. características    34 medidas de cor e textura, só dentro da língua
 |- 7. classificador      probabilidade calibrada                     (Classificador)
 |
 |- 8. decisão            voto de maioria entre as 5 passagens
                          sem maioria clara ou probabilidade perto do limiar -> inconclusivo
```

Rejeitar é resposta bem-sucedida, não erro: a API devolve HTTP 200 com o motivo.

## Dados

- 345 anamneses com nota clínica de 1 a 3, atribuída por avaliador humano
- 321 fotos casadas com essas anamneses, 306 utilizáveis depois do filtro de máscara
- distribuição das 306: nota 1 = 22, nota 2 = 107, nota 3 = 177
- origem: clínica parceira na PUCRS
- divisão estratificada por paciente, 70% treino, 15% validação, 15% teste (semente 42)

As fotos são dados de pacientes e **não ficam no git** nem nesta wiki.

## Instalação e uso

A forma recomendada é a imagem Docker `hality-ai`, que já vem com os pesos dos modelos e roda
sem internet. Ela é distribuída como asset do release, porque os pesos passam de 100 MB e
foram treinados com dados de pacientes. Não redistribua fora do grupo.

```bash
gh release download v0.2.0 --repo Hality-CYB/Halitoses-AI --pattern "hality-ai.tar.gz"
docker load -i hality-ai.tar.gz
docker run -d --name hality-ai -p 8000:8000 --restart unless-stopped hality-ai:latest
```

Testar:

```bash
curl http://127.0.0.1:8000/saude
curl -F "foto=@lingua.jpg" http://127.0.0.1:8000/analisar
```

| Rota | O que faz |
|---|---|
| `POST /analisar` | recebe a foto (multipart, campo `foto`, até 20 MB) e devolve o veredito |
| `GET /saude` | informa o segmentador carregado, o limiar e a faixa de abstenção |
| `GET /docs` | documentação interativa (Swagger) |

Exemplo de resposta:

```json
{
  "veredito": "indicio",
  "motivo": "Indicio compativel com halitose. Procure um dentista para avaliacao. Isto nao e um diagnostico.",
  "probabilidade": 0.8995,
  "area_lingua": 0.4512,
  "nitidez": 210.4,
  "confianca_lingua": 0.9994,
  "dispersao": 0.0049,
  "passagens": 5,
  "arquivo": "lingua.jpg",
  "tempo_ms": 3120,
  "aviso": "Triagem, nao diagnostico. Procure um dentista para avaliacao."
}
```

Sem Docker: Python 3.12 (o 3.14 não serve, o PyTorch falha no treino).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
tar -xzf hality-models.tar.gz
.venv/bin/python -m uvicorn hality.api:app --port 8000
```

## Classificador

**Autor:** Eduardo Alcaria Lopes

Responde a pergunta "esta língua indica halitose?". Recebe a foto e a máscara da língua
produzida pelo [Segmentador](#segmentador) e devolve uma probabilidade calibrada.

### Descrição

| Item | Valor |
|---|---|
| Modelo | `HistGradientBoostingClassifier` (scikit-learn, `max_iter=300`) |
| Calibração | `CalibratedClassifierCV`, isotônica, 5 folds |
| Entrada | 34 características de cor e textura medidas só dentro da máscara |
| Alvo | binário: nota clínica 3 contra o resto |
| Saída | probabilidade em [0,02, 0,98] |
| Artefato | `models/classificador.pkl` |
| Código | `hality/features.py`, `hality/train_classifier.py`, `hality/pipeline.py` |

### Características (features)

A foto é reduzida para 128×128 e todas as medidas são feitas só nos pixels da máscara.

| Grupo | Características | Quantidade |
|---|---|---|
| Cor RGB | média, desvio, percentil 10 e percentil 90 de R, G e B | 12 |
| Cor HSV | média, desvio, percentil 10 e percentil 90 de H, S e V | 12 |
| Gerais | saturação média, área da máscara | 2 |
| Textura | média e desvio do gradiente vertical do brilho | 2 |
| Setores | saturação e brilho em ponta, meio e base da língua | 6 |

A característica que mais pesa é `HSV_S_p10`, o percentil 10 da saturação: **quão pálida é a
porção mais pálida da língua**. É a camada de saburra, medida de forma contínua.

A lista `FEATURE_NAMES` (nome e ordem) é gravada junto com o modelo. Mudar a extração invalida
o modelo treinado, e `python -m hality.features` tem um teste de estabilidade para detectar isso.

### Por que alvo binário

Três classes colapsam: a nota 1 tem só 22 fotos e o modelo acertava 4. O corte é nota 3 contra
o resto, o caso de maior consequência e de melhor separabilidade. A maioria dos falsos
positivos atuais vem da nota 2: o modelo erra na fronteira entre moderado e grave.

### Por que só a imagem, sem questionário

| Entrada | AUC |
|---|---|
| Imagem sozinha | 0,797 |
| Imagem + anamnese sem `Q6` | 0,799 |
| Anamnese sozinha sem `Q6` | 0,615 |

A pergunta `Q6` correlaciona 0,528 com o rótulo e pode ser o próprio rótulo disfarçado. Sem
ela, o questionário não agrega nada. O produto passa a depender só da foto.

### Limiar e abstenção

- **Limiar:** escolhido na validação, entre os pontos com sensibilidade ≥ 0,80, o que
  maximiza sensibilidade + especificidade − 1.
- **Faixa de abstenção:** `limiar ± 2 × ruído`, com ruído medido de 0,043 entre refotos da mesma
  língua. Probabilidade dentro da faixa vira `inconclusivo`. Sem o ruído no artefato, usa os
  percentis 40 a 60 da validação.
- **Voto de 5 passagens:** o pipeline roda o classificador sobre 5 variações geométricas da foto
  (rotação até 5°, recorte até 7%, espelhamento) e decide por maioria. Sem maioria clara a
  resposta é `inconclusivo`.

| Versão | Veredito mudou entre refotos |
|---|---|
| Uma passagem | 48,3% |
| Média da probabilidade sobre 5 passagens | 36,7% |
| Voto de maioria sobre 5 passagens | 23,3% |

A API também devolve uma `classe` 1 a 3: 3 se a probabilidade passa do limiar, 2 se passa da
metade do limiar, 1 abaixo disso.

### Resultados

**AUC por validação cruzada repetida** (306 fotos, 5 folds × 20 repetições): **0,77**, intervalo
0,68 a 0,86. É a estimativa mais estável e o número a citar.

**Pipeline completo** (validação + teste, n = 93, fotos nunca vistas no treino do classificador):

| Métrica | Valor |
|---|---|
| Cobertura (fotos com veredito definitivo) | 72,0% |
| Acurácia | 0,776 |
| F1 macro | 0,759 |
| Sensibilidade | 0,875 |
| Especificidade | 0,630 |
| Precisão | 0,778 |
| AUC | 0,748 |

Referência: chutar sempre a classe majoritária acerta 0,597, com F1 macro 0,374.

Métrica sem cobertura é meia informação: quanto mais o sistema rejeita, melhor ele parece.

### O que foi testado e não ajudou

| Experimento | Resultado |
|---|---|
| DINOv2 (backbone pré-treinado grande) no lugar das features | AUC 0,761 contra 0,801 |
| Normalizar iluminação (gray-world, white-patch) | AUC cai de 0,786 para ~0,69 |
| Jitter de cor no aumento de dados (±5% a ±35%) | piora em todas as intensidades |
| Aumentar saburra em fotos nota 2 e rotular como nota 3 | AUC −0,055 |
| Segmentar a saburra explicitamente (`coating.py`) | nenhum ganho |
| Modelo ordinal, restrição monotônica | nenhum ganho |
| Feature limiarizada de pixels pálidos | AUC 0,553 sozinha, não agrega |

Modelos grandes pré-treinados aprendem a ignorar cor, e aqui a cor é o sinal. Aumento de dados
geométrico (espelhamento) é seguro e entra no treino.

**Teste de confundidor:** prever o rótulo só com metadados do arquivo (largura, altura, bytes,
formato) dá AUC 0,498 no nosso conjunto, ou seja, acaso. Um dataset externo deu 0,995 e foi
descartado.

### Como treinar

Precisa das fotos da clínica em `Hality-Project-main/`, fora do git.

```bash
python -m hality.train_classifier
```

Leva cerca de 3 minutos em CPU. Gera `models/classificador.pkl` e `models/metricas.json`.

Auto-testes:

```bash
python -m hality.features
python -m hality.data
python -m hality.pipeline
```

### Limitações

- 306 fotos é pouco. A curva de aprendizado ainda sobe (AUC 0,761 com 208 fotos de treino,
  0,773 com 244). Não existe dataset público com rótulo de halitose.
- O conjunto de teste já foi aberto uma vez antes de uma mudança no pipeline, então não é mais
  um holdout limpo.
- O TongueNet foi treinado com fotos que incluem as de validação e teste do classificador. Os
  números do pipeline estão otimistas até o TongueNet ser retreinado sem elas.
- 23,3% de refotos ainda mudam de veredito. É o principal item aberto.

## Segmentador

**Autor:** Paulo Augusto

Responde a pergunta "onde exatamente está a língua?". Recebe a foto e devolve uma máscara
binária com os pixels da língua. Todas as medidas do [Classificador](#classificador) são
feitas só dentro dessa máscara, então um erro aqui vira veredito errado lá.

### Descrição

| Item | Valor |
|---|---|
| Modelo principal | TongueNet, DeepLabV3 com backbone ResNet50 (torchvision), 2 classes |
| Entrada | foto RGB redimensionada para 384×384, normalização ImageNet |
| Saída | máscara 128×128, pixel é língua se a probabilidade da classe 1 > 0,5 |
| Artefato | `models/tonguenet.pt` |
| Reserva | U-Net própria, `models/segmentador.pt` |
| Código | `hality/segmentacao.py` |

Na subida, se `models/tonguenet.pt` existir, o sistema usa o TongueNet. Se não existir, cai
para a U-Net de reserva. `GET /saude` informa qual está carregado.

### Resultados

| Segmentador | IoU contra máscara de especialista (BioHit, n = 30) | IoU contra máscara Roboflow (validação, n = 45) |
|---|---|---|
| **TongueNet** | **0,903** | 0,823 |
| U-Net de reserva | 0,729 | 0,843 |

A máscara da Roboflow tem defeitos conhecidos (em várias fotos ela pega só um fragmento sobre a
mancha de saburra), por isso a coluna do especialista é a régua honesta.

Nas 15 fotos em que a segmentação antiga só encontrava fragmentos, o TongueNet recupera a língua
inteira em 14.

### Checagens da máscara no pipeline

Depois de segmentar, o pipeline confere se a máscara é plausível antes de classificar:

| Checagem | Regra | Mensagem ao usuário |
|---|---|---|
| Área mínima | máscara cobre ≥ 12% da foto | "Não identifiquei uma língua. Aproxime a câmera e mostre a língua." |
| Fragmento | maior região conectada tem ≥ 55% da máscara | "Não consegui delimitar a língua. Refaça a foto." |
| Pixels mínimos | ≥ 200 pixels na máscara 128×128 | "Região da língua pequena demais. Aproxime a câmera." |

Não existe limite superior de área: máscaras cobrindo 62% a 70% do quadro são close-ups
legítimos e estão entre as melhores fotos do conjunto.

### U-Net de reserva

| Item | Valor |
|---|---|
| Arquitetura | U-Net de 3 níveis, largura 16, 483 mil parâmetros |
| Entrada | 192×192 |
| Perda | BCE + Dice |
| Treino | AdamW, lr 3e-3, cosine, até 60 épocas, early stopping com paciência 12 |
| Aumento de dados | espelhamento horizontal e rotação de 90° |
| Rótulos | máscaras da API da Roboflow (destilação) |

Se a máscara da U-Net cobrir menos de 20% da foto, ela tenta de novo sobre a foto com realce de
contraste (estica cada canal entre os percentis 2 e 98) e fica com a máscara maior. O realce
entra **só** na segmentação, nunca nas características do classificador, porque o brilho
absoluto carrega o sinal clínico.

A U-Net treina só na partição de treino, para não produzir máscaras melhores justamente nas
fotos em que o classificador é medido.

### Experimentos

| Experimento | Arquivo | Resultado |
|---|---|---|
| DeepLabV3 com backbone congelado, 192×192 | `train_deeplab.py` | alternativa, não adotada |
| SegFormer | `train_segformer.py` | alternativa, não adotada |
| Contorno por raios radiais a partir de um ponto | `raios.py` | IoU 0,37, descartado |
| SAM com pontos gerados pela U-Net | `sam_segmenter.py` | resgata fotos difíceis, não integrado |
| Encoder do SAM em ONNX na GPU AMD (DirectML) | `sam_onnx.py` | 5,80 s para 0,16 s por imagem (35×) |

O contorno por raios falhou porque a fronteira entre língua e lábio não se distingue por cor
local (d' = 0,54), e o lábio é 60,7% do contorno. Cavidade e dente separam bem (d' > 3) e
viraram os pontos negativos do SAM.

### Como usar no código

```python
import numpy as np
from PIL import Image
from hality.segmentacao import Segmentador

seg = Segmentador()
rgb = np.asarray(Image.open("lingua.jpg").convert("RGB"))
mascara = seg(rgb)
print(seg.nome, mascara.shape, mascara.mean())
```

Auto-teste:

```bash
python -m hality.segmentacao
```

### Limitações

- O TongueNet foi treinado com fotos da clínica que incluem as de validação e teste do
  classificador. Os números do pipeline completo ficam otimistas até um retreino sem essas fotos.
- O código de treino do TongueNet não está neste repositório; só o peso
  (`tonguenet.pt`) é carregado.
- Selfies caseiras, com luz de teto e enquadramento livre, não estão representadas no treino.

## Latência

Cerca de 3 s por foto aceita em CPU local e 5,6 s dentro do Docker. Foto rejeitada leva
menos de 1 s.

## Limitações gerais

- **Protocolo do rótulo desconhecido.** Não sabemos se a nota 1 a 3 veio de halímetro, teste
  organoléptico ou impressão clínica. Se o avaliador olhou a língua para dar a nota, parte do
  resultado é circular.
- **Todas as fotos vêm da mesma clínica.** Selfie caseira, com luz de teto, não está representada.
- **Prevalência.** Na clínica (58% de casos graves), 78% dos sinalizados têm a condição. Numa
  população aberta com 20% de prevalência, cai para cerca de 37%.
- **O gate nunca viu boca fechada.** Os negativos dele são fotos do COCO.

## Roadmap

1. Descobrir com a clínica o protocolo da nota 1 a 3, e se o avaliador via a língua.
2. Retreinar o TongueNet sem as fotos de validação e teste do classificador e remedir tudo.
3. Medir a constância de veredito com o TongueNet.
4. Coletar negativos difíceis para o gate (boca fechada, língua não exposta).
5. Coletar mais fotos rotuladas, de preferência fora da clínica.

## Autores

- **Eduardo Alcaria Lopes**: classificador, pipeline de inferência, API e imagem Docker
- **Paulo Augusto**: segmentador (TongueNet)

## Status do projeto

Em desenvolvimento ativo. Versão atual da imagem: `v0.2.0`.
