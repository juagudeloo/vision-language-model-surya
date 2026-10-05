# Study plan — Forward pass de Surya: de imagen a embeddings

Guía paso a paso para la tarea de Notion "Primer forward pass: imagen a embeddings Surya"
(Sprint 1). Cada sección dice qué explorar, qué código de referencia existe ya en el repo, y qué
deberías obtener antes de pasar a la siguiente. **Vos escribís el notebook/script — esto es la
guía, no la solución.**

Repo de Surya: `/scratchsan/observatorio/juagudeloo/Doctorado/Surya` (clon de `NASA-IMPACT/Surya`).
Ambiente: `pytorch_jupyter` (conda, ya tiene torch 2.6.0+cu124 + einops/timm/peft agregados el
13 sep).

## Notebooks

Un notebook agrupa varias secciones cuando forman un mismo arco narrativo con un entregable claro
al final — no es un notebook por sección. Uno nuevo arranca solo cuando el tema cambia de verdad,
aunque reutilice objetos del notebook anterior (en ese caso, se reconstruyen — no se importa estado
entre notebooks).

| Notebook | Secciones | Tema | Estado |
|---|---|---|---|
| [`01-surya_forward_pass.ipynb`](../notebook/01-surya_forward_pass.ipynb) | 1–8 | Poner a andar el forward pass de Surya: ambiente, checkpoint, arquitectura, datos reales, pesos reales, verificación visual | Terminado |
| [`02-projector_prototype.ipynb`](../notebook/02-projector_prototype.ipynb) | 9 | Prototipo del projector: token-cropping, pooling, MLP, congelar Surya | Terminado |
| `03-projector_training.ipynb` (propuesto) | 10–11 | Entrenar el projector: armar la secuencia con tokens visuales, pérdida enmascarada, bucle | En curso |

## 1. Activar el ambiente

**Qué obtener al final**: una shell con CUDA visible y `import surya` funcionando.

```bash
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter
cd /scratchsan/observatorio/juagudeloo/Doctorado/Surya
python3 -c "import torch, surya; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Explicación**: reusamos el env que ya arma MUISCA en vez de crear uno desde
`surya_workshop/environment.yml` — ya traía casi todo (`torch`, `sunpy`, `huggingface_hub`,
`safetensors`); solo faltaron `einops`/`timm`/`peft`/`pytest`. `surya` se importa por path porque
estás parado en el repo, no porque esté instalado (`pip install -e`) — no hace falta instalarlo.

## 2. Ensayo con un modelo liviano análogo (antes de tocar Surya)

**Qué obtener al final**: un tensor de tokens `(1, 197, 192)` de un ViT diminuto, corriendo en
CPU en segundos.

Surya es 366M parámetros, código propio (no `transformers`), y solo corre cómodo en GPU. Antes de
depurar la mecánica completa (descargar checkpoint, instanciar, cargar pesos, extraer tokens) sobre
ese modelo pesado, conviene ensayar el mismo patrón general sobre algo que descarga y corre al
toque:

```python
import timm, torch

model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
model.eval()

x = torch.randn(1, 3, 224, 224)  # imagen dummy
with torch.no_grad():
    tokens = model.forward_features(x)

print(tokens.shape)  # (1, 197, 192) -- 196 parches de 16x16 + 1 token CLS
```

**Explicación**: `timm` ya trae `forward_features()`, que corta el forward justo antes de la
cabeza de clasificación — el mismo lugar conceptual de donde vamos a sacar los embeddings de
Surya. La diferencia clave que vas a encontrar en la sección 6: **Surya no expone un método así**,
por eso hay que llamar sus submódulos a mano. Si este paso corre limpio, ya validaste que el env
funciona y que entendés qué forma de tensor estás buscando (tokens antes de la cabeza) antes de
pelearte con la complejidad real de Surya.

**Si te da `RepositoryNotFoundError` / "OAuth token has expired" al descargar pesos de HF**
(afectó este mismo paso el 22 sep 2026): no es un problema del repo ni de la red — es un token
OAuth vencido guardado en `$HF_HOME/token`. Tu `HF_HOME` está en
`/scratchsan/observatorio/juagudeloo/.cache/huggingface` (no en el default de tu home), así que si
alguna vez hiciste login ahí (`huggingface-cli login`/`hf auth login`) y ese token venció, **toda**
petición falla con 401 aunque el repo sea público — `huggingface_hub` no cae a modo anónimo cuando
el error es 401. Arreglo (no hace falta estar logueado para nada de este documento, todos los repos
son públicos):
```bash
huggingface-cli logout   # o: hf auth logout
```
Confirmá qué `HF_HOME` tenés vos con `echo $HF_HOME` antes de tocar nada — si no coincide con la
ruta de arriba, el archivo del token está en otro lado.

## 3. Descargar checkpoint + datos de validación

**Qué obtener al final**: `data/Surya-1.0/{config.yaml,scalers.yaml,surya.366m.v1.pt}` y
23 archivos `.nc` en `data/Surya-1.0_validation_data/`, corriendo el comando **desde la raíz del
repo** (para que las rutas relativas del índice de prueba calcen).

Referencia: [`tests/test_surya.py:59-70`](../../Surya/tests/test_surya.py#L59-L70) (fixture `download_data`).

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nasa-ibm-ai4science/Surya-1.0",
    local_dir="data/Surya-1.0",
    allow_patterns=["config.yaml", "scalers.yaml", "surya.366m.v1.pt"],
)
snapshot_download(
    repo_id="nasa-ibm-ai4science/Surya-1.0_validation_data",
    repo_type="dataset",
    local_dir="data/Surya-1.0_validation_data",
    allow_patterns="20140107_1[5-9]??.nc",
)
```

**Explicación**: el checkpoint es ~1.5 GB (según lo documentado en el [`CLAUDE.md`](../CLAUDE.md) del repo de VLM,
no hay tamaño exacto en el repo de Surya). Los datos de validación son un día real de SDO
(2014-01-07, 15:00–19:59, cadencia 12 min) ya en los 13 canales que Surya necesita — esto te
resuelve el problema de "qué imagen uso" sin depender del pipeline de Julio, que sigue bloqueado.

## 4. Construir el modelo

**Qué obtener al final**: un objeto `HelioSpectFormer` con arquitectura completa pero **pesos
aleatorios** (los pesos reales se cargan en la sección 6), y poder señalar sus tres piezas:
`model.embedding`, `model.backbone` y `model.unembed`. No hay forward pass todavía: el objetivo es
que puedas leer qué contiene el modelo y por qué cada pieza está ahí.

Referencia: [`tests/test_surya.py:83-110`](../../Surya/tests/test_surya.py#L83-L110) (fixture `model`), lee `config.yaml` descargado en el
paso 3.

```python
import yaml
import torch
from surya.models.helio_spectformer import HelioSpectFormer

with open("data/Surya-1.0/config.yaml") as fp:
    config = yaml.safe_load(fp)

model = HelioSpectFormer(
    img_size=config["model"]["img_size"],
    patch_size=config["model"]["patch_size"],
    in_chans=len(config["data"]["sdo_channels"]),
    embed_dim=config["model"]["embed_dim"],
    time_embedding={
        "type": "linear",
        "time_dim": len(config["data"]["time_delta_input_minutes"]),
    },
    depth=config["model"]["depth"],
    n_spectral_blocks=config["model"]["n_spectral_blocks"],
    num_heads=config["model"]["num_heads"],
    mlp_ratio=config["model"]["mlp_ratio"],
    drop_rate=config["model"]["drop_rate"],
    window_size=config["model"]["window_size"],
    dp_rank=config["model"]["dp_rank"],
    learned_flow=config["model"]["learned_flow"],
    rpe=config["model"]["rpe"],
    ensemble=config["model"]["ensemble"],
    finetune=config["model"]["finetune"],
)

print(type(model.embedding).__name__, type(model.backbone).__name__, type(model.unembed).__name__)
print("spectral:", len(model.backbone.blocks_spectral_gating), "attention:", len(model.backbone.blocks_attention))
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parametros")
```

Salida real (probado en el cluster, desde la raíz del repo, con el `config.yaml` de la sección 3):
```
LinearEmbedding SpectFormer LinearDecoder
spectral: 2 attention: 8
366.2 M parametros
```

Lo que **no** pasé respecto al fixture del test, y por qué: `dtype` (el constructor dice que no se usa),
`init_weights` (los pesos se sobrescriben al cargar el checkpoint en la sección 6),
`checkpoint_layers` (ahorra memoria solo en entrenamiento) y `use_latitude_in_learned_flow` (solo
aplica con `learned_flow: true`, que aquí es `false`). Las 366.2 M de parámetros cuadran con las
366M que anuncia Surya, señal de que la arquitectura quedó bien armada.

**Qué es `HelioSpectFormer` realmente.** Es un *contenedor* de tres piezas en serie, definido en
[`helio_spectformer.py`](../../Surya/surya/models/helio_spectformer.py); no es él quien mezcla spectral gating
con atención:

```
ts (B,13,2,4096,4096) → embedding → backbone → unembed (solo pretraining)
                        patches+tiempo   SpectFormer     reconstruye la imagen
```

1. **`model.embedding`** (`LinearEmbedding`, en [`embedding.py`](../../Surya/surya/models/embedding.py)): convierte la imagen en
   tokens. Cada parche de 16×16 píxeles pasa a ser un vector de 1280 números, y el tiempo entra
   como canales extra (ver "Encoder shapes" en el [`CLAUDE.md`](../CLAUDE.md) del repo).
2. **`model.backbone`** (`SpectFormer`, en [`spectformer.py:183`](../../Surya/surya/models/spectformer.py#L183)): **aquí sí** están los dos
   tipos de bloque. Es una pila de `depth` bloques, y es la parte que llamaremos "el encoder".
3. **`model.unembed`** (`LinearDecoder`): revierte el paso 1 (tokens → imagen de 13 canales), sirve
   para la tarea de pretraining (predecir la imagen una hora después). Para nosotros es ruido.

**Tu pregunta: ¿une spectral gating con long-short attention?** Casi: lo hace el `backbone`
(`SpectFormer`), no `HelioSpectFormer`. Y "une" es engañoso — **no los mezcla ni los pone en
paralelo, los apila**: primero los `n_spectral_blocks` bloques espectrales, después los
`depth - n_spectral_blocks` de atención ([`spectformer.py:288`](../../Surya/surya/models/spectformer.py#L288),
`chain(blocks_spectral_gating, blocks_attention)`). Con el `config.yaml` real: `depth: 10`,
`n_spectral_blocks: 2` → **2 bloques espectrales y luego 8 de atención**.

Qué hace cada tipo, y por qué van en ese orden:
- **Spectral gating** ([`SpectralGatingNetwork.forward`](../../Surya/surya/models/spectformer.py#L50)):
  reacomoda los 65,536 tokens en su rejilla 256×256, aplica una FFT 2D, multiplica cada frecuencia
  por un peso aprendido y vuelve con la FFT inversa. Es un filtro **global** aprendido: cada
  token queda influido por toda la imagen de una vez, y es barato (no hay matriz de atención de
  65,536×65,536). Analogía: un ecualizador de audio, pero sobre las frecuencias espaciales de la imagen.
- **Long-short attention** (`BlockAttention`, [`spectformer.py:110`](../../Surya/surya/models/spectformer.py#L110)):
  atención tipo transformer, pero por ventanas locales (`window_size`) más un resumen de largo
  alcance (`dp_rank`), para que sea tratable a esta resolución. Aquí se refinan relaciones
  finas entre regiones.
- El orden espectral → atención es primero mezclar contexto global barato, después razonar en detalle.

Es lo que hace que Surya no sea un ViT plano, y es la razón por la que la receta de SigLIP2 no se
transfiere directamente.

**Parámetros del config que importan** (valores reales de
[`config.yaml`](../notebook/data/Surya-1.0/config.yaml)):
- `embed_dim: 1280`: ancho de cada token; será el `in_dim` de tu projector.
- `depth: 10`, `n_spectral_blocks: 2`: como arriba.
- `window_size: 2`, `dp_rank: 4`: solo afectan a los bloques de atención.
- `learned_flow: false`, `rpe: false`, `ensemble: null`: opciones desactivadas en este checkpoint;
  puedes ignorarlas.
- `finetune: false`: por eso `model.unembed` **sí existe** (con `finetune: true` el constructor
  no lo crea). Ese es el modelo completo de pretraining; en la sección 6 no lo llamas.

## 5. Cargar el dataset real

**Qué obtener al final**: un `batch_data` (dict) con clave `"ts"` de forma `(1, 13, 2, 4096, 4096)`
y `"time_delta_input"` de forma `(1, 2)`.

Referencia: [`tests/test_surya.py:113-155`](../../Surya/tests/test_surya.py#L113-L155) (fixtures `scalers`, `dataset`, `dataloader`), usa
`tests/test_surya_index.csv` tal cual (ya apunta a `data/Surya-1.0_validation_data/...`, que
coincide con donde bajaste todo en el paso 3).

**Explicación**: `scalers.yaml` guarda parámetros de **`StandardScaler`** por canal — no min-max —
y el dataset aplica `signum-log` seguido de z-score antes de devolver el tensor
([`surya/datasets/helio.py`](../../Surya/surya/datasets/helio.py), [`transformations.py:129`](../../Surya/surya/datasets/transformations.py#L129)). Para una extracción "limpia" del embedding
(no estás entrenando ni robusteciendo el modelo), pasá `drop_hmi_probability=0.0` y
`num_mask_aia_channels=0` — si copiás los valores de `config["data"]` tal cual, vas a estar usando
los de *pretraining*, que enmascaran canales al azar. `rollout_steps=1` alcanza: no te interesa el
forecast, solo necesitás un batch de entrada válido.

**Por qué necesitás igual el target, aunque no te interese**: [`HelioNetCDFDataset`](../../Surya/surya/datasets/helio.py#L190)
solo cuenta un timestamp como "muestra válida" si *también* existen los timestamps del forecast
(`filter_valid_indices`, [`helio.py:300`](../../Surya/surya/datasets/helio.py#L300)) — así que tu índice necesita cubrir `t-60`, `t`, `t+60` y
`t+120` min, aunque `ts`/`time_delta_input` sean lo único que uses de la salida.

Corriendo esto desde `notebook/` (donde ya tenés los datos de la sección 3), sin tocar nada del
repo de Surya:

```python
import yaml
from torch.utils.data import DataLoader

from surya.datasets.helio import HelioNetCDFDataset
from surya.utils.data import build_scalers, custom_collate_fn

with open("data/Surya-1.0/config.yaml") as fp:
    config = yaml.safe_load(fp)

scalers_info = yaml.safe_load(open("data/Surya-1.0/scalers.yaml"))
scalers = build_scalers(info=scalers_info)

dataset = HelioNetCDFDataset(
    index_path="../../Surya/tests/test_surya_index.csv",  # el índice real del repo de Surya
    time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
    time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
    n_input_timestamps=len(config["data"]["time_delta_input_minutes"]),
    rollout_steps=1,
    channels=config["data"]["sdo_channels"],
    drop_hmi_probability=0.0,
    num_mask_aia_channels=0,
    use_latitude_in_learned_flow=config["data"]["use_latitude_in_learned_flow"],
    scalers=scalers,
    phase="valid",
    pooling=config["data"]["pooling"],
    random_vert_flip=False,
)
print(f"{len(dataset)} sample(s) en el dataset")

dataloader = DataLoader(
    dataset, shuffle=False, batch_size=1, num_workers=0,
    pin_memory=False, drop_last=False, collate_fn=custom_collate_fn,
)

batch_data, batch_metadata = next(iter(dataloader))
print("ts:", batch_data["ts"].shape, batch_data["ts"].dtype)
print("time_delta_input:", batch_data["time_delta_input"].shape, batch_data["time_delta_input"])
```
Salida real (probado desde `notebook/`, sin symlinks ni cambios en el repo de Surya):
```
8 sample(s) en el dataset
ts: torch.Size([1, 13, 2, 4096, 4096]) torch.float32
time_delta_input: torch.Size([1, 2]) tensor([[1., 0.]])
```

`time_delta_input = [1., 0.]` no son minutos, son **horas antes del frame más reciente**: el primer
input está 1 hora antes del segundo, y el segundo es el propio instante de referencia (0 horas antes
de sí mismo). Por eso no ves literalmente `-60`/`0` — es la misma información (`[-60, 0]` min), pero
reexpresada como horas-antes-del-último-frame. Como ya vimos en el `CLAUDE.md`, `forward` acepta
este tensor pero no lo usa.

## 6. Cargar los pesos y extraer el embedding del encoder

**Qué obtener al final**: un tensor `(1, 65536, 1280)` — el embedding documentado el 27 de agosto,
ahora confirmado con un forward pass real en vez de solo leído de la config.

Referencia: [`tests/test_surya.py:259-266`](../../Surya/tests/test_surya.py#L259-L266) para cargar pesos; [`surya/models/helio_spectformer.py:242`](../../Surya/surya/models/helio_spectformer.py#L242)
(`forward`) para ver por qué **no** hay que llamarlo directo.

```python
model.to(device)
weights = torch.load("data/Surya-1.0/surya.366m.v1.pt", map_location=device)
model.load_state_dict(weights, strict=True)
model.eval()

with torch.no_grad():
    tokens = model.embedding(batch_data["ts"].to(device), batch_data["time_delta_input"].to(device))
    tokens = model.backbone(tokens)

print(tokens.shape)  # esperado: (1, 65536, 1280)
```

Salida real (probado en GPU real, batch de la sección 5):
```
tokens.shape: (1, 65536, 1280)
```

**Explicación**: `model(batch)` completo no es "el embedding más algo de más que se descarta" —
literalmente **no devuelve el embedding en absoluto** con este checkpoint. Mirá el `forward` real,
[`helio_spectformer.py:242-280`](../../Surya/surya/models/helio_spectformer.py#L242-L280):

```python
tokens = self.backbone(tokens)
if self.finetune:
    return tokens          # con finetune=True, model(batch) SÍ devolvería el embedding
forecast_hat = self.unembed(tokens)
...
return forecast_hat        # con finetune=False (nuestro caso), esto es lo que llega
```

Con `finetune: false` (nuestro `config.yaml`), el `return` final es `forecast_hat` — la imagen
reconstruida, forma `(B, C, H, W)` — no los tokens. El tensor `(1, 65536, 1280)` nunca sale del
método con este flag. Por eso hace falta llamar `embedding` y `backbone` manualmente: no es
evitar cómputo de más, es que lo que necesitás no está en lo que `model()` devuelve.

Dos detalles prácticos del snippet, ya verificados:
- `model.eval()` desactiva dropout — pero este checkpoint tiene `drop_rate: 0.0` en el
  `config.yaml`, así que acá no cambia nada numéricamente; sigue siendo buena práctica general.
- `torch.no_grad()` no es una traba contra "entrenar por accidente" (para eso haría falta además
  `.backward()` y un optimizador) — evita que PyTorch construya el grafo de autograd, que cuesta
  memoria real aunque nunca lo uses. Medido en esta misma GPU: este forward con `no_grad()` usa
  **7.1 GB** de pico; el mismo modelo con forward+backward completo (fine-tuning real) llega a
  **~19-20 GB** y no entra en la A4500 de 20 GB ni con gradient checkpointing + bf16 — la
  diferencia es en gran parte ese grafo que `no_grad()` evita construir.

## 7. Visualizar y verificar

**Qué obtener al final**: una figura guardada mostrando estructura espacial coherente con el disco
solar (más "señal" en el centro que en las esquinas).

```python
import matplotlib.pyplot as plt

norms = tokens[0].norm(dim=-1).reshape(256, 256).cpu()
plt.imshow(norms, cmap="turbo")
plt.title("Norma L2 por token — grilla 256x256")
plt.colorbar()
plt.savefig("forward_pass_embedding_norm.png", dpi=100, bbox_inches="tight")
```

Salida real (probado en GPU real, encadenando pesos reales + `batch_data` real de las secciones
5-6):
```
norms.shape: (256, 256), min=17.5, max=361.0
```
El disco solar se ve completo y nítido, con *limb brightening* en el borde y puntos de norma alta
coincidiendo con regiones activas — nada de ruido uniforme. Confirma que el pipeline completo
(pesos reales + datos reales + forward pass) funciona de punta a punta, no solo por separado.

**Por qué la norma promedio (no la media) es la métrica correcta**: un promedio simple sobre los
1280 valores de un token puede dar cerca de cero aunque el token esté muy activo, si sus valores
positivos y negativos se cancelan entre sí. La norma L2 no tiene ese problema — solo es chica si
**todos** los valores son chicos a la vez, nunca por cancelación de signos. Por eso es una medida
más confiable de "cuánta actividad real hay" en cada token.

**Qué hacer si el mapa sale ruido, sin forma del disco**: la explicación con más probabilidad a
priori **no** es que Surya no aprendió nada útil — es un modelo maduro, entrenado 9 años sobre
datos reales. La sospecha debe recaer primero sobre el propio código: orden de canales, escalado,
seguir usando pesos aleatorios en vez de los reales, o un error al reacomodar los tokens en la
grilla 256×256. Regla general: con un modelo fundacional ya validado, un resultado raro es mucho
más probable que sea un bug propio que una falla del modelo.

## 8. Cerrar el loop

- Actualizar la tarea en Notion a `Done`, con la forma real obtenida y el tiempo/VRAM que tardó
  (dato que hoy no existe documentado en ningún lado del repo de Surya).
- Con esto quedan desbloqueadas "Seleccionar LLM final" (ya sin dependencias pendientes) y
  "Prototipo: encoder Surya congelado + MLP dummy" (Sprint 1).

## 9. Prototipo: encoder Surya congelado + MLP dummy

### Development notebook

[`02-projector_prototype.ipynb`](../notebook/02-projector_prototype.ipynb)

**Empezó notebook nuevo:** las secciones 1-8 terminaron en un entregable cerrado (embedding real
confirmado y verificado visualmente); esto arrancó un arco distinto, construir el projector, y
aunque reutiliza `tokens` conceptualmente, se recalculó desde cero en el notebook nuevo en vez de
depender de estado del notebook anterior.

**Qué obtener al final**: partiendo del embedding real `(1, 65536, 1280)`, llegar a lo que recibiría
el LLM — `(1, 256, 4096)`, los "tokens visuales" para una región activa — pasando por token-cropping,
pooling, y un projector MLP sin entrenar (pesos aleatorios). Además, verificar que el gradiente solo
llega al projector, nunca a Surya.

### Diseño: token-cropping (original, sin referencia existente)

No hay código de referencia para esto — es la parte nueva del proyecto, no algo que Surya o LLaVA-1.5
ya resuelvan. El mecanismo: cada token cubre un parche de 16×16 px sobre la grilla 4096/16=256, así
que un bbox en píxeles se mapea a índices de token por división entera entre 16 (techo para el borde
superior, para no perder el último parche parcial).

```python
def crop_tokens(tokens, bbox_px, patch_size=16, grid_size=256):
    """bbox_px: (x_min, y_min, x_max, y_max) en pixeles sobre la imagen 4096x4096."""
    x_min, y_min, x_max, y_max = bbox_px
    tx_min, ty_min = x_min // patch_size, y_min // patch_size
    tx_max, ty_max = -(-x_max // patch_size), -(-y_max // patch_size)  # techo (ceil)
    grid = tokens.reshape(tokens.shape[0], grid_size, grid_size, tokens.shape[-1])
    cropped = grid[:, ty_min:ty_max, tx_min:tx_max, :]
    return cropped  # (B, h_tokens, w_tokens, D)
```

Salida real, con una región activa de ejemplo de 512×512 px:
```
cropped shape: (1, 32, 32, 1280)
```
32×32 = 1024 tokens, como documenta el `CLAUDE.md` del repo — confirmado con código real, no solo
aritmética a mano.

**Por qué recortar en el espacio de tokens y no en píxeles**: ya lo vimos — Surya solo vio el disco
completo en preentrenamiento; recortar antes del encoder le cambia la distribución de entrada sin
fine-tuning que le enseñe a interpretarla (misma familia de riesgo que la idea del multiplicador de
intensidad, descartada por ahora — ver [`design_decisions.md`](design_decisions.md#9-prototipo-encoder-surya-congelado--mlp-dummy)).
Limitación conocida y aceptada: el VLM, tal como está diseñado ahora, solo puede describir la región
recortada, no el contexto alrededor — también en `design_decisions.md`.

### Pooling: 1024 → 256 tokens

```python
import torch.nn.functional as F

def pool_tokens(cropped, out_size=16):
    B, h, w, D = cropped.shape
    x = cropped.permute(0, 3, 1, 2)  # (B, D, h, w) -- pooling 2D espera canales primero
    x = F.adaptive_avg_pool2d(x, output_size=(out_size, out_size))
    x = x.permute(0, 2, 3, 1).reshape(B, out_size * out_size, D)
    return x  # (B, out_size*out_size, D)
```

Salida real:
```
pooled shape: (1, 256, 1280)
```
`adaptive_avg_pool2d` trata los `D=1280` valores de cada token como "canales" de una imagen de
32×32, y reduce el plano espacial a 16×16=256 — el `~256` que ya estaba anotado como objetivo.

### El projector — arquitectura de referencia: LLaVA-1.5

A diferencia del token-cropping, esto sí tiene una plantilla real: el conector `mlp2x_gelu` de
LLaVA-1.5. Su detalle exacto — verificado, no asumido — es que la dimensión **oculta** del MLP es
igual a la dimensión de salida (el `hidden_size` del LLM), no un valor intermedio arbitrario:
`Linear(in_dim, out_dim) → GELU → Linear(out_dim, out_dim)`.

```python
from torch import nn

class Projector(nn.Module):
    def __init__(self, in_dim=1280, out_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)
```

### Congelar Surya, y verificar que el gradiente respeta eso

`requires_grad=False` en los parámetros de Surya (no `no_grad()` — son mecanismos distintos, ver la
sección anterior) es lo que garantiza que el optimizador nunca lo toque, sin importar en qué bloque
de código se llame su forward más adelante.

```python
for p in model.parameters():
    p.requires_grad = False

projector = Projector().to(device)
visual_tokens = projector(pooled)  # (1, 256, 4096) -- esto recibiria el LLM

loss = visual_tokens.sum()  # loss de juguete, solo para verificar el flujo de gradientes
loss.backward()
proj_grad_ok = all(p.grad is not None for p in projector.parameters())
surya_has_grad = any(p.grad is not None for p in model.parameters())
```

Salida real:
```
visual_tokens shape: (1, 256, 4096)
projector params entrenables: 22.03 M
parametros de Surya con requires_grad=True (deben ser 0): 0
projector tiene gradientes: True
Surya tiene gradientes (debe ser False): False
```
`(1, 256, 4096)` coincide con el `out_dim` de Qwen3-8B, decidido el 25 de septiembre — el projector
entero es solo 22 M de parámetros, frente a los 366 M congelados de Surya. Confirmado con código
real: el gradiente llega al projector y nunca a Surya.

## 10. Entrenamiento del projector: secuencia, pérdida y verificación

**Empieza notebook nuevo:** `03-projector_training.ipynb` — es otro arco: hasta la sección 9 se construyó
el projector, ahora se entrena. Reutiliza los tokens `(256, 1280)` del notebook 02, pero se leen desde
el dataset cacheado ([`cached_dataset.py`](../scripts/cached_dataset.py)), no se importa estado de aquel notebook.

**Qué obtener al final**: dado un batch del dataset cacheado, producir las tres cosas que recibe el LLM:
`inputs_embeds` de forma `(B, S, 4096)` con los 256 vectores del projector puestos en las posiciones
marcadas, el `attention_mask`, y las `labels`, con `-100` en todo salvo el caption y su fin. Y comprobar
con código real que el gradiente llega al projector y a nada más.

### Qué se verificó antes de decidir

El texto de entrenamiento de Qwen3 empieza exactamente con el prompt que se usará al generar, incluido el
bloque `<think>` vacío (`enable_thinking=False`). Corrido para este proyecto:

```
INFER: <|im_start|>user\n<IMG> Describe this solar observation.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
TRAIN: ...lo mismo... A prominence eruption at the north-west limb.<|im_end|>\n
el texto de entrenamiento empieza con el de inferencia: True
```

Eso fija la frontera de qué se supervisa: todo lo que viene después del prompt de inferencia. Además el
tokenizador de Qwen3 ya trae `<|vision_start|>` (151652), `<|image_pad|>` (151655) y `<|vision_end|>`
(151653), cada uno como un solo token. Y su `pad` (151643, `<|endoftext|>`) es distinto de su token de
fin (151645, `<|im_end|>`), lo que importa para el enmascarado (ver la decisión B).

### Diseño: marcador de imagen (original, con referencia parcial)

**Qué se decidió:** reservar en el mensaje del usuario `<|vision_start|>` + 256 × `<|image_pad|>` +
`<|vision_end|>`, tokenizar todo el chat normalmente, y sustituir después los embeddings de esas 256
posiciones por la salida del projector.

**Alternativas consideradas:**
- Token nuevo `<|image|>`, como en el libro — funcionalmente equivalente, pero obliga a registrar un token
  en el tokenizador y a mantener coherente el checkpoint; no aporta nada si ya existen los de Qwen3.
- Pegar los embeddings visuales delante de todo el texto del chat, sin marcador — es lo que se usó en las
  pruebas de VRAM; deja la imagen fuera de la estructura del chat (antes de `<|im_start|>`) y no escala si
  algún día hay más de una imagen o texto antes de ella.

**Por qué esta:** los tres tokens ya están en el vocabulario (verificado arriba), no se toca el
tokenizador, y el embedding original de `<|image_pad|>` nunca se usa porque se sustituye. El mecanismo es el
del libro de Noyan et al., cap. 3, pp. 118–120 del PDF (`referencias/Vision Language Models (...)_bibis.ir.pdf`,
local y no versionado): el texto completo pasa por el embedder, y las posiciones del marcador se reemplazan
por los embeddings reales de la imagen.

**Juicio, no hecho establecido:** no sé si los embeddings de esos marcadores se entrenaron en Qwen3-8B (un
modelo solo de texto); no importa, porque se reemplazan, pero conviene saberlo si alguna vez se decide
dejarlos sin sustituir.

### Diseño: sobre qué tokens se calcula la pérdida (original, con referencia)

**Qué se decidió:** supervisar solo el caption más `<|im_end|>`. Se enmascaran (`-100`) la imagen, el prompt
del usuario, las marcas del chat y el relleno, y también el `\n` que la plantilla pone después de
`<|im_end|>`, porque es pegamento de la plantilla y no algo que se deba aprender.

**Alternativas consideradas:**
- Todo el texto salvo la imagen, como muestra el libro (p. 120: se enmascaran los tokens de imagen y el
  relleno, así que el prompt también cuenta) — con un prompt idéntico en todas las muestras, esos tokens se
  aprenden enseguida y diluyen la señal del caption.
- Usar `pad_token_id` como `ignore_index`, como hace el libro — en Qwen3 el `pad` y el fin son tokens
  distintos, pero es más claro usar `-100`, la convención de Hugging Face, y evita enmascarar por accidente
  un caption que contenga ese id.

**Por qué esta:** LLaVA (Liu et al., 2023, [arXiv:2304.08485](https://arxiv.org/abs/2304.08485), sección 4.2)
entrena "to predict the assistant answers and where to stop", y en su etapa 1 deja congelados el codificador
visual y el LLM, entrenando solo la matriz de proyección ("a compatible visual tokenizer for the frozen LLM").
Es el mismo montaje que este proyecto. Incluir `<|im_end|>` en la pérdida es lo que enseña a parar.

**Juicio, no hecho establecido:** no se comparó la calidad de las dos variantes. Con un prompt fijo se
espera poca diferencia; si más adelante se varían los prompts, enmascararlos sigue siendo lo correcto.

### Diseño: precisión del LLM y memoria (original, con medición)

**Qué se decidió:** Qwen3-8B congelado en bf16 sin cuantizar, con gradient checkpointing, batch 2 y
gradient accumulation para llegar al batch efectivo que se quiera, calculando los logits solo en las
posiciones del caption.

**Alternativas consideradas** (medido en la A4500 de 21.5 GB, secuencia de 256 + 40 + 120 tokens, logits
solo en el caption):

| Pesos | Checkpointing | Batch | Pico | Muestras/s | 1000 muestras |
|---|---|---|---|---|---|
| 4 bits (nf4) | no | 2 | 11.7 GB | 3.69 | 4.5 min |
| 8 bits | no | 2 | 16.2 GB | 2.78 | 6.0 min |
| bf16 | sí | 2 | 17.5 GB | 2.75 | 6.1 min |

**Por qué esta:** a velocidad equivalente a 8 bits, bf16 no degrada el LLM, y entrenamiento y evaluación usan
los mismos pesos. Un paso es limitado por cómputo, así que un batch mayor casi no acelera (bf16 con
checkpointing queda plano en 2.7–2.8 muestras/s desde batch 2): el batch efectivo se sube con acumulación, no
con memoria.

**Juicio, no hecho establecido:** la calidad de los captions con cada precisión no se midió. El margen en
bf16 es de unos 4 GB y captions de más de 120 tokens lo reducen; la precisión es un campo de la configuración
para bajar a 4 bits sin tocar código.

### Qué construir

Tres funciones, sin implementación de referencia aquí: es el núcleo y lo escribe Juan. Contrato:

- `build_example(tokenizer, caption, n_visual=256, prompt="Describe this solar observation.")` → `(input_ids, labels)`
  como listas de enteros, de la misma longitud; `labels` vale `-100` salvo en el caption y su `<|im_end|>`.
- `collate(tokenizer, examples)` → `(input_ids, labels, attention_mask)` como tensores `(B, S_max)`, con relleno
  a la derecha: `pad_token_id` en `input_ids`, `-100` en `labels` y `0` en `attention_mask`.
- `inject_visual(llm, input_ids, visual_emb)` → `inputs_embeds` de forma `(B, S, D)`, con la salida del
  projector `(B, 256, D)` en las posiciones donde `input_ids == <|image_pad|>` y el resto sin cambios.

Pistas que son hechos verificados y no la solución: `tokenizer(texto, add_special_tokens=False)` convierte
cada `<|image_pad|>` del texto en un único id; `apply_chat_template(..., enable_thinking=False)` con y sin
`add_generation_prompt` da el prefijo y el texto completo, y el segundo empieza con el primero.

### Objetivo verificable

Material: [`fixtures/section10_sequence_fixture.json`](fixtures/section10_sequence_fixture.json), con los dos
captions de abajo y el resultado esperado. Hay que obtener exactamente esto con la implementación propia:

| Caption | `n_tokens` | `vision_start` | `n_image_pad` | `vision_end` | `n_masked` | primer supervisado | `n_supervised` |
|---|---|---|---|---|---|---|---|
| `A prominence eruption at the north-west limb.` | 286 | 3 | 256 | 260 | 276 | 275 | 10 |
| `White-light flare in AR 11515 seen in AIA 1600.` | 298 | 3 | 256 | 260 | 276 | 275 | 22 |

Lo supervisado en el primero, decodificado: `A prominence eruption at the north-west limb.<|im_end|>`, y la
etiqueta del último token es `-100`. Con `collate` sobre los dos: `input_ids` de forma `(2, 298)`, suma de
`attention_mask` por fila `[286, 298]`, y el relleno de la fila más corta con todas sus etiquetas en `-100`.

Para probar `inject_visual` y el gradiente sin cargar Qwen3-8B, un LLM aleatorio diminuto con el vocabulario
real (corre en CPU en segundos):

```python
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
llm = Qwen3ForCausalLM(cfg).eval()
for p in llm.parameters():
    p.requires_grad_(False)
projector = nn.Sequential(nn.Linear(1280, 64), nn.GELU(), nn.Linear(64, 64))
visual = torch.randn(2, 256, 1280)
```

Con el batch anterior debe cumplirse (resultado esperado, obtenido con una implementación de referencia):
`inputs_embeds` de forma `(2, 298, 64)`; las 512 posiciones de imagen contienen exactamente la salida del
projector; las demás posiciones coinciden con `llm.get_input_embeddings()(input_ids)`; la pérdida es finita;
y tras `loss.backward()` el gradiente del projector es distinto de cero mientras que el del LLM no existe.

## 11. Bucle de entrenamiento

Sobre las funciones de la sección 10: cargar el LLM en bf16 con checkpointing, entrenar solo el projector con
AdamW y acumulación de gradiente, y registrar la pérdida por paso.

## 12. Lanzarlo en el clúster

Configuración, `sbatch` siguiendo el patrón de `MUISCA/tools/`, y la prueba de humo con datos de juguete.
