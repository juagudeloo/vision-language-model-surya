# Decisiones de diseño — VLM Surya

Registro vivo de decisiones de diseño originales (sin implementación de referencia a la que
apuntar) que surgen al trabajar la sección correspondiente de
[`surya_forward_pass_study_plan.md`](surya_forward_pass_study_plan.md). Mismos números de sección
que ese documento, para que los dos se puedan cruzar. Solo se agrega una sección cuando ya hay un
hallazgo real adentro.

## 9. Prototipo: encoder Surya congelado + MLP dummy

### Intensificar la región de interés antes de Surya, en vez de recortar tokens después

Qué es: en vez de pasar el disco completo sin modificar por Surya y recortar los tokens de la
región activa después del encoder, aplicar un multiplicador que intensifique esa región *antes* de
que la imagen entre a Surya — para que el encoder "preste más atención" a la zona de interés desde
el arranque.

A favor:
- El VLM podría en principio seguir teniendo acceso al contexto de todo el disco, no solo a la
  región recortada — resolvería la limitación que sí tiene el enfoque actual (recorte de tokens
  post-encoder), donde el modelo no puede decir nada de lo que pasa alrededor de la región activa.

En contra:
- Modifica la distribución de entrada respecto a lo que Surya vio en preentrenamiento (9 años de
  datos reales de SDO, sin ninguna intensificación artificial), sin que haya fine-tuning que le
  enseñe a interpretar esa modificación. Es la misma familia de riesgo por la que ya se descartó
  recortar la imagen antes del encoder — "el crop destruye las pistas de escala y posición en el
  disco que el modelo aprendió" — un multiplicador de intensidad es otra forma de alterar esas
  mismas pistas.

Evidencia: ninguna todavía — es una idea, no una decisión tomada ni probada. Se descarta por ahora
en favor del recorte de tokens post-encoder (más simple, sin este riesgo), y queda anotada acá por
si en algún momento se vuelve relevante — por ejemplo, si la limitación de "no contexto global" del
enfoque actual resulta ser un problema real una vez haya captions generados para evaluar.

## 10. Entrenamiento del projector: secuencia, pérdida y verificación

### Marcadores de imagen propios de Qwen3, en vez de un token nuevo

Qué es: reservar `<|vision_start|>` + 256 × `<|image_pad|>` + `<|vision_end|>` en el mensaje del usuario y
sustituir esas 256 posiciones por la salida del projector, en vez de añadir un token `<|image|>` al
tokenizador.

A favor:
- Los tres tokens ya existen en el vocabulario de Qwen3 y cada uno es un único id; no se toca el tokenizador.
- El embedding de los marcadores nunca se usa, porque se reemplaza.

En contra:
- No se sabe si esos embeddings se entrenaron en un modelo solo de texto; es irrelevante mientras se
  reemplacen, pero habría que revisarlo si algún día se dejan sin sustituir.
- Se aleja del libro, que añade un token propio; el mecanismo de sustitución es el mismo.

Evidencia: `tokenizer("<|vision_start|><|image_pad|><|image_pad|><|vision_end|>")` da `[151652, 151655,
151655, 151653]`. Resultados esperados de las funciones del núcleo, obtenidos con una implementación de
referencia, en [`fixtures/section10_sequence_fixture.json`](fixtures/section10_sequence_fixture.json).

### Pérdida solo sobre el caption y su fin, en vez de todo el texto

Qué es: supervisar el caption más `<|im_end|>` y enmascarar con `-100` la imagen, el prompt, las marcas del
chat, el relleno y el `\n` final de la plantilla.

A favor:
- Es lo que hace LLaVA (sec. 4.2, [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)): "predict the
  assistant answers and where to stop".
- Con un prompt fijo, los tokens del prompt se aprenden enseguida y diluirían la pérdida del caption.
- Incluir `<|im_end|>` enseña al modelo a parar.

En contra:
- El libro (p. 120) supervisa también el prompt; la calidad de ambas variantes no se comparó aquí.
- Si el prompt fuera parte de lo que se quiere aprender (por ejemplo, para preguntas variables), habría que
  revisarlo.

Evidencia: el texto de entrenamiento de Qwen3 empieza exactamente con el prompt de inferencia (comprobado
con `apply_chat_template`), y en la implementación de referencia lo supervisado del primer caso es
`A prominence eruption at the north-west limb.<|im_end|>`, 10 tokens de 286.

### bf16 con gradient checkpointing, en vez de 4 u 8 bits

Qué es: cargar Qwen3-8B congelado sin cuantizar, con checkpointing, batch 2 y gradient accumulation para el
batch efectivo, calculando los logits solo en las posiciones del caption.

A favor:
- A velocidad equivalente a 8 bits (2.75 contra 2.78 muestras/s), el LLM no se degrada, y entrenamiento y
  evaluación usan los mismos pesos.
- Un paso es limitado por cómputo: un batch mayor casi no acelera, así que el batch efectivo se sube con
  acumulación y no con memoria.

En contra:
- Margen de unos 4 GB sobre 21.5 GB (pico de 17.5 GB), que captions de más de 120 tokens reducen.
- 4 bits es cerca de un 30% más rápido (3.69 muestras/s) y deja mucha más holgura.
- No se midió la calidad de los captions con ninguna precisión.

Evidencia: barrido completo en 4 bits, 8 bits y bf16, con y sin checkpointing y con dos formas de calcular
la pérdida, en la tabla de `CLAUDE.md`; reproducible con
[`scripts/probes/throughput_probe.py`](../scripts/probes/throughput_probe.py). La primera medición dejó el
LLM en `eval()`, donde el checkpointing no actúa, y por eso concluyó mal que no bajaba el pico; la segunda,
en `train()`, lo corrige.

