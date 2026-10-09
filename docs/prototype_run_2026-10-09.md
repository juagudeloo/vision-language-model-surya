# Primera corrida del prototipo (9 oct 2026)

Resultado de entrenar el proyector de punta a punta con datos provisionales. **Es una prueba de que el
pipeline entrena, no un resultado sobre qué tan bien describe el Sol.**

## Qué se corrió

- Surya congelado, Qwen3-8B congelado en bf16 con gradient checkpointing, solo el proyector entrena (22.03 M parámetros).
- 20 épocas, batch 2 con acumulación 4 (efectivo 8), `learning_rate` 3e-4 con calentamiento del 3% y decaimiento coseno, 220 pasos, 703 s, pico de memoria 17.4 GB.
- La pérdida es solo sobre el caption y su `<|im_end|>`. Configuración: `configs/prototype.yaml`.
- **Se usó una implementación de respaldo de las tres funciones de `sequence.py`**, no las del usuario. Hay que repetir la corrida con las suyas; deberían dar los mismos números, porque ambas deben cumplir el mismo fixture.

## Datos

| Origen | Muestras | Nota |
|---|---|---|
| 7 enero 2014, 6 regiones activas × 18 instantes (X1.2 en AR 11944) | 72 train + 36 apartadas | Apartadas: AR 11946 y AR 11942, todos sus instantes |
| Labrosse et al. 2012, Figs. 2 a 5 | 7 train | Recuadro puesto a ojo sobre AIA 304; caption del texto del paper |
| Song et al. 2018, 2012-07-05 03:36 | 4 train | Incluye AR 11515 con una erupción M4.7 en curso |

Los captions salen de datos del catálogo NOAA (ubicación, manchas, clase de McIntosh, clase magnética,
erupción en curso) sin el número de región.

## Resultados

![curvas](../runs/prototype/loss_curves.png) (local, `runs/` no se versiona)

| Medida | Valor |
|---|---|
| Pérdida de entrenamiento | 3.54 al inicio, 0.021 al final |
| Pérdida en regiones apartadas | mejor 2.125 (época 9), final 2.286: sobreajuste desde la época 9 |
| Diferencia de imagen (pérdida con la imagen de otra región menos con la correcta) | positiva desde la época 2; promedio 0.535 entre las épocas 5 y 20 |
| Captions de entrenamiento idénticos a la referencia | 65 de 83; 8 difieren solo en 1° de coordenadas, 1 se cortó por el límite de tokens, 9 son paráfrasis de los eventos de papers |
| Captions de las 36 muestras apartadas | 7 degenerados ("Okay, ..."), 25 bien formados |
| De los 25 bien formados: hemisferio correcto / número de manchas correcto | 5 / 5 |

## Qué se puede decir y qué no

- **Sí:** el entrenamiento funciona de punta a punta; el modelo reproduce los captions de entrenamiento; y la imagen aporta información que también ayuda en regiones que no vio, porque la pérdida apartada sube en promedio 0.54 nats por token si se le da la imagen equivocada.
- **No:** que describa bien una región nueva. En las dos regiones apartadas, ambas en el hemisferio norte, el hemisferio salió correcto en 5 de 25 captions bien formados (por debajo del azar) y 7 de 36 salidas se desvían al estilo de razonamiento de Qwen3, señal de que esas imágenes caen fuera de lo que el proyector aprendió.
- **Por qué es esperable:** 83 muestras de entrenamiento, casi todas de un solo día y de ~6 regiones distintas, no bastan para aprender qué se ve en una región nueva; las 36 apartadas son del mismo día. El conjunto de prueba es pequeño y de dos regiones, así que los porcentajes tienen mucho ruido.
- El mejor punto de control se eligió con las mismas muestras apartadas, de modo que 2.125 es algo optimista.

## Qué sigue

Más regiones distintas: es lo que más falta. Descargar un instante de unas 20 fechas repartidas entre 2011 y 2014 daría
del orden de 150 regiones nuevas, y permitiría apartar días enteros en vez de regiones del mismo día.
