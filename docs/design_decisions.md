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
