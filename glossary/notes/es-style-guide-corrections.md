# Spanish: glossary corrections from the style guide (2026-11-26)

Source: `es/style.md` §12 (with §1.1, §3.3, §7, §9.1). Applied to
`glossary/terms/es.jsonl`; every changed entry was `proposed` and stays
`proposed`, with a note "Style guide 2026-11-26: …". No verified entry was
touched.

| Concept | Before | After | Reason |
|---|---|---|---|
| transformer | transformador | Transformer | Architecture name kept in English like ResNet; *transformador* is the electrical device. |
| transformer-block | bloque transformador | bloque Transformer | Same. |
| transformer-encoder | codificador transformador | codificador del Transformer | Same. |
| transformer-decoder | decodificador transformador | decodificador del Transformer | Same. |
| vision-transformer | transformador de visión | Vision Transformer; alt. ViT | Same. |
| diffusion-transformer | transformador de difusión | Diffusion Transformer; alt. DiT | Same. |
| word-embedding | incrustación de palabras | embedding de palabras | Compounds agree with the head entry *embedding* (loan). |
| position-embedding | incrustación posicional | embedding posicional | Same. |
| rotary-position-embedding | incrustación posicional rotatoria | embedding posicional rotatorio | Same (masculine agreement). |
| patch-embedding | incrustación de parche | embedding de parches | Same. |
| user-embedding | incrustación de usuario | embedding de usuario | Same. |
| item-embedding | incrustación de elemento | embedding de elemento | Same. |
| embedding-layer | capa de incrustación | capa de embedding | Same. |
| embedding-matrix | matriz de incrustaciones | matriz de embeddings | Same. |
| embedding-table | tabla de incrustaciones | tabla de embeddings | Same. |
| embedding-dimension | dimensión de la incrustación | dimensión del embedding | Same. |
| pooling | agrupación | pooling | *agrupación* is also clustering; Spanish CNN texts write *pooling*. |
| max-pooling | agrupación máxima | max pooling | Same. |
| average-pooling | agrupación por promedio | average pooling; alt. pooling por promedio | Same; guide gives both. |
| global-pooling | agrupación global | pooling global | Same. |
| global-average-pooling | agrupación global por promedio | global average pooling; alt. pooling global por promedio | Same; form derived from *average pooling*. |
| attention-pooling | agrupación por atención | pooling por atención | Same; form derived (guide lists the concept but no target). |
| pooling-layer | capa de agrupación | capa de pooling | Same. |
| pooling-window | ventana de agrupación | ventana de pooling | Same. |
| floating-point | coma flotante | punto flotante | Agrees with the decimal point (§4); majority usage in Latin America. |
| floating-point-precision | precisión de coma flotante | precisión de punto flotante | Same. |
| floating-point-format | formato de coma flotante | formato de punto flotante | Same. |
| softmax-temperature (alias *temperature*) | temperatura de softmax | temperatura de softmax; alt. temperatura | Bare *temperatura* for sampling temperature. |
| score-function | función score | función score; alt. función de score | Both forms occur. |
| cost | costo; alt. coste | costo | §1.1 fixes one form. |
| image-patch ("patch") | parche de imagen | unchanged (skipped) | Existing entry already agrees with *parche*; English *patch* alone would need an alias on the concept. |

New concepts (not added; reported to the coordinator): matrix-entry
(elemento), image-edge (borde), edge-map (mapa de bordes), flatten (aplanar),
banded-matrix (matriz banda), built-in (integrado), deep-learning (aprendizaje
profundo), machine-learning (aprendizaje automático), framework (framework,
m.), array (array, m.), optimizer (optimizador), post-training
(posentrenamiento), standard-normal-distribution (distribución normal
estándar).
