<!--en:325423f8-->
# Transformer

<!--en:e785897f-->
In :numref:`chap_attention` haben wir die Multi-Head-Attention-Schicht, Positionskodierungen und Attention-Varianten für lange Kontexte entwickelt. Attention allein mischt jedoch nur Informationen zwischen den Token. Ohne nichtlineare Transformationen für jedes einzelne Token, Residualverbindungen (engl. residual connections) und Normalisierung ergibt selbst ein Stapel von Attention-Schichten kein trainierbares tiefes Netz. In diesem Kapitel bauen wir aus diesen Komponenten vollständige **Transformer**-Architekturen auf :cite:`Vaswani.Shazeer.Parmar.ea.2017`. Wir untersuchen, wie wir sie trainieren, für die Inferenz bereitstellen und skalieren.

<!--en:2bb20a08-->
Zunächst bauen wir Schritt für Schritt einen Stapel kausaler Blöcke auf und trainieren ihn. In :numref:`sec_transformer-block` kombinieren wir Multi-Head-Attention und ein positionsweises Feedforward-Netz in einem Residualstrom (engl. residual stream) mit Pre-Norm (`RMSNorm` und `SwiGLU`). Wir zeigen, warum die Skalierung der Residualprojektionen ($1/\sqrt{2L}$) und die Normalisierung von Abfragen und Schlüsseln die Signalnormen und Attention-Logits auch bei zunehmender Tiefe beschränkt halten. In :numref:`sec_gpt` stapeln wir diese kausalen Blöcke zwischen den Tokeneinbettungen (engl. token embeddings) und dem Ausgabekopf des Sprachmodells, der dieselben Gewichte wie die Einbettungen verwendet. Wir trainieren ein GPT-Modell auf Byteebene mit WikiText-103 und vergleichen es mit einem entsprechend ausgelegten rekurrenten Referenzmodell. Anschließend laden wir vortrainierte GPT-2-Gewichte (`124M`) und untersuchen den Einfluss der Softmax-Temperatur sowie das Top-$k$-Sampling. Daraus arbeiten wir die gemeinsamen Architekturmerkmale moderner großer Sprachmodelle mit offenen Gewichten heraus (:numref:`sec_modern-recipe`). Bei der naiven autoregressiven Generierung berechnet das Modell in jedem Schritt das gesamte Präfix neu. Deshalb führen wir in :numref:`sec_kv-cache` den KV-Cache für Schlüssel und Werte ein. Anhand der Hardware-Roofline vergleichen wir den rechenbegrenzten Prefill mit der Dekodierung, die durch die Speicherbandbreite begrenzt ist. Wir verkleinern den Cache in drei Dimensionen. Bei der Anzahl der Köpfe setzen wir Grouped-Query-Attention und Multi-Query-Attention ein, bei der Kanalbreite Schlüssel-Wert-Projektionen von niedrigem Rang und bei der Sequenzlänge gleitende Fenster mit Attention-Sinks.

<!--en:77ccb35a-->
Die Aufgaben reichen über die kausale Textgenerierung hinaus. In :numref:`sec_transformer` vergleichen wir kausale Decoder mit zwei weiteren Architekturen. Zum einen trainieren wir bidirektionale Encoder durch maskierte Sprachmodellierung (engl. masked language modeling) :cite:`Devlin.Chang.Lee.ea.2018`. Zum anderen koppeln wir für Sequenz-zu-Sequenz-Aufgaben Encoder und Decoder durch Cross-Attention, auch über latente Flaschenhälse fester Größe :cite:`Jaegle.Gimeno.Brock.ea.2021`. In :numref:`sec_vision-transformer` wenden wir den bidirektionalen Encoder dann direkt auf Bilder an. Dazu projizieren wir nicht überlappende Pixelausschnitte in Tokeneinbettungen :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`. Wir vergleichen die erlernte zweidimensionale Positionsstruktur und die Dateneffizienz mit denen eines ResNet mit gleicher Parameterzahl.

<!--en:fe1fa0b4-->
Zum Schluss wenden wir uns der Skalierung und der bedingten Berechnung (engl. conditional computation) zu. In :numref:`sec_scaling-laws` zählen wir die Parameter ($N \approx 12 L d^2$) und die FLOPs für das Training ($C \approx 6 N D$). Wir trainieren 15 Modelle in einem Versuchsraster. Mit den veröffentlichten Anpassungen der Chinchilla-Potenzgesetze :cite:`kaplan2020scaling,hoffmann2022training` zeichnen wir die IsoFLOP-Kurven und stellen die rechenoptimale Aufteilung des Budgets dar. Wir erklären, warum die Inferenzkosten über die gesamte Nutzungsdauer beim heutigen Pre-Training für kleinere Modelle sprechen, die mit weit mehr Token pro Parameter trainiert werden, als das Chinchilla-Verhältnis vorsieht. In :numref:`sec_moe` entkoppeln wir die Zahl der gespeicherten Parameter von den FLOPs pro Token. Dazu ersetzen wir das dichte Feedforward-Netz durch eine Gruppe von Experten, aus denen ein Router für jedes Token auswählt :cite:`Shazeer.Mirhoseini.Maziarz.ea.2017,fedus2022switch`. Für den Lastausgleich vergleichen wir einen Hilfsverlust mit einer Regelung des Router-Bias, die ohne Hilfsverlust auskommt :cite:`Wang.Gao.Zeng.ea.2024`. In :numref:`sec_transformers-further-reading` geben wir einen Überblick über Erweiterungen, von der Platzierung der Normalisierung über latente Attention bis zur spekulativen Dekodierung.

<!--en:e6c857bf-->
Transformer vom einzelnen Block bis zu heutigen Spitzenarchitekturen<br>**Residualstrom · Pre-Norm & SwiGLU · GPT · KV-Cache & Roofline · Encoder & ViT · Skalierungsgesetze · Mixture of Experts**

<!--en:296b0aa1-->
::: {.slide title="Warum Attention allein nicht ausreicht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:8fb6c38e-->
- **Fehlende Nichtlinearität pro Token:** Ohne positionsweises Feedforward-Netz können gestapelte Attention-Schichten innerhalb der einzelnen Tokenvektoren keine hierarchischen nichtlinearen Merkmale bilden.

<!--en:5b46cad7-->
- **Instabilität bei großer Tiefe und begrenzte Bandbreite:** Tiefe Stapel brauchen Residualströme mit Pre-Norm. Bei der Generierung müssen alle $12 L d^2$ Blockgewichte und der KV-Cache aus dem HBM gelesen werden.

<!--en:4068a1dd-->
::: {.slide title="Die vier Säulen der Architektur in Kapitel §{num:chap_transformers}"}

<!--en:c7d201c1-->
[Architekturen im Überblick]{.kicker}

<!--en:253ff008-->
- **1. Aufbauen und trainieren (§§{sec_transformer-block}–§{num:sec_gpt}):** Pre-Norm, `RMSNorm`, `QK-Norm`, Skalierung mit $1/\sqrt{2L}$, `SwiGLU` und Pre-Training eines GPT-Modells, das nur aus einem Decoder besteht.

<!--en:af465b30-->
- **2. Für die Inferenz bereitstellen und komprimieren (§{sec_kv-cache}):** KV-Cache, Roofline-Analyse, GQA, MLA mit niedrigem Rang und Attention-Sinks.

<!--en:a69656bd-->
- **3. Neu verschalten und skalieren (§§{sec_transformer}–§{num:sec_moe}):** Encoder, Cross-Attention, Bildausschnitte im ViT, Chinchilla-Skalierungsgesetze und Mixture of Experts (MoE) mit wenigen aktiven Experten pro Token.

<!--en:bf119ab2-->
::: {.slide title="Benchmarks auf einer einzelnen GPU in Kapitel §{num:chap_transformers}"}

<!--en:ff9e6bb1-->
[Durchgehende Beispiele]{.kicker}

<!--en:b30403f7-->
- Jeden Architekturbaustein prüfen wir an reproduzierbaren Modellen. Auf einer einzelnen GPU dauert das Training oder die Leistungsmessung nur Minuten.

<!--en:f8ebc089-->
- **Validierung von Anfang bis Ende:** Von der Prüfung der Initialisierung bei 32 Schichten und dem Pre-Training auf Byteebene mit WikiText-103 bis zu Roofline-Messungen auf der RTX 4090, einem Versuchsraster mit 15 Modellen und MoE-Routing mit 8 Experten.

<!--en:9913bfa1-->
::: {.slide title="Vom Transformer aus dem Jahr 2017 zum heutigen Bauplan"}

<!--en:688e3215-->
[Der heutige Bauplan]{.kicker}

<!--en:aeced3fa-->
- Alle Modelle dieses Kapitels erzeugen wir mit demselben parametrisierbaren Konstruktor. Seine Voreinstellungen entsprechen denen führender großer Sprachmodelle mit offenen Gewichten (`RMSNorm`, `RoPE`, `SwiGLU`, GQA/MLA, MoE).

<!--en:b3f79a21-->
- **Als Nächstes (§{sec_transformer-block} Der Transformer-Block):** Wie die Anordnung der Normalisierung bei Pre-Norm, `RMSNorm`, `QK-Norm` und `SwiGLU` den Kollaps der Tokenrepräsentationen verhindern und die Gradienten über die Schichten hinweg stabil halten.
