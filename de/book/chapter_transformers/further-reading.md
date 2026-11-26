<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:66927834-->
Die in diesem Kapitel entwickelten Architekturen haben denselben Grundaufbau. Sie verwenden einen Residualstrom (engl. residual stream) mit vorgeschalteter Normalisierung (:numref:`sec_transformer-block`), kausale oder bidirektionale Multi-Head-Attention (:numref:`sec_gpt`, :numref:`sec_transformer`) und einen KV-Cache, der über Köpfe, Token oder Kanäle hinweg komprimiert ist (:numref:`sec_kv-cache`). Die Feedforward-Schichten lassen sich entweder als dichte Schichten skalieren (:numref:`sec_scaling-laws`) oder indem für jedes Token nur wenige Experten ausgewählt werden (:numref:`sec_moe`). Dieser Abschnitt gibt einen Überblick über wichtige Architekturvarianten und Forschungsrichtungen, die auf diesen Grundlagen aufbauen.

<!--en:caa58cfa-->
## Position der Normalisierung und Varianten des Residualstroms

<!--en:85ee2203-->
Die meisten modernen Transformer verwenden Pre-Norm (`RMSNorm` vor jeder Teilschicht). So bleibt der Identitätspfad (engl. identity path) von der Ausgabe zurück zur ersten Schicht ungedämpft (:numref:`sec_transformer-block`). In sehr tiefen Modellen oder bei langen Trainingsläufen mit FP16/BF16 können die Normen der Aktivierungen in einem unnormalisierten Residualstrom jedoch von Schicht zu Schicht anwachsen. Zugleich können die Beträge der Attention-Scores so weit steigen, dass die Softmax-Funktion in Sättigung geht. Zwei Gruppen von Verfahren setzen hier an (:numref:`fig_norm-taxonomy`).

<!--en:70d88e79-->
![Vier Möglichkeiten zur Normalisierung an einer Teilschicht $f$. Post-LN normalisiert den Residualstrom selbst, Pre-LN nur die Eingabe der Teilschicht. OLMo 2 normalisiert nur die Ausgabe der Teilschicht, bevor es sie zum Residualstrom addiert. Peri-LN (Sandwich-LN, Gemma 3) normalisiert sowohl die Eingabe als auch die Ausgabe der Teilschicht.](../img/mdl-transformers-norm-taxonomy.svg)

<!--en:ee13b226-->
- **Normalisierung der Teilschichtausgabe (Sandwich-LN / Peri-LN) und Post-Norm-Varianten.** Normalisiert man sowohl die Eingabe als auch die Ausgabe jeder Teilschicht ($x + \operatorname{Norm}(f(\operatorname{Norm}(x)))$), bleibt der Beitrag jedes Updates zum Residualstrom begrenzt. Die Skip-Verbindung, die die Identität abbildet, bleibt dabei erhalten. Dieses Muster verwenden CogView :cite:`Ding.Yang.Hong.ea.2021` und Gemma 2/3 :cite:`Gemma.Team.2024,Gemma.Team.2025`. OLMo 2 :cite:`OLMo.2025` verschiebt die RMSNorm an die Ausgabe jeder Teilschicht ($x + \operatorname{Norm}(f(x))$) und kombiniert sie mit QK-Norm. Das begrenzt das Wachstum des Residualstroms, ohne dass der Identitätspfad nach der Addition durch eine Normalisierung führt.

<!--en:98fd5eb8-->
- **Normalisierung von Abfragen und Schlüsseln (QK-Norm).** Jeder Kopf normalisiert Abfragen und Schlüssel mit RMSNorm, bevor er $QK^\top / \sqrt{d_h}$ berechnet (:numref:`subsec_qk-norm`). Nach der Cauchy-Schwarz-Ungleichung sind die Logits vor der Softmax-Funktion damit beschränkt, unabhängig davon, wie stark der Residualstrom wächst :cite:`Henry.Dachapally.Pawar.ea.2020,Dehghani.Djolonga.Mustafa.ea.2023`. Dieses Verfahren ist in ViT-22B, Gemma 3, Qwen3 und OLMo 2 Standard.

<!--en:e0c720ff-->
- **Tiefenabhängig skalierte Post-Norm und glatte Begrenzung der Logits.** DeepNorm :cite:`Wang.Ma.Dong.ea.2024` behält den Post-LN-Block bei und verstärkt die Skip-Verbindung durch einen expliziten, tiefenabhängigen Faktor. Der Block berechnet $\operatorname{Norm}(\alpha x + f(x))$ mit $\alpha = (2L)^{1/4}$. Zugleich skaliert DeepNorm die Gewichte des anderen Zweigs bei der Initialisierung mit dem Faktor $\beta = (8L)^{-1/4}$ herunter. Durch beide Maßnahmen bleibt jedes Update im Verhältnis zum Residualstrom klein. Das stabilisiert Post-LN-Transformer mit bis zu 1000 Schichten. Gemma 2 begrenzt alternativ die Attention-Logits und die Ausgabe-Logits mit einer glatten tanh-Funktion ($c \tanh(z / c)$). Bei den Attention-Logits hat QK-Norm diese glatte Begrenzung in neueren Modellen weitgehend abgelöst.

<!--en:9ff7da3e-->
## Feedforward-Netze und Aktivierungsfunktionen

<!--en:d954cc7d-->
Der ursprüngliche Transformer verwendete ein Feedforward-Netz mit zwei Matrizen, ReLU und einer Zwischenschicht der Breite $d_{\textrm{ff}} = 4d$ :cite:`Vaswani.Shazeer.Parmar.ea.2017`. GPT-2 und BERT ersetzten ReLU durch GELU :cite:`Hendrycks.Gimpel.2016`. Nach dem Vorbild von :citet:`Shazeer.2020` verwenden heute fast alle Sprachmodelle lineare Einheiten mit Gate :cite:`Dauphin.Fan.Auli.ea.2017`, entweder **SwiGLU** (SiLU-Gate) oder **GeGLU** (GELU-Gate). Die Breite der Zwischenschicht beträgt $d_{\textrm{ff}} \approx \frac{8}{3}d$, gerundet auf ein für die Hardware günstiges Vielfaches von 128 oder 256. Damit haben die drei Projektionsmatrizen ($W_{\textrm{gate}}, W_{\textrm{up}}, W_{\textrm{down}}$) zusammen $8d^2$ Parameter und benötigen ebenso viele FLOPs wie ein Feedforward-Netz (FFN) der Breite $4d$ ohne Gate (:numref:`sec_transformer-block`). Bei extrem großen Schichtbreiten berechnen **parallele Transformer-Blöcke** (PaLM :cite:`chowdhery2022palm`, ViT-22B :cite:`Dehghani.Djolonga.Mustafa.ea.2023`) die Attention und die Ausgabe des FFN parallel aus derselben normalisierten Eingabe ($x + \operatorname{Attn}(\operatorname{Norm}(x)) + \operatorname{FFN}(\operatorname{Norm}(x))$). So lassen sich die Eingabeprojektionen für Attention und FFN zu einer einzigen großen Matrizenmultiplikation zusammenfassen.

<!--en:0ca03e62-->
## Inferenz mit langem Kontext und Kompression des KV-Caches

<!--en:f2465935-->
Die Speicherbandbreite begrenzt die Geschwindigkeit der Dekodierung (:numref:`subsec_kv-cache-prefill-decode`). Für einen effizienten Inferenzbetrieb kommt es daher darauf an, den Speicherverkehr beim Zugriff auf den KV-Cache zu verringern.

<!--en:bfef77ed-->
- **Grouped-Query-Attention und Multi-Head Latent Attention.** Bei Multi-Query-Attention (MQA) :cite:`Shazeer.2019` und Grouped-Query-Attention (GQA) :cite:`Ainslie.Lee-Thorp.Jong.ea.2023` teilen sich die Query-Köpfe einer Gruppe jeweils einen Schlüsselkopf und einen Wertkopf (:numref:`sec_kv-cache-reduction`). **Multi-Head Latent Attention (MLA)** kam erstmals in DeepSeek-V2 :cite:`DeepSeek-AI.2024` zum Einsatz. Auch DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` verwendet sie. MLA verallgemeinert die KV-Kompression mit niedrigem Rang (:numref:`subsec_kv-cache-low-rank`). Sie projiziert Schlüssel und Werte gemeinsam auf einen einzigen niedrigdimensionalen latenten Vektor $c_t^{\textrm{KV}} \in \mathbb{R}^{d_c}$ ($d_c \ll H d_h$) und speichert ihn im Cache. Hinzu kommt ein entkoppelter, niedrigdimensionaler RoPE-Schlüsselvektor. Damit bleiben rotierende Positionskodierungen mit dem Zusammenfassen der Gewichtsmatrizen bei der Dekodierung vereinbar.

<!--en:0dafd3cd-->
- **Gleitende Fenster, Attention-Sinks und hybride Schichtfolgen.** Neben einem gleitenden Fenster mit den jüngsten Token behält **StreamingLLM** die ersten Token des Prompts bei (:cite:`Xiao.Tian.Chen.ea.2024`, :numref:`subsec_kv-cache-sinks`). So bleibt die Qualität der generierten Sequenz auch bei fortlaufender Inferenz erhalten. Der Grund liegt im Nenner der Softmax-Funktion, über den überschüssige Wahrscheinlichkeitsmasse auf die ersten Token, die sogenannten Sink-Token, entfällt. Mehrere neuere Architekturen kombinieren lokale Attention-Schichten mit gleitendem Fenster und Schichten mit vollständiger globaler kausaler Attention. Bei Gemma 3 kommen fünf lokale Schichten auf eine globale, bei gpt-oss wechseln sich beide ab. Andere Architekturen kombinieren Attention-Schichten mit Zustandsraumschichten oder rekurrenten Schichten, deren Laufzeit linear mit der Sequenzlänge wächst (:numref:`chap_modern_rnn`).

<!--en:776959da-->
- **Spekulative Dekodierung.** Die parallele Prüfung von $m$ möglichen Token beansprucht fast genauso viel Speicherbandbreite wie die Dekodierung eines einzigen Tokens (:numref:`subsec_kv-cache-prefill-decode`). Bei der **spekulativen Dekodierung** (engl. speculative decoding) :cite:`Leviathan.Kalman.Matias.2023,Chen.Borgeaud.Irving.ea.2023` schlägt daher ein schnelles Entwurfsmodell $m$ Token vor. Alternativ übernehmen zusätzliche Köpfe am Zielmodell die Vorhersage mehrerer Token. Das Zielmodell prüft die Vorschläge in einem einzigen Vorwärtsdurchlauf. Eine Regel für das Rejection-Sampling erhält dabei die Zielverteilung exakt.

<!--en:777177e0-->
## Encoder, Cross-Attention und Basisnetze für die Bildverarbeitung

<!--en:cdf6bdea-->
- **Bidirektionale Encoder und Retrieval.** Bei der freien Textgenerierung überwiegen kausale Decoder. Bidirektionale Encoder (:numref:`sec_transformer`) sind weiterhin die Standardarchitektur für dichte Texteinbettungen, das erneute Ordnen von Suchergebnissen und die Klassifikation einzelner Token. RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` und DeBERTa :cite:`He.Liu.Gao.ea.2021` verfeinern das Pre-Training von BERT mittels maskierter Sprachmodellierung. ModernBERT :cite:`Warner.Chaffin.Clavie.ea.2024` kombiniert Pre-Norm, RoPE, SwiGLU und FlashAttention ohne Padding in einem bidirektionalen Encoder.

<!--en:b9998588-->
- **Vision Transformer und hierarchische Hybridarchitekturen.** Um ViTs mit mäßig großen Datenmengen effektiv zu trainieren, kommen mehrere Verfahren zum Einsatz. Dazu gehören starke Datenaugmentierung und Regularisierung (DeiT :cite:`touvron2021training`), die selbstüberwachte Vorhersage maskierter Patches (MAE :cite:`he2022masked`) oder die Selbstdestillation (DINOv2 :cite:`Oquab.Darcet.Moutakanni.ea.2024`) sowie zweidimensionale Rotary Position Embeddings. Dichte Vorhersageaufgaben wie Objekterkennung und Segmentierung erfordern mehrskalige Merkmalskarten. Für solche Aufgaben beschränken Swin Transformer :cite:`liu2021swin` die Self-Attention auf verschobene lokale 2D-Fenster auf den Stufen einer hierarchischen Pyramide. ConvNeXt :cite:`liu2022convnet` wiederum überträgt Gestaltungsprinzipien des Transformer-Blocks zurück auf rein konvolutionäre Netze mit tiefenweise separierbaren Faltungen.

<!--en:66531f99-->
## Skalierungsgesetze und Entwurf von Mixture-of-Experts-Modellen

<!--en:cfbf9027-->
- **Rechenoptimale Skalierung und Skalierung unter Berücksichtigung der Inferenzkosten.** :citet:`kaplan2020scaling` zeigten erstmals, dass der Kreuzentropieverlust von Sprachmodellen in Abhängigkeit von der Modellgröße $N$, der Größe des Datensatzes $D$ und dem Rechenaufwand $C$ Potenzgesetzen folgt. :citet:`hoffmann2022training` (Chinchilla) zeigten anhand von IsoFLOP-Kurven, wann der Trainingsverlust bei festem Rechenbudget für das Pre-Training minimal ist. Dazu müssen $N$ und $D$ im gleichen Verhältnis wachsen ($D \approx 20 N$). Bei modernen Modellen mit offenen Gewichten wie Llama 3 8B (trainiert mit 15 Billionen Token) oder Qwen3 trainiert man kleinere Modelle bewusst mit weit mehr Token, als das Chinchilla-Verhältnis vorgibt (:numref:`sec_scaling-laws`). Damit sollen die Einsparungen bei den Inferenzkosten über die gesamte Einsatzdauer den zusätzlichen Trainingsrechenaufwand ausgleichen :cite:`Sardana.Portes.Doubov.ea.2024`.

<!--en:bca33215-->
- **Mixture of Experts mit wenigen aktiven Experten pro Token.** Moderne MoE-Modelle bauen auf Top-$K$-Gating mit Rauschen :cite:`Shazeer.Mirhoseini.Maziarz.ea.2017`, GShard :cite:`Lepikhin.Lee.Xu.ea.2021` und Switch Transformer :cite:`fedus2022switch` auf. Sie unterscheiden sich vor allem darin, wie fein sie die Berechnung auf Experten aufteilen (:numref:`sec_moe`). Mixtral 8x7B :cite:`Jiang.Sablayrolles.Roux.ea.2024` leitet jedes Token an 2 von 8 Experten voller Breite weiter. Qwen3 :cite:`Yang.Li.Yang.ea.2025` und DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` wählen dagegen 8 von mehr als hundert schmalen Experten aus. DeepSeek-V3 verwendet zusätzlich einen stets aktiven gemeinsamen Experten :cite:`Dai.Deng.Zhao.ea.2024`. Für den Lastausgleich passt der Router hauptsächlich einen eigenen Bias pro Experte dynamisch an :cite:`Wang.Gao.Zeng.ea.2024`. Hinzu kommt nur noch ein sehr kleiner zusätzlicher Strafterm. Mit **Sparse Upcycling** :cite:`Komatsuzaki.Puigcerver.Lee-Thorp.ea.2023` lassen sich auch dichte Checkpoints in MoE-Modelle umwandeln. Dazu initialisiert man jeden Experten mit den Gewichten des trainierten dichten FFN und setzt dann das Pre-Training fort.

<!--en:1a8fd496-->
[Diskussion](https://d2l.discourse.group/t/36853)
