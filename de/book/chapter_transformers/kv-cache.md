<!--en:4ebbe344-->
# Der KV-Cache und seine Verkleinerung

<!--en:a53c9ef7-->
Betrachten wir `GPT.generate` (:numref:`subsec_gpt-sampling`) im Dekodierschritt $t$ genauer. Um Token $t+1$ zu ziehen, führt die Methode einen vollständigen Vorwärtsdurchlauf (engl. forward pass) über das gesamte Präfix $x_{1:t}$ der Länge $t$ aus. Sie liest nur den letzten Logit-Vektor $\mathbf{z}_t \in \mathbb{R}^V$, hängt $x_{t+1}$ an und schickt im Schritt $t+1$ das Präfix $x_{1:t+1}$ erneut durch alle $L$ Blöcke. Die Matrix der kausalen Attention hat untere Dreiecksform. Daher hängt die Repräsentation an Position $s \le t$ in jeder Schicht nur von den Token $1, \dots, s$ ab. Berechnen wir die Schlüssel und Werte an den Positionen $1, \dots, t-1$ im Schritt $t$ erneut, erhalten wir exakt dieselben Tensoren wie im Schritt $t-1$. Über $T$ erzeugte Token hinweg durchlaufen so $O(T^2)$ Token die Gewichtsmatrizen, obwohl $T$ genügen würden.

<!--en:c3194b5d-->
Speichern wir frühere Schlüssel und Werte in einem **KV-Cache** (engl. key–value cache) (:numref:`fig_kv-cache`), entfällt diese Mehrarbeit. Im Schritt $t$ projiziert jede Schicht nur das neue Token $x_t$ auf $(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t)$. Sie schreibt $(\mathbf{k}_t, \mathbf{v}_t)$ auf Platz $t$ eines vorab reservierten Puffers und berechnet die Attention für $\mathbf{q}_t$ mit dem gespeicherten Präfix $(\mathbf{K}_{1:t}, \mathbf{V}_{1:t})$.

<!--en:1062e509-->
Wenn die überflüssigen FLOPs entfallen, wird jedoch ein Engpass der Hardware sichtbar. **Speicherkapazität und Speicherbandbreite** begrenzen die Dekodierung einzelner Token. Für jedes erzeugte Token müssen wir die Modellgewichte und den gesamten KV-Cache aus dem GPU-Speicher (HBM) lesen, um einige wenige Matrix-Vektor-Produkte zu berechnen. Bei langem Kontext oder großen Batches wächst der KV-Cache auf mehrere zehn Gigabyte und begrenzt den Durchsatz bei der Dekodierung. In der ersten Hälfte dieses Abschnitts implementieren wir den Vorwärtsdurchlauf mit Cache. Anhand der Hardware-Roofline untersuchen wir **Prefill**, **Dekodierung** (engl. decode) und **Batching**. In der zweiten Hälfte (:numref:`sec_kv-cache-reduction`) komprimieren wir den KV-Cache entlang seiner drei Dimensionen. Wir verringern die Anzahl der Köpfe (**GQA/MQA**), die Kanalbreite (**Schlüssel-Wert-Projektionen mit niedrigem Rang**) und die Sequenzlänge (**gleitende Fenster mit Attention-Sinks**).

<!--en:f6154d1c-->
## Von der Neuberechnung zum Zwischenspeichern

<!--en:27b3595e-->
![Ein Dekodierschritt an Position $t$. Die naive Generierung berechnet für das gesamte Präfix jede Zeile der Schlüssel-, Wert- und Score-Matrizen neu und verwendet nur die letzte Zeile. Mit einem KV-Cache liest der Dekodierschritt frühere Schlüssel und Werte aus dem Speicher und berechnet jeweils nur eine neue Zeile.](../img/mdl-transformers-kv-cache.svg)

<!--en:91a2b96a-->
### Vorwärtsdurchlauf mit Cache

<!--en:f1eb1560-->
Warum speichern wir in jedem Block die Schlüssel und Werte $(\mathbf{K}_{1:t}, \mathbf{V}_{1:t})$ zwischen und nicht die Abfragen $\mathbf{Q}_{1:t}$ oder die Residualzustände $\mathbf{H}_{1:t}$? Bei kausaler Maskierung brauchen wir die Abfrage $\mathbf{q}_s$ ($s < t$) ausschließlich zur Ausgabe von Token $s+1$. Speichern wir die Schlüssel $\mathbf{k}_s$ *nach der RoPE-Rotation* sowie die Werte $\mathbf{v}_s$ (jeweils mit der Form $(B, H_{\textrm{kv}}, 1, d_h)$), müssen wir frühere Token weder erneut mit $\mathbf{W}_k, \mathbf{W}_v$ projizieren noch erneut mit RoPE rotieren.

<!--en:d56be464-->
`init_cache` reserviert im Voraus die mit Nullen gefüllten Schlüssel- und Wertpuffer `ks, vs` der Form $(B, H_{\textrm{kv}}, T_{\max}, d_h)$ für jeden der $L$ Blöcke. :numref:`fig_mdl-transformers-kv-buffer` zeigt einen solchen Puffer während der Generierung. Die Plätze vor Position $t$ sind belegt. Der aktuelle Schritt beschreibt Platz $t$. Die übrigen Plätze enthalten noch Nullen und sind maskiert.

<!--en:a47f30bc-->
![Cache-Puffer eines Blocks im Schritt an Position $t$ (eine Sequenz, ein Schlüssel-Wert-Kopf). Die Plätze $0, \dots, t-1$ enthalten die Schlüssel und Werte früherer Schritte. Der aktuelle Schritt schreibt $\mathbf{k}_t$ und $\mathbf{v}_t$ auf Platz $t$. Die Plätze nach $t$ enthalten noch Nullen und sind maskiert. Die Abfrage an Position $t$ berücksichtigt daher die Plätze $0, \dots, t$.](../img/mdl-transformers-kv-buffer.svg)

<!--en:718cb812-->
`forward_cached(model, X, ks, vs, t)` verarbeitet einen Slice `X` aus $n$ neuen Token ab dem Positionsversatz `t`. Dazu übergibt die Funktion `offset=t` an RoPE oder greift auf `pos_emb[t:t+n]` zu. Sie schreibt die Schlüssel und Werte der neuen Token nach `[:, :, t:t+n, :]` und berechnet die Attention über `[:, :, :t+n, :]`.

<!--en:b5e297bb-->
Mithilfe von `forward_cached` teilt `generate_cached` die Generierung in zwei Phasen. (1) Der **Prefill** verarbeitet den gesamten Prompt der Länge $T_p$ in einem parallelen Aufruf (`t = 0, n = T_p`). (2) Die **Dekodierung** verarbeitet pro Schritt ein neu gezogenes Token (`n = 1` mit dem Versatz `t = T_p, T_p+1, ...`).

<!--en:3cc1ed95-->
### Prüfung auf exakte Übereinstimmung

<!--en:55fda0b8-->
Bevor wir die Laufzeit von `generate_cached` messen, prüfen wir die Logits. Wenn wir den Cache zunächst mit dem Prompt füllen und dann Token für Token weiterrechnen, müssen sich **exakt dieselben Logits** ergeben (bis auf Gleitkomma-Rundungsfehler der Größenordnung $\sim 10^{-6}$) wie bei einem Aufruf von `model(X)` für die gesamte Sequenz. Wir prüfen dies für `pos='rope'` und `pos='learned'`.

<!--en:a5cbd2bb-->
Wir erzeugen nun je eine Instanz von `GPT` mit 3 Blöcken, einmal mit rotierenden (`pos='rope'`) und einmal mit erlernten absoluten Positionseinbettungen (`pos='learned'`). Mit einem Aufruf füllen wir den Cache mit den ersten $16$ Token einer Sequenz aus $48$ Token. Anschließend verarbeiten wir die übrigen $32$ Positionen Token für Token und messen die maximale absolute Abweichung vom vollständigen Vorwärtsdurchlauf `model(x)` ohne Cache.

<!--en:5be1fa40-->
### Laufzeitmessung beim Übergang von quadratischem zu linearem Aufwand

<!--en:ec74c8d8-->
An einem Modell mit 12 Schichten in der Größe von GPT-2 ($d = 768, H = 12, L = 12$) vergleichen wir zwei Laufzeiten. Zunächst messen wir einen Vorwärtsdurchlauf ohne Cache über $n$ Token (`time_forward`), den die naive Dekodierung bei *jedem* Schritt wiederholt. Dann messen wir einen Dekodierschritt mit Cache bei Kontextlänge $n$ (`time_decode`). Der Vergleich zeigt, warum ein Cache für die Generierung mit langem Kontext unverzichtbar ist.

<!--en:5833f957-->
Für die Zeitmessung definieren wir zwei Hilfsfunktionen mit `d2l.Benchmark`. Die Funktion `time_forward(model, n)` misst einen kompilierten Vorwärtsdurchlauf über das gesamte Präfix aus $n$ Token. Die Funktion `time_decode(model, batch_size, n)` mittelt die gemessene Latenz über $32$ Dekodierschritte mit Cache, die ab der Kontextlänge $n$ jeweils ein Token verarbeiten.

<!--en:e5763145-->
Wir variieren die Kontextlänge $n$ zunächst in Zweierpotenzen ab $256$ und abschließend bis auf $4{,}032$ Token. Ohne Cache wächst die Laufzeit jedes Schritts linear, während die Latenz eines Dekodierschritts mit Cache nahezu konstant bleibt.

<!--en:4b5ab363-->
Tragen wir beide Latenzkurven pro Token doppelt logarithmisch auf, sehen wir, dass die Kosten eines naiven Schritts proportional zur Kontextlänge $n$ wachsen. Die Kurve für den Schritt mit Cache verläuft fast waagerecht. Wir berechnen zwar weiterhin die Scores der neuen Abfrage für alle $n$ gespeicherten Schlüssel, doch der konstante Faktor ist klein.

<!--en:dfc965db-->
Über die gesamte Generierung von $256$ neuen Token nach einem Prompt aus $3{,}584$ Token ergibt sich eine Beschleunigung um mehr als eine Größenordnung, wenn wir `model.generate` durch `model.generate_cached` ersetzen.

<!--en:cd91e1c8-->
## Speicherkapazität und Hardware-Roofline

<!--en:5f90fd33-->
Betrachten wir ein Modell mit $L$ Blöcken, der Batchgröße $B$, der Sequenzlänge $n$ und $H_{\textrm{kv}}$ Schlüssel-Wert-Köpfen der Breite $d_h$. Jede gespeicherte Zahl belegt $b$ Byte ($b = 4$ bei FP32, $2$ bei BF16/FP16, $1$ bei FP8). Jeder Block speichert zwei Tensoren ($\mathbf{K}$ und $\mathbf{V}$) der Form $(B, H_{\textrm{kv}}, n, d_h)$. Der gesamte Speicherbedarf des KV-Caches beträgt daher

<!--en:f6881c86-->
Wir werten `kv_cache_bytes` für unser FP32-Modell mit 124 Millionen Parametern bei $n = 4{,}096$ Token aus. Die reservierten Cache-Puffer belegen genau den nach :eqref:`eq_kv-cache-bytes` erwarteten Speicher ($288\textrm{ MiB}$). Das ist bereits mehr als die Hälfte der $471\textrm{ MiB}$, die die Modellgewichte belegen.

<!--en:a97d9621-->
### Prefill und Dekodierung im Roofline-Modell

<!--en:a6320d44-->
Warum lag die Laufzeit eines Prefill-Aufrufs mit 2048 Token etwa in derselben Größenordnung wie die von rund fünfzehn Dekodierschritten mit je einem Token? Das **Roofline-Modell** (:numref:`fig_kv-cache-roofline`) erklärt den Unterschied anhand der **arithmetischen Intensität**. Sie gibt an, wie viele FLOPs pro Byte ausgeführt werden, das zwischen HBM und Recheneinheiten übertragen wird.

<!--en:765b16fe-->
- **Prefill ($n = 2048$ Token parallel, rechenbegrenzt).** Wir lesen jede Gewichtsmatrix $\mathbf{W}$ einmal aus dem HBM und multiplizieren sie mit $B n$ Tokenvektoren ($2 B n N_{\textrm{params}}$ FLOPs für $b_w N_{\textrm{params}}$ Byte an Gewichten). Die arithmetische Intensität von $\sim 2 B n / b_w \gg 80\textrm{ FLOPs/byte}$ liegt weit über dem Knickpunkt (engl. ridge point) der GPU. Die Tensorkerne sind damit ausgelastet.

<!--en:40694422-->
- **Dekodierung einer einzelnen Sequenz ($B = 1, n = 1$ Token pro Schritt, durch die Speicherbandbreite begrenzt).** Jeder Schritt benötigt nur $2 N_{\textrm{params}}$ FLOPs für Matrix-Vektor-Produkte. Dafür liest er **alle** $b_w N_{\textrm{params}}$ Byte der Gewichte sowie $M_{\textrm{KV}}(n, 1)$ Byte aus dem Cache aus dem HBM. Die arithmetische Intensität beträgt $\le 2 / b_w \approx 0.5\textrm{ FLOPs/byte}$ (in FP32, in BF16 sind es $1.0\textrm{ FLOP/byte}$). Der erreichbare Durchsatz liegt damit um mehr als den Faktor $100\times$ unter der maximalen Rechenleistung.

<!--en:4c39b888-->
![FP32-Roofline einer RTX 4090. Der erreichbare Durchsatz ist das Minimum aus der maximalen Rechenleistung von etwa 83 TFLOP/s und dem Produkt aus arithmetischer Intensität und Speicherbandbreite (etwa 1 TB/s). Beide Grenzen treffen sich am Knickpunkt bei etwa 82 FLOPs pro Byte. Ein Dekodierschritt eines Modells in der Größe von GPT-2 erreicht bei 2048 Token Kontext nur 0.38 FLOPs pro Byte und liegt weit unten auf dem ansteigenden Ast der Roofline. Ein Prefill mit 2048 Token liegt mit etwa 780 FLOPs pro Byte unter dem waagerechten Dach, wenn man nur Gewichte und Cache zählt.](../img/mdl-transformers-roofline.svg)

<!--en:dcb9f087-->
Wir vergleichen die gemessenen Tokendurchsätze und arithmetischen Intensitäten mit einer verfügbaren HBM-Bandbreite von $1\textrm{ TB/s}$. Damit wird deutlich, warum die Dekodierung einer einzelnen Sequenz fast die Grenze der Speicherbandbreite erreicht, während der Prefill mit $2{,}048$ Token einen um mehr als den Faktor $100\times$ höheren Durchsatz erzielt.

<!--en:c463d52e-->
### Dekodierung im Batch

<!--en:684fbc1b-->
Ein Dekodierschritt für eine einzelne Sequenz ($B = 1$) verbringt seine Zeit damit, die $M_{\textrm{weights}} = b_w N_{\textrm{params}}$ Byte der Modellgewichte aus dem HBM zu lesen. Dekodieren wir einen Batch aus $B$ Sequenzen gleichzeitig, nutzen alle $B$ Token jedes gelesene Gewicht gemeinsam. Allerdings müssen wir $B$ verschiedene KV-Caches lesen ($B \, M_{\textrm{KV}}(n, 1)$ Byte). Bei einer Speicherbandbreite von $\mathcal{B}_{\textrm{HBM}}$ (Byte/s) ergibt sich für den bandbreitenbegrenzten Durchsatz der Dekodierung (Token/s)

<!--en:d03e0ba6-->
Bei kleinen Batchgrößen ($B \, M_{\textrm{KV}} \ll M_{\textrm{weights}}$) wächst der Durchsatz nahezu linear mit $B$ („kostenloses Batching“). Wird $B$ oder $n$ groß ($B \, M_{\textrm{KV}} \gg M_{\textrm{weights}}$), dominieren die Lesezugriffe auf den KV-Cache den HBM-Speicherverkehr. Der Durchsatz erreicht die **durch die KV-Bandbreite gesetzte Obergrenze** $\mathcal{B}_{\textrm{HBM}} / M_{\textrm{KV}}(n, 1)$.

<!--en:26ac2e35-->
## Verkleinerung des KV-Caches

<!--en:aee8b43c-->
Wie viele Sequenzen in den Speicher passen (:eqref:`eq_kv-cache-bytes`) und welchen Durchsatz die Dekodierung bei großen Batches höchstens erreicht (:eqref:`eq_kv-cache-batch-throughput`), hängt von der Cache-Größe pro Sequenz ab. Sie beträgt $M_{\textrm{KV}}(n, 1) = 2 b L n H_{\textrm{kv}} d_h$. Moderne Transformer-Architekturen verkleinern daher das Produkt $n \cdot H_{\textrm{kv}} \cdot d_h$ entlang dreier orthogonaler Achsen.

<!--en:e2d3d42e-->
1. **Weniger Schlüssel-Wert-Köpfe ($H_{\textrm{kv}} < H$).** Bei Grouped-Query-Attention (GQA) :cite:`Ainslie.Lee-Thorp.Jong.ea.2023` und Multi-Query-Attention (MQA) :cite:`Shazeer.2019` teilen sich jeweils $G = H / H_{\textrm{kv}}$ Query-Köpfe einen Schlüssel-Wert-Kopf (:numref:`fig_gqa`).

<!--en:7db8b26d-->
2. **Schmalere Repräsentationen pro Token ($2 H_{\textrm{kv}} d_h \to r$).** Schlüssel-Wert-Projektionen mit niedrigem Rang und Multi-Head Latent Attention (MLA) :cite:`Liu.Feng.Xue.ea.2024` fassen die Schlüssel und Werte aller Köpfe in einem niedrigdimensionalen latenten Vektor zusammen (:numref:`subsec_kv-cache-low-rank`).

<!--en:16b2dd8a-->
3. **Kürzerer Kontext im Cache ($n \to w + s$).** Attention mit gleitendem Fenster entfernt Token, die aus dem Fenster der Breite $w$ fallen. Sie behält jedoch $s \ge 1$ Token vom Anfang als **Attention-Sinks** :cite:`Xiao.Tian.Chen.ea.2024` bei. So bleibt der Nenner der Softmax-Funktion stabil (:numref:`subsec_kv-cache-sinks`).

<!--en:42bbda8c-->
![Query-Köpfe und die Schlüssel-Wert-Köpfe, auf die sie zugreifen. Bei Multi-Head-Attention hat jeder Query-Kopf einen eigenen Schlüssel-Wert-Kopf. Bei Grouped-Query-Attention teilt sich jede Gruppe von Query-Köpfen einen Schlüssel-Wert-Kopf. Bei Multi-Query-Attention teilen sich alle einen einzigen. Die Cache-Größe wächst proportional zur Anzahl der Schlüssel-Wert-Köpfe.](../img/mdl-transformers-gqa.svg)

<!--en:3ba8b913-->
### Erste Achse – Grouped-Query-Attention und Weitertraining eines Checkpoints

<!--en:098f4fcb-->
Unsere Klasse `CausalAttention` (:numref:`sec_gpt`) unterstützt bereits `num_kv_heads`. Setzen wir $H_{\textrm{kv}} = H / G$, schrumpft $\mathbf{W}_{qkv}$ und der Speicherbedarf $M_{\textrm{KV}}$ sinkt um den Faktor $G = H / H_{\textrm{kv}}$. Außerdem zeigten :citet:`Ainslie.Lee-Thorp.Jong.ea.2023`, wie sich ein vorhandener Multi-Head-Checkpoint ($H_{\textrm{kv}} = H$) in ein GQA- oder MQA-Modell umwandeln lässt. Dazu bilden wir durch **Mittelwert-Pooling** (engl. mean-pooling) für jede Gruppe von $G$ Köpfen den Mittelwert der Schlüssel- und Wertprojektionen (`pool_heads`). Für das anschließende **Weitertraining** (engl. uptraining) genügt ein kleiner Bruchteil ($\sim 5\text{--}10\%$) des ursprünglichen Trainingsbudgets.

<!--en:e5952525-->
In zwei Durchläufen vergleichen wir das Training von Grund auf mit der Umwandlung eines Checkpoints. Dazu trainieren wir unser bytebasiertes `GPT` mit 6 Blöcken ($H = 8$) für 1000 Schritte mit $H_{\textrm{kv}} \in \{8, 2\}$ (und mit $H_{\textrm{kv}} = 2$ auch für nur 50 Schritte). Den trainierten Checkpoint mit $H_{\textrm{kv}} = 8$ stellen wir auf $H_{\textrm{kv}} = 2$ um. Dabei verwenden wir entweder Mittelwert-Pooling oder behalten nur den ersten Kopf jeder Gruppe. Wir prüfen beide Varianten direkt nach der Umwandlung (0 Schritte) sowie nach 50 und 100 Schritten ($10\%$) Weitertraining.

<!--en:d61a3099-->
Wir trainieren nun unser bytebasiertes `GPT` mit 6 Blöcken auf `WikiText103` mit $H_{\textrm{kv}} \in \{8, 2\}$. Anschließend werten wir `pool_heads` für beide Pooling-Varianten `'mean'` und `'first'` nach $0$, $50$ und $100$ Schritten Weitertraining aus.

<!--en:8a3f1ad0-->
Teilen sich 4 Query-Köpfe die Schlüssel und Werte ($H_{\textrm{kv}} = 2$), schrumpft der KV-Cache um den Faktor $4\times$. Beim Training von Grund auf messen wir keinen Qualitätsverlust (1.34–1.35 gegenüber 1.34 Nat/Byte). Unmittelbar nach dem Zusammenfassen der Köpfe liegt der Verlust bei etwa 2.8 Nat/Byte. Nach 100 Schritten Weitertraining erreicht das Modell mit Mittelwert-Pooling etwa 1.47 und hat damit rund 90% des Rückstands aufgeholt. Mit 50 Trainingsschritten von Grund auf erreichen wir nur etwa 2.7. Behalten wir den ersten Kopf jeder Gruppe, schneidet das Modell nach dem Weitertraining schlechter ab als mit Mittelwert-Pooling.

<!--en:5303cab8-->
### Zweite Achse – Schlüssel-Wert-Projektionen mit niedrigem Rang

<!--en:34edc0b3-->
Können wir die Anzahl der Köpfe $H_{\textrm{kv}}$ beibehalten und stattdessen den $2 d$-dimensionalen Vektor $[\mathbf{k}_t; \mathbf{v}_t] \in \mathbb{R}^{2d}$ aus den aneinandergereihten Schlüsseln und Werten jedes Tokens in Schicht $i$ komprimieren? Dazu beschränken wir ihn auf einen Unterraum vom Rang $r$ ($r \ll 2d$). Wir testen dies direkt am vortrainierten **GPT-2 Small** (`124M` mit $d = 768$ und $2d = 1536$) anhand einer Passage aus The Time Machine mit $T = 1024$ Token.

<!--en:b25809ca-->
Um die Schlüssel und Werte in jeder Schicht abzufangen und zu transformieren, ohne `GPT` umzuschreiben, definieren wir `forward_blocks(model, X, attn_fn)`. Diese Funktion überträgt die Attention in Block $i$ an die Rückruffunktion `attn_fn(q, k, v, i)`. Wir vergleichen die gewöhnliche kausale Attention mit `gpt2(x)` und prüfen, ob beide bis auf $10^{-5}$ übereinstimmen. Dabei bestimmen wir den Referenzwert von `tail_loss` an den Positionen $512 \dots 1023$.

<!--en:4138db14-->
Mit den nächsten 1024 Token des Buchs (`calib`) bestimmen wir für jede Schicht eine empirische PCA/SVD-Basis $\mathbf{U}_i \in \mathbb{R}^{2d \times r}$ für die aneinandergereihten Schlüssel- und Wertzeilen (`fit_bases`). Anschließend projizieren wir die Schlüssel und Werte der untersuchten Passage `x` auf die ersten $r \in \{512, 256, 128\}$ Richtungen (`projected`). Damit messen wir, wie stark sich der KV-Cache von GPT-2 auch ohne erneutes Training komprimieren lässt.

<!--en:5e43661f-->
Jeder Block berechnet seine $2d$ Schlüssel- und Wertkomponenten aus einer einzigen $d$-dimensionalen Eingabe mit einer Matrix und einem Bias. Die Vektoren liegen daher in einem affinen Unterraum mit nur $d = 768$ Dimensionen. Darauf bezieht sich „of 768“ in der Ausgabe. Schon eine nachträgliche lineare Projektion komprimiert die KV-Vektoren von GPT-2 um den Faktor $6\times$ ($r = 256$ von $2d = 1536$) ohne messbare Verschlechterung (3.61 gegenüber 3.58 Nat). Bei $r = 128$ steigt der Verlust dagegen um mehr als ein halbes Nat. Bei der **Multi-Head Latent Attention (MLA)** von DeepSeek-V2/V3 :cite:`Liu.Feng.Xue.ea.2024` wird der Flaschenhals mit niedrigem Rang durchgängig mittrainiert. Dann ersetzt ein einziger 512-dimensionaler latenter Vektor die Schlüssel und Werte aller Köpfe bei einer Kompression um den Faktor $>16\times$.

<!--en:69f7ea47-->
### Dritte Achse – gleitende Fenster und Attention-Sinks

<!--en:215527fd-->
Entlang der dritten Achse beschränken wir die im Cache gespeicherte Sequenzlänge $n$ auf ein gleitendes Fenster mit den $w$ jüngsten Token (:numref:`sec_windowed-linear-attention`). Wenden wir jedoch ein reines gleitendes Fenster ($w = 256$) auf das vortrainierte GPT-2 an, springt die Kreuzentropie für die hinteren Token ($t \ge 512$) einer Passage aus 1024 Token von $\sim 3.6$ Nat auf über $8$ Nat.

<!--en:b58bef88-->
Um die Ursache zu finden, messen wir mit `sink_mass` in jeder der 12 Schichten von GPT-2, welche Attention-Wahrscheinlichkeit die Abfragen an den Positionen $t \ge 64$ im Mittel **Token 0** zuweisen, dem allerersten Token der Sequenz.

<!--en:9cb64c07-->
In den mittleren und oberen Schichten von GPT-2 legen Abfragen **30% bis fast 60% ihres gesamten Attention-Gewichts auf Token 0**, unabhängig vom Wort an Position 0. Die Wahrscheinlichkeiten der Softmax-Funktion müssen sich zu $1$ summieren. Findet sich kein passendes lokales Token, nutzen die Attention-Köpfe daher das erste oder die ersten Token als **Attention-Sink** :cite:`Xiao.Tian.Chen.ea.2024`. Nur diese Positionen sind für *jede* kausale Abfrage sichtbar. Sobald ein reines gleitendes Fenster Position 0 hinter sich lässt ($t \ge w$), entfällt Token 0 im Nenner der Softmax-Funktion. Die Wahrscheinlichkeitsmasse von $30\text{--}60\%$ verteilt sich dann zwangsläufig auf die übrigen Token im Fenster und bringt den Residualstrom durcheinander.

<!--en:b46ea181-->
Behalten wir nur $s \in \{1, 4\}$ anfängliche Sink-Token fest am Anfang des Caches und speichern daneben die $w = 256$ jüngsten Token (insgesamt $w + s$ Token im Cache), erreicht der Validierungsverlust von GPT-2 fast wieder den Referenzwert bei vollem Kontext.

<!--en:4304831a-->
## Vergleich der Verfahren zur Cache-Verkleinerung

<!--en:434460fc-->
:Verfahren zur Verkleinerung des KV-Caches anhand der drei Faktoren in $M_{\textrm{KV}} = 2 b B L n H_{\textrm{kv}} d_h$.

<!--en:7fa37808-->
| Verfahren | Verkleinerter Faktor | Gespeicherte Zahlen pro Sequenz | Training erforderlich? |

<!--en:726b8869-->
| **Gemeinsame KV-Köpfe (GQA, MQA)** :cite:`Shazeer.2019,Ainslie.Lee-Thorp.Jong.ea.2023` | Kopfanzahl $H_{\textrm{kv}} < H$ | $2 L n H_{\textrm{kv}} d_h$ | Ja, Training von Grund auf oder Weitertraining ($\sim 5\text{--}10\%$) |

<!--en:68887846-->
| **Latenter Vektor mit niedrigem Rang (MLA)** :cite:`Liu.Feng.Xue.ea.2024` | Breite $2 H_{\textrm{kv}} d_h \to r$ | $\approx L n r$ (plus ein schmaler Schlüssel für die Rotation) | Ja für MLA. Eine nachträglich bestimmte SVD-Basis genügt bis zum Faktor $\sim 6\times$ |

<!--en:54c1e51d-->
| **Gleitendes Fenster mit Sinks** :cite:`Xiao.Tian.Chen.ea.2024` | Länge $n \to w + s$ | $2 L (w + s) H_{\textrm{kv}} d_h$ | Bei $s \ge 1$ direkt auf vortrainierte Modelle anwendbar |

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:6614a6e0-->
Der **KV-Cache** speichert die Schlüssel nach der RoPE-Rotation und die Werte früherer Token. So projiziert jeder Dekodierschritt nur das neue Token (`n = 1`). Bei der Generierung von $T$ Token sinkt der Projektionsaufwand von $O(T^2)$ auf $O(T)$ Tokenauswertungen und der Aufwand für die Attention-Scores von $O(T^3)$ auf $O(T^2)$. Die Logits bleiben identisch. Auf der Hardware-Roofline (:numref:`fig_kv-cache-roofline`) lassen sich beim Inferenzbetrieb zwei Phasen unterscheiden. Der **Prefill** ist rechenbegrenzt ($n = T_p$ Prompt-Token parallel bei hoher arithmetischer Intensität). Die **Dekodierung** ist durch die Speicherbandbreite begrenzt ($n = 1$ Token pro Schritt bei geringer arithmetischer Intensität). Bei der **Dekodierung im Batch** verteilen sich die Lesezugriffe auf die Gewichte auf $B$ gleichzeitig verarbeitete Sequenzen (:eqref:`eq_kv-cache-batch-throughput`). Der Durchsatz steigt linear mit $B$, bis die Zugriffe auf den KV-Cache den Speicherverkehr dominieren und der Durchsatz seinen Grenzwert $\mathcal{B}_{\textrm{HBM}} / M_{\textrm{KV}}(n, 1)$ erreicht. Damit mehr Sequenzen in den Speicher passen und der Durchsatz bei großen Batches steigt, verkleinern moderne Architekturen $M_{\textrm{KV}} = 2 b L n H_{\textrm{kv}} d_h$ (:eqref:`eq_kv-cache-bytes`) entlang dreier orthogonaler Achsen (:numref:`tab_kv-cache-reduction`). Sie verringern die Anzahl der Köpfe (**GQA/MQA**), die Kanalbreite (**KV-Projektionen mit niedrigem Rang / MLA**) und die Sequenzlänge (**gleitende Fenster mit Attention-Sinks**).

<!--en:2294243a-->
## Übungen

<!--en:3ef60c5b-->
1. **Cache-Speicher und maximale Batchgröße eines 70B-Modells.** Llama 3.1 70B hat $L = 80$ Schichten mit versteckter Dimension $d = 8192$, $H = 64$ Query-Köpfe ($d_h = 128$) und $H_{\textrm{kv}} = 8$ Schlüssel-Wert-Köpfe, deren Schlüssel und Werte in BF16 gespeichert werden ($b = 2$ Byte).
   1. Berechnen Sie mit :eqref:`eq_kv-cache-bytes` die Größe des KV-Caches $M_{\textrm{KV}}(n, 1)$ einer einzelnen Sequenz bei voller Kontextlänge $n = 131{,}072$ ($128\textrm{K}$). Vergleichen Sie das Ergebnis mit dem Speicherbedarf bei gewöhnlicher MHA ($H_{\textrm{kv}} = 64$).
   2. Ein Rechenknoten mit 8 GPUs verfügt insgesamt über $8 \times 80 = 640\textrm{ GB}$ HBM. Davon belegen die BF16-Modellgewichte $140\textrm{ GB}$. Welche maximale Batchgröße $B$ für Sequenzen mit je $128\textrm{K}$ Token passt in die verbleibenden $500\textrm{ GB}$ bei GQA ($H_{\textrm{kv}} = 8$) beziehungsweise bei MHA ($H_{\textrm{kv}} = 64$)?
   :exlabel:`ex_kv-cache-70b`

<!--en:b7661909-->
2. **Dekodierdurchsatz im Batch und Obergrenze durch die KV-Bandbreite.** Betreiben Sie das 70B-Modell aus :exref:`ex_kv-cache-70b` bei einer Kontextlänge von $n = 32{,}768$ auf einem System mit einer gesamten HBM-Bandbreite von $\mathcal{B}_{\textrm{HBM}} = 24\textrm{ TB/s}$ und $M_{\textrm{weights}} = 140\textrm{ GB}$.
   1. Berechnen Sie die durch die Bandbreite gesetzte Obergrenze :eqref:`eq_kv-cache-batch-throughput` für die Batchgrößen $B \in \{1, 8, 32, 128, 512\}$ sowohl bei GQA ($H_{\textrm{kv}} = 8$) als auch bei MHA ($H_{\textrm{kv}} = 64$).
   2. Bei welcher Batchgröße $B_{1/2}$ erreicht die jeweilige Konfiguration die Hälfte ihres asymptotischen Grenzwerts $\lim_{B \to \infty} \textrm{Throughput}(B, n) = \mathcal{B}_{\textrm{HBM}} / M_{\textrm{KV}}(n, 1)$?
   :exlabel:`ex_kv-cache-70b-batch`

<!--en:6df49ae2-->
3. **Gemeinsames Präfix für $K$ Fortsetzungen.** Angenommen, $K = 16$ Sequenzen in einem Batch teilen sich einen System-Prompt aus $n_{\textrm{shared}} = 4096$ Token und jede Sequenz hat $n_{\textrm{unique}} = 256$ eigene Token erzeugt. Passen Sie den Nenner von :eqref:`eq_kv-cache-batch-throughput` für den Fall an, dass der KV-Cache des gemeinsamen Präfixes pro Schritt nur **einmal** für alle $K$ Sequenzen gespeichert und gelesen wird. Berechnen Sie den daraus folgenden Durchsatzgewinn bei $K = 16$ für das 70B-Modell mit GQA.
   :exlabel:`ex_kv-cache-shared-reads`

<!--en:1a8799aa-->
4. **INT8-/FP8-Quantisierung des KV-Caches.** Speichern Sie in `forward_blocks` die Schlüssel und Werte von GPT-2 mit symmetrischer 8-Bit-Ganzzahlquantisierung pro Token. Berechnen Sie für jeden Tokenvektor $\mathbf{u} \in \mathbb{R}^d$ den Skalierungsfaktor $s = \max_j |u_j| / 127$, quantisieren Sie mit $\tilde{\mathbf{u}} = s \cdot \textrm{round}(\mathbf{u} / s)$ und verwenden Sie $\tilde{\mathbf{u}}$ in der Attention. Messen Sie die Änderung von `tail_loss` auf `x` bei Quantisierung der Schlüssel und Werte auf 8 Bit (Cache-Speicherbedarf um den Faktor $2\times$ kleiner) und auf 4 Bit (`max / 7`, Verringerung um den Faktor $4\times$).
   :exlabel:`ex_kv-cache-quantization`

<!--en:0bdb69b2-->
5. **Attention-Sinks und ein virtuelles Sink-Logit ($\textrm{softmax}_1$).** :citet:`Xiao.Tian.Chen.ea.2024` und gpt-oss (:numref:`tab_modern-recipe`) weisen darauf hin, dass ein Kopf einen Attention-Sink braucht, weil sich die Attention-Gewichte bei der gewöhnlichen Softmax-Funktion zu $1$ summieren müssen.
   1. Ersetzen Sie die Softmax-Funktion in `forward_blocks` durch $\textrm{softmax}_1(\mathbf{s})_j = \frac{e^{s_j}}{e^{b} + \sum_{m} e^{s_m}}$. Damit ergänzen Sie den Nenner um den Beitrag eines Sink-Logits $b = 5.0$ ohne zugehöriges Token und fügen keinen Wertvektor hinzu.
   2. Werten Sie das reine gleitende Fenster ($w = 256, s = 0$ beibehaltene Sink-Token) auf GPT-2 mit $b = 5.0$ aus. Wie viel vom Verlustanstieg bei $s = 0$ verhindert das virtuelle Logit im Nenner?
   :exlabel:`ex_kv-cache-sink-logit`

<!--en:08408049-->
6. **Integration der Schlüssel-Wert-Rekonstruktionen mit niedrigem Rang in $\mathbf{W}_q$ und $\mathbf{W}_o$.** Angenommen, Schicht $\ell$ komprimiert jedes Token $j$ zu einem einzigen latenten Vektor $\mathbf{c}_j \in \mathbb{R}^r$ ($r \ll d_h H$). Für Kopf $m$ lassen sich daraus Schlüssel und Wert mit $\mathbf{k}_{j,m} = \mathbf{c}_j \mathbf{B}_{k,m}$ und $\mathbf{v}_{j,m} = \mathbf{c}_j \mathbf{B}_{v,m}$ rekonstruieren ($\mathbf{B}_{k,m}, \mathbf{B}_{v,m} \in \mathbb{R}^{r \times d_h}$).
   1. Zeigen Sie, dass $\mathbf{q}_{t,m}\mathbf{k}_{j,m}^\top = (\mathbf{x}_t \mathbf{W}_{q,m} \mathbf{B}_{k,m}^\top)\mathbf{c}_j^\top$ und $\bigl(\sum_j \alpha_{t,j}^{(m)} \mathbf{v}_{j,m}\bigr)\mathbf{W}_{o,m} = \bigl(\sum_j \alpha_{t,j}^{(m)} \mathbf{c}_j\bigr)(\mathbf{B}_{v,m}\mathbf{W}_{o,m})$ gelten.
   2. Erklären Sie, warum wir bei der Dekodierung $\mathbf{B}_{k,m}^\top$ in $\mathbf{W}_{q,m}$ und $\mathbf{B}_{v,m}$ in $\mathbf{W}_{o,m}$ integrieren können und dadurch die vollständigen Schlüssel $\mathbf{k}_{j,m}$ und Werte $\mathbf{v}_{j,m}$ nie im HBM ablegen müssen. Erklären Sie auch, warum DeepSeek-V2/V3 RoPE auf einen separaten schmalen, entkoppelten Schlüssel $\mathbf{k}_j^{\textrm{RoPE}}$ anwendet, anstatt $\mathbf{c}_j \mathbf{B}_{k,m}$ zu rotieren.
   :exlabel:`ex_kv-cache-absorbed`

<!--en:a3c66b42-->
7. **Implementierung eines Streaming-Caches mit Ringpuffer fester Größe.** Schreiben Sie eine `StreamingCache`-Variante von `init_cache` und `forward_cached`, die pro Schicht nur $s + w$ Plätze für Token reserviert. Behalten Sie die ersten $s = 4$ Sink-Positionen auf den Plätzen $0 \dots s-1$ bei und überschreiben Sie im Ringpuffer jeweils den Platz $s + ((t - s) \bmod w)$ für $t \ge s$. Prüfen Sie, dass die Variante 2048 Token erzeugt und dabei mit konstantem Cache-Speicher der Größenordnung $O(s + w)$ auskommt. Das Modell `gpt2` hat nur 1024 erlernte Positionen. Verwenden Sie daher entweder ein Modell mit rotierenden Positionseinbettungen (`pos='rope'`), dessen Positionsindex unbegrenzt wachsen kann, oder weisen Sie jedem Token von `gpt2` die Position seines Cache-Platzes zu, wie es :citet:`Xiao.Tian.Chen.ea.2024` tun.
   :exlabel:`ex_kv-cache-streaming`

<!--en:9f984824-->
[Diskussion](https://d2l.discourse.group/t/36776)

<!--en:003eaccb-->
KV-Cache, Hardware-Roofline und Cache-Verkleinerung<br>**KV-Cache · Prefill und Dekodierung · arithmetische Intensität · Durchsatz im Batch · GQA und Weitertraining · MLA mit niedrigem Rang · Attention-Sinks**

<!--en:e58995e3-->
::: {.slide title="Naive Generierung verursacht 250× so viel Arbeit"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:b40bd735-->
- **Naive Generierung mit `GPT.generate` (Aufwand $O(T^2)$).** Für jedes erzeugte Token durchläuft das Modell erneut das gesamte Präfix der Länge $3{,}584, 3{,}585, \dots, 3{,}839$ ($950{,}144$ Tokenauswertungen).

<!--en:8f8de9a3-->
- **Warum die Neuberechnung überflüssig ist.** Bei einer kausalen Maske bleibt $(\mathbf{k}_s, \mathbf{v}_s)$ an Position $s < t$ unverändert, wenn Token $t$ hinzukommt.

<!--en:8f1123af-->
- **Generierung mit Cache (Aufwand $O(T)$).** Das Modell verarbeitet den Prompt und jedes neue Token genau einmal ($3{,}840$ Auswertungen, **um den Faktor $250\times$ weniger Arbeit**).

<!--en:0e978d61-->
::: {.slide title="Kausalität hält frühere Schlüssel und Werte konstant"}

<!--en:e49b26ee-->
[Prinzip des KV-Caches]{.kicker}

<!--en:af8f48dd-->
- **Was jeder Block speichert.** Die Abfrage $\mathbf{q}_s$ ($s < t$) wird nur zur Ausgabe von Token $s+1$ benötigt. Spätere Schritte lesen ausschließlich die Schlüssel $\mathbf{K}_{1:t}$ nach der RoPE-Rotation und die Werte $\mathbf{V}_{1:t}$.

<!--en:845101a4-->
- **Phase 1 – Prefill ($n = T_p$ Token).** Ein paralleler Vorwärtsdurchlauf verarbeitet den gesamten Prompt (`offset = 0`) und füllt die Plätze $0, \dots, T_p - 1$ im Cache.

<!--en:5dce16dd-->
- **Phase 2 – Dekodierung ($n = 1$ Token/Schritt).** Jeder Block projiziert nur das neueste Token $x_t$ auf $(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t)$, beschreibt Platz $t$ und berechnet die Attention für $\mathbf{q}_t$ über die gespeicherten $(\mathbf{K}_{0:t}, \mathbf{V}_{0:t})$.

<!--en:d9855c47-->
::: {.slide title="Vorab reservierte statische Puffer und Positionsversatz"}

<!--en:a67ece75-->
[Pufferaufbau]{.kicker}

<!--en:5675e9a5-->
- **Puffer mit fester Form (`init_cache`).** Vor der Generierung reserviert die Funktion für jeden Block mit Nullen gefüllte Tensoren `ks[i], vs[i]` der Form $(B, H_{\text{kv}}, T_{\max}, d_h)$.

<!--en:6a6ecab7-->
- **Positionsversatz `t` und boolesche Maske.** Abfragen und Schlüssel rotieren entsprechend dem Versatz $t$. Unbeschriebene Plätze $> t$ bleiben maskiert (`arange(T_max) <= pos[:, None]`). Dank der festen Tensorformen kompiliert der Compiler den Dekodierschritt nur **einmal**.

<!--en:796ed340-->
::: {.slide title="Dekodierschritt mit Cache – einen Platz beschreiben, das Präfix lesen" only="pytorch"}

<!--en:ce0f4436-->
[Implementierung]{.kicker}

<!--en:d1dc40ac-->
In `forward_cached` projiziert jeder Block mit Vornormierung nur die neuen Token `X`. Er schreibt `k, v` direkt an die Positionen `pos = t + arange(T)` und berechnet die Attention über den Puffer.

<!--en:5c86a123-->
Markieren wir die Cache-Tensoren mit `torch._dynamo.mark_static_address(u)`, kann `torch.compile` mit `triton.cudagraphs=True` den gesamten Dekodierschritt wiederholt ausführen. Für die Aufrufsteuerung durch Python fällt dabei fast kein Aufwand an.

<!--en:cfef7cd4-->
::: {.slide title="Dekodierschritt mit Cache – einen Platz beschreiben, das Präfix lesen" only="jax"}

<!--en:ce0f4436-2-->
[Implementierung]{.kicker}

<!--en:7b6f6602-->
In `forward_cached` projiziert jeder Block mit Vornormierung nur die neuen Token `X`. Er aktualisiert den Slice ab `(0, t, 0, 0)` in `ks[i], vs[i]` und berechnet die Attention über den Puffer.

<!--en:311d0222-->
Übergeben wir `donate_argnames=('ks', 'vs')` an `@jax.jit` für die Schleife `jax.lax.fori_loop`, kann XLA die überlassenen KV-Cache-Puffer direkt ändern, ohne Speicher für Kopien zu reservieren.

<!--en:05debd3e-->
::: {.slide title="Prüfung auf exakt gleiche Logits"}

<!--en:8cd49386-->
[Korrektheitsprüfung]{.kicker}

<!--en:47ed7dfd-->
Der KV-Cache liefert **keine Näherung**. Ein Prefill mit $16$ Token und anschließende Einzelschritte für $32$ weitere Token liefern **dieselben Logits** wie ein Durchlauf ohne Cache über alle $48$ Token (bis auf Rundung, $\sim 10^{-7}$). Das gilt für rotierende wie für erlernte Positionseinbettungen.

<!--en:bd7b7215-->
::: {.slide title="Gemessene Beschleunigung bei 4000 Token um mehr als 20×"}

<!--en:974e8e9e-->
[Laufzeitmessung]{.kicker}

<!--en:e58adc22-->
- **Ohne Cache wächst die Schrittdauer linear mit der Kontextlänge $n$.** Die Neuberechnung des gesamten Präfixes bei einem Modell mit 124 Mio. Parametern dauert $33\text{--}53\text{ ms/token}$ bei $n = 4{,}032$. Eine Fortsetzung um 256 Token braucht damit mehr als $8\text{ seconds}$.

<!--en:1fa95fb7-->
- **Mit Cache bleibt die Schrittdauer nahezu konstant ($\sim 1\text{ ms/token}$).** Die Projektion eines einzelnen Tokens dauert $0.8\text{--}1.3\text{ ms}$ über den gesamten Bereich bis $4{,}096$ Token.

<!--en:ae501d39-->
::: {.slide title="Formel für den Speicherbedarf des KV-Caches"}

<!--en:55c7ed9c-->
[Speicherbedarf]{.kicker}

<!--en:10482bf3-->
- Wir speichern $\mathbf{K}$ und $\mathbf{V}$ für $L$ Blöcke, Batchgröße $B$, Länge $n$ und $H_{\text{kv}}$ Köpfe der Breite $d_h$ mit $b$ Byte pro Element. Dafür benötigen wir
  $$M_{\mathrm{KV}}(n, B) = 2 \, b \, B \, L \, n \, H_{\mathrm{kv}} \, d_h \;\;\text{bytes}$$

<!--en:c93815c9-->
- Bei GPT-2 Small (FP32, $72\text{ KiB/token}$) belegt ein Cache der Länge $n = 4{,}096$ insgesamt $288\text{ MiB}$ bei $B = 1$ und **$4.5\text{ GiB}$ bei $B = 16$** ($10\times$ so viel wie die Gewichte).

<!--en:2c081b7b-->
::: {.slide title="Warum die Speicherbandbreite die Dekodierung einzelner Token begrenzt"}

<!--en:463ae47f-->
[Mindestdauer des Datentransfers]{.kicker}

<!--en:e7761e84-->
- **Rechenzeit und Datentransfer.** Bei GPT-2 Small ($B = 1$) benötigen $2N \approx 2.5 \times 10^8\text{ FLOPs}$ nur $0.003\text{ ms}$ bei $83\text{ TFLOP/s}$. Das Lesen von $498\text{ MB}$ an FP32-Gewichten aus dem HBM mit $1\text{ TB/s}$ dauert dagegen mindestens **$0.50\text{ ms/token}$**.

<!--en:b50dc764-->
- **Gemessene Latenz ($0.8\text{--}1.3\text{ ms/token}$).** Sie liegt nur um den Faktor $1.6\text{--}2\times$ über dem physikalischen Minimum bei gegebener HBM-Bandbreite.

<!--en:efcc2cae-->
::: {.slide title="Prefill und Dekodierung auf der Hardware-Roofline"}

<!--en:eacee862-->
[Roofline-Modell]{.kicker}

<!--en:ef22d533-->
- **Arithmetische Intensität ($I = \text{FLOPs} / \text{Bytes}$).** Das Verhältnis von maximaler Rechenleistung ($83\text{ TFLOP/s}$) zu HBM-Bandbreite ($1\text{ TB/s}$) bestimmt den **Knickpunkt** der Hardware bei $I^* \approx 82\text{ FLOPs/byte}$.

<!--en:09c131c4-->
- **Dekodierung ($n = 1, B = 1$).** Für $2N$ FLOPs liest das Modell $4N + M_{\text{KV}}$ Byte ($I \approx 0.38\text{ FLOPs/byte}$ bei Kontextlänge $2{,}048$). Das liegt um den Faktor $200\times$ unter dem Knickpunkt (**bandbreitenbegrenzt**).

<!--en:e0900634-->
- **Prefill ($n = 2{,}048$ parallel).** Jedes gelesene Gewicht dient zur Verarbeitung von $2{,}048$ Token ($I \approx 780\text{ FLOPs/byte}$). Das liegt weit über dem Knickpunkt (**rechenbegrenzt**).

<!--en:e0184ea1-->
::: {.slide title="Durchsatzmessung für Prefill und Dekodierung"}

<!--en:d66f3092-->
[Überprüfung des Roofline-Modells]{.kicker}

<!--en:a1ad9234-->
- Der Vergleich eines Prefills mit 2048 Token und eines Dekodierschritts mit einem Token bei Kontextlänge $2{,}048$ bestätigt den Faktor $100\times$ zwischen den Durchsätzen im Inferenzbetrieb.

<!--en:0939a94e-->
- Der rechenbegrenzte Prefill bestimmt die **TTFT**, die durch die Speicherbandbreite begrenzte Dekodierung die **ITL**.

<!--en:674445eb-->
::: {.slide title="Dekodierung im Batch und Durchsatzgrenze der KV-Bandbreite"}

<!--en:e5656e41-->
[Gemeinsames Lesen der Gewichte]{.kicker}

<!--en:d490a647-->
Bei der gleichzeitigen Dekodierung von $B$ Sequenzen liest das Modell die Gewichte $M_{\text{weights}}$ pro Schritt nur **einmal** für alle $B$ Token. Es muss jedoch $B$ verschiedene KV-Caches lesen ($B \cdot M_{\text{KV}}(n, 1)$ Byte).

<!--en:c8d0e459-->
- **Kleine Batches ($B \, M_{\text{KV}} \ll M_{\text{weights}}$).** Beim Lesen aus dem HBM überwiegen die Gewichte. Daher wächst der Gesamtdurchsatz nahezu linear mit $B$ („kostenloses Batching“).

<!--en:868a4297-->
- **Batchgröße bei halber Sättigung ($B_{1/2} = M_{\text{weights}} / M_{\text{KV}}(n, 1)$).** Bei dieser Batchgröße belegen alle KV-Caches zusammen so viele Byte wie die Modellgewichte ($B_{1/2} \approx 53$ bei $n = 128$, aber $B_{1/2} \approx 6.6$ bei $n = 1{,}024$).

<!--en:e12c7117-->
::: {.slide title="Gemessener Dekodierdurchsatz in Abhängigkeit von der Batchgröße"}

<!--en:e5656e41-2-->
[Gemeinsames Lesen der Gewichte]{.kicker}

<!--en:aa7aa3d9-->
::: {.slide title="Drei orthogonale Achsen zur Verkleinerung des KV-Caches"}

<!--en:4a87d871-->
[Verfahren zur Cache-Kompression]{.kicker}

<!--en:cba7b250-->
- Die Zahl der Sequenzen, die in den VRAM passen, und die Durchsatzgrenze bei großen Batches $\mathcal{B}_{\text{HBM}} / M_{\text{KV}}(n, 1)$ sind umgekehrt proportional zu $M_{\text{KV}}(n, 1) = 2 b L n H_{\text{kv}} d_h$. Moderne LLMs komprimieren daher alle drei Faktoren.
  1. **Anzahl der Köpfe ($H_{\text{kv}}$).** GQA und MQA ($4\text{--}8\times$).
  2. **Merkmalsbreite ($2 H_{\text{kv}} d_h$).** SVD mit niedrigem Rang oder MLA ($6\text{--}16\times$).
  3. **Sequenzlänge ($n$).** Gleitendes Fenster der Länge $w$ und $s$ Sinks.

<!--en:e87cb65e-->
::: {.slide title="Erste Achse – Grouped-Query-Attention (GQA) und Multi-Query-Attention"}

<!--en:7f410f60-->
[Gemeinsame Schlüssel-Wert-Köpfe]{.kicker}

<!--en:7c95c119-->
- **Multi-Head-Attention (MHA, $H_{\text{kv}} = H$).** Jeder Query-Kopf hat seinen eigenen Schlüssel-Wert-Kopf.

<!--en:c768d884-->
- **Multi-Query-Attention (MQA, $H_{\text{kv}} = 1$, Shazeer 2019).** Alle $H$ Query-Köpfe teilen sich einen einzigen Schlüssel-Wert-Kopf. Der Cache schrumpft um den Faktor $H\times$.

<!--en:0cfd992d-->
- **Grouped-Query-Attention (GQA, $1 < H_{\text{kv}} < H$, Ainslie et al. 2023).** Die $H$ Query-Köpfe bilden $H_{\text{kv}}$ Gruppen der Größe $G = H / H_{\text{kv}}$. Der Cache schrumpft um den Faktor $G\times$ bei gleicher Güte wie mit MHA. Llama 3 und Qwen3 verwenden GQA.

<!--en:e1735094-->
::: {.slide title="Umwandlung von MHA-Checkpoints durch GQA-Weitertraining"}

<!--en:933f125e-->
[Weitertraining eines Checkpoints]{.kicker}

<!--en:dc081b09-->
Können wir einen bereits trainierten Multi-Head-Checkpoint ($H_{\text{kv}} = H$) in ein GQA-Modell ($H_{\text{kv}} = H/G$) umwandeln, ohne das Training von vorn zu beginnen?

<!--en:cffe9aaf-->
- **Training von Grund auf.** GQA ($H_{\text{kv}} = 2$, um den Faktor $4\times$ kleinerer Cache) erreicht wie MHA ($H_{\text{kv}} = 8$) einen Verlust von **$1.34\text{ nats/byte}$**.

<!--en:9c3e11f6-->
- **Mittelwert-Pooling und $5\text{--}10\%$ Weitertraining.** Direkt nach der Umwandlung liegt der Verlust bei etwa $2.8\text{ nats}$. Nach nur $100$ weiteren Schritten ($10\%$) sinkt er auf **$1.47\text{--}1.48\text{ nats}$**. Das ist deutlich besser, als nur den ersten Kopf zu behalten (`first`) oder 50 Schritte von Grund auf zu trainieren ($2.7\text{ nats}$).

<!--en:a28cf49e-->
::: {.slide title="Zweite Achse – KV-Unterräume mit niedrigem Rang und latente Attention (MLA)"}

<!--en:24d8b5d1-->
[KV-Kompression mit niedrigem Rang]{.kicker}

<!--en:daa261e7-->
- **Nachträgliche SVD-Projektion ($6\times$).** Projizieren wir bei GPT-2 den $1{,}536$-dimensionalen Vektor $[\mathbf{k}_t; \mathbf{v}_t]$ auf die ersten $r = 256$ Singulärvektoren, bleibt die Perplexität nahezu gleich ($37$ gegenüber $36$).

<!--en:5f36d824-->
- **Multi-Head Latent Attention (MLA, DeepSeek-V2/V3).** MLA speichert einen gemeinsamen latenten Vektor $\mathbf{c}_t \in \mathbb{R}^{r}$ ($r = 512$) und integriert die Projektionen in den höherdimensionalen Raum bei der Dekodierung in $\mathbf{W}_q$ und $\mathbf{W}_o$.

<!--en:d090ebf8-->
::: {.slide title="Dritte Achse – gleitende Fenster versagen ohne Attention-Sinks"}

<!--en:b249eb03-->
[Das Phänomen der Attention-Sinks]{.kicker}

<!--en:6a3f98f2-->
- Bei vortrainiertem GPT-2 lässt ein reines gleitendes Fenster ($w = 256, s = 0$) die Perplexität am Ende der Passage von **$36$ auf $\approx 3{,}840$** springen.

<!--en:7f9ea3c9-->
- **Warum?** Die Softmax-Gewichte müssen sich zu $1$ summieren. Köpfe ohne passenden Schlüssel legen daher $30\%\text{--}57\%$ ihres Gewichts auf **Token 0**. Entfernen wir Token 0, sobald $t \ge w$ gilt, gerät der Residualstrom durcheinander.

<!--en:f8b4fed1-->
::: {.slide title="Sink-Token senken die Perplexität bei gleitendem Fenster"}

<!--en:c1175535-->
[StreamingLLM und Sinks]{.kicker}

<!--en:835ccf30-->
Es genügt, $s \in \{1, 4\}$ anfängliche Token als **Attention-Sinks** dauerhaft auf den Cache-Plätzen $0 \dots s-1$ zu halten, zusätzlich zu den $w = 256$ Plätzen des gleitenden Fensters (insgesamt $w + s$ Token im Cache, Xiao et al. 2024). Damit verschwindet der Einbruch ohne weiteres Training.

<!--en:722b1ecc-->
Ein einziges Sink-Token ($s = 1$) senkt die Perplexität von $\approx 3{,}840$ wieder auf $47$ (mit $s = 4$ auf $43$, gegenüber $36$ bei voller Attention über $1{,}024$ Token). Moderne Architekturen wie `gpt-oss` gehen einen Schritt weiter. Jeder Attention-Kopf erhält einen lernbaren virtuellen Sink-Bias im Nenner der Softmax-Funktion.

<!--en:89811416-->
::: {.slide title="Verständnisfragen zu Roofline und Größe des KV-Caches"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:c216c9f9-->
Lösen Sie diese drei Rechenaufgaben zum Inferenzbetrieb.

<!--en:34afe91d-->
1. **Speicherbedarf des GQA-Caches beim 70B-Modell.** Llama 3.1 70B hat $L = 80$ Schichten, $H = 64$ Query-Köpfe und $H_{\text{kv}} = 8$ KV-Köpfe der Breite $d_h = 128$. Gewichte ($140\text{ GB}$) und Cache ($b = 2\text{ B}$) liegen in BF16 vor. Wie viele Kilobyte pro Token benötigt der KV-Cache? Ab welcher Kontextlänge $n$ belegt der Cache einer einzelnen Sequenz bei GQA bzw. MHA so viel Speicher wie die Gewichte mit $140\text{ GB}$?

<!--en:5b77ce44-->
2. **Batchgröße bei halber Sättigung.** Betrachten Sie GPT-2 Small in FP32 ($M_{\text{weights}} \approx 498\text{ MB}$, $72\text{ KiB/token}$). Bei welcher Batchgröße $B_{1/2}$ und Kontextlänge $n = 4{,}096$ erzeugen KV-Cache und Gewichte gleich viel Speicherverkehr? Welcher asymptotische Grenzwert ergibt sich für den Dekodierdurchsatz bei $1\text{ TB/s}$?

<!--en:579607ca-->
3. **Speicherbedarf von Fenster und Sinks.** Wie viel Speicher benötigt ein Streaming-Cache mit Fensterbreite $w = 256$ und $s = 4$ Sinks auf GPT-2 Small (FP32), wenn das Modell $100{,}000$ Token erzeugt?

<!--en:46176696-->
::: {.slide title="Verständnisfragen zu Roofline und Größe des KV-Caches (Antworten)"}

<!--en:929b7671-2-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:557b98b6-->
**Antworten**

<!--en:e81fb346-->
1. Bei GQA sind es **$2 \times 2 \times 80 \times 8 \times 128 = 327{,}680\text{ B} = 320\text{ KiB/token}$** (um den Faktor $8\times$ weniger als die $2.5\text{ MiB/token}$ bei MHA). Eine Sequenz belegt so viel Speicher wie die Gewichte mit $140\text{ GB}$, wenn sie bei GQA **$n \approx 427\text{K}$ Token** erreicht. Bei MHA genügen **$53\text{K}$ Token**.

<!--en:841e163f-->
2. Bei $n = 4{,}096$ belegt ein Cache $302\text{ MB}$. Damit erhalten wir **$B_{1/2} = 498 / 302 \approx 1.6$ Sequenzen**. Unabhängig von der Batchgröße bleibt der Durchsatz auf höchstens $10^{12} / (3.02 \times 10^8) \approx \mathbf{3{,}310\text{ tokens/s}}$ begrenzt.

<!--en:86430f96-->
3. **Konstant $(256 + 4) \times 72\text{ KiB} = 18.3\text{ MiB}$** bei jeder Sequenzlänge (gegenüber $6.9\text{ GiB}$ für einen Cache mit $100\text{K}$ Token ohne Fenster).

<!--en:8ae0b12a-->
::: {.slide title="Zusammenfassung – Inferenzbetrieb und Kompression des KV-Caches"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:043ae25d-->
| Konzept oder Verfahren | Mathematischer Zusammenhang oder Regel | Bedeutung für System und Architektur |

<!--en:c5e28c2c-->
| **Dekodierung mit Cache** | $(\mathbf{K}_{1:t}, \mathbf{V}_{1:t})$ nach RoPE speichern, Schritte mit $n = 1$ | Die FLOPs für die Gewichtsmatrizen beim Erzeugen von $T$ Token sinken von $O(T^2)$ auf $O(T)$. Die Logits bleiben identisch |

<!--en:e033a9e7-->
| **Speicherbedarf des KV-Caches** | $M_{\text{KV}}(n, B) = 2 b B L n H_{\text{kv}} d_h$ | Wächst linear mit Batchgröße $B$ und Kontextlänge $n$ ($288\text{ MiB}$ bei $n = 4096$ für GPT-2) |

<!--en:87d891f9-->
| **Prefill und Dekodierung** | $I_{\text{prefill}} \approx \frac{2Bn}{b_w} \gg I^*$ und $I_{\text{decode}} \le \frac{2}{b_w} \ll I^*$ | Prefill ist **rechenbegrenzt** ($>10^5\text{ tok/s}$), Dekodierung **bandbreitenbegrenzt** ($\sim 10^3\text{ tok/s}$) |

<!--en:8574c720-->
| **Dekodierung im Batch** | $\text{Throughput}(B,n) \le \frac{B \mathcal{B}_{\text{HBM}}}{M_{\text{weights}} + B M_{\text{KV}}(n,1)}$ | Lineare Beschleunigung für $B \ll B_{1/2}$, Sättigung bei $\mathcal{B}_{\text{HBM}} / M_{\text{KV}}(n,1)$ |

<!--en:755966f5-->
| **GQA / MQA** | Je $G = H / H_{\text{kv}}$ Query-Köpfe teilen sich 1 KV-Kopf | Cache schrumpft um den Faktor $G\times$. MHA-Checkpoints lassen sich mit Mittelwert-Pooling und $5\text{--}10\%$ Weitertraining umwandeln |

<!--en:f4776996-->
| **Niedriger Rang / MLA und Sinks** | Latente Darstellung vom Rang $r$ mit $\mathbf{c}_t \in \mathbb{R}^r$, insgesamt $w + s$ Plätze für Fenster und Sinks | Breite sinkt um den Faktor $6\text{--}16\times$. Bei $s \ge 1$ Sink-Token bleibt die Softmax-Funktion im gleitenden Fenster stabil |
