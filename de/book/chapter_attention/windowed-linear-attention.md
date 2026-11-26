<!--en:2665d492-->
# Attention mit gleitendem Fenster und lineare Attention

<!--en:41521434-->
Fusionierte Kernel (engl. fused kernels, :numref:`sec_attention-at-scale`) ersparen uns die Score-Matrix der Größe $O(n^2)$ als Zwischenergebnis. Die Zahl der Rechenoperationen beträgt jedoch $\Theta(n^2 d)$ FLOPs und wächst ebenso wie der KV-Cache mit der Sequenzlänge $n$. Bei einem Dokument mit 100 000 Token hängen die meisten Token vor allem vom lokalen Kontext ab. Nur gelegentlich müssen wir Informationen aus größerer Entfernung abrufen.

<!--en:db30ddfc-->
Zwei grundlegende Ansätze beim Entwurf der Architektur senken die Rechenkosten der Attention auf $O(n)$:

<!--en:be450e5b-->
1. **Attention mit gleitendem Fenster** (engl. sliding-window attention) :cite:`child2019generating,beltagy2020longformer` beschränkt jede Abfrage auf die $w$ jüngsten Schlüssel. Dadurch sinkt der Rechenaufwand pro Schicht auf $O(nw)$, und der KV-Cache fasst höchstens $w$ Token.

<!--en:44fd3ece-->
2. **Lineare Attention** :cite:`Katharopoulos.Vyas.Pappas.ea.2020` ersetzt den Softmax-Kernel $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ durch einen faktorisierten Kernel $\phi(\mathbf{q})^\top \phi(\mathbf{k})$. So lassen sich Schlüssel und Werte in einer rekurrenten Zustandsmatrix der festen Größe $d_h \times d_v$ aufsummieren.

<!--en:535cd64b-->
Beide Ansätze berechnen eine andere Funktion als die dichte Softmax-Attention. Wir untersuchen, wie ein gleitendes Fenster die Reichweite eines einzelnen Schritts begrenzt und dadurch viele Kacheln bei der Berechnung ausspart. Dann leiten wir die parallele, die rekurrente und die blockweise Form der linearen Attention her. Wir zeigen, warum sich die Verteilung bei faktorisierten Merkmalsabbildungen anders als bei der Softmax-Funktion nicht über die Temperatur zuspitzen lässt.

<!--en:bb9568e1-->
## Attention mit gleitendem Fenster

<!--en:f7acf14d-->
### Die Bandmaske

<!--en:095cd014-->
Bei kausaler *Attention mit gleitendem Fenster* der Breite $w$ berücksichtigt die Abfrage $i$ den Schlüssel $j$ nur dann, wenn $j \le i$ und $i - j < w$ gelten. Diese Verknüpfung von Kausalität und Lokalität ersetzt das untere Dreieck aus $n(n+1)/2$ aktiven Paaren durch ein diagonales Band mit höchstens $nw$ Paaren:

<!--en:f90d5f37-->
### Wachstum des rezeptiven Feldes mit der Tiefe

<!--en:32100ae2-->
Bei aufeinandergestapelten Faltungen wächst das rezeptive Feld mit der Tiefe (:numref:`chap_cnn`). Ähnlich reicht der Informationsfluss bei $L$ übereinanderliegenden Fensterschichten der Breite $w$ pro Schicht $w - 1$ Positionen weiter zurück. Damit umfasst das theoretische rezeptive Feld

<!--en:1878f4e7-->
Token. Mistral 7B v0.1 :cite:`Jiang.Sablayrolles.Mensch.ea.2023` verwendete beispielsweise eine Fensterbreite von $w = 4{,}096$ in $L = 32$ Schichten und erreichte damit bis zu $131{,}041$ Token. Die theoretische Reichweite allein sagt jedoch wenig über den direkten Abruf aus. Volle Self-Attention verbindet zwei beliebige Token in einem Schritt. Bei gestapelten Fenstern braucht es $\lceil \Delta / (w - 1) \rceil$ aufeinanderfolgende Schritte von Schicht zu Schicht, um eine Information über die Entfernung $\Delta$ weiterzugeben. Zudem schwächt jeder Schritt ihren Beitrag ab. :numref:`fig_window-receptive-field` zeigt den kleinsten Fall. Wenn wir innerhalb jedes Fensters gleichmäßig mitteln ($w = 4$, $L = 4$), erreicht ein Token am Rand bei $\Delta = 12$ die Ausgabe über einen einzigen Pfad aus vier Schritten mit Gewicht $(1/4)^4 = 1/256$. In einer einzigen globalen Schicht beträgt sein Gewicht dagegen $1/13$.

<!--en:ed7cdccc-->
![Gestapelte Fenster im Vergleich zu einer globalen Schicht. Links: Bei vier Fensterschichten der Breite $w = 4$ hängt die Ausgabe oben rechts von einem Kegel früherer Positionen ab. Die Eingabe $\Delta = 12$ Positionen weiter links erreicht die Ausgabe über einen einzigen Pfad aus vier Schritten (blau), bei gleichmäßigen Attention-Gewichten mit Gewicht $(1/4)^4 = 1/256$. Rechts: Eine globale Schicht verbindet dieselben Positionen in einem Schritt mit Gewicht $1/13$.](../img/mdl-attention-window-receptive-field.svg)

<!--en:82b6aa27-->
### Globale Schichten

<!--en:a67aff3c-->
Moderne Architekturen für lange Kontexte ermöglichen den Abruf über große Entfernungen in einem Schritt und behalten dabei den Großteil der Speicher- und Rechenersparnis lokaler Fenster bei. Dazu folgt auf jeweils $r$ Fensterschichten eine volle (*globale*) Attention-Schicht. Bei Gemma 3 sind es beispielsweise $r = 5$ Fensterschichten ($w = 1{,}024$) pro globaler Schicht :cite:`Gemma.Team.2025`. Eine Fensterschicht speichert bei der Generierung nur die letzten $\min(n, w)$ Schlüssel und Werte zwischen (:numref:`sec_kv-cache`). Durch den Wechsel der Schichten schrumpft der gesamte KV-Cache auf den Anteil

<!--en:36f18588-->
des Speicherbedarfs einer Architektur mit ausschließlich globalen Schichten ($\approx 27\%$ bei $n = 8{,}192$ und asymptotisch $1/6$ für $r = 5, w = 1{,}024$).

<!--en:93d6aeee-->
### Kacheln in einem fusionierten Kernel

<!--en:6f238f9d-->
Wenden wir `window_mask` innerhalb einer dichten $n \times n$-Matrizenmultiplikation an, berechnen wir weiterhin alle $n^2$ Scores. Um Zeit- und Speicherbedarf auf $O(nw)$ zu begrenzen, darf die Implementierung nur das aktive Band auswerten. Dazu können wir die Abfragen zu Blöcken der Größe $w$ zusammenfassen und pro Block nur den aktuellen und den unmittelbar vorhergehenden Schlüsselblock berücksichtigen (insgesamt $2nw$ Scores):

<!--en:5d535bd9-->
GPU-Kernel wie FlashAttention oder FlexAttention :cite:`Dong.Feng.Guessous.ea.2025` arbeiten mit $b \times b$-Kacheln (bei FlexAttention standardmäßig $b = 128$). Der Kernel überspringt eine Kachel nur dann, wenn die Maske *alle* $b^2$ Einträge ausschließt. Bei $n = 2{,}048$ und $b = w = 128$ verarbeitet er mit kausaler Maske $T(T+1)/2$ der $T^2$ Kacheln ($53.1\%$ für $T = n/b = 16$). Beim gleitenden Fenster verarbeitet er in jeder Kachelzeile nur die Diagonalkachel und, falls vorhanden, ihren linken Nachbarn ($2T - 1 = 31$ Kacheln oder $12.1\%$, siehe :numref:`fig_window-tiles`):

<!--en:3abe58c9-->
![Kacheln, die ein fusionierter Kernel bei $T = 16$ Kachelzeilen verarbeitet. Kräftig eingefärbte Einträge sind zulässige Scores. Der blasse Rest eines Quadrats kennzeichnet eine Kachel, die der Maskenrand schneidet und die der Kernel dennoch verarbeiten muss. Mit der kausalen Maske (links) verarbeitet er $T(T+1)/2 = 136$ Kacheln. Der Maskenrand schneidet nur die $16$ Kacheln auf der Diagonalen. Mit dem Fenster der Breite $w = b$ (rechts) verarbeitet er in jeder Zeile die Diagonalkachel und, falls vorhanden, ihren linken Nachbarn. Das sind $2T - 1 = 31$ Kacheln, die der Maskenrand alle schneidet.](../img/mdl-attention-window-tiles.svg)

<!--en:520c915c-->
Sowohl `windowed_attention` als auch der fusionierte Fensterkernel des Frameworks liefern bis auf float32-Rundungsfehler dieselben Ergebnisse wie die dichte Berechnung mit Maske:

<!--en:c89c427d-->
Als fusionierten Kernel verwenden wir `flex_attention` mit `create_block_mask`.

<!--en:b1dc7d9b-->
Als fusionierten Kernel verwenden wir `dot_product_attention` mit `local_window_size=(w - 1, 0)`.

<!--en:10b9afdb-->
### Fensterkernel und dichte Kernel im Vergleich

<!--en:75e43f39-->
Bei $T = n / 128$ Kachelzeilen beträgt das Verhältnis der aktiven Kacheln bei kausaler Maske zu denen beim gleitenden Fenster ($w = b = 128$) genau $T(T + 1) / (2(2T - 1)) \approx T/4$. Theoretisch sinkt die Zahl der Kacheln damit um den Faktor $64\times$ bei $n = 32{,}768$ ($T =
256$). Wir messen die Laufzeit beider Masken im selben fusionierten bfloat16-Kernel für $n \in [512, 32{,}768]$ ($H = 32$ Köpfe, $d_h = 64$):

<!--en:d5394fd5-->
Oberhalb von $n \approx 2{,}048$ wächst die Laufzeit des dichten kausalen Kernels quadratisch, die des Fensterkernels linear. Der Beschleunigungsfaktor erreicht $> 30\times$ bei $n =
32{,}768$. Das entspricht der Hälfte bis drei Vierteln des anhand der Kachelzahl vorhergesagten Faktors $64\times$. Die Rechnung setzt für jede verarbeitete Kachel denselben Aufwand an. Sie liefert daher nur eine Obergrenze für die Beschleunigung.

<!--en:0005a3d1-->
In FlexAttention schneidet der Maskenrand bei einem Band der Breite $b$ jede verarbeitete Kachel (:numref:`fig_window-tiles`). Jede dieser Kacheln erfordert daher eine elementweise Maskierung. Im Inneren des kausalen Dreiecks entfällt diese Maskierung.

<!--en:e92c71e0-->
Die Dokumentation von cuDNN gibt nicht an, wie der Kernel die Score-Matrix kachelt. Wir nehmen hier $b = 128$ an.

<!--en:245ed4f7-->
## Lineare Attention

<!--en:5dddedf6-->
### Ein Kernel für den Score

<!--en:837adff3-->
Die *lineare Attention* :cite:`Katharopoulos.Vyas.Pappas.ea.2020` beschränkt die Auswahl der bewerteten Paare nicht. Sie ersetzt die exponentielle Ähnlichkeitsfunktion $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ durch einen faktorisierten Kernel $K(\mathbf{q}, \mathbf{k}) = \phi(\mathbf{q})^\top \phi(\mathbf{k})$ mit einer nichtnegativen, elementweise angewandten Merkmalsabbildung (engl. feature map) $\phi: \mathbb{R}^{d_h} \to
\mathbb{R}_+^{d_h}$. Wegen der Assoziativität der Matrizenmultiplikation gilt

<!--en:90221679-->
wodurch wir die $n \times n$-Matrix $\phi(\mathbf{Q})\phi(\mathbf{K})^\top$ vermeiden. Stattdessen fassen wir die Informationen in einer $d_h \times d_v$-Matrix zusammen. Unter einer kausalen Maske ($s \le t$) können wir $\phi(\mathbf{q}_t)^\top$ aus den Summen im Zähler und im Nenner ausklammern. Damit erhalten wir

<!--en:ca86aed8-->
Wie :citet:`Katharopoulos.Vyas.Pappas.ea.2020` wählen wir elementweise $\phi(x) =
\mathrm{elu}(x) + 1 > 0$ (mit einem kleinen $\epsilon = 10^{-6}$ im Nenner zum Schutz vor einem float32-Unterlauf). Die Präfixsummen $(\mathbf{S}_t,
\mathbf{z}_t)$ erfüllen die lineare Rekurrenz

<!--en:715e1249-->
und machen die kausale Attention damit zu einem RNN. Sein versteckter Zustand ist eine $d_h \times d_v$-Matrix, die als assoziativer Speicher dient und sich durch äußere Produkte aktualisiert (ein *Fast Weight Programmer*, :cite:`Schlag.Irie.Schmidhuber.2021`). Wir können :eqref:`eq_linear-attn` entweder parallel über kumulative Summen (`cumsum`) auswerten oder rekurrent Schritt für Schritt mit einem Speicherbedarf von $O(d_h d_v)$:

<!--en:5dbd7fcc-->
### Blockweise Berechnung

<!--en:077f177e-->
Wenn wir beim Training auf GPUs alle $n$ Zwischenmatrizen $\mathbf{S}_t \in \mathbb{R}^{d_h \times d_v}$ in `linear_attention_parallel` explizit speichern, benötigen wir $O(n d_h d_v)$ Speicher (256 MiB pro Kopf bei $n = 16{,}384, d_h = d_v
= 64$). Dagegen führt `linear_attention_recurrent` nacheinander $n$ Vektoroperationen aus und lastet damit die Matrix-Recheneinheiten der GPU kaum aus. Wir kombinieren beide Ansätze, indem wir die Sequenz in Blöcke der Länge $c$ unterteilen :cite:`Yang.Wang.Shen.ea.2024`. Für einen Block, der an Position $t_0$ mit dem bisherigen Zustand $(\mathbf{S}, \mathbf{z})$ beginnt, gilt

<!--en:1ffb2719-->
sodass sowohl die Attention innerhalb eines Blocks ($c \times c$) als auch die Weitergabe des Zustands zwischen den Blöcken dichte Matrizenmultiplikationen nutzen. Das erfordert $O(nc(d_h + d_v) + n d_h d_v)$ FLOPs, und wir speichern nur $n/c$ Blockzustände:

<!--en:b56ce3a6-->
Alle drei Formen liefern bis auf float32-Rundungsfehler identische Ausgaben. Der rekurrente Zustand belegt unabhängig von der Sequenzlänge rund 16 KiB pro Kopf ($64 \times 64 + 64$ Gleitkommazahlen):

<!--en:15de51ec-->
### Kapazität und Schärfe

<!--en:546a03e8-->
Eine beliebige Zahl von Schlüssel-Wert-Paaren in einer $d_h \times d_v$-Zustandsmatrix zu verdichten, bringt zwei grundlegende Einschränkungen mit sich:

<!--en:7f2398c8-->
1. **Begrenzte assoziative Kapazität:** Da $\phi(\mathbf{k}_s) \in
   \mathbb{R}_+^{d_h}$ nichtnegativ ist, kann eine Abfrage einen einzelnen gespeicherten Wert nur dann ohne Störbeiträge anderer Werte isolieren, wenn die Merkmalsvektoren der Schlüssel nahezu orthogonal sind (nahezu disjunkte Träger haben). Damit ist ein exakter Abruf für höchstens $d_h$ Schlüssel möglich.

<!--en:d4398c65-->
2. **Keine Zuspitzung über die Temperatur:** Skalieren wir bei Softmax-Attention eine Abfrage mit $\beta > 1$, gehen die Score-Abstände exponentiell in das Gewichtsverhältnis ein ($e^{\beta \Delta}$). Für $\beta \to \infty$ konzentriert sich die Verteilung auf einen einzigen Eintrag. Bei linearer Attention mit $\phi(x) = \mathrm{elu}(x) + 1$ ergibt die Skalierung von $\mathbf{q}$ mit großem $\beta$ dagegen $\phi(\beta \mathbf{q}) \approx \beta\,
   \mathrm{relu}(\mathbf{q})$. Der Skalenfaktor $\beta$ kürzt sich also in Zähler und Nenner heraus (Aufgabe :exref:`ex_attention-at-scale-feature-map`):

<!--en:29ab3538-->
### Trefferquoten der einzelnen Mechanismen

<!--en:e6db1d18-->
Beide Einschränkungen und die begrenzte Reichweite eines Fensters zeigen sich schon bei der einfachsten Abrufaufgabe. Wir speichern $N$ zufällige Schlüssel-Wert-Paare der Breite $d_h = 64$ und verwenden einen der gespeicherten Schlüssel als Abfrage. Als Treffer zählen wir jede Ausgabe, deren inneres Produkt mit dem richtigen Wert größer ist als mit jedem anderen gespeicherten Wert. Vier Mechanismen beantworten die Abfrage: dichte Softmax-Attention, ein Fenster mit Zugriff nur auf die letzten $w = 16$ Paare, lineare Attention mit der Merkmalsabbildung elu+1 und der *Zustand aus äußeren Produkten* $\mathbf{q}^\top \sum_i \mathbf{k}_i \mathbf{v}_i^\top$. Letzterer entspricht linearer Attention mit der Identität als Merkmalsabbildung und ohne Normierung. So sehen wir, welche Kapazität der $d_h
\times d_v$-Zustand allein bietet:

<!--en:f41101d4-->
Die Softmax-Attention findet das Paar bei jedem $N$. Das Fenster findet es nur, wenn es unter den letzten $w$ Paaren liegt. Die Trefferquote beträgt daher etwa $w/N$. Bei $N = 16$ findet das Fenster jedes Paar, bei $N = 64$ ein Viertel und bei $N = 1{,}024$ nur wenige Prozent. Auffälliger ist das Ergebnis der linearen Attention mit elu+1. Sie scheitert schon bei $N = 16$, obwohl der Zustand viermal so viele Zeilen hat, wie Paare zu speichern sind. Die Gewichte lassen sich nicht zuspitzen. Die Ausgabe liegt daher nahe am Mittelwert aller gespeicherten Werte, und der richtige Wert hebt sich kaum ab.

<!--en:39c07a5c-->
Der Zustand aus äußeren Produkten schneidet deutlich besser ab und zeigt die Kapazitätsgrenze besonders klar. Für die Abfrage $\mathbf{q} = \mathbf{k}_j$ ergibt sich als Ausgabe $\|\mathbf{k}_j\|^2\, \mathbf{v}_j \approx d_h \mathbf{v}_j$ zuzüglich des Störbeitrags $\sum_{i \ne j} (\mathbf{k}_j^\top \mathbf{k}_i)\, \mathbf{v}_i$ der anderen Paare. Jeder Koeffizient im Störterm hat einen Betrag von ungefähr $\sqrt{d_h}$. Damit beträgt das Verhältnis von Rauschen zu Signal etwa $\sqrt{(N - 1)/d_h}$. Bis $N \approx d_h$ ist die Trefferquote nahezu perfekt. Bei $N = 4 d_h$ sinkt sie auf etwa zwei Drittel, bei $N = 16 d_h$ bricht sie ein. Softmax-Attention behält alle $N$ Paare und findet den richtigen Wert bei jedem $N$. Ein Zustand fester Größe kann ihn nur so lange zuverlässig abrufen, wie $N$ in der Größenordnung seiner Breite bleibt.

<!--en:e1c036ad-->
## Wahl des Attention-Mechanismus

<!--en:06b82224-->
* **Exakte fusionierte Attention** ist die Standardwahl für kurze bis mittlere Kontexte. Sie ermöglicht mit $O(n)$ Arbeitsspeicher einen exakten Abruf in einem Schritt.

<!--en:0d13a5f4-->
* **Fensterschichten im Wechsel mit globalen Schichten** sind bei Transformern für lange Kontexte üblich (zum Beispiel bei Gemma 3). Die lokalen Schichten benötigen weniger KV-Cache und weniger FLOPs für die Kacheln. Die regelmäßig eingefügten globalen Schichten verbinden weiterhin beliebige Positionen in einem Schritt.

<!--en:a42e3a7f-->
* **Lineare Attention** benötigt datenabhängige Vergessensgates, damit sie alte Einträge vergessen kann (siehe Aufgabe :exref:`ex_attention-at-scale-decaying-state` und :numref:`chap_modern_rnn`). Für den exakten assoziativen Abruf muss sie außerdem im Wechsel mit vollen Softmax-Schichten eingesetzt werden (z. B. bei MiniMax-01, :cite:`MiniMax.2025`) :cite:`Arora.Eyuboglu.Zhang.ea.2024,Yang.Wang.Shen.ea.2024`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:1bd57f3e-->
Attention mit gleitendem Fenster beschränkt jede Abfrage auf die $w$ jüngsten Schlüssel. Dadurch sinkt der Rechenaufwand auf $O(nw)$ und der Speicherbedarf des Caches auf $O(w)$. Das effektive rezeptive Feld wächst auf $1 + L(w - 1)$ Token über $L$ Schichten. Moderne Modelle für lange Kontexte fügen zwischen diesen lokalen Schichten regelmäßig globale Attention-Schichten ein. So bleibt der Abruf in einem Schritt möglich. Auf GPUs erzielen Kernel diese Beschleunigung, indem sie vollständig maskierte $b \times b$-Kacheln der Score-Matrix überspringen. Am Rand entsteht zusätzlicher Aufwand durch die Maskierung. Die lineare Attention nutzt dagegen eine algebraische Umformung. Sie ersetzt den exponentiellen Score durch einen faktorisierten Ähnlichkeitskernel $\phi(\mathbf{q})^\top \phi(\mathbf{k})$. Damit lässt sich die kausale Attention als Rekurrenz mit Aufwand $O(n)$ und einer Zustandsmatrix der festen Größe $d_h \times d_v$ formulieren. Das Training kann blockweise parallel ablaufen. Bei einem Zustand fester Größe mit nichtnegativen Merkmalen lassen sich die Gewichte durch Skalieren der Abfrage nicht zuspitzen. Auch lassen sich höchstens $O(d_h)$ orthogonale Schlüssel ohne Störbeiträge speichern. In der Praxis nutzen Architekturen mit linearer Attention daher datenabhängige Vergessensgates und kombinieren lineare Attention mit Softmax-Schichten.

<!--en:2294243a-->
## Übungen

<!--en:5486f914-->
1. [code] **Kacheln eines fusionierten Kernels.** Ein fusionierter Kernel arbeitet mit Kacheln aus je $b \times b$ Scores. Er überspringt eine Kachel nur dann, wenn die Maske sie vollständig ausschließt. Sei $T = n/b$.
    1. Zeigen Sie, dass bei einer kausalen Maske $T(T+1)/2$ der $T^2$ Kacheln verbleiben. Zeigen Sie außerdem, dass bei einem Fenster der Breite $w = kb$ für eine ganze Zahl $k < T$ noch $T(k + 1) - k(k + 1)/2$ Kacheln verbleiben. Prüfen Sie beide Anzahlen mit `tiles_visited`. Wie ändert sich die Anzahl, wenn $w$ kein Vielfaches von $b$ ist?
    1. Mehrere Dokumente sind in eine Sequenz von $n = 2048$ Token gepackt. Wie in :numref:`sec_attention-scoring-functions` verbindet die Maske die Kausalitätsbedingung mit der Bedingung „gleiches Dokument“. Vergleichen Sie mithilfe von `tiles_visited` bei $b = 128$ vier Dokumente mit je 512 Token mit vier Dokumenten von 1000, 600, 300 und 148 Token. Erklären Sie, warum mehr Kacheln verarbeitet werden müssen, wenn die Dokumentgrenzen nicht mit den Kachelgrenzen zusammenfallen.

<!--en:371f2083-->
1. [code] **Die Merkmalsabbildung.** Die Merkmalsabbildung $\phi(x) =
   \mathrm{elu}(x) + 1$ in `phi` ist eine von vielen Möglichkeiten. Vergleichen Sie sie mit $\phi(x) = \mathrm{relu}(x)$ und $\phi(x) = \exp(x)$. Wenden Sie dazu beide Funktionen elementweise auf $n = 512$ zufällige Abfragen und Schlüssel der Breite `d_h` an.
    1. Berechnen Sie für jede Abbildung die Gewichtsmatrix der kausalen linearen Attention mit den Einträgen
       $$\frac{\phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_s)}{\sum_{s' \leq t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_{s'})}, \qquad s \leq t,$$
       und bestimmen Sie die maximale absolute Abweichung von der kausalen Softmax-Gewichtsmatrix `d2l.masked_softmax(Q @ K.T / math.sqrt(d_h),
       d2l.causal_mask(n))` sowie die mittlere Entropie der Zeilen der berechneten Matrix.
    1. Unter welcher Bedingung an $\mathbf{q}_t$ und die Schlüssel $\mathbf{k}_1, \ldots, \mathbf{k}_t$ ergibt die ReLU-Abbildung eine undefinierte Zeile? Leiten Sie die Wahrscheinlichkeit dieses Ereignisses für Zeile $t$ her, wenn alle Einträge unabhängig und standardnormalverteilt sind. Berechnen Sie die Wahrscheinlichkeit für $t = 1$ bei Breite $64$ und bei Breite $4$.
    1. In diesem Abschnitt haben wir die Abfragen mit $\beta \in \{1, 4, 16\}$ skaliert und das größte Gewicht betrachtet. Wiederholen Sie dies für die ReLU- und die Exponentialabbildung, messen Sie dabei aber die mittlere Zeilenentropie. Leiten Sie für jede der drei Abbildungen den Grenzwert der normierten Gewichte für $\beta \to \infty$ her. Erklären Sie, warum keine der drei Abbildungen das Gewicht auf einen einzigen Eintrag konzentriert, wenn die Softmax-Funktion dies tut.
   :exlabel:`ex_attention-at-scale-feature-map`

<!--en:ffa857b6-->
1. **Eine exakte Merkmalsabbildung.** ● Auch der Softmax-Kernel $K(\mathbf{q},
   \mathbf{k}) = \exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d_h})$ lässt sich faktorisieren, allerdings nur mit unendlich vielen Merkmalen.
    1. Entwickeln Sie $K$ in seine Taylorreihe. Zeigen Sie, dass der Term vom Grad $p$ gleich $\phi_p(\mathbf{q})^\top \phi_p(\mathbf{k})$ ist, wenn Sie die Merkmalsabbildung $\phi_p$ der Dimension $d_h^p$ aus dem $p$-fachen Tensorprodukt ihres Arguments bilden.
    1. Brechen Sie die Reihe nach Grad $2$ ab. Zeigen Sie, dass der so erhaltene Kernel für jedes Paar aus $\mathbf{q}$ und $\mathbf{k}$ positiv ist. Der Normierungsfaktor in :eqref:`eq_linear-attn` kann also nicht verschwinden. Geben Sie die Dimension der Merkmalsabbildung an, die $\phi_0$, $\phi_1$ und $\phi_2$ aneinanderhängt. Berechnen Sie für $d_h = 64$ und Werte der Breite $64$ die Größe des Zustands $(\mathbf{S}, \mathbf{z})$ bei einfacher Genauigkeit sowie die Länge $n$, bei der der KV-Cache eines Softmax-Kopfes dieselbe Größe erreicht. Based nutzt eine auf diese Weise abgebrochene Reihe als Merkmalsabbildung :cite:`Arora.Eyuboglu.Zhang.ea.2024`.
    1. Setzen Sie voraus, dass die $n \times n$-Matrix mit Einträgen $K(\mathbf{x}_i, \mathbf{x}_j)$ für paarweise verschiedene $\mathbf{x}_1, \ldots, \mathbf{x}_n$ positiv definit ist. Zeigen Sie, dass keine Merkmalsabbildung der Dimension $D < n$ den Kernel $K$ auf diesen Punkten exakt wiedergibt, wenn Abfragen und Schlüssel übereinstimmen.

<!--en:e3fd15be-->
1. [code] **Ein abklingender Zustand.** ● Ergänzen Sie den Zustand in :eqref:`eq_linear-attn-recurrence` um einen skalaren Abklingfaktor $\gamma \in (0, 1)$, sodass gilt:
   $$\mathbf{S}_t = \gamma\, \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\,\mathbf{v}_t^\top, \qquad \mathbf{z}_t = \gamma\, \mathbf{z}_{t-1} + \phi(\mathbf{k}_t).$$
    1. Zeigen Sie, dass die parallele Form den Beitrag von Position $s$ zur Ausgabe an Position $t$ nun mit $\gamma^{t-s}$ gewichtet. Implementieren Sie diese Form, indem Sie die kumulativen Summen in `linear_attention_parallel` mit Potenzen von $\gamma$ umskalieren.
    1. Ergänzen Sie den Abklingfaktor in `linear_attention_recurrent`. Prüfen Sie, ob beide Formen für $n = 512$ und $\gamma = 0.99$ übereinstimmen.
    1. Die Umskalierung multipliziert den Beitrag an Position $s$ vor der kumulativen Summation mit $\gamma^{-s}$. Für welche $n$ führt dies bei einfacher Genauigkeit zu einem Überlauf, wenn $\gamma = 0.9$ beziehungsweise $\gamma = 0.5$ gilt?
    1. Ergänzen Sie den Abklingfaktor in `linear_attention_chunked`. Skalieren Sie jede Position relativ zum Anfang ihres Blocks um und geben Sie den Zustand mit der abklingenden Rekurrenz von Block zu Block weiter. Zeigen Sie, dass kein Faktor größer als $\gamma^{-c}$ auftritt. Führen Sie Ihre Implementierung für $\gamma = 0.5$, $n = 512$ und $c = 64$ aus und vergleichen Sie das Ergebnis mit der rekurrenten Form.
    1. Berechnen Sie den Abstand, bei dem das Gewicht $\gamma^{t-s}$ unter $0.01$ fällt, wenn $\gamma = 0.99$ gilt. In einer Aufgabe soll das Modell ein Token abrufen, das $2000$ Positionen zurückliegt. Dazwischen stehen nur Fülltoken. Erklären Sie, warum kein festes $\gamma$ sowohl dem gesuchten Token als auch den Fülltoken gerecht wird. Was ließe sich mit einem Abklingfaktor erreichen, der für jedes Token aus dessen eigener Eingabe berechnet wird?
   :exlabel:`ex_attention-at-scale-decaying-state`

<!--en:d968aeda-->
[Diskussionen](https://d2l.discourse.group/t/36882)

<!--en:5c10d5ce-->
Attention mit gleitendem Fenster, dünn besetzte und lineare Attention<br>**Masken für gleitende Fenster · rezeptive Felder über mehrere Schritte · Sparsität auf Kachelebene · lineare Attention mit Kernel · rekurrente und blockweise Berechnung**

<!--en:d039d0df-->
::: {.slide title="Warum fusionierte exakte Attention weiterhin O(n²) Zeit benötigt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:d9617a11-->
- FlashAttention spart die Score-Matrix der Größe $O(n^2)$ im HBM ein. Bei $n = 32{,}768$ ($T = 256$ Kachelzeilen, Kantenlänge je Kachel $b = 128$) verarbeitet der Kernel aber weiterhin $T(T+1)/2 = \mathbf{32{,}896}$ Kacheln und hält die Schlüssel und Werte aller $32{,}768$ Token im Cache.

<!--en:2069a223-->
- Um den Rechenaufwand unter $\Theta(n^2 d)$ zu senken und den Speicherbedarf bei der Generierung zu begrenzen, müssen wir **weniger Paare bewerten** (Fenster: $511$ Kacheln) oder **den Ähnlichkeitskernel faktorisieren** (lineare Attention: $d_h \times d_v$-Zustand).

<!--en:c512d92b-->
::: {.slide title="Zwei Architekturen für Attention in O(n)"}

<!--en:5d233c39-->
[Abwägungen beim Entwurf]{.kicker}

<!--en:c4f28ad4-->
- **Ansatz 1 – Attention mit gleitendem Fenster:** Jede Abfrage $i$ greift nur auf die $w$ jüngsten Schlüssel zu ($i - w < j \le i$). Lokal bleibt die Schärfe der Softmax-Funktion bei Kosten von $O(nw)$ exakt erhalten.

<!--en:9d141887-->
- **Ansatz 2 – lineare Attention:** An die Stelle von $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ tritt das Produkt $\phi(\mathbf{q})^\top\phi(\mathbf{k})$ nichtnegativer Merkmalsvektoren. Alle bisherigen Schlüssel und Werte gehen in eine Zustandsmatrix der festen Größe $d_h \times d_v$ ein.

<!--en:4013534e-->
::: {.slide title="Attention mit gleitendem Fenster und kausaler Bandmaske"}

<!--en:b0d9df37-->
[Lokale Attention]{.kicker}

<!--en:2edb9e62-->
Bei einem kausalen gleitenden Fenster der Breite $w$ darf die Abfrage $i$ den Schlüssel $j$ genau dann berücksichtigen, wenn $j \le i$ (Kausalität) und $i - j < w$ (Lokalität) gelten:

<!--en:2e3195dd-->
- **Boolesche Konjunktion:** `window_mask(n, w)` ist die logische UND-Verknüpfung (`&`) von `d2l.causal_mask(n)` und `i[:, None] - i[None, :] < w`. Letzterer begrenzt den Abstand.

<!--en:3838cff9-->
- **Lineare Anzahl von Paaren:** Im kausalen Dreieck bleiben $n(n+1)/2$ Paare aktiv, im gleitenden Band höchstens $nw$ Paare ($\approx 4.2 \times 10^6$ gegenüber $5.4 \times 10^8$ bei $n = 32{,}768, w = 128$).

<!--en:27a1a8a5-->
::: {.slide title="Wachstum des rezeptiven Feldes über mehrere Fensterschichten"}

<!--en:74d84392-->
[Informationsfluss über mehrere Schritte]{.kicker}

<!--en:44a7314d-->
- Bei $L$ Fensterschichten der Breite $w$ reicht das rezeptive Feld pro Schicht um bis zu $w - 1$ Positionen weiter zurück:
  $$\textrm{Receptive Field}(L, w) = 1 + L(w - 1)$$

<!--en:5fa10706-->
- **Theoretische Reichweite und tatsächliche Abschwächung:** Für die Entfernung $\Delta$ braucht es $\lceil \Delta / (w - 1) \rceil$ Schritte von Schicht zu Schicht. Bei gleichmäßigen Fenstergewichten ($w = 4, L = 4$) erreicht ein Token am Rand bei $\Delta = 12$ die Ausgabe mit Gewicht $(1/4)^4 = 1/256$.

<!--en:19fbeaa7-->
::: {.slide title="Fensterschichten im Wechsel mit globalen Attention-Schichten"}

<!--en:e878e77c-->
[Hybride Architekturen für lange Kontexte]{.kicker}

<!--en:ac07ac06-->
- Bei modernen Transformern für lange Kontexte folgt auf jeweils $r$ Fensterschichten $1$ volle globale Schicht:
  $$\frac{\textrm{Hybrid Cache}}{\textrm{All-Global Cache}} = \frac{r \min(n, w) + n}{(r + 1)\,n} \;\xrightarrow{\;n \gg w\;}\; \frac{1}{r + 1}$$

<!--en:5f253d61-->
- **Gemma 3 ($r = 5, w = 1{,}024$):** Der KV-Cache schrumpft auf **$17.3\%$** bei $n = 128\text{K}$. Jede sechste Schicht garantiert einen globalen Attention-Pfad mit nur einem Schritt.

<!--en:8bb505fb-->
::: {.slide title="Blockweise Auswertung des Bandes mit Aufwand O(nw)"}

<!--en:bc1cb38c-->
[Blockweise Implementierung]{.kicker}

<!--en:d4e67c74-->
Auch mit `window_mask(n, w)` berechnet ein dichtes $n \times n$-Matrizenprodukt alle $n^2$ Scores. Um Zeit- und Speicherbedarf auf $O(nw)$ zu begrenzen, zerlegt `windowed_attention` die Abfragen in Blöcke der Größe $w$. Für jeden Block berechnet die Funktion nur die Scores mit dem eigenen und dem vorherigen Schlüsselblock ($2nw$ Scores):

<!--en:0f2239b1-->
::: {.slide title="Wie fusionierte Kernel die Sparsität der Blockmaske nutzen"}

<!--en:507b5f02-->
[Überspringen von Kacheln auf der GPU]{.kicker}

<!--en:acca214e-->
- GPU-Kernel mit Kachelung (FlashAttention, Kernel nach Art von FlexAttention, cuDNN) zerlegen die $n \times n$-Score-Matrix in $b \times b$-Kacheln (typisch $b = 128$). Sie überspringen eine Kachel **nur dann, wenn die Maske alle $b^2$ Einträge ausschließt**.

<!--en:03709e14-->
- Bei $n = 2{,}048$ und $b = w = 128$ ($T = n/b = 16$ Kachelzeilen) gilt:
  - **Kausale Maske:** Der Kernel verarbeitet $T(T+1)/2 = 136$ von $256$ Kacheln (**$53.1\%$**).
  - **Gleitendes Fenster ($w = 128$):** Pro Zeile verarbeitet er nur die Diagonalkachel und, falls vorhanden, ihren linken Nachbarn, insgesamt $2T - 1 = 31$ Kacheln (**$12.1\%$**).

<!--en:07b4d257-->
::: {.slide title="Laufzeitvergleich von Fensterkerneln und dichten kausalen Kerneln" layout="tight"}

<!--en:56da99b6-->
[Gemessene Laufzeit bis 32K]{.kicker}

<!--en:2dc398dd-->
- Wir messen die Laufzeiten der dichten kausalen Attention und der Attention mit gleitendem Fenster ($w = 128$) im selben kompilierten fusionierten `bfloat16`-Kernel ($H = 32, d_h = 64$) von $n = 512$ bis $n = 32{,}768$:

<!--en:d7ba6423-->
- **Lineares und quadratisches Wachstum:** Oberhalb von $n = 2{,}048$ vervierfacht sich die Laufzeit des dichten Kernels bei jeder Verdopplung von $n$. Die Laufzeit des Fensterkernels wächst linear ($2\times$ pro Verdopplung).

<!--en:37e4d4b6-->
- **Beschleunigung um den Faktor $> 30\times$ bei $32\text{K}$:** Das entspricht der Hälfte bis drei Vierteln des Kachelzahlverhältnisses von $64.4\times$. Da die Zählung jeder verarbeiteten Kachel denselben Aufwand zuweist, liefert sie eine Obergrenze für die Beschleunigung.

<!--en:6a9ac9a5-->
::: {.slide title="Lineare Attention durch Faktorisierung des Ähnlichkeitskernels"}

<!--en:eb42d192-->
[Faktorisierte Kernel]{.kicker}

<!--en:10813951-->
- **Lineare Attention** (Katharopoulos et al., 2020) ersetzt $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ durch $K(\mathbf{q}, \mathbf{k}) = \phi(\mathbf{q})^\top \phi(\mathbf{k})$ mit $\phi(x) = \mathrm{elu}(x) + 1 > 0$:
  $$\bigl(\phi(\mathbf{Q})\,\phi(\mathbf{K})^\top\bigr)\,\mathbf{V} \;=\; \phi(\mathbf{Q})\,\bigl(\phi(\mathbf{K})^\top \mathbf{V}\bigr)$$

<!--en:2e7187ec-->
- **Assoziativität nutzen:** Berechnen wir $\phi(\mathbf{K})^\top \mathbf{V} \in \mathbb{R}^{d_h \times d_v}$ zuerst, sinkt der Rechenaufwand von $\Theta(n^2 d_h)$ auf **$\Theta(n d_h d_v)$**, ohne dass wir eine $n \times n$-Matrix bilden.

<!--en:632b15be-->
::: {.slide title="Kausale lineare Attention als RNN mit schnellen Gewichten"}

<!--en:6cf6f5bb-->
[Rekurrenter Matrixzustand]{.kicker}

<!--en:e39a46e9-->
- Unter einer kausalen Maske ($s \le t$) können wir $\phi(\mathbf{q}_t)^\top$ aus den Summen ausklammern. So erhalten wir für die kausale lineare Attention eine exakte Rekurrenz:
  $$\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}{\phi(\mathbf{q}_t)^\top \mathbf{z}_t + \epsilon}, \qquad \mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\,\mathbf{v}_t^\top$$

<!--en:818751a9-->
- **Konstanter Speicherbedarf bei der Generierung:** $(\mathbf{S}_t, \mathbf{z}_t)$ hat die feste Größe $d_h d_v + d_h$ ($16.25\text{ KiB}$ in `float32` bei $d_h = d_v = 64$), unabhängig von der Sequenzlänge $n$.

<!--en:ebf61ddf-->
::: {.slide title="Parallele und rekurrente lineare Attention im Code"}

<!--en:3e003ea9-->
[Zwei gleichwertige Berechnungen]{.kicker}

<!--en:70efb962-->
Die Zustandsrekurrenz $\mathbf{S}_t = \sum_{s \le t} \phi(\mathbf{k}_s)\mathbf{v}_s^\top$ ist eine Präfixsumme. Wir können sie daher parallel mit `cumsum` oder sequenziell Schritt für Schritt auswerten. Die parallele Form für das Training lautet:

<!--en:f1f2d486-->
::: {.slide title="Parallele und rekurrente lineare Attention im Code (Fortsetzung)"}

<!--en:3e003ea9-2-->
[Zwei gleichwertige Berechnungen]{.kicker}

<!--en:d9e27076-->
Die rekurrente Form für die Generierung führt einen Zustand der Größe $(d_h, d_v)$ mit und aktualisiert ihn Token für Token:

<!--en:47f458fd-->
::: {.slide title="Blockweise lineare Attention für effizientes Training auf der GPU"}

<!--en:2be2e366-->
[Blockweise parallele Berechnung]{.kicker}

<!--en:dc37a9db-->
- **Blockweise Zerlegung (Yang et al., 2024):** Wir teilen die Sequenz in $n/c$ Blöcke der Länge $c$ ($c = 64$). Für einen Block mit Anfangsposition $t_0$ und Anfangszustand $(\mathbf{S}, \mathbf{z})$ gilt:
  $$\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S} + \sum_{t_0 \le s \le t} \big(\phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)\big)\,\mathbf{v}_s^\top}{\phi(\mathbf{q}_t)^\top \mathbf{z} + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s) + \epsilon}$$

<!--en:11ae8593-->
- Sowohl der Term zwischen Blöcken ($\mathbf{Q}_p \mathbf{S}$) als auch der Term innerhalb eines Blocks ($\mathrm{tril}(\mathbf{Q}_p \mathbf{K}_p^\top)\mathbf{V}_c$) lassen sich als dichte Matrizenprodukte auf Tensor Cores berechnen. Nur $n/c$ Blockzustände müssen gespeichert werden.

<!--en:78c1bba5-->
::: {.slide title="Implementierung und Prüfung der blockweisen linearen Attention"}

<!--en:7c74fdb6-->
[Blockweiser Code und Nachweis der Gleichwertigkeit]{.kicker}

<!--en:0cab97ec-->
`linear_attention_chunked` stimmt mit beiden anderen Formen bis auf `float32`-Genauigkeit überein ($\sim 10^{-7}$):

<!--en:9b0ee779-->
::: {.slide title="Drei Berechnungsformen der linearen Attention im Vergleich"}

<!--en:eb93d8b3-->
[Vergleich der Implementierungen]{.kicker}

<!--en:86d14685-->
- **Parallel (`cumsum`):** Speichert alle $n$ Zustandsmatrizen ($256\text{ MiB}$ bei $n = 16\text{K}$). Nur für kurze Testsequenzen zur Fehlersuche sinnvoll.

<!--en:f301ba9f-->
- **Rekurrente Schleife:** Der Zustand belegt konstant $16\text{ KiB}$ ($O(1)$ pro Schritt). Ideal für die Generierung Token für Token.

<!--en:ad504271-->
- **Blockweise ($c = 64$):** Dichte Matrizenprodukte auf Tensor Cores und ein um den Faktor $64\times$ geringerer Speicherbedarf für die Zustände ermöglichen schnelles paralleles Training und Prefill.

<!--en:c822fc80-->
::: {.slide title="Kapazität und Schärfe als Grenzen der linearen Attention"}

<!--en:beae4b52-->
[Grundlegende Einschränkungen]{.kicker}

<!--en:a260c2f9-->
Verdichten wir eine unbegrenzt lange Folge von Schlüssel-Wert-Paaren in einer festen $d_h \times d_v$-Matrix mit endlichdimensionaler Merkmalsabbildung $\phi$, ergeben sich zwei grundlegende Einschränkungen für den Abruf:

<!--en:a2443622-->
- **1. Begrenzte Kapazität für orthogonale Schlüssel ($\le d_h$ Schlüssel):** Da $\mathbf{S}_t = \sum_{s \le t} \phi(\mathbf{k}_s)\mathbf{v}_s^\top$ äußere Produkte in $\mathbb{R}^{d_h \times d_v}$ summiert, gelingt der Abruf von $\mathbf{v}_s$ ohne Störbeiträge nur, wenn $\phi(\mathbf{q}_t)$ orthogonal zu $\phi(\mathbf{k}_{s'})$ für alle $s' \neq s$ ist. Das ist für höchstens $d_h$ Schlüssel möglich.

<!--en:da78c2d3-->
- **2. Keine Zuspitzung über die Temperatur:** Bei der Softmax-Funktion gehen die Score-Abstände nach der Skalierung $\mathbf{q} \mapsto \beta \mathbf{q}$ exponentiell in das Gewichtsverhältnis ein ($e^{\beta\Delta} \to \infty$). Im Grenzfall erhält ein einziger Eintrag das gesamte Gewicht. Für $\phi(x) = \mathrm{elu}(x) + 1$ und große $\beta$ gilt dagegen $\phi(\beta \mathbf{q}) \approx \beta\,\mathrm{relu}(\mathbf{q})$, und $\beta$ **kürzt sich in Zähler und Nenner heraus**.

<!--en:aa1a6cf0-->
::: {.slide title="Wahl eines Attention-Mechanismus in der Praxis"}

<!--en:79da16e4-->
[Entscheidungshilfe für die Architektur]{.kicker}

<!--en:611485e1-->
- **Kurze und mittlere Kontexte ($n \le 32\text{K}$):** Exakte fusionierte FlashAttention (Zeitbedarf $\Theta(n^2 d)$, vollständiger Abruf in einem Schritt).

<!--en:9aa817bd-->
- **Lange Kontexte ($n = 128\text{K}\text{--}1\text{M}$):** Fensterschichten im Wechsel mit regelmäßig eingefügten globalen Schichten (z. B. Gemma 3: Das Verhältnis $5{:}1$ senkt den KV-Cache auf $17\%$).

<!--en:10dee32b-->
- **Streaming mit konstanter Zustandsgröße:** Lineare Attention mit Gates oder moderne RNNs mit eingabeabhängigen Vergessensgates $\gamma_t$, kombiniert mit Softmax-Schichten.

<!--en:0ace03ab-->
::: {.slide title="Verständnisfragen zu Kacheln, Caches und linearen Kerneln"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:e2f7e05b-->
1. **Fenster schmaler als eine Kachel:** Ein fusionierter Kernel mit Kachelgröße $b = 128$ verarbeitet bei Sequenzlänge $n = 2{,}048$ ($T = 16$ Kachelzeilen) und einem kausalen Fenster der Breite $w = 128$ insgesamt $2T - 1 = 31$ Kacheln. Wie viele $128 \times 128$-Kacheln verarbeitet er, wenn Sie das Fenster auf $w = 32$ verkleinern?

<!--en:dacbd070-->
2. **Skalierung der ReLU-Merkmalsabbildung:** Sie verwenden $\phi(x) = \mathrm{relu}(x)$ in der linearen Attention und multiplizieren einen Abfragevektor $\mathbf{q}_t$ mit $\beta = 16$. Um wie viel ändern sich die Attention-Gewichte unter der Annahme $\phi(\mathbf{q}_t)^\top \mathbf{z}_t > 0$?

<!--en:d80f2efd-->
3. **Taylor-Merkmalsabbildung zweiten Grades (Based):** Die Entwicklung $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h}) \approx 1 + \frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{d_h}} + \frac{(\mathbf{q}^\top\mathbf{k})^2}{2d_h}$ lässt sich mit einer Merkmalsabbildung der Dimension $D = 1 + d_h + d_h^2$ faktorisieren. Wie groß ist bei $d_h = 64$ der Zustand $\mathbf{S}_t \in \mathbb{R}^{D \times d_v}$ ($d_v = 64$), und ab welcher Sequenzlänge $n$ ist ein Softmax-KV-Cache größer?

<!--en:07dcea37-->
::: {.slide title="Verständnisfragen zu Kacheln, Caches und linearen Kerneln (Antworten)"}

<!--en:929b7671-2-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:557b98b6-->
**Antworten:**

<!--en:bc4d5998-->
1. **Weiterhin $31$ Kacheln.** Der Kernel überspringt nur vollständig leere $128 \times 128$-Kacheln. Jedes Fenster mit $2 \le w \le 128$ ragt in die linke Nachbarkachel hinein. Deshalb verarbeitet der Kernel dieselben $2T - 1 = 31$ Kacheln.

<!--en:31cbd9c5-->
2. **Keine Änderung.** Wegen $\mathrm{relu}(16x) = 16\,\mathrm{relu}(x)$ ist die ReLU positiv homogen vom Grad 1. Der Faktor $16$ kürzt sich in Zähler und Nenner exakt heraus.

<!--en:356a13ac-->
3. **$D = 1 + 64 + 4{,}096 = 4{,}161$**. Damit enthält $\mathbf{S}_t$ genau $4{,}161 \times 64$ Gleitkommazahlen ($1.02\text{ MiB}$ in fp32). Ein Softmax-KV-Cache ($2 \times 64$ Gleitkommazahlen pro Token) erreicht diese Größe bei **$n = 2{,}081$ Token**.

<!--en:7da8e1ec-->
::: {.slide title="Zusammenfassung zu Attention mit gleitendem Fenster und linearer Attention"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:51cbf751-->
| Konzept oder Variante | Mathematische Regel | Zeit- und Speicherbedarf | Abwägung für die Praxis |

<!--en:dd7faa32-->
| **Kausales gleitendes Fenster** | $j \le i$ und $i - j < w$ | $\Theta(nwd)$ FLOPs, Cache der Größe $O(wd)$ | $L$ Schichten erreichen über mehrere Schritte $1 + L(w-1)$ Token |

<!--en:bd018e26-->
| **Fenster- und globale Schichten** | $r$ Fensterschichten je $1$ globale Schicht | Cache-Anteil $\to \dfrac{1}{r+1}$ für $n \gg w$ | Abruf über große Entfernungen in einem Schritt mit einem Bruchteil des KV-Caches |

<!--en:2a9e9fbf-->
| **Blocksparsität auf Kachelebene** | $b \times b$-Kachel genau dann überspringen, wenn alle $b^2$ Einträge `False` sind | Fenster mit $w=b$ verarbeitet $2T-1$ statt $T(T+1)/2$ Kacheln | Beschleunigungsfaktor $> 30\times$ bei $n=32\text{K}$. Fensterbreite $w$ und Grenzen gepackter Dokumente an Kachelgröße $b$ ausrichten |

<!--en:b76af88d-->
| **Lineare Attention** | $\mathbf{o}_t^\top = \dfrac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}{\phi(\mathbf{q}_t)^\top \mathbf{z}_t + \epsilon}$ | $\Theta(n d_h d_v)$ FLOPs, **Zustand der Größe $O(d_h d_v)$** | Durch Assoziativität ($(\phi(\mathbf{Q})\phi(\mathbf{K})^\top)\mathbf{V} = \phi(\mathbf{Q})(\phi(\mathbf{K})^\top\mathbf{V})$) entfällt der Faktor $n^2$ |

<!--en:1ed986e8-->
| **Blockweise lineare Form** | Innerhalb eines Blocks $c \times c$, zwischen Blöcken $\mathbf{Q}_p\mathbf{S}$ | $\Theta(nc d_h + n d_h d_v)$ FLOPs | Nutzt die Tensor Cores der GPU, ohne $n$ vollständige Zustandsmatrizen zu speichern |

<!--en:42af3a35-->
| **Kapazität und Schärfe** | $\operatorname{rank}(\mathbf{S}_t) \le d_h$, bei der Skalierung kürzt sich $\beta$ für $\mathrm{elu}+1$ heraus | Speichert höchstens $d_h$ orthogonale Schlüssel | Erfordert datenabhängige Vergessensgates $\gamma_t$ und die Kombination mit Softmax-Schichten |
