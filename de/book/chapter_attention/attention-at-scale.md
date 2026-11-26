<!--en:21ed8922-->
# Die Kosten der Attention

<!--en:16f930dc-->
Sprachmodelle mit langem Kontext verarbeiten in einem einzigen Durchlauf Zehntausende oder Hunderttausende Token. Die dichte Self-Attention berechnet für jedes Positionspaar einen Score. Deshalb wachsen der Rechenaufwand und bei naiver Implementierung auch der Speicherbedarf (engl. memory footprint) quadratisch mit der Sequenzlänge $n$. Bei $n = 8{,}192$ wertet ein einzelner Attention-Kopf $6.7 \times 10^7$ Paare aus Abfrage und Schlüssel aus (256 MiB in float32).

<!--en:c0082157-->
Jahrelang galt ein Speicherbedarf von $O(n^2)$ weithin als unvermeidlicher Preis für exakte Attention. Dann zeigten :citet:`Rabe.Staats.2021` auf Grundlage des *Online-Softmax-Algorithmus* von :citet:`Milakov.Gimelshein.2018`, dass exakte Attention ohne die explizite Speicherung der $n \times n$-Score-Matrix auskommt. *FlashAttention* :cite:`Dao.Fu.Ermon.ea.2022` setzt die kachelweise Online-Softmax-Berechnung in einem GPU-Kernel um, der die Datentransfers berücksichtigt. Er ist schneller als die naive Attention und hat einen linearen Speicherbedarf.

<!--en:48391ea4-->
Wir bestimmen die FLOPs und den Speicherbedarf der dichten Attention und messen ihr Laufzeit- und Speicherverhalten auf einer GPU. Dann leiten wir die blockweise Online-Softmax-Berechnung her. Wir vergleichen eine blockweise Implementierung und fusionierte FlashAttention-Kernel mit der naiven Implementierung, die die gesamte Matrix speichert.

<!--en:7029e66f-->
## Quadratischer Zeit- und Speicherbedarf

<!--en:6450278a-->
### Anzahl der FLOPs

<!--en:dfc24cb3-->
Betrachten wir eine Multi-Head-Self-Attention-Schicht der Breite $d$ mit $H$ Köpfen der Breite $d_h = d/H$ für $n$ Token. Aus dem Rechenaufwand für mehrere Köpfe (:numref:`sec_multihead-attention`) und dem Aufwand für die zeilenweise Softmax-Funktion ergibt sich

<!--en:9cb0e42f-->
Der quadratische Term $4n^2 d$ ($2n^2 d_h$ für $\mathbf{Q}\mathbf{K}^\top$ und $2n^2 d_h$ für $\mathbf{A}\mathbf{V}$, jeweils pro Kopf bei $H$ Köpfen) übersteigt den linearen Projektionsterm $8nd^2$, sobald $n > 2d$ gilt. Bei GPT-2 small ($d = 768$, $n =
1{,}024$) überwiegen noch die Projektionen. Für ein Modell mit $d = 4{,}096$ gilt bei $n = 131{,}072$ hingegen $n / (2d) = 16$. Die Berechnung der Scores und die gewichtete Kombination der Werte benötigen damit $16\times$ so viele FLOPs wie die Attention-Projektionen. Beziehen wir das Feedforward-Netz ein, benötigen beide zusammen ungefähr $5\times$ so viele FLOPs wie die übrigen Berechnungen des gesamten Transformer-Blocks (siehe :numref:`chap_transformers`). Für den Block verschiebt sich die Schwelle damit auf $n \approx 6d$.

<!--en:48a706c9-->
### Speicherbedarf

<!--en:edae7f4e-->
Bei langem Kontext wird der Speicher oft früher zum Engpass als die Rechenleistung. Eine naive Implementierung speichert sowohl die $n \times n$-Score-Matrix als auch die $n \times n$-Matrix der Softmax-Gewichte. In float32 sind das $8n^2$ Byte pro Kopf und Sequenz (gegenüber $O(nd)$ für die Projektionen). Für einen Kopf der Breite $d_h = 64$ belegen bei $n = 8{,}192$ die Eingaben $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ zusammen nur 6 MiB, die beiden $n \times n$-Puffer hingegen 512 MiB. Bei $n =
131{,}072$ würde eine einzige Attention-Matrix in float32 bereits 64 GiB benötigen. Wollten wir beim Training von GPT-2 small die Attention-Gewichte für alle 12 Schichten mit je 12 Köpfen speichern, bräuchten wir bei $n = 8{,}192$ für jede Sequenz $144 \times 256\textrm{ MiB} = 36\textrm{
GiB}$.

<!--en:d44239df-->
Wir überprüfen den temporären Speicherbedarf von $8n^2$ Byte für einen einzelnen Kopf in float32 ($d_h = 64$) mit `d2l.peak_memory` (:numref:`sec_use_gpu`):

<!--en:2eb22e82-->
Wir werten `d2l.peak_memory(attention, Q, Q, Q)` für die Sequenzlängen $n \in \{2048, 4096, 8192, 16384\}$ aus. Wie erwartet vervierfacht jede Verdopplung von $n$ den temporären Speicherbedarf, genau wie es die Formel $8n^2$ Byte vorhersagt:

<!--en:5a6686fd-->
### Laufzeitmessung

<!--en:ade9406e-->
Wir messen die Laufzeit desselben Vorwärtsdurchlaufs mit `d2l.Benchmark` (:numref:`sec_use_gpu`). Sie wächst wie erwartet quadratisch, sobald $n$ so groß ist, dass die Latenz beim Kernelstart kaum noch ins Gewicht fällt ($n \ge 8{,}192$):

<!--en:046ed865-->
## Exakte Attention ohne die Matrix

<!--en:58795325-->
### Blockweise Online-Softmax-Berechnung

<!--en:9e4ecd91-->
Auf den ersten Blick scheinen wir für den Softmax-Nenner $\sum_{j'} \exp(a_{j'})$ und das zur numerischen Stabilität verwendete Zeilenmaximum $\max_{j'} a_{j'}$ (:numref:`subsec_softmax-implementation-revisited`) alle $n$ Scores zu benötigen, bevor wir auch nur einen Wert gewichten können. Der *Online-Softmax-Algorithmus* :cite:`Milakov.Gimelshein.2018` kommt mit einem Durchlauf aus. Er verarbeitet Schlüssel in Blöcken der Größe $c$ und führt für jede Abfrage drei Größen mit: den bisher größten Score $m$, die laufende Summe im Nenner $s =
\sum_j e^{a_j - m}$ und die unnormierte Ausgabesumme $\mathbf{o} =
\sum_j e^{a_j - m} \mathbf{v}_j$.

<!--en:020f3bc1-->
Kommt ein neuer Block mit Scores $\{a_j\}$ und Werten $\{\mathbf{v}_j\}$ hinzu, aktualisieren wir das laufende Maximum auf $m' = \max(m, \max_j a_j)$. Dazu müssen wir die bisherige Summe $s$ und die bisherige Ausgabe $\mathbf{o}$ mit $e^{m - m'}$ multiplizieren:

<!--en:fad3f39b-->
Hat etwa Block 1 die Scores $(1, 3)$, so gilt $m = 3$ und $s = e^{-2} +
e^0 \approx 1.135$. Hat Block 2 die Scores $(5, 2)$, so ist das neue Maximum $m' =
5$. Wir multiplizieren daher die Summe aus Block 1 mit $e^{3 - 5} = e^{-2}$ und addieren $e^{5 - 5} + e^{2 - 5}$. Das Ergebnis $s' = e^{-4} + e^{-2} + e^0 + e^{-3}
\approx 1.203$ ist genau der übliche Softmax-Nenner, den wir bei vorab bekanntem globalem Maximum $5$ erhalten. Nach dem letzten Block stimmt $\mathbf{o} / s$ bis auf Gleitkomma-Rundungsfehler mit $\mathrm{softmax}(\mathbf{a})\mathbf{V}$ überein. Dabei liegt immer nur ein $n \times c$-Streifen der Score-Matrix im Speicher (:numref:`fig_online-softmax`, :cite:`Rabe.Staats.2021`).

<!--en:0bf51322-->
![Der Online-Softmax-Algorithmus verarbeitet die Score-Matrix Schlüsselblock für Schlüsselblock. Von den Scores liegt nur der aktuelle $n \times c$-Streifen im Speicher, neben den laufenden Maxima, Summen und Ausgaben $\mathbf{m}, \mathbf{s}, \mathbf{O}$ aller Abfragen. Die vollständige $n \times n$-Matrix wird nie gespeichert.](../img/mdl-attention-online-softmax.svg)

<!--en:c31639d0-->
### Blockweise Implementierung

<!--en:9d2dadf5-->
Wir vergleichen `causal_attention`, das die vollständige Matrix berechnet, mit `chunked_attention`. Diese Funktion verarbeitet Blöcke von Schlüsseln und Werten der Breite `chunk_size=512` nach :eqref:`eq_online-softmax`. Zur Initialisierung von $m$ und zur Maskierung künftiger Positionen verwendet sie `finfo(dtype).min`. So vermeidet sie $-\infty - (-\infty) = \mathrm{NaN}$, wenn die kausale Maske einen ganzen Block abdeckt (siehe Übung :exref:`ex_attention-at-scale-running-maximum`d).

<!--en:06220664-->
`chunked_attention` stimmt mit `causal_attention` bis auf die Genauigkeit von float32 ($\sim
10^{-6}$) überein und senkt dabei den maximalen Speicherbedarf von $O(n^2)$ auf $O(nc)$:

<!--en:2275195b-->
Ein Vergleich mit `d2l.peak_memory` für $n \in \{2048, 4096, 8192, 16384\}$ bestätigt, dass `chunked_attention` den quadratischen Speicherbedarf $O(n^2)$ von `causal_attention` auf einen linear wachsenden Bedarf von $O(nc)$ senkt:

<!--en:764ff8bd-->
### Speicherverkehr und fusionierte Kernel

<!--en:d72194f6-->
Die Recheneinheiten einer GPU führen Matrixoperationen um Größenordnungen schneller aus, als der Gerätespeicher außerhalb des Chips Daten übertragen kann (RTX 4090: $\sim 165$ TFLOP/s in float16 bei $\sim 1$ TB/s Speicherbandbreite, siehe :numref:`chap_performance`). Grafikkarten für Rechenzentren verwenden dafür High Bandwidth Memory (HBM), die RTX 4090 verwendet GDDR6X. Wie in der Arbeit zu FlashAttention bezeichnen wir beide Arten von Gerätespeicher als HBM. Die naive Attention ist *durch die Speicherbandbreite begrenzt*: Sie schreibt die $n \times n$-Score-Matrix in den HBM, liest sie für die Softmax-Funktion zurück, schreibt die $n \times n$ Gewichte in den HBM und liest sie für die Multiplikation mit $\mathbf{V}$ erneut. Damit überträgt sie $4n^2$ Gleitkommazahlen bei geringer arithmetischer Intensität (engl. arithmetic intensity, siehe Übungen :exref:`ex_attention-at-scale-arithmetic-intensity` und :exref:`ex_attention-at-scale-tiled-traffic`). Eine Python-Schleife wie die in `chunked_attention` startet weiterhin separate GPU-Kernel. Diese schreiben jeden $n \times c$-Streifen in den HBM und lesen ihn wieder aus. Damit sinkt der maximale Speicherbedarf, der gesamte HBM-Verkehr bleibt jedoch gleich.

<!--en:ed6b492c-->
*FlashAttention* :cite:`Dao.Fu.Ermon.ea.2022` zerlegt sowohl die Abfragen als auch die Schlüssel in Kacheln. Jeder Score-Block und die zugehörigen laufenden Größen $(m, s, \mathbf{o})$ bleiben während der gesamten Schleife im schnellen SRAM auf dem Chip und gelangen nie in den HBM. Für den Rückwärtsdurchlauf speichert der Vorwärtsdurchlauf anstelle von $n \times n$ Attention-Gewichten nur den logarithmierten Normierungsfaktor $\ell = m + \log s$ pro Abfrage (Speicherbedarf $O(n)$). Bei der Backpropagation berechnet der Kernel die Score-Kacheln auf dem Chip erneut.

<!--en:2713da21-->
Wir vergleichen den fusionierten FlashAttention-Kernel des Frameworks mit `causal_attention` für $B = 2$ Sequenzen, $H = 8$ Köpfe, $d_h = 64$ und $n = 8{,}192$ in float16. Ein zusätzlicher Lauf des fusionierten Kernels ohne Maske trennt zwei Effekte: den Laufzeitgewinn durch den Wegfall des HBM-Verkehrs der Ordnung $O(n^2)$ und die Beschleunigung um den Faktor $\sim 1.5\times$ durch das Überspringen kausal maskierter Kacheln im oberen Dreieck:

<!--en:65dda751-->
Der fusionierte Kernel ist `F.scaled_dot_product_attention`.

<!--en:2eb60036-->
Der fusionierte Kernel ist `jax.nn.dot_product_attention` mit `implementation='cudnn'`.

<!--en:68692f2e-->
Der fusionierte Kernel senkt den temporären Speicherbedarf von $\sim 4$ GiB ($16 \cdot 4n^2$ Byte) auf ein kleines Vielfaches der 16 MiB großen Ausgabe. Er läuft um ein Mehrfaches schneller und erreicht nahezu den maximalen Matrixdurchsatz der GPU. Auch mit fusionierten Kerneln erfordert exakte Attention jedoch weiterhin $\Theta(n^2 d)$ Rechenoperationen. Für lineare Zeitkomplexität müssen wir den Attention-Mechanismus selbst ändern (:numref:`sec_windowed-linear-attention`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:4277568f-->
Dichte Self-Attention erfordert $8nd^2 + 4n^2d$ FLOPs pro Schicht. Der quadratische Aufwand für die Scores und die gewichtete Kombination der Werte übersteigt den linearen Aufwand für die Projektionen, sobald $n > 2d$ gilt. Schwerer wiegt der Speicherbedarf: Wer die $n \times n$-Matrizen für Scores und Gewichte vollständig speichert, belegt in float32 pro Kopf $8n^2$ Byte. Auf modernen GPUs begrenzt daher die Speicherbandbreite die Geschwindigkeit der naiven Attention. Der Online-Softmax-Algorithmus kommt ohne die beiden $n \times n$-Puffer und ohne Näherung aus. Er verarbeitet Schlüssel blockweise und führt das laufende Maximum $m$, den Normierungsfaktor $s$ und die gewichtete Wertsumme $\mathbf{o}$ mit. Die Teilsummen werden mit $e^{m - m'}$ multipliziert, sobald $m$ steigt. FlashAttention führt die Schritte dieser Rekursion kachelweise im schnellen SRAM auf dem Chip zusammen. Dadurch sinkt der Speicherbedarf außerhalb des Chips auf $O(n)$ und die gemessene Laufzeit verkürzt sich um ein Mehrfaches. Die Attention bleibt exakt und benötigt weiterhin $\Theta(n^2 d)$ Rechenoperationen.

<!--en:2294243a-->
## Übungen

<!--en:bb716442-->
1. **Online-Softmax von Hand.** Für eine Abfrage liegen vier Schlüssel in zwei Blöcken zu je zwei Schlüsseln vor. Die Scores lauten $(0, 2)$ im ersten Block und $(1, 3)$ im zweiten. Die zugehörigen Werte sind $1, 2$ im ersten Block und $3, 4$ im zweiten.
    1. Wenden Sie :eqref:`eq_online-softmax` blockweise an. Beginnen Sie mit $m = -\infty$, $s = 0$ und $o = 0$. Notieren Sie nach jedem Block $m$, $s$ und $o$ und bestimmen Sie die Ausgabe durch Division.
    1. Berechnen Sie die vier Softmax-Gewichte direkt und prüfen Sie, ob sie dieselbe Ausgabe ergeben.
    1. Bilden Sie nach dem letzten Block $\ell = m + \log s$. Berechnen Sie jedes der vier Gewichte erneut als $e^{a_j - \ell}$ allein aus dem zugehörigen Score und dieser Größe. Prüfen Sie, ob die Gewichte mit denen aus der vorigen Teilaufgabe übereinstimmen und sich zu eins summieren. Erklären Sie, warum es für den Rückwärtsdurchlauf eines fusionierten Kernels genügt, $\ell$ anstelle von $m$ und $s$ zu speichern.

<!--en:c6f562c4-->
1. [code] **Das laufende Maximum.** Entfernen Sie das laufende Maximum aus `chunked_attention`, sodass der Zustand direkt die Terme $e^{a_j}$ aufsummiert. Führen Sie die geänderte Funktion mit $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ und den Zustandsgrößen $s$, $\mathbf{O}$ in float16 aus. Vergrößern Sie die Abfragen um den Faktor $10$.
    1. Berechnen Sie vor der Ausführung den größten Exponenten $x$, für den $e^x$ in float16 endlich ist. Schätzen Sie, welcher Anteil der skalierten Scores diese Schwelle überschreitet, wenn die Einträge der Abfragen und Schlüssel der Breite $64$ unabhängig und standardnormalverteilt sind.
    1. Führen Sie die geänderte Funktion aus. Geben Sie an, bei welcher Operation der erste unendliche oder undefinierte Wert auftritt.
    1. Erklären Sie, warum das Mitführen von $m$ jeden Exponenten nach oben durch null beschränkt. Ein Term $e^{a_j - m'}$, der höchstens halb so groß ist wie die kleinste positive float16-Zahl, wird bei Rundung zur nächsten Zahl mit gerader letzter Stelle im Gleichstandsfall auf null gerundet. Geben Sie eine obere Schranke für den Gesamtbeitrag solcher Terme zu $s$ im Verhältnis zum Term des größten Scores an. Betrachten Sie dazu eine Zeile mit $n$ Schlüsseln.
    1. `chunked_attention` initialisiert $m$ mit dem kleinsten endlichen Wert des Datentyps und maskiert Scores mit demselben Wert. Wählen Sie $n = 2c$ für Blöcke mit je $c$ Schlüsseln. Lassen Sie die Schleife die beiden Schlüsselblöcke in umgekehrter Reihenfolge durchlaufen. Für eine Abfrage aus dem ersten Block sind dann alle Scores des zuerst besuchten Blocks maskiert. Verfolgen Sie $m$, $s$ und $\mathbf{O}$ für diese Abfrage über beide Schleifendurchläufe und zeigen Sie, dass die endgültige Ausgabe weiterhin exakt ist.
   :exlabel:`ex_attention-at-scale-running-maximum`

<!--en:e5f155a7-->
1. **Teilzustände zusammenführen.** Betrachten Sie eine feste Abfrage mit Scores $a_j$ und Werten $\mathbf{v}_j$. Für eine Schlüsselmenge $B$ nennen wir das Tripel $(m_B, s_B,
   \mathbf{o}_B)$, das :eqref:`eq_online-softmax` allein aus den Schlüsseln in $B$ berechnet, den Teilzustand von $B$. Damit ist $\mathbf{o}_B / s_B$ die auf $B$ beschränkte Attention-Ausgabe.
    1. Leiten Sie den Teilzustand von $B \cup B'$ für disjunkte Mengen $B$ und $B'$ aus deren Teilzuständen her. Zeigen Sie, dass :eqref:`eq_online-softmax` dem Spezialfall entspricht, in dem $B'$ aus einem einzigen Block besteht.
    1. Zeigen Sie, dass diese Verknüpfung assoziativ und kommutativ ist. Folgern Sie, dass sich die Schlüsselblöcke in beliebiger Reihenfolge verarbeiten lassen. Alternativ lassen sich die Blöcke parallel verarbeiten und ihre Teilzustände in einem Baum der Tiefe $\lceil \log_2 (n/c) \rceil$ für Blöcke mit je $c$ Schlüsseln zusammenführen. Erklären Sie, warum das bei der Generierung wichtig ist, wenn eine einzelne Abfrage die Attention über einen langen Cache berechnet.
    1. Ersetzen Sie das Paar $(m_B, s_B)$ durch die einzelne Zahl $\ell_B = m_B +
       \log s_B$ und das Tripel durch $(\ell_B, \bar{\mathbf{o}}_B)$ mit $\bar{\mathbf{o}}_B = \mathbf{o}_B / s_B$. Drücken Sie die zusammengeführten Größen $\ell$ und $\bar{\mathbf{o}}$ durch diese Größen aus. Zeigen Sie, dass $\bar{\mathbf{o}}$ eine konvexe Kombination von $\bar{\mathbf{o}}_B$ und $\bar{\mathbf{o}}_{B'}$ ist, deren Gewichte nur von $\ell_B$ und $\ell_{B'}$ abhängen.

<!--en:98279925-->
1. **Arithmetische Intensität der Attention.** Betrachten Sie einen Kopf mit je $n$ Abfragen und Schlüsseln der Breite $d_h$ sowie Werten der Breite $d_v$. Seine beiden Matrixprodukte, $\mathbf{Q}\mathbf{K}^\top$ und das Produkt der Gewichte mit $\mathbf{V}$, benötigen $2n^2 d_h$ beziehungsweise $2n^2 d_v$ Gleitkommaoperationen. Die Softmax-Funktion benötigt einige wenige Operationen pro Score.
    1. Nehmen Sie an, die Berechnung schreibt die Score-Matrix einmal in den Speicher und die Softmax-Funktion liest sie einmal. Deren Ausgabe wird ebenfalls einmal geschrieben und für die Multiplikation mit den Werten einmal gelesen, alles in einfacher Genauigkeit. Zählen Sie die bei diesen vier Durchläufen übertragenen Byte. Zählen Sie auch die Byte für das Lesen von $\mathbf{Q}$, $\mathbf{K}$ und $\mathbf{V}$ sowie für das Schreiben der Ausgabe.
    1. Vernachlässigen Sie den Rechenaufwand der Softmax-Funktion und bilden Sie das Verhältnis von Operationen zu übertragenen Byte. Betrachten Sie $n \to \infty$ bei festen $d_h$ und $d_v$. Zeigen Sie, dass der Grenzwert $(d_h + d_v)/8$ beträgt, und geben Sie an, wie er sich bei Berücksichtigung des Softmax-Rechenaufwands ändert.
    1. Bestimmen Sie den Grenzwert erneut für Eingaben und Scores in einem 16-Bit-Format. Entnehmen Sie den 16-Bit-Durchsatz und die Speicherbandbreite einer H100 oder B200 aus :numref:`tab_gpu_specs`. Berechnen Sie daraus die Zahl der Operationen pro Byte und entscheiden Sie, ob die Rechenleistung oder der Speicherverkehr die Attention mit $d_h = d_v = 64$ auf dieser GPU begrenzt.
   :exlabel:`ex_attention-at-scale-arithmetic-intensity`

<!--en:54636535-->
1. **Speicherverkehr bei kachelweiser Berechnung.** ● Nach Übung :exref:`ex_attention-at-scale-arithmetic-intensity`a überträgt die naive Schicht zusätzlich zu den in der Sequenzlänge linearen Termen $4n^2$ Gleitkommazahlen zwischen Hauptspeicher und Recheneinheiten. Die linearen Terme hängen von $n$ ab. Der quadratische Term entsteht, weil die Schicht die Score-Matrix und die Gewichtsmatrix jeweils einmal schreibt und einmal liest. Vernachlässigen Sie alle Transfers, deren Umfang linear in $n$ ist.
    1. Nehmen Sie an, `chunked_attention` führt die Operationen einzeln aus. Jede Operation liest ihre Eingaben aus dem Hauptspeicher und schreibt ihre Ausgabe dorthin zurück. Ermitteln Sie in Abhängigkeit von $n$, $c$ und $d_h$, wie viele Gleitkommazahlen für die Score-Streifen übertragen werden, wie viele für $\mathbf{Q}$ und die laufende Ausgabesumme $\mathbf{O}$ und wie viele Operationen gestartet werden. Welche dieser Größen verringert die blockweise Verarbeitung gegenüber der naiven Schicht?
    1. Ein fusionierter Kernel hält dagegen eine Kachel mit $b$ Abfragen, deren laufende Ausgabesummen und je eine Kachel mit $b$ Schlüsseln und $b$ Werten in einem schnellen Speicher. Dieser fasst $M$ Gleitkommazahlen, sodass $4bd_h \leq M$ gilt. Auch die Scores einer Kachel und die laufenden Größen $m$ und $s$ bleiben im schnellen Speicher. Der Kernel lädt jede Abfragekachel einmal, verarbeitet dazu nacheinander alle Schlüssel- und Wertkacheln und schreibt jede Ausgabekachel einmal. Zeigen Sie, dass er abgesehen von den linearen Termen $2n^2 d_h / b$ Gleitkommazahlen überträgt und dass sich für die größte zulässige Kachel $\Theta(n^2 d_h^2 / M)$ ergibt. Diese Schranke beweisen :citet:`Dao.Fu.Ermon.ea.2022` für FlashAttention.
    1. Berechnen Sie das Verhältnis des Speicherverkehrs des fusionierten Kernels zu dem der naiven Schicht für $d_h \in \{64, 128, 256\}$ bei $M = 2^{16}$. Geben Sie an, wie die Kopfbreite den Nutzen der Kachelung begrenzt.
   :exlabel:`ex_attention-at-scale-tiled-traffic`

<!--en:57c95bb8-->
[Diskussion](https://d2l.discourse.group/t/36772)

<!--en:68620334-->
Attention bei langen Sequenzen: Speicherbandbreite, Online-Softmax und FlashAttention<br>**quadratischer Rechen- und Speicherbedarf · arithmetische Intensität · blockweise Online-Softmax-Berechnung · Kachelung im SRAM · FlashAttention**

<!--en:c754923b-->
::: {.slide title="Warum der Speicher die Attention bei langem Kontext begrenzt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:bcbffd69-->
- **Quadratischer temporärer Speicherbedarf ($8n^2$ Byte pro Kopf in FP32):** Die $n \times n$-Matrizen für Scores und Softmax-Gewichte belegen $8\text{ MiB}$ bei $n = 1{,}024$, $512\text{ MiB}$ bei $n = 8{,}192$ und **$128\text{ GiB}$ pro Kopf** bei $n = 131{,}072$.

<!--en:c300a3a8-->
- **Mehrbedarf beim Training:** Die Gewichte aller $144$ Köpfe von GPT-2 small belegen bei $n = 8{,}192$ insgesamt $36\text{ GiB}$ pro Sequenz.

<!--en:f28faad3-->
- **Der Durchbruch:** Der Online-Softmax-Algorithmus (Milakov & Gimelshein, 2018, Rabe & Staats, 2021) und **FlashAttention** (Dao et al., 2022) berechnen *exakte* Attention mit **Speicherbedarf $O(n)$**.

<!--en:b965d9fe-->
::: {.slide title="FLOPs zählen: Wann überwiegt die Score-Berechnung gegenüber den Projektionen?"}

<!--en:a6091560-->
[Skalierung des Rechenaufwands]{.kicker}

<!--en:2c16ea03-->
- **FLOPs pro Schicht nach Rechenschritt:**
  $$\textrm{FLOPs} = 8 n d^2 + 4 n^2 d + O(H n^2)$$

<!--en:117c16eb-->
- **Verhältnis der Rechenkosten:** Teilen wir den quadratischen Aufwand für Scores und gewichtete Summen ($4n^2d$) durch den linearen Aufwand für die Projektionen ($8nd^2$), erhalten wir $n / (2d)$.

<!--en:d13ad3d5-->
- **Kurzer und langer Kontext:** Bei $d = 768, n = 1{,}024$ überwiegen die Projektionen ($n/2d = 0.67$). Bei $d = 4{,}096, n = 131{,}072$ erfordern Scores und gewichtete Summen der Attention **$16\times$** so viele FLOPs wie alle Projektionen zusammen.

<!--en:09b52f6b-->
::: {.slide title="Speicherbedarf: 8n² Byte für Zwischenergebnisse"}

<!--en:4c20aac5-->
[Speicherbelegung]{.kicker}

<!--en:743f1884-->
Warum belegt die naive Attention $8n^2$ Byte temporären Speicher pro Kopf in `float32`?

<!--en:ae373635-->
- **Zwei $n \times n$-Matrizen:** `scores = Q @ K.T / sqrt(d_h)` belegt $4n^2$ Byte. `softmax(scores)` reserviert weitere $4n^2$ Byte für die Gewichte, bevor diese mit `V` multipliziert werden.

<!--en:14e5ccac-->
- **Die Eingaben sind vergleichsweise klein:** Bei $n = 8{,}192, d_h = 64$ belegen die Eingabetensoren $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ nur $3 \times 8{,}192 \times 64 \times 4\text{ B} = 6\text{ MiB}$. Die beiden $n \times n$-Puffer benötigen **$512\text{ MiB}$** ($85\times$ so viel).

<!--en:fc378388-->
::: {.slide title="Quadratischen Speicherbedarf und quadratische Laufzeit auf der GPU messen"}

<!--en:05c5a477-->
[Empirische Messungen]{.kicker}

<!--en:127ad23d-->
Wir messen den maximalen temporären Speicherbedarf (`d2l.peak_memory`) und die verstrichene Zeit (`d2l.Benchmark`) für einen einzelnen Kopf in `float32` ($d_h = 64$). Die Messung bestätigt den Bedarf von genau $8n^2$ Byte und den Anstieg der Laufzeit um den Faktor $4\times$ bei jeder Verdopplung von $n$:

<!--en:8f6533fb-->
::: {.slide title="Warum die naive Softmax-Berechnung zwei Durchläufe zu erfordern scheint"}

<!--en:04fd6f49-->
[Zwei Durchläufe als Engpass]{.kicker}

<!--en:28225dbf-->
- **Zwei globale Größen:** Die übliche numerisch stabile Softmax-Funktion benötigt das globale Zeilenmaximum $m = \max_j a_j$ und den Nenner $s = \sum_j \exp(a_j - m)$, bevor sie einen Wert $\mathbf{v}_j$ gewichten kann.

<!--en:6042461a-->
- **Wozu speichern wir $m$?** Ohne Subtraktion von $m$ läuft $\exp(a_j)$ in `float16` oberhalb von $a_j \approx 11.09$ über, in `float32` oberhalb von $a_j \approx 88.7$.

<!--en:51bfce1e-->
- **Die Idee des Online-Verfahrens:** Wir verarbeiten Schlüssel blockweise und *skalieren die laufenden Summen neu*, sobald ein Block ein größeres Maximum $m' > m$ enthält.

<!--en:61aea113-->
::: {.slide title="Herleitung der blockweisen Online-Softmax-Berechnung"}

<!--en:3fefa837-->
[Rekursion des Online-Softmax-Algorithmus]{.kicker}

<!--en:1080ebdb-->
- **Drei laufende Größen pro Abfrage:** Maximum $m$, unnormierte Summe $s = \sum_j e^{a_j - m}$ und unnormierte Ausgabe $\mathbf{o} = \sum_j e^{a_j - m}\mathbf{v}_j$.

<!--en:fd64fed4-->
- **Aktualisierung pro Block:**
  $$m' = \max\!\big(m, \max_j a_j\big), \quad s' = s\,e^{m - m'} + \sum_j e^{a_j - m'}$$
  $$\mathbf{o}' = \mathbf{o}\,e^{m - m'} + \sum_j e^{a_j - m'}\mathbf{v}_j$$

<!--en:3c47feb3-->
- **Exakte Ausgabe:** Nach dem letzten Block gilt $\mathbf{o} / s = \mathrm{softmax}(\mathbf{a})\mathbf{V}$.

<!--en:93cef56d-->
::: {.slide title="Rechenbeispiel: Online-Softmax über zwei Blöcke"}

<!--en:2d3436d4-->
[Rechnung Schritt für Schritt]{.kicker}

<!--en:01f5d34b-->
- **Block 1 (Scores $1, 3$):** Wir erhalten $m_1 = 3$ und $s_1 = e^{-2} + e^0 \approx 1.1353$.

<!--en:e9ce6919-->
- **Block 2 (Scores $5, 2$):** Wir setzen das Maximum auf $m_2 = \max(3, 5) = 5$, multiplizieren $s_1$ mit $e^{3-5} = e^{-2}$ und addieren $e^0 + e^{-3}$. Das ergibt $s_2 \approx 1.2034$.

<!--en:cc2c9131-->
- **Assoziativ und kommutativ:** Verarbeiten wir Block 2 vor Block 1, bleibt $m = 5$ erhalten ($e^{m-m'} = 1$). Wir erhalten genau dieselbe Summe $1.2034$.

<!--en:2e6d6ac2-->
::: {.slide title="Speicherbelegung bei streifenweiser Online-Softmax-Berechnung"}

<!--en:3e8c1625-->
[Linearer Arbeitsspeicherbedarf]{.kicker}

<!--en:484eb9dd-->
- **Ein Streifen nach dem anderen:** Von der $n \times n$-Score-Matrix halten wir nur den $n \times c$-Streifen für den aktuellen Block von $c$ Schlüsseln im Speicher.

<!--en:928ee460-->
- **Laufender Zustand der Abfragen:** Neben dem Streifen speichern wir $\mathbf{m} \in \mathbb{R}^{n \times 1}$, $\mathbf{s} \in \mathbb{R}^{n \times 1}$ und $\mathbf{O} \in \mathbb{R}^{n \times d_v}$ (insgesamt $O(n d_v)$ Gleitkommazahlen).

<!--en:3b03e140-->
- **Speicherersparnis:** Der maximale Speicherbedarf für Scores sinkt von $O(n^2)$ auf $O(nc)$. Er wächst **linear mit $n$**, wenn die Blockgröße $c$ fest bleibt.

<!--en:aa244c06-->
![Der Online-Softmax-Algorithmus durchläuft die Schlüsselblöcke in Streifen der Größe n × c und aktualisiert dabei m, s und O.](../img/mdl-attention-online-softmax.svg){.nostretch width=100%}

<!--en:4bd9c37f-->
::: {.slide title="Implementierung der blockweisen kausalen Attention"}

<!--en:a4fbb6b9-->
[Implementierung]{.kicker}

<!--en:4a8689ba-->
Die Referenzfunktion `causal_attention` speichert die vollständige $n \times n$-Score-Matrix. Jeder schnellere Kernel muss dasselbe Ergebnis liefern:

<!--en:0fd25624-->
::: {.slide title="Implementierung der blockweisen kausalen Attention (Fortsetzung)"}

<!--en:a4fbb6b9-2-->
[Implementierung]{.kicker}

<!--en:f976b9ab-->
Die Maskierung mit `finfo(dtype).min` anstelle von $-\infty$ verhindert bei vollständig maskierten Blöcken den Ausdruck $-\infty - (-\infty) = \mathrm{NaN}$:

<!--en:aead7d3a-->
::: {.slide title="Prüfung auf Exaktheit und linearen Speicherbedarf"}

<!--en:478ce0c3-->
[Blockweise Berechnung und vollständige Matrix im Vergleich]{.kicker}

<!--en:261bbdbb-->
- **Exakt bis auf Maschinengenauigkeit:** Bei $n = 2{,}048, d_h = 64$ stimmt `chunked_attention` mit `causal_attention` bis auf $\sim 10^{-7}$ überein (Rundung in `float32`):

<!--en:78fb2402-->
- **Speicherbedarf: quadratisch $\to$ linear:** Bei $n = 16{,}384$ belegt die Attention mit vollständiger Matrix $2{,}048\text{--}2{,}308\text{ MiB}$ ($>2\text{ GiB}$). Dagegen benötigt `chunked_attention` bei $c = 512$ nur $\approx 36\text{--}132\text{ MiB}$ (um einen Faktor $>17\times$ weniger).

<!--en:dd7ffd54-->
::: {.slide title="Der Hardwareengpass: HBM der GPU und SRAM auf dem Chip"}

<!--en:8b98d8a8-->
[Speicherhierarchie der GPU]{.kicker}

<!--en:f8d7fed5-->
- **Tensor Cores und HBM:** Eine GPU erreicht $\sim 165$ bis $1{,}000\text{ TFLOP/s}$ bei 16-Bit-Matrixoperationen. Der Gerätespeicher außerhalb des Chips überträgt nur $\sim 1.0\text{ TB/s}$ (GDDR6X, RTX 4090) bis $3.35\text{ TB/s}$ (HBM, H100). Auf der RTX 4090 entspricht das $\sim 164$ FLOPs pro übertragenem Byte. Beide Speicherarten bezeichnen wir als HBM.

<!--en:c858d87b-->
- **SRAM auf dem Chip (einige zehn $\text{TB/s}$):** Register und L1-Cache bzw. gemeinsam genutzter Speicher sind um eine Größenordnung schneller als der Gerätespeicher, fassen aber insgesamt nur $\sim 0.1\text{ GB}$.

<!--en:f69776af-->
- **Warum die blockweise Berechnung in Python langsam bleibt:** Die separat gestarteten Kernel lesen und schreiben jeden $n \times c$-Streifen im langsamen HBM.

<!--en:2cffb5e2-->
::: {.slide title="Arithmetische Intensität der naiven Attention"}

<!--en:7a3f7d8a-->
[Roofline-Analyse]{.kicker}

<!--en:2ed823a4-->
- **Vier HBM-Zugriffe auf je $n^2$ Gleitkommazahlen:** Die naive Attention schreibt $\mathbf{S}$ in den HBM, liest $\mathbf{S}$ für die Softmax-Funktion, schreibt $\mathbf{A}$ in den HBM und liest $\mathbf{A}$ für die Multiplikation mit $\mathbf{V}$.

<!--en:4b5e2b29-->
- **Geringe arithmetische Intensität:**
  $$\textrm{Intensity} \;\xrightarrow{n \to \infty}\; \frac{d_h + d_v}{4}\;\textrm{FLOPs/byte (fp16)}$$

<!--en:89f5080f-->
- **Speicherbegrenzter Betrieb:** Für $d_h = d_v = 64$ in FP16 beträgt die Intensität nur $32\text{ FLOPs/byte}$. Das liegt weit unter dem Knickpunkt der Roofline der RTX 4090 bei $\sim 164\text{ FLOPs/byte}$.

<!--en:35fa42c4-->
::: {.slide title="FlashAttention: Kacheln aus Abfragen und Schlüsseln im SRAM"}

<!--en:fbf0f657-->
[Kernelfusion für effiziente Speicherzugriffe]{.kicker}

<!--en:fb37b88c-->
- **Kein HBM-Verkehr für Scores:** FlashAttention lädt eine Abfragekachel $\mathbf{Q}_i$ und $(m_i, s_i, \mathbf{O}_i)$ in den SRAM auf dem Chip, liest nacheinander die $(\mathbf{K}_j, \mathbf{V}_j)$-Kacheln aus dem HBM und schreibt $\mathbf{O}_i / s_i$ nur einmal in den HBM.

<!--en:60b5f14f-->
- **Schranke für den HBM-Verkehr:** Bei maximaler Kachelgröße $b \propto M / d_h$ sinkt der quadratische HBM-Verkehr von $4n^2$ auf $\Theta(n^2 d_h^2 / M)$ Gleitkommazahlen.

<!--en:38412659-->
- **Neuberechnung im Rückwärtsdurchlauf mit Speicherbedarf $O(n)$:** Der Kernel speichert nur $\ell_i = m_i + \log s_i$ ($1$ Gleitkommazahl pro Abfrage) und berechnet $\alpha_{ij} = \exp(a_{ij} - \ell_i)$ bei der Backpropagation auf dem Chip erneut.

<!--en:e59d5454-->
::: {.slide title="Fusionierte FlashAttention und naive Attention im Benchmark" except="jax"}

<!--en:9e2a9d20-->
[Code und gemessene Beschleunigung]{.kicker}

<!--en:4bd9411a-->
Wir vergleichen `causal_attention` mit dem fusionierten FlashAttention-Kernel des Frameworks für $B = 2$ Sequenzen, $H = 8$ Köpfe, $d_h = 64$ und $n = 8{,}192$ in `float16`:

<!--en:28347c77-->
::: {.slide title="Fusionierte FlashAttention und naive Attention im Benchmark" only="jax"}

<!--en:9e2a9d20-2-->
[Code und gemessene Beschleunigung]{.kicker}

<!--en:5d907989-->
Wir vergleichen `causal_attention` mit dem fusionierten FlashAttention-Kernel des Frameworks. Der fusionierte cuDNN-Kernel erwartet die Achsenfolge (Batch, Länge, Köpfe, Breite). Die aufrufende Funktion vertauscht daher die Achsen bei der Ein- und Ausgabe:

<!--en:d293a506-->
::: {.slide title="Fusionierte FlashAttention und naive Attention im Benchmark (Fortsetzung)" only="jax"}

<!--en:9e2a9d20-3-->
[Code und gemessene Beschleunigung]{.kicker}

<!--en:3bba3560-->
Wir messen die Laufzeit aller drei Varianten für $B = 2$ Sequenzen, $H = 8$ Köpfe, $d_h = 64$ und $n = 8{,}192$ in `float16`:

<!--en:e2db3b81-->
::: {.slide title="Beschleunigung durch Kernelfusion und Überspringen kausal maskierter Kacheln"}

<!--en:68dd96c9-->
[Auswertung des Benchmarks]{.kicker}

<!--en:19d334b2-->
Der Vergleich von `naive`, `fused` (kausal) und `fused, no mask` trennt zwei Ursachen der Beschleunigung:

<!--en:6d83682e-->
- **1. Wegfall des HBM-Verkehrs für Scores (`naive` $\to$ `fused, no mask`):** Der maximale temporäre Speicherbedarf sinkt von $\sim 4{,}096\text{ MiB}$ ($16 \times 4n^2$ Byte) auf ein kleines Vielfaches des $16\text{ MiB}$ großen Ausgabetensors (Verringerung um einen Faktor $> 80\times$). Die Ausführung ist um ein Mehrfaches schneller, weil die Tensor Cores nahezu ihren maximalen Durchsatz erreichen ($2.7 \times 10^{11}\text{ FLOPs}$).

<!--en:b50ce8b4-->
- **2. Überspringen kausal maskierter Kacheln (`fused, no mask` $\to$ `fused`):** Eine kausale Maske setzt alle Einträge oberhalb der Diagonalen auf null ($j > i$). Der Kernel kann daher das gesamte obere Dreieck aus leeren Kacheln überspringen, ohne sie zu berechnen.

<!--en:7a45ae72-->
- **Warum das Überspringen nur um den Faktor $\sim 1.5\times$ statt $2\times$ beschleunigt:** Kacheln auf der Diagonalen erstrecken sich über beide Seiten von $j = i$ und erfordern eine elementweise Maskierung. Bei kausalen Zeilen unterschiedlicher Länge verursacht die Zuteilung der Threadblöcke einen kleinen Mehraufwand für den Lastausgleich.

<!--en:25efbe7f-->
::: {.slide title="Was fusionierte Kernel lösen und welcher Aufwand O(n²) bleibt"}

<!--en:ad5ae2a9-->
[Architektur]{.kicker}

<!--en:a3855349-->
- **Was FlashAttention löst:** Die Score- und Gewichtsmatrizen der Größe $\Theta(n^2)$ im HBM entfallen. Der maximale temporäre Speicherbedarf sinkt auf $\Theta(n d_h)$ und der HBM-Verkehr auf $\Theta(n^2 d_h^2 / M)$.

<!--en:7eb39658-->
- **Welcher Aufwand von der Ordnung $\Theta(n^2)$ bleibt:** Die Anzahl der FLOPs wächst weiterhin wie $\Theta(n^2 d)$, der autoregressive KV-Cache wie $O(n d)$. Das motiviert **gleitende Fenster** und **lineare Attention** (§{sec_windowed-linear-attention}).

<!--en:142a7c07-->
::: {.slide title="Verständnisfragen zu Online-Softmax und Speicherverkehr"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:e0943ca8-->
Bearbeiten Sie diese drei Fragen zu Hardware und Algorithmen:

<!--en:b7e59a4f-->
1. **Rekonstruktion mit Log-Sum-Exp im Rückwärtsdurchlauf:** Der letzte Block des Online-Softmax-Algorithmus liefert $(m, s, \mathbf{o})$. Wie lässt sich im Rückwärtsdurchlauf das exakte Softmax-Gewicht $\alpha_j$ des Schlüssels $j$ allein aus seinem Score $a_j$ und dem einzigen gespeicherten Skalar $\ell = m + \log s$ rekonstruieren?

<!--en:a32b4380-->
2. **Überlauf in float16 ohne laufendes Maximum:** Sie rechnen in `float16` (größter endlicher Wert $65{,}504 \approx e^{11.09}$) und bilden direkt die Summe $\sum_j e^{a_j}$, ohne das laufende Maximum $m$ abzuziehen. Was geschieht, wenn ein skalierter Score $a_j = 12$ erreicht?

<!--en:c41e15d9-->
3. **Kachelgröße und Kopfbreite:** Für FlashAttention gilt die Schranke $\Theta(n^2 d_h^2 / M)$ für den HBM-Verkehr bei einer SRAM-Kapazität von $M$ auf dem Chip. Wie wirkt sich eine Verdopplung der Kopfbreite von $d_h = 64$ auf $d_h = 128$ (bei fester Gesamtbreite $d$) auf die maximale Kachelgröße $b$ und den HBM-Verkehr aus?

<!--en:7d30ecd2-->
::: {.slide title="Verständnisfragen zu Online-Softmax und Speicherverkehr (Antworten)"}

<!--en:929b7671-2-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:557b98b6-->
**Antworten:**

<!--en:152111f6-->
1. **$\alpha_j = \exp(a_j - \ell)$**, denn es gilt $\exp(a_j - m - \log s) = \exp(a_j - m) / s = \alpha_j$. Daher genügt $1$ Gleitkommazahl pro Abfrage anstelle des Paars $(m, s)$ oder aller $n$ Gewichte.

<!--en:ac9cb8c1-->
2. **Sofortiger Überlauf zu `inf`**, gefolgt von `inf / inf = NaN` bei der Normierung der Ausgabe. Führen wir $m$ mit, gilt stets $a_j - m' \le 0$. Damit liegt jeder Exponentialwert in $(0, 1]$.

<!--en:837d27ce-->
3. **Die maximale Kachelgröße $b \le M / (4d_h)$ halbiert sich.** Daher sind doppelt so viele Durchläufe über $\mathbf{K}, \mathbf{V}$ nötig. Der quadratische HBM-Verkehr steigt (bei fester Gesamtbreite insgesamt um den Faktor $2\times$, bei fester Kopfzahl um den Faktor $4\times$).

<!--en:9ef04be0-->
::: {.slide title="Zusammenfassung: Rechenkosten der Attention"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:4f2a98da-->
- **HBM als Engpass bei Speicherbedarf $O(n^2)$:** Die naive Attention belegt $8n^2$ Byte pro Kopf ($128\text{ GiB}$ bei $n = 128\text{K}$). Die HBM-Bandbreite begrenzt ihre Geschwindigkeit stark.

<!--en:2d71fd16-->
- **Rekursion des Online-Softmax-Algorithmus:** In einem einzigen Durchlauf aktualisiert er die laufenden Größen $(m, s, \mathbf{o})$ und skaliert die bisherigen Summen mit $e^{m - m'}$ neu.

<!--en:c71d304e-->
- **Kachelung bei FlashAttention:** Der Kernel fusioniert $\mathbf{Q}\mathbf{K}^\top$, die Softmax-Funktion und $\mathbf{P}\mathbf{V}$ im SRAM auf dem Chip. Er benötigt $\Theta(n)$ Speicher im HBM und überspringt leere, kausal maskierte Kacheln im oberen Dreieck.
