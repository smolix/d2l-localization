<!--en:837bd13b-->
# Attention

<!--en:86a1bc05-->
Die rekurrenten Modelle aus :numref:`chap_rnn` lesen eine Sequenz Token für Token. Alles bisher Gelesene verdichten sie zu einem versteckten Zustand (engl. hidden state) fester Größe. Nehmen wir an, eine Vorhersage gegen Ende eines langen Dokuments hängt von einem Detail aus dem ersten Absatz ab. Dann muss das rekurrente Netz dieses Detail schon in seinem versteckten Zustand gespeichert haben, lange bevor klar war, dass es darauf ankommt. Muss das Netz die gesamte Vorgeschichte in einem einzigen Vektor verdichten? Oder kann es an jeder Position eine Repräsentation speichern und gezielt abrufen, was es für die aktuelle Vorhersage braucht?

<!--en:07ae40d2-->
Genau das leistet *Attention*. Bei jedem Zugriff gibt ein Vektor an, was an der aktuellen Position gesucht wird. Er dient als *Abfrage* (engl. query). Für jede infrage kommende Position ergibt der Vergleich mit ihrem *Schlüssel* (engl. key), ebenfalls einem Vektor, einen Attention-Score. Die Softmax-Funktion wandelt die Scores in nichtnegative Gewichte um, deren Summe eins ist. Die Schicht gibt den gewichteten Mittelwert der an diesen Positionen gespeicherten *Werte* (engl. values) zurück, die ebenfalls als Vektoren vorliegen. Ein gewichteter Mittelwert ist eine glatte Funktion. Deshalb lassen sich die Projektionen, die Abfragen, Schlüssel und Werte erzeugen, mit Gradientenabstieg durchgängig trainieren. Führen wir mehrere solche Zugriffe parallel aus, erhalten wir *Multi-Head-Attention*. Ursprünglich sollte Attention bei der neuronalen maschinellen Übersetzung dem Decoder direkten Zugriff auf jedes Wort des Ausgangstextes geben :cite:`Bahdanau.Cho.Bengio.2014`. Im Transformer ist Attention die einzige Operation, die Informationen aus verschiedenen Positionen zusammenführt :cite:`Vaswani.Shazeer.Parmar.ea.2017`.

<!--en:ffdd19f3-->
Wenn wir die Rekurrenz durch einen Zugriff über alle Positionspaare ersetzen, stellen sich drei neue Herausforderungen:

<!--en:1cf1e970-->
1. **Unempfindlichkeit gegenüber der Reihenfolge:** Eine gewichtete Summe ist invariant gegenüber Permutationen ihrer Eingaben. Deshalb müssen wir die Positionsinformation ausdrücklich einbringen.

<!--en:b0f30b52-->
2. **Quadratische Skalierung:** Berechnen wir Scores für alle Paare von $n$ Token, wachsen die Zahl der Operationen und der Speicherbedarf wie $O(n^2)$. Das motiviert exakte Kernel, die Speichertransfers berücksichtigen, ebenso wie subquadratische Näherungen.

<!--en:4e253e03-->
3. **Mechanistische Interpretation:** Die Attention-Gewichte (engl. attention weights) zeigen, *wo* ein Attention-Kopf (engl. attention head) liest. Aus ihnen allein geht nicht hervor, *welche* Funktion er über mehrere Schichten hinweg berechnet.

<!--en:4bce203b-->
In den ersten drei Abschnitten bauen wir die Schicht auf. Wir beginnen mit dem differenzierbaren Zugriff auf Werte über ihre Schlüssel (:numref:`sec_queries-keys-values`). Dann berechnen wir Scores mit skalierten Skalarprodukten und wenden boolesche Masken an (:numref:`sec_attention-scoring-functions`). Darauf folgt Self-Attention mit mehreren Köpfen (:numref:`sec_multihead-attention`). :numref:`sec_positional-information` zeigt, wie absolute, rotationsbasierte (RoPE) und relative (ALiBi) Positionskodierungen die Reihenfolge der Sequenz erfassen. Die nächsten beiden Abschnitte untersuchen, wie die Kosten mit der Sequenzlänge wachsen. In :numref:`sec_attention-at-scale` leiten wir Kernel für die blockweise Online-Berechnung der Softmax-Funktion her (FlashAttention). Sie kommen ohne die explizite Speicherung der $n \times n$-Matrix der Scores aus. In :numref:`sec_windowed-linear-attention` untersuchen wir Attention mit gleitendem Fenster und Varianten der linearen Attention. In :numref:`sec_what-attention-computes` analysieren wir schließlich trainierte Attention-Schaltkreise anhand des Residualstroms (engl. residual stream), der Zerlegung in QK- und OV-Schaltkreise und der Induktionsköpfe (engl. induction heads), die über zwei Schichten wirken. :numref:`sec_attention-further-reading` gibt einen Überblick über Erweiterungen und Primärliteratur.

<!--en:41cabfbb-->
Wenn Sie direkt zu :numref:`chap_transformers` übergehen möchten, genügen die ersten vier Abschnitte und die Beschreibung mithilfe des Residualstroms zu Beginn von :numref:`sec_what-attention-computes`. Die Abschnitte zu den Rechenkosten können Sie zusammen mit :numref:`sec_kv-cache` und :numref:`chap_modern_rnn` lesen.

<!--en:bb504911-->
Attention-Mechanismen: differenzierbares inhaltsbasiertes Retrieval<br>**Abfragen, Schlüssel und Werte · skaliertes Skalarprodukt · Multi-Head-Attention und GQA · RoPE und ALiBi · FlashAttention · lineare Attention · Schaltkreise**

<!--en:359d614e-->
::: {.slide title="Vom Flaschenhals eines Zustands fester Größe zum direkten Zugriff"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:451a4c71-->
- **Der rekurrente Flaschenhals:** Ein Modell der Breite $d = 768$ (GPT-2 small) liest $n = 1{,}024$ Token. Es verdichtet $1{,}024 \times 768 \approx 786{,}000$ Zahlen aus der Vorgeschichte zu $768$ Gleitkommazahlen. Der Gradient durchläuft dabei eine Kette von $O(n)$ Schritten.

<!--en:4aaa32a2-->
- **Inhaltsbasiertes Retrieval:** An jeder Position eine Repräsentation speichern und bei Bedarf den relevanten Kontext abrufen.

<!--en:68e80f47-->
- **Differenzierbarer weicher Zugriff:** An die Stelle des unstetigen Abgleichs in einer Datenbank tritt ein mit der Softmax-Funktion gewichteter Mittelwert der gespeicherten Werte. So lässt sich der Zugriff mit Backpropagation durchgängig trainieren.

<!--en:91807bca-->
::: {.slide title="Die Grundoperation mit Abfragen, Schlüsseln und Werten"}

<!--en:053f73b8-->
[Grundprinzip]{.kicker}

<!--en:8abcaa58-->
- **Drei verschiedene Rollen:** Bei jedem Zugriff vergleichen wir eine **Abfrage** $\mathbf{q} \in \mathbb{R}^d$ (was die aktuelle Position sucht) mit $m$ **Schlüsseln** $\mathbf{k}_i \in \mathbb{R}^d$ (was die jeweilige Position anbietet).

<!--en:79df554e-->
- **Konvexkombination der Werte:** Die Softmax-Funktion wandelt die Scores $a(\mathbf{q}, \mathbf{k}_i)$ für die Übereinstimmung in nichtnegative Gewichte $\alpha_i$ mit Summe $1$ um:
  $$\textrm{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i)\,\mathbf{v}_i$$

<!--en:92664e51-->
- **Parallele Köpfe:** Ein Zugriff bildet einen **Kopf** der Attention-Schicht. Mit $H$ parallelen Köpfen kann ein Token mehrere Argumente gleichzeitig abrufen.

<!--en:058f7782-->
![Beim weichen Zugriff vergleichen wir eine Abfrage mit den Schlüsseln und erhalten eine Konvexkombination der Werte.](../img/mdl-attention-soft-lookup.svg){.nostretch width=100%}

<!--en:04721d10-->
::: {.slide title="Drei Herausforderungen beim Zugriff über alle Positionspaare"}

<!--en:a83731ae-->
[Zielkonflikte beim Entwurf]{.kicker}

<!--en:1763c6c1-->
- **1. Unempfindlichkeit gegenüber der Reihenfolge:** Eine gewichtete Summe $\sum_i \alpha_i \mathbf{v}_i$ ist permutationsinvariant und erfasst die Wortreihenfolge nicht. Deshalb brauchen wir Tabellen für absolute Positionen, **RoPE** oder **ALiBi** (**§{sec_positional-information}**).

<!--en:d74f2d0e-->
- **2. Quadratische Kosten:** Die Scores für alle Paare von $n$ Token erfordern $\Theta(n^2 d)$ FLOPs und bei naiver Implementierung $O(n^2)$ Speicher ($64\text{ GiB}$ bei $n=128\text{K}$). Das motiviert **FlashAttention** (**§{sec_attention-at-scale}**), gleitende Fenster und **lineare Attention** (**§{sec_windowed-linear-attention}**).

<!--en:022fa63b-->
- **3. Schwer durchschaubarer Mechanismus:** Attention-Gewichte zeigen, *wo* ein Kopf liest. Sie verraten nicht, *welche* Funktion er mithilfe von **QK- und OV-Schaltkreisen** über den Residualstrom berechnet (**§{sec_what-attention-computes}**).

<!--en:0ff6467b-->
::: {.slide title="Architektur und Aufbau von Kapitel §{num:chap_attention}"}

<!--en:99e0030a-->
[Kapitelübersicht]{.kicker}

<!--en:c19c3523-->
- **Teil 1: Aufbau der Schicht (§§{sec_queries-keys-values}–§{num:sec_multihead-attention}):** Weicher Zugriff auf Werte über ihre Schlüssel, Kernel-Pooling nach Nadaraya und Watson, mit $1/\sqrt{d}$ skalierte Skalarprodukte als Scores, boolesche Algebra für Masken sowie Multi-Head-Attention und Grouped-Query-Attention.

<!--en:6adc4d8f-->
- **Teil 2: Reihenfolge in der Sequenz (§{sec_positional-information}):** Permutationsäquivarianz, sinusförmige Rotationen, **RoPE**, **ALiBi** und Erweiterung des Kontexts ohne zusätzliches Training.

<!--en:7bce3df4-->
- **Teile 3 und 4: Skalierung und Schaltkreise (§§{sec_attention-at-scale}–§{num:sec_what-attention-computes}):** Blockweise Online-Berechnung der Softmax-Funktion (**FlashAttention**), gleitende Fenster, Rekurrenzen bei linearer Attention und **Induktionsköpfe**, die über zwei Schichten wirken.

<!--en:72b72fbb-->
::: {.slide title="Benchmarks im Kapitel und Hinweise zur Lektüre"}

<!--en:05adf916-->
[Roter Faden]{.kicker}

<!--en:5bb5449d-->
In jedem Abschnitt prüfen wir die Aussagen mit einem ausführbaren Benchmark. Als Referenz dient ein Kopf von GPT-2 small ($d = 768, H = 12, d_h = 64$), für den wir FLOPs und Bytes exakt zählen:

<!--en:d79273b6-->
| Abschnitt | Ausführbarer Benchmark oder Modell | Wichtigstes quantitatives Ergebnis |

<!--en:ba5609c8-->
| **§{sec_queries-keys-values}–§{num:sec_attention-scoring-functions}** | Eindimensionale Regression nach Nadaraya und Watson, Permutationsaufgabe mit `ToyAligner` | Die Bandbreite $\sigma$ und der Faktor $1/\sqrt{d}$ wirken als Softmax-Temperaturen und verhindern die Sättigung der Gradienten. |

<!--en:2e74e22e-->
| **§{sec_multihead-attention}** | Beispielsatz *„Alice gave Bob the book“*, **Copy-Both**-Aufgabe mit gaußverteilten Daten | Mit einem Kopf beträgt der relative MSE $\ge 1/2$, mit zwei Köpfen nur $\sim 10^{-7}$, bei jeweils $4d^2$ Parametern. |

<!--en:d443cbff-->
| **§{sec_positional-information}** | `TinyCharLM` auf den Bytes von WikiText-103 ($n = 128$ $\to$ $512$) | RoPE erzielt innerhalb des trainierten Längenbereichs die niedrigste Perplexität ($7.3$). Interpolation der langsam rotierenden Paare ermöglicht wieder gute Vorhersagen bei $4\times$ so langem Kontext. |

<!--en:446d11f9-->
| **§{sec_attention-at-scale}–§{num:sec_windowed-linear-attention}** | Blockweise Online-Berechnung der Softmax-Funktion, fusionierte Kernel für Flash/FlexAttention bis $n = 32\text{K}$ | Die Online-Berechnung der Softmax-Funktion senkt den Speicherbedarf von $O(n^2)$ auf $O(n)$. Mit Fensterbreite $w = 128$ ist die Berechnung $> 30\times$ so schnell. |

<!--en:e6b7ec00-->
| **§{sec_what-attention-computes}** | `TinyCharLM` auf sich wiederholenden Zufallsmustern mit variabler Periodenlänge | Zwei Blöcke lernen die Induktion im Kontext ($\sim 90\%$ Genauigkeit). Ein Block bleibt bei $\sim 10\%$. |

<!--en:fd9600fe-->
**Kurzer Weg zu Kapitel §{num:chap_transformers} (Transformer):** Lesen Sie **§{sec_queries-keys-values}–§{num:sec_positional-information}** und die Einführung in den Residualstrom zu Beginn von **§{sec_what-attention-computes}**. Lesen Sie **§{sec_attention-at-scale}–§{num:sec_windowed-linear-attention}** zusammen mit den Abschnitten zum Zwischenspeichern im KV-Cache und zu modernen RNNs.
