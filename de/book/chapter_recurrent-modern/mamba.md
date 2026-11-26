<!--en:74dd50bf-->
# Selektive Zustandsraummodelle

<!--en:71a1a449-->
Die S4D-Schicht aus :numref:`sec_ssm` behandelt alle Token mit demselben zeitlichen Abstand $k$ gleich. Eine Eingabe, die $k$ Schritte zurückliegt, trägt mit dem festen Faltungsgewicht $K_k$ zur aktuellen Ausgabe bei, ob sie nun das zentrale Substantiv eines Absatzes war oder ein Komma. Natürliche Sprache erfordert eine *inhaltsabhängige Auswahl*. Die meisten Token kann das Modell rasch verwerfen. Einige wenige muss es jedoch mit hohem Gewicht speichern und über Hunderte von Schritten behalten. Ihre Positionen unterscheiden sich von Satz zu Satz. Kann ein Zustandsraummodell für jedes Token entscheiden, was es wie lange behält?

<!--en:3df61953-->
Zwei Modelle aus früheren Abschnitten liefern je eine Hälfte der Antwort:

<!--en:d6694381-->
- `MinGRU` (:numref:`sec_linear-recurrence`) berechnet sein Vergessensgate aus dem aktuellen Eingabetoken. So kann es ein Token behalten und das nächste überschreiben. Allerdings beträgt die anfängliche Halbwertszeit der Gates nur einen Schritt, und jeder Kanal speichert nur einen skalaren Zustand.

<!--en:f8e4f3e1-->
- Bei `S4D` (:numref:`sec_ssm`) reichen die anfänglichen Halbwertszeiten von $1$ bis $\sim 700$ Schritten, verteilt auf $N$ Zustände pro Kanal. Die Dynamik ist jedoch zeitinvariant.

<!--en:c765368b-->
*Selektive* Zustandsraummodelle (engl. selective state space models), die :citet:`Gu.Dao.2023` mit der Mamba-Architektur eingeführt haben, verbinden beide Ansätze. Die Schrittweite $\Delta_t$ des diskretisierten Zustandsraummodells (SSM), der Schreibvektor $\mathbf{B}_t$ und der Lesevektor $\mathbf{C}_t$ hängen nun von der aktuellen Eingabe $\mathbf{x}_t$ ab. Für $\Delta_t \to 0$ bleibt der Zustand eingefroren, und ein informationsarmes Fülltoken hat keinen Einfluss. Bei großem $\Delta_t$ schreibt das Modell ein wichtiges Token in den Zustand und schwächt den bisherigen Inhalt ab. Da sich die Dynamik von Schritt zu Schritt ändert, entfällt die FFT-Faltung. Mit dem parallelen assoziativen Scan aus :numref:`sec_linear-recurrence` lässt sich die Schicht weiterhin in $O(\log_2 T)$ Runden trainieren. Im Folgenden vergleichen wir `S4D` und `MinGRU` beim selektiven Kopieren. Wir zeigen, dass ein selektives SSM mit 1 Zustand mathematisch einem Gate von `MinGRU` entspricht. Dann implementieren wir `SelectiveSSM` und `Mamba` mit einem eigenen speichereffizienten Rückwärtsscan (`adjoint_scan`). Wir untersuchen, was eine trainierte Mamba-Schicht in ihrem Zustand speichert, und bewerten sie auf WikiText-103 auf Byteebene.

<!--en:f8143667-->
## Selektives Kopieren

<!--en:7291086b-->
Der synthetische Benchmark zum *selektiven Kopieren* (engl. selective copying) von :citet:`Gu.Dao.2023` prüft gezielt, ob ein Modell Informationen abhängig vom Inhalt behält (:numref:`fig_selective_copy`). In unserem Versuch belegen zunächst $1{,}024$ Fülltoken (`0`) die Sequenz. An *zufällig* gewählten Positionen stehen stattdessen $4$ markierte Symbole, die wir gleichverteilt aus den Token `2..9` ziehen. Am Ende folgen $4$ Abfragetoken (`1`). Beim $j$-ten Abfragetoken muss das Modell das $j$-te markierte Symbol ausgeben, also die Reihenfolge einhalten (Genauigkeit bei zufälligem Raten: $1/8 = 0.125$). Die Symbolpositionen ändern sich zufällig von Sequenz zu Sequenz. Daher muss jeder zeitinvariante Filter scheitern, der seine Eingaben allein nach ihrem zeitlichen Abstand $T - s$ gewichtet.

<!--en:07424092-->
![Selektives Kopieren. Wenige markierte Symbole (farbig) stehen zwischen Fülltoken (grau). Ihre Positionen wechseln von Sequenz zu Sequenz. An den Abfragetoken muss das Modell die Symbole in ihrer ursprünglichen Reihenfolge ausgeben.](../img/mdl-modernrnn-selective-copy.svg)

<!--en:6f064d9f-->
Zur Bewertung beim selektiven Kopieren trainiert `train_copy` ein Netz über 1500 Adam-Schritte (`lr=3e-3`, Batchgröße 64) mit `d2l.train_queries` und misst die Genauigkeit der Abfragen an 1024 neuen Testsequenzen:

<!--en:dd4944da-->
Die Hilfsfunktion `copy_runs` trainiert eine Architektur in `num_runs=2` Läufen mit unabhängigen Zufallsstartwerten. Sie gibt die Parameteranzahl und die Testgenauigkeiten aus und liefert die trainierten Modelle zurück:

<!--en:24462ae2-->
Zunächst vergleichen wir `d2l.SeqLM`-Modelle mit zwei Blöcken der Breite $d = 48$. Als Sequenzmischer verwenden wir das zeitinvariante `d2l.S4D` oder das eingabeselektive `d2l.MinGRU`:

<!--en:ecb89d7b-->
Wie erwartet liegt das zeitinvariante `S4D` nahe am Zufallsniveau ($\le 0.20$). Das eingabeselektive `minGRU` erreicht je nach Lauf etwa $0.75$ bis $0.95$. Warum ist `minGRU` trotz der anfänglichen Halbwertszeit seiner Gates von nur einem Schritt erfolgreich? Beim Abruf mit einer festen Verzögerung von 1000 Schritten in :numref:`sec_ssm` reichte das nicht aus. Beim selektiven Kopieren liegen markierte Symbole gelegentlich nahe an den Abfragen. Die Gradienten über diese kurzen Abstände lehren die Gates nach und nach, Symbole über längere Folgen von Fülltoken zu behalten (Aufgabe :exref:`ex_mamba-short-memories`).

<!--en:c7b30ca8-->
## Selektive Dynamik

<!--en:13ac8514-->
### Eingabeabhängige Schrittweiten

<!--en:d15b8501-->
Wir bauen das selektive SSM in zwei Schritten auf. Betrachten wir zunächst einen Kanal mit einem einzigen Zustand ($N = 1$) und der Rate $\lambda = -1$. Mit einem Halteglied nullter Ordnung (engl. zero-order hold, ZOH, siehe :numref:`subsec_zoh`) gilt $h_t = e^{-\Delta} h_{t-1} + (1 - e^{-\Delta}) x_t$. Statt $\Delta$ als festen Parameter zu behandeln, berechnen wir $\Delta_t = \textrm{softplus}(s_t) = \log(1 + e^{s_t}) > 0$ aus einer linearen Projektion $s_t = \mathbf{w}^\top \mathbf{x}_t + b$ des aktuellen Tokens. Wenden wir die Exponentialfunktion auf $-\Delta_t$ an, erhalten wir die algebraische Identität:

<!--en:1e1018f9-->
Mit $z_t = \sigma(-s_t)$ wird die ZOH-Aktualisierung zu $h_t = z_t h_{t-1} + (1 - z_t) x_t$. Das ist genau die Konvexkombination von `MinGRU` (:numref:`subsec_mingru`, Satz 1 aus :citet:`Gu.Dao.2023`). Liefert ein Fülltoken $s_t = -4.6$, so gilt $\Delta_t \approx 0.01$ und $z_t = 0.99$. Damit bleiben $99\%$ des Zustands erhalten, die Halbwertszeit beträgt 69 Schritte. Liefert ein markiertes Symbol $s_t = 2$, so gilt $\Delta_t \approx 2.1$ und $z_t \approx 0.12$. Das neue Symbol überschreibt also den Zustand weitgehend. Mit der Parametrisierung des Gates durch $\Delta_t = \textrm{softplus}(s_t)$ lässt sich auch die Initialisierung von S4D auf mehreren Zeitskalen direkt übernehmen. Wählen wir den Bias $b$ so, dass $\textrm{softplus}(b) \in [0.001, 0.1]$ gilt, beginnt das Gate mit Halbwertszeiten von $7$ bis $693$ Schritten. Entscheidend ist, dass eine einzige Schrittweite $\Delta_{t,c}$ pro Kanal gleichzeitig $N$ Zustände mit unterschiedlichen Raten $\lambda_1, \ldots, \lambda_N$ steuern kann.

<!--en:b5f04c69-->
### Schreib- und Lesevektoren

<!--en:3bac1b6d-->
Wir geben nun jedem der $H$ Kanäle der Schicht (bei Mamba $H = 2d$) einen $N$-dimensionalen Zustandsvektor $\mathbf{h}_t^{(c)} \in \mathbb{R}^N$. Seine Raten $\boldsymbol{\lambda} = (\lambda_1, \ldots, \lambda_N)^\top$ initialisieren wir mit $(-1, \ldots, -N)^\top$. Aus der aktuellen $H$-dimensionalen Eingabe $\mathbf{x}_t$ berechnet die Schicht die Schrittweiten $\boldsymbol{\Delta}_t \in \mathbb{R}_+^H$ für die einzelnen Kanäle sowie eine gemeinsame Schreibrichtung $\mathbf{B}_t \in \mathbb{R}^N$ und Leserichtung $\mathbf{C}_t \in \mathbb{R}^N$:

<!--en:6dcae741-->
Jeder Kanal $c \in \{1, \ldots, H\}$ aktualisiert dann seine $N$ Zustände und liest daraus seine Ausgabe ab:

<!--en:e9179c3d-->
![Ein Schritt des selektiven SSM. Aus der Eingabe $\mathbf{x}_t$ mit $H$ Kanälen berechnen drei Projektionen die Schrittweiten $\boldsymbol{\Delta}_t$ (eine pro Kanal), den Schreibvektor $\mathbf{B}_t$ und den Lesevektor $\mathbf{C}_t$ (beide der Länge $N$). Zeile $c$ des Zustands der Form $H \times N$ klingt mit dem Faktor $e^{\Delta_{t,c}\boldsymbol{\lambda}}$ ab und erhält den Beitrag $\Delta_{t,c}\, x_{t,c}\, \mathbf{B}_t$. Für die Ausgabe $y_{t,c}$ lesen wir diese Zeile mit $\mathbf{C}_t$ aus und addieren den direkten Beitrag $D_c\, x_{t,c}$.](../img/mdl-modernrnn-selective-ssm.svg)

<!--en:5947f02d-->
:numref:`fig_selective_ssm` zeigt, wo die drei Projektionen wirken. $\boldsymbol{\Delta}_t$ skaliert die Zeilen des Zustands, eine pro Kanal. Dagegen wirken $\mathbf{B}_t$ und $\mathbf{C}_t$ entlang der Spalten und gelten für alle Kanäle gemeinsam. Wie bei :citet:`Gu.Dao.2023` verwendet :eqref:`eq_selective_ssm` den exakten ZOH-Abklingfaktor $e^{\Delta_{t,c}\boldsymbol{\lambda}}$. Das ZOH-Eingabegewicht $(e^{\Delta\lambda} - 1)/\lambda$ nähern wir durch das Euler-Gewicht erster Ordnung $\Delta_{t,c}$ an. Für $\Delta|\lambda| \ll 1$ stimmen die Gewichte näherungsweise überein. Große Schritte $\Delta_{t,c}$ erlauben Schreibgewichte über $1/|\lambda|$ (siehe Aufgabe :exref:`ex_mamba-approximate-write`). Die Tabelle zeigt die Formen aller Größen in :eqref:`eq_selective_ssm`:

<!--en:46679d73-->
| Größe | Form | Berechnet aus |

<!--en:2e3f1cfe-->
| Eingabe $\mathbf{x}_t$ | $H$ | Eingabe der Schicht im Schritt $t$ |

<!--en:835819c6-->
| Schrittweiten $\boldsymbol{\Delta}_t$ | $H$, eine pro Kanal | $\mathbf{x}_t$ über eine Faktorisierung vom Rang $R$ der Matrix $\mathbf{W}_\Delta \in \mathbb{R}^{H \times H}$ |

<!--en:5eddf2e4-->
| Schreib- und Lesevektor $\mathbf{B}_t$, $\mathbf{C}_t$ | je $N$, für alle Kanäle gemeinsam | $\mathbf{x}_t$ über $\mathbf{W}_B, \mathbf{W}_C \in \mathbb{R}^{N \times H}$ |

<!--en:10a71d35-->
| Raten $\boldsymbol{\lambda}$ | $N$ pro Kanal | gelernte Parameter mit Anfangswerten $-1, \ldots, -N$ |

<!--en:f54727a9-->
| Zustand $\mathbf{h}_t^{(1)}, \ldots, \mathbf{h}_t^{(H)}$ | $H \times N$ | Rekurrenz :eqref:`eq_selective_ssm` |

<!--en:0a63a890-->
| Ausgabe $\mathbf{y}_t$ | $H$ | Auslesen mit $\mathbf{C}_t$ plus direkter Beitrag $D_c x_{t,c}$ |

<!--en:11df5dd5-->
## Die Mamba-Schicht

<!--en:7838f754-->
### Projektionen, Faltung und Gate

<!--en:dd8885ff-->
Der `Mamba`-Block :cite:`Gu.Dao.2023` bettet das selektive SSM in eine Architektur mit drei Bausteinen ein: Projektionen, die Kanäle mischen, eine kanalweise kausale Faltung der Breite 4 und ein multiplikatives Ausgabegatter nach dem Vorbild von SwiGLU (:numref:`fig_mamba_block`):

<!--en:11710b31-->
1. Eine Eingabeprojektion $\mathbf{W}_\textrm{in}$ erweitert die $d$-dimensionale Eingabe auf zwei Zweige der Breite $H = 2d$.

<!--en:68104803-->
2. Im ersten Zweig durchläuft die Eingabe eine kanalweise kausale Faltung der Breite 4 und eine SiLU-Aktivierung, bevor sie das selektive SSM erreicht. Dadurch hängen $\boldsymbol{\Delta}_t, \mathbf{B}_t, \mathbf{C}_t$ von einem lokalen Fenster mit 4 Token ab und nicht nur von einem einzelnen Token.

<!--en:9b3c8a18-->
3. Wir multiplizieren die SSM-Ausgabe elementweise mit der SiLU-Ausgabe des zweiten Zweigs und projizieren das Ergebnis auf die Breite $d$ zurück. Dazu dient $\mathbf{W}_\textrm{out}$.

<!--en:fde4c352-->
4. Um Parameter zu sparen, faktorisieren wir $\mathbf{W}_\Delta \in \mathbb{R}^{H \times H}$ über einen Flaschenhals vom Rang $R = \lceil d / 16 \rceil$.

<!--en:ce92a697-->
![Die Mamba-Schicht. Eine Projektion teilt die Eingabe in zwei Zweige der Breite $H = 2d$ auf. Im ersten Zweig folgen eine kurze kausale Faltung, eine SiLU und das selektive SSM. Das SSM berechnet Schrittweiten, Schreib- und Lesevektoren aus seiner Eingabe. Im zweiten Zweig steuert eine SiLU, welcher Anteil des Ergebnisses durchgelassen wird. Eine abschließende Projektion führt zur Breite $d$ zurück.](../img/mdl-modernrnn-mamba-block.svg)

<!--en:7f9cb1aa-->
### Training mit einem Scan

<!--en:b56096ab-->
Auch :eqref:`eq_selective_ssm` ist in jedem Schritt eine affine Rekurrenz $\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$ für einen Zustand der Form $H \times N$. Daher berechnet `d2l.linear_scan` alle $T$ Zustände parallel. Mit der Erweiterung auf $H \times N$ Zahlen pro Token wächst der Aktivierungsspeicher allerdings um den Faktor $2N$. Die direkte Differentiation durch $\lceil \log_2 T \rceil$ Runden eines Verdopplungsscans hält $\sim 2\lceil \log_2 T \rceil$ Tensoren der Form $(B, T, H, N)$ im Speicher. Schon unser kleines Kopiermodell benötigt dafür $\sim 2\textrm{ GB}$ pro Schicht, ein großes Sprachmodell Hunderte von Gigabyte.

<!--en:75fe9810-->
Wir brauchen die Zwischenergebnisse der Scan-Runden nicht zu speichern. Dazu nutzen wir aus, dass der bei der Backpropagation berechnete Zustandsgradient $\mathbf{g}_t = \partial \ell / \partial \mathbf{h}_t$ die folgende, zeitlich rückwärts laufende affine Rekurrenz erfüllt:

<!--en:ba632191-->
Die zugehörigen Parametergradienten sind $\partial \ell / \partial \mathbf{b}_t = \mathbf{g}_t$ und $\partial \ell / \partial \mathbf{a}_t = \mathbf{g}_t \odot \mathbf{h}_{t-1}$. Der Rückwärtsdurchlauf besteht somit aus einem zweiten Aufruf von `linear_scan` in umgekehrter Zeitrichtung. Er benötigt aus dem Vorwärtsdurchlauf nur die Abklingfaktoren $\mathbf{a}$ und die Zustände $\mathbf{h}$ (zwei Tensoren statt $\sim 20$). Der fusionierte CUDA-Kernel von Mamba geht noch weiter. Er hält $\mathbf{h}$ im SRAM auf dem Chip und berechnet die Zustände im Rückwärtsdurchlauf neu, wie in :numref:`sec_memory_precision` beschrieben. Implementieren wir `adjoint_scan` mit einem eigenen Rückwärtsdurchlauf und prüfen wir die Ergebnisse anhand der üblichen automatischen Differentiation.

<!--en:887f27cc-->
An Zufallssequenzen der Länge $T = 100$ prüfen wir, ob die Ableitungen von `adjoint_scan` bis auf float32-Genauigkeit mit denen der automatischen Differentiation durch `d2l.linear_scan` übereinstimmen:

<!--en:d4f06fae-->
Der Gradient aus unserem Rückwärtsscan stimmt bis auf $\sim 10^{-6}$ mit dem der automatischen Differentiation überein.

<!--en:869d223f-->
### Implementierung

<!--en:70333e7b-->
`SelectiveSSM` implementiert :eqref:`eq_selective_ssm`. Wir initialisieren `dt_bias` so, dass $\textrm{softplus}(\textrm{dt\_bias})$ auf $[\textrm{dt\_min}, \textrm{dt\_max}] = [0.001, 0.1]$ logarithmisch gleichverteilt ist. Mit `selective=False` ersetzen wir $\boldsymbol{\Delta}_t, \mathbf{B}_t, \mathbf{C}_t$ durch gelernte, für jeden Kanal konstante Werte. So erhalten wir eine lineare zeitinvariante Kontrollvariante im ansonsten identischen Mamba-Block.

<!--en:4ffe2f84-->
Als Nächstes ergänzen wir `log_coefficients`. Die Methode berechnet für jeden Schritt die logarithmierten Abklingfaktoren $\Delta_{t,c}\boldsymbol{\lambda}$ und die Schreibbeiträge $\Delta_{t,c} u_{t,c}\mathbf{B}_t$, jeweils in der Form `(batch_size, T, H, N)`. Der Vorwärtsdurchlauf der Schicht ruft `adjoint_scan` auf und kontrahiert die Zustände mit $\mathbf{C}_t$:

<!--en:0ec0e493-->
Das Modul `Mamba` bettet `SelectiveSSM` in die zweigeteilte Architektur aus :numref:`fig_mamba_block` ein. Es erweitert die Breite $d$ auf $2H = 4d$ mithilfe von `W_in`. Im SSM-Zweig wendet es die kanalweise kausale Faltung `conv` und SiLU an. Es gewichtet das Ergebnis mit `silu(Z)` und projiziert es mit `W_out` zurück:

<!--en:4134d059-->
## Trainierte Mamba-Modelle

<!--en:6b8f7e8a-->
### Selektives Kopieren mit Mamba

<!--en:9d5b5988-->
Wir trainieren nun `Mamba` ($N = 4$ Zustände pro Kanal) und seine nichtselektive Variante `Mamba, not selective` auf selektives Kopieren. Bei dieser Ablation (`selective=False`) bleiben die Projektionen, die Faltung der Breite 4 und das SiLU-Gate identisch:

<!--en:2a4d0b44-->
`Mamba` kopiert fast jedes Symbol. Die Genauigkeit beträgt mindestens $95\%$ und liegt meist nahe bei $100\%$, also höher als bei `minGRU`. Die nichtselektive Kontrollvariante verliert mehr als $30$ Prozentpunkte, und ihre Ergebnisse schwanken stark von Lauf zu Lauf. Das entspricht dem Ergebnis der Ablation in :citet:`Gu.Dao.2023`. Beide Modelle haben dieselben Projektionen, dieselbe Faltung und dasselbe Gate. Der Unterschied geht also allein auf die Auswahl zurück. In Aufgabe :exref:`ex_mamba-longer-sequences` untersuchen Sie, ob das Modell ohne weiteres Training auf Sequenzen mit $4{,}096$ Fülltoken extrapoliert.

<!--en:d256100e-->
### Beiträge zum Zustand

<!--en:26e52489-->
Um zu verstehen, *wie* `Mamba` die vier markierten Symbole behält und die Fülltoken übergeht, zerlegen wir den Endzustand am ersten Abfragetoken in die Beiträge der früheren Positionen $s \le T$. Betrachten wir die entfaltete Rekurrenz $\mathbf{h}_T = \sum_{s=1}^T (\prod_{r=s+1}^T \mathbf{a}_r) \odot \mathbf{b}_s$. Für Position $s$ berechnen wir die euklidische Norm des abgeklungenen Terms, also $\|\exp(\sum_{r=s+1}^T \log \mathbf{a}_r) \odot \mathbf{b}_s\|_2$. Sie misst, wie viel vom Schreibbeitrag der Position $s$ im Schritt $T$ noch im Zustand enthalten ist.

<!--en:6d8bc512-->
Wir ergänzen `log_coefficients` in `d2l.S4D`. Damit erhalten wir die logarithmierten Abklingfaktoren und die Schreibbeiträge jedes Schritts im selben Format `(batch_size, T, d, N)` wie bei `SelectiveSSM`:

<!--en:ca84bbf6-->
Welche der beiden Schichten sollen wir untersuchen? Das Training gibt dem Modell nicht vor, welche Schicht die Auswahl treffen soll. Deshalb untersuchen wir beide. Die Funktion `layer_contributions` führt die Blöcke unterhalb der untersuchten Schicht aus. Für eine Testsequenz `X`, die mit dem ersten Abfragetoken endet (`X[:, :-3]`), entnimmt sie der Schicht die logarithmierten Abklingfaktoren und die Schreibbeiträge und wertet `state_contributions` aus. Wir wenden sie auf beide Schichten von `S4D` und auf beide Schichten der zwei trainierten `Mamba`-Modelle an:

<!--en:6a43e6bd-->
Die kausale Faltung der Breite 4 verteilt jedes markierte Symbol auf bis zu 4 aufeinanderfolgende Positionen. Daher erfasst `symbol_windows` für jedes Symbol das an seiner Position beginnende Fenster der Länge 4 (insgesamt $\le 1.6\%$ der Sequenz). Für jede Schicht geben wir aus, welcher Anteil des Endzustands aus diesen Fenstern stammt. Dann plotten wir für jedes Modell die Beiträge aller $1{,}025$ Positionen in der Schicht mit dem größeren Anteil:

<!--en:dd011d86-->
Dieser Anteil hängt davon ab, wo die Symbole in einer einzelnen Sequenz zufällig liegen. Ein lokales Maß ist das *Spitzenverhältnis* (engl. spike ratio). Dazu teilen wir den größten Beitrag im Fenster eines Symbols von 4 Schritten durch den Beitrag des unmittelbar vorhergehenden Fülltokens. Wieder berechnen wir den Wert für beide Schichten und bilden jeweils den Median über die Symbole aus 64 neuen Testsequenzen:

<!--en:cb5123e9-->
Bei `S4D` zählt vor allem, wie kurz ein Token zurückliegt. Das Profil steigt zum Sequenzende hin gleichmäßig an. Die Symbolfenster liefern nur wenige Prozent des Zustands, kaum mehr als ihren Anteil an den Positionen. Ein Symbol hebt sich höchstens um einen Faktor von einigen Zehn vom vorhergehenden Fülltoken ab. Ein fester Filter kann auf ein Token stärker reagieren als auf ein anderes, es aber nicht länger behalten. In einer selektiven Schicht von `Mamba` tragen Symbole dagegen mehrere Hundert bis mehrere Zehntausend Mal so viel bei wie benachbarte Fülltoken. Die vier Fenster liefern einen großen Teil des Zustands, oft sogar den größten. Welche Schicht diese Auswahl lernt, steht vor dem Training nicht fest. In manchen Läufen treffen beide Schichten eine Auswahl. In anderen übernimmt das nur die erste oder nur die zweite Schicht. Das Spitzenverhältnis der anderen Schicht ist dann höchstens so groß wie bei `S4D`. Deshalb führen die Tabellen beide Schichten auf, während das Diagramm die jeweils selektivere zeigt.

<!--en:b5f6e63a-->
Wie erzeugt eine Schicht eine solche Spitze? Es liegt nahe, dass sie bei jedem Symbol einen großen Schritt macht, der das Symbol schreibt und die vorherigen Fülltoken löscht. Die Schrittweiten der trainierten Modelle zeigen jedoch etwas anderes. Im Mittel über die $H = 96$ Kanäle ist der Schritt bei einem Symbol meist nicht größer als bei einem Fülltoken. Nur wenige Kanäle treffen die Auswahl. Zwei oder drei von ihnen tragen das jeweilige Symbol. Bei Fülltoken liegen ihre Schrittweiten $\Delta_{t,c}$ bei etwa $10^{-5}$. Dadurch klingt ihr Zustand praktisch nicht ab, und es kommt kein nennenswerter Schreibbeitrag hinzu. Der Zustand bleibt eingefroren. Beim Symbol wählen dieselben Kanäle Schrittweiten zwischen etwa $0.1$ und $1.5$. Da der Schreibbeitrag $\Delta_{t,c}\, x_{t,c}\, \mathbf{B}_t$ proportional zur Schrittweite ist, geht das Symbol mit vollem Gewicht in den Zustand ein. Ein solcher Schritt schwächt auch den bisherigen Kanalinhalt ab, allerdings nur geringfügig. Das Löschen der vorherigen Fülltoken trägt daher wenig zur Spitze bei. Die Auswahl beruht hier darauf, dass wenige, sonst eingefrorene Kanäle das Symbol mit hohem Gewicht schreiben. Der Zustand wird dabei nicht geleert.

<!--en:adb5ad39-->
### Sprachmodellierung

<!--en:986607fd-->
Abschließend vergleichen wir `Mamba`, `minGRU` und `attention` auf WikiText-103 auf Byteebene (`d2l.SeqLM`, 4 Blöcke der Breite $d = 256$, Fenster von 128 Byte, 1000 Schritte). Zunächst laden wir den Datenstrom von `WikiText103` in Fenstern zu 128 Byte:

<!--en:ac9e82df-->
Dann definieren wir `text_model(mixer, d)` und konfigurieren die vier Versuchsvarianten in `arms`. Dazu gehört `minGRU, d=309` mit $3.99\textrm{M}$ Parametern. Diese Variante hat ebenso viele Parameter wie `Mamba` mit seinen zusätzlichen Kanälen:

<!--en:6d9eadb8-->
Mit `d2l.text_table` trainieren wir jede Architektur mit zwei Zufallsstartwerten über je $1{,}000$ Schritte. Die Tabelle zeigt den Validierungsverlust nach Positionsbereichen im Fenster sowie die Latenz pro Schritt:

<!--en:72084a19-->
`Mamba` erreicht $1.30$ Nats/Byte und damit denselben Wert wie die vollständigen Modelle `LSTM` und `GRU` aus :numref:`subsec_deep-rnn`, obwohl das Training einen parallelen Scan nutzt. Gegenüber `minGRU` gleicher Breite sinkt der Verlust um $0.05$ Nats/Byte, gegenüber `minGRU, d=309` mit gleicher Parameteranzahl um $0.03$ Nats/Byte. Der Scan von `Mamba` verarbeitet pro Token einen Zustand mit $H \times N = 512 \times 16 = 8{,}192$ Zahlen. Das ist $32\times$ so viel wie bei `minGRU` mit 256 Zahlen. Ohne fusionierten Kernel dauert ein Trainingsschritt daher $2\times$ bis $4\times$ so lange wie bei `minGRU`. Bei der Inferenz entsprechen die $32d$ Zahlen des Zustands und die $6d$ Zahlen im Puffer der Faltung der Breite 4 aber nur dem KV-Cache von 19 Token bei Multi-Head-Attention. Den Mamba-Zustand der Form $H \times N$ können wir auch als Matrixspeicher $\mathbf{S}_t$ auffassen. Das äußere Produkt $(\boldsymbol{\Delta}_t \odot \mathbf{x}_t)\mathbf{B}_t^\top$ aktualisiert ihn, und $\mathbf{C}_t$ fragt ihn ab. Diese Sicht führt direkt zu den Rekurrenzen mit Matrixzustand und zum Training mit blockweiser Matrixmultiplikation in :numref:`sec_matrix-state`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:f723656f-->
Zeitinvariante Zustandsraummodelle gewichten frühere Eingaben allein nach ihrem zeitlichen Abstand. Daher scheitern sie an Aufgaben wie dem selektiven Kopieren, die eine inhaltsabhängige Auswahl erfordern. Selektive Zustandsraummodelle berechnen aus jedem Eingabetoken die Schrittweiten $\boldsymbol{\Delta}_t = \textrm{softplus}(\mathbf{W}_\Delta \mathbf{x}_t + \mathbf{b}_\Delta)$ pro Kanal sowie die gemeinsamen Schreib- und Lesevektoren $\mathbf{B}_t, \mathbf{C}_t$. Bei einem einzigen Zustand ($N = 1$) ergibt $e^{-\textrm{softplus}(s_t)} = \sigma(-s_t)$ wieder das Update-Gatter von `MinGRU`. Initialisieren wir $N$ Zustände mit den Raten $-1, \ldots, -N$ und wählen die Bias-Werte so, dass $\Delta$ logarithmisch gleichverteilt ist, verbindet `Mamba` die inhaltsabhängige Auswahl mit einem Gedächtnis über mehrere Zeitskalen. Differenzieren wir den parallelen Scan mit einem Rückwärtsscan (`adjoint_scan`), müssen wir die Zwischenergebnisse der $O(\log_2 T)$ Verdopplungsrunden nicht speichern. Beim selektiven Kopieren erreicht `Mamba` eine Genauigkeit von mindestens $95\%$. Ohne Selektivität sinkt sie um $>30$ Prozentpunkte. Für diese Genauigkeit genügen wenige Kanäle, die bei Fülltoken eingefroren bleiben und jedes Symbol mit einem großen Schritt schreiben. Auf WikiText-103 auf Byteebene erreicht das Modell denselben Verlust wie die vollständigen LSTM- und GRU-Modelle und unterbietet `MinGRU` mit gleicher Parameteranzahl um $0.03$ Nats/Byte.

<!--en:2294243a-->
## Übungen

<!--en:a04cb1a8-->
1. **Näherung für den Schreibbeitrag.** Mamba behält den Abklingfaktor $e^{\Delta\lambda}$ des Halteglieds nullter Ordnung bei, schreibt aber mit dem Gewicht $\Delta$ anstelle von $(e^{\Delta\lambda} - 1)/\lambda$.
   :exlabel:`ex_mamba-approximate-write`
    1. Bestimmen Sie für $\lambda = -1$ die obere Grenze der Schrittweiten $\Delta$, bei denen sich die beiden Gewichte um weniger als 10% unterscheiden.
    1. Betrachten Sie einen Zustand mit $\lambda = -1$, dem Schreibvektor 1 und dem exakten Gewicht. Zeigen Sie, dass ein bei null beginnender Zustand in dem Intervall bleibt, das null und alle bisherigen Eingaben aufspannen. Finden Sie eine Folge von Eingaben aus $[-1,
       1]$, für die der Zustand mit dem Gewicht $\Delta$ dieses Intervall verlässt. Was hält die Werte eines trainierten Mamba-Modells in ihrem Wertebereich?

<!--en:bc69a30c-->
1. **Speicherbedarf des Scans.** Betrachten Sie eine Mamba-Schicht mit $H$ Kanälen und $N$ Zuständen, einen Minibatch aus $B$ Sequenzen der Länge $T$ und 32-Bit-Zahlen. Bestimmen Sie, wie viele Byte die Backpropagation für den Scan speichert: (a) bei Differentiation durch die Runden des Verdopplungsscans, (b) beim adjungierten Scan und (c) bei einem Kernel, der nur die Eingaben der Schicht speichert und die Zustände neu berechnet. Werten Sie alle drei Fälle für $H = 4096$, $N = 16$, $B = 8$ und $T =
   8192$ aus. Vergleichen Sie die Ergebnisse mit der Speicherkapazität einer GPU.

<!--en:690e2632-->
1. [code] **Längere Sequenzen.** Das Mamba-Modell wurde mit 1024 Positionen vor den Abfragen trainiert. Testen Sie es ohne weiteres Training auf Sequenzen mit 2048 und 4096 solchen Positionen (`num_fillers`).
   :exlabel:`ex_mamba-longer-sequences`
    1. Kopiert das Modell die Symbole weiterhin? Vergleichen Sie es mit der minGRU.
    1. Berechnen Sie für jeden Kanal der selektiveren Schicht, wie stark sein langsamster Zustand über 1000 Fülltoken abklingt. Wie viele Kanäle können ein Symbol über 4096 Fülltoken hinweg behalten?

<!--en:1c64297d-->
1. [code] **Welcher Teil muss selektiv sein?** ●
    1. Zwischen den beiden Mamba-Modellen dieses Abschnitts liegen zwei weitere Varianten. Die erste hat eingabeabhängige Schrittweiten und feste Schreib- und Lesevektoren $\mathbf{B}$ und $\mathbf{C}$. Die zweite hat feste Schrittweiten, aber eingabeabhängige Vektoren $\mathbf{B}_t$ und $\mathbf{C}_t$. Trainieren Sie beide Varianten je zweimal. Welche lösen die Aufgabe? Setzen Sie das Ergebnis zu den oben gemessenen Spitzenverhältnissen in Beziehung.
    1. Ersetzen Sie im trainierten Mamba-Modell die Schrittweiten beider Schichten in den vier Positionen ab jedem Symbol (den oben untersuchten Fenstern) durch ihren Median über die Füllpositionen. Damit kann keine Schicht die Symbole mit einem großen Schritt schreiben. Um wie viel sinkt die Genauigkeit? Ersetzen Sie im nächsten Versuch die Schrittweiten an allen anderen Positionen durch ihren Median über die Symbolfenster. Damit kann keine Schicht die Fülltoken überspringen. Verändern Sie anschließend jeweils nur eine Schicht. Stimmt das Ergebnis mit den oben gemessenen Anteilen überein?

<!--en:81b109d3-->
1. [code] **Generierung.** Ergänzen Sie eine Methode `step` in `SelectiveSSM` und in `Mamba`. Sie soll ausgehend vom SSM-Zustand und den letzten drei Eingaben der Faltung ein einzelnes Token verarbeiten und die Ausgabe sowie den neuen Zustand zurückgeben. Prüfen Sie, ob die schrittweise Verarbeitung einer Sequenz die Ausgaben des Vorwärtsdurchlaufs reproduziert. Wie viele Zahlen gibt die Schicht von Token zu Token weiter?

<!--en:c8772a6f-->
1. [code] **Kurzes Gedächtnis zu Trainingsbeginn.** Die minGRU hat das Kopieren gelernt, obwohl ihre Gates anfangs nur ein Gedächtnis von etwa einem Schritt haben. Prüfen Sie die Erklärung, dass die Gates von den Symbolen lernen, die den Abfragen am nächsten liegen.
   :exlabel:`ex_mamba-short-memories`
    1. Zeichnen Sie vor und nach dem Training die Gate-Werte der minGRU bei Fülltoken auf. Bestimmen Sie für das trainierte Modell die Genauigkeit für jede Abfrage getrennt, aufgeschlüsselt nach dem Abstand des Symbols zu seiner Abfrage. Welche Symbole kopiert es am besten?
    1. Trainieren Sie das Mamba-Modell mit anfänglichen Schrittweiten zwischen 0.5 und 1 (`dt_min=0.5, dt_max=1.0`). Damit hat jeder Zustand anfangs ein Gedächtnis von höchstens zwei Schritten. Lernt das Modell innerhalb des Trainingsbudgets, die Symbole zu kopieren? Falls ja, plotten Sie die Schrittweiten der ersten Schicht bei Fülltoken vor und nach dem Training.
    1. Wiederholen Sie beide Versuche mit 4096 Positionen vor den Abfragen, sodass ein Symbol seltener nahe an einer Abfrage liegt. Initialisieren Sie die minGRU dabei wie im Experiment zum Abruf über lange Zeitabstände in :numref:`sec_ssm`.

<!--en:5b6bf7a0-->
[Diskussion](https://d2l.discourse.group/t/36782)

<!--en:c42b91b5-->
Selektive Zustandsraummodelle (Mamba)<br>**Selektives Kopieren · eingabeabhängige Schrittweiten · Dualität von Softplus und Sigmoidfunktion · adjungiertes Vektor-Jacobi-Produkt per Rückwärtsscan · Zustandsbeiträge**

<!--en:8c4b3d87-->
::: {.slide title="Warum zeitinvariante SSMs bei der inhaltsabhängigen Auswahl scheitern"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:79883ac6-->
- **Grenze fester Faltungskerne:** In §{sec_ssm} gewichtet `S4D` jedes Token, das $k$ Schritte zurückliegt, mit demselben Faltungskoeffizienten $K_k$. Dabei kann das Token eine zentrale Entität bezeichnen oder nur als Fülltoken dienen.

<!--en:73347de6-->
- **Natürliche Sprache erfordert inhaltsabhängige Auswahl:** Beim Lesen müssen wir wichtige Substantive fest im Gedächtnis verankern und über unterschiedlich lange Teilsätze hinweg behalten. Fülltoken dazwischen können wir übergehen.

<!--en:8308fd11-->
- **Warum keine LTI-Faltung selektiv kopieren kann:** Stehen die Zielsymbole in jeder Sequenz an anderen, zufälligen Positionen, kann kein festes, nur vom Abstand abhängiges Gewicht $K_k$ die Zielsymbole erfassen.

<!--en:3197f7d5-->
::: {.slide title="Die Selektivität von minGRU und die Zeitskalen von S4D"}

<!--en:ed2509af-->
[Zwei Ansätze in einem Modell]{.kicker}

<!--en:3653a55f-->
- **Was `MinGRU` (§{sec_linear-recurrence}) bietet:** ein eingabeabhängiges Gate $\mathbf{z}_t(\mathbf{x}_t)$, aber nur $N = 1$ Skalar pro Kanal mit anfänglich $\tau_{1/2} = 1$.

<!--en:01ffa86d-->
- **Was `S4D` (§{sec_ssm}) bietet:** $N = 16$ Abklingmoden pro Kanal in logarithmisch gleichen Abständen ($\tau_{1/2} \in [0.4, 693]$), aber eine feste LTI-Schrittweite $\Delta_c$.

<!--en:3eed9347-->
- **Was `Mamba` (§{sec_mamba}) verbindet:** tokenabhängige $(\boldsymbol{\Delta}_t, \mathbf{B}_t, \mathbf{C}_t)$ für einen Zustand der Form $H \times N$ mit mehreren Zeitskalen. Das Training nutzt parallele und adjungierte Scans.

<!--en:97bc5d37-->
::: {.slide title="Selektives Kopieren bei zufälligen Symbolabständen"}

<!--en:00ffc4b0-->
[Synthetischer Belastungstest]{.kicker}

<!--en:0f986767-->
`copy_batch` prüft die inhaltsabhängige Auswahl über einen Horizont von $1{,}028$ Token:

<!--en:2393cf88-->
- **Aufbau der Sequenz:** Unter $1{,}024$ Fülltoken (`0`) stehen $4$ markierte Symbole (gleichverteilt aus den Token `2..9` gezogen) an **zufälligen Positionen**. Es folgen $4$ Abfragetoken (`1`) an den Positionen $1{,}025 \ldots 1{,}028$.

<!--en:b06ff76c-->
- **Vorhersageziel:** Beim $j$-ten Abfragetoken das $j$-te markierte Symbol ausgeben (Genauigkeit beim Raten $= 1/8 = 0.125$).

<!--en:5ad85082-->
::: {.slide title="Vergleich von zeitinvariantem S4D und selektiver minGRU"}

<!--en:d48e8fc2-->
[Ergebnisse der Baselines]{.kicker}

<!--en:c6827c56-->
Wir trainieren `SeqLM` mit 2 Blöcken ($d = 48$) über $1{,}500$ Adam-Schritte ($\text{lr} = 3 \times 10^{-3}$, Batchgröße $64$, je zwei Läufe):

<!--en:498aa7b0-->
- **`S4D` bleibt nahe am Zufallsniveau ($\le 0.20$):** Feste, nur vom Abstand abhängige Gewichte können Symbolen nicht folgen, wenn deren Positionen zufällig wechseln.

<!--en:d747e48e-->
- **Warum `minGRU` trotz anfänglicher Gate-Halbwertszeiten von 1 Schritt erfolgreich ist ($\approx 0.75$ bis $0.95$):** Beim Abruf mit fester Verzögerung von $1{,}000$ Schritten (§{sec_ssm}) ist der Abstand immer gleich. Hier liegen markierte Symbole gelegentlich nur wenige Schritte vor den Abfragen. Gradienten über kurze Abstände lehren die Gates nach und nach, bei Fülltoken `0` geschlossen zu bleiben.

<!--en:8aaa421c-->
::: {.slide title="Exakte Dualität zwischen Softplus-Schrittweiten und Sigmoid-Gates"}

<!--en:55c9f799-->
[Satz 1 (Gu & Dao, 2023)]{.kicker}

<!--en:1c97577e-->
- **Exakte Äquivalenz in einer Dimension ($N = 1, \lambda = -1$):** Diskretisieren wir $h'(t) = -h(t) + x(t)$ mit der Schrittweite $\Delta_t = \textrm{softplus}(s_t)$, erhalten wir $h_t = z_t h_{t-1} + (1 - z_t)x_t$ (`MinGRU`).

<!--en:ee45d29d-->
- **Variabler Takt:** Fülltoken ($s_t = -4.6 \implies \Delta_t \approx 0.01, z_t \approx 0.99$) frieren den Zustand ein, markierte Symbole ($s_t = +2.0 \implies \Delta_t \approx 2.13, z_t \approx 0.12$) überschreiben ihn.

<!--en:45d00d72-->
::: {.slide title="Selektive SSM-Dynamik mit Δt, Bt und Ct pro Token"}

<!--en:34f219aa-->
[Gleichungen des selektiven SSM]{.kicker}

<!--en:48b69c47-->
- **Eingabeabhängige Projektionen:** Aus $\mathbf{x}_t \in \mathbb{R}^H$ berechnet eine Projektion vom Rang $R$ die Schrittweiten $\boldsymbol{\Delta}_t \in \mathbb{R}_+^H$. Weitere Projektionen liefern die gemeinsamen Vektoren $\mathbf{B}_t, \mathbf{C}_t \in \mathbb{R}^N$.

<!--en:0377c6c0-->
- **Euler-Schreibgewicht:** Der exakte ZOH-Abklingfaktor $e^{\Delta_{t,c}\boldsymbol{\lambda}}$ bleibt erhalten. Der Schreibterm $\Delta_{t,c} x_{t,c}\mathbf{B}_t$ ist von erster Ordnung.

<!--en:8cfe9031-->
::: {.slide title="Tensorformen und Schrittweitenprojektion mit niedrigem Rang"}

<!--en:b6285020-->
[Parametrisierung]{.kicker}

<!--en:c164efbf-->
Jeder Tensor in `SelectiveSSM` hat eine bestimmte Funktion für die $H = 2d$ Kanäle nach der Erweiterung und die $N$ Zustandsdimensionen:

<!--en:94d07df3-->
| Größe | Form | Parametrisierung oder Berechnung |

<!--en:0d86ba93-->
| **Kanaleingabe** $\mathbf{x}_t$ | $(H,)$ | Ausgabe der kausalen Faltung der Breite 4 mit anschließender SiLU im erweiterten Zweig |

<!--en:646bbaa1-->
| **Schrittweiten** $\boldsymbol{\Delta}_t$ | $(H,)$ | $\textrm{softplus}(\mathbf{W}_\Delta \mathbf{x}_t + \mathbf{b}_\Delta)$ mit Flaschenhals vom Rang $R$: $\mathbf{W}_\Delta \in \mathbb{R}^{H \times R} \times \mathbb{R}^{R \times H}$, $R = \lceil d/16 \rceil$ |

<!--en:e8bcd9a2-->
| **Anfänglicher Bias** $\mathbf{b}_\Delta$ | $(H,)$ | so gewählt, dass $\textrm{softplus}(\mathbf{b}_\Delta) \in [0.001, 0.1]$ logarithmisch gleichverteilt ist (`dt + log(-expm1(-dt))`) |

<!--en:b8c2ab85-->
| **Schreib- und Lesevektor** $\mathbf{B}_t, \mathbf{C}_t$ | je $(N,)$ | lineare Projektionen $\mathbf{W}_B \mathbf{x}_t$ und $\mathbf{W}_C \mathbf{x}_t$, gemeinsam für alle $H$ Kanäle |

<!--en:ef5a1a73-->
| **Abklingraten** $\boldsymbol{\lambda}$ | $(H, N)$ | $-\exp(\text{log\_neg\_rate})$, in jedem Kanal mit $-1, -2, \ldots, -N$ initialisiert |

<!--en:6789646f-->
| **Rekurrenter Zustand** $\mathbf{h}_t$ | $(H, N)$ | Aktualisierung durch elementweises Abklingen mit $e^{\Delta_{t,c}\lambda_{c,n}}$ und Schreiben mit dem äußeren Produkt $(\boldsymbol{\Delta}_t \odot \mathbf{x}_t)\mathbf{B}_t^\top$ vom Rang 1 |

<!--en:59c4e371-->
::: {.slide title="Aufbau des vollständigen Mamba-Blocks"}

<!--en:25c78882-->
[Blockarchitektur]{.kicker}

<!--en:8a324631-->
1. **Erweiterung der Eingabe ($\mathbf{W}_\textrm{in}$):** Die Projektion bildet die Breite $d$ auf zwei Zweige $(\mathbf{U}, \mathbf{Z})$ mit erweiterter Breite $H = 2d$ ab.

<!--en:9c5b3585-->
2. **Kurze kausale Faltung:** Eine kanalweise kausale `Conv1d` der Breite $4$ mit anschließendem $\textrm{SiLU}$ mischt $4$ aufeinanderfolgende Token vor der Berechnung von $\boldsymbol{\Delta}_t, \mathbf{B}_t, \mathbf{C}_t$. So berücksichtigt die Auswahl den lokalen Kontext.

<!--en:490d7a79-->
3. **Selektives SSM und Gate nach dem Vorbild von SwiGLU:** Das elementweise Produkt $\textrm{SSM}(\mathbf{U}) \odot \textrm{SiLU}(\mathbf{Z})$ wird auf die Breite $d$ zurückprojiziert. Dazu dient $\mathbf{W}_\textrm{out}$.

<!--en:a776225f-->
::: {.slide title="Hoher Aktivierungsspeicherbedarf bei vierdimensionalen Scans"}

<!--en:2fc3255e-->
[Speicher als Flaschenhals]{.kicker}

<!--en:97b78b22-->
- **Keine Abkürzung per FFT:** Zeitlich veränderliches $\boldsymbol{\Delta}_t$ schließt eine LTI-Faltung aus. Wir brauchen einen vierdimensionalen parallelen Scan über $(B, T, H, N)$.

<!--en:7681259a-->
- **32× so großer Zustand ($H = 2d, N = 16$):** Bei direkter automatischer Differentiation bleiben $\sim 20$ Zwischentensoren aus den Verdopplungsrunden gespeichert. Das kostet $\sim 2\text{ GB}$ pro Schicht in kleinen Versuchen und $\sim 190\text{ GB}$ pro Schicht bei großen Sprachmodellen.

<!--en:879dbab0-->
::: {.slide title="Der adjungierte Scan als Rückwärtsscan zur Differentiation"}

<!--en:30b2cfb2-->
[Speichereffiziente Backpropagation]{.kicker}

<!--en:2966c4d5-->
- **Zeitlich rückwärts laufende lineare Rekurrenz:** Ein einziger Rückwärtsscan berechnet $\mathbf{g}_{T:1}$ mit `linear_scan`.

<!--en:63d58673-->
- **10× geringerer Speicherbedarf (`adjoint_scan`):** Nur $2$ Tensoren $(\mathbf{a}, \mathbf{h})$ bleiben gespeichert, statt der Zwischenergebnisse von $\sim 20$ Scan-Runden.

<!--en:2e4a6ea6-->
::: {.slide title="Prüfung des eigenen Vektor-Jacobi-Produkts im adjungierten Scan"}

<!--en:5a9f82c4-->
[Vergleich der Gradienten]{.kicker}

<!--en:859ae12c-->
Wir vergleichen die Gradienten $\partial \ell / \partial \mathbf{a}$ und $\partial \ell / \partial \mathbf{b}$ aus `adjoint_scan` mit der automatischen Differentiation durch `linear_scan` für einen Batch mit Sequenzlänge $T = 100$:

<!--en:785bef0c-->
Der zeitlich rückwärts laufende adjungierte Scan liefert bis auf `float32`-Genauigkeit ($\sim 10^{-6}$) dieselben Gradienten und benötigt nur einen Bruchteil des Aktivierungsspeichers.

<!--en:9310a8a7-->
::: {.slide title="Selektives Kopieren mit und ohne Auswahl in Mamba"}

<!--en:337d8cee-->
[Kontrollierte Ablation]{.kicker}

<!--en:86e36aea-->
Wir trainieren `Mamba` mit 2 Blöcken ($N = 4$) sowie `Mamba, not selective` (`selective=False`). Die Eingabe- und Ausgabeprojektionen, die kausale Faltung der Breite 4 und das SiLU-Ausgabegatter sind identisch. In der zweiten Variante sind $\boldsymbol{\Delta}, \mathbf{B}, \mathbf{C}$ gelernte Konstanten pro Kanal:

<!--en:569bfac3-->
**Die Selektivität entscheidet:** `Mamba` erreicht **eine Genauigkeit von mindestens $0.95$**, meist nahe $1.00$. Bleiben $\boldsymbol{\Delta}, \mathbf{B}, \mathbf{C}$ im ansonsten identischen Block über alle Token konstant, sinkt die Genauigkeit um **$>30$ Prozentpunkte**. Die Ergebnisse streuen stark mit dem Zufallsstartwert.

<!--en:6f6c6828-->
::: {.slide title="Zerlegung des Endzustands nach Tokenposition"}

<!--en:0d7e8d09-->
[Mechanistische Interpretierbarkeit]{.kicker}

<!--en:a7348a4c-->
- **Norm des verbleibenden Beitrags je Position:** `state_contributions` misst für die Position $s$ die euklidische Norm des Schreibbeitrags, der in $\mathbf{h}_T$ noch erhalten ist.

<!--en:e093d374-->
- **Faltungsfenster der Breite 4:** Jedes markierte Symbol verteilt sich auf $\{s, \ldots, s+3\}$ (zusammen $\le 1.6\%$ der $1{,}025$ Token).

<!--en:7f0da91b-->
::: {.slide title="Spitzen bei markierten Symbolen in Mamba, stärkere Gewichtung jüngerer Token in S4D"}

<!--en:ac37ff83-->
[Profil der Zustandsbeiträge]{.kicker}

<!--en:350b591a-->
Für jede Schicht zeigen wir den Anteil des Endzustands aus den Symbolfenstern ($1.6\%$ der Positionen). Für die selektivere Schicht jedes Modells zeigen wir zudem die Beiträge aller $1{,}025$ Positionen:

<!--en:a315ceb3-->
::: {.slide title="Auswahl messen mit dem Spitzenverhältnis zwischen Symbol und Fülltoken"}

<!--en:5acea507-->
[Große Schritte in wenigen Kanälen]{.kicker}

<!--en:094e1515-->
Für das **Spitzenverhältnis** teilen wir den höchsten Beitrag im Fenster von 4 Token ab einem Symbol durch den Beitrag des unmittelbar vorhergehenden Fülltokens. Für beide Schichten bilden wir den Median über die Symbole aus $64$ Testsequenzen:

<!--en:e7522159-->
- **Verstärkung um Größenordnungen:** In einer selektiven Schicht von `Mamba` tragen die Symbole **Hunderte bis Zehntausende Male** so viel bei wie benachbarte Fülltoken. Bei `S4D` beträgt der Faktor höchstens einige Zehn. Ein fester Filter kann auf ein Token stärker reagieren als auf ein anderes, es aber nicht länger behalten. Beide Schichten können eine Auswahl treffen oder nur eine. Welche es ist, variiert von Lauf zu Lauf.

<!--en:416bdf3d-->
- **Wenige Kanäle steuern die Auswahl:** Jedes Symbol wird von **zwei oder drei der $H = 96$ Kanäle** getragen. Bei Fülltoken gilt $\Delta_{t,c} \approx 10^{-5}$. Ihr Zustand bleibt eingefroren, und der Schreibbeitrag ist verschwindend klein. Beim Symbol wählen sie Schrittweiten von $\Delta_{t,c} \approx 0.1$ bis $1.5$ und **schreiben es mit hohem Gewicht** ($\Delta_{t,c} x_{t,c}\mathbf{B}_t$). Der frühere Zustand klingt dabei nur wenig ab. Die Spitze entsteht durch starkes Schreiben in sonst eingefrorene Kanäle und nicht durch Leeren des Zustands.

<!--en:403564a1-->
::: {.slide title="Mamba, minGRU und Attention auf WikiText-103 auf Byteebene"}

<!--en:a94e4287-->
[Benchmark zur Sprachmodellierung]{.kicker}

<!--en:0d92956e-->
Vergleich von `Mamba`, `minGRU` und kausaler `attention` ($d = 256$) mit je 4 Blöcken sowie `minGRU, d=309` mit angeglichener Parameteranzahl ($3.99\text{M}$ Parameter) auf Fenstern von 128 Byte aus WikiText-103:

<!--en:323f8488-->
`Mamba` erreicht **$1.30$ Nats/Byte** und liegt damit gleichauf mit dem vollständigen sequenziellen LSTM und der GRU aus §{sec_lstm}, obwohl das Training einen parallelen Scan nutzt. Gegenüber `minGRU, d=309` mit gleicher Parameteranzahl sinkt der Verlust um **$0.03$ Nats/Byte**.

<!--en:dfd1fd50-->
::: {.slide title="Verständnisfragen zu Softplus-Gates, Zustandsgröße und adjungiertem Scan"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:aa6b5a39-->
Beantworten Sie die folgenden drei Fragen:

<!--en:a6c53d9c-->
1. Ein selektives SSM hat 1 Zustand ($N = 1, \lambda = -1$). Welcher Wert $s_t$ vor der Aktivierung der Schrittweite ergibt den diskreten Erhaltungsfaktor $z_t = e^{-\textrm{softplus}(s_t)} = 0.999$ (Halbwertszeit $\approx 693$ Schritte)?

<!--en:5c336082-->
2. Betrachten Sie eine `Mamba`-Schicht der Modellbreite $d = 256$ ($H = 2d = 512$, $N = 16$, kausale Faltung der Breite 4). Wie viele Zahlen speichert sie bei der autoregressiven Generierung insgesamt pro Sequenz? Wie vielen Token im KV-Cache einer Attention-Schicht ($2d$ Zahlen/Token) entspricht das?

<!--en:10174a7c-->
3. Warum senkt `adjoint_scan` den Aktivierungsspeicherbedarf des Scans von $O(B T H N \log_2 T)$ auf $O(B T H N)$?

<!--en:557b98b6-->
**Antworten:**

<!--en:eff4cdaf-->
1. Da $e^{-\textrm{softplus}(s_t)} = \sigma(-s_t) = 0.999$ gilt, brauchen wir $-s_t = \log(0.999 / 0.001) = \log 999 \approx 6.91 \implies \mathbf{s_t \approx -6.91}$ ($\Delta_t \approx 0.001$).

<!--en:e716cf2a-->
2. Der SSM-Zustand umfasst $H \times N = 512 \times 16 = 8{,}192 = 32d$ Zahlen ($16$ Token im KV-Cache). Hinzu kommen die letzten $3$ Faltungseingaben der Breite $H = 2d$ ($3 \times 2d = 6d$ Zahlen oder $3$ Token im KV-Cache). Zusammen sind das **$38d = 9{,}728$ Zahlen, entsprechend nur $19$ Token im KV-Cache der Attention** bei jeder Sequenzlänge.

<!--en:f22b1e91-->
3. Die übliche automatische Differentiation speichert die Ausgabe jeder Verdopplungsrunde ($\lceil \log_2 T \rceil$ Runden). Dagegen speichert `adjoint_scan` aus dem Vorwärtsdurchlauf nur die fertig berechneten Zustände $\mathbf{h}$ und die Abklingfaktoren $\mathbf{a}$. Ein einziger Rückwärtsscan berechnet $\mathbf{g}_t = \mathbf{g}_t^{\text{direct}} + \mathbf{a}_{t+1}\odot \mathbf{g}_{t+1}$.

<!--en:d63c43f8-->
::: {.slide title="Zusammenfassung zu selektiven Zustandsraummodellen (Mamba)"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:2d580cec-->
| Komponente oder Modell | Mathematischer Mechanismus | Fähigkeit und empirisches Ergebnis |

<!--en:2e27813e-->
| **Identität für Softplus und Gate** | $e^{-\textrm{softplus}(s_t)} = \sigma(-s_t)$ | verbindet die kontinuierliche Schrittweite $\Delta_t$ mit dem `MinGRU`-Update-Gatter $z_t$ |

<!--en:43566dd1-->
| **Selektives SSM (`Mamba`)** | $\mathbf{h}_t^{(c)} = e^{\Delta_{t,c}\boldsymbol{\lambda}} \odot \mathbf{h}_{t-1}^{(c)} + \Delta_{t,c} x_{t,c}\mathbf{B}_t$ | tokenabhängige $\boldsymbol{\Delta}_t, \mathbf{B}_t, \mathbf{C}_t$ und Raten $\lambda_n = -n$ für mehrere Zeitskalen |

<!--en:22e64d30-->
| **Vektor-Jacobi-Produkt per adjungiertem Scan** | $\mathbf{g}_t = \nabla_{\mathbf{h}_t}\ell + \mathbf{a}_{t+1} \odot \mathbf{g}_{t+1}$ | differenziert den Scan durch einen Rückwärtsscan und speichert $2$ Tensoren statt $\sim 20$ |

<!--en:bd861032-->
| **Selektives Kopieren** | $4$ Symbole an zufälligen Positionen unter $1{,}024$ Fülltoken | `Mamba` erreicht **mindestens $0.95$** (`S4D`: $\le 0.20$, Mamba ohne Auswahl verliert $>30$ Prozentpunkte) |

<!--en:f2045d14-->
| **WikiText-103 auf Byteebene** | `SeqLM` mit 4 Blöcken ($d = 256$, $128$ Byte) | `Mamba` erreicht **$1.30$ Nats/Byte** (gleichauf mit LSTM und GRU, gegenüber `minGRU, d=309` um $0.03$ besser) |
