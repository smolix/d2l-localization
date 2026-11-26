<!--en:6fe17d03-->
# Sequenz-zu-Sequenz-Lernen

<!--en:58851b82-->
Die Sprachmodelle der vorigen Kapitel setzen eine einzelne Sequenz fort. Sie lesen Text und sagen das nächste Token vorher. Viele Aufgaben bilden eine Sequenz auf eine andere mit unterschiedlicher Länge ab. Dabei entsprechen die Positionen einander nicht eins zu eins. Ein Übersetzer etwa liest einen englischen Satz und schreibt einen französischen, eine Spracherkennung überträgt eine Tonaufnahme in Wörter. Bei der Übersetzung erreichten neuronale Netze 2014 erstmals die Qualität statistischer phrasenbasierter Systeme :cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`. Ihre Architektur war einfach. Ein rekurrentes Netz liest die Quellsequenz, ein zweites schreibt die Zielsequenz. Das erste übergibt dem zweiten einen einzigen Zustandsvektor fester Breite.

<!--en:2f5fea8f-->
Die Nachfolger dieser Architektur kennen Sie bereits: die Cross-Attention (:numref:`chap_attention`) und den Transformer-Encoder-Decoder (:numref:`sec_transformer`). Beide wurden entwickelt, um den Flaschenhals (engl. bottleneck) zu umgehen, der durch den einzelnen Vektor entsteht. Warum greifen wir die Architektur von 2014 wieder auf? Genau dieser Flaschenhals steht im Mittelpunkt des Kapitels. Jedes rekurrente Modell in den folgenden Abschnitten verdichtet seine Vergangenheit zu einem Zustand fester Größe. Wir führen hier das Encoder-Decoder-Prinzip und die gängigen lexikalischen Metriken für die Übersetzung (`chrF` und `BLEU`) ein. Anschließend messen wir beim Umkehren von Zeichenketten direkt, wie viel Information ein einzelner Zustandsvektor fester Breite bei wachsender Eingabelänge übertragen kann.

<!--en:1ba44304-->
## Das Encoder-Decoder-Prinzip

<!--en:5eab12a7-->
Die Abbildung einer Sequenz auf eine andere variabler Länge (*Sequenz-zu-Sequenz-Lernen* (engl. sequence-to-sequence learning) oder *Sequenztransduktion*) lässt sich in zwei Module gliedern (:numref:`fig_encoder_decoder`). Ein *Encoder* liest die Quellsequenz und fasst sie in einem Zustand zusammen. Ein *Decoder* übernimmt diesen Zustand und erzeugt die Zielsequenz Token für Token. Alles, was der Decoder über die Quellsequenz erfährt, muss diesen Zustand durchlaufen.

<!--en:1e283721-->
![Die Encoder-Decoder-Architektur: Der Encoder fasst die Quellsequenz in einem Zustand zusammen. Aus diesem Zustand erzeugt der Decoder die Zielsequenz.](../img/mdl-modernrnn-encoder-decoder.svg)

<!--en:8a81b761-->
Gegeben seien eine Quellsequenz $x_1, \ldots, x_T$ und eine Zielsequenz $y_1, \ldots, y_{T'}$ (wobei $T'$ von $T$ abweichen darf). Wie ein Sprachmodell (:numref:`sec_language-model`) faktorisiert der Decoder die bedingte Wahrscheinlichkeit autoregressiv. Hier hängt jedoch jeder Faktor zusätzlich vom Encoder-Zustand $\mathbf{c}$ ab:

<!--en:ade5b45d-->
Auf der rechten Seite verschwindet die Quellsequenz, nur $\mathbf{c}$ bleibt. Die Gleichung gilt also nur, wenn $\mathbf{c}$ alles über die Quellsequenz bewahrt, was der Decoder braucht. Beim Training minimieren wir die Kreuzentropie für das nächste Token mit Teacher-Forcing (:numref:`subsec_teacher-forcing`). Dabei geben wir die jeweils vorangegangenen Zieltoken aus der Ground Truth vor. Bei der Inferenz verwendet der Decoder seine eigenen Ausgaben als nächste Eingabe, bis er ein Sequenzendtoken erzeugt.

<!--en:4f6adba9-->
Zwei Forschungsgruppen stellten diese rekurrente Architektur 2014 vor. :citet:`Cho.Van-Merrienboer.Gulcehre.ea.2014` bewerteten mit einem GRU-Encoder-Decoder (:numref:`sec_gru`) Phrasenpaare in einem statistischen Übersetzungssystem. :citet:`Sutskever.Vinyals.Le.2014` übersetzten mit tiefen LSTMs, die über einen einzigen Vektor verbunden waren, durchgängig vom Englischen ins Französische. Sutskever et al. gaben den Quellsatz *rückwärts* in den Encoder ein. So lagen zwischen den ersten Wörtern der Übersetzung und den entsprechenden Quellwörtern nur wenige rekurrente Schritte. Das verkürzte den Gradientenpfad bei gleicher Informationsmenge, die der Zustand insgesamt speichern musste.

<!--en:428a8be0-->
Transformer lösten die rekurrenten Encoder und Decoder später ab, etwa Whisper für gesprochene Sprache :cite:`radford2023whisper` oder Bild-Sprach-Modelle für Bildbeschreibungen. Das Encoder-Decoder-Prinzip blieb bestehen. In :numref:`sec_transformer` änderte sich der Zustand, den der Decoder erhält. Aus einem einzelnen Vektor für die gesamte Quellsequenz wurde eine Sequenz mit je einem Vektor pro Quelltoken. Hier behalten wir den einzelnen Zustandsvektor von 2014 bei, weil wir gerade seine Kapazitätsgrenze messen wollen.

<!--en:fdb729d2-->
## Bewertung von Übersetzungen

<!--en:cfad23db-->
Die Bewertung durch Menschen ist für eine Trainingsschleife zu langsam. Modelle für die maschinelle Übersetzung bewertet man daher anhand der lexikalischen Übereinstimmung zwischen den $n$-Grammen der erzeugten Übersetzung und denen einer Referenzübersetzung. Die Arbeiten von 2014 gaben BLEU-Werte an. Moderne Benchmarks empfehlen stattdessen chrF auf Zeichenebene :cite:`Popovic.2015,Kocmi.Federmann.Grundkiewicz.ea.2021`. Wir implementieren hier beide Metriken und verwenden sie in :numref:`sec_lm_scoring` erneut.

<!--en:555b3daa-->
**chrF** entfernt Leerzeichen und vergleicht $n$-Gramme auf Zeichenebene für die Ordnungen $n = 1, \ldots, 6$. Damit ist das Ergebnis unabhängig von der Tokenisierung. Gemeinsame Wortstämme oder nur leicht abweichende Flexionsformen zählen als Teiltreffer. Für jede Ordnung $n$ gibt die Präzision den Anteil der $n$-Gramme des Kandidaten an, die in der Referenz vorkommen. Die Trefferquote (engl. recall) gibt umgekehrt den Anteil der $n$-Gramme der Referenz an, die im Kandidaten vorkommen. Der Schnitt der Multimengen `p & r` zählt wiederholte $n$-Gramme höchstens so oft, wie sie in der Referenz vorkommen. Aus den Mittelwerten von Präzision und Trefferquote über alle sechs Ordnungen, $P$ und $R$, berechnet chrF den $F_\beta$-Score:

<!--en:861b7ddb-->
Bei $\beta = 2$ hat die Trefferquote mehr Gewicht als die Präzision.

<!--en:59397f27-->
**BLEU** :cite:`Papineni.Roukos.Ward.ea.2002` bestimmt für Wort-$n$-Gramme die Präzisionen $p_n$ ($n = 1, \ldots, k$), wobei Treffer höchstens so oft zählen wie in der Referenz. BLEU multipliziert ihr gewichtetes geometrisches Mittel mit einer *Brevity Penalty*. So erzielen zu kurze Fragmente nicht allein wegen ihrer hohen Präzision einen hohen Score:

<!--en:881910e8-->
Da der BLEU-Wert ein Produkt ist, genügt eine einzige Präzision von $p_n = 0$ bei einer beliebigen Ordnung $n \le k$, damit der BLEU-Wert auf Satzebene auf $0$ fällt. (Auf Korpusebene gewichtet das übliche BLEU-Verfahren alle Ordnungen $n \le 4$ gleich mit $1/4$ und summiert die Häufigkeiten über den gesamten Testdatensatz. Dabei kommen Nullhäufigkeiten selten vor.)

<!--en:71b3c80a-->
Wie unterschiedlich bewerten die beiden Metriken einen einzelnen Satz? Wir vergleichen eine exakte Kopie, eine fast richtige Übersetzung mit einem Flexionsfehler beim dritten von sechs Wörtern und eine korrekte Paraphrase. Jedes 4-Gramm eines Satzes aus sechs Wörtern enthält das dritte Wort. Für die fast richtige Übersetzung gilt deshalb $p_4 = 0$. Damit fällt der BLEU-Wert bei $k = 4$ auf null, während bei chrF nur die Zeichen-$n$-Gramme mit dem überzähligen *s* als Treffer entfallen.

<!--en:ee3269c5-->
Die fast richtige Übersetzung erreicht bei chrF 0.91 und bei BLEU 0.00, die sinngleiche Paraphrase bei chrF 0.67 und bei BLEU 0.00. Beide Metriken zählen Übereinstimmungen in der äußeren Form und erfassen die Bedeutung nicht. Deshalb kombiniert man chrF (oder für historische Vergleiche sacreBLEU auf Korpusebene :cite:`Post.2018`) heute mit trainierten neuronalen Bewertungsmodellen wie COMET :cite:`Rei.Stewart.Farinha.ea.2020` oder mit LLMs als Gutachtern.

<!--en:ae313e6e-->
## Der Zustand fester Größe als Flaschenhals

<!--en:80e94f5f-->
Wie viel von einer Eingabesequenz lässt sich über einen einzigen Zustandsvektor übertragen? In natürlicher Sprache hängt die Sequenzlänge auch damit zusammen, wie selten eine syntaktische Konstruktion ist. Um die Zustandskapazität isoliert zu betrachten, verwenden wir die synthetische Aufgabe aus :numref:`subsec_reversal-task`. Wir kehren eine zufällige Zeichenkette aus $T$ unabhängigen Buchstaben um, die wir gleichverteilt aus einem Alphabet mit 16 Symbolen ziehen. Wegen dieser Unabhängigkeit verraten die bereits dekodierten Buchstaben nichts über den nächsten. Jeder Buchstabe muss den Endzustand $\mathbf{c}$ des Encoders durchlaufen. Bei der Umkehrung gibt der Decoder zudem das letzte Token des Encoders zuerst aus. Das entspricht der Anordnung, die Sutskever et al. durch Umkehren der Eingabe erzielten.

<!--en:614c9d24-->
Für jeden Minibatch ziehen wir neue zufällige Zeichenketten. `Reversal` fasst die Quellzeichenkette und die um eine Position nach rechts verschobene, umgekehrte Zielzeichenkette (mit vorangestelltem Starttoken `V`) in einem einzigen Tensor `X` für `d2l.train_steps` zusammen.

<!--en:8dca99f6-->
Jeder der 16 gleich wahrscheinlichen Buchstaben trägt $\log_2 16 = 4$ Bit Information. Eine Zeichenkette der Länge $T$ enthält also $4T$ Bit (32 Bit bei $T = 8$ und 256 Bit bei $T = 64$). Rein rechnerisch belegen 64 Gleitkommazahlen einfacher Genauigkeit 2048 Bit. Das reicht für 512 Buchstaben. Wie viele Buchstaben kann eine trainierte GRU tatsächlich in 64 oder 256 Gleitkommazahlen kodieren und daraus wieder dekodieren?

<!--en:1ed9d7eb-->
Wie in :numref:`fig_seq2seq_details` und bei :citet:`Cho.Van-Merrienboer.Gulcehre.ea.2014` initialisiert der Endzustand $\mathbf{c}$ des GRU-Encoders den GRU-Decoder ($\mathbf{s}_0 = \mathbf{c}$). Zusätzlich hängen wir ihn bei jedem Dekodierschritt an die Einbettung der Eingabe an:

<!--en:a152926a-->
![Ein über die Zeit entfalteter GRU-Encoder-Decoder beim Umkehren einer Zeichenkette. Der Encoder liest die Quellsequenz. Sein Endzustand $\mathbf{c}$ initialisiert den Decoder und geht in jeden Dekodierschritt ein. Der Decoder liest die umgekehrte, hinter ein Starttoken nach rechts verschobene Zeichenkette und sagt die umgekehrte Zeichenkette vorher.](../img/mdl-modernrnn-seq2seq.svg)

<!--en:a4ba1544-->
Wir trainieren jedes Modell 1000 Schritte lang mit AdamW auf Minibatches aus je 128 neu gezogenen Zeichenketten. Die Lernrate folgt einem Kosinus-Abfall von $3 \times 10^{-3}$ auf null (`d2l.LMOptimizer` und `d2l.train_steps`), wie in :numref:`sec_transformer`. An 1000 weiteren, neu gezogenen Zeichenketten messen wir mit Teacher-Forcing die Genauigkeit pro Buchstabe. Da die Buchstaben unabhängig sind, verraten die vorgegebenen korrekten Token nichts über den nächsten Zielbuchstaben. Zugleich verhindern wir, dass frühe Dekodierfehler die Messung der Zustandskapazität verfälschen.

<!--en:b890c9bd-->
Für die Zustandsbreiten 64 und 256 und die Sequenzlängen $T \in \{8, 16, 32, 64\}$ verwenden wir jeweils zwei Startwerte für den Zufallszahlengenerator (zufälliges Raten erreicht eine Genauigkeit von $1/16 \approx 0.06$).

<!--en:e7fbab9a-->
Bei Breite 64 sinkt die Genauigkeit von 0.99 bei $T = 8$ auf etwa 0.73 bei $T = 16$ und auf 0.26–0.27 bei $T = 64$. Vervierfachen wir die Zustandsbreite auf 256, bleibt die Genauigkeit bis $T = 16$ bei 0.99. Bei $T = 64$ rekonstruiert das Modell noch etwa die Hälfte der Buchstaben. Dennoch nutzt es nur einen kleinen Bruchteil der verfügbaren Gleitkommabits. Bei $T = 16$ trägt die Zeichenkette nur 64 Bit (1 Bit pro Gleitkommazahl im Zustand der Breite 64), doch ein Viertel der Buchstaben geht verloren. Selbst wenn die Zustandsbreite proportional zur Sequenzlänge wächst, sinkt die Genauigkeit: Bei $T = 8$ und Breite 64 erreicht sie 0.99, bei $T = 32$ und Breite 256 nur etwa 0.80. Das Netz muss die frühen Buchstaben auch über eine längere Kette nichtlinearer rekurrenter Zustandsübergänge weitergeben.

<!--en:0cca73db-->
Auf denselben Flaschenhals stieß die neuronale maschinelle Übersetzung 2014 :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014`. Deshalb führten :citet:`Bahdanau.Cho.Bengio.2014` die Cross-Attention ein, mit der der Decoder auf jeden Encoder-Zustand zurückgreifen kann (:numref:`sec_attention-scoring-functions`, :numref:`sec_transformer`). Damit gelingt die fehlerfreie Umkehrung von Zeichenketten der Länge 12 innerhalb von 300 Schritten. Der dafür nötige Cache wächst linear mit der Eingabelänge. Im weiteren Verlauf dieses Kapitels untersuchen wir den ergänzenden Ansatz. Wir behalten einen Zustand fester Größe bei und ermöglichen paralleles Training. Dann erweitern wir den Vektor zu einem Matrixspeicher für Schlüssel-Wert-Paare und kombinieren ihn schließlich mit einem kleinen Anteil an Attention-Schichten.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:57460646-->
Die Encoder-Decoder-Architektur bildet eine Quellsequenz variabler Länge auf eine Zielsequenz ebenfalls variabler Länge ab. Der Encoder fasst die Quellsequenz in einem Zustand $\mathbf{c}$ zusammen. Ein autoregressiver Decoder erzeugt die Zielsequenz bedingt auf $\mathbf{c}$. Die Übersetzungsqualität misst man üblicherweise anhand der Übereinstimmung von Zeichen-$n$-Grammen (chrF) oder der Präzision von Wort-$n$-Grammen mit einer Brevity Penalty (BLEU). Gelernte semantische Metriken ergänzen diese Maße. Beim Umkehren synthetischer Zeichenketten sinkt die Genauigkeit mit wachsender Sequenzlänge rasch, wenn nur ein GRU-Zustandsvektor fester Breite zur Verfügung steht. Er speichert deutlich weniger als ein effektives Bit pro Zustandskoordinate. Das motiviert sowohl Attention-Mechanismen als auch die strukturierten rekurrenten Zustände höherer Kapazität, die wir im weiteren Verlauf dieses Kapitels entwickeln.

<!--en:2294243a-->
## Übungen

<!--en:f2ca1f07-->
1. **BLEU und chrF im Vergleich.** Bewerten Sie die fast richtige Übersetzung und die Paraphrase mit `bleu` für $k = 1, 2, 3, 4$. Erklären Sie jeden Wert anhand der Häufigkeiten der $n$-Gramme. Konstruieren Sie dann einen Kandidaten, bei dem chrF und BLEU möglichst weit auseinanderliegen. Die Richtung ist beliebig. Erklären Sie, welche Eigenschaft der jeweiligen Metrik der Kandidat ausnutzt.

<!--en:7eaba032-->
1. **Informationsübertragung durch den Zustand.** Nach der Abzählung in diesem Abschnitt kann ein Zustand aus $N$ Zahlen mit je $b$ Bit höchstens $2^{Nb}$ Werte annehmen.
    1. Zeigen Sie, dass $T$ Buchstaben aus einem Alphabet mit 16 Buchstaben nur dann für jede mögliche Zeichenkette fehlerfrei in umgekehrter Reihenfolge ausgegeben werden können, wenn $Nb \ge 4T$ gilt.
    1. Sei $p$ die Genauigkeit des Decoders pro Buchstabe, gemessen wie in diesem Abschnitt mit den jeweils korrekten vorangegangenen Buchstaben als Eingabe. Zeigen Sie mithilfe der Fano-Ungleichung (:numref:`subsec_mdl-fano`) und der Kettenregel der gegenseitigen Information, dass mindestens $T\,(4 - H_\textrm{b}(1 - p) - (1 - p)\log_2 15)$ Bit Information über die Zeichenkette durch $\mathbf{c}$ übertragen wurden. Dabei bezeichnet $H_\textrm{b}$ die binäre Entropie in Bit.
    1. Berechnen Sie diese Schranke für jeden Tabelleneintrag und vergleichen Sie sie mit $32N$. Wie viele Bit nutzt der trainierte Encoder pro Zahl, und hängt diese Anzahl von $T$ ab?
    1. Trainieren Sie das Modell der Breite 64 für jede Sequenzlänge viermal so lange. Bei welchen Längen steigt die Genauigkeit am stärksten? Verschwindet der Rückgang bei wachsender Länge? Was legt Ihr Ergebnis nahe: Setzt der Zustand oder das Training die Grenze?

<!--en:37702c16-->
1. **Sutskevers Trick rückgängig machen.** Unsere Umkehraufgabe hat bereits die Anordnung, die Sutskever et al. durch Umkehren der Quellsequenz erreichten: Der zuerst auszugebende Buchstabe ist der zuletzt gelesene. Machen Sie dies rückgängig, indem Sie dem Encoder die Quellsequenz rückwärts zuführen. Der Decoder muss die Zeichenkette dann in der Reihenfolge kopieren, in der der Encoder sie gelesen hat.
    1. Zählen Sie für jeden Buchstaben die Schritte zwischen Lesen und Schreiben, sowohl beim Umkehren als auch beim Kopieren. Bei welcher Aufgabe sind die Abhängigkeiten im Mittel kürzer? Bei welcher Aufgabe tritt die kürzeste einzelne Abhängigkeit auf?
    1. Trainieren Sie für beide Aufgaben je ein Modell mit Breite 64 und $T = 16$. Messen Sie die Genauigkeit insgesamt und an jeder Ausgabeposition. Welche Aufgabe ist leichter zu lernen? Stimmt das Ergebnis mit dem Argument von :citet:`Sutskever.Vinyals.Le.2014` für die Umkehrung der Quellsequenz überein?

<!--en:220d0aa1-->
1. **Kontext bei jedem Schritt.** Übergeben Sie den Kontext $\mathbf{c}$ wie bei Sutskever et al. nur als Anfangszustand an den Decoder. Lassen Sie ihn als zusätzliche Eingabe bei den einzelnen Schritten weg. Wie verändert sich die Genauigkeit bei den jeweiligen Sequenzlängen? Warum könnte es einem kleinen Modell helfen, den Kontext bei jedem Schritt erneut zu erhalten?

<!--en:b9b75206-->
[Diskussionen](https://d2l.discourse.group/t/1062)

<!--en:28d28eff-->
[Diskussionen](https://d2l.discourse.group/t/18022)

<!--en:8a96e8a4-->
Sequenz-zu-Sequenz-Lernen und der Zustand fester Größe als Flaschenhals<br>**Encoder-Decoder-Prinzip · Bewertung mit chrF und BLEU · Umkehrung von Zeichenketten · effektive Kapazität in Bit**

<!--en:3b0ea163-->
::: {.slide title="Von der Vorhersage in einer einzelnen Sequenz zur Sequenztransduktion"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:1069d597-->
- **Transduktion ohne feste Zuordnung:** Übersetzung, Spracherkennung und Zusammenfassung bilden eine Quellsequenz $x_{1:T}$ auf eine Zielsequenz $y_{1:T'}$ mit $T' \neq T$ ab. Dabei kann sich die Reihenfolge nichtmonoton ändern.

<!--en:a3f0445e-->
- **Architektur mit Flaschenhals von 2014:** Ein Encoder-RNN verdichtet $x_{1:T}$ zu einem einzigen Vektor $\mathbf{c} \in \mathbb{R}^d$ fester Breite. Daraus erzeugt ein Decoder-RNN die Zielsequenz $y_{1:T'}$.

<!--en:45b4a084-->
- **Ziel der Messung:** Wie viele Bit kann ein einzelner rekurrenter Zustandsvektor genau übertragen?

<!--en:fe6d59cb-->
::: {.slide title="Die Encoder-Decoder-Architektur"}

<!--en:a44094ec-->
[Architekturprinzip]{.kicker}

<!--en:0859368c-->
- **Encoder:** Er liest die Quelltoken $x_1, \ldots, x_T$ einer Sequenz beliebiger Länge $T$ und fasst sie in einem Zustand $\mathbf{c}$ fester Dimension zusammen.

<!--en:e9bede1f-->
- **Decoder:** Er übernimmt $\mathbf{c}$ und erzeugt die Zieltoken $y_1, \ldots, y_{T'}$ Schritt für Schritt, bis er ein Sequenzendtoken ausgibt.

<!--en:09099778-->
**Prinzip des Informationsflaschenhalses:** Alles, was der Decoder über die Quellsequenz weiß, muss den Zustand $\mathbf{c}$ durchlaufen.

<!--en:453e60b0-->
::: {.slide title="Der Decoder als bedingtes autoregressives Modell"}

<!--en:8ce8ef8f-->
[Probabilistische Formulierung]{.kicker}

<!--en:fa83f982-->
- **Annahme einer suffizienten Statistik:** Die Gleichung gilt nur, wenn $\mathbf{c}$ jedes für $y_{1:T'}$ benötigte Detail der Quellsequenz bewahrt.

<!--en:d5d63c30-->
- **Training und Inferenz:** Paralleles Training mit **Teacher-Forcing** und korrekten Zielpräfixen, autoregressives Dekodieren bis `<eos>`.

<!--en:739a7c01-->
::: {.slide title="Der Durchbruch der neuronalen maschinellen Übersetzung 2014"}

<!--en:589b5289-->
[Historischer Kontext]{.kicker}

<!--en:6f509bf6-->
- **Cho et al. und Sutskever et al. (2014):** Sie führten die Bewertung von Phrasenpaaren mit GRUs und die Übersetzung Englisch$\to$Französisch mit 4 LSTM-Schichten ein. Beide Ansätze verwenden einen einzigen Vektor $\mathbf{c}$.

<!--en:a32e041f-->
- **Umkehrung der Quellsequenz:** Durch die Eingabereihenfolge $x_T, \ldots, x_1$ lag $x_1$ direkt vor $y_1$. Damit sank der anfängliche Abstand von $T+1$ auf $2$ Schritte.

<!--en:e89313f7-->
- **Die 20-Wort-Grenze:** Bei Sätzen mit mehr als $20$–$30$ Wörtern brach die Qualität dennoch ein.

<!--en:e6ac03a7-->
::: {.slide title="Lexikalische Bewertung mit dem F-Score für Zeichen-N-Gramme (chrF)"}

<!--en:7802915d-->
[Übersetzungsmetriken]{.kicker}

<!--en:5e5aaf70-->
- **Zeichen-$1\ldots 6$-Gramme:** chrF entfernt Leerzeichen und zählt Teilzeichenketten. Gemeinsame Wortstämme zählen unabhängig von der Tokenisierung als Teiltreffer.

<!--en:6541414b-->
- **Höheres Gewicht für die Trefferquote ($\beta = 2$):** chrF begrenzt die Häufigkeiten des Kandidaten auf die der Referenz (`p & r`). Die Trefferquote erhält gegenüber der Präzision den Gewichtungsfaktor $4\times$.

<!--en:2a44d986-->
::: {.slide title="Präzision von Wort-N-Grammen und Brevity Penalty (BLEU)"}

<!--en:7802915d-2-->
[Übersetzungsmetriken]{.kicker}

<!--en:8df5a05b-->
- **Brevity Penalty (BP):** Ein exponentieller Faktor bestraft zu kurze Kandidaten, die allein mit hoher 1-Gramm-Präzision gut abschneiden würden.

<!--en:b871abac-->
- **Ein fehlender Treffer genügt:** Schon ein einziges $p_n = 0$ setzt das gesamte BLEU-Produkt auf Satzebene auf $0$.

<!--en:31a6cb4f-->
::: {.slide title="Schwachstelle von BLEU auf Satzebene: Nullwerte bei Paraphrasen"}

<!--en:0096d524-->
[Grenzen der Metriken]{.kicker}

<!--en:3663512c-->
Vergleich von `chrF` und `BLEU` auf Satzebene ($k = 4$) für die Referenz `'we will visit our grandparents tomorrow'` mit 6 Wörtern:

<!--en:8df77fa6-->
- **Fast richtige Übersetzung (`'visits'` an Position 3):** Ein Satz mit 6 Wörtern hat nur drei Wort-4-Gramme (`1..4`, `2..5`, `3..6`). *Jedes einzelne* enthält das Wort an Position 3. Daher gilt $p_4 = 0$ und der BLEU-Wert fällt auf **$0.00$**, während `chrF` noch **$0.91$** erreicht.

<!--en:c6a4c725-->
- **Sinngleiche Paraphrase (`'are going to see'`):** Sie erreicht bei BLEU **$0.00$** und bei $0.67$ **`chrF`**. Damit bewertet chrF sie schlechter als die fast richtige, grammatisch fehlerhafte Übersetzung.

<!--en:31a4b1ae-->
Kombinieren Sie `chrF` auf Zeichenebene (oder `sacreBLEU` auf Korpusebene) mit trainierten semantischen Bewertungsmodellen wie COMET oder mit LLMs als Gutachtern.

<!--en:95cfadfa-->
::: {.slide title="Isolierte Messung der Zustandskapazität durch Umkehren zufälliger Zeichenketten"}

<!--en:da06fac8-->
[Experiment zum Flaschenhals]{.kicker}

<!--en:5a780f3c-->
- **Unabhängige, gleichverteilte Buchstaben ($V = 16$):** Beim Umkehren von $T$ zufälligen Buchstaben ($4\text{ bits/letter}$) müssen alle $4T$ Bit den Zustand $\mathbf{c} \in \mathbb{R}^d$ durchlaufen.

<!--en:90b7eff0-->
- **Keine Abkürzung über das Sprachmodell:** Die gegenseitige Information zwischen den bereits dekodierten Buchstaben $y_{<t'}$ und dem nächsten Buchstaben $y_{t'}$ ist null.

<!--en:aca19258-->
- **Optimale Reihenfolge für den ersten Schritt:** Die letzte Eingabe $x_T$ wird zuerst ausgegeben ($y_1 = x_T$), während $x_1$ über $2T$ Schritte erhalten bleiben muss.

<!--en:1fd21ff3-->
::: {.slide title="Entfalteter GRU-Encoder-Decoder beim Umkehren von Zeichenketten"}

<!--en:1201dc49-->
[Modellarchitektur]{.kicker}

<!--en:a0c915d1-->
- Wie bei Cho et al. (2014) initialisiert der Endzustand $\mathbf{c} \in \mathbb{R}^d$ des Encoders den GRU-Decoder ($\mathbf{s}_0 = \mathbf{c}$). Bei jedem Dekodierschritt $t'$ hängen wir ihn auch an die Tokeneinbettung an. So muss der Decoder keine Zustandskapazität dafür aufwenden, sich $\mathbf{c}$ zu merken.

<!--en:fd2a4ffc-->
::: {.slide title="Informationstheoretische und erlernte Speicherkapazität"}

<!--en:a8009656-->
[Kapazitätsanalyse in Bit]{.kicker}

<!--en:cbb7359d-->
- **Speicherplatz und Entropie der Zeichenkette:** Ein Vektor der Dimension $d = 64$ aus `float32`-Zahlen umfasst $2{,}048$ Bit. Für $T \in \{8, 16, 64\}$ Buchstaben genügen dagegen $32, 64, 256$ Bit ($0.5$ bis $4.0$ Bit pro Gleitkommazahl).

<!--en:73cd036c-->
- **Messung mit Teacher-Forcing:** Wie viele Bit speichert der Gradientenabstieg tatsächlich in $\mathbf{c}$? Bei dieser Messung pflanzen sich Fehler aus früheren Vorhersagen nicht fort.

<!--en:28518481-->
::: {.slide title="Gemessene Zustandskapazität nach Sequenzlänge und Zustandsbreite"}

<!--en:51ab32f2-->
[Ergebnisse der Umkehraufgabe]{.kicker}

<!--en:ed030305-->
Genauigkeit pro Buchstabe mit Teacher-Forcing auf $1{,}000$ neuen Zeichenketten nach $1{,}000$ AdamW-Schritten (Batchgröße $128$, zwei Zufallsstartwerte pro Eintrag, Genauigkeit bei zufälligem Raten $= 1/16 \approx 0.06$):

<!--en:55e06ecb-->
::: {.slide title="Warum trainierte rekurrente Zustände weniger als 1 Bit pro Gleitkommazahl speichern"}

<!--en:bc40f265-->
[Deutung des Flaschenhalses]{.kicker}

<!--en:8773da3c-->
Die Tabelle zur Zustandskapazität bei der Umkehraufgabe zeigt drei quantitative Zusammenhänge:

<!--en:ab86072d-->
- **Starker Rückgang oberhalb von 0.5 Bit pro Gleitkommazahl:** Bei Breite $d = 64$ sinkt die Genauigkeit von **$0.99$** bei $T = 8$ ($0.5\text{ bits/float}$) auf **$0.72\text{--}0.74$** bei $T = 16$ ($1\text{ bit/float}$) und **$0.26\text{--}0.27$** bei $T = 64$ ($4\text{ bits/float}$).

<!--en:9cb80a0d-->
- **Ein breiterer Zustand verschiebt die Kurve nach rechts:** Bei vierfacher Zustandsbreite $d = 256$ bleibt die Genauigkeit bei **$0.99$** für $T = 16$. Etwa die Hälfte der Buchstaben ($0.50$–$0.51$) lässt sich bei $T = 64$ rekonstruieren.

<!--en:87372511-->
- **Auch bei proportional wachsender Breite sinkt die Genauigkeit:** Bei gleich vielen Buchstaben pro Gleitkommazahl im Zustand ($T/d = 1/8$) erreicht das Modell für $T = 8$ und $d = 64$ eine Genauigkeit von $0.99$. Für $T = 32$ und $d = 256$ sinkt sie auf $\sim 0.80$, weil die frühen Zeichen $4\times$ so viele nichtlineare GRU-Übergänge überdauern müssen.

<!--en:1742d396-->
::: {.slide title="Zwei Auswege aus dem Flaschenhals eines einzelnen Vektors"}

<!--en:b55f05af-->
[Alternativen für die Architektur]{.kicker}

<!--en:4dd601f2-->
- **Weg 1: Cross-Attention (§{num:chap_transformers}):** Alle $T$ Encoder-Zustände $[\mathbf{h}_1, \ldots, \mathbf{h}_T]$ bleiben für den exakten dynamischen Zugriff erhalten. Die Speicherkosten betragen $O(T)$.

<!--en:4d4d8d7b-->
- **Weg 2: Strukturierte Zustände und Matrixzustände (§§{sec_linear-recurrence}–§{num:sec_hybrids}):** Der Inferenzzustand hat weiterhin die Größe $O(1)$. Dazu dienen parallele lineare Scans, Schlüssel-Wert-Matrizen der Größe $d_v \times d_k$ aus äußeren Produkten, gezieltes Löschen mit der Delta-Regel und Hybridmodelle mit wenigen Attention-Schichten im Verhältnis 3:1.

<!--en:1d07be5f-->
::: {.slide title="Verständnisfragen zu N-Gramm-Metriken und Zustandskapazität in Bit"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:d3534d8c-->
Prüfen Sie Ihr Verständnis anhand dieser drei Fragen:

<!--en:fd827085-->
1. Warum ergibt BLEU auf Satzebene bei fest vorgegebener Ordnung $k = 4$ für jede Übersetzung aus 3 Wörtern genau $0.0$, selbst wenn alle 3 Wörter mit einer Referenz aus 3 Wörtern exakt übereinstimmen?

<!--en:0786d398-->
2. Warum bestimmen wir die Genauigkeit beim Umkehren von Zeichenketten mit Teacher-Forcing, wenn wir die Kapazität des Encoder-Zustands $\mathbf{c}$ messen?

<!--en:465ff839-->
3. Ein Encoder der Breite $64$ erreicht beim Umkehren zufälliger Zeichenketten eine Genauigkeit von $100\%$. Die Zeichenketten haben die Länge $T = 8$ und stammen aus einem Alphabet mit $16$ Buchstaben. Wie viele effektive Bit überträgt der Encoder pro Zustandskoordinate?

<!--en:557b98b6-->
**Antworten:**

<!--en:c7277ffa-->
1. Ein Satz aus 3 Wörtern enthält kein Wort-4-Gramm. (Im Code endet die Schleife wegen `min(k, len_pred)` bei $n=3$. Erzwingt man $k=4$ ohne Begrenzung auf `len_pred`, gilt $p_4 = 0$. Auch mit Begrenzung gibt es bei einem Kandidaten mit mindestens 4 Wörtern Fälle ohne übereinstimmendes 4-Gramm: Weicht bei 6 Wörtern das Wort an Position 3 ab, gilt $p_4 = 0$.)

<!--en:3a57eec0-->
2. Beim freien Dekodieren verändert ein einziges falsches Token im Schritt $t'=1$ die rekurrente Trajektorie des Decoders und verfälscht spätere Schritte. Da die zufälligen Buchstaben unabhängig sind, verrät Teacher-Forcing nichts über $y_{t'}$. So messen wir isoliert, welche Information $\mathbf{c}$ enthält.

<!--en:93f8978d-->
3. Jeder Buchstabe trägt $\log_2 16 = 4$ Bit. Also verteilen sich bei $8$ Buchstaben $32$ Bit auf $64$ Gleitkommazahlen: **$0.5\text{ effective bits per float}$** (von $32$ verfügbaren Bit).

<!--en:67fc34a4-->
::: {.slide title="Zusammenfassung: Sequenz-zu-Sequenz-Lernen und Zustandskapazität"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:48b1e2c3-->
| Komponente oder Konzept | Mathematische Form | Zentrale Eigenschaft oder Grenze |

<!--en:b6e62c2c-->
| **Encoder** | $\mathbf{c} = f_{\text{enc}}(x_1, \ldots, x_T) \in \mathbb{R}^d$ | Verdichtet die Quellsequenz variabler Länge zu einem Vektor $\mathbf{c}$ fester Größe |

<!--en:7e20f4e1-->
| **Autoregressiver Decoder** | $\prod_{t'=1}^{T'} P(y_{t'} \mid y_{<t'}, \mathbf{c})$ | Setzt voraus, dass $\mathbf{c}$ alle für die Zielsequenz nötigen Informationen über die Quellsequenz bewahrt |

<!--en:9188f53d-->
| **chrF-Metrik** | $F_2 = \frac{5PR}{4P + R}$ für Zeichen-$1\ldots 6$-Gramme | Robust gegenüber Flexion und Tokenisierung, zählt gemeinsame Teile von Wortstämmen als Teiltreffer |

<!--en:9b9ab70b-->
| **BLEU-Metrik** | $\exp(\min(0, 1 - \ell_{\text{ref}}/\ell_{\text{cand}})) \prod_{n=1}^k p_n^{w_n}$ | Das geometrische Produkt fällt auf Satzebene auf $0$, sobald für eine Ordnung $p_n = 0$ gilt |

<!--en:f9a6bfdb-->
| **Einzelner Zustandsvektor als Flaschenhals** | $4T$ Bit einer zufälligen Zeichenkette in $d$ Gleitkommazahlen | Die trainierte GRU speichert $< 1\text{ bit/float}$ (Genauigkeit $0.99 \to 0.26$ für $T=8 \to 64$ bei $d=64$) |

<!--en:9bd260f2-->
| **Architektonische Lösungen** | Cross-Attention (Cache der Größe $O(T)$) oder Matrixzustand ($O(1)$) | Attention speichert alle Token. In §{sec_linear-recurrence}–§{num:sec_hybrids} entwickeln wir schnelle Zustände hoher Kapazität |
