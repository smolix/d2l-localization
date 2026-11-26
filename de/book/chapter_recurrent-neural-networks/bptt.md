<!--en:7c88427c-->
# Backpropagation durch die Zeit

<!--en:b330af27-->
Unsere Experimente mit dem RNN-Sprachmodell in :numref:`sec_rnn-scratch` führten zu zwei auffälligen Beobachtungen. Ohne Gradienten-Clipping wurden die meisten Trainingsläufe instabil. Mit Clipping konvergierte das Training zuverlässig, doch das trainierte Netz nutzte aus seinem Kontextfenster von 32 Token nur die letzten paar Token. Beide Phänomene haben dieselbe mathematische Ursache. Ein rekurrentes Netz verwendet in jedem Zeitschritt dieselbe Gewichtsmatrix. Läuft ein Gradient rückwärts durch eine Sequenz, wird er daher in jedem Schritt mit der Jacobi-Matrix des Zustandsübergangs multipliziert. Über viele Schritte wirken diese Faktoren wie Zinseszinsen. Der Gradient wächst oder schrumpft geometrisch.

<!--en:fd67a699-->
Schon an der skalaren Rekurrenz $h_t = w_\textrm{h} h_{t-1} + x_t$ sehen wir, warum. Rollen wir sie aus, erhalten wir $h_t = \sum_{s \leq t} w_\textrm{h}^{t-s} x_s$ und damit $\partial h_t / \partial h_{t-k} = w_\textrm{h}^k$. Über vierzig Schritte schrumpft ein Gradient für $w_\textrm{h} = 0.9$ auf $0.9^{40} \approx 0.015$, während er für $w_\textrm{h} = 1.1$ auf $1.1^{40} \approx 45$ anwächst. Explodierende Gradienten (engl. exploding gradients) machen die Optimierung sofort instabil und zwingen uns, die Aktualisierungsschritte zu begrenzen. Verschwindende Gradienten (engl. vanishing gradients) fallen weniger auf, sind aber tückischer. Sie schwächen die Fehlersignale weit zurückliegender Token so stark ab, dass das Modell keine weitreichenden Abhängigkeiten lernen kann.

<!--en:0e1ad48c-->
Um Gradienten in einem RNN zu berechnen, brauchen wir keinen neuen Algorithmus zur Differentiation. Wir rollen die Rekurrenz über die Zeit zu einem tiefen Feedforward-Netz mit gemeinsamen Gewichten aus und wenden die übliche Backpropagation an (:numref:`sec_backprop`). Dieses Verfahren heißt *Backpropagation durch die Zeit* (engl. backpropagation through time, BPTT) :cite:`Werbos.1990`. Im Folgenden leiten wir die Rückwärtsrekursion direkt in Matrixform her. Wir messen, wie weit Gradienten in unserem trainierten RNN aus :numref:`sec_rnn-scratch` tatsächlich zurückreichen, und untersuchen die abgeschnittene Backpropagation durch die Zeit (engl. truncated backpropagation through time).

<!--en:599549e0-->
## Verschwindende und explodierende Gradienten

<!--en:aad65e67-->
Betrachten wir ein über $T$ Schritte ausgerolltes RNN mit gemeinsamen rekurrenten Parametern und dem mittleren Sequenzverlust $L = \frac{1}{T}\sum_{t=1}^T l(\mathbf{y}_t,
\mathbf{o}_t)$. Bei der Gewichtsteilung (engl. weight sharing) summiert die Backpropagation die lokalen Gradientenbeiträge aus allen Zeitschritten, in denen eine Gewichtsmatrix vorkommt. Um die Struktur des Rückwärtsdurchlaufs zu verdeutlichen, lassen wir beim RNN aus :numref:`sec_rnn` vorübergehend die Bias-Terme und die $\tanh$-Nichtlinearität weg. Gleich danach setzen wir $\tanh$ wieder ein. Für einen Minibatch aus $b$ Sequenzen mit den Eingaben $\mathbf{X}_t \in \mathbb{R}^{b\times d}$ im jeweiligen Zeitschritt berechnet das lineare RNN

<!--en:c8d44cc2-->
mit $\mathbf{W}_\textrm{xh}\in\mathbb{R}^{d\times h}$, $\mathbf{W}_\textrm{hh}\in\mathbb{R}^{h\times h}$ und $\mathbf{W}_\textrm{hq}\in\mathbb{R}^{h\times q}$. :numref:`fig_rnn_bptt` zeigt drei ausgerollte Zeitschritte dieses Netzes. Der Vorwärtsdurchlauf ist grau, der Rückwärtsdurchlauf rot dargestellt.

<!--en:cc9edc8c-->
![Vorwärtsdurchlauf (grau) und Rückwärtsdurchlauf (rot) eines über drei Zeitschritte ausgerollten linearen RNN. Bis auf den letzten erhält jeder versteckte Zustand die Summe zweier Gradientenbeiträge. Ein roter Pfeil führt von seiner eigenen Ausgabe über $\mathbf{W}_\textrm{hq}^\top$ zu ihm, der andere vom nächsten Zustand über $\mathbf{W}_\textrm{hh}^\top$.](../img/mdl-rnn-bptt.svg)

<!--en:483c38e1-->
Sei $\mathbf{G}_t = \partial L/\partial \mathbf{H}_t$ der Gradient des Gesamtverlusts bezüglich $\mathbf{H}_t$. Der Zustand $\mathbf{H}_t$ geht sowohl in seine unmittelbare Ausgabe $\mathbf{O}_t$ als auch in den nächsten versteckten Zustand $\mathbf{H}_{t+1}$ ein. Eine Multiplikation von rechts wird bei der Backpropagation zur Multiplikation mit der transponierten Gewichtsmatrix (:numref:`sec_backprop`). Daher erfüllt $\mathbf{G}_t$ die Rückwärtsrekursion

<!--en:73cf4fb4-->
entsprechend den beiden roten Pfeilen, die in :numref:`fig_rnn_bptt` an jedem versteckten Zustand eintreffen. Rollen wir :eqref:`eq_bptt_partial_L_ht_recur` von $\mathbf{G}_T$ bis $\mathbf{G}_t$ rückwärts aus, erhalten wir

<!--en:8dd8a3d1-->
Der Verlust im zukünftigen Schritt $j$ wirkt also auf den Zustand zurück, der $k = j - t$ Schritte zurückliegt. Dabei geht die $k^\textrm{th}$ Potenz von $\mathbf{W}_\textrm{hh}^\top$ ein. Bei einem allgemeinen nichtlinearen Übergang $\mathbf{H}_t = g(\mathbf{H}_{t-1}, \mathbf{X}_t, \mathbf{W}_\textrm{hh})$ tritt an die Stelle dieser Matrixpotenz das Produkt der Jacobi-Matrizen der $j - t$ dazwischenliegenden Zustandsübergänge, also $\prod_{i=t+1}^j \partial \mathbf{H}_i / \partial \mathbf{H}_{i-1}$. Haben wir $\mathbf{G}_1, \ldots, \mathbf{G}_T$ in einem einzigen Rückwärtsdurchlauf berechnet, erhalten wir jeden Parametergradienten als Summe über je ein äußeres Produkt pro Schritt:

<!--en:d041f7cf-->
Wie wir in :numref:`subsec_lin-alg-eigenvalues` und :numref:`sec_numerical_stability` gesehen haben, skaliert die $k^\textrm{th}$ Potenz einer Matrix jeden Eigenvektor um den Faktor $\lambda^k$. Sei $\rho$ der *Spektralradius* (der größte Betrag $|\lambda|$) von $\mathbf{W}_\textrm{hh}$. Dann skaliert die Norm von $(\mathbf{W}_\textrm{hh}^\top)^k$ asymptotisch wie $\rho^k$. Für $\rho < 1$ verschwinden Beiträge weit zurückliegender Schritte exponentiell, für $\rho > 1$ wachsen sie exponentiell an. Setzen wir die $\tanh$-Aktivierung wieder ein, multiplizieren wir den Gradienten in jedem Rückwärtsschritt zunächst elementweise mit der lokalen Ableitung $1 - \mathbf{H}_t^2 \in (0, 1]$ und anschließend mit $\mathbf{W}_\textrm{hh}^\top$. Wegen $1 - h^2 \leq 1$ kann $\tanh$ einen Gradienten nie verstärken. Geraten versteckte Einheiten nahe $\pm 1$ in Sättigung, schwächt die Aktivierung den Gradienten stark ab. Er kann dennoch explodieren, wenn $\mathbf{W}_\textrm{hh}$ groß ist und die Aktivierungen nahe null bleiben.

<!--en:7c396b80-->
Messen wir nun, was in unserem trainierten Sprachmodell aus :numref:`sec_rnn-scratch` tatsächlich geschieht. Zuerst laden wir den Datensatz `TimeMachine` mit Fenstern von je 32 Token.

<!--en:5361699a-->
Als Nächstes erzeugen wir eine Instanz unseres selbst implementierten rekurrenten Sprachmodells `RNNLMScratch` mit Einbettungsdimension $d = 512$, $h = 128$ versteckten Einheiten und Dropout.

<!--en:d30b4af5-->
Wir trainieren `model` über 10 Epochen mit der Lernrate `lr=4` und globalem Gradienten-Clipping mit dem Schwellenwert `gradient_clip_val=1`.

<!--en:a1dd3f4d-->
Um den Gradientenfluss über einen längeren Zeitraum zu untersuchen, ziehen wir aus den Validierungsdaten zufällig 256 Segmente `seqs` der Länge $L = 64$ (doppelte Trainingsfensterlänge) samt den jeweils folgenden Token `targets`.

<!--en:5ad30106-->
Nun führen wir `model.rnn` auf `seqs` vorwärts aus. Dabei erfassen wir die versteckten Zustände aller 64 Zeitschritte in `H`, die Logits des letzten Schritts in `logits` und die Gewichtsmatrizen `W_hh` und `W_hq`.

<!--en:a90e6313-->
Ausgehend vom Verlustgradienten $\partial L_L / \partial
\mathbf{O}_L = \textrm{softmax}(\mathbf{O}_L) - \mathbf{Y}_L$ im letzten Schritt gehen wir mit $\mathbf{G}_{t-1} = (\mathbf{G}_t \odot (1 -
\mathbf{H}_t^2))\mathbf{W}_\textrm{hh}^\top$ schrittweise in der Zeit zurück. Wir tragen das geometrische Mittel der relativen Gradientennorm $\|\mathbf{G}_{L-k}\| / \|\mathbf{G}_L\|$ gegen den zeitlichen Abstand $k$ auf.

<!--en:ba77ea2b-->
Die trainierte Matrix $\mathbf{W}_\textrm{hh}$ hat einen Spektralradius von etwa $1.8$. In einer rein linearen Rekurrenz würde der Gradient damit explodieren. Tatsächlich verschwindet er rasch. Innerhalb von etwa acht Schritten fällt seine Norm unter 1% ihres Anfangswerts, über 32 Token schrumpft sie auf wenige Milliardstel. Bei jedem Rückwärtsschritt bleibt etwas mehr als die Hälfte der Gradientennorm erhalten (aus dem ausgegebenen Wert beim Abstand 8 folgt $0.008^{1/8} \approx 0.55$). Das kommt der groben Schätzung $1.8 \times 0.27 \approx 0.5$ nahe, für die wir den Spektralradius mit der mittleren $\tanh$-Ableitung multiplizieren. Dieser Abfall passt zu unserer Beobachtung aus :numref:`sec_rnn-scratch`, dass die Kurve des positionsabhängigen Validierungsverlusts nach zwei oder drei Token Kontext abflacht. Er beweist aber nicht, dass verschwindende Gradienten die Ursache sind. Bei einem Abstand von 2 beziehungsweise 4 beträgt die Gradientennorm noch etwa ein Drittel beziehungsweise ein Zehntel ihres Anfangswerts. Mit SGD gelingt das Lernen bei einer solchen Abschwächung in der Regel noch. Auch das Kneser-Ney-Verfahren aus :numref:`sec_rnn-scratch` erreicht bei derselben Kontextlänge die Sättigung, obwohl es nur Häufigkeiten zählt und ganz ohne Gradienten auskommt. Was der Abfall ausschließt, ist das Lernen aus Token, die wesentlich weiter zurückliegen, unabhängig davon, welche Information die Daten enthalten.

<!--en:60d339e6-->
### Gegenmaßnahmen bei explodierenden und verschwindenden Gradienten

<!--en:0742aa08-->
Explodierende und verschwindende Gradienten erfordern grundverschiedene Gegenmaßnahmen. Bei einem explodierenden Gradienten kann schon ein einziger zu großer Aktualisierungsschritt die Parameter unbrauchbar machen. Dagegen bietet globales Gradienten-Clipping (:numref:`sec_rnn`) einen einfachen Schutz. Multiplizieren wir jedoch die gesamte Gradientensumme :eqref:`eq_bptt_partial_L_ht` mit einem Skalar, bleibt das *relative* Gewicht weit zurückliegender Beiträge gegenüber jüngeren Beiträgen gleich. Um verschwindenden Gradienten entgegenzuwirken, müssen wir daher die Rekurrenz selbst ändern. In den Richtungen, in denen der Zustand Information speichert, müssen die Jacobi-Matrizen annähernd wie die Identität wirken. Das gelingt mit *gattergesteuerten* rekurrenten Zellen wie LSTMs und GRUs, deren erlernte Gatter einen additiven Speicherpfad steuern. Eine weitere Möglichkeit sind *strukturierte lineare Rekurrenzen* und Zustandsraummodelle. Sie verzichten im Zustandsübergang auf die Nichtlinearität und beschränken die Eigenwerte auf die Nähe des Einheitskreises. Beide Modellfamilien behandeln wir in :numref:`chap_modern_rnn`.

<!--en:412f1bb0-->
## Abgeschnittene Backpropagation durch die Zeit

<!--en:7d0508f3-->
Werten wir :eqref:`eq_bptt_partial_L_ht_recur` über ein ganzes Korpus von $T$ Token aus, müssen wir alle $T$ versteckten Zustände des Vorwärtsdurchlaufs speichern und für eine einzige Parameteraktualisierung alle $T$ Schritte durchlaufen. Die *abgeschnittene Backpropagation durch die Zeit* begrenzt Speicherbedarf und Rechenaufwand pro Aktualisierung. Dazu zerlegen wir lange Datenströme in Segmente der festen Länge $T_0$ und stoppen die Backpropagation an jeder Segmentgrenze (:numref:`fig_truncated_bptt`).

<!--en:172987d3-->
![Vollständige und abgeschnittene Backpropagation durch die Zeit über eine Tokensequenz. Beide verwenden denselben Vorwärtsdurchlauf (grau). Bei der vollständigen Backpropagation läuft der Gradient (rot) die gesamte Sequenz entlang zurück. Die abgeschnittene Variante zerlegt die Sequenz in Segmente von je vier Token und trennt den Zustand an jeder Grenze vom Graphen ab. Sein Wert geht in das nächste Segment ein, doch über die Grenze fließt kein Gradient zurück.](../img/mdl-rnn-truncated-bptt.svg)

<!--en:e641541f-->
Wir können jedes Fenster mit $\mathbf{H}_0 = \mathbf{0}$ beginnen (wie in :numref:`sec_rnn-scratch`). Wir können auch den Endzustand des vorherigen Segments übernehmen und ihn dabei vom Berechnungsgraphen *abtrennen* (wie in :numref:`fig_truncated_bptt`). In beiden Fällen überschreitet kein Gradient eine Segmentgrenze. Der Lernhorizont ist damit auf höchstens $T_0$ Schritte begrenzt :cite:`Jaeger.2002`. In einem einfachen RNN schwächen verschwindende Gradienten das rückwärts laufende Signal jedoch schon nach etwa acht Schritten so stark ab, dass es praktisch erlischt. Das geschieht weit innerhalb unseres Fensters von 32 Token. Ein längeres Fenster allein erweitert daher das effektive Gedächtnis des Modells nicht.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:2a0b99a0-->
Bei der Backpropagation durch die Zeit wenden wir die übliche Backpropagation auf ein ausgerolltes RNN an. Wir übertragen die Sensitivitäten des Verlusts gemäß :eqref:`eq_bptt_partial_L_ht_recur` rückwärts und summieren die Parametergradienten über alle Zeitschritte (:eqref:`eq_bptt_param_grads`). Läuft ein Gradient $k$ Schritte zurück, wird er mit dem Produkt von $k$ Jacobi-Matrizen der Zustandsübergänge multipliziert. Im linearen RNN reduziert sich dieses Produkt auf eine Matrixpotenz mit dem asymptotischen Verhalten $(\mathbf{W}_\textrm{hh}^\top)^k \sim \rho^k$. Daher explodieren Gradienten für $\rho > 1$ und verschwinden für $\rho < 1$. Verwenden wir $\tanh$, kommt in jedem Schritt der Faktor $1 - \mathbf{H}_t^2 \in (0, 1]$ hinzu. Dadurch sinkt die Gradientennorm in unserem trainierten RNN innerhalb von etwa acht Schritten auf unter 1% ihres Ausgangswerts, trotz $\rho \approx 1.8$. Gradienten-Clipping begrenzt explodierende Spitzen. Abgeschnittene BPTT begrenzt Speicherbedarf und Rechenaufwand pro Aktualisierung. Um Gradienten über lange Zeiträume zu erhalten, brauchen wir Änderungen an der Architektur (:numref:`chap_modern_rnn`).

<!--en:2294243a-->
## Übungen

<!--en:2e4e3622-->
1. **Rückwärtsdurchlauf von Hand.** Ein Schritt von `RNNScratch` (:numref:`sec_rnn-scratch`) berechnet
   $$\mathbf{H}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}).$$
   Sei $\mathbf{G}_t = \partial L / \partial \mathbf{H}_t$ der Gradient, der aus späteren Schritten und von der Ausgabe im Schritt $t$ eintrifft.
    1. Leiten Sie die Beiträge dieses Schritts zu $\partial L / \partial \mathbf{W}_{\textrm{xh}}$, $\partial L / \partial \mathbf{W}_{\textrm{hh}}$, $\partial L / \partial \mathbf{b}_\textrm{h}$ und $\partial L / \partial \mathbf{H}_{t-1}$ als Funktionen von $\mathbf{G}_t$, $\mathbf{X}_t$, $\mathbf{H}_t$, $\mathbf{H}_{t-1}$ und $\mathbf{W}_{\textrm{hh}}$ her.
    1. Addieren Sie den Beitrag der Ausgabe im Schritt $t-1$ zum Beitrag zu $\partial L / \partial \mathbf{H}_{t-1}$. So erhalten Sie eine Rekursion für $\mathbf{G}_{t-1}$. Zeigen Sie, dass sie sich auf :eqref:`eq_bptt_partial_L_ht_recur` reduziert, wenn Sie $\tanh$ und den Bias weglassen. Nennen Sie den Faktor, der durch $\tanh$ in jedem Schritt hinzukommt. Erklären Sie auch, wie sich die Parametergradienten über die $T$ Schritte eines Fensters aufsummieren, und vergleichen Sie mit :eqref:`eq_bptt_param_grads`.

    *Nach CMU 11-785, [Hausaufgabe 3, Teil 1](https://deeplearning.cs.cmu.edu/F22/document/homework/HW3/HW3P1_F22.pdf).*
   :exlabel:`ex_bptt-backward-pass-hand`

<!--en:d6b83c5d-->
1. **Potenziteration und Ausrichtung.** Sei $\mathbf{M} \in \mathbb{R}^{h \times h}$ symmetrisch mit Eigenwerten $\lambda_i$ und Eigenvektoren $\mathbf{v}_i$ für $i = 1, \ldots, h$, geordnet nach $|\lambda_i| \geq |\lambda_{i+1}|$.
    1. Zeigen Sie, dass $\mathbf{M}^k$ die Eigenwerte $\lambda_i^k$ mit denselben Eigenvektoren hat.
    1. Betrachten Sie einen Zufallsvektor $\mathbf{x} \in \mathbb{R}^h$ mit unabhängigen, standardnormalverteilten Einträgen. Zeigen Sie, dass sich $\mathbf{M}^k \mathbf{x}$ an $\mathbf{v}_1$ ausrichtet, wenn $k$ zunimmt. Geben Sie an, welche Bedingung die Eigenwerte dafür erfüllen müssen und in welchem Sinne die Ausrichtung gilt.
    1. Was folgt daraus für den Gradienten :eqref:`eq_bptt_partial_L_ht` eines RNN über die Änderung seiner Norm hinaus?

<!--en:1eb79e18-->
1. [code] **Implementierung des Rückwärtsdurchlaufs.** ● Trainieren Sie `d2l.RNNLMScratch` auf `d2l.TimeMachine` wie in :numref:`sec_rnn-scratch`. Verwenden Sie die Gradienten eines einzelnen Schritts und die Rekursion aus :exref:`ex_bptt-backward-pass-hand`, um den Rückwärtsdurchlauf der `RNNScratch`-Zelle über ein ganzes Fenster zu implementieren. Nutzen Sie ausschließlich Tensoroperationen. Gehen Sie vom Gradienten des Verlusts bezüglich der Logits aus (:numref:`subsec_softmax_and_derivatives`). Führen Sie den Rückwärtsdurchlauf für einen Trainingsminibatch aus. Vergleichen Sie Ihre Gradienten für `W_xh`, `W_hh` und `b_h` mit den Gradienten, die automatisches Differenzieren liefert.

    *Nach CMU 11-785, [Hausaufgabe 3, Teil 1](https://deeplearning.cs.cmu.edu/F22/document/homework/HW3/HW3P1_F22.pdf).*

<!--en:657be6de-->
1. [code] **Jacobi-Matrizen nichtlinearer Übergänge.** ● Ziehen Sie eine zufällige symmetrische $100 \times 100$-Matrix und skalieren Sie sie auf jeden der Spektralradien $\rho \in \{0.9, 1.0, 1.1\}$. Tragen Sie die Norm ihrer $k^\textrm{th}$ Potenz, also $\|\mathbf{J}^k\|$, gegen $k$ auf. Verwenden Sie für die Norm eine logarithmische Achse. Sie erhalten drei Geraden, wie es die Eigenwerte vorhersagen. Verlassen Sie nun den symmetrischen linearen Fall.
    1. Ersetzen Sie die symmetrische Matrix durch eine allgemeine, nicht symmetrische Matrix $\mathbf{W}_\textrm{hh}$. Skalieren Sie sie auf jeden der drei Radien und tragen Sie $\|\mathbf{J}^k\|$ gegen $k$ auf. Geben Sie den größten Wert von $\|\mathbf{J}^k\|$ für $\rho = 0.9$ und den zeitlichen Abstand an, bei dem er auftritt. Erklären Sie, warum das Verhältnis $\|\mathbf{J}^k\| / \rho^k$ größer als eins sein kann und warum es bei wachsendem $k$ beschränkt bleibt (eine Zufallsmatrix ist mit Wahrscheinlichkeit eins diagonalisierbar).
    1. Kehren Sie zur symmetrischen Matrix $\mathbf{W}_\textrm{hh}$ zurück. Führen Sie das $\tanh$-RNN
       $$\mathbf{H}_t = \tanh(\mathbf{H}_{t-1}\mathbf{W}_\textrm{hh} + \mathbf{X}_t), \qquad \mathbf{H}_0 = \mathbf{0},$$
       für eine einzelne Sequenz ($\mathbf{H}_t \in \mathbb{R}^{1 \times h}$) mit Eingaben $\mathbf{X}_t$ aus, deren Einträge Sie unabhängig aus $\mathcal{N}(0, s^2)$ ziehen. Tragen Sie für die beiden Eingabeskalen die Norm des Produkts der Jacobi-Matrizen $\partial \mathbf{H}_j / \partial \mathbf{H}_{j-1}$ entlang der Trajektorie auf. Verwenden Sie $s = 0.1$ und $s = 1$. Wie verändern die Nichtlinearität und die Eingabeskala das Wachstum der Norm und warum?
    1. Erklären Sie, warum das Produkt schrumpft, wenn der größte Singulärwert von $\mathbf{W}_\textrm{hh}$ kleiner als eins ist. Warum ist ein größter Singulärwert über eins für das Wachstum notwendig, aber nicht hinreichend :cite:`bengio1994learning,Pascanu.Mikolov.Bengio.2013`? Setzen Sie diese Überlegung in Beziehung zu Ihren $\tanh$-Kurven für $\rho = 0.9$ und $\rho = 1.1$.

<!--en:28c06f49-->
1. [code] **Fensterlänge und Reichweite des Gedächtnisses.** ● Im Modell aus :numref:`subsec_bptt_analysis` sinkt die Norm des Gradienten des letzten Verlusts innerhalb von etwa acht Schritten unter 1% ihres Anfangswerts. Das liegt weit innerhalb des Trainingsfensters von 32 Token.
    1. Überlegen Sie vor dem Ausführen des Codes, ob Sie bei erneutem Training mit `num_steps=128` eine Änderung dieser Reichweite erwarten. Begründen Sie Ihre Antwort mit dem Produkt der Jacobi-Matrizen der Zustandsübergänge in :eqref:`eq_bptt_partial_L_ht`.
    1. Trainieren Sie das Modell erneut mit dem Wert 128 für `num_steps`. Messen Sie den Gradienten in Abhängigkeit vom zeitlichen Abstand auf Abschnitten von je 256 Validierungstoken. Vergleichen Sie das Ergebnis mit Ihrer Vorhersage.
    1. Wiederholen Sie für das neu trainierte Modell die Messung des positionsabhängigen Verlusts aus :numref:`sec_rnn-scratch`. Sinkt der Verlust nun auch nach dem dritten Kontexttoken weiter?

<!--en:a44b5276-->
[Diskussion](https://d2l.discourse.group/t/334)

<!--en:e035fd62-->
Backpropagation durch die Zeit<br>**ausgerollte Berechnungsgraphen · Produkte von Jacobi-Matrizen · Matrixpotenzen und Spektralradius · verschwindende und explodierende Gradienten · abgeschnittene BPTT**

<!--en:15559302-->
::: {.slide title="Zwei Beobachtungen mit derselben mathematischen Ursache"}

<!--en:cb967ea7-->
[Motivation und skalares Beispiel]{.kicker}

<!--en:4523d2e4-->
- **Wiederholte Multiplikation über die Zeit:** In der eindimensionalen linearen Rekurrenz $h_t = w_{\textrm{h}} h_{t-1} + x_t$ skaliert die Sensitivität über einen Abstand von $k$ Schritten geometrisch mit $w_{\textrm{h}}^k$.

<!--en:9e2d85dd-->
- **Wie viel nach 40 Schritten vom Gradienten bleibt:** Für $w_{\textrm{h}} = 0.9$ schrumpft er über 40 Rückwärtsschritte auf $0.9^{40} \approx 0.015$ (**verschwindet**), für $w_{\textrm{h}} = 1.1$ wächst er auf $1.1^{40} \approx 45.3$ (**explodiert**).

<!--en:3e74c16d-->
- **Backpropagation durch die Zeit (BPTT):** Ausrollen der Rekurrenz über $T$ Schritte und automatisches Differenzieren im Rückwärtsmodus erklären beide Phänomene in Matrixform.

<!--en:4ea3f0f1-->
::: {.slide title="Ausrollen einer Rekurrenz zu einem Graphen mit gemeinsamen Gewichten"}

<!--en:51621992-->
[Struktur des Berechnungsgraphen]{.kicker}

<!--en:8c779f72-->
- **Gewichtsbindung über alle Schichten:** Ausrollen über $T$ Schritte ergibt ein Netz der Tiefe $T$, das dasselbe Gewicht $w_{\textrm{h}}$ in jedem Schritt $t \in \{1, \ldots, T\}$ verwendet.

<!--en:2ef4b4e3-->
- **Totale Ableitung bei Gewichtsteilung:** Nach der mehrdimensionalen Kettenregel ergibt sich der Gradient bezüglich $w_{\textrm{h}}$ als **Summe** der lokalen Beiträge aus allen $T$ Schritten.

<!--en:cf96696c-->
- **Rekursive Sensitivität:** Ändert sich $w_{\textrm{h}}$, so ändert sich $h_t = g(h_{t-1}, x_t, w_{\textrm{h}})$ sowohl direkt im Schritt $t$ als auch indirekt über $h_{t-1}$.

<!--en:a2ad175a-->
::: {.slide title="Die Kettenregel über die Zeit und die Vorwärtsrekursion der Sensitivität"}

<!--en:fb14519d-->
[Entwicklung als Produkt von Jacobi-Matrizen]{.kicker}

<!--en:786394d0-->
- **Sensitivitätsrekursion erster Ordnung:** Die Kettenregel liefert $\frac{\partial h_t}{\partial w_{\textrm{h}}} = u_t + c_t \frac{\partial h_{t-1}}{\partial w_{\textrm{h}}}$ mit dem direkten Beitrag $u_t = \partial g_t / \partial w_{\textrm{h}}$ und der Jacobi-Matrix $c_t = \partial g_t / \partial h_{t-1}$ des jeweiligen Schritts.

<!--en:a795fbbb-->
- **Vorteil der BPTT im Rückwärtsmodus:** Die getrennte Auswertung von $\prod_{j=i+1}^t c_j$ für jedes Paar $(i, t)$ kostet $O(T^2)$ Produkte von Jacobi-Matrizen. Automatisches Differenzieren im Rückwärtsmodus faktorisiert die Summe in einem einzigen Durchlauf mit Aufwand $O(T)$.

<!--en:feea68f2-->
::: {.slide title="Vorwärts- und Rückwärtsdurchlauf im ausgerollten linearen RNN"}

<!--en:59e69589-->
[BPTT in Matrixform]{.kicker}

<!--en:1732734a-->
Die genaue Matrixstruktur des Rückwärtsdurchlaufs zeigt sich an einem linearen RNN für einen Minibatch ($\phi = \textrm{id}$, alle Bias-Terme null) mit Eingaben $\mathbf{X}_t \in \mathbb{R}^{b \times d}$, Zuständen $\mathbf{H}_t \in \mathbb{R}^{b \times h}$ und Ausgaben $\mathbf{O}_t \in \mathbb{R}^{b \times q}$:

<!--en:cad8d85e-->
- **Gleichungen des Vorwärtsdurchlaufs (graue Pfeile):**
  $$\mathbf{H}_t = \mathbf{X}_t\mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1}\mathbf{W}_{\textrm{hh}}$$
  $$\mathbf{O}_t = \mathbf{H}_t\mathbf{W}_{\textrm{hq}}$$

<!--en:c94ed2a5-->
- **Zwei Nachfolger je verstecktem Zustand:** Im ausgerollten Graphen geht jeder Zustand $\mathbf{H}_t$ (für $t < T$) in zwei Knoten ein, in seine unmittelbare Ausgabe $\mathbf{O}_t$ (über $\mathbf{W}_{\textrm{hq}}$) und in den nächsten Zustand $\mathbf{H}_{t+1}$ (über $\mathbf{W}_{\textrm{hh}}$).

<!--en:27d22530-->
- **Rückwärtsdurchlauf (rote Pfeile):** Bei der Differentiation im Rückwärtsmodus ist die adjungierte Größe $\mathbf{G}_t = \partial L / \partial \mathbf{H}_t$ die Summe der Beiträge der beiden roten Pfeile, die am Knoten $\mathbf{H}_t$ eintreffen.

<!--en:1765444d-->
::: {.slide title="Rückwärtsrekursion für die Gradienten der versteckten Zustände"}

<!--en:a925d76d-->
[Dynamische Programmierung im Rückwärtsmodus]{.kicker}

<!--en:d60d608c-->
- **Zwei eingehende Pfade bei $\mathbf{H}_t$:** Der Zustand $\mathbf{H}_t$ geht sowohl in $\mathbf{O}_t$ als auch in $\mathbf{H}_{t+1}$ ein. Seine adjungierte Größe $\mathbf{G}_t = \partial L / \partial \mathbf{H}_t \in \mathbb{R}^{b \times h}$ summiert die lokale Sensitivität der Ausgabe und die rückwärts übertragene adjungierte Größe des Folgeschritts (mit $\mathbf{G}_{T+1} = \mathbf{0}$).

<!--en:bb10d0ce-->
- **Ein Rückwärtsdurchlauf mit Aufwand $O(T)$:** Jeder Schritt von $t = T$ bis $1$ erfordert nur eine Matrixmultiplikation (GEMM) der Form $(b \times h) \times (h \times h)$ mit $\mathbf{W}_{\textrm{hh}}^\top$.

<!--en:3cc52e3f-->
::: {.slide title="Ausrollen von G_t über die Zeit mit Matrixpotenzen"}

<!--en:1c69ce36-->
[Explizite Entwicklung nach Matrixpotenzen]{.kicker}

<!--en:ba8f388f-->
- **Operator für den Abstand $k$:** Ein Verlustsignal aus Schritt $j$ erreicht über die $k^\textrm{th}$ Matrixpotenz $(\mathbf{W}_{\textrm{hh}}^\top)^k$ ($k = j - t$) den Zustand $\mathbf{H}_t$.

<!--en:e8de1fc6-->
- **Summe äußerer Produkte:** Liegen $\mathbf{G}_{1:T}$ vor, ergibt sich jeder Gewichtsgradient als Summe über je ein äußeres Produkt pro Schritt.

<!--en:1ca26e94-->
::: {.slide title="Der Spektralradius bestimmt das Wachstum der Matrixpotenzen"}

<!--en:6de34a91-->
[Analyse der Eigenwerte und Singulärwerte]{.kicker}

<!--en:4c93226c-->
- **Asymptotisch exponentielle Skalierung ($\|(\mathbf{W}_{\textrm{hh}}^\top)^k\| \sim \rho^k$):** Sei $\rho = \max_i |\lambda_i(\mathbf{W}_{\textrm{hh}})|$ der **Spektralradius**. Beim Erheben von $\mathbf{W}_{\textrm{hh}}^\top$ in die $k^\textrm{th}$ Potenz skaliert jeder Eigenvektor mit $\lambda_i^k$.
  - **Verschwinden ($\rho < 1$):** $\|(\mathbf{W}_{\textrm{hh}}^\top)^k\| \to 0$ mit exponentiellem Abfall.
  - **Explosion ($\rho > 1$):** Im führenden Eigenraum wachsen Vektoren wie $\rho^k \to \infty$.
  - **Grenzfall ($\rho = 1$):** Nur für $|\lambda_i| = 1$ bleibt die Norm über lange Zeit erhalten.

<!--en:d26f64dd-->
- **Vorübergehendes Wachstum bei nichtnormalen Matrizen:** Ist $\mathbf{W}_{\textrm{hh}}$ asymmetrisch mit $\sigma_{\max}(\mathbf{W}_{\textrm{hh}}) > 1 > \rho$, können die Normen bei kleinen Abständen zunächst ansteigen, bevor der Abfall gemäß $\rho^k$ einsetzt.

<!--en:446273d8-->
::: {.slide title="Wie die tanh-Nichtlinearität die lineare BPTT verändert"}

<!--en:ebd463cf-->
[Jacobi-Matrizen nichtlinearer Zustandsübergänge]{.kicker}

<!--en:249fccea-->
- **Lokale Kontraktion:** Wegen $H_{t,i} \in (-1, 1)$ liegt jeder Eintrag von $\tanh'(z) = 1 - \mathbf{H}_t^2$ in $(0, 1]$ und verstärkt den Gradienten nie.

<!--en:e6891d48-->
- **Sättigung beschleunigt das Verschwinden:** Gerät eine versteckte Einheit nahe $\pm 1$ in Sättigung, kommt der rückwärts gerichtete Gradientenfluss wegen $1 - H_{t,i}^2 \approx 0$ fast zum Erliegen.

<!--en:ec353276-->
- **Effektiver Faktor pro Schritt:** Selbst bei $\rho(\mathbf{W}_{\textrm{hh}}) > 1$ kann der resultierende Faktor $\overline{(1 - H^2)} \cdot \rho(\mathbf{W}_{\textrm{hh}})$ pro Schritt deutlich unter $1$ liegen.

<!--en:200ac4fa-->
::: {.slide title="Versuchsaufbau zur Reichweite der Gradienten (L = 64)"}

<!--en:ff4f0b47-->
[Messung am trainierten RNN]{.kicker}

<!--en:250e4d3a-->
Wie weit reichen Gradienten in unserem trainierten Sprachmodell aus §{sec_rnn} ($d = 512, h = 128$, Dropout-Rate $0.5$, Clipping bei $\theta = 1$)? Zur Messung durchläuft das trainierte Netz in Vorwärtsrichtung 256 Validierungssegmente der Länge $L = 64$ Token:

<!--en:627e7c1f-->
- **Isolierte Sensitivität beim Abstand $k$:** Vom Kreuzentropieverlust an der letzten Position $t = L$ ausgehen und $\mathbf{G}_t$ mit $\mathbf{G}_{t-1} = (\mathbf{G}_t \odot (1 - \mathbf{H}_t^2))\mathbf{W}_{\textrm{hh}}^\top$ über 64 Schritte zurückführen, ohne die Verluste dazwischenliegender Ausgaben zu addieren.

<!--en:7ca5c6b8-->
::: {.slide title="Gemessener Gradientenabfall im trainierten RNN"}

<!--en:84c7aa8a-->
[Ergebnisse der Messung]{.kicker}

<!--en:2f5122d5-->
Die Kurve des geometrischen Mittels der relativen Gradientennorm $\|\mathbf{G}_{L-k}\| / \|\mathbf{G}_L\|$ über dem zeitlichen Abstand $k \in \{0, \ldots, 63\}$ erklärt die kurze Reichweite aus §{sec_rnn}:

<!--en:ae347504-->
- **Große Gewichte, rasch verschwindende Gradienten:** Obwohl $\rho(\mathbf{W}_{\textrm{hh}}) \approx 1.80$ gilt, senkt die Sättigung von $\tanh$ die mittlere Ableitung auf $\overline{1 - H^2} \approx 0.27$.

<!--en:9551ad91-->
- **$\approx 0.55\times$ pro Schritt:** Die Gradienten schrumpfen pro Schritt um fast die Hälfte ($1.80 \times 0.27 \approx 0.49$). Sie fallen **unter $1\%$ beim Abstand $k = 8$** und auf $\sim 10^{-9}$ beim Abstand 32.

<!--en:8d6f10ae-->
- **Kurzes effektives Gedächtnis:** Bei Abständen von 2 bis 4 bleiben $10\text{--}35\%$ des Signals erhalten (hier erreicht auch das Kneser-Ney-Verfahren die Sättigung). Jenseits von 8 Schritten erreicht fast kein Gradient mehr $\mathbf{W}_{\textrm{hh}}$.

<!--en:7e351ec2-->
::: {.slide title="Warum Gradienten-Clipping nur bei explodierenden Gradienten hilft"}

<!--en:0b812df0-->
[Unterschiede zwischen den beiden Gradientenproblemen]{.kicker}

<!--en:e9eb2e91-->
- **Schutz vor explodierenden Spitzen:** Führt eine Trajektorie ohne Sättigung zu $\|\mathbf{g}\| \gg \theta$, so begrenzt die Skalierung mit $\theta/\|\mathbf{g}\|$ die Schrittweite auf $\eta\theta$. So verhindert sie ein Auseinanderlaufen der Parameter.

<!--en:3b61fa10-->
- **Grenze des Clippings bei verschwindenden Beiträgen:** Ein Skalar vor der gesamten Gradientensumme $\mathbf{g}$ skaliert die Terme für die Abstände $0$ und $20$ gleichermaßen. Das Verhältnis $1 : 10^{-6}$ zwischen Signalen aus nahen und weit zurückliegenden Schritten bleibt erhalten.

<!--en:3bf611a7-->
::: {.slide title="Gatter und lineare Rekurrenzen als architektonische Gegenmaßnahmen"}

<!--en:2e595180-->
[Ausblick auf das Kapitel über Zustandsraummodelle]{.kicker}

<!--en:5bd64a85-->
Gegen verschwindende Gradienten hilft eine Änderung des rekurrenten Zustandsübergangs $g$. Seine Jacobi-Matrix $\mathbf{J}_t = \partial \mathbf{H}_t / \partial \mathbf{H}_{t-1}$ soll in den für das Gedächtnis relevanten Richtungen Eigenwerte nahe $1$ haben:

<!--en:d4306b64-->
| Architekturfamilie | Prinzip des Zustandsübergangs | Verhalten der Jacobi-Matrizen relativ zur Identität ($\mathbf{J}_t \approx \mathbf{I}$) |

<!--en:4308db6a-->
| **Einfaches $\tanh$-RNN (§{sec_rnn})** | $\mathbf{H}_t = \tanh(\mathbf{X}_t\mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1}\mathbf{W}_{\textrm{hh}} + \mathbf{b}_{\textrm{h}})$ | Bei Sättigung der Einheiten verkleinert $\mathbf{J}_t = \textrm{diag}(1 - \mathbf{H}_t^2)\mathbf{W}_{\textrm{hh}}^\top$ den Gradienten pro Schritt um den Faktor $\sim 0.5\times$ |

<!--en:2e2b0097-->
| **Gattergesteuerte RNNs (LSTM, GRU, Abschnitt über gattergesteuerte rekurrente Netze)** | Additiver Zellzustand $\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t$ mit erlernten Sigmoidgattern $\mathbf{F}_t \in (0, 1)^h$ | Für ein Vergessensgatter mit $\mathbf{F}_t \approx \mathbf{1}$ gilt $\partial \mathbf{C}_t / \partial \mathbf{C}_{t-1} = \textrm{diag}(\mathbf{F}_t) \approx \mathbf{I}$. So entsteht ein Pfad, auf dem der Gradient ungeschwächt fließt |

<!--en:566c7afb-->
| **Strukturierte Zustandsraummodelle (SSMs bzw. lineare RNNs, Abschnitte über lineare Rekurrenzen und Zustandsraummodelle)** | Lineare Zustandsrekurrenz $\mathbf{h}_t = \mathbf{A}\mathbf{h}_{t-1} + \mathbf{B}\mathbf{x}_t$ mit nichtlinearen Ausgabeschichten | Die rekurrente $\tanh$-Sättigung entfällt. Die Eigenwerte von $\mathbf{A}$ sind auf die Einheitskreisscheibe beschränkt ($|\lambda_i| \le 1$) |

<!--en:6830b809-->
::: {.slide title="Speicherbedarf und Rechenaufwand der BPTT über die gesamte Sequenz"}

<!--en:1076c29f-->
[Engpass bei langen Datenströmen]{.kicker}

<!--en:e83275d1-->
- **Der Aktivierungsspeicher wächst linear mit $T$ ($O(Tbh)$):** Der Rückwärtsdurchlauf in Schritt $t$ benötigt sowohl $\mathbf{H}_t$ (für $1 - \mathbf{H}_t^2$) als auch $\mathbf{H}_{t-1}$ (für $\mathbf{H}_{t-1}^\top \mathbf{G}_t$). Beim automatischen Differenzieren im Rückwärtsmodus müssen alle $T$ Zustände bis zum Ende des Durchlaufs im GPU-Speicher bleiben.

<!--en:29396e09-->
- **Latenz pro Aktualisierung von der Ordnung $O(T)$:** Eine vollständige BPTT über *The Time Machine* ($T = 65{,}741$ Token) bräuchte $65{,}741$ Vorwärts- und $65{,}741$ Rückwärtsschritte für **eine einzige Aktualisierung der Gewichte**.

<!--en:4801353c-->
::: {.slide title="Abgeschnittene BPTT und Abtrennen des Zustands an Segmentgrenzen"}

<!--en:60f8f826-->
[Begrenzter Speicherbedarf und Rechenaufwand pro Aktualisierung]{.kicker}

<!--en:957d7954-->
Die **abgeschnittene Backpropagation durch die Zeit (TBPTT)** zerlegt eine lange Sequenz in Segmente der festen Länge $T_0$ (z. B. $T_0 = 32$ Token). Der Rückwärtsdurchlauf endet an jeder Segmentgrenze:

<!--en:d7729d4b-->
- **Der Zustand läuft vorwärts weiter, der Gradient stoppt:** Bei aufeinanderfolgenden Segmenten dient der Endzustand $\mathbf{H}_{T_0}$ als Anfangszustand des nächsten Segments. Das vorherige **Abtrennen** vom Autograd-Graphen stoppt seinen Gradienten.

<!--en:81dfcba7-->
- **Speicherbedarf $O(T_0 bh)$ und häufige Aktualisierungen:** Autograd gibt den Berechnungsgraphen jedes Segments unmittelbar nach der Aktualisierung frei. So genügt Speicher für $T_0$ Zustände.

<!--en:dda28e5c-->
::: {.slide title="Statistische Folgen der Wahl der Segmentlänge"}

<!--en:2f19acaa-->
[Was beim Abschneiden verloren geht]{.kicker}

<!--en:3e5ef95a-->
Backpropagation in Fenstern der Länge $T_0$ begrenzt den Speicherbedarf und erlaubt es, Minibatches zu mischen und parallel zu verarbeiten. Dafür lässt sich der Verlust über den Gradienten nur noch zeitlich nahen Schritten zuschreiben:

<!--en:c7e246ec-->
- **Feste Grenze von $T_0$ Schritten für die Zuordnung:** In einem Segment der Länge $T_0 = 32$ läuft der Gradient des Verlusts an Position $j \in \{1, \ldots, 32\}$ über $j - 1$ Schritte innerhalb des Segments zurück (höchstens $31$ Schritte, im Mittel $15.5$). **Kein Gradient überschreitet die abgetrennte Grenze** zum vorherigen Segment.

<!--en:8fb22010-->
- **Nullzustand oder Fortführung des abgetrennten Zustands:** In §{sec_rnn} zog `d2l.TimeMachine` gemischte Fenster von 32 Token mit Anfangszustand $\mathbf{H}_0 = \mathbf{0}$. Damit sind die Minibatches für SGD unabhängig und identisch verteilt. Bei fortlaufender Verarbeitung liefert $\mathbf{H}_{T_0}$ über Segmentgrenzen hinweg Kontext für die Inferenz. Der Gradient stoppt weiterhin bei $T_0$.

<!--en:9a9cb213-->
- **Welche Grenze greift zuerst?** In unserem einfachen $\tanh$-RNN fallen Gradienten bereits unter $1\%$, sobald der Abstand $k = 8 \ll T_0 = 32$ erreicht ist. Die Abschwächung durch die Jacobi-Matrizen der Zelle ist also schon innerhalb des Fensters von 32 Token der entscheidende Engpass.

<!--en:ed072157-->
::: {.slide title="Verständnisfragen zu BPTT, Jacobi-Matrizen, Clipping und Gradientenstopp"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:de28a550-->
Fragen zu Matrixpotenzen, zum Abtrennen vom Berechnungsgraphen und zur Reichweite von Gradienten:

<!--en:83dedc98-->
1. **Sensitivität über ein Fenster von 32 Token:** Wie viele Jacobi-Matrizen liegen in einem linearen RNN mit einem Fenster von $T_0 = 32$ Token zwischen dem letzten Verlust bei $t = 32$ und dem Anfangszustand $\mathbf{H}_1$? Mit welchem Faktor skaliert die Sensitivität für $\rho(\mathbf{W}_{\textrm{hh}}) = 0.9$ und für $\rho(\mathbf{W}_{\textrm{hh}}) = 1.1$?

<!--en:6e08b242-->
2. **Fehlender Gradientenstopp:** Ein RNN trainiert auf einer GPU über 1000 aufeinanderfolgende Segmente von je 32 Token. Dabei geht `state` aus Segment $s$ in Segment $s+1$ über. Am Ende jedes Segments fehlt jedoch der Aufruf, der `state` vom Graphen abtrennt und so den Gradienten stoppt. Wie entwickeln sich GPU-Speicherbedarf und Laufzeit des Rückwärtsdurchlaufs für Segment $s = 1, 2, 3, \ldots$?

<!--en:41775312-->
3. **Größeres Fenster beim einfachen RNN:** Das einfache $\tanh$-RNN aus §{sec_rnn} trainiert erneut mit `num_steps = 128` anstelle von `num_steps = 32`. Sinkt der positionsabhängige Validierungsverlust nun auch nach Token 3 weiter? Warum oder warum nicht?

<!--en:c48c9bf0-->
::: {.slide title="Verständnisfragen zu BPTT, Jacobi-Matrizen, Clipping und Gradientenstopp (Lösungen)"}

<!--en:929b7671-2-->
[Wissensprüfung]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:85faa7e0-->
1. **$31$ Jacobi-Matrizen:** Bei $\rho = 0.9$ beträgt der Faktor $0.9^{31} \approx 0.038$ ($96\%$ Verlust), bei $\rho = 1.1$ beträgt er $1.1^{31} \approx 19.2$. Variiert $\rho$ um $\pm 10\%$ gegenüber $1.0$, unterscheiden sich die Gradientennormen über das Fenster von 32 Schritten um den Faktor **$500\times$**.

<!--en:c0b51654-->
2. **Unbegrenztes Wachstum des Graphen (Speicher erschöpft):** Ohne Abtrennen bleibt Segment $s$ mit den Berechnungsgraphen aller vorherigen Segmente $1, \ldots, s-1$ verbunden. Laufzeit des Rückwärtsdurchlaufs und GPU-Speicherbedarf wachsen mit jedem Schritt linear gemäß $O(s \cdot T_0)$, bis der Speicher beim Rückwärtsdurchlauf erschöpft ist.

<!--en:8593d284-->
3. **Nein:** Unabhängig von der Fensterlänge sinken die Gradienten beim gemessenen Faktor von $\approx 0.55$ pro Schritt innerhalb von 8 Schritten unter $1\%$. Die Vergrößerung des Fensters $T_0$ gemäß $32 \to 128$ liefert daher kein zusätzliches nutzbares Gradientensignal von weit zurückliegenden Token.

<!--en:e91860ab-->
::: {.slide title="Zusammenfassung der Backpropagation durch die Zeit (BPTT)"}

<!--en:408fcbdb-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:45722085-->
| Konzept oder Verfahren | Mathematischer Ausdruck | Skalierung und asymptotisches Verhalten | Praktische Konsequenz |

<!--en:7a492e95-->
| **Lineare Rückwärtsrekursion** | $\mathbf{G}_t = \nabla_{\mathbf{O}_t} L\,\mathbf{W}_{\textrm{hq}}^\top + \mathbf{G}_{t+1}\mathbf{W}_{\textrm{hh}}^\top$ | Rückwärtsdurchlauf mit Aufwand $O(T)$ ($1$ GEMM pro Schritt) | Exaktes automatisches Differenzieren im Rückwärtsmodus auf dem ausgerollten Graphen |

<!--en:411a1b53-->
| **Ausgerollte Rekurrenz für den Abstand $k$** | $\mathbf{G}_t = \sum_{j=t}^T \nabla_{\mathbf{O}_j} L\,\mathbf{W}_{\textrm{hq}}^\top (\mathbf{W}_{\textrm{hh}}^\top)^{j-t}$ | $\|(\mathbf{W}_{\textrm{hh}}^\top)^k\| \sim \rho(\mathbf{W}_{\textrm{hh}})^k$ | $\rho < 1 \implies$ verschwindende Gradienten, $\rho > 1 \implies$ explodierende Gradienten |

<!--en:629e20c2-->
| **Nichtlineare BPTT mit $\tanh$** | $\mathbf{G}_{t-1} = \nabla_{\mathbf{O}_{t-1}} L\,\mathbf{W}_{\textrm{hq}}^\top + (\mathbf{G}_t \odot (1 - \mathbf{H}_t^2))\mathbf{W}_{\textrm{hh}}^\top$ | Gemessener Abklingfaktor $\approx 0.55$ pro Schritt (grobe Schätzung $\overline{(1 - H^2)}\,\rho \approx 0.49$) | Die Gradientennorm sinkt unter $1\%$ in $\sim 8$ Schritten, trotz $\rho \approx 1.8$ |

<!--en:d8f2b09c-->
| **Gradienten gemeinsamer Parameter** | $\nabla_{\mathbf{W}_{\textrm{hh}}} L = \sum_{t=1}^T \mathbf{H}_{t-1}^\top \mathbf{G}_t$, $\nabla_{\mathbf{W}_{\textrm{xh}}} L = \sum_{t=1}^T \mathbf{X}_t^\top \mathbf{G}_t$ | Summe aus $T$ äußeren Produkten, erfordert das Speichern von $\mathbf{H}_{1:T}$ | Bei verschwindenden Gradienten dominieren Terme mit kleinem Abstand ($j - t \le 3$) |

<!--en:a5323c02-->
| **Globales Gradienten-Clipping** | $\mathbf{g} \leftarrow \min(1, \theta / \|\mathbf{g}\|)\mathbf{g}$ | Einheitlicher Skalar für die gesamte Summe $\mathbf{g}$ | Begrenzt Spitzen der Aktualisierungsschritte. Das Verhältnis naher zu weit zurückliegenden Beiträgen bleibt gleich |

<!--en:4adf1b16-->
| **Abgeschnittene BPTT (`detach`)** | Gradientenstopp an den Grenzen von Segmenten der Länge $T_0$ | Begrenzt den Speicherbedarf auf $O(T_0 bh)$ pro Aktualisierung | Beschränkte Reichweite der Zuordnung über Gradienten ($k < T_0$), notwendig bei langen Datenströmen |
