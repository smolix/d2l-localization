<!--en:b7e6a72e-->
# Skalierungsgesetze

<!--en:c4c0d6c2-->
Beim Pre-Training eines Transformer-Sprachmodells (:numref:`sec_gpt`) bestimmen zwei Größen den Rechenaufwand. Die erste ist die **Anzahl $N$ der Parameter ohne Einbettungen** in den $L$ Transformer-Blöcken, die zweite die **Anzahl $D$ der Trainingstoken**, die der Optimierer verarbeitet. Jedes Token erfordert pro Parameter eine vorhersagbare Anzahl von Gleitkommaoperationen (engl. floating-point operations, FLOPs). Bei festem Rechenbudget $C$ (in FLOPs) müssen wir daher zwischen Modellgröße $N$ und Datenbudget $D$ abwägen. Trainieren wir ein größeres Modell über weniger Schritte oder ein kleineres Modell mit mehr Token?

<!--en:71ccda42-->
Empirische **Skalierungsgesetze für neuronale Netze** (engl. neural scaling laws) :cite:`kaplan2020scaling` zeigen, dass der Kreuzentropieverlust auf den Validierungsdaten einer glatten, durch Potenzgesetze beschriebenen Fläche $\mathcal{L}(N, D)$ folgt. Das gilt über mehrere Größenordnungen von $N$ und $D$ hinweg. :citet:`hoffmann2022training` zeigten anschließend, wie man den Verlust bei festem Rechenbudget $C \approx 6 N D$ für das Pre-Training minimiert. Dazu müssen die Anzahl der Modellparameter $N^*(C)$ und die Anzahl der Trainingstoken $D^*(C)$ **im gleichen Verhältnis** wachsen ($N^* \propto C^{1/2}, D^* \propto C^{1/2}$, mit rund $D^* / N^* \approx 20$ Token pro Parameter, siehe :numref:`fig_mdl-transformers-chinchilla`). In diesem Abschnitt leiten wir die Parameteranzahl $N \approx 12 L d^2$ und die FLOP-Regel $C \approx 6 N D$ her und prüfen beide anhand exakter Zählungen auf Operatorebene. Auf WikiText-103 trainieren wir 15 Modelle in einem Versuchsraster mit 5 Modellgrößen und 3 Tokenbudgets. Schließlich erklären wir, warum man unter Berücksichtigung der **Inferenzkosten** (engl. inference cost) bei modernen offenen Modellen kleinere Modelle wählt und sie weit über das Chinchilla-Verhältnis hinaus trainiert.

<!--en:b9a7e99d-->
## Anzahl der Parameter ($N \approx 12 L d^2$) und der FLOPs ($C \approx 6 N D$)

<!--en:73aba09e-->
### Blöcke und Einbettungen

<!--en:3160c592-->
Wir zählen die Parameter in den $L$ Blöcken eines dichten `GPT` der versteckten Breite $d$. Es verwendet herkömmliche Multi-Head-Attention ($H_{\textrm{kv}} = H$) und `SwiGLU` mit der Breite $d_{\textrm{ff}} = \operatorname{round}(\frac{8}{3}d)$ (:numref:`sec_transformer-block`).

<!--en:d6cebebe-->
- Jede Attention-Teilschicht enthält vier Projektionsmatrizen der Form $d \times d$ ($\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$), also **$4 d^2$** Parameter.

<!--en:fdce0f52-->
- Jedes `SwiGLU`-FFN enthält drei Matrizen ($\mathbf{W}_g, \mathbf{W}_1 \in \mathbb{R}^{d \times \frac{8}{3}d}$ und $\mathbf{W}_2 \in \mathbb{R}^{\frac{8}{3}d \times d}$), also **$8 d^2$** Parameter.

<!--en:6a6fc888-->
- Jeder Block enthält somit **$12 d^2$** Matrixparameter (zuzüglich $2d$ Verstärkungsfaktoren für `RMSNorm`). Auf $L$ Blöcke entfallen daher

  $$N \;=\; 12 L d^2$$

  Matrixparameter ohne Einbettungen. Die für Ein- und Ausgabe gemeinsam genutzte Tokeneinbettungstabelle steuert weitere $V d$ Parameter bei.

<!--en:8a9d89b6-->
Wir prüfen, ob die Parameteranzahl ohne Einbettungen mit $12 L d^2$ übereinstimmt. Dabei vernachlässigen wir die $O(Ld)$ Verstärkungsfaktoren und Bias-Terme der Normalisierung. Wir prüfen dies sowohl für unser bytebasiertes `GPT` mit 6 Blöcken als auch für GPT-2 Small mit 12 Blöcken.

<!--en:da12482e-->
### Sechs FLOPs pro Parameter und Token ($C \approx 6 N D$)

<!--en:ab1ec4af-->
Warum kostet es ungefähr $6 N$ FLOPs, ein Trainingstoken mit den $N = 12 L d^2$ Parametern der linearen Projektionen zu verarbeiten?

<!--en:b42a6bac-->
1. **Vorwärtsdurchlauf ($2 N$ FLOPs/Token):** Multiplizieren wir einen Tokenvektor $\mathbf{x} \in \mathbb{R}^{d_{\textrm{in}}}$ mit einer Gewichtsmatrix $\mathbf{W} \in \mathbb{R}^{d_{\textrm{in}} \times d_{\textrm{out}}}$, so benötigen wir $d_{\textrm{in}} d_{\textrm{out}}$ Multiplikationsadditionen $= 2 P$ FLOPs (dabei ist $P = d_{\textrm{in}} d_{\textrm{out}}$ die Parameteranzahl der Matrix). Summieren wir über alle $N$ Matrixparameter und $D$ Token, ergeben sich $2 N D$ FLOPs für den Vorwärtsdurchlauf.

<!--en:3811d602-->
2. **Rückwärtsdurchlauf ($4 N$ FLOPs/Token):** Die Backpropagation durch $\mathbf{Y} = \mathbf{X}\mathbf{W}$ erfordert pro Token *zwei* Matrizenmultiplikationen der Größe $P$. Für $\nabla_{\mathbf{X}}\mathcal{L} = (\nabla_{\mathbf{Y}}\mathcal{L})\mathbf{W}^\top$ fallen $2P$ FLOPs an und für $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^\top(\nabla_{\mathbf{Y}}\mathcal{L})$ weitere $2P$ FLOPs. Damit ergeben sich insgesamt $4 N D$ FLOPs für den Rückwärtsdurchlauf.

<!--en:6753554b-->
Addieren wir den Aufwand für den Vorwärts- und den Rückwärtsdurchlauf, erhalten wir die grundlegende Näherungsformel für den Trainingsrechenaufwand :cite:`kaplan2020scaling`:

<!--en:c4d8ba6a-->
(Berücksichtigen wir auch die Skalarprodukte $Q K^\top$ und $A V$ der Attention für ein Kontextfenster der Länge $T$, kommen $12 L T d D$ FLOPs hinzu. Dadurch wächst $6 N D$ um den Faktor $1 + \frac{T}{6d}$. Dieser liegt nahe bei $1$, wenn $T \ll 6d$ gilt. Überspringt eine Implementierung die maskierte Hälfte der kausalen Attention-Matrix, halbiert sich die Korrektur.)

<!--en:e35a756b-->
### Überprüfung der Rechnung mit Hardwarezählern

<!--en:b0e9cd2e-->
Wir können :eqref:`eq_six_nd` direkt mit der exakten FLOP-Anzahl vergleichen, die das Framework für einen einzelnen Trainingsschritt von `gpt(256, act='gelu')` ermittelt.

<!--en:be97e161-->
In PyTorch zählt `FlopCounterMode` die FLOPs jedes Operators.

<!--en:2f0d4996-->
Die Kostenanalyse von XLA zählt die FLOPs des kompilierten Programms auf der CPU.

<!--en:3a148e0d-->
Der gemessene Rückwärtsdurchlauf erfordert etwa doppelt so viele FLOPs wie der Vorwärtsdurchlauf ($2.0\text{--}2.1\times$). Die Näherung $6 N D$ stimmt bis auf wenige Prozent mit der exakten Zählung auf Operatorebene überein, wenn wir die mit der Einbettung geteilten Gewichte des Ausgabekopfs und den Term $12 L T d D$ für die Attention-Scores berücksichtigen.

<!--en:84b18ab8-->
## Eine kleine Skalierungsstudie auf WikiText-103

<!--en:be99b4fa-->
### Fünf Modellgrößen mit jeweils drei Datenbudgets

<!--en:1a4381eb-->
Um das Skalierungsverhalten selbst zu beobachten, trainieren wir **15 Modelle** auf WikiText-103. Wir kombinieren fünf Architekturen mit $N \in [0.3\textrm{M}, 14\textrm{M}]$ Parametern ohne Einbettungen (`(d, L, H)` von `(96, 3, 4)` bis `(384, 8, 8)`) mit jeweils drei Schrittbudgets von $s \in \{500, 1000, 2000\}$ Schritten (das entspricht $D \in \{4.10\textrm{M}, 8.19\textrm{M}, 16.38\textrm{M}\}$ Trainingsbytes).

<!--en:6f3071c2-->
Nach dem Laden des Korpus trainieren wir alle $5 \times 3 = 15$ Kombinationen `(size, budget)`. Dabei skalieren wir die maximale Lernrate $\eta = 2 \times 10^{-3} \cdot (256 / d)$ mit der Modellbreite. Für jedes Modell erfassen wir die Parameteranzahl ohne Einbettungen in `Ns` und die Kreuzentropie auf den Validierungsdaten in `val`.

<!--en:99d9b157-->
### Verlust in Abhängigkeit von Modellgröße, Datenbudget und Rechenaufwand

<!--en:544bdad0-->
Wir tragen die Kreuzentropie auf den Validierungsdaten gegen die Parameteranzahl $N$ ohne Einbettungen auf. Die Parameterachse ist logarithmisch skaliert, und jedes Tokenbudget $D$ erhält eine eigene Kurve. Anschließend tragen wir dieselben Werte gegen den gesamten Trainingsrechenaufwand $C = 6 N D$ auf.

<!--en:32fec6eb-->
Wir zeichnen dieselben 15 Validierungsverluste erneut, diesmal als Funktion des gesamten Trainingsrechenaufwands $C = 6 N D$. Zusätzlich geben wir die Ergebnisse dreier Läufe mit ähnlichem Rechenaufwand aus ($C \approx 2\text{--}3 \times 10^{14}\textrm{ FLOPs}$, mit höchstens einem Faktor $2\times$ zwischen den Läufen). So sehen wir, wie die rechenoptimale Modellgröße mit dem FLOP-Budget wächst.

<!--en:9c2394cd-->
Zwei Muster fallen auf.

<!--en:800641f1-->
1. Beim größten Tokenbudget ($2{,}000$ Schritte, $D = 16.4\textrm{M}$ Byte) sinkt der Validierungsverlust für die kleinen und mittleren Modelle fast linear mit $\log N$. Beim größten Modell ($N = 14.16\textrm{M}$) flacht die Kurve ab. Selbst $2{,}000$ Schritte ($D/N \approx 1.2$ Byte pro Parameter) liefern für $14\textrm{M}$ Parameter noch zu wenig Daten.

<!--en:0d8dc7f6-->
2. Tragen wir die Ergebnisse aller 15 Läufe gegen die Gesamtzahl der Trainings-FLOPs $C = 6 N D$ auf, ergibt die untere Hüllkurve die **Pareto-Front für den Rechenaufwand**. Bei geringem Rechenaufwand erzielen kleine Modelle den niedrigsten Verlust. Bei höherem Rechenaufwand überholen größere Modelle die kleinen, sobald sie genügend Trainingsschritte erhalten.

<!--en:0caaf108-->
## Rechenoptimale Aufteilung des Budgets (Chinchilla-Skalierung)

<!--en:fa3d0698-->
### Das parametrische Chinchilla-Gesetz

<!--en:dcf8a1fa-->
:citet:`hoffmann2022training` modellieren die Landschaft des Validierungsverlusts wie folgt. Auch die Neuanpassung von :citet:`Besiroglu.Erdil.Barnett.ea.2024` verwendet diesen Ansatz. Der Verlust $\mathcal{L}(N, D)$ ist die Summe aus einer durch die Textentropie gegebenen Untergrenze $L_\infty$ und zwei Zusatztermen, die Potenzgesetzen in $N$ und $D$ folgen.

<!--en:0bd0635c-->
### Die rechenoptimale Aufteilung ($N^*(C)$ und $D^*(C)$)

<!--en:e67a1825-->
Wir minimieren :eqref:`eq_chinchilla_law` bezüglich $N$ und $D$ unter der Nebenbedingung $C = 6 N D$. Dazu setzen wir $D = C / (6N)$ ein und fordern $\partial \mathcal{L} / \partial N = 0$. Damit ergeben sich Potenzgesetze sowohl für die optimale Modellgröße $N^*(C)$ als auch für das optimale Tokenbudget $D^*(C)$:

<!--en:b48111c1-->
Dabei gilt

<!--en:3c95af28-->
![Niveaulinien des Verlusts nach dem Chinchilla-Gesetz in Abhängigkeit von Parameteranzahl und Anzahl der Trainingstoken. Die Konstanten stammen aus der weiter unten vorgestellten Neuanpassung von Besiroglu et al. (2024). Die gestrichelten Linien verbinden Punkte mit gleichem Rechenaufwand $C = 6ND$. Die Punkte markieren das rechenoptimale Modell für jedes Budget, die orange Linie den rechenoptimalen Pfad. Das Quadrat für Chinchilla liegt genau auf dem Optimum seines Budgets und verdeckt den Punkt. Llama 3 8B hat ein ähnliches Budget wie Chinchilla, liegt aber mit einem Zehntel der optimalen Größe weit vom Pfad entfernt.](../img/mdl-transformers-chinchilla.svg)

<!--en:fa5e0893-->
Wegen $\alpha \approx \beta \approx 0.34\text{--}0.37$ gilt für die Exponenten $a \approx b \approx 0.5$. Wächst der Trainingsrechenaufwand $C$ um den Faktor $100\times$, wachsen sowohl die Modellgröße $N^*$ als auch die Anzahl der Trainingstoken $D^*$ um den Faktor $\sim 10\times$. Das rechenoptimale Verhältnis von $D^* / N^* \approx 20$ Token pro Parameter bleibt also annähernd konstant.

<!--en:c3d85ec4-->
Dies war nicht die erste Antwort auf die Frage. Zwei Jahre zuvor hatten :citet:`kaplan2020scaling` Potenzgesetze derselben Art angepasst. Sie folgerten, dass der größte Teil des zusätzlichen Rechenbudgets in das Modell fließen sollte. Für die Modellgröße fanden sie $N^* \propto C^{0.73}$, für die Daten blieb nur $D^* \propto C^{0.27}$ (:numref:`fig_mdl-transformers-kaplan-chinchilla`). Der Unterschied zwischen den Studien liegt vor allem am Lernratenplan. Kaplan et al. verwendeten für alle Läufe denselben Plan. Für kleinere Tokenbudgets lasen sie den Verlust an früheren Checkpoints ab, an denen die Lernrate noch nicht abgeklungen war. Dadurch erscheint das Training mit weniger Token schlechter, als es ist, und der Nutzen zusätzlicher Daten wird unterschätzt. Zudem waren ihre Modelle mit höchstens $1.5\textrm{B}$ Parametern klein. Hoffmann et al. trainierten Modelle mit bis zu $16\textrm{B}$ Parametern und passten den Kosinusplan an die Länge jedes Laufs an. So sind auch wir im Versuchsraster oben vorgegangen. Deshalb hat jeder der 15 Läufe einen eigenen Zeithorizont für den Kosinusplan (in Aufgabe :exref:`ex_scaling-laws-frozen-learning-rate` messen Sie, was andernfalls geschieht).

<!--en:6bd3bd5e-->
![Rechenoptimale Modellgröße in Abhängigkeit vom Trainingsrechenaufwand für die beiden veröffentlichten Exponenten $0.73$ (Kaplan et al.) und $0.5$ (Hoffmann et al.). Beide Geraden verlaufen bei $10^{18}$ FLOPs durch dasselbe Modell. Nur die Steigungen stammen aus den Veröffentlichungen. Ein um den Faktor $100\times$ größeres Budget erfordert beim ersten Exponenten ein um den Faktor $29\times$ größeres Modell, beim zweiten ein um den Faktor $10\times$ größeres. Das Quadrat markiert Chinchilla.](../img/mdl-transformers-kaplan-chinchilla.svg)

<!--en:d1633d65-->
Die folgende Funktion `optimum` wertet :eqref:`eq_chinchilla_optimum` für das Trainingsbudget von Chinchilla aus, also für $C = 5.76 \times 10^{23}$ FLOPs. Wir verwenden einmal die von Hoffmann et al. veröffentlichten Konstanten und einmal die von Besiroglu et al. neu angepassten Konstanten.

<!--en:5168d889-->
### Training über das Chinchilla-Optimum hinaus für die Inferenz

<!--en:e7c562b1-->
Warum trainierte man für Llama 3 8B (:numref:`fig_mdl-transformers-chinchilla`) ein Modell mit 8 Milliarden Parametern auf $15\textrm{T}$ Token ($D / N \approx 1{,}875$ Token pro Parameter, fast $100\times$ so viel wie nach dem Chinchilla-Verhältnis)? Mit demselben Rechenbudget für das Pre-Training hätte man auch ein Chinchilla-optimales Modell mit $70\textrm{B}$ Parametern auf $1.4\textrm{T}$ Token trainieren können, bei einem Aufwand von $C \approx 7 \times 10^{23}$ FLOPs.

<!--en:1fcebce4-->
Weil Chinchilla **allein die FLOPs $6 N D$ des Pre-Trainings** minimiert. Die $2 N D_{\textrm{inf}}$ FLOPs für die Inferenz und der Speicher für den KV-Cache, die beim späteren Betrieb des Modells anfallen, bleiben unberücksichtigt :cite:`Sardana.Portes.Doubov.ea.2024`. Folgen wir einer Niveaulinie konstanten Verlusts $\mathcal{L}(N, D) = \mathcal{L}_{\textrm{target}}$, sehen wir, dass ein mit $15\textrm{T}$ Token weit über das Optimum hinaus trainiertes Modell mit 8 Milliarden Parametern fast denselben Verlust erreicht wie ein nach Chinchilla trainiertes Modell mit $30\textrm{B}\text{--}40\textrm{B}$ Parametern. Dabei sinken die Kosten für jedes später erzeugte Inferenztoken um den Faktor $4\text{--}5\times$.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ac4cdbb6-->
Ein dichter Transformer mit $L$ Schichten der Breite $d$ und SwiGLU ($d_{\textrm{ff}} \approx \frac{8}{3}d$) enthält **$N \approx 12 L d^2$** Matrixparameter ohne Einbettungen ($4 L d^2$ in der Attention und $8 L d^2$ im FFN). Sein Training erfordert **$C \approx 6 N D$** FLOPs für $D$ Token ($2 N D$ im Vorwärts- und $4 N D$ im Rückwärtsdurchlauf, siehe :eqref:`eq_six_nd`). Nach dem **Chinchilla-Skalierungsgesetz** $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta}$ (:eqref:`eq_chinchilla_law`) gilt bei festem Rechenbudget $C = 6 N D$ für das Pre-Training: Der Validierungsverlust ist minimal, wenn die Anzahl der Parameter und die Anzahl der Token mit den Exponenten $a = \beta/(\alpha+\beta) \approx 0.5$ und $b = \alpha/(\alpha+\beta) \approx 0.5$ wachsen (:eqref:`eq_chinchilla_optimum` bis :eqref:`eq_chinchilla_allocation`). Das ergibt rund $D^* / N^* \approx 20$ Token pro Parameter. Berücksichtigen wir den **Rechenaufwand für die Inferenz über die gesamte Lebensdauer** ($2 N D_{\textrm{inf}}$) und die Latenz im Betrieb, verschiebt sich das Optimum für den gesamten Rechenaufwand entlang der Niveaulinie konstanten Verlusts. Es liegt dann bei kleineren Modellen der Größe $N$, die wir mit Hunderten oder Tausenden Token pro Parameter weit über das Chinchilla-Optimum hinaus trainieren.

<!--en:2294243a-->
## Übungen

<!--en:517384ad-->
1. **FLOP-Korrektur für die Attention-Scores ($1 + T / (6d)$).** In :eqref:`eq_six_nd` fehlen die Produkte $\mathbf{Q}\mathbf{K}^\top$ und $\mathbf{A}\mathbf{V}$ für die Attention-Scores und die Gewichtung der Werte sowie der Ausgabekopf $\mathbf{H}^{(L)}\mathbf{E}^\top$.
   1. Betrachten Sie ein Modell mit $L$ Blöcken der Breite $d$ und einem Kontextfenster der Länge $T$. Zeigen Sie, dass die Skalarprodukte der Attention im Vorwärtsdurchlauf $4 L T^2 d$ FLOPs pro Fenster erfordern ($4 L T d$ FLOPs pro Token). Zeigen Sie weiter, dass für Vorwärts- und Rückwärtsdurchlauf zusammen $12 L T d$ FLOPs pro Token anfallen (oder $6 L T d$, wenn FlashAttention die Symmetrie der kausalen Maske ausnutzt).
   2. Bilden Sie das Verhältnis von $6 L T d$ zu $6 N = 72 L d^2$. Zeigen Sie damit, dass die kausale Attention die FLOP-Anzahl relativ um $T / (12 d)$ erhöht (oder um $T / (6 d)$, wenn keine kausal maskierten Kacheln übersprungen werden). Bei welcher Kontextlänge $T^*$ benötigt die Attention so viele FLOPs wie die linearen Projektionen von Llama 3 8B ($d = 4096$)?

<!--en:f84661f4-->
2. **Parameteranzahl mit und ohne Einbettungen im Versuchsraster.** Für alle 5 Modellgrößen in unserer Versuchsreihe auf WikiText-103 verwendeten wir ein Byte-Vokabular ($V = 256$). Dabei beträgt $V d$ höchstens etwa $7\%$ von $12 L d^2$. Führen Sie `census` für dieselben fünf Basisnetze `(d, L, H)` erneut aus, diesmal mit einem BPE-Vokabular von $V = 128{,}256$ Token (Llama 3). Welchen Anteil an allen Parametern hätte die Einbettungstabelle beim kleinsten (`d=96, L=3`) und beim größten (`d=384, L=8`) Modell? Warum passen :citet:`kaplan2020scaling` ihre Potenzgesetze als Funktion von $N_{\textrm{non-emb}} = 12 L d^2$ statt von $N_{\textrm{total}}$ an?
   :exlabel:`ex_scaling-laws-embedding-count`

<!--en:8e91ef28-->
3. **Angepasster oder fester Zeithorizont für den Kosinus-Abfall.** In unserer Skalierungsstudie mit 15 Läufen hatte jedes Budget $s \in \{500, 1000, 2000\}$ einen eigenen Kosinusplan, bei dem die Lernrate in Schritt $s$ auf null sinkt. Den Plan an die Länge des Laufs anzupassen war die entscheidende Korrektur von :citet:`hoffmann2022training` an der Methode von :citet:`kaplan2020scaling`. Wiederholen Sie die Läufe für alle drei Budgets mit dem mittelgroßen Modell `(192, 4, 6)`. Verwenden Sie einen festen Zeithorizont `num_steps=2000` für den Kosinusplan in `LMOptimizer` und lesen Sie den Validierungsverlust bei den Schritten 500, 1000 und 2000 ab. Um wie viel überschätzen Sie den Verlust kurzer Läufe, wenn die Lernrate bei $s = 500$ und $s = 1000$ noch nicht abgeklungen ist?
   :exlabel:`ex_scaling-laws-frozen-learning-rate`

<!--en:a47d04bb-->
4. **Anpassung des Chinchilla-Gesetzes an unser Raster mit 15 Messpunkten.** Verwenden Sie `scipy.optimize.minimize` (oder eine Rastersuche in Verbindung mit der nichtlinearen Methode der kleinsten Quadrate). Passen Sie damit die Parameter $(L_\infty, A, B, \alpha, \beta)$ in :eqref:`eq_chinchilla_law` an die 15 Messpunkte `(N, D, val_loss)` unserer Versuchsreihe auf WikiText-103 an. Welche Skalierungsexponenten $a = \beta/(\alpha+\beta)$ und $b = \alpha/(\alpha+\beta)$ ergeben sich aus den 15 Messpunkten der bytebasierten Modelle?
   :exlabel:`ex_scaling-laws-fit-grid`

<!--en:a20fe21b-->
5. **Minimaler Rechenaufwand über die Lebensdauer bei einem Inferenzbedarf von $D_{\textrm{inf}}$.** Ein Modell der Größe $N$ wird auf $D$ Token trainiert und erzeugt im Betrieb $D_{\textrm{inf}}$ Inferenztoken. Über seine gesamte Lebensdauer benötigt es $C_{\textrm{tot}}(N, D) = 6 N D + 2 N D_{\textrm{inf}}$ FLOPs.
   1. Verwenden Sie Lagrange-Multiplikatoren mit der Nebenbedingung $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta} = \mathcal{L}_{\textrm{target}}$. Zeigen Sie, dass am Minimum $(N^*, D^*)$ des Rechenaufwands über die Lebensdauer für das Verhältnis der marginalen Beiträge gilt:

      $$\frac{\beta B (D^*)^{-\beta}}{\alpha A (N^*)^{-\alpha}} \;=\; \frac{6 N^* D^*}{6 N^* D^* + 2 N^* D_{\textrm{inf}}} \;=\; \frac{1}{1 + \frac{D_{\textrm{inf}}}{3 D^*}}.$$

   2. Prüfen Sie, dass sich für $D_{\textrm{inf}} = 0$ wieder die Chinchilla-Bedingung $\alpha A (N^*)^{-\alpha} = \beta B (D^*)^{-\beta}$ ergibt. Bestimmen Sie numerisch $(N^*, D^*)$ für den Zielverlust $\mathcal{L}_{\textrm{target}} = \mathcal{L}(70\textrm{B}, 1.4\textrm{T})$ und einen Inferenzbedarf von $D_{\textrm{inf}} \in \{10\textrm{T}, 100\textrm{T}, 500\textrm{T}\}$ Token.
   :exlabel:`ex_scaling-laws-training-for-inference`

<!--en:ad025382-->
6. **Aktivierungs-Checkpointing und die Regel für $8 N D$.** Vollständiges Aktivierungs-Checkpointing auf Blockebene speichert im Vorwärtsdurchlauf nur die Eingabe $\mathbf{X}^{(\ell-1)}$ jedes Blocks ($2 N D$ FLOPs für den Vorwärtsdurchlauf). Bei der Backpropagation wird der Vorwärtsdurchlauf jedes Blocks einmal wiederholt, bevor die Matrixprodukte des Rückwärtsdurchlaufs berechnet werden ($4 N D$ FLOPs). Zeigen Sie, dass sich die Formel für den Rechenaufwand des Pre-Trainings durch Aktivierungs-Checkpointing von $C \approx 6 N D$ auf $C \approx 8 N D$ ändert. Bestimmen Sie, wie sich der Ersatz von $6$ durch $8$ in :eqref:`eq_chinchilla_optimum` bei der Skalierung von $N^*(C)$ und $D^*(C)$ für ein festes FLOP-Budget $C$ auswirkt.

<!--en:128c43b6-->
[Diskussionen](https://d2l.discourse.group/t/36780)

<!--en:e771fb7a-->
Skalierungsgesetze für neuronale Netze und rechenoptimales Training (Chinchilla)<br>**N ≈ 12Ld² und C ≈ 6ND · Versuchsraster mit 15 Modellen · IsoFLOP-Linien · Chinchilla-Optimum · längeres Training für die Inferenz**

<!--en:01b31777-->
::: {.slide title="Warum ein Lauf mit 10²⁵ FLOPs im Voraus geplant werden muss"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:57bce442-->
- Das Pre-Training eines Sprachmodells an der Spitze der Entwicklung ist eine einmalige Investition, die sich nicht wiederholen lässt. Sie bindet Tausende GPUs über Monate ($3.8 \times 10^{25}\text{ FLOPs}$ für Llama 3 405B).

<!--en:c7f196dd-->
- **Wie teilen wir das Budget auf?** Bei festem Rechenbudget $C \approx 6 N D$ für das Training stellt sich die Frage, wie wir $C$ auf Modellparameter $N$ und Trainingstoken $D$ verteilen, um den Validierungsverlust $\mathcal{L}(N, D)$ zu minimieren.

<!--en:6e893d87-->
::: {.slide title="Parameteranzahl in den Blöcken (12Ld²) und Einbettungen (Vd)"}

<!--en:90b25bbf-->
[Parameteranzahl]{.kicker}

<!--en:4fe3c052-->
In einem dichten Transformer mit $L$ Blöcken der Breite $d$, $H_{\text{kv}} = H$ und `SwiGLU` ($d_{\text{ff}} \approx \frac{8}{3}d$) enthält jeder Block $4d^2$ Attention-Gewichte und $8d^2$ FFN-Gewichte.

<!--en:c85e2d1a-->
- **Kleines Vokabular ($V = 256$ Byte):** Die Einbettungstabelle enthält nur $1.4\%$ der insgesamt $4.79\text{M}$ Parameter unseres Modells ($N_{\text{blocks}} = 12.01\,L d^2$).

<!--en:d5514c9c-->
- **Großes BPE-Vokabular ($V = 50{,}257$ bei GPT-2 Small):** Einbettungs- und Positionstabelle enthalten zusammen $39.4\text{M}$ ($31.6\%$) der $124.4\text{M}$ Parameter. Auf die 12 Blöcke entfallen $85.0\text{M}$ ($12.02\,L d^2$).

<!--en:fd08bfb3-->
::: {.slide title="Parameter im Code zählen"}

<!--en:90b25bbf-2-->
[Parameteranzahl]{.kicker}

<!--en:445b2c32-->
::: {.slide title="Warum das Training sechs FLOPs pro Parameter und Token kostet"}

<!--en:65fdc80c-->
[Die Regel C ≈ 6ND]{.kicker}

<!--en:4998f802-->
- Für jede lineare Projektion $\mathbf{Y} = \mathbf{X}\mathbf{W}$ mit $P = d_{\text{in}} d_{\text{out}}$ Parametern über $D$ Token gilt insgesamt:
  $$C \approx \underbrace{2 N D}_{\text{forward}} + \underbrace{4 N D}_{\text{backward}} = 6 N D$$

<!--en:3f5f8bbe-->
- **Vorwärtsdurchlauf ($2P$ FLOPs/Token):** $\mathbf{Y} = \mathbf{X}\mathbf{W}$.

<!--en:d3bf5a09-->
- **Rückwärtsdurchlauf ($4P$ FLOPs/Token):** Aktivierungsgradient $\nabla_{\mathbf{X}}\mathcal{L} = (\nabla_{\mathbf{Y}}\mathcal{L})\mathbf{W}^\top$ ($2P$) + Gewichtsgradient $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^\top(\nabla_{\mathbf{Y}}\mathcal{L})$ ($2P$).

<!--en:80ec9019-->
::: {.slide title="Skalarprodukte der Attention in der FLOP-Bilanz"}

<!--en:547e34fe-->
[FLOP-Term der Attention]{.kicker}

<!--en:b0e8eced-->
- In jedem Block berechnet die Attention zusätzlich $\mathbf{Q}\mathbf{K}^\top$ und $\mathbf{A}\mathbf{V}$ ($4Td$ FLOPs/Token im Vorwärtsdurchlauf gegenüber $24d^2$ FLOPs/Token für lineare Projektionen). Die relative Korrektur beträgt $T / (6d)$ (oder $T / (12d)$, wenn die kausale Maske das Überspringen ganzer Kacheln erlaubt).

<!--en:8416a8ec-->
- **Kurzer Kontext ($T \ll 6d$):** Nur $\approx +8\%$ bei unserem bytebasierten GPT ($T = 128, d = 256$).

<!--en:5e7f7910-->
- **Schwelle bei langem Kontext ($T \ge 6d$):** Bei $T = 6d = 24{,}576$ Token benötigt die Attention in Llama 3 8B ($d = 4{,}096$) so viele FLOPs wie die Projektionen.

<!--en:04d9fdad-->
::: {.slide title="C ≈ 6ND mit exakten Hardwarezählern überprüfen" only="pytorch"}

<!--en:f795cc80-->
[Empirische FLOP-Zählung]{.kicker}

<!--en:6eebfcbf-->
Wir messen einen vollständigen Vorwärts- und Rückwärtsdurchlauf von `gpt(256, act='gelu')` für einen Batch der Form $(64, 128)$ ($D_{\text{step}} = 8{,}192$ Token). Die Messung mit `FlopCounterMode` bestätigt die Herleitung.

<!--en:447e8b6d-->
Die exakte FLOP-Anzahl für die Matrizenmultiplikationen beträgt **$1.082 \times 6 N D$**. Das stimmt auf drei Nachkommastellen mit $1 + \frac{4 L T d}{2 N} = 1.082$ überein. Das Verhältnis von Rückwärts- zu Vorwärtsdurchlauf beträgt **$\mathbf{2.00}$** ($4 N D$ gegenüber $2 N D$).

<!--en:1bfbea00-->
::: {.slide title="C ≈ 6ND mit exakten Hardwarezählern überprüfen" only="jax"}

<!--en:f795cc80-2-->
[Empirische FLOP-Zählung]{.kicker}

<!--en:83d14f1a-->
Wir messen einen vollständigen Vorwärts- und Rückwärtsdurchlauf von `gpt(256, act='gelu')` für einen Batch der Form $(64, 128)$ ($D_{\text{step}} = 8{,}192$ Token). Die Messung mit der HLO-Methode `cost_analysis()` für den kompilierten Schritt bestätigt die Herleitung.

<!--en:518ebe35-->
Die HLO-Kostenanalyse ergibt **$1.116 \times 6 N D$** FLOPs und ein Verhältnis von Rückwärts- zu Vorwärtsdurchlauf von **$2.06$**. Beides liegt wenige Prozent über der Vorhersage für die Matrizenmultiplikationen $1 + \frac{4 L T d}{2 N} = 1.082$ und dem Verhältnis $2$ ($4 N D$ gegenüber $2 N D$).

<!--en:25750c16-->
::: {.slide title="Empirische Skalierungsstudie mit 15 Modellen auf WikiText-103"}

<!--en:e531ac0c-->
[Aufbau des Versuchsrasters]{.kicker}

<!--en:43d4b08f-->
Um die Skalierungsgesetze für neuronale Netze selbst zu beobachten, trainieren wir in einem $5 \times 3$-Raster **15 bytebasierte GPT-Modelle**. Zwischen kleinster und größter Parameteranzahl ohne Einbettungen liegt der Faktor $43\times$. Jede Parameteranzahl $N$ kombinieren wir mit drei Datenbudgets $D$.

<!--en:7e8a068b-->
| Versuchsgröße | Werte im Raster mit 15 Läufen | Umfang und Zweck |

<!--en:bdfc8c0d-->
| **5 Größen des Basisnetzes `(d, L, H)`** | `(96, 3, 4)`, `(128, 3, 4)`, `(192, 4, 6)`, `(256, 6, 8)`, `(384, 8, 8)` | $N \in [0.33\text{M},\; 14.16\text{M}]$ Parameter ohne Einbettungen (Faktor $43\times$) |

<!--en:b44b3ffd-->
| **3 Schrittbudgets ($s$ Schritte)** | $500$, $1{,}000$, $2{,}000$ Schritte ($8{,}192$ Byte/Schritt) | $D \in \{4.1\text{M},\; 8.2\text{M},\; 16.4\text{M}\}$ Byte ($\le 6\%$ des Korpus mit $270\text{M}$ Byte) |

<!--en:0844045d-->
| **Breitenabhängige maximale Lernrate** | $\eta_{\max}(d) = 2 \times 10^{-3} \cdot (256 / d)$ | Die optimale Lernrate für AdamW ist umgekehrt proportional zur Breite $d$ |

<!--en:d94ec85a-->
| **Angepasster Zeithorizont des Kosinusplans** | Aufwärmphase über $\max(100, s/10)$ Schritte, dann Kosinus-Abfall auf $0$ bei Schritt $s$ | Wir werten jeden Lauf am Ende seines eigenen Lernratenplans aus, wenn die Lernrate abgeklungen ist |

<!--en:c95a1b47-->
::: {.slide title="Feste Zeithorizonte verfälschen die Skalierungsexponenten"}

<!--en:80b628f4-->
[Fehler im Versuchsaufbau]{.kicker}

<!--en:2db1feaa-->
- **Korrektur von Kaplan et al. (2020) durch Hoffmann et al. (2022):** Bei einem einheitlichen Kosinusplan über $2{,}000$ Schritte liegt die Lernrate in Schritt $500$ noch nahe am Maximum ($\eta_{500} \approx 0.9\,\eta_{\max}$).

<!--en:46964393-->
- **Systematische Verzerrung bei kleinen Budgets:** Lesen wir den Verlust an Checkpoints ab, bevor die Lernrate abgeklungen ist, überschätzen wir den Verlust kurzer Läufe. Zusätzliche Token $D$ erscheinen so weniger wertvoll als zusätzliche Parameter $N$.

<!--en:9b405c36-->
::: {.slide title="Abnehmender Nutzen, wenn nur N wächst"}

<!--en:3819a9f6-->
[Empirische Skalierungsfläche]{.kicker}

<!--en:96a546c8-->
- **Größeres $N$ bei fester Tokenanzahl $D$:** Eine Verdreifachung der Modellgröße ($0.59\text{M} \to 1.77\text{M}$) senkt den Verlust um $0.05\text{--}0.09\text{ nats}$. Eine weitere Verdreifachung ($4.72\text{M} \to 14.16\text{M}$) bringt nur $0.02\text{--}0.04\text{ nats}$, weil $500\text{--}2{,}000$ Schritte für das Modell mit $14.16\text{M}$ Parametern zu wenig Daten liefern.

<!--en:c6ff6572-->
- **Größeres $D$ bei fester Modellgröße $N$:** Vervierfachen wir die Tokenanzahl ($4.1\text{M} \to 16.4\text{M}$), sinkt der Validierungsverlust bei jeder Modellgröße um **$0.20\text{--}0.27\text{ nats}$**.

<!--en:dc0bfcc0-->
::: {.slide title="IsoFLOP-Vergleich: Länger trainierte kleinere Modelle schneiden besser ab"}

<!--en:cc89a756-->
[Pareto-Front für den Rechenaufwand]{.kicker}

<!--en:4a2a1b2c-->
- **Alle 15 Läufe in Abhängigkeit von der Gesamtzahl der FLOPs $C = 6 N D$:** Wo sich die Kurven verschiedener Modellgrößen im Rechenaufwand $C$ überlappen, erzielt das **mit mehr Token trainierte kleinere Modell** durchweg einen niedrigeren Validierungsverlust.

<!--en:e41fb840-->
- **IsoFLOP-Vergleich im Einzelnen:**
  - $N = 14.16\text{M}$ mit $D = 4.1\text{M}$ ($C = 3.5 \times 10^{14}$): **$1.42\text{ nats}$**
  - $N = 4.72\text{M}$ mit $D = 8.2\text{M}$ ($C = 2.3 \times 10^{14}$): **$1.34\text{ nats}$**
  - $N = 1.77\text{M}$ mit $D = 16.4\text{M}$ ($C = 1.7 \times 10^{14}$): **$1.29\text{ nats}$** (niedrigster Verlust bei der *Hälfte* der FLOPs).

<!--en:aa7ee6b2-->
::: {.slide title="Das parametrische Chinchilla-Skalierungsgesetz"}

<!--en:ad7f80de-->
[Parametrische Verlustlandschaft]{.kicker}

<!--en:87174072-->
- Hoffmann et al. (2022) zerlegen den Validierungsverlust $\mathcal{L}(N, D)$ in drei additive Terme:
  $$\mathcal{L}(N, D) = L_\infty + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

<!--en:d00ebf57-->
- **Entropieuntergrenze ($L_\infty \approx 1.7\text{--}1.8$):** Irreduzible Entropie natürlicher Sprache.

<!--en:c0b133b6-->
- **Modellkapazität ($A / N^\alpha$) und Datenstichprobe ($B / D^\beta$):** Zusatzterme, die Potenzgesetzen mit $\alpha \approx 0.35, \beta \approx 0.37$ folgen.

<!--en:65d4d0d7-->
::: {.slide title="Herleitung der rechenoptimalen Aufteilung N*(C) und D*(C)"}

<!--en:96ee8bde-->
[Beschränkte Optimierung]{.kicker}

<!--en:6a370cef-->
Um $\mathcal{L}(N, D)$ bei festem Rechenbudget $C = 6 N D$ zu minimieren, setzen wir $D = C / (6N)$ ein. In $\mathcal{L}(N, C/(6N))$ setzen wir dann die Ableitung nach $N$ gleich null.

<!--en:73fcc85d-->
- **Gleicher Grenznutzen ($\alpha A (N^*)^{-\alpha} = \beta B (D^*)^{-\beta}$):** Am Rechenoptimum schrumpfen die Zusatzterme für begrenzte Modellgröße und begrenzte Datenmenge mit zueinander proportionalen Raten. Es gilt
  $$N^*(C) = G\left(\frac{C}{6}\right)^a, \qquad D^*(C) = G^{-1}\left(\frac{C}{6}\right)^b$$
  mit $a = \frac{\beta}{\alpha + \beta}$, $b = \frac{\alpha}{\alpha + \beta}$ und $G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}$.

<!--en:293c5ae9-->
- **Gleiche Skalierung ($a \approx b \approx 0.5$):** Wegen $\alpha \approx \beta$ gilt: Steigt der Rechenaufwand $C$ um den Faktor $100\times$, steigen **sowohl $N^*$ als auch $D^*$ um den Faktor $\sim 10\times$**.

<!--en:837229fa-->
::: {.slide title="IsoFLOP-Linien und rechenoptimaler Pfad"}

<!--en:9914b045-->
[Geometrische Deutung]{.kicker}

<!--en:8e4846b4-->
- **IsoFLOP-Budgetgeraden (gestrichelt, Steigung $-1$ bei doppelt logarithmischer Darstellung):** Geraden mit konstantem Trainingsrechenaufwand $C = 6 N D$.

<!--en:8c24a580-->
- **Rechenoptimaler Pfad (orange Linie, Steigung $\approx +1$):** Er verbindet die Berührpunkte $(N^*(C), D^*(C))$, an denen jede Budgetgerade die Niveaulinie des niedrigsten erreichbaren Verlusts berührt.

<!--en:a3700fa1-->
- **Die Regel der 20 Token pro Parameter:** Entlang des orangefarbenen Pfads gilt ein Verhältnis von $D^* / N^* \approx 20$ Token pro Parameter (`Chinchilla` 70B mit $1.4\text{T}$ Token).

<!--en:bc13df1d-->
::: {.slide title="IsoFLOP-Linien und rechenoptimaler Pfad (Fortsetzung)"}

<!--en:9914b045-2-->
[Geometrische Deutung]{.kicker}

<!--en:6c12c439-->
Wir berechnen das Optimum $N^*(C), D^*(C)$ in geschlossener Form für zwei veröffentlichte Parameterschätzungen beim Budget von Chinchilla von $C = 5.76 \times 10^{23}$ FLOPs.

<!--en:9569ad11-->
::: {.slide title="Wie die Ergebnisse von Kaplan (2020) und Chinchilla (2022) zusammenpassen"}

<!--en:13b166b0-->
[Warum sich die Exponenten geändert haben]{.kicker}

<!--en:f3ded24d-->
- **Kaplan et al. (2020):** Sie erhielten $N^* \propto C^{0.73}, D^* \propto C^{0.27}$, indem sie den Verlust vor dem Abklingen der Lernrate an Zwischenständen des Trainings ablasen und die Einbettungsparameter $V d$ ausklammerten.

<!--en:019278e0-->
- **Chinchilla (2022) und Replikationsstudien:** Passt man die Dauer des Kosinus-Abfalls an jedes Budget $s$ an und berücksichtigt alle Parameter und FLOPs, ergibt sich $N^* \propto C^{0.50}, D^* \propto C^{0.50}$ ($D^* / N^* \approx 20$).

<!--en:d0c703e0-->
::: {.slide title="Warum Modelle für den Einsatz weit über das Chinchilla-Verhältnis hinaus trainiert werden"}

<!--en:acef1204-->
[Skalierung unter Berücksichtigung der Inferenz]{.kicker}

<!--en:b34a3f48-->
Betrachten wir, wo neuere Modelle mit offenen Gewichten im Verhältnis zu ihrem Chinchilla-Optimum für das Pre-Training liegen.

<!--en:4f730c05-->
- **Chinchilla optimiert allein die Kosten des Pre-Trainings ($6 N D$):** Für die Inferenz-FLOPs ($2 N$ pro erzeugtem Token) und den VRAM-Bedarf im Betrieb nach der Veröffentlichung setzt Chinchilla **Kosten von null** an.

<!--en:f49b6d65-->
- **Flache Verlustlandschaft nahe dem Optimum:** Trainiert man `Llama 3 8B` auf $15\text{T}$ Token ($1{,}875\text{ tokens/param}$, nur $0.10\times$ der Chinchilla-optimalen Größe für dieses Budget), steigt der Verlust um lediglich $+0.054\text{ nats}$ gegenüber einem Chinchilla-optimalen Modell mit $\sim 80\text{B}$ Parametern. Zugleich sinken **die Kosten für jedes erzeugte Inferenztoken um den Faktor $\sim 10\times$**.

<!--en:8f110e0c-->
::: {.slide title="Rechenaufwand über die Lebensdauer für Pre-Training und Betrieb"}

<!--en:ffcf5006-->
[Gesamtkosten über die Lebensdauer]{.kicker}

<!--en:dc1a06a2-->
- Für $D$ Trainingstoken und $D_{\text{inf}}$ erzeugte Inferenztoken beträgt der gesamte Rechenaufwand über die Lebensdauer
  $$C_{\mathrm{lifetime}}(N, D) = 6 N D + 2 N D_{\mathrm{inf}}$$

<!--en:f82da7cc-->
- **Wann sich Llama 3 8B ($D = 15\text{T}$) gegenüber einem Chinchilla-Modell mit 34 Mrd. Parametern und gleichem Verlust rechnet:** Die zusätzlichen $5.9 \times 10^{23}$ FLOPs beim Training sparen $5.2 \times 10^{10}\text{ FLOPs}$ pro erzeugtem Token. Sie amortisieren sich nach **$D_{\text{inf}} \approx 11.3\text{T}$ Inferenztoken**.

<!--en:3fd13d7a-->
::: {.slide title="Verständnisfragen zu FLOP-Bilanz und Budgetaufteilung"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:025f4585-->
Prüfen Sie Ihr Verständnis der Skalierungsgesetze anhand von drei Rechnungen.

<!--en:684dcdda-->
1. **Rechenaufwand und Attention-Korrektur für GPT-2 Small:** GPT-2 Small hat insgesamt $N = 124\text{M}$ Parameter ($d = 768$) und wurde auf rund $D = 10\text{B}$ Token bei einer Kontextlänge von $T = 1{,}024$ trainiert. Schätzen Sie die $6ND$ FLOPs des Trainings und die relative Korrektur $T / (6d)$ für die Attention-Scores.

<!--en:26099519-->
2. **100-faches Rechenbudget:** Ein Labor erhöht sein Rechenbudget $C$ für das Pre-Training um den Faktor $100\times$ (von $10^{23}$ auf $10^{25}\text{ FLOPs}$). Um welche Faktoren sollte es die Anzahl der Modellparameter $N^*$ und der Trainingstoken $D^*$ erhöhen, um auf dem rechenoptimalen Chinchilla-Pfad zu bleiben ($a \approx b \approx 0.5$)?

<!--en:86abd34e-->
3. **Aktivierungs-Checkpointing ($8ND$):** Beim vollständigen Aktivierungs-Checkpointing auf Blockebene wird der Vorwärtsdurchlauf ($2ND$) während der Backpropagation ($4ND$) einmal neu berechnet. Wie ändert sich dadurch die Formel für die Trainingskosten pro Token?

<!--en:557b98b6-->
**Antworten:**

<!--en:8f73aac5-->
1. **$C \approx 6 \times (1.24 \times 10^8) \times 10^{10} \approx \mathbf{7.4 \times 10^{18}\text{ FLOPs}}$**. Der Attention-Term erhöht den Aufwand gegenüber den linearen Projektionen um $T / (6d) = 1024 / (6 \times 768) \approx \mathbf{22.2\%}$.

<!--en:59176b1f-->
2. Sowohl $N^*$ als auch $D^*$ wachsen um den Faktor **$100^{0.5} = \mathbf{10\times}$**. Das Verhältnis von $D^* / N^* \approx 20$ Token pro Parameter bleibt damit unverändert.

<!--en:055a1c49-->
3. Der gesamte Trainingsrechenaufwand steigt um $33\%$ von $C \approx 6 N D$ auf **$C \approx (2 + 2 + 4) N D = \mathbf{8 N D}$**. Dafür genügt ein Aktivierungsspeicher von $O(1)$ pro Schicht.

<!--en:e3222d32-->
::: {.slide title="Zusammenfassung der Skalierungsgesetze für Parameter, FLOPs und Chinchilla"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:8702d295-->
| Gesetz oder Größe | Formel | Praktische Bedeutung und Faustregel |

<!--en:e6dac5e6-->
| **Parameter ohne Einbettungen** | $N \approx 12 L d^2$ ($4 L d^2$ für Attention $+ 8 L d^2$ für das FFN) | Bei dichten Modellen auf $<1\%$ genau. Für gemeinsam genutzte Einbettungen kommen $V d$ hinzu |

<!--en:a9c049c2-->
| **Trainings-FLOPs** | $C \approx 6 N D \cdot \bigl(1 + \frac{T}{6d}\bigr)$ | $2ND$ im Vorwärtsdurchlauf $+ 4ND$ im Rückwärtsdurchlauf. Attention dominiert erst für $T \gtrsim 6d$ |

<!--en:9d4de3d2-->
| **Chinchilla-Verlustlandschaft** | $\mathcal{L}(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta}$ | Summe aus Entropieuntergrenze, Fehler durch begrenzte Modellkapazität und Fehler durch begrenzte Datenmenge |

<!--en:02db57ea-->
| **Rechenoptimale Aufteilung** | $N^*(C) \propto C^{\frac{\beta}{\alpha+\beta}} \approx C^{0.5},\; D^*(C) \propto C^{0.5}$ | $N$ und $D$ wachsen gleichermaßen mit $\sqrt{C}$. Das Verhältnis $D^* / N^* \approx 20\text{ tokens/param}$ bleibt erhalten |

<!--en:93494dbf-->
| **Verschiebung des Optimums durch die Inferenz** | Minimiere $6 N D + 2 N D_{\text{inf}}$ entlang von $\mathcal{L}(N,D)=\mathcal{L}_0$ | Kleinere Modelle mit $200\text{--}2{,}000\text{ tokens/param}$ trainieren, um die Betriebskosten stark zu senken |
