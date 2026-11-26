<!--en:f389f79e-->
# Der Transformer-Block

<!--en:243267c5-->
Self-Attention (:numref:`chap_attention`) mischt Informationen zwischen den Positionen einer Sequenz. Die Ausgabe für jedes Token ist ein gewichteter Mittelwert der Wertvektoren an den Positionen, auf die das Token zugreift. Vor und nach der gewichteten Summe nutzt eine Attention-Schicht für sich genommen jedoch nur lineare Projektionen ($W_Q, W_K, W_V, W_O$). Ohne positionsweise Nichtlinearität, Residualverbindungen (engl. residual connections) und Normalisierung können gestapelte Attention-Schichten weder hierarchische Merkmale für einzelne Token bilden noch Signale zuverlässig über Dutzende von Schichten weitergeben.

<!--en:dd0fd990-->
Der **Transformer-Block** :cite:`Vaswani.Shazeer.Parmar.ea.2017` verbindet Multi-Head-Attention mit einem positionsweisen **Feedforward-Netz (FFN)** (engl. feed-forward network). Er umgibt jede Teilschicht mit einer additiven **Residualverbindung** :cite:`He.Zhang.Ren.ea.2016` und stabilisiert die Aktivierungen durch **Schichtnormalisierung** (engl. layer normalization, :numref:`fig_transformer-block`). Moderne Blöcke unterscheiden sich in drei Punkten vom ursprünglichen Transformer aus dem Jahr 2017:

<!--en:b735c7dc-->
1. **Position der Normalisierung:** Verlagert man die Normalisierung vom Residualstrom (engl. residual stream) selbst (**Post-Norm**) an den Eingang jeder Teilschicht (**Pre-Norm**), bleibt über alle Schichten ein Identitätspfad ohne Abschwächung erhalten.

<!--en:e85f95c2-->
2. **Form der Normalisierung:** An die Stelle von `LayerNorm` mit Mittelwertzentrierung tritt die Normalisierung mit dem quadratischen Mittel (**`RMSNorm`**), optional ergänzt um eine **Normalisierung von Abfragen und Schlüsseln (`QK-Norm`)** je Kopf. Das begrenzt Aktivierungen und Attention-Logits und beansprucht weniger zusätzliche Speicherbandbreite.

<!--en:9135e9ae-->
3. **Feedforward-Netze mit Gate:** An die Stelle des ReLU/GELU-MLP mit zwei Matrizen ($d_{\textrm{ff}} = 4d$) tritt ein **`SwiGLU`**-Netz mit Gate ($d_{\textrm{ff}} \approx \frac{8}{3}d$). Es liefert bessere Repräsentationen bei exakt demselben Parameter- und FLOP-Budget von $8d^2$.

<!--en:36d515b2-->
## Komponenten eines Transformer-Blocks

<!--en:4f960f94-->
Ein Transformer-Block arbeitet mit einem gemeinsamen **Residualstrom** $\mathbf{X} \in \mathbb{R}^{B \times n \times d}$ (:numref:`sec_what-attention-computes`). Zwei aufeinanderfolgende Teilschichten lesen aus dem Strom. Statt $\mathbf{X}$ zu überschreiben, berechnet jede Teilschicht eine Aktualisierung und addiert sie zum Strom:

<!--en:dcd98a4b-->
- **Multi-Head-Attention (Mischen zwischen Token)** ist die einzige Stelle, an der die Sequenzpositionen miteinander wechselwirken. Jedes Token fragt die Sequenz ab (:numref:`sec_multihead-attention`) und addiert den abgerufenen Kontextvektor zu seinem Anteil am Strom.

<!--en:76a8fffb-->
- Das **positionsweise Feedforward-Netz (Mischen zwischen Kanälen)** wendet dasselbe MLP unabhängig auf jeden Tokenvektor $\mathbf{x}_i \in \mathbb{R}^d$ an. So transformiert es die Merkmale, die die Attention zusammengetragen hat. Rund zwei Drittel der Parameter eines dichten Blocks entfallen auf das FFN.

<!--en:dcf28fc1-->
Moderne Transformer verwenden **Pre-Norm** (:numref:`fig_transformer-block`). Dabei normalisiert jede Teilschicht ihre eigene Eingabe, bevor sie ihre Aktualisierung berechnet:

<!--en:dc8e315a-->
![Der Transformer-Block. Attention und positionsweises FFN lesen aus dem Residualstrom und addieren ihre Ergebnisse zum Strom. Links die Anordnung von 2017. Die Normalisierung wirkt nach jeder Addition auf den Strom selbst. Rechts die heutige Anordnung. Die Normalisierung liegt in den einzelnen Zweigen, und der Strom verläuft ohne Unterbrechung von der Eingabe zur Ausgabe.](../img/mdl-transformers-block-anatomy.svg)

<!--en:3e8d0762-->
## Normalisierung

<!--en:b8348ffb-->
### Pre-Norm und Post-Norm im Vergleich

<!--en:4e2ee620-->
Der ursprüngliche Transformer von 2017 :cite:`Vaswani.Shazeer.Parmar.ea.2017` und BERT :cite:`Devlin.Chang.Lee.ea.2018` verwendeten **Post-Norm**. Nach jeder Addition normalisierten sie *den Residualstrom selbst*:

<!--en:a76eb4ef-->
Warum sind moderne Architekturen von :eqref:`eq_postln-block` zu :eqref:`eq_preln-block` übergegangen? Den Unterschied sehen wir schon bei der Initialisierung, ohne einen einzigen Trainingsschritt. Dazu verwenden wir einen minimalen Block `MiniBlock`, dessen Schalter `pre_norm` zwischen den beiden Anordnungen wechselt:

<!--en:bc2135a5-->
### Signalausbreitung und Kollaps der Tokenrepräsentationen bei der Initialisierung

<!--en:11e790d5-->
Wir leiten einen Batch zufälliger Tokenvektoren $\mathbf{X}^{(0)} \in \mathbb{R}^{B \times n \times d}$ mit Varianz eins durch $L = 32$ frisch initialisierte Blöcke. Die folgende Funktion `signal_stats` zieht jede Gewichtsmatrix des Stapels neu. Dazu nutzt sie die Xavier-Initialisierung wie im Transformer von 2017 oder zieht die Einträge aus $\mathcal{N}(0, \textrm{std}^2)$, wenn `std` angegeben ist. An jedem Block $\ell \in \{1, \dots, L\}$ verfolgen wir drei Größen:

<!--en:bdc8c7f7-->
1. die **Standardabweichung des Residualstroms** über alle Einträge von $\mathbf{X}^{(\ell)}$,

<!--en:57f46e3e-->
2. die **Tokenstreuung** (engl. token spread), gemessen als Standardabweichung über die $n$ Sequenzpositionen (sie verschwindet, wenn alle $n$ Token auf denselben Vektor zusammenfallen), und

<!--en:50534d1d-->
3. die **Gradientennorm der Abfrageprojektion** $\|\partial \mathcal{L} / \partial W_q^{(\ell)}\|_F$ bei der Rückpropagation ausgehend von einem skalaren Verlust mit Einheitsgradienten am Ende des Stapels:

<!--en:c29f371c-->
Warum nimmt bei Post-Norm die Vielfalt der Token mit der Tiefe ab? Wir zerlegen jedes Token in einen allen Token gemeinsamen und einen tokenspezifischen Anteil, $\mathbf{x}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$. Dabei ist $\boldsymbol{\mu} = \frac{1}{n}\sum_j \mathbf{x}_j$ der Mittelwert über die $n$ Positionen (:numref:`fig_mdl-transformers-token-collapse`). Bei der Initialisierung sind die Attention-Gewichte fast gleich, sodass jedes Token nahezu dieselbe Aktualisierung $\mathbf{u} \approx \mathbf{W}_o\mathbf{W}_v\boldsymbol{\mu}$ erhält. Die Abweichungen mitteln sich heraus, und nur der Mittelwert wird weitergegeben. Jede der beiden Projektionen multipliziert die quadrierte Länge ihrer Eingabe mit $g^2 = d\sigma^2$. Dabei ist $\sigma^2$ die Varianz eines Gewichtseintrags, und der Zufallsvektor $\mathbf{u}$ ist nahezu orthogonal zu $\boldsymbol{\mu}$. Die Energie des gemeinsamen Anteils wächst also um den Faktor $1 + g^4$, während die Abweichungen gleich bleiben. Die anschließende Normalisierung kann das nicht ausgleichen. Sie skaliert jedes Token mit demselben Faktor. Damit stellt sie die Länge des Stroms wieder her, aber nicht das Verhältnis zwischen Mittelwert und Abweichungen.

<!--en:9a6bb650-->
![Kollaps der Tokenrepräsentationen in einem Post-Norm-Block. Jedes Token setzt sich aus dem gemeinsamen Mittelwert $\boldsymbol{\mu}$ und seiner eigenen Abweichung $\boldsymbol{\delta}_i$ zusammen. Bei nahezu gleichmäßigen Attention-Gewichten erhält jedes Token dieselbe Aktualisierung $\mathbf{u} \approx \mathbf{W}_o\mathbf{W}_v\boldsymbol{\mu}$, sodass nur der Mittelwert wächst. Die anschließende Normalisierung skaliert alle Token mit demselben Faktor und verkleinert dadurch die Abweichungen. Die Abbildung zeigt die Xavier-Initialisierung ($g = 1$). Hier ist $\mathbf{u}$ so lang wie $\boldsymbol{\mu}$, und der Faktor beträgt $1/\sqrt{2}$.](../img/mdl-transformers-token-collapse.svg)

<!--en:1e4e7366-->
Sei $\rho = n\|\boldsymbol{\mu}\|^2 / \sum_i \|\boldsymbol{\delta}_i\|^2$ das Verhältnis der gemeinsamen zur tokenspezifischen Energie. Jeder Block multipliziert $\rho$ mit $1 + g^4$. Die Tokenstreuung, also der Anteil der Abweichungen an der Norm des Stroms, beträgt $s = (1 + \rho)^{-1/2}$. Für $n$ unabhängige zufällige Token gilt anfangs $\rho_0 = 1/(n-1)$. Nach $k$ Blöcken erwarten wir daher

<!--en:f7e9a4c4-->
Sobald der Mittelwert dominiert, sinkt die Streuung mit jedem Block auf das $1/\sqrt{1 + g^4}$-Fache ihres vorherigen Werts. Bei der Xavier-Initialisierung beträgt dieser Faktor $1/\sqrt{2}$ ($g = 1$). Das FFN kommt in :eqref:`eq_collapse-rate` nicht vor. Es wirkt auf jedes Token einzeln und kann den Mittelwert nicht von den Abweichungen unterscheiden. Deshalb ändert es ihr Verhältnis nur wenig (Übung :exref:`ex_transformer-block-ffn-width-rounding`).

<!--en:6e39a545-->
Bei Pre-Norm (:eqref:`eq_preln-block`) wird der Residualstrom an keiner Stelle durch den Nenner der Normalisierung geteilt. Schreiben wir die Rekursion für $L$ Blöcke aus, erhalten wir direkt die Summe $\mathbf{X}^{(L)} = \mathbf{X}^{(0)} + \sum_{\ell=1}^{2L} \mathbf{U}^{(\ell)}$. Die Ausgaben von $2L$ unabhängigen Teilschichten tragen zur Varianz des Stroms bei. Ohne Skalierung wächst sein quadratisches Mittel daher mit der Tiefe wie $\sqrt{1 + 2\ell\sigma^2}$. Die gesamte Varianz, die alle $2L$ Teilschichten zusammen beitragen, soll von der Ordnung $O(1)$ bleiben, unabhängig von der Tiefe $L$. Dazu skaliert GPT-2 :cite:`Radford.Wu.Child.ea.2019` bei der Initialisierung die Standardabweichung der residualen Ausgabeprojektionen `W_o` und `W_2` jedes Blocks mit $1/\sqrt{2L}$. Das implementieren wir in :numref:`sec_gpt` und prüfen es in Übung :exref:`ex_transformer-block-stream-variance`. Bei Post-Norm verlangsamt BERT :cite:`Devlin.Chang.Lee.ea.2018` den Kollaps aus :eqref:`eq_collapse-rate` durch eine kleinere Standardabweichung $\sigma = 0.02$ bei der Initialisierung aller Gewichte der linearen Schichten. Damit gelten $g^2 \approx 0.10$ und $g^4 \approx 0.01$ bei $d = 256$. Vergleichen wir die drei Konfigurationen (`post-norm`, `pre-norm` und `post-norm, std 0.02`) über 32 Blöcke:

<!--en:66cc7b98-->
Für die Blöcke $k = 1, \ldots, 32$ zeichnen wir die gemessene Tokenstreuung zusammen mit der Vorhersage :eqref:`eq_collapse-rate` für die beiden Post-Norm-Varianten (gepunktet, mit $n = 64$):

<!--en:93166c88-->
Bei Post-Norm mit der üblichen Xavier-Initialisierung (`post-norm`) sinkt die Tokenstreuung bis Block 32 auf nahezu null ($\sim 10^{-4}$). Das entspricht weitgehend der Vorhersage :eqref:`eq_collapse-rate`. Alle Token der Sequenz haben nun fast identische Vektoren. Die gemessene Kurve liegt etwas unterhalb der gepunkteten. Diese kleine Differenz geht auf das FFN zurück. Kleinere Anfangsgewichte bei Post-Norm (`post-norm, std 0.02`) verlangsamen die Kontraktion pro Block so weit, dass die Tokenstreuung über 32 Blöcke erhalten bleibt. Auch Pre-Norm (`pre-norm`) ist nicht immun, denn die Attention-Teilschichten tragen ebenfalls zum gemeinsamen Mittelwert bei. Der Strom wächst jedoch mit der Tiefe. Daher fällt jede Aktualisierung weniger ins Gewicht als die vorherige, und die Streuung nimmt langsam ab. Selbst bei Xavier-Initialisierung bleibt nach 32 Blöcken mehr als ein Drittel der Streuung erhalten. Das quadratische Mittel des Stroms wächst dabei wie $\sqrt{L}$, sofern die Ausgabeprojektionen nicht mit $1/\sqrt{2L}$ skaliert werden (Übung :exref:`ex_transformer-block-stream-variance`).

<!--en:f45fd701-->
### Gradienten bei der Initialisierung

<!--en:f7555db1-->
Der Kollaps der Tokenrepräsentationen tritt im Vorwärtsdurchlauf auf, entscheidet aber auch darüber, welche Gewichte lernen können. Wir zeichnen die Gradientennorm $\|\partial \mathcal{L} / \partial W_q^{(\ell)}\|_F$ der Abfrageprojektion über dem Blockindex $\ell \in \{1, \dots, 32\}$ auf:

<!--en:32430eb0-->
Bei `post-norm` ist der Gradient der Abfrageprojektion in Block 1 mit rund $10^{-2}$ am größten und fällt bis Block 32 auf etwa $10^{-9}$. Bei den verschwindenden Gradienten eines tiefen MLP verhält es sich umgekehrt. Dort erhalten die Schichten mit dem größten Abstand zum Verlust die kleinsten Gradienten. Hier geht der Gradient nicht auf dem Rückweg verloren. Die Ursache liegt in den Eingaben, die die tiefen Blöcke beim Vorwärtsdurchlauf erhalten. Sobald die Token am Eingang eines Blocks nahezu übereinstimmen, erhält jeder Schlüssel denselben Score, und alle Werte sind gleich. Eine Änderung der Abfrage hat dann keinen Einfluss auf die Ausgabe. Der Gradient der Abfrageprojektion schrumpft daher mit dem Quadrat der Tokenstreuung. Ein Faktor entsteht, weil sich die Scores nur durch die Abweichungen $\boldsymbol{\delta}_i$ unterscheiden, der zweite, weil dasselbe für die Werte gilt. Der Gradient der Schlüsselprojektion schrumpft ebenfalls. Die übrigen Gewichte sind davon nicht betroffen. In Block 32 haben die Gradienten von `W_v` und `W_o` bei `post-norm` und `pre-norm` dieselbe Größenordnung (jeweils die letzte Ausgabezeile der beiden Varianten oben).

<!--en:8351bfcd-->
Bei `pre-norm` trägt der Identitätsterm $I$ in $\partial \mathbf{X}^{(L)} / \partial \mathbf{X}^{(\ell)} = I + \cdots$ den Gradienten vom Verlust zu jedem Block. Auch hier ist der Gradient der Abfrageprojektion nicht überall gleich groß. Er nimmt über den Stapel etwa um den Faktor 30 ab, da die Tokenstreuung langsam sinkt. Er bleibt aber über $10^{-3}$, während er bei Post-Norm um mehr als sechs Größenordnungen abnimmt. :citet:`xiong2020layer` untersuchen ein anderes Ungleichgewicht bei der Initialisierung von Post-Norm-Transformern. Die Parameter nahe der Ausgabe haben große Gradienten. Die Autoren zeigen, dass das Training mit Post-Norm deshalb eine Aufwärmphase für die Lernrate benötigt, während Pre-Norm ohne sie auskommt.

<!--en:bb93ce31-->
### RMSNorm

<!--en:032a9029-->
Die übliche Schichtnormalisierung (LayerNorm, :numref:`sec_batch_norm`) zentriert jeden Tokenvektor $\mathbf{x} \in \mathbb{R}^d$ durch Subtraktion seines Mittelwerts $\mu = \frac{1}{d}\sum_j x_j$ und dividiert ihn durch seine Standardabweichung. Dann wendet sie den erlernten Verstärkungsfaktor $\boldsymbol{\gamma}$ und den Bias $\boldsymbol{\beta}$ an. :citet:`Zhang.Sennrich.2019` zeigten, dass die Mittelwertzentrierung und der additive Bias wenig zur Stabilität des Trainings beitragen. Entscheidend ist die Normalisierung mit dem quadratischen Mittel. Lassen wir $\mu$ und $\boldsymbol{\beta}$ weg, erhalten wir **RMSNorm**:

<!--en:d9879676-->
Dabei initialisieren wir alle Einträge von $\boldsymbol{\gamma} \in \mathbb{R}^d$ mit eins, und es gilt $\epsilon = 10^{-5}$. RMSNorm berechnet pro Token nur den Mittelwert der Quadrate $\frac{1}{d}\sum_j x_j^2$. Da Mittelwert und Varianz nicht mehr getrennt berechnet werden müssen, braucht RMSNorm weniger Reduktionen über die gespeicherten Daten und erreicht dieselbe Genauigkeit wie LayerNorm:

<!--en:cbbe1c47-->
### Normalisierung von Abfragen und Schlüsseln (QK-Norm)

<!--en:beb09a8e-->
Pre-Norm normalisiert die Eingabe $\mathbf{X}$ der Attention-Teilschicht. Innerhalb der Teilschicht entstehen die Abfragen $\mathbf{Q} = \textrm{Norm}(\mathbf{X})\mathbf{W}_q$ und Schlüssel $\mathbf{K} = \textrm{Norm}(\mathbf{X})\mathbf{W}_k$ jedoch durch Multiplikation mit den erlernten Projektionsmatrizen $\mathbf{W}_q, \mathbf{W}_k$. Wachsen die Singulärwerte von $\mathbf{W}_q$ und $\mathbf{W}_k$ während des Trainings, wachsen die Logits des skalierten Skalarprodukts $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}$ quadratisch mit der Gewichtsskala. Dadurch gerät die Softmax-Funktion in die One-Hot-Sättigung, in der ihre Gradienten verschwinden :cite:`Henry.Dachapally.Pawar.ea.2020,Dehghani.Djolonga.Mustafa.ea.2023`.

<!--en:37339a69-->
**QK-Norm** wendet in jedem Kopf `RMSNorm` der Breite $d_h$ auf $\mathbf{Q}$ und $\mathbf{K}$ an, unmittelbar bevor die Attention-Scores berechnet werden. Nach der Cauchy-Schwarz-Ungleichung ist das Skalarprodukt zweier Vektoren mit quadratischem Mittel eins betragsmäßig durch $d_h$ beschränkt. Der Betrag des skalierten Logits $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}$ ist daher höchstens $\sqrt{d_h}$ (bei zufälligen Richtungen hat das Logit Varianz eins), unabhängig davon, wie groß $\|\mathbf{W}_q\|$ und $\|\mathbf{W}_k\|$ werden:

<!--en:3cc4f5fd-->
Skalieren wir $Q$ und $K$ ohne QK-Norm mit $\lambda = 4\times$ bzw. $\lambda = 16\times$, wächst die Standardabweichung der Logits quadratisch (um den Faktor $\approx 16$ bzw. $\approx 256$), und die Entropie der Softmax-Verteilung fällt gegen null. Wenden wir `rms_norm` auf $Q$ und $K$ an, bleiben die Streuung der Logits und die Attention-Entropie von der Gewichtsskala vollkommen unabhängig.

<!--en:ca6d3eb0-->
## Das Feedforward-Netz: von GELU zu SwiGLU

<!--en:d639a04d-->
Das Feedforward-Netz des ursprünglichen Transformers ist ein zweischichtiges MLP. Es erhöht die Kanalbreite von $d$ auf $d_{\textrm{ff}} = 4d$, wendet elementweise eine Aktivierungsfunktion an ($\textrm{ReLU}$ bei :citet:`Vaswani.Shazeer.Parmar.ea.2017`, $\textrm{GELU}$ bei GPT-2 und BERT) und projiziert das Ergebnis zurück auf die Breite $d$:

<!--en:b0fa6e7c-->
Dafür benötigt es $2 \cdot d \cdot 4d = 8d^2$ Gewichtsparameter.

<!--en:f916b73a-->
Nach :citet:`Dauphin.Fan.Auli.ea.2017` und :citet:`Shazeer.2020` ersetzen moderne Transformer :eqref:`eq_ffn` durch eine lineare Einheit mit Gate und ohne Bias, **SwiGLU**. Sie bildet das elementweise Produkt aus der mit $\textrm{SiLU}$ aktivierten Gate-Projektion $\mathbf{x}\mathbf{W}_g$ ($\textrm{SiLU}(z) = z \sigma(z)$) und der linearen Projektion $\mathbf{x}\mathbf{W}_1$ in die höhere Dimension. Das Ergebnis projiziert sie zurück nach $\mathbb{R}^d$, und zwar mit $\mathbf{W}_2$:

<!--en:618dc43c-->
`SwiGLU` verwendet *drei* Matrizen. Zwei haben die Form $d \times d_{\textrm{ff}}$ ($\mathbf{W}_g, \mathbf{W}_1$), die dritte die Form $d_{\textrm{ff}} \times d$ ($\mathbf{W}_2$). Wählen wir $d_{\textrm{ff}} = \operatorname{round}(\frac{8}{3}d)$, ergibt sich $3 d d_{\textrm{ff}} \approx 8d^2$. Damit stimmen sowohl die Parameteranzahl als auch die Anzahl der FLOPs mit denen des $4d$ breiten GELU-MLP überein:

<!--en:68a0e4ca-->
## Ein konfigurierbarer `TransformerBlock`

<!--en:8df01549-->
Nun bauen wir `TransformerBlock` auf. Wir wählen zwischen Pre-Norm und Post-Norm (`pre_norm`), zwischen `RMSNorm` und `LayerNorm` (`norm`) sowie zwischen einem FFN mit `SwiGLU` oder `GELU` (`act`). Der Bias lässt sich zuschalten (`bias`). Über die austauschbaren Funktionen `attn_factory` und `ffn_factory` können wir später kausale RoPE-Attention in :numref:`sec_gpt` oder Mixture-of-Experts-Schichten in :numref:`sec_moe` einsetzen, ohne den Block umzuschreiben:

<!--en:bb6567af-->
### Parameteranzahl

<!--en:e774321e-->
Wir vergleichen die moderne Standardkonfiguration (`RMSNorm`, `SwiGLU`, `bias=False`) mit den Varianten von 2017 und GPT-2 (`GELU`, `LayerNorm`, `bias=True`) bei $d = 256, H = 8$. Der Vergleich bestätigt, dass jede Konfiguration $\approx 12 d^2 = 786{,}432$ Parameter enthält ($4d^2$ in der Attention und $8d^2$ im FFN):

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:d646c5ec-->
Ein **Transformer-Block** trennt das **Mischen zwischen Token** (Multi-Head-Attention über die Sequenzpositionen) vom **Mischen zwischen Kanälen** (ein positionsweises Feedforward-Netz über die $d$ Merkmale). Beide Aktualisierungen addiert er zu einem gemeinsamen Residualstrom. **Pre-Norm** (:eqref:`eq_preln-block`) normalisiert nur die Eingabe jeder Teilschicht. Der Residualstrom bleibt ein durchgängiger Identitätspfad. **Post-Norm** (:eqref:`eq_postln-block`) normalisiert den Strom selbst. Bei Xavier-Initialisierung nehmen die Unterschiede zwischen den Token in einem tiefen Post-Norm-Stapel exponentiell ab (:eqref:`eq_collapse-rate`). Dadurch schwinden in den oberen Blöcken die Gradienten der Abfrage- und Schlüsselprojektionen. Skaliert man dagegen bei Pre-Norm die residualen Ausgabeprojektionen jedes Blocks (`W_o` und `W_2`) mit $1/\sqrt{2L}$, bleibt die Varianz des Stroms von der Ordnung $O(1)$, unabhängig von der Tiefe $L$. Moderne Blöcke verwenden **RMSNorm** (:eqref:`eq_rmsnorm`), das auf die Subtraktion des Mittelwerts und den additiven Bias von LayerNorm verzichtet. Hinzu kommt **QK-Norm** je Kopf. Nach der Cauchy-Schwarz-Ungleichung sind die Attention-Logits damit beschränkt, unabhängig davon, wie stark die Projektionsgewichte wachsen. Schließlich ersetzt **SwiGLU** (:eqref:`eq_swiglu`) das $4d$ breite Feedforward-Netz mit zwei Matrizen durch ein Produkt mit Gate und drei Matrizen bei einer Breite von $d_{\textrm{ff}} = \operatorname{round}(\frac{8}{3}d)$. Damit hält es das Parameter- und FLOP-Budget von $8d^2$ eines herkömmlichen FFN ein.

<!--en:2294243a-->
## Übungen

<!--en:ede77121-->
1. **Skaleninvarianz von RMSNorm und die zugehörige Jacobi-Matrix.** Sei $\mathbf{y} = \textrm{RMSNorm}(\mathbf{x})$ wie in :eqref:`eq_rmsnorm` mit $\epsilon = 0$ und $\boldsymbol{\gamma} = \mathbf{1}$ definiert.
   1. Zeigen Sie, dass $\textrm{RMSNorm}(c\mathbf{x}) = \textrm{RMSNorm}(\mathbf{x})$ für jeden Skalar $c > 0$ gilt.
   2. Differenzieren Sie $\mathbf{y} = \sqrt{d}\, \mathbf{x} / \|\mathbf{x}\|_2$ und zeigen Sie so, dass die Jacobi-Matrix durch $J(\mathbf{x}) = \frac{\sqrt{d}}{\|\mathbf{x}\|_2}\bigl(\mathbf{I} - \frac{\mathbf{x}\mathbf{x}^\top}{\|\mathbf{x}\|_2^2}\bigr)$ gegeben ist. Prüfen Sie, dass $J(\mathbf{x})\mathbf{x} = \mathbf{0}$ gilt. Warum muss jede radiale Störung parallel zu $\mathbf{x}$ im Nullraum von $J(\mathbf{x})$ liegen?

<!--en:c581ee64-->
2. **Ausgeschriebener Residualstrom und Identitätspfad des Gradienten.** Nummerieren Sie die $2L$ Teilschichten eines Pre-Norm-Stapels aus $L$ Blöcken als $f_1, f_2, \dots, f_{2L}$. Ungerade Indizes bezeichnen Attention-Teilschichten, gerade Indizes FFN-Teilschichten.
   1. Schreiben Sie die Rekursion :eqref:`eq_preln-block` aus, um $\mathbf{X}^{(L)}$ explizit als $\mathbf{X}^{(0)} + \sum_{m=1}^{2L} f_m(\textrm{Norm}(\mathbf{X}^{(m-1)/2}))$ darzustellen.
   2. Drücken Sie $\partial \mathbf{X}^{(L)} / \partial \mathbf{X}^{(0)}$ mit der Kettenregel als Produkt von $2L$ Jacobi-Matrizen aus. Multiplizieren Sie die führenden Terme aus und vergleichen Sie den entstehenden Identitätsterm $\mathbf{I}$ mit dem Produkt bei Post-Norm. Dort wird jeder Faktor von links mit $J_{\textrm{Norm}}$ multipliziert.

<!--en:a9c38797-->
3. **QK-Norm in `TransformerBlock`.** Schreiben Sie eine Funktion `attn_factory`, die `d2l.MultiHeadAttention` kapselt und vor der Berechnung der Attention-Scores in jedem Kopf RMSNorm der Breite `d // h` auf Abfragen und Schlüssel anwendet. Multiplizieren Sie die Projektionsgewichte eines Stapels aus 16 Blöcken mit $4\times$ (oder $16\times$). Prüfen Sie, dass die Attention-Entropie durch eine positive Konstante nach unten beschränkt bleibt.

<!--en:bb51fc17-->
4. **Varianzbudget des Residualstroms über die Tiefe.** Bei Pre-Norm addiert jede Teilschicht eine Aktualisierung zum Residualstrom. Führen Sie `signal_stats(pre_norm=True, std=0.02 / math.sqrt(2 * L), num_blks=L)` für $L \in \{8, 16, 32, 64\}$ aus und vergleichen Sie die Ergebnisse mit der unskalierten Variante `std=None`. Prüfen Sie, dass die Standardabweichung des Stroms im letzten Block $L$ ohne Skalierung wie $\sqrt{L}$ wächst. Prüfen Sie auch, dass sie unabhängig von $L$ nahezu konstant bleibt, wenn Sie die Skalierungsregel mit $1/\sqrt{2L}$ anwenden.
   :exlabel:`ex_transformer-block-stream-variance`

<!--en:b6f90639-->
5. **Rate des Kollapses der Tokenrepräsentationen.** Die Vorhersage :eqref:`eq_collapse-rate` enthält keinen Faktor für das FFN. Die Begründung lautet, dass nur die Attention-Teilschicht Informationen zwischen den Token mischt, während das FFN positionsweise arbeitet.
   1. Ändern Sie `MiniBlock` so, dass der Block die Attention-Teilschicht überspringt ($\mathbf{H} = \textrm{Norm}(\mathbf{X})$) und nur das Post-Norm-FFN ausführt. Führen Sie `signal_stats(pre_norm=False)` erneut aus. Entfernen Sie vorher die Zeilen zur Gradientenberechnung, da keine Attention-Gewichte mehr zum Einsatz kommen. Wie viel Tokenstreuung bleibt nach 32 Blöcken übrig?
   2. Überspringen Sie nun das FFN und behalten Sie die Attention-Teilschicht bei. Wie gut stimmt die gemessene Streuung mit :eqref:`eq_collapse-rate` für $k = 8, 16, 32$ überein?
   3. Auch im reinen FFN-Stapel nimmt die Streuung etwas ab. Welche Eigenschaft von GELU ermöglicht es einem positionsweisen Netz, einen kleinen Vektor zu addieren, der allen Token gemeinsam ist?
   :exlabel:`ex_transformer-block-ffn-width-rounding`

<!--en:3cce5cba-->
[Diskussionen](https://d2l.discourse.group/t/36774)

<!--en:26de144d-->
Der Transformer-Block: Pre-Norm, SwiGLU und Signalausbreitung<br>**Residualstrom · Pre-Norm und Post-Norm · Kollaps der Tokenrepräsentationen · RMSNorm & QK-Norm · SwiGLU · Parameteranzahl 12d²**

<!--en:ddf34141-->
::: {.slide title="Warum Attention allein bei großer Tiefe versagt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:3bca3ab5-->
- **Fehlende Nichtlinearität pro Token:** Ohne positionsweises FFN zwischen den Attention-Schichten reduzieren sich die linearen Projektionen auf eine einzige Abbildung von niedrigem Rang über die Kanäle jedes Tokens.

<!--en:e7849f3e-->
- **Kollaps der Tokenrepräsentationen und schwindende Gradienten:** Bei $L = 32$ gestapelten Post-Norm-Blöcken mit Xavier-Initialisierung sinkt die Tokenstreuung auf $\sim 10^{-4}$. In Block 32 fällt die Gradientennorm der Abfrageprojektion $\|\partial \mathcal{L} / \partial \mathbf{W}_q^{(32)}\|_F$ auf $\sim 10^{-9}$.

<!--en:fd196e07-->
::: {.slide title="Mischen zwischen Token und zwischen Kanälen im Residualstrom"}

<!--en:37774c41-->
[Aufbau eines Blocks]{.kicker}

<!--en:a9005933-->
- **Gemeinsamer Residualstrom $\mathbf{X} \in \mathbb{R}^{B \times n \times d}$:** Jede Teilschicht liest aus dem Strom, berechnet eine Aktualisierung in $\mathbb{R}^{B \times n \times d}$ und addiert sie zum Strom, statt die Repräsentationen zu überschreiben.

<!--en:b15b71b1-->
- **Multi-Head-Attention (Mischen zwischen Token):** Der *einzige* Operator, bei dem die Sequenzpositionen $1, \dots, n$ Informationen austauschen ($4d^2$ Parameter).

<!--en:87e27c8f-->
- **Positionsweises FFN (Mischen zwischen Kanälen):** Dasselbe MLP verarbeitet jeden Tokenvektor $\mathbf{x}_i \in \mathbb{R}^d$ unabhängig und transformiert so den durch Attention gesammelten Kontext ($8d^2$ Parameter, also zwei Drittel der Blockparameter).

<!--en:03cf153e-->
::: {.slide title="Position der Normalisierung bei Post-Norm und Pre-Norm"}

<!--en:c0dae5ea-->
[Position der Normalisierung]{.kicker}

<!--en:ba828e07-->
- **Post-Norm (2017 / BERT):** Die Normalisierung wirkt direkt auf den Residualstrom:
  $$\mathbf{H} = \mathrm{Norm}_1(\mathbf{X} + \mathrm{Attn}(\mathbf{X})), \quad \mathbf{Y} = \mathrm{Norm}_2(\mathbf{H} + \mathrm{FFN}(\mathbf{H}))$$

<!--en:62335f5d-->
- **Pre-Norm (GPT-2 / Llama / Qwen):** Die Normalisierung liegt im Eingabezweig jeder Teilschicht. Der Hauptstrom bleibt durchgängig:
  $$\mathbf{H} = \mathbf{X} + \mathrm{Attn}(\mathrm{Norm}_1(\mathbf{X})), \quad \mathbf{Y} = \mathbf{H} + \mathrm{FFN}(\mathrm{Norm}_2(\mathbf{H}))$$

<!--en:755cfb88-->
::: {.slide title="Zwei Codezeilen unterscheiden Post-Norm und Pre-Norm"}

<!--en:ce0f4436-->
[Implementierung]{.kicker}

<!--en:5d873470-->
Der Testblock `MiniBlock` verwendet für beide Anordnungen identische Gewichte. Nur der Ansatzpunkt von `norm1` und `norm2` ändert sich: die Summe im Residualstrom oder die Eingaben der Zweige.

<!--en:45f2a54f-->
Ein untrainierter Stapel aus $L$ Blöcken ist bei gegebenen Zufallsgewichten eine deterministische Funktion. So lässt sich schon bei der Initialisierung ($t = 0$) messen, wie sich Signalvarianz, Unterschiede zwischen Token und Gradienten in beiden Anordnungen über die Schichten entwickeln.

<!--en:8041dbf5-->
::: {.slide title="Warum sich Token bei Post-Norm mit jeder Schicht angleichen"}

<!--en:7b76940e-->
[Mechanismus des Kollapses der Tokenrepräsentationen]{.kicker}

<!--en:6d16f412-->
- Jeden Tokenvektor $\mathbf{x}_i = \boldsymbol{\mu} + \boldsymbol{\delta}_i$ in den Sequenzmittelwert $\boldsymbol{\mu} = \frac{1}{n}\sum_j \mathbf{x}_j$ und die tokenspezifische Abweichung $\boldsymbol{\delta}_i$ zerlegen.

<!--en:b1fd1ba4-->
- **Zufällig initialisierte Attention verstärkt $\boldsymbol{\mu}$:** Gleichmäßige Anfangsgewichte addieren $\mathbf{W}_o\mathbf{W}_v\boldsymbol{\mu}$ zu jedem Token, während sich die Abweichungen $\boldsymbol{\delta}_i$ aufheben.

<!--en:517d8b6d-->
- **Post-Norm skaliert nach unten:** Die Division aller Token durch denselben Faktor stellt die Länge des Stroms wieder her. Dabei schrumpft $\boldsymbol{\delta}_i$ in jeder Schicht relativ zu $\boldsymbol{\mu}$ (`LayerNorm` zentriert die *Kanäle*, nicht die Token).

<!--en:c5b14f15-->
::: {.slide title="Vorhersage der exponentiellen Rate des Kollapses der Tokenrepräsentationen"}

<!--en:de057a66-->
[Mathematische Herleitung]{.kicker}

<!--en:ee777553-->
- Sei $\rho = n\|\boldsymbol{\mu}\|^2 / \sum_{i=1}^n \|\boldsymbol{\delta}_i\|^2$ das Verhältnis der gemeinsamen zur tokenspezifischen Energie. Dann beträgt die relative Tokenstreuung $s = (1 + \rho)^{-1/2}$.

<!--en:8ba6b756-->
- Beim Varianzgewinn $g^2 = d\sigma^2$ je Projektion erhöht jeder Post-Norm-Block das Verhältnis gemäß $\rho \mapsto (1 + g^4)\rho$:
  $$s_k \approx \left(1 + \frac{(1 + g^4)^k}{63}\right)^{-1/2}$$

<!--en:f722f38b-->
::: {.slide title="Gemessene Tokenstreuung im Einklang mit der analytischen Vorhersage"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:0a5054d9-->
- **Post-Norm + Xavier (`C0`):** Die Tokenstreuung fällt mit der Tiefe exponentiell, bis auf $\sim 10^{-4}$ in Block 32. Sie liegt knapp unter der gepunkteten Kurve für Attention allein, weil der positive Mittelwert von `GELU` auch im FFN einen kleinen gemeinsamen Vektor beiträgt.

<!--en:c1be2f73-->
- **Post-Norm + $\mathcal{N}(0, 0.02^2)$ (`C2`):** Kleine Gewichte sorgen für $g^4 \approx 0.01$. Damit bleibt die Tokenstreuung über 32 Schichten bei $0.97$, wenn die Breite $d = 256$ beträgt.

<!--en:f14628e7-->
- **Pre-Norm + Xavier (`C1`):** Selbst bei großen Xavier-Gewichten ($g = 1$) bleiben deutliche Unterschiede zwischen den Token erhalten ($s_{32} \approx 0.37$). Ein exponentieller Kollaps bleibt aus.

<!--en:0c156638-->
::: {.slide title="Fallstrick bei fester Gewichtsskala 0.02 und wachsender Modellbreite"}

<!--en:af5b852a-->
[Skalierung mit der Breite]{.kicker}

<!--en:131ee3a1-->
- Für $\mathbf{W} \in \mathbb{R}^{d \times d}$ mit fester Varianz $\sigma^2 = 0.02^2$ der Einträge wächst der Varianzgewinn je Projektion linear mit der Breite $d$:
  $$g^2 = d \sigma^2 = 0.0004\,d$$

<!--en:42e34be6-->
- Ein festes $\sigma = 0.02$ ist nur für $d \ll 2{,}500$ klein. Bei Breiten heutiger Spitzenmodelle ($d = 4{,}096$) führt $g^2 = 1.64 > 1$ in Post-Norm-Stapeln noch schneller zum Kollaps als die Xavier-Initialisierung.

<!--en:ab1747ed-->
::: {.slide title="Schwindende Gradienten der Abfrage- und Schlüsselprojektionen beim Kollaps der Tokenrepräsentationen"}

<!--en:f410a91e-->
[Signalausbreitung im Rückwärtsdurchlauf]{.kicker}

<!--en:a02928ae-->
- **Verschwindende Gradienten von $\mathbf{W}_q, \mathbf{W}_k$ am Ende des Stapels:** Sind die Eingabetoken $\mathbf{x}_1, \dots, \mathbf{x}_n$ von Block 32 nahezu identisch, verschiebt eine Störung von $\mathbf{W}_q$ oder $\mathbf{W}_k$ alle Attention-Scores gleichermaßen. Der mit der Softmax-Funktion gewichtete Mittelwert bleibt unverändert ($\|\nabla_{\mathbf{W}_q^{(32)}} \mathcal{L}\|_F \sim 10^{-9}$).

<!--en:96c2d2a6-->
- **Keine Abschwächung auf dem Rückweg:** Bei Post-Norm ist der Gradient von $\mathbf{W}_q$ in Block 1 am größten ($\sim 10^{-2}$) und schrumpft mit dem Quadrat der Tokenstreuung. Die Gradienten von $\mathbf{W}_v$ und $\mathbf{W}_o$ in Block 32 haben dieselbe Größenordnung wie bei Pre-Norm. Bei **Pre-Norm** (`C1`) fällt der Abfragegradient über 32 Blöcke um einen Faktor von etwa 30, nicht um mehr als sechs Größenordnungen.

<!--en:887544c3-->
::: {.slide title="Ausgeschriebener Pre-Norm-Strom und Residualskalierung mit 1/√(2L)"}

<!--en:ee2de8be-->
[Varianz bei Pre-Norm]{.kicker}

<!--en:88fa45c7-->
- Schreibt man die Rekursion für $L$ Pre-Norm-Blöcke aus, wird der durchgängige Identitätspfad sichtbar:
  $$\mathbf{X}^{(L)} = \mathbf{X}^{(0)} + \sum_{\ell=1}^{L} \bigl(\mathrm{Attn}_\ell(\mathrm{Norm}(\mathbf{X}^{(\ell-1)})) + \mathrm{FFN}_\ell(\mathrm{Norm}(\mathbf{H}^{(\ell)}))\bigr)$$

<!--en:d265f579-->
- **Skalierung mit $1/\sqrt{2L}$ bei GPT-2:** Multipliziert man die anfängliche Standardabweichung der beiden Matrizen, die pro Block zum Residualstrom beitragen (`W_o` und `W_2`), mit $1/\sqrt{2L}$, gilt für die aufsummierte Varianz des Stroms $\sum_{m=1}^{2L} \frac{\sigma^2}{2L} = \sigma^2 = O(1)$ bei jeder Tiefe $L$.

<!--en:6bdb215e-->
::: {.slide title="RMSNorm: schneller ohne Mittelwertzentrierung"}

<!--en:d25cce71-->
[Form der Normalisierung]{.kicker}

<!--en:2925b450-->
Zhang und Sennrich (2019) zeigten, dass die Stabilität von `LayerNorm` auf der Begrenzung des quadratischen Mittels beruht. Die Zentrierung um den Kanalmittelwert $\mu = \frac{1}{d}\sum_j x_j$ ist dafür nicht ausschlaggebend:

<!--en:10844088-->
**Invarianz bei Skalierung und Verschiebung:** Beide Normalisierungen sind gegenüber positiver Skalierung ($\mathbf{x} \mapsto 3\mathbf{x}$) invariant. Eine konstante Verschiebung ($\mathbf{x} \mapsto \mathbf{x} + 10$) ändert das Ergebnis von `RMSNorm`, während das von `LayerNorm` gleich bleibt. Ohne $\mu$ und $\boldsymbol{\beta}$ entfällt je Token ein Reduktionsdurchlauf über den HBM.

<!--en:36fa3566-->
::: {.slide title="Begrenzung der Logits durch Normalisierung von Abfragen und Schlüsseln (QK-Norm)"}

<!--en:0821ecab-->
[Stabilität der Attention]{.kicker}

<!--en:f7c75717-->
- **Quadratisches Anwachsen der Logits:** Wachsen die Spektralnormen von $\mathbf{W}_q$ und $\mathbf{W}_k$ beim Training um den Faktor $\lambda$, steigen die Attention-Logits $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}$ um den Faktor **$\lambda^2$**. Die Softmax-Funktion gerät in Sättigung.

<!--en:0d385a2e-->
- **`QK-Norm` je Kopf:** Bei Kopfbreite $d_h$ erzwingt `RMSNorm` für Abfragen und Schlüssel die Normen $\|\hat{\mathbf{q}}_i\|_2 = \|\hat{\mathbf{k}}_j\|_2 = \sqrt{d_h}$. Daraus folgt mit der Cauchy-Schwarz-Ungleichung die feste Schranke:
  $$\frac{|\hat{\mathbf{q}}_i^\top \hat{\mathbf{k}}_j|}{\sqrt{d_h}} \le \sqrt{d_h}$$

<!--en:71030f11-->
::: {.slide title="Empirische Prüfung der Invarianz von Logits und Entropie bei QK-Norm"}

<!--en:7a0e5db5-->
[Überprüfung von QK-Norm]{.kicker}

<!--en:18b06421-->
Die Skalierung von $\mathbf{W}_q$ und $\mathbf{W}_k$ mit $\lambda \in \{1, 4, 16\}$ bei $n = 64$ zufälligen Token ($d_h = 64$, Entropie der Gleichverteilung $\ln 64 \approx 4.16$ Nat) zeigt, wie die Softmax-Verteilung kollabiert und wie `QK-Norm` dem entgegenwirkt:

<!--en:1175030b-->
::: {.slide title="Konstante Logitstreuung und Entropie durch QK-Norm"}

<!--en:7a0e5db5-2-->
[Überprüfung von QK-Norm]{.kicker}

<!--en:df5cff3c-->
Ohne `QK-Norm` erhöht eine Skalierung der Gewichte um den Faktor $16\times$ die Standardabweichung der Logits um den Faktor $16^2 = 256\times$. Die Attention-Entropie fällt auf $\approx 0.01\text{--}0.03$ Nat (One-Hot-Verteilung). Mit `QK-Norm` bleiben Logitstreuung ($\approx 1.0$) und Entropie ($\approx 3.7$) völlig unabhängig von der Gewichtsskala.

<!--en:ef75e1ef-->
::: {.slide title="Das Feedforward-Netz als Schlüssel-Wert-Speicher"}

<!--en:d1fa7d93-->
[Mischen zwischen Kanälen]{.kicker}

<!--en:0787ea63-->
- Schreibt man das zweischichtige FFN mithilfe der Zeilen $\mathbf{w}_{1,j} \in \mathbb{R}^d$ von $\mathbf{W}_1$ und der Spalten $\mathbf{w}_{2,j} \in \mathbb{R}^d$ von $\mathbf{W}_2$, erkennt man einen inhaltsadressierten Speicher (Geva et al., 2021):
  $$\mathrm{FFN}_{\mathrm{GELU}}(\mathbf{x}) = \sum_{j=1}^{4d} \mathrm{GELU}(\mathbf{x}^\top \mathbf{w}_{1,j})\, \mathbf{w}_{2,j}$$

<!--en:62aa7037-->
- **Größter Parameteranteil:** Zwei Matrizen der Form $d \times 4d$ enthalten $8d^2$ Parameter ($3{,}072$ Detektoren pro Block in GPT-2 Small).

<!--en:c3d7788e-->
::: {.slide title="Vom GELU-MLP zum SwiGLU-Netz mit Gate"}

<!--en:05e292cd-->
[Feedforward-Netze mit Gate]{.kicker}

<!--en:0e6886c9-->
- Moderne Transformer ersetzen das `GELU`-MLP mit zwei Matrizen durch ein **`SwiGLU`**-Netz mit Gate und ohne Bias (Shazeer, 2020):
  $$\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x}) = \bigl(\mathrm{SiLU}(\mathbf{x}\mathbf{W}_g) \odot (\mathbf{x}\mathbf{W}_1)\bigr)\mathbf{W}_2$$

<!--en:091bf24c-->
- **Gleiches Budget von $8d^2$:** Sinkt die Breite der versteckten Schicht von $4d$ auf $d_{\text{ff}} = \operatorname{round}(\frac{8}{3}d)$, beträgt die Parameteranzahl der drei Matrizen weiterhin $3 \times d \times \frac{8}{3}d = 8d^2$.

<!--en:37fb06bc-->
::: {.slide title="Beide FFN-Varianten mit gleichem Parameterbudget in einer Klasse"}

<!--en:12e5b97f-->
[Implementierung von FeedForward]{.kicker}

<!--en:aea5dd9f-->
Das Modul `FeedForward` bietet zwei Varianten: bei Breite $4d$ ein `GELU`-MLP, bei Breite $\frac{8}{3}d$ ein `SwiGLU`-Netz. Für beide gilt dasselbe Parameter- und FLOP-Budget von $8d^2$:

<!--en:9ff50588-->
Bei $d = 256$ benötigt `act='gelu'` insgesamt $2 \times 256 \times 1024 = 524{,}288$ Parameter, `act='swiglu'` dagegen $3 \times 256 \times 683 = 524{,}544$ Parameter ($8.004\,d^2$ durch Rundung von $\frac{8}{3}\cdot 256 = 682.67 \to 683$).

<!--en:61b15b2f-->
::: {.slide title="Ein konfigurierbarer TransformerBlock und seine Parameteranzahl"}

<!--en:3deca60b-->
[Gemeinsamer Block und Parameteranzahl]{.kicker}

<!--en:a4a7966c-->
`TransformerBlock` vereint `norm` (`'rms'` oder `'layer'`), `act` (`'swiglu'` oder `'gelu'`), `pre_norm`, `bias` und die austauschbaren Funktionen `attn_factory` und `ffn_factory`. In allen Varianten enthält der Block $\approx 12d^2$ Parameter:

<!--en:c0bba112-->
**Faustregel für $12d^2$ Parameter:** In jedem dichten Block mit $H_{\text{kv}} = H$ enthält die Multi-Head-Attention **$4d^2$** Parameter ($\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$) und das FFN **$8d^2$** Parameter. Zusammen sind das **$12d^2$** pro Schicht ($7.08\text{M}$ pro Block bei der GPT-2-Breite $d = 768$).

<!--en:f4f3c16f-->
::: {.slide title="Verständnisfragen zum Entwurf eines Transformer-Blocks"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:38a668a3-->
Drei Fragen zum Entwurf beantworten:

<!--en:361209a4-->
1. **Fallstrick beim Ablationsvergleich:** In einem Experiment wird ein $4d$ breites `GELU`-FFN bei $d = 768$ durch ein `SwiGLU`-FFN ersetzt. Versehentlich bleibt die Breite der versteckten Schicht bei $4d = 3{,}072$, statt auf $\frac{8}{3}d = 2{,}048$ zu sinken. Wie viele zusätzliche Parameter enthält dadurch jeder Block?

<!--en:7a60198c-->
2. **Skalierung mit `QK-Norm`:** Ohne `QK-Norm` werden $\mathbf{W}_q$ und $\mathbf{W}_k$ mit $\lambda = 8$ multipliziert. Um welchen Faktor ändert sich dadurch die Standardabweichung der Attention-Logits? Wie groß ist der Faktor mit `QK-Norm`?

<!--en:77c416a9-->
3. **Skalierung mit der Tiefe:** Mit welchem Faktor skaliert GPT-2 in einem Pre-Norm-Modell aus 32 Blöcken die anfängliche Standardabweichung von $\mathbf{W}_o$ und $\mathbf{W}_2$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:ca381658-->
1. **$+4d^2 \approx 2.36\text{M}$ zusätzliche Parameter pro Block** ($12d^2 = 7.08\text{M}$ gegenüber $8d^2 = 4.72\text{M}$, also $50\%$ mehr Parameter im FFN). So vermischen sich im Vergleich die Einflüsse von Architekturgüte und Modellkapazität.

<!--en:fb855a40-->
2. **$\lambda^2 = 64\times$** ohne `QK-Norm` (die Softmax-Verteilung nähert sich einer One-Hot-Verteilung), **$1\times$ (invariant)** mit `QK-Norm`.

<!--en:8bcfa971-->
3. **$1 / \sqrt{2L} = 1 / \sqrt{64} = 1/8 = 0.125$**. Damit bleibt die Summe der Varianzen aller $2L = 64$ Residualzweige von der Ordnung $O(1)$.

<!--en:94e38d41-->
::: {.slide title="Aufbau und Stabilität des Transformer-Blocks im Überblick"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:315df129-->
- **Mischen zwischen Token ($4d^2$) und zwischen Kanälen ($8d^2$):** Multi-Head-Attention und `SwiGLU`-FFN ($d_{\text{ff}} = \frac{8}{3}d$) haben zusammen $12d^2$ Parameter pro Block.

<!--en:ad760aa0-->
- **Pre-Norm und Ausgabeskalierung mit $1/\sqrt{2L}$:** Der Identitätspfad für den Gradienten bleibt erhalten, und die aufsummierte Varianz des Stroms bleibt von der Ordnung $O(1)$ bei jeder Tiefe $L$.

<!--en:19742837-->
- **`RMSNorm` und `QK-Norm`:** Schnelle Skalennormalisierung und Cauchy-Schwarz-Schranke für die Logits ($\le \sqrt{d_h}$)
