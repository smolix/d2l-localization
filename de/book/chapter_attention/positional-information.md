<!--en:eaf2b34b-->
# Positionsinformation

<!--en:81c8f7a4-->
Die Sätze *„Hund beißt Mann“* und *„Mann beißt Hund“* enthalten dieselben drei Wörter. Die unmaskierte Self-Attention-Schicht aus :numref:`sec_multihead-attention` liefert für beide exakt dieselben Ausgabevektoren in vertauschter Reihenfolge, denn die Attention-Gewichte hängen allein vom Token-Inhalt ab. Damit das Modell die Wortreihenfolge unterscheiden kann, muss die Positionsinformation in die Darstellung oder in die Berechnung der Scores eingehen.

<!--en:9437f17f-->
Die Verfahren lassen sich in drei Gruppen einteilen. Die erste addiert einen Positionsvektor $\mathbf{p}_i$ zu jeder Token-Einbettung (engl. token embedding). Die zweite dreht Abfragen (engl. queries) und Schlüssel (engl. keys) um positionsabhängige Winkel (*Rotary Position Embeddings*, RoPE). Die dritte verringert die Attention-Scores um eine Abstandsstrafe (engl. distance penalty) (*ALiBi*). Alle drei unterscheiden Positionen innerhalb der Kontextlänge des Trainings. Bei der Extrapolation auf längere Sequenzen verhalten sie sich jedoch sehr unterschiedlich. Zunächst formulieren wir die Permutationssymmetrie der Self-Attention mathematisch. Anschließend konstruieren und vergleichen wir fünf Verfahren anhand eines Byte-Sprachmodells mit zwei reinen Attention-Schichten (`TinyCharLM`). Wir trainieren es bei Länge $128$ und werten es bis zur Länge $512$ aus.

<!--en:d55116c7-->
## Permutationsäquivarianz

<!--en:bbac62b1-->
Die Matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ enthalte $n$ Token-Vektoren. Es bezeichne $f(\mathbf{X}) = \mathrm{softmax}(\mathbf{S})\,\mathbf{X}\mathbf{W}_v$ mit $\mathbf{S} = \mathbf{X}\mathbf{W}_q(\mathbf{X}\mathbf{W}_k)^\top/\sqrt{d_h}$ die unmaskierte Self-Attention (:numref:`sec_multihead-attention`) bei ausgeschaltetem Dropout. Ordnen wir die $n$ Token um, so entspricht das einer Multiplikation von $\mathbf{X}$ von links mit einer $n \times n$-Permutationsmatrix $\boldsymbol{\Pi}$.

<!--en:588907e2-->
**Proposition (Permutationsäquivarianz).** Für jede Permutationsmatrix $\boldsymbol{\Pi}$ gilt $f(\boldsymbol{\Pi} \mathbf{X}) = \boldsymbol{\Pi}
f(\mathbf{X})$.

<!--en:1e186a78-->
**Beweis.** Da lineare Projektionen zeilenweise wirken, gilt $(\boldsymbol{\Pi}\mathbf{X})\mathbf{W} = \boldsymbol{\Pi}(\mathbf{X}\mathbf{W})$ für $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$. Die Score-Matrix geht in $\boldsymbol{\Pi} \mathbf{S} \boldsymbol{\Pi}^\top$ über. Die zeilenweise angewandte Softmax-Funktion vertauscht mit Zeilen- und Spaltenpermutationen. Daher gilt $\mathrm{softmax}(\boldsymbol{\Pi} \mathbf{S} \boldsymbol{\Pi}^\top) =
\boldsymbol{\Pi}\, \mathrm{softmax}(\mathbf{S})\, \boldsymbol{\Pi}^\top$. Multiplizieren wir mit den permutierten Werten $\boldsymbol{\Pi}\mathbf{X}\mathbf{W}_v$ und nutzen die Identität $\boldsymbol{\Pi}^\top \boldsymbol{\Pi} = \mathbf{I}$, so erhalten wir $\boldsymbol{\Pi} f(\mathbf{X})$. Die Äquivarianz bleibt erhalten, wenn wir die Ausgaben der Köpfe konkatenieren und $\mathbf{W}_o$ zeilenweise anwenden. $\blacksquare$

<!--en:a448c398-->
Wir überprüfen die Permutationsäquivarianz numerisch an einer `MultiHeadAttention`-Schicht mit vier Köpfen:

<!--en:7794bbee-->
Auch wenn wir äquivariante Schichten oder auf einzelne Token wirkende mehrschichtige Perzeptrons (engl. multilayer perceptrons, MLPs) stapeln, bleibt die Äquivarianz erhalten. Eine *kausale Maske* bricht jedoch die Permutationssymmetrie, denn die Attention an Position $i$ berücksichtigt $i+1$ Positionen einschließlich der eigenen. Diese Asymmetrie nutzt weiter unten *NoPE*.

<!--en:8ed36138-->
## Absolute Positionseinbettungen

<!--en:8953f510-->
Der ursprüngliche Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017` bricht die Permutationssymmetrie bereits an der Eingabe. Dazu addiert er einen Positionsvektor $\mathbf{p}_i
\in \mathbb{R}^d$ (Zeile $i$ von $\mathbf{P}$) zu jeder Token-Einbettung $\mathbf{x}_i$. Eine Permutation der Token bildet somit $\mathbf{X} + \mathbf{P}$ auf $\boldsymbol{\Pi}\mathbf{X} + \mathbf{P} \neq
\boldsymbol{\Pi}(\mathbf{X} + \mathbf{P})$ ab.

<!--en:f0d50527-->
### Gelernte Positionen

<!--en:7382974a-->
Modelle wie BERT :cite:`Devlin.Chang.Lee.ea.2018` und GPT-2 :cite:`Radford.Wu.Child.ea.2019` behandeln $\mathbf{P} \in \mathbb{R}^{L_{\max}
\times d}$ als trainierbare Einbettungstabelle. Bis zu einer festen Maximallänge $L_{\max}$ enthält sie für jede Position eine Parameterzeile. Da die Zeilen jenseits der Kontextlänge des Trainings nie Gradienten erhalten, kann eine gelernte Tabelle nicht auf unbekannte Positionen extrapolieren.

<!--en:b383b8bf-->
### Sinusförmige Kodierungen

<!--en:66cb9e89-->
Um Positionsvektoren für beliebige Längen ohne gelernte Parameter zu definieren, füllen :citet:`Vaswani.Shazeer.Parmar.ea.2017` die Zeile $i$ von $\mathbf{P}$ mit Sinus- und Kosinuswerten zu $d/2$ geometrisch gestaffelten Frequenzen $\omega_m =
10000^{-2m/d}$:

<!--en:aa50e297-->
Die Frequenz $\omega_m$ sinkt von $1$ Radiant pro Token ($m = 0$) bis nahe an $10^{-4}$ ($m = d/2 - 1$). So wirken die Spaltenpaare wie ein kontinuierlicher Binärzähler. Spalten mit kleinen Indizes schwingen schnell und unterscheiden benachbarte Positionen. Spalten mit großen Indizes ändern sich über weite Bereiche nur langsam.

<!--en:bcda5c59-->
Die Heatmap der vollständigen $60 \times 32$-Kodierungsmatrix `P` zeigt links die schnell schwingenden Paare mit hohen Frequenzen. Nach rechts gehen sie in langsam veränderliche Paare mit niedrigen Frequenzen über, ähnlich den Binärstellen eines ganzzahligen Zählers:

<!--en:69b3a8eb-->
### Sinusförmige Kodierungen als Drehungen

<!--en:d9fb6f03-->
Verschieben wir die Position $i$ um den Versatz $\delta$, so ergibt sich nach den trigonometrischen Additionstheoremen eine $2 \times 2$-Drehmatrix, die nur von $\delta$ abhängt. Sie wirkt auf das jeweilige Frequenzpaar $(2m, 2m+1)$:

<!--en:ff9ec4d9-->
Da wir jedoch $\mathbf{p}_i$ zu $\mathbf{x}_i$ *addieren*, ergeben sich für den bilinearen Attention-Score mit $\mathbf{A} = \mathbf{W}_q \mathbf{W}_k^\top /
\sqrt{d_h}$ beim Ausmultiplizieren vier Terme:

<!--en:cff8dc2e-->
Die beiden gemischten Terme koppeln den Inhalt an die *absolute* Position. Sie ändern sich, wenn wir $i$ und $j$ gemeinsam verschieben. Selbst $\mathbf{p}_i^\top \mathbf{A}\,
\mathbf{p}_j$ hängt nur dann allein von $j - i$ ab, wenn $\mathbf{A}$ blockdiagonal ist und die $2 \times 2$-Blöcke skalierte Drehungen beschreiben (:exref:`ex_positional-information-additive-codes`).

<!--en:31aead68-->
## Rotary Position Embeddings

<!--en:c5810dcc-->
*Rotary Position Embeddings* (RoPE) :cite:`Su.Lu.Pan.ea.2021` wenden die Positionstransformationen *multiplikativ* auf Abfragen und Schlüssel an. Dadurch hängt jeder Score hinsichtlich der Position nur vom relativen Versatz $j - i$ ab:

<!--en:5fdb4a90-->
Fordern wir $\mathbf{R}_i^\top \mathbf{R}_j = \mathbf{R}_{j-i}$ und hat $\mathbf{R}_0$ vollen Rang, so folgt (1) $\mathbf{R}_0 = \mathbf{I}$ aus $\mathbf{R}_0^\top
\mathbf{R}_0 = \mathbf{R}_0$. Weiter gilt (2) $\mathbf{R}_i^\top \mathbf{R}_i =
\mathbf{I}$, also ist jede Matrix $\mathbf{R}_i$ orthogonal und normerhaltend. Schließlich gilt (3) $\mathbf{R}_{i+j} = \mathbf{R}_i \mathbf{R}_j$ und damit $\mathbf{R}_i =
\mathbf{R}_1^i$. In einer geeigneten Orthonormalbasis (der Basiswechsel lässt sich in $\mathbf{W}_q, \mathbf{W}_k$ einbeziehen) zerfällt die hier verwendete orthogonale Matrix $\mathbf{R}_1$ in $2 \times 2$-Blöcke für Drehungen in der Ebene. RoPE unterteilt die $d_h$ Koordinaten jeder Abfrage und jedes Schlüssels in $d_h/2$ Paare. Das Paar $m$ dreht sich an Position $i$ um den Winkel $i\omega_m$ mit $\omega_m = 10000^{-2m/d_h}$ (:numref:`fig_rope-rotation`):

<!--en:5c2cf492-->
Die Werte bleiben ungedreht. Dadurch bleibt die Ausgabe der Schicht invariant, wenn wir alle Positionen gemeinsam verschieben. RoPE ist die übliche Positionskodierung in modernen Sprachmodellen mit offenen Gewichten.

<!--en:71e13ed8-->
![Rotary Position Embeddings drehen jedes zweidimensionale Merkmalspaar der Abfrage und des Schlüssels um einen Winkel, der proportional zur Position ist. Verschieben sich beide Positionen um denselben Betrag, hier um 3, so drehen sich beide Vektoren gemeinsam. Der Winkel zwischen ihnen ($2\theta + \varphi$ in beiden Teilbildern) und damit auch der Attention-Score bleiben gleich. Dabei gibt $\theta = \omega_m$ an, um welchen Winkel sich das Paar $m$ pro Positionsschritt dreht. Der Winkel zwischen der ungedrehten Abfrage und dem ungedrehten Schlüssel ist $\varphi$.](../img/mdl-attention-rope-rotation.svg)

<!--en:ebe96f42-->
Verschieben wir alle Positionen um einen beliebigen Versatz, so bleiben die RoPE-Scores bis auf Rundungsfehler in float32 gleich. Bei additiver sinusförmiger Kodierung ändern sich die Scores dagegen um $O(1)$. Dafür sind die gemischten Terme aus Inhalt und Position in :eqref:`eq_pos-four-terms` verantwortlich:

<!--en:8af2b07b-->
## Abstandsstrafe und Verzicht auf Positionskodierung

<!--en:34a5e94e-->
Zwei weitere Verfahren lassen die Vektoren für Token, Abfragen und Schlüssel unverändert:

<!--en:20be502d-->
* **ALiBi** (*Attention with Linear Biases*, :cite:`Press.Smith.Lewis.2022`) zieht direkt von den Scores der kausalen Attention in Kopf $h \in \{1, \ldots, H\}$ eine lineare Abstandsstrafe ab:

  $$
  \mathrm{score}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}} - m_h\,(i - j), \qquad m_h = 2^{-8h/H}.
  $$
  :eqlabel:`eq_alibi-def`

  Die geometrisch gestaffelten Steigungen $m_h$ (z. B. $1/4, 1/16, 1/64, 1/256$ für $H = 4$) geben jedem Kopf eine andere Reichweite für die Gewichtung zurückliegender Positionen. Eine feste Tabellengröße ist dafür überflüssig.

<!--en:e425cbf7-->
* **NoPE** (*No Positional Encoding*, :cite:`Kazemnejad.Padhi.Ramamurthy.ea.2023`) fügt keinerlei Positionssignal hinzu. Allein die kausale Maske bricht die Permutationssymmetrie, denn sie macht $i + 1$ Schlüssel an Position $i$ sichtbar.

<!--en:9d2f8ab2-->
## Extrapolation über die Trainingslänge hinaus

<!--en:8fa07ec4-->
Wir vergleichen die fünf Verfahren innerhalb der Trainingsverteilung und bei der Extrapolation auf längere Kontexte. Dazu trainieren wir bei einer Länge von $L = 128$ Byte und bestimmen die Perplexität bei den Längen $128$, $256$ und $512$.

<!--en:10838bd3-->
### Das Sprachmodell TinyCharLM

<!--en:a32739ea-->
`TinyCharLM` bettet die 256 Bytewerte in Vektoren der Breite $d = 128$ ein und wendet zwei kausale Self-Attention-Blöcke an ($H = 4$ Köpfe der Breite $d_h = 32$). Da die Blöcke weder ein MLP noch Normalisierungsschichten enthalten, geht die Position ausschließlich über `pos` ein. Mit denselben Gewichten wie bei der Einbettung projiziert das Modell anschließend zurück auf Byte-Logits :cite:`Press.Wolf.2017`.

<!--en:bc8f79b0-->
In `embed` skalieren wir die sinusförmige Tabelle mit $1/\sqrt{d}$. Damit haben ihre Einträge dieselbe Größenordnung wie die $\mathcal{N}(0, 0.02^2)$-verteilten Token-Einbettungen, ohne dass sich die Ausgabe-Logits aus den gekoppelten Gewichten ändern:

<!--en:ad338f01-->
In `attend` dreht RoPE die Abfragen und Schlüssel. ALiBi dagegen addiert den abstandsabhängigen Bias $-m_h(i - j)$ zu den Scores vor der Softmax-Funktion:

<!--en:060eb7bf-->
### Das Experiment

<!--en:336100b8-->
Wir trainieren jedes Verfahren mit drei verschiedenen Zufallsstartwerten jeweils $3{,}000$ AdamW-Schritte lang auf Fenstern von 128 Byte aus `d2l.WikiText103` (:numref:`subsec_tinylm`).

<!--en:9d0ab10b-->
Wir bestimmen die Validierungsperplexität für dieselben Validierungsbytes, die wir jeweils zu Fenstern der Länge $n \in \{128, 256, 512\}$ gruppieren. Als Referenz dient ein Byte-Bigrammmodell, dessen Perplexität sich in geschlossener Form berechnen lässt.

<!--en:f4880414-->
Um diese Zahlen einzuordnen, zählen wir im Trainingskorpus die Übergänge zwischen aufeinanderfolgenden Bytes. Aus den nach Laplace geglätteten Häufigkeiten berechnen wir die exakte Bigrammperplexität auf Byteebene für dieselben Validierungsfenster:

<!--en:cd2e6a1f-->
Bei der Trainingslänge ($n = 128$) erzielt RoPE die niedrigste Perplexität ($7.3$), gefolgt von gelernten Einbettungen ($9.6$), sinusförmigen Kodierungen ($9.7$–$9.8$) und ALiBi ($9.9$–$10.0$). Alle vier unterbieten die Byte-Bigrammreferenz ($11.6$). NoPE bleibt in diesem kleinen Modell mit zwei Blöcken auf Bigrammniveau.

<!--en:4bcb9820-->
Bei $n = 512$ schneiden die drei Verfahren deutlich schlechter ab, deren Darstellungen Werte außerhalb des Trainingsbereichs annehmen. Gelernte Einbettungen greifen auf untrainierte Zeilen zu ($128$–$511$). Die langsamen sinusförmigen Spalten nehmen unbekannte Werte an, und die niederfrequenten Paare von RoPE drehen sich in Winkelbereiche, die im Training nie vorkamen. Die Perplexität von ALiBi bleibt konstant bei $9.9$. Um zu verstehen, warum, messen wir für die vier Köpfe von ALiBi den Anteil des Attention-Gewichts, der auf Schlüssel im Abstand $\ge 128$ entfällt. Zum Vergleich maskieren wir alle Schlüssel im Abstand $\ge 128$:

<!--en:9876812d-->
Das überprüfen wir direkt mit `WindowedALiBi`. Diese Variante kopiert die trainierten ALiBi-Gewichte und blendet alle Schlüssel im Abstand $\ge 128$ vollständig aus (`window=train_len`), wenn wir das Modell bei Länge $T = 512$ auswerten:

<!--en:7a33f43c-->
Entfernen wir alle Schlüssel im Abstand $\ge 128$, bleibt die Perplexität von ALiBi gleich. Die Köpfe mit steiler Steigung ignorieren weit entfernte Schlüssel ohnehin. Die Köpfe mit flacher Steigung mitteln über einen weiten Bereich und gewichten jüngere Bytes stärker. Bytes, die mehr als 128 Positionen zurückliegen, bringen ihnen keinen Vorteil. Eine flache Extrapolationskurve zeigt also, dass weit entfernte Schlüssel das Verfahren nicht beeinträchtigen. Ob es ihnen über große Entfernungen hinweg Informationen entnimmt, lässt sich daraus allein nicht schließen.

<!--en:a0e5a8a0-->
### Kontexterweiterung bei einem trainierten Modell

<!--en:3430fec0-->
Warum versagt RoPE jenseits der Trainingslänge $L = 128$, und wie lässt sich das beheben? Bei einer Kopfbreite von $d_h = 32$ haben die $16$ Koordinatenpaare die Perioden $2\pi / \omega_m = 2\pi \cdot 10000^{m/16}$. Die ersten $6$ *schnellen* Paare haben Perioden $\le 112 < 128$ und durchlaufen bereits innerhalb des Trainingsfensters volle Umdrehungen. Die übrigen $10$ *langsamen* Paare haben Perioden von $199$ bis $35{,}000$ und drehen sich bei $n = 512$ in unbekannte Winkelbereiche.

<!--en:16186e4c-->
Die *Positionsinterpolation* :cite:`Chen.Wong.Chen.ea.2023` skaliert alle Winkel mit $128 / 512 = 1/4$. So bildet sie den neuen Kontextbereich auf das Trainingsintervall $[0, 128)$ ab. Ohne Fine-Tuning zerstört die Skalierung der schnellen Paare jedoch die lokalen Phasendifferenzen, anhand derer das Modell benachbarte Bytes unterscheidet. Wir lassen deshalb die schnellen Paare unverändert und skalieren *nur* die langsamen. Das ist die Grundidee von *YaRN* :cite:`Peng.Quesnelle.Fan.ea.2024` und der *NTK-bewussten Skalierung* (die die Basis gemäß $b \mapsto b\, s^{d_h/(d_h - 2)}$ erhöht). So machen wir den Anstieg der Perplexität bei der Extrapolation ohne zusätzliche Trainingskosten weitgehend rückgängig:

<!--en:ec44e8a7-->
Skalieren wir nur die langsamen Paare neu, sinkt die Perplexität bei Länge 512 von etwa $27$ auf etwa $9$, ohne dass wir ein einziges Gewicht aktualisieren. Ein kurzes Fine-Tuning bei der Ziellänge schließt die verbleibende Lücke (:exref:`ex_positional-information-position-interpolation`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:767d02ff-->
Unmaskierte Self-Attention ist permutationsäquivariant: Es gilt $f(\boldsymbol{\Pi} \mathbf{X}) = \boldsymbol{\Pi} f(\mathbf{X})$ für jede Permutation $\boldsymbol{\Pi}$ der Token. Additive Positionseinbettungen, ob gelernt oder sinusförmig, brechen diese Symmetrie an der Eingabe. Dabei entstehen jedoch gemischte Terme aus Inhalt und Position, die von den absoluten Indizes abhängen. Jenseits der Trainingslänge nimmt die Güte ab. Fordern wir, dass die Attention-Scores hinsichtlich der Position allein vom relativen Versatz $j - i$ abhängen, so ergeben sich orthogonale Drehungen in der Ebene für Abfragen und Schlüssel (*RoPE*). *ALiBi* berücksichtigt den Abstand dagegen durch einen kopfspezifischen linearen Strafterm, indem es $-m_h(i - j)$ zum Score addiert. Jenseits des Trainingsfensters verliert RoPE gerade bei den niederfrequenten Koordinatenpaaren an Güte, deren Wellenlänge die Trainingslänge übersteigt. Interpolieren wir gezielt diese langsamen Paare und lassen die hochfrequenten Paare unverändert, erreicht das Modell auch bei langem Kontext wieder eine hohe Güte.

<!--en:2294243a-->
## Übungen

<!--en:f3b22fbe-->
1. **Verschiedene Positionen, verschiedene Codes.** Es sei $\mathbf{p}_i$ die Zeile $i$ der sinusförmigen Tabelle aus :eqref:`eq_sinusoidal-def` mit der Breite $d$ und den Frequenzen $\omega_m = 1/10000^{2m/d}$.
    1. Zeigen Sie, dass folgende Gleichung gilt:
       $$\mathbf{p}_i^\top \mathbf{p}_{i+\delta} = \sum_{m=0}^{d/2-1} \cos(\delta\,\omega_m),$$
       Leiten Sie daraus $\|\mathbf{p}_i - \mathbf{p}_{i+\delta}\|^2$ ab und folgern Sie, dass beide Größen allein vom Versatz $\delta$ abhängen.
    1. Folgern Sie, dass $\mathbf{p}_i = \mathbf{p}_{i'}$ nur dann gilt, wenn $(i - i')\,\omega_m$ ein Vielfaches von $2\pi$ für jedes $m$ ist. Schließen Sie daraus, dass zwei verschiedene ganzzahlige Positionen nie dieselben Zeileneinträge haben, unabhängig vom Wert von `max_len`.
    1. Das schnellste Paar hat die Periode $2\pi$. Es wiederholt sich also bei ganzzahligen Verschiebungen nahe einem Vielfachen von $2\pi$ fast genau. Finden Sie den positiven ganzzahligen Versatz $\delta < 512$, bei dem sein Winkel einem Vielfachen von $2\pi$ am nächsten kommt, und geben Sie den Restwinkel an. Vergleichen Sie für $d = 32$ den Wert von $\|\mathbf{p}_i - \mathbf{p}_{i+\delta}\|$ bei diesem Versatz mit dem Wert bei $\delta = 1$. Welche Paare machen die beiden Positionen trotz der beinahe identischen Werte des schnellsten Paares unterscheidbar? Stellen Sie den Zusammenhang mit dem Binärzählen aus :numref:`subsec_pos-sinusoidal` her.
   :exlabel:`ex_positional-information-distinct-positions-distinct-codes`

<!--en:0de0f84e-->
1. **Additive Codes im Score.** Bei der additiven Kodierung aus :numref:`subsec_pos-sinusoidal` berechnet ein Kopf der ersten Schicht den Score für die Eingaben an den Positionen $i$ und $j$ aus den vier Termen in :eqref:`eq_pos-four-terms` mit $\mathbf{A} = \mathbf{W}_q \mathbf{W}_k^\top / \sqrt{d_h}$.
    1. Es sei $\mathbf{J}$ die Drehung der Ebene um $\pi/2$. Zeigen Sie mithilfe von :eqref:`eq_sinusoidal-rotation`, dass der rein positionsabhängige Term $\mathbf{p}_i^\top \mathbf{A}\, \mathbf{p}_j$ nur von $j - i$ abhängt, wenn $\mathbf{A}$ blockdiagonal ist und für jedes Frequenzpaar einen $2 \times 2$-Block der Form $a_m \mathbf{I} + b_m \mathbf{J}$ enthält. In :exref:`ex_positional-information-distinct-positions-distinct-codes` geht es um den Fall $\mathbf{A} = \mathbf{I}$. Zeigen Sie anschließend, dass ein einziger von null verschiedener Eintrag von $\mathbf{A}$, der zwei verschiedene Frequenzpaare koppelt, diese Eigenschaft zerstört.
    1. Bestimmen Sie, wann der gemischte Term aus Inhalt und Position $\mathbf{x}_i^\top \mathbf{A}\, \mathbf{p}_j$ bei jeder Verschiebung von $j$ unverändert bleibt. Erklären Sie damit, warum sich der Score der additiven Kodierung beim Verschiebungstest in :numref:`subsec_pos-rope` ändert und warum RoPE keinen gemischten Term aus Inhalt und Position enthält.
   :exlabel:`ex_positional-information-additive-codes`

<!--en:b50c4679-->
1. [code] **Wo die Extrapolation scheitert.** Die Perplexitäten in :numref:`subsec_pos-experiment` für die Kontextlänge 512 beruhen auf dem mittleren Verlust über alle Positionen jeder Sequenz.
    1. Prüfen Sie für jedes der fünf Modelle numerisch, ob die Logits an den Positionen 0 bis 127 einer Sequenz der Länge 512 bis auf Gleitkomma-Rundungsfehler mit den Logits übereinstimmen, die sich allein aus ihren ersten 128 Token ergeben. Welche beiden Eigenschaften von `TinyCharLM` garantieren dies für jedes Verfahren?
    1. Berechnen Sie für die fünf Modelle den Validierungsverlust an jeder Position der Sequenzen der Länge 512. Mitteln Sie ihn über Fenster von je 16 Positionen und tragen Sie ihn gegen die Position auf. Wo verlässt die Kurve bei jedem Verfahren das Niveau, das sie unterhalb von Position 128 hält? Steigt der Verlust an Position 128 sprunghaft oder nimmt er allmählich zu?
    1. Beschränken Sie jede Abfrage von `models['rope']` auf die 128 jüngsten Schlüssel einschließlich des Schlüssels an der eigenen Position. Überschreiben Sie dazu `attend` in einer Unterklasse und begrenzen Sie die kausale Maske auf ein Band. Kopieren Sie die trainierten Gewichte in das neue Modell und wiederholen Sie die Messung. Vergleichen Sie die Kurve mit Fenster oberhalb von Position 128 mit der Kurve ohne Fenster unterhalb dieser Position. Erklären Verschiebungen um mehr als 127 Positionen, warum RoPE bei Kontextlänge 512 an Güte verliert?
   :exlabel:`ex_positional-information-where-extrapolation-fails`

<!--en:dcdade14-->
1. **Bias-Terme, die die Symmetrie erhalten.** ● Nehmen Sie an, zur Score-Matrix aus dem Beweis kommt vor der Softmax-Funktion eine feste Matrix $\mathbf{B} \in \mathbb{R}^{n \times n}$ hinzu. Mit der Score-Matrix $\mathbf{S}$ berechnet die Self-Attention-Schicht (:numref:`sec_multihead-attention`) also $\mathrm{softmax}(\mathbf{S} + \mathbf{B})\, \mathbf{X}\mathbf{W}_v$.
    1. Zeigen Sie, dass die Proposition weiterhin für jede Permutationsmatrix $\boldsymbol{\Pi}$ mit $\boldsymbol{\Pi}^\top \mathbf{B}
       \boldsymbol{\Pi} = \mathbf{B}$ gilt.
    1. Bestimmen Sie alle Matrizen $\mathbf{B}$, die diese Bedingung für jede Permutation erfüllen. Beschreiben Sie, wie jede dieser Matrizen die Attention-Gewichte verändert.
    1. Ohne kausale Maske lässt sich der ALiBi-Bias aus :eqref:`eq_alibi-def` symmetrisch zu $B_{ij} = -m_h\,|i - j|$ erweitern. Bestimmen Sie die Permutationen, die diese Matrix unverändert lassen. Was folgt daraus für eine unmaskierte Schicht, die diesen Bias als einziges Positionssignal erhält? Erklären Sie, warum das kausale ALiBi aus :eqref:`eq_alibi-def` diese Symmetrie nicht aufweist.

<!--en:2d2411b9-->
1. **Bevorzugter Versatz bei RoPE.** ● Zerlegen Sie eine Abfrage $\mathbf{q}$ und einen Schlüssel $\mathbf{k}$ in die $d_h/2$ zweidimensionalen Paare $\mathbf{q}^{(m)},
   \mathbf{k}^{(m)} \in \mathbb{R}^2$, auf die :eqref:`eq_rope-def` wirkt. Es sei $\delta = j - i$ der Versatz der Schlüsselposition $j$ gegenüber der Abfrageposition $i$.
    1. Zeigen Sie, dass der RoPE-Score aus :eqref:`eq_rope-goal` durch folgende Summe gegeben ist:
       $$\sum_{m=0}^{d_h/2-1} \|\mathbf{q}^{(m)}\|\, \|\mathbf{k}^{(m)}\| \cos(\delta\,\omega_m - \psi_m),$$
       Dabei ist $\psi_m$ der Winkel von $\mathbf{k}^{(m)}$ nach $\mathbf{q}^{(m)}$ (der Winkel $\varphi$ aus :numref:`fig_rope-rotation` mit umgekehrtem Vorzeichen).
    1. Wählen Sie die Winkel $\psi_m$ so, dass jedes Paar beim Versatz $\delta = -1$, also beim vorherigen Token, seinen maximalen Beitrag liefert. Alle Produkte $\|\mathbf{q}^{(m)}\|\, \|\mathbf{k}^{(m)}\|$ sollen denselben Wert $c$ haben. Zeigen Sie, dass der Score als Funktion von $\delta$ gleich dem $c$-Fachen des verschobenen Skalarprodukts $\mathbf{p}_i^\top
       \mathbf{p}_{i+\delta}$ aus :exref:`ex_positional-information-distinct-positions-distinct-codes` für eine sinusförmige Tabelle der Breite $d_h$ ist.
    1. Betrachten Sie $d_h = 32$ und die Abfrage an Position 127, die auf die Positionen mit dem Versatz $\delta = -127, \ldots, 0$ zugreift. Setzen Sie die Attention-Logits gleich diesen Scores und beziehen Sie den Faktor $1/\sqrt{d_h}$ in $c$ ein. Bestimmen Sie den Wert von $c$, bei dem auf den Versatz $-1$ die Hälfte des Softmax-Gewichts entfällt. Welche anderen Verschiebungen konkurrieren am stärksten mit ihm? Welche Paare bestimmen die Differenz zwischen dem Score bei $-1$ und den Scores an den Nachbarpositionen?
   :exlabel:`ex_positional-information-offset-preference-rope`

<!--en:317a171f-->
1. [extended] **Positionsinterpolation.** ● Die letzte Codezelle in :numref:`subsec_pos-context-extension` skaliert die Winkel der trainierten RoPE-Modelle ohne weiteres Training neu. In dieser Aufgabe erklären Sie das Ergebnis, ergänzen den Basiswechsel der NTK-bewussten Skalierung und schließen das Verfahren mit einem Fine-Tuning ab.
    1. Betrachten Sie ein Paar, das bei den im Training vorkommenden Verschiebungen von 0 bis 127 eine volle Umdrehung durchläuft. Zeigen Sie, dass es dabei jeden Winkel einer größeren Verschiebung bis auf eine Abweichung von höchstens $\omega_m/2$ modulo $2\pi$ erreicht. Berechnen Sie wie in :exref:`ex_positional-information-offset-preference-rope` die Differenz zwischen dem Score beim Versatz $-1$ und den Scores an den Nachbarpositionen für $d_h = 32$. Vergleichen Sie drei Fälle: vor der Skalierung, nach einheitlicher Skalierung mit $128/512$ und nach Skalierung nur der Paare, deren Periode 128 Positionen übersteigt. Setzen Sie das Ergebnis in Beziehung zu den Perplexitäten bei Kontextlänge 128 aus der Codezelle.
    1. Zeigen Sie, dass der Wechsel zur Basis $b' = b\, s^{d_h/(d_h-2)}$ die Frequenz $\omega_0$ unverändert lässt und $\omega_{d_h/2-1}$ durch $s$ teilt. Setzen Sie `rope_base` von `models['rope']` für $s = 4$ auf diese Basis. Werten Sie das Modell ohne weiteres Training bei den Längen 128, 256 und 512 aus und vergleichen Sie die Ergebnisse mit denen der beiden Skalierungen aus der Codezelle.
    1. Stimmen Sie das Modell für jede der beiden Skalierungen aus der Codezelle einige hundert Schritte lang bei Kontextlänge 512 auf `d2l.WikiText103(batch_size=16, num_steps=512)` fein ab. Beginnen Sie jeweils mit einer Kopie des trainierten Modells `models['rope']`. Werten Sie es erneut bei den Längen 128, 256 und 512 aus. Vergleichen Sie die Ergebnisse mit denen des unskalierten Modells und des ALiBi-Modells aus dem Experiment in :numref:`subsec_pos-experiment` bei denselben Längen.
   :exlabel:`ex_positional-information-position-interpolation`

<!--en:b1c6e1ed-->
[Diskussion](https://d2l.discourse.group/t/36771)

<!--en:5b235496-->
Positionsinformation mit sinusförmigen und gelernten Kodierungen, RoPE und ALiBi<br>**Permutationsäquivarianz · absolute Einbettungen · Rotary Position Embeddings (RoPE) · ALiBi und NoPE · Längenextrapolation**

<!--en:7169724b-->
::: {.slide title="Warum Self-Attention allein die Wortreihenfolge nicht erkennt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:04bc160b-->
- **Wortreihenfolge:** *„Hund beißt Mann“* und *„Mann beißt Hund“* enthalten dieselben drei Wörter. Unmaskierte Self-Attention liefert exakt dieselben Ausgabevektoren in vertauschter Reihenfolge.

<!--en:7ffa190c-->
- **Extrapolation auf längere Sequenzen:** Modelle wie GPT-2 small lernen eine feste Tabelle mit $1{,}024$ Positionsvektoren. Für Token $1{,}025$ gibt es keine trainierte Darstellung.

<!--en:75d51c34-->
- **Ziel:** Positionsinformation so einbringen, dass der relative Abstand erhalten bleibt und das Verfahren über die Kontextlänge des Trainings hinaus verallgemeinert

<!--en:5339050d-->
::: {.slide title="Satz: Äquivarianz der unmaskierten Self-Attention"}

<!--en:2eb43f08-->
[Permutationssymmetrie]{.kicker}

<!--en:8d858e88-->
- **Permutationsäquivarianz:** Für jede $n \times n$-Permutationsmatrix $\boldsymbol{\Pi}$ ($\boldsymbol{\Pi}^\top \boldsymbol{\Pi} = \mathbf{I}$) gilt:
  $$f(\boldsymbol{\Pi}\mathbf{X}) = \boldsymbol{\Pi} f(\mathbf{X})$$

<!--en:4d6468e6-->
- **Beweis in vier Schritten:** Zeilenweise Projektionen vertauschen mit $\boldsymbol{\Pi}$. Die Score-Matrix geht in $\boldsymbol{\Pi}\mathbf{S}\boldsymbol{\Pi}^\top$ über. Die zeilenweise Softmax-Funktion erhält die Form $\boldsymbol{\Pi}\mathbf{A}\boldsymbol{\Pi}^\top$. Bei der Multiplikation mit $\boldsymbol{\Pi}\mathbf{V}$ fallen die beiden benachbarten Permutationsmatrizen wegen $\boldsymbol{\Pi}^\top\boldsymbol{\Pi} = \mathbf{I}$ weg.

<!--en:06011a8a-->
- **Unabhängigkeit von der Tiefe:** Beliebig viele unmaskierte Attention-Schichten und MLP-Schichten hintereinander erhalten die Permutationsäquivarianz.

<!--en:916900c4-->
::: {.slide title="Numerische Prüfung der Permutationsäquivarianz"}

<!--en:7484a6ca-->
[Empirische Prüfung]{.kicker}

<!--en:5b455e43-->
Durchlaufen $8$ zufällige Token-Vektoren vor und nach einer zufälligen Permutation eine Schicht mit $4$ Köpfen vom Typ `MultiHeadAttention`, stimmen die entsprechend umgeordneten Ausgaben bis auf `float32`-Rundungsfehler überein:

<!--en:0a33d6a6-->
Eine **kausale Maske** bricht die Permutationssymmetrie selbst ohne Positionsvektoren (**NoPE**), denn Abfrage $i$ mittelt über $i + 1$ Positionen einschließlich der eigenen.

<!--en:be01a357-->
::: {.slide title="Absolute Positionseinbettungen mit gelernten Tabellen"}

<!--en:542058a6-->
[Absolute Kodierungen]{.kicker}

<!--en:5b60b768-->
- **Symmetriebruch an der Eingabe:** Die Addition eines Positionsvektors $\mathbf{p}_i \in \mathbb{R}^d$ (Zeile $i$ von $\mathbf{P}$) zu jedem Token $\mathbf{x}_i$ bricht die Äquivarianz:
  $$\boldsymbol{\Pi}\mathbf{X} + \mathbf{P} \neq \boldsymbol{\Pi}(\mathbf{X} + \mathbf{P})$$

<!--en:832853f3-->
- **Gelernte Positionstabellen (BERT, GPT-2):** $\mathbf{P} \in \mathbb{R}^{L_{\max} \times d}$ als freie Parametermatrix behandeln

<!--en:d32a8900-->
- **Ausbleibender Gradient verhindert Extrapolation:** Beim Training mit Länge $L = 128$ erhalten die Zeilen $128, \ldots, L_{\max}-1$ den Gradienten null und enthalten weiterhin nur zufälliges Rauschen.

<!--en:7107bb2b-->
::: {.slide title="Sinusförmige Positionskodierungen bei verschiedenen Frequenzen"}

<!--en:385c3c56-->
[Deterministische absolute Codes]{.kicker}

<!--en:07d2de36-->
- Vaswani et al. (2017) füllen Zeile $i$ von $\mathbf{P} \in \mathbb{R}^{n \times d}$ mit Sinus- und Kosinuswerten zu $d/2$ geometrisch gestaffelten Frequenzen $\omega_m = 10000^{-2m/d}$:
  $$p_{i, 2m} = \sin(i\,\omega_m)$$
  $$p_{i, 2m+1} = \cos(i\,\omega_m)$$

<!--en:a76addc7-->
- **Geometrisch gestaffelte Wellenlängen:** Die Perioden $2\pi / \omega_m$ wachsen von $2\pi \approx 6.28$ Token ($m = 0$) in Richtung $2\pi \times 10^4$ Token (etwa $35{,}000$ bei $m = d/2 - 1$ für $d = 32$).

<!--en:a9bd3a1d-->
::: {.slide title="Sinusschwingungen als kontinuierlicher Binärzähler"}

<!--en:50843f62-->
[Mehrskalige Geometrie]{.kicker}

<!--en:5d4fefb7-->
- **Kontinuierlicher Binärzähler:** Beim Binärzählen wechselt das niedrigstwertige Bit bei jedem Schritt, das nächste alle $2$ Schritte, das übernächste alle $4$. Sinusförmige Spaltenpaare leisten dasselbe mit stetigen Bewegungen auf dem Einheitskreis.

<!--en:ca38b7ce-->
- **Inneres Produkt als Funktion des Versatzes:** Das Skalarprodukt zweier Zeilen der sinusförmigen Tabelle hängt nur von ihrem relativen Versatz $\delta$ ab:
  $$\mathbf{p}_i^\top \mathbf{p}_{i+\delta} = \sum_{m=0}^{d/2-1} \cos(\delta\,\omega_m)$$

<!--en:5769ecc9-->
::: {.slide title="Positionsverschiebungen als 2×2-Drehungen in der Ebene"}

<!--en:1c761583-->
[Trigonometrische Verschiebungsformel]{.kicker}

<!--en:0920a24c-->
- **Linearer Verschiebungsoperator:** Nach den Additionstheoremen entspricht die Verschiebung $i \mapsto i + \delta$ einer $2 \times 2$-Drehung, die *nur* von $\delta$ abhängt:
  $$\begin{bmatrix} \cos(\delta \omega_m) & \sin(\delta \omega_m) \\ -\sin(\delta \omega_m) & \cos(\delta \omega_m) \end{bmatrix} \begin{bmatrix} \sin(i \omega_m) \\ \cos(i \omega_m) \end{bmatrix} = \begin{bmatrix} p_{i+\delta, 2m} \\ p_{i+\delta, 2m+1} \end{bmatrix}$$

<!--en:0a82d622-->
- **Relative Verschiebungen in $\mathbf{P}$:** Für jedes Frequenzpaar $m$ ergibt sich der Code an Position $i + \delta$ durch eine feste lineare Drehung des Codes an Position $i$.

<!--en:b6e30302-->
::: {.slide title="Warum additive Positionsvektoren den Score mitbestimmen"}

<!--en:80cb6803-->
[Zerlegung in vier Terme]{.kicker}

<!--en:e5f0f791-->
- **Ausmultiplizieren von $(\mathbf{x}_i + \mathbf{p}_i)^\top \mathbf{A}\,(\mathbf{x}_j + \mathbf{p}_j)$** mit $\mathbf{A} = \mathbf{W}_q \mathbf{W}_k^\top / \sqrt{d_h}$ ergibt vier bilineare Terme.

<!--en:4883efba-->
- **Kopplung an absolute Positionen:** Die gemischten Terme $\mathbf{x}_i^\top \mathbf{A}\,\mathbf{p}_j$ und $\mathbf{p}_i^\top \mathbf{A}\,\mathbf{x}_j$ koppeln den Token-Inhalt direkt an *absolute* Positionsvektoren. Sie ändern sich, wenn sich $i$ und $j$ gemeinsam verschieben.

<!--en:5edffb21-->
- **Mischung verschiedener Frequenzen:** Selbst $\mathbf{p}_i^\top \mathbf{A}\,\mathbf{p}_j$ verliert die Verschiebungsinvarianz, wenn $\mathbf{A}$ verschiedene Frequenzpaare mischt.

<!--en:3d021a65-->
::: {.slide title="Herleitung von RoPE aus der Verschiebungsinvarianz"}

<!--en:973e861d-->
[Rotary Position Embeddings (RoPE)]{.kicker}

<!--en:07b02b83-->
Statt $\mathbf{p}_i$ zu $\mathbf{x}_i$ zu addieren, transformieren **Rotary Position Embeddings** (Su et al., 2021) Abfragen und Schlüssel *multiplikativ* mit Matrizen $\mathbf{R}_i$. Dadurch hängt jedes innere Produkt hinsichtlich der Position nur von $j - i$ ab:

<!--en:766f7427-->
- **1. Identität im Nullpunkt ($i = j = 0$):** Es gilt $\mathbf{R}_0^\top \mathbf{R}_0 = \mathbf{R}_0$. Bei vollem Rang folgt $\mathbf{R}_0 = \mathbf{I}$.

<!--en:73702c1b-->
- **2. Orthogonalität ($i = j$):** Es gilt $\mathbf{R}_i^\top \mathbf{R}_i = \mathbf{R}_0 = \mathbf{I}$. Somit ist jede Matrix $\mathbf{R}_i$ orthogonal und erhält die Normen von Abfrage und Schlüssel.

<!--en:220abbf2-->
- **3. Gruppenhomomorphismus:** Aus $\mathbf{R}_{-i} = \mathbf{R}_i^\top$ und $\mathbf{R}_{i+j} = \mathbf{R}_i \mathbf{R}_j$ folgt $\mathbf{R}_i = \mathbf{R}_1^i$. Alle Matrizen sind Potenzen eines orthogonalen Erzeugers $\mathbf{R}_1$, der bei RoPE in $2 \times 2$-Blöcke für Drehungen in der Ebene zerfällt.

<!--en:adbef69c-->
::: {.slide title="RoPE mit blockdiagonalen 2×2-Drehmatrizen"}

<!--en:1f4fdb09-->
[Geometrie von RoPE]{.kicker}

<!--en:03977158-->
- $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_h}$ in $d_h/2$ Koordinatenpaare $(2m, 2m+1)$ aufteilen. Paar $m$ an Position $i$ um den Winkel $i\,\omega_m$ mit $\omega_m = 10000^{-2m/d_h}$ drehen:
  $$R(i\omega_m) = \begin{bmatrix} \cos(i\omega_m) & -\sin(i\omega_m) \\ \sin(i\omega_m) & \cos(i\omega_m) \end{bmatrix}$$

<!--en:cdc192dd-->
- **Verschiebungsinvarianz:** Eine Verschiebung beider Positionen um $+3$ dreht $\mathbf{q}$ und $\mathbf{k}$ um exakt denselben Winkel $3\omega_m$. Der Winkel zwischen ihnen und ihr Skalarprodukt bleiben gleich.

<!--en:efdf3e51-->
- **Werte bleiben ungedreht:** Nur Abfragen und Schlüssel drehen sich. Die Wertevektoren bleiben verschiebungsinvariant.

<!--en:7456c787-->
![RoPE dreht jedes zweidimensionale Merkmalspaar um einen Winkel proportional zur Position. Bei Verschiebungen bleiben die relativen Winkel erhalten.](../img/mdl-attention-rope-rotation.svg){.nostretch width=100%}

<!--en:fce52cc3-->
::: {.slide title="Implementierung von RoPE und Prüfung der Verschiebungsinvarianz"}

<!--en:8e3495db-->
[Code und Überprüfung]{.kicker}

<!--en:a2f437e0-->
- `rope` trennt Koordinaten mit geraden (`0::2`) und ungeraden (`1::2`) Indizes. Für die $2 \times 2$-Drehung wandelt die Funktion die `float32`-Winkel in `x.dtype` um. Anschließend fügt sie die Koordinaten der Paare wieder abwechselnd zusammen.

<!--en:60825335-->
- Eine Verschiebung aller $32$ Positionen um $1$, $17$ oder $480$ ändert die RoPE-Scores nur um den `float32`-Rundungsfehler ($\sim 10^{-6}$ bis $10^{-5}$). Die Scores der additiven sinusförmigen Kodierung ändern sich dagegen um $2.7$.

<!--en:12c12710-->
::: {.slide title="Abstandsstrafe (ALiBi) und kausale Maske allein (NoPE)"}

<!--en:a58c6a2f-->
[Weitere Verfahren]{.kicker}

<!--en:97ca71f3-->
- **ALiBi (Press et al., 2022):** Eine lineare Abstandsstrafe mindert die kausalen Scores vor der Softmax-Funktion in Kopf $h$:
  $$\mathrm{score}_{ij}^{(h)} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}} - m_h\,(i - j), \quad m_h = 2^{-8h/H}$$

<!--en:f7b00ff2-->
- **Steigungen mit unterschiedlicher Reichweite:** Köpfe mit steiler Steigung ($m_1 = 1/4$) gewichten vor allem nahe Positionen. Köpfe mit flacher Steigung ($m_4 = 1/256$) beziehen einen weiten Bereich ein.

<!--en:6642e74d-->
- **NoPE (Kazemnejad et al., 2023):** Allein die kausale Maske bricht die Symmetrie. Sie macht $i + 1$ Schlüssel an Position $i$ sichtbar.

<!--en:d2cd6ea1-->
::: {.slide title="Fünf Verfahren für Positionsinformation in TinyCharLM"}

<!--en:d1dbc8f7-->
[Modellarchitektur im Experiment]{.kicker}

<!--en:02e9aeb1-->
- **Vergleichsmodell (`TinyCharLM`):** Byte-Sprachmodell mit zwei reinen Attention-Blöcken ($d = 128, H = 4, d_h = 32$, gekoppelte Ausgabegewichte), trainiert auf WikiText-103 bei $n = 128$

<!--en:206e2518-->
- **Verfahren an der Eingabe (`learned`, `sinusoidal`):** Einmalige Addition der Positionsvektoren in `embed(X)`

<!--en:4bb9935a-->
- **Verfahren in der Attention-Schicht (`rope`, `alibi`, `none`):** Die Einbettungen bleiben unverändert. Das Verfahren greift in jedem Block beim Aufruf von `attend`.

<!--en:f19504de-->
::: {.slide title="Implementierung von embed, alibi und attend"}

<!--en:0b473ed6-->
[Implementierung von TinyCharLM]{.kicker}

<!--en:285cd9c6-->
RoPE dreht `q` und `k` in jedem Block. ALiBi addiert den Bias `-m_h * (i - j)` zu den Scores:

<!--en:aebe575f-->
::: {.slide title="Training bei Länge 128, Test bei 128, 256 und 512"}

<!--en:4934aaaa-->
[Vergleich der Extrapolation]{.kicker}

<!--en:4e1116ab-->
Für jedes Verfahren laufen $3$ Trainingsläufe mit verschiedenen Zufallsstartwerten über jeweils $3{,}000$ AdamW-Schritte bei Kontextlänge $n = 128$. Für die Validierungsperplexität pro Byte dienen dieselben WikiText-103-Bytes, neu gruppiert in Fenster der Länge $n \in \{128, 256, 512\}$:

<!--en:d504b150-->
::: {.slide title="Perplexität innerhalb und außerhalb des Trainingsbereichs"}

<!--en:de1bc839-->
[Warum die Extrapolation scheitert]{.kicker}

<!--en:f8f859e3-->
Dank kausaler Maskierung ergeben sich an den Positionen $0 \ldots 127$ unabhängig von der Fensterlänge dieselben Logits. Jeder Anstieg der Perplexität bei $n \in \{256, 512\}$ stammt daher von Positionen $\ge 128$:

<!--en:0f2a51f8-->
- **RoPE liegt im Trainingsbereich vorn ($7.3$ bei $n = 128$), verliert aber bei $512$ ($\approx 27$):** Drehungen allein nach dem Versatz liefern im Trainingsfenster den stärksten induktiven Bias. Bei einem Versatz $> 127$ drehen sich niederfrequente Paare jedoch in Winkelsektoren, die im Training nie vorkamen.

<!--en:041727e7-->
- **Gelernt ($9.6 \to 20$–$25$) und sinusförmig ($\approx 9.7 \to \approx 15.5$):** Gelernte Einbettungen greifen auf zufällig initialisierte, untrainierte Zeilen $128 \ldots 511$ zu. Die langsamen sinusförmigen Spalten nehmen neue Wertekombinationen an, die mit den gemischten Termen aus Inhalt und Position in Konflikt geraten.

<!--en:b7f585cf-->
- **NoPE bleibt auf dem Niveau der Byte-Bigrammreferenz ($11.6$):** Ohne expliziten Positionscode stützt ein Modell mit zwei reinen Attention-Blöcken seine Vorhersagen fast ausschließlich auf das aktuelle Byte im Residualstrom.

<!--en:7b6c3b50-->
::: {.slide title="Bedeutet konstante Perplexität, dass das Modell weit zurückliegende Bytes nutzt?"}

<!--en:866a7db4-->
[ALiBi jenseits des Trainingsfensters]{.kicker}

<!--en:6ba9a296-->
Die Perplexität von ALiBi bleibt konstant bei $\approx 9.9$, von $n = 128$ bis $n = 512$. Gewinnt ALiBi aus Bytes, die $\ge 128$ Positionen zurückliegen, tatsächlich nützliche Information, oder ignoriert es sie?

<!--en:2fe4c0bc-->
**Diagnose:** Das vollständige Ausblenden aller Schlüssel im Abstand $\ge 128$ (`WindowedALiBi`) lässt die Perplexität von ALiBi bei Länge $512$ unverändert ($9.9$). Köpfe mit steiler Steigung ($1, 2$) vergeben das Gewicht $0.000$ für Abstände jenseits von $128$. Köpfe mit flacher Steigung ($3, 4$) mitteln über einen weiten Bereich und gewichten jüngere Positionen stärker. Eine flache Extrapolationskurve zeigt, dass das Verfahren entfernte Schlüssel *verkraftet*. Ob es daraus *Informationen abruft*, bleibt offen.

<!--en:b42514ca-->
::: {.slide title="Warum RoPE versagt: schnelle und langsame Frequenzpaare"}

<!--en:ef8b95ed-->
[Frequenzzerlegung]{.kicker}

<!--en:92dc3930-->
- **6 schnelle Paare ($m = 0 \ldots 5$, Perioden $\le 112 < 128$):** Sie durchlaufen volle $2\pi$-Drehungen innerhalb des Trainingsfensters von $128$ Byte. Jeder Winkel beim Versatz $400$ liegt modulo $2\pi$ bereits im überstrichenen Winkelbereich des Trainings.

<!--en:e41f760c-->
- **10 langsame Paare ($m = 6 \ldots 15$, Perioden von $199$ bis $35{,}000 > 128$):** Bei $n = 512$ erreichen sie unbekannte Winkelbereiche. Das verfälscht die Skalarprodukte von Abfragen und Schlüsseln.

<!--en:cdda6d8a-->
- **Selektive Skalierung:** Schnelle Paare beim Faktor $1.0$ belassen, um lokale Phasenunterschiede zwischen Bytes zu erhalten. Nur die langsamen Paare mit $128/512 = 0.25$ skalieren.

<!--en:821a05f7-->
::: {.slide title="Kontexterweiterung ohne weiteres Training durch Skalierung langsamer Paare"}

<!--en:880d147a-->
[Selektive Interpolation]{.kicker}

<!--en:4e52af0e-->
Die $6$ schnellen Paare bleiben unverändert (`scale = 1.0`). Die Multiplikation *nur* der $10$ langsamen Paare mit $128/512 = 0.25$ senkt die Perplexität bei Länge $512$ von $\approx 27$ auf $\approx 9$, **ohne ein einziges Modellgewicht zu aktualisieren**:

<!--en:d280c1fc-->
Eine einheitliche Skalierung aller Paare (`all pairs`) ohne Fine-Tuning zerstört bei jeder Kontextlänge die lokale Auflösung auf Byteebene (Perplexität $\sim 40\text{--}41$, schlechter als ein Bigrammmodell). Die selektive Skalierung der langsamen Paare erhält die lokalen Phasen und staucht nur die langsamen Winkel außerhalb der Trainingsverteilung.

<!--en:62ac33bb-->
::: {.slide title="Verfahren aus der Praxis: YaRN und NTK-bewusste Skalierung"}

<!--en:5ab26556-->
[Große Sprachmodelle mit langem Kontext]{.kicker}

<!--en:6fc3f2cc-->
Moderne Sprachmodelle mit offenen Gewichten erweitern den RoPE-Kontext von $4\text{K}$/$8\text{K}$ auf $128\text{K}+$ mit zwei eng verwandten Verfahren zur Frequenzskalierung:

<!--en:5704506c-->
- **NTK-bewusste Skalierung der Basis:** Für die Erweiterung des Kontexts wird die RoPE-Basis $b = 10000$ durch $b' = b \cdot s^{d_h / (d_h - 2)}$ ersetzt. Der Erweiterungsfaktor beträgt $s = L_{\textrm{new}} / L_{\textrm{train}}$.
  - Paar $m = 0$ behält die Frequenz $\omega_0 = (b')^0 = 1$ Radiant pro Token. Die lokale Auflösung bleibt erhalten.
  - Beim langsamsten Paar $m = (d_h - 2)/2$ sinkt die Frequenz um den Faktor $(b'/b)^{(d_h-2)/d_h} = s$. Dadurch passt der $s$-fach längere Kontext in den Winkelbereich des Trainings.

<!--en:6d49bda8-->
- **YaRN (Yet another RoPE extensioN, Peng et al., 2024):** Schnelle Frequenzen bleiben unverändert ($\text{scale} = 1$), langsame werden linear interpoliert ($\text{scale} = 1/s$). Eine glatte Rampe verbindet die Zwischenfrequenzen. Hinzu kommen eine Skalierung der Softmax-Temperatur und ein kurzes Fine-Tuning.

<!--en:2c379d9a-->
::: {.slide title="Verständnisfragen zu Symmetrie, RoPE und Extrapolation"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:fbc3c5d2-->
Fragen zu Positionskodierungen und Kontexterweiterung:

<!--en:b47fd322-->
1. **Relativer Winkel bei RoPE:** Paar $m = 0$ dreht sich um $\omega_0 = 1\text{ rad/token}$. Abfrage und Schlüssel haben für Paar $0$ identische ungedrehte Vektoren. Welcher Winkel liegt nach der Drehung zwischen ihren Vektoren für Paar $0$, wenn die Abfrage an Position $i = 3$ und der Schlüssel an Position $j = 5$ steht? Wie groß ist der Winkel bei $i = 1{,}003$ und $j = 1{,}005$?

<!--en:0dc7c8f7-->
2. **Additive gemischte Terme:** Welche Terme in $(\mathbf{x}_i + \mathbf{p}_i)^\top \mathbf{A}\,(\mathbf{x}_j + \mathbf{p}_j)$ ändern sich bei einer Verschiebung beider Token um $+17$ Positionen ($i \mapsto i+17$, $j \mapsto j+17$)?

<!--en:463de279-->
3. **Berechnung der NTK-bewussten Basis:** Ein Kopf der Breite $d_h = 64$ soll mittels NTK-bewusster Skalierung von $4\text{K}$ auf $16\text{K}$ Token erweitert werden ($s = 4$). Welchen Wert ergibt die ursprüngliche Basis $b = 10{,}000$ für die neue Basis $b'$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:2f75b359-->
1. **$2\text{ radians}$ in beiden Fällen** ($(5 - 3)\omega_0 = (1005 - 1003)\omega_0 = 2$). Nur der relative Versatz $j - i$ beeinflusst den Winkel und das Skalarprodukt.

<!--en:464c366f-->
2. **Die gemischten Terme aus Inhalt und Position** $\mathbf{x}_i^\top\mathbf{A}\mathbf{p}_j$ und $\mathbf{p}_i^\top\mathbf{A}\mathbf{x}_j$ (sowie $\mathbf{p}_i^\top\mathbf{A}\mathbf{p}_j$, es sei denn, $\mathbf{A}$ ist blockdiagonal und besteht aus skalierten $2 \times 2$-Drehmatrizen).

<!--en:88789151-->
3. **$b' = 10{,}000 \times 4^{64/62} \approx 41{,}826$**. Dabei bleibt $\omega_0 = 1$ unverändert, während die niedrigste Frequenz $\omega_{31}$ durch $4$ geteilt wird.

<!--en:963d3627-->
::: {.slide title="Zusammenfassung: Positionsinformation in der Attention"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:037bce8a-->
| Verfahren | Einbindung der Position | Invariant bei gemeinsamem Versatz? | Perplexität im Trainingsbereich ($128$) | Perplexität bei Extrapolation ($512$) | Stärke und Grenze |

<!--en:771888a7-->
| **NoPE** | Nur kausale Maske $j \le i$ | Entfällt | $\approx 11.7$ | $\approx 11.8$ | Kein Zusatzaufwand, geringe Positionsauflösung in flachen Modellen |

<!--en:037bbfdf-->
| **Gelernt** | $\mathbf{x}_i + \mathbf{p}_i$ (trainierbare Tabelle) | Nein | $9.6$ | $20$–$25$ | Einfach und flexibel im Trainingsbereich. Zeilen $\ge L_{\textrm{train}}$ erhalten den Gradienten null |

<!--en:aa6ceb2c-->
| **Sinusförmig** | $\mathbf{x}_i + \mathbf{p}_i/\sqrt{d}$ (feste Sinusschwingungen) | Nein (gemischte Terme) | $\approx 9.7$ | $\approx 15.5$ | Parameterfreier Zähler. Additive gemischte Terme ändern sich bei Verschiebungen |

<!--en:c5d3ea58-->
| **ALiBi** | $s_{ij} - m_h(i - j)$ | Ja | $\approx 10$ | **$9.9$** | Konstante Perplexität bei Extrapolation durch Bevorzugung jüngerer Positionen. Entfernte Schlüssel erhalten weniger Gewicht |

<!--en:e797bed4-->
| **RoPE** | $(\mathbf{R}_i\mathbf{q}_i)^\top(\mathbf{R}_j\mathbf{k}_j) = \mathbf{q}_i^\top\mathbf{R}_{j-i}\mathbf{k}_j$ | **Ja (exakt)** | **$7.3$** | $\approx 27 \to \mathbf{\approx 9}^*$ | Beste Güte im Trainingsbereich. $*$Skalierung der langsamen Paare oder YaRN-Skalierung verbessert die Güte bei langen Kontexten |
