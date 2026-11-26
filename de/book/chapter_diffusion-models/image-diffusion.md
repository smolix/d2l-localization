<!--en:27f116f9-->
# Diffusion für Bilder

<!--en:9cb27159-->
Die Geschwindigkeitsregression, die Wahrscheinlichkeitsfluss-ODE (engl. probability-flow ODE) und die ancestrale Kette aus :numref:`sec_diffusion-gaussian-paths` und :numref:`sec_diffusion-integration` gelten in jeder Dimension $d$. Beim Übergang von der Ebene ($d = 2$) zu den $28 \times 28$ Pixel großen Bildern von Fashion-MNIST ($d = 784$) kommen vier praktische Komponenten hinzu:

<!--en:25684d19-->
1. **Architektur:** Ein U-Net mit mehreren Auflösungen :cite:`Ronneberger.Fischer.Brox.2015,ho2020denoising` erfasst am groben $7 \times 7$-Flaschenhals (engl. bottleneck) die globale Bildstruktur. Skip-Verbindungen in voller Auflösung erhalten feine räumliche Details. Zeit- und Klasseneinbettungen (engl. embeddings) konditionieren jeden Residualblock.

<!--en:1503cba9-->
2. **Trainingsrezept:** Wir bilden einen exponentiellen gleitenden Mittelwert (engl. exponential moving average, EMA) der Modellparameter. Zufälliges *Label-Dropout* beim Training sorgt dafür, dass ein einziges Netz sowohl bedingte als auch unbedingte Geschwindigkeitsfelder lernt.

<!--en:b70055c9-->
3. **Auswertung:** Wir messen den Fréchet-Abstand (engl. Fréchet distance, FD) :cite:`Heusel.Ramsauer.Unterthiner.ea.2017` zwischen den Merkmalsverteilungen realer und generierter Bilder. Die Merkmale liefert ein Klassifikator für diese Bilddomäne. Als Maßstab dient der Referenzwert des Abstands bei endlicher Stichprobengröße.

<!--en:579cb42d-->
4. **Guidance:** Die *Classifier-Free Guidance* (CFG) :cite:`Dhariwal.Nichol.2021,Ho.Salimans.2022` extrapoliert entlang der Differenz zwischen bedingter und unbedingter Vorhersage. Dadurch werden die Samples auf Kosten der Sample-Vielfalt typischer für ihre Klasse.

<!--en:4b6d1696-->
Wir implementieren alle vier Komponenten und untersuchen sie auf Fashion-MNIST mit einem U-Net mit 1.2 Millionen Parametern, das sich auf einer einzigen GPU trainieren lässt.

<!--en:eaad75ed-->
## Das U-Net zur Entrauschung

<!--en:bddf861d-->
### Skip-Verbindungen, Auflösungen und Zeiteinbettung

<!--en:d786d18b-->
Das Netz aus :numref:`sec_diffusion-gaussian-paths`, `TimeMLP`, erhielt einen Punkt und einen Zeitpunkt und gab einen Punkt derselben Dimension zurück. Ein Netz für Bilder folgt demselben Prinzip. Es erhält ein verrauschtes Bild $\mathbf{x}_t$ und den Zeitpunkt $t$ und liefert eine Vorhersage derselben Form, hier die Geschwindigkeit. Faltungen (engl. convolutions) eignen sich für Bilder. Bei einer Folge von $3 \times 3$-Faltungen in voller Auflösung hängt jedoch jedes Ausgabepixel nur von einem kleinen Ausschnitt der Eingabe ab. Bei starkem Rauschen braucht das Netz das ganze Bild, um zu entscheiden, ob ein verschwommener Fleck einen Schuh oder eine Tasche darstellt.

<!--en:ad467f8a-->
Das U-Net arbeitet mit mehreren Auflösungen. Ein Encoder halbiert die Auflösung zweimal, von $28 \times 28$ über $14 \times 14$ auf $7 \times 7$, und verdoppelt die Kanalzahl einmal von 64 auf 128. So erfasst eine $3 \times 3$-Faltung auf der untersten Stufe den größten Teil des Bildes. Ein Decoder erhöht die Auflösung wieder auf $28 \times 28$. Im Flaschenhals allein gingen feine Details verloren, etwa die Kante eines Ärmels oder die Streifen auf einem Turnschuh. Deshalb erhält der Decoder auf jeder Stufe zusätzlich die Merkmale des Encoders in derselben Auflösung und hängt sie an seine eigenen an. Diese *Skip-Verbindungen* geben dem Netz seine U-Form (:numref:`fig_diffusion-unet`). Ronneberger et al. entwarfen es zur Segmentierung von Strukturen in Mikroskopbildern. Diese Aufgabe verlangt ein Label pro Pixel, also eine Ausgabe in der Größe der Eingabe, die sowohl vom ganzen Bild als auch von den Details am jeweiligen Pixel abhängt. Dasselbe gilt für einen Entrauscher (engl. denoiser).

<!--en:74a12e60-->
![Das U-Net dieses Abschnitts. Auf zwei Downsampling-Stufen folgen ein Flaschenhals bei $7 \times 7$ und zwei Upsampling-Stufen. Skip-Verbindungen hängen die Merkmale des Encoders an die des Decoders in derselben Auflösung an. Die Einbettungen von Zeitpunkt und Klassenlabel durchlaufen ein kleines MLP, dessen Ausgabe in jedem Block addiert wird.](../img/mdl-diffusion-unet.svg)

<!--en:ffc3e4b4-->
Ein einziges Netz entrauscht bei jedem Rauschpegel. Die beste Vorhersage hängt vom Pegel ab. Nahe $t = 0$ entspricht das unverrauschte Bild fast der Eingabe, bei $t = 1$ lässt es sich nur als Mittelwert über viele Bilder schätzen. Deshalb geht die Zeit in jeden Block ein. `d2l.time_embedding` aus :numref:`sec_diffusion-denoising` bildet $t$ auf Sinus- und Kosinuswerte bei 64 Frequenzen ab. Die Frequenzen bilden eine geometrische Folge von 1 bis 64 Radiant pro Zeiteinheit. Das entspricht der Konstruktion der Positionskodierungen in :numref:`sec_positional-information`. Hier sind die Frequenzen auf Zeiten zwischen 0 und 1 ausgelegt statt auf Positionen, die bis in die Tausende reichen. So erhalten benachbarte Rauschpegel deutlich verschiedene Merkmale. Ein zweischichtiges MLP bildet daraus einen Vektor. Jeder Block addiert eine gelernte Projektion dieses Vektors zu seinen Merkmalskarten, einen Bias pro Kanal.

<!--en:02724180-->
Jeder Block wendet zwei $3 \times 3$-Faltungen mit Gruppennormalisierung an. Dazwischen addiert er eine Projektion der Einbettung, die den Block konditioniert. Eine Residualverbindung überbrückt beide Faltungen. Die Gruppennormalisierung berechnet ihre Statistiken über Gruppen von Kanälen innerhalb eines Bildes. Anders als die Batchnormalisierung vermischt sie daher keine Statistiken von Bildern mit unterschiedlichen Rauschpegeln.

<!--en:eb29fac0-->
`d2l.Conv` entspricht `nnx.Conv`, zieht aber Gewichte und Bias-Terme aus $U(-a, a)$ mit $a = 1/\sqrt{\textrm{fan-in}}$, wie es `d2l.Linear` für vollständig verbundene Schichten tut. Flax verwendet standardmäßig LeCun-normalverteilte Gewichte und setzt die Bias-Terme auf null.

<!--en:bf715ecd-->
Zwei Faltungen mit Stride 2 senken die Auflösung im U-Net auf $7 \times 7$. Anschließend erhöht die Nächster-Nachbar-Interpolation sie jeweils auf die Größe der zugehörigen Skip-Verbindung.

<!--en:e36c3e4b-->
Auf welchen Stufen liegen die Parameter eines solchen Netzes, und wo fällt der Rechenaufwand an? Eine $3 \times 3$-Faltung hat bei jeder Auflösung dieselben Gewichte, wird aber einmal pro Pixel ausgeführt. Deshalb kostet dieselbe Schicht bei $28 \times 28$ sechzehnmal so viel wie bei $7 \times 7$. Wir bauen das Netz mit 64 Kanälen bei voller Auflösung für zehn Klassen und zählen für jede Auflösung Parameter und Rechenoperationen.

<!--en:5f11c1a9-->
Auf die beiden unteren Stufen entfallen vier Fünftel der Parameter und zwei Fünftel der Rechenoperationen. Die Stufe mit voller Auflösung enthält ein Sechstel der Parameter, benötigt aber mehr als die Hälfte der Rechenoperationen. Bei $7 \times 7$ liegen im Flaschenhals 38% der Gewichte, doch dort fallen nur 8% der Multiplikationsadditionen an. Zusätzliche Tiefe kostet bei niedriger Auflösung wenig. Deshalb liegen die breitesten Schichten eines U-Nets ganz unten. Eine Auswertung des Netzes kostet etwa eine Viertelmilliarde Multiplikationsadditionen. Behalten wir diese Zahl im Kopf, wenn wir die Auswertungen pro Sample zählen.

<!--en:7d6a41f2-->
Wir halten dieses U-Net bewusst einfach. Seine 1.2 Millionen Parameter sind ein Dreißigstel der 35.7 Millionen Parameter des CIFAR-10-Modells von :citet:`ho2020denoising`. Dieses Modell arbeitet mit vier Auflösungen und je zwei Residualblöcken. Hinzu kommen Self-Attention bei $16 \times 16$ und Dropout. Der veröffentlichte Code initialisiert außerdem die letzte Faltung jedes Blocks mit null. So entspricht jeder Block anfangs seinem Skip-Pfad. Wir verzichten auf all diese Maßnahmen. Dennoch sind die unten erzeugten Kleidungsstücke erkennbar. Seit 2024 verwenden die meisten neuen großen Bildgeneratoren anstelle des U-Nets einen Transformer, der Patches eines komprimierten Bildes verarbeitet. Ein Text-Prompt übernimmt die Rolle der Klasse bei der Konditionierung (:numref:`sec_diffusion-latent`). Der Rest dieses Abschnitts lässt sich unverändert übertragen.

<!--en:8dbecdbc-->
### Klassenkonditionierung

<!--en:bdc05c46-->
Das Klassenlabel ist eine weitere Eingabe. Das U-Net hat dafür schon einen Platz, nämlich die Einbettung, die jeder Block zu seinen Merkmalen addiert. Eine Tabelle mit einem gelernten Vektor pro Klasse liefert den Vektor des Labels. Vor dem MLP addieren wir ihn zu den Zeitmerkmalen. Die Tabelle enthält eine zusätzliche Zeile für das *Null-Label* (engl. null label) $\varnothing$ (hier Index 10). Es steht für „keine Klasse angegeben“.

<!--en:c6f1203e-->
Beim Training ersetzen wir jedes Label mit Wahrscheinlichkeit $0.1$ durch das Null-Label. Diesen Schritt nennen wir *Label-Dropout*. Das eine Netz lernt damit zwei Vorhersagen. $\mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, c)$ ist die Geschwindigkeit der Bilder der Klasse $c$. $\mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \varnothing)$ ist die Geschwindigkeit über alle Bilder von Fashion-MNIST hinweg. Das Netz lernt sie anhand jedes zehnten Beispiels im Mittel. Warum bringen wir dem Netz bei, das vorgegebene Label zu ignorieren? Die Guidance am Ende dieses Abschnitts (:numref:`subsec_diffusion-images-guided`) braucht die Vorhersage ohne Label als Bezugsgröße für die Vorhersage mit Label. Label-Dropout liefert sie im selben Netz fast ohne zusätzliche Kosten. Bei :citet:`Ho.Salimans.2022` funktionierten Dropout-Raten von $0.1$ und $0.2$ besser als $0.5$. Die Bezugsgröße benötigt weniger Daten als die Vorhersagen, die sich auf sie stützen.

<!--en:a9246ac1-->
Ein Sampler integriert ein Feld, das allein von $(\mathbf{x}, t)$ abhängt. `conditional` legt die Labels $c$ eines Batches fest und macht das Netz so zum Feld $(\mathbf{x}, t) \mapsto \mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}, t, c)$.

<!--en:da03adbc-->
## Training und Sampling

<!--en:d3627ede-->
### Verlust und Rechenkosten

<!--en:675be86d-->
Wir verwenden denselben Verlust wie in :numref:`sec_diffusion-gaussian-paths`, diesmal für Bilder anstelle von Punkten. Wir ziehen ein unverrauschtes Bild $\mathbf{x}_0$ mit Werten in $[-1, 1]$ samt Label, einen auf $[0, 1]$ gleichverteilten Zeitpunkt $t$ und Rauschen $\boldsymbol{\epsilon}$. Daraus bilden wir $\mathbf{x}_t = (1 - t)\, \mathbf{x}_0 + t\, \boldsymbol{\epsilon}$ auf dem linearen Pfad. Ziel der Regression ist die bedingte Geschwindigkeit $\boldsymbol{\epsilon} - \mathbf{x}_0$. Wir mitteln den quadratischen Fehler der Netzausgabe über die Pixel. Hinzu kommt lediglich das Label-Dropout, das `batch_fn` auf jeden Batch von 128 Bildern anwendet.

<!--en:3a912870-->
`d2l.train_regression` aus :numref:`sec_diffusion-denoising` minimiert diesen Verlust mit Adam und gibt das trainierte Netz, einen gleitenden Mittelwert seiner Gewichte und die Verlustwerte zurück. Wir verwenden für 15 000 Aktualisierungen eine konstante Lernrate von $10^{-3}$. :citet:`ho2020denoising` trainierten mit $2 \cdot 10^{-4}$ über 800 000 Aktualisierungen, also mehr als fünfzigmal so viele.

<!--en:cc23ff34-->
Das Training ist der günstigere Teil dieses Abschnitts. Dabei durchlaufen $15{,}000 \times 128
\approx 1.9$ Millionen Bilder das Netz vorwärts und rückwärts. Die Samples weiter unten benötigen teils je tausend Schritte, insgesamt etwa 20 Millionen Netzauswertungen. Zählt man einen Rückwärtsdurchlauf wie zwei Vorwärtsdurchläufe, ist das dreieinhalbmal so viel Rechenaufwand wie beim Training. Samples aus einem Diffusionsmodell zu ziehen kann mehr kosten, als es dafür zu trainieren. Ein Großteil des übrigen Kapitels behandelt Verfahren, die diese Kosten senken. Auf einer RTX 4090 dauert dieser Abschnitt weniger als eine Stunde, auf einer CPU wären es viele Stunden. Ohne GPU können Sie 1000 statt 5000 Samples auswerten und das Ergebnis mit dem Referenzwert für 1000 Bilder vergleichen, den wir weiter unten berechnen.

<!--en:b3078e83-->
Der über jeweils 100 Aktualisierungen gemittelte Verlust pro Pixel sinkt von etwa 0.5 auf etwa 0.2. Der größte Teil des Rückgangs fällt in die ersten tausend Aktualisierungen. Der Verlust pendelt sich deutlich oberhalb von null ein, denn die bedingte Varianz der Zielgeschwindigkeit bildet seine Untergrenze (:numref:`sec_diffusion-gaussian-paths`). Bei großem $t$ passen viele unverrauschte Bilder zu demselben verrauschten Bild.

<!--en:04def3e0-->
### Samples und Vorhersagen

<!--en:5662be3d-->
Für das Sampling macht `d2l.make_pred` aus :numref:`sec_diffusion-integration` ein Feld zum Prädiktor für das unverrauschte Bild und das Rauschen. `d2l.ode_sample` integriert die Wahrscheinlichkeitsfluss-ODE von $t = 1$ bis $t = 0$, hier mit 50 Euler-Schritten. `generate` erhält das Feld als Argument, standardmäßig `conditional`. Für die Guidance übergeben wir später ein anderes Feld. Die Funktion zieht gruppenweise $n$ Bilder, das $i$-te aus Klasse $i \bmod 10$. Ein Startwert bestimmt das Anfangsrauschen. Sofern nicht anders angegeben, ist er 0. Zwei Aufrufe mit demselben Startwert unterscheiden sich daher nur in dem, was wir zwischen ihnen ändern.

<!--en:124974fe-->
Jede Spalte enthält vier Samples einer Klasse. Die meisten lassen sich der vorgegebenen Klasse zuordnen und zeigen unterschiedliche Formen, Aufdrucke und Farbtöne. Manche sind verformt. Hemden, Pullover und Mäntel lassen sich wie auf realen Bildern schwer voneinander unterscheiden.

<!--en:2edc7e6d-->
Bei jedem Rauschpegel stellt der Sampler dem Netz dieselbe Frage: Welches unverrauschte Bild steckt dahinter? In der ersten Zeile unten sehen wir dazu ein Testbild einer Stiefelette mit Rauschen zu sechs Zeitpunkten auf dem linearen Pfad. Die zweite Zeile zeigt für jeden Zeitpunkt das unverrauschte Bild $\hat{\mathbf{x}}_0 = \mathbf{x}_t - t\, \mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, c)$, das sich bei gegebenem Label aus der Netzvorhersage ergibt. Beide Enden der zweiten Zeile können wir vorhersagen. Bei $t = 0$ ist die Eingabe das Bild selbst. Bei $t = 1$ besteht sie aus reinem Rauschen. Ein mit quadratischem Fehler trainiertes Netz sagt den A-posteriori-Mittelwert (engl. posterior mean) voraus (:numref:`sec_diffusion-denoising`). Als Information bleibt nur das Label, also ist die Vorhersage das mittlere Trainingsbild der Klasse. Die ersten beiden Zeilen der Textausgabe geben für jedes $\hat{\mathbf{x}}_0$ den Abstand zum Testbild und zu diesem Mittelwert an, jeweils als quadratisches Mittel der Pixeldifferenzen.

<!--en:b00d9f06-->
Die dritte Zeile zeigt den Sampler selbst. Er beginnt mit neu gezogenem Rauschen und nähert sich mit den 50 Euler-Schritten von `euler50` einer Stiefelette. Bei $t = 1, 0.8, \ldots, 0.2$ zeichnet er jeweils das unverrauschte Bild auf, das sich unterwegs aus der Netzvorhersage ergibt. Die Spalte $t = 0$ enthält das fertige Sample. Die letzte Zeile der Textausgabe gibt den Abstand jeder Vorhersage dazu an.

<!--en:ada20c17-->
Beide Enden entsprechen der Erwartung. Bei $t = 1$ beträgt der Abstand der Vorhersage zur mittleren Stiefelette höchstens 0.07 und zum Testbild mehr als 0.4. Dazwischen erkennt das Netz mehr als das menschliche Auge. Bei $t = 0.6$ ist die Stiefelette im Rauschen kaum auszumachen, doch die Vorhersage gibt ihren Umriss wieder. Ihr Abstand zu dieser Stiefelette liegt unter 0.2, zur mittleren dagegen über 0.4. Zuerst verschwinden die Details, zuletzt der Umriss.

<!--en:0232c0a7-->
Ein Sampler durchläuft die Rauschpegel in umgekehrter Reihenfolge. Die dritte Zeile zeigt die Wirkung. Bei $t = 1$ entspricht die Vorhersage der verschwommenen mittleren Stiefelette, mit einem Abstand von 0.3 bis 0.4 zum späteren Sample. Bei $t = 0.6$ beträgt der Abstand nur noch 0.12 bis 0.14, und der endgültige Umriss steht bereits fest. Dabei enthält das Bild, aus dem die Vorhersage entsteht, noch sechs Teile Rauschen auf vier Teile Stiefelette. Die übrigen Schritte schärfen die Kanten und ergänzen Schattierung und Textur. Die Form eines Samples steht früh fest, seine Oberfläche erst spät. Das nutzt Aufgabe :exref:`ex_image-diffusion-guidance-interval`.

<!--en:16751782-->
### Der gleitende Mittelwert

<!--en:79b7a0c8-->
Zum Sampling haben wir `ema` verwendet, einen gleitenden Mittelwert der Gewichte anstelle der Gewichte nach der letzten Aktualisierung. Bei konstanter Lernrate kommen die Iterierten von Adam nicht zur Ruhe. Jede Aktualisierung trägt das Rauschen ihres Minibatches. Die Iterierten streuen daher um einen günstigen Bereich, und der Mittelwert der jüngsten Iterierten liegt näher an dessen Zentrum :cite:`Polyak.Juditsky.1992`. `d2l.EMA` aus :numref:`sec_dcgan` führt einen exponentiellen gleitenden Mittelwert. Nach jeder Aktualisierung bewegt er sich einen kleinen Schritt auf die aktuellen Gewichte zu,

<!--en:7b84e9b9-->
sodass die Gewichte von vor $n_{1/2}$ Aktualisierungen halb so viel zählen wie die aktuellen. $n_{1/2}$ ist die *Halbwertszeit*. Wir wählen $n_{1/2} = 1000$. Damit verringert sich der Abstand bei jeder Aktualisierung um 0.07%. Veröffentlichungen und Bibliotheken geben stattdessen den *Abklingfaktor* $2^{-1/n_{1/2}}$ an, hier 0.9993. Daher lohnt sich die Umrechnung.

<!--en:cae34e39-->
Die Halbwertszeit muss zur Trainingsdauer passen. Ist sie zu kurz, bleibt das Rauschen der letzten Schritte im Mittelwert erhalten. Ist sie zu lang, gehen frühe Gewichte ein, die noch weit vom günstigen Bereich entfernt sind. :citet:`ho2020denoising` verwenden einen Abklingfaktor von 0.9999. Bei insgesamt 800 000 Aktualisierungen entspricht das einer Halbwertszeit von etwa 7000 Aktualisierungen. Sobald wir ein Maß haben, untersuchen wir als Erstes, wie viel der Mittelwert bringt.

<!--en:9492db63-->
## Messung der Sample-Qualität

<!--en:d4afab61-->
### Fréchet-Abstand im Raum der gelernten Merkmale

<!--en:7f521943-->
Ein Raster mit vierzig Samples zeigt, ob die Bilder plausibel aussehen. Es verrät uns aber nicht, ob 5000 davon die richtige Vielfalt aufweisen, ob es zu viele Taschen und zu wenige Sandalen gibt oder welches von zwei augenscheinlich guten Modellen besser ist. Dafür brauchen wir eine Zahl, die die Verteilung der Samples mit der Verteilung realer Bilder vergleicht.

<!--en:ea00bf76-->
Der Pixelraum eignet sich schlecht für diesen Vergleich. Verschieben wir einen Turnschuh um zwei Pixel, ergibt sich ein großer euklidischer Abstand, obwohl das Bild dasselbe zeigt. Deshalb vergleichen wir die Bilder im Merkmalsraum eines Klassifikators. Dort liegen Bilder mit ähnlichem Inhalt nahe beieinander. Für den *Fréchet-Abstand* (FD) nach :citet:`Heusel.Ramsauer.Unterthiner.ea.2017` passen wir Gauß-Verteilungen $\mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$ und $\mathcal{N}(\boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q)$ an die Merkmale realer und generierter Bilder an. Ihr quadrierter Wasserstein-2-Abstand ergibt sich nach :numref:`sec_dcgan` zu

<!--en:21a52f9e-->
Der erste Term wächst, wenn das mittlere Sample vom mittleren realen Bild abweicht, etwa weil der Generator eine Klasse zu häufig erzeugt. Der zweite wächst, wenn sich die Streuungen unterscheiden, etwa weil alle generierten Mäntel ähnlich aussehen. Beide verschwinden nur dann zugleich, wenn die beiden Gauß-Verteilungen übereinstimmen.

<!--en:1fdd92fe-->
Veröffentlichte Werte heißen FID und beruhen auf den Merkmalen eines auf ImageNet trainierten Inception-Netzes. Wie in :numref:`sec_dcgan` trainieren wir stattdessen ein eigenes Netz. `d2l.FeatureCNN` (:numref:`subsec_utils_feature_cnn`) ist ein kleiner konvolutionärer Klassifikator mit 128 Merkmalen. Wir trainieren ihn 15 Epochen lang auf Fashion-MNIST und verwenden den Grauwertkanal dreifach. Da wir ein eigenes Merkmalsnetz nutzen, nennen wir das Maß FD und vergleichen es nie mit veröffentlichten Werten. `fashion_feature_cnn` trainiert das Netz, und `fd_features` berechnet die Merkmale von jeweils 1000 Bildern. Der Ausgabekopf des Netzes ordnet jedem Bild eine Klasse zu.

<!--en:97bde827-->
Das Merkmalsnetz klassifiziert etwa 93% der Testbilder korrekt. Seine Merkmale trennen also die Klassen. `fd_and_agreement` berechnet den FD einer Menge von Samples gegenüber Referenzmerkmalen sowie die *Klassenübereinstimmung* (engl. class agreement). Das ist der Anteil der Samples, die das Merkmalsnetz der gewünschten Klasse zuordnet. Wir vergleichen 5000 Samples mit den ersten 5000 Testbildern.

<!--en:ee36f89d-->
Die Samples erreichen einen FD von etwa 0.8, und das Merkmalsnetz ordnet fast 90% der gewünschten Klasse zu. Ist ein FD von 0.8 gut? Die Zahl hängt vom Merkmalsnetz und von der Anzahl der Bilder ab. Für sich allein sagt sie nichts aus. Der nächste Unterabschnitt liefert einen Maßstab.

<!--en:826428ab-->
### Ein Referenzwert für den Fréchet-Abstand

<!--en:b9b4ebcb-->
Auch zwischen zwei Mengen realer Bilder ist der FD positiv. Ihre Stichprobenmittelwerte und Kovarianzen unterscheiden sich zufallsbedingt. Setzen wir sie als Schätzwerte in :eqref:`eq_diffusion-fd` ein, ergibt sich vor allem im Spurterm eine Verzerrung nach oben. Ihr Betrag ist annähernd proportional zu $1/n$ bei $n$ Bildern je Menge. Um ihre Größe zu bestimmen, vergleichen wir fünf disjunkte Mengen realer Trainingsbilder mit den ersten $n$ Testbildern, der Referenzmenge für die Samples. Diesen Vergleich führen wir bei $n = 1{,}000$ und $n = 5{,}000$ durch. Das Ergebnis nennen wir den *Referenzwert* des FD bei dieser Stichprobengröße. Ist die Verzerrung proportional zu $1/n$, sollte der Wert bei 5000 Bildern etwa ein Fünftel des Werts bei 1000 Bildern betragen. Daneben geben wir die Kernel Inception Distance (KID) nach :citet:`Binkowski.Sutherland.Arbel.ea.2018` aus. Sie schätzt die quadrierte Maximum Mean Discrepancy mit einem kubischen Kernel auf denselben Merkmalen erwartungstreu (:numref:`sec_dcgan`). Für zwei Mengen realer Bilder ist ihr Erwartungswert bei jedem $n$ null.

<!--en:cecc3788-->
Die Vorhersage trifft zu. Bei 1000 Bildern liegt der Referenzwert knapp unter 0.9, bei 5000 beträgt er etwa 0.17, also ein Fünftel davon. Die KID weist keine solche Verzerrung auf, schwankt aber zufallsbedingt. Bei 1000 Bildern streuen ihre fünf Werte um einige Hundertstel zu beiden Seiten von null. Ein einzelner Wert sagt also wenig aus. Bei 5000 Bildern streuen sie um einige Tausendstel. Veröffentlichungen geben den FD an. Das tun wir in diesem Kapitel ebenfalls und berücksichtigen seine Verzerrung durch einen Vergleich statt durch eine Formel.

<!--en:3a84d934-->
Im weiteren Verlauf des Kapitels vergleichen wir FD-Werte nur bei gleicher Bildanzahl und mit demselben Merkmalsnetz. Unterscheiden sich zwei Werte für 5000 Bilder um weniger als etwa 0.05, behandeln wir sie als gleich. Das entspricht der Streubreite der fünf realen Bildmengen. Bei 1000 Bildern beträgt sie 0.15 bis 0.2. Spätere Abschnitte geben den Referenzwert neben ihren Messwerten als *Untergrenze* an. Diesen Wert würde ein perfekter Generator im Mittel erreichen.

<!--en:3fba772a-->
Mit etwa 0.8 ist der FD unserer Samples etwa fünfmal so hoch wie der Referenzwert bei 5000 Bildern. Die Differenz ist der Fehler unseres Modells. Zwei Mengen von je 1000 realen Bildern erreichen im Vergleich miteinander aber einen ähnlich hohen Wert. Ein Fréchet-Abstand ohne Angabe der Bildanzahl ist bedeutungslos.

<!--en:21555037-->
Mit diesem Maßstab können wir untersuchen, wie viel der gleitende Mittelwert bringt und ob ein Sampler mit tausend Schritten besser abschneidet als fünfzig Euler-Schritte. Als Sampler mit tausend Schritten verwenden wir die ancestrale Kette aus :numref:`sec_diffusion-integration` auf dem varianzerhaltenden Pfad aus :citet:`ho2020denoising`. Zwar haben wir unser Netz auf dem linearen Pfad trainiert, doch das spielt keine Rolle. `make_pred` findet auf dem Trainingspfad den Zeitpunkt mit demselben Signal-Rausch-Verhältnis und skaliert entsprechend um. So lässt sich jedes trainierte Netz auf jedem Pfad nutzen. Wir führen einen gepaarten Vergleich durch. Alle drei Sampler starten mit demselben Anfangsrauschen, sodass Unterschiede zwischen den Rauschrealisierungen den Vergleich nicht beeinflussen. Wir wiederholen ihn für drei Realisierungen.

<!--en:54aab2c8-->
Für drei gemeinsame Startwerte des Anfangsrauschens erzeugen wir nun jeweils 5000 Bilder mit den EMA-Gewichten und `euler50`, mit den ungemittelten Endgewichten `net` und `euler50` sowie mit den EMA-Gewichten und `ancestral1000`. Wir bestimmen ihre Fréchet-Abstände gegenüber `f_test[:5000]`:

<!--en:942f7279-->
Bei diesem Netz schneiden die Gewichte der letzten Aktualisierung bei jeder Ziehung um 0.4 bis 0.6 schlechter ab als ihr gleitender Mittelwert. Der FD des Mittelwerts schwankt zwischen den Ziehungen dagegen nur um wenige Hundertstel. Wie viel schlechter die letzten Gewichte abschneiden, hängt davon ab, wo die letzte Aktualisierung zufällig in der Wolke der Iterierten endet. Ein anderer Trainingslauf würde an einer anderen Stelle enden. Über die letzten paar tausend Aktualisierungen zu mitteln verringert diesen Zufallseinfluss erheblich. Deshalb verwenden wir zum Sampling den Mittelwert. Die Kette mit tausend Schritten erzielt einen um 0.1 bis 0.2 niedrigeren FD als fünfzig Euler-Schritte, wiederum bei jeder Ziehung. Dabei ändern sich Pfad, Sampler und Schrittzahl zugleich. Der Vergleich misst also die Wirkung ihrer Kombination. Für zwanzigmal so viele Auswertungen ist der Gewinn bescheiden. Ab jetzt messen wir mit fünfzig Euler-Schritten.

<!--en:07cf17f8-->
## Guidance

<!--en:0dcbe776-->
### Die Bayes-Regel für Scores

<!--en:67bec9f5-->
Unsere bedingt erzeugten Samples gehören schon in fast 90% der Fälle zur gewünschten Klasse. Bei einigen gelingt die Zuordnung nicht. Hemden, Pullover und Mäntel waren schwer zu unterscheiden. Der Schieberegler in der Einführung verlangt mehr, nämlich einen Sampler, der dem Label genauer folgt als das trainierte Netz. Um zu verstehen, wie wir einen solchen Sampler erhalten könnten, betrachten wir die Wirkung eines Labels auf den Score. Die Bayes-Regel zum Zeitpunkt $t$ lautet $p_t(\mathbf{x} \mid c) = p_t(\mathbf{x})\, p_t(c \mid \mathbf{x})
/ p(c)$. Dabei ist $p_t(c \mid \mathbf{x})$ die Wahrscheinlichkeit, dass ein verrauschtes Bild $\mathbf{x}$ zum Zeitpunkt $t$ von einem unverrauschten Bild der Klasse $c$ stammt. Bilden wir den Gradienten des Logarithmus, fällt $p(c)$ weg:

<!--en:ca5f46d7-->
Der Score der Klasse ist die Summe aus dem Score aller Daten und dem Gradienten der Log-Wahrscheinlichkeit eines Klassifikators. Die *Klassifikator-Guidance* (engl. classifier guidance) :cite:`Dhariwal.Nichol.2021` berechnet den zweiten Term mit einem separaten Klassifikator $p_{\boldsymbol{\phi}}(c \mid \mathbf{x}, t)$ und multipliziert ihn mit einer *Guidance-Skala* $\gamma$:

<!--en:32135e85-->
Bei $\gamma = 1$ ergibt sich die Bayes-Regel. Bei :citet:`Dhariwal.Nichol.2021` passten die Samples mit $\gamma > 1$ deutlich besser zu ihren Klassen. Der Klassifikator muss auf verrauschten Bildern trainiert sein, denn der Sampler wertet seinen Gradienten bei $\mathbf{x}_t$ aus. Dort liefert ein Klassifikator für unverrauschte Bilder keine aussagekräftigen Gradienten. Der Anhang untersucht diese Regel an einer eindimensionalen Mischverteilung (:numref:`sec_mdl-ddim-guidance`). In Aufgabe :exref:`ex_image-diffusion-clean-classifier-guide` wenden Sie sie auf unsere Bilder an.

<!--en:22de41d2-->
### Classifier-Free Guidance

<!--en:89b3fdea-->
Klassifikator-Guidance braucht ein zweites, auf verrauschten Bildern trainiertes Netz, das eigene Fehler macht. :citet:`Ho.Salimans.2022` lesen stattdessen :eqref:`eq_diffusion-bayes-score` von rechts nach links:

<!--en:b62c7d66-->
Der Gradient des Klassifikators ist die Differenz aus bedingtem und unbedingtem Score. Ein mit Label-Dropout trainiertes Netz liefert beide. Setzen wir seine beiden Scores $\mathbf{s}_c$ und $\mathbf{s}_\varnothing$ in :eqref:`eq_diffusion-classifier-guidance` ein, erhalten wir den durch Guidance gesteuerten Score $\tilde{\mathbf{s}} = \mathbf{s}_\varnothing + \gamma(\mathbf{s}_c -
\mathbf{s}_\varnothing)$. Das entspricht einer Klassifikator-Guidance, deren Klassifikator implizit im Netz steckt.

<!--en:32b2b65f-->
Unser Netz sagt Geschwindigkeiten voraus. An einem festen Punkt $(\mathbf{x}, t)$ ist die Geschwindigkeit aber eine affine Funktion des Scores. Für den linearen Pfad liefert die Umrechnung aus :numref:`sec_diffusion-gaussian-paths`

<!--en:e4a1fedd-->
Der Term $-\mathbf{x}/(1-t)$ hängt nicht vom Label ab. Da sich die Gewichte $1 - \gamma$ und $\gamma$ zu eins addieren, bleibt der Term bei der Kombination unverändert. Für die gesteuerte Geschwindigkeit ergibt sich damit dieselbe Kombination der beiden Geschwindigkeiten:

<!--en:c3eb457a-->
`cfg` setzt diese Regel um. Die Funktion wertet das Netz auf einem verdoppelten Batch aus. Darin steht jedes Bild einmal mit seinem Label und einmal mit dem Null-Label. Ein gesteuertes Sample kostet also zwei Auswertungen pro Schritt. Bei $\gamma = 1$ vereinfacht sich die Regel zum bedingten Feld, und `cfg` gibt `conditional(net, c)` ohne die zweite Hälfte des Batches zurück.

<!--en:6434549f-->
Das Argument über affine Funktionen gilt für jede Zielgröße der Umrechnung (Aufgabe :exref:`ex_image-diffusion-equivalent-guidance`). :citet:`Ho.Salimans.2022` schreiben die Regel als $(1 + w)\, \mathbf{u}_c - w\, \mathbf{u}_\varnothing$. Ihr $w$ entspricht also unserem $\gamma - 1$. :numref:`fig_diffusion-guidance` veranschaulicht die Regel an unserem durchgehenden Beispiel. Für $\gamma > 1$ liegt die gesteuerte Vorhersage auf der Geraden durch die unbedingte und die bedingte Vorhersage, jenseits der bedingten.

<!--en:9d8b1dd2-->
![Classifier-Free Guidance für die Mischverteilung mit drei Modi bei einem festen Rauschpegel. Von einem verrauschten Punkt $\mathbf{x}_t$ zeigen Pfeile zum A-posteriori-Mittelwert $\hat{\mathbf{x}}_0^{\varnothing}$ aller Daten, zum A-posteriori-Mittelwert $\hat{\mathbf{x}}_0^{c}$ bei gegebenem oberen Modus und zur gesteuerten Vorhersage $\hat{\mathbf{x}}_0^{\varnothing} + \gamma(\hat{\mathbf{x}}_0^{c} - \hat{\mathbf{x}}_0^{\varnothing})$ bei $\gamma = 2$ jenseits von $\hat{\mathbf{x}}_0^{c}$. Die Konturlinien zeigen die Dichte nach dem Hinzufügen von Rauschen für die gesamte Verteilung sowie für den oberen Modus.](../img/mdl-diffusion-guidance.svg)

<!--en:2677961d-->
Um die Wirkung von $\gamma$ zu sehen, erzeugen wir für jede von sieben Skalen 5000 Samples aus demselben Rauschen. Der FD misst, wie gut sie die Verteilung realer Bilder abdecken, die Klassenübereinstimmung, wie gut sie ihren Labels folgen.

<!--en:a6f3d16a-->
Ohne Klassenkonditionierung ($\gamma = 0$) beträgt die Klassenübereinstimmung $10\%$ und liegt damit auf Zufallsniveau. Der FD steigt auf $2.7$–$2.8$ und ist mehr als dreimal so hoch wie der bedingte FD bei $\gamma = 1$ ($\approx 0.8$ bei fast $90\%$ Klassenübereinstimmung). Wie viel vom Anstieg des FD bei $\gamma = 0$ geht auf verschobene Klassenanteile zurück, wie viel auf schlechtere Bilder innerhalb der Klassen? Dazu vergleichen wir in der nächsten Zelle den Mittelwertterm $\|\boldsymbol{\mu}_p - \boldsymbol{\mu}_q\|^2$ aus :eqref:`eq_diffusion-fd` und untersuchen das Histogramm der generierten Klassen. Außerdem berechnen wir den FD für 5000 reale Trainingsbilder, die wir entsprechend den Klassenanteilen bei $\gamma = 0$ neu ziehen.

<!--en:9db787f1-->
Der unbedingte Sampler erzeugt zu viele Pullover ($13\%$–$14\%$) und zu wenige Sandalen ($6\%$–$7\%$). Erzwingen wir dieselben Klassenanteile bei realen Bildern, steigt der FD nur um $0.2$–$0.3$ (von $0.19$ auf $0.4$–$0.5$). Das erklärt weniger als ein Sechstel der FD-Differenz von $1.9$ zwischen $\gamma = 1$ und $\gamma = 0$. Der Rest entsteht, weil das unbedingte Feld $\mathbf{u}_{\boldsymbol\theta}(\mathbf{x}_t, t, \varnothing)$ mehr verformte oder mehrdeutige Kleidungsstücke erzeugt. Es lernt aus nur $10\%$ der Beispiele in den Minibatches und muss dabei alle zehn Klassen zugleich abdecken. Da wir :eqref:`eq_diffusion-cfg` als $\mathbf{u}_c + (\gamma - 1)(\mathbf{u}_c - \mathbf{u}_\varnothing)$ schreiben können, geht jeder Fehler der unbedingten Bezugsgröße $\mathbf{u}_\varnothing$ mit Gewicht $1 - \gamma$ in die gesteuerten Samples ein.

<!--en:547a8e94-->
Erhöhen wir $\gamma$ über $1$ hinaus, zeigt sich der Zielkonflikt zwischen Vielfalt und Klassentypizität. Schwache Guidance ($\gamma = 1.25$) erhöht die Klassenübereinstimmung um etwa fünf Prozentpunkte auf $93\%$–$94\%$. Der FD steigt dabei nicht über die Schwankungen zwischen den Läufen hinaus. Für $\gamma \ge 1.5$ übersteigt die Klassenübereinstimmung $96\%$. Sie liegt sogar über $99\%$ bei $\gamma = 3$ und $4$ und damit über der Genauigkeit des Klassifikators auf realen Testbildern. Gleichzeitig steigt der FD stetig auf $5.5$–$6$, denn die Vielfalt innerhalb der Klassen schwindet und übrig bleiben kontrastreiche, prototypische Bilder:

<!--en:42aa12da-->
In der Praxis verwenden Text-zu-Bild-Systeme hohe Guidance-Skalen. Wer solche Systeme nutzt, beurteilt an einzelnen Bildern, wie gut sie zum Prompt passen und wie klar sie aussehen. Die Kovarianz eines Batches von 5000 Samples spielt dabei keine Rolle.

<!--en:776ca69c-->
### Die Verteilung eines gesteuerten Samplers

<!--en:005395f5-->
Bei $\gamma = 1$ zieht der Sampler Bilder aus der vorgegebenen Klasse, bei $\gamma = 0$ aus allen Bildern. Für andere Skalen wurde die Regel aus :eqref:`eq_diffusion-cfg` nicht aus einer Verteilung hergeleitet. Suchen wir dennoch nach einer solchen Verteilung. Der gesteuerte Score $\nabla \log p_t(\mathbf{x}) + \gamma\, \nabla \log p_t(c \mid \mathbf{x})$ ist der Gradient des Logarithmus der *umgewichteten* (engl. tilted) Dichte

<!--en:cc8fa6e1-->
Für $\gamma > 1$ wird die vom Klassifikator angegebene Wahrscheinlichkeit potenziert. Dadurch verlagert sich Wahrscheinlichkeitsmasse zu den Bildern, bei denen sich der Klassifikator am sichersten ist. Gleichbedeutend damit ist, einen Teil der unbedingten Dichte herauszudividieren. Das drängt Samples aus Bereichen, die sich die Klasse mit anderen Klassen teilt.

<!--en:1a3ce173-->
Wie groß ist dieser Effekt? Im einfachsten Fall lässt sich die Antwort in geschlossener Form angeben. Die Bilder der Klasse $c$ seien gemäß $\mathcal{N}(\boldsymbol{\mu}_c, s^2\mathbf{I})$ verteilt. Für alle Bilder zusammen nehmen wir stark vereinfachend eine einzige breitere Gauß-Verteilung $\mathcal{N}(\mathbf{0}, S^2\mathbf{I})$ mit $S^2 > s^2$ an. Das ist die gewagteste Annahme des Abschnitts, denn eine Mischung aus zehn Klassen ist keine Gauß-Verteilung. Gerade diese Annahme ermöglicht aber exakte Antworten. Der Logarithmus einer Gauß-Dichte ist quadratisch. Daher ist auch die logarithmierte umgewichtete Dichte der unverrauschten Bilder, bis auf eine Konstante $\gamma \log p(\mathbf{x} \mid c) + (1 - \gamma) \log
p(\mathbf{x})$, wieder der Logarithmus einer Gauß-Dichte. Fassen wir die quadratischen und linearen Terme zusammen, erhalten wir die inverse Varianz $1/v$ und den Mittelwert:

<!--en:2cdbfb4e-->
Die inverse Varianz der Klasse zählt $\gamma$-mal, die aller Bilder ziehen wir $\gamma - 1$-mal ab. Für $\gamma > 1$ übersteigt das Ergebnis $1/s^2$ um $(\gamma - 1)(1/s^2 - 1/S^2) > 0$, und der Faktor vor $\boldsymbol{\mu}_c$ ist größer als eins. Die Guidance verengt also die Klassenverteilung und verschiebt sie vom Mittelwert aller Bilder weg. In einer Dimension mit $\mu_c = 1$, $s = 0.5$ und $S^2 = 2$ verringert die Skala $\gamma = 4$ die Standardabweichung der Klasse von 0.5 auf 0.26 und verschiebt ihren Mittelwert von 1 auf 1.10. So lässt sich der Tausch von Vielfalt gegen Klassentypizität in einer Formel ausdrücken.

<!--en:0fe954d4-->
Endet der gesteuerte Sampler wenigstens bei dieser schmaleren Gauß-Verteilung? Nein, der Unterschied ist sogar groß. In einer Dimension ist die gesteuerte Geschwindigkeit in diesem Beispiel affin in $x$ für jedes $t$. Die Wahrscheinlichkeitsfluss-ODE führt daher eine Gauß-Verteilung wieder in eine Gauß-Verteilung über. Mittelwert und Varianz lassen sich exakt integrieren. Der Sampler schießt über die Umgewichtung hinaus. Startet er bei $\gamma = 4$ mit reinem Rauschen, endet er bei einem Mittelwert von 1.23 und einer Standardabweichung von 0.02. Die Klasse ist damit um mehr als den Faktor zwanzig geschrumpft. Ihre Breite beträgt weniger als ein Zehntel der Breite der umgewichteten Gauß-Verteilung.

<!--en:01b6eb13-->
Der Sampler folgt nicht der umgewichteten Verteilung bei $t = 0$. Die umgewichteten Dichten bei $t > 0$ entstehen nämlich weder aus ihr noch aus einer anderen unverrauschten Verteilung durch Hinzufügen von Rauschen. Eine Dichte, die durch Addition von gaußschem Rauschen der Varianz $\sigma_t^2$ entsteht, kann keine stärkere Krümmung aufweisen als das Rauschen selbst. Kein Eigenwert von $-\nabla^2 \log p_t$ übersteigt $1/\sigma_t^2$ (Beweis in :numref:`sec_mdl-denoising-score-matching`). Auf dem linearen Pfad gilt $\sigma_t = t$. Die beiden Gauß-Verteilungen bleiben gaußförmig, mit Varianzen $(1 - t)^2 s^2 + t^2$ und $(1 - t)^2 S^2 + t^2$. Dieselbe Rechnung wie bei $t = 0$ zeigt, dass die Umgewichtung nahe $t = 1$ die Schranke verletzt, sobald $\gamma > S^2/(S^2 - s^2)$ gilt. Der Schwellenwert beträgt in unserem Beispiel $8/7$. Bei $\gamma = 1.5$ ist die Schranke für jedes $t$ oberhalb von 0.47 verletzt, bei $\gamma = 4$ für jedes $t$ oberhalb von 0.24 (Aufgabe :exref:`ex_image-diffusion-gaussian-guidance`). Über einen großen Teil seines Pfades folgt der gesteuerte Sampler also Feldern, deren Zug stärker ist, als es bei einer verrauschten Verteilung möglich wäre. Diese Wirkung verstärkt sich von Schritt zu Schritt. Der Anhang kommt für zwei Klassen in einer Dimension zum selben Ergebnis (:numref:`sec_mdl-ddim-guidance`).

<!--en:c4e169c6-->
Da das Problem bei starkem Rauschen auftritt, schalten :citet:`Kynkaanniemi.Aittala.Karras.ea.2024` die Guidance dort ab und setzen sie nur bei mittleren Rauschpegeln ein. So bleibt der Gewinn an Klassenübereinstimmung größtenteils erhalten, während weniger Vielfalt verloren geht. In Aufgabe :exref:`ex_image-diffusion-guidance-interval` erproben Sie das an unserem Modell. :citet:`Karras.Aittala.Kynkaanniemi.ea.2024` ändern stattdessen die Bezugsgröße. Sie ersetzen die unbedingte Vorhersage durch die einer kleineren oder weniger intensiv trainierten Version des bedingten Modells. Ihre Fehler ähneln denen des gesteuerten Modells. Daher verringert die Extrapolation von ihr weg die Fehler statt der Vielfalt.

<!--en:6c1f8028-->
All das kostet Rechenzeit. Jedes gesteuerte Sample oben erforderte hundert Netzauswertungen, also fünfzig Schritte mit verdoppeltem Batch. Die Kette mit tausend Schritten schnitt nur wenig besser ab als fünfzig Schritte. Mit wie wenigen Auswertungen wir auskommen, ist das Thema von :numref:`sec_diffusion-samplers`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:50d58468-->
Ein U-Net zur Entrauschung verbindet Flaschenhalsschichten niedriger Auflösung mit Skip-Verbindungen in voller Auflösung. Erstere erfassen mit dem Großteil der Parameter und einem Bruchteil der Rechenoperationen den globalen Kontext. Letztere erhalten scharfe Bildkanten. Zeit- und Klasseneinbettungen gehen in jeden Residualblock ein. Verwenden wir beim Sampling den exponentiellen gleitenden Mittelwert (EMA) der Gewichte anstelle der letzten Adam-Iterierten, sinkt der Fréchet-Abstand erheblich. Der Fréchet-Abstand selbst hat bei endlichen Stichproben eine positive Verzerrung der Ordnung $O(1/n)$. Als Maßstab brauchen wir daher immer den Referenzwert zwischen disjunkten Mengen realer Bilder derselben Größe $n$. Durch zufälliges Label-Dropout lernt ein einziges Netz bedingte und unbedingte Felder. Damit lässt sich die Bilderzeugung durch Classifier-Free Guidance $\tilde{\mathbf{u}} = \mathbf{u}_\varnothing + \gamma(\mathbf{u}_c - \mathbf{u}_\varnothing)$ steuern. Kleine Skalen ($\gamma \approx 1.25$) erhöhen die Klassenübereinstimmung, ohne den FD zu verschlechtern. Größere Skalen ($\gamma > 1.5$) erhöhen die Klassentypizität auf Kosten der Sample-Vielfalt. Bei starkem Rauschen verringern die umgewichteten Scores die Varianz nämlich stärker, als es für eine mit einer Gauß-Dichte gefaltete Dichte möglich ist.

<!--en:2294243a-->
## Übungen

<!--en:09f1c585-->
1. **Äquivalente Guidance-Regeln.** Nehmen Sie an, dass die beiden Geschwindigkeiten des Netzes exakt sind. Nach den Umrechnungen aus :numref:`sec_diffusion-gaussian-paths` sind sie dann affine Funktionen der Scores $\nabla \log p_t(\mathbf{x} \mid c)$ und $\nabla \log p_t(\mathbf{x})$. Der Klassifikator liefere außerdem die exakte A-posteriori-Verteilung $p_t(c \mid \mathbf{x})$ der Klassen für die verrauschten Bilder.
    1. Zeigen Sie, dass die Klassifikator-Guidance :eqref:`eq_diffusion-classifier-guidance` mit Skala $\gamma$ und die Classifier-Free Guidance :eqref:`eq_diffusion-cfg` mit demselben $\gamma$ dieselbe gesteuerte Geschwindigkeit ergeben.
    1. Jede Zielgröße der Umrechnung (Rauschen, unverrauschtes Bild, Score, $\mathbf{v}$ und Geschwindigkeit) ist bei festem $(\mathbf{x}, t)$ eine affine Funktion $a(\mathbf{x}, t) + b(t)\, \hat{\mathbf{x}}_0$ des A-posteriori-Mittelwerts $\hat{\mathbf{x}}_0$. Zeigen Sie, dass die Anwendung der Regel auf eine beliebige Zielgröße mit anschließender Umrechnung dieselbe gesteuerte Vorhersage ergibt wie die direkte Anwendung auf jede andere Zielgröße. Welche Eigenschaft der Gewichte $1 - \gamma$ und $\gamma$ nutzt das Argument?
    1. Wäre die Regel noch für alle Zielgrößen dieselbe, wenn ein Netz ein auf $[-1, 1]$ begrenztes $\hat{\mathbf{x}}_0$ vorhersagte?
   :exlabel:`ex_image-diffusion-equivalent-guidance`

<!--en:a99b47e4-->
1. [code] **Guidance mit einem Klassifikator für unverrauschte Bilder.** ● Klassifikator-Guidance benötigt einen Klassifikator für verrauschte Bilder. Man kann aber auch einen Klassifikator für unverrauschte Bilder zur Steuerung verwenden, indem man ihn auf die Schätzung des unverrauschten Bildes anwendet.
    1. Bestimmen Sie die Genauigkeit des Merkmalsnetzes `cnn` auf Testbildern, die Sie auf dem linearen Pfad zu $\mathbf{x}_t$ bei $t \in \{0.25, 0.5, 0.75\}$ verrauschen. Wie erklären die Genauigkeiten, warum der Text einen auf verrauschten Bildern trainierten Klassifikator verlangt?
    1. Verwenden Sie stattdessen den A-posteriori-Mittelwert $\hat{\mathbf{x}}_0(\mathbf{x}_t)$ der unbedingten Vorhersage (`cfg(ema, c, 0.0)`) als Eingabe für den Klassifikator. Addieren Sie $\gamma\, \nabla_{\mathbf{x}_t} \log p_{\textrm{cnn}}(c \mid \hat{\mathbf{x}}_0(\mathbf{x}_t))$ zum unbedingten Score und rechnen Sie das Ergebnis in eine Geschwindigkeit um. Ziehen Sie bei $\gamma \in \{1, 3, 10\}$ mit 50 Euler-Schritten jeweils 1000 Bilder.
    1. Bewerten Sie die Samples mit `fd_and_agreement`. Nun misst dasselbe Netz die Klassenübereinstimmung, das den Sampler gesteuert hat. Trainieren Sie ein zweites `d2l.FeatureCNN` mit einem anderen Startwert und bewerten Sie die Samples auch damit. Was sagt der Vergleich über den Einsatz desselben Klassifikators zur Steuerung und Auswertung aus?
   :exlabel:`ex_image-diffusion-clean-classifier-guide`

<!--en:5d8c457d-->
1. [code] **Guidance in einem begrenzten Intervall.** :citet:`Kynkaanniemi.Aittala.Karras.ea.2024` wenden Guidance nur innerhalb eines Intervalls von Rauschpegeln an und verwenden sonst $\gamma = 1$. Sie messen den Rauschpegel mit $\bar\sigma_t = \sigma_t / \alpha_t$. Bei einem ihrer ImageNet-Modelle schnitt $0.28 < \bar\sigma_t \leq 2.9$ am besten ab.
    1. Bestimmen Sie die Zeitpunkte $t$ auf dem linearen Pfad, zu denen $\bar\sigma_t$ die Werte $0.28$ und $2.9$ annimmt.
    1. Schreiben Sie eine Version von `cfg`, die die Skala $\gamma$ nur für $t$ in einem vorgegebenen Intervall anwendet. Berechnen Sie bei $\gamma = 4$ den FD und die Klassenübereinstimmung von 5000 Samples jeweils für Guidance im oberen, mittleren und unteren Rauschintervall. Vergleichen Sie die Ergebnisse mit Guidance in jedem Schritt und mit $\gamma = 1$.
    1. Welches Intervall trägt am meisten zum Anstieg der Übereinstimmung bei, welches zum Anstieg des FD? Setzen Sie Ihre Antwort in Beziehung zu dem Zeitpunkt, an dem die Klasse eines Samples feststeht (Aufgabe :exref:`ex_gaussian-paths-commitment-time` in :numref:`sec_diffusion-integration`), und zur dritten Zeile der Bildfolge mit den Vorhersagen in diesem Abschnitt. Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-guidance-levels` in :numref:`sec_mdl-score-matching-diffusion-flow` stellt dieselbe Frage für zwei Klassen in einer Dimension.
   :exlabel:`ex_image-diffusion-guidance-interval`

<!--en:b2ffc7b5-->
1. [code] **Skip-Verbindungen.** Schreiben Sie eine Variante von `UNet` ohne die beiden Skip-Verbindungen. Dabei erhalten `up1` und `up0` nur die hochskalierten Merkmale, sodass ihre Eingabebreiten auf `2 * ch` beziehungsweise `ch` sinken.
    1. Sagen Sie vor dem Training voraus, zu welchen Zeitpunkten $t$ die Skip-Verbindungen am wichtigsten sind und wann sie kaum eine Rolle spielen.
    1. Trainieren Sie beide Netze für 5000 Aktualisierungen. Bestimmen Sie für 1000 Testbilder jeweils den quadratischen Fehler des unverrauschten Bildes, das sich aus der Geschwindigkeitsvorhersage des jeweiligen Netzes ergibt, also $\hat{\mathbf{x}}_0 = \mathbf{x}_t - t\, \hat{\mathbf{u}}$. Werten Sie ihn bei $t \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ aus und zeichnen Sie beide Kurven.
    1. Vergleichen Sie den FD von je 5000 Samples der beiden Netze. Lässt sich der Unterschied anhand der Fehlerkurve aus (b) vorhersagen?
   :exlabel:`ex_image-diffusion-skip-connections`

<!--en:556a8987-->
1. [code] **Begrenzung der vorhergesagten Pixelwerte.** ● Die gesteuerte Geschwindigkeit bestimmt auf dem linearen Pfad ein unverrauschtes Bild $\hat{\mathbf{x}}_0 = \mathbf{x}_t - t\, \tilde{\mathbf{u}}$. :citet:`saharia2022photorealistic` führen die Sättigung der Samples bei großen Guidance-Skalen auf Vorhersagen $\hat{\mathbf{x}}_0$ außerhalb des Wertebereichs der Daten zurück.
    1. Erfassen Sie für $\gamma \in \{1, 4, 16\}$ bei jedem Euler-Schritt den Anteil der Pixel von $\hat{\mathbf{x}}_0$ außerhalb von $[-1, 1]$ und tragen Sie ihn gegen $t$ auf. Wo auf dem Pfad verlässt die Vorhersage den Wertebereich, und welchen Einfluss hat $\gamma$ darauf?
    1. Bei der *statischen Begrenzung* beschneiden Sie $\hat{\mathbf{x}}_0$ vor dem Schritt auf $[-1, 1]$. Bei der *dynamischen Begrenzung* setzen Sie für jedes Bild $m$ auf das Maximum aus $1$ und dem 99.5. Perzentil von $|\hat{\mathbf{x}}_0|$. Dann beschneiden Sie die Werte auf $[-m, m]$ und teilen sie durch $m$. Schreiben Sie den Euler-Schritt so um, dass er die Geschwindigkeit aus dem begrenzten $\hat{\mathbf{x}}_0$ bildet, und implementieren Sie beide Varianten.
    1. Definieren Sie die *Sättigung* einer Bildmenge als den Anteil ihrer Pixel, deren Wert nach der Begrenzung gleich $1$ ist. Vergleichen Sie Sättigung, FD und Klassenübereinstimmung der drei Sampler bei $\gamma \in \{4, 8, 16\}$.
   :exlabel:`ex_image-diffusion-thresholding`

<!--en:0281106a-->
1. **Guidance bei gaußverteilten Klassen.** ● Betrachten Sie die gaußverteilten Klassen dieses Abschnitts. Die Klasse $c$ ist gemäß $\mathcal{N}(\boldsymbol{\mu}_c, s^2 \mathbf{I})$ verteilt, alle Bilder zusammen gemäß $\mathcal{N}(\mathbf{0}, S^2 \mathbf{I})$ mit $S^2 > s^2$. Auf dem linearen Pfad kommt Rauschen hinzu.
    1. Leiten Sie die im Text angegebenen Ausdrücke für die inverse Varianz und den Mittelwert der umgewichteten Gauß-Verteilung bei $t = 0$ her. Zeigen Sie dann, dass für jedes $t$ der gesteuerte Score $\gamma \nabla \log p_t(\mathbf{x} \mid c) + (1 - \gamma) \nabla \log p_t(\mathbf{x})$ der Score einer Gauß-Verteilung ist. Bestimmen Sie ihre Varianz und zeigen Sie, dass diese nahe $t = 1$ genau dann unter $t^2$ fällt, wenn $\gamma > S^2/(S^2 - s^2)$ gilt.
    1. In einer Dimension mit $\mu_c = 1$, $s^2 = 0.25$ und $S^2 = 2$ ist die gesteuerte Geschwindigkeit affin in $x$ für jedes $t$. Stellen Sie die ODEs für den Mittelwert und die Varianz auf. Integrieren Sie sie ausgehend von $\mathcal{N}(0, 1)$ bei $t = 0.999$ bis $t = 0$ für $\gamma \in \{1, 2, 4\}$. Prüfen Sie bei $\gamma = 4$ den im Text angegebenen Endpunkt (Mittelwert 1.23, Standardabweichung 0.02). Vergleichen Sie jeden Endpunkt mit der umgewichteten Gauß-Verteilung bei $t = 0$ und erklären Sie den Unterschied anhand der Schranke aus dem Text.
    1. Sowohl der Trunkierungstrick aus Aufgabe :exref:`ex_conditional-biggan-s-truncation-trick` in :numref:`sec_gan_conditional`, angewandt auf den Generator $\boldsymbol{\mu}_c + s \mathbf{z}$, als auch das Sampling aus $p(\mathbf{x} \mid c)^{1/T}$ bei einer Temperatur $T < 1$ verengen die Klassenverteilung. Welche der drei Methoden verschiebt auch den Klassenmittelwert und warum?
   :exlabel:`ex_image-diffusion-gaussian-guidance`

<!--en:9040a6a3-->
[Diskussion](https://d2l.discourse.group/t/36822)

<!--en:9e3bef89-->
Architekturen, Auswertung und Guidance für die Bilddiffusion<br>**mehrskaliges U-Net · Zeit- und Klassenkonditionierung · EMA-Gewichte · Fréchet-Abstand und 1/n-Referenzwert · Classifier-Free Guidance (CFG)**

<!--en:25e570af-->
::: {.slide title="Diffusion von der 2D-Ebene zu 784-dimensionalen Bildern"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:22e08607-->
- **1. U-Net mit mehreren Auflösungen:** Globales rezeptives Feld von $28 \times 28$ bei starkem Rauschen ($t \to 1$) und Skip-Verbindungen in voller Auflösung bei schwachem Rauschen ($t \to 0$)

<!--en:7427e074-->
- **2. EMA der Gewichte und Label-Dropout:** Der EMA dämpft das Minibatch-Rauschen bei SGD. Mit $10\%$ zufälligem Label-Dropout lernt ein Netz bedingte und unbedingte Felder.

<!--en:1c29789e-->
- **3. Kalibrierter FD und Classifier-Free Guidance (CFG):** $5{,}000$ Samples mit dem $1/n$-Referenzwert aus zwei realen Stichproben vergleichen und von $\mathbf{u}_\varnothing$ in Richtung $\mathbf{u}_c$ mit Skala $\gamma$ extrapolieren

<!--en:36c69910-->
::: {.slide title="Zeit- und klassenkonditioniertes U-Net zur Entrauschung"}

<!--en:09abd9eb-->
[Grundarchitektur mit mehreren Auflösungen]{.kicker}

<!--en:86b2dd18-->
Bei voller Auflösung brauchen $3 \times 3$-Faltungen $14$ Schichten, damit das rezeptive Feld eines Pixels $28$ Pixel abdeckt. Das **U-Net** (Ronneberger et al., 2015; Ho et al., 2020) verbindet groben globalen Kontext mit Skip-Verbindungen für feine räumliche Details:

<!--en:bb4570c4-->
- **Encoder-Decoder-Hierarchie:** Zwei Faltungen mit Stride 2 halbieren die Auflösung zweimal ($28 \times 28 \to 14 \times 14 \to 7 \times 7$), während sich die Kanalzahl verdoppelt ($64 \to 128$). So erfasst ein $3 \times 3$-Faltungskern am Flaschenhals den größten Teil des Bildes.

<!--en:38d191d7-->
- **Skip-Verbindungen in voller Auflösung:** Das Anhängen der Merkmalskarten des Encoders an die Decoderstufen erhält scharfe Kanten, die am $7 \times 7$-Flaschenhals verschwimmen würden.

<!--en:55215043-->
- **Konditionierung je Block (`UNetBlock`):** GroupNorm (8 Gruppen, getrennte Statistiken für verschiedene Rauschpegel im Batch) + SiLU + $3 \times 3$-Faltung. Eine lineare Projektion der gemeinsamen Zeit- und Label-Einbettung liefert einen additiven Bias pro Kanal.

<!--en:98d1d8d2-->
::: {.slide title="Verteilung der Parameter und FLOPs" layout="tight"}

<!--en:4c8ebbdd-->
[Ressourcenbedarf des U-Nets]{.kicker}

<!--en:10657333-->
- **Hohe Kapazität bei niedriger Auflösung:** Die Stufen mit $14 \times 14$ und $7 \times 7$ enthalten **$80\%$ der $1.20\text{M}$ Parameter** ($38\%$ im $7 \times 7$-Flaschenhals bei nur $8\%$ des Rechenaufwands).

<!--en:0fe553bc-->
- **Größter Rechenaufwand bei voller Auflösung:** Die $28 \times 28$-Blöcke enthalten $18\%$ der Gewichte, benötigen aber **$57\%$ der $266\text{M}$ MACs** (pro Gewicht etwa $15\times$ so viel Rechenaufwand wie bei $7 \times 7$).

<!--en:5df76a3a-->
::: {.slide title="Verteilung der Parameter und FLOPs (Code)" layout="tight"}

<!--en:4c8ebbdd-2-->
[Ressourcenbedarf des U-Nets]{.kicker}

<!--en:98dd7f3c-->
::: {.slide title="Gemeinsame Konditionierung durch zufälliges Label-Dropout"}

<!--en:0c4b2635-->
[Ein Netz, zwei Vektorfelder]{.kicker}

<!--en:25d3f0c2-->
- **Token für das Null-Label ($\varnothing = 10$):** In jedem Minibatch aus $128$ Bildern werden zufällig $10\%$ der Klassenlabels durch $\varnothing = 10$ ersetzt. Das entspricht Zeile 11 der Tabelle für die Label-Einbettung mit den Maßen $11 \times$ `emb_dim`.

<!--en:f9592cff-->
- **Gemeinsame Repräsentation:** Ein U-Net lernt ohne zusätzliche Parameter sowohl die klassenbedingte Geschwindigkeit $\mathbf{u}_{\boldsymbol\theta}(\mathbf{x}_t, t, c)$ ($90\%$) als auch die unbedingte Geschwindigkeit $\mathbf{u}_{\boldsymbol\theta}(\mathbf{x}_t, t, \varnothing)$ ($10\%$).

<!--en:6b50668d-->
::: {.slide title="Trainings- und Samplingkosten bei der Bilddiffusion"}

<!--en:7d5862c2-->
[Trainingsdynamik und Rechenbudget]{.kicker}

<!--en:fe621354-->
Training des U-Nets mit $1.2\text{M}$ Parametern und Adam ($\text{lr} = 10^{-3}$, Batchgröße $128$, EMA-Halbwertszeit $1{,}000$) über $15{,}000$ Aktualisierungen:

<!--en:7e73cb0f-->
- **Schnelle Konvergenz zur Fehleruntergrenze:** Der mittlere quadratische Fehler der Geschwindigkeit pro Pixel fällt von $\approx 0.50$ auf $\approx 0.20$ innerhalb der ersten $1{,}000$ Schritte. Er pendelt sich an der durch die bedingte Varianz gegebenen Untergrenze ein.

<!--en:24926406-->
- **Warum Sampling mehr kosten kann als Training:** $15{,}000$ Aktualisierungen verarbeiten $1.92\text{M}$ Trainingsbilder (Gegenwert von $\approx 5.8\text{M}$ Vorwärtsdurchläufen). Dagegen erfordern die Auswertungen dieses Abschnitts mit je $5{,}000$ generierten Bildern **$\approx 20\text{M}$ U-Net-Auswertungen** ($3.5\times$ so viel Rechenaufwand wie beim Training).

<!--en:25db1b55-->
::: {.slide title="Klassenbedingte Erzeugung von Kleidungsstücken in 50 Schritten"}

<!--en:9f10895f-->
[Samples aus der ODE mit dem Euler-Verfahren]{.kicker}

<!--en:bf08903e-->
Die Integration des Geschwindigkeitsfelds des EMA-U-Nets mit $50$ Euler-Schritten (`ode_sample` auf dem linearen Pfad) erzeugt $40$ klassenkonditionierte Kleidungsstücke (eine Klasse je Spalte):

<!--en:70b9faee-->
::: {.slide title="Trajektorie vom Groben zum Feinen: erst die Form, dann die Details"}

<!--en:8ab5eb2b-->
[Vorhergesagtes unverrauschtes Bild im Zeitverlauf]{.kicker}

<!--en:365fdf67-->
Die Geschwindigkeit bestimmt den A-posteriori-Mittelwert $\hat{\mathbf{x}}_0(\mathbf{x}_t) = \mathbf{x}_t - t\,\mathbf{u}_{\boldsymbol\theta}(\mathbf{x}_t, t, c)$ des unverrauschten Bildes. Vorwärts hinzugefügtes Rauschen (Zeilen 1–2) und eine Rückwärtstrajektorie mit 50 Schritten (Zeile 3) zeigen, wann Struktur entsteht:

<!--en:cb3268f5-->
- **Bei reinem Rauschen ($t = 1.0$):** Ist nur das Label *„Stiefelette“* bekannt, entspricht die MSE-optimale Vorhersage $\hat{\mathbf{x}}_0$ der verschwommenen mittleren Stiefelette ($\textrm{RMS} = 0.07$ Abstand vom Klassenmittel).

<!--en:e64aeea2-->
- **Umriss steht bei $t = 0.6$ fest:** Obwohl der Zustand $\mathbf{x}_{0.6}$ zu $60\%$ aus gaußschem Rauschen besteht, zeigt $\hat{\mathbf{x}}_0$ bereits die Silhouette des endgültigen Samples ($\textrm{RMS} \le 0.14$ Abstand zum fertigen Sample bei $t = 0$).

<!--en:f8ee7c61-->
- **Späte Schritte ($t < 0.4$):** Verfeinerung hochfrequenter Kanten, der Schattierung und der Textur

<!--en:5de347cc-->
::: {.slide title="Exponentieller gleitender Mittelwert (EMA) der Modellgewichte"}

<!--en:98c991a7-->
[Stabilisierung der Iterierten beim stochastischen Gradientenabstieg]{.kicker}

<!--en:e5f7e7a0-->
- **Parametrisierung über die Halbwertszeit ($n_{1/2} = 1{,}000$):** Gewichte von vor $1{,}000$ Aktualisierungen zählen halb so viel wie die aktuelle Iterierte ($\beta_{\textrm{EMA}} \approx 0.9993$). Das dämpft zufällige Schwankungen durch Minibatches bei $(t, \boldsymbol\epsilon)$.

<!--en:18509400-->
- **Anpassung von $n_{1/2}$ an das Budget:** Zu kurze Halbwertszeiten erhalten das SGD-Rauschen, zu lange beziehen frühe, noch unzureichend angepasste Gewichte ein.

<!--en:31317eb0-->
::: {.slide title="Bewertung von Sample-Verteilungen mit dem Fréchet-Abstand"}

<!--en:41fe8f78-->
[Wasserstein-2-Abstand im Merkmalsraum des Klassifikators]{.kicker}

<!--en:2fbbd520-->
Der euklidische Abstand im Pixelraum erfasst die wahrgenommene Qualität schlecht (ein um 2 Pixel verschobener Turnschuh ergibt einen großen $L_2$-Fehler). Für den **Fréchet-Abstand (FD)** (Heusel et al., 2017) werden Gauß-Verteilungen $\mathcal{N}(\boldsymbol\mu_p, \boldsymbol\Sigma_p)$ und $\mathcal{N}(\boldsymbol\mu_q, \boldsymbol\Sigma_q)$ an $128$-dimensionale Merkmale eines domänenspezifischen Klassifikators angepasst (`FeatureCNN`, Testgenauigkeit etwa $93\%$):

<!--en:75b2886b-->
::: {.slide title="Kalibrierung des FD: 1/n-Referenzwert für endliche Stichproben" layout="tight"}

<!--en:3d3b2cd3-->
[Warum FD-Werte einen Maßstab brauchen]{.kicker}

<!--en:041cac57-->
- **Warum disjunkte reale Mengen einen FD $> 0$ ergeben:** Stichprobenkovarianzen schwanken um $O(1/\sqrt{n})$. Über $(\boldsymbol\Sigma_p^{1/2}\boldsymbol\Sigma_q\boldsymbol\Sigma_p^{1/2})^{1/2}$ entsteht eine Verzerrung nach oben von $+O(1/n)$.

<!--en:32be8d4e-->
- **Zwei feste Regeln:** (1) FD nur bei gleicher Stichprobengröße $n$ vergleichen (der reale Referenzwert sinkt um den Faktor $5\times$ von $0.87$ bei $n = 1{,}000$ auf $0.17$ bei $n = 5{,}000$). (2) Unterschiede unterhalb der Streubreite bei mehreren Startwerten ignorieren ($\pm 0.05$ bei $n = 5{,}000$).

<!--en:e3c3ff6f-->
::: {.slide title="Kalibrierung des FD: 1/n-Referenzwert für endliche Stichproben (Code)" layout="tight"}

<!--en:3d3b2cd3-2-->
[Warum FD-Werte einen Maßstab brauchen]{.kicker}

<!--en:06e62637-->
::: {.slide title="Gepaarte Ablation: EMA-Gewichte und Kette mit 1000 Schritten"}

<!--en:998be979-->
[Einfluss von EMA und Schrittzahl bei gleichem Rauschen]{.kicker}

<!--en:233c1a32-->
Drei gemeinsame Ziehungen von je $5{,}000$ Rauschtensoren: 50 Euler-Schritte mit EMA-Gewichten (`ema`) und mit den ungemittelten Endgewichten (`net`) sowie 1000 ancestrale Schritte auf dem varianzerhaltenden Pfad (mit `make_pred`):

<!--en:1b1b39f0-->
- **Deutlicher FD-Gewinn von $\approx +0.5$ durch EMA:** Bei jeder Ziehung erreichen die ungemittelten Endgewichte $\approx 1.3$–$1.5$, die EMA-Gewichte dagegen $\approx 0.8$. Der EMA-Wert schwankt bei verschiedenen Startwerten des Rauschens nur um $\pm 0.02$.

<!--en:5cb91fc9-->
- **Ancestrale Kette mit 1000 Schritten verbessert den FD um $\approx 0.15$:** Sie erreicht einen FD von $\approx 0.65$ bei **$20\times$ so hohen Rechenkosten** wie 50 Euler-Schritte. Das motiviert die schnellen Löser höherer Ordnung in §{sec_diffusion-samplers}.

<!--en:f51c25ee-->
::: {.slide title="Bayes-Regel für Scores und Klassifikator-Guidance"}

<!--en:1d6f8365-->
[Gezielte Erzeugung von Bildern einer Klasse]{.kicker}

<!--en:e2747968-->
Wie lassen sich Samples stärker zur Klasse $c$ hin lenken als mit der gewöhnlichen bedingten Verteilung $p_t(\mathbf{x} \mid c)$? Die Bayes-Regel bei Rauschpegel $t$ lautet $p_t(\mathbf{x} \mid c) = p_t(\mathbf{x})\,p_t(c \mid \mathbf{x}) / p(c)$. Anwenden von $\nabla_{\mathbf{x}} \log(\cdot)$ eliminiert die A-priori-Wahrscheinlichkeit $p(c)$:

<!--en:1a255709-->
- **Klassifikator-Guidance (Dhariwal & Nichol, 2021):** Verstärkt den Gradienten der Log-Likelihood eines externen, auf verrauschten Bildern trainierten Klassifikators $p_{\boldsymbol\phi}(c \mid \mathbf{x}, t)$ mit der **Guidance-Skala** $\gamma \ge 0$:
  $$\tilde{\mathbf{s}}(\mathbf{x}, t, c) = \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) + \gamma\,\nabla_{\mathbf{x}} \log p_{\boldsymbol\phi}(c \mid \mathbf{x}, t)$$

<!--en:63c46422-->
- **Nachteil externer Klassifikator-Guidance:** Ein separater, auf den Rauschpegel konditionierter Klassifikator $p_{\boldsymbol\phi}(c \mid \mathbf{x}_t, t)$ benötigt Training für alle $t \in [0, 1]$. Der Aufstieg entlang $\nabla_{\mathbf{x}} \log p_{\boldsymbol\phi}$ kann adversarielle Richtungen des Klassifikators ausnutzen.

<!--en:4d6090d8-->
::: {.slide title="Classifier-Free Guidance (CFG): implizite Steuerung"}

<!--en:bc98af52-->
[Wegfall des externen Klassifikators]{.kicker}

<!--en:d5b0bfed-->
- **Impliziter Klassifikatorgradient nach Bayes (Ho & Salimans, 2022):** Umstellen der Bayes-Regel ergibt $\nabla_{\mathbf{x}}\log p_t(c \mid \mathbf{x}) = \mathbf{s}_c(\mathbf{x}, t) - \mathbf{s}_\varnothing(\mathbf{x}, t)$.

<!--en:e3a4437b-->
- **Invarianz bei jeder Zielumrechnung:** Wegen $(1 - \gamma) + \gamma = 1$ gilt dieselbe lineare Extrapolation für die Vorhersagen von $\hat{\mathbf{x}}_0$, $\hat{\boldsymbol\epsilon}$, $\hat{\mathbf{v}}$, Score und Geschwindigkeit.

<!--en:4726b46f-->
::: {.slide title="Geometrie der CFG und Implementierung mit Batches"}

<!--en:ae0c2513-->
[Extrapolation über den bedingten Mittelwert hinaus]{.kicker}

<!--en:0cb15296-->
Für $\gamma > 1$ extrapoliert $\hat{\mathbf{x}}_0^\gamma = \hat{\mathbf{x}}_0^\varnothing + \gamma(\hat{\mathbf{x}}_0^c - \hat{\mathbf{x}}_0^\varnothing)$ über den A-posteriori-Mittelwert der Klasse hinaus:

<!--en:bf8f983a-->
- **Verdoppelter Batch je Schritt:** Für $\gamma \neq 1$ verkettet `cfg` die Bilder zu $[\mathbf{x}, \mathbf{x}]$ mit den Labels $[c, \varnothing]$. Das verdoppelt den Rechenaufwand je Schritt ($100$ U-Net-Durchläufe für $50$ Euler-Schritte).

<!--en:175d3d29-->
::: {.slide title="Guidance zwischen Vielfalt und Klassentypizität"}

<!--en:960e6173-->
[5000 Samples je Guidance-Skala γ bei gleichem Rauschen]{.kicker}

<!--en:31495441-->
Guidance verbessert die Klassenübereinstimmung auf Kosten der Verteilungsabdeckung (FD):

<!--en:d678f042-->
- **$\gamma = 0$:** Übereinstimmung auf Zufallsniveau (10%), FD nahe $2.7$

<!--en:1f1e380d-->
- **$\gamma \approx 1$:** Niedrigster FD (etwa $0.8$), Übereinstimmung nahe $90\%$ und damit fast so hoch wie die $93\%$ des Merkmalsnetzes auf realen Testbildern

<!--en:96589553-->
- **$\gamma \ge 3$:** Übereinstimmung über $99\%$, aber der FD steigt auf $4$–$6$, weil fast nur noch Klassenprototypen entstehen.

<!--en:084d5e4a-->
::: {.slide title="Unbedingte (γ = 0) und gesteuerte Samples (γ = 4) im Bild" layout="tight"}

<!--en:41a539e2-->
[Klassenprototypen und mehrdeutige Samples]{.kicker}

<!--en:96ef21b9-->
Dieselben $40$ Realisierungen des Anfangsrauschens bei $\gamma = 0$ (links) und $\gamma = 4$ (rechts) zeigen, warum Menschen oft hohe Werte von $\gamma$ trotz schlechteren FD bevorzugen:

<!--en:b5cc4ca2-->
::: {.slide title="Ursachen für den höheren unbedingten FD (γ = 0)" layout="code"}

<!--en:f267ca21-->
[Verschobene Klassenanteile und Sample-Qualität innerhalb der Klassen]{.kicker}

<!--en:6ba71050-->
Warum ist der unbedingte FD ($\gamma = 0$, $\mathrm{FD} \approx 2.75$) mehr als dreimal so hoch wie der bedingte FD ($\gamma = 1$, $\mathrm{FD} \approx 0.82$)? Erklären verschobene Klassenanteile die Differenz?

<!--en:c773f584-->
- **Verschobene Klassenanteile erklären weniger als ein Sechstel der Differenz:** Bei $\gamma = 0$ entstehen zu viele Pullover ($13\%$–$14\%$) und zu wenige Sandalen ($6\%$–$7\%$). Werden $5{,}000$ reale Bilder mit genau diesen Klassenhäufigkeiten gezogen, steigt ihr FD nur um $0.2$–$0.3$ (von $0.19$ auf $0.4$–$0.5$).

<!--en:4d2c2f69-->
- **Hauptursache (mehr als fünf Sechstel der Differenz):** Nur $10\%$ der Beispiele in den Minibatches dienen dem Training ohne Klassenhinweis. Daher erzeugt $\mathbf{u}_{\boldsymbol\theta}(\mathbf{x}_t, t, \varnothing)$ mehr mehrdeutige Mischformen von Kleidungsstücken. Seine Fehler gehen mit Gewicht $(1 - \gamma)$ in jeden gesteuerten Schritt ein.

<!--en:e7b6226a-->
::: {.slide title="Warum gesteuerte ODE-Trajektorien über die Umgewichtung hinausschießen"}

<!--en:5394b5a0-->
[Exakte Analyse für gaußverteilte Klassen]{.kicker}

<!--en:18771401-->
- **Statische Umgewichtung und Überschießen der ODE ($\mu_c = 1, s = 0.5, S^2 = 2, \gamma = 4$):**
  - Bei $t = 0$ verringert die statisch umgewichtete Dichte $\tilde{p}_0 \propto p_0^{1-\gamma}p_0(\cdot \mid c)^\gamma$ die Breite um den Faktor $2\times$ ($0.50 \to 0.26$) und verschiebt den Mittelwert auf $1.10$.
  - Die Integration der gesteuerten ODE über $t = 1 \to 0$ verringert die Breite **um etwa den Faktor $23\times$ auf $0.02$** und verschiebt den Mittelwert auf $1.23$.

<!--en:14115144-->
::: {.slide title="Warum Guidance bei starkem Rauschen die Krümmungsschranke verletzt"}

<!--en:b9009d37-->
[Inkonsistenz des Pfades und neuere Lösungsansätze]{.kicker}

<!--en:7561e99f-->
- **Krümmungsschranke bei gaußscher Faltung:** Jede zulässige Randdichte $p_t = p_0 * \mathcal{N}(\mathbf{0}, \sigma_t^2\mathbf{I})$ erfüllt $-\nabla^2\log p_t(\mathbf{x}) \preceq (1/\sigma_t^2)\mathbf{I}$.

<!--en:6dc208c6-->
- **Verletzung bei starkem Rauschen:** Für $\gamma > S^2/(S^2 - s^2) = 8/7$ übersteigt die gesteuerte inverse Varianz $1/v_t$ die Schranke $1/t^2$ nahe $t = 1$ (Verletzung für $t \in (0.24, 1]$ bei $\gamma = 4$).

<!--en:3920e360-->
- **Neuere Lösungsansätze:** $\gamma > 1$ nur in einem mittleren **Guidance-Intervall** anwenden (Kynkäänniemi et al., 2024) oder **Autoguidance** mit einem schwächeren bedingten Modell als Bezugsgröße nutzen (Karras et al., 2024)

<!--en:7ce32aa6-->
::: {.slide title="Verständnisfragen zu rezeptiven Feldern, FD-Referenzwerten und CFG"}

<!--en:929b7671-->
[Wissenskontrolle]{.kicker}

<!--en:31f6717f-->
Fragen zur Architektur des U-Nets, zur Kalibrierung des Fréchet-Abstands und zur Classifier-Free Guidance:

<!--en:4574e17f-->
1. **Downsampling und rezeptives Feld im U-Net:** Jede $3 \times 3$-Faltung erweitert das rezeptive Feld eines Pixels um $2$ Pixel (auf $1 + 2k$ nach $k$ Schichten). Wie viele Schichten $k$ sind nötig, um $28$ Pixel bei voller Auflösung bzw. am $7 \times 7$-Flaschenhals abzudecken?

<!--en:f1e25516-->
2. **FD bei unterschiedlichen Stichprobengrößen:** Modell A erreicht $\mathrm{FD} = 0.85$ bei $n = 1{,}000$ Bildern, Modell B erreicht $\mathrm{FD} = 0.80$ bei $n = 5{,}000$ Bildern (mit demselben `FeatureCNN`). Welches Modell liegt näher an der realen Verteilung?

<!--en:f5d5ea52-->
3. **Guidance für Gauß-Verteilungen:** Eine Klasse $\mathcal{N}(1, 0.5^2)$ liegt innerhalb von $\mathcal{N}(0, 2)$. Welche Varianz $v$, Standardabweichung $\sqrt{v}$ und welchen Mittelwert ergibt die statische Umgewichtung bei Guidance-Skala $\gamma = 2$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:164913dd-->
1. Bei $28 \times 28$ sind $1 + 2k \ge 28 \implies \mathbf{k = 14}$ Schichten nötig. Bei $7 \times 7$ (jede Position deckt einen $4 \times 4$-Patch ab) genügen $1 + 2k \ge 7 \implies \mathbf{k = 3}$ Schichten.

<!--en:86297328-->
2. **Modell A schneidet deutlich besser ab:** Bei $n = 1{,}000$ beträgt der Referenzwert zwischen realen Mengen $\approx 0.87$. Modell A ($0.85$) ist damit nicht von realen Daten zu unterscheiden. Bei $n = 5{,}000$ liegt der reale Referenzwert bei $\approx 0.17$. Der FD von Modell B ($0.80$) ist also fast $5\times$ so hoch.

<!--en:5ca8b4f9-->
3. $1/v = \gamma/s^2 - (\gamma - 1)/S^2 = 2/0.25 - 1/2 = 7.5 \implies v = 2/15 \approx 0.133$ (**Breite $\sqrt{v} \approx \mathbf{0.37} < 0.5$**) und **$\text{mean} = \gamma v / s^2 = 2(2/15)/0.25 \approx \mathbf{1.07} > 1$**.

<!--en:3da1e225-->
::: {.slide title="Zusammenfassung: Bilddiffusion, Auswertung und Guidance"}

<!--en:c5a222f8-->
[Zentrale Ergebnisse]{.kicker}

<!--en:70902993-->
| Komponente | Mechanismus / Formel | Messergebnis auf Fashion-MNIST | Empfehlung für die Praxis |

<!--en:335f1564-->
| **U-Net zur Entrauschung** | Encoder und Decoder mit $28 \to 14 \to 7$ und Skip-Verbindungen | $80\%$ der Parameter bei niedriger Auflösung, $57\%$ der MACs bei $28 \times 28$ | Viele Kanäle bei niedriger Auflösung vorsehen und GroupNorm verwenden |

<!--en:01ea7032-->
| **EMA der Gewichte** | $\bar{\boldsymbol\theta} \leftarrow \bar{\boldsymbol\theta} + (1 - 2^{-1/n_{1/2}})(\boldsymbol\theta - \bar{\boldsymbol\theta})$ | FD um **$0.4$–$0.6$** niedriger als mit der letzten Iterierten | Zum Sampling stets EMA-Gewichte verwenden ($n_{1/2} \approx 10^3$–$10^4$) |

<!--en:b4ea6e86-->
| **Fréchet-Abstand (FD)** | $\|\boldsymbol\mu_p - \boldsymbol\mu_q\|^2 + \operatorname{tr}(\boldsymbol\Sigma_p + \boldsymbol\Sigma_q - 2(\boldsymbol\Sigma_p^{1/2}\boldsymbol\Sigma_q\boldsymbol\Sigma_p^{1/2})^{1/2})$ | Realer Referenzwert sinkt um den Faktor $5\times$ von $0.87$ ($1\text{k}$) auf $0.17$ ($5\text{k}$) | Immer Stichprobengröße $n$ und Referenzwert zwischen realen Mengen angeben |

<!--en:6cf6f186-->
| **Label-Dropout** | Klassenlabel gemäß $c \leftarrow \varnothing$ mit Wahrscheinlichkeit $p_{\textrm{drop}} = 0.1$ ersetzen | Gemeinsames Training von $\mathbf{u}_c$ ($90\%$) und $\mathbf{u}_\varnothing$ ($10\%$) | $p_{\textrm{drop}} \in [0.1, 0.2]$ genügt für die unbedingte Bezugsgröße |

<!--en:917d3a02-->
| **Classifier-Free Guidance** | $\tilde{\mathbf{u}} = \mathbf{u}_\varnothing + \gamma(\mathbf{u}_c - \mathbf{u}_\varnothing)$ | Bei $\gamma = 1.25$ steigt die Übereinstimmung gemäß $89\% \to 94\%$ bei unverändertem FD | Schwache Guidance mit $\gamma \approx 1.25$ für niedrigen FD, $\gamma > 2$ für Klassentypizität |
