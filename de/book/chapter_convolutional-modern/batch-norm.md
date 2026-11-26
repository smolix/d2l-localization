<!--en:d8eab9a9-->
# Normalisierungsschichten

<!--en:e56249eb-->
Als wir LeNet in :numref:`sec_lenet` trainierten, versagte die Lernrate (engl. learning rate), mit der sich ein MLP problemlos trainieren ließ. Bei 0.1 blieb die Validierungsgenauigkeit mehrere Epochen lang auf Zufallsniveau und erreichte nach zehn Epochen nur etwa 73%. Für rund 89% mussten wir die Lernrate auf das Zwanzigfache, also auf 2.0, erhöhen und 30 Epochen lang trainieren. Ebenso empfindlich reagierten die Netze in :numref:`sec_blocks`. Mit einer maximalen Lernrate von 0.1 ließ sich VGG problemlos trainieren, während das Training von NiN und GoogLeNet ins Stocken geriet oder divergierte. Ein Netz mit fünf Schichten sollte auf die Schrittweite weniger empfindlich reagieren. Warum hängt die Wahl einer geeigneten Lernrate so stark von der Architektur ab? Können wir eine Schicht in das Netz einfügen, die diese Empfindlichkeit beseitigt?

<!--en:e8b34625-->
Die Ursache liegt darin, dass sich die Größenordnungen von Signalen und Gradienten auf ihrem Weg durch die Schichten verschieben. Jede der vier Sigmoidschichten von LeNet verkleinert die rückwärts propagierten Gradienten auf höchstens einen Anteil von $1/4$ ihres bisherigen Werts. Die Gewichte der frühen Schichten ändern sich deshalb nur wenig im Vergleich zu denen der späteren Schichten. Keine einzelne Lernrate passt zum ganzen Netz. Mit zunehmender Tiefe verschärft sich das Problem, denn jede Schicht skaliert die Aktivierungen für die nächste Schicht neu. ReLU-Aktivierungen und eine sorgfältige Initialisierung (:numref:`subsec_he_init`) gleichen die Varianzen der Aktivierungen in Schritt null aus. Sobald sich die Gewichte durch die Gradientenschritte ändern, gibt es jedoch nichts mehr, was diese Größenordnungen im Gleichgewicht hält.

<!--en:e37f2d69-->
Die *Batchnormalisierung* (engl. batch normalization) :cite:`Ioffe.Szegedy.2015` löst dieses Problem während des gesamten Trainings. Unmittelbar vor der nichtlinearen Aktivierungsfunktion einer Schicht standardisiert sie jedes Merkmal anhand von Mittelwert und Varianz des aktuellen Minibatchs. Anschließend skaliert und verschiebt sie es mit gelernten Parametern. Diese einfache Transformation stabilisiert sowohl die Aktivierungen im Vorwärtsdurchlauf als auch die Gradienten im Rückwärtsdurchlauf. Zugleich macht sie jede normalisierte Schicht invariant gegenüber einer Umskalierung der Gewichte. Wie wir gleich sehen, lässt sich LeNet mit Batchnormalisierung bei der Lernrate des MLP und sogar über einen Bereich von Schrittweiten trainieren, die sich um den Faktor 300 unterscheiden. Zusammen mit den Residualverbindungen des nächsten Abschnitts ermöglichte die Batchnormalisierung Netze mit mehr als hundert Schichten. Welche Mechanismen ihre empirischen Erfolge im Einzelnen erklären, ist weiterhin umstritten. Da die Batchnormalisierung die Beispiele eines Minibatchs miteinander koppelt, verliert sie bei sehr kleinen Minibatches an Wirksamkeit. In der zweiten Hälfte dieses Abschnitts stellen wir die *Schichtnormalisierung* (engl. layer normalization) und die *Gruppennormalisierung* (engl. group normalization) vor, die jedes Beispiel für sich normalisieren.

<!--en:22ec41e3-->
## Batchnormalisierung

<!--en:8b3ff613-->
### Standardisierung über einen Minibatch

<!--en:e224ffde-->
Betrachten wir ein einzelnes Merkmal vor der Aktivierung. Eine vollständig verbundene Schicht berechnet dafür in einem Minibatch $\mathcal{B}$ mit $m$ Beispielen die skalaren Werte $x_1, \ldots, x_m$. Die Batchnormalisierung berechnet deren Stichprobenmittelwert und Stichprobenvarianz über den Minibatch,

<!--en:3cc0f4e3-->
und transformiert jeden Wert gemäß

<!--en:14583b99-->
Der Bruch standardisiert das Merkmal über den Minibatch auf den Mittelwert null und die Varianz $\hat{\sigma}^2_\mathcal{B} / (\hat{\sigma}^2_\mathcal{B} +
\epsilon)$, die knapp unter 1 liegt. Die kleine Konstante $\epsilon > 0$ (typischerweise $10^{-5}$) verhindert eine Division durch null, wenn alle Einträge im Minibatch übereinstimmen. Bei mehreren Merkmalen normalisiert die Schicht jedes Merkmal unabhängig mit dessen eigenem Minibatch-Mittelwert und dessen eigener Minibatch-Varianz.

<!--en:8a3b4680-->
Rechnen wir das an einem Beispiel nach. Die vier Werte $(2, 4, 6, 8)$ haben den Mittelwert 5 und die Varianz $(9 + 1 + 1 + 9)/4 = 5$. Die Standardisierung ergibt bis auf den Einfluss von $\epsilon$ die Werte $(x_i - 5)/\sqrt{5} \approx (-1.34, -0.45, 0.45, 1.34)$. Auch $(20, 40, 60, 80)$ und $(102, 104, 106, 108)$ ergeben genau diese vier Zahlen. Die Standardisierung entfernt sowohl die beliebige Skalierung als auch die Verschiebung des Eingangsmerkmals.

<!--en:9120362f-->
Mittelwert null und Varianz eins sind jedoch nicht für jede nachgeschaltete Aktivierungsfunktion ideal. Die Sigmoidfunktion etwa verhält sich bei Eingaben mit Varianz eins nahezu linear. Damit das Netz wieder alle Darstellungen lernen kann, führt :eqref:`eq_batchnorm` für jedes Merkmal einen lernbaren *Skalierungsfaktor* $\gamma$ und eine lernbare *Verschiebung* $\beta$ ein. Wir initialisieren sie mit $\gamma = 1$ und $\beta = 0$ und trainieren sie zusammen mit den Gewichten des Netzes. Setzen wir $\gamma$ auf die wahre Standardabweichung und $\beta$ auf den wahren Mittelwert des Datensatzes, erhalten wir bis auf Stichprobenschwankungen im Minibatch das ursprüngliche, unnormalisierte Merkmal zurück. Die Batchnormalisierung schränkt die Ausdruckskraft des Netzes also nicht ein.

<!--en:b651f2b7-->
### Skaleninvarianz

<!--en:cf7aaf74-->
Dass bei der Standardisierung die ursprüngliche Skalierung eines Merkmals verschwindet, hat zwei wichtige Folgen für die Optimierung. Die erste erklärt unmittelbar, warum die Batchnormalisierung bei LeNet hilft. Variiert ein Merkmal innerhalb eines Minibatchs nur wenig, etwa weil die Ausgaben der Sigmoidfunktionen dicht bei $1/2$ liegen, so stellt die Division durch seine kleine Standardabweichung $\hat{\sigma}_\mathcal{B}$ im Vorwärtsdurchlauf die Varianz eins wieder her. Nach der Kettenregel verstärkt sie den rückwärts propagierten Gradienten um denselben Faktor $1/\hat{\sigma}_\mathcal{B}$. Verschwindende Signale und verschwindende Gradienten werden also automatisch neu skaliert.

<!--en:23531496-->
Die zweite Folge ist weniger offensichtlich und betrifft die Gewichte der Schicht selbst. Schreiben wir $x_i = \mathbf{w}^\top \mathbf{h}_i + b$ für die Präaktivierung von Beispiel $i$. Multiplizieren wir $\mathbf{w}$ und $b$ mit einer beliebigen positiven Konstanten $a > 0$, so multiplizieren wir damit jedes $x_i$ sowie den Mittelwert und die Standardabweichung dieser Werte mit $a$. Die normalisierte Ausgabe bleibt also gleich, wenn wir $\epsilon$ vernachlässigen:

<!--en:a30e3f63-->
Damit hängt der Verlust nur von der *Richtung* von $\mathbf{w}$ ab und ist unabhängig von der euklidischen Norm. Leiten wir beide Seiten nach $\mathbf{w}$ ab, so sehen wir, dass der Gradient an der Stelle $a \mathbf{w}$ das $1/a$-Fache des Gradienten an der Stelle $\mathbf{w}$ beträgt :cite:`Ioffe.Szegedy.2015`. Kleine Gewichte erhalten also größere Gradienten, große Gewichte kleinere.

<!--en:e2a9f2c0-->
Was geschieht, wenn wir eine zu hohe Lernrate $\eta$ wählen? Da der Verlust auf dem Strahl vom Ursprung durch $\mathbf{w}$ konstant ist, steht der Gradient $\mathbf{g}$ senkrecht auf $\mathbf{w}$. Mit jedem SGD-Schritt wächst die Gewichtsnorm daher strikt: $\|\mathbf{w} - \eta \mathbf{g}\|^2 = \|\mathbf{w}\|^2 + \eta^2 \|\mathbf{g}\|^2$. Ein zu großer Schritt lässt $\|\mathbf{w}\|$ rasch anwachsen. Dadurch schrumpfen alle nachfolgenden Gradienten um denselben Faktor. Bei großen Lernraten stabilisiert sich das Training durch diese Rückkopplung von selbst :cite:`Arora.Li.Lyu.2019`.

<!--en:c1c22d9c-->
Diese Analyse hat Grenzen. Streng genommen setzt sie einfaches SGD ohne Gewichtszerfall voraus. Mit Gewichtszerfall erreichen die Gewichtsnormen ein Gleichgewicht, in dem die Gradientenschritte die Normen ebenso stark vergrößern, wie der Gewichtszerfall sie verkleinert. Die Analyse gilt außerdem nur für Schichten, auf die unmittelbar eine Normalisierung folgt, also etwa nicht für die letzte Ausgabeschicht eines Klassifikators. Auch kann die Skaleninvarianz in den ersten Schritten, bevor die Gewichtsnormen anwachsen, eine Divergenz nicht verhindern. Trotz dieser Einschränkungen liefert sie uns die klarste Erklärung dafür, warum normalisierte Netze einen so viel größeren Bereich von Lernraten vertragen.

<!--en:5a5aaace-->
### Vollständig verbundene Schichten und Faltungsschichten

<!--en:6e69d93b-->
In einer vollständig verbundenen Schicht mit Gewichten $\mathbf{W}$, Bias $\mathbf{b}$ und nichtlinearer Aktivierungsfunktion $\phi$ liegt die Batchnormalisierung zwischen der affinen Transformation und der Aktivierungsfunktion, wie von :citet:`Ioffe.Szegedy.2015` vorgeschlagen:

<!--en:5692d9dc-->
Jede Ausgabeeinheit gilt hier als eigenes Merkmal mit eigenem Minibatch-Mittelwert, eigener Varianz und eigenen Parametern $\gamma_j$ und $\beta_j$.

<!--en:0b65ca2f-->
Bei einer Faltungsschicht erhält die Batchnormalisierung die Translationsäquivarianz, indem sie jeden *Kanal* der Ausgabe als ein Merkmal behandelt. Sie berechnet für ihn einen Mittelwert und eine Varianz über die $m$ Beispiele des Minibatchs *und* alle $p \times q$ räumlichen Positionen ($m \cdot p \cdot q$ Werte pro Kanal). In der ersten Faltungsschicht von LeNet (6 Kanäle der Größe $28 \times 28$ bei Batchgröße 128) gehen in die Statistik jedes Kanals $128 \cdot 28 \cdot 28 = 100{,}352$ Einträge ein. Das liefert wesentlich stabilere Schätzungen als die 128 Einträge pro Einheit in einer vollständig verbundenen Schicht. Ebenso verwendet jeder Kanal dieselben skalaren Parameter $\gamma$ und $\beta$ an allen räumlichen Positionen. Dadurch bleibt die normalisierte Schicht translationsäquivariant.

<!--en:8ac5cc0f-->
### Training und Vorhersage

<!--en:875adeaa-->
Bei der Inferenz verarbeitet ein Modell im praktischen Einsatz häufig nur ein Bild auf einmal. Seine Vorhersage für eine gegebene Eingabe muss deterministisch sein, unabhängig davon, welche anderen Bilder zufällig im selben Minibatch liegen. Die Batchnormalisierung hat deshalb zwei Betriebsarten. Im *Trainingsmodus* normalisiert sie wie in :eqref:`eq_batchnorm` anhand der Statistik des aktuellen Minibatchs. Nach jedem Schritt aktualisiert sie die exponentiellen gleitenden Mittelwerte $\hat{\mu}$ und $\hat{\sigma}^2$, die anfangs auf 0 beziehungsweise 1 gesetzt sind:

<!--en:07007bfa-->
Der *Abklingfaktor* $\rho$ bestimmt das effektive Zeitfenster dieser gleitenden Schätzungen. Ein Minibatch, der $k$ Schritte zurückliegt, hat das Gewicht $(1 - \rho)\rho^k$. Bei $\rho = 0.9$ halbieren sich die Gewichte etwa alle sieben Schritte. Die Mittelung erfasst damit effektiv etwa die letzten $1/(1 - \rho) = 10$ Minibatches (von 469 Minibatches der Größe 128 pro Fashion-MNIST-Epoche). Im *Vorhersagemodus* hält die Schicht diese Schätzwerte fest und normalisiert damit jede Eingabe. Die Batchnormalisierung vereinfacht sich dann zu einer festen, deterministischen affinen Transformation. Wie Dropout (:numref:`sec_dropout`) verhält sie sich also beim Training anders als bei der Auswertung.

<!--en:7fa1e9f4-->
Achten Sie auf einen leicht zu übersehenden Unterschied zwischen den Programmierschnittstellen der Bibliotheken. Einige Frameworks bezeichnen den Parameter $\rho$ als `momentum`, andere bezeichnen $1 - \rho$ als `momentum`. Mit dem Momentum bei der Optimierung hat keiner der beiden Parameter etwas zu tun. In diesem Kapitel verwenden wir einheitlich $\rho = 0.9$ und $\epsilon = 10^{-5}$ und übergeben diese Werte ausdrücklich, wenn ein Framework andere Standardwerte hat.

<!--en:6275dcac-->
## Implementierung von Grund auf

<!--en:82349654-->
Die folgende Funktion `batch_norm` implementiert :eqref:`eq_batchnorm` und :eqref:`eq_bn_running` für den Trainings- und den Vorhersagemodus. Sie verarbeitet sowohl Eingaben für vollständig verbundene Schichten (2D) als auch Eingaben für Faltungsschichten (4D).

<!--en:36418d60-->
Die Bilder sind in der Reihenfolge (Beispiele, Kanäle, Höhe, Breite) gespeichert. Die Kanalachse ist also Achse 1.

<!--en:739885b6-->
Die Bilder sind in der Reihenfolge (Beispiele, Höhe, Breite, Kanäle) gespeichert. Die Kanalachse ist also die letzte Achse.

<!--en:de3e3273-->
Wenn wir `batch_norm` in eine Schicht namens `BatchNorm` einbetten, müssen wir zwei Arten von Zustand verwalten. Die lernbaren Parameter (`gamma` und `beta`) aktualisiert der Optimierer. Die nicht trainierbaren Puffer (`moving_mean` und `moving_var`) ignoriert er. Das Modul speichert sie dennoch zusammen mit den Gewichten auf dem Datenträger und überträgt sie mit ihnen auf GPUs.

<!--en:dcb6990a-->
`autograd.is_training()` liefert innerhalb von `autograd.record()` den Wert wahr, da der Trainer jeden Trainingsschritt aufzeichnet. Der Modus braucht daher kein eigenes Flag. Die gleitenden Schätzwerte sind Parameter mit `grad_req='null'` und erhalten keinen Gradienten.

<!--en:1716b73d-->
`register_buffer` registriert einen Tensor, den `model.to(device)` verschiebt und `state_dict()` speichert. Bei einem Aufruf von `parameters()` wird er jedoch nicht zurückgegeben. Das Attribut `self.training` setzen die Methoden `model.train()` und `model.eval()`, die der Trainer vor dem Training beziehungsweise vor der Validierung aufruft.

<!--en:02781d76-->
`add_weight(..., trainable=False)` erzeugt Variablen, die die Schicht speichert und der Optimierer ignoriert. Bei Trainingsschritten übergibt Keras `training=True` an `call`.

<!--en:b37493b3-->
NNX speichert die gleitenden Schätzwerte in Variablen vom Typ `BatchStat`. Der Optimierer ignoriert sie, da er nur Variablen vom Typ `Param` aktualisiert. Der Trainer erzeugt mit `deterministic=False` eine Trainingsansicht des Modells und mit `deterministic=True` eine Vorhersageansicht. Dieses Flag wird an `set_view` übergeben.

<!--en:41d61185-->
Um unsere Implementierung zu prüfen, wenden wir `BatchNorm` und die eingebaute Schicht des Frameworks (jeweils mit $\rho = 0.9$ und $\epsilon = 10^{-5}$) auf denselben Minibatch an. Er enthält 128 Merkmalskarten der Größe $28 \times 28$ mit je sechs Kanälen, deren Werte aus einer Verteilung mit Mittelwert 2 und Standardabweichung 3 stammen. Wir prüfen, ob jeder Ausgabekanal den Mittelwert 0 und die Varianz 1 hat, ob unsere Ausgabe mit der Ausgabe der eingebauten Schicht übereinstimmt und ob die gleitenden Mittelwerte nach einem Schritt korrekt aktualisiert sind.

<!--en:a9c1549e-->
Bis auf Rundungsfehler bei einfacher Genauigkeit hat nun jeder Kanal den Mittelwert null und die Varianz eins. Unsere Implementierung stimmt bis auf $10^{-6}$ mit der eingebauten Schicht überein. Nach einem Minibatch haben die gleitenden Mittelwerte den Anteil $1 - \rho = 0.1$ der Strecke von 0 zum Minibatch-Mittelwert 2 zurückgelegt.

<!--en:867943c9-->
`nn.BatchNorm2d(6)` verwendet den PyTorch-Standardwert `momentum=0.1`. Das entspricht unserem $\rho = 0.9$.

<!--en:43014826-->
## LeNet mit Batchnormalisierung

<!--en:a445abe5-->
Kehren wir zu LeNet (:numref:`sec_lenet`) zurück und erproben wir die Batchnormalisierung. Bei unveränderter Xavier-Initialisierung fügt `BNLeNet` nach jeder der vier versteckten Schichten von LeNet eine eingebaute Batchnormalisierungsschicht ein, unmittelbar vor der jeweiligen Sigmoidfunktion. Das betrifft die Faltungen mit 6 und 16 Kanälen und die vollständig verbundenen Schichten mit 120 und 84 Einheiten. In diesen vier Schichten deaktivieren wir den Bias. Das Subtrahieren des Minibatch-Mittelwerts hebt jeden additiven Bias unmittelbar vor der Batchnormalisierung vollständig auf, und $\beta$ übernimmt seine Rolle. Für alle $6 + 16 + 120 + 84 = 226$ normalisierten Merkmale zusammen ersetzen wir also 226 Bias-Terme durch 226 Paare $(\gamma, \beta)$. Zu den 61 706 trainierbaren Parametern von LeNet kommen damit nur 226 hinzu.

<!--en:6a087edb-->
Gluons `BatchNorm` verwendet standardmäßig `momentum=0.9`, was $\rho$ entspricht, und `epsilon=1e-5`. Mit `init.Xavier()` werden die Gewichte der Faltungsschichten und der vollständig verbundenen Schichten aus der Xavier-Gleichverteilung gezogen. Skalierungsfaktor und Verschiebung jeder Batchnormalisierungsschicht bleiben bei 1 und 0.

<!--en:42205d61-->
`LazyBatchNorm2d` normalisiert jeden Kanal einer Eingabe mit vier Achsen, `LazyBatchNorm1d` jedes Merkmal einer Eingabe mit zwei Achsen. Ihre Standardwerte `momentum=0.1` und `eps=1e-5` entsprechen unseren Werten für $\rho = 0.9$ und $\epsilon$. `BNLeNet` verwendet die Funktion `weight_init` von LeNet. Nur die Ausgabeschicht hat einen Bias, den sie auf null setzen kann.

<!--en:5733f14b-->
Die Keras-Schicht `BatchNormalization` verwendet standardmäßig `momentum=0.99` und `epsilon=1e-3`. Daher übergeben wir $\rho = 0.9$ und $\epsilon = 10^{-5}$. Ihr Parameter `momentum` entspricht wie in unserer Implementierung dem Wert $\rho$. Xavier-Initialisierung (`glorot_uniform`) und ein Bias von null für die Ausgabeschicht sind bei Keras Standard.

<!--en:41658cf6-->
Die Flax-Schicht `nnx.BatchNorm` verwendet standardmäßig `momentum=0.99`. Deshalb übergeben wir $\rho = 0.9$. Für `epsilon` gilt der Standardwert $10^{-5}$. Außerdem berechnet die Schicht die Varianz standardmäßig als Mittelwert der Quadrate minus Quadrat des Mittelwerts. Dabei geht Genauigkeit verloren, wenn der Mittelwert im Vergleich zur Standardabweichung groß ist. Mit `use_fast_variance=False` zieht die Schicht wie unsere Implementierung den Mittelwert vor dem Quadrieren ab. Für das Padding verwendet Flax standardmäßig `'SAME'`. Daher geben wir für die zweite Faltung `'VALID'` an. Die zu einem Vektor umgeformte Ausgabe hat dann $16 \cdot 5 \cdot 5 = 400$ Einträge.

<!--en:e5fbde30-->
Vor dem Training lohnt ein Blick darauf, wie der Vorhersagemodus und die Lernrate zusammenwirken. Die gleitenden Mittelwerte hinken um etwa zehn Minibatches hinterher. Solange sich die Gewichte rasch ändern, beschreiben diese Mittelwerte also einen etwas älteren Zustand des Netzes. Das zeigt sich auch in der Praxis. Bei einer konstanten Lernrate von 0.1 kann die Validierungsgenauigkeit von `BNLeNet` im Vorhersagemodus in einzelnen Epochen plötzlich um 10 Prozentpunkte oder mehr einbrechen, auch in der letzten Epoche. Senken wir die Lernrate allmählich auf null, löst sich das Problem. Mit abnehmender Schrittweite stabilisieren sich die Gewichte, und die gleitenden Mittelwerte passen sich dem endgültigen Netz an. Wir verwenden deshalb erneut den Kosinus-Lernratenplan mit Aufwärmphase aus :numref:`sec_alexnet`, den wir in :numref:`chap_optimization` genauer untersuchen. Die Aufwärmphase dauert das erste Zehntel von `num_steps`. Wir leiten von `d2l.LeNet` die Unterklasse `CosineLeNet` ab, damit beide Modelle genau denselben Lernratenplan verwenden.

<!--en:74f1f1c6-->
Trainieren wir `CosineLeNet` und `BNLeNet` unter identischen Bedingungen (maximale Lernrate 0.1, Batchgröße 128 und 10 Epochen) und verfolgen wir die Validierungsgenauigkeit von Epoche zu Epoche.

<!--en:ac6f42a5-->
Zeichnen wir die in `curves` gespeicherte Validierungsgenauigkeit über die Epochen auf, können wir direkt vergleichen, wie schnell `CosineLeNet` und `BNLeNet` konvergieren.

<!--en:e5d7e7da-->
Für beide Modelle geben wir außerdem die abschließende Validierungsgenauigkeit `acc[-1]` aus `curves` aus. So können wir die Ergebnisse nach 10 Epochen vergleichen.

<!--en:a2d5d289-->
Ohne Batchnormalisierung verharrt LeNet in den ersten zwei bis fünf Epochen auf Zufallsniveau (10%) und erreicht nach zehn Epochen nur rund 60%. Das ist weniger als die 73% in :numref:`sec_lenet`, vermutlich weil der Kosinus-Abfall die durchschnittliche Schrittweite halbiert. Mit Batchnormalisierung übertrifft dieselbe Architektur schon in der ersten Epoche 70% und erreicht nach zehn Epochen rund 88%. Das kommt den 88–90% nahe, für die LeNet ohne Normalisierung die zwanzigfache Lernrate und dreimal so viele Epochen brauchte. Beachten Sie auch, wie gleichmäßig die Kurve in den letzten Epochen abflacht, während die Lernrate auf null sinkt. Die 226 zusätzlichen Parameter haben die Ausdruckskraft des Netzes nicht erhöht. Sie haben lediglich die Aktivierungen und Gradienten jeder Schicht in einen günstigen Größenbereich gebracht.

<!--en:356e3195-->
Der Vergleich bei einer einzigen Lernrate lässt jedoch eine wichtige Frage offen. Verschiebt die Batchnormalisierung nur die optimale Schrittweite oder vergrößert sie tatsächlich den Bereich brauchbarer Lernraten? Dazu variieren wir die maximale Lernrate von 0.03 bis 10, also um einen Faktor von rund 300. Für beide Modelle zeichnen wir die Validierungsgenauigkeit nach 10 Epochen auf.

<!--en:e97c29d2-->
Die Versuchsreihe gibt eine klare Antwort. LeNet ohne Normalisierung lernt nur im engen Bereich von etwa 1 bis 3 erfolgreich und erreicht dort 84–87%. Bei kleineren Lernraten lernt es in 10 Epochen zu langsam. Bei 10 wird das Training unzuverlässig und bleibt mitunter auf Zufallsniveau stehen. Dagegen übertrifft `BNLeNet` bei *jeder* Lernrate von 0.03 bis 10 die Marke von rund 85%. Ab einer Lernrate von 1 erreicht es etwa 91%, ganze drei Prozentpunkte mehr als bei 0.1. Beim unnormalisierten Netz darf die Lernrate also nur um einen Faktor von drei variieren. Das normalisierte Netz lernt über einen Bereich mit Faktor 300 zuverlässig, wie unsere Analyse der Skaleninvarianz vorhersagt. Da die letzte Ausgabeschicht unnormalisiert bleibt, muss es allerdings auch hier eine Obergrenze geben.

<!--en:736faea4-->
Die interne Kovariatenverschiebung (engl. internal covariate shift) :cite:`Ioffe.Szegedy.2015` erklärt nur zum Teil, warum die Batchnormalisierung tiefen Netzen so sehr hilft. Spätere Untersuchungen zeigten, dass sie die Verlustlandschaft und die Gradienten glättet :cite:`Santurkar.Tsipras.Ilyas.ea.2018` und die Größenordnungen von Aktivierungen und Gradienten über die Schichten hinweg im Gleichgewicht hält. Sie macht die Gewichte versteckter Schichten skaleninvariant, sodass sich das Training bei hohen Lernraten selbst stabilisiert. Außerdem erzeugt sie ein leichtes Minibatch-Rauschen, das ähnlich wie Dropout regularisiert :cite:`Luo.Wang.Shao.ea.2018,brock2021nfnet`.

<!--en:172ed380-->
## Normalisierung ohne Batchstatistik

<!--en:02ca3136-->
Dass die Ausgabe für ein Beispiel von den anderen Beispielen im selben Minibatch abhängt, ist für eine Schicht eines neuronalen Netzes ungewöhnlich. Daraus ergeben sich zwei praktische Schwächen. Erstens werden die Schätzwerte aus dem Minibatch mit sinkender Batchgröße unzuverlässiger. Im Extremfall einer vollständig verbundenen Schicht mit nur einem Beispiel im Minibatch ist jeder Eintrag gleich seinem eigenen Mittelwert. Jede standardisierte Aktivierung ist damit 0, und die Schicht gibt unabhängig von der Eingabe den konstanten Wert $\beta$ aus. Bei speicherintensiven Aufgaben wie Objekterkennung und semantischer Segmentierung passen häufig nur ein oder zwei hochauflösende Bilder auf eine GPU. Gerade dann rauschen die Batchstatistiken am stärksten. Zweitens berechnet die Batchnormalisierung im Trainings- und im Vorhersagemodus unterschiedliche Funktionen. Passen die gleitenden Schätzwerte nicht mehr zu den Eingaben beim Test, etwa nach einem Fine-Tuning für eine andere Domäne, kann die Genauigkeit unbemerkt sinken. Die Ursache ist dann oft schwer zu finden.

<!--en:97dc7ae3-->
Beide Einschränkungen entfallen, wenn wir die Statistiken für die Normalisierung *für jedes Beispiel unabhängig von den anderen* berechnen. :numref:`fig_norm_axes` vergleicht die so entstehenden Normalisierungsschichten anhand des Aktivierungstensors einer Faltungsschicht mit Batchachse $N$, Kanalachse $C$ und räumlichen Achsen $H \cdot W$. Alle Varianten verwenden dieselbe Formel zur Standardisierung. Sie unterscheiden sich allein darin, welche Einträge sie für einen gemeinsamen Mittelwert und eine gemeinsame Varianz zusammenfassen.

<!--en:9204beca-->
![Normalisierungsverfahren für Aktivierungen mit $N$ Beispielen, $C$ Kanälen und $H\cdot W$ räumlichen Positionen. Für die schattierten Zahlen gelten derselbe Mittelwert und dieselbe Varianz. Die Batchnormalisierung mittelt für jeden Kanal über den Minibatch und alle Positionen. Die Schichtnormalisierung mittelt für jedes Beispiel über alle Kanäle und Positionen, die Gruppennormalisierung über eine Gruppe von Kanälen und alle Positionen, die Instanznormalisierung über die Positionen eines einzelnen Kanals. Die Schichtnormalisierung bei Transformern und ConvNeXt mittelt über die Kanäle an einer Position.](../img/arch-norm-axes.svg)

<!--en:166c5c32-->
### Schichtnormalisierung

<!--en:c131c077-->
Die *Schichtnormalisierung* :cite:`Ba.Kiros.Hinton.2016` standardisiert für jedes Beispiel unabhängig von den anderen alle $n$ Merkmale seines Merkmalsvektors $\mathbf{x} = (x_1, \ldots, x_n)$:

<!--en:c45dacf4-->
Das entspricht :eqref:`eq_batchnorm` mit vertauschten Achsen. Statt für ein Merkmal über die Beispiele zu mitteln, mitteln wir für ein Beispiel über seine Merkmale. Dabei behalten wir den lernbaren Skalierungsfaktor $\gamma_j$ und die lernbare Verschiebung $\beta_j$ pro Merkmal bei. Da kein anderes Beispiel in die Rechnung eingeht, verhält sich die Schichtnormalisierung bei jeder Batchgröße sowie im Trainings- und Vorhersagemodus gleich. Sie braucht keinerlei gleitende Schätzwerte. Über welche Achsen die „Schichtnormalisierung“ in der Praxis mittelt, hängt von der Architektur ab. Bei Tensoren in Faltungsschichten kann sie über alle Kanäle und räumlichen Positionen eines Beispiels mitteln (zweites Teilbild in :numref:`fig_norm_axes`). Transformer hingegen normalisieren den Merkmalsvektor jedes Tokens unabhängig. Ebenso normalisiert ConvNeXt (:numref:`sec_convnext`) den Kanalvektor an jedem Pixel für sich, ohne über räumliche Positionen zu mitteln (Teilbild ganz rechts).

<!--en:fdefa096-->
### Gruppen- und Instanznormalisierung

<!--en:a2b632d1-->
Die *Gruppennormalisierung* :cite:`wu2018groupnorm` ist speziell auf Faltungsschichten zugeschnitten. Sie teilt die $c$ Kanäle in $G$ Gruppen mit je $c/G$ Kanälen auf und standardisiert jedes Beispiel über die $(c/G) \cdot p \cdot q$ Einträge der jeweiligen Gruppe. Anschließend wendet sie pro Kanal einen gelernten Skalierungsfaktor und eine gelernte Verschiebung an. Die Gruppenzahl erlaubt einen fließenden Übergang zwischen zwei Extremen. Mit $G = 1$ mitteln wir über alle Kanäle und Positionen (räumliche Schichtnormalisierung in :numref:`fig_norm_axes`). Mit $G = c$ normalisieren wir jeden Kanal jedes Beispiels für sich. Das heißt *Instanznormalisierung* (engl. instance normalization). Bei mittleren Gruppenzahlen (typischerweise $G = 32$) enthält jede Gruppe genügend Elemente für zuverlässige Schätzwerte, ohne dass voneinander unabhängige Merkmalskanäle denselben Mittelwert und dieselbe Varianz verwenden müssen. Im Folgenden prüfen wir an einem Batch aus 32 Tensoren (16 Kanäle, 4 Gruppen), ob jeder Slice `(example, group)` auf Mittelwert null und Varianz eins standardisiert wird. Außerdem prüfen wir, ob sich für das erste Beispiel allein genau dieselbe Ausgabe ergibt wie im vollständigen Batch.

<!--en:66985a96-->
Wie erwartet hat jede Gruppe bis auf Rundungsfehler den Mittelwert null und die Varianz eins. Die Ausgabe für das erste Beispiel ist vom Rest des Batchs völlig unabhängig. Kostet diese Unabhängigkeit Genauigkeit? Bei großen Batchgrößen nur wenig. Für ResNet-50 auf ImageNet mit 32 Bildern pro GPU berichten :citet:`wu2018groupnorm` eine Top-1-Fehlerrate von 23.6% mit Batchnormalisierung gegenüber 24.1% mit Gruppennormalisierung. Bei nur 2 Bildern pro GPU springt die Fehlerrate mit Batchnormalisierung jedoch auf 34.7%, während sie mit Gruppennormalisierung bei 24.1% bleibt. Deshalb ist die Gruppennormalisierung bei Modellen zur Objekterkennung und Segmentierung, die mit kleinen Batches pro GPU trainiert werden, das übliche Verfahren.

<!--en:7737984e-->
### Netze ohne Normalisierungsschichten

<!--en:1d14a901-->
Auch ohne Normalisierung der Aktivierungen lassen sich die Signalgrößen in tiefen konvolutionären Netzen stabil halten. Mit *Netzen ohne Normalisierungsschichten* (engl. normalizer-free networks, NFNets) :cite:`brock2021nfnet` erreichten tiefe Architekturen nach dem Vorbild von ResNet ohne eine einzige Normalisierungsschicht auf ImageNet eine der höchsten Genauigkeiten unter den ausschließlich auf ImageNet trainierten Modellen. NFNets standardisieren direkt die Gewichte jedes Faltungskerns. So erhalten sie während des gesamten Trainings das Gleichgewicht der Varianzen, das die Initialisierung (:numref:`subsec_he_init`) nur in Schritt null herstellt. Sie kombinieren dies mit adaptivem Gradienten-Clipping relativ zur Norm des jeweiligen Gewichtstensors. Wir implementieren NFNets hier nicht, doch ihr Erfolg ist aufschlussreich. Tiefe konvolutionäre Netze lassen sich ebenso gut trainieren, wenn die Signalgröße über die Gewichte statt über die Aktivierungen geregelt wird. Das spricht dafür, dass die Regelung der Signalgröße die zentrale Aufgabe einer Normalisierungsschicht ist.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3391a45d-->
Die Batchnormalisierung standardisiert jedes Merkmal einer vollständig verbundenen Schicht beziehungsweise jeden Kanal einer Faltungsschicht anhand von Mittelwert und Varianz des aktuellen Minibatchs. Anschließend wendet sie einen gelernten Skalierungsfaktor und eine gelernte Verschiebung an. Sie hält die Signale im Vorwärtsdurchlauf und die Gradienten im Rückwärtsdurchlauf in einem günstigen Größenbereich und macht normalisierte Schichten invariant gegenüber Änderungen der Gewichtsnorm. So lässt sich LeNet mit maximalen Lernraten von 0.03 bis 10 zuverlässig trainieren. Das entspricht einem Faktor von 300 gegenüber einem Faktor von 3 ohne Batchnormalisierung. Eine vollständige theoretische Erklärung für ihren Erfolg in tiefen Netzen steht noch aus. Bei der Inferenz verwendet die Batchnormalisierung die nun festgehaltenen exponentiellen gleitenden Mittelwerte der Trainingsstatistiken. Weil die Kopplung der Beispiele bei sehr kleinen Batches versagt, berechnen Schichtnormalisierung und Gruppennormalisierung ihre Statistiken ausschließlich innerhalb jedes einzelnen Beispiels. Damit entfallen sowohl die Empfindlichkeit gegenüber der Batchgröße als auch die Unterscheidung zwischen Trainings- und Vorhersagemodus. Im nächsten Abschnitt kombinieren wir Batchnormalisierung mit Residualverbindungen, um Netze mit mehr als hundert Schichten zu trainieren.

<!--en:2294243a-->
## Übungen

<!--en:d11bcf56-->
1. **Bias vor der Batchnormalisierung.** `BNLeNet` baut seine vier normalisierten Schichten ohne Bias auf. Zeigen Sie anhand von :eqref:`eq_batchnorm`, dass ein zusätzlicher Bias $\mathbf{b}$ vor der Batchnormalisierung weder die Ausgabe im Trainingsmodus noch die Gradienten der Gewichte ändert und dass $\partial L / \partial \mathbf{b} = \mathbf{0}$ gilt. Wie wirkt sich ein solcher Bias im Vorhersagemodus aus?

<!--en:e2e993ff-->
1. [code] **Obergrenze der Lernrate.** In der Versuchsreihe dieses Abschnitts ließ sich `BNLeNet` bei jeder maximalen Lernrate bis 10 problemlos trainieren.
    1. Erweitern Sie die Versuchsreihe für `BNLeNet` um die maximalen Lernraten 30, 100 und 300. Ab wann versagt das Training?
    1. Die lineare Ausgabeschicht ist die einzige Schicht mit Gewichten, auf die keine Batchnormalisierung folgt. Prüfen Sie, ob sie die Obergrenze bestimmt. Halten Sie dazu die Lernrate der Ausgabeschicht bei 0.1 fest und testen Sie für die normalisierten Schichten maximale Lernraten von 30, 100 und 300.
    1. Ersetzen Sie die Sigmoidfunktionen in `CosineLeNet` und `BNLeNet` durch ReLUs mit He-Initialisierung und wiederholen Sie die Versuchsreihe. Wie verändert sich der Bereich brauchbarer Lernraten beim unnormalisierten Netz und warum?

<!--en:e4dcb016-->
1. [code] **Gleitende Mittelwerte und neu berechnete Statistiken.** Der Abklingfaktor $\rho$ in :eqref:`eq_bn_running` bestimmt, über wie viele zurückliegende Minibatches die gleitenden Schätzwerte effektiv mitteln.
    1. Nach wie vielen Schritten fällt das Gewicht eines Minibatchs im gleitenden Mittelwert bei $\rho = 0.9$ und bei $\rho = 0.99$ unter 1% des Gewichts des neuesten Minibatchs? Vergleichen Sie beide Zeitspannen mit den 469 Schritten einer Fashion-MNIST-Epoche.
    1. Trainieren Sie `BNLeNet` für 10 Epochen mit einer konstanten Lernrate von 0.1 (`d2l.Classifier.configure_optimizers`). Werten Sie das Modell nach jeder Epoche zweimal aus, einmal im Vorhersagemodus mit seinen gleitenden Mittelwerten und einmal, nachdem Sie bei festgehaltenen Gewichten die Statistiken über 100 Trainings-Minibatches neu berechnet haben. Zeichnen Sie beide Validierungskurven zusammen mit der Kurve für den Kosinus-Lernratenplan und erklären Sie den Abstand.

<!--en:0090525b-->
1. [code] **Kopplung im Minibatch.** Die Batchnormalisierung koppelt die Beispiele eines Minibatchs miteinander. Werten Sie deshalb ein trainiertes `BNLeNet` auf dem Validierungsdatensatz auf drei Arten aus: (i) im Vorhersagemodus, (ii) im Trainingsmodus mit Validierungs-Batchgröße 128 und (iii) im Trainingsmodus mit Validierungs-Batchgröße 2 (oder 1). Messen Sie zuerst im Vorhersagemodus, damit die Durchläufe im Trainingsmodus die gleitenden Schätzwerte nicht vorher überschreiben. Erklären Sie die drei resultierenden Genauigkeiten.

<!--en:79dc5ec7-->
1. [code] **Training mit kleinen Batches: BatchNorm im Vergleich zu GroupNorm und LayerNorm.** ● Bauen Sie eine Variante von `BNLeNet`, die `BatchNorm` nach den beiden Faltungsschichten durch Gruppennormalisierung (2 Gruppen) und nach den beiden versteckten vollständig verbundenen Schichten durch Schichtnormalisierung (:numref:`subsec_layer-normalization-in-bn`) ersetzt. Trainieren Sie sowohl `BNLeNet` als auch diese Variante jeweils 2 Epochen lang mit Batchgröße 4 und mit Batchgröße 128. Wie ändert sich die Validierungsgenauigkeit der beiden Modelle, wenn die Batchgröße von 128 auf 4 sinkt, und warum?

<!--en:bb9c5cac-->
1. [code] **Rückwärtsdurchlauf der Batchnormalisierung.** ● Da $\hat{\mu}_\mathcal{B}$ und $\hat{\sigma}^2_\mathcal{B}$ in :eqref:`eq_batchnorm` von jedem Element des Minibatchs abhängen, koppelt der Rückwärtsdurchlauf der Batchnormalisierung alle $m$ Beispiele miteinander.
    1. Leiten Sie von Hand den Gradienten $\partial L / \partial x_i$ eines skalaren Verlusts $L$ nach jeder Eingabe $x_i$ einer eindimensionalen Batchnormalisierungsschicht her. Zeigen Sie anhand Ihrer Formel, dass $\sum_{i=1}^m \partial L / \partial x_i = 0$ identisch gilt.
    1. Implementieren Sie Ihre Formel für den Rückwärtsdurchlauf und prüfen Sie, ob sie bis auf Maschinengenauigkeit mit dem Ergebnis des automatischen Differenzierens übereinstimmt.

<!--en:47c258e3-->
[Diskussion](https://d2l.discourse.group/t/83)

<!--en:53204b95-->
[Diskussion](https://d2l.discourse.group/t/84)

<!--en:d0e38981-->
[Diskussion](https://d2l.discourse.group/t/330)

<!--en:035119cf-->
[Diskussion](https://d2l.discourse.group/t/18005)

<!--en:57d0028c-->
Normalisierungsschichten<br>**Standardisierung im Minibatch · Skaleninvarianz und Selbststabilisierung bei hohen Lernraten · Trainings- und Vorhersagemodus · LayerNorm und GroupNorm**

<!--en:430c6eb6-->
::: {.slide title="Warum unnormalisierte Netze empfindlich auf die Lernrate reagieren"}

<!--en:d6899680-->
[Motivation und Drift der Größenordnungen]{.kicker}

<!--en:1fb35e28-->
- **Drift bei Sigmoidfunktion und ReLU:** Bei $\eta = 0.1$ stagniert LeNet ($\approx 60\%$). Es lernt nur im engen Bereich $3\times$ ($\eta_0 \in [1, 3]$). NiN und GoogLeNet divergieren bei $\eta_0 = 0.1$.

<!--en:355c0085-->
- **Die Initialisierung wirkt nur in Schritt 0:** Sobald Gradientenschritte die Gewichte ändern, driften die Größenordnungen der Aktivierungen und Gradienten ohne Normalisierung auseinander.

<!--en:ab784ce0-->
::: {.slide title="Batchnormalisierung über einen Minibatch"}

<!--en:6f14e397-->
[Mathematische Formulierung]{.kicker}

<!--en:ac889cd0-->
- **Momente des Minibatchs:** $\hat{\mu}_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m x_i$ und $\hat{\sigma}^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m (x_i - \hat{\mu}_\mathcal{B})^2$ ($\epsilon = 10^{-5}$).

<!--en:b1a56146-->
- **Affine Transformation ($\gamma, \beta$):** Lernbarer Skalierungsfaktor (anfangs $1$) und lernbare Verschiebung (anfangs $0$) pro Merkmal erhalten die Ausdruckskraft.

<!--en:19d8c7d8-->
::: {.slide title="Standardisierung am Beispiel und warum ein Bias vor der Batchnormalisierung verschwindet"}

<!--en:b8b322d9-->
[Invarianz gegenüber affinen Verschiebungen]{.kicker}

<!--en:a24eef5e-->
- **Verschiebungs- und Skaleninvarianz:** $(2, 4, 6, 8)$, $(20, 40, 60, 80)$ und $(102, 104, 106, 108)$ ergeben nach der Standardisierung jeweils $(-1.34, -0.45, 0.45, 1.34)$.

<!--en:e0ad67ea-->
- **Warum `bias=False` vor der Batchnormalisierung genügt:** In $x_i = \mathbf{w}^\top \mathbf{h}_i + b$ verschiebt der Bias $b$ den Mittelwert $\hat{\mu}_\mathcal{B}$ um $+b$ und fällt in $x_i - \hat{\mu}_\mathcal{B}$ vollständig weg ($\partial L / \partial b = 0$).

<!--en:d718880e-->
::: {.slide title="Skaleninvarianz und Selbststabilisierung bei hohen Lernraten"}

<!--en:4fa35305-->
[Optimierungstheorie]{.kicker}

<!--en:6234647c-->
- **Orthogonale Gradienten ($\mathbf{w}^\top \mathbf{g} = 0$):** Nach dem Satz des Pythagoras gilt $\|\mathbf{w} - \eta \mathbf{g}\|^2 = \|\mathbf{w}\|^2 + \eta^2 \|\mathbf{g}\|^2$.

<!--en:27414d8a-->
- **Selbststabilisierung bei hohen Lernraten:** Ein großes $\eta$ lässt $\|\mathbf{w}\|$ wachsen. Das dämpft nachfolgende Gradienten automatisch mit dem Faktor $1/\|\mathbf{w}\|$.

<!--en:1b3a4b35-->
::: {.slide title="Batchnormalisierung in vollständig verbundenen Schichten und Faltungsschichten"}

<!--en:80a97033-->
[Räumliche Äquivarianz]{.kicker}

<!--en:3bbb8e85-->
- **Vollständig verbundene Schichten ($m \times d$):** Mittelung über Achse `0` ($m$ Beispiele pro Merkmal, Form $1 \times d$).

<!--en:c8cc1b00-->
- **Faltungsschichten ($m$ Beispiele, $c$ Kanäle, $p \times q$ Pixel):** Mittelung über die Batchachse und beide Raumachsen ($m \cdot p \cdot q$ Werte pro Kanal, eine Statistik pro Kanal) erhält die räumliche Translationsäquivarianz ($100{,}352$ Werte pro Kanal in der ersten Faltungsschicht von LeNet).

<!--en:46d80023-->
::: {.slide title="Trainingsmodus, Vorhersagemodus und gleitende Mittelwerte"}

<!--en:e2c40921-->
[Zwei Betriebsarten]{.kicker}

<!--en:8111d88a-->
- **Training und Auswertung:** Das Training nutzt die aktuelle Batchstatistik $(\hat{\mu}_\mathcal{B}, \hat{\sigma}^2_\mathcal{B})$ und aktualisiert die Puffer. Bei der Auswertung bleiben $(\hat{\mu}, \hat{\sigma}^2)$ für deterministische Vorhersagen mit $N=1$ fest.

<!--en:a923807b-->
- **Stolperfalle:** Die BatchNorm-Schichten der Frameworks nennen das Mischgewicht `momentum`. Einige definieren es als $1-\rho$ (z. B. $0.1$), andere als $\rho$ (z. B. $0.9$). Bei unterschiedlichen Standardwerten das Mischgewicht ausdrücklich angeben.

<!--en:9a2c3acb-->
::: {.slide title="Batchnormalisierung von Grund auf implementieren"}

<!--en:c8bf2621-->
[Implementierung von Grund auf]{.kicker}

<!--en:16345209-->
Unsere Funktion `batch_norm` unterscheidet die Modi anhand von `training`. Bei vollständig verbundenen Schichten mittelt sie über Achse `0`, bei Faltungsschichten über die Achsen `(0, 2, 3)` (bzw. `(0, 1, 2)` bei hinten liegender Kanalachse):

<!--en:9e945765-->
::: {.slide title="Eine BatchNorm-Schicht mit eigenem Zustand implementieren"}

<!--en:e36546be-->
[Trainierbare Parameter und Puffer für gleitende Schätzwerte]{.kicker}

<!--en:857d6950-->
`BatchNorm` trennt die lernbaren Parameter `gamma` und `beta` von den Puffern für die gleitenden Schätzwerte `moving_mean` und `moving_var`:

<!--en:4f9f2937-->
::: {.slide title="Vergleich mit der eingebauten Schicht des Frameworks"}

<!--en:1b718c47-->
[Numerische Überprüfung]{.kicker}

<!--en:bc7266f6-->
::: {.slide title="LeNet um Batchnormalisierung erweitern (BNLeNet)"}

<!--en:0854bfd4-->
[Kompakte Implementierung]{.kicker}

<!--en:87676cc6-->
BatchNorm vor den vier versteckten Sigmoidfunktionen von LeNet erhöht die Parameterzahl um nur $226$ gegenüber den ursprünglichen $61{,}706$:

<!--en:f9435a20-->
::: {.slide title="Warum der Kosinus-Abfall die gleitenden Schätzwerte stabilisiert"}

<!--en:224db193-->
[Zusammenspiel von Lernratenplan und gleitenden Schätzwerten]{.kicker}

<!--en:e1d844af-->
Der Vergleich von `CosineLeNet` und `BNLeNet` über 10 Epochen bei maximaler Lernrate $\eta_0 = 0.1$ zeigt, wie viel schneller das normalisierte Netz konvergiert und warum der Lernratenabfall für die Batchnormalisierung entscheidend ist:

<!--en:9dc5b527-->
- **Sofortige Konvergenz bei $\eta_0 = 0.1$:** LeNet ohne Normalisierung bleibt mehrere Epochen auf Zufallsniveau (10%) und erreicht am Ende $\approx 60\%$. Dagegen übertrifft `BNLeNet` schon in Epoche 1 die Marke von $70\%$ und erreicht **$\approx 88\%$**.

<!--en:d74675bf-->
- **Verzögerung der gleitenden Schätzwerte ausgleichen:** Bei konstantem $\eta = 0.1$ ändern sich die Gewichte sprunghaft. Die gleitenden Mittelwerte hinken um $\sim 10$ Schritte hinterher, die Validierungsgenauigkeit bricht um 10 Prozentpunkte ein. Bei $\eta_t \to 0$ stabilisieren sich die Gewichte, sodass sich $(\hat{\mu}, \hat{\sigma}^2)$ dem endgültigen Netz anpassen.

<!--en:993a9787-->
::: {.slide title="Nutzbare Lernraten über einen Faktor von 300"}

<!--en:50374458-->
[Skaleninvarianz im Experiment]{.kicker}

<!--en:bd00240a-->
Jeweils 10 Epochen mit maximaler Lernrate $\eta_0 \in \{0.03, 0.1, 0.3, 1, 3, 10\}$ trainieren. Die Ergebnisse bestätigen die Vorhersage der Skaleninvarianz: Das Training stabilisiert sich selbst.

<!--en:ee5c1d97-->
- **Unnormalisiertes LeNet (Bereich $3\times$):** Lernt nur im engen Bereich um $\eta_0 \in [1, 3]$ erfolgreich ($84\text{--}87\%$). Unterhalb von $1$ sind die Gradienten zu schwach. Bei $10$ werden die Trainingsläufe unzuverlässig oder stagnieren auf Zufallsniveau.

<!--en:3b88f3fe-->
- **Normalisiertes `BNLeNet` (Bereich $300\times$):** Übertrifft $\approx 85\%$ bei **jeder** maximalen Lernrate von $0.03$ bis $10$ und erreicht $\approx 91\%$ für $\eta_0 \ge 1$. Bei noch höheren Lernraten setzt allein die unnormalisierte lineare Ausgabeschicht eine Grenze.

<!--en:85f544ff-->
::: {.slide title="Wann Batchstatistiken versagen: kleine Batches und Sequenzen"}

<!--en:cb93cc31-->
[Grenzen der Kopplung im Minibatch]{.kicker}

<!--en:94305245-->
- **Zusammenbruch bei Kleinstbatches ($m \le 2$):** In einer vollständig verbundenen Schicht gilt bei $m=1$ die Beziehung $x_1 = \hat{\mu}_\mathcal{B} \implies \textrm{BN}(x_1) \equiv \beta$ (Gradient null). Bei 2 Bildern pro GPU steigt die Fehlerrate von ResNet-50 mit Batchnormalisierung auf ImageNet von $23.6\%$ auf **$34.7\%$**.

<!--en:554f16a8-->
- **Drift zwischen Training und Test, Sequenzen:** Beim Fine-Tuning für eine andere Domäne verschieben sich die gleitenden Schätzwerte. Bei kausalen autoregressiven Sequenzen kann man weder über Token noch über Beispiele mitteln.

<!--en:47f93a49-->
::: {.slide title="Schicht-, Gruppen- und Instanznormalisierung"}

<!--en:4de0874f-->
[Normalisierung einzelner Beispiele]{.kicker}

<!--en:1032c542-->
Alle Alternativen verwenden dieselbe Formel $\gamma (x - \hat{\mu})/\sqrt{\hat{\sigma}^2 + \epsilon} + \beta$, berechnen $(\hat{\mu}, \hat{\sigma}^2)$ aber *ausschließlich innerhalb jedes einzelnen Beispiels* (keine Mittelung über die Achse $N$):

<!--en:e54f0cf6-->
- **Schichtnormalisierung (LN):** Normalisiert innerhalb eines Beispiels über alle Kanäle oder Merkmale (bei Transformern und ConvNeXt über die $C$ Kanäle jedes einzelnen Tokens bzw. Pixels).

<!--en:e4885e2f-->
- **Gruppennormalisierung (GN):** Teilt $C$ Kanäle in $G$ Gruppen (typischerweise $G=32$) und mittelt pro Beispiel über $(C/G) \times H \times W$.

<!--en:9a7d90b8-->
- **Instanznormalisierung (IN):** Grenzfall $G = C$. Jede einzelne räumliche Karte zum Indexpaar $(N, C)$ wird über $H \times W$ normalisiert.

<!--en:0da434fa-->
::: {.slide title="Gruppennormalisierung bleibt bei 2 Bildern pro GPU stabil"}

<!--en:9ea88d7b-->
[Unabhängigkeit von der Batchgröße]{.kicker}

<!--en:c304e362-->
Die Überprüfung von `GroupNorm` im Code und auf ImageNet zeigt, warum Verfahren zur Normalisierung einzelner Beispiele ohne gleitende Mittelwerte auskommen und auch bei kleinen Batchgrößen ihre Leistung halten:

<!--en:2cfcf3f3-->
- **Exakte Invarianz für einzelne Beispiele:** `gn(X[:1])` liefert genau denselben Tensor wie `gn(X)[:1]` (kein Unterschied zwischen Training und Auswertung).

<!--en:6f56dd9b-->
- **Top-1-Fehlerrate von ResNet-50 auf ImageNet (Wu und He, 2018):**
  - Bei **32 Bildern/GPU:** BatchNorm $23.6\%$ gegenüber GroupNorm $24.1\%$ (Vorteil von $0.5\%$ durch das Minibatch-Rauschen bei BN).
  - Bei **2 Bildern/GPU:** Die Fehlerrate bei BatchNorm springt auf **$34.7\%$**, bei GroupNorm bleibt sie bei **$24.1\%$**.

<!--en:c4df3146-->
::: {.slide title="Netze ohne Normalisierungsschichten und direkte Skalierung der Gewichte"}

<!--en:a43607cf-->
[Ein anderer Weg zur Regelung der Signalgröße]{.kicker}

<!--en:5a3958a2-->
- **NFNets (Brock et al., 2021):** Standardisieren die Filtergewichte direkt und verwenden adaptives Gradienten-Clipping ($\|\mathbf{G}_l\|_F / \|\mathbf{W}_l\|_F \le \lambda$).

<!--en:2eec1f98-->
- **Fazit:** Die Signalgröße über die Schichten hinweg zu erhalten, erweist sich als zentrale Aufgabe der Normalisierung.

<!--en:bf7bbaf0-->
::: {.slide title="Verständnisfragen zu Normalisierungsachsen, Puffern und Fehlern bei der Auswertung"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:aaceeb94-->
Die folgenden drei Aufgaben zur Normalisierung lösen:

<!--en:94b02109-->
1. **Anzahl der Elemente pro Mittelwert:** Eine Faltungsschicht verarbeitet $N = 16$ Beispiele mit $C = 64$ Kanälen und $H \times W = 32 \times 32$ räumlichen Positionen. Über wie viele skalare Aktivierungen wird für einen einzelnen Mittelwert $\hat{\mu}$ gemittelt, und zwar (a) bei der 2D-Batchnormalisierung, (b) bei der Gruppennormalisierung ($G = 16$) und (c) bei der Schichtnormalisierung über die Kanäle eines Pixels (wie in ConvNeXt)?

<!--en:06591bb6-->
2. **Trainierbare Parameter und Puffer:** Wie viele trainierbare Parameter und wie viele nicht trainierbare Pufferwerte für die gleitenden Schätzwerte kommen durch eine 2D-Batchnormalisierungsschicht mit 64 Kanälen hinzu?

<!--en:f668323d-->
3. **Trainingsmodus bei $N=1$:** Was geschieht, wenn ein vollständig verbundenes Netz mit 1D-Batchnormalisierung versehentlich im Trainingsmodus mit Batchgröße $N=1$ ausgewertet wird?

<!--en:557b98b6-->
**Lösungen:**

<!--en:92f01a3e-->
1. **(a) BatchNorm:** mittelt über $(N, H, W) \implies 16 \cdot 32 \cdot 32 = 16{,}384$ Elemente. **(b) GroupNorm ($G=16$):** mittelt über $(C/G, H, W) = (4, 32, 32) \implies 4{,}096$ Elemente. **(c) LayerNorm pro Pixel:** mittelt über $C \implies 64$ Elemente.

<!--en:bd9a94bf-->
2. **128 trainierbare Parameter** ($\gamma, \beta \in \mathbb{R}^{64}$) und **128 nicht trainierbare Pufferwerte** (`moving_mean`, `moving_var` $\in \mathbb{R}^{64}$).

<!--en:c9a2bc81-->
3. Jeder Merkmalswert ist sein eigener Mittelwert ($x_1 - \hat{\mu}_\mathcal{B} = 0$). Jede normalisierte Schicht liefert daher unabhängig vom Eingabebild die Konstante $\beta$ (oder das Framework löst einen `ValueError` aus).

<!--en:af89cf6a-->
::: {.slide title="Normalisierungsschichten im Überblick"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:d137458d-->
- **BatchNorm (BN):** Mittelt über $(N, H, W)$ pro Kanal $C$. Verwendet bei der Auswertung die gleitenden Schätzwerte $(\hat{\mu}, \hat{\sigma}^2)$ und lässt sich in die Faltungsgewichte einrechnen ($N \ge 16$).

<!--en:4914ee94-->
- **GroupNorm (GN, $G=32$):** Mittelt über $(C/G, H, W)$ pro Beispiel $N$. Unabhängig von der Batchgröße bei Objekterkennung und Segmentierung ($N=1\text{--}2$).

<!--en:94e5df68-->
- **LN pro Pixel und InstanceNorm:** LN mittelt über $(C,)$ pro Token oder Pixel (ViT, ConvNeXt), IN über $(H, W)$ pro Kanal (Stiltransfer).
