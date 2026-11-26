<!--en:8e665c76-->
# Initialisierung

<!--en:5a0276f0-->
Jedes neuronale Netz, das wir bisher trainiert haben, begann mit zufälligen Gewichten. Meist hat sie der Standardinitialisierer des Frameworks gezogen. Wie empfindlich das Netz auf die Anfangswerte reagiert, haben wir in :numref:`sec_numerical_stability` gesehen. Verstärkt oder dämpft jede Schicht die Aktivierungen auch nur geringfügig, potenziert sich dieser Faktor mit der Tiefe. Das Training stockt, bevor es überhaupt begonnen hat.

<!--en:6ce14970-->
Dort haben wir auch eine Regel für eine einzelne Schicht hergeleitet. Wir ziehen Gewichte mit der Standardabweichung $g/\sqrt{n_\textrm{in}}$. Dabei ist $n_\textrm{in}$ die Eingabebreite der Schicht. Der *Verstärkungsfaktor* (engl. gain) $g$ gleicht den Einfluss der nachfolgenden Aktivierungsfunktion aus. Vor einer ReLU gilt $g = \sqrt{2}$ (He-Initialisierung, :numref:`subsec_he_init`), bei einer linearen Ausgabe gilt $g = 1$. So erhält jede Schicht das mittlere Quadrat (engl. mean square) ihrer Eingabe.

<!--en:290b36c7-->
Man könnte erwarten, dass diese Regel für jede Schicht die Frage der Initialisierung ein für alle Mal klärt. Überraschenderweise bleibt ein Problem. In einem Residualnetz explodieren die Aktivierungen mit zunehmender Tiefe selbst dann, wenn jede einzelne Schicht nach He initialisiert ist. Jeder Residualblock addiert die Ausgabe seines Residualzweigs $f$ (:numref:`sec_model_construction`) zur Eingabe. Erhält der Zweig das mittlere Quadrat der Eingabe und ist seine Ausgabe mit der Eingabe unkorreliert, hat die Summe ein *doppelt* so großes mittleres Quadrat. Über 32 Residualblöcke wächst die Standardabweichung der Aktivierungen um den Faktor $2^{16} \approx 65{,}000$, noch vor dem ersten Gradientenschritt.

<!--en:9d316d5d-->
Eine geeignete Initialisierung hängt also neben den einzelnen Schichttypen auch davon ab, *wie die Schichten in der Architektur verschaltet sind*. Wir brauchen deshalb ein einfaches Verfahren, um eigene Regeln auf einen ganzen Modulbaum anzuwenden. In der ersten Hälfte dieses Abschnitts geht es um die Umsetzung. Wir wenden Initialisierer auf einen ganzen Modulbaum an, begrenzen extreme Werte mit gestutzten Normalverteilungen (engl. truncated normals), schreiben eigene Initialisierer und ändern Gewichte direkt im Speicher. Nehmen Sie sich für die zweite Hälfte Zeit. Dort untersuchen wir, wie die Varianz von Residualblock zu Residualblock wächst. Wir begrenzen dieses Wachstum, indem wir die Gewichte der Ausgabeprojektion jedes Blocks verkleinern oder auf null setzen.

<!--en:d794788b-->
## Anwendung von Initialisierern

<!--en:9f3f4f2a-->
Wie wir in :numref:`sec_numerical_stability` gesehen haben, verwenden lineare Schichten standardmäßig keine He-Initialisierung (:numref:`subsec_he_init`). Um ein ganzes Modell nach He zu initialisieren, durchlaufen wir den Modulbaum (:numref:`sec_model_construction`). Dasselbe Verfahren eignet sich für die tiefenabhängige Skalierung der Residualzweige, die wir in Kürze herleiten. Wir vergleichen die Standardinitialisierung mit der He-Initialisierung an einem Stapel aus 10 ReLU-Schichten der Breite 256 ohne Bias. Als Eingabe dienen 1024 standardnormalverteilte Vektoren.

<!--en:09971a39-->
Gluon zieht die Gewichte standardmäßig aus $U(-0.07, 0.07)$. Der Aufruf `net.initialize(init.MSRAPrelu(factor_type='in', slope=0), force_reinit=True)` wendet die He-Initialisierung ($g = \sqrt{2}$) auf den gesamten Blockbaum an und überschreibt dabei die vorhandenen Gewichte.

<!--en:2330a90c-->
PyTorch zieht die Gewichte standardmäßig aus $U(-1/\sqrt{n_\textrm{in}}, 1/\sqrt{n_\textrm{in}})$. Die Varianz beträgt ein Sechstel der Varianz bei He-Initialisierung. Mit `net.apply(init_he)` rufen wir `init_he` für jedes Untermodul auf. Die Funktion `nn.init.kaiming_normal_` mit dem abschließenden Unterstrich überschreibt die Gewichte direkt, sodass vorhandene Verweise auf `nn.Parameter` gültig bleiben.

<!--en:90400139-->
Keras verwendet standardmäßig die Glorot-Initialisierung mit Gleichverteilung. Um ein bestehendes Keras-Modell neu zu initialisieren, durchlaufen wir `model.layers` und weisen mit `assign` direkt neu gezogene Werte zu. Für jede Schicht erzeugen wir mit `initializer_class()` eine neue Instanz. Verwenden wir eine Keras-Initialisiererinstanz ohne vorgegebenen Startwert erneut, liefert sie dieselben Zahlen.

<!--en:2509bf53-->
NNX verwendet standardmäßig die LeCun-Initialisierung mit Normalverteilung. Die Varianz ist halb so groß wie bei der He-Initialisierung. Wir durchlaufen `nnx.iter_modules(model)`, spalten für jede `nnx.Linear`-Schicht einen neuen Zufallsschlüssel ab und aktualisieren `layer.kernel[...]` direkt mit `nnx.initializers.he_normal()`.

<!--en:f7fa322f-->
Bei der Standardinitialisierung sinkt das mittlere Quadrat der Ausgabe um Größenordnungen, weil jede Schicht die Varianz mit einem Faktor kleiner als eins multipliziert. Mit He-Initialisierung weicht das mittlere Quadrat der Ausgabe nur noch um etwa einen Faktor zwei von 1 ab, dem mittleren Quadrat der Eingabe. Die verbleibende Streuung entsteht allein durch zufällige Schwankungen aufgrund der endlichen Schichtbreite, die sich über zehn Schichten hinweg multiplizieren.

<!--en:a0c185f9-->
Mehr verlangt die Schnittstelle nicht. In späteren Kapiteln fassen wir dieses Muster in Hilfsfunktionen zusammen, etwa in `init_cnn` für konvolutionäre Netze. Jede davon ist wie `init_he` eine Funktion, die wir an `apply` übergeben.

<!--en:68e70fa3-->
### Gestutzte Normalverteilungen

<!--en:031a2497-->
Eine gewöhnliche Gauß-Verteilung liefert die gewünschte Varianz, lässt aber beliebig große Beträge zu. Bei einer kleinen Schicht spielt das selten eine Rolle. Unter den $10^8$ Gewichten eines großen Modells liegen jedoch mit an Sicherheit grenzender Wahrscheinlichkeit Dutzende mehr als fünf Standardabweichungen vom Mittelwert entfernt. Um extreme Ausreißer bei der Initialisierung zu vermeiden, ziehen moderne Sprach- und Bildmodelle wie BERT und der Vision Transformer ihre Gewichte aus einer *gestutzten Normalverteilung*. Liegt der Betrag eines Samples über einem festen Vielfachen von $\sigma$, wird es verworfen und neu gezogen :cite:`Devlin.Chang.Lee.ea.2018,Dosovitskiy.Beyer.Kolesnikov.ea.2021`. Durch die Stutzung sinkt die Varianz leicht. Einige Frameworks gleichen das durch erneutes Skalieren aus, andere übernehmen die geringere Varianz.

<!--en:dde7d87f-->
Da `mxnet.init` keine gestutzte Normalverteilung anbietet, ziehen wir Einträge außerhalb von $\pm 2\sigma$ in einer Schleife neu.

<!--en:865db9a3-->
`nn.init.trunc_normal_` erwartet *absolute* Grenzen `a` und `b` (standardmäßig $\pm 2$). Für eine Stutzung bei $\pm 2\sigma$ und `std=0.02` müssen wir daher `a=-0.04, b=0.04` übergeben.

<!--en:e7538756-->
`tf.keras.initializers.TruncatedNormal` stutzt bei $\pm 2\sigma$ ohne anschließende Skalierung. `HeNormal` hingegen stutzt und skaliert anschließend auf die gewünschte Varianz.

<!--en:c81d9fca-->
`nnx.initializers.truncated_normal` erwartet die Grenzen als Vielfache von $\sigma$ (standardmäßig `lower=-2.0, upper=2.0`).

<!--en:3070ac10-->
Unter einer Million normalverteilter Gewichte mit $\sigma = 0.02$ erreichen die extremsten Gewichte annähernd $5\sigma = 0.1$. Bei der gestutzten Normalverteilung gilt dagegen garantiert $|w| \leq 0.04$. Durch die Stutzung sinkt die empirische Standardabweichung von $0.02$ auf etwa $0.0176$. Verfahren mit festem $\sigma$ wie bei BERT nehmen diese leichte Abnahme hin. Varianzerhaltende Verfahren skalieren nach der Stutzung erneut und erreichen so genau die gewünschte Varianz.

<!--en:571b69e3-->
## Eigene Initialisierer und direkte Änderungen

<!--en:9cf8b1b0-->
Wenn die eingebauten Initialisierer den Vorgaben einer Publikation oder einer Architektur nicht entsprechen, können wir einen eigenen Initialisierer schreiben oder die Parametertensoren direkt ändern. Beide Verfahren nutzen die Initialisiererschnittstelle des jeweiligen Frameworks.

<!--en:6250ce76-->
In Gluon ist ein Initialisierer eine Unterklasse von `init.Initializer`. Ihre Methode `_init_weight` befüllt das übergebene Array direkt.

<!--en:a1fe1c6a-->
In PyTorch ist ein Initialisierer eine Funktion, die einen Parameter direkt im Speicher überschreibt.

<!--en:c8d611cc-->
In Keras ist ein Initialisierer ein Objekt, das `(shape, dtype)` auf einen Tensor abbildet. Dazu leiten wir eine Klasse von `tf.keras.initializers.Initializer` ab und implementieren `__call__`. Den Schichten übergeben wir dann eine Instanz über ihre Initialisiererargumente.

<!--en:aa45a973-->
In NNX kann jede Funktion mit der Signatur `(key, shape, dtype)
-> array` als Initialisierer dienen.

<!--en:902e89a0-->
### Eine Regel mit frei wählbarem Verstärkungsfaktor

<!--en:e90c85b3-->
Wir implementieren nun die Regel aus der Einleitung. Für die Gewichte verwenden wir eine Normalverteilung mit der Standardabweichung $g/\sqrt{n_\textrm{in}}$ und einem frei wählbaren Verstärkungsfaktor $g$. Die He-Initialisierung entspricht $g = \sqrt{2}$. In unseren Experimenten mit Residualnetzen testen wir für die Ausgabeprojektion jedes Blocks außerdem $g = 1$, $g = 1/\sqrt{N}$ (bei $N$ Blöcken) und $g = 0$. Ein einziger wiederverwendbarer Initialisierer deckt alle vier Fälle ab.

<!--en:3c1cfdd3-->
Die Unterklasse speichert den Verstärkungsfaktor. Ein Gewichtstensor in Gluon hat die Form $(n_\textrm{out}, n_\textrm{in})$.

<!--en:d2114cd4-->
Die Funktion folgt der Konvention von `torch.nn.init`. Sie überschreibt den übergebenen Tensor, und ihr Name endet mit einem Unterstrich. Ein Gewichtstensor in PyTorch hat die Form $(n_\textrm{out}, n_\textrm{in})$.

<!--en:708ae98b-->
Die Unterklasse speichert den Verstärkungsfaktor. Eine Gewichtsmatrix in Keras hat die Form $(n_\textrm{in}, n_\textrm{out})$.

<!--en:a166fe8a-->
Eine Funktion nimmt den Verstärkungsfaktor entgegen und gibt den Initialisierer zurück. Eine Gewichtsmatrix in NNX hat die Form $(n_\textrm{in}, n_\textrm{out})$.

<!--en:1621841d-->
### Direkte Änderungen

<!--en:394186ab-->
Gelegentlich möchten wir Parameterwerte direkt ändern, etwa unmittelbar nach der Initialisierung oder nach dem Laden von Gewichten. Da die ReLU positiv homogen ist, halbiert sich jede Ausgabeaktivierung, wenn wir die Gewichte der letzten Schicht unseres zehnschichtigen Stapels halbieren. Das mittlere Quadrat der Ausgabe sollte daher auf ein Viertel sinken.

<!--en:d09c81f1-->
`weight.data()[:] *= 0.5` ändert das zugrunde liegende Array direkt, außerhalb von `autograd.record()`.

<!--en:b59385f9-->
Führen Sie direkte Änderungen an `nn.Parameter`-Tensoren, die Blätter im Rechengraphen sind, innerhalb von `with torch.no_grad():` aus, damit Autograd sie zulässt.

<!--en:1ca11600-->
`w.assign(0.5 * w)` aktualisiert die `tf.Variable` direkt außerhalb jedes `GradientTape`.

<!--en:f1fbd632-->
`kernel[...] *= 0.5` ändert das Array in der von NNX verfolgten Variablen vom Typ `nnx.Param`.

<!--en:fc0e538e-->
Tatsächlich sinkt das mittlere Quadrat der Ausgabe auf genau ein Viertel seines vorherigen Werts.

<!--en:3a31b6a7-->
## Initialisierung von Residualnetzen

<!--en:e05a429d-->
### Wachstum der Varianz von Block zu Block

<!--en:675db8eb-->
Wir haben behauptet, dass die Aktivierungen in einem tiefen Stapel von Residualblöcken schon bei der Initialisierung explodieren, selbst wenn jede Schicht der Regel von He folgt. Zum Beweis betrachten wir die Abbildung $\mathbf{x}_{k+1} = \mathbf{x}_k + f_k(\mathbf{x}_k)$ eines Residualblocks. Wir verwenden den zweischichtigen Residualzweig aus :numref:`sec_model_construction` und vervierfachen seine Breite in der Mitte, wie es bei vielen modernen Residualblöcken und Transformern üblich ist. Eine lineare Schicht erweitert die Breite von $d$ auf $4d$. Darauf folgen eine ReLU und eine lineare *Ausgabeprojektion* von $4d$ zurück auf $d$. Alle Bias-Terme sind anfangs null. Die erste Schicht verwendet die He-Initialisierung ($g = \sqrt{2}$), weil ihr eine ReLU folgt. Für die Ausgabeprojektion wählen wir die Standardabweichung $g/\sqrt{4d}$ mit einem zunächst freien Verstärkungsfaktor $g$. Dabei entspricht $g = 1$ der üblichen Regel für lineare Schichten aus :numref:`sec_numerical_stability`.

<!--en:666bb71c-->
![Mittleres Quadrat des Signals an jeder Stelle eines Residualblocks. Die nach He initialisierte Schicht verdoppelt $m_k$, die ReLU halbiert es wieder, und die Ausgabeprojektion multipliziert es mit $g^2$. Der Identitätspfad führt $m_k$ am Residualzweig vorbei. Beim Addieren der beiden Signale addieren sich ihre mittleren Quadrate.](../img/bg-res-variance.svg)

<!--en:f9ef1c3c-->
Sei $m_k$ das mittlere Quadrat der Einträge von $\mathbf{x}_k$. Wie es sich innerhalb des Blocks verändert, zeigt :numref:`fig_bg_res-variance`. Die erste lineare Schicht verdoppelt $m_k$, die ReLU halbiert es wieder auf $m_k$ (:numref:`subsec_he_init`). Die Ausgabeprojektion skaliert das mittlere Quadrat anschließend um den Faktor $4d \cdot g^2/(4d) = g^2$. Die Ausgabe $f_k(\mathbf{x}_k)$ des Residualzweigs hat also das mittlere Quadrat $g^2 m_k$. Nun folgt der entscheidende Schritt. Die Gewichte der Ausgabeprojektion haben den Erwartungswert null und sind unabhängig von $\mathbf{x}_k$. Daher ist die Ausgabe $f_k(\mathbf{x}_k)$ des Residualzweigs mit dem Signal $\mathbf{x}_k$ der Skip-Verbindung unkorreliert. Die mittleren Quadrate addieren sich also:

<!--en:9558c174-->
Jeder Block vergrößert das mittlere Quadrat der Aktivierungen um den Faktor $1 + g^2 > 1$, sofern $g \neq 0$. Bei $g = 1$ verdoppelt sich die Varianz in jedem Block. Über $N$ Blöcke wächst die Standardabweichung der Aktivierungen daher um den Faktor $2^{N/2}$. Das entspricht dem Faktor $2\times$ nach 2 Blöcken, $16\times$ nach 8 Blöcken und $65{,}536\times$ nach 32 Blöcken. Jede einzelne Schicht hat die Varianzregel für eine Schicht eingehalten. Trotzdem sind die Aktivierungen im gesamten Netz explodiert, denn die lokale Regel berücksichtigt die Skip-Verbindung nicht, über die jeder Block seine Eingabe wieder addiert.

<!--en:041f0e21-->
### Skalierung der Ausgabeprojektion mit einem kleineren Faktor

<!--en:071c6ddb-->
Über $N$ Blöcke multiplizieren sich die Faktoren der einzelnen Blöcke zu $(1 + g^2)^N$. Damit die Aktivierungen bei beliebiger Tiefe ihre Größenordnung beibehalten, muss $g$ mit wachsendem $N$ abnehmen. Wählen wir $g^2 = 1/N$, skalieren wir die Ausgabeprojektion jedes Residualzweigs mit $1/\sqrt{N}$. Damit lässt sich das Wachstum begrenzen:

<!--en:647ea4c7-->
denn $(1 + 1/N)^N$ wächst monoton gegen $e$. Die Standardabweichung der Aktivierungen wächst nun unabhängig von der Tiefe höchstens um den Faktor $\sqrt{e} \approx 1.65$ (Faktor $1.5$ nach 2 Blöcken, $1.60$ nach 8 und $1.64$ nach 32). Dieselbe Skalierung mit $1/\sqrt{N}$ ist in der Praxis weit verbreitet. GPT-2 :cite:`Radford.Wu.Child.ea.2019` multipliziert beispielsweise die anfängliche Standardabweichung der Ausgabeprojektion jedes Residual*zweigs* mit $1/\sqrt{N}$. Dabei ist $N$ die Gesamtzahl der Residualzweige im Netz. Die Blöcke von GPT-2 normalisieren ihre Eingaben. Deshalb ist das Wachstum additiv statt multiplikativ (siehe Übungen), und dieselbe Skalierung hält die Summe beschränkt. Fixup :cite:`Zhang.Dauphin.Ma.2019` leitet eine solche tiefenabhängige Skalierung für Residualnetze her, die ganz ohne Normalisierungsschichten auskommen.

<!--en:688036a4-->
### Ausgabeprojektionen mit Anfangswert null

<!--en:d2050dd3-->
Was passiert, wenn wir diese Idee bis zum Äußersten treiben und $g = 0$ setzen? Auf den ersten Blick scheint eine Initialisierung mit null riskant. Setzt man *jedes* Gewicht auf null, bleiben alle versteckten Einheiten aufgrund der Symmetrie für immer identisch (:numref:`sec_numerical_stability`). Ganz anders verhält es sich, wenn wir *nur* die Ausgabeprojektion jedes Residualzweigs auf null setzen. In Schritt null gibt jeder Zweig null aus. Jeder Block bildet seine Eingabe also exakt auf sich selbst ab. Das gesamte Netz startet so mit stabilen Aktivierungen, und seine Tiefe kommt während des Trainings allmählich zum Tragen. Die Symmetrie blockiert das Lernen nicht, denn die erste Schicht des Residualzweigs behält ihre zufälligen Gewichte. Schon im ersten Rückwärtsdurchlauf erhält die Ausgabeprojektion einen von null verschiedenen Gradienten, weil ihre Eingaben aus der ReLU von null verschieden sind. Ihre Gewichte ändern sich also sofort. Ab dem zweiten Schritt fließen die Gradienten auch in die erste Schicht. :numref:`fig_bg_zero-out-proj` zeigt die drei Phasen und stellt ihnen den Fall gegenüber, in dem alle Gewichte bei null bleiben.

<!--en:4fa35449-->
![Initialisierung allein der Ausgabeprojektion $W_{\mathrm{out}}$ eines Residualzweigs mit null. Beim ersten Vorwärtsdurchlauf gibt der Zweig $f = 0$ aus. Der Block bildet seine Eingabe also auf sich selbst ab, doch die versteckten Aktivierungen $\mathbf{h}$ sind von null verschieden. Im ersten Rückwärtsdurchlauf erhält $W_{\mathrm{out}}$ daher den von null verschiedenen Gradienten $\boldsymbol{\delta}\mathbf{h}^\top$. Dabei ist $\boldsymbol{\delta}$ der Gradient, der an der Ausgabe des Zweigs ankommt. Für $W_{\mathrm{in}}$ ergibt sich kein Gradient. Ab dem nächsten Schritt lernen beide Schichten. Untere Reihe: Sind beide Schichten mit null initialisiert, gilt $\mathbf{h} = 0$ und keine der beiden Schichten erhält je einen von null verschiedenen Gradienten.](../img/bg-zero-out-proj.svg)

<!--en:2bd65afb-->
Die Initialisierung von Residualzweigen mit null lässt sich über mehrere Arbeiten zurückverfolgen. :citet:`Goyal.Dollar.Girshick.ea.2017` setzten den letzten Skalierungsfaktor jedes Residualblocks auf null, um ResNets mit großen Minibatches zu trainieren. Fixup kombiniert mit null initialisierte Ausgabeprojektionen mit seiner tiefenabhängigen Skalierung. Spätere Arbeiten ersetzen die Nullmatrix durch einen einzigen lernbaren Skalar pro Zweig, dessen Anfangswert null ist :cite:`De.Smith.2020,Bachlechner.Majumder.Mao.ea.2021`.

<!--en:f4419e59-->
### Messung der Varianz in Abhängigkeit von der Tiefe

<!--en:9f32ca44-->
Nun überprüfen wir alle drei Vorhersagen empirisch. `WideResidualBlock` implementiert unseren Residualblock der Form $d \to 4d \to d$ mit unserem eigenen Initialisierer für normalverteilte Gewichte. Für die Erweiterungsschicht gilt der Verstärkungsfaktor $\sqrt{2}$. Für die Ausgabeprojektion ist der Faktor über `out_gain` einstellbar.

<!--en:f6b03778-->
Als Nächstes baut `std_across_depth` einen Stapel aus `num_blocks` Blöcken der Breite 64 auf. Die Funktion speist 256 Eingabevektoren aus $\mathcal{N}(0, 1)$ ein und erfasst die Standardabweichung der Aktivierungen an der Eingabe und nach jedem Block.

<!--en:b7b94990-->
Wir vergleichen die drei Varianten des Verstärkungsfaktors: unskaliert ($g = 1$), skaliert ($g = 1/\sqrt{N}$) und null ($g = 0$). Für $N = 2, 8, 32$ geben wir die Standardabweichung am Ende des Stapels aus. Für $N = 32$ zeichnen wir den gesamten Verlauf über die Tiefe.

<!--en:49d922c0-->
Die Messkurven bestätigen unsere Analyse. Im unskalierten Stapel wächst die Standardabweichung pro Block etwa um den Faktor $\sqrt{2}$. Auf der logarithmischen Skala ergibt sich eine Gerade. Nach 32 Blöcken liegen die Werte etwa zwischen 20 000 und 200 000. Die meisten einzelnen Durchläufe liegen etwas unter dem theoretischen Wert $\sqrt{\mathbb{E}[m_{32}]} = 65{,}536$. Das Produkt zufälliger positiver Faktoren hat hier eine rechtsschiefe Verteilung. Der Median der Standardabweichung liegt daher unter der Quadratwurzel des erwarteten mittleren Quadrats. Bei anfänglichen Aktivierungen in der Größenordnung von Zehntausenden lohnt sich ein Trainingsversuch mit dem unskalierten Stapel kaum. Im mit $1/\sqrt{N}$ skalierten Stapel bleibt die Standardabweichung dagegen bei allen untersuchten Tiefen zwischen 1.4 und 1.9. Das passt zu unserer Vorhersage von 1.5–1.65. Setzen wir die Ausgabeprojektionen auf null, bleibt die Standardabweichung über alle 32 Blöcke hinweg konstant bei 1.

<!--en:86496eb7-->
Mit einem solchen kurzen Vorwärtsdurchlauf erkennen wir eine ungeeignete Initialisierung sofort, bevor wir GPU-Zeit für das Training verschwenden. Ob die Optimierung mit der Skalierung um $1/\sqrt{N}$ oder mit der Initialisierung bei null schneller vorankommt, verrät dieser Test allerdings nicht. Beide erhalten in Schritt null die Größenordnung des Signals. Sie unterscheiden sich nur darin, ob die Residualzweige von Anfang an aktiv sind oder erst im Lauf der ersten Aktualisierungen aktiv werden.

<!--en:b1cb85bb-->
Die Initialisierung bestimmt nur den Zustand in Schritt null. Sobald Gradientenschritte die Gewichte verändern, kann die Initialisierung allein nicht mehr gewährleisten, dass die Aktivierungen ihre Größenordnung beibehalten. Hier kommen Normalisierungsschichten (:numref:`sec_batch_norm`) ins Spiel.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:1d69c352-->
Frameworks verwenden standardmäßig keine He-Initialisierung. Tiefe Netze ohne Normalisierungsschichten müssen wir daher explizit initialisieren, etwa über Konstruktorargumente, über eine Rückruffunktion für den gesamten Modulbaum oder durch direkte Änderungen im Speicher. Gestutzte Normalverteilungen verhindern seltene extreme Ausreißer in großen Modellen. Für Residualnetze reichen Varianzregeln für einzelne Schichten (:numref:`sec_numerical_stability`) allerdings nicht aus. Jeder Block addiert die Ausgabe seines Residualzweigs zur Eingabe und multipliziert damit das mittlere Quadrat der Aktivierungen mit $1 + g^2$. Bei $g = 1$ wächst die Standardabweichung also pro Block um den Faktor $\sqrt{2}$. Skalieren wir die Ausgabeprojektion jedes Blocks mit $1/\sqrt{N}$, bleibt der Wachstumsfaktor der Standardabweichung bei jeder Tiefe auf $\sqrt{e}$ begrenzt, unabhängig von der Zahl $N$ der Blöcke. Setzen wir die Ausgabeprojektion auf null, beginnt jeder Block als Identitätsabbildung, ohne dass die Symmetrie das Lernen verhindert.

<!--en:2294243a-->
## Übungen

<!--en:0b8f0959-->
1. **Gestutzte Normalverteilung.** Welcher Anteil der gezogenen Werte entfällt, wenn Sie eine Gauß-Verteilung bei $\pm 2\sigma$ stutzen? Um wie viel sinkt die Standardabweichung? Für keine der beiden Größen gibt es eine geschlossene Darstellung durch elementare Funktionen. Drücken Sie beide Größen durch die Dichte und die Verteilungsfunktion der Standardnormalverteilung aus und berechnen Sie sie mit Tabellenwerten. Vergleichen Sie die Standardabweichung mit der Ausgabe des Beispiels in :numref:`subsec_truncated_normals`. Überprüfen Sie den verworfenen Anteil, indem Sie in der dort gezogenen ungestutzten Stichprobe die Einträge außerhalb von $\pm 2\sigma$ zählen.

<!--en:e7513012-->
2. [code] **Normalisierte Eingaben.** Viele Residualnetze normalisieren die Eingabe jedes Residualzweigs. Der Zweig erhält $\mathbf{x}_k / r_k$, wobei $r_k$ das quadratische Mittel der Einträge von $\mathbf{x}_k$ für das jeweilige Beispiel ist. Die unnormalisierte Eingabe wäre $\mathbf{x}_k$.
    1. Zeigen Sie, dass die Ausgabe des Residualzweigs dann das mittlere Quadrat $g^2$ hat, unabhängig von der Größenordnung von $\mathbf{x}_k$. Zeigen Sie außerdem, dass bei $m_0 = 1$ das mittlere Quadrat nach $N$ Blöcken gleich $1 + N g^2$ ist. Wie unterscheidet sich das Wachstum mit der Tiefe von dem der nicht normalisierten Blöcke in diesem Abschnitt?
    1. Bei welchem Verstärkungsfaktor ist das mittlere Quadrat am Ende des Stapels für jedes $N$ gleich 2? Vergleichen Sie den Faktor mit der Skalierung von GPT-2.
    1. Ergänzen Sie `WideResidualBlock` um die Normalisierung. Messen Sie die Standardabweichung in Abhängigkeit von der Tiefe für $g = 1$ und $g = 1/\sqrt{N}$ bei $N = 32$. Vergleichen Sie das Ergebnis mit Ihren Vorhersagen und der Messkurve in :numref:`subsec_variance_across_depth`.

<!--en:8ea92892-->
3. [code] **Kaiming-Initialisierung.** Implementieren Sie die beiden Kaiming-Initialisierer (He-Initialisierer) aus :numref:`subsec_he_init`, einen mit Gleichverteilung und einen mit Normalverteilung. Sehen Sie ein explizites Argument `nonlinearity` vor, das je nach Wert (`'relu'` oder `'linear'`) den passenden Verstärkungsfaktor auswählt.
    1. Prüfen Sie beide Initialisierer anhand von Mittelwert, Standardabweichung und Wertebereich der Gewichte einer Schicht mit 512 Eingängen und 256 Ausgängen. Vergleichen Sie den Initialisierer mit Normalverteilung mit dem skalierten Initialisierer für normalverteilte Gewichte aus diesem Abschnitt. Vergleichen Sie beim Initialisierer mit Gleichverteilung die Messwerte mit der vorhergesagten Grenze und Standardabweichung. Erklären Sie etwaige Unterschiede im Wertebereich.
    1. Wenden Sie Ihren Initialisierer mit Normalverteilung auf den Stapel aus :numref:`subsec_variance_across_depth` an. Verwenden Sie für beide Schichten jedes Blocks den Verstärkungsfaktor für `'relu'`. Sagen Sie voraus, mit welchem Faktor jeder Block das mittlere Quadrat multipliziert. Ergänzen Sie das Diagramm aus :numref:`subsec_variance_across_depth` um eine Messkurve und überprüfen Sie daran Ihre Vorhersage.

    *Nach CMU 10-714, [Hausaufgabe 2](https://github.com/dlsyscourse/hw2) und [Hausaufgabe 4](https://github.com/dlsyscourse/hw4).*

<!--en:87579008-->
4. [code] **Gradienten in Abhängigkeit von der Tiefe.** ● Der Rückwärtsdurchlauf durch einen Block multipliziert den Gradienten mit der transponierten Jacobi-Matrix von $\mathbf{x} \mapsto \mathbf{x} + f(\mathbf{x})$.
    1. Übertragen Sie die Herleitung aus diesem Abschnitt auf den Rückwärtsdurchlauf. Zeigen Sie, dass der Residualzweig das mittlere Quadrat des Gradienten mit $g^2$ multipliziert, obwohl der Rückwärtsdurchlauf über die Ausgänge jeder Schicht statt über ihre Eingänge summiert. Damit multipliziert jeder Block das mittlere Quadrat wieder mit $1 + g^2$.
    1. Überprüfen Sie die Vorhersage für alle drei Verstärkungsfaktoren bei $N = 32$. Berechnen Sie den Gradienten der Summe aller Ausgaben des Stapels bezüglich seiner Eingabe. Vergleichen Sie sein mittleres Quadrat mit dem des Gradienten an der Ausgabe.

<!--en:4e24f4d9-->
5. [code] **Datenbasierte Kalibrierung.** ● Initialisieren Sie den Stapel aus :numref:`subsec_variance_across_depth` mit dem unskalierten Verstärkungsfaktor. Kalibrieren Sie ihn anschließend Block für Block mit einem einzigen Minibatch von Eingaben mit Varianz eins. Lassen Sie den Minibatch nacheinander bis zum jeweiligen Block vorwärts durchlaufen. Messen Sie die Standardabweichung der Ausgabe $f_k(\mathbf{x}_k)$ seines Residualzweigs und skalieren Sie die Ausgabeprojektion des Blocks so, dass diese Standardabweichung $1/\sqrt{N}$ beträgt. Damit übertragen Sie die schichtweise sequenzielle Initialisierung auf Einheitsvarianz von :citet:`Mishkin.Matas.2016` auf einen Stapel von Residualblöcken.
    1. Erklären Sie, warum Sie die Blöcke der Reihe nach kalibrieren müssen und nicht alle auf einmal anhand eines einzigen Vorwärtsdurchlaufs kalibrieren können.
    1. Erklären Sie, warum die Kalibrierung an der Ausgabe des Residualzweigs ansetzt und nicht an der Blockausgabe $\mathbf{x}_k + f_k(\mathbf{x}_k)$.
    1. Sagen Sie die Standardabweichung am Ende des kalibrierten Stapels voraus. Zeichnen Sie die Standardabweichung des kalibrierten Stapels in Abhängigkeit von der Tiefe. Stellen Sie dieser Kurve die Kurve für den skalierten Verstärkungsfaktor gegenüber und erklären Sie den Unterschied.

<!--en:c9cd8213-->
[Diskussionen](https://d2l.discourse.group/t/36754)

<!--en:63ca25c0-->
Parameterinitialisierung<br>**Framework-Standards · Initialisierer für den ganzen Modulbaum · gestutzte Normalverteilungen · Varianz im Residualnetz $(1 + g^2)^N$ · Skalierung mit $1/\sqrt{N}$ oder null**

<!--en:ead987e9-->
::: {.slide title="Warum Aktivierungen in ResNets trotz He-Initialisierung jeder Schicht explodieren"}

<!--en:4bb8ae0b-->
[Das Paradox der Initialisierung von Residualnetzen]{.kicker}

<!--en:9f7049f6-->
- **Additive Skip-Verbindungen:** Jeder Residualblock berechnet $\mathbf{x}_{k+1} = \mathbf{x}_k + f_k(\mathbf{x}_k)$.

<!--en:99a6e699-->
- **Varianzen unkorrelierter Signale addieren sich:** Haben $\mathbf{x}_k$ und $f_k(\mathbf{x}_k)$ jeweils das mittlere Quadrat eins und sind sie unkorreliert, hat ihre Summe das **doppelte** mittlere Quadrat (Varianz pro Block $2\times$).

<!--en:0950e4cd-->
- **Exponentielles Wachstum mit der Tiefe:** Über $N = 32$ Residualblöcke wächst die Standardabweichung der Aktivierungen um den Faktor $2^{32/2} = 2^{16} = 65{,}536\times$, schon in Schritt $0$.

<!--en:3d87873d-->
::: {.slide title="Keine He-Initialisierung in der Standardeinstellung" only="pytorch"}

<!--en:c9f9870b-->
[Warum die Aktivierungen in tiefen ReLU-Stapeln standardmäßig schrumpfen]{.kicker}

<!--en:3f3a2187-->
- **Varianzfaktor pro ReLU-Schicht bei Standardinitialisierung ($\frac{1}{2}n_{\text{in}}\sigma^2$):** `nn.Linear` zieht aus $U(\pm 1/\sqrt{n_{\text{in}}})$. Damit gilt $\sigma^2 = \frac{1}{3n_{\text{in}}}$ und jede Schicht multipliziert das mittlere Quadrat mit $\frac{1}{6}$ ($6^{-10} \approx 1.7 \times 10^{-8}$ nach 10 Schichten).

<!--en:0324cefa-->
- **He-Initialisierung** ($\sigma^2 = 2/n_{\text{in}}$) hält den Faktor bei genau $1$.

<!--en:48656355-->
::: {.slide title="Keine He-Initialisierung in der Standardeinstellung" only="tensorflow"}

<!--en:c9f9870b-2-->
[Warum die Aktivierungen in tiefen ReLU-Stapeln standardmäßig schrumpfen]{.kicker}

<!--en:667b5732-->
- **Varianzfaktor pro ReLU-Schicht bei Standardinitialisierung ($\frac{1}{2}n_{\text{in}}\sigma^2$):** `Dense` verwendet die Glorot-Gleichverteilung. Bei $n_{\text{in}} = n_{\text{out}}$ ergibt das $\sigma^2 = \frac{1}{n_{\text{in}}}$ und den Faktor $\frac{1}{2}$ ($2^{-10} \approx 9.8 \times 10^{-4}$ nach 10 Schichten).

<!--en:0324cefa-2-->
- **He-Initialisierung** ($\sigma^2 = 2/n_{\text{in}}$) hält den Faktor bei genau $1$.

<!--en:d12b2438-->
::: {.slide title="Keine He-Initialisierung in der Standardeinstellung" only="jax"}

<!--en:c9f9870b-3-->
[Warum die Aktivierungen in tiefen ReLU-Stapeln standardmäßig schrumpfen]{.kicker}

<!--en:97d082c1-->
- **Varianzfaktor pro ReLU-Schicht bei Standardinitialisierung ($\frac{1}{2}n_{\text{in}}\sigma^2$):** `nnx.Linear` verwendet die LeCun-Normalverteilung mit $\sigma^2 = \frac{1}{n_{\text{in}}}$. Jede Schicht multipliziert das mittlere Quadrat mit $\frac{1}{2}$ ($2^{-10} \approx 9.8 \times 10^{-4}$ nach 10 Schichten).

<!--en:0324cefa-3-->
- **He-Initialisierung** ($\sigma^2 = 2/n_{\text{in}}$) hält den Faktor bei genau $1$.

<!--en:cdc8aaf7-->
::: {.slide title="Keine He-Initialisierung in der Standardeinstellung" only="mxnet"}

<!--en:c9f9870b-4-->
[Warum die Aktivierungen in tiefen ReLU-Stapeln standardmäßig schrumpfen]{.kicker}

<!--en:2e2cb1d0-->
- **Varianzfaktor pro ReLU-Schicht bei Standardinitialisierung ($\frac{1}{2}n_{\text{in}}\sigma^2$):** `nn.Dense` zieht aus $U(-0.07, 0.07)$. Das ergibt den Faktor $0.21$ bei einer Breite von $256$ ($\approx 1.6 \times 10^{-7}$ nach 10 Schichten).

<!--en:0324cefa-4-->
- **He-Initialisierung** ($\sigma^2 = 2/n_{\text{in}}$) hält den Faktor bei genau $1$.

<!--en:97915821-->
::: {.slide title="He-Initialisierung eines Modulbaums mit 10 Schichten"}

<!--en:40fcfc2c-->
[Initialisierung des gesamten Modulbaums]{.kicker}

<!--en:3327a490-->
- Die He-Initialisierung ($\sigma = \sqrt{2 / n_{\text{in}}}$) im gesamten Modulbaum bringt das mittlere Quadrat der Ausgabe eines Stapels aus 10 ReLU-Schichten wieder auf die Größenordnung eins.

<!--en:6c981907-->
::: {.slide title="He-Initialisierung beim Durchlaufen des Modulbaums"}

<!--en:40fcfc2c-2-->
[Initialisierung des gesamten Modulbaums]{.kicker}

<!--en:bc871219-->
::: {.slide title="Gestutzte Normalverteilungen verhindern extreme Anfangsgewichte"}

<!--en:2e227be1-->
[Begrenzung extremer Gewichte in großen Modellen]{.kicker}

<!--en:50d85ee0-->
- Bei $10^8$ Gewichten liefert eine Gauß-Verteilung Dutzende Ausreißer mit Beträgen $>5\sigma$. BERT und ViT verwenden eine **gestutzte Normalverteilung** und ziehen Samples außerhalb von $\pm 2\sigma$ neu (Standardabweichung $\approx 0.88\times$).

<!--en:b382ea61-->
::: {.slide title="Ein eigener Initialisierer mit einstellbarem Verstärkungsfaktor" only="pytorch,mxnet"}

<!--en:b3fa61f9-->
[Schnittstelle für eigene Initialisierer]{.kicker}

<!--en:65b4a3cf-->
- Ein wiederverwendbarer Initialisierer zieht normalverteilte Gewichte mit Standardabweichung $\sigma = g / \sqrt{n_{\text{in}}}$ (Speicherform der Gewichte `(n_out, n_in)`).

<!--en:3c28b707-->
::: {.slide title="Ein eigener Initialisierer mit einstellbarem Verstärkungsfaktor" only="tensorflow,jax"}

<!--en:b3fa61f9-2-->
[Schnittstelle für eigene Initialisierer]{.kicker}

<!--en:aef389a6-->
- Ein wiederverwendbarer Initialisierer zieht normalverteilte Gewichte mit Standardabweichung $\sigma = g / \sqrt{n_{\text{in}}}$ (Speicherform der Gewichtsmatrizen `(n_in, n_out)`).

<!--en:9de1d271-->
::: {.slide title="Direkte Änderungen an Parametertensoren"}

<!--en:f4dd4886-->
[Direkte Änderungen an Gewichten]{.kicker}

<!--en:32be834a-->
Parametertensoren lassen sich direkt im Speicher skalieren oder anderweitig ändern. Da die ReLU positiv homogen ist, sinkt das mittlere Quadrat der Ausgabe auf ein Viertel, wenn die Gewichte der letzten Schicht halbiert werden.

<!--en:52e8d812-->
::: {.slide title="Herleitung des Varianzwachstums in einem Residualblock"}

<!--en:c82ffbad-->
[Varianzanalyse eines Residualblocks]{.kicker}

<!--en:5cb33b84-->
1. **Erweiterung und ReLU erhalten die Eingabeskala:** Die nach He initialisierte Schicht der Form $d \to 4d$ ($\text{Var} = 2/d$) verdoppelt das mittlere Quadrat auf $2m_k$, die ReLU halbiert es wieder auf $m_k$.

<!--en:1e112da2-->
2. **Die Ausgabeprojektion skaliert mit $g^2$:** Die Projektion der Form $4d \to d$ ($\text{Var} = g^2/(4d)$) ergibt am Ausgang des Residualzweigs das mittlere Quadrat $g^2 m_k$.

<!--en:024d4efc-->
3. **Unkorrelierte Signale aus Skip-Verbindung und Residualzweig addieren sich:** $\mathbb{E}[\mathbf{x}_k^\top f_k(\mathbf{x}_k)] = 0 \implies m_N = (1 + g^2)^N m_0$.

<!--en:7fb08e55-->
::: {.slide title="Drei Verstärkungsfaktoren für Residualnetze der Tiefe N"}

<!--en:44f46334-->
[Begrenzung des Wachstums mit der Tiefe]{.kicker}

<!--en:125ac519-->
- **Ohne Skalierung ($g = 1$):** Die Varianz verdoppelt sich pro Block ($m_N = 2^N m_0$). Die Standardabweichung der Aktivierungen wächst um den Faktor $2^{16} = 65{,}536\times$ bei $N = 32$.

<!--en:e701b842-->
- **Tiefenabhängig skaliert ($g = 1/\sqrt{N}$, GPT-2):** Die Varianz wächst pro Block um den Faktor $1 + 1/N$. Damit gilt $m_N \le e\,m_0$ ($\text{std} \le \sqrt{e} \approx 1.65\times$).

<!--en:6f339093-->
- **Mit null initialisiert ($g = 0$, Fixup):** Jeder Residualblock beginnt als Identitätsabbildung ($m_N = m_0$, $\text{std} = 1.00\times$).

<!--en:685cfc63-->
::: {.slide title="Warum der Faktor 1/√N das Wachstum bei jeder Tiefe auf √e begrenzt"}

<!--en:ccee0831-->
[Die Skalierungsregel für Residualzweige in GPT-2]{.kicker}

<!--en:10773cc4-->
- **Tiefenunabhängige Schranke:** Wegen $(1 + 1/N)^N \nearrow e$ wächst die Standardabweichung der Aktivierungen höchstens um den Faktor $\sqrt{e} \approx 1.65\times$ gegenüber der Eingabe, ob das Modell 8, 32 oder 1000 Blöcke hat.

<!--en:7c37133f-->
- **Anwendung in der Praxis:** GPT-2 multipliziert die anfängliche Standardabweichung der Ausgabeprojektion jedes Residualzweigs mit $1/\sqrt{N}$.

<!--en:781f5f9d-->
::: {.slide title="Warum Ausgabeprojektionen mit Anfangswert null die Symmetriefalle umgehen"}

<!--en:75f1be44-->
[Ausgabeprojektionen mit Anfangswert null (g = 0)]{.kicker}

<!--en:0c83a622-->
- **Schritt 0 (Netz als Identitätsabbildung):** Werden *nur* die Gewichte der Ausgabeprojektion auf null gesetzt ($\mathbf{W}_{\text{out}} = \mathbf{0}$), gilt $f_k(\mathbf{x}_k) = \mathbf{0}$ und damit $\mathbf{x}_{k+1} = \mathbf{x}_k$ bei Varianz eins.

<!--en:ed7d43e8-->
- **Erster Rückwärtsdurchlauf (erster Schritt weg von null):** Die Erweiterungsschicht ($d \to 4d$) hat zufällige, nach He initialisierte Gewichte. Ihre Aktivierungen $\mathbf{h}_k$ nach der ReLU sind daher von null verschieden und unterscheiden sich voneinander.

<!--en:560efa37-->
- **Sofortige Aktualisierung:** $\nabla_{\mathbf{W}_{\text{out}}} L = (\nabla_{\mathbf{x}_{k+1}} L)\mathbf{h}_k^\top \neq \mathbf{0}$ aktualisiert $\mathbf{W}_{\text{out}}$ im ersten Schritt. Ab dem zweiten Schritt fließen Gradienten auch in den Residualzweig.

<!--en:23468837-->
::: {.slide title="WideResidualBlock mit einstellbarem Verstärkungsfaktor für die Ausgabe"}

<!--en:5e7332f1-->
[Versuchsaufbau]{.kicker}

<!--en:ca2d0705-->
Test aller drei Varianten an einem Residualblock der Form $d \to 4d \to d$ (`WideResidualBlock`, Breite $d = 64$) mit He-Initialisierung ($g = \sqrt{2}$) in der Erweiterungsschicht und `out_gain` in der Ausgabeprojektion:

<!--en:cc2c4972-->
::: {.slide title="Messung der Standardabweichung der Aktivierungen über die Tiefe"}

<!--en:70e4f6ac-->
[Diagnose im Vorwärtsdurchlauf]{.kicker}

<!--en:7901f818-->
`std_across_depth` stapelt `num_blocks` Blöcke der Breite $d = 64$, speist einen Minibatch mit 256 Vektoren aus $\mathcal{N}(0, 1)$ ein und erfasst nach jedem Block die empirische Standardabweichung.

<!--en:94131e87-->
::: {.slide title="Empirische Überprüfung an 32 Residualblöcken"}

<!--en:24d6ce89-->
[Theorie und Messung]{.kicker}

<!--en:bd9175e1-->
Messungen für $g = 1$, $1/\sqrt{N}$ und $0$ über $N \in \{2, 8, 32\}$ Blöcke bestätigen den Faktor $(1+g^2)^N$:

<!--en:81229fdc-->
- **Vorwärtsdurchlauf immer schon in Schritt 0 prüfen:** Ein Batch mit Varianz eins macht in einer neuen Architektur eine explodierende Varianz binnen Millisekunden sichtbar.

<!--en:36c90c3c-->
::: {.slide title="Verständnisfragen zu Initialisierung und Skalierung von Residualzweigen"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:4f7fedec-->
Die Varianzfaktoren für diese drei Initialisierungen bestimmen:

<!--en:3355259e-->
1. Ein 10-schichtiges ReLU-MLP ($256 \to 256$, ohne Bias) zieht Anfangsgewichte aus $\mathcal{U}(-1/\sqrt{n_{\text{in}}}, 1/\sqrt{n_{\text{in}}})$. Mit welchem Faktor multipliziert jede Schicht das mittlere Quadrat der Aktivierungen, und was ergibt sich nach 10 Schichten?

<!--en:0c1c1141-->
2. Ein Initialisierer zieht aus $\mathcal{N}(0, \sigma^2)$, stutzt bei $\pm 2\sigma$ (dadurch sinkt die Standardabweichung um den Faktor $0.88$) und skaliert anschließend auf die Standardabweichung $\sigma = 0.02$. Wie groß kann der Betrag $|w|$ nach der Skalierung höchstens sein?

<!--en:ebce58c3-->
3. Wie groß darf der Verstärkungsfaktor $g$ der Ausgabeprojektion höchstens sein, damit das mittlere Quadrat nach $N = 100$ Residualblöcken höchstens $3\times$ so groß ist wie das mittlere Quadrat der Eingabe?

<!--en:557b98b6-->
**Lösungen:**

<!--en:c774da1f-->
1. **$\frac{1}{6}$ pro Schicht ($6^{-10} \approx 1.7 \times 10^{-8}$ nach 10 Schichten):** Bei $\text{Var}(w) = \frac{1}{3 n_{\text{in}}}$ wird die Varianz mit $\frac{1}{3}$ multipliziert. Die ReLU halbiert diesen Faktor auf $\frac{1}{6}$.

<!--en:f2eadc11-->
2. **$|w|_{\max} = \frac{2\sigma}{0.88} \approx 2.27\sigma = 0.0455$:** Die Skalierung dividiert die Grenzen $\pm 0.04$ durch $0.88$.

<!--en:547ac629-->
3. **$g \le \sqrt{3^{1/100} - 1} \approx \sqrt{\ln(3)/100} \approx 0.105$** (nahe bei $1/\sqrt{100} = 0.10$, womit sich $1.01^{100} \approx 2.70$ ergibt).

<!--en:efdf9274-->
::: {.slide title="Zusammenfassung zur Parameterinitialisierung"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:f2767a5e-->
- **Initialisierung einzelner Schichten und ganzer Modulbäume:** $\sigma = g/\sqrt{n_{\text{in}}}$ ($g = \sqrt{2}$ für ReLU) im gesamten Modulbaum verhindert, dass das Signal mit der Tiefe abklingt. Gestutzte Normalverteilungen schließen Ausreißer mit Beträgen $>2\sigma$ aus.

<!--en:1e256ed2-->
- **Varianzwachstum über Residualblöcke:** $m_{k+1} = (1 + g^2)m_k \implies m_N = (1 + g^2)^N m_0$ wächst wie $2^N$, wenn $g = 1$.

<!--en:554487e1-->
- **Skalierung der Residualzweige:** Mit dem Verstärkungsfaktor $g = 1/\sqrt{N}$ für die Ausgabeprojektion wächst die Standardabweichung höchstens um den Faktor $\sqrt{e} \approx 1.65\times$. Bei $g = 0$ beginnt jeder Block als Identitätsabbildung.
