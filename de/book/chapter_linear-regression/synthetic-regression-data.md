<!--en:dbe3d8d5-->
# Synthetische Daten

<!--en:c9dd53f3-->
Reale Datensätze sind oft unübersichtlich, und vor allem fehlt ihnen die Musterlösung. Trainieren Sie einen neuen Algorithmus mit echten Immobiliendaten, erhalten Sie einen Gewichtsvektor $\hat{\mathbf{w}}$. Doch weder die wahren Parameter $\mathbf{w}^*$ der Datenerzeugung noch die durch das Rauschen gesetzte Untergrenze sind Ihnen bekannt. Wer Forschungserfahrung hat, testet eine Implementierung deshalb zunächst an *synthetischen Daten* aus einem bekannten Modell, bevor er sie auf reale Daten anwendet.

<!--en:007751a8-->
Bei einem synthetischen Datensatz legen wir die Merkmalsverteilung, die wahren Gewichte $\mathbf{w}^*$ und den Bias $b^*$, den Stichprobenumfang $n$ und den Rauschpegel $\sigma$ selbst fest. Damit können wir unseren Code auf zweierlei Weise prüfen. Wir können testen, ob die gelernten Parameter $(\hat{\mathbf{w}}, \hat{b})$ den Werten $(\mathbf{w}^*, b^*)$ nahekommen. Und wir können den endgültigen Verlust mit der theoretischen, durch das Rauschen gesetzten Untergrenze $\frac{1}{2}\sigma^2$ vergleichen. In diesem Abschnitt erzeugen wir den synthetischen Datensatz für die nächsten drei Abschnitte, betrachten seine Streudiagramme und berechnen mit der Methode der kleinsten Quadrate (engl. least squares) eine Referenzlösung.

<!--en:ad7b0083-->
## Erzeugen der Daten

<!--en:ef4135b1-->
Wir erzeugen $n = 1{,}000$ Trainingsbeispiele mit $d = 2$ Merkmalen. So bleiben die Daten leicht darstellbar, und wir können trotzdem mit mehreren Merkmalen in Matrixform rechnen. Für die Designmatrix $\mathbf{X} \in \mathbb{R}^{1000 \times 2}$ ziehen wir unabhängig standardnormalverteilte Einträge $X_{ij} \sim \mathcal{N}(0, 1)$. Die Labels erzeugen wir aus dem linearen Modell

<!--en:ceffe8eb-->
mit den wahren Parametern

<!--en:ddd6fb7a-->
und additivem gaußschem Rauschen $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ mit Standardabweichung $\sigma = 0.01$. Das Signal $(\mathbf{X}\mathbf{w}^* + b^*)$ schwankt um mehrere Einheiten. Das Rauschen $\sigma = 0.01$ ist dagegen um einen Faktor von mehreren Hundert kleiner. Ein gut funktionierender Optimierer sollte $\mathbf{w}^*$ und $b^*$ daher bis auf wenige Tausendstel genau bestimmen.

<!--en:11e04bc9-->
Da JAX Pseudozufallszahlen funktional mit expliziten Schlüsseln erzeugt (:numref:`sec_ndarray`), erhält unser Datengenerator ein zusätzliches Argument, einen `key`. Diesen Schlüssel beziehen wir von der zustandsbehafteten Hilfsfunktion `d2l.get_key`.

<!--en:0f375d73-->
Ziehen wir $2{,}000$ Beispiele mit $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$ und $\sigma = 0.01$. Die ersten $1{,}000$ verwenden wir für das Training, die übrigen $1{,}000$ für die Validierung. Die Validierungsbeispiele stammen aus genau derselben Verteilung, gehen aber nie in die Anpassung ein. Der Verlust auf diesen Beispielen zeigt uns daher, wie genau das Modell bisher unbekannte Daten vorhersagt. Diesen Unterschied untersuchen wir in :numref:`sec_generalization_basics` genauer.

<!--en:5245bfbb-->
Jede Zeile von `X` enthält einen zweidimensionalen Merkmalsvektor und jede Zeile von `y` das zugehörige skalare Label. Für eine erste Plausibilitätsprüfung betrachten wir das erste Beispiel und berechnen seinen Rauschanteil $\epsilon^{(1)} = y^{(1)} - (\mathbf{w}^{*\top}\mathbf{x}^{(1)} + b^*)$.

<!--en:3ad23f9c-->
Rechnen Sie für das erste Beispiel von Hand nach: $2 x_1^{(1)} - 3.4 x_2^{(1)} + 4.2$ stimmt mit $y^{(1)}$ bis auf die kleine Störung aus $\mathcal{N}(0, 0.01^2)$ überein.

<!--en:27181f46-->
## Untersuchung der Daten

<!--en:f9f75fac-->
Bevor wir ein Modell anpassen, tragen wir das Label $y$ gegen jedes der beiden Merkmale $x_1$ und $x_2$ einzeln auf:

<!--en:91fc7890-->
Jedes Streudiagramm zeigt einen klaren linearen Trend mit dem erwarteten Vorzeichen: $y$ steigt mit $x_1$ (Steigung $+2$) und fällt mit $x_2$ steiler ab (Steigung $-3.4$). Warum sehen wir in jedem 2D-Streudiagramm trotz des winzigen Rauschpegels $\sigma = 0.01$ ein breites Band anstelle einer hauchdünnen Linie? Tragen wir $y$ nur gegen $x_2$ auf, bleibt die Variation der Punkte in $x_1$ unsichtbar. Der nicht dargestellte Term $2 x_1$ hat die Standardabweichung $2$. Dadurch verteilen sich die Punkte in vertikaler Richtung über mehrere Einheiten. In den marginalen 2D-Projektionen eines multivariaten Zusammenhangs geht die Variation der nicht dargestellten Koordinaten stets in die scheinbare vertikale Streuung ein.

<!--en:afe3cea8-->
## Schätzung der Parameter

<!--en:956882f2-->
In :numref:`subsec_linear-regression-analytic-solution` haben wir die Normalgleichungen für die geschlossene Lösung hergeleitet. Berechnen wir nun die exakte Lösung der kleinsten Quadrate $(\hat{\mathbf{w}}, \hat{b})$ für unsere 1000 Beispiele, indem wir `lstsq` auf die erweiterte Designmatrix $[\mathbf{X}, \mathbf{1}]$ anwenden. Das ist die Referenzlösung für diese endliche Stichprobe. Minibatch-SGD soll sie in :numref:`sec_linear_scratch` möglichst genau erreichen:

<!--en:7c381e18-->
Nun geben wir die tatsächlichen Schätzfehler zusammen mit dem vorhergesagten Standardfehler $\sigma / \sqrt{n}$ aus.

<!--en:fd42669e-->
Obwohl `lstsq` den empirischen Verlust bis auf Maschinengenauigkeit minimiert, unterscheidet sich $(\hat{\mathbf{w}}, \hat{b})$ in der vierten Nachkommastelle von den wahren Werten $(2, -3.4, 4.2)$. Diese kleine Abweichung ist ein *statistischer Schätzfehler* (engl. statistical estimation error). Sie entsteht nicht durch numerische Ungenauigkeit: Die 1000 Rauschwerte $\boldsymbol{\epsilon}$ korrelieren zufällig leicht mit den Spalten von $\mathbf{X}$. Dadurch neigt sich die Ausgleichsebene um einen Betrag der Größenordnung $\sigma / \sqrt{n} = 0.01 / \sqrt{1000} \approx 3 \times 10^{-4}$ (Übung :exref:`ex_synthetic-regression-data-recovery-under-noise`). Diese Grenze setzen die Daten. Kein erwartungstreuer Schätzer aus diesen 1000 verrauschten Punkten hat einen kleineren erwarteten quadratischen Fehler (*Cramér-Rao-Schranke*). Ein einzelner Versuch kann die wahren Werte zufällig besser treffen. In Übung :exref:`ex_synthetic-regression-data-misspecified` untersuchen wir, was bei einer nichtlinearen datenerzeugenden Funktion geschieht: Die Methode der kleinsten Quadrate findet dann die beste lineare Näherung bezüglich der Merkmalsverteilung.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:80072e72-->
Erzeugen wir synthetische Daten mit einem bekannten linearen Modell $\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$ und den Werten $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$ und $\sigma = 0.01$, gewinnen wir zwei quantitative Prüfmöglichkeiten, die bei realen Daten fehlen: die unvermeidbare Verlustuntergrenze $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$ und die bekannten wahren Parameter. Tragen wir $y$ gegen eine einzelne Koordinate $x_j$ auf, geht die Variation der nicht dargestellten Merkmale in die vertikale Streuung ein. So erscheint eine nahezu rauschfreie Ebene als breites 2D-Band. Selbst die exakte Lösung der kleinsten Quadrate weicht infolge des statistischen Schätzfehlers von $(\mathbf{w}^*, b^*)$ ab. Dessen Größenordnung $\sigma/\sqrt{n}$ ($\approx 3 \times 10^{-4}$ bei $n = 1{,}000$) gibt im nächsten Abschnitt die angestrebte Genauigkeit der iterativen Optimierung vor.

<!--en:2294243a-->
## Übungen

<!--en:26bd0fb8-->
1. [code] **Parameterschätzung bei Rauschen.** ● Wiederholen Sie die Anpassung mit der Methode der kleinsten Quadrate aus :numref:`subsec_synthetic-regression-data-recovery` für jede der folgenden Einstellungen auf 1000 unabhängigen Datensätzen. Messen Sie die Standardabweichung des Schätzfehlers $\hat{w}_1 - w_1^*$ für das erste Gewicht.
    1. Variieren Sie bei 1000 Trainingsbeispielen `noise` über $\{0.001, 0.01, 0.1, 1\}$ und die Anzahl $n$ der Trainingsbeispiele über $\{10, 100, 1{,}000, 10{,}000\}$ bei `noise=0.01`. Tragen Sie beide Messreihen zusammen mit der Vorhersage $\sigma/\sqrt{n}$ auf logarithmischen Achsen auf.
    1. Bestimmen Sie das kleinste $n$, bei dem der Messwert um höchstens 10% von $\sigma/\sqrt{n}$ abweicht. Welche Näherung, auf der die Vorhersage beruht, versagt bei weniger Beispielen?
    1. Schreiben Sie $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}]$, $\tilde{\mathbf{w}}^* = (\mathbf{w}^*, b^*)$ und $\hat{\tilde{\mathbf{w}}} = (\hat{\mathbf{w}}, \hat{b})$. Zeigen Sie anhand der Normalgleichungen, dass $\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon}$ gilt, und leiten Sie daraus $\textrm{Cov}(\hat{\tilde{\mathbf{w}}}) = \sigma^2(\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}$ her. Erzeugen Sie dann korrelierte Merkmale $x_2 = \rho x_1 + \sqrt{1 - \rho^2}\, z$ mit unabhängigen, standardnormalverteilten Größen $x_1$ und $z$ für $\rho \in \{0, 0.9, 0.99\}$. Sagen Sie mit dieser Formel den Standardfehler von $\hat{w}_1$ voraus und prüfen Sie ihn numerisch. Erklären Sie, warum die Vorhersagen $\hat{\tilde{\mathbf{w}}}^\top \tilde{\mathbf{x}}$ mit $\tilde{\mathbf{x}} = (\mathbf{x}, 1)$ auf neuen Beispielen aus derselben Verteilung trotz ungenauer Gewichte genau bleiben.
   :exlabel:`ex_synthetic-regression-data-recovery-under-noise`

<!--en:7fb8973a-->
1. [code] **Ein Modell, das nicht zu den Daten passt.** ● Addieren Sie zu jedem Label den Term $c\,x_1^2$ mit $c = 0.5$, sodass die Labels keine lineare Funktion der Merkmale mehr sind. Passen Sie das lineare Modell wie in diesem Abschnitt mit der Methode der kleinsten Quadrate an.
    1. Sagen Sie mithilfe von $\mathbb{E}[x_1^2] = 1$ und $\mathbb{E}[x_1^3] = 0$ voraus, welche der Schätzwerte $\hat{w}_1$, $\hat{w}_2$ und $\hat{b}$ von den wahren Werten abweichen und um wie viel ungefähr.
    1. Prüfen Sie die Vorhersage für $n \in \{100, 1{,}000, 10{,}000\}$ Trainingsbeispiele. Welche Schätzfehler schrumpfen mit wachsendem $n$, welche nicht? Wie verhalten sich die schrumpfenden Fehler im Vergleich zu $\sigma/\sqrt{n}$?
    1. Ein Test meldet jeden Parameterfehler über $10^{-3}$, also über etwa drei Standardfehlern. Eine korrekte Implementierung würde diesen Test mit den Daten des Abschnitts bestehen. Würde der Test eine korrekte Implementierung mit den hier veränderten Daten als fehlerhaft einstufen? Erklären Sie, was ein bestandener oder fehlgeschlagener Test aussagt, wenn das Modell nicht zu den Daten passt.
   :exlabel:`ex_synthetic-regression-data-misspecified`

<!--en:b667324b-->
1. **Die Verlustuntergrenze und ihre Schwankungen.** Im nächsten Abschnitt beurteilen wir einen Trainingslauf anhand des Validierungsverlusts, also des Mittelwerts von $\frac{1}{2}(\hat{y} - y)^2$ über die 1000 Validierungsbeispiele.
    1. Zeigen Sie, dass ein Modell mit den wahren Parametern auf einem neuen Beispiel den erwarteten Verlust $\sigma^2/2$ erreicht und dass dieser für keine anderen Parameter kleiner ist. Bei einem korrekten Trainingslauf pendelt sich der Verlust an dieser Untergrenze ein.
    1. Für $\epsilon \sim \mathcal{N}(0, \sigma^2)$ gilt $\mathbb{E}[\epsilon^4] = 3\sigma^4$. Berechnen Sie die Standardabweichung des Verlusts $\frac{1}{2}\epsilon^2$ eines einzelnen Beispiels. Bestimmen Sie daraus für die wahren Parameter den Standardfehler des Validierungsverlusts über $n' = 1{,}000$ Beispiele als Bruchteil der Untergrenze.
    1. [code] Prüfen Sie beide Zahlen, indem Sie die wahren Parameter auf vielen unabhängigen Validierungsdatensätzen auswerten. Bestimmen Sie dann mit dem Ergebnis aus Teil (a) den kleinsten Parameterfehler, der den erwarteten Verlust um einen Standardfehler des Validierungsverlusts erhöht. Vergleichen Sie ihn mit dem Schwellenwert $10^{-3}$ für den Parameterfehler aus der vorherigen Übung.
   :exlabel:`ex_synthetic-regression-data-loss-floor`

<!--en:e071e12b-->
[Diskussion](https://d2l.discourse.group/t/6662)

<!--en:be388e90-->
[Diskussion](https://d2l.discourse.group/t/6663)

<!--en:6f0f1fa4-->
[Diskussion](https://d2l.discourse.group/t/6664)

<!--en:b998e753-->
[Diskussion](https://d2l.discourse.group/t/17975)

<!--en:2ed41750-->
Synthetische Regressionsdaten: Prüfung anhand der wahren Werte<br>
**bekannte Parameter · gaußscher Datengenerator · rauschbedingte Untergrenze · statistischer Schätzfehler**

<!--en:ff3e9dca-->
::: {.slide title="Warum Code für maschinelles Lernen mit synthetischen Daten prüfen?"}

<!--en:66f66102-->
[Fehlersuche]{.kicker}

<!--en:f6e6f2c3-->
- **Mehrdeutigkeit bei realen Daten:** Der Verlust sinkt nicht weiter. Ist das Modell falsch spezifiziert, sind die Labels fehlerhaft oder steckt ein unbemerkter Fehler im Code?

<!--en:54e67ecc-->
- **Bekannte wahre Werte:** Erzeugen wir $\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$ mit $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$ und $\sigma = 0.01$, sind das lineare Modell und der quadratische Verlust konstruktionsbedingt statistisch optimal.

<!--en:4d99c362-->
- **Quantitatives Prüfkriterium:** Verfehlt der Verlust die Untergrenze $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$ oder liegen die Schätzwerte nicht nahe genug an $(\mathbf{w}^*, b^*)$ (Toleranz $\sim 3\sigma/\sqrt{n} \approx 10^{-3}$), deutet dies auf einen Fehler im Code hin.

<!--en:d7691e62-->
::: {.slide title="Das bekannte Modell der Datenerzeugung"}

<!--en:697851d4-->
[Mathematische Beschreibung]{.kicker}

<!--en:4dc08b7b-->
- **Wahre Parameter und hohes Signal-Rausch-Verhältnis:** Gewichte $\mathbf{w}^* = (2, -3.4)^\top$, Bias $b^* = 4.2$ und Standardabweichung des Signals $\sqrt{2^2 + (-3.4)^2} \approx 3.94$ (etwa $400\times$ so groß wie der Rauschpegel $\sigma = 0.01$).

<!--en:e01df669-->
- **Training und Validierung:** Wir ziehen $2{,}000$ unabhängige, identisch verteilte Beispiele: $1{,}000$ Trainingsbeispiele für die Anpassung und $1{,}000$ Validierungsbeispiele für die Auswertung auf neuen Daten.

<!--en:820c5e2f-->
::: {.slide title="Implementierung des Datengenerators"}

<!--en:88a2d6d8-->
[Vektorisierte Datenerzeugung]{.kicker}

<!--en:b4b76810-->
- **Explizite Spaltenform `(-1, 1)`:** Formen wir `w` zu `(d, 1)` um, liefert `matmul(X, w)` eine Spalte der Form `(n, 1)`. Sie passt zu `eps` der Form `(n, 1)`. So vermeiden wir eine unbeabsichtigte äußere Summe der Form `(n, n)` durch Broadcasting.

<!--en:20fb2ed6-->
::: {.slide title="Explizite Aufteilung der PRNG-Schlüssel in JAX" only="jax"}

<!--en:901bc94e-->
[Zufall ohne globalen Zustand · JAX]{.kicker}

<!--en:0e314633-->
- **Warum Schlüssel aufteilen?** Der *gleiche* `key` für `X` und `eps` würde dieselbe Zufallsquelle für beide Normalziehungen verwenden. Mit `jax.random.split` entstehen deterministisch unabhängige Teilschlüssel `key1` und `key2`.

<!--en:c17ceb2e-->
::: {.slide title="Erzeugte Tensoren und gezogene Rauschwerte untersuchen"}

<!--en:0c466f91-->
[Plausibilitätsprüfung]{.kicker}

<!--en:f25e1fa3-->
Wir ziehen $2{,}000$ Beispiele und betrachten das erste Trainingspaar $(\mathbf{x}^{(1)}, y^{(1)})$ zusammen mit seinem gezogenen Rauschwert $\epsilon^{(1)} = y^{(1)} - (\mathbf{w}^{*\top}\mathbf{x}^{(1)} + b^*)$:

<!--en:4c44d0b0-->
Prüfen Sie die Rechnung direkt: $2 x_1^{(1)} - 3.4 x_2^{(1)} + 4.2$ stimmt mit dem beobachteten Label $y^{(1)}$ bis auf die kleine gaußsche Störung $\epsilon^{(1)}$ der Größenordnung $\sigma = 0.01$ überein.

<!--en:2fbb2298-->
::: {.slide title="Warum marginale 2D-Streudiagramme breite Bänder zeigen"}

<!--en:9914b045-->
[Geometrische Anschauung]{.kicker}

<!--en:458d70d1-->
Warum ergeben sich trotz des winzigen Rauschpegels $\sigma = 0.01$ beim Auftragen von $y$ gegen $x_1$ und $x_2$ senkrecht mehrere Einheiten breite Punktwolken statt hauchdünner Linien?

<!--en:149b790b-->
- **Projektion ohne eine Variable:** Im 3D-Raum $(x_1, x_2, y)$ liegen von den $1{,}000$ Punkten bis auf wenige alle innerhalb von $\pm 0.03$ ($3\sigma$) um die geneigte Ebene $y = 2x_1 - 3.4x_2 + 4.2$.

<!--en:7c7cd578-->
- **Varianz des nicht dargestellten Merkmals:** Bei der Projektion auf $(x_2, y)$ geht die nicht dargestellte Koordinate $2x_1 \sim \mathcal{N}(0, 4)$ jedes Punkts in die vertikale Streuung ein ($\textrm{std} = 2.0$). Bei der Projektion auf $(x_1, y)$ übernimmt $-3.4x_2$ diese Rolle ($\textrm{std} = 3.4$).

<!--en:93b1bc7d-->
In marginalen 2D-Streudiagrammen einer multivariaten Funktion geht die Variation der nicht dargestellten Merkmale stets in die vertikale Streuung ein.

<!--en:7360c17c-->
::: {.slide title="Die unvermeidbare Verlustuntergrenze bei den wahren Parametern"}

<!--en:42df1fda-->
[Theoretischer Vergleichswert]{.kicker}

<!--en:5ca1df46-->
- **Rauschbedingte Untergrenze:** Da neue Rauschwerte $\epsilon \sim \mathcal{N}(0, \sigma^2)$ unabhängig von $\mathbf{x}$ sind, kann *kein* Prädiktor einen erwarteten Validierungsverlust unter $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$ erreichen.

<!--en:15095610-->
- **Schwankung bei endlicher Stichprobe:** Über $n' = 1{,}000$ Validierungsbeispiele schwankt der Stichprobenmittelwert von $\frac{1}{2}\epsilon^2$ um $5 \times 10^{-5}$. Sein Standardfehler beträgt $\sigma^2/\sqrt{2n'} \approx 2.2 \times 10^{-6}$.

<!--en:06a33aae-->
::: {.slide title="Parameterschätzung I: Der Schätzfehler ist linear im Rauschen"}

<!--en:58261667-->
[Statistische Analyse der Schätzung]{.kicker}

<!--en:2eff2f8a-->
- **Das Signal hebt sich auf:** Setzen wir $\mathbf{y} = \tilde{\mathbf{X}}\tilde{\mathbf{w}}^* + \boldsymbol{\epsilon}$ in die Normalgleichungen ein, fällt der wahre Parametervektor $\tilde{\mathbf{w}}^*$ heraus.

<!--en:1075f988-->
- **Erwartungstreue und rauschfreier Grenzfall:** Wegen $\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$ ist $\hat{\tilde{\mathbf{w}}}$ erwartungstreu ($\mathbb{E}[\hat{\tilde{\mathbf{w}}}] = \tilde{\mathbf{w}}^*$) und liefert $\tilde{\mathbf{w}}^*$ bei $\sigma = 0$ bis auf Maschinengenauigkeit.

<!--en:95fa89b9-->
::: {.slide title="Parameterschätzung II: Isotrope Merkmale ergeben X̃ᵀX̃ ≈ nI"}

<!--en:e1feba22-->
[Gesetz der großen Zahlen für die Gram-Matrix]{.kicker}

<!--en:3168b44a-->
- **Konzentration auf der Diagonalen:** Die empirischen zweiten Momente $\frac{1}{n}\sum_{i=1}^n (x_j^{(i)})^2$ konzentrieren sich eng um $\mathbb{E}[x_j^2] = 1$.

<!--en:9d5dac77-->
- **Entkopplung abseits der Diagonalen:** Die empirischen Kreuzprodukte $\frac{1}{n}\sum_{i=1}^n x_j^{(i)} x_k^{(i)}$ konzentrieren sich um $\mathbb{E}[x_j x_k] = 0$. Dadurch entkoppeln sich die drei Parameterfehler.

<!--en:7bea5ef7-->
::: {.slide title="Parameterschätzung III: Der Standardfehler σ/√n als Untergrenze"}

<!--en:51037c3f-->
[Die statistische Grenze]{.kicker}

<!--en:0933f563-->
- **Wurzelgesetz für den Stichprobenumfang:** Um den Schätzfehler um den Faktor $10\times$ zu senken, brauchen wir $100\times$ so viele Trainingsbeispiele ($n = 100{,}000$). Senken wir dagegen $\sigma$ um den Faktor $10\times$, sinkt der Fehler unmittelbar um den Faktor $10\times$.

<!--en:ba157fba-->
- **Exakte Kovarianz:** $\operatorname{Cov}(\hat{\tilde{\mathbf{w}}} \mid \tilde{\mathbf{X}}) = \sigma^2 (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}$ wächst bei kollinearen Merkmalen an.

<!--en:3cc55571-->
::: {.slide title="Die Referenzlösung der kleinsten Quadrate im Code prüfen"}

<!--en:bad5a28e-->
[Referenzlösung]{.kicker}

<!--en:a6d08a77-->
Wir lösen die Normalgleichungen für unsere $n = 1{,}000$ Trainingsbeispiele mit `lstsq` und vergleichen die tatsächlichen Parameterfehler mit $\sigma / \sqrt{n} \approx 3 \times 10^{-4}$:

<!--en:a571df4e-->
- **Prüftoleranz:** Jeder konvergierte iterative Optimierer in diesem Kapitel sollte bis auf $\sim 10^{-3}$ ($\approx 3\text{ SE}$) an $(\mathbf{w}^*, b^*)$ herankommen.

<!--en:c26d20c4-->
::: {.slide title="Verständnisfragen: Verlustuntergrenze, Stichprobenumfang und Kollinearität"}

<!--en:929b7671-->
[Verständnisfrage]{.kicker}

<!--en:a53f3b8a-->
Sagen Sie für jede Änderung am synthetischen Benchmark die quantitative Auswirkung voraus:

<!--en:bf23bb97-->
1. **Stichprobenumfang und Rauschen:** Wie groß ist der Standardfehler von $\hat{w}_1$ für (a) $n = 100, \sigma = 0.01$ und (b) $n = 1{,}000, \sigma = 0.1$?

<!--en:8c48b484-->
2. **Nicht modellierter quadratischer Term:** Die wahren Labels entstehen gemäß $y = 2x_1 - 3.4x_2 + 4.2 + 0.5x_1^2 + \epsilon$ mit $x_1, x_2 \sim \mathcal{N}(0, 1)$. Wir passen das lineare Modell $\hat{y} = w_1 x_1 + w_2 x_2 + b$ an. Welcher Parameter nimmt den nicht modellierten Term $0.5x_1^2$ auf, wenn $n \to \infty$ gilt?

<!--en:628ea3a3-->
3. **Korrelierte Merkmale:** Wenn $x_2 = 0.99 x_1 + \sqrt{1 - 0.99^2}\,z$ gilt, warum wachsen die Fehler der einzelnen Gewichte $|\hat{w}_j - w_j^*|$ um den Faktor $\sim 7\times$, während die Vorhersagen $\hat{\mathbf{w}}^\top\mathbf{x}$ auf neuen Testpunkten aus derselben Verteilung genau bleiben?

<!--en:557b98b6-->
**Antworten:**

<!--en:ca2b99de-->
1. (a) $\sigma / \sqrt{n} = 0.01 / \sqrt{100} = 10^{-3}$, (b) $0.1 / \sqrt{1000} \approx 3.16 \times 10^{-3}$.

<!--en:b068f1be-->
2. Wegen $\mathbb{E}[x_1^2] = 1$ und $\mathbb{E}[x_1 \cdot x_1^2] = \mathbb{E}[x_1^3] = 0$ ist der quadratische Term $0.5x_1^2$ orthogonal zu $x_1$ und $x_2$. Er verschiebt aber den Mittelwert der Labels um $+0.5$. Daher gilt $\hat{b} \to 4.2 + 0.5 = 4.7$ und zugleich $\hat{\mathbf{w}} \to (2, -3.4)^\top$.

<!--en:b38b5dce-->
3. Bei $x_1 \approx x_2$ ist die Varianz entlang der normierten Richtung der Differenz $x_1 - x_2$ klein ($1 - \rho = 0.01$). Dadurch wächst $\operatorname{Var}(\hat{w}_1 - \hat{w}_2)$. Für neue Testpunkte aus derselben Verteilung gilt aber ebenfalls $x_1 \approx x_2$. Sie hängen daher fast nur von der gut bestimmten Summe $\hat{w}_1 + \hat{w}_2$ ab.

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und wichtigste Ergebnisse"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:4f24074f-->
| Vergleichsgröße | Formel / Definition | Wert im Beispiel | Verwendung zur Prüfung in späteren Abschnitten |

<!--en:c31a22e8-->
| **Datengenerator** | $\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$ | $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$, $\sigma = 0.01$ | Lineares Testmodell mit bekannten Parametern ($n = 1{,}000$ Trainings-, $1{,}000$ Validierungsbeispiele) |

<!--en:e3ccc6b9-->
| **Unvermeidbare Verlustuntergrenze** | $\mathbb{E}[L(\mathbf{w}^*, b^*)] = \frac{1}{2}\sigma^2$ | $5 \times 10^{-5}$ (oder $\sigma^2 = 10^{-4}$ ohne $\frac{1}{2}$) | Erwartetes Plateau der Trainings- und Validierungsverlustkurven |

<!--en:6b9d7afe-->
| **Schätzfehler** | $\hat{\tilde{\mathbf{w}}} - \tilde{\mathbf{w}}^* = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^\top\boldsymbol{\epsilon}$ | Genau null für $\sigma = 0$ | Trennt statistisch bedingten Fehler vom Optimierungsfehler |

<!--en:3e6f3f25-->
| **Standardfehler der Parameter** | $\operatorname{SE}(\hat{w}_j) \approx \sigma / \sqrt{n}$ | $0.01 / \sqrt{1000} \approx 3.2 \times 10^{-4}$ | Legt die Prüftoleranz $3\text{ SE} \approx 10^{-3}$ für die geschätzten Gewichte fest |
