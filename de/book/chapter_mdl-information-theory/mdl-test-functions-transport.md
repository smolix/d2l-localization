<!--en:cf1ba284-->
# Testfunktionen und Transport

<!--en:ba6476bb-->
Ein Modell, dessen Samples die Daten verfehlen, lernt aus einer f-Divergenz nichts. Bei disjunkten Trägern ist jede Divergenz dieser Familie konstant (:numref:`sec_mdl-f-divergences`), und der Gradient einer Konstanten ist null. Zu diesem Schluss kamen wir an einem Extrembeispiel mit zwei Punktmassen. Reale Verteilungen weisen meist eine Streuung auf. Zwei Gauß-Verteilungen überlappen sich überall, weil ihre Dichten auch in den Ausläufern nie null erreichen. Genügt eine geringe Überlappung, damit wir wieder einen Gradienten erhalten? Die f-Divergenzen brachten auch ein praktisches Problem mit sich. Sie hängen vom Dichteverhältnis $p/q$ ab, das die Stichproben allein nicht liefern. Schon für die Schätzung der Divergenz mussten wir daher ein Kritikernetz trainieren (:numref:`sec_mdl-f-gan-dual`). Können wir zwei Verteilungen allein anhand ihrer Stichproben vergleichen, ganz ohne Kritiker?

<!--en:72dc0f38-->
Um die erste Frage mit Zahlen zu beantworten, wählen wir als Datenverteilung $P = \mathcal{N}(0, 1)$ und als Modell $Q = \mathcal{N}(d, 1)$. Das Modell hat dieselbe Glockenkurve, verschoben um $d$ Standardabweichungen. Dabei ist $d$ sein einziger Parameter. Wir importieren die benötigten Bibliotheken und berechnen für dieses Paar die fünf Vergleichsgrößen aus :numref:`tab_mdl-point-mass-table` samt ihren Steigungen bezüglich $d$. Diese Steigungen sind die Gradienten, die das Modell beim Training erhielte. Die Formeln im Code leiten wir im vorigen und im vorliegenden Abschnitt her. Betrachten Sie zunächst nur die ausgegebenen Zahlen. Wir bezeichnen sie im Folgenden als Eingangstabelle.

<!--en:5c8f20c6-->
Liegen die beiden Gauß-Verteilungen eine halbe Standardabweichung auseinander, so liegen alle fünf Steigungen zwischen $0.1$ und $1$. Mit jeder dieser Zielfunktionen ließe sich das Modell also wirksam trainieren. Bei $d = 8$ hat die Jensen-Shannon-Divergenz jedoch ihren Sättigungswert $0.693$ erreicht, der Totalvariationsabstand (engl. total variation distance) den Wert $1.000$. Das sind dieselben Konstanten $\log 2$ und $1$ wie bei disjunkten Punktmassen. Beide Steigungen sind auf etwa $10^{-4}$ gesunken. Daran ändern auch die überlappenden Ausläufer nichts. Die KL-Divergenz hat weiterhin eine beträchtliche Steigung, nämlich $d$ selbst. Das liegt aber nur daran, dass ein gaußsches Modell jedem Datenpunkt eine positive Dichte zuweist. Ein impliziter Generator, dessen Samples auf einer niedrigdimensionalen Fläche liegen, besitzt keine solche Dichte. Bleiben die beiden Abstände, die bei Punktmassen noch aussagekräftig waren. $\textrm{MMD}^2$ wächst zunächst, stagniert aber bei $d = 8$ und erreicht $1.155$ mit einer Steigung unter $10^{-4}$. Nur $W_1$ bleibt linear. Sein Wert ist $d$ und seine Steigung beträgt bei jedem Abstand $1$.

<!--en:2ac64249-->
Weder MMD noch $W_1$ benötigt das Dichteverhältnis $p/q$. Die MMD vergleicht, wie $P$ und $Q$ Funktionen einer Klasse *mitteln*. Mittelwerte lassen sich direkt aus Stichproben schätzen. Die Wahl der Funktionenklasse bestimmt, bis zu welchem Abstand zwei Verteilungen noch ein brauchbares Signal liefern. $W_1$ geht von einer geometrischen Frage aus. Wie weit muss Wahrscheinlichkeitsmasse wandern, um eine Verteilung in die andere zu überführen? Überraschenderweise läuft auch das auf einen Vergleich von Mittelwerten hinaus. Zwei Ergebnisse prägen den Rest dieses Abschnitts. Für eine feste Menge von Merkmalen lässt sich der Vergleich der Mittelwerte in geschlossener Form berechnen, ohne ein Hilfsnetz zu trainieren. Die minimalen Kosten für den Massentransport entsprechen wiederum der größten Mittelwertdifferenz über alle Funktionen mit Steigungsbetrag höchstens eins. Auf dieser Dualität beruht das Wasserstein-GAN (:numref:`sec_gan_objectives`). Durchgehend steht $P$ für die Datenverteilung und $Q$ für das Modell. Wir betrachten konkrete Beispiele auf der reellen Geraden.

<!--en:acd77795-->
## Integrale Wahrscheinlichkeitsmetriken und MMD

<!--en:c14a36c7-->
### Klassen von Testfunktionen

<!--en:58935604-->
Der Totalvariationsabstand ist eine Metrik und flacht in der Eingangstabelle dennoch ab. Seine Variationsdarstellung erklärt den Grund und zeigt einen Ausweg. Der Totalvariationsabstand ist der größte Unterschied zwischen den Wahrscheinlichkeiten, die $P$ und $Q$ einem Ereignis $A$ zuweisen (:numref:`sec_mdl-tv-pinsker`). Es gilt also $\textrm{TV}(P, Q) = \sup_A |P(A) - Q(A)|$. Die Wahrscheinlichkeit eines Ereignisses ist der Erwartungswert seiner Indikatorfunktion. Der Totalvariationsabstand ist somit das Supremum der Differenz zweier Mittelwerte über eine Klasse von Funktionen. Wir können auch andere Funktionen als Indikatorfunktionen zulassen. Für eine beliebige Klasse $\mathcal{G}$ von *Testfunktionen* $g$ ist die **integrale Wahrscheinlichkeitsmetrik** (engl. integral probability metric, IPM) :cite:`Muller.1997` definiert durch

<!--en:9e6ee49b-->
Stellen Sie sich $\mathcal{G}$ als eine Gruppe von Prüfern vor, die jeweils die Punkte im Raum bewerten. Die IPM gibt den größten Unterschied der mittleren Bewertungen an, den ein Prüfer der Gruppe aufdecken kann. Alle hier betrachteten Klassen sind symmetrisch. Das heißt, $-g$ gehört zu $\mathcal{G}$, sobald $g$ dazugehört. Die maximale Differenz ist daher stets nichtnegativ und symmetrisch in $P$ und $Q$. Verschwindet die IPM, wie das Wort „Metrik“ verspricht, nur für $P = Q$? Das hängt davon ab, ob $\mathcal{G}$ vielfältig genug ist, um je zwei verschiedene Verteilungen zu unterscheiden. Der Kritiker aus :numref:`sec_mdl-f-gan-dual` hatte eine ähnliche Prüfaufgabe, allerdings mit einem entscheidenden Unterschied. In der Variationsschranke $D_f(P\|Q) \geq \mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)]$ hält die konvex Konjugierte $f^*$ einen unbeschränkten Kritiker $T$ automatisch in Schach. Bei einer IPM mitteln wir $g$ direkt unter $P$ und $Q$. Die einzige Einschränkung für $g$ ist die Zugehörigkeit zu $\mathcal{G}$. Daher kommt es auf die Wahl der richtigen Klasse an.

<!--en:7f5ae241-->
Drei Funktionenklassen stehen hier im Mittelpunkt.

<!--en:71e2bcd7-->
- Beschränkte Funktionen mit $|g| \leq \tfrac{1}{2}$ liefern den Totalvariationsabstand. Zieht man von einer Indikatorfunktion $\tfrac{1}{2}$ ab, hält sie diese Schranke ein. In der Differenz der Mittelwerte fällt die Konstante weg, und $g = \tfrac{1}{2}\,\textrm{sign}(p - q)$ erreicht das Supremum.

<!--en:47ad212c-->
- Funktionen, die linear in einer festen Menge von Merkmalen sind, liefern die **Maximum Mean Discrepancy** (MMD) :cite:`Gretton.Borgwardt.Rasch.ea.2012`. Diese konstruieren wir als Nächstes.

<!--en:7458ca3c-->
- Funktionen mit Steigungsbetrag höchstens eins liefern den Wasserstein-Abstand. Diesen behandeln wir ausführlich in :numref:`sec_mdl-optimal-transport`.

<!--en:0a9528e8-->
Warum wählen wir für $\mathcal{G}$ nicht immer die größtmögliche Klasse? Eine brauchbare Klasse muss zwei gegenläufige Anforderungen erfüllen. Ist $\mathcal{G}$ zu groß, verletzt sie die erste. Erstens müssen Stichprobenmittelwerte die wahren Erwartungswerte gut wiedergeben. Zwar können wir Erwartungswerte ohne Dichteverhältnis direkt aus Stichproben schätzen. Eine zu vielfältige Gruppe von Prüfern findet zwischen endlichen Stichproben aber immer scheinbare Unterschiede. Zwei unabhängige Stichproben mit je $1{,}000$ Punkten aus *derselben* stetigen Verteilung haben beispielsweise fast sicher keinen Punkt gemeinsam. Die Indikatorfunktion der ersten Stichprobe hat auf der ersten Stichprobe den Mittelwert $1$ und auf der zweiten den Mittelwert $0$. Der empirische Totalvariationsabstand beträgt daher $1$, selbst wenn $P$ und $Q$ identisch sind. Um das zu vermeiden, muss $\mathcal{G}$ klein genug sein oder hinreichend glatte Funktionen verlangen, sodass solche scharf zugespitzten Funktionen ausgeschlossen sind.

<!--en:3042f47c-->
Zweitens soll die Differenz weiter wachsen, wenn $P$ und $Q$ weiter auseinanderrücken. Dafür muss $\mathcal{G}$ Funktionen enthalten, die mit dem Abstand wachsen. Beim Totalvariationsabstand fehlen solche Funktionen. Da seine Testfunktionen dem Betrag nach durch $\tfrac{1}{2}$ beschränkt sind, kann die Differenz nie größer als $1$ sein. Der größte Unterschied tritt beim Ereignis $\{p > q\}$ auf. Für die beiden Gauß-Verteilungen der Eingangstabelle ist das die Halbgerade $x < d/2$. Berechnen wir die Wahrscheinlichkeit dieser Halbgeraden unter beiden Verteilungen, erhalten wir $\textrm{TV} = \Phi(d/2) - \Phi(-d/2) = 2\Phi(d/2) - 1$. Dabei ist $\Phi$ die Verteilungsfunktion (engl. cumulative distribution function, CDF) der Standardnormalverteilung. Der Totalvariationsabstand erreicht $0.954$ bei $d = 4$ und weicht um weniger als $10^{-4}$ von $1$ ab, sobald $d = 8$ gilt. Trennt eine Sprungfunktion die beiden Glockenkurven fast vollständig, ändert sich ihr Wert beim weiteren Auseinanderrücken praktisch nicht mehr. Der Totalvariationsabstand verfehlt beide Anforderungen. An beiden messen wir auch die nächsten zwei Klassen.

<!--en:ea07253b-->
### Mittlere Merkmalsvektoren und Kernel

<!--en:de983e31-->
Bauen wir ausgehend von der kleinstmöglichen Gruppe von Prüfern eine bessere Klasse auf. Jeden Schritt erproben wir an demselben Paar. Wir vergleichen $P = \mathcal{N}(0, 1)$ mit $Q = \mathcal{N}(0, 4)$. Beide haben denselben Mittelwert, aber unterschiedliche Varianzen.

<!--en:a723c7ac-->
Beginnen wir mit der einfachsten Wahl, den linearen Testfunktionen $g(x) = w\, x$ mit $|w| \leq 1$. Hier gilt $\mathbb{E}_P g - \mathbb{E}_Q g = w\,(\mathbb{E}_P X - \mathbb{E}_Q X)$. Das Supremum über $|w| \leq 1$ ergibt daher $|\mathbb{E}_P X - \mathbb{E}_Q X|$, den Abstand der beiden Mittelwerte. Diese Prüfer erkennen keine Varianzunterschiede und liefern für unser Testpaar $0$.

<!--en:611c687f-->
Um neben der Lage auch die Streuung zu erfassen, ergänzen wir ein zweites Merkmal. Wir definieren die Merkmalsabbildung $\boldsymbol{\varphi}(x) = (x, x^2)$ und bilden Linearkombinationen $g(x) = \mathbf{w}^\top \boldsymbol{\varphi}(x)$ mit $\|\mathbf{w}\| \leq 1$. Schreiben wir $\mathbf{m}_P = \mathbb{E}_P[\boldsymbol{\varphi}(x)]$ für den *mittleren Merkmalsvektor* (engl. mean feature) von $P$ (und entsprechend $\mathbf{m}_Q$ für $Q$), so beträgt die Differenz der Mittelwerte $\mathbb{E}_P g - \mathbb{E}_Q g = \mathbf{w}^\top(\mathbf{m}_P - \mathbf{m}_Q)$. Nach der Cauchy-Schwarz-Ungleichung ist das Maximum einer linearen Funktion auf der Einheitskugel die euklidische Norm ihres Koeffizientenvektors:

<!--en:d370aecb-->
Damit ist die MMD der Abstand zwischen den mittleren Merkmalsvektoren von $P$ und $Q$. Das Supremum kommt in der Formel nicht mehr vor. Für unser Testpaar lauten die mittleren Merkmalsvektoren $\mathbf{m}_P = (0, 1)$ und $\mathbf{m}_Q = (0, 4)$. Das ergibt $\textrm{MMD}^2 = 0^2 + 3^2 = 9$. Der Vergleich erfasst nun Unterschiede sowohl in den Mittelwerten als auch in den zweiten Momenten.

<!--en:f7bcd665-->
Multiplizieren wir als Nächstes das Quadrat der Norm aus. Bei unabhängigen Ziehungen ist das Produkt der Erwartungswerte gleich dem Erwartungswert des Produkts. Daher gilt $\|\mathbf{m}_P\|^2 = \mathbb{E}\langle\boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x')\rangle$ für unabhängige $x, x' \sim P$. Ebenso verfahren wir mit unabhängigen $y, y' \sim Q$ und mit dem gemischten Term und erhalten $\|\mathbf{m}_P - \mathbf{m}_Q\|^2 =
\mathbb{E}\langle\boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x')\rangle +
\mathbb{E}\langle\boldsymbol{\varphi}(y), \boldsymbol{\varphi}(y')\rangle -
2\,\mathbb{E}\langle\boldsymbol{\varphi}(x), \boldsymbol{\varphi}(y)\rangle$. Die Merkmale $\boldsymbol{\varphi}(x)$ treten ausschließlich in inneren Produkten auf. Definieren wir den **Kernel** $k(x, x') = \langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x')\rangle$, so erhalten wir

<!--en:e9f8e003-->
Dieser Ausdruck lässt sich geometrisch deuten. Die ersten beiden Terme messen die mittlere Ähnlichkeit *innerhalb* von $P$ und *innerhalb* von $Q$, der dritte die Ähnlichkeit *zwischen* den Verteilungen. Mischen sich die Ziehungen aus $P$ und $Q$, so stimmen die Ähnlichkeiten innerhalb und zwischen den Stichproben überein. Die drei Terme heben sich auf. Bilden die Punkte getrennte Cluster, überwiegen die Terme innerhalb der Stichproben. Unser erster Schritt entspricht dem *linearen Kernel* $k(x, x') = x x'$, der zweite dem *quadratischen Kernel* $k(x, x') = x x' + x^2 x'^2$.

<!--en:abaae335-->
Auch der quadratische Kernel hat noch einen blinden Fleck. Zwei Verteilungen mit gleichem Mittelwert und gleicher Varianz haben stets den Abstand null. Wir könnten $x^3$ hinzufügen, dann $x^4$ und so weiter, ohne je fertig zu werden. Der entscheidende Schritt besteht deshalb darin, den Kernel direkt als Ähnlichkeitsmaß zu wählen, ohne eine endliche Liste von Merkmalen aufzustellen. Der *RBF-Kernel* $k(x, x') = e^{-(x - x')^2/(2\ell^2)}$ betrachtet zwei Punkte als ähnlich, wenn sie höchstens ungefähr eine *Längenskala* (engl. length scale) $\ell$ auseinanderliegen. Auch dieser direkt gewählte Kernel ist ein inneres Produkt von Merkmalsvektoren. Zerlegen wir ihn in $e^{-x^2/(2\ell^2)}\, e^{-x'^2/(2\ell^2)}\, e^{x x'/\ell^2}$ und entwickeln den letzten Faktor in eine Potenzreihe, erkennen wir ein inneres Produkt über unendlich viele Merkmale $\varphi_j(x) = x^j e^{-x^2/(2\ell^2)} / (\ell^j \sqrt{j!})$ für $j = 0, 1, 2, \ldots$. Es handelt sich um gedämpfte Monome jeder Ordnung mit rasch abnehmenden Gewichten. Unsere Gruppe umfasst nun unendlich viele Prüfer. Für ihr Urteil brauchen wir dennoch nur die drei Mittelwerte in :eqref:`eq_mdl-mmd2`. In $\mathbb{R}^n$ ersetzen wir lediglich $(x - x')^2$ durch $\|x - x'\|^2$.

<!--en:fdf78b6e-->
Mit diesem Kernel verschwindet der blinde Fleck. Es gilt $\textrm{MMD}(P, Q) = 0$ genau dann, wenn $P = Q$ :cite:`Gretton.Borgwardt.Rasch.ea.2012`. Die RBF-MMD ist somit eine Metrik. Kernel mit dieser Eigenschaft heißen *charakteristisch*. Dem linearen und dem quadratischen Kernel fehlt diese Eigenschaft.

<!--en:97a5dc91-->
Für Gauß-Verteilungen lassen sich die Erwartungswerte des RBF-Kernels in geschlossener Form berechnen. Gilt $Z \sim \mathcal{N}(\mu, s^2)$, so liefert ein bekanntes Gauß-Integral

<!--en:c42c8d9d-->
Die Differenz zweier unabhängiger gaußscher Zufallsvariablen ist wieder gaußverteilt. Ihre Varianz ist die Summe der beiden Varianzen. Damit lassen sich alle drei Terme in :eqref:`eq_mdl-mmd2` direkt ablesen. Für unser Testpaar $\mathcal{N}(0, 1)$ und $\mathcal{N}(0, 4)$ mit $\ell = 1$ haben die Differenzen $x - x'$, $y - y'$ und $x - y$ jeweils Mittelwert null und die Varianzen $2$, $8$ und $5$. Daraus folgt $\textrm{MMD}^2 = 3^{-1/2} + 9^{-1/2} - 2 \cdot 6^{-1/2} \approx 0.094$. Für die beiden verschobenen Gauß-Verteilungen der Eingangstabelle liefert dasselbe Integral $\tfrac{2}{\sqrt{3}}\big(1 - e^{-d^2/6}\big)$ (Aufgabe :exref:`ex_mdl-divergences-distances-q-mathcaln-delta`a). Bei den Punktmassen aus :numref:`tab_mdl-point-mass-table` entfällt die Mittelung ganz. Aus :eqref:`eq_mdl-mmd2` folgt unmittelbar $1 + 1 - 2k(0, d) = 2 - 2e^{-d^2/2}$, wie in der vierten Tabellenzeile. In beiden Fällen hängt nur der gemischte Term von $d$ ab. Er fällt rasch ab, sobald $d$ einige Längenskalen übersteigt. Die Differenz flacht daher ab und verfehlt unsere zweite Anforderung. Ein RBF-Prüfer ist glatt, aber kurzsichtig.

<!--en:0ee0f0d4-->
### Schätzung aus Stichproben

<!--en:4a747d74-->
Bei unserer ersten Anforderung zeigt sich die Stärke eines Kernels. Er ermöglicht eine zuverlässige Schätzung aus Stichproben. Für endliche Stichproben aus $P$ und $Q$ ersetzen wir jeden Erwartungswert in :eqref:`eq_mdl-mmd2` durch einen Mittelwert über Punktepaare. Um einen erwartungstreuen Schätzer :cite:`Gretton.Borgwardt.Rasch.ea.2012` zu erhalten, lassen wir die Diagonalterme weg. Diese paaren einen Punkt mit sich selbst und würden den Mittelwert innerhalb jeder Stichprobe nach oben verzerren. Im Folgenden werten wir den Schätzer für $N = 1{,}000$ Ziehungen aus $\mathcal{N}(0, 1)$ und $1{,}000$ Ziehungen aus $\mathcal{N}(0, 4)$ aus. Wir verwenden den linearen, den quadratischen und den RBF-Kernel, für die wir gerade die Populationswerte $0$, $9$ und $0.094$ berechnet haben. Für den RBF-Kernel bilden wir zwei weitere Paare, einmal mit einer zweiten unabhängigen Stichprobe aus $\mathcal{N}(0, 1)$ (Populationswert $0$) und einmal mit einer verschobenen Stichprobe aus $\mathcal{N}(0.5, 1)$ (Populationswert $0.047$ in der Eingangstabelle). Wie nahe kommen die Schätzwerte diesen Populationswerten?

<!--en:bcfb3752-->
Der lineare Kernel liefert einen Wert nahe null, weil beide Mittelwerte übereinstimmen. Genauer gesagt entspricht sein erwartungstreuer Schätzwert der quadrierten Differenz der beiden Stichprobenmittelwerte abzüglich der geschätzten Varianz dieser Differenz, die ungefähr $(1 + 4)/1{,}000 = 0.005$ beträgt. Der Schätzwert fällt daher selten wesentlich unter $-0.005$. Etwa jeder zwanzigste Durchlauf liefert einen Wert über $+0.015$. Der quadratische Kernel ergibt rund $9$ mit Schwankungen um etwa $3$, weil vierte Potenzen das Stichprobenrauschen verstärken. Beim RBF-Kernel liegt die Schätzung für das Paar mit unterschiedlichen Varianzen bei etwa $0.094 \pm 0.03$ und für das Paar mit unterschiedlichen Mittelwerten bei $0.047 \pm 0.02$. Der Stichprobenfehler kann also einen merklichen Teil des Signals ausmachen. Bei zwei unabhängigen Stichproben aus *derselben* Verteilung $\mathcal{N}(0, 1)$ streut die RBF-Schätzung hingegen erstaunlich wenig. Sie bleibt bis auf wenige Tausendstel bei null und fällt gelegentlich knapp darunter, weil sich die Stichprobenfehler erster Ordnung in den drei Mittelwerten aufheben. Wo der Totalvariationsabstand zwischen zwei Stichproben aus derselben Verteilung einen Unterschied von $1$ vortäuschte, erkennt der RBF-Kernel keinen.

<!--en:ba61cbad-->
Bei einer von null verschiedenen Differenz ist der Fehler des RBF-Schätzers also relativ groß. Für $P = Q$ ist er dagegen außerordentlich genau. Das macht ihn zu einem leistungsfähigen *Zweistichprobentest*. Bei $1{,}000$ Punkten je Stichprobe sind schon wenige Hundertstel für die RBF-basierte $\textrm{MMD}^2$ ein starkes Indiz für unterschiedliche Verteilungen. Dafür müssen wir weder einen Kritiker trainieren noch eine einzige Dichte auswerten.

<!--en:f9a67e01-->
### Die Zeugenfunktion

<!--en:df6f7e84-->
Da eine IPM als Supremum definiert ist, liegt die Frage nahe, welche Testfunktion die maximale Differenz erreicht. Bei der MMD wird das Supremum in :eqref:`eq_mdl-mmd-features` für $\mathbf{w}^\star \propto \mathbf{m}_P - \mathbf{m}_Q$ erreicht. Die resultierende Testfunktion heißt **Zeugenfunktion** (engl. witness) und lautet $g^\star(x) \propto \langle \mathbf{m}_P - \mathbf{m}_Q, \boldsymbol{\varphi}(x)\rangle = \mathbb{E}_P[k(x, X)] - \mathbb{E}_Q[k(x, Y)]$. Erneut genügen Auswertungen des Kernels. Bei unseren verschobenen Gauß-Verteilungen $\mathcal{N}(0, 1)$ und $\mathcal{N}(0.5, 1)$ sind die beiden Dichten Spiegelbilder bezüglich der Geraden $x = 0.25$. Die Zeugenfunktion der Population wechselt daher genau bei $0.25$ das Vorzeichen. Im Folgenden schätzen wir die Zeugenfunktion aus unseren Stichproben auf einem Gitter und zeichnen sie zusammen mit den beiden Dichten. Außerdem geben wir $\mathbb{E}_P g - \mathbb{E}_Q g$ für die unskalierte Zeugenfunktion $g = \langle \mathbf{m}_P - \mathbf{m}_Q, \boldsymbol{\varphi}\rangle$ aus. Diese Differenz entspricht $\|\mathbf{m}_P - \mathbf{m}_Q\|^2 = \textrm{MMD}^2$. Bei ihrer Schätzung berücksichtigen wir hier auch die Paare eines Punktes mit sich selbst.

<!--en:fe350ea2-->
Wie erwartet ist die Zeugenfunktion positiv, wo $P$ die höhere Dichte hat, und negativ, wo $Q$ die höhere Dichte hat. Ihre Nullstelle liegt nahe $x = 0.25$ (bei $1{,}000$ Punkten je Stichprobe schwankt sie von Lauf zu Lauf um ein bis zwei Zehntel). Die ausgegebene Differenz stimmt bis auf etwa $0.001$ mit der erwartungstreuen Schätzung aus der vorigen Zelle überein. Die kleine Abweichung entsteht durch die Paare identischer Punkte.

<!--en:02706763-->
Noch mehr als die Nullstelle verraten die Ausläufer der gezeichneten Zeugenfunktion. Um die Differenz $\mathbb{E}_P[g] - \mathbb{E}_Q[g]$ zu verkleinern, muss ein generatives Modell jedes seiner Samples entlang $g$ bergauf verschieben. Die Steigung der Zeugenfunktion an einem Modellsample ist also das Gradientensignal, das dieses Sample erhält. Zwischen den Glockenkurven ist die Zeugenfunktion steil und zieht die Samples kräftig zu den Daten hin. Einige Längenskalen weiter außen flacht die Zeugenfunktion gegen null ab. Ein weit entferntes Modellsample erhält daher kaum einen Gradienten. Hier sehen wir die Spalte für $\textrm{MMD}^2$ aus der Eingangstabelle aus der Sicht eines einzelnen Samples.

<!--en:62baba5f-->
Die MMD mit festem Kernel erfüllt in der Praxis also zwei Aufgaben. Sie dient als nichtparametrischer Zweistichprobentest und als Trainingsziel für generative Modelle, das ohne adversarielle Optimierung auskommt :cite:`Li.Swersky.Zemel.2015`. Die Kernel Inception Distance zur Bewertung von Bildgeneratoren (:numref:`sec_dcgan`) etwa ist der erwartungstreue MMD-Schätzer mit einem kubischen Kernel auf Bildmerkmalen aus einem tiefen Netz. Die sorgfältige Wahl der Längenskala $\ell$ ist entscheidend. Für $\ell = 1$ fiel die Steigung bei $d = 8$ auf $7 \times 10^{-5}$ (größere Längenskalen untersucht Aufgabe :exref:`ex_mdl-divergences-distances-q-mathcaln-delta`b). In der Praxis wählt man für $\ell$ häufig den Median der paarweisen Abstände zwischen Stichprobenpunkten oder summiert RBF-Kernel über mehrere Längenskalen. Letzteres Verfahren verwenden wir in der MMD-Variante des Trainings eines GAN-*Generators* in :numref:`sec_gan_objectives`.

<!--en:6da0e5db-->
## Optimaler Transport und Wasserstein-Abstand

<!--en:df8d0bff-->
Eine Spalte der Eingangstabelle flachte nie ab. $W_1$ behielt die konstante Steigung $1$, weil dieser Abstand die Geometrie des zugrunde liegenden Raums berücksichtigt. Die f-Divergenzen ignorieren sie, und ein fester Kernel erfasst sie nur lokal. Gemessen mit $W_1$ liegt die Punktmasse $\delta_{0.01}$ nahe bei $\delta_0$, weil ihre Masse nur die Strecke $0.01$ zurücklegen muss. Die Idee geht auf das Jahr 1781 zurück. Damals fragte Gaspard Monge, wie man einen Erdhaufen mit möglichst wenig Arbeit in eine Grube gleichen Volumens befördern kann. Diese minimale Arbeit ist ein natürliches Maß für den Abstand zweier Verteilungen.

<!--en:62e979c0-->
### Transportpläne

<!--en:e63a891a-->
Betrachten wir zwei diskrete Verteilungen auf den sechs Punkten $0, 1, \ldots, 5$ der reellen Geraden. Die Masse von $P$ liegt überwiegend links, die von $Q$ rechts, wie :numref:`fig_mdl-ot-transport-plan` oben links zeigt. Um $P$ in $Q$ zu überführen, müssen wir festlegen, wie viel Masse von jeder Quelle $x$ zu jedem Ziel $y$ gelangt. Ein solcher *Transportplan* (engl. transport plan) ist eine gemeinsame Verteilung $\gamma$ über Paare $(x, y)$ mit den Randverteilungen $P$ und $Q$. Von jedem Punkt $x$ geht insgesamt die Masse $P(x)$ aus, und an jedem Punkt $y$ kommt insgesamt die Masse $Q(y)$ an. In der Wahrscheinlichkeitstheorie heißt ein Transportplan auch *Kopplung* (engl. coupling) von $P$ und $Q$. Der **Wasserstein-1-Abstand** (auch Earth Mover’s Distance) ist das Minimum der erwarteten Weglänge über alle zulässigen Pläne. Bezeichnen wir mit $\Pi(P, Q)$ die Menge aller Kopplungen von $P$ und $Q$ auf $\mathbb{R}^n$ mit endlichen ersten Momenten, so gilt

<!--en:7ac359f7-->
Der Index $1$ steht für die erste Potenz des Abstands. Die Variante $W_2$ mit quadriertem Abstand lernen wir gleich kennen. Für zwei Punktmassen $\delta_0$ und $\delta_d$ gibt es nur einen Plan. Er verschiebt die gesamte Masse von $0$ nach $d$. Somit gilt $W_1(\delta_0, \delta_d) = |d|$ mit Gradient $\pm 1$ für $d \neq 0$. Bei den verschobenen Gauß-Verteilungen der Eingangstabelle ergibt die Verschiebung jedes Punkts um $d$ einen zulässigen Plan mit Kosten $d$. Kein Plan kann günstiger sein, denn es gilt $\mathbb{E}_\gamma|x - y| \geq |\mathbb{E}_\gamma[x - y]| = d$ für jede Kopplung $\gamma$. In beiden Fällen liefert $W_1$ den Abstand $d$ selbst und damit das klare Gradientensignal, das den f-Divergenzen fehlt.

<!--en:ea3eaca1-->
![Transport auf der Geraden. Oben links ein Transportplan zwischen zwei Verteilungen auf sechs Punkten. Jedes Band führt von einem Punkt von $P$ (oben) zu einem Punkt von $Q$ (unten). Seine Breite ist proportional zur transportierten Masse. Unten links die Verteilungsfunktionen von $P$ und $Q$. Die schattierte Fläche dazwischen beträgt $1.7$. Rechts der Vergleich für zwei Punktmassen im Abstand $d$. Die Jensen-Shannon-Divergenz (in Nat) bleibt konstant bei $\log 2$, während $W_1$ (in Einheiten von $x$) linear mit $d$ wächst.](../img/mdl-it-ot-transport-plan.svg)

<!--en:56cebc27-->
### Kantorovich-Rubinstein-Dualität

<!--en:b28f6b41-->
Als Rechenvorschrift wirkt :eqref:`eq_mdl-w1-primal` zunächst abschreckend. Wir minimieren über gemeinsame Verteilungen $\gamma$, die sich aus Stichproben nur schwer parametrisieren lassen. Auf endlichen Punktmengen handelt es sich jedoch um ein vertrautes Optimierungsproblem, ein lineares Programm. Die Unbekannten sind die Einträge des Transportplans. Sowohl die Transportkosten als auch die Nebenbedingungen für die Randverteilungen sind linear in diesen Einträgen. Zu jedem linearen Programm gehört ein *duales* Problem, bei dem man über die Multiplikatoren der Nebenbedingungen maximiert (:numref:`subsec_mdl-dual-problem`). Beim optimalen Transport ersetzt das duale Problem die Suche über gemeinsame Verteilungen durch die Maximierung über eine einzige Funktion. Erst diese Umformulierung macht $W_1$ für Deep Learning praktisch nutzbar.

<!--en:48bd65fc-->
**Proposition (Kantorovich-Rubinstein-Dualität).** *Für Verteilungen $P$ und $Q$ auf $\mathbb{R}^n$ mit endlichen ersten Momenten gilt*

<!--en:fdaf3b32-->
*Dabei läuft das Supremum über alle 1-Lipschitz-Funktionen $g$, das heißt, es gilt $|g(x) - g(y)| \leq \|x - y\|$ für alle $x, y$.*

<!--en:55b965d2-->
Aus dem Infimum über Kopplungen ist ein Supremum über eine einzige Testfunktion geworden. $W_1$ ist genau die IPM :eqref:`eq_mdl-ipm-def` für Funktionen mit Steigungsbetrag höchstens $1$. Das ist die dritte der oben angekündigten Klassen. Ein 1-Lipschitz-Prüfer kann keine Sprünge machen und daher einzelne Stichprobenpunkte nicht wie eine Indikatorfunktion auswendig lernen. Anders als beschränkte Funktionen oder RBF-Kernel kann er aber unbegrenzt wachsen und erfüllt damit unsere zweite Anforderung. Für die beiden verschobenen Gauß-Verteilungen der Eingangstabelle ist beispielsweise $g(x) = -x$ eine 1-Lipschitz-Funktion mit Erwartungswert $0$ unter $P$ und $-d$ unter $Q$. Sie erreicht somit die Differenz $d = W_1(P, Q)$. Ihr Gradient zeigt sogar überall mit Betrag $1$ in Richtung der Daten, unabhängig davon, wie weit ein Modellsample von den Daten entfernt liegt.

<!--en:b4d4c1bf-->
Da ein neuronales Netz eine solche Testfunktion approximieren kann, optimiert das **WGAN** :cite:`Arjovsky.Chintala.Bottou.2017` die duale Zielfunktion :eqref:`eq_mdl-kr-dual` direkt. Die Rolle von $g$ übernimmt ein neuronaler Kritiker, der die 1-Lipschitz-Bedingung näherungsweise erfüllen muss. In :numref:`sec_gan_objectives` vergleichen wir dieses Trainingsziel mit anderen adversariellen Verlusten und trainieren einen solchen Kritiker. Ein Strafterm für den Gradienten zieht den Betrag seiner Steigung in Richtung eins. In :numref:`sec_gan_convergence` sehen wir, warum spätere Verfahren den Strafterm stattdessen bei null zentrieren. Ein Strafterm, der den Steigungsbetrag eins bevorzugt, treibt einen Generator auch dann weiter an, wenn er die Datenverteilung bereits trifft. Ein bei null zentrierter Strafterm hält lediglich den Gradienten des Kritikers an den Samples klein und erzwingt keine Lipschitz-Bedingung. Erfüllt der Kritiker die Lipschitz-Bedingung nur näherungsweise und ist er auf eine neuronale Architektur beschränkt, so ist sein Score kein exakter Wert für $W_1$ mehr.

<!--en:676af8d2-->
### Transport auf der Geraden

<!--en:8e8faf6f-->
Wir beweisen die Kantorovich-Rubinstein-Dualität in einer Dimension. Dabei erhalten wir auch eine geschlossene Formel, die ganz ohne Optimierung auskommt. Den Beweis für höhere Dimensionen geben :citet:`Peyre.Cuturi.2019`. Die Idee besteht darin, für jeden Punkt $t \in \mathbb{R}$ zu zählen, wie viel Masse ihn überquert. Links von $t$ (einschließlich $t$) hat $P$ die Masse $F_P(t)$ und $Q$ die Masse $F_Q(t)$. Jeder zulässige Transportplan muss also mindestens die Masse $|F_P(t) - F_Q(t)|$ über $t$ hinweg transportieren. Ein Masseteilchen auf dem Weg von $x$ nach $y$ überquert außerdem jedes $t$ zwischen $x$ und $y$. Sein Weg $|x - y|$ entspricht daher der Länge des durchquerten Intervalls. Summieren wir über alle Teilchen, erhalten wir die gesamten Transportkosten als Integral über $t$ der insgesamt über $t$ transportierten Masse.

<!--en:02e24dd9-->
**Proposition ($W_1$ auf der Geraden).** *Für Verteilungen $P$ und $Q$ auf $\mathbb{R}$ mit Verteilungsfunktionen $F_P$, $F_Q$ und endlichen ersten Momenten gilt*

<!--en:fbeadf87-->
*Damit gilt die Kantorovich-Rubinstein-Dualität auf der reellen Geraden.*

<!--en:e6036242-->
**Beweis.** *Schritt 1: Jeder Plan kostet mindestens so viel wie die Fläche zwischen den Verteilungsfunktionen.* Für $(X, Y) \sim \gamma$ schreiben wir $|X - Y| = \int \mathbf{1}(t \textrm{ lies between } X \textrm{ and } Y)\, dt$. Da der Integrand nichtnegativ ist, dürfen wir Erwartungswert und Integration vertauschen:

<!--en:7addd8a0-->
Entscheidend ist die Vereinfachung der Differenz im letzten Integral. Addieren und subtrahieren wir $\gamma(X \leq t, Y \leq t)$, so erhalten wir $\gamma(X \leq t < Y) - \gamma(Y \leq t < X) = \gamma(X \leq t) - \gamma(Y \leq t) = F_P(t) - F_Q(t)$, denn $\gamma$ hat die Randverteilungen $P$ und $Q$. Jeder Transportplan kostet also mindestens $\int |F_P - F_Q|$.

<!--en:7ed0c138-->
*Schritt 2: Ein Plan erreicht diese untere Schranke.* Wir paaren die Quantile direkt. Das unterste Zehntel von $P$ schicken wir zum untersten Zehntel von $Q$ und fahren so entlang der Geraden fort. Formal sei $U$ gleichverteilt auf $(0, 1)$. Wir setzen $X = F_P^{-1}(U)$ und $Y = F_Q^{-1}(U)$. Für die Quantilfunktion $F^{-1}(\lambda) = \inf\{x : F(x) \geq \lambda\}$ mit $\lambda \in (0, 1)$ gilt dabei $F^{-1}(\lambda) \leq t$ genau dann, wenn $\lambda \leq F(t)$. Dies definiert eine zulässige Kopplung, denn es gilt $\Pr(X \leq t) = \Pr(U \leq F_P(t)) = F_P(t)$ und entsprechend für $Y$. Wir nennen sie den *monotonen Plan*, weil $X$ und $Y$ gemeinsam wachsen. Unter dieser Kopplung gelten $\gamma(X \leq t < Y) = \Pr\big(F_Q(t) < U \leq F_P(t)\big)$ und $\gamma(Y \leq t < X) = \Pr\big(F_P(t) < U \leq F_Q(t)\big)$. Bei jedem $t$ ist mindestens eine der beiden Wahrscheinlichkeiten null. Der monotone Plan transportiert also nie Masse in beide Richtungen über denselben Punkt. In der Ungleichung aus Schritt 1 gilt somit Gleichheit. Die minimalen Transportkosten entsprechen der Fläche zwischen den Verteilungsfunktionen.

<!--en:3f1db409-->
*Schritt 3: Der Wert keiner 1-Lipschitz-Testfunktion übersteigt die Kosten eines zulässigen Plans.* Dieses Argument der *schwachen Dualität* gilt in jeder Dimension. Für jede 1-Lipschitz-Funktion $g$ und jede Kopplung $\gamma$ gilt punktweise $g(x) - g(y) \leq \|x - y\|$. Bilden wir den Erwartungswert unter $\gamma$, folgt unmittelbar $\mathbb{E}_P[g] - \mathbb{E}_Q[g] \leq \mathbb{E}_\gamma\|x - y\|$ (vgl. die schwache Dualität in :numref:`subsec_mdl-dual-problem`).

<!--en:6d8fa4e2-->
*Schritt 4: Eine 1-Lipschitz-Testfunktion erreicht die Fläche.* Es genügt, Funktionen der Form $g(x) = g(0) + \int_0^x g'(t)\, dt$ mit $|g'| \leq 1$ zu betrachten. Jedes solche $g$ ist 1-Lipschitz-stetig, und für alle übrigen liefert Schritt 3 bereits eine obere Schranke. Die Konstante $g(0)$ fällt in $\mathbb{E}_P[g] - \mathbb{E}_Q[g]$ weg. Wir schreiben $\int_0^X g'(t)\,dt = \int_0^{\infty} g'(t)\, \mathbf{1}(X > t)\, dt - \int_{-\infty}^0 g'(t)\, \mathbf{1}(X \leq t)\, dt$ und vertauschen Erwartungswert und Integration. Das erlaubt der Satz von Fubini aus :numref:`sec_mdl-integral_calculus`, da die ersten Momente endlich sind. Damit erhalten wir

<!--en:f843a4f9-->
Mit der Wahl $g'(t) = \textrm{sign}\big(F_Q(t) - F_P(t)\big)$ wird die rechte Seite zu $\int |F_P(t) - F_Q(t)|\, dt$. Fassen wir die vier Schritte zusammen. Das Supremum über alle 1-Lipschitz-Funktionen ist mindestens so groß wie die Fläche (Schritt 4) und höchstens so groß wie die minimalen Plankosten (Schritt 3). Diese entsprechen genau der Fläche (Schritte 1 und 2). $\blacksquare$

<!--en:482e27b2-->
Auf der reellen Geraden ist $W_1$ also die Fläche zwischen den beiden Verteilungsfunktionen. Bei zwei gleich großen Stichproben genügt es, die sortierten Werte paarweise zu vergleichen und ihre absoluten Differenzen zu mitteln (Aufgabe :exref:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`a). Erstaunlich, dass sich ein ganzes lineares Programm so weit vereinfachen lässt. Prüfen wir das am Sechspunktbeispiel aus :numref:`fig_mdl-ot-transport-plan`. In den fünf Intervallen zwischen benachbarten Atomen betragen die Abstände der Verteilungsfunktionen $0.25$, $0.35$, $0.5$, $0.35$ und $0.25$. Ihre Summe $1.7$ ist die schattierte Fläche in der Abbildung. Der folgende Code berechnet $W_1$ auf drei Wegen. Zuerst summieren wir die fünf Abstände. Dann lösen wir das lineare Programm mit $36$ Variablen für Transportmatrizen der Form $6 \times 6$ mit Zeilensummen $p$ und Spaltensummen $q$. Schließlich konstruieren wir den monotonen Plan aus Schritt 2 und berechnen seine Kosten. Jede Route trägt dabei die Masse, die der Überlappung der zugehörigen Quantilintervalle entspricht. Zum Schluss prüfen wir, ob das lineare Programm den monotonen Plan geliefert hat.

<!--en:3c91e9d7-->
Alle drei Verfahren ergeben $W_1 = 1.7$. Fünf Abstände zwischen Verteilungsfunktionen zu addieren ergibt genau dasselbe wie das Lösen eines linearen Programms mit $36$ Variablen. Die letzte Zeile überrascht jedoch. Das lineare Programm hat einen anderen als den monotonen Plan gefunden. Er verursacht genau dieselben Kosten und transportiert ebenfalls keine Masse nach links.

<!--en:cd005b0d-->
Der optimale Plan für $W_1$ ist hier also nicht eindeutig. Der Grund ist leicht zu erkennen. Weil überall $F_P \geq F_Q$ gilt, muss keine Masse nach links fließen. Für *jeden* Plan, der Masse nur nach rechts verschiebt ($y \geq x$), gilt $|x - y| = y - x$. Seine Gesamtkosten entsprechen daher der Differenz der Mittelwerte, also $3.35 - 1.65 = 1.7$, unabhängig davon, wie sich die Routen kreuzen. Der in der Abbildung gezeigte monotone Plan ist nur einer von vielen gleich guten Plänen.

<!--en:8efc7206-->
Zwei Verfahren, die uns an anderen Stellen im Buch begegnen, ersetzen den absoluten Abstand durch sein Quadrat. Die Fréchet Inception Distance bewertet Bildgeneratoren (:numref:`sec_dcgan`). Der optimale Transport in Minibatches beim Flow-Matching paart Rauschen mit Daten, indem er den quadrierten Abstand minimiert (:numref:`sec_mdl-flow-matching-transport`). Der **Wasserstein-2-Abstand** mittelt den quadrierten Abstand und zieht anschließend die Quadratwurzel. Es gilt $W_2(P, Q) = \big(\inf_{\gamma} \mathbb{E}_\gamma \|x - y\|^2\big)^{1/2}$. Bei quadratischen Kosten ist es stets teurer, zwei Transportbänder zu kreuzen, als ihre Kreuzung aufzulösen. Auf der Geraden ist der monotone Plan daher das *eindeutige* Optimum (Aufgabe :exref:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`d). Für zwei eindimensionale Gauß-Verteilungen mit Mittelwerten $\mu_1, \mu_2$ und Standardabweichungen $\sigma_1, \sigma_2$ ergibt sich $W_2^2 = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$ (Aufgabe :exref:`ex_mdl-test-functions-transport-w2-gaussians`a). Die Fréchet Inception Distance verwendet die multivariate Entsprechung dieser Formel für Gauß-Verteilungen, die an Bildmerkmale angepasst wurden. An die Stelle der Varianzen treten Kovarianzmatrizen.

<!--en:9b825899-->
### Entropische Regularisierung und Sinkhorn-Iterationen

<!--en:d4fd933b-->
Diesen Unterabschnitt können Sie bei der ersten Lektüre überspringen. In höheren Dimensionen gibt es im Allgemeinen keine geschlossene Formel über Verteilungsfunktionen, und das lineare Programm mit $N$ Punkten auf jeder Seite hat $N^2$ Variablen. Das Standardverfahren, mit dem optimaler Transport schnell und differenzierbar wird, ist die **entropische Regularisierung** :cite:`Cuturi.2013`. Für eine Kostenmatrix $\mathbf{C}$ und eine Regularisierungsstärke $\varepsilon > 0$ addieren wir $\varepsilon \sum_{ij} \gamma_{ij} (\log \gamma_{ij} - 1)$ zu den Transportkosten $\sum_{ij} \gamma_{ij} C_{ij}$. Für jeden zulässigen Plan gilt $\sum_{ij}\gamma_{ij} = 1$. Der zusätzliche Term entspricht daher $-\varepsilon$ mal der Entropie von $\gamma$ zuzüglich einer Konstanten. Er benachteiligt Pläne, die ihre gesamte Masse auf wenige scharf abgegrenzte Routen konzentrieren. Die entropische Regularisierung hat den optimalen Transport als Verlustfunktion für Deep Learning praktikabel gemacht. Damit lassen sich generative Modelle direkt anhand regularisierter Transportkosten trainieren :cite:`Genevay.Peyre.Cuturi.2018`.

<!--en:f28246a0-->
Das regularisierte Problem lässt sich erstaunlicherweise fast in geschlossener Form lösen. Wir bilden die Lagrange-Funktion (:numref:`subsec_mdl-dual-problem`) mit einem Multiplikator $\xi_i$ für jede Zeilenbedingung $\sum_j \gamma_{ij} = p_i$ und einem Multiplikator $\zeta_j$ für jede Spaltenbedingung $\sum_i \gamma_{ij} = q_j$. Setzen wir die Ableitung nach $\gamma_{ij}$ gleich null, erhalten wir $C_{ij} + \varepsilon \log \gamma_{ij} - \xi_i - \zeta_j = 0$ oder gleichbedeutend $\gamma_{ij} = e^{-(C_{ij} - \xi_i - \zeta_j)/\varepsilon}$. Die Masse auf einer Route nimmt exponentiell damit ab, wie weit ihre Kosten $C_{ij}$ die Summe $\xi_i + \zeta_j$ übersteigen, gemessen in Einheiten von $\varepsilon$. In Matrixform gilt

<!--en:4fe4f3a8-->
Dabei sind $\mathbf{a} = e^{\boldsymbol{\xi}/\varepsilon}$ und $\mathbf{b} = e^{\boldsymbol{\zeta}/\varepsilon}$ positive Skalierungsvektoren. Den regularisierten Plan erhalten wir, indem wir den *Gibbs-Kernel* $\mathbf{K}$ zeilen- und spaltenweise skalieren. Dieser ist eine Matrix positiver Gewichte und hat mit den Kerneln der MMD nichts zu tun. Zu bestimmen sind nur noch $\mathbf{a}$ und $\mathbf{b}$. Jede Gruppe von Randbedingungen legt einen der beiden Vektoren fest. Halten wir $\mathbf{b}$ fest und setzen eintragsweise $\mathbf{a} = \mathbf{p} / (\mathbf{K}\mathbf{b})$, erhalten wir die Zeilensummen $\mathbf{p}$. Halten wir $\mathbf{a}$ fest, so erzwingt $\mathbf{b} = \mathbf{q} / (\mathbf{K}^\top \mathbf{a})$ die Spaltensummen $\mathbf{q}$.

<!--en:9149ad5e-->
Die **Sinkhorn-Iterationen** führen diese beiden Aktualisierungen abwechselnd aus. Wir aktualisieren erst $\mathbf{b}$, dann $\mathbf{a}$, bis beide Randverteilungen stimmen. Nach der Aktualisierung von $\mathbf{a}$ sind die Zeilensummen exakt. Wir stoppen daher, sobald die Spaltensummen von $\mathbf{q}$ um höchstens $10^{-9}$ abweichen. Jeder Schritt benötigt nur zwei Matrix-Vektor-Produkte. Der Algorithmus läuft daher auf GPUs schnell, und seine Schritte lassen sich leicht differenzieren. Im Folgenden geben wir die *entropischen Kosten* an. Damit meinen wir die Transportkosten $\sum_{ij} (\gamma_\varepsilon)_{ij} C_{ij}$ des regularisierten Plans ohne den Entropiestrafterm. Für $\varepsilon \to 0$ nähern sich diese Kosten dem exakten unregularisierten Optimum. Allerdings verlangsamt sich die Konvergenz drastisch :cite:`Peyre.Cuturi.2019`, und die Einträge von $\mathbf{K}$ können durch numerischen Unterlauf zu null werden. Implementierungen für den produktiven Einsatz rechnen daher im Logarithmus mit $\log \mathbf{a}$ und $\log \mathbf{b}$ (Aufgabe :exref:`ex_mdl-divergences-distances-sinkhorn-divergences`b).

<!--en:cbf52298-->
Um zu sehen, wie klein $\varepsilon$ sein muss und wie teuer ein kleines $\varepsilon$ wird, wenden wir das Sinkhorn-Verfahren mit den quadratischen Kosten $C_{ij} = (x_i - x_j)^2$ auf das Sechspunktbeispiel an. Der monotone Plan aus :numref:`fig_mdl-ot-transport-plan` ist hier das eindeutige Optimum. Seine zehn aktiven Routen verursachen Kosten von $W_2^2 = 3.5$. Die folgende Zelle überprüft dieses Optimum mit einem linearen Programm und führt das Sinkhorn-Verfahren für drei Werte von $\varepsilon$ aus. Sie gibt die Iterationszahl, die entropischen Kosten und die Anzahl der Routen mit einer Masse über $10^{-3}$ aus.

<!--en:94d64b0f-->
Das lineare Programm bestätigt $W_2^2 = 3.5$. Bei $\varepsilon = 2$ verteilt der regularisierte Plan die Masse jeder Quelle auf viele Ziele. Auf $25$ der $36$ Routen liegt mehr Masse als $10^{-3}$, und die Kosten steigen auf $4.06$. Bei $\varepsilon = 0.5$ konzentriert sich der Plan auf $14$ aktive Routen mit Kosten von $3.57$. Bei $\varepsilon = 0.1$ bleiben nur die zehn Routen des monotonen Plans übrig. Die Kosten betragen $3.50002$ und liegen damit bis auf $2 \times 10^{-5}$ am exakten Optimum. In den Heatmaps wird aus einem unscharfen Band eine klar abgegrenzte Treppe. Die natürliche Skala für $\varepsilon$ ist der Kostenunterschied zwischen konkurrierenden Routen. Auf diesem Gitter ist er von der Größenordnung $1$. Setzen wir $\varepsilon$ auf ein Zehntel dieser Skala, erhalten wir das Optimum mit hoher Genauigkeit.

<!--en:5fddf6d5-->
Die Iterationszahlen zeigen den Preis dafür. Das Sinkhorn-Verfahren benötigte etwa $50$ Iterationen bei $\varepsilon = 2$ und etwa $190$ bei $\varepsilon = 0.5$, aber rund $150{,}000$ bei $\varepsilon = 0.1$. Als wir $\varepsilon$ um den Faktor fünf verkleinerten, wuchs der Aufwand um mehr als den Faktor $800$. Sinkhorn-Iterationen sind nur dann schnell, wenn ein unscharfer Transportplan genügt (Aufgabe :exref:`ex_mdl-test-functions-transport-sinkhorn-iterations`b untersucht dieses Wachstum quantitativ).

<!--en:29740e0e-->
## Feste Kernel und trainierte Kritiker

<!--en:e1d1b7f4-->
MMD und $W_1$ erfüllen jeweils eine unserer beiden Anforderungen. Ein fester Kernel definiert eine kleine Klasse glatter Funktionen. Die MMD reduziert sich auf drei Stichprobenmittelwerte, und ihr Schätzfehler aus $N$ Punkten fällt wie $O(N^{-1/2})$ in *jeder* Dimension :cite:`Gretton.Borgwardt.Rasch.ea.2012`. Dem Fluch der Dimensionalität entgehen Kernel damit allerdings nicht. Die Fehlerrate ist dimensionsunabhängig, die Signalstärke hingegen nicht. Bei festem Kernel schrumpft die MMD auf Populationsebene zwischen zwei gegebenen Verteilungen typischerweise mit wachsender Dimension :cite:`Ramdas.Reddi.Poczos.ea.2015`. Deshalb berechnet man die Kernel Inception Distance und MMD-Verluste auf gelernten Merkmalen statt auf rohen Pixelwerten. Außerdem verliert ein fester Kernel die Daten jenseits weniger Längenskalen aus dem Blick. 1-Lipschitz-Funktionen liefern dagegen bei jedem Abstand einen beständigen Gradienten, genau das, was ein Modell weit entfernt von seinen Daten braucht. Zwar können sie keine Sprünge machen wie Indikatorfunktionen, doch die Klasse aller 1-Lipschitz-Funktionen ist in hohen Dimensionen immer noch enorm groß. Für zwei Stichproben in Dimension $n > 2$ fällt der Fehler des empirischen $W_1$ nur wie $O(N^{-1/n})$ :cite:`Weed.Bach.2019`. In $100$ Dimensionen bräuchte man für eine Halbierung dieses Schätzfehlers die $2^{100}$-fache Datenmenge. Niemand schätzt $W_1$ auf diesem Weg direkt aus hochdimensionalen Stichproben. Stattdessen sucht ein WGAN-Kritiker in einer viel kleineren Klasse neuronaler Netze. Für kleinere diskrete Zuordnungsprobleme eignen sich Sinkhorn-Iterationen.

<!--en:1ef58726-->
Sowohl bei Kerneln als auch bei adversariellen Kritikern ziehen wir zunächst Samples aus dem Modell. Bei manchen generativen Modellen ist bereits das Sampling der Engpass. In :numref:`sec_mdl-scores` fragen wir daher, wie wir zwei Verteilungen vergleichen können, ohne überhaupt Modellsamples zu ziehen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:60807c25-->
Überlappende Ausläufer helfen den f-Divergenzen wenig. Liegen zwei Gauß-Verteilungen mit Einheitsvarianz acht Standardabweichungen auseinander, haben die Jensen-Shannon-Divergenz und der Totalvariationsabstand Steigungen nahe $10^{-4}$. Integrale Wahrscheinlichkeitsmetriken kommen ohne Dichteverhältnisse aus. Sie vergleichen, wie $P$ und $Q$ die Funktionen einer Klasse von Testfunktionen mitteln. Die Klasse bestimmt sowohl die Stichprobeneffizienz als auch die Reichweite der Metrik. Testfunktionen, die linear in festen Merkmalen sind, ergeben die MMD, den Abstand zwischen den mittleren Merkmalsvektoren. Sie lässt sich ohne Training eines Kritikers in geschlossener Form aus drei Kernel-Mittelwerten berechnen. Die Schätzung konvergiert unabhängig von der Dimension mit der Rate $O(N^{-1/2})$. Allerdings schrumpft die MMD selbst meist mit wachsender Dimension, und ein RBF-Kernel erfasst Unterschiede jenseits weniger Längenskalen kaum noch. Funktionen mit Steigungsbetrag höchstens eins liefern $W_1$. Dieser Abstand gibt die minimalen Kosten (Masse mal Weg) für den Transport von $P$ nach $Q$ an. Er behält bei jeder Verschiebung die Steigung eins und bildet die Grundlage für den Kritiker des Wasserstein-GAN. In einer Dimension reduziert sich $W_1$ auf die Fläche zwischen den beiden Verteilungsfunktionen. In höheren Dimensionen braucht man entweder einen trainierten Kritiker oder, bei kleinen diskreten Problemen, Sinkhorn-Iterationen mit entropischer Regularisierung.

<!--en:2294243a-->
## Übungen

<!--en:f1169e0a-->
1. [code] **MMD in geschlossener Form.** Verwenden Sie das Gauß-Integral :eqref:`eq_mdl-rbf-gaussian-integral` für den RBF-Kernel mit Längenskala $\ell$ und die Verteilungen $P = \mathcal{N}(0, 1)$ und $Q = \mathcal{N}(d, 1)$.
    1. Zeigen Sie, dass $\textrm{MMD}^2 = \frac{2\ell}{\sqrt{\ell^2 + 2}}
       \big(1 - e^{-d^2/(2(\ell^2 + 2))}\big)$ gilt. Prüfen Sie, dass Sie für $\ell = 1$ die Spalte für $\textrm{MMD}^2$ aus der Eingangstabelle erhalten. Welche Längenskala maximiert $\textrm{MMD}^2$ für kleine Verschiebungen $d$? Was geschieht für $\ell \to 0$ und für $\ell \to \infty$?
    1. Berechnen Sie die Steigung bezüglich $d$ an der Stelle $d = 8$ für $\ell = 1$, $4$ und $16$. Was sagen diese drei Werte über die Wahl der Längenskala für ein Modell aus, das weit von seinen Daten entfernt startet?
   :exlabel:`ex_mdl-divergences-distances-q-mathcaln-delta`

<!--en:4ee24eea-->
1. [code] **Eindimensionaler Transport.** ● Auf der reellen Geraden ist $W_1$ die Fläche zwischen den Verteilungsfunktionen :eqref:`eq_mdl-w1-cdf`.
    1. Betrachten Sie zwei empirische Verteilungen zu gleich großen Stichproben $x_1, \ldots, x_N$ und $y_1, \ldots, y_N$. Zeigen Sie mit dem monotonen Plan aus dem Beweis, dass $W_1 = \tfrac{1}{N} \sum_i |x_{(i)} - y_{(i)}|$ gilt, also die mittlere absolute Differenz der sortierten Werte den Abstand ergibt. Prüfen Sie das Ergebnis mit dem linearen Programm aus dem Sechspunktbeispiel an zwei Zufallsstichproben vom Umfang $7$.
    1. Berechnen Sie aus den sortierten Stichproben des MMD-Experiments den Abstand $W_1$ zwischen `x_p` und `y_shift` sowie zwischen `x_p` und `y_same`. Mitteln Sie diesen Abstand dann über $20$ Paare unabhängiger Stichproben vom Umfang $N$ aus $\mathcal{N}(0, 1)$ für $N \in \{10^2, 10^3, 10^4\}$. Bestimmen Sie die Rate, mit der er abnimmt. Der Schätzwert ist nie negativ. Warum schließt dies Erwartungstreue für $P = Q$ aus?
    1. Ändern Sie `q_w` in $(0.40, 0.05, 0.05, 0.10, 0.10, 0.30)$, sodass sich die beiden Verteilungsfunktionen kreuzen. Führen Sie die Zelle zur Berechnung von $W_1$ erneut aus. Konstruieren Sie wie in Schritt 4 des Beweises die Testfunktion $g$ mit $g(0) = 0$ und der Steigung $\textrm{sign}\big(F_Q(t) - F_P(t)\big)$ zwischen benachbarten Atomen. Prüfen Sie, ob $\mathbb{E}_P[g] - \mathbb{E}_Q[g]$ dem LP-Wert entspricht. Vergleichen Sie für jede Route $(i, j)$, die in der LP-Lösung `plan_lp` Masse trägt, die Werte $g(x_i) - g(x_j)$ und $|x_i - x_j|$. Warum müssen sie bei jedem optimalen Plan und jeder optimalen Testfunktion auf allen solchen Routen übereinstimmen?
    1. *Eindeutigkeit bei quadratischen Kosten.* Ein Plan transportiere Masse von $x$ nach $y'$ und von $x'$ nach $y$, wobei $x < x'$ und $y < y'$ gelten. Die beiden Bänder kreuzen sich also. Zeigen Sie, dass es bei quadratischen Kosten günstiger ist, dieselbe Masse von $x$ nach $y$ und von $x'$ nach $y'$ zu schicken. Die Ersparnis beträgt $2(x' - x)(y' - y)$ je Masseneinheit. Folgern Sie daraus, dass allein der monotone Plan optimal ist. Wie viel spart derselbe Tausch bei den Kosten $|x - y|$, wenn alle vier Wege nach rechts führen? Finden Sie zwei sich kreuzende Bänder in `plan_lp`, dem Plan des linearen Programms für das Sechspunktbeispiel. Prüfen Sie daran Ihre Antwort.
   :exlabel:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`

<!--en:fbd0a3ed-->
1. **Optimale Kritiker auf der Geraden.** Schritt 4 im Beweis von :eqref:`eq_mdl-w1-cdf` zeigt, dass eine Testfunktion mit der Steigung $g'(t) = \textrm{sign}\big(F_Q(t) - F_P(t)\big)$ den Wert $W_1$ erreicht.
    1. Zeigen Sie für $P = \mathcal{N}(0, 1)$ und $Q = \mathcal{N}(d, 1)$ mit $d > 0$, dass überall $F_P > F_Q$ gilt. Schritt 4 liefert also $g(x) = -x$, den Kritiker, den wir im Text direkt ablesen konnten.
    1. Zeigen Sie für $P = \mathcal{N}(0, 1)$ und $Q = \mathcal{N}(0, 4)$, dass sich die beiden Verteilungsfunktionen nur bei $0$ kreuzen, dass $g(x) = -|x|$ optimal ist und dass $W_1 = \sqrt{2/\pi} \approx 0.80$ gilt. (Für $X \sim \mathcal{N}(0, \sigma^2)$ gilt $\mathbb{E}|X| = \sigma\sqrt{2/\pi}$.) Der lineare Kernel lieferte für dieses Paar $0$. In welche Richtung verschiebt der Kritiker die Samples von $Q$?
    1. Es seien $P = \tfrac12 \delta_0 + \tfrac12 \delta_3$ und $Q = \tfrac12 \delta_1 + \tfrac12 \delta_3$. Bestimmen Sie $W_1(P, Q)$ und beschreiben Sie alle optimalen Kritiker der in Schritt 4 verwendeten Form. Zeigen Sie dann allgemein, dass ein solcher Kritiker genau dann optimal ist, wenn $g'(t) = \textrm{sign}\big(F_Q(t) - F_P(t)\big)$ an fast jedem Punkt $t$ mit $F_P(t) \neq F_Q(t)$ gilt. Das sind die Punkte, die Masse überqueren muss. An den übrigen Punkten ist die Steigung im Rahmen der Lipschitz-Bedingung frei wählbar.
    1. Beim Training eines WGAN-Kritikers verwendet man häufig einen Strafterm für $(|g'(x)| - 1)^2$ an Punkten $x$ zwischen den Daten und den generierten Samples :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017`. Erklären Sie diese Wahl anhand von (c). Welche Kritiker sind für $P = Q$ optimal, und welche davon bevorzugt der Strafterm? In :numref:`sec_gan_convergence` sehen Sie, was daraus für das Training folgt.
   :exlabel:`ex_mdl-test-functions-transport-optimal-critics`

<!--en:c17f05cf-->
1. [code] **$W_2$ zwischen Gauß-Verteilungen und der Fréchet-Abstand.** Auf der Geraden ist der monotone Plan $X = F_P^{-1}(U)$, $Y = F_Q^{-1}(U)$ für quadratische Kosten optimal.
    1. Zeigen Sie, dass der monotone Plan für $P = \mathcal{N}(\mu_1, \sigma_1^2)$ und $Q = \mathcal{N}(\mu_2, \sigma_2^2)$ durch $Y = \mu_2 + (\sigma_2/\sigma_1)(X - \mu_1)$ gegeben ist. Folgern Sie daraus $W_2^2(P, Q) = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$. Leiten Sie mit demselben Plan für gleiche Varianzen die Formel $W_1 = |\mu_1 - \mu_2|$ her, also die Spalte für $W_1$ aus der Eingangstabelle.
    1. Beide Standardabweichungen seien $\sigma$, und die Mittelwerte sollen im Abstand $d$ liegen. Dann gilt $W_2^2 = d^2$ für jedes $\sigma$, während die KL-Divergenz zwischen den beiden Gauß-Verteilungen $d^2/(2\sigma^2)$ beträgt. Was ergeben beide Größen für $\sigma \to 0$, wenn die Gauß-Verteilungen zu den Punktmassen aus :numref:`tab_mdl-point-mass-table` werden?
    1. In $\mathbb{R}^n$ ist das Quadrat von $W_2$ zwischen $\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ und $\mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$ der quadrierte *Fréchet-Abstand* $\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|^2 + \textrm{tr}\big(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2 - 2\,(\boldsymbol{\Sigma}_1^{1/2} \boldsymbol{\Sigma}_2 \boldsymbol{\Sigma}_1^{1/2})^{1/2}\big)$ :cite:`Dowson.Landau.1982`. Prüfen Sie, dass sich dieser Ausdruck für $n = 1$ auf (a) reduziert und für diagonale Kovarianzmatrizen auf die Summe der Ausdrücke aus (a) über alle Koordinaten. Beweisen Sie den diagonalen Fall direkt. Die quadratischen Kosten sind eine Summe über Koordinaten. Daher kostet kein Plan weniger als die Summe der koordinatenweisen Optima, und die monotonen Pläne für die einzelnen Koordinaten erreichen diese Summe gemeinsam.
    1. Für die Fréchet Inception Distance passt man an zwei Merkmalsverteilungen jeweils eine Gauß-Verteilung an und berechnet den quadrierten Fréchet-Abstand zwischen den angepassten Verteilungen. Sei $P$ die Mischung $0.7\,\mathcal{N}(-2, 0.6^2) + 0.3\,\mathcal{N}(2, 0.6^2)$ und sei $Q = \mathcal{N}(-0.8, 3.72)$ die Gauß-Verteilung mit demselben Mittelwert und derselben Varianz. Wie groß ist der Fréchet-Abstand zwischen den angepassten Verteilungen? Schätzen Sie $W_2^2(P, Q)$ aus je $10^5$ sortierten Ziehungen (Sie sollten etwa $0.63$ erhalten). Was lässt sich aus einem Fréchet-Abstand von null schließen, und was bleibt offen?
   :exlabel:`ex_mdl-test-functions-transport-w2-gaussians`

<!--en:f58c2eb6-->
1. [code] **Entropischer Transport und Sinkhorn-Divergenzen.** ●● Regularisieren Sie das lineare Transportprogramm des Sechspunktbeispiels mithilfe der Divergenz gegenüber dem unabhängigen Plan:

   $$
   \textrm{OT}_\varepsilon(P, Q) = \min_{\gamma}\, \sum_{i, j} \gamma_{ij} C_{ij}
   + \varepsilon\, D_{\textrm{KL}}\big(\gamma \,\|\, p\, q^\top\big),
   $$

   Dabei durchläuft $\gamma$ alle Pläne mit Zeilensummen $p$ und Spaltensummen $q$.
    1. Zeigen Sie, dass $D_{\textrm{KL}}(\gamma \,\|\, p\, q^\top) = -H(\gamma) + H(P) + H(Q)$ für jeden solchen Plan gilt. Dabei bezeichnet $H$ die Entropie. Folglich liefert `sinkhorn` den minimierenden Plan. Zeigen Sie, dass $\textrm{OT}_\varepsilon(P, Q) \to \sum_{i, j} p_i q_j C_{ij}$ für $\varepsilon \to \infty$ gilt und dass $\textrm{OT}_\varepsilon(P, P) > 0$ für jedes $\varepsilon > 0$ gilt, sofern $P$ keine Punktmasse ist. Verwenden Sie die Kosten $C_{ij} = |x_i - x_j|$. Zeigen Sie, dass die auf den Atomen liegenden Verteilungen $P$, die den Grenzwert $\sum_{i, j} p_i q_j C_{ij}$ minimieren, ihre gesamte Masse auf Mediane von $Q$ legen. Für `q_w` sind dies die Atome $3$ und $4$. Was würde ein Modell lernen, das mit $\textrm{OT}_\varepsilon$ bei großem $\varepsilon$ trainiert wird?
    1. Die Sinkhorn-Divergenz $S_\varepsilon(P, Q) = \textrm{OT}_\varepsilon(P, Q)
       - \tfrac{1}{2} \textrm{OT}_\varepsilon(P, P)
       - \tfrac{1}{2} \textrm{OT}_\varepsilon(Q, Q)$ beseitigt diese Verzerrung :cite:`Genevay.Peyre.Cuturi.2018,Feydy.Sejourne.Vialard.ea.2019`. Berechnen Sie $S_\varepsilon(P, Q)$ für `p_w` und `q_w` mit den absoluten Kosten `C` für $\varepsilon \in \{10^{-3}, 10^{-2}, \ldots, 10^3\}$. Bei $\varepsilon = 10^{-3}$ tritt für Einträge von $\mathbf{K}$ numerischer Unterlauf auf. Führen Sie die Sinkhorn-Aktualisierungen dort für $\log \mathbf{a}$ und $\log \mathbf{b}$ mithilfe von `scipy.special.logsumexp` aus. Prüfen Sie, dass $S_\varepsilon$ gegen $W_1$ für $\varepsilon \to 0$ und gegen $\tfrac{1}{2} \mathcal{E}(P, Q)$ für $\varepsilon \to \infty$ konvergiert. Dabei ist $\mathcal{E}(P, Q) = 2\mathbb{E}|X - Y| - \mathbb{E}|X - X'| - \mathbb{E}|Y - Y'|$ der Energieabstand :cite:`Szekely.Rizzo.2013`. Er entspricht dem Ausdruck :eqref:`eq_mdl-mmd2` mit $-|x - y|$ anstelle des Kernels. Leiten Sie den zweiten Grenzwert aus (a) her.
   :exlabel:`ex_mdl-divergences-distances-sinkhorn-divergences`

<!--en:5c27582e-->
1. [code] **Sinkhorn-Iterationen für kleine $\varepsilon$.** ● Verwenden Sie `sinkhorn`, `p_w` und `q_w` aus dem Sechspunktbeispiel.
    1. *Vorzeitiger Abbruch.* Rufen Sie `sinkhorn` mit den quadratischen Kosten `C2` für $\varepsilon = 0.1$ und `max_iters=5000` auf, sodass die Schleife endet, bevor sich die Spaltensummen stabilisiert haben. Die zurückgegebenen Kosten liegen unter dem Optimum $3.5$. Geben Sie die Spaltensummen der zurückgegebenen Matrix abzüglich `q_w` aus. Welche Ziele erhalten zu wenig Masse, und warum sinken dadurch die Kosten? Warum kann keine Matrix mit den Zeilensummen `p_w` und den Spaltensummen `q_w` weniger als $3.5$ kosten?
    1. *Iterationszahl.* Zählen Sie die Iterationen, die `sinkhorn` mit `C2` für $\varepsilon \in \{2, 1, 0.5, 0.3, 0.2, 0.15, 0.1\}$ benötigt. Tragen Sie den Logarithmus der Iterationszahl gegen $1/\varepsilon$ auf. Welche Art von Wachstum zeigen die Punkte für kleine $\varepsilon$, und welche Iterationszahl lässt sich daraus für $\varepsilon = 0.05$ vorhersagen?
    1. *Mehrere gleich gute Pläne.* Bei den absoluten Kosten `C` ist jeder Plan, der Masse nur nach rechts verschiebt, für `p_w` und `q_w` optimal. Führen Sie `sinkhorn` mit `C` für $\varepsilon = 0.1$ und $0.02$ aus. Wie viele Routen tragen Masse? Zeigen Sie, dass der Grenzplan die Form $\gamma_{ij} = a_i b_j$ für $j \geq i$ und sonst $0$ hat. Für zulässige Pläne entspricht der Regularisierungsterm dem Produkt aus $-\varepsilon$ und der Entropie zuzüglich einer Konstanten. Erklären Sie damit, warum der Grenzplan unter den optimalen Plänen die größte Entropie hat.
   :exlabel:`ex_mdl-test-functions-transport-sinkhorn-iterations`

<!--en:2270e891-->
[Diskussion](https://d2l.discourse.group/t/36895)

<!--en:be67b4bd-->
Testfunktionen und optimaler Transport<br>**integrale Wahrscheinlichkeitsmetriken · Kernel-MMD · Zeugenfunktionen · Wasserstein-1 · Kantorovich-Rubinstein-Dualität · Sinkhorn**

<!--en:fca0bbb8-->
::: {.slide title="Hilft die Überlappung der Ausläufer bei Verlusten auf Basis des Dichteverhältnisses?"}

<!--en:ca82c91d-->
[Motivation anhand verschobener Gauß-Verteilungen]{.kicker}

<!--en:1e24d024-->
Vergleich zweier überlappender Gauß-Verteilungen $P = \mathcal{N}(0, 1)$ und $Q = \mathcal{N}(d, 1)$ im Abstand von $d = 8$ Standardabweichungen.

<!--en:15d54190-->
- **Glatte Ausläufer helfen weder JS und TV noch der lokalen MMD:** Bei $d = 8$ sind $\textrm{JS} \approx \log 2$ und $\textrm{TV} \approx 1$ gesättigt. Die Ableitungen $\partial\textrm{JS}/\partial d$, $\partial\textrm{TV}/\partial d$ und $\partial\textrm{MMD}^2/\partial d$ (für $\ell = 1$) fallen auf $\sim 10^{-4}$.

<!--en:f37a2010-->
- **Wasserstein-1 ($W_1$) behält seine Steigung:** Die Steigung $\partial W_1/\partial d = 1.00$ bleibt bei jedem Abstand $d$ erhalten, auch ohne explizite Modelldichte $q(x)$.

<!--en:7cc6d33f-->
::: {.slide title="Werte und Gradienten beim Abstand d"}

<!--en:decb28dc-->
[Fünf Vergleichsgrößen für verschobene Gauß-Verteilungen]{.kicker}

<!--en:d095fa0e-->
Für alle fünf Größen lassen sich die Werte berechnen und die Ableitungen nach $d$ durch zentrale Differenzen annähern. Das zeigt, wie schnell beschränkte Verhältnisgrößen und lokale Kernel entfernte Daten aus dem Blick verlieren.

<!--en:3fd810ca-->
::: {.slide title="Integrale Wahrscheinlichkeitsmetriken als Gruppe von Prüfern"}

<!--en:f49c1bcf-->
[Vergleiche mit Testfunktionen unter einem Dach]{.kicker}

<!--en:54c2f5b1-->
Eine **integrale Wahrscheinlichkeitsmetrik (IPM)** vergleicht $P$ und $Q$ mithilfe einer symmetrischen Klasse $\mathcal{G}$ von Testfunktionen („Prüfern“) $g$:

<!--en:8aa71ed8-->
- **Drei Möglichkeiten für $\mathcal{G}$:** Die Schranke $|g| \le \tfrac12$ liefert **TV**, die Einheitskugel im RKHS mit $\|\mathbf{w}\|_2 \le 1$ die **MMD** (in geschlossener Form) und die 1-Lipschitz-Klasse $\textrm{Lip}_1$ den **Wasserstein-Abstand $W_1$** (WGAN).

<!--en:fea7bb8d-->
::: {.slide title="Zwei gegenläufige Anforderungen an die Testklasse G"}

<!--en:2fb74b4b-->
[Stichprobeneffizienz und Gradienten mit großer Reichweite]{.kicker}

<!--en:4800670c-->
- **Problem 1 ($\mathcal{G}$ zu groß oder mit zu spitzen Funktionen):** Unter der Schranke $|g(x)| \le \tfrac12$ haben zwei unabhängige Stichproben vom Umfang $N$ aus $\mathcal{N}(0, 1)$ fast sicher keinen gemeinsamen Punkt. Deshalb liefert $g(x) = \mathbf{1}_{x \in \{X_i\}} - \tfrac12$ den empirischen Abstand $\textrm{TV}(\hat{P}_N, \hat{Q}_N) = 1.00$, selbst wenn $P = Q$ gilt.

<!--en:fc941298-->
- **Problem 2 (Funktionswerte in $\mathcal{G}$ beschränkt):** Für jede Klasse mit $|g(x)| \le M$ ist die Differenz auf $2M$ begrenzt. Bei weit entfernten Modi tritt Sättigung ein, und der Gradient verschwindet.

<!--en:e2ae02c7-->
- **Lösung durch 1-Lipschitz-Funktionen:** Die Steigungsschranke $|\nabla g| \le 1$ verhindert das Auswendiglernen der Stichprobe *und* erlaubt lineares Wachstum mit dem Abstand $d$.

<!--en:201f6d01-->
::: {.slide title="Von linearen und quadratischen Merkmalen zur Mittelwerteinbettung"}

<!--en:311653e3-->
[Herleitung der Maximum Mean Discrepancy]{.kicker}

<!--en:688c35cf-->
Für die Merkmalsabbildung $\boldsymbol{\varphi}(x)$ und die Mittelwerteinbettung $\mathbf{m}_P = \mathbb{E}_{x \sim P}[\boldsymbol{\varphi}(x)]$ liefert die Cauchy-Schwarz-Ungleichung das Supremum über $\|\mathbf{w}\|_2 \le 1$ in geschlossener Form:

<!--en:ad70793a-->
- **Vielfalt der Merkmale:** Das lineare Merkmal $\varphi(x) = x$ erkennt den Varianzunterschied zwischen $\mathcal{N}(0,1)$ und $\mathcal{N}(0,4)$ nicht. Dagegen trennt $\boldsymbol{\varphi}(x) = (x, x^2)$ die Verteilungen ($\|\mathbf{m}_P - \mathbf{m}_Q\|_2 = 3$), und der charakteristische RBF-Kernel bettet alle Momente $j = 0, \ldots, \infty$ ein.

<!--en:8764f183-->
::: {.slide title="Mit dem Kernel-Trick entfällt das Supremum"}

<!--en:45044519-->
[Paarweise Erwartungswerte in geschlossener Form]{.kicker}

<!--en:06acf4a4-->
Ausmultiplizieren von $\|\mathbf{m}_P - \mathbf{m}_Q\|_2^2$ mithilfe von $k(x, x') = \langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x') \rangle$ liefert drei Erwartungswerte über Punktepaare:

<!--en:51730f41-->
- **Kein Kritiker zu trainieren:** Die Ähnlichkeiten innerhalb der Gruppen abzüglich der doppelten Ähnlichkeit zwischen ihnen ergeben bei charakteristischen Kerneln genau dann $0$, wenn $P = Q$ gilt.

<!--en:f9e0cfdc-->
::: {.slide title="Charakteristische Kernel und RBF-MMD in geschlossener Form"}

<!--en:f95f4d1b-->
[Unendlichdimensionale Merkmalsräume]{.kicker}

<!--en:7fe823e5-->
Jeder polynomiale Kernel endlichen Grades (etwa $k(x, x') = xx' + x^2 x'^2$) ist für höhere Momente blind. Mit dem **RBF-Kernel (Gauß-Kernel)** $k(x, x') = \exp\!\big(-\frac{\|x - x'\|^2}{2\ell^2}\big)$ mit Längenskala $\ell > 0$ lassen sich alle Momente zugleich vergleichen:

<!--en:4e145042-->
- **Entwicklung in unendlich viele Merkmale:** Die Zerlegung $e^{-(x-x')^2/(2\ell^2)} = e^{-x^2/(2\ell^2)}e^{-x'^2/(2\ell^2)}\sum_{j=0}^\infty \frac{(xx'/\ell^2)^j}{j!}$ zeigt ein inneres Produkt über unendlich viele gedämpfte Monome $\varphi_j(x) = \frac{x^j}{\ell^j\sqrt{j!}}e^{-x^2/(2\ell^2)}$ als Merkmale.

<!--en:8a11cc3e-->
- **Charakteristischer Kernel:** Da diese Merkmale alle Verteilungen unterscheiden, gilt $\textrm{MMD}(P, Q) = 0 \iff P = Q$. Die RBF-MMD ist damit eine Metrik auf Wahrscheinlichkeitsverteilungen (Gretton et al., 2012).

<!--en:a6da5bbc-->
- **Geschlossene Form für Gauß-Verteilungen ($\ell = 1$):** Es gilt $\mathbb{E}[e^{-Z^2/(2\ell^2)}] = (1 + s^2/\ell^2)^{-1/2}e^{-\mu^2/(2(\ell^2+s^2))}$ für $Z \sim \mathcal{N}(\mu, s^2)$.
  - Für $\mathcal{N}(0, 1)$ und $\mathcal{N}(0, 4)$ gelten $x - x' \sim \mathcal{N}(0, 2)$, $y - y' \sim \mathcal{N}(0, 8)$ und $x - y \sim \mathcal{N}(0, 5)$. Daraus folgt $\textrm{MMD}^2 = \frac{1}{\sqrt{3}} + \frac{1}{\sqrt{9}} - \frac{2}{\sqrt{6}} \approx 0.0942$.
  - Für die verschobenen Gauß-Verteilungen $\mathcal{N}(0, 1)$ und $\mathcal{N}(d, 1)$ folgt $\textrm{MMD}^2 = \frac{2}{\sqrt{3}}(1 - e^{-d^2/6})$. Für $d \gg \ell$ tritt Sättigung ein.

<!--en:82330c4f-->
::: {.slide title="Erwartungstreue Schätzung aus Stichproben mit einer U-Statistik"}

<!--en:4b32e6e7-->
[Zweistichprobentest ohne Kritiker]{.kicker}

<!--en:55ce74ed-->
Für endliche Stichproben $\{x_i\}_{i=1}^n \sim P$ und $\{y_j\}_{j=1}^m \sim Q$ entfallen die Diagonalpaare ($i = j$). Sie würden die Ähnlichkeiten innerhalb einer Stichprobe um $k(x_i, x_i)/n = 1/n$ nach oben verzerren. Das ergibt eine erwartungstreue U-Statistik:

<!--en:4bf38f5e-->
::: {.slide title="Die Zeugenfunktion und verschwindende Gradienten in den Ausläufern"}

<!--en:d7eb4770-->
[Warum die MMD mit fester Bandbreite kurzsichtig ist]{.kicker}

<!--en:6ce68d3e-->
Welche Testfunktion $g^\star \in \mathcal{G}$ erreicht die maximale Differenz in $\textrm{MMD}(P, Q)$?

<!--en:575e5de5-->
- **Zeugenfunktion in geschlossener Form:** Nach der Cauchy-Schwarz-Ungleichung gilt $\mathbf{w}^\star \propto \mathbf{m}_P - \mathbf{m}_Q$. Der optimale Prüfer (die **Zeugenfunktion**) ist somit
  $$g^\star(x) \propto \mathbb{E}_{X \sim P}[k(x, X)] - \mathbb{E}_{Y \sim Q}[k(x, Y)]$$

<!--en:2d769f81-->
- **Vorzeichenwechsel:** Die Funktion ist positiv, wo $P$ überwiegt, und negativ, wo $Q$ überwiegt (Durchgang durch $0$ bei $x = 0.25$ für $\mathcal{N}(0, 1)$ und $\mathcal{N}(0.5, 1)$).

<!--en:8f0d9442-->
- **Verschwindende Gradienten in den Ausläufern:** Ein Generator verschiebt Samples entlang $\nabla g^\star(y)$ bergauf. Mehrere Längenskalen $\ell$ weiter außen flacht $g^\star(x)$ jedoch auf $0$ ab.

<!--en:7edcacb1-->
::: {.slide title="Optimaler Transport und Wasserstein-1-Abstand"}

<!--en:c4c11ecd-->
[Transport von Wahrscheinlichkeitsmasse]{.kicker}

<!--en:95ddc565-->
Damit der Gradient bei *jedem* Abstand $d$ erhalten bleibt, muss der zugrunde liegende Abstand $\|x - y\|$ im Raum eingehen. Ein **Transportplan** (oder eine *Kopplung*) $\gamma \in \Pi(P, Q)$ ist eine gemeinsame Verteilung von $(x, y)$ mit den Randverteilungen $P$ und $Q$:

<!--en:87de636d-->
- **Physikalische Bedeutung (Monge, 1781):** $\gamma(x, y)$ gibt an, wie viel Erde von der Quelle $x$ zur Grube $y$ gelangt. $W_1(P, Q)$ ist die minimale Gesamtarbeit (**Masse $\times$ Weg**), die nötig ist, um $P$ in $Q$ zu überführen.

<!--en:ada6153d-->
- **Exakter Abstand bei jeder Verschiebung $d$:**
  - **Disjunkte Punktmassen ($\delta_0$ und $\delta_d$):** Die einzige Kopplung legt die gesamte Masse auf $(0, d)$. Das ergibt $W_1(\delta_0, \delta_d) = |d|$ mit Gradient $\pm 1$.
  - **Verschobene Gauß-Verteilungen ($\mathcal{N}(0, 1)$ und $\mathcal{N}(d, 1)$):** Die Verschiebung $y = x + d$ kostet $d$. Nach der Jensen-Ungleichung gilt $\mathbb{E}_\gamma|x - y| \ge |\mathbb{E}_P[X] - \mathbb{E}_Q[Y]| = |d|$. Daher ist $W_1 = |d|$ mit konstanter Steigung $1$ für alle $d > 0$.

<!--en:f7f276e8-->
::: {.slide title="Transportpläne und Flächen zwischen Verteilungsfunktionen in einer Dimension"}

<!--en:70c5b66c-->
[Geometrie nach Monge und Kantorovich]{.kicker}

<!--en:80e7dc69-->
Um auf der reellen Geraden eine linkslastige Verteilung $P$ in eine rechtslastige Verteilung $Q$ auf den Atomen $\{0, 1, 2, 3, 4, 5\}$ zu überführen, muss Masse die Intervalle dazwischen durchqueren. Die vertikalen Abstände $|F_P(t) - F_Q(t)|$ über diesen Einheitsintervallen summieren sich zu $W_1 = 1.7$:

<!--en:b1d55e7c-->
::: {.slide title="Kantorovich-Rubinstein-Dualität und 1-Lipschitz-Kritiker"}

<!--en:80bb47a7-->
[Vom primalen linearen Programm zur dualen IPM]{.kicker}

<!--en:dec3ead4-->
Die starke Dualität linearer Programme überführt das primale Infimum der Transportkosten über Kopplungen $\gamma \in \Pi(P, Q)$ in das duale Supremum einer IPM über **1-Lipschitz-Kritiker**:

<!--en:eaabee37-->
- **Schwache Dualität in einer Zeile:** Es gilt punktweise $g(x) - g(y) \le \|x - y\|$ für $\|g\|_{\textrm{Lip}} \le 1$. Mit dem Erwartungswert $\mathbb{E}_\gamma$, folgt $\mathbb{E}_P[g] - \mathbb{E}_Q[g] \le \mathbb{E}_\gamma[\|x - y\|]$.

<!--en:1b677f5b-->
::: {.slide title="Exakter optimaler Transport in einer Dimension als Fläche zwischen Verteilungsfunktionen"}

<!--en:cf399190-->
[Beweis in vier Schritten auf der reellen Geraden]{.kicker}

<!--en:905d3628-->
In einer Dimension ergeben sowohl die duale Darstellung nach Kantorovich und Rubinstein als auch das primale Transportproblem die Fläche zwischen den Verteilungsfunktionen $F_P(t) = \Pr(X \le t)$ und $F_Q(t) = \Pr(Y \le t)$:

<!--en:097ceeb6-->
- **Schritt 1 (untere Schranke für jeden Plan):** Die Darstellung $|X - Y| = \int \mathbf{1}(t \text{ between } X \text{ and } Y)\,dt$ und der Erwartungswert $\mathbb{E}_\gamma$ zeigen, dass über $t$ mindestens die Nettomasse $|\gamma(X \le t < Y) - \gamma(Y \le t < X)| = |F_P(t) - F_Q(t)|$ transportiert werden muss.

<!--en:82c6ada4-->
- **Schritt 2 (monotone Quantilkopplung erreicht die Fläche):** Sei $U \sim \mathrm{Unif}(0, 1)$. Die Kopplung paart die Quantile $X = F_P^{-1}(U)$ und $Y = F_Q^{-1}(U)$. Da bei dieser Kopplung von $X$ und $Y$ über keine Schwelle $t$ Masse in beide Richtungen fließt, betragen die Kosten $\int |F_P(t) - F_Q(t)|\,dt$.

<!--en:472e7d9a-->
- **Schritte 3 und 4 (optimaler dualer 1-Lipschitz-Kritiker):** Nach dem Satz von Fubini gilt $\mathbb{E}_P[g] - \mathbb{E}_Q[g] = \int_{-\infty}^\infty g'(t)(F_Q(t) - F_P(t))\,dt$. Dieser Ausdruck erreicht $\int |F_P(t) - F_Q(t)|\,dt$ für die Steigung $g'(t) = \textrm{sign}(F_Q(t) - F_P(t)) \in \{\pm 1\}$. $\blacksquare$

<!--en:fd129a4b-->
::: {.slide title="Transport in einer Dimension prüfen: Fläche zwischen Verteilungsfunktionen und LP mit 36 Variablen"}

<!--en:e0b5c40b-->
[Rechenbeispiel mit sechs Punkten]{.kicker}

<!--en:63f816d6-->
Im Beispiel mit 6 Punkten ergeben die Summe der fünf Abstände zwischen den Verteilungsfunktionen ($0.25 + 0.35 + 0.50 + 0.35 + 0.25 = 1.7$), die Lösung des linearen Programms mit 36 Variablen und die Kosten der monotonen Quantilkopplung denselben Wert $W_1 = 1.7000$:

<!--en:3d182dad-->
::: {.slide title="Quadratische Kosten, W₂ und Fréchet Inception Distance"}

<!--en:75397cce-->
[Strikte Konvexität und geschlossene Form für Gauß-Verteilungen]{.kicker}

<!--en:3d3fb126-->
Der Ersatz von $|x - y|$ durch den quadrierten Abstand $\|x - y\|^2$ definiert den **Wasserstein-2-Abstand ($W_2$)**:

<!--en:f60054eb-->
- **Kreuzungen in einer Dimension auflösen:** Für $x < x'$ und $y < y'$ spart das Auflösen einer Kreuzung $2(x' - x)(y' - y) > 0$. Daher ist die monotone Quantilkopplung eindeutig optimal.

<!--en:61f99f2c-->
- **Fréchet Inception Distance (FID):** Der Wert von $W_2^2$ für multivariate Gauß-Verteilungen dient zur Bewertung von Bildgeneratoren im Merkmalsraum.

<!--en:d85e368b-->
::: {.slide title="Entropische Regularisierung und Gibbs-Kernel"}

<!--en:6a6e33c0-->
[Schneller, differenzierbarer Transport]{.kicker}

<!--en:e49297bd-->
In Dimensionen $n \ge 2$ entfällt die eindimensionale Formel über die Verteilungsfunktionen. Das Lösen eines linearen Programms mit $N^2$ Variablen kostet $\mathcal{O}(N^3 \log N)$. **Entropischer optimaler Transport** (Cuturi, 2013) ergänzt einen Entropiestrafterm der Stärke $\varepsilon > 0$ und begünstigt so glatte Pläne:

<!--en:e2711366-->
- **Diagonalskalierung aus der Lagrange-Funktion:** Mit Multiplikatoren $\xi_i$ für die Zeilensummen $\sum_j \gamma_{ij} = p_i$ und $\zeta_j$ für die Spaltensummen $\sum_i \gamma_{ij} = q_j$ folgt aus $\frac{\partial \mathcal{L}}{\partial \gamma_{ij}} = C_{ij} + \varepsilon \log \gamma_{ij} - \xi_i - \zeta_j = 0$
  $$\gamma_{ij} = e^{\xi_i/\varepsilon} e^{-C_{ij}/\varepsilon} e^{\zeta_j/\varepsilon} \iff \boldsymbol{\gamma}_\varepsilon = \textrm{diag}(\mathbf{a})\,\mathbf{K}\,\textrm{diag}(\mathbf{b}), \qquad K_{ij} = e^{-C_{ij}/\varepsilon}$$

<!--en:f571ab7c-->
- Die optimale regularisierte Kopplung ergibt sich aus dem eintragsweise berechneten **Gibbs-Kernel** $\mathbf{K} = e^{-\mathbf{C}/\varepsilon}$ durch Skalierung der Zeilen und Spalten mit positiven Vektoren $\mathbf{a}, \mathbf{b} > \mathbf{0}$.

<!--en:6c6b1d37-->
::: {.slide title="Matrixskalierung nach Sinkhorn und die Wahl von ε"}

<!--en:598903a0-->
[Abwechselnde Projektionen auf die Randbedingungen]{.kicker}

<!--en:4e8d1f1f-->
Damit $\boldsymbol{\gamma}_\varepsilon^\top\mathbf{1} = \mathbf{q}$ und $\boldsymbol{\gamma}_\varepsilon\mathbf{1} = \mathbf{p}$ für $\boldsymbol{\gamma}_\varepsilon = \textrm{diag}(\mathbf{a})\mathbf{K}\textrm{diag}(\mathbf{b})$ gelten, wechselt das **Sinkhorn-Knopp-Verfahren** zwischen zwei Skalierungen mit Aufwand $\mathcal{O}(N^2)$, jeweils einer Division nach einem Matrix-Vektor-Produkt:

<!--en:f6de1946-->
- **Differenzierbar auf GPUs:** Die Backpropagation führt direkt durch die entrollten Sinkhorn-Schritte.

<!--en:069fc4fc-->
- **Wahl von $\varepsilon$:** Ein großes $\varepsilon$ führt zu schneller Konvergenz, macht aber $\boldsymbol{\gamma}_\varepsilon$ unscharf. Ein kleines $\varepsilon \to 0$ nähert die exakte LP-Lösung an, doch die Schrittzahl wächst rasch (um mehr als $800\times$ beim Übergang von $\varepsilon = 0.5$ auf $0.1$).

<!--en:a74f6faa-->
::: {.slide title="Konvergenz des Sinkhorn-Verfahrens für verschiedene ε auf dem 6-Punkte-Gitter"}

<!--en:45dde93b-->
[Schärfe und Iterationsaufwand]{.kicker}

<!--en:944cae14-->
Bei quadratischen Kosten $C_{ij} = (x_i - x_j)^2$ ($W_2^2 = 3.50000$ auf $10$ monotonen Routen) sinkt $\varepsilon$ von $2.0$ auf $0.1$. In der Heatmap tritt dadurch die exakte Treppenform hervor, während die Iterationszahl von $50$ auf $\sim 150{,}000$ steigt:

<!--en:71d18e80-->
::: {.slide title="Feste Kernel, Kritiker und der Fluch der Dimensionalität"}

<!--en:495176f8-->
[Statistische Genauigkeit und Rechenaufwand]{.kicker}

<!--en:aa94fd64-->
- **MMD mit festem Kernel:** U-Statistik in geschlossener Form mit Aufwand $\mathcal{O}(N^2)$ und **dimensionsunabhängiger Stichprobenfehlerrate $\mathcal{O}(N^{-1/2})$** in jeder Dimension $n$. Für $d \gg \ell$ verschwinden jedoch die Gradienten in den Ausläufern.

<!--en:c345af60-->
- **Wasserstein-1 ($W_1$):** Steigung eins bei jedem Abstand $d$. Der empirische Abstand $W_1$ zwischen Stichproben unterliegt jedoch dem **Fluch der Dimensionalität mit der Rate $\mathcal{O}(N^{-1/n})$** für $n > 2$. Deshalb beschränkt das WGAN die Suche auf neuronale Kritiker.

<!--en:4d28ff36-->
::: {.slide title="Verständnisfragen zu empirischem TV, W₁ auf der Geraden und Kritikern"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:143d94c2-->
Fragen zu IPMs und optimalem Transport:

<!--en:3c981388-->
1. **Problem des empirischen Totalvariationsabstands:** Zwei unabhängige Stichproben mit je $N = 1{,}000$ Punkten aus $\mathcal{N}(0, 1)$ ziehen. Wie groß ist der exakte Totalvariationsabstand zwischen den empirischen Verteilungen $\hat{P}_N$ und $\hat{Q}_N$, und welchen Wert liefert der erwartungstreue Schätzer für die RBF-$\textrm{MMD}^2$?

<!--en:e65f8acc-->
2. **Wasserstein-Abstand und TV bei Skalierung auf der Geraden:** Die Verteilungen $P = (0.6, 0.3, 0.1)$ und $Q = (0.2, 0.5, 0.3)$ liegen auf den Atomen $\{0, 1, 2\}$. Gesucht sind $\textrm{TV}(P, Q)$ und $W_1(P, Q)$. Wie ändern sich beide Größen, wenn die Atome auf $\{0, 2, 4\}$ auseinanderrücken?

<!--en:e0d01c7c-->
3. **Optimaler Kritiker bei unterschiedlicher Varianz:** Für $P = \mathcal{N}(0, 1)$ und $Q = \mathcal{N}(0, 4)$ verwendet die monotone Kopplung auf der Geraden die Abbildung $X \mapsto Y = 2X$. Mit dieser Kopplung $W_1(P, Q)$ berechnen und den optimalen 1-Lipschitz-Kritiker $g^\star(x)$ bestimmen.

<!--en:7d39c680-->
::: {.slide title="Verständnisfragen zu empirischem TV, W₁ auf der Geraden und Kritikern (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständniskontrolle]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:a4ccc99b-->
1. **Empirischer TV:** Da Stichproben aus einer stetigen Verteilung fast sicher keinen Punkt gemeinsam haben, liefert die Indikatorfunktion der ersten Stichprobe $\textrm{TV}(\hat{P}_N, \hat{Q}_N) = 1.0$. Bei der erwartungstreuen RBF-$\textrm{MMD}^2$ hebt sich das Rauschen erster Ordnung auf. Das ergibt $\approx 0.000 \pm 0.002$.

<!--en:3b021fee-->
2. **$W_1$ auf 3 Atomen der Geraden:** Es gilt $\textrm{TV} = 0.4$. Die Verteilungsfunktionen $F_P = (0.6, 0.9, 1.0)$ und $F_Q = (0.2, 0.7, 1.0)$ haben über den Einheitsintervallen die Abstände $|F_P - F_Q| = (0.4, 0.2)$. Daraus folgt $W_1 = 0.4(1) + 0.2(1) = 0.6$. Rücken die Atome auf $\{0, 2, 4\}$ auseinander, verdoppeln sich die Intervallbreiten ($\Delta x = 2$). Damit verdoppelt sich der Wasserstein-Abstand ($W_1 \to 1.2$), während $\textrm{TV} = 0.4$ unverändert bleibt.

<!--en:5f61d7a1-->
3. **Gauß-Verteilungen mit unterschiedlicher Varianz:** Für $Y = 2X$ gilt $|X - Y| = |X|$, also $W_1 = \mathbb{E}_{X \sim \mathcal{N}(0,1)}[|X|] = \sqrt{2/\pi} \approx 0.798$. Da $F_P(t) < F_Q(t)$ für $t < 0$ und $F_P(t) > F_Q(t)$ für $t > 0$ gilt, hat der optimale Kritiker die Steigung $g'(t) = -\textrm{sign}(t)$. Das ergibt $g^\star(x) = -|x|$ und zieht die breit gestreuten Samples nach innen.

<!--en:156441ab-->
::: {.slide title="Zusammenfassung: integrale Wahrscheinlichkeitsmetriken und Transport"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:94793579-->
| Verfahren oder Abstand | Mathematische Definition | Stichprobenkomplexität und Berechnung | Geometrische Reichweite und Rolle im Deep Learning |

<!--en:aa94c719-->
| **Allgemeine IPM** | $\sup_{g \in \mathcal{G}}(\mathbb{E}_P[g] - \mathbb{E}_Q[g])$ | Abhängig von der Vielfalt der Prüferklasse $\mathcal{G}$ | Vereint TV ($|g|\le 1/2$), MMD (Einheitskugel im RKHS) und $W_1$ ($\textrm{Lip}_1$) |

<!--en:8360916c-->
| **Kernel-MMD** | $\|\mathbf{m}_P - \mathbf{m}_Q\|_2 = (\mathbb{E}[k_{PP} + k_{QQ} - 2k_{PQ}])^{1/2}$ | $\mathcal{O}(N^{-1/2})$ in jeder Dimension $n$, U-Statistik in geschlossener Form | Charakteristisch bei RBF-Kernel, ideal für Zweistichprobentests, flach für $d \gg \ell$ |

<!--en:2c5aefee-->
| **Wasserstein-1 ($W_1$)** | $\inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_\gamma\|x-y\| = \sup_{\|g\|_{\textrm{Lip}}\le 1}(\mathbb{E}_P g - \mathbb{E}_Q g)$ | In einer Dimension $\int |F_P - F_Q|\,dt$, in $n$ Dimensionen empirische Rate $\mathcal{O}(N^{-1/n})$ | Konstante Steigung eins bei jedem Abstand $d$, Grundlage des WGAN |

<!--en:1defb0bb-->
| **Wasserstein-2 ($W_2$)** | $(\inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_\gamma\|x-y\|^2)^{1/2}$ | Eindeutiger monotoner Optimalplan auf der Geraden, Fréchet-Abstand für Gauß-Verteilungen in geschlossener Form | Bestraft sich kreuzende Routen, Grundlage der FID-Bewertung und des optimalen Transports beim Flow-Matching |

<!--en:f5f6e816-->
| **Entropischer Transport nach Sinkhorn** | $\min_{\gamma \in \Pi(\mathbf{p},\mathbf{q})} \langle \boldsymbol{\gamma}, \mathbf{C} \rangle + \varepsilon \langle \boldsymbol{\gamma}, \log \boldsymbol{\gamma} - \mathbf{1} \rangle$ | $\boldsymbol{\gamma}_\varepsilon = \textrm{diag}(\mathbf{a})e^{-\mathbf{C}/\varepsilon}\textrm{diag}(\mathbf{b})$, Aufwand $\mathcal{O}(N^2)$ je Schritt | Schneller, auf GPUs differenzierbarer Transport, Abwägung zwischen Unschärfe ($\varepsilon$) und Iterationszahl |
