<!--en:b9c1626a-->
# Schätzung der gegenseitigen Information

<!--en:0ace983f-->
Bisher haben wir die gegenseitige Information (engl. mutual information) stets aus einer gemeinsamen Dichte in geschlossener Form berechnet. In der Praxis liefern die Daten keine Dichte mit. Wir haben nur eine Sammlung von Paaren, etwa gesprochene Sätze mit ihren Transkripten oder Suchanfragen mit den Dokumenten, die sie beantworten. Allein aus diesen Samples wollen wir bestimmen, wie viel ein Teil eines Paares über den anderen verrät. Das Repräsentationslernen (engl. representation learning) geht einen Schritt weiter. Es trainiert Modelle so, dass diese gemeinsame Information groß wird. Damit wird der Schätzwert selbst zum Trainingsziel.

<!--en:a1a90e6b-->
Ein bekannter kontrastiver Verlust (engl. contrastive loss) verdeutlicht die Schwierigkeit. Systeme wie SimCLR und CLIP trainieren mit Paaren von *Ansichten* (engl. views) desselben Beispiels, etwa zwei zufälligen Bildausschnitten eines Fotos oder einem Bild und seiner Beschreibung. Zu einer Ansicht lernt das Modell, den richtigen Partner unter beispielsweise $128$ Kandidaten eines Batches zu erkennen. Dazu minimiert es einen Kreuzentropieverlust, den InfoNCE-Verlust. In der Fachliteratur wird dieser Verlust häufig von $\ln 128 = 4.85$ abgezogen und die Differenz als gegenseitige Information der beiden Ansichten angegeben. Doch die Differenz kann höchstens $4.85$ Nat betragen, unabhängig davon, wie viel Information die Ansichten tatsächlich teilen. Ein Modell, das jedes Mal den richtigen Partner wählt, hat lediglich gezeigt, dass es $128$ Kandidaten unterscheiden kann. Da wir solche Größen sowohl für das Training als auch als Messwerte verwenden, müssen wir zwischen einer Messung und einer unteren Schranke unterscheiden. Ebenso müssen wir den Abstand der Schranke zum wahren Wert vom Fehler des gelernten Modells trennen.

<!--en:6779c040-->
Um die gegenseitige Information aus Samples zu schätzen, gibt es zwei wesentliche Ansätze. Der direkte Ansatz zählt Samples in Histogrammklassen oder in Umgebungen nächster Nachbarn. Für niedrigdimensionale Variablen mit wenig gemeinsamer Information funktioniert das gut, bei Vektoren stößt das Verfahren an seine Grenzen. Der zweite Ansatz ersetzt die unbekannten Dichten durch ein gelerntes Modell. Es ist so konstruiert, dass der resultierende Erwartungswert eine untere Schranke für $I(X; Y)$ ist. Diese Schranke wird maximiert. Zwei Ergebnisse stehen im Mittelpunkt des weiteren Abschnitts. Das erste begrenzt grundsätzlich, was ein Schätzer ohne Verteilungsannahmen garantieren kann (:numref:`subsec_mdl-mi-distribution-free`). Das zweite beschreibt die variationelle Schranke hinter dem InfoNCE-Verlust und ihre logarithmische Obergrenze (:numref:`sec_mdl-infonce`).

<!--en:c6e8ed9c-->
## Ein Paar mit bekannter gegenseitiger Information

<!--en:4c163f0c-->
Um einen Schätzer zu beurteilen, brauchen wir einen Vergleichsfall, dessen gegenseitige Information wir genau kennen. In diesem Abschnitt verwenden wir das standardisierte bivariate Gauß-Paar aus :numref:`subsec_mdl-mi-gaussian` mit Korrelation $\rho$. Seine gegenseitige Information beträgt

<!--en:e6dcafb0-->
Nat, also $0.83$ Nat bei $\rho = 0.9$ und $1.96$ Nat bei $\rho = 0.99$. Alle Logarithmen in diesem Abschnitt sind natürlich.

<!--en:77ddf4be-->
Die variationellen Schranken, die wir herleiten, beruhen auf der *punktweisen gegenseitigen Information* (engl. pointwise mutual information, PMI) aus :numref:`subsec_mdl-pmi`, $\textrm{pmi}(x, y) = \log \frac{p(x, y)}{p(x)\,p(y)}$. Sie misst, um welchen Faktor ein bestimmtes Paar $(x, y)$ unter der gemeinsamen Verteilung wahrscheinlicher ist als bei Unabhängigkeit, und zwar auf logarithmischer Skala. Für unser standardisiertes Gauß-Paar ist die PMI eine einfache quadratische Funktion:

<!--en:8d2cdde1-->
Aus dieser Definition folgen unmittelbar zwei Identitäten. Erstens ergibt der Mittelwert von $\textrm{pmi}(x, y)$ über Paare aus der gemeinsamen Verteilung $I(X; Y)$. Zweitens hat die Exponentialfunktion $e^{\textrm{pmi}(x, y)}$ bei *unabhängigen* Ziehungen von $x$ und $y$ den Mittelwert $1$, denn das Integral von $p(x)\,p(y)\,e^{\textrm{pmi}(x, y)} = p(x, y)$ ist $1$. Prüfen wir beide Identitäten anhand von Samples. Für die zweite heben wir die Abhängigkeit auf, indem wir jedes $x$ mit dem $y$ des nächsten Samples im Array paaren. Wir nennen sie *neu zusammengestellte Paare* (engl. mismatched pairs). Jedes entsteht durch eine unabhängige Ziehung der Komponenten gemäß $P_X \otimes P_Y$. Diesen Trick verwenden wir immer dann, wenn eine Schranke unabhängige Paare erfordert.

<!--en:c80783c2-->
Beide Stichprobenmittelwerte stimmen bis auf etwa $0.01$ mit ihren theoretischen Werten überein. Jede variationelle Schranke in diesem Abschnitt beruht auf diesen beiden Identitäten. Fast alle statistischen Schwierigkeiten gehen auf die zweite zurück. Bei $\rho = 0.9$ verhält sich $e^{\textrm{pmi}}$ gut. Die Varianz über unabhängige Paare beträgt $e^{2I} - 1$ (Übung :exref:`ex_mdl-mi-estimation-heavy-tails`), hier also nur $4.3$. Bei $I = 6$ steigt diese Varianz jedoch auf $160{,}000$. Selbst der Mittelwert über dieselben $200{,}000$ Samples hat damit eine Unsicherheit von rund $0.9$.

<!--en:aa1c96a6-->
## Nichtparametrische Schätzung

<!--en:111673c8-->
Am direktesten schätzen wir eine Wahrscheinlichkeitsverteilung aus Daten, indem wir zählen, wie viele Samples in jedes Gebiet des Raums fallen. Entscheidend ist, wie wir diese Gebiete festlegen. Wir beginnen mit der einfachsten Wahl, einem festen Gitter.

<!--en:0ffd3d7a-->
### Histogramme

<!--en:77f76b15-->
Der Histogrammschätzer unterteilt den Wertebereich jeder Variablen in $b$ gleich breite Intervalle. Er zählt die Samples in jeder der $b \times b$ Zellen und setzt die empirischen Häufigkeiten in die diskrete Formel für die gegenseitige Information ein. Bei diesem einfachen Schätzer treten zwei entgegengesetzte Verzerrungen auf.

<!--en:ef3cbfa1-->
Die erste Verzerrung zieht den Schätzwert *nach unten*. Ersetzen wir einen stetigen Wert durch seinen Klassenindex, wenden wir eine deterministische Funktion auf $X$ an. Nach der Datenverarbeitungsungleichung (:numref:`subsec_mdl-dpi`) teilt jede Funktion von $X$ höchstens so viel Information mit $Y$ wie $X$ selbst. Die Klasseneinteilung kann also nur Information vernichten. Der Verlust ist besonders groß, wenn die Abhängigkeit auf einer feineren Skala liegt als die Klassenbreite. Bei $\rho = 0.99$ etwa beträgt die bedingte Standardabweichung von $y$ bei gegebenem $x$ nur $\sqrt{1 - \rho^2} = 0.14$. Teilen wir eine Stichprobe mit einem Wertebereich von ungefähr $[-4.5, 4.5]$ in $10$ Klassen, erhalten wir Zellen der Breite $0.9$. Das ist mehr als das Sechsfache der bedingten Streuung.

<!--en:f061c66c-->
Die zweite Verzerrung zieht den Schätzwert *nach oben*. Der Plug-in-Schätzer berechnet die KL-Divergenz zwischen der empirischen gemeinsamen Häufigkeitsverteilung und dem Produkt ihrer Randverteilungen über Zeilen und Spalten. Er ist daher stets nichtnegativ und nur dann null, wenn die Tabelle exakt faktorisiert. Selbst bei vollständiger Unabhängigkeit von $X$ und $Y$ verhindern zufällige Stichprobenschwankungen eine exakte Faktorisierung. So täuscht Rauschen eine Abhängigkeit vor. Je weniger Samples jede Zelle enthält, desto stärker wird diese Verzerrung nach oben.

<!--en:4f6eff78-->
Zu diesen beiden Fehlern kommt eine feste Obergrenze hinzu. Der Schätzwert kann die Entropie der in Klassen eingeteilten Variablen $X$ nicht überschreiten. Dabei können $b$ Klassen höchstens $\log b$ Nat tragen, bei zehn Klassen also nur $2.30$ Nat, unabhängig davon, wie eng $X$ und $Y$ zusammenhängen. Eine Variante dieser Obergrenze begegnet uns bei jedem Schätzer in diesem Abschnitt. Kein Verfahren kann mehr Information angeben als den Logarithmus der Anzahl von Zuständen, die es unterscheiden kann.

<!--en:d168df66-->
Implementieren wir diesen Schätzer als `binned_mi` und testen wir ihn an $200{,}000$ Samples aus vier Verteilungen mit $b = 10$, $40$ und $160$ Klassen. Wir betrachten ein unabhängiges Paar, das Gauß-Paar bei $\rho = 0.9$ und $\rho = 0.99$ sowie das parabelförmige Paar $Y = X^2 + 0.3\,\epsilon$ aus :numref:`sec_mdl-mi-nonlinear-correlation`. Dessen lineare Korrelation ist null, seine gegenseitige Information beträgt dennoch $1.141$ Nat.

<!--en:f1c6851c-->
Die Tabelle zeigt beide Verzerrungen deutlich. Bei $b = 10$ fällt der Schätzwert für jedes abhängige Paar zu niedrig aus. Er beträgt rund $0.6$ statt $0.83$ Nat bei $\rho = 0.9$ und nur etwa die Hälfte des wahren Werts bei $\rho = 0.99$ sowie für die Parabel. Bei $b = 160$ schreibt der Schätzer dem unabhängigen Paar einige Hundertstel Nat zu. Für das Paar bei $\rho = 0.9$ überschätzt er sogar den wahren Wert. Eine grobe Klasseneinteilung allein könnte das nie verursachen. Bei $\rho = 0.99$ und $b = 160$ liegt der Schätzwert bis auf etwa $0.01$ beim wahren Wert, aber nur, weil sich entgegengesetzte Fehler zufällig aufheben. In der Zeile darüber liefert dasselbe Gitter einen zu hohen Wert. Kennen wir den wahren Wert nicht, verrät uns die Tabelle nicht, welcher Spalte wir vertrauen können.

<!--en:5b2eec4f-->
Eine dritte Schwäche ist die Abhängigkeit vom Koordinatensystem. Eine invertierbare Transformation einer der beiden Variablen lässt die gegenseitige Information unverändert (:numref:`sec_mdl-mi-nonlinear-correlation`), verzerrt aber ein Gitter mit festen Klassenbreiten. Ersetzen wir etwa $x$ durch $x^3$ im Gauß-Paar bei $\rho = 0.9$. Die Extremwerte unter $200{,}000$ standardnormalverteilten Ziehungen liegen bei ungefähr $\pm 4.5$, sodass $x^3$ etwa bis $\pm 90$ reicht. Jede von $40$ gleich breiten Klassen hat dann eine Breite von etwa $4.5$. Dadurch fallen rund neun von zehn Samples in die zwei oder drei Klassen um null. Abhilfe schafft eine Klasseneinteilung nach *Rängen*, also nach der Position jedes Werts in der sortierten Stichprobe. Jede streng monoton wachsende Abbildung erhält die Ränge, und jede Rangklasse enthält automatisch gleich viele Samples.

<!--en:e15191d0-->
Obwohl die wahre gegenseitige Information bei $0.830$ Nat bleibt, sinkt durch das Kubieren von $x$ die lineare Korrelation von $0.90$ auf etwa $0.7$. Der Histogrammschätzwert auf den Rohdaten fällt auf einen Wert zwischen $0.3$ und $0.5$ Nat (der genaue Wert hängt von den wenigen extremen Ziehungen ab, die die Klassengrenzen bestimmen). Die Einteilung nach Rängen hingegen ergibt für $x$ und $x^3$ bis auf die letzte Stelle denselben Schätzwert. Er weicht um höchstens $0.02$ Nat vom wahren Wert ab.

<!--en:34cd3a71-->
Ränge lösen das Koordinatenproblem. In hohen Dimensionen stoßen Histogramme dennoch an ihre Grenzen. Ein Paar fünfdimensionaler Vektoren erfordert ein zehndimensionales Gitter. Schon zehn Klassen pro Achse ergeben $10^{10}$ Zellen, von denen bei jeder realistischen Stichprobengröße fast alle leer bleiben. Für Vektoren muss der Schätzer die Auflösung an die Daten anpassen. Wo die Samples dicht liegen, braucht er kleine Gebiete, wo sie weit auseinanderliegen, größere.

<!--en:c4a1534a-->
### Der KSG-Schätzer

<!--en:8408982f-->
Der als KSG bekannte Schätzer von :citet:`Kraskov.Stogbauer.Grassberger.2004` ersetzt feste Histogrammzellen durch adaptive Umgebungen der nächsten Nachbarn jedes Datenpunkts. Er ist das nichtparametrische Standardverfahren für Paare niedrigdimensionaler Variablen. (Diesen und den nächsten Unterabschnitt können Sie bei der ersten Lektüre überspringen.)

<!--en:1c50fbb2-->
KSG misst Abstände anhand der größten absoluten Koordinatendifferenz (Chebyshev- oder $\ell_\infty$-Abstand). Eine Kugel mit Radius $r$ ist daher in zwei Dimensionen ein Quadrat und in höheren Dimensionen ein Hyperwürfel. Um jeden Stichprobenpunkt $i$ legen wir im gemeinsamen $(x, y)$-Raum den kleinsten Kasten, der seinen $m$-ten nächsten Nachbarn erreicht. Seine halbe Breite bezeichnen wir mit $r_i$. Nach Konstruktion enthält dieser Kasten einen Anteil $m/n$ der $n$ Samples. Damit erhalten wir eine lokale Schätzung der gemeinsamen Wahrscheinlichkeitsmasse in der Nähe von $(x_i, y_i)$.

<!--en:2b1e5db2-->
Als Nächstes messen wir die beiden Randwahrscheinlichkeiten auf genau derselben Skala $r_i$. Sei $n_x^{(i)}$ die Anzahl anderer Samples strikt innerhalb des vertikalen Streifens $|x - x_i| < r_i$ und $n_y^{(i)}$ die entsprechende Anzahl strikt innerhalb des horizontalen Streifens $|y - y_i| < r_i$. Dank der gemeinsamen halben Breite $r_i$ können wir Dichten durch Anzahlen ersetzen. Sind die Dichten über den Kasten annähernd konstant, enthält er eine Wahrscheinlichkeitsmasse von etwa $p(x_i, y_i)\,(2r_i)^2$. Die beiden Streifen enthalten etwa $p(x_i)\,2r_i$ und $p(y_i)\,2r_i$. Teilen wir den Anteil im Kasten durch das Produkt der beiden Streifenanteile, kürzen sich die geometrischen Faktoren $2r_i$. Der Logarithmus ergibt eine lokale Schätzung der punktweisen gegenseitigen Information allein durch Zählen. :numref:`fig_mdl-ksg-box` veranschaulicht die drei Zählungen an einem einzigen Punkt.

<!--en:f3bf2455-->
![KSG-Zählung an einem Stichprobenpunkt $i$ (Stern) mit $m = 3$. Der Kasten mit halber Breite $r_i$ reicht bis zum drittnächsten Nachbarn in der gemeinsamen Ebene (Ring am linken Rand, der im horizontalen Streifen liegt und nur zu $n_y^{(i)}$ zählt). Zwei Nachbarn liegen strikt im Inneren. Der vertikale Streifen mit derselben halben Breite enthält $n_x^{(i)}$ weitere Samples, der horizontale $n_y^{(i)}$ (blaue und grüne Punkte liegen jeweils nur in einem Streifen). Jeder Summand des Schätzers beträgt ungefähr $\log\frac{m/n}{(n_x/n)(n_y/n)}$, eine aus Anzahlen berechnete punktweise gegenseitige Information.](../img/mdl-it-ksg-box.svg)

<!--en:2c7c3f99-->
Das Mittel dieser lokalen logarithmischen Verhältnisse über alle $n$ Punkte schätzt $I(X; Y)$, mit einer Korrektur für endliche Stichproben. Der Logarithmus einer kleinen Anzahl liefert eine verzerrte Schätzung einer Log-Wahrscheinlichkeit. Für den Kasten, der bis zum $m$-ten von $n - 1$ Nachbarn reicht, beträgt die erwartete logarithmische Wahrscheinlichkeitsmasse $\mathbb{E}[\log p] = \psi(m) - \psi(n)$ anstelle von $\log(m/n)$. Dabei ist $\psi = (\log \Gamma)'$ die Digammafunktion (wir lassen die Herleitung aus). Ersetzen wir die Logarithmen durch $\psi$, erhalten wir den KSG-Schätzer:

<!--en:9cf5b418-->
Wegen $\psi(j) \approx \log j$ für große $j$ beträgt jeder Summand ungefähr $\log\frac{m\,n}{n_x^{(i)}\, n_y^{(i)}}$. Dies entspricht dem lokalen logarithmischen Verhältnis zwischen dem Anteil im Kasten und dem Produkt der Streifenanteile. Das $+1$ in den Streifentermen stimmt diese auf den Kasten ab. Der Term $\psi(m)$ entspricht einem Kasten, dessen Rand durch den $m$-ten Nachbarn verläuft. Einen Streifen mit $n_x^{(i)}$ Punkten strikt im Inneren behandeln wir daher so, als verliefe sein Rand durch den $(n_x^{(i)} + 1)$-ten Punkt. In beiden Streifen zählen wir mit strikten Ungleichungen und lassen den Punkt $i$ selbst aus. Ein Nachbar genau auf dem Streifenrand zählt also nicht mit. Da sich jeder Radius $r_i$ automatisch an die lokale Dichte der Samples anpasst, müssen wir weder eine Klassenbreite noch eine Kernbandbreite einstellen.

<!--en:01189a36-->
Aus den $n \times n$-Matrizen der Chebyshev-Abstände in $x$ und $y$ können wir mit `cdist` alle Nachbarzahlen auf einmal bestimmen. Ihr elementweises Maximum ergibt den gemeinsamen Abstand.

<!--en:8cec3862-->
Da jeder Summand in :eqref:`eq_mdl-ksg` ausschließlich von Nachbarzahlen abhängt, begrenzt bereits das Zählen den Wert, den KSG liefern kann.

<!--en:33689d70-->
**Proposition (Obergrenze von KSG durch die Stichprobengröße).** *Für jede Stichprobe aus $n$ Punkten, in der von keinem Punkt aus zwei gemeinsame Abstände gleich sind, gilt*

<!--en:fbaff4ae-->
*wobei für $Y = X$ Gleichheit gilt.*

<!--en:3d497165-->
**Beweis.** Sind alle Abstände von einem Punkt aus verschieden, liegen genau $m - 1$ Samples strikt näher an Punkt $i$ als sein $m$-ter Nachbar. Der gemeinsame Chebyshev-Abstand ist der größere der beiden Koordinatenabstände. Jeder dieser $m - 1$ Punkte erfüllt also sowohl $|x_j - x_i| < r_i$ als auch $|y_j - y_i| < r_i$ und liegt damit in beiden Streifen. Folglich gelten $n_x^{(i)} \geq m - 1$ und $n_y^{(i)} \geq m - 1$. Da $\psi$ streng monoton wächst, beträgt jeder Summand höchstens $\psi(m) + \psi(n) - 2\psi(m) = \psi(n) - \psi(m)$. Für $Y = X$ stimmen die gemeinsamen und die marginalen Abstände überein. Beide Streifen enthalten dann nur die $m - 1$ Punkte strikt im Kasteninneren, und jeder Summand erreicht die Schranke. $\blacksquare$

<!--en:89f60418-->
Für $n = 2{,}000$ und $m = 5$ beträgt diese Obergrenze $\psi(2000) - \psi(5) = 6.09$ Nat und liegt damit nahe bei $\log(n/m) = 5.99$. Selbst für $Y = X$ mit stetigem $X$, also bei unendlicher gegenseitiger Information, liefert KSG nur $6.1$ Nat. Eine Verdopplung der Stichprobengröße hebt die Obergrenze um lediglich etwa $\log 2$ an. Hier begegnet uns die Obergrenze $\log b$ des Histogramms wieder, diesmal mit adaptiven Nachbarschaften anstelle von Klassen. Ein Schätzer, der Nachbarn zählt, liefert höchstens den Logarithmus der Anzahl unterscheidbarer Nachbarschaften in der Stichprobe. Diese Anzahl beträgt ungefähr $n/m$.

<!--en:d60936b0-->
### KSG in hohen Dimensionen

<!--en:93054143-->
In höheren Dimensionen versagt KSG aus geometrischen Gründen schon weit unterhalb der durch die Stichprobengröße gegebenen Obergrenze. Betrachten wir $d$ unabhängige Kopien unseres Gauß-Paares, jeweils mit $\rho = 0.9$. Dann sind $X$ und $Y$ zwei $d$-dimensionale Vektoren. Der Logarithmus des Dichteverhältnisses zerfällt in eine Summe von $d$ unabhängigen Termen. Die gegenseitige Information addiert sich also über unabhängige Koordinaten, und ihr wahrer Wert beträgt $d \times 0.83$ Nat.

<!--en:cbeebe65-->
KSG setzt voraus, dass die gemeinsame Dichte und die Randdichten in jedem Nachbarschaftskasten annähernd konstant sind. Mit wachsender Dimension bricht diese Annahme zusammen. Für $n = 2{,}000$ und $m = 5$ enthält jeder Kasten einen Anteil $m/n = 0.0025$ der Stichprobe. Bei gleichmäßig über $2d$ Dimensionen verteilten Punkten würde ein solcher Kasten entlang jeder Achse einen Anteil $(m/n)^{1/(2d)}$ der Datenbreite abdecken. Dieser Anteil steigt von $0.05$ bei $d = 1$ auf $0.47$ bei $d = 4$ und $0.83$ bei $d = 16$. Die Abhängigkeit liegt jedoch in einem schmalen Band um $y = \rho x$. Bei gegebenem $x$ hat jede Koordinate von $y$ die Standardabweichung $\sqrt{1 - \rho^2} = 0.44$. Das entspricht rund $0.07$ der gesamten Spannweite der Stichprobe (etwa $-3.3$ bis $3.3$). Ein um ein Vielfaches breiterer Kasten verwischt die Abhängigkeit ebenso wie eine zu grobe Histogrammklasse. Wenden wir KSG auf $2{,}000$ Samples für $d = 1, 2, 4, 8, 16$ an und erfassen wir den Schätzwert und die mittlere Punktzahl je Streifen. Zum Vergleich betrachten wir den Fall $Y = X$, der die Obergrenze erreicht.

<!--en:ab61d4da-->
In einer und zwei Dimensionen weicht KSG nur um etwa $0.1$ Nat vom wahren Wert ab. Schon bei $d = 4$ fällt die Schätzung auf rund $2.5$ Nat gegenüber einem wahren Wert von $3.3$, also fast ein ganzes Nat zu niedrig. Danach stagniert sie. Sowohl bei $d = 8$ als auch bei $d = 16$ liefert KSG knapp $3$ Nat, obwohl sich die wahre gegenseitige Information von $6.6$ auf $13.3$ verdoppelt. Die Stichprobengröße setzt hier noch nicht die entscheidende Obergrenze. Denn $3$ Nat sind weniger als die Hälfte der Obergrenze von $6.1$ Nat für $Y = X$.

<!--en:f70f0c6d-->
Die mittleren Anzahlen in den Streifen erklären die Stagnation. Jeder Summand beträgt ungefähr $\log\frac{m\,n}{n_x n_y}$. Der Schätzwert kann also nur steigen, wenn weniger Punkte in die Randstreifen fallen. Ihre Anzahl sinkt von etwa $70$ Punkten bei $d = 1$ auf rund $31$ bei $d = 4$ und pendelt sich dann bei ungefähr $26$ ein. Damit die Kästen bei $d = 16$ so lokal bleiben wie bei $d = 1$, bräuchten wir $m/n = 0.05^{32}$, also eine Stichprobe mit rund $10^{42}$ Punkten. Die Stichprobengröße, die für lokale Nachbarschaften nötig ist, wächst exponentiell mit der Dimension.

<!--en:6409dd96-->
### Grenzen der verteilungsfreien Schätzung

<!--en:847d5a74-->
Vielleicht kann ein geschickterer Schätzer diesen Engpass umgehen. Prüfen wir diese Hoffnung anhand einer schwächeren Forderung als einer exakten Messung, nämlich einer *unteren Schranke*. Gesucht ist eine Funktion $B$ von $n$ Samples, sodass $B \leq I(X; Y)$ mit Wahrscheinlichkeit mindestens $0.95$ *für jede Verteilung* des Paares $(X, Y)$ gilt. (Jedes Konfidenzniveau über $1/2$ wäre ebenso geeignet.) Auch beim Repräsentationslernen suchen wir untere Schranken, denn wir möchten sicher sein, dass zwei Ansichten viel Information teilen.

<!--en:abefb29f-->
Schon der einfachste Fall zeigt die grundsätzliche Grenze. Setzen wir $Y = X$ für eine diskrete Variable $X$. Dann gilt $I(X; Y) = H(X)$. Behandelt ein Schätzer die diskreten Symbole als beliebige Bezeichnungen ohne vorgegebene Metrik oder Ordnung, kann er nicht ausnutzen, *welche* Symbole aufgetreten sind. Er kann nur das Muster der Wiederholungen betrachten. Wie viele verschiedene Werte kamen einmal vor, wie viele zweimal und so weiter? Diese Zusammenfassung heißt *Typ* der Stichprobe. Wir nehmen an, dass $B$ allein davon abhängt. Vier kurze Schritte ergeben die Grenze, wobei der dritte Schritt entscheidend ist:

<!--en:c712b58c-->
1. Sei $X$ gleichverteilt auf $M$ Symbolen. Ziehen wir $n$ Samples. Zwei gegebene Ziehungen stimmen mit Wahrscheinlichkeit $1/M$ überein. Nach der booleschen Ungleichung beträgt die Wahrscheinlichkeit einer Kollision bei *irgendeinem* der $n(n-1)/2$ Paare daher höchstens $n(n-1)/(2M)$.

<!--en:874ebf9a-->
1. Wenden wir die Garantie mit Konfidenzniveau $0.95$ auf die Gleichverteilung auf $M = n^2$ Symbolen an. Ihre Entropie ist $2 \log n$. Für eine Stichprobe aus *dieser* Verteilung gilt $B \leq 2 \log n$ mit Wahrscheinlichkeit mindestens $0.95$.

<!--en:f6b84371-->
1. Nach dem ersten Schritt besteht eine Stichprobe der Größe $n$ aus $M = n^2$ Symbolen aus $n$ verschiedenen Werten, und zwar mit Wahrscheinlichkeit größer als $\tfrac{1}{2}$. Zwei Ereignisse, deren Wahrscheinlichkeiten zusammen größer als $1$ sind, müssen sich überschneiden. Es gibt daher mindestens eine Stichprobe ohne Wiederholungen, für die $B \leq 2 \log n$ gilt.

<!--en:8dce8dad-->
1. Jede Stichprobe aus $n$ verschiedenen Werten hat genau denselben Typ. Daher muss $B$ denselben Wert (höchstens $2 \log n$) für *jede* Stichprobe ohne Wiederholungen liefern, unabhängig von ihrer tatsächlichen Verteilung.

<!--en:a4b70352-->
Prüfen wir diese Schranke nun an einer Gleichverteilung auf $M = 2^{256}$ Symbolen mit einer Entropie von $256 \ln 2 = 177$ Nat. Bei $n = 10^6$ Samples sind mit Wahrscheinlichkeit mindestens $1 - 4.3 \times 10^{-66}$ alle Werte verschieden. Für eine solche Stichprobe kann $B$ höchstens $2 \log 10^6 = 27.6$ Nat garantieren. Eine Million Samples garantiert keine $177$ Nat, nicht einmal $30$. Übung :exref:`ex_mdl-mutual-information-entropy-bounds-samples` verbessert die Konstante und fragt, wie viele Samples nötig wären.

<!--en:a2269804-->
:citet:`McAllester.Stratos.2020` bewiesen den allgemeinen Satz, ohne den Schätzer auf Funktionen des Stichprobentyps zu beschränken:

<!--en:ef794072-->
> **Satz (McAllester–Stratos, informell).** Jede verteilungsfreie untere Schranke für $I(X; Y)$, die bei festem Konfidenzniveau für $n$ Samples gilt, ist $O(\log n)$.

<!--en:cf8b4c36-->
Hinter dieser Grenze steht die Invarianz der gegenseitigen Information. Jede umkehrbare Umbenennung von $X$ oder $Y$ lässt $I(X; Y)$ unverändert (:numref:`sec_mdl-mi-nonlinear-correlation`). Ein Schätzer ohne Annahmen über die Koordinaten muss sich daher bei jeder möglichen Umbenennung konsistent verhalten. Ihm bleibt nur das Wiederholungsmuster. Um über $O(\log n)$ hinauszukommen, muss ein Schätzer Annahmen über die Daten treffen. KSG setzt aussagekräftige Koordinatenabstände und glatte Dichten voraus. Dafür gibt es bei endlichen Stichproben keine Garantie für seinen Schätzwert. Die variationellen Schranken aus :numref:`sec_mdl-mi-variational-bounds` können größere Werte als $\log n$ liefern. In :numref:`subsec_mdl-exact-critics` sehen wir, wie verlässlich diese dann sind.

<!--en:94492634-->
### Die Debatte um die Kompressionsphase

<!--en:b5fd2fab-->
Unsere beiden Beobachtungen, die Koordinatenabhängigkeit von Histogrammen und die logarithmische Obergrenze durch die Stichprobengröße, erhellen eine bekannte Debatte im Deep Learning. :citet:`Shwartz-Ziv.Tishby.2017` trainierten einen kleinen Klassifikator auf den $4{,}096$ binären Mustern aus $12$ Bits. Während des Trainings teilten sie die Aktivierungen jeder versteckten Schicht $Z$ in Klassen ein und zeichneten die Schätzwerte für $I(X; Z)$ und $I(Y; Z)$ über den Trainingsverlauf auf, mit Eingabe $X$ und Label $Y$. Sie beobachteten eine kurze *Anpassungsphase*, in der beide Größen stiegen. Darauf folgte eine lange *Kompressionsphase*, in der $I(X; Z)$ fiel und $I(Y; Z)$ hoch blieb. Sie vermuteten, diese Kompression erkläre, warum tiefe Netze generalisieren. Später zeigten :citet:`Saxe.Bansal.Dapello.ea.2018`, dass diese Kompressionsphase bei sättigenden Aktivierungen wie $\tanh$ auftritt, bei ReLU-Aktivierungen aber verschwindet, obwohl ReLU-Netze ebenso gut generalisieren.

<!--en:4c37c980-->
Was maß das Histogramm tatsächlich? In einem deterministischen Netz ist die diskretisierte Aktivierung $Z_{\textrm{bin}}$ eine deterministische Funktion der Eingabe $X$. Daher gilt $H(Z_{\textrm{bin}} \mid X) = 0$, und der Histogrammschätzwert vereinfacht sich zu $I(X; Z_{\textrm{bin}}) = H(Z_{\textrm{bin}})$, der Entropie der diskretisierten Aktivierungen. Diese Größe beträgt höchstens den Logarithmus der Anzahl der Eingaben, also $\ln 4096 = 8.3$ Nat. Sie sinkt, wenn verschiedene Eingaben in dieselbe Histogrammzelle fallen. Genau das geschieht, wenn sättigende $\tanh$-Einheiten beim Training die beiden äußeren Klassen nahe $\pm 1$ erreichen. Die wahre gegenseitige Information $I(X; Z)$ bleibt unverändert. Solange verschiedene Eingaben verschiedene Aktivierungen erzeugen, gilt $I(X; Z) = H(X)$ für eine endliche Eingabemenge. Bei stetigen Eingaben ist $I(X; Z)$ unendlich (:numref:`subsec_mdl-mi-gaussian`). Die Kurven zeigen, wie sich Aktivierungen in festen Klassen häufen. Sie beschreiben damit eine Eigenschaft des Schätzers und geben keinen Aufschluss über eine Änderung der wahren gegenseitigen Information des Netzes (Übung :exref:`ex_mdl-mutual-information-compression-phase`).

<!--en:428685aa-->
## Variationelle untere Schranken

<!--en:a3930210-->
Da sich eine große gegenseitige Information in hohen Dimensionen nicht durch Zählen messen lässt, verfolgen variationelle Methoden ein anderes Ziel. Anstatt die unbekannten Dichten in $\log\frac{p(x, y)}{p(x)\,p(y)}$ direkt zu schätzen, führen wir ein Hilfsmodell ein. Sein erwarteter Score ist garantiert eine *untere Schranke* für $I(X; Y)$, wie unvollkommen das Modell auch sein mag. Trainieren wir das Modell so, dass dieser Erwartungswert wächst, wird die Schranke schärfer, ohne je über $I(X; Y)$ zu liegen. Aus Schätzung wird damit Optimierung. Ihr Ergebnis hat zwei verschiedene Fehlerquellen. Ein Approximationsfehler entsteht, wenn das Hilfsmodell das Optimum verfehlt. Hinzu kommt ein Stichprobenfehler, weil wir Erwartungswerte der Grundgesamtheit durch Mittelwerte über endliche Batches ersetzen müssen.

<!--en:7f574edb-->
Wir untersuchen nacheinander vier variationelle Schranken, die :citet:`Poole.Oord.Alemi.ea.2019` in einem einheitlichen Rahmen dargestellt haben. Die erste nutzt einen normierten Decoder (Barber–Agakov). Die nächsten beiden verwenden eine unnormierte Bewertungsfunktion. Das ergibt die Schranken von Donsker und Varadhan (DV) sowie von Nguyen, Wainwright und Jordan (NWJ). Die vierte, InfoNCE, ist so wichtig, dass wir ihr einen eigenen Abschnitt widmen. Alle Beweise beruhen auf einem einzigen Hilfsmittel, der Gibbs-Ungleichung (:numref:`sec_mdl-information_theory`). Sie besagt, dass eine KL-Divergenz stets nichtnegativ ist.

<!--en:add03a2d-->
### Die Barber–Agakov-Schranke

<!--en:1f196e36-->
Die einfachste variationelle Schranke ersetzt die unbekannte A-posteriori-Verteilung $p(x \mid y)$ durch ein berechenbares Modell $q(x \mid y)$.

<!--en:b175d34a-->
**Proposition (Barber–Agakov-Schranke).** *Für jede bedingte Dichte $q(x \mid y)$ gilt*

<!--en:5e9dca84-->
*wobei genau dann Gleichheit gilt, wenn $q(x \mid y) = p(x \mid y)$. Für stetiges $X$ tritt die differentielle Entropie $h(X)$ an die Stelle von $H(X)$.*

<!--en:70e5b252-->
**Beweis.** Wir gehen von $I(X; Y) = H(X) - H(X \mid Y)$ aus und ziehen die rechte Seite von :eqref:`eq_mdl-ba_bound` ab:

<!--en:4442bf1c-->
Die Ungleichung folgt aus der Gibbs-Ungleichung für jedes $y$ durch Mittelung über $P_Y$. Gleichheit gilt genau dann, wenn die KL-Divergenz für jedes $y$ verschwindet, also wenn $q(x \mid y) = p(x \mid y)$. $\blacksquare$

<!--en:19a47021-->
Die Barber–Agakov-Schranke :cite:`Barber.Agakov.2003` liefert eine informationstheoretische Deutung eines Encoder-Decoder-Modells. Rekonstruiert ein Decoder $q(x \mid y)$ die Variable $X$ aus $Y$ mit geringer Kreuzentropie, müssen $X$ und $Y$ viel Information teilen. In der Praxis schränkt der meist unbekannte Term $H(X)$ die Anwendbarkeit ein. Allerdings hängt $H(X)$ nicht von $q$ ab. Um die Schranke über $q$ zu *maximieren*, müssen wir ihn also nie kennen. Mit der Schreibweise $H(X) = -\mathbb{E}[\log p(x)]$ ergibt sich die äquivalente Verhältnisform $I(X; Y) \geq \mathbb{E}\left[\log \left(q(x \mid y)/p(x)\right)\right]$. Sie liefert den entscheidenden Schritt im Beweis der InfoNCE-Schranke.

<!--en:97f2d977-->
In diesem Abschnitt sagen wir, ein Modell *garantiere* den Wert seiner Schranke. Diese Garantie bezieht sich auf Erwartungswerte der Grundgesamtheit. Ein Mittelwert über einen endlichen Batch schätzt den garantierten Wert lediglich und kann darüber oder darunter liegen.

<!--en:e4dd48ee-->
Ein gewöhnlicher Klassifikator ist ein vertrauter Spezialfall. Wegen der Symmetrie der gegenseitigen Information können wir $X$ und $Y$ vertauschen und einen Klassifikator $q(y \mid x)$ als Decoder des Labels auffassen. So erhalten wir wieder die Garantie aus dem Verlust in :numref:`subsec_mdl-loss-certificate`. Für $k$ gleich häufige Klassen und eine auf zurückgehaltenen Daten gemessene Kreuzentropie $L$ gilt $I(X; Y) \geq \log k - L$. Wegen $L \geq 0$ kann ein Klassifikator mit $10$ Klassen nie mehr als $\ln 10 = 2.30$ Nat garantieren. InfoNCE hat genau dieselbe Form.

<!--en:d131eb52-->
Unser Gauß-Paar zeigt, wie viel Information einem unpassenden Decoder entgehen kann. Trainieren wir einen Prädiktor allein mit dem quadratischen Fehler, entspricht das einem Gauß-Decoder mit Einheitsvarianz $q(x \mid y) = \mathcal{N}(x; a y, 1)$. Wegen $h(X) = \tfrac{1}{2}\log(2\pi e)$ ergibt sich für die Barber–Agakov-Schranke $\tfrac{1}{2} - \tfrac{1}{2}\mathbb{E}[(X - aY)^2] = \tfrac{1}{2} - \tfrac{1}{2}(1 - 2a\rho + a^2)$. Sie erreicht bei $a = \rho$ ihr Maximum $\rho^2/2$. Bei $\rho = 0.9$ garantiert der Decoder nur $0.41$ Nat von den tatsächlichen $0.83$. Wie nahe $\rho$ auch an $1$ heranrückt, er kann nie mehr als $\tfrac{1}{2}$ Nat garantieren. Die fehlende Information ist der Preis dafür, dass wir die Varianz des Decoders auf $1$ festlegen, anstatt sie an die wahre bedingte Varianz $1 - \rho^2$ anzupassen (Übung :exref:`ex_mdl-mutual-information-ba-gaussian-decoder`).

<!--en:fb24f022-->
### Die Donsker–Varadhan-Schranke

<!--en:7abcc768-->
Unsere zweite Schranke kommt ohne die beiden Voraussetzungen von Barber–Agakov aus. Sie benötigt weder $H(X)$ noch eine normierte Dichte $q$. Die gegenseitige Information ist die KL-Divergenz der gemeinsamen Verteilung vom Produkt ihrer Randverteilungen (:numref:`sec_mdl-mutual-information`). Daher ergibt jede untere Schranke für die KL-Divergenz unmittelbar eine untere Schranke für $I(X; Y)$. Die Donsker–Varadhan-Schranke nutzt dazu eine beliebige Bewertungsfunktion $T$, den *Kritiker* (wie in :numref:`sec_mdl-f-gan-dual`). Sie unterliegt keinerlei Normierungsbedingung.

<!--en:84d548f5-->
**Proposition (Donsker–Varadhan-Schranke).** *Für Verteilungen $P$ und $Q$ sowie jede Funktion $T$, für die $\mathbb{E}_P[T]$ definiert ist und $\mathbb{E}_Q[e^T] < \infty$ gilt, haben wir*

<!--en:b82e0765-->
*wobei genau dann Gleichheit gilt, wenn $T = \log \frac{dP}{dQ} + c$ für eine Konstante $c$ gilt. Falls $P \ll Q$ und die Divergenz endlich ist, ergibt das Supremum über $T$ den Wert $D_{\textrm{KL}}(P \| Q)$. Dies ist die Donsker–Varadhan-Darstellung der KL-Divergenz* :cite:`Donsker.Varadhan.1983`.

<!--en:d0bcebbb-->
**Beweis.** Ist $P$ nicht absolut stetig bezüglich $Q$, ist die linke Seite unendlich und die Ungleichung gilt unmittelbar. Andernfalls gewichten wir $Q$ mithilfe des Kritikers neu. Wir definieren die *exponentiell umgewichtete Verteilung* (engl. tilted distribution) $\tilde{Q}$ mit der Dichte $\tilde{q}(x) = q(x)\, e^{T(x)} / Z$, wobei $Z = \mathbb{E}_Q[e^T]$ die Normierungskonstante ist. Dann gilt

<!--en:47a0418a-->
nach der Gibbs-Ungleichung. Gleichheit gilt genau dann, wenn $P = \tilde{Q}$, also $p/q = e^{T}/Z$ oder äquivalent $T = \log(p/q) + \log Z$. $\blacksquare$

<!--en:3b8f10b6-->
Der Abstand der Schranke zum wahren Wert beträgt $D_{\textrm{KL}}(P \| \tilde{Q})$. Ein Kritiker ist also genau dann exakt, wenn die exponentielle Umgewichtung von $Q$ mit $e^T$ die Verteilung $P$ ergibt. Verschieben wir $T$ um eine beliebige Konstante $c$, heben sich die Änderungen der beiden Terme auf. Setzen wir $P = P_{X,Y}$ und $Q = P_X \otimes P_Y$, erhalten wir für jeden Kritiker $T(x, y)$ eine untere Schranke für die gegenseitige Information:

<!--en:b36b5b54-->
Der optimale Kritiker ist bis auf eine beliebige additive Konstante $\textrm{pmi}(x, y)$. In der Praxis schätzen wir den ersten Erwartungswert durch Mittelung von $T$ über beobachtete Paare und den zweiten durch Mittelung von $e^T$ über neu zusammengestellte Paare aus demselben Minibatch. Parametrisieren wir $T$ durch ein neuronales Netz, erhalten wir *MINE* (Mutual Information Neural Estimation) :cite:`Belghazi.Baratin.Rajeshwar.ea.2018`.

<!--en:76a55418-->
### Die NWJ-Schranke

<!--en:3434f65c-->
Obwohl die Donsker–Varadhan-Schranke für Erwartungswerte scharf ist, verursacht der Logarithmus um $\mathbb{E}_Q[e^T]$ bei Minibatches eine leicht zu übersehende Verzerrung. Da $\log$ konkav ist, folgt aus der Jensen-Ungleichung, dass der Logarithmus eines Batchmittelwerts im Mittel kleiner als der Logarithmus des wahren Erwartungswerts ist. Ziehen wir eine kleinere Zahl ab, verschiebt sich der Batchschätzwert $\overline{T} - \log \overline{e^T}$ im Mittel *nach oben*. Für den Schätzer einer unteren Schranke ist das unerwünscht.

<!--en:0bf96af9-->
Mit der elementaren Tangentenungleichung $\log z \leq z/e$ lässt sich der Logarithmus beseitigen. Sie gilt für alle $z > 0$ mit Gleichheit bei $z = e$. Wenden wir sie auf $Z = \mathbb{E}_Q[e^T]$ in :eqref:`eq_mdl-dv_bound` an, ergibt sich für jeden Kritiker $T$

<!--en:29a65e98-->
Dies ist die *NWJ-Schranke* :cite:`Nguyen.Wainwright.Jordan.2010`. Für jeden festen Kritiker $T$ ist NWJ höchstens so groß wie DV und stimmt nur dann mit DV überein, wenn $Z = e$ gilt. Bei $T^* = 1 + \textrm{pmi}$ ist die Schranke scharf. Während DV gegenüber einer additiven Konstante im Kritiker invariant ist, legt NWJ diese Konstante auf $1$ fest. Die Schreibweise $e^{-1}\, \mathbb{E}_Q[e^T] = \mathbb{E}_Q[e^{T-1}]$ zeigt, dass NWJ die allgemeine duale Schranke für f-Divergenzen aus :numref:`sec_mdl-f-gan-dual` im Spezialfall der KL-Divergenz ist, $D_{\textrm{KL}}(P \| Q) \geq \mathbb{E}_P[T] - \mathbb{E}_Q[e^{T-1}]$. Dort haben wir sie aus der konvex Konjugierten von $u \log u$ hergeleitet. Für die KL-Divergenz verschärft Donsker–Varadhan diese allgemeine Schranke strikt.

<!--en:4199ee47-->
Da der zweite Erwartungswert außerhalb des Logarithmus steht, liefert die Auswertung von :eqref:`eq_mdl-nwj_bound` auf einem Minibatch eine *erwartungstreue* Schätzung der unteren Schranke. Erwartungstreue allein sichert jedoch keine Verlässlichkeit. Beim optimalen Kritiker ist $e^{T^*-1} = e^{\textrm{pmi}}$ das Dichteverhältnis $p(x,y)/(p(x)p(y))$. Sein Erwartungswert über unabhängige Paare ist $1$. Hängen $X$ und $Y$ stark voneinander ab, bestimmen wenige sehr große Werte unter vielen Werten nahe null diesen Erwartungswert. Fehlen diese seltenen Spitzen zufällig in einem Batch, unterschätzt er den zweiten Term erheblich und überschätzt daher die Schranke. In :numref:`subsec_mdl-exact-critics` berechnen und messen wir diesen Effekt.

<!--en:e42a6222-->
## InfoNCE

<!--en:3cf9d5d7-->
### Der kontrastive Verlust

<!--en:99929156-->
Der Exponentialterm $e^T$ in der DV- und der NWJ-Schranke ist unnormiert. Daher kann ein einziges extremes Paar einen ganzen Batch dominieren. InfoNCE vermeidet diese Instabilität durch Normierung *innerhalb* des Batches. Dadurch treten die Kandidaten gegeneinander an. Wir ziehen ein *positives* Paar $(x, y^+) \sim P_{X,Y}$ und bezeichnen $x$ als *Anker* (engl. anchor). Hinzu kommen $N - 1$ *negative Kandidaten* $y_2, \ldots, y_N$, unabhängig gezogen aus der Randverteilung $P_Y$. Wir bewerten alle $N$ Kandidaten mit einem Kritiker $T(x, y)$. Auf seine Scores wenden wir die Softmax-Funktion an, um den wahren Partner vorherzusagen. Die resultierende Kreuzentropie ist der *InfoNCE-Verlust* :cite:`Oord.Li.Vinyals.2018`:

<!--en:fedf1620-->
Dabei erstreckt sich der Erwartungswert sowohl über das positive Paar als auch über die $N - 1$ negativen Kandidaten. Die zugehörige Schätzung der gegenseitigen Information erhalten wir, indem wir diesen Verlust von $\log N$ abziehen:

<!--en:7be06814-->
Einige Arbeiten schreiben den Nenner als Mittelwert $\frac{1}{N}\sum_{j=1}^N e^{T(x, y_j)}$. Dann ist das erwartete logarithmische Verhältnis direkt gleich $\hat{I}_{\textrm{NCE}}$. Im Kern ist $\mathcal{L}_{\textrm{NCE}}$ eine Kreuzentropie für $N$ Klassen. Die Klassen sind hier die anderen Beispiele im Batch anstelle fester Labels. Dies ist das Trainingsziel von CPC, SimCLR und CLIP :cite:`Oord.Li.Vinyals.2018,Chen.Kornblith.Norouzi.ea.2020,radford2021learning`. Dabei sind $x$ und $y$ zwei Ansichten oder Modalitäten desselben Beispiels. Die übrigen Beispiele im Batch dienen als negative Kandidaten.

<!--en:983c9e5c-->
![Die InfoNCE-Aufgabe. Links bewertet der Kritiker einen Anker $x$ zusammen mit seinem gemeinsam gezogenen positiven Partner $y^+$ und mit $N-1$ unabhängigen negativen Kandidaten aus der Randverteilung. Die Strahlen tragen die Scores $T(x, y_j)$. Rechts soll die Softmax-Funktion über die $N$ Scores den positiven Kandidaten erkennen. Die resultierende Kreuzentropie garantiert $I(X;Y) \geq \log N - \mathcal{L}_{\textrm{NCE}}$, eine Schranke, die höchstens $\log N$ beträgt.](../img/mdl-it-infonce-pos-neg.svg)

<!--en:24110e1d-->
:numref:`fig_mdl-infonce-pos-neg` veranschaulicht den Aufbau. Im kleinsten nichttrivialen Fall $N = 2$ gibt es einen einzigen negativen Kandidaten. Der Verlust reduziert sich auf die binäre Kreuzentropie für die Unterscheidung des wahren Partners von einem falschen. Selbst ein perfektes Modell zeigt dann nur, dass es aus zwei Möglichkeiten die richtige gewählt hat. Der Schätzwert beträgt also höchstens $\ln 2 = 0.69$ Nat.

<!--en:897a201e-->
### Die InfoNCE-Schranke

<!--en:42b91ade-->
Die folgende Proposition bestätigt, dass die Minimierung des InfoNCE-Verlusts eine untere Schranke für die gegenseitige Information garantiert. Sie zeigt auch, woher die Obergrenze $\log N$ kommt :cite:`Poole.Oord.Alemi.ea.2019`.

<!--en:14fcad0d-->
**Proposition (InfoNCE-Schranke).** *Für jeden messbaren Kritiker $T$ und jedes $N \geq 1$ gilt*

<!--en:9a253bce-->
*Außerdem beträgt die rechte Seite höchstens $\log N$.*

<!--en:39f46a17-->
**Beweis.** Bezeichnen wir die Kandidaten mit $y_1 = y^+, y_2, \ldots, y_N$ und fassen wir sie in einer zufälligen Gruppe $V = (Y_1, \ldots, Y_N)$ zusammen. Die negativen Kandidaten $Y_{2:N}$ sind unabhängig von $(X, Y_1)$. Wir ersetzen $Y$ durch die Gruppe $V$, konstruieren eine normierte bedingte Verteilung über Gruppen bei gegebenem $X$ und wenden die Barber–Agakov-Schranke an.

<!--en:c8aa1ae5-->
*Schritt 1: Unabhängige negative Kandidaten liefern keine zusätzliche Information.* Nach der Kettenregel für die gegenseitige Information (:numref:`sec_mdl-mutual-information`) gilt $I(X; Y_1, Y_{2:N}) = I(X; Y_1) + I(X; Y_{2:N} \mid Y_1)$. Da $Y_{2:N}$ unabhängig von $(X, Y_1)$ ist, ist es auch unabhängig von $X$, bedingt auf $Y_1$. Der zweite Term ist daher $0$, und es folgt $I(X; V) = I(X; Y)$.

<!--en:74a797fb-->
*Schritt 2: Das Verhältnis aus der Softmax-Funktion definiert ein normiertes Modell.* Entscheidend ist die folgende Rechnung. Wir definieren den umskalierten Softmax-Score

<!--en:13f02a28-->
mit Werten in $(0, N]$. Sei $p(v)$ die Verteilung einer Gruppe von $N$ unabhängigen Ziehungen aus $P_Y$, einschließlich der ersten Position. Da alle $N$ Einträge unabhängig und identisch verteilt sind, verändert eine Permutation die Verteilung $p(v)$ nicht. Alle $N$ Verhältnisse $e^{T(x, Y_i)} / \left(\frac{1}{N}\sum_j e^{T(x, Y_j)}\right)$ haben daher unter $p(v)$ denselben Erwartungswert. Für jedes feste $x$ gilt

<!--en:39e308f0-->
Somit ist $q(v \mid x) = p(v)\, \gamma(x, v)$ eine gültige bedingte Wahrscheinlichkeitsverteilung über Gruppen.

<!--en:0463ecd4-->
*Schritt 3: Barber–Agakov auf die Gruppe anwenden.* Da $Y_1$ die Randverteilung $P_Y$ hat und unabhängig von $Y_{2:N}$ ist, hat auch $V$ die Randverteilung $p(v)$. Wenden wir die Verhältnisform von :eqref:`eq_mdl-ba_bound` auf $(X, V)$ an und vertauschen die Rollen der beiden Variablen (wegen der Symmetrie von $I$), ergibt sich

<!--en:e469f5b6-->
Die letzte Gleichheit gilt, weil $\log \gamma$ gleich $\log N$ plus dem Logarithmus der Softmax-Wahrscheinlichkeit des positiven Kandidaten ist. Zusammen mit Schritt 1 beweist dies die Schranke. Schließlich kann die Softmax-Wahrscheinlichkeit des positiven Kandidaten höchstens $1$ betragen. Es gilt also $\mathcal{L}_{\textrm{NCE}} \geq 0$, und die rechte Seite beträgt höchstens $\log N$. $\blacksquare$

<!--en:51344a5a-->
Anschaulich weist selbst die fehlerfreie Auswahl des wahren Partners unter $N$ Kandidaten höchstens $\log N$ Nat nach. So viel Information enthält eine einzige Wahl unter $N$ Alternativen. Kein Kritiker kann dieser Aufgabe mehr Information entnehmen, ebenso wie ein Klassifikator mit $10$ Klassen nie mehr als $\ln 10$ Nat garantieren kann. Ein größeres $N$ hebt die Obergrenze an. Das ist einer der Gründe für die großen Batches in kontrastiven Verfahren wie CPC, SimCLR und CLIP. Wie wir am Ende des Abschnitts sehen, macht eine schärfere Schranke allein aber noch keine gute Repräsentation aus. Jeder Kritiker der Form $T^*(x, y) = \textrm{pmi}(x, y) + c(x)$ minimiert den Verlust, wobei $c(x)$ eine beliebige Funktion allein des Ankers ist (Übung :exref:`ex_mdl-mutual-information-optimal-critic`).

<!--en:094579b7-->
### Die Obergrenze bei exaktem Kritiker

<!--en:8185c195-->
Man könnte erwarten, dass beim exakten Kritiker $T = \textrm{pmi}$ die Schätzung $\hat{I}_{\textrm{NCE}}$ dem wahren Wert $I(X; Y)$ bis zur Obergrenze $\log N$ folgt und dort scharf abknickt. Setzen wir $T = \textrm{pmi}$ in :eqref:`eq_mdl-infonce_est` ein, zeigt sich ein glatterer und ernüchternderer Verlauf:

<!--en:c0c6758f-->
Der Schätzwert entspricht der wahren gegenseitigen Information abzüglich eines Korrekturterms, der nach der Proposition nichtnegativ ist. Seine Größe lässt sich überschlagen. Im Logarithmus steht ein Mittelwert über $N$ Terme. Die $N - 1$ negativen Kandidaten sind unabhängig von $x$. Jeder zugehörige Term hat daher den Mittelwert $1$ (die zweite Identität, die wir zu Beginn des Abschnitts überprüft haben). Zusammen tragen sie ungefähr $1$ zum Mittelwert bei. Der positive Kandidat trägt $e^{\textrm{pmi}(x, y^+)}/N$ bei, für ein typisches Paar also etwa $e^I / N$. Ersetzen wir den zufälligen Mittelwert durch diesen typischen Wert, ergibt sich

<!--en:f30197db-->
Die rechte Seite ist ein glattes Minimum von $I$ und $\log N$. Sie folgt $I$ eng für $e^I \ll N$ und nähert sich $\log N$ für $e^I \gg N$. Im Übergangsbereich liegt sie deutlich unter beiden Größen. Genau bei $I = \log N$ ergibt sich $\log N - \log 2$, also ein Abstand von $0.69$ Nat. Unsere anfängliche Vermutung übersah, dass der positive Kandidat in seinem eigenen Nenner steht, neben negativen Kandidaten mit mittlerem Beitrag $1$. Der positive Kandidat dominiert den Nenner erst, wenn $e^I$ wesentlich größer als $N$ ist.

<!--en:f3caa92d-->
Einen Mittelwert mit schweren Verteilungsrändern durch seinen Erwartungswert zu ersetzen, ist eine grobe Heuristik. Prüfen wir daher :eqref:`eq_mdl-infonce-smooth-min` empirisch. Für $N = 128$ und $I = 4$ sagt die Näherung $3.6$ Nat voraus, obwohl die Obergrenze $\ln 128 = 4.85$ noch weit entfernt scheint. Im Folgenden bewerten wir alle $N \times N$ Paare eines Batches gleichzeitig. In Zeile $i$ wird Anker $x_i$ mit jedem $y_j$ gepaart. Der Diagonaleintrag gehört zum positiven Paar, die $N - 1$ Einträge außerhalb der Diagonalen zu seinen negativen Kandidaten. Obwohl sich die Zeilen Kandidaten teilen, bleibt die Schranke gültig (Übung :exref:`ex_mdl-mi-estimation-in-batch-negatives`). Variieren wir $I(X; Y)$ für $N = 16$, $128$ und $1024$ (Obergrenzen $2.77$, $4.85$ und $6.93$ Nat) und vergleichen wir die empirischen Schätzwerte mit den wahren Werten.

<!--en:9c31c223-->
Keine Kurve hat einen scharfen Knick. Jede entfernt sich früh von der Diagonalen und nähert sich allmählich ihrer Obergrenze. Alle neun ausgegebenen Werte weichen um höchstens $0.1$ Nat von :eqref:`eq_mdl-infonce-smooth-min` ab. Bei $I = 1$ liegen die Schätzungen für jede Batchgröße bis auf etwa $0.1$ Nat beim wahren Wert. Bei $I = 4$ liefert ein Batch der Größe $128$ nur noch $3.7$ Nat, und selbst bei $N = 1024$ bleibt die Schätzung etwas zurück. Bei $I = 7$, knapp jenseits der Obergrenze $6.93$, liefert $N = 1024$ nur $6.3$.

<!--en:98750ce8-->
Zwei exakte Aussagen erklären dieses Verhalten. Erstens ist der Mittelwert im Logarithmus mindestens so groß wie der Beitrag $e^{\textrm{pmi}(x, y^+)}/N$ des positiven Kandidaten. Der Abstand beträgt also mindestens $\mathbb{E}[\textrm{pmi}(x, y^+)] - \log N = I - \log N$, woraus wieder die Obergrenze $\log N$ folgt. Zweitens verschwindet der Abstand erst im Grenzfall $N \to \infty$ (Übung :exref:`ex_mdl-mutual-information-optimal-critic`). Die feste Obergrenze $\log N$ ist nur der gröbste Ausdruck dieses Abstands bei endlicher Batchgröße.

<!--en:0457c836-->
Durch diese glatte Kompression lässt sich auch schwer vom Schätzwert auf den wahren Wert zurückschließen. Bei $N = 128$ entspricht ein Schätzwert von $3.7$ einem wahren Wert von $4$ Nat und ein Schätzwert nahe $4.75$ einem wahren Wert von $7$ Nat. Damit werden alle wahren Werte von $4$ Nat bis unendlich in die obersten $1.2$ Nat der Skala gedrängt.

<!--en:057c7acd-->
Die Bildklassifikation macht diese Zahlen anschaulich. Angenommen, jede von zwei Ansichten eines Bildes bestimmt seine Klasse unter $1{,}000$ gleich häufigen Kategorien eindeutig. Da das Label eine Funktion jeder einzelnen Ansicht ist, folgt aus der Datenverarbeitungsungleichung (:numref:`subsec_mdl-dpi`), dass die Ansichten mindestens die Entropie des Labels teilen, also $\ln 1000 = 6.91$ Nat. Ein InfoNCE-Verlust mit $N = 128$ Kandidaten kann keinen annähernd so großen Wert garantieren. Selbst mit $N = 1024$ ergibt die Schätzung nur $6.3$ bei einer wahren gemeinsamen Information von $7$. Ob die beiden Ansichten genug Information für eine feingranulare Klassifikation bewahren, lässt sich an einer solchen Zahl nicht ablesen.

<!--en:e2d82e61-->
### Exakte Kritiker im Vergleich

<!--en:537cb5ba-->
Wir können DV, NWJ und InfoNCE nun unter gleichen Bedingungen vergleichen. Die Batchgröße ist $N = 128$, und jeder Schätzer erhält seinen exakten optimalen Kritiker $T^* = 1 + \textrm{pmi}$ (eine additive Konstante lässt InfoNCE unverändert). DV und NWJ verwenden als unabhängige Paare die $N$ neu zusammengestellten Paare des Batches. Dank der zuvor beobachteten schweren Verteilungsränder können wir das Verhalten vorhersagen, noch bevor wir eine einzige Codezeile ausführen.

<!--en:dfa2561c-->
Ein typisches Paar aus der gemeinsamen Verteilung hat eine PMI nahe $I$. Unter der Produktverteilung $P_X \otimes P_Y$ tritt ein solches Paar mit Wahrscheinlichkeit ungefähr $e^{-I}$ auf, denn es gilt $p(x)\,p(y) = p(x, y)\,e^{-\textrm{pmi}}$. Unter etwa $e^I$ neu zusammengestellten Paaren sieht also eines plausibel genug aus, um als echtes Paar durchzugehen. Jedes dieser *seltenen Paare* hat ein sehr großes Gewicht $e^{\textrm{pmi}} \approx e^I$, und zusammen erklären sie den gesamten Erwartungswert $\mathbb{E}[e^{\textrm{pmi}}] = 1$. Ein Batch der Größe $N$ enthält im Mittel ungefähr $N e^{-I}$ solche Paare. Das sind etwa sechs bei $I = 3$, zwei bei $I = 4$ und nur $0.3$ bei $I = 6$.

<!--en:19a0e6c5-->
Solange ein Batch mehrere seltene Paare enthält, bleibt sein Stichprobenmittelwert von $e^{\textrm{pmi}}$ nahe $1$, und die drei Schranken sollten übereinstimmen. Sinkt die erwartete Anzahl $N e^{-I}$ unter $1$, enthalten die meisten Batches gar kein seltenes Paar. Wir sprechen dann von einem *Batch ohne seltenes Paar* (engl. starved batch). In einem solchen Batch liegt der Mittelwert von $e^{\textrm{pmi}}$ nahe null. NWJ zieht dann vom Batchmittelwert von $T^*$ fast nichts ab und liefert etwa $I + 1$, ein Nat zu viel. DV hingegen zieht den Logarithmus einer winzigen Zahl ab und überschätzt den wahren Wert um viele Nat.

<!--en:94d14f76-->
Da NWJ erwartungstreu ist, muss es ein Gegengewicht zu den Batches ohne seltenes Paar geben. Es ist ein *Batch mit Treffer* (engl. hit batch), der zufällig ein neu zusammengestelltes Paar mit $e^{\textrm{pmi}} \geq N$ enthält. Ein einziges solches Paar trägt mindestens $1$ zum Batchmittelwert von $e^{\textrm{pmi}}$ bei und drückt NWJ um mindestens ein Nat nach unten. Der Wert liegt dann nahe $I$ oder gelegentlich weit darunter. Bei $I = 6$ gilt für ein seltenes Paar $e^{\textrm{pmi}} \approx e^6 = 403 > N$. Jeder Batch mit einem solchen Paar ist daher ein Batch mit Treffer. Bei durchschnittlich $0.3$ seltenen Paaren pro Batch erwarten wir in etwa jedem vierten Batch einen Treffer. Die meisten übrigen Batches enthalten kein seltenes Paar.

<!--en:2b5330df-->
InfoNCE verhält sich ganz anders. Durch die Normierung innerhalb des Batches beeinflusst ein überzeugender negativer Kandidat höchstens eine Zeile der Softmax-Funktion. InfoNCE sollte stabil bei dem von :eqref:`eq_mdl-infonce-smooth-min` vorhergesagten Wert bleiben, also bei etwa $4.6$ Nat für $I = 6$. Der Übergang $N e^{-I} \approx 1$ tritt bei $I \approx \ln N$ ein. Der Logarithmus der Batchgröße begrenzt DV und NWJ ebenso wie InfoNCE. Bei DV und NWJ äußert sich diese Grenze jedoch in einer explosionsartig wachsenden Varianz statt in einer sichtbaren Obergrenze.

<!--en:774b2ff0-->
Prüfen wir diese Vorhersagen, indem wir $200$ Batches für jeden Wert von $I$ ziehen. Da schwere Verteilungsränder Stichprobenmittelwerte und Standardabweichungen stark beeinflussen, geben wir für jede Schranke den Median und den Interquartilsabstand an, für NWJ zusätzlich den Stichprobenmittelwert. Die beiden letzten Spalten zeigen den Anteil der Batches ohne seltenes Paar (Mittelwert $e^{\textrm{pmi}} < 0.1$) und den Anteil der Batches mit Treffer ($\max e^{\textrm{pmi}} \geq N$). Das Diagramm vergleicht die drei Mediane.

<!--en:02bcc338-->
Bis $I = 3$ weichen alle drei Mediane um höchstens etwa $0.2$ Nat vom wahren Wert ab. Fast kein Batch erfüllt eines der beiden Kriterien, und InfoNCE streut am wenigsten. Jenseits dieses Punkts treten wie vorhergesagt Batches ohne seltenes Paar auf. Bei $I = 4$ ist es höchstens etwa jeder zehnte, bei $I = 5$ etwa jeder dritte und bei $I = 6$ mehr als die Hälfte.

<!--en:a3da71dd-->
In der Zeile für $I = 6$ steigt der Median von NWJ auf etwa $7 = I + 1$. Das obere Quartil liegt praktisch auf demselben Wert. Die Hälfte aller Batches überschätzt den wahren Wert also um ungefähr ein ganzes Nat. Das knappe Viertel der Batches mit Treffer zieht den Stichprobenmittelwert wieder bis auf etwa ein halbes Nat an $6$ heran. Bei DV fällt das Ergebnis noch schlechter aus. Der Median liegt mehrere Nat über dem wahren Wert, das obere Quartil erreicht zweistellige Werte, und die Kurve reicht über den oberen Rand des Diagramms hinaus. InfoNCE hingegen liefert stabile $4.6$ Nat mit einem Interquartilsabstand unter $0.1$.

<!--en:bac98070-->
Die Streuung von NWJ lässt sich auch analytisch berechnen. Der Term für neu zusammengestellte Paare ist ein Batchmittelwert von $e^{\textrm{pmi}}$ mit Varianz $(e^{2I} - 1)/N$. Bei $I = 6$ und $N = 128$ ergibt sich daraus eine Standardabweichung von $36$ Nat. Sie geht auf extreme Batches mit Treffer zurück, die so selten sind, dass selbst unter $200$ Ziehungen kaum einer vorkommt (Übung :exref:`ex_mdl-mi-estimation-heavy-tails`).

<!--en:a1f9bc96-->
All das ist mit den variationellen Ungleichungen vereinbar, denn diese gelten für Erwartungswerte der Grundgesamtheit. Wir sehen hier, wie die variationellen Schranken mit der verteilungsfreien Grenze aus :numref:`subsec_mdl-mi-distribution-free` zusammenpassen. InfoNCE kann $\log N$ nicht überschreiten (wobei $N$ höchstens der Stichprobengröße entspricht) und hält damit die logarithmische Grenze konstruktionsbedingt ein. DV und NWJ können größere Zahlen liefern. Geschieht das, sind ihre Batchschätzungen jedoch keine verlässlichen unteren Schranken mehr.

<!--en:4105d2a8-->
### Training eines Kritikers

<!--en:30099ef7-->
In Anwendungen kennen wir die wahre PMI nicht und müssen den Kritiker aus Daten lernen. Liefert der kontrastive Verlust eine Schätzung unterhalb der wahren gegenseitigen Information, wie unterscheiden wir dann den Fehler eines unvollkommenen Kritikers vom Abstand der Schranke selbst? Trainieren wir dazu einen Kritiker mit zwei Encodern, genau wie bei CLIP und SimCLR. Zwei kleine Netze $g, h : \mathbb{R} \to \mathbb{R}^{16}$ betten $x$ und $y$ ein, und der Kritiker berechnet das Skalarprodukt $T(x, y) = g(x)^\top h(y)$. Alle $N \times N$ Scores für einen Batch ergeben sich dann aus einer einzigen Multiplikation der beiden Einbettungsmatrizen. Statt $N^2$ benötigen wir nur $2N$ Netzauswertungen. Wir erzeugen zwei disjunkte Mengen mit je $4{,}096$ Paaren bei $\rho = 0.99$ (wahre Information $1.96$ Nat), eine für das Training und eine für die Auswertung.

<!--en:b4a0664f-->
Jeder Encoder ist ein mehrschichtiges Perzeptron mit zwei versteckten Schichten aus je $64$ ReLU-Einheiten und gleichverteilter Glorot-Initialisierung. Für die resultierende $N \times N$-Score-Matrix ist der InfoNCE-Verlust die übliche Kreuzentropie für mehrere Klassen. Die Zielklasse für Zeile $i$ ist $i$, der Index ihres positiven Paars auf der Diagonalen.

<!--en:816d31e6-->
Wir trainieren mit $1{,}000$ Adam-Schritten bei einer Lernrate von $10^{-3}$ auf Minibatches aus je $N = 128$ Paaren aus der Trainingsmenge. Anschließend kapseln wir das trainierte Modell in `critic_np`, damit es wie `exact_scores` NumPy-Arrays entgegennimmt.

<!--en:ed28a5ed-->
Um den Fehler des Kritikers vom Abstand der Schranke zu trennen, werten wir den gelernten und den exakten Kritiker auf *denselben* zurückgehaltenen Batches aus. Dazu teilen wir die Auswertungsmenge in disjunkte Batches der Größe $N \in \{2, 8, 32, 128\}$ und mitteln die InfoNCE-Schranke über diese Batches. Einen bei $N = 128$ trainierten Kritiker können wir bei jeder anderen Batchgröße auswerten, weil der optimale Kritiker $\textrm{pmi}(x, y) + c(x)$ nicht von $N$ abhängt.

<!--en:d2097037-->
Bei jeder Batchgröße stimmt der gelernte Kritiker bis auf wenige Hundertstel Nat mit dem exakten überein. Der verbleibende Abstand zum wahren Wert von $1.96$ Nat geht daher ganz auf die Schranke zurück. Er schließt sich allmählich mit wachsendem $N$, bis beide Kritiker etwa $1.9$ Nat bei $N = 128$ erreichen. Betrachten wir $N = 8$ genauer. Die Obergrenze $\ln 8 = 2.08$ liegt bereits über dem wahren Wert $1.96$. Der Fehlbetrag von rund $0.55$ Nat entsteht also nicht dadurch, dass die Obergrenze erreicht wäre. Er ist der Abstand durch das glatte Minimum, den :eqref:`eq_mdl-infonce-smooth-min` auf etwa $0.6$ Nat schätzt. Ein skalares Gauß-Paar stellt dem Kritiker freilich eine leichte Aufgabe. In hohen Dimensionen ist sein Approximationsfehler nicht mehr vernachlässigbar. Ohne bekannten Referenzwert lassen sich die beiden Fehlerquellen nicht trennen.

<!--en:83445408-->
Kontrastive Systeme wie SimCLR ändern den Kritiker leicht ab. Sie normieren die Einbettungen $z_x$ und $z_y$ auf Einheitslänge und teilen ihre Kosinusähnlichkeit durch eine *Temperatur* $\tau$. Damit setzen sie $T(x, y) = \textrm{sim}(z_x, z_y)/\tau$ :cite:`Chen.Kornblith.Norouzi.ea.2020`. Die Temperatur steuert, wie scharf die Softmax-Funktion die Kandidaten trennt, und führt eine zweite Obergrenze ein. Da die Kosinusähnlichkeit in $[-1, 1]$ liegt, unterscheiden sich zwei Scores um höchstens $2/\tau$. Ein Kosinuskritiker kann daher nie $2/\tau$ Nat oder mehr garantieren, wie groß $N$ auch ist (Übung :exref:`ex_mdl-mutual-information-temperature`). Die Schranke bleibt damit unter $4$ Nat bei $\tau = 0.5$ und unter $20$ Nat bei $\tau = 0.1$. Eine hohe Temperatur kann die Schranke also schon deutlich früher begrenzen als die Batchgröße.

<!--en:25dc3d95-->
Welchen Schätzer sollten wir in der Praxis wählen? Für zwei ein- oder zweidimensionale Variablen mit höchstens einigen Nat gemeinsamer Information eignet sich das Zählen nächster Nachbarn. KSG lag hier bis auf etwa $0.1$ Nat beim wahren Wert. Bei hochdimensionalen Vektoren kann keiner unserer Schätzer eine große gegenseitige Information genau messen. Die Frage ist dann nicht mehr, *wie viel* Information zwei Ansichten teilen, sondern *wie viel mindestens*. InfoNCE beantwortet diese Frage verlässlich. Trainieren wir mit so vielen negativen Kandidaten, wie der Speicher erlaubt, und interpretieren wir $\log N - \mathcal{L}_{\textrm{NCE}}$ strikt als untere Schranke. Ein Wert, der höchstens etwa ein Nat unter $\log N$ liegt, besagt nur, dass die Ansichten mindestens so viel Information teilen. Ein kleiner Wert könnte auch auf einen schwachen Kritiker zurückgehen. Für DV und NWJ gibt es keinen Bereich mit eigenem Vorteil. Mehrere Nat unterhalb von $\log N$, wo ihre Batchschätzungen verlässlich bleiben, ist InfoNCE fast ebenso genau und deutlich stabiler. Weiter oben überschätzen DV und NWJ den Wert auf einem typischen Minibatch.

<!--en:f5dd1714-->
Eine schärfere untere Schranke führt schließlich nicht automatisch zu einer besseren Repräsentation. :citet:`Tschannen.Djolonga.Rubenstein.ea.2020` zeigten, dass die Güte bei nachgelagerten Aufgaben nur schwach damit korreliert, wie scharf die maximierte Schranke für die gegenseitige Information ist. Weniger scharfe Schranken ergeben oft bessere Repräsentationen, und die Architektur von Encoder und Kritiker spielt mindestens eine ebenso große Rolle wie der geschätzte Informationsgehalt. Zwei grundlegende Fragen bleiben offen. Wie lässt sich eine große gegenseitige Information in hohen Dimensionen verlässlich schätzen? Und warum liefert die Maximierung wenig scharfer variationeller Schranken dennoch so wirksame Repräsentationen?

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:b126196d-->
Allein aus endlichen Stichproben lässt sich die gegenseitige Information nur dann direkt messen, wenn die Variablen niedrigdimensional sind und wenig Information teilen. Darüber hinaus begrenzt der Logarithmus der Anzahl unterscheidbarer Ergebnisse jeden Schätzer. Die Obergrenze beträgt $\log b$ für ein Histogramm mit $b$ Klassen. Für KSG mit seiner Zählung nächster Nachbarn liegt sie ungefähr bei $\log(n/m)$ (in hohen Dimensionen tritt die Sättigung schon weit unterhalb dieser Grenze ein). Für jede verteilungsfreie untere Schranke gilt die Obergrenze $O(\log n)$. Variationelle Verfahren umgehen die Dichteschätzung, indem sie eine durch einen Decoder oder Kritiker parametrisierte untere Schranke maximieren. InfoNCE, das auf der Kreuzentropie beruhende Trainingsziel des kontrastiven Lernens, hat geringe Varianz und verhält sich stabil. Mit exaktem Kritiker ergibt sich ungefähr $-\log(e^{-I} + 1/N)$ und nie mehr als $\log N$ bei $N$ Kandidaten. Donsker–Varadhan und NWJ haben keine feste Obergrenze, doch ihre Minibatchschätzungen bleiben nur bis einige Nat unterhalb von $\log N$ verlässlich. Bei unserem Gauß-Paar stimmte ein gelernter Kritiker mit zwei Encodern bis auf wenige Hundertstel Nat mit dem exakten PMI-Kritiker überein. Das bestätigte, dass der beobachtete Abstand auf die Schranke selbst zurückging. Dennoch sagt die Schärfe einer Informationsschranke allein wenig über die Güte einer gelernten Repräsentation aus.

<!--en:2294243a-->
## Übungen

<!--en:11b8af35-->
1. **Schwere Verteilungsränder der exponenzierten PMI.** ● Sei $(X, Y)$ das Gauß-Paar mit Korrelation $\rho$, $I = -\tfrac{1}{2}\log(1 - \rho^2)$ und $Q = P_X \otimes P_Y$. Unter $Q$ ist der Mittelwert von $e^{\textrm{pmi}}$ gleich $1$ für jedes $\rho$. Bei einem typischen unabhängigen Paar liegt $e^{\textrm{pmi}}$ dennoch nahe $0$, wenn $\rho$ nahe $1$ ist.
    1. Zeigen Sie, dass $\mathbb{E}_Q[e^{2\,\textrm{pmi}}] = \int p^2/q = 1/(1 - \rho^2) = e^{2I}$ gilt und $e^{\textrm{pmi}}$ somit die Varianz $e^{2I} - 1$ unter $Q$ hat. *Hinweis:* Der Integrand ist proportional zu einer Gauß-Dichte mit Präzisionsmatrix $2\boldsymbol{\Sigma}^{-1} - \mathbf{I}$. Dabei ist $\boldsymbol{\Sigma}$ die Kovarianzmatrix des Paares, und es gilt $\det(2\boldsymbol{\Sigma}^{-1} - \mathbf{I}) = 1$.
    1. Bei einer zufälligen Permutation eines Batches bleibt im Mittel ein $y$ mit seinem eigenen $x$ gepaart. Zeigen Sie, dass ein solcher Fixpunkt $e^{\textrm{pmi}}/N$ zum Batchmittelwert von $e^{\textrm{pmi}}$ über die permutierten Paare beiträgt. Der Erwartungswert dieses Beitrags ist $e^{2I}/N$, also $0.43$, $23$ und $1270$ bei $I = 2, 4, 6$ mit $N = 128$. Warum vermeidet `np.roll` diesen Effekt?
    1. Zeigen Sie für den NWJ-Kritiker $T^* = 1 + \textrm{pmi}$, dass die Batchschätzung den Mittelwert $I$ hat, wenn die neu zusammengestellten Paare unabhängige Ziehungen sind. Zeigen Sie weiter, dass der Term für diese Paare dann die Varianz $(e^{2I} - 1)/N$ hat. Das ergibt Standardabweichungen von $0.65$, $4.8$ und $36$ Nat bei $I = 2, 4, 6$ mit $N = 128$. Zeigen Sie, dass die zyklisch verschobenen Paare $(x_j, y_{j+1})$ aus der Codezelle für $N \geq 3$ dieselbe Varianz ergeben, obwohl benachbarte Terme eine Variable gemeinsam haben. *Hinweis:* Bedingen Sie auf das gemeinsame Paar.
   :exlabel:`ex_mdl-mi-estimation-heavy-tails`

<!--en:2fb71156-->
1. **Barber–Agakov mit einem Gauß-Decoder.** Betrachten Sie das standardisierte bivariate Gauß-Paar mit Korrelation $\rho$ und gegenseitiger Information $-\tfrac{1}{2}\log(1 - \rho^2)$ sowie die Decoder $q(x \mid y) = \mathcal{N}(x; a y, s^2)$ mit Parametern $a \in \mathbb{R}$ und $s^2 > 0$. Verwenden Sie in der Schranke :eqref:`eq_mdl-ba_bound` die differentielle Entropie $h(X) = \tfrac{1}{2}\log(2\pi e)$.
    1. Berechnen Sie die rechte Seite von :eqref:`eq_mdl-ba_bound` als Funktion von $a$ und $s^2$. Maximieren Sie sie über beide Parameter und zeigen Sie, dass das Maximum der wahren Information entspricht.
    1. Im Text setzen wir $s^2 = 1$, was dem Training allein mit dem quadratischen Fehler entspricht, und erhalten als beste Schranke $\rho^2/2$. Zeigen Sie, dass ihr Abstand zur wahren Information gleich $D_{\textrm{KL}}\big(\mathcal{N}(0, 1 - \rho^2) \,\|\, \mathcal{N}(0, 1)\big)$ ist, wie es aus dem Beweis der Proposition folgt. Wie verhält sich der Abstand für $\rho \to 1$?
    1. Sei $\hat{x}(y)$ ein beliebiger Prädiktor von $X$ mit mittlerem quadratischem Fehler $D$. Zeigen Sie mit dem Decoder $\mathcal{N}(x; \hat{x}(y), D)$, dass $I(X; Y) \geq -\tfrac{1}{2}\log D$ gilt. Wie lautet die entsprechende Schranke für $X \sim \mathcal{N}(0, v)$? Vergleichen Sie sie mit der gaußschen Rate-Verzerrungs-Funktion $R(D) = \tfrac{1}{2}\log(v/D)$ für $D < v$ aus :numref:`subsec_mdl-rate-distortion`. Diese gibt an, wie viel Information eine Beschreibung von $X$ mindestens benötigt, um den quadratischen Fehler $D$ zu erreichen.
   :exlabel:`ex_mdl-mutual-information-ba-gaussian-decoder`

<!--en:5e54e335-->
1. **Temperatur eines Kosinuskritikers.** Systeme nach dem Vorbild von SimCLR bewerten Kandidaten mit $T(x, y) = s(x, y)/\tau$. Dabei ist $s(x, y) = \textrm{sim}(z_x, z_y) \in [-1, 1]$ eine Kosinusähnlichkeit und $\tau > 0$ die Temperatur. Schreiben Sie für einen Anker $x$ kurz $s_j = s(x, y_j)$, wobei $y_1 = y^+$ der positive Kandidat ist. Sei $\ell$ der Verlust innerhalb des Erwartungswerts :eqref:`eq_mdl-infonce_def`.
    1. Zeigen Sie, dass $\mathcal{L}_{\textrm{NCE}} \geq \log\big(1 + (N-1)e^{-2/\tau}\big)$ gilt und daher $\hat{I}_{\textrm{NCE}} < 2/\tau$ für jedes $N$. Berechnen Sie die Obergrenze $\log N - \log\big(1 + (N-1)e^{-2/\tau}\big)$ für $N = 4096$ und $\tau \in \{1, 0.5, 0.1\}$ (Sie sollten etwa $2.0$, $4.0$ und $8.3$ Nat erhalten). Wann begrenzt die Temperatur die Schranke früher als die Batchgröße?
    1. Zeigen Sie, dass $\partial \ell / \partial s_j = p_j/\tau$ für jeden negativen Kandidaten $j \geq 2$ gilt, wobei $p_j$ die Softmax-Wahrscheinlichkeit des Kandidaten $j$ ist. Drücken Sie $\partial \ell / \partial s_1$ durch die $p_j$ aus.
    1. Zeigen Sie, dass $\tau \ell \to \max\{0, \max_{j \geq 2} s_j - s_1\}$ für $\tau \to 0$ gilt. Zeigen Sie außerdem:

       $$\ell = \log N - \frac{1}{\tau}\Big(s_1 - \frac{1}{N}\sum_{j=1}^N s_j\Big) + O(\tau^{-2}) \quad \textrm{as } \tau \to \infty.$$

       Welche negativen Kandidaten bestimmen in den beiden Grenzfällen den Gradienten :cite:`Wang.Liu.2021`?
   :exlabel:`ex_mdl-mutual-information-temperature`

<!--en:87a2d7d2-->
1. **Entropieschranken aus Samples.** ● In :numref:`subsec_mdl-mi-distribution-free` haben wir den Satz von :citet:`McAllester.Stratos.2020` informell formuliert und eine Variante für die Entropie bewiesen. In dieser Aufgabe verfeinern Sie den Beweis und wenden ihn an. Ziehen Sie eine Stichprobe $S$ mit $n$ unabhängigen, identisch verteilten Werten aus einer diskreten Verteilung $p$. Der *Typ* von $S$ gibt für jedes $i \geq 1$ an, wie viele verschiedene Werte genau $i$-mal in $S$ vorkommen. Welche Werte das sind, hält er nicht fest. Sei $B$ eine Funktion des Typs, die für jede diskrete Verteilung eine untere Konfidenzschranke liefert. Es gilt also $P_{S \sim p^n}\big(B \leq H(p)\big) \geq 0.95$ für jedes $p$. Sei $\mathcal{T}_0$ der Typ einer Stichprobe mit $n$ verschiedenen Werten.
    1. Im Text wenden wir die Garantie auf die Gleichverteilung auf $n^2$ Werten an. Zeigen Sie, dass jede ganze Zahl $M \geq n^2/1.9$ ebenfalls geeignet ist, sodass $B(\mathcal{T}_0) \leq \log M$ gilt. Um wie viel verbessert dies die Schranke $2 \log n$?
    1. Sei $p$ gleichverteilt auf $M = 2^{256}$ Werten. Wie groß muss $n$ sein, damit $2 \log n$ den Wert $H(p)$ erreicht? Was folgt daraus für die Messung der Entropie eines kryptografischen Schlüssels anhand von Samples?
   :exlabel:`ex_mdl-mutual-information-entropy-bounds-samples`

<!--en:f351d130-->
1. **Optimaler InfoNCE-Kritiker.** ● Schreiben Sie für einen Anker $x$ und eine Gruppe $v = (y_1, \ldots, y_N)$ mit $y_1 = y^+$ kurz $\ell(x, v) = -\log\big(e^{T(x, y_1)} / \sum_{j=1}^N e^{T(x, y_j)}\big)$. Dann ist $\mathcal{L}_{\textrm{NCE}}$ aus :eqref:`eq_mdl-infonce_def` der Erwartungswert dieses Verlusts.
    1. Zeigen Sie, dass das Ersetzen von $T(x, y)$ durch $T(x, y) + c(x)$ für jede Funktion $c$ den Verlust $\mathcal{L}_{\textrm{NCE}}$ unverändert lässt.
    1. Mischen Sie die Gruppe so, dass der positive Kandidat an einer gleichverteilt gezogenen Position $J$ steht und die anderen $N - 1$ Positionen die negativen Kandidaten enthalten. Zeigen Sie:

       $$P(J = i \mid x, y_1, \ldots, y_N) = \frac{e^{\textrm{pmi}(x, y_i)}}{\sum_{j=1}^N e^{\textrm{pmi}(x, y_j)}}.$$

       Folgern Sie mit der Gibbs-Ungleichung, dass jeder Kritiker $T(x, y) = \textrm{pmi}(x, y) + c(x)$ den Verlust $\mathcal{L}_{\textrm{NCE}}$ minimiert.
    1. Zeigen Sie für diesen Kritiker:

       $$\hat{I}_{\textrm{NCE}} = I(X;Y) - \mathbb{E}\left[\log \frac{1}{N}\sum_{j=1}^N e^{\textrm{pmi}(x, y_j)}\right].$$

       Begründen Sie mit dem Gesetz der großen Zahlen, dass die Korrektur für $N \to \infty$ gegen null geht, wenn $I(X;Y)$ endlich ist. Berechnen Sie die Korrektur mit der Monte-Carlo-Methode bei $\rho = 0.99$ und $N = 8$. Vergleichen Sie sie mit dem Abstand zwischen dem Wert des exakten Kritikers (etwa $1.4$) und dem wahren Wert ($1.96$) in :numref:`subsec_mdl-infonce-learned-critic`.
    1. Die Näherung :eqref:`eq_mdl-infonce-smooth-min` ersetzt den Anteil der negativen Kandidaten am Mittelwert durch $1$ und $\textrm{pmi}(x, y^+)$ durch $I$. Zeigen Sie mit der Jensen-Ungleichung, dass die erste Ersetzung den Schätzwert senkt und die zweite ihn erhöht. Welcher Effekt überwiegt in der Tabelle in :numref:`subsec_mdl-infonce-ceiling`?
   :exlabel:`ex_mdl-mutual-information-optimal-critic`

<!--en:3a1672b5-->
1. **Negative Kandidaten aus demselben Batch und symmetrischer Verlust.** Die Codezellen in :numref:`subsec_mdl-infonce-learned-critic` bewerten alle $N \times N$ Kombinationen in einem Batch aus $N$ unabhängigen Paaren $(x_i, y_i)$. Die Einträge außerhalb der Diagonalen in Zeile $i$ dienen als negative Kandidaten für Anker $x_i$.
    1. Zeigen Sie, dass jede Zeile die Voraussetzungen der InfoNCE-Proposition erfüllt. Ihre Kandidaten $y_j$ für $j \neq i$ sind Ziehungen aus $P_Y$, die unabhängig von $(x_i, y_i)$ sind. Folgern Sie, dass der mittlere Zeilenverlust $\mathcal{L}$ im Erwartungswert $I(X; Y) \geq \log N - \mathcal{L}$ garantiert.
    1. Die Zeilen teilen sich Kandidaten, daher sind ihre Verluste abhängig. Beeinflusst das die Gültigkeit der Schranke, den Erwartungswert des mittleren Zeilenverlusts oder seine Varianz?
    1. CLIP verwendet den symmetrischen Verlust, das Mittel aus zeilenweiser und spaltenweiser Kreuzentropie. Zeigen Sie, dass er $I(X; Y)$ auf dieselbe Weise beschränkt, mit derselben Obergrenze $\log N$.
    1. Ein gemeinsamer Kritiker $T(x, y)$, ein Netz mit den verketteten Komponenten des Paares als Eingabe, benötigt $N^2$ Netzauswertungen für die Scores eines Batches. Dem separierbaren Kritiker $g(x)^\top h(y)$ genügen $2N$. Berechnen Sie beide Anzahlen für die CLIP-Batches mit $N = 32{,}768$. Ein separierbarer Kritiker kann nicht jede Funktion des Paares darstellen. Zeigen Sie, dass sich die PMI des Gauß-Paares mit dreidimensionalen Einbettungen separierbar darstellen lässt und dass für InfoNCE zwei Dimensionen genügen.
   :exlabel:`ex_mdl-mi-estimation-in-batch-negatives`

<!--en:708fe972-->
1. **Negative Sampling als verschobene PMI.** ● Skip-Gram mit Negative Sampling, das Trainingsverfahren von word2vec, bewertet ein Zentrumswort $w$ und ein Kontextwort $o$ anhand des Skalarprodukts $s(w, o) = \mathbf{u}_o^\top \mathbf{v}_w$ ihrer Vektoren. Es lernt, Paare aus einem Korpus von Paaren zu unterscheiden, deren Kontextwort eines von $K$ aus einer Verteilung $P$ gezogenen Rauschwörtern ist. Sei $p(w, o)$ die Häufigkeit des Paares unter allen Paaren (Zentrum, Kontext) im Korpus, mit Randhäufigkeiten $p(w)$ und $p(o)$. Über den Korpus gemittelt ergibt sich der Verlust

   $$\ell = -\sum_{w, o} p(w, o) \log \sigma\big(s(w, o)\big) - K \sum_{w} p(w) \sum_{o} P(o) \log \sigma\big(-s(w, o)\big),$$

   wobei $\sigma(s) = 1/(1 + e^{-s})$ die Sigmoidfunktion ist. Nehmen Sie an, die Einbettungen haben genügend Dimensionen, um jeden Score $s(w, o)$ unabhängig wählen zu können.
    1. Zeigen Sie, dass $\ell$ bei $s^*(w, o) = \log \frac{p(w, o)}{p(w)\, P(o)} - \log K$ minimal wird. Dieser Wert ist gleich $\textrm{pmi}(w, o) - \log K$, wenn $P$ die Unigrammverteilung $p(o)$ ist :cite:`Levy.Goldberg.2014`.
    1. word2vec zieht Rauschwörter mit einer Wahrscheinlichkeit proportional zur Anzahl ihrer Vorkommen hoch $0.75$. Zeigen Sie, dass sich $s^*$ dann von $\textrm{pmi}(w, o) - \log K$ um einen Term unterscheidet, der nur von $o$ abhängt. Für welche Kontextwörter ist dieser Term negativ? Wie wirkt er auf Paare seltener Wörter wie „sri lanka“ aus Übung :exref:`ex_mdl-mutual-information-rare-pairs` in :numref:`sec_mdl-mutual-information`?
    1. Vergleichen Sie $s^*$ mit dem optimalen InfoNCE-Kritiker aus Übung :exref:`ex_mdl-mutual-information-optimal-critic`b. Warum legt die binäre Klassifikation beim Negative Sampling den additiven Term fest, den InfoNCE offenlässt?
   :exlabel:`ex_mdl-mutual-information-negative-sampling-pmi`

<!--en:472046d9-->
1. [extended] **Kompressionsphase im Kleinen.** ●● :citet:`Shwartz-Ziv.Tishby.2017` untersuchten vollständig verbundene tanh-Netze mit den Schichtbreiten 12-10-7-5-4-3-2 auf den $4096$ binären Eingabemustern aus $12$ Bits. Für jede versteckte Schicht $Z$ teilten sie die Aktivierung jeder Einheit in $30$ gleich breite Intervalle von $[-1, 1]$ ein. Aus den diskretisierten Werten berechneten sie $I(X;Z)$ und $I(Y;Z)$. In :numref:`subsec_mdl-compression-debate` geht es um die Debatte über die von ihnen beobachtete Kompressionsphase.
    1. Sei $X$ gleichverteilt auf den $4096$ Mustern und $Z_{\textrm{bin}}$ die diskretisierte Schicht eines deterministischen Netzes. Zeigen Sie, dass der Histogrammschätzwert von $I(X;Z)$ gleich $H(Z_{\textrm{bin}})$ ist und nie mehr als $\ln 4096 \approx 8.32$ Nat beträgt. Wann sinkt er während des Trainings?
    1. Fixieren Sie einen Zufallsvektor $\mathbf{w} \in \mathbb{R}^{12}$ und setzen Sie $Y = 1$, wenn $\mathbf{w}^\top \mathbf{x}$ seinen Median über die $4096$ Muster übersteigt. Trainieren Sie das tanh-Netz mit einer Softmax-Funktion über seinen beiden Ausgabeeinheiten. Verwenden Sie SGD mit dem Kreuzentropieverlust auf zufällig gewählten $85\%$ der Muster. Zeichnen Sie für jede versteckte Schicht ihren Weg in der Informationsebene $\big(I(X;Z), I(Y;Z)\big)$ während des Trainings.
    1. Wiederholen Sie (b) mit versteckten ReLU-Einheiten. Verwenden Sie je Einheit $30$ gleich breite Intervalle zwischen $0$ und ihrer größten Aktivierung im gesamten Lauf.
    1. Erfassen Sie während des Trainings den Anteil der tanh-Einheiten mit $|z| > 0.99$. Folgt die Abnahme des diskretisierten $I(X;Z)$ in (b) diesem Anteil? Welche Aussagen über die Kompression gelten auch nach dem Wechsel der Aktivierung in (c) :cite:`Saxe.Bansal.Dapello.ea.2018`?
   :exlabel:`ex_mdl-mutual-information-compression-phase`

<!--en:3b56af0e-->
[Diskussion](https://d2l.discourse.group/t/36866)

<!--en:b7ba163b-->
Gegenseitige Information aus Samples schätzen<br>**Histogramme und Ränge · nächste Nachbarn bei KSG · Obergrenze von McAllester–Stratos · DV- und NWJ-Schranke · InfoNCE**

<!--en:1e4be2f8-->
::: {.slide title="Warum InfoNCE höchstens ln N Nat liefert"}

<!--en:d7d72c48-->
[Motivation: kontrastives Lernen und Messung]{.kicker}

<!--en:9b42ab25-->
Beim selbstüberwachten kontrastiven Lernen (SimCLR, CLIP, CPC) bewertet ein Netz eine Ankeransicht $x$ zusammen mit ihrem positiven Partner $y^+$ und $N - 1$ negativen Kandidaten. Dazu dient der Kreuzentropieverlust für $N$ Klassen, $\mathcal{L}_{\textrm{NCE}} \ge 0$:

<!--en:60a85fc3-->
- **Obergrenze $\log N$ durch die Batchgröße:** Bei $N = 128$ Kandidaten kann $\hat{I}_{\textrm{NCE}}$ niemals $\ln 128 = 4.85\text{ nats}$ übersteigen, selbst wenn die beiden Ansichten $50\text{ nats}$ an Information teilen. Wer $1$ Partner unter $128$ Kandidaten erkennt, weist damit höchstens $\ln 128\text{ nats}$ nach.

<!--en:585bff34-->
- **Zwei Ansätze zur Schätzung der gegenseitigen Information aus Samples:**
  1. **Direktes nichtparametrisches Zählen (Histogramme und KSG):** Punkte in Klassen oder adaptiven Nachbarschaftskästen zählen. Bei $1\text{--}2$ Dimensionen und kleinem $I$ genau, in hohen Dimensionen unzuverlässig.
  2. **Variationelle untere Schranken (Barber–Agakov, DV, NWJ, InfoNCE):** Einen Hilfsdecoder oder Kritiker $T(x, y)$ trainieren, um eine garantierte untere Schranke für $I(X; Y)$ in der Grundgesamtheit zu maximieren.

<!--en:978dc099-->
::: {.slide title="Vergleichsfall: Gauß-Paar und Verteilungsränder der exponenzierten PMI"}

<!--en:b5925f69-->
[Referenzwert und schwere Verteilungsränder von Dichteverhältnissen]{.kicker}

<!--en:4f724dbc-->
Einen exakten Referenzwert für die Schätzer liefert das standardisierte bivariate Gauß-Paar $(X, Y)$ mit Korrelation $\rho$. Es gilt $I(X; Y) = -\tfrac12\log(1 - \rho^2)$, und die punktweise gegenseitige Information ist quadratisch:

<!--en:2b0aa803-->
- **Zwei grundlegende Identitäten:** Bei Ziehungen $(x, y) \sim P_{X,Y}$ aus der gemeinsamen Verteilung gilt $\mathbb{E}_{P_{X,Y}}[\textrm{pmi}(x, y)] = I(X; Y)$. Bei unabhängigen Ziehungen $(x, y) \sim P_X \otimes P_Y$ (durch zyklisches Verschieben von $y$ um einen Index) gilt $\mathbb{E}_{P_X \otimes P_Y}[e^{\textrm{pmi}(x, y)}] = 1$, aber die Varianz $\textrm{Var}_{P_X \otimes P_Y}(e^{\textrm{pmi}}) = e^{2I} - 1$ wächst mit $I$ explosionsartig.

<!--en:47280868-->
::: {.slide title="Plug-in-Schätzung mit Histogrammen: zwei gegenläufige Verzerrungen"}

<!--en:0671d19d-->
[Grobe Klassen und Rauschen bei endlichen Stichproben]{.kicker}

<!--en:437422dc-->
Die Einteilung jeder Achse in $b$ gleich breite Klassen und die Berechnung der diskreten gegenseitigen Information aus der empirischen $b \times b$-Häufigkeitstabelle ergeben zwei gegenläufige Fehler und eine feste Obergrenze $\log b$:

<!--en:f5e72409-->
- **Verzerrung nach unten durch Datenverarbeitung (grobe Klassen):** Nach der Datenverarbeitungsungleichung gilt $I(X_{\textrm{bin}}; Y_{\textrm{bin}}) \le I(X; Y)$. Übersteigt die Klassenbreite ($\approx 9/b$) die bedingte Streuung $\sqrt{1 - \rho^2}$ ($0.14$ bei $\rho = 0.99$), geht die feine Abhängigkeitsstruktur verloren.

<!--en:615253a7-->
- **Verzerrung nach oben bei endlichen Stichproben (feine Klassen):** Selbst bei Unabhängigkeit schwanken die empirischen Zellhäufigkeiten zufällig. Wegen $D_{\textrm{KL}} \ge 0$ täuscht Stichprobenrauschen stets positive gegenseitige Information vor.

<!--en:5f65dd5e-->
::: {.slide title="Koordinatenabhängigkeit und Klasseneinteilung nach Rängen"}

<!--en:4c1d16b8-->
[Verzerrungen des Gitters in einer Dimension beheben]{.kicker}

<!--en:7b325718-->
Die wahre Information $I(X; Y)$ ist unter jeder Bijektion $X \mapsto \phi(X)$ invariant. Die Transformation $X \mapsto X^3$ dehnt den Stichprobenbereich jedoch auf $\pm 90$ aus, und $90\%$ der Punkte fallen in 2–3 gleich breite Klassen. Eine Klasseneinteilung anhand der sortierten **Ränge** stellt die exakte Invarianz unter streng monoton wachsenden Abbildungen wieder her:

<!--en:99514537-->
**Warum Histogramme bei $d = 2$ an ihre Grenzen stoßen:** Bei zwei $d$-dimensionalen Vektoren und $b = 10$ Klassen je Achse hat das gemeinsame Gitter $b^{2d} = 10^{2d}$ Zellen ($10^{10}$ Zellen bei $d = 5$). Fast jede Zelle bleibt leer.

<!--en:c9b0ec59-->
::: {.slide title="KSG: adaptiver Schätzer anhand nächster Nachbarn"}

<!--en:aed88951-->
[Lokale Kästen und Streifen im Chebyshev-Abstand]{.kicker}

<!--en:3bb72e94-->
Der Schätzer von **Kraskov, Stögbauer und Grassberger (KSG)** ersetzt feste Gitter durch lokale Kästen, deren Größe sich an die Dichte der Samples anpasst:

<!--en:290afaf8-->
- **Gemeinsamer Chebyshev-Kasten ($\ell_\infty$):** Für Punkt $i$ den Abstand $r_i$ zu seinem $m$-ten nächsten Nachbarn gemäß $\max(\|x - x_i\|_\infty, \|y - y_i\|_\infty)$ bestimmen. Der Kasten umfasst den Anteil $\approx m/n$ der gemeinsamen Masse.

<!--en:3f2ed7be-->
- **Randstreifen mit genau demselben Radius $r_i$:** Die Anzahlen $n_x^{(i)}$ der Punkte mit $\|x_j - x_i\|_\infty < r_i$ und $n_y^{(i)}$ der Punkte mit $\|y_j - y_i\|_\infty < r_i$ bestimmen. Die Seitenlängen $(2r_i)$ kürzen sich im Verhältnis $\frac{m/n}{(n_x/n)(n_y/n)}$.

<!--en:e1539f2e-->
- **Korrektur der Verzerrung mit der Digammafunktion ($\psi = \Gamma'/\Gamma$):** $$\hat{I}_{\textrm{KSG}} = \psi(m) + \psi(n) - \overline{\psi(n_x + 1) + \psi(n_y + 1)}$$

<!--en:ee13d3f0-->
::: {.slide title="Implementierung von KSG und Beweis der Stichprobenobergrenze"}

<!--en:300be232-->
[Exakte obere Schranke mit der Digammafunktion]{.kicker}

<!--en:55b636e2-->
Jeder Punkt strikt innerhalb des gemeinsamen $\ell_\infty$-Kastens mit halber Breite $r_i$ erfüllt sowohl $\|x_j - x_i\|_\infty < r_i$ als auch $\|y_j - y_i\|_\infty < r_i$. Beide Randstreifen enthalten daher mindestens $m - 1$ Punkte ($n_x^{(i)}, n_y^{(i)} \ge m - 1$). Da $\psi$ monoton wächst, gilt:

<!--en:fbdc1711-->
::: {.slide title="Warum KSG in höheren Dimensionen versagt"}

<!--en:faecdb0d-->
[Der Fluch der Dimensionalität]{.kicker}

<!--en:c118d7e9-->
Bei $d$ unabhängigen Kopien des Gauß-Paares mit $\rho = 0.9$ (wahrer Wert $I = 0.830\,d\text{ nats}$) ist KSG ($n = 2{,}000, m = 5$) für $d \le 2$ genau. Der Schätzwert stagniert jedoch unter $3\text{ nats}$ für $d \ge 8$, weit unter der Stichprobenobergrenze von $6.09$ Nat:

<!--en:12cab317-->
- **Warum lokale Konstanz geometrisch scheitert:** Um einen Anteil $m/n = 0.0025$ der Samples in $2d$ Dimensionen zu erfassen, muss ein Kasten den Anteil $(m/n)^{1/(2d)}$ jeder Achse überdecken ($5\%$ bei $d = 1$, bereits $83\%$ bei $d = 16$). Dadurch verwischt er das bedingte Band der Breite $7\%$.

<!--en:356ffa82-->
::: {.slide title="Die verteilungsfreie Grenze von McAllester und Stratos"}

<!--en:14a43fd0-->
[Kollisionsbeweis in vier Schritten]{.kicker}

<!--en:689079ad-->
- **Satz (McAllester und Stratos, 2020):** Jeder verteilungsfreie Schätzer $B(S_n)$, der $\Pr(B(S_n) \le I(X; Y)) \ge 0.95$ für alle Verteilungen erfüllt, ist durch $\mathcal{O}(\log n)$ beschränkt ($\approx 2\log n$).

<!--en:79f1042f-->
- **Beweis mit dem Geburtstagsproblem:** Bei $M = n^2$ gleich wahrscheinlichen Symbolen ($H = 2\log n$) sind alle $n$ Ziehungen mit Wahrscheinlichkeit $> 1/2$ verschieden. Jede unter Umbenennungen invariante Schranke muss daher auf *jeder* kollisionsfreien Stichprobe einen Wert $\le 2\log n$ liefern.

<!--en:5d05a658-->
::: {.slide title="Was die Kompressionsdebatte bei tiefen Netzen erklärt"}

<!--en:020fa673-->
[Was diskretisierte Aktivierungen tatsächlich messen]{.kicker}

<!--en:937cfdc7-->
In einem deterministischen Netz gilt für die diskretisierten Aktivierungen $Z_{\textrm{bin}} = f_{\textrm{bin}}(X)$ die Beziehung $H(Z_{\textrm{bin}} \mid X) = 0$. Daher:

<!--en:15f5eaac-->
- **Sättigung und wahre gegenseitige Information:** Wachsen die Gewichte, sättigt $\tanh(z)$ nahe $\pm 1$. Eingaben fallen in die beiden äußeren Histogrammklassen, wodurch $H(Z_{\textrm{bin}})$ sinkt. ReLU hingegen sättigt nicht, und der wahre Wert $I(X; Z)$ ohne Diskretisierung sinkt nie.

<!--en:01a363c9-->
::: {.slide title="Variationelle untere Schranken: Schätzung durch Optimierung"}

<!--en:6c9fd280-->
[Der einheitliche Rahmen von Poole et al.]{.kicker}

<!--en:71c5bb24-->
- **Barber–Agakov (BA):** Ein normierter Decoder $q(x \mid y)$ garantiert $I(X; Y) \ge H(X) + \mathbb{E}[\log q(x \mid y)]$.

<!--en:50c3fec2-->
- **Donsker–Varadhan (DV / MINE) und NWJ:** Ein unbeschränkter skalarer Kritiker $T(x, y)$ ($T^* = \textrm{pmi} + c$) genügt, allerdings mit dem Nachteil $\textrm{Var}_Q(e^{\textrm{pmi}}) = e^{2I} - 1$.

<!--en:954b9516-->
- **InfoNCE (kontrastiv):** Normiert $e^{T(x,y)}$ über $N$ Kandidaten im Batch. Die geringe Varianz geht mit einer sanften Sättigung an der Obergrenze $\log N$ einher.

<!--en:bd6925d3-->
::: {.slide title="Die variationelle Decoderschranke von Barber und Agakov"}

<!--en:b5b5ead6-->
[Rekonstruktion garantiert Information]{.kicker}

<!--en:1285afb2-->
Der Ersatz der unbekannten wahren A-posteriori-Verteilung $p(x \mid y)$ in $I(X; Y) = H(X) - H(X \mid Y) = H(X) + \mathbb{E}[\log p(x \mid y)]$ durch einen beliebigen berechenbaren Decoder $q(x \mid y)$ ergibt die **Barber–Agakov-Schranke** (2003):

<!--en:87f96f56-->
- **Exakter Abstand:** Die Differenz zwischen $I(X; Y)$ und der Schranke beträgt $\mathbb{E}_{y \sim P_Y}\big[D_{\textrm{KL}}(p(\cdot \mid y) \,\|\, q(\cdot \mid y))\big] \ge 0$. Dieser Term verschwindet genau dann, wenn $q(x \mid y) = p(x \mid y)$ gilt.

<!--en:8f9bf1c3-->
- **Nachteil eines MSE-Decoders mit fester Varianz:** Für unser Gauß-Paar mit Einheitsvarianz ($h(X) = \tfrac12\log(2\pi e)$) garantiert der Gauß-Decoder mit fester Varianz $q(x \mid y) = \mathcal{N}(x; ay, 1)$ die Schranke $\tfrac12 - \tfrac12\mathbb{E}[(X - aY)^2] = \rho^2/2$ bei $a^* = \rho$. Sie erfasst nur $0.405$ von $0.830\text{ nats}$ bei $\rho = 0.9$ und übersteigt nie $0.5\text{ nats}$, selbst für $\rho \to 1$.

<!--en:79749cb0-->
::: {.slide title="Die Kritikerschranke von Donsker–Varadhan (DV) und MINE"}

<!--en:d9005362-->
[Exponentielle Umgewichtung ohne H(X)]{.kicker}

<!--en:081c4f12-->
Für die exponentiell umgewichtete Dichte $\tilde{q}(x) = q(x)e^{T(x)} / Z$ ($Z = \mathbb{E}_Q[e^T]$) beträgt der Abstand der DV-Schranke $D_{\textrm{KL}}(P \| \tilde{Q}) \ge 0$:

<!--en:6d44a99d-->
- **Spezialfall MINE:** Die Wahl $P = P_{X,Y}$ und $Q = P_X \otimes P_Y$ liefert $I(X; Y)$ bei $T^*(x, y) = \textrm{pmi}(x, y) + c$.

<!--en:a920a910-->
- **Linearisierung für NWJ:** Aus $\log Z \le Z/e$ folgt $\mathbb{E}_P[T] - e^{-1}\mathbb{E}_Q[e^T]$ (auf Minibatches erwartungstreu, exakt bei $T^* = 1 + \textrm{pmi}$).

<!--en:8019bfb7-->
::: {.slide title="Die NWJ-Schranke: erwartungstreue Batchschätzungen"}

<!--en:c65e3e4e-->
[Linearisierung des Logarithmus]{.kicker}

<!--en:32e320ee-->
Bei Donsker–Varadhan bewirkt der äußere Logarithmus in $-\log \mathbb{E}_Q[e^T]$, dass die Minibatchschätzungen $\overline{T} - \log \overline{e^T}$ **nach oben verzerrt** sind (nach der Jensen-Ungleichung gilt $\mathbb{E}[\log \overline{e^T}] < \log \mathbb{E}[e^T]$). Die Tangentenungleichung $\log Z \le Z/e$ (Gleichheit bei $Z = e$) linearisiert den zweiten Term und liefert die **NWJ-Schranke** (Nguyen, Wainwright und Jordan, 2010):

<!--en:6cec4214-->
- **Zusammenhang mit §{sec_mdl-divergences-distances} und der Dualität für $f$-GANs:** Die Schreibweise $e^{-1}\mathbb{E}_Q[e^T] = \mathbb{E}_Q[e^{T-1}]$ zeigt, dass NWJ genau die Fenchel–Young-Kritikerschranke $\mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)]$ für $f(u) = u\log u$ ist ($f^*(t) = e^{t-1}$).

<!--en:94f2410a-->
- **Festgelegte additive Konstante:** Anders als bei DV ändert eine Verschiebung von $T$ den Wert von $\log Z - Z/e$. NWJ erreicht Gleichheit $I(X; Y)$ ausschließlich bei $T^*(x, y) = 1 + \textrm{pmi}(x, y)$ (dort gilt $Z = \mathbb{E}_Q[e^{1+\textrm{pmi}}] = e$).

<!--en:9098c598-->
- **Erwartungstreu, aber mit schweren Verteilungsrändern:** Da kein $\log$ den Erwartungswert umschließt, sind die Minibatchmittelwerte von NWJ exakt erwartungstreu. Ihre Varianz $\textrm{Var}_Q(e^{\textrm{pmi}}) = e^{2I} - 1$ wächst jedoch explosionsartig.

<!--en:b9391a22-->
::: {.slide title="InfoNCE: kontrastive Suche als Schranke für die gegenseitige Information"}

<!--en:1f2858b1-->
[Softmax-Klassifikation innerhalb eines Batches]{.kicker}

<!--en:d678f137-->
Anstatt einen unnormierten Exponentialterm $e^{T(x, y)}$ zu mitteln, normiert **InfoNCE** (van den Oord et al., 2018) die Größe $e^{T}$ *innerhalb* einer Gruppe von $N$ Kandidaten. Sie enthält einen positiven Partner $y_1 = y^+$ aus $p(y \mid x)$ und $N - 1$ negative Kandidaten $y_{2:N}$, unabhängig und identisch verteilt gemäß $P_Y$:

<!--en:1a3b8aba-->
- **Kontrastiver Kreuzentropieverlust:** $$\mathcal{L}_{\textrm{NCE}} = -\mathbb{E}\!\left[\log\frac{e^{T(x, y^+)}}{e^{T(x, y^+)} + \sum_{j=2}^N e^{T(x, y_j)}}\right]$$

<!--en:794ad171-->
- **Untere Schranke für die gegenseitige Information:** $$\hat{I}_{\textrm{NCE}} = \log N - \mathcal{L}_{\textrm{NCE}} \le I(X; Y)$$

<!--en:af61d567-->
- Die selbstnormierende Softmax-Funktion begrenzt jeden Batchterm auf höchstens $\log N$ (das Verhältnis $\gamma$ liegt in $(0, N]$).

<!--en:db2612d9-->
::: {.slide title="Beweis der unteren InfoNCE-Schranke in drei Schritten"}

<!--en:a7e0578e-->
[Barber–Agakov für eine Gruppe von N Kandidaten]{.kicker}

<!--en:9fc3e11d-->
Für eine Gruppe $V = (Y_1, \ldots, Y_N)$ mit $Y_1 = Y^+$ und $Y_{2:N} \sim P_Y^{\otimes(N-1)}$ gilt:

<!--en:c209660c-->
- **Begründung:** Unabhängige negative Kandidaten tragen keine gegenseitige Information bei ($I(X; V) = I(X; Y_1)$). Wegen der Permutationssymmetrie von $p(v) = \prod_j p_Y(y_j)$ ergibt das Integral von $q(v \mid x) = p(v)\frac{N e^{T(x,y_1)}}{\sum_j e^{T(x,y_j)}}$ den Wert $1$.

<!--en:f0e73911-->
::: {.slide title="Warum InfoNCE schon unterhalb der Obergrenze ln N abflacht"}

<!--en:d921038d-->
[Analyse des Abstands bei exaktem Kritiker]{.kicker}

<!--en:143db7f7-->
Jeder Kritiker $T^*(x, y) = \textrm{pmi}(x, y) + c(x)$ minimiert InfoNCE. Einsetzen von $T = \textrm{pmi}$ in $\hat{I}_{\textrm{NCE}} = \log N - \mathcal{L}_{\textrm{NCE}}$ ergibt den exakten Abstand der Schranke für die Grundgesamtheit:

<!--en:2d9cb14c-->
- **Näherung durch ein glattes Minimum:** Beim Mittelwert im Nenner hat jeder der $N - 1$ unabhängigen negativen Kandidaten den Erwartungswert $\mathbb{E}[e^{\textrm{pmi}(x, y_j)}] = 1$. Ein typisches positives Paar trägt $\frac{1}{N}e^{\textrm{pmi}(x, y^+)} \approx \frac{e^I}{N}$ bei:
  $$\hat{I}_{\textrm{NCE}} \approx I - \log\!\left(1 + \frac{e^I}{N}\right) = -\log\!\left(e^{-I} + \frac{1}{N}\right)$$

<!--en:fdc89362-->
- **Warum InfoNCE früh abflacht:** Genau bei $I = \log N$ (also $e^I/N = 1$) tragen der positive Kandidat und die $N-1$ negativen Kandidaten gleich viel zum Nenner bei. Es bleibt $\hat{I}_{\textrm{NCE}} \approx \log N - \ln 2$, also $0.69\text{ nats}$ unter der Obergrenze.

<!--en:aa01da40-->
::: {.slide title="Simulation der glatten InfoNCE-Sättigung für verschiedene N"}

<!--en:4e179240-->
[Batchgrößen N = 16, 128, 1024]{.kicker}

<!--en:e3975872-->
Bei Variation des wahren Werts $I(X; Y) \in [0.25, 7]$ mit dem exakten Kritiker $T^* = \textrm{pmi}$ für $N \in \{16, 128, 1024\}$ folgt jede Kurve dem glatten Minimum $-\log(e^{-I} + 1/N)$ bis auf $0.1\text{ nat}$:

<!--en:a2e2ab40-->
::: {.slide title="Batches ohne seltenes Paar und Batches mit Treffer bei DV und NWJ"}

<!--en:2e4ca39f-->
[Warum unnormierte Kritiker oberhalb von ln N instabil werden]{.kicker}

<!--en:183fb29f-->
- **Seltene Paare unter $P_X \otimes P_Y$:** Ein plausibles Paar ($\textrm{pmi} \approx I$, Gewicht $e^I$) tritt zufällig mit Wahrscheinlichkeit $\sim e^{-I}$ auf. Ein Batch mit $N$ negativen Kandidaten enthält also im Mittel $N e^{-I}$ Treffer.

<!--en:44682e3d-->
- **Batches ohne seltenes Paar und Batches mit Treffer ($I > \log N \implies N e^{-I} < 1$):** Die meisten Batches *enthalten kein seltenes Paar* ($\overline{e^{\textrm{pmi}}} \ll 1$), sodass DV und NWJ den Wert $I$ stark überschätzen. Seltene Batches *mit Treffer* lassen die Streuung auf eine Standardabweichung von $\sqrt{(e^{2I}-1)/N}$ anwachsen.

<!--en:ac57db4d-->
::: {.slide title="DV, NWJ und InfoNCE mit exakten Kritikern im Vergleich"}

<!--en:acdaadf7-->
[Quartile aus 200 Batches]{.kicker}

<!--en:fb8067ed-->
Die Auswertung von $200$ Batches der Größe $N = 128$ mit exakten Kritikern zeigt, dass DV und NWJ instabil werden, sobald $I > 3$ gilt. Bei InfoNCE bleibt der Interquartilsabstand unter $0.1\text{ nat}$:

<!--en:eef09ce0-->
::: {.slide title="DV, NWJ und InfoNCE mit exakten Kritikern im Vergleich (Fortsetzung)"}

<!--en:acdaadf7-2-->
[Quartile aus 200 Batches]{.kicker}

<!--en:d4e3693a-->
Mediane und Interquartilsabstände zum Diagramm. Die letzten beiden Spalten zeigen den Anteil der Batches mit Mittelwert $e^{\textrm{pmi}} < 0.1$ über die neu zusammengestellten Paare und den Anteil mit einem solchen Paar mit $e^{\textrm{pmi}} \ge N$:

<!--en:1b9f5953-->
::: {.slide title="Training eines kontrastiven Kritikers mit zwei Encodern"}

<!--en:1fab1f24-->
[Separierbare Architektur mit Skalarprodukt]{.kicker}

<!--en:cdae6b2b-->
In der Praxis ist $\textrm{pmi}(x, y)$ unbekannt. Systeme wie SimCLR und CLIP trainieren einen **separierbaren Kritiker mit zwei Encodern** $T(x, y) = g(x)^\top h(y)$ und Einbettungen $g, h : \mathcal{X} \to \mathbb{R}^d$. Ein einziges Matrixprodukt liefert alle $N \times N$ Scores im Batch ($2N$ Encoder-Durchläufe statt $N^2$):

<!--en:2dee848c-->
::: {.slide title="Training eines kontrastiven Kritikers mit zwei Encodern (Fortsetzung)"}

<!--en:1fab1f24-2-->
[Separierbare Architektur mit Skalarprodukt]{.kicker}

<!--en:aa1701de-->
::: {.slide title="Training des Kritikers mit zwei Encodern durch InfoNCE"}

<!--en:1fab1f24-3-->
[Separierbare Architektur mit Skalarprodukt]{.kicker}

<!--en:7093700e-->
::: {.slide title="Fehler des Kritikers und Abstand der Schranke unterscheiden"}

<!--en:2fd8f035-->
[Gelernter und exakter Kritiker bei verschiedenen Batchgrößen]{.kicker}

<!--en:25e9cd56-->
Der Vergleich des trainierten Kritikers mit zwei Encodern und des exakten $\textrm{pmi}$-Kritikers auf zurückgehaltenen Batches bei $\rho = 0.99$ ($I = 1.96\text{ nats}$) trennt den Approximationsfehler des Netzes vom Abstand der Schranke:

<!--en:3be28ea1-->
- **Abstand der Schranke und Temperaturobergrenze:** Der gelernte Kritiker stimmt bis auf $0.02\text{ nats}$ bei jedem $N$ mit dem exakten überein. Der Schätzwert $\approx 1.4\text{ nat}$ bei $N = 8$ ($\ln 8 = 2.08 > 1.96$) bleibt also allein wegen des glatten Minimums hinter dem wahren Wert zurück. Bei Kosinusähnlichkeit $T(x, y) = \mathrm{sim}(z_x, z_y)/\tau \in [-1/\tau, 1/\tau]$ unterscheiden sich Scores zudem um höchstens $2/\tau$. Daraus folgt eine zweite Obergrenze $\hat{I}_{\textrm{NCE}} < 2/\tau$.

<!--en:70827540-->
::: {.slide title="Gegenseitige Information in der Praxis schätzen und zum Lernen nutzen"}

<!--en:d0ed0cf6-->
[Wann zählen, wann eine Schranke bestimmen?]{.kicker}

<!--en:71d4d3a1-->
Wie lässt sich in Forschung und Praxis ein Schätzer der gegenseitigen Information auswählen und interpretieren?

<!--en:f3a9c78a-->
- **Niedrige Dimension ($d \le 2$) und mäßige Abhängigkeit ($I \lesssim 4\text{ nats}$):** Mit **KSG und nächsten Nachbarn** (in einer Dimension auch mit Ranghistogrammen) direkt und nichtparametrisch messen. Ohne Parameterabstimmung liegt die Abweichung bei höchstens etwa $\sim 0.1\text{ nat}$.

<!--en:cd60349d-->
- **Hochdimensionale Repräsentationen ($d \gg 2$):** Die Grenze $\mathcal{O}(\log n)$ von McAllester und Stratos und das exponentielle Wachstum der Kästen verhindern eine direkte Schätzung. **InfoNCE** mit vielen negativen Kandidaten $N$ verwenden und $\log N - \mathcal{L}_{\textrm{NCE}}$ strikt als garantierte untere Schranke interpretieren.

<!--en:254624b7-->
- **Unnormierte exponentielle Kritiker (DV / NWJ) bei $I \gtrsim \log N - 2$ vermeiden:** Dass sie auf Minibatches scheinbar Werte über $\log N$ messen können, liegt an der Verzerrung nach oben in Batches ohne seltenes Paar und einer Standardabweichung von $\mathcal{O}(e^I/\sqrt{N})$.

<!--en:e33d50bc-->
- **Mehr gegenseitige Information bedeutet nicht immer eine bessere Repräsentation (Tschannen et al., 2020):** Die Genauigkeit eines linearen Klassifikators auf nachgelagerten Aufgaben hängt stark von dem induktiven Bias des Encoders und der Kritikertemperatur $\tau$ ab. Sie erreicht ihr Maximum oft bei weniger scharfen Informationsschranken.

<!--en:1b461429-->
::: {.slide title="Verständnisfragen: KSG, Temperatur und Abstand der Schranke"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:75b69e39-->
Fragen zur Schätzung der gegenseitigen Information und zu kontrastiven Schranken:

<!--en:27ccffbc-->
1. **KSG bei deterministischen Kopien:** KSG mit $m = 5$ Nachbarn auf $n = 2{,}000$ Samples eines stetigen Paares mit $Y = X \sim \mathcal{N}(0, 1)$ anwenden (wahrer Wert $I(X; Y) = \infty$). Welchen Zahlenwert liefert KSG und warum?

<!--en:cbcd1e13-->
2. **Temperaturobergrenze beim Kosinuskritiker:** Gegeben sei SimCLR mit Batchgröße $N = 256$ und Kosinuskritiker $T(x, y) = \mathrm{sim}(z_x, z_y)/\tau$ mit $\mathrm{sim} \in [-1, 1]$. Wie groß kann $\hat{I}_{\textrm{NCE}}$ bei einem unbeschränkten Kritiker werden und wie groß bei der Temperatur $\tau = 0.5$?

<!--en:929db1fa-->
3. **Ursache einer zu kleinen kontrastiven Schätzung:** Für einen Datensatz mit wahrem Wert $I(X; Y) = 1.96\text{ nats}$ liefert ein trainierter InfoNCE-Kritiker bei $N = 8$ ($\ln 8 = 2.08\text{ nats}$) den Wert $\hat{I}_{\textrm{NCE}} = 1.40\text{ nats}$. Warum fehlen $0.56\text{ nats}$, obwohl $\ln 8 > 1.96$ gilt?

<!--en:bcce5d29-->
::: {.slide title="Verständnisfragen: KSG, Temperatur und Abstand der Schranke (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständniskontrolle]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:87a3618e-->
1. **KSG-Obergrenze:** Für $Y = X$ ist der gemeinsame Chebyshev-Abstand gleich dem Randabstand. Jeder Randstreifen enthält daher nur die $m - 1 = 4$ Punkte im Kasteninneren. Das ergibt $\psi(2000) - \psi(5) = 6.09\text{ nats} \approx \ln(400)$.

<!--en:b6c0abeb-->
2. **Obergrenze durch Temperatur oder Batchgröße:** Bei einem unbeschränkten Kritiker liegt die Obergrenze bei $\ln 256 = 5.55\text{ nats}$. Mit $\mathrm{sim} \in [-1, 1]$ und $\tau = 0.5$ unterscheiden sich die Scores um höchstens $2/\tau = 4$. Daher gelten $\mathcal{L}_{\textrm{NCE}} \ge \ln(1 + 255 e^{-4}) = 1.74$ und $\hat{I}_{\textrm{NCE}} \le 5.55 - 1.74 = 3.81 < 2/\tau = 4\text{ nats}$ (die Temperatur begrenzt den Wert früher als die Batchgröße).

<!--en:0c20f6ff-->
3. **Abstand durch das glatte Minimum:** Selbst beim exakten Kritiker $T^* = \textrm{pmi}$ steht der positive Kandidat in seinem eigenen Nenner neben $N-1$ negativen Kandidaten mit Mittelwert $1$. Das ergibt näherungsweise $-\ln(e^{-1.96} + 1/8) = 1.33\text{ nats}$. Der Fehlbetrag geht auf die Schranke selbst zurück und ist kein Fehler des Kritikers.

<!--en:a6666aae-->
::: {.slide title="Zusammenfassung: Schätzer und Schranken für die gegenseitige Information"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:309ab664-->
| Schätzer / Schranke | Formel | Obergrenze / Wachstum der Varianz | Einsatzbereich und wichtigste Einschränkung |

<!--en:5b7ce624-->
| **Histogramm / Ränge** | $\sum_{i,j} \hat{p}_{ij}\log\frac{\hat{p}_{ij}}{\hat{p}_i\hat{p}_j}$ auf einem $b \times b$-Gitter | $\le \log b$, $b^{2d}$ Zellen in $\mathbb{R}^{2d}$ | Eindimensionale Ranghistogramme sind koordinateninvariant. Für $d > 2$ versagt das Verfahren. |

<!--en:cb0a793c-->
| **KSG-Nachbarn** | $\psi(m) + \psi(n) - \overline{\psi(n_x+1) + \psi(n_y+1)}$ | $\le \psi(n) - \psi(m) \approx \log(n/m)$ | Referenzverfahren für $d \le 2$. Kästen überdecken je Achse einen Anteil $(m/n)^{1/(2d)}$ und versagen bei großem $d$. |

<!--en:57adffc5-->
| **McAllester–Stratos** | Beliebige verteilungsfreie untere 95%-Konfidenzschranke $B(S_n)$ | $B(S_n) \le \mathcal{O}(\log n)$ ($\approx 2\log n$) | Zeigt, dass sich große Werte von $I$ ohne Modellannahmen nicht garantieren lassen |

<!--en:b34f0b35-->
| **Barber–Agakov** | $H(X) + \mathbb{E}_{P_{X,Y}}[\log q(x \mid y)]$ | Abstand $\mathbb{E}_y[D_{\textrm{KL}}(p(\cdot\mid y) \| q(\cdot\mid y))]$ | Garantiert $I \ge \log k - L$ für Klassifikatoren und setzt bekanntes $H(X)$ voraus |

<!--en:7ca6d38b-->
| **DV (MINE) und NWJ** | $\mathbb{E}_P[T] - \log \mathbb{E}_Q[e^T]$ und $\mathbb{E}_P[T] - e^{-1}\mathbb{E}_Q[e^T]$ | Keine Obergrenze, aber $\textrm{Var}_Q(e^{\textrm{pmi}}) = e^{2I}-1$ | Minibatches ohne seltenes Paar überschätzen den Wert stark, sobald $I \gtrsim \log N$ gilt |

<!--en:13837ea8-->
| **InfoNCE** | $\log N + \mathbb{E}\big[\log\frac{e^{T(x,y^+)}}{\sum_{j=1}^N e^{T(x,y_j)}}\big]$ | $\le \min(\log N, 2/\tau)$, $\approx -\log(e^{-I}+1/N)$ | Varianzarmes Standardverfahren für SimCLR und CLIP mit sanfter Sättigung bei $\log N$ |
