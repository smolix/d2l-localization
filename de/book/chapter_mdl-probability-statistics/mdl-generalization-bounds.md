<!--en:aa79663f-->
# Gleichmäßige Konvergenz und Generalisierung

<!--en:5367ada9-->
Im vorigen Abschnitt haben wir gesehen, wie wir einen aus Daten berechneten Mittelwert mit einem Fehlerbalken versehen. Testen wir einen Klassifikator an 10 000 zurückgehaltenen Bildern, so weicht seine wahre Fehlerrate mit 95% Konfidenz um höchstens etwa 1.4 Prozentpunkte vom Testfehler ab. Diese Garantie hat eine leicht zu übersehende Voraussetzung. Der Klassifikator muss feststehen, *bevor* wir die Testbilder ziehen. In der Praxis gehen wir selten so vor. Wir trainieren viele Konfigurationen, vergleichen ihre Validierungsscores und behalten die beste. Wir berichten also den Score der Gewinnerin eines Wettbewerbs. Sie hat auch deshalb gewonnen, weil sie bei der Stichprobe am meisten Glück hatte.

<!--en:e3a4c85d-->
:numref:`chap_classification_generalization` hat dieses Problem und die Lösung von Vapnik und Chervonenkis vorgestellt :cite:`Vapnik.Chervonenkis.1971`. Wir verlangen eine Garantie, die für alle Kandidatenfunktionen gleichzeitig gilt. Dann erfasst sie automatisch die Funktion, die das Lernverfahren auswählt. Bei einer endlichen Liste wächst die Schranke nur logarithmisch mit der Anzahl der Kandidaten. Wir greifen dieses Ergebnis für endliche Klassen kurz auf. Ein Experiment zeigt, wie viel Spielraum der Zufall bei der Auswahl lässt. Wir werfen jede von 1000 fairen Münzen 100-mal und behalten die Münze, die am häufigsten Kopf zeigt. Wie oft zeigt sie Kopf?

<!--en:54d68adb-->
Im Mittelpunkt dieses Abschnitts stehen jedoch die unendlichen Funktionenklassen, aus denen wir in der Praxis Modelle lernen. Ein später entwickelter Ansatz misst, wie vielfältig eine solche Klasse ist. Ihre *Rademacher-Komplexität* gibt an, wie gut sie durch reine Münzwürfe erzeugte Labels nachbilden kann. Das Maß erklärt, warum die Kapazität eines linearen Modells von der Norm seiner Gewichte statt von der Parameteranzahl abhängt. Es zeigt auch die Grenze der gleichmäßigen Konvergenz (engl. uniform convergence). Die tiefen Netze des Haupttexts etwa sind flexibel genug, um zufällige Labels nachzubilden. Zum Schluss greifen wir das Experiment zum doppelten Abstieg (engl. double descent) aus :numref:`subsec_double_descent` wieder auf. Unser Modell ist so einfach, dass wir jeden Teil des Mechanismus in geschlossener Form berechnen können.

<!--en:0024ab22-->
## Gleichmäßige Konvergenz

<!--en:0d42aa0c-->
Ob wir ein festes Modell auf einem Testdatensatz auswerten oder ein Modell trainieren, unterscheidet sich nur in der Reihenfolge zweier Schritte. Wir wählen die Funktion und ziehen die Stichprobe. Wir messen, wie stark sich der Score verzerrt, wenn wir die Reihenfolge umkehren. Wie viele Daten brauchen wir zusätzlich, sobald die Wahl von der Stichprobe abhängt?

<!--en:e2cfd7e7-->
### Die beste von vielen Münzen

<!--en:60c5decf-->
Formulieren wir zunächst die Garantie für eine einzelne Funktion in der Notation dieses Abschnitts. Eine Stichprobe $S=\{(\mathbf x_1,y_1),\ldots,(\mathbf x_n,y_n)\}$ besteht aus $n$ unabhängigen Ziehungen aus einer unbekannten Verteilung. Ein Verlust $\ell$ mit Werten in $[0,1]$ bewertet jede Vorhersage. Der 0-1-Verlust etwa beträgt $1$ bei einem Fehler und sonst $0$.

<!--en:43122124-->
Das *empirische Risiko* einer Funktion $f$ ist ihr mittlerer Verlust auf der Stichprobe, $\hat R(f)=\frac1n\sum_{i=1}^n\ell(f(\mathbf x_i),y_i)$. Ihr *Risiko* ist der erwartete Verlust bei einer neuen Ziehung, $R(f)=\mathbb{E}[\ell(f(\mathbf x),y)]$. (Beim 0-1-Verlust sind dies der empirische Fehler $\epsilon_\mathcal{S}(f)$ und der Populationsfehler $\epsilon(f)$ aus dem Haupttext.) Steht $f$ schon fest, bevor wir $S$ ziehen, so ist $\hat R(f)$ der Mittelwert von $n$ unabhängigen Termen in $[0,1]$ mit Erwartungswert $R(f)$. Die Hoeffding-Ungleichung (:numref:`sec_mdl-concentration-generalization`) beschränkt daher $P(|\hat R(f)-R(f)|\ge t)$ durch $2e^{-2nt^2}$. Setzen wir diese Ausfallwahrscheinlichkeit gleich $\delta=0.05$ und lösen nach $t$ auf, so erhalten wir die Halbbreite $\sqrt{\log(2/\delta)/(2n)}$. Im Haupttext heißt sie Hoeffding-Radius. Für $n=10{,}000$ ergibt sich $0.0136$, also rund 1.4 Prozentpunkte wie oben angegeben.

<!--en:650361e3-->
Ein Lernverfahren kehrt die Reihenfolge um. Es durchsucht eine Funktionenklasse $\mathcal F$ und liefert den Prädiktor $\hat f$ mit dem kleinsten empirischen Risiko $\hat R$ auf der beobachteten Stichprobe. Jedes $\hat R(f)$ schwankt um das wahre Risiko $R(f)$. So wählt das Verfahren bevorzugt eine Funktion, bei der die Schwankung zufällig günstig ausfiel. Da $\hat R(\hat f)$ das Minimum vieler schwankender Größen ist, greift eine Ungleichung für jedes einzeln festgelegte $f$ hier nicht mehr.

<!--en:431042e8-->
Unser Münzexperiment zeigt, wie groß dieser Auswahleffekt sein kann. Wir werfen jede von $k=1{,}000$ fairen Münzen $m=100$-mal. Für eine vorab benannte Münze erlaubt die Hoeffding-Ungleichung bei $\delta=0.05$ eine Abweichung des Kopfanteils von $\sqrt{\log 40/200}=0.136$. Mit mindestens 95% Wahrscheinlichkeit zeigt sie also 37- bis 63-mal Kopf (exakt beträgt die Wahrscheinlichkeit 99.3%). Für die beste der 1000 Münzen gilt diese Garantie nicht. Wiederholen wir das Experiment 2000-mal und messen, wie weit die Gewinnermünze vom Erwartungswert abweicht und wie oft sie außerhalb des Bands für eine Münze liegt.

<!--en:b11c94e4-->
In einem typischen Experiment zeigt die beste Münze 66-mal Kopf, in neun von zehn Experimenten 64- bis 69-mal. In 96% bis 97% der Durchläufe liegt sie außerhalb des Bands für eine Münze. Eine Garantie für jede vorab benannte Münze versagt also fast immer für die nachträglich ausgewählte Münze. Alle Münzen sind fair. Die Abweichung von 16 Prozentpunkten bei der Gewinnermünze entsteht allein dadurch, dass wir unter 1000 zufälligen Schwankungen die größte ausgewählt haben.

<!--en:da2dd7ac-->
:numref:`fig_mdl-gen-best-of-many` veranschaulicht den Grund. Jede einzelne Münze überschreitet die Obergrenze des Bands mit Wahrscheinlichkeit $0.0033$. Für sich genommen ist das selten, doch unter 1000 Münzen erwarten wir etwa $3.3$ solche Fälle. Werfen wir tausend Münzen, finden sich also etliche in einem Randbereich der Verteilung, den eine einzelne Münze fast nie erreicht.

<!--en:f40b6f2c-->
![Erwartetes Histogramm des Kopfanteils für 1000 faire Münzen mit je 100 Würfen in logarithmischer Darstellung. Das 95%-Hoeffding-Band für eine Münze erfasst 99% der Münzen. Dennoch erwarten wir etwa 3.3 Münzen oberhalb des Bands, und die beste Münze erreicht typischerweise einen Kopfanteil nahe 0.66. Das breitere Band aus der booleschen Ungleichung für alle 1000 Münzen erfasst auch die beste Münze.](../img/mdl-prob-gen-best-of-many.svg)

<!--en:d5b4d039-->
Die Modellauswahl auf einem Validierungsdatensatz entspricht genau diesem Experiment mit anderen Bezeichnungen. Jede Konfiguration ist eine Münze, jedes Validierungsbeispiel ein Wurf. Anders als bei unseren unabhängigen fairen Münzen verwenden wir für alle Konfigurationen dieselben Beispiele, und die wahren Fehler unterscheiden sich. Die folgende Schranke setzt jedoch weder Unabhängigkeit zwischen den Kandidaten noch gleiche Erwartungswerte voraus.

<!--en:1b67ed81-->
Wie erhalten wir also eine Garantie, die auch die ausgewählte Funktion erfasst? Nehmen wir an, die größte Abweichung in der gesamten Klasse,

<!--en:a8287891-->
betrage höchstens $t$. Dann weicht das empirische Risiko jeder Funktion um höchstens $t$ von ihrem wahren Risiko ab. Das gilt insbesondere für die ausgewählte Funktion. Schranken für dieses Supremum heißen Schranken für die *gleichmäßige Konvergenz* :cite:`Vapnik.Chervonenkis.1971`. Gilt die Garantie schon vor der Datenerhebung für alle Kandidaten, können wir die Wahl bedenkenlos nach der Datenerhebung treffen. Jeder mögliche Kandidat ist bereits erfasst (:numref:`fig_mdl-gen-uniform-sup`).

<!--en:19bf688f-->
![Gleichmäßige Konvergenz. Das empirische Risiko $\hat R(f)$ (orange) weicht um höchstens $t$ vom wahren Risiko $R(f)$ (blau) ab, und zwar für alle Funktionen der Klasse zugleich. Das Lernverfahren wählt den Minimierer $\hat f$ des empirischen Risikos. Dort liegt die Kurve in einer Senke unter dem wahren Risiko. Das Band erfasst auch diese Funktion, also gilt $R(\hat f)\le\hat R(\hat f)+t$.](../img/mdl-prob-gen-uniform-sup.svg)

<!--en:3af55a6f-->
### Endliche Klassen und die boolesche Ungleichung

<!--en:7ff93c45-->
Im einfachsten Fall steht eine endliche Kandidatenliste vorab fest, etwa ein vorgegebenes Gitter von Hyperparameterkonfigurationen. Das Ergebnis für diesen Fall kennen wir aus :numref:`chap_classification_generalization`. Der Beweis braucht nur zwei Zeilen. Nach der Hoeffding-Ungleichung gilt für ein festes $f$: Eine Abweichung von mindestens $t$ hat höchstens Wahrscheinlichkeit $2e^{-2nt^2}$. Wenden wir die boolesche Ungleichung (engl. union bound, siehe :numref:`sec_mdl-concentration-generalization`) auf alle $|\mathcal F|$ Kandidaten an, so beträgt die Wahrscheinlichkeit einer Abweichung von mindestens $t$ für *irgendeinen* der Kandidaten höchstens $2|\mathcal F|e^{-2nt^2}$. Setzen wir diese gesamte Ausfallwahrscheinlichkeit gleich $\delta$ und lösen nach $t$ auf, erhalten wir unseren ersten Satz zur gleichmäßigen Konvergenz. Für einen Verlust mit Werten in $[0,1]$ und eine unabhängig und identisch verteilte Stichprobe vom Umfang $n$ gilt mit Wahrscheinlichkeit mindestens $1-\delta$:

<!--en:77248975-->
Gegenüber der Auswertung einer einzigen festen Funktion kommt bei der Wahl unter $|\mathcal F|$ Kandidaten nur der Term $\log|\mathcal F|$ unter der Quadratwurzel hinzu.

<!--en:c8457aa8-->
Im Münzexperiment liefern $|\mathcal F|=1{,}000$ und $n=100$ die Halbbreite $\sqrt{\log(2{,}000/0.05)/200}=0.230$. Mit mindestens 95% Wahrscheinlichkeit zeigen alle Münzen, auch die Gewinnermünze, 27- bis 73-mal Kopf. Die Schranke ist recht konservativ. Die wahre Wahrscheinlichkeit, dass die Anzahl der Kopfwürfe bei irgendeiner Münze außerhalb des Bereichs von 27 bis 73 liegt, beträgt etwa 0.17% und liegt damit deutlich unter den zulässigen 5%. Fast der gesamte Faktor 30 geht auf die Hoeffding-Ungleichung für eine einzelne Münze zurück. Diese verlässt den Bereich mit Wahrscheinlichkeit $1.7\times10^{-6}$, während Hoeffding $5\times10^{-5}$ zulässt. Die boolesche Ungleichung selbst ist für seltene unabhängige Ereignisse nahezu scharf. Trotz seiner Breite gilt dieses Band auch für die Gewinnermünze, anders als das Band für eine einzelne Münze.

<!--en:f5523c2f-->
Für einen Validierungsdatensatz mit 10 000 Beispielen nennt der Haupttext die Halbbreiten $0.014$ für eine einzelne Funktion und $0.030$ für eine Million Funktionen. Lösen wir :eqref:`eq_mdl-finite-class` nach dem Stichprobenumfang $n$ auf, sehen wir deutlicher, wie viele Beispiele die Suche kostet:

<!--en:387316c5-->
Jede zusätzliche Einheit von $\log|\mathcal F|$ erfordert $1/(2t^2)$ weitere Beispiele. Bei einer Toleranz von $t=0.01$ kostet eine Verdopplung der Anzahl der Modellkandidaten $\log 2/(2\cdot 10^{-4})\approx3{,}500$ zusätzliche Validierungsbeispiele.

<!--en:c5452f15-->
Diese logarithmische Abhängigkeit folgt unmittelbar aus dem exponentiellen Abfall der Hoeffding-Schranke. Verwenden wir stattdessen die Tschebyscheff-Ungleichung, so hat ein Verlust mit Werten in $[0,1]$ eine Varianz von höchstens $\tfrac14$. Dieselbe boolesche Ungleichung liefert dann $t=\sqrt{|\mathcal F|/(4n\delta)}$. Für 2000 Konfigurationen auf 10 000 Beispielen bei $\delta=0.05$ ergibt Tschebyscheff $t=1$. Das umfasst den gesamten Wertebereich des Verlusts und sagt daher nichts aus. Hoeffding liefert hingegen $t=0.024$.

<!--en:a950a4c0-->
Schon der erste Beweisschritt braucht eine entscheidende Voraussetzung. Die Kandidatenklasse $\mathcal F$ muss vor dem Ziehen der Stichprobe feststehen. Eine öffentliche Benchmark-Rangliste verletzt diese Annahme, sobald neue Einreichungen von den Testscores früherer Einreichungen abhängen :cite:`Dwork.Feldman.Hardt.ea.2015`. Passen sich die Kandidaten an die zurückgehaltenen Daten an, verschlechtert sich die Fehlerschranke. Statt der Größenordnung $\sqrt{\log k/n}$ für $k$ feste Kandidaten ergibt sich die Größenordnung $\sqrt{k/n}$ für ein adaptives Mehrheitsvotum aus $k$ zufällig ratenden Prädiktoren (Übung :exref:`ex_mdl-concentration-generalization-leaderboard-attack`).

<!--en:8376d04b-->
## Rademacher-Komplexität

<!--en:975406ed-->
Die Funktionenklassen, aus denen wir in der Praxis Modelle lernen, sind unendlich. Jeder Gewichtsvektor $\mathbf w\in\mathbb{R}^d$ definiert einen eigenen linearen Klassifikator. Die boolesche Ungleichung über unendlich viele Ereignisse liefert hier keine Aussage. Die klassische Lerntheorie umgeht das Problem auf zwei Wegen. Sie nähert die Klasse durch endlich viele Repräsentanten an (Überdeckungszahlen) oder zählt, wie viele verschiedene Labelzuweisungen sie auf $n$ Punkten realisieren kann (VC-Dimension, siehe :numref:`chap_classification_generalization`). Wir wählen einen direkteren Ansatz :cite:`Bartlett.Mendelson.2002`. Die zugrunde liegende Frage lässt sich in der Praxis prüfen: Wie gut kann die Klasse reines Rauschen nachbilden?

<!--en:893148e5-->
Hier leiten wir die beiden zentralen Ergebnisse dieses Abschnitts her: den Satz zur gleichmäßigen Konvergenz :eqref:`eq_mdl-rademacher-bound` und die Komplexitätsschranke $Br/\sqrt n$ für normbeschränkte lineare Modelle, :eqref:`eq_mdl-linear-rademacher`. Nur ein Schritt erfordert besondere Aufmerksamkeit: die Symmetrisierung im Beweis des ersten Satzes.

<!--en:eebb0e93-->
### Korrelation mit zufälligen Vorzeichen

<!--en:6db7a048-->
Kann eine Funktionenklasse Labels ohne jedes Signal nachbilden, so kann sie ebenso leicht das Rauschen in echten Labels nachbilden. Ihr empirisches Risiko sagt dann wenig über ihr wahres Risiko aus. Um zu prüfen, wie gut eine Klasse Rauschen auswendig lernen kann, geben wir ihr fiktive Zufallslabels. Seien $\varepsilon_1,\ldots,\varepsilon_n$ unabhängige zufällige Vorzeichen, die jeweils den Wert $+1$ oder $-1$ mit Wahrscheinlichkeit $\tfrac12$ annehmen. Sie heißen *Rademacher-Variablen*. Für einen gegebenen Vorzeichenvektor $\boldsymbol\varepsilon$ misst der Mittelwert $\frac1n\sum_i\varepsilon_if(\mathbf x_i)$, wie gut $f$ auf der Stichprobe mit diesen Vorzeichen korreliert. Das Supremum über $f\in\mathcal F$ misst die beste Korrelation, die die Klasse erreichen kann.

<!--en:3d9f4acc-->
**Definition.** Die *empirische Rademacher-Komplexität* einer Klasse $\mathcal F$ reellwertiger Funktionen auf den Stichprobenpunkten $S=(\mathbf x_1,\ldots,\mathbf x_n)$ ist der Erwartungswert der besten Korrelation, die die Klasse mit zufälligen Vorzeichen erreicht:

<!--en:7e4707d5-->
Ihre *Rademacher-Komplexität* $\mathfrak R_n(\mathcal F)=\mathbb{E}_S[\widehat{\mathfrak R}_S(\mathcal F)]$ ist der Mittelwert dieser Größe über zufällige Stichproben vom Umfang $n$.

<!--en:4ae16202-->
Eine Klasse, die jedes Vorzeichenmuster nachbilden kann, erreicht einen hohen Wert, eine Klasse mit wenig Spielraum einen Wert nahe null. Die genaue Obergrenze hängt vom Wertebereich der Funktionen ab. Nimmt jedes $f\in\mathcal F$ Werte in $[-1,1]$ an, so ist jede Korrelation höchstens $1$. Die Klasse *aller* Funktionen mit Werten in $[-1,1]$ erreicht auf paarweise verschiedenen Punkten $1$, indem sie für jedes Vorzeichenmuster $f(\mathbf x_i)=\varepsilon_i$ setzt. Für eine Klasse mit Werten in $[0,1]$ beträgt die größtmögliche Komplexität $\tfrac12$. Die beste Wahl setzt $f(\mathbf x_i)=1$ überall dort, wo $\varepsilon_i=+1$ gilt, und sonst $0$. Ihre Korrelation ist der Anteil der Pluszeichen, im Mittel also $\tfrac12$. Eine unbeschränkte Klasse wie die linearen Funktionen, die wir gleich untersuchen, hat keine solche Obergrenze. Kann sie jedes Vorzeichenmuster auf der Stichprobe interpolieren, beträgt ihre Komplexität mindestens $1$ und kann noch größer sein.

<!--en:65c2aae9-->
Um ein Gefühl für die Zahlen zu bekommen, betrachten wir die einfachste nichttriviale Klasse: die zwei konstanten Funktionen $\{+1,-1\}$. Ihre beste Korrelation mit einem Vorzeichenvektor ist $\max(\bar\varepsilon,-\bar\varepsilon)=|\bar\varepsilon|$, wobei $\bar\varepsilon$ den Stichprobenmittelwert der Vorzeichen bezeichnet. Für $n=2$ liefern die vier gleich wahrscheinlichen Muster $(+,+),(+,-),(-,+),(-,-)$ die Werte $|\bar\varepsilon|=1,0,0,1$, also gilt $\widehat{\mathfrak R}_S=\tfrac12$. Für $n=4$ beträgt die Summe der Vorzeichen $\pm4$ mit Wahrscheinlichkeit $\tfrac2{16}$, $\pm2$ mit Wahrscheinlichkeit $\tfrac8{16}$ und sonst $0$. Das ergibt $\widehat{\mathfrak R}_S=\tfrac38$. Für große $n$ ist $\bar\varepsilon$ annähernd normalverteilt mit Mittelwert $0$ und Varianz $1/n$, also gilt $\widehat{\mathfrak R}_S\approx\sqrt{2/(\pi n)}$. Für $n=100$ beträgt der exakte Wert $0.0796$ gegenüber $0.0798$ aus der Normalapproximation. Selbst eine Klasse mit nur zwei Funktionen korreliert mit Rauschen in der Größenordnung $1/\sqrt n$, weil der Mittelwert von $n$ zufälligen Vorzeichen in dieser Größenordnung schwankt. Eine einelementige Klasse $\{f\}$ hat hingegen die Komplexität $0$. Ohne Auswahlmöglichkeit gilt $\mathbb{E}[\frac1n\sum_i\varepsilon_if(\mathbf x_i)]=0$.

<!--en:f47f10d7-->
Für eine lineare Klasse hat die beste Korrelation eine einfache geometrische Bedeutung, wie :numref:`fig_mdl-gen-rademacher` zeigt. Sie ist die Länge einer Summe der mit Vorzeichen versehenen Eingabevektoren, geteilt durch $n$. Dies beweisen wir gleich in :numref:`subsec_mdl-linear-class`.

<!--en:e954f19e-->
![Rademacher-Komplexität einer linearen Klasse für acht Einheitsvektoren in der Ebene und lineare Funktionen mit Gewichtsnorm eins. (a) Für ein Vorzeichenmuster stehen ausgefüllte Punkte für $+1$ und offene Punkte für $-1$. Der Einheitsgewichtsvektor $\mathbf w^*$ in Richtung der vorzeichenbehafteten Summe $\mathbf s=\sum_i\varepsilon_i\mathbf x_i$ erzielt die beste Korrelation $\|\mathbf s\|/n=0.27$. (b) Exakte Verteilung der besten Korrelation über alle 256 Vorzeichenmuster. Ihr Mittelwert, die empirische Rademacher-Komplexität, beträgt 0.316 gegenüber der im Text bewiesenen Schranke $Br/\sqrt n=0.354$.](../img/mdl-prob-gen-rademacher.svg)

<!--en:f2e9129f-->
Für eine endliche Klasse mit Werten in $[-1,1]$ gilt das *Lemma von Massart*, $\widehat{\mathfrak R}_S(\mathcal F)\le\sqrt{2\log|\mathcal F|/n}$ (Übung :exref:`ex_mdl-concentration-generalization-massart-l1`). Ihre Komplexität wächst also nur logarithmisch mit der Anzahl der Funktionen. Allgemein erfasst die Rademacher-Komplexität, wie frei eine Klasse ihre Werte auf den tatsächlich beobachteten Punkten variieren kann, unabhängig von der Anzahl ihrer Parameter oder Funktionen.

<!--en:a2d98617-->
### Der Satz zur gleichmäßigen Konvergenz

<!--en:c0368e92-->
Nun können wir zeigen, dass die Rademacher-Komplexität die größte Differenz zwischen empirischem und wahrem Risiko in der gesamten Klasse beschränkt. Dazu unterscheiden wir zunächst zwei Funktionenklassen, denn Vorhersagen und Verluste gehören zu verschiedenen Klassen. Die *Prädiktorklasse* $\mathcal F$ enthält die Modelle, die wir anpassen. Die Größen, deren Mittelwerte wir vergleichen, sind jedoch die Verluste. Daher definieren wir die *Verlustklasse*

<!--en:d38e7f64-->
mit Funktionswerten in $[0,1]$. Schreiben wir $z=(\mathbf x,y)$ für ein Beispiel mit Label und setzen wir $R(g)=\mathbb{E}[g(z)]$ und $\hat R_S(g)=\frac1n\sum_ig(z_i)$ für $g\in\mathcal G$. Das Risiko eines Prädiktors ist das Risiko der zugehörigen Verlustfunktion.

<!--en:6262cf92-->
**Satz (gleichmäßige Konvergenz über die Rademacher-Komplexität)** :cite:`Bartlett.Mendelson.2002`. *Sei $S=(z_1,\ldots,z_n)$ eine unabhängig und identisch verteilte Stichprobe und $\mathcal G$ eine Klasse von Funktionen mit Werten in $[0,1]$. Dann gilt mit Wahrscheinlichkeit mindestens $1-\delta$:*

<!--en:e1bcacc5-->
Diese Schranke hat dieselben drei Bestandteile wie die Schranke für endliche Klassen: das empirische Risiko auf den Trainingsdaten, einen Komplexitätsterm für die Vielfalt der Klasse und einen Konfidenzterm, der selbst bei einer einzigen festen Funktion aufträte. Der Komplexitätsterm ist hier das Doppelte des Maßes dafür, wie gut die Klasse zufällige Vorzeichen nachbilden kann.

<!--en:dc1ff9dc-->
**Beweis.** Zunächst beschränken wir die erwartete maximale Differenz $\mathbb{E}_S\bigl[\sup_g(R(g)-\hat R_S(g))\bigr]\le2\mathfrak R_n(\mathcal G)$ in drei Schritten. Die Aussage mit hoher Wahrscheinlichkeit folgt dann aus der Konzentration.

<!--en:50bc79b0-->
I. *Geisterstichprobe* (engl. ghost sample). Wir führen eine unabhängige Kopie $S'=(z'_1,\ldots,z'_n)$ der Stichprobe $S$ ein. Sie dient allein als Hilfsmittel im Beweis. Es gilt $R(g)=\mathbb{E}_{S'}[\hat R_{S'}(g)]$, und das Supremum eines Erwartungswerts ist höchstens so groß wie der Erwartungswert des Supremums. Daher erhalten wir

<!--en:e9708118-->
An die Stelle des unbekannten Populationsrisikos tritt das empirische Risiko auf einer neuen Stichprobe.

<!--en:6e399da0-->
II. *Vorzeichenwechsel.* Vertauschen wir $z_i$ mit $z'_i$, so bleibt die gemeinsame Verteilung von $(S,S')$ unverändert, weil beide Punkte unabhängige Ziehungen aus derselben Verteilung sind. Dabei wechselt jedoch die $i$-te Differenz $g(z'_i)-g(z_i)$ für alle $g\in\mathcal G$ zugleich ihr Vorzeichen. Bei jeder festen Wahl der Vorzeichen $\varepsilon_1,\ldots,\varepsilon_n\in\{-1,+1\}$ können wir folglich die $i$-te Differenz mit $\varepsilon_i$ multiplizieren, ohne die rechte Seite von :eqref:`eq_mdl-symmetrization` zu ändern. Auch beim Mitteln über zufällige Vorzeichen bleibt sie unverändert:

<!--en:41b59aef-->
III. *Zerlegung.* Das Supremum einer Summe ist höchstens so groß wie die Summe der Suprema. Die rechte Seite ist also nach oben beschränkt durch

<!--en:b18ef4ec-->
denn $-\boldsymbol\varepsilon$ hat dieselbe Verteilung wie $\boldsymbol\varepsilon$. Beide Erwartungswerte sind daher gleich $\mathfrak R_n(\mathcal G)$. Aus dieser Zerlegung stammt der Faktor $2$.

<!--en:c06c8e6e-->
Für den Übergang vom Erwartungswert zu einer Schranke mit hoher Wahrscheinlichkeit definieren wir $\Phi(S)=\sup_g(R(g)-\hat R_S(g))$. Ersetzen wir einen einzelnen Stichprobenpunkt $z_i$, so ändert sich jedes empirische Risiko $\hat R_S(g)$ um höchstens $1/n$ (da $g$ Werte in $[0,1]$ annimmt). Damit ändert sich auch das Supremum $\Phi(S)$ um höchstens $1/n$. Die *McDiarmid-Ungleichung* :cite:`McDiarmid.1989` verallgemeinert die Hoeffding-Ungleichung von Summen auf Funktionen, deren Wert sich durch eine einzelne Eingabe nur wenig ändern kann. Sie liefert $P(\Phi-\mathbb{E}\Phi\ge t)\le e^{-2nt^2}$ (Übung :exref:`ex_mdl-concentration-generalization-mcdiarmid-s-inequality`). Setzen wir die rechte Seite gleich $\delta$ und verwenden $\mathbb{E}\Phi\le2\mathfrak R_n(\mathcal G)$, so folgt :eqref:`eq_mdl-rademacher-bound`. $\blacksquare$

<!--en:b40be63d-->
Die Idee hinter dieser *Symmetrisierung* ist einfach. Die Geisterstichprobe vertritt die Grundgesamtheit. Da die beobachteten Punkte und die der Geisterstichprobe austauschbar sind, ändert das Vertauschen eines beliebigen Paars nichts an der Verteilung. Entscheiden wir über jedes Vertauschen zufällig, bleiben nur Münzwürfe übrig. Wie groß die Differenz zwischen Trainings- und Populationsrisiko schlimmstenfalls ausfällt, hängt davon ab, wie gut die Klasse zufälligen Vorzeichen folgen kann.

<!--en:48dc4a70-->
Für die Praxis bleibt ein Detail. Im Satz steht die erwartete Komplexität $\mathfrak R_n(\mathcal G)$. Auf einem realen Datensatz können wir nur die empirische Komplexität $\widehat{\mathfrak R}_S(\mathcal G)$ der vorliegenden Stichprobe auswerten. Ändern wir einen Stichprobenpunkt, verschiebt sich auch $\widehat{\mathfrak R}_S(\mathcal G)$ um höchstens $1/n$. Eine zweite Anwendung der McDiarmid-Ungleichung erlaubt uns daher, $\widehat{\mathfrak R}_S(\mathcal G)$ anstelle von $\mathfrak R_n(\mathcal G)$ in :eqref:`eq_mdl-rademacher-bound` einzusetzen. Dafür wächst der Konfidenzterm auf $3\sqrt{\log(2/\delta)/(2n)}$. In den folgenden numerischen Überprüfungen verwenden wir diese empirische Form.

<!--en:0615e51f-->
Betrachten wir die Extremfälle des Satzes. Für eine einelementige Klasse gilt $\mathfrak R_n=0$, und :eqref:`eq_mdl-rademacher-bound` reduziert sich auf die einseitige Hoeffding-Ungleichung. Für eine endliche Klasse ergibt das Lemma von Massart wieder die Form von :eqref:`eq_mdl-finite-class`, wenn auch mit ungünstigeren Konstanten. Für 2000 Konfigurationen auf 10 000 Beispielen beträgt allein der Komplexitätsterm $2\sqrt{2\log 2000/n}=0.078$, gegenüber der gesamten Halbbreite $0.024$ aus der booleschen Ungleichung. Die Rademacher-Komplexität kostet bei endlichen Klassen einen konstanten Faktor, erfasst dafür aber auch unendliche Klassen. Für binäre Klassifikatoren können wir die auf $n$ Punkten realisierbaren Labelzuweisungen zählen. Damit lässt sich $\mathfrak R_n$ durch ein konstantes Vielfaches von $\sqrt{d_{\mathrm{VC}}\log n/n}$ beschränken :cite:`boucheron2005theory`. Hier ist $d_{\mathrm{VC}}$ die VC-Dimension aus dem Haupttext, also die größte Anzahl von Punkten, auf denen die Klasse jede mögliche Labelzuweisung realisieren kann.

<!--en:c8ad57b7-->
Am anderen Ende des Spektrums gilt für jede Verlustklasse mit Werten in $[0,1]$ die Schranke $\mathfrak R_n(\mathcal G)\le\tfrac12$. Bei Gleichheit beträgt der Komplexitätsterm $2\mathfrak R_n(\mathcal G)$ genau $1$. Die Schranke lässt dann jedes Risiko bis 100% zu und sagt nichts aus. Bei binären Klassifikatoren mit Werten in $\{-1,+1\}$ und dem 0-1-Verlust $g(\mathbf x,y)=(1-yf(\mathbf x))/2$ unterscheiden sich die Komplexitäten von Verlustklasse und Prädiktorklasse um genau den Faktor zwei. Ausmultiplizieren der Korrelation liefert $\sum_i\varepsilon_ig(z_i)=\tfrac12\sum_i\varepsilon_i+\tfrac12\sum_i(-\varepsilon_iy_i)f(\mathbf x_i)$. Die erste Summe hat Erwartungswert null und hängt nicht von $f$ ab. Bei festen Labels $y_i\in\{-1,+1\}$ sind die Produkte $-\varepsilon_iy_i$ wieder unabhängige, gleich wahrscheinliche Zufallsvorzeichen. Bilden wir das Supremum und den Erwartungswert, ergibt sich

<!--en:67c564ee-->
Kann eine binäre Prädiktorklasse jedes Vorzeichenmuster auf der Stichprobe nachbilden, so gilt also $\widehat{\mathfrak R}_S(\mathcal F)=1$. Die doppelte Komplexität der Verlustklasse beträgt $1$, und der Satz zur gleichmäßigen Konvergenz garantiert keine bessere Fehlerschranke als 100%.

<!--en:5946db08-->
### Die lineare Klasse

<!--en:9430ba2d-->
Ein Generalisierungssatz hilft nur, wenn wir seinen Komplexitätsterm für die Klassen auswerten können, die uns interessieren. Bei linearen Prädiktoren mit beschränkter Gewichtsnorm genügen vier Zeilen. Überraschenderweise ist das Ergebnis unabhängig von der Eingabedimension.

<!--en:6afe7784-->
**Satz (Rademacher-Komplexität einer normbeschränkten linearen Klasse).** *Sei $\mathcal F=\{\mathbf x\mapsto\langle\mathbf w,\mathbf x\rangle :
\|\mathbf w\|_2\le B\}$, und für jede Eingabe der Stichprobe gelte $\|\mathbf x_i\|_2\le r$. Dann gilt*

<!--en:80a2f5f6-->
**Beweis.** Das innere Supremum hat eine geschlossene Form. Aus der Linearität des inneren Produkts und der Cauchy-Schwarz-Ungleichung folgt, dass der Gewichtsvektor $\mathbf w$ der Länge $B$ in Richtung der vorzeichenbehafteten Summe das Maximum annimmt:

<!--en:179161aa-->
Bilden wir den Erwartungswert über $\boldsymbol\varepsilon$, erhalten wir die Gleichheit im Satz. Für die obere Schranke nutzen wir die Konkavität der Quadratwurzel ($\mathbb{E}\|\cdot\|\le\sqrt{\mathbb{E}\|\cdot\|^2}$) und multiplizieren die quadrierte Norm aus:

<!--en:034cc58d-->
Alle gemischten Terme verschwinden, weil für unabhängige faire Vorzeichen $\mathbb{E}[\varepsilon_i\varepsilon_j]=0$ gilt, sobald $i\ne j$ ist. Multiplikation mit $B/n$ liefert $Br/\sqrt n$. $\blacksquare$

<!--en:213392f6-->
Teil (a) von :numref:`fig_mdl-gen-rademacher` zeigt den ersten Beweisschritt. Bei jedem Vorzeichenmuster zeigt der optimale Gewichtsvektor in Richtung der vorzeichenbehafteten Summe der Eingaben.

<!--en:5c54286f-->
Nur in einem Schritt des Beweises verlieren wir an Schärfe. Da die Identität $\mathbb{E}\|\sum_i\varepsilon_i\mathbf x_i\|^2=\sum_i\|\mathbf x_i\|^2$ exakt gilt, entsteht die gesamte Differenz beim Ersetzen der mittleren Länge der vorzeichenbehafteten Summe durch das quadratische Mittel ihrer Länge. Die Ungleichung ist nahezu scharf, wenn die Länge über die Vorzeichenmuster hinweg wenig schwankt. Für die acht Punkte in der Abbildung beträgt der Abstand 11% ($0.316$ gegenüber $0.354$). In zwei Dimensionen verhält sich die vorzeichenbehaftete Summe vieler gut verteilter Einheitsvektoren wie ein zweidimensionaler Gauß-Vektor. Seine mittlere Länge beträgt das $\sqrt\pi/2\approx0.89$-Fache ihres quadratischen Mittels. Der Abstand bleibt daher unabhängig von $n$ bei etwa 11%. In zwanzig Dimensionen konzentriert sich die Länge der Summe hingegen eng um ihr quadratisches Mittel. Wegen des Dünnschalenphänomens aus :numref:`subsec_mdl-thin-shell` sinkt der Abstand auf etwa 1%.

<!--en:97a7a577-->
In :eqref:`eq_mdl-linear-rademacher` fehlt die Eingabedimension $d$. Sie geht nur indirekt über $B$ und $r$ ein. In dieser Schranke bestimmt das Produkt aus Gewichtsnorm und Eingaberadius die Kapazität eines linearen Modells, ob es mit zehn Merkmalen oder mit einer Million arbeitet. Das begründet den Gewichtszerfall (:numref:`sec_weight_decay`) theoretisch: Halten wir $\|\mathbf w\|$ klein, verkleinern wir die Funktionenklasse, die das trainierte Modell enthält. Diese Dimensionsunabhängigkeit gilt auch für andere Modelle. Bei einem zweischichtigen ReLU-Netz sei die $\ell_1$-Norm der Ausgabegewichte höchstens $B_1$ und die $\ell_2$-Norm der Gewichtsvektoren der versteckten Schicht höchstens $B_2$. Dann beträgt die Rademacher-Komplexität unabhängig von der Breite des Netzes höchstens $2B_1B_2r/\sqrt n$ (Übung :exref:`ex_mdl-generalization-bounds-two-layer-norm-bound`).

<!--en:ec80e8a3-->
Hier liegt ein Einwand nahe. Im Satz muss der Radius $B$ *vor* der Beobachtung der Daten feststehen. Die Norm eines trainierten Gewichtsvektors kennen wir aber erst *nach* dem Training. Diesen berechtigten Einwand entkräften wir mit der booleschen Ungleichung über eine Folge von Radien. Wir wenden den Satz auf jeden Radius $B_j=2^j$ mit Ausfallwahrscheinlichkeit $\delta_j=\delta/(j(j+1))$ an (diese Wahrscheinlichkeiten summieren sich zu $\delta$). Die Schranke werten wir beim kleinsten Radius $B_j$ aus, dessen Kugel den angepassten Gewichtsvektor enthält. Für $\|\mathbf w\|>1$ kostet das höchstens einen Faktor $2$ im Komplexitätsterm und einen kleinen Zuschlag der Größenordnung $\sqrt{\log\log\|\mathbf w\|/n}$ (Übung :exref:`ex_mdl-concentration-generalization-margin-bound`).

<!--en:6ea4c509-->
Ein Schritt fehlt noch, um aus :eqref:`eq_mdl-linear-rademacher` eine Generalisierungsschranke für lineare Klassifikatoren zu gewinnen. Der Satz zur gleichmäßigen Konvergenz braucht die Komplexität der *Verlustklasse* $\mathcal G$. Unser Satz beschränkt dagegen die Komplexität der *Prädiktorklasse* $\mathcal F$. Eine Lipschitz-stetige Verlustfunktion verbindet die beiden.

<!--en:35cd9970-->
**Lemma (Kontraktion)** :cite:`Boucheron.Lugosi.Massart.2013`. *Die Funktionen $\phi_1,\ldots,\phi_n:\mathbb{R}\to\mathbb{R}$ seien jeweils $L$-Lipschitz-stetig. Dann gilt für jede Funktionenklasse $\mathcal F$ und beliebige Punkte $\mathbf x_1,\ldots,\mathbf x_n$:*

<!--en:41a6b654-->
Verketten wir die Funktionen einer Klasse mit Abbildungen, die Abstände höchstens um den Faktor $L$ strecken, so wächst ihre Rademacher-Komplexität höchstens um den Faktor $L$. Dass jedes $\phi_i$ verschieden sein darf, ist gerade für den Verlust beim überwachten Lernen wichtig. Am Beispiel $i$ ist $\phi_i$ die Abbildung $u\mapsto\ell(u,y_i)$ mit festem Label $y_i$. Eine Zentrierung ist unnötig. Addieren wir eine Konstante $c$ zu $\phi_i$, so verschiebt sich die Summe um $c\,\varepsilon_i$, und dieser Term hat Erwartungswert null. Das Kontraktionslemma geben wir ohne Beweis an.

<!--en:802b4f9f-->
Der Satz zur gleichmäßigen Konvergenz verlangt außerdem, dass der Verlust Werte in $[0,1]$ annimmt. Übliche Trainingsverluste wie der Hinge-Verlust und der logistische Verlust sind unbeschränkt. Auf unserer normbeschränkten linearen Klasse können sie ungefähr $1+Br$ erreichen. Für eine direkte Anwendung des Satzes müssten wir also durch dieses Maximum teilen und den skalierten Verlust beschränken. Für die Analyse ist ein von vornherein beschränkter Verlust einfacher: der *Rampenverlust* (engl. ramp loss) für die Klassifikationsmargin $u=yf(\mathbf x)$,

<!--en:699da616-->
Bei einem Fehler beträgt der Rampenverlust $1$ ($u\le0$). Er fällt auf $0$, sobald eine Vorhersage mit mindestens der Margin $\gamma$ richtig ist ($u\ge\gamma$). Dazwischen interpoliert er linear. Er beschränkt den 0-1-Verlust überall nach oben und ist $(1/\gamma)$-Lipschitz-stetig, wie :numref:`fig_mdl-gen-ramp-margin` zeigt.

<!--en:3a4471fa-->
![Rampenverlust $\phi_\gamma$ (blau) als Funktion der Margin $u=yf(\mathbf{x})$ zusammen mit dem 0-1-Verlust $\mathbf{1}\{u\le0\}$ (gestrichelt). Beide stimmen außer für $0<u<\gamma$ (schattiert) überein. Dort ist die Vorhersage richtig, erreicht aber die geforderte Margin nicht, und die Rampe fällt mit der Steigung $-1/\gamma$.](../img/mdl-prob-gen-ramp-margin.svg)

<!--en:b17f9931-->
Schreiben wir $\hat R_\gamma(\mathbf w)=\frac1n\sum_i\phi_\gamma(y_i\langle\mathbf w,\mathbf x_i\rangle)$ für den empirischen Rampenverlust auf der Stichprobe. Erfüllt jede Eingabe $\|\mathbf x\|\le r$, so beschränkt das Kontraktionslemma zusammen mit dem Satz für lineare Klassen die Komplexität der Verlustklasse durch $Br/(\gamma\sqrt n)$. Einsetzen in :eqref:`eq_mdl-rademacher-bound` liefert eine *Margin-Schranke*: Mit Wahrscheinlichkeit mindestens $1-\delta$ gilt gleichzeitig für alle $\mathbf w$ mit $\|\mathbf w\|\le B$:

<!--en:a1badb83-->
Links steht die wahre Fehlerrate der Klassifikation, rechts der Rampenverlust auf den Trainingsdaten plus zwei berechenbare Terme. Um die Größenordnung zu sehen, wählen wir beispielhaft $Br/\gamma=10$, $n=50{,}000$ und $\delta=0.05$ (das Verhältnis geben wir vor, statt es auf einem konkreten Datensatz zu messen). Der Komplexitätsterm beträgt $0.089$, der Konfidenzterm $0.005$, zusammen rund $0.095$. Legen wir $B=10\gamma/r$ vorab fest, gilt also für jeden linearen Klassifikator der Klasse, der seine Trainingsdaten mit Margin $\gamma$ trennt: Seine wahre Fehlerrate liegt mit 95% Konfidenz unter $9.5\%$. Zwingen wir hingegen ein lineares Modell, zufällige Labels mit Margin $\gamma$ nachzubilden, so wächst $Br/\gamma$. Oberhalb von $Br/\gamma\approx10.6$ überschreitet die Summe der Zuschläge $0.1$. Bei etwa $Br/\gamma\approx110$ erreicht allein der Komplexitätsterm $1$, und die Schranke wird aussagelos.

<!--en:6129f6ce-->
### Rademacher-Komplexität in Zahlen

<!--en:4b2cb5ac-->
Da :eqref:`eq_mdl-linear-rademacher` die empirische Rademacher-Komplexität einer linearen Klasse als Erwartungswert einer euklidischen Norm ausdrückt, können wir sie direkt schätzen, indem wir über zufällige Vorzeichenvektoren mitteln. Das tun wir für 50 Punkte mit Norm eins in $d=2$, $20$ und $1{,}000$ Dimensionen. Nach unserer Analyse sollte der Schätzwert in zwei Dimensionen rund 11% unter der Schranke $Br/\sqrt n=1/\sqrt{50}$ liegen. In zwanzig Dimensionen sollte er nur noch etwa 1% von ihr abweichen.

<!--en:90c584b1-->
Die Monte-Carlo-Schätzung ergibt etwa $0.125$ in der Ebene und Werte zwischen $0.139$ und $0.141$ in 20 und 1000 Dimensionen. Wie vorhergesagt liegen Letztere unmittelbar an der Schranke $0.141$. Erhöhen wir die Dimension von 20 auf 1000, also um den Faktor fünfzig, bleibt die Komplexität unverändert.

<!--en:6e331220-->
Am anderen Ende des Spektrums steht eine Klasse, die *interpolieren* kann. Sie kann jede Labelzuweisung auf der Trainingsstichprobe exakt wiedergeben. Betrachten wir die Methode der kleinsten Quadrate für $p=2n$ standardnormalverteilte Merkmale. Die Merkmalsmatrix $\boldsymbol\Phi\in\mathbb{R}^{n\times p}$ enthält die Merkmalsvektoren der $n$ Punkte als Zeilen. Mit der Pseudoinversen aus :numref:`subsec_mdl-pseudoinverse` liefert die Lösung $\mathbf w=\boldsymbol\Phi^+\boldsymbol\varepsilon$ für jeden Vorzeichenvektor $\boldsymbol\varepsilon$ den Interpolanten kleinster Norm. Nach Konstruktion erreicht sie für jedes Muster die Korrelation $1$. Unser Satz zeigt, wie groß eine Kugel im Gewichtsraum sein muss, um alle diese Lösungen zu enthalten. Für jede Klasse, die jedes Vorzeichenmuster interpoliert, gilt $\widehat{\mathfrak R}_S\ge1$. Andererseits liefert :eqref:`eq_mdl-linear-rademacher` die obere Schranke $Br/\sqrt n$. Für jede Kugel, die alle Interpolanten enthält, muss also $Br/\sqrt n\ge1$ gelten. Wie nahe kommen tatsächliche Lösungen dieser Untergrenze? Wir passen Modelle an 2000 zufällige Vorzeichenvektoren an und vergleichen ihre Gewichtsnormen mit der theoretischen Vorhersage $\sqrt{n/(p-n-1)}$ (Herleitung in Übung :exref:`ex_mdl-concentration-generalization-fitting-random-signs`). Dann berechnen wir $Br/\sqrt n$ für die kleinste Kugel, die sie alle enthält.

<!--en:3bee8755-->
Jede Lösung hat eine Gewichtsnorm nahe $1$, und die Zeilen von $\boldsymbol\Phi$ haben im Mittel eine Länge nahe $\sqrt p=10$. Für eine typische Lösung gilt also $\|\mathbf w\|r/\sqrt n\approx1.4$. Über 2000 Vorzeichenmuster liegt die größte Gewichtsnorm zwischen $1.3$ und $1.8$ und die Länge der längsten Zeile zwischen $11$ und $12.5$. Für die Kugel, die alle 2000 Lösungen enthält, liegt $Br/\sqrt n$ somit zwischen $2$ und $3$. Zur Interpolation reinen Rauschens reicht also eine Norm vom 1.5- bis 3-Fachen der theoretischen Untergrenze. Die Schranke $Br/\sqrt n$ für lineare Klassen bleibt nahezu scharf, selbst dort, wo sie keine Aussage über die Generalisierung mehr erlaubt.

<!--en:7b58756d-->
Dieselbe Rechnung gilt unmittelbar für neuronale Netze. Betrachten wir alle Klassifikatoren, die eine gegebene Architektur darstellen kann. Findet das Training auf zufälligen Vorzeichen $\boldsymbol\varepsilon$ eine Funktion, die diese an fast jedem Punkt trifft, liegt das Supremum in :eqref:`eq_mdl-rademacher` für dieses Vorzeichenmuster nahe $1$. Geschieht das für typische Vektoren $\boldsymbol\varepsilon$, so liegt die Rademacher-Komplexität selbst nahe $1$. In :numref:`subsec_random_labels` bildete ein MLP zufällige Labels auf seinen Trainingsbildern fehlerfrei nach, wie in den Experimenten zur Bildklassifikation von :citet:`zhang2021understanding`. (Diese Experimente prüften einige Labelzuweisungen mit zehn Klassen. Sie liefern empirische Belege, aber keinen formalen Beweis.) Die doppelte Komplexität der Verlustklasse liegt dann nahe $1$, und :eqref:`eq_mdl-rademacher-bound` lässt jeden Testfehler zu. Dasselbe Netz generalisiert aber gut, wenn wir es mit echten Labels trainieren. Was seinen Testfehler bestimmt, entgeht also jeder Schranke für den ungünstigsten Fall über die gesamte Funktionenklasse. Der Testfehler hängt von den Daten, dem Optimierer und der konkreten Lösung ab, die das Training erreicht. Um das zu verstehen, betrachten wir nun ein Modell, dessen gelernte Lösung wir in geschlossener Form untersuchen können.

<!--en:205cc8f0-->
## Interpolation und doppelter Abstieg

<!--en:9b28a140-->
In :numref:`subsec_double_descent` variierten wir die Breite $h$ eines Netzes, dessen versteckte Schicht nach der zufälligen Initialisierung unverändert blieb. Bei $h=n$ schoss der Testfehler in die Höhe. Genau dort wurde erstmals Interpolation möglich, und der kleinste Singulärwert der versteckten Darstellung brach ein. Als $h$ weiter wuchs, sank der Testfehler wieder. Wir wiederholen dieses Experiment mit synthetischen Daten, für die wir jeden Teil des Mechanismus explizit berechnen können. Dabei ergänzen wir den Haupttext um zwei Ergebnisse: eine geschlossene Formel für die angepassten Gewichte anhand der Singulärwerte und eine Zerlegung des Testfehlers in die Beiträge der wahren Zielfunktion und des Labelrauschens. Die Merkmale sind $\boldsymbol\phi(\mathbf x)=\mathrm{ReLU}(\mathbf V\mathbf x)$ mit einer festen Zufallsmatrix $\mathbf V$. Dies heißt Modell mit *zufälligen Merkmalen* :cite:`Rahimi.Recht.2007`. Wir passen nur den linearen Ausgabekopf $\mathbf w$ an. Bei der ersten Lektüre können Sie diesen Teil überspringen, ohne den Faden des Kapitels zu verlieren.

<!--en:40c58eab-->
### Der Interpolant kleinster Norm

<!--en:311f313e-->
Um das auffällige Verhalten an der Interpolationsschwelle $p=n$ zu verstehen, schreiben wir die Lösung in geschlossener Form. Bei $p$ Merkmalen und $n$ Trainingspunkten bilden die Merkmalsvektoren der Trainingsdaten die Zeilen von $\boldsymbol\Phi\in\mathbb{R}^{n\times p}$. Wir bestimmen die Lösung $\mathbf w=\boldsymbol\Phi^+\mathbf y$. Unterhalb von $p=n$ ist dies die gewöhnliche Kleinste-Quadrate-Lösung, oberhalb von $p=n$ der Interpolant kleinster euklidischer Norm.

<!--en:47cc6269-->
Unter den vielen Interpolanten zeichnet sich der Interpolant kleinster Norm dadurch aus, dass das Training genau ihn findet. Der Gradient des quadratischen Verlusts, $\boldsymbol\Phi^\top(\boldsymbol\Phi\mathbf w-\mathbf y)$, ist eine Linearkombination der Merkmalsvektoren der Trainingsdaten. Ein bei $\mathbf w=\mathbf 0$ gestarteter Gradientenabstieg verlässt daher nie den Zeilenraum von $\boldsymbol\Phi$. Der einzige Interpolant in diesem Zeilenraum ist $\boldsymbol\Phi^+\mathbf y$. Der Gradientenabstieg konvergiert auch ohne Strafterm für die Norm gegen diesen Interpolanten. Bevorzugt ein Optimierer unter vielen gleich gut an die Daten angepassten Lösungen eine bestimmte, so heißt diese Präferenz sein *impliziter Bias*.

<!--en:349521ab-->
Die Singulärwertzerlegung (engl. singular value decomposition, SVD) erklärt das Verhalten unterhalb, an und oberhalb der Schwelle. Die Matrix $\boldsymbol\Phi$ habe vollen Rang $q=\min(n,p)$ und die Singulärwertzerlegung $\boldsymbol\Phi=\sum_{i=1}^q\sigma_i\mathbf u_i\mathbf v_i^\top$ mit $\sigma_1\ge\cdots\ge\sigma_q>0$. Dann lautet die Lösung mit der Pseudoinversen

<!--en:61f45455-->
Jede Komponente von $\mathbf y$ entlang eines linken Singulärvektors $\mathbf u_i$ wird durch seinen Singulärwert $\sigma_i$ geteilt. Der kleinste Singulärwert $\sigma_{\min}=\sigma_q$ bewirkt daher die stärkste Verstärkung. Die Komponente von $\mathbf y$ entlang $\mathbf u_q$ geht in $\mathbf w$ mit dem Verstärkungsfaktor $1/\sigma_{\min}$ ein. Das verstärkt sowohl das Labelrauschen als auch den Teil der wahren Zielfunktion, den die Merkmale schlecht approximieren.

<!--en:31217cda-->
Für $p<n$ projizieren die angepassten Vorhersagen $\boldsymbol\Phi\mathbf w$ den Vektor $\mathbf y$ auf den $p$-dimensionalen Spann der $\mathbf u_i$. Der Rest bleibt als Trainingsresiduum übrig. Bei $p=n$ spannen diese $n$ Richtungen hingegen ganz $\mathbb{R}^n$ auf. Die Anpassung ist exakt, und kein Residuum kann die schlecht dargestellte Komponente mehr aufnehmen. Eine quadratische Zufallsmatrix aus ReLU-Merkmalen ist fast sicher invertierbar, aber selten *gut* konditioniert. Ihr kleinster Singulärwert ist typischerweise winzig. Wie die Analyse der Konditionszahl in :numref:`subsec_mdl-condition-number` zeigt, erstreckt sich die Lösung dann sehr weit in Richtung $\mathbf v_q$. Fügen wir jenseits von $p=n$ Merkmale hinzu, hat die Matrix mehr Spielraum, um $\mathbb{R}^n$ aufzuspannen, und $\sigma_{\min}$ wächst wieder.

<!--en:3f81429f-->
Oberhalb der Schwelle kann die Norm der Lösung durch zusätzliche Merkmale nur sinken oder gleich bleiben, sofern die Merkmalsmengen geschachtelt sind (das Modell mit $(p+1)$ Merkmalen behält alle $p$ bisherigen Merkmale und fügt eines hinzu). Das folgt aus dem Argument der Nullergänzung in :numref:`subsec_double_descent`. Geben wir dem neuen Merkmal Gewicht null, bleibt der Interpolant kleinster Norm für $p$ Merkmale auch für $(p+1)$ Merkmale ein gültiger Interpolant derselben Norm. Die neue kleinste Norm kann also nicht größer sein. Diese Monotonie setzt exakte Interpolation voraus. Unterhalb von $p=n$ interpoliert die Kleinste-Quadrate-Lösung nicht, und ihre Norm *wächst* sogar mit $p$. Im folgenden Experiment steigt sie von rund $0.5$ bei $p=2$ auf fast $2$ bei $p=15$.

<!--en:c33a0f68-->
Unsere Schranke für lineare Klassen legt nahe, beim Variieren der Breite das *Kapazitätsprodukt* $\|\mathbf w\|\cdot\max_i\|\boldsymbol\phi(\mathbf x_i)\|/\sqrt n$ zu beobachten. Es entspricht $Br/\sqrt n$ für die kleinste Kugel, die die angepassten Gewichte enthält. Wir nutzen es nur zur Diagnose. Eine formale Schranke liefert es hier nicht, weil wir den quadratischen Verlust für unbeschränkte Regressionsziele statt eines beschränkten Klassifikationsverlusts verwenden. Jenseits der Schwelle wächst der Merkmalsradius wie $\sqrt p$, während $\|\mathbf w\|$ sinkt. Wie sich ihr Produkt verhält, ist daher vorab unklar.

<!--en:2e47432d-->
### Ein Experiment zum doppelten Abstieg

<!--en:e42d6931-->
Beobachten wir das Zusammenspiel dieser Größen an einem konkreten Problem. Wir ziehen $n=40$ Trainingspunkte nach einem linearen Lehrermodell $y=\langle\boldsymbol\beta,\mathbf x\rangle$ in $d=15$ Dimensionen mit $\|\boldsymbol\beta\|=1$ und gaußschem Labelrauschen der Standardabweichung $0.1$. Wir konstruieren geschachtelte Mengen zufälliger ReLU-Merkmale und passen den linearen Ausgabekopf mit `pinv` an. Unterhalb von $p=40$ ergibt das die Kleinste-Quadrate-Lösung, oberhalb den Interpolanten kleinster Norm. Wir variieren $p$ von $2$ bis $400$ und berücksichtigen auch $p=39$, $40$ und $41$, um die Spitze an der Schwelle genau zu erfassen. Für jedes $p$ protokollieren wir den mittleren quadratischen Fehler (engl. mean squared error, MSE) auf Test- und Trainingsdaten, $\|\mathbf w\|$, $\sigma_{\min}$ und das Kapazitätsprodukt. Hinzu kommen die beiden in :numref:`subsec_mdl-benign-overfitting` untersuchten Komponenten des Testfehlers: der *rauschfreie Anteil* (Testfehler bei Anpassung an rauschfreie Labels) und der *Rauschanteil* (zusätzlicher Testfehler durch Labelrauschen). Damit die Kurven gut lesbar bleiben, bilden wir den Median über 20 unabhängige Durchläufe.

<!--en:0fc6fdc4-->
Die folgende Abbildung zeigt die Mediane von Test-MSE, Lösungsnorm und Kapazitätsprodukt auf logarithmischen Achsen.

<!--en:8540aa71-->
Die folgende Tabelle schlüsselt die Ergebnisse für sieben repräsentative Merkmalsanzahlen vollständig auf. Ihre letzten beiden Spalten gehören zu :numref:`subsec_mdl-benign-overfitting`. Die letzte Zeile zeigt als klassischen Vergleichswert die Kleinste-Quadrate-Lösung direkt auf den Roheingaben.

<!--en:72a78b8b-->
Betrachten wir die Tabelle getrennt für drei Bereiche: unterhalb, an und oberhalb der Schwelle.

<!--en:94b8a861-->
*Unterhalb der Schwelle* hat das Modell weniger Parameter als Trainingspunkte und verhält sich klassisch. Sein Testfehler fällt zunächst von rund $1$ bei $p=2$ auf etwa $0.65$ bis $0.85$ nahe $p=15$. Nähert sich $p$ dann dem Wert $40$, steigt er wieder. So ergibt sich die bekannte klassische U-Kurve. Für Vergleiche mit diesem Bereich sind zwei Einschränkungen wichtig. Erstens ist diese klassische Anpassung selbst im besten Fall schwach. Die Zielwerte haben Varianz $1$ (der Fehler bei konstanter Vorhersage null). Selbst am Minimum der U-Kurve erklärt das Modell also höchstens etwa ein Drittel der Varianz. Zweitens gibt es ein weitaus besseres klassisches Modell. Das wahre Lehrermodell ist linear in den 15 Roheingaben. Die gewöhnliche Methode der kleinsten Quadrate auf diesen Eingaben hat daher den erwarteten Testfehler $0.01\cdot15/(40-15-1)\approx0.006$ (Rauschvarianz mal $d/(n-d-1)$ für $d=15$), wie die letzte Ausgabezeile bestätigt. Das Modell mit zufälligen Merkmalen ist durch die Konstruktion im Nachteil. Es muss eine lineare Funktion aus ReLU-Kammfunktionen rekonstruieren, und dafür sind vierzig Punkte viel zu wenig.

<!--en:2a9de28e-->
*An der Schwelle* ändern sich drei Spalten gleichzeitig. Der Trainingsfehler stürzt von einigen Vielfachen von $10^{-3}$ bei $p=38$ auf rund $10^{-28}$. Bis auf Gleitkommagenauigkeit ist die Interpolation exakt. Zugleich sinkt der kleinste Singulärwert auf etwa $0.01$ bis $0.04$ und damit auf einen Bruchteil seines Werts bei $p=38$ oder $p=42$. Da in :eqref:`eq_mdl-svd-interpolant` durch $\sigma_{\min}$ geteilt wird, nimmt die Lösung extreme Werte an. Die Norm $\|\mathbf w\|$ springt auf Werte zwischen $10$ und $30$, der Test-MSE auf Werte zwischen $10$ und $70$, ein bis zwei Größenordnungen über dem Minimum der U-Kurve. Diese Bereiche sind breit, weil die Höhe der Spitze eine Verteilung mit schweren Rändern hat, wie bei einer quadratischen Gauß-Matrix mit $\mathbb{E}[1/\sigma_{\min}^2]=\infty$. Selbst der Median über 20 Versuche schwankt von Durchlauf zu Durchlauf merklich.

<!--en:4046b56c-->
*Oberhalb der Schwelle* gehen alle Größen, die zuvor in die Höhe schossen, wieder zurück. Der kleinste Singulärwert erholt sich auf rund $2.8$ bei $p=400$. Wie das Argument der Nullergänzung garantiert, sinkt $\|\mathbf w\|$ in jedem einzelnen Versuch monoton (also auch in den Medianen) und erreicht rund $0.4$. Der Test-MSE sinkt ebenfalls, auf etwa $0.2$ bei $p=80$ und $0.04$ bis $0.07$ bei $p=400$. Eine Spalte liefert nun keine Information mehr: Der Trainingsfehler bleibt jenseits von $p=40$ überall bis auf Gleitkommagenauigkeit null. Kein Kriterium auf Basis des Trainingsverlusts kann daher $p=42$ von $p=400$ unterscheiden, obwohl ihre Testfehler mehr als eine Größenordnung auseinanderliegen. Die Lösungsnorm, $\sigma_{\min}$ und das Kapazitätsprodukt unterscheiden die beiden deutlich.

<!--en:e1f9c350-->
Was zeigt dieses Experiment tatsächlich? Die auffällige Kurve allein kann einen zu weit reichenden Schluss nahelegen. Innerhalb der geschachtelten Familie zufälliger Merkmale schlägt das breiteste Modell jedes schmalere. Selbst gegenüber dem besten unterparametrisierten Modell beträgt der Vorteil etwa einen Faktor zehn, weit jenseits des Punkts, an dem das Modell das gesamte Labelrauschen auswendig lernt. Das ist ein echtes und überraschendes Phänomen. Dennoch schneidet die gewöhnliche Methode der kleinsten Quadrate auf den Roheingaben nochmals um einen Faktor zehn besser ab. Ob ein interpolierendes Modell ein korrekt spezifiziertes kleines Modell schlagen kann, ist eine andere Frage. Hier lautet die Antwort nein.

<!--en:8fd4a639-->
Das Kapazitätsprodukt erreicht bei $p=40$ Spitzenwerte von etwa 10 bis 35 und fällt wieder auf rund $1.2$ bis $1.3$ bei $p=400$. Es folgt der Form der Testfehlerkurve, liefert aber keine Generalisierungsgarantie. Keiner der Sätze dieses Abschnitts erfasst den unbeschränkten quadratischen Verlust. Das Produkt ändert sich bei Umskalierung der Zielwerte, und $B$ haben wir an der angepassten Lösung abgelesen, statt es vor Beobachtung der Daten festzulegen.

<!--en:206b889c-->
Die Spitze entsteht *nicht* durch das Labelrauschen. Wiederholen wir das Experiment mit `noise = 0`, erhalten wir eine fast ebenso hohe Spitze. Die Tabelle zeigt den Grund: Bei $p=40$ ist der rauschfreie Anteil des Testfehlers bereits größer als der Rauschanteil, typischerweise um ein Mehrfaches. Da sich das lineare Lehrermodell mit den gegebenen ReLU-Merkmalen nicht exakt darstellen lässt, wird der Approximationsfehler entlang $\mathbf u_q$ durch $\sigma_{\min}$ geteilt. Ersetzen wir das Lehrermodell durch eine exakte Linearkombination der ersten zehn Merkmale und entfernen das Rauschen, verschwindet die Spitze ganz (Übung :exref:`ex_mdl-generalization-bounds-square-case`).

<!--en:fdd6b019-->
### Gutartige Überanpassung

<!--en:f1bc5ab2-->
Eine Frage bleibt. Bei $p=400$ interpoliert das Modell jedes verrauschte Trainingslabel ohne Residuum und erzielt dennoch den kleinsten Testfehler im Experiment. Warum richtet die Anpassung an das Rauschen hier so wenig Schaden an?

<!--en:9e2591f8-->
Die Antwort erhalten wir, indem wir den Testfehler in zwei berechenbare Anteile zerlegen. Schreiben wir die verrauschten Trainingslabels als $\mathbf y=\mathbf y_0+\boldsymbol\xi$. Dabei sind $\mathbf y_0$ die rauschfreien Label und $\boldsymbol\xi$ ist unabhängiges Rauschen mit Varianz $\sigma_\xi^2=0.01$ je Beispiel. Wegen der Linearität der Pseudoinversen gilt $\mathbf w=\boldsymbol\Phi^+\mathbf y_0+\boldsymbol\Phi^+\boldsymbol\xi$. Die Testvorhersagen zerfallen also in $\boldsymbol\Phi_{\mathrm{te}}\boldsymbol\Phi^+\mathbf y_0+\boldsymbol\Phi_{\mathrm{te}}\boldsymbol\Phi^+\boldsymbol\xi$. Da $\boldsymbol\xi$ den Erwartungswert null hat, verschwindet der gemischte Term im Erwartungswert. Damit erhalten wir eine exakte Zerlegung des erwarteten Test-MSE über $n_{\mathrm{te}}$ Testpunkte:

<!--en:759513ff-->
Die Spalten `clean` und `noise` unserer Tabelle geben diese beiden Terme an. Bei $p=40$ beträgt der Rauschanteil rund $1$ bis $4$ (mindestens das Hundertfache der Varianz $0.01$ des Labelrauschens), und der rauschfreie Anteil ist noch größer. Bei $p=42$ sinkt der Rauschanteil auf einige Zehntel, bei $p=80$ auf etwa $0.02$ und bei $p=400$ auf knapp $0.01$. Auf derselben Strecke fällt der rauschfreie Anteil von rund 8 bis 60 an der Spitze auf etwa $0.05$ bei $p=400$. Der größte Teil der Verbesserung jenseits der Schwelle geht also darauf zurück, dass das Modell die Zielfunktion besser approximiert.

<!--en:24508430-->
Warum ist die Division durch kleine Singulärwerte, die bei $p=40$ die Spitze erzeugt hat, bei $p=400$ fast harmlos? Nach :eqref:`eq_mdl-svd-interpolant` geht eine Rauschkomponente entlang $\mathbf u_i$ mit dem Faktor $1/\sigma_i$ in die Gewichte entlang $\mathbf v_i$ ein. Sie verschiebt eine Testvorhersage proportional zur Komponente des Testmerkmalsvektors entlang $\mathbf v_i$. Mit $\|\boldsymbol\Phi_{\mathrm{te}}\boldsymbol\Phi^+\|_F^2=\sum_i\|\boldsymbol\Phi_{\mathrm{te}}\mathbf v_i\|^2/\sigma_i^2$ und $\sigma_i^2=\|\boldsymbol\Phi\mathbf v_i\|^2$ erhält der Rauschanteil für $p\ge n$ die aufschlussreiche Form

<!--en:d7d7a973-->
also die Rauschvarianz mal dem mittleren Energieverhältnis. Dazu bilden wir für jede rechte Singulärrichtung $\mathbf v_i$ den Quotienten aus den mittleren quadrierten Merkmalskomponenten der Test- und der Trainingsdaten. Ein winziger Singulärwert $\sigma_i$ verursacht nur dann Probleme, wenn die Testeingaben in derselben Richtung erhebliche Energie tragen. Bei $p=n$ ist ein winziger Wert von $\sigma_{\min}$ eine zufällige Besonderheit der vierzig Trainingspunkte. Entlang $\mathbf v_q$ sind die mittleren quadrierten Testmerkmale typischerweise mehrere tausend Mal größer als die mittleren quadrierten Trainingsmerkmale. Bei $p=400$ hingegen kennzeichnet ein kleiner Wert von $\sigma_i$ eine Richtung, in der sowohl Trainings- als auch Testmerkmale klein sind. Die letzten beiden Tabellenspalten bestätigen das. `ratio_q` (entlang $\mathbf v_q$) beträgt nur etwa $2$, und `ratio` (gemittelt über alle Richtungen) liegt knapp unter $1$. Ein kleiner Zähler gleicht also den kleinen Nenner aus. In diesem Sinne heißt die Überanpassung *gutartig* (engl. benign overfitting). Gutartig bedeutet *beschränkt*. Bei $p=400$ bleibt die Varianz der Vorhersagen ungefähr so groß wie die eines einzelnen verrauschten Trainingslabels.

<!--en:2c3e834c-->
:citet:`Bartlett.Long.Lugosi.Tsigler.2020` präzisieren dieses Phänomen für die lineare Regression anhand zweier effektiver Ränge der Kovarianzmatrix der Eingaben. In Übung :exref:`ex_mdl-concentration-generalization-benign-spectra` berechnen Sie diese direkt. In tiefen neuronalen Netzen werden auch die Merkmale gelernt, und ihr Spektrum ändert sich während des Trainings. Eine Theorie, die aus Grundprinzipien den Testfehler eines trainierten Netzes vorhersagt, ist daher noch außer Reichweite. Jede formale Schranke in diesem Abschnitt erfasst die ungünstigste Funktion einer vorab festgelegten Klasse. In der Praxis setzen wir jedoch die konkrete Lösung ein, die ein Optimierer findet. Wie lässt sich diese Lücke mit Garantien schließen, die vom Algorithmus und von den Daten abhängen und über die Funktionenklasse allein hinausgehen? Das ist die zentrale offene Frage dieses Abschnitts.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ef788a32-->
Eine Konzentrationsschranke für eine einzelne, vor der Datenerhebung festgelegte Funktion erfasst kein Modell, das wir wegen seines besten Abschneidens auf der Stichprobe auswählen. Gleichmäßige Konvergenz schließt diese Lücke, indem sie den Schätzfehler aller Funktionen einer Klasse gleichzeitig beschränkt. Für eine endliche Klasse $\mathcal F$ leisten das die Hoeffding-Ungleichung und die boolesche Ungleichung mit $\log|\mathcal F|/(2t^2)$ zusätzlichen Beispielen bei Toleranz $t$. Für unendliche Klassen führt die Symmetrisierung mit einer Geisterstichprobe die größtmögliche Differenz auf die Korrelation mit zufälligen Vorzeichen zurück. Zum empirischen Risiko kommt die doppelte Rademacher-Komplexität der Verlustklasse hinzu. Bei normbeschränkten linearen Modellen ($\|\mathbf w\|_2\le B$, $\|\mathbf x_i\|_2\le r$) beträgt diese Komplexität unabhängig von der Dimension höchstens $Br/\sqrt n$. Die gleichmäßige Konvergenz stößt bei Klassen an ihre Grenze, die beliebige Vorzeichenmuster interpolieren können, etwa bei tiefen Netzen, die zufällige Labels nachbilden. Hier wird die Schranke aussagelos. Unser Experiment mit zufälligen Merkmalen zeigt, warum Generalisierung dennoch gelingen kann. Der Testfehler schießt dort in die Höhe, wo $\sigma_{\min}$ einbricht, und sinkt, wenn die Norm des Interpolanten kleinster Norm abnimmt. Entscheidend sind hier Eigenschaften der gelernten Lösung innerhalb der Funktionenklasse.

<!--en:2294243a-->
## Übungen

<!--en:a03642eb-->
1. **McDiarmid-Ungleichung.** ● Die Ungleichung von :citet:`McDiarmid.1989` erweitert die Hoeffding-Ungleichung von Summen auf Funktionen mit *beschränkten Differenzen*. Wenn sich nur das $i$-te Argument von $g(x_1,\ldots,x_n)$ ändert, ändere sich der Funktionswert um höchstens $c_i$. Ferner seien $X_1,\ldots,X_n$ unabhängig. Dann gilt

   $$P\bigl(g(X_1,\ldots,X_n)-\mathbb{E}[g(X_1,\ldots,X_n)]\ge t\bigr)\le\exp\left(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\right),$$

   und dieselbe Schranke gilt für Abweichungen nach unten. Verwenden Sie diese Aussage ohne Beweis. Ihr Beweis nutzt Martingale.
    1. Leiten Sie daraus die Hoeffding-Ungleichung für den Mittelwert unabhängiger Terme $X_i\in[a_i,b_i]$ her.
    1. Sei $\Phi(S)=\sup_{g\in\mathcal G}\bigl(R(g)-\hat R_S(g)\bigr)$ für eine Verlustklasse $\mathcal G$ mit Werten in $[0,1]$. Zeigen Sie, dass $\Phi$ beschränkte Differenzen $c_i=1/n$ hat. Kombinieren Sie die McDiarmid-Ungleichung mit der Symmetrisierungsschranke $\mathbb{E}_S[\Phi(S)]\le 2\mathfrak R_n(\mathcal G)$ aus dem Text und leiten Sie :eqref:`eq_mdl-rademacher-bound` her.
    1. Für Daten in $[0,1]$ hat der Stichprobenmittelwert beschränkte Differenzen $1/n$. Konstruieren Sie eine Stichprobe von ungeradem Umfang $n$ in $[0,1]$, bei der die Änderung eines Punkts den Stichprobenmedian um $1$ verschiebt. Was bedeutet das, wenn Sie mit der McDiarmid-Ungleichung einen Fehlerbalken für die Bootstrap-Schätzung des Medians in :numref:`sec_mdl-statistics` angeben wollen?
   :exlabel:`ex_mdl-concentration-generalization-mcdiarmid-s-inequality`

<!--en:89033090-->
1. **Das Lemma von Massart und $\ell_1$-Kugeln.** ● Das Lemma von Massart für endliche Klassen und die Rademacher-Komplexität $\ell_1$-beschränkter linearer Klassen beruhen auf derselben Schranke für den Erwartungswert des Maximums subgaußscher Variablen.
    1. Seien $Z_1,\ldots,Z_N$ zentriert und subgaußsch mit Varianzparameter $\sigma^2$, das heißt, es gelte $\mathbb{E}[e^{\lambda Z_j}]\le e^{\lambda^2\sigma^2/2}$ für alle $\lambda$. Die Variablen müssen nicht unabhängig sein. Zeigen Sie $\mathbb{E}[\max_j Z_j]\le\sigma\sqrt{2\log N}$, indem Sie $e^{\lambda \mathbb{E}[\max_j Z_j]}$ durch $\sum_j \mathbb{E}[e^{\lambda Z_j}]$ beschränken und über $\lambda>0$ optimieren. Leiten Sie daraus das Lemma von Massart, $\widehat{\mathfrak R}_S(\mathcal F)\le\sqrt{2\log|\mathcal F|/n}$, für eine endliche Klasse mit Werten in $[-1,1]$ her.
    1. Sei $\mathcal F=\{\mathbf x\mapsto\langle\mathbf w,\mathbf x\rangle:
       \|\mathbf w\|_1\le B\}$ und es gelte $\|\mathbf x_i\|_\infty\le r$ für alle $i$. Folgen Sie dem Beweis von :eqref:`eq_mdl-linear-rademacher` und verwenden Sie die Dualität $\sup_{\|\mathbf w\|_1\le B}\langle\mathbf w,\mathbf
       v\rangle=B\|\mathbf v\|_\infty$ anstelle der Cauchy-Schwarz-Ungleichung. Zeigen Sie so $\widehat{\mathfrak R}_S(\mathcal F)\le Br\sqrt{2\log(2d)/n}$. Warum geht die Dimension $d$ nun ein, und warum nur logarithmisch?
    1. Die Eingaben mögen in $\{-1,+1\}^d$ liegen, und der Zielgewichtsvektor habe $k$ von null verschiedene Einträge, jeweils gleich $\pm1$. Vergleichen Sie die Schranke aus (b) mit :eqref:`eq_mdl-linear-rademacher`. Wählen Sie jeweils den kleinsten Radius $B$, dessen Kugel den Zielvektor enthält. Für welche $k$ ist die $\ell_1$-Schranke bei $d=10^4$ kleiner?
   :exlabel:`ex_mdl-concentration-generalization-massart-l1`

<!--en:0eb0b0ad-->
1. **Margin-Schranke.** ● Sei $\mathcal F=\{\mathbf x\mapsto\langle\mathbf
   w,\mathbf x\rangle:\|\mathbf w\|_2\le B\}$ mit $\|\mathbf x\|_2\le r$. Die Labels seien $y\in\{-1,+1\}$, und der Rampenverlust $\phi_\gamma(u)=\min\{1,\max\{0,1-u/\gamma\}\}$ mit festem $\gamma>0$ wirke auf die Margin $u=yf(\mathbf x)$.
    1. Zeigen Sie, dass $\phi_\gamma(yf(\mathbf x))$ mindestens so groß wie der 0-1-Verlust $\mathbf 1\{yf(\mathbf x)\le 0\}$ und dass $\phi_\gamma$ $(1/\gamma)$-Lipschitz-stetig ist. Zeigen Sie, dass die Multiplikation jeder Vorhersage $f(\mathbf x_i)$ mit einem festen Label $y_i$ die Komplexität $\widehat{\mathfrak R}_S(\mathcal F)$ unverändert lässt.
    1. Kombinieren Sie (a), das Kontraktionslemma :eqref:`eq_mdl-contraction`, :eqref:`eq_mdl-rademacher-bound` und :eqref:`eq_mdl-linear-rademacher` zu :eqref:`eq_mdl-margin-bound`. Bestimmen Sie für $n=50{,}000$ und $\delta=0.05$ das größte Verhältnis $Br/\gamma$, für das Komplexitäts- und Konfidenzterm zusammen unter $0.1$ bleiben.
    1. Die Schranke gilt für einen Radius $B$, der vor dem Ziehen der Daten feststeht. Die Norm eines trainierten Gewichtsvektors kennen wir erst danach. Wenden Sie die Schranke mit Radius $B_j=2^j$ und Ausfallwahrscheinlichkeit $\delta_j=\delta/(j(j+1))$ für jedes $j\ge1$ an. Nutzen Sie dann die boolesche Ungleichung über $j$. Geben Sie die Schranke für einen Gewichtsvektor mit $\|\mathbf w\|>1$ an, den Sie nach Beobachtung der Daten wählen. Zeigen Sie, dass die freie Wahl des Radius höchstens einen Faktor $2$ im Komplexitätsterm und einen zusätzlichen Term der Ordnung $\sqrt{\log\log\|\mathbf w\|/n}$ kostet.
   :exlabel:`ex_mdl-concentration-generalization-margin-bound`

<!--en:4f5adf8b-->
1. [code] **Anpassung an zufällige Vorzeichen.** ● Die zweite Codezelle in :numref:`subsec_mdl-coin-flips-code` passt Modelle an $2{,}000$ Vorzeichenvektoren $\boldsymbol\varepsilon\in\{-1,+1\}^n$ an. Sie berechnet dazu $\mathbf w=\boldsymbol\Phi^+\boldsymbol\varepsilon$ auf $p=2n$ standardnormalverteilten Merkmalen und misst die Normen der Lösungen.
    1. Es gelte $p<n$, und $\boldsymbol\Phi$ habe vollen Spaltenrang. Zeigen Sie, dass die Korrelation $\frac1n\sum_i\varepsilon_i(\boldsymbol\Phi\mathbf w)_i$ bezüglich der Vorzeichen den Erwartungswert $p/n$ hat. Die Matrix $\boldsymbol\Phi\boldsymbol\Phi^+$ ist eine Projektion.
    1. Es gelte $p>n$, und $r$ sei die größte Zeilennorm von $\boldsymbol\Phi$. Der Radius $B$ sei so groß, dass die Klasse $\{\|\mathbf w\|\le B\}$ für jeden Vorzeichenvektor einen Interpolanten enthält. Erklären Sie, warum $Br/\sqrt n$ dann nicht kleiner als $1$ sein kann. Für eine standardnormalverteilte Zufallsmatrix $\boldsymbol\Phi$ mit $p>n+1$ gilt $\mathbb{E}[(\boldsymbol\Phi\boldsymbol\Phi^\top)^{-1}]=\mathbf I_n/(p-n-1)$. Sagen Sie mit dieser Identität den typischen Wert von $\|\mathbf w\|$ vorher. Bestimmen Sie auch den typischen Wert des Produkts aus $\|\mathbf w\|$ und einer typischen Zeilennorm von $\boldsymbol\Phi$, geteilt durch $\sqrt n$. Geben Sie beide Werte als Funktionen von $p$ und $n$ an.
    1. Führen Sie die Zelle mit $n=50$ für $p\in\{10,25,45,55,100,400,1000\}$ erneut aus (geben Sie für $p<n$ die Korrelation anstelle der Norm aus). Vergleichen Sie die Ergebnisse mit (a) und (b). Wie verhält sich der Kapazitätswert aus (b) für $p\to\infty$, und warum bleibt die Variante mit der größten Zeilennorm über $1$?
   :exlabel:`ex_mdl-concentration-generalization-fitting-random-signs`

<!--en:dbecd714-->
1. [code] **Angriff auf eine Rangliste.** ●● Eine Rangliste meldet für jeden eingereichten Labelvektor den 0-1-Verlust auf einem zurückgehaltenen Datensatz aus $n=4{,}000$ Beispielen mit Labels $\mathbf y\in\{0,1\}^n$. Der *Boosting-Angriff* von :citet:`Blum.Hardt.2015` erfordert keine Kenntnis der Aufgabe. Reichen Sie $k$ gleichverteilt zufällige Labelvektoren ein, behalten Sie diejenigen mit gemeldetem Verlust von höchstens $\tfrac12$ und reichen Sie ihr koordinatenweises Mehrheitsvotum ein. Auf neuen Daten hat das Mehrheitsvotum den Verlust $\tfrac12$.
    1. Implementieren Sie den Angriff für gleichverteilte zufällige $\mathbf y$ und $k\in\{10,30,100,300,1000,3000\}$. Halten Sie den gemeldeten Verlust des Mehrheitsvotums fest. Vergleichen Sie seinen Abstand von $\tfrac12$ mit der Abweichung, die :eqref:`eq_mdl-finite-class` für eine vorab registrierte Menge von $k+1$ Modellen bei $\delta=0.05$ zulässt. Ab welchem $k$ überschreitet der Angriff diese Abweichung, und welche Voraussetzung des Satzes verletzt er?
    1. :citet:`Blum.Hardt.2015` beweisen, dass der gemeldete Verlust $\tfrac12-\Omega(\sqrt{k/n})$ beträgt, geben aber die Konstante nicht an. Leiten Sie diese heuristisch mit Normalapproximationen her. Schätzen Sie zunächst, um wie viel die erwartete Übereinstimmung eines behaltenen Vektors mit $\mathbf y$ über dem Zufallsniveau liegt. Schätzen Sie dann die Wahrscheinlichkeit, dass die Mehrheit von etwa $k/2$ solchen Vektoren bei einer Koordinate richtig liegt. Vergleichen Sie die Konstante mit den Messwerten aus (a).
    1. Der *Ladder*-Mechanismus von :citet:`Blum.Hardt.2015` veröffentlicht einen neuen Score nur, wenn eine Einreichung den besten veröffentlichten Score um mehr als die Schrittweite $\eta$ verbessert. Er rundet ihn auf ein Vielfaches von $\eta$ und wiederholt andernfalls den vorigen Score. Bei diesem Verfahren kann der Angreifer nur die Vektoren behalten, die den veröffentlichten Score gesenkt haben. Implementieren Sie den Mechanismus und den Angriff für $\eta\in\{0.005,0.01\}$ und $k\in\{100,1000,3000\}$. Wie weit unter $\tfrac12$ gelangt das Mehrheitsvotum jetzt? Vergleichen Sie dies mit der Abweichung aus (a).
   :exlabel:`ex_mdl-concentration-generalization-leaderboard-attack`

<!--en:dcb17416-->
1. [code] **Der quadratische Fall.** ● Verwenden Sie die Versuchsreihe zum doppelten Abstieg aus :numref:`subsec_mdl-double-descent-example`.
    1. Zeigen Sie anhand der gespeicherten Normen, dass die Norm unterhalb der Schwelle trotz geschachtelter Merkmalsmengen nicht monoton fällt. Erklären Sie, warum das Argument der Nullergänzung dort nicht greift.
    1. Zerlegen Sie bei $p=n$ den Vektor $\mathbf w$ gemäß :eqref:`eq_mdl-svd-interpolant`. Geben Sie den Koeffizienten $|\langle\mathbf u_n,\mathbf y\rangle|/\sigma_{\min}$ der Richtung des kleinsten Singulärwerts zusammen mit $\|\mathbf w\|$ an, jeweils als Median über die Versuche. Welchen Anteil der Norm trägt diese eine Richtung?
    1. Wiederholen Sie die Versuchsreihe mit `noise = 0`. Ersetzen Sie dann das Lehrermodell durch eine Zielfunktion $y=\langle\mathbf c,\boldsymbol\phi_{1:10}(\mathbf x)\rangle$, die eine exakte Linearkombination der ersten zehn Merkmale mit $\|\mathbf c\|=1$ und ohne Rauschen ist. Beschreiben Sie die beiden Kurven. Welche Größe wird jeweils durch $\sigma_{\min}$ geteilt? Warum wird der Interpolant kleinster Norm mit zusätzlichen Merkmalen *schlechter*, wenn die Zielfunktion nur zehn davon nutzt?
    1. Fügen Sie der Zielfunktion aus (c) Labelrauschen mit Standardabweichung $0.1$ hinzu. Wo ist der Testfehler nun am höchsten, wie groß ist die Spitze im Vergleich zur Rauschvarianz und wo ist der Fehler am kleinsten?
   :exlabel:`ex_mdl-generalization-bounds-square-case`

<!--en:149b7556-->
1. [extended] **Gutartige Spektren.** ●● :citet:`Bartlett.Long.Lugosi.Tsigler.2020` charakterisieren, wann der Interpolant kleinster Norm bei der linearen Regression ein kleines Überschussrisiko hat. Das ist der Mechanismus, den :numref:`subsec_mdl-benign-overfitting` in Worten beschreibt. Sei $\mathbf x\sim\mathcal N(\mathbf 0,\boldsymbol\Sigma)$ mit $\boldsymbol\Sigma=\operatorname{diag}(\lambda_1,\ldots,\lambda_p)$ und $\lambda_1\ge\lambda_2\ge\cdots$. Ferner sei $y=\langle\boldsymbol\theta^*,\mathbf x\rangle+\xi$ mit $\xi\sim\mathcal N(0,\sigma^2)$. Fassen Sie $n<p$ Trainingseingaben in $\mathbf X\in\mathbb R^{n\times p}$ zusammen und setzen Sie $\hat{\boldsymbol\theta}=\mathbf X^+\mathbf y$. Das Überschussrisiko beträgt $(\hat{\boldsymbol\theta}-\boldsymbol\theta^*)^\top\boldsymbol\Sigma
   (\hat{\boldsymbol\theta}-\boldsymbol\theta^*)$.
    1. Zeigen Sie, dass der Erwartungswert bezüglich des Rauschens die Summe aus einem Verzerrungsterm $\boldsymbol\theta^{*\top}(\mathbf I-\mathbf P)\boldsymbol\Sigma
       (\mathbf I-\mathbf P)\boldsymbol\theta^*$ mit $\mathbf P=\mathbf X^+\mathbf X$ und einem Varianzterm $\sigma^2\operatorname{tr}\bigl((\mathbf X^+)^\top\boldsymbol\Sigma
       \mathbf X^+\bigr)$ ist. Implementieren Sie beide Terme. Dazu brauchen Sie keinen Testdatensatz.
    1. Die effektiven Ränge der Arbeit sind $\rho_k=\sum_{i>k}\lambda_i/\lambda_{k+1}$ und $\varrho_k=\bigl(\sum_{i>k}\lambda_i\bigr)^2/\sum_{i>k}\lambda_i^2$. Sei $k^*$ der kleinste Index $k$ mit $\rho_k\ge bn$. Dann ist der Varianzterm bis auf konstante Faktoren von der Ordnung $\sigma^2(k^*/n+n/\varrho_{k^*})$. Setzen Sie $b=1$. Verwenden Sie $n=100$, $\sigma=1$ und fünf starke Richtungen mit $\lambda_i=1$, deren Spann den Einheitsvektor $\boldsymbol\theta^*$ enthält. Nehmen Sie $q$ schwache Richtungen mit jeweils $\lambda_i=1/q$ hinzu, für $q\in\{100,300,10^3,3\cdot10^3,10^4\}$. Vergleichen Sie die gemessene Varianz mit $\sigma^2(k^*/n+n/\varrho_{k^*})$. Was geschieht bei $q=100$ und warum?
    1. Wiederholen Sie das Experiment mit Potenzgesetzspektren $\lambda_i=i^{-\alpha}$, abgeschnitten bei $p=40n$, für $\alpha\in\{0.5,1,2\}$, $\boldsymbol\theta^*=\mathbf e_1$ und $n\in\{50,100,200,400\}$. Bei welchen Spektren sinkt das Überschussrisiko mit $n$? Liegt es bei den übrigen an der Verzerrung oder an der Varianz, und woran erkennen Sie das in $\rho_0/n$, $k^*/n$ und $n/\varrho_{k^*}$?
   :exlabel:`ex_mdl-concentration-generalization-benign-spectra`

<!--en:ff65f71d-->
1. **Normen statt Breiten bei einem zweischichtigen Netz.** ●● Seien $\mathcal F_1=\{\mathbf x\mapsto\mathrm{ReLU}(\langle\mathbf v,\mathbf x\rangle):\|\mathbf v\|_2\le B_2\}$ und $\mathcal F_2=\{\mathbf x\mapsto\sum_{j=1}^hw_j\,\mathrm{ReLU}(\langle\mathbf v_j,\mathbf x\rangle):\|\mathbf w\|_1\le B_1,\ \|\mathbf v_j\|_2\le B_2\}$ auf Punkten mit $\|\mathbf x_i\|\le r$ gegeben.
    1. Zeigen Sie $\widehat{\mathfrak R}_S(\mathcal F_1)\le B_2r/\sqrt n$.
    1. Zeigen Sie, dass $\mathcal F_2$ im $B_1$-Fachen der konvexen Hülle von $\mathcal F_1\cup(-\mathcal F_1)$ enthalten ist. Zeigen Sie ferner, dass der Übergang zur konvexen Hülle einer Klasse ihre empirische Rademacher-Komplexität unverändert lässt. Hinweis: $\sum_jw_jf_j=\|\mathbf w\|_1\sum_j\frac{|w_j|}{\|\mathbf w\|_1}(\pm f_j)$.
    1. Zeigen Sie mithilfe von $0\in\mathcal F_1$, dass $\mathbb{E}_{\boldsymbol\varepsilon}\sup_{f\in\mathcal F_1}|\sum_i\varepsilon_if(\mathbf x_i)|\le2n\,\widehat{\mathfrak R}_S(\mathcal F_1)$ gilt, und folgern Sie daraus $\widehat{\mathfrak R}_S(\mathcal F_2)\le2B_1B_2r/\sqrt n$.
    1. Die Breite $h$ kommt in der Schranke nicht vor. Was sagt die Schranke über Gewichtszerfall auf beiden Schichten aus? Warum erklärt sie das Experiment mit zufälligen Labels in :numref:`subsec_random_labels` nicht?
   :exlabel:`ex_mdl-generalization-bounds-two-layer-norm-bound`

<!--en:05c394e3-->
[Diskussion](https://d2l.discourse.group/t/36863)

<!--en:30f784c5-->
Generalisierungsschranken, Symmetrisierung und Rademacher-Komplexität<br>**gleichmäßige Konvergenz · boolesche Ungleichung · Symmetrisierung mit einer Geisterstichprobe · Rademacher-Komplexität · Margin-Schranken · doppelter Abstieg und gutartige Überanpassung**

<!--en:dcb2c367-->
::: {.slide title="Die beste von 1000 fairen Münzen zeigt 66-mal Kopf"}

<!--en:81f68873-->
[Motivation und Selektionsbias]{.kicker}

<!--en:dc098f08-->
- **Vorab benannte Münze ($m = 100$ Würfe):** Nach der Hoeffding-Ungleichung zeigt eine faire Münze $50 \pm 13.6$-mal Kopf ($[37, 63]$), mit mindestens $95\%$ Wahrscheinlichkeit (exakt $99.3\%$).

<!--en:67fec8a6-->
- **Auswahl der besten von $k = 1{,}000$ fairen Münzen:** Mehr als $63$-mal Kopf zeigt eine einzelne Münze nur mit Wahrscheinlichkeit $0.0033$. Unter $1{,}000$ Münzen erwarten wir aber rund $3.3$ oberhalb dieser Schwelle.

<!--en:464dbc26-->
- **Zufällige Schwankung wirkt wie ein Signal:** Die beste Münze zeigt typischerweise **$66$-mal Kopf** ($[64, 69]$ in $90\%$ der Durchläufe), obwohl jede Münze fair ist.

<!--en:d3f8992d-->
::: {.slide title="Simulation des Fluchs des Gewinners bei der Modellauswahl"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:3a274c7d-->
Wiederholen wir das Experiment mit $1{,}000$ Münzen $2{,}000$-mal, so verfehlt das $95\%$-Konfidenzband für eine Münze die nachträglich ausgewählte Gewinnermünze fast immer ($96\%\text{--}97\%$):

<!--en:a07ccea9-->
- **Validierung als Münzwettbewerb:** Jede Hyperparameterkonfiguration und jeder Checkpoint im Training entspricht einer Münze, jedes Validierungsbeispiel einem Wurf.

<!--en:5a42a8cb-->
- **Umgekehrte Reihenfolge von Auswahl und Stichprobenziehung:** Wählen wir $\hat{f} = \operatorname{argmin}_{f \in \mathcal{F}} \hat{R}_S(f)$ *nach* Beobachtung der Stichprobe $S$, greifen wir systematisch die günstigste Schwankung nach unten heraus.

<!--en:e6cfd0f2-->
::: {.slide title="Von punktweiser Konzentration zu gleichmäßiger Konvergenz"}

<!--en:0c02929b-->
[Das Supremum über eine Funktionenklasse]{.kicker}

<!--en:a10bcbcd-->
- **Warum punktweise Schranken versagen:** Da $\hat{f} = \operatorname{argmin}_{f \in \mathcal{F}} \hat{R}_S(f)$ von $S$ abhängt, fällt die Wahl systematisch auf die tiefste Senke im empirischen Risiko.

<!--en:7408f311-->
- **Abhilfe durch gleichmäßige Konvergenz:** Eine Schranke für das Supremum über alle $f \in \mathcal{F}$ gilt für jeden Kandidaten, auch für $\hat{f}(S)$.

<!--en:8f014ddb-->
::: {.slide title="Endliche Funktionenklassen und die boolesche Ungleichung"}

<!--en:31179a44-->
[Logarithmische Kosten der Suche]{.kicker}

<!--en:ccfb2568-->
- **Logarithmischer Zuschlag für die Suche:** Dank des exponentiellen Abfalls der Hoeffding-Schranke wächst das Band bei der Suche unter $\lvert\mathcal{F}\rvert$ festen Modellen auf $n = 10{,}000$ Beispielen nur langsam. Für $95\%$ wächst es von $\pm 1.36\%$ ($1$ Modell) auf $\pm 2.38\%$ ($2{,}000$ Modelle) und $\pm 2.96\%$ ($10^6$ Modelle).

<!--en:fe5849fc-->
- **Polynomieller Abfall reicht nicht:** Bei Tschebyscheffs Abfall mit $1/n$ liefert die boolesche Ungleichung für $2{,}000$ Modelle die aussagelose Schranke $\pm 100\%$.

<!--en:a9fd9b2e-->
::: {.slide title="Fallstrick: Adaptive Ranglisten verletzen die Voraussetzung der booleschen Schranke"}

<!--en:e7a166e0-->
[Nichtadaptive und adaptive Abfragen]{.kicker}

<!--en:f46833e8-->
- **Nichtadaptive Auswahl ($\sqrt{\log k / n}$):** Stehen alle $k$ Kandidaten *vor* der Einsicht in die Testscores fest, gilt die logarithmische Schranke aus der booleschen Ungleichung selbst für $k = 10^6$.

<!--en:dab51e40-->
- **Adaptiver Angriff nach Blum und Hardt ($\sqrt{k / n}$):** Wer nach jeder Abfrage die Rangliste einsieht, kann $k$ zufällige Münzwurfvektoren einreichen und die mit Fehler $\le 1/2$ behalten. Ihr Mehrheitsvotum senkt den Fehler auf den zurückgehaltenen Daten auf $\frac{1}{2} - \Omega(\sqrt{k/n})$, während der wahre Fehler $50\%$ bleibt.

<!--en:b9e878a6-->
::: {.slide title="Rademacher-Komplexität: Korrelation mit Münzwurfrauschen"}

<!--en:5d7ff5c0-->
[Ein Maß für unendliche Funktionenklassen]{.kicker}

<!--en:2a1ad40f-->
- **Messbare Vielfalt:** Wie gut kann $\mathcal{F}$ auf den beobachteten Eingaben mit den zufälligen Münzwurfvorzeichen $\varepsilon_i \in \{\pm 1\}$ übereinstimmen? Die Eingaben sind $S$.

<!--en:1bd004a0-->
- **Drei Kapazitätsstufen:** Für die einelementige Klasse $\{f\}$ gilt $\widehat{\mathfrak{R}}_S = 0$, für die beiden Konstanten $\{\pm 1\}$ gilt $\mathbb{E}|\bar{\varepsilon}| \approx \sqrt{2/(\pi n)}$. Eine Klasse in $[-1, 1]$, die die Stichprobe zertrümmert, erreicht $\widehat{\mathfrak{R}}_S = 1.0$.

<!--en:e75129ea-->
::: {.slide title="Das Spektrum der Rademacher-Komplexität"}

<!--en:10de9d42-->
[Von einelementigen Klassen bis zum Zertrümmern]{.kicker}

<!--en:38ba7ba4-->
Die Rademacher-Komplexität reicht stufenlos von null für eine feste Funktion bis $1$ für eine Klasse, die beliebige Vorzeichenmuster auf $S = (\mathbf{x}_1, \dots, \mathbf{x}_n)$ auswendig lernt:

<!--en:b3baa3b6-->
| Funktionenklasse $\mathcal{F}$ | Empirische Rademacher-Komplexität $\widehat{\mathfrak{R}}_S(\mathcal{F})$ | Abhängigkeit vom Stichprobenumfang $n$ |

<!--en:a10b91ed-->
| **Einelementige Klasse** $\{f\}$ | $\mathbb{E}_{\boldsymbol{\varepsilon}}\bigl[\frac{1}{n}\sum_i \varepsilon_i f(\mathbf{x}_i)\bigr] = 0$ | Identisch $0$ (keine Auswahl) |

<!--en:c18b5387-->
| **Zwei Konstanten** $\{+1, -1\}$ | $\mathbb{E}_{\boldsymbol{\varepsilon}}\bigl[\lvert\bar{\varepsilon}\rvert\bigr] \approx \sqrt{\frac{2}{\pi n}}$ | Fällt mit Rate $O(n^{-1/2})$ |

<!--en:7e9d822a-->
| **Endliche Klasse** in $[-1, 1]$ | $\le \sqrt{\frac{2\log\lvert\mathcal{F}\rvert}{n}}$ (**Lemma von Massart**) | Logarithmisch in $\lvert\mathcal{F}\rvert$, $O(\sqrt{\log\lvert\mathcal{F}\rvert / n})$ |

<!--en:0b0a9e1e-->
| **Binäre VC-Klasse** in $\{-1, +1\}$ | $\le C\sqrt{\frac{d_{\mathrm{VC}}\log n}{n}}$ | Bestimmt durch die VC-Dimension $d_{\mathrm{VC}}$ |

<!--en:263b612a-->
| **Alle Funktionen** mit Werten in $[0, 1]$ | $\mathbb{E}_{\boldsymbol{\varepsilon}}\bigl[\frac{1}{n}\sum_i \mathbf{1}\{\varepsilon_i = +1\}\bigr] = \frac{1}{2}$ | Konstant $\frac{1}{2}$ (keine Generalisierung) |

<!--en:98fb6b89-->
| **Alle Funktionen** mit Werten in $[-1, 1]$ | $\mathbb{E}_{\boldsymbol{\varepsilon}}\bigl[\frac{1}{n}\sum_i \varepsilon_i^2\bigr] = 1$ (für paarweise verschiedene $\mathbf{x}_i$) | Konstant $1$ (vollständiges Auswendiglernen) |

<!--en:e51ea401-->
::: {.slide title="Satz: Gleichmäßige Konvergenz über die Rademacher-Komplexität"}

<!--en:c29774f2-->
[Generalisierungsschranke für unendliche Klassen]{.kicker}

<!--en:84db3e25-->
Um das Generalisierungsrisiko zu beschränken, betrachten wir die Rademacher-Komplexität der **Verlustklasse mit Werten in $[0, 1]$** $\mathcal{G} = \{z = (\mathbf{x}, y) \mapsto \ell(f(\mathbf{x}), y) : f \in \mathcal{F}\}$. Mit Wahrscheinlichkeit mindestens $1 - \delta$ bezüglich einer unabhängig und identisch verteilten Stichprobe $S$ vom Umfang $n$ gilt gleichzeitig für jedes $g \in \mathcal{G}$:

<!--en:7940ea2f-->
- **Datenabhängige empirische Form:** Eine zweite Anwendung der McDiarmid-Ungleichung ersetzt den Populationsmittelwert $\mathfrak{R}_n(\mathcal{G})$ durch die berechenbare Stichprobengröße $\widehat{\mathfrak{R}}_S(\mathcal{G})$. Dafür wächst der Konfidenzterm auf $3\sqrt{\frac{\log(2/\delta)}{2n}}$.

<!--en:aa8ec4b0-->
- **Exakter Zusammenhang bei binärer $0\text{--}1$-Klassifikation:** Für Prädiktoren $f(\mathbf{x}) \in \{-1, +1\}$ und Labels $y \in \{-1, +1\}$ ist der $0\text{--}1$-Verlust gleich $g(\mathbf{x}, y) = \frac{1 - yf(\mathbf{x})}{2}$. Da $-\varepsilon_i y_i$ genauso verteilt ist wie $\varepsilon_i$, gelten $\widehat{\mathfrak{R}}_S(\mathcal{G}) = \frac{1}{2}\widehat{\mathfrak{R}}_S(\mathcal{F})$ und $2\widehat{\mathfrak{R}}_S(\mathcal{G}) = \widehat{\mathfrak{R}}_S(\mathcal{F})$.

<!--en:0ac40014-->
::: {.slide title="Beweis durch Symmetrisierung: Geisterstichprobe und Vorzeichenwechsel"}

<!--en:6e807739-->
[Symmetrisierung in drei Schritten]{.kicker}

<!--en:ad28734e-->
- **1. Geisterstichprobe $S'$:** Ziehen wir den Erwartungswert $R(g) = \mathbb{E}_{S'}[\hat{R}_{S'}(g)]$ mithilfe der Jensen-Ungleichung vor das Supremum.

<!--en:4535b4c8-->
- **2. Vertauschen der Paare ($z_i \leftrightarrow z'_i$):** Multiplizieren wir jede Differenz $g(z'_i) - g(z_i)$ mit einem unabhängigen Rademacher-Vorzeichen $\varepsilon_i \in \{\pm 1\}$ und zerlegen das Supremum. So ergibt sich $2\,\mathfrak{R}_n(\mathcal{G})$.

<!--en:0909e3a4-->
::: {.slide title="Geometrie der Rademacher-Komplexität linearer Modelle"}

<!--en:f9f2b37f-->
[Normbeschränkte lineare Prädiktoren]{.kicker}

<!--en:2977f42f-->
- **Normbeschränkte lineare Klasse:** Betrachten wir $\mathcal{F} = \bigl\{\mathbf{x} \mapsto \langle\mathbf{w}, \mathbf{x}\rangle : \|\mathbf{w}\|_2 \le B\bigr\}$ für Stichprobeneingaben mit $\|\mathbf{x}_i\|_2 \le r$.

<!--en:e25908f2-->
- **Exaktes inneres Supremum mit Cauchy-Schwarz:** Aus der Linearität des inneren Produkts folgt $\frac{1}{n}\sum_{i=1}^n \varepsilon_i \langle\mathbf{w}, \mathbf{x}_i\rangle = \bigl\langle\mathbf{w},\, \frac{1}{n}\sum_{i=1}^n \varepsilon_i \mathbf{x}_i\bigr\rangle$. Das Maximum liegt dort, wo $\mathbf{w}^*$ in Richtung der vorzeichenbehafteten Vektorsumme $\mathbf{s} = \sum_{i=1}^n \varepsilon_i \mathbf{x}_i$ zeigt:
  $$\widehat{\mathfrak{R}}_S(\mathcal{F}) = \frac{B}{n}\,\mathbb{E}_{\boldsymbol{\varepsilon}}\Biggl[\Biggl\|\sum_{i=1}^n \varepsilon_i \mathbf{x}_i\Biggr\|_2\Biggr]$$

<!--en:ddb902b2-->
::: {.slide title="Dimensionsunabhängige Rademacher-Komplexität linearer Modelle"}

<!--en:6ec931cf-->
[Beweis von R_S(F) <= B r / sqrt(n)]{.kicker}

<!--en:8866d33b-->
Den Erwartungswert der euklidischen Norm von $\mathbf{s} = \sum_{i=1}^n \varepsilon_i \mathbf{x}_i$ beschränken wir mithilfe der Jensen-Ungleichung ($\mathbb{E}\|\mathbf{s}\|_2 \le \sqrt{\mathbb{E}\|\mathbf{s}\|_2^2}$) durch das quadratische Mittel der Norm. Damit erhalten wir eine obere Schranke in geschlossener Form:

<!--en:73042baa-->
- **Orthogonalität unabhängiger Vorzeichen:** Jeder gemischte Term $i \neq j$ verschwindet, weil $\mathbb{E}[\varepsilon_i \varepsilon_j] = \mathbb{E}[\varepsilon_i]\mathbb{E}[\varepsilon_j] = 0$ gilt. Dagegen ist $\varepsilon_i^2 = 1$.

<!--en:e1efa9d1-->
- **Die Dimension $d$ des umgebenden Raums spielt keine Rolle:** Ob $\mathbf{x}_i$ in $\mathbb{R}^{10}$ oder $\mathbb{R}^{10^6}$ liegt, nur das Normprodukt $Br$ bestimmt die Komplexität. Das begründet den **$L_2$-Gewichtszerfall**.

<!--en:0434993d-->
- **Erweiterung auf zweischichtige ReLU-Netze:** Für $f(\mathbf{x}) = \sum_{j=1}^h w_j \mathrm{ReLU}(\langle\mathbf{v}_j, \mathbf{x}\rangle)$ mit $\|\mathbf{w}\|_1 \le B_1$ und $\|\mathbf{v}_j\|_2 \le B_2$ gilt $\widehat{\mathfrak{R}}_S \le \frac{2B_1 B_2 r}{\sqrt{n}}$ unabhängig von der Breite $h$ der versteckten Schicht.

<!--en:91f775f7-->
::: {.slide title="Talagrands Kontraktionslemma und Margin-Schranken"}

<!--en:fcd56210-->
[Von reellwertigen Scores zum Klassifikationsrisiko]{.kicker}

<!--en:6b666fb0-->
- **Rampenverlust $\phi_\gamma(u)$:** Beschränkt den stufenförmigen $0\text{--}1$-Verlust überall nach oben und ist $(1/\gamma)$-Lipschitz-stetig auf $[0, \gamma]$.

<!--en:d4aa5625-->
- **Talagrands Kontraktion:** Die lineare Rademacher-Komplexität $Br/\sqrt{n}$ erhält den Faktor $1/\gamma$. Eine große Margin $\gamma / B$ garantiert somit einen kleinen Testfehler in jeder Dimension $d$.

<!--en:a7b4efd2-->
::: {.slide title="Dimensionsunabhängige Komplexität im Monte-Carlo-Experiment"}

<!--en:fae9edc0-->
[Empirische Rademacher-Komplexität im Code]{.kicker}

<!--en:53cf799e-->
Mitteln wir das exakte Supremum $\frac{B}{n}\bigl\|\sum_{i=1}^n \varepsilon_i \mathbf{x}_i\bigr\|_2$ über $2{,}000$ zufällige Vorzeichenvektoren für $n = 50$ Einheitsvektoren ($B = r = 1$) in den Dimensionen $d \in \{2, 20, 1000\}$, bestätigt sich die Theorie:

<!--en:6720488c-->
- **Warum die Schranke in hohen Dimensionen schärfer wird:** Die einzige Ungleichung im Beweis war die Jensen-Ungleichung $\mathbb{E}\|\mathbf{s}\|_2 \le \sqrt{\mathbb{E}\|\mathbf{s}\|_2^2}$. Bei $d = 2$ beträgt die mittlere Länge eines zweidimensionalen Gauß-Vektors das $\frac{\sqrt{\pi}}{2} \approx 0.89$-Fache ihres quadratischen Mittels ($0.125$ gegenüber $0.141$). Bei $d \ge 20$ ist die Norm $\|\mathbf{s}\|_2$ wegen der Konzentration auf einer dünnen Schale (§{sec_mdl-concentration-generalization}) für alle Vorzeichenmuster fast gleich. Der Abstand sinkt auf $< 1\%$.

<!--en:bfa308da-->
::: {.slide title="Die Grenze der gleichmäßigen Konvergenz: Anpassung an zufällige Labels"}

<!--en:f80ab8e6-->
[Die Interpolationsgrenze]{.kicker}

<!--en:2b3b1b3f-->
Was geschieht, wenn eine Modellfamilie überparametrisiert ist ($p = 2n = 100$ standardnormalverteilte Merkmale auf $n = 50$ Punkten) und jeden zufälligen Vorzeichenvektor $\boldsymbol{\varepsilon} \in \{\pm 1\}^n$ durch $\mathbf{w} = \boldsymbol{\Phi}^+\boldsymbol{\varepsilon}$ interpolieren kann?

<!--en:af31ef88-->
- **Interpolation von Rauschen erzwingt $Br/\sqrt{n} \ge 1$:** Da $\boldsymbol{\Phi}\mathbf{w} = \boldsymbol{\varepsilon}$ für jedes Vorzeichenmuster die Korrelation $1$ erreicht, gelten für jede Gewichtskugel $\|\mathbf{w}\|_2 \le B$, die diese Interpolanten enthält, $\widehat{\mathfrak{R}}_S(\mathcal{F}) \ge 1$ und $Br/\sqrt{n} \approx 2.4 > 1$.

<!--en:ed69113e-->
- **Das Rätsel der zufälligen Labels bei Zhang et al. (2017):** Tiefe neuronale Netze können zufällige Labels ohne Trainingsfehler nachbilden. Für die unbeschränkte Klasse der Architektur gilt daher $2\mathfrak{R}_n(\mathcal{G}) \approx 1$. Schranken für die gleichmäßige Konvergenz über die ganze Klasse sagen dann nichts aus. Die Generalisierung hängt von der konkreten Lösung ab, die der Optimierer auswählt.

<!--en:1ed47356-->
::: {.slide title="Interpolation mit kleinster Norm und das SVD-Spektrum"}

<!--en:6ad125e8-->
[Der Mechanismus des doppelten Abstiegs]{.kicker}

<!--en:98e51cd5-->
- **Warum bei $p = n$ die Norm explodiert:** An der Interpolationsschwelle $p = n$ verschwindet das Residuum. Eine quadratische Zufallsmatrix hat einen winzigen Singulärwert $\sigma_n \ll 1$. Dadurch werden $1/\sigma_n$ und $\|\mathbf{w}\|_2$ sehr groß.

<!--en:c4e6987a-->
- **Warum sich die Lage bei $p \gg n$ bessert:** Zusätzliche Merkmale erhöhen $\sigma_n$. Die Norm des Interpolanten kleinster Norm fällt monoton ($\|\mathbf{w}_{p+1}\|_2 \le \|\mathbf{w}_p\|_2$).

<!--en:68684069-->
::: {.slide title="Doppelter Abstieg mit zufälligen ReLU-Merkmalen in der Simulation"}

<!--en:7c26f661-->
[Die drei Bereiche im Code]{.kicker}

<!--en:3010b370-->
- **Versuchsaufbau ($n = 40, d_{\mathrm{in}} = 15, \sigma_\xi = 0.1$):** Wir variieren die Anzahl $p \in [2, 400]$ geschachtelter zufälliger ReLU-Merkmale in $20$ Durchläufen und erhalten die vollständige Kurve des doppelten Abstiegs.

<!--en:20253ebb-->
- **Unterparametrisiert ($p < 40$):** Die klassische U-Kurve erreicht nahe $p = 15$ ihren Tiefpunkt ($\mathrm{MSE} \approx 0.7$) und steigt zur Schwelle hin wieder an, während $\|\mathbf{w}\|_2$ wächst.

<!--en:5e3740f1-->
- **Interpolationsspitze ($p = n = 40$) und zweiter Abstieg ($p > 40$):** Der Test-MSE springt auf rund $10\text{--}70$ bei $p = 40$, ein bis zwei Größenordnungen über dem Minimum der U-Kurve. Dann fällt er monoton auf $\approx 0.06$ bei $p = 400$, während $\|\mathbf{w}\|_2$ auf $0.4$ sinkt.

<!--en:cca2c9a5-->
::: {.slide title="Die Ergebnistabelle zum doppelten Abstieg im Detail"}

<!--en:0be10957-->
[Singulärwerte, Normen und Vergleichswerte]{.kicker}

<!--en:f047046f-->
Die neun Messgrößen über $p \in \{15, 38, 40, 42, 50, 80, 400\}$ zeigen drei wichtige Einzelheiten:

<!--en:554ca998-->
- **Trainingsfehler jenseits von $p = 40$ ohne Aussagekraft:** Für jedes $p \ge 40$ ist der Trainings-MSE bis auf Gleitkommagenauigkeit null ($\sim 10^{-28}$). Der Test-MSE sinkt dagegen um mehr als den Faktor $100\times$ beim Übergang von $p = 40$ auf $400$. Dies zeigen auch $\sigma_{\min}$ ($0.02 \to 2.8$) und $\|\mathbf{w}\|_2$ ($10$ bis $30 \to 0.4$).

<!--en:d946dd92-->
- **Eine falsch spezifizierte Zielfunktion genügt für die Spitze:** Schon bei $p = 40$ übersteigt der rauschfreie Fehler `clean` den Rauschfehler `noise`, weil $40$ ReLU-Kammfunktionen das lineare Lehrermodell entlang $\mathbf{u}_n$ schlecht approximieren.

<!--en:8d50675e-->
- **Korrekt spezifizierte kleine Modelle gewinnen weiterhin:** Das Modell mit $p = 400$ schlägt jedes kleinere ReLU-Modell. Die gewöhnliche Methode der kleinsten Quadrate direkt auf den $15$ Roheingaben erreicht aber $\mathrm{MSE} \approx 0.006$, nochmals um den Faktor $10\times$ weniger.

<!--en:a2b8ee5f-->
::: {.slide title="Zerlegung interpolierender Lösungen in Verzerrung und Varianz"}

<!--en:ea128bdb-->
[Rauschfreier Fehler und Rauschfehler]{.kicker}

<!--en:3fce2bad-->
- **Linearität von $\boldsymbol{\Phi}^+$:** Die Darstellung $\mathbf{w} = \boldsymbol{\Phi}^+\mathbf{y}_0 + \boldsymbol{\Phi}^+\boldsymbol{\xi}$ trennt den Approximationsfehler für die rauschfreie Zielfunktion von der Verstärkung des Labelrauschens.

<!--en:190e17de-->
- **Mittleres Energieverhältnis:** Für $p \ge n$ ist der Rauschfehler gleich $\sigma_\xi^2$ mal dem mittleren Verhältnis der Merkmalsenergien auf Test- und Trainingsdaten über die $n$ Singulärrichtungen.

<!--en:eafb138c-->
::: {.slide title="Warum Überanpassung bei großer Breite (p >> n) gutartig wird"}

<!--en:607e82dc-->
[Verhältnis der Merkmalsenergien auf Test- und Trainingsdaten]{.kicker}

<!--en:49638831-->
- **Schwelle $p = n = 40$ (katastrophal):** Entlang $\mathbf{v}_n$ ist die Energie der $40$ Trainingspunkte zufällig winzig ($\sigma_{\min}^2/n \approx 10^{-5}$). Testeingaben haben dagegen Energie der Größenordnung $O(1)$. Das Energieverhältnis beträgt $\sim 10^4\times$.

<!--en:5532095f-->
- **Große Breite $p = 400$ (gutartige Überanpassung):** Das Labelrauschen verteilt sich auf varianzarme Richtungen, in denen *auch* die Projektionen der Testeingaben klein sind. Das mittlere Energieverhältnis bleibt bei $\approx 0.9\times$.

<!--en:0b4839fe-->
::: {.slide title="Diagnose und Garantie: Grenzen gleichmäßiger Schranken"}

<!--en:a201c34f-->
[Was die Theorie garantiert und was offenbleibt]{.kicker}

<!--en:184dea20-->
Wie passt unsere Rademacher-Margin-Schranke $\frac{Br}{\gamma\sqrt{n}}$ zum beobachteten doppelten Abstieg?

<!--en:ac3e4c9f-->
- **Warum sich $B$ nicht nachträglich einsetzen lässt:** Für die lineare Rademacher-Schranke muss der Normradius $B$ *vor* der Beobachtung von $S$ feststehen. Für eine datenabhängige Norm $\|\hat{\mathbf{w}}\|_2$ nutzt die **strukturelle Risikominimierung** die boolesche Ungleichung über eine geometrische Folge von Radien $B_j = 2^j$ mit Ausfallwahrscheinlichkeiten $\delta_j = \frac{\delta}{j(j+1)}$ ($\sum_{j=1}^\infty \delta_j = \delta$). Das kostet höchstens einen Faktor $2$ bei $\|\hat{\mathbf{w}}\|_2$ und einen Konfidenzterm der Ordnung $O\bigl(\sqrt{\frac{\log\log\|\hat{\mathbf{w}}\|_2}{n}}\bigr)$.

<!--en:de987bf8-->
- **Wann das Kapazitätsprodukt zur Diagnose dient:** Im Regressionsexperiment folgt $\|\hat{\mathbf{w}}\|_2 r/\sqrt{n}$ der Spitze und dem Abstieg des Test-MSE ($15 \to 1.2$). Es zeigt, in *welche* Teilklasse der Interpolant kleinster Norm fällt, obwohl der Satz für Verluste in $[0, 1]$ den unbeschränkten quadratischen Verlust nicht erfasst.

<!--en:1f7befbc-->
- **Offene Frage der Theorie des Deep Learning:** Gleichmäßige Konvergenz beschränkt die *ungünstigste* Funktion einer vorab festgelegten Klasse $\mathcal{F}$. Beim Training überparametrisierter Modelle wählt der **implizite Bias des Optimierers** (z. B. kleinste Norm oder größte Margin) eine gutartige Lösung aus einer Klasse, die sogar Rauschen nachbilden kann.

<!--en:a6fa62f5-->
::: {.slide title="Ansätze zur Beschreibung der Generalisierung im Vergleich"}

<!--en:c9af3aca-->
[Einzelmodell, endliche Klasse, lineare Margin und Interpolation]{.kicker}

<!--en:13735e1a-->
| Fall | Komplexitätsmaß | Schranke für die Generalisierungslücke oder Skalierung | Voraussetzung oder Einschränkung |

<!--en:a8f3986d-->
| **Einzelnes festes Modell** $\{f\}$ | $\mathfrak{R}_n = 0$ | $\sqrt{\frac{\log(2/\delta)}{2n}}$ | $f$ muss feststehen, *bevor* wir den Testdatensatz $S$ ziehen |

<!--en:3da97a0d-->
| **Endliche Kandidatenklasse** $\mathcal{F}$ | $\log\lvert\mathcal{F}\rvert$ | $\sqrt{\frac{\log(2\lvert\mathcal{F}\rvert/\delta)}{2n}}$ | Klasse $\mathcal{F}$ muss nichtadaptiv sein (keine Rückmeldung aus der Rangliste) |

<!--en:48726893-->
| **Allgemeine unendliche Klasse** $\mathcal{G}$ | $2\,\mathfrak{R}_n(\mathcal{G})$ | $2\,\mathfrak{R}_n(\mathcal{G}) + \sqrt{\frac{\log(1/\delta)}{2n}}$ | Erfordert Verluste in $[0, 1]$ und unabhängig und identisch verteilte Daten |

<!--en:8f2cadbb-->
| **Linearer Margin-Klassifikator** | $\frac{Br}{\gamma\sqrt{n}}$ | $\hat{R}_\gamma(\mathbf{w}) + \frac{2Br}{\gamma\sqrt{n}} + \sqrt{\frac{\log(1/\delta)}{2n}}$ | Dimensionsunabhängig, aussagelos ab $Br/\gamma \gtrsim \sqrt{n}$ |

<!--en:6bbb9297-->
| **Zweischichtiges ReLU-Netz** | $\frac{2B_1 B_2 r}{\sqrt{n}}$ | $\ell_1 \times \ell_2$-Pfadnormen bestimmen die Schranke | Breitenunabhängig, aber die Normen nach dem Training übersteigen oft $\sqrt{n}$ |

<!--en:da427d51-->
| **Interpolant kleinster Norm** | $\sigma_{\min}(\boldsymbol{\Phi})$ und effektive Ränge | Rauschfehler $\frac{\sigma_\xi^2}{n}\sum_i \frac{\|\boldsymbol{\Phi}_{\mathrm{te}}\mathbf{v}_i\|^2/n_{\mathrm{te}}}{\|\boldsymbol{\Phi}\mathbf{v}_i\|^2/n}$ | Algorithmusabhängig, Spitze bei $p = n$, gutartig für $p \gg n$ |

<!--en:81c25b55-->
::: {.slide title="Verständnisfragen: Gleichmäßige Konvergenz und doppelter Abstieg"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:6b1fc529-->
1. **Datenbedarf bei Verdopplung eines Hyperparametergitters:** Wie viele zusätzliche Validierungsbeispiele erfordert nach der booleschen Ungleichung für endliche Klassen $n = \frac{\log(2/\delta) + \log\lvert\mathcal{F}\rvert}{2t^2}$ eine Toleranz von $t = 0.01$, wenn Sie die Anzahl $\lvert\mathcal{F}\rvert$ der Modellkandidaten verdoppeln?

<!--en:68165fb2-->
2. **Rademacher-Komplexität eines interpolierenden Klassifikators:** Eine Familie binärer Klassifikatoren $\mathcal{F} \subseteq \{\pm 1\}^{\mathcal{X}}$ könne jedes Vorzeichenmuster $\boldsymbol{\varepsilon} \in \{\pm 1\}^n$ auf dem Trainingsdatensatz $S$ nachbilden, ihn also zertrümmern. Wie groß sind $\widehat{\mathfrak{R}}_S(\mathcal{F})$ und die resultierende Rademacher-Schranke für das $0\text{--}1$-Risiko?

<!--en:3db094a8-->
3. **Zwei Lösungen ohne Trainingsfehler unterscheiden:** Im Experiment mit zufälligen Merkmalen erreichen sowohl $p = 42$ als auch $p = 400$ einen Trainings-MSE von $\approx 10^{-28}$. Nennen Sie drei Größen aus der SVD-Analyse, die ihr Generalisierungsverhalten unterscheiden.

<!--en:557b98b6-->
**Lösungen:**

<!--en:58e81bc1-->
1. Verdoppeln wir $\lvert\mathcal{F}\rvert$, so kommt $\log 2 \approx 0.693$ zu $\log\lvert\mathcal{F}\rvert$ hinzu. Das erfordert unabhängig von der bisherigen Kandidatenzahl $\frac{\log 2}{2(0.01)^2} \approx 3{,}466$ zusätzliche Validierungsbeispiele.

<!--en:89419820-->
2. Da $\mathcal{F}$ jedes $\boldsymbol{\varepsilon}$ exakt nachbildet, gilt $\widehat{\mathfrak{R}}_S(\mathcal{F}) = 1$. Für die $0\text{--}1$-Verlustklasse $\mathcal{G}$ gilt $\widehat{\mathfrak{R}}_S(\mathcal{G}) = \frac{1}{2}\widehat{\mathfrak{R}}_S(\mathcal{F}) = \frac{1}{2}$. Der Komplexitätsterm $2\widehat{\mathfrak{R}}_S(\mathcal{G}) = 1.0$ ($100\%$) macht die Schranke daher aussagelos.

<!--en:15266594-->
3. Beim Übergang von $p = 42$ zu $p = 400$ sinkt (1) die Lösungsnorm $\|\mathbf{w}\|_2$ von $\sim 5$ auf $0.4$, steigt (2) der kleinste Singulärwert $\sigma_{\min}$ von $\sim 0.08$ auf $2.8$ und sinkt (3) das Verhältnis der Merkmalsenergien auf Test- und Trainingsdaten entlang $\mathbf{v}_{\min}$ von $\sim 500$ bis $800$ auf $\sim 2$.

<!--en:058097ff-->
::: {.slide title="Zusammenfassung: Generalisierung und doppelter Abstieg"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:f932efc8-->
| Begriff | Mathematische Form | Bedeutung für Statistik und Deep Learning |

<!--en:141689ba-->
| **Gleichmäßige Konvergenz** | $\sup_{f \in \mathcal{F}} \lvert\hat{R}_S(f) - R(f)\rvert \le t$ | Sichert die datenabhängige Modellauswahl $\hat{f} = \operatorname{argmin}_f \hat{R}_S(f)$ ab |

<!--en:a9e7841d-->
| **Boolesche Ungleichung für endliche Klassen** | $\lvert\hat{R}_S(f) - R(f)\rvert \le \sqrt{\frac{\log(2\lvert\mathcal{F}\rvert/\delta)}{2n}}$ | Dank exponentiellen Abfalls kostet die Suche unter $\lvert\mathcal{F}\rvert$ festen Modellen nur $\log\lvert\mathcal{F}\rvert$ |

<!--en:e422bab1-->
| **Rademacher-Komplexität** | $\widehat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_{\boldsymbol{\varepsilon}}\bigl[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum_i \varepsilon_i f(\mathbf{x}_i)\bigr]$ | Symmetrisierung mit einer Geisterstichprobe führt die Generalisierungslücke auf die Anpassung an zufällige Vorzeichen zurück |

<!--en:574ed392-->
| **Schranken für lineare Modelle und Margin-Schranken** | $\widehat{\mathfrak{R}}_S(\mathcal{F}) \le \frac{Br}{\sqrt{n}}$ und $R_{0\text{--}1} \le \hat{R}_\gamma + \frac{2Br}{\gamma\sqrt{n}} + \sqrt{\frac{\log(1/\delta)}{2n}}$ | Die Kapazität hängt vom Normprodukt $Br/\gamma$ ab und ist unabhängig von der Dimension $d$ |

<!--en:53b2afa5-->
| **Interpolationsspitze in der SVD** | $\mathbf{w} = \boldsymbol{\Phi}^+\mathbf{y} = \sum_{i=1}^q \frac{\langle\mathbf{u}_i, \mathbf{y}\rangle}{\sigma_i}\mathbf{v}_i$ | Der Einbruch von $\sigma_{\min}$ bei $p = n$ lässt $\|\mathbf{w}\|_2$ und den Testfehler stark ansteigen |

<!--en:829a33b5-->
| **Gutartige Überanpassung** | $\text{Noise MSE} = \frac{\sigma_\xi^2}{n}\sum_{i=1}^n \frac{\|\boldsymbol{\Phi}_{\mathrm{te}}\mathbf{v}_i\|^2/n_{\mathrm{te}}}{\|\boldsymbol{\Phi}\mathbf{v}_i\|^2/n}$ | Für $p \gg n$ nehmen energiearme Richtungen, in denen auch die Testmerkmale klein sind, das Rauschen auf |
