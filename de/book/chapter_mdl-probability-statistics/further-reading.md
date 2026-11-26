<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:0f8c5028-->
Am Ende dieses Abschnitts finden Sie Bücher und Kurse zu den Themen des Kapitels. Zunächst wenden wir uns den Fragen zu, die das Kapitel offenlässt. Die kurzen Beiträge folgen dem Aufbau des Kapitels. Jeder geht von einer Grenze einer bereits behandelten Methode aus und zeigt, wie die Fachliteratur darüber hinausgeht. Zwei Ansätze, die konforme Vorhersage und die Konfidenzsequenzen, helfen unmittelbar in der Praxis. Mit ihnen lassen sich die Intervalle und Tests aus :numref:`sec_mdl-statistics` auch unter den weniger idealen Bedingungen realer Experimente einsetzen. Andere Konzepte wie die VC-Dimension sind vor allem theoretisch interessant. Manche Fragen sind nach wie vor umstritten, etwa ob Gradientenrauschen tatsächlich schwere Verteilungsränder hat. Die späteren Kapitel setzen diesen Abschnitt nicht voraus. Nutzen Sie ihn als Wegweiser für die weitere Lektüre.

<!--en:4e312377-->
## Wahrscheinlichkeitsmodelle

<!--en:f70877d7-->
Wir ergänzen unsere Betrachtung von Verteilungen um drei Themen. Das erste ist eine Klasse von Verteilungen, die sich der üblichen Einteilung in diskret und stetig entzieht. Das zweite sind Grenzverteilungen für Mittelwerte bei unendlicher Varianz. Das dritte ist eine bekannte Besonderheit der Softmax-Funktion bei ähnlichen Wahlmöglichkeiten.

<!--en:3f2eee7e-->
### Verteilungen ohne Atome und ohne Dichte

<!--en:e9632933-->
In :numref:`sec_mdl-random_variables` haben wir festgehalten, dass eine Zufallsvariable mit Dichte keine Punktmassen (Atome) hat. Dort haben wir jede Verteilung als diskret, als stetig mit Dichte oder als Mischung aus beidem behandelt. Überraschenderweise gilt die Umkehrung nicht. Auch eine Verteilung ohne Atome kann ohne Dichte sein. Um das zu sehen, konstruieren wir eine Zufallszahl in $[0, 1]$ zur Basis $3$. Als Ziffern verwenden wir nur $0$ und $2$ und wählen jede Ziffer unabhängig durch einen fairen Münzwurf. Die Verteilungsfunktion ist überall stetig. Deshalb hat kein einzelner Punkt positive Wahrscheinlichkeit. Dennoch liegt die gesamte Wahrscheinlichkeitsmasse auf der Cantor-Menge, deren Gesamtlänge null ist. Über eine solche Menge kann keine Dichte das Integral $1$ ergeben.

<!--en:b67974c8-->
Die *Lebesgue-Zerlegung* vervollständigt diese Einteilung. Jede Wahrscheinlichkeitsverteilung lässt sich in einen diskreten Anteil, einen Anteil mit Dichte und einen *singulär stetigen* Anteil wie bei der Cantor-Verteilung zerlegen. Eine Verteilung besitzt genau dann eine Dichte, wenn sie bezüglich des Längenmaßes *absolut stetig* ist. In diesem Fall liefert der Satz von Radon und Nikodym die Dichte. :citet:`Billingsley.1995` entwickelt die Wahrscheinlichkeitstheorie auf dieser maßtheoretischen Grundlage. Der Erwartungswert ist dann ein einziges Integral bezüglich der Verteilung. Die Summen und Integrale aus diesem Kapitel ergeben sich als zwei Spezialfälle.

<!--en:b9156b2b-->
### Schwere Verteilungsränder und stabile Verteilungen

<!--en:995d4630-->
Wie in :numref:`sec_mdl-random_variables` erwähnt, hat der Mittelwert unabhängiger Ziehungen aus derselben Cauchy-Verteilung wieder genau diese Cauchy-Verteilung. Aufgabe :exref:`ex_mdl-random-variables-heavy-tail-averages` zeigt, dass sich Mittelwerte bei Variablen mit potenzgesetzlich abfallenden Verteilungsrändern viel langsamer stabilisieren, als die bekannte Rate $1/\sqrt{n}$ erwarten lässt. Wogegen konvergieren sie, wenn sie sich keiner Gauß-Verteilung annähern? Fällt die Randwahrscheinlichkeit $P(|X| > x)$ wie $x^{-\alpha}$ mit $0 < \alpha < 2$ ab, ist die Varianz unendlich. Teilen wir geeignet zentrierte Summen durch $n^{1/\alpha}$ statt durch $\sqrt{n}$, konvergieren sie gegen eine *$\alpha$-stabile* Verteilung. Die Cauchy-Verteilung ist die symmetrische stabile Verteilung mit $\alpha = 1$, die Gauß-Verteilung entspricht dem Grenzfall $\alpha = 2$. :citet:`Gnedenko.Kolmogorov.1954` entwickeln die klassische Theorie dieser Grenzverteilungen.

<!--en:4875a291-->
Im Deep Learning sind schwere Verteilungsränder beim Gradientenrauschen von Minibatches auf Interesse gestoßen. :citet:`Simsekli.Sagun.Gurbuzbalaban.2019` modellieren dieses Rauschen als $\alpha$-stabil. Auch in den Eigenwertspektren trainierter Gewichtsmatrizen hat man schwere Verteilungsränder untersucht :cite:`Martin.Mahoney.2021`. Solche empirischen Aussagen sind mit Vorsicht zu lesen. Jede beruht auf einer Schätzung des Randindex $\alpha$, und der Schätzwert kann stark davon abhängen, wie weit außen man den Verteilungsrand untersucht.

<!--en:e9f6ae4a-->
### Wahlmodelle und irrelevante Alternativen

<!--en:2709f438-->
In :numref:`subsec_mdl-softmax-origins` haben wir die Softmax-Funktion hergeleitet, indem wir zum Score jeder Option unabhängiges Gumbel-Rauschen addiert haben. Aus dieser Unabhängigkeit folgt, dass das Verhältnis der Wahlwahrscheinlichkeiten zweier Optionen, $p_i / p_j = e^{z_i - z_j}$, von allen anderen verfügbaren Optionen unabhängig ist. Sobald zwei Optionen nahezu identisch sind, beschreibt diese Eigenschaft das Wahlverhalten nicht mehr angemessen. Stellen Sie sich einen Pendler vor, der ebenso oft mit dem Auto wie mit dem roten Bus fährt. Setzt die Stadt einen bis auf seine blaue Farbe identischen zweiten Bus ein, sollte der Pendler weiterhin die Hälfte der Fahrten mit dem Auto zurücklegen und die andere Hälfte auf die beiden Busse verteilen. Die Softmax-Funktion weist jedoch allen drei Optionen die Wahrscheinlichkeit $\tfrac{1}{3}$ zu.

<!--en:c3d6b8c6-->
Lassen wir für ähnliche Optionen korreliertes Rauschen zu, bleibt die Wahrscheinlichkeit für das Auto nahe bei $\tfrac{1}{2}$. Beispiele sind ein verschachteltes Logit-Modell oder ein Probit-Modell mit korreliertem gaußschem Rauschen. Dafür verlieren wir die geschlossene Formel für die Wahlwahrscheinlichkeiten. :citet:`McFadden.1974` leitete das Logit-Modell aus Gumbel-Rauschen her und hob die *Unabhängigkeit von irrelevanten Alternativen* nach Luce als zentrale Einschränkung des Modells hervor.

<!--en:89afc9a1-->
## Schätzen und Testen

<!--en:e1cc6019-->
Die Konfidenzintervalle und Hypothesentests in :numref:`sec_mdl-statistics` stützten sich auf drei Vereinfachungen. Wir verwendeten eine Gauß-Approximation, kannten das Fehlermodell oder legten den Stichprobenumfang vor der Sichtung der Daten fest. Die folgenden drei Themen zeigen jeweils, was passiert, wenn eine dieser Voraussetzungen entfällt.

<!--en:5d8f1379-->
### Bessere Bootstrap-Intervalle

<!--en:0649f53c-->
Das einfache Bootstrap-Verfahren mit Perzentilen aus :numref:`sec_mdl-statistics` übernimmt jede Verzerrung und Schiefe der Schätzwerte aus den Bootstrap-Stichproben. Bei kleinen Stichproben oder schief verteilten Statistiken kann die tatsächliche Überdeckung daher merklich unter dem nominalen Niveau liegen. Zwei Verfeinerungen schließen diese Lücke weitgehend. Beim *studentisierten* Bootstrap-Verfahren berechnen wir für die Bootstrap-Stichproben die selbstnormierte Pivotgröße $(\hat\theta^* - \hat\theta)/\widehat{\textrm{se}}^*$ anstelle von $\hat\theta^*$. Das *BCa*-Intervall (bias-corrected and accelerated) verschiebt die Perzentilgrenzen. Es berücksichtigt dabei die Verzerrung und die Abhängigkeit des Standardfehlers von $\theta$. Bei einseitigen Intervallen senken beide Verfahren den Überdeckungsfehler von der Größenordnung $1/\sqrt{n}$ auf $1/n$ :cite:`Hall.1992`. Wenn Sie die Genauigkeit anhand von Tausenden von Testbeispielen angeben, brauchen Sie diese Verfahren selten. Bei kleinen Evaluierungsdatensätzen oder bei Quotienten von Fehlerraten zahlen sie sich unmittelbar aus. :citet:`DiCiccio.Efron.1996` geben einen verständlichen Überblick über diese Methoden.

<!--en:21c854e6-->
### Konforme Vorhersage

<!--en:3d46456b-->
Ein trainiertes neuronales Netz liefert uns eine Punktvorhersage. Für ein Vorhersageintervall mit garantierter Überdeckung müssten wir jedoch ein parametrisches Fehlermodell annehmen. Die *konforme Vorhersage mit Datenteilung* (engl. split conformal prediction) kommt ganz ohne Modellannahmen aus. Für jeden von $n$ zurückgehaltenen Kalibrierungspunkten messen wir mit einem Score, wie weit die Vorhersage danebenliegt, etwa mit dem Betrag des Residuums $|y - f(\mathbf{x})|$. Für eine neue Eingabe geben wir dann alle Label zurück, deren Score höchstens so groß ist wie der $\lceil(n+1)(1-\alpha)\rceil$-kleinste Kalibrierungsscore.

<!--en:ad73e69b-->
Sind Kalibrierungs- und Testpunkte austauschbar, enthält die resultierende Vorhersagemenge das wahre Label mit einer Wahrscheinlichkeit von mindestens $1 - \alpha$. Das gilt für *jedes* Modell und *jede* Datenverteilung. Der Beweis nutzt dieselbe Symmetrie wie der Permutationstest. Bei Austauschbarkeit ist der Rang des Test-Scores unter allen $n + 1$ Scores gleichverteilt. Für die Praxis ist eine Einschränkung wichtig. Die Garantie gilt im Mittel über die Eingaben. Für eine einzelne vorgegebene Eingabe muss sie nicht gelten. :citet:`Vovk.Gammerman.Shafer.2022` entwickeln die vollständige Theorie. :citet:`Angelopoulos.Bates.2023` bieten eine leicht zugängliche Einführung mit Code für Klassifikation und Regression.

<!--en:69bbbe7a-->
### Konfidenzsequenzen

<!--en:4a8e3d01-->
Der $z$-Test aus :numref:`sec_mdl-statistics` setzt voraus, dass Sie den Stichprobenumfang $n$ festlegen, bevor Sie das erste Ergebnis sehen. Prüfen Sie ein laufendes Experiment dagegen nach jedem Batch und brechen ab, sobald $p < 0.05$ gilt, ist ein Fehlalarm bei hinreichend langer Beobachtung fast sicher. In Aufgabe :exref:`ex_mdl-statistics-peeking` konstruieren wir mit einem Gauß-Mischmartingal einen Test, der unabhängig von der Zahl der Zwischenprüfungen gültig bleibt. *Konfidenzsequenzen* übertragen diese Idee auf ganze Intervalle, die für alle Stichprobenumfänge $n$ gleichzeitig gelten. Sie erfassen beliebige beschränkte oder subgaußsche Daten und sind damit nicht auf ein einzelnes parametrisches Modell beschränkt. Der Preis für diese Freiheit sind breitere Intervalle. Bei den schärfsten Schranken wächst der zusätzliche Breitenfaktor allerdings bemerkenswert langsam, nur wie $\sqrt{\log\log n}$. :citet:`Howard.Ramdas.McAuliffe.ea.2021` konstruieren solche Sequenzen aus nichtnegativen Supermartingalen. :citet:`Johari.Koomen.Pekelis.ea.2017` beschreiben den Einsatz jederzeit gültiger $p$-Werte auf einer kommerziellen Plattform für A/B-Tests.

<!--en:1ba803d4-->
## Rechnen mit A-posteriori-Verteilungen

<!--en:c000dade-->
In :numref:`sec_mdl-bayesian-computation` haben wir mit einer einzigen Markov-Kette gearbeitet. Mit dem Metropolis-Algorithmus auf Basis einer Irrfahrt haben wir eine A-posteriori-Verteilung für zwei Parameter untersucht. Bayessche Modelle in der Praxis haben Tausende oder Millionen von Parametern. Dafür brauchen wir effizientere Sampling-Verfahren und Approximationen, die auch in dieser Größenordnung funktionieren.

<!--en:33fc1c76-->
### Gibbs-Sampling und Hamiltonian Monte Carlo

<!--en:4fe7ca5e-->
Beim Metropolis-Algorithmus mit Irrfahrt (:numref:`sec_mdl-bayes-mcmc`) müssen wir für jede Richtung die Skala der Vorschlagsverteilung abstimmen. Die Kette kommt nur in kleinen diffusionsartigen Schritten voran. Bei einer hochdimensionalen A-posteriori-Verteilung oder bei starken Korrelationen verbringt sie die meiste Zeit damit, hin und her zu wandern. Beim *Gibbs-Sampling* entfällt die Abstimmung. Wir aktualisieren jeweils eine Koordinate durch eine Ziehung aus ihrer exakten bedingten Verteilung. Damit wird jeder Schritt angenommen. Allerdings muss es leicht sein, aus diesen bedingten Verteilungen zu ziehen, und starke Korrelationen zwischen den Koordinaten bremsen das Verfahren weiterhin :cite:`Geman.Geman.1984`.

<!--en:ee0fee8d-->
*Hamiltonian Monte Carlo* orientiert sich an der Physik. Das Verfahren behandelt $-\log\tilde{p}$ als potentielle Energie, gibt einen zufälligen Anfangsimpuls vor und simuliert die resultierende Trajektorie mithilfe von Gradienten über viele Schritte. Da die Dynamik die Energie nahezu erhält, kann ein einziger Vorschlag einen weiten Weg durch die A-posteriori-Verteilung zurücklegen und dennoch mit hoher Wahrscheinlichkeit angenommen werden. :citet:`Neal.2011` führt anschaulich in das Verfahren und seine Abstimmung ein. :citet:`Hoffman.Gelman.2014` bestimmen mit dem No-U-Turn Sampler die Länge der Trajektorie automatisch. Stan verwendet diesen Sampler inzwischen standardmäßig.

<!--en:87c7a178-->
### A-posteriori-Verteilungen neuronaler Netze

<!--en:9d5b75e9-->
Jeder Algorithmus in :numref:`sec_mdl-bayesian-computation` wertet bei jedem Schritt die gemeinsame Log-Dichte für den gesamten Datensatz aus. Für die Laplace-Approximation müssen wir zudem eine $d \times d$ große Hesse-Matrix speichern und invertieren. Beides ist für ein neuronales Netz mit Millionen von Parametern und Millionen von Trainingsbeispielen nicht praktikabel. Die *Langevin-Dynamik mit stochastischen Gradienten* (engl. stochastic-gradient Langevin dynamics) baut auf dem Langevin-Schritt ohne Akzeptanzkorrektur aus Aufgabe :exref:`ex_mdl-bayesian-computation-mala` auf. Sie setzt Minibatch-Gradienten und eine abnehmende Schrittweite ein. Die Metropolis-Akzeptanzprüfung entfällt, dafür sind die Kosten pro Schritt genauso hoch wie beim gewöhnlichen SGD :cite:`Welling.Teh.2011`. Laplace-Approximationen lassen sich auf andere Weise auf große Netze übertragen. Man behält nur den aussagekräftigsten Teil der Krümmungsinformation, etwa pro Schicht einen in Kronecker-Faktoren zerlegten Block oder allein die Hesse-Matrix der letzten Schicht. :citet:`Daxberger.Kristiadi.Immer.ea.2021` zeigen, dass diese sparsamen Approximationen oft mit weit aufwendigeren bayesschen Methoden mithalten und nur einen Bruchteil des Rechenaufwands erfordern.

<!--en:2323a87b-->
## Konzentration und Lerntheorie

<!--en:ea66818a-->
Die ersten beiden Themen erweitern die Werkzeuge aus :numref:`sec_mdl-concentration-generalization` und :numref:`sec_mdl-generalization-bounds` über einfache Mittelwerte und die Rademacher-Komplexität hinaus. Die letzten beiden greifen die offene Frage vom Ende unserer theoretischen Abschnitte auf. Wie lassen sich Garantien für die konkrete Lösung beweisen, die ein Algorithmus tatsächlich findet, statt nur für die ungünstigste Funktion seiner Hypothesenklasse?

<!--en:e0b68a2d-->
### Beschränkte Differenzen und Lipschitz-stetige Funktionen

<!--en:f1d9fa26-->
Die Chernoff-Methode in :numref:`sec_mdl-concentration-generalization` setzt eine Summe unabhängiger Terme voraus. Viele für uns wichtige Größen hängen zwar von unabhängigen Eingaben ab, lassen sich aber nicht als einfache Summen schreiben. Ein Beispiel ist die größte Differenz zwischen wahrem und empirischem Risiko über eine ganze Modellklasse. Die *Ungleichung von McDiarmid* verallgemeinert die Hoeffding-Schranke auf solche Funktionen. An die Stelle der Annahme „Jeder Term liegt in einem Intervall der Länge $c_i$“ tritt die Annahme „Ändern wir nur die Eingabe $i$, so ändert sich der Funktionswert um höchstens $c_i$“. Dann gilt dieselbe Schranke für die Randwahrscheinlichkeit, $e^{-2t^2/\sum_i c_i^2}$. Für den Beweis ersetzt man das Produkt der momenterzeugenden Funktionen durch ein Martingal.

<!--en:616e9323-->
Für gaußsche Zufallsvektoren gilt eine noch einfachere Aussage. Bei einem standardnormalverteilten Vektor in beliebiger Dimension $d$ erfüllt jede $L$-Lipschitz-stetige Funktion $f$ die Ungleichung $P(|f - \mathbb{E}f| \ge t) \le 2e^{-t^2/(2L^2)}$. Da die euklidische Norm $1$-Lipschitz-stetig ist, bleiben ihre Schwankungen um den Radius der dünnen Kugelschale von der Größenordnung $1$, gleichgültig wie groß $d$ wird. :citet:`Boucheron.Lugosi.Massart.2013` behandeln beide Ergebnisse und die zugrunde liegenden Methoden ausführlich.

<!--en:9780de59-->
### VC-Dimension und Überdeckungszahlen

<!--en:30ae3ef8-->
In :numref:`sec_mdl-generalization-bounds` haben wir unendliche Hypothesenklassen mithilfe ihrer Rademacher-Komplexität untersucht. Der klassische Ansatz zählt, wie viele verschiedene Labelzuweisungen eine Klasse auf einer endlichen Punktmenge erzeugen kann. Angenommen, eine Familie binärer Klassifikatoren kann nie alle $2^{d+1}$ Labelzuweisungen realisieren, gleichgültig welche $d+1$ Punkte wir wählen (für lineare Halbräume in $\mathbb{R}^m$ gilt zum Beispiel $d = m + 1$). Nach dem Lemma von Sauer erzeugt die Klasse dann höchstens $(en/d)^d$ Labelzuweisungen auf beliebigen $n \ge d$ Punkten. Wenden wir auf diese Zuweisungen die boolesche Ungleichung an, erhalten wir eine Generalisierungslücke der Größenordnung $\sqrt{d \log n / n}$. Das ist das berühmte Argument von :citet:`Vapnik.Chervonenkis.1971`. Bei reellwertigen Funktionen überdeckt man die Klasse für jede Auflösung $\epsilon$ durch endlich viele Repräsentanten. Ein als *Chaining* bezeichnetes Verfahren summiert über die verschiedenen Skalen und beseitigt so den zusätzlichen Faktor $\log n$ :cite:`Vershynin.2018`.

<!--en:5a925ffc-->
### PAC-Bayes-Schranken

<!--en:f21e276c-->
Jede Schranke in :numref:`sec_mdl-generalization-bounds` sichert uns gegen die ungünstigste Funktion einer Hypothesenklasse ab, die schon vor der Sichtung der Daten feststeht. Eine *PAC-Bayes*-Schranke wechselt die Perspektive. Sie bewertet eine A-posteriori-Verteilung $Q$ über Funktionen. Diese Verteilung wählen wir *nach* Beobachtung der Daten. Als Vergleich dient eine vorab festgelegte A-priori-Verteilung $P$. An die Stelle des Terms für die Klassenkomplexität tritt dann $D_{\textrm{KL}}(Q\|P)/n$ unter einer Quadratwurzel :cite:`McAllester.1999`. Selbst bei einer riesigen Hypothesenklasse bleibt die Komplexitätsstrafe gering, wenn die A-posteriori-Verteilung nahe bei der A-priori-Verteilung liegt. Auf dieser Grundlage optimierten :citet:`Dziugaite.Roy.2017` einen Gauß-Posterior um die mit SGD gefundenen Gewichte. Für eine binäre MNIST-Aufgabe erhielten sie aussagekräftige Generalisierungsschranken für stochastische Netze mit weit mehr Parametern als Trainingsbeispielen.

<!--en:ac75992d-->
### Algorithmische Stabilität

<!--en:418d2157-->
Das Experiment mit zufälligen Labeln (:numref:`subsec_random_labels`) zeigt, dass die Klasse der Funktionen, die ein neuronales Netz darstellen *kann*, weitaus größer ist als die Teilmenge, die SGD bei realen Daten tatsächlich durchläuft. *Algorithmische Stabilität* liefert Schranken für den Lernalgorithmus selbst statt für die Hypothesenklasse. Angenommen, wir tauschen ein beliebiges einzelnes Trainingsbeispiel aus und der Verlust des gelernten Modells ändert sich dadurch an jedem Testpunkt um höchstens $\beta$. Dann ist auch die erwartete Generalisierungslücke von der Größenordnung $\beta$. Die Ungleichung von McDiarmid liefert daraus eine Schranke, die mit hoher Wahrscheinlichkeit gilt :cite:`Bousquet.Elisseeff.2002`. Für glatte, Lipschitz-stetige konvexe Verlustfunktionen ist SGD stabil. Dabei ist $\beta$ proportional zur Summe der Schrittweiten geteilt durch $n$. Weniger oder kleinere Trainingsschritte ergeben also engere Garantien :cite:`Hardt.Recht.Singer.2016`.

<!--en:c6669a52-->
## Klassifikatoren aus Häufigkeiten

<!--en:73a72916-->
Zum Schluss vertiefen wir zwei Aspekte aus :numref:`sec_mdl-naive_bayes`. Wie korrigieren wir schlecht kalibrierte Wahrscheinlichkeiten, wenn eine einzige Temperatur nicht ausreicht? Und wie kann ein generativer Klassifikator aus Daten ohne Label lernen? Bei einer rein diskriminativen Anpassung ist Letzteres nicht direkt möglich.

<!--en:2a6d198f-->
### Kalibrierung über eine einzige Temperatur hinaus

<!--en:32a74feb-->
Die Temperaturskalierung (Aufgabe :exref:`ex_mdl-naive-bayes-temperature-scaling`) funktioniert, wenn die Scores eines Klassifikators bis auf einen gemeinsamen multiplikativen Faktor stimmen. Bei komplexeren Verzerrungen helfen flexiblere Abbildungen. Die *Platt-Skalierung* passt sowohl die Steigung als auch den Achsenabschnitt einer logistischen Kurve an, die Scores für binäre Entscheidungen auf Wahrscheinlichkeiten abbildet :cite:`Platt.1999`. Bei der isotonen Regression lässt sich dagegen jede monoton nicht fallende Funktion anpassen. :citet:`Zadrozny.Elkan.2002` nutzten die isotone Regression, um Naive Bayes und andere Klassifikatoren zu kalibrieren. Hier gilt die bekannte Abwägung. Eine flexiblere Kalibrierung benötigt mehr zurückgehaltene Daten, um Überanpassung zu vermeiden. Bei modernen tiefen Netzen funktioniert eine einzige Temperatur oft ebenso gut :cite:`Guo.Pleiss.Sun.Weinberger.2017`.

<!--en:894a7b15-->
### Halbüberwachtes Lernen mit Naive Bayes

<!--en:a0547d40-->
In vielen praktischen Anwendungen sind Beispiele mit Label teuer, während Eingaben ohne Label reichlich vorhanden sind. Naive Bayes modelliert für jede Klasse, wie ihre Merkmale entstehen. Ein Beispiel ohne Label lässt sich daher in der Likelihood berücksichtigen, indem wir seine unbekannte Klasse als latente Variable behandeln. Der EM-Algorithmus aus :numref:`sec_mdl-latent-em-elbo` wechselt dann zwischen zwei Schritten. Zuerst berechnet er für die Beispiele ohne Label die A-posteriori-Wahrscheinlichkeiten der Klassen. Dann verwendet er diese Wahrscheinlichkeiten als anteilige Label, um die Wort- oder Pixelhäufigkeiten zu aktualisieren. :citet:`Nigam.McCallum.Thrun.ea.2000` erzielten so bei der Textklassifikation eine deutlich höhere Genauigkeit, wenn nur eine Handvoll Dokumente ein Label trug. Sie wiesen zugleich darauf hin, dass Daten ohne Label schaden können, wenn das generative Modell die Daten schlecht beschreibt.

<!--en:35e6f3e0-->
## Bücher und Lehrmaterialien zur Wahrscheinlichkeitstheorie und Statistik

<!--en:f3e47eb0-->
**Bücher**

<!--en:ea6b0504-->
- [Introduction to Probability – Blitzstein & Hwang](https://probabilitybook.net/) bietet einen zugänglichen Einstieg in Zufallsvariablen, Erwartungswert, Konditionierung und die in :numref:`sec_mdl-random_variables` und :numref:`sec_mdl-distributions` vorgestellten Verteilungen mit eigenen Namen. Zum Harvard-Kurs Stat 110 steht das Buch kostenlos als PDF bereit.

<!--en:9d567148-->
- [Mathematics for Machine Learning – Deisenroth, Faisal & Ong](https://mml-book.github.io/) gibt in Kapitel 6 einen in sich geschlossenen Überblick über die Grundlagen der Wahrscheinlichkeitstheorie und der Verteilungen, einschließlich der Exponentialfamilie aus :numref:`sec_mdl-exponential-family`.

<!--en:17001375-->
- [All of Statistics – Wasserman](https://www.stat.cmu.edu/~larry/all-of-statistics/) behandelt Schätzen, Testen, Konfidenzintervalle und das Bootstrap-Verfahren knapp und verständlich für Leser aus der Informatik. Das Buch deckt den Stoff aus :numref:`sec_mdl-statistics` und :numref:`sec_mdl-maximum_likelihood` ab. Die Begleitseite enthält Korrekturen, Code und Datensätze.

<!--en:44e50319-->
- [Computer Age Statistical Inference – Efron & Hastie](https://hastie.su.domains/CASI/) untersucht das Bootstrap-Verfahren, statistisches Testen im großen Maßstab und den Zusammenhang zwischen frequentistischer und bayesscher Inferenz. Damit ergänzt das Buch :numref:`sec_mdl-statistics`.

<!--en:88cd4628-->
- [Pattern Recognition and Machine Learning – Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) behandelt die Exponentialfamilie, Mischmodelle und den EM-Algorithmus (:numref:`sec_mdl-latent-em-elbo`) sowie die Laplace- und Variationsapproximationen aus :numref:`sec_mdl-bayesian-computation`. Microsoft Research stellt das Buch kostenlos als PDF zur Verfügung.

<!--en:6e73de1a-->
- [Probabilistic Machine Learning: An Introduction – Murphy](https://probml.github.io/pml-book/book1.html) erschließt das maschinelle Lernen über die Wahrscheinlichkeits- und Entscheidungstheorie. Zu den Themen gehören die MAP-Schätzung (:numref:`sec_mdl-maximum_likelihood`) und generative Klassifikatoren (:numref:`sec_mdl-naive_bayes`). Eine kostenlose Vorabfassung und ausführbare Notebooks stehen zur Verfügung.

<!--en:e2cdd7ce-->
- [Information Theory, Inference, and Learning Algorithms – MacKay](https://www.inference.org.uk/itila/book.html) verbindet Inferenz mit Informationstheorie. Die Kapitel zur Laplace-Methode, zu Monte-Carlo-Verfahren und zur Variationsinferenz begleiten :numref:`sec_mdl-bayesian-computation`.

<!--en:f053e7e6-->
- [Bayesian Data Analysis – Gelman et al.](https://sites.stat.columbia.edu/gelman/book/) behandelt A-priori- und A-posteriori-Verteilungen, Markov-Chain-Monte-Carlo samt diagnostischen Verfahren und das praktische Vorgehen bei bayesschen Analysen passend zu :numref:`sec_mdl-bayesian-computation`. Die dritte Auflage ist kostenlos als PDF erhältlich.

<!--en:69719ecd-->
- [High-Dimensional Probability – Vershynin](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html) vertieft den Stoff aus der Wahrscheinlichkeitstheorie mit Konzentrationsungleichungen, subgaußschen Zufallsvariablen und hochdimensionaler Geometrie und führt so über :numref:`sec_mdl-concentration-generalization` hinaus.

<!--en:62bd0c45-->
- [Understanding Machine Learning: From Theory to Algorithms – Shalev-Shwartz & Ben-David](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) leitet wie in :numref:`sec_mdl-generalization-bounds` die gleichmäßige Konvergenz, die Rademacher-Komplexität und Margin-Schranken her. Die Autoren stellen das Buch kostenlos als PDF zur Verfügung.

<!--en:51311a3d-->
**Kurse und Vorlesungsvideos**

<!--en:92bc4104-->
- [Statistics 110: Probability – Harvard (Blitzstein)](https://stat110.hsites.harvard.edu/) bietet Vorlesungsvideos, Aufgabenblätter und mehr als 250 Übungsaufgaben mit Lösungen zu :numref:`sec_mdl-random_variables` und :numref:`sec_mdl-distributions`.

<!--en:7a737a10-->
- [Vorlesungsvideos zu Stat 110 – YouTube](https://stat110.hsites.harvard.edu/youtube) zeigen die gesamte Vorlesungsreihe zu Wahrscheinlichkeit, Verteilungen und Konditionierung.

<!--en:613e8301-->
**Einführungen und Skripte**

<!--en:ea2da4c6-->
- [Probability Theory Review – Stanford CS229 (Maleki & Do)](https://cs229.stanford.edu/section/cs229-prob.pdf) wiederholt Zufallsvariablen, Erwartungswert, Varianz, Gauß-Verteilungen und multivariate Wahrscheinlichkeitsrechnung für das maschinelle Lernen. Das Skript dient als knappe Ergänzung zu :numref:`sec_mdl-random_variables`.

<!--en:30918443-->
[Diskussion](https://d2l.discourse.group/t/36864)
