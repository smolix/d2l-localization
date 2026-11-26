<!--en:abc623fb-->
# Naive Bayes

<!--en:2528f4d3-->
In :numref:`sec_softmax_concise` haben wir die Softmax-Regression auf Fashion-MNIST trainiert. Dazu haben wir den Gradientenabstieg direkt auf die Wahrscheinlichkeit des Labels bei gegebenem Bild angewandt. So erhielten wir 7850 Parameter (ein Gewicht für jedes Paar aus Pixel und Klasse sowie zehn additive Konstanten). Das Modell erreichte etwa 83% Genauigkeit und bestand die Prüfung der Kalibrierung aus :numref:`sec_classification`. Doch ein Klassifikator lässt sich auch ohne Gradientenabstieg entwickeln. Wir können beschreiben, wie jede Klasse ihre Bilder erzeugt, also wie ein typischer Mantel oder eine typische Sandale aussieht. Der Satz von Bayes kehrt diese Beschreibung um und liefert eine Vorhersage. Wie gut funktioniert dieser Ansatz, und was geben wir dafür auf?

<!--en:1430b057-->
Die erste Hürde ist die Zahl möglicher Bilder. Um das Aussehen von Mänteln zu beschreiben, müssen wir jedem möglichen Bild eine Wahrscheinlichkeit zuweisen. Selbst wenn wir jedes Pixel auf ein Bit reduzieren (gesetzt oder leer), gibt es weit mehr mögliche $28\times28$-Bilder als Atome im beobachtbaren Universum. Uns stehen aber nur 60 000 Trainingsbilder zur Verfügung. Der *Naive-Bayes*-Klassifikator umgeht diese Hürde mit einer einzigen gewagten Annahme: Sobald die Klasse bekannt ist, verrät uns kein Merkmal mehr etwas über ein anderes.

<!--en:50359633-->
Diese Idee geht auf Marons frühe Arbeiten zur Einordnung von Dokumenten anhand der darin vorkommenden Wörter zurück :cite:`Maron.1961`. Auf ihr beruhten die Spamfilter der 1990er-Jahre :cite:`Sahami.Dumais.Heckerman.ea.1998`. Unter der Unabhängigkeitsannahme braucht ein Spamfilter nur zu wissen, wie oft jedes Wort in Spam und in gewöhnlichen E-Mails vorkommt. Für einen Klassifikator für Kleidungsstücke genügt es zu wissen, wie oft jedes Pixel in jeder Klasse gesetzt ist. Für das Training reicht es zu zählen, für die Vorhersage zu multiplizieren.

<!--en:5dfbbf4b-->
Naive Bayes ist der einfachste *generative* Klassifikator. Ihm liegt ein Modell zugrunde, das beschreibt, wie jede Klasse ihre Eingaben erzeugt. Zum Anpassen genügt ein Durchlauf durch die Daten, ganz ohne Optimierer. Fehlen Merkmale, lässt der Klassifikator ihre Faktoren weg. Bei Bildern ist seine Unabhängigkeitsannahme offensichtlich falsch, wie bei den meisten realen Daten. Kein anderes Modell zeigt so deutlich, wie sich eine falsche Unabhängigkeitsannahme auf die Entscheidungen und Wahrscheinlichkeiten eines Klassifikators auswirkt.

<!--en:052c1db3-->
Dieser Abschnitt bildet die Fallstudie des Kapitels. Nach den Generalisierungsschranken aus :numref:`sec_mdl-generalization-bounds` ist die Mathematik hier einfach. Dafür kommen die Werkzeuge des Kapitels an einem konkreten Modell zusammen. Mit Maximum-Likelihood schätzen wir seine Parameter, mit einer Beta-A-posteriori-Verteilung glätten wir sie. Das Bootstrap-Verfahren liefert einen Fehlerbalken für die Genauigkeit, und anhand der beobachteten Häufigkeiten prüfen wir die Zuverlässigkeit der Wahrscheinlichkeiten.

<!--en:f2b9cc2a-->
Für die folgende Implementierung genügt NumPy. Das Deep-Learning-Framework liefert nur den Datenlader.

<!--en:f509a7a0-->
## Klassifikation mit dem Satz von Bayes

<!--en:baf7914a-->
Es gibt zwei Wege zur Wahrscheinlichkeit eines Labels bei gegebener Eingabe. Bisher haben wir diese Wahrscheinlichkeit direkt gelernt. Hier nehmen wir den Umweg über ein Modell der Eingaben. Wir sehen, warum dieser Weg im allgemeinen Fall nicht praktikabel ist und wie er durch die Annahme gangbar wird, der der Klassifikator seinen Namen verdankt.

<!--en:23886dc9-->
### Der generative Ansatz

<!--en:55333ae3-->
Ein Klassifikator ordnet ein Label $y\in\{1,\ldots,K\}$ einer Eingabe $\mathbf x$ zu. Kennten wir die wahre A-posteriori-Verteilung (engl. posterior distribution) $p(y\mid\mathbf x)$, also die Wahrscheinlichkeit jedes Labels bei gegebener Eingabe, wäre die beste Entscheidungsregel klar. Am häufigsten liegt richtig, wer das wahrscheinlichste Label wählt:

<!--en:2adc04ad-->
Für Fashion-MNIST gilt $K=10$. Halten wir für jedes der 784 Pixel fest, ob es gesetzt oder leer ist, dann ist $\mathbf x$ eines von $2^{784}$ binären Mustern. Wegen $784\log_{10}2=236.0$ entspricht das rund $10^{236}$ verschiedenen Bildern. Eine Tabelle für $p(y\mid\mathbf x)$ bräuchte für jedes Muster eine Zeile. Bei 60 000 Trainingsbildern blieben fast alle Zeilen leer.

<!--en:78c00ee4-->
Mit dem Satz von Bayes (:numref:`sec_mdl-joint-distributions`) schreiben wir die A-posteriori-Verteilung als $p(y\mid\mathbf x)=p(\mathbf x\mid y)\,p(y)/p(\mathbf x)$. Da der Nenner $p(\mathbf x)$ unabhängig von $y$ ist, hat er keinen Einfluss darauf, welche Klasse das Maximum erreicht:

<!--en:804afcdf-->
Für jede Klasse multipliziert der Klassifikator die Wahrscheinlichkeit des Bildes unter dieser Klasse mit der Wahrscheinlichkeit der Klasse insgesamt. Dieses Produkt $p(\mathbf x\mid y)\,p(y)$ nennen wir *Bayes-Zähler* (engl. Bayes numerator). Teilen wir die $K$ Zähler durch ihre Summe, so erhalten wir wieder den Nenner $p(\mathbf x)$ und damit die vollständige A-posteriori-Verteilung.

<!--en:5e9fcc3d-->
Damit ergeben sich für die Klassifikation zwei große Familien von Verfahren. Ein *generativer Klassifikator* modelliert, wie die Eingaben jeder Klasse entstehen. Er verbindet die klassenbedingte Verteilung $p(\mathbf x\mid y)$ mit der A-priori-Verteilung (engl. prior distribution) $p(y)$ und wendet dann den Satz von Bayes an. Ein *diskriminativer Klassifikator* modelliert $p(y\mid\mathbf x)$ direkt, ohne zu beschreiben, wie die Eingaben entstehen. Das typische Beispiel ist die Softmax-Regression (:numref:`sec_softmax`).

<!--en:d1524dfe-->
Hier liegt ein Einwand nahe: Der Satz von Bayes löst das Größenproblem nicht, denn $p(\mathbf x\mid y)$ ist immer noch eine Verteilung über $2^{784}$ Muster. Der Einwand ist berechtigt. Die Kettenregel der Wahrscheinlichkeitsrechnung zeigt genau, wo die kombinatorische Explosion entsteht. Zerlegen wir die Wahrscheinlichkeit Merkmal für Merkmal, so erhalten wir

<!--en:cd96d3d4-->
Jeder Faktor hängt von allen vorhergehenden Merkmalen ab. Allein der letzte Faktor benötigt daher eine Tabelle für $2^{d-1}$ mögliche Vorgeschichten. Im nächsten Unterabschnitt verzichten wir ganz auf diese Vorgeschichten.

<!--en:01f7eb83-->
### Die naive Annahme

<!--en:2922b056-->
Um $p(\mathbf x\mid y)$ aus einem Datensatz realistischer Größe zu schätzen, brauchen wir deutlich weniger Parameter. Die naive Annahme besagt Folgendes: Kennen wir die Klasse des Kleidungsstücks, so verrät uns die Beobachtung eines gesetzten Pixels nichts mehr über irgendein anderes Pixel. Die Merkmale sind also *bei gegebenem Label bedingt unabhängig* (engl. conditionally independent given the label). Damit entfällt in jedem Faktor der Kettenregel die Vorgeschichte:

<!--en:346e9efb-->
Bei binären Pixeln bestimmt eine einzige Zahl jeden Faktor: die Wahrscheinlichkeit, dass Pixel $i$ in Klasse $y$ gesetzt ist. Für ein uneingeschränktes $p(\mathbf x\mid y)$ brauchen wir $2^{784}-1$ freie Wahrscheinlichkeiten pro Klasse, für das faktorisierte Modell nur $784$. Jede dieser 784 Zahlen ist eine relative Häufigkeit. Mit 6000 Trainingsbildern pro Klasse schätzen wir sie mit einem Standardfehler von höchstens $0.5/\sqrt{6000}\approx0.0065$ (:numref:`sec_mdl-statistics`). Beide Modelle benötigen außerdem $K-1=9$ Zahlen für die A-priori-Verteilung der Klassen.

<!--en:56282045-->
Die Annahme betrifft alle $d$ Merkmale gemeinsam. Um sie zu überprüfen, genügt es nicht, Merkmalspaare zu betrachten. Seien innerhalb einer Klasse $x_1$ und $x_2$ unabhängige, gleichverteilte Bits. Dann hat $x_3=x_1\oplus x_2$ genau dann den Wert $1$, wenn genau eines der beiden Bits den Wert $1$ hat. Jedes Merkmalspaar ist unabhängig, doch die ersten beiden Merkmale legen das dritte vollständig fest. Daher hat das Muster $(1,1,1)$ die wahre Wahrscheinlichkeit $0$, während das Produkt der drei Randwahrscheinlichkeiten ihm $\tfrac18$ zuweist (Aufgabe :exref:`ex_mdl-naive-bayes-xor`).

<!--en:32dd066f-->
Stellen wir die Faktorisierung als gerichteten Graphen dar, so zeigt von jeder Variablen ein Pfeil zu den Variablen, die sie direkt beeinflusst. Es entsteht ein Stern: Das Label $y$ ist der Elternknoten aller Merkmale, und keine Kante verbindet die Merkmale untereinander (:numref:`fig_mdl-naive-independence`, links). Ein reichhaltigeres Modell, das bedingte Abhängigkeiten zwischen Merkmalen berücksichtigt, fügt solche Kanten hinzu (rechts).

<!--en:a7cfdb0c-->
![Strukturen bedingter Unabhängigkeit. Links: Bei Naive Bayes ist das Label $y$ ein Elternknoten jedes Merkmals. Keine Kante verbindet die Merkmale untereinander. Das entspricht $p(\mathbf x\mid y)=\prod_i p(x_i\mid y)$. Rechts: Ein Modell, das bedingte Abhängigkeiten zwischen Merkmalen berücksichtigt, fügt Kanten zwischen ihnen hinzu.](../img/mdl-prob-naive-independence.svg)

<!--en:20a644af-->
Setzen wir die Faktorisierung :eqref:`eq_mdl-naive_assumption` in den Satz von Bayes ein, erhalten wir die *Naive-Bayes-Regel* :cite:`Maron.1961`:

<!--en:9a362235-->
Für Bilder ist diese Annahme offensichtlich falsch. Benachbarte Pixel eines Kleidungsstücks sind gemeinsam gesetzt oder gemeinsam leer, weil sie zum selben Ärmel oder Saum gehören. Im Folgenden messen wir, wie stark Fashion-MNIST von dieser Annahme abweicht und wie sich das auf den Klassifikator auswirkt.

<!--en:635cadee-->
### Ein Spamfilter für zwei Wörter

<!--en:808a1cf3-->
Bevor wir zu 784 Pixeln übergehen, rechnen wir das kleinstmögliche Beispiel von Hand durch. Ein E-Mail-Filter prüft nur zwei binäre Merkmale: ob eine Nachricht das Wort „free“ enthält und ob sie „meeting“ enthält. Unser Trainingsdatensatz bestehe aus zehn Nachrichten (4 Spam-Nachrichten und 6 gewöhnlichen E-Mails) mit folgenden Häufigkeiten:

<!--en:d758ca69-->
| | Spam (4 E-Mails) | gewöhnlich (6 E-Mails) |

<!--en:98230d4c-->
| enthält „free“ | 3 | 1 |

<!--en:6dc72ec6-->
| enthält „meeting“ | 0 | 4 |

<!--en:d34c566d-->
Durch Zählen erhalten wir alle Zahlen, die das Modell benötigt. Die A-priori-Wahrscheinlichkeit beträgt $\hat p(\text{spam})=4/10=0.4$. Das Wort „free“ kommt in $3/4$ der Spam-Nachrichten und in $1/6$ der gewöhnlichen E-Mails vor. „meeting“ steht in $0/4$ der Spam-Nachrichten und in $4/6$ der gewöhnlichen E-Mails. Eine neue Nachricht enthalte nun „free“, aber nicht „meeting“. Ihre beiden Bayes-Zähler lauten

<!--en:df1d6b32-->
Daraus ergibt sich für Spam eine A-posteriori-Wahrscheinlichkeit von $0.30/(0.30+0.033)=0.90$. Noch übersichtlicher wird die Rechnung mit den Chancen (engl. odds). Jedes beobachtete Merkmal multipliziert die A-priori-Chancen mit seinem Likelihood-Verhältnis. Dieses ist die Wahrscheinlichkeit der Beobachtung bei Spam geteilt durch ihre Wahrscheinlichkeit bei einer gewöhnlichen E-Mail:

<!--en:92aa849d-->
Chancen von $9$ zu $1$ entsprechen einer Wahrscheinlichkeit von $9/10$. Unser Klassifikator für 784 Pixel führt genau dieselbe Rechnung durch, nur mit 784 Likelihood-Verhältnissen statt zwei. Wir kommen noch dreimal auf diesen kleinen Filter zurück: wenn eine Häufigkeit null ist, wenn wir zu Logarithmen übergehen und wenn wir ein Merkmal doppelt zählen.

<!--en:dcc5b874-->
## Lernen durch Zählen

<!--en:b7a14c19-->
Die Naive-Bayes-Regel benötigt die A-priori-Verteilung der Klassen und eine Wahrscheinlichkeit für jedes Paar aus Klasse und Merkmal. Woher kommen diese Schätzwerte? Wie kann ein Rechner 785 Wahrscheinlichkeiten multiplizieren, ohne dass das Ergebnis durch einen numerischen Unterlauf verloren geht?

<!--en:4b6feeb1-->
### Maximum-Likelihood und Glättung

<!--en:c50f2d7c-->
In unseren Spamfilter haben wir die beobachteten relativen Häufigkeiten eingesetzt. Maximum-Likelihood begründet diese Wahl und zeigt zugleich ihre Grenzen. Sei $\theta_{iy}=p(x_i=1\mid y)$ die Wahrscheinlichkeit, dass Merkmal $i$ in Klasse $y$ aktiv ist. Klasse $y$ komme $n_y$-mal unter $n$ Trainingsbeispielen vor, und Merkmal $i$ sei in $n_{iy}$ dieser Beispiele aktiv. Mit der Naive-Bayes-Faktorisierung zerfällt die Log-Likelihood des gesamten Trainingsdatensatzes in einen Term für die A-priori-Verteilung der Klassen und je einen unabhängigen Term für jedes Paar aus Klasse und Merkmal:

<!--en:ab49bbbb-->
Jeder Klammerausdruck ist die Log-Likelihood dafür, $n_{iy}$-mal Kopf bei $n_y$ Würfen mit einer Münze zu beobachten, deren Wahrscheinlichkeit für Kopf $\theta_{iy}$ beträgt. Wir können also jeden Parameter einzeln schätzen. Der Maximum-Likelihood-Schätzer für die Wahrscheinlichkeit von Kopf ist die beobachtete relative Häufigkeit (:numref:`sec_mdl-maximum_likelihood`). Damit erhalten wir $\hat\theta_{iy}=n_{iy}/n_y$ und $\hat p(y)=n_y/n$.

<!--en:e88fffc2-->
Ungeglättete Häufigkeiten haben allerdings einen gravierenden Nachteil. Kommt Merkmal $i$ in keinem Trainingsbeispiel der Klasse $y$ vor, so gilt $\hat\theta_{iy}=0$. Jedes künftige Beispiel mit aktivem Merkmal $i$ erhält dann den Bayes-Zähler $0$ für Klasse $y$, unabhängig davon, was alle anderen Merkmale aussagen. Ein einziges Merkmal schließt die Klasse aus. Unser Spamfilter für zwei Wörter enthält bereits eine solche Null: „meeting“ kam in keiner der vier Spam-Nachrichten im Training vor. Eine Nachricht mit „free“ und „meeting“ erhält daher für Spam den Zähler $0.4\cdot\frac34\cdot0=0$ und gilt mit völliger Sicherheit als gewöhnliche E-Mail. Weiter unten prüfen wir, wie oft ein Pixel in einer bestimmten Klasse von Fashion-MNIST nie gesetzt ist.

<!--en:8a6cb7da-->
Das übliche Gegenmittel ist die *Laplace-Glättung* (engl. Laplace smoothing) :cite:`Laplace.1814`. Wir ergänzen ein gedachtes Beispiel mit aktivem und eines mit inaktivem Merkmal und erhalten

<!--en:042dae21-->
Dadurch rückt jeder Schätzwert ein wenig in Richtung $\tfrac12$. Zugleich ist keine geschätzte Wahrscheinlichkeit jemals $0$ oder $1$.

<!--en:eb1e1212-->
Die Laplace-Glättung lässt sich auch bayessch deuten. Bei gleichförmiger A-priori-Verteilung auf $\theta$ ergeben $n_{iy}$ beobachtete Erfolge in $n_y$ Versuchen die A-posteriori-Verteilung $\mathrm{Beta}(n_{iy}+1,\,n_y-n_{iy}+1)$. Ihr Mittelwert ist genau $(n_{iy}+1)/(n_y+2)$ (:numref:`sec_mdl-exponential-family`). Laplace nannte dies die Sukzessionsregel.

<!--en:178e0bd3-->
Bei unserem Spamfilter ändern sich durch die Glättung die geschätzten Wahrscheinlichkeiten für „free“ und „meeting“ auf $4/6$ und $1/6$ bei Spam und auf $2/8$ und $5/8$ bei gewöhnlichen E-Mails. Die A-priori-Wahrscheinlichkeit bleibt bei $0.4$. Die Nachricht mit beiden Wörtern hat nun die Bayes-Zähler $0.4\cdot\frac46\cdot\frac16=0.044$ und $0.6\cdot\frac28\cdot\frac58=0.094$. Daraus ergibt sich eine Spam-Wahrscheinlichkeit von $0.32$ statt $0$. Bei der ersten Nachricht („free“ ohne „meeting“) sinkt sie von $0.90$ auf $0.80$. Vier Spam-Nachrichten liefern schließlich nur einen schwachen Beleg dafür, dass Spam nie ein Treffen erwähnt.

<!--en:f8975eb4-->
### Scores im logarithmischen Raum

<!--en:6b883235-->
Auch mit Glättung multiplizieren wir bei direkter Auswertung der Naive-Bayes-Regel $d+1$ Zahlen zwischen $0$ und $1$. Bei einem Bild ist das Produkt verschwindend klein. Wäre jeder Faktor gleich $\tfrac12$, ergäbe das Produkt aus 784 Faktoren $2^{-784}\approx10^{-236}$. Die kleinste positive Gleitkommazahl bei einfacher Genauigkeit beträgt etwa $1.4\cdot10^{-45}$, bei doppelter Genauigkeit etwa $4.9\cdot10^{-324}$. Bei einfacher Genauigkeit führt der Unterlauf daher unmittelbar zu $0$. Bei doppelter Genauigkeit lässt sich $2^{-784}$ noch darstellen. Bei einigen der realen Bildscores, die wir unten berechnen, liegen die Werte jedoch nahe bei $10^{-1000}$.

<!--en:58741644-->
Durch Logarithmieren wird das Produkt zur Summe. Da $\log$ streng monoton wächst, bleibt die Klasse mit dem größten Wert dieselbe (:numref:`sec_mdl-numerical-stability-conditioning` behandelt dieses Verfahren allgemein). Definieren wir den *Klassenscore* (engl. class score) $s_y(\mathbf x)$ als Logarithmus des Bayes-Zählers, so lautet die Entscheidungsregel

<!--en:71226b70-->
An die Stelle eines Produkts aus 785 Wahrscheinlichkeiten tritt eine Summe aus 785 Logarithmen. Jeder Summand bleibt endlich, solange keine Wahrscheinlichkeit gleich $0$ ist. Dafür sorgt die Glättung. Beim ungeglätteten Spamfilter lauten die logarithmischen Scores für die erste Nachricht $\log0.30=-1.20$ und $\log\frac1{30}=-3.40$. Ihre Differenz $2.20=\log 9$ ist der Logarithmus der A-posteriori-Chancen, die wir von Hand berechnet haben.

<!--en:2f2c76ae-->
### Lineare Entscheidungsgrenzen

<!--en:73adb544-->
Für binäre Merkmale nimmt der Klassenscore eine vertraute Form an. An ihr erkennen wir, welche Klassifikatoren Naive Bayes darstellen kann. Ein binäres Merkmal trägt $\log\theta_{iy}$ bei, wenn es aktiv ist, und $\log(1-\theta_{iy})$, wenn es inaktiv ist. Also gilt $\log p(x_i\mid y)=x_i\log\theta_{iy}+(1-x_i)\log(1-\theta_{iy})$. Fassen wir die Terme mit dem Faktor $x_i$ und diejenigen ohne diesen Faktor jeweils zusammen, erhalten wir

<!--en:2a520fa6-->
Jedes gesetzte Pixel trägt $w_{iy}$ bei, die Log-Odds dafür, dass Pixel $i$ in Klasse $y$ gesetzt ist. Die additive Konstante $b_y$ ist der Score eines vollständig leeren Bildes. Jeder Klassenscore ist affin in $\mathbf x$. Die Entscheidungsgrenze zwischen je zwei Klassen, an der ihre Scores übereinstimmen, ist also eine Hyperebene :cite:`Bishop.2006`. Damit haben wir das erste von zwei besonders wichtigen Ergebnissen dieses Abschnitts: Bernoulli-Naive-Bayes ist ein linearer Klassifikator aus genau derselben Familie wie die Softmax-Regression. Selbst die Parameterfelder haben dieselbe Form ($784\times10$ Gewichte und $10$ additive Konstanten). Die beiden Verfahren unterscheiden sich nur darin, wie sie diese Zahlen bestimmen: durch Zählen von Häufigkeiten oder durch Gradientenabstieg auf $p(y\mid\mathbf x)$. Bei zwei Klassen ergibt sich die A-posteriori-Wahrscheinlichkeit aus der Sigmoidfunktion der Scoredifferenz, $p(y=1\mid\mathbf x)=\operatorname{sigmoid}\bigl((\mathbf w_1-\mathbf w_0)^\top\mathbf x+b_1-b_0\bigr)$. Diese Form stimmt mit der logistischen Regression überein (Aufgabe :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq`).

<!--en:0ead35cf-->
Ob die Grenze linear bleibt, hängt von der Merkmalsverteilung ab. Gauß-Naive-Bayes modelliert jedes stetige Merkmal $x_i$ in Klasse $y$ durch eine Gauß-Verteilung mit Mittelwert $\mu_{iy}$ und Varianz $\sigma_{iy}^2$. Sein Beitrag zum Klassenscore ist $-(x_i-\mu_{iy})^2/(2\sigma_{iy}^2)$. Hat jedes Merkmal in allen Klassen dieselbe Varianz, sind die $x_i^2$-Terme in allen Scores identisch. Sie fallen bei jedem paarweisen Vergleich weg, und die Grenze bleibt linear. Bei klassenspezifischen Varianzen bleiben die $x_i^2$-Terme erhalten. Die Grenze wird quadratisch in jedem $x_i$, enthält aber weiterhin keine gemischten Terme $x_ix_j$ (Aufgabe :exref:`ex_mdl-naive-bayes-gaussian-boundaries`).

<!--en:1a697642-->
Am Beispiel der Gauß-Verteilung können wir veranschaulichen, wie wichtig das Schätzverfahren ist. In :numref:`fig_mdl-naive-genvdisc` sind die beiden Merkmale innerhalb jeder Klasse stark korreliert. Beide Punktwolken verlaufen deshalb schräg. Naive Bayes passt für jede Klasse eine achsenparallele Gauß-Verteilung an (gleiche Varianzen in beiden Klassen, also eine Gerade als Grenze). Die achsenparallelen Ellipsen berücksichtigen die Neigung nicht. Die resultierende Grenze schneidet beide Wolken, und 19% der Punkte werden falsch klassifiziert. Rechts sehen wir die lineare Grenze, die wir erhalten, wenn wir $p(y\mid\mathbf x)$ mit genügend Daten direkt anpassen. Sie folgt der Neigung und erreicht eine Fehlerquote von 7%. Das ist für dieses Problem optimal: Kein Klassifikator kann besser abschneiden.

<!--en:fb6999d3-->
![Zwei Wege zu einem linearen Klassifikator auf denselben Daten. Die beiden Merkmale sind innerhalb jeder Klasse korreliert. Links: Das naive generative Modell beschreibt jede Klasse durch eine Gauß-Verteilung mit unabhängigen Merkmalen (Ellipsen bei einer und zwei Standardabweichungen). Die Grenze folgt aus dem Satz von Bayes. Rechts: die direkt an $p(y\mid\mathbf x)$ angepasste lineare Grenze. Die Fehlerquoten gelten für die Grundgesamtheit bei gleichen A-priori-Wahrscheinlichkeiten.](../img/mdl-prob-naive-genvdisc.svg)

<!--en:06059a11-->
Historisch war die Textverarbeitung das wichtigste Einsatzgebiet von Naive Bayes. Ein echter Spamfilter erweitert unser Beispiel von zwei auf Zehntausende von Wörtern :cite:`Manning.Raghavan.Schutze.2008`. Ein Dokument lässt sich dadurch darstellen, welche Wörter vorkommen oder wie oft jedes Wort vorkommt. In beiden Fällen ist der Klassenscore linear. Hier bleiben wir bei Pixeln. In Aufgabe :exref:`ex_mdl-naive-bayes-two-event-models` entwickeln Sie beide Ereignismodelle für Texte und sehen, wie sie beim selben Dokument zu unterschiedlichen Entscheidungen kommen können.

<!--en:6108c417-->
## Fashion-MNIST durch Zählen klassifizieren

<!--en:8af2e41b-->
Einen Spamfilter für zwei Wörter können wir leicht von Hand überprüfen. Ob bloßes Zählen auch bei 784 offensichtlich korrelierten Pixeln funktioniert, lässt sich nur durch Ausprobieren klären. Mit Fashion-MNIST können wir das Ergebnis direkt mit dem der Softmax-Regression auf denselben Bildern vergleichen.

<!--en:392ba6f4-->
### Binäre Pixel

<!--en:7a702270-->
Fashion-MNIST :cite:`Xiao.Rasul.Vollgraf.2017` umfasst 60 000 Trainingsbilder und 10 000 Testbilder aus zehn Kleidungsklassen. Jedes Bild ist ein Graustufenraster der Größe $28\times28$ (:numref:`sec_fashion_mnist`). Hintergrundpixel haben stets den Wert $0$. Wir können daher jedes Bild in einen binären Vektor umwandeln: Wir setzen $x_i=1$, wo Pixel $i$ positiv ist (Teil des Kleidungsstücks), und $x_i=0$ für den Hintergrund. Der Datenlader liefert die Rohbilder für beide Teildatensätze. Wir formen sie in Vektoren mit je 784 binären Merkmalen um.

<!--en:c352e32b-->
Etwa die Hälfte aller Pixel ist gesetzt. Zu jeder Klasse gehören 6000 Trainingsbilder und 1000 Testbilder. Die geschätzte A-priori-Wahrscheinlichkeit beträgt daher einheitlich $0.1$ und trägt zu jedem Klassenscore dieselbe Konstante bei. Wir behalten den A-priori-Term im Code bei, damit die Implementierung allgemein einsetzbar bleibt.

<!--en:83adc49d-->
### Klassenvorlagen und generierte Bilder

<!--en:6947299e-->
Für das Training müssen wir nur zwei Dinge zählen: die Bilder jeder Klasse, $n_y$, und für jedes Pixel die Bilder dieser Klasse, in denen es gesetzt ist, $n_{iy}$. Die folgende Zelle speichert die mit Laplace-Glättung berechneten Schätzwerte $\hat\theta_{iy}$ in `P_xy`, mit einer Zeile pro Klasse. Zugleich beantwortet sie unsere praktische Frage zur Glättung: Wie viele Pixel sind in den Trainingsbildern einer Klasse nie gesetzt, und in wie vielen Testbildern ist eines davon gesetzt?

<!--en:5a5dd9f8-->
Nullhäufigkeiten sind hier recht selten. Vier der zehn Klassen haben überhaupt keine, bei Hosen sind es 13 und bei Turnschuhen 57 (Turnschuhe sind flach, sodass der obere Bildrand leer bleibt). Im gesamten Testdatensatz gibt es nur 5 Bilder mit einem gesetzten Pixel, das in ihrer wahren Klasse im Training immer leer war. Ohne Glättung erhielte jedes dieser fünf Bilder für seine wahre Klasse den Score $-\infty$. Es würde falsch klassifiziert, selbst wenn alle anderen 783 Pixel klar für diese Klasse sprächen.

<!--en:2ca4132a-->
Jede Zeile von `P_xy` gibt für jedes Pixel den Anteil der Trainingsbilder an, in denen es gesetzt ist. Formen wir die zehn Zeilen wieder zu $28\times28$-Bildern um, sehen wir, was das Modell für jede Kleidungsklasse gelernt hat.

<!--en:03c0b14b-->
Die resultierenden Vorlagen zeigen weiche, verschwommene Umrisse der zehn Kleidungsstücke. Jede Vorlage ist der pixelweise Mittelwert ihrer Klasse. Die Unschärfe spiegelt also echte Unterschiede in Form und Lage der Kleidungsstücke wider. Jedes Modell, das die Häufigkeiten der einzelnen Pixel in den Daten wiedergibt, erzeugt genau dieselben Vorlagen, ob es nun Unabhängigkeit annimmt oder nicht. Die Unschärfe kommt aus den Daten. Die naive Annahme besagt darüber hinaus, dass diese Mittelwerte zusammen mit der A-priori-Verteilung alles Wissenswerte enthalten. Um zu sehen, was das bedeutet, ziehen wir synthetische Bilder aus dem Modell. Für jedes Pixel werfen wir unabhängig eine Münze mit Erfolgswahrscheinlichkeit $\hat\theta_{iy}$.

<!--en:af45edc5-->
Jedes generierte Bild trifft die Silhouette im Großen und Ganzen. Seine Ränder sind jedoch ausgefranst und gesprenkelt, weil über jedes Randpixel ein eigener, unabhängiger Münzwurf entscheidet. Diesen sichtbaren Unterschied können wir direkt messen. Gehen wir eine Zeile eines echten Bildes entlang, wechseln gesetzte Pixel und Hintergrund nur wenige Male. In einem generierten Bild sollten sie viel häufiger wechseln. Im Folgenden vergleichen wir den Gesamtanteil gesetzter Pixel und die mittlere Anzahl der Wechsel zwischen gesetzten Pixeln und Hintergrund pro Zeile. Dazu betrachten wir den Trainingsdatensatz und 1000 generierte Bilder pro Klasse.

<!--en:cfb189a6-->
In den generierten Bildern ist insgesamt derselbe Anteil an Pixeln gesetzt wie in den echten Daten. Das muss so sein, weil jedes Pixel die richtige Randwahrscheinlichkeit hat. Die Bilder wechseln aber viel häufiger zwischen gesetzten Pixeln und Hintergrund: bei Pullovern rund 5 statt 3 Wechsel pro Zeile, bei Sandalen und Taschen etwa 8 statt 3. Echte Kleidungsstücke haben glatte Konturen, weil benachbarte Pixel meist übereinstimmen. Genau diese lokale Übereinstimmung geht mit der Unabhängigkeitsannahme verloren.

<!--en:0ab99c4f-->
### Überprüfung der Annahme

<!--en:6bcb9b92-->
Die generierten Bilder zeigen, wie die Unabhängigkeitsannahme im Bild als Ganzem scheitert. An einem einzigen Pixelpaar lässt sich das mit konkreten Zahlen belegen. Betrachten wir zwei benachbarte Pixel am unteren Saum von Pullovern, in Zeile 26 an den Spalten 14 und 15. Wären die beiden Pixel bei gegebener Klasse bedingt unabhängig, so wäre der Anteil der Pullover mit zwei gesetzten Pixeln gleich dem Produkt ihrer einzelnen relativen Häufigkeiten. Vergleichen wir dieses Produkt mit der tatsächlich beobachteten gemeinsamen Häufigkeit bei den Pullovern im Trainingsdatensatz und bei den 1000 generierten Pullovern.

<!--en:f62c59a8-->
Jedes der beiden Pixel ist bei rund 55% der Pullover gesetzt. Bei Unabhängigkeit sollten beide bei $0.55\times0.55\approx30\%$ der Pullover gemeinsam gesetzt sein. In den echten Daten sind bei 54% der Pullover beide gesetzt, fast so oft wie jedes Pixel für sich. Der Saum eines Pullovers reicht entweder bis zur Zeile 26 oder endet davor. Die beiden Nachbarpixel verhalten sich daher fast wie ein einziges Merkmal, das wir zweimal erfassen. Bei den generierten Bildern entspricht die gemeinsame Häufigkeit konstruktionsbedingt dem Produkt.

<!--en:58e1c0c4-->
### Genauigkeit auf dem Testdatensatz

<!--en:7f16ad43-->
Zur Klassifikation des Testdatensatzes verwenden wir die affine Form :eqref:`eq_mdl-naive_bayes_affine`. Wir berechnen die Gewichte $w_{iy}$ und die additiven Konstanten $b_y$ aus `P_xy` und `P_y`. Eine einzige Matrizenmultiplikation liefert für jedes Testbild die Scores aller zehn Klassen. Die Vorhersagen speichern wir für die folgende Analyse. Die Softmax-Regression erreichte mit Scores genau dieser Form etwa 83% Genauigkeit. Wie viel Genauigkeit kostet uns das Zählen?

<!--en:e53b6e8d-->
Naive Bayes klassifiziert etwa 71% der Testbilder richtig, deutlich mehr als die 10% beim zufälligen Raten. Da beim Zählen der Zufall keine Rolle spielt, erhalten wir bei jedem Durchlauf genau denselben Wert. Die Unsicherheit aufgrund der Stichprobenauswahl geben wir gleich mit einem Fehlerbalken an.

<!--en:17c62bb0-->
Ebenso aufschlussreich ist die Trainingsgenauigkeit. Auf den 60 000 Bildern, an die wir das Modell angepasst haben, liegt es in 72% der Fälle richtig. Das ist nur ein Prozentpunkt mehr als bei unbekannten Testbildern. Bei Überanpassung an den Trainingsdatensatz wäre der Abstand groß. Hier ist er kaum zu erkennen. Die Fehler entstehen also nicht durch einen Mangel an Trainingsdaten. Das Modell macht auf seinem Trainingsdatensatz dieselben Fehler wie bei neuen Bildern, weil seine Annahmen zu restriktiv sind. In der Sprache von :numref:`sec_mdl-statistics` dominiert beim Fehler die Verzerrung gegenüber der Varianz. Zusätzliche Bilder würden kaum etwas ändern. Wir brauchen bessere Parameter. Der folgende Vergleich mit der Softmax-Regression bestätigt, dass dieselbe Modellfamilie einen wesentlich besseren linearen Klassifikator enthält.

<!--en:245ffa89-->
Mit den logarithmischen Scores können wir auch prüfen, was bei direkter Auswertung des Produkts :eqref:`eq_mdl-naive_bayes` geschehen wäre. Jeder Score ist der natürliche Logarithmus eines Bayes-Zählers. Teilen wir ihn durch $\log 10$, erhalten wir dessen Exponenten zur Basis 10. Vergleichen wir die Zähler mit der kleinsten positiven Zahl bei einfacher Genauigkeit und der kleinsten *normalisierten* Zahl bei doppelter Genauigkeit. Unterhalb dieser Grenze lässt die Genauigkeit auch bei doppelter Genauigkeit nach.

<!--en:e62c2add-->
Die Bayes-Zähler reichen ungefähr von $10^{-32}$ bis hinunter zu $10^{-1000}$. Bei einem typischen Bild liegt der Zähler der siegreichen Klasse nahe $10^{-81}$. Bei 96% der Testbilder ist selbst der größte der zehn Zähler kleiner als die kleinste positive Gleitkommazahl bei einfacher Genauigkeit. Direkt berechnete Produkte ergäben zehn Nullen und erlaubten für diese Bilder überhaupt keine Klassifikation. Bei doppelter Genauigkeit liegen 44% aller Zähler unter der kleinsten normalisierten float64-Zahl. Fast alle gehören jedoch zu unterlegenen Klassen. Bis auf drei von zehntausend Bildern bleibt der größte Zähler darstellbar. Die Genauigkeit des direkt berechneten float64-Produkts weicht daher um höchstens ein Bild je zehntausend vom Ergebnis der logarithmischen Rechnung ab. Selbst doppelte Genauigkeit bietet bei 784 Pixeln nur wenig Spielraum: Bei $10^{-81}$ für einen typischen siegreichen Score käme es mit einer viermal so langen Eingabe auch bei float64 zum Unterlauf. Im logarithmischen Raum bleiben alle Scores problemlos im üblichen Gleitkommabereich.

<!--en:98467770-->
## Auswertung des Klassifikators

<!--en:38e2726e-->
Die Genauigkeit von 71% wirft drei Fragen auf. Wie präzise ist dieser Schätzwert? Welche Verwechslungen erklären die übrigen 29%? Und können wir den Wahrscheinlichkeiten vertrauen, die der Klassifikator angibt?

<!--en:205f914b-->
### Fehlerbalken

<!--en:f2fc7cf4-->
Unsere Testgenauigkeit ist ein Mittelwert über 10 000 zufällige Testbilder. Eine andere Teststichprobe ergäbe daher einen etwas anderen Wert. Jedes Testbild liefert ein einziges Bernoulli-Ergebnis: richtig oder falsch. Das entspricht dem Münzwurf vom Anfang des Kapitels, mit einem Wurf pro Testbild. Die beobachtete Genauigkeit $\hat p$ ist also der Stichprobenmittelwert aus $n$ Würfen und hat den Standardfehler $\sqrt{\hat p(1-\hat p)/n}$ (:numref:`sec_mdl-statistics`). Für $\hat p=0.706$ und $n=10{,}000$ gilt

<!--en:7824612d-->
Damit erhalten wir ein 95%-Konfidenzintervall, das in beide Richtungen etwa $0.9$ Prozentpunkte reicht. Das Bootstrap-Verfahren (:numref:`sec_mdl-statistics`) bietet eine zweite Prüfung ohne Formel: Wir ziehen Testbilder mit Zurücklegen, berechnen für jede so gewonnene Stichprobe die Genauigkeit neu und nehmen die mittleren 95% der resultierenden Verteilung.

<!--en:d0c9c079-->
Das Bootstrap-Intervall reicht von etwa 0.70 bis 0.715. Das passt zur halben Breite $0.009$ des binomialen Intervalls. Wir sollten die Genauigkeit daher als $71\%\pm0.9$ Punkte angeben. Die Nachkommastellen in „70.59%“ sind dem Stichprobenzufall geschuldet. Vergleichen wir zwei Klassifikatoren auf genau demselben Testdatensatz, so sind ihre Fehler korreliert. Eine gepaarte Analyse der Differenzen pro Bild ist daher wesentlich präziser als der Vergleich zweier getrennter Konfidenzintervalle (Aufgabe :exref:`ex_mdl-naive-bayes-paired-comparison`).

<!--en:1bf52104-->
### Verwechslungen und Gewichte

<!--en:b93f8ffb-->
Die Gesamtgenauigkeit sagt uns, wie oft der Klassifikator irrt. Die Konfusionsmatrix (engl. confusion matrix) aus :numref:`sec_classification` verrät uns, *wie* er irrt. Wie in :numref:`sec_softmax_scratch` stehen die Zeilen für vorhergesagte Klassen und die Spalten für wahre Klassen. An jeder Spalte lässt sich also ablesen, welchen Klassen der Klassifikator die 1000 Testbilder eines bestimmten Kleidungsstücks zuordnet.

<!--en:92025f49-->
Eine $10\times10$-Häufigkeitstabelle lässt sich leichter lesen, wenn wir die häufigsten Fehler herausgreifen. Unten geben wir die sechs größten Einträge außerhalb der Diagonalen aus, jeweils in der Reihenfolge „wahre Klasse, vorhergesagte Klasse“. Dazu kommt die Genauigkeit jeder einzelnen Klasse, also ihr Diagonaleintrag geteilt durch ihre Spaltensumme.

<!--en:dc9bf60b-->
Die Fehler konzentrieren sich auf wenige Klassenpaare. Mäntel werden 372-mal als Pullover klassifiziert und Pullover 242-mal als Mäntel. Hemden werden jeweils mehr als 200-mal für Pullover, Mäntel oder T-Shirts gehalten und Sandalen 232-mal für Turnschuhe. Hosen (91%) und Turnschuhe (89%) sind leicht zu erkennen, Mäntel (48%) und Hemden (17%) deutlich schwerer. Genau diese beiden Gruppen bereiteten der Softmax-Regression in :numref:`sec_softmax_scratch` Schwierigkeiten: die vier Oberbekleidungsarten und die drei Schuharten. Bei der Oberbekleidung schneidet das durch Zählen angepasste Modell aber wesentlich schlechter ab. Die Softmax-Regression erkannte Hemden in 40% bis 60% der Fälle, Naive Bayes erkennt nur etwa jedes sechste.

<!--en:1c9cad54-->
Sind 71% bei binären Pixeln die Obergrenze für einen linearen Klassifikator? Die Softmax-Regression definiert Hyperebenen genau derselben Form, wählt ihre Gewichte aber durch Minimierung der Kreuzentropie von $p(y\mid\mathbf x)$. Trainieren wir die Softmax-Regression mit stochastischem Gradientenabstieg auf Minibatches derselben binären Pixel. So unterscheidet sich nur das Schätzverfahren. Für die Aktualisierung nutzen wir das Ergebnis aus :numref:`sec_softmax`: Der Gradient der Kreuzentropie bezüglich der Logits ist der vorhergesagte Wahrscheinlichkeitsvektor abzüglich des One-Hot-Labels.

<!--en:e36a88c8-->
Die Softmax-Regression erreicht etwa 83% Genauigkeit, zwölf Prozentpunkte mehr als Naive Bayes, und verwechselt Pullover und Mäntel nur etwa halb so oft. Die lineare Modellfamilie selbst ist also kein Flaschenhals. Der Unterschied liegt allein darin, wie das Zählen die Gewichte festlegt. Warum es diese Gewichte verzerrt, sehen wir gleich bei der Prüfung der Kalibrierung.

<!--en:3165f0cf-->
Betrachten wir zunächst die Gewichte von Naive Bayes. Sie zeigen, mit welchen Merkmalen der Klassifikator zwei Klassen unterscheidet. Die Scoredifferenz zwischen Pullover und Mantel ist eine affine Funktion mit Gewichtsvektor $\mathbf w_{\text{pullover}}-\mathbf w_{\text{coat}}$. Ist Pixel $i$ gesetzt statt leer, verschiebt sich der Score zugunsten von Pullover um den Eintrag $i$ dieses Vektors. Stellen wir den Differenzvektor als $28\times28$-Bild dar. Rot spricht für Pullover, Blau für Mantel. Außerdem geben wir den größten Eintrag zusammen mit den zugrunde liegenden Häufigkeiten aus.

<!--en:5adc5400-->
Rote Pixel verlaufen an beiden Seiten des Bildes und entlang der oberen Zeilen. Dort sind bei Pullovern häufiger Pixel gesetzt als bei Mänteln. Der Rumpf ist mit Gewichten nahe $-1$ blau gefärbt. Daraus zu schließen, ein gesetztes blaues Pixel spreche stark für einen Mantel, wäre jedoch falsch. Jedes Gewicht misst den Score eines gesetzten Pixels *relativ* zu einem leeren. Ein Rumpfpixel ist bei rund 99% der Pullover und mehr als 99.5% der Mäntel gesetzt. Ist es gesetzt, ändert das den Score kaum, nämlich um etwa $-0.01$ Nat. Das Gewicht von $-1$ erklärt sich aus dem seltenen Fall eines leeren Rumpfpixels: Dann erhält der Pullover einen entsprechenden Vorteil. In der Praxis fällt die Entscheidung an den Seiten. Das größte Einzelgewicht von rund 3 Nat liegt hingegen in einer Ecke, die bei etwa zwanzig Pullovern und keinem Mantel gesetzt ist. Seine Größe hängt also hauptsächlich von der Laplace-Pseudohäufigkeit ab und stützt sich nur auf wenige Daten.

<!--en:df319108-->
Bevor wir die generative Anpassung näher untersuchen, betrachten wir einen echten Vorteil gegenüber einem diskriminativen Modell. Fehlen beim Test einige Pixel, etwa weil sie verdeckt sind, lässt Naive Bayes ihre Faktoren im Produkt weg. Unter dem generativen Modell ist das mathematisch exakt: Summieren wir $p(x_i\mid y)$ über beide Werte eines unbeobachteten Pixels $x_i$, erhalten wir $1$. Die Softmax-Regression hat keine entsprechende Marginalisierungsregel und muss für die fehlenden Pixel Ersatzwerte einsetzen (Aufgabe :exref:`ex_mdl-naive-bayes-missing-pixels`).

<!--en:0f5e0631-->
### Kalibrierung

<!--en:30c7f4fc-->
Die Klassenscores liefern mehr als nur eine Rangfolge der zehn Klassen. Durch Exponenzieren erhalten wir wieder die Bayes-Zähler und beim Normieren den zuvor weggelassenen Nenner $p(\mathbf x)$. Die A-posteriori-Wahrscheinlichkeiten des Modells ergeben sich also aus der Softmax-Funktion seiner Scores:

<!--en:0f451339-->
Zu jeder Vorhersage gehört somit eine angegebene Konfidenz, nämlich die größte dieser A-posteriori-Wahrscheinlichkeiten. Wie in :numref:`sec_classification` nennen wir einen Klassifikator *kalibriert*, wenn seine angegebenen Konfidenzen mit den empirischen Häufigkeiten übereinstimmen: Unter allen Vorhersagen mit Konfidenz $c$ sollte tatsächlich ein Anteil von $c$ richtig sein. Zur Prüfung der Kalibrierung teilen wir die Testbilder in Konfidenzbereiche ein. Für jeden Bereich vergleichen wir die mittlere angegebene Konfidenz mit der tatsächlich erreichten Genauigkeit.

<!--en:e374e2c6-->
Der *erwartete Kalibrierungsfehler* (engl. expected calibration error, ECE) fasst diese Tabelle in einer Zahl zusammen. Er ist der gewichtete Mittelwert der absoluten Differenzen zwischen angegebener Konfidenz und erreichter Genauigkeit in den einzelnen Bereichen. Das Gewicht jedes Bereichs ist sein Anteil an den Testbildern. Unten berechnen wir die Softmax-Funktion in ihrer numerisch stabilen Form, indem wir zunächst den größten Score abziehen (:numref:`subsec_mdl-stable-softmax`). Außerdem geben wir den Median der Differenz zwischen höchstem und zweithöchstem Klassenscore aus.

<!--en:30976603-->
Das Ergebnis fällt deutlich aus: Das Modell gibt im Mittel 99% Konfidenz an, doch nur 71% seiner Vorhersagen sind richtig. Rund 92% aller Testbilder fallen in den obersten Bereich. Dort gibt das Modell auf drei Nachkommastellen Gewissheit an, liegt aber nur in etwa 73% der Fälle richtig. In den drei unteren Bereichen gibt es zwischen 73% und 99.6% Konfidenz an, erreicht aber nur 35% bis 42% Genauigkeit. Da die Konfidenz in jedem Bereich überhöht ist, entspricht der erwartete Kalibrierungsfehler der Differenz zwischen mittlerer Konfidenz und Genauigkeit. Er beträgt etwa 0.29 gegenüber weniger als 0.03 bei der Softmax-Regression in :numref:`sec_classification`. Die Rangfolge der Konfidenzen ist dennoch aussagekräftig, denn die Genauigkeit steigt von Bereich zu Bereich. Auf die angegebenen Wahrscheinlichkeiten selbst ist jedoch kein Verlass.

<!--en:ce03c085-->
Die Ursache kennen wir vom Pulloversaum: Das Modell zählt korrelierte Pixel als unabhängige Hinweise. Im einfachsten Fall wird ein einzelnes Merkmal mehrfach erfasst. Die Rechnung dazu braucht nur drei Zeilen. Betrachten wir zwei Klassen mit A-priori-Wahrscheinlichkeiten $\pi_0$ und $\pi_1$ sowie ein Merkmal $x$ mit Likelihood-Verhältnis $r(x)=p(x\mid y=1)/p(x\mid y=0)$. Unser Datensatz enthalte $k$ identische Kopien von $x$. Da alle $k$ Kopien stets übereinstimmen, enthalten sie nur so viel Information wie eine einzige. Die wahren A-posteriori-Log-Odds betragen also $\log(\pi_1/\pi_0)+\log r(x)$. Naive Bayes behandelt die Kopien jedoch als bedingt unabhängig, multipliziert $k$ identische Faktoren $p(x\mid y)$ und liefert

<!--en:0b78aafa-->
Mehrfach gezählte Hinweise multiplizieren die Log-Odds bei gleichen A-priori-Wahrscheinlichkeiten mit $k$. In diesem Fall ändert die Multiplikation mit einer positiven Konstante nie das Vorzeichen. Naive Bayes sagt daher für jedes $k$ dieselbe Klasse vorher, gibt aber A-posteriori-Chancen von $\bar r^k$ an, obwohl $\bar r$ richtig wäre, mit $\bar r=\max(r,1/r)$. Das ist das zweite besonders wichtige Ergebnis dieses Abschnitts: Die Rangfolge bleibt erhalten, die Konfidenz verliert ihre Aussagekraft.

<!--en:31f35936-->
Ein Zahlenbeispiel verdeutlicht, wie stark die Konfidenz anwächst. Ein Hinweis liege in 80% der Fälle richtig, mit $p(x=1\mid y=1)=0.8$ und $p(x=1\mid y=0)=0.2$. Ist der Hinweis aktiv, gilt also $r=4$. Eine einzige Kopie liefert die richtige Konfidenz von $0.80$. Speichern wir zwei, drei, vier oder fünf Kopien, steigt die von Naive Bayes angegebene Konfidenz auf $0.94$, $0.985$, $0.996$ beziehungsweise $0.999$. Trotzdem liegen die Vorhersagen weiterhin nur in 80% der Fälle richtig. Erfassen wir in unserem Spamfilter das Wort „free“ doppelt, passiert dasselbe: Die Chancen der ersten Nachricht steigen von $9$ auf $\frac23\cdot4.5^2\cdot3=40.5$, ihre angegebene Spam-Wahrscheinlichkeit steigt auf $0.976$.

<!--en:2a135536-->
Auch die Rangfolge ist nur unter bestimmten Bedingungen sicher: Das vorige Argument setzt gleiche A-priori-Wahrscheinlichkeiten voraus. Bei A-priori-Chancen $\pi_1/\pi_0=1/5$ und $r=4$ ergibt eine einzige Kopie A-posteriori-Chancen von $4/5$ und sagt korrekterweise Klasse $0$ vorher. Zwei Kopien ergeben $16/5$ und ändern die Vorhersage zu Klasse $1$. Ein mehrfach gezählter Hinweis kann die A-priori-Chancen überwiegen, obwohl eine einzelne Beobachtung dafür nicht ausreichen sollte. Die Rangfolge ist nur dann vor einer solchen Umkehr sicher, wenn der Hinweis bereits mit der A-priori-Verteilung übereinstimmt oder eine Kopie allein sie überwiegt.

<!--en:efcbca16-->
Die 784 Pixel in Fashion-MNIST sind keine exakten Kopien voneinander, aber in größeren Bildbereichen korrelieren sie stark. Unsere beiden Saumpixel verhielten sich fast wie ein doppelt gezähltes Merkmal. Dutzende solcher korrelierten Bereiche summieren sich, sodass der Median der Differenz zwischen höchstem und zweithöchstem Klassenscore etwa 60 Nat beträgt. Wegen $e^{-60}\approx10^{-26}$ macht die Softmax-Funktion aus einer Differenz von 60 Nat eine angegebene Konfidenz von $1.000$. Dass Naive Bayes die Klassen trotz extrem überhöhter Konfidenz sinnvoll ordnet, ist eine bekannte Eigenschaft des Modells :cite:`Domingos.Pazzani.1997`. Ganz unbeschadet bleibt die Rangfolge aber nicht: Korrelierte Bereiche sind unterschiedlich groß. Ein großer Bereich mit redundanten Pixeln kann leicht einen kleineren mit unabhängiger Information überstimmen. Dieses Ungleichgewicht könnte erklären, warum die Softmax-Regression zwölf Prozentpunkte besser abschneidet. Sie kann Pixel geringer gewichten, die lediglich ihre Nachbarn wiederholen. Unser Experiment allein isoliert diesen Mechanismus allerdings nicht.

<!--en:50b9c716-->
Überhöhte Konfidenz lässt sich nachträglich auf einfache Weise korrigieren. Hier skizzieren wir das Verfahren nur: Bei der Temperaturskalierung (engl. temperature scaling, :numref:`sec_classification`) teilen wir jeden Klassenscore durch denselben Skalar $T$, den wir anhand zurückgehaltener Validierungsdaten schätzen. Die Rangfolge der Klassen bleibt dabei vollständig erhalten. Für $k$ identische Kopien und gleiche A-priori-Wahrscheinlichkeiten liefert $T=k$ die exakte A-posteriori-Verteilung. In Aufgabe :exref:`ex_mdl-naive-bayes-temperature-scaling` schätzen Sie $T$ für unser durch Zählen angepasstes Modell. Überhöhte Konfidenz tritt auch bei tiefen neuronalen Netzen auf. Diese Verzerrung ist dort meist schwächer ausgeprägt und hängt von Architektur und Trainingsrezept ab :cite:`Guo.Pleiss.Sun.Weinberger.2017`. Es empfiehlt sich daher, die Zuverlässigkeit jedes Klassifikators zu prüfen, dessen Wahrscheinlichkeiten in Entscheidungen einfließen.

<!--en:9d34a5aa-->
Eine Frage bleibt offen. Bei 6000 Bildern pro Klasse lag die diskriminative Anpassung klar vorn. Bei sehr wenigen Trainingsbeispielen kehrt sich die Rangfolge jedoch oft um: :citet:`Ng.Jordan.2002` haben bewiesen, dass für die Konvergenz der Naive-Bayes-Parameter eine in der Dimension $d$ logarithmische Anzahl von Beispielen genügt. Die logistische Regression kann hingegen eine in $d$ lineare Anzahl benötigen. In Aufgabe :exref:`ex_mdl-naive-bayes-generative-vs-discriminative` untersuchen Sie diesen Wechsel anhand von Pullovern und Mänteln.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3dcb4f74-->
Der Satz von Bayes macht aus einem generativen Modell, das die Entstehung der Eingaben jeder Klasse beschreibt, einen Klassifikator. Eine uneingeschränkte Tabelle für $p(\mathbf x\mid y)$ über $10^{236}$ binäre Bilder lässt sich allerdings nicht schätzen. Nehmen wir an, dass die Merkmale bei gegebener Klasse bedingt unabhängig sind, genügt für jedes Paar aus Klasse und Merkmal eine einzige relative Häufigkeit. Wir glätten sie mit einer Laplace-Pseudohäufigkeit von eins. Für das Training genügt also Zählen. Da der Logarithmus des Naive-Bayes-Zählers affin in den binären Merkmalen ist, gehört Bernoulli-Naive-Bayes als linearer Klassifikator zu genau derselben Modellfamilie wie die Softmax-Regression.

<!--en:697b29bb-->
Auf Fashion-MNIST erreicht dieser durch Zählen angepasste Klassifikator eine Genauigkeit von etwa $71\%\pm0.9$ Prozentpunkten. Das sind zwölf Punkte weniger als bei der direkt an dieselben binären Pixel angepassten Softmax-Regression. Er gibt im Mittel eine Konfidenz von 99% an, weil er korrelierte Pixel als unabhängige Hinweise behandelt und mehrfach gezählte Hinweise die Log-Odds vervielfachen. Diese extrem überhöhte Konfidenz zeigt, warum wir die Kalibrierung prüfen sollten, wann immer wir die Wahrscheinlichkeiten eines Klassifikators in der Praxis verwenden.

<!--en:2294243a-->
## Übungen

<!--en:7da238f9-->
1. **XOR und die naive Annahme.** Die vier Punkte $(0,0)$, $(0,1)$, $(1,0)$ und $(1,1)$ tragen die Labels $y=x_1\oplus x_2$, also $0$, $1$, $1$ und $0$. Jeder Punkt kommt im Trainingsdatensatz einmal vor.
    1. Berechnen Sie die Maximum-Likelihood-Schätzwerte für $p(y)$ und $\theta_{iy}=p(x_i=1\mid y)$ sowie die Klassenscores aller vier Punkte. Welche Vorhersagen liefert die Naive-Bayes-Regel :eqref:`eq_mdl-naive_bayes`?
    1. Zeigen Sie anhand der affinen Form :eqref:`eq_mdl-naive_bayes_affine` der Scores, dass keine Parameterwahl bei Bernoulli-Naive-Bayes alle vier Punkte richtig klassifiziert. Warum helfen auch entartete Parameter $\theta_{iy}\in\{0,1\}$ nicht? (Jede Klasse muss ihren eigenen beiden Punkten positive Wahrscheinlichkeit zuweisen.)
    1. Ersetzen Sie die beiden Faktoren $p(x_1\mid y)\,p(x_2\mid y)$ durch eine gemeinsame Tabelle $p(x_1,x_2\mid y)$. In der Sprache bayesscher Netze :cite:`Koller.Friedman.2009` fügt dies die Kante zwischen $x_1$ und $x_2$ hinzu, die rechts in :numref:`fig_mdl-naive-independence` zu sehen ist. Schätzen Sie dieses Modell anhand der vier Punkte und zeigen Sie, dass es sie richtig klassifiziert. Warum ist sein Klassenscore nicht mehr affin in $\mathbf x$?
    1. Innerhalb einer Klasse seien $x_1$ und $x_2$ unabhängige, gleichverteilte Bits, und es gelte $x_3=x_1\oplus x_2$. Zeigen Sie, dass jedes Paar der drei Merkmale unabhängig ist. Vergleichen Sie die Wahrscheinlichkeit jedes der acht Muster mit dem Produkt der drei Randwahrscheinlichkeiten. Warum kann eine Prüfung aller paarweisen Korrelationen innerhalb einer Klasse die Annahme :eqref:`eq_mdl-naive_assumption` nicht bestätigen?
   :exlabel:`ex_mdl-naive-bayes-xor`

<!--en:e80a7c37-->
1. **Naive Bayes als logistische Regression.** Betrachten Sie zwei Klassen $y\in\{0,1\}$ mit A-priori-Wahrscheinlichkeit $\pi_1=p(y=1)$ und Bernoulli-Parametern $\theta_{iy}=p(x_i=1\mid y)$.
    1. Zeigen Sie, dass für die A-posteriori-Wahrscheinlichkeit von Naive Bayes $p(y=1\mid\mathbf x)=\operatorname{sigmoid}(\mathbf w^\top\mathbf x+b)$ gilt, mit $\operatorname{sigmoid}(z)=1/(1+e^{-z})$ und $w_i=\log\frac{\theta_{i1}(1-\theta_{i0})}{\theta_{i0}(1-\theta_{i1})}$. Drücken Sie $b$ durch die Parameter $\theta_{iy}$ und $\pi_1$ aus. Welches Vorzeichen hat $w_i$ für ein Pixel, das in Klasse $1$ häufiger gesetzt ist als in Klasse $0$?
    1. Zeigen Sie umgekehrt, dass jedes Paar $(\mathbf w,b)$ durch eine geeignete Wahl von $\theta_{iy}\in(0,1)$ und $\pi_1\in(0,1)$ entsteht. Naive Bayes und logistische Regression definieren also dieselbe Menge von A-posteriori-Verteilungen und unterscheiden sich darin, wie sie die Verteilung an die Daten anpassen.
    1. Die beiden Klassen im Trainingsdatensatz seien linear separierbar. Wie ändert sich die bedingte Likelihood der logistischen Regression, wenn Sie ein trennendes Parameterpaar $(\mathbf w,b)$ mit einem Faktor $t\to\infty$ multiplizieren? Was folgt daraus für die Maximum-Likelihood-Schätzung von $(\mathbf w,b)$? Welches Paar $(\mathbf w,b)$ liefert Naive Bayes mit Laplace-Glättung auf denselben Daten?
   :exlabel:`ex_mdl-naive-bayes-discussion-after-eqref-eq`

<!--en:3f340057-->
1. **Klassenbedingte Gauß-Verteilungen.** Für ein einzelnes Merkmal gelten $x\mid y=0\sim\mathcal N(0,1)$ und $x\mid y=1\sim\mathcal N(0,4)$. Beide Klassen haben die A-priori-Wahrscheinlichkeit $1/2$.
    1. Bestimmen Sie die Menge der Werte $x$, die der Bayes-Klassifikator der Klasse $1$ zuordnet. Warum kann kein einzelner Schwellenwert für $x$ diese Menge liefern? Für welches transformierte Merkmal genügt ein Schwellenwert als Grenze?
    1. In $d$ Dimensionen hat Gauß-Naive-Bayes mit klassenspezifischen Varianzen den Klassenscore
       $$\log p(y)-\sum_{i=1}^d\left[\frac{(x_i-\mu_{iy})^2}{2\sigma_{iy}^2}+\frac12\log\left(2\pi\sigma_{iy}^2\right)\right].$$
       Zeigen Sie, dass die Grenze zwischen zwei Klassen eine quadratische Fläche ohne gemischte Terme $x_ix_j$ ist.
    1. Zwei Klassen in zwei Dimensionen haben gleiche A-priori-Wahrscheinlichkeiten und die klassenbedingten Verteilungen $\mathcal N(\mathbf 0,\boldsymbol\Sigma_+)$ und $\mathcal N(\mathbf 0,\boldsymbol\Sigma_-)$ mit $\boldsymbol\Sigma_\pm=\left(\begin{smallmatrix}1&\pm\rho\\\pm\rho&1\end{smallmatrix}\right)$ für ein $0<\rho<1$. Zeigen Sie, dass die Differenz der beiden Klassenscores $2\rho x_1x_2/(1-\rho^2)$ beträgt. Der Bayes-Klassifikator wählt also genau dann Klasse $+$, wenn $x_1x_2>0$ gilt. Zeigen Sie außerdem, dass jeder einzelne Score auch von $x_1^2+x_2^2$ abhängt. Die beiden Klassen haben dieselben Randverteilungen, sodass Gauß-Naive-Bayes selbst mit unbegrenzt vielen Daten nur das Niveau zufälligen Ratens erreicht. Unter $\mathcal N(\mathbf 0,\boldsymbol\Sigma_+)$ gilt $P(x_1x_2>0)=\frac12+\frac1\pi\arcsin\rho$ (Sheppards Formel). Bestimmen Sie den Bayes-Fehler als Funktion von $\rho$ und berechnen Sie ihn für $\rho=0.1$ und $\rho=0.9$.
   :exlabel:`ex_mdl-naive-bayes-gaussian-boundaries`

<!--en:ae5df3bc-->
1. **Zwei Ereignismodelle für Texte.** Ein Vokabular $V$ umfasst die vier Wörter *goal*, *match*, *vote* und *poll*. Der Trainingsdatensatz enthält drei Sportdokumente, „goal goal match“, „goal match match“ und „match vote“, sowie drei Politikdokumente, „vote poll poll“, „vote vote match“ und „poll vote“. Das Multinomialmodell behandelt ein Dokument als Sequenz von Wörtern, die unabhängig aus einer klassenspezifischen Verteilung über $V$ gezogen werden. Das Bernoulli-Modell erfasst für jedes Wort aus $V$ nur, ob es im Dokument vorkommt.
    1. Schätzen Sie beide Modelle mit Laplace-Glättung. Addieren Sie beim Multinomialmodell $1$ zur absoluten Häufigkeit jedes Wortes einer Klasse und $|V|$ zur Gesamtzahl der Wörter dieser Klasse. Schätzen Sie beim Bernoulli-Modell die Wahrscheinlichkeit für das Vorkommen eines Wortes durch $(n_{iy}+1)/(n_y+2)$. Dabei zählt $n_{iy}$ die Dokumente der Klasse $y$, die Wort $i$ enthalten.
    1. Klassifizieren Sie das Dokument „goal match poll poll“ mit beiden Modellen. Welche Wörter geben jeweils den Ausschlag?
    1. Verketten Sie ein Dokument $k$-mal mit sich selbst. Wie ändern sich die A-posteriori-Log-Odds beider Modelle? In echten Texten tritt ein Wort, das bereits einmal in einem Dokument vorgekommen ist, mit erhöhter Wahrscheinlichkeit erneut auf :cite:`Church.Gale.1995`. Aufgabe :exref:`ex_mdl-distributions-word-burstiness` aus :numref:`sec_mdl-distributions` reproduziert diesen Befund. Welche Annahme des Multinomialmodells verletzt er?
   :exlabel:`ex_mdl-naive-bayes-two-event-models`

<!--en:9dfa4f0e-->
1. [code] **Gepaarter Vergleich.** Im Text haben wir Naive Bayes und Softmax-Regression anhand ihrer Genauigkeit verglichen, also `pred` mit `pred_sr`.
    1. Wenden Sie das Bootstrap-Verfahren auf die Differenz der Richtigkeitsindikatoren pro Bild an. Verwenden Sie für beide Modelle dieselben neu gezogenen Indizes und geben Sie ein 95%-Intervall für die Differenz der Genauigkeiten an.
    1. Vergleichen Sie die Breite dieses Intervalls mit den Breiten der beiden einzelnen Intervalle und mit dem ungepaarten Standardfehler $\sqrt{p_1(1-p_1)/n+p_2(1-p_2)/n}$. Berechnen Sie die Korrelation zwischen den beiden Vektoren der Richtigkeitsindikatoren und erklären Sie damit, warum die Paarung hilft.
    1. Zählen Sie die Testbilder, die beide Modelle falsch, nur Naive Bayes richtig und nur die Softmax-Regression richtig klassifiziert. Welche dieser Anzahlen bestimmen die Differenz der Genauigkeiten? Wie sähe die Tabelle für ein Modell aus, das sich von Naive Bayes nur dadurch unterscheidet, dass es einige seiner Fehler korrigiert?
   :exlabel:`ex_mdl-naive-bayes-paired-comparison`

<!--en:a0d57b26-->
1. **Mehrfach erfasste und korrelierte Hinweise.** Zwei Klassen haben gleiche A-priori-Wahrscheinlichkeiten. Für ein binäres Merkmal gelten $p(x=1\mid y=1)=0.8$ und $p(x=1\mid y=0)=0.2$.
    1. Eine Verarbeitungskette speichert das Merkmal $k$-mal. Bestimmen Sie das kleinste $k$, bei dem Naive Bayes mindestens 99.9% Konfidenz für Vorhersagen angibt, die in 80% der Fälle richtig sind.
    1. Speichern Sie nun zwei Kopien, die nur teilweise übereinstimmen: Bei gegebenem $y$ gleicht die zweite Kopie der ersten mit Wahrscheinlichkeit $q$. Andernfalls ziehen Sie sie neu aus derselben klassenbedingten Verteilung. Zeigen Sie, dass beide Kopien dieselbe Randverteilung haben wie $x$. Naive Bayes gibt also weiterhin $16/17\approx0.94$ an, wenn beide Kopien den Wert $1$ haben. Zeigen Sie, dass die korrekte A-posteriori-Wahrscheinlichkeit $r/(1+r)$ mit $r=4(0.8+0.2q)/(0.2+0.8q)$ beträgt. Berechnen Sie sie für $q\in\{1,0.75,0.5,0.25,0\}$. Wie hängt die Überbewertung des Hinweises von der Korrelation der Kopien ab?
    1. Zeigen Sie, dass Sie in (a) die korrekte A-posteriori-Verteilung erhalten, wenn Sie jeden Klassenscore vor Anwendung der Softmax-Funktion durch $T=k$ teilen. Warum müssen die A-priori-Wahrscheinlichkeiten dafür gleich sein? (Auch der A-priori-Term wird durch $k$ geteilt.)
   :exlabel:`ex_mdl-naive-bayes-duplicated-features`

<!--en:40c7328a-->
1. [code] **Labelverteilungsverschiebung.** Bei einer Labelverteilungsverschiebung (:numref:`sec_environment-and-distribution-shift`) unterscheidet sich die A-priori-Verteilung im Zieldatensatz $p_T(y)$ von derjenigen im Quelldatensatz $p_S(y)$, während $p(\mathbf x\mid y)$ gleich bleibt.
    1. Zeigen Sie, dass für die A-posteriori-Verteilung unter der A-priori-Verteilung des Zieldatensatzes $p_T(y\mid\mathbf x)\propto p_S(y\mid\mathbf x)\,p_T(y)/p_S(y)$ gilt. Warum kann Naive Bayes sie ohne erneutes Training verwenden?
    1. Erstellen Sie einen Testdatensatz, in dem 90% der Bilder Pullover zeigen. Verwenden Sie dazu alle 1000 Testbilder von Pullovern und 111 zufällig gezogene Bilder aus den übrigen Klassen. Vergleichen Sie die Genauigkeit von `pred` auf diesem Datensatz mit der Genauigkeit der korrigierten Regel aus (a), mit $p_T(\text{pullover})=0.9$ und $p_T(y)=1/90$ für jede andere Klasse.
    1. Die Korrektur addiert $\log 9\approx2.2$ Nat zum Score für Pullover und zieht $2.2$ von den übrigen Scores ab. Vergleichen Sie diese Werte mit den Scoredifferenzen aus der Zelle `#mdl-naive-bayes-calibration-3` und erklären Sie das Ausmaß der in (b) gemessenen Änderung.
   :exlabel:`ex_mdl-naive-bayes-label-shift`

<!--en:7b102511-->
1. [code] **Temperaturskalierung.** ● Aufgabe :exref:`ex_mdl-naive-bayes-duplicated-features` zeigt, dass die Division der Klassenscores durch $T=k$ eine $k$-fache Wiederholung der Merkmale rückgängig macht. Bei Fashion-MNIST ist die effektive Wiederholungszahl unbekannt. Die Temperaturskalierung :cite:`Guo.Pleiss.Sun.Weinberger.2017` schätzt $T$ stattdessen anhand zurückgehaltener Daten. Die A-posteriori-Verteilung ergibt sich dann aus der Softmax-Funktion von $s_y(\mathbf x)/T$.
    1. Passen Sie das Modell erneut an, diesmal an die ersten $50{,}000$ Trainingsbilder. Wählen Sie $T$ so, dass die mittlere negative Log-Likelihood der A-posteriori-Verteilung auf den verbleibenden $10{,}000$ Bildern minimal ist. Warum ändert $T$ die Testgenauigkeit nicht?
    1. Wenden Sie das geschätzte $T$ auf die Testscores desselben mit $50{,}000$ Bildern trainierten Modells an. Erstellen Sie die Zuverlässigkeitstabelle aus der Zelle `#mdl-naive-bayes-calibration-3` erneut, mit den Bereichsgrenzen $0$, $0.5$, $0.7$, $0.9$, $0.99$ und $1$. Vergleichen Sie die mittlere angegebene Konfidenz mit der Genauigkeit.
    1. Zeigen Sie für die Situation aus Aufgabe :exref:`ex_mdl-naive-bayes-duplicated-features`, dass die Standardabweichung der Scoredifferenz bei echten Daten das $\sqrt k$-Fache der vom Modell vorhergesagten Standardabweichung beträgt. Berechnen Sie das Quadrat dieses Verhältnisses für die Paare Pullover/Mantel, T-Shirt/Hemd und Pullover/Hemd. Vergleichen Sie es mit dem geschätzten $T$.
    1. Wenden Sie die A-priori-Korrektur aus Aufgabe :exref:`ex_mdl-naive-bayes-label-shift` auf die temperaturskalierte A-posteriori-Verteilung an. Verwenden Sie den dort erstellten Testdatensatz. Warum ändert die korrigierte A-priori-Verteilung nach der Temperaturskalierung so viel mehr Vorhersagen?
   :exlabel:`ex_mdl-naive-bayes-temperature-scaling`

<!--en:4cb400c9-->
1. [code] **Fehlende Pixel.** Verdecken Sie in jedem Testbild eine zufällig ausgewählte Hälfte der Pixel mit einer booleschen Maske `M` in der Form von `X_test`, die für sichtbare Pixel den Wert `True` hat.
    1. Zeigen Sie, dass im Modell beim Herausmarginalisieren eines verdeckten Pixels aus $p(\mathbf x\mid y)$ sein Faktor entfällt. Der Score der sichtbaren Pixel beträgt also $\log p(y)+\sum_{i\text{ visible}}\log p(x_i\mid y)$. Implementieren Sie ihn mit zwei Matrixprodukten, `(X_test * M) @ np.log(P_xy).T + ((1 - X_test) * M) @ np.log(1 - P_xy).T`, zu denen Sie die logarithmierten A-priori-Wahrscheinlichkeiten addieren. Vergleichen Sie die Genauigkeit mit den 71% bei vollständigen Bildern.
    1. Füllen Sie die verdeckten Pixel stattdessen mit $0$, mit $1$ oder mit dem Mittelwert der Trainingsbilder auf. Berechnen Sie die Scores mit der affinen Form. Erklären Sie die jeweilige Genauigkeit anhand von :eqref:`eq_mdl-naive_bayes_affine`. Was addiert das Auffüllen mit $0$ zum Score jeder Klasse?
    1. Berechnen Sie die Scores der Softmax-Regression aus dem Text für die Bilder, in denen die verdeckten Pixel auf $0$ gesetzt sind. Welche Operation aus (a) steht dem generativen, aber nicht dem diskriminativen Modell zur Verfügung?
   :exlabel:`ex_mdl-naive-bayes-missing-pixels`

<!--en:905d616c-->
1. [extended] **Generatives und diskriminatives Lernen.** ●● :citet:`Ng.Jordan.2002` vergleichen Naive Bayes mit der logistischen Regression. Wie Aufgabe :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq` zeigt, passt die logistische Regression dieselbe Familie von A-posteriori-Verteilungen durch Maximierung der bedingten Likelihood an. Die Autoren beweisen, dass sich die Naive-Bayes-Parameter nach einer in $d$ logarithmischen Anzahl von Beispielen ihren Populationswerten nähern. Bei der logistischen Regression wächst die Anzahl der Beispiele, die zur Annäherung an ihren asymptotischen Fehler nötig ist, höchstens linear in $d$. Im ungünstigsten Fall wächst sie auch mindestens linear in $d$. Auf vielen ihrer $15$ Datensätze hat Naive Bayes bei kleinen Trainingsdatensätzen den geringeren Testfehler, die logistische Regression bei großen.
    1. Jede Klasse enthalte mindestens $m$ Trainingsbeispiele. Zeigen Sie mit der Hoeffding-Ungleichung (:numref:`sec_mdl-concentration-generalization`) und der booleschen Ungleichung: Alle $2d$ ungeglätteten Schätzwerte $\hat\theta_{iy}$ eines Modells mit zwei Klassen weichen um höchstens $\varepsilon$ von ihren Populationswerten ab, und zwar mit Wahrscheinlichkeit mindestens $1-\delta$, wenn gilt:
       $$m\ge\frac{\log(4d/\delta)}{2\varepsilon^2}.$$
       Werten Sie die Schranke für $d=784$, $\varepsilon=0.05$ und $\delta=0.05$ aus.
    1. Eine gleichmäßige Schranke für $|\hat\theta_{iy}-\theta_{iy}|$ beschränkt für sich genommen die Änderung der Klassenscores nicht. Warum gilt das für Pixel mit $\theta_{iy}$ nahe $0$ oder $1$, etwa für das Eckpixel mit dem größten Gewicht im Vergleich von Pullover und Mantel? Was müssen Sie über die $\theta_{iy}$ annehmen, damit die Scores mit derselben Rate wie die Parameter konvergieren?
    1. Beschränken Sie die Daten auf Pullover und Mäntel. Ziehen Sie für die Gesamtgrößen $n\in\{10,20,50,100,200,500,1000,2000,5000\}$ des Trainingsdatensatzes und für alle $12{,}000$ Bilder zufällige Trainingsteilmengen, die beide Klassen enthalten: $20$ für jede Größe bis $n=200$ und $5$ für jede größere. Passen Sie Naive Bayes mit Laplace-Glättung und die logistische Regression ohne Gewichtszerfall an. Für kleine $n$ sind die Trainingsdaten linear separierbar. Nach Aufgabe :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq`c hat der logistische Verlust dann keinen Minimierer. Führen Sie daher ausgehend von der Initialisierung mit null eine feste Anzahl von Gradientenschritten mit dem gesamten Trainingsdatensatz aus. Plotten Sie die mittleren Testfehler beider Klassifikatoren gegen $n$ auf einer logarithmischen Achse mit `d2l.plot`. In welchem Bereich von $n$ hat Naive Bayes gegebenenfalls den geringeren Fehler?
    1. Wiederholen Sie (c) mit synthetischen Daten, für die die naive Annahme gilt. Ziehen Sie Labels mit gleicher Wahrscheinlichkeit und jedes Pixel unabhängig aus $\mathrm{Bernoulli}(\theta_{iy})$. Schätzen Sie dabei die $\theta_{iy}$ aus allen Trainingsbildern der beiden Klassen. Wie ändern sich die beiden Lernkurven und ihre asymptotischen Fehler?
    1. Satz 4 von :citet:`Ng.Jordan.2002` beschränkt den Überschussfehler von Naive Bayes durch die Wahrscheinlichkeit, dass die Scoredifferenz im Populationsmodell nahe bei null liegt. Vergleichen Sie diese Wahrscheinlichkeit für echte Pixel und für die synthetischen Daten aus (d). Erklären Sie, wie dieser Vergleich zusammen mit den asymptotischen Fehlern aus (c) und (d) die beiden Lernkurvenpaare erklärt.
   :exlabel:`ex_mdl-naive-bayes-generative-vs-discriminative`

<!--en:2fcf4a1f-->
[Diskussion](https://d2l.discourse.group/t/418)

<!--en:f8d644e1-->
[Diskussion](https://d2l.discourse.group/t/1100)

<!--en:847c7565-->
[Diskussion](https://d2l.discourse.group/t/1101)

<!--en:847c7565-2-->
[Diskussion](https://d2l.discourse.group/t/1101)

<!--en:ad1a4b34-->
Naive Bayes und der Vergleich generativer und diskriminativer Klassifikatoren<br>**generativ oder diskriminativ · bedingte Unabhängigkeit · Laplace-Glättung · Linearität der Log-Odds · Kalibrierung und überhöhte Konfidenz**

<!--en:cdbb7a1f-->
::: {.slide title="10^236 binäre Bilder, nur 60 000 Trainingsbeispiele"}

<!--en:ccb1ec6c-->
[Motivation und kombinatorische Explosion]{.kicker}

<!--en:06bfbcc7-->
In früheren Kapiteln kam die Softmax-Regression auf Fashion-MNIST zum Einsatz. Der Gradientenabstieg setzte direkt bei der bedingten Wahrscheinlichkeit des Labels $p(y \mid \mathbf{x})$ an ($7{,}850$ Parameter, $\approx 83\%$ Genauigkeit). Was ergibt sich stattdessen aus einem Modell der Bildentstehung in jeder Klasse ($p(\mathbf{x} \mid y)$), wenn der Satz von Bayes die Bedingung umkehrt?

<!--en:8bcfc1e2-->
- **Die kombinatorische Hürde:** Selbst bei einer Darstellung der $d = 28 \times 28 = 784$ Pixel durch je ein Bit $x_i \in \{0, 1\}$ (gesetzt oder leer) gibt es $2^{784} \approx 10^{236}$ verschiedene binäre Bilder.

<!--en:a36d653c-->
- **Vollständige Tabellen sind nicht praktikabel:** Für eine beliebige Verteilung $p(\mathbf{x} \mid y)$ brauchen wir $2^{784} - 1$ freie Wahrscheinlichkeiten pro Klasse. Unser gesamter Trainingsdatensatz umfasst nur $60{,}000$ Bilder ($6{,}000$ pro Klasse).

<!--en:a80fe847-->
- **Die Vereinfachung bei Naive Bayes:** Bei bekannter Klasse $y$ seien die $d$ Merkmale bedingt unabhängig. Dann genügt es, beim Training Häufigkeiten einzelner Pixel zu zählen und bei der Vorhersage Log-Odds zu addieren.

<!--en:c2e62719-->
::: {.slide title="Der Satz von Bayes: generativer und diskriminativer Ansatz"}

<!--en:140639c5-->
[Zwei Familien von Klassifikatoren]{.kicker}

<!--en:3c842c97-->
- **Generativer Ansatz (Naive Bayes):** $p(\mathbf{x} \mid y)$ und $p(y)$ für jede Klasse getrennt schätzen, dann die Bedingung mit dem Bayes-Zähler $p(y)\prod_i p(x_i \mid y)$ umkehren.

<!--en:1e793eea-->
- **Diskriminativer Ansatz (Softmax-Regression):** Ohne Modell für $p(\mathbf{x} \mid y)$ die Entscheidungsgrenze $p(y \mid \mathbf{x})$ direkt über die bedingte negative Log-Likelihood optimieren.

<!--en:28e6bee8-->
::: {.slide title="Die naive Annahme der bedingten Unabhängigkeit"}

<!--en:d1d2a0bb-->
[Faktorisierung der klassenbedingten Dichte]{.kicker}

<!--en:fe882cdf-->
- **Die Vorgeschichte entfällt:** Naive Bayes nimmt an, dass bei gegebener Klasse $y$ die Kenntnis eines Merkmals $x_i$ nichts weiter über ein anderes Merkmal $x_j$ aussagt:
  $$p(\mathbf{x} \mid y) = \prod_{i=1}^d p(x_i \mid y)$$

<!--en:313fa164-->
- **Exponentiell weniger Parameter:** Für $d = 784$ binäre Merkmale und $K = 10$ Klassen genügen statt $2^{784} - 1$ Parametern pro Klasse nur $784$ Bernoulli-Häufigkeiten $\theta_{iy} = p(x_i = 1 \mid y)$ (jeweils mit Standardfehler $\le \frac{0.5}{\sqrt{6000}} \approx 0.0065$ geschätzt).

<!--en:5f5bdd82-->
- **Gemeinsame und paarweise Unabhängigkeit:** Paarweise unkorrelierte Merkmale garantieren keine gemeinsame bedingte Unabhängigkeit (sind etwa $x_1, x_2$ unabhängige, gleichverteilte Bits und gilt $x_3 = x_1 \oplus x_2$, ist jedes Paar unabhängig, und doch gilt $p(1,1,1) = 0 \neq \frac{1}{8}$).

<!--en:1c9ce51b-->
::: {.slide title="Rechenbeispiel: ein Spamfilter für zwei Wörter"}

<!--en:b08ecb18-->
[Multiplikation von Likelihood-Verhältnissen]{.kicker}

<!--en:e41684d8-->
- **Beobachtungen verknüpfen:** Die A-priori-Chancen ($2/3$), der Faktor für das Vorkommen von `"free"` ($\times 4.5$) und der Faktor für das Fehlen von `"meeting"` ($\times 3.0$) ergeben multipliziert die A-posteriori-Chancen $9:1$ ($P = 0.90$).

<!--en:70533c54-->
- **Additive Log-Odds:** Durch Logarithmieren wird aus der Multiplikation von Likelihood-Verhältnissen eine Addition von Log-Odds ($-0.41 + 1.50 + 1.10 = +2.20$ Nat).

<!--en:773dd669-->
::: {.slide title="Maximum-Likelihood durch Zählen: Wie eine Null eine Klasse ausschließt"}

<!--en:39c3e00f-->
[Laplace-Glättung durch Addition von eins]{.kicker}

<!--en:353bfa84-->
Bei bedingter Unabhängigkeit zerfällt die Log-Likelihood des Trainingsdatensatzes in unabhängige Log-Likelihoods von Münzwürfen, je eine für jede Klasse $y$ und jedes Merkmal $i$. Dabei zählt $n_y$ die Beispiele der Klasse und $n_{iy}$ die Beispiele mit aktivem Merkmal in Klasse $y$:

<!--en:89ffa8ef-->
- **Nullhäufigkeiten beim ungeglätteten Maximum-Likelihood-Schätzer ($\hat{\theta}_{iy} = n_{iy}/n_y$):** Gilt $n_{iy} = 0$ (wie für `"meeting"` in unseren $4$ Spam-Nachrichten im Training), so folgt $\hat{\theta}_{iy} = 0$. Jedes künftige Testbeispiel mit $x_i = 1$ erhält den Bayes-Zähler $0$ für Klasse $y$. Ein einziges Merkmal schließt die ganze Klasse aus.

<!--en:1e8ed807-->
- **Laplace-Glättung (A-posteriori-Mittelwert der Beta-Verteilung):** Ein gedachtes positives und ein gedachtes negatives Beispiel ergänzen. Das ergibt bei gleichförmiger A-priori-Verteilung $\mathrm{Beta}(1,1)$ den A-posteriori-Mittelwert:
  $$\hat{\theta}_{iy} = \frac{n_{iy} + 1}{n_y + 2} \in (0, 1)$$
  Mit Glättung erhält eine E-Mail, die sowohl `"free"` als auch `"meeting"` enthält, die Spam-Wahrscheinlichkeit $0.32$ statt $0$.

<!--en:134262d9-->
::: {.slide title="Gleitkommaunterlauf durch Logarithmieren vermeiden"}

<!--en:1d7f3c62-->
[Additive Klassenscores]{.kicker}

<!--en:1d1ce785-->
Bei direkter Gleitkommamultiplikation von $d = 784$ Pixelwahrscheinlichkeiten $\prod_{i=1}^{784} p(x_i \mid y)$ kann das Ergebnis vollständig verloren gehen: Hat jeder Faktor den Wert $1/2$, so ist das Produkt $2^{-784} \approx 10^{-236}$. Das liegt weit unter der kleinsten positiven subnormalen Zahl vom Typ `float32` ($1.4 \times 10^{-45}$). Durch Logarithmieren wird das Produkt zum additiven **Klassenscore**:

<!--en:077c822d-->
Die Dezimalexponenten $\log_{10} = s_y(\mathbf{x}) / \log 10$ für die $10{,}000$ Testbilder von Fashion-MNIST zeigen, warum wir logarithmisch rechnen müssen:

<!--en:ddd90c3c-->
::: {.slide title="Satz: Bernoulli-Naive-Bayes ist ein linearer Klassifikator"}

<!--en:2568fb5b-->
[Gewichtsmatrix aus Log-Odds und additive Konstante]{.kicker}

<!--en:9b4be85c-->
Für binäre Merkmale $x_i \in \{0, 1\}$ gilt $\log p(x_i \mid y) = x_i \log\theta_{iy} + (1 - x_i)\log(1 - \theta_{iy})$. Terme mit dem Faktor $x_i$ und die von $x_i$ unabhängigen Terme jeweils zusammenfassen. Daraus folgt: Jeder Klassenscore ist eine **affine Funktion von $\mathbf{x}$**:

<!--en:ad8ebdc8-->
- **Dieselbe Architektur wie bei der Softmax-Regression:** Bernoulli-Naive-Bayes speichert eine $10 \times 784$-Gewichtsmatrix $\mathbf{W}$ (dabei sind $w_{iy}$ die Log-Odds dafür, dass Pixel $i$ in Klasse $y$ gesetzt ist) und einen $10$-elementigen Bias-Vektor $\mathbf{b}$ (dabei ist $b_y$ der logarithmische Score eines vollständig leeren Bildes).

<!--en:00d4aa93-->
- **Äquivalenz zur logistischen Regression im binären Fall:** Für $K = 2$ Klassen gilt $p(y = 1 \mid \mathbf{x}) = \operatorname{sigmoid}\bigl((\mathbf{w}_1 - \mathbf{w}_0)^\top \mathbf{x} + b_1 - b_0\bigr)$.

<!--en:23497d37-->
- **Entscheidungsgrenzen bei Gauß-Naive-Bayes:** Für stetige Merkmale $x_i \mid y \sim \mathcal{N}(\mu_{iy}, \sigma_{iy}^2)$ fallen bei klassenübergreifend gleichen Varianzen ($\sigma_{iy}^2 = \sigma_i^2$) die quadratischen $x_i^2$-Terme weg. Die Grenze ist **linear**. Klassenspezifische Varianzen ergeben eine **diagonal-quadratische** Grenze ohne gemischte Terme $x_i x_j$.

<!--en:9784bba2-->
::: {.slide title="Geometrie generativ und diskriminativ bestimmter Hyperebenen"}

<!--en:d8f3bd2f-->
[Warum korrelierte Merkmale die Entscheidungsgrenze neigen]{.kicker}

<!--en:947782ef-->
- **Dieselbe lineare Hypothesenklasse, unterschiedliche Anpassungen:** Warum erreichen zwei lineare Klassifikatoren aus derselben Modellfamilie unterschiedliche Fehlerquoten?

<!--en:2a4bc89b-->
- **Naive generative Anpassung (links, $19\%$ Fehler):** Sind die Merkmale $x_1, x_2$ innerhalb jeder Klasse korreliert, verlaufen die tatsächlichen Punktwolken schräg. Gauß-Naive-Bayes passt achsenparallele Ellipsen an ($\operatorname{Cov} = \operatorname{diag}$). Wenden wir auf diese falsch ausgerichteten Ellipsen den Satz von Bayes an, erhalten wir eine Hyperebene mit falscher Neigung.

<!--en:5acd1b58-->
- **Direkte diskriminative Anpassung (rechts, $7\%$ Fehler):** Die direkte Anpassung von $p(y \mid \mathbf{x})$ richtet die Hyperebene entlang der schrägen Punktwolken aus und erreicht die Bayes-optimale Fehlerquote.

<!--en:47516cf7-->
::: {.slide title="Naive Bayes auf Fashion-MNIST in vier Codezeilen trainieren"}

<!--en:5b479bd3-->
[Zählen in einem Durchlauf]{.kicker}

<!--en:fab6ee17-->
Fashion-MNIST binarisieren ($x_i = 1$ überall dort, wo für den Grauwert $> 0$ gilt, was einen Gesamtanteil gesetzter Pixel von $0.50$ ergibt). Für alle $7{,}850$ Parameter genügt dann ein einziger Zähldurchlauf ohne Gradientenschritte:

<!--en:e79b6bb7-->
- **Warum die Laplace-Glättung unverzichtbar ist:** Bei Turnschuhen bleiben im Training $57$ Pixel nahe dem oberen Bildrand stets leer. In $5$ Testbildern ist ein Pixel gesetzt, das in den $6{,}000$ Trainingsbeispielen der wahren Klasse immer leer war. Ohne die Pseudohäufigkeiten $+1 / +2$ erhielten alle $5$ Bilder den Score $-\infty$ für ihre wahre Klasse.

<!--en:b093df2d-->
::: {.slide title="Gelernte Klassenvorlagen und generierte Pixelbilder"}

<!--en:47027d6d-->
[Was das generative Modell erfasst und was verloren geht]{.kicker}

<!--en:4f04e8b6-->
**Pixelweise Klassenvorlagen ($\hat{\theta}_{iy}$)**

<!--en:a931097e-->
**Bilder aus unabhängigen Münzwürfen pro Pixel**

<!--en:18b14184-->
- **Vorlagen spiegeln die Randverteilungen der Daten wider:** Jede Vorlage links ist der exakte pixelweise Mittelwert von $6{,}000$ Trainingsbildern. Jedes Modell mit denselben Randverteilungen hat dieselben Vorlagen.

<!--en:d4b16fde-->
- **Generierte Bilder offenbaren die falsche Unabhängigkeitsannahme:** Aus $784$ unabhängigen Münzwürfen mit Wahrscheinlichkeiten $\hat{\theta}_{iy}$ (rechts) entstehen gesprenkelte, ausgefranste Ränder, weil benachbarte Randpixel nicht aufeinander abgestimmt sind.

<!--en:bee84fa3-->
::: {.slide title="Die Abweichung von der Unabhängigkeit bei echten Kleidungsstücken messen"}

<!--en:2d6f9bcd-->
[Wechsel pro Zeile und paarweise Korrelation am Saum]{.kicker}

<!--en:a471f981-->
Global lässt sich die Abweichung von der bedingten Unabhängigkeit an der Zahl der Wechsel zwischen gesetzten Pixeln und Hintergrund entlang jeder Bildzeile messen. Für die lokale Messung genügen zwei benachbarte Pixel in Zeile $26$, Spalten $14$ und $15$, am unteren Saum von Pullovern:

<!--en:19620fc2-->
- **Fast wie ein doppelt erfasstes Merkmal:** Jedes Saumpixel ist bei $55\%$ der Pullover gesetzt. Bei Unabhängigkeit wären beide bei $0.55^2 = 30\%$ der Pullover gemeinsam gesetzt. In den echten Daten sind es **$54\%$** der Pullover.

<!--en:0672c48b-->
::: {.slide title="Vektorisierte Inferenz und Bootstrap-Fehlerbalken"}

<!--en:456a1645-->
[70.6% +/- 0.9% Genauigkeit auf dem Testdatensatz]{.kicker}

<!--en:199ab188-->
Mit der affinen Form $s_y(\mathbf{x}) = \mathbf{w}_y^\top \mathbf{x} + b_y$ klassifiziert eine einzige Matrizenmultiplikation `X_test @ W.T + b` alle $10{,}000$ Testbilder. $1{,}000$ Bootstrap-Stichproben bestätigen den binomialen Standardfehler:

<!--en:86d95e34-->
- **Hohe Verzerrung, kaum Anzeichen für Überanpassung:** Die Trainingsgenauigkeit ($71.7\%$) liegt nur etwa $1$ Prozentpunkt über der Testgenauigkeit ($70.6\% \pm 0.9\%$). Mehr Trainingsdaten beheben die verbleibenden $29\%$ Fehler nicht. Die Unabhängigkeitsannahme begrenzt die Genauigkeit durch ihre **Verzerrung**.

<!--en:84f5e5a4-->
::: {.slide title="Naive Bayes und Softmax-Regression im direkten Vergleich"}

<!--en:1474b258-->
[Dieselbe lineare Familie, unterschiedliche Zielfunktionen]{.kicker}

<!--en:a12090b4-->
Ist $71\%$ das beste Ergebnis, das ein linearer Klassifikator mit Hyperebenen auf binarisierten $28 \times 28$-Pixeln erreichen kann? Die Antwort liefert die Softmax-Regression auf denselben binären Vektoren $\mathbf{X}$, trainiert durch Minimierung der Kreuzentropie $-\sum_i \log p(y_i \mid \mathbf{x}_i)$:

<!--en:3984e686-->
- **$12$ Prozentpunkte mehr in derselben Modellfamilie:** Die Softmax-Regression erreicht **etwa $83\%$** Genauigkeit und verwechselt Pullover$\leftrightarrow$Mantel nur etwa halb so oft (rund $300$ gegenüber $614$). Der diskriminative Gradientenabstieg passt die Gewichte gemeinsam an und vermeidet so, korrelierte Pixelbereiche doppelt zu zählen.

<!--en:9ec9e2c6-->
::: {.slide title="Die Konfusionsmatrix nach Kleidungsgruppen auswerten"}

<!--en:d09f07b7-->
[Wo das Zählen die meisten Schwierigkeiten bereitet]{.kicker}

<!--en:e0eb0c4b-->
Die sechs größten Einträge außerhalb der Diagonalen der $10 \times 10$-Konfusionsmatrix $N_{ij}$ (Spalte $j$ gibt die wahre Klasse an, Zeile $i$ die vorhergesagte) zeigen, wo Naive Bayes die $12$ Punkte verliert:

<!--en:b8f9c7af-->
- **Zwei Gruppen mit häufigen Verwechslungen:** Auf die vier Oberbekleidungsarten (Mantel, Pullover, Hemd, T-Shirt) und die drei Schuhklassen (Sandale, Turnschuh, Stiefelette) entfällt der weitaus größte Teil der Fehler. Hemden werden nur in **$17\%$** der Fälle erkannt.

<!--en:b922892b-->
- **Markante Silhouetten bleiben leicht erkennbar:** Klassen mit einer charakteristischen räumlichen Form, etwa Hosen ($91\%$) und Turnschuhe ($89\%$), lassen sich selbst mit unabhängig gezählten Pixeln gut trennen.

<!--en:6b88886d-->
::: {.slide title="Die Log-Odds-Gewichte zur Klassentrennung sichtbar machen"}

<!--en:47f35a52-->
[Interpretation von w_pullover - w_coat]{.kicker}

<!--en:fa42270c-->
- **Lineare Differenz der Log-Odds ($\mathbf{w}_{\text{pullover}} - \mathbf{w}_{\text{coat}}$):** Ist Pixel $i$ gesetzt ($x_i = 1$) statt leer ($x_i = 0$), verschiebt es die Log-Odds zwischen Pullover und Mantel um $w_{i,\text{pullover}} - w_{i,\text{coat}}$ Nat.

<!--en:6ea25d8b-->
- **Warum ein gesetztes Pixel am blauen Rumpf ($-1$ Nat) kaum für Mantel spricht:** Rumpfpixel sind bei $\approx 99.0\%$ der Pullover und $\approx 99.6\%$ der Mäntel gesetzt. Ein gesetztes Rumpfpixel verschiebt das logarithmische Verhältnis daher um $\log\frac{0.990}{0.996} \approx -0.006$ Nat. Das Gewicht von $-1$ Nat erklärt sich aus $-\log\frac{1-0.990}{1-0.996}$: Ein *leeres* Rumpfpixel spricht also für Pullover.

<!--en:e71d0431-->
- **Die Entscheidung fällt an den roten Seiten:** Bei den abgebildeten Kleidungsstücken geben Pixel an den Außenkanten von Ärmeln und Schultern den Ausschlag.

<!--en:11ee39e9-->
::: {.slide title="Kalibrierung prüfen: 99.2% Konfidenz bei 70.6% Genauigkeit"}

<!--en:149cbf91-->
[Erwarteter Kalibrierungsfehler (ECE)]{.kicker}

<!--en:95c46a29-->
Exponenzieren und Normieren der Klassenscores liefert die A-posteriori-Wahrscheinlichkeiten $p(y \mid \mathbf{x}) = \frac{\exp s_y(\mathbf{x})}{\sum_{y'} \exp s_{y'}(\mathbf{x})}$ des Modells. Die Einteilung der $10{,}000$ Testvorhersagen in Konfidenzbereiche zeigt eine stark **überhöhte Konfidenz**:

<!--en:83580eca-->
**Für 92% der Testbilder gibt das Modell eine Konfidenz von $\ge 99.9\%$ an (`1.000`):** Im Testdatensatz beträgt der Median der Differenz zwischen den beiden höchsten Klassenscores **$60$ Nat** ($e^{-60} \approx 10^{-26}$). Der erwartete Kalibrierungsfehler liegt bei $\mathrm{ECE} = 0.29$ (gegenüber $< 0.03$ bei der Softmax-Regression).

<!--en:052d1692-->
::: {.slide title="Warum korrelierte Merkmale die Konfidenz so stark überhöhen"}

<!--en:b4b10ceb-->
[Ein Satz über mehrfach gezählte Hinweise]{.kicker}

<!--en:b25afe35-->
- **Vervielfachung multipliziert die Log-Odds mit $k$:** Liegt ein Hinweis mit Likelihood-Verhältnis $r = 4$ (wahre Genauigkeit $80\%$) in $k = 5$ Kopien vor, gibt Naive Bayes Chancen von $4^5 = 1024$ an ($99.9\%$ Konfidenz).

<!--en:d7df09cb-->
- **Korrektur:** Temperaturskalierung $s_y(\mathbf{x})/T$ mit $T = k$ stellt bei gleichen A-priori-Wahrscheinlichkeiten die kalibrierten Log-Odds wieder her und erhält die Rangfolge der Klassen.

<!--en:95e73be7-->
::: {.slide title="Nachträgliche Korrektur: Temperaturskalierung und Labelverteilungsverschiebung"}

<!--en:bed13cdf-->
[Wahrscheinlichkeiten ohne erneutes Training korrigieren]{.kicker}

<!--en:12482acf-->
Zwei Anpassungen korrigieren die Wahrscheinlichkeiten von Naive Bayes beim Test, ohne die $7{,}840$ gelernten Pixelhäufigkeiten $\hat{\theta}_{iy}$ zu ändern:

<!--en:5ccd06b8-->
- **1. Temperaturskalierung ($p_T(y \mid \mathbf{x}) = \operatorname{softmax}\bigl(s_y(\mathbf{x})/T\bigr)$):** Jeden Klassenscore durch dieselbe skalare Temperatur $T > 1$ teilen, geschätzt durch Minimierung der negativen Log-Likelihood auf zurückgehaltenen Validierungsdaten. Das reduziert die Scoredifferenzen von $60$ Nat auf kalibrierte Log-Odds, ohne $\operatorname{argmax}_y s_y(\mathbf{x})$ zu ändern. Bei $k$ identischen Merkmalen und gleichen A-priori-Wahrscheinlichkeiten liefert $T = k$ die exakte A-posteriori-Verteilung.

<!--en:78adc8b5-->
- **2. Anpassung an eine Labelverteilungsverschiebung ohne erneutes Training:** Weicht die A-priori-Verteilung der Klassen im Einsatz $p_T(y)$ von der im Training $p_S(y)$ ab, während $p(\mathbf{x} \mid y)$ gleich bleibt (*Labelverteilungsverschiebung*), aktualisiert ein generativer Klassifikator seine additive Konstante in einer Zeile:
  $$s_y^{\mathrm{target}}(\mathbf{x}) = s_y^{\mathrm{source}}(\mathbf{x}) + \log\frac{p_T(y)}{p_S(y)}$$
  (Die Temperaturskalierung muss *vor* der Addition der A-priori-Korrektur erfolgen, damit die Merkmalssumme von $60$ Nat die Verschiebung um $2.2$ Nat nicht überdeckt.)

<!--en:aa24b3d7-->
::: {.slide title="Ein Vorteil generativer Klassifikatoren: fehlende Merkmale"}

<!--en:327674dd-->
[Exakte analytische Marginalisierung]{.kicker}

<!--en:f32707ad-->
- **Exakte Marginalisierung:** Beim Summieren über verdeckte Pixel $j \in \mathcal{U}$ entfallen ihre Faktoren im Score $s_y(\mathbf{x}_{\mathcal{V}}) = \log p(y) + \sum_{i \in \mathcal{V}} \log p(x_i \mid y)$.

<!--en:38fb3677-->
- **Fallstrick beim Ersetzen durch null:** Mit $x_j = 0$ in $\mathbf{w}_y^\top\mathbf{x} + b_y$ behandelt das Modell verdeckte Pixel über $b_y = \log p(y) + \sum_j \log(1-\theta_{jy})$ als beobachteten leeren Hintergrund.

<!--en:9ffe30cf-->
::: {.slide title="Stichprobenbedarf nach Ng und Jordan (2002): Wann wechselt die Rangfolge?"}

<!--en:6ac9ebaf-->
[Lernkurven für O(log d) und O(d)]{.kicker}

<!--en:331b6f07-->
- **Kleines $n$ (Naive Bayes kann besser abschneiden):** Da die $1\text{D}$-Pixelhäufigkeiten entkoppelt sind, genügen nach der booleschen Ungleichung für $2d$ Parameter schon $n = O(\log d)$ Beispiele, um das asymptotische Fehlerniveau zu erreichen (geringe Varianz, höhere Verzerrung).

<!--en:df3f03f7-->
- **Großes $n$ (der diskriminative Ansatz gewinnt):** Die logistische Regression bzw. die Softmax-Regression passt $d$ gekoppelte Gewichte an und kann $n = O(d)$ Beispiele benötigen, erreicht aber einen niedrigeren asymptotischen Fehler. Auf dem gesamten Trainingsdatensatz von Fashion-MNIST ergibt sich eine Genauigkeit von etwa $83\%$ gegenüber $71\%$.

<!--en:bb42bee3-->
- **Die Kurven veranschaulichen den Satz schematisch.** Ob und wo sie sich bei Fashion-MNIST schneiden, ist Gegenstand einer Übung.

<!--en:8df44132-->
::: {.slide title="Generative und diskriminative Klassifikatoren im Vergleich"}

<!--en:9f41cf81-->
[Die Unterschiede im Überblick]{.kicker}

<!--en:1e4a6c96-->
| Kriterium | Bernoulli-Naive-Bayes (generativ) | Logistische Regression / Softmax-Regression (diskriminativ) |

<!--en:adcf4423-->
| **Probabilistische Zielfunktion** | Gemeinsame Likelihood $\prod_i p(\mathbf{x}_i \mid y_i)\,p(y_i)$ | Bedingte Likelihood $\prod_i p(y_i \mid \mathbf{x}_i)$ |

<!--en:b5cd81d3-->
| **Entscheidungsscore $s_y(\mathbf{x})$** | Affin $\mathbf{w}_y^\top \mathbf{x} + b_y$ ($w_{iy} = \log\frac{\theta_{iy}}{1-\theta_{iy}}$) | Affin $\mathbf{w}_y^\top \mathbf{x} + b_y$ (direkt mit SGD angepasst) |

<!--en:54ab405b-->
| **Trainingsalgorithmus** | Geschlossene Lösung durch Zählen in einem Durchlauf ($O(nd)$) | Iterative Gradientenoptimierung über mehrere Epochen |

<!--en:dc2bcbbd-->
| **Stichprobenbedarf** | **$O(\log d)$** Beispiele bis zum asymptotischen Fehlerniveau | **$O(d)$** Beispiele bis zum asymptotischen Fehlerniveau |

<!--en:c657efae-->
| **Asymptotische Genauigkeit ($n \gg d$)** | Niedriger ($70.6\%$ auf Fashion-MNIST, hohe Verzerrung) | Höher ($\approx 83\%$ auf Fashion-MNIST, geringe Verzerrung) |

<!--en:0e5a3a14-->
| **Kalibrierung der Wahrscheinlichkeiten** | Stark überhöhte Konfidenz ($\mathrm{ECE} = 0.29$, Doppelzählung) | Gut kalibriert ($\mathrm{ECE} < 0.03$) |

<!--en:ae32d906-->
| **Fehlende Merkmale beim Test** | Exakte Marginalisierung in geschlossener Form (Faktoren weglassen) | Keine Marginalisierungsregel, Ersatzwerte erforderlich |

<!--en:db472332-->
::: {.slide title="Verständnisfragen: Naive Bayes und Kalibrierung"}

<!--en:929b7671-->
[Selbsttest]{.kicker}

<!--en:24a5f3cd-->
1. **Kann Bernoulli-Naive-Bayes XOR lernen?** Es gelte $y = x_1 \oplus x_2$ auf den vier binären Punkten $(0,0), (0,1), (1,0), (1,1)$. Warum kann keine Wahl der Bernoulli-Naive-Bayes-Parameter $\theta_{iy} \in (0, 1)$ alle vier Punkte richtig klassifizieren?

<!--en:c3fcefad-->
2. **Mehrfach erfasster Hinweis gegen starke A-priori-Chancen:** Zwei Klassen haben A-priori-Chancen von $\frac{\pi_1}{\pi_0} = \frac{1}{5}$. Ein binäres Merkmal $x=1$ hat das Likelihood-Verhältnis $r = \frac{p(x=1\mid y=1)}{p(x=1\mid y=0)} = 4$. Die Verarbeitung speichert $x$ in drei Kopien ($k = 3$). Welche Klasse sagt die wahre A-posteriori-Verteilung vorher, welche Naive Bayes?

<!--en:527ecfca-->
3. **Warum Auffüllen mit null keine Marginalisierung ist:** Warum genügt es beim affinen Score $s_y(\mathbf{x}) = \mathbf{w}_y^\top \mathbf{x} + b_y$ nicht, für ein verdecktes Pixel $x_i = 0$ zu setzen, um $x_i$ aus $p(\mathbf{x} \mid y)$ herauszusummieren?

<!--en:55575ea1-->
::: {.slide title="Verständnisfragen: Naive Bayes und Kalibrierung (Fortsetzung)"}

<!--en:929b7671-2-->
[Selbsttest]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:549fd1ab-->
1. Bernoulli-Naive-Bayes liefert einen affinen Log-Odds-Score $(\mathbf{w}_1 - \mathbf{w}_0)^\top \mathbf{x} + (b_1 - b_0)$ mit einer einzigen Hyperebene als Entscheidungsgrenze. XOR ist jedoch nicht linear separierbar (tatsächlich gilt für jede Randhäufigkeit $\hat{\theta}_{iy} = 1/2$, sodass jedes Gewicht $w_{iy} = \log 1 = 0$ ist).

<!--en:df2c568e-->
2. Die wahren A-posteriori-Chancen betragen $\frac{1}{5} \times 4 = \frac{4}{5} < 1$. Damit lautet die Vorhersage **Klasse $0$** ($p(y=1\mid x=1) = 4/9 \approx 44\%$). Naive Bayes berechnet Chancen von $\frac{1}{5} \times 4^3 = \frac{64}{5} = 12.8 > 1$ und ändert so die Vorhersage zu **Klasse $1$** mit einer angegebenen Konfidenz von $92.8\%$.

<!--en:75647432-->
3. Da $b_y = \log p(y) + \sum_{j=1}^d \log(1 - \theta_{jy})$ bereits $\log(1 - \theta_{iy})$ für jedes Pixel enthält, wertet das Modell bei $x_i = 0$ das Pixel $i$ als **beobachteten leeren Hintergrund** ($p(x_i=0\mid y) = 1-\theta_{iy}$). Der Faktor für Pixel $i$ entfällt also nicht ($1$).

<!--en:c8912351-->
::: {.slide title="Zusammenfassung: Naive Bayes und generative Klassifikation"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:f932efc8-->
| Konzept | Mathematische Form | Bedeutung für Statistik und Deep Learning |

<!--en:29f45595-->
| **Generative Klassifikation mit dem Satz von Bayes** | $\hat{y} = \operatorname{argmax}_y p(\mathbf{x} \mid y)\,p(y)$ | Macht aus einem klassenbedingten Datenmodell durch Umkehrung einen Klassifikator |

<!--en:cbfaa886-->
| **Naive Faktorisierung** | $p(\mathbf{x} \mid y) = \prod_{i=1}^d p(x_i \mid y)$ | Reduziert $2^d - 1$ Tabelleneinträge pro Klasse auf $d$ unabhängige Randverteilungen |

<!--en:039f7267-->
| **Laplace-Glättung** | $\hat{\theta}_{iy} = \frac{n_{iy} + 1}{n_y + 2}$ (Mittelwert von $\mathrm{Beta}(1,1)$) | Verhindert, dass unbeobachtete Kombinationen aus Merkmal und Klasse ($n_{iy} = 0$) die Klasse $y$ ausschließen |

<!--en:6b9c92a7-->
| **Affiner Score mit Log-Odds** | $s_y(\mathbf{x}) = \mathbf{w}_y^\top \mathbf{x} + b_y$, $w_{iy} = \log\frac{\theta_{iy}}{1-\theta_{iy}}$ | Zeigt, dass Bernoulli-Naive-Bayes ein linearer Klassifikator aus der Softmax-Familie ist |

<!--en:7bba0e4b-->
| **Verzerrung durch mehrfach gezählte Hinweise** | $\log\text{odds}_{\mathrm{NB}} = \log\frac{\pi_1}{\pi_0} + k\log r(x)$ | Korrelierte Merkmale vervielfachen die Log-Odds und überhöhen die Konfidenz ($\mathrm{ECE} = 0.29$) |

<!--en:39894a9b-->
| **Abwägung nach Ng und Jordan** | Naive Bayes: $O(\log d)$ Beispiele, logistische Regression: $O(d)$ | Generatives Zählen konvergiert bei kleinem $n$ schneller, diskriminativer SGD gewinnt bei großem $n$ |
