<!--en:101c63b8-->
# Gegenseitige Information

<!--en:e42daee2-->
In den vorigen Abschnitten haben wir zwei Verteilungen derselben Variablen verglichen, etwa eine Datenverteilung und ein Modell. Viele Fragen im Deep Learning betreffen dagegen ein Paar von Variablen, etwa ein Bild und seine Beschreibung, zwei Ausschnitte desselben Fotos oder einen Satz und seine Übersetzung. Wie viel verrät uns die eine Variable über die andere?

<!--en:0443e33a-->
Das übliche Abhängigkeitsmaß, der Korrelationskoeffizient, erfasst nur lineare Zusammenhänge. Betrachten wir $X\sim\mathcal N(0,1)$ und setzen wir $Y=X^2+0.3\,\epsilon$, wobei $\epsilon\sim\mathcal N(0,1)$ unabhängiges Rauschen ist. Da $X^3$ symmetrisch um null verteilt ist, gilt $\textrm{Cov}(X,Y)=\mathbb{E}[X^3]=0$. Die Korrelation zwischen $X$ und $Y$ ist also null. Dennoch bestimmt $X$ den Wert von $Y$ fast vollständig. Kennen wir $X$, so schwankt $Y$ nur noch durch Rauschen mit Standardabweichung $0.3$. Für sich genommen streut $Y$ dagegen ungefähr von $0$ bis $9$. Die Größe, die wir hier definieren, erfasst diese Abhängigkeit. Sie weist diesem Paar sogar mehr gemeinsame Information zu als einem bivariat normalverteilten Paar mit Korrelation $0.9$.

<!--en:c98899c8-->
Wir suchen ein Maß, das genau dann null ist, wenn zwei Variablen unabhängig sind. Es soll mit der Stärke der Abhängigkeit wachsen und unverändert bleiben, wenn wir eine der Variablen in anderen Koordinaten ausdrücken. Das Werkzeug dafür kennen wir bereits. Es ist die KL-Divergenz $D_{\textrm{KL}}(P\|Q)$ aus :numref:`sec_mdl-information_theory`. Nach der Gibbs-Ungleichung ist sie stets nichtnegativ und verschwindet nur für $P=Q$. Wählen wir für $P$ die wahre gemeinsame Verteilung des Paares und für $Q$ die Produktverteilung, die bei unabhängigen Komponenten vorläge, erhalten wir die *gegenseitige Information* (engl. mutual information) $I(X;Y)$.

<!--en:3d10eb95-->
Die gegenseitige Information ist leicht zu definieren. Was sie so wertvoll macht, sind die Folgerungen. Wir leiten drei grundlegende Ergebnisse her. Erstens kann die Verarbeitung einer Variablen ihre gegenseitige Information mit keiner anderen Variablen erhöhen. Daher enthält keine Schicht eines neuronalen Netzes mehr Information über die Eingabe als die darunterliegende Schicht. Zweitens kann ein Klassifikator eine vorgegebene Fehlerrate nur erreichen, wenn seine Eingabe ein Mindestmaß an Information über das Label enthält. Drittens liefert der Testverlust jedes probabilistischen Modells unmittelbar eine untere Schranke für diese Information. Anschließend wenden wir uns Shannons beiden Grundfragen zu. Wie viel Information kann ein verrauschter Kanal übertragen? Wie wenig Information muss ein verlustbehafteter Kompressor bewahren? Aus der Kompression ergibt sich eine dritte Frage. Was soll eine gelernte Repräsentation bewahren? In diesem Abschnitt setzen wir die zugrunde liegenden Verteilungen als bekannt voraus. Die gegenseitige Information allein aus Stichproben zu schätzen, ist schwieriger. Damit befassen wir uns in :numref:`sec_mdl-mi-estimation`.

<!--en:b2cdb548-->
Wie zuvor rechnen wir in Nat, wobei $1$ Bit $=\ln 2\approx0.693$ Nat gilt. Alle Logarithmen sind natürlich. In allgemeinen Formeln schreiben wir $\log$, bei der Berechnung von Zahlenwerten $\ln$. Die binäre Entropie bezeichnen wir mit $H_b(p)=-p\log p-(1-p)\log(1-p)$.

<!--en:fe0861fb-->
## Abhängigkeitsmaße

<!--en:5668d440-->
Wir werten jede Definition zunächst an derselben $2 \times 2$-Wahrscheinlichkeitstabelle aus. Sie dient im gesamten Abschnitt als Beispiel.

<!--en:5c4e12c1-->
### Gemeinsame und bedingte Entropie

<!--en:eeb1e15c-->
Sei $X$ ein gleichverteiltes Bit und $Y$ eine Kopie von $X$, die mit Wahrscheinlichkeit $\delta=0.1$ kippt. Wir können $X$ als Bit auffassen, das über eine verrauschte Leitung gesendet wird, und $Y$ als empfangenes Bit. Ebenso können wir $Y$ als binäres Label und $X$ als verrauschtes Merkmal deuten, das in $90\%$ der Fälle mit dem Label übereinstimmt. Da die Tabelle symmetrisch ist, spielt diese Wahl im Folgenden keine Rolle. Ordnen wir die Zeilen nach $x\in\{0,1\}$ und die Spalten nach $y\in\{0,1\}$, so lautet die gemeinsame Verteilung

<!--en:984850c3-->
und die Zeilen- und Spaltensummen ergeben die Randverteilungen $p_X=p_Y=(\tfrac12,\tfrac12)$. Wir nennen dieses Paar das *gekippte Bit*. Tiefgestellte Indizes bezeichnen die Variablen einer Verteilung. Wir lassen sie weg, wenn die Argumente die Verteilung eindeutig kennzeichnen, wie bei der bedingten Verteilung $p(y\mid x)$.

<!--en:09415ac1-->
Neben den einzelnen Entropien $H(X)$ und $H(Y)$ ergeben sich für ein Paar von Zufallsvariablen zwei weitere natürliche Entropien. Beide kennen wir aus :numref:`subsec_mdl-joint-conditional-entropy`. Die *gemeinsame Entropie* (engl. joint entropy) ist die Entropie des Paares als einer einzigen Zufallsvariablen, hier mit vier möglichen Ergebnissen:

<!--en:a08f5161-->
Für unsere Tabelle erhalten wir $H(X,Y)=-2(0.45\ln0.45)-2(0.05\ln0.05)=1.018$ Nat. Das ist mehr als die $0.693$ Nat eines einzelnen gleichverteilten Bits, aber weniger als die $1.386$ Nat zweier unabhängiger gleichverteilter Bits.

<!--en:b22fd943-->
Die *bedingte Entropie* (engl. conditional entropy) misst, wie viel Überraschung in $Y$ verbleibt, wenn $X$ bekannt ist. Sie ist die mittlere Entropie der bedingten Verteilungen in den Zeilen $p(y\mid x)=p_{X,Y}(x,y)/p_X(x)$:

<!--en:6e1c3fb8-->
Beim gekippten Bit lauten die beiden Zeilen $(0.9,0.1)$ und $(0.1,0.9)$. Beide haben die Entropie $H_b(0.1)=0.325$ Nat, also gilt $H(Y\mid X)=0.325$ Nat. Selbst nach Beobachtung von $X$ verbleiben in $Y$ noch $0.325$ der ursprünglichen $0.693$ Nat an Überraschung.

<!--en:cd05760f-->
Die drei Zahlen erfüllen $1.018=0.693+0.325$. Darin erkennen wir die Kettenregel $H(X,Y)=H(X)+H(Y\mid X)$ aus :numref:`subsec_mdl-joint-conditional-entropy`. Die Überraschung im Paar ist die Summe aus der Überraschung in $X$ und der zusätzlichen Überraschung in $Y$, wenn $X$ bereits bekannt ist.

<!--en:7dc55780-->
### Gegenseitige Information

<!--en:6b7f5b0a-->
Die bedingte Entropie erfasst den Anteil der Überraschung in $Y$, den $X$ nicht erklären kann. Nun interessiert uns der Anteil, den beide Variablen gemeinsam haben. Wären $X$ und $Y$ unabhängig, so wäre ihre gemeinsame Verteilung das Produkt ihrer Randverteilungen, $P_X \otimes P_Y$. In jeder Zelle stünde dann $p_X(x)\,p_Y(y)$. Die *gegenseitige Information* ist die KL-Divergenz der tatsächlichen gemeinsamen Verteilung von dieser Produktverteilung:

<!--en:e68b25e0-->
Beim gekippten Bit steht im Produkt der Randverteilungen in jeder Zelle $0.25$. Das Verhältnis der gemeinsamen Wahrscheinlichkeit zum Produkt beträgt daher $0.45/0.25=1.8$ auf der Diagonalen, wo $X$ und $Y$ übereinstimmen, und $0.05/0.25=0.2$ außerhalb der Diagonalen, wo sie verschieden sind. Mitteln wir die Logarithmen dieser Verhältnisse unter der gemeinsamen Verteilung, so ergibt sich

<!--en:245c58e4-->
Übereinstimmungen treten $1.8$-mal so häufig auf wie bei Unabhängigkeit, Unterschiede nur ein Fünftel so häufig. Die gegenseitige Information ist der Erwartungswert des Logarithmus dieses Faktors. Die KL-Divergenz misst auch die zusätzliche Codelänge bei Verwendung des falschen Modells (:numref:`sec_mdl-coding-compression`). Daher gibt $I(X;Y)$ an, wie viele Nat pro Paar ein Code verschwendet, der $X$ und $Y$ als unabhängig behandelt. Genau diese zusätzlichen Kosten begegnen uns in :numref:`sec_diffusion-discrete`, wenn ein Netz zwei maskierte Token unabhängig voneinander vorhersagt.

<!--en:98c5546c-->
Wir können :eqref:`eq_mdl-mut_ent_def` auch zeilenweise lesen. Teilen wir Zähler und Nenner durch $p_X(x)$, so wird das Verhältnis zu $p(y\mid x)/p_Y(y)$. Der Erwartungswert über die Paare lässt sich damit als Mittelwert über $x$ schreiben. Gemittelt wird die KL-Divergenz zwischen der bedingten Verteilung und der Randverteilung von $Y$:

<!--en:e9d68ba3-->
In dieser Form misst die gegenseitige Information, wie weit die Beobachtung von $x$ die Verteilung von $Y$ im Mittel von ihrer Randverteilung verschiebt. Beim gekippten Bit hat jede der beiden bedingten Verteilungen, $(0.9,0.1)$ oder $(0.1,0.9)$, eine KL-Divergenz von $0.9\ln1.8+0.1\ln0.2=0.368$ Nat zur Randverteilung $(0.5,0.5)$. Diese asymmetrische Sichtweise bietet sich besonders an, wenn die Variablen verschiedene Rollen spielen, etwa als Eingabe und Ausgabe eines verrauschten Encoders. In :numref:`sec_mdl-information-bottleneck` greifen wir sie wieder auf.

<!--en:48512817-->
**Satz (Eigenschaften der gegenseitigen Information).** *Für jedes Paar $(X,Y)$ gilt $I(X;Y)=I(Y;X)\ge0$. Der Wert null wird genau dann erreicht, wenn $X$ und $Y$ unabhängig sind. Sind die beteiligten Entropien endlich, so gilt*

<!--en:b650beb7-->
**Beweis.** Vertauschen wir $X$ und $Y$, bleibt das Verhältnis in :eqref:`eq_mdl-mut_ent_def` unverändert. Damit ist die Symmetrie gezeigt. Die Nichtnegativität und die Bedingung für Gleichheit folgen aus der Gibbs-Ungleichung für $P_{X,Y}$ und $P_X \otimes P_Y$. Die KL-Divergenz ist nichtnegativ und verschwindet genau dann, wenn überall $p_{X,Y}(x,y) = p_X(x)\,p_Y(y)$ gilt. Das ist die Definition der Unabhängigkeit. Für die Entropiegleichungen zerlegen wir den Logarithmus des Verhältnisses in drei Terme und bilden die Erwartungswerte:

<!--en:cebdd00f-->
Die beiden übrigen Formen folgen aus der Kettenregel in beiden Reihenfolgen, $H(X,Y) = H(Y) + H(X \mid Y)=H(X)+H(Y\mid X)$. $\blacksquare$

<!--en:40bc1030-->
Somit verrät $X$ genau so viel über $Y$ wie $Y$ über $X$. Die letzte Gleichung deutet die gegenseitige Information als Abnahme der Unsicherheit. Beobachten wir $X$, so sinkt die Überraschung in $Y$ von $H(Y)$ auf $H(Y\mid X)$, beim gekippten Bit also von $0.693$ auf $0.325$ Nat. Aus der Nichtnegativität folgt auch die Subadditivität $H(X,Y)\le H(X)+H(Y)$, mit Gleichheit nur bei Unabhängigkeit. Die Überraschung im Paar ist nie größer als die Summe der Überraschungen in den einzelnen Komponenten. :numref:`fig_mdl-mi-overlap` veranschaulicht diese Beziehungen. Jede Gleichung des Satzes gibt einen anderen Weg an, den Flächeninhalt der Überlappung abzulesen.

<!--en:23273e6a-->
![Die gegenseitige Information $I(X;Y)$ als Überlappung der Entropien $H(X)$ und $H(Y)$. Die Sicheln entsprechen den bedingten Entropien $H(X\mid Y)$ und $H(Y\mid X)$, die Vereinigungsfläche der gemeinsamen Entropie $H(X,Y)$. Aus den Flächen lassen sich die drei gleichwertigen Ausdrücke für $I(X;Y)$ ablesen.](../img/mdl-it-mi-overlap.svg)

<!--en:59ef5d84-->
Die folgende Funktion `mutual_information` berechnet die Divergenzform für jede diskrete Tabelle einer gemeinsamen Verteilung. Wir verwenden sie im gesamten Abschnitt. Die Codezelle prüft auch die Identität $I(X;Y) = H(Y) - H(Y\mid X)$, indem sie die Entropien der Zeilen mittelt.

<!--en:28d26b1e-->
Beide Ausdrücke ergeben $0.3681$ Nat oder $0.531$ Bit. Eine Kopie, die in einem von zehn Fällen kippt, verrät also noch etwas mehr als ein halbes Bit über das Original.

<!--en:1e69c8a0-->
### Das gaußsche Paar

<!--en:5e9063de-->
Bisher haben wir mit diskreten Tabellen gearbeitet. Die meisten Variablen im Deep Learning, etwa Aktivierungen, Gradienten und Einbettungen, sind jedoch stetig. Einen stetigen Fall können wir vollständig von Hand durchrechnen, nämlich die bivariate Normalverteilung. Sie dient uns als zweites durchgängiges Beispiel und bildet später die Grundlage für die Rate-Verzerrungs-Kurve und den Informationsflaschenhals (engl. information bottleneck). Bei stetigen Dichten treten an die Stelle der diskreten Entropien die differentiellen Entropien $h$ aus :numref:`subsec_mdl-differential-entropy`. Diese können negativ sein und hängen von den Maßeinheiten ab. Die KL-Definition der gegenseitigen Information hat keinen dieser Nachteile. Die Herleitung über die Kettenregel setzt keine diskreten Variablen voraus. Daher gilt $I(X;Y)=h(X)+h(Y)-h(X,Y)=h(Y)-h(Y\mid X)$, sofern diese Größen endlich sind. Dabei bezeichnet $h(Y\mid X)$ die differentielle Entropie der bedingten Dichte $p(y\mid x)$, gemittelt über $x$.

<!--en:e8e3b1c9-->
**Satz (gegenseitige Information eines gaußschen Paares).** *Ist $(X, Y)$ bivariat normalverteilt mit Varianzen $\sigma_X^2,\sigma_Y^2>0$ und Korrelationskoeffizient $\rho$, wobei $\rho^2<1$ gilt, so folgt*

<!--en:66c57ba9-->
**Beweis.** Wir verwenden $I(X;Y)=h(Y)-h(Y\mid X)$. Eine gaußsche Zufallsvariable mit Varianz $\sigma^2$ hat die differentielle Entropie $\tfrac12\log(2\pi e\sigma^2)$ (:numref:`subsec_mdl-gaussian-max-entropy`), also gilt $h(Y)=\tfrac12\log(2\pi e\sigma_Y^2)$. Bedingt auf $X=x$ bleibt $Y$ normalverteilt mit Varianz $\sigma_Y^2(1-\rho^2)$, unabhängig vom Wert von $x$. Das ist die Varianz des Residuums der besten linearen Vorhersage von $Y$ aus $X$. Bei einer bivariaten Normalverteilung ist dieses Residuum unabhängig von $X$ (:numref:`subsec_mdl-mvn-conditioning`). Daher gilt $h(Y\mid X)=\tfrac12\log\big(2\pi e\sigma_Y^2(1-\rho^2)\big)$, und die Differenz ergibt $-\tfrac12\log(1-\rho^2)$. $\blacksquare$

<!--en:dcab4b28-->
Das Ergebnis hängt allein von $1-\rho^2$ ab, dem Anteil der Varianz von $Y$, den $X$ nicht erklärt. Die Maßeinheiten von $Y$ haben sich herausgekürzt. Hier zeigt sich erstmals die Koordinateninvarianz, die wir in :numref:`sec_mdl-mi-nonlinear-correlation` beweisen.

<!--en:0c118726-->
Für $\rho = 0$ sind die Variablen unabhängig und es gilt $I(X;Y) = 0$. Für $\rho \to \pm 1$ zieht sich die gemeinsame Dichte auf eine Gerade zusammen und $I(X;Y) \to \infty$. Zwischen diesen Extremen erhalten wir für $\rho = 0.5$ den Wert $0.144$ Nat, für $\rho = 0.9$ den Wert $0.830$ Nat und für $\rho = 0.99$ den Wert $1.959$ Nat. Die bivariate Normalverteilung mit standardnormalen Randverteilungen und $\rho=0.9$ begegnet uns in diesem Abschnitt mehrfach. Wir nennen sie das *gaußsche Paar mit $\rho=0.9$*.

<!--en:4b21b2dd-->
Wie der Grenzübergang $\rho \to \pm 1$ zeigt, kann die gegenseitige Information unendlich sein. Diesem Phänomen begegnen wir später zweimal, bei deterministischen Schichten eines neuronalen Netzes und beim Encoder des Informationsflaschenhalses. Gilt $Y=X$ für eine stetige Zufallsvariable $X$, so legt die Kenntnis von $X$ den Wert von $Y$ auf unendlich viele Dezimalstellen fest. Dasselbe gilt, wenn $Y$ eine deterministische Funktion von $X$ ist, die keinen einzelnen Wert mit positiver Wahrscheinlichkeit annimmt. Ist dagegen $Y$ diskret und hat endliche Entropie, so kann die gegenseitige Information nicht unendlich sein. Dann gilt $I(X;Y)=H(Y)-H(Y\mid X)\le H(Y)$ unabhängig von $X$, wobei $H(Y\mid X)$ die Entropie von $p(\cdot\mid x)$ über $x$ mittelt. Für standardnormalverteiltes $X$ enthält das Vorzeichen von $X$ beispielsweise genau $\ln 2$ Nat. Dieselbe obere Schranke begrenzt die Jensen-Shannon-Divergenz aus :numref:`sec_basic_gan` auf $\ln 2$. Wie dort gezeigt, entspricht sie der gegenseitigen Information zwischen einem Beispiel und dem binären Label, das dessen Quelle angibt.

<!--en:b4dc2f6a-->
### Punktweise gegenseitige Information

<!--en:5f9e5e7b-->
Die gegenseitige Information mittelt über alle Ergebnispaarungen. Dabei bleibt verborgen, welche Paare zur Abhängigkeit beitragen. Oft interessiert uns ein einzelnes Paar. Ist etwa nach dem Wort *new* das Wort *york* viel wahrscheinlicher, als es der Zufall erwarten ließe? Die Antwort liefert der Term, der in :eqref:`eq_mdl-mut_ent_def` gemittelt wird. Es ist die *punktweise gegenseitige Information* (engl. pointwise mutual information, PMI),

<!--en:197f55d6-->
die angibt, wie häufig das konkrete Paar $(x, y)$ im Vergleich zu dem Fall auftritt, dass $X$ und $Y$ unabhängig sind. Eine positive PMI zeigt an, dass sich die Ergebnisse anziehen, also häufiger gemeinsam auftreten, als der Zufall erwarten lässt. Eine negative PMI zeigt an, dass sie sich abstoßen. Ihre mittlere Anziehung beträgt $I(X;Y) = \mathbb{E}_{(x,y) \sim P_{X,Y}}[\textrm{pmi}(x,y)]$. Beim gekippten Bit beträgt die PMI $\ln1.8=+0.588$ in den beiden Zellen mit übereinstimmenden Werten und $\ln0.2=-1.609$ in den beiden anderen Zellen. Genau diese Logarithmen haben wir zuvor gemittelt.

<!--en:cf3a3aaf-->
Die punktweise gegenseitige Information ist ein klassisches Werkzeug der natürlichen Sprachverarbeitung :cite:`Church.Hanks.1990`. Sie unterscheidet *Kollokationen*, also Wortpaare, die sprachlich zusammengehören, von Wörtern, die nur deshalb gemeinsam auftreten, weil beide häufig sind. Angenommen, wir zählen aufeinanderfolgende Wortpaare (Bigramme) in einem kleinen Korpus. Dabei halten wir fest, wie oft auf *new*, *machine* oder *the* jeweils *york*, *learning* oder *day* folgt:

<!--en:79e8a1bc-->
| Anzahl | york | learning | day |

<!--en:800fba53-->
| **new** | 38 | 3 | 9 |

<!--en:b25b4118-->
| **machine** | 1 | 46 | 3 |

<!--en:cc61eb68-->
| **the** | 11 | 21 | 168 |

<!--en:6333785b-->
Die absoluten Häufigkeiten täuschen hier. „the day“ ist mit Abstand das häufigste Paar ($168$ Vorkommen gegenüber $38$ für „new york“), weil *the* und *day* beide sehr häufig sind. Der folgende Code teilt jede gemeinsame Häufigkeit durch das Produkt der beiden Randhäufigkeiten. Er gibt die so berechnete PMI für vier Paare und die gegenseitige Information der gesamten Tabelle aus.

<!--en:57ef1162-->
Berücksichtigen wir die Randhäufigkeiten, stehen die tatsächlichen Kollokationen an der Spitze. „new york“ erreicht $1.52$ Nat und „machine learning“ $1.37$, das häufige Paar „the day“ dagegen nur $0.34$. Für „the york“ ist die PMI mit $-1.11$ Nat negativ. Das Paar tritt etwa ein Drittel so häufig auf, wie bei Unabhängigkeit zu erwarten wäre. Die gegenseitige Information der Tabelle beträgt $0.41$ Nat. Sie fasst zusammen, wie stark das erste Wort im Mittel die Wahl des zweiten einschränkt. Dieselbe Korrektur um die Randhäufigkeiten begegnet uns bei gelernten Worteinbettungen (:numref:`sec_lm_word_embeddings`). Trainieren wir ein Skip-Gram-Modell darauf, beobachtete Paare benachbarter Wörter von $K$ zufälligen Paaren zu unterscheiden, so faktorisieren wir implizit die Matrix der verschobenen PMI-Werte, $\textrm{pmi}(x,y)-\log K$ :cite:`Levy.Goldberg.2014`. Dies leiten wir in Aufgabe :exref:`ex_mdl-mutual-information-negative-sampling-pmi` in :numref:`sec_mdl-mi-estimation` her.

<!--en:84bb4f3c-->
### Gegenseitige Information und Korrelation

<!--en:0e5d3d2c-->
Der Korrelationskoeffizient $\rho(X,Y)=\textrm{Cov}(X,Y)/(\sigma_X\sigma_Y)$ aus :numref:`sec_mdl-joint-distributions` erfasst nur lineare Abhängigkeit. Aus $\rho(X,Y)=0$ folgt keine Unabhängigkeit. Die gegenseitige Information verschwindet dagegen genau dann, wenn die Variablen unabhängig sind. Unser Eingangsbeispiel verdeutlicht diesen Unterschied.

<!--en:a3b51e4e-->
Seien $X \sim \mathcal{N}(0, 1)$ und $Y = X^2 + 0.3\,\epsilon$ mit unabhängigem $\epsilon \sim \mathcal{N}(0,1)$. Die Korrelation dieses Paares ist $\rho=0$. Mit $I(X;Y)=h(Y)-h(Y\mid X)$ können wir die gegenseitige Information über eindimensionale Integrale berechnen. Bedingt auf $X=x$ stammt der Zufall in $Y$ allein vom additiven Rauschen. Die Verschiebung einer Verteilung um die bekannte Konstante $x^2$ ändert ihre differentielle Entropie nicht. Also gilt $h(Y\mid X)=h(0.3\,\epsilon)=\tfrac12\ln(2\pi e\cdot0.09)=0.215$ Nat. Bevor wir $h(Y)$ numerisch berechnen, können wir die Größe analytisch abschätzen. $Y$ hat die Varianz $\textrm{Var}(X^2)+0.09=2.09$. Unter allen Dichten mit vorgegebener Varianz maximiert die Normalverteilung die differentielle Entropie (:numref:`subsec_mdl-gaussian-max-entropy`). Das ergibt $h(Y)\le\tfrac12\ln(2\pi e\cdot2.09)=1.79$ Nat und damit $I(X;Y)\le1.79-0.215=1.57$ Nat. Die Randdichte von $Y$ ist eine stetige Mischung über $x$, nämlich $p_Y(y)=\int\mathcal N(x;0,1)\,\mathcal N(y;x^2,0.09)\,dx$. Da sie keine geschlossene Form hat, bestimmen wir $h(Y)$ durch numerische Integration.

<!--en:52ac87e2-->
Das Paar teilt $1.141$ Nat. Dieser Wert liegt deutlich unter unserer gaußschen oberen Schranke von $1.57$ Nat. Die Dichte von $Y$ konzentriert sich nahe null und hat einen langen rechten Ausläufer, weicht also stark von einer Normalverteilung ab. Zugleich liegt der Wert über den $0.830$ Nat des gaußschen Paares mit $\rho=0.9$, obwohl die lineare Korrelation unseres Paares null ist. :numref:`fig_mdl-mi-corr-vs-mi` zeigt alle drei Fälle nebeneinander.

<!--en:a69280fc-->
![Drei Paare im Vergleich anhand ihrer Korrelation und ihrer gegenseitigen Information. (a) Ein gaußsches Paar mit Korrelation $0.9$. (b) $Y=X^2+0.3\epsilon$. Die Korrelation verschwindet aus Symmetriegründen, obwohl $X$ den Wert von $Y$ fast vollständig bestimmt. Die vertikale Skala in (b) ist anders gewählt. (c) Ein unabhängiges Paar. Die Korrelation unterscheidet (b) und (c) nicht, die gegenseitige Information stuft (b) sogar höher ein als (a).](../img/mdl-it-mi-corr-vs-mi.svg)

<!--en:190ed4d7-->
Korrelation und gegenseitige Information unterscheiden sich in einem zweiten grundlegenden Punkt, ihrem Verhalten bei einem Koordinatenwechsel. Ob wir Temperatur in Fahrenheit statt in Celsius messen oder einen Preis auf einer logarithmischen Skala angeben, ändert nichts am Informationsgehalt der Variablen über andere Größen. Ein geeignetes Abhängigkeitsmaß sollte daher invariant bleiben.

<!--en:c93a9b4c-->
**Satz (Invarianz unter Reparametrisierung).** *Seien $\phi$ und $\psi$ invertierbare Abbildungen der Werte von $X$ beziehungsweise $Y$. Dann gilt $I(\phi(X);\psi(Y))=I(X;Y)$.*

<!--en:23df772b-->
**Beweis.** Bei diskreten Variablen benennen invertierbare Abbildungen die Ergebnisse nur um. Die gemeinsame Verteilung von $(\phi(X), \psi(Y))$ weist den umbenannten Ergebnissen dieselben Wahrscheinlichkeiten zu wie die Verteilung von $(X, Y)$ den ursprünglichen. Die Summe in :eqref:`eq_mdl-mut_ent_def` bleibt daher unverändert. Für stetige Variablen beweisen wir hier den glatten Fall, um den Mechanismus zu zeigen. In :numref:`subsec_mdl-dpi` entfällt die Glattheitsannahme. Seien $\phi$ und $\psi$ stetig differenzierbar mit stetig differenzierbaren Umkehrabbildungen. Wir wechseln die Variablen gemäß $(x, y) \mapsto (\phi(x), \psi(y))$. Die gemeinsame Dichte des transformierten Paares an der Stelle $(\phi(x),\psi(y))$ ist $p_{X,Y}(x,y)\,|\det D\phi(x)|^{-1}|\det D\psi(y)|^{-1}$. Die beiden Randdichten erhalten die Faktoren $|\det D\phi(x)|^{-1}$ und $|\det D\psi(y)|^{-1}$. Diese Jacobi-Faktoren kürzen sich im Dichteverhältnis. Der Integrand in :eqref:`eq_mdl-mut_ent_def` hat daher an entsprechenden Punkten denselben Wert, und sein Erwartungswert bleibt unverändert. Ausnahmemengen mit Wahrscheinlichkeit null, etwa der Ursprung bei $x\mapsto x^3$, ändern das Integral nicht. $\blacksquare$

<!--en:94da0b0b-->
Der differentiellen Entropie fehlt diese Invarianz. Ihr Jacobi-Faktor hat keinen Gegenfaktor, mit dem er sich kürzen könnte. Deshalb hängt $h$ von den Maßeinheiten ab. Die Korrelation bleibt bis auf das Vorzeichen nur unter affinen Transformationen allgemein erhalten. Wenden wir etwa die invertierbare Abbildung $x\mapsto x^3$ auf die erste Koordinate des gaußschen Paares mit $\rho=0.9$ an, bleibt die gegenseitige Information bei $0.830$ Nat. Die Korrelation sinkt dagegen von $0.9$ auf $3\rho/\sqrt{15}=0.697$ (denn es gelten $\mathbb{E}[X^3Y]=3\rho$ und $\textrm{Var}(X^3)=15$ für standardnormalverteiltes $X$). Die gegenseitige Information fragt, ob zwei Variablen in *beliebigen* Koordinaten Information übereinander enthalten. Ob sie zufällig längs einer Geraden liegen, ist dabei unerheblich.

<!--en:49768418-->
## Verarbeitung und Vorhersage

<!--en:c38ba4fc-->
Wir leiten nun die drei in der Einleitung angekündigten Ergebnisse her. Die ersten beiden setzen grundlegende Grenzen. Datenverarbeitung kann keine Information erzeugen, und eine genaue Vorhersage erfordert ein Mindestmaß an Information. Beides folgt aus einer einzigen Identität, der Kettenregel für gegenseitige Information. Das dritte Ergebnis liefert aus dem Kreuzentropieverlust jedes probabilistischen Modells direkt eine untere Schranke für die Information in seiner Eingabe.

<!--en:f53934d2-->
### Bedingte gegenseitige Information und Kettenregel

<!--en:5397a8eb-->
Erweitern wir das Beispiel des gekippten Bits um einen Schritt. Angenommen, $Z$ entsteht, indem $Y$ unabhängig vom ersten Kippen mit Wahrscheinlichkeit $0.2$ erneut kippt. Das entspricht einem gleichverteilten Bit auf zwei hintereinandergeschalteten verrauschten Leitungen oder einem Label, gefolgt von einem verrauschten Merkmal und dessen verrauschter Kopie (:numref:`fig_mdl-it-bsc-chain`). Die beiden Schritte wirken wie ein einziger. Das Bit kommt genau dann invertiert an, wenn es auf genau einer der Leitungen kippt. Die Wahrscheinlichkeit dafür ist $0.1\cdot0.8+0.9\cdot0.2=0.26$. Das Paar $(X,Z)$ ist also ein gekipptes Bit mit $\delta=0.26$. Damit erhalten wir $I(X;Z)=\ln2-H_b(0.26)=0.120$ Nat. Die zweite verrauschte Leitung hat zwei Drittel der $0.368$ Nat ausgelöscht, die $Y$ enthielt.

<!--en:22e280fd-->
![Zweimaliges Kippen in Folge. Ein gleichverteiltes Bit $X$ kippt mit Wahrscheinlichkeit $0.1$ und ergibt $Y$. Anschließend kippt $Y$ mit Wahrscheinlichkeit $0.2$ und ergibt $Z$. Zusammengenommen entspricht das einem einzigen Kippvorgang mit Wahrscheinlichkeit $0.26$. Der zweite Schritt entfernt zwei Drittel der Information. Dabei besagt $I(X;Z\mid Y)=0$, dass $Z$ nichts über $X$ weiß, was $Y$ nicht bereits weiß.](../img/mdl-it-bsc-chain.svg)

<!--en:d1a95bb9-->
Bei drei Variablen stellt sich eine neue Frage. Wie viel Information teilen $X$ und $Y$, wenn $Z$ bereits bekannt ist? Wie für die Entropie gibt es auch für die gegenseitige Information eine bedingte Variante:

<!--en:873d43c4-->
Dabei ist $H(X\mid Y,Z)$ die bedingte Entropie von $X$ bei gegebenem Paar $(Y,Z)$. Schreiben wir jede bedingte Entropie als Erwartungswert und fassen die Logarithmen zusammen, erhalten wir die Divergenzform $I(X; Y \mid Z) = \mathbb{E}_{z}\big[D_{\textrm{KL}}(P_{X,Y \mid z}\,\|\, P_{X \mid z} \otimes P_{Y \mid z})\big]$. Sie mittelt die gewöhnliche gegenseitige Information der bedingten Verteilungen über die Werte von $z$. Als Erwartungswert von KL-Divergenzen ist $I(X; Y \mid Z)$ stets nichtnegativ. Der Wert ist genau dann null, wenn $X$ und $Y$ bei gegebenem $Z$ bedingt unabhängig sind.

<!--en:852dd854-->
Mit dieser Definition können wir die Information, die das Paar $(Y,Z)$ über $X$ enthält, in zwei Teile zerlegen. Subtrahieren und addieren wir $H(X\mid Z)$ in $I(X;Y,Z)=H(X)-H(X\mid Y,Z)$, so erhalten wir

<!--en:4e9e1e79-->
Diese Identität heißt *Kettenregel für gegenseitige Information* (engl. chain rule for mutual information). In der KL-Darstellung aller Terme gilt sie ohne Einschränkung. Was $(Y,Z)$ über $X$ verrät, ist also die Summe dessen, was $Z$ verrät und was $Y$ bei bekanntem $Z$ hinzufügt. Ebenso können wir zuerst auf $Y$ konditionieren. Bei unseren beiden Kippvorgängen zerlegen die beiden Reihenfolgen dieselbe Summe auf zwei Arten. Da $Z$ aus $Y$ und unabhängigem Rauschen entsteht, gilt bei bekanntem $Y$, dass $Z$ nichts mehr über $X$ verrät. Es gilt $I(X;Z\mid Y)=0$. Beginnen wir mit $Y$, so erhalten wir $I(X;Y,Z)=I(X;Y)+0=0.368$ Nat. Beginnen wir mit $Z$, so zerfallen dieselben $0.368$ Nat gemäß $I(X;Z)+I(X;Y\mid Z)=0.120+I(X;Y\mid Z)$, und es verbleiben $I(X;Y\mid Z)=0.248$ Nat. Hier verringert die Konditionierung auf $Z$ das, was $Y$ über $X$ verrät, weil $Z$ bereits einen Teil dieser Information enthält. Wie wir im nächsten Unterabschnitt sehen, kann Konditionieren die gegenseitige Information aber auch *erhöhen*.

<!--en:d2282988-->
### Die Datenverarbeitungsungleichung

<!--en:3f8a6e7a-->
In einem neuronalen Netz berechnet sich jede versteckte Schicht aus der darunterliegenden Schicht und diese wiederum aus der Eingabe. Kann diese Verarbeitung Information über die Eingabe erzeugen, die in der früheren Schicht noch fehlte? Bei den beiden verrauschten Leitungen war die Antwort nein. Der Grund dafür hat nichts mit den Besonderheiten von Bits zu tun.

<!--en:c14ceaab-->
Drei Zufallsvariablen $X, Y, Z$ bilden eine *Markov-Kette*, geschrieben $X \to Y \to Z$, wenn $Z$ von $(X, Y)$ nur über $Y$ abhängt, also $p(z \mid x, y) = p(z \mid y)$ gilt. Genau diese Struktur hat eine schrittweise Verarbeitung. $Y$ entsteht aus $X$, etwa als Messung, Merkmalskarte oder versteckte Schicht. Anschließend berechnet sich $Z$ allein aus $Y$, gegebenenfalls unter erneutem Zufallseinfluss. Gleichwertig dazu sind $X$ und $Z$ bei gegebenem $Y$ bedingt unabhängig. Also gilt $I(X; Z \mid Y) = 0$.

<!--en:0280bcdf-->
**Satz (Datenverarbeitungsungleichung).** *Ist $X \to Y \to Z$ eine Markov-Kette und sind die beteiligten Entropien endlich, so gilt*

<!--en:62d792b6-->
*mit Gleichheit genau dann, wenn $I(X; Y \mid Z) = 0$ gilt, also genau dann, wenn auch $X \to Z \to Y$ eine Markov-Kette ist* :cite:`Cover.Thomas.1999`.

<!--en:29c52487-->
**Beweis.** Wir zerlegen $I(X; Y, Z)$ mit der Kettenregel in beiden Reihenfolgen:

<!--en:cbde2c4e-->
Wegen der Markov-Eigenschaft verschwindet der letzte Term. Da die bedingte gegenseitige Information nichtnegativ ist, folgt $I(X; Z) = I(X; Y) - I(X; Y \mid Z) \leq I(X; Y)$. Gleichheit gilt genau dann, wenn $I(X; Y \mid Z) = 0$ gilt. $\blacksquare$

<!--en:77a85d9d-->
Keine Verarbeitung von $Y$, ob deterministisch oder zufällig, kann die darin enthaltene Information über $X$ erhöhen. Für unsere beiden Kippvorgänge lautet die Ungleichung $0.120 \le 0.368$.

<!--en:16d0730e-->
Die Ungleichung beruht auf der Struktur der Markov-Kette. Beim Konditionieren außerhalb einer Markov-Kette kann eine zuvor nicht vorhandene Abhängigkeit entstehen. Seien $X$ und $Y$ unabhängige gleichverteilte Bits und $Z = X \oplus Y$ ihr exklusives Oder. Für sich genommen teilen $X$ und $Y$ keine Information. Sobald $Z$ bekannt ist, bestimmt $Y$ jedoch $X$ vollständig. Der folgende Code berechnet die gegenseitige Information sowohl für die beiden Kippvorgänge als auch für dieses XOR-Tripel.

<!--en:c543dbf4-->
Für die beiden Kippvorgänge bestätigt $I(X; Z \mid Y) = 0$ numerisch die Markov-Eigenschaft. Beide Zerlegungen von $I(X; Y, Z)$ nach der Kettenregel ergeben $0.3681$ Nat. Beim exklusiven Oder erhöht die Konditionierung auf $Z$ die gegenseitige Information zwischen $X$ und $Y$ von $0$ auf die vollen $\ln2=0.693$ Nat eines Bits.

<!--en:e556ae03-->
Wann geht bei der Verarbeitung überhaupt keine Information verloren? Nach dem Satz gilt Gleichheit genau dann, wenn auch $X\to Z\to Y$ eine Markov-Kette ist. Dann bewahrt $Z$ alles, was $Y$ über $X$ verrät. Eine invertierbare Transformation $Z=g(Y)$ leistet das immer, weil sich $Y$ aus $Z$ zurückgewinnen lässt. Eine Abbildung darf aber auch einen Teil von $Y$ verwerfen, ohne Information über $X$ zu verlieren, sofern das Verworfene keinen Bezug zu $X$ hat. So bewahrt die Anzahl der Kopfwürfe alles, was eine Folge von Münzwürfen über die Kopfwahrscheinlichkeit verrät. Die Reihenfolge der Würfe enthält keine Information darüber. Jede Zusammenfassung $T(Y)$ mit dieser Eigenschaft heißt *suffiziente Statistik* (engl. sufficient statistic). Dabei hängt $p(x \mid y)$ von $y$ nur über $T(y)$ ab, und daher gilt $I(X;T(Y))=I(X;Y)$. Die suffizienten Statistiken einer Exponentialfamilie (:numref:`sec_mdl-exponential-family`) sind genau solche Zusammenfassungen. Dabei übernimmt $Y$ die Rolle der Daten und $X$ die des Parameters $\theta$ (mit einer A-priori-Verteilung, damit $I(\theta;Y)$ wohldefiniert ist).

<!--en:68b7cba0-->
Die Datenverarbeitungsungleichung selbst setzt weder endliche Entropien noch Dichten voraus. Die gegenseitige Information ist eine KL-Divergenz, und keine zufällige oder deterministische Abbildung der Daten kann eine KL-Divergenz erhöhen. In :numref:`subsec_mdl-f-div-data-processing` finden wir den Beweis für abzählbare Mengen, die Aussage gilt aber allgemein. Aufgabe :exref:`ex_mdl-mutual-information-dpi-from-f-divergence` nutzt dies für einen allgemeinen Beweis der Ungleichung. Wenden wir sie einmal auf eine invertierbare Abbildung und einmal auf deren Umkehrabbildung an, erhalten wir auch die Invarianz unter Reparametrisierung aus :numref:`sec_mdl-mi-nonlinear-correlation` ohne Glattheitsannahme.

<!--en:2e54e127-->
Bei tiefen Netzen lässt sich die Datenverarbeitungsungleichung leicht missverstehen. Die Schichten eines Netzes bilden eine Markov-Kette $X \to Z_1 \to Z_2 \to \cdots$. Daher kann $I(X; Z_\ell)$ mit der Tiefe $\ell$ nicht zunehmen. Bis zum Ende dieses Teils bezeichnet $Y$ nun das vorherzusagende Label anstelle der mittleren Variablen einer Kette. Alles, was das Netz über das Label weiß, erreicht es über die Eingabe. Daher ist auch $Y \to X \to Z_1 \to Z_2 \to \cdots$ eine Markov-Kette, und auch $I(Y; Z_\ell)$ kann mit $\ell$ nicht zunehmen. Keine Architektur kann Information über das Label erzeugen, die der Eingabe fehlt. Keine spätere Schicht kann Information zurückgewinnen, die eine frühere verworfen hat.

<!--en:d40c7a80-->
Daraus folgt *nicht*, dass Tiefe nutzlos ist. Versteckte Schichten ordnen die vorhandene Information über das Label so um, dass eine einfache letzte Schicht sie auslesen kann, ohne dass Information hinzukommt. Sei das Label $Y$ ein gleichverteiltes Bit und die Eingabe $X \in \mathbb{R}^2$ gleichverteilt auf dem Kreis mit Radius $1$ für $Y=0$ und auf dem Kreis mit Radius $2$ für $Y=1$. Die Eingabe bestimmt das Label exakt, also gilt $I(Y;X)=\ln2$. Dennoch trennt keine Gerade zwei konzentrische Kreise. Der beste lineare Klassifikator legt seine Entscheidungsgrenze tangential an den inneren Kreis. Er klassifiziert diesen vollständig korrekt und ebenso das Drittel des äußeren Kreises jenseits der Geraden. Seine Genauigkeit beträgt $2/3$. Bilden wir nun ein einziges nichtlineares Merkmal $Z=\|X\|^2$. Es nimmt nur die Werte $1$ und $4$ an. Eine Schwelle dazwischen klassifiziert jeden Punkt korrekt, obwohl $I(Y;Z)=\ln2$ nicht größer ist als $I(Y;X)$. Die Transformation hat keine Information hinzugefügt, aber die gesamte Information linear auslesbar gemacht. Aufgabe :exref:`ex_mdl-mutual-information-usable-information` formalisiert einen Begriff der nutzbaren Information, der nur erfasst, was eine eingeschränkte Familie von Klassifikatoren auslesen kann.

<!--en:1ad93300-->
Für die Eingabekette $X \to Z_1 \to Z_2 \to \cdots$ ist noch eine Besonderheit zu beachten. Bei einer deterministischen Schicht gilt $H(Z_\ell\mid X)=0$. Ist die Repräsentation diskret oder quantisiert, so reduziert sich $I(X;Z_\ell)$ auf die Entropie $H(Z_\ell)$. Ist sie stetig, so ist $I(X;Z_\ell)$ in der Regel unendlich (:numref:`subsec_mdl-mi-gaussian`). Als Maß dafür, wie viel eine Schicht über ihre Eingabe bewahrt, ist $I(X;Z_\ell)$ daher nur aussagekräftig, wenn die Schicht Rauschen hinzufügt oder eine diskrete Ausgabe erzeugt.

<!--en:4968dd42-->
### Die Fano-Ungleichung

<!--en:91fffbba-->
Genaue Vorhersagen erfordern gegenseitige Information. Erreicht ein Klassifikator eine Fehlerrate von $5\%$ bei $1{,}000$ Klassen, so muss seine Eingabe den größten Teil der Entropie des Labels enthalten. Die Fano-Ungleichung gibt an, wie viel genau erforderlich ist.

<!--en:cefdef4c-->
**Satz (Fano-Ungleichung).** *Sei $Y$ eine Variable mit Werten in einer Menge von $k$ Elementen. Sei $\hat Y=g(X)$ eine beliebige Schätzung von $Y$ aus $X$ mit Werten in derselben Menge und $P_e=P(\hat Y\ne Y)$ ihre Fehlerwahrscheinlichkeit. Dann gilt*

<!--en:37450ab4-->
Die linke Seite lässt sich als Kodierungsaufwand für die Korrektur der Schätzung $\hat Y$ deuten. Es kostet $H_b(P_e)$ Nat, anzugeben, ob $\hat Y$ falsch ist. Im Fehlerfall (mit Wahrscheinlichkeit $P_e$) genügen höchstens $\log(k-1)$ Nat, um unter den übrigen $k-1$ Labels das richtige zu bezeichnen. Wer Schätzung und Korrektur kennt, kennt $Y$ genau. Die Korrektur kostet also mindestens so viel wie die Unsicherheit über $Y$, die bei bekanntem $\hat Y$ verbleibt. Nach der Datenverarbeitungsungleichung ist diese mindestens so groß wie die Unsicherheit bei bekanntem $X$.

<!--en:2448cdd3-->
Der Beweis macht aus dieser Überlegung eine exakte Schranke. Dazu benötigen wir nur eine Indikatorvariable, die anzeigt, ob ein Fehler aufgetreten ist. Den Rest liefert die Kettenregel.

<!--en:116d0fb8-->
**Beweis.** Sei $W = \mathbf{1}\{\hat Y \neq Y\}$ der Fehlerindikator. Wir zerlegen $H(W, Y \mid \hat Y)$, die bei bekannter Schätzung verbleibende Unsicherheit über den Fehler und das wahre Label, auf zwei Arten. Erstens ist $W$ durch $(Y,\hat Y)$ vollständig bestimmt und trägt somit keine zusätzliche Entropie bei. Daher gilt $H(W, Y \mid \hat Y) = H(Y \mid \hat Y)$. Zweitens liefert die Kettenregel $H(W, Y \mid \hat Y) = H(W \mid \hat Y) + H(Y \mid W, \hat Y)$. Wir schätzen beide Terme einzeln ab:

<!--en:a03058b1-->
* Konditionieren kann die Entropie nicht erhöhen, denn $H(W)-H(W\mid\hat Y)=I(W;\hat Y)\ge0$. Daher gilt $H(W \mid \hat Y) \leq H(W) = H_b(P_e)$.

<!--en:b4727a8a-->
* Für $W=0$ ist das Label gleich $\hat Y$ und die Unsicherheit null. Für $W=1$ (mit Wahrscheinlichkeit $P_e$) muss das Label einen der $k-1$ Werte außer $\hat Y$ annehmen. Seine Entropie beträgt also höchstens $\log(k-1)$. Mitteln über $W$ ergibt $H(Y \mid W, \hat Y) \leq P_e \log(k-1)$.

<!--en:87d032b9-->
Zusammen ergeben beide Schranken $H(Y\mid\hat Y)\le H_b(P_e)+P_e\log(k-1)$. Da sich $\hat Y$ ausschließlich aus $X$ berechnet, bildet das Tripel $Y \to X \to \hat Y$ eine Markov-Kette. Die Datenverarbeitungsungleichung ergibt $I(Y; \hat Y) \leq I(Y; X)$ oder gleichwertig $H(Y \mid \hat Y) \geq H(Y \mid X)$. $\blacksquare$

<!--en:f278d399-->
Die Anzahl von $k-1$ verbleibenden Labels im zweiten Schritt setzt voraus, dass $\hat Y$ Werte in derselben Labelmenge annimmt. Darf der Klassifikator mit „Enthaltung“ antworten, muss $\log k$ an die Stelle von $\log(k-1)$ treten. Die angegebene Ungleichung gilt dann nicht mehr. Für ein gleichverteiltes Bit $Y$ unabhängig von $X$ und einen Klassifikator, der sich immer enthält, gilt $P_e=1$. Die linke Seite wird zu $H_b(1)+\ln1=0$ und liegt damit unter $H(Y\mid X)=\ln2$.

<!--en:9b230c79-->
Im einfachsten Fall ist die Fano-Schranke exakt. Für das gekippte Bit mit der Schätzung $\hat Y=X$ gelten $P_e=0.1$ und $k=2$. Die linke Seite ist also gleich $H_b(0.1)+0.1\ln1=0.325$ und stimmt genau mit $H(Y\mid X)$ überein.

<!--en:1b828271-->
Kein Klassifikator kann die Fano-Schranke unterbieten, unabhängig von Architektur und Trainingsverfahren. Für $1{,}000$ gleich wahrscheinliche Klassen gilt $H(Y)=\ln1000=6.91$ Nat. Stellen wir :eqref:`eq_mdl-fano` nach $I(X;Y)$ um, erhalten wir die gegenseitige Information, die für die jeweilige angestrebte Fehlerrate mindestens erforderlich ist.

<!--en:a594b16d-->
Eine Fehlerrate von $5\%$ erfordert mindestens $6.36$ der $6.91$ Nat an Labelentropie ($92\%$ der Gesamtentropie). Selbst ein Klassifikator, der sich in der Hälfte der Fälle irrt, benötigt $2.76$ Nat, also zwei Fünftel der Gesamtentropie.

<!--en:eee66f95-->
### Der Verlust als Informationsnachweis

<!--en:44130fe4-->
Die Fano-Ungleichung liefert aus der Fehlerrate einer Klassifikation eine untere Schranke für die gegenseitige Information. Ein Sprachmodell wird jedoch kaum anhand der Fehlerrate seiner eindeutigen Entscheidungen bewertet. Wie bei fast jedem mittels Maximum-Likelihood trainierten Modell gibt man seinen Kreuzentropieverlust an. Auch dieser Verlust liefert eine untere Schranke für die gegenseitige Information. Der Beweis benötigt nur drei Zeilen.

<!--en:e01c2f93-->
**Satz (Informationsnachweis durch den Verlust).** *Sei $q(y\mid x)$ ein beliebiges Modell für das Label bei gegebener Eingabe mit Kreuzentropieverlust $L=\mathbb{E}[-\log q(Y\mid X)]$ unter $P_{X,Y}$. Dann gilt*

<!--en:00387ed3-->
*und für $k$ gleich wahrscheinliche Klassen gilt $I(X;Y)\ge\log k-L$.*

<!--en:7d2fe50e-->
**Beweis.** Nach Definition gilt $I(X;Y)=H(Y)-H(Y\mid X)$. Für jedes $x$ übersteigt die Kreuzentropie von $q(\cdot\mid x)$ unter $p(\cdot\mid x)$ die wahre bedingte Entropie von $p(\cdot\mid x)$ um $D_{\textrm{KL}}(p(\cdot\mid x)\,\|\,q(\cdot\mid x))\ge0$ (:numref:`sec_mdl-information_theory`). Mitteln über $x$ ergibt $H(Y\mid X)\le L$. Ziehen wir beide Seiten jeweils von $H(Y)$ ab, folgt die Behauptung. $\blacksquare$

<!--en:407f8237-->
Der Verlust auf zurückgehaltenen Testdaten schätzt $L$. Daher liefert der Testverlust jedes Klassifikators oder Sprachmodells direkt einen Nachweis für eine untere Schranke der Information in seiner Eingabe. Bei $10$ gleich wahrscheinlichen Klassen und einem Testverlust von $0.3$ Nat enthält die Eingabe zum Beispiel mindestens $\ln10-0.3=2.0$ der $2.30$ Nat des Labels. Die Schranke gilt auch für ein schlecht kalibriertes oder schlecht trainiertes Modell. Ein schwächeres Modell weist lediglich eine kleinere untere Schranke nach. Dies ist die Variationsschranke aus :citet:`Barber.Agakov.2003` mit einem Klassifikator als Decoder. In :numref:`sec_mdl-mi-variational-bounds` entwickeln wir daraus eine umfassendere Familie von Schätzern der gegenseitigen Information.

<!--en:fddc9ba5-->
Die Schranke ist genau dann scharf, wenn $q(y\mid x)$ mit der wahren bedingten Verteilung $p(y\mid x)$ übereinstimmt. Sagt ein Modell beim gekippten Bit $\hat Y=X$ mit Konfidenz $c$ vorher, beträgt sein erwarteter Verlust $L=-0.9\log c-0.1\log(1-c)$. Wir berechnen ihn im Folgenden für vier Konfidenzniveaus.

<!--en:b5bf877e-->
Nur das kalibrierte Modell ($c=0.9$) weist die vollen $0.368$ Nat nach. Ein zu selbstsicheres Modell ($c=0.99$) erleidet bei jedem Fehler einen hohen Verlust und weist nur $0.22$ Nat nach. Ein zu vorsichtiges Modell ($c=0.6$) weist $0.14$ Nat nach. Die Klassenvorhersagen sind jedoch bei allen vier Modellen gleich, und alle irren sich in $10\%$ der Fälle. Die Fano-Ungleichung berücksichtigt nur die Fehlerrate und weist für jedes Modell die vollen $0.368$ Nat nach. Der Kreuzentropieverlust berücksichtigt auch die vorhergesagten Wahrscheinlichkeiten und bestraft Fehlkalibrierung. Beide Ergebnisse beschränken dieselbe bedingte Entropie $H(Y\mid X)$ nach oben, einmal über die Fehlerrate und einmal über den Verlust. Aus jeder oberen Schranke folgt eine untere Schranke für $I(X;Y)$. Der Zusammenhang ist sogar noch enger. Aufgabe :exref:`ex_mdl-mutual-information-certificate-and-fano` zeigt, dass die Fano-Schranke selbst der Informationsnachweis aus dem Verlust eines bestimmten Modells ist. Ein gut kalibrierter Klassifikator, der weiß, wann er unsicher ist, kann eine echt größere Schranke nachweisen.

<!--en:a19060e9-->
## Kommunikation und Kompression

<!--en:dfe25c75-->
Shannon stellte zwei grundlegende technische Fragen. Wie viel Information kann ein verrauschter Kanal zuverlässig übertragen? Wie wenig Information muss ein verlustbehafteter Kompressor bewahren? Die Antworten erweisen sich als Maximum und Minimum der gegenseitigen Information. Wenn Sie sich vor allem für Repräsentationslernen interessieren, können Sie die Kanalkapazität beim ersten Lesen überfliegen. Die Rate-Verzerrungs-Theorie ist dagegen wesentlich, denn der letzte Teil dieses Abschnitts baut unmittelbar darauf auf.

<!--en:0efd7e03-->
### Kanalkapazität

<!--en:8a9e12e9-->
Wie viel Information übersteht die Übertragung auf einer Leitung, auf der jedes Bit unabhängig mit Wahrscheinlichkeit $\delta$ kippt? Ein gedächtnisloser *Kanal* ist durch eine bedingte Verteilung $p(y\mid x)$ gegeben, die für jedes Eingabesymbol unabhängig gilt. Jedes übertragene Symbol zählt als eine Kanalnutzung. Unser gekipptes Bit ist genau so ein Kanal mit Eingabe $X$ und Ausgabe $Y$. Das Kanalrauschen ist vorgegeben, doch der Sender kann die Eingabeverteilung $p(x)$ frei wählen. Die *Kapazität* des Kanals ist die maximale gegenseitige Information über alle Eingabeverteilungen, gemessen in Nat pro Kanalnutzung:

<!--en:56bb54e2-->
Ein Sender, der fast nur Nullen überträgt, nutzt den Kanal schlecht aus, denn eine solche Eingabe hat von vornherein wenig Entropie. Shannons Satz zur Kanalkodierung gibt der Kapazität ihre operationale Bedeutung. Definieren wir die *Rate* eines Codes als die Anzahl der pro Kanalnutzung übermittelten Nat an Nachrichteninformation. Shannon zeigte, dass lange Symbolblöcke bei jeder Rate unter $C$ eine zuverlässige Übertragung ermöglichen, also mit beliebig kleiner Fehlerwahrscheinlichkeit. Bei jeder Rate über $C$ ist zuverlässige Kommunikation unmöglich :cite:`Shannon.1948,Cover.Thomas.1999`. Wir geben den Satz ohne Beweis an.

<!--en:6cbd7c60-->
Für den Kanal, auf dem Bits kippen, den *binären symmetrischen Kanal* (engl. binary symmetric channel), genügt zur Berechnung der Kapazität eine Zeile. Schreiben wir $I(X;Y)=H(Y)-H(Y\mid X)$. Jede Zeile von $p(y\mid x)$ ist bis auf die Reihenfolge gleich $(1-\delta,\delta)$, also gilt $H(Y\mid X)=H_b(\delta)$ unabhängig von der Eingabeverteilung. Außerdem gilt $H(Y)\le\ln2$ mit Gleichheit genau dann, wenn $Y$ gleichverteilt ist. Wegen $P(Y=1)=\delta+P(X=1)(1-2\delta)$ ist für $\delta\ne\tfrac12$ die Ausgabe genau dann gleichverteilt, wenn es die Eingabe ist. Somit gilt

<!--en:1625d486-->
und die gleichverteilte Eingabe erreicht diesen Wert. Bei $\delta=0.1$ erhalten wir wieder die $0.368$ Nat unseres gekippten Bits, dessen Eingabe bereits gleichverteilt war. Bei zwei Leitungen in Reihe ($\delta=0.26$) sinkt die Kapazität auf $0.120$ Nat. Bei $\delta=\tfrac12$ ist die Ausgabe von der Eingabe unabhängig und die Kapazität verschwindet. Bei $\delta=1$ erreicht die Kapazität wieder die vollen $\ln2$ Nat. Weiß der Empfänger, dass jedes Bit invertiert wird, kann er es wieder umkehren. Rauschen zerstört Information nur in dem Maße, in dem es für den Empfänger unvorhersehbar ist. Der folgende Code variiert zunächst die Eingabeverteilung für den Kanal mit $\delta=0.1$ und anschließend die Kippwahrscheinlichkeit $\delta$ bei gleichverteilter Eingabe.

<!--en:fd72b7d0-->
Eine Eingabe, die eine $1$ nur in $10\%$ der Fälle sendet, bietet nur $H_b(0.1)=0.325$ Nat und überträgt davon $0.146$. Das sind zwei Fünftel dessen, was die gleichverteilte Eingabe erreicht. Links in :numref:`fig_mdl-it-capacity-rd` sehen wir die Kapazität als Funktion von $\delta$. Auch neuronale Netze enthalten Kanäle. Dropout löscht ein Merkmal mit Wahrscheinlichkeit $\alpha$ und lässt es andernfalls passieren. Aufgabe :exref:`ex_mdl-information-theory-binary-channels` zeigt, dass ein binäres Merkmal auf diesem Weg höchstens $(1-\alpha)\ln2$ Nat übertragen kann.

<!--en:f3270016-->
![Shannons zwei Kurven für Übertragung und Kompression, in Nat. Links die Kapazität $C=\ln2-H_b(\delta)$ des binären symmetrischen Kanals. Markiert sind das gekippte Bit ($\delta=0.1$) und die beiden Kippvorgänge in Folge ($\delta=0.26$). Ein Kanal mit Kippwahrscheinlichkeit $1/2$ überträgt keine Information. Kippt das Bit immer, überträgt er die vollen $\ln2$. Rechts die gaußsche Rate-Verzerrungs-Funktion für eine Quelle mit Varianz $\sigma_X^2=1$. Die Rate divergiert für $D\to0$ und erreicht null, sobald $D$ die Quellenvarianz erreicht.](../img/mdl-it-capacity-rd.svg)

<!--en:f485ea58-->
### Rate und Verzerrung

<!--en:e0a42511-->
Verlustfreie Kompression erfordert $H(X)$ Nat pro Symbol (:numref:`sec_mdl-coding-compression`). Ein exakt verlustfreier Code für eine stetige Quelle würde damit unendlich viele Nat erfordern. In der Praxis genügt meist eine näherungsweise Rekonstruktion. Ein Bildcodec darf Pixelwerte leicht verändern, und eine gelernte Repräsentation kann irrelevante Details gezielt verwerfen. Wir wählen ein nichtnegatives *Verzerrungsmaß* $d(x,\hat x)\ge0$, das bewertet, wie gut $\hat x$ den Wert $x$ rekonstruiert, und erlauben eine mittlere Verzerrung von höchstens $D$. Wie wenige Nat pro Quellenwert genügen? Die *Rate-Verzerrungs-Funktion* (engl. rate–distortion function) gibt die minimale gegenseitige Information an, die ein Encoder $p(\hat x\mid x)$ bewahren muss, um diese Verzerrungsschranke einzuhalten:

<!--en:7d71cff4-->
Eine engere Verzerrungsschranke verlangt mehr Information. Shannons Rate-Verzerrungs-Satz ist das Gegenstück zum Kanalkodierungssatz. Die Rate misst jetzt, wie viele Nat ein Code pro Quellenwert aufwendet. Bei jeder Rate über $R(D)$ können Blockcodes die mittlere Verzerrung $D$ erreichen. Bei einer Rate unter $R(D)$ ist das mit keinem Code möglich :cite:`Shannon.1959`.

<!--en:ebfc90b1-->
In der Praxis ist es meist einfacher, die feste Nebenbedingung für die Verzerrung durch einen Strafterm zu ersetzen. Dazu minimieren wir $I(X;\hat X)+\lambda\,\mathbb{E}[d(X,\hat X)]$ über alle Encoder für einen Multiplikator $\lambda>0$. Jeder Encoder entspricht einem Punkt in der Verzerrungs-Raten-Ebene auf oder oberhalb der Kurve $R(D)$. Die Zielfunktion mit Strafterm ist auf Geraden der Steigung $-\lambda$ konstant. Da $R(D)$ konvex ist, liegt das Minimum dort, wo eine Gerade mit Steigung $-\lambda$ die Kurve von unten berührt. Ein größerer Wert von $\lambda$ führt zu geringerer Verzerrung bei höherer Rate. Der Informationsflaschenhals in :numref:`sec_mdl-information-bottleneck` verwendet genau diese Lagrange-Form.

<!--en:db6dcac7-->
**Satz (gaußsche Rate-Verzerrungs-Funktion).** *Für eine Quelle $X\sim\mathcal N(0,\sigma_X^2)$ mit quadratischem Fehler $d(x,\hat x)=(x-\hat x)^2$ gilt*

<!--en:32b03cfb-->
Lösen wir diese Formel nach der Verzerrung auf, ergibt sich $D=\sigma_X^2 e^{-2R}$. Jedes zusätzliche Bit an Rate ($\ln2$ Nat) senkt den quadratischen Fehler auf ein Viertel.

<!--en:babea475-->
**Beweis.** Der wesentliche Schritt ist die untere Schranke. Sie beruht auf einem einzigen Ergebnis aus :numref:`subsec_mdl-gaussian-max-entropy`. Unter allen Dichten mit Varianz $s^2$ erreicht die Normalverteilung die maximale differentielle Entropie $\tfrac12\log(2\pi e s^2)$. Diese wächst mit $s^2$. Betrachten wir einen beliebigen Encoder, der $\mathbb{E}[(X-\hat X)^2]\le D$ erfüllt. Dann gilt

<!--en:3fa9d5f6-->
Die zweite Gleichheit gilt, weil die Subtraktion des bekannten Werts $\hat X$ die bedingte Dichte nur verschiebt, ohne ihre Entropie zu ändern. Die erste Ungleichung nutzt aus, dass Konditionieren die Entropie nicht erhöhen kann. Die zweite nutzt die gaußsche Schranke für die maximale Entropie, denn der Rekonstruktionsfehler $X-\hat X$ hat höchstens die Varianz $\mathbb{E}[(X-\hat X)^2]\le D$. Für $D\ge\sigma_X^2$ erfüllt bereits ein konstanter Encoder die Vorgabe, der nichts überträgt und stets den Mittelwert $0$ ausgibt. Also gilt $R(D)=0$.

<!--en:54a2851c-->
Nun benötigen wir noch einen Encoder, der diese Schranke für $D<\sigma_X^2$ erreicht. Setzen wir $\sigma_X^2=1$ und addieren unabhängiges gaußsches Rauschen: $Z=X+\sigma\epsilon$ mit $\epsilon\sim\mathcal N(0,1)$ und $\sigma>0$. Dann gelten $\textrm{Cov}(X,Z)=1$ und $\textrm{Var}(Z)=1+\sigma^2$, also beträgt die quadrierte Korrelation $\rho_{XZ}^2=1/(1+\sigma^2)$. Mit :eqref:`eq_mdl-gaussian_mi` erhalten wir die gegenseitige Information, die dieser Encoder bewahrt:

<!--en:930c035d-->
Rekonstruieren wir nun $X$ mit dem bedingten Mittelwert $\hat X=\mathbb{E}[X\mid Z]=Z/(1+\sigma^2)$. Das ist die optimale lineare Regression von $X$ auf $Z$ mit Steigung $\textrm{Cov}(X,Z)/\textrm{Var}(Z)$. Ihr mittlerer quadratischer Fehler entspricht der bedingten Varianz $D=1-1/(1+\sigma^2)=\sigma^2/(1+\sigma^2)$, die das gesamte Intervall $(0,1)$ durchläuft, wenn wir $\sigma$ variieren. Da $\hat X$ aus $Z$ durch invertierbare Skalierung hervorgeht, gilt $I(X;\hat X)=\tfrac12\log(1+\sigma^{-2})=\tfrac12\log(1/D)$. Für eine allgemeine Varianz $\sigma_X^2$ skalieren wir $X$ und $\hat X$ entsprechend. $\blacksquare$

<!--en:b7a7aa6a-->
Der Encoder aus dem Beweis, der ein Signal mit unabhängigem gaußschem Rauschen überlagert, heißt *gaußscher Kanal* mit Signal-Rausch-Verhältnis $\sigma^{-2}$. Seine Kapazität beträgt :eqref:`eq_mdl-gaussian-channel` (Aufgabe :exref:`ex_mdl-mutual-information-gaussian-channel`). Das schrittweise Hinzufügen von Rauschen im Vorwärtsprozess eines Diffusionsmodells (:numref:`chap_diffusion`) ist genau so ein Kanal. Unabhängig von der Datenverteilung enthält eine bis zum Signal-Rausch-Verhältnis $s$ verrauschte Koordinate höchstens $\tfrac12\log(1+s)$ Nat über ihren unverrauschten Wert. Der folgende Code simuliert diesen Encoder für eine Million Samples bei $\sigma^2=1/9$, $1$ und $3$ (entsprechend $D=0.1$, $0.5$ und $0.75$) und überprüft seinen quadratischen Fehler.

<!--en:e0c725cf-->
Die empirischen quadratischen Fehler stimmen bis auf wenige Tausendstel mit $D$ überein. Eine Verzerrung von $D=0.1$ kostet $1.15$ Nat pro Quellenwert, $D=0.5$ kostet $0.35$ Nat (ein halbes Bit) und $D=0.75$ kostet $0.14$ Nat.

<!--en:5defefa6-->
Dieser verrauschte Encoder erreicht $R(D)$ zunächst als gegenseitige Information. Für eine tatsächliche Dateigröße muss ein praktisch einsetzbarer Code die Quellenwerte auf diskrete Bitfolgen abbilden. Der Rate-Verzerrungs-Kurve nähert er sich nur, wenn er lange Blöcke von Quellenwerten gemeinsam kodiert. Jeder skalare Code, der jeweils einen einzelnen Quellenwert quantisiert, bleibt strikt oberhalb der Kurve. Das einfachste Beispiel bewahrt nur das Vorzeichen von $X$. Es benötigt $\ln2=0.693$ Nat und rekonstruiert $\pm\sqrt{2/\pi}$ (im Betrag den Mittelwert von $|X|$). Dabei entsteht ein quadratischer Fehler von $1-2/\pi=0.363$, während die Rate-Verzerrungs-Kurve für diese Verzerrung nur $0.506$ Nat verlangt.

<!--en:2bc2a9e3-->
Rechts in :numref:`fig_mdl-it-capacity-rd` sehen wir die vollständige Kurve. Für $D\to0$ wächst die erforderliche Rate unbegrenzt, denn eine exakte Kopie einer stetigen Variablen enthält unendlich viel Information. Die Rate sinkt auf $0$, sobald $D$ den Wert $\sigma_X^2$ erreicht. Der steile Anstieg nahe $D=0$ erklärt, warum moderne Bildgeneratoren Bilder vor der Modellierung komprimieren. Die feinsten, kaum wahrnehmbaren Pixeldetails beanspruchen den weitaus größten Teil der Bits. Daher überlässt :numref:`sec_diffusion-latent` sie einem Autoencoder.

<!--en:e756e257-->
## Der Informationsflaschenhals

<!--en:2a9ea7e5-->
Eine gute Repräsentation sollte bewahren, was zur Vorhersage des Labels beiträgt, und den Rest verwerfen. Zunächst mag jeder Informationsverlust nachteilig erscheinen. Nach der Datenverarbeitungsungleichung kann er den Bayes-optimalen Prädiktor niemals verbessern. In der Praxis verursachen irrelevante Details jedoch Kosten. Das Foto eines Hundes hält auch die Beleuchtung, den Rasen und die zufällige Position des Hundes im Bild fest. All diese Einzelheiten zu speichern kostet zusätzliche Nat. Ein Klassifikator, der mit endlich vielen Bildern trainiert wurde, kann sie leicht für echte Hinweise auf das Label halten. Der Informationsflaschenhals formalisiert diese Abwägung mithilfe zweier gegenseitiger Informationen. Es handelt sich um ein Rate-Verzerrungs-Problem, bei dem das Label selbst die Verzerrung definiert.

<!--en:d301883b-->
### Die Zielfunktion des Informationsflaschenhalses

<!--en:edcfbabd-->
Sei $Z$ eine Repräsentation der Eingabe $X$, die ein stochastischer Encoder $p(z\mid x)$ erzeugt, und sei $Y$ das Ziellabel. Ist $Z$ stetig, muss der Encoder stochastisch sein, da eine deterministische Abbildung in der Regel $I(X;Z)=\infty$ ergäbe (:numref:`subsec_mdl-mi-gaussian`). Der *Informationsflaschenhals* :cite:`Tishby.Pereira.Bialek.1999` sucht eine Repräsentation $Z$, die möglichst wenig von $X$ und zugleich möglichst viel über $Y$ bewahrt:

<!--en:0b1b220c-->
Dabei bezeichnet $I(X;Z)$ die Rate, also die Information, die die Repräsentation über die Eingabe bewahrt. Die über das Label bewahrte Information $I(Y;Z)$ erhält das Gewicht $\beta>0$. Da der Encoder nur $X$ sieht, bilden die Variablen eine Markov-Kette $Y\to X\to Z$. Die Datenverarbeitungsungleichung liefert die Schranke $I(Y; Z) \leq \min\{I(X; Y),\, I(X; Z)\}$. Eine Repräsentation kann nie mehr über das Label wissen als die Eingabe. Ebenso wenig kann sie gut vorhersagen, ohne Information über die Eingabe zu bewahren.

<!--en:634e091e-->
Der Multiplikator $\beta$ steuert diese Abwägung. Seine beiden Extremfälle sind leicht zu verstehen. Für $\beta\le1$ ist es optimal, nichts zu bewahren. Nach der obigen Ungleichung gilt $I(X;Z)-\beta I(Y;Z)\ge(1-\beta)\,I(X;Z)\ge0$, und ein konstanter Encoder, der $X$ vollständig ignoriert, erreicht $I(X;Z)=I(Y;Z)=0$. Ist der konstante Encoder optimal, sagen wir, der Flaschenhals *kollabiert*. Für große Werte von $\beta$ dominiert dagegen der Labelterm. Der Encoder bewahrt dann alles, was zur Vorhersage von $Y$ beiträgt. In diesem Grenzfall nähert sich $I(Y;Z)$ seiner oberen Schranke $I(X;Y)$ und die Repräsentation einer *minimal suffizienten Statistik*. Das ist die gröbste Zusammenfassung von $X$, mit der sich $Y$ noch genauso gut vorhersagen lässt wie mit $X$ selbst. Ist der Encoder wie jedes neuronale Netz auf eine parametrische Familie beschränkt, erreicht er diese Schranke möglicherweise nicht.

<!--en:b183f06e-->
**Satz (der Informationsflaschenhals als Rate-Verzerrungs-Problem).** *Ist $Y\to X\to Z$ eine Markov-Kette, so gilt*

<!--en:7ed4580c-->
**Beweis.** Wir zerlegen $I(Y;X,Z)$ mit der Kettenregel in beiden Reihenfolgen und erhalten $I(Y;Z)+I(Y;X\mid Z)=I(Y;X)+I(Y;Z\mid X)$. Aus der Markov-Eigenschaft folgt $I(Y;Z\mid X)=0$. Schreiben wir die verbleibende bedingte gegenseitige Information in KL-Form, so ergibt sich $I(Y;X\mid Z)=\mathbb{E}_{(x,z)}\big[D_{\textrm{KL}}(p(y\mid x,z)\,\|\,p(y\mid z))\big]$. Wegen der Markov-Eigenschaft vereinfacht sich $p(y\mid x,z)$ zu $p(y\mid x)$. $\blacksquare$

<!--en:0870d5ca-->
Setzen wir :eqref:`eq_mdl-ib-as-rd` in die Zielfunktion ein, wird aus :eqref:`eq_mdl-ib_lagrangian` der Ausdruck $I(X;Z)+\beta\,\mathbb{E}[d(X,Z)]-\beta I(X;Y)$ mit Verzerrung $d(x,z)=D_{\textrm{KL}}(p(y\mid x)\,\|\,p(y\mid z))$. Der letzte Term $\beta I(X;Y)$ hängt nicht vom Encoder ab. Damit ist der Informationsflaschenhals genau die Lagrange-Form eines Rate-Verzerrungs-Problems. Das Verzerrungsmaß muss niemand von Hand entwerfen, denn das Label gibt es vor. Ein Code $z$ stellt $x$ originalgetreu dar, wenn er $Y$ genauso genau vorhersagt wie $x$. Anders als beim gewöhnlichen Rate-Verzerrungs-Problem hängt diese Verzerrung jedoch über $p(y\mid z)$ vom Encoder selbst ab. Das erschwert die Optimierung erheblich.

<!--en:208ec15c-->
### Der gaußsche Informationsflaschenhals

<!--en:246ec7e3-->
In einem Spezialfall lassen sich alle Größen des Informationsflaschenhalses in geschlossener Form berechnen :cite:`Chechik.Globerson.Tishby.ea.2005`. Wählen wir für $(X, Y)$ unser gaußsches Paar mit $\rho = 0.9$, für das $I(X; Y) = 0.830$ Nat gilt, und verwenden wir den Encoder mit additivem Rauschen aus :numref:`subsec_mdl-rate-distortion`:

<!--en:65f444da-->
Sein einziger Parameter ist die Rauschstärke $\sigma$. Die Rate $I(X;Z)$ kennen wir bereits aus :eqref:`eq_mdl-gaussian-channel`. Für das Label gelten $\textrm{Cov}(Y,Z)=\textrm{Cov}(Y,X)=\rho$ und $\textrm{Var}(Z)=1+\sigma^2$, also $\rho_{YZ}^2=\rho^2/(1+\sigma^2)$. Mit :eqref:`eq_mdl-gaussian_mi` erhalten wir beide gegenseitigen Informationen:

<!--en:0e917b0d-->
Sinkt $\sigma$ von $\infty$ auf $0$, so geht die Repräsentation stetig von einem Extrem ins andere über. Anfangs weiß sie nichts über $X$, am Ende kennt sie $X$ exakt. Ihre Information über das Label steigt von $0$ bis zur Schranke der Datenverarbeitungsungleichung von $0.830$ Nat.

<!--en:0e17b955-->
Um diese Abwägung direkt darzustellen, schreiben wir $u=I(X;Z)$ für die Rate ($R$ behalten wir der Rate-Verzerrungs-Funktion $R(D)$ vor). Durch Umkehren von :eqref:`eq_mdl-gaussian-ib-info` erhalten wir $\sigma^{-2}=e^{2u}-1$ und damit $1/(1+\sigma^2)=1-e^{-2u}$. So können wir die Information über das Label direkt als Funktion der Rate ausdrücken:

<!--en:dec7cfba-->
Die Ableitung fällt monoton von $\rho^2$ bei $u=0$ gegen $0$. Die Kurve ist also streng konkav. Im Ursprung hat sie die Steigung $\rho^2=0.81$. Das erste Nat an Eingabeinformation liefert höchstens $0.81$ Nat an Information über das Label. Jedes weitere Nat trägt weniger bei, während sich $I(Y;Z)$ dem Wert $I(X;Y)$ nähert. Wie :citet:`Chechik.Globerson.Tishby.ea.2005` bewiesen haben, kann für ein gaußsches Paar kein linearer oder nichtlinearer Encoder diese Kurve übertreffen. Sie ist damit die exakte optimale Grenzkurve. Wir verwenden dieses Ergebnis ohne Beweis. Für jedes gewählte $\beta$ legt der Minimierer der Lagrange-Funktion einen einzelnen *Arbeitspunkt* (engl. operating point) auf der Grenzkurve fest.

<!--en:4390507e-->
**Satz (Arbeitspunkt und Kollaps).** *Beim gaußschen Informationsflaschenhals mit $\rho^2<1$ ist die Lagrange-Funktion $u-\beta\,I(Y;Z)$ konvex in $u$. Ihre Ableitung beträgt $1-\beta\rho^2$ bei $u=0$. Für $\beta\le1/\rho^2$ liegt das Minimum bei $u=0$, dem konstanten Encoder. Für $\beta>1/\rho^2$ lauten der Minimierer und die zugehörige Information über das Label*

<!--en:bb9b312d-->
**Beweis.** Da $I(Y;Z)$ konkav in $u$ ist, ist die Lagrange-Funktion $u-\beta I(Y;Z)$ konvex mit Ableitung $1-\beta\rho^2/(\rho^2+(1-\rho^2)e^{2u})$. Bei $u=0$ beträgt diese Ableitung $1-\beta\rho^2$. Für $\beta\rho^2\le1$ ist sie bei $u=0$ nichtnegativ und wächst streng monoton. Daher liegt das Minimum über $u\ge0$ bei $u=0$. Andernfalls setzen wir die Ableitung gleich null und erhalten $\rho^2+(1-\rho^2)e^{2u}=\beta\rho^2$ oder $e^{2u}=\rho^2(\beta-1)/(1-\rho^2)$. Einsetzen von $\rho^2e^{-2u}=(1-\rho^2)/(\beta-1)$ in $I(Y;Z)=-\tfrac12\log(1-\rho^2+\rho^2e^{-2u})$ liefert den zweiten Ausdruck. $\blacksquare$

<!--en:1ba7e94f-->
Die allgemeine Argumentation mit der Datenverarbeitungsungleichung garantierte einen Kollaps für $\beta\le1$. Für das gaußsche Paar mit $\rho=0.9$ reicht der Kollaps sogar bis $\beta\le1/\rho^2=1.235$. Unterhalb dieser Schwelle erhöht jedes Nat an Rate die Zielfunktion um $1$ und senkt sie zugleich um höchstens $\beta\rho^2\le1$. Denn die Steigung der Grenzkurve übertrifft nie ihren Anfangswert $\rho^2$. Oberhalb der Schwelle verschiebt ein wachsendes $\beta$ den Arbeitspunkt auf der Grenzkurve in Richtung der oberen Schranke. Für $\beta=2$ erhalten wir eine Rate von $u^*=\tfrac12\ln(0.81/0.19)=0.725$ Nat und $0.484$ Nat an Information über das Label, für $\beta=16$ erreichen wir $2.079$ und $0.798$ Nat. :numref:`fig_mdl-ib-tradeoff` zeigt die Grenzkurve mit ihren Arbeitspunkten in der *Informationsebene*. Auf der horizontalen Achse steht die Rate $I(X;Z)$, auf der vertikalen die Information über das Label $I(Y;Z)$.

<!--en:1dd2c63d-->
![Informationsebene des gaußschen Informationsflaschenhalses mit $\rho = 0.9$. Die konkave Grenzkurve gibt den größten erreichbaren Wert von $I(Y;Z)$ bei jeder Rate $I(X;Z)$ an. Der schattierte Bereich darüber ist unerreichbar. Die gestrichelte Linie markiert die obere Schranke der Datenverarbeitungsungleichung bei $I(X;Y) = -\tfrac{1}{2}\log(1-\rho^2) \approx 0.830$ Nat. Die Grenzkurve beginnt im Ursprung mit Steigung $\rho^2$. Der Multiplikator wählt den Punkt mit Steigung $1/\beta$. Für $\beta\le1/\rho^2\approx1.23$ fällt dieser Punkt in den Ursprung.](../img/mdl-it-ib-tradeoff.svg)

<!--en:f205f04a-->
Einen so scharfen Phasenübergang überprüfen wir numerisch. Nach dem Satz bewahrt der optimale Encoder bei $\beta=1.2$ nichts, bei $\beta=1.25$ dagegen bereits Information. Im Folgenden minimieren wir die Lagrange-Funktion direkt auf einem Gitter von $4{,}001$ Rauschstärken, ohne die geschlossene Lösung zu verwenden.

<!--en:2d026be7-->
Gittersuche und geschlossene Lösung stimmen bis auf ein Tausendstel Nat überein. Die Kollapsschwelle liegt genau dort, wo der Satz sie vorhersagt. Bei $\beta=1.2$ ist der konstante Encoder optimal, bei $\beta=1.25$ bewahrt der Encoder bereits $0.032$ Nat. Genau dieser Encoder mit Rauschstärke $\sigma$ erreicht für quadratischen Fehler auch die Rate-Verzerrungs-Kurve $R(D)$ bei $D=\sigma^2/(1+\sigma^2)$. Die Rate-Verzerrungs-Theorie bewertet den Encoder danach, wie gut er $X$ rekonstruiert. Der Informationsflaschenhals bewertet ihn danach, wie gut er $Y$ vorhersagt.

<!--en:d8c7a5d0-->
### Der variationelle Informationsflaschenhals

<!--en:0bfdcea0-->
Für ein gaußsches Paar konnten wir beide gegenseitigen Informationen in der Zielfunktion analytisch berechnen. Bei realen Daten kennen wir die wahre gemeinsame Verteilung nicht und können daher keine der beiden Größen berechnen. Wir beobachten nur Beobachtungspaare $(x,y)$ und wählen dazu einen Encoder $p(z\mid x)$. Der *variationelle Informationsflaschenhals* (engl. variational information bottleneck, VIB) :cite:`Alemi.Fischer.Dillon.ea.2017` ermöglicht das Training, indem er beide gegenseitigen Informationen durch handhabbare Schranken ersetzt, die sich aus Stichproben schätzen lassen.

<!--en:c3497264-->
Für den Labelterm kennen wir die benötigte Schranke bereits. Wir führen einen Decoder $q(y\mid z)$ mit Kreuzentropieverlust $L_Z=\mathbb{E}[-\log q(Y\mid Z)]$ ein, also einen Klassifikator, der das Label aus der Repräsentation vorhersagt. Der Informationsnachweis durch den Verlust :eqref:`eq_mdl-info-certificate` liefert für das Paar $(Z,Y)$ unmittelbar $I(Y;Z)\ge H(Y)-L_Z$.

<!--en:48219d90-->
Da wir die Rate $I(X;Z)$ *minimieren* wollen, brauchen wir für sie eine obere Schranke. Schreiben wir die Rate in der bedingten Form :eqref:`eq_mdl-mi-conditional-form` als $I(X;Z)=\mathbb{E}_x\big[D_{\textrm{KL}}(p(z\mid x)\,\|\,p(z))\big]$. Der Encoder $p(z\mid x)$ ist bekannt, die aggregierte Randverteilung $p(z)$ dagegen nicht, denn sie mittelt $p(z\mid x)$ über die unbekannte Datenverteilung. Wir ersetzen $p(z)$ durch eine feste A-priori-Dichte $r(z)$ unserer Wahl, etwa die Standardnormaldichte. Mit der Identität $\log\frac{p(z\mid x)}{r(z)}=\log\frac{p(z\mid x)}{p(z)}+\log\frac{p(z)}{r(z)}$ und den Erwartungswerten über $x$ und $z$ erhalten wir

<!--en:3b2c638d-->
Jede Abweichung unserer gewählten A-priori-Verteilung $r$ von der wahren Randverteilung $p(z)$ kann die Ratenschranke nur erhöhen. Aufgabe :exref:`ex_mdl-mutual-information-vae-rate` berechnet die Differenz für einen gaußschen Encoder explizit. Zusammen zeigen beide Schranken, dass der KL-Term auf der linken Seite von :eqref:`eq_mdl-vib-rate-bound` plus $\beta L_Z$ die Zielfunktion :eqref:`eq_mdl-ib_lagrangian` bis auf die Konstante $\beta H(Y)$ nach oben beschränkt. Beide Terme sind Erwartungswerte über Trainingsbeispiele. Ein neuronales Netz kann diese Schranke daher direkt durch stochastischen Gradientenabstieg minimieren.

<!--en:17f259a8-->
Wer schon einen Variationsautoencoder trainiert hat, erkennt die linke Seite von :eqref:`eq_mdl-vib-rate-bound` sofort wieder. Ein Variationsautoencoder (:numref:`sec_diffusion-latent`) verwendet genau diese Konstruktion mit der Eingabe als ihrem eigenen Label. Er besteht aus einem Encoder $p(z\mid x)$, einer festen A-priori-Verteilung $r(z)$ und einem Decoder, der $x$ aus $z$ rekonstruiert, anstatt ein separates Klassenlabel vorherzusagen. Sein Trainingsverlust ist die Summe aus Rekonstruktionsfehler und gewichtetem KL-Strafterm. Aus unserer Sicht ist dieser KL-Strafterm eine obere Schranke für die Rate. Er begrenzt, wie viele Nat der latente Code über die Eingabe speichern kann. Der Rekonstruktionsfehler übernimmt die Rolle der Verzerrung. Der Unterschied liegt allein in der Gewichtung. In :eqref:`eq_mdl-ib_lagrangian` wird die bewahrte Information mit $\beta$ multipliziert. Ein Variationsautoencoder gewichtet dagegen den KL-Term für die Rate. Ein großes KL-Gewicht dort entspricht also einem kleinen $\beta$ hier.

<!--en:951b1d19-->
Der Kreuzentropieverlust des Decoders und die KL-Divergenz zu einer A-priori-Verteilung sind die einzigen Größen dieses Abschnitts, die sich allein anhand von Stichproben auswerten lassen. Sie liefern eine untere Schranke für die eine gegenseitige Information und eine obere für die andere. Für alle übrigen Zahlen benötigten wir die gemeinsame Verteilung, die wir bei realen Bildern oder Texten nie kennen. In :numref:`sec_mdl-mi-estimation` fragen wir, inwieweit sich die gegenseitige Information allein aus Stichproben schätzen lässt.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:4fd3ebe5-->
Die gegenseitige Information ist die KL-Divergenz zwischen einer gemeinsamen Verteilung und dem Produkt ihrer Randverteilungen, $I(X;Y)=D_{\textrm{KL}}(P_{X,Y}\,\|\,P_X\otimes P_Y)$. Sie verschwindet genau dann, wenn $X$ und $Y$ unabhängig sind, und bleibt unter invertierbaren Reparametrisierungen jeder der beiden Variablen invariant. Anders als die lineare Korrelation erfasst sie nichtlineare Abhängigkeiten. Sie weist $1.14$ Nat für $X$ und $Y=X^2+0.3\epsilon$ aus, obwohl deren Korrelation null ist. Nach der Datenverarbeitungsungleichung kann schrittweise Verarbeitung keine Information erzeugen. Schichten neuronaler Netze können vorhandene Information jedoch so umordnen, dass ein Klassifikator sie leichter auslesen kann. Die Fano-Ungleichung liefert für jede angestrebte Fehlerrate eine Mindestanforderung an die gegenseitige Information. Der Kreuzentropieverlust $L$ jedes probabilistischen Modells weist $I(X;Y)\ge H(Y)-L$ nach. Shannons Kanalkapazität ergibt sich als Maximum der gegenseitigen Information über die Eingabeverteilungen, seine Rate-Verzerrungs-Funktion als Minimum über die verlustbehafteten Encoder. Der Informationsflaschenhals $\min I(X;Z)-\beta I(Y;Z)$ formuliert Repräsentationslernen als Rate-Verzerrungs-Problem, bei dem das Label die Verzerrung vorgibt. Für ein gaußsches Paar kollabiert er, solange $\beta$ den Wert $1/\rho^2$ nicht übersteigt. In der variationellen Form wird der Strafterm für die Rate zum bekannten KL-Regularisierungsterm eines Variationsautoencoders.

<!--en:2294243a-->
## Übungen

<!--en:b9514c15-->
1. [code] **Seltene Paare und PMI.** Die punktweise gegenseitige Information :eqref:`eq_mdl-pmi_def` korrigiert die Häufigkeit der Kookkurrenz um die Häufigkeiten der beiden Wörter (:numref:`subsec_mdl-pmi`).
    1. Zeigen Sie, dass $\textrm{pmi}(x, y) \leq \min\{-\log p_X(x), -\log p_Y(y)\}$ gilt. Wann wird die jeweilige Schranke erreicht?
    1. Erweitern Sie `counts` um ein viertes erstes Wort *sri* und ein viertes zweites Wort *lanka*, die nur im Bigramm „sri lanka“ mit Häufigkeit $2$ vorkommen. Berechnen Sie die PMI von „new york“, „machine learning“, „the day“ und „sri lanka“ neu und wiederholen Sie die Rechnung mit Häufigkeit $1$. Bei Häufigkeit null gilt $\textrm{pmi} = -\infty$, sodass `(p * pmi).sum()` den Wert `nan` zurückgibt. Beschränken Sie die Summen daher wie in `mutual_information` auf Zellen mit positiver Häufigkeit.
    1. Vergleichen Sie die Beiträge $p_{X,Y}(x, y)\,\textrm{pmi}(x, y)$ von „new york“ und „sri lanka“ zu $I(X;Y)$. Addieren Sie anschließend $1$ zu jeder Häufigkeit und berechnen Sie beide PMI-Werte neu. Welche dieser Größen würden Sie für das Ranking von Kollokationen verwenden und warum?
   :exlabel:`ex_mdl-mutual-information-rare-pairs`

<!--en:967370ca-->
1. **Information und nutzbare Information.** Das Label $Y$ sei ein gleichverteiltes Bit. Für $Y = 0$ sei die Eingabe $X \in \mathbb{R}^2$ gleichverteilt auf dem Kreis mit Radius $1$ und für $Y = 1$ auf dem Kreis mit Radius $2$. Eine Repräsentation berechnet $Z = \|X\|^2$.
    1. Zeigen Sie, dass $I(Y;Z) = I(Y;X) = \ln 2$ gilt, obwohl $x \mapsto \|x\|^2$ nicht invertierbar ist. Welche Bedingung für Gleichheit in der Datenverarbeitungsungleichung aus :numref:`subsec_mdl-dpi` ist erfüllt?
    1. Zeigen Sie, dass kein linearer Klassifikator $\hat y = \mathbf{1}(\mathbf{w}^\top \mathbf{x} + b > 0)$ jede Eingabe korrekt klassifiziert, ein Schwellenwert für $Z$ dagegen schon.
    1. Die Schichten eines Netzes bilden eine Markov-Kette $X \to Z_1 \to Z_2 \to \cdots$. Zeigen Sie, dass der Fehler des bestmöglichen Klassifikators beliebiger Bauart mit Eingabe $Z_\ell$ für wachsendes $\ell$ nicht sinken kann. Die Genauigkeit eines an $Z_\ell$ angepassten linearen Klassifikators steigt dennoch häufig mit $\ell$. Warum ist das kein Widerspruch?
    1. :citet:`Xu.Zhao.Song.ea.2020` messen die Information, die eine Familie $\mathcal{V}$ von Prädiktoren $q(y \mid x)$ gewinnen kann. Sei $H_{\mathcal{V}}(Y \mid X) = \inf_{q \in \mathcal{V}} \mathbb{E}[-\log q(Y \mid X)]$, und sei $H_{\mathcal{V}}(Y)$ dasselbe Infimum über diejenigen Prädiktoren in $\mathcal{V}$, die ihre Eingabe ignorieren. Die *nutzbare Information* ist

       $$I_{\mathcal{V}}(X \to Y) = H_{\mathcal{V}}(Y) - H_{\mathcal{V}}(Y \mid X).$$

       Sei $\mathcal{V}$ die Familie der logistischen Regressionen auf der gegebenen Eingabe. Zeigen Sie, dass $I_{\mathcal{V}}(X \to Y) = 0$ und $I_{\mathcal{V}}(Z \to Y) = \ln 2$ gelten. Nutzen Sie für $X$ aus, dass der erwartete logistische Verlust konvex in $(\mathbf{w}, b)$ ist und sich bei Drehung von $\mathbf{w}$ nicht ändert. Welche Ungleichung aus :numref:`subsec_mdl-dpi` verletzt $I_{\mathcal{V}}$?
   :exlabel:`ex_mdl-mutual-information-usable-information`

<!--en:d6f095fe-->
1. **Synergie und Venn-Diagramm.** Seien $X$ und $Y$ unabhängige gleichverteilte Bits und $Z = X \oplus Y$ ihr exklusives Oder. Dafür haben wir im Abschnitt $I(X;Y) = 0$ und $I(X;Y \mid Z) = \ln 2$ berechnet.
    1. Welche der Markov-Ketten $X \to Z \to Y$, $X \to Y \to Z$ und $Y \to X \to Z$ liegen vor? Was sagt die Datenverarbeitungsungleichung demnach über dieses Tripel aus?
    1. In einer Variante von :numref:`fig_mdl-mi-overlap` für drei Variablen entspräche die allen drei Kreisscheiben gemeinsame Fläche der *Interaktionsinformation* $I(X;Y;Z) = I(X;Y) - I(X;Y \mid Z)$. Beweisen Sie $I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)$ und folgern Sie, dass $I(X;Y;Z)$ symmetrisch in seinen drei Argumenten ist.
    1. Berechnen Sie $I(X;Y;Z)$ für das Tripel aus (a) und für drei Kopien $X = Y = Z$ eines gleichverteilten Bits. Was bedeuten die beiden Werte für ein Entropiediagramm mit drei Kreisscheiben?

   *Nach Thomas M. Cover und Joy A. Thomas, [Elements of Information Theory, 2. Auflage](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959), Aufgabe 2.25, und David J. C. MacKay, [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/itprnn/book.pdf), Aufgabe 8.8.*
   :exlabel:`ex_mdl-mutual-information-synergy-conditioning-can-create`

<!--en:da4042cd-->
1. [code] **Datenverarbeitung mit $f$-Divergenzen.** Für jeden Kanal $\kappa$, der eine Eingabe auf eine zufällige Ausgabe abbildet, und je zwei Eingabeverteilungen $P$ und $Q$ erfüllen die Ausgabeverteilungen die Ungleichung $D_f(P\kappa\,\|\,Q\kappa)\le D_f(P\,\|\,Q)$ für jede $f$-Divergenz (:numref:`subsec_mdl-f-div-data-processing`). Zwei Beispiele sind die KL-Divergenz und $\chi^2(P\|Q)=\sum_x p(x)^2/q(x)-1$.
    1. Sei $X\to Y\to Z$ eine Markov-Kette. Wählen Sie $P=P_{X,Y}$, $Q=P_X\otimes P_Y$ und den Kanal $\kappa$, der $(x,y)$ auf $(x,z)$ mit $z\sim p(z\mid y)$ abbildet. Zeigen Sie, dass $P\kappa=P_{X,Z}$ und $Q\kappa=P_X\otimes P_Z$ gelten, und folgern Sie $I(X;Z)\le I(X;Y)$.
    1. Welchen Teil des Satzes zur Datenverarbeitungsungleichung liefert diese Herleitung nicht?
    1. Betrachten Sie die beiden Kippvorgänge in Folge mit Wahrscheinlichkeiten $0.1$ und $0.2$. Berechnen Sie die KL-Divergenz und die $\chi^2$-Divergenz jeder gemeinsamen Verteilung vom Produkt ihrer Randverteilungen vor und nach dem zweiten Kippvorgang.
    1. Zeigen Sie für die gleichverteilte Eingabe eines Kanals mit Kippwahrscheinlichkeit $\delta$, dass $\chi^2(P_{X,Y}\,\|\,P_X\otimes P_Y)=(1-2\delta)^2$ gilt. Überprüfen Sie Ihre Werte aus (c). Folgern Sie, dass ein weiteres Kippen mit Wahrscheinlichkeit $\delta'$ diese Divergenz mit $(1-2\delta')^2$ multipliziert, hier also mit $0.36$. Ist das Verhältnis $I(X;Z)/I(X;Y)$ aus (c) gleich diesem Faktor?
   :exlabel:`ex_mdl-mutual-information-dpi-from-f-divergence`

<!--en:6c786c14-->
1. **Fano-Ungleichung und eine umgekehrte Abschätzung.** ● Die Fano-Ungleichung :eqref:`eq_mdl-fano` beschränkt den Fehler jeder Schätzung $\hat Y = g(X)$ eines Labels $Y$ mit $k$ Werten nach unten.
    1. Sei $\hat Y(x) = \arg\max_y p(y \mid x)$ das wahrscheinlichste Label. Zeigen Sie, dass für seinen Vorhersagefehler $P_e \leq 1 - e^{-H(Y \mid X)}$ gilt. Schätzen Sie $\max_y p(y \mid x)$ nach unten durch $\sum_y p(y \mid x)^2=\mathbb{E}[p(Y\mid X)\mid X=x]$ ab und wenden Sie dann einmal die Jensen-Ungleichung an, und zwar auf $\mathbb{E}[p(Y\mid X)]=\mathbb{E}\big[e^{\log p(Y\mid X)}\big]$.
    1. Eine Beobachtung $X$ enthält $I(X;Y) = 9$ Nat über ein unter $10{,}000$ Klassen gleichverteiltes Label $Y$. Grenzen Sie mit :eqref:`eq_mdl-fano` und (a) den Fehler des besten Klassifikators ein. Die Ungleichung :eqref:`eq_mdl-fano` hat keine geschlossene Lösung für $P_e$. Lösen Sie sie daher numerisch. Was sagt die Breite dieses Intervalls darüber aus, wie gut sich aus gegenseitiger Information eine nachgelagerte Fehlerrate vorhersagen lässt?
   :exlabel:`ex_mdl-mutual-information-fano-converse`

<!--en:e39471f2-->
1. **Informationsnachweis durch Verlust und Fano-Ungleichung.** Ein Klassifikator für $k=10$ gleich wahrscheinliche Klassen hat den Testfehler $P_e=0.08$ und den Testverlust $L=0.30$ Nat.
    1. Wie viele Nat von $I(X;Y)$ weist die Fano-Ungleichung :eqref:`eq_mdl-fano` nach, wie viele der Informationsnachweis durch den Verlust :eqref:`eq_mdl-info-certificate`?
    1. Betrachten Sie eine beliebige Schätzung $\hat Y=g(X)$ mit Fehlerwahrscheinlichkeit $P_e$. Das Modell $q(y\mid x)$ weise $1-P_e$ als Wahrscheinlichkeit für $y=g(x)$ zu und verteile $P_e$ gleichmäßig auf die übrigen $k-1$ Labels. Zeigen Sie, dass sein Kreuzentropieverlust $H_b(P_e)+P_e\log(k-1)$ beträgt. Folgern Sie, dass die Fano-Ungleichung der Informationsnachweis durch den Verlust dieses Modells ist. Welchem der vier in :numref:`subsec_mdl-loss-certificate` ausgewerteten Modelle entspricht es?
    1. Der Klassifikator aus (a) liefert einen kleineren Verlust als das Modell, das (b) aus seinen Entscheidungen konstruiert. Wie müssen sich seine Konfidenzen verhalten, damit dies möglich ist? Geben Sie ein Paar $(X,Y)$ mit binärem Label und ein Modell mit Fehlerrate $P_e$ an, das mehr Information nachweist als die Fano-Ungleichung.
   :exlabel:`ex_mdl-mutual-information-certificate-and-fano`

<!--en:814ffd62-->
1. **Binäre Kanäle.** ● Rechnen Sie in Nat mit der binären Entropie $H_b$.
    1. Ein *Z-Kanal* überträgt eine $0$ unverändert und wandelt eine $1$ in eine $0$ um, und zwar mit Wahrscheinlichkeit $\tfrac12$. Berechnen Sie $I(X;Y)=H(Y)-H(Y\mid X)$ als Funktion von $a=P(X=1)$. Zeigen Sie, dass das Maximum bei $a=\tfrac25$ liegt und $\ln\tfrac54$ Nat beträgt. Warum sendet die optimale Eingabe das unzuverlässige Symbol in weniger als der Hälfte der Fälle, aber auch nicht nie?
    1. Ein Merkmal nimmt die Werte $\pm 1$ an und durchläuft Dropout mit Rate $\alpha$. Dropout ersetzt das Merkmal durch $0$ mit Wahrscheinlichkeit $\alpha$ und skaliert es andernfalls um. Zeigen Sie, dass dieser *binäre Auslöschungskanal* die Kapazität $(1 - \alpha)\ln2$ Nat hat.
    1. Bestimmen Sie die Kippwahrscheinlichkeit $\delta$, bei der der binäre symmetrische Kanal dieselbe Kapazität hat wie Dropout mit $\alpha = 0.5$. Warum kostet ein gekipptes Bit mehr Kapazität als ein gelöschtes?
   :exlabel:`ex_mdl-information-theory-binary-channels`

<!--en:2eb79670-->
1. **Der gaußsche Kanal.** Sei $Y=X+N$ mit Rauschen $N\sim\mathcal N(0,\sigma_N^2)$ unabhängig von $X$. Für die Eingabe gelte die Leistungsbeschränkung $\mathbb{E}[X^2]\le P$.
    1. Zeigen Sie, dass $I(X;Y)=h(Y)-h(N)$ für jede Eingabe mit einer Dichte gilt.
    1. Unter allen Dichten mit Varianz $s^2$ hat die Normalverteilung die größte differentielle Entropie $\tfrac12\log(2\pi e s^2)$. Zeigen Sie damit $h(Y)\le\tfrac12\log\big(2\pi e(P+\sigma_N^2)\big)$ und folgern Sie, dass die Kapazität $C=\tfrac12\log(1+P/\sigma_N^2)$ beträgt und von einer gaußschen Eingabe erreicht wird.
    1. Berechnen Sie $C$ für die Signal-Rausch-Verhältnisse $P/\sigma_N^2=1,10,100$ in Nat und in Bit. Zeigen Sie, dass bei hohem Signal-Rausch-Verhältnis jede Verdopplung von $P$ etwa $\tfrac12\ln2$ Nat hinzufügt, also ein halbes Bit.
    1. Ein Diffusionsmodell ersetzt eine Koordinate $X$ der Daten, die so skaliert ist, dass $\mathbb{E}[X^2]=1$ gilt, durch $\sqrt{a}\,X+\sqrt{1-a}\,\epsilon$ mit $0<a<1$ und unabhängigem $\epsilon\sim\mathcal N(0,1)$. Zeigen Sie, dass die verrauschte Koordinate höchstens $\tfrac12\log\frac{1}{1-a}$ Nat über $X$ enthält, unabhängig von der Verteilung von $X$. Werten Sie die Schranke bei $a=0.5$ und $a=0.01$ aus. Für welche Daten wird sie erreicht?
   :exlabel:`ex_mdl-mutual-information-gaussian-channel`

<!--en:5157012c-->
1. **Die Ratenschranke eines Variationsautoencoders.** Sei $X\sim\mathcal N(0,1)$. Der Encoder sei durch $Z=X+\sigma\epsilon$ gegeben (:numref:`subsec_mdl-gaussian-bottleneck`), sodass $p(z\mid x)=\mathcal N(z;x,\sigma^2)$ gilt. Die A-priori-Verteilung sei $r=\mathcal N(0,1)$. Verwenden Sie $D_{\textrm{KL}}\big(\mathcal N(\mu,s^2)\,\|\,\mathcal N(0,1)\big)=\tfrac12(\mu^2+s^2-1-\log s^2)$.
    1. Berechnen Sie die drei Terme in :eqref:`eq_mdl-vib-rate-bound` und überprüfen Sie die Identität. Um wie viele Nat überschätzt der KL-Strafterm die Rate bei $\sigma=1$?
    1. Skalieren Sie den Code zu $Z'=Z/\sqrt{1+\sigma^2}$ um. Zeigen Sie, dass $I(X;Z')=I(X;Z)$ gilt und die Schranke nun mit Gleichheit erfüllt ist.
    1. Die Rate ist unabhängig von den Einheiten des Codes, die Schranke hängt davon ab. Für welche A-priori-Verteilung $r$ wird bei gegebenem Encoder die Schranke mit Gleichheit erfüllt? Was bewirkt die Minimierung des KL-Strafterms also außer der Senkung der Rate für den Code?
   :exlabel:`ex_mdl-mutual-information-vae-rate`

<!--en:d2e9212f-->
[Diskussionen](https://d2l.discourse.group/t/36836)

<!--en:95fa049a-->
Gegenseitige Information und Informationsflaschenhals<br>**Abhängigkeit · PMI · Datenverarbeitung · Fano-Ungleichung · Kanalkapazität · Rate und Verzerrung · IB**

<!--en:7beee3f5-->
::: {.slide title="Warum Korrelation nichtlineare Abhängigkeit übersehen kann"}

<!--en:acee9d16-->
[Keine Korrelation trotz hoher gegenseitiger Information]{.kicker}

<!--en:c2c68ba5-->
Die lineare Korrelation $\rho(X, Y) = \frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$ erfasst nur geradlinige Zusammenhänge. Betrachten wir $X \sim \mathcal{N}(0, 1)$ und die verrauschte Parabel $Y = X^2 + 0.3\epsilon$ mit $\epsilon \sim \mathcal{N}(0, 1)$:

<!--en:473fc21f-->
- Da $X^3$ symmetrisch um null verteilt ist, gilt $\textrm{Cov}(X, Y) = \mathbb{E}[X^3] = 0$ $\implies$ $\rho(X, Y) = 0$. Dennoch bestimmt $X$ den Wert von $Y$ bis auf Rauschen mit Standardabweichung $0.3$. Beide teilen **$1.14\text{ nats}$**, mehr als das gaußsche Paar mit $\rho = 0.9$ ($0.83\text{ nats}$).

<!--en:d33e01db-->
::: {.slide title="Gemeinsame und bedingte Entropie beim gekippten Bit"}

<!--en:7b67a4e4-->
[Durchgängiges Beispiel: binärer symmetrischer Kanal]{.kicker}

<!--en:cd2d76cd-->
Sei $X \in \{0, 1\}$ ein gleichverteiltes Bit ($H(X) = \ln 2 \approx 0.693\text{ nats}$), das mit Wahrscheinlichkeit $\delta = 0.1$ kippt und so $Y$ ergibt:

<!--en:502e9476-->
- **Bedingte und gemeinsame Entropie:** Jede Zeile hat die Entropie $H(Y \mid X = x) = H_b(0.1) = 0.325\text{ nats}$. Somit gelten $H(X, Y) = 0.693 + 0.325 = 1.018\text{ nats}$ und $I(X; Y) = 0.693 - 0.325 = 0.368\text{ nats}$.

<!--en:acb783c7-->
::: {.slide title="Gegenseitige Information als KL-Divergenz zur Unabhängigkeit"}

<!--en:55623f5d-->
[Definition]{.kicker}

<!--en:3c47a1ee-->
Wären $X$ und $Y$ unabhängig, wäre ihre gemeinsame Verteilung das Produkt ihrer Randverteilungen $P_X \otimes P_Y$. Die **gegenseitige Information** ist die KL-Divergenz der wahren gemeinsamen Verteilung von dieser Produktverteilung:

<!--en:0275e927-->
- **Berechnung beim gekippten Bit ($\delta = 0.1$):** In der Produktverteilung steht $0.25$ in jeder Zelle. Daher beträgt $p_{X,Y}/(p_X p_Y)$ auf der Diagonalen der Übereinstimmungen $0.45/0.25 = 1.8$ (Gewicht $0.9$) und außerhalb $0.05/0.25 = 0.2$ (Gewicht $0.1$):
  $$I(X; Y) = 0.9\ln 1.8 + 0.1\ln 0.2 = 0.529 - 0.161 = 0.368\text{ nats} \approx 0.531\text{ bits}$$

<!--en:c24932c4-->
- **Bedingte KL-Divergenz (zeilenweise):** Wegen $\frac{p_{X,Y}(x, y)}{p_X(x)p_Y(y)} = \frac{p(y \mid x)}{p_Y(y)}$ misst $I(X; Y) = \mathbb{E}_{x \sim P_X}\big[D_{\textrm{KL}}(p(y \mid x) \,\|\, p_Y(y))\big]$, wie weit die Beobachtung von $x$ die Verteilung von $Y$ von ihrer A-priori-Randverteilung $p_Y$ verschiebt.

<!--en:1af8dd7c-->
::: {.slide title="Symmetrie, Nichtnegativität und Überlappung der Entropien"}

<!--en:f311954a-->
[Grundlegende Identitäten]{.kicker}

<!--en:874ca09c-->
Wie hängt die gegenseitige Information mit den einzelnen Entropien, der gemeinsamen und der bedingten Entropie zusammen?

<!--en:8b315f4f-->
- **Nichtnegativität und Unabhängigkeit:** Nach der Gibbs-Ungleichung gilt $I(X; Y) = I(Y; X) \ge 0$, und es gilt $I(X; Y) = 0 \iff P_{X,Y} = P_X \otimes P_Y$ ($X$ und $Y$ sind unabhängig).

<!--en:4e8dc7c7-->
- **Drei gleichwertige Entropiezerlegungen:** Ausmultiplizieren von $\log\frac{p(x,y)}{p(x)p(y)}$ ergibt
  $$I(X; Y) = H(X) + H(Y) - H(X, Y)$$
  $$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$$

<!--en:78aafe80-->
- **Abnahme der Unsicherheit:** Beim gekippten Bit verringert die Beobachtung von $X$ die Unsicherheit über $Y$ um $0.693 - 0.325 = 0.368\text{ nats}$.

<!--en:d7544852-->
::: {.slide title="Diskrete gegenseitige Information im Code überprüfen"}

<!--en:da8d42f3-->
[KL-Divergenz und Entropieabnahme]{.kicker}

<!--en:e344bb1e-->
Werten wir die KL-Divergenz und die Abnahme der Entropie durch Konditionieren $H(Y) - H(Y \mid X)$ für die $2 \times 2$-Tabelle des gekippten Bits aus, bestätigen beide $I(X; Y) = 0.3681\text{ nats}$ ($0.531\text{ bits}$):

<!--en:25395426-->
::: {.slide title="Diskrete gegenseitige Information im Code überprüfen (Fortsetzung)"}

<!--en:da8d42f3-2-->
[KL-Divergenz und Entropieabnahme]{.kicker}

<!--en:6c5248e1-->
Mit der gemeinsamen Tabelle und beiden Hilfsfunktionen erhalten wir aus den Randverteilungen $H(Y \mid X)$. Beide Berechnungen von $I(X; Y)$ stimmen überein:

<!--en:29ba9d06-->
::: {.slide title="Gegenseitige Information eines bivariat normalverteilten Paares"}

<!--en:4b8bedf4-->
[Geschlossene Formel für stetige Variablen]{.kicker}

<!--en:689b667d-->
Für ein bivariat normalverteiltes Paar $(X, Y)$ mit Korrelation $\rho \in (-1, 1)$ senkt die Konditionierung auf $X$ die Varianz von $Y$ von $\sigma_Y^2$ auf $\sigma_Y^2(1 - \rho^2)$:

<!--en:57ef379b-->
- **Unerklärte Varianz:** Das Ergebnis hängt nur von $1 - \rho^2$ ab ($0$ Nat für $\rho = 0$, $0.830$ Nat für $\rho = 0.9$, $1.959$ Nat für $\rho = 0.99$ und $\infty$ für $\rho \to \pm 1$).

<!--en:198eacca-->
::: {.slide title="Punktweise gegenseitige Information und Kollokationen"}

<!--en:bbafe2a7-->
[Anziehung und Abstoßung einzelner Paare]{.kicker}

<!--en:1a59ea52-->
Während $I(X; Y)$ über alle Ergebnispaarungen mittelt, ist das logarithmierte Verhältnis für ein einzelnes Paar $(x, y)$ die **punktweise gegenseitige Information**:

<!--en:3affe2db-->
- Die Division durch die Randhäufigkeiten $p_X(x)p_Y(y)$ unterscheidet echte Kollokationen (*new york*, *machine learning*) von Wörtern, die nur deshalb gemeinsam auftreten, weil beide häufig sind (*the day*):

<!--en:7d465a73-->
::: {.slide title="Gegenseitige Information der verrauschten Parabel"}

<!--en:f19f3738-->
[Eindimensionale Quadratur und gaußsche obere Schranke]{.kicker}

<!--en:f5909264-->
Für $X \sim \mathcal{N}(0, 1)$ und $Y = X^2 + 0.3\epsilon$ ($\rho = 0$) bleibt nach Konditionierung auf $X = x$ nur das additive gaußsche Rauschen $0.3\epsilon$. Es folgt $h(Y \mid X) = \tfrac12\ln(2\pi e \cdot 0.09) = 0.215\text{ nats}$. Eindimensionale numerische Quadratur von $p_Y(y)$ ergibt $h(Y) = 1.356\text{ nats}$ und $I(X; Y) = 1.141\text{ nats}$:

<!--en:dd051826-->
::: {.slide title="Invarianz unter bijektiver Reparametrisierung"}

<!--en:d70da6fe-->
[Koordinatenunabhängigkeit]{.kicker}

<!--en:c80c449c-->
Die differentielle Entropie $h(X)$ ändert sich mit den Einheiten, die lineare Korrelation $\rho(X, Y)$ bei nichtlinearer Streckung. Die gegenseitige Information bleibt unter jeder invertierbaren Transformation der Variablen invariant:

<!--en:f3dc5902-->
- **Warum sich die Jacobi-Faktoren kürzen:** Für glatte Bijektionen $\tilde{x} = \phi(x)$ und $\tilde{y} = \psi(y)$ transformiert sich die gemeinsame Dichte gemäß $p_{\tilde{X},\tilde{Y}}(\tilde{x}, \tilde{y}) = p_{X,Y}(x, y)|\det D\phi(x)|^{-1}|\det D\psi(y)|^{-1}$. Das Produkt der Randdichten $p_{\tilde{X}}(\tilde{x})p_{\tilde{Y}}(\tilde{y})$ erhält dieselben beiden Jacobi-Faktoren. Im Verhältnis $\frac{p_{\tilde{X},\tilde{Y}}}{p_{\tilde{X}}p_{\tilde{Y}}}$ kürzen sie sich.

<!--en:94cb1038-->
- **Beispiel für das gaußsche Paar mit $\rho = 0.9$:** Die kubische Bijektion $X \mapsto X^3$ lässt $I(X^3; Y) = I(X; Y) = 0.830\text{ nats}$ unverändert. Die lineare Korrelation sinkt dagegen von $\rho = 0.900$ auf $\frac{\mathbb{E}[X^3 Y]}{\sqrt{\textrm{Var}(X^3)\textrm{Var}(Y)}} = \frac{3\rho}{\sqrt{15}} = 0.697$.

<!--en:8b20a5b6-->
::: {.slide title="Bedingte gegenseitige Information und Kettenregel"}

<!--en:000b53ac-->
[Zerlegungen für drei Variablen]{.kicker}

<!--en:606a4192-->
Wie viel Information teilen $X$ und $Y$ nach Beobachtung einer dritten Variablen $Z$?

<!--en:a9802757-->
- **Bedingte gegenseitige Information:**
  $$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z) \ge 0$$
  Sie ist gleich $\mathbb{E}_z[D_{\textrm{KL}}(P_{X,Y\mid z} \| P_{X\mid z}\otimes P_{Y\mid z})]$ und verschwindet genau dann, wenn $X \perp Y \mid Z$ gilt.

<!--en:59e2e25b-->
- **Kettenregel für gegenseitige Information:**
  $$I(X; Y, Z) = I(X; Z) + I(X; Y \mid Z) = I(X; Y) + I(X; Z \mid Y)$$

<!--en:9400d48e-->
- **Zweimaliges Kippen in Folge ($\delta_1 = 0.1, \delta_2 = 0.2 \implies \delta = 0.26$):** $I(X; Y, Z) = 0.368 = 0.120 + 0.248\text{ nats}$.

<!--en:8f4cb2a6-->
::: {.slide title="Die Datenverarbeitungsungleichung (DPI)"}

<!--en:16ea4e4a-->
[Schrittweise Verarbeitung kann keine Information erzeugen]{.kicker}

<!--en:c11f3f68-->
Drei Variablen bilden eine **Markov-Kette** $X \to Y \to Z$, wenn $Z$ von $(X, Y)$ nur über $Y$ abhängt ($p(z \mid x, y) = p(z \mid y)$ oder gleichwertig $I(X; Z \mid Y) = 0$).

<!--en:82a7d2dc-->
**Satz (Datenverarbeitungsungleichung):** Ist $X \to Y \to Z$ eine Markov-Kette, so gilt
$$I(X; Z) \le I(X; Y)$$ mit Gleichheit genau dann, wenn $I(X; Y \mid Z) = 0$ gilt (also auch $X \to Z \to Y$ eine Markov-Kette ist).

<!--en:d7df146b-->
- **Beweis in zwei Zeilen mit der Kettenregel:** Wir zerlegen $I(X; Y, Z)$ in beiden Reihenfolgen:
  $$\begin{aligned} I(X; Z) + I(X; Y \mid Z) &= I(X; Y, Z) = I(X; Y) + \underbrace{I(X; Z \mid Y)}_{= 0 \text{ (Markov)}} \\ \implies I(X; Z) &= I(X; Y) - \underbrace{I(X; Y \mid Z)}_{\ge 0} \le I(X; Y) \quad \blacksquare \end{aligned}$$

<!--en:53c747b4-->
- **Suffiziente Statistiken:** Gleichheit $I(X; T(Y)) = I(X; Y)$ gilt genau dann, wenn $T(Y)$ alles bewahrt, was $Y$ über $X$ verrät ($p(x \mid y) = p(x \mid T(y))$, etwa die Anzahl der Kopfwürfe für die Kopfwahrscheinlichkeit einer Münze).

<!--en:1c9ffe2f-->
::: {.slide title="Konditionieren und Verarbeiten: Synergie beim exklusiven Oder"}

<!--en:af0f3f9e-->
[Wann Konditionieren die gegenseitige Information erhöht]{.kicker}

<!--en:8905644d-->
Die Ungleichung $I(X; Y \mid Z) \le I(X; Y)$ gilt in einer Markov-Kette $X \to Y \to Z$, aber *nicht* allgemein. Sind $X, Y$ unabhängige gleichverteilte Bits und gilt $Z = X \oplus Y$, so steigt ihre gegenseitige Information bei Beobachtung von $Z$ von $0$ auf $\ln 2 = 0.693\text{ nats}$:

<!--en:fefa4e5b-->
::: {.slide title="Was Schichten tiefer Netze leisten: nutzbare Information"}

<!--en:478ba0ba-->
[Die Datenverarbeitungsungleichung im Deep Learning]{.kicker}

<!--en:cc3a9c8e-->
Entlang eines Feedforward-Netzes $Y \to X \to Z_1 \to Z_2 \to \cdots$ garantiert die Datenverarbeitungsungleichung:

<!--en:7c81ec36-->
- **Warum Tiefe hilft:** Versteckte Schichten können keine neue Shannon-Information erzeugen. Sie können jedoch konzentrische Mannigfaltigkeiten trennen (etwa durch $Z_1 = \|X\|_2^2 \in \{1, 4\}$), sodass selbst eine lineare Ausgabefunktion $\mathcal{V}$ alle $\ln 2$ Nat auslesen kann (**nutzbare Information**).

<!--en:40185af0-->
::: {.slide title="Fano-Ungleichung: Vorhersage erfordert Information"}

<!--en:bdb5f4d8-->
[Untere Schranke für den Klassifikationsfehler]{.kicker}

<!--en:a3abf1c7-->
Wie viel gegenseitige Information $I(X; Y)$ muss die Eingabe $X$ über ein $k$-wertiges Label $Y$ enthalten, damit *überhaupt ein* Klassifikator $\hat{Y} = g(X) \in \{1, \ldots, k\}$ die Fehlerwahrscheinlichkeit $P_e = \Pr(\hat{Y} \neq Y)$ erreichen kann?

<!--en:d5cf21ad-->
**Satz (Fano-Ungleichung):** Für jede $k$-wertige Zielvariable $Y$ und jeden Schätzer $\hat{Y} = g(X) \in \{1, \ldots, k\}$ mit Fehlerrate $P_e = \Pr(\hat{Y} \neq Y)$ gilt
$$H_b(P_e) + P_e \log(k - 1) \ge H(Y \mid X) = H(Y) - I(X; Y)$$

<!--en:f6fe6a32-->
- **Kodierung als Korrektur von $\hat{Y}$:** Um $Y$ aus $\hat{Y}$ zu rekonstruieren, senden wir $H_b(P_e)$ Nat, die angeben, ob $\hat{Y}$ falsch lag. Im Fehlerfall geben höchstens $\log(k-1)$ weitere Nat an, welche der übrigen $k-1$ Klassen vorlag (Fehlerwahrscheinlichkeit $P_e$).

<!--en:8942ac01-->
::: {.slide title="Beweis der Fano-Ungleichung in drei Schritten"}

<!--en:b5050e3a-->
[Fehlerindikator und Kettenregel]{.kicker}

<!--en:a5cd4e08-->
Sei $W = \mathbf{1}\{\hat{Y} \neq Y\} \in \{0, 1\}$ der binäre Fehlerindikator ($P(W = 1) = P_e$). Wir zerlegen die bedingte Entropie $H(W, Y \mid \hat{Y})$ in beiden Reihenfolgen:

<!--en:72f63103-->
- **Schritt 1 ($W$ ist eine deterministische Funktion von $Y$ und $\hat{Y}$):**
  $$H(W, Y \mid \hat{Y}) = H(Y \mid \hat{Y}) + \underbrace{H(W \mid Y, \hat{Y})}_{= 0} = H(Y \mid \hat{Y})$$

<!--en:f3a8d82e-->
- **Schritt 2 (zuerst nach $W$ zerlegen und beide Terme abschätzen):**
  $$H(W, Y \mid \hat{Y}) = \underbrace{H(W \mid \hat{Y})}_{\le H(W) = H_b(P_e)} + \underbrace{H(Y \mid W, \hat{Y})}_{= (1-P_e)\cdot 0 + P_e H(Y \mid W=1, \hat{Y}) \le P_e \log(k-1)}$$

<!--en:1d01ebcc-->
- **Schritt 3 (Datenverarbeitungsungleichung auf $Y \to X \to \hat{Y}$ anwenden):** Aus $I(Y; \hat{Y}) \le I(Y; X)$ folgt $H(Y \mid X) \le H(Y \mid \hat{Y}) \le H_b(P_e) + P_e \log(k - 1)$. $\blacksquare$

<!--en:1adb090d-->
::: {.slide title="Informationsnachweis durch den Testverlust: I(X; Y) ≥ H(Y) − L"}

<!--en:8ea1d6d6-->
[Variationsschranke von Barber und Agakov]{.kicker}

<!--en:9a1ac2c9-->
Die Fano-Ungleichung nutzt die Fehlerrate $P_e$ der eindeutigen Entscheidungen. Jeder probabilistische Prädiktor $q(y \mid x)$ mit Kreuzentropieverlust $L = \mathbb{E}_{(x,y) \sim P_{X,Y}}[-\log q(y \mid x)]$ auf zurückgehaltenen Daten liefert direkt eine untere Schranke für $I(X; Y)$:

<!--en:2e08dc2e-->
::: {.slide title="Shannons zwei Kurven: Kapazität und Rate-Verzerrungs-Funktion"}

<!--en:ca5d986f-->
[Extremwerte der gegenseitigen Information]{.kicker}

<!--en:3200c44b-->
Shannons grundlegende Grenzen für Kommunikation und verlustbehaftete Kompression ergeben sich aus zwei dualen Extremwertproblemen für die gegenseitige Information:

<!--en:dbde9b07-->
- **Kanalkapazität (links):** Bei festem verrauschtem Kanal $p(y \mid x)$ *maximieren* wir über die Eingabeverteilung des Senders: $C = \max_{p(x)} I(X; Y)$.

<!--en:6371804c-->
- **Rate und Verzerrung (rechts):** Bei fester Quelle $p(x)$ *minimieren* wir über den verlustbehafteten Encoder $p(\hat{x} \mid x)$ unter der Verzerrungsschranke $D$: $R(D) = \inf_{p(\hat{x} \mid x): \mathbb{E}[d(X, \hat{X})] \le D} I(X; \hat{X})$.

<!--en:59c37705-->
::: {.slide title="Kapazität des binären symmetrischen Kanals"}

<!--en:bdc88c69-->
[Maximierung über die Eingabeverteilungen]{.kicker}

<!--en:4417ed13-->
Kippt jedes übertragene Bit in einem binären symmetrischen Kanal unabhängig mit Wahrscheinlichkeit $\delta \in [0, 1]$, so hat jede Zeile von $p(y \mid x)$ die Entropie $H_b(\delta)$. Daher wird $I(X; Y) = H(Y) - H_b(\delta)$ maximal, wenn $Y$ (und damit $X$) gleichverteilt ist ($\tfrac12, \tfrac12$):

<!--en:3e0b5023-->
::: {.slide title="Gaußsche Rate-Verzerrungs-Funktion: 1 Bit halbiert die Wurzel des mittleren quadratischen Fehlers"}

<!--en:ff022d96-->
[Minimierung über verlustbehaftete Encoder]{.kicker}

<!--en:8a735c16-->
Für eine gaußsche Quelle $X \sim \mathcal{N}(0, \sigma_X^2)$ unter der Schranke für den quadratischen Fehler $\mathbb{E}[(X - \hat{X})^2] \le D < \sigma_X^2$ lautet die Rate-Verzerrungs-Funktion:

<!--en:a6b4c083-->
- **Beweis der unteren Schranke:**
  $$\begin{aligned} I(X; \hat{X}) &= h(X) - h(X - \hat{X} \mid \hat{X}) \ge h(X) - h(X - \hat{X}) \\ &\ge \tfrac12\log(2\pi e\sigma_X^2) - \tfrac12\log(2\pi e D) = \tfrac12\log(\sigma_X^2/D). \end{aligned}$$

<!--en:dacee9a1-->
- **Die Schranke mit einem additiven gaußschen Kanal erreichen:** Wir kodieren $Z = X + \sigma\epsilon$ und rekonstruieren den A-posteriori-Mittelwert $\hat{X} = \mathbb{E}[X \mid Z] = \frac{\sigma_X^2}{\sigma_X^2 + \sigma^2}Z$:

<!--en:25032209-->
::: {.slide title="Der Informationsflaschenhals als Rate-Verzerrungs-Problem"}

<!--en:7785b855-->
[Aufgabenbezogenes Repräsentationslernen]{.kicker}

<!--en:775cbdfe-->
Entlang $Y \to X \to Z$ wägt der **Informationsflaschenhals** (Tishby et al., 1999) die Kompressionsrate gegen die bewahrte Information über das Label ab:

<!--en:fe9e26ce-->
- **Rate-Verzerrungs-Identität:** $I(Y; Z) = I(X; Y) - \mathbb{E}[D_{\textrm{KL}}(p(y \mid x) \| p(y \mid z))]$. Der Encoder kollabiert ($I(X; Z) = 0$), sobald $\beta \le 1$ gilt.

<!--en:18d47a00-->
::: {.slide title="Geschlossene Lösung für den gaußschen Informationsflaschenhals"}

<!--en:f3999d7e-->
[Konkave Grenzkurve und Phasenübergang]{.kicker}

<!--en:58bd9e8f-->
Für das bivariat normalverteilte Paar $(X, Y)$ mit Korrelation $\rho = 0.9$ ($I(X; Y) = 0.830\text{ nats}$) und additivem Encoder $Z = X + \sigma\epsilon$ erhalten wir mit der Rate $u = I(X; Z) = \tfrac12\log(1 + \sigma^{-2})$ die exakte optimale Grenzkurve:

<!--en:d5cf71e6-->
- **Konkave Kurve in der Informationsebene:**
  $$I(Y; Z) = -\frac{1}{2}\log\Big(1 - \rho^2\big(1 - e^{-2u}\big)\Big)$$

<!--en:eaafdacf-->
- **Phasenübergang bei $\beta_c = 1/\rho^2$:** Die Grenzkurve beginnt bei $(0, 0)$ mit Steigung $\frac{dI(Y;Z)}{du}\big|_{u=0} = \rho^2 = 0.81$. Für alle $\beta \le 1/\rho^2 \approx 1.235$ **kollabiert** der optimale Encoder ($u^* = 0$).

<!--en:da41a32d-->
- **Arbeitspunkt für $\beta > 1/\rho^2$:**
  $$u^* = \frac{1}{2}\log\frac{\rho^2(\beta - 1)}{1 - \rho^2}$$

<!--en:a5abeba0-->
::: {.slide title="Phasenübergang des Informationsflaschenhalses überprüfen und VIB"}

<!--en:bb361bb0-->
[Gittersuche und Variationsautoencoder]{.kicker}

<!--en:3a09721b-->
Eine Gittersuche über $4{,}001$ Rauschstärken $\sigma$ bestätigt den Kollaps bei $\beta = 1.20$ und die positive Rate bei $\beta = 1.25$:

<!--en:a03ce4b3-->
- **Variationeller Informationsflaschenhals (VIB) und VAEs:** Ist $p(z)$ unbekannt, liefert der Ersatz durch eine A-priori-Verteilung $r(z) = \mathcal{N}(0, I)$ eine obere Schranke für die Rate gemäß $\mathbb{E}_x[D_{\textrm{KL}}(p(z \mid x) \| r(z))] = I(X; Z) + D_{\textrm{KL}}(p(z) \| r(z)) \ge I(X; Z)$. Das ist genau der KL-Regularisierungsterm eines Variationsautoencoders.

<!--en:ec8af6e0-->
::: {.slide title="Verständnisfragen: Kanäle in Reihe, Kapazität und Informationsflaschenhals"}

<!--en:929b7671-->
[Überprüfen Sie Ihr Verständnis]{.kicker}

<!--en:e140a4fa-->
Lösen Sie die drei Aufgaben:

<!--en:90ca82e6-->
1. **Zwei verrauschte Leitungen in Reihe:** Ein gleichverteiltes Bit $X$ durchläuft einen Kanal mit Kippwahrscheinlichkeit $\delta_1 = 0.1$ und ergibt $Y$. Anschließend durchläuft $Y$ einen unabhängigen Kanal mit Kippwahrscheinlichkeit $\delta_2 = 0.2$ und ergibt $Z$. Wie groß sind $I(X; Y)$, $I(X; Z)$, $I(X; Z \mid Y)$ und $I(X; Y \mid Z)$?

<!--en:10a664c4-->
2. **Kapazität des binären symmetrischen Kanals:** Warum hat ein solcher Kanal mit Kippwahrscheinlichkeit $\delta = 1.0$ dieselbe Kapazität ($\ln 2\text{ nats}$) wie ein rauschfreier Kanal ($\delta = 0$), während die Kapazität für $\delta = 0.5$ nur $0$ beträgt?

<!--en:82db2a1b-->
3. **Arbeitspunkt des gaußschen Informationsflaschenhalses:** Berechnen Sie für das gaußsche Paar mit $\rho = 0.9$ ($\rho^2 = 0.81$) die optimale Rate $u^* = I(X; Z)^*$ und die bewahrte Information über das Label $I(Y; Z)^*$ beim Multiplikator $\beta = 2$.

<!--en:29b5f4fc-->
::: {.slide title="Verständnisfragen: Kanäle in Reihe, Kapazität und Informationsflaschenhals (Fortsetzung)"}

<!--en:929b7671-2-->
[Überprüfen Sie Ihr Verständnis]{.kicker}

<!--en:557b98b6-->
**Antworten:**

<!--en:61ccedf4-->
1. **Zweimaliges Kippen in Folge:** $I(X; Y) = \ln 2 - H_b(0.1) = 0.368\text{ nats}$. Die zusammengesetzte Kippwahrscheinlichkeit beträgt $0.1(0.8) + 0.9(0.2) = 0.26$, also gilt $I(X; Z) = \ln 2 - H_b(0.26) = 0.120\text{ nats}$. Aus der Markov-Kette $X \to Y \to Z$ folgt $I(X; Z \mid Y) = 0$, und die Kettenregel liefert $I(X; Y \mid Z) = 0.368 - 0.120 = 0.248\text{ nats}$.

<!--en:1c996b23-->
2. **Kapazität des binären symmetrischen Kanals:** $C(\delta) = \ln 2 - H_b(\delta)$. Bei $\delta = 1$ wird jedes Bit deterministisch invertiert ($H_b(1) = 0$) und der Empfänger kann es wieder umkehren. Bei $\delta = 0.5$ ist die Ausgabe wie ein fairer Münzwurf unabhängig von der Eingabe ($H_b(0.5) = \ln 2 \implies C = 0$).

<!--en:7fa4c5a5-->
3. **Gaußscher Informationsflaschenhals bei $\beta = 2$:** Wegen $\beta = 2 > 1/0.81 = 1.235$ gelten $u^* = \tfrac12\ln\frac{0.81(1)}{0.19} = 0.725\text{ nats}$ und $I(Y; Z)^* = \tfrac12\ln\frac{1}{2(0.19)} = 0.484\text{ nats}$.

<!--en:58236a55-->
::: {.slide title="Zusammenfassung: Gegenseitige Information und Informationsflaschenhals"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:aa4f0aae-->
| Begriff oder Satz | Formel | Invarianz oder Schranke | Rolle im Deep Learning |

<!--en:ee62a718-->
| **Gegenseitige Information** | $I(X; Y) = D_{\textrm{KL}}(P_{X,Y} \| P_X \otimes P_Y)$ | $I(\phi(X); \psi(Y)) = I(X; Y)$ für Bijektionen | Misst beliebige nichtlineare Abhängigkeit in Nat |

<!--en:b2b0f861-->
| **Gegenseitige Information eines gaußschen Paares** | $I(X; Y) = -\tfrac12\log(1 - \rho^2)$ | Hängt nur vom unerklärten Varianzanteil $1-\rho^2$ ab | Exakter Vergleichswert, divergiert für $\rho \to \pm 1$ ($Y = X$) |

<!--en:a02b1212-->
| **Punktweise gegenseitige Information (PMI)** | $\textrm{pmi}(x, y) = \log\frac{p(x,y)}{p(x)p(y)}$ | $\mathbb{E}_{P_{X,Y}}[\textrm{pmi}] = I(X; Y)$ und $\mathbb{E}_{P_X\otimes P_Y}[e^{\textrm{pmi}}] = 1$ | Ranking von Kollokationen in der Sprachverarbeitung, optimaler Kritiker für Word2Vec und InfoNCE |

<!--en:ee010d62-->
| **Datenverarbeitungsungleichung (DPI)** | $X \to Y \to Z \implies I(X; Z) \le I(X; Y)$ | Differenz $I(X; Y \mid Z) \ge 0$ (genau 0 bei suffizienter Statistik) | Schichten können Information über das Label linear auslesbar machen, aber keine erzeugen |

<!--en:25ba3240-->
| **Informationsnachweis durch Fano-Ungleichung und Verlust** | $H_b(P_e) + P_e\log(k-1) \ge H(Y \mid X)$ und $I \ge H(Y) - L$ | Gilt für jeden Klassifikator $\hat{Y} = g(X)$ oder jedes Modell $q(y \mid x)$ | Liefert aus Testgenauigkeit oder negativer Log-Likelihood eine Untergrenze für die Information in der Eingabe |

<!--en:b457bcb6-->
| **Kapazität und Rate-Verzerrungs-Funktion** | $C = \max_{p(x)} I(X; Y)$ und $R(D) = \min_{\mathbb{E}d \le D} I(X; \hat{X})$ | Binärer symmetrischer Kanal: $\ln 2 - H_b(\delta)$, gaußsche Quelle: $\tfrac12\log(\sigma_X^2/D)$ | Grundlegende Grenzen für verrauschte Kanäle und verlustbehaftete Autoencoder |

<!--en:d145d996-->
| **Informationsflaschenhals** | $\min_{p(z\mid x)} I(X; Z) - \beta I(Y; Z)$ | Gaußscher Kollaps für $\beta \le 1/\rho^2$, VIB-Rate $\le \mathbb{E}[D_{\textrm{KL}}(p \| r)]$ | Aufgabenspezifische Kompression, erklärt den KL-Regularisierungsterm des VAE |
