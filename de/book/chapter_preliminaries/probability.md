<!--en:5e7faec5-->
# Wahrscheinlichkeit und Statistik

<!--en:45ce2983-->
Beim maschinellen Lernen ziehen wir Schlüsse unter Unsicherheit. Wie wahrscheinlich ist es, dass ein Patient einen Herzinfarkt erleidet? Ist ein Sensormesswert ungewöhnlich? Die Genauigkeit fällt von $0.90$ ($230/256$ Testbilder richtig erkannt) auf $0.87$ beim nächsten Minibatch. Hat sie sich tatsächlich verändert, oder sehen wir eine zufällige Stichprobenschwankung?

<!--en:67be9aaf-->
Der Münzwurf ist noch immer das einfachste Gedankenexperiment der Wahrscheinlichkeitstheorie. Eine Münze zeigt $56$-mal Kopf bei $100$ Würfen. Sollten wir vermuten, dass sie unfair ist? Selbst eine faire Münze zeigt selten genau $50$-mal Kopf. Wir müssen also fragen, ob eine Abweichung von $6$ Köpfen größer ist, als wir sie durch Zufall regelmäßig beobachten. Die *Wahrscheinlichkeitstheorie* geht von einem bekannten Modell (einer fairen Münze) aus und bestimmt, wie wahrscheinlich die möglichen Ergebnisse sind. Die *Statistik* geht den umgekehrten Weg. Sie schließt aus beobachteten Daten ($56$-mal Kopf bei $100$ Würfen) auf Eigenschaften des zugrunde liegenden Prozesses.

<!--en:64f8ff51-->
In diesem Abschnitt simulieren wir Münzwürfe und beobachten, wie die Ergebnisse von Stichprobe zu Stichprobe schwanken. Wir führen Ergebnisräume, Ereignisse, Zufallsvariablen (engl. random variables), gemeinsame Verteilungen (engl. joint distributions) und Unabhängigkeit ein. Aus Erwartungswert und Varianz leiten wir den *Standardfehler* (engl. standard error) $\sigma/\sqrt{n}$ her, um die Frage zur Münze zu beantworten. Anschließend erarbeiten wir die bedingte Wahrscheinlichkeit und den Satz von Bayes anhand eines zweistufigen medizinischen Tests.

<!--en:21d45b56-->
## Sampling

<!--en:116bea33-->
### Simulierte Münzwürfe

<!--en:2b66d2b6-->
Bei einer *fairen* Münze sind Kopf und Zahl symmetrisch, also gilt $P(\textrm{heads}) = 0.5$. Das Ziehen von Realisierungen eines stochastischen Prozesses heißt *Sampling*. Wir simulieren $100$ Würfe zunächst, indem wir gleichverteilte Werte aus $[0, 1)$ ziehen und prüfen, ob sie größer als $0.5$ sind. Anschließend ziehen wir mit unserem Framework aus der *Multinomialverteilung* mit $n = 100$ und den Wahrscheinlichkeiten $(0.5, 0.5)$:

<!--en:633c48bd-->
In JAX teilen wir einen expliziten PRNG-Schlüssel auf, den wir mit einem Startwert aus `random` initialisieren. So erhalten wir bei jedem Durchlauf neue Anzahlen.

<!--en:8885d18f-->
### Häufigkeiten und Wahrscheinlichkeiten

<!--en:ecd75a0b-->
Teilen wir die beobachteten Anzahlen durch $n$, erhalten wir die relative *Häufigkeit* $\hat{p}$. Sie ist ein *Schätzer* (engl. estimator) für die wahre Wahrscheinlichkeit $p = P(\textrm{heads}) = 0.5$. Führen wir das Experiment mit 100 Würfen fünfmal parallel aus, erhalten wir eine $5 \times 2$-Matrix relativer Häufigkeiten. Sie schwanken merklich um $0.5$:

<!--en:433a88e9-->
Verhundertfachen wir den Stichprobenumfang auf $n = 10{,}000$ Würfe, sinkt der typische Stichprobenfehler (engl. sampling error) um den Faktor $\sqrt{100} = 10$. Die relativen Häufigkeiten von Kopf und Zahl liegen dann viel näher bei $(0.5, 0.5)$:

<!--en:6d42a65e-->
Verfolgen wir nach jedem Wurf von $1$ bis $10{,}000$ mit `cumsum` die relative Häufigkeit, sehen wir das *Gesetz der großen Zahlen* (engl. law of large numbers) in Aktion (:numref:`sec_mdl-statistics`). Es garantiert $\hat{p} \to p$ für $n \to \infty$:

<!--en:bcf814a9-->
Um zu verstehen, *wie schnell* sich $\hat{p}$ stabilisiert und ob $56/100$ auffällig ist, beschreiben wir nun Zufallsvariablen und ihre Streuung mathematisch.

<!--en:60a76152-->
## Ereignisse und Zufallsvariablen

<!--en:22e6763c-->
### Ergebnisräume und Ereignisse

<!--en:9ebdac3a-->
Die Menge aller möglichen Ergebnisse eines Zufallsexperiments heißt *Ergebnisraum* (engl. sample space) $\mathcal{S}$. Für einen Würfelwurf gilt $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$. Jede Teilmenge $\mathcal{A} \subseteq \mathcal{S}$ heißt *Ereignis*. Beispiele sind „eine gerade Zahl würfeln“, $\mathcal{A} = \{2, 4, 6\}$, und „mindestens $5$ würfeln“, $\mathcal{B} = \{5, 6\}$. Ein *Wahrscheinlichkeitsmaß* $P$ ordnet jedem Ereignis $\mathcal{A} \subseteq \mathcal{S}$ eine reelle Zahl zu. Dabei gelten die drei Axiome von Kolmogorov :cite:`Kolmogorov.1933`:

<!--en:e0616503-->
* **Nichtnegativität:** $P(\mathcal{A}) \geq 0$.

<!--en:f7cdeb8b-->
* **Normierung:** $P(\mathcal{S}) = 1$.

<!--en:44515ed4-->
* **Abzählbare Additivität:** Für paarweise disjunkte Ereignisse $\mathcal{A}_1, \mathcal{A}_2, \ldots$ ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ für $i \neq j$) gilt $P(\bigcup_i \mathcal{A}_i) = \sum_i P(\mathcal{A}_i)$.

<!--en:49a04015-->
Aus diesen Axiomen folgen die Gegenwahrscheinlichkeit $P(\mathcal{A}') = 1 - P(\mathcal{A})$, die Gleichung $P(\emptyset) = 0$ und die Formel für das Prinzip der Inklusion und Exklusion (:numref:`fig_prob_venn`):

<!--en:11562f8e-->
![Zwei Ereignisse $\mathcal{A}$ und $\mathcal{B}$ im Ergebnisraum $\mathcal{S}$. Die Summe $P(\mathcal{A}) + P(\mathcal{B})$ zählt die Schnittmenge $\mathcal{A} \cap \mathcal{B}$ doppelt.](../img/probability-venn.svg)

<!--en:4198d9ae-->
### Zufallsvariablen und Verteilungen

<!--en:2aa73bd4-->
Eine *Zufallsvariable* $X$ ordnet jedem Ergebnis in $\mathcal{S}$ einen Zahlenwert zu. Zählen wir zum Beispiel die Köpfe bei $100$ Münzwürfen, bilden wir $2^{100}$ mögliche Folgen auf $\{0, \ldots, 100\}$ ab. Bei einer *diskreten* Zufallsvariablen entfällt auf jeden Wert eine Wahrscheinlichkeitsmasse. Die Masse $P(X = v)$ für jeden Wert $v$ bestimmt ihre *Verteilung* $P(X)$. Eine *stetige* Zufallsvariable (etwa die exakte Körpergröße eines Menschen) beschreiben wir durch eine *Wahrscheinlichkeitsdichte* $p(x)$ (:numref:`fig_prob_density`). Die Wahrscheinlichkeit, dass ihr Wert in $[a, b]$ liegt, entspricht der Fläche unter der Dichtekurve über diesem Intervall, also $P(a \leq X \leq b) = \int_a^b p(x)\,dx$ (:numref:`sec_mdl-random_variables`).

<!--en:8b29ebbd-->
![Eine diskrete Zufallsvariable weist einzelnen Werten Wahrscheinlichkeiten zu (links). Eine stetige Zufallsvariable hat eine Dichte. Die Wahrscheinlichkeit eines Intervalls entspricht der Fläche unter der Dichtekurve über diesem Intervall (rechts).](../img/probability-density.svg)

<!--en:44ed8621-->
### Gemeinsame Verteilung, Randverteilung und Unabhängigkeit

<!--en:e30c626d-->
Die *gemeinsame Verteilung* $P(X = x, Y = y)$ zweier Zufallsvariablen gibt an, mit welcher Wahrscheinlichkeit $X = x$ und $Y = y$ zusammen eintreten. Betrachten wir zum Beispiel, ob es an einem zufällig gewählten Tag regnet ($X$) und ob jemand einen Schirm mitnimmt ($Y$):

<!--en:460ae05a-->
| $P(X, Y)$ | mit Schirm | ohne Schirm |

<!--en:17ff837f-->
| Regen | 0.25 | 0.05 |

<!--en:a07cf1d6-->
| kein Regen | 0.10 | 0.60 |

<!--en:60710b96-->
Summieren wir über alle Werte einer Variablen, erhalten wir die *Randverteilung* (engl. marginal distribution) der anderen (*Marginalisierung*, :numref:`fig_prob_joint`):

<!--en:7dc9e814-->
Damit gilt $P(\textrm{rain}) = 0.25 + 0.05 = 0.30$ und $P(\textrm{umbrella}) = 0.25 + 0.10 = 0.35$.

<!--en:fcbddb37-->
![Eine gemeinsame Verteilung $P(X, Y)$ in Tabellenform. Die Zeilen- oder Spaltensummen ergeben jeweils eine Randverteilung. Division einer Zeile durch ihre Summe ergibt eine bedingte Verteilung, hier $P(Y \mid X = x_2)$.](../img/probability-joint-grid.svg)

<!--en:66134e2b-->
Zwei Zufallsvariablen $X$ und $Y$ sind *unabhängig* ($X \perp Y$), wenn ihre gemeinsame Verteilung das Produkt ihrer Randverteilungen ist:

<!--en:cd9362d1-->
Aufeinanderfolgende Münzwürfe sind unabhängig. Ob es regnet und ob jemand einen Schirm mitnimmt, hängt hingegen zusammen ($0.25 \neq 0.30 \cdot 0.35 = 0.105$).

<!--en:20d1800c-->
## Erwartungswert und Varianz

<!--en:42b5db2a-->
### Erwartungswert

<!--en:9be8230c-->
Der *Erwartungswert* (engl. expectation) oder *Mittelwert* einer diskreten Zufallsvariablen $X$ ist der mit den Wahrscheinlichkeiten gewichtete Durchschnitt ihrer Werte:

<!--en:8087d91f-->
Für jede Funktion $f$ gilt entsprechend $\mathbb{E}[f(X)] = \sum_x f(x)\,P(X = x)$. Eine Anlage zahlt $0\times$ den Einsatz mit Wahrscheinlichkeit $0.5$ aus, $2\times$ mit Wahrscheinlichkeit $0.4$ und $10\times$ mit Wahrscheinlichkeit $0.1$. Ihr Erwartungswert beträgt $\mathbb{E}[X] = 0.5(0) + 0.4(2) + 0.1(10) = 1.8$. Für die Indikatorvariable $X = \mathbf{1}(\textrm{heads}) \in \{0, 1\}$ eines Münzwurfs gilt $\mathbb{E}[X] = 1\cdot p + 0\cdot(1-p) = p$.

<!--en:e4c01879-->
Der Erwartungswert ist stets *linear*. Die Gleichung $\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$ gilt unabhängig davon, ob $X$ und $Y$ unabhängig sind. Für die relative Häufigkeit $\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i$ folgt daraus $\mathbb{E}[\hat{p}] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i] = p$. Also ist $\hat{p}$ ein *erwartungstreuer* (engl. unbiased) Schätzer für $p$.

<!--en:95c6df73-->
### Varianz und Summen unabhängiger Variablen

<!--en:3045b6ff-->
Die *Varianz* misst die Streuung um $\mu = \mathbb{E}[X]$ als Mittelwert der quadrierten Abweichungen. Die *Standardabweichung* $\sigma = \sqrt{\textrm{Var}(X)}$ hat wieder die ursprüngliche Maßeinheit:

<!--en:4e3a9569-->
Für unsere Anlage gilt $\mathbb{E}[X^2] = 0.5(0) + 0.4(4) + 0.1(100) = 11.6$. Daraus folgt $\textrm{Var}(X) = 11.6 - 1.8^2 = 8.36$ ($\sigma \approx 2.9$). Für die Indikatorvariable $X \in \{0, 1\}$ eines einzelnen Münzwurfs gilt $X^2 = X$ und damit $\textrm{Var}(X) = p - p^2 = p(1 - p)$. Diese Varianz erreicht ihr Maximum $1/4$ bei $p = 0.5$. Bei $n$ Beobachtungen schätzen wir die Varianz wie in :numref:`sec_pandas` anhand der Abweichungen vom Stichprobenmittelwert. Dabei teilen wir durch $n - 1$ (*Bessel-Korrektur*). Damit gleichen wir die leichte Verzerrung nach unten aus, die beim Teilen durch $n$ entsteht.

<!--en:d769985f-->
Für die Summe zweier Zufallsvariablen multiplizieren wir das Quadrat in $\mathbb{E}[(X - \mu_X + Y - \mu_Y)^2]$ aus und erhalten

<!--en:dcb17034-->
Dabei messen die *Kovarianz* $\textrm{Cov}(X, Y)$ und die dimensionslose *Korrelation* $\rho = \textrm{Cov}(X, Y)/(\sigma_X\sigma_Y) \in [-1, 1]$ den linearen Zusammenhang (:numref:`fig_prob_covariance`). Betrachten wir allgemeiner einen Zufallsvektor $\mathbf{x} \in \mathbb{R}^d$ mit Mittelwert $\boldsymbol{\mu} = \mathbb{E}[\mathbf{x}]$. Seine paarweisen Kovarianzen $\Sigma_{ij} = \textrm{Cov}(x_i, x_j)$ bilden die symmetrische $d \times d$-*Kovarianzmatrix* $\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^\top]$. Jede Linearkombination $\mathbf{v}^\top\mathbf{x}$ hat die Varianz $\textrm{Var}(\mathbf{v}^\top\mathbf{x}) = \mathbf{v}^\top\boldsymbol{\Sigma}\mathbf{v}$.

<!--en:e9f8d942-->
![Stichproben von drei Paaren von Zufallsvariablen mit positiver, nahezu verschwindender und negativer Kovarianz.](../img/probability-covariance.svg)

<!--en:2d95b396-->
Sind $X$ und $Y$ *unabhängig*, zerfällt der Erwartungswert ihres Produkts in das Produkt der Erwartungswerte. Damit gilt $\textrm{Cov}(X, Y) = \mathbb{E}[X - \mu_X]\mathbb{E}[Y - \mu_Y] = 0$, und **die Varianzen unabhängiger Variablen addieren sich**:

<!--en:0ef2ea53-->
Aus einer Kovarianz von null folgt allerdings keine Unabhängigkeit. Nehmen wir an, $X$ nimmt die Werte $-1, 0, +1$ mit gleicher Wahrscheinlichkeit an und es gilt $Y = X^2$. Dann ist $Y$ durch $X$ vollständig bestimmt. Aus Symmetriegründen gelten jedoch $\mathbb{E}[X] = 0$ und $\mathbb{E}[XY] = \mathbb{E}[X^3] = 0$ und damit $\textrm{Cov}(X, Y) = 0$.

<!--en:7ea660aa-->
## Schätzfehler

<!--en:c38a5ba5-->
### Standardfehler einer relativen Häufigkeit

<!--en:b9f8257e-->
Bei $n$ unabhängigen Münzwürfen hat jede Indikatorvariable die Varianz $p(1 - p)$. Ihre Summe hat daher die Varianz $np(1 - p)$. Teilen wir die Summe durch $n$, ändert sich ihre Varianz um den Faktor $1/n^2$:

<!--en:1c8b4ca0-->
Ziehen wir die Quadratwurzel, erhalten wir den *Standardfehler* von $\hat{p}$. Er beträgt $\sqrt{p(1-p)/n}$, für eine faire Münze ($p = 0.5$) also $0.5/\sqrt{n}$ ($0.05$ bei $n = 100$ und $0.005$ bei $n = 10{,}000$). Wegen der Abnahme proportional zu $1/\sqrt{n}$ brauchen wir zum Halbieren des Fehlers $4\times$ so viele Daten. Allgemein hat der Mittelwert aus $n$ unabhängigen Ziehungen mit Varianz $\sigma^2$ den Standardfehler $\sigma/\sqrt{n}$. Simulieren wir $1000$ Wiederholungen für jeden Stichprobenumfang $n \in \{10, 100, 1000, 10000\}$, bestätigt sich für $0.5/\sqrt{n}$ die Gerade mit Steigung $-1/2$:

<!--en:566725a3-->
### Test auf eine unfaire Münze und Umgang mit Unsicherheit

<!--en:59563b5e-->
Nun können wir die Frage zur Münze beantworten. Bei $100$ Würfen liegt $\hat{p} = 0.56$ um $0.06 / 0.05 = 1.2$ Standardfehler von $0.5$ entfernt. Bei $10{,}000$ Würfen weicht dieselbe relative Häufigkeit $\hat{p} = 0.56$ ($5{,}600$-mal Kopf) um $0.06 / 0.005 = 12$ Standardfehler ab. Wir simulieren $1000$ Versuche mit einer fairen Münze. Wie oft die Abweichung mindestens so groß ausfällt, liefert uns einen Schätzwert für den *p-Wert*:

<!--en:b72daa07-->
Bei rund einem Viertel der Versuche mit 100 Würfen einer fairen Münze weicht die Anzahl der Köpfe um $6$ oder mehr vom Erwartungswert ab. Damit spricht $56/100$ noch nicht für eine unfaire Münze. Bei keinem der $1000$ Versuche mit je 10 000 Würfen einer fairen Münze kommt die Anzahl der Köpfe dagegen auch nur in die Nähe von $5{,}600$. Zwei allgemeine Sätze beschreiben die Wahrscheinlichkeit solcher Abweichungen. Die *Chebyshev-Ungleichung* (:numref:`sec_mdl-random_variables`) garantiert $P(|X - \mu| \geq k\sigma) \leq 1/k^2$ für jede Verteilung mit endlicher Varianz. Nach dem *zentralen Grenzwertsatz* (engl. central limit theorem, :numref:`sec_mdl-distributions`) nähert sich die Verteilung der Mittelwerte unabhängiger Ziehungen einer gaußschen Glockenkurve. Innerhalb von $2$ Standardfehlern um den Erwartungswert liegen dann ungefähr $95\%$ der Mittelwerte. Dieselbe Rechnung gilt bei der Modellauswertung. Hat ein Klassifikator die wahre Genauigkeit $0.9$, beträgt der Standardfehler $\sqrt{0.9 \cdot 0.1 / 256} \approx 0.019$ bei einem einzelnen Minibatch mit $256$ Bildern (Schwankungen um $\pm 2\%$ sind also gewöhnliches Rauschen). Dagegen beträgt er nur $0.003$ bei einem Testdatensatz mit $10{,}000$ Bildern.

<!--en:c1f39df0-->
Mit mehr Daten sinkt also die *epistemische Unsicherheit* über den Modellparameter $p$ proportional zu $1/\sqrt{n}$. Die *aleatorische Unsicherheit* nimmt dadurch nie ab. Beispiele sind die zufällige 50/50-Entscheidung beim nächsten Münzwurf und ein von sich aus mehrdeutiges Bild :cite:`Der-Kiureghian.Ditlevsen.2009`.

<!--en:86264e99-->
## Bedingte Wahrscheinlichkeit

<!--en:ae4c0a7d-->
### Bedingen, Satz von Bayes und bedingte Unabhängigkeit

<!--en:c52bf91f-->
Wir beschränken uns auf Ergebnisse mit $X = x$ ($P(X = x) > 0$) und normieren die Wahrscheinlichkeiten erneut. So erhalten wir die *bedingte Wahrscheinlichkeit* von $Y = y$ unter der Bedingung $X = x$:

<!--en:cc8bde37-->
In unserer Wettertabelle teilen wir unter der Bedingung „Regen“ die Regenzeile $(0.25, 0.05)$ durch $0.30$. Wir erhalten $P(\textrm{umbrella} \mid \textrm{rain}) = 0.25/0.30 \approx 0.83$ (ohne die Bedingung sind es $0.35$). Schreiben wir die gemeinsame Wahrscheinlichkeit in beiden Reihenfolgen, gilt $P(X, Y) = P(Y \mid X)P(X) = P(X \mid Y)P(Y)$. Teilen wir durch $P(Y) = \sum_x P(Y \mid X = x)P(X = x)$, erhalten wir den *Satz von Bayes* :cite:`Bayes.1763`:

<!--en:4234dfe3-->
Für eine Hypothese $H$ und beobachtete Daten $D$ aktualisiert der Satz von Bayes die *A-priori-Wahrscheinlichkeit* (engl. prior) $P(H)$. Aus der *Likelihood* $P(D \mid H)$ und der Evidenz $P(D)$ ergibt sich die *A-posteriori-Wahrscheinlichkeit* (engl. posterior) $P(H \mid D) \propto P(D \mid H)P(H)$.

<!--en:1382fa82-->
Zwei Variablen $X$ und $Y$ sind gegeben $Z$ *bedingt unabhängig* ($X \perp Y \mid Z$), wenn $P(X, Y \mid Z) = P(X \mid Z)P(Y \mid Z)$ gilt. Durch Bedingen kann eine Abhängigkeit verschwinden oder entstehen (:numref:`fig_prob_explaining_away`). Beobachten wir eine *gemeinsame Ursache* $Z$, etwa das Alter eines Kindes, verschwindet die Korrelation zwischen Schuhgröße und Lesefähigkeit. Kennen wir den gegenwärtigen Zustand $Y$ in einer *Markov-Kette* $X \to Y \to Z$, sind Vergangenheit $X$ und Zukunft $Z$ unabhängig. Umgekehrt entsteht zwischen zwei unabhängigen Ursachen eine Abhängigkeit, wenn wir auf eine *gemeinsame Wirkung* $Z$ bedingen (*Wegerklären*, engl. explaining away). Erfahren wir zum Beispiel, dass ein Patient im Krankenhaus einen Knochenbruch hat, sinkt die Wahrscheinlichkeit, dass Lungenkrebs der Grund für die Aufnahme war.

<!--en:f6a16d94-->
![Bedingen verändert Abhängigkeiten. Eine gemeinsame Ursache $Z$ macht $X$ und $Y$ abhängig. Bei beobachtetem $Z$ verschwindet die Abhängigkeit. In einer Kette sind $X$ und $Z$ unabhängig, wenn $Y$ gegeben ist. Zwei unabhängige Ursachen mit gemeinsamer Wirkung $Z$ werden abhängig, sobald $Z$ beobachtet wird.](../img/probability-explaining-away.svg)

<!--en:02da8c89-->
### Beispiel eines diagnostischen Tests

<!--en:b1037728-->
Angenommen, ein Patient macht einen HIV-Suchtest $D_1 \in \{0, 1\}$. Sein tatsächlicher HIV-Status sei $H \in \{0, 1\}$. Die Sensitivität des Tests beträgt $100\%$, seine Falsch-Positiv-Rate $1\%$:

<!--en:f9874071-->
| $P(D_1 \mid H)$ | $H=1$ (HIV-positiv) | $H=0$ (HIV-negativ) |

<!--en:97ba14ac-->
| Test positiv, $D_1 = 1$ | 1.00 | 0.01 |

<!--en:a2f58449-->
| Test negativ, $D_1 = 0$ | 0.00 | 0.99 |

<!--en:44608e23-->
Die Prävalenz in der Bevölkerung sei $P(H = 1) = 0.0015$ ($P(H = 0) = 0.9985$). Der Patient erhält ein positives Testergebnis ($D_1 = 1$). Durch Marginalisierung erhalten wir $P(D_1 = 1) = 0.01(0.9985) + 1(0.0015) = 0.011485$ und mit dem Satz von Bayes

<!--en:d4bea9ed-->
Unter $10{,}000$ Personen (:numref:`fig_prob_natural_freq`) erhalten alle $15$ Infizierten ein positives Testergebnis, aber auch $1\%$ der $9{,}985$ Gesunden ($\approx 100$ Fehlalarme). Unter den positiv Getesteten sind demnach nur $15 / 115 \approx 13\%$ tatsächlich infiziert.

<!--en:bb6b676b-->
![HIV-Test bei 10 000 Personen. Alle 15 Infizierten und rund 100 der 9985 Gesunden werden positiv getestet. Von etwa 115 positiven Ergebnissen sind also nur 15 richtig.](../img/probability-natural-frequencies.svg)

<!--en:d4a11661-->
Nehmen wir an, auch ein Bestätigungstest $D_2$ fällt positiv aus. Für ihn gelten $P(D_2 = 1 \mid H = 1) = 0.98$ und die Falsch-Positiv-Rate $P(D_2 = 1 \mid H = 0) = 0.03$. Sind $D_1$ und $D_2$ gegeben $H$ bedingt unabhängig, sinkt die gemeinsame Falsch-Positiv-Rate auf $0.01 \cdot 0.03 = 0.0003$. Die Sensitivität beträgt $1 \cdot 0.98 = 0.98$. Durch Marginalisierung erhalten wir $P(D_1 = 1, D_2 = 1) = 0.0003(0.9985) + 0.98(0.0015) = 0.00176955$. Mit dem Satz von Bayes steigt die A-posteriori-Wahrscheinlichkeit einer Infektion (:numref:`fig_prob_update`) auf

<!--en:c1784fa1-->
Simulieren wir zwei Millionen virtuelle Patienten, erhalten wir einen empirischen Anteil von etwa $0.82\text{--}0.83$. Er liegt innerhalb von zwei Standardfehlern zu je $\pm 0.006$ ($\sqrt{0.83 \cdot 0.17 / 3500}$) um den exakten Wert $0.8307$ für die $\approx 3{,}500$ Patienten mit zwei positiven Testergebnissen:

<!--en:d7015574-->
![Wahrscheinlichkeit einer HIV-Infektion beim getesteten Patienten: vor dem Test, nach einem positiven Ergebnis und nach zwei bedingt unabhängigen positiven Ergebnissen.](../img/probability-bayes-update.svg)

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:50d0cf2b-->
Die Wahrscheinlichkeitstheorie schließt von einem Modell darauf, wie wahrscheinlich die möglichen Ergebnisse sind. Die Statistik schließt umgekehrt aus beobachteten Daten auf Eigenschaften des zugrunde liegenden Prozesses. Für gemeinsame Verteilungen $P(X, Y)$ gelten die drei Axiome von Kolmogorov. Durch Summieren über Variablen erhalten wir Randverteilungen. Bei unabhängigen Variablen zerfällt die gemeinsame Verteilung in das Produkt $P(X)P(Y)$. Der Erwartungswert $\mathbb{E}[X]$ ist stets linear. Die Varianz $\textrm{Var}(X) = \mathbb{E}[(X-\mu)^2]$ addiert sich bei unabhängigen Variablen. Deshalb hat der Mittelwert aus $n$ unabhängigen Ziehungen den Standardfehler $\sigma/\sqrt{n}$. Messen wir die Abweichung in Standardfehlern, verstehen wir, warum $56/100$ Köpfe ($1.2$ Standardfehler) eine gewöhnliche Zufallsschwankung sind und $5{,}600/10{,}000$ ($12$ Standardfehler) ein eindeutiger Hinweis auf eine unfaire Münze ist. Mehr Daten verringern die *epistemische* Unsicherheit proportional zu $1/\sqrt{n}$. Die *aleatorische* Unsicherheit bleibt gleich. Mit dem Satz von Bayes $P(H \mid D) = P(D \mid H)P(H)/P(D)$ aktualisieren wir unsere ursprüngliche Einschätzung anhand der Beobachtungen. Bei einer seltenen Erkrankung mit $0.15\%$ Prävalenz und einem Test mit $1\%$ Falsch-Positiv-Rate ergibt ein positives Ergebnis eine A-posteriori-Wahrscheinlichkeit von nur $13.1\%$. Erst ein zweiter, bedingt unabhängiger positiver Test erhöht sie auf $83.1\%$.

<!--en:2294243a-->
## Übungen

<!--en:1d2fb6a6-->
1. **Rendite und Risiko eines Portfolios.** Sie verwalten ein Portfolio aus $n$ Aktien. Ihre Renditen $\mathbf{s} \in \mathbb{R}^n$ sind zufällig mit Mittelwert $\boldsymbol{\mu} = \mathbb{E}[\mathbf{s}]$ und Kovarianzmatrix $\boldsymbol{\Sigma}$. Sie investieren den Anteil $\alpha_i \geq 0$ Ihres Kapitals in Aktie $i$, wobei $\sum_i \alpha_i = 1$ gilt.
    1. Berechnen Sie den Erwartungswert und die Varianz der Rendite des Portfolios $\boldsymbol{\alpha}$.
    1. Jede Aktie habe die Varianz $\sigma^2$, und je zwei verschiedene Aktien haben die Korrelation $\rho \geq 0$. Berechnen Sie die Varianz des gleichgewichteten Portfolios $\alpha_i = 1/n$ und ihren Grenzwert für $n \to \infty$. Vergleichen Sie den Fall $\rho = 0$ mit :eqref:`eq_prob_var_phat`.
    1. (Optional.) Formulieren Sie ein Optimierungsproblem, das die erwartete Rendite maximiert und dabei eine Obergrenze für die Varianz einhält. Dies ist das Problem der [Portfoliooptimierung nach Markowitz](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`Markowitz.1952`. In :numref:`sec_mdl-constrained-optimization-duality` entwickeln wir Methoden für solche Probleme mit Nebenbedingungen.

<!--en:e424b986-->
1. [code] **Chebyshev-Ungleichung für den Schätzer beim Münzwurf.** :eqref:`eq_prob_var_phat` gibt die Varianz der relativen Häufigkeit $\hat{p}$ von Kopf nach $n$ Würfen an.
    1. Bestimmen Sie für eine faire Münze und $n = 100$ mit der Chebyshev-Ungleichung eine obere Schranke für $P(|\hat{p} - p| \geq 0.1)$.
    1. Schätzen Sie dieselbe Wahrscheinlichkeit, indem Sie 10 000 Versuche mit je 100 Würfen simulieren. Vergleichen Sie den Schätzwert mit der Schranke und erklären Sie den Unterschied.
    1. Bestimmen Sie, wie viele Würfe die Chebyshev-Ungleichung bei einer fairen Münze erfordert, um $P(|\hat{p} - p| \geq 0.01) \leq 0.05$ zu garantieren. Vergleichen Sie die Anzahl mit dem Wert von $n$, bei dem der Standardfehler $\sqrt{p(1-p)/n}$ selbst gleich $0.01$ ist.

<!--en:822f8189-->
1. **Vergleich zweier Klassifikatoren.** ● Wir werten zwei Klassifikatoren auf demselben Testdatensatz mit $n = 10{,}000$ Bildern aus. Klassifikator A ordnet 9050 Bildern das richtige Label zu, Klassifikator B 9000 Bildern. Behandeln Sie die Testbilder als unabhängige Ziehungen.
    1. Berechnen Sie den Standardfehler jeder der beiden Genauigkeiten.
    1. Nehmen Sie an, die beiden Genauigkeiten seien unabhängig, und berechnen Sie den Standardfehler ihrer Differenz. Ist die Differenz von $0.005$ im Vergleich zum Rauschen groß?
    1. Wir messen die beiden Genauigkeiten an denselben Bildern, daher sind sie nicht unabhängig. Angenommen, die Klassifikatoren kommen bei 300 Bildern zu unterschiedlichen Ergebnissen. Bei 175 davon liegt A richtig, bei 125 liegt B richtig. Setzen Sie für Bild $i$ den Wert $Z_i = 1$, wenn nur A richtig liegt, $Z_i = -1$, wenn nur B richtig liegt, und sonst $Z_i = 0$. Dann ist die Differenz der Genauigkeiten der Mittelwert der $Z_i$. Berechnen Sie den Standardfehler dieses Mittelwerts und vergleichen Sie ihn mit Ihrem Ergebnis aus der vorigen Teilaufgabe.
    1. Erklären Sie den Unterschied mithilfe von $\textrm{Var}(X - Y) = \textrm{Var}(X) + \textrm{Var}(Y) - 2\,\textrm{Cov}(X, Y)$. Welches Vorzeichen hat die Kovarianz zwischen den Indikatorvariablen, die für jeden Klassifikator angeben, ob er bei einem Bild richtig liegt? Warum?

<!--en:8c8c6158-->
1. **Faktorisierung einer Kette.** Es seien $X$, $Y$ und $Z$ Zufallsvariablen. Dabei ist $Z$ bedingt unabhängig von $X$ bei gegebenem $Y$, wie in der Kette in :numref:`fig_prob_explaining_away`.
    1. Schreiben Sie die gemeinsame Wahrscheinlichkeit $P(X, Y, Z)$ als Produkt von Faktoren, die jeweils höchstens zwei der Variablen enthalten.
    1. Verallgemeinern Sie das Ergebnis auf eine Folge $X_1, \ldots, X_T$, in der jedes $X_t$ nur über $X_{t-1}$ von den früheren Variablen abhängt. Wie viele Zahlen legen die gemeinsame Verteilung fest, wenn jedes $X_t$ genau $k$ Werte annimmt? Vergleichen Sie mit einer beliebigen gemeinsamen Verteilung von $T$ solchen Variablen.

<!--en:3d5abd85-->
1. **Bedingen auf eine gemeinsame Wirkung.** Werfen Sie zwei unabhängige faire Münzen. Die Indikatorvariablen $X, Y \in \{0, 1\}$ geben an, ob die jeweilige Münze Kopf zeigt. Die Variable $Z = \max(X, Y)$ gibt an, ob mindestens eine Münze Kopf zeigt. Berechnen Sie $P(X = 1)$, $P(X = 1 \mid Z = 1)$ und $P(X = 1 \mid Z = 1, Y = 1)$. Entscheiden Sie anhand dieser Werte, ob $X$ und $Y$ gegeben $Z = 1$ bedingt unabhängig sind. Erklären Sie den Unterschied zwischen dem zweiten und dritten Wert anhand von :numref:`fig_prob_explaining_away`.

<!--en:1f0dc65d-->
1. **Ein zweiter, korrelierter Test.** ● Betrachten Sie :numref:`subsec_probability_hiv_app` erneut, diesmal mit zwei neuen Tests. Ihre Ergebnisse sind bei Patienten mit HIV bedingt unabhängig, bei gesunden Patienten hingegen abhängig. Jeder Test für sich hat eine Falsch-Positiv-Rate von 10% und eine Falsch-Negativ-Rate von 1%:

   $$P(D_i = 1 \mid H = 0) = 0.1, \quad P(D_i = 0 \mid H = 1) = 0.01 \quad \textrm{for } i = 1, 2.$$

   Für $H = 1$ zerfällt die gemeinsame Verteilung in das Produkt $P(D_1, D_2 \mid H = 1) = P(D_1 \mid H = 1)\, P(D_2 \mid H = 1)$. Für $H = 0$ hängen die Ergebnisse gemäß $P(D_1 = D_2 = 1 \mid H = 0) = 0.02$ zusammen. Die Ausgangswahrscheinlichkeit beträgt wie zuvor $P(H = 1) = 0.0015$.
    1. Erstellen Sie die Tabelle der gemeinsamen Wahrscheinlichkeiten für $D_1$ und $D_2$ bei gegebenem $H = 0$.
    1. Leiten Sie die Wahrscheinlichkeit her, dass der Patient HIV hat, nachdem der erste Test positiv ausgefallen ist und bevor der zweite Test stattfindet.
    1. Leiten Sie die Wahrscheinlichkeit her, dass der Patient HIV hat, nachdem beide Tests positiv ausgefallen sind.
    1. Wiederholen Sie die letzte Rechnung für Tests, die auch bei gegebenem $H = 0$ bedingt unabhängig sind. Erklären Sie, warum sich die beiden A-posteriori-Wahrscheinlichkeiten unterscheiden.

<!--en:153ec9f7-->
1. **Die Statistik eines Verteidigers.** Die Ehefrau eines Verdächtigen wurde ermordet. Es ist belegt, dass er sie zuvor misshandelt hatte. Sein Verteidiger argumentiert, statistisch ermorde nur einer von tausend Männern, die ihre Frau misshandeln, später seine Frau. Die früheren Misshandlungen seien also nur ein schwaches Indiz, und die Geschworenen sollten den Angeklagten freisprechen. Schreiben Sie $M$ für „Der Verdächtige hat seine Frau ermordet“, $K$ für „Seine Frau wurde getötet“ und $B$ für „Er hatte sie zuvor misshandelt“.
    1. Drücken Sie die Angabe „einer von tausend“ und die A-posteriori-Wahrscheinlichkeit der Schuld mit diesen Ereignissen aus. Welche Wahrscheinlichkeit hat der Verteidiger mit welcher verwechselt?
    1. Nehmen Sie an, dass eine misshandelte Ehefrau in dem Zeitraum, auf den sich die Statistik des Verteidigers bezieht, mit Wahrscheinlichkeit $10^{-5}$ von einer anderen Person als ihrem Ehemann ermordet wird. Berechnen Sie die A-posteriori-Wahrscheinlichkeit der Schuld und vergleichen Sie sie mit der Angabe des Verteidigers.

    *Nach MIT 18.05, [Übungsblatt 7](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_pset07_sol.pdf), Aufgabe 4.*

<!--en:28f298cb-->
[Diskussion](https://d2l.discourse.group/t/36)

<!--en:f725f385-->
[Diskussion](https://d2l.discourse.group/t/37)

<!--en:52633beb-->
[Diskussion](https://d2l.discourse.group/t/198)

<!--en:11272efa-->
[Diskussion](https://d2l.discourse.group/t/17971)

<!--en:f5de5028-->
Wahrscheinlichkeit und Statistik: Schlüsse unter Unsicherheit beim maschinellen Lernen<br>**Aleatorische und epistemische Unsicherheit · Verteilungen und Bedingen · Satz von Bayes · Erwartungswert und Kovarianz · Fehlerbalken der Länge $\sigma/\sqrt{n}$**

<!--en:27d2cbe9-->
::: {.slide title="Warum Wahrscheinlichkeit und Statistik die Sprache des maschinellen Lernens sind"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:92c3aaaf-->
- **Kalibrierte Vorhersagen und Verluste:** Die Ausgaben $P(Y \mid \mathbf{x})$ der Softmax-Funktion quantifizieren das Vorhersagerisiko. Kreuzentropie und mittlerer quadratischer Fehler (MSE) folgen direkt aus der Maximum-Likelihood-Schätzung anhand von $P(\mathcal{D} \mid \mathbf{w})$.

<!--en:25686f2a-->
- **Monte-Carlo-Training und -Auswertung:** Minibatch-SGD schätzt $\nabla \mathbb{E}[\ell]$ aus $B = 256$ Beispielen. Standardfehler $\sigma/\sqrt{n}$ trennen echte Verbesserungen von Zufallsschwankungen im Testdatensatz.

<!--en:6eefb9d8-->
- **Dualität der Schlussrichtungen:** Die Wahrscheinlichkeitstheorie sagt bei bekanntem $\boldsymbol{\theta}$ Daten vorher. Die Statistik bestimmt Schätzwerte $\hat{\boldsymbol{\theta}}$ und Fehlerbalken aus Daten $\mathcal{D}$.

<!--en:9e26fdc0-->
::: {.slide title="Zwei Quellen der Unsicherheit: aleatorische und epistemische Unsicherheit"}

<!--en:c5327de9-->
[Arten der Unsicherheit]{.kicker}

<!--en:724ae8fe-->
- **Aleatorische Unsicherheit der Daten – *nicht reduzierbar*:** Zufall oder Mehrdeutigkeit in der Beobachtung selbst.
  - Auch bei einer bekanntermaßen fairen Münze ($p = 0.5$) stehen die Chancen beim nächsten Wurf 50/50.
  - Ein unscharfes $28 \times 28$-Bild aus Fashion-MNIST, das Menschen im Verhältnis 60/40 als *Pullover* oder *Mantel* einstufen, bleibt mehrdeutig, unabhängig von der Zahl der Trainingsbilder.

<!--en:f43613bf-->
- **Epistemische Unsicherheit des Modells – *reduzierbar*:** Die wahren Parameter ($p$ oder die Gewichte $\mathbf{w}$) sind wegen des endlichen Stichprobenumfangs $n$ nur unvollständig bekannt.
  - Mit mehr Daten sinkt diese Unsicherheit proportional zu **$1/\sqrt{n}$**.

<!--en:262cf78b-->
::: {.slide title="Sampling und relative Häufigkeiten"}

<!--en:80dfa118-->
[Vom Modell zu den Daten]{.kicker}

<!--en:c1c41b79-->
Das Ziehen von Realisierungen aus einer Verteilung $P$ heißt **Sampling**. Bei $n$ unabhängigen Würfen einer fairen Münze mit $P = (0.5, 0.5)$ folgt der Vektor der Anzahlen einer **Multinomialverteilung**:

<!--en:5c4c0934-->
- Division der beobachteten Anzahlen durch $n$ ergibt die relative **Häufigkeit** $\hat{P}_n = \textrm{counts}/n$ als Schätzer für die wahre Wahrscheinlichkeit $P = (0.5, 0.5)$.

<!--en:ebc6bbfe-->
- Bei einem Versuch mit $100$ Würfen weichen die Anzahlen oft um mehrere Köpfe von $(50, 50)$ ab:

<!--en:b95b7f4e-->
- Bei fünf unabhängigen Versuchen mit je $100$ Würfen schwankt $\hat{p}_n$ merklich um $0.50$. Bei $n = 10{,}000$ schrumpfen die Schwankungen um den Faktor zehn auf $\approx \pm 0.005$:

<!--en:2419d8f6-->
::: {.slide title="Das Gesetz der großen Zahlen: Konvergenz der relativen Häufigkeiten"}

<!--en:6acdd8cf-->
[Asymptotische Konsistenz]{.kicker}

<!--en:4853ed3d-->
- Die relative Häufigkeit $\hat{p}_n = \frac{1}{n}\sum_{i=1}^n X_i$ nach jedem Wurf $n = 1, 2, \ldots, 10{,}000$ verfolgen. Die Berechnung mit `cumsum` veranschaulicht das **Gesetz der großen Zahlen (LLN)**:
  $$\hat{p}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{n \to \infty} p = 0.5$$

<!--en:879ae180-->
- Anfangs, bei $n < 100$, schwankt die Kurve stark. Für $n \to 10{,}000$ nähert sie sich $P(\textrm{heads}) = 0.5$ an.

<!--en:b6b320fb-->
**Die offene Frage:** Das Gesetz der großen Zahlen garantiert die Konvergenz *im Grenzwert* $n \to \infty$. Es sagt jedoch nichts darüber aus, **wie schnell** der Schätzfehler bei endlichem $n$ sinkt.

<!--en:7cbca09a-->
::: {.slide title="Wie schnell konvergieren Schätzwerte? Die 1/√n-Hüllkurve"}

<!--en:f733ea99-->
[Konvergenzgeschwindigkeit]{.kicker}

<!--en:d3f26f29-->
- Die Simulation von $1{,}000$ Versuchen für jeden Stichprobenumfang $n \in \{10, 100, 1{,}000, 10{,}000\}$ ergibt im doppelt logarithmischen Diagramm eine Gerade mit **Steigung $-1/2$**:
  $$\textrm{std}(\hat{p}_n) = \frac{0.5}{\sqrt{n}}$$

<!--en:b8a497c7-->
- **Der Aufwand für höhere Präzision wächst quadratisch:**
  - Für einen um den Faktor **$2\times$** kleineren Fehler braucht es **$4\times$** so viele Beispiele zur Auswertung.
  - Für einen um den Faktor **$10\times$** kleineren Fehler braucht es **$100\times$** so viele Beispiele ($n = 100 \to \pm 5\%$ gegenüber $n = 10{,}000 \to \pm 0.5\%$).

<!--en:50db443f-->
- Die Herleitung dieses $1/\sqrt{n}$-Gesetzes aus den Grundlagen erfordert Definitionen für Ereignisse, Zufallsvariablen, Erwartungswert und Varianz.

<!--en:d5ba0483-->
::: {.slide title="Ergebnisräume, Ereignisse und die Axiome von Kolmogorov"}

<!--en:9fb9ca65-->
[Grundlagen der Wahrscheinlichkeitstheorie]{.kicker}

<!--en:3751a251-->
- Der **Ergebnisraum** $\mathcal{S}$ eines Zufallsexperiments enthält alle möglichen Ergebnisse. Jede Teilmenge $\mathcal{A} \subseteq \mathcal{S}$ ist ein **Ereignis**.

<!--en:4b83eaae-->
- **Die drei Axiome von Kolmogorov** (1933):
  1. **Nichtnegativität:** $P(\mathcal{A}) \ge 0$ für jedes $\mathcal{A} \subseteq \mathcal{S}$.
  2. **Normierung:** $P(\mathcal{S}) = 1$ (und $P(\emptyset) = 0$).
  3. **Additivität:** Für paarweise disjunkte Ereignisse ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$) gilt $P(\bigcup_i \mathcal{A}_i) = \sum_i P(\mathcal{A}_i)$.

<!--en:fc2c3744-->
- **Gegenwahrscheinlichkeit ($P(\mathcal{A}^\complement) = 1 - P(\mathcal{A})$) und Prinzip der Inklusion und Exklusion:**
  $$P(\mathcal{A} \cup \mathcal{B}) = P(\mathcal{A}) + P(\mathcal{B}) - P(\mathcal{A} \cap \mathcal{B})$$

<!--en:a7d6022d-->
*Würfelwurf ($\mathcal{S} = \{1,\ldots,6\}$):* Für $\mathcal{A} = \{2, 4, 6\}$ (gerade) und $\mathcal{B} = \{5, 6\}$ ($\ge 5$) gilt $\mathcal{A} \cap \mathcal{B} = \{6\}$, also $P(\mathcal{A} \cup \mathcal{B}) = \frac{3}{6} + \frac{2}{6} - \frac{1}{6} = \frac{2}{3}$.

<!--en:06c4d216-->
::: {.slide title="Diskrete und stetige Zufallsvariablen"}

<!--en:300af185-->
[Zufallsvariablen und Verteilungen]{.kicker}

<!--en:f4780548-->
- Eine **Zufallsvariable** $X: \mathcal{S} \to \mathbb{R}$ ordnet Ergebnissen Zahlen zu (z. B. den $2^{100}$ möglichen Münzwurffolgen die Anzahl $\{0, \ldots, 100\}$ der Köpfe).

<!--en:9d9819b7-->
- **Diskret (Wahrscheinlichkeitsfunktion):** Jedem einzelnen Wert kommt eine Masse $P(X = v) \ge 0$ zu, mit $\sum_v P(X = v) = 1$ (z. B. bei der Softmax-Funktion über Klassen oder Token des Vokabulars).

<!--en:8f3cbe73-->
- **Stetig (Wahrscheinlichkeitsdichte):** Bei einer Dichte $p(x) \ge 0$ entspricht die Wahrscheinlichkeit der **Fläche unter der Kurve**:
  $$P(a \le X \le b) = \int_a^b p(x)\,dx, \quad \int_{-\infty}^{\infty} p(x)\,dx = 1$$
  Hinweis: $p(x)$ kann größer als $1$ sein, während an jedem einzelnen Punkt $P(X = x) = 0$ gilt.

<!--en:8ffd3b4f-->
::: {.slide title="Gemeinsame Wahrscheinlichkeit, Randwahrscheinlichkeit und bedingte Wahrscheinlichkeit"}

<!--en:ab69b515-->
[Summenregel und Produktregel]{.kicker}

<!--en:8b275ef2-->
- **Gemeinsame Verteilung $P(X = x, Y = y)$:** Wahrscheinlichkeit, dass $X = x$ und $Y = y$ zusammen eintreten (zweidimensionale Tabelle mit Summe $1$ über alle Einträge).

<!--en:50f56265-->
- **Summenregel (Marginalisierung):** Über Störvariablen oder latente Variablen entlang einer Achse summieren (`sum(axis=1)`):
  $$P(X = x) = \sum_y P(X = x, Y = y)$$

<!--en:f8e831c3-->
- **Bedingen und Produktregel (Kettenregel):** Auf die Zeile $X = x$ beschränken und durch Division durch $P(X = x)$ neu normieren:
  $$P(Y \mid X) = \frac{P(X, Y)}{P(X)} \iff P(X, Y) = P(Y \mid X)\,P(X)$$
  Wiederholtes Anwenden der Produktregel ergibt die Faktorisierung autoregressiver Sprachmodelle: $P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t})$.

<!--en:1c01486f-->
::: {.slide title="Satz von Bayes: die Richtung der Bedingung umkehren"}

<!--en:8877c5c6-->
[Inverse Wahrscheinlichkeit]{.kicker}

<!--en:0434ccb2-->
- Die gemeinsame Verteilung in beiden Reihenfolgen schreiben: $P(H, D) = P(D \mid H)P(H) = P(H \mid D)P(D)$. Division durch $P(D)$ ergibt den **Satz von Bayes**:
  $$P(H \mid D) = \frac{P(D \mid H)\,P(H)}{P(D)} = \frac{\textrm{likelihood} \times \textrm{prior}}{\textrm{evidence}}$$
  Dabei normiert $P(D) = \sum_h P(D \mid H = h)\,P(H = h)$ die A-posteriori-Verteilung über alle Hypothesen.

<!--en:08f5a2e4-->
- **Vorwärtsmodell $\to$ Rückschluss:** Generative Modelle beschreiben, wie verborgene Ursachen oder Label $H$ beobachtete Merkmale oder Testergebnisse $D$ hervorbringen ($P(D \mid H)$). Der Satz von Bayes kehrt die Schlussrichtung um und liefert $P(H \mid D)$.

<!--en:8ec28ab9-->
::: {.slide title="Abhängigkeit, Unabhängigkeit und Wegerklären"}

<!--en:0f393ff5-->
[Graphenstruktur und Bedingen]{.kicker}

<!--en:ad9bbd54-->
- **Marginale Unabhängigkeit ($X \perp Y$):** $P(X, Y) = P(X)P(Y)$.

<!--en:1df2d13d-->
- **Bedingte Unabhängigkeit ($X \perp Y \mid Z$):** $P(X, Y \mid Z) = P(X \mid Z)P(Y \mid Z)$. Bedingen kann Abhängigkeit *aufheben* oder *erzeugen*:
  1. **Gemeinsame Ursache ($X \leftarrow Z \to Y$):** Die Variablen sind marginal abhängig, aber unabhängig, sobald die gemeinsame Ursache $Z$ bekannt ist.
  2. **Markov-Kette ($X \to Y \to Z$):** Vergangenheit $X \perp$ Zukunft $Z$ bei gegebener Gegenwart $Y$.
  3. **Gemeinsame Wirkung ($X \to Z \leftarrow Y$, *Wegerklären*):** Unabhängige Ursachen (Regen $\perp$ Rasensprenger) werden **negativ abhängig**, sobald ihre gemeinsame Wirkung $Z = 1$ (nasses Gras) bekannt ist.

<!--en:9f73dcd8-->
::: {.slide title="Erwartungswert: Schwerpunkt und erwartungstreue Schätzer"}

<!--en:45e66ad7-->
[Erstes Moment]{.kicker}

<!--en:02e2ca85-->
- Der **Erwartungswert** (Mittelwert $\mu = \mathbb{E}[X]$) ist der mit den Wahrscheinlichkeiten gewichtete Schwerpunkt:
  $$\mathbb{E}[X] = \sum_x x\,P(X = x), \qquad \mathbb{E}[f(X)] = \sum_x f(x)\,P(X = x)$$

<!--en:c42211d9-->
- *Beispiele:* Erwarteter Auszahlungsfaktor der Anlage: $0.5(0) + 0.4(2) + 0.1(10) = 1.8$. Für eine Bernoulli-Indikatorvariable $X \in \{0, 1\}$ gilt $\mathbb{E}[X] = p$.

<!--en:b5510b37-->
- **Linearität des Erwartungswerts (gilt auch bei abhängigen $X, Y$):**
  $$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$$
  Damit gilt $\mathbb{E}[\hat{p}_n] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i] = p$. Relative Häufigkeiten und die Gradienten bei Minibatch-SGD sind **erwartungstreue** Schätzer.

<!--en:c3c77644-->
::: {.slide title="Varianz und Standardabweichung als Streuungsmaße"}

<!--en:8269df49-->
[Zweites zentrales Moment]{.kicker}

<!--en:65d4ecb1-->
- Die **Varianz** ist die erwartete quadratische Abweichung von $\mu = \mathbb{E}[X]$. Die **Standardabweichung** $\sigma = \sqrt{\mathrm{Var}(X)}$ hat dieselbe Maßeinheit wie $X$:
  $$\mathrm{Var}(X) = \mathbb{E}\!\left[(X - \mu)^2\right] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

<!--en:f5e046ac-->
- **Verschiebungsinvarianz und quadratische Skalierung:** $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$.

<!--en:cda4da9c-->
- *Auszahlung der Anlage:* $\mathbb{E}[X^2] = 0.5(0) + 0.4(4) + 0.1(100) = 11.6$, also $\mathrm{Var}(X) = 11.6 - 1.8^2 = 8.36$ ($\sigma \approx 2.9$).

<!--en:aec83d02-->
- *Bernoulli-Indikatorvariable $X \in \{0, 1\}$:* Wegen $X^2 = X$ gilt $\mathrm{Var}(X) = p - p^2 = p(1 - p) \le \frac{1}{4}$ (Maximum bei $p = 0.5$).

<!--en:7e8bdda9-->
::: {.slide title="Kovarianz, Korrelation und Varianz eines Mittelwerts"}

<!--en:15c0aab1-->
[Linearer Zusammenhang und 1/√n-Skalierung]{.kicker}

<!--en:0341ff07-->
- **Kovarianz und Korrelation** ($\rho = \frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$) messen den linearen Zusammenhang:
  $$\mathrm{Cov}(X, Y) = \mathbb{E}\!\left[(X - \mu_X)(Y - \mu_Y)\right]$$

<!--en:ecb28b4e-->
- **Varianz einer Summe:** Ausmultiplizieren des Quadrats in $\mathbb{E}[(X - \mu_X + Y - \mu_Y)^2]$ ergibt
  $$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)$$

<!--en:533580da-->
- **Warum die Varianz beim Mitteln um den Faktor $n$ sinkt:** Bei $n$ unabhängigen Ziehungen ($\mathrm{Cov} = 0$) addieren sich die Varianzen ($\mathrm{Var}(\sum_{i=1}^n X_i) = n\sigma^2$). Division der Summe durch $n$ skaliert die Varianz mit $1/n^2$:
  $$\mathrm{Var}(\bar{X}_n) = \frac{\sigma^2}{n} \implies \mathrm{SE}(\bar{X}_n) = \frac{\sigma}{\sqrt{n}}$$

<!--en:1865614f-->
::: {.slide title="Chebyshev-Ungleichung, zentraler Grenzwertsatz und Test auf eine unfaire Münze"}

<!--en:0812d2b4-->
[Statistische Signifikanz]{.kicker}

<!--en:08c4e4c4-->
- Für eine faire Münze ($p = 0.5$) gilt $\mathrm{SE}(\hat{p}_n) = \sqrt{p(1-p)/n} = 0.5/\sqrt{n}$. Vergleich von $\hat{p} = 0.56$ bei zwei Stichprobenumfängen:
  - **$56/100$ Würfe mit Kopf ($\mathrm{SE} = 0.05$):** Die Abweichung $0.06$ entspricht **$1.2\,\mathrm{SE}$** ($p\text{-value} \approx 0.27$, gewöhnliche Zufallsschwankung).
  - **$5{,}600/10{,}000$ Würfe mit Kopf ($\mathrm{SE} = 0.005$):** Die Abweichung $0.06$ entspricht **$12\,\mathrm{SE}$** ($p\text{-value} \approx 0$, eindeutiger Hinweis auf eine unfaire Münze).

<!--en:7b048ff4-->
- **Chebyshev-Ungleichung (jede Verteilung mit endlicher Varianz):** $P(|X - \mu| \ge k\sigma) \le 1/k^2$ (für $k=2$ beträgt die Wahrscheinlichkeit einer solchen Abweichung $\le 25\%$).

<!--en:a8c62a3f-->
- **Zentraler Grenzwertsatz (näherungsweise gaußsche Verteilung der Mittelwerte):** $\bar{X}_n$ liegt innerhalb von $\pm 1\sigma / \pm 2\sigma / \pm 3\sigma$ um den Erwartungswert mit Wahrscheinlichkeit **$68\% / 95\% / 99.7\%$** ($95\%$-Konfidenzintervall: $\hat{p} \pm 1.96\,\mathrm{SE}$).

<!--en:cb8f37b3-->
::: {.slide title="Klassifikatoren auswerten: Minibatch-Rauschen und gepaarte Vergleiche"}

<!--en:31e88643-->
[Modellauswertung und Fehlerbalken]{.kicker}

<!--en:958e0905-->
Ob eine Vorhersage im Test richtig ist, ist bernoulliverteilt. Für die geschätzte Genauigkeit gilt $\mathrm{SE}(\hat{p}_n) = \sqrt{p(1-p)/n}$:

<!--en:bd98f10b-->
- **Minibatch ($n = 256$) und Testdatensatz ($n = 10{,}000$) bei $p = 0.90$:** Beim Minibatch gilt $\mathrm{SE} \approx \mathbf{1.9\%}$ (ein Rückgang um $3\%$ ist gewöhnliches Rauschen), beim Testdatensatz $\mathrm{SE} = \mathbf{0.30\%}$.

<!--en:365f0b25-->
- **Vorteil der gepaarten Auswertung auf *demselben* Testdatensatz:** Bei $10{,}000$ gemeinsamen Bildern fallen $9{,}700$ Übereinstimmungen in der Differenz weg ($\mathrm{Cov} > 0$). Damit sinkt $\mathrm{SE}(\hat{p}_A - \hat{p}_B)$ von $0.42\%$ auf **$\mathbf{0.17\%}$** (Verbesserung um $+0.5\%$ $= 2.9\,\mathrm{SE}$).

<!--en:247dd6d0-->
::: {.slide title="Rechenbeispiel: Vernachlässigung der Basisrate bei Diagnostik und Anomalieerkennung"}

<!--en:c1a9134c-->
[Der Satz von Bayes in der Praxis]{.kicker}

<!--en:c56e8af6-->
- Ein diagnostischer Test $D_1 \in \{0, 1\}$ auf den HIV-Status $H \in \{0, 1\}$ hat **$100\%$ Sensitivität** ($P(D_1{=}1 \mid H{=}1) = 1.00$) und **$99\%$ Spezifität** ($P(D_1{=}1 \mid H{=}0) = 0.01$). Die Prävalenz beträgt $P(H{=}1) = 0.0015$.

<!--en:63dd198d-->
- Nach der Summenregel gilt $P(D_1{=}1) = 1.00(0.0015) + 0.01(0.9985)$ $= 0.011485$. Der Satz von Bayes ergibt:
  $$P(H{=}1 \mid D_1{=}1) = \frac{0.0015}{0.011485} = \mathbf{0.1306} \;(13.1\%)$$

<!--en:d89a9fc6-->
- **Warum so niedrig?** Unter $10{,}000$ Personen werden alle $15$ Infizierten positiv getestet. Hinzu kommen $1\%$ der $9{,}985$ Gesunden, also **$\approx 100$ Fehlalarme** ($15 / 115 \approx 13.1\%$).

<!--en:9abbc32a-->
::: {.slide title="Schrittweise Aktualisierung mit dem Satz von Bayes und korrelierte Beobachtungen"}

<!--en:0da9f409-->
[Zweistufiger Suchtest und Monte-Carlo-Simulation]{.kicker}

<!--en:8f39ef7a-->
- **Bestätigungstest $D_2$** (Sensitivität $0.98$, Falsch-Positiv-Rate $0.03$): Bei bedingter Unabhängigkeit $D_1 \perp D_2 \mid H$ sinkt die gemeinsame Falsch-Positiv-Rate auf $0.01 \times 0.03 = 0.0003$:
  $$P(H{=}1 \mid D_1{=}1, D_2{=}1) = \frac{0.00147}{0.00147 + 0.00030} = \mathbf{0.8307}$$

<!--en:1d50146a-->
- **Korrelierte Fehler:** Hängt ein falsch positives Ergebnis bei beiden Tests von denselben biochemischen Faktoren ab ($P(D_1{=}1, D_2{=}1 \mid H{=}0) > 0.0003$), häufen sich Fehlalarme. Die A-posteriori-Wahrscheinlichkeit sinkt (z. B. auf $32.9\%$, wenn sich die gemeinsame Falsch-Positiv-Rate auf $0.003$ verzehnfacht).

<!--en:84c48cec-->
::: {.slide title="Verständnisfragen: Standardfehler und Aktualisierung mit dem Satz von Bayes"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:9268f8b6-->
Drei praktische Aufgaben aus dem maschinellen Lernen zur Prüfung der eigenen Intuition:

<!--en:0c168fc2-->
1. **Größe eines Testdatensatzes:** Ein Klassifikator hat die Genauigkeit $p \approx 0.90$. Wie viele Testbeispiele $n$ sind nötig für einen Standardfehler von $\mathrm{SE}(\hat{p}_n) = 0.005$ ($0.5\%$)? Wie viele für $\mathrm{SE} = 0.0025$?

<!--en:015aa304-->
2. **Wegerklären:** Beim Wurf zweier unabhängiger fairer Münzen $X, Y \in \{0, 1\}$ gilt $Z = \max(X, Y) = 1$ (mindestens einmal Kopf). Wie groß ist $P(X = 1 \mid Z = 1)$ und wie ändert sich diese Wahrscheinlichkeit, wenn anschließend $Y = 1$ bekannt wird?

<!--en:b2c31618-->
3. **Geänderte A-priori-Wahrscheinlichkeit:** Eine Klinik für Risikopatienten verwendet den ersten HIV-Test ($100\%$ Sensitivität, $1\%$ Falsch-Positiv-Rate). Die Prävalenz beträgt dort $P(H = 1) = 0.01$ ($1\%$) statt $0.15\%$. Wie groß ist $P(H = 1 \mid D_1 = 1)$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:dc4b901c-->
1. $\sqrt{0.9 \times 0.1 / n} = 0.005 \implies n = 0.09 / 0.000025 = \mathbf{3{,}600}$ Bilder. Die Halbierung von $\mathrm{SE}$ auf $0.0025$ erfordert $4\times$ so viele Daten, also **$\mathbf{14{,}400}$** Bilder.

<!--en:555b5988-->
2. $P(X = 1 \mid Z = 1) = \mathbf{2/3}$ (unter den Möglichkeiten $\{(1,0), (0,1), (1,1)\}$). Die Beobachtung $Y = 1$ reicht aus, um $Z = 1$ *wegzuerklären*. Daher sinkt $P(X = 1 \mid Z = 1, Y = 1)$ wieder auf **$\mathbf{1/2}$**.

<!--en:0b39d077-->
3. $P(H = 1 \mid D_1 = 1) = \frac{1.00 \times 0.01}{1.00(0.01) + 0.01(0.99)} = \frac{0.01}{0.0199} \approx \mathbf{50.3\%}$ (richtig positive und falsch positive Ergebnisse stehen nun etwa im Verhältnis $1{:}1$).

<!--en:97369ca1-->
::: {.slide title="Zusammenfassung: Werkzeuge der Wahrscheinlichkeitstheorie und ihre Rolle im Deep Learning"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:0df08f04-->
| Begriff oder Regel | Mathematische Form | Skalierung oder Verhalten | Hauptanwendung im Deep Learning |

<!--en:cbf6a3dc-->
| **Marginalisierung (Summenregel)** | $P(X) = \sum_y P(X, Y)$ | Summiert über eine Achse (`sum(axis=...)`) | Mitteln über latente Variablen und Ensembles |

<!--en:2df3342e-->
| **Kettenregel (Produktregel)** | $P(X, Y) = P(Y \mid X)\,P(X)$ | Exakte Faktorisierung jeder gemeinsamen Verteilung $P(\mathbf{x})$ | Autoregressive LLMs $\prod_t P(x_t \mid x_{<t})$ |

<!--en:e5e74da6-->
| **Satz von Bayes** | $P(H \mid D) = \frac{P(D \mid H)P(H)}{P(D)}$ | A-posteriori-Verteilung $\propto$ Likelihood $\times$ A-priori-Verteilung | MAP-Schätzung, Anomalieerkennung und medizinische Suchtests |

<!--en:d42c3d5d-->
| **Linearität des Erwartungswerts** | $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ | Gilt auch für abhängige $X, Y$ | Erwartungstreue Gradientenschätzung bei Minibatch-SGD |

<!--en:ccdf4e8f-->
| **Standardfehler eines Mittelwerts** | $\mathrm{SE}(\bar{X}_n) = \sigma / \sqrt{n}$ | $4\times$ so viele Daten $\to \frac{1}{2}$ des Fehlers, $100\times$ so viele Daten $\to \frac{1}{10}$ des Fehlers | Größe des Testdatensatzes und gepaarte Modellvergleiche |

<!--en:47b5876e-->
| **Aleatorische und epistemische Unsicherheit** | Datenrauschen und Unkenntnis der Parameter | Aleatorisch: nicht reduzierbar. Epistemisch: $\propto 1/\sqrt{n}$ | Kalibrierung der Softmax-Funktion und aktive Datenerhebung |
