<!--en:c8fb8f93-->
# Bayessche Rechenverfahren

<!--en:dd8b4548-->
Im Grundlagenteil dieses Buches haben wir für jedes Modell beim Training einen einzigen Parametervektor bestimmt. Dazu haben wir entweder Maximum-Likelihood verwendet oder mit einem Regularisierungsterm die Maximum-a-posteriori-Schätzung (engl. maximum a posteriori, MAP) am Maximum der A-posteriori-Verteilung (engl. posterior distribution) berechnet. In :numref:`sec_mdl-maximum_likelihood` sind wir einen Schritt weiter gegangen und haben diese Schätzung mit einem Fehlerbalken versehen. Doch auch dann beruhten unsere Vorhersagen auf diesem einen Vektor, obwohl viele andere Parametervektoren die Trainingsdaten fast ebenso gut erklären. Wie stark unterscheiden sich die Vorhersagen dieser plausiblen Modelle, und was sollten wir vorhersagen, wenn sie voneinander abweichen?

<!--en:a5eeb3dc-->
Betrachten wir eine logistische Regression mit Achsenabschnitt und Steigung. Wir passen sie an zwanzig Punkte mit bekannten Labels an, deren Eingaben zwischen $0$ und $4$ liegen. Bei einer Testeingabe von $-3$, weit unterhalb der Trainingsdaten, ergibt die angepasste MAP-Gerade für das Label $1$ eine Wahrscheinlichkeit von etwa vier zu tausend. Andere Parametervektoren passen fast ebenso gut zu den zwanzig Punkten, doch ihre Vorhersagen unterscheiden sich dort um Größenordnungen. Mitteln wir die Vorhersagen und gewichten dabei jeden Vektor danach, wie plausibel er angesichts der Daten ist, ergibt sich eine fast fünfmal so hohe Wahrscheinlichkeit. Bei einer Eingabe von $2$, mitten im Trainingsbereich, unterscheiden sich die Vorhersage des einzelnen Vektors und die gemittelte Vorhersage um höchstens $0.01$. Bei einem seltenen Ereignis, etwa einem Betrugsalarm oder einer medizinischen Diagnose, kann ein Faktor fünf in der vorhergesagten Wahrscheinlichkeit unsere Entscheidung ändern.

<!--en:f8a496ba-->
Dieser gewichtete Mittelwert ist die bayessche Vorhersage. Ihre Berechnung ist ein Problem, das so alt ist wie der Satz von Bayes selbst. Um die A-posteriori-Verteilung zu erhalten, teilen wir durch die Evidenz, ein Integral über alle Parametervektoren, das selten eine geschlossene Form hat. Laplace umging das Integral 1774, indem er die A-posteriori-Verteilung eines Anteils durch eine Gauß-Verteilung ersetzte, die sich ihrem Gipfel anschmiegt. 1953 zeigten Metropolis und seine Kollegen, wie eine Irrfahrt (engl. random walk) Zustände proportional zu einer Verteilung mit unbekannter Normierungskonstante besuchen kann :cite:`Metropolis.Rosenbluth.Rosenbluth.ea.1953`. Bis zu den 1990er-Jahren hatte die Forschung das Integrationsproblem auch in ein Optimierungsproblem überführt. Alle diese Verfahren nutzen denselben Kunstgriff. Wir können über eine Verteilung mitteln, die wir nur bis auf eine Normierungskonstante kennen, weil jedes Verfahren allein auf Dichteverhältnissen beruht, in denen sich die unbekannte Konstante herauskürzt.

<!--en:dc8db4dd-->
In diesem Abschnitt entwickeln wir anhand von vier Fragen vier Ansätze. Erstens, können wir die A-posteriori-Verteilung durch eine Gauß-Verteilung annähern, die ihren Gipfel nachbildet? Zweitens, können wir aus einer einfacheren Verteilung ziehen und jede Ziehung korrigieren? Drittens, kann eine Irrfahrt in jedem Gebiet einen Zeitanteil verbringen, der proportional zu dessen A-posteriori-Masse ist? Viertens, können wir eine einfache Verteilung so optimieren, dass sie der A-posteriori-Verteilung möglichst nahekommt? Die ersten drei Verfahren beheben jeweils eine Einschränkung des vorherigen Ansatzes. Das vierte ersetzt Sampling durch Optimierung. An einer niedrigdimensionalen A-posteriori-Verteilung lässt sich der Fall einer nur bis auf eine Konstante bekannten Verteilung am klarsten untersuchen. Dieselben Ideen begegnen uns an vielen Stellen im Buch wieder. Einen Überblick gibt :numref:`sec_mdl-bayes-decision-map`. Mit allen vier Verfahren wollen wir die **posterior-prädiktive** Verteilung (engl. posterior predictive distribution) eines Labels $y_\star$ bei einer neuen Eingabe $\mathbf x_\star$ berechnen,

<!--en:56d3a49b-->
die die Vorhersagen aller Parametervektoren mittelt und sie dabei mit ihrer jeweiligen A-posteriori-Dichte bei gegebenen Daten $\mathcal D$ gewichtet.

<!--en:50b3a1d0-->
## Die A-posteriori-Verteilung der bayesschen logistischen Regression

<!--en:3c881b17-->
Für einen fairen Vergleich der Näherungsverfahren brauchen wir eine A-posteriori-Verteilung ohne geschlossene Form, für die wir dennoch das exakte Ergebnis zum Vergleich heranziehen können. Eine A-priori-Verteilung (engl. prior distribution) heißt *konjugiert* zu einer Likelihood, wenn die A-posteriori-Verteilung derselben Familie angehört wie die A-priori-Verteilung. So bleibt eine Beta-Verteilung für die Kopfwahrscheinlichkeit einer Münze nach beliebig vielen Würfen eine Beta-Verteilung (:numref:`sec_mdl-exponential-family`). Ein Gauß-Prior ist zur logistischen Likelihood nicht konjugiert. Wir können die unnormierte A-posteriori-Dichte an jedem Parametervektor auswerten, sie aber nicht analytisch integrieren. Unser durchgängiges Beispiel hat jedoch nur zwei Parameter. Daher liefert ein feines zweidimensionales Gitter einen exakten Referenzwert für jede Näherung.

<!--en:f4e69488-->
### Das Modell und seine unnormierte A-posteriori-Dichte

<!--en:ec26581d-->
Bei der eindimensionalen logistischen Regression hat jedes binäre Label $y_i\in\{0,1\}$ eine Wahrscheinlichkeit, die von einer linearen Funktion der Eingabe $x_i$ abhängt,

<!--en:782f71ac-->
wobei $\mathbf z_i=(1,x_i)^\top$ gilt. Die vorhergesagte Wahrscheinlichkeit durchläuft $\tfrac12$ an der Nullstelle der linearen Funktion, also bei $x=-\theta_0/\theta_1$. Die Steigung $\theta_1$ bestimmt, wie steil die Kurve dort ansteigt. Für Achsenabschnitt und Steigung wählen wir unabhängige Gauß-Priors $\theta_j\sim\mathcal N(0,\tau^2)$ mit A-priori-Standardabweichung $\tau$. Nach der Proposition von Bayes ist die A-posteriori-Dichte proportional zum Produkt aus der Likelihood der $n$ Labels und der A-priori-Dichte:

<!--en:29d185d3-->
Das Produkt in eckigen Klammern ist die Likelihood, der Exponentialfaktor die A-priori-Dichte. Die fehlende Normierungskonstante ist die **Evidenz** $p(\mathcal D)=\int p(\mathcal D,\boldsymbol\theta)\,d\boldsymbol\theta$, ein Integral über alle Parametervektoren ohne geschlossene Form. Wir bezeichnen die rechte Seite mit $\widetilde p(\boldsymbol\theta)=p(\mathcal D,\boldsymbol\theta)$ und nennen sie die *unnormierte A-posteriori-Dichte*. Die normierte A-posteriori-Dichte lautet dann $\widetilde p/p(\mathcal D)$.

<!--en:40dd6cec-->
Für unseren Beispieldatensatz ziehen wir $n=20$ Eingaben gleichverteilt aus $[0,4]$ und erzeugen die Labels mit dem wahren Parametervektor $\boldsymbol\theta^\star=(-3,1.5)$. Seine Wahrscheinlichkeitskurve durchläuft $\tfrac12$ bei $x=2$ und steigt von $\operatorname{sigmoid}(-3)\approx0.05$ bei $x=0$ auf $\operatorname{sigmoid}(3)\approx0.95$ bei $x=4$. Die A-priori-Varianz setzen wir auf $\tau^2=4$. Bei zwanzig Datenpunkten bleibt die A-posteriori-Verteilung breit genug, dass alle hier untersuchten Näherungseffekte deutlich hervortreten. Damit wir konkrete Referenzwerte angeben können, verwenden wir in jedem Lauf dieselbe repräsentative Stichprobe von zwanzig Punkten. Unter 1500 simulierten Datensätzen war sie nach den unten bestimmten A-posteriori-Kenngrößen die typischste. Jeder Sampler verwendet weiterhin neue Zufallszahlen. Seine Ergebnisse geben wir daher als typische Wertebereiche an. Im Code berechnet `log_joint` den Wert von $\log\widetilde p$. Dabei verwenden wir `logaddexp(0, z)` für $\log(1+e^z)$, um einen Überlauf bei großen Logits zu vermeiden.

<!--en:2f12e9a6-->
### Gradient und Hesse-Matrix

<!--en:6d4ca8a0-->
Mehrere der folgenden Verfahren nutzen Steigung und Krümmung von $\log\widetilde p$. Daher leiten wir beide vorab her. Die Ableitung jedes Log-Likelihood-Terms $y_i\,\mathbf z_i^\top\boldsymbol\theta-\log(1+e^{\mathbf z_i^\top\boldsymbol\theta})$ ergibt $(y_i-p_i)\,\mathbf z_i$ mit $p_i=\operatorname{sigmoid}(\mathbf z_i^\top\boldsymbol\theta)$. Der Gauß-Prior trägt $-\boldsymbol\theta/\tau^2$ bei. Fassen wir die Eingabezeilen $\mathbf z_i^\top$ zur $n\times2$-Matrix $\mathbf Z$ zusammen, erhalten wir den Gradienten $\nabla\log\widetilde p(\boldsymbol\theta)=\mathbf Z^\top(\mathbf y-\mathbf p)-\boldsymbol\theta/\tau^2$. Seine Geometrie lässt sich direkt ablesen. Jeder Trainingspunkt verschiebt $\boldsymbol\theta$ in Richtung $\mathbf z_i$, proportional zum Residuum $y_i-p_i$ seiner Vorhersage. Die A-priori-Verteilung zieht die Parameter leicht zum Ursprung zurück. Erneutes Ableiten ergibt die negative Hesse-Matrix

<!--en:cffc8b91-->
Der Datenterm ist eine gewichtete Gram-Matrix der Eingaben. Jedes Gewicht $p_i(1-p_i)\le\tfrac14$ erreicht bei $p_i=\tfrac12$ sein Maximum. Daher tragen die Punkte, bei denen das Modell am unsichersten ist, am stärksten zur Krümmung bei. Der additive Prior-Term $\tau^{-2}\mathbf I$ erhöht jeden Eigenwert um mindestens $\tau^{-2}$. Somit ist $\mathbf H(\boldsymbol\theta)$ überall positiv definit. Die Log-Dichte der A-posteriori-Verteilung ist also streng konkav und hat ein eindeutiges Maximum. Selbst bei linear trennbaren Trainingsdaten bleibt dieses Maximum endlich, denn die Log-Likelihood ist nie positiv und der quadratische Strafterm der A-priori-Verteilung fällt in jeder Richtung unbeschränkt ab.

<!--en:8c12720e-->
Bei nur zwei Parametern lassen sich diese Ableitungen leicht von Hand berechnen. Für größere Modelle würden wir nur $\log\widetilde p$ implementieren und den Gradienten sowie die Hesse-Matrix durch automatisches Differenzieren bestimmen.

<!--en:da52d37e-->
Alle Verfahren in diesem Abschnitt greifen ausschließlich über $\widetilde p$ auf die A-posteriori-Verteilung zu, sei es über die Funktionswerte, den Gradienten oder die Hesse-Matrix.

<!--en:7d140906-->
### A-posteriori-Modus und Laplace-Approximation

<!--en:26e0f3c2-->
Unsere erste und kostengünstigste Näherung baut direkt auf einer bereits bekannten Optimierung auf. Am Gipfel der A-posteriori-Verteilung liegt die MAP-Schätzung, die Schätzung mit Strafterm aus :numref:`sec_mdl-maximum_likelihood`. Das Newton-Verfahren $\boldsymbol\theta\leftarrow\boldsymbol\theta+\mathbf H(\boldsymbol\theta)^{-1}\nabla\log\widetilde p(\boldsymbol\theta)$ erreicht sie in wenigen Iterationen. Eine zusätzliche Auswertung der Hesse-Matrix genügt, um aus dieser Punktschätzung eine Gauß-Verteilung zu gewinnen.

<!--en:39b55646-->
Dazu passen wir eine Gauß-Verteilung mit Zentrum am Modus $\boldsymbol\theta_{\mathrm{MAP}}$ an. Die Entwicklung von $\log\widetilde p$ bis zur zweiten Ordnung um $\boldsymbol\theta_{\mathrm{MAP}}$ ergibt mit $\mathbf H=\mathbf H(\boldsymbol\theta_{\mathrm{MAP}})$

<!--en:7036c38f-->
Der lineare Term entfällt, weil der Gradient am Modus verschwindet. Bis auf eine additive Konstante ist die rechte Seite die Log-Dichte einer Gauß-Verteilung mit Mittelwert $\boldsymbol\theta_{\mathrm{MAP}}$ und Kovarianz $\mathbf H^{-1}$. Exponenzieren ergibt daher die **Laplace-Approximation**,

<!--en:3b37f64c-->
Damit diese Näherung wohldefiniert ist, muss $\log\widetilde p$ an einem Modus im Inneren zweimal differenzierbar und die negative Hesse-Matrix $\mathbf H$ dort positiv definit sein. Beides gilt hier. Anschaulich hat ein spitzer Gipfel eine große Krümmung und ergibt deshalb eine schmale Gauß-Verteilung mit kleiner Varianz. Die Gauß-Verteilung erfasst ausschließlich die Lage und die lokale Krümmung des Gipfels der A-posteriori-Verteilung.

<!--en:faa0870f-->
Ein eindimensionales Beispiel zeigt dieses Verhalten unmittelbar. Betrachten wir die schiefe A-posteriori-Verteilung $\mathrm{Gamma}(4,1)$ mit unnormierter Dichte $\theta^3e^{-\theta}$ für $\theta>0$. Ihre Log-Dichte $3\log\theta-\theta$ hat die Ableitung $3/\theta-1$, die am Modus $\theta=3$ verschwindet. Die negative zweite Ableitung beträgt dort $3/\theta^2=\tfrac13$. Die Laplace-Approximation ist folglich $\mathcal N(3,3)$ mit Standardabweichung $\sqrt{3}\approx1.73$. Die exakte A-posteriori-Verteilung hat den Mittelwert $4$ und die Varianz $4$. Die Gauß-Verteilung liegt also zu weit links, ist zu schmal und legt sogar $\Phi(-\sqrt3)\approx4.2\%$ ihrer Masse unterhalb von $\theta=0$ ab, wo die wahre Dichte null ist (:numref:`fig_mdl-prob-bayes-laplace`).

<!--en:a1d53f64-->
Wenden wir dieselbe Rechnung mit einer gleichförmigen A-priori-Verteilung auf eine Münze an, die 9-mal Kopf und 4-mal Zahl zeigte, erhalten wir wieder die Maximum-Likelihood-Schätzung $9/13$ und ihren Standardfehler $0.128$ aus :numref:`subsec_mdl-coin-error-bar`. Bei einer flachen A-priori-Verteilung entspricht die Laplace-Approximation tatsächlich der Maximum-Likelihood-Schätzung mitsamt ihrem asymptotischen Fehlerbalken (Übung :exref:`ex_mdl-bayesian-computation-laplace-parameterization`).

<!--en:414bf299-->
![Laplace-Approximation einer schiefen eindimensionalen A-posteriori-Verteilung, der Gamma(4,1)-Dichte mit Modus 3. Links stimmt die quadratische Näherung der Log-Dichte nur in Wert, Steigung und Krümmung am Modus mit ihr überein. Rechts passt die Gauß-Verteilung $\mathcal{N}(3,3)$ nach dem Exponenzieren zwar zum Gipfel, erfasst aber die Schiefe nicht. 4.2% ihrer Masse (orange schattiert) liegen unterhalb der Trägergrenze $\theta=0$, wo die A-posteriori-Dichte null ist. Den rechten Randbereich unterschätzt sie. Der wahre Mittelwert beträgt 4.](../img/mdl-prob-bayes-laplace.svg)

<!--en:1adf7a25-->
Die unbekannte Evidenz geht an keiner Stelle in die Rechnung ein. In der Log-Dichte tritt $\log p(\mathcal D)$ als additive Konstante in $\log p(\boldsymbol\theta\mid\mathcal D)$ auf und verschwindet beim Ableiten. Integrieren wir die Exponentialfunktion der rechten Seite von :eqref:`eq_mdl-bayes-laplace-taylor`, gewinnen wir sogar eine Schätzung der Evidenz selbst, nämlich $p(\mathcal D)\approx\widetilde p(\boldsymbol\theta_{\mathrm{MAP}})(2\pi)^{d/2}\det(\mathbf H)^{-1/2}$ in $d$ Dimensionen (Übung :exref:`ex_mdl-bayesian-computation-evidence`).

<!--en:7f32260d-->
Die Laplace-Approximation verwendet nur lokale Ableitungen an einem einzigen Gipfel. Daher erfasst sie weder Schiefe noch schwere Verteilungsränder, Parametergrenzen oder weitere Modi. Ihr Zentrum liegt am Modus statt am Mittelwert. Auch die Skalierung ist problematisch. Bei einem neuronalen Netz mit $d$ Parametern hat $\mathbf H$ insgesamt $d^2$ Einträge. In der Praxis beschränkt man sich daher meist auf ihre Diagonale oder wendet die Näherung nur auf die letzte Schicht an.

<!--en:a706f670-->
Im Code bestimmen wir den Modus mit zehn Newton-Schritten vom Ursprung aus. Eine einzige Matrixinversion liefert die Laplace-Kovarianz.

<!--en:0e9551ad-->
Der A-posteriori-Modus liegt bei $(-2.14,\,1.12)$. Die zugehörige Gerade ist flacher als die Gerade mit den datenerzeugenden Parametern $(-3,1.5)$. Die Laplace-Standardabweichungen betragen $1.00$ für den Achsenabschnitt und $0.46$ für die Steigung. In jeder Koordinate liegt der wahre Wert also etwas weniger als eine Standardabweichung entfernt. Bemerkenswert ist auch die starke negative Korrelation $-0.86$. Eine steilere Gerade muss tiefer beginnen, um durch dieselbe Punktwolke zu verlaufen.

<!--en:7dc2464a-->
### Ein Gitter als Referenz

<!--en:615237fb-->
Um die Genauigkeit der Laplace-Approximation zu beurteilen, brauchen wir die exakte A-posteriori-Verteilung als Vergleichsmaßstab. Bei nur zwei Parametern können wir $\log\widetilde p$ auf einem feinen $m\times m$-Gitter auswerten. Für numerische Stabilität ziehen wir das Maximum ab, exponenzieren und normieren die Gewichte auf Summe eins. In $d$ Dimensionen erfordert ein solches Gitter $m^d$ Auswertungen. Die Gitterintegration eignet sich daher nur in wenigen Dimensionen als Referenz. Selbst in zwei Dimensionen müssen wir wissen, wo wir das Gitter platzieren. Wir zentrieren es bei $\boldsymbol\theta_{\mathrm{MAP}}$ und gehen in jede Richtung sieben Laplace-Standardabweichungen weit. So bleibt nur eine vernachlässigbare Randmasse außerhalb.

<!--en:55ca50c1-->
Das Gitter zeigt die Schiefe, die der Laplace-Approximation entgeht. Der A-posteriori-Mittelwert der Steigung, $1.24$, liegt eine Viertelstandardabweichung der A-posteriori-Verteilung über der MAP-Schätzung von $1.12$. Der Mittelwert des Achsenabschnitts, $-2.35$, liegt eine Fünftelstandardabweichung unter seiner MAP-Schätzung. Die Laplace-Standardabweichungen fallen um $4$ bis $5\%$ zu klein aus. Die Korrelation stimmt auf zwei Stellen überein. Diese Abweichungen in der Breite sind gering. Wichtiger ist, dass die A-posteriori-Verteilung zu steileren Geraden hin schief ist und ihr Mittelwert deshalb vom Modus abweicht.

<!--en:30e37a9b-->
### Plug-in-Vorhersage und posterior-prädiktive Vorhersage

<!--en:b1e61987-->
Jetzt können wir unsere Ausgangsfrage direkt beantworten. Die *Plug-in*-Vorhersage $\operatorname{sigmoid}(\mathbf z_\star^\top\boldsymbol\theta_{\mathrm{MAP}})$ verwendet nur den Modus. Die posterior-prädiktive Vorhersage :eqref:`eq_mdl-bayes-predictive` hingegen mittelt $\operatorname{sigmoid}(\mathbf z_\star^\top\boldsymbol\theta)$ mit den A-posteriori-Gewichten über das Gitter. Vergleichen wir beide Vorhersagen an drei Testeingaben $x_\star\in\{-3,2,7\}$ unterhalb, innerhalb und oberhalb des Trainingsbereichs. Dazu betrachten wir auch das Intervall, das die mittleren 90% der Vorhersagen plausibler Parametervektoren enthält.

<!--en:93c6f9b2-->
Innerhalb des Trainingsbereichs bei $x_\star=2$ stimmen beide Vorhersagen ($0.524$ und $0.529$) bis auf $0.01$ überein. Die plausiblen Parametervektoren ergeben einen mäßig breiten Bereich von $0.32$ bis $0.74$. Weit außerhalb der Daten, bei $x_\star=-3$, erstrecken sich die mittleren 90% der plausiblen Vorhersagen über mehr als drei Größenordnungen, von $4\times10^{-5}$ bis $0.09$. Die posterior-prädiktive Vorhersage ($0.019$) beträgt das $4.6$-Fache des Plug-in-Werts ($0.0041$). Bei $x_\star=7$ zeigt sich dieselbe Verstärkung für die Wahrscheinlichkeit von Label $0$ ($0.017$ gegenüber $0.003$). Wahrscheinlichkeiten so nahe bei $0$ oder $1$ lassen sich am besten auf der Log-Odds-Skala $\log\frac{P}{1-P}$ vergleichen. Wir zeichnen deshalb beide Kurven über einen größeren Bereich und markieren die Trainingseingaben am unteren Rand.

<!--en:f40090dd-->
Auf der Log-Odds-Skala ist die Plug-in-Vorhersage die Gerade $\theta_0+\theta_1x$. Innerhalb von $[0,4]$ folgt ihr die posterior-prädiktive Vorhersage fast genau. Außerhalb der Trainingsdaten biegt sie zurück zur Null. Zwei gegenläufige Effekte erklären die Abweichung. Erstens ist die Steigung beim A-posteriori-Mittelwert größer als bei der MAP-Schätzung. Das verschiebt extreme Vorhersagen *weg* von $\tfrac12$. Zweitens zieht die Mittelung der Sigmoidfunktion über unterschiedlich große Logits die Vorhersagen *hin* zu $\tfrac12$. Fern der Daten liefern plausible Parametervektoren mit flacherer Steigung viel größere Wahrscheinlichkeiten als der typische Vektor. Diese größeren Wahrscheinlichkeiten dominieren den Mittelwert.

<!--en:aac97208-->
Welcher Effekt überwiegt und wie stark? Eine Überschlagsrechnung zeigt die Größenordnung des Mittelungseffekts. Bei kleinen Wahrscheinlichkeiten gilt $\operatorname{sigmoid}(a)\approx e^a$. Die vorhergesagte Wahrscheinlichkeit entspricht daher ungefähr dem A-posteriori-Mittelwert von $e^a$ für den Logit $a=\mathbf z_\star^\top\boldsymbol\theta$. Wäre $a$ gaußverteilt mit Mittelwert $\mu$ und Varianz $s^2$, so ergäbe seine momenterzeugende Funktion $\mathbb{E}[e^a]=e^{\mu+s^2/2}$. Die Log-Wahrscheinlichkeit läge damit um $s^2/2$ über ihrem Wert beim mittleren Logit. Bei $x_\star=-3$ erhalten wir auf dem Gitter $s^2/2=2.87$, was einen Faktor von $e^{2.87}\approx18$ erwarten lässt. Tatsächlich beträgt der Faktor aber $4.6$, ein Anstieg von nur $1.51$ auf der logarithmischen Skala. Wo bleiben die fehlenden $1.36$ Nat? Die letzten beiden Codezeilen oben verfolgen den Weg von der Plug-in-Vorhersage zur posterior-prädiktiven Vorhersage Schritt für Schritt:

<!--en:1d525b78-->
| Schritt bei $x_\star=-3$ | Log-Wahrscheinlichkeit | Änderung |

<!--en:cf528004-->
| Plug-in: Logit am Modus | $-5.49$ | |

<!--en:5be0ec2c-->
| A-posteriori-Mittelwert des Logits | $-6.06$ | $-0.57$ |

<!--en:26d899a2-->
| $s^2/2$ addieren wie bei gaußverteiltem Logit | $-3.19$ | $+2.87$ |

<!--en:333e3ef5-->
| $e^a$ über die tatsächliche Verteilung des Logits mitteln | $-3.66$ | $-0.47$ |

<!--en:bbc246f9-->
| $\operatorname{sigmoid}(a)$ statt $e^a$ mitteln: posterior-prädiktive Vorhersage | $-3.98$ | $-0.32$ |

<!--en:2fcbe33e-->
Drei Korrekturen nach unten erklären den gesamten Unterschied zwischen $2.87$ und $1.51$. Erstens geht der Anstieg um $s^2/2$ vom mittleren Logit aus. Dieser liegt wegen der größeren Steigung beim A-posteriori-Mittelwert um $0.57$ unter dem Plug-in-Logit. Zweitens ist der Logit schief verteilt. Sein linker Rand reicht weit zu sehr negativen Werten, sein rechter Rand in Richtung null ist kürzer. Aus diesem rechten Rand stammt fast die gesamte Masse im Mittelwert von $e^a$. Das kostet weitere $0.47$. Drittens liefern die Vektoren, die den Mittelwert dominieren, bei $x_\star=-3$ keine sehr kleinen Wahrscheinlichkeiten mehr. Dort verläuft die Sigmoidfunktion unterhalb von $e^a$, was die letzten $0.32$ abzieht. Insgesamt gilt $2.87-0.57-0.47-0.32=1.51$.

<!--en:cb044fdc-->
Die Rechnung mit der Gauß-Verteilung gibt also eine Orientierung. Richtung, Größenordnung und asymptotische Abhängigkeit von $x_\star$ stimmen. Die Logit-Varianz $\textrm{Var}(\theta_0)+2x_\star\textrm{Cov}(\theta_0,\theta_1)+x_\star^2\textrm{Var}(\theta_1)$ wächst quadratisch mit der Entfernung von den Daten, die Verschiebung des Mittelwerts hingegen nur linear. Fern der Trainingseingaben überwiegt daher zwangsläufig der Varianzterm. Innerhalb der Daten bleiben beide Effekte vernachlässigbar (Übung :exref:`ex_mdl-bayesian-computation-moderated-predictions`).

<!--en:1013a8a4-->
Die größere vorhergesagte Wahrscheinlichkeit liegt nicht zwangsläufig näher am wahren Wert des datenerzeugenden Modells. Die wahre Gerade liefert $\operatorname{sigmoid}(-3-1.5\cdot3)=\operatorname{sigmoid}(-7.5)\approx0.0006$ bei $x_\star=-3$. Das liegt unter beiden Vorhersagen, weil zwanzig Punkte die Steigung nicht genau festlegen und die A-priori-Verteilung sie zur Null hin schrumpfen lässt. Die wahre Wahrscheinlichkeit liegt aber deutlich innerhalb des 90%-Intervalls der A-posteriori-Verteilung, wie bei einem typischen Datensatz zu erwarten. Für die posterior-prädiktive Vorhersage müssen wir den wahren Parameter nicht kennen. Sie gibt die exakte Wahrscheinlichkeit von $y_\star=1$ an, gemittelt über alle Parametervektoren, die mit der A-priori-Verteilung und den zwanzig Beobachtungen vereinbar sind. Im Rest des Abschnitts untersuchen wir, wie sich solche Mittelwerte berechnen lassen, wenn der Parameterraum für ein Gitter zu groß ist.

<!--en:92689905-->
## Rejection-Sampling und Importance-Sampling

<!--en:3279fdef-->
Die Laplace-Approximation ist schnell, hat aber ihr Zentrum am Modus und ignoriert die Schiefe. Könnten wir direkt aus der exakten A-posteriori-Verteilung ziehen, ließe sich jeder A-posteriori-Erwartungswert durch den empirischen Mittelwert der Ziehungen schätzen. Schiefe und Randbereiche würden dabei automatisch berücksichtigt. Die beiden Verfahren dieses Abschnitts wandeln Ziehungen aus einer leicht handhabbaren *Vorschlagsverteilung* (engl. proposal distribution) $q$ in exakte oder gewichtete Ziehungen aus der A-posteriori-Verteilung um. Dafür brauchen sie nur das Verhältnis $\widetilde p/q$.

<!--en:3095c9b8-->
### Rejection-Sampling

<!--en:f8945dc5-->
Der einfachste exakte Sampler zieht Kandidaten aus $q$, behält einige und verwirft den Rest. Wir wählen eine Vorschlagsverteilung $q$, aus der wir leicht ziehen und deren Dichte wir leicht auswerten können. Dazu wählen wir eine Konstante $M$, sodass $Mq(\boldsymbol\theta)\ge\widetilde p(\boldsymbol\theta)$ für jedes $\boldsymbol\theta$ gilt. So erhalten wir eine obere Hüllfunktion für die unnormierte A-posteriori-Dichte. Jede Ziehung $\boldsymbol\theta\sim q$ nehmen wir mit folgender Wahrscheinlichkeit an:

<!--en:df2d12f8-->
Geometrisch ist die Annahmewahrscheinlichkeit das Verhältnis der Höhe der Zieldichte zur Höhe der Hüllfunktion am vorgeschlagenen Punkt. Jede angenommene Ziehung stammt exakt aus $p(\boldsymbol\theta\mid\mathcal D)$, und die Annahmerate beträgt insgesamt $p(\mathcal D)/M$. Das lässt sich in zwei Schritten nachweisen. Ein Vorschlag fällt in ein infinitesimales Gebiet $d\boldsymbol\theta$ um $\boldsymbol\theta$ mit Wahrscheinlichkeit $q(\boldsymbol\theta)\,d\boldsymbol\theta$ und wird dort mit Wahrscheinlichkeit $a(\boldsymbol\theta)$ angenommen. Die gemeinsame Wahrscheinlichkeit, dort zu landen und angenommen zu werden, beträgt also $\widetilde p(\boldsymbol\theta)\,d\boldsymbol\theta/M$. Integrieren über alle $\boldsymbol\theta$ liefert die marginale Annahmewahrscheinlichkeit $\int\widetilde p/M=p(\mathcal D)/M$. Teilen wir die gemeinsame Dichte durch diese Wahrscheinlichkeit, ergibt sich die bedingte Dichte einer angenommenen Ziehung, $\widetilde p(\boldsymbol\theta)/p(\mathcal D)=p(\boldsymbol\theta\mid\mathcal D)$.

<!--en:267bcbee-->
Entscheidend ist, dass die unbekannte Evidenz nur in der Annahmerate vorkommt. Die Regel für Annahme oder Ablehnung kommt ohne sie aus. Wählen wir die A-priori-Verteilung als Vorschlagsverteilung $q$, so ist $\widetilde p/q$ die Likelihood und $M$ ihr Maximalwert. Wir ziehen also Parametervektoren aus der A-priori-Verteilung und nehmen jeden mit einer Wahrscheinlichkeit an, die seiner Likelihood geteilt durch die maximale Likelihood entspricht.

<!--en:27b1ee9c-->
Wie schnell verliert Rejection-Sampling mit zunehmender Datenmenge an Effizienz? Betrachten wir wieder die Münze mit 9-mal Kopf und 4-mal Zahl bei gleichförmiger A-priori-Verteilung. Die Likelihood $\theta^9(1-\theta)^4$ erreicht bei $\theta=9/13$ ihr Maximum $M=(9/13)^9(4/13)^4\approx3.3\times10^{-4}$. Die Evidenz beträgt $p(\mathcal D)=\int_0^1\theta^9(1-\theta)^4\,d\theta=9!\,4!/14!\approx1.0\times10^{-4}$, also ergibt sich eine Annahmerate von $0.305$. Verzehnfachen wir die Datenmenge (90-mal Kopf und 40-mal Zahl), fällt die Annahmerate auf $0.10$. Bei hundertfacher Datenmenge sinkt sie auf $0.03$. Die Breite der A-posteriori-Verteilung schrumpft wie $1/\sqrt n$, während die A-priori-Verteilung fest bleibt. Daher fällt die Annahmerate in einer Dimension wie $1/\sqrt n$ und in mehreren Dimensionen wie $n^{-d/2}$ bei $d$ Dimensionen. Eine kurze Simulation mit 100 000 Ziehungen aus der gleichförmigen A-priori-Verteilung bestätigt die Rechnung.

<!--en:fe49fcb2-->
Zwei wesentliche Hindernisse begrenzen den praktischen Einsatz von Rejection-Sampling. Erstens müssen wir eine gültige obere Schranke $M$ kennen, die sich außerhalb einfacher Modelle selten angeben lässt. Zweitens fällt die Annahmerate selbst bei fester Beobachtungszahl pro Parameter mindestens exponentiell mit der Dimension $d$.

<!--en:2dbf84a4-->
### Wichtigkeitsgewichte

<!--en:46c8db7a-->
Rejection-Sampling verwirft die meisten erzeugten Ziehungen. Statt mit dem Verhältnis $w=\widetilde p/q$ eine Münze zu werfen und Kandidaten zu verwerfen, können wir jede Ziehung behalten und mit $w$ gewichten. Um zu sehen, warum das funktioniert, schreiben wir einen beliebigen A-posteriori-Erwartungswert als Verhältnis zweier Integrale. In beiden Integranden multiplizieren und dividieren wir durch $q$:

<!--en:dd7c4a0b-->
Der Nenner $\mathbb{E}_q[w]$ ist gleich der unbekannten Evidenz $p(\mathcal D)=\int\widetilde p$. Wir brauchen ihn nie explizit zu berechnen, weil dieselbe Normierungskonstante Zähler und Nenner skaliert. Ersetzen wir die beiden Erwartungswerte durch Stichprobenmittel über Ziehungen $\boldsymbol\theta_1,\ldots,\boldsymbol\theta_N\sim q$, erhalten wir das **selbstnormierte Importance-Sampling**, $\sum_sw_sh(\boldsymbol\theta_s)/\sum_sw_s$. Nach dem Gesetz der großen Zahlen konvergiert dieses Verhältnis für $N\to\infty$ gegen den exakten A-posteriori-Erwartungswert. Bei endlichem $N$ weist es eine kleine Verzerrung auf, weil es ein Quotient zweier zufälliger Mittelwerte ist. Produkte vieler Likelihood-Terme führen in der Gleitkommaarithmetik schnell zum Unterlauf. Daher berechnen wir stets logarithmierte Gewichte und ziehen vor dem Exponenzieren das größte logarithmierte Gewicht ab, wie beim Log-Sum-Exp-Trick. Ist $q$ die A-priori-Verteilung, reduziert sich das Wichtigkeitsgewicht auf die Likelihood. Jede A-priori-Ziehung erhält ihr Gewicht also danach, wie gut sie die beobachteten Daten erklärt.

<!--en:cd1210d6-->
### Die effektive Stichprobengröße

<!--en:b1c35442-->
Auch wenn wir jede Ziehung behalten, trägt nicht jede gleich viel bei. Entfällt nahezu das gesamte Gewicht einer gewichteten Stichprobe auf eine einzige Ziehung, ist die Stichprobe effektiv nur eine Ziehung wert, wie groß $N$ auch sein mag. Das quantifizieren wir mit der **effektiven Stichprobengröße** (engl. effective sample size, ESS), die wir über die normierten Gewichte $\bar w_s=w_s/\sum_rw_r$ definieren:

<!--en:fb7dab73-->
Bei gleichen Gewichten ($\bar w_s=1/N$) gilt $N_{\mathrm{eff}}=N$. Trägt eine Ziehung das gesamte Gewicht, gilt $N_{\mathrm{eff}}=1$. Die ESS schätzt grob, wie vielen unabhängigen Ziehungen aus der exakten A-posteriori-Verteilung die gewichtete Stichprobe entspricht. Wie wir gleich sehen, kann sie uns jedoch in zwei wichtigen Punkten täuschen.

<!--en:3e1b4251-->
**Proposition.** Für $N\to\infty$ gilt

<!--en:ac0a96ed-->
wobei $p$ die normierte A-posteriori-Dichte ist. Für dieselbe Vorschlagsverteilung $q$ ist diese Effizienz im Grenzwert mindestens so hoch wie die Annahmerate $p(\mathcal D)/M$ des Rejection-Samplings.

<!--en:0a0c04d1-->
**Beweis.** Teilen wir Zähler und Nenner von :eqref:`eq_mdl-bayes-is-ess` durch $N^2$, erhalten wir $N_{\mathrm{eff}}/N=(\frac1N\sum_sw_s)^2/(\frac1N\sum_sw_s^2)$. Auf beide Mittelwerte lässt sich das Gesetz der großen Zahlen anwenden. Einsetzen von $w=p(\mathcal D)\,p/q$ liefert $\mathbb{E}_q[w^2]=p(\mathcal D)^2\int p^2/q$ und $\mathbb{E}_q[w]=p(\mathcal D)$ und damit den Ausdruck mit der $\chi^2$-Divergenz. Schließlich folgt aus $w\le M$ die Ungleichung $\mathbb{E}_q[w^2]\le M\,\mathbb{E}_q[w]$. Der Grenzwert beträgt also mindestens $\mathbb{E}_q[w]/M=p(\mathcal D)/M$. $\blacksquare$

<!--en:837ca066-->
Die ESS ist damit ein empirisches Maß dafür, wie gut $q$ mit $p$ übereinstimmt. Im Grenzfall großer $N$ liefert Importance-Sampling mindestens so viele effektive Ziehungen, wie Rejection-Sampling annimmt, und benötigt dafür keine Schranke $M$ für die Hüllfunktion. Bei unserer Münze mit gleichförmiger A-priori-Verteilung als Vorschlagsverteilung gilt $\mathbb{E}_q[w^2]=\int_0^1\theta^{18}(1-\theta)^8\,d\theta$. Daraus ergibt sich im Grenzwert ein ESS-Anteil von $0.42$ gegenüber einer Annahmerate von $0.305$ beim Rejection-Sampling. Wir können dieselben Ziehungen aus der Codezelle zum Rejection-Sampling wiederverwenden und neu gewichten, statt zwei Drittel zu verwerfen.

<!--en:e11f396c-->
Die Grenzwertformel zeigt auch, wie stark eine schlecht skalierte Vorschlagsverteilung die Effizienz beeinträchtigen kann. Für eine eindimensionale Gauß-Zielverteilung mit Standardabweichung $s_p$ und eine gaußsche Vorschlagsverteilung mit gleichem Zentrum und Standardabweichung $s_q=rs_p$ beträgt der ESS-Anteil im Grenzwert $\sqrt{2r^2-1}/r^2$ (Übung :exref:`ex_mdl-bayesian-computation-weight-ess`). Diese Kurve hat den Wert $0.83$ bei $r=0.8$, erreicht ihr Maximum $1$ bei $r=1$ und fällt dann langsam auf $0.66$ bei $r=2$ und $0.46$ bei $r=3$. Für $r\le1/\sqrt2$ fällt sie jedoch abrupt auf null, weil $\mathbb{E}_q[w^2]=\infty$ gilt. Eine zu schmale Vorschlagsverteilung ist also weitaus problematischer als eine zu breite.

<!--en:8647e334-->
Zwei Bedingungen gewährleisten die Identität :eqref:`eq_mdl-bayes-importance` für das Importance-Sampling. Es muss $q(\boldsymbol\theta)>0$ überall dort gelten, wo $\widetilde p(\boldsymbol\theta)>0$ gilt, und außerdem $\mathbb{E}_q|wh|<\infty$. Für einen zentralen Grenzwertsatz und damit für Fehlerbalken der Ordnung $O(1/\sqrt N)$ müssen zusätzlich $\mathbb{E}_q[w^2]$ und $\mathbb{E}_q[w^2h^2]$ endlich sein. Selbst dann kann die ESS bei endlicher Stichprobe gefährlich optimistisch ausfallen, wie :numref:`fig_mdl-prob-bayes-importance` zeigt.

<!--en:351a090f-->
![Importance-Sampling für eine Zielverteilung mit zwei Modi und zwei verschiedenen Vorschlagsverteilungen. Die Linien zeigen die normierten Gewichte von 24 Ziehungen. Links erreicht eine schmale Vorschlagsverteilung den rechten Modus nie. Ihre Gewichte sehen daher fast gleich aus (ESS 23 von 24), obwohl ihre Schätzung weit danebenliegt. Rechts deckt eine breitere Vorschlagsverteilung beide Modi ab. Ihre Gewichte sind ungleicher (ESS 17 von 24), ihre Schätzung ist aber wesentlich genauer. Die ESS misst die Ungleichheit der Gewichte. Ob die Vorschlagsverteilung die gesamte A-posteriori-Masse erfasst hat, lässt sich daran nicht erkennen.](../img/mdl-prob-bayes-importance.svg)

<!--en:0936512a-->
### Vorschlagsverteilungen für die A-posteriori-Verteilung der logistischen Regression

<!--en:2fec1000-->
Wie wählen wir eine Vorschlagsverteilung für die A-posteriori-Verteilung unserer logistischen Regression? Vergleichen wir zwei Gauß-Verteilungen, die A-priori-Verteilung mit 20 000 Ziehungen und die Laplace-Approximation mit 5000 Ziehungen. Dazu erfassen wir die ESS als Anteil aller Ziehungen, das größte normierte Gewicht und die Abweichung des geschätzten A-posteriori-Mittelwerts vom Gitterreferenzwert in Einheiten der A-posteriori-Standardabweichung.

<!--en:44f1d12f-->
Mit der A-priori-Verteilung als Vorschlagsverteilung beträgt die ESS nur etwa 4% der Zahl der Ziehungen, weil die meisten Ziehungen in Bereiche mit vernachlässigbarer Likelihood fallen. Richten wir die Vorschlagsverteilung mit der Laplace-Approximation an der A-posteriori-Verteilung aus, steigt die ESS deutlich. Typischerweise sind 80% bis 90% der Ziehungen effektiv, gelegentlich aber auch weit weniger. Mit beiden Vorschlagsverteilungen weicht der geschätzte A-posteriori-Mittelwert meist nur um wenige Hundertstel einer A-posteriori-Standardabweichung vom Gitterreferenzwert ab. Nur selten übersteigt der Fehler $0.15$.

<!--en:0b19c230-->
Die Laplace-Vorschlagsverteilung hat jedoch einen verborgenen Mangel, den eine typische ESS-Ausgabe nie erkennen lässt. Fern vom Ursprung bestimmt die A-priori-Verteilung das Randverhalten der A-posteriori-Verteilung. Entlang jedes Strahls $\boldsymbol\theta=t\mathbf v$ fällt die Log-Likelihood höchstens linear in $t$ (Übung :exref:`ex_mdl-bayesian-computation-posterior-tails`), die logarithmierte A-priori-Dichte hingegen quadratisch wie $-t^2/(2\tau^2)$. Bei einer gaußschen Vorschlagsverteilung $\mathcal N(\mathbf m,\mathbf S)$ divergiert das Integral $\mathbb{E}_q[w^2]=\int\widetilde p^2/q$ folglich, sobald ein Eigenwert von $\mathbf S$ unter $\tau^2/2=2$ liegt. Hier liegen beide Eigenwerte der Laplace-Kovarianz (etwa $1.17$ und $0.05$) deutlich unter $2$. Die Varianz der Wichtigkeitsgewichte ist also in *jeder* Richtung unendlich, nicht nur entlang einer einzelnen schmalen Achse.

<!--en:d406ff38-->
Warum sah unser Lauf mit 5000 Ziehungen dann so unauffällig aus? Der Bereich, in dem die Gewichte anwachsen, liegt viele Laplace-Standardabweichungen entfernt. Die meisten Läufe mit 5000 Ziehungen erreichen ihn nie. In einigen wenigen von hundert Läufen entfällt jedoch ein enormer Anteil des Gesamtgewichts auf eine einzige weit entfernte Ziehung, und die ESS bricht ein. Ein einzelner Lauf kann diese Instabilität nicht aufdecken. Das gelingt nur durch wiederholte Läufe oder eine Analyse der Verteilungsränder. Zwei einfache Änderungen stellen eine endliche Gewichtsvarianz sicher. Wir können die Gauß-Verteilung durch eine Student-$t$-Vorschlagsverteilung mit schwereren Rändern ersetzen (Übung :exref:`ex_mdl-bayesian-computation-posterior-tails`) oder eine *defensive Mischung* wie $q=0.9\,\mathcal N(\boldsymbol\theta_{\mathrm{MAP}},\mathbf H^{-1})+0.1\,\text{prior}$ verwenden. Die Dichte der Mischung beträgt überall mindestens ein Zehntel der A-priori-Dichte. Jedes Gewicht ist daher höchstens zehnmal so groß wie die Likelihood, die nie größer als $1$ ist.

<!--en:5b7007f1-->
Beim Importance-Sampling kürzt sich die unbekannte Evidenz automatisch heraus. Das Verfahren benötigt aber weiterhin eine einzige globale Vorschlagsverteilung $q$, die die gesamte A-posteriori-Verteilung abdeckt. In hohen Dimensionen kann das keine einfache globale Verteilung leisten.

<!--en:7b8ebcaf-->
## Markov-Chain-Monte-Carlo

<!--en:c12186a0-->
Statt eine globale Vorschlagsverteilung für die gesamte A-posteriori-Verteilung zu suchen, können wir den Parameterraum mit lokalen Schritten erkunden. Dazu konstruieren wir eine Irrfahrt, die in jedem Gebiet einen Zeitanteil proportional zu dessen A-posteriori-Masse verbringt. Eine solche Irrfahrt ist eine *Markov-Kette*. Ihr nächster Zustand hängt nur vom aktuellen Zustand ab, und zwar über eine *Übergangsregel* (engl. transition rule) $T(\boldsymbol\theta'\mid\boldsymbol\theta)$, die Wahrscheinlichkeit eines Übergangs von $\boldsymbol\theta$ nach $\boldsymbol\theta'$. Eine Verteilung $\pi$ heißt *stationär* für die Kette, wenn bei einem aktuellen Zustand aus $\pi$ auch der nächste Zustand gemäß $\pi$ verteilt ist:

<!--en:dbc86a4f-->
Bei kontinuierlichen Parametern tritt an die Stelle der Summe ein Integral. Wir suchen eine einfache Übergangsregel mit der A-posteriori-Verteilung als stationärer Verteilung.

<!--en:0c305597-->
### Die Metropolis-Regel

<!--en:124a16e9-->
Der Metropolis-Algorithmus konstruiert eine solche Kette aus einer beliebigen symmetrischen Vorschlagsverteilung $g(\boldsymbol\theta'\mid\boldsymbol\theta)=g(\boldsymbol\theta\mid\boldsymbol\theta')$, etwa einem lokalen gaußschen Schritt $\boldsymbol\theta'=\boldsymbol\theta+\mathbf L\boldsymbol\epsilon$. Vom aktuellen Zustand $\boldsymbol\theta$ aus schlagen wir einen Kandidaten $\boldsymbol\theta'\sim g(\cdot\mid\boldsymbol\theta)$ vor. Hat er eine höhere oder gleich hohe A-posteriori-Dichte, gilt also $\widetilde p(\boldsymbol\theta')\ge\widetilde p(\boldsymbol\theta)$, wechseln wir stets zu $\boldsymbol\theta'$. Führt der Schritt bergab, nehmen wir ihn mit Wahrscheinlichkeit $\widetilde p(\boldsymbol\theta')/\widetilde p(\boldsymbol\theta)$ an. Andernfalls bleiben wir bei $\boldsymbol\theta$. Bei einem abgelehnten Vorschlag wiederholen wir also den aktuellen Zustand. Entscheidend ist, dass diese Wiederholung als nächster Zustand der Kette zählt (:numref:`fig_mdl-prob-bayes-metropolis`).

<!--en:2429edc3-->
![Metropolis-Irrfahrt auf den Höhenlinien der A-posteriori-Dichte. Vom Startpunkt (Quadrat) aus schlägt die Kette bei jedem Schritt eine lokale Bewegung vor. Schritte bergauf gelingen immer (Pfad), Schritte bergab nur mit einer gewissen Wahrscheinlichkeit. Bei abgelehnten Vorschlägen (Kreuze) bleibt die Kette einen weiteren Schritt an derselben Stelle. Langfristig entspricht der Zeitanteil in jedem Gebiet dessen A-posteriori-Masse.](../img/mdl-prob-bayes-metropolis.svg)

<!--en:13fe41c9-->
Hastings :cite:`Hastings.1970` verallgemeinerte die Regel auf asymmetrische Vorschlagsverteilungen. Dazu multiplizierte er das Dichteverhältnis mit einem Korrekturfaktor $g(\boldsymbol\theta\mid\boldsymbol\theta')/g(\boldsymbol\theta'\mid\boldsymbol\theta)$, der jede Richtungspräferenz der Vorschlagsverteilung ausgleicht. Wir nehmen einen Kandidaten dann mit folgender Wahrscheinlichkeit an:

<!--en:667a55e3-->
Schritte zu Zuständen, die die Vorschlagsverteilung selten besucht, nehmen wir eher an. Ist $g$ symmetrisch, beträgt der Korrekturfaktor $1$. Die unbekannte Evidenz kürzt sich wieder im Verhältnis der Werte von $\widetilde p$ heraus.

<!--en:ccde05db-->
Schon die kleinstmögliche Kette zeigt, warum wir Ablehnungen mitzählen müssen. Betrachten wir zwei Zustände mit Zielverteilung $\pi=(\tfrac14,\tfrac34)$ und einem symmetrischen Vorschlag, der stets den *anderen* Zustand wählt. Vom Zustand $1$ aus beträgt das Verhältnis der Zieldichten $3$. Die Kette springt also immer zu Zustand $2$. Von Zustand $2$ aus beträgt das Verhältnis $\tfrac13$. Die Kette springt also zu Zustand $1$ mit Wahrscheinlichkeit $\tfrac13$ und bleibt in Zustand $2$ mit Wahrscheinlichkeit $\tfrac23$. Schreiben wir die Übergangsmatrix so, dass die Zeile jeweils dem aktuellen Zustand entspricht, können wir die Stationarität direkt prüfen:

<!--en:c4b83b45-->
Würden wir Ablehnungen verwerfen und nur angenommene Schritte aufzeichnen, ergäbe sich die alternierende Folge $2, 1, 2, 1$ mit Verteilung $(\tfrac12,\tfrac12)$ statt $(\tfrac14,\tfrac34)$. Erst die wiederholten Besuche in Zustand $2$ nach Ablehnungen geben Zustand $2$ das richtige Gewicht (:numref:`fig_mdl-prob-bayes-two-state`).

<!--en:bf95b399-->
![Metropolis-Kette mit zwei Zuständen und Zielverteilung $\pi=(1/4,3/4)$. Der Wahrscheinlichkeitsfluss von Zustand 1 nach Zustand 2 beträgt $\frac14\cdot1$, der Rückfluss $\frac34\cdot\frac13$. Beide sind gleich $1/4$, sodass die Belegungsverteilung $(1/4,3/4)$ bei jedem Schritt erhalten bleibt. Die Selbstschleife steht für abgelehnte Schritte aus Zustand 2. Allein aus den angenommenen Schritten ergäbe sich $(1/2,1/2)$.](../img/mdl-prob-bayes-two-state.svg)

<!--en:73c56569-->
### Detailliertes Gleichgewicht und Stationarität

<!--en:0b78695d-->
:numref:`fig_mdl-prob-bayes-two-state` zeigt den Mechanismus hinter Metropolis-Hastings. In einem Schritt fließt genauso viel Wahrscheinlichkeitsmasse von Zustand $1$ nach Zustand $2$ wie zurück. Daher gewinnt oder verliert keiner der Zustände an Wahrscheinlichkeit. Gilt dieses paarweise Gleichgewicht für jedes Zustandspaar, so ist die Zielverteilung stationär. Wenn Sie sich aus diesem Abschnitt ein theoretisches Ergebnis merken, dann dieses.

<!--en:973278d7-->
**Proposition.** Erfüllt eine Übergangsregel das *detaillierte Gleichgewicht* (engl. detailed balance),

<!--en:7acaadf4-->
so ist $\pi$ für die Kette stationär. Außerdem erfüllt die Metropolis-Hastings-Kette das detaillierte Gleichgewicht mit $\pi\propto\widetilde p$.

<!--en:961c1997-->
**Beweis.** Summieren wir beide Seiten von :eqref:`eq_mdl-bayes-detailed-balance` über $\boldsymbol\theta$, erhalten wir links die linke Seite von :eqref:`eq_mdl-bayes-stationary`. Rechts vereinfacht sich der Ausdruck zu $\pi(\boldsymbol\theta')\sum_{\boldsymbol\theta}T(\boldsymbol\theta\mid\boldsymbol\theta')=\pi(\boldsymbol\theta')$, weil die Übergangswahrscheinlichkeiten aus $\boldsymbol\theta'$ in der Summe $1$ ergeben. Für die zweite Aussage gilt bei $\boldsymbol\theta'=\boldsymbol\theta$ automatisch detailliertes Gleichgewicht. Gilt $\boldsymbol\theta'\ne\boldsymbol\theta$, kommt ein Übergang genau dann zustande, wenn wir $\boldsymbol\theta'$ vorschlagen und annehmen. Also gilt

<!--en:46fba1f1-->
und dieser Ausdruck ist offensichtlich symmetrisch in $\boldsymbol\theta$ und $\boldsymbol\theta'$. Da sich die Normierungskonstante von $\pi$ in $\alpha$ herauskürzt, können wir $\widetilde p$ für $\pi$ einsetzen. $\blacksquare$

<!--en:efd6d385-->
Der entscheidende algebraische Schritt besteht darin, $\pi g$ in das Minimum zu ziehen. So wird die Symmetrie unmittelbar sichtbar. In einem kontinuierlichen Zustandsraum erfordert die Punktmasse der abgelehnten Schritte etwas mehr Sorgfalt (Übung :exref:`ex_mdl-bayesian-computation-mh-acceptance`).

<!--en:17537c07-->
Stationarität allein garantiert noch nicht, dass eine Kette von einem beliebigen Startpunkt aus gegen $\pi$ konvergiert. Die Kette muss auch jedes Gebiet mit positiver A-posteriori-Masse erreichen können und darf nicht deterministisch zwischen Gebieten wechseln. Unter diesen beiden milden Bedingungen konvergiert der Zeitanteil, den die Kette langfristig in einem Gebiet verbringt, gegen dessen A-posteriori-Wahrscheinlichkeit.

<!--en:d23f8352-->
Der Metropolis-Algorithmus lässt sich in wenigen Zeilen implementieren. Im folgenden Code geben wir der gaußschen Vorschlagsverteilung mit dem Cholesky-Faktor der Laplace-Kovarianz ihre Form, $\mathbf L\mathbf L^\top=\mathbf H^{-1}$.

<!--en:3f21c814-->
Wir starten vier unabhängige Ketten mit je 12 000 Schritten. Ihre Startpunkte streuen doppelt so breit wie die Laplace-Verteilung, $\boldsymbol\theta_{\mathrm{MAP}}+2\mathbf L\boldsymbol\epsilon$. Wir schlagen Schritte der Größe $1.1\,\mathbf L\boldsymbol\epsilon$ vor. Da die frühen Zustände noch vom jeweiligen Startpunkt abhängen, verwerfen wir die ersten 2000 Iterationen jeder Kette, die *Aufwärmphase* (engl. warm-up), bevor wir Mittelwerte berechnen.

<!--en:6caf92ff-->
In jeder Kette nehmen wir etwa die Hälfte aller Vorschläge an. Die stationäre Verteilung wird durch die Metropolis-Annahmeregel bestimmt. Eine ungünstig geformte gaußsche Vorschlagsverteilung würde die Kette daher verlangsamen, ohne eine Verzerrung einzuführen.

<!--en:b096b003-->
### Diagnose der Durchmischung von Markov-Ketten

<!--en:c308e2ac-->
Anders als bei unabhängigen Stichproben sind aufeinanderfolgende Zustände einer Markov-Kette korreliert. Zudem beginnt jede Kette an dem Punkt, den wir für sie gewählt haben. Bevor wir einem Mittelwert aus einer Kette vertrauen, stellen wir drei diagnostische Fragen und beantworten jede anhand einer konkreten Kennzahl.

<!--en:fda0e09e-->
*Haben die Ketten ihre Startpunkte vergessen und sind sie gegen dieselbe Verteilung konvergiert?* Ein Verlaufsdiagramm (engl. trace plot) jeder Kette macht Stillstand oder Drift sichtbar. Die Kennzahl $\widehat R$ misst die Abweichungen zwischen den Ketten. Für mehrere Ketten mit jeweils $N$ Zuständen bezeichne $W$ die mittlere Varianz innerhalb der Ketten und $B$ das $N$-Fache der Varianz der Kettenmittelwerte. Dann gilt

<!--en:69baab83-->
Der Zähler schätzt die gesamte A-posteriori-Varianz. Solange die Ketten getrennt bleiben, überschätzt er sie tendenziell. Dagegen unterschätzt $W$ sie, solange jede Kette nur einen Teil der A-posteriori-Verteilung erkundet hat. Somit vergleicht $\widehat R$ zwei Varianzschätzungen und nähert sich $1$, wenn sich die Ketten durchmischen. In der Praxis berechnen wir **Split-$\widehat R$**. Dazu teilen wir zunächst jede Kette in zwei Hälften. So erscheint selbst eine einzelne Kette mit Drift als zwei voneinander abweichende Ketten. Bevor wir den Ergebnissen vertrauen, muss die Abweichung innerhalb von $0.01$ um $1$ liegen. Das allein genügt aber nie. Ketten, die alle denselben entfernten Modus verfehlen, können gut übereinstimmen.

<!--en:6f702770-->
*Wie vielen unabhängigen Ziehungen entsprechen diese korrelierten Zustände tatsächlich?* Sei $\rho_k$ die Autokorrelation zwischen Zuständen im Abstand von $k$ Schritten. Die integrierte Autokorrelationszeit $\kappa=1+2\sum_{k\ge1}\rho_k$ gibt an, wie viele Kettenschritte denselben Informationsgehalt haben wie eine unabhängige Ziehung. Für $c$ Ketten mit je $N$ Zuständen ergibt sich folglich eine effektive Stichprobengröße von $N_{\mathrm{eff}}=cN/\kappa$. Bedeutung und Symbol sind dieselben wie beim Importance-Sampling.

<!--en:a3055de2-->
Das einfachste Modell für zeitliche Korrelation ist der autoregressive Prozess erster Ordnung, AR(1), mit $x_t=\phi x_{t-1}+\varepsilon_t$, unabhängigem Rauschen $\varepsilon_t$ und $|\phi|<1$. Bei jedem Schritt behält die Kette einen Anteil $\phi$ ihres vorherigen Werts bei. Die Autokorrelation im Abstand $k$ beträgt $\rho_k=\phi^k$. Damit gilt

<!--en:528dfd5e-->
Bei $\phi=0.9$ ergibt sich $\kappa=19$. 10 000 Zustände entsprechen also nur etwa 530 unabhängigen Ziehungen. Bei $\phi=0.99$ gilt $\kappa=199$. Dann bleiben kaum 50 unabhängige Ziehungen.

<!--en:e8f26ee2-->
Bei sehr großen Schrittabständen dominiert das Rauschen die geschätzten Autokorrelationen. Unsere Implementierung fasst sie daher zu aufeinanderfolgenden Paaren $\rho_{2k}+\rho_{2k+1}$ zusammen und bricht die Summe beim ersten Paar mit negativer Summe ab. Inferenzpakete für den praktischen Einsatz verfeinern beide Diagnosegrößen weiter. Sie verwenden rangnormiertes $\widehat R$ und getrennte effektive Stichprobengrößen für den Hauptteil der Verteilung und ihre Ränder.

<!--en:96064404-->
*Wie genau ist unser geschätzter A-posteriori-Mittelwert?* Nach dem zentralen Grenzwertsatz für Markov-Ketten ist der **Monte-Carlo-Standardfehler** (engl. Monte Carlo standard error, MCSE) eines geschätzten A-posteriori-Mittelwerts gleich der A-posteriori-Standardabweichung geteilt durch $\sqrt{N_{\mathrm{eff}}}$.

<!--en:e6ba25a8-->
Prüfen wir diese Diagnosegrößen. Ist unser Fehlerbalken kalibriert, sollte der MCMC-Mittelwert aus unseren vier Ketten höchstens etwa einen MCSE vom exakten Gittermittelwert entfernt liegen. Außerdem prüfen wir, wie gut die gemessene Autokorrelationszeit $\kappa$ der Steigung mit der AR(1)-Formel für die Autokorrelation $\rho_1$ bei einem Schritt Abstand übereinstimmt.

<!--en:caddfb1c-->
Ein Verlaufsdiagramm für die Steigung über die ersten 600 Iterationen nach der Aufwärmphase vervollständigt unsere Diagnose.

<!--en:f9785364-->
Die Diagnose ist unauffällig. Split-$\widehat R$ liegt innerhalb von $0.01$ um $1$, und die vier Kettenverläufe überlagern sich eng. Die ESS beträgt etwa ein Zehntel der 40 000 beibehaltenen Zustände. Daraus folgt $\kappa\approx10$, in guter Übereinstimmung mit der AR(1)-Vorhersage aus der Autokorrelation $\rho_1\approx0.8$ bei einem Schritt Abstand. Besonders aufschlussreich ist, dass der MCMC-Mittelwert um etwa einen MCSE vom exakten Gittermittelwert abweicht. Unsere Unsicherheitsschätzung passt also zum tatsächlichen Schätzfehler.

<!--en:6954678e-->
### Vorschlagsskala und unzureichende Durchmischung

<!--en:9760e788-->
Die Annahmerate lässt sich bei einem Metropolis-Lauf besonders leicht ausgeben, sagt allein aber wenig aus. Lassen wir die vier Ketten erneut laufen, diesmal mit Vorschlagsskalen von $0.05$ und $10$ statt $1.1$. Überlegen Sie vor dem Blick auf die Zahlen, welches der beiden Extreme weniger effektive Ziehungen liefert.

<!--en:29da28e0-->
Bei winzigen Schritten (Skala $0.05$) nehmen wir etwa 97% der Vorschläge an. Dennoch steigt $\widehat R$ deutlich über $1.01$. Aus 40 000 Iterationen gewinnen wir nur wenige Dutzend effektive Ziehungen, weil jeder Schritt kaum vom Fleck kommt. Bei riesigen Schritten (Skala $10$) nehmen wir nur etwa 2% der Vorschläge an. Die Kette wiederholt über lange Strecken denselben Zustand, erreicht aber immerhin einige Hundert effektive Ziehungen. Die optimale Schrittweite liegt deutlich zwischen beiden Extremen (Übung :exref:`ex_mdl-bayesian-computation-proposal-scale`). Eine blinde Irrfahrt bewegt sich nur durch Zufall bergauf. Ein Vorschlag, der dem Gradienten von $\log\widetilde p$ folgt, hat dagegen gezielt eine Drift bergauf. Solche gradientengestützten Schritte ergeben sich aus der Langevin-Dynamik (:numref:`sec_mdl-langevin-sampling`). Ihre Metropolis-korrigierte Variante erarbeiten Sie in Übung :exref:`ex_mdl-bayesian-computation-mala`.

<!--en:90c551e8-->
## Variationelle Inferenz

<!--en:c92e716d-->
Sampling-Verfahren schätzen jeden A-posteriori-Erwartungswert aufs Neue und sind stets mit Monte-Carlo-Rauschen behaftet. Die *variationelle Inferenz* verfolgt einen grundsätzlich anderen Ansatz. Sie macht aus der Integration eine Optimierung. Wir wählen eine handhabbare Familie von Verteilungen $q_\psi$ mit Variationsparametern $\psi$. Dann optimieren wir $\psi$ so, dass $q_\psi$ der A-posteriori-Verteilung möglichst nahekommt. Anschließend berechnen wir Erwartungswerte unter $q_\psi$ entweder in geschlossener Form oder durch schnelles, unabhängiges Sampling.

<!--en:64feb52c-->
### Die untere Evidenzschranke

<!--en:daa3fcfc-->
Zur Optimierung von $q_\psi$ brauchen wir eine Zielfunktion, die die Nähe zur A-posteriori-Verteilung ohne die unbekannte Normierungskonstante $p(\mathcal D)$ misst. Eine solche Zielfunktion haben wir bereits in :numref:`sec_mdl-latent-em-elbo` hergeleitet. Dort lieferte eine Näherungsverteilung $q$ über einer latenten Variablen eine untere Schranke für die Log-Evidenz. Der Abstand zur Schranke war eine KL-Divergenz. Behandeln wir den Parametervektor $\boldsymbol\theta$ als latente Variable und den Datensatz $\mathcal D$ als Beobachtung, erhalten wir dieselbe Identität:

<!--en:b34126f5-->
Da die KL-Divergenz stets nichtnegativ ist, bildet $\mathcal L(\psi)$ die **untere Evidenzschranke** (engl. evidence lower bound, ELBO) für $q_\psi$,

<!--en:0bd810cb-->
Die beiden Terme auf der rechten Seite von :eqref:`eq_mdl-bayes-elbo-gap` summieren sich zur festen Konstanten $\log p(\mathcal D)$. Steigt $\mathcal L(\psi)$ um eine Einheit, sinkt $D_{\textrm{KL}}(q_\psi\,\|\,p(\cdot\mid\mathcal D))$ daher um genau eine Einheit. Die ELBO über $\psi$ zu maximieren ist somit gleichbedeutend damit, die KL-Divergenz zu minimieren. Die unbekannte Evidenz fällt als additive Konstante weg. Jeden Term innerhalb des Erwartungswerts können wir punktweise auswerten. Den Erwartungswert selbst können wir leicht mit Ziehungen aus $q_\psi$ schätzen.

<!--en:ecfac8d0-->
### Die Richtung der KL-Divergenz

<!--en:4f863e79-->
Beachten Sie die Reihenfolge der Argumente in :eqref:`eq_mdl-bayes-elbo-gap`. Die variationelle Inferenz minimiert $D_{\textrm{KL}}(q\|p)=\int q\log(q/p)$, die *rückwärts gerichtete* KL-Divergenz, mit $q$ an erster Stelle. Ihr Integrand divergiert überall dort, wo $q>0$ und zugleich $p\approx0$ gilt. Daher bestraft die Rückwärts-KL es stark, wenn $q$ Masse in Bereiche legt, in denen die A-posteriori-Verteilung keine hat. Folglich zieht sich $q$ aus Bereichen geringer Dichte zurück und konzentriert sich meist auf einen einzigen Modus. Die Rückwärts-KL ist *modussuchend*. Die *vorwärts gerichtete* Divergenz $D_{\textrm{KL}}(p\|q)=\int p\log(p/q)$ bestraft hingegen Bereiche mit $p>0$ und $q\approx0$. Dadurch muss sich $q$ über die gesamte Masse der A-posteriori-Verteilung erstrecken. Die Vorwärts-KL ist *massendeckend*. Minimieren wir die Vorwärts-KL für eine Familie von Gauß-Verteilungen, erhalten wir den exakten Mittelwert und die exakte Kovarianz der A-posteriori-Verteilung. Dazu benötigen wir aber Erwartungswerte unter der schwer handhabbaren A-posteriori-Verteilung $p$. Für die Rückwärts-KL brauchen wir nur Erwartungswerte unter $q$. :numref:`fig_mdl-prob-bayes-kl-modes` zeigt beide Verhaltensweisen für eine bimodale Zielverteilung.

<!--en:c14a657c-->
![Die Richtung der KL-Divergenz bestimmt, auf welche Weise eine zu einfache Näherung scheitert. Bei einer Zielverteilung mit zwei Modi konzentriert sich die durch variationelle Inferenz gefundene Gauß-Verteilung der Rückwärts-KL auf einen einzelnen Modus und unterschätzt die Unsicherheit. Die Gauß-Verteilung der Vorwärts-KL deckt beide Modi ab und legt ihren Gipfel in das Tal geringer Dichte zwischen ihnen.](../img/mdl-prob-bayes-kl-modes.svg)

<!--en:7c40c745-->
### Anpassung einer Mean-Field-Gauß-Verteilung

<!--en:67e31f22-->
Die einfachste Variationsfamilie ist die *Mean-Field*-Gauß-Verteilung $q=\mathcal N(\mathbf m,\operatorname{diag}(\mathbf s^2))$, die die Parameter als unabhängig modelliert. Um die Gradienten der ELBO nach $\mathbf m$ und $\log\mathbf s$ zu berechnen, reparametrisieren wir eine Ziehung als $\boldsymbol\theta=\mathbf m+\mathbf s\odot\boldsymbol\epsilon$ mit $\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I)$. Da die Verteilung des Rauschens nicht von den Variationsparametern abhängt, können wir den Gradienten in den Erwartungswert ziehen (:numref:`subsec_mdl-stochastic-gradient-estimators`). Die Entropie $-\mathbb{E}_q[\log q]$ ist bis auf eine additive Konstante gleich $\sum_j\log s_j$. Damit erhalten wir die erwartungstreuen Gradientenformeln

<!--en:382bc2fe-->
wobei $\mathbf 1$ aus der Ableitung der Entropie $\sum_j\log s_j$ nach $\log\mathbf s$ stammt. Der erste Gradient verschiebt den Mittelwert im Mittel bergauf. Der zweite vergrößert $q$, bis der nach außen gerichtete Einfluss der Entropie die abwärts gerichtete Krümmung der A-posteriori-Log-Dichte ausgleicht. Bevor wir den Optimierer auf unsere logistische Regression anwenden, lösen wir den Fall einer gaußschen Zielverteilung exakt. So können wir vorhersagen, was geschieht.

<!--en:14adc6fc-->
**Proposition.** Für eine gaußsche Zielverteilung $p=\mathcal N(\boldsymbol\mu,\boldsymbol\Lambda^{-1})$ mit Präzisionsmatrix $\boldsymbol\Lambda$ hat die Mean-Field-Gauß-Verteilung, die $D_{\textrm{KL}}(q\|p)$ minimiert, den Mittelwert $\boldsymbol\mu$ und die Varianzen $s_j^2=1/\Lambda_{jj}$. Dies sind die *bedingten* Varianzen der Zielverteilung. In zwei Dimensionen mit Korrelation $\rho$ beträgt jedes $s_j$ das $\sqrt{1-\rho^2}$-Fache der wahren marginalen Standardabweichung.

<!--en:8f33faeb-->
**Beweis.** Lassen wir die von $q$ unabhängigen Konstanten weg, gilt $D_{\textrm{KL}}(q\|p)=-\sum_j\log s_j+\tfrac12\mathbb{E}_q[(\boldsymbol\theta-\boldsymbol\mu)^\top\boldsymbol\Lambda(\boldsymbol\theta-\boldsymbol\mu)]$. Unter $q$ sind die Koordinaten unabhängig mit Mittelwerten $m_j$ und Varianzen $s_j^2$. Daher verschwinden alle Kreuzterme der Abweichungen im Erwartungswert, und es bleibt $\sum_j\Lambda_{jj}s_j^2+(\mathbf m-\boldsymbol\mu)^\top\boldsymbol\Lambda(\mathbf m-\boldsymbol\mu)$. Da $\boldsymbol\Lambda$ positiv definit ist, erreicht der quadratische Ausdruck in $\mathbf m$ sein Minimum bei $\mathbf m=\boldsymbol\mu$. Setzen wir die Ableitung nach $s_j$ gleich null, $-1/s_j+\Lambda_{jj}s_j=0$, erhalten wir $s_j^2=1/\Lambda_{jj}$. Für $d=2$ ergibt die Inversion der $2\times2$-Kovarianzmatrix den Ausdruck $1/\Lambda_{jj}=\sigma_j^2(1-\rho^2)$, wobei $\sigma_j$ die marginale Standardabweichung ist. $\blacksquare$

<!--en:863a0819-->
Eine Gauß-Verteilung mit diagonaler Kovarianz kann ihre Achsen nicht an einer korrelierten A-posteriori-Verteilung ausrichten. Bei der Rückwärts-KL zahlt sie für Masse in Ecken geringer Dichte einen hohen Preis. Deshalb schrumpft sie auf die Breite eines bedingten Schnitts durch die A-posteriori-Verteilung, statt die volle marginale Breite zu erfassen (:numref:`fig_mdl-prob-bayes-mean-field`). Diese Schrumpfung entsteht gerade aus dem Zusammenspiel der faktorisierten Familie mit der Rückwärts-KL. Passen wir eine Gauß-Verteilung mit voller Kovarianzmatrix mit genau derselben ELBO an eine gaußsche Zielverteilung an, erhalten wir sie fehlerfrei zurück. Im Beispiel der logistischen Regression gilt $\rho=-0.86$. Nach der Proposition betragen die Mean-Field-Standardabweichungen also ungefähr das $\sqrt{1-0.86^2}=0.51$-Fache der wahren marginalen Standardabweichungen, etwa die Hälfte.

<!--en:6e36d1c7-->
![Gauß-Verteilungen mit diagonaler Kovarianz, angepasst an einen korrelierten Gauß-Posterior mit Standardabweichungen $(1,0.5)$ und Korrelation $\rho=-0.85$. Die Ellipsen zeigen zwei Standardabweichungen. Da die Achsen der faktorisierten Familie festliegen, schrumpft die Näherung der Rückwärts-KL (orange) auf die bedingte Breite, also auf das $\sqrt{1-\rho^2}\approx0.53$-Fache der marginalen Standardabweichung. Die Näherung der Vorwärts-KL (grün) erhält die marginale Breite, legt dabei aber Masse in Bereiche geringer Dichte.](../img/mdl-prob-bayes-mean-field.svg)

<!--en:dc890e46-->
Prüfen wir diese Vorhersage numerisch. Wir starten bei der Laplace-Approximation und führen mit :eqref:`eq_mdl-bayes-vi-gradients` insgesamt 3000 Iterationen des stochastischen Gradientenaufstiegs aus. Pro Schritt verwenden wir 256 reparametrisierte Ziehungen und eine Lernrate, die wie $1/\sqrt t$ abfällt. Wir mitteln die letzten 1000 Iterierten, um das Rauschen der Optimierung zu glätten. So können wir die angepassten Standardabweichungen direkt mit der Vorhersage der Proposition für die Gitterkorrelation $\rho$ vergleichen.

<!--en:d364ff39-->
Der Mean-Field-Mittelwert liegt höchstens etwa eine Zehntelstandardabweichung der A-posteriori-Verteilung vom Gittermittelwert entfernt. Die Standardabweichungen schrumpfen auf etwa die Hälfte der wahren Breiten und stimmen mit unseren analytischen Vorhersagen ($0.54$ und $0.25$) bis auf $2\%$ überein. Der Optimierer ist konvergiert. Die verbleibende Abweichung geht daher vollständig auf die eingeschränkte Variationsfamilie zurück. Reichhaltigere Familien wie Gauß-Verteilungen mit voller Kovarianzmatrix, Mischungen oder normalisierende Flüsse schließen diese Lücke bei höherem Rechenaufwand. Eine konvergierte ELBO bestätigt nur, dass die Optimierung *innerhalb der gewählten Familie* abgeschlossen ist. Ob die Familie die A-posteriori-Verteilung gut approximiert, lässt sich daraus allein nie ableiten.

<!--en:d6c685d9-->
## Vergleich der Verfahren

<!--en:b6570dc9-->
Wir haben nun vier unterschiedliche Näherungen für dieselbe A-posteriori-Verteilung. Ein Diagramm stellt die Ergebnisse für die marginale Dichte der Steigung $\theta_1$ gegenüber. Es zeigt die exakte Gitterkurve, die Gauß-Verteilungen der Laplace- und der Mean-Field-Approximation sowie ein Histogramm der Metropolis-Ziehungen.

<!--en:9720b015-->
Das Metropolis-Histogramm gibt die exakte A-posteriori-Verteilung einschließlich ihrer Schiefe gut wieder. Die Laplace-Kurve ist am Modus links vom wahren Mittelwert zentriert und etwas zu schmal. Die Mean-Field-Gauß-Verteilung ist etwa halb so breit wie die exakte Randverteilung. Konstruktionsbedingt ergibt sie eine Korrelation von null gegenüber dem wahren Wert $\rho=-0.86$.

<!--en:6d216921-->
| Verfahren | Darstellung | Wichtigste Diagnose | Typisches Problem |

<!--en:7df88fb5-->
| Gitter | gewichtete Gitterpunkte | Masse am Gitterrand | Aufwand $m^d$ in Dimension $d$ |

<!--en:0a3449ff-->
| Laplace | eine Gauß-Verteilung am Modus | Vergleich mit Ziehungen | Schiefe, Verteilungsränder, Grenzen, weitere Modi |

<!--en:ae394a3e-->
| Rejection-Sampling | unabhängige exakte Ziehungen | Annahmerate | Rate fällt wie $n^{-d/2}$, Schranke $M$ erforderlich |

<!--en:608328f8-->
| Importance-Sampling | unabhängige gewichtete Ziehungen | ESS der Gewichte, größtes Gewicht, wiederholte Läufe | Vorschlagsverteilung mit zu leichten Rändern oder falscher Lage |

<!--en:4bcf8211-->
| MCMC | korrelierte A-posteriori-Ziehungen | Kettenverläufe, Split-$\widehat R$, ESS, MCSE | langsame Durchmischung, unbesuchte Modi |

<!--en:d64da247-->
| Variationelle Inferenz | optimiertes Mitglied einer Verteilungsfamilie | Vergleich mit Ziehungen | zu enge Familie, faktorisiertes $q$ unterschätzt Unsicherheit |

<!--en:9ca29cf6-->
Welches Verfahren sollten Sie in der Praxis wählen? Bei nur wenigen Parametern bietet sich ein Gitter an. Für eine glatte, unimodale A-posteriori-Verteilung liefert die Laplace-Approximation schnell ein erstes Ergebnis und eine gute erste Vorschlagsverteilung. Brauchen Sie für einen Erwartungswert einen verlässlichen Fehlerbalken, verwenden Sie MCMC mit seinen drei Diagnosegrößen. Die variationelle Inferenz eignet sich besonders dann, wenn die Näherung innerhalb einer größeren Optimierungsschleife liegen muss. Dabei ist zu beachten, dass faktorisierte Familien die Unsicherheit unterschätzen. Da keine der beiden gaußschen Näherungen selbst ihre Genauigkeit prüfen kann, sollten wir beide nach Möglichkeit mit einem Sampler überprüfen.

<!--en:0793fe94-->
Für große Modelle stoßen alle vier Verfahren an Grenzen. Keines lässt sich ohne Weiteres auf die gemeinsame A-posteriori-Verteilung aller Parameter eines großen neuronalen Netzes skalieren. Mit steigender Dimension degenerieren die Wichtigkeitsgewichte, und MCMC mit Irrfahrt benötigt immer mehr Schritte pro unabhängiger Ziehung. Die volle Hesse-Matrix hat $d^2$ Einträge, und die Mean-Field-Näherung $q$ unterschätzt die Unsicherheit um den gerade hergeleiteten Korrelationsfaktor. Hinzu kommt, dass architekturbedingte Symmetrien eines neuronalen Netzes jeden A-posteriori-Modus bei jeder Permutation und jedem Vorzeichenwechsel seiner versteckten Einheiten vervielfachen (Übung :exref:`ex_mdl-bayesian-computation-two-modes`).

<!--en:2a594ba3-->
Für große Netze ist die verbreitetste Antwort auf unsere Ausgangsfrage einfacher als jedes der vier Verfahren. Wir trainieren dasselbe Netz mehrmals mit unterschiedlichen Zufallsinitialisierungen und mitteln die vorhergesagten Wahrscheinlichkeiten. Ein solches *tiefes Ensemble* (engl. deep ensemble) :cite:`Lakshminarayanan.Pritzel.Blundell.2017` liefert eine grobe Monte-Carlo-Schätzung der posterior-prädiktiven Vorhersage :eqref:`eq_mdl-bayes-predictive`. Eine Handvoll unabhängig trainierter Netze steht dabei für A-posteriori-Ziehungen. Jeder Lauf endet an einem anderen Modus. Der Mittelwert erfasst daher mehrere Modi, was eine einzelne Laplace- oder Mean-Field-Approximation nicht leisten kann. Wo die Mitglieder voneinander abweichen, typischerweise fern der Trainingsdaten, fällt die gemittelte Vorhersage weniger sicher aus. Die Mitglieder sind allerdings keine Ziehungen aus der A-posteriori-Verteilung, und nichts garantiert, dass sie gleich gewichtet werden sollten. Auch ihre Fehler sind korreliert, sodass zusätzliche Mitglieder nur begrenzt helfen. Bei paarweiser Korrelation $\rho$ fällt die Varianz des Mittelwerts nie unter den Anteil $\rho$ der Varianz eines einzelnen Mitglieds (Übung :exref:`ex_mdl-random-variables-using-textrmvar-x-y` in :numref:`sec_mdl-joint-distributions`).

<!--en:b69318d0-->
Die Grundideen dieses Abschnitts begegnen uns im Deep Learning immer wieder, wenn wir eine Verteilung nur bis auf eine Normierungskonstante kennen. Mit der unteren Evidenzschranke trainieren wir Variationsautoencoder :cite:`Kingma.Welling.2014` (:numref:`sec_diffusion-latent`), deren Encoder-Netz $q$ ausgibt, sowie diskrete Diffusionsmodelle (:numref:`sec_diffusion-discrete`). Mit Wichtigkeitsgewichten und effektiver Stichprobengröße können Policy-Gradient-Verfahren frühere Trajektorien wiederverwenden (:numref:`sec_ppo`). Lassen wir bei gradientengestützten Vorschlägen den Annahme- und Ablehnungsschritt weg, erhalten wir die Langevin-Sampler, auf denen scorebasierte generative Modelle beruhen (:numref:`sec_diffusion-denoising`). Jedes Sampling-Verfahren liefert letztlich einen empirischen Mittelwert aus endlich vielen Ziehungen. Im nächsten Abschnitt untersuchen wir, wie eng sich solche Mittelwerte um ihre Erwartungswerte konzentrieren, wenn die Ziehungen unabhängig sind.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:8a485d4e-->
Trainieren wir ein Modell auf einen einzigen Parametervektor, behandeln wir diese Punktschätzung, als wäre sie mit Sicherheit bekannt. Fern der Trainingsdaten können sich die Vorhersagen plausibler Parametervektoren um Größenordnungen unterscheiden. Die posterior-prädiktive Mittelung kann dann die Wahrscheinlichkeit seltener Ereignisse um ein Mehrfaches verschieben. Für diesen Mittelwert müssen wir über eine A-posteriori-Verteilung integrieren, die wir nur bis auf ihre Normierungskonstante kennen. Alle vier Verfahren umgehen die unbekannte Konstante durch Verhältnisse oder Differenzen, in denen sie wegfällt. Die Laplace-Approximation nutzt die lokale Krümmung am Modus, Rejection- und Importance-Sampling nutzen Dichteverhältnisse $\widetilde p/q$ und Metropolis-Hastings nutzt Annahmeverhältnisse $\widetilde p(\boldsymbol\theta')/\widetilde p(\boldsymbol\theta)$. Die variationelle Inferenz optimiert eine Zielfunktion, die bis auf eine additive Konstante der negativen Rückwärts-KL-Divergenz entspricht. Die Sampler liefern eigene Diagnosegrößen, nämlich die ESS der Gewichte beim Importance-Sampling sowie Split-$\widehat R$, Autokorrelationszeit $\kappa$ und MCSE bei MCMC. Gaußsche Näherungen müssen wir hingegen mit Ziehungen vergleichen. Obwohl die vollständige bayessche Inferenz für große neuronale Netze kaum berechenbar ist, begegnen uns die untere Evidenzschranke, Wichtigkeitsgewichte und Langevin-Dynamik im Buch immer wieder beim Umgang mit unnormierten Verteilungen.

<!--en:2294243a-->
## Übungen

<!--en:5840c830-->
1. **Annahme bei Metropolis-Hastings.** Mit der Annahmewahrscheinlichkeit :eqref:`eq_mdl-bayes-mh` lässt sich aus jeder Vorschlagsverteilung eine Kette mit der A-posteriori-Verteilung als Zielverteilung konstruieren.
    1. Auf einem kontinuierlichen Zustandsraum hat der Übergangskern einen Dichteanteil $g(\boldsymbol\theta'\mid\boldsymbol\theta)\alpha(\boldsymbol\theta,\boldsymbol\theta')$ und ein Atom am aktuellen Zustand $\boldsymbol\theta$ mit Masse $r(\boldsymbol\theta)=1-\int g(\boldsymbol\theta'\mid\boldsymbol\theta)\alpha(\boldsymbol\theta,\boldsymbol\theta')\,d\boldsymbol\theta'$, der Wahrscheinlichkeit einer Ablehnung. Zeigen Sie unter Berücksichtigung des Atoms, dass die A-posteriori-Verteilung stationär ist. *Hinweis:* Nach einem Schritt trägt das Atom $\pi(\boldsymbol\theta')r(\boldsymbol\theta')$ zur Dichte bei $\boldsymbol\theta'$ bei.
    1. Ein Unabhängigkeitssampler schlägt unabhängig vom aktuellen Zustand $\boldsymbol\theta'\sim q$ vor. Zeigen Sie, dass seine Annahmewahrscheinlichkeit $\min(1,w(\boldsymbol\theta')/w(\boldsymbol\theta))$ beträgt, mit dem Wichtigkeitsgewicht $w$ aus :eqref:`eq_mdl-bayes-importance`. Was tut die Kette, nachdem sie einen Zustand erreicht hat, dessen Gewicht weit über den Gewichten typischer Vorschläge liegt?
    1. Die Barker-Regel nimmt einen symmetrischen Vorschlag mit Wahrscheinlichkeit $\widetilde p(\boldsymbol\theta')/(\widetilde
       p(\boldsymbol\theta)+\widetilde p(\boldsymbol\theta'))$ an. Zeigen Sie, dass sie das detaillierte Gleichgewicht erfüllt. Geben Sie ihre Übergangsmatrix für die Kette mit zwei Zuständen aus dem Text an. Zeigen Sie, dass die Metropolis-Kette von beiden Zuständen aus häufiger wechselt. Für eine Kette mit zwei Zuständen beträgt die Autokorrelation jeder Funktion des Zustands $\rho_k=\lambda^k$, wobei $\lambda$ der zweite Eigenwert der Übergangsmatrix ist. Berechnen Sie $\lambda$ und $\kappa=(1+\lambda)/(1-\lambda)$ für beide Regeln. Welche ergibt bei gleicher Schrittzahl den kleineren Monte-Carlo-Fehler? *Hinweis:* Bei Barker lauten die Einträge außerhalb der Diagonalen $3/4$ und $1/4$, bei Metropolis $1$ und $1/3$.
   :exlabel:`ex_mdl-bayesian-computation-mh-acceptance`

<!--en:dbfc9132-->
1. [code] **Vorschlagsskala.** Die Funktion `run_chains` multipliziert den Cholesky-Faktor von `laplace_cov` mit einem Skalenfaktor, im Text $1.1$.
    1. Variieren Sie den Skalenfaktor über $\{0.05,0.2,0.5,1.1,1.7,2.4,3,10\}$. Behalten Sie die 12 000 Schritte pro Kette und die Aufwärmphase bei. Erfassen Sie für jeden Wert die Annahmerate, Split-$\widehat R$ und die kleinste ESS aus `chain_ess` geteilt durch die $48{,}000$ Auswertungen der logarithmierten gemeinsamen Dichte aller vier Ketten. Tragen Sie die ESS pro Auswertung gegen die Annahmerate auf. In welchem Bereich der Annahmeraten erreicht die ESS pro Auswertung mindestens die Hälfte ihres Maximums?
    1. Für eine $d$-dimensionale Standardnormalverteilung als Ziel und die Vorschlagsverteilung $\mathcal N(\boldsymbol\theta,(\ell^2/d)\mathbf I)$ zeigen :citet:`Roberts.Gelman.Gilks.1997`, dass für $d\to\infty$ die Effizienz bei $\ell\approx2.38$ am größten ist. Die Annahmerate beträgt dort etwa $0.234$. Die A-posteriori-Verteilung der logistischen Regression liegt nahe an ihrer Laplace-Approximation. Mit dem Cholesky-Faktor lässt sie sich daher näherungsweise standardisieren. Welche Skala legt dieses Ergebnis für $d=2$ nahe? Wie schneiden die beste Skala Ihrer Versuchsreihe und ihre Annahmerate im Vergleich ab?
    1. Welche Diagnosegrößen aus dem Text schlagen beim Lauf mit Skala $0.05$ an? Gehört die Annahmerate dazu? Wie weit liegt der A-posteriori-Mittelwert dieses Laufs vom Gittermittelwert entfernt, gemessen in Einheiten seines eigenen MCSE?
   :exlabel:`ex_mdl-bayesian-computation-proposal-scale`

<!--en:b3a197bb-->
1. **Laplace-Approximation und Parametrisierung.** Eine Münze zeigt 9-mal Kopf und 4-mal Zahl. Bei gleichförmiger A-priori-Verteilung hat die Kopfwahrscheinlichkeit $\theta$ die A-posteriori-Verteilung $\mathrm{Beta}(10,5)$ mit Mittelwert $2/3$, Standardabweichung $0.118$ und zentralem $95\%$-Intervall $[0.419,0.872]$.
    1. Wie im Text berechnet, hat die Laplace-Approximation :eqref:`eq_mdl-bayes-laplace` im Parameter $\theta$ den Mittelwert $9/13$ und die Standardabweichung $0.128$. Geben Sie ihr zentrales $95\%$-Intervall und die Wahrscheinlichkeit an, die sie außerhalb von $[0,1]$ zuweist.
    1. In den Log-Odds $\eta=\log(\theta/(1-\theta))$ ist die A-posteriori-Dichte proportional zu $\operatorname{sigmoid}(\eta)^{10}(1-\operatorname{sigmoid}(\eta))^5$, weil die Jacobi-Determinante $d\theta/d\eta=\theta(1-\theta)$ auf jeder Seite einen Zähler hinzufügt. Zeigen Sie, dass der Modus bei $\eta=\log2$ liegt. Berechnen Sie dort die Krümmung der Log-Dichte und transformieren Sie das zentrale $95\%$-Intervall der Laplace-Approximation zurück auf $\theta$.
    1. Welche der beiden Näherungen liegt näher an der exakten A-posteriori-Verteilung? Warum hängt die Laplace-Approximation von der Parametrisierung ab, die exakte A-posteriori-Wahrscheinlichkeit eines Ereignisses wie $\theta>1/2$ dagegen nicht?
   :exlabel:`ex_mdl-bayesian-computation-laplace-parameterization`

<!--en:66c19aae-->
1. [code] **Abgeschwächte Vorhersagen.** ● Unter der Laplace-Approximation :eqref:`eq_mdl-bayes-laplace` ist der Logit $a=\mathbf z_\star^\top\boldsymbol\theta$ an einer Testeingabe mit $\mathbf z_\star=(1,x_\star)^\top$ gaußverteilt mit Mittelwert $\mu_a=\mathbf z_\star^\top\boldsymbol\theta_{\mathrm{MAP}}$ und Varianz $s_a^2=\mathbf z_\star^\top\mathbf H^{-1}\mathbf z_\star$. Die Probit-Approximation $\operatorname{sigmoid}(a)\approx\Phi(\lambda a)$ mit $\lambda^2=\pi/8$, wobei $\Phi$ die Verteilungsfunktion der Standardnormalverteilung ist, liefert für das Vorhersageintegral :eqref:`eq_mdl-bayes-predictive` eine geschlossene Form :cite:`MacKay.1992`.
    1. Zeigen Sie, dass $\mathbb{E}[\Phi(\lambda a)]=\Phi\bigl(\lambda\mu_a/\sqrt{1+\lambda^2s_a^2}\bigr)$ für $a\sim\mathcal N(\mu_a,s_a^2)$ gilt. *Hinweis:* Schreiben Sie $\Phi(\lambda a)$ als $P(U\le\lambda a)$ für eine unabhängige standardnormalverteilte Zufallsvariable $U$. Folgern Sie, dass die abgeschwächte Vorhersage $\operatorname{sigmoid}\bigl(\mu_a/\sqrt{1+\pi s_a^2/8}\bigr)$ zwischen $\operatorname{sigmoid}(\mu_a)$ und $1/2$ liegt.
    1. Berechnen Sie die abgeschwächten Vorhersagen bei $x_\star\in\{-3,2,7\}$ aus `theta_map` und `laplace_cov`. Vergleichen Sie sie mit den Plug-in- und posterior-prädiktiven Werten, die Ihr Lauf ausgibt.
    1. Berechnen Sie an denselben Eingaben drei Größen: die Plug-in-Vorhersage $\operatorname{sigmoid}(\mathbf z_\star^\top\boldsymbol\theta_{\mathrm{MAP}})$, die Vorhersage am A-posteriori-Mittelwert $\operatorname{sigmoid}(\mathbf z_\star^\top\mathbb{E}[\boldsymbol\theta\mid\mathcal D])$ und die posterior-prädiktive Vorhersage $\mathbb{E}[\operatorname{sigmoid}(\mathbf z_\star^\top\boldsymbol\theta)\mid\mathcal D]$. Welcher der beiden Schritte, vom Modus zum Mittelwert und von dort zur gemittelten Vorhersage, verschiebt die Vorhersage von $1/2$ weg und welcher zu diesem Wert hin? Warum sind beide Schritte bei $x_\star=2$ klein, bei $x_\star=-3$ oder $x_\star=7$ aber nicht? Was folgt daraus für den Einsatz der Laplace-Approximation zur Abschwächung übermäßig sicherer Vorhersagen?
   :exlabel:`ex_mdl-bayesian-computation-moderated-predictions`

<!--en:c3417d5d-->
1. [code] **Vorhersage der ESS der Gewichte.** ● Nach der Proposition aus dem Text nähert sich $N_{\mathrm{eff}}/N$ dem Wert $(\mathbb{E}_q[w])^2/\mathbb{E}_q[w^2]$ an, wenn die Zahl $N$ der Ziehungen wächst.
    1. Die normierte Zielverteilung in einer Dimension sei $\mathcal N(\mu,s_p^2)$, die Vorschlagsverteilung $\mathcal N(\nu,s_q^2)$. Es gelte $r=s_q/s_p$. Zeigen Sie, dass für $s_q^2>s_p^2/2$ die Gleichung

       $$\frac{\mathbb{E}_q[w^2]}{(\mathbb{E}_q[w])^2}=\frac{r^2}{\sqrt{2r^2-1}}\exp\!\left(\frac{(\mu-\nu)^2}{2s_q^2-s_p^2}\right),$$

       gilt und dass das Verhältnis für $s_q^2\le s_p^2/2$ unendlich ist. Welches $r$ minimiert das Verhältnis, wenn $\mu=\nu$ gilt?
    1. Nähern Sie die A-posteriori-Verteilung durch unabhängige Gauß-Verteilungen mit den Mittelwerten und Standardabweichungen vom Gitter an. Vernachlässigen Sie die Korrelation, sodass das Verhältnis über die Koordinaten faktorisiert. Sagen Sie die ESS der 20 000 Ziehungen mit der A-priori-Verteilung als Vorschlagsverteilung vorher und vergleichen Sie sie mit dem ausgegebenen Wert. Wiederholen Sie dies für die A-priori-Vorschläge mit $\tau=0.5$ und $\tau=10$. Setzen Sie dazu `prior_var` $=\tau^2$ in der Modellzelle, führen Sie die Zellen bis einschließlich der Zelle zum Importance-Sampling erneut aus und vergleichen Sie Vorhersage und Messung. Welcher Faktor der Formel macht eine schmale A-priori-Verteilung zu einer schlechten Vorschlagsverteilung und welcher eine breite?
    1. Für welche Koordinaten Ihres Laufs hat die Vorschlagsverteilung $\mathcal N(\boldsymbol\theta_{\mathrm{MAP}},0.3\,\mathbf I)$ unter derselben Näherung unendliche Gewichtsvarianz? Führen Sie zehn Läufe mit je 5000 Ziehungen aus und erfassen Sie die ESS, das größte normierte Gewicht und den geschätzten A-posteriori-Mittelwert. Warum zeigt ein Lauf mit plausibel erscheinender Schätzung noch nicht, dass die Vorschlagsverteilung geeignet ist?
    1. Rejection-Sampling aus der A-priori-Verteilung behält eine Ziehung mit Wahrscheinlichkeit $L(\boldsymbol\theta)/\max L$. Dabei ist $L$ die Likelihood und $\max L$ ihr Wert bei der Maximum-Likelihood-Schätzung (bei trennbaren Daten ihr Supremum 1). Messen Sie die Annahmerate für Ihre Daten und für Daten mit zehnmal so vielen Punkten. Erklären Sie den Rückgang mit $n$ anhand des Volumens der A-posteriori-Verteilung. Vergleichen Sie die Rate mit dem ESS-Anteil des Importance-Samplings aus derselben A-priori-Verteilung.
   :exlabel:`ex_mdl-bayesian-computation-weight-ess`

<!--en:36facba3-->
1. [code] **Schätzung der Evidenz.** ● Sei $q$ eine normierte Vorschlagsverteilung, sei $w=p(\mathcal D,\boldsymbol\theta)/q(\boldsymbol\theta)$ und sei $\hat Z_N=\frac1N\sum_{s=1}^Nw(\boldsymbol\theta_s)$ für unabhängige Ziehungen $\boldsymbol\theta_s\sim q$.
    1. Zeigen Sie, dass $\hat Z_N$ ein erwartungstreuer Schätzer für $p(\mathcal D)$ ist und dass $\mathbb{E}[\log\hat Z_N]\le\log p(\mathcal D)$ gilt. Zeigen Sie, dass $\mathbb{E}[\log\hat Z_1]$ die ELBO :eqref:`eq_mdl-bayes-vi-elbo` von $q$ ist und dass $\mathbb{E}[\log\hat Z_N]$ mit wachsendem $N$ nicht abnimmt :cite:`Burda.Grosse.Salakhutdinov.2016`.
    1. Die Funktion `log_joint` lässt die logarithmierte Normierungskonstante $-\log(2\pi\tau^2)$ der zweidimensionalen A-priori-Verteilung weg. Ergänzen Sie diese und berechnen Sie $\log p(\mathcal D)$ auf drei Wegen: durch Summieren der Dichte über das Gitter und Multiplikation mit der Fläche einer Gitterzelle, mithilfe von $\hat Z_N$ aus 5000 Ziehungen aus der Laplace-Vorschlagsverteilung und mit der Laplace-Näherung der Log-Evidenz $\log\widetilde p(\boldsymbol\theta_{\mathrm{MAP}})+\log2\pi-\tfrac12\log\det\mathbf H$. Die Laplace-Schätzung von $p(\mathcal D)$ fällt um einige Prozent zu niedrig aus, was etwa $0.04$ in $\log p(\mathcal D)$ entspricht. Welche Eigenschaft der A-posteriori-Verteilung, die in der Codezelle zum Gitter sichtbar wird, erklärt das Vorzeichen?
    1. Schätzen Sie die ELBO der Laplace-Approximation und der Mean-Field-Näherung mit jeweils $10^5$ Ziehungen und geben Sie den MCSE jeder Schätzung an. Nach :eqref:`eq_mdl-bayes-elbo-gap` ist der Abstand zwischen $\log p(\mathcal D)$ und der ELBO von $q$ gleich $D_{\textrm{KL}}(q\|p(\cdot\mid\mathcal D))$. Vergleichen Sie die beiden Gauß-Verteilungen anhand dieser Divergenz.
   :exlabel:`ex_mdl-bayesian-computation-evidence`

<!--en:7397c8b0-->
1. [code] **Eine A-posteriori-Verteilung mit zwei Modi.** ● Wir ziehen fünfzig Beobachtungen gemäß $y_i\sim\mathcal N(\theta^2,1)$ mit dem wahren Wert $\theta=1$. Der Parameter $\theta$ hat die A-priori-Verteilung $\mathcal N(0,4)$. Die Likelihood hängt von $\theta$ nur über $\theta^2$ ab. Ebenso bleibt der Verlust eines neuronalen Netzes unverändert, wenn die eingehenden Gewichte, der Bias und die ausgehenden Gewichte einer versteckten $\tanh$-Einheit gemeinsam das Vorzeichen wechseln. Die A-posteriori-Verteilung ist daher symmetrisch unter $\theta\mapsto-\theta$.
    1. Berechnen Sie die A-posteriori-Verteilung auf einem eindimensionalen Gitter. Geben Sie $\mathbb{E}[\theta\mid\mathcal D]$, $\mathbb{E}[\theta^2\mid\mathcal D]$ und das Verhältnis der A-posteriori-Dichte bei $\theta=0$ zu ihrem Wert an einem Modus an.
    1. Zeigen Sie, dass die Gauß-Verteilung, die die Vorwärtsdivergenz $D_{\textrm{KL}}(p\|q)$ minimiert, mit Mittelwert und Varianz von $p$ übereinstimmt. Wo liegt der Modus dieser Gauß-Verteilung bei der vorliegenden A-posteriori-Verteilung? Wie plausibel ist dieser Punkt unter $p$?
    1. Passen Sie drei lokale Näherungen an: die Laplace-Approximation am positiven Modus, eine eindimensionale Gauß-Verteilung mit dem ELBO-Aufstieg aus der Codezelle zur variationellen Inferenz ab $m=0.5$ und vier Metropolis-Ketten mit Vorschlagsstandardabweichung $0.2$ und Startpunkten $\theta\in\{0.5,1,1.5,2\}$. Starten Sie die Ketten dann erneut bei $\theta\in\{-2,-0.5,0.5,2\}$. Welche Verfahren liefern nur einen Modus? Welche Diagnosegrößen (Split-$\widehat R$ und ESS für die Ketten, ELBO für die variationelle Näherung) decken das Problem im jeweiligen Lauf auf?
    1. Welche A-posteriori-Erwartungswerte schätzt eine auf einen Modus beschränkte Näherung dennoch korrekt? Ein Netz mit $h$ versteckten $\tanh$-Einheiten erlaubt für jede Einheit den Vorzeichenwechsel aus dieser Übung und zusätzlich die $h!$ Permutationen der versteckten Einheiten (Übung :exref:`ex_mdl-maximum-likelihood-relu-symmetries`). Wie viele äquivalente Kopien eines Modus hat seine A-posteriori-Verteilung bei isotropem Gauß-Prior? Was folgt daraus für die Näherung der A-posteriori-Verteilung eines neuronalen Netzes mit einem der Verfahren aus diesem Abschnitt?
   :exlabel:`ex_mdl-bayesian-computation-two-modes`

<!--en:c17f3a86-->
1. [code] **Randverhalten der A-posteriori-Verteilung bei logistischer Regression.** ●● Ein zentraler Grenzwertsatz für Importance-Sampling setzt ein endliches zweites Moment der Gewichte voraus. Die Laplace-Vorschlagsverteilung in der Codezelle zum Importance-Sampling erreicht meist eine hohe ESS. Durchgehend bezeichnet $\tau^2=$ `prior_var` $=4$ die A-priori-Varianz in der A-posteriori-Verteilung :eqref:`eq_mdl-bayes-logistic-posterior`.
    1. Fixieren Sie einen Einheitsvektor $\mathbf v$ und setzen Sie $\boldsymbol\theta=t\mathbf v$. Zeigen Sie, dass die Log-Likelihood in :eqref:`eq_mdl-bayes-logistic-posterior` den Wert $-c(\mathbf v)t+O(1)$ für $t\to\infty$ hat. Dabei ist $c(\mathbf v)$ die Summe von $|\mathbf z_i^\top\mathbf v|$ über die Beobachtungen mit $y_i=1$ und $\mathbf z_i^\top\mathbf v<0$ oder mit $y_i=0$ und $\mathbf z_i^\top\mathbf v>0$. Für welche Datensätze gilt $c(\mathbf v)>0$ für jedes $\mathbf v$?
    1. Folgern Sie, dass für eine gaußsche Vorschlagsverteilung $\mathcal N(\mathbf m,\mathbf S)$ die Beziehung $\mathbb{E}_q[w^2]=\infty$ gilt, wenn ein Eigenwert von $\mathbf S$ kleiner als $\tau^2/2$ ist, und $\mathbb{E}_q[w^2]<\infty$, wenn jeder Eigenwert von $\mathbf S$ größer als $\tau^2/2$ ist. Welche der beiden Vorschlagsverteilungen aus der Codezelle zum Importance-Sampling hat Gewichte mit endlicher Varianz? Vergleichen Sie Ihre Antwort mit dem Ergebnis der Formel aus Übung :exref:`ex_mdl-bayesian-computation-weight-ess`a für die Laplace-Vorschlagsverteilung, wenn die A-posteriori-Verteilung durch eine Gauß-Verteilung angenähert wird.
    1. Bestimmen Sie entlang jedes Eigenvektors von `laplace_cov` in beiden Richtungen den Mahalanobis-Abstand von $\boldsymbol\theta_{\mathrm{MAP}}$, ab dem $\widetilde p^2/q$ seinen Wert bei $\boldsymbol\theta_{\mathrm{MAP}}$ übersteigt. Wie wahrscheinlich ist es, dass eine einzelne Ziehung aus der Laplace-Vorschlagsverteilung so weit außen liegt? Bringen Sie Ihre Antwort mit der ESS in Einklang, die Ihr Lauf ausgibt.
    1. :citet:`Geweke.1989` konstruiert Vorschlagsdichten für das Importance-Sampling aus multivariaten Student-$t$-Näherungen am A-posteriori-Modus. Verwenden Sie eine Student-$t$-Vorschlagsverteilung mit $3$ Freiheitsgraden, Lageparameter $\boldsymbol\theta_{\mathrm{MAP}}$ und Skalenmatrix `laplace_cov`. Zeigen Sie, dass ihre Gewichte endliche Varianz haben. Vergleichen Sie sie über 20 Läufe mit der Laplace-Vorschlagsverteilung anhand der ESS und der Schätzung der Randwahrscheinlichkeit $P(\theta_1>\bar\theta_1+3s_1\mid\mathcal D)$. Dabei sind $\bar\theta_1$ und $s_1$ Gittermittelwert und Gitterstandardabweichung der Steigung. Berechnen Sie einen Referenzwert auf einem doppelt so breiten Gitter. Probieren Sie auch die defensive Mischung aus dem Text aus und erklären Sie, warum ihre Gewichte beschränkt sind.
   :exlabel:`ex_mdl-bayesian-computation-posterior-tails`

<!--en:fcb38077-->
1. [code] **Langevin-Vorschläge.** ●● Der Metropolis-korrigierte Langevin-Algorithmus schlägt $\boldsymbol\theta'=\boldsymbol\theta+\tfrac{\epsilon^2}{2}\mathbf
   C\nabla\log\widetilde p(\boldsymbol\theta)+\epsilon\,\mathbf
   C^{1/2}\boldsymbol\xi$ mit $\boldsymbol\xi\sim\mathcal N(\mathbf
   0,\mathbf I)$ und einem Vorkonditionierer $\mathbf C$ vor. Verwenden Sie für $\mathbf C$ die Laplace-Kovarianz $\mathbf H^{-1}$ und für $\mathbf C^{1/2}$ ihren Cholesky-Faktor.
    1. Geben Sie die Vorschlagsdichte $g(\boldsymbol\theta'\mid\boldsymbol\theta)$ und den Hastings-Faktor der Annahmeregel :eqref:`eq_mdl-bayes-mh` an. Warum ist die Vorschlagsverteilung nicht symmetrisch?
    1. Implementieren Sie den Sampler mit `grad_log_joint` und führen Sie vier Ketten für $\epsilon\in\{0.5,1,1.5\}$ aus. Geben Sie die Annahmerate, Split-$\widehat R$ und die ESS pro Gradientenauswertung an. Vergleichen Sie diese mit der ESS der Irrfahrt pro Auswertung der logarithmierten gemeinsamen Dichte.
    1. Ohne Annahmeschritt entspricht die Kette dem unkorrigierten Langevin-Algorithmus. Führen Sie ihn mit $\epsilon=1$ aus und vergleichen Sie seinen Mittelwert und seine Standardabweichungen mit den Gitterwerten. Welchen der beiden Fehler beseitigt der Metropolis-Schritt, die Verzerrung durch die Diskretisierung oder den Monte-Carlo-Fehler?
   :exlabel:`ex_mdl-bayesian-computation-mala`

<!--en:948b0387-->
[Diskussion](https://d2l.discourse.group/t/36833)

<!--en:814a7aa8-->
Bayessche Rechenverfahren: Laplace-Approximation, Importance-Sampling, MCMC und variationelle Inferenz<br>**posterior-prädiktive Vorhersage · Laplace-Approximation · Importance-Sampling und ESS · Metropolis-Hastings · Diagnose der Durchmischung · variationelle Inferenz**

<!--en:97f05614-->
::: {.slide title="Warum Punktschätzungen fern der Trainingsdaten versagen"}

<!--en:25d9aa8a-->
[Motivation und Ausgangsproblem]{.kicker}

<!--en:62ca0561-->
- **Eindimensionale logistische Regression mit $n = 20$ Punkten ($x_i \in [0, 4]$):** Im Trainingsbereich bei $x_\star = 2$ stimmen MAP-Plug-in ($0.524$) und A-posteriori-Mittelung ($0.529$) bis auf $0.01$ überein.

<!--en:977be4ff-->
- **Unterschiede um Größenordnungen außerhalb der Daten:** Bei $x_\star = -3$ reichen die Vorhersagen plausibler Parametervektoren von $4 \times 10^{-5}$ bis $0.09$.

<!--en:53f39350-->
- **Faktor fünf bei seltenem Ereignis:** Der posterior-prädiktive Mittelwert beträgt $0.019$, also **etwa $4.6\times$ so viel** wie der MAP-Plug-in-Wert ($0.0041$). Die Log-Odds-Kurve biegt außerhalb der Daten zur Null zurück.

<!--en:22c18a96-->
::: {.slide title="Posterior-prädiktive Vorhersage und unnormierte Zieldichte"}

<!--en:e8756717-->
[Das Integrationsproblem]{.kicker}

<!--en:251fba46-->
- **Unnormierte gemeinsame Dichte:** Bei der bayesschen logistischen Regression lassen sich $\widetilde{p}(\boldsymbol{\theta}) = p(\mathcal{D} \mid \boldsymbol{\theta})p(\boldsymbol{\theta})$ und die Ableitungen $\nabla \log \widetilde{p}, \mathbf{H}(\boldsymbol{\theta})$ in Zeit $O(n)$ auswerten.

<!--en:bd975028-->
- **Wie $p(\mathcal{D})$ wegfällt:** Die Evidenz $p(\mathcal{D}) = \int \widetilde{p}(\boldsymbol{\theta})\,d\boldsymbol{\theta}$ lässt sich kaum berechnen. Laplace-Approximation, Importance-Sampling, MCMC und variationelle Inferenz nutzen daher Verhältnisse oder Differenzen, in denen $p(\mathcal{D})$ wegfällt.

<!--en:cd58bbd6-->
::: {.slide title="Steigung und Krümmung der logarithmierten A-posteriori-Dichte"}

<!--en:5d6bf713-->
[Gradient und negative Hesse-Matrix]{.kicker}

<!--en:d1c04b43-->
Ableiten von $\log \widetilde{p}(\boldsymbol{\theta})$ nach $\boldsymbol{\theta}$ liefert den Gradienten und die negative Hesse-Matrix für die Optimierung, die Laplace-Approximation und gradientengestützte Sampler:

<!--en:7ef297a2-->
- **Gradient aus Residuen:** Jede Beobachtung $i$ zieht $\boldsymbol{\theta}$ entlang $\mathbf{z}_i$ proportional zu ihrem Vorhersagefehler $y_i - p_i$. Die A-priori-Verteilung zieht $\boldsymbol{\theta}$ radial zum Ursprung.

<!--en:9a7d3ff3-->
- **Mit Unsicherheit gewichtete Krümmung:** Da $p_i(1-p_i) \le 1/4$ bei $p_i = 1/2$ maximal ist, tragen Beispiele nahe der Entscheidungsgrenze am stärksten zur Krümmung $\mathbf{H}(\boldsymbol{\theta})$ bei.

<!--en:78511512-->
- **Strenge Log-Konkavität:** Die A-priori-Verteilung addiert $\tau^{-2}\mathbf{I}$ und hebt jeden Eigenwert von $\mathbf{H}(\boldsymbol{\theta})$ auf mindestens $\tau^{-2} > 0$. Selbst bei linear trennbaren Daten hat die A-posteriori-Verteilung daher einen eindeutigen, endlichen globalen Modus $\boldsymbol{\theta}_{\mathrm{MAP}}$.

<!--en:1e3faa7b-->
::: {.slide title="Laplace-Approximation durch eine Gauß-Verteilung am Modus"}

<!--en:026901f8-->
[Verfahren 1: lokale quadratische Näherung]{.kicker}

<!--en:f7086e65-->
- **Taylor-Entwicklung zweiter Ordnung bei $\boldsymbol{\theta}_{\mathrm{MAP}}$:** Wegen $\nabla \log \widetilde{p}(\boldsymbol{\theta}_{\mathrm{MAP}}) = \mathbf{0}$ ergibt die Entwicklung um den Modus mit $\mathbf{H} = \mathbf{H}(\boldsymbol{\theta}_{\mathrm{MAP}})$:
  $$\log \widetilde{p}(\boldsymbol{\theta}) \approx \log \widetilde{p}(\boldsymbol{\theta}_{\mathrm{MAP}}) - \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}_{\mathrm{MAP}})^\top \mathbf{H}(\boldsymbol{\theta} - \boldsymbol{\theta}_{\mathrm{MAP}})$$

<!--en:b74c2367-->
- **Exponenzieren ergibt eine Gauß-Verteilung:** $p(\boldsymbol{\theta} \mid \mathcal{D}) \approx \mathcal{N}\bigl(\boldsymbol{\theta}_{\mathrm{MAP}},\, \mathbf{H}^{-1}\bigr)$.

<!--en:e2f8c750-->
- **Schiefe und Grenzen bleiben unberücksichtigt:** Für $\mathrm{Gamma}(4,1) \propto \theta^3 e^{-\theta}$ ($\theta > 0$) liefert die Laplace-Approximation $\mathcal{N}(3, 3)$ am Modus $\theta = 3$. Sie verfehlt den wahren Mittelwert $4$ und legt $4.2\%$ der Masse unterhalb von $\theta = 0$ ab.

<!--en:3aeb3658-->
::: {.slide title="Laplace-Approximation mit dem Newton-Verfahren"}

<!--en:e94df919-->
[Modus und Krümmung im Code]{.kicker}

<!--en:7c718fac-->
Zehn Newton-Schritte $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \mathbf{H}(\boldsymbol{\theta})^{-1}\nabla \log \widetilde{p}(\boldsymbol{\theta})$ vom Ursprung aus liefern $\boldsymbol{\theta}_{\mathrm{MAP}}$. Eine Matrixinversion $\mathbf{H}(\boldsymbol{\theta}_{\mathrm{MAP}})^{-1}$ ergibt die Kovarianz:

<!--en:5b033b47-->
- **Schrumpfung zum Ursprung:** Die A-priori-Verteilung verschiebt den geschätzten Modus $\boldsymbol{\theta}_{\mathrm{MAP}} = (-2.14, 1.12)$ gegenüber den wahren datenerzeugenden Parametern $\boldsymbol{\theta}^\star = (-3.0, 1.5)$ zur Null hin.

<!--en:cad30328-->
- **Starke negative Korrelation ($\rho = -0.86$):** Alle Trainingseingaben $x_i \in [0, 4]$ sind positiv. Steigt die Steigung $\theta_1$, muss der Achsenabschnitt $\theta_0$ sinken, damit die Entscheidungsgrenze $-\theta_0/\theta_1 \approx 2$ innerhalb der Punktwolke bleibt.

<!--en:75d1c848-->
::: {.slide title="Exakter Gittervergleich: Schiefe verschiebt den Mittelwert"}

<!--en:42213f8e-->
[Referenzwerte in wenigen Dimensionen]{.kicker}

<!--en:c3981d84-->
Bei $d = 2$ Parametern liefert die Auswertung von $\widetilde{p}(\boldsymbol{\theta})$ auf einem $241 \times 241$-Gitter über $\boldsymbol{\theta}_{\mathrm{MAP}} \pm 7$ Laplace-Standardabweichungen einen exakten Vergleichsmaßstab:

<!--en:9cbe6d37-->
- **Schiefe trennt Mittelwert und Modus:** Sobald alle Punkte fern der Grenze mit Wahrscheinlichkeit nahe $0$ oder $1$ klassifiziert sind, bleiben steilere Geraden plausibel. Die A-posteriori-Verteilung ist zu größeren Steigungen hin schief. Die wahre mittlere Steigung ($1.24$) liegt $+0.25$ Standardabweichungen über der MAP-Steigung ($1.12$).

<!--en:a9c6f4d2-->
- **Fluch der Dimensionalität:** Ein Gitter mit $m$ Punkten pro Achse erfordert $m^d$ Auswertungen ($241^2 \approx 5.8 \times 10^4$ in $2\text{D}$, aber $241^{100} \approx 10^{238}$ in $100\text{D}$). Gitterintegration dient daher nur in wenigen Dimensionen als Referenz.

<!--en:74ceb86c-->
::: {.slide title="Extrapolation mit Plug-in- und posterior-prädiktiver Vorhersage"}

<!--en:e35d94ad-->
[Abschwächung extremer Vorhersagen]{.kicker}

<!--en:7a59b12a-->
Der Vergleich des MAP-Plug-in-Werts $\operatorname{sigmoid}(\mathbf{z}_\star^\top \boldsymbol{\theta}_{\mathrm{MAP}})$ mit der exakten posterior-prädiktiven Vorhersage auf dem Gitter bei $x_\star \in \{-3, 2, 7\}$ zeigt, wie Parameterunsicherheit extreme Vorhersagen abschwächt:

<!--en:a90ce13e-->
- **Warum Unsicherheit Vorhersagen in Richtung $1/2$ verschiebt:** Für negative Logits $a = \mathbf{z}_\star^\top \boldsymbol{\theta} \ll 0$ ist $\operatorname{sigmoid}(a) \approx e^a$ streng konvex. Nach der Jensen-Ungleichung gilt $\mathbb{E}[e^a] > e^{\mathbb{E}[a]}$. Modelle mit flacheren Steigungen liefern bei $x_\star = -3$ viel größere Wahrscheinlichkeiten und dominieren den A-posteriori-Mittelwert.

<!--en:7ce15002-->
::: {.slide title="Woher die Verschiebung um 1.51 Nat bei x = -3 kommt"}

<!--en:db89576e-->
[Zerlegung der Log-Wahrscheinlichkeit]{.kicker}

<!--en:5591c3d6-->
- **Quadratischer Varianzbeitrag ($+2.87$ Nat):** Für einen gaußverteilten Logit $a \sim \mathcal{N}(\mu, s^2)$ gilt $\log \mathbb{E}[e^a] = \mu + s^2/2$. Damit steigt die Log-Wahrscheinlichkeit von $-6.06$ auf $-3.19$.

<!--en:893b8e82-->
- **Drei Korrekturen (insgesamt $-1.36$ Nat):** Die schiefebedingte Verschiebung vom Modus zum Mittelwert ($-0.57$), der kurze obere Rand der Logit-Verteilung ($-0.47$) und die Sättigung der Sigmoidfunktion $\operatorname{sigmoid}(a) < e^a$ ($-0.32$) ergeben netto einen Anstieg um $+1.51$ Nat ($4.6\times$).

<!--en:3dd40e9e-->
::: {.slide title="Rejection-Sampling und der Einbruch der Annahmerate in hohen Dimensionen"}

<!--en:a3b4dd2b-->
[Verfahren 2A: Annahme anhand einer Hüllfunktion]{.kicker}

<!--en:8022ba7c-->
Für eine handhabbare Vorschlagsverteilung $q(\boldsymbol{\theta})$ und eine Konstante $M$ mit $M q(\boldsymbol{\theta}) \ge \widetilde{p}(\boldsymbol{\theta})$ im gesamten Raum nimmt **Rejection-Sampling** jede Ziehung $\boldsymbol{\theta} \sim q$ mit Wahrscheinlichkeit $a(\boldsymbol{\theta}) = \frac{\widetilde{p}(\boldsymbol{\theta})}{M q(\boldsymbol{\theta})}$ an:

<!--en:7b4b33a5-->
- **Exakte Ziehungen ohne Kenntnis von $p(\mathcal{D})$:** Die angenommenen Ziehungen folgen exakt $\widetilde{p}(\boldsymbol{\theta})/p(\mathcal{D}) = p(\boldsymbol{\theta} \mid \mathcal{D})$. Die unbekannte Evidenz bestimmt nur die marginale Annahmerate $\int \frac{\widetilde{p}(\boldsymbol{\theta})}{M}\,d\boldsymbol{\theta} = \frac{p(\mathcal{D})}{M}$.

<!--en:cdd573bd-->
- **Exponentieller Rückgang mit $d$:** Die A-posteriori-Verteilung zieht sich um $\boldsymbol{\theta}^\star$ mit Rate $n^{-1/2}$ entlang jeder der $d$ Koordinaten zusammen. Ihr Volumen schrumpft relativ zur A-priori-Verteilung wie $n^{-d/2}$ ($30.5\%$ für $9\text{H}, 4\text{T}$ $\to$ $10\%$ für $90\text{H}, 40\text{T}$ $\to$ $3\%$ für $900\text{H}, 400\text{T}$).

<!--en:e028ef08-->
::: {.slide title="Selbstnormiertes Importance-Sampling"}

<!--en:f0912c34-->
[Verfahren 2B: alle Ziehungen neu gewichten]{.kicker}

<!--en:5e6ce204-->
**Importance-Sampling** behält jede Ziehung $\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_N \sim q$ und gewichtet sie mit dem unnormierten Verhältnis $w(\boldsymbol{\theta}) = \frac{\widetilde{p}(\boldsymbol{\theta})}{q(\boldsymbol{\theta})}$, statt Vorschläge zu verwerfen:

<!--en:3c3fd87d-->
Dabei sind $\bar{w}_s = \frac{w_s}{\sum_{r=1}^N w_r}$ die selbstnormierten Gewichte (numerisch stabil mit Log-Sum-Exp berechnet). Im Verhältnis kürzt sich $\mathbb{E}_q[w] = p(\mathcal{D})$ heraus:

<!--en:da80690f-->
::: {.slide title="Effektive Stichprobengröße und Chi-Quadrat-Divergenz"}

<!--en:a19461e1-->
[Degeneration der Gewichte erkennen]{.kicker}

<!--en:af76ffe2-->
- **Effektive Stichprobengröße (ESS):** Trägt eine Ziehung fast das gesamte Gewicht, sind $N$ gewichtete Ziehungen nur eine Ziehung wert. Die ESS misst das anhand der normierten Gewichte $\bar{w}_s$:
  $$N_{\mathrm{eff}} = \frac{1}{\sum_{s=1}^N \bar{w}_s^2}$$

<!--en:e9f691e7-->
- **Grenzwert bei großer Stichprobe:** Für $N \to \infty$ gilt $\frac{N_{\mathrm{eff}}}{N} \to \frac{(\mathbb{E}_q[w])^2}{\mathbb{E}_q[w^2]} = \frac{1}{1 + \chi^2(p \,\|\, q)} \ge \frac{p(\mathcal{D})}{M}$. Damit schneidet Importance-Sampling stets mindestens so gut ab wie Rejection-Sampling ($42.0\%$ gegenüber $30.5\%$ bei der Münze).

<!--en:cb2c40e0-->
- **Unentdeckte Masse:** Die empirische ESS misst die Ungleichheit der Gewichte unter den beobachteten Ziehungen. Ob $q$ alle Modi von $p$ erfasst, bleibt offen.

<!--en:a37cafea-->
::: {.slide title="Leichte Verteilungsränder und unendliche Gewichtsvarianz"}

<!--en:c609fb25-->
[Ränder der Vorschlagsverteilung bei logistischer Regression]{.kicker}

<!--en:caac2a71-->
Der Vergleich des breiten Gauß-Priors ($N = 20{,}000$) mit der gezielt angepassten Laplace-Verteilung ($N = 5{,}000$) für unsere logistische Regression zeigt ein Problem an den Verteilungsrändern:

<!--en:6175c610-->
- **Warum die Laplace-Vorschlagsverteilung unendliche Gewichtsvarianz hat ($\mathbb{E}_q[w^2] = \infty$):** Entlang jedes Strahls $\boldsymbol{\theta} = t\mathbf{v}$ fällt die logistische Log-Likelihood nur linear in $t$. Daher haben die Ränder der A-posteriori-Verteilung gaußsches Verhalten mit der Varianz *der A-priori-Verteilung* $\tau^2 = 4$. Folglich gilt $\int (\widetilde{p}^2 / q)\,d\boldsymbol{\theta} = \infty$, sobald ein Eigenwert der Vorschlagskovarianz $\mathbf{S}$ die Bedingung $\lambda_{\min}(\mathbf{S}) \le \tau^2/2 = 2$ erfüllt.

<!--en:5e3560f9-->
- **Abhilfe:** Beide Laplace-Eigenwerte ($1.17$ und $0.05$) liegen unter $2$. Bei wiederholten Läufen bricht die ESS deshalb gelegentlich drastisch ein. Eine **Student-$t$-Vorschlagsverteilung** mit schwereren Rändern oder eine **defensive Mischung** $q = 0.9\,\mathcal{N}(\boldsymbol{\theta}_{\mathrm{MAP}}, \mathbf{H}^{-1}) + 0.1\,\text{prior}$ liefert beschränkte Gewichte. Für die Mischung gilt $w \le 10$.

<!--en:0068e53c-->
::: {.slide title="Markov-Chain-Monte-Carlo und die Metropolis-Regel"}

<!--en:c229cc50-->
[Verfahren 3: lokale Irrfahrten]{.kicker}

<!--en:dcfb7a04-->
- **Stationäre Markov-Kette:** Anstelle einer einzigen globalen Vorschlagsverteilung $q$ konstruiert MCMC einen Übergangskern $T(\boldsymbol{\theta}' \mid \boldsymbol{\theta})$, der die A-posteriori-Verteilung invariant lässt: $\int \pi(\boldsymbol{\theta})T(\boldsymbol{\theta}' \mid \boldsymbol{\theta})\,d\boldsymbol{\theta} = \pi(\boldsymbol{\theta}')$.

<!--en:a5f7adb9-->
- **Annahmewahrscheinlichkeit bei Metropolis-Hastings:** Vom Zustand $\boldsymbol{\theta}$ aus einen Kandidaten $\boldsymbol{\theta}' \sim g(\cdot \mid \boldsymbol{\theta})$ vorschlagen und mit folgender Wahrscheinlichkeit annehmen:
  $$\alpha(\boldsymbol{\theta}, \boldsymbol{\theta}') = \min\!\left(1,\, \frac{\widetilde{p}(\boldsymbol{\theta}')\,g(\boldsymbol{\theta} \mid \boldsymbol{\theta}')}{\widetilde{p}(\boldsymbol{\theta})\,g(\boldsymbol{\theta}' \mid \boldsymbol{\theta})}\right)$$

<!--en:f10761b5-->
- **Die Evidenz kürzt sich im Verhältnis heraus:** Bei symmetrischen Vorschlägen $g(\boldsymbol{\theta}'\mid\boldsymbol{\theta}) = g(\boldsymbol{\theta}\mid\boldsymbol{\theta}')$ gelingen Schritte bergauf ($\widetilde{p}' \ge \widetilde{p}$) immer, Schritte bergab mit Wahrscheinlichkeit $\widetilde{p}'/\widetilde{p}$.

<!--en:a8090c4d-->
::: {.slide title="Warum die Kette bei Ablehnung den aktuellen Zustand wiederholen muss"}

<!--en:35279e42-->
[Metropolis-Beispiel mit zwei Zuständen]{.kicker}

<!--en:a61d3d6e-->
- **Zielverteilung $\pi = (1/4, 3/4)$ mit Zustandswechsel als Vorschlag:** Ist der Vorschlag stets der *andere* Zustand, führt der Schritt $1 \to 2$ bergauf ($\widetilde{p}_2/\widetilde{p}_1 = 3 \implies \alpha = 1$) und der Schritt $2 \to 1$ bergab ($\alpha = 1/3$).

<!--en:b3238733-->
- **Prüfung auf Stationarität:** Die Übergangsmatrix $\mathbf{T} = \begin{pmatrix}0 & 1 \\ 1/3 & 2/3\end{pmatrix}$ lässt $\pi$ invariant:
  $$\bigl(\tfrac{1}{4},\, \tfrac{3}{4}\bigr)\begin{pmatrix}0 & 1 \\ 1/3 & 2/3\end{pmatrix} = \bigl(\tfrac{1}{4},\, \tfrac{3}{4}\bigr)$$

<!--en:fc257f5c-->
- **Warum Wiederholungen zählen:** Die Selbstschleife $T_{22} = 2/3$ erfasst abgelehnte Schritte aus Zustand $2$. Ohne die Ablehnungen würden die Zustände gemäß $2, 1, 2, 1$ wechseln und die relativen Häufigkeiten gegen $(1/2, 1/2)$ konvergieren.

<!--en:b95ceddc-->
::: {.slide title="Detailliertes Gleichgewicht garantiert Stationarität der A-posteriori-Verteilung"}

<!--en:06ec9fa2-->
[Der grundlegende Satz für MCMC]{.kicker}

<!--en:07795ef0-->
- **Symmetrischer Fluss bei Metropolis-Hastings:** Für $\boldsymbol{\theta}' \ne \boldsymbol{\theta}$ gilt $\pi(\boldsymbol{\theta})g(\boldsymbol{\theta}'\mid\boldsymbol{\theta})\alpha(\boldsymbol{\theta}, \boldsymbol{\theta}') = \min\bigl\{\pi(\boldsymbol{\theta})g(\boldsymbol{\theta}'\mid\boldsymbol{\theta}),\, \pi(\boldsymbol{\theta}')g(\boldsymbol{\theta}\mid\boldsymbol{\theta}')\bigr\}$.

<!--en:116f41e8-->
- **Detailliertes Gleichgewicht $\implies$ Stationarität:** Integration beider Seiten über $\boldsymbol{\theta}$ ergibt $\int \pi(\boldsymbol{\theta})T(\boldsymbol{\theta}'\mid\boldsymbol{\theta})\,d\boldsymbol{\theta} = \pi(\boldsymbol{\theta}')$.

<!--en:46959de6-->
::: {.slide title="Vorkonditioniertes Metropolis-Verfahren in NumPy"}

<!--en:8c47a1bc-->
[Irrfahrt mit Cholesky-Skalierung]{.kicker}

<!--en:cf3d44b6-->
Der lokale gaußsche Vorschlag $\boldsymbol{\theta}' = \boldsymbol{\theta} + s\mathbf{L}\boldsymbol{\epsilon}$ nutzt den Cholesky-Faktor der Laplace-Kovarianz ($\mathbf{L}\mathbf{L}^\top = \mathbf{H}^{-1}$). So folgt die Irrfahrt der Korrelation $-0.86$ der A-posteriori-Verteilung:

<!--en:38f91ca2-->
Für $4$ Ketten mit weit gestreuten Startpunkten $\boldsymbol{\theta}_{\mathrm{MAP}} + 2\mathbf{L}\boldsymbol{\epsilon}$ und Skala $s = 1.1$ ergibt sich nach dem Verwerfen von $2{,}000$ Aufwärmschritten eine Annahmerate von etwa $50\%$:

<!--en:94435a23-->
::: {.slide title="Drei Diagnosegrößen für Durchmischung und Genauigkeit von MCMC"}

<!--en:3df29087-->
[Split-R-Hat, Autokorrelationszeit und MCSE]{.kicker}

<!--en:7062ad56-->
Die Zustände einer Markov-Kette sind zeitlich korreliert und starten außerhalb des Gleichgewichts. Deshalb braucht jeder MCMC-Lauf drei quantitative Prüfungen:

<!--en:da5e69cc-->
- **1. Split-$\widehat{R}$ (potentieller Skalenreduktionsfaktor):** $c$ Ketten der Länge $N$ in $2c$ Halbketten der Länge $N/2$ teilen. Die Varianz $B$ zwischen den Ketten mit der mittleren Varianz $W$ innerhalb der Ketten vergleichen:
  $$\widehat{R} = \sqrt{\frac{\frac{N-1}{N}W + \frac{1}{N}B}{W}} \qquad (\text{require } \widehat{R} < 1.01)$$

<!--en:ff766be4-->
- **2. Integrierte Autokorrelationszeit $\kappa$ und $N_{\mathrm{eff}}$:** Bei Schrittabstand $k$ und Autokorrelation $\rho_k$ gibt $\kappa = 1 + 2\sum_{k \ge 1} \rho_k$ an, wie viele korrelierte Schritte einer unabhängigen Ziehung entsprechen. Daraus folgt $N_{\mathrm{eff}} = \frac{cN}{\kappa}$. Für eine AR(1)-Kette $x_t = \phi x_{t-1} + \varepsilon_t$ gilt $\rho_k = \phi^k \implies \kappa = \frac{1+\phi}{1-\phi}$ ($\phi = 0.9 \implies \kappa = 19$; $\phi = 0.99 \implies \kappa = 199$).

<!--en:4cbf736e-->
- **3. Monte-Carlo-Standardfehler (MCSE):** Nach dem zentralen Grenzwertsatz für Markov-Ketten beträgt der Standardfehler des Kettenmittelwerts $\mathrm{MCSE} = \frac{\hat{\sigma}_{\mathrm{post}}}{\sqrt{N_{\mathrm{eff}}}}$.

<!--en:c9233858-->
::: {.slide title="MCMC-Konvergenz bei der logistischen Regression prüfen"}

<!--en:961ad551-->
[Diagnosegrößen und Verlaufsdiagramme in der Praxis]{.kicker}

<!--en:8723366f-->
- **Gute Durchmischung:** Es gilt Split-$\widehat{R} \le 1.002$, und die $4$ Ketten überlagern sich im Verlaufsdiagramm eng.

<!--en:b8e5d5fc-->
- **Kalibrierte Unsicherheit:** Bei $\rho_1 \approx 0.80$ gilt $\kappa \approx 10$ ($N_{\mathrm{eff}} \approx 4{,}000$ von $40{,}000$ Schritten). Der MCMC-Mittelwert liegt innerhalb von $\sim 1\ \mathrm{MCSE}$ um den exakten Gittermittelwert.

<!--en:d8b5a917-->
::: {.slide title="Passende Vorschlagsskala und Orientierung am Gradienten"}

<!--en:42d26c77-->
[Abwägung bei der Wahl der Schrittweite]{.kicker}

<!--en:e1137f85-->
Warum kann die Annahmerate allein nie Konvergenz belegen? Eine Wiederholung der $4$ Ketten mit den extremen Vorschlagsskalen $s = 0.05$ und $s = 10$ zeigt beide Probleme:

<!--en:048c27f1-->
- **Winzige Schritte ($s = 0.05$, Annahmerate $97\%$):** Fast jeder Schritt wird angenommen. Die Irrfahrt breitet sich aber so langsam aus, dass $\widehat{R} > 1.04$ gilt und $40{,}000$ Schritte nur $\sim 40$ effektive Ziehungen liefern.

<!--en:0070a280-->
- **Riesige Schritte ($s = 10$, Annahmerate $2\%$):** Vorschläge landen fast immer in Randbereichen geringer Dichte und werden abgelehnt. Die Kette verharrt Dutzende Schritte im selben Zustand.

<!--en:90fa679c-->
- **Skalierung in hohen Dimensionen und Langevin-Dynamik:** Das Metropolis-Verfahren mit Irrfahrt benötigt eine Vorschlagsvarianz $O(1/d)$ (angestrebte Annahmerate $23.4\%$). **Langevin-Vorschläge** $\boldsymbol{\theta}' = \boldsymbol{\theta} + \frac{\epsilon^2}{2}\mathbf{C}\nabla\log\widetilde{p}(\boldsymbol{\theta}) + \epsilon\mathbf{C}^{1/2}\boldsymbol{\xi}$ nutzen die Drift entlang des Gradienten, um sich gezielt bergauf zu bewegen.

<!--en:23b839cc-->
::: {.slide title="Variationelle Inferenz: Integration durch Optimierung"}

<!--en:166a35ac-->
[Verfahren 4: parametrische Maximierung der ELBO]{.kicker}

<!--en:540c1792-->
Die **variationelle Inferenz (VI)** wählt eine handhabbare Familie von Dichten $q_{\boldsymbol{\psi}}(\boldsymbol{\theta})$ und optimiert $\boldsymbol{\psi}$, um die KL-Divergenz zur A-posteriori-Verteilung zu minimieren. So kommt sie ohne korrelierte Ziehungen aus. Mit $\boldsymbol{\theta}$ in der ELBO-Identität aus §{sec_mdl-latent-em-elbo} als latenter Variable ergibt sich:

<!--en:ff6de8bc-->
- **Die unbekannte Evidenz fällt weg:** Da $\log p(\mathcal{D})$ nicht von $\boldsymbol{\psi}$ abhängt, senkt jeder Anstieg der berechenbaren ELBO um $1$ Nat ($\mathcal{L}(\boldsymbol{\psi})$) die Divergenz $D_{\mathrm{KL}}(q_{\boldsymbol{\psi}} \,\|\, p(\cdot \mid \mathcal{D}))$ um $1$ Nat.

<!--en:6652d74c-->
- **Reparametrisierungsgradienten:** Die Darstellung $\boldsymbol{\theta} = \mathbf{m} + \mathbf{s} \odot \boldsymbol{\epsilon}$ mit $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ erlaubt es, nach $\boldsymbol{\psi} = (\mathbf{m}, \log\mathbf{s})$ direkt innerhalb von $\log \widetilde{p}$ abzuleiten. So entstehen erwartungstreue stochastische Gradienten $\nabla_{\mathbf{m}}\mathcal{L} = \mathbb{E}[\nabla_{\boldsymbol{\theta}}\log \widetilde{p}]$ und $\nabla_{\log\mathbf{s}}\mathcal{L} = \mathbb{E}[\nabla_{\boldsymbol{\theta}}\log \widetilde{p} \odot \mathbf{s} \odot \boldsymbol{\epsilon}] + \mathbf{1}$ mit geringer Varianz.

<!--en:f9404cd1-->
::: {.slide title="Rückwärts-KL (modussuchend) und Vorwärts-KL"}

<!--en:5ee82e8f-->
[Asymmetrie der KL-Divergenz]{.kicker}

<!--en:e3ada351-->
- **Rückwärts-KL ($D_{\mathrm{KL}}(q \,\|\, p) = \int q \log\frac{q}{p}$, bei VI):** Sie wächst auf $+\infty$, wo $p(\boldsymbol{\theta} \mid \mathcal{D}) \approx 0$ und $q(\boldsymbol{\theta}) > 0$ gelten. Diese Strafe **erzwingt Nullen** und wirkt modussuchend. Sie zwingt $q$, sich auf einen einzelnen Modus zu konzentrieren und die Breite des Trägers zu unterschätzen.

<!--en:3615bacf-->
- **Vorwärts-KL ($D_{\mathrm{KL}}(p \,\|\, q) = \int p \log\frac{p}{q}$):** Sie divergiert, wo $p > 0$ und $q \approx 0$ gelten. Diese Strafe erzwingt eine **Abdeckung der Masse**. Dadurch überbrückt $q$ alle Modi und legt seinen Gipfel in ein leeres Tal. Die Berechnung erfordert Erwartungswerte unter der schwer handhabbaren Verteilung $p$.

<!--en:038c5e34-->
::: {.slide title="Warum Mean-Field-VI die A-posteriori-Unsicherheit unterschätzt"}

<!--en:577f4f42-->
[Bedingte und marginale Varianz]{.kicker}

<!--en:7cecf3a6-->
- **Mean-Field-Satz für eine korrelierte Gauß-Verteilung:** Hat $p = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1})$ die Präzisionsmatrix $\boldsymbol{\Lambda}$ und Korrelation $\rho$, so liefert die Minimierung von $D_{\mathrm{KL}}(q \,\|\, p)$ über faktorisierte Gauß-Verteilungen $q = \mathcal{N}(\mathbf{m}, \operatorname{diag}(\mathbf{s}^2))$ den Mittelwert $\mathbf{m} = \boldsymbol{\mu}$ und:
  $$s_j^2 = \frac{1}{\Lambda_{jj}} = \sigma_j^2(1 - \rho^2)$$

<!--en:28f54ae1-->
- **Schnittbreite und Projektionsbreite:** Eine achsenparallele Ellipse kann sich nicht drehen. Die Rückwärts-KL verkleinert daher jedes $s_j$ um den Faktor $\sqrt{1 - \rho^2}$ ($0.51\times$ für $\rho = -0.86$), um Ecken geringer Dichte zu vermeiden:

<!--en:69546134-->
::: {.slide title="Näherungen der Randverteilung der Steigung im Vergleich"}

<!--en:ffafbd96-->
[Die Ergebnisse im Vergleich]{.kicker}

<!--en:28a3740b-->
- **Metropolis (Histogramm):** Gibt die exakte Randverteilung vom Gitter einschließlich ihrer Rechtsschiefe und ihrer wahren Breite zuverlässig wieder.

<!--en:2cfd0af2-->
- **Laplace (orange):** Am MAP-Modus ($1.12$) links vom A-posteriori-Mittelwert ($1.24$) zentriert und um $4\%$ zu schmal.

<!--en:6c1b5082-->
- **Mean-Field-VI (grün):** Erfasst den A-posteriori-Mittelwert gut ($1.22$), schrumpft aber auf die Hälfte der wahren marginalen Breite ($s_1 = 0.25$ gegenüber $0.48$) und setzt $\rho = 0$ voraus.

<!--en:d2a43501-->
::: {.slide title="Verständnisfragen zu bayesschen Rechenverfahren"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:1359b166-->
1. **Autokorrelation und effektive Stichprobengröße:** Eine MCMC-Kette mit $N = 19{,}000$ Schritten nach der Aufwärmphase verhält sich wie ein AR(1)-Prozess mit Autokorrelation $\phi = 0.90$ bei einem Schritt Abstand. Wie groß ist ihre integrierte Autokorrelationszeit $\kappa$, und wie vielen unabhängigen Ziehungen entspricht die Kette?

<!--en:d6f28c6e-->
2. **Diagnose bei einer Annahmerate von $97\%$:** Warum erkundet eine Metropolis-Irrfahrt mit einer Annahmerate von $97\%$ die A-posteriori-Verteilung mit ziemlicher Sicherheit unzureichend? Welche Diagnosegröße zeigt das?

<!--en:b5c9e527-->
3. **Varianzschrumpfung bei Mean-Field-VI:** Ein $2\text{D}$-Gauß-Posterior hat die marginalen Standardabweichungen $(\sigma_1, \sigma_2) = (2.0, 1.0)$ und die Korrelation $\rho = -0.80$. Gegen welche Standardabweichungen $(s_1, s_2)$ konvergiert die variationelle Mean-Field-Inferenz?

<!--en:557b98b6-->
**Lösungen:**

<!--en:5fc2f7e8-->
1. Es gilt $\kappa = \frac{1 + \phi}{1 - \phi} = \frac{1.90}{0.10} = 19$. Die Kette entspricht also $N_{\mathrm{eff}} = \frac{19{,}000}{19} = 1{,}000$ unabhängigen Ziehungen. Die Monte-Carlo-Standardfehler verringern sich um den Faktor $\sqrt{1000} \approx 31.6$ statt um $\sqrt{19000} \approx 137.8$.

<!--en:c947eb0e-->
2. Eine Annahmerate von $97\%$ weist auf eine winzige Vorschlagsschrittweite hin ($s \ll 1$). Die Kette breitet sich daher nur langsam durch Diffusion aus. **Split-$\widehat{R} \gg 1.01$** und ein winziges **$N_{\mathrm{eff}}$** zeigen die langsame Durchmischung sofort an.

<!--en:14803730-->
3. Aus $s_j = \sigma_j\sqrt{1 - \rho^2}$ mit $\sqrt{1 - (-0.80)^2} = \sqrt{0.36} = 0.60$ folgt, dass Mean-Field-VI gegen $(s_1, s_2) = (1.20, 0.60)$ konvergiert. Damit unterschätzt sie die marginale Unsicherheit um $40\%$.

<!--en:9e36a305-->
::: {.slide title="Zusammenfassung der bayesschen Rechenverfahren"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:bbc2f3a5-->
| Verfahren | Darstellung der A-posteriori-Verteilung | Wie die Normierungskonstante $p(\mathcal{D})$ wegfällt | Wichtigste Diagnose | Typisches Problem |

<!--en:4222d812-->
| **Laplace-Approximation** | $\mathcal{N}(\boldsymbol{\theta}_{\mathrm{MAP}}, \mathbf{H}^{-1})$ am Modus | Additive Konstante $\log p(\mathcal{D})$ verschwindet in $\nabla, \nabla^2$ | Vergleich mit MCMC-Ziehungen | Erfasst Schiefe, schwere Ränder, Grenzen und mehrere Modi nicht |

<!--en:bb29db69-->
| **Rejection-Sampling** | unabhängige exakte Ziehungen | Erscheint nur in der Annahmerate $p(\mathcal{D})/M$ | empirische Annahmerate | Rate sinkt wie $n^{-d/2}$, globale Schranke $M$ nötig |

<!--en:ca3ca017-->
| **Importance-Sampling** | gewichtete Ziehungen $(\boldsymbol{\theta}_s, \bar{w}_s)$ | Kürzt sich im selbstnormierten Verhältnis $\frac{\sum w_s h_s}{\sum w_s}$ heraus | ESS der Gewichte $1/\sum \bar{w}_s^2$ und größtes Gewicht | Unendliche Gewichtsvarianz möglich, wenn $q$ leichtere Ränder als $p$ hat |

<!--en:3f9cbf21-->
| **Metropolis-Hastings** | korrelierte Zustände einer Markov-Kette | Kürzt sich im Annahmeverhältnis $\frac{\widetilde{p}(\boldsymbol{\theta}')}{\widetilde{p}(\boldsymbol{\theta})}$ heraus | Split-$\widehat{R} < 1.01$, ESS $cN/\kappa$, MCSE | Langsame Durchmischung der Irrfahrt bei hohem $d$, Verharren an einem Modus |

<!--en:11f74953-->
| **Variationelle Inferenz** | optimierte parametrische Verteilung $q_{\boldsymbol{\psi}}(\boldsymbol{\theta})$ | Verschiebt ELBO $\mathcal{L}(\boldsymbol{\psi})$ um eine feste Konstante | Vergleich mit MCMC oder NLL auf zurückgehaltenen Daten | Modussuchende Rückwärts-KL, Mean-Field verkleinert Standardabweichung um Faktor $\sqrt{1-\rho^2}$ |
