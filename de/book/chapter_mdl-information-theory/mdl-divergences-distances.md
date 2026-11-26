<!--en:949c2ad4-->
# f-Divergenzen

<!--en:5590f6c1-->
Immer wenn wir in diesem Buch ein Modell nach dem Maximum-Likelihood-Prinzip anpassen, messen wir seine Übereinstimmung mit den Daten anhand einer einzigen Zahl, der KL-Divergenz. Am Ende von :numref:`sec_mdl-information_theory` haben wir jedoch eine Schwäche der KL-Divergenz gesehen. Sie springt ins Unendliche, sobald das Modell einem tatsächlich eintretenden Ereignis die Wahrscheinlichkeit null zuweist. Ein Klassifikator stößt selten auf dieses Problem, weil eine Softmax-Schicht nie exakt null ausgibt. Ein generatives Netz, das einen niedrigdimensionalen Rauschvektor in den Bildraum abbildet, stößt sofort darauf. Seine Samples liegen auf einer dünnen Fläche im Bildraum, die die echten Daten vollständig verfehlen kann. Wie können wir zwei Verteilungen sonst vergleichen? Welche Vergleiche unterscheiden noch zwischen einem Modell, das die Daten fast trifft, und einem, das weit danebenliegt?

<!--en:680cfe59-->
Schon das einfachste Beispiel zeigt, was schiefgehen kann. Nehmen wir an, die Daten liegen in einem einzigen Punkt, $P = \delta_0$. Auch das Modell konzentriert seine gesamte Masse in einem Punkt, allerdings im Abstand $d$ von den Daten, $Q = \delta_d$. Anschaulich liegt ein Modell bei $d = 0.01$ fast richtig und eines bei $d = 100$ weit daneben. Ein nützlicher Verlust sollte sie daher unterscheiden. :numref:`tab_mdl-point-mass-table` zeigt, welche Werte fünf gängige Vergleichsmaße liefern.

<!--en:bc775f45-->
:Fünf Vergleiche der Punktmassen $\delta_0$ und $\delta_d$. Die MMD-Zeile verwendet den RBF-Kernel mit Längenskala $1$.

<!--en:6e8a594e-->
| Vergleichsmaß | $d = 0.01$ | $d = 1$ | $d = 100$ |

<!--en:aa88dc72-->
| KL-Divergenz $D_{\textrm{KL}}(P\|Q)$ | $\infty$ | $\infty$ | $\infty$ |

<!--en:ba9a6195-->
| Jensen-Shannon-Divergenz | $\log 2$ | $\log 2$ | $\log 2$ |

<!--en:f1c9f3e1-->
| Totalvariationsabstand | $1$ | $1$ | $1$ |

<!--en:0e7c00fd-->
| Maximum Mean Discrepancy $\textrm{MMD}^2$ | $0.0001$ | $0.79$ | $2.00$ |

<!--en:0c7949ed-->
| Wasserstein-Abstand $W_1$ | $0.01$ | $1$ | $100$ |

<!--en:b20f00b9-->
Betrachten wir die Tabelle genauer. In drei der fünf Zeilen ändert sich nichts, wenn wir $d$ variieren. Trainieren wir ein Modell mit einem dieser drei Maße, so ist der Verlust bei $d = 0.01$ genauso groß wie bei $d = 100$. Der Gradient bezüglich $d$ ist daher null. Genau vor diesem Problem stand das ursprüngliche generative adversariale Netz (engl. generative adversarial network, GAN) von :numref:`chap_gans`, denn seine Zielfunktion entspricht bis auf Konstanten der Jensen-Shannon-Zeile.

<!--en:8668418c-->
Diese drei konstanten Zeilen gehören zu einer Familie. Die KL-Divergenz vergleicht $P$ und $Q$ über das Verhältnis $p/q$ ihrer Dichten. Behalten wir dieses Dichteverhältnis (engl. density ratio) bei und variieren die darauf angewandte konvexe Funktion, so erhalten wir eine große Familie, die *f-Divergenzen*. Für alle Mitglieder gilt derselbe Beweis der Nichtnegativität. Wir können sie auf dieselbe Weise aus Stichproben schätzen, und sie alle erfassen den Abstand $d$ aus demselben strukturellen Grund nicht. Die beiden übrigen Zeilen ändern sich mit $d$ und kommen ganz ohne das Dichteverhältnis aus. Wir untersuchen sie in :numref:`sec_mdl-test-functions-transport`.

<!--en:7462d368-->
Sobald wir diese Familie kennen, gehen wir drei Fragen nach, die im Deep Learning immer wieder auftreten. Erstens, spielt es bei der KL-Divergenz eine Rolle, welche Verteilung zuerst steht – die Datenverteilung wie bei Maximum-Likelihood oder das Modell wie bei der Variationsinferenz (:numref:`sec_mdl-fwd-vs-rev-kl`)? Zweitens, was garantiert eine kleine Divergenz, wenn wir $P$ von $Q$ unterscheiden wollen (:numref:`sec_mdl-tv-pinsker`)? Drittens, was lässt sich noch berechnen, wenn wir beide Verteilungen nur aus Stichproben kennen (:numref:`sec_mdl-f-gan-dual`)? Mit dieser letzten Frage endete :numref:`sec_mdl-coding-compression`. Auf ihrer Antwort beruht das adversariale Training.

<!--en:eb807b32-->
In diesem Abschnitt verwenden wir ausschließlich natürliche Logarithmen. Wir messen die KL- und die Jensen-Shannon-Divergenz daher in Nat. Wir schreiben $P$ für die Daten- oder Zielverteilung und $Q$ für das Modell, $p$ und $q$ für ihre Wahrscheinlichkeitsfunktionen oder Dichten und $u = p/q$ für ihr *Dichteverhältnis*.

<!--en:d91be9ae-->
Wir können jeden Eintrag in :numref:`tab_mdl-point-mass-table` direkt im Code überprüfen. Die nächste Zelle stellt $\delta_0$ und $\delta_d$ als die Wahrscheinlichkeitsvektoren $(1, 0)$ und $(0, 1)$ auf dem zweipunktigen Träger $\{0, d\}$ dar und berechnet jedes Vergleichsmaß nach seiner Definition. Drei dieser Definitionen folgen in diesem Abschnitt, die beiden anderen im nächsten.

<!--en:ff87e4b2-->
## Die Familie der f-Divergenzen

<!--en:26ce30b4-->
Bevor wir die KL-Divergenz verallgemeinern, halten wir fest, welche Eigenschaften ein Vergleichsmaß für zwei Verteilungen mindestens haben sollte. Die Familie der f-Divergenzen ergibt sich dann aus der Frage, welche Voraussetzungen der Beweis der Nichtnegativität der KL-Divergenz tatsächlich benötigt.

<!--en:82996d4e-->
### Divergenzen und Metriken

<!--en:99ce8e45-->
Ein „Abstand zwischen Verteilungen“ sollte zumindest dann verschwinden, wenn beide Verteilungen übereinstimmen, und strikt positiv sein, wenn sie sich unterscheiden. Jede Funktion zweier Verteilungen mit dieser Eigenschaft heißt **Divergenz**:

<!--en:9ecde5d7-->
Weitere Anforderungen gibt es nicht. Eine **Metrik** wie der euklidische Abstand $\|\mathbf{x} - \mathbf{y}\|$ zwischen zwei Vektoren (:numref:`sec_linear-algebra`) muss zusätzlich symmetrisch sein, $D(P, Q) = D(Q, P)$, und die Dreiecksungleichung erfüllen, $D(P, R) \leq D(P, Q) + D(Q, R)$. Die KL-Divergenz ist eine Divergenz. Die Gibbs-Ungleichung (:numref:`sec_mdl-information_theory`) garantiert $D_{\textrm{KL}}(P\|Q) \geq 0$, wobei Gleichheit nur für $P = Q$ gilt. Sie verletzt jedoch beide zusätzlichen Metrikaxiome, wie schon eindimensionale Gaußverteilungen zeigen. Wie wir dort berechnet haben, beträgt die KL-Divergenz zwischen zwei Gaußverteilungen

<!--en:f9bd0b20-->
Für $P = \mathcal{N}(0, 1)$ und $Q = \mathcal{N}(0, 4)$ ergibt diese Formel $D_{\textrm{KL}}(P\|Q) \approx 0.318$ Nat in der einen Richtung und $D_{\textrm{KL}}(Q\|P) \approx 0.807$ Nat in der anderen. Die KL-Divergenz ist also asymmetrisch. Die Dreiecksungleichung scheitert aus einem anderen Grund. Betrachten wir drei Gaußverteilungen mit Varianz eins und den Mittelwerten $0$, $1$ und $2$. Die Formel ergibt $D_{\textrm{KL}} = \tfrac{1}{2}$ für jedes benachbarte Paar, aber $2$ für das äußere Paar, und es gilt $2 > \tfrac{1}{2} + \tfrac{1}{2}$. Da die KL-Divergenz mit dem *Quadrat* des Abstands der Mittelwerte wächst, verhält sie sich wie ein quadrierter euklidischer Abstand und nicht wie eine Länge.

<!--en:bc19e042-->
Keine dieser beiden Eigenschaften müssen wir korrigieren. Um eine Divergenz zu minimieren, brauchen wir weder Symmetrie noch die Dreiecksungleichung. Die Asymmetrie der KL-Divergenz eröffnet uns sogar eine Wahl bei der Modellierung (:numref:`sec_mdl-fwd-vs-rev-kl`). Sobald ein Beweis jedoch die Vergleiche von $P$ mit $Q$ und von $Q$ mit $R$ verknüpfen muss, ist die Dreiecksungleichung unverzichtbar. Die Familie, die wir nun konstruieren, enthält glücklicherweise auch echte Metriken.

<!--en:77fa242b-->
### Allgemeine Form der f-Divergenz

<!--en:649f569c-->
Dank der Gibbs-Ungleichung ist die KL-Divergenz tatsächlich eine Divergenz. Betrachten wir den Beweis genauer, denn jede andere Funktion mit denselben beiden Eigenschaften liefert ebenfalls eine Divergenz.

<!--en:f0c1afb4-->
Zuerst schreiben wir die KL-Divergenz als Erwartungswert unter $Q$. Dabei mitteln wir eine Funktion des Dichteverhältnisses $u(x) = p(x)/q(x)$:

<!--en:721b18fb-->
So betrachtet gewichtet die KL-Divergenz jedes Ergebnis gemäß $Q$ und bewertet sein Dichteverhältnis mit der Straffunktion $u \log u$. Prüfen wir diese Formel an unserem durchgängigen Beispiel aus :numref:`sec_mdl-information_theory`, dem diskreten Verteilungspaar $P = (0.6, 0.3, 0.1)$ und $Q = (0.2, 0.5, 0.3)$. Die Verhältnisse betragen $u = 3$, $0.6$ und $\tfrac13$, die zugehörigen Gewichte $0.2$, $0.5$ und $0.3$:

<!--en:3220e14c-->
Das entspricht der zuvor berechneten KL-Divergenz.

<!--en:e72e2b7f-->
Betrachten wir zweitens den Beweis der Nichtnegativität. In :numref:`sec_mdl-information_theory` haben wir die Jensen-Ungleichung einmal auf $-\log(q/p)$ unter $P$ angewandt. Führen wir denselben Schritt unter $Q$ für die Funktion $u \log u$ aus, so brauchen wir nur zwei Tatsachen. Die Funktion ist konvex und hat bei $u = 1$ den Wert null. Wir können $u \log u$ also durch jede Funktion mit diesen beiden Eigenschaften ersetzen. Sei $f : (0, \infty) \to \mathbb{R}$ konvex mit $f(1) = 0$. Wir definieren $f(0) = \lim_{u \downarrow 0} f(u)$ und lassen dabei auch den Wert $+\infty$ zu. Nehmen wir vorerst an, dass $p(x) = 0$ überall dort gilt, wo $q(x) = 0$ ist. Dafür schreiben wir $P \ll Q$. So erfasst der Erwartungswert unter $Q$ die gesamte Masse von $P$. Die **f-Divergenz** von $P$ bezüglich $Q$ ist definiert durch :cite:`Csiszar.1967`

<!--en:fb3f5742-->
Das Verhältnis $u = p/q$ misst an jedem Punkt, wie stark $P$ das Ergebnis $x$ im Vergleich zu $Q$ über- oder unterrepräsentiert. Die Funktion $f$ weist dem Verhältnis einen Strafwert zu, der für $u = 1$ verschwindet. Wir nennen $f$ die *Divergenzfunktion*. In der Literatur heißt sie häufig Generator der Divergenz. Unser Code speichert diese Funktionen in einem Wörterbuch namens `generators`. Im Text verwenden wir das Wort *Generator* jedoch für ein Netz, das Samples erzeugt, wie in :numref:`sec_gan_objectives`.

<!--en:4541be03-->
Um genau anzugeben, wann die Divergenz null ist, brauchen wir eine etwas stärkere Form der Konvexität. Wir nennen $f$ *streng konvex bei $1$*, wenn es eine Gerade durch $(1, 0)$ gibt, die überall sonst strikt unter $f$ liegt. Das heißt, es gilt $f(u) > c\,(u - 1)$ für alle $u \neq 1$ bei einer geeigneten Steigung $c$.

<!--en:8e09e209-->
**Satz (Nichtnegativität).** *Für jede konvexe Funktion $f$ mit $f(1) = 0$ und für $P \ll Q$ gilt*

<!--en:1020c70b-->
*Ist zudem $f$ bei $1$ streng konvex, so gilt $D_f(P\|Q) = 0$ genau dann, wenn $P = Q$.*

<!--en:cf32a969-->
**Beweis.** Nach der Jensen-Ungleichung (:numref:`subsec_mdl-jensen`) gilt $\mathbb{E}[f(U)] \geq f(\mathbb{E}[U])$ für jede konvexe Funktion $f$ und jede Zufallsvariable $U$. Wir wenden sie auf $U = u(x)$ mit $x \sim Q$ an. Entscheidend ist eine einzige Identität. Das Dichteverhältnis hat unter $Q$ den Erwartungswert eins, denn $\mathbb{E}_Q[u] = \sum_x q(x)\, p(x)/q(x) = \sum_x p(x) = 1$. Also gilt

<!--en:b1486db0-->
Für den Gleichheitsfall betrachten wir die Stützgerade $c\,(u - 1)$, die strikt unter $f$ liegt, außer bei $1$. Dann gilt $D_f(P\|Q) \geq \mathbb{E}_Q[c\,(u - 1)] = 0$. Die Differenz ist der Erwartungswert von $f(u) - c\,(u - 1)$, und dieser Ausdruck ist überall dort strikt positiv, wo $u \neq 1$ gilt. Aus $D_f(P\|Q) = 0$ folgt daher $u(x) = 1$ für jedes $x$ mit $q(x) > 0$, also $P = Q$. $\blacksquare$

<!--en:1950d86f-->
Die Identität $\mathbb{E}_Q[u] = 1$ aus diesem Beweis hat eine zweite Konsequenz, die wir wiederholt nutzen. Addieren wir ein beliebiges Vielfaches $c\,(u - 1)$ zu $f$, so ändert sich keine Divergenz, denn $\mathbb{E}_Q[c\,(u - 1)] = 0$. Beispielsweise definieren die Funktionen $u \log u$ und $u \log u - u + 1$ dieselbe KL-Divergenz.

<!--en:1c818a77-->
Bisher haben wir $P \ll Q$ vorausgesetzt. Die Punktmassen aus unserem Eingangsbeispiel verletzen diese Bedingung jedoch. Zu Beginn des Trainings befindet sich ein generatives Modell in derselben Lage, wenn seine Samples und die Daten in disjunkten Gebieten liegen. Wir müssen $D_f$ daher auf Nullstellen von $q$ erweitern. Ergebnisse mit $p(x) = q(x) = 0$ tragen nichts bei. Wo $p(x) = 0 < q(x)$ gilt, beträgt der Beitrag gemäß der Definition $q(x)\, f(0)$. Für $q(x) = 0 < p(x)$ wächst das Verhältnis ins Unendliche. Hier definieren wir den Beitrag als Grenzwert von $q\, f(p/q)$ für $q \downarrow 0$. Mit der Substitution $t = p/q$ ergibt sich $p\, f(t)/t$ mit dem Grenzwert $p(x)\, f'(\infty)$, wobei $f'(\infty) := \lim_{t \to \infty} f(t)/t$. Für eine konvexe Funktion $f$ haben die Sekantensteigung $f(t)/t$ und die Ableitung $f'(t)$ denselben Grenzwert im Unendlichen. Daher rührt die Schreibweise. Sowohl $f(0)$ als auch $f'(\infty)$ können den Wert $+\infty$ annehmen.

<!--en:9704bf33-->
Wenden wir diese Regel auf die Punktmassen $\delta_0$ und $\delta_d$ für beliebiges $d \neq 0$ an. Bei $x = 0$ gilt $q = 0 < p = 1$ und bei $x = d$ gilt $p = 0 < q = 1$. Also folgt

<!--en:f2fb3d0b-->
Auf der rechten Seite kommt $d$ nicht mehr vor. Das sagt weit mehr aus, als :numref:`tab_mdl-point-mass-table` allein gezeigt hat. *Jede* f-Divergenz ist auf disjunkten Trägern (engl. disjoint supports) konstant. Sie kann nicht unterscheiden, ob das Modell die Daten knapp oder weit verfehlt. Beim Training liefert der Abstand zwischen den Trägern also keinen Gradienten. Für die KL-Divergenz ergibt $f(u) = u \log u$ die Werte $f(0) = 0$ und $f'(\infty) = \lim_{t \to \infty} \log t = \infty$. Damit erhalten wir die erste Zeile von :numref:`tab_mdl-point-mass-table`. Die beiden anderen konstanten Zeilen ergeben sich, sobald wir ihre Divergenzfunktionen angeben.

<!--en:8092dc62-->
### Gängige Divergenzfunktionen

<!--en:f622d520-->
Jede Zeile in :numref:`tab_mdl-f-div-generators` entspricht einer Wahl von $f$. In :numref:`fig_mdl-f-div-generators` sehen wir sechs dieser Kurven im Vergleich. Die Spalte $f''(1)$ gibt die Krümmung bei $u = 1$ an. Sie spielt im nächsten Unterabschnitt eine zentrale Rolle.

<!--en:707a8742-->
:Gängige Divergenzfunktionen. Jede Zeile zeigt für eine Divergenz die Funktion $f$ und ihre Krümmung bei $u = 1$. Außerdem gibt sie an, ob die Divergenz symmetrisch ist und ob sie selbst oder ihre Quadratwurzel eine Metrik ist.

<!--en:8c37f10f-->
| Divergenz | $f(u)$ | $f''(1)$ | Symmetrisch? | Metrik? |

<!--en:efbacb76-->
| Kullback-Leibler (vorwärts) | $u \log u$ | $1$ | nein | nein |

<!--en:e6d7146a-->
| Rückwärts-KL | $-\log u$ | $1$ | nein | nein |

<!--en:e24bc821-->
| Pearson-$\chi^2$-Divergenz | $(u - 1)^2$ | $2$ | nein | nein |

<!--en:7ec89116-->
| quadrierter Hellinger-Abstand $d_{\mathrm{H}}^2$ | $(\sqrt{u} - 1)^2$ | $\tfrac12$ | ja | $d_{\mathrm{H}}$ ist eine Metrik |

<!--en:359cffef-->
| Totalvariationsabstand | $\tfrac{1}{2}\lvert u - 1 \rvert$ | keine (Knick) | ja | ja |

<!--en:10b5f4f5-->
| Jensen-Shannon | $\tfrac{u}{2} \log u - \tfrac{u+1}{2} \log \tfrac{u+1}{2}$ | $\tfrac14$ | ja | $\sqrt{\textrm{JS}}$ ist eine Metrik |

<!--en:d71b73d7-->
| $\alpha$-Divergenz ($\alpha \neq 0, 1$) | $\dfrac{u^\alpha-\alpha u+\alpha-1}{\alpha(\alpha-1)}$ | $1$ | nur bei $\alpha = \tfrac{1}{2}$ | Quadratwurzel bei $\alpha = \tfrac{1}{2}$ |

<!--en:a5fe3676-->
![Sechs Divergenzfunktionen. Wir haben die glatten Kurven um den Term $f'(1)(u-1)$ verschoben, sodass jede bei $u=1$ die horizontale Achse berührt, ohne dass sich die zugehörige Divergenz ändert. In der Nähe von $u=1$ verlaufen die glatten Kurven parabelförmig mit den in der Tabelle angegebenen Krümmungen $1, 1, 2, \tfrac12, \tfrac14$. Der Totalvariationsabstand hat dort einen Knick. Fern von $u=1$ bestrafen die Funktionen Abweichungen sehr unterschiedlich. Die Rückwärts-KL wächst für $u \to 0$ ins Unendliche, die KL- und die $\chi^2$-Divergenz wachsen überlinear und der Totalvariationsabstand wächst linear.](../img/mdl-it-f-div-generators.svg)

<!--en:46af4513-->
Die **Rückwärts-KL** (engl. reverse KL) verwendet $f(u) = -\log u$, denn $\mathbb{E}_Q[-\log(p/q)] = \mathbb{E}_Q[\log(q/p)] = D_{\textrm{KL}}(Q\|P)$. Die Argumente der KL-Divergenz zu vertauschen entspricht also einer anderen Wahl von $f$. Deshalb stehen beide Richtungen in derselben Tabelle. Aufgabe :exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`a zeigt allgemein, wie sich die Argumente einer beliebigen f-Divergenz vertauschen lassen.

<!--en:9dedb414-->
Der **Totalvariationsabstand** (engl. total variation distance) verwendet $f(u) = \tfrac{1}{2}|u - 1|$. Damit vereinfacht sich die Divergenz zu $\sum_x q(x)\, \tfrac12 |p(x)/q(x) - 1| = \tfrac{1}{2}\sum_x |p(x) - q(x)|$. Wir schreiben dafür $\textrm{TV}(P, Q)$. Er ist der halbe $\ell_1$-Abstand der Wahrscheinlichkeitsvektoren. Für unser Beispielpaar ergibt sich $\textrm{TV} = \tfrac12(0.4 + 0.2 + 0.2) = 0.4$. Was diese Zahl für die Unterscheidung von $P$ und $Q$ bedeutet, untersuchen wir in :numref:`sec_mdl-tv-pinsker`. Für die Punktmassen liefert die Divergenzfunktion $f(0) + f'(\infty) = \tfrac12 + \tfrac12 = 1$. Das erklärt die dritte Zeile in :numref:`tab_mdl-point-mass-table`.

<!--en:223f7b9d-->
Die **Pearson-$\chi^2$-Divergenz** verwendet die Parabel $f(u) = (u - 1)^2$. Das ergibt $\chi^2(P\|Q) = \sum_x q(x)\,\big(p(x)/q(x) - 1\big)^2
= \sum_x (p(x) - q(x))^2 / q(x)$. Für unser Beispielpaar erhalten wir $0.16/0.2 + 0.04/0.5 + 0.04/0.3 = 1.0133$. Beachten Sie, wie stark eine Abweichung dort ins Gewicht fällt, wo $Q$ klein ist. Allein das erste Ergebnis mit $q = 0.2$ trägt $0.8$ zur Gesamtsumme bei.

<!--en:1f5b448a-->
Die **Jensen-Shannon-Divergenz** :cite:`Lin.1991` macht die KL-Divergenz symmetrisch, indem sie beide Verteilungen mit ihrer gleichgewichteten Mischung $M = \tfrac{1}{2}(P + Q)$ (mit Wahrscheinlichkeitsfunktion $m$) vergleicht:

<!--en:29f6f94c-->
Da $M$ überall dort Masse hat, wo $P$ oder $Q$ Masse hat, bleiben die Verhältnisse $p/m$ und $q/m$ beschränkt. Es gilt $p/m = 2p/(p + q) \leq 2$, also beträgt jeder KL-Term höchstens $\log 2$. Damit ist JS symmetrisch und beschränkt, $0 \leq \textrm{JS}(P, Q) \leq \log 2$. Den Höchstwert $\log 2$ erreicht JS nur dann, wenn $P$ und $Q$ disjunkte Träger haben (dort gilt $p/m = 2$ überall, wo $p > 0$). Bemerkenswerterweise ist ihre Quadratwurzel eine echte Metrik :cite:`Endres.Schindelin.2003`. Drücken wir :eqref:`eq_mdl-js-def` durch das Verhältnis $u$ aus, so erhalten wir die Divergenzfunktion aus der Tabelle (Aufgabe :exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`c). Für diese gilt $f(0) = f'(\infty) = \tfrac12 \log 2$. Die Summe beträgt $\log 2$, wie in der zweiten Zeile von :numref:`tab_mdl-point-mass-table`. Die eine Identität $D_f(\delta_0\|\delta_d) = f(0) + f'(\infty)$ erklärt damit alle drei konstanten Zeilen.

<!--en:c956e313-->
Der **quadrierte Hellinger-Abstand** ist $d_{\mathrm{H}}^2(P, Q) = \sum_x \big(\sqrt{p(x)} - \sqrt{q(x)}\big)^2$. Seine Quadratwurzel $d_{\mathrm{H}}$ ist der euklidische Abstand zwischen den Vektoren $\sqrt{p}$ und $\sqrt{q}$. Damit ist $d_{\mathrm{H}}$ unmittelbar eine Metrik (Aufgabe :exref:`ex_mdl-divergences-distances-prove-that-total-variation`b).

<!--en:b34e944d-->
Die letzte Tabellenzeile enthält eine kontinuierliche Familie innerhalb der Familie, die **$\alpha$-Divergenzen** :cite:`Amari.2016` mit der Divergenzfunktion

<!--en:f2e99370-->
Für jedes $\alpha \neq 0, 1$ ist diese Funktion konvex mit $f_\alpha(1) = 0$ und der Krümmung $f_\alpha''(1) = 1$. Für $\alpha \to 1$ nähert sie sich $u\log u - u + 1$ und für $\alpha \to 0$ nähert sie sich $-\log u + u - 1$. Da sich bei Addition eines Vielfachen von $u - 1$ keine Divergenz ändert, ergeben die beiden Grenzwerte die Vorwärts-KL (engl. forward KL) und die Rückwärts-KL. In der Mitte, bei $\alpha = \tfrac12$, erhalten wir $2(\sqrt{u} - 1)^2$ (das Doppelte des quadrierten Hellinger-Abstands), während $\alpha = 2$ die Hälfte von $\chi^2$ ergibt. Ein einziger Parameter $\alpha$ vermittelt also stetig zwischen der Rückwärts-KL, dem Hellinger-Abstand, der Vorwärts-KL und schließlich der $\chi^2$-Divergenz (Aufgabe :exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`d stellt den Zusammenhang mit der Rényi-Divergenz her).

<!--en:1a4a0959-->
Werten wir die ersten sechs Zeilen für unser Beispielpaar in beiden Argumentreihenfolgen aus. Für die KL-Divergenz erwarten wir $0.396$ und $0.365$ Nat wie in :numref:`sec_mdl-information_theory`. Der TV- und der $\chi^2(P\|Q)$-Wert sollten mit unserer Rechnung von Hand übereinstimmen.

<!--en:c4634e5f-->
Wie erwartet ist die KL-Zeile asymmetrisch. In der Zeile für die Rückwärts-KL sind die beiden Spalten der KL-Zeile vertauscht. Auch die $\chi^2$-Zeile ist asymmetrisch ($1.0133$ gegenüber $0.8000$). Dagegen sind der quadrierte Hellinger-Abstand ($0.1862$), TV ($0.4$) und JS symmetrisch. Mit $0.0911$ Nat liegt JS deutlich unter ihrem Höchstwert $\log 2 \approx 0.693$. Für dasselbe Verteilungspaar liefern sechs Divergenzen Werte, die von $0.09$ bis $1.01$ reichen und damit eine Größenordnung überspannen. Multiplizieren wir jedoch die KL-Divergenz mit $2$, den quadrierten Hellinger-Abstand mit $4$ und JS mit $8$, so erhalten wir $0.79$, $0.74$ und $0.73$. Warum stimmen drei so unterschiedliche Funktionen des Dichteverhältnisses annähernd überein? Woher kommen die Faktoren $2$, $4$ und $8$?

<!--en:148febe2-->
### Lokale quadratische Näherung

<!--en:8e6b00c9-->
Die Kurven in :numref:`fig_mdl-f-div-generators` verhalten sich fern von $u = 1$ sehr unterschiedlich. In der Nähe von $u = 1$ stimmen sie jedoch bis auf einen konstanten Faktor überein. Das ist vielleicht die überraschendste Eigenschaft der ganzen Familie. Da der Rest dieses Abschnitts nicht auf diesem Unterabschnitt aufbaut, können Sie ihn beim ersten Lesen überspringen.

<!--en:fe3da86a-->
Da ein zusätzlicher linearer Term $c\,(u - 1)$ keine Divergenz ändert, beeinflussen weder der Wert $f(1) = 0$ noch die Steigung $f'(1)$ die Divergenz $D_f$. Die erste Eigenschaft, in der sich glatte Divergenzfunktionen unterscheiden, ist ihre Krümmung $f''(1)$. In der Nähe von $u = 1$ gleicht jede glatte Funktion $f$ der Parabel $\tfrac12 f''(1)(u - 1)^2$. Der Erwartungswert von $(u - 1)^2$ unter $Q$ ist gerade die $\chi^2$-Divergenz, $\mathbb{E}_Q[(u - 1)^2] = \sum_x (p(x) - q(x))^2/q(x)$. Mit $r = \max_x |u(x) - 1|$ messen wir, wie weit das Dichteverhältnis von $1$ abweicht.

<!--en:ac01e00d-->
**Satz (lokale quadratische Näherung).** *Es gelte $P \ll Q$ und $0 < r < 1$. Die Funktion $f$ sei auf $[1 - r, 1 + r]$ dreimal differenzierbar und erfülle dort $|f'''| \leq B$. Ferner gelte $|u(x) - 1| \leq r$ für jedes $x$ mit $q(x) > 0$. Dann gilt*

<!--en:232b9793-->
**Beweis.** Wegen $f(1) = 0$ liefert der Satz von Taylor mit Restglied die Entwicklung $f(u) = f'(1)(u - 1) + \tfrac12 f''(1)(u - 1)^2 + R(u)$ mit $|R(u)| \leq B |u - 1|^3 / 6$. Wir bilden gliedweise $\mathbb{E}_Q$. Der lineare Term verschwindet wegen $\mathbb{E}_Q[u - 1] = 0$. Damit entfällt der einzige Unterschied erster Ordnung zwischen zwei Funktionen $f$. Der quadratische Term ergibt $\tfrac12 f''(1)\, \chi^2(P\|Q)$, und für das Restglied gilt $\mathbb{E}_Q|R| \leq \tfrac{B}{6}\,\mathbb{E}_Q|u - 1|^3 \leq \tfrac{B r}{6}\, \mathbb{E}_Q(u - 1)^2 = \tfrac{B r}{6}\,\chi^2$. $\blacksquare$

<!--en:a96be2ff-->
Kurz gesagt gilt $D_f \approx \tfrac12 f''(1)\, \chi^2$ mit einem relativen Fehler proportional zu $r$. Liegt $P$ nahe bei $Q$, so sind alle glatten f-Divergenzen näherungsweise skalare Vielfache voneinander. Die Faktoren $2$, $4$ und $8$, mit denen wir die KL-Divergenz, den quadrierten Hellinger-Abstand und JS multipliziert haben, sind gerade $2/f''(1)$. Alle drei Produkte nähern denselben Wert $\chi^2(P\|Q) = 1.013$ an. Für unser Beispielpaar liegen sie noch etwa ein Viertel darunter. Das überrascht kaum, denn das größte Verhältnis ist $3$. Damit gilt $r = 2$, was die Voraussetzung $r < 1$ des Satzes verletzt. Um zu sehen, wie die quadratische Näherung genauer wird, verschieben wir $P$ in Richtung $Q$ entlang der Strecke $P_t = Q + t\,(P - Q)$. Dadurch skaliert jedes $u - 1$ mit $t$, sodass $r = 2t$ gilt. Für vier glatte Divergenzen geben wir das Verhältnis $D_f(P_t\|Q) / \big(\tfrac12 f''(1)\, \chi^2(P_t\|Q)\big)$ aus. Es sollte gegen $1$ konvergieren, und zwar mit einer zu $t$ proportionalen Rate.

<!--en:01412936-->
Bei $t = 0.5$ erreichen die vier Divergenzen etwa $75\%$ bis $85\%$ ihrer quadratischen Näherungen. Bei $t = 0.1$ übersteigen sie $91\%$ und bei $t = 0.01$ sogar $99\%$. Zwischen den letzten beiden Zeilen sinkt $t$ um den Faktor zehn. Auch die Abweichungen sinken jeweils fast um diesen Faktor, passend zum linearen Faktor $r$ im Satz. Die Schranke selbst ist konservativ. Für die KL-Divergenz bei $t = 0.1$ gilt $r = 0.2$ und $|f'''(u)| = 1/u^2 \leq 1/0.8^2$. Das lässt einen relativen Fehler von etwa $10\%$ zu, tatsächlich beträgt die Abweichung nur $4\%$. Für die Optimierung bedeutet das, dass sich glatte f-Divergenzen nur bei weit auseinanderliegenden Verteilungen wesentlich unterscheiden. Das ist zu Beginn des Trainings der Fall, wenn das Modell noch weit von den Daten entfernt ist.

<!--en:76c088bc-->
Innerhalb einer parametrischen Familie bestimmt die Fisher-Informationsmatrix $\mathbf{I}(\boldsymbol{\theta}) = \mathbb{E}\big[\nabla_{\boldsymbol{\theta}}
\log q_{\boldsymbol{\theta}}\, \nabla_{\boldsymbol{\theta}} \log
q_{\boldsymbol{\theta}}^\top\big]$ aus :numref:`sec_mdl-maximum_likelihood` diese gemeinsame lokale quadratische Näherung. Für einen kleinen Parameterschritt $\mathbf{h}$ hat das Dichteverhältnis von $q_{\boldsymbol{\theta} + \mathbf{h}}$ zu $q_{\boldsymbol{\theta}}$ die Entwicklung $u \approx 1 + \mathbf{h}^\top \nabla_{\boldsymbol{\theta}} \log q_{\boldsymbol{\theta}}$. Daher gilt $\chi^2(q_{\boldsymbol{\theta} + \mathbf{h}}\| q_{\boldsymbol{\theta}}) =
\mathbb{E}[(u - 1)^2] \approx \mathbf{h}^\top \mathbf{I}(\boldsymbol{\theta})\,
\mathbf{h}$. Jede glatte Divergenz $D_f$ hat also die Hesse-Matrix $f''(1)\, \mathbf{I}(\boldsymbol{\theta})$ bei $\mathbf{h} = \mathbf{0}$. Der natürliche Gradient aus :numref:`sec_mdl-adaptive-stochastic-methods` wird gewöhnlich über die KL-Divergenz begründet, würde aber mit jeder glatten f-Divergenz dasselbe Update liefern. Einzig der Totalvariationsabstand bildet in unserer Tabelle eine Ausnahme. Seine Divergenzfunktion hat bei $u = 1$ einen Knick und dort keine zweite Ableitung. In Aufgabe :exref:`ex_mdl-divergences-distances-sharpness-pinsker-coins-biases`b untersuchen wir, wie sich das auswirkt, wenn sich $P$ an $Q$ annähert.

<!--en:6e2603a2-->
## Vorwärts- und Rückwärts-KL

<!--en:51e4536b-->
Maximum-Likelihood ist das Trainingskriterium für die meisten Modelle in diesem Buch. Es minimiert die KL-Divergenz mit der Datenverteilung als erstem Argument. Die Variationsinferenz minimiert sie mit dem Modell als erstem Argument. Beide Reihenfolgen haben in :numref:`tab_mdl-f-div-generators` eine eigene Zeile. Passen wir ein Modell $Q_{\boldsymbol{\theta}}$ (mit Dichte $q_{\boldsymbol{\theta}}$) an eine Zielverteilung $P$ außerhalb der Modellfamilie an, so treiben sie das Modell in entgegengesetzte Richtungen:

<!--en:08c69487-->
* Die **Vorwärts-KL**, $D_{\textrm{KL}}(P \,\|\, Q_{\boldsymbol{\theta}}) = \mathbb{E}_P[\log(p/q_{\boldsymbol{\theta}})]$, bildet den Erwartungswert unter der *wahren Verteilung*. Gilt $q_{\boldsymbol{\theta}}(x) \to 0$ an einem Punkt mit $p(x) > 0$, so wird die Divergenz unendlich. Die Vorwärts-KL *vermeidet Nullstellen* (engl. zero-avoiding) und *deckt Modi ab* (engl. mode-covering). Das Modell muss sich über alle Modi der Daten erstrecken, also über alle Gebiete, in denen sich $p$ konzentriert. Dabei muss es unter Umständen auch leeren Gebieten Masse zuweisen, in denen $P$ nie Daten erzeugt.

<!--en:00b76e57-->
* Die **Rückwärts-KL**, $D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \,\|\, P) = \mathbb{E}_{Q_{\boldsymbol{\theta}}}[\log(q_{\boldsymbol{\theta}}/p)]$, bildet den Erwartungswert unter dem *Modell*. Weist das Modell einem Gebiet Masse zu, in dem die wahre Verteilung keine hat, fällt die Strafe hoch aus. Ein Modus, den das Modell auslässt, trägt dagegen nichts zur Zielfunktion bei. Die Rückwärts-KL *erzwingt Nullstellen* (engl. zero-forcing) und *sucht einen Modus* (engl. mode-seeking). Das Modell konzentriert sich auf den Teil der Zielverteilung, den es gut erklären kann.

<!--en:32b740e5-->
In der Praxis entscheidet oft die Berechenbarkeit über die Richtung. Die Vorwärts-KL benötigt Stichproben aus $P$ und die Dichte des Modells. Sie zu minimieren entspricht dem Maximum-Likelihood-Prinzip. Wie :numref:`subsec_mdl-nll-crossentropy` zeigt, ist die mittlere negative Log-Likelihood gleich der Kreuzentropie der empirischen Verteilung bezüglich des Modells. Diese unterscheidet sich von der Vorwärts-KL nur um eine von $\boldsymbol{\theta}$ unabhängige Konstante. Die Rückwärts-KL benötigt Samples des Modells und die Dichte der Zielverteilung. Sie funktioniert auch dann, wenn wir die Zieldichte nur bis auf eine Normierungskonstante kennen. Das ist bei der Variationsinferenz (:numref:`sec_mdl-latent-em-elbo`) der Fall. Ihre Zielverteilung $P$ ist eine A-posteriori-Verteilung, deren Normierungskonstante sich nicht praktikabel berechnen lässt. Daher minimiert sie $D_{\textrm{KL}}(Q_{\boldsymbol{\theta}}\|P)$ über berechenbare Näherungen $Q_{\boldsymbol{\theta}}$.

<!--en:ef060cc4-->
Kennen wir auf beiden Seiten Stichproben und Dichten, können wir die Richtung bei der Modellierung frei wählen. Bei der Destillation eines Schülernetzes aus einem Lehrernetz (:numref:`sec_lm_distillation`) hängt das Ergebnis von der Reihenfolge ab. Steht der Lehrer zuerst, deckt der Schüler alle möglichen Antworten des Lehrers ab. Steht der Schüler zuerst, konzentriert er sich auf die wahrscheinlichsten Ausgaben des Lehrers. Ähnlich setzt der KL-Strafterm in :numref:`sec_regularized` die erlernte Strategie an die erste Stelle. Das hält sie von Aktionen ab, die die Referenzstrategie nie wählt.

<!--en:96ffbb83-->
Enthält die Modellfamilie $P$, so liefern beide Richtungen $P$. Sie unterscheiden sich, wenn das Modell *fehlspezifiziert* ist. Am deutlichsten zeigt das die Anpassung einer einzigen Gaußverteilung an eine bimodale Mischung. Für die Vorwärts-KL ist nicht einmal eine numerische Optimierung nötig. Sie ist gleich einer Konstanten minus $\mathbb{E}_P[\log q(X)]$. Ihr Minimum liegt daher bei der Maximum-Likelihood-Gaußverteilung, wobei $P$ die empirische Verteilung ersetzt. In einer Exponentialfamilie gleicht Maximum-Likelihood die Erwartungswerte der suffizienten Statistiken ab (:numref:`subsec_mdl-moment-matching`). Es gelten also $\mu^* = \mathbb{E}_P[X]$ und $(\sigma^*)^2 = \textrm{Var}_P(X)$, sofern die Varianz endlich ist. Für die Rückwärts-KL gibt es keine solche geschlossene Lösung. Nahe jedem Modus kann ein eigenes lokales Minimum liegen.

<!--en:1fdc9c50-->
Für die bimodale Mischung $P = 0.7\,\mathcal{N}(-2, 0.6^2) + 0.3\,\mathcal{N}(2, 0.6^2)$ können wir die Anpassung mit Vorwärts-KL vorhersagen. Ihr Mittelwert beträgt $0.7(-2) + 0.3(2) = -0.8$. Ihre Varianz ist die Summe aus der Varianz innerhalb der Komponenten und der Varianz zwischen ihnen. Bei den Gewichten $w_1$ und $w_2$ und dem Abstand $\Delta$ zwischen den Mittelwerten beträgt letztere $w_1 w_2 \Delta^2$. Insgesamt ergibt sich $0.36 + 0.7 \cdot 0.3 \cdot 4^2 = 3.72$. Für die Anpassung mit Rückwärts-KL verwenden wir einen ableitungsfreien Optimierer für $\boldsymbol{\theta} = (\mu, \log\sigma)$. Wir starten einmal im Ursprung und einmal nahe dem kleineren Modus. Jede KL-Divergenz berechnen wir durch numerische Quadratur auf einem Gitter. Die nächste Zelle gibt für jede Anpassung Mittelwert, Standardabweichung und beide Divergenzen aus.

<!--en:29f0588e-->
Die Anpassung mit Vorwärts-KL bestätigt unsere Rechnung mit $\mu = -0.800$ und $\sigma = 1.929 \approx \sqrt{3.72}$. Die breite Glockenkurve überspannt beide Modi und legt beträchtliche Masse in das Tal dazwischen, wo $P$ fast keine hat. Die erste Anpassung mit Rückwärts-KL liefert $\mu = -1.998$ und $\sigma = 0.603$. Sie schmiegt sich an die dominante Komponente an und lässt der kleineren fast keine Masse. Ihre Rückwärts-KL von $0.356$ Nat liegt nahe bei $\log(1/0.7) \approx 0.357$. So viel kostet es, die Komponente mit Gewicht $70\%$ wie die gesamte Verteilung zu behandeln. Startet der Optimierer nahe dem kleineren Modus, findet er stattdessen $\mu = +1.995$. Die Rückwärts-KL beträgt dann $1.202$ Nat, nahe bei $\log(1/0.3) \approx 1.204$. Welchen Modus eine Variationsnäherung erfasst, hängt also von ihrer Initialisierung ab.

<!--en:25fee1af-->
Da die Rückwärts-KL unter dem Modell mittelt, geht ein Modus, den das Modell nie erreicht, nicht in die Rechnung ein. Die Zielfunktion erfasst nur, dass der gewählte Modus das Gewicht $0.7$ (oder $0.3$) statt $1$ hat. Daher unterschätzen durch Variationsinferenz gewonnene A-posteriori-Verteilungen bekanntermaßen die Unsicherheit. Hier hat die Anpassung mit Rückwärts-KL die Standardabweichung $0.60$, die Zielverteilung hingegen $1.93$.

<!--en:9a119b4c-->
Die letzten beiden Spalten zeigen, wie unterschiedlich die Kriterien dasselbe Modell bewerten. Die Vorwärts-KL, also das Maximum-Likelihood-Kriterium, bewertet die Anpassung an den dominanten Modus mit $5.98$ Nat (mehr als dem Zehnfachen der $0.558$ Nat für die breite Anpassung) und die an den kleineren Modus mit $14.5$ Nat. In beiden Fällen liegt der ausgelassene Modus mehr als sechs Standardabweichungen des Modells von dessen Mittelwert entfernt. Dort ist $\log q$ stark negativ (Aufgabe :exref:`ex_mdl-divergences-distances-mode-dropping`c). Die Rückwärts-KL ordnet die ersten beiden Anpassungen umgekehrt ein, mit $0.356$ Nat gegenüber $1.27$. Sie bestraft die Masse, die die breite Anpassung in das leere Tal zwischen den Modi legt.

<!--en:c75a036a-->
Unsere Regeln für verschwindende Dichten erfassen diesen Unterschied im Grenzfall. Für die Vorwärts-KL mit $f(u) = u \log u$ gilt $f'(\infty) = \infty$. Weist das Modell einem Gebiet keine Masse zu, obwohl $P$ dort positive Masse hat, sind die Kosten also unendlich. Wegen $f(0) = 0$ entsteht dagegen kein Randterm, wenn das Modell Masse dorthin legt, wo $P$ keine hat. Bei der Rückwärts-KL mit $f(u) = -\log u$ tauschen die beiden Werte ihre Rollen.

<!--en:e09bb560-->
## Totalvariationsabstand, Pinsker-Ungleichung und Datenverarbeitung

<!--en:0dfac395-->
Der Wert $0.4$ einer Divergenz sagt für sich genommen wenig aus. Was garantiert uns diese Zahl? Beim Totalvariationsabstand lässt sich diese Frage unmittelbar beantworten. Die Pinsker-Ungleichung überträgt einen großen Teil dieser Aussage auf die KL-Divergenz.

<!--en:30ace03e-->
### Totalvariationsabstand

<!--en:5a0b84ce-->
Die Divergenzfunktion $\tfrac12|u - 1|$ ist der einfachste Eintrag in :numref:`tab_mdl-f-div-generators`. Sie bestraft nur den absoluten Abstand des Verhältnisses von $1$. Ihre Divergenz ist der halbe $\ell_1$-Abstand zwischen $p$ und $q$. Eine zweite, gleichwertige Darstellung macht die Zahl unmittelbar verständlich. Der Totalvariationsabstand ist die größte Differenz zwischen den Wahrscheinlichkeiten, die $P$ und $Q$ demselben Ereignis $A$ zuweisen:

<!--en:b10f82d3-->
Gilt $\textrm{TV}(P, Q) = 0.03$, so unterscheiden sich die Wahrscheinlichkeiten unter den beiden Verteilungen für kein Ereignis um mehr als drei Prozentpunkte. Das Supremum wird beim Ereignis $A^\star = \{x : p(x) > q(x)\}$ erreicht. Es umfasst alle Ergebnisse, die unter $P$ wahrscheinlicher sind als unter $Q$. Um das zu zeigen, wählen wir ein beliebiges Ereignis $A$. Dann gilt $P(A) - Q(A) = \sum_{x \in A} (p(x) - q(x)) \leq \sum_{x \in A^\star} (p(x) - q(x))$, denn beim Übergang von $A$ zu $A^\star$ kommen nur nichtnegative Terme hinzu und nur nichtpositive fallen weg. Wegen $\sum_x (p(x) - q(x)) = 0$ haben der positive und der negative Anteil von $p - q$ betragsmäßig dieselbe Summe. Die rechte Seite ist also gleich $\tfrac12 \sum_x |p(x) - q(x)|$. Dieselbe Schranke gilt für $Q(A) - P(A)$.

<!--en:acf01501-->
Bei unserem Beispielpaar enthält $A^\star$ nur das erste Ergebnis. Das ergibt wie zuvor $\textrm{TV} = 0.6 - 0.2 = 0.4$. Für zwei stetige Dichten zeigt :numref:`fig_mdl-tv-area` dieselben geometrischen Verhältnisse. Der Totalvariationsabstand ist zudem eine echte Metrik. Beide Darstellungen sind offensichtlich symmetrisch, und die Dreiecksungleichung folgt direkt aus der $\ell_1$-Formel (Aufgabe :exref:`ex_mdl-divergences-distances-prove-that-total-variation`a).

<!--en:69f00d14-->
![Zwei Dichten mit gleicher Varianz und die Gebiete, in denen sie sich unterscheiden. Die Fläche zwischen den Kurven im Bereich $p > q$ ist genauso groß wie im Bereich $q > p$. Dieser gemeinsame Flächeninhalt ist der Totalvariationsabstand. Das Ereignis, bei dem das Supremum erreicht wird, ist $A^\star = \{p > q\}$, hier die Halbgerade links vom Schnittpunkt. Der optimale Test anhand einer einzelnen Beobachtung entscheidet dort für $P$ und sonst für $Q$.](../img/mdl-it-tv-area.svg)

<!--en:1bf64935-->
Die Darstellung als Supremum gibt dem TV-Abstand eine konkrete statistische Bedeutung. Stellen wir uns einen Test vor, der eine einzelne Beobachtung erhält. Sie stammt mit gleicher Wahrscheinlichkeit $\tfrac12$ aus $P$ oder aus $Q$. Der Test soll entscheiden, welche Verteilung sie erzeugt hat. Jeder deterministische Test entscheidet für „$P$“, wenn die Beobachtung in einem bestimmten Gebiet $A$ liegt, und sonst für „$Q$“. Seine Erfolgswahrscheinlichkeit beträgt

<!--en:64ad1ff1-->
wobei für $A = A^\star$ Gleichheit gilt. Jeder Test hat für ein geeignetes $A$ diese Form (ein randomisierter Test mittelt lediglich über deterministische Tests und kann nicht besser abschneiden). Selbst der optimale Test übertrifft die Erfolgswahrscheinlichkeit eines fairen Münzwurfs also um höchstens $\textrm{TV}/2$. Bei unserem Beispielpaar entscheidet der optimale Test beim ersten Ergebnis für „$P$“ und liegt in $70\%$ der Fälle richtig. Sowohl der Diskriminator eines GAN, der echte Daten von erzeugten Samples unterscheiden soll, als auch ein Detektor für maschinell erzeugten Text sind solche Tests. Der Totalvariationsabstand begrenzt die Genauigkeit, die beide mit einer einzelnen Beobachtung erreichen können.

<!--en:59b67f66-->
### Pinsker-Ungleichung

<!--en:ea79c1e6-->
Der Totalvariationsabstand begrenzt die Erfolgswahrscheinlichkeit jedes statistischen Tests. In der Praxis können wir meist die KL-Divergenz berechnen oder beschränken, weil die Verlustfunktionen neuronaler Netze auf logarithmierten Wahrscheinlichkeiten beruhen. Die Pinsker-Ungleichung verbindet beide Größen. Eine kleine KL-Divergenz garantiert, dass kein Test zuverlässig $P$ von $Q$ unterscheiden kann. Ein klassisches Beispiel sind die Trust-Regionen für Strategien in :numref:`sec_ppo`. Die Garantie einer monotonen Verbesserung wird mithilfe des Totalvariationsabstands bewiesen, aber durch eine Schranke für die KL-Divergenz durchgesetzt.

<!--en:dd4919eb-->
**Satz (Pinsker-Ungleichung).** :cite:`Pinsker.1964,Csiszar.1967` *Für Verteilungen $P$ und $Q$ auf einer endlichen oder abzählbaren Menge gilt*

<!--en:660dbf0f-->
**Beweis.** *Schritt 1: zwei Ergebnisse.* Wir schreiben $\textrm{Ber}(a)$ für eine Münze, die mit Wahrscheinlichkeit $a$ Kopf zeigt. Für $a \in [0, 1]$ und $b \in (0, 1)$ gilt $D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b)) = a \log\frac{a}{b} + (1-a) \log\frac{1-a}{1-b}$ und $\textrm{TV}(\textrm{Ber}(a), \textrm{Ber}(b)) = |a - b|$. Wir müssen also zeigen, dass $\rho(a) = D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b)) - 2(a - b)^2$ nichtnegativ ist. Halten wir $b$ fest. Offenbar gilt $\rho(b) = 0$. Auch die erste Ableitung $\rho'(a) = \log\frac{a}{b} - \log\frac{1-a}{1-b} - 4(a - b)$ verschwindet bei $a = b$. Für $0 < a < 1$ ist die zweite Ableitung

<!--en:b1a61d77-->
denn es gilt $a(1-a) \leq \tfrac{1}{4}$. Eine konvexe Funktion, deren Wert und Steigung bei $a = b$ null sind, ist auf ganz $(0, 1)$ nichtnegativ. Aufgrund der Stetigkeit (mit $0 \log 0 = 0$) gilt das auch an den Randpunkten $a = 0$ und $a = 1$. Ist $b$ gleich $0$ oder $1$, so ist die KL-Divergenz außer für $a = b$ unendlich. Die Ungleichung gilt dann unmittelbar.

<!--en:c929925d-->
*Schritt 2: Reduktion auf zwei Ergebnisse.* Sei $A^\star = \{p > q\}$. Setzen wir $a = P(A^\star)$ und $b = Q(A^\star)$, so gilt $\textrm{TV}(P, Q) = a - b$. Wir vergröbern den Ergebnisraum, indem wir alle Ergebnisse innerhalb von $A^\star$ zusammenfassen und ebenso alle außerhalb von $A^\star$. Dadurch kann die KL-Divergenz nur *abnehmen*. Das folgt aus der Log-Summen-Ungleichung. Für nichtnegative Zahlen $v_i$ und $w_i$ gilt

<!--en:08b6b7ed-->
wie die Anwendung der Jensen-Ungleichung auf $t \mapsto t\log t$ zeigt. Wenden wir die Log-Summen-Ungleichung getrennt auf $A^\star$ und sein Komplement an, so ergibt sich

<!--en:a186e1e1-->
und nach Umstellen :eqref:`eq_mdl-pinsker`. $\blacksquare$

<!--en:2e33030b-->
Für unser Beispielpaar lautet die Pinsker-Ungleichung $0.4 \leq \sqrt{0.396/2} = 0.445$. Schritt 2 lässt sich von Hand nachvollziehen. Fassen wir das zweite und dritte Ergebnis zusammen, bleiben zwei Münzen mit den Verteilungen $(0.6, 0.4)$ und $(0.2, 0.8)$. Ihr TV-Abstand bleibt bei $0.4$. Ihre KL-Divergenz sinkt von $0.396$ auf $0.6 \log 3 + 0.4 \log \tfrac12 = 0.382$ Nat, und Schritt 1 liefert dafür die untere Schranke $2\,(0.4)^2 = 0.32$.

<!--en:9306d0fc-->
Vergleichen wir eine Münze mit Kopfwahrscheinlichkeit $0.6$ mit einer fairen Münze, so beträgt die KL-Divergenz $0.6 \log 1.2 + 0.4 \log 0.8 = 0.0201$ Nat, und es gilt $2\,(0.1)^2 = 0.02$. Für nahezu gleiche Münzen ist die Pinsker-Ungleichung also fast scharf. Entscheidend ist jedoch, dass es keine brauchbare Umkehrung gibt. Setzt ein Modell auch nur für ein seltenes Ergebnis $q = 0$, so ist die KL-Divergenz unendlich, selbst bei winzigem TV-Abstand. Auch für weit auseinanderliegende Verteilungen verliert die Schranke ihre Aussagekraft, weil TV nie größer als $1$ ist. Für zwei Gaußverteilungen mit Varianz eins und Mittelwerten im Abstand $50$ gilt $\textrm{TV} \approx 1$, während :eqref:`eq_mdl-dd-gaussian-kl` eine KL-Divergenz von $= 50^2/2 = 1250$ Nat ergibt.

<!--en:01d859ab-->
Wie scharf ist die Pinsker-Schranke bei typischen Verteilungen? Lässt sich die Konstante $\tfrac12$ verbessern? Für die erste Frage ziehen wir $10{,}000$ zufällige Paare von Verteilungen auf fünf Ergebnissen gleichverteilt aus dem Simplex. Für die zweite lassen wir die Kopfwahrscheinlichkeiten zweier Münzen bei $a = b = \tfrac12$ zusammenrücken. Dort ging in Schritt 1 bis zur zweiten Ordnung nichts verloren. Das Verhältnis $\textrm{TV}/\sqrt{\textrm{KL}/2}$ sollte daher bis auf $1$ steigen.

<!--en:22062bea-->
Alle Punkte im Streudiagramm liegen unter der Kurve $\sqrt{\textrm{KL}/2}$. Im Median erreichen die Paare etwa das $0.8$-Fache der Schranke. Unter den $10{,}000$ zufälligen Paaren kommt das mit dem größten Verhältnis bis auf wenige Hundertstel an $1$ heran. Sobald $\textrm{KL} > 2$ gilt, steigt die Kurve über $1$ und die Schranke verliert ihre Aussagekraft. Für nahezu faire Münzen steigt das Verhältnis über $0.9899$ und $0.9999$ auf $0.999999$. Das bestätigt, dass sich die Konstante $\tfrac{1}{2}$ nicht weiter verbessern lässt (Aufgabe :exref:`ex_mdl-divergences-distances-sharpness-pinsker-coins-biases`c). Pinskers ursprünglicher Beweis lieferte eine schwächere Konstante. Der scharfe Faktor $\tfrac{1}{2}$ geht auf :citet:`Csiszar.1967` und unabhängig davon auf Kullback und Kemperman zurück.

<!--en:092dab87-->
### Datenverarbeitung

<!--en:b40cdb54-->
Können wir zwei Verteilungen leichter unterscheiden, wenn wir Merkmale extrahieren, Werte quantisieren oder die Daten anders nachverarbeiten? In Pinskers Beweis konnte das Zusammenfassen von Ergebnissen die KL-Divergenz nur verringern. Die Kettenregel aus :numref:`subsec_mdl-kl-chain-rule` zeigte dasselbe für das Weglassen einer Variablen eines Paares. Dieses Prinzip gilt für jede f-Divergenz und jede Art der Verarbeitung. Ein *Kanal* $\kappa(y \mid x)$ ist eine beliebige deterministische oder randomisierte Vorschrift, die eine Eingabe $x$ in eine Ausgabe $y$ umwandelt. Schicken wir $P$ durch den Kanal, so erhalten wir die Ausgabeverteilung $(P\kappa)(y) = \sum_x \kappa(y \mid x)\, p(x)$. Für $Q$ ergibt sich entsprechend $Q\kappa$.

<!--en:34c174eb-->
**Satz (Datenverarbeitung für f-Divergenzen).** *Für jeden Kanal $\kappa$, jede konvexe Funktion $f$ mit $f(1) = 0$ und $P \ll Q$ auf einer endlichen oder abzählbaren Menge gilt*

<!--en:d748665a-->
**Beweis.** Wegen $\sum_y \kappa(y \mid x) = 1$ können wir $D_f(P\|Q) = \sum_y \sum_x \kappa(y \mid x)\, q(x)\, f\big(p(x)/q(x)\big)$ schreiben. Wir halten eine Ausgabe $y$ mit $(Q\kappa)(y) > 0$ fest. Die normierten Gewichte $\kappa(y \mid x)\, q(x) / (Q\kappa)(y)$ summieren sich über $x$ zu eins. Mitteln wir die Dichteverhältnisse $p(x)/q(x)$ mit diesen Gewichten, erhalten wir $(P\kappa)(y)/(Q\kappa)(y)$. Die Jensen-Ungleichung liefert für die innere Summe daher die untere Schranke $(Q\kappa)(y)\, f\big((P\kappa)(y)/(Q\kappa)(y)\big)$. Das ist der $y$-te Term von $D_f(P\kappa\|Q\kappa)$. Für jede Ausgabe $y$ mit $(Q\kappa)(y) = 0$ muss $\kappa(y \mid x) = 0$ überall dort gelten, wo $q(x) > 0$ ist. Damit gilt auch $(P\kappa)(y) = 0$, und die Ausgabe trägt auf keiner Seite etwas bei. Summieren über $y$ schließt den Beweis ab. $\blacksquare$

<!--en:05a11642-->
Unabhängig von der gewählten Divergenzfunktion kann die Verarbeitung der Daten die Unterscheidbarkeit zweier Verteilungen nie erhöhen. Im Deep Learning lesen wir diese Ungleichung oft in umgekehrter Richtung. Messen wir eine Divergenz anhand erlernter Merkmale, quantisierter Ausgaben oder einer anderen Zusammenfassung der Daten, so erhalten wir stets eine untere Schranke für die Divergenz zwischen den ursprünglichen Eingaben. Das Zusammenfassen von Ergebnissen in Pinskers Beweis entspricht dem deterministischen Kanal $x \mapsto \mathbf{1}(x \in A^\star)$. Bei unserem Beispielpaar senkte er die KL-Divergenz von $0.396$ auf $0.382$. Die $\chi^2$-Divergenz sinkt von $1.013$ auf $0.16/0.2 + 0.16/0.8 = 1$. Auf dem Spezialfall der KL-Divergenz beruht auch die Datenverarbeitungsungleichung für die gegenseitige Information in :numref:`subsec_mdl-dpi`.

<!--en:32d6d2da-->
Eine umkehrbare Transformation wirkt in beiden Richtungen als Kanal. Daher gilt in :eqref:`eq_mdl-f-div-dpi` Gleichheit. Jede f-Divergenz bleibt also unverändert, wenn wir Ergebnisse umbenennen oder Achsen umskalieren. Die letzten beiden Zeilen von :numref:`tab_mdl-point-mass-table` ändern sich hingegen bei einer Umskalierung der Achse. Diese Abstände erfassen die Geometrie über $d$, verlieren dabei aber die Skaleninvarianz.

<!--en:a8ffb947-->
## Fenchel-Konjugierte und Kritikerschranke

<!--en:5fc56f42-->
Angenommen, ein generatives Netz erzeugt Samples aus einer Verteilung $Q$. Daneben haben wir einen Datensatz mit Beobachtungen aus der wahren Verteilung $P$. Keine der beiden Stichproben liefert das Dichteverhältnis $p(x)/q(x)$, das wir für :eqref:`eq_mdl-f-div-def` brauchen. Können wir $D_f$ dennoch allein aus Stichproben schätzen? Tatsächlich reichen Erwartungswerte unter $P$ und $Q$ aus, um jede f-Divergenz nach unten abzuschätzen. Über diese untere Schranke können wir jede Divergenz als Zielfunktion für adversariales Training nutzen.

<!--en:2be8bfb9-->
### Herleitung der Schranke

<!--en:a3980efa-->
Wir verwenden die konvex Konjugierte aus :numref:`subsec_mdl-convex-conjugate`. Für eine konvexe Funktion $f$ ist ihre Konjugierte definiert durch

<!--en:fbcd0ec7-->
Damit ist $-f^*(t)$ der vertikale Achsenabschnitt der Tangente an $f$ mit Steigung $t$ (:numref:`fig_mdl-f-gan-tangent-bound`). Für $f(u) = u \log u$ liegt das Supremum dort, wo $\log u + 1 = t$ gilt. Das ergibt $f^*(t) = e^{t-1}$. Da der Graph einer konvexen Funktion überall auf oder über jeder seiner Tangenten liegt, erhalten wir die Fenchel-Young-Ungleichung:

<!--en:feb71d18-->
Gleichheit gilt, wenn $t$ mit der Steigung von $f$ bei $u$ übereinstimmt, also für $t = f'(u)$.

<!--en:44641f2d-->
![Eine konvexe Divergenzfunktion, hier $f(u)=u\log u$, ist die obere Einhüllende ihrer Tangenten. Die Tangente mit Steigung $t$ hat den vertikalen Achsenabschnitt $-f^*(t)$, hier $-e^{t-1}$. Eine einzelne Tangente liefert die untere Fenchel-Young-Schranke. Das Supremum über alle Steigungen ergibt wieder $f(u)$. Wenden wir diese Ungleichung punktweise auf das Dichteverhältnis an, so erhalten wir die Kritikerschranke.](../img/mdl-it-f-gan-tangent-bound.svg)

<!--en:12dea593-->
Wenden wir nun :eqref:`eq_mdl-fenchel-young` *punktweise innerhalb der Divergenz* an und lassen wir die Steigung von $x$ abhängen. Dazu wählen wir eine beliebige Funktion $T(x)$, den **Kritiker** (engl. critic), und setzen $u = p(x)/q(x)$ und $t = T(x)$ ein.

<!--en:432775c7-->
**Satz (Kritikerschranke).** *Sei $f$ konvex und differenzierbar mit $f(1) = 0$. Es gelte $P \ll Q$, und $T$ sei ein beliebiger Kritiker mit $\mathbb{E}_P|T| < \infty$ und $\mathbb{E}_Q|f^*(T)| < \infty$. Dann gilt*

<!--en:f2057d1b-->
*Außerdem ist das Supremum der rechten Seite über alle Kritiker gleich $D_f(P\|Q)$. Der Kritiker $T^\star(x) = f'\big(p(x)/q(x)\big)$ erreicht es, sofern $T^\star$ überall endlich ist* :cite:`Nguyen.Wainwright.Jordan.2010,Nowozin.Cseke.Tomioka.2016`.

<!--en:95e83cc5-->
**Beweis.** Wir multiplizieren :eqref:`eq_mdl-fenchel-young` an den Stellen $u = p(x)/q(x)$ und $t = T(x)$ mit $q(x) \geq 0$:

<!--en:275c6a9f-->
Summieren (oder Integrieren) über $x$ ergibt :eqref:`eq_mdl-f-gan-bound`. In der Fenchel-Young-Ungleichung gilt bei $t = f'(u)$ Gleichheit. Wählen wir $T^\star(x) = f'(p(x)/q(x))$, so gilt die Gleichheit für jedes $x$ und damit auch für den Erwartungswert. $\blacksquare$

<!--en:7684a090-->
Die rechte Seite von :eqref:`eq_mdl-f-gan-bound` enthält zwei Erwartungswerte. Wir mitteln $T$ über echte Daten und $f^*(T)$ über erzeugte Samples. Explizite Dichten benötigen wir dafür nicht. Wir können $T$ daher als neuronales Netz parametrisieren und es mit Minibatches darauf trainieren, die Schranke zu maximieren. Für die KL-Divergenz lautet die Schranke $D_{\textrm{KL}}(P\|Q) \geq \mathbb{E}_P[T] - \mathbb{E}_Q[e^{T - 1}]$ für jeden Kritiker $T$. Das **f-GAN** von :citet:`Nowozin.Cseke.Tomioka.2016` (nach dem :eqref:`eq_mdl-f-gan-bound` auch f-GAN-Schranke heißt) macht aus dieser Variationsdarstellung ein Spiel mit zwei Spielern. Der Kritiker maximiert die Schranke, damit sie die Divergenz möglichst genau erfasst. Der Generator, der $Q$ erzeugt, minimiert sie. In :numref:`sec_gan_objectives` implementieren und trainieren wir solche Paare.

<!--en:0bcb2e8f-->
### Das ursprüngliche GAN

<!--en:b0362136-->
Das ursprüngliche GAN von :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` ergibt sich direkt aus diesem Ansatz, wenn wir $f(u) = u\log u - (u+1)\log(u+1)$ wählen. Seine Divergenzfunktion hat die konvex Konjugierte $f^*(t) = -\log(1 - e^t)$ für $t < 0$. Schreiben wir den Kritiker als Logarithmus eines Diskriminators $D(x) \in (0, 1)$ und setzen $T = \log D$, so ergibt die Kritikerschranke das bekannte Spiel mit binärer Kreuzentropie:

<!--en:1812c35f-->
Der optimale Kritiker $T^\star = f'(p/q) = \log\big(p/(p+q)\big)$ entspricht dem Bayes-optimalen Diskriminator $D^\star = p/(p + q)$. Der Vergleich mit der Jensen-Shannon-Divergenzfunktion $f_{\textrm{JS}}$ in :numref:`tab_mdl-f-div-generators` ergibt $f(u) = 2 f_{\textrm{JS}}(u) - (u - 1)\log 2 - \log 4$. Wir erhalten also das Doppelte der Jensen-Shannon-Funktion plus einen linearen Term in $u - 1$ (der keine Divergenz ändert) und die Konstante $f(1) = -\log 4$ (die nur den Wert verschiebt). Bei $D^\star$ erreicht das Minimax-Spiel daher den Wert $2\,\textrm{JS}(P, Q) - \log 4$, für unser Beispielpaar also $2(0.0911) - 1.386 = -1.204$.

<!--en:a7be20cc-->
### Grenzen der Schranke

<!--en:70bec6aa-->
In der Praxis hat die Kritikerschranke zwei Einschränkungen. Die erste können wir direkt messen. Da die Schranke nur bei $T^\star$ scharf ist, liefert ein unzureichend trainierter Kritiker einen Wert unterhalb der wahren Divergenz. Beobachten wir, wie sich diese Lücke beim Training schließt.

<!--en:5211c233-->
Für unser diskretes Beispielpaar sind die Erwartungswerte einfache Summen. Der Kritiker ist ein Vektor aus drei Zahlen, einer für jedes Ergebnis. Betrachten wir die $\chi^2$-Divergenz mit $f(u) = (u-1)^2$, der Konjugierten $f^*(t) = t + t^2/4$ und dem optimalen Kritiker $T^\star = 2(p/q - 1)$. (Hier bilden wir für die Konjugierte das Supremum über alle reellen $u$. In Aufgabe :exref:`ex_mdl-divergences-distances-derive-convex-conjugate`a leiten Sie die Konjugierte her und prüfen, dass diese Wahl bei $T^\star$ nichts ändert.) Wir starten bei $T = 0$, wo die Schranke $0$ beträgt, und maximieren sie durch Gradientenaufstieg mit Schrittweite $1$. Ihre Ableitung nach $T(x)$ ist $p(x) - q(x)\,\big(1 + T(x)/2\big)$. Nach dem Satz muss sich die Schranke dabei von unten an $\chi^2(P\|Q) = 1.0133$ annähern. Zur Kontrolle wertet die letzte Zeile auch einen übermäßig selbstsicheren Kritiker $2T^\star$ aus, der in die richtige Richtung weist, aber um den Faktor zwei zu weit geht.

<!--en:c1300424-->
Nach fünf Schritten weist der Kritiker $0.70$ von $1.01$ nach (etwa $70\%$ der wahren Divergenz). Nach zehn Schritten erreicht er $90\%$ und nach zwanzig Schritten $99\%$. Die Schranke überschreitet den wahren Wert nie. Selbst der übermäßig selbstsichere Kritiker $2T^\star$ liefert null und schneidet damit nicht besser ab als ein untrainierter Kritiker. Bei exakten Erwartungswerten liegt jede adversariale Schätzung der Divergenz höchstens beim wahren Wert. An der Zahl selbst erkennen wir nie, wie viel fehlt. Ersetzen wir Erwartungswerte durch Mittelwerte endlicher Stichproben, so gilt die untere Schranke im Erwartungswert nur für neue Stichproben, die nicht zum Training des Kritikers dienten.

<!--en:1edf7ce9-->
Der Gradientenschritt zeigt auch, woher der fehlende Teil der Divergenz kommt. Da sich $T(x)$ mit einer zu $q(x)$ proportionalen Rate ändert, lernt der Kritiker dort am langsamsten, wo das Modell am seltensten Samples erzeugt. Bei unserem Beispielpaar ist das das erste Ergebnis. Dort ist das Dichteverhältnis am größten, und von dort stammt der größte Teil der Divergenz (Aufgabe :exref:`ex_mdl-divergences-distances-derive-convex-conjugate`b).

<!--en:bf99148e-->
Die zweite Einschränkung führt uns zurück zu disjunkten Trägern, diesmal aus Sicht des Kritikers. Erweitern wir $T^\star = f'(p/q)$ formal über die Voraussetzung $P \ll Q$ hinaus, so ist das Verhältnis auf dem Träger der Daten gleich $\infty$ und auf dem Träger des Modells gleich $0$. Damit nimmt $T^\star$ auf dem einen den konstanten Wert $f'(\infty)$ an und auf dem anderen $f'(0)$. Der Generator aktualisiert seine Parameter mit dem Gradienten des Kritikers an seinen eigenen Samples. Ist der Kritiker dort lokal konstant, so ist sein Gradient null. Der Generator erhält kein Signal, in welche Richtung er sich bewegen soll.

<!--en:9880b201-->
Beide Einschränkungen haben dieselbe Ursache, das Dichteverhältnis $p/q$. Endliche Stichproben liefern uns $p/q$ nie direkt. Sind die Träger disjunkt, nimmt das Verhältnis nur noch die Werte $0$ oder $\infty$ an. Wie weit die Verteilungen auseinanderliegen, ist daran nicht mehr zu erkennen. In :numref:`sec_mdl-test-functions-transport` wenden wir uns Vergleichsmaßen zu, die ganz ohne Dichteverhältnisse auskommen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:84115fb8-->
Die KL-Divergenz gehört zu einer großen Familie. Eine f-Divergenz mittelt eine konvexe Funktion des Dichteverhältnisses $p/q$ unter $Q$. Die Jensen-Ungleichung garantiert die Nichtnegativität aller Familienmitglieder. Bis auf den Faktor ihrer Krümmung $f''(1)$ stimmen alle glatten Mitglieder näherungsweise überein, wenn $P$ und $Q$ nahe beieinanderliegen. Keine Nachverarbeitung der Daten kann eine f-Divergenz erhöhen. Die Richtung der KL-Divergenz wählen wir bei der Modellierung. Die Vorwärts-KL (Maximum-Likelihood) dehnt das Modell über alle Modi aus, während die Rückwärts-KL (Variationsinferenz) es auf einen einzigen Modus konzentrieren kann. Der beste Test, der anhand einer einzelnen Beobachtung $P$ von $Q$ unterscheiden soll, entscheidet mit Wahrscheinlichkeit $\tfrac12(1 + \textrm{TV})$ richtig. Die Pinsker-Ungleichung beschränkt TV nach oben durch $\sqrt{D_{\textrm{KL}}/2}$. Mithilfe der konvex Konjugierten liefert jeder Kritiker eine untere Schranke für $D_f$, die nur Erwartungswerte unter $P$ und $Q$ benötigt. So schätzt adversariales Training Divergenzen aus Stichproben. Alle Mitglieder der Familie haben jedoch dieselbe Schwäche, die auf das Dichteverhältnis zurückgeht. Auf disjunkten Trägern nimmt $D_f$ den konstanten Wert $f(0) + f'(\infty)$ an, unabhängig davon, wie nahe die Träger beieinanderliegen.

<!--en:2294243a-->
## Übungen

<!--en:27ef4aef-->
1. **Divergenzfunktionen und Reihenfolge der Argumente.** Jede Zeile von :numref:`tab_mdl-f-div-generators` entspricht einer Wahl von $f$ in der allgemeinen Form :eqref:`eq_mdl-f-div-def`.
    1. Definieren Sie für eine Divergenzfunktion $f$ die Funktion $f^\diamond(u) = u\, f(1/u)$. Zeigen Sie, dass $f^\diamond$ konvex ist mit $f^\diamond(1) = 0$ und dass $D_f(Q\|P) = D_{f^\diamond}(P\|Q)$ gilt. Berechnen Sie $f^\diamond$ für die KL-Funktion $u \log u$ und die $\chi^2$-Funktion $(u - 1)^2$. Schreiben Sie die Divergenz, die letztere definiert, als Summe über $x$.
    1. Die Addition von $c\,(u - 1)$ zu $f$ ändert die Divergenz nicht. Zeigen Sie, dass $D_f$ symmetrisch ist, wenn $f^\diamond(u) = f(u) + c\,(u - 1)$ für eine Konstante $c$ gilt. Prüfen Sie diese Bedingung für die als symmetrisch gekennzeichneten Zeilen in :numref:`tab_mdl-f-div-generators`. Zeigen Sie, dass $D_{f_\alpha}(Q\|P) = D_{f_{1-\alpha}}(P\|Q)$ für die Funktion :eqref:`eq_mdl-alpha-generator` gilt. Bestätigen Sie mit dieser Identität den Eintrag der $\alpha$-Divergenz in der Spalte „Symmetrisch?“.
    1. Drücken Sie :eqref:`eq_mdl-js-def` durch das Verhältnis $u = p/q$ aus und leiten Sie daraus die Jensen-Shannon-Funktion in :numref:`tab_mdl-f-div-generators` her.
    1. *Rényi-Divergenz.* Definieren Sie für $\alpha > 0$, $\alpha \neq 1$ die Divergenz $D_\alpha(P\|Q) = \frac{1}{\alpha - 1} \log \sum_x p(x)^\alpha\,
       q(x)^{1-\alpha}$ :cite:`Renyi.1961`. Zeigen Sie, dass $\sum_x p^\alpha q^{1-\alpha} = 1 + \alpha(\alpha - 1)\,
       D_{f_\alpha}(P\|Q)$ gilt und dass $D_\alpha$ eine wachsende Funktion von $D_{f_\alpha}$ ist. Folgern Sie, dass $D_\alpha \geq 0$ gilt und $D_\alpha$ die Datenverarbeitungsungleichung :eqref:`eq_mdl-f-div-dpi` erfüllt. Bestimmen Sie $\lim_{\alpha \to 0} D_\alpha(P\|Q)$. Erklären Sie, warum dieser Grenzwert gleich $0$ ist, wenn $p$ und $q$ auf denselben Ergebnissen positiv sind. Warum konvergiert die Rényi-Familie daher für $\alpha \to 0$ *nicht* gegen die Rückwärts-KL, im Gegensatz zu $D_{f_\alpha}$?
   :exlabel:`ex_mdl-divergences-distances-recover-kl-reverse-kl`

<!--en:92bd821f-->
1. **Metriken unter den Divergenzen.** Die letzte Spalte von :numref:`tab_mdl-f-div-generators` gibt an, welche Divergenzen Metriken sind.
    1. Beweisen Sie die Dreiecksungleichung für TV mithilfe der Darstellung als halber $\ell_1$-Abstand.
    1. Zeigen Sie, dass der Hellinger-Abstand $d_{\mathrm{H}} = \sqrt{d_{\mathrm{H}}^2}$ der euklidische Abstand zwischen den Vektoren $\big(\sqrt{p(x)}\big)_x$ und $\big(\sqrt{q(x)}\big)_x$ ist und somit $d_{\mathrm{H}}$ eine Metrik ist. Finden Sie drei Verteilungen auf zwei Ergebnissen, für die sowohl $d_{\mathrm{H}}^2$ als auch JS die Dreiecksungleichung verletzen.
   :exlabel:`ex_mdl-divergences-distances-prove-that-total-variation`

<!--en:495d33d7-->
1. **Kosten eines ausgelassenen Modus.** Ein Modell $Q$ erfasst einen Teil einer Datenverteilung $P$ nicht.
    1. Sei $P = (1 - w)\, Q + w R$ mit $0 < w < 1$. Dabei sei $R$ eine Verteilung auf einer Menge, der $Q$ die Wahrscheinlichkeit null zuweist. Berechnen Sie $D_{\textrm{KL}}(P\|Q)$, $D_{\textrm{KL}}(Q\|P)$, $\textrm{TV}(P, Q)$ und $d_{\mathrm{H}}^2(P, Q)$ als Funktionen von $w$. Welche Richtung der KL-Divergenz bestraft die fehlende Masse? Wie verhalten sich die endlichen Divergenzen für $w \to 0$?
    1. Sei nun $P = w_1 \mathcal{N}(m_1, s^2) + w_2 \mathcal{N}(m_2, s^2)$ mit dem Abstand $\Delta = |m_2 - m_1|$ zwischen den Mittelwerten. Das Modell $Q = \mathcal{N}(m_1, s^2)$ behalte nur die erste Komponente bei. Zeigen Sie, dass $D_{\textrm{KL}}(Q\|P) \leq \log(1/w_1)$ für jedes $\Delta$ gilt, mit Gleichheit im Grenzfall $\Delta \to \infty$. Stellen Sie den Zusammenhang mit (a) her. Erklären Sie anhand der Schranke die beiden Werte der Rückwärts-KL von $0.356$ und $1.202$ Nat, die der Code bei der Anpassung einer Gaußverteilung an die Mischung ausgibt.
    1. Nehmen Sie an, dass die Komponenten so weit auseinanderliegen, dass für fast jedes $x$ höchstens eine Komponente eine nicht vernachlässigbare Dichte hat. Zeigen Sie, dass $D_{\textrm{KL}}(P\|Q) \approx w_2 \Delta^2 / (2 s^2)
       + w_1 \log w_1 + w_2 \log w_2$ gilt. Werten Sie die Näherung für die Mischung aus dem Experiment aus und überprüfen Sie sie mit `kl(log_p(x), log_q(x, [-2.0, np.log(0.6)]), x)`. Zum Vergleich gibt das Experiment in der Spalte für die Vorwärts-KL zwei Werte aus. Bei der Anpassung mit Rückwärts-KL sind es $5.981$ Nat, da die Parameter $(-1.998, 0.603)$ leicht von $(-2, 0.6)$ abweichen. Bei der Anpassung mit Vorwärts-KL sind es $0.558$ Nat.
   :exlabel:`ex_mdl-divergences-distances-mode-dropping`

<!--en:32ff2231-->
1. **Lokale quadratische Näherung.** Der Satz :eqref:`eq_mdl-f-div-local` begrenzt die Abweichung zwischen $D_f$ und $\tfrac12 f''(1)\, \chi^2$.
    1. In Aufgabe :exref:`ex_mdl-information-theory-gaussian-kl`b aus :numref:`sec_mdl-information_theory` bestimmen Sie die Ordnung, bei der sich die beiden Richtungen der KL-Divergenz für Gaußverteilungen erstmals unterscheiden. Welche Eigenschaft der Funktionen für KL und Rückwärts-KL erklärt diese Ordnung? Führen Sie die Taylor-Entwicklung aus dem Beweis um einen Term weiter. Zeigen Sie, dass das im Experiment zur lokalen quadratischen Näherung ausgegebene Verhältnis $1 + c\, t\, f'''(1)/f''(1) + O(t^2)$ beträgt, wobei $c = \mathbb{E}_Q[(u - 1)^3] / \big(3 \chi^2(P\|Q)\big)$ für unser Beispielpaar etwa $0.49$ ist. Berechnen Sie $f'''(1)/f''(1)$ für die vier Funktionen. Warum ist die Abweichung nach unten bei der Rückwärts-KL für $t = 0.01$ doppelt so groß wie bei der KL-Divergenz? Warum ergibt sich dort für den quadrierten Hellinger-Abstand und JS derselbe Wert?
    1. Die Funktion $\tfrac{1}{2}|u - 1|$ des Totalvariationsabstands hat bei $1$ keine zweite Ableitung. Sei $P_t = Q + t\,(P - Q)$ für $t \in (0, 1]$. Wie skalieren $\textrm{TV}(P_t, Q)$ und $\chi^2(P_t\|Q)$ mit $t$? Erklären Sie damit, warum die Pinsker-Ungleichung :eqref:`eq_mdl-pinsker` den TV-Abstand durch die Quadratwurzel der KL-Divergenz beschränkt und nicht durch ein Vielfaches der KL-Divergenz.
    1. Seien $P$ und $Q$ Münzen mit den Kopfwahrscheinlichkeiten $b$ und $b + \delta$. Zeigen Sie mit dem Satz, dass $D_{\textrm{KL}}(P\|Q) = \delta^2 / \big(2 b (1 - b)\big) + O(\delta^3)$ gilt. In Aufgabe :exref:`ex_mdl-information-theory-two-part-coin`a aus :numref:`sec_mdl-coding-compression` erhalten Sie dieselbe Entwicklung direkt aus der KL-Divergenz für Bernoulli-Verteilungen. Der Faktor $1/\big(b(1-b)\big)$ ist die Fisher-Information der Münze (:numref:`sec_mdl-maximum_likelihood`). Bestimmen Sie den Grenzwert von $\textrm{TV}/\sqrt{D_{\textrm{KL}}/2}$ für $\delta \to 0$ und die Werte von $b$, für die die Pinsker-Ungleichung in diesem Grenzfall scharf ist. Das Münzexperiment nach der Pinsker-Ungleichung verwendet $b = \tfrac{1}{2}$ und liefert ein Verhältnis von $0.999999$ bei $\delta = 10^{-3}$. Warum nähert sich das Verhältnis dem Wert $1$ mit der Rate $\delta^2$ und nicht mit $\delta$?
   :exlabel:`ex_mdl-divergences-distances-sharpness-pinsker-coins-biases`

<!--en:16f7f244-->
1. **Hellinger-Schranken und statistische Tests.** ● Die Pinsker-Ungleichung verliert ihre Aussagekraft, sobald $D_{\textrm{KL}} > 2$ gilt. Sei $\textrm{BC}(P, Q) = \sum_x \sqrt{p(x) q(x)} = 1 - \tfrac{1}{2} d_{\mathrm{H}}^2(P, Q)$ der Bhattacharyya-Koeffizient.
    1. Beweisen Sie $\tfrac{1}{2} d_{\mathrm{H}}^2(P, Q) \leq \textrm{TV}(P, Q)$ mithilfe von $|p - q| = |\sqrt{p} - \sqrt{q}|\,(\sqrt{p} + \sqrt{q})$.
    1. Beweisen Sie $\textrm{TV}(P, Q) \leq d_{\mathrm{H}}(P, Q) \sqrt{1 - \tfrac{1}{4} d_{\mathrm{H}}^2(P, Q)}$ mit der Cauchy-Schwarz-Ungleichung aus :numref:`sec_mdl-geometry-linear-algebraic-ops` und schreiben Sie die Schranke in der Form $\textrm{TV}^2 \leq 1 - \textrm{BC}^2$.
    1. Zeigen Sie mit der Jensen-Ungleichung, dass $\textrm{BC}(P, Q) \geq e^{-D_{\textrm{KL}}(P\|Q)/2}$ gilt, und folgern Sie daraus die Bretagnolle-Huber-Ungleichung $\textrm{TV}(P, Q) \leq \sqrt{1 - e^{-D_{\textrm{KL}}(P\|Q)}}$ :cite:`Bretagnolle.Huber.1978,Canonne.2022`. Für welche Werte von $D_{\textrm{KL}}$ ist sie schärfer als :eqref:`eq_mdl-pinsker`?
    1. Ein Test erhält $n$ unabhängige Beobachtungen, entweder alle aus $P$ oder alle aus $Q$. Beide Fälle treten mit Wahrscheinlichkeit $\tfrac{1}{2}$ ein. Verwenden Sie (c) und die Additivität der KL-Divergenz für unabhängige Beobachtungen aus :numref:`subsec_mdl-kl-chain-rule`. Zeigen Sie damit, dass die Fehlerwahrscheinlichkeit jedes Tests mindestens $\tfrac{1}{4} e^{-n D_{\textrm{KL}}(P\|Q)}$ beträgt. Wie viele Beobachtungen sind für eine Fehlerwahrscheinlichkeit von höchstens $\eta$ nötig? Warum kann die Pinsker-Ungleichung keine solche Schranke liefern?
   :exlabel:`ex_mdl-divergences-distances-hellinger-bretagnolle-huber`

<!--en:984b47c7-->
1. **Konjugierte und Kritiker.** Die Kritikerschranke :eqref:`eq_mdl-f-gan-bound` benötigt die Konjugierte $f^*$ der Divergenzfunktion.
    1. Leiten Sie die Konjugierte $f^*(t) = t + t^2/4$ der $\chi^2$-Funktion $f(u) = (u - 1)^2$ her, indem Sie das Supremum über $u \in \mathbb{R}$ bilden. Bestimmen Sie auch den optimalen Kritiker $T^\star = 2(p/q - 1)$. Wiederholen Sie die Rechnung auf dem Definitionsbereich $u \in (0, \infty)$ der Funktion und zeigen Sie, dass das Supremum gleich $t + t^2/4$ für $t \geq -2$ und gleich $-1$ für $t \leq -2$ ist.
    1. Verwenden Sie die Konjugierte auf $\mathbb{R}$. Zeigen Sie, dass für jeden Kritiker $T$ die Differenz zwischen $\chi^2(P\|Q)$ und der Schranke gleich $\tfrac{1}{4} \mathbb{E}_Q\big[(T - T^\star)^2\big]$ ist. Folgern Sie, dass ein skalierter Kritiker $T = c\,T^\star$ die untere Schranke $\chi^2(P\|Q)\,(2c - c^2)$ liefert. Das erklärt den im Experiment ausgegebenen Wert für $2T^\star$. Zeigen Sie, dass ein Aufstiegsschritt im Experiment $T(x) - T^\star(x)$ mit $1 - q(x)/2$ multipliziert. Welches Ergebnis unseres Beispielpaars trägt nach zehn Schritten am meisten zur Differenz bei und warum? Für welche Schrittweiten konvergiert der Gradientenaufstieg?
    1. Berechnen Sie die Konjugierte der Divergenzfunktion des Totalvariationsabstands $f(u) = \tfrac{1}{2}|u - 1|$ auf $u \in (0, \infty)$. Zeigen Sie, dass Kritiker mit Werten in $[-\tfrac{1}{2}, \tfrac{1}{2}]$ genügen, um die Schranke zu maximieren. Folgern Sie, dass $\textrm{TV}(P, Q) = \sup_g \big(\mathbb{E}_P[g] - \mathbb{E}_Q[g]\big)$ gilt, wobei das Supremum über alle Funktionen $g$ mit $|g| \leq \tfrac{1}{2}$ läuft. Ein solches Supremum untersuchen wir in :numref:`sec_mdl-ipm-mmd`. Welches $g$ erreicht es?
   :exlabel:`ex_mdl-divergences-distances-derive-convex-conjugate`

<!--en:4d617747-->
[Diskussionen](https://d2l.discourse.group/t/36835)

<!--en:6ca3bb72-->
f-Divergenzen und duale Variationsdarstellungen<br>**Dichteverhältnisse · Vorwärts- und Rückwärts-KL · Totalvariationsabstand · Pinsker-Ungleichung · Fenchel-Dualität · f-GAN**

<!--en:de78093a-->
::: {.slide title="Warum Verluste auf Basis von Dichteverhältnissen bei disjunkten Trägern versagen"}

<!--en:874e7af4-->
[Motivation: Punktmassen als Testfall]{.kicker}

<!--en:66714233-->
Wir vergleichen zwei Punktmassen $P = \delta_0$ und $Q = \delta_d$ im Abstand $d > 0$:

<!--en:1ff9c940-->
- **Konstante $f$-Divergenzen ($u = p/q$):** Für jedes $d > 0$ gelten $D_{\textrm{KL}} = \infty$, $\textrm{JS} = \log 2 \approx 0.693$ und $\textrm{TV} = 1.000$. Der **Gradient ist null** und zieht $Q$ daher nicht zu $P$.

<!--en:156175fd-->
- **Metriken, die die Geometrie berücksichtigen:** Die RBF-basierte $\textrm{MMD}^2 = 2(1 - e^{-d^2/(2\ell^2)})$ liefert lokale Gradienten, der Wasserstein-Abstand $W_1(\delta_0, \delta_d) = |d|$ bei jedem Abstand einen Gradienten vom Betrag eins, $\pm 1$.

<!--en:1afc5b3a-->
::: {.slide title="Punktmassen als Testfall im Code"}

<!--en:def310aa-->
[Fünf Vergleiche auf disjunkten Atomen]{.kicker}

<!--en:674cc3ca-->
Stellen wir $\delta_0$ und $\delta_d$ als Wahrscheinlichkeitsvektoren $(1, 0)$ und $(0, 1)$ auf den Atomen $\{0, d\}$ dar, so erhalten wir direkt die Tabelle:

<!--en:11386146-->
- Anhand der **$f$-Divergenzen**, die auf dem Dichteverhältnis beruhen, beantworten wir drei Fragen. (1) Wie unterscheiden sich Vorwärts- und Rückwärts-KL bei fehlspezifizierten Modellen? (2) Was garantieren Totalvariationsabstand und Pinsker-Ungleichung für statistische Tests? (3) Wie lässt sich jede Divergenz $D_f$ mithilfe konvexer Dualität und eines Kritikernetzes aus Stichproben schätzen?

<!--en:c537fa59-->
::: {.slide title="Divergenzen und echte Wahrscheinlichkeitsmetriken"}

<!--en:ae07c4b9-->
[Mathematische Grundlagen]{.kicker}

<!--en:5a717e83-->
- **Axiome einer Divergenz:** $D(P, Q) \ge 0$ und $D(P, Q) = 0 \iff P = Q$.

<!--en:63dc5846-->
- **Axiome einer Metrik:** Zusätzlich gelten Symmetrie $D(P, Q) = D(Q, P)$ und Dreiecksungleichung $D(P, R) \le D(P, Q) + D(Q, R)$ (erfüllt für TV, $d_{\mathrm{H}}$, $\sqrt{\textrm{JS}}$, MMD und $W_1, W_2$).

<!--en:6b2741b1-->
- **Warum KL keine Metrik ist:** KL ist asymmetrisch ($0.318 \neq 0.807$ für $\mathcal{N}(0,1)$ gegenüber $\mathcal{N}(0,4)$) und verhält sich wie ein *quadrierter* Abstand $\tfrac12(\mu_1 - \mu_2)^2$. Das verletzt die Dreiecksungleichung ($2 > \tfrac12 + \tfrac12$).

<!--en:52cdf097-->
::: {.slide title="Allgemeine Form und Nichtnegativität der f-Divergenz"}

<!--en:ff368488-->
[Divergenzen aus Dichteverhältnissen]{.kicker}

<!--en:de2eafc9-->
Schreiben wir die KL-Divergenz als $\mathbb{E}_{x \sim Q}[u(x)\log u(x)]$ mit dem Dichteverhältnis $u(x) = p(x)/q(x)$, erkennen wir die allgemeine Form. Wir ersetzen $u \log u$ durch eine beliebige konvexe Funktion $f : (0, \infty) \to \mathbb{R}$ mit $f(1) = 0$ (unter der Annahme $P \ll Q$):

<!--en:64cc3cae-->
- **Nichtnegativität in einer Zeile (Jensen-Ungleichung):** Das Dichteverhältnis hat den Erwartungswert $1$ unter $Q$ ($\mathbb{E}_Q[u] = \sum_x q(x)\frac{p(x)}{q(x)} = \sum_x p(x) = 1$). Aus der Konvexität von $f$ folgt direkt
  $$D_f(P \| Q) = \mathbb{E}_Q[f(u)] \ge f(\mathbb{E}_Q[u]) = f(1) = 0$$

<!--en:85d021bd-->
- **Strikte Positivität und affine Invarianz:** Ist $f$ bei $u = 1$ streng konvex ($f(u) > c(u-1)$ für alle $u \neq 1$), so gilt $D_f(P \| Q) = 0 \iff u(x) = 1$ überall ($P = Q$). Außerdem ändert die Addition von $c(u - 1)$ zu $f(u)$ keine Divergenz, da $\mathbb{E}_Q[c(u - 1)] = 0$.

<!--en:21402485-->
::: {.slide title="Gängige Divergenzfunktionen und ihre Eigenschaften"}

<!--en:27cec8d8-->
[Übersicht der f-Divergenzen]{.kicker}

<!--en:83882999-->
- **Asymmetrische Divergenzen:** Vorwärts-KL ($u\log u$, $f''(1)=1$), Rückwärts-KL ($-\log u$, $f''(1)=1$) und Pearson-$\chi^2$-Divergenz ($(u-1)^2$, $f''(1)=2$).

<!--en:359a5925-->
- **Symmetrische Divergenzen und Metriken:** quadrierter Hellinger-Abstand ($(\sqrt{u}-1)^2$, $f''(1)=1/2$), Jensen-Shannon-Divergenz ($f''(1)=1/4$, $\sqrt{\textrm{JS}} \le \sqrt{\log 2}$) und Totalvariationsabstand ($\tfrac12|u-1|$, V-förmiger Knick bei $u=1$).

<!--en:3f9b0b7f-->
::: {.slide title="Geometrie der Divergenzfunktionen in der Nähe von u = 1"}

<!--en:5bbf20c8-->
[Form der Straffunktionen im Vergleich]{.kicker}

<!--en:a6633afd-->
Verschieben wir jede glatte Divergenzfunktion um $f'(1)(u - 1)$, so berühren alle Kurven bei $u = 1$ die horizontale Achse. Lokal stimmen sie annähernd überein, an den Rändern unterscheiden sie sich:

<!--en:e991c21f-->
- **Nahe $u = 1$ (lokale Übereinstimmung):** Jede glatte Kurve gleicht einer Parabel $\tfrac12 f''(1)(u - 1)^2$. Der Totalvariationsabstand hat einen V-förmigen Knick.

<!--en:b0be3128-->
- **Für $u \to 0$ ($p \ll q$, das Modell legt Masse in leere Gebiete):** Die Rückwärts-KL ($-\log u$) wächst gegen $+\infty$ und bestraft so fälschlich vom Modell erzeugte Modi stark.

<!--en:e81cf494-->
- **Für $u \to \infty$ ($p \gg q$, das Modell verfehlt Daten):** Die Vorwärts-KL ($u \log u$) und die $\chi^2$-Divergenz ($(u-1)^2$) wachsen überlinear, TV und JS nur linear.

<!--en:6c3c1492-->
::: {.slide title="Randfälle: Warum D_f auf disjunkten Trägern konstant ist"}

<!--en:4b3708f9-->
[Singuläre Maße und disjunkte Träger]{.kicker}

<!--en:42e35a08-->
Was geschieht mit $D_f(P \| Q)$, wenn $P$ und $Q$ jeweils Gebieten Masse zuweisen, in denen die andere Verteilung verschwindet?

<!--en:a121db39-->
- **Grenzwerte bei $p = 0$ und $q = 0$:**
  - Für $p(x) = 0 < q(x)$ ($u = 0$) beträgt der Beitrag $q(x)\,f(0)$ mit $f(0) = \lim_{u \downarrow 0} f(u)$.
  - Für $q(x) = 0 < p(x)$ ($u \to \infty$) liefert der Grenzübergang $\lim_{q \downarrow 0} q\,f(p/q) = p \lim_{t \to \infty} f(t)/t$ den Beitrag $p(x)\,f'(\infty)$ mit $f'(\infty) = \lim_{t \to \infty} f(t)/t$.

<!--en:c13c8c08-->
- **Universelle Konstante für disjunkte Punktmassen ($d \neq 0$):** Bei $x = 0$ ($p=1, q=0$) und $x = d$ ($p=0, q=1$) ergibt sich
  $$D_f(\delta_0 \| \delta_d) = f(0) + f'(\infty) \quad \text{for every } d \neq 0$$
  - **Vorwärts-KL ($u\log u$):** $f(0) = 0$, $f'(\infty) = \infty$ $\implies$ $D_{\textrm{KL}}(\delta_0 \| \delta_d) = \infty$.
  - **Jensen-Shannon:** $f(0) = \tfrac12\log 2$, $f'(\infty) = \tfrac12\log 2$ $\implies$ $\textrm{JS}(\delta_0, \delta_d) = \log 2$.
  - **Totalvariationsabstand ($\tfrac12|u-1|$):** $f(0) = \tfrac12$, $f'(\infty) = \tfrac12$ $\implies$ $\textrm{TV}(\delta_0, \delta_d) = 1$.

<!--en:8bc73778-->
::: {.slide title="Sechs f-Divergenzen für unser Beispielpaar"}

<!--en:5dba7865-->
[Numerischer Vergleich]{.kicker}

<!--en:6d2b2ecb-->
Für $P = (0.6, 0.3, 0.1)$ und $Q = (0.2, 0.5, 0.3)$ überspannen die sechs Divergenzen eine Größenordnung ($0.091$ bis $1.013$). Nach Skalierung mit $2/f''(1)$ stimmen $2\,\textrm{KL} = 0.79$, $4\,d_{\mathrm{H}}^2 = 0.74$ und $8\,\textrm{JS} = 0.73$ jedoch annähernd überein:

<!--en:de0b39ff-->
::: {.slide title="Lokale quadratische Näherung: Übereinstimmung aller glatten Divergenzen D_f"}

<!--en:4e7b57e6-->
[Taylor-Entwicklung zweiter Ordnung]{.kicker}

<!--en:75673fae-->
Entwickeln wir $f(u)$ um $u = 1$ und bilden $\mathbb{E}_Q$, so verschwindet der Term erster Ordnung $f'(1)\mathbb{E}_Q[u - 1] = 0$:

<!--en:6eab6dd2-->
- **Universelle lokale Geometrie:** In der Nähe von $P = Q$ gilt $2\,\textrm{KL} \approx 4\,d_{\mathrm{H}}^2 \approx 8\,\textrm{JS} \approx \chi^2$.

<!--en:ced83017-->
- **Fisher-Informationsmetrik:** Beim Vergleich von $q_{\boldsymbol{\theta}+\mathbf{h}}$ mit $q_{\boldsymbol{\theta}}$ gilt $D_f(q_{\boldsymbol{\theta}+\mathbf{h}} \| q_{\boldsymbol{\theta}}) \approx \tfrac12 f''(1)\mathbf{h}^\top\mathbf{I}(\boldsymbol{\theta})\mathbf{h}$.

<!--en:e1a13ee1-->
::: {.slide title="Numerische Prüfung der lokalen quadratischen Näherung"}

<!--en:85e7b08c-->
[Asymptotische Übereinstimmung]{.kicker}

<!--en:ed728bed-->
Verschieben wir $P_t = Q + t(P - Q)$ in Richtung $Q$, so skaliert $r = 2t$. Sinkt $t$ um den Faktor zehn von $0.1$ auf $0.01$, so sinkt auch die relative Abweichung zwischen $D_f(P_t \| Q)$ und $\tfrac12 f''(1)\chi^2(P_t \| Q)$ um den Faktor zehn auf $< 1\%$:

<!--en:bfded807-->
::: {.slide title="Vorwärts- und Rückwärts-KL: Modi abdecken oder einen Modus suchen"}

<!--en:82c756dd-->
[Asymmetrie als Modellierungsentscheidung]{.kicker}

<!--en:b2b9ebe5-->
- **Vorwärts-KL $D_{\textrm{KL}}(P \| Q_{\boldsymbol{\theta}})$ (vermeidet Nullstellen, deckt Modi ab):** Wächst ins Unendliche, wenn $q_{\boldsymbol{\theta}}(x) \to 0$ an Datenpunkten gilt ($f'(\infty) = \infty$). Das Modell dehnt sich über alle Modi von $P$ aus (Maximum-Likelihood).

<!--en:3e98d833-->
- **Rückwärts-KL $D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \| P)$ (erzwingt Nullstellen, sucht einen Modus):** Wächst ins Unendliche, wenn $p(x) \to 0$ für Samples des Modells gilt ($f(0) = \infty$). Das Modell konzentriert sich auf einen einzigen Modus von $P$ zu Kosten von $-\log w_k$ (Variationsinferenz, RLHF).

<!--en:09a0577a-->
::: {.slide title="Anpassung einer Gaußverteilung an eine bimodale Mischung"}

<!--en:c42c4b1c-->
[Verhalten eines fehlspezifizierten Modells]{.kicker}

<!--en:30d8b99c-->
Wir passen $Q_{\boldsymbol{\theta}} = \mathcal{N}(\mu, \sigma^2)$ an die asymmetrische bimodale Mischung $P = 0.7\,\mathcal{N}(-2, 0.6^2) + 0.3\,\mathcal{N}(2, 0.6^2)$ an:

<!--en:3132f1ec-->
- **Exakte Anpassung mit Vorwärts-KL (Momentenabgleich):** Minimieren wir $D_{\textrm{KL}}(P \| \mathcal{N}(\mu, \sigma^2))$, stimmen Mittelwert und Varianz des Modells mit denen von $P$ überein:
  $$\mu^* = 0.7(-2) + 0.3(2) = -0.800, \qquad (\sigma^*)^2 = 0.6^2 + 0.7(0.3)(-2 - 2)^2 = 3.72 \implies \sigma^* = 1.929$$

<!--en:dd8619d7-->
- **Lokale Minima der Rückwärts-KL (Auslassen von Modi):** Stimmt $Q_{\boldsymbol{\theta}}$ mit einer einzelnen Komponente $\mathcal{N}(\mu_k, 0.6^2)$ zum Mischungsgewicht $w_k$ überein, so gilt im Bereich dieses Modus $p(x) \approx w_k q_{\boldsymbol{\theta}}(x)$. Daraus folgt
  $$D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \| P) \approx \mathbb{E}_{Q_{\boldsymbol{\theta}}}\!\left[\log\frac{q_{\boldsymbol{\theta}}(x)}{w_k q_{\boldsymbol{\theta}}(x)}\right] = \log\frac{1}{w_k}$$
  - Die Konzentration auf den Modus mit Gewicht $70\%$ ($\mu \approx -2, \sigma \approx 0.6$) kostet $\ln(1/0.7) \approx 0.357\text{ nats}$, die auf den Modus mit Gewicht $30\%$ ($\mu \approx +2$) kostet $\ln(1/0.3) \approx 1.204\text{ nats}$.

<!--en:72e32557-->
::: {.slide title="Anpassungen, die alle Modi abdecken oder einen Modus suchen"}

<!--en:d03a91fc-->
[Numerische Optimierung im Beispiel]{.kicker}

<!--en:cddb9623-->
Die Rückwärts-KL unterschätzt die Standardabweichung der Zielverteilung um den Faktor $>3\times$ ($\sigma = 0.603$ gegenüber $1.929$). Sie konvergiert zu dem Modus, der näher am Startpunkt liegt:

<!--en:f562d955-->
::: {.slide title="Totalvariationsabstand und optimale Tests mit einer Beobachtung"}

<!--en:5ea1228d-->
[Statistische Bedeutung des TV-Abstands]{.kicker}

<!--en:4a226d0e-->
Was garantiert ein Totalvariationsabstand von $\textrm{TV}(P, Q) = 0.4$ konkret?

<!--en:35fe36e8-->
- **Größter Unterschied der Ereigniswahrscheinlichkeiten:**
  $$\textrm{TV}(P, Q) = \frac{1}{2}\sum_x |p(x) - q(x)| = \sup_{A} |P(A) - Q(A)|$$

<!--en:affdb15d-->
- **Optimales trennendes Ereignis:** Das Supremum wird bei $A^\star = \{x : p(x) > q(x)\}$ erreicht, weil die Flächen zwischen dem Graphen von $p - q$ und der horizontalen Achse oberhalb und unterhalb der Achse gleich groß sind.

<!--en:83297bbe-->
- **Grenze für Hypothesentests:** Eine einzelne Beobachtung stammt mit gleicher Wahrscheinlichkeit aus $P$ oder $Q$. *Jeder* Detektor (GAN-Diskriminator oder Detektor für KI-Texte) entscheidet mit Wahrscheinlichkeit höchstens $\tfrac12(1 + \textrm{TV}(P, Q))$ richtig.

<!--en:b5378fe9-->
::: {.slide title="Pinsker-Ungleichung: Eine TV-Schranke aus der KL-Divergenz"}

<!--en:996426ba-->
[Beweis in zwei Schritten mit der Log-Summen-Ungleichung]{.kicker}

<!--en:7c094f5d-->
Die **Pinsker-Ungleichung** begrenzt die maximale Änderung einer Ereigniswahrscheinlichkeit mithilfe der Divergenz aus logarithmierten Wahrscheinlichkeiten:

<!--en:c70cf2da-->
- **Schritt 1 (Vergröberung auf $A^\star = \{p > q\}$):** Für $a = P(A^\star), b = Q(A^\star)$ ($a - b = \textrm{TV}$) ergibt die Datenverarbeitung $D_{\textrm{KL}}(P \| Q) \ge D_{\textrm{KL}}(\textrm{Ber}(a) \| \textrm{Ber}(b))$.

<!--en:a5d8f31b-->
- **Schritt 2 (Bernoulli-Krümmung $\ge 4$):** $\frac{d^2}{da^2}D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b)) = \frac{1}{a(1-a)} \ge 4 \implies D_{\textrm{KL}} \ge 2(a - b)^2$.

<!--en:398ee9f9-->
::: {.slide title="Schärfe und Grenzen der Pinsker-Ungleichung"}

<!--en:7213e337-->
[Numerische Prüfung mit der Monte-Carlo-Methode]{.kicker}

<!--en:45cf88e7-->
Für $10{,}000$ zufällige Verteilungspaare auf je 5 Ergebnissen hat $\textrm{TV}/\sqrt{\textrm{KL}/2}$ einen Median von etwa $0.78$ und übersteigt nie $1$. Nahezu faire Münzen ($\delta \to 0$) zeigen, dass sich die Konstante $\tfrac12$ nicht verbessern lässt:

<!--en:9f3e4b59-->
::: {.slide title="Die Datenverarbeitungsungleichung für f-Divergenzen"}

<!--en:92e120a0-->
[Monotonie unter der Wirkung eines Kanals]{.kicker}

<!--en:87468d20-->
Können Merkmalsextraktion, Quantisierung oder eine randomisierte Transformation zwei Verteilungen $P$ und $Q$ leichter unterscheidbar machen?

<!--en:f6bf90dc-->
- **Transformation durch einen Markov-Kanal:** Ein beliebiger Übergangskern $\kappa(y \mid x)$ mit der Abbildung $x \mapsto y$ erzeugt die Ausgabeverteilungen $(P\kappa)(y) = \sum_x \kappa(y \mid x)p(x)$ und $(Q\kappa)(y) = \sum_x \kappa(y \mid x)q(x)$.

<!--en:78178fc3-->
- **Allgemeiner Kontraktionssatz:** Wenden wir die Jensen-Ungleichung mit den A-posteriori-Gewichten $\frac{\kappa(y \mid x)q(x)}{(Q\kappa)(y)}$ zur jeweiligen Ausgabe $y$ an, so folgt für *jede* $f$-Divergenz:
  $$D_f(P\kappa \,\|\, Q\kappa) \le D_f(P \| Q)$$

<!--en:17a147dd-->
- **Zwei Konsequenzen für Deep Learning:**
  1. Jede an extrahierten Merkmalen $\phi(x)$ gemessene Divergenz ist eine *untere Schranke* für die Divergenz $D_f(P \| Q)$ im ursprünglichen Raum.
  2. Jede umkehrbare Transformation $\phi$ wirkt in beiden Richtungen als Kanal. Daher gilt $D_f(\phi_\# P \| \phi_\# Q) = D_f(P \| Q)$ (**Invarianz gegenüber der Skalierung der Koordinaten**).

<!--en:0f3c0d69-->
::: {.slide title="Legendre-Fenchel-Konjugierte und Tangentenschranken"}

<!--en:dd532c1c-->
[Grundlagen der konvexen Dualität]{.kicker}

<!--en:9bb70ea3-->
Wie schätzen wir $D_f(P \| Q)$, wenn wir nur Stichproben aus $P$ und $Q$ haben und $p(x)/q(x)$ nicht auswerten können?

<!--en:6e12250f-->
- **Konvex Konjugierte:** Jede konvexe Funktion $f(u)$ ist die obere Einhüllende ihrer Tangenten mit Steigung $t$ und Achsenabschnitt $-f^*(t)$:
  $$f^*(t) = \sup_{u}\big(ut - f(u)\big)$$

<!--en:0c78ae36-->
- **Fenchel-Young-Ungleichung:** Für jedes $u$ und jede Steigung $t$ gilt
  $$f(u) \ge ut - f^*(t)$$
  mit Gleichheit bei $t = f'(u)$.

<!--en:fc7e0846-->
- **Beispiele:** $f(u) = u\log u$ $\implies$ $f^*(t) = e^{t-1}$ und $f(u) = (u-1)^2$ $\implies$ $f^*(t) = t + t^2/4$.

<!--en:c82e92cb-->
::: {.slide title="Variationsschranke des Kritikers und Training eines f-GAN"}

<!--en:5f4a78ae-->
[Schätzung von Divergenzen aus Stichproben]{.kicker}

<!--en:550d7b1b-->
Wenden wir $f(p/q) \ge (p/q)T(x) - f^*(T(x))$ an und mitteln unter $Q$, so entfällt das unbekannte Dichteverhältnis:

<!--en:0bec12ef-->
- **Optimaler Kritiker:** Bei $T^\star(x) = f'(p(x)/q(x))$ gilt Gleichheit ($1 + \log(p/q)$ für KL, $2(p/q - 1)$ für $\chi^2$).

<!--en:3a60eb48-->
- **Ursprüngliches GAN:** Aus $T(x) = \log D(x)$ und $-f^*(T) = \log(1 - D(x))$ ergibt sich $2\,\textrm{JS}(P, Q) - \log 4$.

<!--en:4ed23151-->
::: {.slide title="Das ursprüngliche GAN als duale Darstellung der JS-Divergenz"}

<!--en:ff411223-->
[Diskriminator mit binärer Kreuzentropie]{.kicker}

<!--en:b4518337-->
Die ursprüngliche GAN-Zielfunktion von Goodfellow et al. (2014) ist ein Spezialfall der Kritikerschranke für das $f$-GAN:

<!--en:5022494d-->
- **1. Divergenzfunktion und Konjugierte wählen:** Wir wählen $f(u) = u\log u - (u+1)\log(u+1)$. Für $t < 0$ lautet die Konjugierte $f^*(t) = -\log(1 - e^t)$.

<!--en:5bcd9439-->
- **2. Kritiker über einen sigmoidalen Diskriminator $D(x) \in (0, 1)$ parametrisieren:** Setzen wir $T(x) = \log D(x)$, so wird $-f^*(T(x))$ zu $\log(1 - D(x))$. Daraus ergibt sich das Spiel mit binärer Kreuzentropie:
  $$\mathbb{E}_{x \sim P}[T(x)] - \mathbb{E}_{x \sim Q}[f^*(T(x))] = \mathbb{E}_{x \sim P}[\log D(x)] + \mathbb{E}_{x \sim Q}[\log(1 - D(x))]$$

<!--en:f5f91702-->
- **3. Optimaler Diskriminator und JS-Wert:** Setzen wir $T^\star(x) = f'(p/q) = \log\frac{p(x)}{p(x)+q(x)}$, so erhalten wir den Bayes-optimalen Diskriminator $D^\star(x) = \frac{p(x)}{p(x)+q(x)}$. Wegen $f(u) = 2f_{\textrm{JS}}(u) - (u-1)\log 2 - \log 4$ beträgt der Wert bei $D^\star$ genau $2\,\textrm{JS}(P, Q) - \log 4$.

<!--en:7a208edc-->
::: {.slide title="Praktische Grenzen adversarialer Kritikerschranken"}

<!--en:db4d022c-->
[Unterschätzung und verschwindende Gradienten]{.kicker}

<!--en:451e39b2-->
Trainieren wir einen $\chi^2$-Kritiker durch Gradientenaufstieg für $P = (0.6, 0.3, 0.1)$ und $Q = (0.2, 0.5, 0.3)$ ($\chi^2 = 1.0133$), so zeigen sich zwei strukturelle Grenzen der Variationsschranke:

<!--en:56b2d661-->
::: {.slide title="Praktische Grenzen adversarialer Kritikerschranken (Fortsetzung)"}

<!--en:db4d022c-2-->
[Unterschätzung und verschwindende Gradienten]{.kicker}

<!--en:a8ecc767-->
- **Verzerrung nach unten und Sättigung bei disjunkten Trägern:** Jeder suboptimale Kritiker ($T \neq T^\star$, auch $2T^\star$) unterschätzt $D_f$ um $\tfrac14\mathbb{E}_Q[(T - T^\star)^2]$. Bei disjunkten Trägern ist $T^\star$ auf jedem Träger konstant ($f'(\infty)$ auf Daten, $f'(0)$ auf Samples des Modells). Der Generator erhält den Gradienten null.

<!--en:8a0fca79-->
::: {.slide title="Verständnisfragen zu f-Divergenzen, Pinsker-Ungleichung und Kritikern"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:46195ee5-->
Führen Sie die folgenden drei Rechnungen aus:

<!--en:8607d451-->
1. **Rechnung von Hand:** Berechnen Sie für $P = (0.6, 0.3, 0.1)$ und $Q = (0.2, 0.5, 0.3)$ den Abstand $\textrm{TV}(P, Q)$ und die Pearson-Divergenz $\chi^2(P \| Q)$. Welches Ergebnis trägt am meisten zu $\chi^2(P \| Q)$ bei und warum?

<!--en:36c84f7d-->
2. **Grenze für die Erkennung von KI-Texten:** Für menschliche Texte $P$ und Texte eines LLM $Q$ gelte $D_{\textrm{KL}}(P \| Q) = 0.02\text{ nats}$. Welche Genauigkeit kann ein Detektor anhand eines einzelnen Textes bei einer Mischung im Verhältnis $50/50$ höchstens erreichen? Hilft es, zunächst Merkmale aus dem Text zu extrahieren?

<!--en:1e76f097-->
3. **Interpretation eines Kritikerwerts:** Ein mit der $\chi^2$-Konjugierten $f^*(t) = t + t^2/4$ trainierter Kritiker liefert $\mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)] = 0.70$ auf einem neuen, zurückgehaltenen Testdatensatz. Was folgt daraus für den wahren Wert von $\chi^2(P \| Q)$?

<!--en:557b98b6-->
**Antworten:**

<!--en:74040711-->
1. **TV und $\chi^2$:** $\textrm{TV} = \tfrac12(0.4 + 0.2 + 0.2) = 0.40$ (erreicht für $A^\star = \{1\}$). Es gilt $\chi^2 = \frac{0.4^2}{0.2} + \frac{0.2^2}{0.5} + \frac{0.2^2}{0.3} = 0.80 + 0.08 + 0.133 = 1.0133$. Ergebnis $1$ trägt $0.80$ bei, weil bei $\chi^2$ durch $q_1 = 0.2$ dividiert wird.

<!--en:28134c63-->
2. **Grenze für den Detektor:** Nach der Pinsker-Ungleichung gilt $\textrm{TV}(P, Q) \le \sqrt{0.02/2} = 0.10$. Die Genauigkeit beträgt also höchstens $\tfrac12(1 + \textrm{TV}) \le 55\%$. Wegen der Datenverarbeitungsungleichung kann keine Merkmalsextraktion $D_{\textrm{KL}}$ oder $\textrm{TV}$ erhöhen.

<!--en:6b89d838-->
3. **Kritikerschranke:** Auf neuen Stichproben gilt $\chi^2(P \| Q) \ge 0.70$ im Erwartungswert. Das liefert eine untere Schranke. Die Divergenz kann deutlich größer sein, wenn $T \neq T^\star$ gilt (wie nach Schritt 5 unseres Gradientenaufstiegs, wo der wahre Wert $1.0133$ betrug).

<!--en:9bc079c4-->
::: {.slide title="Zusammenfassung: f-Divergenzen und ihre dualen Darstellungen"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:ed3ddf33-->
| Begriff oder Verfahren | Mathematische Form | Eigenschaft oder Satz | Bedeutung für Deep Learning und Einschränkungen |

<!--en:55389cd3-->
| **$f$-Divergenz** | $D_f(P \| Q) = \mathbb{E}_Q[f(p/q)]$ | $D_f \ge 0$ nach der Jensen-Ungleichung, lokal $D_f \approx \tfrac12 f''(1)\chi^2$ | Umfasst KL, Rückwärts-KL, $\chi^2$, Hellinger, TV und JS. Auf disjunkten Trägern konstant bei $f(0)+f'(\infty)$ |

<!--en:270c333f-->
| **Vorwärts- und Rückwärts-KL** | $\mathbb{E}_P[\log(p/q)]$ und $\mathbb{E}_Q[\log(q/p)]$ | $f'(\infty) = \infty$ (vorwärts) und $f(0) = \infty$ (rückwärts) | Vorwärts-KL (MLE) deckt alle Modi ab. Rückwärts-KL (VI) konzentriert sich auf einen Modus und unterschätzt die Varianz |

<!--en:ebb109ef-->
| **Totalvariationsabstand (TV)** | $\tfrac12\|p-q\|_1 = \sup_A |P(A)-Q(A)|$ | Genauigkeit des optimalen Tests mit 1 Beobachtung: $\tfrac12(1 + \textrm{TV})$ | Echte Metrik mit Werten in $[0, 1]$, erreicht auf disjunkten Trägern $1$ |

<!--en:0cdbbbb2-->
| **Pinsker-Ungleichung** | $\textrm{TV}(P, Q) \le \sqrt{\tfrac12 D_{\textrm{KL}}(P \| Q)}$ | Scharfe Konstante $1/2$ für $P \to Q$, keine Umkehrung | Liefert aus KL-Trust-Regionen (PPO) Schranken für die maximale Änderung von Ereigniswahrscheinlichkeiten |

<!--en:ab02c584-->
| **Datenverarbeitung** | $D_f(P\kappa \| Q\kappa) \le D_f(P \| Q)$ | Kontraktion unter jedem Kanal $\kappa(y \mid x)$ | Divergenzen zwischen Merkmalen liefern untere Schranken für Divergenzen zwischen Eingaben. Skaleninvarianz |

<!--en:f32d6c05-->
| **Fenchel-Kritikerschranke** | $D_f(P \| Q) \ge \mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)]$ | Scharf bei $T^\star = f'(p/q)$, für das GAN gilt $2\,\textrm{JS} - \log 4$ | Ermöglicht adversariales Training allein mit Stichproben ($f$-GAN), unterschätzt aber die Divergenz bei suboptimalem $T$ |
