<!--en:ebe0b943-->
# Optimierungslandschaften und Gradientenabstieg

<!--en:c74a46d7-->
Die gängigen Optimierer im Deep Learning unterscheiden sich im Momentum, in der adaptiven Vorkonditionierung (engl. preconditioning) oder im Lernratenplan. Dennoch wiederholen alle denselben Grundschritt: Sie berechnen den Gradienten des Verlusts und gehen einen Schritt in die entgegengesetzte Richtung. Warum kommen sie damit bei manchen Zielfunktionen rasch voran, bei anderen aber nur schleppend? Und warum divergiert das Verfahren plötzlich, sobald die Lernrate eine bestimmte Schwelle überschreitet?

<!--en:8d6f7d34-->
Ein Netz mit $d$ Parametern definiert ein empirisches Risiko $f: \mathbb{R}^d \to \mathbb{R}$ als Funktion eines $d$-dimensionalen Parametervektors $\mathbf{x}$. Für die Generalisierung ist letztlich das erwartete Risiko auf bisher ungesehenen Daten entscheidend (:numref:`subsec_empirical-risk-and-risk`). Ein Optimierungsalgorithmus soll zunächst den empirischen Zielfunktionswert $f(\mathbf{x})$ möglichst schnell und zuverlässig senken. Klassische Darstellungen der nichtkonvexen Optimierung betonen oft die Gefahr, in ungünstigen lokalen Minima stecken zu bleiben. In hochdimensionalen neuronalen Netzen bestimmen jedoch zwei andere geometrische Eigenschaften die Trainingsdynamik: **schlechte Konditionierung** (engl. ill-conditioning), also große Unterschiede in der Krümmung je nach Richtung, und **Sattelpunkte** (engl. saddle points).

<!--en:134ffc29-->
Wir untersuchen zunächst den Gradientenabstieg (engl. gradient descent) in einem schlecht konditionierten zweidimensionalen quadratischen Tal. Daran leiten wir den genauen Zusammenhang zwischen Krümmung, maximaler stabiler Lernrate und Iterationszahl her. Anschließend betrachten wir kritische Punkte, Sattelpunkte und Plateaus in hochdimensionalen Verlustlandschaften und beweisen die allgemeine Abstiegsgarantie für glatte Funktionen. Wir zeigen, wie die Lernrate lokale Minima nach ihrer Schärfe (engl. sharpness) auswählt und wie das Training neuronaler Netze dadurch an den *Rand der Stabilität* (engl. edge of stability) gelangt. Schließlich lernen wir die *Vorkonditionierung* als Mittel gegen schlechte Konditionierung kennen.

<!--en:c523e94a-->
## Optimierungslandschaften und Konditionszahl

<!--en:85c9be8f-->
Um den Einfluss unterschiedlich starker Krümmung für sich zu betrachten, untersuchen wir die zweidimensionale quadratische Zielfunktion

<!--en:83fb3963-->
mit ihrem einzigen Minimum im Ursprung. Die zweite Ableitung (Krümmung) beträgt $\lambda_1 = 0.2$ entlang der flachen Achse $x_1$ und $\lambda_2 = 4$ entlang der steilen Achse $x_2$. Dadurch entstehen lang gestreckte elliptische Niveaulinien, die ein schmales Tal bilden.

<!--en:2efd081f-->
Der Gradientenabstieg mit Lernrate $\eta > 0$ aktualisiert die Iterierte gemäß $\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$. Da $\nabla f(\mathbf{x}) = (0.2 x_1, 4 x_2)$ gilt, sind die beiden Koordinaten entkoppelt. Bei jedem Schritt ändert sich jede um einen festen Faktor:

<!--en:4f80d4cf-->
Die beiden Koordinaten stellen konkurrierende Anforderungen an $\eta$:

<!--en:d505b673-->
* **Stabilitätsgrenze (steile Richtung):** Der Betrag der steilen Koordinate $x_2$ schrumpft genau dann, wenn $|1 - 4\eta| < 1$ gilt. Daraus folgt die strikte Obergrenze $\eta < 0.5$.

<!--en:140eeafb-->
* **Konvergenzgeschwindigkeit (flache Richtung):** Für jede stabile Schrittweite $\eta < 0.5$ erfüllt der Kontraktionsfaktor in der flachen Richtung die Ungleichung $1 - 0.2\eta > 0.9$. Jeder Schritt baut also weniger als $10\%$ des verbleibenden Fehlers entlang $x_1$ ab.

<!--en:949b3a7e-->
Wir speichern die Zielfunktion des Tals `f_valley`, ihren Gradienten `valley_grad`, die Funktion `train_2d` zur Berechnung zweidimensionaler Trajektorien ab $(-5, -2)$ und den einfachen Gradientenabstiegsschritt `gd_2d` in `d2l`. So können wir in den folgenden Abschnitten alle Optimierer auf genau derselben Fläche vergleichen.

<!--en:d2765e7d-->
Bei $\eta = 0.45$, knapp unter der Stabilitätsgrenze, beträgt der Faktor in der steilen Richtung $1 - 4(0.45) = -0.8$. Daher schießt $x_2$ bei jedem Schritt über die Talsohle hinaus und wechselt das Vorzeichen. Dabei sinkt der Betrag um $20\%$. In der flachen Richtung beträgt der Faktor dagegen $1 - 0.2(0.45) = 0.91$. Nach $30$ Schritten vom Startpunkt $(-5, -2)$ liegt die flache Koordinate deshalb noch bei etwa $-5 \times 0.91^{30} \approx -0.3$.

<!--en:dd2480cb-->
Wir variieren $\eta$ im Intervall $(0, 0.5]$ und zählen die Schritte, bis der Abstand beider Koordinaten vom Ursprung auf höchstens $1\%$ ihres jeweiligen Anfangsabstands gesunken ist. So sehen wir, wie die Konvergenzzeit vom Zusammenspiel der beiden Faktoren abhängt:

<!--en:03ac74cf-->
Bei der optimalen Schrittweite haben die Faktoren in der flachen und der steilen Richtung denselben Betrag: $1 - 0.2\eta = -(1 - 4\eta)$. Daraus folgt $\eta^\star = 2 / 4.2 \approx 0.476$. Beide Faktoren haben dann den Betrag $0.905$, und bis zur Konvergenz auf $1\%$ dauert es $47$ Schritte. Auffällig ist die Asymmetrie nahe der Stabilitätsschwelle: Gehen wir von $\eta^\star$ etwas herunter auf $\eta = 0.45$, brauchen wir nur zwei Schritte mehr ($49$). Gehen wir dagegen etwas hinauf auf $\eta = 0.49$, steigt die Schrittzahl auf mehr als das Doppelte ($113$).

<!--en:33000aa0-->
Diese Analyse gilt ebenso für jede $d$-dimensionale positiv definite quadratische Funktion $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x}$ mit symmetrischer Hesse-Matrix $\mathbf{H} \succ \mathbf{0}$. Wir entwickeln $\mathbf{x} = \sum_{i=1}^d c_i \mathbf{v}_i$ in der orthonormalen Eigenbasis von $\mathbf{H}$. Dabei gelten $\mathbf{H}\mathbf{v}_i = \lambda_i \mathbf{v}_i$ und $0 < \lambda_{\min} = \lambda_1 \le \cdots \le \lambda_d = \lambda_{\max}$. In dieser Basis zerfällt der Gradientenabstiegsschritt in unabhängige Schritte entlang der einzelnen Eigenvektoren:

<!--en:0f957214-->
Für Stabilität muss $|1 - \eta\lambda_i| < 1$ für alle $i$ gelten. Die Grenze bestimmt also der größte Eigenwert:

<!--en:15c7183b-->
Unter dieser Beschränkung hängt der Fortschritt entlang des Eigenvektors $\mathbf{v}_1$ in der flachsten Richtung allein von der **Konditionszahl** (engl. condition number) ab:

<!--en:521a64a5-->
Multiplizieren wir die Zielfunktion $f$ mit einer Konstanten $c > 0$, so ändern sich alle Eigenwerte um den Faktor $c$. Teilen wir die stabile Lernrate durch $c$, bleibt die Trajektorie gleich. Das dimensionslose Verhältnis $\kappa$ erfasst also, wie schwierig die Optimierung aufgrund der Geometrie der Fläche ist. Setzen wir die extremen Kontraktionsfaktoren dem Betrag nach gleich, gilt $1 - \eta\lambda_{\min} = -(1 - \eta\lambda_{\max})$. Wir erhalten die optimale konstante Lernrate $\eta^\star = 2 / (\lambda_{\min} + \lambda_{\max})$ und für den ungünstigsten Fall den Kontraktionsfaktor $(\kappa - 1)/(\kappa + 1) \approx 1 - 2/\kappa$ pro Schritt (:numref:`subsec_mdl-quadratic-model`). Um den Fehler auf das $\epsilon$-Fache zu senken, benötigen wir also

<!--en:ee1ae004-->
Schritte. Für unser Tal ($\kappa = 4 / 0.2 = 20$, $\epsilon = 0.01$) sagt die Formel $10 \log 100 \approx 46$ Schritte voraus, was gut zu den oben beobachteten $47$ Schritten passt. Da $t$ linear mit $\kappa$ wächst, lässt sich die schlechte Konditionierung mit keiner skalaren Lernrate überwinden. Für eine schnellere Konvergenz müssen wir die Richtung des Schritts selbst ändern.

<!--en:7515d192-->
## Kritische Punkte, Sattelpunkte und Plateaus

<!--en:47e4b056-->
Anders als eine konvexe quadratische Schale ist die Verlustlandschaft eines neuronalen Netzes nichtkonvex. Sie besitzt viele **kritische Punkte**, an denen $\nabla f(\mathbf{x}) = \mathbf{0}$ gilt, und **Plateaus**, auf denen $\|\nabla f(\mathbf{x})\|$ verschwindend klein ist. An jedem kritischen Punkt einer zweimal differenzierbaren Funktion $f: \mathbb{R}^d \to \mathbb{R}$ lässt sich die lokale Geometrie anhand der Eigenwerte der Hesse-Matrix $\mathbf{H} = \nabla^2 f(\mathbf{x})$ einordnen:

<!--en:5c047447-->
* **Lokales Minimum:** Alle $d$ Eigenwerte sind strikt positiv ($\mathbf{H} \succ \mathbf{0}$).

<!--en:10bfbbf9-->
* **Lokales Maximum:** Alle $d$ Eigenwerte sind strikt negativ ($\mathbf{H} \prec \mathbf{0}$).

<!--en:c852b5d9-->
* **Sattelpunkt:** $\mathbf{H}$ hat sowohl positive als auch negative Eigenwerte. Entlang mancher Eigenvektoren krümmt sich die Fläche nach oben, entlang anderer nach unten. Ein Beispiel ist $f(x, y) = x^2 - y^2$ im Ursprung (:numref:`fig_mdl-opt-saddle`).

<!--en:4f246c5d-->
* **Entarteter kritischer Punkt:** $\mathbf{H}$ hat Eigenwerte null, und alle von null verschiedenen Eigenwerte haben dasselbe Vorzeichen. Ableitungen höherer Ordnung entscheiden daher, ob ein Extremum (wie bei $x^4$) oder ein Wendepunkt (wie bei $x^3$) vorliegt.

<!--en:7b36a4b4-->
![Die Sattelfläche $z=x^2-y^2$ hat entlang $x$ ein Minimum und entlang $y$ ein Maximum (links). Das zeigen die beiden Schnitte durch den Ursprung (rechts).](../img/mdl-opt-saddle.svg)

<!--en:a7df9d17-->
In hohen Dimensionen gibt es weitaus mehr Sattelpunkte als lokale Minima. Für ein lokales Minimum müssen an einem nichtentarteten kritischen Punkt alle $d$ Hesse-Eigenwerte positiv sein. Nehmen wir heuristisch unabhängige, gleich wahrscheinliche Vorzeichen an, sind alle $d$ Eigenwerte mit Wahrscheinlichkeit $2^{-d}$ positiv. Die Theorie der Zufallsmatrizen bestätigt, dass sich die Eigenwerte symmetrischer Zufallsmatrizen gleichmäßig auf beide Seiten der Null verteilen, solange das Verlustniveau nicht sehr niedrig ist (siehe :exref:`ex_optimization-intro-eigenvalues-symmetric-random-matrices`). Messungen der Hesse-Matrix an neuronalen Netzen bestätigen, dass kritische Punkte bei hohem Verlust fast ausschließlich Sattelpunkte mit vielen negativen Eigenwerten sind. Erst bei niedrigen Verlustwerten sinkt der Anteil negativer Eigenwerte gegen null :cite:`Dauphin.Pascanu.Gulcehre.ea.2014`.

<!--en:7a9b13c5-->
Entscheidend ist, dass der Gradientenabstieg an **strikten Sattelpunkten** nicht stecken bleibt. Das sind kritische Punkte mit mindestens einem negativen Eigenwert $\lambda < 0$. Entlang eines Eigenvektors mit negativer Krümmung $-\mu < 0$ vergrößert jeder Gradientenschritt die Auslenkung um den Faktor $(1 + \eta\mu) > 1$. Die stabile Mannigfaltigkeit aller Startpunkte, von denen aus das Verfahren zu einem strikten Sattelpunkt konvergiert, hat Maß null. Bei Zufallsinitialisierung entkommt der Gradientenabstieg daher mit Wahrscheinlichkeit eins :cite:`Lee.Simchowitz.Jordan.ea.2016`. Vom Anfangsabstand $\delta$ zum Grat bis zum Abstand $1$ in der instabilen Richtung dauert es $\log(1/\delta)/\log(1 + \eta\mu) \approx \log(1/\delta)/(\eta\mu)$ Schritte. Die Schrittzahl hängt also logarithmisch von $\delta$ ab und ist umgekehrt proportional zum Betrag der negativen Krümmung $\mu$. Das Rauschen des Minibatch-Gradienten stößt die Iterierten sogar dann vom Grat, wenn $\delta = 0$ gilt (siehe :exref:`ex_optimization-intro-saddle-escape-time`).

<!--en:663c7e0f-->
![Bei der Minimierung von $f(x)=\tanh x$ ab $x=4$ kommt der Gradientenabstieg auf einem Aktivierungsplateau kaum voran. Dort gilt $f'(4) \approx 0.0013$, obwohl es keinen kritischen Punkt gibt.](../img/mdl-opt-tanh-flat.svg)

<!--en:88102d7b-->
**Plateaus** können dagegen ganz ohne kritischen Punkt entstehen, wenn Aktivierungen sättigen und ihre Ableitungen klein werden. Ein Beispiel ist $f(x) = \tanh(x)$ bei $x = 4$ (:numref:`fig_mdl-opt-tanh-flat`). Dort gilt $f'(4) = 1 - \tanh^2(4) \approx 0.0013$, sodass der Gradientenabstieg mit $\eta = 1$ ganze $370$ Schritte für den Weg von $x = 4$ nach $x = 2$ braucht. In tiefen Netzen verhindert vor allem die Architektur solche verschwindenden Gradienten (:numref:`sec_numerical_stability`): Aktivierungen ohne Sättigung (:numref:`subsec_activation-functions`), Normalisierungsschichten (:numref:`sec_batch_norm`) und Residualverbindungen (:numref:`sec_resnet`). Damit bleibt die schlechte Konditionierung die zentrale Herausforderung für den Optimierer.

<!--en:756fffb4-->
Die Verlustfunktion eines neuronalen Netzes ist global nichtkonvex. Vertauschen wir etwa die $d$ versteckten Einheiten einer Schicht, erhalten wir $d!$ verschiedene Parametervektoren mit gleichem Verlust. Ihr Mittelwert entspricht einem Netz mit nur einer versteckten Einheit und höherem Verlust (siehe :exref:`ex_optimization-intro-permutation-symmetric-minima`). Die lokale Geometrie um jede Iterierte bestimmt hingegen das Spektrum der Hesse-Matrix. In empirisch gemessenen Spektren tiefer Netze liegen sehr viele Eigenwerte nahe null, daneben gibt es wenige große positive Ausreißer :cite:`Ghorbani.Krishnan.Xiao.2019`. Der größte Eigenwert, die **Schärfe** $\lambda_{\max}$, begrenzt die stabile Lernrate. In den vielen flachen Richtungen kommt der Gradientenabstieg nur mit der Rate $\eta\lambda_i \ll 1$ voran. Damit tritt der Engpass des schmalen Tals auch in hohen Dimensionen auf.

<!--en:9c9f1231-->
## Abstiegsgarantie und Stabilität von Minima

<!--en:0e58b42f-->
Bei einer nichtquadratischen Zielfunktion ändert sich die Krümmung von Punkt zu Punkt. Warum bestimmt die Regel $2/\lambda_{\max}$ dennoch, welche Schrittweiten stabil sind?

<!--en:70f552e9-->
Aus der Taylor-Entwicklung erster Ordnung $f(\mathbf{x} + \boldsymbol{\epsilon}) \approx f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x})$ und der Cauchy-Schwarz-Ungleichung folgt, dass der negative Gradient $-\nabla f(\mathbf{x})$ in die Richtung des lokal steilsten Abstiegs zeigt. Damit ein endlicher Schritt zum Punkt $\mathbf{x} - \eta \nabla f(\mathbf{x})$ den Wert von $f$ tatsächlich senkt, nehmen wir an, dass $f$ **$L$-glatt** ist. Der Gradient ist also Lipschitz-stetig mit Konstante $L$ (in einer Variablen gilt $|f'(y) - f'(x)| \le L|y - x|$ beziehungsweise $|f''(x)| \le L$). Integrieren wir die Steigungsdifferenz längs eines Schritts $\epsilon$, so erhalten wir für $f$ eine obere Schranke: eine im Ausgangspunkt tangential anliegende Parabel mit Krümmung $L$.

<!--en:c89c9e7f-->
Setzen wir den Gradientenschritt $\epsilon = -\eta f'(x)$ ein (oder $\boldsymbol{\epsilon} = -\eta \nabla f(\mathbf{x})$ in $d$ Dimensionen mit $\|\nabla^2 f(\mathbf{x})\|_2 \le L$, siehe :numref:`subsec_mdl-gd-smoothness`), erhalten wir das **Abstiegslemma** (engl. descent lemma):

<!--en:19f11827-->
Für $0 < \eta < 2/L$ senkt jeder Gradientenschritt den Wert von $f$ an jedem nichtkritischen Punkt strikt. Die größte garantierte Abnahme pro Schritt beträgt $\|\nabla f(\mathbf{x})\|^2 / (2L)$ und ergibt sich bei $\eta = 1/L$.

<!--en:7e01987e-->
Kehren wir nun die Betrachtung um. Wir halten die Lernrate $\eta$ auf einer Verlustlandschaft mit mehreren lokalen Minima unterschiedlicher Krümmung fest. Nahe einem lokalen Minimum $x^\star$ mit Krümmung $\lambda = f''(x^\star) > 0$ multipliziert ein Gradientenschritt eine kleine Auslenkung $x - x^\star$ in linearer Näherung mit $1 - \eta\lambda$. Daraus folgt:

<!--en:71382c94-->
* Für $\lambda < 2/\eta$ gilt $|1 - \eta\lambda| < 1$. Das Minimum ist **linear stabil** und zieht benachbarte Iterierte an.

<!--en:1720bba9-->
* Für $\lambda > 2/\eta$ gilt $1 - \eta\lambda < -1$. Das Minimum ist **instabil** und stößt jede Iterierte ab, die nicht genau auf $x^\star$ trifft.

<!--en:c2b5109c-->
Die Lernrate $\eta$ wählt lokale Minima also nach ihrer **Schärfe** aus, unabhängig von ihrer Tiefe. Der Gradientenabstieg kann nur zu Minima konvergieren, deren Schärfe die Bedingung $\lambda_{\max} \le 2/\eta$ erfüllt. Diese Auswahl können wir direkt an $f(x) = x\cos(0.15\pi x)$ beobachten. Die Funktion hat ein tiefes, schärferes Minimum nahe $x = 7.27$ ($f'' \approx 1.81$) und ein weniger tiefes, flacheres Minimum nahe $x = -1.83$ ($f'' \approx 0.98$). Die Werte stehen in :numref:`tab_gd-minima-factors`.

<!--en:83c657aa-->
:Lokale Krümmung $f''$ und Faktor $1 - \eta f''$, mit dem jeder Schritt die Auslenkung an den beiden Minima von $f(x) = x \cos(0.15\pi x)$ multipliziert.

<!--en:ead7e82f-->
| Minimum | Krümmung $f''$ | Faktor bei $\eta = 0.5$ | Faktor bei $\eta = 2$ |

<!--en:3b521c87-->
| Tief ($x = 7.27$) | $1.81$ | $0.09$ (stabil) | $-2.62$ (instabil) |

<!--en:2bedc04a-->
| Weniger tief ($x = -1.83$) | $0.98$ | $0.51$ (stabil) | $-0.96$ (stabil) |

<!--en:e8267423-->
Zunächst definieren wir `gd` für den eindimensionalen Gradientenabstieg und `show_trace` zum Zeichnen der Trajektorie:

<!--en:23790a13-->
Als Nächstes definieren wir $f(x) = x\cos(0.15\pi x)$ sowie die Funktionen `f_cos_grad` für die erste und `f_cos_hess` für die zweite Ableitung:

<!--en:daa4da53-->
Wir werten `f_cos_hess` an den beiden lokalen Minima aus und starten `gd` bei $x = 10$ mit $\eta \in \{0.5, 2\}$. So sehen wir, wie die Stabilitätsgrenze $2/\eta$ zwischen dem tiefen, scharfen Minimum und dem weniger tiefen, flachen Minimum auswählt:

<!--en:65595e50-->
Vom Startpunkt $x = 10$ aus konvergiert das Verfahren mit $\eta = 0.5$ ($2/\eta = 4 > 1.81$) rasch zum tiefen Minimum bei $x = 7.269$. Bei $\eta = 2$ ($2/\eta = 1 < 1.81$) ist das tiefe Minimum instabil ($1 - \eta f'' = -2.62$). Die Iterierte nähert sich stattdessen dem flacheren Minimum nahe $x = -1.83$, wo $f'' = 0.98 < 2/\eta$ gilt.

<!--en:bcc1ee4c-->
## Am Rand der Stabilität

<!--en:6f3a90a7-->
Beim Training neuronaler Netze verändert sich das Verhältnis zwischen der Schärfe $\lambda_{\max}$ und der Stabilitätsgrenze $2/\eta$ im Verlauf des Trainings. :citet:`Cohen.Kaur.Li.ea.2021` zeigten, dass beim Gradientenabstieg auf dem gesamten Trainingssatz die Schärfe $\lambda_{\max}(\nabla^2 f(\mathbf{x}_t))$ der Hesse-Matrix mit sinkendem Verlust stetig steigt (**fortschreitende Zunahme der Schärfe**, engl. progressive sharpening). Sobald sie die Stabilitätsschwelle $2/\eta$ erreicht, pendelt sie sich für den Rest des Trainings in der Nähe von $2/\eta$ ein. Das Verfahren divergiert dabei nicht: Der Verlust sinkt weiter, wenn auch nicht monoton. Man bezeichnet diesen Bereich als **Rand der Stabilität**.

<!--en:98db8fe3-->
Wir können dieses Phänomen direkt an einem Netz mit 25 Parametern, einer versteckten Schicht und $\tanh$-Aktivierung messen. Wir passen das Netz an $16$ Punkte von $y = \sin(3x)$ auf dem Intervall $[-2, 2]$ an. Wegen $d = 25$ können wir die exakte $25 \times 25$-Hesse-Matrix und ihren größten Eigenwert $\lambda_{\max}$ alle $500$ Schritte berechnen. Das tun wir für $10$ Zufallsinitialisierungen mit den Lernraten $\eta = 0.4$ ($2/\eta = 5$) und $\eta = 0.25$ ($2/\eta = 8$).

<!--en:0dbea375-->
Wir starten `gd_sharpness` jeweils von denselben zehn Initialisierungen `P0` mit $\eta = 0.4$ ($2/\eta = 5$) und $\eta = 0.25$ ($2/\eta = 8$). Die Tabelle zeigt die Schärfe zu Beginn und am Ende, den Endverlust und den Anteil der Schritte gegen Ende des Trainings, bei denen der Verlust stieg:

<!--en:476ad81c-->
Wir tragen die auf die Stabilitätsschwelle normierte Schärfe $\lambda_{\max} / (2/\eta)$ über alle $20{,}000$ Schritte auf. Bei beiden Lernraten steigt die Schärfe bis in die Nähe des Stabilitätsrands beim normierten Wert $1$:

<!--en:3e5078ba-->
Von denselben Initialisierungen aus steigt die normierte Schärfe $\lambda_{\max} / (2/\eta)$ auf $1$ und bleibt auf diesem Niveau. Bei $\eta = 0.4$ pendelt sich die Schärfe bei $\lambda_{\max} \approx 5.0$ ein. Bei $\eta = 0.25$ erreichen die meisten Läufe $\lambda_{\max} \approx 8.0$. Es gibt Ausnahmen: Bei $\eta = 0.25$ passen manche Läufe die Daten bereits gut an, bevor die Schärfe $8$ erreicht. Sie bleiben knapp unter der Schwelle, und der Verlust sinkt fast bei jedem Schritt. Bei den Läufen an der Schwelle $2/\eta$ steigt der Verlust typischerweise bei etwa $50\%$ der Schritte. Die Iterierte schwingt entlang des Eigenvektors mit der größten Krümmung ($1 - \eta\lambda_{\max} \approx -1$), während sie in den übrigen $24$ flacheren Richtungen weiter absteigt. Sobald $\lambda_{\max}$ kurzzeitig über $2/\eta$ liegt, wächst die Schwingung in der Richtung größter Krümmung. Im Zusammenspiel mit den dritten Ableitungen des Verlusts entsteht eine rückstellende Drift in flachere Bereiche. Dadurch stabilisiert sich $\lambda_{\max}$ bei $2/\eta$ :cite:`Damian.Nichani.Lee.2023`. Diese Selbststabilisierung erklärt auch die Wirkung einer *Aufwärmphase für die Lernrate*: Erhöhen wir $\eta$ zu Beginn des Trainings allmählich (:numref:`sec_scheduler`), gelangt das Netz in einen Bereich, der flach genug für die maximale Lernrate ist :cite:`Kalra.Barkeshli.2024`.

<!--en:d08779fa-->
In diesem Experiment haben wir den Gradientenabstieg auf dem gesamten Trainingssatz untersucht. Im weiteren Verlauf des Kapitels trainieren wir mit Momentum, Adam und Minibatches. Der Rand der Stabilität tritt in allen drei Fällen auf, jeweils mit einer anderen Schwelle. Heavy-Ball-Momentum mit dem Koeffizienten $\beta$ (:numref:`sec_momentum`) hebt die Stabilitätsgrenze von $\eta\lambda_{\max} < 2$ auf $\eta\lambda_{\max} < 2(1 + \beta)$ an. Die Schärfe pendelt sich entsprechend bei $2(1 + \beta)/\eta$ ein :cite:`Cohen.Kaur.Li.ea.2021`. Adam (:numref:`sec_adam`) teilt jede Koordinate des Schritts durch die für sie geschätzte Gradientenskala. Hier stabilisiert sich die Schärfe der entsprechend skalierten Hesse-Matrix, die *vorkonditionierte Schärfe*. Mit dem für Adam üblichen Momentum-Koeffizienten $\beta_1 = 0.9$ liegt ihre Schwelle bei etwa $38/\eta$ :cite:`Cohen.Ghorbani.Krishnan.ea.2022`. Bei Minibatch-Gradienten endet der Anstieg der Schärfe bereits unterhalb von $2/\eta$, und zwar umso früher, je kleiner der Batch ist. In allen Fällen bedeutet eine höhere Lernrate eine niedrigere Obergrenze für die Schärfe. Die Erklärung für die Wirkung der Aufwärmphase gilt also auch für die Optimierer der folgenden Abschnitte.

<!--en:e8d4eee5-->
## Newton-Verfahren und Vorkonditionierung

<!--en:ec1d3e77-->
### Schritte zweiter Ordnung und Newton-Verfahren

<!--en:f517e9d9-->
Warum kommt der Gradientenabstieg im Tal :eqref:`eq_opt-intro-valley` nur mühsam voran? Am Punkt $(-5, -2)$ zeigt der negative Gradient $(1, 8)$ fast quer zur steilen Talwand und damit kaum in Richtung des Ursprungs $(0, 0)$. Für die steile Koordinate $x_2$ führt nur die Lernrate $\eta = 1/4 = 0.25$ in einem Schritt auf die Talsohle. Die flache Koordinate $x_1$ erreicht ihr Ziel in einem Schritt erst bei $\eta = 1/0.2 = 5$.

<!--en:1e595d30-->
Das **Newton-Verfahren** (engl. Newton's method) skaliert jede Eigenrichtung mit dem Kehrwert ihrer lokalen Krümmung. Dazu minimiert es die Taylor-Näherung zweiter Ordnung um $\mathbf{x}$:

<!--en:d6c8fd3b-->
Setzen wir den Gradienten nach $\boldsymbol{\epsilon}$ gleich null, erhalten wir den Newton-Schritt:

<!--en:cd95b0ad-->
Für jede positiv definite quadratische Funktion ist die Entwicklung zweiter Ordnung exakt. Ein einziger Newton-Schritt erreicht das Minimum unabhängig von der Konditionszahl $\kappa$. Nahe einem regulären lokalen Minimum einer allgemeinen glatten Funktion erreicht das Newton-Verfahren *quadratische Konvergenz*: Mit jedem Schritt verdoppelt sich die Anzahl der korrekten Stellen (:numref:`sec_mdl-curvature-methods`).

<!--en:3d9d55e9-->
Trotz dieser schnellen lokalen Konvergenz sind exakte Newton-Schritte für tiefe neuronale Netze aus drei Gründen unpraktisch:

<!--en:938d161e-->
1. **Rechenkosten:** Mit Hesse-Vektor-Produkten (:numref:`subsec_mdl-hvp`) müssen wir die $d \times d$-Hesse-Matrix mit ihren $10^{18}$ Einträgen bei $d = 10^9$ nicht speichern. Die iterative Lösung von $\mathbf{H}\boldsymbol{\epsilon} = -\nabla f(\mathbf{x})$ erfordert dennoch Dutzende Rückwärtsdurchläufe mit Backpropagation pro Parameterupdate.

<!--en:cde0ae3e-->
2. **Negative Krümmung:** Da $\mathbf{H}$ alle kritischen Punkte gleich behandelt, kehren negative Eigenwerte die Schrittrichtung um. Sie treiben die Iterierten bergauf zu Sattelpunkten oder lokalen Maxima. Abhilfe schaffen eine Dämpfung oder eine Projektion, die Eigenwerte durch ihre Beträge ersetzt (:numref:`subsec_mdl-why-not-newton`).

<!--en:328b7123-->
3. **Empfindlichkeit gegenüber Minibatch-Rauschen:** Bilden wir die Kehrwerte verrauschter Krümmungsschätzungen nahe null, verstärkt das die Stichprobenschwankungen enorm.

<!--en:f2476681-->
### Diagonale Vorkonditionierung

<!--en:7ade00dd-->
Gängige Optimierer im Deep Learning greifen die Grundidee der Verfahren zweiter Ordnung auf, ohne die gesamte Hesse-Matrix zu invertieren. Sie skalieren jede Koordinate (oder jede Schichtmatrix) anhand einer kostengünstigen Schätzung ihrer Krümmung. Allgemein heißt eine Aktualisierung der Form $\mathbf{x} \leftarrow \mathbf{x} - \eta \mathbf{P}^{-1} \nabla f(\mathbf{x})$ **vorkonditionierter Gradientenabstieg**. Der **Vorkonditionierer** $\mathbf{P}$ senkt dabei die effektive Konditionszahl von $\mathbf{P}^{-1/2}\mathbf{H}\mathbf{P}^{-1/2}$.

<!--en:024afb21-->
Die einfachste Wahl ist der **diagonale Vorkonditionierer** $\mathbf{P} = \operatorname{diag}(\mathbf{H})$. Er teilt jede Komponente des Gradienten durch die zweite Ableitung nach der jeweiligen Koordinate:

<!--en:cecab97c-->
In unserem achsenparallelen Tal gilt $\operatorname{diag}(\mathbf{H}) = (0.2, 4)$. Nach :eqref:`eq_gd-diag-precond` schrumpfen beide Koordinaten bei jedem Schritt um denselben Faktor $1 - \eta$. So sinkt die effektive Konditionszahl von $\kappa = 20$ auf $1$. Bei $\eta = 0.5$ fallen die Beträge beider Koordinaten unter $1\%$ ihres Anfangsbetrags, und zwar nach nur $7$ Schritten ($0.5^7 \approx 0.0078$) statt nach $47$:

<!--en:1cc82bdf-->
Bei tiefen Netzen kennen wir $\operatorname{diag}(\mathbf{H})$ nicht unmittelbar. Außerdem verlaufen die Hauptkrümmungsrichtungen im Allgemeinen schräg zu den Koordinatenachsen. Drehen wir etwa das Tal um $45^\circ$, so sind die Diagonaleinträge von $\mathbf{H}$ gleich, und die diagonale Vorkonditionierung wird wirkungslos (siehe :exref:`ex_gd-diagonal-preconditioning`). Spätere Abschnitte behandeln beide Einschränkungen. Adam (:numref:`sec_adam`) schätzt während des Trainings für jede Koordinate einen Skalenfaktor aus den quadrierten Gradienten. Momentum (:numref:`sec_momentum`) und Muon (:numref:`sec_muon`) beschleunigen den Abstieg auch in Richtungen, die schräg zu den Achsen verlaufen. Zunächst untersuchen wir jedoch, was geschieht, wenn wir exakte Gradienten durch verrauschte Schätzungen aus Minibatches ersetzen (:numref:`sec_sgd`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ec32a070-->
Für eine positiv definite quadratische Funktion mit Hesse-Eigenwerten $\lambda_{\min} \le \cdots \le \lambda_{\max}$ ist der Gradientenabstieg mit positiver Lernrate genau dann stabil, wenn $\eta < 2/\lambda_{\max}$ gilt. Für eine relative Genauigkeit $\epsilon$ bei der optimalen konstanten Schrittweite $\eta^\star = 2/(\lambda_{\min} + \lambda_{\max})$ benötigt er $t \approx (\kappa/2)\log(1/\epsilon)$ Schritte, wobei $\kappa = \lambda_{\max}/\lambda_{\min}$ die Konditionszahl ist. In hochdimensionalen nichtkonvexen Landschaften sind kritische Punkte bei hohem Verlust überwiegend strikte Sattelpunkte. Der Gradientenabstieg entkommt ihnen entlang der Eigenvektoren mit negativer Krümmung. Moderne Aktivierungen, Normalisierungsschichten und Residualverbindungen verhindern Aktivierungsplateaus weitgehend. Für jede $L$-glatte Zielfunktion garantiert das Abstiegslemma eine monotone Abnahme, sofern $0 < \eta < 2/L$ gilt. Umgekehrt destabilisiert eine feste Lernrate $\eta$ jedes Minimum mit einer Schärfe über $2/\eta$. Beim Training neuronaler Netze auf dem gesamten Trainingssatz steigt die Schärfe allmählich, bis $\lambda_{\max}$ am *Rand der Stabilität* um $2/\eta$ schwankt. Skalieren wir den Gradienten mit einem Vorkonditionierer, der die Krümmung berücksichtigt, sinkt die effektive Konditionszahl. Diagonale Vorkonditionierung beseitigt eine achsenparallele schlechte Konditionierung mit einem Speicherbedarf von $O(d)$. Auf dieser Idee beruhen die adaptiven koordinatenweisen Verfahren in :numref:`sec_adam`.

<!--en:2294243a-->
## Übungen

<!--en:66808e51-->
1. [code] **Fluchtzeit an einem Sattelpunkt.** Betrachten Sie die Sattelfläche $f(x, y) = x^2 - \mu y^2$ mit $\mu > 0$ (Krümmung vom Betrag $2\mu$ nach unten). Starten Sie den Gradientenabstieg mit Lernrate $\eta$ bei $(1, \delta)$.
    1. Zeigen Sie, dass die Anzahl der Schritte bis $|y| \geq 1$ näherungsweise $\log(1/\delta)/(2\eta\mu)$ beträgt, wenn $\eta\mu$ klein ist. Prüfen Sie diese Schätzung numerisch für $\mu \in \{1, 0.01\}$ bei $\delta = 10^{-2}$ und $\eta = 0.1$. Vergleichen Sie sie mit der Schrittzahl, die eine Koordinate bei positiver Krümmung $2\mu$ benötigt, um auf das $\delta$-Fache ihres Anfangsbetrags zu schrumpfen.
    1. Setzen Sie $\mu = 1$ und $\delta = 0$. Addieren Sie bei jeder Gradientenauswertung gaußsches Rauschen mit Mittelwert null und Standardabweichung $\sigma \in \{10^{-3}, 10^{-1}\}$ und messen Sie die Fluchtzeit. Wie passt Ihr Ergebnis dazu, dass die stabile Mannigfaltigkeit Maß null hat (siehe :numref:`subsec_opt-intro-saddles`)?
   :exlabel:`ex_optimization-intro-saddle-escape-time`

<!--en:c64c26be-->
1. [code] **Lernrate und Schärfe.** Ein zweischichtiges skalares lineares Netz liefert die Vorhersage $ab$ bei Eingabe $1$. Sein quadratischer Verlust ist $\ell(a, b) = \frac{1}{2}(ab - 1)^2$. Damit ist jeder Punkt auf der Hyperbel $ab = 1$ ein globaler Minimierer.
    1. Zeigen Sie, dass die Hesse-Matrix an einem Minimierer $(a, b)$ auf $ab = 1$ den größten Eigenwert $\lambda_{\max} = a^2 + b^2$ hat. Welche Minimierer sind am flachsten?
    1. Führen Sie $200$ Schritte des Gradientenabstiegs vom Startpunkt $(a, b) = (4, 0.1)$ mit $\eta \in \{0.05, 0.1, 0.15, 0.2\}$ aus. Vergleichen Sie die Schärfe $\lambda_{\max}$ am Ende mit der Stabilitätsschwelle $2/\eta$.
   :exlabel:`ex_optimization-intro-learning-rate-sharpness`

<!--en:c83dc797-->
1. **Permutationssymmetrische Minima und Nichtkonvexität.** ● Sei $\boldsymbol{\theta}^\star$ ein lokales Minimum eines MLP mit einer einzigen versteckten Schicht aus $d \ge 2$ versteckten Einheiten. Keine zwei dieser Einheiten haben im Minimum identische Eingangsgewichte.
    1. Zeigen Sie, dass sich durch Permutation der versteckten Einheiten $d!$ verschiedene Parametervektoren mit gleichem Verlust ergeben. Beweisen Sie, dass bei konvexem Verlust auch ihr Mittelwert ein globaler Minimierer wäre, der einem Netz mit nur einer versteckten Einheit entspricht.
    1. Bestimmen Sie für eine $\tanh$-Aktivierung die Symmetrie jeder versteckten Einheit unter Vorzeichenwechsel. Geben Sie an, wie viele äquivalente Minima sich daraus ergeben.
   :exlabel:`ex_optimization-intro-permutation-symmetric-minima`

<!--en:6a1d80b4-->
1. **Eigenwerte symmetrischer Zufallsmatrizen.** ● Sei $\mathbf{M}$ eine symmetrische $k \times k$-Zufallsmatrix mit unabhängigen Einträgen $M_{ij} = M_{ji}$ für $i \le j$, gezogen aus symmetrischen Verteilungen $p_{ij}(x) = p_{ij}(-x)$ :cite:`Wigner.1958`.
    1. Beweisen Sie, dass ein gleichverteilt gewählter Eigenwert $\lambda$ von $\mathbf{M}$ die Gleichung $P(\lambda > 0) = P(\lambda < 0)$ erfüllt. Geben Sie mithilfe des wignerschen Halbkreisgesetzes auf $[-2\sigma, 2\sigma]$ den Grenzwert für den Anteil negativer Eigenwerte in einer einzelnen großen Realisierung von $\mathbf{M}/\sqrt{k}$ an.
    1. Berechnen Sie für $k = 2$ mit $M_{ii} \sim \mathcal{N}(0, 1)$ und $M_{12} \sim \mathcal{N}(0, 1/2)$ die exakte Wahrscheinlichkeit, dass beide Eigenwerte positiv sind. Vergleichen Sie das Ergebnis mit der Schätzung $2^{-2}$ für unabhängige Münzwürfe.
   :exlabel:`ex_optimization-intro-eigenvalues-symmetric-random-matrices`

<!--en:aa3677ed-->
1. **Abstiegsgarantie über viele Schritte.** Sei $f$ eine $L$-glatte Funktion mit unterer Schranke $f^\star$. Starten Sie den Gradientenabstieg mit $\eta = 1/L$ bei $\mathbf{x}_0$.
    1. Summieren Sie :eqref:`gd-guarantee` über $T$ Schritte und zeigen Sie so, dass $\min_{0 \le t < T} \|\nabla f(\mathbf{x}_t)\|^2 \le 2L\,(f(\mathbf{x}_0) - f^\star)/T$ gilt. Setzt diese Schranke Konvexität voraus?
    1. Starten Sie im Tal bei $(-5, -2)$ und vergleichen Sie für $T = 100$ die obere Schranke mit dem tatsächlich erreichten Quadrat der Gradientennorm.
   :exlabel:`ex_gd-guarantee-long-run`

<!--en:12e58a20-->
1. [code] **Erreichbare Minima und Training unterhalb der Stabilitätsgrenze.**
    1. Bestimmen Sie die vier lokalen Minima von $f(x) = x\cos(0.15\pi x)$ auf $(-20, 30)$. Berechnen Sie für jedes Minimum $f''$ und die Stabilitätsschwelle $2/f''$. Starten Sie den Gradientenabstieg im Abstand $0.01$ vom jeweiligen Minimum und testen Sie die Schrittweiten $0.9\times$ und $1.1\times$ der zugehörigen Schwelle.
       :exlabel:`ex_gd-minima-within-reach`
    1. Sagen Sie für das Netz mit 25 Parametern aus :numref:`subsec_gd-edge-of-stability` die endgültige Schärfe von `gd_sharpness(0.5, P0)` voraus, bevor Sie die Funktion ausführen. Testen Sie anschließend $\eta = 0.1$ für $50{,}000$ Schritte ($2/\eta = 20$). Erklären Sie, warum der Verlust monoton fällt, wenn das Netz die Daten bereits gut anpasst, bevor die Schärfe $2/\eta$ erreicht (`rising = 0%`).
       :exlabel:`ex_gd-below-the-edge`

<!--en:aca2846c-->
1. [code] **Diagonale Vorkonditionierung bei Drehung und Konvergenzordnung des Newton-Verfahrens.** ●
    1. Drehen Sie das Tal um den Winkel $\theta$ mithilfe von $\mathbf{H}_\theta = \mathbf{R}_\theta \operatorname{diag}(0.2, 4)\mathbf{R}_\theta^\top$. Berechnen Sie die Konditionszahl der diagonal vorkonditionierten Hesse-Matrix $\mathbf{D}^{-1/2}\mathbf{H}_\theta\mathbf{D}^{-1/2}$ mit $\mathbf{D} = \operatorname{diag}(\mathbf{H}_\theta)$ als Funktion von $\theta$. Prüfen Sie für $\theta = 45^\circ$ numerisch nach, dass die diagonale Vorkonditionierung nur eine skalare Änderung der Lernrate bewirkt und $\kappa = 20$ gilt.
       :exlabel:`ex_gd-diagonal-preconditioning`
    1. Wenden Sie das eindimensionale Newton-Verfahren $x_{t+1} = x_t - f'(x_t)/f''(x_t)$ auf $f(x) = \cosh(0.5x)$ mit Startwert $x_0 = 10$ an. Schätzen Sie die lokale Konvergenzordnung $p$ in $|x_{t+1}| \approx C|x_t|^p$ nahe $0$. Leiten Sie $p = 3$ aus der Taylor-Entwicklung von $x - 2\tanh(0.5x)$ her.
       :exlabel:`ex_gd-newton-order`

<!--en:29170301-->
[Diskussion](https://d2l.discourse.group/t/351)

<!--en:4001df12-->
Optimierungslandschaften und Gradientenabstieg<br>**schlecht konditionierte Täler · Konditionszahl κ · Sattelpunkte und Plateaus · Abstiegslemma · Rand der Stabilität · Vorkonditionierung**

<!--en:95a80165-->
::: {.slide title="Warum der Gradientenabstieg kaum vorankommt oder plötzlich divergiert"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:b528e02e-->
- **Empirische Risikominimierung:** Das Verhalten bei der Aktualisierung $\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$ in hohen Dimensionen ($d \gg 10^6$) bestimmen vor allem drei geometrische Eigenschaften, nicht ungünstige lokale Minima:

<!--en:173f3909-->
- **1. Schlechte Konditionierung ($\kappa = \lambda_{\max}/\lambda_{\min} \gg 1$):** Große Krümmungsverhältnisse begrenzen die stabile Schrittweite auf $\eta < 2/\lambda_{\max}$. In flachen Richtungen kommt das Verfahren kaum voran.

<!--en:47e0f47f-->
- **2. Hochdimensionale Sattelpunkte und zunehmende Schärfe:** Bei hohem Verlust sind kritische Punkte fast immer Sattelpunkte. Die Schärfe $\lambda_{\max}(\nabla^2 f(\mathbf{x}_t))$ steigt im Trainingsverlauf, bis sie sich bei $2/\eta$ einpendelt.

<!--en:79117233-->
::: {.slide title="Das quadratische Tal mit den Krümmungen 0.2 und 4.0"}

<!--en:805524a0-->
[Das quadratische Modell]{.kicker}

<!--en:0c66987e-->
Gegeben ist das achsenparallele Tal $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$ mit Gradient $\nabla f(\mathbf{x}) = (0.2x_1, 4x_2)$ und Krümmungen $\lambda_1 = 0.2, \lambda_2 = 4.0$:

<!--en:91bab1ba-->
- **Entkoppelte multiplikative Aktualisierungen:**
  $$x_i \leftarrow (1 - \lambda_i \eta)\, x_i$$

<!--en:35a21bd3-->
- **Stabilitätsgrenze ($x_2$, $\lambda_2 = 4$):** Für eine Kontraktion muss $|1 - 4\eta| < 1$ gelten, also **$\eta < 0.5$**.

<!--en:ce115c2a-->
- **Konvergenzgeschwindigkeit ($x_1$, $\lambda_1 = 0.2$):** Für jede stabile Schrittweite $\eta < 0.5$ gilt in der flachen Richtung $1 - 0.2\eta > 0.9$. Pro Schritt sinkt der Fehler um $<10\%$ seines Restbetrags entlang $x_1$.

<!--en:bec8f0c0-->
::: {.slide title="Dynamik bei η = 0.45: Schwingungen quer zum Tal, langsamer Abstieg entlang der Sohle"}

<!--en:06892055-->
[Trajektorie im Tal]{.kicker}

<!--en:3558836f-->
- Bei $\eta = 0.45$ (knapp unter der Stabilitätsgrenze $2/\lambda_2 = 0.5$) entwickeln sich die beiden Koordinaten vom Startpunkt $(-5, -2)$ aus sehr unterschiedlich:
  - **Steile Talwand ($x_2$):** Durch den Faktor $1 - 4(0.45) = -0.80$ wechselt die Koordinate bei jedem Schritt das Vorzeichen. Sie schießt über die Talsohle hinaus, ihr Betrag sinkt dabei um $20\%$ pro Schritt.
  - **Flache Talsohle ($x_1$):** Mit dem Faktor $1 - 0.2(0.45) = 0.91$ nähert sich die Koordinate langsam und ohne Vorzeichenwechsel der Null. Nach $30$ Schritten gilt $x_1 \approx -5 \times 0.91^{30} \approx -0.30$.

<!--en:231a1261-->
- Der negative Gradient $(1, 8)$ am Punkt $(-5, -2)$ weist fast genau quer über die steile Talflanke und am Ursprung vorbei.

<!--en:cddce95a-->
::: {.slide title="Ausgleich zwischen beiden Achsen bei der optimalen Schrittweite η* = 0.476"}

<!--en:9a07be39-->
[Variation der Schrittweite]{.kicker}

<!--en:e9e5fc20-->
- Für Konvergenz müssen *beide* Koordinaten betragsmäßig unter $1\%$ ihres Anfangsabstands sinken. Die optimale Schrittweite erfüllt
  $$1 - \lambda_1 \eta^\star = -(1 - \lambda_2 \eta^\star)$$

<!--en:9dcbfe02-->
- Auflösen ergibt $\eta^\star = \frac{2}{\lambda_1 + \lambda_2} = \frac{2}{4.2} \approx 0.476$. Beide Faktoren haben den Betrag $0.905$, sodass der Fehler die Marke von $1\%$ in **$47$ Schritten** erreicht.

<!--en:520e0613-->
- **Steiler Anstieg oberhalb von $\eta^\star$:** Knapp darunter, bei $\eta = 0.45$, sind nur $2$ zusätzliche Schritte nötig ($49$). Knapp darüber, bei $\eta = 0.49$, sind es mehr als doppelt so viele ($113$ Schritte).

<!--en:9c5f5f8e-->
::: {.slide title="Allgemeine quadratische Funktionen und die Konditionszahl κ"}

<!--en:369fb24d-->
[Analyse in der Eigenbasis]{.kicker}

<!--en:018a23dd-->
- **Entkoppelte Eigenmoden:** Für $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{H}\mathbf{x}$ mit Eigenwerten $0 < \lambda_{\min} \le \cdots \le \lambda_{\max}$ schrumpft die Komponente entlang jedes Eigenvektors $\mathbf{v}_i$ unabhängig um den Faktor $(1 - \eta\lambda_i)^t$.

<!--en:64bd6682-->
- **Globale Stabilitätsgrenze:** Damit jede Mode schrumpft, muss $|1 - \eta\lambda_i| < 1$ gelten. Die Richtung mit der größten Krümmung bestimmt die Grenze: **$\eta < 2/\lambda_{\max}$**.

<!--en:b546660e-->
- **Konditionszahl $\kappa = \lambda_{\max}/\lambda_{\min}$:** Bei $\eta^\star = \frac{2}{\lambda_{\min} + \lambda_{\max}}$ beträgt der Kontraktionsfaktor im ungünstigsten Fall $\frac{\kappa - 1}{\kappa + 1} \approx 1 - \frac{2}{\kappa}$. Damit sind **$t \approx \frac{\kappa}{2}\ln\frac{1}{\epsilon}$** Schritte nötig.

<!--en:3ba3d9ca-->
::: {.slide title="Kritische Punkte: Warum in hohen Dimensionen Sattelpunkte überwiegen"}

<!--en:a864f85b-->
[Nichtkonvexe Geometrie]{.kicker}

<!--en:de5f0258-->
An jedem kritischen Punkt ($\nabla f(\mathbf{x}) = \mathbf{0}$) lässt sich die lokale Geometrie anhand der Eigenwerte der Hesse-Matrix $\mathbf{H} = \nabla^2 f(\mathbf{x})$ einordnen:

<!--en:7cd7cdfc-->
- **Lokales Minimum ($\mathbf{H} \succ \mathbf{0}$):** Alle $d$ Eigenwerte sind strikt positiv.

<!--en:bbb81a1a-->
- **Lokales Maximum ($\mathbf{H} \prec \mathbf{0}$):** Alle $d$ Eigenwerte sind strikt negativ.

<!--en:99b4182e-->
- **Sattelpunkt:** Es gibt positive und negative Eigenwerte (z. B. bei $z = x^2 - y^2$).

<!--en:694c6396-->
- **Warum Minima bei hohem Verlust selten sind:** Sind die Vorzeichen der Eigenwerte bei hohem Verlust annähernd symmetrisch verteilt, so sind alle $d$ Eigenwerte mit Wahrscheinlichkeit $\sim 2^{-d}$ positiv. Die kritischen Punkte sind daher fast ausschließlich **Sattelpunkte**.

<!--en:7d69cbdc-->
::: {.slide title="Warum der Gradientenabstieg strikten Sattelpunkten entkommt"}

<!--en:2254ea91-->
[Sattelpunkten entkommen]{.kicker}

<!--en:51809a55-->
- **Exponentielle Abstoßung in instabilen Richtungen:** An einem strikten Sattelpunkt mit negativem Eigenwert $\lambda = -\mu < 0$ verstärkt jeder Gradientenschritt die Auslenkung $c_t$:
  $$c_{t+1} = (1 + \eta\mu)\, c_t$$

<!--en:f614500b-->
- **Stabile Mannigfaltigkeit vom Maß null:** Bei Zufallsinitialisierung ist die Projektion auf den instabilen Unterraum mit Wahrscheinlichkeit $1$ von null verschieden (Lee et al., 2016).

<!--en:70cc3e66-->
- **Logarithmische Fluchtzeit:** Vom Anfangsabstand $\delta > 0$ bis zum Abstand $1$ dauert es nur $t \approx \frac{\ln(1/\delta)}{\eta\mu}$ Schritte. Minibatch-Rauschen stößt die Iterierten selbst bei $\delta = 0$ vom Grat.

<!--en:600960cb-->
::: {.slide title="Aktivierungsplateaus und Ausreißer im Hesse-Spektrum"}

<!--en:74fc93cd-->
[Plateaus und Spektren neuronaler Netze]{.kicker}

<!--en:53449900-->
- **Aktivierungsplateaus:** Sättigende Nichtlinearitäten erzeugen ausgedehnte flache Bereiche ohne kritischen Punkt. Für $f(x) = \tanh(x)$ gilt bei $x = 4$ die Beziehung $f'(4) = 1 - \tanh^2(4) \approx 0.0013$. Der Gradientenabstieg mit $\eta = 1$ braucht daher $370$ Schritte allein für den Weg von $x = 4$ nach $x = 2$.

<!--en:a5a6c253-->
- **Abhilfe durch die Architektur:** Moderne Netze vermeiden Sättigungsplateaus durch ReLU- oder GELU-Aktivierungen, LayerNorm oder BatchNorm und Residualverbindungen.

<!--en:b5f2df5e-->
- **Empirisches Hesse-Spektrum:** Es bleibt eine besonders **schlechte Konditionierung**. Sehr vielen Eigenwerten nahe null stehen wenige große positive Ausreißer gegenüber. Der größte davon (die **Schärfe** $\lambda_{\max}$) begrenzt die Lernrate auf $\eta < 2/\lambda_{\max}$.

<!--en:2567003f-->
::: {.slide title="L-Glattheit und Abstiegslemma"}

<!--en:93d69e5c-->
[Garantien für nichtquadratische Funktionen]{.kicker}

<!--en:b83f2c8f-->
- **Quadratische obere Schranke ($L$-Glattheit):** Gilt überall $\|\nabla^2 f(\mathbf{x})\|_2 \le L$, so folgt $f(\mathbf{x}+\boldsymbol{\epsilon}) \le f(\mathbf{x}) + \boldsymbol{\epsilon}^\top\nabla f(\mathbf{x}) + \frac{L}{2}\|\boldsymbol{\epsilon}\|^2$.

<!--en:47ef1e19-->
- **Abstiegslemma ($\boldsymbol{\epsilon} = -\eta\nabla f(\mathbf{x})$):**
  $$f(\mathbf{x} - \eta\nabla f(\mathbf{x})) \le f(\mathbf{x}) - \eta\Big(1 - \frac{\eta L}{2}\Big)\|\nabla f(\mathbf{x})\|^2$$

<!--en:f6b484df-->
- **Garantierte Abnahme:** Für jedes $0 < \eta < 2/L$ sinkt $f$ außerhalb kritischer Punkte strikt. Die größte garantierte Abnahme pro Schritt beträgt $\frac{\|\nabla f\|^2}{2L}$ bei $\eta = 1/L$.

<!--en:086e89cd-->
::: {.slide title="Wie die Lernrate Minima nach ihrer Schärfe auswählt"}

<!--en:787edb90-->
[Lineare Stabilität von Minima]{.kicker}

<!--en:7cc06f81-->
- **Krümmungsfilter:** Nahe einem Minimum $x^\star$ mit Krümmung $\lambda = f''(x^\star) > 0$ multipliziert ein Schritt die Auslenkung $x - x^\star$ mit $1 - \eta\lambda$:
  - **Stabil ($\lambda < 2/\eta$):** Wegen $|1 - \eta\lambda| < 1$ nähern sich die Iterierten dem Minimum $x^\star$.
  - **Instabil ($\lambda > 2/\eta$):** Wegen $1 - \eta\lambda < -1$ verlassen die Iterierten das Einzugsgebiet.

<!--en:b20f8be8-->
- **Beispiel $f(x) = x\cos(0.15\pi x)$:** Bei $\eta = 0.5$ ($2/\eta = 4$) sind das tiefe, scharfe Minimum ($f'' = 1.81$) und das seichte, flache Minimum ($f'' = 0.98$) beide stabil. Steigt die Lernrate auf $\eta = 2.0$ ($2/\eta = 1$), wird das tiefe Minimum instabil (Faktor $-2.62$). Das Verfahren konvergiert dann zum flachen Minimum.

<!--en:5230d79b-->
::: {.slide title="Auswahl nach Schärfe bei η = 0.5 und η = 2.0"}

<!--en:acde40e1-->
[Experiment]{.kicker}

<!--en:d7a6b308-->
- Start bei $x_0 = 10$ für die Funktion $f(x) = x\cos(0.15\pi x)$:
  - **$\eta = 0.5$ ($2/\eta = 4 > 1.81$):** Das Verfahren konvergiert rasch zum tiefen, schärferen Minimum bei $x = 7.269$.
  - **$\eta = 2.0$ ($2/\eta = 1 < 1.81$):** Das tiefe Minimum ist instabil ($1 - \eta f'' = -2.62$). Die Iterierte springt aus seinem Einzugsgebiet und nähert sich dem weniger tiefen, flacheren Minimum nahe $x = -1.83$ ($f'' = 0.98 < 1$).

<!--en:998dde88-->
- **Grundprinzip:** Die Lernrate $\eta$ wählt Lösungen nach ihrer **Krümmung ($\lambda_{\max} \le 2/\eta$)** aus. Der Verlustwert im Minimum ist dafür unerheblich.

<!--en:a8e66899-->
::: {.slide title="Fortschreitende Zunahme der Schärfe und Rand der Stabilität"}

<!--en:da8a976d-->
[Dynamik neuronaler Netze]{.kicker}

<!--en:f98549cb-->
- Beim Gradientenabstieg auf dem gesamten Trainingssatz bleibt die Schärfe $\lambda_{\max}(\nabla^2 f(\mathbf{x}_t))$ der Hesse-Matrix eines neuronalen Netzes nicht konstant. Sie **steigt stetig**, während der Verlust sinkt (**fortschreitende Zunahme der Schärfe**, Cohen et al., 2021).

<!--en:e5230551-->
- Für ein MLP mit 25 Parametern und $\tanh$-Aktivierung, angepasst an $y = \sin(3x)$, lässt sich die exakte $25 \times 25$-Hesse-Matrix alle $500$ Schritte für $10$ Zufallsinitialisierungen berechnen:
  - Die normierte Schärfe $\frac{\lambda_{\max}}{2/\eta}$ steigt bis auf **$1.0$** ($\lambda_{\max} \approx 2/\eta$) und bleibt dort bis zum Trainingsende.

<!--en:2c875903-->
::: {.slide title="Am Rand der Stabilität: nichtmonotoner Abstieg" layout="tall-output"}

<!--en:b892d649-->
[Selbststabilisierung bei 2/η]{.kicker}

<!--en:621b785e-->
- **Alle $10$ Zufallsinitialisierungen:** Bei $\eta = 0.4$ pendelt sich die Schärfe bei $\lambda_{\max} \approx 5.0 = 2/0.4$ ein. Bei $\eta = 0.25$ steigt sie meist höher, bis auf $\lambda_{\max} \approx 8.0 = 2/0.25$. Einige Läufe passen die Daten schon vorher gut an und bleiben knapp darunter. Bei ihnen steigt der Verlust in $\approx 0\%$ der Schritte.

<!--en:61634499-->
- **Warum divergiert das Verfahren nicht?** Übersteigt $\lambda_{\max}$ kurzzeitig $2/\eta$, wächst die Schwingung in der Richtung größter Krümmung ($1 - \eta\lambda_{\max} \approx -1$, der Verlust steigt bei $\sim 50\%$ der Schritte). Zusammen mit den dritten Ableitungen des Verlusts treibt sie die Iterierte in flachere Bereiche (Damian et al., 2023).

<!--en:f1709aa0-->
::: {.slide title="Das Newton-Verfahren und die Grenzen der exakten Inversion der Hesse-Matrix"}

<!--en:f20fdef3-->
[Optimierung zweiter Ordnung]{.kicker}

<!--en:2102cae4-->
- **Newton-Schritt ($\mathbf{x} \leftarrow \mathbf{x} - \mathbf{H}^{-1}\nabla f(\mathbf{x})$):** Jede Eigenrichtung erhält den Faktor $1/\lambda_i$. Damit gilt $\kappa \to 1$, und das Verfahren minimiert jede positiv definite quadratische Funktion in $1$ Schritt.

<!--en:e2201d53-->
- **Grenzen exakter Newton-Schritte im Deep Learning:**
  1. **$O(d^2)$ Speicherbedarf und $O(d^3)$ Rechenaufwand für die Inversion:** $\mathbf{H}$ hat $10^{18}$ Einträge bei $d = 10^9$.
  2. **Anziehung durch Sattelpunkte:** Negative Eigenwerte $\lambda_i < 0$ kehren das Vorzeichen des Schritts um und lenken die Iterierten *bergauf* zu Sattelpunkten oder Maxima.
  3. **Rauschverstärkung:** Der Kehrwert kleiner, verrauschter Krümmungsschätzungen lässt die Varianz über Minibatches stark anwachsen.

<!--en:6c1d068c-->
::: {.slide title="Diagonale Vorkonditionierung: Skalierung jeder Koordinate"}

<!--en:5e0396e7-->
[Kostengünstige Skalierung anhand der Krümmung]{.kicker}

<!--en:bf70c3dc-->
- Der **vorkonditionierte Gradientenabstieg** $\mathbf{x} \leftarrow \mathbf{x} - \eta\mathbf{P}^{-1}\nabla f(\mathbf{x})$ ersetzt $\mathbf{H}$ durch eine kostengünstige $O(d)$-Näherung $\mathbf{P}$. Sie senkt die effektive Konditionszahl von $\mathbf{P}^{-1/2}\mathbf{H}\mathbf{P}^{-1/2}$.

<!--en:1312c5d0-->
- **Diagonale Vorkonditionierung ($\mathbf{P} = \operatorname{diag}(\mathbf{H})$):**
  $$x_i \leftarrow x_i - \eta\,\frac{\partial_i f(\mathbf{x})}{H_{ii}}$$

<!--en:8248b32e-->
- Im achsenparallelen Tal ($\operatorname{diag}(\mathbf{H}) = (0.2, 4)$) schrumpfen beide Koordinaten um den Faktor $1 - \eta$. Die Konditionszahl sinkt gemäß $\kappa = 20 \to 1$. Ein Restfehler von $1\%$ ist nach **$7$ Schritten** bei $\eta = 0.5$ ($0.5^7 \approx 0.0078$) erreicht, statt nach $47$.

<!--en:e97a3476-->
::: {.slide title="Verständnisfragen zu Konditionierung, Stabilität und Schärfe"}

<!--en:929b7671-->
[Wissen prüfen]{.kicker}

<!--en:14696181-->
Für jeden Fall die numerische Lösung berechnen:

<!--en:e48c33c7-->
1. **Konditionszahl und Schrittzahl:** Die Eigenwerte der Hesse-Matrix einer quadratischen Zielfunktion reichen von $\lambda_{\min} = 0.05$ bis $\lambda_{\max} = 10$. Wie groß sind die Obergrenze der stabilen Lernrate $\eta_{\max}$, die Konditionszahl $\kappa$ und die geschätzte Schrittzahl $t \approx \frac{\kappa}{2}\ln(1/\epsilon)$ für einen Fehler von $\epsilon = e^{-4} \approx 1.8\%$ bei $\eta^\star$?

<!--en:f64231bc-->
2. **Vorhersage am Rand der Stabilität:** Das Netz mit 25 Parametern und $\tanh$-Aktivierung wird per Gradientenabstieg auf dem gesamten Trainingssatz mit $\eta = 0.5$ trainiert. Bei welcher Schärfe $\lambda_{\max}$ der Hesse-Matrix endet der Anstieg?

<!--en:351a2bed-->
3. **Drehung des Tals:** Nach einer Drehung von $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$ um $45^\circ$: Welche Werte haben die Diagonaleinträge $H_{11}, H_{22}$ der gedrehten Hesse-Matrix? Wie groß ist die Konditionszahl nach diagonaler Vorkonditionierung?

<!--en:557b98b6-->
**Lösungen:**

<!--en:babefb6b-->
1. **$\eta_{\max} = 2/\lambda_{\max} = 0.2$**, **$\kappa = 10 / 0.05 = 200$** und **$t \approx \frac{200}{2} \times 4 = 400$ Schritte**.

<!--en:03562ec1-->
2. Die Schärfe pendelt sich an der Stabilitätsgrenze **$\lambda_{\max} \approx 2/\eta = 2/0.5 = 4.0$** ein.

<!--en:a57d0640-->
3. Bei $45^\circ$ sind die Diagonaleinträge gleich: $H_{11} = H_{22} = \frac{0.2 + 4.0}{2} = 2.1$. Die Division durch $\operatorname{diag}(\mathbf{H})$ skaliert lediglich $\eta$ mit dem Faktor $1/2.1$. Damit bleibt **$\kappa = 20$ unverändert**. Das motiviert Momentum (§{sec_momentum}) und die Vorkonditionierung mit der vollständigen Matrix (§{sec_muon}).

<!--en:780920cf-->
::: {.slide title="Zusammenfassung: Optimierungslandschaften, Konditionierung und Gradientenabstieg"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:75261ab7-->
| Begriff oder Phänomen | Mathematische Formel | Skalierung und Stabilität | Praktische Bedeutung |

<!--en:ace98ed6-->
| **Stabilitätsgrenze für quadratische Funktionen** | $\eta < \frac{2}{\lambda_{\max}}$ | Komponente in steilster Richtung wechselt bei $\eta\lambda_{\max} \ge 2$ das Vorzeichen und klingt an der Grenze nicht ab, oberhalb wächst ihr Betrag | Größter Hesse-Eigenwert begrenzt die globale Lernrate |

<!--en:74069387-->
| **Konditionszahl $\kappa$** | $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \ge 1$ | Schrittzahl für Genauigkeit $\epsilon$ wächst gemäß $t \approx \frac{\kappa}{2}\log\frac{1}{\epsilon}$ bei $\eta^\star = \frac{2}{\lambda_{\min}+\lambda_{\max}}$ | Kein skalares $\eta$ beschleunigt den Abstieg in schlecht konditionierten Tälern ($\kappa \gg 1$) über diese Grenze hinaus |

<!--en:e241f550-->
| **Entkommen aus strikten Sattelpunkten** | $c_{t+1} = (1 + \eta\mu)c_t$ ($\lambda = -\mu < 0$) | Bei Zufallsinitialisierung genügen $O(\frac{1}{\eta\mu}\log\frac{1}{\delta})$ Schritte zum Entkommen | An hochdimensionalen Sattelpunkten bleibt der Gradientenabstieg nicht stecken |

<!--en:7fa69be4-->
| **Abstiegslemma für $L$-glatte Funktionen** | $\Delta f \le -\eta(1 - \frac{\eta L}{2})\|\nabla f\|^2$ | Strikte Abnahme außerhalb kritischer Punkte für $0 < \eta < 2/L$, beste garantierte Abnahme bei $\eta = 1/L$ | Schranke für den Fortschritt pro Schritt, sofern $\|\nabla^2 f(\mathbf{x})\|_2 \le L$ |

<!--en:8fdf881b-->
| **Rand der Stabilität** | $\lambda_{\max}(\nabla^2 f(\mathbf{x}_t)) \to \frac{2}{\eta}$ | Schärfe pendelt sich bei $2/\eta$ ein, Verlust sinkt nichtmonoton | Größeres $\eta$ wählt flachere Einzugsgebiete, Aufwärmphase lenkt in flache Bereiche |

<!--en:4e0e3516-->
| **Diagonale Vorkonditionierung** | $\mathbf{x} \leftarrow \mathbf{x} - \eta\,\operatorname{diag}(\mathbf{H})^{-1}\nabla f(\mathbf{x})$ | Bei achsenparallelem Tal sinkt die Konditionszahl gemäß $\kappa \to 1$ mit Speicherbedarf $O(d)$ ($47 \to 7$ Schritte) | Grundlage der koordinatenweisen Skalierung während des Trainings mit AdaGrad, RMSProp und Adam |
