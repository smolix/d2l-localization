<!--en:adf410d9-->
# Entropie, Kreuzentropie und KL-Divergenz

<!--en:f457903e-->
Beim Training eines Klassifikators betrachten wir zuerst die Verlustkurve. Ihre Form lernen wir schnell zu deuten: Fällt sie, ist das ein gutes Zeichen. Verläuft sie flach, werden wir misstrauisch. Die Zahlen auf der vertikalen Achse sind schwerer zu deuten. Bisher können wir nur eine davon erklären. Ein frisch initialisierter Klassifikator mit zehn Klassen liefert einen Verlust von etwa $2.30$. In :numref:`sec_mlp-implementation` diente dieser Wert als Plausibilitätsprüfung für ein untrainiertes Netz. Die Zahl ist $\ln 10$. So viel kostet es, eines von zehn Labels zu benennen, wenn das Modell alle zehn noch für gleich wahrscheinlich hält. Das andere Ende der Kurve wirft weitergehende Fragen auf. Ist ein Endwert von $0.6$ gut? Wenn die Kurve abflacht, liegt das Plateau an einem Fehler im Training oder unvermeidlich an den Daten? Und wie niedrig kann der Verlust überhaupt werden, wenn einige Labels falsch sind?

<!--en:28fe09f3-->
Der Kreuzentropieverlust (engl. cross-entropy loss) ist der Mittelwert von $-\log q(y)$. Dieser Ausdruck misst, wie sehr das tatsächlich beobachtete Label $y$ das Modell überrascht. Der Mittelwert zerfällt auf natürliche Weise in zwei Teile. Der erste ist die *Entropie*. Sie gibt an, wie groß die Überraschung im Mittel für jemanden ist, der die wahre Verteilung der Labels bereits kennt. Die Daten selbst legen die Entropie fest, daher kann kein Modell sie senken. Der zweite Teil ist die *Kullback-Leibler-Divergenz (KL-Divergenz)*. Sie misst die zusätzliche Überraschung, wenn wir statt der wahren Verteilung ein unvollkommenes Modell verwenden. Sie ist nie negativ und verschwindet nur, wenn das Modell mit der wahren Verteilung übereinstimmt. Einen Verlustwert können wir daher nur im Verhältnis zu seiner irreduziblen Verlustuntergrenze (engl. loss floor) beurteilen.

<!--en:0e108da0-->
Zwei dieser Aussagen kennen wir aus dem vorigen Kapitel. Das Maximum-Likelihood-Verfahren minimiert eine Kreuzentropie (:numref:`subsec_mdl-nll-crossentropy`), und die KL-Divergenz ist nie negativ (:numref:`subsec_mdl-max-entropy-derivation`). Die zweite Aussage heißt Gibbs-Ungleichung. Dort dienten uns diese Ergebnisse als Werkzeuge, hier stehen sie im Mittelpunkt. Wir beginnen mit einem Kartenspiel und einer Quelle mit vier Symbolen. Dabei zählt die Entropie, wie viele Ja-Nein-Fragen wir brauchen. Anschließend vergleichen wir Verteilungen paarweise und kehren zu unserem Klassifikator zurück, um die Untergrenze seines Verlusts zu bestimmen. Zum Schluss untersuchen wir, was sich bei weichen Zielverteilungen oder bei stetigen reellwertigen Ergebnissen ändert.

<!--en:7d145ddb-->
Die Basis des Logarithmus bestimmt lediglich die Einheit. Zur Basis $2$ messen wir Information in *Bit*, zur Basis $e$ in *Nat*. **Wir rechnen durchgehend in Nat.** Das entspricht dem natürlichen Logarithmus `log` in numerischen Bibliotheken und bei der Angabe von Trainingsverlusten. Zum Zählen von Ja-Nein-Fragen eignen sich Bit besser. Größen in Bit kennzeichnen wir daher mit dem Index $2$, etwa $H_2$ für die Entropie. Ein Bit entspricht $\ln 2 \approx 0.693$ Nat. :numref:`tab_mdl-it-notation` fasst die Symbole dieses Kapitels zusammen.

<!--en:0c3e62aa-->
:Symbole im Kapitel zur Informationstheorie. Das Argument der binären Entropie $H_b(p)$ ist eine Zahl, keine Zufallsvariable. Die gegenseitige Information $I(X;Y)$ und die Größen in den letzten fünf Zeilen werden in späteren Abschnitten definiert.

<!--en:1f8afa60-->
| Symbol | Bedeutung | Einheit |

<!--en:a4ac8c20-->
| $I(x) = -\log p(x)$ | Informationsgehalt (Überraschungswert) des Ergebnisses $x$ (denselben Buchstaben verwendet man üblicherweise für die gegenseitige Information $I(X;Y)$) | Nat |

<!--en:ee222ec3-->
| $H(X)$, $H(P)$ | Entropie einer diskreten Zufallsvariable oder Verteilung. Der Index $2$ wie in $H_2$ und $\textrm{CE}_2$ kennzeichnet Angaben in Bit | Nat |

<!--en:7bbb2df2-->
| $H_b(p)$ | binäre Entropie einer Münze mit Wahrscheinlichkeit $p$ für Kopf | Nat |

<!--en:afd824fa-->
| $h(X)$ | differentielle Entropie einer stetigen Zufallsvariable | Nat, abhängig von der Maßeinheit von $X$ |

<!--en:288cf38b-->
| $D_{\textrm{KL}}(P \Vert Q)$ | KL-Divergenz von $P$ zu $Q$ | Nat |

<!--en:75c1fada-->
| $\textrm{CE}(P, Q)$ | Kreuzentropie von $P$ zu $Q$ | Nat |

<!--en:c0862614-->
| $k$ | Anzahl der Klassen (Größe der Ergebnismenge) | |

<!--en:b51256c3-->
| $n$ | Anzahl der Beispiele oder der Symbole einer Nachricht, Dimension in $\mathbb{R}^n$ | |

<!--en:1561cbed-->
| $\mathbf{p}$, $\mathbf{q}$ | Wahrscheinlichkeitsvektoren für $k$ Klassen (wahre Verteilung und Modell) | |

<!--en:51aa484c-->
| $\mathbf{z}$ | Logits mit $\mathbf{q} = \mathrm{softmax}(\mathbf{z})$ | |

<!--en:841b4dfe-->
| $\epsilon$ | beim Label-Smoothing umverteilte Wahrscheinlichkeitsmasse (in späteren Abschnitten eine Rauschvariable) | |

<!--en:3cc400e1-->
| $\tau$ | Temperatur | |

<!--en:6f2a81a8-->
| $\rho$ | Rate des Labelrauschens, also die Wahrscheinlichkeit, dass das erfasste Label von der wahren Klasse abweicht | |

<!--en:ad75907d-->
| $H(X \mid Y)$, $I(X;Y)$ | bedingte Entropie, gegenseitige Information | Nat |

<!--en:f4713f11-->
| $D_f(P \Vert Q)$ | f-Divergenz mit Divergenzfunktion $f$ | |

<!--en:0b3c9cbe-->
| $T$ | Kritiker, eine Funktion, deren Mittelwerte unter zwei Verteilungen eine Divergenz beschränken | |

<!--en:ef4a119e-->
| $g$ | Testfunktion einer integralen Wahrscheinlichkeitsmetrik | |

<!--en:80a4a113-->
| $\gamma$ | Kopplung (Transportplan) zweier Verteilungen | |

<!--en:da40b878-->
| $N$ | Anzahl der Punkte in jeder von zwei verglichenen Stichproben oder Anzahl der Kandidaten in einem kontrastiven Batch | |

<!--en:6d941be0-->
## Information und Entropie

<!--en:af667dcc-->
### Informationsgehalt

<!--en:d09c1777-->
Betrachten wir vier Aussagen über ein frisch gemischtes Kartenspiel. „Ich sehe eine Karte“ verrät uns nichts, denn das Ereignis ist sicher. „Ich sehe Herz“ grenzt das Ergebnis auf eine von vier gleich wahrscheinlichen Kartenfarben ein. „Es ist die Pik-Drei“ bezeichnet eine von $52$ gleich wahrscheinlichen Karten. Wer die Reihenfolge aller Karten nennt, gibt eine von $52!$ möglichen Anordnungen an. Je unwahrscheinlicher ein Ereignis ist, desto mehr erfahren wir, wenn wir es beobachten. Shannon :cite:`Shannon.1948` fasste diesen Gedanken in eine Zahl, den *Informationsgehalt* (engl. self-information) eines Ergebnisses $x$ mit der Wahrscheinlichkeit $p(x)$:

<!--en:3873a44b-->
Diese Größe heißt auch *Überraschungswert* (engl. surprisal) des Ergebnisses, wie in :numref:`subsec_info_theory_basics`. Je weniger wir mit einem Ergebnis gerechnet haben, desto mehr überrascht es uns. Entscheidend ist, wer die Wahrscheinlichkeit angesetzt hat. Ein Modell, das sich seiner falschen Vorhersagen sicher ist, findet selbst alltägliche Ereignisse höchst überraschend. Ein sicheres Ereignis hat den Informationsgehalt null. Halbiert sich die Wahrscheinlichkeit eines Ergebnisses, steigt sein Überraschungswert um $\ln 2$ Nat (ein Bit). Die ersten drei Aussagen über die Karten liefern $0$, $\ln 4 \approx 1.39$ und $\ln 52 \approx 3.95$ Nat. Ein Bit entspricht der Information aus einem Wurf einer fairen Münze. In Bit liefern die drei Aussagen $0$, $2$ und $5.70$.

<!--en:25909466-->
Warum wählen wir einen Logarithmus? Wir verlangen drei Eigenschaften. Ein sicheres Ereignis liefert keine Information, seltenere Ereignisse liefern mehr, und bei zwei unabhängigen Ereignissen addieren sich die Informationsgehalte. Da sich die Wahrscheinlichkeiten unabhängiger Ereignisse multiplizieren, lautet die dritte Forderung $I(p_1 p_2) = I(p_1) + I(p_2)$. Die einzigen fallenden Funktionen mit dieser Eigenschaft sind positive Vielfache von $-\log p$ (:citet:`Csiszar.2008`, den Beweis führen Sie in Aufgabe :exref:`ex_mdl-information-theory-show-that-self-information`). Der konstante Faktor legt nur die Einheit fest.

<!--en:6835bc6a-->
![Informationsgehalt $I(x) = -\log p(x)$ in Nat als Funktion der Wahrscheinlichkeit $p$. Sichere Ereignisse ($p = 1$) liefern keine Information, ein fairer Münzwurf ($p = \tfrac{1}{2}$) liefert $\ln 2 \approx 0.693$ Nat. Für $p \to 0$ divergiert die Kurve.](../img/mdl-it-self-info-curve.svg)

<!--en:68cafe39-->
:numref:`fig_mdl-self-info-curve` zeigt die fallende, konvexe Kurve. Sie erreicht bei $p = 1$ den Wert null und divergiert für $p \to 0$. Die vierte Aussage, die Reihenfolge des gesamten Kartenspiels, liegt weit außen auf dieser Kurve bei $\ln 52!$. Mit dem Logarithmus der Gammafunktion, $\ln 52! = \ln \Gamma(53)$, können wir ihren Informationsgehalt berechnen, ohne die 68-stellige ganze Zahl $52!$ zu bilden.

<!--en:e47ec18e-->
Wer die Reihenfolge der gemischten Karten angibt, liefert $156$ Nat, etwa so viel Information wie $226$ faire Münzwürfe.

<!--en:87229dba-->
### Entropie

<!--en:64fccc31-->
Den Überraschungswert können wir erst berechnen, wenn das Ergebnis vorliegt. Vorher können wir fragen, wie viel Überraschung wir im Mittel erwarten. Sei $X$ eine diskrete Zufallsvariable, die den Wert $x_i$ mit Wahrscheinlichkeit $p_i$ annimmt. Ihre *Entropie* ist der erwartete Informationsgehalt ihres Ergebnisses:

<!--en:92d985a1-->
Dabei setzen wir $0 \log 0 = 0$. Das folgt aus der Stetigkeit ($t \log t \to 0$ für $t \to 0$): Ein unmögliches Ergebnis trägt nichts bei. Da die Entropie nur von der Verteilung $P$ der Zufallsvariable $X$ abhängt, verwenden wir $H(P)$ und $H(X)$ gleichbedeutend.

<!--en:7890b09e-->
Auf einer endlichen Menge hat die Entropie in Bit eine anschauliche Bedeutung. Sie zählt, wie viele Ja-Nein-Fragen man bei geschickter Wahl im Mittel braucht, um das Ergebnis zu bestimmen. Betrachten wir die Verteilung $P = (\tfrac12, \tfrac14, \tfrac18, \tfrac18)$ auf vier Symbolen $a, b, c, d$. Dieses Beispiel begleitet uns durch den Abschnitt. Wir nennen es die *dyadische Quelle*, weil alle Wahrscheinlichkeiten Potenzen von $\tfrac12$ sind. Eine geschickte erste Frage lautet „Ist es $a$?“. Damit ist die Sache in der Hälfte der Fälle geklärt. Andernfalls fragen wir „Ist es $b$?“ und dann „Ist es $c$?“. Bei „Nein“ bleibt nur $d$ übrig. Die erwartete Anzahl der Fragen beträgt

<!--en:25406378-->
und entspricht damit der Entropie in Bit: $H_2(P) = \tfrac12 \log_2 2 + \tfrac14 \log_2 4 + 2 \cdot \tfrac18 \log_2 8
= 1.75$ Bit oder $1.213$ Nat. Jedes Ergebnis lässt sich mit $\log_2(1/p_i)$ Fragen bestimmen. Das ist sein Informationsgehalt in Bit.

<!--en:5f7a2ea5-->
Wer die Wahrscheinlichkeiten außer Acht lässt, fragt dagegen „Ist es $a$ oder $b$?“ und dann „Welches der beiden?“. Dafür sind immer $2$ Fragen nötig. In jeder Fragestrategie steckt ein Binärcode. Schreiben wir $1$ für Ja und $0$ für Nein, wird die Antwortfolge für ein Ergebnis zu dessen Codewort. Hier sind das $1$, $01$, $001$ und $000$ für $a$, $b$, $c$ und $d$. Die mittlere Anzahl der Fragen entspricht hier genau der Entropie, weil jede Wahrscheinlichkeit eine Potenz von $\tfrac12$ ist. Mit den Mitteln der Kodierung zeigt :numref:`sec_mdl-coding-compression`, dass die Abweichung stets klein bleibt. Für jede Verteilung auf einer endlichen Menge braucht die beste Strategie im Mittel zwischen $H_2$ und $H_2 + 1$ Fragen.

<!--en:90789fe9-->
Im Code blendet die Funktion `entropy` Wahrscheinlichkeiten von null aus und setzt so $0 \log 0 = 0$ um. Wir verwenden sie im gesamten Abschnitt, zunächst für die dyadische Quelle.

<!--en:d2e90601-->
Sie liefert genau die $1.75$ Bit, die wir beim Fragen gezählt haben.

<!--en:cdf9c16c-->
Wegen $0 \le p_i \le 1$ ist jeder Term $-p_i \log p_i$ nichtnegativ. Daher gilt $H(X) \ge 0$, mit Gleichheit nur bei einer Punktmasse, also wenn das Ergebnis sicher feststeht. Die Entropie hängt zudem nur von der Liste der Wahrscheinlichkeiten ab. Die Zahlenwerte oder Namen der Ergebnisse spielen keine Rolle. Nimmt eine Variable die Werte $1$, $100$, $10^6$ und $10^9$ mit den dyadischen Wahrscheinlichkeiten an, beträgt ihre Entropie weiterhin $1.75$ Bit.

<!--en:5c4e12c1-->
### Gemeinsame und bedingte Entropie

<!--en:6f122b6c-->
Beim maschinellen Lernen gehört zu jedem Label eine Eingabe, und jedes Zeichen steht im Kontext des vorangegangenen Textes. Wir brauchen daher die Entropie beider Variablen zusammen und die Entropie, die in einer Variable verbleibt, sobald die andere bekannt ist. Für die erste Größe brauchen wir keinen neuen Begriff. Ein Paar $(X, Y)$ mit gemeinsamer Wahrscheinlichkeitsfunktion $p(x, y)$ ist selbst eine Zufallsvariable, deren Ergebnisse Paare sind. Seine *gemeinsame Entropie* ist also $H(X, Y) = -\mathbb{E}[\log p(X, Y)]$. Die *bedingte Entropie* gibt an, welcher Überraschungswert für $Y$ im Mittel verbleibt, sobald wir $X$ beobachtet haben:

<!--en:3f380818-->
Dabei ist $H(Y \mid X = x)$ die Entropie der bedingten Verteilung $p(\cdot \mid x)$. Wegen $\log p(x, y) = \log p(x) + \log p(y \mid x)$ erhalten wir durch Bildung der Erwartungswerte auf beiden Seiten die *Kettenregel* für die Entropie:

<!--en:a0115464-->
In Worten ist die Entropie eines Paares die Entropie der ersten Variable zuzüglich der Unsicherheit, die bei bekannter erster Variable in der zweiten verbleibt. Durch wiederholtes Anwenden der Kettenregel erhalten wir $H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_1, \ldots, X_{i-1})$. Die Entropie einer Sequenz ist die Summe der Entropien ihrer Symbole, jeweils bedingt auf alle vorangegangenen Symbole. In derselben Reihenfolge sagt ein Sprachmodell Text vorher, Symbol für Symbol von links nach rechts.

<!--en:68d4519e-->
Das Fragespiel veranschaulicht diese Identität. Sei $X$ die Antwort auf die erste Frage „Ist es $a$?“, und sei $Y$ das Ergebnis der dyadischen Quelle. Da beide Antworten wie bei einem fairen Münzwurf gleich wahrscheinlich sind, gilt $H_2(X) = 1$ Bit. Nach „Ja“ bleibt nichts zu fragen ($0$ Bit). Nach „Nein“ lautet das Ergebnis $b$, $c$ oder $d$ mit den Wahrscheinlichkeiten $(\tfrac12, \tfrac14, \tfrac14)$. Die Entropie dieser Verteilung beträgt $1.5$ Bit. Im Mittel über beide Antworten erhalten wir $H_2(Y \mid X) = \tfrac12 \cdot 0 + \tfrac12 \cdot 1.5 = 0.75$ Bit. Nach der Kettenregel gilt $H_2(X, Y) = 1 + 0.75 = 1.75$ Bit. Die erste Antwort liefert ein Bit, die weiteren Fragen liefern den Rest. Diese Summe muss $H_2(Y)$ ergeben. Denn das Ergebnis $Y$ legt die Antwort $X$ vollständig fest. Es gilt also $H(X \mid Y) = 0$, und in umgekehrter Reihenfolge lautet die Kettenregel $H(X, Y) = H(Y) + 0$.

<!--en:d4d2990c-->
### Maximale Entropie auf einer endlichen Menge

<!--en:f2d1aae7-->
Wie groß kann die Entropie einer Variable mit $k$ Ergebnissen sein? Beginnen wir mit dem kleinsten Fall, einem einzelnen Münzwurf. Eine Bernoulli-Variable mit Wahrscheinlichkeit $p$ für Kopf hat die *binäre Entropie*

<!--en:58797f63-->
als Funktion des Skalars $p$. Ihr Argument ist also keine Zufallsvariable. :numref:`fig_mdl-bernoulli-entropy` zeigt die Kurve. Die Entropie verschwindet bei $p \in \{0, 1\}$, wo das Ergebnis sicher ist. Ihr Maximum erreicht sie bei der fairen Münze mit $H_b(\tfrac12) = \ln 2 \approx 0.693$ Nat (einem Bit). Eine verzerrte Münze mit $p = 0.1$ hat eine Entropie von $H_b(0.1) \approx 0.325$ Nat, weil ihr Ergebnis viel leichter zu erraten ist.

<!--en:fa354ae4-->
![Die binäre Entropie $H_b(p)$ in Nat ist symmetrisch um $p = \tfrac{1}{2}$. Dort erreicht sie ihr Maximum von $\ln 2 \approx 0.693$ Nat. An den deterministischen Endpunkten $p = 0$ und $p = 1$ steht das Ergebnis fest, und die Entropie verschwindet. Die Kurve ist konkav, jede Sehne liegt unter ihr.](../img/mdl-it-bernoulli-entropy.svg)

<!--en:afd29438-->
Die Kurve ist konkav: Jede Sehne zwischen zwei ihrer Punkte liegt unter der Kurve. Das Maximum liegt bei der fairen Münze. Dasselbe Prinzip gilt für jede endliche Anzahl von Ergebnissen. Nimmt $X$ höchstens $k$ Werte an, dann gilt

<!--en:14279248-->
mit Gleichheit genau dann, wenn $X$ auf allen $k$ Werten gleichverteilt ist. Die Gleichverteilung verteilt ihre Wahrscheinlichkeitsmasse so gleichmäßig wie möglich. Dadurch ist das Ergebnis so schwer wie möglich zu erraten. Für vier Ergebnisse ist die obere Schranke $\log_2 4 = 2$ Bit. Sie liegt $0.25$ Bit über den $1.75$ Bit der dyadischen Quelle. Den Beweis verschieben wir, bis uns die Gibbs-Ungleichung zur Verfügung steht. Dann genügt eine Zeile. Da diese Ungleichung zwei Verteilungen vergleicht, wenden wir uns nun dem Vergleich von Verteilungen zu.

<!--en:9d469bf5-->
## KL-Divergenz und Kreuzentropie

<!--en:3e606c39-->
### Die Kullback-Leibler-Divergenz

<!--en:d5e71d28-->
Wie stark weicht ein Modell $Q$ von einer wahren Verteilung $P$ ab? Die Informationstheorie misst das mit der *Kullback-Leibler-Divergenz (KL-Divergenz)*, auch *relative Entropie* genannt :cite:`Kullback.Leibler.1951`. Sie gibt den mittleren zusätzlichen Überraschungswert an, wenn wir von $Q$ ausgehen, die Ergebnisse aber aus $P$ stammen. Im Fragespiel entspricht die KL-Divergenz in Bit der mittleren Anzahl zusätzlicher Fragen, die eine auf $Q$ ausgelegte Strategie für Ergebnisse aus $P$ braucht, bis auf die Rundung auf ganze Fragen. Die Verteilungen $P$ und $Q$ hätten die Wahrscheinlichkeitsfunktionen (oder Dichten) $p$ und $q$. Die KL-Divergenz von $P$ zu $Q$ ist

<!--en:7257bc6e-->
Im Erwartungswert steht $\log p(x) - \log q(x) = (-\log q(x)) - (-\log p(x))$, also der Überraschungswert aus Sicht von $Q$ abzüglich des Überraschungswerts aus Sicht von $P$. Die Differenz ist positiv, wo $Q$ die Wahrscheinlichkeit eines Ergebnisses unterschätzt ($q(x) < p(x)$), und negativ, wo $Q$ sie überschätzt. Die KL-Divergenz mittelt diese Differenz unter der wahren Verteilung $P$.

<!--en:491eea98-->
Da $P$ sowohl die Stichproben als auch den Zähler liefert, ist die Definition asymmetrisch. Im Allgemeinen gilt $D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P)$. Ergebnisse mit $p(x) = 0$ tragen gemäß der Konvention $0 \log(0/q) = 0$ nichts bei. Gibt es jedoch ein Ergebnis mit $p(x) > 0$ und zugleich $q(x) = 0$, dann gilt $D_{\textrm{KL}}(P\|Q) = +\infty$.

<!--en:0d345e6f-->
Ein kleines Zahlenbeispiel veranschaulicht die Definition und begleitet uns durch den restlichen Abschnitt. Sei $P = (0.6, 0.3, 0.1)$ die wahre Verteilung und $Q = (0.2, 0.5, 0.3)$ ein Modell für drei mögliche Ergebnisse. Dann gilt

<!--en:d08211c5-->
während sich in umgekehrter Richtung $D_{\textrm{KL}}(Q\|P) = 0.365$ Nat ergeben. In Bit betragen die Werte $0.571$ und $0.527$. Sie unterscheiden sich, und der erste Wert hat eine konkrete Bedeutung. Wer von $Q$ ausgeht, stellt pro Ergebnis $0.57$ Fragen mehr als nötig, gemittelt über lange Folgen von Ergebnissen aus $P$.

<!--en:82b781df-->
Im Code berechnen wir die diskrete KL-Divergenz wie folgt. Die Konvention $0 \log 0 = 0$ setzen wir um, indem wir die Nullwahrscheinlichkeiten von $P$ wie in `entropy` ausblenden. Wir werten die Funktion für unser Beispielpaar in beiden Richtungen aus und außerdem für ein Modell $Q' = (0.5, 0.5, 0)$, das das dritte Ergebnis für unmöglich erklärt.

<!--en:443ec452-->
Besonders aufschlussreich ist die zweite Zeile. Das Modell $Q'$ weist einem Ergebnis, das in einem von zehn Fällen eintritt, die Wahrscheinlichkeit null zu. Seine KL-Divergenz ist unendlich. Tritt ein Ereignis ein, das das Modell für unmöglich hält, ist sein Überraschungswert unendlich groß.

<!--en:0f9d1ad9-->
### Die Gibbs-Ungleichung

<!--en:767a295a-->
Kann der mittlere zusätzliche Überraschungswert negativ sein? Dann wäre ein falsches Modell im Mittel weniger überrascht als jemand, der die wahre Verteilung kennt. Die Gibbs-Ungleichung :cite:`Gibbs.1902` schließt das aus. Im Mittel schneidet kein Modell besser ab als die wahre Verteilung. Da alles Weitere auf diesem Ergebnis beruht, beweisen wir es vollständig, indem wir eine konkave Funktion mitteln. Für jede konkave Funktion $f$ und jede Zufallsvariable $X$ mit endlichem Mittelwert besagt die *Jensen-Ungleichung*:

<!--en:332c8906-->
Wenn wir zuerst mitteln und dann $f$ anwenden, erhalten wir also den größeren Wert. Nimmt $X$ zwei Werte an, besagt die Ungleichung, dass die Sehne in :numref:`fig_mdl-bernoulli-entropy` unter der Kurve liegt. Ist $f$ streng konkav, gilt Gleichheit nur dann, wenn $X$ mit Wahrscheinlichkeit eins konstant ist (den allgemeinen Fall beweist :numref:`subsec_mdl-jensen`). Wir brauchen die Jensen-Ungleichung nur für eine Funktion, den streng konkaven Logarithmus. Der Logarithmus eines Mittelwerts ist mindestens so groß wie der Mittelwert der Logarithmen.

<!--en:17cce663-->
**Proposition (Gibbs-Ungleichung).** *Für beliebige Verteilungen $P$ und $Q$ auf demselben Raum gilt*

<!--en:43ed68a2-->
*mit Gleichheit genau dann, wenn $P = Q$.*

<!--en:7c669841-->
**Beweis.** Gilt $q(x) = 0$ für ein $x$ mit $p(x) > 0$, so ist die Divergenz $+\infty$ und die Behauptung folgt unmittelbar. Betrachten wir also den übrigen Fall. Wir schreiben die Divergenz als negativen Mittelwert von $\log(q(x)/p(x))$ unter $x \sim P$ und wenden die Jensen-Ungleichung auf den Logarithmus an:

<!--en:bbce3aab-->
Das Minuszeichen kehrt das „höchstens“ der Jensen-Ungleichung in ein „mindestens“ um. Den Rest des Beweises liefert die zweite Zeile. Der Erwartungswert erstreckt sich über den Träger von $P$, also über alle Ergebnisse mit $p(x) > 0$. Die Faktoren $p(x)$ kürzen sich. Übrig bleibt die gesamte $Q$-Wahrscheinlichkeit dieses Trägers. Sie beträgt höchstens $1$, ihr negativer Logarithmus ist also nichtnegativ. Für stetige Dichten verläuft der Beweis genauso, mit Integralen anstelle der Summen. Gleichheit setzt voraus, dass beide Ungleichungen mit Gleichheit erfüllt sind. Wegen der strengen Konkavität des Logarithmus gilt in der ersten nur dann Gleichheit, wenn $q(x)/p(x)$ gleich einer Konstanten $c$ ist, und zwar für $P$-fast alle $x$ (also außerhalb einer Menge mit $P$-Wahrscheinlichkeit null). Die zweite verlangt $Q(\textrm{supp}\,P) = 1$. Summieren oder integrieren wir $q = c\,p$ über den Träger von $P$, folgt zwingend $c = 1$ und damit $q = p$ (bei Dichten wiederum außerhalb einer Nullmenge). $\blacksquare$

<!--en:cd80021e-->
Der Beweis beruht auf einem einzigen Gedanken. Unter der wahren Verteilung beträgt das Verhältnis $q/p$ im Mittel höchstens $1$. Sein Logarithmus ist daher im Mittel höchstens $0$. Gleichheit gilt nur, wenn $q/p$ überhaupt nicht variiert (also $q = p$). Nur ein zutreffendes Modell verursacht keine zusätzliche Überraschung.

<!--en:47eed681-->
Die Gibbs-Ungleichung liefert sofort den ausstehenden Beweis für die obere Entropieschranke. Für jede Verteilung $P$ auf $k$ Ergebnissen und die Gleichverteilung $U$ auf derselben Menge gilt $D_{\textrm{KL}}(P\|U) = \sum_i p_i \log(k\,p_i) = \log k - H(P)$. Dieser Ausdruck ist nichtnegativ und verschwindet nur für $P = U$.

<!--en:f0fcdff1-->
Damit können wir auch zeigen, dass die Entropie durch Bedingen im Mittel nicht steigen kann. Vergleichen wir die gemeinsame Verteilung eines Paares $(X, Y)$ mit dem Produkt seiner Randverteilungen. Der Logarithmus ihres Verhältnisses lautet $\log \frac{p(x, y)}{p(x)\,p(y)} = \log p(y \mid x) - \log p(y)$. Sein Erwartungswert unter der gemeinsamen Verteilung beträgt $H(Y) - H(Y \mid X)$. Da dieser Erwartungswert eine KL-Divergenz ist, gilt

<!--en:edffd633-->
mit Gleichheit genau dann, wenn $X$ und $Y$ unabhängig sind. Im Mittel erhöht die Kenntnis von $X$ die Unsicherheit über $Y$ nie. Eine *einzelne* Beobachtung kann sie jedoch erhöhen. Angenommen, $Y$ zeigt bei $X = 0$ sicher Zahl und verhält sich bei $X = 1$ wie eine faire Münze. Dabei tritt $X = 1$ in einem von fünf Fällen ein. Insgesamt zeigt $Y$ mit Wahrscheinlichkeit $0.1$ Kopf, also gilt $H(Y) = H_b(0.1) \approx 0.325$ Nat. Beobachten wir $X = 1$, so steigt unsere Unsicherheit über $Y$ auf $\ln 2 \approx 0.693$ Nat. Die bedingte Entropie mittelt jedoch über beide Ergebnisse von $X$. Sie beträgt $H(Y \mid X) = 0.2 \cdot 0.693 \approx 0.139$ Nat, weniger als die Hälfte von $H(Y)$.

<!--en:a20658a9-->
### Kreuzentropie

<!--en:f5ee3160-->
Zum Trainieren von Klassifikatoren verwenden wir eine nahe Verwandte der KL-Divergenz. Die *Kreuzentropie* von $P$ zu $Q$ ist der mittlere Überraschungswert, wenn wir von $Q$ ausgehen und die Ergebnisse aus $P$ stammen:

<!--en:cd24a0c6-->
Im Erwartungswert können wir $\log p(x)$ addieren und wieder subtrahieren, sofern $H(P)$ endlich ist. Damit zerfällt der mittlere Überraschungswert des Modells in die Entropie der wahren Verteilung und die zusätzliche Divergenz:

<!--en:06c9ac1b-->
Die Kreuzentropie setzt sich also aus einer irreduziblen Untergrenze, der Entropie, und einem nichtnegativen Zusatzterm zusammen. Nach der Gibbs-Ungleichung gilt $\textrm{CE}(P, Q) \ge H(P)$ mit Gleichheit genau dann, wenn $Q = P$. Für unser Beispielpaar erhalten wir $\textrm{CE}(P, Q) = -(0.6 \ln 0.2 + 0.3 \ln 0.5 + 0.1 \ln 0.3) = 1.294$ Nat und $H(P) = 0.898$ Nat. Die Differenz von $0.396$ Nat ist die zuvor berechnete KL-Divergenz. Da $H(P)$ nicht von $Q$ abhängt, ist das Minimieren der Kreuzentropie bezüglich $Q$ gleichbedeutend mit dem Minimieren der KL-Divergenz. Genau das leistet Maximum-Likelihood bei diskreten Daten. Die mittlere negative Log-Likelihood ist die Kreuzentropie von der empirischen Datenverteilung zum Modell (:numref:`subsec_mdl-nll-crossentropy`).

<!--en:34d1a24b-->
:numref:`fig_mdl-code-length-bars` zeigt diese Zerlegung im Fragespiel für die dyadische Quelle $P$ und das gleichverteilte Modell $U = (\tfrac14, \tfrac14, \tfrac14, \tfrac14)$. Mit Fragen, die auf $P$ zugeschnitten sind, brauchen wir im Mittel $H_2(P) = 1.75$ Fragen. Die Strategie für die Gleichverteilung stellt immer zwei feste Fragen, braucht also $\textrm{CE}_2(P, U) = 2$ Fragen. Damit verschwenden wir $D_{\textrm{KL}}(P\|U) = 0.25$ Bit. Da die Fragestrategie einen Binärcode darstellt, geben dieselben Zahlen die mittleren Codewortlängen in Bit pro Symbol an. (Bei diesem speziellen Paar ergibt die umgekehrte Divergenz $D_{\textrm{KL}}(U\|P)$ zufällig ebenfalls $0.25$ Bit. Deshalb haben wir die Asymmetrie der KL-Divergenz zuvor an einem anderen Paar gezeigt.)

<!--en:a0762d01-->
![Die Zerlegung :eqref:`eq_mdl-ce_decomp` für die dyadische Quelle $P = (\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}, \tfrac{1}{8})$ und das gleichverteilte Modell $U$ in Bit. Mit Fragen, die auf $P$ zugeschnitten sind, brauchen wir im Mittel $H_2(P) = 1.75$ Fragen. Mit zwei festen Fragen brauchen wir $\textrm{CE}_2(P, U) = 2.0$ und verschwenden damit $D_{\textrm{KL}}(P\|U) = 0.25$ Bit.](../img/mdl-it-code-length-bars.svg)

<!--en:d7a5f8ef-->
### Die Kettenregel für die KL-Divergenz

<!--en:0e352f05-->
Für ein Paar zerfällt die Entropie in zwei Teile. Die Überraschung über $(X, Y)$ besteht aus der Überraschung über $X$ und der verbleibenden Überraschung über $Y$ (:numref:`subsec_mdl-joint-conditional-entropy`). Die KL-Divergenz zerfällt auf dieselbe Weise und aus demselben Grund. Seien $P_{X,Y}$ und $Q_{X,Y}$ zwei gemeinsame Verteilungen mit den Zerlegungen $p(x, y) = p(x)\,p(y \mid x)$ und $q(x, y) = q(x)\,q(y \mid x)$. Der Logarithmus des Verhältnisses $p(x, y)/q(x, y)$ ist dann eine Summe zweier Terme. Bilden wir den Erwartungswert unter $P_{X,Y}$, erhalten wir die *Kettenregel* für die KL-Divergenz:

<!--en:e75ed93a-->
Ein fehlerhaftes Modell des Paares bezahlt für seinen Fehler bei $X$ und anschließend im Mittel für seinen Fehler bei $Y$, sobald $X$ bekannt ist. Dabei mitteln wir über die wahre Verteilung $P_X$ und nicht über das Modell. Die Identität lässt sich in einer Zeile herleiten. Drei ihrer Folgerungen verwenden wir im weiteren Verlauf des Buches.

<!--en:d426c390-->
Erstens addiert sich die KL-Divergenz über unabhängige Koordinaten. Sind $X$ und $Y$ sowohl unter $P$ als auch unter $Q$ unabhängig, hängt der bedingte Term nicht mehr von $x$ ab. Die Divergenz des Paares ist dann die Summe der beiden einzelnen Divergenzen. Bei zehn unabhängigen Ziehungen aus unserem Beispiel steigt die Divergenz von $P$ zu $Q$ auf $10 \cdot 0.396 = 3.96$ Nat. Daher lassen sich zwei Verteilungen anhand langer Stichprobenfolgen viel besser unterscheiden als anhand kurzer. Zweitens ist die Divergenz zweier gemeinsamer Verteilungen mindestens so groß wie die Divergenz ihrer Randverteilungen. Denn der bedingte Term ist nach der Gibbs-Ungleichung nichtnegativ. Lassen wir eine Variable weg, können die Verteilungen dadurch nur schwerer unterscheidbar werden. Das nutzen latente Variablenmodelle: Die berechenbare Zielfunktion ist bis auf eine Konstante die gemeinsame Divergenz über Daten und latente Variablen. Sie übersteigt die Divergenz über die Daten allein um den bedingten Term. Dieser entspricht der Lücke zwischen der Log-Evidenz und der unteren Evidenzschranke (:numref:`sec_mdl-latent-em-elbo`). Drittens können wir die Regel wiederholt auf eine Sequenz anwenden und erhalten

<!--en:7bf29762-->
Der zusätzliche Verlust eines autoregressiven Modells auf einer ganzen Sequenz ist also die Summe der zusätzlichen Verluste der einzelnen Schritte. Jedes nächste Token gut vorherzusagen, ist daher gleichbedeutend damit, die gesamte Sequenz gut zu modellieren. In :numref:`subsec_mdl-perplexity` liefert diese Zerlegung eine untere Schranke für den Verlust jedes Sprachmodells.

<!--en:996ce8b6-->
## Der Klassifikationsverlust

<!--en:fce7da77-->
Bei der Klassifikation mit $k$ Klassen ist die Zielverteilung für ein einzelnes Beispiel eine One-Hot-Verteilung. Ihre gesamte Masse liegt auf dem erfassten Label $y$, während das Modell eine vorhergesagte Verteilung $\mathbf{q} = (q_1, \ldots, q_k)$ ausgibt. Die Kreuzentropie :eqref:`eq_mdl-ce_def` zwischen beiden reduziert sich auf einen einzigen Term, $-\log q_y$. Er gibt an, wie sehr das beobachtete Label das Modell überrascht. Mitteln wir über $n$ Beispiele mit Labels $y_i$ und Vorhersagen $\mathbf{q}_i$, erhalten wir den *Kreuzentropieverlust*

<!--en:4f10b806-->
in Nat. Deep-Learning-Frameworks berechnen diesen Verlust direkt aus den *Logits* $\mathbf{z}$. Das sind die unnormierten Bewertungen, aus denen die Softmax-Funktion die Verteilung $\mathbf{q}$ berechnet. Intern verwenden die Frameworks ein numerisch stabiles Log-Softmax. Denn bei der direkten Berechnung der Wahrscheinlichkeiten droht für große Logits ein Überlauf, für sehr negative ein Unterlauf auf null und damit $\log 0$ (:numref:`subsec_softmax-implementation-revisited`, die stabile Form leitet :numref:`subsec_mdl-stable-softmax` her). Im Folgenden untersuchen wir, was dieser Verlust über Rauschen in den Daten, die Kalibrierung von Wahrscheinlichkeiten und weiche Zielverteilungen aussagt.

<!--en:0b39acbd-->
### Der irreduzible Verlust

<!--en:ccb51b6b-->
Zu Beginn des Abschnitts haben wir gefragt, wie klein der Verlust werden kann, wenn einige Labels falsch sind. Auf einer festen Trainingsmenge sagt die Antwort wenig aus. Ein hinreichend großes Netz kann den Datensatz auswendig lernen, jedes erfasste Label einschließlich der falschen reproduzieren und den Trainingsverlust gegen null treiben. Aussagekräftig wird die Frage erst bei neuen Beispielen, die das Modell noch nie gesehen hat. Dort beschreibt das One-Hot-Ziel die Wahrheit nur unvollständig. Bei realen Daten ist das Label selten eine deterministische Funktion der Eingabe. Dasselbe Bild kann wegen echter Mehrdeutigkeit oder durch Fehler bei der Annotation verschiedene Labels erhalten. Für eine Eingabe $\mathbf{x}$ beschreibt die bedingte Verteilung, aus der ihr Label gezogen wird, den wahren Sachverhalt. Das erfasste Label ist eine einzelne Ziehung daraus.

<!--en:b8998ea3-->
Eingabe $\mathbf{x}$ und Label $y \in \{1, \ldots, k\}$ seien gemeinsam verteilt. Dabei sei $p(\cdot \mid \mathbf{x})$ die wahre bedingte Verteilung des Labels und $q(\cdot \mid \mathbf{x})$ die Vorhersage eines beliebigen Modells. Mit wachsender Zahl neuer Beispiele konvergiert ihr mittlerer Verlust gegen den Erwartungswert über $(\mathbf{x}, y)$. Diesen nennen wir *Populationsverlust*. Kein Modell kann ihn unter die mittlere Entropie des Labels bei gegebener Eingabe senken.

<!--en:ab014162-->
**Proposition (irreduzibler Klassifikationsverlust).** *Für den Kreuzentropieverlust eines beliebigen Modells auf der Population gilt*

<!--en:5c9a8584-->
*mit Gleichheit genau dann, wenn $q(\cdot \mid \mathbf{x}) = p(\cdot \mid
\mathbf{x})$ für fast jedes $\mathbf{x}$ gilt (also für alle Eingaben außerhalb einer Menge mit Wahrscheinlichkeit null).*

<!--en:18283561-->
**Beweis.** Bedingen wir auf $\mathbf{x}$. Der innere Erwartungswert über $y$ ist die Kreuzentropie $\textrm{CE}(p(\cdot \mid \mathbf{x}), q(\cdot \mid \mathbf{x}))$. Damit gilt die Zerlegung :eqref:`eq_mdl-ce_decomp` punktweise für jedes $\mathbf{x}$ (alle Entropien sind endlich und betragen höchstens $\log k$). Der äußere Erwartungswert über $\mathbf{x}$ liefert die Identität. Nach der Gibbs-Ungleichung ist jeder KL-Term nichtnegativ. Ein Mittelwert nichtnegativer Terme verschwindet genau dann, wenn fast alle Terme null sind. $\blacksquare$

<!--en:ea089feb-->
Diese Untergrenze ist die bedingte Entropie des Labels bei gegebener Eingabe (:numref:`subsec_mdl-joint-conditional-entropy`). Sie ist eine Eigenschaft der Daten selbst. Weder eine andere Architektur noch zusätzliche Trainingsdaten oder längeres Training können sie senken. Behalten Sie vor allem dieses Ergebnis im Kopf, wenn Sie eine Verlustkurve lesen. Drei Fälle verdeutlichen seine Bedeutung:

<!--en:e39c5dfb-->
* Sagt ein Modell die Gleichverteilung vorher, beträgt seine Kreuzentropie für jede Eingabe $\ln k$, unabhängig von den Labels. Das ergibt die bekannten $2.30$ Nat für $k = 10$.

<!--en:1941d01f-->
* Ist das Label eine deterministische Funktion der Eingabe, so ist jedes $H(p(\cdot \mid \mathbf{x}))$ null. Ein perfektes Modell erreicht dann einen Verlust von $0$.

<!--en:c3aabe82-->
* Bei symmetrischem Labelrauschen legt die Eingabe die wahre Klasse fest. Das erfasste Label stimmt mit Wahrscheinlichkeit $1 - \rho$ mit ihr überein und nimmt jede der übrigen $k - 1$ Klassen mit Wahrscheinlichkeit $\rho/(k-1)$ an. Bei jeder Eingabe beträgt die bedingte Entropie $H = -(1-\rho)\log(1-\rho) - \rho\log\tfrac{\rho}{k-1} = H_b(\rho) + \rho\log(k-1)$. Für $\rho = 0.1$ und $k = 10$ liegt diese Untergrenze bei $0.325 + 0.220 = 0.545$ Nat. Solange $1 - \rho > \rho/(k-1)$ gilt, bleibt die Vorhersage der wahren Klasse optimal. Die höchstmögliche Genauigkeit beträgt daher $1 - \rho$ (hier $90\%$).

<!--en:5e6488a0-->
Bevor Sie die nächste Zelle ausführen, schätzen Sie den Verlust eines selbstsicheren Modells, das der wahren Klasse bei diesem Rauschen die Wahrscheinlichkeit $0.99$ zuweist. Seine Vorhersage stimmt in neun von zehn Fällen, und es ist sich jedes Mal fast sicher. Der Code vergleicht es mit der Gleichverteilung und mit dem *Bayes-Prädiktor*, der die wahre bedingte Verteilung $q = p$ ausgibt.

<!--en:c69a5e60-->
Das übermäßig selbstsichere Modell erleidet einen Verlust von $0.689$ und liegt damit deutlich über der Untergrenze. Jedes verfälschte Label überrascht es. Sein hohes Vertrauen kostet bei dem Zehntel verfälschter Labels weit mehr, als es bei den übrigen gewinnt. Nur der Bayes-Prädiktor erreicht die Untergrenze.

<!--en:afb0a593-->
Nun können wir die Eingangsfrage beantworten: Ist ein Endverlust von $0.6$ gut? Bei fehlerfreien Labels mit Untergrenze null lässt $0.6$ noch viel Raum für ein besseres Modell. Bei den verrauschten Labels liegt derselbe Wert hingegen nur $0.06$ Nat über der Untergrenze. Es bleibt kaum Spielraum. Ein Plateau an der Untergrenze ergibt sich aus den Daten. Nur ein Plateau oberhalb der Untergrenze weist auf Grenzen des Modells oder seines Trainings hin. All dies gilt für neue Daten. Ein Verlust *unterhalb* der Untergrenze auf einem zurückgehaltenen Teil der Daten lässt sich nicht durch ein besseres Modell erklären. In Aufgabe :exref:`ex_mdl-information-theory-label-noise-floor` untersuchen Sie, worauf er stattdessen hinweist.

<!--en:2439503e-->
Bisher haben wir die Verlustuntergrenze für eine bestimmte Strafe ($-\log q_y$), unveränderte One-Hot-Labels und abzählbare Ergebnisse hergeleitet. Nun lockern wir diese drei Annahmen nacheinander. Wir betrachten andere Strafen, weichere Zielverteilungen und stetige reellwertige Ergebnisse.

<!--en:b6613594-->
### Echte Bewertungsregeln

<!--en:0113616e-->
Nur die wahren bedingten Wahrscheinlichkeiten erreichen die Untergrenze des Verlusts. Genau diese Eigenschaft brauchen wir, wenn wir die Ausgaben eines Klassifikators als Wahrscheinlichkeiten deuten wollen. Die Entscheidungstheorie gibt ihr einen Namen. Eine *Bewertungsregel* (engl. scoring rule) legt die Strafe $S(\mathbf{q}, y)$ dafür fest, dass wir den Wahrscheinlichkeitsvektor $\mathbf{q}$ angeben, wenn tatsächlich Klasse $y$ eintritt. Die Regel heißt *echt* (engl. proper), wenn es optimal ist, die wahre Verteilung anzugeben. Folgen die Labels der Verteilung $\mathbf{p}$, so minimiert die Angabe der wahren Verteilung die erwartete Strafe $\mathbb{E}_{y \sim \mathbf{p}}[S(\mathbf{q}, y)]$, und zwar bei $\mathbf{q} = \mathbf{p}$. Die Regel heißt *streng echt*, wenn diese Angabe der einzige Minimierer ist. Der Kreuzentropieverlust ist der *logarithmische Score* $S(\mathbf{q}, y) = -\log q_y$. Die Gibbs-Ungleichung zeigt, dass er streng echt ist: Seine erwartete Strafe beträgt $\textrm{CE}(\mathbf{p}, \mathbf{q})$ und hat ihr einziges Minimum bei $\mathbf{q} = \mathbf{p}$.

<!--en:5459c1ec-->
Enthält die Modellklasse die wahre Verteilung, so gibt ein mit Kreuzentropie trainierter Klassifikator am Populationsoptimum folglich die wahren bedingten Wahrscheinlichkeiten aus. Insbesondere sind seine Vorhersagen *kalibriert*. Betrachten wir alle Eingaben, denen das Modell für eine bestimmte Klasse die Wahrscheinlichkeit $0.8$ zuweist, so tritt diese Klasse tatsächlich mit relativer Häufigkeit $80\%$ auf. (In der Praxis kann ein trainiertes Netz wegen einer endlichen Trainingsmenge oder einer eingeschränkten Modellklasse dennoch schlecht kalibriert sein.)

<!--en:6cf3c037-->
Der logarithmische Score ist nicht die einzige streng echte Bewertungsregel. Auch der *Brier-Score* $S(\mathbf{q}, y) = \sum_j (q_j - \mathbf{1}(j = y))^2$ ist streng echt. Anders als der logarithmische Score bestraft er selbstsichere Fehlvorhersagen nur begrenzt, selbst wenn $-\log q_y$ divergiert :cite:`Gneiting.Raftery.2007`. In Aufgabe :exref:`ex_mdl-information-theory-scoring-rule-is-strictly` berechnen Sie seine zusätzliche Strafe und untersuchen eine scheinbar sinnvolle Regel, die sich als nicht echt erweist.

<!--en:b3b9e624-->
### Label-Smoothing

<!--en:d5d86cc2-->
Bisher haben wir das One-Hot-Label als Trainingsziel verwendet. Eine weichere Zielverteilung verändert sowohl die Anforderungen an die Logits als auch die Untergrenze des Verlusts. Dafür gibt es einen praktischen Grund. One-Hot-Ziele verlangen etwas, das eine Softmax-Schicht nicht leisten kann. Damit die wahre Klasse die Wahrscheinlichkeit $1$ erhält, müsste ihr Logit die übrigen um unendlich viel übersteigen. Der Verlust hat daher keinen endlichen Minimierer. Der Gradientenabstieg treibt die Logits während des gesamten Trainings immer weiter auseinander. *Label-Smoothing* :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` behebt das Problem. Dazu mischen wir das One-Hot-Ziel $\mathbf{e}_y$ mit einem Anteil $\epsilon$ der Gleichverteilung über alle $k$ Klassen:

<!--en:307f1981-->
Bei dieser Zielverteilung erhält die wahre Klasse die Wahrscheinlichkeit $1 - \epsilon +
\epsilon/k$ und jede andere Klasse $\epsilon/k$. Als Trainingsziel dient die gewöhnliche Kreuzentropie $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ bezüglich der geglätteten Zielverteilung (derselbe Verlust wie in :numref:`sec_training_recipes`). Ihr Minimum liegt bei $\mathbf{q} = \mathbf{p}^\epsilon$. Die Softmax-Funktion erreicht es mit einem endlichen Abstand zwischen dem Logit der wahren Klasse und jedem konkurrierenden Logit:

<!--en:0285be0d-->
Für $k = 10$ Klassen und $\epsilon = 0.1$ beträgt dieser optimale Logit-Abstand $\ln 91 \approx 4.51$. Der Verlust am Optimum entspricht der Entropie der geglätteten Zielverteilung, also $H(\mathbf{p}^\epsilon) = 0.500$ Nat.

<!--en:ccc7fd3d-->
Wie schnell driften die Logits ohne Glättung auseinander? Betrachten wir ein einzelnes Beispiel. Seine Logits starten bei null und ändern sich durch Gradientenabstieg mit Schrittweite $1$. Der Gradient der Kreuzentropie bezüglich der Logits lautet $\mathbf{q} - \mathbf{p}^\epsilon$. Bei einem One-Hot-Ziel bleiben die Logits aller falschen Klassen gleich. Mit jedem Schritt wächst der Abstand $g$ zwischen der wahren Klasse und den übrigen um $(1 - q_y) + q_j = k/(e^g + k - 1)$, näherungsweise also um $k e^{-g}$, sobald $g$ groß ist. Erhöhen wir $g$ um $k e^{-g}$, so multipliziert das $e^g$ mit $e^{k e^{-g}} \approx 1 + k e^{-g}$. Bei jedem Schritt kommt also ungefähr $k$ zu $e^g$ hinzu. Nach $t$ Schritten erwarten wir deshalb $e^g \approx kt$ oder $g \approx \ln(kt)$. Der Abstand wächst unbegrenzt, aber nur logarithmisch. Zehnmal so viele Schritte vergrößern ihn lediglich um $\ln 10 \approx 2.30$. Nach $10{,}000$ Schritten erwarten wir $g \approx \ln 10^5 \approx 11.51$. Die nächste Zelle führt den Gradientenabstieg einmal ohne und einmal mit Glättung aus.

<!--en:5b2a77f6-->
Mit One-Hot-Zielen erreicht der Logit-Abstand die Werte $4.29$, $6.87$, $9.21$ und $11.51$. Unsere Näherung $\ln(kt)$ sagt $4.61$, $6.91$, $9.21$ und $11.51$ voraus. Anfangs liegt der tatsächliche Abstand etwas unter der Schätzung, weil $k e^{-g}$ noch eine grobe Näherung ist. Ab $1{,}000$ Schritten stimmen die Werte auf zwei Nachkommastellen überein. Währenddessen sinkt der Verlust immer langsamer, ohne je null zu erreichen. In einem neuronalen Netz, das groß genug ist, um die Trainingsmenge genau zu reproduzieren, zeigt sich diese unbegrenzte Drift in stetig wachsenden Gewichten. Die vorhergesagten Wahrscheinlichkeiten liegen für jedes Trainingsbeispiel nahe $1$, auch für falsch beschriftete Beispiele. Bei $\epsilon = 0.1$ hingegen erreicht der Abstand den Wert $4.51$ schon bei Schritt $100$ und bleibt dort, während sich der Verlust bei $0.500$ einpendelt.

<!--en:977615e9-->
Das Trainingsziel hat nun ein endliches Optimum. Ein mit Glättung trainiertes Modell müssen wir allerdings mit Bedacht interpretieren. Selbst wenn es die geglättete Zielverteilung fehlerfrei vorhersagt, erreicht sein Trainingsverlust ein Plateau bei $H(\mathbf{p}^\epsilon)$ statt bei null. Seine Ausgaben entsprechen zudem nicht mehr den wahren bedingten Wahrscheinlichkeiten. Keine Klasse kann eine höhere Wahrscheinlichkeit als $1 - \epsilon + \epsilon/k$ erhalten (hier $0.91$). In der Praxis ist $\epsilon = 0.1$ die übliche Wahl, auch beim Training in :numref:`sec_training_recipes`. Aufgabe :exref:`ex_mdl-information-theory-label-smoothing` untersucht, wie sich Label-Smoothing auf die Logits der falschen Klassen auswirkt.

<!--en:19e6ffa3-->
### Wissensdestillation

<!--en:29956a40-->
Weiche Zielverteilungen kann auch ein anderes trainiertes Modell liefern. Bei der *Wissensdestillation* (engl. knowledge distillation) :cite:`Hinton.Vinyals.Dean.2015` lernt ein kompaktes Netz als *Schüler*, ein größeres Netz, den *Lehrer*, nachzuahmen. In :numref:`sec_lm_distillation` kommt dieses Verfahren bei Sprachmodellen zum Einsatz. Der Schüler lernt anhand der gesamten Wahrscheinlichkeitsverteilung des Lehrers über die Klassen statt anhand eines One-Hot-Labels. Eine handgeschriebene „2“ ähnelt einer „7“ viel stärker als einer „0“. Ein One-Hot-Label verwirft diese Information, doch die Logits des Lehrers für die falschen Klassen enthalten sie. (Aufgabe :exref:`ex_mdl-information-theory-label-smoothing` fragt, warum ein mit Label-Smoothing trainierter Lehrer weniger von dieser Information weitergibt.)

<!--en:052d96e4-->
Wie bei jeder festen Zielverteilung erreicht der Verlust sein Minimum, wenn der Schüler die Zielverteilung selbst reproduziert. Die Zerlegung :eqref:`eq_mdl-ce_decomp` liefert hier also keine neue Erkenntnis. Erklärungsbedürftig ist die Skalierung des Verlusts durch einen auf den ersten Blick willkürlichen Faktor.

<!--en:c08a9951-->
Die Softmax-Ausgabe eines gut trainierten Lehrers ist fast ein One-Hot-Vektor. Die Wahrscheinlichkeiten der falschen Klassen sind daher meist so klein, dass sie das Lernen des Schülers kaum beeinflussen. Wir machen deshalb beide Verteilungen zunächst *weicher*. Dazu teilen wir die Logits vor Anwendung der Softmax-Funktion durch eine *Temperatur* $\tau > 0$:

<!--en:a6a1e311-->
Mit $\tau > 1$ verringern sich die Abstände zwischen den skalierten Logits, und Wahrscheinlichkeitsmasse verschiebt sich zu den falschen Klassen. Der Destillationsverlust ist die KL-Divergenz zwischen den weicheren Verteilungen, multipliziert mit $\tau^2$:

<!--en:7577611f-->
Dieser Verlust verschwindet genau dann, wenn der Schüler die weiche Verteilung des Lehrers trifft. Um den Faktor $\tau^2$ zu verstehen, leiten wir den unskalierten KL-Term nach einem Logit des Schülers ab. Nur der Kreuzentropieterm $-\sum_j p_j^{(\tau)} \log q_j^{(\tau)}$ hängt vom Schüler ab. Bei Temperatur $\tau$ ergibt die Ableitung wieder den bekannten Ausdruck „Softmax minus Zielverteilung“, skaliert mit $1/\tau$:

<!--en:28fdebbc-->
Jedes Logit des Schülers erhält seinen Anstoß also aus der Differenz der weicheren Wahrscheinlichkeiten von Schüler und Lehrer für diese Klasse, geteilt durch $\tau$.

<!--en:35d18054-->
Bei hohen Temperaturen schrumpft auch diese Wahrscheinlichkeitsdifferenz. Ziehen wir den mittleren Logit-Wert $\bar z$ ab, bleibt der Wert der Softmax-Funktion gleich. Für große $\tau$ können wir $e^{(z_j - \bar z)/\tau} \approx 1 + (z_j - \bar z)/\tau$ bis zur ersten Ordnung entwickeln. Da sich die zentrierten Abweichungen $z_j - \bar z$ zu null summieren, beträgt der Softmax-Nenner ungefähr $k$. Das ergibt $\mathrm{softmax}(\mathbf{z}/\tau)_j \approx \tfrac{1}{k}\big(1 + (z_j - \bar z)/\tau\big)$. Folglich schrumpft $q_j^{(\tau)} - p_j^{(\tau)}$ wie $1/\tau$ und der unskalierte Gradient wie $1/\tau^2$:

<!--en:b5e7b760-->
Multiplizieren wir den Verlust mit $\tau^2$, gleichen wir diesen Rückgang aus. Der Betrag des Gradienten bleibt dann bei veränderlichem $\tau$ ungefähr konstant. Das ist in der Praxis wichtig, weil der Destillationsverlust selten allein zum Einsatz kommt. Meist addieren wir ihn mit einem festen Gewicht zum üblichen Kreuzentropieverlust für die harten Labels. Ohne den Faktor $\tau^2$ würde eine Anpassung von $\tau$ unbeabsichtigt das relative Gewicht der beiden Verluste verändern. Mit dem Faktor steuert die Temperatur nur, wie deutlich der Schüler die kleinen Wahrscheinlichkeiten des Lehrers für die falschen Klassen sieht.

<!--en:de80765b-->
Prüfen wir diese Skalierung numerisch mit den Lehrer-Logits $(5, 2, -1)$ und den Schüler-Logits $(3, 3, 0)$. Beide Vektoren haben den Mittelwert $2$. Daher sollte das $\tau^2$-Fache des Gradienten gegen $\tfrac13\big((1, 1, -2) - (3, 0, -3)\big) = (-0.667, 0.333, 0.333)$ mit der Norm $0.816$ konvergieren. In der folgenden Zelle berechnen wir durch automatisches Differenzieren den Gradienten des unskalierten KL-Terms für fünf Temperaturen. Wir vergleichen seine Norm vor und nach der Multiplikation mit $\tau^2$.

<!--en:b9baa118-->
Ohne den Faktor $\tau^2$ bricht die Gradientennorm ein. Sie sinkt von $0.64$ bei $\tau = 1$ auf $0.0021$ bei $\tau = 20$, also ungefähr auf ein Dreihundertstel. Die letzte Verdopplung von $\tau$ verringert die Norm etwa um den Faktor vier, wie es $1/\tau^2$ vorhersagt. Mit dem Faktor bleibt die Norm zwischen $0.64$ und $0.92$. Sie überschreitet den asymptotischen Grenzwert $0.816$ leicht, erreicht ihr Maximum nahe $\tau = 5$ und nähert sich dem Grenzwert dann langsam mit einer Rate der Ordnung $1/\tau$ (noch $0.854$ bei $\tau = 20$). Bei $\tau = 1$ reduziert sich der Verlust zudem auf die gewöhnliche KL-Divergenz. Für einen Lehrer mit One-Hot-Ausgabe und damit Entropie null ergibt sich der übliche Kreuzentropieverlust. Das gewöhnliche überwachte Training ist also ein Spezialfall.

<!--en:113ca48a-->
## Entropie und KL-Divergenz für Dichten

<!--en:ca753667-->
Bisher waren die möglichen Ergebnisse diskret und abzählbar. Ein Regressionsziel ist hingegen eine reelle Zahl, die wir durch eine Wahrscheinlichkeitsdichte beschreiben. An die Stelle von Summen treten also Integrale. Die KL-Divergenz behält dabei ihre Eigenschaften, die Entropie nur zum Teil. Die Entropie einer stetigen Variable kann negativ sein und hängt von der Maßeinheit ab. Daher kann die negative Log-Likelihood als Verlust eines stetigen Modells unter null fallen. Ihre Untergrenze verschiebt sich, wenn wir die Daten umskalieren. Der Abstand zur Untergrenze behält seine absolute Bedeutung und damit auch die Verlustdifferenz zwischen zwei Modellen, die wir auf denselben Daten auswerten. (Beim ersten Lesen können Sie die folgenden drei Unterabschnitte überspringen und beim letzten Absatz dieses Abschnitts weiterlesen.)

<!--en:b3cc656e-->
### Differentielle Entropie

<!--en:e38956d6-->
Sei $X$ eine stetige Zufallsvariable mit Dichte $p$. Ersetzen wir die Summe in der diskreten Entropie durch ein Integral, erhalten wir die *differentielle Entropie*:

<!--en:88a68453-->
Das ist die erwartete negative Log-Dichte. Dichtewerte unterscheiden sich von Wahrscheinlichkeiten: Sie haben die Einheit $1/x$ und können durchaus größer als $1$ sein. Mit dem Kleinbuchstaben $h$ unterscheiden wir die differentielle Entropie von der diskreten Entropie $H$, denn beide verhalten sich sehr unterschiedlich.

<!--en:e85bb139-->
Schon das einfachste Beispiel zeigt den Grund. Die Dichte der Gleichverteilung auf $[0, a]$ beträgt im gesamten Intervall $1/a$. Daher gilt $h = -\int_0^a \tfrac1a \log \tfrac1a \,dx = \log a$. Für $a = \tfrac12$ ergibt das $-\ln 2 \approx -0.693$ Nat, eine negative Entropie, die bei einer diskreten Verteilung niemals vorkommt. Mehr noch: Messen wir dieselbe Länge in Zentimetern statt in Metern, multipliziert sich $a$ mit $100$. Dadurch kommen $\ln 100 \approx 4.61$ Nat zu $h$ hinzu, obwohl sich an der zugrunde liegenden Unsicherheit nichts geändert hat.

<!--en:463498b7-->
Eine solche Verschiebung ergibt sich bei jeder glatten Variablentransformation. Ist $y = g(x)$ invertierbar und hat die Ableitung $g'$, so beträgt die Dichte von $Y$ an der Stelle $g(x)$ gerade $p_X(x)/\lvert g'(x)\rvert$. Daher gilt

<!--en:16e0f1db-->
Für einen Zufallsvektor $\mathbf{x} \in \mathbb{R}^n$ behalten sowohl die Definition als auch die Transformationsregel ihre Form. Wir integrieren über $\mathbb{R}^n$ und verwenden die Jacobi-Determinante $\lvert\det \partial g/\partial \mathbf{x}\rvert$ anstelle von $\lvert g'\rvert$. Eine Streckung erhöht die differentielle Entropie einer Variable. Durch Stauchung lässt sich die Entropie beliebig weit senken. Wie wir in :numref:`subsec_mdl-gaussian-max-entropy` herleiten, hat eine Gauß-Verteilung mit Standardabweichung $\sigma$ die differentielle Entropie $h = \tfrac12 \log(2\pi e \sigma^2)$. Diese wird negativ, sobald $\sigma < (2\pi e)^{-1/2} \approx 0.24$ gilt.

<!--en:e477902f-->
Die KL-Divergenz hängt glücklicherweise nicht in dieser Weise von der Maßeinheit ab. Wenden wir dieselbe Variablentransformation auf die Datendichte $p$ und die Modelldichte $q$ an, werden beide durch denselben Jacobi-Faktor geteilt. Im Verhältnis $p/q$ kürzt er sich. Die KL-Divergenz ist also invariant unter invertierbaren Umparametrisierungen. Betrachten wir etwa die konstanten Dichten $p_1$ auf $[0, a_1]$ und $p_2$ auf $[0, a_2]$ mit $a_1 \le a_2$. Dann gilt $\int p_1 \log(p_1/p_2)\,dx = \log(a_2/a_1)$. Für $a_1 = \tfrac12$ und $a_2 = 1$ ergibt das $\ln 2 \approx 0.693$ Nat, ob wir in Metern oder in Zentimetern messen. Die Kreuzentropie $-\mathbb{E}_P[\log q]$ beim Training eines stetigen Modells verschiebt sich um dieselbe Konstante wie $h$. Da diese Konstante nicht vom Modell abhängt, bleiben die Gradienten und der Minimierer gleich. Ist $h(p)$ endlich, gilt weiterhin die Zerlegung :eqref:`eq_mdl-ce_decomp` mit $h$ anstelle von $H$. Maximum-Likelihood für Dichten minimiert also auf Populationsebene $D_{\textrm{KL}}(p\|q_{\boldsymbol\theta})$.

<!--en:0de1c3a5-->
Die differentielle Entropie ergibt sich auch nicht unmittelbar als Grenzwert der diskreten Entropie bei immer feinerer Unterteilung. Zerlegen wir die reelle Achse in Zellen der Breite $\Delta$. Dabei halte $X_\Delta$ fest, in welcher Zelle $X$ liegt. Für einen Punkt $x_i$ in der Zelle $i$ gilt $P(X_\Delta = i) \approx p(x_i)\Delta$ und damit

<!--en:995c2b26-->
sofern $p$ so regulär ist, dass die Riemann-Summen gegen die Integrale konvergieren (etwa wenn $p$ stetig ist und sowohl $p$ als auch $p \log p$ Riemann-integrierbar sind). Die diskrete Entropie $H(X_\Delta)$ selbst divergiert wie $\log(1/\Delta)$, denn ein feineres Gitter erfordert immer mehr Ja-Nein-Fragen. Die differentielle Entropie $h(X)$ ist der endliche Rest, der nach Abzug dieses Wachstums bleibt. Bei einer Gleichverteilung auf $[0, 1]$ mit $\Delta = \tfrac18$ sind alle acht Zellen gleich wahrscheinlich. Wir erhalten $H_2(X_\Delta) = 3$ Bit und $\log_2 \Delta = -3$, zusammen also $0 = h$. In Aufgabe :exref:`ex_mdl-information-theory-quantized-entropy` beweisen Sie diesen Grenzwert und prüfen ihn numerisch an einer Gauß-Verteilung.

<!--en:772dd800-->
### KL-Divergenz zwischen Gauß-Verteilungen

<!--en:29339f9e-->
Wenn ein Regressionsmodell neben einer reellen Zahl auch deren Unsicherheit vorhersagt, gibt es meist eine Gauß-Verteilung aus, also einen Mittelwert und eine Varianz. Um ein solches Modell mit den Daten oder zwei gaußsche Modelle miteinander zu vergleichen, brauchen wir die KL-Divergenz zwischen zwei Gauß-Verteilungen. Sie hat eine einfache geschlossene Form. Für $P = \mathcal{N}(\mu_1, \sigma_1^2)$ und $Q = \mathcal{N}(\mu_2, \sigma_2^2)$ gilt

<!--en:44fe2e47-->
Die Formel verbindet den Logarithmus des Verhältnisses der Standardabweichungen mit der Varianz von $P$ und der quadrierten Differenz der Mittelwerte, beide bezogen auf die Modellvarianz $\sigma_2^2$. Wie die Gibbs-Ungleichung garantiert, ist die Divergenz nichtnegativ und verschwindet nur, wenn $\mu_1 = \mu_2$ und $\sigma_1 = \sigma_2$ gelten. Ihre Asymmetrie ist direkt erkennbar, denn $\sigma_1$ und $\sigma_2$ gehen unterschiedlich in die Formel ein. In Aufgabe :exref:`ex_mdl-information-theory-gaussian-kl` leiten Sie den Ausdruck direkt aus der Definition her.

<!--en:1472a758-->
Beginnen wir mit gleichen Varianzen und um eins verschobenen Mittelwerten, also $P = \mathcal{N}(0, 1)$ und $Q = \mathcal{N}(1, 1)$. Die Formel ergibt in beiden Richtungen $(\mu_1 - \mu_2)^2/(2\sigma^2) = \tfrac12$ Nat. Betrachten wir nun verschiedene Varianzen: Für $P = \mathcal{N}(0, 1)$ und $Q = \mathcal{N}(0, 4)$ gilt $\sigma_2 = 2$. Einsetzen liefert $D_{\textrm{KL}}(P\|Q) = \ln 2 + \tfrac18 - \tfrac12 \approx 0.318$ Nat. In umgekehrter Richtung erhalten wir $D_{\textrm{KL}}(Q\|P) = -\ln 2 + 2 - \tfrac12 \approx 0.807$ Nat, mehr als das $2.5$-Fache.

<!--en:91cb02d2-->
:numref:`fig_mdl-kl-integrand` zeigt, woher die Divergenz jeweils stammt. In beiden Richtungen wechselt der Integrand bei $|x| = (8 \ln 2/3)^{1/2} \approx 1.36$ das Vorzeichen, am Schnittpunkt der Dichten auf jeder Seite. In Teilbild (a) erklärt das breite Modell schmal verteilte Daten. Im Zentrum ist sein Überraschungswert zu hoch ($|x| < 1.36$ mit Beitrag $+0.424$), in den Randbereichen etwas niedriger als unter der wahren Verteilung (Beitrag $-0.106$). In Teilbild (b) muss das schmale Modell breit verteilte Daten erklären. Das Zentrum trägt $-0.240$ bei. In den Randbereichen weist das schmale Modell tatsächlich eintretenden Ergebnissen nur winzige Wahrscheinlichkeiten zu. Sie tragen daher beträchtliche $+1.046$ bei.

<!--en:746d7dbb-->
![Beiträge zur KL-Divergenz zwischen $\mathcal{N}(0,1)$ und $\mathcal{N}(0,4)$. (a) Das breite Modell erklärt die schmale Verteilung. Ein positiver Beitrag im Zentrum und leicht negative Beiträge in den Randbereichen ergeben zusammen $0.318$ Nat. (b) Das schmale Modell erklärt die breite Verteilung. Die Randbereiche, denen das Modell wenig Wahrscheinlichkeit zuweist, tragen $1.05$ Nat bei, das Zentrum $-0.24$. Insgesamt ergibt sich $0.807$. Das Vorzeichen wechselt in beiden Richtungen bei $|x| = 1.36$.](../img/mdl-it-kl-integrand.svg)

<!--en:ae0a3365-->
Breit verteilte Daten überraschen ein schmales Modell viel stärker, als schmal verteilte Daten ein breites Modell überraschen. Allgemein bestraft die KL-Divergenz ein Modell stark, wenn Ergebnisse eintreten, die es beinahe ausgeschlossen hat. Verteilt es dagegen nur zusätzliche Wahrscheinlichkeit auf Bereiche ohne Daten, bleibt die Strafe gering. Was diese Asymmetrie für die Anpassung generativer Modelle je nach Richtung des Vergleichs bedeutet, untersucht :numref:`sec_mdl-fwd-vs-rev-kl`.

<!--en:843e8264-->
Die Definition zeigt auch, wie wir eine KL-Divergenz ohne geschlossene Form schätzen können. Wir ziehen Stichproben aus $P$ und mitteln $\log p(x) - \log q(x)$. Da wir später auf solche Schätzungen angewiesen sind, prüfen wir ihre Genauigkeit an diesem Paar von Gauß-Verteilungen. Die nächste Zelle implementiert die geschlossene Form sowie den Monte-Carlo-Schätzer mit seinem Standardfehler.

<!--en:757f5f39-->
Für die Verschiebung des Mittelwerts können wir die Genauigkeit des Schätzers vorhersagen. Der Logarithmus des Verhältnisses lautet $\log p(x) - \log q(x) = \tfrac12 - x$ und hat die Varianz $1$ unter $P$. Der Mittelwert aus $n = 2 \cdot 10^5$ Ziehungen hat daher den Standardfehler $1/\sqrt{n} = 0.0022$.

<!--en:30101f12-->
Bei dem Paar mit verschiedenen Varianzen lassen sich die beiden Richtungen dagegen unterschiedlich gut schätzen. Bis auf das Vorzeichen mitteln wir dieselbe Funktion $\ln 2 - \tfrac38 x^2$. Einmal ziehen wir jedoch aus der schmalen Gauß-Verteilung, das andere Mal aus der breiten, bei der $x^2$ viermal so stark schwankt. Schätzen wir alle drei Divergenzen und prüfen wir das Ergebnis.

<!--en:b2eb0f54-->
Die Standardfehler betragen $0.0022$, $0.0012$ und $0.0047$. Jede Schätzung liegt innerhalb weniger Standardfehler um den Wert der geschlossenen Formel. Bei Ziehungen aus der breiten Gauß-Verteilung ist der Standardfehler viermal so groß wie in umgekehrter Richtung. Für dieselbe Genauigkeit brauchen wir daher sechzehnmal so viele Ziehungen. Teilbild (b) in :numref:`fig_mdl-kl-integrand` zeigt den Grund. Der größte Teil dieser Divergenz stammt aus den Randbereichen. Dort ist der Logarithmus des Verhältnisses groß, aber es fallen nur wenige Stichprobenwerte dorthin. Auf dasselbe Problem bei der Stichprobenziehung stoßen wir in :numref:`sec_mdl-mi-estimation` in viel größerem Maßstab.

<!--en:4b8f970a-->
### Maximale Entropie und Gauß-Verteilung

<!--en:e4c863ce-->
Auf einer endlichen Menge maximiert die Gleichverteilung die Unsicherheit. Welche Dichte spielt dieselbe Rolle auf der reellen Achse? Ohne Nebenbedingung gibt es keine solche Dichte. Keine Dichte kann auf ganz $\mathbb{R}$ konstant sein, und durch Strecken lässt sich die differentielle Entropie unbegrenzt erhöhen. Geben wir stattdessen die beiden Momente vor, die ein gaußsches Modell ausgibt, den Mittelwert $\mu$ und die Varianz $\sigma^2$. Unter allen Dichten mit diesen beiden Momenten hat die Gauß-Verteilung die größte differentielle Entropie. Genau in diesem Sinne trifft ein Modell mit gaußschem Rauschen die wenigsten Annahmen. Es legt nur Mittelwert und Varianz fest.

<!--en:bc1a80ec-->
Der Beweis folgt mit einer Änderung unserem einzeiligen Argument für die Gleichverteilung. Im diskreten Fall war $-\log u(x) = \log k$ für alle Ergebnisse konstant. Daher hatte jede Verteilung $P$ dieselbe Kreuzentropie $\log k$ zum gleichverteilten Modell. Hier bezeichne $\varphi$ die gaußsche Dichte mit Mittelwert $\mu$ und Varianz $\sigma^2$. Ihr Überraschungswert $-\log \varphi(x) = \tfrac{1}{2}\log(2\pi\sigma^2) + \tfrac{(x-\mu)^2}{2\sigma^2}$ ist nicht mehr konstant, aber eine quadratische Funktion von $x$. Der Erwartungswert einer quadratischen Funktion hängt nur über die beiden vorgegebenen Momente von $P$ ab. Für jede Dichte $P$ mit Mittelwert $\mu$ und Varianz $\sigma^2$ gilt

<!--en:5046add7-->
Die Kreuzentropie zu $\varphi$ hängt also bemerkenswerterweise nicht davon ab, welches dieser $P$ die Daten erzeugt hat. Setzen wir $P = \varphi$, wird die Kreuzentropie zur Entropie. Die Konstante ist daher die differentielle Entropie der Gauß-Verteilung selbst, deren geschlossene Form wir zuvor angegeben haben. Für jede andere Dichte $P$ mit denselben Momenten liefern die Zerlegung :eqref:`eq_mdl-ce_decomp` (mit $h$ anstelle von $H$) und die Gibbs-Ungleichung

<!--en:13244b51-->
mit Gleichheit genau dann, wenn $P = \mathcal{N}(\mu, \sigma^2)$. (Für $h(P) = -\infty$ gilt die Ungleichung unmittelbar.) Auch der Abstand zur Schranke, $D_{\textrm{KL}}(P\|\varphi)$, hat eine direkte Bedeutung. Er ist der zusätzliche Verlust des bestmöglichen gaußschen Modells, wenn die Daten keiner Gauß-Verteilung folgen (Aufgabe :exref:`ex_mdl-information-theory-gaussian-excess`).

<!--en:fdf340d7-->
Die Gleichverteilung und die Gauß-Verteilung sind zwei Beispiele für dasselbe Prinzip. Hat der Überraschungswert einer Kandidatendichte unter jeder Verteilung, die den Nebenbedingungen genügt, denselben Mittelwert, dann maximiert diese Dichte die Entropie unter all diesen Verteilungen. :numref:`subsec_mdl-max-entropy-derivation` beweist das Ergebnis allgemein für Nebenbedingungen an die Erwartungswerte beliebiger Statistiken. Die maximierenden Verteilungen bilden dabei eine Exponentialfamilie. Ein drittes Beispiel ist die geometrische Verteilung auf den nichtnegativen ganzen Zahlen $\{0, 1, 2, \ldots\}$. Ihr Überraschungswert ist eine affine Funktion des Ergebnisses. Daher maximiert sie die Entropie unter allen Verteilungen auf den nichtnegativen ganzen Zahlen mit gegebenem Mittelwert.

<!--en:e665849e-->
Welche Fragen bleiben offen? Das diskrete Modell $Q'$ wies zuvor einem tatsächlich eingetretenen Ergebnis die Wahrscheinlichkeit null zu. Damit wurde seine KL-Divergenz unendlich. Die schmale Gauß-Verteilung in :numref:`fig_mdl-kl-integrand` kam dem nahe. Der Großteil ihrer Divergenz stammte aus Randbereichen, die sie fast ausgeschlossen hatte. Ein generatives Modell, das wenige Zufallszahlen auf ein Bild abbildet, legt seine gesamte Wahrscheinlichkeitsmasse auf eine dünne Fläche im Pixelraum. Liegen die realen Daten nicht genau auf dieser Fläche, ist die KL-Divergenz unendlich, ob die Fläche nun ganz nah an den Daten vorbeiläuft oder weit entfernt. Wie :numref:`sec_mdl-divergences-distances` zeigt, kann keine Divergenz aus dem Dichteverhältnis $p/q$ diesen Unterschied erfassen. In :numref:`sec_mdl-test-functions-transport` entwickeln wir Vergleichsmaße, die nah und fern auch hier unterscheiden. Eine zweite Frage knüpft unmittelbar an unser Fragespiel an. Wir haben Ja-Nein-Fragen gezählt und die Antworten als binäre Codewörter gelesen. Das haben wir bisher nur für eine dyadische Quelle getan, deren Überraschungswerte in Bit ganzzahlig waren. In :numref:`sec_mdl-coding-compression` konstruieren wir Codes für beliebige Quellen und fragen, wie nahe ein reales Kompressionsverfahren der Entropie kommen kann.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:5e2d16a0-->
Der Kreuzentropieverlust misst den mittleren Überraschungswert eines Modells auf den Daten. Er zerfällt in die Entropie der Daten, die kein Modell senken kann, und die KL-Divergenz von den Daten zum Modell. Nach der Gibbs-Ungleichung ist diese Divergenz nie negativ und verschwindet nur für die wahre Verteilung. Einen Verlustwert müssen wir daher an seiner irreduziblen Untergrenze messen, der bedingten Entropie des Labels bei gegebener Eingabe. Auf neuen Daten erreichen nur die wahren bedingten Wahrscheinlichkeiten diese Grenze. Bestimmt die Eingabe das Label eindeutig, ist die Untergrenze null. Bei symmetrischem Labelrauschen beträgt sie $H_b(\rho) + \rho \log(k-1)$ für die Rauschrate $\rho$.

<!--en:638f85df-->
Die Entropie einer endlichen Quelle in Bit gibt bis auf eine Frage an, wie viele Ja-Nein-Fragen wir im Mittel brauchen, um ein Ergebnis zu bestimmen. Sie beträgt höchstens $\log_2 k$ bei $k$ möglichen Ergebnissen. Weichere Zielverteilungen verschieben das Optimum. Label-Smoothing erzeugt einen endlichen optimalen Logit-Abstand und eine positive Verlustuntergrenze. Bei der Wissensdestillation gleicht sich der Schüler an die weichere Verteilung des Lehrers an. Der Faktor $\tau^2$ hält die Größenordnung des Gradienten stabil, wenn sich die Temperatur ändert. Bei stetigen Dichten kann die differentielle Entropie negativ sein und verschiebt sich mit der Maßeinheit. Die KL-Divergenz und der Minimierer der Kreuzentropie bleiben dagegen invariant.

<!--en:2294243a-->
## Übungen

<!--en:fe1ed2af-->
1. **Eindeutigkeit des Informationsgehalts.** ● Zeigen Sie, dass der Informationsgehalt bis auf die Skalierung eindeutig ist. Erfüllt eine streng fallende Funktion $I(p)$ auf $(0, 1]$ die Bedingungen $I(1) = 0$ und $I(p_1 p_2) = I(p_1) + I(p_2)$ für alle $p_1, p_2 \in (0, 1]$, dann gilt $I(p) = -c \log p$ für eine Konstante $c > 0$. Verwenden Sie die Funktion $g(t) = I(e^{-t})$ für $t \geq 0$.
   :exlabel:`ex_mdl-information-theory-show-that-self-information`

<!--en:d24c9049-->
1. [code] **Untergrenze bei Labelrauschen.** Bei $k$ Klassen ist das Label mit Wahrscheinlichkeit $1-\rho$ richtig und andernfalls gleichverteilt über die übrigen $k-1$ Klassen, unabhängig von der Eingabe.
    1. Nehmen Sie an, das unverfälschte Label sei eine Funktion der Eingabe. Zeigen Sie, dass die Kreuzentropie jedes Klassifikators auf der Population mindestens $H_b(\rho) + \rho\log(k-1)$ beträgt. Zeigen Sie außerdem, dass die optimale Vorhersage die verrauschte bedingte Verteilung selbst ist und die optimale Genauigkeit $1-\rho$ beträgt, sofern $1 - \rho > \rho/(k-1)$ gilt.
    1. Berechnen Sie die Untergrenze für $k = 10$ und $\rho = 0.05, 0.1, 0.2$.
    1. Ein Kollege meldet einen Validierungsverlust von $0.40$ Nat für einen Datensatz mit 10 Klassen, bei dem jede Teilmenge nachweislich $10\%$ solches Rauschen enthält. Nennen Sie mögliche Erklärungen. Welche davon schließt die Untergrenze aus, welche lässt sie zu?
    1. Trainieren Sie die Softmax-Regression aus :numref:`sec_softmax_scratch` auf Fashion-MNIST. Verfälschen Sie dabei $10\%$ der Trainings- und Testlabels auf die beschriebene Weise. Vergleichen Sie den Testverlust mit $0.545$ und die Testgenauigkeit mit $90\%$. Der Abstand ist groß. Welche Annahme aus (a) verletzt Fashion-MNIST? Welchen zusätzlichen Beitrag liefert der Fehler des linearen Modells selbst? Wie könnten Sie die tatsächlich geltende Untergrenze schätzen?
   :exlabel:`ex_mdl-information-theory-label-noise-floor`

<!--en:89192a16-->
1. **Weitere Bewertungsregeln.** Eine Bewertungsregel $S(\mathbf{q}, y)$ heißt streng echt, wenn die erwartete Strafe $\mathbb{E}_{y \sim \mathbf{p}}[S(\mathbf{q}, y)]$ nur bei $\mathbf{q} = \mathbf{p}$ minimal ist. Für den logarithmischen Score folgt diese Eigenschaft aus der Gibbs-Ungleichung.
    1. Zeigen Sie für den Brier-Score mit $S(\mathbf{q}, y) = \sum_j (q_j - \mathbf{1}(j = y))^2$, dass sein Erwartungswert denjenigen bei Angabe der wahren Verteilung um $\|\mathbf{q} - \mathbf{p}\|^2$ übersteigt. Der Brier-Score ist somit streng echt :cite:`Gneiting.Raftery.2007`.
    1. Der lineare Score $S(\mathbf{q}, y) = -q_y$ belohnt die Wahrscheinlichkeit, die das Modell der beobachteten Klasse zuweist. Welche Angabe minimiert seine erwartete Strafe? Ist der lineare Score echt? Was gibt ein damit trainierter Klassifikator für eine Eingabe aus, deren Label zur Klasse $1$ mit Wahrscheinlichkeit $0.6$ und zur Klasse $2$ mit Wahrscheinlichkeit $0.4$ gehört?
   :exlabel:`ex_mdl-information-theory-scoring-rule-is-strictly`

<!--en:67265319-->
1. **Label-Smoothing bei mehrdeutigen Eingaben.** ● Beim Label-Smoothing dient die Verteilung :eqref:`eq_mdl-label-smoothing-target` als Trainingsziel.
    1. In der Trainingsverteilung trägt eine Eingabe $\mathbf{x}$ das Label $y$ mit Wahrscheinlichkeit $p(y \mid \mathbf{x})$. Zeigen Sie, dass der erwartete geglättete Verlust für diese Eingabe bei $\mathbf{q}^*(\cdot \mid \mathbf{x}) = (1 - \epsilon)\,
       p(\cdot \mid \mathbf{x}) + \tfrac{\epsilon}{k}\mathbf{1}$ minimal ist. Wie lässt sich $p(\cdot \mid \mathbf{x})$ aus $\mathbf{q}^*$ zurückgewinnen? Welche Wahrscheinlichkeit kann der Minimierer einer Klasse höchstens zuweisen?
    1. Für ein Beispiel mit Logits $\mathbf{z}$ und $\mathbf{q} = \mathrm{softmax}(\mathbf{z})$ lautet der Gradient von $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ bezüglich $z_j$ gerade $q_j - p^\epsilon_j$. Zeigen Sie für $\epsilon > 0$, dass an jedem stationären Punkt alle Logits der falschen Klassen gleich sind.
    1. Bei One-Hot-Zielen ($\epsilon = 0$) beträgt der Gradient bezüglich eines Logits einer falschen Klasse $q_j$, und der Verlust hat keinen stationären Punkt. Zeigen Sie, dass der Gradientenabstieg auf den Logits eines Beispiels den Abstand zwischen zwei Logits falscher Klassen dennoch verringert. Erklären Sie, warum dieser Rückgang für $q_y \to 1$ ins Stocken gerät. In einem Netz ergeben sich die Logits aller Beispiele aus gemeinsamen Merkmalen. Welche Unterschiede zwischen Logits falscher Klassen können daher das Training mit $\epsilon = 0$ überdauern? Warum beseitigt ein Training mit $\epsilon > 0$ diese Unterschiede?
    1. :citet:`Muller.Kornblith.Hinton.2019` beobachten, dass mit Label-Smoothing trainierte Lehrer beim Destillationsverlust :eqref:`eq_mdl-distill-loss` schlechtere Schüler hervorbringen. Erklären Sie diese Beobachtung anhand der Ergebnisse aus (b) und (c).
   :exlabel:`ex_mdl-information-theory-label-smoothing`

<!--en:720a9793-->
1. [code] **Destillation bei hoher Temperatur.** ● :citet:`Hinton.Vinyals.Dean.2015` führen in Abschnitt 2.1 ihrer Arbeit aus, dass die Destillation bei hoher Temperatur darauf hinausläuft, zentrierte Logits anhand des quadratischen Fehlers abzugleichen.
    1. Schreiben Sie $q_j = p_j(1 + u_j)$, sodass $\sum_j p_j u_j = 0$ gilt, und entwickeln Sie $D_{\textrm{KL}}(\mathbf{p} \| \mathbf{q}) = -\sum_j p_j \log(1 + u_j)$ bis zur zweiten Ordnung. Sie erhalten

       $$
       D_{\textrm{KL}}(\mathbf{p} \| \mathbf{q}) =
       \frac{1}{2}\sum_j \frac{(q_j - p_j)^2}{p_j}
       + O\big(\|\mathbf{q} - \mathbf{p}\|^3\big).
       $$

       Kombinieren Sie diese Entwicklung mit der für große $\tau$ geltenden Näherung von $\mathrm{softmax}(\mathbf{z}/\tau)$ aus :numref:`subsec_mdl-distillation`. Zeigen Sie damit, dass für $\tau \to \infty$ gilt:

       $$
       \tau^2 D_{\textrm{KL}}\big(\mathbf{p}^{(\tau)} \,\|\, \mathbf{q}^{(\tau)}\big)
       \to \frac{1}{2k} \sum_{j=1}^k
       \Big(\big(z^{\textrm{stu}}_j - \bar z^{\textrm{stu}}\big)
       - \big(z^{\textrm{tea}}_j - \bar z^{\textrm{tea}}\big)\Big)^2.
       $$

    1. Berechnen Sie $\tau^2 D_{\textrm{KL}}(\mathbf{p}^{(\tau)} \| \mathbf{q}^{(\tau)})$ für $\tau = 1, 2, 5, 10, 100$ mit den Logits `z_tea` $= (5, 2, -1)$ und `z_stu` $= (3, 3, 0)$ aus :numref:`subsec_mdl-distillation` und vergleichen Sie das Ergebnis mit dem Grenzwert.
    1. Welche Logits des Lehrers haben bei niedriger Temperatur fast keinen Einfluss auf den Verlust? Laut :citet:`Hinton.Vinyals.Dean.2015` kann das sowohl ein Vorteil als auch ein Nachteil sein. Geben Sie für beide Seiten je ein Argument an. Gehen Sie davon aus, wie das Training des Lehrers mit Kreuzentropie diese Logits beeinflusst.
   :exlabel:`ex_mdl-information-theory-distillation-high-temperature`

<!--en:bd46fede-->
1. [code] **Quantisierte und differentielle Entropie im Vergleich.** Sei $X$ eine Zufallsvariable mit Dichte $p$. Die Variable $X_\Delta$ gebe an, welche Zelle $[i\Delta, (i+1)\Delta)$ den Wert $X$ enthält.
    1. Zeigen Sie, dass $H(X_\Delta) + \log \Delta \to h(X)$ für $\Delta \to 0$ gilt, wenn $p$ stetig ist und $p$ sowie $p \log p$ Riemann-integrierbar sind. Schreiben Sie mithilfe des Mittelwertsatzes der Integralrechnung $P(X_\Delta = i) = p(x_i)\Delta$ für ein $x_i$ in der Zelle.
    1. Prüfen Sie den Grenzwert von Hand für die Gleichverteilung auf $[0, 1]$ mit $\Delta = \tfrac18$. Was passiert bei einer Zellbreite, die $1$ nicht ganzzahlig teilt, etwa $\Delta = 0.3$?
    1. Berechnen Sie $H(X_\Delta) + \log \Delta$ numerisch für $X \sim \mathcal{N}(0, 1)$ mit $\Delta = 0.5, 0.1, 0.01$. Bestimmen Sie die Zellwahrscheinlichkeiten mit der Verteilungsfunktion der Gauß-Verteilung und vergleichen Sie das Ergebnis mit $h = \tfrac12\log(2\pi e) \approx 1.4189$ Nat. (Bei $\Delta = 0.5$ sollten Sie $1.4292$ erhalten.)
    1. Warum divergiert $H(X_\Delta)$ selbst für $\Delta \to 0$? Ein Regressionsziel wird in Zellen der Breite $\Delta$ eingeteilt und mit einem Softmax-Klassifikator vorhergesagt. Welcher Größe nähert sich sein Verlust im besten Fall? Was müssen Sie addieren, um ihn mit der negativen Log-Dichte eines stetigen Modells vergleichen zu können?
   :exlabel:`ex_mdl-information-theory-quantized-entropy`

<!--en:9851aa0d-->
1. **Gaußsche KL-Divergenz.** Sei $P = \mathcal{N}(\mu_1, \sigma_1^2)$.
    1. Leiten Sie die geschlossene Form :eqref:`eq_mdl-gaussian_kl` her. Schreiben Sie dazu $\mathbb{E}_{x\sim P}[\log p(x) - \log q(x)]$ aus und verwenden Sie $\mathbb{E}_{x \sim P}[x] = \mu_1$ und $\mathbb{E}_{x\sim P}[(x-\mu_1)^2] = \sigma_1^2$.
    1. Die Verteilung $Q$ habe den Mittelwert $\mu_1 + \delta$ und die Standardabweichung $\sigma_1 e^{\eta}$. Entwickeln Sie sowohl $D_{\textrm{KL}}(P\|Q)$ als auch $D_{\textrm{KL}}(Q\|P)$ bis zur zweiten Ordnung in $\delta$ und $\eta$. In welcher Ordnung unterscheiden sich die beiden Richtungen erstmals?
    1. Werten Sie die Näherung zweiter Ordnung für $P = \mathcal{N}(0, 1)$ und $Q = \mathcal{N}(0, 4)$ aus. Vergleichen Sie sie für die beiden Richtungen mit den exakten Werten $0.318$ und $0.807$ Nat.
   :exlabel:`ex_mdl-information-theory-gaussian-kl`

<!--en:ad8edede-->
1. **Zusätzlicher Verlust eines gaußschen Modells.** Ein Regressionsmodell gibt eine Gauß-Verteilung $\mathcal{N}(m, s^2)$ für eine Zielgröße aus, deren Dichte $P$ den Mittelwert $\mu$, die Varianz $\sigma^2$ und eine endliche differentielle Entropie hat.
    1. Zeigen Sie, dass $\textrm{CE}(P, \mathcal{N}(m, s^2))$ bei $m = \mu$ und $s^2 = \sigma^2$ am kleinsten ist. Zeigen Sie, dass der Verlust dieses bestmöglichen gaußschen Modells die Untergrenze $h(P)$ um $D_{\textrm{KL}}(P\|\varphi) = h(\varphi) - h(P)$ übersteigt. Dabei ist $\varphi$ die Dichte von $\mathcal{N}(\mu, \sigma^2)$.
    1. Berechnen Sie den zusätzlichen Verlust für die Gleichverteilung auf $[0, a]$ mit Varianz $a^2/12$ und für die Laplace-Dichte $\tfrac{1}{2b} e^{-|x|/b}$ mit Varianz $2b^2$ und differentieller Entropie $1 + \log(2b)$. (Sie sollten etwa $0.176$ bzw. $0.072$ Nat erhalten.)
    1. Keines der Ergebnisse hängt von der Skala $a$ oder $b$ ab. Begründen Sie dies ohne Rechnung anhand der Aussagen über den Wechsel der Maßeinheit in :numref:`subsec_mdl-differential-entropy`.
    1. Bei vorgegebener Varianz hat die Gauß-Verteilung die größte Entropie. Keine Dichte hat also eine höhere Verlustuntergrenze. Gibt es auch einen größten zusätzlichen Verlust? Betrachten Sie Dichten, die sich in der Nähe der beiden Punkte $\mu \pm \sigma$ konzentrieren. Was bedeutet Ihre Antwort für ein gaußsches Modell einer Zielgröße mit zwei deutlich getrennten Modi?
   :exlabel:`ex_mdl-information-theory-gaussian-excess`

<!--en:0846e3a3-->
[Diskussion](https://d2l.discourse.group/t/1104)

<!--en:231af68f-->
Entropie, Kreuzentropie und KL-Divergenz<br>**Informationsgehalt · Shannon-Entropie · Gibbs-Ungleichung · irreduzibler Verlust · Label-Smoothing · Destillation**

<!--en:2747c0b4-->
::: {.slide title="Was bedeuten die Zahlen auf einer Verlustkurve?"}

<!--en:d0ab186a-->
[Motivation und Leitfrage]{.kicker}

<!--en:bf6b1c51-->
- **Untrainiertes Modell mit 10 Klassen ($\ln 10 \approx 2.30$ Nat):** Wer gleichverteilt rät ($q_j = 1/10$), erhält einen Überraschungswert von $2.30$ Nat pro Beispiel.

<!--en:31b00edb-->
- **Untergrenze mit und ohne Rauschen ($0$ bzw. $0.545$ Nat):** Für fehlerfreie deterministische Labels gilt die Untergrenze $0$. Bei $10\%$ symmetrischem Labelrauschen liegt die irreduzible Untergrenze bei $H(Y \mid X) = 0.545$ Nat.

<!--en:b1547b3b-->
- **Plateaus deuten:** Ein Plateau bei $0.60$ Nat ist für fehlerfreie Labels schlecht, liegt aber nur $0.055$ Nat vom Bayes-Optimum entfernt, wenn das Labelrauschen $10\%$ beträgt.

<!--en:214ee31a-->
::: {.slide title="Der Informationsgehalt misst die Überraschung über ein Ereignis"}

<!--en:ea40a500-->
[Informationsgehalt]{.kicker}

<!--en:153b9b66-->
Wie viel Information liefert die Beobachtung eines Ergebnisses $x$ mit Wahrscheinlichkeit $p(x)$?

<!--en:352711ee-->
- **Shannons drei Forderungen:**
  1. Ein sicheres Ereignis ($p = 1$) überrascht uns nicht: $I(1) = 0$.
  2. Seltenere Ereignisse überraschen uns stärker: $I(p)$ fällt streng mit $p$.
  3. Bei unabhängigen Ereignissen addieren sich die Überraschungswerte: $I(p_1 p_2) = I(p_1) + I(p_2)$.

<!--en:7b8c6f2f-->
- **Eindeutige mathematische Lösung:** Bis auf die Wahl der Skala (Logarithmusbasis) erfüllt nur der **Informationsgehalt** (oder *Überraschungswert*) diese Axiome:
  $$I(x) = -\log p(x)$$

<!--en:c4e40c65-->
- **Einheiten:** Die Basis $e$ ergibt Information in **Nat**, die Basis $2$ in **Bit** ($1\text{ bit} = \ln 2 \approx 0.693\text{ nats}$).

<!--en:4982250d-->
::: {.slide title="Überraschungswerte berechnen: 52! mögliche Kartenfolgen"}

<!--en:76cb8e63-->
[Informationsgehalt in der Praxis]{.kicker}

<!--en:8af76847-->
Der Überraschungswert wächst logarithmisch und lässt sich vom einzelnen Münzwurf bis zu riesigen kombinatorischen Ergebnisräumen ohne numerischen Überlauf berechnen:

<!--en:78135b6e-->
| Beobachtung am gemischten Kartenspiel | Wahrscheinlichkeit $p(x)$ | Überraschungswert in Nat ($-\ln p$) | Überraschungswert in Bit ($-\log_2 p$) |

<!--en:ea9e5a05-->
| „Ich sehe eine Karte“ (sicheres Ereignis) | $1$ | $0.00$ | $0.00$ |

<!--en:1bf22cbf-->
| „Ich sehe Herz“ (1 von 4 Farben) | $1/4$ | $\ln 4 \approx 1.39$ | $2.00$ |

<!--en:6a4cd7d0-->
| „Es ist die Pik-Drei“ (1 von 52 Karten) | $1/52$ | $\ln 52 \approx 3.95$ | $5.70$ |

<!--en:26a23c69-->
| Reihenfolge aller 52 Karten | $1/52!$ | $\ln \Gamma(53) \approx 156.36$ | $225.58$ |

<!--en:b46e4dba-->
::: {.slide title="Die Shannon-Entropie ist der erwartete Informationsgehalt"}

<!--en:423e0ace-->
[Diskrete Entropie]{.kicker}

<!--en:6b02d1b8-->
Für eine diskrete Zufallsvariable $X \sim P$ mit Wahrscheinlichkeiten $p_i$ ist die **Shannon-Entropie** der erwartete Überraschungswert:

<!--en:269bc5a8-->
- **Wahrscheinlichkeit null ($0\log 0 = 0$):** Unmögliche Ereignisse tragen nichts zum erwarteten Überraschungswert bei.

<!--en:15ad163d-->
- **Wertebereich und Invarianz:** $0 \le H(P) \le \log k$ ($0$ genau im deterministischen Fall, $\log k$ genau bei Gleichverteilung). Die Entropie hängt nur von $\mathbf{p}$ ab, unabhängig von den Werten der Ergebnisse.

<!--en:c92fdd5a-->
::: {.slide title="Entropie als Anzahl optimal gewählter Ja-Nein-Fragen"}

<!--en:b6e6576d-->
[Entropie und Binärcodes]{.kicker}

<!--en:3ba89a05-->
Auf einer endlichen Menge zählt $H_2(P)$ in Bit die mittlere Anzahl von Ja-Nein-Fragen, mit denen eine optimale Strategie das Ergebnis bestimmt:

<!--en:1c681f23-->
- **Dyadische Quelle:** Die Verteilung $P = (\tfrac12, \tfrac14, \tfrac18, \tfrac18)$ auf vier Symbolen $\{a, b, c, d\}$ liefert die Überraschungswerte $(1, 2, 3, 3)$ Bit.

<!--en:7f743b89-->
- **Adaptive Fragestrategie:** Zuerst *„Ist es $a$?“* (in einem Anteil von $50\%$ der Fälle genügt $1$ Frage), dann *„Ist es $b$?“*, dann *„Ist es $c$?“*. Das ergibt die Codewörter `1`, `01`, `001`, `000`:
  $$\mathbb{E}[\text{questions}] = \tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac18 \cdot 3 + \tfrac18 \cdot 3 = 1.75\text{ bits} = H_2(P)$$

<!--en:fbbbc818-->
::: {.slide title="Gemeinsame Entropie, bedingte Entropie und Kettenregel"}

<!--en:a3dae932-->
[Paare und Sequenzen]{.kicker}

<!--en:e267c4e6-->
Wegen $\log p(x, y) = \log p(x) + \log p(y \mid x)$ zerfällt die gemeinsame Unsicherheit bei der Bildung des Erwartungswerts in eine Summe:

<!--en:17e36b1d-->
- **Kettenregel für autoregressive Sequenzen:** $H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_{<i})$.

<!--en:89cf9a72-->
- **Probe an der dyadischen Quelle:** Die Frage *„Ist es $a$?“* liefert $H_2(X) = 1.00$ Bit. Es verbleiben $H_2(Y \mid X) = \tfrac12(0) + \tfrac12(1.5) = 0.75$ Bit, insgesamt also $1.75$ Bit.

<!--en:3184a07a-->
::: {.slide title="Binäre Entropie und obere Schranke bei Gleichverteilung"}

<!--en:f9b6fa37-->
[Maximale diskrete Entropie]{.kicker}

<!--en:d9f7276e-->
Wie groß kann die Unsicherheit einer diskreten Zufallsvariable mit $k$ Ergebnissen werden?

<!--en:b446d1c2-->
- **Binäre Entropiefunktion:** Für einen Bernoulli-Münzwurf mit Wahrscheinlichkeit $p \in [0, 1]$ für Kopf gilt:
  $$H_b(p) = -p \log p - (1-p)\log(1-p)$$

<!--en:a7152e61-->
- **Strenge Konkavität und Maximum:** $H_b(p)$ verschwindet an den deterministischen Endpunkten $p \in \{0, 1\}$. Das einzige Maximum liegt bei der fairen Münze mit $p = \tfrac12$ und beträgt $H_b(\tfrac12) = \ln 2 \approx 0.693$ Nat ($1$ Bit).

<!--en:08faf7e5-->
- **Allgemeine Obergrenze bei $k$ Ergebnissen:** Jede Verteilung $P$ auf $k$ Ergebnissen erfüllt $H(P) \le \log k$, mit Gleichheit genau dann, wenn $P$ gleichverteilt ist ($p_i = 1/k$).

<!--en:36d8d4ed-->
::: {.slide title="Die KL-Divergenz misst den mittleren zusätzlichen Überraschungswert"}

<!--en:09699190-->
[Kullback-Leibler-Divergenz]{.kicker}

<!--en:1f496139-->
Die Daten stammen aus der wahren Verteilung $P$, ihre Wahrscheinlichkeiten bewertet jedoch ein Modell $Q$. Die **KL-Divergenz** (oder *relative Entropie*) ist der mittlere zusätzliche Überraschungswert unter $P$:

<!--en:26c4dd28-->
- **Bedeutung in Bit:** $D_{\textrm{KL}}^{(2)}(P \| Q)$ ist die erwartete Anzahl *zusätzlicher* Ja-Nein-Fragen pro Symbol, wenn die Fragestrategie auf $Q$ zugeschnitten ist, die Ergebnisse aber aus $P$ stammen.

<!--en:ec1ea8fa-->
- **Rechnung für unser Beispielpaar:** Für $P = (0.6, 0.3, 0.1)$ und $Q = (0.2, 0.5, 0.3)$ gilt:
  $$D_{\textrm{KL}}(P \| Q) = 0.6 \ln\frac{0.6}{0.2} + 0.3 \ln\frac{0.3}{0.5} + 0.1 \ln\frac{0.1}{0.3} = 0.659 - 0.153 - 0.110 = 0.396\text{ nats}$$

<!--en:132a9c8a-->
- **Asymmetrie:** Beim Vertauschen von wahrer Verteilung und Modell ergibt sich $D_{\textrm{KL}}(Q \| P) = 0.365\text{ nats} \neq D_{\textrm{KL}}(P \| Q)$.

<!--en:309abea8-->
::: {.slide title="Asymmetrie der KL-Divergenz und Kosten einer Nullwahrscheinlichkeit"}

<!--en:2889fd05-->
[KL-Divergenz im Code]{.kicker}

<!--en:d3ce88e9-->
Gilt $q'(x) = 0$ an einer Stelle mit $p(x) > 0$, so ist der Überraschungswert $-\log q'(x)$ unendlich und es gilt $D_{\textrm{KL}}(P \| Q') = +\infty$:

<!--en:b27a8d8d-->
**Warum die Softmax-Funktion nie null liefert:** Wegen $e^{z_j} > 0$ für alle endlichen Logits $z_j$ garantiert ein Softmax-Klassifikator überall $q_j > 0$ und vermeidet so einen unendlichen KL-Verlust.

<!--en:6ccb87ee-->
::: {.slide title="Gibbs-Ungleichung: Kein Modell übertrifft die wahre Verteilung"}

<!--en:2a08e6cc-->
[Grundlegender Satz]{.kicker}

<!--en:47c40450-->
Für beliebige Verteilungen $P$ und $Q$ besagt die **Gibbs-Ungleichung**, dass $D_{\textrm{KL}}(P \| Q) \ge 0$ gilt, mit Gleichheit genau dann, wenn $P = Q$:

<!--en:7c4015cb-->
- **Jensen-Ungleichung:** Die strenge Konkavität von $\log$ liefert die Schranke $-\mathbb{E}_P[\log(q/p)] \ge -\log\mathbb{E}_P[q/p] = -\log Q(\textrm{supp}\,P) \ge 0$.

<!--en:96b1bc61-->
- **Gleichheitsfall:** Wegen strenger Konkavität gilt $q(x)/p(x) = 1$ auf ganz $\textrm{supp}\,P$ und somit $P = Q$.

<!--en:e950ac5b-->
::: {.slide title="Zwei unmittelbare Folgerungen aus der Gibbs-Ungleichung"}

<!--en:db351716-->
[Gleichverteilung und bedingte Entropie]{.kicker}

<!--en:cf547a8c-->
Zwei grundlegende Eigenschaften der Entropie folgen in jeweils einer Zeile aus der Gibbs-Ungleichung:

<!--en:3c783b04-->
- **1. Die Gleichverteilung maximiert die diskrete Entropie ($H(P) \le \log k$):**
  Eine beliebige Verteilung $P$ auf $k$ Ergebnissen mit der Gleichverteilung $U = (1/k, \ldots, 1/k)$ vergleichen:
  $$0 \le D_{\textrm{KL}}(P \| U) = \sum_{i=1}^k p_i \log\frac{p_i}{1/k} = \log k - H(P) \implies H(P) \le \log k$$

<!--en:8bb3fb03-->
- **2. Bedingen erhöht die Entropie im Mittel nie ($H(Y \mid X) \le H(Y)$):**
  Die gemeinsame Verteilung $p_{X,Y}$ mit dem Produkt ihrer Randverteilungen $p_X p_Y$ vergleichen:
  $$H(Y) - H(Y \mid X) = \mathbb{E}_{p_{X,Y}}\!\left[\log\frac{p(x, y)}{p(x)p(y)}\right] = D_{\textrm{KL}}(p_{X,Y} \,\|\, p_X p_Y) \ge 0$$
  *(Für eine einzelne Realisierung $X = x$ kann dennoch $H(Y \mid X = x) > H(Y)$ gelten, etwa wenn ein seltenes Signal $X=1$ anzeigt, dass eine sonst deterministische Münze fair ist.)*

<!--en:4387eed4-->
::: {.slide title="Die Kreuzentropie ist die Summe aus Entropie und KL-Divergenz"}

<!--en:0b7af27d-->
[Zerlegung des Verlusts]{.kicker}

<!--en:76c7f62f-->
Die **Kreuzentropie** von $P$ zu $Q$ ist der mittlere Überraschungswert des Modells $Q$ für Daten aus $P$:

<!--en:f0122dde-->
- **Exakte Zerlegung:** Addition und Subtraktion von $\log p(x)$ innerhalb von $\textrm{CE}(P, Q) = -\mathbb{E}_P[\log q(x)]$ ergeben:
  $$\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P \| Q)$$

<!--en:23de2e96-->
- **Warum Maximum-Likelihood die KL-Divergenz minimiert:** Die Datenentropie $H(P)$ hängt nicht von den Modellparametern $\boldsymbol{\theta}$ ab. Somit ist die Minimierung von $\textrm{CE}(P, Q_{\boldsymbol{\theta}})$ gleichbedeutend mit der Minimierung von $D_{\textrm{KL}}(P \| Q_{\boldsymbol{\theta}})$.

<!--en:13897af5-->
- **Beispiele:** Für $(P, Q)$ ergibt sich $1.294 = 0.898 + 0.396$ Nat, für die dyadische Quelle und das gleichverteilte Modell $U$ dagegen $2.00 = 1.75 + 0.25$ Bit.

<!--en:36ed8790-->
::: {.slide title="One-Hot-Klassifikation und numerisch stabiles Log-Softmax"}

<!--en:2ea46926-->
[Verlust bei überwachter Klassifikation]{.kicker}

<!--en:1014f8e1-->
Wie wird die allgemeine Formel für die Kreuzentropie bei der Klassifikation mit $k$ Klassen zum Trainingsverlust?

<!--en:ebf33364-->
- **Verlust eines Beispiels:** Für ein erfasstes Label $y \in \{1, \ldots, k\}$ ist der One-Hot-Vektor $\mathbf{e}_y$ die empirische Zielverteilung. Also gilt $\textrm{CE}(\mathbf{e}_y, \mathbf{q}) = -\sum_{j=1}^k \mathbf{1}(j = y)\log q_j = -\log q_y$.

<!--en:412b6af3-->
- **Empirische negative Log-Likelihood des Datensatzes:** Mitteln über $n$ Trainingsbeispiele $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ ergibt:
  $$\mathcal{L}_{\textrm{CE}} = -\frac{1}{n}\sum_{i=1}^n \log q_{i, y_i}$$

<!--en:13badb2c-->
- **Numerische Stabilität durch LogSumExp:** Wer zuerst $q_j = e^{z_j}/\sum_\ell e^{z_\ell}$ bildet, riskiert bei großen Logits einen Überlauf und bei negativen Logits einen Unterlauf auf $\log 0 = -\infty$. Frameworks verschieben die Logits direkt im logarithmischen Raum um $m = \max_\ell z_\ell$:
  $$\log q_y = (z_y - m) - \log \sum_{\ell=1}^k e^{z_\ell - m}$$

<!--en:94d3803f-->
::: {.slide title="Die irreduzible Untergrenze des Klassifikationsverlusts"}

<!--en:e64ac5df-->
[Zerlegung des Populationsverlusts]{.kicker}

<!--en:6090a944-->
Auf neuen Testbeispielen $(\mathbf{x}, y) \sim P$ zerfällt die Kreuzentropie der Population in die bedingte Entropie der Labels und die erwartete KL-Divergenz:

<!--en:452e0aa6-->
- **Eindeutiges globales Optimum:** Der Verlust erreicht $H(Y \mid X)$ genau dann, wenn fast überall $q(\cdot \mid \mathbf{x}) = p(\cdot \mid \mathbf{x})$ gilt.

<!--en:6db37113-->
- **Untergrenze bei symmetrischem Rauschen (Rate $\rho$, $k$ Klassen):** $H(Y \mid X) = H_b(\rho) + \rho\log(k - 1)$.

<!--en:adfa136f-->
::: {.slide title="Symmetrisches Labelrauschen und die Kosten übermäßiger Sicherheit"}

<!--en:a1ecbdd8-->
[Rechenbeispiel mit 10% Labelrauschen]{.kicker}

<!--en:55195f9c-->
Bei $k = 10$ Klassen und $\rho = 0.1$ Labelrauschen liegt die Untergrenze bei $H_b(0.1) + 0.1 \ln 9 = 0.325 + 0.220 = 0.545$ Nat. Drei Prädiktoren auf dieser verrauschten Testverteilung im Vergleich:

<!--en:d99dbb04-->
**Warum übermäßige Sicherheit schadet:** Ein Klassifikator, der der wahren Klasse die Wahrscheinlichkeit $0.99$ zuweist, erreicht mit $90\%$ die höchstmögliche Genauigkeit. Dennoch beträgt sein Verlust $0.689$ Nat, weil bei den $10\%$ verfälschten Labels jedes einzelne einen Überraschungswert von $-\ln(0.01/9) = 6.80$ Nat hat.

<!--en:da17232c-->
::: {.slide title="Streng echte Bewertungsregeln und Kalibrierung"}

<!--en:20e5bbeb-->
[Entscheidungstheorie und Kalibrierung]{.kicker}

<!--en:24b153a2-->
- **Streng echte Bewertungsregel:** Eine Strafe $S(\mathbf{q}, y)$, deren Erwartungswert $\mathbb{E}_{y \sim \mathbf{p}}[S(\mathbf{q}, y)]$ nur für die Angabe $\mathbf{q} = \mathbf{p}$ minimal ist. Das garantiert kalibrierte Wahrscheinlichkeiten.

<!--en:450a208e-->
- **Logarithmischer Score ($-\log q_y$):** Die zusätzliche Strafe gegenüber der wahren Angabe beträgt $D_{\textrm{KL}}(\mathbf{p} \| \mathbf{q}) \ge 0$. Der Score divergiert für $q_y \to 0$ und bestraft selbstsichere Fehler daher besonders stark.

<!--en:f643e758-->
- **Brier-Score ($\sum_j(q_j - \mathbf{1}_{j=y})^2$):** Die zusätzliche Strafe gegenüber der wahren Angabe beträgt $\|\mathbf{q} - \mathbf{p}\|_2^2 \ge 0$. Der Score ist durch $2$ beschränkt, selbst wenn $q_y = 0$ gilt.

<!--en:be2d474a-->
::: {.slide title="Label-Smoothing ergibt einen endlichen optimalen Logit-Abstand"}

<!--en:a1a64269-->
[Weiche Zielverteilungen]{.kicker}

<!--en:4887a82b-->
Mit One-Hot-Zielen ($\mathbf{e}_y$) erfordert $q_y = 1$ einen unendlichen Logit-Abstand $z_y - z_j \to \infty$. Die Gewichte wachsen unbegrenzt, das Modell wird übermäßig selbstsicher. **Label-Smoothing** mischt $\mathbf{e}_y$ mit gleichverteilter Masse $\epsilon \in (0, 1)$:

<!--en:d89249ab-->
- **Endlicher optimaler Logit-Abstand:** Das Minimum von $\textrm{CE}(\mathbf{p}^\epsilon, \mathbf{q})$ liegt bei $q_y = 1 - \epsilon + \epsilon/k$ und $q_j = \epsilon/k$ ($j \neq y$). Die Softmax-Funktion erreicht es bei einem endlichen Abstand:
  $$z_y - z_j = \log\frac{1 - \epsilon + \epsilon/k}{\epsilon/k} = \log\!\left(1 + \frac{k(1 - \epsilon)}{\epsilon}\right)$$

<!--en:a768790a-->
- **Zahlenbeispiel ($k = 10, \epsilon = 0.1$):** Der optimale Logit-Abstand beträgt $\ln 91 \approx 4.51$, die größte vorhergesagte Wahrscheinlichkeit höchstens $0.91$. Der Trainingsverlust erreicht ein Plateau bei $H(\mathbf{p}^\epsilon) = 0.500$ Nat statt bei $0$.

<!--en:b35c6005-->
::: {.slide title="Logit-Drift bei One-Hot-Zielen und bei geglätteten Zielen"}

<!--en:1edfba85-->
[Dynamik des Gradientenabstiegs]{.kicker}

<!--en:2d529053-->
Beim Gradientenabstieg für ein Beispiel ($\nabla_{\mathbf{z}} \textrm{CE} = \mathbf{q} - \mathbf{p}^\epsilon$) vergrößert ein One-Hot-Ziel ($\epsilon = 0$) den Logit-Abstand $g_t$ in jedem Schritt um $\approx k e^{-g_t}$. Dadurch wächst er logarithmisch gemäß $g_t \approx \ln(kt)$. Bei $\epsilon = 0.1$ kommt er hingegen bei $\ln 91 = 4.51$ zum Stillstand:

<!--en:4d8f3e7d-->
::: {.slide title="Wissensdestillation und Temperaturskalierung"}

<!--en:20e325c8-->
[Destillation vom Lehrer zum Schüler]{.kicker}

<!--en:aeb207dd-->
Bei der **Wissensdestillation** übernimmt ein kompaktes Schülernetz mit Logits $\mathbf{z}^{\textrm{stu}}$ das verborgene Wissen aus den Wahrscheinlichkeiten des Lehrers für falsche Klassen. Die Temperatur $\tau > 0$ macht beide Verteilungen weicher:

<!--en:a6111841-->
- **Warum mit $\tau^2$ multiplizieren?** Der Gradient von unskaliertem $D_{\textrm{KL}}(\mathbf{p}^{(\tau)} \| \mathbf{q}^{(\tau)})$ bezüglich des Schüler-Logits $z_j^{\textrm{stu}}$ lautet $\frac{1}{\tau}(q_j^{(\tau)} - p_j^{(\tau)})$.

<!--en:719c4e82-->
- **Taylor-Entwicklung bei hoher Temperatur ($\tau \gg 1$):** Die Entwicklung von $\mathrm{softmax}(\mathbf{z}/\tau)_j \approx \frac{1}{k}(1 + \frac{z_j - \bar{z}}{\tau})$ zeigt, dass die Wahrscheinlichkeitsdifferenz wie $1/\tau$ schrumpft. Der unskalierte Gradient fällt daher wie $1/\tau^2$ ab:
  $$\frac{\partial}{\partial z_j^{\textrm{stu}}} D_{\textrm{KL}}\!\left(\mathbf{p}^{(\tau)} \,\|\, \mathbf{q}^{(\tau)}\right) \approx \frac{1}{k\tau^2}\Big(\big(z_j^{\textrm{stu}} - \bar{z}^{\textrm{stu}}\big) - \big(z_j^{\textrm{tea}} - \bar{z}^{\textrm{tea}}\big)\Big)$$

<!--en:e2f2149f-->
::: {.slide title="Skalierung des Gradienten mit τ² im Code prüfen"}

<!--en:8d8183a3-->
[Numerik der Destillation]{.kicker}

<!--en:ec627a5b-->
Ohne den Faktor $\tau^2$ sinkt die Gradientennorm der Destillation bei einer Erhöhung von $\tau$ von $1$ auf $20$ um den Faktor $300\times$ (von $0.64$ auf $0.0021$). Durch Multiplikation mit $\tau^2$ bleibt der Betrag nahe dem Grenzwert für den Abgleich zentrierter Logits:

<!--en:40154215-->
::: {.slide title="Die differentielle Entropie hängt von der Maßeinheit ab"}

<!--en:005cc84b-->
[Stetige Zufallsvariablen]{.kicker}

<!--en:87af9b7f-->
Für eine stetige Zufallsvariable $X$ mit Dichte $p(x)$ beträgt die **differentielle Entropie** $h(X) = -\int p(x)\log p(x)\,dx$. Anders als die diskrete Entropie $H(X)$ kann sie negativ sein und hängt von der Skala ab:

<!--en:61995073-->
- **Negative Werte und Maßeinheiten:** Für $X \sim \mathrm{Unif}[0, a]$ ergibt $p(x) = 1/a$ den Wert $h(X) = \log a$. Bei $a = \tfrac12\text{ m}$ gilt $h(X) = -\ln 2 \approx -0.693$ Nat. Bei Angabe derselben Länge in Zentimetern ($a = 50\text{ cm}$) kommen $\ln 100 \approx 4.61$ Nat hinzu.

<!--en:b41ed89e-->
- **Variablentransformation:** Bei einer glatten Bijektion $\mathbf{y} = g(\mathbf{x})$ verschiebt die Jacobi-Determinante die Entropie $h$:
  $$h(g(\mathbf{X})) = h(\mathbf{X}) + \mathbb{E}\big[\log |\det Dg(\mathbf{X})|\big]$$

<!--en:5d73723f-->
- **Quantisierungsgrenzwert und KL-Invarianz:** Nach Einteilung von $X$ in Zellen der Breite $\Delta$ gilt $H(X_\Delta) + \log \Delta \to h(X)$ für $\Delta \to 0$. In $D_{\textrm{KL}}(P \| Q) = \int p(x)\log\frac{p(x)}{q(x)}\,dx$ kürzen sich die Jacobi-Faktoren im Verhältnis $p/q$. Die **KL-Divergenz bleibt bei einer Umskalierung der Koordinaten vollständig invariant**.

<!--en:0bf88c1c-->
::: {.slide title="KL-Divergenz zwischen eindimensionalen Gauß-Verteilungen"}

<!--en:abb8289b-->
[Geschlossene Form und Asymmetrie in den Randbereichen]{.kicker}

<!--en:3442f48a-->
Für zwei Gauß-Verteilungen $P = \mathcal{N}(\mu_1, \sigma_1^2)$ und $Q = \mathcal{N}(\mu_2, \sigma_2^2)$ hat die KL-Divergenz eine exakte geschlossene Form:

<!--en:b4b97bcb-->
- **Exakte Formel für die gaußsche KL-Divergenz:**
  $$D_{\textrm{KL}}(P \| Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

<!--en:8b5e1fef-->
- **Breites Modell, schmale Datenverteilung:** $D_{\textrm{KL}}(\mathcal{N}(0, 1) \| \mathcal{N}(0, 4)) = \ln 2 + \tfrac18 - \tfrac12 \approx 0.318$ Nat.

<!--en:9c3ee9ef-->
- **Schmales Modell, breite Datenverteilung:** $D_{\textrm{KL}}(\mathcal{N}(0, 4) \| \mathcal{N}(0, 1)) = -\ln 2 + 2 - \tfrac12 \approx 0.807$ Nat ($2.5\times$ so groß, da die Überraschungswerte in den Randbereichen für $|x| > 1.36$ hoch sind).

<!--en:af8b22e8-->
::: {.slide title="Monte-Carlo-Schätzung der KL-Divergenz und maximale Entropie"}

<!--en:0c49d67f-->
[Stichprobenvarianz und Optimalität der Gauß-Verteilung]{.kicker}

<!--en:1ecd928f-->
Bei der Schätzung von $D_{\textrm{KL}}(P \| Q)$ aus $n = 2 \times 10^5$ Ziehungen ist der Standardfehler für die breite Gauß-Verteilung $4\times$ so groß (für gleiche Genauigkeit sind $16\times$ so viele Ziehungen nötig). Die Divergenz stammt vor allem aus den selten erfassten Randbereichen:

<!--en:e599512c-->
**Satz über die maximale Entropie der Gauß-Verteilung:** Da der gaußsche Überraschungswert $-\log \varphi(x) = \tfrac12\log(2\pi\sigma^2) + \frac{(x-\mu)^2}{2\sigma^2}$ quadratisch ist, gilt $\textrm{CE}(P, \varphi) = \tfrac12\log(2\pi e\sigma^2) = h(\varphi)$ für *jede* Dichte $P$ mit Mittelwert $\mu$ und Varianz $\sigma^2$. Somit gilt $h(P) = h(\varphi) - D_{\textrm{KL}}(P \| \varphi) \le \tfrac12\log(2\pi e\sigma^2)$ mit Gleichheit genau dann, wenn $P = \mathcal{N}(\mu, \sigma^2)$.

<!--en:6d441018-->
::: {.slide title="Verständnisfragen zu Entropie, Rauschuntergrenzen und Einheiten"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:8a68ba3a-->
Drei Rechnungen zur Einordnung der Ergebnisse:

<!--en:75723d69-->
1. **Untrainiertes Modell:** Welchen Kreuzentropieverlust liefert ein frisch initialisierter Klassifikator mit gleichverteilten Vorhersagen für $3$ Klassen in Nat und für $10$ Klassen in Bit?

<!--en:5bbf3a9e-->
2. **Auffälliger Validierungsverlust:** Ein Datensatz mit 10 Klassen enthält in jeder Teilmenge $20\%$ symmetrisches Labelrauschen. Wo liegt die irreduzible Untergrenze des Populationsverlusts? Welcher Schluss liegt nahe, wenn ein Kollege einen Validierungsverlust von $0.40$ Nat meldet?

<!--en:47905f59-->
3. **Umskalieren stetiger Zielwerte:** Die Körpergröße Erwachsener ist näherungsweise gaußverteilt mit Standardabweichung $\sigma = 0.07\text{ m}$. Wie ändern sich die differentielle Entropie $h(X)$ und die Divergenz $D_{\textrm{KL}}(P \| Q)$, bei der Umrechnung in Zentimeter?

<!--en:557b98b6-->
**Lösungen:**

<!--en:570cf6c8-->
1. **Untrainiertes Modell:** $\ln 3 \approx 1.10\text{ nats}$ für $3$ Klassen und $\log_2 10 = \frac{\ln 10}{\ln 2} \approx 3.32\text{ bits}$ ($2.30\text{ nats}$) für $10$ Klassen.

<!--en:2b8467bc-->
2. **Untergrenze bei Rauschen:** $H_b(0.2) + 0.2\ln 9 = 0.500 + 0.439 = 0.940\text{ nats}$ (maximale Genauigkeit $80\%$). Ein Validierungsverlust von $0.40 < 0.940$ lässt sich nicht durch ein besseres Modell erklären. Er deutet auf eine Datenleckage oder bereinigte Validierungslabels hin.

<!--en:727f5886-->
3. **Einheiten:** In Metern gilt $h = \tfrac12\ln(2\pi e) + \ln 0.07 = 1.42 - 2.66 = -1.24\text{ nats}$. Die Umrechnung in Zentimeter multipliziert $\sigma$ mit $100$. Zur Entropie kommen $\ln 100 = 4.61\text{ nats}$ hinzu, sie beträgt dann $+3.36\text{ nats}$. Die Divergenz $D_{\textrm{KL}}(P \| Q)$ bleibt unverändert.

<!--en:ce4150a6-->
::: {.slide title="Informationstheoretische Größen im Überblick"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:4f747513-->
| Größe | Mathematische Definition | Wertebereich und Invarianz | Bedeutung für Deep Learning |

<!--en:96643d3c-->
| **Informationsgehalt** | $I(x) = -\log p(x)$ | $[0, \infty]$, additiv für unabhängige Ereignisse | Verlust eines einzelnen Beispiels als negative Log-Likelihood |

<!--en:f2ee06ff-->
| **Diskrete Entropie** | $H(P) = -\sum_{i=1}^k p_i \log p_i$ | $[0, \log k]$, maximal bei der Gleichverteilung $U$ | Irreduzible Untergrenze des Verlusts $H(Y \mid X)$, erwartete Anzahl von Fragen |

<!--en:74edb845-->
| **Differentielle Entropie** | $h(P) = -\int p(x)\log p(x)\,dx$ | $(-\infty, \infty)$, verschiebt sich um $\mathbb{E}[\log|\det Dg|]$ | Maximale Entropie $\tfrac12\log(2\pi e\sigma^2)$ der Gauß-Verteilung bei fester Varianz |

<!--en:41a5a6e6-->
| **KL-Divergenz** | $D_{\textrm{KL}}(P \| Q) = \mathbb{E}_P[\log(p/q)]$ | $[0, \infty]$, $0 \iff P = Q$, skaleninvariant | Zusätzlicher Überraschungswert als Strafe für die Abweichung des Modells |

<!--en:ac8a245c-->
| **Kreuzentropie** | $\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P \| Q)$ | $[H(P), \infty]$, streng echter logarithmischer Score | Übliches Trainingsziel für Klassifikatoren und große Sprachmodelle |

<!--en:dda6b1e2-->
| **Label-Smoothing** | $\mathbf{p}^\epsilon = (1-\epsilon)\mathbf{e}_y + (\epsilon/k)\mathbf{1}$ | Optimaler Logit-Abstand $\log(1 + k(1-\epsilon)/\epsilon)$ | Verhindert unbegrenzte Logit-Drift und übermäßige Sicherheit |

<!--en:ab9390b1-->
| **Destillationsverlust** | $\tau^2 D_{\textrm{KL}}(\mathbf{p}^{(\tau)} \| \mathbf{q}^{(\tau)})$ | Größenordnung des Gradienten $\mathcal{O}(1)$ für alle $\tau > 0$ | Überträgt das verborgene Wissen des Lehrers über falsche Klassen |
