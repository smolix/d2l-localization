<!--en:6e4c07ac-->
# Soft Actor-Critic

<!--en:36dcbe6a-->
Am Ende von :numref:`sec_deeprl` lernte ein Strategienetz (engl. policy network), ein Pendel aufzuschwingen und aufrecht zu halten. Dazu brauchte es allerdings einige Hunderttausend Umgebungsschritte. Jeder Batch von 1 600 Schritten diente einem einzigen Update. Danach verwarfen wir ihn. In einer Simulation kosten solche Schritte wenig. Bei einem Roboterarm kostet jeder Schritt Zeit, beansprucht die Hardware und birgt das Risiko eines Schadens. Wie kann ein Agent denselben Regler aus wenigen Tausend Schritten lernen?

<!--en:bb369864-->
Den größten Teil der Einsparung verdanken wir zwei bekannten Ideen. Beim Replay verwenden wir frühere Übergänge aus einem Puffer erneut, wie beim Training in :numref:`sec_dqn`. So kann jeder Übergang zu vielen Updates beitragen, sofern ein Kritiker aus Aktionen einer früheren Strategie lernen kann. Der pfadweise Gradient (engl. pathwise gradient) aus :numref:`sec_deeprl` passt dann den Akteur an, indem er den Kritiker durch die Aktion selbst differenziert. DDPG :cite:`Lillicrap.Hunt.Pritzel.ea.2016` verband beides mit einem deterministischen Akteur. Das Verfahren erwies sich als störanfällig, weil ein Akteur beim Aufstieg entlang des Kritikers auch dessen Schätzfehler ausnutzt. TD3 :cite:`Fujimoto.vanHoof.Meger.2018` führte das zu einem großen Teil auf die Maximierungsverzerrung (engl. maximization bias) aus :numref:`subsec_dqn-max-bias` zurück und trainierte zwei Kritiker. Dabei zählte die niedrigere Schätzung. Soft Actor-Critic (SAC) :cite:`Haarnoja.Zhou.Abbeel.ea.2018,Haarnoja.Zhou.Hartikainen.ea.2018` behält beide Verbesserungen bei und ergänzt sie um die Entropiebelohnung aus :numref:`sec_regularized`. Die Strategie (engl. policy) erhält eine Belohnung dafür, dass ihre Aktionen zufällig bleiben. Ein Akteur, der anhand seines eigenen Kritikers lernt, exploriert also weiter, statt sich auf die gerade bevorzugte Aktion festzulegen.

<!--en:7d62f33b-->
Kurz gesagt addiert SAC die Entropie der Strategie zur Belohnung, lernt den resultierenden Wert mit Off-Policy-Updates aus gespeicherten Übergängen und passt den Akteur an, indem er den Kritiker durch die vom Akteur gezogene Aktion differenziert. Zwei Einzelheiten verlangen besondere Sorgfalt. Für ein Drehmoment im Intervall $[-2, 2]$ brauchen wir eine beschränkte Strategie mit korrekter Dichte, weil der Gradient nun durch die Aktion fließt. Außerdem muss das Gewicht der Entropie zur Skala der Belohnung passen. SAC lernt dieses Gewicht deshalb mit. Wir leiten die Schritte zur Bewertung und Verbesserung für die entropieregularisierte Zielfunktion her, konstruieren und prüfen die beschränkte Strategie und setzen das Update zusammen. Dann messen wir, wie viele Schritte das Pendel braucht und wie zuverlässig die Vorhersagen des Kritikers sind.

<!--en:37de964a-->
Als durchgängiges Beispiel dient Pendulum aus :numref:`sec_deeprl`. Ein Pendel hängt an einer motorgetriebenen Achse. Der Agent beobachtet $(\cos\vartheta, \sin\vartheta, \dot\vartheta)$, wobei $\vartheta$ den Winkel zur aufrechten Lage bezeichnet, und legt ein Drehmoment $u \in [-2, 2]$ an. Jeder Schritt kostet $\vartheta^2 + 0.1\,\dot\vartheta^2 + 0.001\,u^2$. Hängt das Pendel unten in Ruhe, gilt $\vartheta = \pi$. Dann kostet jeder Schritt $\pi^2 \approx 9.9$ und damit mehr als in jeder anderen Ruhelage. Eine Episode dauert 200 Schritte. Gleichverteilte zufällige Drehmomente erzielen einen Ertrag (engl. return) von etwa $-1{,}200$. Ein Regler, der das Pendel aufschwingt und aufrecht hält, erreicht etwa $-200$ oder mehr. Jeder der folgenden Läufe umfasst höchstens 20 000 Umgebungsschritte. Wir trainieren jede Konfiguration mit drei Zufallsstartwerten. Der Skalenfaktor für die Aktion beträgt $c = 2$. Die angegebenen Spannweiten der Schrittzahlen und die Häufigkeiten über mehrere Läufe haben wir anhand von zwanzig Läufen je Konfiguration überprüft.

<!--en:c4abc28d-->
## Die Zielfunktion für maximale Entropie

<!--en:b2c3e389-->
### Entropie als Belohnung

<!--en:24fb225a-->
In :numref:`sec_regularized` hielten wir eine Strategie nahe an einer festen Referenzstrategie $\pi_{\textrm{ref}}$. Dazu zogen wir $\beta\, D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}})$ von der erwarteten Belohnung ab. Der Faktor $\beta > 0$ gewichtet die Divergenz gegenüber der Belohnung. Ist die Referenz eine Gleichverteilung über $|\mathcal{A}|$ Aktionen, gilt $D_{\textrm{KL}}(\pi \Vert \textrm{uniform}) = \log|\mathcal{A}| - H(\pi)$, wobei $H(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a)]$ die Entropie der Strategie ist. Der Strafterm ergibt dann einen Entropiebonus zuzüglich einer Konstanten, die keine Strategie ändern kann. Dasselbe gilt für das Drehmomentintervall von Pendulum. Die Gleichverteilung auf $[-2, 2]$ hat die Entropie $\log 4 \approx 1.39$ Nat, die größte aller Dichten auf diesem Intervall. Dabei tritt $\log 4$ an die Stelle von $\log|\mathcal{A}|$. SAC berücksichtigt den Bonus bei jedem Schritt:

<!--en:fa8bfbc3-->
Der Agent erhält in jedem besuchten Zustand $\beta$ Belohnungseinheiten je Nat Entropie. Die *Temperatur* $\beta$ hat also die Einheit Belohnung pro Nat. Wir behalten die Bezeichnung $\beta$ aus :numref:`sec_regularized` bei. Die Arbeiten zu SAC schreiben $\alpha$. In diesem Buch bezeichnet dieser Buchstabe die Schrittweite beim Lernen aus zeitlichen Differenzen.

<!--en:26d25071-->
Anders als der optionale Entropiebonus im Akteurverlust aus :numref:`sec_ppo` gehört dieser Term zur Zielfunktion. Er verändert das Optimum. Wo mehrere Aktionen annähernd gleich gut sind, bleibt die beste Strategie stochastisch. Bei kontinuierlichen Aktionen ist $H$ eine differentielle Entropie, also das Integral über $-\pi \log \pi$. Für eine Dichte, deren Masse auf einem Intervall der Länge kleiner als eins liegt, ist sie negativ. Reinforcement Learning mit maximaler Entropie ist älter als Deep Learning :cite:`Ziebart.Maas.Bagnell.ea.2008`. SAC ist das zugehörige Off-Policy-Akteur-Kritiker-Verfahren.

<!--en:70fa3051-->
Wie die anderen Verfahren dieses Kapitels wechselt SAC zwischen zwei Schritten der verallgemeinerten Strategieiteration (:numref:`fig_rl_gpi`). Die Bewertung nähert die Wertschätzungen denen der aktuellen Strategie an. Die Verbesserung verschiebt die Strategie hin zu den Aktionen, die nach diesen Schätzungen besser sind. Die Entropie verändert beide Schritte. Wie das geschieht, sehen wir schon ohne Netz an einem Zustand mit zwei Aktionen.

<!--en:73554caf-->
### Zwei Aktionen von Hand durchgerechnet

<!--en:d8bf959d-->
Betrachten wir einen Zustand mit zwei Aktionen und den Werten $Q = (1, 0)$. Es gelte $\beta = 0.5$. Wie :numref:`sec_regularized` zeigt, erhalten wir das KL-regularisierte Optimum, indem wir die Referenz mit $e^{Q/\beta}$ multiplizieren und erneut normieren. Sein Wert ist $\beta \log Z$. Für eine gleichverteilte Referenz gilt

<!--en:a0892825-->
Hier gilt $\pi^\star = (e^2, 1)/(e^2 + 1) = (0.88, 0.12)$. Die Entropie beträgt $0.37$ Nat und es gilt $\mathbb{E}_{\pi^\star}[Q] = 0.88$. Der entropieregularisierte Wert (engl. soft value) ist also $0.88 + 0.5 \cdot 0.37 = 1.06$, gleich $0.5 \log(e^2 + 1)$. Er übersteigt das harte Maximum $1$. Die exponentiell umgewichtete Strategie gibt $0.12$ an erwartetem Wert auf und gewinnt $0.18$ an Entropiebonus. Eine zweite Beobachtung ist für das Folgende wichtig. Der Klammerausdruck $Q(a) - \beta \log \pi^\star(a)$ liefert für beide Aktionen denselben Wert. Die Rechnungen $1 + 0.5 \cdot 0.13$ und $0 + 0.5 \cdot 2.13$ ergeben beide $1.06$. Im Optimum genügt eine gezogene Aktion, um den entropieregularisierten Wert fehlerfrei zu bestimmen. Die folgende Zelle vergleicht $\pi^\star$ mit der Gleichverteilungsstrategie und einer nahezu gierigen Strategie:

<!--en:2d7e293c-->
Die exponentiell umgewichtete Strategie erreicht $\beta \log Z = 1.063$. Die beiden anderen Strategien bleiben darunter. Die nahezu gierige Strategie erreicht $1.018$ und die Gleichverteilungsstrategie $0.847$. Bei ihnen liefert der Klammerausdruck für die beiden Aktionen unterschiedliche Werte. Bei der Gleichverteilungsstrategie beträgt die Differenz eine Einheit. Eine einzelne gezogene Aktion liefert daher eine verrauschte Schätzung des entropieregularisierten Werts. Nur im Optimum ist sie exakt.

<!--en:bcbbe7ee-->
### Entropieregularisierte Strategiebewertung

<!--en:50a48faf-->
Der entropieregularisierte Wert eines Zustands berücksichtigt neben der zukünftigen Belohnung die dort erzielte Entropie. Es gilt $V^\pi(s) = \mathbb{E}_{a \sim \pi}\big[Q^\pi(s, a) - \beta \log \pi(a \mid s)\big]$. Dabei enthält $Q^\pi(s, a)$ die Entropieboni aller späteren Zustände, aber nicht den von $s$ selbst. Der Kritiker aus :numref:`sec_actorcritic` lernte durch Regression auf das Einschrittziel $r + \gamma V(s')$, das sich in einem Endzustand auf $r$ reduziert. Der Kritiker von SAC lernt $Q$. Er braucht also den Wert des Folgezustands als Funktion von $Q$. Diesen liefert die obige Identität mit einer gezogenen Aktion. Sei $Q_{w^-}$ eine Kopie des Kritikers $Q_w$, die sich langsam ändert. Dieses Zielnetz aus :numref:`sec_dqn` verhindert, dass sich das Regressionsziel mit jedem Update verschiebt. Das Ziel lautet

<!--en:b9a64894-->
Der einzige neue Term, $-\beta \log \pi_\theta(\tilde{a}' \mid s')$, ist der Entropiebonus bei $s'$, geschätzt anhand der gerade gezogenen Aktion. Der Übergang $(s, a, r, s')$ stammt aus dem Replay-Puffer. Die Aktion $\tilde{a}'$ ziehen wir dagegen neu aus der aktuellen Strategie, denn $V^\pi$ ist ein Erwartungswert unter der zu bewertenden Strategie. Der Puffer liefert, was geschehen ist. Die Strategie liefert, was sie als Nächstes tun würde.

<!--en:496c6d05-->
Eine einzelne Ziehung schätzt den entropieregularisierten Wert der aktuellen Strategie. Dieser Schritt ist also eine entropieregularisierte Strategiebewertung. Das Log-Sum-Exp-Backup aus :numref:`sec_regularized` hingegen berechnet den Wert des exponentiell umgewichteten Optimums. Beide stimmen nur dann überein, wenn die aktuelle Strategie optimal ist. Dann hat der Klammerausdruck wie im Beispiel mit zwei Aktionen für jede Aktion denselben Wert. Die entropieregularisierte Bewertung ist eine gewöhnliche Strategiebewertung mit einer um den Entropiebonus erhöhten Belohnung. Daher gilt das Argument aus :numref:`sec_valueiter`. Jede Anwendung des Bewertungsoperators verkleinert den größten Fehler um den Faktor $\gamma$. Wiederholte Anwendung konvergiert gegen $Q^\pi$. Das Argument setzt beschränkte erweiterte Belohnungen voraus. Also müssen sowohl die Belohnungen als auch die Entropien beschränkt sein. Das gilt bei endlich vielen Aktionen und für kontinuierliche Strategien, deren Dichte beschränkt bleibt, wie wir hier annehmen.

<!--en:e534471b-->
### Entropieregularisierte Strategieverbesserung

<!--en:c738412e-->
Auch der Verbesserungsschritt folgt aus :numref:`sec_regularized`. Dort haben wir gezeigt, dass der Wert der regularisierten Zielfunktion einer beliebigen Strategie um eine mit der Temperatur gewichtete KL-Divergenz unter dem Optimum liegt:

<!--en:76e2ad1a-->
Dabei ist $\pi^\star \propto \pi_{\textrm{ref}}\, e^{r/\beta}$ das exponentiell umgewichtete Optimum. Ersetzen wir $r$ durch $Q(s, \cdot)$ und wählen eine gleichverteilte Referenz, so entspricht die Maximierung von $\mathbb{E}_\pi[Q] + \beta H(\pi)$ in einem Zustand der Minimierung von $D_{\textrm{KL}}\big(\pi \Vert e^{Q/\beta}/Z\big)$. Die temperaturgewichtete Divergenz und der Zielfunktionswert ergänzen sich zur Konstanten $\beta \log Z$, die nicht von $\pi$ abhängt. Ein Strategienetz kann die umgewichtete Verteilung meist nicht darstellen. SAC nähert die Strategie daher dem Mitglied ihrer Familie an, das ihr gemessen an dieser Divergenz am nächsten liegt. Das ist die *Projektion bezüglich der umgekehrten KL-Divergenz* (engl. reverse-KL projection). In dieser Richtung bestraft die Divergenz Wahrscheinlichkeitsmasse dort, wo $e^{Q/\beta}$ klein ist. Hat die umgewichtete Verteilung zwei getrennte Gipfel, konzentriert sich eine Gauß-Verteilung deshalb auf einen davon, statt sich über beide auszubreiten (:numref:`subsec_kl_direction`). Schreiben wir die Aktion als differenzierbare Funktion des Rauschens, erhalten wir den Akteurverlust

<!--en:ae6c4e07-->
mit $z \sim \mathcal{N}(0, I)$ und Zuständen aus dem Replay-Puffer $\mathcal{D}$. Der Verlust ist der negative entropieregularisierte Wert einer neu gezogenen Aktion. Seine Minimierung erhöht also den Kritikerwert und die Entropie gemeinsam. Die Stauchung (engl. squashing) $c \tanh$ hält die Aktion im zulässigen Bereich. Wir behandeln sie im nächsten Teil. Der Gradient des Verlusts ist der pfadweise Gradient aus :numref:`sec_deeprl`. Für eine gaußverteilte Aktion $a = \mu + \sigma z$ lautet er $\nabla_\mu \mathbb{E}[Q(a)] = \mathbb{E}[Q'(\mu + \sigma z)]$. Er differenziert durch die Aktion, statt einen Score zu gewichten. Im dortigen Beispiel betrug seine Varianz bei einer einzelnen Ziehung $1.0$, gegenüber $21.75$ für die Score-Funktion. Dazu muss der Kritiker bezüglich der Aktion differenzierbar sein. Bei diskreten Aktionen berechnet man den Erwartungswert stattdessen durch Summation über die Aktionen.

<!--en:0e967b2c-->
Bei exakter Maximierung in jedem Zustand würde der Verbesserungsschritt den entropieregularisierten Wert überall erhöhen oder beibehalten.

<!--en:d1c07b76-->
**Satz.** Belohnungen und Entropien seien beschränkt. Die Familie $\Pi$ enthalte $\pi_{\textrm{old}}$ und erlaube es wie eine Tabelle, die Strategie in jedem Zustand unabhängig zu wählen. Maximiert $\pi_{\textrm{new}}(\cdot \mid s)$ den Ausdruck $\mathbb{E}_{a \sim \pi}\big[ Q^{\pi_{\textrm{old}}}(s, a) \big] + \beta H(\pi(\cdot \mid s))$ über $\Pi$ in jedem Zustand $s$, dann gilt überall $V^{\pi_{\textrm{new}}} \geq V^{\pi_{\textrm{old}}}$.

<!--en:df01ce48-->
**Beweis.** Sei $H_t$ die Entropie von $\pi_{\textrm{new}}$ bei $s_t$. Wegen $\pi_{\textrm{old}} \in \Pi$ folgt aus der Optimalität bei $s_0 = s$ und anschließend aus $Q^{\pi_{\textrm{old}}}(s, a) = r(s, a) + \gamma\, \mathbb{E}[V^{\pi_{\textrm{old}}}(s_1)]$ die Ungleichung

<!--en:481a4431-->
Wenden wir dieselbe Ungleichung auf $V^{\pi_{\textrm{old}}}(s_1)$ an, dann auf $V^{\pi_{\textrm{old}}}(s_2)$ und so weiter, insgesamt $n$-mal, so erhalten wir $V^{\pi_{\textrm{old}}}(s) \leq \mathbb{E}_{\pi_{\textrm{new}}}\big[ \sum_{t < n} \gamma^t (r_t + \beta H_t) \big] + \gamma^n\, \mathbb{E}\big[V^{\pi_{\textrm{old}}}(s_n)\big]$. Die Werte sind beschränkt. Daher verschwindet der letzte Term für $n \to \infty$ und die Summe konvergiert gegen $V^{\pi_{\textrm{new}}}(s)$. $\blacksquare$

<!--en:700227f8-->
SAC macht für jeden Verlust einen Gradientenschritt, während der Kritiker noch lernt. Der Satz beschreibt somit den exakten Algorithmus. Die Trainingsschleife führt nur eine Näherung aus. Den vollständigen Beweis geben :citet:`Haarnoja.Zhou.Abbeel.ea.2018` in Anhang B.

<!--en:757e46a5-->
## Die gestauchte Gauß-Strategie

<!--en:45adbc80-->
### Clipping und der pfadweise Gradient

<!--en:32bbb7d0-->
Die Gauß-Strategie aus :numref:`sec_deeprl` zog ein unbeschränktes Drehmoment. Die Umgebung begrenzte es durch Clipping auf $[-2, 2]$. Dem Score-Funktionsschätzer schadete das nicht, weil er nie durch die Aktion differenziert. Der pfadweise Schätzer von SAC tut das jedoch. Dadurch verursacht Clipping zwei Probleme. Außerhalb des Intervalls ist die Ableitung der Clipping-Funktion null. Eine Ziehung jenseits einer Grenze liefert dem Akteur also keinen Gradienten, selbst wenn das Drehmoment an der Grenze für diesen Zustand optimal ist. Außerdem verschiebt Clipping die gesamte Wahrscheinlichkeitsmasse jenseits jeder Grenze auf die Grenze selbst. Die resultierende Verteilung hat dort keine Dichte. Damit ist $\log \pi$ in :eqref:`eq_sac_target` und :eqref:`eq_sac_actor` undefiniert. Eine beschränkte Verteilung, die durch eine glatte, umkehrbare Abbildung aus einer Gauß-Verteilung entsteht, löst beide Probleme. :numref:`fig_rl_clip_vs_tanh` vergleicht Clipping mit der Abbildung, die wir verwenden, nämlich $2 \tanh u$.

<!--en:2c4fc1ea-->
![Clipping und Stauchung eines gaußverteilten Werts $u$ auf $[-2, 2]$. (a) Beim Clipping ist die Steigung jenseits der Grenzen null. Eine Ziehung in diesem Bereich liefert dem Akteur keinen Gradienten. Die Funktion $2 \tanh u$ hat überall positive Steigung. (b) Für $u \sim \mathcal{N}(0.7, 0.8^2)$ verschiebt Clipping die Wahrscheinlichkeitsmasse oberhalb der oberen Grenze, etwa $0.05$, auf die einzelne Aktion $a = 2$. Dort besitzt die Verteilung keine Dichte. Die Stauchung liefert eine Dichte auf dem ganzen Intervall.](../img/mdl-rl-clip-vs-tanh.svg)

<!--en:53ca4a82-->
### Variablentransformation mit tanh

<!--en:b8dfebb7-->
Wir ziehen $u \sim \mathcal{N}\big(\mu_\theta(s), \sigma_\theta(s)^2\big)$ und stauchen den Wert gemäß $a = c \tanh u$. Die Abbildung ist glatt und streng monoton steigend. Ein kurzes Intervall um $u$ und sein Bild um $a$ haben also dieselbe Wahrscheinlichkeit, doch die Länge des Bilds ist um den Faktor $da/du = c\,(1 - \tanh^2 u)$ gestaucht. Die Dichte von $a$ ergibt sich daher koordinatenweise aus der Gauß-Dichte dividiert durch diesen Faktor:

<!--en:ea588970-->
Jeder Summand korrigiert die Gauß-Log-Dichte um die Stauchung einer Koordinate. :numref:`fig_rl_squashed_density` zeigt den Effekt für einen bestimmten Mittelwert und eine bestimmte Streuung. Nahe $a = 2$ drängt die Stauchung den gesamten oberen Rand der Verteilung von $u$ in ein kurzes Aktionsintervall. Daher steigt die Dichte dort an. Ohne Korrektur fehlt diese Verdichtung und das Integral des Ausdrucks beträgt $1.13$ statt $1$.

<!--en:155bcc5f-->
![Die gestauchte Gauß-Verteilung. (a) Dichte von $u \sim \mathcal{N}(0.7, 0.8^2)$. (b) Dichte der Aktion $a = 2\tanh u$. Mit dem Jacobi-Term aus :eqref:`eq_tanh_logdet` hat die Dichte das Integral 1 und steigt nahe $a = 2$ an. Dort staucht die Transformation den oberen Rand der Verteilung von $u$ auf ein kurzes Intervall. Ohne den Term entspricht der Ausdruck der an der Stelle $\operatorname{arctanh}(a/2)$ ausgewerteten Gauß-Dichte. Ihr Integral über die Aktionen beträgt 1.13 und die Häufung fehlt.](../img/mdl-rl-squashed-density.svg)

<!--en:2ace714b-->
Der Term $\log c$ ändert den Gradienten des Akteurs nicht. Er verschiebt aber jede Log-Dichte und damit auch die ausgewiesene Entropie und den Term $\beta \log \pi$ im Ziel um $\log 2 \approx 0.69$ Nat. Wir behalten ihn bei, damit die angegebenen Entropien zur tatsächlichen Drehmomentverteilung gehören.

<!--en:be13197f-->
Als Strategienetz dient der `GaussianHead` aus :numref:`sec_deeprl`. Ein gemeinsamer *Stamm* (engl. trunk) berechnet Merkmale. Eine lineare Schicht bildet sie auf den Mittelwert $\mu_\theta(s)$ ab. Mit `state_std=True` liefert eine zweite lineare Schicht $\log \sigma_\theta(s)$, begrenzt auf $[-5, 2]$. SAC braucht eine zustandsabhängige Streuung, weil der Entropiebonus in jedem Zustand anfällt. Beim Balancieren in aufrechter Lage braucht das Pendel eine schmale Strategie. Beim Aufschwingen kann es eine breite nutzen. Ein gemeinsames $\sigma$ kann nicht beides leisten. Für den Stamm verwenden wir zwei ReLU-Schichten mit je 64 Einheiten, genau wie bei den Kritikern weiter unten. Sie ersetzen die einzelne tanh-Schicht aus :numref:`sec_deeprl`. Veröffentlichte SAC-Implementierungen verwenden denselben Aufbau mit 256 Einheiten. In unseren Läufen greift die Begrenzung nie, denn $\log \sigma$ bleibt über etwa $-3.5$. Ohne sie bleiben die Ergebnisse gleich. Wir behalten sie zur Absicherung bei. Sie hält $\sigma$ von null fern, damit $\log\pi$ im Ziel bei schrumpfender Streuung nicht divergiert. Die Entropie der gestauchten Aktion beschränkt sie allerdings nicht. Ein Mittelwert weit im Sättigungsbereich drückt diese Entropie weiter nach unten. Gängige Implementierungen begrenzen auf $[-20, 2]$.

<!--en:2f3ebfc4-->
`SquashedGaussianPolicy` kapselt den Ausgabekopf. Für die Verlustberechnung zieht `sample` den Wert $u = \mu + \sigma z$ und gibt die gestauchte Aktion mit ihrer Log-Wahrscheinlichkeit zurück. Beide sind differenzierbar. Die Methode `log_prob_u` berechnet :eqref:`eq_tanh_logdet` aus dem vor der Stauchung gespeicherten Wert $u$. Daher kommt der Arkustangens hyperbolicus auf dem differenzierbaren Pfad nicht vor. Für die Umgebung zieht `act` eine Aktion als NumPy-Array. Für die Messungen am Ende dieses Abschnitts liefert `act_logp` eine solche Aktion zusammen mit ihrer Log-Wahrscheinlichkeit. Beide beziehen den gezogenen Wert $u$ von `_draw`. Die Methode `act_greedy` liefert den gestauchten Mittelwert $c \tanh \mu_\theta(s)$. Wir speichern die Klasse in der Bibliothek, weil wir sie im nächsten Abschnitt erneut verwenden.

<!--en:579c1fce-->
`_draw`, das sich `act` und `act_logp` teilen, sowie `act_greedy` werten den Ausgabekopf über `d2l.forward_np` aus. Diese Funktion kompiliert pro Netz einen Vorwärtsdurchlauf und liest bei jedem Aufruf die aktuellen Parameter (:numref:`sec_utils`). So kann der Akteur einzelne Zustände verarbeiten, ohne den Durchlauf nach jedem Update neu zu kompilieren.

<!--en:908645f6-->
In `log_prob_u` summieren wir sowohl die Gauß-Terme als auch die Jacobi-Terme über die Aktionskoordinaten, bevor wir über den Batch mitteln. Summieren wir die einen und mitteln die anderen, stimmt das Ergebnis für die einzige Koordinate von Pendulum, ist aber für jeden höherdimensionalen Aktionsraum falsch.

<!--en:95dc4a84-->
### Eine numerisch stabile logarithmierte Determinante

<!--en:9703249c-->
Die direkte Formel `log(1 - tanh(u)**2)` subtrahiert zwei nahezu gleiche Zahlen, sobald $\tanh u$ nahe bei $1$ liegt. Sie versagt, wenn float32 den Wert $\tanh u$ auf exakt $1$ rundet. Wegen $1 - \tanh^2 u = 4 e^{-2u} / (1 + e^{-2u})^2$ ergibt sich durch Logarithmieren

<!--en:98ad3da1-->
Diese Identität gilt für jedes $u$ und kommt ohne die Subtraktion nahezu gleicher Zahlen aus. `log_prob_u` verwendet sie. Die folgende Rechnung in float32 vergleicht die direkte Formel, eine übliche Absicherung durch Addition von $10^{-6}$ und die stabile Formel:

<!--en:afe5bd16-->
Schon bei $u = 8$ weicht die direkte Formel leicht ab, weil float32 nur noch wenige gültige Stellen für $1 - \tanh^2 u$ hat. Sie liefert $-\infty$ ab $u = 10$. Die Absicherung vermeidet den unendlichen Wert, liegt aber bei $u = 8$ um mehr als ein Nat zu hoch. Sie bleibt bei $\log 10^{-6} = -13.8155$, sobald $u = 10$ erreicht ist, und erfasst die weitere Sättigung nicht mehr. Die ausgewiesene Entropie und der Entropieterm aus :eqref:`eq_sac_actor` sind dann falsch, während die Ertragskurve unauffällig aussieht.

<!--en:6d8cd9de-->
Die Normierung liefert eine zweite, unabhängige Prüfung der Variablentransformation. Wir werten die Dichte auf einem feinen Gitter von Aktionen aus. Dazu kehren wir die Stauchung an den Gitterpunkten um und integrieren :eqref:`eq_tanh_logdet` mit der Trapezregel, einmal mit und einmal ohne Jacobi-Term:

<!--en:1354c8cb-->
Mit Korrektur ergibt das Integral der Dichte bei beiden Einstellungen $1.000000$. Ohne sie sind die Integrale $1.65$ und $1.13$. Dieser Fehler würde den Ertrag auf Pendulum kaum verändern, aber jede ausgewiesene Entropie verfälschen. Die Prüfung deckt ihn ohne Training auf. Das Gitter endet im Abstand $10^{-6}$ von den Grenzen $\pm c$, an denen $\operatorname{arctanh}$ unendlich wird. Darum verwenden wir für die Prüfung mäßig große Werte von $\sigma$.

<!--en:cb93618d-->
## Der Algorithmus

<!--en:359b81a7-->
### Zwei Kritiker und Zielnetze mit Polyak-Mittelung

<!--en:8afdde55-->
Das Maximum verrauschter Schätzungen ist nach oben verzerrt (:numref:`subsec_dqn-max-bias`). Bei zwei unabhängigen Schätzungen desselben Werts mit mittelwertfreiem Rauschen der Standardabweichung $\sigma$ liegt das Maximum im Mittel um etwa $0.56\sigma$ über dem wahren Wert. Ein Akteur, der dem Gradienten seines Kritikers folgt, tut dasselbe bei kontinuierlichen Aktionen. Er bewegt sich zu Aktionen, deren Wert der Kritiker zufällig überschätzt. Durch Bootstrapping geht der Fehler in das nächste Ziel ein. Wie TD3 trainiert SAC zwei unterschiedlich initialisierte Kritiker $Q_{w_1}$ und $Q_{w_2}$. Ihr punktweises Minimum geht in das Ziel ein,

<!--en:a1d4078b-->
und ebenso in den Akteurverlust :eqref:`eq_sac_actor`. Dort ersetzt $\min_j Q_{w_j}$ den Ausdruck $Q_w$. Beide Kritiker lernen durch Regression auf dasselbe Ziel $y$. Wo sie voneinander abweichen, zählt die niedrigere Schätzung. Das Minimum ist also bewusst pessimistisch. Am Ende dieses Abschnitts messen wir, was dieser Pessimismus kostet und was er verhindert.

<!--en:f3f753b7-->
Jeder Kritiker hat wie bei DQN eine Zielkopie $w_j^-$. Anders als in :numref:`sec_dqn` überschreiben wir sie jedoch nicht alle 250 Umgebungsschritte. Nach jedem Update legt sie einen kleinen Bruchteil $\nu$ des Wegs zu den aktuellen Gewichten zurück:

<!--en:aa298a67-->
Diese *Polyak-Mittelung* (engl. Polyak averaging) ist ein exponentiell gleitender Mittelwert mit einer Halbwertszeit von $\ln 2/\nu \approx 139$ Updates. Das Regressionsziel ändert sich also langsam, statt erst stillzustehen und dann zu springen. Nur die Kritiker haben Zielkopien. Die Aktion $\tilde{a}'$ im Ziel stammt aus der aktuellen Strategie. Das Ziel braucht keine zusätzliche Glättung dieser Aktion, weil eine stochastische Strategie durch ihre Streuung selbst über die Zielwerte mittelt. :numref:`fig_rl_sac_update` zeigt, wie die Teile eines Updates zusammenwirken.

<!--en:81184639-->
![Ein SAC-Update. Obere Zeile: Für das entropieregularisierte Ziel $y$ ziehen wir bei der aktuellen Strategie eine neue Aktion $\tilde a'$ im Zustand $s'$. Vom Minimum der beiden Zielkritiker an dieser Stelle ziehen wir $\beta \log \pi_\theta(\tilde a' \mid s')$ ab. Mittlere Zeile: Die Kritiker passen $Q_{w_j}(s, a)$ durch Regression an $y$ an. Untere Zeile: Der Akteurverlust wirkt über die Kritiker und die neu gezogene Aktion $\tilde a$ auf den Akteur (rot, pfadweiser Gradient). Die Temperatur hat einen eigenen Verlust mit derselben Log-Wahrscheinlichkeit. Der gestrichelte orangefarbene Pfeil zeigt das Polyak-Update. Nur diese Verbindung ändert die Zielkritiker.](../img/mdl-rl-sac-update.svg)

<!--en:85cdf46e-->
### Automatische Temperaturwahl

<!--en:ab4f4804-->
Ein festes $\beta$ legt den Umrechnungskurs zwischen Belohnung und Entropie fest. Welcher Kurs passt, hängt von der Einheit der Belohnung ab. Multiplizieren wir jede Belohnung mit $10$, hat das auf die optimale Strategie dieselbe Wirkung wie die Division von $\beta$ durch $10$ (Aufgabe :exref:`ex_sac-quadratic-critic`). Eine für eine Aufgabe abgestimmte Temperatur passt also schon dann nicht mehr, wenn wir dieselbe Aufgabe in anderen Einheiten messen. Die zweite Arbeit zu SAC :cite:`Haarnoja.Zhou.Hartikainen.ea.2018` gibt stattdessen die Entropie vor. Gesucht ist die beste Strategie, deren Entropie in den besuchten Zuständen im Mittel mindestens den Zielwert $\bar H$ erreicht,

<!--en:32564c77-->
wobei $\beta$ als Lagrange-Multiplikator der Nebenbedingung dient. Üblicherweise wählt man $\bar H = -\dim \mathcal{A}$, hier also $-1$ Nat. Wir lernen den Multiplikator durch Gradientenabstieg auf

<!--en:4ae23dac-->
Die Ableitung nach $\log\beta$ beträgt $\hat H - \bar H$. Sie gibt an, um wie viel die aus dem Batch geschätzte Entropie $\hat H = -\mathbb{E}[\log \pi_\theta]$ den Zielwert übersteigt. Ein Schritt senkt also $\beta$, solange die Strategie zufälliger ist als gewünscht. Der Verzicht auf Zufälligkeit kostet dann weniger. Ist die Entropie zu klein, erhöht der Schritt $\beta$. Wir lernen $\log\beta$, damit $\beta$ positiv bleibt. Wir beginnen mit $\beta = 1$, dem Standardwert gängiger Implementierungen, und verwenden Adam mit derselben Lernrate wie für die Netze, nämlich $3 \times 10^{-4}$. Solange der Gradient sein Vorzeichen behält, verschiebt Adam einen Parameter je Schritt um etwa eine Lernrate. Bleibt die Entropie über dem Zielwert, sinkt $\log \beta$ also um etwa $3 \times 10^{-4}$ pro Update. Für 6 000 Updates ergibt das die Vorhersage $\beta \approx e^{-1.8} \approx 0.17$, die wir mit dem gemessenen Verlauf vergleichen.

<!--en:47b7e476-->
Schon die erste Arbeit zu SAC trainierte zwei Kritiker, zusammen mit einem separaten Netz für den Zustandswert. Wie in der zweiten Arbeit kommt unser Algorithmus ohne Wertnetz aus und lernt seine Temperatur. Diese Version verwenden auch die gängigen Implementierungen.

<!--en:a22f4fb4-->
### Update und Trainingsschleife

<!--en:279d12c3-->
Der Replay-Puffer aus :numref:`sec_dqn`, `d2l.ReplayBuffer`, speichert Übergänge $(s, a, r, s', \textrm{term})$ in vorab reservierten Arrays. Die Methode `add` speichert einen Übergang und überschreibt den ältesten, sobald der Puffer voll ist. Die Methode `sample` zieht gleichverteilt einen Batch und gibt ihn als `d2l.Batch` zurück. Die Aktionsspalte enthält ganze Zahlen. `ReplayBufferC` erweitert nur diese Spalte auf Vektoren von Gleitkommazahlen. Wie die Strategie speichern wir auch diese Klasse, weil wir sie im nächsten Abschnitt erneut verwenden.

<!--en:a3f33be1-->
SAC lernt Off-Policy ohne Wichtigkeitsquotienten. Das Kritikerziel :eqref:`eq_sac_target_twin` hängt nicht von der Strategie ab, die den Übergang erzeugt hat, denn wir ziehen die nächste Aktion neu. Auch für den Akteurverlust ziehen wir eigene Aktionen. Replay verändert dennoch die Zustandsverteilung, über die wir die Verluste mitteln. Die Zustände im Puffer stammen aus Besuchen früherer Strategien. Kein Quotient von Aktionswahrscheinlichkeiten korrigiert diesen Unterschied.

<!--en:deb748b4-->
Der Container `ActorCritic` aus :numref:`sec_deeprl` enthält eine Strategie mit einem Zustandswertnetz und dessen Optimierer. Für SAC passt er nicht, denn SAC braucht Aktionswertkritiker, deren Zielkopien und eine Temperatur. `SoftActorCritic` enthält den Akteur, je eine Liste der Kritiker und ihrer Zielkopien, $\log\beta$ sowie je einen Adam-Optimierer für den Akteur, die Kritiker und $\log\beta$. Mit `beta=None` lernen wir die Temperatur ausgehend von $\beta = 1$. Eine Zahl legt sie fest. Die Methode `beta` liefert die Temperatur als skalares Array. Die Methode `min_q` berechnet das Minimum über eine Liste von Kritikern, standardmäßig über die aktuell trainierten Kritiker. Für die abschließenden Messungen erledigt `q_np` dasselbe für NumPy-Eingaben.

<!--en:d213fd0f-->
`make_critic` erzeugt einen Kritiker. Das Netz bildet den verketteten Zustands- und Aktionsvektor mit zwei ReLU-Schichten zu je 64 Einheiten auf eine Zahl ab. `make_sac` erzeugt einen Agenten, dessen Zielkopien zu Beginn mit ihren Kritikern übereinstimmen.

<!--en:3f9f2d09-->
Ein Update folgt :numref:`fig_rl_sac_update`. Drei Funktionen berechnen die Größen für die Verluste. `soft_target` berechnet das Ziel :eqref:`eq_sac_target_twin` an einer neu aus der aktuellen Strategie gezogenen Aktion. Für die Regression ist das Ziel ein Datenwert. Daher fließt kein Gradient hinein. `critic_loss` passt alle Kritiker durch Regression an dieses eine Ziel an. `actor_loss` entspricht :eqref:`eq_sac_actor` mit dem Minimum der Kritiker an einer neu gezogenen, reparametrisierten Aktion. Die Funktion liefert außerdem die Log-Wahrscheinlichkeiten für den Temperaturverlust.

<!--en:604d6f23-->
`sac_step` führt das Update aus. Die Kritiker machen einen Schritt für `critic_loss`, dann macht der Akteur einen Schritt für `actor_loss`. Lernt der Agent $\beta$, folgt für die Temperatur ein Schritt für $J(\beta)$ mit den Log-Wahrscheinlichkeiten des Akteurs. Schließlich verschieben sich die Zielkritiker um den Anteil $\nu$, im Code `polyak` genannt. Der Schritt gibt den Batchmittelwert von $\log\pi_\theta$ zurück. Dessen negativer Wert schätzt die Entropie der Strategie. Das Ergebnis bleibt auf dem Gerät, damit die Trainingsschleife nicht auf jedes Update warten muss.

<!--en:d9cbc8cc-->
`descend` macht einen Optimierungsschritt für einen Verlust. `lerp_` verschiebt jedes Zielgewicht um den Bruchteil `polyak` der Differenz zum entsprechenden Gewicht des Kritikers.

<!--en:7fdf2c2a-->
Wir kompilieren `sac_step` mit `nnx.jit`. Die Funktion erhält daher die Arrays des Batchs und einen Zufallsschlüssel anstelle eines `d2l.Batch`. `nnx.grad` differenziert jeden Verlust nach seinem ersten Argument, also nach den Kritikern oder nach dem Akteur. Das Ziel `y` geht als Datenwert in den Kritikerverlust ein. Die Form des Batchs bleibt gleich. Daher genügt es, den Schritt einmal zu kompilieren. `optax.incremental_update` berechnet das Polyak-Mittel aus den Parametern jedes Kritikers und seiner Kopie.

<!--en:20dedf9d-->
`train_sac` lässt einen Agenten auf Pendulum laufen. Während der ersten `warmup` Schritte ziehen wir die Drehmomente gleichverteilt. Danach wählt der Akteur die Aktionen, und auf jeden Umgebungsschritt folgen `utd` Updates. Dieses *Verhältnis von Updates zu Daten* (engl. update-to-data ratio) gibt die Zahl der Gradientenschritte pro Umgebungsschritt an. Die Schleife speichert `terminated` anstelle von `truncated`. Pendulum erreicht nie einen Endzustand. Seine Episoden enden nur wegen der Zeitgrenze von 200 Schritten. Das gespeicherte Kennzeichen ist also immer null und der Kritiker verwendet über den Abbruch hinweg Bootstrapping, wie es eine Zeitgrenze erfordert (:numref:`sec_deeprl`). Am Ende jeder Episode liefert die Schleife die Schrittzahl, den Episodenertrag, den über die Updates der Episode gemittelten Wert von $-\log\pi_\theta$ als Entropieschätzung und das aktuelle $\beta$. Wir übergeben das Verhältnis von Updates zu Daten und den Puffer als Argumente. So nutzen alle folgenden Experimente dieselbe Schleife. Sie entspricht der Off-Policy-Schleife von `train_dqn` aus :numref:`sec_dqn` mit kontinuierlichen Aktionen. `make_update` bindet `sac_step` an den Agenten und wandelt einen gezogenen `d2l.Batch` in Arrays um. Die Funktion setzt auch den Zufallsstartwert für das Rauschen bei der Aktionswahl. So liefert ein Lauf allein und nach anderen Läufen dasselbe Ergebnis.

<!--en:90b2f004-->
`nnx.cached_partial` bindet den Agenten einmal an den kompilierten Schritt. So muss die Schleife die Module des Agenten nicht bei jedem Aufruf durchlaufen.

<!--en:3f3d8ba7-->
`run_arm` trainiert eine Konfiguration mit mehreren Zufallsstartwerten und bewahrt die Agenten für spätere Messungen auf. Der Puffer fasst jeden Übergang eines Laufs. Wir beginnen mit zwei Konfigurationen mit je zwei Kritikern und gelernter Temperatur. Die erste macht 20 000 Umgebungsschritte mit einem Update pro Schritt, die zweite 4 000 Schritte mit zehn Updates pro Schritt.

<!--en:9e1dbd17-->
## Dateneffizienz und Kalibrierung

<!--en:f439845f-->
### Ertrag in Abhängigkeit von den Umgebungsschritten

<!--en:e83ebda9-->
Die Grafik zeigt den über die letzten fünf Episoden gemittelten Episodenertrag in Abhängigkeit von der Zahl der Umgebungsschritte. Die Markierung bei $-200$ entspricht dem Niveau eines Reglers, der das Pendel aufschwingt und aufrecht hält. Für jeden Zufallsstartwert einer Versuchsvariante gibt `crossings` den ersten Schritt an, bei dem der Mittelwert über fünf Episoden dieses Niveau erreicht. Den Schritt bestimmt `d2l.steps_to` aus :numref:`sec_actorcritic`. Der Wert `inf` kennzeichnet einen Lauf, der die Schwelle nie erreicht.

<!--en:cd024928-->
Bei einem Update pro Umgebungsschritt überschreitet jeder Lauf die Schwelle $-200$ nach etwa fünf- bis elftausend Umgebungsschritten. REINFORCE mit gelernter Baseline erreichte dasselbe Niveau bei derselben Aufgabe erst nach einigen Hunderttausend Schritten (:numref:`sec_deeprl`), also ein bis zwei Größenordnungen später. Drei Änderungen erklären den Unterschied. Mit Replay ermöglicht jeder Umgebungsschritt einen Gradientenschritt. REINFORCE brauchte für ein Update 1 600 Schritte. Durch das Bootstrapping-Ziel lernt der Kritiker aus einzelnen Übergängen statt aus vollständigen Erträgen. Und der pfadweise Gradient vermittelt dem Akteur direkt, was der Kritiker über die Abhängigkeit des Werts vom Drehmoment gelernt hat.

<!--en:f5188bbd-->
Bevor die Kurven steigen, fallen sie. Episoden mit zufälligen Drehmomenten in der Aufwärmphase erzielen einen Ertrag von etwa $-1{,}200$. Sobald der Akteur übernimmt, sinkt der Ertrag zwischen Schritt 1 500 und 3 000 zunächst auf etwa $-1{,}600$, also unter das Zufallsniveau. Erst danach steigt er. Zwei Verhaltensweisen könnten so hohe Kosten verursachen. Der Akteur könnte das Pendel schnell drehen und damit hohe Geschwindigkeitskosten verursachen. Oder er könnte es unten halten, wo die Winkelkosten am höchsten sind. Um das zu unterscheiden, messen wir, wie sich das Pendel unter einer Strategie bewegt. Die Funktion `behavior` führt zehn Episoden aus. Sie liefert den mittleren Ertrag, den Mittelwert von $\cos\vartheta$ (unten $-1$, oben $1$) und den mittleren Betrag der Winkelgeschwindigkeit $|\dot\vartheta|$. Als Referenz dienen drei feste Regler. Der erste übt kein Drehmoment aus. Der zweite treibt das Pendel mit dem Drehmoment $2\,\mathrm{sign}(\dot\vartheta)$ immer in Bewegungsrichtung an. Der dritte dämpft mit $-2\tanh(2\dot\vartheta)$ jede Bewegung.

<!--en:235802e8-->
Wir vergleichen diese Regler für jeden Zufallsstartwert mit dem Akteur nach 2 000 Schritten, mitten im Ertragstief, sowie mit einem fertig trainierten Agenten. Da jeder Lauf seinen eigenen Zufallsstartwert setzt, wiederholt ein erneutes Training über 2 000 Schritte genau den Anfang des entsprechenden Laufs von oben. Wir messen das Verhalten bei der mittleren Aktion des Akteurs $c \tanh \mu_\theta(s)$:

<!--en:26768077-->
Nach 2 000 Schritten hält jeder Akteur das Pendel fast in Ruhe nahe der unteren Lage. Dabei liegt $\cos\vartheta$ bei etwa $-0.9$ und der Betrag der Winkelgeschwindigkeit bei etwa einem Radiant pro Sekunde oder weniger. Der Akteur verhält sich wie der dämpfende Regler, der unter den drei Referenzen am schlechtesten abschneidet. Der trainierte Agent hält das Pendel dagegen nahe der oberen Lage. Der antreibende Regler erzielt einen ähnlich niedrigen Ertrag wie der dämpfende, aber mit einer ganz anderen Bewegung. Das Pendel dreht sich schnell und $\cos\vartheta$ liegt im Mittel nahe null. Dieses Verhalten zeigt keiner der Akteure. Der Akteur lernt also zunächst, die Schwingung zu dämpfen. Dadurch bleibt das Pendel in seiner teuersten Lage hängen. Erst später lernt er, Energie zuzuführen und das Pendel aufzuschwingen. In allen zwanzig Läufen hielt die mittlere Aktion des Akteurs das Pendel nach 2 000 Schritten auf dieselbe Weise unten. Eine plausible Erklärung haben wir nicht gesondert geprüft. Ein wenig trainierter Kritiker berücksichtigt erst die Kosten der nächsten paar Schritte. Über so wenige Schritte verändert eine Aktion vor allem die Geschwindigkeits- und Drehmomentkosten. Daher erscheint Dämpfung am günstigsten.

<!--en:bba4361e-->
Für die zweite Kurve machen wir zehn Updates pro Umgebungsschritt. Sie überschreitet $-200$ nach 2 400 bis 4 000 Umgebungsschritten. Bei einem Update pro Schritt sind es 5 400 bis 11 200. Der Bedarf sinkt also etwa um den Faktor 2.5. Das Ertragstief tritt schon nach rund 1 200 Schritten auf und ist nach 1 600 Schritten vorbei. Die eingesparten Umgebungsschritte kosten Rechenzeit. Bis zur Schwelle braucht die Variante mit mehr Updates etwa dreimal so viele Gradientenschritte. Im gesamten Lauf über 4 000 Umgebungsschritte sind es 30 000 Gradientenschritte, etwa das 1.6-Fache der 19 000 im Lauf über 20 000 Umgebungsschritte. Mehr Updates sparen also Interaktionen auf Kosten von Rechenaufwand. Für einen Roboterarm ist dieser Tausch günstig. Unterhalb von 2 400 Schritten lässt sich so kaum noch etwas gewinnen. Die ersten 1 000 Schritte sind zufällig, und ein Mittelwert über fünf Episoden kann $-200$ frühestens bei Schritt 2 000 erreichen. Bei schwierigeren Aufgaben stößt dieser Tausch an zwei Grenzen. Die erste ist die weiter unten untersuchte Überschätzung. Die zweite ist eine *Übergewichtung früher Erfahrungen* (engl. primacy bias). Lernt ein Netz sehr oft aus seinen ersten wenigen Übergängen, passt es sich so eng an sie an, dass es aus späteren Daten schlecht lernt. Eine übliche Abhilfe besteht darin, die Netze gelegentlich neu zu initialisieren und den Puffer beizubehalten :cite:`Nikishin.Schwarzer.DOro.ea.2022`.

<!--en:0698b6c4-->
### Entropie und Temperatur

<!--en:7fd7d762-->
Um die Wirkung der gelernten Temperatur zu untersuchen, vergleichen wir sie mit einem festen Wert $\beta = 0.2$, den viele Implementierungen verwenden. Dazu trainieren wir drei weitere Läufe mit einem Update pro Umgebungsschritt:

<!--en:0ea71f26-->
Mit fester Temperatur erreicht der Ertrag die Schwelle $-200$ im selben Schrittbereich wie mit gelernter Temperatur. Keine Variante ist durchgehend früher. Die folgende Grafik zeigt die Entropieschätzungen beider Varianten im Vergleich zum Zielwert von $-1$ Nat:

<!--en:abf9f08e-->
Bei gelernter Temperatur sinkt die Entropie stetig von mehr als einem Nat auf den Zielwert. Ab etwa 15 000 Schritten bleibt sie in seiner Nähe. Bei festem $\beta = 0.2$ sinkt sie beim Erlernen des Aufschwingens auf einige Zehntel Nat unter null und steigt dann wieder. Sobald das Pendel balanciert, ist der Kritiker in der Nähe des besten Drehmoments fast flach. Streuung kostet dann wenig Wert. Bei einem in der Aktion quadratischen Kritiker mit Krümmung $h$ hat die optimale Gauß-Verteilung die Varianz $\beta/h$ (Aufgabe :exref:`ex_sac-quadratic-critic`). Ein flacherer Kritiker erlaubt also eine breitere Strategie. Nach 20 000 Schritten steigt die Entropie noch. Bei fester Temperatur hängt ihr Endwert deshalb davon ab, wann der Lauf endet. Die folgende Grafik zeigt die Temperatur beider Konfigurationen auf einer logarithmischen Skala:

<!--en:a9884e84-->
Bis etwa Schritt 10 000 fällt die gelernte Temperatur $\beta$ ausgehend von $1$ auf der logarithmischen Skala entlang einer Geraden. Solange die Entropie ihren Zielwert übersteigt, macht Adam mit jedem Update einen Schritt von etwa einer Lernrate. Die Temperatur liegt nahe $0.2$, wenn die Läufe die Schwelle $-200$ überschreiten. Das passt zur Vorhersage von $0.17$ nach 6 000 Updates. Sobald die Entropie ihren Zielwert erreicht, verlangsamt sich der Rückgang. Am Ende beträgt $\beta$ einige Hundertstel. Indem die automatische Anpassung $\beta$ bei flacher werdendem Kritiker senkt, verhindert sie das vorübergehende Absinken und den Wiederanstieg der Entropie. Wie schnell der Agent das Pendel zu regeln lernt, ändert sich dadurch nicht.

<!--en:77760a16-->
Die Trainingserträge stammen von der stochastischen Strategie, denn die Entropie gehört zu ihrer Zielfunktion. Zusätzlich werten wir die mittlere Aktion $c\tanh\mu_\theta(s)$ jedes trainierten Agenten über je zwanzig Episoden aus:

<!--en:03df6812-->
Mit der mittleren Aktion liegt der Ertrag jedes Agenten zwischen etwa $-120$ und $-190$. Ziehen wir die Aktionen stattdessen aus der Strategie, ändert sich der Ertrag um höchstens etwa zehn Punkte. Dank der gelernten zustandsabhängigen Streuung bleibt die Strategie mit geringen Ertragseinbußen stochastisch.

<!--en:45c7d4a9-->
### Kalibrierung der Kritiker

<!--en:0ed24a86-->
Der Akteur folgt dem Gradienten seines Kritikers und übernimmt damit dessen Fehler. Erfüllt das Minimum der beiden Kritiker seinen Zweck? Dazu vergleichen wir die Vorhersage im ersten Zustand einer Episode mit dem Ertrag, den die Strategie anschließend erzielt. Für jeden Agenten führen wir zwanzig stochastische Episoden aus. Wir erfassen $\min_j Q_{w_j}(s_0, a_0)$ und den tatsächlich erzielten diskontierten entropieregularisierten Ertrag. Der Kritiker sagt die Größe $r_0 + \sum_{t \geq 1} \gamma^t \big(r_t - \beta \log \pi_\theta(a_t \mid s_t)\big)$ vorher. Der gemessene Ertrag enthält zusätzlich $-\beta \log \pi_\theta(a_0 \mid s_0)$. Dessen Mittelwert ist $\beta$ mal die Entropie bei $s_0$ und liegt für diese Agenten betragsmäßig unter $0.1$. Die Messung endet nach 200 Schritten. Der Kritiker verwendet dagegen Bootstrapping über die Zeitgrenze hinweg und sagt einen Ertrag über einen unendlichen Horizont vorher. Der fehlende Rest beträgt $\gamma^{200} \approx 0.13$ mal den Wert bei Schritt 200. Dieser Wert liegt für ein balancierendes Pendel nahe null.

<!--en:adcf44e8-->
Zum Vergleich trainieren wir Agenten mit nur einem Kritiker, also ohne Minimum. Wir verwenden zehn Updates pro Umgebungsschritt, weil die beiden Kritiker oben mit dieser Einstellung am schnellsten lernten:

<!--en:4fb1bc40-->
Mit diesem Verhältnis erreicht der einzelne Kritiker $-200$ im selben Schrittbereich wie die beiden Kritiker. Am Ertrag lassen sich die Varianten also nicht unterscheiden. Die Kalibrierung zeigt dagegen einen Unterschied:

<!--en:c817f4b7-->
In fast allen Läufen liegt das Minimum der beiden Kritiker unter dem erzielten Ertrag, in den übrigen höchstens einige Punkte darüber. Der einzelne Kritiker sagt in jedem Lauf mehr als das Minimum und in den meisten Läufen mehr als den erzielten Ertrag vorher. Über die zwanzig Läufe je Konfiguration bestätigten sich diese Richtungen. Die Größe der Abweichungen schwankte von Lauf zu Lauf um mehrere zehn Punkte. Das Minimum verhindert also die Überschätzung, die ein einzelner Kritiker hier zeigt. Es unterschätzt den Ertrag allerdings selbst. Auf Pendulum ändert sich der Ertrag in beiden Fällen nicht. Ob ein einzelner Kritiker bei mehr Updates pro Umgebungsschritt stärker überschätzt, untersuchen wir in Aufgabe :exref:`ex_sac-update-ratio`.

<!--en:229aaa71-->
Der Agent lernt die Pendelregelung nun aus wenigen Tausend Umgebungsschritten. Mehr Updates sparen auf Kosten von Rechenaufwand weitere Umgebungsschritte. Jeder Übergang liefert dem Kritiker aber nur eine Zahl, das Ziel einer Regression. Selbst mit zehn Updates pro Schritt erforderte das Aufschwingen einige Tausend Übergänge. Wer eine Schaukel ein paarmal angeschoben hat, kann vorhersagen, wohin sie sich als Nächstes bewegt. In :numref:`sec_model_based` fragen wir, ob ein Agent die Physik des Pendels aus wenigen Hundert Übergängen lernen und damit das Aufschwingen planen kann.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3d809c0c-->
SAC maximiert mit einem Off-Policy-Akteur-Kritiker-Verfahren die Summe aus Belohnung und gewichteter Entropie. Die entropieregularisierte Strategiebewertung addiert $-\beta \log \pi$ zum Bootstrapping-Ziel des Kritikers. Die entropieregularisierte Strategieverbesserung nähert den Akteur der Projektion von $e^{Q/\beta}$ bezüglich der umgekehrten KL-Divergenz an. Dazu verwendet sie einen pfadweisen Gradienten durch den Kritiker. Eine tanh-Stauchung hält die Aktionen im zulässigen Bereich. Für die Log-Dichte brauchen wir den Term der Variablentransformation, den wir in numerisch stabiler Form berechnen. Zwei Kritiker mit Polyak-gemittelten Kopien verhindern eine Überschätzung durch den Kritiker. Eine gelernte Temperatur hält die Entropie auf einem Zielwert, statt einen festen Umrechnungskurs zwischen Belohnung und Entropie vorzugeben. Auf Pendulum lernt SAC das Aufschwingen bei einem Update pro Schritt nach etwa fünf- bis elftausend Umgebungsschritten. Bei zehn Updates genügen etwa dreitausend. Das sind ein bis zwei Größenordnungen weniger als bei REINFORCE.

<!--en:2294243a-->
## Aufgaben

<!--en:06619925-->
1. **Entropie im Ziel.** Durch den Term $-\beta \log \pi_\theta$ in :eqref:`eq_sac_target` berücksichtigt der Kritiker die Entropie späterer Zustände. Der Entropiebonus aus :numref:`sec_ppo` wirkt dagegen nur im Akteurverlust. Betrachten Sie einen Anfangszustand $s_0$ mit zwei Aktionen L und R, die beide die Belohnung $0$ haben. Aktion L führt in einen Zustand $s_L$ mit einer einzigen verfügbaren Aktion. Aktion R führt in einen Zustand $s_R$ mit $n$ verfügbaren Aktionen. Jede Aktion in $s_L$ und in $s_R$ liefert die Belohnung $0$ und beendet die Episode.
    1. Berechnen Sie für :eqref:`eq_maxent_objective` die optimalen entropieregularisierten Werte von $s_L$ und $s_R$ sowie die optimale Wahrscheinlichkeit für R in $s_0$. Werten Sie die Wahrscheinlichkeit für $n = 10$ und $\gamma = 0.99$ aus. Wie hängt sie von $\beta$ ab?
    1. Angenommen, das Kritikerziel lässt $-\beta \log \pi_\theta$ weg, während der Akteurverlust $\beta \log \pi_\theta$ beibehält. Gegen welche Wahrscheinlichkeit für R konvergiert der Akteur? Welche Zielfunktion optimiert diese Variante?
    1. Der KL-Strafterm aus :numref:`sec_regularized` mit einer Referenz, die in jedem Zustand über die verfügbaren Aktionen gleichverteilt ist, entspricht bis auf eine Konstante dem Entropiebonus. Wiederholen Sie Teil a mit diesem Strafterm. Warum verändert die Konstante hier die optimale Strategie? Warum verändert sie diese auf Pendulum nicht?
   :exlabel:`ex_sac-entropy-in-target`

<!--en:35f1fb9f-->
1. [code] **Quadratischer Kritiker.** Nahe seiner besten Aktion $a^\star$ sei der Kritiker quadratisch in der eindimensionalen Aktion. Es gelte $Q(s, a) = Q(s, a^\star) - \frac{h}{2} (a - a^\star)^2$ mit der Krümmung $h > 0$. Die Strategie sei die ungestauchte Gauß-Verteilung $\mathcal{N}(\mu, \sigma^2)$.
    1. Maximieren Sie $\mathbb{E}_{a \sim \pi}[Q(s, a)] + \beta H(\pi)$ bezüglich $\mu$ und $\sigma$. Zeigen Sie, dass die exponentiell umgewichtete Verteilung $e^{Q(s, \cdot)/\beta}/Z$ das Maximum annimmt. Damit ist die Projektion bezüglich der umgekehrten KL-Divergenz aus :numref:`subsec_sac-soft-improvement` für diese Familie exakt.
    1. Zeigen Sie, dass die optimale Strategie für jede Krümmung einen Wertverlust von $Q(s, a^\star) - \mathbb{E}_{a \sim \pi}[Q(s, a)] = \beta/2$ erleidet, unabhängig von $h$.
    1. In Aufgabe :exref:`ex_regularized-identifiability`c aus :numref:`sec_regularized` haben Sie für eine einzelne Entscheidung gezeigt, dass die Skalierung der Belohnung genauso wirkt wie die Division des KL-Koeffizienten durch denselben Faktor. Zeigen Sie für einen beliebigen MDP, dass die Multiplikation jeder Belohnung mit $k > 0$ unter :eqref:`eq_maxent_objective` dieselbe optimale Strategie ergibt wie die Division von $\beta$ durch $k$. Warum muss ein festes $\beta$ neu abgestimmt werden, wenn sich die Einheit der Belohnung ändert?
    1. Betrachten Sie den Grenzfall $\beta \to 0$. Was geschieht mit $\sigma$? Für $\beta = 0$ und $\sigma_\theta = 0$ wird der Verlust :eqref:`eq_sac_actor` zu $-Q_w\big(s, c \tanh \mu_\theta(s)\big)$, dem Verlust des deterministischen Akteurs von DDPG :cite:`Lillicrap.Hunt.Pritzel.ea.2016`. Woher muss bei einem solchen Agenten die Exploration kommen? Wie hängt dieser Grenzfall mit Aufgabe :exref:`ex_deep-rl-gaussian-score`b aus :numref:`sec_deeprl` zusammen?
    1. Multiplizieren Sie jede Belohnung mit $10$, bevor sie in den Puffer gelangt. Geben Sie den Episodenertrag weiterhin unskaliert an. Trainieren Sie einen Lauf mit festem $\beta = 0.2$ und einen mit gelernter Temperatur. Sagen Sie anhand von Teil c vorher, wie sich die Entropie des ersten Laufs und der Endwert von $\beta$ im zweiten von den Läufen in :numref:`subsec_sac-entropy` unterscheiden. Prüfen Sie Ihre Vorhersage. :citet:`Haarnoja.Zhou.Abbeel.ea.2018` zeigen in ihrer Abbildung 3(b), wie empfindlich SAC bei fester Temperatur auf die Belohnungsskala reagiert.
   :exlabel:`ex_sac-quadratic-critic`

<!--en:71252ca2-->
1. [code] **Entropie der gestauchten Strategie.** Es gelte $u \sim \mathcal{N}(\mu, \sigma^2)$ und $a = c \tanh u$ mit $c = 2$ wie in `SquashedGaussianPolicy`.
    1. Zeigen Sie mithilfe von :eqref:`eq_tanh_logdet`, dass die folgende Gleichung gilt:

       $$H(\pi) = \tfrac{1}{2} \log(2 \pi e \sigma^2) + \mathbb{E}_u\big[\log(1 - \tanh^2 u)\big] + \log c.$$

       Welcher Term wächst bei $\mu = 0$ mit $\sigma$ und welcher nimmt ab?
    1. Berechnen Sie $H$ bei $\mu = 0$ für $\sigma \in \{0.1, 0.3, 1, 3\}$ als Erwartungswert über $u$. Verwenden Sie dazu eine Monte-Carlo-Schätzung mit `log_prob_u` oder eine Quadratur bezüglich $u$. In der Nähe welches Werts von $\sigma$ erreicht $H$ sein Maximum? Vergleichen Sie es mit $\log 2c$, der Entropie der Gleichverteilung auf $[-c, c]$. Keine Dichte auf diesem Intervall erreicht eine höhere Entropie.
    1. Initialisieren Sie eine neue Instanz von `SquashedGaussianPolicy` für Pendulum. Schätzen Sie ihre Entropie durch Monte-Carlo-Sampling mit `sample`. Mitteln Sie über $1{,}000$ Zustände, die Sie wie in der Aufwärmphase von `train_sac` mit gleichverteilten zufälligen Drehmomenten besuchen. Wie nahe liegt die Schätzung am Wert aus Teil b bei $\mu = 0$ und $\sigma = 1$? Welche Ausgaben des neu initialisierten Netzes für diese Zustände erklären eine etwaige Abweichung? Angenommen, der Akteurverlust enthielte nur den Term $\beta \log \pi_\theta$. Welchem Wert würde sich dann der Entropieverlauf aus :numref:`subsec_sac-entropy` nähern?

    *Nach Berkeley CS 285, [Aufgabenblatt 3](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf), Abschnitt 3.3.*
   :exlabel:`ex_sac-squashed-entropy`

<!--en:59171369-->
1. **Off-Policy-Lernen ohne Quotienten.** Der Puffer von `train_sac` speichert jeden Übergang eines Laufs einschließlich der $1{,}000$ gleichverteilten zufälligen Aktionen der Aufwärmphase. Dagegen ziehen :eqref:`eq_sac_target` und :eqref:`eq_sac_actor` neue Aktionen aus der aktuellen Strategie.
    1. Eine Variante von `sac_step` ersetzt $\tilde{a}'$ durch die Aktion, die die datenerhebende Strategie als Nächstes im Zustand $s'$ gewählt hat. Den Wert welcher Strategie schätzt der Kritiker dann? Was misst der Term $-\beta \log \pi_\theta(a' \mid s')$ bei einer solchen Aktion? Welcher Quotient würde das Ziel korrigieren und warum lässt er sich aus diesem Puffer nicht berechnen?
    1. Beide Verluste mitteln über Zustände aus dem Puffer statt über Zustände aus der aktuellen Strategie. Der Satz in :numref:`subsec_sac-soft-improvement` verbessert die Strategie in jedem Zustand. Warum spielt die Gewichtung der Zustände bei exakter Maximierung für jeden einzelnen Zustand keine Rolle? Warum spielt sie für ein Akteurnetz mit beschränkter Kapazität eine Rolle?
   :exlabel:`ex_sac-off-policy-without-ratios`

<!--en:e771d181-->
1. [code] **Rückstellkraft des Jacobi-Terms.** ● Im Akteurverlust :eqref:`eq_sac_actor` enthält der Term $\beta \log \pi_\theta$ für jede Aktionskoordinate den Ausdruck $-\beta \log(1 - \tanh^2 u)$ mit $u = \mu_\theta(s) + \sigma_\theta(s) z$.
    1. Zeigen Sie mit :eqref:`eq_tanh_logdet_stable`, dass dieser Term zum Gradienten des Verlusts den Ausdruck $2 \beta \tanh(u)\, \partial u / \partial \theta$ beiträgt. In welche Richtung verschiebt er $u$ und wie stark kann diese Wirkung werden?
    1. Berechnen Sie durch automatisches Differenzieren in float32 die Ableitungen nach $u$ für die stabile Formel und den abgesicherten Ausdruck `log(1 - tanh(u)**2 + 1e-6)` an den Stellen $u \in \{3, 5, 7, 10, 20\}$. Vergleichen Sie beide mit $-2 \tanh u$. Ab welchem Betrag $|u|$ verliert ein mit dem abgesicherten Ausdruck trainierter Akteur die Rückstellkraft aus Teil a?
    1. Entfernen Sie `logdet` aus `log_prob_u`. Zeigen Sie, dass der verbleibende Entropieterm den Verlust um $-\beta/\sigma$ pro Einheit von $\sigma$ ändert und nicht von $\mu$ abhängt. Warum kann der Kritikerterm im Verlust das Wachstum von $\sigma$ nicht bremsen, sobald $u$ in Sättigung geht? Welche Entropie zeigt die Messkurve, wenn $\sigma$ die Grenze $e^2$ erreicht? Wie verhält sich dieser Wert zur Schranke aus Aufgabe :exref:`ex_sac-squashed-entropy`b?
    1. Trainieren Sie nach dem Entfernen des Terms erneut mit einem Zufallsstartwert. Zeichnen Sie die Ertragskurve und den Entropieverlauf auf. Prüfen Sie die geänderte Dichte durch Quadratur mit `squashed_logp`. Welche der drei Messungen decken den Fehler auf?
   :exlabel:`ex_sac-jacobian-force`

<!--en:8fd00e6d-->
1. [code] **Verhältnis von Updates zu Daten.** Die Läufe aus :numref:`subsec_sac-sample-efficiency` verwendeten ein und zehn Updates pro Umgebungsschritt.
    1. Trainieren Sie mit drei Zufallsstartwerten und `utd=20` über 3 000 Umgebungsschritte, einmal mit zwei Kritikern und einmal mit einem einzelnen Kritiker. Erreichen die Läufe die Schwelle früher als bei zehn Updates pro Schritt? Wiederholen Sie den Versuch mit `warmup=200`.
    1. Untersuchen Sie die Agenten aus Teil a mit `calibration`. Nimmt die Abweichung des einzelnen Kritikers beim Übergang von zehn auf zwanzig Updates pro Schritt zu? Bleibt das Minimum beider Kritiker auf oder unter dem erzielten Ertrag?
    1. Zählen Sie für jeden Lauf die Gradientenschritte bis zum Überschreiten von $-200$, und zwar bei einem, zehn und zwanzig Updates pro Schritt. Welchen Aufwand tauscht ein höheres Verhältnis gegen Umgebungsschritte ein? Was begrenzt die mögliche Einsparung?
   :exlabel:`ex_sac-update-ratio`

<!--en:be9a20d9-->
1. [extended] **Diskretes Soft Actor-Critic.** ● Bei einer endlichen Aktionsmenge gibt das Strategienetz über eine Softmax-Funktion die Wahrscheinlichkeiten $\pi_\theta(\cdot \mid s)$ aus. Jeder Kritiker liefert wie das Q-Netz aus :numref:`sec_dqn` einen Wert pro Aktion :cite:`Christodoulou.2019`.
    1. Geben Sie das Kritikerziel und den Akteurverlust an. Berechnen Sie dabei die Erwartungswerte über $\tilde{a}'$ und $\tilde{a}$ exakt durch Summation über die Aktionen. Welche Teile von `SquashedGaussianPolicy` entfallen? Warum muss der Kritiker für den Akteurgradienten nicht mehr bezüglich der Aktion differenzierbar sein?
    1. Warum ist der Zielwert $\bar{H} = -\dim \mathcal{A}$ für die automatische Temperaturwahl in :numref:`subsec_sac-temperature` bei einer diskreten Strategie bedeutungslos? In welchem Intervall muss eine Zielentropie für $|\mathcal{A}|$ Aktionen liegen?
    1. Implementieren Sie das Verfahren auf CartPole mit zwei Kritikern und Polyak-Zielnetzen. Vergleichen Sie den Ertrag in Abhängigkeit von der Zahl der Umgebungsschritte mit `train_dqn` aus :numref:`sec_dqn` bei gleichem Budget und denselben Zufallsstartwerten.
   :exlabel:`ex_sac-discrete`

<!--en:e28a5f88-->
[Diskussion](https://d2l.discourse.group/t/36805)

<!--en:dc7582ed-->
Kontinuierliche Regelung mit DDPG, TD3 und Soft Actor-Critic<br>**Reinforcement Learning mit maximaler Entropie · pfadweise Gradienten · Jacobi-Determinante der gestauchten Gauß-Verteilung · zwei Kritiker · automatische Temperaturwahl · Updates pro Umgebungsschritt**

<!--en:1f6db362-->
::: {.slide title="Warum kontinuierliche Regelung Replay und pfadweise Ableitungen braucht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:6f63e2a0-->
- **Warum DQN nicht direkt anwendbar ist:** Bei kontinuierlichen Drehmomenten $u \in [-2, 2]$ erfordert die Auswertung von $\max_{u'} Q_w(s', u')$ in jedem Bellman-Ziel bei jedem Schritt eine innere Optimierung.

<!--en:6169a044-->
- **Pfadweiser Akteur und Replay:** Wir übernehmen Replay-Puffer und Zielnetze von DQN und trainieren einen Akteur $\tilde{a}_\theta(s, z)$ direkt anhand der Steigung $\partial Q_w(s, a)/\partial a$. Damit sinkt der Datenbedarf auf Pendulum von $\sim 300{,}000$ auf $\sim 3{,}000$ Schritte.

<!--en:df56e321-->
::: {.slide title="Von DDPG und TD3 zu Soft Actor-Critic (SAC)"}

<!--en:ea151974-->
[Entwicklung der Algorithmen]{.kicker}

<!--en:211a39bd-->
- **DDPG (2016):** Deterministischer Akteur $\mu_\theta(s)$ und ein Kritiker $Q_{w^-}$. Das Verfahren ist störanfällig, weil der Akteur schmale Spitzen im Schätzfehler des Kritikers ausnutzt.

<!--en:55a8cd3c-->
- **TD3 (2018):** Verwendet **zwei Kritiker** $\min_{j=1,2} Q_{w_j^-}(s', \tilde{a}')$, um die Maximierungsverzerrung nach oben zu beseitigen, braucht aber weiterhin eigens gewähltes Rauschen zur Exploration.

<!--en:ef4f305d-->
- **SAC (2018):** Kombiniert zwei Kritiker mit einem **stochastischen Akteur mit gestauchter Gauß-Verteilung** und der Belohnung $-\beta \log \pi_\theta$ für maximale Entropie.

<!--en:2a328e2a-->
::: {.slide title="Die Zielfunktion beim Reinforcement Learning mit maximaler Entropie"}

<!--en:9b0446a6-->
[Reinforcement Learning mit maximaler Entropie]{.kicker}

<!--en:e39250ce-->
- **Temperatur $\beta > 0$:** Legt den Umrechnungskurs in Belohnungseinheiten pro Nat differentieller Entropie $H(\pi) = -\mathbb{E}[\log \pi(a \mid s)]$ fest.

<!--en:b48f45ca-->
- **Bonus für zukünftige Entropie:** Der Term $\beta H(\pi(\cdot \mid s'))$ im Ziel des Kritikers belohnt Aktionen, die in Zustände mit hoher Entropie und vielen Handlungsmöglichkeiten führen.

<!--en:fb0d5ed6-->
::: {.slide title="Zwei Aktionen von Hand durchgerechnet: exakter entropieregularisierter Wert aus einer Ziehung"}

<!--en:21490583-->
[Rechenbeispiel]{.kicker}

<!--en:2ce78d51-->
Betrachten wir einen Zustand mit zwei Aktionen, den Werten $Q = (1, 0)$ und der Temperatur $\beta = 0.5$. Nach §{sec_regularized} ist das exponentiell umgewichtete Optimum $\pi^\star \propto e^{Q/\beta} = (e^2, 1)/(e^2 + 1) = (0.88, 0.12)$ mit dem entropieregularisierten Wert $\beta \log(e^2 + 1) = 1.063$:

<!--en:ddd39126-->
**Varianz null bei $\pi^\star$:** Im exponentiell umgewichteten Optimum $\pi^\star(a) = e^{Q(a)/\beta}/Z$ hat der Klammerausdruck $Q(a) - \beta \log \pi^\star(a) = \beta \log Z = 1.063$ **für jede Aktion $a$ exakt denselben Wert**.

<!--en:a4494992-->
::: {.slide title="Entropieregularisierte Strategiebewertung mit einer neu gezogenen Aktion im Ziel"}

<!--en:aff7f43d-->
[Entropieregularisierte Strategiebewertung]{.kicker}

<!--en:b94e2eb0-->
- **Keine Wichtigkeitsquotienten nötig:** Der Übergang $(s, a, r, s')$ stammt aus dem Replay-Puffer. Für die Fortsetzung ziehen wir $\tilde{a}' \sim \pi_\theta(\cdot \mid s')$ *neu* aus der aktuellen Strategie.

<!--en:5cce934f-->
- **Garantierte Kontraktion:** Die exakte entropieregularisierte Strategiebewertung ist bei beschränkten Belohnungen und Entropien eine $\gamma$-Kontraktion.

<!--en:0a78767d-->
::: {.slide title="Entropieregularisierte Strategieverbesserung durch Projektion bezüglich der umgekehrten KL-Divergenz"}

<!--en:6e8f02fe-->
[Entropieregularisierte Strategieverbesserung]{.kicker}

<!--en:76eb3867-->
In jedem Zustand $s$ entspricht die Maximierung von $\mathbb{E}_{a \sim \pi}[Q_w(s, a)] + \beta H(\pi(\cdot \mid s))$ der Minimierung der **umgekehrten KL-Divergenz** zur Boltzmann-Verteilung $e^{Q_w(s, \cdot)/\beta} / Z(s)$:

<!--en:72ad1e5b-->
- **Konzentration auf einen Modus:** Hat $Q_w(s, \cdot)$ zwei getrennte Gipfel, bestraft die Minimierung der umgekehrten KL-Divergenz $D_{\textrm{KL}}(\pi_\theta \Vert e^{Q/\beta}/Z)$ Wahrscheinlichkeitsmasse im Tal dazwischen. Eine unimodale Strategie konzentriert sich daher auf einen Gipfel.

<!--en:52946d31-->
- **Reparametrisierung (pfadweiser Gradient):** Schreiben wir $\tilde{a}_\theta(s, z) = c \tanh\big(\mu_\theta(s) + \sigma_\theta(s) z\big)$, so fließt $\nabla_\theta L_\pi(\theta)$ über die Kettenregel direkt durch $\partial Q_w(s, a) / \partial a$.

<!--en:a9e3f451-->
- **Satz zur entropieregularisierten Strategieverbesserung:** Verbessert die Projektion bezüglich der umgekehrten KL-Divergenz in jedem Zustand die Strategie $\pi_{\textrm{old}}$, dann gilt $V^{\pi_{\textrm{new}}}(s) \ge V^{\pi_{\textrm{old}}}(s)$ für alle Zustände $s$.

<!--en:d66c160e-->
::: {.slide title="Warum hartes Clipping den pfadweisen Gradienten unterbricht"}

<!--en:1a02c719-->
[Beschränkte kontinuierliche Aktionen]{.kicker}

<!--en:4f72deb8-->
- **Warum hartes Clipping versagt:** Für $|u| > c$ verschwindet wegen $\frac{d}{du}\textrm{clip}(u, -c, c) = 0$ der pfadweise Akteurgradient. Die bei $\pm c$ angehäufte Masse aus den Verteilungsrändern lässt dort keine kontinuierliche Log-Dichte $\log \pi(a \mid s)$ zu.

<!--en:35b3cbdc-->
- **Glatt gestauchte Gauß-Verteilung:** Ziehen wir $u \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$ und setzen $a = c\tanh(u) \in (-c, c)$, so gilt überall $da/du = c(1 - \tanh^2 u) > 0$.

<!--en:f26f37a7-->
::: {.slide title="Die Jacobi-Determinante der tanh-Transformation"}

<!--en:adc8d257-->
[Herleitung der Dichte]{.kicker}

<!--en:f135fbf4-->
- Die Abbildung $a_i = c \tanh(u_i)$ staucht Intervalle nahe den Grenzen $\pm c$ um den Jacobi-Faktor $da_i / du_i = c(1 - \tanh^2 u_i)$. Aus der Erhaltung der Wahrscheinlichkeit $\pi(a)\,da = \mathcal{N}(u; \mu, \sigma)\,du$ folgt:
  $$\log \pi(a \mid s) = \sum_{i=1}^d \Big[\log \mathcal{N}(u_i; \mu_i, \sigma_i) - \log(1 - \tanh^2 u_i) - \log c\Big]$$

<!--en:699f940c-->
- Nahe $a = \pm 2$ staucht die Transformation den oberen Rand von $u$. Dadurch entsteht an der Grenze eine Dichtespitze. Ohne den Jacobi-Faktor fehlt diese Häufung und das Integral beträgt $1.13 \neq 1$.

<!--en:0e9ec8e3-->
::: {.slide title="Implementierung der gestauchten Gauß-Strategie"}

<!--en:927d2e90-->
[Implementierung der Strategie]{.kicker}

<!--en:99758412-->
`SquashedGaussianPolicy` kapselt einen Stamm mit zwei ReLU-Schichten und zustandsabhängige Ausgabeköpfe für Mittelwert und logarithmierte Standardabweichung (`state_std=True`). Die Klasse berechnet $\log \pi(a \mid s)$ direkt aus dem vor der Stauchung gezogenen Wert $u$ und benötigt keinen Aufruf von `arctanh`:

<!--en:cbfb6e63-->
::: {.slide title="Implementierung der gestauchten Gauß-Strategie (Fortsetzung)"}

<!--en:927d2e90-2-->
[Implementierung der Strategie]{.kicker}

<!--en:4e5c3e98-->
Bei der Variablentransformation bewertet `log_prob_u` die Ziehung $u$ vor der Stauchung. Die Methode `sample` zieht eine reparametrisierte Aktion und liefert ihre Log-Wahrscheinlichkeit mit:

<!--en:6f38d51d-->
::: {.slide title="Implementierung der gestauchten Gauß-Strategie (Aktionswahl)"}

<!--en:927d2e90-3-->
[Implementierung der Strategie]{.kicker}

<!--en:286a115b-->
Für die Aktionswahl in der Umgebung liegen die Zustände als NumPy-Arrays vor. Die Methode `_draw` zieht $u$ aus der Verteilung des Ausgabekopfs mit dem Rauschen von `rng`. Anschließend staucht `act` den Wert:

<!--en:503b5857-->
::: {.slide title="Implementierung der gestauchten Gauß-Strategie (Fortsetzung der Aktionswahl)"}

<!--en:927d2e90-4-->
[Implementierung der Strategie]{.kicker}

<!--en:855b5fe8-->
`act_logp` gibt zusätzlich $\log \pi(a \mid s)$ zurück. Für die Auswertung staucht `act_greedy` den Mittelwert:

<!--en:e70decf0-->
::: {.slide title="Numerisch stabile logarithmierte Determinante und Prüfung durch Quadratur"}

<!--en:16100594-->
[Numerische Genauigkeit]{.kicker}

<!--en:3dbc339e-->
In `float32` rundet $\tanh(u)$ auf $1.000000$ für $|u| \ge 10$. Damit gilt $\log(1 - \tanh^2 u) = -\infty$. Eine einfache Absicherung durch $+10^{-6}$ bleibt bei $-13.82$ stehen und lässt den rücktreibenden Gradienten $-2\tanh u$ verschwinden. Algebraisches Umformen vermeidet die katastrophale Auslöschung:

<!--en:6b9ddb5d-->
::: {.slide title="Zwei Kritiker und Zielnetze mit Polyak-Mittelung"}

<!--en:5dd2ff1d-->
[Aufbau der Kritiker]{.kicker}

<!--en:b5770f1d-->
- **Clipped Double-Q (zwei Kritiker):** Ein Akteur, der entlang $\nabla_a Q_w(s, a)$ aufsteigt, sucht systematisch überschätzte Aktionen auf. SAC trainiert daher zwei unabhängige Kritiker $Q_{w_1}, Q_{w_2}$ und verwendet ihr **punktweises Minimum** sowohl im Ziel als auch im Akteurverlust:
  $$y = r + \gamma(1 - \textrm{term})\Big(\min_{j=1,2} Q_{w_j^-}(s', \tilde{a}') - \beta \log \pi_\theta(\tilde{a}' \mid s')\Big)$$

<!--en:e060fc71-->
- **Polyak-Mittelung ($\nu = 0.005$):** Anstelle abrupter Sprünge alle $250$ Schritte verschieben sich die Zielgewichte nach jedem Update ein wenig gemäß $w_j^- \leftarrow \nu w_j + (1 - \nu) w_j^-$ (Halbwertszeit $\ln 2 / \nu \approx 139$ Updates).

<!--en:15576e8a-->
::: {.slide title="Automatische Temperaturwahl durch dualen Gradientenabstieg"}

<!--en:c830629f-->
[Entropienebenbedingung]{.kicker}

<!--en:3877a89e-->
- **Zielentropie $\bar{H} = -\dim(\mathcal{A})$:** Erzwingt auf Pendulum unabhängig von der Skalierung der Belohnung die Bedingung $-1.0\text{ nat}$.

<!--en:654b0ab0-->
- **Automatische Rückkopplung:** Nach $\partial J / \partial \log\beta = \hat{H}_{\textrm{batch}} - \bar{H}$ senkt eine zu hohe Entropie ($\hat{H} > \bar{H}$) den Wert von $\log\beta$. Eine zu niedrige Entropie ($\hat{H} < \bar{H}$) erhöht $\log\beta$.

<!--en:8be8012b-->
::: {.slide title="Der Container SoftActorCritic und die Verlustfunktionen"}

<!--en:d5663700-->
[Bausteine]{.kicker}

<!--en:2dcea2e4-->
`SoftActorCritic` umfasst den Akteur mit gestauchter Gauß-Verteilung, die beiden Aktionswertkritiker $Q_{w_1}, Q_{w_2}$ mit verketteter Eingabe $[s, a] \in \mathbb{R}^4$, ihre Polyak-Zielkopien und $\log \beta$:

<!--en:556bceaf-->
::: {.slide title="Ein vollständiges SAC-Update" layout="tight" only="pytorch"}

<!--en:61ab2693-->
[Der Update-Schritt]{.kicker}

<!--en:51b6823b-->
Jeder Aufruf von `sac_step` passt beide Kritiker durch Regression an das gemeinsame entropieregularisierte Ziel $y$ mit dem Minimum beider Schätzungen an. Dann folgen das Akteurupdate mit dem pfadweisen Gradienten von $\beta \log \pi_\theta - \min_j Q_{w_j}$, die Anpassung von $\log \beta$ zur Regelung der Strategieentropie auf $\bar{H} = -1$ und die Mittelung der Zielnetze mit $\nu = 0.005$:

<!--en:291e231e-->
::: {.slide title="Ein vollständiges SAC-Update (Fortsetzung)" layout="tight" only="pytorch"}

<!--en:61ab2693-2-->
[Der Update-Schritt]{.kicker}

<!--en:c495f3c8-->
Der Temperaturschritt passt $\log \beta$ an, damit sich die Entropie ihrem Zielwert nähert. Die Polyak-Mittelung verschiebt die Zielkritiker:

<!--en:2dd30582-->
::: {.slide title="Ein vollständiges SAC-Update" layout="tight" only="jax"}

<!--en:61ab2693-3-->
[Der Update-Schritt]{.kicker}

<!--en:51b6823b-2-->
Jeder Aufruf von `sac_step` passt beide Kritiker durch Regression an das gemeinsame entropieregularisierte Ziel $y$ mit dem Minimum beider Schätzungen an. Dann folgen das Akteurupdate mit dem pfadweisen Gradienten von $\beta \log \pi_\theta - \min_j Q_{w_j}$, die Anpassung von $\log \beta$ zur Regelung der Strategieentropie auf $\bar{H} = -1$ und die Mittelung der Zielnetze mit $\nu = 0.005$:

<!--en:31bd7697-->
::: {.slide title="Ein vollständiges SAC-Update (Fortsetzung)" layout="tight" only="jax"}

<!--en:61ab2693-4-->
[Der Update-Schritt]{.kicker}

<!--en:c495f3c8-2-->
Der Temperaturschritt passt $\log \beta$ an, damit sich die Entropie ihrem Zielwert nähert. Die Polyak-Mittelung verschiebt die Zielkritiker:

<!--en:1b95cb3b-->
::: {.slide title="Die Off-Policy-Trainingsschleife für kontinuierliche Aktionen"}

<!--en:b0a65151-->
[Trainingsschleife]{.kicker}

<!--en:c1aad572-->
Nach `warmup = 1000` Schritten mit gleichverteilten zufälligen Aktionen macht `train_sac` pro Umgebungsschritt `utd` Gradientenschritte (**Verhältnis von Updates zu Daten**). So können wir Interaktionen in der realen Welt durch Rechenaufwand einsparen:

<!--en:db5e9a83-->
::: {.slide title="Die Off-Policy-Trainingsschleife für kontinuierliche Aktionen (Fortsetzung)"}

<!--en:b0a65151-2-->
[Trainingsschleife]{.kicker}

<!--en:d6de6c4d-->
Bei jedem Umgebungsschritt wählen wir eine Aktion (in der Aufwärmphase gleichverteilt), speichern den Übergang und machen `utd` Updates. Am Episodenende geben wir Ertrag, Entropie und $\beta$ aus:

<!--en:87ff1839-->
::: {.slide title="Dateneffizienz auf Pendulum bei einem und bei zehn Updates pro Schritt (UTD 1 und UTD 10)"}

<!--en:a7187495-->
[Dateneffizienz]{.kicker}

<!--en:b12ed616-->
- **1 Update pro Schritt (`UTD 1`):** Die 20 Läufe überschreiten $-200$ nach **$5{,}400$ bis $11{,}200$ Schritten** (die drei gezeigten nach $5{,}400$–$6{,}600$). Das sind fast um den Faktor $50\times$ weniger Schritte als bei REINFORCE.

<!--en:d96b3565-->
- **10 Updates pro Schritt (`UTD 10`):** Die 20 Läufe überschreiten $-200$ nach **$2{,}400$ bis $4{,}000$ Schritten** (die drei gezeigten nach $2{,}400$–$2{,}600$, nahe der durch Aufwärmphase und Mittelungsfenster vorgegebenen Untergrenze von $2{,}000$ Schritten). Bis dahin brauchen sie $\sim 3\times$ so viele Gradientenschritte.

<!--en:089ce766-->
::: {.slide title="Warum der Ertrag anfangs sinkt: erst dämpfen, dann Energie zuführen"}

<!--en:0ef4395b-->
[Verhaltensanalyse]{.kicker}

<!--en:3782b1e8-->
Warum sinken die Erträge zunächst von $-1{,}200$ bei der Zufallsstrategie auf $-1{,}700$ um Schritt $2{,}000$, bevor sie auf $-170$ steigen? Der Vergleich des Akteurs nach $2{,}000$ Schritten mit drei analytischen Reglern erklärt den Einbruch:

<!--en:0d4228ff-->
**Kurzsichtige Dämpfung:** Nach $2{,}000$ Schritten berücksichtigt der Kritiker erst die Belohnungen der nächsten wenigen Schritte. Dort überwiegt der Vorteil geringerer Kosten $0.1\dot\vartheta^2$ für die Winkelgeschwindigkeit. Der Akteur **dämpft daher die Bewegung** ($|\dot\vartheta| < 1\text{ rad/s}$). Das Pendel hängt unten fast still ($\cos\vartheta \approx -0.93$, Kosten von $\pi^2 \approx 9.9$ pro Schritt), bis der Kritiker den über mehrere Schritte erzielbaren Wert des Aufschwingens erfasst.

<!--en:3d5c2b80-->
::: {.slide title="Entropie der Strategie bei gelerntem und festem Beta"}

<!--en:9e35e966-->
[Entropieverlauf]{.kicker}

<!--en:ff697da5-->
- **Gelernte Temperatur $\beta$ (blau):** Die Entropie der Strategie sinkt gleichmäßig von etwa $+1.3\text{ nats}$ auf den Zielwert $\bar{H} = -1.0\text{ nat}$ und bleibt ab $15{,}000$ Schritten dort.

<!--en:9c5bcf97-->
- **Feste Temperatur $\beta = 0.2$ (orange):** Beim Erlernen des Aufschwingens sinkt die Entropie auf einige Zehntel Nat unter null. Sobald der Agent balancieren kann, **steigt sie wieder**. Beim Balancieren hat der Kritiker eine geringe Krümmung $h$. Für einen quadratischen Kritiker gilt für die optimale Gauß-Varianz $\sigma^2 = \beta / h$.

<!--en:24f4b275-->
::: {.slide title="Verlauf der dualen Temperatur und Kosten der zufälligen Aktionswahl" layout="tight"}

<!--en:ebde68c7-->
[Temperatur und Auswertung]{.kicker}

<!--en:a8e6633d-->
- **Log-linearer Rückgang:** Solange $\hat{H} > -1$ gilt, senkt Adam $\log \beta$ pro Update um etwa seine Lernrate $\approx 3 \times 10^{-4}$. Der Verlauf entspricht der Vorhersage $e^{-1.8} \approx 0.17$ nach $6{,}000$ Updates ($7\text{k}$ Umgebungsschritten). Später pendelt sich die Temperatur nahe $0.03$ ein.

<!--en:8a2dbc78-->
::: {.slide title="Verlauf der dualen Temperatur und Kosten der zufälligen Aktionswahl (Fortsetzung)" layout="tight"}

<!--en:ebde68c7-2-->
[Temperatur und Auswertung]{.kicker}

<!--en:7d15b2bb-->
- **Fast kostenlose Exploration bei Konvergenz:** Der Vergleich der deterministischen mittleren Aktion $c\tanh\mu_\theta(s)$ mit stochastisch gezogenen Aktionen zeigt, dass die zustandsabhängige Streuung höchstens $\sim 10$ Ertragspunkte kostet.

<!--en:69256d1a-->
::: {.slide title="Kalibrierung beim Minimum zweier Kritiker und bei einem einzelnen Kritiker"}

<!--en:b9106305-->
[Kalibrierung der Werte]{.kicker}

<!--en:6d78db23-->
Der Vergleich der Vorhersage $\min_j Q_{w_j}(s_0, a_0)$ im Anfangszustand mit dem tatsächlich erzielten diskontierten entropieregularisierten Ertrag $\sum_{t=0}^{199} \gamma^t (r_t - \beta \log \pi_\theta(a_t \mid s_t))$ über 200 Schritte zeigt, was das Minimum zweier Kritiker verhindert:

<!--en:2ef729ed-->
Beide Varianten erreichen auf Pendulum $-200$. Bei fast jedem Zufallsstartwert **überschätzt der einzelne Kritiker systematisch** den wahren entropieregularisierten Ertrag. Das **Minimum der beiden Kritiker** liegt in fast jedem Lauf unter dem erzielten Ertrag und sonst höchstens wenige Punkte darüber.

<!--en:b104f0fc-->
::: {.slide title="Verständnisfragen zu Grenzfällen, Krümmung und Rückstellkraft des Jacobi-Terms"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:29bcd7d9-->
Prüfen Sie Ihr Verständnis der mathematischen Grundlagen von SAC:

<!--en:a43955d3-->
1. **Deterministischer Grenzfall:** Setzen Sie im Akteurverlust und im Ziel der beiden SAC-Kritiker die Temperatur auf $\beta = 0$ und die Standardabweichung der Strategie auf $\sigma_\theta = 0$. Welchen Algorithmus erhalten Sie und wie muss er explorieren?

<!--en:a1fcfdf0-->
2. **Skalierung der Belohnung:** Sie multiplizieren alle Belohnungen auf Pendulum mit $10$. Wie muss sich $\beta$ ändern, damit die exakt optimale Strategie $\pi^\star$ gleich bleibt? Wie reagiert die automatische Temperaturwahl?

<!--en:f67053a0-->
3. **Rückstellkraft des Jacobi-Terms:** Welchen Wert liefert $\log(1 - \tanh^2 u) = 2(\log 2 - u - \operatorname{softplus}(-2u))$ für die Ableitung $\frac{\partial}{\partial u}\big[-\beta \log(1 - \tanh^2 u)\big]$ im Akteurverlust? Was geschieht, wenn Sie die abgesicherte Formel $\log(1 - \tanh^2 u + 10^{-6})$ bei $u = 12$ in `float32` verwenden?

<!--en:557b98b6-->
**Antworten:**

<!--en:7ecb603d-->
1. Der Akteurverlust wird zu $-\min_{j=1,2} Q_{w_j}(s, c\tanh\mu_\theta(s))$. Wir erhalten **DDPG mit den beiden Kritikern von TD3**. Weil die Strategie deterministisch ist, braucht sie bei der Datenerhebung zusätzliches Rauschen auf den Aktionen.

<!--en:aad6789f-->
2. Die Skalierung $r \to 10r$ erfordert **$\beta \to 10\beta$**, damit $r/\beta$ konstant bleibt. Die automatische Temperaturwahl stellt einen um den Faktor $10\times$ größeren Wert von $\beta$ ein, denn $\partial J(\beta)/\partial \log\beta = \hat{H} - \bar{H}$ hängt nur von der Entropie ab.

<!--en:a69fac6c-->
3. Die exakte Ableitung ist **$+2\beta \tanh(u)$**. Beim Gradientenabstieg zieht sie gesättigte Aktivierungen $u$ vor der Stauchung zu $0$ zurück. In `float32` ergibt bei $u = 12$ der Ausdruck $1 - \tanh^2(12)$ durch Rundung $0$. Damit verschwindet die Ableitung der abgesicherten Formel und beträgt **$0$**.

<!--en:b0e32c95-->
::: {.slide title="Überblick über die Bestandteile von Soft Actor-Critic und ihre Aufgaben"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:a0b98ceb-->
| Bestandteil von SAC | Mathematischer Ausdruck | Gelöstes Problem | Gemessene Wirkung auf Pendulum |

<!--en:1fe5cca1-->
| **Entropieregularisiertes Kritikerziel** | $r + \gamma(1-\textrm{term})\big(\min_j Q_{w_j^-}(s', \tilde{a}') - \beta \log \pi_\theta(\tilde{a}' \mid s')\big)$ | Off-Policy-Wertschätzung ohne Wichtigkeitsquotienten | Bewertet Belohnung und zukünftige Entropie anhand einer neu gezogenen Aktion $\tilde{a}'$ |

<!--en:946fa5b5-->
| **Pfadweiser Akteurverlust** | $\mathbb{E}_{s, z}\big[\beta \log \pi_\theta(\tilde{a}_\theta \mid s) - \min_j Q_{w_j}(s, \tilde{a}_\theta)\big]$ | Hohe Varianz der Score-Funktionsgradienten | Nutzt direkt die Steigung $\partial Q / \partial a$ und braucht um den Faktor $\sim 50\times$ weniger Schritte als REINFORCE |

<!--en:cb175832-->
| **Logarithmierte Jacobi-Determinante der tanh-Transformation** | $-2\big(\log 2 - u - \operatorname{softplus}(-2u)\big) - \log c$ | Undefinierte Dichte und verschwindende Gradienten bei hartem Clipping | Das Integral der Dichte beträgt $1.000000$ ohne Unterlauf in `float32` |

<!--en:3cf2f648-->
| **Zwei Kritiker mit Polyak-Mittelung** | $\min(Q_{w_1^-}, Q_{w_2^-})$, $w_j^- \leftarrow 0.005 w_j + 0.995 w_j^-$ | Maximierungsverzerrung beim Aufstieg des Akteurs entlang $Q$ im kontinuierlichen Aktionsraum | Die vorhergesagten entropieregularisierten Werte liegen in fast jedem Lauf bei oder unter den erzielten Erträgen |

<!--en:55257f7e-->
| **Duale Temperatur ($\beta$)** | $\min_{\log\beta} -\log\beta\, \mathbb{E}[\log \pi_\theta(a \mid s) + \bar{H}]$ | Abhängigkeit einer festen Temperatur $\beta$ von Belohnungsskala und Kritikerkrümmung | Hält die Strategieentropie bei $\bar{H} = -1\text{ nat}$, während der Kritiker flacher wird |
