<!--en:b69328c5-->
# Lernen aus Demonstrationen

<!--en:6b6b50e3-->
Im Grundlagenteil hat jedes Modell aus Beispielen mit Labels gelernt, etwa aus einem Bild und seiner Klasse oder einem Satz und dem nächsten Wort. Im vorigen Abschnitt haben wir gesehen, wie viel ein Agent verliert, wenn ihm niemand Labels liefert und er gute Aktionen finden muss, indem er schlechte ausprobiert. Oft kann uns jemand vormachen, wie es geht. Angenommen, wir zeichnen auf, was ein geübter Fahrer sieht und tut. Zu jedem Zeitpunkt erfassen wir das Kamerabild und den Lenkwinkel, den er daraufhin wählt. Könnten wir mit diesen Paaren eine Fahrstrategie (engl. policy) trainieren, so wie wir einen Klassifikator auf Fashion-MNIST trainiert haben, mit dem Bild als Eingabe und dem Lenkwinkel als Label?

<!--en:9d6af0a1-->
Dean Pomerleau baute Ende der 1980er-Jahre an der Carnegie Mellon University ein solches Netz. Sein Netz ALVINN leitete mit einer einzigen versteckten Schicht aus Straßenbildern eine Lenkrichtung ab und steuerte so einen Kleintransporter eine Straße entlang :cite:`Pomerleau.1989`. Eine spätere Version lernte das Fahren in wenigen Minuten, indem sie einem Menschen dabei zusah :cite:`Pomerleau.1991`. Heute heißt das Verfahren *Verhaltensklonen* (engl. behavior cloning). Es braucht weder ein Modell der Umgebung noch eine Belohnung, sondern nur einen Experten, dessen Entscheidungen sich aufzeichnen lassen. Heute ist es das gängigste Verfahren, um von einem Experten zu lernen. Auch das Fine-Tuning eines Sprachmodells anhand kuratierter Antworten ist Verhaltensklonen. Der bisherige Text ist der Zustand und das nächste Token die Aktion.

<!--en:c92bd798-->
Pomerleau erkannte auch die Schwäche des Verfahrens. Der Fahrer hielt den Wagen in der Mitte der Fahrspur. Deshalb zeigten die Aufnahmen weder die Sicht vom Straßenrand aus noch ein Beispiel für das Zurücklenken. Sobald der Wagen unter der Steuerung des Netzes von der Spurmitte abkam, sah das Netz unbekannte Bilder. Pomerleau musste deshalb künstlich seitlich versetzte Straßenansichten erzeugen. Als Label gab er jeweils den Lenkwinkel an, der den Wagen zurückführen würde. Einen Klassifikator testen wir mit ähnlichen Eingaben wie beim Training. Eine Strategie bestimmt durch ihre Aktionen selbst, welche Eingaben sie als Nächstes sieht. In diesem Abschnitt messen wir am Beispiel des Sees, was das kostet, und zeigen, wie die Kosten mit der Länge der Aufgabe wachsen können. Dann ändern wir die Datenerhebung so, dass der Experte die Zustände mit Labels versieht, die der lernende Agent selbst erreicht.

<!--en:986d80b0-->
## Verhaltensklonen

<!--en:6afad21b-->
Eine *Demonstration* ist eine aufgezeichnete Trajektorie eines Experten. Beim Verhaltensklonen behalten wir nur die Zustände, die der Experte besucht hat, und die dort gewählten Aktionen. Belohnungen und Übergangsmodell brauchen wir dazu nicht. Auf dem See dient die optimale Strategie $\pi^*$ als Experte. Wir berechnen sie wie in :numref:`sec_valueiter` durch Wertiteration. Daher bezeichnet $\pi^*$ in diesem Abschnitt sowohl die optimale Strategie als auch den Experten. Wir verwenden griffiges Eis, auf dem jeder Befehl den Agenten in die gewünschte Richtung bewegt. Der Experte geht dann in jeder Episode denselben Weg in sechs Zügen vom Start zum Ziel. Seine Zustände bilden einen schmalen Korridor über den See, ähnlich der Spurmitte in Pomerleaus Aufnahmen. Da wir das Modell des Sees kennen, können wir jede Erfolgsrate exakt berechnen und den lernenden Agenten ohne Stichprobenrauschen mit dem Experten vergleichen.

<!--en:25d83aba-->
### Rückführung auf ein Klassifikationsproblem

<!--en:35a34e73-->
Die folgende Codezelle stellt den See mit griffigem Eis als `TabularMDP` dar. Dazu verwendet sie die Arrays $P$ und $r$ mit dem Diskontfaktor $\gamma$ aus :numref:`sec_mdp`. Sie löst das Modell mit `d2l.value_iteration` und liest die Expertenstrategie aus `mdp.backup` ab. Das ist der Bellman-Operator angewandt auf $V^*$. In jedem Zustand wählt sie daraus die beste Aktion. `demonstrations` lässt den Experten in der Umgebung agieren und zeichnet nur auf, was er gesehen und getan hat.

<!--en:9fce0f97-->
Drei Episoden ergeben 18 Paare mit Labels, dreimal dieselben sechs Paare. Auf griffigem Eis verläuft jede Expertenepisode gleich: *abwärts*, *abwärts*, *rechts*, *abwärts*, *rechts*, *rechts*. In den Zuständen 0 und 9 sind die Wege über *abwärts* und *rechts* gleich kurz. Bei Gleichstand fällt die Wahl auf *abwärts*. Die Eisfelder 1, 2, 3, 6 und 10 besucht der Experte nie.

<!--en:e0f8a284-->
Beim Verhaltensklonen behandeln wir diese Paare so wie die Bilder und Labels in :numref:`sec_softmax`. Der Zustand ist die Eingabe, die Aktion des Experten die Klasse. Wir passen eine bedingte Verteilung $\pi_\theta(a \mid s)$ mit den Parametern $\theta$ an, indem wir die Log-Likelihood $\sum_i \log \pi_\theta(a_i \mid s_i)$ maximieren. Das entspricht der Minimierung der Kreuzentropie. Wir brauchen dazu nur eine parametrisierte Strategie und eine Möglichkeit, ihre Parameter mit Gradientenschritten anzupassen.

<!--en:87eac8f6-->
### Die Strategieklasse

<!--en:800c18c8-->
`Policy` enthält ein Netz `policy`, das für einen Zustand je einen Score pro Aktion liefert. Damit ergibt sich $\pi_\theta(\cdot \mid s)$ durch Anwendung der Softmax-Funktion auf die Scores im Zustand $s$. Zur Klasse gehört auch ein Adam-Optimierer `opt_pi` für die Netzparameter. `log_prob` liefert $\log \pi_\theta(a \mid s)$ für einen Batch aus Zuständen und den dort gewählten Aktionen. Genau diese Größe maximieren wir beim Verhaltensklonen.

<!--en:ecb2886a-->
Der Konstruktor `tabular` erzeugt ein Modell für die Softmax-Regression. Eine Einbettungstabelle speichert für jedes Zustand-Aktion-Paar eine Präferenz $\theta_{s,a}$. Die ganzzahlige Zustandsnummer gibt die Zeile an, und die Softmax-Funktion liefert $\pi_\theta(\cdot\mid s)$. Durch die Initialisierung mit Nullen sind anfangs alle Aktionen gleich wahrscheinlich.

<!--en:56149477-->
Das Argument `seed` legt die Zufallsinitialisierung der Netze fest, die dieser und spätere Konstruktoren erzeugen. `seeded` initialisiert den CPU-Zufallsgenerator von PyTorch mit `seed` innerhalb eines `with`-Blocks und stellt anschließend den vorherigen Generatorzustand wieder her. Der Startwert beeinflusst also keine Ziehung außerhalb des Konstruktors.

<!--en:8e8cfc6e-->
Für die Tabelle verwenden wir `d2l.Embed` (:numref:`subsec_utils_embed`), eine Variante von `nnx.Embed`. Auf einem Beschleuniger berechnet sie den Gradienten mit einem einzigen Matrixprodukt anstelle atomarer Additionen. Atomare Additionen in derselben Zeile geschehen in wechselnder Reihenfolge. Dadurch würden sich die letzten Bits eines Schritts von Ausführung zu Ausführung unterscheiden. Nach vielen Schritten mit großer Schrittweite beträfe das ganze Trainingsläufe.

<!--en:a578aee7-->
Die Umgebung tauscht Ganzzahlen und Arrays im NumPy-Format aus. Drei Methoden übernehmen die Umwandlung an der Schnittstelle. `act` zieht eine Aktion aus $\pi_\theta(\cdot \mid s)$ mit einem NumPy-Zufallsgenerator, den der Aufrufer übergibt. Damit hat der gesamte Zufall, der auf die Umgebung wirkt, eine einzige sichtbare Quelle. Die Methode folgt der Schnittstelle zur Aktionswahl von `d2l.evaluate` aus :numref:`sec_valueiter`. Diese Funktion mittelt die Erträge der per Sampling erzeugten Episoden. `act_greedy` liefert die wahrscheinlichste Aktion und `log_prob_np` die logarithmischen Wahrscheinlichkeiten als NumPy-Array.

<!--en:19c129d1-->
Die drei Methoden laufen unter `torch.inference_mode`, weil wir keine ihrer Berechnungen differenzieren.

<!--en:599c107e-->
`act` und `act_greedy` führen das Netz mit `d2l.forward_np` aus. `log_prob_np` und die weiter unten definierte Funktion `policy_step` füllen ihren Batch mit `d2l.padded` auf die Länge der jeweiligen Größenklasse auf. So lässt sich der kompilierte Code für verschiedene Batchgrößen nutzen. Mit `d2l.shared_adam` teilen sich alle Agenten mit derselben Lernrate einen Optimierer und damit die kompilierten Aktualisierungen. Diese Hilfsfunktionen stehen in :numref:`sec_utils`. `log_prob` akzeptiert ein optionales Argument `policy`, weil `nnx.value_and_grad` nach dem übergebenen Modul differenziert. Bei der Aktualisierung übergeben wir dieses Modul wieder an die Funktion. JAX berechnet Gradienten nur innerhalb einer solchen Transformation.

<!--en:c169ca01-->
Für eine Strategie mit wenigen Zuständen liest `probs` die gesamte Tabelle $\pi_\theta(a \mid s)$ des lernenden Agenten als Array aus. Damit führen wir in diesem Kapitel exakte Berechnungen durch.

<!--en:68fa13d0-->
### Anpassung des Klons

<!--en:6e72879f-->
Wir passen den Klon durch Gradientenaufstieg auf $\frac{1}{n}\sum_i w_i \log \pi_\theta(a_i \mid s_i)$ an, wobei jedes Gewicht $w_i$ gleich eins ist. `policy_step` führt für einen Batch aus Zuständen, Aktionen und Gewichten einen Optimierungsschritt für diese Zielfunktion aus.

<!--en:480f0f64-->
Die Gewichte liegen als NumPy-Array vor, also als Daten. In sie fließt kein Gradient. Beim Verhaltensklonen sind sie alle gleich eins.

<!--en:fd83b020-->
Wie lange sollen wir trainieren? Trainierten wir eine Tabelle mit einer Zeile pro Zustand bis zur Konvergenz, würde sie die sechs demonstrierten Aktionen exakt kopieren. Auf griffigem Eis würde der Klon dann jedes Mal den Weg des Experten nehmen. In der Praxis arbeiten Klassifikatoren nicht fehlerfrei. Ähnliche Eingaben, begrenzte Kapazität und verrauschte Labels führen auch auf der Trainingsverteilung zu Fehlern. Damit unser Klon ebenfalls einen solchen Fehler aufweist, ruft `clone` die Funktion `policy_step` mit Gewichten von eins auf, bis $\pi_\theta$ mit einer Wahrscheinlichkeit von höchstens $\varepsilon = 0.2$ von den demonstrierten Aktionen abweicht, gemittelt über die Paare. Das Abbruchkriterium gibt $\varepsilon$ vor. So bleibt der Anpassungsfehler bei allen Vergleichen in diesem Abschnitt konstant.

<!--en:4216b5ce-->
Vierzehn Adam-Schritte senken die Wahrscheinlichkeit einer Abweichung auf 0.182. Jeder demonstrierte Zustand kommt gleich oft vor und erhält denselben Gradienten. Daher beträgt in jeder der sechs Zeilen die Wahrscheinlichkeit für die Expertenaktion am Ende 0.818 und für jede der drei anderen Aktionen 0.061, wie die Zeile für Zustand 9 zeigt. Zustand 10 kam in keiner Demonstration vor. Seine Zeile erhält keinen Gradienten und behält die Gleichverteilung bei. Dort weicht der Klon also in drei von vier Fällen vom Experten ab. Eine Wiederholung der Daten ändert die Anpassung nicht. Der mittlere Verlust über drei Kopien der sechs Paare ist derselbe wie über eine Kopie. Das Klonen einer einzigen Expertenepisode ergibt daher dieselbe Tabelle wie das Klonen von hundert Episoden.

<!--en:94193d26-->
## Verteilungsverschiebung

<!--en:6a40f976-->
### Training und Einsatz

<!--en:417e3491-->
Die Garantie eines Klassifikators gilt für die Verteilung, aus der seine Trainingsdaten stammen. Beim Verhaltensklonen trainieren wir mit den Zuständen, die der Experte besucht. Im Einsatz trifft der Klon auf Zustände, die er durch seine eigenen Aktionen erreicht. Wir setzen ihn auf zwei Arten ein. Bei der gierigen Aktionswahl wählt er stets die wahrscheinlichste Aktion. Beim Sampling zieht er mit der Strategie $\pi_\theta$ und der Methode `act` eine Aktion, so wie ein Sprachmodell sein nächstes Token zieht.

<!--en:3fbc8f76-->
Da wir den See kennen, können wir die Zustandsverteilung einer Strategie direkt berechnen, statt sie durch Sampling zu schätzen. Sei $d^\pi_t(s)$ die Wahrscheinlichkeit, dass sich der Agent unter der Strategie $\pi$ im Zustand $s$ befindet, nachdem er $t$ Schritte gemacht hat. Anfangs liegt die gesamte Wahrscheinlichkeitsmasse im Startzustand. In einem Schritt verteilt sie sich gemäß der Strategie und dem Übergangskern wie folgt.

<!--en:15a42a7f-->
Die Masse in $s'$ im nächsten Schritt ist die Summe der Massen, die aus den einzelnen Zuständen durch die gewählten Aktionen dorthin gelangen. `visitation` führt diese Rekursion bis zum Zeitlimit von 100 Schritten auf dem See aus. Das Ziel und die Löcher beenden die Episode und behalten ihre Masse. Daher entspricht die Masse im Ziel nach 100 Schritten der Erfolgswahrscheinlichkeit. Mit einem Monte-Carlo-Lauf von `d2l.evaluate` überprüfen wir den exakten Wert für den Klon mit Sampling.

<!--en:7c9ff1fd-->
Der Experte erreicht jedes Mal das Ziel, ebenso der gierige Klon. In jeder demonstrierten Zeile ist die Expertenaktion die wahrscheinlichste, sodass der gierige Klon dem Weg des Experten folgt. Der Klon mit Sampling erreicht das Ziel in 66% seiner Episoden. In den übrigen fällt er in ein Loch. Die 1000 per Sampling erzeugten Episoden bestätigen dieses Ergebnis im Rahmen der Zufallsschwankungen. Auf griffigem Eis sind nur die Entscheidungen des Klons zufällig. Jeder Schritt, mit dem er den Weg verlässt, geht also auf eine eigene Fehlentscheidung zurück.

<!--en:92d1bf5b-->
Liegt das am Klassifikator? Die nächste Codezelle misst für drei Gruppen von Zuständen, wie oft der Klon vom Experten abweicht. Wir betrachten die Trainingspaare, die Zustände aus 100 neuen Expertenepisoden und die vom Klon selbst besuchten Zustände, gewichtet mit der jeweiligen Besuchshäufigkeit. Anschließend klonen wir 1, 10 und 100 Expertenepisoden.

<!--en:02dfcbe6-->
In neuen Expertenepisoden macht der Klon genauso oft Fehler wie auf seinen Trainingspaaren, nämlich in 18% der Fälle. Es gibt also keine Generalisierungslücke. Die Demonstrationen decken die Verteilung des Experten ab. Weitere Expertendaten helfen hier nicht, weil kein neuer Zustand hinzukommt. Jede Wiederholung der sechs Paare ergibt beim Klonen dieselbe Tabelle mit einer Erfolgsrate von 66%. In den selbst besuchten Zuständen macht der Klon häufiger Fehler, nämlich in 22% der Fälle. Für einige dieser Felder fehlen ihm Daten, und dort weicht er mit Wahrscheinlichkeit $3/4$ vom Experten ab. Die Verteilung der Zustände, in denen wir die Strategie testen, hat sich durch ihre eigenen Aktionen verschoben. Das ist eine *Verteilungsverschiebung* (engl. distribution shift). Bei einer Aufgabe mit sechs Schritten ist die Verschiebung gering. In :numref:`subsec_il_state_divergence` messen wir, welchen Anteil des Verlusts sie verursacht.

<!--en:c28616b4-->
### Fehlende Abdeckung

<!--en:c837160f-->
Fehlende Abdeckung hat eine andere Ursache. Auf rutschigem Eis rutscht auch der Experte aus. Welche Zustände er besucht, hängt daher vom Zufall ab. Bei wenigen Episoden kann ein Zustand fehlen, in dem es auf die richtige Aktion ankommt. Die nächste Codezelle erzeugt den rutschigen See und seine optimale Strategie. Wir ziehen 200 Datensätze mit je $n$ Expertenepisoden für jedes $n$ und berechnen für jeden Datensatz die exakte Erfolgsrate des gierigen Klons. In jedem demonstrierten Zustand wählt dieser Klon die Expertenaktion. In allen anderen Zeilen sind die Aktionen gleich wahrscheinlich, und bei Gleichstand wählt er Aktion 0. Die Codezelle erstellt die Tabelle direkt, sodass wir keine 600 Klone trainieren müssen.

<!--en:8f888116-->
Bei einer Episode ergibt etwa jeder dritte Datensatz einen Klon, dessen Erfolgsrate unter der Hälfte der Erfolgsrate des Experten liegt. Bei drei Episoden trifft das auf einen oder zwei von hundert Datensätzen zu, bei zehn auf keinen der 200. Mit mehr Expertendaten verschwindet die fehlende Abdeckung. Die Verteilungsverschiebung bleibt jedoch bestehen, weil der Klon durch seine Fehler Zustände erreicht, die der Experte nie besucht.

<!--en:97eac60b-->
## Fehlerakkumulation

<!--en:e48dbf40-->
### Quadratisches und lineares Fehlerwachstum

<!--en:0d4ab126-->
Der Weg des Experten über den See ist sechs Züge lang. Wie wachsen die Kosten eines Fehlers pro Schritt, wenn die Aufgabe länger dauert? Beim überwachten Lernen haben wir eine Schranke für die Fehlerwahrscheinlichkeit bei einer einzelnen Entscheidung. Bei einer sequenziellen Aufgabe verändert ein früher Fehler alle späteren Zustände und kann alle späteren Belohnungen kosten. Die Zustände in den Demonstrationen stammen aus den Verteilungen $d^{\pi^*}_t$ des Experten. Nur für diese Verteilungen begrenzt das Verhaltensklonen die Fehlerrate des Klons.

<!--en:f6d9ac75-->
**Proposition (Fehlerakkumulation).** *Betrachten wir eine episodische Aufgabe mit $T$ Schritten und Belohnungen in $[0, 1]$ bei jedem Schritt. Eine Strategie $\hat\pi$ weiche in jedem Schritt $t$ mit Wahrscheinlichkeit höchstens $\varepsilon$ vom Experten ab, wenn der Zustand aus der Verteilung $d^{\pi^*}_t$ des Experten gezogen wird. (i) Der erwartete Ertrag von $\hat\pi$ liegt um höchstens $\varepsilon T(T+1)/2 \leq \varepsilon T^2$ unter dem des Experten. Es gibt Aufgaben, bei denen die Differenz diese Größenordnung hat, sobald $\varepsilon T \leq 1$ gilt. (ii) Nehmen wir nun an, dass $\hat\pi$ mit Wahrscheinlichkeit höchstens $\varepsilon$ vom Experten abweicht, wenn wir den Zustand aus ihrer eigenen Verteilung $d^{\hat\pi}_t$ ziehen. Kostet eine falsche Aktion mit anschließendem Expertenverhalten höchstens $u$, dann beträgt die Ertragsdifferenz höchstens $u \varepsilon T$* :cite:`Ross.Bagnell.2010`.

<!--en:e717e562-->
**Beweis.** (i) Lassen wir $\hat\pi$ und den Experten im selben Zustand starten und in der Umgebung dieselben Zufallsergebnisse erleben. Ihre Trajektorien stimmen überein, bis $\hat\pi$ erstmals eine andere Aktion wählt als der Experte. Der Zustand des Experten im Schritt $t$ stammt aus $d^{\pi^*}_t$. Die erste Abweichung kann nur dann im Schritt $t$ auftreten, wenn $\hat\pi$ in diesem Zustand abweicht. Das geschieht mit Wahrscheinlichkeit höchstens $\varepsilon$. Von da an kann $\hat\pi$ höchstens die Belohnungen der Schritte $t, \ldots, T$ verlieren, also höchstens $T-t+1$ Belohnungen. Summieren wir über $t$, so beträgt die Ertragsdifferenz höchstens $\sum_{t=1}^{T} \varepsilon (T-t+1) = \varepsilon T(T+1)/2$. (ii) Die Strategie $\pi_k$ folge der Strategie $\hat\pi$ für die ersten $k$ Schritte und danach dem Experten. Somit entspricht $\pi_0$ dem Experten und $\pi_T$ der Strategie $\hat\pi$. Zwei aufeinanderfolgende Mischstrategien $\pi_{k-1}$ und $\pi_k$ unterscheiden sich nur in der Aktion im Schritt $k$. Der Zustand in diesem Schritt stammt aus $d^{\hat\pi}_k$. Die Aktion weicht mit Wahrscheinlichkeit höchstens $\varepsilon$ vom Experten ab. Eine Abweichung mit anschließendem Expertenverhalten kostet höchstens $u$. Der Ertrag des Experten abzüglich des Ertrags von $\hat\pi$ ist die Summe der $T$ Differenzen zwischen aufeinanderfolgenden Mischstrategien und damit höchstens $u \varepsilon T$. $\blacksquare$

<!--en:56768b55-->
Der zweite Teil verwendet das Teleskopsummenargument von :citet:`Kakade.Langford.2002`. Bezeichnet $\varepsilon$ den Mittelwert der Fehlerraten über die $T$ Schritte anstelle ihres Maximums, beträgt die Summe in Teil (i) immer noch höchstens $\varepsilon T^2$. Teil (i) rechnet einem ersten Fehler sämtliche nachfolgenden Kosten zu, denn danach befindet sich der Agent in Zuständen, für die keine Fehlerschranke gilt. In Teil (ii) messen wir $\varepsilon$ auf den Zuständen der Strategie selbst. Die Schranke hilft nur, wenn $u$ klein ist, wenn sich also eine einzelne falsche Aktion ausgleichen lässt. Ist ein einziger Fehler endgültig, so ist $u$ so groß wie der verbleibende Horizont. Teil (ii) liefert dann wieder die Größenordnung $\varepsilon T^2$.

<!--en:330cdd33-->
Der Horizont auf dem See ist fest. Daher untersuchen wir das Wachstum mit $T$ anhand einer Kette mit $T$ Schritten. Der Experte geht in jedem Schritt nach rechts und erhält dafür jeweils eine Belohnung von eins. Zwei Nachahmer weichen in jedem Schritt unabhängig mit Wahrscheinlichkeit $\varepsilon$ vom Experten ab. Beim *geklonten* Nachahmer wirkt ein Fehler dauerhaft. Er verlässt die Kette, gerät in Zustände, die ihm nie gezeigt wurden, und erhält danach keine Belohnung mehr. Somit erhält er die Belohnung im Schritt $t$ mit Wahrscheinlichkeit $(1-\varepsilon)^t$. Der *zurückkehrende* Nachahmer ist nach einem Schritt wieder auf der Kette. Ein Fehler kostet ihn also genau eine Belohnung, wie in Teil (ii) mit $u = 1$. Er erhält die Belohnung im Schritt $t$ mit Wahrscheinlichkeit $1-\varepsilon$. Die Codezelle berechnet den entgangenen Ertrag exakt für $\varepsilon = 0.01$ und Werte von $T$ bis 64.

<!--en:fcb2ad1a-->
Auf logarithmischen Achsen liegt die Kurve des geklonten Nachahmers nahe an der quadratischen Referenzkurve $\varepsilon T(T+1)/2$. Die Abweichung beträgt bei $T = 10$ höchstens 3% und bei $T = 30$ höchstens 10%. Mit wachsendem $\varepsilon T$ bleibt die Kurve zunehmend unter der Referenzkurve, weil ein erster Fehler nur einmal auftreten kann. Die erste Abweichung fällt in den Schritt $t$ mit Wahrscheinlichkeit $\varepsilon(1-\varepsilon)^{t-1}$. Diese Wahrscheinlichkeit ist kleiner als der Wert $\varepsilon$, mit dem die Schranke rechnet. Wie Teil (i) verlangt, steigt die Kurve nie über die Referenzkurve. Der zurückkehrende Nachahmer verliert genau $\varepsilon T$. Bei $T = 64$ entgehen dem geklonten Nachahmer etwa 17 Belohnungen, dem zurückkehrenden 0.64, der Verlust des geklonten Nachahmers ist also etwa 27-mal so groß. In Aufgabe :exref:`ex_imitation-chain-closed-form` leiten Sie für beide Kurven eine geschlossene Formel her.

<!--en:8c4c625c-->
### Zerlegung des Verlusts des Klons

<!--en:4f23af7c-->
Zurück zum See. Wie viel vom Verlust des Klons von 34 Prozentpunkten geht auf die Verteilungsverschiebung zurück? Der Klon kann auf zwei Arten scheitern. Er kann direkt vom Weg des Experten in ein Loch fallen, wenn er in einem der sechs Trainingszustände eine falsche Aktion wählt. Das entspricht dem Fehler $\varepsilon$ auf der Verteilung des Experten, den nur eine bessere Anpassung behebt. Oder er kann den Weg verlassen und auf ein Feld geraten, für das ihm Daten fehlen, und von dort aus scheitern. Darin zeigt sich die Verteilungsverschiebung. In der nächsten Codezelle machen wir die Felder abseits des Weges absorbierend. So misst `visitation`, wie oft der Klon den Weg verlässt. Wir vergleichen, wie häufig er das Ziel erreicht, wenn er auf dem Weg bleibt und wenn er ihn verlässt. In der letzten Zeile bewerten wir einen lernenden Agenten, der die Fehlerrate des Klons von 18% auf jedem Feld aufweist, auf und abseits des Weges.

<!--en:a8d93a85-->
Der Klon verlässt den Weg in 17% seiner Episoden. Stets führt ihn eine falsche Aktion von Zustand 0 auf Feld 1 oder von Zustand 9 oder 14 auf Feld 10 (:numref:`fig_rl_compounding_error`). Bleibt der Klon auf dem Weg, erreicht er in 72% der Fälle das Ziel. Verlässt er ihn, sind es 39%. Entsprechend lässt sich der Verlust aufteilen. Von den 34 Prozentpunkten entfallen 24 auf Stürze direkt vom Weg in ein Loch ($0.828 \times 0.285$) und 10 auf Fehlschläge nach dem Verlassen des Weges ($0.172 \times 0.606$). Ein lernender Agent mit der Fehlerrate des Klons auf jedem Feld würde das Ziel in 73.5% der Fälle erreichen. Labels abseits des Weges können also etwa 7.5 der 34 Prozentpunkte zurückgewinnen. Das ist weniger als 10, denn selbst auf einem Feld mit Label liegt die Fehlerrate noch bei 18%. Auf diesem See erklärt die Verteilungsverschiebung etwa ein Fünftel des Verlusts des Klons. Der Rest geht auf die Fehlerrate von 18% selbst zurück, auf und abseits des Weges. Daran ändert auch eine andere Datenerhebung nichts. Einen längeren Weg verlässt der Klon wesentlich häufiger. In Aufgabe :exref:`ex_imitation-dagger-s-budget` untersuchen Sie auf dem See der Größe $8 \times 8$, wie sich das auf die beiden Verlustanteile auswirkt.

<!--en:d28756ab-->
![Welche Felder der Klon aus diesem Abschnitt auf dem See mit griffigem Eis erreicht. Blaue Pfeile markieren den sechs Züge langen Weg des Experten. Im Einsatz zieht der Klon seine Aktionen per Sampling. In den Zuständen des Experten weicht er mit Wahrscheinlichkeit $\varepsilon = 0.18$ von ihm ab. Jedes Feld zeigt die Wahrscheinlichkeit, dass der Klon es innerhalb der 100 Schritte betritt. Dunklere Felder betritt er häufiger. Die gestrichelten Pfeile zeigen, wo der Klon den Weg verlässt und auf Felder gerät, für die ihm Daten fehlen und auf denen er alle Aktionen mit gleicher Wahrscheinlichkeit wählt. Er gelangt von 0 nach 1 und von 9 oder 14 nach 10. Mit Wahrscheinlichkeit $0.66$ erreicht er das Ziel. Der größte Teil der übrigen Wahrscheinlichkeitsmasse entfällt auf die Löcher 5 und 12 neben dem Weg.](../img/mdl-rl-compounding-error.svg)

<!--en:1c571f9c-->
## Interaktive Datenerhebung

<!--en:a81a5e9e-->
### Aggregation von Datensätzen

<!--en:fa9487b6-->
Weitere Demonstrationen können die Verteilungsverschiebung nicht beheben, weil der Experte die Felder nie besucht, auf denen der Klon vom Weg abkommt. Der lernende Agent muss sie selbst besuchen, und der Experte muss sie mit Labels versehen. Genau das leistet *DAgger*, kurz für „Dataset Aggregation“ :cite:`Ross.Gordon.Bagnell.2011`. Das Verfahren lässt den aktuellen Agenten agieren und fragt den Experten für jeden besuchten Zustand, welche Aktion er dort gewählt hätte. Die neuen Paare kommen zum Datensatz hinzu, und wir trainieren den Agenten erneut (:numref:`fig_rl_dagger_loop`). Modell, Verlust, Optimierer und Abbruchkriterium bleiben gleich. Nur die Daten ändern sich.

<!--en:71585537-->
![Eine Runde DAgger. Der lernende Agent bestimmt durch seine Aktionen die Zustände, der Experte liefert die Labels. Für jeden besuchten Zustand nennt er die Aktion, die er dort wählen würde. Diese Paare ergänzen den Datensatz für das erneute Training des Agenten. Der erste Datensatz besteht aus den Demonstrationen des Experten.](../img/mdl-rl-dagger-loop.svg)

<!--en:5d92c5f0-->
`relabel` lässt den lernenden Agenten mit `act` seine Aktionen wie im Einsatz per Sampling wählen. Als Label erhält jeder besuchte Zustand die Aktion des Experten.

<!--en:4d3d38e4-->
`dagger` beginnt mit den drei Demonstrationen und lässt den lernenden Agenten vier Runden mit je zehn Episoden durchlaufen. Nach jeder Runde trainieren wir ihn mit `clone` erneut und bewerten ihn exakt. Mit `d2l.run_seeds` aus :numref:`sec_utils` wiederholen wir das für zehn Zufallsstartwerte und fassen die Ergebnisse in einem Array zusammen.

<!--en:96c9c87f-->
Runde 0 entspricht dem Verhaltensklonen mit einer Erfolgsrate von 0.66. Jede Runde fügt etwa 65 Labels hinzu, die meisten für die sechs Zustände auf dem Weg, gelegentlich auch eines für ein Feld abseits davon. Innerhalb von vier Runden kommen so einige hundert Labels zusammen. Der Median über zehn Zufallsstartwerte erreicht je nach Auswahl der zehn Werte 0.70 bis 0.735. Damit liegt er nahe bei oder genau auf der Erfolgsrate von 0.735 eines Agenten, der auf jedem Feld die Fehlerrate des Klons aufweist. Bei jedem Startwert schneidet der Agent der vierten Runde besser ab als das Verhaltensklonen. Dieses bleibt unabhängig von der Zahl der Expertenlabels bei 0.66. DAgger gleicht den Teil des Verlusts aus, der auf Zustände abseits des Weges zurückgeht. Der verbleibende Abstand zum Experten ergibt sich aus dem Anpassungsfehler von 18%, auf den DAgger keinen Einfluss hat. Ein Agent kann die 0.735 exakt erreichen. Adam teilt die Schrittweite für jeden Parameter durch ein Maß für die Größe seiner eigenen Gradienten. Der Gradient einer Zeile mit Label ist die Anzahl ihrer Labels mal einem Faktor, der für alle Zeilen mit Label gleich ist. Daher erreicht jede solche Zeile am Ende dieselbe Wahrscheinlichkeit von 0.818, unabhängig von der Zahl ihrer Labels. Welche Felder ein Label tragen, legt den Agenten also vollständig fest. Sobald jedes Feld, das er erreicht, ein Label trägt, ist er genau der Agent mit Erfolgsrate 0.735.

<!--en:21049617-->
Die Erfolgsrate muss nicht von Runde zu Runde steigen. Bei manchen Zufallsstartwerten sinkt sie in einer Runde leicht, im ungünstigsten Fall auf 0.649. In Aufgabe :exref:`ex_imitation-dagger-guarantee` untersuchen Sie die Ursache. Die Garantie von :citet:`Ross.Gordon.Bagnell.2011` gilt für den besten Agenten der Folge oder für einen zufällig daraus gezogenen Agenten. Für den letzten allein gibt sie keine Zusicherung. Sie setzt viele Runden der Datenerhebung voraus und einen Agenten, der auf dem angesammelten Datensatz mit verschwindendem mittlerem Regret lernt. In der ursprünglichen Analyse stammen die Aktionen bei der Datenerhebung teils vom Experten, teils vom lernenden Agenten. Der Experte muss außerdem für jeden erreichten Zustand ein Label liefern können. Da wir die Daten aller früheren Runden behalten, hängt die angepasste Strategie auch von den früheren Trajektorien ab.

<!--en:eb460ec4-->
### Überwachtes Fine-Tuning als Verhaltensklonen

<!--en:c267efa5-->
Im Token-MDP aus :numref:`sec_mdp` ist das überwachte Fine-Tuning eines Sprachmodells ein Fall von Verhaltensklonen. Der Zustand ist der Kontext, also der Prompt zusammen mit den bisher geschriebenen Token. Die Aktion ist das nächste Token, das beim Zustandsübergang mit Wahrscheinlichkeit eins angehängt wird. Die kuratierten Antworten sind die Demonstrationen des Experten. Die Kreuzentropie beim Fine-Tuning ist die negative Log-Likelihood des Verhaltensklonens, summiert über die Token jeder Antwort. Der Horizont $T$ ist die Länge der Antwort. Die Größe $u$ gibt an, was ein falsches Token kostet, wenn das Modell anschließend wieder gut fortfährt. Bei einem ungewöhnlichen, aber harmlosen Wort sind die Kosten gering. Bei einer falschen Ziffer am Anfang einer Rechnung, auf der jeder spätere Schritt aufbaut, sind sie hoch. Im Einsatz zieht das Modell seine Token selbst und verwendet sie als Kontext. Je länger eine Antwort wird, desto weiter entfernt sich ihr Kontext daher von denen der kuratierten Antworten.

<!--en:f1db1044-->
DAgger für ein Sprachmodell braucht einen Experten, der die vom Modell selbst erzeugten Präfixe mit Labels versieht. Für jeden vom Schüler erzeugten Kontext muss er angeben, welches Token folgen sollte. Für Millionen von Präfixen können Menschen das nicht leisten, ein größeres Modell hingegen schon. Bei der On-Policy-Destillation trainieren wir den Schüler mit seinen eigenen Samples. Als Labels dienen die Verteilungen des Lehrermodells für das nächste Token :cite:`Agarwal.Vieillard.Zhou.ea.2024`. Das entspricht DAgger mit dem Lehrer als jederzeit verfügbarem Experten. Verhaltensklonen ist auch das übliche erste Vergleichsverfahren beim bestärkenden Offline-Lernen, wo nur ein fester Datensatz zur Verfügung steht. In :numref:`sec_offline` vergleichen wir Offline-Verfahren damit.

<!--en:8db2ea81-->
Beim Verhaltensklonen genügt ein Klassifikator, um aus den aufgezeichneten Entscheidungen eines Experten eine Strategie zu lernen. DAgger gleicht allein durch eine Änderung der Daten den Verlustanteil aus, der auf die selbst besuchten Zustände zurückgeht. Beide Verfahren brauchen einen Experten, der das Verhalten demonstriert. Für DAgger muss er außerdem jederzeit Labels für die vom lernenden Agenten erreichten Zustände liefern können. Steht kein Experte zur Verfügung, bleiben zwei Wege. Sind die Regeln der Umgebung bekannt und differenzierbar, können wir gute Aktionen daraus berechnen. In :numref:`sec_optimal_control` tun wir das für kontinuierliche Zustände und Aktionen. Sind auch die Regeln unbekannt, können wir den Aktualisierungsschritt des Klonens dennoch verwenden. Dazu ersetzen wir das Gewicht $w_i = 1$ durch eine Zahl, die angibt, wie gut die Aktion ausgegangen ist. In :numref:`sec_policygradient` leiten wir dieses Gewicht aus der Belohnung her.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:8ef07467-->
Beim Verhaltensklonen passen wir eine Strategie $\pi_\theta(a \mid s)$ wie bei der Softmax-Regression durch Minimierung der Kreuzentropie an die Zustand-Aktion-Paare eines Experten an. Dafür brauchen wir weder ein Übergangsmodell noch eine Belohnung. Jeder Anpassungsschritt ist ein Gradientenschritt für logarithmische Wahrscheinlichkeiten mit Gewichten von eins. Die Anpassung begrenzt den Fehler der Strategie auf der Zustandsverteilung des Experten. Im Einsatz besucht der Agent jedoch die Zustände, die sich aus seinen eigenen Aktionen ergeben. Ein Fehler von $\varepsilon$ pro Schritt auf den Expertenzuständen kann bis zu $\varepsilon T(T+1)/2$ an Ertrag über $T$ Schritte kosten. Derselbe Fehler auf den eigenen Zuständen der Strategie kostet höchstens $u \varepsilon T$, wenn ein einzelner Fehler höchstens $u$ kostet. Auf dem sechs Züge langen Weg über den See mit griffigem Eis ändern zusätzliche Expertendaten den Klon nicht. Die Verschiebung hin zu Zuständen, die der Experte nie besucht, erklärt etwa ein Fünftel des Verlusts. DAgger versieht die selbst besuchten Zustände des lernenden Agenten mit den Aktionen des Experten als Labels. Allein durch die Änderung der Daten gleicht das Verfahren diesen Teil des Verlusts aus.

<!--en:2294243a-->
## Übungen

<!--en:709f0dd5-->
1. **Geschlossene Formeln für die Kette.** Im Kettenexperiment aus :numref:`subsec_il_error_bounds` weicht ein Nachahmer mit Wahrscheinlichkeit $\varepsilon$ in jedem Schritt $t = 1, \ldots, T$ unabhängig vom Ausgang der anderen Schritte vom Experten ab. Der geklonte Nachahmer erhält die Belohnung im Schritt $t$ nur dann, wenn er bis einschließlich Schritt $t$ jedes Mal mit dem Experten übereingestimmt hat. Der zurückkehrende Nachahmer erhält sie immer dann, wenn er im Schritt $t$ mit ihm übereinstimmt.
    1. Leiten Sie eine geschlossene Formel für den erwarteten Ertrag $G(T)$ her, der dem geklonten Nachahmer gegenüber dem Experten entgeht. Zeigen Sie, dass dem zurückkehrenden Nachahmer $\varepsilon T$ entgeht. Vergleichen Sie beide Ergebnisse mit den ausgegebenen Werten für $\varepsilon = 0.01$ und $T = 64$.
    1. Entwickeln Sie $G(T)$ bis zur zweiten Ordnung in $\varepsilon$. Zeigen Sie, dass der Term erster Ordnung die Schranke $\varepsilon T(T+1)/2$ aus der Proposition ergibt, und bestimmen Sie den Term zweiter Ordnung. Was sagt sein Vorzeichen über die Lage der Kurve des geklonten Nachahmers relativ zur Referenzkurve aus? Bis zu welchem Horizont weicht bei $\varepsilon = 0.01$ der Term erster Ordnung um höchstens 10% von $G(T)$ ab?
    1. Wie wächst $G(T)$, sobald $\varepsilon T \gg 1$ gilt? Welche Bedingung an $\varepsilon T$ ist demnach für quadratisches Wachstum nötig? Berechnen Sie das Verhältnis der beiden Verluste für $\varepsilon = 0.01$ und $T = 64$ am rechten Rand des Diagramms. Vergleichen Sie es mit dem Verhältnis $(T+1)/2$ der beiden Referenzwerte.
   :exlabel:`ex_imitation-chain-closed-form`

<!--en:12ae6aa0-->
1. **Fine-Tuning als Verhaltensklonen.** In :numref:`subsec_il_sft` haben wir das überwachte Fine-Tuning eines Sprachmodells als Verhaltensklonen aufgefasst. Der Kontext ist der Zustand, das nächste Token die Aktion. Die kuratierten Antworten sind die Demonstrationen.
    1. Die Proposition aus :numref:`subsec_il_error_bounds` setzt in jedem Schritt dieselbe Fehlerrate $\varepsilon$ voraus. Entlang kuratierter Antworten kann ein Modell nach dem Fine-Tuning eine kleine, konstante Fehlerrate aufweisen. In seinen eigenen Antworten kann sie dagegen mit der Position des Tokens zunehmen. Erklären Sie, warum das so ist und welcher Teil der Proposition das Modell dann beschreibt.
    1. Warum sind Menschen, abgesehen von ihrer begrenzten Zahl, als Quelle für DAgger-Labels für ein Sprachmodell wenig geeignet? Bei der On-Policy-Destillation erhält jedes Präfix des Schülers als Label die gesamte Verteilung des Lehrers für das nächste Token anstelle eines einzelnen Tokens. Geben Sie den resultierenden Verlust an. Zeigen Sie, dass er in den Verlust des Verhaltensklonens übergeht, wenn der Lehrer einem einzigen Token die gesamte Wahrscheinlichkeit zuweist.

<!--en:8caec13f-->
1. [code] **Erforderliche Demonstrationen für die Abdeckung.** ● Die Codezelle zur Abdeckung in :numref:`subsec_il_distribution_shift` erzeugt den rutschigen See `mdp_ice` und seine optimale Strategie `pi_ice`. Die Wahrscheinlichkeit, dass eine deterministische Strategie `pi` das Ziel erreicht, ist die Wahrscheinlichkeitsmasse im Ziel nach dem Aufruf von `visitation(mdp_ice, one_hot[pi])`.
    1. Berechnen Sie für jeden nichtterminalen Zustand $s$ die Erfolgswahrscheinlichkeit der Strategie, die `pi_ice` folgt, aber in $s$ die Aktion 0 wählt. Nennen wir $s$ kritisch, wenn diese Abweichung die Erfolgswahrscheinlichkeit auf weniger als die Hälfte der Erfolgswahrscheinlichkeit von `pi_ice` senkt. Welche Zustände sind kritisch?
    1. Sei $q_s$ die Wahrscheinlichkeit, dass der Experte in einer Episode den Zustand $s$ besucht. Berechnen Sie $q_s$ exakt mit dem Argument `absorbing` von `visitation`. Machen Sie dazu $s$ absorbierend und lesen Sie die Masse in $s$ nach 100 Schritten ab. Für welchen nichtterminalen Zustand gilt $q_s = 0$, und was folgt daraus für beliebig große Mengen an Expertendaten? Wie wahrscheinlich ist es, dass in einem Datensatz aus drei Episoden der Zustand 13 nie besucht wird?
    1. Nehmen Sie an, dass die Besuche verschiedener kritischer Zustände unabhängig voneinander sind. Sagen Sie anhand ihrer Besuchswahrscheinlichkeiten $q_s$ voraus, mit welcher Wahrscheinlichkeit in $n \in \{1, 3, 10\}$ Expertenepisoden mindestens ein kritischer Zustand undemonstriert bleibt. Vergleichen Sie Ihre Vorhersage mit den von der Codezelle ausgegebenen Häufigkeiten. Ist die Vorhersage zu hoch oder zu niedrig? Was sagt das über Besuche in verschiedenen kritischen Zuständen aus?
   :exlabel:`ex_imitation-demonstrations-coverage`

<!--en:9e1e5755-->
1. [code] **DAgger auf einem längeren Weg.** ● Auf dem See der Größe $4 \times 4$ mit griffigem Eis erklärt die Verteilungsverschiebung etwa ein Fünftel des Verlusts des Klons. Wiederholen Sie das Experiment auf dem See der Größe $8 \times 8$ mit griffigem Eis. Erzeugen Sie ihn mit `gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False)`. Sein optimaler Weg umfasst 14 Züge. Erstellen Sie `env`, `mdp`, `pi_star`, `demo_s` und `demo_a` neu und führen Sie die Codezellen dieses Abschnitts erneut aus.
    1. Berechnen Sie die Erfolgswahrscheinlichkeit des Klons mit Sampling, die Wahrscheinlichkeit, dass er den Weg des Experten verlässt, und die Erfolgswahrscheinlichkeit eines Agenten, der auf jedem Feld die Fehlerrate des Klons aufweist. Welcher Anteil des Verlusts entsteht, nachdem der Klon den Weg verlassen hat? Welchen Anteil können Labels abseits des Weges ausgleichen? Warum wächst der erste Anteil mit der Länge des Weges, der zweite aber nicht?
    1. Führen Sie DAgger für zehn Zufallsstartwerte aus und tragen Sie die Erfolgsrate gegen die Anzahl der Anfragen an den Experten auf. Stellen Sie daneben die Ergebnisse des Verhaltensklonens mit mindestens ebenso vielen Expertenlabels dar.
    1. Eine tabellarische Strategie braucht höchstens ein Label pro Zustand. Zählen Sie nach jeder Runde, wie viele verschiedene Zustände DAgger bis dahin mit Labels versehen hat. Schätzen Sie, wie viele Anfragen nötig wären, wenn DAgger den Experten nur nach Zuständen ohne Label fragte.
   :exlabel:`ex_imitation-dagger-s-budget`

<!--en:2a7f3bd8-->
1. **Die Garantie von DAgger.** ● In der Analyse von :citet:`Ross.Gordon.Bagnell.2011` wählt die Strategie $\pi_i$ zur Datenerhebung in Iteration $i$ bei jedem Schritt mit Wahrscheinlichkeit $\beta_i$ die Expertenaktion und andernfalls die Aktion des aktuellen lernenden Agenten $\hat\pi_i$. Der nächste Agent $\hat\pi_{i+1}$ minimiert den Verlust auf den zusammengeführten Daten der Iterationen $1, \ldots, i$, wie beim erneuten Training mit `clone`. Schreiben Sie $d_\pi = \frac{1}{T}\sum_{t=1}^T d^\pi_t$ für die mittlere Zustandsverteilung von $\pi$. Sei $c(s, \pi) \in [0, c_{\max}]$ ein Klassifikationsverlust von $\pi$ im Zustand $s$, zum Beispiel $\mathbf{1}(\pi(s) \neq \pi^*(s))$. Setzen Sie $c_i(\pi) = \mathbb{E}_{s \sim d_{\pi_i}}[c(s, \pi)]$.
    1. Zeigen Sie, dass $\|d_{\pi_i} - d_{\hat\pi_i}\|_1 \leq 2\min(1, T\beta_i)$ gilt.
    1. Sei $\varepsilon_N = \min_\pi \frac{1}{N}\sum_{i=1}^N c_i(\pi)$ der Verlust der im Nachhinein besten Strategie. Nehmen Sie an, dass der mittlere Regret der lernenden Agenten höchstens $\delta_N$ beträgt.
       $$\frac{1}{N}\sum_{i=1}^N c_i(\hat\pi_i) - \varepsilon_N \leq \delta_N.$$
       Zeigen Sie, dass für mindestens einen Agenten $\hat\pi_j$ der Folge die folgende Schranke gilt.
       $$\mathbb{E}_{s \sim d_{\hat\pi_j}}[c(s, \hat\pi_j)] \leq \varepsilon_N + \delta_N + \frac{2c_{\max}}{N}\sum_{i=1}^N \min(1, T\beta_i).$$
       Im Beweis von Satz 4.1 der Veröffentlichung wird diese Schranke hergeleitet, bevor die letzte Summe vereinfacht wird.
    1. Die Schleife aus :numref:`subsec_il_dagger` verwendet $\beta_1 = 1$ und $\beta_i = 0$ für $i \geq 2$. Spezialisieren Sie die Schranke auf diese Wahl der Parameter und den 0-1-Verlust. Kombinieren Sie sie mit Teil (ii) der Proposition aus :numref:`subsec_il_error_bounds`. Wie muss $N$ mit $T$ wachsen, damit der Term, der weder $\varepsilon_N$ noch $\delta_N$ enthält, beschränkt bleibt?
    1. Bei manchen Zufallsstartwerten im ausgegebenen Lauf schneidet der Agent der ersten Runde schlechter ab als das Verhaltensklonen. Zeigen Sie, dass ein zusätzliches Label allein für Zustand 1 die exakte Erfolgsrate des Klons auf griffigem Eis senkt. Erklären Sie den Rückgang anhand der Expertenaktion im Zustand 1 und der Tabellenzeilen des lernenden Agenten für die Zustände 1 und 2 vor und nach Vergabe des Labels. Was wäre nötig, um den besten Agenten einer Folge zu bestimmen?
   :exlabel:`ex_imitation-dagger-guarantee`

<!--en:0804f968-->
1. **Belohnungen hinter den Demonstrationen.** ● In diesem Abschnitt haben wir aus den Aktionen des Experten gelernt. Beim *inversen bestärkenden Lernen* erschließen wir eine Belohnung, unter der die Demonstrationen optimal sind, und planen anschließend mit dieser Belohnung.
    1. Nach dem Teleskopsummenargument aus :numref:`subsec_vi_shaping` bleibt der Experte bei jeder potenzialbasierten Belohnungsgestaltung $\tilde r(s, a, s') = r(s, a) + \gamma
       \Phi(s') - \Phi(s)$ von $r$ mit $\Phi = 0$ in terminalen Zuständen optimal. Zeigen Sie, dass der Experte auch bei einer Belohnung von konstant null optimal ist. Was folgt aus diesen beiden Tatsachen für die Rekonstruktion von $r$ aus Demonstrationen?
    1. Inverses bestärkendes Lernen mit maximaler Entropie :cite:`Ziebart.Maas.Bagnell.ea.2008` modelliert die Trajektorien einer deterministischen episodischen Aufgabe mit festem Startzustand durch die Verteilung $P_\theta(\tau) \propto \exp\big(\sum_t r_\theta(s_t, a_t)\big)$ über die möglichen Trajektorien. Es passt $\theta$ an, indem es die Likelihood der Demonstrationen maximiert. Warum verwirft dieses Kriterium die Belohnung von konstant null für einen Experten, der das Ziel erreicht? Angenommen, jede mögliche Trajektorie endet in einem terminalen Zustand. Warum kann das Kriterium dann $r$ und die daraus mit $\gamma = 1$ gestaltete Belohnung nicht unterscheiden? Welcher Schritt Ihrer Argumentation gilt für Trajektorien nicht mehr, die wegen eines Zeitlimits abbrechen?

<!--en:98c64ec1-->
[Diskussion](https://d2l.discourse.group/t/36796)

<!--en:2c4e9eab-->
Lernen aus Demonstrationen<br>**Verhaltensklonen · Kovariaten- und Verteilungsverschiebung · quadratische Fehlerakkumulation · DAgger · überwachtes Fine-Tuning und Destillation**

<!--en:61882b83-->
::: {.slide title="Warum ein Klon mit 82% Genauigkeit nur in 66% der Durchläufe Erfolg hat"}

<!--en:b0445376-->
[Motivation und Grundidee]{.kicker}

<!--en:7466c255-->
- **Pomerleaus ALVINN (1989):** Da der Fahrer die Spurmitte hielt, zeigten die Kameraaufnahmen nie, wie man vom Straßenrand zurücklenkt.

<!--en:a1707bed-->
- **FrozenLake auf griffigem Eis:** Ein Klon, der in demonstrierten Zuständen die Aktion von $\pi^*$ mit Wahrscheinlichkeit $0.818$ wählt, erreicht sein Ziel nur in **$66.0\%$** der Episoden, ob nach dem Training mit $3$ Expertenepisoden oder mit $1{,}000$.

<!--en:047aee1b-->
::: {.slide title="Demonstrationen des Experten auf griffigem Eis aufzeichnen"}

<!--en:d3bf98f8-->
[Rückführung auf überwachte Klassifikation]{.kicker}

<!--en:81a21615-->
Eine **Demonstration** ist eine Trajektorie $(s_0, a_0^*, s_1, a_1^*, \ldots)$ des Experten. Das **Verhaltensklonen** verwirft Belohnungen und Dynamik und behält nur die Paare $(s_i, a_i^*)$.

<!--en:51722905-->
- **Schmaler Korridor im Zustandsraum:** Auf griffigem Eis folgt jede Expertenepisode genau derselben Trajektorie über 6 Felder $[0, 4, 8, 9, 13, 14]$ (*abwärts, abwärts, rechts, abwärts, rechts, rechts*). Die Eisfelder $1, 2, 3, 6, 10$ bleiben unbesucht.

<!--en:76e8cf7b-->
::: {.slide title="Die differenzierbare Strategieklasse" only="pytorch"}

<!--en:360bda5c-->
[Softmax-Regression über Präferenzen für Zustand-Aktion-Paare]{.kicker}

<!--en:1a87b3e6-->
`Policy` verbindet ein neuronales oder tabellarisches Score-Modell mit einem Adam-Optimierer und berechnet $\log \pi_\theta(a \mid s)$ mit `log_softmax`.

<!--en:b9e2573c-->
- **Tabellarischer Spezialfall (`Policy.tabular`):** Eine mit Nullen initialisierte Einbettungstabelle der Größe $16 \times 4$ speichert die Präferenzen $\theta_{s,a}$. Daher hat $\pi_\theta(a \mid s) = \exp(\theta_{s,a}) / \sum_{a'} \exp(\theta_{s,a'})$ anfangs für alle Aktionen denselben Wert $1/4$.

<!--en:4794fdbe-->
::: {.slide title="Die differenzierbare Strategieklasse" only="jax"}

<!--en:360bda5c-2-->
[Softmax-Regression über Präferenzen für Zustand-Aktion-Paare]{.kicker}

<!--en:1a87b3e6-2-->
`Policy` verbindet ein neuronales oder tabellarisches Score-Modell mit einem Adam-Optimierer und berechnet $\log \pi_\theta(a \mid s)$ mit `log_softmax`.

<!--en:b9e2573c-2-->
- **Tabellarischer Spezialfall (`Policy.tabular`):** Eine mit Nullen initialisierte Einbettungstabelle der Größe $16 \times 4$ speichert die Präferenzen $\theta_{s,a}$. Daher hat $\pi_\theta(a \mid s) = \exp(\theta_{s,a}) / \sum_{a'} \exp(\theta_{s,a'})$ anfangs für alle Aktionen denselben Wert $1/4$.

<!--en:06b636c4-->
::: {.slide title="Aufstiegsschritt für die gewichtete Log-Likelihood"}

<!--en:cbdfc1b3-->
[Gemeinsamer Grundschritt der Strategieoptimierung]{.kicker}

<!--en:628ece46-->
Sowohl das Verhaltensklonen als auch Policy-Gradient-Verfahren (§{sec_policygradient}–§{sec_deeprl}) rufen `policy_step` auf, um einen Aufstiegsschritt für $\frac{1}{n}\sum_{i=1}^n w_i \log \pi_\theta(a_i \mid s_i)$ auszuführen.

<!--en:2ef2cb89-->
- **Gewichte von eins beim Klonen:** Das Verhaltensklonen setzt für jedes demonstrierte Paar $w_i = 1$ und minimiert damit den üblichen Kreuzentropieverlust für mehrere Klassen.

<!--en:d99fd3c9-->
::: {.slide title="Anpassung des Klons an eine vorgegebene Abweichungsrate"}

<!--en:75dd128a-->
[Kontrolle des Klassifikationsfehlers auf dem Trainingsdatensatz]{.kicker}

<!--en:963d33d3-->
Um reale Klassifikatoren mit begrenzter Kapazität, Aliasing von Merkmalen oder verrauschten menschlichen Labels nachzubilden, stoppt `clone`, sobald die mittlere Abweichungsrate beim Training unter $\varepsilon = 0.20$ fällt.

<!--en:77dca41a-->
- **Zeilen mit und ohne Demonstration:** Nach vierzehn Adam-Schritten gilt auf jedem demonstrierten Feld (etwa $s = 9$) bereits $\pi_{\textrm{bc}}(a^* \mid s) = 0.818$ ($\varepsilon = 0.182$). Das nie demonstrierte Feld $s = 10$ erhält den Gradienten null und behält die Gleichverteilung bei ($0.250$, Abweichung von $\pi^*$ in $3/4$ der Fälle).

<!--en:28e0cd74-->
::: {.slide title="Exakte Fortschreibung der Zustandsverteilung"}

<!--en:ac0cee2b-->
[Strategien ohne Stichprobenrauschen bewerten]{.kicker}

<!--en:f13559ca-->
- **Exakte Erfolgswahrscheinlichkeit für 100 Schritte:** Sind die terminalen Löcher und das Ziel in $P_\pi$ absorbierend, entspricht $d_{100}^\pi(\textrm{goal})$ genau der Erfolgsrate nach 100 Schritten.

<!--en:427a392f-->
::: {.slide title="Der Klon im Einsatz mit gieriger Aktionswahl und mit Sampling"}

<!--en:31ed7e26-->
[Erfolg im Einsatz messen]{.kicker}

<!--en:74721f19-->
Wählt der trainierte Klon seine Aktionen gierig ($\mathrm{argmax}_a \pi_{\textrm{bc}}(a \mid s)$) oder per Sampling ($a_t \sim \pi_{\textrm{bc}}(\cdot \mid s)$, wie bei der Tokenwahl eines Sprachmodells), zeigt sich ein Unterschied von $34$ Prozentpunkten.

<!--en:cf2ae308-->
- **Warum die gierige Wahl zu $100\%$ gelingt:** Auf jedem Feld des 6 Züge langen Weges hat die Expertenaktion die höchste Wahrscheinlichkeit ($0.818 > 0.061$). Mit `argmax` bleibt der Klon daher immer im Korridor.

<!--en:2728f5e4-->
- **Warum die Erfolgsrate mit Sampling auf $66.0\%$ fällt:** Bei $\varepsilon = 0.182$ pro Schritt beträgt die Wahrscheinlichkeit für 6 fehlerfreie Züge in Folge nur $0.818^6 \approx 0.300$. Abseits des Weges sind alle Aktionen mit Wahrscheinlichkeit $1/4$ gleich wahrscheinlich.

<!--en:f3ee12e5-->
::: {.slide title="Verteilungsverschiebung durch die eigene Strategie erkennen"}

<!--en:efb18f6b-->
[Warum mehr Expertendaten nicht helfen]{.kicker}

<!--en:6a6e8950-->
Der Vergleich der Abweichungsrate $1 - \pi_{\textrm{bc}}(\pi^*(s) \mid s)$ des Klons auf drei Zustandsverteilungen zeigt die Ursache des Scheiterns. Es handelt sich um eine **Verteilungsverschiebung** und nicht um Überanpassung.

<!--en:24ccee18-->
- **Keine Generalisierungslücke auf $d^{\pi^*}$:** Die Fehlerrate des Klons beträgt $18.2\%$ auf seinen 18 Trainingspaaren und $18.2\%$ auf 100 neuen Expertenepisoden. Das Klonen von 1, 10 oder 100 Expertenepisoden ergibt dieselbe Tabelle mit $66.0\%$ Erfolgsrate.

<!--en:cd800605-->
- **Höherer Fehler auf $d^{\pi_{\textrm{bc}}}$:** Gewichtet nach der Besuchshäufigkeit des Klons steigt seine Fehlerrate auf **$21.6\%$**. Seine eigenen Fehler führen ihn auf zuvor unbesuchte Felder mit einer Fehlerrate von $75\%$.

<!--en:d5c52488-->
::: {.slide title="Verteilungsverschiebung und fehlende Abdeckung durch den Experten"}

<!--en:49105e94-->
[Zwei verschiedene Lücken in den Daten]{.kicker}

<!--en:5a76f01c-->
Auf *rutschigem* Eis rutscht auch der Experte in verschiedenen Episoden auf unterschiedliche Felder. Das erlaubt einen Vergleich von fehlender Abdeckung bei endlicher Stichprobe und Verteilungsverschiebung.

<!--en:a70a6879-->
- **Fehlende Abdeckung verschwindet mit wachsendem $n$:** Bei $1$ Expertenepisode fehlen in $32.0\%$ der Fälle kritische Felder, bei $3$ Episoden in $1.5\%$ der Fälle. Bei $10$ Episoden fällt kein Klon unter die halbe Erfolgsrate des Experten ($0.0\%$ Ausfälle).

<!--en:6dbbf56d-->
- **Die Verteilungsverschiebung bleibt für jedes $n$ bestehen:** Selbst bei unendlich vielen Expertendaten ($n \to \infty$) erhalten Zustände außerhalb des Trägers der Expertenverteilung ($d^{\pi^*}(s) = 0$) nie ein Label.

<!--en:a36a2b43-->
::: {.slide title="Der Satz zur Fehlerakkumulation"}

<!--en:4d30d636-->
[Quadratisches und lineares Wachstum mit dem Horizont]{.kicker}

<!--en:a3cd7cc3-->
- **$O(\varepsilon T^2)$ beim Verhaltensklonen:** Ein erster Fehler im Schritt $t$ kann die Belohnungen aller $T - t + 1$ verbleibenden Schritte kosten.

<!--en:4e8282e0-->
- **$O(u\varepsilon T)$ mit einer Teleskopsumme über Mischstrategien:** Aufeinanderfolgende Mischstrategien $\pi_{k-1}, \pi_k$ unterscheiden sich nur im Schritt $k$ auf $d_k^{\hat{\pi}}$. Pro Schritt gehen höchstens $u\varepsilon$ verloren.

<!--en:cd2d5070-->
::: {.slide title="Quadratisches und lineares Fehlerwachstum auf einer Kette mit 64 Schritten"}

<!--en:5e8f4a2f-->
[Wie der Verlust mit dem Horizont wächst]{.kicker}

<!--en:92fecc07-->
Auf einer Kette mit $T$ Schritten erhält der Experte pro Schritt $1$, und die Nachahmer irren mit Wahrscheinlichkeit $\varepsilon = 0.01$. Die Verluste beim Klonen und bei möglicher Rückkehr ($u = 1$) entwickeln sich unterschiedlich.

<!--en:4757eb80-->
- **Geklonter Nachahmer:** Er erhält die Belohnung im Schritt $t$ mit Wahrscheinlichkeit $(1-\varepsilon)^t$. Ihm entgehen **$17.03$** Belohnungen bei $T = 64$ (nahe der Referenz $\varepsilon T(T+1)/2 = 20.80$).

<!--en:92243490-->
- **Zurückkehrender Nachahmer:** Er kehrt nach 1 verlorenen Schritt auf den Weg zurück ($u = 1$). Ihm entgehen genau $\varepsilon T = \mathbf{0.64}$ Belohnungen, **um den Faktor $27\times$ weniger**.

<!--en:3b9b41e8-->
::: {.slide title="Wo der Klon auf griffigem Eis in FrozenLake vom Weg abkommt"}

<!--en:c28ab8f7-->
[Besuchswahrscheinlichkeiten der Zustände]{.kicker}

<!--en:a036ea11-->
Die Besuchswahrscheinlichkeiten der 16 Felder zeigen genau, wo der Klon im Einsatz den 6 Züge langen Weg des Experten verlässt.

<!--en:73de3916-->
- **Drei Ausgänge vom Weg (gestrichelte Pfeile):** In **$17.2\%$** der Episoden verlässt der Klon den blauen Expertenweg. Er geht von Feld 0 nach rechts auf Feld 1, von Feld 9 nach rechts oder von Feld 14 nach oben auf Feld 10.

<!--en:0b819fe4-->
- **Zufallsbewegung ohne Labels:** Auf Feld 1 oder 10 wählt der Klon zufällig unter allen Richtungen ($1/4$ pro Richtung) und gerät häufig in die Löcher 5, 7 und 11.

<!--en:e68e0353-->
::: {.slide title="Wo der Klon 34 Prozentpunkte an Erfolgsrate verliert"}

<!--en:ccc6dce7-->
[Anpassungsfehler auf dem Weg und Verteilungsverschiebung abseits davon]{.kicker}

<!--en:8e9c777e-->
Sind die Felder abseits des Weges absorbierend, zerfällt der Verlust von $34.0$ Prozentpunkten ($1.000 \to 0.660$) in Fehler auf dem Weg und Fehler nach der Verteilungsverschiebung abseits des Weges.

<!--en:7f04a97f-->
- **Stürze vom Weg in ein Loch ($23.6$ Prozentpunkte):** In den $82.8\%$ der Episoden, in denen der Klon den Weg nie verlässt, erreicht er in $71.5\%$ der Fälle das Ziel ($0.828 \times 0.285 = 0.236$ entfallen auf die benachbarten Löcher 5 und 12).

<!--en:dbde9acd-->
- **Fehlschläge abseits des Weges ($10.4$ Prozentpunkte):** In den $17.2\%$ der Episoden, in denen der Klon den Weg verlässt, erreicht er nur in $39.4\%$ der Fälle doch noch das Ziel ($0.172 \times 0.606 = 0.104$ gehen verloren). Erhalten alle Felder abseits des Weges Labels mit derselben Fehlerrate von $18.2\%$, steigt die Gesamterfolgsrate **von $0.660$ auf $0.735$**.

<!--en:4cf2d70b-->
::: {.slide title="DAgger und die Aggregation von Datensätzen"}

<!--en:26b4af2b-->
[Zustände vom lernenden Agenten, Labels vom Experten]{.kicker}

<!--en:f910b1f9-->
- **Interaktive Vergabe von Labels (Ross et al., 2011):** Den aktuellen Agenten $\hat{\pi}_k$ agieren lassen und den Experten $\pi^*$ nach der Aktion für jeden tatsächlich besuchten Zustand fragen.

<!--en:be67064f-->
- **Daten aller Runden zusammenführen:** Die neuen Paare zum Datensatz hinzufügen gemäß $\mathcal{D}_k = \mathcal{D}_{k-1} \cup \mathcal{D}_{\textrm{new}}$ und $\hat{\pi}_{k+1}$ erneut anpassen. So bleiben frühere Zustände im Training erhalten.

<!--en:d985e8e8-->
::: {.slide title="Vier DAgger-Runden gleichen den Verlust durch fehlende Labels abseits des Weges aus"}

<!--en:fbcd4abc-->
[Beobachtete Konvergenz zum Grenzwert bei vollständiger Abdeckung]{.kicker}

<!--en:2def91f0-->
In 4 DAgger-Runden (10 Episoden des lernenden Agenten pro Runde, 10 Zufallsstartwerte) schließt sich die Lücke durch die Verteilungsverschiebung. Die Erfolgsrate steigt von $0.660$ auf $0.735$, den Grenzwert bei vollständiger Abdeckung.

<!--en:670727a6-->
- **Warum DAgger bei $0.735$ stehen bleibt:** Sobald der Agent die Felder 1, 2, 6 und 10 besucht und dort Expertenlabels erhält, gilt für jeden erreichbaren Zustand $\varepsilon = 0.182$. Die verbleibende Lücke ($0.735$ gegenüber $1.000$) entsteht durch den Klassifikationsfehler $\varepsilon$ pro Schritt.

<!--en:a02cd973-->
::: {.slide title="Überwachtes Fine-Tuning und On-Policy-Destillation"}

<!--en:ef2970b1-->
[Imitationslernen bei großen Sprachmodellen]{.kicker}

<!--en:b8f3919f-->
Im MDP auf Tokenebene aus §{sec_mdp} entsprechen gängige Schritte des Post-Trainings großer Sprachmodelle unmittelbar dem Verhaltensklonen und DAgger.

<!--en:caa2f926-->
| Konzept des Imitationslernens | Entsprechung im Post-Training eines Sprachmodells |

<!--en:450f4710-->
| **Zustand $s_t$ und Aktion $a_t$** | Prompt mit erzeugtem Präfix $(x, y_{<t})$ und nächstes Token $y_t$ |

<!--en:44572b2c-->
| **Verhaltensklonen auf $\mathcal{D}^*$** | **Überwachtes Fine-Tuning (SFT):** Kreuzentropie auf kuratierten Expertenantworten |

<!--en:ec50c1a1-->
| **Verteilungsverschiebung im Einsatz** | Autoregressiver Exposure-Bias, weil das Modell seine selbst gezogenen Token als Kontext verwendet |

<!--en:59ef564c-->
| **Kosten der Fehlerkorrektur $u$ und Horizont $T$** | $u$ ist bei einem stilistischen Synonym klein, bei einem frühen Vorzeichenfehler in einer Rechnung gilt dagegen $u \approx T - t$ |

<!--en:971925b9-->
| **DAgger mit interaktivem Experten** | **On-Policy-Destillation:** Der Schüler erzeugt Präfixe $y_{<t} \sim \pi_\theta$, der Lehrer liefert die Labels $\pi^*(\cdot \mid x, y_{<t})$ |

<!--en:2c49e808-->
::: {.slide title="Verständnisfragen zu Imitationslernen und DAgger"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:7a1a4355-->
Fragen zu Verteilungsverschiebung, Fehlerakkumulation und DAgger

<!--en:2ca377af-->
1. **Mehr Daten:** Auf griffigem Eis in FrozenLake wächst der Datensatz für das Verhaltensklonen von $3$ auf $1{,}000$ Expertenepisoden. Warum bleibt die Erfolgsrate des Klons mit Sampling bei genau $0.660$?

<!--en:5eb90420-->
2. **Quadratische und lineare Schranke:** Bei welcher Art von Aufgabe ergibt Teil (ii) der Proposition zur Fehlerakkumulation ($u\,\varepsilon\,T$) wieder die quadratische Größenordnung $O(\varepsilon\,T^2)$, selbst wenn $\varepsilon$ auf $d_t^{\hat{\pi}}$ klein ist?

<!--en:1bf2b8bd-->
3. **Die verbleibende Lücke schließen:** Nach 4 DAgger-Runden auf griffigem Eis in FrozenLake steigt die Erfolgsrate des lernenden Agenten nicht mehr über $0.735$. Was muss sich ändern, damit sie von $0.735$ weiter in Richtung $1.000$ steigt?

<!--en:557b98b6-->
**Antworten:**

<!--en:6235286f-->
1. Auf deterministischem, griffigem Eis folgt der Experte in jeder Episode demselben Weg über 6 Felder. Weitere $1{,}000$ Episoden wiederholen diese 6 Paare. Die Felder $1, 2, 3, 6, 10$ abseits des Weges erhalten nie ein Label.

<!--en:4a68f0ae-->
2. Ist ein einzelner Fehler endgültig oder nicht mehr auszugleichen (etwa ein Sturz in ein Loch), entsprechen die Kosten $u$ eines einzigen Fehlers dem verbleibenden Horizont $\approx T - t$. Damit gilt $u\,\varepsilon\,T = \Theta(\varepsilon\,T^2)$.

<!--en:325b514d-->
3. Die Fehlerwahrscheinlichkeit $\varepsilon$ pro Schritt muss unter $0.182$ sinken. Das gelingt durch längeres Training oder durch gierige Aktionswahl im Einsatz. DAgger behebt nur die Verteilungsverschiebung. Der zugrunde liegende Klassifikationsfehler $\varepsilon$ bleibt bestehen.

<!--en:0e99ede7-->
::: {.slide title="Zusammenfassung der Verfahren zum Imitationslernen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:0d3fd178-->
| Verfahren oder Einsatzart | Trainingsverteilung | Erwartete Ertragsdifferenz und wesentliche Eigenschaften |

<!--en:c1c97352-->
| **Verhaltensklonen (BC)** | Expertenzustände $s \sim d_t^{\pi^*}$ | $\leq \varepsilon T(T+1)/2 = O(\varepsilon T^2)$. Weder Interaktion mit der Umgebung noch Belohnung erforderlich |

<!--en:e889f802-->
| **Gierige Aktionswahl beim Einsatz von BC** | Expertenzustände $s \sim d_t^{\pi^*}$ | Bei deterministischen Aufgaben erfolgreich, wenn auf dem Weg $\mathrm{argmax}_a \hat{\pi}(a \mid s) = \pi^*(s)$ gilt |

<!--en:6cf19062-->
| **DAgger** | Zusammengeführte Zustände der lernenden Agenten $s \sim d_t^{\hat{\pi}_k}$ | $\leq u\,\varepsilon\,T = O(u\varepsilon T)$ bei Aufgaben mit korrigierbaren Fehlern. Experte muss auf Anfrage Labels liefern |

<!--en:a6815f37-->
| **SFT für Sprachmodelle** | Kuratierte Präfixe von Menschen oder Lehrermodellen | Exposure-Bias bei langen Argumentationsketten, weil Fehler den Kontext verschieben |

<!--en:ce713421-->
| **On-Policy-Destillation** | Präfixe des Schülers mit neuen Labels vom Lehrer | Setzt DAgger für LLMs mit der Tokenverteilung des Lehrers $\pi^*(\cdot \mid x, y_{<t})$ um |

<!--en:4da0c222-->
**Ausblick (§{sec_optimal_control} Optimale Regelung und §{sec_policygradient} Policy-Gradient):** Fehlt ein Experte zum Nachahmen, lässt sich mittels LQR, iLQR und MPC direkt durch bekannte kontinuierliche physikalische Gleichungen $\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$ differenzieren (§{sec_optimal_control}). Alternativ lässt sich eine parametrisierte Strategie $\pi_\theta(a \mid s)$ anhand gezogener skalarer Belohnungen mit dem Policy-Gradienten über die Score-Funktion optimieren (§{sec_policygradient}).
