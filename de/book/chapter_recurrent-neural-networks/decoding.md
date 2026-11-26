<!--en:7d2738dd-->
# Dekodierung und Textgenerierung

<!--en:08fb384c-->
Bisher haben wir Sprachmodelle in diesem Kapitel allein danach beurteilt, wie gut sie die Wahrscheinlichkeiten für das nächste Token angeben. Um Text zu *generieren*, muss das Modell in jedem Schritt auch ein konkretes Token aus der vorhergesagten Verteilung auswählen. Diese Entscheidung hat überraschend großen Einfluss: Je nach Auswahlregel erzeugt ein und dasselbe Modell völlig unterschiedliche Texte. In :numref:`sec_rnn-scratch` haben wir gesehen, wo die einfachste Regel scheitert: Wählt der Generator in jedem Schritt gierig das wahrscheinlichste Token, gerät er schnell in Wiederholungsschleifen. Wie gewinnen wir also aus den Verteilungen für das nächste Token einen flüssigen Text?

<!--en:783cf422-->
Das Verfahren, das eine Folge von Verteilungen für das jeweils nächste Token in Ausgabetoken umsetzt, heißt *Dekodierung* :cite:`Graves.2013,Holtzman.Buys.Du.ea.2020`. Die Dekodierstrategien lassen sich in zwei große Familien einteilen. Verfahren zur *Maximierung* suchen nach einer Ausgabesequenz mit hoher gemeinsamer Wahrscheinlichkeit unter dem Modell. Beim *Sampling* ziehen wir jedes Token zufällig, nachdem wir den unzuverlässigen Rand der Verteilung umgeformt oder abgeschnitten haben. Im Folgenden entwickeln wir die Verfahren beider Familien und vergleichen sie miteinander. Entscheidend ist, dass jede Dekodierregel vom Sprachmodell nur die Verteilung für das nächste Token benötigt. Unsere Algorithmen eignen sich daher unverändert für jedes autoregressive Sprachmodell, ob RNN, Zustandsraummodell oder Transformer.

<!--en:bb75ee22-->
## Das Dekodierproblem

<!--en:e8bcff04-->
Ein trainiertes Sprachmodell liefert bedingte Verteilungen für das nächste Token. Zu jedem Präfix $x_{<t} = x_1, \ldots, x_{t-1}$ gibt es $P(x_t \mid x_{<t})$ aus. Nach der Kettenregel der Wahrscheinlichkeitsrechnung (:numref:`sec_sequence`) bestimmen diese bedingten Verteilungen die gemeinsame Wahrscheinlichkeit jeder vollständigen Sequenz gemäß $P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$.

<!--en:1d6d7b00-->
Bei der Dekodierung stellt sich die umgekehrte Frage: Welche Sequenz sollen wir ausgeben, wenn wir diese bedingten Verteilungen kennen? Es liegt nahe, die *wahrscheinlichste* Sequenz zu suchen. Eine exakte Maximierung ist jedoch praktisch unmöglich. Bei einem Vokabular $\mathcal{V}$ und einem Horizont $T$ gibt es $|\mathcal{V}|^T$ mögliche Fortsetzungen. Für unser Vokabular mit 1027 Token und eine Fortsetzung aus 50 Token sind das $1027^{50} > 10^{150}$ Kandidaten, weit mehr, als wir je aufzählen könnten. Jeder praktikable Decoder kann daher nur einen winzigen Teil des Suchbaums erkunden und muss in jedem Schritt entscheiden, welche Präfixe er beibehält.

<!--en:2c25784d-->
Selbst wenn eine exakte Suche möglich wäre, ist die wahrscheinlichste Sequenz häufig *nicht* die gewünschte. Lässt eine Aufgabe nur wenige korrekte Ausgaben zu, etwa beim Transkribieren von Sprache oder beim Übersetzen eines technischen Satzes, bewährt sich die Suche nach einer wahrscheinlichen Sequenz. Sind hingegen viele Fortsetzungen gleichermaßen gültig, etwa im Dialog oder beim kreativen Schreiben, entfällt auf jede einzelne zwangsläufig nur wenig Wahrscheinlichkeitsmasse. Denn ein gut kalibriertes Modell verteilt die Wahrscheinlichkeit auf viele plausible Pfade. Zwingen wir das Modell, den wahrscheinlichsten Pfad zu wählen, geht diese natürliche Vielfalt verloren. Sampling erhält sie, weil wir direkt aus der erlernten Verteilung ziehen.

<!--en:020919dc-->
:numref:`fig_decoding-taskmap` ordnet die Dekodierstrategien nach dieser Unterscheidung. Maximierungsverfahren wie gieriges Dekodieren (engl. greedy decoding) und Beam-Search sind nach wie vor die Standardwahl für eng umrissene Aufgaben mit wenigen gültigen Ausgaben. Für die offene Textgenerierung sind Sampling-Verfahren mit Temperaturskalierung oder mit abgeschnittener Verteilung üblich.

<!--en:9442431c-->
![Zwei Familien von Dekodierregeln. Die Suche nach wahrscheinlichen Sequenzen eignet sich für Aufgaben mit wenigen zulässigen Ausgaben. Sampling erhält die Vielfalt bei der offenen Textgenerierung.](../img/mdl-rnn-decoding-taskmap.svg)

<!--en:2dec320f-->
### Ein Sprachmodell für die Dekodierung

<!--en:2949d534-->
Für unsere Experimente verwenden wir die kompakte Implementierung des RNN-Sprachmodells aus :numref:`sec_rnn-concise`. Es hat 512-dimensionale Einbettungen, 128 versteckte Einheiten und eine Ausgabeschicht für die 1027 BPE-Token aus *The Time Machine*. Dropout wirkt auf die Einbettungen und die versteckten Zustände. Zunächst laden wir die Trainings- und Validierungsfenster mit je 32 Token aus `d2l.TimeMachine`.

<!--en:534b9908-->
Als Nächstes erstellen wir die kompakte rekurrente Zelle `d2l.RNN` und das Sprachmodell `d2l.RNNLM` mit $d = 512$, $h = 128$, Dropout und der Lernrate `lr=4`.

<!--en:411f46c6-->
Trainieren wir `model` über 10 Epochen mit Gradienten-Clipping bei `gradient_clip_val=1`, erreichen wir für *The Time Machine* erneut eine Validierungsperplexität von etwa 60.

<!--en:366c915e-->
Alle Dekodieralgorithmen in diesem Abschnitt greifen über dieselbe modellunabhängige Schnittstelle `step_fn(ids)` auf das Sprachmodell zu. Sie nimmt die Token-IDs des bisherigen Präfixes entgegen und liefert einen eindimensionalen NumPy-Vektor mit den Logits für das nächste Token. Dank einfacher NumPy-Arrays als Rückgabewerte schreiben wir jeden Decoder nur einmal für alle vier Frameworks. Wir verzichten bewusst auf einen expliziten versteckten Zustand: Jedes autoregressive Modell, ob rekurrentes Netz oder Transformer, passt zu dieser Signatur.

<!--en:24cee4d5-->
`tf.function` mit `jit_compile=True` kompiliert den Vorwärtsdurchlauf mit XLA einmal pro Eingabeform. Wir füllen daher jedes Präfix mit Nullen auf ein Vielfaches von 32 Token auf, sodass nur wenige Formen kompiliert werden. Das Netz liest die Sequenz von links nach rechts. Die Logits an der letzten echten Position bleiben durch das Padding unverändert. Genau diese Logits lesen wir aus.

<!--en:780a38f6-->
`nnx.jit` kompiliert den Vorwärtsdurchlauf einmal pro Eingabeform. Wir füllen daher jedes Präfix mit Nullen auf ein Vielfaches von 32 Token auf, sodass nur wenige Formen kompiliert werden. Das Netz liest die Sequenz von links nach rechts. Die Logits an der letzten echten Position bleiben durch das Padding unverändert. Genau diese Logits lesen wir aus.

<!--en:d0f0069b-->
Diese Schnittstelle kostet Rechenzeit: Bei jedem Aufruf werten wir das Netz erneut für das *gesamte* Präfix aus, nur um einen einzigen Vektor mit Logits zu erhalten. Bei einem RNN lässt sich diese Mehrarbeit vermeiden. Auf die Rechenkosten kommen wir am Ende des Abschnitts zurück.

<!--en:820462cc-->
## Gieriges Dekodieren

<!--en:9b9bef28-->
Der einfachste Decoder wählt in jedem Schritt das eine Token mit der höchsten bedingten Wahrscheinlichkeit:

<!--en:b61727ba-->
Wir bewerten die möglichen Fortsetzungen des Präfixes, wählen das Token mit dem höchsten Wert, hängen es an und wiederholen den Vorgang. Gieriges Dekodieren benötigt nur eine Modellauswertung pro erzeugtem Token und ist vollständig deterministisch: Dasselbe Präfix führt immer zu genau derselben Fortsetzung. Die folgende Hilfsfunktion `show` dekodiert Token-IDs zu einer Zeichenkette in Anführungszeichen. Zeilenumbrüche zeigt sie als `\n` an und nicht druckbare Bytes ersetzt sie durch `U+FFFD`.

<!--en:6afb7af1-->
Wie in :numref:`sec_rnn-scratch` geraten beide Fortsetzungen schnell in Wiederholungsschleifen. Gieriges Dekodieren hat zwei grundlegende Schwächen. Die erste ist seine *Kurzsichtigkeit*: Eine Folge lokal optimaler Entscheidungen ergibt nicht unbedingt die global optimale Sequenz. Angenommen, für das erste Token gelten $P(A) = 0.5$ und $P(B) = 0.4$. Nach $A$ ist der zweite Schritt unsicher (höchste Wahrscheinlichkeit einer Fortsetzung: $0.3$), nach $B$ dagegen nahezu eindeutig (höchste Wahrscheinlichkeit: $0.9$). Gieriges Dekodieren legt sich sofort auf $A$ fest und erreicht eine gemeinsame Wahrscheinlichkeit von $0.5 \times 0.3 = 0.15$. Damit übersieht es die weitaus wahrscheinlichere Sequenz mit Anfang $B$ und Wahrscheinlichkeit $0.4 \times 0.9 = 0.36$ (:numref:`fig_greedy-myopia`).

<!--en:fb320ca7-->
![Die Kurzsichtigkeit des gierigen Dekodierens an einem Beispiel mit zwei Schritten. Das Verfahren wählt das wahrscheinlichere erste Token $A$ und erhält die Sequenz $AC$ mit der gemeinsamen Wahrscheinlichkeit $0.5 \times 0.3 = 0.15$. Der Pfad über $B$ bleibt unberücksichtigt, obwohl er $0.4 \times 0.9 = 0.36$ erreicht.](../img/mdl-rnn-greedy-myopia.svg)

<!--en:3cd4f59e-->
Die zweite Schwäche ist die Wiederholung selbst. Beim gierigen Dekodieren fließt in jedem Schritt das deterministisch gewählte Maximum wieder in die Rekursion ein. Der nächste versteckte Zustand ist damit eine deterministische Funktion des aktuellen Zustands. Gerät die Trajektorie in einen anziehenden Zyklus, gibt es keinen Zufallseinfluss, der sie daraus befreien könnte. Das betrifft auch wesentlich größere Netze: :citet:`Holtzman.Buys.Du.ea.2020` beobachteten genau diese *Degeneration neuronal erzeugter Texte* (engl. neural text degeneration) in tausendmal größeren Sprachmodellen. Sie zeigten, dass die Maximierung der Likelihood bei offener Textgenerierung regelmäßig eintönige Texte mit vielen Wiederholungen liefert, die sich deutlich von menschlichen Texten unterscheiden.

<!--en:38fbc93f-->
## Beam-Search

<!--en:fefe0f47-->
Gieriges Dekodieren legt sich sofort auf einen Pfad fest, die vollständige Suche verfolgt jeden möglichen Pfad. Die *Beam-Search* liegt dazwischen. Ein einziger Hyperparameter bestimmt, wie breit sie sucht: die *Beam-Breite* (engl. beam width) $w$. In jedem Schritt behält die Beam-Search die $w$ Präfixe mit dem höchsten Score. Sie beginnt mit den $w$ wahrscheinlichsten ersten Token. In jedem weiteren Schritt erweitert sie jedes verbliebene Präfix um jedes Token des Vokabulars, bewertet alle $w \cdot |\mathcal{V}|$ möglichen Erweiterungen nach ihrer kumulierten Log-Wahrscheinlichkeit und behält die besten $w$ Sequenzen. Mit $w = 1$ erhalten wir gieriges Dekodieren. Ist $w$ so groß, dass nie ein Kandidat ausscheidet, ergibt sich die vollständige Suche.

<!--en:831a3bd7-->
![Beam-Search mit Beam-Breite 2 und einem Vokabular aus fünf Token. In jedem Schritt wird jedes verbliebene Präfix um jedes Token des Vokabulars erweitert. Nur die beiden Präfixe mit dem höchsten Score bleiben erhalten: hier zunächst $A$ und $C$, dann $AB$ und $CE$ und schließlich $ABD$ und $CED$.](../img/mdl-rnn-beam-tree.svg)

<!--en:eaf1d5c2-->
:numref:`fig_beam-search` zeigt drei Schritte der Beam-Search mit $w = 2$ für das Vokabular $\mathcal{V} = \{A, B, C, D, E\}$ aus fünf Token. Den gemeinsamen Prompt lassen wir in der Notation weg. Angenommen, $A$ und $C$ sind die beiden wahrscheinlichsten ersten Token. In Schritt 2 bewerten wir alle zehn Erweiterungen um ein Token $x' \in \mathcal{V}$,

<!--en:da32bd52-->
und behalten die beiden besten, etwa $AB$ und $CE$. In Schritt 3 wiederholen wir die Erweiterung und behalten $ABD$ und $CED$. Von den $5^3 = 125$ möglichen Sequenzen bewertet die Beam-Search also nur $5 + 10 + 10 = 25$. Allgemein kostet die Erzeugung von $T$ Token $O(w |\mathcal{V}| T)$. Das ist nur das $w$-Fache des Aufwands beim gierigen Dekodieren und exponentiell weniger als bei der vollständigen Suche.

<!--en:9ad4f8c0-->
Darf eine Sequenz durch Ausgabe eines Sequenzendtokens vorzeitig enden, müssen wir beim Vergleich unterschiedlich langer Kandidaten ihre Länge berücksichtigen. Da die Log-Wahrscheinlichkeit jedes Tokens nichtpositiv ist, sinkt die kumulierte Log-Wahrscheinlichkeit mit wachsender Sequenzlänge monoton. Das bevorzugt kurze Hypothesen zu Unrecht. Üblicherweise verwenden wir daher eine *Längennormalisierung* (engl. length normalization) :cite:`Wu.Schuster.Chen.ea.2016`. Für einen Kandidaten, der $T$ Token an ein Präfix der Länge $t_0$ anhängt, berechnen wir den Score gemäß

<!--en:3726c695-->
Dabei interpoliert der Exponent $\beta$ (typischerweise etwa $0.75$) zwischen der unnormierten Log-Wahrscheinlichkeit ($\beta = 0$) und der mittleren Log-Wahrscheinlichkeit pro Token ($\beta = 1$). Im folgenden Code bezeichnen wir dieses Argument als `alpha`, wie bei :citet:`Wu.Schuster.Chen.ea.2016`.

<!--en:d1f9e36b-->
Unsere Implementierung von `beam_search` übernimmt abgeschlossene Hypothesen (mit `eos_id` am Ende) unverändert und erweitert nur aktive Präfixe. Dabei können wir Rechenaufwand sparen: Unter den insgesamt besten $w$ Kandidaten können höchstens $w$ Erweiterungen eines einzelnen Präfixes sein. Daher genügen pro aktivem Präfix die besten $w$ Folgetoken. Wir brauchen nicht alle $|\mathcal{V}|$ Erweiterungen zu sortieren.

<!--en:854ff1b2-->
Dekodieren wir dieselben beiden Prompts mit $w = 1$ (also gierig) und mit $w = 4$. Für die jeweils beste Hypothese geben wir den längennormalisierten Score aus :eqref:`eq_beam-search-score` aus. Da unser Korpus kein Sequenzendtoken besitzt, erreichen alle Kandidaten die Länge 30. Der konstante Nenner $30^{0.75} \approx 12.8$ lässt ihre Rangfolge unverändert.

<!--en:c38b9477-->
Mit größerer Beam-Breite finden wir im Allgemeinen wahrscheinlichere Fortsetzungen als beim gierigen Dekodieren. Garantiert ist das nicht, denn bei der näherungsweisen Auswahl kann weiterhin ein Präfix ausscheiden, das sich später gelohnt hätte. Eine höhere Wahrscheinlichkeit bedeutet auch nicht automatisch *bessere* Prosa. Bei der offenen Textgenerierung erzeugen große Sprachmodelle mit Beam-Search deutlich mehr Wiederholungen, als sie in menschlichen Texten vorkommen :cite:`Holtzman.Buys.Du.ea.2020`. Diesen Unterschied messen wir in Kürze. Für eng umrissene Aufgaben wie Spracherkennung und maschinelle Übersetzung bleibt die Beam-Search mit mäßiger Breite (typischerweise 4 bis 10) dagegen der Standard :cite:`Sutskever.Vinyals.Le.2014,Wu.Schuster.Chen.ea.2016`.

<!--en:21d45b56-->
## Sampling

<!--en:247e3da9-->
Hat unser Modell eine sinnvolle Verteilung über Fortsetzungen gelernt, ist direktes *Ancestral Sampling* die naheliegendste Alternative zur Maximierung: Wir ziehen $x_t \sim P(x_t \mid x_{<t})$, hängen das Token an und wiederholen den Vorgang. Sampling stellt die Vielfalt wieder her, die bei der Maximierung verloren geht. Beim reinen Sampling stoßen wir jedoch auf ein Problem am Rand der Verteilung. Die Softmax-Funktion weist jedem Eintrag im Vokabular eine positive Wahrscheinlichkeit zu (hier sind es rund tausend Token, in modernen großen Sprachmodellen über hunderttausend). In jedem Schritt summieren sich die winzigen Wahrscheinlichkeiten tausender unplausibler Token zu einer beträchtlichen Masse. Beim reinen Sampling ziehen wir daher regelmäßig ein völlig unplausibles Token. Fließt es in den Kontext ein, kann der ganze restliche Text entgleisen. Genau diese Fehlerfortpflanzung bei der rekursiven Vorhersage haben wir in :numref:`sec_sequence` besprochen. :citet:`Holtzman.Buys.Du.ea.2020` bezeichneten dieses Phänomen als *unzuverlässigen Rand* (engl. unreliable tail): Für die wahrscheinlichsten Token ist das Sprachmodell gut kalibriert, weit draußen am Rand sind seine Wahrscheinlichkeiten jedoch verrauscht. In der Praxis formen wir die Verteilung vor dem Sampling um oder schneiden ihren Rand ab. Vier verbreitete Verfahren leisten dies auf unterschiedliche Weise.

<!--en:69a3e2c4-->
Bei der **Temperaturskalierung** teilen wir jedes Logit $o_x$ durch einen positiven Skalar $\tau > 0$, bevor die Softmax-Funktion zum Einsatz kommt:

<!--en:45022ec3-->
Senken wir $\tau$ unter 1, konzentriert sich die Verteilung stärker auf die wahrscheinlichsten Token (für $\tau \to 0$ ergibt sich im Grenzfall gieriges Dekodieren). Erhöhen wir $\tau$ über 1, wird die Verteilung flacher und nähert sich der Gleichverteilung. Die Temperatur skaliert jedoch die gesamte Verteilung auf dieselbe Weise und ist daher ein grobes Werkzeug: Senken wir die Temperatur der Softmax-Funktion so weit, dass der unzuverlässige Rand kaum noch Gewicht hat, geht zugleich die erwünschte Vielfalt unter den plausiblen Token verloren.

<!--en:26068160-->
**Top-$k$-Sampling** schneidet nach Rang ab: Wir behalten nur die $k$ wahrscheinlichsten Token, normieren ihre Wahrscheinlichkeiten erneut und ziehen daraus :cite:`Fan.Lewis.Dauphin.2018,Radford.Wu.Child.ea.2019`. Beim **Top-$p$**-Sampling (auch *Nucleus*-Sampling) entscheidet dagegen die kumulierte Wahrscheinlichkeitsmasse. Wir behalten von den wahrscheinlichsten Token gerade so viele, dass ihre Wahrscheinlichkeiten in der Summe mindestens $p$ ergeben (üblich ist $p = 0.9$) :cite:`Holtzman.Buys.Du.ea.2020`. Der Unterschied zählt, weil die Entropie des Modells von Token zu Token stark schwankt. Steht das nächste Wort nahezu fest, lässt ein festes $k = 20$ neunzehn schlechte Kandidaten zu. Passen dagegen Dutzende Wörter gleich gut, kann $k = 20$ die tatsächliche Vielfalt beschneiden. Top-$p$ passt sich dynamisch an: Bei stark konzentrierten Verteilungen schrumpft die Kandidatenmenge, bei flachen wächst sie. Überschreitet schon das wahrscheinlichste Token die Schwelle $p$, kann die Auswahl dennoch zu gierig ausfallen (etwa bei Wahrscheinlichkeit $0.95$ und $p = 0.9$, wenn dadurch ein guter zweiter Kandidat ausscheidet).

<!--en:acb0f461-->
**Min-$p$-Sampling** setzt die Schwelle relativ zur höchsten vom Modell vorhergesagten Wahrscheinlichkeit :cite:`Nguyen.Baker.Neo.ea.2025`. Es behält jedes Token, dessen Wahrscheinlichkeit mindestens das $p_{\min}$-Fache der höchsten Wahrscheinlichkeit beträgt (zum Beispiel $p_{\min} = 0.05$). Ist das Modell sicher, steigt die Schwelle automatisch und lässt nur aussichtsreiche Kandidaten zu. Ist es unsicher, sinkt die Schwelle, sodass mehr Möglichkeiten zur Wahl stehen. :citet:`Nguyen.Baker.Neo.ea.2025` zeigen, dass die Qualität bei erhöhter Temperatur mit dieser relativen Schwelle deutlich langsamer abnimmt als mit Top-$p$. Dort erfasst eine feste Massenschwelle einen großen Teil des abgeflachten Randes.

<!--en:7fba9ca8-->
### Eine gemeinsame Funktion für das Sampling

<!--en:32830180-->
Temperatur, Top-$k$, Top-$p$ und Min-$p$ wirken alle auf dieselbe Verteilung für das nächste Token. Daher genügt für jede Kombination eine einzige kompakte Funktion. Die folgende Funktion `sample_next` liefert für `temperature == 0` das Token mit dem höchsten Logit. Andernfalls berechnet sie $P_\tau$ nach :eqref:`eq_temperature` und sortiert die Token nach absteigender Wahrscheinlichkeit. Für jede aktive Regel bestimmt sie, wie viele der führenden Token bleiben dürfen, und wählt die engste Grenze. Anschließend normiert sie die verbleibenden Wahrscheinlichkeiten erneut und zieht mit dem NumPy-Zufallsgenerator `rng` ein Token.

<!--en:6ffea234-->
Die Funktion `generate` fasst die Schleife für autoregressives Dekodieren zusammen: Sie ruft `step_fn` auf, zieht mit `sample_next` das nächste Token und hängt es an. Dabei reicht sie die Schlüsselwortargumente für die gewählte Strategie weiter. Gibt das Modell `eos_id` aus, endet die Schleife vorzeitig. Da die offene Prosa aus *The Time Machine* kein Sequenzendtoken enthält, erzeugen wir in unseren Beispielen eine feste Anzahl von Token.

<!--en:02e37da8-->
An einer Beispielverteilung über fünf Token mit den Wahrscheinlichkeiten $0.45, 0.25, 0.15, 0.10$ und $0.05$ können wir jede Regel direkt überprüfen. Top-$k$ mit $k = 2$ behält nur die Token 0 und 1. Top-$p$ mit $p = 0.8$ behält die Token 0, 1 und 2 (ihre kumulierte Masse erreicht $0.85$). Min-$p$ mit $p_{\min} = 0.5$ behält alle Token mit einer Wahrscheinlichkeit von mindestens $0.5 \times 0.45 = 0.225$, also wieder 0 und 1. Je 1000 Ziehungen nach jeder Regel bestätigen die erwarteten Trägermengen.

<!--en:f4ee7f40-->
Wie erwartet zieht jede Regel nur Token aus ihrer verbleibenden Trägermenge: 0 und 1 bei Top-$k$ und Min-$p$ sowie 0 bis 2 bei Top-$p$.

<!--en:f8bd92cc-->
### Abschneidegrenzen bei einem trainierten Modell

<!--en:4c4c2c5a-->
Betrachten wir die drei Regeln bei einem echten Sprachmodell. Dazu tragen wir die sortierten Wahrscheinlichkeiten unseres RNN für das nächste Token nach `"The Time Traveller"` auf doppelt logarithmischen Achsen auf. So lässt sich jede Regel geometrisch deuten: Top-$k$ zieht unabhängig von der Kurvenform eine feste senkrechte Linie bei Rang $k$. Top-$p$ zieht eine senkrechte Linie bei dem Rang, an dem die kumulierte Masse $p$ erreicht. Min-$p$ zieht eine *waagerechte* Linie beim $p_{\min}$-Fachen der höchsten Wahrscheinlichkeit. Ihre Höhe steigt und fällt mit der Sicherheit des Modells. Alles rechts von einer senkrechten Grenze oder unterhalb der waagerechten Grenze entfällt.

<!--en:c1bb947f-->
Auf doppelt logarithmischen Achsen fallen die sortierten Wahrscheinlichkeiten stetig von einem Wert zwischen $0.1$ und $0.25$ auf Rang 1 bis auf etwa $10^{-3}$ bei Rang 100. Über die verbleibenden Hunderte von Token fallen sie noch steiler ab. Dieser ausgedehnte Bereich geringer Wahrscheinlichkeit ist der Rand, den wir abschneiden. Um zu sehen, wie die Regeln auf die Sicherheit des Modells reagieren, vergleichen wir `"The Time Traveller"` mit dem kürzeren Präfix `"The Time"`. Darauf folgt in 11 von 13 Fällen in der Novelle `" Traveller"`, sodass die Verteilung ein ausgeprägtes Maximum hat.

<!--en:f5f10d10-->
Top-$k$ behält unabhängig vom Kontext stets 20 Token. Die beiden adaptiven Regeln schränken die Auswahl dagegen stark ein, sobald das Modell sicher ist: Nach `"The Time"` dominiert das wahrscheinlichste Token. Beide Regeln schneiden dann großzügig ab, bei Min-$p$ bleibt nur eine Handvoll Kandidaten. Ist das Modell wie nach `"The Time
Traveller"` unsicher, unterscheiden sich die Regeln: Top-$p$ muss weit in den Rand hineinreichen, um 90% der Masse zu erfassen. Es lässt deutlich über hundert Token zu, die einzeln meist weniger als 1% Wahrscheinlichkeit haben. Min-$p$ orientiert sich allein am Spitzenreiter und behält höchstens einige Dutzend Token, deren Wahrscheinlichkeit mindestens ein Zwanzigstel der höchsten beträgt.

<!--en:62eec77a-->
### Vergleich der Strategien an einem Präfix

<!--en:50395db5-->
Betrachten wir die Fortsetzungen, die die einzelnen Strategien tatsächlich erzeugen. Wir beginnen mit reiner Temperaturskalierung bei $\tau = 0.5$, $1.0$ und $2.0$.

<!--en:d6bd7bf6-->
Bei $\tau = 0.5$ zieht das Verfahren vorwiegend Token mit hoher Wahrscheinlichkeit. Bei $\tau = 1.0$ werden die Ausgaben merklich vielfältiger und enthalten gelegentlich verstümmelte Wörter. Bei $\tau = 2.0$ ist die Verteilung so flach, dass Token vom Rand die Ausgabe dominieren. Der Text ist dann kein durchgängiges Englisch mehr und enthält sogar Bytefolgen, die keine gültigen UTF-8-Zeichen bilden. Vergleichen wir nun unsere wichtigsten Dekodierstrategien in ihren Standardeinstellungen anhand desselben Prompts.

<!--en:18aaf663-->
Ein kleines einschichtiges RNN, das nur an einer Novelle trainiert wurde, kann mit keinem Decoder makellose Prosa erzeugen. Auch ist ein einzelnes Sample pro Regel wenig aussagekräftig. Dennoch ist der qualitative Unterschied unverkennbar: Gieriges Dekodieren ist deterministisch und gerät in einen Wiederholungszyklus. Jedes Sampling-Verfahren erzeugt dagegen bei jedem Aufruf neuen Text ohne Schleifen. In der Praxis bringt eine mäßige Temperatur in Verbindung mit Top-$p$ oder Min-$p$ als Abschneideregel Sprachfluss und Vielfalt gut ins Gleichgewicht. Diese Kombination ist der Standard für die offene Textgenerierung.

<!--en:0e60f4c1-->
## Bewertung und Rechenaufwand

<!--en:9b628c08-->
### Qualität der erzeugten Texte

<!--en:9abb864c-->
Wie können wir Dekodierstrategien systematisch vergleichen, statt nur einzelne Textpassagen zu begutachten? Die Perplexität beantwortet diese Frage nicht. Sie bewertet das *Modell* anhand menschlicher Texte, die beim Training unberücksichtigt blieben. Welche Dekodierregel wir verwenden, spielt dabei keine Rolle. Würden wir erzeugte Samples anhand ihrer Likelihood unter demselben Modell bewerten, stünde die Maximierung schon durch das Kriterium als Siegerin fest. Beim Sampling nehmen wir aber bewusst Abstriche bei der maximalen Likelihood in Kauf, um Vielfalt zu gewinnen.

<!--en:dd3e4cc7-->
Automatische Kennzahlen können dennoch bestimmte Schwächen aufdecken. Um Wiederholungen zu messen, verwenden wir *Distinct-3*: Wir teilen die Anzahl unterschiedlicher 3-Gramme durch die Gesamtzahl aller 3-Gramme in einer erzeugten Fortsetzung. Der Wert beträgt 1, wenn keine Folge aus drei Token zweimal vorkommt. Er nähert sich 0, wenn der Text in einen Zyklus gerät. Im Folgenden wählen wir zufällig 20 Präfixe aus je sechs Token aus den Validierungsdaten und erzeugen dazu Fortsetzungen mit je 100 Token. Dafür verwenden wir gieriges Dekodieren, Beam-Search ($w = 4$) und jedes der Sampling-Verfahren. Wir geben sowohl die mittlere Log-Wahrscheinlichkeit pro Token unter dem Modell als auch den Mittelwert von Distinct-3 an.

<!--en:853abca4-->
Die beiden Kennzahlen weisen deutlich in entgegengesetzte Richtungen. Beide Maximierungsstrategien erreichen eine merklich höhere Log-Wahrscheinlichkeit pro Token als jedes Sampling-Verfahren. Dabei liegt die Beam-Search meist knapp vor dem gierigen Dekodieren. Ihre Distinct-3-Werte fallen jedoch stark ab, weil sie die meisten ihrer 3-Gramme wiederholen. Alle vier Sampling-Verfahren erreichen Distinct-3-Werte nahe 1. Damit sehen wir im Kleinen genau den Zielkonflikt zwischen Likelihood und Vielfalt, den :citet:`Holtzman.Buys.Du.ea.2020` beschrieben haben.

<!--en:7dceee42-->
Innerhalb der Sampling-Familie erreichen alle vier Regeln fast die größtmögliche Vielfalt. Die Log-Wahrscheinlichkeit pro Token unterscheidet sie dagegen deutlich. Reines Ancestral Sampling ($\tau = 1$) ergibt die niedrigste mittlere Log-Wahrscheinlichkeit (etwa $-4$ Nat pro Token). Alle drei Verfahren mit abgeschnittener Verteilung erhöhen die Likelihood erheblich, ohne bei Distinct-3 einzubüßen. Wie die zuvor gezählten Kandidaten erwarten lassen, liegt Top-$p$ ($0.9$) dem reinen Sampling am nächsten, weil es die größte Trägermenge behält. Top-$k$ ($20$) und Min-$p$ ($0.05$) schneiden mehr Token ab und steigern die Log-Wahrscheinlichkeit pro Token um etwa ein ganzes Nat. Hier sehen wir direkt, dass das Abschneiden des Randes seinen Zweck erfüllt.

<!--en:b390c2f2-->
Automatische Kennzahlen wie Distinct-3 helfen, konkrete Schwächen zu erkennen. Kohärenz, sachliche Richtigkeit und Stil können sie jedoch nicht messen. Bei der offenen Textgenerierung urteilen letztlich Menschen, die die Texte lesen, oder entsprechend kalibrierte Ersatzverfahren. Bei eng umrissenen Aufgaben mit Referenztexten, etwa der Übersetzung, können wir die Ausgaben mit diesen Referenzen vergleichen. Kurz gesagt: Bei der Wahl des *Modells* hilft die Perplexität, bei der Wahl des *Decoders* helfen menschliche Präferenzen oder aufgabenspezifische Benchmarks.

<!--en:caeb89ac-->
### Rechenaufwand der Textgenerierung

<!--en:4b9d2b19-->
Wir haben `step_fn` ohne eigenen Zustand implementiert, damit die Funktion jedes autoregressive Modell einbinden kann. Daher läuft das RNN bei jedem neuen Token erneut über das *gesamte* Präfix. Eine Fortsetzung aus $T$ Token kostet so $O(T^2)$ Zellaktualisierungen. Autoregressive Generierung ist außerdem ihrem Wesen nach *sequenziell*. Beim Training mit Teacher-Forcing sind alle Positionen von Anfang an bekannt. Bei der Generierung können wir Token $t + 1$ erst bewerten, wenn Token $t$ gewählt ist. Diese Abhängigkeit ist unvermeidlich, der quadratische Aufwand für die Neuberechnung bei einem RNN dagegen völlig unnötig. Sein versteckter Zustand fasst das Präfix bereits zusammen. Geben wir ihn von Schritt zu Schritt weiter (wie `predict` in :numref:`sec_rnn-scratch`), genügt unabhängig von der Sequenzlänge genau eine Zellaktualisierung pro Token. Modelle ohne rekurrenten Zustand vermeiden dieselbe Mehrfachberechnung, indem sie die Ergebnisse für jedes frühere Token zwischenspeichern. Der Speicherbedarf wächst dabei mit der Länge des Präfixes (:numref:`chap_attention`).

<!--en:6b36e163-->
Konstanter Rechenaufwand und konstanter Speicherbedarf pro erzeugtem Token sind die entscheidenden Effizienzvorteile rekurrenter Sprachmodelle. Zugleich ist der Zustand fester Größe der Flaschenhals unseres Modells: Alles, was es sich vom Präfix merkt, muss in nur 128 Gleitkommazahlen passen (:numref:`subsec_rnn-constant-memory`). Erfasst das Modell in seinen Verteilungen für das nächste Token keine weitreichenden Strukturen, kann ihm keine Dekodierstrategie diese Feinheiten entlocken. Da unsere Decoder das Modell aber ausschließlich als Blackbox behandeln, lassen sich alle hier entwickelten Algorithmen unverändert auf weit leistungsfähigere Modelle übertragen.

<!--en:b8e3a8a0-->
Damit endet der Teil *Grundlagen* an einer entscheidenden Weggabelung für die Modellarchitektur. Unser RNN sagt Text Token für Token bei konstantem Aufwand vorher, nutzt aber effektiv nur die letzten zwei oder drei Token der Vorgeschichte (:numref:`sec_rnn-scratch`). Da die Gradienten schon nach wenigen Schritten verschwinden (:numref:`sec_bptt`), kann es kaum lernen, wesentlich weiter zurückzugreifen. Von hier aus führen zwei Wege weiter. Der erste behält einen Zustand fester Größe bei und gestaltet die Rekursion so um, dass Gradienten auch über lange Zeiträume erhalten bleiben (:numref:`chap_modern_rnn`). Der zweite verzichtet ganz auf die Kompression auf eine feste Größe. Er behält die Repräsentationen aller früheren Token und gewichtet sie direkt, wobei der Aufwand mit der Sequenzlänge wächst. Dieser *Attention*-Mechanismus (:numref:`chap_attention`) bildet das Herzstück moderner großer Sprachmodelle. Unabhängig von der Architektur lassen sich die Subwort-Tokenizer, die Bewertung in Bit pro Byte und die Dekodieralgorithmen dieses Kapitels direkt übertragen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:afbbd661-->
Bei der Dekodierung entstehen aus den Verteilungen eines Sprachmodells für das jeweils nächste Token konkrete Tokensequenzen. Maximierungsverfahren suchen nach Ausgaben mit hoher Wahrscheinlichkeit. Gieriges Dekodieren trifft in jedem Schritt eine lokale Entscheidung, ist kurzsichtig und gerät in Wiederholungsschleifen. Die Beam-Search verfolgt die besten $w$ Hypothesen mit dem $w$-Fachen des Aufwands. Können Kandidaten vorzeitig enden, verwendet sie eine Längennormalisierung. Ihre Stärke liegt bei eng umrissenen Aufgaben mit wenigen möglichen Ausgaben. Bei der offenen Textgenerierung erhält Sampling die natürliche Vielfalt. Temperaturskalierung (:eqref:`eq_temperature`) und das Abschneiden des Randes mit Top-$k$, Top-$p$ oder Min-$p$ verhindern, dass ein unwahrscheinliches Token vom Rand die rekursive Vorhersage entgleisen lässt. Die erzeugten Texte erreichen bei Distinct-3 die Vielfalt des reinen Samplings und sind zugleich wesentlich wahrscheinlicher. Die Perplexität bewertet das Modell unabhängig vom Decoder. Geben wir den versteckten Zustand eines rekurrenten Netzes weiter, sparen wir uns die Neuberechnung mit Aufwand $O(T^2)$ beim naiven zustandslosen Dekodieren.

<!--en:2294243a-->
## Übungen

<!--en:7ab29bc8-->
1. **Eigenschaften der Beam-Search.**
    1. Verwenden Sie die Beam-Breite 2, das Vokabular mit fünf Token aus :numref:`fig_beam-search` und drei Schritte. Geben Sie bedingte Wahrscheinlichkeiten an, bei denen die Beam-Search die wahrscheinlichste Sequenz aus drei Token verfehlt. Berechnen Sie die Wahrscheinlichkeiten beider Sequenzen. Was zeigt Ihr Beispiel darüber, wie die Beam-Search die wahrscheinlichste Sequenz übersehen kann?
    1. Welche Beam-Breite ist mindestens nötig, damit die Beam-Search über $T$ Schritte unabhängig vom Modell garantiert die wahrscheinlichste aller $|\mathcal{V}|^T$ Sequenzen findet?
    1. Es gelte $w \leq |\mathcal{V}|$, und es trete kein Sequenzendtoken auf. Zählen Sie die Aufrufe von `step_fn`, die `beam_search` über $T$ Schritte ausführt. Zählen Sie auch die Aufrufe bei vollständiger Suche, wenn diese jedes Präfix einmal auswertet. Berechnen Sie beide Anzahlen für $|\mathcal{V}| = 1{,}027$, $T = 50$ und $w = 4$.

<!--en:48af1f40-->
1. [code] **Längennormalisierung.** Führen Sie `beam_search` mit $w = 4$ für bis zu 40 Token nach `'The Time Traveller'` und `'It seemed to me that'` aus. Übergeben Sie die ID des Zeilenumbruchs, `data.tokenizer.encode('\n')[0]`, als `eos_id`. Variieren Sie $\beta$ (das Argument `alpha`) über die Werte $\{0, 0.75, 1.5\}$. Vergleichen Sie die Längen und die kumulierten Log-Wahrscheinlichkeiten der jeweils besten Kandidaten. Erklären Sie den Trend anhand von :eqref:`eq_beam-search-score`.

<!--en:253af518-->
1. **Abschneiden bei Temperaturskalierung.** `sample_next` wendet jede Abschneideregel auf die temperaturskalierte Verteilung $P_\tau$ aus :eqref:`eq_temperature` an. Formulieren Sie die Min-$p$-Bedingung $P_\tau(x \mid x_{<t}) \geq p_{\min} \max_{x'} P_\tau(x' \mid x_{<t})$ als Bedingung an die ursprünglichen Wahrscheinlichkeiten. Betrachten Sie die Verteilung 0.45, 0.25, 0.15, 0.10, 0.05 über fünf Token. Berechnen Sie, welche Mengen Top-$k$ mit $k = 2$, Top-$p$ mit $p = 0.85$ und Min-$p$ mit $p_{\min} = 0.5$ bei $\tau = 2$ beibehalten. Welche der drei Regeln behält bei jeder Temperatur dieselbe Menge?

<!--en:12cbddb1-->
1. [code] **Top-$p$ und Min-$p$ im Vergleich.** ● Variieren Sie bei $\tau = 1.5$ den Parameter $p$ für Top-$p$ und $p_{\min}$ für Min-$p$ über ein Werteraster. Wählen Sie fünf Präfixe und erzeugen Sie für jede Einstellung Fortsetzungen aus 50 Token. Bestimmen Sie über die erzeugten Token zwei Mittelwerte: die Anzahl der Token, die die Regel zur Auswahl behält, und die Log-Wahrscheinlichkeit des gezogenen Tokens unter dem ursprünglichen Modell ohne Temperaturskalierung. Tragen Sie den zweiten Mittelwert gegen den ersten auf, mit einer Kurve pro Regel. Welche Regel zieht bei gleicher mittlerer Anzahl beibehaltener Token wahrscheinlichere Token? Vergleichen Sie Ihr Ergebnis mit denen von :citet:`Nguyen.Baker.Neo.ea.2025`.

<!--en:059c475b-->
1. [code] **Konstanter Aufwand pro Token.** ● `step_fn` wertet für jedes erzeugte Token das Netz erneut über das gesamte Präfix aus. Daher kostet eine Fortsetzung aus $T$ Token $O(T^2)$ Zellaktualisierungen (:numref:`subsec_cost-of-generation`).
    1. Schreiben Sie einen zustandsbehafteten Ersatz für `step_fn`, der den versteckten Zustand von `model.rnn` speichert und nur das neueste Token verarbeitet. Jeder Aufruf soll die Zelle genau einmal aktualisieren. Prüfen Sie, ob `generate` mit beiden Versionen dieselbe gierige Fortsetzung erzeugt.
    1. Messen Sie für beide Versionen die Laufzeit von `generate` für 50, 200 und 800 Token. Schätzen Sie anhand der Laufzeitverhältnisse, wie die Laufzeit jeweils mit der Tokenzahl wächst. Erklären Sie Abweichungen vom quadratischen beziehungsweise linearen Wachstum, das die Anzahl der Zellaktualisierungen erwarten lässt.
    1. Warum lässt sich derselbe Ersatz nicht direkt auf `beam_search` übertragen? Welche Informationen müssten Sie dafür zusätzlich verwalten?

<!--en:53a0047d-->
[Diskussion](https://d2l.discourse.group/t/338)

<!--en:e51f0e77-->
Dekodierung und Textgenerierung<br>**Gierige Suche · Beam-Search und Längennormalisierung · Temperaturskalierung · Top-k-, Nucleus- (Top-p) und Min-p-Sampling · Distinct-n und Komplexität**

<!--en:382a8777-->
::: {.slide title="Das Dekodierproblem: 10¹⁵⁰ mögliche Fortsetzungen"}

<!--en:6a8b1bbc-->
[Motivation und kombinatorischer Suchraum]{.kicker}

<!--en:733a6174-->
- **Exakte globale Maximierung ist praktisch unmöglich:** Bei unserem BPE-Vokabular mit 1027 Token und einem Horizont von 50 Token gibt es $1027^{50} > 10^{150}$ Fortsetzungen. Kein Decoder kann den gesamten Suchbaum aufzählen.

<!--en:487ca1bd-->
- **Das Dilemma zwischen Maximierung und Sampling:** Die gierige Wahl des Maximums führt schnell in deterministische Wiederholungsschleifen. Sampling ohne Abschneiden liefert dagegen auch unwahrscheinliche Token vom Rand der Verteilung, die die rekursive Vorhersage entgleisen lassen.

<!--en:e8e02fea-->
::: {.slide title="Zwei Decoderfamilien: Maximierung und Sampling"}

<!--en:84ed5415-->
[Decoderwahl nach Aufgabe]{.kicker}

<!--en:425a8216-->
Die **Entropie der Zielaufgabe** entscheidet darüber, ob die Suche nach einer wahrscheinlichen Sequenz oder die zufällige Auswahl von Token geeigneter ist:

<!--en:f3c1f84e-->
- **Decoder mit Maximierung (gierige Suche und Beam-Search):** Sie nähern $\operatorname*{argmax}_{x_{1:T}} P(x_{1:T})$ an. Am besten eignen sie sich für **eng umrissene Aufgaben mit geringer Entropie** (Spracherkennung, maschinelle Übersetzung, OCR), bei denen die Eingabe nur wenige korrekte Ausgaben zulässt.

<!--en:c3b39c67-->
- **Decoder mit Sampling aus abgeschnittenen Verteilungen (Temperatur, Top-$k$, Top-$p$, Min-$p$):** Sie formen den unzuverlässigen Rand von $P(x_t \mid x_{<t})$ mit seinen kleinen Wahrscheinlichkeiten um oder schneiden ihn ab und ziehen dann zufällig. Das ist der Standard für die **offene Textgenerierung** (Dialog, kreatives Schreiben), bei der ein gut kalibriertes Modell die Masse auf viele gültige Fortsetzungen verteilt.

<!--en:06a3d771-->
::: {.slide title="Eine modellunabhängige Schnittstelle für die Logits des nächsten Tokens"}

<!--en:a4cfca2d-->
[Eine gemeinsame Schnittstelle für alle Decoder]{.kicker}

<!--en:fec58a6a-->
Jeder Dekodieralgorithmus in diesem Abschnitt greift auf unser trainiertes RNN-Sprachmodell (`RNNLM`, $d = 512, h = 128$, Validierungsperplexität $\approx 60$) ausschließlich über die Blackboxfunktion `step_fn(ids) -> logits` zu. Sie liefert einen eindimensionalen NumPy-Vektor der Länge $|\mathcal{V}| = 1{,}027$:

<!--en:b9eb8441-->
- **Unabhängig von Architektur und Framework:** `step_fn` nimmt eine Liste ganzzahliger Token-IDs entgegen und gibt die Logits für das nächste Token als NumPy-Array zurück. So funktioniert derselbe Decodercode für RNNs, Zustandsraummodelle und Transformer in jedem Framework.

<!--en:6d5809a0-->
::: {.slide title="Gieriges Dekodieren und seine Kurzsichtigkeit"}

<!--en:7e9b065b-->
[Maximierungsverfahren I: Gierige Suche]{.kicker}

<!--en:bf9249cd-->
- **Ein Modellaufruf pro Token:** Das Verfahren ist deterministisch und schnell. Es legt sich in jedem Schritt unwiderruflich auf das wahrscheinlichste Token fest.

<!--en:86283f26-->
- **Warum die lokale Wahl des Maximums kurzsichtig ist:** Die Wahl von $A$ in Schritt 1 ($0.50 > 0.40$) führt zu einem unsicheren Schritt 2 ($0.50 \times 0.30 = 0.15$). Dabei entfällt der Zweig $B$, dessen wahrscheinliche Fortsetzung $0.40 \times 0.90 = 0.36$ erreicht ($2.4\times$ so hoch).

<!--en:1840bcec-->
::: {.slide title="Degeneration neuronal erzeugter Texte: deterministische Grenzzyklen"}

<!--en:b898b6fc-->
[Warum gierige Vorhersagen in Schleifen geraten]{.kicker}

<!--en:6dff4cdc-->
Gieriges Dekodieren nach `'The Time Traveller'` und `'It seemed to me that'` über jeweils 50 Token zeigt die zweite, noch gravierendere Schwäche: **Wiederholungsschleifen**.

<!--en:0beb3d70-->
- **Deterministische Attraktoren im Zustandsraum:** In einem autoregressiven RNN mit gierigem Dekodieren sind $\mathbf{H}_t = g(\mathbf{H}_{t-1}, x_t)$ und $x_t = \operatorname*{argmax} \mathbf{O}_{t-1}$ deterministische Abbildungen. Gerät die Trajektorie in die Nähe eines versteckten Zustands, den sie wenige Token zuvor durchlaufen hat, kann sie für immer in einem periodischen Grenzzyklus gefangen bleiben.

<!--en:86dc4ffd-->
- **Auch bei großen Modellen:** Holtzman et al. (2020) beobachteten dieselbe **Degeneration neuronal erzeugter Texte** bei GPT-2 Large (774 Mio. Parameter), wenn sie offene Texte durch Maximierung der Likelihood dekodierten.

<!--en:c195cb3a-->
::: {.slide title="Beam-Search: die w besten Teilhypothesen beibehalten"}

<!--en:c03324e3-->
[Maximierungsverfahren II: Beam-Search]{.kicker}

<!--en:086e105d-->
Die **Beam-Search** mildert die Kurzsichtigkeit der gierigen Suche, indem sie in jedem Schritt die $w$ Präfixe mit dem höchsten Score beibehält. Mit **Beam-Breite** $w = 1$ ergibt sich die gierige Suche, für $w \to \infty$ nähert sich das Verfahren der vollständigen Suche:

<!--en:b8a8db5a-->
- **Wiederholt erweitern und auswählen (Breite $w = 2$, $|\mathcal{V}| = 5$):**
  1. **Schritt 1:** Die $w = 2$ wahrscheinlichsten ersten Token beibehalten ($A$ und $C$).
  2. **Schritt 2:** Jedes aktive Präfix um jedes der $|\mathcal{V}|$ Token erweitern ($w|\mathcal{V}| = 10$ Kandidaten), nach kumulierter Log-Wahrscheinlichkeit $\log P(x_1) + \log P(x_2 \mid x_1)$ bewerten und nur die besten $w = 2$ behalten ($AB$ und $CE$).
  3. **Schritt $T$:** Die Gesamtzahl der Bewertungen wächst linear gemäß $O(w|\mathcal{V}|T)$ ($5 + 10 + 10 = 25$ statt $|\mathcal{V}|^3 = 125$).

<!--en:552f8d43-->
::: {.slide title="Längennormalisierung bei vorzeitig beendeten Hypothesen"}

<!--en:67f853f9-->
[Vergleich unterschiedlich langer Hypothesen]{.kicker}

<!--en:157714a7-->
- **Warum die unnormierte Log-Likelihood ($\beta = 0$) kurze Sequenzen bevorzugt:** Jeder Summand $\log P \le 0$ ist nichtpositiv. Mit zusätzlichen Token sinkt die kumulierte Log-Wahrscheinlichkeit monoton, was ein vorzeitiges `<eos>` begünstigt.

<!--en:9943224d-->
- **Exponent $\beta \in [0, 1]$ zur Interpolation (Wu et al., 2016):** $\beta \approx 0.75$ vermittelt zwischen kumulierter Likelihood und Mittelwert pro Token ($\beta = 1$).

<!--en:2645036f-->
::: {.slide title="Implementierung der Beam-Search mit Längennormalisierung"}

<!--en:98121462-->
[Eigene Implementierung und Auswertung]{.kicker}

<!--en:ea32d07b-->
Pro aktivem Präfix können höchstens $w$ Erweiterungen in der Auswahl bleiben. Daher genügen jeweils die besten $w$ Folgetoken:

<!--en:6a78f7c7-->
::: {.slide title="Implementierung der Beam-Search mit Längennormalisierung (Fortsetzung)"}

<!--en:98121462-2-->
[Eigene Implementierung und Auswertung]{.kicker}

<!--en:68b42c71-->
- **Höhere Modell-Likelihood, weiterhin Wiederholungen:** Eine größere Beam-Breite ($w = 1 \to 4$) liefert Fortsetzungen mit höherer Log-Wahrscheinlichkeit und weniger negativen längennormalisierten Scores. Bei offener Prosa wiederholen sich dennoch ganze Satzteile.

<!--en:3fdbf6ef-->
::: {.slide title="Ancestral Sampling und der unzuverlässige Rand"}

<!--en:43fe5c69-->
[Motivation für Sampling-Verfahren]{.kicker}

<!--en:27761e91-->
- **Warum reines Sampling entgleist (Holtzman et al., 2020):** Die Softmax-Funktion weist allen $|\mathcal{V}| = 1{,}027$ Token eine positive Wahrscheinlichkeit zu. Nach `'The Time Traveller'` entfallen $90\%$ der Masse auf die 140–174 wahrscheinlichsten Token. Die übrigen $\sim 850$–890 unplausiblen Token am Rand teilen sich **in jedem Schritt bis zu $10\%$**.

<!--en:9d242715-->
- **Fehlerfortpflanzung über 50 Schritte:** Liegt die Wahrscheinlichkeit für ein Token vom Rand pro Schritt bei $10\%$, kommt nur ein Anteil von $(1 - 0.1)^{50} \approx 0.005$ aller Fortsetzungen über 50 Token ohne ein solches Token aus, das den weiteren Kontext entgleisen lässt.

<!--en:a50fdd45-->
::: {.slide title="Temperaturskalierung und Entropie der Softmax-Verteilung"}

<!--en:deed951b-->
[Sampling steuern I: Temperatur]{.kicker}

<!--en:4ddd5f76-->
Die **Temperaturskalierung** teilt den Logit-Vektor $\mathbf{o} \in \mathbb{R}^{|\mathcal{V}|}$ durch $\tau > 0$, bevor die Softmax-Funktion zum Einsatz kommt:

<!--en:09de5c76-->
- **Drei Bereiche:** Für $\tau \to 0^+$ ergibt sich im Grenzfall die deterministische gierige Wahl des Maximums. Bei $\tau = 1$ ergibt sich unverändertes Ancestral Sampling. Für $\tau > 1$ nähert sich die Verteilung der Gleichverteilung (bei $\tau = 2.0$ zerstören Bytes vom Rand die englischen Wörter und sogar die gültige UTF-8-Kodierung).

<!--en:bfc1116f-->
- **Warum die Temperatur allein nur grob steuert:** Ein niedrigeres $\tau$ (z. B. $\tau = 0.5$) unterdrückt den unzuverlässigen Rand nur auf Kosten der Vielfalt unter den 3–5 plausibelsten Token.

<!--en:01ad918e-->
::: {.slide title="Drei Regeln zum Abschneiden des Randes: Top-k, Top-p und Min-p"}

<!--en:d439d5db-->
[Sampling steuern II: den Rand abschneiden]{.kicker}

<!--en:e83c25f2-->
Beim **Sampling aus abgeschnittenen Verteilungen** bleiben die relativen Gewichte der wahrscheinlichsten Token erhalten. Die Token absteigend sortieren: $P(x_{(1)}) \ge P(x_{(2)}) \ge \cdots \ge P(x_{(|\mathcal{V}|)})$. Alle Wahrscheinlichkeiten jenseits eines Grenzrangs $K$ auf null setzen, die übrigen erneut normieren und daraus ziehen:

<!--en:c37bbcaf-->
| Abschneideregel | Verbleibende Trägermenge $\mathcal{S}$ | Beispielverteilung $(0.45, 0.25, 0.15, 0.10, 0.05)$ | Anpassung an die Modellentropie |

<!--en:ac77fc54-->
| **Top-$k$** (Fan et al., 2018) | Feste Anzahl von Rängen: $\{x_{(1)}, \ldots, x_{(k)}\}$ | $k = 2 \implies \{0, 1\}$ (Masse $0.70$) | **Keine (starr):** Behält $k$ Token, ob die Entropie nahe $0$ liegt oder hoch ist |

<!--en:28486a6e-->
| **Top-$p$ / Nucleus** (Holtzman et al., 2020) | Kleinste Menge führender Token mit $\sum_{i=1}^K P(x_{(i)}) \ge p$ | $p = 0.80 \implies \{0, 1, 2\}$ (kum. Masse $0.85 \ge 0.80$) | **Kumulierte Masse:** Schrumpft bei ausgeprägtem Maximum, wächst bei flacher Verteilung |

<!--en:3e64fabe-->
| **Min-$p$** (Nguyen et al., 2025) | Relative Untergrenze: $\{x : P(x) \ge p_{\min}\,P(x_{(1)})\}$ | $p_{\min} = 0.50 \implies P(x) \ge 0.225 \implies \{0, 1\}$ | **Relativ zum Spitzenreiter:** Schwelle skaliert direkt mit der höchsten Wahrscheinlichkeit $P(x_{(1)})$ |

<!--en:f65aa0b3-->
::: {.slide title="Kombinierbare Sampling-Regeln in fünfzehn Zeilen"}

<!--en:c8bf2621-->
[Eigene Implementierung]{.kicker}

<!--en:bc8537c5-->
Jede Regel bestimmt lediglich die Anzahl `keep` der führenden Token, die bei der erneuten Normierung übrig bleiben:

<!--en:76185249-->
::: {.slide title="Kombinierbare Sampling-Regeln in fünfzehn Zeilen (Fortsetzung)"}

<!--en:c8bf2621-2-->
[Eigene Implementierung]{.kicker}

<!--en:0d683c4f-->
::: {.slide title="Geometrische Deutung der Abschneidegrenzen bei einem trainierten Modell"}

<!--en:15f6d797-->
[Abschneiden auf doppelt logarithmischen Achsen]{.kicker}

<!--en:2d78347d-->
Die sortierten Wahrscheinlichkeiten des trainierten RNN für das nächste Token nach `'The Time Traveller'` doppelt logarithmisch auftragen. So wird jede Abschneideregel geometrisch sichtbar:

<!--en:3ce6ad75-->
- **Spitze und Rand der Verteilung:** Über die Ränge $1\text{--}100$ fallen die Wahrscheinlichkeiten zunächst langsam ($\approx 0.1 \to 10^{-3}$), dann steil in einen verrauschten Rand mit 900 Token ab.

<!--en:00c3c71b-->
- **Senkrechte und waagerechte Grenzen:**
  - **Top-$k$ ($k = 20$, magenta, gestrichelt):** Feste senkrechte Grenze bei Rang 20.
  - **Top-$p$ ($p = 0.9$, grün):** Senkrechte Grenze dort, wo die kumulierte Fläche $90\%$ erreicht.
  - **Min-$p$ ($p_{\min} = 0.05$, rot):** Waagerechte Untergrenze bei $5\%$ des Maximums auf Rang 1 ($0.05\,P(x_{(1)})$). Alles darunter entfällt.

<!--en:ccb91191-->
::: {.slide title="Wie adaptive Grenzen auf die Sicherheit des Modells reagieren"}

<!--en:a769b164-->
[Kontexte mit hoher und geringer Vorhersagesicherheit]{.kicker}

<!--en:03b551d9-->
Das Präfix `'The Time Traveller'`, dessen Fortsetzung unsicher ist, mit dem gut vorhersagbaren Präfix `'The Time'` vergleichen. Auf Letzteres folgt in 11 von 13 Fällen in der Novelle `' Traveller'`. So wird sichtbar, wie sich die drei Regeln bei unterschiedlicher Entropie verhalten:

<!--en:4d43d74a-->
- **Bei sicherer Vorhersage (`'The Time'`, hohes $P_{\max}$):** Top-$k = 20$ lässt unnötig 19 unwahrscheinliche Token zu. Top-$p$ und Min-$p$ beschränken sich dagegen auf wenige Kandidaten.

<!--en:ade848a3-->
- **Bei unsicherer Vorhersage (`'The Time Traveller'`, flache Spitze):** Um $90\%$ der Masse zu erfassen, **muss Top-$p = 0.9$ etwa 140–175 Token behalten und weit in den Rand hineinreichen**. **Min-$p = 0.05$ behält nur Token, deren Wahrscheinlichkeit höchstens um den Faktor $20\times$ unter dem Spitzenwert liegt** (etwa 10 bis 30 plausible Möglichkeiten).

<!--en:5bd961bb-->
::: {.slide title="Textgenerierung mit verschiedenen Dekodierstrategien im direkten Vergleich" layout="tall-output"}

<!--en:3661dca8-->
[Qualitativer Vergleich der rekursiven Vorhersagen]{.kicker}

<!--en:ad606adc-->
Jeweils 50 Token nach `'The Time Traveller'` durch gieriges Dekodieren, reines Sampling ($\tau = 1$) und die drei Verfahren mit abgeschnittenen Verteilungen erzeugen. Die qualitative Verbesserung ist sofort sichtbar:

<!--en:16aa8159-->
- **Aus der Schleife heraus, ohne wirre Token vom Rand:** Gieriges Dekodieren gerät in eine deterministische Schleife. Reines Sampling mit $\tau = 1$ fügt gelegentlich Subwörter unplausibel zusammen. `top-k=20`, `top-p=0.9` und `min-p=0.05` erzeugen jeweils vielfältige Fortsetzungen ohne Schleifen.

<!--en:4978ff99-->
::: {.slide title="Likelihood und Vielfalt messen (Distinct-3)"}

<!--en:dee4a510-->
[Systematischer Vergleich anhand von 20 Prompts]{.kicker}

<!--en:2be1a899-->
Als Maß für den Zielkonflikt zwischen **Modell-Likelihood** (mittlere Log-Wahrscheinlichkeit pro erzeugtem Token) und **lexikalischer Vielfalt** dient **$\textrm{distinct-}3$** für 20 zufällige Fortsetzungen aus je 100 Token. Die Kennzahl ist das Verhältnis der Anzahl verschiedener 3-Gramme zur Gesamtzahl der 3-Gramme. Bei $1.00$ gibt es keine Wiederholungen:

<!--en:ff1fb113-->
- **Einbruch der Vielfalt bei Maximierung:** Gierige Suche und Beam-Search ($w = 4$) erzielen die höchste Log-Wahrscheinlichkeit pro Token ($\approx -1.5$ bis $-2.2$ Nat, mit höherem Wert für Beam-Search). Ihre `distinct-3`-Werte fallen jedoch stark ab ($\approx 0.15\text{--}0.30$), weil über $70\%$ ihrer Ausgabe aus wiederholten 3-Grammen besteht.

<!--en:81167cad-->
- **Günstiger Kompromiss beim Sampling aus abgeschnittenen Verteilungen:** Reines Sampling ($\tau = 1$) erreicht `distinct-3` $\approx 0.99$ bei geringer Likelihood ($\approx -4.0$ Nat). `top-k=20` und `min-p=0.05` erzielen fast dieselbe Vielfalt (`distinct-3` $\approx 0.97\text{--}0.99$) und erhöhen die Log-Wahrscheinlichkeit pro Token um ein ganzes Nat (auf $\approx -2.9$ Nat).

<!--en:632b4a2f-->
::: {.slide title="Warum die Perplexität das Modell bewertet, unabhängig vom Decoder"}

<!--en:943cfd09-->
[Modellgüte und Decoderwahl]{.kicker}

<!--en:9f904d20-->
Ein häufiger Fehler bei der Sprachmodellierung besteht darin, Dekodieralgorithmen anhand der Perplexität oder der Modell-Log-Likelihood erzeugter Texte zu vergleichen:

<!--en:87baffb2-->
- **Perplexität bewertet das Modell anhand menschlicher Texte:** $\textrm{PPL} = \exp(-\frac{1}{T}\sum_t \log P(x_t \mid x_{<t}))$ misst, wie viel Wahrscheinlichkeitsmasse die Modellverteilungen **zurückgehaltenen Referenzpassagen** zuweisen, bevor überhaupt ein Decoder zum Einsatz kommt.

<!--en:59ed3865-->
- **Warum die Modell-Likelihood erzeugter Texte zu einem Zirkelschluss führt:** Bei Bewertung der erzeugten Ausgaben anhand von $\frac{1}{T}\sum_t \log P(\hat{x}_t \mid \hat{x}_{<t})$ gewinnt immer die gierige Suche oder die Beam-Search, selbst wenn der Text in einer Schleife stecken bleibt (`"and the time traveller and the time traveller..."`).

<!--en:72463ab3-->
- **Geeignete Bewertung von Decodern:** Einfache Textmerkmale (`distinct-n`, Wiederholungsrate) decken Degeneration auf. Für eng umrissene Aufgaben eignen sich Referenzmetriken (BLEU, COMET, WER), für offene Textgenerierung verblindete menschliche Präferenzurteile oder kalibrierte Bewertungsverfahren.

<!--en:28fc4d68-->
::: {.slide title="Inferenzaufwand: O(T²) ohne Zustand, O(T) mit Zwischenspeicherung"}

<!--en:372b3dfb-->
[Rechen- und Speicheraufwand der autoregressiven Generierung]{.kicker}

<!--en:c61c7b41-->
- **Erneute Präfixauswertung ohne Zustand (`step_fn(ids)`):** Läuft das Netz über das Präfix $x_{1:t}$ in jedem Schritt $t$ erneut, fallen insgesamt $\sum_{t=1}^T t = O(T^2)$ Zellaktualisierungen an.

<!--en:88289b0e-->
- **Weitergabe des rekurrenten Zustands (`predict`):** Bei Weitergabe von $\mathbf{H}_{t-1} \in \mathbb{R}^h$ genügt **1 Zellaktualisierung pro Token** (insgesamt $O(T)$). Der Zustand benötigt bei jeder Sequenzlänge genau $512$ Byte.

<!--en:1e1e25d2-->
- **Ausblick (§{num:chap_attention}–§{num:chap_modern_rnn}):** Gates oder Zustandsraummodelle erweitern $\mathbf{H}_t$ so, dass Gradienten bei $O(1)$ Speicherbedarf erhalten bleiben (§{chap_modern_rnn}). Alternativ behält Self-Attention alle bisherigen Schlüssel- und Wertvektoren der Token bei einem Speicherbedarf von $O(T)$ (§{chap_attention}–§{num:chap_transformers}).

<!--en:c7f49bfa-->
::: {.slide title="Verständnisfragen: Beam-Search, Abschneidegrenzen und Komplexität"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:0cb208d9-->
Fragen zu Dekodieralgorithmen und ihrem Rechenaufwand:

<!--en:a02a797a-->
1. **Funktionsauswertungen bei der Beam-Search:** Das Vokabular habe die Größe $|\mathcal{V}| = 1{,}027$, die erzeugte Sequenz die Länge $T = 50$ und die Beam-Breite sei $w = 4$ (ohne vorzeitiges `<eos>`). Wie oft ruft `beam_search` in den 50 Schritten `step_fn` auf? Wie verhält sich das zur vollständigen Suche über alle $|\mathcal{V}|^T$ Sequenzen?

<!--en:b0675967-->
2. **Top-$p$ und Min-$p$ bei ausgeprägtem Maximum:** Im Schritt $t$ habe das wahrscheinlichste Token die Wahrscheinlichkeit $P(x_{(1)}) = 0.95$, das zweitwahrscheinlichste $P(x_{(2)}) = 0.04$, und die übrigen 1025 Token teilen sich $0.01$. Welche Token bleiben genau bei (a) Top-$p$ mit $p = 0.90$ und (b) Min-$p$ mit $p_{\min} = 0.05$?

<!--en:bdbca0f0-->
3. **Temperaturunabhängige Abschneidegrenzen:** `sample_next` wendet Top-$k$, Top-$p$ und Min-$p$ *nach* der Temperaturskalierung $P_\tau(x) \propto P(x)^{1/\tau}$ an. Welche der drei Regeln behält bei jeder Temperatur $\tau > 0$ **genau dieselbe Trägermenge von Token**?

<!--en:ff395747-->
::: {.slide title="Verständnisfragen: Beam-Search, Abschneidegrenzen und Komplexität (Lösungen)"}

<!--en:929b7671-2-->
[Wissensprüfung]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:3491b2b0-->
1. **`beam_search` benötigt $1 + 49 \times 4 = 197$ Aufrufe von `step_fn`** (1 Aufruf für den ersten Prompt in Schritt 1, danach $w = 4$ Aufrufe pro Schritt in den Schritten $2, \ldots, 50$, also insgesamt $1027 + 49 \times 4 \times 1027 = 202{,}319$ bewertete Erweiterungen). Bei der vollständigen Suche sind es dagegen $> 10^{150}$ Sequenzen.

<!--en:3ed3a13a-->
2. **(a) Top-$p$ ($p = 0.90$) behält nur $1$ Token ($x_{(1)}$)**, denn $0.95 \ge 0.90$ überschreitet bereits die Schwelle. Die Regel verhält sich hier rein gierig und verwirft das zweitwahrscheinlichste Token mit $4\%$. **(b) Min-$p$ ($p_{\min} = 0.05$) behält $1$ Token ($x_{(1)}$)**, denn für die Schwelle gilt $0.05 \times 0.95 = 0.0475 > 0.04$ (auch $x_{(2)}$ bleibt erhalten, sobald $P(x_{(2)}) \ge 0.0475$ gilt).

<!--en:d384d9c4-->
3. **Top-$k$ ist invariant gegenüber $\tau > 0$:** Da $p \mapsto p^{1/\tau}$ für alle $\tau > 0$ streng monoton wächst, bleibt die Rangfolge aller Token erhalten. Die besten $k$ Token sind daher immer dieselben. Bei Top-$p$ und Min-$p$ ändert sich dagegen die Größe der Trägermenge, wenn $\tau$ die Verteilung abflacht oder stärker konzentriert.

<!--en:6f976ef6-->
::: {.slide title="Zusammenfassung: Dekodierstrategien im Vergleich"}

<!--en:408fcbdb-->
[Das Wichtigste aus diesem Abschnitt]{.kicker}

<!--en:a28594fa-->
| Strategie | Auswahl- oder Abschneideregel | Wichtigster Hyperparameter | Anpassung an die Verteilungsform | Modellaufrufe pro Schritt | Einsatzgebiet und Schwäche |

<!--en:88b79d39-->
| **Gieriges Dekodieren** | $x_t = \operatorname*{argmax}_{x} P(x \mid x_{<t})$ | Keiner ($\tau = 0$) | Deterministische Wahl des Maximums | $1$ | Schnelle Baseline, aber kurzsichtig und anfällig für deterministische Wiederholungsschleifen |

<!--en:92908a68-->
| **Beam-Search** | Beste $w$ Präfixe nach $\frac{1}{T^\beta}\sum \log P$ behalten | Breite $w$, Längenexponent $\beta \approx 0.75$ | Verfolgt $w$ Pfade parallel | $w$ | Standard bei eng umrissenen Aufgaben (MT, ASR), weiterhin Schleifen bei offener Textgenerierung |

<!--en:891be1ca-->
| **Temperatur-Sampling** | Aus $P_\tau(x) \propto \exp(o_x / \tau)$ ziehen | Temperatur $\tau > 0$ | Konzentriert ($\tau < 1$) oder verflacht ($\tau > 1$) die Verteilung einheitlich | $1$ | Ändert die Entropie global. Beim Abkühlen geht auch Vielfalt unter den wahrscheinlichen Token verloren |

<!--en:d211dd93-->
::: {.slide title="Zusammenfassung: Dekodierstrategien im Vergleich (Fortsetzung)"}

<!--en:408fcbdb-2-->
[Das Wichtigste aus diesem Abschnitt]{.kicker}

<!--en:a28594fa-2-->
| Strategie | Auswahl- oder Abschneideregel | Wichtigster Hyperparameter | Anpassung an die Verteilungsform | Modellaufrufe pro Schritt | Einsatzgebiet und Schwäche |

<!--en:de4e8616-->
| **Top-$k$-Sampling** | Aus den ersten $k$ Rängen von $P_\tau$ ziehen | Anzahl $k \in \{1, \ldots, \lvert\mathcal{V}\rvert\}$ | **Feste Größe $k$**, unabhängig von der Sicherheit des Modells | $1$ | Schneidet nach Rang $k$ ab. Bei ausgeprägtem Maximum bleiben $k-1$ schlechte Kandidaten |

<!--en:f6e50e7f-->
| **Top-$p$ (Nucleus)** | Kleinste Menge führender Token mit $\sum_{i=1}^K P_\tau(x_{(i)}) \ge p$ | Kumulierte Masse $p \in (0, 1)$ (z. B. $0.9$) | **An die Masse angepasst** (klein bei ausgeprägtem Maximum, größer bei flacher Verteilung) | $1$ | Passt sich der Entropie an. Bei hohem $\tau$ kann die Auswahl Hunderte von Token weit in den flachen Rand reichen |

<!--en:34b4d761-->
| **Min-$p$-Sampling** | $\{x : P_\tau(x) \ge p_{\min} P_\tau(x_{(1)})\}$ beibehalten | Relative Untergrenze $p_{\min}$ (z. B. $0.05$) | **An die Modellsicherheit angepasst** (Schwelle skaliert direkt mit dem Spitzenwert $P_{\max}$) | $1$ | Robust über verschiedene Temperaturen. Verhindert bei flacher Verteilung die Aufnahme des verrauschten Randes |
