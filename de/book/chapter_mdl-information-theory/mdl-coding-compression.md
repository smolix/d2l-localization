<!--en:955d8db8-->
# Codes, Kompression und Perplexität

<!--en:b529caba-->
In :numref:`sec_mdl-information_theory` haben wir Entropie und Kreuzentropie als mittlere Codewortlängen interpretiert. Demnach müsste ein Modell mit einem Verlust von $x$ Bit pro Symbol einen Text auf etwa $x$ Bit pro Symbol komprimieren können. Überprüft haben wir das bisher nur an einer besonders einfachen Quelle, deren Überraschungswerte (engl. surprisal) zufällig ganzzahlige Bitzahlen waren. Für andere Verteilungen haben wir keinen Code konstruiert. Bis wir einen solchen Code tatsächlich angeben, bleibt der Zusammenhang zwischen logarithmischem Verlust (engl. log-loss) und Dateigröße eine Analogie. Um diese Lücke zu schließen, gehen wir zwei Fragen nach. Können wir einen Code konstruieren, dessen Länge dem logarithmischen Verlust eines Modells entspricht? Wie nahe kommen gängige Kompressionsverfahren dieser Grenze?

<!--en:ba75c91c-->
Gängige Kompressionsprogramme geben uns schon einen Hinweis. Betrachten wir *The Time Machine*, den Roman aus :numref:`sec_text-sequence`. Nach der Bereinigung umfasst er 173 428 Zeichen aus einem Alphabet mit 27 Symbolen (die 26 Kleinbuchstaben und das Leerzeichen). Das universell einsetzbare Kompressionsprogramm `zlib` ersetzt wiederkehrende Zeichenfolgen durch Rückverweise und kodiert dann die Symbole einzeln. Es speichert den Text mit $2.84$ Bit pro Zeichen. Ein kleines Zählmodell sagt jedes Zeichen aus den drei vorhergehenden Zeichen vorher und aktualisiert beim Lesen seine Zähler. Es komprimiert denselben Text auf $2.55$ Bit pro Zeichen. Warum führt die bessere Vorhersage zu einer kürzeren Datei? Das untersuchen wir in diesem Abschnitt. Sobald ein Code den logarithmischen Verlust in eine Dateigröße übersetzt, gibt ein Verlust wie die $1.2$ Nat pro Zeichen aus der Kapiteleinleitung einen konkreten Speicherbedarf an. Zwei Modelle lassen sich dann direkt anhand der Größe der von ihnen erzeugten Dateien vergleichen.

<!--en:a6359cbd-->
Wir gehen in vier Schritten vor. Zunächst fragen wir, wie kurz Codewörter sein dürfen, wenn der Empfänger einen Bitstrom ohne Trennzeichen zwischen den Symbolen zerlegen muss. Dann bestimmen wir, wie viele zusätzliche Bits nötig sind, weil jedes Codewort eine ganzzahlige Länge haben muss. Wir konstruieren einen Code für den gesamten Datenstrom, bei dem diese Kosten nicht bei jedem Symbol anfallen. Im dritten Schritt betrachten wir echten Text, in dem jedes Zeichen von den vorhergehenden abhängt. Wie kurz kann die Datei überhaupt werden, und wie gehen die Kosten für das Lernen des Modells in die Gesamtlänge ein? Schließlich verknüpfen wir diese Codelängen mit der Perplexität, dem Maß, mit dem wir Sprachmodelle auf zurückgehaltenem Text bewerten. Da Codewörter Bitfolgen sind, rechnen wir in diesem Abschnitt in Bit statt in Nat. Dabei entsprechen $x$ Bit genau $x \ln 2$ Nat.

<!--en:6ef38913-->
Zusätzlich zu den üblichen Importen des Buchs benötigen wir `re` für die Vorverarbeitung des Textes und `zlib` als Referenz für die Kompression.

<!--en:d5e24735-->
## Symbolcodes

<!--en:64cdf60f-->
Beginnen wir mit dem einfachsten Kommunikationsproblem. Ein Sender beobachtet einen Strom von Symbolen, die unabhängig voneinander aus einer Verteilung $P$ mit $k$ möglichen Ergebnissen stammen. Er muss sie als eine einzige Bitfolge ohne Trennzeichen zwischen den Symbolen übertragen. Der Empfänger muss daraus die ursprüngliche Sequenz eindeutig rekonstruieren können.

<!--en:bf141446-->
### Präfixcodes und die Kraft-Ungleichung

<!--en:4691b3f7-->
Ein Symbolcode ordnet jedem Ergebnis $i$ eine Bitfolge zu, sein *Codewort* der Länge $\ell_i$. Woran erkennt der Empfänger ohne Trennzeichen, wo ein Codewort endet und das nächste beginnt? Am einfachsten lässt sich das garantieren, wenn kein Codewort Präfix eines anderen ist. Ein solcher Code heißt *präfixfrei* oder kurz *Präfixcode*. Telefonnummern folgen derselben Regel. Sobald die gewählten Ziffern eine gültige Nummer ergeben, beginnt keine längere Nummer mit dieser Ziffernfolge. Die Vermittlungsstelle erkennt also sofort, dass die Nummer vollständig ist. Für vier Symbole ist der Code $0, 10, 110, 111$ präfixfrei. Der eingehende Bitstrom `110010111` lässt sich nur auf eine Weise zerlegen, nämlich als `110|0|10|111`.

<!--en:5f326970-->
![Der präfixfreie Code $0, 10, 110, 111$ als Binärbaum. Jedes Codewort entspricht dem Pfad von der Wurzel zu einem Blatt (linke Kante $0$, rechte Kante $1$). Kein solcher Pfad führt durch das Blatt eines anderen Codeworts. Ein Blatt in Tiefe $\ell$ beansprucht den Anteil $2^{-\ell}$ des Baums. Hier schöpfen die vier Blätter das gesamte Budget aus: $2^{-1} + 2^{-2} + 2^{-3} + 2^{-3} = 1$. Die Symbole $a, b, c, d$ und ihre Wahrscheinlichkeiten gehören zur dyadischen Quelle aus dem vorigen Abschnitt.](../img/mdl-it-kraft-tree.svg)

<!--en:e86777f8-->
Stellen wir einen Präfixcode als Binärbaum dar (:numref:`fig_mdl-kraft-tree`), erkennen wir sofort, welche Nebenbedingung für die Codewortlängen gilt. Ein Codewort der Länge $\ell$ endet an einem Blatt in Tiefe $\ell$ und schließt alle darunterliegenden Nachkommen aus. Damit beansprucht es den Anteil $2^{-\ell}$ des vollständigen Baums. Da sich die von verschiedenen Codewörtern belegten Teile nicht überschneiden dürfen, müssen alle Anteile zusammen in einen Baum passen. Das Gesamtbudget beträgt also $1$, und kurze Codewörter verbrauchen besonders viel davon. Die Längen $(1, 2, 3, 3)$ in der Abbildung beanspruchen $\tfrac12 + \tfrac14 + \tfrac18 + \tfrac18 = 1$ und schöpfen den Baum vollständig aus. Vier Codewörter der Länge $1$ bräuchten dagegen ein Budget von $2$. Das ist unmöglich, denn es gibt nur zwei Bitfolgen der Länge $1$. Die Längen $(1, 2, 4)$ verbrauchen $\tfrac12 + \tfrac14 + \tfrac1{16} = 0.8125$ und lassen noch Platz, etwa bei der Wahl $0, 10, 1100$.

<!--en:7f131411-->
Derselbe Baum beschreibt eine Strategie für das Spiel mit Ja-Nein-Fragen aus :numref:`sec_mdl-information_theory`. Jeder innere Knoten stellt eine binäre Frage. Jedes Codewort hält die Folge der Antworten fest, die zu einem Symbol führt. Steht $0$ für Ja, fragt der Code $0, 10, 110, 111$ zunächst „Ist es das erste Symbol?“, dann „Ist es das zweite?“ und schließlich „Ist es das dritte?“. Im vorigen Abschnitt stand $1$ für Ja, sodass sich das Spiegelbild ergab, $1, 01, 001, 000$.

<!--en:c86975af-->
**Proposition (Kraft-Ungleichung).** *Ein präfixfreier Binärcode mit Codewortlängen $\ell_1, \ldots, \ell_k$ existiert genau dann, wenn gilt:*

<!--en:b2696b2c-->
**Beweis.** Ordnen wir jedem binären Codewort $c$ der Länge $\ell$ das halboffene Intervall $[0.c,\, 0.c + 2^{-\ell})$ zu. Es enthält alle reellen Zahlen in $[0, 1)$, deren Binärdarstellung mit $c$ beginnt. So entspricht das Codewort $10$ dem Intervall $[0.10_2, 0.11_2) = [\tfrac{1}{2},
\tfrac{3}{4})$ der Breite $2^{-2}$. In :numref:`fig_mdl-kraft-tree` steht dieses Intervall für das Blatt zu $b$ samt dem gesamten darunter entfernten Teilbaum. Zwei solche *dyadischen* Intervalle sind entweder ineinander enthalten oder disjunkt. Ist $c$ ein Präfix von $c'$, beginnt jede Zahl mit den Anfangsziffern $c'$ auch mit $c$. Dann liegt ein Intervall im anderen. Andernfalls unterscheiden sich $c$ und $c'$ an einer Stelle, die noch im kürzeren Wort liegt. Keine Zahl kann mit beiden beginnen, also sind die Intervalle disjunkt. Ein präfixfreier Code entspricht somit disjunkten Intervallen in $[0, 1)$. Ihre Gesamtbreite $\sum_i 2^{-\ell_i}$ beträgt höchstens $1$.

<!--en:4f90e716-->
Nehmen wir umgekehrt an, die Längen erfüllen :eqref:`eq_mdl-kraft`. Wir sortieren sie aufsteigend, $\ell_1 \leq \cdots \leq \ell_k$, und legen die Intervalle der Breite $2^{-\ell_i}$ von $0$ aus lückenlos nebeneinander. Das $i$-te Intervall beginnt bei $s_i = \sum_{m < i} 2^{-\ell_m}$ und endet bei $s_i + 2^{-\ell_i} \leq 1$. Für jede vorhergehende Länge gilt $\ell_m \leq \ell_i$. Somit ist jeder Summand $2^{-\ell_m}$ in $s_i$ ein ganzzahliges Vielfaches von $2^{-\ell_i}$, also ist auch $s_i$ ein ganzzahliges Vielfaches von $2^{-\ell_i}$. Das $i$-te Intervall ist daher das dyadische Intervall zu dem Codewort, das aus den ersten $\ell_i$ Binärziffern von $s_i$ besteht. Da die Intervalle disjunkt sind, ist nach dem ersten Teil des Beweises kein Codewort Präfix eines anderen. $\blacksquare$

<!--en:d699db34-->
Das Kraft-Budget ergibt sich keineswegs nur aus der Forderung nach Präfixfreiheit. Ein Code heißt *eindeutig dekodierbar* (engl. uniquely decodable), wenn sich jede Verkettung von Codewörtern nur auf eine Weise zerlegen lässt. Dabei dürfen einzelne Codewörter Präfixe anderer sein. McMillan bewies, dass auch jeder eindeutig dekodierbare Code die Ungleichung :eqref:`eq_mdl-kraft` erfüllen muss :cite:`McMillan.1956` (Aufgabe :exref:`ex_mdl-coding-compression-mcmillan`). Alle mit einem eindeutig dekodierbaren Code erreichbaren Codewortlängen lassen sich also bereits mit einem Präfixcode erzielen. Wir verlieren nichts, wenn wir uns auf Präfixcodes beschränken.

<!--en:d56e45b1-->
### Shannon-Längen und die Entropieschranke

<!--en:d4dd8c9a-->
Nun kennen wir das Budget und können fragen, welche Codewortlängen die erwartete Länge $\mathbb{E}[\ell] = \sum_i p_i \ell_i$ minimieren. Häufige Symbole sollten offensichtlich kürzere Codewörter erhalten. Doch schon ein Codewort der Länge $1$ verbraucht die Hälfte des gesamten Kraft-Budgets.

<!--en:dd4cb698-->
Um das Optimum zu finden, lassen wir vorübergehend die Ganzzahligkeitsbedingung fallen und behandeln die Längen $\ell_i$ als reelle Zahlen. Budget übrig zu lassen, bringt keinen Vorteil, denn jede Verkürzung eines Codeworts verringert $\mathbb{E}[\ell]$ strikt. Im Optimum muss also $\sum_i 2^{-\ell_i} = 1$ gelten. Nach dem Lagrange-Verfahren zur Optimierung unter Gleichungsnebenbedingungen (:numref:`subsec_mdl-lagrange`) muss der Gradient der Zielfunktion im Minimum proportional zum Gradienten der Nebenbedingung sein. Die Ableitung von $\mathbb{E}[\ell]$ nach $\ell_i$ ist $p_i$, die Ableitung von $\sum_i 2^{-\ell_i}$ entsprechend $-2^{-\ell_i}\ln 2$. Damit gilt $2^{-\ell_i} = c\, p_i$ mit derselben Konstanten $c$ für alle $i$. Summieren wir beide Seiten über $i$, folgt $c = 1$. Damit ergeben sich die optimalen reellen Längen und die zugehörige erwartete Länge:

<!--en:459e951b-->
Die ideale Länge für jedes Symbol ist sein *Überraschungswert* $\log_2(1/p_i)$, also die in Bit gemessene Selbstinformation aus :numref:`sec_mdl-information_theory`. Die erwartete Länge ist dann die binäre Entropie $H_2(P)$. Tatsächliche Codewörter müssen allerdings ganzzahlige Längen haben. Bei der dyadischen Quelle $P = (\tfrac12, \tfrac14, \tfrac18, \tfrac18)$ sind die Überraschungswerte $1, 2, 3, 3$ bereits ganzzahlig. Der Code aus :numref:`fig_mdl-kraft-tree` erreicht daher ohne Mehrbedarf $\mathbb{E}[\ell] = 1.75$ Bit $= H_2(P)$. Für $P = (0.6, 0.3, 0.1)$, die erste Verteilung aus dem Beispielpaar dieses Kapitels, betragen die Überraschungswerte $0.74$, $1.74$ und $3.32$ Bit. Runden wir jeden Wert auf die nächste ganze Zahl auf, erhalten wir die *Shannon-Längen* $1, 2, 4$. Diese Längen passen, wie wir gesehen haben, in den Binärbaum und lassen noch Platz. Ihre Kraft-Summe beträgt $0.8125$, ihre erwartete Länge $\mathbb{E}[\ell] = 0.6 + 0.6 + 0.4 = 1.6$ Bit, verglichen mit $H_2(P) = 1.295$.

<!--en:05aed058-->
Selbst bei sorgfältiger Wahl der ganzzahligen Längen kann kein Präfixcode die Entropie im Mittel unterschreiten. Die Shannon-Längen überschreiten sie im Mittel um höchstens ein Bit. Beide Schranken folgen aus derselben Identität aus :numref:`sec_mdl-information_theory`, der Zerlegung der Kreuzentropie. Wir geben sie hier in Bit an (der tiefergestellte Index $2$ und der hochgestellte Index $(2)$ bezeichnen Logarithmen zur Basis $2$). Der erwartete logarithmische Verlust eines Modells $Q$ auf Daten aus $P$ zerfällt in die Entropie von $P$ und die Kullback-Leibler-Divergenz von $P$ zu $Q$:

<!--en:ede2f048-->
Dass die KL-Divergenz nichtnegativ ist, besagt die Gibbs-Ungleichung. Kein Modell kann einen kleineren erwarteten logarithmischen Verlust erreichen als die wahre Entropie der Quelle. Gleichheit gilt nur für $Q = P$. Für unser Beispielpaar $P = (0.6, 0.3,
0.1)$ und $Q = (0.2, 0.5, 0.3)$ ergibt sich für die drei Größen $1.867 = 1.295 + 0.571$ Bit (bis auf Rundung in der letzten Dezimalstelle).

<!--en:e8739e72-->
**Proposition (Entropie als untere Schranke der Codelänge).** *Sei $P$ eine Verteilung mit $p_i > 0$ für alle $i$ (Symbole mit Wahrscheinlichkeit null brauchen kein Codewort). Für jeden präfixfreien Binärcode gilt $\mathbb{E}[\ell] \geq H_2(P)$, mit Gleichheit genau dann, wenn $p_i = 2^{-\ell_i}$ für jedes $i$ gilt. Umgekehrt ist der* Shannon-Code *mit Längen $\ell_i = \lceil \log_2 (1/p_i) \rceil$ präfixfrei und erfüllt $\mathbb{E}[\ell] < H_2(P) + 1$.*

<!--en:80da588d-->
**Beweis.** Für die untere Schranke vergleichen wir den Code mit der Wahrscheinlichkeitsverteilung, die er implizit festlegt. Sei $Z = \sum_i 2^{-\ell_i}$. Nach der Kraft-Ungleichung gilt $Z \leq 1$. Wir definieren die normierte Verteilung $q_i = 2^{-\ell_i}/Z$. Auflösen nach $\ell_i$ ergibt $\ell_i = -\log_2 q_i - \log_2 Z$. Bilden wir den Erwartungswert unter $P$, erhalten wir

<!--en:d9cab754-->
Beide Zusatzterme sind nichtnegativ. Nach der Gibbs-Ungleichung gilt $D_{\textrm{KL}}^{(2)}(P \| Q) \geq 0$, und es gilt $-\log_2 Z \geq 0$ wegen $Z \leq 1$. Gleichheit setzt voraus, dass beide Terme verschwinden, also $Q = P$ und $Z = 1$ gelten. Gleichbedeutend damit ist $p_i = 2^{-\ell_i}$ für alle $i$. Für die obere Schranke beachten wir, dass die Shannon-Längen $2^{-\ell_i} \leq p_i$ erfüllen. Damit gilt $\sum_i 2^{-\ell_i} \leq \sum_i p_i = 1$. Sie erfüllen also die Kraft-Ungleichung und definieren einen gültigen Präfixcode. Da beim Aufrunden stets weniger als ein Bit hinzukommt, gilt $\ell_i < \log_2(1/p_i) + 1$. Mitteln unter $P$ ergibt $\mathbb{E}[\ell] < H_2(P) + 1$. $\blacksquare$

<!--en:48a72520-->
Die Identität :eqref:`eq_mdl-code-length-decomp` sagt mehr aus als die daraus abgeleitete Schranke. Ein Präfixcode kann nur auf zwei Arten Bits verschwenden. Entweder impliziert er eine Verteilung $Q$, die von $P$ abweicht, oder er lässt einen Teil des Kraft-Budgets ungenutzt ($Z < 1$). Für unsere Beispielverteilung $P$ verbraucht der Shannon-Code $1.6 - 1.295 = 0.305$ Bit pro Symbol zu viel. Nur $0.005$ Bit davon gehen auf die abweichende Verteilung zurück, die übrigen $0.300$ Bit auf ungenutztes Budget.

<!--en:0c4d7300-->
Zusammen grenzen die beiden Teile der Proposition die optimale erwartete Länge eines Symbolcodes auf den Bereich zwischen $H_2(P)$ und $H_2(P) + 1$ Bit ein. Das ist genau die in :numref:`sec_mdl-information_theory` angekündigte Schranke für die erwartete Anzahl von Ja-Nein-Fragen. Die Entropie als untere Grenze erreichen wir nur, wenn jede Wahrscheinlichkeit eine exakte Potenz von $\tfrac12$ ist.

<!--en:8ff84aed-->
Wie viel von diesem einen zusätzlichen Bit lässt sich mit einem geschickter gewählten Symbolcode einsparen? Der Huffman-Algorithmus :cite:`Huffman.1952` konstruiert einen optimalen Präfixcode, indem er wiederholt die beiden unwahrscheinlichsten Symbole zu einem Symbol zusammenfasst (Aufgabe :exref:`ex_mdl-coding-compression-huffman`). Nach diesem Verfahren kodiert auch `zlib` seine Symbole. Für unsere Beispielverteilung $P$ wählt der Huffman-Algorithmus die Längen $1, 2, 2$. Damit benötigt er im Mittel $1.4$ Bit, nur $0.1$ Bit mehr als die Entropie. So entfallen zwei Drittel des Mehrbedarfs des Shannon-Codes. Doch selbst ein optimaler Symbolcode stößt an eine Grenze. Kein Codewort kann kürzer als $1$ Bit sein. Bei einer sehr ungleichen Verteilung macht diese Untergrenze von einem Bit fast die gesamte mögliche Ersparnis zunichte.

<!--en:2f06cde1-->
Betrachten wir eine Quelle, die `a` mit Wahrscheinlichkeit $0.9$ und `b` mit Wahrscheinlichkeit $0.1$ ausgibt. Ihre Entropie beträgt $0.47$ Bit pro Symbol, und jedes `a` trägt nur $\log_2(1/0.9) = 0.15$ Bit zum Überraschungswert bei. Dennoch benötigt jeder Symbolcode für jedes `a` mindestens ein ganzes Bit, mehr als das Sechsfache seines Informationsgehalts. Tritt `a` mit Wahrscheinlichkeit $0.999$ auf, sinkt die Entropie auf $0.011$ Bit pro Symbol. Jeder Symbolcode benötigt weiterhin $1$ Bit. Dieser Zuschlag „$+1$“ entsteht, weil wir für jedes Symbol einzeln aufrunden. Kodieren wir Blöcke von jeweils $n$ Symbolen, verteilen sich die Kosten. Wir behandeln jeden Block aus $n$ Symbolen als ein einziges Symbol eines größeren Alphabets. Damit runden wir nur einmal pro Block auf, und der Mehrbedarf pro Symbol sinkt unter $1/n$. Schon wenn wir *Paare* von `a`- und `b`-Symbolen mit dem Huffman-Algorithmus kodieren, sinkt die Rate von $1$ Bit auf $0.645$ Bit pro ursprünglichem Symbol.

<!--en:aca22a85-->
### Codes für abweichende Verteilungen und Kreuzentropie

<!--en:4b029850-->
In der Praxis kennen wir die wahre Datenverteilung $P$ selten. Daher konstruieren wir unseren Code aus einem Modell $Q$. Der Shannon-Code für $Q$ weist die Längen $\ell_i = \lceil \log_2 (1/q_i) \rceil$ zu. Das setzt $q_i > 0$ für jedes Symbol voraus, das $P$ ausgeben kann. Verwenden wir diesen Code für Daten, die tatsächlich von $P$ stammen, gilt für seine erwartete Länge

<!--en:9583aa69-->
denn $\log_2(1/q_i) \leq \ell_i < \log_2(1/q_i) + 1$ gilt für jedes Symbol $i$, und beim Mitteln unter $P$ bleiben beide Ungleichungen erhalten. Bis auf weniger als ein Bit Rundungsaufwand entspricht die Kreuzentropie also der erwarteten Länge eines für das Modell $Q$ entworfenen Codes. Nach :eqref:`eq_mdl-ce2-decomp` gibt die KL-Divergenz genau den Mehraufwand an, wenn wir $Q$ anstelle der wahren Verteilung $P$ verwenden.

<!--en:73c6da0b-->
Überprüfen wir diese Schranken numerisch. Die folgende Zelle konstruiert den Shannon-Code für ein Modell `q`, berechnet seine erwartete Länge unter einer Datenverteilung `p` und gibt die Kraft-Summe sowie die Kreuzentropie aus. Für beide Quellen aus diesem Abschnitt testen wir sowohl passende Codes als auch Codes für abweichende Verteilungen. Bei der dyadischen Quelle verwenden wir als abweichendes Modell die Gleichverteilung, bei unserer Beispielverteilung $P$ das Modell $Q = (0.2, 0.5, 0.3)$.

<!--en:cafef30b-->
Die beiden Zeilen mit passenden Codes bestätigen unsere Rechnung. Hier entspricht `CE2` der Entropie, und die erwarteten Längen betragen $1.75$ und $1.6$ Bit. Die Zeilen mit abweichendem Modell zeigen die Kosten der falschen Modellwahl. Der aus $Q$ konstruierte Shannon-Code hat die Längen $3, 1, 2$. Damit bekommt ausgerechnet das häufigste Symbol von $P$ das längste Codewort. Auf Daten aus $P$ benötigt der Code im Mittel $2.3$ Bit pro Symbol. Das liegt im Intervall $1.867 \leq 2.3 < 2.867$, das :eqref:`eq_mdl-wrong-code-length` vorhersagt. Der Code für die Gleichverteilung weist jedem Symbol der dyadischen Quelle zwei Bit zu. Es gibt keinen Rundungsverlust, die untere Schranke wird also exakt erreicht. Die Zerlegung $2.0 = 1.75 + 0.25$ Bit ergibt die Entropie zuzüglich einer KL-Divergenz von $0.25$ Bit. Die Spalte mit der Kraft-Summe zeigt schließlich, wie viel Budget jeder Code ungenutzt lässt. Bei der dyadischen Quelle bleibt nichts übrig, $0.1875$ bleiben beim Shannon-Code für $P$ übrig und $0.125$ beim Code aus $Q$.

<!--en:9a10ed87-->
Beachten wir eine Feinheit. Die Schranke :eqref:`eq_mdl-wrong-code-length` gilt speziell für den Shannon-Code von $Q$. Andere aus $Q$ konstruierte Codes müssen sie nicht erfüllen. Der Huffman-Code für $Q$ hat die Längen $2, 1, 2$, schöpft das Kraft-Budget aus und benötigt im Mittel nur $1.2 + 0.3 + 0.2 = 1.7 < 1.867$ Bit auf Daten aus $P$. Das ist kein Widerspruch. Der Code impliziert die Verteilung $(\tfrac14, \tfrac12, \tfrac14)$ anstelle von $Q$, und diese dyadische Verteilung liegt zufällig näher an $P$. Wie :eqref:`eq_mdl-code-length-decomp` zeigt, hängt die zusätzliche Codelänge nur von der impliziten Verteilung und dem ungenutzten Budget ab. $Q$ geht nur über die daraus bestimmten Codewortlängen ein.

<!--en:28b96ce2-->
## Stromcodes und Beschreibungslänge

<!--en:7242c1d1-->
Bei einem Symbolcode fallen für jedes Symbol Rundungskosten an. Konstruieren wir nun einen Code, der nur einmal pro Nachricht rundet, und verbinden wir ihn direkt mit einem Sprachmodell. Anschließend fragen wir, wie wir die Kosten für die Beschreibung des Modells selbst berücksichtigen.

<!--en:12d7fa0c-->
### Arithmetische Kodierung

<!--en:5949bb80-->
Kodieren wir ganze Symbolblöcke, vermeiden wir den Rundungsverlust pro Symbol. Doch die Zahl der möglichen Blöcke wächst rasch. Ein Codebuch für Blöcke aus $100$ binären Symbolen bräuchte $2^{100}$ Einträge, viel zu viele, um sie zu speichern. Wir suchen also die Effizienz eines Blockcodes ohne dessen Nachschlagetabelle. Dazu wollen wir für die gesamte Nachricht ein einziges Codewort direkt beim Lesen berechnen. Solche Codes heißen *Stromcodes* (engl. stream codes). Die klassische Konstruktion ist die *arithmetische Kodierung* :cite:`Rissanen.1976,Witten.Neal.Cleary.1987`. Sie schlägt kein Codewort nach, sondern stellt die Nachricht durch ein Teilintervall von $[0, 1)$ dar. :numref:`fig_mdl-arith-intervals` veranschaulicht das Verfahren für die Nachricht `aaaaaaaaba` aus zehn Symbolen. Sie stammt von unserer Quelle mit ungleichen Wahrscheinlichkeiten ($q(\texttt{a}) = 0.9$) und wird mit dem wahren Modell kodiert.

<!--en:2c139e91-->
![Arithmetische Kodierung der Nachricht aaaaaaaaba mit dem Modell $q(\texttt{a}) = 0.9$. (a) Jedes Symbol schränkt das aktuelle Intervall auf das Teilintervall ein, das ihm das Modell zuweist. Nach zehn Symbolen beträgt die verbleibende Breite $w = 0.9^9 \cdot 0.1 = 0.0387$, die Wahrscheinlichkeit der Nachricht. (b) Ein vollständig im Endintervall enthaltenes dyadisches Intervall kodiert das Endintervall. Keines der beiden dyadischen Intervalle der Länge $2^{-5}$, die es schneiden, passt ganz hinein. Das Intervall der Länge $2^{-6}$ von $0.390625$ bis $0.40625$ hingegen liegt vollständig darin. Seine Binäradresse 011001 ist das Codewort: sechs Bit für eine Nachricht mit einem Überraschungswert von $4.69$ Bit.](../img/mdl-it-arith-intervals.svg)

<!--en:cd063fac-->
Der Encoder verwaltet ein Intervall, zu Beginn $[0, 1)$. Um das nächste Symbol $x_i$ zu kodieren, zerlegt er das aktuelle Intervall in je ein Teilintervall pro möglichem Symbol. Die Breiten sind proportional zu den bedingten Wahrscheinlichkeiten $q(\cdot \mid x_{<i})$ des Modells. Der Encoder behält nur das Teilintervall des tatsächlich aufgetretenen Symbols. Bei jedem Schritt multipliziert sich die Intervallbreite mit der Wahrscheinlichkeit des beobachteten Symbols. Nach der Verarbeitung von $x_1, \ldots, x_n$ hat das Endintervall also die Breite

<!--en:b3446d4e-->
Dies ist genau die Wahrscheinlichkeit, die das Modell der gesamten Nachricht zuweist. Verschiedene Nachrichten der Länge $n$ führen zu disjunkten Intervallen. Daher legt jeder Punkt im Endintervall die Nachricht eindeutig fest. Der Decoder vollzieht die Schritte mit demselben Modell nach. Er prüft, welches Teilintervall die übertragene Zahl enthält, gibt das zugehörige Symbol aus und schränkt sein Intervall im Gleichschritt mit dem Encoder ein. Der Decoder muss zudem das Ende erkennen können. Dazu kennt er entweder $n$ im Voraus, oder wir verwenden ein eigenes Symbol für das Nachrichtenende. Die Kosten dafür lassen wir hier außer Acht. Wir müssen nur noch einen Punkt im Endintervall binär angeben. Dazu eignen sich die dyadischen Intervalle aus dem Beweis der Kraft-Ungleichung.

<!--en:883afea4-->
Jedes Intervall der Breite $w$ enthält ein dyadisches Intervall der Länge $2^{-\ell}$, wenn wir $\ell = \lceil \log_2(1/w) \rceil + 1$ wählen. Warum? Diese Wahl garantiert $2^{-\ell} \leq w/2$. Somit liegt das erste Vielfache von $2^{-\ell}$ am oder rechts vom linken Rand noch in der linken Intervallhälfte. Das dort beginnende dyadische Intervall der Länge $2^{-\ell}$ reicht daher höchstens bis zum rechten Rand. Teil (b) von :numref:`fig_mdl-arith-intervals` zeigt beide Fälle. Die dyadischen Intervalle der Länge $2^{-5}$ ragen über die Grenzen des Endintervalls hinaus, während eines der Länge $2^{-6}$ vollständig hineinpasst. Wegen $\ell < \log_2(1/w) + 2$ gilt für die Länge des resultierenden Codeworts

<!--en:0189a65d-->
Zur Kodierung benötigen wir also den gesamten Überraschungswert der Nachricht unter dem Modell plus höchstens zwei zusätzliche Bits (bei bekannter Nachrichtenlänge $n$). Wir runden nun einmal für die gesamte Nachricht statt einmal pro Symbol. Die Codewörter für verschiedene Nachrichten der Länge $n$ bezeichnen zudem dyadische Intervalle innerhalb disjunkter Nachrichtenintervalle. Nach dem ersten Teil des Beweises der Kraft-Ungleichung bilden sie daher einen Präfixcode für Nachrichten. Die untere Entropieschranke gilt also für ganze Nachrichten ebenso wie für einzelne Symbole. Unabhängig von der Wahl des Modells kann die erwartete Codelänge nie unter die Entropie der Nachricht fallen.

<!--en:1078c72e-->
Wenden wir diese schrittweise Intervalleinschränkung auf `aaaaaaaaba` an und vergleichen wir die Codelänge mit der des symbolweise arbeitenden Shannon-Codes. Die folgende Zelle verfolgt das Intervall und bestimmt das binäre Codewort. Sie prüft, ob das zugehörige dyadische Intervall im Endintervall liegt, und gibt beide Bitzahlen aus.

<!--en:ffb39522-->
Die endgültige Breite $0.0387$ entspricht wie erwartet dem Produkt $0.9^9 \cdot 0.1$. Der Überraschungswert der Nachricht beträgt $4.69$ Bit. Die arithmetische Kodierung speichert sie in $6$ Bit (`011001`). Der symbolweise arbeitende Shannon-Code benötigt dagegen $13$ Bit, denn er verwendet für jedes der neun eigentlich günstigen `a` ein ganzes Bit. Diese Zeichenfolge hat genau die mittlere Zusammensetzung der Quelle (neunmal `a` und einmal `b`). Daher beträgt ihr Überraschungswert $10\, H_2(0.9, 0.1) = 4.69$ Bit. Eine Nachricht, in der `b` zweimal vorkommt, würde mehr Bits benötigen.

<!--en:954162c2-->
Die direkte Intervalleinschränkung mit Gleitkommazahlen funktioniert nur für kurze Nachrichten. Die Breite schrumpft in jedem Schritt um den Faktor $q$. Nach rund hundert Symbolen lassen sich die Endpunkte mit der $53$ Bit langen Mantisse einer Gleitkommazahl doppelter Genauigkeit nicht mehr unterscheiden (Aufgabe :exref:`ex_mdl-information-theory-exact-arithmetic-coding`). Praktische arithmetische Kodierer rechnen mit ganzen Zahlen fester Bitbreite. Sobald beide Endpunkte in den führenden Bits übereinstimmen, geben sie diese aus und skalieren das Arbeitsintervall wieder auf volle Genauigkeit :cite:`Witten.Neal.Cleary.1987`. Mit dieser Skalierung darf die Nachricht beliebig lang sein. Der Zuschlag von zwei Bit wird pro Symbol vernachlässigbar. Für einen Text aus $n$ Symbolen beträgt der Speicherbedarf dann das $n$-Fache des mittleren logarithmischen Verlusts des Modells in Bit.

<!--en:6df165bd-->
### Ein Sprachmodell als Kompressor

<!--en:9fd24f67-->
Bei der arithmetischen Kodierung haben wir nirgendwo vorausgesetzt, dass die Symbole unabhängig sind. Jedes bedingte Modell $q(\cdot \mid x_{<i})$ kann die Intervalleinschränkung steuern, sofern Encoder und Decoder bei endlicher Genauigkeit bitgenau dieselben Wahrscheinlichkeiten für das nächste Symbol berechnen. Kombinieren wir ein autoregressives Sprachmodell mit einem arithmetischen Kodierer, erhalten wir ein verlustfreies Kompressionsverfahren. Verwenden beide Seiten denselben Tokenizer, dasselbe Modell und denselben Kodierer, lässt sich jedes Dokument auf seinen gesamten Überraschungswert unter dem Modell plus höchstens zwei Bit komprimieren. (In praktischen Dateiformaten kommen geringe Kosten für die Abgrenzung der Nachricht hinzu. Ist das Modell vorab festgelegt, müssen beide Seiten es bereits besitzen.)

<!--en:4008e618-->
Das einfachste Modell, das während des Lesens lernt, ist ein Zählmodell. Ein *adaptives Modell der Ordnung $m$* sagt jedes Zeichen anhand der $m$ unmittelbar vorhergehenden Zeichen vorher. Sie bilden seinen *Kontext* $s$. Das Modell verwendet dazu

<!--en:e394b247-->
Dabei zählt $c(s, x)$, wie oft bisher $x$ auf $s$ folgte. $c(s) = \sum_x c(s, x)$ ist die Gesamtzahl der Beobachtungen im Kontext $s$ und $k$ die Größe des Alphabets. Addieren wir $\tfrac12$ zu jedem Zähler, erhält kein Zeichen die Wahrscheinlichkeit null (das würde unendlich viele Bits kosten). Entscheidend ist, dass das Modell seine Zähler erst *nach* der Kodierung des jeweiligen Zeichens aktualisiert. Der Decoder aktualisiert seine Zähler auf dieselbe Weise anhand der bereits rekonstruierten Zeichen. So bleiben Encoder und Decoder im Gleichschritt, und wir müssen keine Modellparameter übertragen.

<!--en:2bab3136-->
Testen wir dieses Verfahren an *The Time Machine*. Die erste Zelle lädt den Text herunter und bereinigt ihn, sodass nur Kleinbuchstaben und Leerzeichen bleiben. Anschließend definiert sie eine Funktion, die die adaptive Codelänge in Bit pro Zeichen berechnet.

<!--en:aed24899-->
Als Nächstes testen wir Kontextlängen von null bis vier und vergleichen sie mit `zlib` auf der höchsten Kompressionsstufe. Längere Kontexte erlauben bessere Vorhersagen, sobald genügend Beobachtungen vorliegen. Allerdings muss das Modell für wesentlich mehr Fälle Häufigkeiten lernen. Die Ausgabe zeigt, ab wann der Nachteil überwiegt.

<!--en:a03802d4-->
Mit Kontext sinkt die Codelänge von $4.08$ Bit pro Zeichen ohne Kontext auf $2.55$ Bit pro Zeichen bei drei Kontextzeichen. Damit unterbieten wir das Ergebnis von `zlib` mit $2.84$. Mit einem vierten Zeichen kehrt sich der Gewinn jedoch um. Beim Modell der Ordnung $4$ steigt die Länge wieder auf $2.78$ Bit pro Zeichen. Die meisten seiner $27^4 \approx 530{,}000$ möglichen Kontexte kommen im Roman zu selten vor, um die Wahrscheinlichkeiten für das Folgezeichen zu lernen. Jedes zusätzliche Bit, das beim Aufbau dieser Zähler anfällt, vergrößert unmittelbar die Datei.

<!--en:3ac9df46-->
Ohne Kontext muss das Modell dagegen nur wenig lernen. Die erste Ausgabezeile nennt die Entropie der empirischen Zeichenhäufigkeiten im Roman, $4.078$ Bit pro Zeichen. Diese Rate würde ein kontextfreier Kodierer erreichen, wenn er die Häufigkeiten im Voraus kennte. Das Modell der Ordnung $0$ beginnt ohne Vorwissen. Die letzte Zeile zeigt die genauen Kosten des Lernens: nur $0.0011$ Bit pro Zeichen oder $183$ Bit für das gesamte Buch. Die $26$ Zeichenhäufigkeiten zu lernen kostet wenig, eine halbe Million Kontexte zu lernen dagegen viel. (Auf die Spalten für Nat und Perplexität kommen wir in :numref:`subsec_mdl-perplexity` zurück.)

<!--en:a0cf9fba-->
Ein adaptives Modell erspart die Übertragung von Parametern. Dafür sagt es die Zeichen am Dateianfang schlecht vorher, weil es noch lernt. Bei einem vortrainierten Modell verhält es sich umgekehrt. Es komprimiert jede Datei vom ersten Byte an gut, sofern beide Seiten seine Gewichte bereits einmal ausgetauscht haben. :citet:`Deletang.Ruoss.Duquenne.ea.2023` zeigen diesen Zusammenhang mit einem Sprachmodell mit 70 Milliarden Parametern. Steuern seine Wahrscheinlichkeiten für das nächste Token einen arithmetischen Kodierer, schrumpfen Bild-Patches aus ImageNet auf $43.4\%$ ihrer Rohgröße (PNG erreicht $58.5\%$) und Audiodaten aus LibriSpeech auf $16.4\%$ (FLAC erreicht $30.3\%$). Diese Zahlen lassen das Modell selbst unberücksichtigt. Bei zwei Byte pro Parameter belegen die Gewichte $140$ GB, gegenüber $1$ GB für jeden Datensatz. Rechnen wir das Modell ein, unterbietet das Verfahren die Standardcodecs nicht mehr.

<!--en:ee707b01-->
### Minimale Beschreibungslänge

<!--en:a2db1bd7-->
In unserem Experiment mit *The Time Machine* trug jedes adaptive Modell während der Kodierung die Kosten für das Lernen seiner Häufigkeiten. Lassen wir diese Kosten weg, kehrt sich die Rangfolge vollständig um. Zählen wir die Vorkommen zuerst im ganzen Roman und bewerten wir danach denselben Roman mit diesen Häufigkeiten, schlägt Ordnung $4$ die Ordnung $3$, Ordnung $5$ die Ordnung $4$ und so weiter. Eine größere Nachschlagetabelle kann den Text auswendig lernen (Aufgabe :exref:`ex_mdl-information-theory-adaptive-compression`). Beurteilen wir ein Modell allein danach, wie gut es nachträglich zu den Daten passt, verliert ein reichhaltigeres Modell nie. Ein Empfänger, der das Modell noch nicht besitzt, braucht jedoch zuerst dessen Beschreibung. Das Prinzip der *minimalen Beschreibungslänge* (engl. minimum description length, MDL) berücksichtigt dies. Es wählt das Modell $\mathcal{M}$, das die gemeinsame Codelänge von Modell und Daten minimiert:

<!--en:024a6e2e-->
Dabei ist $L(\mathcal{M})$ die Anzahl der Bits, die wir benötigen, um das Modell anzugeben. $L(\mathcal{D}\mid\mathcal{M})$ ist der logarithmische Verlust für die Daten $\mathcal{D}$ unter diesem Modell, also nach den vorigen Unterabschnitten die Codelänge in Bit. MDL bevorzugt ein aufwendigeres Modell nur dann, wenn es bei den Daten mehr Bits einspart, als seine Beschreibung zusätzlich kostet. In diesem Sinne drückt MDL Ockhams Rasiermesser in Bit aus. Ergibt sich die Modellbeschreibung über $L(\mathcal{M}) = -\log_2 P(\mathcal{M})$ aus einer A-priori-Verteilung, minimieren wir mit der Summe die negative Log-A-priori-Wahrscheinlichkeit plus die negative Log-Likelihood. Dann entspricht MDL der Maximum-a-posteriori-Schätzung (:numref:`subsec_mdl-map`).

<!--en:766e626a-->
An einem Münzwurfbeispiel lassen sich beide Terme ausrechnen. Nehmen wir an, $n = 1{,}000$ Würfe ergeben $600$ Mal Kopf. Für die Hypothese einer fairen Münze $\mathcal{M}_0$ müssen wir keine Parameter übertragen. Jeder Wurf benötigt $1$ Bit, insgesamt also $L(\mathcal{D} \mid \mathcal{M}_0) = 1{,}000$ Bit. Für die Hypothese einer unfairen Münze $\mathcal{M}_1$ müssen wir zuerst ihre Kopfwahrscheinlichkeit $\theta$ übertragen und dazu eine endliche Genauigkeit wählen. Der beobachtete Kopfanteil bei $n$ Würfen schwankt in der Größenordnung $1/\sqrt{n}$. Geben wir $\theta$ genauer als bis auf $1/\sqrt{n}$ an, verschwenden wir Bits zur Beschreibung von Stichprobenrauschen (Aufgabe :exref:`ex_mdl-information-theory-two-part-coin`). Ein Gitter mit Abstand $1/\sqrt{n}$ auf $(0, 1)$ hat $\sqrt{n}$ Punkte. Den nächstgelegenen Gitterpunkt anzugeben kostet also $L(\mathcal{M}_1) = \log_2 \sqrt{n} = \tfrac12 \log_2 1{,}000 \approx 5$ Bit. Bei gegebenem $\theta$ benötigen die Daten $L(\mathcal{D} \mid \mathcal{M}_1) \approx 1{,}000\, H_2(0.6, 0.4) = 971$ Bit, insgesamt ergeben sich $976 < 1{,}000$ Bit. MDL wählt daher die unfaire Münze. Die $5$ Bit für den Parameter sparen $29$ Bit bei den Daten. Hätten wir nur $520$ Mal Kopf beobachtet, betrüge die Ersparnis bei den Daten nur $1.2$ Bit. Das wäre weniger als die $5$ Bit für den Parameter, und MDL bliebe bei der fairen Münze. Der Ansatz, für jeden reellen Parameter $\tfrac12 \log_2 n$ Bit zu veranschlagen, ist als *bayessches Informationskriterium* bekannt.

<!--en:62628466-->
Tatsächlich haben wir in unserem Experiment mit *The Time Machine* bereits MDL angewandt. Eine adaptive Codelänge erfasst eine gültige, vollständige Beschreibung der Daten. Die Kosten für das Lernen des Modells gehen direkt in den Bitstrom ein. Für das Modell der Ordnung $0$ sind diese Kosten gering. Setzen wir $\tfrac12 \log_2 n \approx 8.7$ Bit für jede der $26$ freien Zeichenhäufigkeiten an, erwarten wir etwa $226$ Bit. Das liegt recht nahe an den tatsächlich beobachteten $183$ Bit (das Kriterium berücksichtigt nur den führenden asymptotischen Term). Beim Modell der Ordnung $4$ kostete es mehr Bits, für jeden beobachteten Kontext eine eigene Verteilung zu lernen, als das vierte Kontextzeichen einsparte. Die adaptive Codelänge fiel daher automatisch für Ordnung $3$ günstiger aus.

<!--en:df77b7eb-->
Die Parameter zu zählen hätte hier zu einer groben Fehleinschätzung geführt. Formal besitzt das Modell der Ordnung $4$ insgesamt $27^4 \cdot 26 \approx 14$ Millionen freie Häufigkeiten. Bei jeweils $8.7$ Bit würde das bayessche Informationskriterium dafür $120$ Millionen Bit veranschlagen. Dabei benötigt der Roman unkomprimiert mit acht Bit pro Zeichen nur $1.4$ Millionen Bit. Die asymptotische Rechnung mit der Parameteranzahl setzt wesentlich mehr Daten als Parameter voraus. Tatsächlich treten die meisten Kontexte aus vier Zeichen nie auf, und der adaptive Code berechnet dafür nichts. Hat ein Modell mehr Parameter als Daten, wie es auch bei modernen neuronalen Netzen vorkommt, bleibt die adaptive Codelänge als Beschreibungslänge aussagekräftig.

<!--en:10ecbb25-->
## Entropierate und typische Sequenzen

<!--en:f8fc4d86-->
Bisher haben wir ein Modell vorgegeben und die Länge seines Codes bestimmt. Kehren wir die Frage nun um. Wie kurz kann bei einer Quelle mit Gedächtnis der Code des *bestmöglichen* Modells sein?

<!--en:07d8aed1-->
### Die Entropierate

<!--en:b2290da5-->
Für unabhängige Symbole ist die Entropie eines einzelnen Symbols die grundlegende untere Kompressionsschranke. In echtem Text hängen die Zeichen jedoch stark voneinander ab. Im Englischen folgt auf `q` ein `u`, und auf `th` folgt viel öfter ein `e` oder ein `a` als ein `z`. Um Quellen mit Gedächtnis zu untersuchen, nehmen wir an, die Sequenz $X_1, X_2, \ldots$ sei *stationär*. Das heißt, die gemeinsame Verteilung jedes Symbolblocks bleibt bei einer zeitlichen Verschiebung unverändert.

<!--en:3470695a-->
Wir verwenden die gemeinsame und die bedingte Entropie aus :numref:`subsec_mdl-joint-conditional-entropy`. Im Folgenden bezeichnet $H$ die Entropie von Zufallsvariablen in Bit. Mit $H_2(p_1, \ldots, p_k)$ bezeichnen wir weiterhin die binäre Entropie eines ausdrücklich angegebenen Wahrscheinlichkeitsvektors. Die gemeinsame Entropie $H(X_1, \ldots, X_n)$ behandelt einen ganzen Block aus $n$ Symbolen als eine Zufallsvariable. Die bedingte Entropie $H(X_n \mid X_{<n}) = \mathbb{E}\big[-\log_2 p(X_n \mid X_{<n})\big]$ misst den erwarteten logarithmischen Verlust des bestmöglichen Modells für die Vorhersage des nächsten Symbols, also der wahren Quelle selbst. Durch wiederholtes Anwenden der Kettenregel $H(X, Y) = H(X) + H(Y \mid X)$ zerlegen wir die Blockentropie in eine Summe bedingter Entropien: $H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_{<i})$. Die *Entropierate* der Quelle ist der Grenzwert der bedingten Entropie oder, gleichbedeutend damit, der Grenzwert der Entropie pro Symbol für lange Blöcke:

<!--en:26d85e27-->
Die Blockentropie ist die Summe von $n$ bedingten Entropien. Somit ist der erste Ausdruck der fortlaufende Mittelwert der Terme, deren Grenzwert im zweiten Ausdruck steht. Für jede stationäre Quelle existieren beide Grenzwerte und stimmen überein. Zudem fällt $\tfrac1n H(X_1, \ldots, X_n)$ monoton gegen $H_{\textrm{rate}}$ :cite:`Cover.Thomas.1999`. Diese Ergebnisse setzen wir voraus. Anschaulich erklärt sich das so: Konditionieren erhöht die Entropie nie (:numref:`sec_mdl-information_theory`). Jedes zusätzliche Kontextsymbol kann die Unsicherheit über das nächste Symbol daher allenfalls verringern. Auch der fortlaufende Mittelwert einer monoton fallenden Folge ist monoton fallend. Sind die Symbole unabhängig und identisch verteilt, ergeben beide Grenzwerte unmittelbar $H(X_1)$.

<!--en:110a8af0-->
Eine Markov-Kette mit zwei Zuständen zeigt, wie stark Gedächtnis die untere Schranke senkt. Die Kette gebe `a` oder `b` aus. Mit Wahrscheinlichkeit $0.9$ bleibt sie beim aktuellen Symbol, mit Wahrscheinlichkeit $0.1$ wechselt sie. Aus Symmetrie treten beide Symbole gleich häufig auf. Ein für sich betrachtetes Symbol hat also die Entropie $H(X_1) = 1$ Bit. Kennen wir hingegen das vorige Symbol, verhält sich das nächste wie ein Wurf einer unfairen Münze mit Wechselwahrscheinlichkeit $0.1$. Dann gilt $H(X_n \mid X_{n-1}) = H_2(0.9, 0.1) = 0.47$ Bit. Ein längerer Rückblick liefert keine weitere Information. Allein das Gedächtnis senkt die grundlegende Schranke von $1$ Bit pro Symbol auf $0.47$ (Aufgabe :exref:`ex_mdl-coding-compression-markov-entropy-rate` behandelt die allgemeine asymmetrische Kette mit zwei Zuständen).

<!--en:40b24564-->
Für jede stationäre Quelle benötigt jeder eindeutig dekodierbare Code im Mittel mindestens $H_{\textrm{rate}}$ Bit pro Symbol. Denn jeder Code für Blöcke aus $n$ Symbolen ist ein Symbolcode über dem Alphabet aller Blöcke der Länge $n$. Nach der Entropieschranke, die nach dem Satz von McMillan auch für eindeutig dekodierbare Codes gilt, benötigt er mindestens $H(X_1, \ldots, X_n)$ Bit pro Block. Aus der Monotonie folgt $H(X_1, \ldots, X_n) \geq n H_{\textrm{rate}}$.

<!--en:43464c11-->
Ebenso wichtig ist, dass sich diese Schranke erreichen lässt. Ein arithmetischer Kodierer mit den wahren bedingten Wahrscheinlichkeiten der Quelle benötigt nach :eqref:`eq_mdl-arith-bound` weniger als den Überraschungswert der Nachricht plus zwei Bit. Der erwartete Überraschungswert eines Blocks aus $n$ Symbolen unter seiner eigenen Verteilung beträgt $H(X_1, \ldots, X_n)$. Daher benötigt der Kodierer im Mittel weniger als $\tfrac1n H(X_1, \ldots, X_n) + \tfrac2n$ Bit pro Symbol. Dieser Wert konvergiert gegen $H_{\textrm{rate}}$, wenn $n$ wächst.

<!--en:185fc9ad-->
In einem klassischen Experiment ließ Shannon Versuchspersonen den jeweils nächsten Buchstaben in gedruckten englischen Texten erraten. Er schloss daraus, dass die Entropierate ungefähr zwischen $0.6$ und $1.3$ Bit pro Zeichen liegt :cite:`Shannon.1951`. Das ist weit weniger als die $\log_2 27 = 4.75$ Bit bei gleichverteilten zufälligen Buchstaben und Leerzeichen. Jedes funktionierende Kompressionsverfahren liefert eine *obere* Schranke für die Entropierate. Unser Zählmodell der Ordnung $3$ liefert mit $2.55$ Bit pro Zeichen eine solche obere Schranke. Eine untere Schranke kann dagegen kein Kompressionsverfahren beweisen. Kommen wir auf die Kapiteleinleitung zurück. Ein Sprachmodell mit einem Verlust von $1.2$ Nat pro Zeichen erreicht $1.73$ Bit pro Zeichen. Das liegt deutlich unter den $2.55$ unseres Zählmodells, aber noch über dem oberen Ende des von Shannon angegebenen Bereichs.

<!--en:15c6674d-->
Die besten oberen Schranken liefern heute große Sprachmodelle. In Verbindung mit einem arithmetischen Kodierer komprimiert das Modell von :citet:`Deletang.Ruoss.Duquenne.ea.2023` Wikipedia-Text auf weniger als ein Bit pro Byte und liegt damit in Shannons Bereich. Bevor wir dies als Entropierate der englischen Sprache deuten, müssen wir drei Einschränkungen bedenken. Die Modellgröße ist nicht eingerechnet. Teile des Testtexts könnten bereits im Training vorgekommen sein. Und ein Byte Wikipedia-Auszeichnungstext entspricht keinem Symbol aus Shannons Alphabet mit $27$ Zeichen. Wo unterhalb dieser Schranken die wahre Rate liegt, ist unbekannt.

<!--en:58e85772-->
### Typische Sequenzen

<!--en:294b934f-->
Bisher haben wir *mittlere* Codelängen abgeschätzt. Wie sieht eine einzelne lange Nachricht aus, und wie viele wahrscheinliche Nachrichten gibt es überhaupt? Die Antwort zeigt die Entropie aus einem zweiten Blickwinkel. Als Exponent beschreibt sie die Anzahl der wahrscheinlichen Nachrichten. Der weitere Abschnitt baut nicht auf diesem Unterabschnitt auf. Sie können ihn beim ersten Lesen überspringen. Betrachten wir ein konkretes Zahlenbeispiel mit unabhängigen binären Symbolen, die den Wert $1$ mit Wahrscheinlichkeit $0.3$ annehmen, und Nachrichten der Länge $n = 200$.

<!--en:20a6fea0-->
Die wahrscheinlichste einzelne Nachricht besteht nur aus Nullen. Ihre Wahrscheinlichkeit beträgt $0.7^{200} = 2^{-103}$. Eine Nachricht mit der erwarteten Anzahl von $60$ Einsen hat die Wahrscheinlichkeit $0.3^{60}\, 0.7^{140} = 2^{-176.3}$. Das entspricht $2^{-nH}$ mit $H = H_2(0.3, 0.7) = 0.881$ Bit. Solche Nachrichten gibt es $\binom{200}{60} = 2^{172.2}$. Die meisten Zeichenfolgen gibt es dagegen bei $100$ Einsen, nämlich $2^{195.9}$. Jede einzelne hat aber nur die Wahrscheinlichkeit $2^{-225}$, sodass auf alle zusammen nur $2^{-29}$ der gesamten Wahrscheinlichkeitsmasse entfällt. Die meisten Nachrichten sind unwahrscheinlich, und die wahrscheinlichste Nachricht ist nicht repräsentativ.

<!--en:63c5375f-->
Schreiben wir $x_{1:n}$ für die Sequenz $x_1, \ldots, x_n$. Da die Symbole unabhängig sind, ist der Überraschungswert pro Symbol ein Stichprobenmittelwert unabhängiger und identisch verteilter Zufallsvariablen mit jeweiligem Erwartungswert $H = H_2(P)$. Nach dem Gesetz der großen Zahlen konvergiert dieser Stichprobenmittelwert in Wahrscheinlichkeit gegen seinen Erwartungswert:

<!--en:eae91cc4-->
Für jede feste Toleranz $\delta > 0$ weicht der Überraschungswert einer langen Nachricht also mit einer Wahrscheinlichkeit, die gegen eins geht, um weniger als $n\delta$ von $nH$ Bit ab. Dieses Phänomen heißt *asymptotische Gleichverteilungseigenschaft* (engl. asymptotic equipartition property, AEP). Die Nachrichten, die diese Bedingung erfüllen, bilden die *typische Menge* (engl. typical set) zur Toleranz $\delta$:

<!--en:0b29f1d0-->
**Proposition (AEP).** *Seien $X_1, \ldots, X_n$ unabhängige Ziehungen aus einer Verteilung $P$ auf einem endlichen Alphabet mit binärer Entropie $H$, und sei $\delta > 0$. Dann gilt (i) $P(A_\delta) \to 1$ für $n \to \infty$. (ii) Für jedes $x_{1:n} \in A_\delta$ gilt $2^{-n(H+\delta)} < P(x_{1:n}) < 2^{-n(H-\delta)}$. (iii) Es gilt $|A_\delta| \leq 2^{n(H+\delta)}$ und außerdem $|A_\delta| \geq (1-\eta)\,2^{n(H-\delta)}$, sofern $P(A_\delta) \geq 1-\eta$ gilt.*

<!--en:ab5df52d-->
**Beweis.** Teil (i) folgt unmittelbar aus :eqref:`eq_mdl-aep`. Teil (ii) ergibt sich durch Exponenzieren der Ungleichung, die $A_\delta$ definiert. Für Teil (iii) summieren wir die Schranken aus (ii) über alle Elemente von $A_\delta$. Das ergibt $1 \geq P(A_\delta) > |A_\delta|\, 2^{-n(H+\delta)}$ und $1 - \eta \leq P(A_\delta) < |A_\delta|\, 2^{-n(H-\delta)}$. Daraus folgen direkt beide Schranken für die Größe der Menge. $\blacksquare$

<!--en:59f217fc-->
Auf der Exponentialskala konzentriert sich also fast die gesamte Wahrscheinlichkeitsmasse auf ungefähr $2^{nH}$ typische Sequenzen, jede mit einer Wahrscheinlichkeit von ungefähr $2^{-nH}$. Hinter „ungefähr“ verbirgt sich allerdings ein Faktor $2^{\pm n\delta}$, der bei endlichen Längen beträchtlich sein kann. Bei $n = 200$ und $\delta = 0.1$ beträgt der Faktor $2^{20}$.

<!--en:76eec4c5-->
Aus der AEP folgt direkt Shannons Satz zur Quellenkodierung :cite:`Shannon.1948,Cover.Thomas.1999`. Zur Kompression weisen wir jeder Sequenz in $A_\delta$ einen Index fester Länge von $\lceil n(H+\delta) \rceil$ Bit zu. Bei jeder untypischen Nachricht melden wir einen Fehler. Die Anzahl der Bits pro Quellsymbol heißt *Rate* des Codes. Hier beträgt sie bis auf Rundung $H + \delta$, und für die Fehlerwahrscheinlichkeit gilt $P(A_\delta^c) \to 0$. Da wir $\delta > 0$ beliebig klein wählen können, ist jede Rate oberhalb von $H$ erreichbar. (Stellen wir ein einzelnes Kennbit voran und übertragen untypische Nachrichten unkomprimiert, erhalten wir einen vollständig verlustfreien Code mit derselben asymptotischen Rate.) Umgekehrt hat jeder Code fester Länge mit Rate $R < H$ nur $2^{nR}$ Codewörter. Er kann also eine Menge $B$ von höchstens $2^{nR}$ Nachrichten darstellen. Wählen wir $\delta < H - R$, so erhalten wir

<!--en:49e4c855-->
Bei jedem Code fester Länge mit einer Rate unter $H$ geht die Fehlerwahrscheinlichkeit also gegen eins. Für stationäre ergodische Quellen gilt derselbe Satz mit $H_{\textrm{rate}}$ anstelle von $H$. Diese Verallgemeinerung übernehmen wir ohne Beweis :cite:`Cover.Thomas.1999`.

<!--en:e4a77ede-->
Für unsere Bernoulli-Quelle können wir die Größe der typischen Menge *exakt* berechnen, weil alle Nachrichten mit derselben Anzahl $j$ von Einsen gleich wahrscheinlich sind. Die folgende Zelle gruppiert alle $2^{200}$ Bitfolgen nach $j$. Für die Gruppen, die bei $\delta = 0.1$ die Typizitätsbedingung erfüllen, summiert sie die Anzahlen und die Wahrscheinlichkeitsmassen. Anschließend vergleicht sie die resultierende Größe mit dem Richtwert $2^{nH} = 2^{176.3}$.

<!--en:09a50fc0-->
Die $2^{188.8}$ typischen Folgen enthalten zwischen $44$ und $76$ Einsen. Unter allen Bitfolgen ist nur eine von $2^{11}$ typisch, doch zusammen haben die typischen Folgen die Wahrscheinlichkeit $0.989$. Ihre Anzahl übersteigt $2^{nH}$ um einen Faktor von mehreren Tausend. Denn $\delta = 0.1$ lässt Folgen mit bis zu $76$ Einsen zu, von denen es sehr viel mehr gibt als von denen mit ungefähr $60$ Einsen. Dennoch liegt die Größe klar innerhalb der in der Proposition angegebenen Schranken $0.989 \cdot 2^{156.3}$ und $2^{196.3}$. Tragen wir Wahrscheinlichkeit und Anteil der Folgen gegen $j$ auf, um zu sehen, wie deutlich sie auseinanderliegen.

<!--en:9416aadf-->
Die Maxima der beiden Kurven liegen weit auseinander. Die Wahrscheinlichkeitsmasse erreicht ihr Maximum bei $j = 60$. Dort machen die Folgen nur den Anteil $2^{-28}$ aller Möglichkeiten aus. Die Anzahl der Folgen erreicht ihr Maximum bei $j = 100$, wo ihre Gesamtwahrscheinlichkeit nur $2^{-29}$ beträgt. Die wahrscheinlichste einzelne Folge ($j = 0$, nur Nullen) hat die Wahrscheinlichkeit $2^{-103}$ und liegt ganz außerhalb des dargestellten Bereichs. Typische Sequenzen liegen weder dort, wo die einzelnen Folgen am wahrscheinlichsten sind, noch dort, wo es die meisten Folgen gibt. Sie liegen in dem schmalen Bereich, in dem das Produkt aus Anzahl und Einzelwahrscheinlichkeit am größten ist.

<!--en:d41f6833-->
Für Beweise ist die typische Menge ein wirksames Werkzeug, als praktischer Kodierer dagegen umständlich. Eine direkte Indizierung kostet $189$ Bit pro typischer Folge, also $13$ Bit mehr als $nH = 176.3$, und scheitert bei etwa jeder hundertsten Nachricht. Die arithmetische Kodierung mit dem wahren Modell ist dagegen vollständig verlustfrei und benötigt im Mittel weniger als $nH + 2 = 178.3$ Bit gemäß :eqref:`eq_mdl-arith-bound`. Die typische Menge vermittelt uns eine geometrische Anschauung. Eine lange Nachricht spiegelt fast sicher die wahre Zusammensetzung der Quelle wider, und ungefähr $2^{nH}$ Nachrichten haben diese Zusammensetzung.

<!--en:3d4f72a2-->
## Perplexität

<!--en:afae362e-->
In der Praxis gibt man die Güte eines Sprachmodells selten als Dateigröße an. Stattdessen berichtet man für zurückgehaltenen Text eine einzige Zahl, die *Perplexität*. Sie drückt die Codelänge aus diesem Abschnitt in einer anderen Einheit aus.

<!--en:85ed9a5a-->
### Perplexität als Verzweigungsfaktor

<!--en:a9e52b6f-->
Gegeben seien ein Text aus $n$ Token und ein Modell, das dem jeweils nächsten Token die Wahrscheinlichkeit $q(\cdot \mid x_{<i})$ zuweist. Den mittleren logarithmischen Verlust pro Token in Nat und seinen Exponentialwert, die *Perplexität* :cite:`Jelinek.Mercer.Bahl.ea.1977`, definieren wir durch

<!--en:74789041-->
Ohne Verteilungen als Argumente bezeichnet $\textrm{CE}$ den empirischen Verlust des Modells auf genau diesem Text. Der Erwartungswert über Texte aus der Quelle ist die Kreuzentropie zwischen Quelle und Modell. Der zweite Ausdruck zeigt, dass die Perplexität der Kehrwert des geometrischen Mittels der Wahrscheinlichkeiten ist, die das Modell den beobachteten Token zuweist. Wie in :numref:`subsec_perplexity` eingeführt, lässt sich die Perplexität direkt als *effektiver Verzweigungsfaktor* (engl. effective branching factor) deuten. Ein Modell, das seine Wahrscheinlichkeit bei jedem Schritt gleichmäßig auf $k$ Token verteilt, hat die Perplexität $k$. Ein perfektes Vorhersagemodell für eine deterministische Sequenz hat die Perplexität $1$.

<!--en:360635d7-->
Die Ergebnisse für *The Time Machine* in :numref:`subsec_mdl-lm-compressor` veranschaulichen das auf Zeichenebene. Ein Modell ohne jedes Vorwissen würde jedem der $27$ Symbole dieselbe Wahrscheinlichkeit zuweisen. Seine Perplexität wäre $27$. Allein durch das Lernen der Zeichenhäufigkeiten sinkt sie auf $16.91$ (etwa $17$ gleich wahrscheinliche Möglichkeiten pro Zeichen). Mit drei Kontextzeichen sinkt sie auf $5.84$. Im Mittel ist das nächste Zeichen für das Modell also etwa so ungewiss wie für Sie das Ergebnis eines Wurfs mit einem sechsseitigen Würfel. Unser Programm berechnet die Perplexität als Potenz von $2$ mit den Bit pro Zeichen als Exponenten. Dagegen verwendet :eqref:`eq_mdl-perplexity-def` die Basis $e$ mit den Nat pro Zeichen als Exponenten. Beides ergibt dieselbe Zahl, denn es gilt $\textrm{PPL} = b^{\textrm{CE}_b}$ für jede Logarithmenbasis $b$ (Aufgabe :exref:`ex_mdl-information-theory-show-that-perplexity-does`).

<!--en:fd31f853-->
### Untere Schranke der Perplexität

<!--en:8315cbe8-->
So wie die mittlere Codelänge die Entropierate nicht unterschreiten kann, hat auch die Perplexität eine grundlegende untere Schranke. Für eine stationäre Quelle beträgt die erwartete Perplexität mindestens $2^{H_{\textrm{rate}}}$. Da wir $\textrm{CE}$ in Nat messen, $H_{\textrm{rate}}$ in :eqref:`eq_mdl-entropy_rate` dagegen in Bit, rechnen wir mit dem Faktor $\ln 2$ zwischen beiden Einheiten um.

<!--en:d2952965-->
**Proposition (untere Schranke der Perplexität).** *Seien $X_1, \ldots, X_n$ aus einer stationären Quelle gezogen und $q$ ein beliebiges Modell. Seien $\textrm{CE}$ und $\textrm{PPL}$ der empirische Verlust und die Perplexität gemäß :eqref:`eq_mdl-perplexity-def` auf diesem Text unter $q$. Dann gilt*

<!--en:66160501-->
**Beweis.** Fixieren wir ein beliebiges Präfix $x_{<i}$. Bei gegebenem $x_{<i}$ ist der erwartete Modellverlust für das nächste Token die Kreuzentropie zwischen der wahren bedingten Verteilung $p(\cdot \mid x_{<i})$ und $q(\cdot \mid x_{<i})$. Nach der Gibbs-Ungleichung ist sie mindestens so groß wie die bedingte Entropie der Quelle. Mitteln über die Präfixe ergibt $\mathbb{E}[-\log q(X_i \mid X_{<i})] \geq \mathbb{E}[-\log p(X_i \mid
X_{<i})]$. Die Log-Wahrscheinlichkeit einer Sequenz zerfällt gemäß $\log p(x_{1:n}) = \sum_i \log p(x_i \mid x_{<i})$ in eine Summe. Summieren über $i$ und Dividieren durch $n$ ergibt daher $\mathbb{E}[\textrm{CE}] \geq \tfrac{\ln 2}{n} H(X_1, \ldots, X_n) \geq
H_{\textrm{rate}} \ln 2$. Die zweite Ungleichung folgt daraus, dass die Blockentropie pro Symbol monoton fällt. Da $e^x$ konvex ist, folgt schließlich aus der Jensen-Ungleichung $\mathbb{E}[e^{\textrm{CE}}] \geq e^{\mathbb{E}[\textrm{CE}]}
\geq 2^{H_{\textrm{rate}}}$. $\blacksquare$

<!--en:57618e65-->
Die mittlere Unsicherheit eines Modells kann also nie geringer sein, als es die Zufälligkeit der Quelle zulässt. Um ein Gefühl für die Größenordnung zu bekommen, nehmen wir für einen Text eine Entropierate von einem Bit pro Zeichen an. Dieser Wert liegt in Shannons Bereich für Englisch. Dann könnte kein Zeichenmodell eine erwartete Perplexität unter $2$ erreichen, entsprechend einem fairen Münzwurf pro Buchstaben. Unser Zählmodell der Ordnung $3$ läge mit $5.84$ weit über der Schranke. Bei echtem Text ist diese Schranke ebenso ungewiss wie die Entropierate selbst, für die wir nur obere Schranken kennen.

<!--en:278754d0-->
Die Proposition liefert eine Schranke für einen *Erwartungswert*. Die Perplexität eines einzelnen kurzen Texts kann durchaus darunterliegen. Betrachten wir wieder unsere Quelle, die `a` mit Wahrscheinlichkeit $0.9$ ausgibt. Ihre untere Schranke beträgt $2^H = 1.38$. Selbst mit dem wahren Modell liegt die Perplexität einer Folge aus zehn Symbolen unter $1.38$, wenn kein einziges `b` vorkommt. Das geschieht mit Wahrscheinlichkeit $0.9^{10} = 35\%$. Auch bei langen Texten aus dieser Quelle liegt die Perplexität noch in fast der Hälfte der Fälle unter $2^H$. Der Abstand verschwindet allerdings mit wachsender Textlänge. Allgemeiner gilt für eine ergodische Quelle bei Bewertung mit dem wahren Modell, dass die Perplexität eines einzelnen langen Texts gegen $2^{H_{\textrm{rate}}}$ konvergiert, wenn $n \to \infty$. Das setzen wir hier voraus.

<!--en:9b221863-->
### Perplexität bei verschiedenen Tokenizern

<!--en:d878f4b9-->
Da die Perplexität auf einem Mittelwert pro Token beruht, hängt ihr Wert direkt vom Tokenizer ab. Zwei Modelle lassen sich daher nur dann unmittelbar anhand der Perplexität vergleichen, wenn sie denselben Tokenizer verwenden. :numref:`subsec_bits-per-byte` zeigt das Problem anhand dreier Tokenizer und löst es, indem der gesamte Überraschungswert durch die Anzahl der Bytes im Text geteilt wird. In den Begriffen dieses Abschnitts rechnen wir dazu die Perplexität in eine Codelänge pro Einheit Rohtext zurück. Für einen Text aus $n_{\textrm{char}}$ Zeichen, den der Tokenizer in $n_{\textrm{tok}}$ Token zerlegt, beträgt die gesamte Codelänge $n_{\textrm{tok}} \log_2 \textrm{PPL}$ Bit. Daraus folgt

<!--en:917b7ae8-->
Dabei hängt der Nenner nur vom Text selbst ab. Wenn jedes Zeichen wie in unserem Text mit 27 Symbolen genau ein Byte belegt, stimmen die Bit pro Zeichen mit den im Haupttext verwendeten Bit pro Byte überein.

<!--en:ff43d975-->
Alle Größen in diesem Abschnitt, von der Codelänge bis zur Perplexität, setzen voraus, dass sowohl Encoder als auch Decoder für jedes Symbol die Modellwahrscheinlichkeit $q(x_i \mid x_{<i})$ berechnen können. Sind zwei Verteilungen nur über Stichproben zugänglich, gibt es keinen solchen Code. Das ist etwa beim Vergleich der Ausgaben eines generativen Modells mit realen Daten der Fall. In :numref:`sec_mdl-divergences-distances` und :numref:`sec_mdl-test-functions-transport` entwickeln wir Verfahren, mit denen wir Verteilungen allein anhand von Stichproben vergleichen können.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:b533a5e5-->
Mit dem passenden Code wird der logarithmische Verlust eines Modells zur Dateigröße. Ein symbolweise arbeitender Präfixcode muss jede Codewortlänge auf eine ganze Anzahl von Bits runden. Seine optimale erwartete Länge liegt daher zwischen $H_2(P)$ und $H_2(P) + 1$ Bit. Bei der arithmetischen Kodierung entfällt dieser Rundungsaufwand pro Symbol, da sie nur einmal pro Nachricht rundet. Mit jedem Modell für das nächste Symbol, auch mit einem autoregressiven Sprachmodell, komprimiert sie eine Sequenz auf ihren gesamten Überraschungswert plus höchstens zwei Bit. Bei einem abweichenden Modell $Q$ gibt die Kreuzentropie $\textrm{CE}_2(P, Q)$ die erwartete Codelänge an. Die KL-Divergenz misst den Mehrbedarf, wenn wir $Q$ statt $P$ verwenden. Ein adaptives Zählmodell der Ordnung $3$ komprimiert *The Time Machine* auf $2.55$ Bit pro Zeichen und unterbietet damit `zlib`. Ordnung $4$ schneidet schlechter ab. Das Lernen der selteneren Kontexte kostet mehr Bits, als das zusätzliche Zeichen spart. Das veranschaulicht das Prinzip der minimalen Beschreibungslänge. Bei stationären Quellen mit Gedächtnis benötigt jeder eindeutig dekodierbare Code im Mittel mindestens so viele Bit pro Symbol, wie die Entropierate $H_{\textrm{rate}}$ angibt. Die Perplexität schließlich ist diese Codelänge in exponentieller Form. Sie wirkt als effektiver Verzweigungsfaktor, hat im Erwartungswert die untere Schranke $2^{H_{\textrm{rate}}}$ und lässt sich über verschiedene Tokenizer hinweg erst nach der Rückrechnung in Bit pro Zeichen oder Bit pro Byte vergleichen.

<!--en:2294243a-->
## Übungen

<!--en:20e68d49-->
1. **Satz von McMillan.** Ein Code ist eindeutig dekodierbar, wenn sich jede Verkettung von Codewörtern nur auf eine Weise zerlegen lässt.
    1. Zeigen Sie, dass der Code $\{0, 01, 011\}$ eindeutig dekodierbar, aber nicht präfixfrei ist. Prüfen Sie, ob er die Kraft-Ungleichung erfüllt.
    1. Beweisen Sie, dass jeder eindeutig dekodierbare Code die Kraft-Ungleichung :eqref:`eq_mdl-kraft` erfüllt. Hinweis: Entwickeln Sie $\big(\sum_i 2^{-\ell_i}\big)^m$ zu einer Summe über alle Folgen aus $m$ Codewörtern mit den Summanden $2^{-(\textrm{total length})}$. Gruppieren Sie die Folgen nach ihrer Gesamtlänge $L$, wobei $m \leq L \leq m \ell_{\max}$ gilt. Eindeutige Dekodierbarkeit erlaubt höchstens $2^L$ Folgen der Gesamtlänge $L$. Die Summe beträgt daher höchstens $m \ell_{\max}$. Ziehen Sie die $m$-te Wurzel und betrachten Sie den Grenzübergang $m \to \infty$.
   :exlabel:`ex_mdl-coding-compression-mcmillan`

<!--en:87c2f9b7-->
1. **Huffman-Codes.** Der Huffman-Algorithmus :cite:`Huffman.1952` konstruiert einen Präfixcode, indem er wiederholt die beiden unwahrscheinlichsten Symbole zu einem Symbol zusammenfasst. Dessen Wahrscheinlichkeit ist die Summe der beiden Wahrscheinlichkeiten. Anschließend weist er den beiden Partnern jeder Zusammenfassung $0$ und $1$ zu.
    1. Wenden Sie den Algorithmus auf $P = (0.6, 0.3, 0.1)$ und $(0.4, 0.3, 0.2, 0.1)$ an. Geben Sie für jeden Code die Codewortlängen und die erwartete Länge an und stellen Sie ihnen die Ergebnisse für den Shannon-Code sowie $H_2$ gegenüber.
    1. Beweisen Sie, dass der Huffman-Code unter allen Präfixcodes die kleinste erwartete Länge hat. Zeigen Sie zunächst, dass bei einem optimalen Code aus $p_i > p_j$ die Beziehung $\ell_i \leq \ell_j$ folgt. Zeigen Sie auch, dass die beiden längsten Codewörter gleich lang sind und im Baum als Geschwister gewählt werden können. Zeigen Sie dann, dass das Zusammenfassen der beiden unwahrscheinlichsten Symbole $i, j$ das Problem um ein Symbol verkleinert. Dabei gilt $\mathbb{E}[\ell] = \mathbb{E}[\ell'] + p_i + p_j$, wobei $\ell'$ die Längen des Codes nach dem Zusammenfassen bezeichnet. Schließen Sie den Beweis durch Induktion ab. Hinweis: Vertauschen Sie zwei Codewörter und vergleichen Sie die erwarteten Längen.
    1. Betrachten Sie die Quelle, die `a` mit Wahrscheinlichkeit $0.9$ ausgibt. Zeigen Sie, dass jeder Symbolcode mindestens ein Bit pro Symbol benötigt. Kodieren Sie dann *Symbolpaare* als Quelle mit vier Symbolen mithilfe des Huffman-Algorithmus. Überprüfen Sie den Wert von $0.645$ Bit pro ursprünglichem Symbol aus dem Text. Wiederholen Sie die Rechnung für Tripel. Wie groß ist der Mehrbedarf gegenüber der Entropie jeweils im Vergleich zur Schranke $1/n$?
    1. Betrachten Sie den Code für lineares Raten mit den Längen $1, 2, \ldots, k-1, k-1$. Sortieren Sie die Wahrscheinlichkeiten gemäß $p_1 \geq \cdots \geq p_k$. Zeigen Sie, dass der Huffman-Algorithmus diese Längen liefern kann, wenn $p_i \geq \sum_{j > i} p_j$ für jedes $i$ gilt. Lineares Raten ergibt dann einen optimalen Präfixcode. Die Quelle $(0.4, 0.3, 0.2, 0.1)$ verletzt diese Bedingung. Ist lineares Raten für sie optimal?
   :exlabel:`ex_mdl-coding-compression-huffman`

<!--en:8fbfa6d4-->
1. [code] **Exakte arithmetische Kodierung.** ● Die Intervalleinschränkung aus :numref:`subsec_mdl-arithmetic-coding` speichert `lo` und `hi` als Gleitkommazahlen. Ziehen Sie eine Nachricht aus $n = 1000$ Symbolen. Jedes Symbol sei `a` mit Wahrscheinlichkeit $0.9$ und andernfalls `b`.
    1. Wenden Sie die Intervalleinschränkung in Gleitkommaarithmetik mit dem Modell $q(\texttt{a}) = 0.9$ auf diese Nachricht an. Nach wie vielen Symbolen weicht `hi - lo` zum ersten Mal um mehr als $0.1\%$ vom Produkt der Wahrscheinlichkeiten ab? Nach wie vielen wird die Differenz null? Setzen Sie beide Antworten zur $53$ Bit langen Mantisse einer Zahl doppelter Genauigkeit in Beziehung.
    1. Wiederholen Sie die Intervalleinschränkung mit exakten rationalen Zahlen (`fractions.Fraction`). Geben Sie wie in der Codezelle einen Punkt des Endintervalls an. Schreiben Sie einen Decoder, der die Modellschritte nachvollzieht und so die Nachricht rekonstruiert. Vergleichen Sie die Codelänge mit dem Überraschungswert der Nachricht und mit $n H_2(0.9, 0.1)$.
    1. Kodieren Sie dieselbe Nachricht mit dem abweichenden Modell $q(\texttt{a}) = 0.7$. Vergleichen Sie die Zunahme der Codelänge mit $n D_{\textrm{KL}}^{(2)}(P \| Q)$. Warum unterscheiden sich die beiden Zahlen bei einer einzelnen Nachricht?
   :exlabel:`ex_mdl-information-theory-exact-arithmetic-coding`

<!--en:79ac6bd0-->
1. [extended] **Kompression mit adaptiven Kontextmodellen.** ●● Die Funktion `adaptive_bits` aus dem Text berechnet eine vollständige Beschreibungslänge im Sinne von :eqref:`eq_mdl-mdl` einschließlich der Kosten für das Lernen des Modells. Nach :eqref:`eq_mdl-arith-bound` benötigt ein arithmetischer Kodierer mit diesen Vorhersagen höchstens zwei Bit mehr. :citet:`Blier.Ollivier.2018` verwenden solche präquenziellen Codes, um die Beschreibungslänge tiefer Netze zu messen.
    1. Erweitern Sie das Experiment aus dem Text auf $m = 0, 1, \ldots, 8$. Vergleichen Sie für jedes $m$ die adaptive Codelänge pro Zeichen mit der Kreuzentropie pro Zeichen des Modells der Ordnung $m$, dessen Häufigkeiten aus dem gesamten Text stammen, sowie mit dem Ergebnis von `zlib`.
    1. Erklären Sie, warum die adaptive Codelänge als Funktion von $m$ ein Minimum hat, während die Kreuzentropie des angepassten Modells weiter fällt. Welche Ordnung wählt MDL? Welchen Term aus :eqref:`eq_mdl-mdl` lässt die Kreuzentropie des angepassten Modells unberücksichtigt?
    1. Entwerfen Sie ein adaptives Modell, das Kontexte mehrerer Ordnungen kombiniert. Beispielsweise kann es auf kürzere Kontexte zurückgreifen, wenn ein langer Kontext selten auftritt. Geben Sie für jedes getestete Modell und für `zlib` die Bit pro Zeichen in einer Tabelle an. Beschreiben Sie in einem Absatz, welche Entwurfsentscheidung den größten Einfluss hatte.
   :exlabel:`ex_mdl-information-theory-adaptive-compression`

<!--en:47d4aee8-->
1. **Zweiteiliger Code für eine Münze.** ● Eine Sequenz $\mathcal{D}$ aus $n$ Münzwürfen enthält $j$ Mal Kopf. Es gilt $\hat\theta = j/n$. Die Hypothese $\mathcal{M}_0$ beschreibt eine faire Münze. Es sind keine Parameter anzugeben, und es gilt $L(\mathcal{D} \mid \mathcal{M}_0) = n$ Bit. Die Hypothese $\mathcal{M}_1$ beschreibt eine Münze mit Kopfwahrscheinlichkeit $\theta$. Ihr Encoder sendet den Punkt $\theta_q$, der $\hat\theta$ auf einem Gitter mit Abstand $\Delta$ in $(0, 1)$ am nächsten liegt. Das kostet $L(\mathcal{M}_1) = \log_2(1/\Delta)$ Bit. Der Decoder kennt $n$ und $\Delta$.
    1. Zeigen Sie anhand der Zerlegung :eqref:`eq_mdl-ce2-decomp`, dass $L(\mathcal{D} \mid \mathcal{M}_1) = n H_2(\hat\theta, 1 - \hat\theta)
       + n D_{\textrm{KL}}^{(2)}(\hat\theta \,\|\, \theta_q)$ gilt. Die KL-Divergenz bezieht sich dabei auf die Bernoulli-Verteilungen mit diesen Parametern. In Nat beträgt die KL-Divergenz bis zur zweiten Ordnung $(\theta_q - \hat\theta)^2 / \big(2\hat\theta(1 - \hat\theta)\big)$. Leiten Sie die Entwicklung her. Zeigen Sie, dass die Gesamtlänge :eqref:`eq_mdl-mdl` von $\mathcal{M}_1$ den Wert $n H_2(\hat\theta, 1 - \hat\theta)$ um höchstens etwa $\log_2(1/\Delta) + n\Delta^2 /
       \big(8\hat\theta(1 - \hat\theta) \ln 2\big)$ Bit übersteigt.
    1. Welcher Gitterabstand $\Delta$ minimiert diese Schranke? Der Decoder muss $\Delta$ schon kennen, bevor er $\theta_q$ dekodiert. Warum kann der Code also den Minimierer nicht verwenden? Zeigen Sie, dass der Abstand $\Delta = 1/\sqrt{n}$, der nur von $n$ abhängt, einen Mehrbedarf von $\tfrac{1}{2}\log_2 n + O(1)$ Bit ergibt, sofern $\hat\theta$ einen positiven Mindestabstand von $0$ und $1$ einhält.
    1. Verwenden Sie $n = 1000$ und die Näherung $L(\mathcal{M}_1) + L(\mathcal{D} \mid \mathcal{M}_1) \approx
       n H_2(\hat\theta, 1 - \hat\theta) + \tfrac{1}{2}\log_2 n$ Bit. Wie der Text zeigt, gewinnt $\mathcal{M}_1$ bei $j = 600$ mit einem Vorsprung von $24$ Bit. Welche Hypothese gewinnt bei $j = 520$? Wie weit von $n/2$ muss $j$ ungefähr entfernt sein, damit $\mathcal{M}_1$ gewinnt? Wie vielen Standardabweichungen $\sqrt{n}/2$ von $j$ bei einer fairen Münze entspricht das?
   :exlabel:`ex_mdl-information-theory-two-part-coin`

<!--en:729de451-->
1. **Entropierate einer Kette mit zwei Zuständen.** Eine Kette auf $\{\texttt{a},
   \texttt{b}\}$ wechselt von `a` nach `b` mit Wahrscheinlichkeit $0.1$ und von `b` nach `a` mit Wahrscheinlichkeit $0.3$. Sie startet in ihrer stationären Verteilung.
    1. Bestimmen Sie die stationäre Verteilung sowie $H(X_1)$ in Bit.
    1. Zeigen Sie, dass $H(X_n \mid X_{n-1}) = \sum_s \pi_s H_2(r_s, 1 - r_s)$ Bit gilt, wobei $\pi$ die stationäre Verteilung und $r_s$ die Wahrscheinlichkeit für das Verlassen von Zustand $s$ ist. Zeigen Sie, dass die Entropierate diesem Wert entspricht, und berechnen Sie ihn.
    1. Zeigen Sie, dass $\tfrac1n H(X_1, \ldots, X_n) = \tfrac1n\big(H(X_1) +
       (n-1) H_{\textrm{rate}}\big)$ gilt und dass die Blockentropie pro Symbol monoton gegen $H_{\textrm{rate}}$ fällt.
    1. Betrachten Sie zwei Grenzfälle. Sind beide Wechselwahrscheinlichkeiten gleich $\tfrac12$, sind die Zustände unabhängig und identisch verteilt, und die Rate beträgt $1$ Bit. Sind beide Wechselwahrscheinlichkeiten gleich $r$, geht die Rate gegen $0$ für $r \to 0$, während $H(X_1)$ bei $1$ Bit bleibt. Was sagt $H(X_1)$ darüber aus, wie gut sich die Kette komprimieren lässt?
    1. [code] Simulieren Sie $10^5$ Schritte der ersten Kette und kodieren Sie sie mit `adaptive_bits` aus dem Text für die Ordnungen $0$, $1$ und $2$. Welche Ordnungen erreichen die Entropierate, und wogegen konvergiert die Codelänge pro Symbol bei Ordnung $0$?
   :exlabel:`ex_mdl-coding-compression-markov-entropy-rate`

<!--en:5273ed94-->
1. [code] **Typische Mengen und Mengen hoher Wahrscheinlichkeit.** ● Das Programm aus :numref:`subsec_mdl-aep` stuft eine Bitfolge der Länge $n = 200$ aus einer Bernoulli-Quelle mit $p = 0.3$ als typisch ein, wenn ihr Überraschungswert pro Symbol um weniger als $\delta = 0.1$ Bit von der Entropie $H$ in Bit abweicht. Es findet $2^{188.8}$ typische Folgen. Das liegt innerhalb der Schranken $0.989 \cdot 2^{156.3} \leq |A_\delta| \leq 2^{196.3}$ des Satzes zur AEP und deutlich über $2^{nH} = 2^{176.3}$.
    1. Welche Folge ist am wahrscheinlichsten? Berechnen Sie ihren Überraschungswert pro Symbol und entscheiden Sie, ob sie typisch ist. Warum spielt diese Folge für die Kompression keine Rolle?
    1. Sei $A$ die typische Menge und $B$ eine beliebige Menge von Folgen mit $P(B) \geq 1 - \eta$. Zeigen Sie, dass $P(A \cap B) \geq 1 - \eta - P(A^c)$ gilt, und folgern Sie daraus $|B| \geq \big(1 - \eta - P(A^c)\big)\, 2^{n(H - \delta)}$.
    1. Die kleinste Menge mit einer Wahrscheinlichkeit von mindestens $0.9$ erhält man, indem man die Folgen nach absteigender Wahrscheinlichkeit aufnimmt. Gruppieren Sie die Folgen wie im Programm nach der Anzahl $j$ ihrer Einsen. Berechnen Sie $\tfrac{1}{n}\log_2$ der Größe dieser Menge für $n = 200$, $1000$, $5000$ und $20000$. Vergleichen Sie das Ergebnis mit $H$ und mit $\tfrac{1}{n}\log_2 |A_\delta|$ bei gleichem $n$. Welche der beiden Mengen wächst wie $2^{nH}$? Warum gilt dieses Wachstum bei festem $\delta$ nicht für die typische Menge?

   *Nach David Tse, [EE376A/Stats376A: Informationstheorie, Übungsblatt 2](https://web.stanford.edu/~kedart/ee376a_winter1617/hw2.pdf), Aufgabe 6.*
   :exlabel:`ex_mdl-information-theory-typical-sets`

<!--en:066c53fb-->
1. **Perplexität bei verschiedenen Tokenizern.** Die Perplexität wird pro Token berechnet. Sie hängt also davon ab, wie der Tokenizer den Text zerlegt.
    1. Zeigen Sie, dass $b^{\textrm{CE}_b}$ für jede Logarithmenbasis $b$ denselben Wert hat. Dabei bezeichnet $\textrm{CE}_b$ die Kreuzentropie pro Token in Einheiten zur Basis $b$.
    1. Zwei Sprachmodelle werden auf demselben Text mit $10^6$ Zeichen ausgewertet. Der Tokenizer von Modell A zerlegt ihn in $2.5 \times 10^5$ Token, und A erreicht eine Perplexität pro Token von $20$. Der Tokenizer von Modell B zerlegt ihn in $6.7 \times 10^5$ Token, und B erreicht eine Perplexität pro Token von $8$. Rechnen Sie beide Ergebnisse mit :eqref:`eq_mdl-bpc` in Bit pro Zeichen um. Welches Modell sagt den Text besser vorher?
    1. Nehmen Sie für den Text eine Entropierate von einem Bit pro Zeichen an. Dieser Wert liegt in dem Bereich, den :citet:`Shannon.1951` für Englisch geschätzt hat. Nach :eqref:`eq_mdl-ppl-floor` beträgt die erwartete Perplexität pro Token mindestens $2^{H}$, wobei $H$ die Entropierate pro *Token* in Bit bezeichnet. Welche Perplexität kann ein Modell im Erwartungswert mit dem Tokenizer von A bestenfalls erreichen, welche mit dem von B? Wie weit ist jedes Modell von seiner unteren Schranke entfernt?
    1. Erklären Sie in zwei Sätzen, warum sich die beiden Perplexitäten nicht direkt vergleichen lassen, die beiden Werte in Bit pro Zeichen hingegen schon.
   :exlabel:`ex_mdl-information-theory-show-that-perplexity-does`

<!--en:b609883f-->
[Diskussion](https://d2l.discourse.group/t/36865)

<!--en:4f6d93c6-->
Codes, Kompression und Perplexität<br>**Präfixcodes · Kraft-Ungleichung · arithmetische Kodierung · MDL · Entropierate · AEP · Perplexität**

<!--en:df5ebabc-->
::: {.slide title="Warum führt eine bessere Vorhersage zu einer kürzeren Datei?"}

<!--en:d0ab186a-->
[Motivation und Leitfrage]{.kicker}

<!--en:689f618c-->
- **Kompression von *The Time Machine* ($173{,}428$ Zeichen, $27$ Symbole):** ASCII benötigt $8.00\text{ bpc}$, ein fester Code für 27 Symbole $4.75\text{ bpc}$ und `zlib -9` benötigt $2.84\text{ bpc}$. Ein kleines adaptives Zeichenmodell der Ordnung 3 erreicht **$2.55\text{ bpc}$**.

<!--en:b174d907-->
- **Vorhersage = Kompression:** Jedes kausale Modell $q(x_i \mid x_{<i})$ speichert mit einem arithmetischen Kodierer $n$ Symbole in $-\sum_{i=1}^n \log_2 q(x_i \mid x_{<i}) + 2$ Bit.

<!--en:4d3f989f-->
::: {.slide title="Präfixfreie Codes und binäre Codebäume"}

<!--en:4082f2e0-->
[Eindeutige Zerlegung]{.kicker}

<!--en:762adc7f-->
Wie kann der Empfänger einen Bitstrom ohne Trennzeichen zwischen den Symbolen zerlegen?

<!--en:7e11e393-->
- **Präfixfreiheit:** Ein Symbolcode ist *präfixfrei*, wenn kein Codewort Präfix eines anderen ist. Der Bitstrom `110010111` lässt sich mit $\{0, 10, 110, 111\}$ sofort und eindeutig als `110|0|10|111` zerlegen.

<!--en:51955d5f-->
- **Geometrie des Binärbaums:** Jedes Codewort der Länge $\ell_i$ besetzt ein Blatt in Tiefe $\ell_i$ und schließt alle Nachkommen darunter aus. Damit beansprucht es den Anteil $2^{-\ell_i}$ des vollständigen Binärbaums.

<!--en:a2687b4e-->
- **Vollständig belegter Baum:** Die Längen $(1, 2, 3, 3)$ verbrauchen $\tfrac12 + \tfrac14 + \tfrac18 + \tfrac18 = 1$. Kein Blatt bleibt frei.

<!--en:096af2be-->
::: {.slide title="Beweis der Kraft-Ungleichung mit dyadischen Intervallen"}

<!--en:85a58c15-->
[Existenz von Präfixcodes]{.kicker}

<!--en:7c474ad8-->
Ein präfixfreier Binärcode mit ganzzahligen Längen $\ell_1, \ldots, \ell_k$ existiert genau dann, wenn gilt:

<!--en:efb0a4e8-->
- **Notwendigkeit ($\Rightarrow$):** Dem Codewort $c$ der Länge $\ell$ entspricht das dyadische Intervall $[0.c, 0.c + 2^{-\ell}) \subset [0, 1)$. Präfixfreiheit $\iff$ disjunkte Intervalle $\implies$ Gesamtbreite $\le 1$.

<!--en:ce01114c-->
- **Hinreichende Bedingung ($\Leftarrow$):** Längen gemäß $\ell_1 \le \cdots \le \ell_k$ sortieren und die Intervalle der Breiten $2^{-\ell_i}$ von links nach rechts aneinanderlegen. Jeder Anfangspunkt $s_i$ ist ein ganzzahliges Vielfaches von $2^{-\ell_i}$.

<!--en:47480ced-->
::: {.slide title="Optimale Codewortlängen entsprechen dem Überraschungswert"}

<!--en:8f19752f-->
[Herleitung mit Lagrange-Multiplikatoren]{.kicker}

<!--en:2d4b40a7-->
Welche Codewortlängen minimieren bei gegebenen Symbolwahrscheinlichkeiten $P = (p_1, \ldots, p_k)$ die erwartete Länge $\mathbb{E}[\ell] = \sum_{i=1}^k p_i \ell_i$ unter dem Kraft-Budget $\sum_{i=1}^k 2^{-\ell_i} = 1$?

<!--en:b0b1fca3-->
- **Kontinuierliche Relaxation mit Lagrange-Multiplikatoren:** Ableiten der Lagrange-Funktion $\sum_i p_i \ell_i + \lambda(\sum_i 2^{-\ell_i} - 1)$ nach $\ell_i$ ergibt im Optimum $p_i - \lambda (\ln 2) 2^{-\ell_i} = 0$, also $2^{-\ell_i} \propto p_i$. Summieren über $i$ erzwingt $2^{-\ell_i} = p_i$: $$\ell_i^* = \log_2 \frac{1}{p_i}, \qquad \mathbb{E}[\ell^*] = \sum_{i=1}^k p_i \log_2 \frac{1}{p_i} = H_2(P)$$

<!--en:40fd5b07-->
- **Aufrunden auf ganze Zahlen (Shannon-Code):** Tatsächliche Codewörter haben ganzzahlige Längen. Aufrunden der idealen Längen ergibt die **Shannon-Längen** $\ell_i = \lceil \log_2(1/p_i) \rceil$. Sie erfüllen automatisch $2^{-\ell_i} \le p_i$ und damit die Kraft-Ungleichung ($\sum_i 2^{-\ell_i} \le \sum_i p_i = 1$).

<!--en:c27ed736-->
::: {.slide title="Entropieschranken für Präfixcodes"}

<!--en:a472bb1d-->
[Shannons Schranken für die Quellenkodierung]{.kicker}

<!--en:1d4dbac1-->
Bei ausgeschöpftem Anteil $Z = \sum_i 2^{-\ell_i} \le 1$ des Kraft-Budgets und impliziter Verteilung $q_i = 2^{-\ell_i}/Z$ zerfällt die erwartete Länge in drei Terme:

<!--en:62fad30a-->
- **Shannon-Schranken:** Die Wahl $\ell_i = \lceil \log_2(1/p_i) \rceil < \log_2(1/p_i) + 1$ garantiert $H_2(P) \le \mathbb{E}_P[\ell] < H_2(P) + 1$.

<!--en:a328b4e4-->
::: {.slide title="Shannon- und Huffman-Codes und die Untergrenze von 1 Bit"}

<!--en:2f633b76-->
[Optimale Symbolcodes]{.kicker}

<!--en:1d6955b0-->
- **Gieriges Zusammenfassen nach Huffman:** Wiederholt die beiden seltensten Symbole zusammenfassen, sodass $Z = 1$. Für $P = (0.6, 0.3, 0.1)$ sinkt die erwartete Länge von $1.60$ Bit beim Shannon-Code auf $1.40$ Bit ($H_2 = 1.295$).

<!--en:a0afcaaa-->
- **Mindestens 1 Bit pro Symbol:** Für jedes Symbolcodewort gilt $\ell_i \ge 1$. Daher kommt *kein Symbolcode* unter $1.00$ Bit/Symbol, selbst wenn die Quelle mit $P(\texttt{a}) = 0.9$ eine Entropie von $H_2(P) = 0.469$ Bit hat.

<!--en:9f056d36-->
::: {.slide title="Codes für abweichende Verteilungen: Kreuzentropie als Dateigröße"}

<!--en:21aa5e6d-->
[Kodierung mit einem Modell Q]{.kicker}

<!--en:809fc5c0-->
Der Shannon-Code $\ell_i = \lceil \log_2(1/q_i) \rceil$ stammt aus einem Modell $Q$ und kodiert Daten aus $P$. Dann begrenzt die Kreuzentropie $\textrm{CE}_2(P, Q) = H_2(P) + D_{\textrm{KL}}^{(2)}(P \| Q)$ die erwartete Codelänge wie folgt:

<!--en:0fe7bec0-->
::: {.slide title="Arithmetische Kodierung: ein Intervall pro Nachricht"}

<!--en:76c9b6ec-->
[Kodierung von Symbolströmen]{.kicker}

<!--en:90fe86e4-->
Die **arithmetische Kodierung** erreicht die Effizienz der Blockkodierung ohne Nachschlagetabelle. Sie bildet eine ganze Nachricht aus $n$ Symbolen, $x_{1:n}$, beim Lesen auf ein Teilintervall von $[0, 1)$ der Breite $w = \prod_{i=1}^n q(x_i \mid x_{<i})$ ab:

<!--en:398c375a-->
- Im Schritt $i$ unterteilt der Encoder das aktuelle Intervall $[lo, hi)$ proportional zu $q(\cdot \mid x_{<i})$ und behält das Teilintervall des beobachteten Symbols $x_i$.

<!--en:e07c8192-->
::: {.slide title="Höchstens zwei zusätzliche Bits bei arithmetischer Kodierung"}

<!--en:ad3980ce-->
[Wahl des dyadischen Intervalls]{.kicker}

<!--en:be3a9948-->
Um ein dyadisches Intervall $[m 2^{-\ell}, (m+1)2^{-\ell})$ anzugeben, das vollständig in $[lo, hi)$ der Breite $w = \prod_{i=1}^n q(x_i \mid x_{<i})$ liegt, genügt die Wahl $\ell = \lceil\log_2(1/w)\rceil + 1$ ($2^{-\ell} \le w/2$):

<!--en:5b3e0420-->
- **Verteilte Rundungskosten:** Die Rundung erfolgt nur *einmal* für die gesamte Nachricht. Damit gilt für den Mehrbedarf pro Symbol $2/n \to 0$ für $n \to \infty$.

<!--en:b6ea2d63-->
::: {.slide title="Arithmetische Kodierung im Programm: 6 Bit gegenüber 13 Bit"}

<!--en:c8d0d138-->
[Live-Demonstration]{.kicker}

<!--en:d18a834c-->
Für die Nachricht `aaaaaaaaba` aus 10 Symbolen (Überraschungswert $4.69$ Bit unter $q(\texttt{a}) = 0.9$) liefert die arithmetische Kodierung das 6 Bit lange Codewort `011001`. Der symbolweise arbeitende Shannon-Code benötigt 13 Bit:

<!--en:ee2c364b-->
**Numerisches Problem (53-Bit-Mantisse):** Bei direkter Rechnung mit doppelter Genauigkeit nach IEEE 754 (`float64`) fällt `hi - lo` nach $\sim 100$ Symbolen auf null. Praktische Kodierer verwenden Ganzzahlregister fester Breite. Sie geben führende Bits aus, sobald `lo` und `hi` im höchstwertigen Bit übereinstimmen.

<!--en:8929d7fa-->
::: {.slide title="Jedes autoregressive Sprachmodell kann komprimieren"}

<!--en:12794386-->
[Adaptive und vortrainierte Modelle]{.kicker}

<!--en:0002b5ce-->
Die arithmetische Kodierung benötigt nur die Wahrscheinlichkeiten $q(\cdot \mid x_{<i})$ für das nächste Symbol. Damit wird *jedes* kausale Vorhersagemodell zum verlustfreien Kompressor, von der n-Gramm-Tabelle bis zum großen Transformer-Sprachmodell:

<!--en:a8a09d86-->
- **Adaptives Zählmodell der Ordnung $m$:** Sagt das Zeichen $x_i$ aus dem vorangehenden Kontext von $m$ Zeichen, $s = x_{i-m:i}$, vorher. Es glättet die laufenden Zähler durch Addition von $\tfrac12$ nach Krichevsky und Trofimov ($k = 27$): $$q(x \mid s) = \frac{c(s, x) + 1/2}{c(s) + k/2}$$

<!--en:da253bdd-->
- **Keine Übertragung von Parametern:** Die Zähler ändern sich erst *nach* der Kodierung jedes Zeichens. Der Decoder aktualisiert seine Zähler im Gleichschritt und benötigt keine vorab übertragene Tabelle.

<!--en:9243f885-->
::: {.slide title="Jedes autoregressive Sprachmodell kann komprimieren (Fortsetzung)"}

<!--en:12794386-2-->
[Adaptive und vortrainierte Modelle]{.kicker}

<!--en:288fbc5c-->
::: {.slide title="Kompression von The Time Machine: Wie viel Kontext lohnt sich?"}

<!--en:34ccdfa8-->
[Empirischer Vergleich der Kompression]{.kicker}

<!--en:1e616775-->
Der Vergleich adaptiver Modelle mit Kontextlängen $m \in \{0, 1, 2, 3, 4\}$ auf *The Time Machine* ($173{,}428$ Zeichen) zeigt den Zielkonflikt zwischen Verzerrung, Varianz und Dauer der Aufwärmphase:

<!--en:5b055b1d-->
- **Warum Ordnung 3 gewinnt ($2.55$ Bit/Zeichen gegenüber `zlib` mit $2.84$ Bit/Zeichen):** Drei Zeichen erfassen englische Morpheme und kommen oft genug vor, um rasch Häufigkeiten zu lernen. Bei Ordnung 4 ($27^4 \approx 530{,}000$ Kontexte) kostet das Lernen seltener Kontexte mehr Bits, als das vierte Zeichen spart.

<!--en:29e69773-->
::: {.slide title="Vortrainierte große Sprachmodelle als universelle Kompressoren"}

<!--en:991563eb-->
[Abwägung zwischen festem und adaptivem Modell]{.kicker}

<!--en:f3e3782d-->
- **Festes Sprachmodell mit 70 Milliarden Parametern und arithmetischem Kodierer (Delétang et al., 2023):** Komprimiert rohe ImageNet-Bild-Patches auf **$43.4\%$** (PNG: $58.5\%$) und LibriSpeech-Audiodaten auf **$16.4\%$** (FLAC: $30.3\%$).

<!--en:ae9fa077-->
- **Zusatzkosten beim zweiteiligen MDL-Code:** Fehlen dem Empfänger die $140\text{ GB}$ an Gewichten ($2$ Byte/Parameter), wächst ein Datensatz von $1\text{ GB}$ durch die Modellübertragung auf $>14{,}000\%$ seiner Größe.

<!--en:9fa1a219-->
::: {.slide title="Minimale Beschreibungslänge (MDL) und BIC"}

<!--en:b190758a-->
[Ockhams Rasiermesser in Bit]{.kicker}

<!--en:780a8e39-->
Das Prinzip der **minimalen Beschreibungslänge (MDL)** wählt das Modell $\mathcal{M}$ mit der kleinsten Gesamtdatei aus beiden Teilen:

<!--en:0952e315-->
- **BIC-Regel ($\tfrac12\log_2 n$ Bit/Parameter):** Die empirische Parameterunsicherheit skaliert mit $1/\sqrt{n}$. Ein Gitter mit $\sqrt{n}$ Punkten kostet daher $\approx 5$ Bit bei $n = 1{,}000$.

<!--en:8d8779eb-->
- **Vorteil der adaptiven präquenziellen Kodierung:** Kosten entstehen nur für Kontexte, die im Textstrom tatsächlich vorkommen.

<!--en:dbed82b7-->
::: {.slide title="Entropierate stationärer Quellen mit Gedächtnis"}

<!--en:1b5f9ef6-->
[Grundlegende Schranke für Sequenzen]{.kicker}

<!--en:2858e40b-->
Für einen stationären stochastischen Prozess $X_1, X_2, \ldots$ mit Gedächtnis ist die **Entropierate** die grundlegende untere Kompressionsschranke pro Symbol:

<!--en:daacc86c-->
- **Monotone Abnahme mit wachsendem Kontext:** Nach der Kettenregel $H(X_{1:n}) = \sum_{i=1}^n H(X_i \mid X_{<i})$ und weil Konditionieren die Entropie nie erhöht, fallen $H(X_n \mid X_{<n})$ und ihr Cesàro-Mittel $\frac{1}{n}H(X_{1:n})$ monoton gegen $H_{\textrm{rate}}$.

<!--en:1072edcf-->
- **Markov-Kette mit hoher Verweilwahrscheinlichkeit:** Die Kette $X_n \in \{\texttt{a}, \texttt{b}\}$ bleibt mit Wahrscheinlichkeit $0.9$ im aktuellen Zustand und wechselt mit Wahrscheinlichkeit $0.1$. Aus Symmetrie ist die Randverteilung gleichförmig ($H(X_1) = 1\text{ bit}$). Bei gegebenem $X_{n-1}$ beträgt die Übergangsentropie jedoch $H_{\textrm{rate}} = H(X_n \mid X_{n-1}) = H_2(0.9, 0.1) = 0.469\text{ bits/symbol}$.

<!--en:c1efbf3f-->
::: {.slide title="Die Entropierate erreichen und abschätzen"}

<!--en:df413568-->
[Kompressionsraten für englischen Text]{.kicker}

<!--en:da818ec5-->
- **Untere Schranke:** Kein eindeutig dekodierbarer Blockcode benötigt im Mittel weniger als $H_{\textrm{rate}}$ Bit/Symbol. Die arithmetische Kodierung mit dem wahren bedingten Modell erreicht $\frac{1}{n}H(X_{1:n}) + \frac{2}{n} \to H_{\textrm{rate}}$.

<!--en:5d175a70-->
- **Raten für Englisch ($\textrm{PPL} = 2^{\textrm{bpc}}$):** Von der Gleichverteilung mit $4.75\text{ bpc}$ ($\textrm{PPL} = 27$) sinkt die Rate über Unigramme mit $4.08\text{ bpc}$, `zlib` mit $2.84\text{ bpc}$, Ordnung 3 mit $2.55\text{ bpc}$ ($\textrm{PPL} = 5.84$) und ein neuronales Sprachmodell mit $1.73\text{ bpc}$ bis zu Shannons Schranken aus menschlichen Vorhersagen von $0.6\text{–}1.3\text{ bpc}$ ($\textrm{PPL} \approx 1.5\text{–}2.5$).

<!--en:2b047ccb-->
::: {.slide title="Die asymptotische Gleichverteilungseigenschaft (AEP)"}

<!--en:e40900fb-->
[Typische Sequenzen]{.kicker}

<!--en:e5ccdfce-->
Für unabhängige, identisch verteilte Ziehungen $X_1, \ldots, X_n \sim P$ mit binärer Entropie $H = H_2(P)$ folgt aus dem schwachen Gesetz der großen Zahlen $-\frac{1}{n}\log_2 P(X_{1:n}) = -\frac{1}{n}\sum_{i=1}^n \log_2 P(X_i) \xrightarrow{P} H$. Die **typische Menge** zur Toleranz $\delta > 0$ ist:

<!--en:bdf53575-->
- **1. Konzentration ($P(A_\delta) \to 1$):** Für große Blocklängen liegt fast die gesamte Wahrscheinlichkeitsmasse in $A_\delta$, wenn $n \to \infty$.

<!--en:bb0eafb6-->
- **2. Gleichverteilung der Wahrscheinlichkeit:** Jede typische Sequenz $x_{1:n} \in A_\delta$ hat eine Wahrscheinlichkeit im Bereich $2^{-n(H+\delta)} < P(x_{1:n}) < 2^{-n(H-\delta)}$.

<!--en:76f57354-->
- **3. Exponentielle Mächtigkeit ($|A_\delta| \approx 2^{nH}$):** Summieren der Wahrscheinlichkeitsschranken über $A_\delta$ ergibt $(1-\eta)2^{n(H-\delta)} \le |A_\delta| \le 2^{n(H+\delta)}$. Somit genügen zur Indizierung von $A_\delta$ insgesamt $n(H+\delta)$ Bit, während jeder Code fester Länge mit Rate $R < H$ mit Wahrscheinlichkeit $\to 1$ scheitert.

<!--en:4278d2c7-->
::: {.slide title="Typische Sequenzen und die wahrscheinlichste Folge"}

<!--en:8753fe86-->
[Geometrie der typischen Menge]{.kicker}

<!--en:2d164132-->
Für $n = 200$ unabhängige, identisch nach $\textrm{Bernoulli}(0.3)$ verteilte Bits ($H = 0.881\text{ bits}$, $nH = 176.3\text{ bits}$, $\delta = 0.1$) ergibt die exakte Summe über die Binomialverteilung die Größe der typischen Menge ($44 \le j \le 76$ Einsen):

<!--en:1e5a55c1-->
::: {.slide title="Wahrscheinlichkeitsmasse und Anzahl der Folgen"}

<!--en:f2c792d9-->
[Konzentration bei der AEP]{.kicker}

<!--en:2501cbeb-->
Warum liegt die typische Menge bei etwa $j = np = 60$ Einsen und nicht bei $j = 0$ oder $j = 100$?

<!--en:69b2c525-->
- **Wahrscheinlichste Folge ($j = 0$, nur Nullen):** Ihre Einzelwahrscheinlichkeit beträgt $0.7^{200} = 2^{-103}$. Es gibt aber nur $\binom{200}{0} = 1$ solche Folge, und sie ist untypisch ($0.515\text{ bits/symbol} \notin H \pm \delta$).

<!--en:8c6494e7-->
- **Größte Anzahl von Folgen ($j = 100$ Einsen):** Es gibt $\binom{200}{100} = 2^{195.9}$ Folgen mit der jeweils winzigen Wahrscheinlichkeit $2^{-225}$. Zusammen ergibt das $2^{-29}$.

<!--en:8b53fb2f-->
- **Typischer Bereich ($j \approx 60$):** Das Produkt $\binom{n}{j} p^j (1-p)^{n-j}$ erreicht sein Maximum dort, wo $2^{172.2}$ Folgen jeweils eine Wahrscheinlichkeit von $\approx 2^{-176.3}$ haben.

<!--en:00b96ff0-->
::: {.slide title="Perplexität als exponenzierte Codelänge"}

<!--en:577fed3b-->
[Bewertung von Sprachmodellen]{.kicker}

<!--en:4828e5e6-->
Bei Sprachmodellen gibt man den Kreuzentropieverlust $\textrm{CE} = -\frac{1}{n}\sum_{i=1}^n \log q(x_i \mid x_{<i})$ auf zurückgehaltenen Daten oft in exponentieller Form an, als **Perplexität**:

<!--en:ac23732f-->
- **Unabhängigkeit von der Basis:** Wegen $e^{\textrm{CE}} = 2^{\textrm{CE}/\ln 2} = b^{\textrm{CE}_b}$ bleibt die Perplexität gleich, ob der Verlust in Nat oder in Bit angegeben ist.

<!--en:9f63982f-->
- **Effektiver Verzweigungsfaktor:** Verteilt ein Modell die Wahrscheinlichkeit bei jedem Schritt gleichmäßig auf $k$ plausible nächste Token, gelten $\textrm{CE}_2 = \log_2 k$ Bit und $\textrm{PPL} = k$. Für eine perfekte deterministische Vorhersage gilt $\textrm{PPL} = 1$.

<!--en:f398a658-->
- **Interpretation für *The Time Machine*:** Mit Kontext der Ordnung 3 sinkt die Unsicherheit über das nächste Zeichen von $27$ Möglichkeiten auf einen effektiven Verzweigungsfaktor von $5.84$ (etwa ein Wurf mit einem sechsseitigen Würfel pro Zeichen).

<!--en:d33fea00-->
::: {.slide title="Untere Schranke der erwarteten Perplexität: 2 hoch Entropierate"}

<!--en:40fb7551-->
[Herleitung mit Gibbs- und Jensen-Ungleichung]{.kicker}

<!--en:f9390466-->
Wie klein kann die erwartete Perplexität für Sequenzen aus einer stationären Quelle mit Entropierate $H_{\textrm{rate}}$ Bit/Token werden?

<!--en:0096fb46-->
**Proposition (untere Perplexitätsschranke):** Für jede stationäre Quelle $P$ und jedes autoregressive Modell $q$ gelten $\mathbb{E}[\textrm{CE}] \ge H_{\textrm{rate}} \ln 2$ und $\mathbb{E}[\textrm{PPL}] = \mathbb{E}[e^{\textrm{CE}}] \ge 2^{H_{\textrm{rate}}}$.

<!--en:43fbbe6b-->
- **Schritt 1 (punktweise Gibbs-Ungleichung):** Bei gegebenem Präfix $x_{<i}$ ist der erwartete Modellverlust mindestens so groß wie die wahre bedingte Entropie: $\mathbb{E}[-\log q(X_i \mid X_{<i})] \ge \mathbb{E}[-\log p(X_i \mid X_{<i})]$.

<!--en:72cc6028-->
- **Schritt 2 (Kettenregel und Stationarität):** Mitteln über $i = 1, \ldots, n$ ergibt $\mathbb{E}[\textrm{CE}] \ge \frac{\ln 2}{n}H(X_{1:n}) \ge H_{\textrm{rate}} \ln 2$.

<!--en:4276a7d9-->
- **Schritt 3 (Jensen-Ungleichung für die konvexe Funktion $e^x$):** $\mathbb{E}[\textrm{PPL}] = \mathbb{E}[e^{\textrm{CE}}] \ge e^{\mathbb{E}[\textrm{CE}]} \ge e^{H_{\textrm{rate}}\ln 2} = 2^{H_{\textrm{rate}}}$. $\blacksquare$ *(Die Schranke gilt für den Erwartungswert $\mathbb{E}[\textrm{PPL}]$. Eine Folge aus 10 Zeichen enthält mit etwas Glück nur $\texttt{a}$. Unter $P(\texttt{a})=0.9$ liegt ihre Perplexität dann unter $2^H = 1.38$. Das geschieht mit Wahrscheinlichkeit $0.9^{10} = 35\%$.)*

<!--en:6fb1241a-->
::: {.slide title="Vergleich der Perplexität bei verschiedenen Tokenizern"}

<!--en:e07cd0b7-->
[Einfluss der Tokengröße]{.kicker}

<!--en:0f533f6e-->
Die Perplexität ist *pro Token* definiert. Modelle mit verschiedenen Vokabularen sind daher anhand von **Bit pro Zeichen (bpc)** oder **Bit pro Byte (bpb)** zu vergleichen:

<!--en:0c22ecff-->
- **Grobe Subwörter und feine Zeichen ($10^6$ Zeichen):** Modell A ($2 \times 10^5$ Token, $\textrm{PPL} = 32$) erreicht **$1.00\text{ bpc}$** und unterbietet Modell B ($5 \times 10^5$ Token, $\textrm{PPL} = 8$, **$1.50\text{ bpc}$**) um $33\%$, trotz einer um den Faktor $4\times$ höheren Perplexität pro Token.

<!--en:f8bc10e9-->
::: {.slide title="Verständnisfragen zu Präfixcodes, AEP und Perplexität"}

<!--en:929b7671-->
[Wissenskontrolle]{.kicker}

<!--en:2c6482d1-->
Fragen zur Quellenkodierung und zur Bewertung von Sprachmodellen:

<!--en:926f8480-->
1. **Shannon- und Huffman-Längen:** Welche Codewortlängen und welche erwartete Codelänge ergeben sich für $P = (0.6, 0.3, 0.1)$ ($H_2(P) = 1.295$ Bit) nach Shannons Regel und nach dem Huffman-Algorithmus?

<!--en:08481864-->
2. **Indizierung der typischen Menge und arithmetische Kodierung:** Für $n = 200$ Ziehungen aus $\textrm{Bernoulli}(0.3)$ ($nH = 176.3$ Bit) kostet ein Index für die typische Menge zur Toleranz $\delta = 0.1$ ($2^{188.8}$ Folgen) insgesamt $189$ Bit. Welche erwartete Länge und welche Fehlerquote hat die arithmetische Kodierung im Vergleich dazu?

<!--en:c24a41ca-->
3. **Gleichverteilung als Referenz:** Ein frisch initialisiertes großes Sprachmodell weist jedem Token seines Vokabulars dieselbe Wahrscheinlichkeit $1/50{,}000$ zu. Wie groß sind seine Perplexität und sein Verlust in Bit pro Token sowie in Nat pro Token?

<!--en:557b98b6-->
**Lösungen:**

<!--en:2ab0d8b7-->
1. **Symbolcodes:** Shannon rundet die Überraschungswerte $(0.74, 1.74, 3.32)$ auf $(1, 2, 4)$ auf. Das ergibt $\mathbb{E}[\ell] = 1.60\text{ bits}$ ($Z = 0.8125$). Huffman fasst die beiden seltensten Symbole zusammen und erhält die Längen $(1, 2, 2)$ mit $\mathbb{E}[\ell] = 1.40\text{ bits}$ ($Z = 1$).

<!--en:a976e473-->
2. **AEP und arithmetische Kodierung:** Für die Indizierung von $A_{0.1}$ benötigt der Code $189\text{ bits}$ ($12.7$ Bit mehr als $nH$). Sie scheitert bei den $1.1\%$ der Nachrichten außerhalb von $A_{0.1}$. Die arithmetische Kodierung ist für *alle* Nachrichten verlustfrei und benötigt im Mittel weniger als $nH + 2 = 178.3\text{ bits}$.

<!--en:1859996f-->
3. **Sprachmodell mit Gleichverteilung:** $\textrm{PPL} = 50{,}000$ (der Verzweigungsfaktor). Der Verlust beträgt $\log_2 50{,}000 \approx 15.61\text{ bits/token}$ oder $\ln 50{,}000 \approx 10.82\text{ nats/token}$.

<!--en:fa34cf5b-->
::: {.slide title="Zusammenfassung: Quellenkodierung und Kompressionsmaße"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:8dfa436a-->
| Begriff / Verfahren | Formel / Schranke | Mehrbedarf pro Symbol | Bedeutung für Deep Learning und Kompression |

<!--en:afb4b1b8-->
| **Kraft-Ungleichung** | $\sum_{i=1}^k 2^{-\ell_i} \le 1$ | Exaktes Budget für den Baum bzw. die dyadischen Intervalle | Notwendig und hinreichend für die Existenz präfixfreier bzw. eindeutig dekodierbarer Codes mit gegebenen Längen |

<!--en:e49b6b9d-->
| **Shannon-Symbolcode** | $\ell_i = \lceil \log_2(1/q_i) \rceil$ | $\textrm{CE}_2(P, Q) \le \mathbb{E}_P[\ell] < \textrm{CE}_2 + 1$ | Verknüpft Kreuzentropie und KL-Divergenz mit der Codewortlänge |

<!--en:ee889380-->
| **Arithmetische Kodierung** | Teilintervall der Breite $w = \prod_i q(x_i \mid x_{<i})$ | $< \frac{2}{n}$ Bit/Symbol (insgesamt $\le 2$ Bit) | Macht jedes autoregressive große Sprachmodell zum universellen Kompressor |

<!--en:4be452fc-->
| **MDL-Prinzip** | $\min_{\mathcal{M}}[L(\mathcal{M}) + L(\mathcal{D} \mid \mathcal{M})]$ | $\approx \frac{1}{2}\log_2 n$ Bit pro Parameter (BIC) | Bewertet Überanpassung durch zu viel Kontext oder zu viele Parameter direkt in Bit |

<!--en:3d1e9406-->
| **Entropierate** | $H_{\textrm{rate}} = \lim_{n\to\infty} H(X_n \mid X_{<n})$ | Untere Schranke für stationäre Sequenzen | Grundlegende Kompressionsgrenze für Sprache ($\approx 1$ Bit/Zeichen) |

<!--en:d05d501a-->
| **Perplexität und Bit pro Zeichen** | $\textrm{PPL} = e^{\textrm{CE}} = 2^{\textrm{CE}_2}$, $\textrm{bpc} = \frac{n_{\textrm{tok}}}{n_{\textrm{char}}}\log_2 \textrm{PPL}$ | $\mathbb{E}[\textrm{PPL}] \ge 2^{H_{\textrm{rate}}}$ | Bewertet große Sprachmodelle. $\textrm{bpc}$ erlaubt den Vergleich über Tokenizer hinweg |
