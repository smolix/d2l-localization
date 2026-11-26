<!--en:19cb6299-->
# Numerische Stabilität und Kondition

<!--en:93276587-->
Bislang haben wir in diesem Kapitel in Gleitkommaarithmetik (engl. floating-point arithmetic) gerechnet und die Ergebnisse wie exakte reelle Zahlen gelesen. Meist trägt diese bequeme Annahme. Gelegentlich wird jedoch der Trainingsverlust zu `NaN`, oder ein Verfahren für lineare Gleichungssysteme liefert ein scheinbar plausibles Ergebnis, das weit danebenliegt. Wann bestimmt die endliche Rechengenauigkeit das Ergebnis statt des Optimierungsalgorithmus, und was können wir dagegen tun?

<!--en:a75f84f1-->
Zwei kleine Beispiele zeigen, wie unterschiedlich numerische Probleme zutage treten. Erstens ergeben die drei Logits $(1000, 1001, 1002)$ mit der Softmax-Funktion ganz gewöhnliche Wahrscheinlichkeiten von $(0.09, 0.24, 0.67)$. Werten wir jedoch die Lehrbuchformel direkt aus, erhalten wir dreimal `NaN`, selbst bei doppelter Genauigkeit (fp64). Zweitens liefert `np.linalg.solve` für ein lineares $12\times12$-Gleichungssystem eine Lösung, die um 30% abweicht. Dabei löst sie exakt ein System, das sich erst in der siebzehnten Dezimalstelle vom gegebenen unterscheidet. Im ersten Fall bricht das Training mit dem Verlust `NaN` ab. Der zweite Fehler bleibt unbemerkt.

<!--en:0a0b2a1b-->
Die Ursachen sind grundverschieden. Die naive Formel für die Softmax-Funktion berechnet als Zwischenwert $e^{1000}$, eine Zahl, die für jedes gängige Gleitkommaformat zu groß ist. Eine algebraisch äquivalente Formel kommt ohne $e^{1000}$ aus und ist bis auf wenige Einheiten der letzten Stelle genau. Das lineare $12\times12$-Gleichungssystem hingegen ist so empfindlich, dass ein Rundungsfehler in der sechzehnten Stelle der Daten bereits die erste Stelle der Lösung verändert. Daran ändert auch eine Umformulierung des Lösungsverfahrens nichts.

<!--en:c6fc45f0-->
Um die beiden Fälle zu unterscheiden, untersuchen wir zunächst, wie Computer reelle Zahlen speichern und wie groß der Fehler bei jeder Rundung ist. Dann formen wir die Softmax-Funktion und die Kreuzentropie so um, dass alle Zwischenwerte im darstellbaren Bereich liegen. Die Varianz schreiben wir so, dass wir keine fast gleich großen Zahlen mit hohem Betrag subtrahieren müssen. Als Maß für die Empfindlichkeit führen wir die Konditionszahl (engl. condition number) einer Funktion ein. Mit ihr lässt sich ein schlechter Algorithmus von einem grundsätzlich schwierigen Problem unterscheiden. Schließlich sehen wir, wie die Konditionszahl eines linearen Gleichungssystems bestimmt, wie viele Stellen bei der Methode der kleinsten Quadrate erhalten bleiben und wie viele Schritte der Gradientenabstieg benötigt (siehe auch :numref:`sec_mdl-gradient-based-optimization`). Während :numref:`sec_numerics` den praktischen Einsatz von Formaten geringer Genauigkeit im heutigen Training zeigt, leiten wir hier her, warum Rechnungen in diesen Formaten gelingen oder scheitern.

<!--en:f908a5f0-->
Drei Ideen sind wichtiger als jede einzelne Formel. Die erste ist das relative Rundungsmodell der Gleitkommaarithmetik. Die zweite besagt, dass ein instabiler Algorithmus ein gut konditioniertes Problem über einen schlecht konditionierten Zwischenschritt löst. Die dritte ist eine Schranke, die den Fehler der berechneten Lösung mit der Empfindlichkeit des Problems verknüpft. Die Unterabschnitte zur Summationsreihenfolge und zur Ridge-Regularisierung können Sie beim ersten Lesen überspringen.

<!--en:57f84458-->
## Gleitkommaarithmetik

<!--en:b0a7f44b-->
### Darstellung und Abstand

<!--en:d58287d2-->
Was speichert ein Computer, wenn wir die Zahl $1.1$ speichern wollen? Das Prinzip lässt sich an einem einfachen Binärformat mit nur zwei Bits nach dem Komma ablesen. Zwischen $1$ und $2$ kann dieses Format nur vier Werte darstellen: $(1.00)_2=1$, $(1.01)_2=1.25$, $(1.10)_2=1.5$ und $(1.11)_2=1.75$. Multiplizieren wir diese Werte mit Zweierpotenzen, erhalten wir die übrigen Intervalle: $2, 2.5, 3, 3.5$ zwischen $2$ und $4$ sowie $0.5, 0.625, 0.75, 0.875$ zwischen $0.5$ und $1$. Jede *Oktave*, also jedes Intervall $[2^e, 2^{e+1})$ zwischen aufeinanderfolgenden Zweierpotenzen, enthält genau vier Werte. Verdoppelt sich der Betrag, verdoppelt sich daher auch der absolute Abstand benachbarter Zahlen.

<!--en:5990a4c7-->
Jede reelle Zahl zwischen zwei Gitterpunkten wird auf den nächstgelegenen gerundet: $1.1\mapsto1$ (relativer Fehler $-0.091$), $1.3\mapsto1.25$ (Fehler $-0.038$) und $3.3\mapsto3.5$ (Fehler $+0.061$). Liegt eine Zahl genau in der Mitte, rundet die Standardarithmetik zum Nachbarn mit dem letzten Bit $0$. Der Mittelpunkt $1.125$ wird also auf $1$ gerundet, mit einem Fehler von $-0.111$. Kein relativer Fehler übersteigt $1/8$. Die Rundung verschiebt eine Zahl höchstens um einen halben Gitterabstand, und der beträgt in jeder Oktave $1/8$ ihres kleinsten Werts. :numref:`fig_mdl-opt-fp-number-line` zeigt dieses Gitter auf der Zahlengeraden.

<!--en:920b2144-->
![Gleitkommazahlen auf der Zahlengeraden in einem Beispielformat mit zwei Mantissenbits (vier Werte je Oktave, $\varepsilon_{\mathrm{mach}}=1/4$, kleinste normale Zahl $0.25$). Bei jeder Zweierpotenz verdoppelt sich der absolute Abstand benachbarter Zahlen, während der relative Abstand nahe bei $\varepsilon_{\mathrm{mach}}$ bleibt. Unterhalb der kleinsten normalen Zahl setzen subnormale Zahlen das Gitter mit gleichem Abstand bis null fort. Hinter der Achsenunterbrechung enden reale Formate an ihrer Überlaufschwelle (fp16 bei $65504$, fp32 bei etwa $3.4\times10^{38}$).](../img/mdl-opt-fp-number-line.svg)

<!--en:963c18da-->
Gängige Gleitkommaformate funktionieren genauso, haben aber mehr Bits. Eine Gleitkommazahl hat die wissenschaftliche Darstellung zur Basis zwei mit einer festen Anzahl von $p$ Binärstellen nach dem Komma,

<!--en:3312f236-->
wobei ein Vorzeichenbit $s$, die Bits der *Mantisse* (auch Signifikand genannt) $b_1,\ldots,b_p$ und ein ganzzahliger Exponent $e$ aus einem festen Bereich die Zahl bestimmen. Der Exponent bestimmt den darstellbaren Größenbereich, die Mantisse die relative Genauigkeit. Innerhalb der Oktave $[2^e, 2^{e+1})$ haben benachbarte Werte den Abstand $2^{e-p}$. Der Abstand von $1$ zur nächstgrößeren darstellbaren Zahl, $\varepsilon_{\mathrm{mach}}=2^{-p}$, heißt *Maschinenepsilon* (engl. machine epsilon). In unserem Beispielformat gelten $p=2$ und $\varepsilon_{\mathrm{mach}}=1/4$.

<!--en:946ccf0c-->
Runden wir eine reelle Zahl $x$ auf den nächstgelegenen darstellbaren Wert, ändert sie sich höchstens um einen halben Gitterabstand, also um $2^{e-p-1}$. Wegen $|x|\ge2^e$ in der gesamten Oktave beträgt der relative Fehler höchstens $2^{-p-1}$. Wir nennen einen Wert *normal*, wenn er die Form :eqref:`eq_mdl-opt-float-format` mit einem Exponenten im zulässigen Bereich hat. Für jede reelle Zahl $x$, die auf eine normale Zahl gerundet wird, gilt

<!--en:31669e4b-->
Jede Rundung verändert eine Zahl also um höchstens einen Teil von $2^{p+1}$ Teilen: einen von $8$ in unserem Beispielformat, etwa einen von 17 Millionen in fp32 ($p=23$) und einen von $256$ in bf16 ($p=7$). Die Schranke $u$ heißt *Rundungseinheit* (engl. unit roundoff). Die IEEE-Standardarithmetik rundet das Ergebnis jeder Grundrechenart auf den nächstgelegenen darstellbaren Wert. Die berechnete Summe $x\oplus y$ ist daher gleich $(x+y)(1+\delta)$ mit $|\delta|\le u$. Dieselbe Garantie gilt für $-$, $\times$ und $/$ :cite:`IEEE.754.2019,Goldberg.1991`.

<!--en:b6e3c76c-->
Dieses einfache relative Modell hat drei Ausnahmen. Übersteigt ein Ergebnis den darstellbaren Bereich, kommt es zum *Überlauf* (engl. overflow) auf $\infty$. Unterhalb der kleinsten normalen Zahl kommt es zum *Unterlauf* (engl. underflow). Dort füllen *subnormale* Zahlen die verbleibende Lücke bis null. Undefinierte Operationen wie $0/0$ oder $\infty/\infty$ ergeben `NaN`. Alle drei Effekte hängen von der absoluten Größe ab. Wir kommen darauf zurück, sobald wir die Standardformate kennengelernt haben.

<!--en:85b851a3-->
### Formate

<!--en:b71f496e-->
Im Deep Learning kommen fünf Formate zum Einsatz, die weniger Bits als das Format mit doppelter Genauigkeit (fp64) benötigen. Lesen wir für jedes Format das Maschinenepsilon $\varepsilon_{\mathrm{mach}}$, die kleinste normale Zahl und den größten endlichen Wert direkt aus der Bibliothek aus:

<!--en:012e7947-->
**fp32** hat $p=23$ Mantissenbits (etwa sieben Dezimalstellen) und reicht bis $3.4\times10^{38}$. **fp16** hat $p=10$ Bits (etwa drei Dezimalstellen), aber einen engen Exponentenbereich. Oberhalb von $65504$ tritt ein Überlauf auf, unterhalb von $2^{-14}\approx6.1\times10^{-5}$ sind die Werte subnormal. **bf16** (in den Bibliotheken `bfloat16`) setzt den Schwerpunkt anders. Es behält alle acht Exponentenbits von fp32 und damit dessen großen dynamischen Bereich, hat dafür aber nur $p=7$ Mantissenbits ($\varepsilon_{\mathrm{mach}}=2^{-7}\approx
0.0078$, etwa zwei bis drei Dezimalstellen). Der für bf16 oft angegebene Wert $2^{-8}$ ist die Rundungseinheit $u=\varepsilon_{\mathrm{mach}}/2$. Die beiden fp8-Formate treiben diese Abwägung weiter: E4M3 hat drei Mantissenbits (etwa eine Dezimalstelle) und reicht bis $448$. E5M2 hat zwei Mantissenbits und den Exponentenbereich von fp16. Bei einem so kleinen dynamischen Bereich ist fp8 nur mit einem eigenen Skalierungsfaktor pro Tensor oder Werteblock praktikabel.

<!--en:61a367c8-->
bf16 besteht aus den oberen 16 Bits einer fp32-Zahl: demselben Vorzeichenbit, denselben acht Exponentenbits und den ersten sieben Mantissenbits. Zur Umwandlung von fp32 in bf16 runden wir also die unteren 16 Bits weg. Das lässt sich mit Ganzzahlarithmetik direkt auf dem Bitmuster umsetzen. Die Mitte des Bereichs der verworfenen 16 Bits liegt bei `0x8000`. Wir addieren daher `0x7FFF` (eins unterhalb der Mitte) und zusätzlich eins, falls das niedrigste beibehaltene Bit $1$ ist. Dann löschen wir die unteren 16 Bits. Genau dann entsteht ein Übertrag in die beibehaltenen Bits, wenn der verworfene Teil oberhalb der Mitte liegt oder genau in der Mitte liegt und das letzte beibehaltene Bit eins ist. So runden wir in einer einzigen Zeile durch Bitoperationen zum nächsten Wert und bei Gleichstand zum geraden Nachbarn. Darauf baut die stochastische Rundung in Aufgabe :exref:`ex_mdl-numerical-stability-conditioning-stochastic-rounding` auf. Prüfen wir das Verfahren an einer Eingabe oberhalb eines Mittelpunkts und zwei Eingaben genau in der Mitte:

<!--en:9abf08b0-->
Wie erwartet liegt die erste Eingabe oberhalb der Mitte zwischen $1$ und $1+2^{-7}$ und wird aufgerundet. Die zweite liegt genau in dieser Mitte und wird zum geraden Nachbarn $1$ abgerundet. Die dritte liegt in der Mitte zwischen $1+2^{-7}$ und $1+2^{-6}$ und wird zum geraden Nachbarn $1+2^{-6}$ aufgerundet. Prüfen wir dieselbe Rundungsregel in fp32. Beobachten wir auch, wie der Abstand mit dem Betrag wächst und für welchen größten Logit sich die Exponentialfunktion in jedem Format ohne Überlauf auswerten lässt:

<!--en:51acfb70-->
Hier liegt $1+\varepsilon_{\mathrm{mach}}/2$ genau zwischen $1$ und seinem oberen Nachbarn. Die Rundung zum geraden Nachbarn ergibt also wieder $1$. Der größtmögliche Rundungsfehler von einem halben Gitterabstand tritt tatsächlich auf. Der relative Fehler beträgt $u/(1+u)$ und liegt knapp innerhalb der Schranke $|\delta|\le u$. Bei $2^{20}$ beträgt der Abstand das $2^{20}$-Fache (etwa das Millionenfache) des Abstands bei $1$. Die letzten drei Zeilen zeigen die Überlaufschwelle für $e^x$: oberhalb von $11.09$ in fp16, $88.72$ in fp32 und $709.78$ in fp64.

<!--en:1ee0eae1-->
### Überlauf, Unterlauf und kleine Aktualisierungen

<!--en:eb8be9e2-->
Da die Exponentialfunktion eine additive Skala in eine multiplikative überführt, können schon mäßig große Logits zum Überlauf führen. In fp16 liegt die Grenze $\log 65504\approx11.09$ mitten im üblichen Wertebereich der unnormierten Scores eines Klassifikators.

<!--en:5ea23b76-->
Am anderen Ende kommt es bei $e^{-x}$ zum Unterlauf. Unterhalb der kleinsten normalen Zahl folgen subnormale Zahlen mit konstantem absolutem Abstand. Dabei geht ein signifikantes Bit nach dem anderen verloren, bis die Werte auf $0$ gerundet werden. Berechnen wir später $\log 0$, erhalten wir $-\infty$, und im Rückwärtsdurchlauf treten überall `inf` oder `NaN` auf. In fp16 beginnen die subnormalen Zahlen unterhalb von $2^{-14}\approx6.1\times10^{-5}$. Die kleinste subnormale Zahl ist $2^{-24}\approx6\times10^{-8}$, und alles unterhalb von $2^{-25}\approx3\times10^{-8}$ wird direkt auf null gerundet.

<!--en:0f82036a-->
Die Addition bringt eine zweite absolute Schwelle mit sich, die sich direkt auf das Training auswirkt. Ein Gradientenschritt ersetzt ein Gewicht $w$ durch $w-\eta g$. Ist die Aktualisierung $\eta g$ klein im Vergleich zum Gitterabstand bei $w$, wird die Summe wieder auf $w$ gerundet. Der Schritt verschwindet spurlos.

<!--en:3fb63396-->
**Proposition (kleine Aktualisierungen verschwinden).** *Sei $w>0$ eine Gleitkommazahl mit $2^e\le w<2^{e+1}$ in einem Format mit $p$ Mantissenbits, und sei $\Delta$ reell. Aus $|\Delta|<2^{e-p-1}$ folgt $\mathrm{fl}(w+\Delta)=w$, außer wenn $w=2^e$ und $\Delta<0$ gelten. In diesem Fall gilt dieselbe Aussage für $|\Delta|<2^{e-p-2}$.*

<!--en:6aae5b88-->
**Beweis.** In $[2^e, 2^{e+1}]$ haben die darstellbaren Zahlen den Abstand $2^{e-p}$. Die Nachbarn von $w$ sind daher $w\pm2^{e-p}$. Eine Ausnahme ist die Unterseite von $w=2^e$: Dort beginnt die tiefere Oktave $[2^{e-1},2^e)$, und der Abstand halbiert sich auf $2^{e-p-1}$. Rundung zur nächsten Zahl ergibt $w$, wenn $w+\Delta$ näher bei $w$ liegt als beim benachbarten Gitterpunkt auf derselben Seite. Das gilt genau dann, wenn $|\Delta|$ kleiner als die Hälfte dieses Abstands ist. $\blacksquare$

<!--en:3c50f933-->
Für eine Gewichtsaktualisierung $\Delta=-\eta g$ mit $2^e<w<2^{e+1}$ besagt die Proposition:

<!--en:6d4aca2a-->
Jede Aktualisierung, die kleiner als etwa $u|w|$ ist, geht also verloren. Das relative Modell :eqref:`eq_mdl-opt-rounding-model` beschreibt Produkte und Quotienten gut. Addieren wir jedoch Zahlen mit sehr unterschiedlichen Beträgen, bleibt diese scharfe absolute Schwelle im Modell verborgen.

<!--en:4cb6d3c5-->
Ein Gewicht genau auf einer Zweierpotenz wie $w=1$ fällt unter die Ausnahme der Proposition. Unterhalb von $1$ halbiert sich der Abstand. Eine Erhöhung verschwindet daher unterhalb von $u$ (wie in :numref:`sec_numerics` überprüft), eine Verringerung dagegen unterhalb von $u/2$. In bf16 bleibt ein Gewicht von $1$ also bei jeder Verringerung um weniger als $2^{-9}\approx0.002$ unverändert. Bei einer Lernrate von $10^{-3}$ und einem Gradienten der Größenordnung $1$ bewegt es sich überhaupt nicht. Die entsprechende Schwelle liegt in fp16 bei $2^{-12}\approx2.4\times10^{-4}$ und in fp32 bei $2^{-25}\approx3\times10^{-8}$. Deshalb hält das Training mit gemischter Genauigkeit eine Hauptkopie der Gewichte in fp32. Testen wir Schritte auf beiden Seiten der Schwellen. Dabei berechnen wir die bf16-Aktualisierungen in fp32 und runden mit `to_bf16`:

<!--en:5cdce19b-->
Tatsächlich bleibt das bf16-Gewicht bei einem Schritt von $10^{-3}$ auf $1$ stehen. Ein Schritt von $2\times10^{-3}$ überwindet dagegen die Schwelle von $0.00195$ und führt zum Nachbarn $1-2^{-8}=0.99609375$. Ebenso verschluckt fp16 einen Schritt von $10^{-4}$, während $3\times10^{-4}$ das Gewicht auf $1-2^{-11}\approx0.9995$ verschiebt. fp32 behält die volle Aktualisierung bei.

<!--en:04cbc3df-->
Bei Gradienten ist dagegen die Unterlaufschwelle entscheidend. In fp16 verlieren Gradienten unterhalb von $6.1\times10^{-5}$ an Genauigkeit, und unterhalb von $3\times10^{-8}$ verschwinden sie ganz. Multiplizieren wir vor dem Rückwärtsdurchlauf den Verlust mit einem konstanten Faktor $S$, wachsen alle Gradienten um den Faktor $S$ und gelangen so wieder in den normalen Bereich. Teilen wir danach in fp32 durch $S$, erhalten wir die ursprüngliche Skala zurück. Den praktischen Ablauf behandelt :numref:`subsec_loss_scaling`. In Aufgabe :exref:`ex_mdl-numerical-stability-conditioning-loss-scale-window` bestimmen Sie den zulässigen Bereich für $S$.

<!--en:71c6493a-->
Die *stochastische Rundung* (engl. stochastic rounding) :cite:`Gupta.Agrawal.Gopalakrishnan.ea.2015` beseitigt die Schwelle für verlorene Aktualisierungen auf andere Weise. Sie rundet zufällig auf oder ab. Dabei sind die Wahrscheinlichkeiten so gewählt, dass der Erwartungswert der gespeicherten Zahl dem exakten Ergebnis entspricht. So summieren sich über viele Schritte im Erwartungswert selbst Aktualisierungen, die für die Rundung zum nächsten Gitterpunkt viel zu klein sind (Aufgabe :exref:`ex_mdl-numerical-stability-conditioning-stochastic-rounding`).

<!--en:3e18a123-->
## Stabile Berechnung von Softmax-Funktion und Kreuzentropie

<!--en:bfc0b8aa-->
Softmax-Funktion, Log-Sum-Exp und Kreuzentropie wenden die Exponentialfunktion auf Logits an. Alle drei stoßen daher auf die gerade berechneten Überlauf- und Unterlaufschwellen. Ein Beispiel begleitet uns durch diesen Unterabschnitt: ein Klassifikator mit zwei Klassen und den Logits $\mathbf z=(0,t)$, bei dem die unwahrscheinliche Klasse $0$ das wahre Label ist. Seine Wahrscheinlichkeiten sind $\mathrm{softmax}(\mathbf z)=\bigl(1/(1+e^t),\, e^t/(1+e^t)\bigr)$, sein Log-Sum-Exp ist $\log(1+e^t)$, und der Kreuzentropieverlust für Klasse $0$ beträgt $\log(1+e^t)\approx t$ für große $t$. Keines dieser Ergebnisse liegt außerhalb des üblichen Wertebereichs: Selbst bei $t=100$ sind die Wahrscheinlichkeiten $3.7\times10^{-44}$ und $1$, und der Verlust beträgt nur $100$.

<!--en:63c7097f-->
### Verschiebung um das Maximum

<!--en:04c1aa93-->
Für gegebene Logits $\mathbf z\in\mathbb R^n$ ist der Vektor der Softmax-Funktion definiert durch

<!--en:693f8d27-->
Bei $t=100$ entsteht bei der direkten Auswertung dieser Formel der Zwischenwert $e^{100}$. Er übersteigt die Überlaufschwelle von fp32, die bei $88.72$ liegt. Der zweite Eintrag wird zu `inf/inf` (`NaN`), der erste fällt auf $1/\infty=0$. Die tatsächlichen Wahrscheinlichkeiten lassen sich in fp32 problemlos darstellen. Nur der Zwischenwert der Exponentialfunktion ist zu groß.

<!--en:a67ec9f0-->
Multiplizieren wir Zähler und Nenner mit derselben positiven Konstanten $e^{-c}$, bleibt der Quotient gleich. Die Softmax-Funktion ist also *verschiebungsinvariant*:

<!--en:9386abc8-->
Klammern wir $e^c$ aus der Summe unter dem Logarithmus aus, erhalten wir die entsprechende Identität für **Log-Sum-Exp**, $\mathrm{lse}(\mathbf{z}) = \log \sum_j e^{z_j}$, den Logarithmus des Nenners der Softmax-Funktion:

<!--en:4722c154-->
Beide Identitäten gelten für jedes $c\in\mathbb{R}$. Mit der Wahl $c=\max_j z_j$ sind beide Rechnungen sicher. Jeder verschobene Exponent $z_j-c$ ist dann höchstens $0$, sodass kein Exponentialterm überlaufen kann. Zugleich ist der größte Term gleich $e^0=1$. Die Summe liegt also stets in $[1,n]$, und ihr Logarithmus kann durch Unterlauf nicht zu $-\infty$ werden. Testen wir das für $(0, 100)$ in fp32, für $(1000, 1001, 1002)$ aus der Einleitung in fp64 und für $(0, 10)$, wo beide Varianten endlich bleiben. Liefern sie im letzten Fall exakt dieselben Bits?

<!--en:c69f462c-->
Bei $t=100$ ergibt die verschobene Softmax-Funktion $(3.8\times10^{-44}, 1)$. Der wahre Wert $3.7\times10^{-44}$ ist dabei auf das subnormale Gitter gerundet, fünf signifikante Bits bleiben erhalten. Die direkte Formel liefert dagegen $0$ und `NaN`. Höhere Genauigkeit löst das Problem nicht: Für die Logits $(1000, 1001, 1002)$ aus der Einleitung läuft $e^{1000}$ selbst in fp64 über. Die direkte Formel liefert dreimal `NaN`, die verschobene dagegen wie erwartet $(0.09, 0.24, 0.67)$. Bei $t=10$ stimmen beide Formeln bis aufs letzte Bit überein oder unterscheiden sich um eine oder zwei Einheiten der letzten Stelle (*ulps*, engl. units in the last place). Identitäten für reelle Zahlen garantieren keine identische Rundungsreihenfolge. Ein Grenzfall bleibt: Bei nicht endlichen Logits hilft auch die Verschiebung um das Maximum nicht. Besteht eine ganze Zeile aus $-\infty$, gilt $c=-\infty$, und $z_j-c$ ist undefiniert.

<!--en:a4fdf6c5-->
### Schranken für Log-Sum-Exp

<!--en:dc518a84-->
Log-Sum-Exp ist der logarithmierte Normierungsfaktor jedes Modells der Exponentialfamilie (:numref:`sec_mdl-exponential-family`). Bis auf eine Subtraktion entspricht es dem Kreuzentropieverlust. Die Verschiebung :eqref:`eq_mdl-opt-stable-lse` zeigt unmittelbar, wie nahe es am größten Logit liegt:

<!--en:54859883-->
**Proposition (beidseitige Schranke für Log-Sum-Exp).** *Für jedes $\mathbf{z} \in \mathbb{R}^n$ gilt*

<!--en:02a523bb-->
**Beweis.** Setzen wir $c = \max_j z_j$ in :eqref:`eq_mdl-opt-stable-lse`. Jeder Exponentialterm $e^{z_j - c}$ liegt in $(0, 1]$, und der größte Term ist gleich $1$. Die Summe liegt also in $[1, n]$ und ihr Logarithmus in $[0, \log n]$. Addition von $c$ liefert beide Schranken. $\blacksquare$

<!--en:28d87fc7-->
Log-Sum-Exp ist demnach ein *weiches Maximum*. In unserem Beispiel mit zwei Klassen weicht $\mathrm{lse}(0,t)=\log(1+e^t)$ höchstens um $\log 2\approx0.69$ von $\max(0,t)$ ab. Der Abstand ist bei $t=0$ am größten, wenn beide Logits gleich sind. Mit wachsendem $|t|$ nähert sich die Funktion dem Maximum an (:numref:`fig_mdl-opt-softplus-max`).

<!--en:5d4fa22b-->
![Log-Sum-Exp als weiches Maximum: $\mathrm{lse}(0,t)=\log(1+e^t)$ weicht höchstens um $\log 2$ von $\max(0,t)$ ab, entsprechend der beidseitigen Schranke für $n=2$. Der Abstand erreicht sein Maximum bei $t=0$, wo beide Logits gleich sind, und verschwindet mit wachsendem $|t|$.](../img/mdl-opt-softplus-max.svg)

<!--en:3a908ff0-->
Der Gradient ist zudem die Softmax-Funktion selbst, $\partial\,\mathrm{lse}(\mathbf z)/\partial z_i=e^{z_i}/\sum_j e^{z_j}=\mathrm{softmax}(\mathbf z)_i$. Deshalb vereinfacht sich der Gradient der Kreuzentropie nach den Logits für das Label $y$ zu $\mathrm{softmax}(\mathbf z)-\mathbf e_y$. Wie :numref:`sec_mdl-convexity` außerdem zeigt, ist Log-Sum-Exp konvex.

<!--en:8a0f0472-->
Subtrahieren wir das verschobene Log-Sum-Exp vom verschobenen Logit, erhalten wir **Log-Softmax**. Damit berechnen wir die Log-Wahrscheinlichkeiten direkt, ohne je die Wahrscheinlichkeiten selbst zu bilden:

<!--en:6ec1c224-->
Da wir auf das Endergebnis keine Exponentialfunktion anwenden, kann eine winzige Wahrscheinlichkeit vor dem Logarithmieren nicht durch Unterlauf zu null werden. Beachten wir ein Detail in :eqref:`eq_mdl-opt-log-softmax`: Die algebraisch äquivalente Form $z_i-\mathrm{lse}(\mathbf z)$ ist in Gleitkommaarithmetik merklich ungenauer. Sie bildet zuerst die große Zahl $\mathrm{lse}=c+\log\sum_j e^{z_j-c}$ und subtrahiert sie dann von der großen Zahl $z_i$. Wie viel macht die Klammerung aus? Vergleichen wir beide Formen in fp32 für die Logits $(1000, 1001, 1002)$:

<!--en:092b2f36-->
Die direkte Berechnung läuft über und ergibt `inf`. Das würde selbst in fp64 passieren, denn es gilt $1000>709.78$. Deshalb entstand in der Einleitung aus `inf/inf` dreimal `NaN`. Der mit Verschiebung berechnete Wert $\mathrm{lse}=1002.4076$ liegt deutlich innerhalb der Schranken $[1002, 1002+\log 3]$. Interessanter ist der Unterschied zwischen den beiden Formen von Log-Softmax: Wenden wir die Exponentialfunktion auf $z_i-\mathrm{lse}$ an, summieren sich die Wahrscheinlichkeiten zu $1.0000132$. Mit :eqref:`eq_mdl-opt-log-softmax` ergibt sich dagegen die Summe $1.0000000$.

<!--en:4f332160-->
Warum verliert $z_i-\mathrm{lse}$ an Genauigkeit? Überraschenderweise ist die Subtraktion selbst exakt. Unterscheiden sich zwei Gleitkommazahlen höchstens um den Faktor zwei, ist ihre Differenz exakt darstellbar. Beide sind Vielfache des Gitterabstands bei der kleineren Zahl, und für ihre Differenz sind keine zusätzlichen Bits nötig (Lemma von Sterbenz, Aufgabe :exref:`ex_mdl-numerical-stability-conditioning-exact-subtraction`).

<!--en:08da1cd3-->
Der Fehler entstand schon beim Speichern von $\mathrm{lse}$. Nahe $1002$ haben fp32-Zahlen den Abstand $2^{-14}\approx6.1\times10^{-5}$. Der wahre Wert $1002.4076060$ wird daher um $1.32\times10^{-5}$ abgerundet. Jede als $z_i-\mathrm{lse}$ berechnete Log-Wahrscheinlichkeit ist also um $1.32\times10^{-5}$ zu groß, jede daraus berechnete Wahrscheinlichkeit um den Faktor $e^{1.32\times10^{-5}}$ erhöht, und ihre Summe übersteigt $1$ um $1.32\times10^{-5}$. Ein absoluter Fehler im Exponenten führt nach Anwendung der Exponentialfunktion zu einem relativen Fehler etwa derselben Größe. Die Subtraktion hat lediglich einen Fehler offengelegt, der schon in ihrer Eingabe lag. Genau darum geht es in :numref:`subsec_mdl-catastrophic-cancellation`.

<!--en:e14528f0-->
### Kreuzentropie aus Logits

<!--en:e26906d7-->
Der Kreuzentropieverlust für das Label $y$ ist der negative Wert von Log-Softmax. Mit :eqref:`eq_mdl-opt-log-softmax` erhalten wir ihn direkt aus den Logits:

<!--en:8478d5c1-->
Dazu genügen eine stabile Auswertung von Log-Sum-Exp und eine Subtraktion. Wir müssen keine Wahrscheinlichkeit bilden. In unserem Beispiel beträgt der Verlust für Klasse $0$ genau $\mathrm{lse}(0,t)-0=\log(1+e^t)$. Berechnen wir dagegen zuerst die Wahrscheinlichkeiten, fällt die Wahrscheinlichkeit $e^{-t}/(1+e^{-t})\approx e^{-t}$ der wahren Klasse für $t>87.3$ unter den normalen Bereich von fp32. Bei etwa $t=104$ fällt sie durch Unterlauf auf null und ihr negativer Logarithmus steigt auf $\infty$. Vergleichen wir die beiden Wege mit den eingebauten Bibliotheksfunktionen für $t=103$ und $104$, also beiderseits dieser Schwelle:

<!--en:9cc2a2e6-->
Direkt aus den Logits berechnet ergibt sich in jeder Zeile der Verlust $t$. Der Weg über die Wahrscheinlichkeit liefert bei $t=20$ und $60$ dasselbe Ergebnis. Bei $t=103$ fällt die Wahrscheinlichkeit $e^{-103}$ jedoch in den subnormalen Bereich. Das weitere Ergebnis hängt davon ab, wie die Bibliothek subnormale Zahlen behandelt.

<!--en:c4ef2297-->
Die Exponentialfunktion von MXNet setzt subnormale Ergebnisse auf $0$. Die Wahrscheinlichkeit wird also $0$ und der Verlust springt auf `inf`, schon bei $t=103$ und ebenso bei $104$.

<!--en:9e2a6759-->
PyTorch erhält subnormale Zahlen: $e^{-103}\approx1.32\times2^{-149}$ wird auf die kleinste subnormale fp32-Zahl $2^{-149}$ gerundet. Nur ein signifikantes Bit bleibt erhalten. Der Verlust beträgt daher $149\log 2=103.2789$ und ist ohne jede Warnung bereits in der ersten Nachkommastelle falsch. Bei $t=104$ wird $e^{-104}<2^{-150}$ auf $0$ gerundet und der Verlust wird `inf`.

<!--en:4767a835-->
Die Softmax-Funktion von TensorFlow setzt subnormale Zahlen auf null und liefert damit einen Verlust von `inf` sowohl bei $t=103$ als auch bei $104$. Wie die letzte Spalte zeigt, verdeckt Keras denselben Fehler: Mit `from_logits=False` begrenzt es die Wahrscheinlichkeiten vor dem Logarithmieren auf $[10^{-7}, 1-10^{-7}]$ und liefert $16.1181=-\log 10^{-7}$ für jedes $t$ oberhalb von etwa $16$. Der Verlust bleibt also konstant, der Gradient ist null, und die Logits des Modells spielen keine Rolle mehr.

<!--en:dccb67c2-->
XLA setzt bei der Ausführung kompilierter Operationen subnormale Zahlen auf null. Die Wahrscheinlichkeit wird daher $0$, und der Verlust springt auf `inf`, sowohl bei $t=103$ als auch bei $104$.

<!--en:11607b72-->
Für die Praxis empfiehlt es sich, Verluste, Likelihoods und A-posteriori-Wahrscheinlichkeiten im logarithmischen Raum zu belassen, bis tatsächlich eine Wahrscheinlichkeit benötigt wird. Deshalb implementiert die kombinierte Verlustfunktion in :numref:`subsec_softmax-implementation-revisited` die Formel :eqref:`eq_mdl-opt-ce-from-logits`. Was die Kreuzentropie misst, erläutert :numref:`sec_mdl-information_theory`. Um denselben Unterlauf zu vermeiden, summieren wir in :numref:`sec_mdl-naive_bayes` die Log-Wahrscheinlichkeiten Hunderter einzelner Pixel.

<!--en:6ec8ec1c-->
## Katastrophale Auslöschung

<!--en:40cd247f-->
### Konditionszahl einer Funktion

<!--en:773db722-->
In fp32 ergibt $\log(1+x)$ an der Stelle $x=10^{-8}$ den Wert $0$, also einen relativen Fehler von 100%. Die Bibliotheksfunktion `log1p` liefert dagegen $10^{-8}$. Ist $\log(1+x)$ in der Nähe von $x=0$ grundsätzlich schwer zu berechnen, oder liegt es an der Formel? Um solche Fragen systematisch zu beantworten, brauchen wir eine Größe, die unabhängig vom Algorithmus misst, wie stark eine Funktion Fehler in ihrer Eingabe verstärkt.

<!--en:64352b8f-->
Die **relative Konditionszahl** einer differenzierbaren Funktion $f$ an einer Stelle $x$ mit $f(x)\ne0$ gibt an, um welchen Faktor sich eine kleine relative Störung von $x$ als relative Störung von $f(x)$ auswirkt. Ändern wir $x$ zu $x(1+\delta)$, wird die Ausgabe in erster Ordnung zu $f(x(1+\delta))\approx f(x)+xf'(x)\,\delta$. Teilen wir die Änderung durch $f(x)$, erhalten wir den Verstärkungsfaktor

<!--en:51306372-->
Für die Exponentialfunktion gilt zum Beispiel $\kappa_f(x)=|x|$. Bei $x=88$ wird ein Eingabefehler von einer Rundungseinheit zu $88$ Einheiten in der Ausgabe. Genau diesen Effekt haben wir in der Codezelle zu Log-Softmax gesehen: Ein absoluter Fehler im Exponenten führte zu einem etwa gleich großen relativen Fehler in der Wahrscheinlichkeit. Die Größe $\kappa_f$ beschreibt eine Eigenschaft erster Ordnung der exakten mathematischen Funktion. Sie sagt nichts darüber aus, wie wir $f$ auswerten. Wo $f(x)=0$ gilt, ist sie nicht anwendbar, weil der relative Fehler dort undefiniert ist.

<!--en:eb7dcffa-->
Die Subtraktion hat zwei Eingaben. Auf beide lässt sich dieselbe Überlegung erster Ordnung anwenden. Nehmen wir an, $\hat a=a(1+\delta_a)$ und $\hat b=b(1+\delta_b)$ seien mit relativen Fehlern $|\delta_a|,|\delta_b|\le u$ behaftet und die Subtraktion selbst sei exakt. Dann gilt $\hat a-\hat b=(a-b)+(a\delta_a-b\delta_b)$. Aus der Dreiecksungleichung folgt daher für den relativen Fehler der Differenz die Schranke

<!--en:5a88b0b9-->
Der Faktor $(|a|+|b|)/|a-b|$ ist die Konditionszahl der Subtraktion. Er wächst stark an, sobald $a\approx b$ gilt. Die Subtraktion zweier fast gleicher Zahlen, die bereits Rundungsfehler enthalten, heißt **katastrophale Auslöschung** (engl. catastrophic cancellation). Die genauen führenden Stellen heben sich auf, und das Rauschen in den hinteren Stellen rückt an die führenden Stellen des Ergebnisses. Die folgende Tabelle fasst die Konditionszahlen einiger häufiger Probleme zusammen:

<!--en:5f99fd35-->
| Problem | $\kappa_f$ | Wertebereich |

<!--en:1b0faddd-->
| $a-b$ | $(\lvert a\rvert+\lvert b\rvert)/\lvert a-b\rvert$ | groß für $a\approx b$ |

<!--en:59aef46c-->
| $\log(1+x)$ | $x/((1+x)\log(1+x))$ | $1.000$ bei $x=10^{-8}$, $0.82$ bei $x=0.5$ |

<!--en:450096ac-->
| $e^x$ | $\lvert x\rvert$ | groß für große $\lvert x\rvert$ |

<!--en:5e63c704-->
| $1-\cos x$ | $x\sin x/(1-\cos x)$ | strebt gegen $2$ für $x\to0$ |

<!--en:649d0cbe-->
| $\sqrt x$ | $1/2$ | immer |

<!--en:3615bfdb-->
| Varianz von Daten mit Mittelwert $m$ und Standardabweichung $s$ | höchstens $2(1+\lvert m\rvert/s)$ | Herleitung siehe unten |

<!--en:fca99b2d-->
Die zweite Zeile beantwortet unsere Ausgangsfrage: $\log(1+x)$ ist nahe $x=0$ gut konditioniert ($\kappa_f\approx 1$). Der naive Algorithmus `log(1 + x)` scheitert an einem schlecht konditionierten Zwischenschritt. Er bildet zunächst $y=1+x$ und wertet dann $\log y$ bei $y\approx 1$ aus. Dort hat der Logarithmus die Konditionszahl $\kappa_{\log}(y)=1/|\log y|\approx 1/x=10^8$. Beim Runden von $1+x$ entsteht ein relativer Fehler der Größenordnung $u$ in $y$. Die Funktion $\log$ verstärkt ihn um den Faktor $10^8$, in fp32 auf mehr als 100%. Die Funktion `log1p` ist dagegen so implementiert, dass der Rundungsfehler von $1+x$ nie ins Ergebnis eingeht. Für winzige $x$ liefert sie direkt $x$.

<!--en:acca7b6a-->
Damit haben wir das Leitprinzip dieses Abschnitts: *Ein instabiler Algorithmus löst ein gut konditioniertes Problem über einen schlecht konditionierten Zwischenschritt.* Ist $\kappa_f$ klein und das Ergebnis ungenau, liegt es am Algorithmus. Dann sollten wir ihn umformulieren. Ist dagegen $\kappa_f$ selbst groß, kann kein Algorithmus das Ergebnis retten. Dann müssen wir das Problem ändern. Damit befasst sich :numref:`subsec_mdl-conditioning-revisited`.

<!--en:ab05494f-->
### Subtraktion fast gleicher Zahlen

<!--en:e7a6f330-->
Prüfen wir das im Code. Die folgende Zelle wiederholt das Experiment mit $\log(1+x)$. Dann subtrahiert sie zwei fp32-Zahlen, die in den ersten sieben Dezimalstellen übereinstimmen: $1.0002344$ und $1.0002341$. Ihre wahre Differenz ist $3.0\times10^{-7}$:

<!--en:726a9abc-->
Zwischen $1$ und $2$ ist jede Differenz zweier fp32-Zahlen ein ganzzahliges Vielfaches von $\varepsilon_{\mathrm{mach}}=1.19\times10^{-7}$. Die wahre Differenz $3.0\times10^{-7}$ liegt nicht auf diesem Gitter. Die Subtraktion lieferte zwei ulps, also $2.38\times10^{-7}$ (20% zu wenig). Dennoch war die Subtraktion selbst erneut fehlerfrei. Die gespeicherten Eingaben waren $1.0002343655$ und $1.0002341270$. Die Zelle hat genau deren Differenz ausgegeben. Der Fehler steckte bereits in den gespeicherten Eingaben. Der Auslöschungsfaktor :eqref:`eq_mdl-opt-cancellation-factor` beträgt hier $2.0005/(3\times10^{-7})\approx7\times10^6$. Multipliziert mit $u\approx6\times10^{-8}$ lässt er einen relativen Fehler bis $0.4$ zu. Der beobachtete Fehler von $0.2$ liegt innerhalb dieser Schranke.

<!--en:d3db0741-->
Dieselbe Auslöschung tritt bei $\log(1+x)$ und $e^x-1$ nahe $0$ auf (wo `log1p` und `expm1` sie vermeiden), bei $1-\cos x$ nahe $0$ (umformuliert als $2\sin^2(x/2)$) sowie in der Lösungsformel für quadratische Gleichungen, wenn $b^2\gg 4|ac|$ gilt (Aufgabe :exref:`ex_mdl-numerical-stability-conditioning-blame`). In allen diesen Fällen lässt sich die problematische Subtraktion durch algebraisches Umformen beseitigen.

<!--en:6f93aff2-->
Bei endlichen Differenzen können wir die Subtraktion nicht beseitigen und müssen einen Kompromiss finden. Eine numerische Gradientenprüfung vergleicht das Ergebnis des automatischen Differenzierens mit dem Vorwärtsdifferenzenquotienten $(f(x+h)-f(x))/h$. Wie klein sollten wir die Schrittweite $h$ wählen? Nach dem Satz von Taylor beträgt der mathematische Abbruchfehler etwa $\tfrac h2|f''(x)|$ und schrumpft für $h\to 0$. Zugleich ist jede Funktionsauswertung mit einem Rundungsfehler von ungefähr $u|f(x)|$ behaftet. Teilen wir die Differenz der Funktionswerte durch $h$, wächst dieser Fehler auf etwa $2u|f(x)|/h$ und damit für $h\to 0$ unbegrenzt. Der Gesamtfehler ist am kleinsten, wenn sich beide Beiträge die Waage halten, also bei

<!--en:27c8e225-->
Da der optimale Fehler proportional zu $\sqrt{u}$ ist, bewahrt selbst der beste Vorwärtsdifferenzenquotient nur etwa die Hälfte der Stellen des Formats. Für $f=\sin$ an der Stelle $x=1$ erwarten wir daher $h^\star\approx2\times10^{-8}$ (Fehler $1.8\times10^{-8}$) in fp64 und $h^\star\approx5\times10^{-4}$ in fp32. Verkleinern wir $h$ in beiden Formaten von $10^{-2}$ bis auf $10^{-12}$, um die gegenläufigen Effekte direkt zu beobachten:

<!--en:9c48668a-->
In fp64 fällt der Fehler linear mit $h$ von $4\times10^{-3}$ bei $h=10^{-2}$ (wo der Abbruchfehler dominiert) auf etwa $10^{-8}$ nahe $h=10^{-8}$. Der genaue Wert bei jedem einzelnen $h$ schwankt infolge der Rundung. Dann überwiegt die Auslöschung, und der Fehler steigt wieder auf $4\times10^{-5}$ bei $h=10^{-12}$. In fp32 liefert der günstigste Bereich von $h=10^{-3}$ bis $10^{-4}$ nur etwa drei korrekte Stellen. Ab $h=10^{-8}$ und für kleinere Schritte wird $1+h$ direkt auf $1$ gerundet. Der Differenzenquotient ist dann $0$ und der Fehler so groß wie die ganze Ableitung, $\cos 1=0.54$. Numerische Gradientenprüfungen sollten daher immer in fp64 laufen.

<!--en:9ba266c6-->
Auch höhere Genauigkeit beseitigt die Auslöschung nicht. Sie verschiebt den Fehler lediglich zu kleineren Eingaben. Die letzte Zeile der früheren Zelle zu `log1p` zeigte, dass `log(1 + 1e-20)` in fp64 ebenso $0$ liefert wie fp32 bei $10^{-8}$, während `log1p` in beiden Formaten genau blieb. Eine Umformulierung beseitigt den schlecht konditionierten Schritt selbst :cite:`Higham.2002`.

<!--en:470e738f-->
### Varianz in einem Durchlauf

<!--en:c4da07cc-->
Katastrophale Auslöschung verbirgt sich auch in scheinbar harmlosen Formeln. Das klassische Beispiel ist die Berechnung der Stichprobenvarianz. Sie spielt im Deep Learning eine unmittelbare Rolle, denn Normalisierungsschichten und Verfahren zur Datenvorverarbeitung berechnen fortlaufend Varianzen.

<!--en:a2ca96f6-->
Für Daten $x_1,\ldots,x_N$ mit Mittelwert $m$ und Standardabweichung $s$ legt die bekannte Identität

<!--en:fa423436-->
nahe, die Varianz in einem Durchlauf zu berechnen und dazu nur die Summen von $x_i$ und $x_i^2$ zu bilden. In exakter Arithmetik gilt die Identität uneingeschränkt. In Gleitkommaarithmetik birgt sie eine Falle. Für $s\ll|m|$ liegen beide Terme auf der rechten Seite nahe bei $m^2$, während ihre Differenz $s^2$ beträgt. Der Auslöschungsfaktor :eqref:`eq_mdl-opt-cancellation-factor` liegt also bei etwa $2m^2/s^2$. Für $m=10^9$ und $s=1$ erreicht er $2\times10^{18}$, zwei Größenordnungen mehr als der Faktor $10^{16}$, den fp64 verkraftet.

<!--en:3813abd9-->
Welchen Anteil an diesem Verlust erzwingt das Problem selbst? Bestimmen wir die Konditionszahl der Varianz. Wir stören jede Eingabe $x_i$ um einen relativen Fehler $\delta_i$ mit $|\delta_i|\le u$, ersetzen also $x_i$ durch $x_i+x_i\delta_i$. Die Ableitung der Varianz nach $x_i$ ist $\tfrac2N(x_i-m)$, denn die dadurch bewirkte Änderung von $m$ wird mit $\sum_i(x_i-m)=0$ multipliziert. Schreiben wir $x_i=(x_i-m)+m$, ergibt sich für die Änderung der Varianz in erster Ordnung

<!--en:1bdba575-->
Die erste Summe hat höchstens den Betrag $2s^2 u$. Nach der Cauchy-Schwarz-Ungleichung hat die zweite höchstens den Betrag $2|m|u\cdot\frac1N\sum_i|x_i-m|\le 2|m|s\,u$. Division durch $s^2$ zeigt, dass die relative Konditionszahl der Varianz höchstens $2(1+|m|/s)\approx 2|m|/s$ beträgt, hier also etwa $2\times10^9$ (die Wahl $\delta_i=u\,\mathrm{sign}(x_i-m)$ ergibt $1.6\,|m|/s$ für gaußverteilte Daten). Das Problem selbst lässt daher nur einen Fehler von $2\times10^9 u\approx 2\times10^{-7}$ zu. Das entspricht etwa sieben korrekten Stellen. Die naive Formel mit einem Durchlauf hat dagegen einen Fehler der Größenordnung $(m/s)^2 u$: Sie *quadriert* die Konditionszahl, genau wie die Normalgleichungen in :numref:`subsec_mdl-conditioning-revisited` die Konditionszahl einer Matrix quadrieren.

<!--en:4b4b5842-->
Ein stabiler Algorithmus hält in einem Durchlauf alle Abweichungen in der Größenordnung $s$ statt $m^2$. Das Verfahren von :citet:`Welford.1962` führt einen laufenden Mittelwert $m_k$ und eine laufende Summe von Quadraten *zentrierter* Werte $M_k = \sum_{i \le k} (x_i - m_k)^2$. Beide Größen aktualisiert es anhand der Abweichung $d_k=x_k-m_{k-1}$ jedes neuen Stichprobenwerts vom bisherigen Mittelwert:

<!--en:5d9cb0a3-->
Die Aktualisierung von $M_k$ ist asymmetrisch: Sie multipliziert die Abweichung vom *alten* Mittelwert mit der Abweichung vom *neuen* Mittelwert. Gerade diese Asymmetrie macht die Rekursion exakt:

<!--en:61e61a8c-->
**Proposition (Welfords Rekursion).** *Mit $m_0 = M_0 = 0$ erfüllen die Rekursionsformeln :eqref:`eq_mdl-opt-welford` für jedes $k \ge 1$ in exakter Arithmetik die Identitäten*

<!--en:1bb85142-->
**Beweis.** Die Mittelwertidentität $k\, m_k = (k-1)\, m_{k-1} + x_k$ folgt unmittelbar aus der ersten Rekursionsformel. Für $M_k$ führen wir eine Induktion über $k$. Mit $m_k - m_{k-1} = d_k/k$ und $x_k - m_k = d_k\,(k-1)/k$ spalten wir den $k$-ten Term ab und zentrieren die ersten $k-1$ Terme um $m_{k-1}$:

<!--en:8b2bcf95-->
Der gemischte Term verschwindet wegen $\sum_{i \le k-1} (x_i - m_{k-1}) = 0$. Nach Induktionsannahme ist die erste Summe gleich $M_{k-1}$. Die beiden verbleibenden Terme ergeben zusammen $\frac{k-1}{k}\,d_k^2 = d_k \cdot d_k\frac{k-1}{k} = d_k(x_k - m_k)$, genau den Zuwachs aus der Rekursionsformel. $\blacksquare$

<!--en:d784671e-->
Schwieriger ist der Nachweis, dass Welfords Rekursion auch in Gleitkommaarithmetik genau bleibt. Wir skizzieren hier nur die Idee (die vollständige Analyse findet sich bei :citet:`Higham.2002`). Auf den ersten Blick subtrahiert auch Welfords Verfahren große Zahlen, denn $x_k$ und $m_{k-1}$ liegen beide nahe $10^9$. Nach dem Lemma von Sterbenz ist diese Subtraktion jedoch exakt. Da auch der Eingabewert $x_k$ exakt ist, stammt der einzige Fehler in $d_k$ vom laufenden Mittelwert. Er beträgt höchstens einige Dutzend ulps bei $|m|$ über insgesamt $10^5$ Schritte. Jeder Beitrag zu $M_k$ beruht also auf einer Abweichung der Größenordnung $s$. Rundungsfehler in den Aktualisierungen beziehen sich daher auf $s^2$ statt auf $m^2$. Selbst ein konstanter Versatz $\varepsilon$ im Mittelwert ändert eine zentrierte Quadratsumme nur um $N\varepsilon^2$, weil der Term erster Ordnung mit $\sum_i(x_i-m)=0$ multipliziert wird. In der Praxis entspricht der Fehler von Welfords Verfahren der Empfindlichkeit des Problems, $u|m|/s\approx 1.1\times10^{-7}$. Bei der naiven Formel mit einem Durchlauf beträgt der Fehler dagegen $(m/s)^2 u\approx 100$.

<!--en:9e758a61-->
Vergleichen wir alle drei Verfahren (naive Formel mit einem Durchlauf, Welfords Verfahren und eine Referenz mit zwei Durchläufen) für $10^5$ fp64-Stichprobenwerte mit Mittelwert $10^9$ und Varianz $1$:

<!--en:fbc5c2b1-->
Die naive Schätzung mit einem Durchlauf ergibt ein exaktes Vielfaches von $128$ (meist $0$, $\pm128$ oder $\pm256$). Der Wert liegt weit von der wahren Varianz $1$ entfernt und ist oft negativ. Warum $128$? Sowohl $\frac1N\sum x_i^2$ als auch $m^2$ liegen in $[2^{59}, 2^{60})$. Dort haben benachbarte fp64-Zahlen den Abstand $2^{59-52}=128$. Ihre Differenz kann daher nur ein Vielfaches von $128$ sein. Das ist unser vorhergesagter Fehler von etwa $100$, auf das Gitter quantisiert. Welfords Rekursion stimmt dagegen in mindestens sieben Stellen mit der Referenz aus zwei Durchläufen überein. Oft sind es sogar acht oder neun, weil sich Rundungsfehler mit unterschiedlichem Vorzeichen in der Summe teilweise aufheben.

<!--en:f6f531f6-->
Dieselbe Vorsicht ist immer geboten, wenn wir eine Varianz in einem Durchlauf berechnen, etwa bei fortlaufenden Statistiken über einen Datensatz. (Die laufenden Momente, die die Batchnormalisierung für die Inferenz führt, sind exponentielle gleitende Mittelwerte, :numref:`sec_batch_norm`. Dabei werden keine großen laufenden Summen voneinander subtrahiert.)

<!--en:3442b58e-->
### Summationsreihenfolge

<!--en:235baa6a-->
Sowohl die naive Varianzformel als auch die Referenz mit zwei Durchläufen summieren $10^5$ Zahlen. In Gleitkommaarithmetik spielt die Reihenfolge der Additionen eine Rolle. Jede Addition bringt einen Faktor $(1+\delta_k)$ ein. Summieren wir von links nach rechts, durchläuft der erste Term daher alle $n-1$ Rundungen. In erster Ordnung in $u$ gilt für die Summe von links nach rechts

<!--en:0f7b4446-->
Da sich bei frühen Summanden die Rundungsfehler aller folgenden Additionen ansammeln, wächst der Fehler im ungünstigsten Fall linear mit $n$. Die *paarweise* Summation (engl. pairwise summation) vermeidet dies, indem sie beide Hälften des Arrays rekursiv summiert. Jeder Summand durchläuft dann nur $\log_2 n$ Additionen, und die Schranke sinkt auf $u\log_2 n\sum_i|x_i|$. NumPys Funktion `sum` nutzt eine blockweise Variante dieses Verfahrens. Kahans kompensierte Summation geht noch weiter: Sie führt den Rundungsfehler jeder Addition in einer Korrekturvariablen mit und erreicht so eine von $n$ unabhängige Fehlerschranke (Aufgabe :exref:`ex_mdl-numerical-stability-conditioning-summation-error-growth`). Addieren wir $10^7$ Exemplare der fp32-Zahl $0.1$ (gespeicherter Wert $0.100000001490\ldots$) einmal von links nach rechts und einmal paarweise. Für die paarweise Summation füllen wir das Array mit Nullen auf $2^{24}$ Einträge auf und addieren wiederholt benachbarte Paare:

<!--en:fb42a8b3-->
Die Summe von links nach rechts beträgt $1{,}087{,}937$. Das ist fast 9% zu hoch, liegt aber noch innerhalb der Schranke $(n-1)u\sum_i|x_i|\approx6\times10^5$. Die paarweise Summation liefert $1{,}000{,}000$, weicht also höchstens um $0.015$ von der exakten Summe ab und bleibt deutlich innerhalb ihrer Schranke $u\log_2 n\sum_i|x_i|\approx1.4$.

<!--en:5daecbfb-->
Die Abweichung bei der Summation von links nach rechts hat dieselbe Ursache wie das Verschwinden kleiner Gewichtsaktualisierungen: Jeder Summand wird auf den Gitterabstand der aktuellen Teilsumme gerundet. Überschreitet die Summe $2^{19}=524{,}288$, haben fp32-Zahlen den Abstand $1/16$. Damit entspricht $0.1$ genau $1.6$ Gitterabständen. Jede Addition rundet auf zwei Gitterabstände auf, also auf $0.125$. Das ist 25% zu viel. Oberhalb von $2^{20}$ verdoppelt sich der Abstand auf $1/8$, und jeder Summand wird wieder auf $0.125$ gerundet. Zuvor, zwischen $2^{17}$ und $2^{19}$, wurde auf $0.09375$ abgerundet. Da die Gleitkommaaddition nicht assoziativ ist, liefert eine parallele Reduktion über GPU-Threads je nach Gerät oder Durchlauf möglicherweise unterschiedliche Bitmuster. Auch Welfords Rekursion erlaubt eine paarweise Reduktion. Die Regel in Aufgabe :exref:`ex_mdl-numerical-stability-conditioning-pairwise-merge` führt die Statistiken zweier Teilsummen zusammen. So lassen sich beim verteilten Training laufende Momente über Geräte hinweg zusammenführen.

<!--en:38d6e966-->
## Kondition

<!--en:032fc82d-->
Die Konditionszahl einer Matrix und die Filterfaktoren der Ridge-Regression haben wir in :numref:`sec_mdl-svd-low-rank` hergeleitet. Jetzt betrachten wir Vorwärts- und Rückwärtsfehler, eine Formel zur Bestimmung des Rückwärtsfehlers und den Stellenverlust beim Lösen der Normalgleichungen.

<!--en:a4111e72-->
### Zwei fast parallele Geraden

<!--en:cd772604-->
Bisher war das Ergebnis einer Rechnung eine Zahl. Bei einem linearen Gleichungssystem ist es ein Vektor. In :numref:`subsec_mdl-condition-number` haben wir am System

<!--en:b664cc5e-->
gesehen, wie empfindlich eine solche Lösung sein kann. Die Lösung ist $(1,1)$. Erhöhen wir die zweite Komponente der rechten Seite um $10^{-4}$, eine relative Datenänderung von $3.5\times10^{-5}$, verschiebt sich die Lösung nach $(0,2)$. Jede Gleichung beschreibt eine Gerade in der Ebene, und die Lösung ist der Schnittpunkt. Die beiden Geraden haben die Steigungen $-1$ und $-0.9999$ und schneiden sich unter einem Winkel von $0.003^\circ$. Verschieben wir eine Gerade nur geringfügig, wandert der Schnittpunkt weit entlang der anderen (:numref:`fig_mdl-opt-two-lines`).

<!--en:f1cc3981-->
![Ein lineares Gleichungssystem ist empfindlich, wenn seine Geraden fast parallel sind. (a) Zwei senkrecht zueinander stehende Geraden schneiden sich bei $(1,1)$. Eine Verschiebung der zweiten um $0.1$ verschiebt den Schnittpunkt um 5%: Verstärkung $1$, $\kappa=1$. (b) Die Geraden $x_1+x_2=2$ und $x_1+1.1x_2=2.1$ schneiden sich ebenfalls bei $(1,1)$. Dieselbe Verschiebung verlegt den Schnittpunkt aber nach $(0,2)$. Eine Änderung um 100% bei einer Änderung von $\mathbf{b}$ um 3.4% entspricht einer Verstärkung von $29$, begrenzt durch $\kappa\approx42$. Das System im Text mit $1.0001$ statt $1.1$ hat dieselbe Geometrie mit einem Winkel von $0.003^\circ$ und $\kappa\approx4\times10^4$.](../img/mdl-opt-two-lines.svg)

<!--en:f1eb4ff3-->
Lesen wir dieselben Zahlen in umgekehrter Richtung. Angenommen, ein Lösungsverfahren liefert $\hat{\mathbf x}=(0,2)$ für das ursprüngliche System. Sein **Vorwärtsfehler** (engl. forward error), der relative Abstand zur wahren Lösung, beträgt 100%. Dabei ist $(0,2)$ die exakte Lösung eines benachbarten Problems, bei dem sich $b_2$ um $10^{-4}$ ändert. Die kleinste solche Datenänderung heißt **Rückwärtsfehler** (engl. backward error). Ein Verfahren mit winzigem Rückwärtsfehler hat ein benachbartes Problem exakt gelöst und kann dennoch weit von der gesuchten Lösung entfernt liegen.

<!--en:80af0f46-->
Wie weit, hängt vom Problem ab. Entscheidend ist wieder die Konditionszahl aus :numref:`subsec_mdl-relative-condition-number`, diesmal für die Funktion $\mathbf b\mapsto\mathbf A^{-1}\mathbf b$: der größte Faktor, um den sich eine relative Änderung der Eingabe in der relativen Änderung der Ausgabe verstärken kann. Dabei betrachten wir alle rechten Seiten und alle Richtungen der Änderung. Für eine Matrix mit den Singulärwerten $\sigma_1\ge\cdots\ge\sigma_d>0$ beträgt dieser Faktor $\kappa(\mathbf A)=\sigma_1/\sigma_d$ (:numref:`subsec_mdl-condition-number`), bei unseren beiden Geraden also etwa $4\times10^4$. Die Verschiebung $(0,10^{-4})$ ergibt nur eine Verstärkung um $2.8\times10^4$, weil sie nicht in die am stärksten verstärkte Richtung zeigt.

<!--en:87a0dc96-->
### Rückwärts- und Vorwärtsfehler

<!--en:eccc0dbf-->
Für ein lineares Gleichungssystem $\mathbf A\mathbf x=\mathbf b$ mit berechneter Lösung $\hat{\mathbf x}$ beträgt der Vorwärtsfehler $\|\hat{\mathbf x}-\mathbf x\|/\|\mathbf x\|$. Der Rückwärtsfehler ist die Größe der kleinsten Datenänderung, für die $\hat{\mathbf x}$ exakt ist. Ein Algorithmus ist **rückwärtsstabil**, wenn sein Rückwärtsfehler in der Größenordnung von $u$ liegt. Er liefert dann die exakte Lösung eines Problems, das sich etwa um eine Rundung der Daten vom gegebenen unterscheidet. Die Singulärwertzerlegung (engl. singular value decomposition, SVD) und die QR-Zerlegung sind rückwärtsstabil. In der Praxis gilt dies auch für die LU-Zerlegung mit Pivotisierung, die `np.linalg.solve` verwendet. Für die naive Varianzformel mit einem Durchlauf gilt es nicht: Jede Varianz ist nichtnegativ, also kann ein negatives Ergebnis für keinen Datensatz die exakte Varianz sein.

<!--en:cc361c56-->
Die Definition verlangt eine Minimierung über alle Datenänderungen. Für Änderungen der Matrix gibt es jedoch eine geschlossene Formel, die wir anhand des Residuums auswerten können.

<!--en:ff31aac7-->
**Proposition (Rückwärtsfehler aus dem Residuum).** *Seien $\hat{\mathbf x}\ne\mathbf 0$ und $\mathbf r=\mathbf b-\mathbf A\hat{\mathbf x}$. Der kleinste Wert von $\|\Delta\mathbf A\|_2$ unter der Bedingung $(\mathbf A+\Delta\mathbf A)\hat{\mathbf x}=\mathbf b$ beträgt $\|\mathbf r\|/\|\hat{\mathbf x}\|$. Er wird für $\Delta\mathbf A=\mathbf r\hat{\mathbf x}^\top/\|\hat{\mathbf x}\|^2$ erreicht.*

<!--en:74d7a271-->
**Beweis.** Für jede solche Störung $\Delta\mathbf A$ gilt $\Delta\mathbf A\hat{\mathbf x}=\mathbf b-\mathbf A\hat{\mathbf x}=\mathbf r$, also $\|\mathbf r\|\le\|\Delta\mathbf A\|\,\|\hat{\mathbf x}\|$. Die Rang-eins-Matrix $\mathbf r\hat{\mathbf x}^\top/\|\hat{\mathbf x}\|^2$ bildet $\hat{\mathbf x}$ auf $\mathbf r$ ab und hat die Norm $\|\mathbf r\|\,\|\hat{\mathbf x}\|/\|\hat{\mathbf x}\|^2=\|\mathbf r\|/\|\hat{\mathbf x}\|$. $\blacksquare$

<!--en:84c565ec-->
Bezogen auf die Größe von $\mathbf A$ beträgt der Rückwärtsfehler

<!--en:90c920f0-->
Ein kleines Residuum belegt einen kleinen Rückwärtsfehler, unabhängig von der Kondition des Problems. Für $\hat{\mathbf x}=(0,2)$ im Beispiel mit zwei Geraden gelten $\mathbf r=(0,-10^{-4})$ und $\omega=10^{-4}/(2.00005\cdot2)=2.5\times10^{-5}$. Die Variante, die auch $\mathbf b$ stört, stammt von Rigal und Gaches (siehe :citet:`Higham.2002`). In Gleitkommaarithmetik unterliegt bereits die Berechnung des Residuums Rundungsfehlern. Ausgegebene Werte von $\omega$ unterhalb von etwa $u$ sind daher Rauschen.

<!--en:3101d10e-->
Die Konditionszahl übersetzt den Rückwärtsfehler in eine Schranke für den Vorwärtsfehler. Dieses Ergebnis sollten wir uns merken. Für den Beweis genügen die beiden Normen $\|\mathbf A\|=\sigma_1$ und $\|\mathbf A^{-1}\|=1/\sigma_d$ (:numref:`sec_mdl-svd-low-rank`).

<!--en:a36ca097-->
**Proposition (Vorwärtsfehler $\le$ Konditionszahl $\times$ Rückwärtsfehler).** *Sei $\mathbf{A}$ invertierbar, und sei $\mathbf{A}\mathbf{x} = \mathbf{b}$. Die berechnete Lösung $\hat{\mathbf{x}}$ löse ein benachbartes System exakt, also $(\mathbf{A} + \Delta\mathbf{A})\,\hat{\mathbf{x}} = \mathbf{b}$ mit $\|\Delta\mathbf{A}\| \le \omega \|\mathbf{A}\|$. Dann gilt*

<!--en:a024330e-->
**Beweis.** Subtrahieren wir $\mathbf{A}\mathbf{x} = \mathbf{b}$ von $(\mathbf{A} + \Delta\mathbf{A})\hat{\mathbf{x}} = \mathbf{b}$, erhalten wir $\mathbf{A}(\hat{\mathbf{x}} - \mathbf{x}) = -\Delta\mathbf{A}\,\hat{\mathbf{x}}$. Daraus folgen $\hat{\mathbf{x}} - \mathbf{x} = -\mathbf{A}^{-1}\Delta\mathbf{A}\,\hat{\mathbf{x}}$ und

<!--en:c58a8be2-->
Hier messen wir den Fehler relativ zu $\hat{\mathbf x}$, dem Vektor, der uns vorliegt. Für den Fehler relativ zur wahren Lösung $\mathbf x$ setzen wir $\|\hat{\mathbf x}\|\le\|\mathbf x\|+\|\hat{\mathbf x}-\mathbf x\|$ ein und lösen nach $\|\hat{\mathbf x}-\mathbf x\|$ auf. So erhalten wir die Schranke $\kappa\omega/(1-\kappa\omega)$ für $\kappa\omega<1$, die in erster Ordnung mit der ursprünglichen übereinstimmt. Für das System mit zwei Geraden gilt $\kappa\omega=4\times10^4\cdot
2.5\times10^{-5}=1.0$, und der Fehler relativ zu $\hat{\mathbf x}$ beträgt $\|(-1,1)\|/\|(0,2)\|=0.71$. Er liegt innerhalb der Schranke. Die entsprechende Schranke für einen Fehler in der rechten Seite statt in der Matrix findet sich in :numref:`subsec_mdl-condition-number`.

<!--en:7df9ad92-->
Für ein rückwärtsstabiles Verfahren gilt $\omega\approx u$. Wenden wir $\log_{10}$ auf die Schranke an, erhalten wir die Faustregel

<!--en:e62bc217-->
Die Regel beschreibt den ungünstigsten Fall. Bei einer Matrix mit $\kappa=10^3$ können drei der sieben fp32-Stellen verloren gehen. Das durchgängige Beispiel dieses Kapitels $\tfrac12(x_1^2+1000x_2^2)$ hat dieses $\kappa$ und verliert keine Stelle: Seine Matrix ist diagonal, und für jede Unbekannte genügt eine korrekt gerundete Division.

<!--en:110825a9-->
### Beispiel: die Hilbert-Matrix

<!--en:d25a6d9b-->
Die **Hilbert-Matrix** $\mathbf H_{ij}=1/(i+j-1)$, $i,j=1,\ldots,n$, hat eine Konditionszahl, die exponentiell mit $n$ wächst. Ihre Zeilen werden ebenso wie die beiden Geraden oben fast parallel. Wir lösen $\mathbf H\mathbf x=\mathbf b$ mit $\mathbf b=\mathbf H\mathbf 1$ und der exakten Lösung $\mathbf 1$. Den Vorwärtsfehler vergleichen wir mit der Faustregel und mit dem Rückwärtsfehler :eqref:`eq_mdl-opt-residual-backward-error`. Bevor wir das Lösungsverfahren beurteilen, ist ein Detail wichtig. Berechnen wir $\mathbf b=\mathbf H\mathbf 1$ in fp64, wird jeder Eintrag von $\mathbf b$ gerundet. Die Daten ändern sich also schon vor dem ersten Lösungsschritt um eine relative Größe der Ordnung $u$. (Die Rundung der Einträge $1/3$, $1/7$, … schadet hier nicht, weil wir $\mathbf b$ aus der bereits gespeicherten Matrix bilden.) Wie viel Genauigkeit kostet allein das? Jede fp64-Zahl ist rational. Deshalb können wir das gespeicherte System mit dem Gauß-Jordan-Verfahren und Einträgen vom Typ `Fraction` ganz ohne Rundung lösen. Die Tabelle zeigt den Fehler dieser exakten Lösung neben dem Fehler von `np.linalg.solve`.

<!--en:4f47b596-->
Für die Dimensionen $n \in \{4, 6, 8, 10, 12\}$ konstruieren wir die $n \times n$-Hilbert-Matrix `H` und die rechte Seite `b = H @ x_true`. Wir vergleichen das Gleitkommaverfahren `np.linalg.solve(H, b)` mit `solve_exact(H, b)` anhand von Konditionszahl, Vorwärtsfehler, erhaltenen Stellen und Rückwärtsfehler:

<!--en:256bf733-->
Die Zahl der erhaltenen Stellen fällt von etwa $13$ bei $n=4$ auf deutlich unter eine bei $n=12$. Dabei bleibt der Rückwärtsfehler bei etwa $10^{-16}$ oder darunter. Jeder berechnete Vektor löst also ein System, das sich vom gespeicherten in der sechzehnten oder siebzehnten Stelle unterscheidet. Ist das Lösungsverfahren für den Stellenverlust verantwortlich? Die letzte Spalte entlastet es. Selbst die exakte Lösung des gespeicherten Systems weicht ohne jeden Rundungsfehler in der Rechnung von $\mathbf 1$ ab. Ihr Fehler hat in jeder Zeile dieselbe Größenordnung wie beim Gleitkommaverfahren: etwa $10^{-7}$, $10^{-4}$ und $10^{-1}$ für $n=8$, $10$ und $12$. Die Stellen gingen bereits beim Speichern von $\mathbf b$ verloren, bevor ein Lösungsverfahren zum Einsatz kam. Die Faustregel ist um etwa eine Stelle zu pessimistisch (je nach Zeile um etwa eine halbe bis eineinhalb Stellen), weil der Fehler nicht in die am stärksten verstärkte Richtung zeigt. Bei $n=12$ gilt $\kappa u\approx2$, und die Schranke liefert keine Aussage mehr. Die letzten Nachkommastellen hängen von der verwendeten LAPACK-Kompilierung ab, der Verlauf bleibt gleich.

<!--en:15aeb25f-->
### Kondition der Normalgleichungen

<!--en:1975569c-->
Das Problem der kleinsten Quadrate $\min_{\mathbf{w}}\|\mathbf{A}\mathbf{w}-\mathbf{b}\|^2$ mit einer Matrix $\mathbf A\in\mathbb R^{n\times d}$ von vollem Spaltenrang lässt sich durch die Normalgleichungen $\mathbf{A}^\top\mathbf{A}\mathbf{w}=\mathbf{A}^\top\mathbf{b}$ lösen. Dieser algebraisch äquivalente Weg führt über eine schlechter konditionierte Matrix (:numref:`subsec_mdl-condition-number`):

<!--en:01dee8d8-->
Das haben wir schon bei der Varianzberechnung in einem Durchlauf gesehen: Die bequeme Formel quadriert die Konditionszahl des Problems. Nach der Faustregel können bei den Normalgleichungen etwa $2\log_{10}\kappa$ Stellen verloren gehen. Ein Verfahren mit QR-Zerlegung ($\mathbf A=\mathbf Q\mathbf R$ mit orthonormaler Matrix $\mathbf Q$ und Dreiecksmatrix $\mathbf R$) oder SVD arbeitet direkt mit $\mathbf A$ und verliert etwa $\log_{10}\kappa$ Stellen, *wenn das Residuum $\mathbf b-\mathbf A\mathbf w$ klein ist*. Bei großem Residuum ist das Problem der kleinsten Quadrate selbst empfindlicher: Seine Konditionszahl beträgt etwa $\kappa+\kappa^2\tan\theta$, wobei $\theta$ der Winkel zwischen $\mathbf b$ und dem Spaltenraum von $\mathbf A$ ist. QR und SVD verlieren dann ebenfalls annähernd $2\log_{10}\kappa$ Stellen, aber nie mehr, als das Problem erzwingt. Die Normalgleichungen verlieren so viele Stellen unabhängig vom Residuum.

<!--en:88ed4838-->
Das durchgängige Beispiel dieses Kapitels ist zu gut konditioniert, um den Unterschied zu zeigen. Für seine Designmatrix gilt $\mathbf A^\top\mathbf A=\mathrm{diag}(1,1000)$ und damit $\kappa(\mathbf A)=\sqrt{1000}\approx32$. Die Codezelle konstruiert deshalb aus zufälligen orthonormalen Faktoren schwierigere Designmatrizen mit $\kappa(\mathbf A)=10^3$ und $10^6$. Die rechte Seite liegt im Spaltenraum (Residuum null). Wir erwarten Unterschiede von drei beziehungsweise sechs Stellen:

<!--en:9764538f-->
Bei $\kappa(\mathbf A)=10^3$ bleiben mit den Normalgleichungen etwa $10$ bis $12$ Stellen erhalten, gegenüber etwa $14$ mit `lstsq`. Bei $10^6$ sind es etwa $4$ bis $6$ gegenüber etwa $11$ bis $13$. Die genauen Stellenzahlen schwanken von Durchlauf zu Durchlauf um ein oder zwei Stellen. Die Unterschiede streuen um $\log_{10}\kappa$ und weichen meist um höchstens eine Stelle davon ab. Deshalb beruhen `lstsq` und die Pseudoinverse aus :numref:`subsec_mdl-pseudoinverse` auf der SVD und nicht auf $(\mathbf A^\top\mathbf A)^{-1}\mathbf A^\top$.

<!--en:be001942-->
### Ridge-Regularisierung und Kondition

<!--en:0264bb49-->
Ist $\kappa(\mathbf A)$ selbst groß, weil Merkmale fast kollinear sind oder die Designmatrix keinen vollen Rang hat, hilft keine Wahl des Algorithmus: Wir müssen das Problem ändern. Eine mögliche Änderung erhält die Lösung. Ein **Vorkonditionierer** (engl. preconditioner) transformiert die Variablen so, dass das neue Problem besser konditioniert ist und sich seine Lösung in die ursprüngliche zurücktransformieren lässt. Ein Beispiel ist die Division jeder Spalte von $\mathbf A$ durch ihre Norm (Aufgabe :exref:`ex_mdl-numerical-stability-conditioning-feature-scaling`). Dasselbe leisten innerhalb eines Optimierers die koordinatenweisen Schrittweiten aus :numref:`sec_mdl-adaptive-stochastic-methods`.

<!--en:67ffd154-->
Die Ridge-Regularisierung ändert auch die Lösung. Sie minimiert $\tfrac12\|\mathbf{A}\mathbf{w}-\mathbf{b}\|^2+\tfrac\lambda2\|\mathbf{w}\|^2$. Der Minimierer ist

<!--en:4da7b455-->
die Ridge-Lösung aus :numref:`subsec_mdl-pseudoinverse`. Seien $\sigma_1\ge\cdots\ge\sigma_d\ge0$ die Singulärwerte von $\mathbf A\in\mathbb R^{n\times d}$, mit $\sigma_i=0$ für Indizes oberhalb des Rangs. Die Matrix $\mathbf A^\top\mathbf A+\lambda\mathbf I$ hat dieselben Eigenvektoren wie $\mathbf A^\top\mathbf A$ und die Eigenwerte $\sigma_i^2+\lambda\ge\lambda>0$. Sie ist also positiv definit und damit auch dann invertierbar, wenn $\mathbf A$ keinen vollen Rang hat. Es gilt

<!--en:93033a34-->
Für $\sigma_1>\sigma_d$ fällt dieser Quotient streng monoton mit $\lambda$ (seine Ableitung ist $(\sigma_d^2-\sigma_1^2)/(\sigma_d^2+\lambda)^2<0$) und strebt gegen $1$ für $\lambda\to\infty$. Die Addition von $\lambda$ hebt den unteren Rand des Spektrums an. Solange $\lambda\ll\sigma_1^2$ gilt, verschiebt sich der obere Rand kaum. Daher sinkt $\kappa$ annähernd proportional zu $1/\lambda$, sobald $\lambda\gg\sigma_d^2$ gilt. Der Dämpfungsterm bei Optimierern zweiter Ordnung und der Jitter, den wir bei der Gaußprozessregression zur Kernel-Matrix addieren (:numref:`chap_gp`), sind andere Bezeichnungen für denselben Term $\lambda\mathbf I$.

<!--en:85d189e6-->
Der Preis dafür ist ein anderer Minimierer. In unserer Notation sind $\mathbf v_i$ die rechten Singulärvektoren von $\mathbf A$ und $\mathbf w_0=\mathbf A^+\mathbf b$ die Lösung kleinster Norm für das Problem der kleinsten Quadrate. Die Filterfaktoren aus :numref:`subsec_mdl-pseudoinverse` ergeben damit

<!--en:77623d71-->
Die Ridge-Regularisierung dämpft die Richtungen mit kleinem $\sigma_i$ am stärksten. Gerade in diesen Richtungen verstärkt das unregularisierte Verfahren Datenfehler um $1/\sigma_i$. Wie sich die Regularisierung auf die Niveaumengen auswirkt, zeigt :numref:`fig_mdl-opt-conditioning-ellipse`. Beide Teilbilder sind um den jeweiligen Minimierer zentriert. Seine Verschiebung zum Ursprung ist deshalb nicht eingezeichnet.

<!--en:7782d3d5-->
![Niveaumengen der Zielfunktion der kleinsten Quadrate vor und nach Addition des Ridge-Terms $\tfrac\lambda2\|\mathbf{w}\|^2$. Damit beide Teilbilder lesbar bleiben, gilt hier $\kappa=16$ (mit $\kappa=1000$ aus dem durchgängigen Beispiel wäre nur eine Linie zu sehen). Der Strafterm erhöht jeden Eigenwert von $\mathbf{A}^\top\mathbf{A}$ um $\lambda$. Die Halbachsen sind proportional zu $1/\sqrt{\sigma_i^2+\lambda}$ und dort lang, wo die Krümmung klein ist. Das Tal rundet sich zu einer Schale. Orange: Iterierte des Gradientenabstiegs mit demselben Startpunkt und der jeweils besten festen Schrittweite. Grüne Pfeile: Hauptachsen mit den zugehörigen Krümmungen.](../img/mdl-opt-conditioning-ellipse.svg)

<!--en:d195e7b3-->
Dieselbe Konditionszahl $\kappa$ bestimmt die Schrittzahl des Gradientenabstiegs für die Ridge-Zielfunktion. Bei der besten festen Schrittweite verringert der Gradientenabstieg für eine quadratische Funktion mit Konditionszahl $\kappa>1$ den Fehler in jedem Schritt um den Faktor $\rho=(\kappa-1)/(\kappa+1)$ (:numref:`sec_mdl-gradient-based-optimization`). Beim Start im Ursprung beträgt der Fehler $-\mathbf w^\star$. Für den relativen Fehler gilt $\|\mathbf w_k-\mathbf w^\star\|/\|\mathbf w^\star\|\le\rho^k$ nach $k$ Schritten. Eine Toleranz $0<\mathrm{tol}<1$ erreichen wir also innerhalb von

<!--en:29cd6653-->
Die zweite Ungleichung folgt aus $\log\frac{\kappa+1}{\kappa-1}=2\,\mathrm{artanh}(1/\kappa)\ge2/\kappa$. Für sechs Stellen erlaubt sie $\tfrac12\log10^6=6.9$ Schritte je Einheit von $\kappa$. Bei mehr als zwei Dimensionen kann das Verfahren früher fertig sein (Aufgabe :exref:`ex_mdl-numerical-stability-conditioning-feature-scaling`).

<!--en:6070516e-->
Wie viele Schritte sparen wir bei gegebenem $\lambda$, und was kostet das? Beides messen wir an unserem durchgängigen Beispiel. Seine Zielfunktion $\tfrac12(w_1^2+1000w_2^2)-w_1-1000w_2$ entspricht einem Problem der kleinsten Quadrate mit $\mathbf A^\top\mathbf A=\mathrm{diag}(1,1000)$ und $\mathbf A^\top\mathbf b=(1,1000)$. Ihr Minimierer ist $(1,1)$. Am Ridge-Minimierer lassen sich daher die beiden Filterfaktoren direkt ablesen. Die Funktion `gd_iterations` führt den Gradientenabstieg mit der besten festen Schrittweite aus und liefert die Schrittzahl und den Ridge-Minimierer:

<!--en:b01eb3fd-->
Die Konditionszahl fällt von $1000$ auf $2$, wenn $\lambda$ von $0$ auf $1000$ steigt. Mit ihr fällt die Schrittzahl von $6{,}908$ auf $13$. Das Verhältnis der Schrittzahl zu $\kappa$ bleibt nahe bei $6.9$ (bei kleinem $\kappa$ ist die Näherung $2/\kappa$ grob, und das Aufrunden auf ganze Schritte fällt ins Gewicht). Die letzte Spalte zeigt den Preis der eingesparten Schritte. Bei $\lambda=1$ halbiert sich die Komponente in der langsamen Richtung mit $\sigma^2=1$ auf $0.5$, während sich die Komponente in der steifen Richtung mit $\sigma^2=1000$ nur um $0.1\%$ ändert. Bei $\lambda=1000$ halbiert sich auch die Komponente in der steifen Richtung. Variieren wir $\lambda$, sehen wir $\kappa$ und die Schrittzahl nebeneinander:

<!--en:b0662800-->
Auf doppelt logarithmischen Achsen verlaufen die Kurven parallel und unterscheiden sich um den Faktor $6.9$. Die Steigung liegt nahe $-1$, solange $\lambda$ deutlich zwischen den Eigenwerten $1$ und $1000$ liegt. Die Ridge-Regularisierung verbessert also die Kondition sowohl für das lineare Lösungsverfahren als auch für den Gradientenabstieg. In beiden Fällen verkleinert sie dafür die Lösung. Wie :numref:`subsec_mdl-weight-decay-duality` zeigt, entspricht der Strafterm $\tfrac\lambda2\|\mathbf w\|^2$ einer Nebenbedingung für $\|\mathbf w\|$. Ein aus statistischen Gründen gewähltes $\lambda$ bestimmt damit auch die Kondition der Rechnung.

<!--en:3ce0d1d9-->
Keines dieser Werkzeuge erfasst Fehler, die über viele Iterationsschritte wachsen, etwa bei einem Gradienten über hundert Schichten hinweg. Dies behandelt :numref:`sec_numerical_stability`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:0a12b873-->
Ein Gleitkommaformat speichert Zahlen mit etwa $p$ Bits relativer Genauigkeit und einem endlichen Exponentenbereich. Jede Rundung verursacht einen relativen Fehler von höchstens $u=2^{-p-1}$. Überlauf, Unterlauf und der Verlust kleiner Aktualisierungen sind dagegen absolute Effekte, die das relative Modell nicht erfasst. Softmax-Funktion und Kreuzentropie scheitern an nicht darstellbaren Zwischenwerten, die Varianzformel mit einem Durchlauf an der Subtraktion fast gleicher großer Zahlen. Keines der gesuchten Ergebnisse ist extrem. Die Verschiebung um das Maximum, die Berechnung der Kreuzentropie aus Logits und Welfords Rekursion vermeiden diese Umwege. Die relative Konditionszahl $\kappa_f=|xf'/f|$ unterscheidet schlechte Algorithmen von schwierigen Problemen: Ist sie klein und das Ergebnis ungenau, liegt es am Algorithmus. Bei einem linearen Gleichungssystem begrenzt $\kappa(\mathbf A)$ den Vorwärtsfehler auf das $\kappa$-Fache des Rückwärtsfehlers, den wir aus dem Residuum bestimmen. Es gehen also etwa $\log_{10}\kappa$ Stellen verloren, beim Umweg über die Normalgleichungen doppelt so viele. Die Ridge-Regularisierung addiert $\lambda$ zu jedem quadrierten Singulärwert. Das verringert $\kappa$ sowohl für das lineare Lösungsverfahren als auch für den Gradientenabstieg, verkleinert dafür aber die Komponenten der Lösung um die Filterfaktoren $\sigma_i^2/(\sigma_i^2+\lambda)$.

<!--en:2294243a-->
## Übungen

<!--en:54106c70-->
1. **Zulässiger Bereich für das Loss-Scaling.** Beim Loss-Scaling multiplizieren wir den Verlust mit einem Faktor $S$ und damit jeden Gradienten mit $S$. Nach der Umwandlung in fp32 teilen wir die Gradienten durch $S$. In fp16 ist die kleinste normale Zahl $2^{-14} \approx 6.1 \times 10^{-5}$, die kleinste subnormale Zahl $2^{-24} \approx 6.0 \times 10^{-8}$ (Werte unter $2^{-25}$ werden auf null gerundet) und die größte endliche Zahl $65504$. Angenommen, die von null verschiedenen Einträge des exakten Gradienten eines Netzes haben Beträge zwischen $10^{-9}$ und $10^{-1}$. Der Rückwärtsdurchlauf berechne die skalierten Gradienten in fp16.
    1. Bestimmen Sie alle Zweierpotenzen $S$, für die jeder skalierte Gradient eine normale fp16-Zahl ist.
    1. Die Gradienten eines tieferen Netzes haben Beträge zwischen $10^{-12}$ und $10^{-1}$. Zeigen Sie, dass kein Skalierungsfaktor alle im normalen Bereich hält. Wählen Sie die größte Zweierpotenz $S$, die einen Überlauf vermeidet. Wie viele signifikante Bits behält ein Gradient vom Betrag $10^{-12}$ nach der Skalierung, und wie groß ist sein relativer Rundungsfehler?
    1. Dynamisches Loss-Scaling (:numref:`subsec_loss_scaling`) wählt den größten Faktor, der einen Überlauf vermeidet, und nimmt dafür den Verlust der kleinsten Gradienten in Kauf. Warum ist es günstiger, das untere Ende des Bereichs zu opfern?
    1. Wie ändern sich die Antworten auf die ersten beiden Teilaufgaben, wenn die Gradienten in bf16 berechnet werden?
   :exlabel:`ex_mdl-numerical-stability-conditioning-loss-scale-window`

<!--en:622686bd-->
1. [code] **Liegt es am Problem oder am Algorithmus?** Werten Sie jeden der folgenden Ausdrücke in fp32 aus. Berechnen Sie jeweils die relative Konditionszahl :eqref:`eq_mdl-opt-kappa-f` der *Funktion* im angegebenen Bereich und den Auslöschungsfaktor :eqref:`eq_mdl-opt-cancellation-factor` der *Formel in der angegebenen Form*. Entscheiden Sie, ob der Fehler am Problem oder am Algorithmus liegt. Formen Sie die Formel so um, dass ihr Fehler der Empfindlichkeit des Problems entspricht. Prüfen Sie beide Formen anhand von fp64.
    1. Die betragskleinere Lösung von $ax^2 + bx + c = 0$ für $b > 0$ und $b^2 \gg 4|ac|$ als Funktion von $c$, zum Beispiel für $a = c = 1$ und $b = 10^4$. (Hinweis: Erweitern Sie mit dem konjugierten Ausdruck.)
    1. Die Funktion $\log(1 - e^{-x})$ für $x > 0$, die $\log(1 - q)$ aus $\log q = -x$ berechnet, an den Stellen $x \in \{10^{-6}, 10^{-3}, 1, 20\}$. Finden Sie je eine genaue Formulierung für kleine $x$ und für große $x$ und wählen Sie den Punkt, an dem Sie zwischen beiden wechseln.
    1. Die binäre Kreuzentropie eines Logits $t$ mit Label $1$, also $\log(1+e^{-t})$, für $t\in\{-800,-50,0,50\}$. Bei welchen dieser Werte läuft die angegebene Formel über, bei welchen ergibt sie $0$? Zeigen Sie, dass $\max(-t,0)+\mathrm{log1p}(e^{-|t|})$ denselben Wert hat, nie überläuft und keine Stellen verliert. Welchem Teil der beidseitigen Schranke für Log-Sum-Exp entspricht dieser Ausdruck?
    1. $\sqrt{x+1}-\sqrt x$ bei $x=10^8$.
   :exlabel:`ex_mdl-numerical-stability-conditioning-blame`

    *Teil 2 nach Martin Mächler, [Genaue Berechnung von log(1 − exp(−|a|))](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf), Abschnitt 2.*

<!--en:47f4de5d-->
1. [code] **Normalgleichungen in Gleitkommaarithmetik.** Die Normalgleichungen quadrieren die Konditionszahl :eqref:`eq_mdl-opt-kappa-squared`.
    1. Für $\tau > 0$ besteht die erste Zeile der $(m + 1) \times m$-Matrix $\mathbf{A}$ von :citet:`Lauchli.1961` aus Einsen. Darunter steht $\tau\mathbf{I}$. Berechnen Sie $\mathbf{A}^\top\mathbf{A}$, die Singulärwerte von $\mathbf{A}$ und $\kappa(\mathbf{A})$. Zeigen Sie, dass für $\tau = 10^{-8}$ die Matrix $\mathbf{A}^\top\mathbf{A}$ in fp64 auf eine singuläre Matrix gerundet wird. Wie viele korrekte Stellen sagt die Faustregel aus :numref:`subsec_mdl-conditioning-revisited` für ein QR-basiertes Verfahren der kleinsten Quadrate mit dieser Matrix $\mathbf{A}$ voraus, wenn $\mathbf b$ im Bild von $\mathbf A$ liegt? Wie ändert sich die Vorhersage bei großem Residuum?
    1. Ändern Sie die Zelle `#numerical-stability-conditioning-normal-equations` so, dass $\kappa(\mathbf{A}) = 10^k$ die Werte für $k = 2, \ldots, 10$ durchläuft. Tragen Sie die Anzahl der korrekten Stellen für die Normalgleichungen und für `lstsq` gegen $k$ auf, zusammen mit den beiden Geraden $16-2k$ und $16-k$ aus der Faustregel. Bei welchem $k$ sagt die Regel den Verlust aller korrekten Stellen für die Normalgleichungen voraus, und was beobachten Sie dort?

<!--en:8cfb82fe-->
1. [code] **Merkmalsskalierung als Vorkonditionierung.** In unterschiedlichen Einheiten gemessene Merkmale führen zu sehr unterschiedlichen Normen der Spalten einer Designmatrix. Sei $\mathbf{X}$ eine $200 \times 5$-Matrix mit standardnormalverteilten Einträgen. Weiter seien $\mathbf{A} = \mathbf{X}\,\mathrm{diag}(1, 3, 10, 30, 100)$ und $\mathbf{b} = \mathbf{A}\mathbf{w}^\star$ für einen zufälligen Vektor $\mathbf{w}^\star$.
    1. Wenden Sie die Funktion `gd_iterations` aus der Zelle `#numerical-stability-conditioning-ridge` mit den Argumenten $\mathbf A^\top\mathbf A$ und $\mathbf A^\top\mathbf b$ für diese Designmatrix bei $\lambda = 0$ an. Berechnen Sie $\kappa(\mathbf{A}^\top\mathbf{A})$ und vergleichen Sie die Schrittzahl mit der Schranke :eqref:`eq_mdl-opt-gd-iteration-bound`. Erklären Sie anhand der Komponenten des Anfangsfehlers entlang der Eigenvektoren zu den beiden extremalen Eigenwerten von $\mathbf A^\top\mathbf A$, warum die Schrittzahl unter der Schranke liegt.
    1. Teilen Sie jede Spalte von $\mathbf{A}$ durch ihre Norm und wiederholen Sie die Messung. Wie lässt sich der Minimierer des neu skalierten Problems auf $\mathbf{w}^\star$ zurückführen?
    1. Teil 2 von Aufgabe :exref:`ex_mdl-gradient-based-optimization-newton-show-that-any` in :numref:`sec_mdl-curvature-methods` beschreibt den Gradientenabstieg in neuen Koordinaten $\mathbf{u}$ mit $\mathbf{w} = \mathbf T\mathbf{u}$ als Gradientenabstieg für das ursprüngliche Problem mit einer positiv definiten Matrix $\mathbf P$ als Vorkonditionierer und der Aktualisierung $\mathbf{w} \leftarrow \mathbf{w} - \eta\,\mathbf{P}^{-1}\nabla f(\mathbf{w})$. Welche Matrizen $\mathbf T$ und $\mathbf P$ entsprechen der Spaltenskalierung? Wie hängt $\mathbf P$ mit der Diagonalen von $\mathbf{A}^\top\mathbf{A}$ zusammen?
   :exlabel:`ex_mdl-numerical-stability-conditioning-feature-scaling`

<!--en:4f1001d2-->
1. [code] **Stochastische Rundung.** ● Die stochastische Rundung :cite:`Gupta.Agrawal.Gopalakrishnan.ea.2015` rundet eine reelle Zahl $x$ zwischen benachbarten darstellbaren Werten $x_- < x_+$ auf $x_+$ mit Wahrscheinlichkeit $q = (x - x_-)/(x_+ - x_-)$ und andernfalls auf $x_-$.
    1. Zeigen Sie, dass der gerundete Wert den Erwartungswert $x$ und die Varianz $q(1 - q)\Delta^2 \le \Delta^2/4$ hat, wobei $\Delta = x_+ - x_-$ gilt.
    1. Ein bf16-Gewicht startet bei $w = 1$ und erhält $K = 1000$ Aktualisierungen um jeweils $-10^{-4}$, gefolgt von stochastischer Rundung. Das Gewicht bleibt in $[0.5, 1]$. Dort beträgt der bf16-Gitterabstand $\Delta = 2^{-8}$. Zeigen Sie, dass die Rundungsfehler verschiedener Schritte unkorreliert sind. Sagen Sie Mittelwert und Standardabweichung des Endgewichts voraus. Welches Endgewicht ergibt die Rundung zum nächsten Wert?
    1. Für eine positive fp32-Zahl können wir eine gleichverteilte zufällige ganze Zahl aus $\{0, \ldots, 2^{16} - 1\}$ zum Bitmuster addieren und danach die unteren 16 Bits löschen. Erklären Sie, warum das einer stochastischen Rundung auf bf16 entspricht. Implementieren Sie diese Rundung neben der Funktion `to_bf16` aus :numref:`subsec_mdl-floating-point` und prüfen Sie die Vorhersagen der vorigen Teilaufgabe in 1000 unabhängigen Durchläufen. Vergleichen Sie die Ergebnisse mit einer fp32-Hauptkopie, die dieselben Aktualisierungen erhält.
   :exlabel:`ex_mdl-numerical-stability-conditioning-stochastic-rounding`

<!--en:a75cd6c3-->
1. **Exakte Subtraktion.** ● Eine Subtraktion kann exakt sein und dennoch eine ungenaue Differenz liefern.
    1. Zeigen Sie, dass die Schranke :eqref:`eq_mdl-opt-cancellation-factor` für geeignete $\delta_a$ und $\delta_b$ mit $|\delta_a|,|\delta_b|\le u$ erreicht wird.
    1. Beweisen Sie das Lemma von Sterbenz :cite:`Sterbenz.1974`: Sind $a$ und $b$ Gleitkommazahlen im Format :eqref:`eq_mdl-opt-float-format` mit $b/2 \le a \le 2b$, dann ist $a - b$ eine Gleitkommazahl und die berechnete Differenz daher exakt. Nehmen Sie an, dass das Format subnormale Zahlen enthält.
    1. In der Zelle `#numerical-stability-conditioning-log1p` weicht die fp32-Differenz $a - b$ um etwa 20 Prozent vom wahren Wert $3.0 \times 10^{-7}$ ab. Prüfen Sie mit dem Lemma von Sterbenz und der ausgegebenen Anzahl von ulps, dass die Subtraktion exakt war. Berechnen Sie die relativen Fehler der gespeicherten Werte $a$ und $b$ und prüfen Sie, ob die Schranke aus der ersten Teilaufgabe den beobachteten Fehler abdeckt.
   :exlabel:`ex_mdl-numerical-stability-conditioning-exact-subtraction`

<!--en:7565325a-->
1. [code] **Zusammenführen laufender Momente.** ● Beim datenparallelen Training berechnen wir Statistiken für Teildatensätze und führen sie zusammen. Teildatensatz $A$ enthalte $k_A$ Werte mit Mittelwert $m_A$ und zentrierter Quadratsumme $M_A$. Teildatensatz $B$ enthalte $k_B$ Werte mit $m_B$ und $M_B$.
    1. Leiten Sie exakte Formeln für den Mittelwert $m$ und die zentrierte Quadratsumme $M$ der Vereinigung beider Teildatensätze in Abhängigkeit von $k_A$, $m_A$, $M_A$, $k_B$, $m_B$ und $M_B$ her.
    1. Zeigen Sie, dass sich im Fall $k_B = 1$ Welfords Rekursion :eqref:`eq_mdl-opt-welford` ergibt.
    1. Teilen Sie die $10^5$ Stichprobenwerte der Zelle `#numerical-stability-conditioning-welford` in $2^j$ annähernd gleich große Teildatensätze auf (`np.array_split`). Verwenden Sie $j \in \{1, 4, 8\}$. Berechnen Sie deren Statistiken mit Welfords Rekursion, führen Sie sie paarweise in einem Binärbaum zusammen und vergleichen Sie die Varianz mit der Referenz aus zwei Durchläufen. Führen Sie die Teildatensätze erneut zusammen, indem Sie ihre Summen $\sum_i x_i$ und $\sum_i x_i^2$ addieren und die Formel für einen Durchlauf anwenden. Erklären Sie den Unterschied.
   :exlabel:`ex_mdl-numerical-stability-conditioning-pairwise-merge`

<!--en:a1bf28f2-->
1. [code] **Wachstum des Summationsfehlers.** ● Sei $S_n = \sum_{i=1}^n x_i$ und $\hat S_n$ der berechnete Wert. Für die Summation von links nach rechts gilt im ungünstigsten Fall die Schranke $|\hat S_n - S_n| \le (n - 1)\,u \sum_i |x_i| + O(u^2)$.
    1. Für $n = 2^k$ teilt die rekursive paarweise Summation die Summanden in zwei Hälften, summiert jede Hälfte rekursiv und addiert die beiden Ergebnisse. Beweisen Sie die Schranke $|\hat S_n - S_n| \le u \log_2(n) \sum_i |x_i| + O(u^2)$.
    1. Speichern Sie $n \in \{2^4, 2^8, 2^{12}, 2^{16}, 2^{20}\}$ in $[0, 1)$ gleichverteilte Zufallszahlen in fp32. Messen Sie gegenüber einer float64-Referenz die relativen Fehler der Summation von links nach rechts (`np.cumsum(x)[-1]`), von `np.sum` (einer blockweisen paarweisen Summation) und von Kahans kompensierter Summation :cite:`Kahan.1965`. Kahans Verfahren führt eine Korrektur $c$ mit: Nach jeder Addition $t = s + y$ setzt es $c = (t - s) - y$ und subtrahiert $c$ vom nächsten Summanden. Tragen Sie die drei Fehler zusammen mit den beiden Schranken gegen $n$ auf.
    1. Wie wächst der gemessene Fehler der Summation von links nach rechts mit $n$ im Vergleich zur Schranke für den ungünstigsten Fall, und warum? Warum liegt die Summe von $10^7$ Exemplaren von $0.1$ im Text so viel näher am ungünstigsten Fall?
    1. Was begrenzt für große $n$ den Fehler der beiden genauen Verfahren?
   :exlabel:`ex_mdl-numerical-stability-conditioning-summation-error-growth`

    *Nach Steven G. Johnson, [MIT 18.335 Einführung in numerische Methoden, Aufgabenblatt 1](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/6e6730885b20bd046cbde805d15a4835_MIT18_335JS19_pset1.pdf), Aufgabe 4.*

<!--en:67278b54-->
1. [code] **Iterative Nachbesserung mit gemischter Genauigkeit.** ●● Beim Training mit gemischter Genauigkeit läuft der Großteil der Rechnungen mit geringer Genauigkeit. Größen, in denen sich kleine Korrekturen aufsummieren, behalten wir in hoher Genauigkeit. Die iterative Nachbesserung :cite:`Moler.1967,Carson.Higham.2018` überträgt diese Aufteilung auf $\mathbf{A}\mathbf{x} = \mathbf{b}$: Die Faktorisierung mit Aufwand $O(n^3)$ läuft mit geringer Genauigkeit. Nur die Berechnung des Residuums mit Aufwand $O(n^2)$ erfordert hohe Genauigkeit. Ausgehend von einer fp32-Lösung $\mathbf{x}_0$ wiederholen wir
   $$
   \mathbf{r}_k = \mathbf{b} - \mathbf{A}\mathbf{x}_k, \qquad
   (\mathbf{A} + \Delta\mathbf{A}_k)\,\mathbf{d}_k = \mathbf{r}_k, \qquad
   \mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{d}_k,
   $$
   Dabei berechnen wir Residuum und Aktualisierung in fp64. Die Korrektur $\mathbf{d}_k$ stammt aus der Lösung eines Gleichungssystems in fp32.
    1. Modellieren Sie das fp32-Verfahren als rückwärtsstabil, also $\|\Delta\mathbf{A}_k\| \le c\,u_{32}\,\|\mathbf{A}\|$ mit $u_{32} = 2^{-24}$ und einer moderaten Konstanten $c$. Vernachlässigen Sie die fp64-Rundungsfehler. Zeigen Sie für den Fehler $\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}$ die Beziehung $\mathbf{e}_{k+1} = (\mathbf{A} + \Delta\mathbf{A}_k)^{-1}
       \Delta\mathbf{A}_k\,\mathbf{e}_k$. Leiten Sie einen Kontraktionsfaktor in Abhängigkeit von $c\,\kappa(\mathbf{A})\,u_{32}$ her und geben Sie an, wann er unter $1$ liegt.
    1. Implementieren Sie die Iteration für die Hilbert-Matrizen der Zelle `#numerical-stability-conditioning-hilbert` mit $n = 3, \ldots, 10$. Berechnen Sie die Korrektur mit `np.linalg.solve` aus fp32-Kopien von $\mathbf{H}$ und $\mathbf{r}_k$. Erfassen Sie den Vorwärtsfehler über zehn Iterationen und vergleichen Sie seinen Endwert mit dem eines fp64-Verfahrens. Für welche $n$ erreicht die Nachbesserung die Genauigkeit der fp64-Lösung? Für welche $n$ konvergiert sie noch, aber zu langsam, um diese Genauigkeit innerhalb von zehn Iterationen zu erreichen? Für welche $n$ wächst der Fehler?
    1. Schätzen Sie für jedes $n$ die gemessene Kontraktion pro Iteration. Vergleichen Sie sie mit $\kappa(\mathbf{H})\,u_{32}$ und mit dem Spektralradius von $\mathbf{I} - \hat{\mathbf{H}}^{-1}\mathbf{H}$, wobei $\hat{\mathbf{H}}$ die fp32-Kopie von $\mathbf{H}$ ist und die Anwendung der Inversen in fp64 erfolgt. Welche Störung dominiert $\Delta\mathbf{A}_k$ in diesem Experiment, und wie pessimistisch ist die Schranke aus der ersten Teilaufgabe?

<!--en:351e4bc4-->
[Diskussion](https://d2l.discourse.group/t/36832)

<!--en:c2095750-->
Numerische Stabilität und Kondition<br>**IEEE-Gleitkommaarithmetik · Softmax-Funktion mit Verschiebung um das Maximum · katastrophale Auslöschung · Varianz nach Welford · Matrixkondition und Ridge-Regularisierung**

<!--en:0b96827f-->
::: {.slide title="Zwei Arten numerischer Fehler: NaN und unbemerkte Abweichungen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:18fe607c-->
- **Instabiler Algorithmus für ein einfaches Problem:** Die naive Softmax-Funktion für $\mathbf{z} = (1000, 1001, 1002)$ führt bei $e^{1000} = \infty \implies \text{NaN}$ zum Überlauf. Die Verschiebung um das Maximum $\mathbf{z} - \max(\mathbf{z})$ behebt das Problem vollständig.

<!--en:128cf7f9-->
- **Stabiles Verfahren für ein schlecht konditioniertes Problem:** Beim $12 \times 12$-Hilbert-System ($\kappa \approx 1.7 \times 10^{16}$) hat `np.linalg.solve` den Rückwärtsfehler $\omega \approx 10^{-16}$. Dennoch weicht das Ergebnis um $30\%$ ab. Das Problem erfordert eine Regularisierung.

<!--en:d9df6427-->
::: {.slide title="Binäres Gleitkommagitter und relativer Abstand"}

<!--en:591b2f21-->
[Gleitkommaarithmetik]{.kicker}

<!--en:8d1abec1-->
Eine normale Gleitkommazahl hat eine wissenschaftliche Darstellung zur Basis 2 mit $p$ Mantissenbits (Bits des Signifikanden) nach dem Komma und einem ganzzahligen Exponenten $e$:

<!--en:9a59973f-->
- **Konstanter relativer Abstand:** Jede Oktave $[2^e, 2^{e+1})$ enthält $2^p$ gleichmäßig verteilte Gitterpunkte mit Abstand $2^{e-p}$.

<!--en:3b6616cf-->
- **Maschinenepsilon und Rundungseinheit:** Der Abstand oberhalb von $1$ beträgt $\varepsilon_{\mathrm{mach}} = 2^{-p}$. Beim Runden einer reellen Zahl $x$ zum nächsten Gitterpunkt ändert sie sich höchstens um einen halben Gitterabstand. Somit gilt $\mathrm{fl}(x) = x(1 + \delta)$, und die Rundungseinheit $u$ begrenzt den relativen Fehler:
  $$|\delta| \le u = \tfrac12\varepsilon_{\mathrm{mach}} = 2^{-p-1}$$

<!--en:366c8863-->
::: {.slide title="Deep-Learning-Formate: dynamischer Bereich und Genauigkeit"}

<!--en:591b2f21-2-->
[Gleitkommaarithmetik]{.kicker}

<!--en:9b14c73c-->
Im heutigen Deep Learning stehen Mantissengenauigkeit ($p$) und dynamischer Bereich des Exponenten in fünf Formaten unterhalb von `fp64` ($p = 52$, $u \approx 1.1 \times 10^{-16}$) in unterschiedlichem Verhältnis:

<!--en:6ff83557-->
- **`fp16` und `bfloat16` im Vergleich:** `fp16` hat $p = 10$ Mantissenbits ($\sim 3$ Dezimalstellen), aber nur $5$ Exponentenbits, mit Überlauf bei $65{,}504$. `bfloat16` behält alle $8$ Exponentenbits von `fp32` (bis $3.4 \times 10^{38}$) und kürzt die Mantisse auf $p = 7$ Bits ($\varepsilon_{\mathrm{mach}} = 2^{-7} \approx 7.8 \times 10^{-3}$, $u = 2^{-8} \approx 3.9 \times 10^{-3}$).

<!--en:2033f2d9-->
::: {.slide title="Rundung zum nächsten geraden Nachbarn bei Gleichstand und Überlaufschwellen"}

<!--en:591b2f21-3-->
[Gleitkommaarithmetik]{.kicker}

<!--en:d335d38b-->
`bfloat16` besteht aus den oberen 16 Bits eines `fp32`-Worts. Bei der Umwandlung von `fp32` in `bfloat16` entfallen die unteren 16 Bits durch Rundung. Bei genauem Gleichstand (`0x8000`) gewinnt der Nachbar mit geradem letztem beibehaltenem Bit ($0$):

<!--en:35443918-->
::: {.slide title="Wann kleine Gradientenschritte ganz verschwinden"}

<!--en:591b2f21-4-->
[Gleitkommaarithmetik]{.kicker}

<!--en:695c6b71-->
Multiplikation und Division folgen dem relativen Modell $\mathrm{fl}(x \odot y) = (x \odot y)(1 + \delta)$. Bei der Addition einer kleinen Aktualisierung $\Delta = -\eta g$ zu einem Gewicht $w \in (2^e, 2^{e+1})$ greift eine scharfe absolute Schwelle, sobald $|\eta g|$ kleiner ist als die Hälfte des Gitterabstands $2^{e-p}$ bei $w$:

<!--en:c6d24978-->
In `bfloat16` ($p = 7$) bleibt ein Gewicht bei $w = 1$ für jede Verringerung unter $2^{-9} \approx 0.00195$ unverändert. Bei $\eta g = 10^{-3}$ bewegt es sich also überhaupt nicht. Training mit gemischter Genauigkeit sammelt die Aktualisierungen deshalb in einer **`fp32`-Hauptkopie der Gewichte** (oder verwendet erwartungstreue **stochastische Rundung**).

<!--en:b90f940c-->
::: {.slide title="Verschiebungsinvarianz und Softmax-Funktion mit Verschiebung um das Maximum"}

<!--en:72b516e2-->
[Stabile Softmax-Funktion]{.kicker}

<!--en:76d9a7e6-->
Bei direkter Auswertung von $\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$ kommt es zum Überlauf, sobald ein Logit größer als $\log(\text{max float})$ ist ($11.09$ in `fp16`, $88.72$ in `fp32`, $709.78$ in `fp64`). Die Multiplikation von Zähler und Nenner mit $e^{-c}$ lässt die Softmax-Funktion für jedes $c \in \mathbb{R}$ unverändert:

<!--en:1604dad1-->
- **Warum $c = \max_j z_j$ optimal ist:** Für jeden verschobenen Exponenten gilt $z_j - c \le 0$, ein Überlauf ist also unmöglich. Der größte Term ist $e^0 = 1$. Die Summe im Nenner liegt damit in $[1, n]$ und kann durch Unterlauf nie zu $0$ werden.

<!--en:3eac45ac-->
::: {.slide title="Log-Sum-Exp als weiches Maximum mit beidseitiger Schranke"}

<!--en:72b516e2-2-->
[Stabile Softmax-Funktion]{.kicker}

<!--en:61daf60d-->
Mit $c = \max_j z_j$ in $\mathrm{lse}(\mathbf{z}) = c + \log\sum_{j=1}^n e^{z_j - c}$ folgt, dass $\mathrm{lse}(\mathbf{z})$ eine glatte obere Hülle des harten Maximums ist:

<!--en:8266eafe-->
- **Beweis:** Für jeden Term gilt $e^{z_j - c} \in (0, 1]$, und der größte Term ist gleich $1$. Daher liegt die Summe in $[1, n]$ und ihr Logarithmus in $[0, \log n]$.

<!--en:6b5b663a-->
- **Gradient und Kreuzentropie:** Es gilt $\nabla\mathrm{lse}(\mathbf{z}) = \mathrm{softmax}(\mathbf{z})$. Für $n = 2$ mit $\mathbf{z} = (0, t)$ ist $\mathrm{lse}(0, t) = \log(1 + e^t)$ die **Softplus**-Aktivierung (Abstand höchstens $\log 2 \approx 0.693$ von $\mathrm{ReLU}(t) = \max(0, t)$).

<!--en:c942f1e2-->
::: {.slide title="Klammerung von Log-Softmax zur Vermeidung von Rundungsverlusten"}

<!--en:72b516e2-3-->
[Stabile Softmax-Funktion]{.kicker}

<!--en:53a24fa2-->
Zur Berechnung der Log-Wahrscheinlichkeiten $\log\mathrm{softmax}(\mathbf{z})_i$ ohne Unterlauf kleiner Wahrscheinlichkeiten zwei algebraisch identische Formeln mit $c = \max_j z_j$ und $s = \log\sum_{j=1}^n e^{z_j - c} \in [0, \log n]$ vergleichen:

<!--en:76a28ee6-->
- **Wo verliert $z_i - \mathrm{lse}(\mathbf{z})$ an Genauigkeit?** Nahe $c = 1002$ haben die `fp32`-Gitterpunkte den Abstand $2^{-14} \approx 6.1 \times 10^{-5}$. Beim Bilden von $c + s = 1002.4076$ wird $s$ auf das grobe Gitter bei $1002$ gerundet. Dabei gehen $1.32 \times 10^{-5}$ verloren, noch vor der Differenzbildung mit $z_i$.

<!--en:a5a110bf-->
::: {.slide title="Kreuzentropie aus Logits und Unterlauf der Wahrscheinlichkeit" only="pytorch"}

<!--en:72b516e2-4-->
[Stabile Softmax-Funktion]{.kicker}

<!--en:9e4dbcaf-->
Der Kreuzentropieverlust für die wahre Klasse $y$ ist $-\log\mathrm{softmax}(\mathbf{z})_y = \mathrm{lse}(\mathbf{z}) - z_y$. Für $\mathbf{z} = (0, t)$ mit wahrem Label $y = 0$ beträgt der exakte Verlust $\log(1 + e^t) \approx t$. Bei vorheriger Berechnung der Wahrscheinlichkeiten mit `softmax` fällt $p_0 \approx e^{-t}$ oberhalb von $t = 87.3$ in den subnormalen Bereich und auf $0$ bei $t = 104$:

<!--en:a3b4d8de-->
Bei $t = 103$ rundet PyTorch $e^{-103} \approx 1.32 \times 2^{-149}$ auf die kleinste subnormale `fp32`-Zahl $2^{-149}$ (nur 1 Bit bleibt erhalten) und liefert den Verlust $149\log 2 = 103.2789$. Bei $t = 104$ gelten $p_0 = 0$ und $-\log p_0 = \infty$, während `F.cross_entropy` mit Logits exakt ist.

<!--en:559bdcd6-->
::: {.slide title="Kreuzentropie aus Logits und Unterlauf der Wahrscheinlichkeit" only="mxnet"}

<!--en:72b516e2-5-->
[Stabile Softmax-Funktion]{.kicker}

<!--en:26d1b3ac-->
Der Kreuzentropieverlust für die wahre Klasse $y$ ist $-\log\mathrm{softmax}(\mathbf{z})_y = \mathrm{lse}(\mathbf{z}) - z_y$. Für $\mathbf{z} = (0, t)$ mit wahrem Label $y = 0$ beträgt der exakte Verlust $\log(1 + e^t) \approx t$. Bei vorheriger Berechnung der Wahrscheinlichkeiten mit `softmax` fällt $p_0 \approx e^{-t}$ unter die untere Grenze des normalen `fp32`-Bereichs bei $t > 87.3$:

<!--en:771d0a75-->
Da MXNet subnormale Ergebnisse der Exponentialfunktion direkt auf $0$ setzt, verschwindet die Zwischenwahrscheinlichkeit $p_0$. Der Verlust $-\log p_0$ springt schon auf `inf` bei $t = 103$, während `-npx.log_softmax(logits)` exakt bleibt.

<!--en:dae12fc6-->
::: {.slide title="Kreuzentropie aus Logits und Unterlauf der Wahrscheinlichkeit" only="tensorflow"}

<!--en:72b516e2-6-->
[Stabile Softmax-Funktion]{.kicker}

<!--en:0c523279-->
Der Kreuzentropieverlust für die wahre Klasse $y$ ist $-\log\mathrm{softmax}(\mathbf{z})_y = \mathrm{lse}(\mathbf{z}) - z_y$. Für $\mathbf{z} = (0, t)$ mit wahrem Label $y = 0$ beträgt der exakte Verlust $\log(1 + e^t) \approx t$. Vergleich der direkten Berechnung aus Logits mit der Berechnung aus Wahrscheinlichkeiten:

<!--en:3a58adf3-->
TensorFlow setzt subnormale Wahrscheinlichkeiten auf $0$ (Verlust `inf` bei $t = 103$). Keras begrenzt bei `from_logits=False` die Wahrscheinlichkeiten sogar nach unten auf $10^{-7}$. Der Verlust verharrt bei $-\log 10^{-7} = 16.1181$ mit **Gradient null** für alle $t > 16$.

<!--en:31cbe8e9-->
::: {.slide title="Kreuzentropie aus Logits und Unterlauf der Wahrscheinlichkeit" only="jax"}

<!--en:72b516e2-7-->
[Stabile Softmax-Funktion]{.kicker}

<!--en:04b69c67-->
Der Kreuzentropieverlust für die wahre Klasse $y$ ist $-\log\mathrm{softmax}(\mathbf{z})_y = \mathrm{lse}(\mathbf{z}) - z_y$. Für $\mathbf{z} = (0, t)$ mit wahrem Label $y = 0$ beträgt der exakte Verlust $\log(1 + e^t) \approx t$. Bei vorheriger Berechnung der Wahrscheinlichkeiten mit `softmax` fällt $p_0 \approx e^{-t}$ unter den normalen `fp32`-Bereich bei $t > 87.3$:

<!--en:02fedcd4-->
Da XLA subnormale Gleitkommazahlen auf null setzt, liefert `jax.nn.softmax` die Wahrscheinlichkeit $p_0 = 0$ und damit den Verlust $-\log p_0 = \infty$ schon bei $t = 103$. Dagegen ist `optax.softmax_cross_entropy_with_integer_labels` mit den ursprünglichen Logits exakt.

<!--en:6a8cbd16-->
::: {.slide title="Relative Konditionszahl einer mathematischen Funktion"}

<!--en:551086e2-->
[Katastrophale Auslöschung]{.kicker}

<!--en:75debe66-->
Die **relative Konditionszahl** $\kappa_f(x)$ unterscheidet einen schlechten Algorithmus von einem grundsätzlich empfindlichen mathematischen Problem. Sie misst, um welchen Faktor sich eine kleine relative Eingabestörung $\delta$ im relativen Ausgabefehler $\frac{f(x(1+\delta)) - f(x)}{f(x)} \approx \frac{x f'(x)}{f(x)}\delta$ verstärkt:

<!--en:034a9f70-->
| Mathematisches Problem $f$ | Relative Konditionszahl $\kappa_f$ | Empfindlicher Bereich |

<!--en:e87ef20d-->
| **Subtraktion** $a - b$ | $(|a| + |b|) / |a - b|$ | Schlecht konditioniert für $a \approx b$ (**katastrophale Auslöschung**) |

<!--en:df57ba83-->
| **Verschobener Logarithmus** $\log(1 + x)$ | $\left|\frac{x}{(1+x)\log(1+x)}\right|$ | Gut konditioniert nahe $x = 0$ ($\kappa_f \to 1$ für $x \to 0$) |

<!--en:03deb5c8-->
| **Exponentialfunktion** $e^x$ | $|x|$ | Schlecht konditioniert für große $|x|$ (absoluter Fehler in $x$ $\to$ relativer Fehler) |

<!--en:a593e7e3-->
| **Stichprobenvarianz** (Mittelwert $m$, Standardabweichung $s$) | $\le 2(1 + |m|/s)$ | Verliert $\approx \log_{10}(|m|/s)$ Stellen für $s \ll |m|$ |

<!--en:c95c1374-->
::: {.slide title="Instabiler Algorithmus oder schwieriges Problem?"}

<!--en:551086e2-2-->
[Katastrophale Auslöschung]{.kicker}

<!--en:0dca5efc-->
- **Gut konditioniertes Problem:** $f(x) = \log(1 + x)$ hat nahe $x = 0$ die Konditionszahl $\kappa_f(x) = \left|\frac{x}{(1+x)\log(1+x)}\right| \to 1$.

<!--en:858d0631-->
- **Instabiler Zwischenschritt:** Bei der Auswertung von `log(1.0 + x)` für $x = 10^{-8}$ in `fp32` ergibt die Rundung $y = \mathrm{fl}(1 + 10^{-8}) = 1.0$. Für $\log(y)$ nahe $y = 1$ gilt $\kappa_{\log}(y) = 1/|\log y| = 10^8$. Dadurch wächst der Rundungsfehler von $10^{-8}$ auf $100\%$.

<!--en:6505bb90-->
- **Direkte Reihenentwicklung (`log1p`):** Berechnet $x - x^2/2 + \cdots$ direkt mit voller Maschinengenauigkeit.

<!--en:0baf3579-->
::: {.slide title="Lemma von Sterbenz und Auslöschung bei gerundeten Eingaben"}

<!--en:551086e2-3-->
[Katastrophale Auslöschung]{.kicker}

<!--en:b0609ddc-->
Die Subtraktion der fast gleichen `fp32`-Zahlen $a = 1.0002344$ und $b = 1.0002341$ (wahre Differenz $3.0 \times 10^{-7}$) ergibt $2.384 \times 10^{-7}$. Der Fehler beträgt $20\%$:

<!--en:bec4deb2-->
- **Lemma von Sterbenz (1974):** Erfüllen zwei Gleitkommazahlen $b/2 \le a \le 2b$, so ist ihre Differenz $a - b$ *exakt darstellbar*. Der Rundungsfehler ist null.

<!--en:a7388e42-->
- **Warum das Ergebnis trotzdem um $20\%$ abweicht:** Die Subtraktion selbst war exakt. Die gemeinsamen führenden Bits heben sich auf und legen die Rundungsfehler offen, die schon beim Speichern von $a$ und $b$ auf dem $2^{-23} \approx 1.19 \times 10^{-7}$-Gitter entstanden.

<!--en:90492529-->
::: {.slide title="Abbruchfehler und Auslöschung bei endlichen Differenzen" layout="tall-output"}

<!--en:551086e2-4-->
[Katastrophale Auslöschung]{.kicker}

<!--en:059b19c9-->
Bei der numerischen Gradientenprüfung mit dem Vorwärtsdifferenzenquotienten $\frac{f(x + h) - f(x)}{h}$ lässt sich die Subtraktion fast gleicher Zahlen nicht algebraisch beseitigen:

<!--en:5c938723-->
- **Optimale Schrittweite $h^\star = 2\sqrt{\frac{u|f(x)|}{|f''(x)|}}$:** Das Minimum der Summe ergibt den kleinsten Fehler $\approx 2\sqrt{u|f(x)f''(x)|} \propto \sqrt{u}$. Nur **die Hälfte der Stellen** des Formats bleibt erhalten ($\sim 8$ Stellen bei $h^\star \approx 10^{-8}$ in `fp64` gegenüber $\sim 3.5$ Stellen bei $h^\star \approx 5 \times 10^{-4}$ in `fp32`).

<!--en:d8b0ba63-->
::: {.slide title="Warum die naive Varianzformel die Konditionszahl quadriert"}

<!--en:551086e2-5-->
[Katastrophale Auslöschung]{.kicker}

<!--en:5d8a02c6-->
Normalisierungsschichten und Datenverarbeitung berechnen laufend Stichprobenvarianzen $s^2 = \frac{1}{N}\sum_{i=1}^N (x_i - m)^2$. Ist die Standardabweichung $s$ klein gegenüber dem Mittelwert $|m|$, lohnt der Vergleich zwischen der Empfindlichkeit des Problems und der Lehrbuchformel für einen Durchlauf:

<!--en:7e1f2f73-->
- **Konditionszahl des Varianzproblems:** Eine Störung jeder Eingabe gemäß $x_i \mapsto x_i(1 + \delta_i)$ mit $|\delta_i| \le u$ ändert die Varianz um $\frac{2}{N}\sum_i (x_i - m)x_i\delta_i$. Daraus folgt die relative Konditionszahl $\kappa_{\mathrm{var}} \le 2(1 + |m|/s) \approx 2|m|/s$. Für $m = 10^9, s = 1$ in `fp64` ($u \approx 1.1 \times 10^{-16}$) erlaubt das Problem $\approx 7$ korrekte Stellen ($\kappa_{\mathrm{var}} u \approx 2 \times 10^{-7}$).

<!--en:ec042b93-->
- **Katastrophale Auslöschung in $\frac{1}{N}\sum x_i^2 - m^2$:** Beide Terme haben den Betrag $\approx m^2 = 10^{18}$, während ihre Differenz $s^2 = 1$ beträgt. Die Konditionszahl der Subtraktion ist also $\frac{2m^2}{s^2} = 2 \times 10^{18} > 1/u$. Damit **quadriert sich die Konditionszahl**, und jede einzelne Stelle geht verloren.

<!--en:91d77491-->
::: {.slide title="Welfords stabile Rekursion für die Varianz in einem Durchlauf"}

<!--en:551086e2-6-->
[Katastrophale Auslöschung]{.kicker}

<!--en:125ddb9f-->
**Welfords Algorithmus (1962)** aktualisiert den laufenden Mittelwert $m_k$ und die zentrierte Quadratsumme $M_k = \sum_{i=1}^k(x_i - m_k)^2$ in einem einzigen Durchlauf durch die Daten:

<!--en:a2c70e62-->
- **Keine Auslöschung:** Jeder Zuwachs $d_k(x_k - m_k)$ beruht auf zentrierten Abweichungen und hat die Größenordnung $s^2 = O(1)$ statt $m^2 = 10^{18}$. So erreicht das Verfahren den optimalen Fehler $O(u|m|/s)$.

<!--en:7488d29d-->
::: {.slide title="Naive Varianzformel und Welfords Verfahren bei Mittelwert 10⁹"}

<!--en:551086e2-7-->
[Katastrophale Auslöschung]{.kicker}

<!--en:e8c2d166-->
Für $N = 10^5$ `fp64`-Stichprobenwerte aus $\mathcal{N}(10^9, 1)$ stehen drei Verfahren zum Vergleich: die naive Formel für einen Durchlauf $\mathbb{E}[x^2] - (\mathbb{E}[x])^2$, Welfords Rekursion mit einem Durchlauf und die Referenz mit zwei Durchläufen:

<!--en:89ba5b2a-->
- Wegen $10^{18} \in [2^{59}, 2^{60})$ haben benachbarte `fp64`-Zahlen den Abstand $2^{59-52} = 128$. Die naive Formel kann also nur ganzzahlige Vielfache von $128$ liefern (oft $0$ oder negative Werte). Welfords Ergebnis stimmt dagegen in $7+$ Stellen mit der Referenz aus zwei Durchläufen überein.

<!--en:0fc65029-->
::: {.slide title="Summationsreihenfolge: von links nach rechts oder paarweise im Baum"}

<!--en:c9e670ba-->
[Stabilität der Summation]{.kicker}

<!--en:78f4d6b4-->
Die Gleitkommaaddition ist nicht assoziativ. Beim Summieren von $n$ Zahlen bestimmt der Summationsbaum, wie viele Rundungen jeder Summand durchläuft:

<!--en:14671154-->
- **Warum die Summe von links nach rechts um $+8.8\%$ abweicht, wenn sie $10^7$ Exemplare von $0.1$ addiert:** Überschreitet die laufende `fp32`-Summe $2^{19} = 524{,}288$, beträgt der Gitterabstand $1/16 = 0.0625$. Jeder Summand $+0.1$ wird also auf zwei Gitterschritte ($+0.125$) aufgerundet. Paarweise Summation hält die Teilsummen in derselben Größenordnung und verkleinert die Fehlerschranke um den Faktor $O(n / \log_2 n)$.

<!--en:1edcf4ae-->
::: {.slide title="Geometrische Deutung der Kondition linearer Gleichungssysteme"}

<!--en:3243a144-->
[Kondition von Matrizen]{.kicker}

<!--en:c2e6d914-->
Gegeben sei ein lineares Gleichungssystem $\mathbf{A}\mathbf{x} = \mathbf{b}$. Seine beiden Gleichungen $x_1 + x_2 = 2$ und $x_1 + 1.0001\,x_2 = 2.0001$ beschreiben fast parallele Geraden mit Schnittpunkt $\mathbf{x} = (1, 1)$:

<!--en:46f1c087-->
- **Extreme Empfindlichkeit:** Eine Änderung von $b_2$ um $+10^{-4}$ (relative Datenänderung $3.5 \times 10^{-5}$) verschiebt den Schnittpunkt nach $\hat{\mathbf{x}} = (0, 2)$. Das ergibt einen **Vorwärtsfehler von $100\%$**.

<!--en:1e8ee805-->
- **Vorwärts- und Rückwärtsfehler:** Für das ursprüngliche System hat $\hat{\mathbf{x}} = (0, 2)$ einen **Vorwärtsfehler** von $100\%$. Der **Rückwärtsfehler** (die kleinste relative Datenstörung, für die $\hat{\mathbf{x}}$ exakt ist) beträgt aber nur $3.5 \times 10^{-5}$.

<!--en:8c973a5d-->
::: {.slide title="Rückwärtsfehler, Residuen und Rückwärtsstabilität"}

<!--en:3243a144-2-->
[Kondition von Matrizen]{.kicker}

<!--en:0cf9bcf7-->
Ein Algorithmus für $\mathbf{A}\mathbf{x} = \mathbf{b}$ ist **rückwärtsstabil**, wenn die berechnete Lösung $\hat{\mathbf{x}}$ ein benachbartes System $(\mathbf{A} + \Delta\mathbf{A})\hat{\mathbf{x}} = \mathbf{b}$ exakt löst und der relative Rückwärtsfehler $\omega = \|\Delta\mathbf{A}\|_2 / \|\mathbf{A}\|_2 = O(u)$ erfüllt:

<!--en:b0e85cb9-->
- **Exakte Formel für den Rückwärtsfehler aus dem Residuum $\mathbf{r} = \mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$:**
  - Für jede Störung mit $(\mathbf{A} + \Delta\mathbf{A})\hat{\mathbf{x}} = \mathbf{b}$ muss $\Delta\mathbf{A}\hat{\mathbf{x}} = \mathbf{b} - \mathbf{A}\hat{\mathbf{x}} = \mathbf{r}$ gelten. Daher folgt $\|\mathbf{r}\|_2 \le \|\Delta\mathbf{A}\|_2\,\|\hat{\mathbf{x}}\|_2$.
  - Die Rang-eins-Störung $\Delta\mathbf{A}^\star = \frac{\mathbf{r}\hat{\mathbf{x}}^\top}{\|\hat{\mathbf{x}}\|_2^2}$ erreicht Gleichheit mit der Spektralnorm $\|\Delta\mathbf{A}^\star\|_2 = \frac{\|\mathbf{r}\|_2}{\|\hat{\mathbf{x}}\|_2}$. Damit beträgt der relative Rückwärtsfehler genau
    $$\omega = \min_{(\mathbf{A} + \Delta\mathbf{A})\hat{\mathbf{x}} = \mathbf{b}} \frac{\|\Delta\mathbf{A}\|_2}{\|\mathbf{A}\|_2} = \frac{\|\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}\|_2}{\|\mathbf{A}\|_2\,\|\hat{\mathbf{x}}\|_2}$$

<!--en:95e29308-->
- **Gütekriterium für das Lösungsverfahren:** Ein kleines relatives Residuum $\omega \approx u$ belegt, dass das Verfahren bis auf Maschinengenauigkeit gearbeitet hat, unabhängig von der Größe des Vorwärtsfehlers.

<!--en:5f2e81ea-->
::: {.slide title="Schranke für den Vorwärtsfehler und Faustregel für den Stellenverlust"}

<!--en:3243a144-3-->
[Kondition von Matrizen]{.kicker}

<!--en:05b09151-->
Mit relativem Rückwärtsfehler $\omega = \frac{\|\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}\|_2}{\|\mathbf{A}\|_2\|\hat{\mathbf{x}}\|_2}$ und $\kappa(\mathbf{A}) = \sigma_1/\sigma_d$ gilt:

<!--en:91de36b3-->
- **Faustregel für den Stellenverlust:** Bei jedem rückwärtsstabilen Verfahren ($\omega \approx u$) behält $\hat{\mathbf{x}}$ ungefähr $\log_{10}(1/u) - \log_{10}\kappa(\mathbf{A})$ korrekte Dezimalstellen.

<!--en:2f446046-->
::: {.slide title="Wo bei der Hilbert-Matrix Stellen verloren gehen"}

<!--en:3243a144-4-->
[Kondition von Matrizen]{.kicker}

<!--en:0c43be50-->
Für die Hilbert-Matrix $H_{ij} = \frac{1}{i + j - 1}$ wächst $\kappa(\mathbf{H})$ wie $O\big((1 + \sqrt{2})^{4n}/\sqrt{n}\big)$. Das Lösen von $\mathbf{H}\mathbf{x} = \mathbf{b}$ ($\mathbf{b} = \mathbf{H}\mathbf{1}$, wahre Lösung $\mathbf{1}$) in `fp64` und mit exakter rationaler Arithmetik (`Fraction`) zeigt, wo die Stellen verloren gehen:

<!--en:2f88cc97-->
- Bei $n = 12$ ($\kappa \approx 1.7 \times 10^{16}$) hat `np.linalg.solve` den Rückwärtsfehler $\omega \approx 10^{-16}$. Selbst eine *exakte* rationale Lösung des gespeicherten `fp64`-Systems verfehlt $\mathbf{1}$ um $14\%$. Das entspricht der Größenordnung des $30\%$-Vorwärtsfehlers. Die Stellen gingen schon beim Runden von $\mathbf{b} = \mathbf{H}\mathbf{1}$ auf `fp64` verloren.

<!--en:a3b5fe35-->
::: {.slide title="Warum die Normalgleichungen κ(A) quadrieren"}

<!--en:db5e899a-->
[Kondition bei kleinsten Quadraten]{.kicker}

<!--en:8f0b087a-->
Beim Lösen des linearen Problems der kleinsten Quadrate $\min_{\mathbf{w}}\|\mathbf{A}\mathbf{w} - \mathbf{b}\|_2^2$ ($\mathbf{A} \in \mathbb{R}^{n \times d}$ mit vollem Spaltenrang) quadriert das Bilden der Normalgleichungen $\mathbf{A}^\top\mathbf{A}\mathbf{w} = \mathbf{A}^\top\mathbf{b}$ die Singulärwerte $\sigma_i(\mathbf{A}^\top\mathbf{A}) = \sigma_i(\mathbf{A})^2$ und **die Konditionszahl**:

<!--en:294e5b76-->
- Bei $\kappa(\mathbf{A}) = 10^6$ in `fp64` hat das System $\mathbf{A}^\top\mathbf{A}\mathbf{w} = \mathbf{A}^\top\mathbf{b}$ die Konditionszahl $\kappa(\mathbf{A}^\top\mathbf{A}) = 10^{12}$. Nur $\sim 5$ Stellen bleiben erhalten. SVD/QR (`np.linalg.lstsq`) arbeitet direkt mit $\mathbf{A}$ und behält $\sim 11$ bis $12$ Stellen.

<!--en:11551a76-->
::: {.slide title="Ridge-Regularisierung hebt den unteren Rand des Spektrums an"}

<!--en:55a5d972-->
[Regularisierung und Kondition]{.kicker}

<!--en:a4d05eeb-->
Ist $\kappa(\mathbf{A})$ selbst sehr groß, kann kein Lösungsverfahren die Verstärkung des Datenrauschens verhindern. Wir müssen das Problem regularisieren. Die Addition von $\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$ erhöht jeden Eigenwert von $\mathbf{A}^\top\mathbf{A}$ um $\lambda > 0$:

<!--en:254ef1e2-->
- **SVD-Filterfaktoren:** Mit $\mathbf{w}_0 = \mathbf{A}^+\mathbf{b}$ gilt
  $$\mathbf{w}_\lambda = \sum_{i=1}^d \frac{\sigma_i^2}{\sigma_i^2 + \lambda}\,(\mathbf{v}_i^\top\mathbf{w}_0)\,\mathbf{v}_i$$
  Die Ridge-Regularisierung dämpft verrauschte Richtungen ($\sigma_i^2 \ll \lambda$) und erhält starke Richtungen ($\sigma_i^2 \gg \lambda$).

<!--en:8880a73f-->
::: {.slide title="Wie die Ridge-Regularisierung zugleich den Gradientenabstieg beschleunigt"}

<!--en:55a5d972-2-->
[Regularisierung und Kondition]{.kicker}

<!--en:52db86eb-->
Die Konditionszahl $\kappa(\lambda) = \frac{\sigma_1^2 + \lambda}{\sigma_d^2 + \lambda}$ bestimmt *sowohl* die numerische Empfindlichkeit des linearen Gleichungssystems als auch die Iterationszahl $k \le \big\lceil\frac{\kappa(\lambda)}{2}\log\frac{1}{\mathrm{tol}}\big\rceil$ des Gradientenabstiegs bei $\eta^\star = \frac{2}{L + \mu}$:

<!--en:74d63f4c-->
- Für unser Vergleichsproblem $\mathbf{A}^\top\mathbf{A} = \mathrm{diag}(1, 1000)$ mit wahrem unregularisiertem Minimierer $\mathbf{w}_0 = (1, 1)$ senkt $\lambda = 100$ die Konditionszahl $\kappa$ von $1000$ auf $10.9$. Die Schrittzahl des Gradientenabstiegs fällt von $6{,}908$ auf $76$. Dafür entsteht eine Verzerrung: Die Koordinate in der flachen Richtung schrumpft um den Filterfaktor $\frac{1}{1 + 100} \approx 0.0099$.

<!--en:3b09d6e8-->
::: {.slide title="Konditionszahl und Schrittzahl des Gradientenabstiegs auf logarithmischen Achsen"}

<!--en:55a5d972-3-->
[Regularisierung und Kondition]{.kicker}

<!--en:3f15247e-->
Variiert $\lambda$ von $10^{-2}$ bis $10^3$ für $\mathbf{A}^\top\mathbf{A} = \mathrm{diag}(1, 1000)$, folgt die Schrittzahl des Gradientenabstiegs bis zum relativen Fehler $10^{-6}$ der Konditionszahl $\kappa(\lambda) = \frac{1000 + \lambda}{1 + \lambda}$ über drei Größenordnungen:

<!--en:2144e451-->
- Auf doppelt logarithmischen Achsen verlaufen die Kurven parallel. Sie unterscheiden sich um den konstanten Faktor $\tfrac12\ln(10^6) \approx 6.9$ Schritte je Einheit von $\kappa$.

<!--en:c7bfaa81-->
::: {.slide title="Verständnisfragen: Genauigkeit, Auslöschung und κ"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:0a026afb-->
Fragen zur Gleitkommaarithmetik, zur Auslöschung und zur Kondition:

<!--en:a6d48ecb-->
1. Ein Parameter hat den Wert $w = 1.0$, und der Gradientenabstieg wendet die Aktualisierung $\eta g = 10^{-3}$ an. Ändert sich der gespeicherte Parameter in `bfloat16` ($p = 7$)? Wie ist es in `fp16` ($p = 10$)?

<!--en:7ecff7c9-->
2. Warum versagt in `fp64` ($u \approx 1.1 \times 10^{-16}$) die Varianzformel mit einem Durchlauf $\frac{1}{N}\sum x_i^2 - m^2$ für Daten mit Mittelwert $m = 10^9$ und Standardabweichung $s = 1$ vollständig, während Welfords Algorithmus in einem Durchlauf $\sim 7$ Stellen erhält?

<!--en:4fc68537-->
3. Eine Designmatrix $\mathbf{A}$ hat die Konditionszahl $\kappa(\mathbf{A}) = 10^6$, und das Residuum ist null. Wie viele korrekte Dezimalstellen behalten die Normalgleichungen $\mathbf{A}^\top\mathbf{A}\mathbf{w} = \mathbf{A}^\top\mathbf{b}$ und wie viele QR/SVD in `fp64` ($\sim 16$ Stellen) ungefähr?

<!--en:334b7de8-->
**Lösungen:** **(1) `bfloat16`: Nein.** Eine Verringerung ausgehend von $w = 1$ erfordert $|\eta g| > 2^{-p-2} = 2^{-9} \approx 0.00195 > 10^{-3}$, also gilt $\mathrm{fl}(1 - 10^{-3}) = 1$. **`fp16`: Ja.** Die Schwelle unterhalb von $1$ beträgt $2^{-12} \approx 2.44 \times 10^{-4} < 10^{-3}$. Der neue Wert ist $0.99902$. **(2)** Für das Varianzproblem gilt $\kappa_{\mathrm{var}} \approx 2|m|/s = 2 \times 10^9$. Damit gehen $9$ der in `fp64` verfügbaren $16$ Stellen verloren (Welfords Verfahren erhält $\sim 7$ Stellen). Die naive Formel subtrahiert zwei Zahlen der Größenordnung $m^2 = 10^{18}$. Dadurch quadriert sich der Konditionsfaktor auf $2m^2/s^2 = 2 \times 10^{18} > 1/u$ (Gitterabstand $128$). **(3) Die Normalgleichungen** quadrieren $\kappa$ auf $\kappa(\mathbf{A}^\top\mathbf{A}) = 10^{12}$ und erhalten ungefähr $16 - 12 = \mathbf{4\text{ digits}}$. **QR/SVD** arbeitet direkt mit $\mathbf{A}$ ($\kappa = 10^6$) und erhält ungefähr $16 - 6 = \mathbf{10\text{ digits}}$.

<!--en:8090bf9d-->
::: {.slide title="Zusammenfassung: numerische Stabilität und Kondition"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:556b738b-->
| Phänomen / Aufgabe | Naives / instabiles Verfahren | Numerischer Fehler | Stabiles Verfahren / Abhilfe |

<!--en:8a100c9b-->
| **Aktualisierungen bei geringer Genauigkeit** | Direkte Gewichtsaktualisierung in `bf16`/`fp16` | $\mathrm{fl}(w - \eta g) = w$ für $|\eta g| < u\,2^e$ | Hauptgewichte in `fp32` + Loss-Scaling / stochastische Rundung |

<!--en:099e2547-->
| **Softmax-Funktion und Log-Sum-Exp** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ und $\log\sum_j e^{z_j}$ | $e^{z_i} \to \infty$ (`NaN`) für $z_i > 88.7$ (`fp32`) | Verschiebung um das Maximum $c = \max_j z_j$: $\mathrm{lse}(\mathbf{z}) = c + \log\sum_j e^{z_j - c}$ |

<!--en:b76a3998-->
| **Kreuzentropieverlust** | $-\log(\mathrm{softmax}(\mathbf{z})_y)$ | $p_y \to 0 \implies -\log p_y = \infty$ | Kombinierter Verlust aus Logits $(c - z_y) + \log\sum_j e^{z_j - c}$ |

<!--en:ef615e2b-->
| **Werte nahe null und Varianz** | $\log(1 + x)$ und $\frac{1}{N}\sum x_i^2 - m^2$ | Katastrophale Auslöschung ($\kappa \to \kappa^2$) | `log1p(x)` und Welfords zentrierte Rekursion $M_k = M_{k-1} + d_k(x_k - m_k)$ |

<!--en:9b6958c0-->
| **Summen großer Arrays** | Schleife von links nach rechts $\sum_{i=1}^n x_i$ | Fehler wächst linear: $(n-1)u\sum |x_i|$ | Paarweise Summation im Baum ($u\log_2 n\sum|x_i|$) oder Kompensation nach Kahan |

<!--en:8e5ede0c-->
| **Kleinste Quadrate und Ridge-Regularisierung** | Normalgleichungen $\mathbf{A}^\top\mathbf{A}\mathbf{w} = \mathbf{A}^\top\mathbf{b}$ | Quadriert die Konditionszahl: $\kappa(\mathbf{A}^\top\mathbf{A}) = \kappa(\mathbf{A})^2$ | QR/SVD (`lstsq`) + Ridge-Dämpfung $\kappa = \frac{\sigma_1^2 + \lambda}{\sigma_d^2 + \lambda}$ |
