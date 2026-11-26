<!--en:79fa9fa0-->
# Die Kettenregel auf Berechnungsgraphen

<!--en:9b276d51-->
Jeder Trainingsschritt in diesem Buch beginnt mit einem einzigen Aufruf von `backward()` oder `grad`. Er liefert für jedes Gewicht im Netz die Steigung des Verlusts bezüglich dieses Gewichts. In :numref:`sec_mdl-multivariable_calculus` haben wir gesehen, was diese Steigungen bedeuten. Zusammen bilden sie den Gradienten, also den Vektor, der für jede kleine gemeinsame Änderung der Parameter vorhersagt, wie sich der Verlust ändert. Offen ist noch, wie ein Programm Millionen solcher Steigungen in etwa der Zeit berechnet, die wenige Auswertungen des Netzes brauchen. Beim naiven Ansatz ändern wir jedes Gewicht einzeln um einen kleinen Betrag und werten das Netz jedes Mal neu aus. Das erfordert einen eigenen Durchlauf pro Gewicht und wird schnell undurchführbar.

<!--en:1cc5c285-->
Mit der Kettenregel geht es wesentlich schneller. Den Grund erkennen wir auch ohne Formeln. Ein neuronales Netz entsteht durch tiefe Verkettung einfacher Funktionen. Der Einfluss jedes Gewichts auf den Verlust läuft über die nachfolgenden Schichten. Beim Ableiten nach den vielen Gewichten fallen daher fast überall dieselben Rechnungen an. In einer einzigen verschachtelten Formel bleibt diese gemeinsame Struktur verborgen. Ein *Berechnungsgraph* (engl. computational graph) macht sie sichtbar. Für jeden Zwischenwert gibt es einen Knoten und für jede direkte Abhängigkeit eine gerichtete Kante. Im Graphen nimmt die Kettenregel die Form einer Summe über Pfade an. Wie viel Rechenarbeit wir gemeinsam nutzen können, hängt davon ab, wie wir die Summanden gruppieren. Gruppieren wir die Pfade von der Ausgabe aus rückwärts, erhalten wir in einem einzigen Durchlauf die Ableitungen nach sämtlichen Eingaben. Diese Gruppierung ist die Backpropagation.

<!--en:d586f615-->
Wir arbeiten durchgehend mit kleinen Graphen und skalaren Variablen mit ganzzahligen Werten. So können wir jeden Schritt von Hand nachvollziehen und überprüfen. Die gesamte Herleitung beruht auf einer Erkenntnis aus :numref:`sec_mdl-multivariable_calculus`. In der Nähe jedes Punktes, an dem eine Funktion differenzierbar ist, nähert das Skalarprodukt aus Schritt und Gradient die Änderung der Funktion an. Das ist das lineare Modell aus :numref:`subsec_mdl-gradient`. In den beiden folgenden Abschnitten übertragen wir diesen Ansatz auf Schichten, die Vektoren auf Vektoren abbilden, und messen die Rechenkosten an realen Netzen.

<!--en:c290ea67-->
## Die Kettenregel als Summe über Pfade

<!--en:1672aa14-->
### Ein Graph aus Skalaren

<!--en:d0d323f9-->
Beginnen wir mit einem Beispiel, das wir vollständig von Hand durchrechnen können. Aus vier skalaren Eingaben $w, x, y, z$ entsteht über zwei Schichten von Zwischenvariablen eine einzige skalare Ausgabe:

<!--en:8e2b508f-->
Diese Gleichungen definieren einen gerichteten azyklischen Graphen (:numref:`fig_mdl-chain-1`). Jeder Knoten enthält einen Zwischenwert, jeder Pfeil zeigt eine direkte funktionale Abhängigkeit an. Das Quadrieren hat hier nur den Zweck, dass alle Zwischenwerte und Ableitungen exakt ganzzahlig bleiben. In diesem Abschnitt bezeichnet $f$ stets die skalare Ausgabe des jeweils betrachteten Graphen.

<!--en:77989587-->
![Abhängigkeitsgraph zu :eqref:`eq_mdl-multi_func_def`. Jeder Knoten steht für eine Variable, jede gerichtete Kante für eine direkte funktionale Abhängigkeit.](../img/mdl-cal-chain-net1.svg)

<!--en:1a33eb2a-->
### Zwei Pfade in einem Rechenschritt

<!--en:e8af79fb-->
Betrachten wir zunächst nur den letzten zusammengesetzten Schritt des Graphen, $f(u(a,b), v(a,b))$. Wie reagiert $f$, wenn wir $a$ um einen kleinen Betrag $\epsilon$ ändern? Beide Zwischenknoten ändern sich entsprechend ihren partiellen Ableitungen. In erster Näherung geht $u$ in $u + \epsilon\frac{\partial u}{\partial a}$ über und $v$ in $v + \epsilon\frac{\partial v}{\partial a}$. Setzen wir beide Änderungen in die lineare Näherung erster Ordnung (:numref:`subsec_mdl-gradient`) von $f$ ein, so erhalten wir

<!--en:e949353b-->
Am Koeffizienten von $\epsilon$ lesen wir die Ableitung der zusammengesetzten Funktion nach $a$ ab. Damit erhalten wir die Kettenregel für mehrere Variablen:

<!--en:50a33455-->
Von nun an unterscheiden wir strikt zwischen den beiden Ableitungssymbolen. Das geschwungene $\partial$ bezeichnet die Ableitung der Funktion eines einzelnen Knotens nach einem ihrer direkten Argumente. Dabei halten wir die übrigen Argumente fest. Das aufrechte $\mathrm{d}$ bezeichnet die *totale* Ableitung $\mathrm{d}f/\mathrm{d}a$. Hier dürfen sich alle nachgelagerten Variablen ändern, wenn sich $a$ ändert. (Für einen Eingabeknoten ist die totale Ableitung gleich der gewöhnlichen partiellen Ableitung von $f$ als Funktion allein der Eingaben.) Geometrisch gibt es zwei gerichtete *Pfade*, über die $a$ die Ausgabe $f$ beeinflusst, nämlich $a \to u \to f$ und $a \to v \to f$. Jeder Pfad trägt das Produkt der Ableitungen entlang seiner Kanten bei. Die totale Ableitung summiert diese Beiträge über alle Pfade.

<!--en:2b3a028d-->
Die Herleitung folgt dem Beweis der Kettenregel für eine Variable (:numref:`sec_mdl-derivative_table`). Wir setzen die linearen Änderungen der inneren Funktionen in das lineare Modell der äußeren Funktion ein. Da sich hier zwei Variablen gleichzeitig ändern, muss $f$ im Punkt $(u, v)$ *differenzierbar* sein. Partielle Ableitungen allein reichen nicht aus. Denken wir an das Gegenbeispiel $2xy/(x^2+y^2)$ aus :numref:`subsec_mdl-gradient`. Betrachten wir die Diagonale $x = y = t$. Im Ursprung verschwinden beide partiellen Ableitungen. Die Kettenregel würde daher eine Ableitung von $0$ an der Stelle $t = 0$ vorhersagen. Tatsächlich springt der Funktionswert dort von $0$ auf $1$.

<!--en:c5d798b3-->
### Kantenableitungen und totale Ableitungen

<!--en:f322f94e-->
Mit Induktion lässt sich diese Überlegung auf jeden gerichteten azyklischen Graphen übertragen. Als *Kantenableitung* (engl. edge derivative) eines Knotens nach einem seiner direkten Vorgänger bezeichnen wir die partielle Ableitung seiner Knotenfunktion nach diesem Vorgänger. Alle anderen direkten Vorgänger halten wir dabei fest. Für die totale Ableitung einer Ausgabe $f$ nach einer Eingabe $x$ gilt dann

<!--en:61a80f28-->
In Worten: Für jeden gerichteten Pfad von $x$ nach $f$ multiplizieren wir die lokalen Änderungsraten entlang seiner Kanten. Die Summe dieser Pfadprodukte ist die totale Ableitung.

<!--en:8dbb8527-->
Ein Graph mit einer Skip-Verbindung (engl. skip connection) zeigt, weshalb wir Kantenableitungen und totale Ableitungen auseinanderhalten müssen. Betrachten wir das System aus fünf Knoten

<!--en:aef4de5c-->
Hier hängt $f$ von $r$ und $s$ ab und über eine Skip-Kante auch direkt von $p$ (:numref:`fig_mdl-chain-2`). Damit erreicht die Eingabe $y$ die Ausgabe $f$ auf drei verschiedenen Pfaden, nämlich über $y \to p \to r \to f$, die Abkürzung $y \to p \to f$ und $y \to q \to s \to f$. Die lokalen Kantenableitungen sind $\partial f/\partial r = \partial f/\partial p = \partial f/\partial s = 1$, $\partial r/\partial p = 2p$, $\partial s/\partial q = 2q$, $\partial p/\partial y = 1$ und $\partial q/\partial y = -1$. Die Summe der drei Pfadprodukte ergibt

<!--en:fc5cf4d8-->
Zur Kontrolle setzen wir alle Terme in $f$ ein. Wir erhalten $f = (x+y)^2 + (x+y) + (x-y)^2$, und die Ableitung nach $y$ ist tatsächlich $1 + 4y$. Im Punkt $(x, y) = (1, 2)$ gilt $p = 3$ und $q = -1$. Die Beiträge der drei Pfade ergeben also $6 + 1 + 2 = 9 = 1 + 4 \cdot 2$. Am Knoten $p$ wird der Unterschied deutlich. Die *Kantenableitung* $\partial f/\partial p$ beträgt nur $1$, denn wir halten $r$ fest. Die *totale* Ableitung $\mathrm{d}f/\mathrm{d}p$ berücksichtigt sowohl die Skip-Kante als auch den Umweg über $r$ und beträgt deshalb $2p + 1 = 7$. Der häufigste Fehler bei der Anwendung der Kettenregel für mehrere Variablen besteht darin, diese beiden Größen zu verwechseln.

<!--en:9d185778-->
![Abhängigkeitsgraph zu :eqref:`eq_mdl-multi_func_def_2`. Von der Eingabe $y$ führen drei Pfade zur Ausgabe $f$, nämlich $y\to p\to r\to f$, die Abkürzung $y\to p\to f$ über die Skip-Kante und $y\to q\to s\to f$.](../img/mdl-cal-chain-net2.svg)

<!--en:1d9f6bb7-->
Die Summe über Pfade erklärt auch, wie sich Gradienten durch tiefe Architekturen ausbreiten. Die Gatter eines LSTM (:numref:`sec_lstm`) oder die Skip-Verbindungen eines Residualnetzes (:numref:`sec_resnet`) öffnen oder schließen Pfade. Solche Mechanismen beeinflussen das Lernen unmittelbar, denn sie steuern, über welche Pfade das Gradientensignal zu früheren Schichten zurückfließt.

<!--en:c3c9823b-->
## Backpropagation

<!--en:5cbd0f46-->
Als mathematische Identität gilt die Pfadsumme :eqref:`eq_mdl-chain-rule-paths` ganz allgemein. Sie liefert jede Ableitung in jedem Berechnungsgraphen. Als unmittelbare Rechenvorschrift ist sie jedoch untauglich, denn die Zahl der Pfade kann mit der Tiefe exponentiell wachsen. Schon eine einfache Kette mit $m$ Stufen und zwei parallelen Knoten pro Stufe enthält $2^m$ Pfade vom Anfang bis zum Ende. Diese Pfade überlappen sich jedoch stark. Jedes Teilprodukt, das in vielen Pfaden vorkommt, müssen wir nur einmal berechnen. Die Reihenfolge, in der wir gemeinsame Teilprodukte ausklammern, bestimmt die Rechenkosten.

<!--en:b052fdcc-->
### Vorwärtsmodus

<!--en:0d53ea45-->
Vergleichen wir zwei Arten, die Summe zu faktorisieren. Dazu betrachten wir unsere Funktion mit vier Eingaben aus :eqref:`eq_mdl-multi_func_def` im Punkt $(w, x, y, z) = (1, 1, -2, 1)$. Ein *Vorwärtsdurchlauf* (engl. forward pass) wertet die Knoten in topologischer Reihenfolge aus. Wir erhalten $a = 1^2 = 1$, $b = 3^2 = 9$, $u = 10^2 = 100$, $v = (-8)^2 = 64$ und $f = 164^2 = 26896$. An diesen Knotenwerten lesen wir alle lokalen Kantenableitungen ab:

<!--en:6fe669d2-->
Jede Kantenableitung ist eine lokale Größe. Zu ihrer Berechnung genügen die Werte des Knotens und seiner direkten Vorgänger.

<!--en:0de02097-->
Um $\mathrm{d}f/\mathrm{d}w$ zu bestimmen, betrachten wir die vier Pfade von $w$ nach $f$. Sie führen alle zuerst über $a$ oder $b$ und dann über $u$ oder $v$. Klammern wir gemeinsame Faktoren von der Eingabe $w$ aus vorwärts durch den Graphen aus, erhalten wir

<!--en:fc39b061-->
Die beiden Summen in Klammern sind $\mathrm{d}u/\mathrm{d}w = 160$ und $\mathrm{d}v/\mathrm{d}w = 64$, also die totalen Ableitungen der Zwischenknoten nach $w$. Wenn wir die Sensitivität bezüglich einer Eingabe auf diese Weise vorwärts durch den Graphen weitergeben, sprechen wir vom *Vorwärtsmodus* (engl. forward mode). Wir können die Rechnung in reinem Python überprüfen:

<!--en:01324504-->
Der Vorwärtsmodus liefert $\mathrm{d}f/\mathrm{d}w = 73472$. Für $x$, $y$ oder $z$ können wir jedoch keine der Ableitungen an den Zwischenknoten wiederverwenden. Sowohl $\mathrm{d}u/\mathrm{d}w$ als auch $\mathrm{d}v/\mathrm{d}w$ beziehen sich allein auf $w$. Beim Training eines neuronalen Netzes brauchen wir die Ableitung eines einzigen skalaren Verlusts nach jedem Parameter. Rechnen wir für jede Eingabe einzeln, bleibt der größte Teil der gemeinsamen Struktur ungenutzt.

<!--en:bdf523ee-->
### Adjungierte Variablen und Rückwärtsdurchlauf

<!--en:d99fe186-->
Wechseln wir die Perspektive und wählen statt einer Eingabe eine feste Ausgabe. Im Vorwärtsmodus hatten wir eine Eingabe gewählt und gefragt, wie jeder nachgelagerte Knoten auf sie reagiert. Nun wählen wir die einzige Ausgabe $f$ und fragen, wie sie auf jeden vorgelagerten Knoten reagiert. Die *adjungierte Variable* (engl. adjoint) $\bar{n}$ eines Knotens $n$ definieren wir als die totale Ableitung $\mathrm{d}f/\mathrm{d}n$ der Ausgabe nach $n$. Sie ist die Summe der Produkte der Kantenableitungen über alle gerichteten Pfade von $n$ nach $f$. Jeder solche Pfad verlässt $n$ über einen seiner direkten Nachfolger $c$. Die Beiträge der verbleibenden Pfade von $c$ nach $f$ summieren sich zu $\bar{c}$. Gruppieren wir die Pfadsumme :eqref:`eq_mdl-chain-rule-paths` nach der ersten Kante, erhalten wir eine einfache Rekursion:

<!--en:d431eb95-->
Auf der rechten Seite stehen nur lokale Kantenableitungen und die adjungierten Variablen nachgelagerter Knoten. Wir beginnen bei $\bar f = 1$ und durchlaufen die Knoten in umgekehrter topologischer Reihenfolge. Damit kennen wir die adjungierte Variable jedes direkten Nachfolgers bereits, wenn wir sie brauchen. Für den Graphen mit Skip-Kante aus :eqref:`eq_mdl-multi_func_def_2` liefert die Rekursion beispielsweise $\bar r = 1$ und $\bar p = \bar r \cdot 2p + \bar f \cdot 1 = 7$. Das stimmt mit der zuvor berechneten totalen Ableitung überein. Beim gleichen Durchlauf durch unseren Graphen mit vier Eingaben erhalten wir:

<!--en:058ba842-->
* $\bar{a} = \bar{u}\cdot 20 + \bar{v}\cdot(-16) = 1312$ und
  $\bar{b} = \bar{u}\cdot 20 + \bar{v}\cdot 16 = 11808$,

<!--en:febc6148-->
* $\bar{w} = \bar{x} = \bar{a}\cdot 2 + \bar{b}\cdot 6 = 73472$ und
  $\bar{y} = \bar{z} = \bar{a}\cdot 2 - \bar{b}\cdot 6 = -68224$.

<!--en:f6c81987-->
Ein einziger *Rückwärtsdurchlauf* (engl. backward pass) liefert also die partiellen Ableitungen nach allen vier Eingaben zugleich (:numref:`fig_mdl-cal-backward-sweep`). Führen wir den Rückwärtsdurchlauf in Python aus. Dabei verwenden wir die Kantenableitungen aus der vorigen Codezelle.

<!--en:154bbe90-->
![(a) Der Vorwärtsdurchlauf speichert alle Knotenwerte und lokalen Kantenableitungen. (b) Der Rückwärtsdurchlauf summiert die Beiträge zu den adjungierten Variablen. Ausgehend von $\bar f = 1$ ergeben sich $\bar a = 328\cdot 20 + 328\cdot(-16) = 1312$ und $\bar w = 1312\cdot 2 + 11808\cdot 6 = 73472$.](../img/mdl-cal-backward-sweep.svg)

<!--en:91d07a0e-->
Die Ableitungen nach $w$ und $x$ stimmen überein, ebenso die nach $y$ und $z$. Denn $f$ hängt von seinen Eingaben nur über die beiden Summen $w + x$ und $y + z$ ab.

<!--en:5f9e192e-->
Zählen wir die Operationen, um den Vorteil zu beziffern. Der Vorwärtsmodus braucht sechs Multiplikationen pro Eingabe (je zwei für $\mathrm{d}u/\mathrm{d}w$, $\mathrm{d}v/\mathrm{d}w$ und $\mathrm{d}f/\mathrm{d}w$), insgesamt also $24$ für alle vier Eingaben. Der Rückwärtsdurchlauf braucht vier Multiplikationen für $\bar a$ und $\bar b$ und anschließend zwei pro Eingabe, insgesamt also $12$. Den gemeinsamen oberen Teil des Graphen durchläuft er nur einmal. Bei unserem kleinen Beispielgraphen ist der Faktor zwei bescheiden, doch mit der Zahl der Eingaben wächst der Unterschied rasch. Jede weitere Eingabe kostet im Vorwärtsmodus einen ganzen Durchlauf durch den Graphen. Im Rückwärtsdurchlauf kommen nur zwei Multiplikationen hinzu.

<!--en:c4e1ee21-->
### Backpropagation in einem Framework

<!--en:7d234ed0-->
Berechnen wir Ableitungen, indem wir adjungierte Variablen von der Ausgabe zurück zu den Eingaben weitergeben, so heißt das *Backpropagation* :cite:`Rumelhart.Hinton.Williams.ea.1988`. In :numref:`sec_backprop` wenden wir sie auf mehrschichtige Perzeptronen an. Alle modernen Systeme zum automatischen Differenzieren verwenden dieselben zwei Durchläufe. Im Vorwärtsdurchlauf speichern sie die Zwischenwerte oder die Operationen, mit denen sich die Kantenableitungen bilden lassen. Im Rückwärtsdurchlauf summieren sie die Beiträge zu den adjungierten Variablen. Überprüfen wir unsere Rechnung von Hand, indem wir das Framework für dieselben vier Eingaben automatisch differenzieren lassen.

<!--en:b511dc74-->
Das Framework liefert $(73472, 73472, -68224, -68224)$. Diese Werte stimmen exakt mit den von Hand berechneten adjungierten Variablen überein. Jeder Zwischenwert ist nämlich eine ganze Zahl mit einem Betrag unter $2^{24}$ und lässt sich daher in einfacher Genauigkeit ohne Rundungsfehler darstellen.

<!--en:674d3790-->
Bisher war jeder Knoten in unseren Graphen ein Skalar. Die Multiplikationen konnten wir von Hand zählen. In einem realen neuronalen Netz bildet jede Schicht einen Vektor auf einen Vektor ab. Dadurch werden die Kantenableitungen zu Matrizen, und die Reihenfolge der Matrixmultiplikationen spielt eine weitaus größere Rolle als bei skalaren Multiplikationen. Diesen Schritt gehen wir in :numref:`sec_mdl-matrix-calculus`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:80fc84f9-->
In einem Berechnungsgraphen ist die totale Ableitung einer Ausgabe nach einer Eingabe eine Summe über alle gerichteten Pfade zwischen beiden. Jeder Pfad trägt das Produkt seiner lokalen Kantenableitungen bei. Die adjungierte Variable eines Knotens ist die totale Ableitung der Ausgabe nach diesem Knoten. Sie fasst die Beiträge aller Pfade zusammen, die von ihm ausgehen. Gruppieren wir diese Pfade nach ihrer ersten Kante, erhalten wir eine Rekursion. Von der Ausgabe aus rückwärts angewandt, berechnet sie sämtliche adjungierten Variablen in einem Durchlauf. Diese Rückwärtsakkumulation ist die Backpropagation. Gehen wir von einer Eingabe aus vorwärts durch den Graphen, erhalten wir nur die Ableitung nach dieser einen Eingabe. Deshalb differenzieren wir einen skalaren Verlust nach vielen Parametern immer rückwärts.

<!--en:2294243a-->
## Übungen

<!--en:fd8d1db7-->
1. [code] **Rückwärtsdurchlauf für zwei tanh-Einheiten.** Ein Netz mit einer Eingabe $x$, zwei versteckten Einheiten und einer Ausgabe berechnet für einen Zielwert $t$ folgende Größen:

   $$h_i = \tanh(w_i x + b_i) \quad (i = 1, 2), \qquad f = v_1 h_1 + v_2 h_2, \qquad \ell = \tfrac12 (f - t)^2.$$

    1. Zeichnen Sie den Abhängigkeitsgraphen von $\ell$. Schreiben Sie in reinem Python einen Vorwärtsdurchlauf, der alle Zwischenwerte speichert, und einen Rückwärtsdurchlauf, der die partiellen Ableitungen von $\ell$ nach allen sechs Parametern $w_1, b_1, w_2, b_2, v_1, v_2$ zurückgibt. Überprüfen Sie die Ergebnisse durch automatisches Differenzieren für $x = 1.5$, $t = 0.5$ und $(w_1, b_1, w_2, b_2, v_1, v_2) = (0.5, 0, -1, 0.2, 1, -0.5)$. Welche Ableitung an einem Zwischenknoten geht in alle sechs partiellen Ableitungen ein?
    1. Setzen Sie $w_1 = 5$ und $x = 2$. Behalten Sie die übrigen Werte bei. Berechnen Sie $\partial \ell / \partial w_1$ mit Ihrem Rückwärtsdurchlauf in reinem Python. Erklären Sie die Größe der Ableitung anhand des Produkts der Kantenableitungen entlang ihres Pfades. Vergleichen Sie das Ergebnis mit dem Wert, den automatisches Differenzieren in einfacher Genauigkeit liefert, und erklären Sie den Unterschied. Was folgt daraus für das Training einer Einheit mit großer Präaktivierung?
    1. Setzen Sie $w_1 = w_2$, $b_1 = b_2$ und $v_1 = v_2$. Zeigen Sie anhand des Rückwärtsdurchlaufs, dass beide Einheiten identische Gradienten erhalten und deshalb beim Gradientenabstieg in jedem Schritt identisch bleiben. Was folgt daraus für die Initialisierung versteckter Einheiten?

<!--en:f18a9b71-->
1. **Pfade durch eine Folge von Residualblöcken.** ● In einem eindimensionalen Modell der Residualnetze aus :numref:`sec_resnet` berechnen skalare Blöcke $x_{l+1} = x_l + g_l(x_l)$ für $l = 0, \ldots, m-1$ mit differenzierbaren Funktionen $g_l$.
    1. Schreiben Sie $\partial x_m / \partial x_0$ mithilfe der Kettenregel als Produkt aus $m$ Faktoren.
    1. Multiplizieren Sie das Produkt aus. Ordnen Sie jeden Summanden einem gerichteten Pfad von $x_0$ nach $x_m$ im Abhängigkeitsgraphen zu. Wie viele Pfade gibt es, und wie viele davon durchlaufen genau $k$ der Funktionen $g_l$? Vergleichen Sie die Anzahl der Pfade mit der Anzahl der Multiplikationen, die ein Rückwärtsdurchlauf zur Berechnung von $\partial x_m / \partial x_0$ benötigt.
    1. Für ein trainiertes Netz mit $m = 54$ Residualblöcken stellten :citet:`Veit.Wilber.Belongie.2016` fest, dass fast der gesamte Gradient auf Pfaden durch $5$ bis $17$ Blöcke fließt. Welchen Anteil aller Pfade machen diese Pfade aus? Nehmen Sie nun an, dass $g_l'(x_l) = c > 0$ für jeden Block gilt. Für welchen Wert von $c$ liefern die Pfade durch etwa $11$ Blöcke den größten Beitrag zu $\partial x_m / \partial x_0$? Welchen Anteil der Ableitung tragen dann die Pfade durch $5$ bis $17$ Blöcke?

<!--en:94a5ffd8-->
1. [code] **Gradientenprüfung in einfacher Genauigkeit.** ● Eine Gradientenprüfung vergleicht das Ergebnis eines Rückwärtsdurchlaufs mit dem zentralen Differenzenquotienten $\bigl(f(w + h, x, y, z) - f(w - h, x, y, z)\bigr) / (2h)$. Sei $f$ die Funktion aus :eqref:`eq_mdl-multi_func_def`. Ihre exakte partielle Ableitung im Punkt $(w, x, y, z) = (1, 1, -2, 1)$ beträgt $\partial f / \partial w = 73472$.
    1. Implementieren Sie $f$ in NumPy. Berechnen Sie den zentralen Differenzenquotienten an diesem Punkt für $h = 10^{-1}, 10^{-2}, \ldots, 10^{-12}$, einmal in `float64` und einmal in `float32`. Tragen Sie für beide Genauigkeiten den relativen Fehler gegen $h$ auf. Verwenden Sie für beide Achsen eine logarithmische Skala und lesen Sie jeweils das beste $h$ und den kleinsten relativen Fehler ab.
    1. Der Fehler des zentralen Differenzenquotienten setzt sich aus dem Abbruchfehler der Ordnung $h^2$ und dem Rundungsfehler der Ordnung $\varepsilon_{\textrm{mach}} |f| / h$ zusammen. Dabei beträgt das Maschinenepsilon $\varepsilon_{\textrm{mach}}$ etwa $2.2 \times 10^{-16}$ für `float64` und $1.2 \times 10^{-7}$ für `float32`. Um welche Faktoren sollten sich das beste $h$ und der kleinste relative Fehler beim Wechsel von `float64` zu `float32` ändern? Vergleichen Sie die vorhergesagten Faktoren mit dem Diagramm.
    1. Welche Toleranz für den relativen Fehler kann eine Gradientenprüfung in `float32` verwenden, ohne einen korrekten Rückwärtsdurchlauf zu verwerfen? Warum prüft man Gradienten in doppelter Genauigkeit?
    1. Warum ergibt die Näherung in `float32` für die kleinsten Werte von $h$ exakt null?
   :exlabel:`ex_mdl-multivariable-calculus-gradient-check`

<!--en:c1136f50-->
[Diskussion](https://d2l.discourse.group/t/36877)

<!--en:6517b282-->
Die Kettenregel auf Berechnungsgraphen<br>**Gerichtete azyklische Abhängigkeitsgraphen · Kantenableitungen und totale Ableitungen · Summe über Pfade · Vorwärtsmodus · adjungierte Variablen und Backpropagation**

<!--en:93948f9f-->
::: {.slide title="Warum ein Rückwärtsdurchlauf eine Million Störungen ersetzt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:5cfd0b5d-->
- **Einzelne Störungen als Flaschenhals:** Einzelne Störungen von $N = 10^8$ Parametern erfordern $10^8$ separate Vorwärtsdurchläufe.

<!--en:1368ff45-->
- **Gemeinsame Teilpfade:** Der Einfluss jedes Parameters aus einer frühen Schicht erreicht den skalaren Verlust $f$ über dieselben oberen Schichten.

<!--en:72a78c07-->
- **Backpropagation durch Ausklammern:** Die rückwärtige Gruppierung der Pfadsumme von $f$ aus macht jede nachgelagerte Ableitung für alle Eingaben nutzbar.

<!--en:8eb9ca3e-->
::: {.slide title="Eine Funktion als Berechnungsgraph"}

<!--en:438da543-->
[Gerichtete azyklische Graphen]{.kicker}

<!--en:0932638a-->
- **Ganzzahliges Rechenbeispiel mit 4 Eingaben:** Die vier Eingaben $(w, x, y, z)$ durchlaufen in Schicht 1 die Knoten $a = (w + x + y + z)^2$ und $b = (w + x - y - z)^2$, in Schicht 2 die Knoten $u = (a + b)^2$ und $v = (a - b)^2$. Die skalare Ausgabe lautet
  $$f(u, v) = (u + v)^2$$

<!--en:67869503-->
- **Knoten und gerichtete Kanten:** Jeder Knoten speichert eine Zwischenvariable. Eine gerichtete Kante $n \to c$ bedeutet, dass die Formel für Knoten $c$ den Wert von $n$ direkt als Argument verwendet.

<!--en:4504860f-->
- **Exakte Ganzzahlarithmetik:** Durch das Quadrieren bleiben alle Knotenwerte und Ableitungen ganzzahlig. So lässt sich jeder Schritt ohne Gleitkommarundung überprüfen.

<!--en:7affa82f-->
::: {.slide title="Wie sich eine Störung auf parallele Zweige auswirkt"}

<!--en:27498137-->
[Kettenregel für mehrere Variablen]{.kicker}

<!--en:38f697ae-->
- **1. Änderung an den direkten Nachfolgern bestimmen:** Die Störung $a \mapsto a + \epsilon$ ändert $u$ um $\Delta u \approx \epsilon\frac{\partial u}{\partial a}$ und $v$ um $\Delta v \approx \epsilon\frac{\partial v}{\partial a}$.

<!--en:a85ed1ea-->
- **2. In das lineare Modell von $f(u, v)$ einsetzen:** $\Delta f \approx \frac{\partial f}{\partial u}\Delta u + \frac{\partial f}{\partial v}\Delta v$.

<!--en:d98145bd-->
- **3. Beiträge der parallelen Zweige addieren:** $\frac{\mathrm{d}f}{\mathrm{d}a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial a}$.

<!--en:e6dae895-->
::: {.slide title="Kantenableitungen (∂) und totale Ableitungen (d)"}

<!--en:1fcc4bdb-->
[Notation und Bedeutung]{.kicker}

<!--en:77a0a0c5-->
- **Kantenableitung $\frac{\partial c}{\partial n}$:** Lokale partielle Ableitung der Formel für den Nachfolger $c$ nach dem Vorgänger $n$. Dabei **bleiben alle anderen direkten Vorgänger von $c$ fest**.

<!--en:5d5ec7d4-->
- **Totale Ableitung (adjungierte Variable) $\bar{n} = \frac{\mathrm{d}f}{\mathrm{d}n}$:** Gesamte Änderungsrate der Ausgabe $f$, wenn sich der Knotenwert $n$ leicht ändert und **alle nachgelagerten Knoten reagieren**, und zwar entlang jedes gerichteten Pfades $n \to \cdots \to f$.

<!--en:8e055180-->
::: {.slide title="Die allgemeine Kettenregel als Summe über Pfade"}

<!--en:45b0fbda-->
[Formel für die Pfadsumme]{.kicker}

<!--en:9709fb0f-->
- **Kettenregel als Pfadsumme:** $\frac{\mathrm{d}f}{\mathrm{d}x} = \sum_{\text{paths } x \to \cdots \to f} \prod_{(u \to v)} \frac{\partial v}{\partial u}$ (entlang der Kanten jedes Pfades multiplizieren, die Beiträge paralleler Pfade addieren).

<!--en:a9cf7fbd-->
- **Exponentieller Aufwand beim Aufzählen:** Ein Graph mit $m$ Stufen und 2 parallelen Knoten je Stufe hat $2^m$ Pfade. Dynamische Programmierung nutzt beim automatischen Differenzieren die gemeinsamen Knoten und benötigt nur Zeit der Ordnung $O(|\text{Edges}|)$.

<!--en:8a58818e-->
::: {.slide title="Skip-Verbindungen und das Zusammenspiel mehrerer Pfade"}

<!--en:b02ff776-->
[Rechenbeispiel]{.kicker}

<!--en:44e25374-->
- **Gerichteter azyklischer Graph mit 5 Knoten und einer Skip-Kante:** Es gelte $p = x + y$, $q = x - y$, $r = p^2$, $s = q^2$ und $f(r, p, s) = r + p + s$.

<!--en:c869085c-->
- **Drei Pfade von $y$ nach $f$:** Pfad $y \to p \to r \to f$ liefert $1 \cdot 2p \cdot 1 = 2p$, die Abkürzung $y \to p \to f$ über die Skip-Kante liefert $1 \cdot 1 = 1$ und Pfad $y \to q \to s \to f$ liefert $1 \cdot 2q \cdot (-1) = -2q$. Insgesamt ergibt sich
  $$\frac{\mathrm{d}f}{\mathrm{d}y} = 2p + 1 - 2q = 1 + 4y$$

<!--en:404c69f0-->
- **Im Punkt $(x, y) = (1, 2)$ ($p = 3, q = -1$):** Die drei Pfade liefern zusammen $6 + 1 + 2 = 9 = 1 + 4(2)$.

<!--en:0e5cc713-->
- **Kantenableitung und totale Ableitung am Knoten $p$:** Für die direkte Skip-Kante gilt $\frac{\partial f}{\partial p} = 1$ (bei festem $r$). Die totale Ableitung beträgt $\frac{\mathrm{d}f}{\mathrm{d}p} = 2p + 1 = 7$.

<!--en:321b69d6-->
::: {.slide title="Wie ResNets und Gatter die Pfade bestimmen"}

<!--en:8db841f9-->
[Bezug zum Deep Learning]{.kicker}

<!--en:35b09515-->
- **Folge von Residualblöcken ($x_{l+1} = x_l + g_l(x_l)$):** Jeder Block verzweigt sich in eine Skip-Kante mit Identitätsabbildung (Steigung $1$) und einen Residualzweig (Steigung $g_l'$). Daraus folgt $\frac{\mathrm{d}x_m}{\mathrm{d}x_0} = \prod_{l=0}^{m-1}(1 + g_l'(x_l))$.

<!--en:2b648add-->
- **Gesamtheit der $2^m$ Pfade:** Beim Ausmultiplizieren ergeben sich $\binom{m}{k}$ Pfade, die jeweils $k$ nichtlineare Blöcke durchlaufen und $m - k$ Blöcke mit Steigung eins überspringen (Veit et al., 2016).

<!--en:122b468d-->
::: {.slide title="Vorwärtsdurchlauf und lokale Kantenableitungen"}

<!--en:9b873311-->
[Rechnung von Hand (Schritt 1)]{.kicker}

<!--en:4945d9ff-->
- **Werte im Vorwärtsdurchlauf bei $(1, 1, -2, 1)$:** Schicht 1 liefert $a = 1^2 = 1$ und $b = 3^2 = 9$, Schicht 2 liefert $u = 10^2 = 100$ und $v = (-8)^2 = 64$. Die Ausgabe ist $f = 164^2 = 26{,}896$.

<!--en:ca63b1e8-->
- **Alle 14 Kantensteigungen sind sofort bekannt:** Die lokalen partiellen Ableitungen ($\partial a/\partial w = 2$, $\partial b/\partial w = 6$, $\partial u/\partial a = 20$, $\partial v/\partial a = -16$, $\partial f/\partial u = 328$) hängen nur vom jeweiligen Knoten und seinen direkten Vorgängern ab.

<!--en:8665f67e-->
::: {.slide title="Vorwärtsmodus: Pfadprodukte von einer Eingabe aus zusammenfassen"}

<!--en:7ce6ceb6-->
[Vorwärtsakkumulation]{.kicker}

<!--en:e7cd3689-->
- **Sensitivität für eine Eingabe vorwärts weitergeben:** Für die gewählte Eingabe $w$ ergibt sich $\mathrm{d}a/\mathrm{d}w = 2, \mathrm{d}b/\mathrm{d}w = 6 \implies \mathrm{d}u/\mathrm{d}w = 160, \mathrm{d}v/\mathrm{d}w = 64 \implies \mathrm{d}f/\mathrm{d}w = 328(160 + 64) = 73{,}472$.

<!--en:860f9882-->
- **Warum der Aufwand im Vorwärtsmodus mit $N$ rasch wächst:** Die Tangenten $\mathrm{d}u/\mathrm{d}w$ und $\mathrm{d}v/\mathrm{d}w$ an den Zwischenknoten gelten nur für $w$. Daher braucht der Vorwärtsmodus $N$ separate Durchläufe ($4 \times 6 = 24$ Multiplikationen) für $N$ Eingaben.

<!--en:13504114-->
::: {.slide title="Vorwärtsmodus in reinem Python"}

<!--en:f989df99-->
[Überprüfung mit Code]{.kicker}

<!--en:cd26a198-->
Der Vorwärtsdurchlauf in Python mit anschließender Vorwärtsrechnung für die Sensitivität bezüglich $w$ bestätigt $\mathrm{d}f/\mathrm{d}w = 73472$. Für diese eine Eingabe sind $6$ Multiplikationen nötig:

<!--en:dcf580d5-->
::: {.slide title="Vorwärtsmodus in reinem Python (Fortsetzung)"}

<!--en:f989df99-2-->
[Überprüfung mit Code]{.kicker}

<!--en:592d05ac-->
Für den vollständigen Gradienten bezüglich aller $4$ Eingaben benötigt der Vorwärtsmodus $4 \times 6 = 24$ Multiplikationen ($N$ separate Vorwärtsdurchläufe für $N$ Eingaben).

<!--en:05832826-->
::: {.slide title="Adjungierte Variablen und das Ausklammern im Rückwärtsmodus"}

<!--en:9efce640-->
[Rückwärtsakkumulation]{.kicker}

<!--en:8c634445-->
- **Perspektivwechsel zur festen Ausgabe $f$:** Die **adjungierte Variable** $\bar{n} = \frac{\mathrm{d}f}{\mathrm{d}n}$ eines Knotens $n$ ist die totale Ableitung der Ausgabe $f$ nach $n$. Der Startwert ist $\bar{f} = 1$.

<!--en:5e555481-->
- **Pfade nach ihrer ersten Kante gruppieren:** Jeder Pfad von $n$ nach $f$ führt zuerst zu einem direkten Nachfolger $c$ von $n$. Die Beiträge der verbleibenden Pfade von $c$ nach $f$ summieren sich zu $\bar{c}$:
  $$\bar{n} = \sum_{c \in \mathrm{children}(n)} \bar{c}\,\frac{\partial c}{\partial n}$$

<!--en:d5d3e295-->
- **Ein Rückwärtsdurchlauf für alle Eingaben:** In umgekehrter topologischer Reihenfolge ist die adjungierte Variable $\bar{c}$ jedes direkten Nachfolgers bereits bekannt, wenn die Berechnung von $\bar{n}$ beginnt.

<!--en:c41ef000-->
::: {.slide title="Der Rückwärtsdurchlauf Knoten für Knoten"}

<!--en:ec6dc05e-->
[Rechnung von Hand (Schritt 2)]{.kicker}

<!--en:2b5a03c6-->
- **Rekursion für die adjungierten Variablen von oben nach unten ($\bar{n} = \sum_c \bar{c}\frac{\partial c}{\partial n}$):** Vom Startwert aus ergibt sich $\bar{f} = 1 \implies \bar{u} = \bar{v} = 328 \implies \bar{a} = 1{,}312, \bar{b} = 11{,}808$.

<!--en:1312de57-->
- **Alle 4 Eingaben mit 12 Multiplikationen:** $\bar{w} = \bar{x} = 1312(2) + 11808(6) = 73{,}472$ und $\bar{y} = \bar{z} = 1312(2) - 11808(6) = -68{,}224$.

<!--en:2eb455af-->
::: {.slide title="Rückwärtsdurchlauf in reinem Python"}

<!--en:f989df99-3-->
[Überprüfung mit Code]{.kicker}

<!--en:1052c74e-->
Da $\bar{a} = 1312$ und $\bar{b} = 11808$ bereits den gesamten oberen Teil des Graphen zusammenfassen, kostet jede weitere Eingabe nur $2$ Multiplikationen in der untersten Schicht:

<!--en:d69a416f-->
Symmetrieprüfung: $w$ und $x$ gehen in $a$ und $b$ nur über $w + x$ ein. Daher gilt $\bar{w} = \bar{x} = 73472$. Ebenso gehen $y$ und $z$ nur über $y + z$ ein, mit negativem Vorzeichen in $b$. Daraus folgt $\bar{y} = \bar{z} = -68224$.

<!--en:fd1a39e6-->
::: {.slide title="Adjungierte Variablen aus der Handrechnung mit Autograd überprüfen"}

<!--en:9964fa76-->
[Ausführung im Framework]{.kicker}

<!--en:63e288a5-->
Jedes moderne System zum automatischen Differenzieren verwendet genau diesen zweistufigen Algorithmus. Im Vorwärtsdurchlauf speichert es die Werte, beim Aufruf von `backward()` summiert es die Beiträge zu den adjungierten Variablen:

<!--en:427a0e63-->
Das Framework liefert `[73472., 73472., -68224., -68224.]`. Das Ergebnis stimmt Bit für Bit mit der Handrechnung überein. Alle ganzzahligen Zwischenwerte liegen dem Betrag nach unter $2^{24} \approx 1.68 \times 10^7$. Bis zu dieser Grenze stellt das IEEE-754-Format `float32` jede ganze Zahl exakt dar.

<!--en:61726cf6-->
::: {.slide title="Stolperstein in der Praxis: Gradientenprüfung mit Float32"}

<!--en:1b718c47-->
[Numerische Überprüfung]{.kicker}

<!--en:184a2502-->
- **Gegenläufige Fehler beim zentralen Differenzenquotienten:** Der Abbruchfehler ist von der Ordnung $O(h^2)$ für große $h$, der Fehler durch Auslöschung von der Ordnung $O(\varepsilon_{\text{mach}}|f|/h)$ für kleine $h$. Daraus folgen die optimale Schrittweite $h^\star \propto \varepsilon_{\text{mach}}^{1/3}$ und der minimale Fehler der Ordnung $O(\varepsilon_{\text{mach}}^{2/3})$.

<!--en:fc0dffbe-->
- **Warum `float64` nötig ist:** Mit `float64` sinkt der relative Fehler auf $\sim 10^{-11}$ bei $h^\star \approx 10^{-5}$. Mit `float32` bleibt er bei $\sim 10^{-4}$. Durch Rundung gilt $w + h == w$, und der Differenzenquotient wird für $h \le 10^{-8}$ null.

<!--en:56d4b52f-->
::: {.slide title="Verständnisfragen zur Kettenregel auf Berechnungsgraphen"}

<!--en:b6d9a92f-->
[Wiederholung und Übung]{.kicker}

<!--en:e76c05a1-->
Drei Fragen zu Berechnungsgraphen und adjungierten Variablen beantworten:

<!--en:417a71d1-->
1. **Kantenableitung und totale Ableitung:** Gegeben sei der Graph $p = x + y$, $r = p^2$, $f = r + 3p$. Wie groß sind die Kantenableitung $\frac{\partial f}{\partial p}$ und die totale Ableitung (adjungierte Variable) $\bar{p} = \frac{\mathrm{d}f}{\mathrm{d}p}$ an der Stelle $p = 4$?

<!--en:a47c490d-->
2. **Gemeinsam genutzte adjungierte Variable:** Ein Netz mit zwei versteckten Einheiten $h_1, h_2$ liefert die Vorhersage $f = v_1 h_1 + v_2 h_2$ und den Verlust $\ell = \tfrac{1}{2}(f - t)^2$. Welche adjungierte Variable muss nur einmal berechnet werden und geht dann in die Ableitung nach jedem Gewicht ein?

<!--en:b73da965-->
3. **Symmetrie bei der Gewichtsinitialisierung:** Zwei versteckte Einheiten $h_1 = \tanh(w_1 x + b_1)$ und $h_2 = \tanh(w_2 x + b_2)$ werden identisch initialisiert ($w_1 = w_2$, $b_1 = b_2$, $v_1 = v_2$). Was geschieht bei der Backpropagation mit ihren adjungierten Variablen $\bar{w}_1, \bar{w}_2$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:5bfb03e0-->
1. Bei festem $r$ beträgt die direkte Kantenableitung $\frac{\partial f}{\partial p} = 3$. Die totale Ableitung summiert die Beiträge beider Pfade, die von $p$ ausgehen: $\bar{p} = \frac{\partial f}{\partial r}\frac{\partial r}{\partial p} + \frac{\partial f}{\partial p} = 1(2p) + 3 = 2(4) + 3 = 11$.

<!--en:29209686-->
2. Jeder Pfad von einem Parameter zu $\ell$ führt über $f$. Damit genügt es, die adjungierte Variable der Vorhersage $\bar{f} = \frac{\mathrm{d}\ell}{\mathrm{d}f} = f - t$ einmal zu berechnen und für alle Parameter zu verwenden.

<!--en:fb20f7c0-->
3. Wegen der Symmetrie der beiden Pfade gilt in jedem Schritt $\bar{h}_1 = \bar{h}_2$, $\bar{w}_1 = \bar{w}_2$ und $\bar{b}_1 = \bar{b}_2$. Der Gradientenabstieg aktualisiert beide Einheiten daher bei jedem weiteren Schritt identisch. Die Initialisierung versteckter Einheiten muss diese Symmetrie durch Zufall brechen.

<!--en:ccb5a959-->
::: {.slide title="Zusammenfassung: Berechnungsgraphen und Backpropagation"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:6d283e1c-->
| Begriff oder Verfahren | Mathematische Formel | Bedeutung für Berechnung und Architektur |

<!--en:f1f67057-->
| **Kantenableitung ($\partial$)** | $\frac{\partial c}{\partial n}$ (alle anderen direkten Vorgänger von $c$ bleiben fest) | Lokale Sensitivität einer einzelnen elementaren Operation |

<!--en:d4f5c287-->
| **Totale Ableitung ($\mathrm{d}$ / $\bar{n}$)** | $\bar{n} = \frac{\mathrm{d}f}{\mathrm{d}n}$ (alle nachgelagerten Knoten reagieren) | Gesamte Sensitivität der Ausgabe $f$ bezüglich des Zwischenknotens $n$ |

<!--en:62309da6-->
| **Kettenregel als Pfadsumme** | $\frac{\mathrm{d}f}{\mathrm{d}x} = \sum_{\text{paths } x \to f} \prod_{(u \to v)} \frac{\partial v}{\partial u}$ | Erklärt den direkten Gradientenfluss über Skip-Verbindungen ($2^m$ Pfade in ResNets) |

<!--en:dbf5df8e-->
| **Durchlauf im Vorwärtsmodus** | $\dot{n} = \frac{\mathrm{d}n}{\mathrm{d}x_j} = \sum_{p \in \mathrm{parents}(n)} \frac{\partial n}{\partial p}\dot{p}$ | Gibt den Startwert einer Eingabe vorwärts weiter. Erfordert $N$ Durchläufe für $N$ Parameter |

<!--en:2b231e80-->
| **Rückwärtsmodus (Backpropagation)** | $\bar{f} = 1$, $\bar{n} = \sum_{c \in \mathrm{children}(n)} \bar{c}\frac{\partial c}{\partial n}$ | Gibt den Startwert einer Ausgabe rückwärts weiter. Berechnet alle $N$ Gradienten in $1$ Durchlauf |

<!--en:326e6598-->
| **Prüfung mit zentralem Differenzenquotienten** | $\frac{f(w+h)-f(w-h)}{2h} = f'(w) + O(h^2) + O(\frac{\varepsilon_{\text{mach}}}{h})$ | Erfordert `float64` ($h \approx 10^{-5}$), um die durch Rundung in `float32` bedingte Fehlergrenze zu unterschreiten |
