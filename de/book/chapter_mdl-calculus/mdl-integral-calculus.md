<!--en:81957c7e-->
# Integralrechnung

<!--en:73387004-->
In früheren Kapiteln haben wir stetige Zufallsvariablen verwendet, ohne zu fragen, wie sich ihre Wahrscheinlichkeiten berechnen lassen. Wir haben eine Normalverteilung durch ihre Dichte beschrieben (:numref:`subsec_normal_distribution_and_squared_loss`) und den erwarteten Verlust als Mittelwert bezüglich dieser Dichte geschrieben. Beides haben wir als gegeben hingenommen. Jetzt fragen wir, wie wir eine Größe aufsummieren, die sich stetig über ein Intervall verteilt.

<!--en:5dd8cefa-->
Die Standardnormalverteilung mit ihrer vertrauten Glockenkurve zeigt, warum wir beim stetigen Aufsummieren sorgfältig vorgehen müssen. Eine standardnormalverteilte Zufallsvariable $Z$ hat die Dichte $p(z)=e^{-z^2/2}/\sqrt{2\pi}$. Bei $z=0$ erreicht die Kurve die Höhe $p(0)\approx 0.40$. Dennoch beträgt die Wahrscheinlichkeit, genau null zu treffen, $P(Z=0)=0$. Dasselbe gilt für jeden anderen einzelnen Punkt. Intervalle dagegen können positive Wahrscheinlichkeit haben: $P(-1\le Z\le 1)\approx 0.68$. Geometrisch ist diese Zahl die Fläche unter $p$ zwischen $-1$ und $1$. Der Faktor $1/\sqrt{2\pi}$ sorgt dafür, dass die gesamte Fläche unter der Kurve gleich $1$ ist. Woher kommt $\sqrt{2\pi}$ und wie berechnen wir $0.68$?

<!--en:e7ee3c58-->
Diese Fragen reichen weit über die Glockenkurve hinaus. Jede Wahrscheinlichkeit und jeder erwartete Verlust für eine stetige Zufallsvariable entspricht einer solchen Fläche. Trainieren wir ein Modell durch zufälliges Sampling, wie bei Variationsautoencodern und Policy-Gradient-Verfahren, müssen wir eine solche Fläche auch nach einem Modellparameter ableiten.

<!--en:c8aae28d-->
Wir zerlegen die Fläche in schmale Streifen, addieren deren Flächeninhalte und lassen die Streifenbreite gegen null gehen. Der Grenzwert ist das Integral. Der Hauptsatz der Differential- und Integralrechnung (engl. fundamental theorem of calculus) ersetzt diese unendliche Summation durch die Umkehrung der Differentiation. Damit lassen sich die meisten Integrale in einer Zeile berechnen. Bei der Glockenkurve gelingt das nicht. Für ihre Normierungskonstante $\sqrt{2\pi}$ brauchen wir einen Umweg über zwei Dimensionen. Drei Ergebnisse stehen im Mittelpunkt: der Hauptsatz :eqref:`eq_mdl-ftc`, die Formel für die Variablentransformation in mehreren Dimensionen :eqref:`eq_mdl-change_var_nd` und das Gauß-Integral :eqref:`eq_mdl-gaussian-integral`. Wenn Sie mit der Analysis des ersten Studienjahrs vertraut sind, können Sie den Text bis :numref:`subsec_mdl-improper-integrals` überfliegen. Wie wir ein Integral nach einem Parameter ableiten, behandeln wir auf dieser Grundlage in einem eigenen Abschnitt (:numref:`sec_mdl-expectation-gradients`). Zur Maßtheorie und zum Lebesgue-Integral verweisen wir auf :numref:`sec_mdl-calculus-further-reading`.

<!--en:2211b404-->
## Das bestimmte Integral

<!--en:af6782fc-->
### Riemann-Summen

<!--en:967307db-->
Eine Wahrscheinlichkeitsdichte gibt die Wahrscheinlichkeit pro Längeneinheit an. Für ein kurzes Intervall der Breite $\epsilon$ entspricht die Wahrscheinlichkeit daher näherungsweise dem Produkt aus Dichte und Breite. Ein längeres Intervall zerlegen wir in schmale Streifen. Für jeden Streifen multiplizieren wir Höhe und Breite und addieren die Ergebnisse. Dieses Vorgehen funktioniert für beliebige Funktionen, also auch für Dichten.

<!--en:0ade4caf-->
Beginnen wir mit einem Beispiel, das wir von Hand ausrechnen können. Wir betrachten $f(x)=x^2$ auf $[0,1]$ und zerlegen das Intervall in $N$ Streifen gleicher Breite $\epsilon=1/N$. Auf dem $i$-ten Teilintervall $[i/N,(i+1)/N]$ nähern wir die Fläche durch ein Rechteck an. Seine Höhe ist der Wert von $f$ am linken Randpunkt, also $(i/N)^2$. Addieren wir die Flächen aller $N$ Rechtecke, erhalten wir

<!--en:93004c0d-->
Im mittleren Schritt verwenden wir die bekannte Identität $\sum_{i=0}^{N-1}i^2=(N-1)N(2N-1)/6$. Mit wachsendem $N$ konvergieren die Rechtecksummen gegen $\tfrac13$. Ihre Abweichung vom Grenzwert beträgt ungefähr $\tfrac1{2N}$. Der Fehler ist also von erster Ordnung in der Streifenbreite $\epsilon$.

<!--en:2e9ddb04-->
Die Hilfsfunktion `left_sum` setzt diese **linke Rechteckregel** (engl. left-endpoint rule) für jede Funktion $f$ um. Sie zerlegt $[a,b]$ in Streifen der Breite $\epsilon$ und wertet $f$ am linken Randpunkt jedes Streifens aus, wie :numref:`fig_mdl-riemann` zeigt. Wir verwenden `left_sum` für alle Gittersummen in diesem Abschnitt und prüfen die Funktion zunächst an $x^2$.

<!--en:e08dedd3-->
Die numerischen Summen stimmen in allen ausgegebenen Stellen mit der geschlossenen Formel überein.

<!--en:a9584cc3-->
**Definition.** Diese Konstruktion setzt weder gleich breite Streifen noch die Wahl der linken Randpunkte voraus. Allgemein unterteilt eine *Zerlegung* (engl. partition) $P:a=x_0<x_1<\cdots<x_N=b$ das Intervall $[a,b]$ in Teilintervalle. In jedem wählen wir eine *Stützstelle* (engl. tag) $\xi_i\in[x_{i-1},x_i]$. Die *Feinheit* (engl. mesh) $\|P\|=\max_i(x_i-x_{i-1})$ gibt die Breite des breitesten Teilintervalls an. Eine beschränkte Funktion $f$ heißt **Riemann-integrierbar** auf $[a,b]$, wenn es eine einzige Zahl $I$ gibt, der sich alle so gebildeten Summen nähern, sobald die Feinheit gegen null geht. Dies muss für jede Wahl der Stützstellen gelten:

<!--en:8e16c1b6-->
Das bestimmte Integral ist also der gemeinsame Grenzwert der Summen, wenn die Streifen beliebig schmal werden. Unsere Rechnung von Hand ergibt $\int_0^1x^2\,dx=\tfrac13$. Jede stetige Funktion auf $[a,b]$ ist Riemann-integrierbar. Das gilt auch für jede beschränkte Funktion mit nur endlich vielen Unstetigkeitsstellen, die alle Sprungstellen sind :cite:`Rudin.1976`. Manche pathologischen beschränkten Funktionen besitzen kein Riemann-Integral, etwa die Indikatorfunktion der rationalen Zahlen. Das Lebesgue-Integral (:numref:`sec_mdl-calculus-further-reading`) erfasst auch solche Funktionen und stimmt mit dem Riemann-Integral überein, wo dieses existiert.

<!--en:4ff489cd-->
Aus der Definition folgen unmittelbar zwei elementare Eigenschaften, die wir für die folgenden Beweise brauchen. Erstens ist das Integral **monoton**. Gilt $f\le g$ überall auf $[a,b]$, ist jede Riemann-Summe von $f$ höchstens so groß wie die entsprechende Summe von $g$. Beim Grenzübergang bleibt die Ungleichung erhalten, also gilt $\int_a^b f\,dx\le\int_a^b g\,dx$. Zweitens ist das Integral **additiv über Teilintervalle**. Für beliebige $a\le b\le c$ gilt $\int_a^c f\,dx=\int_a^b f\,dx+\int_b^c f\,dx$, denn jede Zerlegung mit dem Punkt $b$ zerfällt in eine Zerlegung von $[a,b]$ und eine von $[b,c]$.

<!--en:2bba8b76-->
**Unser durchgehendes Beispiel.** Bis zum Gauß-Integral begleitet uns ein konkretes Beispiel:

<!--en:776f4223-->
Warum der Wert $\tfrac12\log5$ beträgt, sehen wir, sobald wir die Auswertungsformel :eqref:`eq_mdl-ftc-eval` hergeleitet haben. :numref:`fig_mdl-riemann` zeigt für drei Streifenbreiten die Rechtecke unter dieser Kurve, deren Höhe der linke Randpunkt bestimmt.

<!--en:d4d913a1-->
![Rechtecke mit linken Stützstellen für $f(x)=x/(1+x^2)$ auf $[0,2]$ bei den Streifenbreiten $\epsilon=0.5$, $0.2$ und $0.05$. Die Höhe jedes Rechtecks ist der Wert von $f$ am linken Randpunkt. Die Rechtecksummen nähern sich der Fläche $\tfrac12\log 5$ für $\epsilon\to0$.](../img/mdl-cal-riemann.svg)

<!--en:33d74fb0-->
Wie schnell konvergieren die linken Rechtecksummen gegen $\tfrac12\log5$? Für $x^2$ auf $[0,1]$ betrug der Fehler näherungsweise $\tfrac1{2N}=\tfrac\epsilon2$. Ist Konvergenz erster Ordnung typisch, sollte ein Zehntel der Streifenbreite den Fehler auf ein Zehntel verringern. Prüfen wir das, indem wir den Fehler neben $0.2\,\epsilon$ ausgeben.

<!--en:2246bccb-->
Jedes Mal, wenn $\epsilon$ um den Faktor zehn sinkt, sinkt auch der Fehler um den Faktor zehn. Jede Summe liegt unter dem exakten Wert. Die Konstante $0.2$ ist die Hälfte von $f(2)-f(0)=0.4$. Das ist kein Zufall. Sobald wir den Hauptsatz kennen, leiten wir die zugehörige Formel her.

<!--en:20ce65d1-->
### Orientierter Flächeninhalt

<!--en:42f8d280-->
Rechtecke unterhalb der horizontalen Achse haben eine negative Höhe $f(\xi_i)<0$ und gehen mit einem Minuszeichen in die Riemann-Summe ein. Das Integral misst daher den *orientierten* Flächeninhalt. Bereiche mit $f<0$ liefern einen negativen Beitrag, etwa bei $\int_0^1(-1)\,dx=-1$. Vereinbarungsgemäß kehrt auch die Integration von rechts nach links das Vorzeichen um: $\int_b^a f\,dx=-\int_a^b f\,dx$. Damit gilt die Additivität unabhängig von der Reihenfolge von $a$, $b$ und $c$. :numref:`fig_mdl-signed-area` veranschaulicht beide Vorzeichenkonventionen an derselben Kurve.

<!--en:dd655c43-->
![Orientierter Flächeninhalt. Die Flächenstücke oberhalb der Achse zählen positiv, das Stück unterhalb zählt negativ. So summiert $\int_a^b f$ die schattierten Flächenstücke mit ihrem jeweiligen Vorzeichen. Bei vertauschten Grenzen verläuft die Integration von rechts nach links, und die Summe wechselt ihr Vorzeichen.](../img/mdl-cal-signed-area.svg)

<!--en:3e4441c9-->
## Der Hauptsatz der Differential- und Integralrechnung

<!--en:8b88b7d2-->
### Der Satz

<!--en:2b543337-->
Durch Aufsummieren der Streifen erhielten wir $\tfrac13$ für $x^2$ nur deshalb, weil es für $\sum i^2$ eine geschlossene Formel gibt. Bei unserem durchgehenden Beispiel lautet die linke Rechtecksumme $\sum_i\epsilon\cdot i\epsilon/(1+i^2\epsilon^2)$. Keine algebraische Identität vereinfacht sie. Wir brauchen ein Verfahren, mit dem wir Integrale ganz ohne Streifensummen berechnen können.

<!--en:cb3d5695-->
Der entscheidende Schritt besteht darin, die obere Grenze als Variable zu betrachten. Wir wählen einen festen Anfangspunkt $a$ und definieren die **Integralfunktion** (engl. area-so-far function)

<!--en:e90860c0-->
die den orientierten Flächeninhalt von $a$ bis $x$ angibt. In dieser einen Funktion stecken alle bestimmten Integrale von $f$. Wegen der Additivität ist der Flächeninhalt über jedem Intervall $[x_1,x_2]$ die Differenz zweier vom Anfangspunkt aus gemessener Flächeninhalte,

<!--en:0cad1a8f-->
wie :numref:`fig_mdl-area-subtract` zeigt. Kennen wir $F$, ist das ganze Problem gelöst.

<!--en:fff7de60-->
![Die Fläche unter $f$ von $x_1$ bis $x_2$ als Differenz zweier Flächen, die beide am Anfangspunkt $a$ am linken Bildrand beginnen: $\int_{x_1}^{x_2} f\,dx=\int_a^{x_2} f\,dx-\int_a^{x_1} f\,dx$.](../img/mdl-cal-sub-area.svg)

<!--en:28ae0544-->
Wie ändert sich $F$, wenn wir $x$ ein wenig nach rechts verschieben? :numref:`fig_mdl-ftc-sliver` zeigt die Fläche bis $x$ und den schmalen Streifen der Breite $\epsilon$, der beim Übergang zu $x+\epsilon$ hinzukommt. Dieser Streifen entspricht annähernd einem Rechteck der Höhe $f(x)$. Daher gilt $F(x+\epsilon)-F(x)\approx\epsilon f(x)$. Division durch $\epsilon$ legt die Beziehung $F'(x)=f(x)$ nahe.

<!--en:235afbdb-->
![Die Fläche $F(x)$ unter $f(y)=y/(1+y^2)$ von $a=0$ bis $x=1.2$ (hell) und der Streifen, der beim Übergang zu $x+\epsilon$ hinzukommt (dunkel). Der Streifen entspricht annähernd dem gestrichelten Rechteck der Höhe $f(x)$, also gilt $F(x+\epsilon)-F(x)\approx\epsilon f(x)$.](../img/mdl-cal-ftc-sliver.svg)

<!--en:e11e10e8-->
**Satz (Hauptsatz der Differential- und Integralrechnung).** *Sei $f$ auf $[a,b]$ Riemann-integrierbar und sei $F(x)=\int_a^x f(y)\,dy$. Ist $f$ an einer Stelle $x\in(a,b)$ stetig, so ist $F$ an der Stelle $x$ differenzierbar und es gilt*

<!--en:168c283b-->
*In Worten: Die Rate, mit der die Fläche wächst, ist gleich der momentanen Höhe der Kurve.*

<!--en:e2549c0c-->
**Beweis.** Verschieben wir $x$ um $\epsilon>0$ und betrachten den Flächenzuwachs. Nach :eqref:`eq_mdl-area-fn` und wegen der Additivität entspricht die Änderung einem einzigen schmalen Streifen:

<!--en:bd2f9058-->
Seien $m_\epsilon = \inf_{[x,x+\epsilon]} f$ und $M_\epsilon = \sup_{[x,x+\epsilon]} f$ die engsten konstanten Schranken für $f$ auf diesem Streifen. Beide sind endlich, da eine Riemann-integrierbare Funktion beschränkt ist. Es gilt $m_\epsilon\le f\le M_\epsilon$ auf ganz $[x,x+\epsilon]$. Wegen der Monotonie des Integrals liegt der Flächeninhalt des Streifens zwischen den Flächeninhalten zweier Rechtecke der Breite $\epsilon$:

<!--en:54357a0c-->
Dividieren wir nun durch $\epsilon$. Da $f$ bei $x$ stetig ist, liegt für jedes $\delta>0$ schließlich das gesamte Intervall $[x,x+\epsilon]$ in dem Bereich, in dem $|f(y)-f(x)|<\delta$ gilt. Sowohl $m_\epsilon$ als auch $M_\epsilon$ konvergieren daher gegen $f(x)$ für $\epsilon \to 0$. Aus diesen Schranken folgt $\lim_{\epsilon\to 0}\frac{F(x+\epsilon)-F(x)}{\epsilon}=f(x)$. Für $\epsilon<0$ liegt der Streifen links von $x$. Es gelten dieselben Schranken, da sich im Differenzenquotienten beide Vorzeichenwechsel aufheben. Der Grenzwert ist $F'(x)$. $\blacksquare$

<!--en:c0757783-->
Im Beweis haben wir nur Monotonie und Additivität verwendet, die beiden Eigenschaften, die unmittelbar aus der Definition folgen. Statt einen Flächeninhalt zu berechnen, können wir nach dem Hauptsatz eine **Stammfunktion** (engl. antiderivative) suchen, also eine Funktion $G$ mit $G'=f$. Oft lässt sie sich direkt aus der Ableitungstabelle in :numref:`sec_mdl-derivative_table` ablesen. Jede Stammfunktion liefert das bestimmte Integral über die **Auswertungsformel**: *Ist $f$ auf $[a,b]$ stetig und gilt $G'=f$ auf ganz $[a,b]$, so gilt*

<!--en:b85506ac-->
Warum funktioniert das für jede Stammfunktion und nicht nur für $F$? Die Funktionen $F$ und $G$ haben dieselbe Ableitung $f$ auf $(a,b)$. Die Ableitung ihrer Differenz $G-F$ ist also null. Nach dem Mittelwertsatz (:numref:`sec_mdl-mvt`) ist die Differenz daher konstant. Diese additive Konstante fällt in $G(b)-G(a)=F(b)-F(a)=\int_a^b f\,dx$ weg.

<!--en:8d02e0d5-->
Die Voraussetzung „auf ganz $[a,b]$“ ist wesentlich. Wegen $(-1/x)'=x^{-2}$ ergäbe blindes Einsetzen $\int_{-1}^1x^{-2}\,dx=[-1/x]_{-1}^1=-2$, also einen negativen Wert für einen streng positiven Integranden. Die Regel ist hier nicht anwendbar, denn $G'=f$ gilt nicht bei $x=0$. Dort ist keine der beiden Funktionen definiert. Tatsächlich divergiert das Integral (:numref:`subsec_mdl-improper-integrals`).

<!--en:7a88cc9f-->
Rückwärts gelesen liefert jede Ableitungsformel ein Integral. Aus $(y^3)'=3y^2$ folgt $\int_0^1y^2\,dy=\tfrac13$, ganz ohne Streifensumme. Aus $(e^t)'=e^t$ folgt $\int_0^xe^t\,dt=e^x-1$. Versuchen wir es bei unserem Beispiel mit $G(x)=\tfrac12\log(1+x^2)$. Nach der Kettenregel gilt $G'(x)=\tfrac12\cdot\frac{2x}{1+x^2}=\frac{x}{1+x^2}$. Damit ergibt sich

<!--en:dda8efd9-->
Das bestätigt den Grenzwert, dem sich unsere Riemann-Summen näherten.

<!--en:300688aa-->
**Der Fehler der linken Rechteckregel.** Die Auswertungsformel erklärt auch die Konstante $0.2$ im Fehler der linken Rechtecksumme. Auf dem Teilintervall mit Anfangspunkt $x_i$ liefert die Entwicklung erster Ordnung $f(x)\approx f(x_i)+f'(x_i)(x-x_i)$. Integrieren wir über $[x_i,x_i+\epsilon]$, so übersteigt die tatsächliche Fläche die Rechteckfläche $\epsilon f(x_i)$ um ungefähr $\tfrac{\epsilon^2}2f'(x_i)$. Addieren wir diese dreieckigen Flächenbeiträge für alle Streifen, erhalten wir $\tfrac\epsilon2$ mal die linke Rechtecksumme für $f'$. Das ist eine Näherung für $\tfrac\epsilon2\int_a^bf'\,dx=\tfrac\epsilon2(f(b)-f(a))$. In erster Ordnung in $\epsilon$ gilt

<!--en:31ed91a8-->
Für $x^2$ auf $[0,1]$ ergibt das $-\tfrac\epsilon2=-\tfrac1{2N}$, genau den führenden Fehlerterm aus unserer Rechnung von Hand. Im durchgehenden Beispiel liefern $f(0)=0$ und $f(2)=0.4$ den Fehlerterm $-0.2\,\epsilon$. Diese Abschätzung gehört zur numerischen Analysis. Sie erklärt den verbleibenden Fehler in drei weiteren Experimenten dieses Abschnitts und dient in :numref:`subsec_mdl-monte-carlo` als Vergleichsmaßstab für zufälliges Sampling.

<!--en:d1600935-->
Der Hauptsatz sagt mehr aus als $\int_0^2f=G(2)-G(0)$. Die Integralfunktion $F(x)$ muss gleich $G(x)-G(0)$ sein, und zwar an jeder Stelle $x$. Prüfen wir das. Dazu berechnen wir $F$ als kumulierte linke Rechtecksumme auf $[0,2]$ und vergleichen sie an jedem Gitterpunkt mit $G(x)-G(0)$. Eine Übereinstimmung bis auf Maschinengenauigkeit ist nicht zu erwarten, denn die linke Rechtecksumme auf $[0,x]$ hat den systematischen Fehler $\tfrac\epsilon2\bigl(f(0)-f(x)\bigr)$. Sein Betrag ist dort am größten, wo $f$ sein Maximum erreicht, also bei $x=1$ mit $f(1)=\tfrac12$. Die größte Abweichung im Intervall sollte daher näherungsweise $\tfrac\epsilon4$ betragen.

<!--en:771398f4-->
Bei jeder Streifenbreite beträgt die maximale Abweichung wie vorhergesagt ungefähr $\tfrac\epsilon4$. Die kumulierte Summe folgt der Stammfunktion auf dem gesamten Intervall bis hin zum rechten Randpunkt $x=2$.

<!--en:8e74dd3a-->
### Uneigentliche Integrale

<!--en:6eadbc44-->
Die Wahrscheinlichkeitsdichten der Normalverteilung, der Exponentialverteilung und vieler weiterer Verteilungen sind auf unbeschränkten Bereichen definiert. Dagegen definiert :eqref:`eq_mdl-riemann` Integrale nur für ein endliches Intervall $[a,b]$. Wir erweitern die Definition mit demselben Mittel, mit dem wir sie gewonnen haben: einem Grenzübergang. Das **uneigentliche Integral** (engl. improper integral) über $[a,\infty)$ ist

<!--en:5dc23f38-->
also der bis $b$ aufsummierte Flächeninhalt im Grenzwert für wachsendes $b$. Existiert dieser Grenzwert und ist er endlich, so *konvergiert* das Integral. Andernfalls *divergiert* es. Für die Integration über die ganze reelle Achse zerlegen wir $\int_{-\infty}^\infty f\,dx$ an einer beliebigen Stelle in zwei einseitige uneigentliche Integrale. *Beide* müssen unabhängig voneinander konvergieren. Ein symmetrischer Grenzwert genügt nicht. So gilt $\int_{-b}^b x\,dx=0$ für jedes $b$, aber $\int_0^\infty x\,dx=\infty$. Daher existiert $\int_{-\infty}^\infty
x\,dx$ nicht.

<!--en:9cf0fe3f-->
Das Potenzgesetz $x^{-s}$ mit Exponent $s>0$ ist das übliche Vergleichskriterium dafür, ob eine Funktion im Unendlichen schnell genug abfällt, um eine endliche Fläche einzuschließen. Wenden wir die Auswertungsformel auf $[1,b]$ an und bilden den Grenzwert für $b\to\infty$, so erhalten wir

<!--en:32554d86-->
(Bei $s=1$ hat der Ausdruck die Form $0/0$. Das bestimmte Integral beträgt dann $\log b$. Das ist der Grenzwert der Formel für $s\to 1$, und er divergiert für $b\to\infty$.) Somit gilt $\int_1^\infty x^{-2}\,dx=1$, aber $\int_1^\infty x^{-1}\,dx=\infty$. Die Grenze zwischen endlichem und unendlichem Flächeninhalt liegt genau bei $s=1$.

<!--en:e894816c-->
Potenzgesetze sind typische Beispiele für Verteilungen mit **schweren Rändern** (engl. heavy-tailed) (siehe :numref:`sec_mdl-random_variables`, wo wir auch ihren Einfluss auf Mittelwerte untersuchen). Fällt eine Dichte im Unendlichen wie $p(x)\propto x^{-s}$ ab, ist die Normierungskonstante nur für $s>1$ endlich. Wenden wir dasselbe Kriterium auf $x\cdot x^{-s}=x^{1-s}$ an, sehen wir, dass der Erwartungswert nur für $s>2$ endlich ist.

<!--en:f39f6695-->
Derselbe Grenzübergang erfasst auch Integranden, die an einem Randpunkt unbeschränkt sind: $\int_0^1 f\,dx=\lim_{a\to0^+}\int_a^1 f\,dx$. Für das Potenzgesetz $x^{-s}$ kehrt sich die Bedingung um:

<!--en:702f2a49-->
Für $s\ge1$ hingegen divergiert das Integral. Bei $s=\tfrac12$ wächst beispielsweise $x^{-1/2}$ in der Nähe von $0$ über alle Grenzen. Dennoch beträgt die Fläche darunter nur $2$. Damit die Fläche endlich bleibt, muss eine Potenzfunktion im Unendlichen schneller fallen als $1/x$. In der Nähe von null muss sie langsamer wachsen als $1/x$.

<!--en:b26a8fb9-->
Wenn wir ein uneigentliches Integral numerisch annähern, entstehen zwei verschiedene Fehler. Wir schneiden den Integrationsbereich bei einem endlichen Wert $b$ ab und summieren Streifen endlicher Breite $\epsilon$. Für $\int_0^\infty e^{-x}\,dx = 1$ können wir beide Fehler getrennt betrachten. Unten geben wir das exakte Teilintegral $\int_0^b e^{-x}\,dx = 1-e^{-b}$ (aus der Stammfunktion $-e^{-x}$) neben einer linken Rechtecksumme mit $\epsilon = 10^{-3}$ aus.

<!--en:c1435161-->
Die exakten Teilintegrale konvergieren schnell gegen $1$. Der *Abbruchfehler* an der oberen Grenze $b$ beträgt $e^{-b}$ und liegt bei $b=20$ bereits unter einer Einheit der sechsten Nachkommastelle. Die linken Rechtecksummen nähern sich hingegen $1.0005$ statt $1$. Die verbleibende Abweichung ist der *Diskretisierungsfehler*, also der systematische Fehler der linken Rechteckregel $\tfrac{\epsilon}{2}\bigl(f(0)-f(b)\bigr)\to\tfrac{\epsilon}{2}=0.0005$. In führender Ordnung addieren sich die beiden Fehler. Für wachsendes $b$ verschwindet der Abbruchfehler, für abnehmendes $\epsilon$ der Diskretisierungsfehler.

<!--en:4c957a4a-->
### Partielle Integration

<!--en:6b0f29d6-->
Über die Auswertungsformel liefert jede Ableitungsregel eine entsprechende Integrationsregel. Aus der Produktregel entsteht die partielle Integration.

<!--en:27a6f6bf-->
Für stetig differenzierbare Funktionen $u$ und $v$ auf $[a,b]$ gilt

<!--en:ad961390-->
Dabei steht $[\,h(x)\,]_a^b$ für $h(b)-h(a)$. Der Beweis passt in eine Zeile: Nach der Produktregel ist $uv$ eine Stammfunktion von $u'v + uv'$. Aus :eqref:`eq_mdl-ftc-eval` folgt daher $\int_a^b (u'v + uv')\,dx = [\,u(x)v(x)\,]_a^b$. Subtrahieren wir auf beiden Seiten $\int_a^b u'v\,dx$, erhalten wir :eqref:`eq_mdl-parts`.

<!--en:7e7f1d5d-->
Die Formel überträgt die Ableitung von einem Faktor auf den anderen. Dabei kommt ein Randterm hinzu. Das lohnt sich, wenn der neue Integrand $u'v$ einfacher ist als $uv'$. Wählen wir $u(x)=x$ und $v'(x)=e^{-x}$, so gilt $u'=1$ und $v=-e^{-x}$. Wenden wir :eqref:`eq_mdl-parts` auf $[0,b]$ an und bilden den Grenzwert für $b\to\infty$, erhalten wir

<!--en:818d9f3f-->
Der Randterm verschwindet, weil $x e^{-x}\to 0$ an beiden Grenzen gilt. Übrig bleibt das gerade berechnete uneigentliche Integral. Das Integral auf der linken Seite ist der Erwartungswert der Exponentialverteilung (:numref:`sec_mdl-distributions`).

<!--en:7d820ae9-->
Da Erwartungswerte Integrale über Produkte sind, verwenden wir in der Wahrscheinlichkeitsrechnung und im maschinellen Lernen häufig denselben Trick: Wir übertragen die Ableitung von einem schwer zu handhabenden Faktor auf einen einfacheren. Besonders deutlich wird das an der Glockenkurve. Betrachten wir die unnormierte Gewichtsfunktion $w(z)=e^{-z^2/2}$ mit $w'(z)=-z\,w(z)$. Sei $g$ stetig differenzierbar mit $g(z)w(z)\to0$ für $z\to\pm\infty$. Außerdem mögen die beiden folgenden Integrale konvergieren. Wir integrieren auf $[-a,b]$ partiell und bilden anschließend getrennt die Grenzwerte $a\to\infty$ und $b\to\infty$. So erhalten wir

<!--en:49d13c7a-->
Unter dem Integral mit diesem Glockengewicht hat die Multiplikation mit $z$ denselben Effekt wie die Differentiation von $g$. Beide Voraussetzungen gelten für jedes Polynom $g$. Setzen wir $g(z)=z$, erhalten wir $\int z^2 w\,dz=\int w\,dz$. Der gewichtete Mittelwert von $z^2$ ist also $1$, unabhängig vom Wert der Normierungskonstante $\int w\,dz$ (sie beträgt $\sqrt{2\pi}$, wie wir in :eqref:`eq_mdl-normal-normalizer` zeigen). Mit $g(z)=z^3$ ergibt sich $\int z^4 w\,dz=3\int z^2 w\,dz=3\int w\,dz$. Der gewichtete Mittelwert von $z^4$ ist also $3$. Normieren wir $w$ zur Dichte der Standardnormalverteilung $p=w/\sqrt{2\pi}$, erhalten wir $\mathbb{E}[g'(Z)]=\mathbb{E}[Z\,g(Z)]$. Das ist die Stein-Identität für die Standardnormalverteilung. Auch beim Score-Matching besteht der entscheidende Schritt darin, eine Ableitung von einem unbekannten auf einen bekannten Faktor zu übertragen (:numref:`sec_mdl-score-matching-diffusion-flow`, :cite:`Hyvarinen.2005`). In $\mathbb{R}^d$ überträgt die entsprechende Regel eine Divergenz anstelle einer eindimensionalen Ableitung. In :numref:`sec_mdl-dyn-vector-calculus` führen wir die Divergenz, den Laplace-Operator und die mehrdimensionale Identität ein, sobald wir sie brauchen.

<!--en:de0b02a1-->
### Substitution

<!--en:abb50eeb-->
Lesen wir die Kettenregel mithilfe der Auswertungsformel rückwärts, erhalten wir die Substitutionsregel. Sei $u$ stetig differenzierbar auf $[a,b]$. Sei $f$ stetig auf einem Intervall, das $u([a,b])$ enthält, und habe dort die Stammfunktion $G$. Nach der Kettenregel gilt $(G\circ u)'(x)=f(u(x))\,u'(x)$, also

<!--en:b05fe747-->
Von rechts nach links gelesen ist das die **Formel für die Variablentransformation**

<!--en:ae4e6292-->
Wir können also ein Integral über $y$ als Integral über $x$ schreiben, wenn wir den Integranden mit dem lokalen Streckungsfaktor $du/dx$ multiplizieren. Dabei muss $u$ nicht einmal monoton sein.

<!--en:3389df6d-->
An einem einzigen schmalen Streifen lässt sich der Streckungsfaktor ablesen. Zerlegen wir $[a,b]$ in Streifen der Breite $\epsilon$. Die Abbildung $u$ bildet den Streifen bei $x$ auf einen Streifen der ungefähren orientierten Breite $\epsilon\,u'(x)$ bei $y=u(x)$ ab. Dort hat die Kurve die Höhe $f(y)=f(u(x))$. Der Streifen trägt also die Fläche $f(u(x))\,u'(x)\,\epsilon$ bei. Genau diese Beiträge summiert das Integral über $x$ (:numref:`fig_mdl-rect-transform`).

<!--en:6d9d886b-->
![Ein schmaler Streifen bei der Substitution $y=u(x)$. Links die $x$-Seite (rechte Formelseite) mit einem Streifen der Breite $\epsilon$ bei $x$ und der Höhe $f(u(x))$. Rechts die $y$-Seite mit dem Bildstreifen der Breite $\epsilon\,u'(x)$ und derselben Höhe. Die Gleichheit der Flächen erfordert den Faktor $du/dx$.](../img/mdl-cal-rect-trans.svg)

<!--en:f8f236fc-->
In der Praxis suchen wir im Integranden nach dem Muster $f(u(x))\,u'(x)$. Setzen wir in $\int_0^1 x\,e^{-x^2}\,dx$ die Funktion $u(x)=x^2$ ein, so gilt $u'(x)=2x$. Der Integrand lautet also $\tfrac12 e^{-u(x)}u'(x)$, und wir erhalten

<!--en:a7c9b2bc-->
Hier liefert der Faktor $x$ die für die Substitution benötigte Ableitung $u'(x)$ bis auf einen konstanten Faktor. Entfernen wir $x$, bleibt $\int e^{-x^2}\,dx$. Dieses Integral hat keine elementare Stammfunktion. Gerade auf ihm beruht die Normierungskonstante $1/\sqrt{2\pi}$ der Glockenkurve. Für die Berechnung in :eqref:`eq_mdl-gaussian-integral` gewinnen wir den fehlenden Faktor durch den Übergang zu zwei Dimensionen und den Wechsel zu Polarkoordinaten.

<!--en:ba1369f1-->
## Mehrfachintegrale

<!--en:8734f6e5-->
Im Folgenden verwenden wir zwei klassische Sätze ohne Beweis. Beide finden sich in :cite:`Folland.1999`: der Satz von Fubini und die Formel für die Variablentransformation in mehreren Dimensionen.

<!--en:09f7dd6b-->
### Doppelintegrale und der Satz von Fubini

<!--en:ee8a7b57-->
Unser Weg zu $\int e^{-x^2}\,dx$ führt über zwei Dimensionen. Daher erweitern wir die Integration von Intervallen auf Gebiete in der Ebene. Betrachten wir die zweidimensionale Glockenfläche $f(x,y)=e^{-x^2-y^2}$ über dem Quadrat $[-2,2]^2$. Sie schließt mit der Grundebene ein Volumen ein. Wir nähern dieses Volumen genauso an wie zuvor die Flächeninhalte. Dazu zerlegen wir die Grundfläche in $\epsilon\times\epsilon$ große Quadrate, errichten auf jedem Quadrat eine Säule der Höhe $f$ und addieren die Volumina der Säulen (:numref:`fig_mdl-bell-surface`). Für eine Funktion $f$ auf einem Rechteck $R=[a,b]\times[c,d]$ ist der Grenzwert dieser Summen für $\epsilon\to0$ das **Doppelintegral** $\iint_R f(x,y)\,dx\,dy$. Es gibt das orientierte Volumen unter der Fläche an.

<!--en:499ce3ba-->
![Die Glockenfläche $z=e^{-x^2-y^2}$ über dem Quadrat $[-2,2]^2$. Die Grundfläche zerfällt in $\epsilon\times\epsilon$ große Quadrate ($\epsilon=0.5$). Auf jedem steht ein Quader der Höhe $f$. Ihre Volumina nähern zusammen das Volumen unter der Fläche an. Die hervorgehobene Spalte aus Quadern bei festem $x$ entspricht einer inneren Summe im Satz von Fubini.](../img/mdl-cal-bell-surface.svg)

<!--en:6d0f838b-->
Wie können wir ein solches Volumen mit unseren eindimensionalen Verfahren berechnen? Bei einer endlichen Summe dürfen wir die Summanden in beliebiger Reihenfolge addieren. Nummerieren wir die Gitterquadrate mit $i$ in $x$-Richtung und mit $j$ in $y$-Richtung, können wir die Quadervolumina spaltenweise summieren (:numref:`fig_mdl-sum-order`):

<!--en:3a56b328-->
Die innere Summe nähert mit der linken Rechteckregel das Integral $\int_c^d f(a+\epsilon i,y)\,dy$ an, also die Querschnittsfläche bei $x=a+\epsilon i$ (hervorgehobene Spalte in :numref:`fig_mdl-bell-surface`). Die äußere Summe integriert diese Querschnittsfläche über $x$.

<!--en:e74d833f-->
![Summation auf einem Gitter aus $\epsilon\times\epsilon$ großen Zellen mit Gitterecke $(a,c)$. Die Indizes laufen mit $i$ in $x$-Richtung und mit $j$ in $y$-Richtung. (1) Bei festem $i$ über $j$ summieren. (2) Die Spaltensummen von links nach rechts addieren. Eine endliche Summe bleibt beim Umordnen gleich. Auf dieser diskreten Eigenschaft beruht die iterierte Integration.](../img/mdl-cal-sum-order.svg)

<!--en:30e9bdcd-->
Eine endliche Gittersumme dürfen wir ohne Weiteres umordnen. Damit die Gleichheit beim Grenzübergang $\epsilon\to0$ erhalten bleibt, brauchen wir einen Satz mit einer zusätzlichen Voraussetzung. Der **Satz von Fubini** besagt: Ist $f$ auf $R$ **absolut integrierbar**, gilt also $\iint_R|f(x,y)|\,dx\,dy<\infty$, dann ist das Doppelintegral gleich jedem der beiden iterierten Integrale:

<!--en:efd8535c-->
Geometrisch bedeutet das, dass wir ein Volumen in beiden Richtungen schichtweise berechnen können. Für $f\ge0$ gilt dieselbe Gleichheit ohne weitere Prüfung der Integrierbarkeit. Alle drei Ausdrücke dürfen dann gleich $+\infty$ sein (*Satz von Tonelli*). Bei Wahrscheinlichkeitsdichten, quadratischen Verlusten und anderen nichtnegativen Integranden entfällt die Prüfung daher. Kann der Integrand beide Vorzeichen annehmen, müssen wir zuerst $\iint|f|<\infty$ prüfen. Wir verwenden den Satz von Fubini jedes Mal, wenn wir eine Variable aus einer gemeinsamen Dichte herausintegrieren oder einen Erwartungswert mit einem anderen Integral vertauschen.

<!--en:aaaf16c2-->
Der häufigste Spezialfall ist ein Integrand der Form $f(x,y)=g(x)h(y)$. Da $g(x)$ bezüglich $y$ konstant ist, können wir diesen Faktor aus dem inneren Integral herausziehen: $\int_c^d g(x)h(y)\,dy=g(x)\int_c^d h(y)\,dy$. Es bleibt

<!--en:94765d32-->
Einen solchen Integranden können wir Faktor für Faktor integrieren. Entscheidend ist, dass auch die zweidimensionale Glockenfunktion so zerfällt: $e^{-x^2-y^2}=e^{-x^2}e^{-y^2}$. Darauf beruht unsere spätere Berechnung des Gauß-Integrals.

<!--en:662d6769-->
Die Voraussetzung der absoluten Integrierbarkeit ist wesentlich. Betrachten wir $f(x,y)=(x^2-y^2)/(x^2+y^2)^2$ auf $[0,1]^2$. Durch Ableiten erhalten wir $\partial_y\bigl(y/(x^2+y^2)\bigr)=f(x,y)$. Die Auswertungsformel liefert daher $\int_0^1 f(x,y)\,dy=\frac1{1+x^2}$ für $x>0$. Integration über $x$ ergibt $\int_0^1\frac{dx}{1+x^2}=\arctan 1=\frac\pi4$. Nun gilt aber $f(x,y)=-f(y,x)$, sodass die umgekehrte Integrationsreihenfolge $-\frac\pi4$ ergibt. Die Ergebnisse unterscheiden sich, weil $\iint|f|=\infty$. In der Nähe des Ursprungs wächst $|f|$ in nichtverschwindenden Winkelbereichen wie der Kehrwert des Abstandsquadrats. Das Gesamtvolumen divergiert.

<!--en:2242680f-->
Ist der Integrationsbereich $D$ kein Rechteck, definieren wir $\int_D f(\mathbf x)\,d\mathbf x$ durch *Fortsetzung mit null*. Dazu setzen wir $f$ auf $0$ außerhalb von $D$ und integrieren über ein beliebiges Rechteck, das $D$ enthält.

<!--en:58d9221d-->
### Variablentransformation in mehreren Dimensionen

<!--en:5ead0ade-->
Der eindimensionale Streckungsfaktor $du/dx$ lässt sich auf höhere Dimensionen verallgemeinern. Schon an linearen Abbildungen erkennen wir seine Form. Eine lineare Abbildung skaliert jeden Flächeninhalt und jedes Volumen mit dem Betrag ihrer Determinante (:numref:`sec_mdl-linear-maps`). Das Vorzeichen der Determinante gibt an, ob die Orientierung erhalten bleibt. Beispielsweise hat die Abbildung $\boldsymbol\phi(u,v)=(au,bv)$ mit $a,b>0$ die Determinante $ab$. Sie streckt die Einheitskreisscheibe (Flächeninhalt $\pi$) zur Ellipsenfläche $x^2/a^2+y^2/b^2\le1$ mit Flächeninhalt $\pi ab$.

<!--en:5cd0035a-->
Eine glatte nichtlineare Abbildung verhält sich lokal wie eine lineare Abbildung. Die lineare Näherung liefert ihre Jacobi-Matrix. Der Volumenfaktor kann sich von Punkt zu Punkt ändern. Sei $\Omega\subseteq\mathbb{R}^n$ offen und sei $\boldsymbol{\phi}:\Omega\to\mathbb{R}^n$ ein **$C^1$-Diffeomorphismus** auf sein Bild. Das heißt, $\boldsymbol{\phi}$ ist stetig differenzierbar und injektiv. Für ihre Jacobi-Matrix $\mathbf J_{\boldsymbol\phi}$ (die Matrix der partiellen Ableitungen aus :numref:`sec_mdl-matrix-calculus`) gilt $\det\mathbf J_{\boldsymbol\phi}(\mathbf{x})\neq 0$ überall auf $\Omega$. Die Injektivität verhindert, dass die Abbildung den Raum auf sich selbst faltet. Die von null verschiedene Determinante verhindert, dass sie einen Bereich auf eine niedrigere Dimension abflacht. Unter diesen Voraussetzungen gilt für jede absolut integrierbare Funktion $f$

<!--en:584659ed-->
Ein infinitesimaler Würfel mit Volumen $d\mathbf x$ am Punkt $\mathbf x$ wird auf ein infinitesimales Parallelotop mit Volumen $|\det\mathbf J_{\boldsymbol\phi}(\mathbf x)|\,d\mathbf x$ am Punkt $\mathbf y=\boldsymbol\phi(\mathbf x)$ abgebildet. Die Jacobi-Determinante berücksichtigt im Integranden diese lokale Volumenänderung. :numref:`fig_mdl-cov-jacobian` zeigt den ein- und den zweidimensionalen Fall nebeneinander.

<!--en:6c4f198b-->
![Links skaliert die eindimensionale Substitution ein orientiertes Streckenstück der Länge $\epsilon$ mit $u'(x)$. Rechts bildet in zwei Dimensionen eine lineare Abbildung $\boldsymbol\phi$ das Einheitsquadrat auf ein Parallelogramm ab. Dessen Fläche entspricht dem Volumenfaktor $|\det\mathbf J_{\boldsymbol\phi}|$.](../img/mdl-cal-cov-jacobian.svg)

<!--en:df3cacee-->
Warum steht in :eqref:`eq_mdl-change_var_nd` der Betrag $|\det\mathbf J_{\boldsymbol\phi}|$ statt $\det\mathbf J_{\boldsymbol\phi}$? Auf der linken Seite integrieren wir über eine Menge ohne Orientierung. In einer Dimension hingegen bestimmt die Reihenfolge der Grenzen das Vorzeichen. Ist $u$ fallend, stehen die Grenzen in :eqref:`eq_mdl-change_var` in umgekehrter Reihenfolge und es gilt $u'<0$. Die beiden Minuszeichen heben sich auf. Gleichwertig können wir also über das Bildintervall mit dem Faktor $|u'|$ integrieren. Gebiete in $\mathbb R^n$ besitzen keine natürliche Anordnung. Daher lässt sich nur diese Form ohne Vorzeichen verallgemeinern.

<!--en:040fad8b-->
Ist $f$ eine Wahrscheinlichkeitsdichte, beschreibt :eqref:`eq_mdl-change_var_nd`, wie sich eine Dichte unter einer glatten umkehrbaren Abbildung transformiert. Auf diese Weise berechnet ein normalisierender Fluss die Dichte seiner Ausgabe (:numref:`sec_mdl-joint-distributions`).

<!--en:b84cc57e-->
### Das Gauß-Integral

<!--en:5186e4a4-->
Die Normierungskonstante $1/\sqrt{2\pi}$ der Glockenkurve ergibt sich aus dem eindimensionalen Integral $\int_{-\infty}^\infty e^{-x^2}\,dx$. Hier liefert der Hauptsatz keine geschlossene Formel. Eine Stammfunktion existiert zwar, weil $e^{-x^2}$ stetig ist. Sie ist jedoch eine grundlegend neue Funktion und lässt sich nicht durch elementare Funktionen ausdrücken. Überraschenderweise können wir das Integral über die ganze reelle Achse dennoch in geschlossener Form berechnen, wenn wir den Umweg über zwei Dimensionen nehmen. Der Beweis besteht aus drei Schritten und ist die eleganteste Rechnung dieses Abschnitts. (Die normierte Stammfunktion mit Wert null im Ursprung ist die **Fehlerfunktion** $\operatorname{erf}(x)=\frac2{\sqrt\pi}\int_0^x
e^{-t^2}\,dt$. Nach dem Ergebnis, das wir gleich beweisen, strebt sie gegen $1$ für $x\to\infty$. Standardbibliotheken stellen sie als `math.erf` bereit.)

<!--en:33d9b2c8-->
**I. Quadrieren und faktorisieren.** Auf dem Quadrat $[-b,b]^2$ liefert die Produktformel :eqref:`eq_mdl-fubini-product` die Gleichung $\iint_{[-b,b]^2}e^{-x^2-y^2}\,dx\,dy=\bigl(\int_{-b}^b e^{-x^2}\,dx\bigr)^2$. Bilden wir auf beiden Seiten den Grenzwert für $b\to\infty$, erhalten wir

<!--en:bc724c0e-->
Das Quadrat unseres eindimensionalen Integrals ist also das Volumen unter der zweidimensionalen Glockenfläche. Diese ist rotationssymmetrisch um die senkrechte Achse durch den Ursprung.

<!--en:fcdfe0e2-->
**II. Polarkoordinaten.** Wegen der Rotationssymmetrie bieten sich Polarkoordinaten an: $\boldsymbol{\phi}(r,\theta)=(r\cos\theta, r\sin\theta)$ auf der offenen Menge $\Omega=(0,\infty)\times(0,2\pi)$. Dort ist die Abbildung injektiv. Ihre Jacobi-Determinante ist

<!--en:e5f62053-->
Eine Polarzelle mit radialer Seitenlänge $dr$ und Bogenlänge $r\,d\theta$ hat näherungsweise den Flächeninhalt $r\,dr\,d\theta$ (:numref:`fig_mdl-polar-cell`). Je weiter die Zelle vom Ursprung entfernt ist, desto breiter ist sie. Entscheidend ist, dass dieser Faktor $r$ die Rolle des fehlenden Faktors $x$ aus unserem Substitutionsbeispiel $\int_0^1 x\,e^{-x^2}\,dx$ übernimmt.

<!--en:6f513685-->
![Polargitter mit einer schattierten Zelle. Aus der radialen Seitenlänge $dr$ und der Bogenlänge $r\,d\theta$ ergibt sich näherungsweise die Fläche $r\,dr\,d\theta$. Beim Übergang zu Polarkoordinaten erhält der Integrand den Faktor $|\det\mathbf J_{\boldsymbol\phi}|=r$.](../img/mdl-cal-polar-cell.svg)

<!--en:22ad8e1d-->
**III. Integrieren.** Wir wenden :eqref:`eq_mdl-change_var_nd` an und verwenden die Stammfunktion $-\tfrac12 e^{-r^2}$ von $r e^{-r^2}$. Damit erhalten wir

<!--en:5952535a-->
Ziehen wir in :eqref:`eq_mdl-gauss-square` die positive Quadratwurzel, erhalten wir

<!--en:d9296166-->
Die gesamte Fläche unter $e^{-x^2}$ beträgt also $\sqrt\pi\approx 1.7725$.

<!--en:678379d4-->
Drei Einzelheiten machen den Beweis vollständig. Erstens ist der Integrand nichtnegativ. Daher konvergieren die Integrale über wachsende Quadrate und über wachsende Kreisscheiben gegen denselben Grenzwert. Jede Kreisscheibe passt in ein hinreichend großes Quadrat und jedes Quadrat in eine hinreichend große Kreisscheibe. Die Form des wachsenden Gebiets spielt also keine Rolle. Zweitens lässt die Polarkoordinatenabbildung den Strahl $\theta=0$ einschließlich des Ursprungs aus. Er hat Flächeninhalt null und trägt nichts zum Integral bei. Drittens wenden wir die Variablentransformation auf die beschränkten Rechtecke $(0,b)\times(0,2\pi)$ an. Dort ist der Integrand beschränkt und stetig. Erst danach bilden wir den Grenzwert für $b\to\infty$.

<!--en:f5bd54cd-->
Die Normierungskonstante der Standardnormalverteilung folgt nun aus einer einzigen Substitution. Setzen wir $z=\sqrt2\,x$ in :eqref:`eq_mdl-change_var` ein, erhalten wir

<!--en:5177a623-->
Teilen wir die Glockenfunktion $e^{-z^2/2}$ durch $\sqrt{2\pi}\approx 2.5066$, so wird die Gesamtfläche unter der Kurve auf $1$ skaliert. Daher stammt die Konstante in der Dichte der Standardnormalverteilung.

<!--en:f6e8e65f-->
Prüfen wir Schritt I numerisch: Schöpfen die wachsenden Quadrate $[-b,b]^2$ die Ebene aus und konvergieren die Integrale dabei tatsächlich gegen $\pi$? Der folgende Code berechnet die zweidimensionale Gittersumme über $[-b,b]^2$ bei $\epsilon=0.01$. Zum Vergleich berechnet er das exakte Volumen über dem Quadrat $\bigl(\int_{-b}^b e^{-x^2}\,dx\bigr)^2=\pi\operatorname{erf}(b)^2$ und gibt den Grenzwert $\pi$ an.

<!--en:a54b9382-->
Die Gittersummen liegen nahe bei $\pi\operatorname{erf}(b)^2$ und steigen auf $\pi$ an. Bei $b=3$ beträgt das Volumen außerhalb des Quadrats nur noch etwa $10^{-4}$. Die Übereinstimmung auf vier oder mehr Nachkommastellen bei $\epsilon=0.01$ lässt sich leicht erklären. Für einen geraden Integranden auf einem symmetrischen Intervall verschwindet der systematische Fehler erster Ordnung der linken Rechteckregel $\tfrac\epsilon2\bigl(f(-b)-f(b)\bigr)$ identisch.

<!--en:62dbbb81-->
Nun können wir auch die Wahrscheinlichkeit $P(-1\le Z\le1)\approx0.68$ für die Standardnormalverteilung berechnen. Mit der Substitution $z=\sqrt2\,t$ lässt sich die Fläche unter $p(z)=e^{-z^2/2}/\sqrt{2\pi}$ durch die Fehlerfunktion ausdrücken:

<!--en:90ddfaba-->
Damit sind beide Fragen zur Glockenkurve beantwortet. Auch der scheinbare Widerspruch zwischen $p(0)\approx 0.40$ und $P(Z=0)=0$ löst sich auf. Eine Dichte misst Wahrscheinlichkeit pro Längeneinheit, und ein einzelner Punkt ist ein Intervall der Länge null.

<!--en:2db71535-->
Eine Frage vom Anfang bleibt offen. Um ein Modell durch zufälliges Sampling zu trainieren, müssen wir wissen, wie sich eine solche Fläche ändert, wenn sich ein Parameter der Dichte ändert. Dazu müssen wir ein Integral ableiten.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:c5d3a035-->
Das bestimmte Integral ist der Grenzwert der Flächensummen schmaler Streifen und misst den orientierten Flächeninhalt unter einer Kurve. Der Hauptsatz der Differential- und Integralrechnung ersetzt diesen Grenzübergang durch das Bestimmen einer Stammfunktion. Da für die Integralfunktion $F'=f$ gilt, liefert jede Stammfunktion $G$ die Formel $\int_a^b f\,dx=G(b)-G(a)$. Partielle Integration und Substitution ergeben sich aus der rückwärts gelesenen Produktregel und Kettenregel. In mehreren Dimensionen führt der Satz von Fubini Volumenintegrale auf iterierte eindimensionale Integrale zurück. Bei der Variablentransformation tritt als Streckungsfaktor $|\det\mathbf J_{\boldsymbol\phi}|$ auf. Er beschreibt die lokale Änderung des Volumens. Beim Übergang zu Polarkoordinaten erhalten wir damit $\int_{-\infty}^\infty e^{-x^2}\,dx=\sqrt\pi$ und die Normierungskonstante $\sqrt{2\pi}$ der Standardnormalverteilung.

<!--en:2294243a-->
## Übungen

<!--en:70d37b4c-->
1. **Aktivierungen als aufsummierte Flächen.** Sei $H$ die Einheitssprungfunktion mit $H(t)=1$ für $t>0$ und $H(t)=0$ für $t\le 0$. Sei $\phi(t)=e^{-t^2/2}/\sqrt{2\pi}$ die Dichte der Standardnormalverteilung und $\Phi(x)=\int_{-\infty}^x \phi(t)\,dt$ ihre Verteilungsfunktion.
    1. Zeigen Sie, dass $\int_{-\infty}^{x} H(t)\,dt=\mathrm{ReLU}(x)$ gilt. Welche Voraussetzung des Hauptsatzes :eqref:`eq_mdl-ftc` ist bei $x=0$ verletzt? Welche einseitigen Ableitungen hat das Integral dort?
    1. Schreiben Sie $\Phi(x)=\Phi(0)+\int_0^x \phi(t)\,dt$ und folgern Sie daraus $\Phi'=\phi$. Zeigen Sie anschließend durch die Substitution $t=\sqrt2\,s$, dass $\Phi(x)=\tfrac12\bigl(1+\operatorname{erf}(x/\sqrt2)\bigr)$ gilt.
    1. Die GELU-Aktivierung aus :numref:`sec_mlp` lautet $x\Phi(x)$. Leiten Sie die Formel für ihre Ableitung her. Bestimmen Sie die Stellen, an denen die Ableitung ihren größten und ihren kleinsten Wert annimmt, und berechnen Sie diese Werte mit `math.erf`. In welchem Intervall liegt die Steigung von GELU? Wie verhält sich dieses Intervall zu den Steigungen $0$ und $1$ von ReLU?
   :exlabel:`ex_mdl-integral-calculus-activation-areas`

<!--en:06b2d838-->
1. **Schwere Verteilungsränder.** Die *Pareto-Dichte* mit Randindex $\alpha>0$ lautet $p(x)=\alpha x^{-\alpha-1}$ für $x\ge 1$ und sonst $p(x)=0$. Die *Cauchy-Dichte* lautet $p(x)=\tfrac{1}{\pi(1+x^2)}$ auf $\mathbb R$.
    1. Zeigen Sie, dass die Pareto-Dichte für jedes $\alpha>0$ normiert ist. Für welche $\alpha$ sind $\mathbb{E}[X]$ und $\mathbb{E}[X^2]$ endlich? Berechnen Sie $\mathbb{E}[X]$ für die Fälle, in denen dieser Erwartungswert endlich ist.
    1. Zeigen Sie mithilfe der Stammfunktion $\arctan x$ von $\tfrac{1}{1+x^2}$, dass die Cauchy-Dichte normiert ist.
    1. Zeigen Sie, dass $\int_{-b}^{b} x\,p(x)\,dx=0$ für die Cauchy-Dichte und jedes $b>0$ gilt. Warum besitzt die Cauchy-Verteilung dennoch keinen Erwartungswert, wenn wir die Definition von $\int_{-\infty}^{\infty}$ im Anschluss an :eqref:`eq_mdl-improper` verwenden? Wie schnell wächst $\int_0^b x\,p(x)\,dx$ mit $b$?
   :exlabel:`ex_mdl-integral-calculus-heavy-tails`

<!--en:2abfc861-->
1. **Die Gammafunktion durch partielle Integration.** Die *Gammafunktion* ist durch $\Gamma(t)=\int_0^\infty u^{t-1}e^{-u}\,du$ gegeben. Für $t<1$ ist der Integrand bei $u=0$ unbeschränkt. Wie in der Nähe von $0$ zu integrieren ist, legt der Grenzwert $\int_0^1=\lim_{a\to 0^+}\int_a^1$ fest (siehe :numref:`subsec_mdl-improper-integrals`).
    1. Für welche $t$ konvergiert das Integral bei $u=0$? Vergleichen Sie die Bedingung mit dem Kriterium für $\int_0^1x^{-s}\,dx$ in :numref:`subsec_mdl-improper-integrals`.
    1. Zeigen Sie durch partielle Integration nach :eqref:`eq_mdl-parts` für beliebiges $t>0$, dass $\Gamma(t+1)=t\,\Gamma(t)$ gilt. Folgern Sie daraus $\Gamma(n+1)=n!$ für jede ganze Zahl $n\ge 0$. Wo geht die Voraussetzung $t>0$ in den Beweis ein? Berechnen Sie mit dem Ergebnis die Varianz der Exponentialverteilung mit Dichte $e^{-u}$ auf $u\ge 0$. Ihren Erwartungswert haben wir in :numref:`subsec_mdl-integration-by-parts` berechnet.
    1. Substituieren Sie $u=x^2$ und berechnen Sie mithilfe des Gauß-Integrals :eqref:`eq_mdl-gaussian-integral` den Wert von $\Gamma(\tfrac12)$.
    1. Die Produktformel :eqref:`eq_mdl-fubini-product` liefert bei $d$-maliger Anwendung $\int_{\mathbb R^d}e^{-\|\mathbf x\|^2}\,d\mathbf x=\pi^{d/2}$. In Polarkoordinaten lautet dasselbe Integral $S_{d-1}\int_0^\infty r^{d-1}e^{-r^2}\,dr$. Dabei ist $S_{d-1}$ der Oberflächeninhalt der Einheitssphäre in $\mathbb R^d$. Zeigen Sie, dass das Radialintegral gleich $\tfrac12\Gamma(d/2)$ ist, und folgern Sie daraus $S_{d-1}=2\pi^{d/2}/\Gamma(d/2)$. Integrieren Sie $S_{d-1}r^{d-1}$ über $r\in[0,1]$ und zeigen Sie so, dass die Einheitskugel das Volumen $V_d=\pi^{d/2}/\Gamma(d/2+1)$ hat. Berechnen Sie $V_d$ für $d\in\{1,2,3,5,10,20\}$ und erklären Sie, warum $V_d\to0$ gilt. Ein Mittelpunktgitter mit drei Punkten pro Achse im Würfel $[-1,1]^{10}$ hat $3^{10}$ Punkte. Welcher Anteil dieser Punkte liegt in der Einheitskugel? Was sagt das über Gitter in hohen Dimensionen aus?
   :exlabel:`ex_mdl-integral-calculus-gamma-function`

<!--en:f960e2d3-->
1. **Die Integrationsreihenfolge vertauschen.** Nach dem Satz von Fubini :eqref:`eq_mdl-fubini` können wir die Integrationsreihenfolge wählen. Bei nicht rechteckigen Gebieten verwenden wir dazu die Fortsetzung mit null.
    1. Im inneren Integral von $\int_0^1\bigl(\int_y^1 e^{-x^2}\,dx\bigr)\,dy$ hat der Integrand keine elementare Stammfunktion. Skizzieren Sie das dreieckige Integrationsgebiet, kehren Sie die Integrationsreihenfolge um und berechnen Sie das Integral in geschlossener Form.
    1. Sei $X\ge 0$ eine Zufallsvariable mit Dichte $p$. Schreiben Sie $x=\int_0^x dt$ in $\mathbb{E}[X]=\int_0^\infty x\,p(x)\,dx$ und vertauschen Sie die beiden Integrale, um $\mathbb{E}[X]=\int_0^\infty P(X>t)\,dt$ zu zeigen. Warum dürfen Sie die Integrale auch dann vertauschen, wenn $\mathbb{E}[X]=\infty$ gilt? Prüfen Sie die Formel an der Dichte $\lambda e^{-\lambda x}$ für $x\ge 0$.
   :exlabel:`ex_mdl-integral-calculus-order-of-integration`

<!--en:b46b3b27-->
1. **Normierungskonstanten und Momente von Gauß-Verteilungen.** Im Text normieren wir die Standardnormalverteilung und berechnen ihre Momente mit der Stein-Identität :eqref:`eq_mdl-stein-1d` für das Glockengewicht. Sei $p(x)=\tfrac{1}{\sqrt{2\pi\sigma^2}}\exp\bigl(-\tfrac{(x-\mu)^2}{2\sigma^2}\bigr)$ mit $\sigma>0$.
    1. Substituieren Sie $x=\mu+\sigma z$ und zeigen Sie mit :eqref:`eq_mdl-normal-normalizer`, dass $p$ normiert ist. Welche Werte haben $\mu$ und $\sigma^2$ für die Dichte $\tfrac{1}{\sqrt\pi}e^{-x^2}$?
    1. Sei $X$ eine Zufallsvariable mit Dichte $p$ und sei $g$ stetig differenzierbar mit $g(x)\,p(x)\to 0$ für $x\to\pm\infty$. Zeigen Sie mithilfe von $p'(x)=-\tfrac{x-\mu}{\sigma^2}\,p(x)$ und partieller Integration, dass folgende Gleichung gilt:

       $$\mathbb{E}\bigl[(X-\mu)\,g(X)\bigr]=\sigma^2\,\mathbb{E}\bigl[g'(X)\bigr].$$

       Folgern Sie daraus $\mathbb{E}[(X-\mu)^2]=\sigma^2$ und $\mathbb{E}[(X-\mu)^4]=3\sigma^4$.
    1. Sei $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$ eine $d\times d$-Kovarianzmatrix mit invertierbarer Matrix $\mathbf L$. Verwenden Sie die Variablentransformation $\mathbf x=\mathbf L\mathbf z$ in :eqref:`eq_mdl-change_var_nd` und die Produktformel :eqref:`eq_mdl-fubini-product`, um folgende Gleichung zu zeigen:

       $$\int_{\mathbb R^d}\exp\bigl(-\tfrac12\mathbf x^\top\boldsymbol\Sigma^{-1}\mathbf x\bigr)\,d\mathbf x=(2\pi)^{d/2}\,(\det\boldsymbol\Sigma)^{1/2}.$$
   :exlabel:`ex_mdl-integral-calculus-gaussian-normalizers`

<!--en:f9848e28-->
[Diskussion](https://d2l.discourse.group/t/414)

<!--en:ef7207b9-->
[Diskussion](https://d2l.discourse.group/t/1092)

<!--en:575bcd2a-->
[Diskussion](https://d2l.discourse.group/t/1093)

<!--en:575bcd2a-2-->
[Diskussion](https://d2l.discourse.group/t/1093)

<!--en:27621000-->
Integralrechnung<br>**Riemann-Summen · Hauptsatz · uneigentliche Integrale · partielle Integration und Substitution · Satz von Fubini · Jacobi-Determinanten · Gauß-Integral**

<!--en:edca331b-->
::: {.slide title="Wahrscheinlichkeiten stetiger Verteilungen als Integrale"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:48151c5e-->
- **Das Paradoxon der Standardnormalverteilung:** Für $Z \sim \mathcal{N}(0, 1)$ erreicht die Dichte ihr Maximum bei einer Höhe von $p(0) \approx 0.40$. Dennoch beträgt die Wahrscheinlichkeit für den einzelnen Punkt null: $P(Z = 0) = 0$. Erst Intervalle liefern positive Fläche ($P(-1 \le Z \le 1) = \int_{-1}^1 p(z)\,dz \approx 0.683$).

<!--en:75e695cd-->
- **Zwei grundlegende Fragen:** Wie berechnen wir bestimmte Integrale mit Stammfunktionen und warum gilt $\int_{-\infty}^\infty e^{-z^2/2}\,dz = \sqrt{2\pi}$?

<!--en:36944eab-->
::: {.slide title="Riemann-Summen aus schmalen Rechtecken"}

<!--en:7bf231b4-->
[Das bestimmte Integral]{.kicker}

<!--en:8ff882e6-->
- **Linke Rechteckregel für $f(x) = x^2$ auf $[0, 1]$:** Die Zerlegung von $[0, 1]$ in $N$ Streifen der Breite $\epsilon = 1/N$ ergibt die exakte Summenformel
  $$\sum_{i=0}^{N-1} \frac{1}{N}\left(\frac{i}{N}\right)^2 = \frac{1}{3} - \frac{1}{2N} + \frac{1}{6N^2}$$
  mit Grenzwert $\tfrac{1}{3}$ und einem Fehler erster Ordnung von $\approx \tfrac{1}{2N} = \tfrac{\epsilon}{2}$.

<!--en:caa36b8e-->
- **Allgemeines Riemann-Integral:** Eine beschränkte Funktion $f$ ist **Riemann-integrierbar** auf $[a, b]$, wenn jede Riemann-Summe $\sum_{i=1}^N f(\xi_i)(x_i - x_{i-1})$ gegen denselben Grenzwert $\int_a^b f(x)\,dx$ konvergiert, sobald die maximale Streifenbreite gegen null geht ($\|P\| \to 0$).

<!--en:dfb5bd3a-->
::: {.slide title="Diskretisierungsfehler erster Ordnung bei linken Rechtecksummen"}

<!--en:2659a6a2-->
[Konvergenzrate]{.kicker}

<!--en:fcb33cec-->
Bei unserem Beispielintegral $\int_0^2 \frac{x}{1 + x^2}\,dx = \tfrac{1}{2}\log 5 \approx 0.804719$ sinkt der Fehler der linken Rechtecksumme auf ein Zehntel, wenn wir die Streifenbreite $\epsilon$ auf ein Zehntel reduzieren:

<!--en:f234c9c0-->
**Warum der führende Fehlerterm $\tfrac{\epsilon}{2}(f(a) - f(b)) = -0.2\epsilon$ beträgt:** Die Summe der dreieckigen Flächenbeiträge $\tfrac{\epsilon^2}{2}f'(x_i)$ über den linken Rechtecken ergibt in führender Ordnung $\tfrac{\epsilon}{2}\int_a^b f'(x)\,dx = \tfrac{\epsilon}{2}(f(b) - f(a))$.

<!--en:3780589b-->
::: {.slide title="Orientierter Flächeninhalt und Monotonie des Integrals"}

<!--en:0f14c9c2-->
[Eigenschaften des Integrals]{.kicker}

<!--en:ee3352e6-->
- **Orientierter Flächeninhalt über und unter der Achse:** Unterhalb der horizontalen Achse ($f(\xi_i) < 0$) gehen die Rechtecke mit negativer Höhe in die Riemann-Summe ein und verringern die Gesamtfläche.

<!--en:e11b39e4-->
- **Orientierte Grenzen:** Bei Integration von rechts nach links wechseln die Streifenbreiten ihr Vorzeichen. Daher gilt $\int_b^a f(x)\,dx = -\int_a^b f(x)\,dx$.

<!--en:ec2341a2-->
- **Monotonie:** Gilt $f(x) \le g(x)$ überall auf $[a, b]$, ist jede Riemann-Summe von $f$ durch die entsprechende Summe von $g$ nach oben beschränkt. Also gilt $\int_a^b f(x)\,dx \le \int_a^b g(x)\,dx$.

<!--en:c52ad75a-->
::: {.slide title="Additivität über Teilintervalle und Integralfunktion"}

<!--en:a70c85ab-->
[Aufsummierte Fläche]{.kicker}

<!--en:af432810-->
- **Additivität über Teilintervalle:** Jede Zerlegung mit dem Punkt $b \in [a, c]$ zerfällt in Zerlegungen von $[a, b]$ und $[b, c]$. Daher gilt $\int_a^c f\,dx = \int_a^b f\,dx + \int_b^c f\,dx$.

<!--en:d359362d-->
- **Die Integralfunktion $F(x)$:** Fester Anfangspunkt $a$ und variable obere Grenze $x$:
  $$F(x) = \int_a^x f(y)\,dy$$

<!--en:64049371-->
- **Jedes Integral ist eine Differenz von Funktionswerten von $F$:** Wegen der Additivität beträgt die Fläche über $[x_1, x_2]$ genau $\int_{x_1}^{x_2} f(x)\,dx = F(x_2) - F(x_1)$.

<!--en:43974c66-->
::: {.slide title="Der Hauptsatz der Differential- und Integralrechnung"}

<!--en:9cf82324-->
[Änderungsrate der Fläche]{.kicker}

<!--en:c5b466c7-->
- **Hauptsatz (Teil 1):** Sei $f$ auf $[a, b]$ Riemann-integrierbar und sei $F(x) = \int_a^x f(y)\,dy$. An jeder inneren Stelle $x \in (a, b)$, an der $f$ stetig ist, ist $F$ differenzierbar und es gilt
  $$\frac{dF}{dx}(x) = f(x)$$

<!--en:cce2750c-->
- **Beweis durch Einschließen mit Monotonie und Additivität:** Bei der Verschiebung $x \mapsto x+\epsilon$ ($\epsilon > 0$) kommt ein schmaler Streifen $F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y)\,dy$ hinzu. Mit $m_\epsilon = \inf_{[x, x+\epsilon]} f$ und $M_\epsilon = \sup_{[x, x+\epsilon]} f$ liegt sein Flächeninhalt wegen der Monotonie zwischen zwei Rechteckflächen:
  $$m_\epsilon \le \frac{F(x+\epsilon) - F(x)}{\epsilon} \le M_\epsilon$$
  Wegen der Stetigkeit von $f$ bei $x$ gilt für beide Schranken $m_\epsilon, M_\epsilon \to f(x)$ für $\epsilon \to 0$.

<!--en:26aad0cf-->
::: {.slide title="Auswertungsformel und Fallstricke bei Stammfunktionen"}

<!--en:08418c91-->
[Berechnung von Integralen]{.kicker}

<!--en:a4e99498-->
- **Auswertungsformel (Hauptsatz, Teil 2):** Ist $f$ auf $[a, b]$ stetig und gilt $G'(x) = f(x)$ für **alle** $x \in [a, b]$, so folgt $\int_a^b f(x)\,dx = G(b) - G(a)$.

<!--en:da638df0-->
- **Unser durchgehendes Beispiel:** $\int_0^2 \frac{x}{1+x^2}\,dx = [\tfrac{1}{2}\log(1+x^2)]_0^2 = \tfrac{1}{2}\log 5$.

<!--en:8afaec73-->
- **Fallstrick bei einer singulären Stammfunktion:** Die Rechnung $\int_{-1}^1 x^{-2}\,dx = [-1/x]_{-1}^1 = -2$ ist falsch, weil $-1/x$ bei $x = 0 \in [-1, 1]$ unbeschränkt ist.

<!--en:fbd21dda-->
::: {.slide title="Prüfung der Integralfunktion auf [0, 2]"}

<!--en:2997b726-->
[Numerische Kontrolle]{.kicker}

<!--en:8f470d3b-->
Der Vergleich der kumulierten linken Rechtecksumme für $F(x) = \int_0^x \frac{y}{1+y^2}\,dy$ mit $G(x) - G(0) = \tfrac{1}{2}\log(1+x^2)$ auf dem gesamten Gitter über $[0, 2]$ bestätigt die Vorhersage. Die größte Abweichung liegt bei $x = 1$ (Maximum $f(1) = \tfrac{1}{2}$) und beträgt näherungsweise $\tfrac{\epsilon}{2}f(1) = \tfrac{\epsilon}{4}$:

<!--en:df5925d0-->
::: {.slide title="Uneigentliche Integrale und Potenzgesetze mit schweren Verteilungsrändern"}

<!--en:7306033f-->
[Unbeschränkte Bereiche und Singularitäten]{.kicker}

<!--en:5b390940-->
- **Konvergenzgrenzen für Potenzgesetze:**
  - Am unendlichen Rand gilt $\int_1^\infty x^{-s}\,dx = \frac{1}{s-1}$ genau für $s > 1$.
  - Am singulären Randpunkt gilt $\int_0^1 x^{-s}\,dx = \frac{1}{1-s}$ genau für $s < 1$.

<!--en:103ddd8a-->
- **Momente von Verteilungen mit schweren Rändern ($p(x) \propto x^{-s}$):** normierbar für $s > 1$, endlicher Erwartungswert für $s > 2$ und endliche Varianz für $s > 3$.

<!--en:b3cacb68-->
::: {.slide title="Abbruchfehler und Diskretisierungsfehler"}

<!--en:524a4b8f-->
[Fallstrick bei der numerischen Integration]{.kicker}

<!--en:554f5dda-->
Die Näherung für das uneigentliche Integral $\int_0^\infty e^{-x}\,dx = 1$ auf einem endlichen Gitter über $[0, b]$ mit Streifenbreite $\epsilon = 10^{-3}$ verursacht zwei unabhängige Fehler: den **Abbruchfehler** $e^{-b}$ durch das Abschneiden des Bereichs und den **Diskretisierungsfehler** von näherungsweise $\tfrac{\epsilon}{2} = 0.0005$ durch die endliche Streifenbreite.

<!--en:ef5683e2-->
Steigt die obere Grenze $b$ von $1$ auf $20$, sinkt der Abbruchfehler $e^{-b}$ unter $10^{-8}$. Die linke Rechtecksumme bleibt aber bei etwa $1.000500$. Auch für $b \to \infty$ bleibt der systematische Diskretisierungsfehler von näherungsweise $\tfrac{\epsilon}{2}(f(0) - 0) = 0.0005$ bestehen.

<!--en:47fefa9b-->
::: {.slide title="Partielle Integration und Stein-Identität"}

<!--en:56883621-->
[Die Produktregel rückwärts gelesen]{.kicker}

<!--en:34809084-->
- **Partielle Integration:** $\int_a^b u(x)v'(x)\,dx = [u(x)v(x)]_a^b - \int_a^b u'(x)v(x)\,dx$.

<!--en:35b43f13-->
- **Stein-Identität für $w(z) = e^{-z^2/2}$ ($w' = -zw$):** Partielle Integration von $g'(z)w(z)$ ergibt $\mathbb{E}[Z\,g(Z)] = \mathbb{E}[g'(Z)]$ für $Z \sim \mathcal{N}(0, 1)$. Daraus folgen $\mathbb{E}[Z^2] = 1$ und $\mathbb{E}[Z^4] = 3$ ohne Berechnung einer Stammfunktion.

<!--en:404c679e-->
::: {.slide title="Variablentransformation durch Substitution in einer Dimension"}

<!--en:4d85c067-->
[Die Kettenregel rückwärts gelesen]{.kicker}

<!--en:93faff63-->
- **Kettenregel rückwärts gelesen:** Gilt $G' = f$, so folgt $(G \circ u)'(x) = f(u(x))\,u'(x)$. Damit liefert die Auswertungsformel die **eindimensionale Formel für die Variablentransformation**:
  $$\int_{u(a)}^{u(b)} f(y)\,dy = \int_a^b f(u(x))\,\frac{du}{dx}\,dx$$

<!--en:e4d8fa37-->
- **Geometrischer Streckungsfaktor $du/dx$:** Ein Streifen der Breite $\epsilon$ bei $x$ geht in einen Bildstreifen bei $y = u(x)$ mit Breite $\Delta y \approx u'(x)\epsilon$ und Höhe $f(u(x))$ über.

<!--en:d3a0c459-->
- **Rechenbeispiel:** In $\int_0^1 x e^{-x^2}\,dx$ liefert die Substitution $y = u(x) = x^2$ ($dy = 2x\,dx$) den Ableitungsfaktor $u'(x)$ und damit $\tfrac{1}{2}\int_0^1 e^{-y}\,dy = \tfrac{1 - e^{-1}}{2} \approx 0.316$.

<!--en:727073f3-->
::: {.slide title="Doppelintegrale und das Volumen unter einer Fläche"}

<!--en:63c272e7-->
[Mehrfachintegrale]{.kicker}

<!--en:8a433bfe-->
- **Vom Intervall zum Rechteck $R = [a, b] \times [c, d]$:** Die Grundfläche zerfällt in $\epsilon \times \epsilon$ große Quadrate der Fläche $\epsilon^2$. Jede Zelle trägt eine senkrechte Säule der Höhe $f(x_i, y_j)$. Für die Volumensumme gilt im Grenzwert $\epsilon \to 0$:
  $$\iint_R f(x, y)\,dx\,dy = \lim_{\epsilon \to 0} \sum_{i, j} \epsilon^2 f(x_i, y_j)$$

<!--en:0f081fb1-->
- **Orientiertes Volumen:** Wie in einer Dimension liefern Bereiche mit $f(x, y) > 0$ einen positiven Volumenbeitrag und Bereiche mit $f(x, y) < 0$ einen negativen.

<!--en:b1b7e542-->
- **Zweidimensionale Glockenkurve:** Die Fläche $z = e^{-x^2 - y^2}$ über $[-2, 2]^2$ ist rotationssymmetrisch. Ihr Funktionsterm zerfällt in das Produkt $e^{-x^2}e^{-y^2}$.

<!--en:08014634-->
::: {.slide title="Der Satz von Fubini und Integrale über Produkte"}

<!--en:a29e0d06-->
[Iterierte Integration]{.kicker}

<!--en:e2809f9b-->
- **Satz von Fubini:** Ist $f$ auf $R$ **absolut integrierbar** ($\iint_R |f| < \infty$) oder gilt überall $f \ge 0$ (**Satz von Tonelli**), lässt sich das Doppelintegral in beiden Reihenfolgen berechnen:
  $$\iint_R f(x,y)\,dx\,dy = \int_a^b\!\left(\int_c^d f(x,y)\,dy\right)dx$$

<!--en:c53c604c-->
- **Integrale über Produktfunktionen zerfallen vollständig:** Gilt $f(x, y) = g(x)h(y)$, so lässt sich der konstante Faktor aus dem inneren Integral herausziehen. Es folgt $\iint_R g(x)h(y)\,dx\,dy = \bigl(\int_a^b g\,dx\bigr)\bigl(\int_c^d h\,dy\bigr)$.

<!--en:b9a1eaee-->
- **Warum $\iint |f| < \infty$ wichtig ist:** Für $f(x,y) = \frac{x^2-y^2}{(x^2+y^2)^2}$ auf $[0,1]^2$ gilt $\iint |f| = \infty$. Die beiden Integrationsreihenfolgen liefern verschiedene Ergebnisse ($+\frac{\pi}{4}$ bzw. $-\frac{\pi}{4}$).

<!--en:26d649a0-->
::: {.slide title="Variablentransformation in mehreren Dimensionen und Jacobi-Determinante"}

<!--en:665ad500-->
[Volumenverzerrung in R^n]{.kicker}

<!--en:a3a7cdc0-->
- **Von der eindimensionalen Streckung $|u'(x)|$ zum $n$-dimensionalen Volumenfaktor $|\det \mathbf{J}_{\boldsymbol{\phi}}|$:** Sei $\boldsymbol{\phi} : \Omega \to \mathbb{R}^n$ ein $C^1$-Diffeomorphismus, also stetig differenzierbar und injektiv mit $\det \mathbf{J}_{\boldsymbol{\phi}}(\mathbf{x}) \ne 0$ überall auf der offenen Menge $\Omega$.

<!--en:d4ebffcd-->
- **Transformationssatz:** Für jede absolut integrierbare Funktion $f$ gilt
  $$\int_{\boldsymbol{\phi}(\Omega)} f(\mathbf{y})\,d\mathbf{y} = \int_{\Omega} f(\boldsymbol{\phi}(\mathbf{x}))\,\bigl|\det \mathbf{J}_{\boldsymbol{\phi}}\bigr|\,d\mathbf{x}$$

<!--en:b8bcabae-->
- **Geometrische Bedeutung:** Lokal wirkt $\boldsymbol{\phi}$ wie ihre Jacobi-Matrix $\mathbf{J}_{\boldsymbol{\phi}}(\mathbf{x})$. Sie bildet einen infinitesimalen Würfel mit Volumen $d\mathbf{x}$ auf ein Parallelotop mit Volumen $|\det \mathbf{J}_{\boldsymbol{\phi}}(\mathbf{x})|\,d\mathbf{x}$ ab.

<!--en:47376c4d-->
::: {.slide title="Transformation von Dichten in normalisierenden Flüssen"}

<!--en:4eaa37a6-->
[Tiefe generative Modelle]{.kicker}

<!--en:7015dbbc-->
- **Erhaltung der Wahrscheinlichkeitsmasse ($\mathbf{Y} = \boldsymbol{\phi}(\mathbf{X})$):** Aus $P(\mathbf{Y} \in \boldsymbol{\phi}(\Omega)) = P(\mathbf{X} \in \Omega)$ folgt $p_{\mathbf{Y}}(\boldsymbol{\phi}(\mathbf{x}))\,|\det\mathbf{J}_{\boldsymbol{\phi}}(\mathbf{x})| = p_{\mathbf{X}}(\mathbf{x})$.

<!--en:9f20ada0-->
- **Exakte Formel für die Log-Likelihood:** $\log p_{\mathbf{Y}}(\boldsymbol{\phi}(\mathbf{x})) = \log p_{\mathbf{X}}(\mathbf{x}) - \log|\det\mathbf{J}_{\boldsymbol{\phi}}(\mathbf{x})|$.

<!--en:9c46888b-->
- **Folge für die Architektur:** Flussschichten mit Dreiecksstruktur oder Kopplung berechnen $\log|\det\mathbf{J}_{\boldsymbol{\phi}}|$ in $O(n)$ statt $O(n^3)$.

<!--en:513e765a-->
::: {.slide title="Berechnung des Gauß-Integrals in Polarkoordinaten"}

<!--en:dfb5f10c-->
[Die Normierungskonstante √(2π)]{.kicker}

<!--en:b29c4bfb-->
Da $e^{-x^2}$ keine elementare Stammfunktion besitzt, lässt sich $I = \int_{-\infty}^\infty e^{-x^2}\,dx$ durch Quadrieren, Integration in zwei Dimensionen und den Wechsel zu Polarkoordinaten $\boldsymbol{\phi}(r, \theta) = (r\cos\theta, r\sin\theta)$ berechnen:

<!--en:4676b943-->
- **1. Quadrieren und faktorisieren (Fubini):** $I^2 = \iint_{\mathbb{R}^2} e^{-x^2-y^2}\,dx\,dy$.

<!--en:d53f6f0f-->
- **2. Jacobi-Determinante in Polarkoordinaten:** Eine Polarzelle mit den Seitenlängen $dr$ und $r\,d\theta$ hat den Volumenfaktor $|\det \mathbf{J}_{\boldsymbol{\phi}}(r, \theta)| = r(\cos^2\theta + \sin^2\theta) = r$.

<!--en:42689474-->
- **3. Mit dem gewonnenen Faktor $r$ integrieren:**
  $$I^2 = \int_0^{2\pi}\!\!\int_0^\infty r e^{-r^2}\,dr\,d\theta = 2\pi \cdot \frac{1}{2} = \pi$$
  Wurzelziehen ergibt $I = \sqrt{\pi} \approx 1.7725$.

<!--en:963c915f-->
::: {.slide title="Ausschöpfung der Ebene und Wahrscheinlichkeiten der Standardnormalverteilung"}

<!--en:f989df99-->
[Numerische Kontrolle]{.kicker}

<!--en:29733ffb-->
Die Substitution $z = \sqrt{2}\,x$ ($dz = \sqrt{2}\,dx$) in $\int_{-\infty}^\infty e^{-x^2}\,dx = \sqrt{\pi}$ liefert die Normierungskonstante der Standardnormalverteilung $\int_{-\infty}^\infty e^{-z^2/2}\,dz = \sqrt{2\pi} \approx 2.5066$. Außerdem gilt $P(-1 \le Z \le 1) = \operatorname{erf}(1/\sqrt{2}) \approx 0.6827$:

<!--en:c6ff3f56-->
Warum stimmen die zweidimensionalen Gittersummen auf $[-b, b]^2$ bei $\epsilon = 0.01$ auf 6 Stellen mit $\pi\operatorname{erf}(b)^2$ überein, obwohl die linke Rechteckregel normalerweise einen Fehler der Ordnung $O(\epsilon)$ hat? Bei einem geraden Integranden auf einem symmetrischen Intervall $[-b, b]$ verschwindet der systematische Fehler erster Ordnung $\tfrac{\epsilon}{2}(f(-b) - f(b))$ identisch.

<!--en:a165e868-->
::: {.slide title="Verständnisfragen zur Integralrechnung"}

<!--en:b6d9a92f-->
[Wiederholung und Übung]{.kicker}

<!--en:f1702164-->
Verständnisfragen zum Hauptsatz, zur Variablentransformation und zu den Momenten der Gauß-Verteilung:

<!--en:d7287dae-->
1. **GELU und ReLU als Integrale:** Sei $\phi(t) = \frac{1}{\sqrt{2\pi}}e^{-t^2/2}$ die Dichte der Standardnormalverteilung und $\Phi(x) = \int_{-\infty}^x \phi(t)\,dt$ ihre Verteilungsfunktion. Welche Ableitung ergibt sich aus dem Hauptsatz und der Produktregel für die $\mathrm{GELU}$-Aktivierung $\mathrm{GELU}(x) = x\Phi(x)$?

<!--en:6cc9f463-->
2. **Integrationsreihenfolge vertauschen (Fubini):** Im inneren Integral von $\int_0^1 \bigl(\int_y^1 e^{-x^2}\,dx\bigr)\,dy$ hat der Integrand keine elementare Stammfunktion. Die Integrationsreihenfolge über dem Dreieck $0 \le y \le x \le 1$ umkehren und das Integral in geschlossener Form berechnen.

<!--en:bec54e73-->
3. **Normierungskonstante der multivariaten Gauß-Verteilung:** Gegeben sei eine $d$-dimensionale Kovarianzmatrix $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ ($\det \mathbf{L} > 0$). Mit $\mathbf{x} = \mathbf{L}\mathbf{z}$ das Integral $\int_{\mathbb{R}^d} \exp(-\tfrac{1}{2}\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x})\,d\mathbf{x}$ berechnen.

<!--en:557b98b6-->
**Lösungen:**

<!--en:8f1e644c-->
1. Nach dem Hauptsatz gilt $\Phi'(x) = \phi(x)$. Die Produktregel liefert somit $\mathrm{GELU}'(x) = \Phi(x) + x\phi(x)$.

<!--en:6ef9a3ae-->
2. Bei senkrechter Zerlegung des Dreiecks $0 \le y \le x \le 1$ lauten die Grenzen $0 \le x \le 1$ und $0 \le y \le x$. Es gilt $\int_0^1 \bigl(\int_0^x e^{-x^2}\,dy\bigr)\,dx = \int_0^1 x e^{-x^2}\,dx = \tfrac{1}{2}(1 - e^{-1})$, denn die innere Integration über $y$ liefert genau den fehlenden Faktor $x$.

<!--en:292d4801-->
3. Bei der Substitution $\mathbf{x} = \mathbf{L}\mathbf{z}$ gilt $|\det \mathbf{J}| = |\det \mathbf{L}| = (\det \boldsymbol{\Sigma})^{1/2}$ und $\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x} = \|\mathbf{z}\|_2^2$. Nach dem Satz von Fubini zerfällt das Integral daher in das Produkt $(\det \boldsymbol{\Sigma})^{1/2}\prod_{i=1}^d \int_{-\infty}^\infty e^{-z_i^2/2}\,dz_i = (2\pi)^{d/2}(\det \boldsymbol{\Sigma})^{1/2}$.

<!--en:89076e84-->
::: {.slide title="Zusammenfassung der Integralrechnung"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:9e71216e-->
| Satz oder Verfahren | Formel | Bedeutung für Wahrscheinlichkeitsrechnung und Deep Learning |

<!--en:082cdf91-->
| **Bestimmtes Integral und systematischer Fehler** | $\int_a^b f(x)\,dx = \lim_{\|P\|\to 0}\sum_i f(\xi_i)\Delta x_i$ | Führender Fehlerterm der linken Rechtecksumme: $\tfrac{\epsilon}{2}(f(a)-f(b))$. Er verschwindet bei symmetrischen Glockenkurven auf symmetrischen Intervallen. |

<!--en:7954d658-->
| **Hauptsatz der Differential- und Integralrechnung** | $\frac{d}{dx}\int_a^x f(y)\,dy = f(x)$ und $\int_a^b f(x)\,dx = G(b)-G(a)$ | Ersetzt die Streifensumme durch die Auswertung einer Stammfunktion ($G'=f$ auf $[a,b]$) |

<!--en:ba75b645-->
| **Konvergenzkriterium für uneigentliche Potenzintegrale** | $\int_1^\infty x^{-s}dx = \frac{1}{s-1}$ ($s>1$) und $\int_0^1 x^{-s}dx = \frac{1}{1-s}$ ($s<1$) | Bestimmt, wann Verteilungen mit schweren Rändern endlichen Erwartungswert bzw. endliche Varianz haben |

<!--en:8c9a0c3b-->
| **Partielle Integration und Stein-Identität** | $\int_a^b u v'\,dx = [uv]_a^b - \int_a^b u'v\,dx$ und $\mathbb{E}[g'(Z)] = \mathbb{E}[Z g(Z)]$ | Überträgt Ableitungen von unbekannten Dichten auf andere Faktoren (Score-Matching, Momente) |

<!--en:5cb74a5b-->
| **Satz von Fubini** | $\iint_R f\,dx\,dy = \int_a^b(\int_c^d f\,dy)dx$ ($\iint\|f\|<\infty$ oder $f\ge 0$) | Ermöglicht die Marginalisierung gemeinsamer Dichten und zerlegt Integrale über unabhängige Faktoren |

<!--en:5083570e-->
| **Variablentransformation in $n$ Dimensionen** | $\int_{\boldsymbol{\phi}(\Omega)} f(\mathbf{y})\,d\mathbf{y} = \int_\Omega f(\boldsymbol{\phi}(\mathbf{x}))\|\det\mathbf{J}_{\boldsymbol{\phi}}(\mathbf{x})\|\,d\mathbf{x}$ | Berücksichtigt die Änderung der Log-Dichte um $\log\|\det\mathbf{J}_{\boldsymbol{\phi}}\|$ in normalisierenden Flüssen |

<!--en:b34ba343-->
| **Gauß-Integral** | $\int_{-\infty}^\infty e^{-x^2}dx = \sqrt{\pi}$ $\implies$ $\int_{-\infty}^\infty e^{-z^2/2}dz = \sqrt{2\pi}$ | Beweis mit ebenen Polarkoordinaten ($\|\det\mathbf{J}\| = r$). Liefert die Normierung von $\mathcal{N}(0,1)$. |
