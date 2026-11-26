<!--en:e79c11de-->
# Zustandsraummodelle

<!--en:2ee9485f-->
Wie lange muss eine Information im Gedächtnis bleiben? Das Subjekt eines Satzes ist für einige Wörter relevant, das Thema eines Absatzes für einige Hundert und die Identität einer Romanfigur für ein ganzes Buch. Im Prinzip kann die minGRU aus :numref:`sec_linear-recurrence` jede dieser Zeitskalen lernen. Ihre Gatter starten jedoch nahe $\sigma(0) = 1/2$, was einer Halbwertszeit von einem Schritt entspricht. Um ein Symbol über 1000 Schritte zu behalten, braucht sie einen Abklingfaktor nahe $0.999$ (Halbwertszeit 693 Schritte). Ein Gradient, den wir bei anfänglichen Abklingfaktoren von $1/2$ über 1000 Schritte zurückverfolgen, schrumpft um den Faktor $2^{-1000}$. Bei einfacher Genauigkeit führt das zu einem Unterlauf auf null. Eine einheitliche Initialisierung der Gatter stellt dem einzelnen Kanal auch kein breites Spektrum an Zeitskalen von Anfang an zur Verfügung.

<!--en:74f4035f-->
Zeitkontinuierliche dynamische Systeme bieten eine natürliche Möglichkeit, das Abklingen auf mehreren Zeitskalen zu parametrisieren. Ein *Zustandsraummodell* (engl. state space model, SSM) beschreibt das Gedächtnis jedes Kanals durch mehrere zeitkontinuierliche lineare Differentialgleichungen mit unterschiedlichen Abklingraten. Wir tasten das System mit der diskreten Schrittweite $\Delta$ ab. Die Diskretisierung ergibt eine lineare Rekurrenz. Das Produkt aus der Rate eines Zustands und der Schrittweite seines Kanals bestimmt, wie stark der Zustand pro Schritt abklingt. So können wir schon vor dem Training ein breites Spektrum an Halbwertszeiten vorgeben, jedem Kanal zugleich $N$ Zeitskalen zuweisen und die zeitinvariante Dynamik mittels schneller Fourier-Transformation (FFT) parallel als kausale Faltung (engl. causal convolution) auswerten. Wir leiten die exakte Diskretisierung mit einem Halteglied nullter Ordnung her und implementieren die diagonale S4D-Schicht von :citet:`Gu.Gupta.Goel.Re.2022`. Beim verzögerten Abruf nach 1000 Schritten prüfen wir, ob es für das Langzeitgedächtnis entscheidend ist, mit mehreren Zeitskalen zu starten.

<!--en:3e48283b-->
## Von der kontinuierlichen Zeit zur Rekurrenz

<!--en:527c09ea-->
### Zeitkontinuierliche Dynamik

<!--en:e642048b-->
Betrachten wir einen skalaren Zustand $h(t)$, der mit der Rate $\lambda$ abklingt und auf ein Eingangssignal $x(t)$ reagiert:

<!--en:6c79490d-->
Für $x(t) = 0$ lautet die Lösung $h(t) = e^{\lambda t} h(0)$. Stabilität erfordert $\lambda < 0$, sodass $h(t)$ exponentiell abklingt. Bei $\lambda = -1$ bleibt nach einer Einheit kontinuierlicher Zeit ein Anteil von $e^{-1} \approx 0.37$ übrig. Für $\lambda > 0$ wächst der Betrag des Zustands dagegen unbegrenzt. Entscheidend ist, dass $\lambda$ die Abklingrate *pro Zeiteinheit* angibt, nicht pro Tokenschritt.

<!--en:99e10c8a-->
Ein *Zustandsraummodell* (SSM) verallgemeinert :eqref:`eq_ssm_scalar` auf einen $N$-dimensionalen Zustandsvektor $\mathbf{h}(t) \in \mathbb{R}^N$. Eine skalare Eingabe $x(t)$ treibt ihn an, und seine Projektion ergibt eine skalare Ausgabe $y(t)$:

<!--en:f3fe1487-->
Dabei projiziert $\mathbf{B} \in \mathbb{R}^N$ die Eingabe in den Zustandsraum. Die *Zustandsmatrix* $\mathbf{A} \in \mathbb{R}^{N \times N}$ bestimmt die zeitliche Entwicklung des Zustands, $\mathbf{C} \in \mathbb{R}^{1 \times N}$ liest ihn aus und $D \in \mathbb{R}$ bildet eine direkte Skip-Verbindung.

<!--en:2ff4dddd-->
Wir beschränken uns auf Diagonalmatrizen $\mathbf{A} = \textrm{diag}(\lambda_1, \ldots, \lambda_N)$. Damit zerfällt :eqref:`eq_ssm_cont` in $N$ unabhängige Gleichungen der Form :eqref:`eq_ssm_scalar`. Ihre Raten sind $\lambda_1, \ldots, \lambda_N$, und $\mathbf{C}$ kombiniert ihre Lösungen linear. Bei einer dicht besetzten Matrix $\mathbf{A}$ betragen die Kosten $O(N^2)$ pro Schritt und $O(N^3)$ pro Verknüpfung im assoziativen Scan. Bei einer diagonalen Matrix $\mathbf{A}$ genügen $O(N)$ skalare Operationen. Über den komplexen Zahlen ist fast jede Matrix diagonalisierbar. Den Basiswechsel zu den Eigenvektoren können $\mathbf{B}, \mathbf{C}$ aufnehmen, sodass die Diagonalstruktur kaum Ausdrucksstärke kostet :cite:`Gu.Gupta.Goel.Re.2022`. Mit *reellen* Raten $\lambda_n < 0$ verzichten wir auf oszillierende Moden ($\lambda = -\alpha + i\omega$, siehe Übungen), können dafür aber rein reell rechnen.

<!--en:53018e88-->
Eine SSM-Schicht verarbeitet eine $d$-dimensionale Eingabesequenz, indem sie für jede Koordinate (*Kanal*) unabhängig ein solches System mit $N$ Zuständen auswertet. Der Gesamtzustand der Schicht umfasst damit $d \times N$ Zahlen. Das sind $N$-mal so viele wie bei einer minGRU der Breite $d$.

<!--en:4dd71554-->
### Diskretisierung

<!--en:964492d5-->
Um :eqref:`eq_ssm_cont` auf eine diskrete Tokensequenz anzuwenden, tasten wir das System mit der Schrittweite $\Delta$ ab. Schritt $t$ entspricht dann dem kontinuierlichen Zeitpunkt $t\Delta$. Halten wir die Eingabe auf dem Wert $x_t$ über das gesamte Intervall $((t-1)\Delta, t\Delta]$ konstant, lässt sich :eqref:`eq_ssm_scalar` exakt integrieren. Dies entspricht einem *Halteglied nullter Ordnung* (engl. zero-order hold, ZOH). Wegen $\frac{\textrm{d}}{\textrm{d}t}(h + x_t/\lambda) = \lambda(h + x_t/\lambda)$ klingt der verschobene Zustand $h + x_t/\lambda$ in einem Schritt um den Faktor $e^{\Delta\lambda}$ ab. Es gilt also $h_t + x_t/\lambda = e^{\Delta\lambda}(h_{t-1} + x_t/\lambda)$. Durch Umstellen erhalten wir die exakte diskrete lineare Rekurrenz:

<!--en:7e0658bd-->
Für $\lambda = -1$ gilt $\bar{b} = 1 - \bar{a}$. Damit wird :eqref:`eq_zoh` zur Konvexkombination $h_t = \bar{a}\, h_{t-1} + (1 - \bar{a})\, x_t$. Sie entspricht genau der Aktualisierung der minimalen GRU (:numref:`subsec_mingru`), wobei $\bar{a} = e^{-\Delta}$ als Update-Gatter dient. Die Schrittweite $\Delta$ steuert direkt die Gedächtnisdauer. Bei $\lambda = -1$ ergibt $\Delta = 0.01$ den Faktor $\bar{a} = 0.990$ (Halbwertszeit 69 Schritte), $\Delta = 0.1$ ergibt $\bar{a} = 0.905$ (Halbwertszeit 7 Schritte) und $\Delta = 1$ ergibt $\bar{a} = 0.37$.

<!--en:3b561892-->
Mit der Parametrisierung $\lambda = -e^\theta < 0$ und $\Delta > 0$ gilt wie bei einem Sigmoid-Gatter stets $\bar{a} = e^{\Delta\lambda} \in (0, 1)$. Neu ist, wie sich die Halbwertszeit $\ln 2 / (\Delta|\lambda|)$ aus zwei Faktoren ergibt (:numref:`fig_ssm_discretization`). Die $N$ Raten $\lambda_1, \ldots, \lambda_N$ verteilen die Zustände eines Kanals auf unterschiedliche relative Zeitskalen. Die gemeinsame Schrittweite $\Delta$ des Kanals streckt oder staucht alle $N$ Zeitskalen zugleich.

<!--en:ef08f5bb-->
![Das Halteglied nullter Ordnung überführt eine reelle Rate $\lambda < 0$ in einen Abklingfaktor pro Schritt $\bar{a} = e^{-\Delta|\lambda|}$, der nur vom Produkt $\Delta|\lambda|$ abhängt. Die Beschriftungen geben die Halbwertszeit in Schritten an, also $\ln 2 / (\Delta|\lambda|)$. Das schattierte Band zeigt den Bereich der S4D-Initialisierung mit $\Delta$ zwischen 0.001 und 0.1 und $|\lambda|$ zwischen 1 und 16.](../img/mdl-modernrnn-discretization.svg)

<!--en:a8e4daf6-->
Für eine diagonale Matrix $\mathbf{A}$ wenden wir ZOH elementweise an. Es gilt $\bar{a}_n = e^{\Delta \lambda_n}$ und $\bar{b}_n = (e^{\Delta \lambda_n} - 1) b_n / \lambda_n$. Den zweiten Ausdruck berechnen wir mit `expm1`, um bei kleinem $\Delta\lambda_n$ katastrophale Auslöschung zu vermeiden (zum Matrixexponential $\bar{\mathbf{A}} = e^{\Delta\mathbf{A}}$ für dicht besetzte Matrizen siehe :numref:`subsec_mdl-ode-driven-linear`).

<!--en:775ff312-->
Die reellwertige Variante von S4D initialisiert $\lambda_n = -n$ für $n = 1, \ldots, N$ (die Diagonale der HiPPO-Projektionsmatrix für Legendre-Polynome :cite:`Gu.Dao.Ermon.ea.2020`). Die Schrittweite $\Delta$ jedes Kanals ziehen wir logarithmisch gleichverteilt aus $[0.001, 0.1]$. Mit $N = 16$ reichen die anfänglichen Halbwertszeiten von weniger als einem halben Schritt ($\Delta = 0.1, \lambda = -16$) bis zu 693 Schritten ($\Delta = 0.001, \lambda = -1$). Damit decken sie bereits vor dem ersten Gradientenschritt das schattierte Band in :numref:`fig_ssm_discretization` ab.

<!--en:cce5006f-->
### Darstellung als Faltung

<!--en:b85c3a7c-->
Wenn $\bar{a}_n$, $\bar{b}_n$ und $\mathbf{C}$ unabhängig von der Tokenposition $t$ sind, ist das System *linear und zeitinvariant* (engl. linear time-invariant, LTI). Wickeln wir :eqref:`eq_zoh` ausgehend von $\mathbf{h}_0 = \mathbf{0}$ ab, sehen wir, dass Zustand $n$ die Summe $\sum_{k \ge 0} \bar{a}_n^k \bar{b}_n x_{t-k}$ akkumuliert. Die Auslesefunktion $\mathbf{C}$ summiert die $N$ Zustände zu einer einzigen *kausalen Faltung*:

<!--en:119e6caa-->
Die Impulsantwort $\mathbf{K} = (K_0, K_1, \ldots)$ jedes Kanals ist eine Linearkombination von $N$ abklingenden Exponentialfunktionen. Ein zeitinvariantes SSM lässt sich daher auf drei gleichwertige Arten darstellen (:numref:`fig_ssm_views`):

<!--en:bed05dd0-->
1. **Zeitkontinuierliche gewöhnliche Differentialgleichung** (engl. ordinary differential equation, ODE), deren Parameter die Raten und Schrittweiten festlegen.

<!--en:1fc07a07-->
2. **Diskrete Rekurrenz** für die autoregressive Generierung mit $O(1)$ Zeit- und Speicherbedarf pro Token.

<!--en:51f52739-->
3. **Kausale Faltung** über alle $T$ Positionen der Trainingssequenz mit einem Rechenaufwand von $O(T \log T)$ mittels schneller Fourier-Transformation (FFT). S4 :cite:`Gu.Goel.Re.2022` löste mithilfe dieser Faltungsdarstellung den Benchmark Path-X mit 16 384 Schritten. Die Darstellung als Faltung setzt eine zeitinvariante Dynamik voraus. Der parallele Scan aus :numref:`sec_linear-recurrence` funktioniert dagegen auch bei Abklingfaktoren, die von Token zu Token variieren.

<!--en:615bdadf-->
![Drei Darstellungen desselben Zustandsraummodells. Links reagiert der zeitkontinuierliche Zustand auf einen Eingangsimpuls. In der Mitte erscheint das diskretisierte System als lineare Rekurrenz mit einem Schritt pro Token. Rechts ergibt sich durch Abwickeln desselben Systems eine kausale Faltung. Ihr Faltungskern ist die Antwort auf einen einzelnen Impuls.](../img/mdl-modernrnn-ssm-views.svg)

<!--en:fce48777-->
### Zeitskalen und gelernte Faltungskerne

<!--en:f87a2adf-->
Was bietet ein diagonales SSM mit mehreren Zuständen zusätzlich zu einer minGRU, deren Gatter wir über die Bias-Terme auf die Abklingfaktoren von S4D initialisieren? Erstens skaliert die gemeinsame Schrittweite $\Delta$ eines Kanals alle $N$ Zeitskalen zugleich. In :numref:`sec_mamba` machen wir diese Skalierung eingabeabhängig. Zweitens kann jeder Kanal durch die Kombination von $N$ Zuständen über die Auslesefunktion $\mathbf{C}$ seinen Faltungskern $K_k = \sum_{n=1}^N c_n \bar{a}_n^k \bar{b}_n$ formen. Dieser ist eine Summe von $N$ Exponentialfunktionen und kann damit mehr darstellen als einen einzelnen monoton abklingenden Verlauf.

<!--en:cb864c26-->
Wie gut können $N$ reelle Exponentialfunktionen einen nichtmonotonen Faltungskern für den verzögerten Abruf annähern? Angenommen, ein Kanal soll die Eingabe von vor 20 Schritten auslesen. Der gewünschte Kern hat entweder eine scharfe Spitze ($K_{20} = 1$, sonst null) oder eine glatte Gauß-Glocke um die Verzögerung 20. Bei einem einzelnen Zustand ($N = 1$) liegt das Maximum immer bei der Verzögerung $0$. Wir passen nun die Auslesegewichte $\mathbf{c}$ von $N \in \{1, 4, 16\}$ Zuständen (mit den S4D-Raten $\lambda_n = -n$ und festem $\Delta = 0.05$) nach der Methode der kleinsten Quadrate an. Anschließend betrachten wir den relativen Fehler $\|\mathbf{K} - \textrm{target}\| / \|\textrm{target}\|$ und den Betrag des größten benötigten Auslesegewichts $|\mathbf{c}|_\infty$.

<!--en:6c5e8e17-->
Für $N = 1$ liegt das Maximum des Faltungskerns bei der Verzögerung 0, und der Kern verfehlt beide Ziele. Für $N = 16$ verschiebt sich das Maximum zur Verzögerung 20. Die glatte Glocke lässt sich mit einem relativen Fehler von 0.03 annähern, die scharfe Spitze dagegen nur schlecht (Fehler 0.89). Beachten Sie jedoch `|c|`. Eine so gute Anpassung mit reellen Exponentialfunktionen erfordert Auslesegewichte der Größenordnung $10^{11}$ mit wechselnden Vorzeichen, deren Beiträge sich fast aufheben. Der Grund ist, dass die reellen Exponentialfunktionen $e^{-n\Delta k}$ eine sehr schlecht konditionierte Basis bilden. Der Gradientenabstieg kann ausgehend von Gewichten der Größenordnung eins keine derart genaue Auslöschung erzielen. Reellwertige SSM-Faltungskerne bilden daher Mittelwerte über breite Zeitfenster. Komplexe Raten mit Schwingungsanteil $\lambda = -\alpha + i\omega$ liefern dagegen eine orthogonale, Fourier-ähnliche Basis (siehe Aufgabe 3).

<!--en:8b3a515e-->
## Eine diagonale Zustandsraumschicht

<!--en:8533ca20-->
### Die S4D-Schicht

<!--en:68f9d1f9-->
Die `S4D`-Schicht führt $N$ reelle Zustände für jeden ihrer $d$ Kanäle. Sie parametrisiert die Raten als $\lambda_n = -e^{\theta_n} < 0$ (`log_neg_rate`) und die Schrittweiten über `log_dt`. Wir setzen $\mathbf{B} = \mathbf{1}$ fest, da nur das Produkt $b_n c_n$ in :eqref:`eq_ssm_kernel` eingeht und die Schicht $\mathbf{C}$ lernt. Das Paar $(\bar{\mathbf{a}}, \bar{\mathbf{b}})$ berechnet die Methode `discretize` nach :eqref:`eq_zoh`.

<!--en:25cd3c62-->
Beim Training bildet der Vorwärtsdurchlauf für jeden Kanal den Faltungskern der Länge $T$ aus :eqref:`eq_ssm_kernel`. Die kausale Faltung berechnet er durch Zero-Padding und eine FFT der Länge $2T$. So vermeidet er, dass Beiträge vom Ende der Sequenz an ihrem Anfang erscheinen.

<!--en:e7479483-->
Bei der autoregressiven Generierung aktualisiert `step` den rekurrenten Zustand der Größe $d \times N$ um ein Token. Das erfordert $O(dN)$ Zeit und konstanten Speicher.

<!--en:457429a8-->
Prüfen wir, ob die FFT-Faltung, die parallele Auswertung mit `linear_scan` und die sequenzielle Auswertung mit `step` innerhalb der Rundungstoleranz für Gleitkommazahlen einfacher Genauigkeit übereinstimmen.

<!--en:452fa502-->
Die drei Darstellungen stimmen bis auf $\sim 10^{-7}$ überein. Bei LTI-Schichten erspart uns die FFT-Faltung den vierdimensionalen Zwischentensor der Form $(B, T, d, N)$ für die Zustände des gesamten Batchs. Sie benötigt nur die Potenzen für den Faltungskern in einem Tensor der Form $(d, N, T)$.

<!--en:9ba405fa-->
### Abruf nach langen Verzögerungen

<!--en:f96efe62-->
Bestimmen die anfänglichen Zeitskalen, wie lange das Gedächtnis zurückreicht? Um das zu prüfen, greifen wir auf den Benchmark zum verzögerten Abruf aus :numref:`sec_lstm` zurück. Wir wählen Verzögerungen von $T = 100$ und $T = 1{,}000$ Schritten und verwenden identische leere Fülltoken (`blank_fillers=True`). So prüft die Aufgabe ausschließlich, wie lange eine Information erhalten bleibt, und erfordert keine inhaltsabhängige Filterung (:numref:`sec_mamba`).

<!--en:574f60f4-->
Wir vergleichen vier Konfigurationen, in denen wir die Schicht und ihre anfänglichen Zeitskalen unabhängig voneinander variieren (:numref:`fig_ssm-recall-design`):

<!--en:d46724b2-->
1. **`minGRU` in der Standardeinstellung** mit anfänglichen Gatterwerten nahe $1/2$.

<!--en:f95d7602-->
2. **`S4D` in der Standardeinstellung** ($d = 64, N = 16$, $\Delta \in [0.001, 0.1]$, Halbwertszeiten bis zu 693 Schritten).

<!--en:7a3b1b74-->
3. **`S4D, short timescales`** mit einer Initialisierung von $\Delta \in [0.1, 1.0]$, bei der keine anfängliche Halbwertszeit 7 Schritte übersteigt.

<!--en:eecfeee3-->
4. **`minGRU, S4D timescales`** mit Bias-Termen der Update-Gatter, die wir über `init_timescales` initialisieren. Damit gilt $\sigma(b) = e^{-\Delta}$ für $\Delta \in [0.001, 0.1]$ ($b = -\log(e^\Delta - 1)$). Jedes Gatter entspricht so dem langsamsten Zustand des jeweiligen S4D-Kanals.

<!--en:5fe7d3d1-->
![Die vier Konfigurationen für den Abruf nach langen Verzögerungen: zwei Schichten (minGRU und S4D), jeweils mit kurzen und mit langen anfänglichen Zeitskalen. Jede Zelle zeigt die längste anfängliche Halbwertszeit und den Anteil eines gespeicherten Werts, den dieser langsamste Zustand nach 1000 Schritten noch enthält. Geben die Zeitskalen den Ausschlag, unterscheiden sich die Spalten. Gibt die Schicht den Ausschlag, unterscheiden sich die Zeilen.](../img/mdl-modernrnn-recall-grid.svg)

<!--en:0ccbeca0-->
Wir trainieren jedes Modell mit zwei verschiedenen Zufallsstartwerten über `d2l.train_delayed_recall` (1500 Schritte mit Adam bei einer Lernrate von $3 \times 10^{-3}$). Für die beiden S4D-Varianten geben wir nach dem Training auch die längste Halbwertszeit $\ln 2 / \min_{d,n}(\Delta|\lambda_n|)$ aus.

<!--en:c35c7df7-->
Das Ergebnis hängt von den anfänglichen Zeitskalen ab, unabhängig vom Schichttyp. Bei $T = 1{,}000$ erreichen `S4D` und `minGRU, S4D timescales` in allen Läufen 100% Genauigkeit. Die Standardversion von `minGRU` und die Variante `S4D, short timescales` bleiben dagegen auf Zufallsniveau ($0.12$). Warum gelingt der Abruf mit `S4D, short timescales` dennoch bei $T = 100$? Die längste anfängliche Halbwertszeit beträgt zwar nur 7 Schritte, doch nach 99 Schritten bleibt vom Signal noch der Anteil $e^{-0.1 \times 99} \approx 5 \times 10^{-5}$ übrig. Das liegt über der relativen Genauigkeitsgrenze von float32. Das Gradientensignal reicht aus, damit Adam die längste Halbwertszeit auf 100 bis 200 Schritte ausdehnen kann. Über 999 Schritte ist der verbleibende Anteil $e^{-100} \approx 10^{-44}$ dagegen so klein, dass praktisch kein Gradient mehr zum Lernen beiträgt. Die längste Halbwertszeit bleibt bei etwa 7 Schritten.

<!--en:d5817aad-->
Eine zeitinvariante Dynamik hat jedoch eine grundlegende Einschränkung. Jedes Token im Abstand $k$ erhält dasselbe Gewicht $K_k$, ob es nun ein entscheidendes Symbol oder ein bedeutungsloses Fülltoken ist. In :numref:`sec_mamba` verbinden wir die verschiedenen Zeitskalen der S4D-Zustände mit einer eingabeabhängigen Auswahl. Dazu lassen wir jedes Token seine eigene Schrittweite $\Delta_t$ wählen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:d59e130a-->
Diskretisieren wir ein zeitkontinuierliches diagonales Zustandsraummodell $\mathbf{h}'(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}x(t)$ mit Raten $\lambda_n < 0$ mithilfe eines Halteglieds nullter Ordnung und der Schrittweite $\Delta$, so erhalten wir eine lineare Rekurrenz mit Abklingfaktoren $\bar{a}_n = e^{\Delta\lambda_n}$ und Halbwertszeiten $\ln 2 / (\Delta|\lambda_n|)$. Wir initialisieren $\lambda_n = -n$ und ziehen $\Delta$ logarithmisch gleichverteilt aus $[0.001, 0.1]$. Damit hat jeder Kanal von `S4D` insgesamt $N$ exponentiell abklingende Zustände, deren Halbwertszeiten von weniger als 1 Schritt bis fast 700 Schritte reichen. Da die Parameter zeitinvariant sind, lässt sich `S4D` bei der Inferenz als Rekurrenz mit $O(1)$ Speicherbedarf für den Zustand auswerten. Im Training liefert die kausale Faltung mittels FFT dasselbe Ergebnis. Beim verzögerten Abruf nach 1000 Schritten hängt der Lernerfolg entscheidend davon ab, dass einige Zustände von Anfang an langsam abklingen. Das gilt sowohl für `S4D` als auch für die Übertragung dieser Zeitskalen auf die Bias-Terme der Gatter von `minGRU`. Mit fester Dynamik lassen sich irrelevante Eingaben jedoch weiterhin nicht gezielt ausblenden.

<!--en:2294243a-->
## Übungen

<!--en:92b9e130-->
1. [code] **Diskretisierungsverfahren.** Das Halteglied nullter Ordnung ist eine von mehreren Möglichkeiten, :eqref:`eq_ssm_scalar` in eine Rekurrenz zu überführen. Das explizite Euler-Verfahren verwendet $\bar{a} = 1 + \Delta\lambda$, das implizite Euler-Verfahren $\bar{a} = 1/(1 - \Delta\lambda)$ und das bilineare Verfahren $\bar{a} = (1 +
   \Delta\lambda/2)/(1 - \Delta\lambda/2)$. In :numref:`sec_mdl-stiffness-implicit` untersuchen wir die beiden Euler-Verfahren zur Lösung einer Differentialgleichung.
    1. Bestimmen Sie für jedes Verfahren die Schrittweiten $\Delta$, für die bei jedem $\lambda < 0$ die Bedingung $|\bar{a}| < 1$ gilt. Welches Verfahren kann ein stabiles zeitkontinuierliches System instabil machen?
    1. Zeigen Sie, dass das implizite Euler-Verfahren für $\lambda = -1$ die Aktualisierung $h_t =
       z h_{t-1} + (1 - z) x_t$ mit $z = \sigma(-\log\Delta)$ ergibt. Dies ist die Mischung einer GRU, deren Update-Gatter den negativen Logarithmus der Schrittweite als Präaktivierung erhält.
    1. Simulieren Sie für jedes Verfahren die Antwort auf einen Einheitssprung für $\lambda = -1$ und $\Delta \in \{0.1, 1, 3\}$. Vergleichen Sie Ihre Ergebnisse mit der exakten Lösung $1 - e^{-t}$.

<!--en:215647ef-->
1. [code] **Verzögerungen und Faltungskerne.** Untersuchen Sie mit `fit_kernel`, welche Gedächtnisinhalte $N$ reelle Exponentialfunktionen darstellen können.
    1. Nähern Sie einen Einheitsimpuls mit den Verzögerungen 5, 20 und 50 durch $N = 16$ Zustände bei $\Delta = 0.05$ an. Wie ändert sich der Fehler mit der Verzögerung? Setzen Sie Ihre Antwort in Beziehung zu den Halbwertszeiten des langsamsten und des schnellsten Zustands.
    1. Nähern Sie den Impuls bei Verzögerung 20 für Schrittweiten $\Delta$ zwischen 0.01 und 0.2 an. Welches $\Delta$ ergibt die beste Anpassung? Was legt das für die Schrittweite nahe, die ein trainiertes S4D für einen Kanal mit einer bestimmten benötigten Gedächtnisdauer wählen sollte?
    1. Passen Sie $\mathbf{c}$ ausgehend von null durch Gradientenabstieg an (Adam, einige Tausend Schritte), statt die Methode der kleinsten Quadrate zu verwenden. Wie nahe kommen Sie dem Fehler dieser Methode bei der Glocke und beim Impuls? Was lässt sich anhand der Gewichte aus der Methode der kleinsten Quadrate vorhersagen?

<!--en:f62ddc7b-->
1. [code] **Komplexe Raten.** Ein Zustand mit der komplexen Rate $\lambda =
   -\alpha + i\omega$, $\alpha > 0$, klingt ab, während er rotiert.
    1. Zeigen Sie, dass das Halteglied nullter Ordnung $\bar{a} = e^{-\alpha\Delta}
       e^{i\omega\Delta}$ ergibt. Zeigen Sie außerdem, dass zwei Zustände mit paarweise konjugiert komplexen Raten, Eingabegewichten und Auslesegewichten einen reellen Beitrag $K_k = r\,
       e^{-\alpha\Delta k} \cos(\omega\Delta k + \varphi)$ zum Faltungskern liefern, für geeignete $r \ge 0$ und $\varphi$. Wie viele reelle Parameter hat ein solches Paar, und wie viele hat ein Paar reeller Zustände?
    1. Welche Faltungskerne lassen sich mit solchen Paaren darstellen, mit reellen Raten aber nicht? Denken Sie an eine Eingabe, die sich mit fester Periode wiederholt.
    1. Ändern Sie `fit_kernel` so, dass die Funktion die komplexen Raten $\lambda_m = -1/2 + i\pi
       m$ für $m = 0, \ldots, N-1$ verwendet (die S4D-Lin-Initialisierung von :citet:`Gu.Gupta.Goel.Re.2022`) und den Realteil des Faltungskerns anpasst. Vergleichen Sie für den Impuls bei Verzögerung 20 den Fehler und die Beträge der Gewichte mit den Ergebnissen für reelle Raten. Erklären Sie den Unterschied.

<!--en:f23f14ba-->
1. [code] **Faltung mittels FFT.** Der Vorwärtsdurchlauf von `S4D` ergänzt die Sequenz und den Faltungskern vor der FFT durch Nullen auf die Länge $2T$.
    1. Entfernen Sie das Padding, sodass die FFT die Länge $T$ hat, und vergleichen Sie die Ausgaben mit denen von `step`. Welche Ausgaben sind falsch, und welche Beiträge wurden zu ihnen addiert?
    1. Welche Mindestlänge ist nach dem Padding nötig, damit sich die exakte kausale Faltung ergibt? Warum runden Implementierungen diese Länge oft auf eine Zweierpotenz auf?

<!--en:bb1cffb5-->
[Diskussion](https://d2l.discourse.group/t/36781)

<!--en:7e974109-->
Strukturierte Zustandsraummodelle (S4D)<br>**zeitkontinuierliche Dynamik · Diskretisierung mit Halteglied nullter Ordnung · ODE, Rekurrenz und Faltung · Initialisierung mit mehreren Zeitskalen**

<!--en:7707b282-->
::: {.slide title="Warum ein Gedächtnis mit mehreren Zeitskalen strukturierte Abklingraten braucht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:c5811638-->
- **Drei Größenordnungen:** In Sprache erstrecken sich syntaktische Beziehungen über 10 Token, Absatzthemen über 200 Token und Bezüge auf dieselbe Entität über 1000 oder mehr Token.

<!--en:a8b45d49-->
- **Unterlauf bei Bias null:** Gatter mit $\sigma(0) = 0.5$ schwächen ein Signal über 1000 Schritte um den Faktor $2^{-1000} = 0$ ab. Das Ergebnis ist in `float32` null.

<!--en:2b72389f-->
- **Strukturierte Zustände:** In einem zeitkontinuierlichen SSM hat jeder Kanal $N$ Abklingraten $\lambda_n = -n$, skaliert mit einer logarithmisch gleichverteilten Schrittweite $\Delta$.

<!--en:18322994-->
::: {.slide title="Zeitkontinuierliche lineare Zustandsraummodelle als Differentialgleichungen"}

<!--en:921fe6a0-->
[Zeitkontinuierliche Dynamik]{.kicker}

<!--en:99303edb-->
- **Exponentielle Stabilität:** Negative Realteile $\textrm{Re}(\lambda_n) < 0$ sorgen dafür, dass die homogenen Lösungen $e^{\lambda_n t}$ stetig abklingen und beschränkt bleiben.

<!--en:a4bb8231-->
- **Vier Operatoren:** Schreibvektor $\mathbf{B}$, Zustandsmatrix $\mathbf{A} \in \mathbb{C}^{N \times N}$, Auslesefunktion $\mathbf{C}$ und Skip-Verbindung $D$.

<!--en:6b9a2f12-->
::: {.slide title="Diagonale Zustandsmatrizen mit N verlustbehafteten Integratoren pro Kanal"}

<!--en:c7674127-->
[Diagonalstruktur]{.kicker}

<!--en:2bdd717c-->
- **Entkoppelte Zustände mit Aufwand $O(N)$:** Die Diagonalmatrix $\mathbf{A} = \textrm{diag}(\lambda_1, \ldots, \lambda_N)$ macht aus einer Matrix-ODE mit Aufwand $O(N^2)$ pro Kanal $N$ parallele skalare verlustbehaftete Integratoren (Gesamtzustand der Größe $d \times N$).

<!--en:3eecd81a-->
- **Reelle und komplexe Diagonale:** Reelle Raten $\lambda_n = -e^{\theta_n}$ ergeben glatt abklingende Fenster. Komplexe Raten ermöglichen zusätzlich Schwingungen.

<!--en:6d2b3497-->
::: {.slide title="Exakte Diskretisierung mit einem Halteglied nullter Ordnung (ZOH)"}

<!--en:9fe1a3e7-->
[Von kontinuierlicher zu diskreter Zeit]{.kicker}

<!--en:18a871ee-->
- **Halteglied nullter Ordnung:** Bleibt $x(\tau) \equiv x_t$ auf $((t-1)\Delta, t\Delta]$ konstant, lässt sich die lineare ODE in geschlossener Form integrieren.

<!--en:dfa42e42-->
- **Numerische Stabilität:** $\bar{b}$ mit `expm1(dt * rate) / rate` berechnen, um für $\Delta|\lambda| \ll 1$ Auslöschung zu vermeiden.

<!--en:d66403c7-->
::: {.slide title="Die Schrittweite Δ steuert die Dehnung der Zeitskalen"}

<!--en:ab5225f7-->
[Schrittweite und Update-Gatter]{.kicker}

<!--en:6daaceda-->
- **Exakte Konvexkombination ($\lambda = -1$):** Die ZOH-Diskretisierung entspricht der Aktualisierung von GRU bzw. `minGRU` mit dem Erhaltungsfaktor $\bar{a} = e^{-\Delta}$ des Gatters.

<!--en:5fbf05a1-->
- **Umgekehrt proportionale Zeitskala:** Sinkt $\Delta$ von $1.0$ auf $0.001$, steigt die diskrete Halbwertszeit des Gedächtnisses von $0.7$ auf $693$ Schritte.

<!--en:ec27d088-->
::: {.slide title="Faktoren der Halbwertszeit und Initialisierungsbereich von S4D"}

<!--en:0fbf111a-->
[Initialisierung mit mehreren Zeitskalen]{.kicker}

<!--en:974aec87-->
- **Faktoren der Halbwertszeit:** Bei $\lambda_n < 0$ und einer Schrittweite $\Delta_c > 0$ des Kanals hat der Zustand $(c, n)$ die Halbwertszeit
  $$\tau_{1/2}(c, n) = \frac{\ln 2}{\Delta_c\, |\lambda_n|}$$

<!--en:881bea9d-->
- **Reelle S4D-Initialisierung:** $\lambda_n = -n$ für $n = 1, \ldots, 16$ setzen (Diagonale der HiPPO-Legendre-Matrix, Gu et al., 2020) und $\Delta_c$ logarithmisch gleichverteilt aus $[0.001, 0.1]$ ziehen.

<!--en:46988ad1-->
- **Spektrum bei Schritt 0:** Die anfänglichen Halbwertszeiten reichen von $\ln 2 / (0.1 \times 16) = \mathbf{0.43\text{ steps}}$ bis $\ln 2 / (0.001 \times 1) = \mathbf{693\text{ steps}}$.

<!--en:ad169f10-->
::: {.slide title="Zeitinvariante SSMs als Faltung"}

<!--en:78aa2b6e-->
[LTI-Dualität]{.kicker}

<!--en:affc721a-->
- **Impulsantwort als Filter:** Sind $(\bar{a}_n, \bar{b}_n, c_n)$ zeitinvariant, ergibt das Abwickeln der Rekurrenz ein kausales Filter der Länge $T$ mit dem Faltungskern $\mathbf{K}$.

<!--en:c821ec5e-->
- **FFT-Training mit Aufwand $O(T \log T)$:** $\mathbf{X}$ und $\mathbf{K}$ mit Nullen auf die Länge $2T$ ergänzen und alle $T$ Ausgaben parallel mit `rfft` und `irfft` berechnen.

<!--en:a616463a-->
::: {.slide title="Ein System in drei Darstellungen: ODE, Rekurrenz und Faltung"}

<!--en:44822927-->
[Drei Darstellungen eines SSM]{.kicker}

<!--en:3511b1ee-->
| Darstellung | Mathematische Form | Hauptanwendung im Deep Learning |

<!--en:411a20aa-->
| **1. Zeitkontinuierliche ODE** | $\mathbf{h}'(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}x(t)$ | Systematische Initialisierung mehrerer Zeitskalen ($\lambda_n, \Delta$) und Unabhängigkeit von der Auflösung |

<!--en:07755cb0-->
| **2. Diskrete Rekurrenz** | $\mathbf{h}_t = \bar{\mathbf{a}} \odot \mathbf{h}_{t-1} + \bar{\mathbf{b}} x_t$ | Schnelle autoregressive Tokengenerierung mit $O(1)$ Speicherbedarf und $O(dN)$ Rechenaufwand pro Token (`step`) |

<!--en:b43be6a6-->
| **3. Kausale Faltung** | $y = \mathbf{K} * x + D x$ | Schnelles paralleles LTI-Training mit Aufwand $O(Td\log T)$ durch Zero-Padding und FFT der Länge $2T$ |

<!--en:15d1b09d-->
::: {.slide title="Was können N reelle Exponentialfunktionen annähern?"}

<!--en:3f6a346f-->
[Ausdrucksstärke und Konditionierung des Faltungskerns]{.kicker}

<!--en:baa5f505-->
Kann eine Summe von $N$ monotonen reellen Exponentialfunktionen $K_k = \sum_{n=1}^N c_n \bar{a}_n^k \bar{b}_n$ ($\lambda_n = -n, \Delta = 0.05$) ihr Maximum bei der Verzögerung $k = 20$ erreichen und so ein früheres Token auslesen? $\mathbf{c} \in \mathbb{R}^N$ nach der Methode der kleinsten Quadrate an zwei Ziele anpassen: eine scharfe **Spitze** bei Verzögerung 20 und eine glatte **Gauß-Glocke** um Verzögerung 20.

<!--en:45106644-->
**Größe der Gewichte in `bump |c|` ($\mathbf{8 \times 10^{10}}$):** Mit $N = 16$ reellen Exponentialfunktionen nähert die Methode der kleinsten Quadrate die Glocke bei Verzögerung 20 gut an (Fehler $0.03$). Dazu braucht sie Auslesegewichte der Größenordnung $\sim 10^{11}$ mit wechselnden Vorzeichen, deren Beiträge sich auf $10$ Dezimalstellen auslöschen.

<!--en:43f24a52-->
::: {.slide title="Schlechte Konditionierung bei Basen aus reellen Exponentialfunktionen"}

<!--en:482d49ef-->
[Warum reelle SSMs über Zeitfenster mitteln]{.kicker}

<!--en:7a421645-->
- **Schlechte Konditionierung der Vandermonde-Matrix:** Reelle Moden $e^{-n\Delta k}$ klingen monoton ab. Eine verzögerte Glocke bei $k^*=20$ erfordert die Subtraktion nahezu kollinearer Kurven mit Gewichten der Größenordnung $\|\mathbf{c}\|_\infty \sim 10^{11}$.

<!--en:24d6c5c2-->
- **Komplex konjugierte Raten (`S4D-Lin`):** Durch Imaginärteile in $\lambda_m = -1/2 \pm i\pi m$ entstehen orthogonale gedämpfte Kosinusfunktionen $e^{-\Delta k/2}\cos(\omega_m \Delta k)$, mit denen sich verzögerte Spitzen mit Gewichten der Größenordnung $O(1)$ annähern lassen.

<!--en:e3432875-->
::: {.slide title="Implementierung von S4D: Diskretisierung, FFT und Einzelschritt"}

<!--en:4b336367-->
[Aufbau des S4D-Codes]{.kicker}

<!--en:1745f88b-->
`S4D` verwendet die Parameter `log_neg_rate` ($\lambda_n = -e^{\theta_n} < 0$) und `log_dt` ($\Delta = e^\phi > 0$). Dabei ist $\mathbf{B} = \mathbf{1}$ fest, da nur $b_n c_n$ in $K_k$ eingeht. Die Schicht unterstützt sowohl FFT-Training als auch Generierung mit einem Zustand konstanter Größe ($O(1)$):

<!--en:24c722e9-->
- **Speichervorteil beim LTI-Training:** Die FFT benötigt pro Batch nur einmal den Potenztensor der Form $(d, N, T)$. Der vierdimensionale Zustand der Form $(B, T, d, N)$ für alle Sequenzen entfällt.

<!--en:1e976f6f-->
::: {.slide title="Numerische Übereinstimmung von Faltung, Scan und Einzelschritten"}

<!--en:76eb996f-->
[Überprüfung der S4D-Schicht]{.kicker}

<!--en:15c10c38-->
Das `S4D`-Modul speichert `log_neg_rate` ($\lambda_n = -e^{\theta_n} < 0$), `log_dt` ($\Delta = e^{\phi} > 0$), die Auslesefunktion `C` und die Skip-Verbindung `D`. Dabei bleibt $\mathbf{B} = \mathbf{1}$ fest, da nur $b_n c_n$ in $K_k$ eingeht. Vergleich von FFT-basiertem `forward`, `linear_scan` und tokenweiser Auswertung mit `step` für $T = 50$:

<!--en:7342022b-->
Alle drei Algorithmen berechnen bis auf Rundungsfehler in `float32` ($\sim 10^{-7}$) exakt dieselbe mathematische Funktion.

<!--en:15881a62-->
::: {.slide title="Einfluss von Zeitskalen und Architektur auf den Abruf"}

<!--en:5fc3bff8-->
[Aufbau der kontrollierten Ablationsstudie]{.kicker}

<!--en:04ba46d0-->
- **2×2-Versuchsplan:** Zwei Architekturen (`minGRU` mit $N=1$ und `S4D` mit $N=16$) und zwei anfängliche Halbwertszeitspektren ($\le 7$ Schritte bzw. bis zu $693$ Schritte) beim verzögerten Abruf ($T \in \{100, 1{,}000\}$).

<!--en:bda7ebd7-->
- **Übertragung der S4D-Zeitskalen auf Gatter:** Mit $b_c = -\log(e^{\Delta_c} - 1)$ erhält jedes Gatter von `minGRU` den Anfangswert $\sigma(b_c) = e^{-\Delta_c}$ und damit denselben logarithmisch gleichverteilten Zeithorizont wie bei S4D.

<!--en:27b81d22-->
::: {.slide title="Abruf nach 1000 Schritten: Die anfänglichen Zeitskalen entscheiden"}

<!--en:935fbeb2-->
[Ergebnisse beim Abruf nach langen Verzögerungen]{.kicker}

<!--en:1b486384-->
Testgenauigkeit (Zufallsniveau $= 0.125$) und längste Halbwertszeit nach dem Training ($\ln 2 / \min(\Delta|\lambda_n|)$) bei zwei Zufallsstartwerten für $T = 100$ und $T = 1{,}000$:

<!--en:7ec9d66c-->
**Die Kontrollvarianten tauschen die Plätze:** Bei $T = 1{,}000$ erreicht `minGRU` mit den anfänglichen Zeitskalen von S4D eine **Genauigkeit von $1.00$**. Dagegen fällt `S4D` mit kurzen anfänglichen Zeitskalen ($\le 7$ Schritte) auf **$0.12$ (Zufallsniveau)**.

<!--en:a5233211-->
::: {.slide title="Warum 7 Schritte Halbwertszeit anfangs für 100 Schritte genügen, für 1000 aber nicht"}

<!--en:48906f30-->
[Reichweite von Gradienten in Gleitkommaarithmetik]{.kicker}

<!--en:56026645-->
Die unteren Zeilen der Ausgabetabelle betrachten (`slowest half-life after training`):

<!--en:f90ef5bc-->
- **Bei $T = 100$ (`S4D, short timescales` lernt):** Mit der langsamsten anfänglichen Rate $\Delta|\lambda_1| = 0.1$ bleibt nach $99$ Füllschritten der Signalanteil $e^{-0.1 \times 99} = e^{-9.9} \approx \mathbf{5 \times 10^{-5}}$ erhalten. Da $5 \times 10^{-5}$ deutlich über der relativen Genauigkeitsgrenze von `float32` ($\sim 10^{-7}$) liegt, erhält Adam einen Gradienten ungleich null. Die längste Halbwertszeit steigt von **$7$ Schritten auf etwa $100$–$200$ Schritte**.

<!--en:20ffc36d-->
- **Bei $T = 1{,}000$ (`S4D, short timescales` stagniert):** Nach $999$ Schritten bleibt vom Signal anfangs nur der Anteil $e^{-0.1 \times 999} \approx \mathbf{4 \times 10^{-44}}$. Bei der Gradientenakkumulation in `float32` geht das Signal durch Unterlauf ganz verloren. Die längste Halbwertszeit bleibt bei **$7$ Schritten**.

<!--en:f270d9d8-->
::: {.slide title="Warum LTI-Modelle den Tokeninhalt beim Gewichten übergehen"}

<!--en:3ad8870e-->
[Von Zeitinvarianz zu Selektivität]{.kicker}

<!--en:2a010b17-->
Warum kommen *identische leere Fülltoken* (`blank_fillers=True`) beim S4D-Test zum Abruf nach $1{,}000$ Schritten zum Einsatz?

<!--en:1d3ffda7-->
- **Feste Gewichte pro Abstand:** Bei einer zeitinvarianten Faltung $y_t = \sum_{k=0}^{t-1} K_k x_{t-k} + D x_t$ hängt das Gewicht $K_k$ **nur vom Abstand $k$** ab, niemals vom Token $x_{t-k}$ selbst.

<!--en:1fa20c86-->
- **Keine Anpassung an störende Fülltoken oder wechselnde Abstände:** Stehen wichtige Token an unvorhersehbaren Positionen zwischen wechselnden Fülltoken, mischt jedes feste Filter $K_k$ Zieltoken und Störsignale mit festen Gewichten.

<!--en:45663d15-->
**Ausblick (§{sec_mamba} Selektive Zustandsraummodelle):** Schrittweite $\boldsymbol{\Delta}_t$, Schreibvektor $\mathbf{B}_t$ und Lesevektor $\mathbf{C}_t$ als Funktionen des Eingabetokens $\mathbf{x}_t$ wählen. Das verbindet die verschiedenen Zeitskalen der S4D-Zustände mit der inhaltsabhängigen Gattersteuerung von `minGRU`.

<!--en:c0622270-->
::: {.slide title="Verständnisfragen zur ZOH-Diskretisierung und zum FFT-Padding"}

<!--en:929b7671-->
[Wissenskontrolle]{.kicker}

<!--en:a3845fe0-->
Drei Fragen zu zeitkontinuierlichen Zustandsraummodellen:

<!--en:a5c7c573-->
1. Ein zeitkontinuierlicher Zustand hat die Rate $\lambda = -2$. Bei welcher Schrittweite $\Delta$ ergibt sich mit einem Halteglied nullter Ordnung eine diskrete Halbwertszeit von $500$ Tokenschritten?

<!--en:7a7bd16f-->
2. Warum ergänzt `S4D.forward` sowohl die Eingabesequenz $\mathbf{X}$ als auch den Faltungskern $\mathbf{K}$ von Länge $T$ auf Länge $2T$ mit Nullen, bevor `rfft` und `irfft` aufgerufen werden?

<!--en:f044c02b-->
3. Wie lässt sich das anfängliche S4D-Spektrum der Abklingfaktoren $\bar{a} = e^{-\Delta}$ ($\Delta \in [0.001, 0.1]$) auf den Bias $b$ des Update-Gatters mit der üblichen Sigmoidfunktion $\sigma(b)$ übertragen?

<!--en:557b98b6-->
**Lösungen:**

<!--en:975df078-->
2. Multiplikation im Frequenzbereich der DFT berechnet eine **zyklische Faltung** mit Periode $N_{\text{FFT}}$. Ohne Zero-Padding auf mindestens $2T - 1$ (aufgerundet auf $2T$) überlagern späte Token bei $t \approx T$ die frühen Ausgaben bei $t \approx 0$ und verletzen damit die Kausalität.

<!--en:a66f8a13-->
3. Die Gleichung nach dem Bias auflösen: $\sigma(b) = 1/(1 + e^{-b}) = e^{-\Delta} \implies e^{-b} = e^\Delta - 1 \implies \mathbf{b = -\log(e^\Delta - 1)}$ (berechnet mit `-log(expm1(dt))`).

<!--en:835f3573-->
::: {.slide title="Zusammenfassung: Strukturierte Zustandsraummodelle (S4D)"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:3d797bd1-->
| Konzept oder Komponente | Mathematische Form | Bedeutung für Anwendung und Implementierung |

<!--en:52257ad4-->
| **Diagonales System von ODEs** | $h_n'(t) = \lambda_n h_n(t) + x(t)$, $\lambda_n = -e^{\theta_n} < 0$ | $N$ unabhängige Raten pro Kanal, $O(dN)$ Speicherbedarf für den Zustand und ebenso viel Rechenaufwand pro Schritt |

<!--en:bf8440a8-->
| **Halteglied nullter Ordnung (ZOH)** | $\bar{a}_n = e^{\Delta \lambda_n}$, $\bar{b}_n = (e^{\Delta \lambda_n} - 1)/\lambda_n$ | Exakte diskrete Rekurrenz für Eingaben, die über jeden Schritt der Länge $\Delta$ konstant bleiben |

<!--en:e71670cb-->
| **Faktoren der Halbwertszeit** | $\tau_{1/2} = \ln 2 / (\Delta|\lambda_n|)$ | Raten $\lambda_n = -n$ unterscheiden die Zustände, $\Delta \in [10^{-3}, 10^{-1}]$ skaliert die Halbwertszeiten auf bis zu $693$ Schritte |

<!--en:3370971e-->
| **Dualität von Rekurrenz und Faltung** | $K_k = \sum_{n=1}^N c_n \bar{a}_n^k \bar{b}_n$, $y = \mathbf{K} * x + D x$ | FFT der Länge $2T$ mit Zero-Padding berechnet den LTI-Vorwärtsdurchlauf beim Training mit Aufwand $O(Td\log T)$ |

<!--en:2cfab0d0-->
| **Reelle und komplexe Raten** | $\lambda_n \in \mathbb{R}_-$ bzw. $\lambda_n = -\alpha + i\omega$ | Reelle Raten mitteln mit glatten Fenstern, komplexe Raten bilden scharfe verzögerte Spitzen nach |

<!--en:6c4ee562-->
| **Grenze der Zeitinvarianz** | Feste Gewichte $K_k$ pro Abstand, unabhängig von $x_{t-k}$ | Abruf mit festem Abstand von $1{,}000$ Schritten gelingt, bei wechselnden Zielpositionen scheitert er (§{sec_mamba}) |
