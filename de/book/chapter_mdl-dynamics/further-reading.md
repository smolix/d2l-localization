<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:e3a1d72f-->
In sieben Abschnitten haben wir den Weg von der Trajektorie eines einzelnen Punktes bis zu einem Fluss verfolgt, der Rauschen in Daten verwandelt. Dabei sind etliche Fragen offengeblieben. Welches stochastische Integral beschreibt das Rauschen in einem physikalischen System? Was wird aus der adjungierten Variablen, wenn das Modell selbst eine stochastische Differentialgleichung (SDE) ist? Jeder der folgenden Einträge greift eine solche Frage auf und zeigt, wo die Antwort zu finden ist.

<!--en:25736441-->
Die Praxis der Diffusionsmodelle behandeln wir in :numref:`chap_diffusion` und den dortigen Literaturhinweisen (:numref:`sec_diffusion-further-reading`). Bei Verfahren, die dort zur Sprache kommen, genügt daher ein Verweis. Was bleibt, ist Mathematik. Wir würden mit drei Einträgen beginnen. Exponentielle Integratoren erklären, warum ein Diffusionssampler mit wenigen Schritten auskommen kann. Wasserstein-Gradientenflüsse zeigen, welche Größe bei der Diffusion einer Dichte abnimmt. Schrödinger-Brücken sind das stochastische Gegenstück zu einer optimalen Paarung. Welche der übrigen Fragen von Interesse sind, hängt vom Leser ab. Wer nach :numref:`sec_mdl-flow-matching-transport` aufhört, hat alle Voraussetzungen für den Rest des Buches.

<!--en:86e5b87a-->
## Numerische Lösungsverfahren

<!--en:96f8e5b4-->
### Exponentielle Integratoren

<!--en:85c914c1-->
Die Wahrscheinlichkeitsfluss-ODE des VP-Prozesses mit gelerntem Score (:numref:`sec_mdl-score-based-generative-modeling`), $\dot{\mathbf x}=-\tfrac12\beta\,\mathbf x-\tfrac12\beta\,\mathbf s_{\boldsymbol\theta}(\mathbf x,t)$, ist semilinear. Sie besteht aus einem linearen Term, der in geschlossener Form bekannt ist, und einem gelernten Anteil. Ein Runge-Kutta-Schritt diskretisiert beide. Dadurch trägt der lineare Term zum Fehler bei und begrenzt über die Stabilitätsgrenze aus :numref:`sec_mdl-stiffness-implicit` die Schrittweite, wie glatt der gelernte Anteil auch sein mag. Diesen Aufwand können wir uns sparen, denn der lineare Teil lässt sich exakt lösen. Ein exponentieller Integrator behandelt den gelernten Anteil als Eingang einer angeregten linearen Gleichung (:numref:`subsec_mdl-ode-driven-linear`). Mit der Formel der Variation der Konstanten lässt sich die Entwicklung des Zustands unter dem linearen Term exakt berechnen. Nur das Integral des gelernten Anteils müssen wir annähern.

<!--en:a0ad20ba-->
Halten wir den gelernten Anteil über den ganzen Schritt konstant, erhalten wir das Halteglied nullter Ordnung aus jenem Unterabschnitt. Als Eingang dient die Ausgabe des Netzes. Bliebe die Ausgabe unverändert, wäre der Schritt exakt. Da sie sich ändert, ist er von erster Ordnung. Diesen Schritt haben wir im Kapitel bereits ausgeführt, ohne ihn so zu nennen. Die Division durch $\sqrt{\bar\alpha_t}$ in :numref:`sec_mdl-ddim-guidance` hat den linearen Term beseitigt, und DDIM hält die Rauschvorhersage über einen ganzen Schritt konstant. DDIM ist der exponentielle Integrator erster Ordnung.

<!--en:6ddb12f6-->
Die Sampler dieser Familie unterscheiden sich darin, welche Ausgabe sie konstant halten (den Score, die Rauschvorhersage oder die Vorhersage des unverrauschten Punktes) und welche Zeitskala sie dafür verwenden. Verfahren höherer Ordnung ersetzen die Konstante durch ein Polynom, das durch mehrere Vorhersagewerte verläuft. Einen Überblick über die Theorie geben :citet:`Hochbruck.Ostermann.2010`. Zu den darauf beruhenden Diffusionssamplern gehören DEIS :cite:`Zhang.Chen.2023` und DPM-Solver :cite:`Lu.Zhou.Bao.ea.2022`. Letzteren behandelt Aufgabe :exref:`ex_samplers-dpm-solver` in :numref:`sec_diffusion-samplers`. Seine Nachfolger sind in :numref:`sec_diffusion-further-reading` aufgeführt. Bei einer SDE führt dieselbe Idee zum exponentiellen Euler-Maruyama-Schritt aus Aufgabe :exref:`ex_mdl-sdes-exact-vp-discretization` in :numref:`sec_mdl-sdes`.

<!--en:c8d19ae2-->
### Geometrische Integration und Leapfrog-Verfahren

<!--en:a097bea8-->
Bei einem reibungsfreien Oszillator nimmt die Energie mit jedem expliziten Euler-Schritt zu und mit jedem impliziten Euler-Schritt ab. Über lange Zeit wächst die Schwingung im ersten Fall unbegrenzt, im zweiten kommt der Oszillator zur Ruhe. Ein Verfahren höherer Ordnung zögert das nur hinaus. Die *geometrische Integration* stellt eine andere Anforderung an ein Verfahren. Es soll eine Struktur des exakten Flusses erhalten, unabhängig davon, wie genau es dem Fluss folgt. Ein *symplektisches* Verfahren wie das symplektische Euler-Verfahren oder das Leapfrog-Verfahren erhält wie der exakte Fluss den Flächeninhalt in der Ebene aus Ort und Impuls. Bei mehr Koordinaten tritt an die Stelle der Fläche ein Volumen.

<!--en:3c680ab6-->
Die Rückwärtsfehleranalyse zeigt, dass ein solches Verfahren dem exakten Fluss zu einer leicht veränderten Energie folgt, bis auf einen Fehler, der exponentiell klein in $1/h$ ist. Der Fehler in der tatsächlichen Energie bleibt daher über sehr lange Zeiten beschränkt, statt mit dem Zeithorizont zu wachsen. In Aufgabe :exref:`ex_mdl-odes-solvers-symplectic-euler` in :numref:`sec_mdl-euler-runge-kutta` bestimmen wir die veränderte Energie für das symplektische Euler-Verfahren am Oszillator. Ihre Niveaulinien sind Ellipsen, die nahe bei den kreisförmigen Niveaulinien der tatsächlichen Energie liegen. Dieselbe Analyse gilt für den Gradientenabstieg, wenn der Verlust die Rolle der Energie übernimmt. Der Term, der zum Verlust hinzukommt, ist der implizite Regularisierungsterm für den Gradienten von :citet:`Barrett.Dherin.2021` (Aufgabe :exref:`ex_mdl-odes-solvers-implicit-gradient-regularization` im selben Abschnitt). Die Theorie entwickeln :citet:`Hairer.Lubich.Wanner.2006`.

<!--en:88e5e63d-->
Das Leapfrog-Verfahren dient auch als Integrator im Hamiltonian Monte Carlo :cite:`Duane.Kennedy.Pendleton.ea.1987,Neal.2011`. Dieser Sampler simuliert ein reibungsfreies Teilchen, dessen potentielle Energie die negative Log-Dichte der Zielverteilung ist. So schlägt er einen Sprung zu einem weit entfernten Punkt vor. Nach der Metropolis-Regel (:numref:`sec_mdl-bayes-mcmc`) nimmt er den Vorschlag an oder verwirft ihn. Eine Annahme ist möglich, weil der Energiefehler des Leapfrog-Verfahrens beschränkt bleibt. Da ein Leapfrog-Schritt umkehrbar ist und das Volumen erhält, muss die Regel nur zwei Energien vergleichen und benötigt keine Jacobi-Matrix.

<!--en:77bb31eb-->
## Modelle in kontinuierlicher Zeit

<!--en:1902d89b-->
### Flüsse mit leicht berechenbaren Determinanten

<!--en:0bb35516-->
Diskrete normalisierende Flüsse schränken jede Schicht so ein, dass sich die Determinante ihrer Jacobi-Matrix mit geringem Aufwand berechnen lässt. In :numref:`sec_mdl-continuous-normalizing-flows` haben wir die übliche Einschränkung genannt, eine Jacobi-Matrix in Dreiecksform. Wie kommt eine Schicht zu dieser Form? Eine Kopplungsschicht :cite:`dinh2014nice,dinh2017density` lässt die Hälfte der Koordinaten unverändert. Die andere Hälfte verschiebt und skaliert sie mithilfe von Funktionen der ersten Hälfte. Ihre Jacobi-Matrix hat Dreiecksform. Zum Umkehren der Schicht genügen eine Subtraktion und eine Division. Da jede Schicht nur die Hälfte des Zustands ändert, schaltet man viele Schichten hintereinander. Dabei tauschen die beiden Hälften von Schicht zu Schicht die Rollen.

<!--en:dbec08fc-->
Ein invertierbares Residualnetz :cite:`Behrmann.Grathwohl.Chen.ea.2019` setzt die Einschränkung an anderer Stelle an. Ein Residualblock entspricht einem Euler-Schritt (:numref:`sec_mdl-neural-odes`). Hat sein Residualzweig eine Lipschitz-Konstante kleiner als eins, ist der Schritt invertierbar, ebenso wie der Fluss eines Lipschitz-stetigen Feldes (:numref:`subsec_mdl-ode-flow-map`). Die Jacobi-Matrix behält ihre allgemeine Form. Den Logarithmus ihrer Determinante schätzt man mit der Potenzreihe aus Aufgabe :exref:`ex_mdl-odes-solvers-residual-logdet` in :numref:`sec_mdl-neural-odes-cnf`. Einen Überblick über die verschiedenen Familien und ihre jeweiligen Einschränkungen geben :citet:`Papamakarios.Nalisnick.Rezende.ea.2021`.

<!--en:7b17430b-->
### Reversible Lösungsverfahren und regularisierte Flüsse

<!--en:c4a80ee0-->
Die adjungierte Methode aus :numref:`sec_mdl-adjoint-method` spart Speicher, indem sie beim Rückwärtsrechnen die Zustände des Vorwärtslaufs rekonstruiert. Ein kontrahierender Fluss verstärkt dabei den Rundungsfehler. Bei der Rekonstruktion entsteht ein zweiter Fehler, auch ohne Kontraktion. Ein Rückwärtsschritt des Lösungsverfahrens kehrt einen Vorwärtsschritt nur näherungsweise um. Die rekonstruierten Zustände weichen daher von denen des Vorwärtslaufs ab. Somit entspricht der Gradient nicht dem Gradienten der tatsächlich ausgeführten Rechnung. Bei einem reversiblen Lösungsverfahren lässt sich jeder Schritt algebraisch rückgängig machen. Damit entfällt der zweite Fehler. Rundungsfehler werden weiterhin verstärkt. Bei einem steifen Feld bleibt Checkpointing daher die sicherere Wahl. Beide Ansätze vergleicht :citet:`Kidger.2022` (Kap. 5).

<!--en:6153f544-->
Beim Training anhand der Likelihood gibt es keinen Anreiz für einen Fluss, sich mit geringem Aufwand integrieren zu lassen. Für die Likelihood zählt nur, wo jeder Punkt ankommt. Der Weg dorthin spielt keine Rolle. Deshalb belegen :citet:`Finlay.Jacobsen.Nurbekyan.ea.2020` die kinetische Energie $\int\mathbb{E}\|\mathbf f_{\boldsymbol\theta}\|^2\,dt$ beim Training mit einem Strafterm. Zwischen zwei festen Endpunkten hat die geradlinige Bahn mit konstanter Geschwindigkeit die geringste kinetische Energie. Der Strafterm begradigt also die Bahnen, und für eine gerade Bahn braucht ein Lösungsverfahren nur wenige Schritte. Flow-Matching verfolgt dasselbe Ziel mit Regressionszielen entlang gerader Bahnen und durch die Paarung von Rauschen mit Daten (:numref:`sec_mdl-ot-connection`).

<!--en:b9a0c8c8-->
### Latente und gesteuerte Differentialgleichungen

<!--en:2a366507-->
Die Modelle dieses Kapitels bilden einen Anfangszustand auf einen Endzustand ab. Bei einer Zeitreihe wie den Messwerten eines Patienten treffen die Beobachtungen in unregelmäßigen Abständen ein. Ein Modell für solche Daten sollte jede Beobachtung erfassen, sobald sie eintrifft. Eine latente ODE :cite:`Rubanova.Chen.Duvenaud.2019` leistet das zum Teil. Sie kodiert die Beobachtungen in einem einzigen Anfangszustand und dekodiert die Trajektorie zu beliebig gewählten Zeitpunkten. Eine nachträglich eintreffende Beobachtung kann den Verlauf daher nicht mehr ändern.

<!--en:91cbfb6c-->
Bei einer neuronalen gesteuerten Differentialgleichung :cite:`Kidger.Morrill.Foster.ea.2020` wirkt jede Beobachtung auf den Zustand ein. Als Antrieb dient der interpolierte Datenpfad $\mathbf X_t$ selbst. Es gilt $d\mathbf z=\mathbf f_{\boldsymbol\theta}(\mathbf z)\,d\mathbf X_t$ mit einer matrixwertigen Funktion $\mathbf f_{\boldsymbol\theta}$. Die Gleichung hat die Form einer SDE, in der die Daten den brownschen Pfad ersetzen. Da der interpolierte Pfad glatt ist, genügt das gewöhnliche Integral. Ihr Euler-Schritt $\mathbf z_{n+1}=\mathbf z_n+\mathbf f_{\boldsymbol\theta}(\mathbf z_n)(\mathbf X_{n+1}-\mathbf X_n)$ entspricht einem rekurrenten Netz, das die Zuwächse der Daten einliest.

<!--en:5a42a7b9-->
### Transformer als System wechselwirkender Teilchen

<!--en:80d6cc3a-->
Ein Residualnetz bewegt einen einzelnen Zustand mit Euler-Schritten (:numref:`sec_mdl-neural-odes`). Ein Transformer bewegt alle Token einer Sequenz zugleich. Durch Attention hängt die Geschwindigkeit jedes Tokens davon ab, wo sich die anderen befinden. Die Token verhalten sich also wie Teilchen, die miteinander wechselwirken. :citet:`Geshkovski.Letrouit.Polyanskiy.ea.2023` untersuchen dieses System in einem idealisierten Modell, in dem die Schichtnormalisierung jedes Token auf der Einheitssphäre hält. Sie zeigen, dass sich die Token mit zunehmender Tiefe zu Clustern zusammenschließen. In einem Punkt unterscheidet sich das System von denen im Kapitel. Das Feld hängt von der Dichte ab, die es bewegt. Damit ist die Kontinuitätsgleichung nicht mehr linear in der Dichte.

<!--en:1842496b-->
## Rauschen

<!--en:30de2a42-->
### Wasserstein-Gradientenflüsse

<!--en:e36d2003-->
Bewegt sich ein Punkt entlang eines Gradientenflusses, nimmt sein Verlust ab (:numref:`sec_mdl-gd-as-solver`). Auch die Dichte eines Diffusionsprozesses entwickelt sich deterministisch gemäß der Fokker-Planck-Gleichung. Nimmt dabei ebenfalls eine Größe ab? Ja, wenn die Drift ein Gradientenfeld ist, $\mathbf f=-\nabla V$, und $g$ konstant ist. Dann bewegt sich die Dichte in Richtung abnehmender freier Energie $\mathcal F(p)=\int Vp\,d\mathbf x+\tfrac{g^2}2\int p\log p\,d\mathbf x$. Der erste Term ist die mittlere potentielle Energie. Sie ist am kleinsten, wenn die gesamte Masse im Minimum von $V$ liegt. Der zweite Term ist die negative Entropie. Sie ist am kleinsten, wenn sich die Masse möglichst weit verteilt. Die Drift verringert den ersten Term, das Rauschen den zweiten. Bei der stationären Dichte halten sich beide Effekte die Waage. Diese Dichte ist proportional zu $e^{-2V/g^2}$.

<!--en:ae635dea-->
Die Richtung des steilsten Abstiegs hängt davon ab, wie wir den Abstand messen. Hier müssen wir den Wasserstein-2-Abstand zwischen zwei Dichten verwenden. Er misst die Kosten des Transports der einen Dichte in die andere. Am impliziten Euler-Schritt aus :numref:`sec_mdl-gd-as-solver` sehen wir, wo der Abstand eingeht. Bei einem konvexen Verlust minimiert der Schritt $\mathcal L(\mathbf x)+\tfrac1{2\eta}\|\mathbf x-\mathbf x_k\|^2$. Das JKO-Schema :cite:`Jordan.Kinderlehrer.Otto.1998` führt denselben Schritt aus, ersetzt aber die Punkte durch Dichten und das Abstandsquadrat durch die Transportkosten. Damit ergibt sich $p_{k+1}=\arg\min_p\,\mathcal F(p)+\tfrac1{2\eta}W_2^2(p,p_k)$. Für $\eta\to0$ konvergieren die Schritte gegen die Lösung der Fokker-Planck-Gleichung.

<!--en:b007d2c8-->
Besonders einprägsam ist die Langevin-Dynamik. Ihre Drift ist der halbe Score der Zielverteilung, und für ihr Rauschen gilt $g=1$ (:numref:`sec_mdl-langevin-sampling`). Bezeichnen wir die Zieldichte mit $p_\infty$, damit $p$ weiterhin für die Dichte der Teilchenwolke steht. Dann gilt $V=-\tfrac12\log p_\infty$, und die freie Energie beträgt $\tfrac12D_{\textrm{KL}}(p\,\|\,p_\infty)$. Eine Wolke von Langevin-Teilchen führt also einen Gradientenabstieg auf der KL-Divergenz zur Zielverteilung aus. Dabei misst sich der Abstand an den Transportkosten. Deshalb begegnet uns der Transport in diesem Kapitel immer wieder, von der Kontinuitätsgleichung bis zur Formulierung von Benamou und Brenier :cite:`Benamou.Brenier.2000`.

<!--en:00509409-->
### Physikalisches Rauschen und der Satz von Wong und Zakai

<!--en:ce3236a7-->
In :numref:`sec_mdl-ito-integral` haben wir das Itô-Integral anhand des linken Intervallrands definiert. Die dort erwähnte Mittelpunktregel definiert das Stratonovich-Integral und erhält die gewöhnliche Kettenregel. Welches der beiden Integrale beschreibt ein physikalisches System? Das Rauschen in einem physikalischen System hat keinen brownschen Pfad. Es hat ein kurzes Gedächtnis und einen glatten Pfad. Für einen glatten Pfad gibt es nur ein Integral, das gewöhnliche. :citet:`Wong.Zakai.1965` haben gezeigt, dass die Lösungen der gewöhnlichen Differentialgleichungen gegen die Stratonovich-Lösung konvergieren, wenn sich solche Pfade einem brownschen Pfad annähern. In einer Dimension entspricht die Stratonovich-Interpretation von $dX=f\,dt+g\,dW$ der Itô-Gleichung mit der Drift $f+\tfrac12g\,\partial_xg$.

<!--en:c76997b9-->
Beim additiven Rauschen von Diffusionsmodellen stimmen beide Interpretationen überein. Bei multiplikativem Rauschen unterscheiden sie sich, und der Satz hilft bei der Wahl. Eine Gleichung, die glattes physikalisches Rauschen idealisiert, ist im Sinne von Stratonovich zu lesen. Bei einem zeitdiskreten Modell, dessen Koeffizient den Zuwachs noch nicht kennt, mit dem er multipliziert wird, gilt die Itô-Interpretation. Ein Schritt des stochastischen Gradientenabstiegs ist von dieser Art.

<!--en:5ec6a72e-->
### Stochastischer Gradientenabstieg als SDE

<!--en:980ad8bf-->
Der Gradientenabstieg entspricht dem expliziten Euler-Verfahren für den Gradientenfluss (:numref:`sec_mdl-gd-as-solver`). Welche Gleichung diskretisiert der stochastische Gradientenabstieg? Ein Minibatch-Gradient ist der vollständige Gradient zuzüglich eines Rauschens, dessen Kovarianz $\boldsymbol\Sigma$ mit wachsender Batchgröße wie $1/B$ abnimmt, wobei $B$ die Batchgröße bezeichnet. :citet:`Li.Tai.E.2017` schlagen die SDE $d\boldsymbol\theta=-\nabla\mathcal L\,dt+\sqrt{\eta}\,\boldsymbol\Sigma^{1/2}d\mathbf W$ vor. Führen wir einen Euler-Maruyama-Schritt der Länge $\eta$ aus. Dabei erhält der Rauschkoeffizient einen weiteren Faktor $\sqrt\eta$. Das ergibt einen Schritt des stochastischen Gradientenabstiegs mit Lernrate $\eta$.

<!--en:8af7bb48-->
Die Lernrate tritt also zweimal auf, als Schrittweite und im Rauschterm. Ist $\boldsymbol\Sigma$ überall dasselbe Vielfache $c\,\mathbf I$ der Einheitsmatrix, so gilt die oben angegebene freie Energie mit $V=\mathcal L$ und $g^2=\eta c$. Letztere Größe gewichtet die Entropie gegenüber dem Verlust. In diesem Sinne ist die Lernrate eine Temperatur. Bei quadratischem Verlust stellt sich für die Iterierten die stationäre Gauß-Verteilung eines Ornstein-Uhlenbeck-Prozesses ein (:numref:`sec_mdl-ornstein-uhlenbeck`). Ihre Varianz wächst mit dem Verhältnis $\eta/B$. :citet:`Mandt.Hoffman.Blei.2017` nutzen dies, um mit dem stochastischen Gradientenabstieg näherungsweise aus der A-posteriori-Verteilung zu ziehen. Das Modell setzt kleine Lernraten und annähernd gaußsches Gradientenrauschen voraus. Beide Annahmen können in der Praxis verletzt sein.

<!--en:a5f93a21-->
### Neuronale SDEs

<!--en:e056f93e-->
Was wird aus der adjungierten Variablen, wenn das gelernte Modell selbst eine SDE mit gelernter Drift und gelernter Rauschamplitude $g$ ist? Sie erfüllt eine Rückwärts-SDE, die derselbe brownsche Pfad antreibt. Diesen Pfad müssen wir speichern oder aus seinem Zufallsstartwert erneut erzeugen :cite:`Li.Wong.Chen.ea.2020`. Auch der Verlust ändert sich. Die Ausgabe eines solchen Modells ist zufällig, sodass wir keinen einzelnen Endpunkt mit einem einzelnen Zielwert vergleichen können. Als Trainingssignal dient eine Divergenz zwischen den Verteilungen ganzer Pfade oder ein Diskriminator, der diese Verteilungen vergleicht :cite:`Kidger.2022`.

<!--en:b376600f-->
Der Satz von Girsanov :cite:`Oksendal.2003` liefert diese Divergenz in geschlossener Form für zwei Prozesse mit demselben $g$ und den Driften $\mathbf f$ und $\mathbf f'$. Die KL-Divergenz zwischen ihren Pfadverteilungen ist die KL-Divergenz zwischen ihren Anfangsverteilungen zuzüglich $\tfrac12\int\mathbb{E}\|\mathbf f-\mathbf f'\|^2/g^2\,dt$. Der Erwartungswert bezieht sich auf die Pfade des ersten Prozesses. Die Bedingung an $g$ ist wesentlich. Zwei Prozesse mit verschiedenem $g$ lassen sich schon an der quadratischen Variation eines einzigen Pfades unterscheiden (:numref:`sec_mdl-quadratic-variation`). Die KL-Divergenz zwischen ihren Pfadverteilungen ist dann unendlich.

<!--en:68bbb405-->
Die Verteilungen der Endpunkte können sich höchstens so stark unterscheiden wie die Pfadverteilungen. Das folgt aus der Datenverarbeitungsungleichung in :numref:`subsec_mdl-f-div-data-processing`. Wenden wir die Formel von Girsanov auf die exakte und die gelernte Rückwärtsdrift eines Diffusionsmodells an, erhalten wir daher eine Schranke für die Likelihood. Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-likelihood-bound` in :numref:`sec_mdl-score-matching-diffusion-flow` leitet dieselbe Schranke ohne Pfadverteilungen her.

<!--en:88b03294-->
## Gelernte Generatoren

<!--en:08b536d0-->
Die meisten Themen, die in dieser Richtung an das Kapitel anschließen, behandelt :numref:`chap_diffusion`. Vorhersageziele, Verlustgewichtungen und Rauschpläne finden sich in :numref:`sec_diffusion-design-space`. Dort ordnet der Logarithmus des Signal-Rausch-Verhältnisses, also $-2\log\bar\sigma_t$ in der Schreibweise von :numref:`sec_mdl-ddim-guidance`, die verschiedenen Möglichkeiten :cite:`Kingma.Salimans.Poole.ea.2021`. Die Erzeugung in einem oder wenigen Schritten behandelt :numref:`sec_diffusion-few-step`. Vier Themen gehen über beide Kapitel hinaus.

<!--en:941c0c9a-->
### Sliced-Score-Matching

<!--en:b796e83f-->
In :numref:`sec_mdl-score-matching` haben wir das implizite Score-Matching zurückgestellt, weil die Berechnung der Spur zu aufwendig war. Einen Ausweg haben wir erwähnt, aber nicht weiterverfolgt. Der Schätzer von Hutchinson benötigt pro Zufallsrichtung nur ein Vektor-Jacobi-Produkt. Trainiert man mit diesem Schätzer, heißt das Verfahren Sliced-Score-Matching :cite:`Song.Garg.Shi.ea.2019`. Es lernt den Score der Daten selbst, ohne dass man einen Rauschpegel wählen muss. Einem Diffusionsmodell bringt das keinen Vorteil. Es braucht die Scores der verrauschten Daten für jeden Rauschpegel. Denoising-Score-Matching liefert sie, ohne eine Spur berechnen zu müssen. Deshalb hat sich die Entrauschung als Standard durchgesetzt.

<!--en:dd71487c-->
### Fehler im Score und beim Sampling

<!--en:c5634f2d-->
Fast jeder Sampler in diesem Kapitel verwendete einen exakten Score. Ein trainierter Score ist hingegen nirgends exakt. Wie gut muss er sein? Es genügt, dass er im Mittel gut ist. :citet:`Chen.Chewi.Li.ea.2023` beschränken den Totalvariationsabstand zwischen der Datenverteilung und der Ausgabeverteilung der diskretisierten Rückwärts-SDE durch eine Summe aus drei Termen. Jeder davon ist uns im Kapitel bereits einzeln begegnet. Der Abstand zwischen $p_T$ und der Gauß-Verteilung, bei der der Sampler startet, fällt wie $e^{-T}$ ab. Der Diskretisierungsfehler wächst mit der Dimension und sinkt mit abnehmender Schrittweite. Beides haben wir in :numref:`sec_mdl-time-reversal` gemessen. Der dritte Term ist die Wurzel des mittleren quadratischen Fehlers des Scores unter der Verteilung der verrauschten Daten. Diesen quadratischen Fehler misst der Entrauschungsverlust oberhalb seines Sockelwerts. Laut dem Satz geht keine weitere Größe ein. Die Voraussetzungen sind schwach. Die Daten müssen ein endliches zweites Moment haben, und die Scores der verrauschten Dichten müssen Lipschitz-stetig sein. Die Verteilung darf mehrere Modi haben. Der Satz erfasst also auch die beiden Modi, an denen die Langevin-Dynamik in :numref:`sec_mdl-langevin-sampling` scheiterte.

<!--en:77190567-->
### Schrödinger-Brücken und entropischer Transport

<!--en:14cbdb7b-->
1931 stellte Schrödinger eine Frage über Teilchen. Wir beobachten eine frei diffundierende Wolke zunächst in einer bestimmten Form und später in einer anderen, unwahrscheinlichen Form. Auf welchem Weg ist sie am wahrscheinlichsten dorthin gelangt? Bei einer großen Wolke unabhängiger Teilchen fällt die Wahrscheinlichkeit exponentiell ab, dass die Wolke einer anderen Pfadverteilung folgt. Die Rate ist durch die KL-Divergenz dieser Verteilung gegenüber der freien Diffusion gegeben. Gesucht ist also unter allen Pfadverteilungen mit den beiden vorgegebenen Randverteilungen diejenige mit der kleinsten KL-Divergenz gegenüber der freien Diffusion. Sie heißt *Schrödinger-Brücke*.

<!--en:636411c2-->
Vergleichen wir die Brücke mit der Paarung innerhalb eines Minibatches aus :numref:`sec_mdl-ot-connection`. Diese Paarung liefert für jeden Batch eine exakte Zuordnung. Der damit trainierte Generator ist eine ODE. Die Brücke paart ihre Endpunkte gemäß dem Plan des entropieregularisierten optimalen Transports. Er lässt sich mit den Sinkhorn-Iterationen aus :numref:`subsec_mdl-sinkhorn` berechnen :cite:`Cuturi.2013,Peyre.Cuturi.2019`. Der Generator der Brücke ist eine SDE. Nimmt das Rauschen der freien Diffusion ab, nähert sich der Plan einem optimalen Transportplan.

<!--en:309ef511-->
Im Vergleich zu einem Diffusionsmodell stellt die Brücke das schwierigere Problem dar. Bei einem Diffusionsmodell steht der Vorwärtsprozess fest. Das Modell lernt nur den Rückwärtsprozess und erreicht seine A-priori-Verteilung nur näherungsweise. Die Brücke verbindet zwei vorgegebene Verteilungen in endlicher Zeit. Daher müssen beide Richtungen gelernt werden. :citet:`DeBortoli.Thornton.Heng.ea.2021` passen abwechselnd eine Vorwärts- und eine Rückwärtsdiffusion an. Die erste Runde dieses Verfahrens entspricht dem Training eines Diffusionsmodells. :citet:`Albergo.Boffi.VandenEijnden.2023` gelangen zu beiden Arten von Generatoren, ODE und SDE, ohne ein Brückenproblem zu lösen. Ziehen wir je einen Wert $\mathbf x_0$ und $\mathbf x_1$ aus zwei beliebigen Verteilungen sowie einen unabhängigen gaußschen Zufallsvektor $\mathbf w$. Der Pfad $a_t\mathbf x_0+b_t\mathbf x_1+c_t\mathbf w$, bei dem $c_t$ an beiden Enden null ist, besitzt Randverteilungen, die sich mit einer ODE oder einer Familie von SDEs erzeugen lassen (Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-stochastic-interpolants` in :numref:`sec_mdl-flow-matching-transport`).

<!--en:ea666aec-->
### Geradlinige und optimale Kopplungen

<!--en:c1e1db73-->
Bei einer Paarung, die Reflow unverändert lässt, sind die Pfade gerade und schneiden sich nicht. In mehr als einer Dimension muss eine solche Paarung nicht optimal sein. Das Beispiel in :numref:`sec_mdl-ot-connection` paart jeden Punkt einer gaußschen Wolke mit seinem Bild unter einer Vierteldrehung. Die Strecken schneiden sich nirgends. Ihre Kosten betragen $4$, obwohl $W_2^2=0$ gilt. Was diese Paarung von einer optimalen unterscheidet, ist die Rotation. Das Geschwindigkeitsfeld eines optimalen Flusses ist ein Gradientenfeld :cite:`Benamou.Brenier.2000`. Ein Feld, das eine Wolke um den Ursprung dreht, hat diese Eigenschaft nicht.

<!--en:9c6d535b-->
:citet:`Liu.2022` beschränkt das Geschwindigkeitsfeld daher auf ein Gradientenfeld $\nabla\phi_t$ und argumentiert, dass genau die optimalen Paarungen unter diesem eingeschränkten Reflow unverändert bleiben. Die Aussage erfordert stärkere Voraussetzungen. :citet:`Hertrich.Chambolle.Delon.2025` beweisen sie für den Fall, dass die Wolke während des Transports zu jedem Zeitpunkt den gesamten Raum ausfüllt. Sie geben Gegenbeispiele an, in denen die Wolke aus getrennten Teilen besteht. Die Einschränkung beantwortet also noch nicht die Frage aus :numref:`sec_mdl-ot-connection`, wie sich allein durch Training eine optimale Paarung finden lässt.

<!--en:77fc72da-->
## Materialien zu dynamischen Systemen und generativen Flüssen

<!--en:d87a6a53-->
Ein Hinweis zur Zeitrichtung vor der Lektüre: Im Skript und im Kurs des MIT, im Flow Matching Guide und in den beiden Arbeiten zum Flow-Matching läuft die Zeit vom Rauschen bei $0$ zu den Daten bei $1$. In diesem Kapitel läuft sie in die umgekehrte Richtung (:numref:`fig_mdl-dyn-time-conventions`).

<!--en:cd0bff15-->
**Bücher und Vorlesungsskripte**

<!--en:89d51682-->
- [Nonlinear Dynamics and Chaos – Steven Strogatz](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering) gibt eine geometrische Einführung in Flüsse, Fixpunkte und deren Stabilität. Das sind die Themen von :numref:`sec_mdl-odes-solvers`.

<!--en:a70411fb-->
- [On Neural Differential Equations – Patrick Kidger (2022)](https://arxiv.org/abs/2202.02435) behandelt auf Buchlänge Neural ODEs sowie gesteuerte und stochastische Differentialgleichungen. Das Buch entwickelt die adjungierte Methode aus :numref:`sec_mdl-neural-odes-cnf` in allen Einzelheiten und gibt Hinweise zur Wahl unter den Lösungsverfahren aus :numref:`sec_mdl-euler-runge-kutta`.

<!--en:5157c74d-->
- [Applied Stochastic Differential Equations – Särkkä & Solin](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf) behandelt den Itô-Kalkül, stochastische Differentialgleichungen und die Fokker-Planck-Gleichung mit Blick auf Anwendungen. Das Buch ergänzt :numref:`sec_mdl-sdes` und :numref:`sec_mdl-fokker-planck-probability-flow`. Die Autoren stellen es vollständig als PDF bereit.

<!--en:bdd1055b-->
- [Stochastic Differential Equations: An Introduction with Applications – Bernt Øksendal](https://link.springer.com/book/10.1007/978-3-642-14394-6) ist ein mathematisch strenges Nachschlagewerk zur brownschen Bewegung, zum Itô-Integral und zur Existenztheorie, auf denen :numref:`sec_mdl-sdes` aufbaut.

<!--en:cc2b7925-->
- [Numerical Solution of Stochastic Differential Equations – Kloeden & Platen](https://link.springer.com/book/10.1007/978-3-662-12616-5) ist das Standardwerk zum Euler-Maruyama-Verfahren, zum Milstein-Verfahren und zu Verfahren höherer Ordnung sowie zu ihren starken und schwachen Konvergenzordnungen. Es ergänzt den Teil über Simulation in :numref:`sec_mdl-sdes`.

<!--en:21d945a4-->
- [An Introduction to Flow Matching and Diffusion Models (Vorlesungsskript zum MIT-Kurs 6.S184) – Holderrieth & Erives](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) leitet Diffusion und Flow-Matching aus stochastischen Differentialgleichungen und der Kontinuitätsgleichung her. Das in sich geschlossene Skript ergänzt :numref:`sec_mdl-score-matching-diffusion-flow` und :numref:`sec_mdl-flow-matching-transport`.

<!--en:e59c5bd6-->
**Kurse und Tutorials**

<!--en:296e4fea-->
- [MIT 6.S184: Generative AI with Stochastic Differential Equations](https://diffusion.csail.mit.edu/) entwickelt in Vorlesungen, Folien und praktischen Übungen ein Diffusionsmodell von Grund auf. Der Kurs folgt dem Weg von :numref:`sec_mdl-fokker-planck-probability-flow` bis :numref:`sec_mdl-flow-matching-transport` und steht unter einer offenen Lizenz.

<!--en:d927865f-->
- [Understanding Diffusion Models: A Unified Perspective – Calvin Luo (2022)](https://arxiv.org/abs/2208.11970) leitet den variationellen und den Score-basierten Zugang zur Diffusion nebeneinander her und ergänzt damit :numref:`sec_mdl-score-matching-diffusion-flow`.

<!--en:594ec7b6-->
- [Flow Matching Guide and Code – Lipman et al. (2024)](https://arxiv.org/abs/2412.06264) ist ein ausführliches Nachschlagewerk zum Flow-Matching für :numref:`sec_mdl-flow-matching-transport`. Dazu gehört die PyTorch-Bibliothek [`facebookresearch/flow_matching`](https://github.com/facebookresearch/flow_matching).

<!--en:d63f4e40-->
**Grundlegende Forschungsarbeiten**

<!--en:c066a683-->
- [Neural Ordinary Differential Equations – Chen et al. (NeurIPS 2018)](https://arxiv.org/abs/1806.07366) parametrisiert die Dynamik eines Netzes als gewöhnliche Differentialgleichung und trainiert das Netz mit der adjungierten Methode wie in :numref:`sec_mdl-neural-odes-cnf`.

<!--en:813c7ef8-->
- [Generative Modeling by Estimating Gradients of the Data Distribution – Song & Ermon (NeurIPS 2019)](https://arxiv.org/abs/1907.05600) beschreibt die Score-basierte Erzeugung mit Denoising-Score-Matching und Langevin-Sampling bei schrittweise sinkendem Rauschpegel. Die Arbeit ergänzt :numref:`sec_mdl-score-matching-diffusion-flow`.

<!--en:2265ceb8-->
- [Denoising Diffusion Probabilistic Models – Ho, Jain & Abbeel (NeurIPS 2020)](https://arxiv.org/abs/2006.11239) stellt die zeitdiskrete DDPM-Formulierung vor, die wir in :numref:`sec_mdl-score-matching-diffusion-flow` als diskretisierte stochastische Differentialgleichung interpretieren.

<!--en:cdc87fc5-->
- [Score-Based Generative Modeling through SDEs – Song et al. (ICLR 2021)](https://arxiv.org/abs/2011.13456) führt Score-Matching und Diffusion über Vorwärts- und Rückwärts-SDEs zusammen und stellt die Wahrscheinlichkeitsfluss-ODE aus :numref:`sec_mdl-fokker-planck-probability-flow` vor.

<!--en:30f50e8f-->
- [Denoising Diffusion Implicit Models – Song, Meng & Ermon (ICLR 2021)](https://arxiv.org/abs/2010.02502) beschreibt den deterministischen Sampler aus :numref:`sec_mdl-score-matching-diffusion-flow`, der ein bereits trainiertes DDPM mit weniger Schritten nutzt.

<!--en:4f139db9-->
- [Flow Matching for Generative Modeling – Lipman et al. (ICLR 2023)](https://arxiv.org/abs/2210.02747) beschreibt das simulationsfreie Training kontinuierlicher Flüsse durch Regression auf bedingte Vektorfelder. Dieses Verfahren behandeln wir in :numref:`sec_mdl-flow-matching-transport`.

<!--en:b2ee27b6-->
- [Flow Straight and Fast: Rectified Flow – Liu, Gong & Liu (ICLR 2023)](https://arxiv.org/abs/2209.03003) stellt geradlinige Interpolationspfade und Reflow für das Sampling in wenigen Schritten vor und ergänzt damit :numref:`sec_mdl-flow-matching-transport`.

<!--en:959bee44-->
**Blogbeiträge und anschauliche Erläuterungen**

<!--en:47132f47-->
- [Generative Modeling by Estimating Gradients of the Data Distribution – Yang Song](https://yang-song.net/blog/2021/score/) zeigt anhand von Abbildungen, wie Score-Matching, Langevin-Dynamik und die Beschreibung durch stochastische Differentialgleichungen aus :numref:`sec_mdl-score-matching-diffusion-flow` zusammenhängen. Der Autor führt Schritt für Schritt durch diese Zusammenhänge.

<!--en:15e13d3e-->
- [What are Diffusion Models? – Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) gibt einen bis 2024 aktualisierten Überblick über Diffusionsmodelle, vom DDPM über Score-SDEs bis zur Steuerung der Erzeugung. Der Beitrag ergänzt :numref:`sec_mdl-score-matching-diffusion-flow`.

<!--en:de445ead-->
- [Perspectives on Diffusion – Sander Dieleman](https://sander.ai/2023/07/20/perspectives.html) zeigt, wie ein Diffusionsmodell zugleich als Entrauscher, Score-Schätzer, Lösungsverfahren für eine Rückwärts-SDE und Fluss aufgefasst werden kann. Genau diese Entsprechungen behandelt :numref:`sec_mdl-score-velocity-dictionary`.

<!--en:b5b5d4a7-->
- [The Annotated Diffusion Model – Hugging Face](https://huggingface.co/blog/annotated-diffusion) implementiert DDPM Zeile für Zeile in PyTorch und verbindet so die Mathematik aus :numref:`sec_mdl-score-matching-diffusion-flow` mit ausführbarem Code.

<!--en:48a210aa-->
[Diskussion](https://d2l.discourse.group/t/36870)
