<!--en:4c69064d-->
# Generative adversariale Netze

<!--en:46798c9f-->
:citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` führten *generative adversariale Netze* (engl. generative adversarial networks, GANs) ein. Sie kombinierten ein Generatornetz mit einem binären Klassifikator, dem *Diskriminator* oder *Kritiker*. Er lernt, echte Daten von generierten Samples zu unterscheiden, ähnlich wie der Domänenklassifikator zur Korrektur der Kovariatenverschiebung in :numref:`subsec_covariate-shift-correction`.

<!--en:7c902b45-->
Daraus ergeben sich unmittelbar zwei grundlegende Fragen:

<!--en:fcc1f072-->
1. Wenn ein Generator den Diskriminator täuscht, hat er dann tatsächlich gelernt, die Datenverteilung $p$ nachzubilden?

<!--en:7f3a4c87-->
2. Wenn der Diskriminator generierte Samples mühelos von echten Daten unterscheiden kann, weist sein Gradient dem Generator dann noch den Weg zu $p$?

<!--en:e455afc8-->
Bei festem Generator lässt sich der optimale Diskriminator in geschlossener Form herleiten. Deshalb können wir beide Fragen analytisch beantworten. Wir überprüfen unsere Analyse zunächst an einem zweidimensionalen Beispiel mit Gauß-Verteilungen, für das die exakten Dichten und KL-Divergenzen in geschlossener Form vorliegen. Anschließend trainieren wir ein mehrschichtiges Perzeptron als Generator mit Bildern aus Fashion-MNIST.

<!--en:6d75088a-->
## Von der Likelihood zum Vergleich zweier Stichproben

<!--en:de1afe1e-->
### Implizite Generatoren

<!--en:b2409eaa-->
Für die Maximum-Likelihood-Schätzung müssen wir die Log-Dichte $\log p_\theta(x)$ des Modells an jeder Beobachtung auswerten. Die Generatoren in diesem Kapitel sind hingegen *implizit*. Sie ziehen einen latenten Vektor $z \sim \mathcal{N}(0, I_{d_z})$ und leiten ihn durch ein deterministisches Netz $x' = G(z)$. Ist die latente Dimension $d_z$ kleiner als die Datendimension $d_x$, liegen die generierten Samples auf einer niedrigdimensionalen Mannigfaltigkeit. Sie besitzen dann keine Dichte bezüglich des Lebesgue-Maßes auf $\mathbb{R}^{d_x}$. Selbst wenn $d_z = d_x$ gilt, setzt die Berechnung der induzierten Dichte mit der Formel für die Variablentransformation voraus, dass $G$ invertierbar und seine Jacobi-Determinante effizient berechenbar ist.

<!--en:0b2e14b3-->
Zudem garantiert eine hohe Log-Likelihood auf den Testdaten keine realistischen Samples :cite:`Theis.Oord.Bethge.2016`. Betrachten wir eine Mischung der wahren Datenverteilung mit gleichverteiltem oder gaußschem Rauschen:

<!--en:f4b7aae5-->
Da überall $\tilde p(x) \geq 0.01\, p_{\textrm{data}}(x)$ gilt, liegt die erwartete Log-Likelihood der Mischung höchstens $\log 100 \approx 4.6$ Nat unter derjenigen der wahren Datenverteilung. Bei Bildern ist dieser Unterschied vernachlässigbar, denn ihre Log-Likelihoods erstrecken sich über Tausende von Nat. Dennoch bestehen 99% der aus $\tilde p$ gezogenen Samples aus reinem Rauschen. Um einen Generator zu beurteilen, müssen wir seine Samples also direkt betrachten.

<!--en:26916f90-->
### Unterscheidung zwischen echten und generierten Beispielen

<!--en:3fda64e0-->
Ein *Zweistichprobentest* (engl. two-sample test) prüft, ob zwei Stichproben aus derselben Verteilung stammen. Dazu bietet es sich an, einen binären Klassifikator zu trainieren, der sie unterscheidet. In einem GAN liefert dieser Klassifikator dem Generator Gradienten. Sei $y \sim \textrm{Bernoulli}(\tfrac12)$ ein Herkunftslabel mit zwei gleich wahrscheinlichen Werten. Für $y = 1$ ziehen wir $x \sim p$ aus der Datenverteilung, für $y = 0$ ziehen wir $x \sim q$ aus der Generatorverteilung. Die Randverteilung von $x$ ist die gleichgewichtete Mischung $m = (p + q)/2$.

<!--en:42b4cb5e-->
Der *Kritiker* (oder *Diskriminator*) ist eine Funktion $D: \mathcal{X} \to \mathbb{R}$. Seine Ausgabe ist ein *Logit für die Echtheit* (engl. realness logit), sodass $\sigma(D(x)) = 1/(1 + e^{-D(x)})$ die A-posteriori-Wahrscheinlichkeit $P(y = 1 \mid x)$ schätzt. Wie in :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` verwenden wir im Code `net_D` und `update_D` für das Diskriminatornetz. Im Text verwenden wir auch den Begriff *Kritiker*, der später im Kapitel auch Netze umfasst, deren Scores keine probabilistische Interpretation haben. (Ein GAN-Kritiker unterscheidet sich vom Kritiker, der beim Akteur-Kritiker-Verfahren im Reinforcement Learning die Wertfunktion schätzt, siehe :numref:`sec_actorcritic`.)

<!--en:ccf16978-->
Wie :numref:`fig_gan` zeigt, trainieren wir die beiden Netze abwechselnd. Wir aktualisieren den Diskriminator so, dass er Minibatches aus echten Daten von generierten Minibatches unterscheidet. Für das Update des Generators propagieren wir Gradienten durch den Diskriminator zurück. So lernt der Generator, Samples zu erzeugen, die schwerer als künstlich zu erkennen sind.

<!--en:42fe6a92-->
![Adversariales Training. Der Generator $G$ bildet latente Zufallsvektoren $z$ auf generierte Samples ab. Der Kritiker $D$ bewertet sowohl generierte als auch echte Beispiele. Er maximiert seinen Score auf echten Daten und minimiert ihn auf generierten Samples. Der Generator nutzt die Gradienten durch $D$, um die beiden Verteilungen ununterscheidbar zu machen.](../img/mdl-gan-architecture.svg)

<!--en:625509f0-->
## Der logarithmische Verlust als Zielfunktion

<!--en:67148fec-->
Trainieren wir den Diskriminator mit binärer Kreuzentropie (logarithmischem Verlust) auf gleich vielen echten und generierten Beispielen, ergibt sich die Minimax-Wertfunktion von :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`:

<!--en:2afd792c-->
Der Diskriminator maximiert $V(D)$. Der Generator beeinflusst nur den zweiten Term und versucht, $V(D)$ zu minimieren. Daraus ergibt sich das Minimax-Problem $\min_G \max_D V(D)$.

<!--en:dbfd9dad-->
### Der optimale Kritiker

<!--en:313ef184-->
Bei fester Generatorverteilung $q$ lässt sich :eqref:`eq_gan_V` für jedes $x$ getrennt maximieren. An jedem Punkt $x$ lautet der Integrand $p(x) \log \sigma(D(x)) + q(x) \log \sigma(-D(x))$. Setzen wir $u = \sigma(D(x)) \in (0, 1)$, so können wir die streng konkave Funktion $p \log u + q \log(1-u)$ mithilfe der Bedingung erster Ordnung $p/u = q/(1-u)$ maximieren. Wir erhalten das eindeutige Optimum:

<!--en:528a3953-->
Der optimale Diskriminator liefert also die A-posteriori-Wahrscheinlichkeit nach Bayes $P(y = 1 \mid x)$. Sein Logit $D^\star(x)$ entspricht dem *logarithmierten Dichteverhältnis* (engl. log density ratio) $\lambda(x) = \log \rho(x)$, wobei $\rho(x) = p(x)/q(x)$ gilt. Obwohl für $q(x)$ keine Dichte in geschlossener Form vorliegt, lässt sich $\log(p/q)$ durch binäre Klassifikation direkt aus Stichproben schätzen. Der logarithmische Verlust legt außerdem die additive Konstante von $D^\star$ eindeutig fest. Verschieben wir $D^\star$ um eine beliebige Konstante $b$ ungleich null, so sinkt $V$ strikt (Aufgabe :exref:`ex_gan-log-loss-game-identifies`). Somit gilt $D^\star(x) > 0$ genau dort, wo $p(x) > q(x)$ gilt.

<!--en:92e3c159-->
Für zwei Gauß-Verteilungen $p = \mathcal{N}(0, 1)$ und $q = \mathcal{N}(\mu, 1)$ mit Einheitsvarianz ist das optimale Logit beispielsweise eine affine Funktion von $x$:

<!--en:607579a0-->
In der Mitte zwischen den beiden Mittelwerten, bei $\mu/2$, wechselt es das Vorzeichen. Haben $p$ und $q$ unterschiedliche Kovarianzmatrizen, so ist $\lambda(x)$ quadratisch in $x$. Dafür brauchen wir einen nichtlinearen Diskriminator.

<!--en:96d3c188-->
### Zusammenhang mit der Jensen-Shannon-Divergenz

<!--en:26a94c3f-->
Setzen wir für den optimalen Diskriminator $\sigma(D^\star) = p/(2m)$ und $\sigma(-D^\star) = q/(2m)$ wieder in :eqref:`eq_gan_V` ein, sehen wir, was der Generator unter einem optimalen Diskriminator minimiert:

<!--en:0513edb9-->
Mit der Definition der Jensen-Shannon-Divergenz, $\mathrm{JS}(p, q) = \tfrac12 D_{\textrm{KL}}(p \,\|\, m) + \tfrac12 D_{\textrm{KL}}(q \,\|\, m)$, erhalten wir:

<!--en:071e6583-->
Es gilt $\mathrm{JS}(p, q) \geq 0$ mit Gleichheit genau dann, wenn $q = p$ gilt. Durch Minimieren von $\max_D V(D)$ erhalten wir daher als eindeutige Lösung die Datenverteilung $q = p$ am globalen Minimum $-2\log 2$. Dort gilt überall $D^\star(x) = 0$, und der Diskriminator kann nur noch raten.

<!--en:48d04427-->
Drücken wir $D_{\textrm{KL}}(p \,\|\, m)$ und $D_{\textrm{KL}}(q \,\|\, m)$ durch differentielle Entropien aus, erhalten wir eine informationstheoretische Interpretation:

<!--en:6ddd18a6-->
Demnach ist $\mathrm{JS}(p, q)$ gleich der gegenseitigen Information $I(x; y)$ zwischen einem Beispiel $x \sim m$ und seinem binären Herkunftslabel $y$. Weil ein binäres Label höchstens $\log 2$ Nat an Information trägt, gilt $0 \leq \mathrm{JS}(p, q) \leq \log 2$. Der Höchstwert $\log 2$ wird immer dann erreicht, wenn $p$ und $q$ disjunkte Träger haben.

<!--en:fde6e151-->
### Disjunkte Träger und verschwindende Gradienten

<!--en:ba7ee6ef-->
Da $\mathrm{JS}(p, q)$ bei $\log 2$ in Sättigung geht, ist die Minimax-Zielfunktion konstant, sobald sich die Träger von $p$ und $q$ nicht überlappen. Sind etwa $p = \delta_0$ und $q_\theta = \delta_\theta$ Punktmassen auf $\mathbb{R}$, so gilt $\mathrm{JS}(p, q_\theta) = \log 2$ für jedes $\theta \neq 0$. Der Gradient bezüglich $\theta$ ist also null, unabhängig davon, wie nahe $\theta$ an $0$ liegt. Selbst wenn sich die Randbereiche der Verteilungen überlappen, wie bei $p = \mathcal{N}(0, 1)$ und $q = \mathcal{N}(\mu, 1)$, beträgt $\mathrm{JS}(p, q)$ bereits $0.11$, $0.34$ und $0.63$ Nat für $\mu = 1, 2, 4$. Bei $\mu = 8$ liegt der Wert weniger als $10^{-4}$ von $\log 2 \approx 0.6931$ entfernt. :numref:`fig_gan_js_saturation` zeigt den gesamten Verlauf. Die Kurve ist schon flach, lange bevor man die beiden Gauß-Verteilungen in einem formalen Sinne als überlappungsfrei bezeichnen könnte.

<!--en:c7f727a2-->
![Jensen-Shannon-Divergenz zwischen $p = \mathcal{N}(0, 1)$ und $q = \mathcal{N}(\mu, 1)$ als Funktion der Verschiebung $\mu$. Die Divergenz erreicht ihre Obergrenze $\log 2$ innerhalb weniger Standardabweichungen. Danach verläuft die Kurve flach und liefert dem Generator keinen Gradienten.](../img/mdl-gan-js-saturation.svg)

<!--en:5a39a98c-->
In hohen Dimensionen sind disjunkte Träger die Regel. Sowohl natürliche Bilder als auch die Ausgaben eines Generators $G: \mathbb{R}^{d_z} \to \mathbb{R}^{d_x}$ mit $d_z \ll d_x$ konzentrieren sich auf niedrigdimensionale Mannigfaltigkeiten, deren Schnittmenge typischerweise das Maß null hat :cite:`Arjovsky.Bottou.2017`. Trennt der Diskriminator sie perfekt, geht der Minimax-Wert in Sättigung.

<!--en:b9cd5f81-->
Die naheliegendste Abhilfe besteht darin, die Träger zur Überlappung zu bringen. Dazu addieren wir zu jedem echten und jedem generierten Beispiel unabhängiges gaußsches Rauschen mit Skalenparameter $s$, bevor der Diskriminator das Beispiel erhält. Er vergleicht dann zwei geglättete, überall positive Dichten. Für die obigen Punktmassen sind das $\mathcal{N}(0, s^2)$ und $\mathcal{N}(\theta, s^2)$. Ihre Divergenz lesen wir auf der Kurve in :numref:`fig_gan_js_saturation` bei $\mu = \theta/s$ ab. Der Generator erhält einen Gradienten, solange $\theta$ höchstens einige Vielfache von $s$ beträgt. Dieses Verfahren heißt *Instance Noise* :cite:`Sonderby.Caballero.Theis.ea.2017,Arjovsky.Bottou.2017`. Allerdings nähert sich der Generator nun den geglätteten Daten an. Deshalb müssen wir $s$ während des Trainings schrittweise gegen null senken. In :numref:`sec_gan_beyond` begegnet uns die Idee erneut. Dort fügt ein Diffusionsmodell den Beispielen Rauschen hinzu.

<!--en:5bc537ed-->
## Der Gradient des Generators

<!--en:67d51059-->
In der Praxis macht der Generator Gradientenschritte bei festem Diskriminator $D$, anstatt $\max_D V(D)$ direkt zu minimieren. Betrachten wir den Gradienten der Zielfunktion des Generators.

<!--en:c9766b3d-->
### Sättigende und nicht sättigende Gewichte

<!--en:685cac51-->
In der ursprünglichen Minimax-Formulierung minimiert der Generator $\mathbb{E}_z[\log \sigma(-D(G(z)))]$ bezüglich seiner Parameter $\theta$. Mit $\tfrac{d}{du} \log \sigma(-u) = -\sigma(u)$ und $x' = G(z)$ ergibt sich der Gradient:

<!--en:ba9fe15b-->
Jedes generierte Sample $x'$ bewegt sich entlang $\nabla_\theta D(G(z))$ in Richtung höherer Werte. Das Gewicht $\sigma(D(x'))$ gibt an, mit welcher Wahrscheinlichkeit $x'$ nach Einschätzung des Diskriminators echt ist. Erkennt der Diskriminator ein Sample sicher als künstlich ($D(x') \ll 0$), wird das Gewicht $\sigma(D(x')) \approx e^{D(x')}$ exponentiell klein. Der Gradient des Generators geht also gerade dann in *Sättigung*, wenn der Generator sich am stärksten verbessern müsste.

<!--en:49cba533-->
Um diese Sättigung zu vermeiden, trainierten :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` den Generator so, dass er die Log-Wahrscheinlichkeit maximiert, mit der seine Samples als echt eingestuft werden. Dazu minimiert er $-\mathbb{E}_z[\log \sigma(D(G(z)))]$. Wegen $\tfrac{d}{du} \log \sigma(u) = \sigma(-u)$ ergibt sich der Gradient:

<!--en:de8bd813-->
Beide Zielfunktionen bewegen jedes Sample in dieselbe Richtung $\nabla_\theta D(G(z))$. Sie unterscheiden sich nur im skalaren Gewicht für das jeweilige Sample:

<!--en:4baa81b8-->
Wie :numref:`fig_gan_weights` zeigt, nähert sich das *nicht sättigende* Gewicht $w_{\textrm{ns}}(x')$ bei sicher als künstlich erkannten Samples dem Wert $1$ ($D(x') \ll 0$). Das Gewicht $w_{\textrm{sat}}(x')$ hingegen fällt wie $e^{D(x')}$ ab. Das Verhältnis beträgt $w_{\textrm{ns}}/w_{\textrm{sat}} = e^{-D(x')}$, also $20\times$ bei $D = -3$ und $148\times$ bei $D = -5$.

<!--en:15f976fd-->
![Gewichte des Generatorgradienten für ein einzelnes Sample in Abhängigkeit vom Diskriminator-Logit $D(x')$, auf logarithmischer Skala. Bei sicher als künstlich erkannten Samples ($D(x') \ll 0$) fällt das sättigende Gewicht $\sigma(D)$ exponentiell ab. Das nicht sättigende Gewicht $\sigma(-D)$ bleibt nahe bei 1.](../img/mdl-gan-weights.svg)

<!--en:ef626eac-->
### Effektive Divergenz des nicht sättigenden Verlusts

<!--en:4ab44761-->
Welche Divergenz verringert der nicht sättigende Generatorverlust, wenn wir den Diskriminator bei seiner optimalen Antwort $D = \lambda = \log(p/q)$ festhalten? Mit $\log \sigma(-D) - \log \sigma(D) = -D$ können wir die nicht sättigende Zielfunktion zerlegen:

<!--en:559927d0-->
Halten wir $D$ bei $\lambda$ fest, so hat der erste Term rechts bezüglich $q$ denselben Gradienten wie die umgekehrte KL-Divergenz $D_{\textrm{KL}}(q \,\|\, p) = \mathbb{E}_q[\log(q/p)]$. Der zweite Term ist der negative Minimax-Verlust. Nach dem Envelope-Theorem hat der Minimax-Verlust denselben Gradienten wie $2\,\mathrm{JS}(p, q) = D_{\textrm{KL}}(p \,\|\, m) + D_{\textrm{KL}}(q \,\|\, m)$ :cite:`Arjovsky.Bottou.2017` (Aufgabe :exref:`ex_gan-generator-weight-divergences`). Fassen wir die Integrale mithilfe von $\log(q/p) - \log(q/m) = \log(m/p)$ und $p + q = 2m$ zusammen, erhalten wir:

<!--en:ea30af27-->
Bei optimalem Kritiker verringert der nicht sättigende Generator also $2\,D_{\textrm{KL}}(m \,\|\, p)$, die *abgeschwächte umgekehrte KL-Divergenz* (engl. softened reverse KL divergence) :cite:`Shannon.Poole.Mariooryad.ea.2020`. Sie hat dasselbe eindeutige globale Minimum $q = p$.

<!--en:a3b6bde0-->
An $2\,D_{\textrm{KL}}(m \,\|\, p)$ sehen wir, dass Fehler unterschiedlich stark ins Gewicht fallen:

<!--en:97db6d1f-->
- Wo $q(x) > 0$, aber $p(x) = 0$ gilt (künstliche Samples außerhalb des Trägers der Datenverteilung), gelten $m(x) > 0 = p(x)$ und $D_{\textrm{KL}}(m \,\|\, p) = \infty$. Unrealistische Samples werden also stark bestraft.

<!--en:e6359f52-->
- Wo $p(x) > 0$, aber $q(x) = 0$ gilt (fehlende Modi der Datenverteilung), gilt $m(x) = p(x)/2$. Der Integrand trägt nur $p(x) \log \tfrac12$ bei, eine kleine, endliche Strafe.

<!--en:949b2087-->
### Modusausfall

<!--en:4f425e59-->
Deckt ein Generator nur einen Teil der Modi einer multimodalen Verteilung $p$ ab, sprechen wir von *Modusausfall* (engl. mode dropping) oder *Moduskollaps* (engl. mode collapse). Sei $p$ eine gleichgewichtete Mischung zweier disjunkter Modi, von denen $q$ den einen perfekt wiedergibt und den anderen auslässt. Auf dem abgedeckten Modus gilt $p = q/2$. Damit ergeben sich $D_{\textrm{KL}}(q \,\|\, p) = \log 2 \approx 0.69$ Nat, $\mathrm{JS}(p, q) \approx 0.22$ Nat und $2\,D_{\textrm{KL}}(m \,\|\, p) \approx 0.26$ Nat. Lässt der Generator die Hälfte der Datenverteilung aus, kostet das nur ein Viertel Nat. Die Maximum-Likelihood-Methode hingegen minimiert die vorwärtsgerichtete KL-Divergenz $D_{\textrm{KL}}(p \,\|\, q)$. Sie ist unendlich, sobald irgendein Teil von $p$ die Dichte null erhält.

<!--en:4e5f79e7-->
Hinzu kommt ein weiteres Problem. Bei der optimalen Antwort $D^\star(x) = \log(p(x)/q(x)) = -\log 2$ des Diskriminators auf dem abgedeckten Modus ist $D^\star$ auf dem gesamten Träger von $q$ konstant. Daher gilt $\nabla_\theta D^\star(G(z)) = 0$. Der lokale Gradient verschwindet auf dem abgedeckten Modus und liefert kein Signal in Richtung des fehlenden Modus. Nehmen wir an, ein nicht optimaler Diskriminator vergibt etwa den Score $D = 3$ für generierte Samples auf dem abgedeckten Modus und $D = 5$ für echte Beispiele auf dem fehlenden Modus. Auch dann hängt das Gewicht jedes generierten Samples über $\sigma(-3) \approx 0.05$ nur vom Vergleich mit der festen Schwelle null ab. Die höheren Scores für echte Beispiele auf dem fehlenden Modus gehen nie in das Update des Generators ein. In :numref:`sec_gan_relativistic` untersuchen wir, ob ein direkter Vergleich der Scores für generierte und echte Beispiele Abhilfe schafft.

<!--en:4fcf3ab3-->
## Überprüfung der Theorie an einer zweidimensionalen Gauß-Verteilung

<!--en:708f0735-->
Um die theoretischen Vorhersagen quantitativ zu prüfen, trainieren wir zunächst einen linearen Generator auf zweidimensionalen gaußverteilten Daten. Hier lassen sich $p$, $q$, $\lambda(x) = \log(p(x)/q(x))$ und $D_{\textrm{KL}}(q \,\|\, p)$ in geschlossener Form berechnen.

<!--en:757ab12d-->
### Abwechselnde Aktualisierung

<!--en:0778f2d6-->
Wir definieren `update_D`, `update_G` und die Trainingsschleife `train_gan` für die abwechselnden Updates einmal und speichern sie zur weiteren Verwendung im Kapitel in `d2l`. Zunächst macht `update_D` einen Diskriminatorschritt, um :eqref:`eq_gan_V` zu maximieren. Dazu minimiert die Funktion die binäre Kreuzentropie mit dem Label $1$ für echte Beispiele und $0$ für generierte Samples, die vom Berechnungsgraphen des Generators getrennt sind. Teilen wir den summierten Verlust durch die Batchgröße, erhalten wir einen Schätzwert für $-V(D)$. Dieser Ausdruck ist gleich $2\log 2 \approx 1.386$, wenn $D = 0$ gilt.

<!--en:b347e687-->
Anschließend macht `update_G` einen Generatorschritt, um den nicht sättigenden Verlust $-\mathbb{E}[\log \sigma(D(G(z)))]$ zu minimieren. Wir berechnen dazu die binäre Kreuzentropie für die neu berechneten Diskriminator-Scores `net_D(net_G(Z))` mit dem Ziellabel $1$. Pro Sample ergibt sich $\log 2 \approx 0.693$, wenn $D = 0$ gilt:

<!--en:5ae9e468-->
`train_gan` wendet auf jeden Minibatch abwechselnd `update_D` und `step_G` an. Wir verwenden Adam mit $\beta_1 = 0.5$ :cite:`Radford.Metz.Chintala.2015`, damit sich das Momentum schnell an die wechselnde Zielfunktion anpasst. Optional unterstützt die Schleife eine Rückruffunktion `on_epoch` für jede Epoche und einen exponentiellen gleitenden Mittelwert `ema`.

<!--en:78d37663-->
### Ein linearer Generator mit bekannten Dichten

<!--en:a6b25145-->
Wir erzeugen 1000 Trainingspunkte nach $x = z A + b$ mit $z \sim \mathcal{N}(0, I_2)$. Die wahre Datenverteilung ist damit die schräg ausgerichtete elliptische Gauß-Verteilung $p = \mathcal{N}(b, A^\top A)$.

<!--en:14dd4b78-->
Der Generator besteht aus einer einzigen linearen Schicht $G(z) = z A_q + \mu_q$ und induziert die Gauß-Verteilung $q = \mathcal{N}(\mu_q, A_q^\top A_q)$. Das logarithmierte Dichteverhältnis zweier Gauß-Verteilungen mit unterschiedlichen Kovarianzen ist quadratisch. Der Diskriminator verwendet deshalb ein dreischichtiges MLP mit zwei versteckten Schichten aus je 16 Leaky-ReLU-Einheiten (Steigung $0.2$ für negative Eingaben). In Aufgabe :exref:`ex_gan-dead-critic` untersuchen wir, was passiert, wenn wir stattdessen sättigende $\tanh$-Aktivierungen verwenden.

<!--en:fa01cc25-->
Wir trainieren 20 Epochen lang mit Batchgröße 8. Dabei wählen wir `lr_D=0.05` und `lr_G=0.005` so, dass der Diskriminator seiner optimalen Antwort folgt, während sich der Generator verändert.

<!--en:de4dea45-->
Nun stellen wir die in `history` erfassten Verluste von Kritiker und Generator pro Epoche ihren theoretischen Gleichgewichtswerten ($2\log 2$ und $\log 2$) gegenüber. Daneben zeigen wir ein Streudiagramm der echten (`data`) und generierten (`fake`) Beispiele:

<!--en:28609d7f-->
Innerhalb von 10 Epochen steigt der Diskriminatorverlust auf den Gleichgewichtswert $2\log 2$, der Generatorverlust pendelt sich bei $\log 2$ ein und die generierte Punktwolke bildet die Zielellipse nach.

<!--en:928e95f5-->
### Anpassungsgüte in Nat

<!--en:246ab2e4-->
Da $G(z) = z A_q + \mu_q$ eine Gauß-Verteilung $q = \mathcal{N}(\mu_q, \Sigma_q)$ mit $\Sigma_q = A_q^\top A_q$ definiert, können wir die exakte KL-Divergenz $D_{\textrm{KL}}(q \,\|\, p)$ in $d_x$ Dimensionen berechnen :cite:`Kullback.1959`:

<!--en:8252d890-->
Das adversariale Training wertet die Dichte von $q$ nie aus. Dennoch senkt es $D_{\textrm{KL}}(q \,\|\, p)$ von $\sim 8$ Nat bei der Initialisierung auf wenige Hundertstel Nat. Den hohen Anfangswert bestimmt vor allem die geringe Entropie der anfänglich gemäß $\mathcal{N}(0, 0.02^2)$ verteilten Gewichte.

<!--en:0cc2fdf4-->
### Überprüfung des gelernten logarithmierten Dichteverhältnisses

<!--en:21d3aa49-->
Wir prüfen, wie genau ein trainierter Diskriminator $\lambda(x) = \log(p(x)/q(x))$ rekonstruiert (:eqref:`eq_gan_dstar`). Dazu wählen wir einen festen Generator $q$ mit $A_q = [[1, 2], [-0.5, 0.5]]$ und $\mu_q = (1.5, 1.5)$ ($D_{\textrm{KL}}(q \,\|\, p) \approx 3.4$ Nat). Gegen dieses feste $q$ trainieren wir einen neuen Diskriminator über 2000 Schritte.

<!--en:245493a6-->
Wir werten das Diskriminator-Logit $D(x)$ und das exakte logarithmierte Dichteverhältnis $\lambda(x)$ an 200 echten und 200 generierten Punkten aus und geben ihre Korrelation sowie den Anteil der erklärten Varianz $R^2$ an.

<!--en:7e9cb07c-->
Wir tragen die Logits des Kritikers `D_out` für die 200 echten und die 200 generierten Beispiele gegen das exakte logarithmierte Dichteverhältnis `lam` auf. So sehen wir, wie eng $D(x)$ der Identitätsgeraden $D = \lambda$ folgt:

<!--en:d973174d-->
Die gelernten Logits folgen dem wahren logarithmierten Dichteverhältnis sehr genau ($\text{correlation} \approx 0.98$, $R^2 > 0.90$). Abweichungen treten nur bei extrem negativen Werten von $\lambda$ auf, also bei generierten Samples mit $p(x) \approx 0$. Das stückweise lineare Leaky-ReLU-Netz extrapoliert in den Randbereichen linear statt quadratisch. Außerdem gilt $\tfrac{d}{dD}[-\log \sigma(-D)] = \sigma(D) \ll 1$ für $D \ll 0$. Sobald sich der Diskriminator sicher ist, liefert der logarithmische Verlust daher kaum noch einen Gradienten, der die Steigung weiter erhöhen könnte.

<!--en:80930c98-->
### Sättigender und nicht sättigender Generatorverlust

<!--en:e0e9eefc-->
Um die sättigenden und nicht sättigenden Gewichte (:eqref:`eq_gan_weights`) bei einem Start weit entfernt von den Daten zu vergleichen, verschieben wir den anfänglichen Bias des Generators um $(-6, 6)$ (Abstand vom Datenmittelwert $\sim 8$ Einheiten, $D_{\textrm{KL}}(q \,\|\, p) \approx 345$ Nat). Wir vergleichen `update_G` mit `update_G_saturating`, das $\mathbb{E}_z[\log \sigma(-D(G(z)))]$ direkt minimiert.

<!--en:af2089e1-->
Für jede Zielfunktion des Generators führen wir drei Läufe mit verschiedenen Zufallsstartwerten durch. Dabei verwenden wir für beide Zielfunktionen jeweils dieselbe Initialisierung und dieselben Minibatches (`lr_G=0.01`, `lr_D=0.05`).

<!--en:1a31487c-->
Um die großen Unterschiede zwischen den abschließenden KL-Divergenzen zu verstehen, zeichnen wir für beide Zielfunktionen die generierten Punktwolken `snaps` aus den Epochen 5 und 20. Daneben zeigen wir den Verlauf des Kritikerverlusts:

<!--en:f7516e01-->
In den ersten beiden Epochen unterscheiden beide Diskriminatoren die weit entfernten generierten Punkte mühelos von den Daten. Der Diskriminatorverlust sinkt fast auf null, und es gilt $D(x') \ll 0$. Wegen $w_{\textrm{sat}}(x') = \sigma(D(x')) \approx 0$ kommt der Generator mit sättigendem Verlust völlig zum Stillstand. Nach 20 Epochen beträgt der Abstand zu den Daten noch $335\text{--}340$ Nat. Das nicht sättigende Gewicht $w_{\textrm{ns}}(x') = \sigma(-D(x')) \approx 1$ erhält hingegen den räumlichen Gradienten $\nabla_\theta D(G(z))$ des Diskriminators. Dieser zieht den Generator innerhalb von 20 Epochen zur Datenverteilung ($D_{\textrm{KL}}(q \,\|\, p)$ von wenigen Hundertstel Nat). Dabei skaliert $w_{\textrm{ns}}$ den Gradienten $\nabla_\theta D$ lediglich. Das Verfahren setzt voraus, dass der Leaky-ReLU-Diskriminator bei $x'$ eine von null verschiedene räumliche Steigung hat.

<!--en:712bdfff-->
## Training eines MLP-GAN auf Fashion-MNIST

<!--en:7979783e-->
Für Bilder kennen wir weder $p(x)$ noch $q(x)$ in geschlossener Form. Deshalb müssen wir die Sample-Qualität und die Trainingsverluste verfolgen. In unserem ersten Bildexperiment trainieren wir zwei Leaky-ReLU-MLPs auf den 60 000 Graustufenbildern der Größe $28 \times 28$ aus Fashion-MNIST (:numref:`sec_fashion_mnist`). Wir skalieren die Pixelwerte auf $[-1, 1]$. Der Generator bildet einen 100-dimensionalen latenten Vektor über versteckte Schichten mit 256 und 512 Einheiten auf 784 Pixel mit $\tanh$-Ausgabeaktivierung ab. Der Diskriminator bildet 784 Pixel über 512 und 256 Einheiten auf ein einziges Logit ab. Wir speichern die Hilfsfunktion `make_mlp`, um in späteren Abschnitten einfache MLPs aufzubauen.

<!--en:e2039d20-->
Für die Netze verwenden wir `make_mlp`. Wir skalieren die 60 000 Trainingsbilder von Fashion-MNIST und stellen sie als $784$-dimensionale Vektoren mit Werten in $[-1, 1]$ dar. Wir erstellen den Datenlader `image_iter` für die Minibatches, den Generator `net_G_img` mit der abschließenden Aktivierung `tanh` und den Diskriminator `net_D_img`:

<!--en:e8df3c9c-->
Wir trainieren beide Netze mit Lernrate $2 \times 10^{-4}$ und Batchgröße 64 über 20 Epochen ($\sim 18{,}750$ Schritte).

<!--en:72ab68e3-->
Nach dem Training ziehen wir 30 latente Vektoren aus $\mathcal{N}(0, I_{100})$ und leiten sie durch `net_G_img`. Wir formen die Ausgaben zu Bildern der Größe $28 \times 28$ um und betrachten die generierten Kleidungsstücke:

<!--en:d45ca2da-->
Außerdem tragen wir die in `history_img` erfassten Verluste von Kritiker und Generator pro Epoche zusammen mit den Referenzwerten $2\log 2$ und $\log 2$ auf:

<!--en:0132f524-->
Die Silhouetten von Hemden, Hosen, Schuhen und Taschen sind auf den generierten Bildern erkennbar. Die Pixel weisen allerdings körniges Rauschen auf, weil einem MLP der räumliche induktive Bias von Faltungen fehlt. Selbst nach 20 Epochen liegt der Diskriminatorverlust ($\sim 1.05$) unter seinem Gleichgewichtswert $2\log 2 \approx 1.39$ und der Generatorverlust ($\sim 1.4$) über $\log 2 \approx 0.69$. Anders als im zweidimensionalen Gauß-Beispiel zeigen die Verlustkurven bei Bildern allein nicht, ob der Generator die Daten unzureichend nachbildet oder beim Diskriminator Überanpassung (engl. overfitting) vorliegt (Aufgabe :exref:`ex_gan-finite-sample-critic`). Auch lässt sich an ihnen nicht ablesen, ob alle Bekleidungskategorien gleich gut vertreten sind (Aufgabe :exref:`ex_gan-class-coverage`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:89a0adac-->
Ein generatives adversariales Netz trainiert einen impliziten Generator $x' = G(z)$ ohne explizite Likelihood. Dazu kombiniert es ihn mit einem Diskriminator (oder Kritiker) $D$, der echte Daten $x \sim p$ von generierten Samples $x' \sim q$ unterscheidet. Beim binären logarithmischen Verlust liefert das optimale Diskriminator-Logit das logarithmierte Dichteverhältnis $D^\star(x) = \lambda(x) = \log(p(x)/q(x))$, und die optimale Wertfunktion vereinfacht sich zu $\max_D V(D) = 2\,\mathrm{JS}(p, q) - 2\log 2$. Dabei ist $\mathrm{JS}(p, q) = I(x; y) \in [0, \log 2]$ die gegenseitige Information zwischen einem Beispiel aus $m = (p + q)/2$ und seinem Herkunftslabel. Erkennt der Diskriminator generierte Samples sicher als künstlich ($D(x') \ll 0$), fällt das Generatorgewicht $w_{\textrm{sat}}(x') = \sigma(D(x'))$ der ursprünglichen Minimax-Formulierung exponentiell gegen null. Beim nicht sättigenden Generatorverlust $-\mathbb{E}[\log \sigma(D(G(z)))]$ bleibt das Gewicht $w_{\textrm{ns}}(x') = \sigma(-D(x')) \approx 1$ dagegen nahe bei eins. Bei optimalem Diskriminator verringert der nicht sättigende Verlust $2\,D_{\textrm{KL}}(m \,\|\, p)$. Diese Divergenz bestraft Samples außerhalb des Datenträgers stark, weist fehlenden Modi aber nur eine beschränkte Strafe zu.

<!--en:2294243a-->
## Aufgaben

<!--en:a8f99211-->
1. **Überschussverlust und additive Konstante.** Sei $\eta = p / (p + q)$ und sei $D_{\textrm{b}}(a \,\|\, u)$ die KL-Divergenz zwischen den Bernoulli-Verteilungen mit Parametern $a$ und $u$.
    1. Zeigen Sie, dass für jeden Kritiker $V(D^\star) - V(D) = 2\, \mathbb{E}_{m}\big[D_{\textrm{b}}(\eta \,\|\, \sigma(D))\big]$ gilt. Damit ist $D^\star$ der eindeutige Maximierer von :eqref:`eq_gan_V` bis auf $m$-Nullmengen.
    1. Folgern Sie, dass $V(D^\star + b) < V(D^\star)$ für jede Konstante $b \neq 0$ gilt.
    1. Entwickeln Sie bis zur zweiten Ordnung in $b$ und zeigen Sie die Beziehung

       $$V(D^\star + b) - V(D^\star) = -\frac{b^2}{2} \int \frac{p q}{p + q}\, dx + O(b^3),$$

       wobei der Koeffizient in $[0, \tfrac12]$ liegt. Wenn $p$ und $q$ nahezu disjunkt sind, wie genau legen endliche Stichproben dann die additive Konstante von $D$ fest?
   :exlabel:`ex_gan-log-loss-game-identifies`

<!--en:df5e90a9-->
1. **Träger vom Maß null.**
    1. Ein Generator $G(z) = z v + \mu_q$ mit eindimensionaler latenter Variable $z \in \mathbb{R}$ erzeugt Samples auf einer Geraden in $\mathbb{R}^2$. Zeigen Sie, dass $\mathrm{JS}(p, q) = \log 2$ für die zweidimensionale gaußsche Datenverteilung $p = \mathcal{N}(\mu_p, A^\top A)$ bei beliebigen $v$ und $\mu_q$ gilt.
    1. Zeigen Sie, dass $\mathrm{JS}(\hat p, q) = \log 2$ für die empirische Verteilung $\hat p$ der 1000 Trainingspunkte und jede stetige Dichte $q$ gilt (einschließlich $q = p$).
    1. Betrachten Sie einen $\kappa$-Lipschitz-stetigen Kritiker $D$. Zeigen Sie mithilfe der Kantorowitsch-Rubinstein-Dualität (:numref:`sec_mdl-optimal-transport`), dass das Ersetzen von $p$ durch $\hat p$ in :eqref:`eq_gan_V` den Wert $V(D)$ um höchstens $\kappa\, W_1(\hat p, p)$ ändert. Warum liefert das Training mit empirischen Stichproben aussagekräftige Ergebnisse, wenn wir die Lipschitz-Konstante beschränken?
   :exlabel:`ex_gan-measure-zero-supports`

<!--en:39a88df8-->
1. [code] **Sättigende Aktivierungen des Kritikers.** ● Ersetzen Sie die Leaky-ReLU-Aktivierungen in `make_nets` durch $\tanh$-Einheiten und wählen Sie für die versteckten Schichten die Breiten 5 und 3.
    1. Wiederholen Sie das Experiment mit nicht sättigendem Verlust aus :numref:`subsec_gan_saturation` für fünf Zufallsstartwerte. Halten Sie jeweils den Endwert von $D_{\textrm{KL}}(q \,\|\, p)$ und den abschließenden Kritikerverlust fest.
    1. Untersuchen Sie bei einem Lauf, der zum Stillstand kommt, die Aktivierungen der versteckten Schichten und $\|\nabla_x D(x)\|$ an echten und generierten Beispielen. Warum kann sich der Generator trotz seines nicht sättigenden Verlusts nicht erholen, wenn die versteckten Einheiten des Kritikers in Sättigung gehen?
    1. Sowohl ein konvergierter Generator ($q = p$) als auch ein konstanter Kritiker ($D(x) = 0$) führen zu einem Kritikerverlust von $2\log 2$. Mit welcher Diagnose aus diesem Abschnitt können Sie die beiden Fälle unterscheiden?
   :exlabel:`ex_gan-dead-critic`

<!--en:3a3e1670-->
1. [code] **Klassenabdeckung des Fashion-MNIST-Generators.** ●
    1. Trainieren Sie LeNet (:numref:`sec_lenet`) auf Fashion-MNIST. Klassifizieren Sie 1000 Samples aus `net_G_img` und vergleichen Sie das Histogramm der vorhergesagten Klassen mit der Gleichverteilung der Klassen in den Daten.
    1. Wiederholen Sie den Versuch mit Generatoren, die Sie über 5 beziehungsweise 40 Epochen trainieren. Wird die Klassenverteilung durch längeres Training gleichmäßiger?
    1. Wie verzerrt die erzwungene Klassifikation in 10 Klassen das Histogramm, wenn einige generierte Bilder unscharf oder mehrdeutig sind? Setzen Sie Ihre Ergebnisse in Beziehung zum Modusausfall (:numref:`subsec_gan_mode_dropping`).
   :exlabel:`ex_gan-class-coverage`

<!--en:ebbd189e-->
1. **Generatorgewichte und zugehörige Divergenzen.** ●● Ein Generatorverlust $\mathbb{E}_z[g(D(G(z)))]$ hat bei festem Kritiker $D$ den Gradienten $-\mathbb{E}_z[w(D)\, \nabla_\theta D(G(z))]$ mit $w = -g'$. Halten Sie $D$ beim optimalen logarithmierten Dichteverhältnis $\lambda = \log(p / q_{\theta_0})$ für den aktuellen Generator $q_{\theta_0}$ fest und setzen Sie $\rho = e^D = p / q_{\theta_0}$.
    1. Zeigen Sie mithilfe von $\int \nabla_\theta q_\theta = 0$, dass der ungewichtete Verlust $\mathbb{E}_z[-D(G(z))]$ ($w = 1$) den Gradienten $\nabla_\theta D_{\textrm{KL}}(q_\theta \,\|\, p)$ an der Stelle $\theta = \theta_0$ hat.
    1. Zeigen Sie für die $f$-Divergenz $D_f(p \,\|\, q_\theta) = \mathbb{E}_{q_\theta}[f(p/q_\theta)]$ (:numref:`sec_mdl-f-divergences`), dass $\nabla_\theta D_f\big|_{\theta_0} = -\mathbb{E}_z\big[\rho^2 f''(\rho)\, \nabla_\theta D(G(z))\big]$ gilt. Einer Gewichtsfunktion $w$ entspricht also die $f$-Divergenz, für die $\rho^2 f''(\rho) = w(\log \rho)$ gilt.
    1. Bestimmen Sie die Divergenzen zu $w_{\textrm{sat}}$, $w = 1$, $w(t) = e^t$ :cite:`Goodfellow.2016` und $w_{\textrm{ns}}$ (Satz 2.5 in :citet:`Arjovsky.Bottou.2017`).
    1. Berechnen Sie mithilfe von :eqref:`eq_gan_ns_divergence` den Wert von $2\,D_{\textrm{KL}}(m \,\|\, p)$, wenn $p$ aus $K$ gleich gewichteten disjunkten Modi besteht und $q$ nur einen davon abdeckt. Vergleichen Sie das Wachstum in $K$ mit $D_{\textrm{KL}}(q \,\|\, p) = \log K$.

   *Nach [CS236: Deep Generative Models](https://cs236.stanford.edu) von Stefano Ermon und Aditya Grover.*
   :exlabel:`ex_gan-generator-weight-divergences`

<!--en:cf73283c-->
1. [extended] **Endliche Stichproben und Kapazität des Kritikers.** ●● Übersteigt der Stichprobenumfang die Anzahl der Kritikerparameter um den Faktor $\tilde O(1/\epsilon^2)$, konzentriert sich der empirische Wert des Kritikers in einer $\epsilon$-Umgebung seines Werts auf der Grundgesamtheit :cite:`Arora.Ge.Liang.ea.2017`. Stellen Sie den Generator fest auf die wahre Datenverteilung ein ($G(z) = z A + \mu_p$, also $q = p$).
    1. Trainieren Sie Kritiker mit je zwei versteckten Schichten der Breiten 4, 16, 128 und 1024 auf $n \in \{100, 1000, 10000\}$ festen Trainingspunkten aus $p$ (Adam mit Lernrate $10^{-3}$, 4000 Schritte mit Batchgröße 256). Tragen Sie die Generalisierungslücke zwischen dem Trainingsverlust und dem an frisch gezogenen Beispielen gemessenen Verlust auf der Grundgesamtheit gegen die Parameteranzahl auf.
    1. Der Fashion-MNIST-Kritiker in :numref:`subsec_gan_fashion_mnist` hat $\sim 530{,}000$ Parameter und trainiert auf 60 000 Bildern. Entwerfen Sie mit dem Testdatensatz von Fashion-MNIST einen Test und führen Sie ihn durch. Prüfen Sie damit, ob der Verlust unter $2\log 2$ liegt, weil der Kritiker den Trainingsdatensatz auswendig gelernt hat oder weil sich $p$ und $q$ tatsächlich unterscheiden.
   :exlabel:`ex_gan-finite-sample-critic`

<!--en:e52af3d9-->
[Diskussionen](https://d2l.discourse.group/t/36808)

<!--en:a05054a7-->
Generative adversariale Netze als Minimax-Spiel mit zwei Spielern<br>**implizite Generatoren · Kritiker mit binärem logarithmischem Verlust · optimales logarithmiertes Dichteverhältnis · Sättigung der Jensen-Shannon-Divergenz · nicht sättigender Verlust und Modusausfall**

<!--en:56791630-->
::: {.slide title="Implizite Generatoren und warum die Likelihood täuschen kann"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:f664bbb6-->
Die Maximum-Likelihood-Schätzung wertet $\log q_\theta(x)$ an jeder Trainingsbeobachtung aus. Ein **impliziter Generator** definiert dagegen eine Verteilung direkt durch ein Sampling-Programm $x' = G_\theta(z)$ mit $z \sim \mathcal{N}(0, I_{d_z})$:

<!--en:84b5fc0e-->
- **Singulärer Träger ($d_z \ll d_x$):** Die Abbildung von $d_z = 100$ gaußverteilten Zahlen auf $d_x = 784$ Pixel eines Fashion-MNIST-Bildes konzentriert $q_\theta$ auf einer $100$-dimensionalen Mannigfaltigkeit. Es gibt weder eine Dichte auf $\mathbb{R}^{784}$ noch eine effizient berechenbare Jacobi-Determinante.

<!--en:6a399338-->
- **Hohe Likelihood $\neq$ realistische Samples (Theis et al., 2016):** Für die Mischung $\tilde{p}(x) = 0.01\,p_{\textrm{data}}(x) + 0.99\,p_{\textrm{noise}}(x)$ gilt $\log \tilde{p}(x) \ge \log p_{\textrm{data}}(x) - \log 100$. Ihre erwartete Log-Likelihood liegt höchstens **$4.6$ Nat** unter derjenigen der wahren Verteilung, obwohl sie **$99\%$ Rauschen** erzeugt.

<!--en:33c6ce76-->
::: {.slide title="Zweistichprobenvergleich mit einem binären Klassifikator"}

<!--en:e6b8ea39-->
[Minimax-Architektur]{.kicker}

<!--en:6b3b477e-->
So wie ein Domänenklassifikator eine Kovariatenverschiebung zwischen Trainings- und Testeingaben erkennt, lernt in einem GAN ein binärer **Diskriminator (Kritiker)** $D: \mathcal{X} \to \mathbb{R}$, echte Daten von generierten Samples zu unterscheiden:

<!--en:f8be1a3c-->
- **Gleichgewichtete Mischung:** Herkunftslabel $y \sim \textrm{Bernoulli}(\frac{1}{2})$ ziehen. Bei $y = 1$ ein Beispiel $x \sim p$ (echte Daten), bei $y = 0$ ein Sample $x' = G(z) \sim q$ (Generator) ziehen. Zusammen folgen die Beispiele der Verteilung $m(x) = \frac{1}{2}(p(x) + q(x))$.

<!--en:ea56a1c0-->
- **Logit für die Echtheit $D(x)$:** Die logistische Sigmoidfunktion $\sigma(t) = 1/(1 + e^{-t})$ bildet den unbeschränkten Kritiker-Score $D(x)$ auf einen Schätzwert für die A-posteriori-Wahrscheinlichkeit $P(y = 1 \mid x)$ ab.

<!--en:ab0ade6a-->
- **Abwechselnde Updates:** $D$ passt seine Gewichte an, um echte von generierten Minibatches zu unterscheiden. $G$ aktualisiert seine Gewichte mit Gradienten, die durch $D$ zurückpropagiert werden.

<!--en:63b372d5-->
::: {.slide title="Minimax-Zielfunktion mit binärem logarithmischem Verlust"}

<!--en:9f50dd3b-->
[Formulierung]{.kicker}

<!--en:38aec3d9-->
- **Diskriminator ($\max_D V(D)$):** Zielt auf $D(x) > 0$ für echte Daten $x \sim p$ und $D(x') < 0$ für generierte Samples $x' \sim q$.

<!--en:0a9d0b37-->
- **Generator ($\min_G \max_D V(D)$):** Verschiebt $x' = G(z)$ in Richtung wachsender Werte von $D$, bis sich generierte Samples nicht mehr von echten Daten unterscheiden lassen.

<!--en:d291b823-->
- **Zufälliges Raten als Baseline ($D = 0$):** Sagt überall $\sigma(0) = \frac{1}{2}$ vorher. Das ergibt $V(0) = -2\log 2 \approx -1.386$.

<!--en:1b547b6c-->
::: {.slide title="Herleitung des optimalen Kritikers als logarithmiertes Dichteverhältnis"}

<!--en:aa142bcf-->
[Punktweises Optimum]{.kicker}

<!--en:79a7bbe8-->
- **Punktweise Zerlegung:** Mit $u = \sigma(D(x)) \in (0, 1)$ liefert die Maximierung von $p(x)\log u + q(x)\log(1 - u)$ das Optimum $\frac{p(x)}{u} = \frac{q(x)}{1 - u}$.

<!--en:4108d7a4-->
- **Exaktes logarithmiertes Dichteverhältnis:** Die Umkehrung der logistischen Sigmoidfunktion liefert $D^\star(x) = \log(p(x)/q(x)) = \lambda(x)$.

<!--en:3b68fbc6-->
- **Feste Nullschwelle:** Es gilt $D^\star(x) > 0 \iff p(x) > q(x)$, und $D^\star(x) = 0$ gilt überall genau dann, wenn $q = p$ gilt.

<!--en:f4d569e0-->
::: {.slide title="Rechenbeispiel zum optimalen Kritiker für zwei Gauß-Verteilungen"}

<!--en:310792c1-->
[Analytische Betrachtung]{.kicker}

<!--en:5ed205f8-->
- **Verschiebung bei gleicher Varianz ($p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(\mu, 1)$):** Die quadratischen Terme in $x^2/2$ heben sich auf. Es bleibt ein **affiner** Kritiker, der in der Mitte bei $x = \mu/2$ das Vorzeichen wechselt.

<!--en:21d18887-->
- **Verschiedene Kovarianzen ($\Sigma_p \neq \Sigma_q$):** Das logarithmierte Dichteverhältnis enthält weiterhin den quadratischen Term $-\frac{1}{2}x^\top(\Sigma_p^{-1} - \Sigma_q^{-1})x$. Es erfordert ein **nichtlineares** neuronales Netz (z. B. ein Leaky-ReLU-MLP).

<!--en:002f7b06-->
::: {.slide title="Minimax-Wert und Jensen-Shannon-Divergenz"}

<!--en:dc61838f-->
[Zielfunktion auf der Grundgesamtheit]{.kicker}

<!--en:42b3708c-->
- **Einsetzen von $\sigma(D^\star) = \frac{p}{2m}$:** Mit der Mischung $m = \frac{1}{2}(p + q)$ lässt sich die optimale Wertfunktion durch die **Jensen-Shannon-Divergenz** $\mathrm{JS}(p, q)$ ausdrücken.

<!--en:d9e766ad-->
- **Eindeutiges globales Gleichgewicht ($q = p$):** Es gilt $\mathrm{JS}(p, q) \ge 0$ mit Gleichheit genau dann, wenn $q = p$ gilt. Die Minimierung von $\max_D V(D)$ liefert daher als eindeutige Lösung $q = p$ mit dem Wert $-2\log 2$.

<!--en:4a441158-->
::: {.slide title="Informationstheoretische Sicht und Sättigung der JS-Divergenz"}

<!--en:203f9717-->
[Problem disjunkter Träger]{.kicker}

<!--en:6151a0e5-->
- **Identität mit der gegenseitigen Information:** $\mathrm{JS}(p, q)$ ist gleich der gegenseitigen Information $I(x; y)$ zwischen einem Beispiel $x \sim m$ und seinem binären Herkunftslabel $y \in \{0, 1\}$.

<!--en:dde14054-->
- **Warum $\mathrm{JS}$ in hohen Dimensionen sättigt:** Liegen $p$ und $q$ auf disjunkten niedrigdimensionalen Mannigfaltigkeiten, verrät jedes Beispiel sein Label $y$ mit Sicherheit. Dann bleibt $\mathrm{JS}(p, q) = \log 2 \approx 0.6931$ Nat konstant und die Steigung ist null.

<!--en:9d7abb0f-->
::: {.slide title="Sättigende und nicht sättigende Generatorgradienten"}

<!--en:e20a71f9-->
[Skalierung des Gradienten]{.kicker}

<!--en:2a48655e-->
In der Praxis macht der Generator einen Gradientenschritt bei festem Kritiker $D$. Beide Verlustfunktionen bewegen jedes Sample $x' = G_\theta(z)$ in dieselbe räumliche Richtung $\nabla_\theta D(G_\theta(z))$. Nur das skalare Gewicht unterscheidet sich:

<!--en:35cfc4ae-->
- **Sättigender Minimax-Verlust** ($\min_\theta \mathbb{E}_z[\log \sigma(-D(x'))]$):
  $$\nabla_\theta \mathcal{L}_{\textrm{sat}} = -\mathbb{E}_z\Big[\underbrace{\sigma(D(x'))}_{w_{\textrm{sat}}(x')}\,\nabla_\theta D(G_\theta(z))\Big]$$
  Wenn $D$ das Sample $x'$ als künstlich erkennt ($D(x') \ll 0$), gilt $w_{\textrm{sat}}(x') \approx e^{D(x')} \to 0$.

<!--en:b99296f0-->
- **Nicht sättigender Verlust** ($\min_\theta -\mathbb{E}_z[\log \sigma(D(x'))]$):
  $$\nabla_\theta \mathcal{L}_{\textrm{ns}} = -\mathbb{E}_z\Big[\underbrace{\sigma(-D(x'))}_{w_{\textrm{ns}}(x')}\,\nabla_\theta D(G_\theta(z))\Big]$$
  Für als künstlich erkannte Samples gilt $w_{\textrm{ns}}(x') \to 1$ ($w_{\textrm{ns}}/w_{\textrm{sat}} = e^{-D(x')}$, also ein um den Faktor $20\times$ größeres Gewicht bei $D = -3$ und um den Faktor $148\times$ bei $D = -5$).

<!--en:daae955c-->
::: {.slide title="Effektive Divergenz des nicht sättigenden Verlusts"}

<!--en:e6be057a-->
[Abgeschwächte umgekehrte KL-Divergenz]{.kicker}

<!--en:76ab0b59-->
- **Identität aus dem Envelope-Theorem:** Mit $-\log\sigma(D) = -D - \log\sigma(-D)$ an der Stelle $D = \lambda = \log(p/q)$ verringert der nicht sättigende Generatorverlust die **abgeschwächte umgekehrte KL-Divergenz** $2\,D_{\textrm{KL}}(m \,\|\, p)$.

<!--en:552f0a4f-->
- **Asymmetrische Strafe:** Unrealistische Samples ($q > 0$ bei $p = 0$) verursachen **unendliche** Kosten ($m\log(m/0) = \infty$). Fehlende Datenmodi ($q = 0$ bei $p > 0$) tragen nur den endlichen Integranden $\tfrac{p}{2}\log\tfrac{1}{2}$ bei. Einen von zwei Modi auszulassen kostet daher nur $\approx 0.26$ Nat.

<!--en:8af4142d-->
::: {.slide title="Modusausfall: Warum der Verlust der halben Datenmenge wenig kostet"}

<!--en:596f0a96-->
[Analyse des Modusausfalls]{.kicker}

<!--en:9fb666c0-->
- **Ausfall eines von zwei gleich gewichteten Modi ($q = 2p$ auf Modus 1, $q = 0$ auf Modus 2):**
  - Maximum-Likelihood $D_{\textrm{KL}}(p \,\|\, q) = \infty$ schließt den Ausfall eines Modus strikt aus.
  - Beim nicht sättigenden GAN kostet der Ausfall der halben Datenmenge nur $2\,D_{\textrm{KL}}(m \,\|\, p) \approx 0.262$ Nat (und $\mathrm{JS} \approx 0.216$ Nat).

<!--en:597c5f0a-->
- **Kein lokaler Gradient zum fehlenden Modus:** Auf dem abgedeckten Modus ist $D^\star(x) = \log(p/2p) = -\log 2$ über $\operatorname{supp}(q)$ konstant. Daher zieht $\nabla_x D^\star(G(z)) = \mathbf{0}$ die Samples nicht zum fehlenden Modus.

<!--en:7144e1dc-->
::: {.slide title="Implementierung der abwechselnden Kritiker- und Generatorschritte" layout="tight"}

<!--en:873e9ee9-->
[Implementierung im Framework]{.kicker}

<!--en:fc334844-->
`update_D` trennt die generierten Samples vom Berechnungsgraphen, sodass kein Gradient in `net_G` fließt. `update_G` berechnet die Kritiker-Scores nach dem Teilschritt von `net_D` neu und minimiert die nicht sättigende Kreuzentropie mit Ziellabel $1$:

<!--en:5b36c026-->
::: {.slide title="Implementierung der abwechselnden Kritiker- und Generatorschritte (Forts.)" layout="tight"}

<!--en:873e9ee9-2-->
[Implementierung im Framework]{.kicker}

<!--en:2b503990-->
::: {.slide title="GAN-Trainingsschleife mit abwechselnden Updates" only="pytorch"}

<!--en:d7e2024d-->
[Trainingsinfrastruktur]{.kicker}

<!--en:f84e15b8-->
`train_gan` führt für jeden Minibatch abwechselnd `update_D` und `step_G` aus. Adam verwendet ein reduziertes Momentum $\beta_1 = 0.5$ (Radford et al., 2015), damit sich die Schätzungen des ersten Moments schnell an den wechselnden Gegner anpassen:

<!--en:3d9ecd5c-->
::: {.slide title="GAN-Trainingsschleife mit abwechselnden Updates (Forts.)" only="pytorch"}

<!--en:d7e2024d-2-->
[Trainingsinfrastruktur]{.kicker}

<!--en:c0087243-->
Pro Minibatch neue latente Vektoren ziehen, je einen Kritiker- und einen Generatorschritt machen und die Verluste auf dem Gerät summieren:

<!--en:6908984b-->
::: {.slide title="GAN-Trainingsschleife mit abwechselnden Updates" only="jax"}

<!--en:d7e2024d-3-->
[Trainingsinfrastruktur]{.kicker}

<!--en:f84e15b8-2-->
`train_gan` führt für jeden Minibatch abwechselnd `update_D` und `step_G` aus. Adam verwendet ein reduziertes Momentum $\beta_1 = 0.5$ (Radford et al., 2015), damit sich die Schätzungen des ersten Moments schnell an den wechselnden Gegner anpassen:

<!--en:9a58562a-->
::: {.slide title="GAN-Trainingsschleife mit abwechselnden Updates (Forts.)" only="jax"}

<!--en:d7e2024d-4-->
[Trainingsinfrastruktur]{.kicker}

<!--en:c0087243-2-->
Pro Minibatch neue latente Vektoren ziehen, je einen Kritiker- und einen Generatorschritt machen und die Verluste auf dem Gerät summieren:

<!--en:39216bea-->
::: {.slide title="Benchmark 1: Linearer Generator für eine schräg ausgerichtete 2D-Gauß-Verteilung"}

<!--en:e3a422f9-->
[Überprüfung in geschlossener Form]{.kicker}

<!--en:1baf7ead-->
Quantitativer Test der theoretischen Vorhersagen: $1{,}000$ Punkte nach $x = zA + b$ ($z \sim \mathcal{N}(0, I_2)$) erzeugen. Damit gilt $p = \mathcal{N}(b, A^\top A)$. Eine einzelne lineare Schicht $G(z) = zA_q + \mu_q$ trainieren:

<!--en:4ab4f9f3-->
- **Gauß-Verteilung $q$ in geschlossener Form:** Da $G$ linear ist, sind Mittelwert und Kovarianz von $q = \mathcal{N}(\mu_q, A_q^\top A_q)$ in jedem Trainingsschritt exakt bekannt. Das Training selbst wertet $q(x)$ nie aus.

<!--en:7761c022-->
- **Nichtlinearer Kritiker erforderlich:** Wegen $\Sigma_p \neq \Sigma_q$ ist das wahre logarithmierte Dichteverhältnis $\lambda(x) = \log(p(x)/q(x))$ quadratisch in $x$. Erforderlich ist daher ein dreischichtiges Leaky-ReLU-MLP als Kritiker ($2 \to 16 \to 16 \to 1$, Steigung $0.2$).

<!--en:0627d10f-->
::: {.slide title="Konvergenz zum Gleichgewicht bei der 2D-Gauß-Verteilung"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:0246a421-->
Trainingsdauer $20$ Epochen bei Batchgröße $8$ (`lr_D = 0.05`, `lr_G = 0.005`, damit der Kritiker seiner optimalen Antwort folgt). Beide Verluste erreichen ihre theoretischen Gleichgewichtswerte innerhalb von $10$ Epochen:

<!--en:515b310e-->
- **Verluste im Gleichgewicht:** Der Kritikerverlust $-V(D)$ steigt auf $2\log 2 \approx 1.386$ (zufälliges Raten). Der Generatorverlust pendelt sich bei $\log 2 \approx 0.693$ ein.

<!--en:555f43b4-->
- **Exakte Anpassungsgüte in Nat:** Die geschlossene Formel für die KL-Divergenz multivariater Gauß-Verteilungen bestätigt, dass $D_{\textrm{KL}}(q \,\|\, p)$ von anfänglich $7.5\text{--}7.9$ Nat auf $0.02\text{--}0.06$ Nat sinkt:

<!--en:e982d74e-->
::: {.slide title="Rekonstruiert der trainierte Kritiker log(p/q)?"}

<!--en:94749d80-->
[Überprüfung des Dichteverhältnisses]{.kicker}

<!--en:e9aef6fa-->
Einen gestörten Generator $q$ ($D_{\textrm{KL}}(q \,\|\, p) \approx 3.4$ Nat) festhalten und einen neuen Leaky-ReLU-Kritiker über $2{,}000$ Schritte gegen ihn trainieren. Der Test zeigt, ob $D(x)$ das analytische logarithmierte Dichteverhältnis $\lambda(x) = \log(p(x)/q(x))$ rekonstruiert:

<!--en:cb51a299-->
- **Hohe Genauigkeit im Überlappungsbereich:** An $200$ echten und $200$ generierten Punkten folgt $D(x)$ dem Wert $\lambda(x)$ mit Korrelation $\approx 0.98$ und einem Anteil erklärter Varianz von $R^2 > 0.90$:

<!--en:a05c970d-->
- **Warum die Kurve bei extrem negativen Werten abflacht ($\lambda \ll 0$):** Das stückweise lineare Leaky-ReLU-Netz extrapoliert in den Randbereichen linear statt quadratisch. Sobald $D(x') \ll 0$ gilt, verschwindet der Gradient $\sigma(D(x')) \approx e^{D(x')}$ des logarithmischen Kritikerverlusts.

<!--en:4a7f9131-->
::: {.slide title="Sättigender und nicht sättigender Verlust bei weit entferntem Startpunkt"}

<!--en:337d8cee-->
[Kontrollierte Ablationsstudie]{.kicker}

<!--en:4cf9c11d-->
Nach einer Verschiebung des anfänglichen Offsets des Generators um $(-6, 6)$ liegt $q$ etwa $8$ Einheiten von den Daten entfernt ($D_{\textrm{KL}}(q \,\|\, p) \approx 345$ Nat). Je drei Läufe mit verschiedenen Zufallsstartwerten und jeweils identischen Minibatches isolieren den Einfluss des Generatorgewichts:

<!--en:f3f1e1fe-->
- Bereits in Epoche $2$ trennen beide Kritiker die weit auseinanderliegenden Punktwolken mühelos ($D(x') \ll 0$, Kritikerverlust $\approx 0$).

<!--en:0bffb05a-->
- **Generator mit sättigendem Verlust ($w_{\textrm{sat}} = \sigma(D(x')) \approx 0$):** Kommt nahe dem Startpunkt völlig zum Stillstand (nach 20 Epochen noch $335\text{--}340$ Nat).

<!--en:4649e9b3-->
- **Generator mit nicht sättigendem Verlust ($w_{\textrm{ns}} = \sigma(-D(x')) \approx 1$):** Erhält die räumliche Steigung $\nabla_x D(x')$ des Leaky-ReLU-Kritikers. Die Divergenz sinkt bis auf wenige Hundertstel Nat.

<!--en:7891f162-->
::: {.slide title="Ein MLP-GAN für Fashion-MNIST-Bilder"}

<!--en:70f46527-->
[Synthese in hohen Dimensionen]{.kicker}

<!--en:1ba5b30c-->
Auf $60{,}000$ Fashion-MNIST-Bildern ($28 \times 28 = 784$ Pixel, skaliert auf $[-1, 1]$) lernen Leaky-ReLU-MLPs ($G$: $100 \to 256 \to 512 \to 784$ mit $\tanh$, $D$: $784 \to 512 \to 256 \to 1$) über $20$ Epochen (Adam mit $\text{lr} = 2 \times 10^{-4}, \beta_1 = 0.5$, Batchgröße $64$):

<!--en:8f3d8f35-->
- **Bildqualität:** Der MLP-Generator bildet erkennbare Silhouetten von Hosen, Hemden, Schuhen und Taschen nach. Die Pixelstruktur ist körnig, weil dichte Schichten die räumliche Nachbarschaft im zweidimensionalen Bild außer Acht lassen.

<!--en:9c52226f-->
- **Bleibender Abstand zum Gleichgewicht:** Nach $20$ Epochen liegt der Kritikerverlust ($\sim 1.05$) weiterhin unter $2\log 2 \approx 1.39$ und der Generatorverlust ($\sim 1.4$) über $\log 2 \approx 0.69$.

<!--en:44c64906-->
- **Mehrdeutige Diagnose:** Ohne Dichten in geschlossener Form lässt sich aus den Verlustkurven allein nicht erkennen, ob der Generator ungenaue Bilder erzeugt, der Kritiker die Daten auswendig lernt oder Bekleidungsklassen fehlen.

<!--en:443611b0-->
::: {.slide title="Verständnisfragen zu optimalen Kritikern und Gradientengewichten"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:c0662d79-->
Fragen zur klassischen Minimax-Analyse für GANs:

<!--en:5b39da61-->
1. **Verhältnis der Gradientengewichte:** Für ein generiertes Sample liefert der Diskriminator das Logit $D(x') = -5$. Um welchen Faktor ist das Gradientengewicht dieses Samples beim nicht sättigenden Verlust größer als beim sättigenden Minimax-Verlust?

<!--en:503d8f60-->
2. **Kritikerverlust auf Gleichgewichtsniveau:** Am Ende eines Trainingslaufs beträgt der Diskriminatorverlust $2\log 2 \approx 1.386$. Beweist das, dass die Generatorverteilung $q$ mit der Datenverteilung $p$ übereinstimmt?

<!--en:bb3f1034-->
3. **Sättigende versteckte Einheiten des Kritikers:** Im Gauß-Beispiel mit weit entferntem Startpunkt die Leaky-ReLU-Aktivierungen des Kritikers durch $\tanh$-Einheiten ersetzen. Warum kommt selbst der Generator mit nicht sättigendem Verlust zum Stillstand?

<!--en:557b98b6-->
**Lösungen:**

<!--en:e9aab23e-->
1. Das Verhältnis der Gewichte beträgt $\frac{w_{\textrm{ns}}(x')}{w_{\textrm{sat}}(x')} = \frac{\sigma(-D(x'))}{\sigma(D(x'))}$ $= e^{-D(x')} = e^5 \approx \mathbf{148.4\times}$ (und $e^3 \approx 20.1\times$ bei $D(x') = -3$).

<!--en:aa6408b7-->
2. **Nein.** Jeder konstante Kritiker $D(x) = 0$ (z. B. ein unzureichend trainierter, inaktiver oder stark regularisierter Kritiker) erreicht den Verlust $2\log 2$, unabhängig vom Abstand zwischen $q$ und $p$. Man muss die Samples oder die Abstände im Merkmalsraum direkt auswerten.

<!--en:8b4e2c35-->
3. Das nicht sättigende Gewicht $w_{\textrm{ns}}(x') = \sigma(-D(x')) \approx 1$ skaliert lediglich den räumlichen Gradienten $\nabla_\theta D(G_\theta(z))$. Gehen die versteckten $\tanh$-Einheiten des Kritikers bei $\pm 1$ in Sättigung, so gelten $\nabla_x D(x') \approx \mathbf{0}$ und $1 \times \mathbf{0} = \mathbf{0}$.

<!--en:bd595cb9-->
::: {.slide title="Zusammenfassung des klassischen Minimax-GAN"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:b2deb230-->
| Begriff oder Komponente | Mathematischer Ausdruck | Gleichgewicht oder Wertebereich | Bedeutung und Schwierigkeiten in der Praxis |

<!--en:a5240a60-->
| **Minimax-Wertfunktion** | $V(D) = \mathbb{E}_p[\log\sigma(D(x))] + \mathbb{E}_q[\log\sigma(-D(x'))]$ | $V(D^\star) = -2\log 2 \approx -1.386$ | Binäre Kreuzentropie für gleich viele echte und generierte Beispiele |

<!--en:c4d5fb5c-->
| **Optimales Kritiker-Logit** | $D^\star(x) = \log \frac{p(x)}{q(x)} = \lambda(x)$ | $D^\star(x) = 0 \iff q(x) = p(x)$ | Schätzt das logarithmierte Dichteverhältnis allein aus Stichproben |

<!--en:254f73a9-->
| **Divergenz auf der Grundgesamtheit** | $\max_D V(D) = 2\,\mathrm{JS}(p, q) - 2\log 2$ | $\mathrm{JS}(p, q) = I(x; y) \in [0, \log 2]$ | Sättigt bei $\log 2$, sobald die Träger disjunkt sind |

<!--en:90e38f4d-->
| **Sättigendes $G$-Gewicht** | $w_{\textrm{sat}}(x') = \sigma(D(x'))$ | $w_{\textrm{sat}} = \frac{1}{2}$ bei $D = 0$ | Fällt wie $e^{D(x')} \to 0$ für $D(x') \ll 0$, sodass der Generator fern der Daten zum Stillstand kommt |

<!--en:28de8998-->
| **Nicht sättigender $G$-Verlust** | $-\mathbb{E}_z[\log\sigma(D(G(z)))]$, $\;w_{\textrm{ns}}(x') = \sigma(-D(x'))$ | Verringert $2\,D_{\textrm{KL}}(m \,\|\, p)$ bei $D^\star$ | Erhält das Gewicht nahe eins für $D(x') \ll 0$, in der Praxis Standard |

<!--en:96f1a3b9-->
| **Modusausfall** | $q$ deckt nur einen Teil der Modi von $p$ ab | Nur $\approx 0.26$ Nat für das Auslassen der Hälfte von $p$ | $\nabla_x D^\star = \mathbf{0}$ auf abgedeckten Modi zieht die Samples nicht zu fehlenden Modi |
