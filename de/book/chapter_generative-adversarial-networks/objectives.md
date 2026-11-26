<!--en:b0bebc98-->
# Adversariale Zielfunktionen und Divergenzen

<!--en:f7a226f3-->
Die meisten GAN-Implementierungen bieten mehrere austauschbare Zielfunktionen für das Training an: den nicht sättigenden logistischen Verlust, kleinste Quadrate (LSGAN), den Hinge-Verlust und den Wasserstein-Verlust. In :numref:`sec_basic_gan` haben wir die logistische Zielfunktion untersucht. Ihr optimaler Diskriminator liefert das logarithmische Dichteverhältnis (engl. log density ratio) $\lambda(x) = \log(p(x)/q(x))$. Ihr Minimax-Wert entspricht bis auf Skalierung und Verschiebung der Jensen-Shannon-Divergenz. Diese sättigt, sobald sich die Träger von $p$ und $q$ nicht mehr überlappen. Welche Divergenzen messen die anderen Zielfunktionen? Wenn die generierten Samples weit von den Daten entfernt liegen, woran liegt dann ein verschwindender Gradient: an der Verlustfunktion (engl. loss function), der Architektur des Kritikers oder der zugrunde liegenden Divergenz?

<!--en:1f477316-->
Die Zielfunktionen dieser Familie unterscheiden sich entweder darin, wie sie die Ausgaben des Kritikers bewerten, oder in der Funktionenklasse $\mathcal{T}$, über die der Kritiker optimiert. Ohne Nebenbedingungen messen Kritiker mit Klassifikationsverlusten (engl. classification losses) $f$-Divergenzen. Diese hängen nur vom punktweisen Dichteverhältnis $p(x)/q(x)$ ab. Beschränken wir den Kritiker auf eine Klasse glatter Funktionen und verwenden lineare Auszahlungen, erhalten wir *integrale Wahrscheinlichkeitsmetriken* (engl. integral probability metrics, IPMs). Dazu zählen die Maximum Mean Discrepancy (MMD) und die Wasserstein-1-Distanz ($W_1$). Beide erfassen räumliche Abstände auch bei disjunkten Trägern. Wir leiten beide Familien her und vergleichen ihr Verhalten an weit auseinanderliegenden eindimensionalen Gauß-Verteilungen und einer zweidimensionalen Gauß-Mischung.

<!--en:7e9338cb-->
## Allgemeine Formulierung

<!--en:e7777891-->
In :numref:`sec_basic_gan` erhielt der Kritiker für echte Samples die Auszahlung $\log \sigma(D(x))$ mit $x \sim p$ und für generierte Samples die Auszahlung $\log \sigma(-D(x'))$ mit $x' \sim q$. Ersetzen wir diese Log-Likelihood-Terme durch allgemeine Auszahlungsfunktionen $\ell_p$ und $\ell_q$ für eine Kritikerklasse $\mathcal{T}$, erhalten wir die allgemeine adversariale Diskrepanz:

<!--en:3f48aaea-->
Setzen wir $\ell_p(t) = \log \sigma(t)$ und $\ell_q(t) = \log \sigma(-t)$ und wählen $\mathcal{T}$ ohne Nebenbedingungen, erhalten wir wieder $2\,\mathrm{JS}(p, q) - 2\log 2$. In diesem Abschnitt maximiert der Kritiker stets die Summe der erwarteten Auszahlungen. Gleichbedeutend damit ist die Minimierung der negativen Summe, die wir als Kritikerverlust bezeichnen.

<!--en:25de18c5-->
:numref:`tab_gan_objective_families` ordnet die Zielfunktionen dieses Kapitels nach zwei Kriterien: Bewertet der Kritiker einzelne Samples oder Paare aus je einem echten und einem generierten Sample? Ist die Kritikerklasse $\mathcal{T}$ frei wählbar oder regularisiert?

<!--en:8659ed60-->
:Einteilung adversarialer Zielfunktionen nach Eingabe des Kritikers und Nebenbedingung an die Funktionenklasse.

<!--en:75ec0b3e-->
| | Kritiker ohne Nebenbedingung | Kritiker mit Nebenbedingung oder Strafterm |

<!--en:e56c1697-->
| **Kritiker für einzelne Samples** | Klassifikationsverluste (logistischer Verlust, kleinste Quadrate, Hinge-Verlust) $\to$ $f$-Divergenzen von $p/q$ | lineare Auszahlungen auf einer RKHS-Kugel (MMD) oder einer Lipschitz-Kugel ($W_1$), Hinge-Verlust mit spektraler Normalisierung |

<!--en:5c5dbbe8-->
| **Kritiker für Sample-Paare** | relativistischer Verlust für Paare (RpGAN, :numref:`sec_gan_relativistic`) | Verlust für Paare mit um null zentrierten Gradientenstrafen $R_1 + R_2$ (R3GAN, :numref:`sec_gan_convergence`, :numref:`sec_dcgan`) |

<!--en:eced794a-->
Für jeden festen Kritiker $T$ ist der Erwartungswert in :eqref:`eq_gan_template` linear in der Verteilung $q$. Das Supremum einer Familie linearer Funktionale ist konvex. Deshalb ist $d(p, q)$ im Wahrscheinlichkeitsmaß $q$ bei festem $p$ stets konvex. Die Menge der Verteilungen $\{q_\theta\}$, die ein neuronales Netz $G_\theta$ parametrisiert, ist jedoch nicht konvex. Aus der Konvexität im Raum der Verteilungen folgt daher keine Konvexität in den Generatorparametern $\theta$ (Aufgabe :exref:`ex_objectives-convexity-parameterization`).

<!--en:88d4aba8-->
## Klassifikationsverluste und $f$-Divergenzen

<!--en:9947de54-->
### Beispiel mit zwei möglichen Ergebnissen

<!--en:5eed8745-->
Betrachten wir einen Münzwurf mit den Ergebnissen Kopf und Zahl. Wir wählen zwischen der Datenverteilung $p = (\tfrac34, \tfrac14)$ und der Generatorverteilung $q = (\tfrac14, \tfrac34)$ mit derselben A-priori-Wahrscheinlichkeit $\tfrac12$ und ziehen daraus ein Ergebnis. Ist das Ergebnis bekannt, beträgt die A-posteriori-Wahrscheinlichkeit dafür, dass es aus $p$ stammt, nach Bayes $\eta = p/(p + q)$. Für Kopf ergibt sich $\tfrac34$ und für Zahl $\tfrac14$. Eine blinde Vorhersage ($D = 0$) liefert unabhängig vom Ergebnis $\tfrac12$. Wie stark die Beobachtung des Ergebnisses den erwarteten Verlust senkt, hängt vom Klassifikationsverlust ab:

<!--en:0ff1989c-->
- *Logarithmischer Verlust*: Die optimale Vorhersage $\eta$ erreicht die binäre Entropie $H_b(\tfrac34) = 0.562$ Nat gegenüber $\log 2 = 0.693$ Nat beim blinden Raten. Der Verlust sinkt um $0.131\text{ nats} = \mathrm{JS}(p, q)$.

<!--en:edfdd385-->
- *Quadratischer Fehler* der Wahrscheinlichkeiten: Die Vorhersage $\eta$ ergibt den erwarteten Verlust $\tfrac34 \cdot \tfrac14 = \tfrac{3}{16}$ gegenüber $\tfrac14$ bei der Vorhersage $\tfrac12$. Der Verlust sinkt um $\tfrac{1}{16}$.

<!--en:998eb833-->
- *Hinge-Verlust* für einen Score $t$: Die optimalen Scores $t = +1$ (Kopf) und $t = -1$ (Zahl) ergeben den erwarteten Verlust $2 \cdot \tfrac14 = \tfrac12$ gegenüber $1$ bei $t = 0$. Der Verlust sinkt um $\tfrac12 = \mathrm{TV}(p, q) = \tfrac12 \sum |p - q|$.

<!--en:c3cf71c3-->
Jeder Verlust ordnet demselben Paar $(p, q)$ auf Grundlage derselben A-posteriori-Wahrscheinlichkeit $\eta$ eine andere nichtnegative Divergenz zu. Wir zeigen nun, dass jeder Klassifikationsverlust auf Basis einer Margin eine $f$-Divergenz definiert.

<!--en:470f8b73-->
### Differenz der Bayes-Risiken

<!--en:20c50e8f-->
Symmetrische Klassifikationsverluste verwenden dieselbe Auszahlungsfunktion $\ell$ für die Margin. Bei echten Samples wenden sie diese auf $D(x)$ an, bei generierten auf $-D(x')$ ($\ell_p(t) = \ell(t)$, $\ell_q(t) = \ell(-t)$):

<!--en:28a91d63-->
Der vorzeichenbehaftete Score $t$ ($+D(x)$ für echte Daten, $-D(x')$ für generierte Daten) ist die *Margin* der Klassifikation. An jedem Punkt $x$ gilt $p(x)\ell(t) + q(x)\ell(-t) = 2m(x)\big[\eta(x)\ell(t) + (1 - \eta(x))\ell(-t)\big]$ mit $m = (p+q)/2$ und $\eta(x) = p(x)/(p(x)+q(x))$. Der kleinste punktweise erwartete Verlust bei gegebener A-posteriori-Wahrscheinlichkeit $\eta$ heißt *bedingtes Bayes-Risiko* (engl. conditional Bayes risk):

<!--en:d1b97b41-->
Als Infimum affiner Funktionen von $\eta$ ist $L(\eta)$ auf $[0, 1]$ konkav. Ein konstanter Kritiker ($t = 0$) erreicht das Risiko $L(\tfrac12)$. Um wie viel die Beobachtung von $x$ das Risiko senkt, misst die *Differenz der Bayes-Risiken* (engl. Bayes-risk gap):

<!--en:ebfced29-->
Diese Differenz bestimmt den optimalen Wert $\sup_D V_\ell(D) = 2\Delta_\ell(p, q) - 2L(\tfrac12)$. Aus der Jensen-Ungleichung und $\mathbb{E}_m[\eta] = \tfrac12$ folgt $\Delta_\ell(p, q) \geq 0$.

<!--en:0e2a300a-->
### Differenzen der Bayes-Risiken als $f$-Divergenzen

<!--en:6fef01ab-->
Wie wir aus :numref:`sec_mdl-f-divergences` wissen, mittelt eine $f$-Divergenz eine konvexe Funktion $f$ (mit $f(1) = 0$) des Dichteverhältnisses $\rho(x) = p(x)/q(x)$ unter $q$: $D_f(p \,\|\, q) = \mathbb{E}_{x \sim q}[f(\rho(x))]$.

<!--en:8f70fb36-->
**Proposition.** *Ist das bedingte Bayes-Risiko $L$ auf $[0, 1]$ endlich, so gilt*

<!--en:23e6a157-->
*wobei $f$ auf $[0, \infty)$ konvex ist und $f(1) = 0$ erfüllt.*

<!--en:d3aa66ae-->
**Beweis.** Setzen wir $u = \rho = p/q$ ein, so erhalten wir $m = q\,(u+1)/2$ und $\eta = u/(u+1)$. Damit gilt

<!--en:c8ed785d-->
Ziehen wir $L(\tfrac12) = \int q\, L(\tfrac12)$ unter das Integral, ergibt sich $\Delta_\ell(p, q) = \mathbb{E}_q[f(\rho)]$. Schreiben wir die konkave Funktion $L(v) = \inf_j \{\alpha_j v + \beta_j\}$ als untere Einhüllende ihrer Tangenten, so sehen wir, dass $(u+1)L(u/(u+1)) = \inf_j \{\alpha_j u + \beta_j(u+1)\}$ in $u$ konkav ist. Folglich ist $f(u)$ konvex, und es gilt $f(1) = L(\tfrac12) - L(\tfrac12) = 0$. $\blacksquare$

<!--en:891a5aec-->
Jeder symmetrische Klassifikationsverlust misst also bei einem Kritiker ohne Nebenbedingungen eine $f$-Divergenz. Der optimale Kritiker $D^\star(x)$ ist eine monotone *Linkfunktion* des logarithmischen Dichteverhältnisses $\lambda(x) = \log(p(x)/q(x))$.

<!--en:1c965359-->
### Divergenzen gängiger Klassifikationsverluste

<!--en:0fa5b553-->
:numref:`tab_gan_loss_divergences` gibt einen Überblick über vier gängige Klassifikationsverluste, ihre bedingten Bayes-Risiken $L(\eta)$, die Differenzen der Bayes-Risiken $\Delta_\ell(p, q)$, die optimalen Linkfunktionen der Kritiker $D^\star(\lambda)$ und die zugehörigen Generatorverluste in der Praxis.

<!--en:aa607e3f-->
:Klassifikationsverluste für einen Kritiker ohne Nebenbedingungen: Margin-Verlust $-\ell(t)$, Bayes-Risiko $L(\eta)$, Differenz der Bayes-Risiken $\Delta_\ell(p, q)$, optimale Linkfunktion des Kritikers $D^\star(\lambda)$ und Generatorverlust in der Praxis für $D = D(x')$.

<!--en:e7af48ed-->
| Verlust | Margin-Verlust $-\ell(t)$ | Bayes-Risiko $L(\eta)$ | Risikodifferenz $\Delta_\ell(p,q)$ | optimale Linkfunktion des Kritikers | Generatorverlust |

<!--en:62955b4c-->
| logistischer Verlust | $\log(1 + e^{-t})$ | $H_b(\eta)$ | $\mathrm{JS}(p,q)$ | $\lambda$ | $-\log \sigma(D)$ |

<!--en:6ddefe3f-->
| kleinste Quadrate (LSGAN) | $\tfrac14 (1 - t)^2$, $t = 2D - 1$ | $\eta(1-\eta)$ | $\tfrac{1}{8}\int \frac{(p-q)^2}{p+q}$ | $\sigma(\lambda)$ | $\tfrac12 (D - 1)^2$ |

<!--en:b68a0e1d-->
| quadrierter Hinge-Verlust | $\max(0, 1 - t)^2$ | $4\eta(1-\eta)$ | $\tfrac{1}{2}\int \frac{(p-q)^2}{p+q}$ | $\tanh(\lambda/2)$ | $-D$ |

<!--en:18a93d8b-->
| Hinge-Verlust | $\max(0, 1 - t)$ | $2\min(\eta, 1-\eta)$ | $\mathrm{TV}(p,q)$ | $\operatorname{sign} \lambda$ | $-D$ |

<!--en:6388f2dc-->
Im LSGAN :cite:`Mao.Li.Xie.ea.2017` lernt der Kritiker durch Regression die Zielwerte $1$ (echt) und $0$ (generiert). Mit $t = 2D - 1$ entspricht dies dem quadratischen Margin-Verlust $\tfrac14(1 - t)^2$ und ergibt ein Viertel der Vincze-Le-Cam-Divergenz $\chi^2(p \,\|\, m) = \tfrac12 \int (p-q)^2/(p+q)$. Der Hinge-Verlust :cite:`Lim.Ye.2017` misst die totale Variation $\mathrm{TV}(p, q) = \tfrac12 \int |p - q|$. Sein optimaler Kritiker liefert $\operatorname{sign} \lambda$ (jeder Wert in $[-1, 1]$ ist bei $\eta = \tfrac12$ optimal). Der quadrierte Hinge-Verlust verwendet dieselbe Margin und dieselbe Generatorzielfunktion $-D(x')$ wie der Hinge-Verlust. Seine glatte Linkfunktion $\tanh(\lambda/2)$ erhält jedoch die Information über den Betrag von $\lambda$. In der Praxis verwendet jeder Verlust eine nicht sättigende Generatorzielfunktion (letzte Spalte). Alle haben dasselbe Gleichgewicht $q = p$.

<!--en:cc8ec623-->
### Variationsschranke nach Fenchel und Young ($f$-GAN)

<!--en:e84895f9-->
Da :eqref:`eq_gan_margin` dieselbe Auszahlung für $D$ und $-D$ verwendet, gilt für das Bayes-Risiko $L(\eta) = L(1 - \eta)$. Die Risikodifferenz jedes solchen Klassifikationsverlusts $\Delta_\ell(p, q)$ ist somit symmetrisch in $p$ und $q$ und nach oben durch $L(\tfrac12)$ beschränkt (Aufgabe :exref:`ex_objectives-divergences-a-loss-can-reach`). Asymmetrische oder unbeschränkte $f$-Divergenzen wie die Vorwärts-KL-Divergenz $D_{\textrm{KL}}(p \,\|\, q)$ erfordern asymmetrische Auszahlungen $\ell_p \neq \ell_q$.

<!--en:4a6bb0e7-->
Sei $f^*(t) = \sup_u \{ut - f(u)\}$ die *konvex Konjugierte* einer konvexen Divergenzfunktion $f$. Nach der Fenchel-Young-Ungleichung gilt $f(u) \geq ut - f^*(t)$ für alle $u, t$, mit Gleichheit für $t = f'(u)$. Wir setzen $u = \rho(x) = p(x)/q(x)$ und $t = T(x)$ und bilden den Erwartungswert unter $q$. Mit $\mathbb{E}_q[\rho\, T] = \mathbb{E}_p[T]$ erhalten wir die untere Variationsschranke des $f$-GAN :cite:`Nowozin.Cseke.Tomioka.2016` (:numref:`sec_mdl-f-gan-dual`):

<!--en:9e3d719f-->
Dies entspricht unserer allgemeinen Formulierung :eqref:`eq_gan_template` mit $\ell_p(t) = t$ und $\ell_q(t) = -f^*(t)$. Für den optimalen Kritiker $T^\star(x) = f'(\rho(x))$ gilt Gleichheit (Aufgabe :exref:`ex_objectives-variational-bound-eqref-eq`). Bei der Vorwärts-KL-Divergenz hat $f(u) = u \log u$ die Konjugierte $f^*(t) = e^{t-1}$ und den optimalen Kritiker $T^\star(x) = 1 + \log \rho(x) = 1 + \lambda(x)$.

<!--en:868e7e2f-->
### Empirische Überprüfung der Linkfunktionen der Kritiker

<!--en:55a85e78-->
Wir prüfen nun, ob Kritiker mit logistischem Verlust, kleinsten Quadraten, Hinge-Verlust und Vorwärts-KL-Zielfunktion ihre jeweilige theoretische Linkfunktion von $\lambda(x)$ lernen: $\lambda$, $\sigma(\lambda)$, $\operatorname{sign} \lambda$ und $1 + \lambda$. Wir wählen für $p$ eine gleich gewichtete Mischung dreier zweidimensionaler Gauß-Verteilungen. Durch Störung der Komponenten konstruieren wir eine Mischung aus 3 Komponenten $q$, sodass wir $\lambda(x)$ überall analytisch kennen.

<!--en:78b24c83-->
Auf einem festen Gitter liefert numerische Quadratur den Wert $\mathrm{JS}(p, q)$ für jede Mischung $q$ dieser Form. Später in diesem Abschnitt bewerten wir mit derselben Funktion nach jedem Trainingsschritt einen Generator.

<!--en:81188469-->
Die von Hand gewählte Verteilung $q$ liegt nur wenige Hundertstel Nat von der Datenverteilung entfernt. Eine Abweichung dieser Größe ist gegen Ende des Generatortrainings typisch. Als Kritiker dient ein kleines Leaky-ReLU-Netz mit zwei versteckten Schichten zu je 64 Einheiten.

<!--en:48ec9dd4-->
Für jede Variante speichern wir den Kritikerverlust und den Generatorverlust aus :numref:`tab_gan_loss_divergences` als Paar. Beide hängen von den Scores des Kritikers ab: `r` für einen echten Batch und `f` für einen generierten.

<!--en:b921cfd1-->
Eine Linkfunktion beschreibt die beste Antwort auf ein festes $q$. Daher trainieren wir für drei der Verluste je einen neuen Kritiker gegen das von Hand gewählte $q$. Einen vierten trainieren wir so, dass er die Vorwärts-KL-Schranke $\mathbb{E}_p[T] - \mathbb{E}_q[e^{T-1}]$ maximiert.

<!--en:08179a25-->
Wir werten jeden trainierten Kritiker an 300 Punkten aus $p$ und 300 Punkten aus $q$ aus. Seine Ausgabe tragen wir gegen das wahre logarithmische Dichteverhältnis $\lambda(x)$ auf und zeichnen die theoretische Linkfunktion gestrichelt ein.

<!--en:3b107ff7-->
Alle vier Netze folgen der jeweils theoretisch vorhergesagten Linkfunktion von $\lambda(x)$: der Identität $\lambda$, der Sigmoidfunktion $\sigma(\lambda)$, der Sprungfunktion $\operatorname{sign} \lambda$ und der verschobenen Identität $1 + \lambda$. Die Ausgaben des Hinge-Kritikers streuen in der Nähe von $\lambda = 0$ stärker. Dort erreicht jede Ausgabe in $[-1, 1]$ das minimale bedingte Risiko.

<!--en:6cad5a0b-->
## Integrale Wahrscheinlichkeitsmetriken

<!--en:ebe34999-->
Bei $f$-Divergenzen kombinieren wir einen Kritiker ohne Nebenbedingungen mit nichtlinearen Auszahlungen. *Integrale Wahrscheinlichkeitsmetriken* (IPMs) verwenden lineare Auszahlungen $\ell_p(t) = t$ und $\ell_q(t) = -t$ und beschränken den Kritiker auf eine symmetrische Funktionenklasse $\mathcal{F}$ (:numref:`sec_mdl-ipm-mmd`):

<!--en:985bbdcf-->
Ohne Nebenbedingung an $\mathcal{F}$ könnten wir $T$ mit einer beliebig großen positiven Konstanten multiplizieren. Die Differenz der Erwartungswerte würde dann gegen $+\infty$ gehen, sofern $p \neq q$ gilt. Besteht $\mathcal{F}$ aus glatten Funktionen, so hängt $d_{\mathcal{F}}(p, q)$ von der räumlichen Geometrie von $\mathcal{X}$ ab und nicht allein vom punktweisen Dichteverhältnis $p(x)/q(x)$.

<!--en:237826fe-->
### Maximum Mean Discrepancy (MMD)

<!--en:9ee89f9c-->
Gegeben sei ein positiv definiter Kernel $k(x, x')$, zum Beispiel der gaußsche RBF-Kernel $k(x, x') = \exp(-\|x - x'\|^2/(2s^2))$ mit Bandbreite $s$. Wählen wir für $\mathcal{F}$ die Einheitskugel des Hilbertraums mit reproduzierendem Kern $k$ (engl. reproducing kernel Hilbert space, RKHS), so erhalten wir die *Maximum Mean Discrepancy* :cite:`Gretton.Borgwardt.Rasch.ea.2012` (:numref:`sec_mdl-ipm-mmd`). Ihr Quadrat lässt sich in geschlossener Form durch Erwartungswerte paarweiser Kernel-Auswertungen ausdrücken:

<!--en:77f3fd92-->
Die bis auf Skalierung optimale Testfunktion ist die *Zeugenfunktion* (engl. witness function) $w(x) = \mathbb{E}_{\tilde x \sim p}[k(x, \tilde x)] - \mathbb{E}_{\tilde x' \sim q}[k(x, \tilde x')]$. Sie misst für einen Punkt die Differenz seiner mittleren Ähnlichkeit zu Punkten aus $p$ und seiner mittleren Ähnlichkeit zu Punkten aus $q$. Da :eqref:`eq_gan_mmd2` kein Kritikernetz erfordert, kann der Generator den erwartungstreuen Schätzer `mmd2_unbiased` für endliche Stichproben direkt minimieren. In :numref:`sec_dcgan` berechnen wir damit auch die KID.

<!--en:3822fbdf-->
### Wasserstein-1-Distanz ($W_1$)

<!--en:730a3889-->
Wählen wir für $\mathcal{F}$ die Klasse der *1-Lipschitz-stetigen* Funktionen ($|T(x) - T(y)| \leq \|x - y\|$ für alle $x, y$), so liefert die Kantorovich-Rubinstein-Dualität die *Wasserstein-1-Distanz* (auch *Earth Mover’s Distance*, :numref:`sec_mdl-optimal-transport`):

<!--en:58e2ba42-->
Dabei bezeichnet $\Pi(p, q)$ die Menge der gemeinsamen Transportpläne mit Randverteilungen $p$ und $q$. Ein optimaler 1-Lipschitz-stetiger Kritiker (ein *Kantorovich-Potential*) hat entlang der optimalen Transportstrahlen von $q$ nach $p$ die Gradientennorm eins. In einer Dimension gilt $W_1(p, q) = \int_{-\infty}^\infty |F_p(t) - F_q(t)|\,dt$. Verschieben wir eine Verteilung um $\mu$, so erhalten wir $W_1(p, q) = |\mu|$ mit konstanter Steigung eins bei jedem Abstand.

<!--en:db599f66-->
In höheren Dimensionen verwenden Wasserstein-GANs (WGANs) vor allem drei Verfahren, um die 1-Lipschitz-Bedingung für einen durch ein neuronales Netz dargestellten Kritiker $T$ näherungsweise durchzusetzen:

<!--en:1a69e02c-->
1. Die *Gewichtsbegrenzung* (engl. weight clipping) :cite:`Arjovsky.Chintala.Bottou.2017` beschränkt die Gewichte auf $[-c, c]$. Damit erzwingt sie $\kappa$-Lipschitz-Stetigkeit mit einer architekturabhängigen Konstanten $\kappa$, schränkt aber die Kapazität des Netzes ein.

<!--en:2fc08a11-->
2. Die *Gradientenstrafe* (engl. gradient penalty, WGAN-GP) :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017` addiert den Term $\gamma\,\mathbb{E}[(\|\nabla_x T(\tilde x)\| - 1)^2]$ für zufällig gewählte Zwischenpunkte $\tilde x = u x + (1-u)x'$ auf den Strecken zwischen echten und generierten Samples.

<!--en:7f48a98e-->
3. Die *spektrale Normalisierung* :cite:`Miyato.Kataoka.Koyama.ea.2018` teilt jede Gewichtsmatrix durch ihren größten Singulärwert $\sigma_{\max}(W)$. Damit beschränkt sie die Lipschitz-Konstante jeder linearen Schicht nach oben durch $1$.

<!--en:0191e76e-->
Der Hinge-Verlust verbindet beide Familien: Für $|D(x)| \leq 1$ sind seine Auszahlungen linear. Ein Hinge-Kritiker ohne weitere Nebenbedingungen entspricht daher der IPM über Funktionen mit $|D(x)| \leq 1$ (der Zielfunktionswert beträgt $2\,\mathrm{TV}(p, q) - 2$). Kombinieren wir den Hinge-Verlust wie in SAGAN und BigGAN mit spektraler Normalisierung, so ist $D$ sowohl beschränkt als auch $\kappa$-Lipschitz-stetig. Der Wert des Kritikers wächst dann linear wie $\kappa\,W_1$ bis zum Abstand $2/\kappa$ und sättigt danach.

<!--en:2ffc5766-->
### Stichprobenkomplexität in hohen Dimensionen

<!--en:63570ae4-->
Die MMD mit festem Kernel und die empirische Distanz $W_1$ unterscheiden sich deutlich in ihrer Stichprobenkomplexität (engl. sample complexity). Der erwartungstreue Schätzer für $\mathrm{MMD}^2$ konvergiert mit der dimensionsunabhängigen Rate $O(n^{-1/2})$ :cite:`Gretton.Borgwardt.Rasch.ea.2012`. Die geschätzte Größe selbst, also $\mathrm{MMD}^2$ für eine gegebene Verschiebung bei festem Kernel, kann allerdings mit wachsender Dimension abnehmen :cite:`Ramdas.Reddi.Poczos.ea.2015`. Die empirische $W_1$-Distanz, bei der wir die empirischen Verteilungen einsetzen, leidet in Dimension $d_x > 2$ mit der Rate $O(n^{-1/d_x})$ unter dem Fluch der Dimensionalität :cite:`Weed.Bach.2019`. Variationsschätzer für $f$-Divergenzen können wiederum eine hohe Varianz aufweisen, wenn $\rho = p/q$ groß ist. In der dualen Schranke für die Vorwärts-KL-Divergenz hat etwa $e^{T^\star(x') - 1} = \rho(x')$ unter $q$ die Varianz $\chi^2(p \,\|\, q)$. Sie kann selbst dann unendlich sein, wenn $D_{\textrm{KL}}(p \,\|\, q) < \infty$ gilt (Aufgabe :exref:`ex_objectives-estimation-and-dimension`).

<!--en:fee94940-->
## Gradienten fern und nah an den Daten

<!--en:f83e453c-->
### Steigung der Divergenz in Abhängigkeit vom Abstand

<!--en:8cdbd260-->
Wie verändern sich Divergenzen auf Populationsebene, wenn zwei Verteilungen auseinanderrücken? Dazu betrachten wir $p = \mathcal{N}(0, 1)$ und $q = \mathcal{N}(\mu, 1)$ in Abhängigkeit vom Abstand $\mu > 0$. Hier hat $W_1(p, q) = \mu$ die konstante Ableitung $1$. Für $\mathrm{JS}(p, q)$ können wir numerisch integrieren. Für $\mathrm{MMD}^2(p, q)$ mit einem gaußschen RBF-Kernel der Bandbreite $s$ ergibt das Gauß-Integral exakt:

<!--en:26b9c8c2-->
Die Ableitung nach $\mu$ erreicht bei $\mu = \sqrt{s^2 + 2}$ ihr Maximum ($\approx 4.2$ für $s = 4$) und klingt danach wie $e^{-\mu^2/(2(s^2+2))}$ ab.

<!--en:3a678a95-->
Wie das Diagramm und :numref:`fig_gan_critics_separated` zeigen, sinkt die Ableitung von $\mathrm{JS}(p, q)$ bis $\mu = 7$ um mehr als zwei Größenordnungen (auf $0.0013$ gegenüber $0.20$ bei $\mu = 1$). Sie erreicht $1.5 \times 10^{-22}$ bei $\mu = 20$. Dagegen behält $W_1$ bei jedem Abstand die Steigung eins. Auch $\mathrm{MMD}^2$ liefert bis zu Abständen in der Größenordnung der Kernel-Bandbreite $s$ einen starken Gradienten.

<!--en:edb4e05b-->
![Optimale Kritiker für $p$ (blau) und $q$ (orange) mit disjunkten Trägern. Links: Der optimale JS-Kritiker $\log(p/q)$ hat den Wert $+\infty$ auf $\operatorname{supp}(p)$ und $-\infty$ auf $\operatorname{supp}(q)$. Auf beiden Trägern ist die Steigung null. Mitte: Das Kantorovich-Potential für $W_1$ ist eine Gerade mit Steigung eins. Rechts: Die MMD-Zeugenfunktion $\mathbb{E}_p[k(x, \cdot)] - \mathbb{E}_q[k(x, \cdot)]$ ist die glatte Differenz zweier glockenförmiger Kernel. Ihr äußerer Ausläufer steigt in Richtung $p$ an.](../img/mdl-gan-critics-separated.svg)

<!--en:98df1a64-->
### Trainierte Kritiker bei großem Anfangsabstand

<!--en:77f76371-->
Theoretische Divergenzkurven setzen einen optimalen Kritiker aus der Klasse aller messbaren Funktionen voraus. In der Praxis ist der Kritiker ein endliches neuronales Netz, das pro Generatoraktualisierung nur wenige Trainingsschritte macht. Wir untersuchen sein Verhalten bei großem Anfangsabstand mit einem eindimensionalen Generator mit Lageparameter $q_\theta = \mathcal{N}(\theta, 1)$ ($x' = \theta + z$). Ausgehend von $\theta_0 = 6$ passen wir ihn an $p = \mathcal{N}(0, 1)$ an. Wir aktualisieren $\theta$ mit gewöhnlichem SGD, sodass die Schrittweiten die Gradientenbeträge direkt widerspiegeln. Der ReLU-Kritiker hat 2 versteckte Schichten und macht 5 Aktualisierungen pro Generatorschritt.

<!--en:fcc72060-->
`train_shift` führt für jeden Generatorschritt fünf Adam-Schritte für den Kritiker aus. So bleibt der Kritiker nahe seiner besten Antwort. Der Generator macht gewöhnliche Gradientenschritte mit Lernrate $0.01$. Adam würde jeden Schritt ungefähr auf die Lernrate skalieren und damit gerade den Gradientenbetrag verdecken, den wir untersuchen wollen. Die Funktion zeichnet bei jedem Schritt $\theta$ und den Gradienten des Generators auf. Ein positiver Wert für `gp` ergänzt den Kritikerverlust um die Gradientenstrafe. Mit `one_sided=True` bestrafen wir nur Steigungen mit Betrag größer als eins, und zwar durch $\max(0, |T'| - 1)^2$. Setzen wir `critic_loss` auf `None`, minimiert der Generator `gen_loss` für die beiden Batches.

<!--en:7dc1085e-->
Wir vergleichen sechs Konfigurationen mit je drei Zufallsstartwerten: sättigenden und nicht sättigenden logistischen Verlust, Hinge-Verlust, WGAN-GP ($\gamma = 10$, zweiseitige Strafe $(|T'| - 1)^2$), WGAN-GP mit einseitiger Strafe $\max(0, |T'| - 1)^2$ und MMD mit der Median-Heuristik für die Bandbreite. Für jeden Lauf geben wir den Median des Gradientenbetrags des Generators über die ersten 100 Schritte und die Trajektorie von $\theta$ an. Außerdem erfassen wir den Schritt der *Ankunft* (erster Schritt mit $|\theta| < 0.5$) und die *größte* Auslenkung $\max |\theta|$ nach der Ankunft.

<!--en:17eacacf-->
Die Ergebnisse zeigen, wie sich neuronale Kritiker in der Praxis von theoretisch optimalen Kritikern ohne Nebenbedingungen unterscheiden:

<!--en:6ddf4f87-->
- **Sättigender und nicht sättigender logistischer Verlust:** Wie in :numref:`sec_basic_gan` ist der Generatorgradient bei sättigendem Verlust winzig ($\sim 0.01$). Daher bewegt sich $\theta$ nur von $6.0$ auf $5.76$. Beim nicht sättigenden Verlust erhält der Generator einen starken Gradienten ($\sim 3$) und erreicht das Ziel vor Schritt 400.

<!--en:a555fb01-->
- **Hinge-Kritiker:** Der theoretisch optimale Hinge-Kritiker $\operatorname{sign} \lambda$ hat innerhalb beider Punktwolken die Steigung null. Ein endliches ReLU-MLP interpoliert dagegen stetig über die Lücke hinweg mit einer Steigung von $1.1\text{--}1.7$. So erreicht der Generator das Ziel nach $700\text{--}950$ Schritten.

<!--en:78404f8c-->
- **MMD mit Median-Bandbreite:** Außer beim sättigenden Verlust ist dies die einzige Variante, die bis Schritt 1500 das Ziel nicht erreicht hat. Hier gilt dann $\theta \approx 1.3$. Der Generator bewegt sich langsam, kommt aber nicht zum Stillstand. Die Median-Heuristik hält die Bandbreite in der Größenordnung des Abstands, sodass der Gradient nie verschwindet. Mit $\sim 0.2$ beträgt er etwa ein Sechstel der Steigung der $W_1$-Kritiker im nächsten Punkt. Gewöhnliche Gradientenschritte mit Schrittweite $0.01$ brauchen entsprechend länger.

<!--en:be83cbd5-->
- **WGAN-GP mit zweiseitiger und einseitiger Strafe:** Beide $W_1$-Varianten ziehen den Generator mit Steigung $\approx 1.26$ zum Ursprung (Minimieren von $-ad + 10(a-1)^2$ ergibt $a = 1 + d/20$) und erreichen ihn etwa bei Schritt 500. Danach schießt der Generator in jedem zweiseitigen Lauf mit `W1 + GP` weit über das Ziel hinaus (`farthest` $> 1$, in der Spitze sogar $> 7$). In jedem einseitigen Lauf bleibt er dagegen höchstens $0.06$ vom Ursprung entfernt. Warum? Sobald $\theta$ die $0$ überquert, muss die Steigung des Kritikers von negativ auf positiv wechseln. Die zweiseitige Strafe $(|T'| - 1)^2$ bestraft einen flachen Kritiker ($T' = 0$) mit $\gamma$ und erzeugt damit bei Steigung null eine Potentialbarriere. Dadurch behält die Steigung ihr altes Vorzeichen ($1 - d/20$), oder sämtliche ReLU-Einheiten werden dauerhaft inaktiv. Dies bestätigen die unten aufgeführten Steigungen der Kritiker und die Anzahlen aktiver ReLU-Einheiten bei Schritt 1500:

<!--en:3b6d49b8-->
Die einseitige Strafe $\max(0, |T'| - 1)^2$ verschwindet für $|T'| \leq 1$. Die Steigung des Kritikers kann daher ungehindert durch null gehen, sobald $\theta$ die Daten erreicht. In :numref:`sec_gan_convergence` untersuchen wir um null zentrierte Strafen ($R_1$ und $R_2$), die die Steigung des Kritikers im Gleichgewicht gezielt gegen null ziehen.

<!--en:c51cc6f3-->
### Vergleich von fünf Zielfunktionen an einer zweidimensionalen Gauß-Mischung

<!--en:07949ab4-->
Nun vergleichen wir fünf Zielfunktionen für den Fall, dass sich die Träger von Generator- und Datenverteilung überlappen: nicht sättigender logistischer Verlust, kleinste Quadrate, Hinge-Verlust, quadrierter Hinge-Verlust und MMD mit mehreren Bandbreiten ($s \in \{0.5, 1.0, 2.0\}$). Der Generator `MixtureGenerator` parametrisiert eine Gauß-Mischung aus 3 Komponenten. Wir initialisieren sie mit verschobenen Mittelwerten `mu_0` und geringerer Streuung `A_0 = 0.4 * I` ($\mathrm{JS}(p, q_0) \approx 0.53$ Nat).

<!--en:02efdecb-->
Zu Beginn beträgt der Abstand von der Datenverteilung etwa ein halbes Nat. Die MMD-Variante minimiert den erwartungstreuen Minibatch-Schätzer für $\mathrm{MMD}^2$. Als Kernel dient die Summe dreier RBF-Kernel mit Bandbreiten $0.5$, $1$ und $2$. Damit erfasst der Kernel mehrere Längenskalen.

<!--en:2c7d4ef8-->
`train_toy` erhält die beiden Verluste als Argumente und führt abwechselnd je einen Kritiker- und einen Generatorschritt aus. Für die MMD-Variante ist als Kritikerverlust `None` gespeichert. Hier gibt es nur den Generatorschritt, der `mmd2_loss(X, fake)` anstelle der Scores des Kritikers verwendet. Alle 200 Schritte zeichnet die Funktion den exakten Wert $\mathrm{JS}(p, q_t)$ auf.

<!--en:1ac9d9d8-->
Für jede Zielfunktion und jeden der drei Zufallsstartwerte geben wir den Median von $\mathrm{JS}(p, q_t)$ über die letzten 2000 Schritte an. Außerdem erfassen wir jeweils den ersten Schritt, in dem $\mathrm{JS}(p, q_t) < 0.1$ beziehungsweise $< 0.01$ Nat gilt.

<!--en:e0236c73-->
Alle fünf Zielfunktionen senken $\mathrm{JS}(p, q_t)$ von $0.53$ Nat auf wenige Tausendstel Nat oder weniger. Die direkte MMD-Minimierung muss keinem sich verändernden Kritiker folgen und liefert am Ende die beste Anpassung ($0.0001$ Nat). Der Hinge-Verlust pendelt sich etwas höher ein ($0.003\text{--}0.007$ Nat) als die Zielfunktionen mit glatter Linkfunktion (logistischer Verlust, kleinste Quadrate und quadrierter Hinge-Verlust, alle $\leq 0.0024$ Nat, Aufgabe :exref:`ex_objectives-plateaus-lag`). In :numref:`sec_gan_relativistic` untersuchen wir als Nächstes Kritiker, die echte und generierte Samples paarweise vergleichen, statt jedes Sample einzeln zu bewerten.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:c3d48236-->
Jede adversariale Zielfunktion für einzelne Samples bildet das Supremum der erwarteten Auszahlungen über eine Kritikerklasse $\mathcal{T}$ (:eqref:`eq_gan_template`). Deshalb ist sie konvex in der Verteilung $q$, bleibt jedoch nichtkonvex in den Generatorparametern $\theta$. Ohne Nebenbedingungen für den Kritiker messen symmetrische Margin-Verluste bei der Klassifikation genau die $f$-Divergenz, die durch die Differenz ihrer Bayes-Risiken $\Delta_\ell(p, q)$ bestimmt ist (:numref:`tab_gan_loss_divergences`). Die optimalen Kritiker liefern monotone Linkfunktionen von $\lambda = \log(p/q)$. Die Variationsschranke nach Fenchel und Young (:eqref:`eq_gan_fgan_bound`) erweitert diese Dualität auf asymmetrische $f$-Divergenzen. Ersetzen wir die nichtlinearen Auszahlungen durch lineare Erwartungswerte über eine eingeschränkte Funktionenklasse $\mathcal{F}$, erhalten wir integrale Wahrscheinlichkeitsmetriken. Beispiele sind die Maximum Mean Discrepancy auf einer RKHS-Einheitskugel und die Wasserstein-1-Distanz auf der Klasse der 1-Lipschitz-stetigen Funktionen. Rücken die Träger auseinander, sättigen die $f$-Divergenzen. Dagegen behält $W_1$ die Steigung eins, und $\mathrm{MMD}^2$ liefert bis zur Kernel-Bandbreite aussagekräftige Gradienten. In der Praxis interpolieren endliche neuronale Kritiker glatt über solche Lücken hinweg, wenn wir sie mit einem nicht sättigenden Generatorverlust kombinieren.

<!--en:2294243a-->
## Übungen

<!--en:f64d284f-->
1. [code] **Konvexität im Raum der Verteilungen und im Parameterraum.** Jede Zielfunktion der Form :eqref:`eq_gan_template` ist konvex in $q$. Ein parametrischer Generator verändert jedoch $\theta$, statt $q$ direkt zu verändern. Verwenden Sie `normal_pdf`, `js_1d` und `xs`.
    1. Seien $p = \mathcal{N}(0, 1)$ und $q_0 = \mathcal{N}(8, 1)$ gegeben. Prüfen Sie anhand zweiter Differenzen, dass $\mathrm{JS}(p, q_t)$ in $t \in [0, 1]$ entlang des linearen Mischungspfads $q_t = (1 - t)\, q_0 + t\, p$ konvex ist.
    1. Seien nun $p = 0.7\, \mathcal{N}(-4, 1) + 0.3\, \mathcal{N}(4, 1)$ und $q_\theta = \mathcal{N}(\theta, 1)$ gegeben. Plotten Sie $\mathrm{JS}(p, q_\theta)$ für $\theta \in [-6, 6]$ und bestimmen Sie die lokalen Minima und Maxima.
    1. Warum steht dieser nichtkonvexe Verlauf in $\theta$ im Einklang mit der Konvexität in $q$?
   :exlabel:`ex_objectives-convexity-parameterization`

<!--en:fe1858c5-->
1. **Konvex Konjugierte zur Jensen-Shannon-Divergenz.** Die doppelte Jensen-Shannon-Divergenz entspricht der Divergenzfunktion $f(u) = u \log u - (u + 1) \log \frac{u + 1}{2}$.
    1. Leiten Sie deren konvex Konjugierte $f^*(t) = -\log(2 - e^t)$ mit Definitionsbereich $t < \log 2$ her.
    1. Zeigen Sie, dass unter der Umparametrisierung $T(x) = \log(2 \sigma(D(x)))$ der optimale Kritiker $T^\star = f'(\rho)$ dem Kritiker $D^\star = \lambda$ entspricht. Zeigen Sie weiter, dass die Variationsschranke :eqref:`eq_gan_fgan_bound` dann die Form $2 \log 2 + V(D)$ annimmt. Dabei ist $V(D)$ die klassische GAN-Zielfunktion mit logarithmischem Verlust aus :numref:`sec_basic_gan`.
   :exlabel:`ex_objectives-variational-bound-eqref-eq`

<!--en:d0822db7-->
1. [code] **Gewichtung im Generator und verzögerte Anpassung des Kritikers bei der Gauß-Mischung.** ● Verwenden Sie `train_toy` und die fünf Zielfunktionen aus :numref:`subsec_gan_objectives_testbed`.
    1. Kombinieren Sie den logistischen Kritikerverlust mit dem linearen Generatorverlust $-D(x')$ (Gewicht $w = 1$). Vergleichen Sie ihn mit dem nicht sättigenden Verlust $-\log \sigma(D(x'))$. Bestimmen Sie dazu für drei Zufallsstartwerte jeweils den ersten Schritt, bei dem $\mathrm{JS}(p, q_t) < 0.1$ Nat gilt. Erklären Sie den Unterschied anhand der Gewichte der einzelnen Samples.
    1. Prüfen Sie, ob die etwas höhere abschließende $\mathrm{JS}$-Divergenz beim Hinge-Verlust an der verzögerten Anpassung des Kritikers oder am Minibatch-Rauschen liegt. Führen Sie dazu zwei Kontrollversuche mit je drei Zufallsstartwerten aus: (i) 5 Kritikeraktualisierungen pro Generatoraktualisierung und (ii) eine $4\times$ so große Batchgröße für den Kritiker.
   :exlabel:`ex_objectives-plateaus-lag`

<!--en:84b3b57e-->
1. [code] **Stichprobenkomplexität und Dimension.** Seien $p = \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_x})$ und $q = \mathcal{N}(\boldsymbol{\mu}, \mathbf{I}_{d_x})$ mit $\|\boldsymbol{\mu}\| = 1$ gegeben. Dann gilt $W_1(p, q) = 1$ in jeder Dimension $d_x$.
    1. Schätzen Sie $W_1$ aus zwei Stichproben der Größe $n$ durch optimale lineare Zuordnung (`scipy.optimize.linear_sum_assignment`) für $d_x \in \{1, 2, 10, 50\}$ und $n \in \{100, 400, 1600\}$. Tragen Sie den relativen Fehler gegen $n$ auf doppelt logarithmischen Achsen auf. Vergleichen Sie das Ergebnis mit dem erwartungstreuen Schätzer für $\mathrm{MMD}^2$ mit einem RBF-Kernel der Bandbreite $s = \sqrt{d_x}$.
    1. Messen Sie die empirische $W_1$-Distanz zwischen zwei unabhängigen Stichproben der Größe $n = 256$ aus *derselben* Verteilung $p$ in Dimension $d_x = 50$.
    1. Berechnen Sie die Variationsschranke für die Vorwärts-KL-Divergenz :eqref:`eq_gan_fgan_bound` für `critics['forward KL']` durch Quadratur auf `grid_2d`. Vergleichen Sie das Ergebnis mit dem wahren Wert $D_{\textrm{KL}}(p \,\|\, q) \approx 0.16$ Nat und mit Schätzungen aus Minibatches der Größe 512. Warum kann ein endlicher Minibatch aus $q$ die Schranke überschätzen, wenn $T(x')$ den Wert $1 + \lambda(x')$ in Bereichen übersteigt, in denen $q(x')$ klein ist?
   :exlabel:`ex_objectives-estimation-and-dimension`

<!--en:c1e425da-->
1. **Symmetrische $f$-Divergenzen aus Klassifikationsverlusten.** ●● Jeder Margin-Verlust der Form :eqref:`eq_gan_margin` hat ein symmetrisches Bayes-Risiko $L(\eta) = L(1 - \eta)$ und misst damit eine symmetrische $f$-Divergenz. Betrachten Sie nun die Umkehrung (:citet:`Nguyen.Wainwright.Jordan.2009`, Korollar 3). Sei $f$ konvex auf $(0, \infty)$ mit $f(1) = 0$, $f(u) = u f(1/u)$ und $f(0) = \lim_{u \to 0} f(u) < \infty$. Definieren Sie

   $$
   L(\eta) = 2 f(0) - 2 (1 - \eta)\, f\!\left( \frac{\eta}{1 - \eta} \right), \qquad 0 < \eta < 1.
   $$

    1. Zeigen Sie, dass $L(\eta)$ konkav und symmetrisch um $\tfrac12$ ist, für $\eta \to 0$ oder $1$ gegen null geht und die Differenz der Bayes-Risiken $\Delta_\ell(p, q) = D_f(p \,\|\, q)$ liefert.
    1. Für differenzierbares $L$ und $\hat\eta = \sigma(t)$ definieren Sie den Verlust für echte Samples als $-\ell(t) = L(\hat\eta) + (1 - \hat\eta) L'(\hat\eta)$. Zeigen Sie, dass $-\ell(-t) = L(\hat\eta) - \hat\eta L'(\hat\eta)$ gilt, dass das bedingte Risiko bei $\hat\eta = \eta$ sein Minimum $L(\eta)$ annimmt und dass $-\ell(t) \geq 0$ gilt.
    1. Überprüfen Sie, dass diese Konstruktion für die Jensen-Shannon-Divergenz wieder den logistischen Verlust ergibt.
   :exlabel:`ex_objectives-divergences-a-loss-can-reach`

<!--en:48fc4d1f-->
[Diskussion](https://d2l.discourse.group/t/36809)

<!--en:c21b2e3b-->
Adversariale Zielfunktionen und Divergenzen<br>**Auszahlungen bei der Klassifikation und $f$-Divergenzen · optimale Linkfunktionen der Kritiker · Fenchel-Young-Dualform des $f$-GAN · MMD und Wasserstein-1 als IPMs · Gradienten bei getrennten Trägern**

<!--en:f42e47c8-->
::: {.slide title="Warum GAN-Implementierungen mehrere Zielfunktionen anbieten"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:6023ceba-->
- **1. Sättigung der Divergenz auf Populationsebene:** Liegen die Träger weit auseinander (z. B. $\mathcal{N}(0, 1)$ und $\mathcal{N}(7, 1)$), weicht $\mathrm{JS}(p, q)$ um weniger als $0.1\%$ von $\log 2$ ab. Die Ableitung beträgt $0.0013$.

<!--en:550fbd6c-->
- **2. Geometrie der Funktionenklasse des Kritikers ($\mathcal{T}$):** Springt der Kritiker $D \in \mathcal{T}$ unstetig zwischen den Trägern, oder behält er in der Lücke einen räumlichen Gradienten $\nabla_x D(x')$ ungleich null?

<!--en:6e0d21c7-->
- **3. Gewichtung durch den Generatorverlust:** Gibt der Generatorverlust $\nabla_x D(x')$ weiter (`w_ns ≈ 1`), oder multipliziert er den Gradienten mit einem exponentiell abklingenden Gewicht (`w_sat ≈ e^D → 0`)?

<!--en:16b48a87-->
::: {.slide title="Allgemeine Formulierung adversarialer Diskrepanzen"}

<!--en:4b1803a3-->
[Gemeinsamer Ansatz]{.kicker}

<!--en:2906fe06-->
- **Auszahlungen bei der Klassifikation ($\mathcal{T}$ ohne Nebenbedingung):** Nichtlineare Margin-Verluste $(\ell_p, \ell_q)$ schätzen monotone Linkfunktionen von $\lambda(x) = \log(p/q)$ und messen **$f$-Divergenzen** (bei Paarbildung die JS-Divergenz auf dem Produktraum).

<!--en:cf5a4753-->
- **Integrale Wahrscheinlichkeitsmetriken ($\mathcal{T}$ mit Nebenbedingung):** Lineare Auszahlungen $\ell_p(t) = t, \ell_q(t) = -t$ auf einer RKHS-Kugel (**MMD**) oder einer Kugel aus 1-Lipschitz-Funktionen (**$W_1$**).

<!--en:7fd92f85-->
- **Konvexität im Raum der Maße:** Als Supremum von Erwartungswerten, die linear in $q$ sind, ist $d(p, q)$ stets konvex in $q$ (jedoch nichtkonvex in den Generatorgewichten $\theta$).

<!--en:e5372b4f-->
::: {.slide title="Von der A-posteriori-Wahrscheinlichkeit zur Differenz der Bayes-Risiken"}

<!--en:ae898812-->
[Beispiel mit zwei Ergebnissen]{.kicker}

<!--en:6b488470-->
- **Beispiel mit zwei Zuständen ($p = (\frac{3}{4}, \frac{1}{4}), q = (\frac{1}{4}, \frac{3}{4})$):** Jeder Klassifikationsverlust vergleicht dieselbe A-posteriori-Wahrscheinlichkeit $\eta(x) = \frac{p(x)}{p(x) + q(x)}$ ($\frac{3}{4}$ bei Kopf, $\frac{1}{4}$ bei Zahl) mit der blinden Vorhersage $\eta = \frac{1}{2}$.

<!--en:3d73a8f6-->
- **Divergenzen aus der Differenz der Bayes-Risiken $\Delta_\ell = L(\frac{1}{2}) - \mathbb{E}_m[L(\eta)]$:**
  - **Logistischer Verlust:** $\log 2 - H_b(\frac{3}{4}) \approx 0.131\text{ nats} = \mathrm{JS}(p, q)$.
  - **Quadratischer Verlust (LSGAN):** $\frac{1}{4} - \frac{3}{16} = \frac{1}{16} = \frac{1}{4}\chi^2(p \,\|\, m)$.
  - **Hinge-Verlust:** $1 - \frac{1}{2} = \frac{1}{2} = \mathrm{TV}(p, q)$.

<!--en:2ac2aa22-->
::: {.slide title="Satz: Jeder Klassifikationsverlust liefert eine f-Divergenz"}

<!--en:bd11cf1b-->
[Differenz der Bayes-Risiken]{.kicker}

<!--en:6329130e-->
- **Konkaves bedingtes Bayes-Risiko:** $L(\eta) = \inf_t \{-\eta\,\ell(t) - (1 - \eta)\,\ell(-t)\}$ ist auf $[0, 1]$ konkav, da es ein Infimum affiner Funktionen von $\eta$ ist.

<!--en:01a22bf8-->
- **Konvexe erzeugende Funktion der $f$-Divergenz:** Der Übergang zum Dichteverhältnis $u = p(x)/q(x)$ (mit $\eta = u/(u+1)$) ergibt eine Funktion $f(u)$, die auf $[0, \infty)$ konvex ist und $f(1) = 0$ erfüllt.

<!--en:79ffb371-->
- **Monotone Linkfunktion des Kritikers $D^\star(\lambda)$:** Der punktweise optimale Kritiker ist eine monotone Funktion des logarithmischen Dichteverhältnisses $\lambda = \log(p/q)$.

<!--en:503e87d4-->
::: {.slide title="Gängige Klassifikationsverluste und Linkfunktionen der Kritiker"}

<!--en:077f2737-->
[Einteilung der Margin-Verluste]{.kicker}

<!--en:5c767302-->
Jeder Margin-Verlust liefert eine Divergenz $\Delta_\ell$ und einen optimalen Kritiker $D^\star$ als monotone **Linkfunktion** von $\lambda = \log(p/q)$:

<!--en:bb2663a3-->
| Zielfunktion | Margin-Verlust $-\ell(t)$ | Bayes-Risiko $L(\eta)$ | Risikodifferenz $\Delta_\ell(p, q)$ | Optimale Linkfunktion $D^\star(\lambda)$ | Verlust für $G$ in der Praxis |

<!--en:e9743e23-->
| **Logistischer Verlust** | $\log(1 + e^{-t})$ | $H_b(\eta)$ | $\mathrm{JS}(p, q)$ | $\lambda$ | $-\log \sigma(D)$ |

<!--en:453b95cb-->
| **Kleinste Quadrate** | $\tfrac{1}{4}(1 - t)^2$ ($t = 2D - 1$) | $\eta(1 - \eta)$ | $\tfrac{1}{8}\int \frac{(p-q)^2}{p+q}$ | $\sigma(\lambda)$ | $\tfrac{1}{2}(D - 1)^2$ |

<!--en:13da453b-->
| **Quadrierter Hinge-Verlust** | $\max(0, 1 - t)^2$ | $4\eta(1 - \eta)$ | $\tfrac{1}{2}\int \frac{(p-q)^2}{p+q}$ | $\tanh(\lambda / 2)$ | $-D$ |

<!--en:3feeb406-->
| **Hinge-Verlust** | $\max(0, 1 - t)$ | $2\min(\eta, 1 - \eta)$ | $\mathrm{TV}(p, q)$ | $\operatorname{sign}(\lambda)$ | $-D$ |

<!--en:ea3091a9-->
::: {.slide title="Variationsschranke nach Fenchel und Young (f-GAN)"}

<!--en:12f3df2b-->
[Asymmetrische und unbeschränkte Divergenzen]{.kicker}

<!--en:0989a0db-->
- **Warum asymmetrische Auszahlungen $\ell_p \neq \ell_q$ nötig sind:** Jeder symmetrische Margin-Verlust liefert eine beschränkte Risikodifferenz $\Delta_\ell \le L(\frac{1}{2})$. Unbeschränkte asymmetrische Divergenzen wie die Vorwärts-KL-Divergenz $D_{\textrm{KL}}(p \,\|\, q)$ kann er daher nicht darstellen.

<!--en:f0b922d8-->
- **Fenchel-Konjugierte $f^*(t) = \sup_u \{ut - f(u)\}$:** Gleichheit bei $T^\star(x) = f'(p(x)/q(x))$. Für die Vorwärts-KL-Divergenz ($f(u) = u\log u$) gelten $f^*(t) = e^{t-1}$ und $T^\star(x) = 1 + \lambda(x)$.

<!--en:c6fe870e-->
::: {.slide title="Empirische Überprüfung der Linkfunktionen der Kritiker"}

<!--en:dd69a946-->
[Vergleich an einer zweidimensionalen Gauß-Mischung]{.kicker}

<!--en:2367903b-->
Training neuer MLP-Kritiker mit 3 Schichten und Leaky ReLU ($2 \to 64 \to 64 \to 1$): $3{,}000$ Schritte gegen eine feste zweidimensionale Gauß-Mischung $q$ aus 3 Komponenten ($\mathrm{JS}(p, q) = 0.033$ Nat). Das logarithmische Dichteverhältnis $\lambda(x) = \log(p(x)/q(x))$ ist in geschlossener Form bekannt:

<!--en:74aa0cea-->
- **Rekonstruktion der exakten Linkfunktionen:** An $300$ echten und $300$ generierten Samples folgen alle vier trainierten Netze ihren theoretischen Linkfunktionen von $\lambda$:
  - **Logistischer Verlust:** Identität $D^\star(\lambda) = \lambda$
  - **Kleinste Quadrate:** Sigmoidfunktion $D^\star(\lambda) = \sigma(\lambda)$
  - **Hinge-Verlust:** Sprungfunktion $D^\star(\lambda) = \operatorname{sign}(\lambda)$ (stärkere Streuung um $\lambda = 0$, wo jedes $D \in [-1, 1]$ das optimale Risiko erreicht)
  - **Vorwärts-KL-Divergenz:** verschobene Gerade $T^\star(\lambda) = 1 + \lambda$

<!--en:29061718-->
::: {.slide title="Integrale Wahrscheinlichkeitsmetriken (IPMs) und MMD"}

<!--en:a857ad5d-->
[Nebenbedingungen für die Kritikerklasse]{.kicker}

<!--en:8ca876e0-->
Bei $f$-Divergenzen hat der Kritiker keine Nebenbedingungen und die Auszahlungen sind nichtlinear. **Integrale Wahrscheinlichkeitsmetriken (IPMs)** verwenden lineare Auszahlungen $\ell_p(t) = t$ und $\ell_q(t) = -t$ auf einer eingeschränkten symmetrischen Funktionenklasse $\mathcal{F}$:

<!--en:4b68dbdf-->
- **Definition der IPM:**
  $$d_{\mathcal{F}}(p, q) = \sup_{T \in \mathcal{F}} \Big\{ \mathbb{E}_{x \sim p}\big[T(x)\big] - \mathbb{E}_{x' \sim q}\big[T(x')\big] \Big\}$$

<!--en:bb9ab86c-->
- **Maximum Mean Discrepancy (MMD, Gretton et al., 2012):** Sei $\mathcal{F}$ die Einheitskugel eines RKHS mit positiv definitem Kernel $k(x, x')$ (z. B. gaußsche RBF $e^{-\|x - x'\|^2 / (2s^2)}$). Dann hat die quadrierte Distanz eine geschlossene Form und erfordert kein Kritikernetz:
  $$\mathrm{MMD}^2(p, q) = \mathbb{E}_{p,p}[k(x, \tilde{x})] + \mathbb{E}_{q,q}[k(x', \tilde{x}')] - 2\mathbb{E}_{p,q}[k(x, x')]$$

<!--en:02e91377-->
::: {.slide title="Wasserstein-1-Distanz und Lipschitz-stetige Kritiker"}

<!--en:d8b43afd-->
[Dualität beim optimalen Transport]{.kicker}

<!--en:539fbcd8-->
- **Kantorovich-Rubinstein-Dualität:** Die Beschränkung der IPM-Testklasse $\mathcal{F}$ auf **1-Lipschitz-stetige Funktionen** ($|T(x) - T(y)| \le \|x - y\|$) liefert die Earth Mover’s Distance $W_1(p, q)$.

<!--en:ba61fc73-->
- **Drei Verfahren für neuronale Netze:** Gewichtsbegrenzung (beschränkt die Kapazität), WGAN-GP mit Gradientenstrafe entlang der Transportstrahlen (zweiseitige Strafe für Abweichungen vom Steigungsbetrag eins erzeugt bei $q = p$ eine Barriere) und spektrale Normalisierung ($W \leftarrow W/\sigma_{\max}(W)$).

<!--en:d27f6b58-->
::: {.slide title="Stichprobenkomplexität in hohen Dimensionen: MMD und W₁"}

<!--en:e192f361-->
[Grenzen statistischer Schätzverfahren]{.kicker}

<!--en:fe432632-->
- **MMD mit festem Kernel ($O(n^{-1/2})$ in jeder Dimension):** Der erwartungstreue Schätzer `mmd2_unbiased` ist eine U-Statistik. Er konvergiert mit der parametrischen Rate $O(n^{-1/2})$ unabhängig von $d_x$.

<!--en:1005b8a9-->
- **Empirische $W_1$-Distanz ($O(n^{-1/d_x})$ für $d_x > 2$):** Der exakte optimale Transport zwischen empirischen Verteilungen leidet unter dem Fluch der Dimensionalität (Weed und Bach, 2019). WGAN funktioniert in der Praxis, weil seine glatte Netzarchitektur eine parametrische Testfunktion liefert.

<!--en:2e0aa962-->
- **Explodierende Varianz beim variationellen $f$-GAN:** In der Dualform der Vorwärts-KL-Divergenz hat $e^{T^\star(x')-1} = p(x')/q(x')$ die Varianz $\chi^2(p \,\|\, q)$. Diese kann selbst dann unendlich sein, wenn $D_{\textrm{KL}}(p \,\|\, q) < \infty$ gilt.

<!--en:2bdc7d7b-->
::: {.slide title="Steigung der Populationsdivergenz in Abhängigkeit vom Abstand" layout="tight"}

<!--en:05498b2f-->
[Gradienten fern von den Daten]{.kicker}

<!--en:a9f4b9f0-->
Zwei eindimensionale Gauß-Verteilungen $p = \mathcal{N}(0, 1)$ und $q = \mathcal{N}(\mu, 1)$ mit Varianz eins im Vergleich: Wie reagiert jede Populationsdivergenz, wenn der Abstand $\mu > 0$ wächst?

<!--en:79ac491b-->
- **Wasserstein-1 ($W_1(p, q) = \mu$):** Die Ableitung $\frac{d}{d\mu}W_1 = 1$ bleibt bei jedem Abstand $\mu$ gleich eins.

<!--en:aba3f514-->
- **$\mathrm{MMD}^2$ mit Gauß-Kernel (Bandbreite $s = 4$):**
  $$\mathrm{MMD}^2(\mu) = \frac{2s}{\sqrt{s^2 + 2}}\!\left(1 - \exp\!\left(-\frac{\mu^2}{2(s^2 + 2)}\right)\right)$$
  Die Steigung erreicht ihr Maximum bei $\mu = \sqrt{s^2 + 2} \approx 4.2$ und klingt dann im Ausläufer der Gauß-Kurve ab.

<!--en:d4bd543e-->
- **Jensen-Shannon-Divergenz $\mathrm{JS}(p, q)$:** Die Steigung sinkt von $0.20$ bei $\mu = 1$ auf $0.0013$ bei $\mu = 7$ und auf $1.5 \times 10^{-22}$ bei $\mu = 20$.

<!--en:3833196c-->
::: {.slide title="Geometrie optimaler Kritiker bei disjunkten Trägern"}

<!--en:d618fbea-->
[Warum glatte Kritiker Gradienten liefern]{.kicker}

<!--en:5dc8586a-->
- **Links (JS-Kritiker $\log(p/q)$ ohne Nebenbedingung):** Er hat den Wert $+\infty$ auf $\operatorname{supp}(p)$ und $-\infty$ auf $\operatorname{supp}(q)$. Die räumliche Steigung ist auf beiden Punktwolken null.

<!--en:cc11939a-->
- **Mitte (Kantorovich-Potential für $W_1$):** Eine Gerade mit Steigung eins verbindet $q$ mit $p$ und liefert über die gesamte Lücke einen konstanten rücktreibenden Gradienten.

<!--en:46e988a7-->
- **Rechts (MMD-Zeugenfunktion $w(x) = \mathbb{E}_p[k(x, \cdot)] - \mathbb{E}_q[k(x, \cdot)]$):** Die glatte Differenz zweier glockenförmiger Kernel steigt an ihrem Ausläufer in Richtung $p$ an, und zwar über Abstände in der Größenordnung der Bandbreite $s$.

<!--en:424bddb9-->
::: {.slide title="Implementierung von MMD und WGAN-GP-Gradientenstrafen" layout="tight"}

<!--en:8f069e89-->
[Grundfunktionen im Code]{.kicker}

<!--en:a64ac00c-->
`mmd2_loss` summiert RBF-Kernel mit mehreren Bandbreiten. `mmd2_median` verwendet eine einzige Bandbreite, den Median der paarweisen Abstände im zusammengefassten Minibatch:

<!--en:2fc319ff-->
::: {.slide title="Implementierung von MMD und WGAN-GP-Gradientenstrafen (Fortsetzung)" layout="tight"}

<!--en:8f069e89-2-->
[Grundfunktionen im Code]{.kicker}

<!--en:11f31c51-->
`gradient_penalty` berechnet entweder die zweiseitige WGAN-GP-Strafe $(|D'(x)| - 1)^2$ oder die einseitige Lipschitz-Strafe $\max(0, |D'(x)| - 1)^2$ an zufälligen Zwischenpunkten $\tilde{x} = uX + (1 - u)X'$:

<!--en:7fa13243-->
::: {.slide title="Trainierte neuronale Kritiker bei großem Anfangsabstand" layout="code"}

<!--en:eac50973-->
[Theorie und Verhalten neuronaler Netze]{.kicker}

<!--en:2ba39704-->
Ein eindimensionaler Generator mit Lageparameter $q_\theta = \mathcal{N}(\theta, 1)$ ($x' = \theta + z$) lernt vom Startwert $\theta_0 = 6.0$ aus die Zielverteilung $p = \mathcal{N}(0, 1)$. Gewöhnliches SGD für $\theta$ macht den Gradientenbetrag direkt an der Schrittweite sichtbar. Der Kritiker ist ein ReLU-MLP mit 2 versteckten Schichten und macht $5$ Aktualisierungen pro Generatorschritt:

<!--en:c705792e-->
**Warum endliche neuronale Kritiker den Generator auch bei logistischem und Hinge-Verlust bewegen:** Der optimale Hinge-Kritiker *ohne Nebenbedingung* $\operatorname{sign}(\lambda)$ ist auf beiden Punktwolken flach. Ein endliches ReLU-MLP interpoliert über die Lücke hinweg linear mit Steigung $1.1\text{--}1.7$. Ein weit entfernter Generator bewegt sich, wenn sein trainierter Kritiker an $x'$ eine Steigung ungleich null hat **und** der Generatorverlust nicht sättigt (`logistic, NS` erreicht das Ziel vor Schritt $400$, `logistic, sat.` bleibt bei $5.76$ stecken).

<!--en:d37035a3-->
::: {.slide title="Warum WGAN-GP mit zweiseitiger Strafe über das Gleichgewicht hinausschießt"}

<!--en:326ec756-->
[Barriere durch die Vorgabe des Steigungsbetrags eins]{.kicker}

<!--en:09522ae7-->
Beide $W_1$-Varianten ziehen den Generator in den Bereich $|\theta| < 0.5$, den er etwa bei Schritt $500$ erreicht. Danach schießt jeder zweiseitige Lauf mit `W1 + GP` über das Ziel hinaus ($|\theta| > 1.1$, in der Spitze $> 7$). Jeder einseitige Lauf mit `W1 + 1-sided GP` pendelt sich in einem Abstand von höchstens $0.06$ vom Ursprung ein:

<!--en:0cef4424-->
- **Barriere bei Steigung null:** Sobald $\theta$ die $0$ überquert, muss die Steigung $D'$ des Kritikers von negativ auf positiv wechseln. Die zweiseitige Strafe $\gamma(|D'| - 1)^2$ beträgt jedoch $\gamma$ bei $D' = 0$. Daher behält die Steigung ihr altes Vorzeichen (und treibt $\theta$ nach außen), oder alle versteckten ReLU-Einheiten werden dauerhaft inaktiv (`active = 0`).

<!--en:663fe114-->
- **Abhilfe durch eine einseitige Strafe:** Der Strafterm $\gamma\max(0, |D'| - 1)^2$ verschwindet für $|D'| \le 1$. Die Steigung des Kritikers kann ungehindert durch null gehen, sobald $q$ die Verteilung $p$ erreicht.

<!--en:2974f004-->
::: {.slide title="Vergleich von fünf Zielfunktionen an einer zweidimensionalen Gauß-Mischung" layout="tight"}

<!--en:0dddbda8-->
[Vergleich bei überlappenden Trägern]{.kicker}

<!--en:f2c956ec-->
Die Generatorverteilung überlappt schon zu Beginn mit den Daten ($\mathrm{JS}(p, q_0) = 0.529$ Nat bei der zweidimensionalen Gauß-Mischung aus 3 Komponenten). Der Vergleich umfasst fünf Zielfunktionen mit je $6{,}000$ alternierenden Schritten und drei Zufallsstartwerten:

<!--en:b58391a0-->
- **Alle fünf nicht sättigenden Zielfunktionen konvergieren:** Jede Variante senkt $\mathrm{JS}(p, q_t)$ von $0.529$ Nat auf wenige Tausendstel Nat oder weniger.

<!--en:77666b70-->
- **Direkte MMD-Minimierung mit mehreren Bandbreiten ($s \in \{0.5, 1, 2\}$):** Sie benötigt kein Kritikernetz und liefert die beste abschließende Anpassung ($0.0001$ Nat).

<!--en:0ae626cc-->
- **Glatte Linkfunktionen und Hinge-Sprungfunktion:** Logistischer Verlust, kleinste Quadrate und quadrierter Hinge-Verlust erreichen alle $\le 0.0024$ Nat. Mit der unstetigen Hinge-Linkfunktion ($\operatorname{sign}\lambda$) liegt das Plateau etwas höher ($0.003\text{--}0.007$ Nat).

<!--en:d14c936c-->
::: {.slide title="Vergleich von fünf Zielfunktionen an einer zweidimensionalen Gauß-Mischung (Fortsetzung)"}

<!--en:0dddbda8-2-->
[Vergleich bei überlappenden Trägern]{.kicker}

<!--en:a5830157-->
Für jede Zielfunktion und jeden Zufallsstartwert: Median der JS-Divergenz über die letzten 2000 Schritte und jeweils erster Schritt, bei dem die JS-Divergenz unter $0.1$ beziehungsweise $0.01$ fällt:

<!--en:4e166d4c-->
::: {.slide title="Verständnisfragen zu Divergenzen, Linkfunktionen und Kritikern"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:8c29a79f-->
Fragen zu adversarialen Zielfunktionen und integralen Wahrscheinlichkeitsmetriken:

<!--en:491bcea7-->
1. **Null-Eins-Verlust und Vorwärts-KL-Divergenz:** Der $0\text{--}1$-Klassifikationsverlust hat das bedingte Bayes-Risiko $L(\eta) = \min(\eta, 1 - \eta)$. Wie groß ist die Differenz der Bayes-Risiken $\Delta_\ell(p, q)$? Warum kann kein symmetrischer nichtnegativer Klassifikationsverlust die Vorwärts-KL-Divergenz $D_{\textrm{KL}}(p \,\|\, q)$ messen?

<!--en:92b810a9-->
2. **Abstand mit größtem MMD-Gradienten:** Gegeben sind die eindimensionalen Gauß-Verteilungen $p = \mathcal{N}(0, 1)$ und $q = \mathcal{N}(\mu, 1)$ mit Varianz eins und ein gaußscher RBF-Kernel mit Bandbreite $s = 4$. Bei welchem Abstand $\mu$ ist der Gradient $\frac{d}{d\mu}\mathrm{MMD}^2(p, q)$ am größten?

<!--en:557b98b6-->
**Lösungen:**

<!--en:88c853ef-->
1. $L(\frac{1}{2}) - \mathbb{E}_m[\min(\eta, 1 - \eta)]$ $= \frac{1}{2}\int |p - q|\,\frac{dx}{2}$ $= \mathbf{\frac{1}{2}\mathrm{TV}(p, q)}$ (die Hälfte der Risikodifferenz beim Hinge-Verlust). Für jeden nichtnegativen Verlust gilt $L(\eta) \ge 0$. Seine Risikodifferenz ist daher nach oben beschränkt: $\Delta_\ell(p, q) \le L(\frac{1}{2}) < \infty$. Dagegen ist $D_{\textrm{KL}}(p \,\|\, q)$ unbeschränkt und asymmetrisch.

<!--en:4ea948b9-->
2. Die Ableitung von $1 - \exp(-\mu^2 / (2(s^2 + 2)))$ nach $\mu$ ist $\propto \mu\exp(-\mu^2 / (2(s^2 + 2)))$ und erreicht ihr Maximum bei $\mu = \sqrt{s^2 + 2} = \sqrt{18} \approx \mathbf{4.24}$.

<!--en:1dd86c36-->
::: {.slide title="Verständnisfragen zu Divergenzen, Linkfunktionen und Kritikern (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständnisfragen]{.kicker}

<!--en:8c09a833-->
3. **Warum Hinge-Kritiker auch einen weit entfernten Generator bewegen:** Im theoretischen Optimum $D^\star(x) = \operatorname{sign}(\lambda(x))$ hat der Hinge-Kritiker überall außer auf der Entscheidungsgrenze die Ableitung $0$. Warum konnte unser trainierter Hinge-Kritiker dennoch die Verschiebung $\theta_0 = 6 \to 0$ bewirken?

<!--en:23db34b8-->
**Lösung:**

<!--en:0484d819-->
3. Ein endliches, stückweise lineares ReLU-Netz springt nach endlich vielen Trainingsschritten nicht unstetig von $+1$ auf $-1$. Es interpoliert linear über die Lücke zwischen den Punktwolken hinweg mit Steigung $\approx 1.1\text{--}1.7$. Der lineare Generatorverlust $-D(x')$ gibt diese Steigung direkt an $\theta$ weiter.

<!--en:015ad4df-->
::: {.slide title="Zusammenfassung: Adversariale Zielfunktionen und Divergenzen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:4872bde4-->
| Familie von Zielfunktionen | Kritikerverlust / Nebenbedingung | Populationsdiskrepanz | Optimaler Kritiker $D^\star$ | Verhalten bei getrennten Trägern |

<!--en:67097a0a-->
| **Logistischer Verlust (nicht sättigend)** | $\operatorname{softplus}(-D(x)) + \operatorname{softplus}(D(x'))$ | $\mathrm{JS}(p, q)$ ($G$ folgt dem Abstieg für $2\,D_{\textrm{KL}}(m \,\|\, p)$) | $\lambda(x) = \log \frac{p(x)}{q(x)}$ | $\mathrm{JS}$ sättigt auf Populationsebene. Neuronaler Kritiker und $w_{\textrm{ns}}$ bewegen $G$ |

<!--en:b0b5722e-->
| **Kleinste Quadrate (LSGAN)** | $\frac{1}{2}(D(x) - 1)^2 + \frac{1}{2}D(x')^2$ | $\frac{1}{8}\int \frac{(p - q)^2}{p + q}$ ($\frac{1}{4}\chi^2(p \,\|\, m)$) | $\sigma(\lambda(x)) = \frac{p(x)}{p(x) + q(x)}$ | Beschränkte Linkfunktion $\sigma(\lambda) \in (0, 1)$ flacht weit außen ab |

<!--en:879ac9d3-->
| **Hinge-Verlust / quadrierter Hinge-Verlust** | $\max(0, 1 - D(x)) + \max(0, 1 + D(x'))$ | $\mathrm{TV}(p, q)$ (bzw. $\chi^2(p \,\|\, m)$ bei quadriertem Hinge-Verlust) | $\operatorname{sign}(\lambda(x))$ (bzw. $\tanh \frac{\lambda}{2}$) | Linearer Anstieg bis $2/\kappa$ mit einem $\kappa$-Lipschitz-stetigen Kritiker |

<!--en:8a648569-->
::: {.slide title="Zusammenfassung: Adversariale Zielfunktionen und Divergenzen (Fortsetzung)"}

<!--en:c5a222f8-2-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:4872bde4-2-->
| Familie von Zielfunktionen | Kritikerverlust / Nebenbedingung | Populationsdiskrepanz | Optimaler Kritiker $D^\star$ | Verhalten bei getrennten Trägern |

<!--en:5adb6d1a-->
| **Duale Schranke des $f$-GAN** | $-\mathbb{E}_p[T(x)] + \mathbb{E}_q[f^*(T(x'))]$ | Beliebige $f$-Divergenz $D_f(p \,\|\, q)$ | $f'(p(x)/q(x))$ | Vorwärts-KL-Divergenz möglich ($1 + \lambda$), aber hohe Varianz des Schätzers |

<!--en:f13db22c-->
| **Kernelbasierte MMD** | U-Statistik in geschlossener Form (ohne Kritikernetz) | $\mathrm{MMD}^2(p, q)$ auf der RKHS-Kugel | Zeugenfunktion $\mathbb{E}_p[k(x, \cdot)] - \mathbb{E}_q[k(x, \cdot)]$ | $O(n^{-1/2})$ in jeder Dimension, große Steigung bis zur Bandbreite $s$ |

<!--en:a339ef3e-->
| **Wasserstein-1 ($W_1$)** | $\mathbb{E}_q[T(x')] - \mathbb{E}_p[T(x)]$ + Lipschitz-Regularisierung | $W_1(p, q)$ (Earth Mover’s Distance) | Kantorovich-Potential ($\|\nabla T^\star\| = 1$) | Steigung eins bei jedem Abstand. Zweiseitige Gradientenstrafe führt bei $q = p$ zum Überschwingen |
