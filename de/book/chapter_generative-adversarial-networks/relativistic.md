<!--en:65379d98-->
# Relativistische Zielfunktionen

<!--en:5ea410fa-->
Ein Kritiker, der als gewöhnlicher Klassifikator arbeitet, vergleicht den Score jedes Beispiels mit der festen Schwelle null ($D(x') > 0$). In vielen Fällen liegt ein paarweiser Vergleich jedoch näher als eine absolute Bewertung. So leiten Präferenzmodelle nach Bradley und Terry (:numref:`sec_regularized`) aus der Score-Differenz zweier Kandidaten eine Gewinnwahrscheinlichkeit ab. :citet:`Jolicoeur-Martineau.2019` führte *relativistische* GAN-Kritiker ein. Ein solcher Kritiker erhält ein echtes Beispiel $x \sim p$ und ein generiertes Sample $x' \sim q$. Aus der Score-Differenz $D(x) - D(x')$ sagt er vorher, welches der beiden echt ist.

<!--en:5045403a-->
Das R3GAN-Verfahren, das wir in :numref:`sec_gan_convergence` entwickeln und im weiteren Kapitel verwenden, beruht auf dieser paarweisen Zielfunktion. Deshalb untersuchen wir drei Fragen:

<!--en:78595a6f-->
1. Liefert der optimale Kritiker für das paarweise Ranking weiterhin das logarithmierte Dichteverhältnis (engl. log density ratio) $\lambda(x) = \log(p(x)/q(x))$?

<!--en:01ace11a-->
2. Welche Divergenz misst die paarweise Zielfunktion an ihrem Optimum, und wie verhält sie sich zu $\mathrm{JS}(p, q)$?

<!--en:7537a329-->
3. Unter welchen Bedingungen beseitigt paarweises Ranking lokale Minima mit Modusausfall (engl. mode dropping)?

<!--en:f37648b9-->
## Bewertung von Paaren

<!--en:1d856b33-->
### Der Paarkritiker nach Bradley und Terry

<!--en:40a9ff30-->
Betrachten wir ein Paar $(x, x')$ mit unabhängig gezogenen Elementen $x \sim p$ und $x' \sim q$. Die Zielfunktion für die *relativistische Paarbildung* (engl. relativistic pairing, RpGAN, :citet:`Jolicoeur-Martineau.2019,Jolicoeur-Martineau.2020`) modelliert die Wahrscheinlichkeit, dass $x$ realistischer ist als $x'$. Dazu nutzt sie die Bradley-Terry-Likelihood $\sigma(D(x) - D(x'))$ :cite:`Bradley.Terry.1952`:

<!--en:e2056be1-->
Der Kritiker maximiert $\Phi(D)$ (minimiert also $-\Phi(D)$). Für einen Minibatch mittelt `rpgan_loss_D` den Ausdruck $-\log \sigma(D(x_i) - D(x'_i)) = \operatorname{softplus}(D(x'_i) - D(x_i))$ über die einander zugeordneten Paare $(x_i, x'_i)$. Die Funktion `rpgan_loss_G` vertauscht die Argumente und berechnet so den nicht sättigenden Generatorverlust:

<!--en:a71a9a67-->
### Verschiebungsinvarianz und optimaler Kritiker

<!--en:25861b39-->
Bei einem Klassifikator für einzelne Beispiele legt der logarithmische Verlust die Schwelle null fest. Die Zielfunktion für das paarweise Ranking hängt dagegen nur von den Differenzen $D(x) - D(x')$ ab (:numref:`fig_gan_pairing`). Addieren wir eine beliebige Konstante $b$ zu $D$, so bleibt der Zielfunktionswert gleich: $\Phi(D + b) = \Phi(D)$. Das Vorzeichen eines einzelnen Scores $D(x')$ ist daher beliebig. Als Genauigkeitsmaß bietet sich der Anteil korrekt geordneter Paare an, also die Genauigkeit des paarweisen Rankings $P(D(x) > D(x'))$.

<!--en:e53f715e-->
![Schwellenvergleich für einzelne Beispiele (links) und Bewertung anhand paarweiser Differenzen (rechts). Eine Verschiebung aller Kritiker-Scores um eine Konstante $b$ verändert die Klassifikation einzelner Beispiele anhand der Schwelle null. Die paarweisen Differenzen $D(x) - D(x')$ bleiben dabei unverändert.](../img/mdl-gan-pairing.svg)

<!--en:2231494b-->
Beim logarithmischen Verlust für einzelne Beispiele ließ sich $V(D)$ für jedes $x$ unabhängig maximieren. In :eqref:`eq_gan_rp` koppeln die Differenzen $D(x) - D(x')$ alle Punkte miteinander. Dank der Konkavität lässt sich das Optimum dennoch in geschlossener Form angeben.

<!--en:2014a401-->
**Proposition.** *$\Phi(D)$ ist konkav in $D$. Haben $p$ und $q$ denselben Träger, so ist $D^\star(x) = \lambda(x) = \log(p(x)/q(x))$ bis auf eine additive Konstante der eindeutige Maximierer.*

<!--en:e7405a75-->
**Beweis.** Da $D \mapsto D(x) - D(x')$ linear und $\log \sigma$ streng konkav ist, ist $\Phi(D)$ konkav in $D$ und streng konkav in den Score-Differenzen. Wir bilden die Funktionalableitung von $\Phi(D)$ nach $D(t)$. Dabei tritt $t$ sowohl als echtes Beispiel $x$ als auch als generiertes Sample $x'$ auf. Mit $\tfrac{d}{du} \log \sigma(u) = \sigma(-u)$ erhalten wir:

<!--en:4714b0f3-->
Für $D = \lambda = \log \rho$ mit $\rho = p/q$ ergibt sich $\sigma(\lambda(x') - \lambda(t)) = \rho(x')/(\rho(x') + \rho(t))$. Wegen $q\rho = p$ sind beide Terme in :eqref:`eq_gan_rp_stationarity` gleich $p(t) \int \frac{p(x)}{\rho(x) + \rho(t)}\,dx$. Somit gilt überall $\frac{\delta \Phi}{\delta D(t)} = 0$. Da $\log \sigma$ in den Score-Differenzen streng konkav ist, ist $D^\star = \lambda + c$ bis auf eine konstante Verschiebung $c$ der eindeutige Maximierer. $\blacksquare$

<!--en:2239329d-->
## Die Paardivergenz

<!--en:4c5af20e-->
### Ein Beispiel mit zwei Zuständen

<!--en:0f51970b-->
Wie hängt der optimale Wert von $\Phi(D)$ mit der Jensen-Shannon-Divergenz zusammen? Betrachten wir den Zustandsraum $\{0, 1\}$ mit $p = (0.8, 0.2)$ und $q = (0.2, 0.8)$. Steht das echte Beispiel an erster Stelle, so folgt das geordnete Paar $(a_1, a_2)$ der Produktverteilung $P(a_1, a_2) = p(a_1)q(a_2)$. Steht es an zweiter Stelle, so folgt das Paar der Verteilung $Q(a_1, a_2) = q(a_1)p(a_2)$:

<!--en:6a711827-->
| Paar $(a_1, a_2)$ | $(0, 0)$ | $(1, 1)$ | $(0, 1)$ | $(1, 0)$ |

<!--en:88b04e94-->
| $P = p \otimes q$ (echtes Beispiel zuerst) | 0.16 | 0.16 | 0.64 | 0.04 |

<!--en:5a42dbbb-->
| $Q = q \otimes p$ (echtes Beispiel an zweiter Stelle) | 0.16 | 0.16 | 0.04 | 0.64 |

<!--en:bbbc711f-->
Die Paare $(0, 0)$ und $(1, 1)$ mit gleichen Elementen verraten nichts darüber, welches Element echt ist. Bei den gemischten Paaren $(0, 1)$ und $(1, 0)$ lässt sich das echte Element mit der A-posteriori-Wahrscheinlichkeit $0.64/(0.64 + 0.04) = 16/17 \approx 0.94$ bestimmen (gegenüber $0.80$ bei einer einzelnen Beobachtung). Bei der optimalen Score-Differenz $d = D^\star(0) - D^\star(1) = \lambda(0) - \lambda(1) = \log 16$ beträgt der Gewinn gegenüber zufälligem Raten ($\Phi(0) = -\log 2$) genau $\sup_D \Phi(D) + \log 2 = 0.68\big(\log 2 - H_b(\tfrac{1}{17})\big) \approx 0.319$ Nat. Das ist $1.66\times$ so viel wie $\mathrm{JS}(p, q) = \log 2 - H_b(0.8) \approx 0.193$ Nat und entspricht genau der Divergenz $\mathrm{JS}(P, Q)$ zwischen den beiden Paarverteilungen $P$ und $Q$:

<!--en:88eff532-->
### Übergang zu Produktmaßen

<!--en:aeadd9cd-->
Für den allgemeinen Beweis dieser Äquivalenz betrachten wir das geordnete Paar $(a_1, a_2) \in \mathcal{X} \times \mathcal{X}$ als eine einzige Beobachtung. Sie stammt entweder aus $P = p \otimes q$ (echtes Beispiel zuerst) oder aus $Q = q \otimes p$ (echtes Beispiel an zweiter Stelle). Wir verwenden die übliche Zielfunktion mit logarithmischem Verlust aus :numref:`sec_basic_gan` auf $\mathcal{X} \times \mathcal{X}$ mit einem Paarkritiker $\mathcal{D}(a_1, a_2)$:

<!--en:76b6ed17-->
**Lemma (Übergang zum Produktraum).** *Es seien $P = p \otimes q$ und $Q = q \otimes p$. Dann gilt:*

<!--en:a0795b87-->
1. *Das logarithmierte Dichteverhältnis auf dem Produktraum lässt sich additiv zerlegen: $\log \frac{dP}{dQ}(a_1, a_2) = \lambda(a_1) - \lambda(a_2)$.*

<!--en:e04eeeb6-->
2. *Für jeden antisymmetrischen Differenzkritiker $\mathcal{D}(a_1, a_2) = D(a_1) - D(a_2)$ gilt $V_{P,Q}(\mathcal{D}) = 2\,\Phi(D)$.*

<!--en:35739379-->
3. *Die Beschränkung von $\mathcal{D}$ auf Differenzkritiker $D(a_1) - D(a_2)$ verringert das Supremum nicht: $\sup_{\mathcal{D}} V_{P,Q}(\mathcal{D}) = \sup_{D} V_{P,Q}\big(D(a_1) - D(a_2)\big)$.*

<!--en:5f8c4f23-->
**Beweis.** (1) folgt unmittelbar aus $\frac{dP}{dQ}(a_1, a_2) = \frac{p(a_1)q(a_2)}{q(a_1)p(a_2)} = \frac{\rho(a_1)}{\rho(a_2)}$. Zu (2): Unter $Q$ gilt $(a_1, a_2) \sim q \otimes p$ und $-\mathcal{D}(a_1, a_2) = D(a_2) - D(a_1)$. Damit sind beide Erwartungswerte in $V_{P,Q}(\mathcal{D})$ gleich $\Phi(D)$. Zu (3): Ohne Einschränkung ist der Maximierer von $V_{P,Q}(\mathcal{D})$ durch $\mathcal{D}^\star(a_1, a_2) = \log \frac{dP}{dQ}(a_1, a_2)$ gegeben. Nach (1) hat er bereits die Differenzform $\lambda(a_1) - \lambda(a_2)$. $\blacksquare$

<!--en:97417d08-->
(Eigenschaft 3 beruht auf der logarithmischen Verknüpfung in der logistischen Auszahlungsfunktion. Bei anderen Auszahlungsfunktionen, etwa der Methode der kleinsten Quadrate, ist der optimale Paarkritiker eine nichtlineare Funktion von $\lambda(a_1) - \lambda(a_2)$ und lässt sich nicht additiv zerlegen. Siehe Übung :exref:`ex_relativistic-separability-choice-loss-replace`.)

<!--en:37effc34-->
**Satz.** *Für die kalibrierte Paardivergenz (engl. pairing divergence) $d_{\mathrm{Rp}}(p, q) := \sup_D \Phi(D) + \log 2$ gilt*

<!--en:e32df4d0-->
**Beweis.** Aus den Teilen 2 und 3 des Lemmas zum Produktraum und der Minimax-Identität für die Jensen-Shannon-Divergenz $\sup_{\mathcal{D}} V_{P,Q}(\mathcal{D}) = 2\,\mathrm{JS}(P, Q) - 2\log 2$ (:numref:`sec_basic_gan`) folgt $2\sup_D \Phi(D) = \sup_{\mathcal{D}} V_{P,Q}(\mathcal{D}) = 2\,\mathrm{JS}(P, Q) - 2\log 2$. Damit ist die erste Gleichheit bewiesen. Die zweite ergibt sich aus der Identität für die Entropiedifferenz $\mathrm{JS}(P, Q) = H(\tfrac{1}{2}(P + Q)) - \tfrac{1}{2}(H(P) + H(Q))$ (:numref:`sec_basic_gan`) zusammen mit $H(p \otimes q) = H(q \otimes p) = H(p) + H(q)$. $\blacksquare$

<!--en:c0e3aa8c-->
Setzen wir $D^\star = \lambda$ direkt in :eqref:`eq_gan_rp` ein, so erhalten wir auch eine explizite Darstellung als Erwartungswert:

<!--en:2d7c6be6-->
und die Identität für die gegenseitige Information (engl. mutual information) $\mathrm{JS}(P, Q) = I((a_1, a_2); y)$ (:numref:`sec_basic_gan`) auf dem Raum der Paare zeigt außerdem: Für ein zufällig angeordnetes Paar $(a_1, a_2)$ mit dem Label $y \in \{0, 1\}$ für die Reihenfolge gilt

<!--en:efe739f7-->
### Informationsschranken

<!--en:85a970cb-->
Mit :eqref:`eq_gan_rp_mi` können wir $d_{\mathrm{Rp}}(p, q)$ direkt mit $\mathrm{JS}(p, q)$ in Beziehung setzen. Jedes Element $a_k$ des zufällig angeordneten Paares hat die Randverteilung $m = (p+q)/2$ und liefert $I(a_k; y) = \mathrm{JS}(p, q)$ Nat Information über das Label $y$ für die Reihenfolge. Dabei sind $a_1$ und $a_2$ bei gegebenem $y$ bedingt unabhängig ($H(a_2 \mid a_1, y) = H(a_2 \mid y)$). Aus der Kettenregel für die gegenseitige Information folgt:

<!--en:df9aaac2-->
Wegen $I(a_2; y \mid a_1) \geq 0$, $I(a_1; a_2) \geq 0$ und $d_{\mathrm{Rp}}(p, q) \leq \log 2$ erhalten wir die folgenden scharfen unteren und oberen Schranken:

<!--en:10136a66-->
Nahe dem Gleichgewicht ($q \to p$) ist $I(a_1; a_2)$ von vierter Ordnung in $p - q$, während $\mathrm{JS}(p, q)$ von zweiter Ordnung ist. Deshalb gilt $d_{\mathrm{Rp}}(p, q) \approx 2\,\mathrm{JS}(p, q)$ (Übung :exref:`ex_relativistic-gap-twice-js`). Haben $p$ und $q$ disjunkte Träger, so erreichen sowohl $\mathrm{JS}(p, q)$ als auch $d_{\mathrm{Rp}}(p, q)$ ihren Sättigungswert $\log 2$.

<!--en:320e9bf1-->
## Ranking und Modusabdeckung

<!--en:2426c181-->
### Das relativistische Ranggewicht

<!--en:6917859b-->
Leiten wir $\Phi(D)$ nach den Generatorparametern $\theta$ ab und berücksichtigen dabei die Abhängigkeit über $x' = G(z)$, so erhalten wir:

<!--en:bbf42228-->
In der sättigenden Nullsummenformulierung minimieren wir $\Phi$. Jedes generierte Sample bewegt sich in Richtung $\nabla_\theta D(G(z))$ mit dem Gewicht $w_{\textrm{Rp}}(x')$. Dieses Ranggewicht (engl. rank weight) ist die geglättete Wahrscheinlichkeit dafür, dass $x'$ einen höheren Rang erhält als ein zufällig gezogenes echtes Beispiel $x \sim p$. In der nicht sättigenden Formulierung (`rpgan_loss_G`, Minimierung von $-\mathbb{E}[\log \sigma(D(x') - D(x))]$) beträgt das Gewicht $1 - w_{\textrm{Rp}}(x') = \mathbb{E}_{x \sim p}[\sigma(D(x) - D(x'))]$. Die Gewichte $\sigma(\pm D(x'))$ für einzelne Samples vergleichen $D(x')$ nur mit null. Das relativistische Gewicht hingegen vergleicht $D(x')$ mit der Verteilung der Scores $D(x)$ echter Beispiele und ist invariant gegenüber konstanten Verschiebungen von $D$.

<!--en:64e667b5-->
### Verlustlandschaften bei fester Paarbildung und bei Mittelung über alle Paare

<!--en:af38e704-->
Beseitigt paarweises Ranking lokale Minima mit Modusausfall? :citet:`Sun.Fang.Schwing.2020` untersuchten den diskreten Fall mit $n$ verschiedenen echten Punkten $x_1, \ldots, x_n$ und $n$ frei beweglichen generierten Punkten $x'_1, \ldots, x'_n$. Sie werteten die Zielfunktionen jeweils beim Kritiker $D$ aus, der die exakte beste Antwort liefert:

<!--en:c88714c2-->
Dabei bezeichnet $\mathcal{E}_{\mathrm{GAN}}$ die klassische Zielfunktion für einzelne Beispiele. Bei der Zielfunktion $\mathcal{E}_{\mathrm{diag}}$ mit *fester Paarbildung* (engl. fixed pairing) bleibt jeder echte Punkt $x_i$ dauerhaft mit dem generierten Punkt $x'_i$ gepaart. Dagegen mittelt $\mathcal{E}_{\mathrm{all}}$ über alle $n^2$ Paare aus einem echten und einem generierten Punkt. Liegen alle $x'_i$ auf echten Punkten und fallen mindestens zwei generierte Punkte zusammen, so bleibt mindestens ein echter Punkt unbesetzt. Wir nennen diese Konfiguration eine *Kollision* (engl. collision). :citet:`Sun.Fang.Schwing.2020` bewiesen, dass jede der $n^n - n!$ Kollisionen ein striktes lokales Minimum von $\mathcal{E}_{\mathrm{GAN}}$ ist. Bei $\mathcal{E}_{\mathrm{diag}}$ führt dagegen aus jeder Kollision ein Weg zum globalen Minimum $-\log 2$, auf dem der Zielfunktionswert nie steigt.

<!--en:cf90bd07-->
Für $n = 2$ und $n = 3$ können wir das direkt überprüfen. Für die empirischen Verteilungen $\hat p$ und $\hat q$ gilt $\mathcal{E}_{\mathrm{GAN}} = \mathrm{JS}(\hat p, \hat q) - \log 2$ und $\mathcal{E}_{\mathrm{all}} = \mathrm{JS}(\hat p \otimes \hat q, \hat q \otimes \hat p) - \log 2$. Dagegen ist $\mathcal{E}_{\mathrm{diag}}$ gleich $-\log 2$ mal dem Anteil der gerichteten Kanten $i \to \text{gen}[i]$, die auf einem Zyklus liegen (Übung :exref:`ex_relativistic-random-repairing`).

<!--en:a052f601-->
Verschieben wir $x'_2$ von $0$ nach $0.1$, so steigen sowohl der klassische Verlust als auch der Verlust über alle Paare. Damit ist die Kollision `[0, 0]` für beide ein striktes lokales Minimum. Der Verlust bei fester Paarbildung bleibt dagegen bei $-\frac{1}{2}\log 2$. Um alle $n^n$ Konfigurationen systematisch zu prüfen, testet `count_bad_minima(n)` jede Zuordnung mit lokalen Störungen aus `small_moves(gen)`:

<!--en:351b1bc5-->
Ein Aufruf von `count_bad_minima` für $n \in \{2, 3\}$ vergleicht die Anzahl suboptimaler strikter lokaler Minima jeder Zielfunktion mit der theoretischen Anzahl der Kollisionen $n^n - n!$:

<!--en:108672dd-->
Die vollständige Aufzählung zeigt: Der Verlust bei fester Kopplung $\mathcal{E}_{\mathrm{diag}}$ hat kein Kollisionsminimum. Der Verlust über alle Paare $\mathcal{E}_{\mathrm{all}}$ besitzt dagegen alle $n^n - n!$ Kollisionsminima, ebenso wie $\mathcal{E}_{\mathrm{GAN}}$. Beim üblichen Minibatch-Training ziehen wir in jedem Schritt unabhängig voneinander neue Minibatches mit echten Beispielen und generierten Samples. Im Erwartungswert über die zufälligen Paarungen ergibt sich deshalb die Zielfunktion $\mathcal{E}_{\mathrm{all}}$ statt $\mathcal{E}_{\mathrm{diag}}$ (Übung :exref:`ex_relativistic-random-repairing`). Der Satz von :citet:`Sun.Fang.Schwing.2020` über die Verlustlandschaft gilt also nur für eine feste Eins-zu-eins-Kopplung. Ob RpGAN mit Minibatches die Modusabdeckung (engl. mode coverage) verbessert, müssen wir empirisch prüfen (:numref:`sec_gan_convergence`).

<!--en:b7bf1cc2-->
### Sättigender und nicht sättigender Verlust bei paarweisem Training

<!--en:24b2d63c-->
Der sättigende Generatorverlust ($\Phi$) und der nicht sättigende Generatorverlust (`rpgan_loss_G`) haben dasselbe globale Minimum $q = p$. Wenn sich $p$ und $q$ bereits überlappen, konvergieren beide Varianten bis auf wenige Tausendstel Nat. Das zeigt der Benchmark mit einer Gauß-Mischverteilung mit 3 Modi aus :numref:`subsec_gan_objectives_testbed`:

<!--en:2288bd7b-->
Beide Varianten nähern die Datenverteilung bis auf wenige Tausendstel Nat oder besser an. Die nicht sättigende Variante unterschreitet in allen Läufen die Schwelle von 0.01 Nat zwischen Schritt 2200 und Schritt 2600. Dasselbe galt für den logistischen Verlust aus :numref:`sec_gan_objectives`. Die Paarbildung kostet also keine Anpassungsgüte. Die sättigende Variante unterschreitet die Schwelle sogar früher, in jedem Lauf bei Schritt 1800, denn ihr Gewicht ist in der Nähe der Daten nicht klein.

<!--en:2b05de8b-->
Startet der Generator dagegen weit entfernt von den Daten ($\theta_0 = 6$ in `d2l.train_shift` aus :numref:`subsec_gan_objectives_separated_start`), so erhält jedes generierte Sample einen viel niedrigeren Score als jedes echte Beispiel ($D(x') \ll D(x)$). Dadurch geht das sättigende Ranggewicht $w_{\textrm{Rp}}(x')$ gegen null:

<!--en:b9dc8c9e-->
Beim Start mit $\theta_0 = 6$ liegt der Median der Gradienten beim Generator mit sättigendem Paarverlust unter $5 \times 10^{-4}$. Selbst nach 1500 Schritten steckt er noch bei $\theta \approx 6.00$ fest. Der Generator mit nicht sättigendem Paarverlust (`rpgan_loss_G`) erhält dagegen Gradienten von $\sim 3$ und erreicht bei allen Zufallsstartwerten etwa in Schritt 330 den Bereich $|\theta| < 0.5$. Daher verwenden wir im restlichen Kapitel den nicht sättigenden Paarverlust.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:637dea50-->
Die Zielfunktion für die relativistische Paarbildung (RpGAN) maximiert die Bradley-Terry-Log-Likelihood $\Phi(D) = \mathbb{E}_{x \sim p, x' \sim q}[\log \sigma(D(x) - D(x'))]$ dafür, dass ein echtes Beispiel höher eingestuft wird als ein generiertes Sample. Damit erhalten wir bis auf eine additive Konstante das logarithmierte Dichteverhältnis $D^\star(x) = \lambda(x) = \log(p(x)/q(x))$. Betrachten wir $(a_1, a_2)$ auf dem Produktraum $\mathcal{X} \times \mathcal{X}$, so ergibt sich für den kalibrierten Optimalwert $d_{\mathrm{Rp}}(p, q) = \sup_D \Phi(D) + \log 2$ der Ausdruck $\mathrm{JS}(p \otimes q, q \otimes p) = 2\,\mathrm{JS}(p, q) - I(a_1; a_2)$ mit den Schranken $\mathrm{JS}(p, q) \leq d_{\mathrm{Rp}}(p, q) \leq \min(2\,\mathrm{JS}(p, q), \log 2)$. Eine feste Eins-zu-eins-Kopplung $\mathcal{E}_{\mathrm{diag}}$ beseitigt im diskreten Fall die Kollisionsminima mit Modusausfall, wenn der Kritiker seine beste Antwort liefert. Ziehen wir dagegen unabhängig neue Minibatches, so erhalten wir im Mittel wieder die Zielfunktion $\mathcal{E}_{\mathrm{all}}$ über alle Paare. Ihre Kollisionsminima bleiben bestehen. Liegen generierte Samples weit von den Daten entfernt, so geht das sättigende Ranggewicht $w_{\textrm{Rp}}(x') = \mathbb{E}_{x \sim p}[\sigma(D(x') - D(x))]$ gegen null. Das nicht sättigende Gewicht $1 - w_{\textrm{Rp}}(x')$ bleibt dagegen nahe $1$ und erhält den Generatorgradienten.

<!--en:2294243a-->
## Übungen

<!--en:d378be3e-->
1. [code] **Sättigende und nicht sättigende Ranggewichte.** Betrachten Sie einen diskreten Raum mit 5 Zuständen, der Datenverteilung $p = (0.35, 0.25, 0.20, 0.12, 0.08)$ und der Generatorverteilung $q = (0.02, 0.05, 0.13, 0.30, 0.50)$.
    1. Berechnen Sie mit `js` aus der Codezelle für zwei Zustände die Divergenzen $\mathrm{JS}(p, q)$ und $d_{\mathrm{Rp}}(p, q)$. Prüfen Sie, dass $\mathrm{JS}(P, Q)$ auf den 25 Paarzuständen mit $\sup_D \Phi(D) + \log 2$ bei $D = \lambda$ übereinstimmt (die beiden Divergenzen betragen $\approx 0.234$ und $0.377$).
    1. Bestimmen Sie für jeden Zustand das sättigende Gewicht $w_{\mathrm{Rp}}$ (:eqref:`eq_gan_rp_weights`) und das nicht sättigende Gewicht $1 - w_{\mathrm{Rp}}$ bei $D = \lambda$. Vergleichen Sie ihre Erwartungswerte unter $q$.
    1. Zeigen Sie für $D = \lambda$, dass $\mathbb{E}_{x' \sim q}[w_{\mathrm{Rp}}(x')] = \iint \frac{P Q}{P + Q}\, da_1\, da_2 \leq \tfrac12$ gilt, mit Gleichheit genau dann, wenn $q = p$. Zeigen Sie außerdem, dass dieser Erwartungswert gegen null geht, wenn sich die Träger von $p$ und $q$ voneinander trennen.
    1. Führen Sie `d2l.train_shift` mit dem Paarkritiker und beiden Generatorverlusten für die Anfangsverschiebungen `theta0` $\in \{1, 2, 3, 4\}$ aus (je drei Zufallsstartwerte). Bestimmen Sie so den Abstand, bei dem der sättigende Generator stecken bleibt.
   :exlabel:`ex_relativistic-zero-sum-weights`

<!--en:d3b008b1-->
1. **Differenz zwischen $d_{\mathrm{Rp}}$ und $2\,\mathrm{JS}(p, q)$.** ● Betrachten Sie das zufällig angeordnete Paar $(a_1, a_2)$ und sein Label $y$ für die Reihenfolge aus :eqref:`eq_gan_rp_mi`.
    1. Zeigen Sie, dass $a_1$ und $a_2$ bei gegebenem $y$ bedingt unabhängig sind und dass $I(a_k; y) = \mathrm{JS}(p, q)$ für $k \in \{1, 2\}$ gilt.
    1. Nehmen Sie an, $p$ und $q$ haben disjunkte Träger. Berechnen Sie $I(a_1; a_2)$ direkt aus der Mischverteilung der Paare $\tfrac12(P + Q)$ und prüfen Sie, dass $I(a_1; a_2) = \mathrm{JS}(p, q) = \log 2$ gilt.
    1. Zeigen Sie für eine lokale Störung $q = p\,(1 + \epsilon v)$ mit $\mathbb{E}_p[v] = 0$, dass $\mathrm{JS}(p, q) = \Theta(\epsilon^2)$ gilt, während $I(a_1; a_2) = \Theta(\epsilon^4)$ gilt. Folgern Sie daraus $d_{\mathrm{Rp}}(p, q) / \mathrm{JS}(p, q) \to 2$ für $\epsilon \to 0$.
    1. Prüfen Sie $d_{\mathrm{Rp}}(p, q) = 2\,\mathrm{JS}(p, q) - I(a_1; a_2)$ numerisch anhand der Verteilung mit 5 Zuständen aus Übung :exref:`ex_relativistic-zero-sum-weights`.
   :exlabel:`ex_relativistic-gap-twice-js`

<!--en:90c57ade-->
1. **Zwei Gauß-Verteilungen.** Gegeben seien $p = \mathcal{N}(0, I_d)$ und $q = \mathcal{N}(\mu, I_d)$ auf $\mathbb{R}^d$.
    1. Zeigen Sie, dass $\lambda(x)$ affin in $x$ und dass $u = \lambda(x') - \lambda(x)$ in :eqref:`eq_gan_rp_explicit` gaußverteilt ist. Bestimmen Sie den Mittelwert und die Varianz dieser Gauß-Verteilung.
    1. Bringen Sie $\mathrm{JS}(p, q)$ in dieselbe Form. Bestimmen Sie die Skalierungskonstante $c$, für die $\mathrm{JS}\big(p, \mathcal{N}(c\mu, I_d)\big) = d_{\mathrm{Rp}}(p, q)$ für alle $\mu$ gilt.
    1. Berechnen Sie $d_{\mathrm{Rp}}(p, q)$ und $\mathrm{JS}(p, q)$ durch eindimensionale Quadratur für $\|\mu\| \in \{0.1, 1, 2, 4\}$. Vergleichen Sie ihr Verhältnis mit den Grenzwerten für kleine und große Abstände.
   :exlabel:`ex_relativistic-two-gaussians`

<!--en:56c1a17f-->
1. **Zerlegbarkeit des Paarkritikers.** ● Ersetzen Sie $\log \sigma$ in :eqref:`eq_gan_rp` durch eine allgemeine konkave Auszahlungsfunktion $\ell$. Damit ergibt sich $\Phi_\ell(D) = \mathbb{E}_{x \sim p, x' \sim q}[\ell(D(x) - D(x'))]$.
    1. Zeigen Sie für die Auszahlungsfunktion der kleinsten Quadrate $\ell(t) = -(1 - t)^2$, dass der uneingeschränkt optimale Paarkritiker auf $\mathcal{X} \times \mathcal{X}$ die Form $2\eta - 1 = \tanh\big(\tfrac{\lambda(a_1) - \lambda(a_2)}{2}\big)$ mit $\eta = P/(P + Q)$ hat und sich nicht additiv als $D(a_1) - D(a_2)$ zerlegen lässt.
    1. Berechnen Sie am Beispiel mit 5 Zuständen aus Übung :exref:`ex_relativistic-zero-sum-weights` die Differenz zwischen dem uneingeschränkten Optimum für Paare und $\sup_{D \in \mathbb{R}^5} 2\,\Phi_\ell(D)$.
    1. Leiten Sie für die logistische Auszahlungsfunktion mit ungleicher A-priori-Wahrscheinlichkeit $\alpha \neq \tfrac12$ für die beiden Reihenfolgen den bayesoptimalen Paarkritiker her. Zeigen Sie, wie Sie die Klasse der Differenzen $D(a_1) - D(a_2)$ erweitern müssen, um dieses Optimum zu erreichen.
   :exlabel:`ex_relativistic-separability-choice-loss-replace`

<!--en:bc16e062-->
1. [code] **Feste Kopplung und zufällige Neuzuordnung der Paare.** ●●
    1. Beweisen Sie die in `losses` für $\mathcal{E}_{\mathrm{diag}}$ verwendete Zyklusregel. Auf jeder gerichteten Kante außerhalb eines Zyklus kann der Kritiker $D(x_i) - D(x'_i) \to +\infty$ erreichen, sodass sein Verlust verschwindet. Auf einem $k$-Zyklus summieren sich die Differenzen dagegen zu null. Wegen der Konkavität von $\log \sigma$ ergibt sich bei gleichen Scores ein Maximum von $-k\log 2$.
    1. Erweitern Sie `count_bad_minima` auf $n = 4$ (256 Konfigurationen) und prüfen Sie die Anzahlen für $\mathcal{E}_{\mathrm{GAN}}$, $\mathcal{E}_{\mathrm{all}}$ und $\mathcal{E}_{\mathrm{diag}}$.
    1. Definieren Sie die über Permutationen gemittelte Zielfunktion $\frac{1}{n!} \sum_\pi \sup_D \frac{1}{n} \sum_i \log \sigma(D(x_i) - D(x'_{\pi(i)}))$ für die jeweils beste Antwort des Kritikers. Zählen Sie ihre strikten suboptimalen lokalen Minima für $n = 3$. Vergleichen Sie diesen Fall mit einem Kritiker, der sich langsam über viele Minibatches mit zufälliger Paarbildung hinweg anpasst (seine Zielfunktion ist $\mathcal{E}_{\mathrm{all}}$).
   :exlabel:`ex_relativistic-random-repairing`

<!--en:adb62eb1-->
[Diskussion](https://d2l.discourse.group/t/36810)

<!--en:6b6051e4-->
Relativistische Zielfunktionen und paarweises Ranking<br>**Paarkritiker nach Bradley und Terry · Verschiebungsinvarianz · JS-Divergenz von Produktmaßen · Informationsschranken · Verlustlandschaften bei fester und zufälliger Paarbildung**

<!--en:fa64a2ec-->
::: {.slide title="Von absoluten Schwellen zum paarweisen Ranking"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:1e28327e-->
Ein Diskriminator für einzelne Beispiele vergleicht den Score jeder Beobachtung $x$ mit der festen Schwelle null ($D(x) \gtrless 0$). Paarweise Präferenzmodelle wie die Bradley-Terry-Belohnungsmodelle beim RLHF sagen anhand der Score-Differenz vorher, welcher von zwei Kandidaten besser ist:

<!--en:5088f224-->
- **Empfindlichkeit gegenüber Score-Verschiebungen:** Eine Konstante $b$ zu allen Kritiker-Scores zu addieren, kann Entscheidungen an der Schwelle $0$ umkehren. Die relative Reihenfolge echter Beispiele und generierter Samples bleibt dabei gleich.

<!--en:a1121de6-->
- **Relativistische Paarbildung (RpGAN, Jolicoeur-Martineau, 2019):** Ein echtes Beispiel $x \sim p$ und ein generiertes Sample $x' \sim q$ bilden ein Paar. Aus der Score-Differenz $D(x) - D(x')$ ergibt sich die Vorhersage, welches Element echt ist.

<!--en:0bfa90d6-->
- **Drei Kernfragen:** (1) Schätzt der optimale Ranking-Kritiker weiterhin $\lambda(x) = \log(p(x)/q(x))$? (2) Welche Divergenz misst er? (3) Wann verhindert paarweises Ranking einen Moduskollaps?

<!--en:5720a4f4-->
::: {.slide title="Die paarweise Zielfunktion nach Bradley und Terry (RpGAN)"}

<!--en:4704b0ac-->
[Formulierung und Implementierung]{.kicker}

<!--en:c1ec06c7-->
Gegeben sei ein Paar $(x, x')$ mit unabhängig gezogenen Elementen $x \sim p$ und $x' \sim q$. Die relativistische Zielfunktion modelliert die Wahrscheinlichkeit, dass $x$ einen höheren Rang als $x'$ erhält, mit der **Bradley-Terry-Likelihood** $\sigma(D(x) - D(x'))$:

<!--en:e37eeb14-->
- Der Kritiker maximiert $\Phi(D)$ (minimiert $-\Phi(D) = \mathbb{E}[\operatorname{softplus}(D(x') - D(x))]$ über einander zugeordnete Paare $(x_i, x'_i)$ im Minibatch). Der in der Praxis verwendete nicht sättigende Generatorverlust `rpgan_loss_G` vertauscht die Argumente:

<!--en:fd363a8e-->
::: {.slide title="Verschiebungsinvarianz und optimaler Paarkritiker"}

<!--en:8fc0c5eb-->
[Satz zur Optimalität]{.kicker}

<!--en:e8eaf26e-->
- **Verschiebungsinvarianz:** Für jede Konstante $b \in \mathbb{R}$ gilt $\Phi(D + b) = \Phi(D)$. Das Vorzeichen eines einzelnen Scores $D(x')$ ist beliebig. Entscheidend ist die **Genauigkeit des paarweisen Rankings** $P(D(x) > D(x'))$.

<!--en:c123e5dc-->
- **Eindeutiger optimaler Ranking-Kritiker:** Wegen der strengen Konkavität von $\log \sigma$ liefert das Nullsetzen der Funktionalableitung $\delta \Phi / \delta D(t)$ den Maximierer $D^\star(x) = \lambda(x) + c$, eindeutig bis auf eine additive Verschiebung $c$.

<!--en:384990e3-->
::: {.slide title="Rechenbeispiel zum paarweisen Ranking bei zwei Zuständen"}

<!--en:b1fd6283-->
[Rechnung im Einzelnen]{.kicker}

<!--en:b6177ede-->
Für den Zustandsraum $\{0, 1\}$ mit $p = (0.8, 0.2)$ und $q = (0.2, 0.8)$ lassen sich die Produktverteilungen $P = p \otimes q$ (echtes Beispiel zuerst) und $Q = q \otimes p$ (echtes Beispiel an zweiter Stelle) vergleichen:

<!--en:5a7ad8de-->
| Geordnetes Paar $(a_1, a_2)$ | $(0, 0)$ (gleich) | $(1, 1)$ (gleich) | $(0, 1)$ (gemischt) | $(1, 0)$ (gemischt) |

<!--en:8ddcabeb-->
| **$P = p \otimes q$ (echtes Beispiel zuerst)** | $0.16$ | $0.16$ | **$0.64$** | $0.04$ |

<!--en:84b72ec3-->
| **$Q = q \otimes p$ (echtes Beispiel an zweiter Stelle)** | $0.16$ | $0.16$ | $0.04$ | **$0.64$** |

<!--en:eb156980-->
- Die Paare $(0, 0)$ und $(1, 1)$ verraten nichts über die Reihenfolge. Beim gemischten Paar $(0, 1)$ lässt sich das echte Element mit der A-posteriori-Wahrscheinlichkeit $\frac{0.64}{0.64 + 0.04} = \frac{16}{17} \approx 94.1\%$ bestimmen (gegenüber $80\%$ bei einer einzelnen Beobachtung).

<!--en:2a3ddb1d-->
- Bei $D^\star(0) - D^\star(1) = \lambda(0) - \lambda(1) = \log 16$ beträgt der Gewinn $\sup_D \Phi(D) + \log 2 \approx 0.319$ Nat und entspricht $\mathrm{JS}(P, Q)$. Das ist $1.66\times$ so viel wie $\mathrm{JS}(p, q) \approx 0.193$ Nat:

<!--en:22143f9d-->
::: {.slide title="Übergang zum Produktraum: Warum Differenzen genügen"}

<!--en:38fb7b94-->
[Das Lemma zum Produktraum]{.kicker}

<!--en:07bca067-->
- **Spiel mit logarithmischem Verlust auf dem Produktraum:** Das geordnete Paar $(a_1, a_2)$ als einzelne Ziehung aus $P = p \otimes q$ (echtes Beispiel zuerst) oder $Q = q \otimes p$ (echtes Beispiel an zweiter Stelle) auffassen.

<!--en:dfad0bd0-->
- **Gleiches Supremum bei Beschränkung:** Der uneingeschränkte Maximierer $\mathcal{D}^\star(a_1, a_2) = \lambda(a_1) - \lambda(a_2)$ für Paare hat bereits Differenzform. Die Beschränkung auf Kritiker der Form $\mathcal{D}(a_1, a_2) = D(a_1) - D(a_2)$ liefert daher dasselbe Supremum $V_{P,Q} = 2\,\Phi(D)$.

<!--en:b6f033e4-->
::: {.slide title="Satz über die kalibrierte Paardivergenz"}

<!--en:fab0cf90-->
[Jensen-Shannon-Divergenz auf dem Produktraum]{.kicker}

<!--en:1096a682-->
- **Darstellung über Entropien und als expliziter Erwartungswert:** Die Divergenz beträgt $H(\frac{1}{2}(p \otimes q + q \otimes p)) - H(p) - H(q) = \log 2 - \mathbb{E}_{p,q}[\log(1 + \frac{q(x)p(x')}{p(x)q(x')})]$.

<!--en:d60f9f9b-->
- **Deutung als gegenseitige Information:** Für ein zufällig angeordnetes Paar $(a_1, a_2)$ mit dem Label $y \in \{0, 1\}$ für die Reihenfolge gibt $d_{\mathrm{Rp}}(p, q) = I((a_1, a_2); y)$ an, wie viel Information das Paar darüber liefert, welches Element echt ist.

<!--en:ac634560-->
::: {.slide title="Informationsschranken für d_Rp im Vergleich zur JS-Divergenz"}

<!--en:63971820-->
[Kettenregel für die gegenseitige Information]{.kicker}

<!--en:b25b22fe-->
- **Nahe dem Gleichgewicht ($q \to p$):** Die Redundanz $I(a_1; a_2) = \Theta(\|p - q\|^4)$ geht schneller gegen null als $\mathrm{JS}(p, q) = \Theta(\|p - q\|^2)$. Daher gilt $d_{\mathrm{Rp}}(p, q) \approx 2\,\mathrm{JS}(p, q)$.

<!--en:80302fd1-->
- **Allgemeine untere und obere Schranke:** $\mathrm{JS}(p, q) \le d_{\mathrm{Rp}}(p, q) \le \min(2\,\mathrm{JS}(p, q), \log 2)$. Bei disjunkten Trägern ist der Sättigungswert weiterhin $\log 2$.

<!--en:7f9bbdc9-->
::: {.slide title="Das relativistische Ranggewicht bei Generatorupdates"}

<!--en:53d1dafc-->
[Gradientengewichtung je Sample]{.kicker}

<!--en:b00f11f1-->
- **Sättigendes und nicht sättigendes Ranggewicht:** $w_{\textrm{Rp}}(x')$ ist die erwartete Gewinnwahrscheinlichkeit von $x'$ gegenüber echten Daten $x \sim p$. `rpgan_loss_G` verwendet $1 - w_{\textrm{Rp}}(x') = \mathbb{E}_p[\sigma(D(x) - D(x'))] \approx 1$, wenn $D(x') \ll D(x)$ gilt.

<!--en:39568240-->
- **Verschiebungsinvarianter Vergleich:** Maßstab für $D(x')$ ist die aktuelle Verteilung der Scores $D(x)$ echter Beispiele anstelle einer festen Schwelle bei null.

<!--en:facbc8de-->
::: {.slide title="Modusabdeckung im diskreten Fall: Drei Verlustlandschaften"}

<!--en:987570b1-->
[Analyse von Sun, Fang und Schwing]{.kicker}

<!--en:ef6f19f6-->
- **Drei Arten der Kopplung:** $\mathcal{E}_{\mathrm{GAN}} = \mathrm{JS}(\hat{p}, \hat{q}) - \log 2$ (einzelne Beispiele), $\mathcal{E}_{\mathrm{diag}}$ (feste 1-zu-1-Kopplung $x_i \leftrightarrow x'_i$) und $\mathcal{E}_{\mathrm{all}} = d_{\mathrm{Rp}}(\hat{p}, \hat{q}) - \log 2$ (alle $n^2$ Paare)

<!--en:17c54863-->
- **Kollisionen ($n^n - n!$ Konfigurationen):** Jeder generierte Punkt $x'_i$ liegt auf einem echten Punkt $\{x_1, \ldots, x_n\}$. Mindestens zwei generierte Punkte fallen zusammen, sodass ein echter Modus unbesetzt bleibt.

<!--en:a8a71a1b-->
::: {.slide title="Ein Schritt aus der Kollision bei festen Paaren und bei allen Paaren"}

<!--en:19d5f5e9-->
[Zyklusregel für gerichtete Graphen]{.kicker}

<!--en:08e935b0-->
Bei fester Paarbildung $\mathcal{E}_{\mathrm{diag}}$ von jedem Index $i$ eines echten Punktes einen Pfeil zur Position des zugeordneten generierten Punktes $x'_i$ zeichnen. Auf jedem gerichteten Zyklus heben sich die Score-Differenzen auf: $\sum (D(x_i) - D(x'_i)) = 0$. Wegen der Konkavität von $\log\sigma$ sind die Scores im Optimum gleich ($-\log 2$ je Zykluskante). Auf Kanten außerhalb eines Zyklus liefert $D(x_i) - D(x'_i) \to +\infty$ den Wert $0$:

<!--en:7f3ea63d-->
- Kollidieren beide generierten Punkte auf $x_1$ (`[0, 0]`) und macht $x'_2$ einen Schritt in Richtung $x_2$ (`[0, 0.1]`), so **steigen** sowohl $\mathcal{E}_{\mathrm{GAN}}$ als auch $\mathcal{E}_{\mathrm{all}}$ strikt (von $-0.477$ und $-0.347$ auf $-0.347$ und $-0.173$). Dadurch bleibt $x'_2$ in der Kollision gefangen.

<!--en:d7661e01-->
- Bei fester Paarbildung $\mathcal{E}_{\mathrm{diag}}$ liefert die Schleife $1 \to 1$ wegen gleicher Scores weiter $-\log 2$. Die Kante $2 \to x'_2$ außerhalb eines Zyklus liefert weiter $0$. Damit bleibt der Verlust **völlig konstant** bei $-0.5\log 2 \approx -0.347$, bis unmittelbar vor $x_2$.

<!--en:c7e13e66-->
::: {.slide title="Abzählung suboptimaler Kollisionsminima"}

<!--en:ea5dd031-->
[Feste Kopplung und Neuzuordnung im Minibatch]{.kicker}

<!--en:a97188e2-->
Über alle Zuordnungen in $\{0, \ldots, n-1\}^n$ ($n = 2, 3$) die Kollisionen zählen, die strikte lokale Minima sind:

<!--en:d13cacf2-->
::: {.slide title="Warum beim Minibatch-Training die Kollisionen aller Paare auftreten"}

<!--en:e623fc07-->
[Stolperstein in der Praxis]{.kicker}

<!--en:8a452706-->
- **Das Ergebnis von Sun et al. (2020):** Die Zielfunktion $\mathcal{E}_{\mathrm{diag}}$ für die feste Eins-zu-eins-Kopplung hat **$0$** suboptimale lokale Kollisionsminima ($0$ gegenüber $n^n - n!$). Jeder kollidierende Punkt $x'_i$ außerhalb eines gerichteten Zyklus kann sich auf einem flachen Plateau in Richtung $x_i$ bewegen.

<!--en:1dc1d638-->
- **Neuzuordnung im Minibatch:** In jedem Schritt unabhängig neue Minibatches aus $p$ und $q$ zu ziehen, mittelt über zufällige Paarungen. Das ergibt den **Erwartungswert über alle Paare** $\mathcal{E}_{\mathrm{all}} = d_{\mathrm{Rp}}(\hat{p}, \hat{q}) - \log 2$ mit allen $n^n - n!$ Kollisionsminima.

<!--en:3bb4f5ae-->
- **Fazit:** In §{sec_gan_convergence} bestätigt das $2 \times 2$-Experiment mit 25 Gauß-Verteilungen, dass die $R_1 + R_2$-Gradientenstrafterme für die Modusabdeckung ausschlaggebend sind. Paarbildung allein genügt nicht.

<!--en:bb42fd8a-->
::: {.slide title="Sättigender und nicht sättigender Paarverlust nahe den Daten"}

<!--en:b06a0c81-->
[Benchmark mit überlappenden Verteilungen]{.kicker}

<!--en:6a6444c0-->
Überlappen sich $p$ und $q$ bereits, wie im Benchmark mit einer 2D-Gauß-Mischverteilung mit 3 Komponenten ($\mathrm{JS}(p, q_0) = 0.529$ Nat), so konvergieren beide Varianten bis auf wenige Tausendstel Nat. Das gilt sowohl für den sättigenden Generatorverlust für Paare ($\min_G \Phi(D)$, Gewicht $w_{\textrm{Rp}}$) als auch für den nicht sättigenden Paarverlust (`rpgan_loss_G`, Gewicht $1 - w_{\textrm{Rp}}$):

<!--en:63532489-->
- Nahe der Datenverteilung erhalten generierte Samples häufig einen höheren Rang als echte Beispiele ($w_{\textrm{Rp}}(x') \approx \frac{1}{2}$). Daher sättigt keines der Gewichte. Die sättigende Variante erreicht $\mathrm{JS} < 0.01$ Nat sogar etwas früher (Schritt $1{,}800$ gegenüber $2{,}200\text{--}2{,}600$).

<!--en:8b6caa9a-->
::: {.slide title="Sechs Einheiten Abstand: Stillstand beim sättigenden Paarverlust"}

<!--en:a50ee01c-->
[Benchmark mit getrennten Startverteilungen]{.kicker}

<!--en:5e65481b-->
Startet der Generator weit von den Daten entfernt bei $\theta_0 = 6.0$ im eindimensionalen Verschiebungsproblem mit Gauß-Verteilungen ($p = \mathcal{N}(0, 1)$, $q_\theta = \mathcal{N}(\theta, 1)$), so rangiert jedes generierte Sample weit unter jedem echten Beispiel ($D(x') \ll D(x)$). Damit gilt $w_{\textrm{Rp}}(x') = \mathbb{E}_p[\sigma(D(x') - D(x))] \to 0$:

<!--en:3e06eb82-->
**Warum stets `rpgan_loss_G` zum Einsatz kommt:** Bei $\theta_0 = 6.0$ liegt der Median der Gradienten für den Generator mit sättigendem Paarverlust unter $5 \times 10^{-4}$. Er steckt bei $\theta = 6.00$ fest, selbst nach $1{,}500$ Schritten. Der Generator mit nicht sättigendem Paarverlust ($1 - w_{\textrm{Rp}} \approx 1$) erhält dagegen einen Gradienten von $\approx 3.1$ und erreicht $|\theta| < 0.5$ etwa bei Schritt $330$.

<!--en:c26da8a5-->
::: {.slide title="Verständnisfragen zu relativistischen Kritikern und Paarbildung"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:b4df2e32-->
Fragen zu relativistischen paarweisen Zielfunktionen:

<!--en:fcb90de7-->
1. **Verhalten des Divergenzverhältnisses $d_{\mathrm{Rp}}(p, q) / \mathrm{JS}(p, q)$:** Im Beispiel mit zwei Zuständen gilt $d_{\mathrm{Rp}}(p, q) / \mathrm{JS}(p, q) = 0.319 / 0.193 \approx 1.66$. Welchem Wert nähert sich das Verhältnis für $q \to p$? Welchen Wert erreicht es, wenn $p$ und $q$ disjunkte Träger haben?

<!--en:64c370cc-->
2. **Ausweg bei fester Paarbildung:** Für $n = 2$ mit der festen Paarbildung $x_1 \leftrightarrow x'_1$ und $x_2 \leftrightarrow x'_2$ mögen beide generierten Punkte auf $x_1$ kollidieren ($x'_1 = x'_2 = x_1$). Welchen Wert hat der Paarverlust $\mathcal{E}_{\mathrm{diag}}$ bei bester Antwort des Kritikers? Wie verhält er sich, wenn sich $x'_2$ stetig von $x_1$ nach $x_2$ bewegt?

<!--en:557b98b6-->
**Lösungen:**

<!--en:b401ae0b-->
1. Aus $d_{\mathrm{Rp}}(p, q) = 2\,\mathrm{JS}(p, q) - I(a_1; a_2)$ folgt für $q \to p$ wegen $I(a_1; a_2) = \Theta(\|p - q\|^4)$ $\ll$ $\mathrm{JS}(p, q) = \Theta(\|p - q\|^2)$, dass das Verhältnis gegen **$2$** geht. Bei disjunkten Trägern erreichen sowohl $d_{\mathrm{Rp}}$ als auch $\mathrm{JS}$ den Sättigungswert $\log 2$. Das Verhältnis geht also gegen **$1$**.

<!--en:7e2c3354-->
2. Bei $x'_1 = x'_2 = x_1$ bildet die Kante $1 \to 1$ einen Zyklus der Länge 1 (Beitrag $-\frac{1}{2}\log 2$). Die Kante $2 \to 1$ liegt außerhalb eines Zyklus (Beitrag $0$). Das ergibt $\mathcal{E}_{\mathrm{diag}} = -\frac{1}{2}\log 2 \approx -0.347$. Bewegt sich $x'_2$ durch den leeren Raum auf $x_2$ zu, bleibt $\mathcal{E}_{\mathrm{diag}}$ **völlig konstant** bei $-\frac{1}{2}\log 2$. Der Verlust fällt erst auf $-\log 2$, wenn $x'_2$ auf $x_2$ trifft und den zweiten Zyklus der Länge 1 schließt ($2 \to 2$).

<!--en:461ab958-->
::: {.slide title="Verständnisfragen zu relativistischen Kritikern und Paarbildung (Forts.)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:aa55925c-->
3. **Deutung einzelner Kritiker-Scores:** Ein trainierter RpGAN-Kritiker liefert für ein generiertes Bild $D(x') = +3.5$. Bedeutet $D(x') > 0$, dass der Kritiker $x'$ eher für echt als für generiert hält?

<!--en:23db34b8-->
**Lösung:**

<!--en:f14baebd-->
3. **Nein.** Wegen der Verschiebungsinvarianz $\Phi(D + b) = \Phi(D)$ ist der Nullpunkt von $D$ beliebig. Aussagekräftig sind nur die paarweisen Differenzen $D(x) - D(x')$ (oder die Ranking-Genauigkeit $P(D(x) > D(x'))$).

<!--en:d3cb3134-->
::: {.slide title="Zusammenfassung: Zielfunktionen für die relativistische Paarbildung"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:68e3f327-->
| Konzept oder Eigenschaft | Mathematischer Ausdruck | Bezug zum klassischen GAN mit logarithmischem Verlust | Bedeutung für die Praxis |

<!--en:17f9a47a-->
| **Kritiker nach Bradley und Terry ($\Phi$)** | $\Phi(D) = \mathbb{E}_{x \sim p,\, x' \sim q}[\log\sigma(D(x) - D(x'))]$ | Paarweises Ranking ersetzt den Schwellenvergleich $D(x) \gtrless 0$ | Invariant gegenüber konstanten Verschiebungen $D \mapsto D + b$ |

<!--en:de2935c3-->
| **Optimaler Ranking-Kritiker** | $D^\star(x) = \lambda(x) + c = \log \frac{p(x)}{q(x)} + c$ | Bis auf die Verschiebung $c$ dasselbe logarithmierte Dichteverhältnis | Genauigkeit des paarweisen Rankings $P(D(x) > D(x'))$ anstelle der Vorzeichen beobachten |

<!--en:1c4fdadb-->
| **Kalibrierte Paardivergenz** | $d_{\mathrm{Rp}}(p, q) = \sup_D \Phi(D) + \log 2$ $= \mathrm{JS}(p \otimes q, q \otimes p)$ | $d_{\mathrm{Rp}}(p, q) = 2\,\mathrm{JS}(p, q) - I(a_1; a_2)$ | $\mathrm{JS} \le d_{\mathrm{Rp}} \le \min(2\,\mathrm{JS}, \log 2)$ mit Sättigung bei $\log 2$ |

<!--en:1f62d369-->
| **Relativistisches Ranggewicht** | $w_{\textrm{Rp}}(x') = \mathbb{E}_{x \sim p}[\sigma(D(x') - D(x))]$ | Vergleich von $D(x')$ mit der Verteilung der Scores $D(x)$ echter Beispiele | Nicht sättigendes Gewicht $1 - w_{\textrm{Rp}}(x')$ verwenden (`rpgan_loss_G`) |

<!--en:d5857cf3-->
| **Verlustlandschaft bei festen Paaren und bei allen Paaren** | $\mathcal{E}_{\mathrm{diag}}$ (feste Zuordnung $x_i \leftrightarrow x'_i$) gegenüber $\mathcal{E}_{\mathrm{all}}$ ($n^2$ Paare) | $\mathcal{E}_{\mathrm{diag}}$ hat $0$ Kollisionsminima, $\mathcal{E}_{\mathrm{all}}$ hat $n^n - n!$ | Zufällige Neuzuordnung im Minibatch ergibt im Mittel $\mathcal{E}_{\mathrm{all}}$ |
