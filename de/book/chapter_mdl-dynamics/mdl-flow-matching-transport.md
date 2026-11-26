<!--en:472a0b2c-->
# Flow-Matching und optimaler Transport

<!--en:45ee1652-->
Die Sampler des vorigen Abschnitts werten den Score einmal pro Schritt aus, tausendmal für die Rückwärtskette und einige Dutzend Mal für ihre deterministischen Sprünge. Sie brauchen so viele Schritte, weil die Pfade vom Rauschen zu den Daten gekrümmt sind. Ein numerisches Verfahren kann einem gekrümmten Pfad nur in kurzen Schritten folgen. Niemand hat diese Pfade entworfen. Sie ergeben sich aus dem Prozess, der die Daten verrauscht. Können wir den Pfad vom Rauschen zu den Daten selbst wählen, und zwar so, dass ein numerisches Verfahren große Schritte machen kann?

<!--en:d3cf585e-->
Stellen wir uns zwei Datenpunkte bei $-2$ und $+2$ vor und eine Wolke aus gaußschem Rauschen, die auf diesen Punkten enden soll. Bewegte sich jedes Teilchen mit konstanter Geschwindigkeit geradlinig auf seinen Zielpunkt zu, brächte es ein einziger Euler-Schritt der Länge eins genau dorthin. Ein Geschwindigkeitsfeld (engl. velocity field) ist aber eine einzige Funktion von Ort und Zeit. Teilchen mit verschiedenen Zielen kommen am selben Ort vorbei, und dort kann das Feld nur in eine Richtung zeigen.

<!--en:666805ff-->
Flow-Matching :cite:`Lipman.Chen.BenHamu.ea.2022` und Rectified Flow :cite:`Liu.Gong.Liu.2022` treffen diese Wahl: Sie geben den Pfad vor und lernen nur die Geschwindigkeit, mit der sich Masse entlang dieses Pfades bewegt. Welche Pfade die geringsten Kosten haben, ist eine viel ältere Frage, das Problem des *optimalen Transports*. Gaspard Monge fragte 1781, wie man einen Erdhaufen in eine Grube befördert. Misst man die Kosten mit dem quadrierten Abstand, erweist sich genau die eben gewünschte Bewegung als die günstigste: Jedes Teilchen bewegt sich geradlinig mit konstanter Geschwindigkeit :cite:`McCann.1997,Benamou.Brenier.2000`.

<!--en:dbeef422-->
Eine einzige Regression lernt die Geschwindigkeit jedes Pfades, den wir aus einfachen Teilen zusammensetzen. Denn diese Geschwindigkeit ist ein Mittelwert über Teilchen, und die Methode der kleinsten Quadrate lernt Mittelwerte (:numref:`sec_mdl-flow-matching`). Auch gerade Teilstücke ergeben einen gekrümmten Fluss. Wenn wir Rauschen und Daten neu paaren, begradigen wir ihn (:numref:`sec_mdl-rectified-flow`). Der optimale Transport beschreibt das Ziel dieser neuen Paarung (:numref:`sec_mdl-ot-connection`). Zwei Teile können bis zur zweiten Lektüre warten: die Umrechnungen zwischen Score, Rauschen und Geschwindigkeit nach der Definition des Gauß-Pfades in :numref:`sec_mdl-score-velocity-dictionary` sowie der Vergleich der numerischen Verfahren in :numref:`sec_mdl-sampling-learned-dynamics`.

<!--en:bbd7cd40-->
::: {.callout-important title="Eine gemeinsame Zeitrichtung"}

<!--en:ea79e332-->
Die Zeit läuft wie im vorigen Abschnitt von den Daten bei $t = 0$ zum Rauschen, das nun bei $t = 1$ liegt. Das gilt für Scores und Geschwindigkeiten gleichermaßen: Beim Training verrauschen wir die Daten in Vorwärtsrichtung, und jeder Sampler integriert von $t = 1$ zurück bis $t = 0$.

<!--en:6040110b-->
Die hier zitierten Arbeiten zum Flow-Matching und der zugehörige Code verwenden die entgegengesetzte Zeitrichtung, vom Rauschen bei $0$ zu den Daten bei $1$ (:numref:`fig_mdl-dyn-time-conventions`). Um eine Formel mit denen dieser Arbeiten zu vergleichen, setzen wir $1 - t$ für deren $t$ ein, lesen ihr Rauschen $\mathbf{x}_0$ als unser $\boldsymbol{\epsilon}$ und ihre Daten $\mathbf{x}_1$ als unser $\mathbf{x}_0$ und kehren das Vorzeichen jeder Geschwindigkeit um.

<!--en:dda09edd-->
![Die Zeitrichtungen in diesem Buch und in der Literatur. In diesem Buch (oben) liegen die Daten bei $t = 0$ und das Rauschen bei $t = 1$. Die Verrauschung läuft vorwärts in der Zeit. Beim Sampling wird ein Score oder eine Geschwindigkeit von $1$ bis $0$ integriert. In den Arbeiten zum Flow-Matching (unten) liegt das Rauschen bei $0$ und die Daten liegen bei $1$. Das Sampling läuft vorwärts. Die Dichten an den Endpunkten sind gleich. Die Substitution $t \mapsto 1 - t$ überführt die eine Zeitkonvention in die andere und kehrt das Vorzeichen jeder Geschwindigkeit um.](../img/mdl-dyn-time-conventions.svg)

<!--en:8ded8d2d-->
Für die Daten gilt $\mathbf{x}_0 \sim p_0$, und $p_t$ ist ein Pfad von Dichten von $p_0$ bis zum Rauschen $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$. Für ein Sample vom Rauschende verwenden wir zwei Bezeichnungen: $\boldsymbol{\epsilon}$ für gaußsches Rauschen wie im vorigen Abschnitt und $\mathbf{x}_1$ in Aussagen über Paarungen, die für zwei beliebige Endpunktverteilungen gelten.

<!--en:88d3a2a1-->
## Flow-Matching

<!--en:151e5647-->
### Wahrscheinlichkeitspfade und Geschwindigkeitsfelder

<!--en:3446f349-->
Bei der Diffusion ergab sich der Pfad aus dem Verrauschungsprozess. Um diesen rückwärts laufen zu lassen, mussten wir den Score lernen. Hier können wir den Pfad selbst wählen. Ein **Wahrscheinlichkeitspfad** (engl. probability path) ist eine Familie von Dichten $(p_t)_{t \in [0, 1]}$, wobei $p_0$ die Daten und $p_1$ das Rauschen beschreibt. Ein Geschwindigkeitsfeld $\mathbf{u}_t$ *realisiert* den Pfad, wenn beide zusammen die Kontinuitätsgleichung aus :numref:`sec_mdl-continuity-equation` erfüllen. Sie besagt, dass sich eine vom Geschwindigkeitsfeld bewegte Dichte um den Nettozufluss an Masse ändert,

<!--en:3c8c5ff5-->
Ist das Feld Lipschitz-stetig, so integrieren wir $\dot{\mathbf{x}} = \mathbf{u}_t(\mathbf{x})$ ausgehend von einem Sample aus $p_1$ bei $t = 1$ zurück bis $t = 0$. Damit erhalten wir zu jedem Zeitpunkt ein Sample aus $p_t$ und bei $t = 0$ eines aus der Datenverteilung. Beachten wir die Richtung: Das Feld zeigt in Richtung der Verrauschung, von den Daten zum Rauschen. Ein Sampler geht in die Gegenrichtung. Approximiert ein Netz $\mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ das Feld $\mathbf{u}_t$, so generieren wir Daten, indem wir ausgehend vom Rauschen eine gewöhnliche Differentialgleichung (engl. ordinary differential equation, ODE) lösen. Ein solches Modell ist ein kontinuierlicher normalisierender Fluss (:numref:`sec_mdl-continuous-normalizing-flows`), der rückwärts in der Zeit läuft. Seine Flussabbildung von $t = 1$ nach $t = 0$ ist der Generator. Mit Flow-Matching trainieren wir ihn, ohne die ODE auch nur einmal zu lösen. Als Zielfunktion bietet sich der **Flow-Matching-Verlust** an,

<!--en:0aaf1a93-->
Er ist der mittlere quadratische Fehler zwischen Netzvorhersage und wahrer Geschwindigkeit, gemittelt über Zeit und Teilchen der Wolke. In dieser Form hilft er uns nicht weiter, denn als Regressionsziel dient ein Feld, das wir nicht auswerten können: $\mathbf{u}_t$ hängt von der gesamten Datenverteilung ab.

<!--en:2b033149-->
Wir bauen den Pfad deshalb aus Teilen auf, die wir explizit angeben können. Die Variable $\mathbf{z}$, auf die wir bedingen, ziehen wir aus einer Verteilung $q$. Sie ist entweder ein Datenpunkt oder ein Paar aus einem Rauschpunkt und einem Datenpunkt. Für jeden Wert von $\mathbf{z}$ wählen wir eine kleine bewegliche Teilwolke $p_t(\mathbf{x} \mid \mathbf{z})$. Bei $t = 0$ sitzt sie auf dem Datenpunkt, bei $t = 1$ endet sie im Rauschen. Ist $\mathbf{z}$ ein Datenpunkt, verteilt sie sich über das gesamte Rauschen. Ist $\mathbf{z}$ ein Paar, endet sie auf dem zugehörigen Rauschpunkt. Für jede Teilwolke kennen wir die Geschwindigkeit $\mathbf{u}_t(\mathbf{x} \mid \mathbf{z})$ in geschlossener Form. Der Pfad ist die Mischung der Teilwolken, $p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{z})\, q(\mathbf{z})\, d\mathbf{z}$. Mit $\mathbf{x}_t$ bezeichnen wir ein Teilchen daraus: Wir ziehen $\mathbf{z}$ aus $q$ und dann einen Punkt aus der zugehörigen Teilwolke zum Zeitpunkt $t$. Welche Geschwindigkeit bewegt die Mischung? Auf der Antwort beruht der gesamte Abschnitt.

<!--en:6a93b6e6-->
**Proposition (die marginale Geschwindigkeit ist ein A-posteriori-Mittelwert).** *Für jedes $\mathbf{z}$ sei die Dichte $p_t(\cdot \mid \mathbf{z})$ positiv und stetig differenzierbar in $(\mathbf{x}, t)$. Sie erfülle die Kontinuitätsgleichung $\partial_t p_t(\mathbf{x} \mid \mathbf{z}) = -\nabla \cdot \left( p_t(\mathbf{x} \mid \mathbf{z})\, \mathbf{u}_t(\mathbf{x} \mid \mathbf{z}) \right)$ mit einer stetig differenzierbaren Geschwindigkeit. Bei den folgenden Integralen sei die Differentiation unter dem Integralzeichen zulässig. Auf $\{p_t > 0\}$ definieren wir die* **marginale Geschwindigkeit**

<!--en:61b78b70-->
*Dann erfüllt $(p_t, \mathbf{u}_t)$ die Kontinuitätsgleichung.*

<!--en:8201c7b6-->
Die Geschwindigkeit der Wolke bei $\mathbf{x}$ ist der Mittelwert der Geschwindigkeiten aller Teilwolken, aus denen ein Teilchen bei $\mathbf{x}$ stammen könnte. Jede Teilwolke geht mit der Wahrscheinlichkeit ein, dass das Teilchen aus ihr stammt.

<!--en:dd59372e-->
**Beweis.** Wir differenzieren die Mischung unter dem Integralzeichen, setzen die bedingte Kontinuitätsgleichung ein und ziehen die Divergenz vor das Integral:

<!--en:3d01c677-->
Im letzten Schritt erweitern wir mit $p_t(\mathbf{x})$ und erkennen :eqref:`eq_mdl-marginal-velocity`. $\blacksquare$

<!--en:56b07787-->
Wo nur eine Teilwolke vorbeizieht, ist die marginale Geschwindigkeit gleich ihrer Geschwindigkeit. Wo sich mehrere treffen, ist sie deren Mittelwert. Genau diese Mittelung krümmt den Fluss.

<!--en:f1bbddbd-->
### Der Satz zum bedingten Flow-Matching

<!--en:2e0b99fc-->
Das Ziel von :eqref:`eq_mdl-fm-loss` ist der A-posteriori-Mittelwert :eqref:`eq_mdl-marginal-velocity` einer Größe, die wir für jedes Sample berechnen können. Nach dem Regressionslemma (:numref:`sec_mdl-denoising-score-matching`) lernt die Methode der kleinsten Quadrate den bedingten Mittelwert einer Zielgröße, wenn für jedes Sample ein Zielwert vorliegt. Wir können also die Größe für das einzelne Sample als Regressionsziel verwenden. Genau das leistet die Verlustfunktion für **bedingtes Flow-Matching** (engl. conditional flow matching, CFM):

<!--en:63905023-->
Sie benötigt nur ein Sample von $(t, \mathbf{z}, \mathbf{x})$ und die bedingte Geschwindigkeit in geschlossener Form.

<!--en:49e2a7bf-->
**Satz (CFM trainiert das marginale Feld).** *Sei $\mathbf{u}_t$ die marginale Geschwindigkeit aus :eqref:`eq_mdl-marginal-velocity` und gelte $\mathbb{E}\|\mathbf{u}_t(\mathbf{x} \mid \mathbf{z})\|^2 < \infty$. Dann gilt*

<!--en:7d3f7fea-->
*wobei $C$ unabhängig von $\boldsymbol{\theta}$ ist. Insbesondere haben beide Zielfunktionen dieselben Gradienten und dieselben Minimierer* :cite:`Lipman.Chen.BenHamu.ea.2022,Tong.Fatras.Malkin.ea.2023`.

<!--en:9d114d28-->
**Beweis.** Wir wenden das Regressionslemma auf $X = (\mathbf{x}, t)$ mit $\mathbf{x} \sim p_t(\cdot \mid \mathbf{z})$ und auf $Y = \mathbf{u}_t(\mathbf{x} \mid \mathbf{z})$ an. Letztere Größe ist nach Voraussetzung quadratintegrierbar. Für die Verteilung von $X$ gilt $t \sim \mathcal{U}[0,1]$ und $\mathbf{x} \sim p_t$. Der bedingte Mittelwert von $Y$ gegeben $X$ ist nach Definition :eqref:`eq_mdl-marginal-velocity` die marginale Geschwindigkeit. Das Lemma zerlegt $\mathcal{L}_{\mathrm{CFM}}$ in $\mathbb{E}\| \mathbf{u}_{\boldsymbol{\theta}}(X) - \mathbf{u}_t(\mathbf{x}) \|^2 = \mathcal{L}_{\mathrm{FM}}$ und $C = \mathbb{E}\| Y - \mathbf{u}_t(\mathbf{x}) \|^2$. Der zweite Term hängt nicht von $\boldsymbol{\theta}$ ab. $\blacksquare$

<!--en:b944b624-->
Dank des Lemmas genügt eine Beweiszeile. Abgesehen von der Quadratintegrierbarkeit der Zielgröße stellt der Satz keine Anforderungen an den Pfad. Er besagt nur, dass die Regression den richtigen Minimierer hat. Dass dieser Minimierer $p_1$ nach $p_0$ transportiert, wenn wir von $t = 1$ bis $t = 0$ integrieren, garantiert die vorangehende Proposition. Die Konstante $C$ ist die bedingte Varianz der Zielgröße. Sie misst, wie stark die Geschwindigkeiten einzelner Samples am selben Punkt voneinander abweichen. Daher hat auch ein perfektes Feld den CFM-Verlust $C$ statt null. Wir nennen $C$ die *Verlustuntergrenze* (engl. loss floor). Denoising-Score-Matching hat dieselbe Struktur: Die Größe für das einzelne Sample ist der Score des Verrauschungskerns, ihr A-posteriori-Mittelwert der marginale Score.

<!--en:ab65d449-->
Diese Verlustfunktion kommt ohne numerisches Lösungsverfahren aus. Beim Training integrieren wir keine Trajektorie. Das Modell aus :numref:`sec_mdl-neural-odes`, das wir über zehn Euler-Schritte trainiert hatten, traf seine Ziele mit zehn Schritten und verfehlte sie bei feinerer Integration: Es hatte ein Programm aus zehn Schritten gelernt. Mit dieser Regression lernen wir ein Feld. Die Schrittzahl wählen wir erst danach.

<!--en:fdbff87d-->
### Score, Rauschen und Geschwindigkeit

<!--en:e34501f4-->
Die Proposition zur marginalen Geschwindigkeit setzt Teilwolken mit bekannten Geschwindigkeiten voraus. Bisher haben wir kein Beispiel genannt, doch eine erste Familie kennen wir bereits. Die Gauß-Pfade des vorigen Abschnitts sind solche Wahrscheinlichkeitspfade, und ihre Geschwindigkeit ist für uns keine neue Größe. Angenommen, wir haben ein Netz wie bei Diffusionsmodellen darauf trainiert, das Rauschen vorherzusagen. Brauchen wir einen zweiten Trainingslauf, um eine Geschwindigkeit zu erhalten? Nein. Der Score, das Rauschen, der rauschfreie Punkt und die Geschwindigkeit sind affine Funktionen desselben A-posteriori-Mittelwerts. Ein Netz, das das Rauschen gelernt hat, hat damit auch die anderen drei Größen gelernt.

<!--en:bfb0f5b5-->
Wir bedingen auf einen Datenpunkt, $\mathbf{z} = \mathbf{x}_0$, und betrachten den Pfad

<!--en:ab90d30d-->
mit glatten Koeffizienten $(\alpha_t, \sigma_t)$, $(\alpha_0, \sigma_0) = (1, 0)$ und $(\alpha_1, \sigma_1) = (0, 1)$. Die Teilwolke um $\alpha_t \mathbf{x}_0$ hat die Breite $\sigma_t$. Bei $t = 0$ ist sie der Datenpunkt, bei $t = 1$ das Rauschen. Dies ist der Verrauschungskern aus dem vorigen Abschnitt, dessen Rauschende jetzt bei $t = 1$ liegt.

<!--en:0e8b071d-->
Beginnen wir mit dem einfachsten Fall: eindimensionalen gaußverteilten Daten $x_0 \sim \mathcal{N}(0, s_0^2)$ auf dem Kosinuspfad $\alpha_t = \cos\frac{\pi t}{2}$, $\sigma_t = \sin\frac{\pi t}{2}$. Die Randverteilung ist $\mathcal{N}(0, s_t^2)$ mit $s_t^2 = \alpha_t^2 s_0^2 + \sigma_t^2$, der Score ist $-x/s_t^2$ und die Geschwindigkeit beträgt $u_t(x) = (\dot{s}_t/s_t)\, x$. Eine Gauß-Wolke dehnt sich nur aus oder zieht sich zusammen (Aufgabe :exref:`ex_mdl-flow-matching-transport-kinetic-energy`). Mit $s_0 = 2$ erhalten wir bei $t = \tfrac12$ die Varianz $s_t^2 = 2.5$. Ein Teilchen bei $x$ hat die Geschwindigkeit $-0.94\, x$. Das Feld zeigt nach innen, weil die Wolke beim Verrauschen schrumpft, von der Breite $2$ bei $t = 0$ auf die Breite $1$ bei $t = 1$. Ein Sampler läuft in die Gegenrichtung und dehnt die Wolke wieder auf die Breite $2$ aus. Für $s_0 = 1$ ist jedes $p_t$ die Standardnormalverteilung und die Wolke bewegt sich nie: $u \equiv 0$.

<!--en:794b3b72-->
**Proposition (Identität zwischen Score und Geschwindigkeit).** *Auf dem Gauß-Pfad :eqref:`eq_mdl-gaussian-path` ist zu jedem Zeitpunkt $t$ mit $\alpha_t, \sigma_t > 0$ und auf $\{p_t > 0\}$ die marginale Geschwindigkeit eine affine Funktion des marginalen Scores:*

<!--en:0c7fccc5-->
Die Geschwindigkeit ist ein Vielfaches der Position plus ein Vielfaches des Scores. Beide Faktoren sind allein durch den Zeitplan bestimmt. Die Daten gehen nur über den Score ein. Die Klammer vor dem Score ist gleich $\sigma_t^2\, \frac{d}{dt} \log(\sigma_t/\alpha_t)$. Dieser Ausdruck ist positiv, wenn das Verhältnis von Rauschen zu Signal zunimmt, wie auf allen hier verwendeten Pfaden. Dann lässt sich die Identität nach dem Score auflösen, und jedes der beiden Felder bestimmt das andere.

<!--en:280ae437-->
**Beweis.** Wir zeigen, dass beide Seiten affine Funktionen desselben A-posteriori-Mittelwerts $\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]$ sind. Durch Differenzieren des Pfades erhalten wir die bedingte Geschwindigkeit $\dot{\mathbf{x}}_t = \dot{\alpha}_t \mathbf{x}_0 + \dot{\sigma}_t \boldsymbol{\epsilon}$, und es gilt $\boldsymbol{\epsilon} = (\mathbf{x} - \alpha_t \mathbf{x}_0)/\sigma_t$. Bilden wir wie in :eqref:`eq_mdl-marginal-velocity` den A-posteriori-Mittelwert, so folgt

<!--en:060c3062-->
Der marginale Score ist der A-posteriori-Mittelwert der Scores der Kerne. Das ist die Formel von Tweedie aus dem vorigen Abschnitt. Der Kern $\mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$ hat den Score $(\alpha_t \mathbf{x}_0 - \mathbf{x})/\sigma_t^2$, also gilt

<!--en:3e784aba-->
Setzen wir diesen Ausdruck für $\hat{\mathbf{x}}_0$ in die Geschwindigkeit ein und fassen die Terme mit $\mathbf{x}$ und $\nabla \log p_t$ zusammen, erhalten wir :eqref:`eq_mdl-score-velocity`. $\blacksquare$

<!--en:a3a0f932-->
Betrachten wir zwei Spezialfälle. Auf einem varianzerhaltenden Pfad gilt $\alpha_t^2 + \sigma_t^2 = 1$ und damit $\sigma_t \dot{\sigma}_t = -\alpha_t \dot{\alpha}_t$. Der Koeffizient des Scores vereinfacht sich zu $\dot{\alpha}_t/\alpha_t$: $\mathbf{u}_t(\mathbf{x}) = \frac{\dot{\alpha}_t}{\alpha_t}\big(\mathbf{x} + \nabla \log p_t(\mathbf{x})\big)$. Für die varianzerhaltende stochastische Differentialgleichung (engl. stochastic differential equation, SDE) des vorigen Abschnitts gilt $\alpha_t = \exp\big(-\tfrac12 \int_0^t \beta(s)\, ds\big)$, also $\dot{\alpha}_t/\alpha_t = -\tfrac12 \beta(t)$. Die Geschwindigkeit ist damit $-\tfrac12 \beta(t) \big(\mathbf{x} + \nabla \log p_t(\mathbf{x})\big)$. Das ist die Geschwindigkeit des Wahrscheinlichkeitsflusses $\mathbf{f} - \tfrac12 g^2 \nabla \log p_t$ (:numref:`sec_mdl-continuity-equation`) mit $\mathbf{f} = -\tfrac12 \beta\, \mathbf{x}$ und $g^2 = \beta$. Auf diesem Pfad beschreiben die Flow-Matching-Geschwindigkeit und die Wahrscheinlichkeitsfluss-ODE dasselbe Feld mit derselben Zeitrichtung. Auch der Kosinuspfad des Gauß-Beispiels ist varianzerhaltend. Wir können die Formel also an ihm prüfen: $\nabla \log p_t = -x/s_t^2$ ergibt $\frac{\dot\alpha_t}{\alpha_t}(1 - 1/s_t^2)\, x = \alpha_t \dot{\alpha}_t (s_0^2 - 1)\, x / s_t^2$, die oben gefundene Dehnrate $\dot{s}_t/s_t$.

<!--en:e8848e47-->
Auf dem geraden Pfad $\alpha_t = 1 - t$, $\sigma_t = t$, den wir im weiteren Verlauf verwenden, lautet die Identität $\mathbf{u}_t(\mathbf{x}) = -\big(\mathbf{x} + t\, \nabla \log p_t(\mathbf{x})\big)/(1 - t)$.

<!--en:20278f14-->
Derselbe A-posteriori-Mittelwert liegt den Zielgrößen zugrunde, mit denen wir Netze in der Praxis trainieren (:numref:`tab_mdl-score-noise-velocity`).

<!--en:12301ce5-->
:Zielgrößen für Netze auf einem Gauß-Pfad. Jede Zeile zeigt die vorhergesagte Größe, den zugehörigen geschätzten A-posteriori-Mittelwert und die Umrechnung aus dem Score. Der Name $\mathbf{v}$-Vorhersage stammt aus der Literatur. Auf dem Kosinuspfad ist ihre Zielgröße $\alpha_t\boldsymbol\epsilon - \sigma_t\mathbf{x}_0$ das $2/\pi$-Fache der Zielgeschwindigkeit.

<!--en:ebe37e77-->
| Netzvorhersage | A-posteriori-Mittelwert | Aus dem Score $\mathbf{s} = \nabla \log p_t$ |

<!--en:bfe51d11-->
| Score $\mathbf{s}_{\boldsymbol{\theta}}$ | $\nabla \log p_t(\mathbf{x})$ | $\mathbf{s}$ |

<!--en:e69a4837-->
| Rauschen $\hat{\boldsymbol{\epsilon}}$ | $\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t = \mathbf{x}]$ | $-\sigma_t\, \mathbf{s}$ |

<!--en:d7f1b1bc-->
| Rauschfreier Punkt $\hat{\mathbf{x}}_0$ | $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]$ | $(\mathbf{x} + \sigma_t^2\, \mathbf{s})/\alpha_t$ |

<!--en:12bf8309-->
| Geschwindigkeit $\mathbf{u}_{\boldsymbol{\theta}}$ | $\mathbb{E}[\dot{\mathbf{x}}_t \mid \mathbf{x}_t = \mathbf{x}]$ | nach der Identität :eqref:`eq_mdl-score-velocity` |

<!--en:eabd397e-->
| $\mathbf{v}$-Vorhersage $\hat{\mathbf{v}}$ | $\mathbb{E}[\alpha_t \boldsymbol{\epsilon} - \sigma_t \mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]$ | $-\tfrac{\sigma_t}{\alpha_t}\, \mathbf{x} - \sigma_t\, \tfrac{\alpha_t^2 + \sigma_t^2}{\alpha_t}\, \mathbf{s}$ |

<!--en:213e5aa6-->
Jede Zeile beschreibt eine von $t$ abhängige affine Funktion desselben A-posteriori-Mittelwerts. Ein Netz, das wir auf eine beliebige dieser Größen trainieren, bestimmt also alle anderen. Welche Größe sich am leichtesten lernen lässt, ist eine andere Frage. Sie betrifft die Varianz der Trainingsziele an beiden Enden des Pfades (Aufgabe :exref:`ex_mdl-flow-matching-transport-target-conditioning`). Ein Netz zur Rauschvorhersage liefert also auch ein Geschwindigkeitsfeld: Nach der Umrechnung gemäß :numref:`tab_mdl-score-noise-velocity` können wir die Sampler dieses Abschnitts darauf anwenden.

<!--en:5fa852f0-->
### Beispiel mit zwei Punkten in einer Dimension

<!--en:ad7ffbae-->
Bevor wir zwischen Pfaden wählen, rechnen wir einen Fall vollständig durch. Für den Datensatz mit zwei Punkten aus der Einleitung können wir alle bisherigen Größen in geschlossener Form angeben. Daran erkennen wir das Problem, das wir in :numref:`sec_mdl-rectified-flow` lösen. Die Daten nehmen die Werte $x_0 = \pm a$ mit Wahrscheinlichkeit $\tfrac12$ je Wert an, wobei $a = 2$ gilt. Das sind die beiden Modi der Mischung, die uns durch das Kapitel begleitet, hier ohne ihre Breite. Das Rauschen $\epsilon \sim \mathcal{N}(0, 1)$ ziehen wir unabhängig von $x_0$. Wir nehmen den Gauß-Pfad mit $\alpha_t = 1 - t$ und $\sigma_t = t$. Er verbindet jeden Datenpunkt geradlinig und mit konstanter Geschwindigkeit mit seinem Rauschpunkt: $x_t = (1 - t)\, x_0 + t\, \epsilon$ mit Geschwindigkeit $\dot{x}_t = \epsilon - x_0$.

<!--en:a7bf46bd-->
Wir bedingen auf den Datenpunkt, $\mathbf{z} = x_0$. Gegeben $x_0$ hat die Position $x_t$ die Verteilung $\mathcal{N}((1 - t)\, x_0, t^2)$. Ihre Dichte ist für $t > 0$ positiv, also gilt die Proposition zur marginalen Geschwindigkeit. Eliminieren wir $\epsilon = (x - (1 - t)\, x_0)/t$, erhalten wir die Geschwindigkeit als Funktion der aktuellen Position, $u_t(x \mid x_0) = (x - x_0)/t$. Rückwärts in der Zeit bewegt sich das Teilchen so schnell auf seinen Datenpunkt zu, dass es genau bei $t = 0$ eintrifft.

<!--en:45fa611d-->
Um diese Geschwindigkeit zu mitteln, brauchen wir die A-posteriori-Verteilung von $x_0$ an einem Punkt $x$. Das A-posteriori-Chancenverhältnis von $+a$ gegenüber $-a$ bei $x$ ist der Quotient der beiden Gauß-Dichten. Im Exponenten heben sich die Terme mit $x^2$ und $(1 - t)^2 a^2$ auf. Nur der gemischte Term bleibt. Das Chancenverhältnis beträgt somit $\exp\big(2 (1 - t)\, a x / t^2\big)$ und der A-posteriori-Mittelwert ist $\mathbb{E}[x_0 \mid x_t = x] = a \tanh\big((1 - t)\, a x / t^2\big)$. Die marginale Geschwindigkeit ist daher

<!--en:5107564b-->
Rückwärts in der Zeit bewegt sich die Wolke bei $x$ auf den Mittelwert der beiden Ziele zu, gewichtet danach, wie plausibel jedes Ziel von $x$ aus ist. Ihre Geschwindigkeit reicht genau aus, um rechtzeitig anzukommen. Bei $(t, x) = (0.5, 0.5)$ ist die Geschwindigkeit $-2.86$, bei $(0.7, -1)$ beträgt sie $0.97$. Aus :eqref:`eq_mdl-two-point-velocity` lassen sich drei Eigenschaften ablesen. Die ersten beiden zeigt :numref:`fig_mdl-dyn-two-point-flow`.

<!--en:185924fb-->
* **Bei $t = 1$ ist das Feld gleich $u_1(x) = x$.** Das Rauschsample enthält keine Information über sein Ziel. Der A-posteriori-Mittelwert ist also der Datenmittelwert $0$. Ein Euler-Schritt von $t = 1$ bis $t = 0$, also $x \mapsto x - u_1(x)$, schickt *jedes* Sample nach $0$. Das gilt bei unabhängiger Paarung für beliebige Daten: $\mathbf{u}_1(\mathbf{x}) = \mathbb{E}[\boldsymbol{\epsilon} - \mathbf{x}_0 \mid \boldsymbol{\epsilon} = \mathbf{x}] = \mathbf{x} - \mathbb{E}[\mathbf{x}_0]$. Ein Schritt mit dem exakten Feld landet also auf dem Datenmittelwert.

<!--en:c34cdaf6-->
* **Der Ursprung zieht Teilchen zunächst an und stößt sie später ab.** Wenn $t$ abnimmt, bewegt sich ein Sampler entgegen $u_t$. Eine positive Steigung am Ursprung zieht Teilchen zu ihm hin, eine negative stößt sie ab. Die Steigung beträgt $\partial_x u_t(0) = \big(1 - (1 - t)\, a^2/t^2\big)/t$: Sie ist $+0.56$ bei $t = 0.9$, null bei $t^* = 2\sqrt{2} - 2 \approx 0.83$ (der Nullstelle von $(1 - t)\, a^2 = t^2$), $-0.31$ bei $t = 0.8$ und $-14$ bei $t = 0.5$. Teilchen in der Nähe von $0$ driften in Richtung $0$, bis $t$ auf $t^*$ gesunken ist. Erst später bewegen sie sich auf einen Modus zu. Der Fluss entscheidet sich also spät für einen Modus. Fern vom Ursprung ist der $\tanh$ gesättigt, und es gilt $u_t \approx (x \mp a)/t$, die bedingte Geschwindigkeit selbst.

<!--en:a517602b-->
* **Die Verlustuntergrenze entsteht vor allem nahe dem Rauschende.** Das mittlere Quadrat der Zielgröße $\epsilon - x_0$ beträgt $a^2 + 1 = 5$. Davon entfallen $C = \mathbb{E}_t \mathbb{E}_x \textrm{Var}(x_0 \mid x_t)/t^2 \approx 2.0$ auf die Verlustuntergrenze, wobei $\textrm{Var}(x_0 \mid x_t = x) = a^2 \operatorname{sech}^2\big((1 - t)\, a x/t^2\big)$ gilt. Die verbleibenden $3.0$ sind das mittlere Quadrat der marginalen Geschwindigkeit. Fast der gesamte Wert von $C$ stammt aus dem Bereich $t > 0.5$. Dort überlappen sich die beiden Teilwolken, und die Position verrät nicht, welchen Punkt das Teilchen ansteuert. Unterhalb von $t = 0.4$ verrät sie es. Nur etwa ein Tausendstel der Verlustuntergrenze stammt aus diesem Bereich, obwohl die bedingte Geschwindigkeit $(x - x_0)/t$ dort unbeschränkt wächst (Aufgabe :exref:`ex_mdl-flow-matching-transport-late-decisions`).

<!--en:36cab35f-->
![Zwei Datenpunkte bei $\pm 2$ und standardnormalverteiltes Rauschen bei unabhängiger Paarung. Das Sampling läuft vom Rauschen bei $t=1$ (rechts) zu den Daten bei $t=0$ (links). (a) Trajektorien des exakten marginalen Flusses vor den Bändern der beiden bedingten Teilwolken, die jeweils eine Standardabweichung um den Mittelwert zeigen. Bis $t$ auf $t^*=0.83$ fällt, bewegt sich jede Trajektorie auf $0$ zu. Trajektorien, die nahe $0$ beginnen, wenden sich erst unterhalb von $t\approx 0.6$ einem Modus zu. (b) Die Geschwindigkeit $u_t(x)$ ist gleich $x$ bei $t=1$. Ihre Steigung am Ursprung wird negativ und steil, wenn $t$ fällt. Fern vom Ursprung nähert sich die Geschwindigkeit $(x\mp 2)/t$.](../img/mdl-dyn-two-point-flow.svg)

<!--en:5ca1050f-->
Die Codezelle berechnet die Steigung am Ursprung für die drei Zeitpunkte aus der zweiten Eigenschaft mit einer endlichen Differenz. Anschließend prüft sie die erste Eigenschaft und untersucht, was zusätzliche Schritte bewirken. Ausgehend von der gesamten Rauschverteilung integriert sie das exakte Feld von $t = 1$ bis $t = 0$ mit $1$, $2$, $4$ und $8$ gleich großen Euler-Schritten. Sie gibt an, welcher Massenanteil höchstens $0.1$ von einem Datenpunkt entfernt landet und welchen Mittelwert $|x|$ am Ende hat. Ein feines Gitter von Startpunkten, gewichtet mit der Gauß-Dichte, stellt die Rauschverteilung dar. Die Ausgabe enthält daher keinen Zufall.

<!--en:58c770fb-->
Die Steigungen stimmen mit der Formel überein. Nach einem Euler-Schritt liegt keinerlei Masse nahe einem Datenpunkt, und der Mittelwert von $|x|$ beträgt $0$: Jedes Sample liegt auf dem Datenmittelwert. Damit scheitert unser Wunsch aus der Einleitung schon beim ersten Versuch. Jeder bedingte Pfad ist gerade, doch der marginale Fluss ist gekrümmt. Der erhoffte große Schritt lässt alle Samples im Mittelwert zusammenfallen.

<!--en:dce1163b-->
Weitere Schritte beheben das Problem schnell. Zwei bringen etwas mehr als ein Drittel der Masse zu einem Datenpunkt, vier die gesamte Masse bis auf etwa ein Prozent und acht praktisch alles. So wenige Schritte genügen, weil der letzte Schritt auf jedem Gitter auf dem A-posteriori-Mittelwert $a \tanh(\cdot)$ landet. Dieser ist gleich $\pm a$, wo immer die vorangehenden Schritte ein Teilchen weit genug vom Ursprung entfernt haben (Aufgabe :exref:`ex_mdl-flow-matching-transport-late-decisions`). Wir wollten aber gerade Schritte einsparen. Wie wir Rauschen und Daten so paaren, dass der Fluss gerade wird und ein Schritt genügt, untersuchen wir in :numref:`sec_mdl-rectified-flow`.

<!--en:ff5472b8-->
## Rectified Flow

<!--en:f7239a13-->
### Gerade Pfade und ihre Kreuzungen

<!--en:952d2453-->
Das Beispiel mit zwei Punkten verwendete gerade bedingte Pfade. In $d$ Dimensionen paaren wir nach demselben Prinzip ein Datensample $\mathbf{x}_0$ mit einem unabhängig gezogenen Rauschsample $\boldsymbol{\epsilon}$. Die Punkte verbinden wir durch eine Gerade, die wir mit konstanter Geschwindigkeit durchlaufen:

<!--en:da05c4ed-->
Die bedingte Geschwindigkeit hängt nicht von $t$ ab. Der CFM-Verlust :eqref:`eq_mdl-cfm-loss` wird damit zur Zielfunktion für **Rectified Flow** :cite:`Liu.Gong.Liu.2022,Lipman.Chen.BenHamu.ea.2022`

<!--en:b2c0e0bd-->
Beim Training ziehen wir Daten und Rauschen, interpolieren dazwischen und verwenden die Differenz als Regressionsziel. Das ist der ganze Algorithmus. Sein Name greift dem Ergebnis vor: *rectified* bedeutet „begradigt“. Gerade wird der Fluss, sobald wir die Paarung korrigiert haben. Bei unabhängigen Paaren ist er, wie wir gerade gesehen haben, gekrümmt.

<!--en:48140606-->
Ein technisches Detail bleibt zu klären. Bei gaußschem Rauschen könnten wir wie im Beispiel mit zwei Punkten allein auf $\mathbf{x}_0$ bedingen. Dann wäre die Proposition zur marginalen Geschwindigkeit anwendbar. Wenn wir auf das Paar bedingen, können wir später die Paarung ändern. Bei gegebenem Paar hat ein Punkt, der sich entlang einer Geraden bewegt, aber keine Dichte. Daher gilt die Proposition in ihrer bisherigen Form nicht. Ihre Aussage bleibt dennoch richtig. Dazu wenden wir dieselbe Rechnung auf den Erwartungswert $\mathbb{E}\,\varphi(\mathbf{x}_t)$ einer glatten Testfunktion $\varphi$ an (Aufgabe :exref:`ex_mdl-flow-matching-transport-reflow-fixed-points`).

<!--en:1d8565b0-->
Warum krümmt sich der Fluss? Zwei Strecken, die sich bei $(\mathbf{x}, t)$ kreuzen, tragen zwei verschiedene Richtungen zum A-posteriori-Mittelwert :eqref:`eq_mdl-marginal-velocity` bei (:numref:`fig_mdl-dyn-fm-paths`). Eine Funktion hat an jedem Punkt nur einen Wert, also mittelt das Feld die Richtungen. Bei einer ODE mit einem in $\mathbf{x}$ Lipschitz-stetigen Feld verläuft durch jeden Punkt genau eine Trajektorie (:numref:`sec_mdl-ode-existence-uniqueness`). Zwei Trajektorien des gelernten Flusses können sich daher nie treffen. Wo sich die Strecken kreuzen, biegen die Trajektorien voneinander weg.

<!--en:5b6d87aa-->
![Drei Daten-Rausch-Paare in einer Dimension. Gerade Strecken (gestrichelt) verbinden die Endpunkte. An zwei Stellen kreuzen sich Strecken (×). Dort mittelt die marginale Geschwindigkeit ihre Richtungen. Daher krümmen sich die Trajektorien des zugehörigen Flusses (durchgezogen) auf dem Weg vom Rauschen bei $t = 1$ zu den Daten bei $t = 0$ und treffen sich nie. Jede Trajektorie endet auf einem anderen Datenpunkt als ihre eigene Strecke. Die Trajektorien kommen in derselben Reihenfolge an, in der sie gestartet sind.](../img/mdl-dyn-fm-paths.svg)

<!--en:2ec1e4c0-->
Unser Ziel ist ein Feld, dessen Trajektorien geradlinig mit konstanter Geschwindigkeit verlaufen. Ein solches Feld ist entlang jeder Trajektorie konstant. Damit ist ein Euler-Schritt beliebiger Länge exakt. Die unabhängige Paarung ist weit von diesem Grenzfall entfernt. Das richtige Feld minimiert den Verlust, und die Teilstücke sind bereits gerade. Wir können also nur noch ändern, welchen Rauschpunkt wir mit welchem Datenpunkt verbinden.

<!--en:2cca61cc-->
### Beispiel mit zwei Halbmonden

<!--en:d38f3796-->
Für die beiden Punkte hatten wir eine Formel, die zeigte, dass ein Schritt nicht ausreicht. Wie viele Schritte braucht ein trainiertes Netz bei Daten, für die es keine solche Formel gibt? Wir nehmen zwei ineinandergreifende Halbmonde in der Ebene, trainieren ein kleines Netz darauf und zählen die Schritte. Dazu brauchen wir ein Maß für die Güte der Samples. Wir verwenden die **quadrierte Energiedistanz** (engl. squared energy distance) :cite:`Szekely.Rizzo.2013` zwischen Verteilungen $P$ und $Q$ mit endlichen ersten Momenten,

<!--en:dbff5550-->
wobei $X, X' \sim P$ und $Y, Y' \sim Q$ unabhängig sind. Sie vergleicht den mittleren Abstand zwischen einem Punkt aus $P$ und einem Punkt aus $Q$ mit den mittleren Abständen innerhalb der jeweiligen Verteilung. Székely und Rizzo nennen diesen Ausdruck selbst Energiedistanz. Wir nennen ihn quadriert, weil seine Quadratwurzel die Metrik ist. Er ist eine quadrierte MMD (:numref:`sec_mdl-ipm-mmd`) und nur dann null, wenn $P = Q$ gilt. Wir schätzen ihn aus zwei Stichproben mit der erwartungstreuen U-Statistik. Alle folgenden Bewertungen geben diese quadrierte Distanz an.

<!--en:c925cdd1-->
Die Codezelle erzeugt die Daten und standardisiert jede Koordinate auf Mittelwert null und Varianz eins. Außerdem erzeugt sie eine zurückgehaltene Stichprobe von $2{,}048$ Punkten und ein Rauscharray `z_eval`, das für alle späteren Vergleiche gleich bleibt. Zwei Referenzen bewertet sie anhand der zurückgehaltenen Stichprobe: eine neue Stichprobe aus den Daten und eine Punktmasse auf dem Datenmittelwert, wie sie ein Euler-Schritt mit dem exakten Feld liefert.

<!--en:b3a1d975-->
Bei zwei Stichproben aus derselben Verteilung weicht die geschätzte Distanz nur um wenige Tausendstel von null ab. Das ist die Auflösung der Metrik bei diesem Stichprobenumfang. Die Punktmasse erreicht einen Wert von etwa $0.9$.

<!--en:31aa1892-->
Das Modell ist ein kleines tanh-Netz, das $(\mathbf{x}, t)$ als Eingabe erhält und eine Geschwindigkeit zurückgibt. `field` wertet es aus. Die Funktion `train_rf` erhält einen Sampler für Paare. So können wir die Paarung später ändern. Sie gibt das Netz und seinen Verlust zurück. Da die Werte einzelner Batches streuen, mitteln wir über die letzten $500$ Batches. Wir mitteln auch über beide Koordinaten, sodass die Ausgabe $\mathcal{L}_{\mathrm{RF}}/2$ entspricht.

<!--en:4f1cdea5-->
Im ersten Lauf paaren wir Rauschen und Daten unabhängig. Nach dem CFM-Satz kann der Verlust nicht auf null sinken. Sein Grenzwert misst, wie stark sich die Strecken kreuzen. Die Verlustuntergrenze $C$ ist die bedingte Varianz von $\boldsymbol{\epsilon} - \mathbf{x}_0$ gegeben $\mathbf{x}_t$, gemittelt über $t$. Können wir sie bereits vor dem Training bestimmen? Für einen endlichen Datensatz geht das ohne Netz. Gegeben $\mathbf{x}_t$ weist die A-posteriori-Verteilung von $\mathbf{x}_0$ jedem Trainingspunkt $\mathbf{x}_0'$ ein Gewicht proportional zu $\exp\big(-\|\mathbf{x}_t - (1 - t)\, \mathbf{x}_0'\|^2 / 2t^2\big)$ zu. Die Zielgröße ist $\boldsymbol{\epsilon} - \mathbf{x}_0 = (\mathbf{x}_t - \mathbf{x}_0)/t$. Ihre bedingte Varianz ist also die A-posteriori-Varianz von $\mathbf{x}_0$ geteilt durch $t^2$. Die Funktion `pairing_floor` mittelt diese Größe über $4{,}000$ Ziehungen von $(t, \mathbf{x}_0, \boldsymbol{\epsilon})$ und halbiert sie, weil `train_rf` den Verlust pro Koordinate ausgibt.

<!--en:710c0a25-->
Die Verlustuntergrenze liegt bei etwa $1.37$ pro Koordinate. Der Schätzwert schwankt von Ziehung zu Ziehung in der zweiten Nachkommastelle. Ein perfektes Netz würde diese Grenze erreichen. Unser Netz hat zwei versteckte Schichten mit je $256$ Einheiten und kann sich der Grenze in $10{,}000$ Schritten mit je $2{,}048$ Paaren nähern.

<!--en:ec4da3ff-->
Der Verlust pendelt sich bei etwa $1.4$ pro Koordinate ein, nur wenige Hundertstel über der Verlustuntergrenze. Ein Verlust von $1.4$ bedeutet also keine schlechte Anpassung. Fast der gesamte Verlust entsteht dadurch, dass sich kreuzende Strecken verschiedene Geschwindigkeiten vorgeben. Das kann kein Netz ändern. Nach dem CFM-Satz ist nur der Anteil oberhalb der Untergrenze ein Fehler des Feldes.

<!--en:e7dbdad2-->
Diese wenigen Hundertstel sind allerdings größer, als sie scheinen. Das mittlere Quadrat der Zielgröße beträgt $2$ pro Koordinate. Jeweils eins stammt vom Rauschen und von den standardisierten Daten, und $1.37$ entfallen auf die Verlustuntergrenze. Das mittlere Quadrat des Feldes selbst beträgt also nur etwa $0.6$. Sein Fehler macht einige Prozent davon aus. Diesem Fehler begegnen wir bei den Samples wieder.

<!--en:3a8c6386-->
Beim Sampling integrieren wir $\dot{\mathbf{x}} = \mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ von $t = 1$ bis $0$ mit dem Euler-Verfahren (:numref:`sec_mdl-euler-runge-kutta`). Wir machen $K$ Schritte der Länge $1/K$, wobei wir jeweils $\mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}, t)/K$ bei $t = 1, 1 - 1/K, \ldots, 1/K$ subtrahieren. Die Funktion `euler_path` gibt die gesamte Trajektorie der Form `(steps + 1, n, 2)` zurück. Ihre letzte Zeile enthält das Sample.

<!--en:424e78e0-->
Ein Schritt mit dem exakten Feld ergäbe die Punktmasse mit dem Wert $0.9$. Für das trainierte Feld integriert die Codezelle das Rauscharray `z_eval` mit $K = 1, 2, 4, \ldots, 64$ Schritten und trägt die quadrierte Energiedistanz zu den zurückgehaltenen Daten auf doppelt logarithmischen Achsen auf. Der Schätzer ist erwartungstreu. An der Auflösungsgrenze der Metrik kann er daher negativ werden. Die Darstellung begrenzt ihn nach unten auf $10^{-3}$.

<!--en:619894fe-->
Ein Schritt erreicht Werte von $0.7$ bis $0.75$, in derselben Größenordnung wie die Punktmasse auf dem Mittelwert ($0.9$). Das Netz hat den Kollaps, den das exakte Feld bei $t = 1$ bewirkt, näherungsweise gelernt. Zwei Schritte ergeben $0.16$ bis $0.19$, vier $0.03$ bis $0.04$, acht etwa $0.01$ oder weniger. Ab sechzehn Schritten bleibt der Wert bis auf wenige Tausendstel bei null. Der Rückgang entspricht dem abnehmenden Diskretisierungsfehler des Euler-Verfahrens. Übrig bleibt der Fehler des trainierten Feldes. Er liegt an der Auflösungsgrenze der Metrik für $2{,}048$ Punkte. Mehr Schritte helfen, wie bei einem ohne numerisches Lösungsverfahren trainierten Feld zu erwarten ist. Das in :numref:`sec_mdl-neural-odes` über zehn Euler-Schritte trainierte Feld wurde bei feineren Schritten schlechter.

<!--en:8ca61300-->
Ein Zahlenwert fasst viele Einzelheiten zusammen. Sehen wir uns an, welche. Die nächste Codezelle zeichnet die Samples nach $1$, $2$, $8$ und $32$ Schritten und daneben die zurückgehaltenen Daten.

<!--en:4ad3186c-->
Ein Schritt ergibt eine Wolke um den Mittelwert, zwei ein schräges Band. Nach acht Schritten bildet die Wolke zwei Halbmonde, zwischen denen noch viele Punkte verstreut liegen. Nach zweiunddreißig Schritten sind die meisten dieser Punkte verschwunden. Wie bei den Daten im letzten Teilbild zeichnen sich zwei getrennte Halbmonde ab. Im direkten Vergleich mit den Daten sind sie allerdings etwas dicker, und in den Lücken bleiben einige Punkte übrig. Auch mehr Schritte beseitigen sie nicht. Diese Abweichungen stammen also vom Feld und nicht vom numerischen Verfahren. So sehen die wenigen Hundertstel zwischen dem Verlust und seiner Untergrenze aus.

<!--en:17b2f67b-->
Der Zahlenwert erfasst diese Abweichungen kaum. Er vergleicht mittlere Abstände zwischen Punkten. Daher zeigt er schon gut an, wo die Masse liegt, lange bevor er die Schärfe der Ränder erfasst. Ein Wert an der Auflösungsgrenze der Metrik garantiert also noch keine so scharfen Konturen wie bei den Daten. Mehr Schritte beseitigen den Fehler des numerischen Verfahrens. Gemessen an der Energiedistanz braucht das sechzehn bis zweiunddreißig Netzauswertungen pro Sample. So viel kostet die Krümmung, die durch die unabhängige Paarung im Fluss entsteht.

<!--en:e53c3e6a-->
### Reflow

<!--en:a61f04d2-->
Die unabhängige Paarung krümmt den Fluss. Das trainierte Modell liefert eine bessere Paarung. Wir lösen seine ODE ausgehend von neuem Rauschen und paaren jeden Rauschpunkt $\boldsymbol{\epsilon}$ mit dem Endpunkt $\hat{\mathbf{x}}_0(\boldsymbol{\epsilon})$, den er erreicht. Trajektorien desselben Flusses kreuzen sich nie. Daher kreuzen sich die Strecken dieser Paare viel seltener als die unabhängiger Paare. Trainieren wir dieselbe Architektur mit den neuen Paaren erneut, sprechen wir von **Reflow** :cite:`Liu.Gong.Liu.2022`.

<!--en:69c5947e-->
Hat die neue Paarung Nachteile? Sie könnte die Endpunktverteilungen verfälschen oder Punkte paaren, die weiter auseinanderliegen als zuvor. Beides geschieht nicht. Wie weit eine Paarung ihre Punkte bewegt, messen wir mit ihren *Transportkosten* $\mathbb{E}\, c(\mathbf{x}_1 - \mathbf{x}_0)$. Dazu mitteln wir eine konvexe Funktion $c$ der Verschiebung, etwa deren quadrierte Länge, über die Paare. In der folgenden Aussage bezeichnen Großbuchstaben zufällige Pfade: $\mathbf{X}_t$ ist die gerade Strecke eines Paares und $\mathbf{Z}_t$ die Trajektorie des Flusses, die am selben Rauschpunkt beginnt.

<!--en:22dfd93f-->
**Proposition (Reflow erhält die Randverteilungen und senkt die Transportkosten).** *Das Paar $(\mathbf{X}_0, \mathbf{X}_1)$ aus einem Datenpunkt und einem Rauschpunkt habe endliche zweite Momente. Sei $\mathbf{X}_t = (1 - t) \mathbf{X}_0 + t \mathbf{X}_1$ und sei $\mathbf{u}_t$ die marginale Geschwindigkeit dieses geraden Pfades. Ferner löse $\mathbf{Z}_t$ die Gleichung $\dot{\mathbf{Z}}_t = \mathbf{u}_t(\mathbf{Z}_t)$ rückwärts in der Zeit vom Rauschende aus, $\mathbf{Z}_1 = \mathbf{X}_1$. Dabei habe $\mathbf{X}_t$ eine positive Dichte für jedes $t > 0$. Das Feld $\mathbf{u}_t$ sei stetig differenzierbar und Lipschitz-stetig in $\mathbf{x}$ mit einer von $t$ unabhängigen Konstanten, sodass dieser Fluss existiert. Dann gilt (i) $\mathbf{Z}_t$ hat dieselbe Verteilung wie $\mathbf{X}_t$ für jedes $t$, insbesondere $\mathbf{Z}_0 \sim p_0$. (ii) Für jede konvexe Funktion $c$ gilt*

<!--en:f59a3811-->
**Beweis.** (i) Die Verteilung von $\mathbf{X}_t$ erfüllt die Kontinuitätsgleichung mit $\mathbf{u}_t$. Dies folgt aus der Proposition zur marginalen Geschwindigkeit mit $\mathbf{z} = (\mathbf{X}_0, \mathbf{X}_1)$ in der schwachen Form, die auch Punktmassen erfasst (Aufgabe :exref:`ex_mdl-flow-matching-transport-reflow-fixed-points`). Die Verteilung von $\mathbf{Z}_t$ erfüllt dieselbe Gleichung, weil sich $\mathbf{Z}_t$ mit dem Feld $\mathbf{u}_t$ bewegt. Beide Verteilungen sind gleich $p_1$ bei $t = 1$. Zwei Lösungen der Kontinuitätsgleichung mit einem solchen Feld, die zu einem Zeitpunkt übereinstimmen, stimmen zu jedem Zeitpunkt überein. Diese Eindeutigkeit ist der einzige Schritt, den wir nicht vollständig beweisen. In :numref:`sec_mdl-probability-flow-ode` haben wir die Begründung für glatte positive Dichten gegeben: Entlang jeder Trajektorie ändern sich beide Log-Dichten mit der Rate $-\nabla \cdot \mathbf{u}_t$. Hier verwenden wir das Ergebnis ohne erneuten Beweis für die schwache Form. (ii) Wir schreiben die Verschiebung als Zeitintegral und wenden zweimal die Jensen-Ungleichung an:

<!--en:50702307-->
Die erste Ungleichung ist die Jensen-Ungleichung für den zeitlichen Mittelwert. Für die Gleichheit ersetzen wir mithilfe von (i) $\mathbf{Z}_t$ durch $\mathbf{X}_t$ und verwenden dann die Definition der marginalen Geschwindigkeit. Die letzte Ungleichung ist die bedingte Jensen-Ungleichung. $\blacksquare$

<!--en:9f16d6fb-->
Für $c = \|\cdot\|^2$ geht im letzten Schritt genau die bedingte Varianz verloren. Die Kosten sinken also mindestens um die Verlustuntergrenze der alten Paarung: $\mathbb{E}\|\mathbf{Z}_1 - \mathbf{Z}_0\|^2 \le \mathbb{E}\|\mathbf{X}_1 - \mathbf{X}_0\|^2 - C$. Die Verlustuntergrenze $C$, beim Training noch ein Hindernis, ist die Einsparung, die eine Runde Reflow garantiert. Wo sich keine Strecken kreuzen, ist die Untergrenze null und eine neue Paarung bringt keinen Gewinn.

<!--en:716e7bdf-->
Die Voraussetzungen sind stärker, als es unsere Beispiele zulassen. Für die beiden Punkte und für jeden endlichen Trainingsdatensatz, den wir unabhängig mit gaußschem Rauschen paaren, ist das marginale Feld auf $[\delta, 1]$ für jedes $\delta > 0$ Lipschitz-stetig, aber nicht bis hin zu $t = 0$. Die Steigung $-14$ des Zweipunktfeldes bei $t = 0.5$ wächst für $t \to 0$ unbeschränkt. Der Beweis erfasst den Fluss dann nur bis zum Zeitpunkt $\delta$. Er liefert (i) für $t \ge \delta$ und (ii) in der Form $\mathbb{E}\, c\big((\mathbf{Z}_1 - \mathbf{Z}_\delta)/(1 - \delta)\big) \le \mathbb{E}\, c(\mathbf{X}_1 - \mathbf{X}_0)$. Den Grenzübergang $\delta \to 0$ beweisen wir nicht. In diesem Sinne gehen die folgenden Beispiele über die Proposition hinaus.

<!--en:6c3d8e88-->
Für die beiden Punkte lässt sich die Aussage direkt prüfen. Ein eindimensionaler Fluss erhält die Reihenfolge. Eine Runde Reflow paart also jeden Rauschpunkt $\epsilon$ mit $a \operatorname{sign}(\epsilon)$ (Aufgabe :exref:`ex_mdl-flow-matching-transport-reflow-two-points`). Diese Paarung kostet $a^2 - 2a\sqrt{2/\pi} + 1 = 1.81$ gegenüber $a^2 + 1 = 5$ für die unabhängige Paarung. Ihre Strecken kreuzen sich nie, also gilt in der zweiten Runde $C = 0$, und ein Euler-Schritt integriert ihr Feld exakt.

<!--en:4ab5f51f-->
Für die beiden Halbmonde zieht die Codezelle $8{,}192$ neue Rauschpunkte und integriert das trainierte Modell über $32$ Schritte, um ihre Endpunkte zu bestimmen. Mit `train_rf` trainiert sie ein neues Netz auf diesen festen Paaren.

<!--en:d387454c-->
Der Reflow-Verlust pendelt sich unter einem Tausendstel ein, während der erste Verlust bei $1.4$ stehen blieb. Die neuen Strecken kreuzen sich also fast nie. Ein Feld, das aus kreuzungsfreien Strecken entsteht, ist entlang seiner Trajektorien fast konstant. Dann sollten wenige Euler-Schritte genügen. Die nächste Codezelle zieht aus beiden Modellen Samples mit $1$, $2$ und $32$ Schritten, jeweils ausgehend von demselben Rauscharray.

<!--en:c1449198-->
Sie gibt außerdem die mittlere quadrierte Länge der alten und der neuen Paare aus, also die Kosten, für die die Proposition eine Schranke liefert. Die alten Paare kosten $4$: Rauschen und standardisierte Daten haben in beiden Koordinaten jeweils Varianz eins, und bei unabhängigen Ziehungen addieren sich die Varianzen. Über beide Koordinaten summiert beträgt die Verlustuntergrenze der unabhängigen Paarung das Doppelte des oben ausgegebenen Werts, etwa $2.75$. Für den exakten marginalen Fluss garantiert die Proposition daher neue Paare mit Kosten von höchstens etwa $1.25$.

<!--en:c7d30a77-->
Ein Euler-Schritt des mit Reflow trainierten Modells erreicht denselben Wert wie zweiunddreißig Schritte des ursprünglichen Modells: höchstens ein bis zwei Tausendstel, die Auflösungsgrenze der Metrik. Ein Schritt des ursprünglichen Modells ergab etwa $0.7$. Ein besseres Ergebnis war nicht zu erwarten. Das Reflow-Netz wurde darauf trainiert, die Endpunkte des ursprünglichen Modells nach $32$ Schritten zu reproduzieren. Mit dessen Samples übernimmt es auch dessen Fehler.

<!--en:5d8cf2cf-->
Die mittlere quadrierte Länge der Paare sinkt von $4$ auf etwa $0.3$, weit unter den garantierten Höchstwert $1.25$. Der Spielraum entsteht im ersten Beweisschritt: Für die gekrümmten Trajektorien des ersten Flusses liegt die Jensen-Ungleichung für das Zeitmittel weit von der Gleichheit entfernt.

<!--en:af54696b-->
Auch die Geradlinigkeit lässt sich messen, und zwar durch den mittleren Abstand jeder Trajektorie aus $32$ Schritten von der Sehne zwischen ihren Endpunkten.

<!--en:56fef9d5-->
Die ursprünglichen Trajektorien weichen um etwa ein Viertel einer Einheit von ihren Sehnen ab. Sie laufen zunächst auf den Mittelwert zu, wo sie nach einem einzigen Euler-Schritt landeten, und biegen erst später nach außen zu den Halbmonden ab. Nach Reflow beträgt ihr Abstand zur Sehne höchstens ein halbes Hundertstel, also etwa fünfzigmal näher. Deshalb genügt ein Schritt. Reflow begradigt einen Fluss, macht ihn aber nicht optimal. In mehr als einer Dimension kann ein Fluss gerade und kreuzungsfrei sein und trotzdem Masse weiter bewegen als nötig.

<!--en:6380f076-->
## Optimaler Transport und Geradlinigkeit

<!--en:c7e4097a-->
### Kopplungen und Wasserstein-Abstand

<!--en:edf38578-->
Reflow hat die Transportkosten gesenkt. Wie weit können sie sinken, und ist der Fluss der günstigsten Paarung gerade? Erinnern wir uns an die Begriffe aus :numref:`sec_mdl-optimal-transport`. Eine **Kopplung** (engl. coupling) $\pi$ von $p_0$ und $p_1$ ist eine gemeinsame Verteilung von Paaren mit diesen Randverteilungen. Sie ist ein Plan, der angibt, wie viel Masse von jedem Ausgangspunkt zu jedem Zielpunkt gelangt. Der **2-Wasserstein-Abstand** ergibt sich aus den Kosten des günstigsten Plans für den quadrierten Abstand,

<!--en:cd8383e7-->
wobei $\Pi(p_0, p_1)$ die Menge der Kopplungen ist. Damit ist $W_2^2$ die kleinste mittlere quadrierte Verschiebung, die eine Verteilung in die andere überführt.

<!--en:8dd3b85e-->
Wie sieht der günstigste Plan aus? Hat eine der beiden Verteilungen eine Dichte und haben beide endliche zweite Momente, so ist er eindeutig und teilt keine Masse auf: Er ist eine Abbildung, die von der Verteilung mit Dichte ausgeht. Bei uns ist das die Rauschverteilung $p_1$. Die Abbildung vom Rauschen zu den Daten ist der Gradient einer konvexen Funktion, die *Brenier-Abbildung*. Das ist der Satz von Brenier :cite:`Brenier.1991`, den wir ohne Beweis anführen. Die Daten brauchen keine Dichte: Die beiden Punkte haben keine, ebenso wenig ein endlicher Trainingsdatensatz wie die $8{,}192$ Punkte unserer Halbmonde. Auf einer Geraden lässt sich der Satz leicht veranschaulichen. Die Ableitung einer konvexen Funktion einer Variablen ist monoton wachsend. Die optimale Abbildung erhält also die Reihenfolge der Punkte.

<!--en:0b33fa13-->
Der einfachste Fall ist eine eindimensionale Gauß-Verteilung mit Daten $\mathcal{N}(m, s^2)$, $s > 0$ und Rauschen $\mathcal{N}(0, 1)$. Die sortierte Paarung bildet jedes Quantil auf das entsprechende Quantil ab, $x_0 = m + s\epsilon$, und kostet $\mathbb{E}(m + s\epsilon - \epsilon)^2 = m^2 + (s - 1)^2$. Die unabhängige Paarung kostet $\mathbb{E}(\epsilon - x_0)^2 = m^2 + 1 + s^2$. In einer Dimension ist die sortierte Paarung für jede konvexe Kostenfunktion optimal: Kreuzen sich zwei Paare, können wir ihre Ziele vertauschen, ohne die Kosten zu erhöhen (Aufgabe :exref:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`d in :numref:`sec_mdl-test-functions-transport`). Damit gilt

<!--en:bd729f9e-->
Der Abstand trennt die Kosten für die Verschiebung des Mittelwerts von denen für die Änderung der Breite. Zwischen $\mathcal{N}(0, 4)$ und $\mathcal{N}(0, 1)$ betragen die minimalen Kosten $1$ gegenüber $5$ bei unabhängigen Paaren (Teilbilder (a) und (b) in :numref:`fig_mdl-dyn-couplings-energy`). Bei unabhängiger Paarung entstehen vier Fünftel der Kosten dadurch, dass Masse an anderer Masse vorbeigeführt wird.

<!--en:f0bc16ce-->
Damit erklärt sich, wo die Trajektorien aus :numref:`fig_mdl-dyn-fm-paths` enden. In einer Dimension können sich die Trajektorien eines Flusses nicht kreuzen. Ein Fluss erhält also die Reihenfolge, und die ordnungserhaltende Paarung ist optimal. Auf der Geraden erreicht daher eine Runde Reflow den optimalen Transport (Aufgabe :exref:`ex_mdl-flow-matching-transport-reflow-two-points`).

<!--en:12016509-->
In zwei Dimensionen reicht Kreuzungsfreiheit nicht aus. Drehen wir eine standardnormalverteilte Wolke innerhalb einer Zeiteinheit um eine Vierteldrehung um den Ursprung. Dazu verwenden wir das Feld, das dem Punkt $(\xi_1, \xi_2)$ die Geschwindigkeit $\tfrac{\pi}{2}\, (-\xi_2, \xi_1)$ zuweist. Keine zwei Trajektorien kreuzen sich, und die Wolke endet dort, wo sie begonnen hat, also gilt $W_2^2(p_0, p_1) = 0$. Dennoch bewegt sich jedes Teilchen, und zwar mit der Geschwindigkeit $\tfrac{\pi}{2}\|\mathbf{x}\|$. Es ist wie auf einem Karussell: Alle fahren mit, doch als Ganzes kommt die Menge nicht vom Fleck (Teilbild (c) in :numref:`fig_mdl-dyn-couplings-energy`).

<!--en:8580af6b-->
Die Kreisbögen zu begradigen behebt das Problem nicht. Wir paaren jedes $\mathbf{x}_0$ mit seiner Vierteldrehung $\mathbf{R}\mathbf{x}_0$ und verbinden die Punkte durch eine Strecke. Die Strecken $\big((1 - t)\mathbf{I} + t\mathbf{R}\big)\mathbf{x}_0$ kreuzen sich nie, weil die Matrix für jedes $t$ invertierbar ist. Jede Strecke wird mit konstanter Geschwindigkeit durchlaufen. Also gilt $C = 0$, und Reflow lässt die Paarung unverändert. Dennoch kostet sie $\mathbb{E}\|\mathbf{R}\mathbf{x}_0 - \mathbf{x}_0\|^2 = 4$ gegenüber $W_2^2 = 0$. Auf einer Geraden lässt sich nur durch Kreuzungen Bewegung verschwenden. In der Ebene kann eine Paarung gerade und kreuzungsfrei und dennoch ungünstig sein. Der Satz von Brenier zeigt, was dieser Paarung fehlt. Eine optimale Abbildung ist der Gradient einer konvexen Funktion. Eine Vierteldrehung ist überhaupt kein Gradient: Die Ableitung eines Gradienten ist eine symmetrische Matrix, und $\mathbf{R}$ ist nicht symmetrisch.

<!--en:a1f4e0bf-->
![Drei Möglichkeiten, eine Gauß-Verteilung zu bewegen. (a) Unabhängige Paarung von Daten $\mathcal{N}(0,4)$ bei $t=0$ mit Rauschen $\mathcal{N}(0,1)$ bei $t=1$, dargestellt für zehn gleichmäßig verteilte Quantilniveaus ($0.05, 0.15, \ldots, 0.95$). Die Strecken kreuzen sich und die mittlere quadrierte Verschiebung beträgt $5$. (b) Die sortierte Paarung $x_0=2\epsilon$ ist kreuzungsfrei und kostet $1$, den quadrierten Wasserstein-Abstand. (c) Eine standardnormalverteilte Wolke in der Ebene nach einer Vierteldrehung. Die Drehbewegung entlang der Kreisbögen (gestrichelt) hat die kinetische Energie $\pi^2/2$. Die Sehnen, die jeden Punkt mit seinem um eine Vierteldrehung versetzten Bild verbinden (Pfeile), sind gerade, kreuzen sich nie und kosten $4$. Die Wolke endet dort, wo sie begonnen hat, also gilt $W_2^2=0$. In zwei Dimensionen muss eine gerade, kreuzungsfreie Paarung nicht optimal sein.](../img/mdl-dyn-couplings-energy.svg)

<!--en:888ad0c6-->
Eine Paarung hält nur fest, wo jedes Teilchen endet. Wie sollen wir die Bewegung dazwischen bewerten? Betrachten wir ein Teilchen, das sich in einer Zeiteinheit von $\mathbf{x}_1$ nach $\mathbf{x}_0$ entlang eines Pfades $\mathbf{x}_t$ bewegt. Nach der Jensen-Ungleichung für das Zeitmittel gilt

<!--en:3655d4a6-->
mit Gleichheit nur bei konstanter Geschwindigkeit. Das Integral des quadrierten Geschwindigkeitsbetrags ist auf einer Geraden bei konstanter Geschwindigkeit am kleinsten. Jeder Umweg und jede Tempoänderung erhöhen es. Sein Mittelwert über die Teilchen ist die **kinetische Energie** eines Flusses, $\int_0^1 \mathbb{E}\|\mathbf{u}_t(\mathbf{x}_t)\|^2\, dt$. Das Karussell hat die kinetische Energie $(\pi/2)^2\, \mathbb{E}\|\mathbf{x}\|^2 = \pi^2/2 \approx 4.9$, und diese ist vollständig verschwendet.

<!--en:84d234af-->
Anhand der kinetischen Energie können wir auch Flüsse mit denselben Endpunkten vergleichen. Betrachten wir drei Flüsse zwischen den Daten $\mathcal{N}(0, 4)$ und dem Rauschen $\mathcal{N}(0, 1)$: den geraden Fluss der sortierten Paarung, den marginalen Fluss der geraden Strecken bei unabhängiger Paarung und den Kosinuspfad aus :numref:`sec_mdl-score-velocity-dictionary`. Welcher ist am günstigsten? Schneidet ein aus geraden Strecken zusammengesetzter Fluss besser ab als einer, der anders entsteht? Alle drei haben gaußsche Randverteilungen $\mathcal{N}(0, s_t^2)$. Daher ist ihre Geschwindigkeit $(\dot{s}_t/s_t)\, x$ und ihre kinetische Energie $\int_0^1 \dot{s}_t^2\, dt$ (Aufgabe :exref:`ex_mdl-flow-matching-transport-kinetic-energy`). Zunächst überprüft die Codezelle durch Sortieren von Samples die beiden Paarungskosten $5$ und $1$.

<!--en:05e0cd24-->
Der gerade Fluss der sortierten Paarung ist am günstigsten. Seine kinetische Energie beträgt $W_2^2 = 1$. Überraschend ist die Reihenfolge der beiden anderen. Die geraden Strecken der unabhängigen Paarung ergeben einen marginalen Fluss, der $1.86$ kostet und damit mehr als der Kosinuspfad mit $1.23$. Seine Varianz $s_t^2 = 4(1 - t)^2 + t^2$ sinkt zunächst, wenn $t$ von $1$ auf $0.8$ fällt, und wächst dann wieder. Seine Teilchen wandern also nach innen und wieder nach außen. Gerade bedingte Strecken ergeben bei der falschen Paarung keinen geraden Fluss.

<!--en:1fe155e5-->
### Der Satz von Benamou und Brenier

<!--en:2ad9cd9e-->
Der günstigste der drei Flüsse kostete genau $W_2^2$, so viel wie die günstigste Paarung. Das ist keine Besonderheit von Gauß-Verteilungen. Der günstigste Fluss zwischen zwei Verteilungen kostet immer so viel wie die günstigste Paarung. Er transportiert jedes Paar dieser Paarung entlang seiner Verbindungsstrecke mit konstanter Geschwindigkeit. Das ist das weitreichendste Ergebnis dieses Abschnitts. Es verbindet das statische Problem :eqref:`eq_mdl-w2`, bei dem wir Endpunkte paaren, mit den Objekten, die wir beim Flow-Matching lernen: den Flüssen.

<!--en:2fc42dfc-->
**Satz (Benamou und Brenier).** *$p_0$ und $p_1$ seien Verteilungen mit endlichen zweiten Momenten. Eine der beiden Verteilungen, hier die Rauschverteilung $p_1$, habe eine Dichte. Ein Paar $(p_t, \mathbf{u}_t)$ heiße zulässig, wenn $\mathbf{u}_t$ stetig differenzierbar und Lipschitz-stetig in $\mathbf{x}$ mit einer von $t$ unabhängigen Konstanten ist und das Paar die Kontinuitätsgleichung mit den Endpunkten $p_0$ und $p_1$ erfüllt. (i) Jedes zulässige Paar hat mindestens die kinetische Energie $W_2^2(p_0, p_1)$. (ii) Wir verbinden die Paare einer optimalen Kopplung durch gerade Strecken. Ist die marginale Geschwindigkeit dieses Pfades zulässig, so beträgt seine kinetische Energie $W_2^2(p_0, p_1)$. Damit gilt für das Minimum über alle zulässigen Paare* :cite:`Benamou.Brenier.2000`

<!--en:8a28541d-->
Die Bedingung in (ii) ist beispielsweise erfüllt, wenn die Brenier-Abbildung ein glatter Diffeomorphismus ist und sowohl ihre Ableitung als auch die Ableitung ihrer Umkehrabbildung beschränkt sind. Ohne diese Bedingung gilt die Gleichheit weiterhin, wenn wir das Minimum über schwache Lösungen der Kontinuitätsgleichung mit quadratintegrierbarer Geschwindigkeit bilden. Diese allgemeine Form beweisen wir nicht, doch unsere Beispiele brauchen sie: Das optimale Feld für die beiden Punkte hat die Steigung $1/t$ und ist nicht zulässig.

<!--en:aa1b2b0f-->
**Beweis.** Wir zeigen, dass $W_2^2$ die kleinste kinetische Energie sowohl von unten als auch von oben beschränkt. Schritt II liefert die untere Schranke (i), Schritt III ergibt (ii).

<!--en:e6608e07-->
I. *Jeder Fluss definiert eine Kopplung.* Sei $\mathbf{Z}_t$ die Lösung von $\dot{\mathbf{Z}}_t = \mathbf{u}_t(\mathbf{Z}_t)$ rückwärts in der Zeit mit dem Anfangswert $\mathbf{Z}_1 \sim p_1$. Nach dem Eindeutigkeitsargument aus der Proposition zu Reflow gilt $\mathbf{Z}_t \sim p_t$ für jedes $t$. Daher ist $(\mathbf{Z}_0, \mathbf{Z}_1)$ eine Kopplung von $p_0$ und $p_1$.

<!--en:9d3d2c87-->
II. *Untere Schranke.* Die Kosten dieser Kopplung betragen mindestens $W_2^2$. Nach der Jensen-Ungleichung für das Zeitmittel sind sie höchstens so groß wie die kinetische Energie:

<!--en:95e42b41-->
III. *Obere Schranke.* Wir wählen eine optimale Kopplung $\pi^\star$ und ihre geradlinige Interpolation $\mathbf{X}_t = (1 - t)\mathbf{X}_0 + t\mathbf{X}_1$ mit marginaler Geschwindigkeit $\mathbf{u}_t(\mathbf{x}) = \mathbb{E}[\mathbf{X}_1 - \mathbf{X}_0 \mid \mathbf{X}_t = \mathbf{x}]$. Nach der schwachen Form der Proposition zur marginalen Geschwindigkeit erfüllt dieses Paar die Kontinuitätsgleichung mit den richtigen Endpunkten. Für seine kinetische Energie gilt nach der bedingten Jensen-Ungleichung

<!--en:f57ac540-->
Die beiden Schranken stimmen überein. $\blacksquare$

<!--en:f402ec81-->
Was sagt der Satz über das Modell aus? Trainieren wir Rectified Flow mit einer optimalen Paarung, erfüllt sich der Wunsch aus der Einleitung: Die Verlustuntergrenze ist null, das exakte Feld ist entlang jeder Trajektorie konstant, und ein Euler-Schritt der Länge eins integriert es fehlerfrei. Beide Eigenschaften folgen aus den Gleichheitsfällen im Beweis. Die beiden Ungleichungen kennen wir aus der Proposition zu Reflow. Ist der Fluss einer optimalen Kopplung zulässig, so erreicht er die untere Schranke. In III gilt für ihn also Gleichheit, und die bedingte Varianz ist null: Die Strecken einer optimalen Kopplung kreuzen sich nicht. In der Jensen-Ungleichung in II gilt Gleichheit nur, wenn sich jedes Teilchen mit konstanter Geschwindigkeit bewegt. Jeder Minimierer bewegt seine Teilchen also geradlinig mit konstanter Geschwindigkeit. Die Umkehrung gilt nicht, wie die Vierteldrehung zeigt. Ein Fluss kann gerade, kreuzungsfrei und in einem Schritt exakt sein, ohne der günstigste zu sein. Er erreicht $W_2^2$ nur, wenn auch seine Paarung optimal ist (Aufgabe :exref:`ex_mdl-flow-matching-transport-reflow-fixed-points`).

<!--en:4566db42-->
Eine optimale Kopplung ist also das Ziel. In der Praxis gibt es zwei Näherungen dafür. Reflow bestimmt die neuen Paare mithilfe des Modells. Eine Paarung durch *optimalen Transport im Minibatch* (engl. minibatch optimal transport) :cite:`Tong.Fatras.Malkin.ea.2023,Pooladian.BenHamu.DomingoEnrich.ea.2023` braucht kein trainiertes Modell. Vor jedem Trainingsschritt ordnet sie die Rausch- und Datenpunkte des Batches so einander zu, dass die Summe der quadrierten Abstände minimal ist. In Aufgabe :exref:`ex_mdl-flow-matching-transport-minibatch-ot` trainieren wir mit dieser Paarung auf den beiden Halbmonden und messen, wie sie das Verlustplateau, die Bewertung nach einem Schritt und die Länge der Paare verändert. Keines der Verfahren findet die optimale Kopplung der beiden Verteilungen. Ein Transportplan für einen Batch hängt vom jeweiligen Batch ab. Reflow bleibt bei jeder kreuzungsfreien Paarung stehen, unabhängig davon, ob sie optimal ist, wie die Vierteldrehung zeigt. Außerdem übernimmt Reflow die Fehler des ersten Modells.

<!--en:eba4c8e0-->
## Sampling mit einem gelernten Feld

<!--en:b534fa50-->
Wenn wir den Fluss begradigen, sparen wir Netzauswertungen. Wir können die Auswertungen aber auch besser nutzen: mit einem numerischen Verfahren höherer Ordnung oder mit Schritten dort, wo auf dem Pfad Bewegung stattfindet. Beides ist mit einem gelernten Score ebenso möglich wie mit einer gelernten Geschwindigkeit. Denn die Wahrscheinlichkeitsfluss-ODE aus :numref:`sec_mdl-score-based-generative-modeling` macht aus einem Score ein Geschwindigkeitsfeld, auf einem varianzerhaltenden Pfad sogar genau das Feld, das Flow-Matching lernen würde. Für ein Sample brauchen wir pro Stufe des numerischen Verfahrens eine Feldauswertung. Wie nutzen wir ein vorgegebenes Budget an Auswertungen? Wir untersuchen das in einem Experiment mit einem Feld ohne Lernfehler. Jeder beobachtete Fehler stammt also vom numerischen Verfahren.

<!--en:cd2ad25d-->
Das Heun-Verfahren ist von zweiter Ordnung, wertet das Feld aber zweimal pro Schritt aus. Lohnen sich die zwei Auswertungen? Wir verwenden die Wahrscheinlichkeitsfluss-ODE mit dem *exakten* Score unserer Beispielmischung mit Komponenten bei $\pm 2$ und Varianz $0.25$. Für den varianzerhaltenden Zeitplan steigt $\beta(t)$ linear von $0.1$ auf $20$. Die Codezelle integriert von $t = 1$ bis $t = 0$ mit dem Euler- und dem Heun-Verfahren (:numref:`sec_mdl-euler-runge-kutta`). Für $8{,}000$ Startpunkte vergleicht sie die Endpunkte mit einer Heun-Lösung aus $800$ Schritten. Beide Verfahren erhalten dasselbe Budget von $10$ bis $80$ Feldauswertungen (Zeile NFE in der ausgegebenen Tabelle, für „number of function evaluations“). Wir verteilen sie auf zwei verschiedene Zeitgitter: gleich große Schritte in $t$ und das *quadratische* Gitter $t_k = (1 - k/K)^2$ für $k = 0, \ldots, K$, dessen Schritte zum Datenende $t = 0$ hin kleiner werden. Der Fehler des Heun-Verfahrens sinkt quadratisch mit der Schrittweite, der des Euler-Verfahrens linear. Bei genügend Auswertungen muss Heun also gewinnen. Reichen zehn?

<!--en:3df3dc47-->
Auf dem gleichmäßigen Gitter erkennen wir die Ordnung an den Konvergenzraten. Jede Verdopplung des Budgets teilt den Euler-Fehler durch etwa zwei. Beim Heun-Verfahren steigt dieser Faktor von etwa drei auf fast vier. Bei gleichen Kosten unterliegt Heun jedoch mit $10$ Auswertungen (etwa $0.08$ gegenüber $0.04$ bei Euler) und mit $20$ (fast $0.03$ gegenüber $0.02$). Erst ab $40$ liegt Heun vorn. Auf dem quadratischen Gitter gewinnt Heun bei jedem Budget. Mit $10$ Auswertungen liegt sein Fehler von etwa $0.03$ unter dem Euler-Fehler auf beiden Gittern. Für Euler wirkt das quadratische Gitter umgekehrt: Der Fehler steigt auf mehr als das Doppelte. Bei diesem Feld entscheidet für kleine Budgets also das Gitter, welches Verfahren gewinnt.

<!--en:9faa611e-->
Wie kann ein Verfahren erster Ordnung eines zweiter Ordnung schlagen? Der Endpunktfehler ist eine Summe vorzeichenbehafteter Schrittfehler, und Fehler mit verschiedenen Vorzeichen können sich aufheben. Die nächste Codezelle misst, in welchem Maß das geschieht. Sie bestimmt den Fehler des $k$-ten groben Schritts, der an einem Punkt des exakten Pfades beginnt. Diesen Fehler multipliziert sie mit der Empfindlichkeit des Endpunkts gegenüber einer Verschiebung zu diesem Zeitpunkt. Diese Empfindlichkeit liest sie an einer benachbarten Trajektorie des exakten Flusses ab. Das Produkt $c_k$ ist in erster Näherung der Beitrag des Schritts zum Endpunktfehler. Die Codezelle gibt zwei Tabellen aus: den Anteil $|\sum_k c_k| / \sum_k |c_k|$, der nach der Summation übrig bleibt, und die Summe $\sum_k |c_k|$ vor jeder Aufhebung, jeweils gemittelt über die Startpunkte. Ihre letzte Zeile zeigt, welcher Anteil der jeweiligen Pfadlänge zwischen $t = 1$ und $t = 0.5$ liegt.

<!--en:8352b3bd-->
Bis $t = 0.5$ haben die Pfade weniger als ein Zehntel ihrer Länge zurückgelegt. Denn die verrauschte Verteilung liegt während des größten Teils dieser Zeit nahe bei $\mathcal{N}(0, 1)$, deren Score die Drift aufhebt. Bei gleich großen Schritten in $t$ verbrauchen wir die Hälfte des Budgets für dieses Zehntel. Deshalb erproben wir überhaupt ein zum Datenende hin dichteres Gitter.

<!--en:76ee7482-->
Die Anteile erklären den Vorsprung von Euler. Bei gleich großen Schritten bleibt im Endpunkt für jedes Budget nur etwa ein Achtel der Euler-Schrittfehler übrig, gegenüber mehr als der Hälfte bei Heun. Vor der Aufhebung sind die Schrittfehler bei Euler größer: Ihre Summen in der zweiten Tabelle betragen bei jedem Budget mehr als das Doppelte der Heun-Werte. Wäre ein ebenso großer Anteil übrig geblieben, hätte Euler überall verloren. Sein Vorsprung beruht auf der Aufhebung von Fehlern. Das ist eine Eigenschaft dieses Feldes auf diesem Gitter, keine Eigenschaft des Verfahrens. Das quadratische Gitter nimmt Euler diesen Vorteil: Dort bleibt etwa ein Drittel der Schrittfehler von Euler übrig, bei Heun ein Viertel bis zwei Fünftel.

<!--en:9d6a2105-->
Erstaunlicherweise ist das fast alles, was sich mit dem quadratischen Gitter ändert. Die Summen sind nahezu unabhängig vom Gitter. Für Heun mit $10$ Auswertungen beträgt die Summe $0.13$ bei gleich großen Schritten und $0.125$ auf dem quadratischen Gitter. Bei jedem Budget liegen die beiden Werte nur wenige Prozent auseinander. Die Summe für Euler ist auf dem quadratischen Gitter sogar kleiner, um etwa ein Siebtel bei $10$ Auswertungen und ein Zehntel bei $80$. Trotzdem steigt sein Endpunktfehler auf mehr als das Doppelte. Insgesamt sind die Schrittfehler durch die Verdichtung also nur wenig gesunken. Geändert hat sich vor allem, wie stark sie sich aufheben, zum Vorteil von Heun und zum Nachteil von Euler. Das quadratische Gitter ist gröber, als es aussieht: Zum Zeitpunkt $t$ beträgt die Schrittweite etwa das $2\sqrt{t}$-Fache der gleichmäßigen Schrittweite. Unterhalb von $t = 1/4$ ist sie kleiner, oberhalb größer.

<!--en:6ab163b0-->
Auch Sampler für Bildmodelle verdichten ihre Schritte zum Datenende hin. Wie stark sie das tun, stimmen sie anhand von Messungen ab :cite:`Karras.Aittala.Aila.ea.2022`, wie in :numref:`sec_diffusion-samplers` beschrieben. Unsere Tabellen zeigen, warum solche Messungen nötig sind: Ein Vergleich numerischer Verfahren auf einem Gitter gilt für eben dieses Gitter.

<!--en:b822572d-->
Zwei Fragen bleiben offen. Niemand weiß, wie man in hohen Dimensionen mit geringem Aufwand die optimale Paarung findet. Reflow und die neue Paarung innerhalb eines Batches nähern sie nur an. Auch die Ausgangsfrage des Kapitels, mit wie wenigen Schritten ein Modell auskommt, beantworten wir für jedes neue Feld weiterhin durch Ausprobieren. Schrittzahl und Lage der Schritte stimmen wir empirisch ab. Eine Herleitung fehlt. :numref:`chap_diffusion` greift beide Fragen für Bilder auf.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:bf785e3e-->
Den Pfad vom Rauschen zu den Daten können wir wählen. Die Geschwindigkeit, die ihn realisiert, ist der A-posteriori-Mittelwert der Geschwindigkeiten der einzelnen Sample-Pfade. Daher können wir sie mit bedingtem Flow-Matching nach der Methode der kleinsten Quadrate lernen. Die Verlustuntergrenze $C$ misst, wie stark die Pfade in ihren Geschwindigkeiten voneinander abweichen. Alle Modelle dieses und des vorigen Abschnitts lernen auf diese Weise den bedingten Mittelwert einer Größe, die für jedes Sample in geschlossener Form bekannt ist. Zum Sampling verwenden sie ein numerisches Verfahren. Auf einem Gauß-Pfad sind Score, Rauschen und Geschwindigkeit affine Funktionen desselben A-posteriori-Mittelwerts. Ein einziges Netz kann also jede dieser Größen liefern. Gerade bedingte Pfade ergeben einen gekrümmten Fluss, wo sich Strecken kreuzen. Reflow paart das Rauschen neu mit den Endpunkten des Flusses selbst. Dabei steigen die Transportkosten nicht, und bei den beiden Halbmonden leistet ein Euler-Schritt so viel wie zweiunddreißig. Die Kosten können nicht unter den quadrierten Wasserstein-Abstand fallen. Er ist die kleinste kinetische Energie aller Flüsse zwischen den Endpunkten. Eine geradlinig gleichförmige Bewegung erreicht sie nur bei optimaler Paarung. Wie viele Schritte ein Feld braucht, hängt auch davon ab, wo sie liegen, denn das Gitter bestimmt, wie stark sich die Schrittfehler aufheben. Auf dem exakten Feld unserer Beispielmischung unterliegt das Heun-Verfahren dem Euler-Verfahren auf einem gleichmäßigen Gitter bei weniger als etwa $40$ Auswertungen. Auf einem zum Datenende hin verdichteten Gitter gewinnt Heun bei jedem Budget.

<!--en:2294243a-->
## Übungen

<!--en:107a5b9d-->
1. **Stochastische Interpolanten.** ● :citet:`Albergo.Boffi.VandenEijnden.2023` schreiben die Pfade dieses und des vorigen Abschnitts in der Form $\mathbf{x}_t = a_t \mathbf{x}_0 + b_t \mathbf{x}_1 + c_t \mathbf{w}$. Dabei ziehen wir $(\mathbf{x}_0, \mathbf{x}_1)$ aus einer Kopplung von $p_0$ und $p_1$. Die Zufallsvariable $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ist von beiden unabhängig, und für die glatten Zeitpläne gelten $a_0 = b_1 = 1$ und $a_1 = b_0 = c_0 = c_1 = 0$. Ist $p_1$ gaußverteilt, so sind die Diffusionspfade und die Pfade von Rectified Flow der einseitige Fall $c \equiv 0$. Der latente Term $c_t \mathbf{w}$ dient für zweiseitige Pfade zwischen beliebigen Verteilungen $p_0$ und $p_1$.
    1. Setzen Sie $\mathbf{z} = (\mathbf{x}_0, \mathbf{x}_1, \mathbf{w})$ in die marginale Geschwindigkeit :eqref:`eq_mdl-marginal-velocity` ein. Zeigen Sie, dass sie gleich $\mathbb{E}[\dot{a}_t \mathbf{x}_0 + \dot{b}_t \mathbf{x}_1 + \dot{c}_t \mathbf{w} \mid \mathbf{x}_t = \mathbf{x}]$ ist, und geben Sie den zugehörigen CFM-Verlust :eqref:`eq_mdl-cfm-loss` an.
    1. Geben Sie Zeitpläne an, die Rectified Flow :eqref:`eq_mdl-rf-path` und den Kosinuspfad mit Daten- und Rauschkoeffizienten $\cos \tfrac{\pi t}{2}$ und $\sin \tfrac{\pi t}{2}$ ergeben, jeweils mit unabhängigen Endpunkten und $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$.
    1. Zeigen Sie für $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$, unabhängige Endpunkte und $c_t > 0$, dass der Interpolant dieselbe marginale Geschwindigkeit hat wie der Interpolant mit $c \equiv 0$ und $\sqrt{b_t^2 + c_t^2}$ anstelle von $b_t$.
    1. Zeigen Sie für $c_t > 0$, dass $\nabla \log p_t(\mathbf{x}) = -\mathbb{E}[\mathbf{w} \mid \mathbf{x}_t = \mathbf{x}] / c_t$ gilt. Was leistet das Rauschen im Inneren des Pfades, wenn $p_1$ eine beliebige Verteilung statt einer Gauß-Verteilung ist?
   :exlabel:`ex_mdl-score-matching-diffusion-flow-stochastic-interpolants`

<!--en:8dd8e4b1-->
1. **Späte Entscheidungen.** Betrachten Sie das Zweipunktbeispiel mit den Daten $\pm a$.
    1. Geben Sie dem Punkt $+a$ das Gewicht $w$ und dem Punkt $-a$ das Gewicht $1 - w$. Wiederholen Sie die Herleitung von :eqref:`eq_mdl-two-point-velocity` und zeigen Sie, dass sich der A-posteriori-Mittelwert $a \tanh\big((1 - t)\, a x/t^2 + \tfrac12 \log\tfrac{w}{1 - w}\big)$ ergibt. Wo landet nun ein Euler-Schritt von $t = 1$ bis $t = 0$?
    1. Zeigen Sie für gleiche Gewichte, dass $\partial_x u_t(0) = \big(1 - (1 - t)\, a^2/t^2\big)/t$ gilt. Zeigen Sie weiter, dass der Ursprung bei abnehmendem $t$ an der Stelle $t^* = \tfrac12\big(\sqrt{a^4 + 4a^2} - a^2\big)$ von anziehend zu abstoßend wechselt. Diese Stelle liegt bei $2\sqrt{2} - 2 \approx 0.83$ für $a = 2$. Wie verhält sich $t^*$ für $a \to \infty$ und für $a \to 0$? Interpretieren Sie beide Grenzwerte.
    1. Zeigen Sie, dass die Steigung $\partial_x u_t$ fern vom Ursprung gegen $1/t$ geht und dass ein Euler-Schritt von $t = h$ bis $t = 0$ genau auf $\pm a$ landet, wenn er an einem Punkt beginnt, an dem der $\tanh$ gesättigt ist. Welcher Teil des Flusses begrenzt also ein Verfahren mit fester Schrittweite, und zu welchen Zeiten?
    1. Die Verlustuntergrenze beträgt $C = \mathbb{E}_t \mathbb{E}_x \textrm{Var}(x_0 \mid x_t)/t^2$ mit $\textrm{Var}(x_0 \mid x_t = x) = a^2 \operatorname{sech}^2\big((1 - t)\, a x/t^2\big)$. Zeigen Sie, dass $C \le a^2 + 1$ gilt. Erklären Sie dann, warum die Zeitpunkte nahe $t = 0$ fast nichts zu $C$ beitragen, obwohl die bedingte Geschwindigkeit $(x - x_0)/t$ dort unbeschränkt ist. Nutzen Sie, dass die Dichte von $x_t$ nahe $0$ die Größenordnung $e^{-(1 - t)^2 a^2/2t^2}/t$ hat.
   :exlabel:`ex_mdl-flow-matching-transport-late-decisions`

<!--en:f06f2ee0-->
1. **Kinetische Energie von Gauß-Pfaden.**
    1. Zeigen Sie für $p_t = \mathcal{N}(0, s_t^2)$ in einer Dimension, dass die Geschwindigkeit $u_t(x) = (\dot{s}_t/s_t)\, x$ die Kontinuitätsgleichung erfüllt und die kinetische Energie $\int_0^1 \dot{s}_t^2\, dt$ beträgt. Warum braucht eine Gauß-Wolke keine andere Geschwindigkeit?
    1. Geben Sie für die Daten $\mathcal{N}(0, 4)$ und das Rauschen $\mathcal{N}(0, 1)$ die Größe $s_t$ für die drei Flüsse der Codezelle an: den geraden Fluss der sortierten Paarung, den marginalen Fluss gerader Strecken bei unabhängiger Paarung und den Kosinuspfad. Zeigen Sie, dass die kinetischen Energien der ersten beiden Flüsse genau $1$ und $5 - \pi$ betragen.
    1. Wiederholen Sie die unabhängige Paarung für die Daten $\mathcal{N}(0, 1)$. Zeigen Sie, dass ihr marginaler Fluss die kinetische Energie $2 - \pi/2 \approx 0.43$ hat, obwohl $W_2^2 = 0$ gilt. Welche Paarung kostet nichts, und wie sieht ihr Fluss aus?
   :exlabel:`ex_mdl-flow-matching-transport-kinetic-energy`

<!--en:0f77541c-->
1. [code] **Eine Reflow-Runde für zwei Punkte.** Betrachten Sie die Daten $\pm a$, das Rauschen $\mathcal{N}(0, 1)$, die unabhängige Paarung und den marginalen Fluss :eqref:`eq_mdl-two-point-velocity` des Zweipunktbeispiels.
    1. Zeigen Sie mithilfe der Proposition zur Ordnungserhaltung aus :numref:`subsec_mdl-flow-order` und der Symmetrie, dass die Flussabbildung von $t = 1$ bis $t = 0$ durch $\epsilon \mapsto a \operatorname{sign}(\epsilon)$ gegeben ist. Eine Reflow-Runde erzeugt also die sortierte Paarung.
    1. Zeigen Sie, dass ihre Kosten $a^2 - 2a\sqrt{2/\pi} + 1$ betragen, also etwa $1.81$ für $a = 2$, gegenüber $a^2 + 1 = 5$ bei unabhängiger Paarung. Erklären Sie, warum diese Kosten gleich $W_2^2$ sind.
    1. Zeigen Sie, dass sich die Strecken $(1 - t)\, a \operatorname{sign}(\epsilon) + t\, \epsilon$ nach Reflow nie kreuzen und die Verlustuntergrenze der zweiten Runde daher $C = 0$ beträgt. Geben Sie das zugehörige Feld an. Ist ein Euler-Schritt mit diesem Feld exakt?
    1. Integrieren Sie das in geschlossener Form bekannte Feld mit RK4 ausgehend von $20{,}000$ gaußverteilten Ziehungen bei $t = 1$ bis hinunter zu $t = 10^{-3}$ (die Steigung des Feldes wächst wie $1/t$, halten Sie also vor $0$ an). Überprüfen Sie die Endpunkte und die Kosten.
    1. Die Verlustuntergrenze der unabhängigen Paarung beträgt $C \approx 2.03$ für $a = 2$. Berechnen Sie sie durch Quadratur aus der Formel in Teil (d) von Aufgabe :exref:`ex_mdl-flow-matching-transport-late-decisions`. Bestimmen Sie den Zeitpunkt, oberhalb dessen $99\%$ der Verlustuntergrenze anfallen. Überprüfen Sie die Schranke $\mathbb{E}\|\mathbf{Z}_1 - \mathbf{Z}_0\|^2 \le \mathbb{E}\|\mathbf{X}_1 - \mathbf{X}_0\|^2 - C$ aus der Proposition zu Reflow. Wie viel Spielraum lässt sie hier?
   :exlabel:`ex_mdl-flow-matching-transport-reflow-two-points`

<!--en:cd1a4b95-->
1. [code] **Paarung durch optimalen Transport im Minibatch.** Ersetzen Sie `independent_pairs` durch einen Sampler, der seinen Batch in Blöcken aus je $256$ Rauschpunkten und $256$ Datenpunkten zieht. Paaren Sie jeden Block neu, indem Sie die Zuordnung mit der kleinsten Summe quadrierter Abstände wählen (`scipy.optimize.linear_sum_assignment` auf der $256 \times 256$-Kostenmatrix). Trainieren Sie mit `train_rf`.
    1. Geben Sie das Verlustplateau, die quadrierte Energiedistanz bei $1$, $2$, $8$ und $32$ Schritten sowie die mittlere quadrierte Paarlänge vor und nach der neuen Paarung an. Vergleichen Sie mit der unabhängigen Paarung und mit Reflow.
    1. Zeigen Sie, dass ein neu gepaarter Rauschpunkt weiterhin standardnormalverteilt ist und ein neu gepaarter Datenpunkt weiterhin aus der Datenverteilung stammt. Das marginale Feld transportiert also nach wie vor $p_1$ nach $p_0$.
    1. Erklären Sie, warum die Paarung im Batch keine optimale Kopplung der beiden Verteilungen ist und was bei wachsender Blockgröße geschieht. Welches der beiden Verfahren zur neuen Paarung braucht ein trainiertes Modell, und welches übernimmt dessen Fehler?
   :exlabel:`ex_mdl-flow-matching-transport-minibatch-ot`

<!--en:0d3460ff-->
1. **Konditionierung der Vorhersageziele.** Betrachten Sie den Gauß-Pfad :eqref:`eq_mdl-gaussian-path` mit eindimensionalen Daten $x_0 \sim \mathcal{N}(0, 1)$ und kosinusförmigem Zeitplan.
    1. Berechnen Sie als Funktionen von $t$ die Varianzen der gezogenen Trainingsziele aus :numref:`tab_mdl-score-noise-velocity`: des bedingten Scores $-\epsilon/\sigma_t$, des Rauschens $\epsilon$, des rauschfreien Punkts $x_0$, der Geschwindigkeit $\dot{\alpha}_t x_0 + \dot{\sigma}_t \epsilon$ und der $\mathbf{v}$-Zielgröße $\alpha_t \epsilon - \sigma_t x_0$. Welche Varianzen wachsen unbeschränkt, und an welchem Ende des Pfades?
    1. Ein auf das Rauschen trainiertes Netz liefert $\hat{\epsilon}$ mit Fehler $\delta$. Zeigen Sie, dass der daraus berechnete rauschfreie Punkt $(x - \sigma_t \hat{\epsilon})/\alpha_t$ bis auf das Vorzeichen den Fehler $\sigma_t \delta/\alpha_t$ aufweist. Zeigen Sie auch, dass die Rauschvorhersage aus einem vorhergesagten rauschfreien Punkt mit Fehler $\delta$ den Fehler $\alpha_t \delta/\sigma_t$ hat. An welchem Ende des Pfades verstärkt sich der jeweilige Fehler?
    1. Zeigen Sie, dass die Zielgeschwindigkeit auf dem Kosinuspfad das $\pi/2$-Fache der $\mathbf{v}$-Zielgröße ist. Die $\mathbf{v}$-Vorhersage ist also eine Geschwindigkeitsvorhersage. Zeigen Sie außerdem, dass die aus einer Rauschvorhersage berechnete Geschwindigkeit den Fehler $\pi\delta/(2\alpha_t)$ hat. Welche Parametrisierungen halten den Fehler an beiden Enden beschränkt?
    1. Die Umrechnungen aus :numref:`tab_mdl-score-noise-velocity` gelten für beliebige Daten. Prüfen Sie :eqref:`eq_mdl-score-velocity` numerisch für die Mischung $\tfrac12 \mathcal{N}(-2, 0.25) + \tfrac12 \mathcal{N}(2, 0.25)$ auf dem Kosinuspfad bei $t = 0.4$. Berechnen Sie auf einem Gitter von Punkten $x$ die marginale Geschwindigkeit auf zwei Arten: einmal als A-posteriori-Mittelwert von $\dot{\alpha}_t x_0 + \dot{\sigma}_t \epsilon$, wobei Sie jede Komponente mit ihrer A-posteriori-Wahrscheinlichkeit gewichten, und einmal aus dem Score der verrauschten Mischung. Beide Ergebnisse sollten bis auf Rundungsfehler übereinstimmen. Welche Zahlen in der Rechnung ändern sich, wenn die Mischungsgewichte nun $0.8$ und $0.2$ betragen, und welche bleiben gleich?
   :exlabel:`ex_mdl-flow-matching-transport-target-conditioning`

<!--en:9d04625d-->
1. **Reflow erhält die Randverteilungen und kann zum Stillstand kommen.**
    1. Beweisen Sie Teil (i) der Proposition zu Reflow im Detail für den geraden Pfad, dessen bedingte Verteilungen Punktmassen sind. Zeigen Sie für eine glatte Testfunktion $\varphi$ mit kompaktem Träger, dass $\frac{d}{dt}\mathbb{E}\,\varphi(\mathbf{X}_t) = \mathbb{E}\big[\nabla\varphi(\mathbf{X}_t) \cdot \mathbf{u}_t(\mathbf{X}_t)\big]$ gilt. Das ist die Kontinuitätsgleichung in schwacher Form. Wo geht die Turmeigenschaft des bedingten Erwartungswerts ein und wo die Voraussetzung der Lipschitz-Stetigkeit?
    1. Sei $\pi_k$ die Paarung nach $k$ Runden. Zeigen Sie, dass $\mathbb{E}_{\pi_k}\|\mathbf{x}_1 - \mathbf{x}_0\|^2$ monoton fällt und durch $W_2^2$ nach unten beschränkt ist, also konvergiert.
    1. Zeigen Sie: Ändert sich die Paarung in einer Runde nicht, gilt also $\pi_{k+1} = \pi_k$, dann ist $C = 0$ für $\pi_k$. Bestimmen Sie den Fixpunkt in einer Dimension.
    1. Paaren Sie in zwei Dimensionen jedes $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ mit $\mathbf{R}\mathbf{x}_0$, wobei $\mathbf{R}$ die Vierteldrehung ist. Zeigen Sie, dass sich die Strecken $\big((1 - t)\mathbf{I} + t\mathbf{R}\big)\mathbf{x}_0$ nie kreuzen (die Matrix ist für jedes $t$ invertierbar, denn ihre Eigenwerte sind $(1 - t) \pm \mathrm{i}\, t$). Damit gilt $C = 0$ und Reflow ändert nichts. Die Kosten betragen jedoch $\mathbb{E}\|\mathbf{R}\mathbf{x}_0 - \mathbf{x}_0\|^2 = 4$ gegenüber $W_2^2 = 0$. Für welche Matrizen $\mathbf{A}$ ist die lineare Abbildung $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$ der Gradient einer konvexen Funktion und damit eine optimale Paarung von $\mathcal{N}(\mathbf{0}, \mathbf{I})$ mit seiner Bildverteilung?
   :exlabel:`ex_mdl-flow-matching-transport-reflow-fixed-points`

<!--en:c6fa5d85-->
1. [code] **Ein stochastischer Sampler für ein Flow-Matching-Modell.** ●
    1. Stellen Sie auf dem geraden Pfad $\alpha_t = 1 - t$, $\sigma_t = t$ die Identität :eqref:`eq_mdl-score-velocity` so um, dass Sie $\nabla \log p_t(\mathbf{x}) = -\big(\mathbf{x} + (1 - t)\, \mathbf{u}_t(\mathbf{x})\big)/t$ erhalten.
    1. Zeigen Sie, dass für jedes $g \ge 0$ die SDE $d\mathbf{X} = \big[\mathbf{u}_t(\mathbf{X}) - \tfrac12 g^2 \nabla \log p_t(\mathbf{X})\big] dt + g\, d\bar{\mathbf{W}}$ bei Integration von $t = 1$ zurück bis $t = 0$ mit einer zeitlich rückwärts laufenden brownschen Bewegung $\bar{\mathbf{W}}$ die Randverteilungen $p_t$ hat. Substituieren Sie $\tau = 1 - t$, schreiben Sie die Fokker-Planck-Gleichung in $\tau$ auf und verwenden Sie $\nabla \cdot (p \nabla \log p) = \Delta p$. Warum geht der Score-Term hier mit einem Minuszeichen ein? Ein trainiertes Geschwindigkeitsfeld definiert somit eine Familie stochastischer Sampler.
    1. Implementieren Sie den Sampler mit dem Euler-Maruyama-Verfahren für das trainierte Feld der beiden Halbmonde mit $g \in \{0, \tfrac12, 1\}$ und $K \in \{8, 32, 128\}$ Schritten. Werten Sie den Score nur bei $t = 1, 1 - 1/K, \ldots, 1/K$ aus, da er bei $t = 0$ singulär ist. Geben Sie die quadrierte Energiedistanz für jeden Lauf an. Welches $g$ braucht bei vorgegebener Qualität die wenigsten Schritte? Hilft Rauschen bei $K = 8$?
    1. Wiederholen Sie das Experiment für das Reflow-Feld. Erklären Sie anhand seiner geraden Trajektorien, warum kein in einem stochastischen Schritt erzeugtes Sample an das in einem deterministischen Schritt erzeugte Sample herankommt.
   :exlabel:`ex_mdl-flow-matching-transport-stochastic-sampler`

<!--en:8d978bc2-->
[Diskussion](https://d2l.discourse.group/t/36869)

<!--en:de8e941a-->
Flow-Matching, Rectified Flow und optimaler Transport<br>**Wahrscheinlichkeitspfade · bedingtes Flow-Matching · Umrechnung zwischen Score und Geschwindigkeit · Reflow · optimaler Transport nach Wasserstein**

<!--en:b6ff0af8-->
::: {.slide title="Warum brauchen Diffusionssampler so viele Schritte?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:4e7e144b-->
- **Gekrümmte Diffusionspfade:** Die Verrauschung durch eine OU- oder VP-SDE ergibt gekrümmte Trajektorien der Wahrscheinlichkeitsfluss-ODE. Bei der Mischung dieses Abschnitts legen sie mehr als $90\%$ ihrer Länge unterhalb von $t = 0.5$ zurück. Ein numerisches Verfahren braucht 20–50 Schritte, um ihnen zu folgen.

<!--en:d758b261-->
- **Idealfall einer geradlinig gleichförmigen Bewegung:** Bewegen sich Teilchen entlang $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$ ohne Kreuzungen ($\dot{\mathbf{x}}_t = \mathbf{x}_1 - \mathbf{x}_0$), ist ein **einziger** Euler-Schritt ($h = 1$) zu $100\%$ exakt.

<!--en:30fadf6b-->
- **Prinzip des Flow-Matchings:** $p_t$ direkt vorgeben und $\mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ mit der Methode der kleinsten Quadrate an die Geschwindigkeiten der einzelnen Samples anpassen.

<!--en:b1b57b59-->
::: {.slide title="Zeitkonvention: Daten bei t = 0 und Rauschen bei t = 1"}

<!--en:50236848-->
[Einheitliche Zeitrichtung im Kapitel]{.kicker}

<!--en:ccf7df02-->
Für den direkten Vergleich von scorebasierter Diffusion (§{sec_mdl-score-matching-diffusion-flow}) und Flow-Matching (§{sec_mdl-flow-matching-transport}) gilt im ganzen Buch **eine Zeitrichtung**: Die Daten $\mathbf{x}_0 \sim p_0$ liegen bei $t = 0$ und das standardnormalverteilte Rauschen $\mathbf{x}_1 = \boldsymbol{\epsilon} \sim p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ bei $t = 1$.

<!--en:230e31fe-->
- **Sampling durch Integration von $t = 1$ zurück bis $t = 0$:** Die Geschwindigkeit $\mathbf{u}_t(\mathbf{x})$ zeigt in Richtung der Verrauschung ($t: 0 \to 1$). Ein generativer Euler-Schritt subtrahiert daher $\Delta t\,\mathbf{u}_t(\mathbf{x})$.

<!--en:ea5879c0-->
- **Umrechnung in die Konvention der Flow-Matching-Literatur:** Für die Konvention mit Rauschen bei $t = 0$ und Daten bei $t = 1$ die Zeit umkehren ($t \mapsto 1 - t$), die Endpunkte $(\mathbf{x}_0, \mathbf{x}_1)$ vertauschen und das Vorzeichen der Geschwindigkeit wechseln ($\mathbf{u} \mapsto -\mathbf{u}$).

<!--en:46f0cd6c-->
::: {.slide title="Wahrscheinlichkeitspfade und marginales Geschwindigkeitsfeld"}

<!--en:49a0efea-->
[Formulierung des Flow-Matchings]{.kicker}

<!--en:ead67c6c-->
Ein Geschwindigkeitsfeld $\mathbf{u}_t: \mathbb{R}^d \to \mathbb{R}^d$ **realisiert** einen Wahrscheinlichkeitspfad $(p_t)_{t \in [0, 1]}$ zwischen $p_0$ und $p_1$, wenn beide zusammen die Kontinuitätsgleichung $\partial_t p_t + \nabla \cdot (p_t\,\mathbf{u}_t) = 0$ erfüllen. Wie lässt sich $(p_t, \mathbf{u}_t)$ konstruieren, wenn $p_0$ nur durch Samples bekannt ist?

<!--en:483186a9-->
- **$p_t$ aus bedingten Teilwolken aufbauen:** Auf eine Variable $\mathbf{z} \sim q$ bedingen (einen rauschfreien Datenpunkt $\mathbf{x}_0$ oder ein gekoppeltes Paar $(\mathbf{x}_0, \mathbf{x}_1)$) und eine handhabbare bedingte Dichte $p_t(\mathbf{x} \mid \mathbf{z})$ mit bekannter Geschwindigkeit $\mathbf{u}_t(\mathbf{x} \mid \mathbf{z})$ wählen, für die $\partial_t p_t(\mathbf{x} \mid \mathbf{z}) = -\nabla\cdot\big(p_t(\mathbf{x} \mid \mathbf{z})\mathbf{u}_t(\mathbf{x} \mid \mathbf{z})\big)$ gilt.

<!--en:d5ed02ce-->
- **Proposition (die marginale Geschwindigkeit ist ein A-posteriori-Mittelwert):** Die Mischung $p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{z})q(\mathbf{z})\,d\mathbf{z}$ erfüllt die Kontinuitätsgleichung $\partial_t p_t + \nabla\cdot(p_t\mathbf{u}_t) = 0$ mit der **marginalen Geschwindigkeit**:
  $$\mathbf{u}_t(\mathbf{x}) = \mathbb{E}\!\left[\mathbf{u}_t(\mathbf{x} \mid \mathbf{z}) \,\middle|\, \mathbf{x}_t = \mathbf{x}\right] = \int \mathbf{u}_t(\mathbf{x} \mid \mathbf{z})\,\frac{p_t(\mathbf{x} \mid \mathbf{z})\,q(\mathbf{z})}{p_t(\mathbf{x})}\,d\mathbf{z}$$

<!--en:2b3c177e-->
- **Beweis in einer Zeile:** Differenzieren unter dem Integralzeichen ergibt $\partial_t p_t(\mathbf{x}) = -\nabla\cdot\int p_t(\mathbf{x} \mid \mathbf{z})\mathbf{u}_t(\mathbf{x} \mid \mathbf{z})q(\mathbf{z})\,d\mathbf{z} = -\nabla\cdot\big(p_t(\mathbf{x})\mathbf{u}_t(\mathbf{x})\big)$.

<!--en:c630b1e8-->
::: {.slide title="Der Satz zum bedingten Flow-Matching (CFM)"}

<!--en:43edd1f9-->
[Training ohne Simulation]{.kicker}

<!--en:4b266ef8-->
Da die marginale Geschwindigkeit der A-posteriori-Mittelwert $\mathbf{u}_t(\mathbf{x}) = \mathbb{E}[\mathbf{u}_t(\mathbf{x} \mid \mathbf{z}) \mid \mathbf{x}_t = \mathbf{x}]$ ist, stimmen nach dem Regressionslemma die marginale und die bedingte Zielfunktion bis auf die konstante Verlustuntergrenze $C$ überein:

<!--en:8576d6b7-->
- **Verlustuntergrenze durch Kreuzungsvarianz $C$:** $C = \mathbb{E}[\operatorname{Var}(\mathbf{u}_t(\mathbf{x} \mid \mathbf{z}) \mid \mathbf{x}_t)]$ verschwindet ($C = 0$) genau dann, wenn sich verschiedene Sample-Pfade nie kreuzen.

<!--en:09cb8b65-->
::: {.slide title="Umrechnung zwischen Score und Geschwindigkeit auf Gauß-Pfaden"}

<!--en:42c84e26-->
[Diffusion und Flow-Matching zusammenführen]{.kicker}

<!--en:e794630f-->
Auf jedem Gauß-Pfad $\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\boldsymbol{\epsilon}$ lassen sich Score $\mathbf{s}$, Rauschen $\hat{\boldsymbol{\epsilon}}$, rauschfreier Punkt $\hat{\mathbf{x}}_0$ und Geschwindigkeit $\mathbf{u}_t$ durch affine Transformationen ineinander umrechnen:

<!--en:deb85976-->
- **Ein A-posteriori-Mittelwert:** Ob das Netz $\mathbf{s}_{\boldsymbol{\theta}}$, $\hat{\boldsymbol{\epsilon}} = -\sigma_t\mathbf{s}$, $\hat{\mathbf{x}}_0 = (\mathbf{x} + \sigma_t^2\mathbf{s})/\alpha_t$ oder $\mathbf{u}_{\boldsymbol{\theta}}$ lernt, der zugrunde liegende bedingte Erwartungswert ist immer derselbe.

<!--en:06c1d693-->
::: {.slide title="Beispiel in geschlossener Form: zwei Datenpunkte bei ±2"}

<!--en:9dc306cb-->
[Analytisches Flow-Matching]{.kicker}

<!--en:fc682c88-->
Zwei Datenpunkte $x_0 \in \{-a, +a\}$ ($a = 2$, jeweils mit Wahrscheinlichkeit $\frac{1}{2}$) **unabhängig** mit Rauschen $\epsilon \sim \mathcal{N}(0, 1)$ entlang des geraden Pfades $x_t = (1 - t)x_0 + t\epsilon$ paaren ($\dot{x}_t = \epsilon - x_0 = \frac{x_t - x_0}{t}$):

<!--en:c37df62b-->
- **Exakte marginale Geschwindigkeit:** Nach dem Satz von Bayes gilt $\mathbb{E}[x_0 \mid x_t = x] = a\tanh\!\big(\frac{(1-t)ax}{t^2}\big)$. Daraus folgt
  $$u_t(x) = \frac{x - a\tanh\!\big(\frac{(1-t)ax}{t^2}\big)}{t}$$

<!--en:7442a4e5-->
- **Späte Wahl des Modus ($t^* = 2\sqrt{2} - 2 \approx 0.83$):** Die Steigung am Ursprung ist $\partial_x u_t(0) = \frac{1}{t}\big(1 - \frac{(1-t)a^2}{t^2}\big)$. Von $t = 1$ bis hinunter zu $t^* = 0.83$ zieht $\partial_x u_t(0) > 0$ die Teilchen *in Richtung* $x = 0$. Erst unterhalb von $t^* = 0.83$ stößt der Ursprung die Teilchen in Richtung $\pm 2$ ab.

<!--en:974f7053-->
::: {.slide title="Warum ein Euler-Schritt unabhängige Paare auf 0 abbildet"}

<!--en:b1785a1c-->
[Ein Schritt zum Kollaps]{.kicker}

<!--en:a6025d39-->
Bei $t = 1$ enthält ein Rauschsample $\mathbf{x}_1 = \boldsymbol{\epsilon}$ keinerlei Information über seinen unabhängig gezogenen Datenpartner $\mathbf{x}_0$. Daher gelten $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_1 = \mathbf{x}] = \mathbb{E}[\mathbf{x}_0]$ und $\mathbf{u}_1(\mathbf{x}) = \mathbf{x} - \mathbb{E}[\mathbf{x}_0]$. Ein einziger Euler-Schritt der Länge $1$ von $t = 1$ bis $t = 0$ bildet also **jedes** Sample auf den Datenmittelwert $\mathbf{x} - \mathbf{u}_1(\mathbf{x}) = \mathbb{E}[\mathbf{x}_0] = \mathbf{0}$ ab:

<!--en:ac4adb7a-->
::: {.slide title="Rectified Flow: gerade Strecken, die sich kreuzen"}

<!--en:3d6d655b-->
[Warum sich der marginale Fluss krümmt]{.kicker}

<!--en:6b62505a-->
**Rectified Flow** paart $\mathbf{x}_0 \sim p_0$ und $\boldsymbol{\epsilon} \sim p_1$ entlang gerader Strecken $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$ bei konstanter Geschwindigkeit und passt $\mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ mit der Methode der kleinsten Quadrate an die konstante Differenz $\boldsymbol{\epsilon} - \mathbf{x}_0$ an:

<!--en:96f0d904-->
- **Sich kreuzende Strecken und kreuzungsfreie ODE-Flüsse:** Kreuzen sich zwei Strecken $(\mathbf{x}_0, \boldsymbol{\epsilon})$ und $(\mathbf{x}_0', \boldsymbol{\epsilon}')$ bei $(\mathbf{x}, t)$, so verlangen sie dort zwei verschiedene Geschwindigkeiten.

<!--en:fc2ebf4f-->
- **Mittelung erzeugt Krümmung:** Das marginale Feld $\mathbf{u}_t(\mathbf{x})$ mittelt die Geschwindigkeiten an der Kreuzung. Da durch jeden Punkt genau eine Trajektorie einer Lipschitz-stetigen ODE verläuft, können sich die integrierten Pfade nicht kreuzen. Sie **biegen voneinander weg** und tauschen ihre Endpunkte.

<!--en:3cc041e9-->
::: {.slide title="Zwei Halbmonde als Benchmark und die quadrierte Energiedistanz"}

<!--en:8c0c25b7-->
[Sample-Qualität in zwei Dimensionen]{.kicker}

<!--en:48748b9e-->
Zum Vergleich in $\mathbb{R}^2$ dienen $8{,}192$ standardisierte Trainingspunkte aus der Verteilung der beiden Halbmonde. Den Abstand der Verteilung zu $2{,}048$ zurückgehaltenen Punkten misst die **quadrierte Energiedistanz**:

<!--en:4d073528-->
- **Kalibrierung der Metrik:** Zwei unabhängige Stichproben der Größe $2{,}048$ aus der wahren Datenverteilung ergeben einen Wert, der nur wenige Tausendstel von null abweicht. Fallen alle Samples auf dem Datenmittelwert $\mathbf{0}$ zusammen (wie bei einem Euler-Schritt mit dem exakten Feld der unabhängigen Paarung), ergibt sich $\approx 0.9$.

<!--en:40f3d9ed-->
::: {.slide title="Die CFM-Verlustuntergrenze vor dem Training vorhersagen"}

<!--en:59d9b795-->
[Verlustuntergrenze durch Kreuzungsvarianz]{.kicker}

<!--en:87bead3e-->
Schon vor dem Training eines neuronalen Netzes lässt sich die exakte, unvermeidbare CFM-Verlustuntergrenze $C = \frac{1}{2}\mathbb{E}_{t, \mathbf{x}_t}\big[\operatorname{Var}(\boldsymbol{\epsilon} - \mathbf{x}_0 \mid \mathbf{x}_t)\big]$ pro Koordinate berechnen. Dazu die A-posteriori-Varianz von $\mathbf{x}_0$ über die $8{,}192$ Trainingspunkte mitteln:

<!--en:9116056d-->
- **Fast 70% der gesamten Zielvarianz entfallen auf die Verlustuntergrenze:** Unabhängige Strecken kreuzen sich zwischen den beiden Halbmonden und $\mathcal{N}(\mathbf{0}, \mathbf{I})$ häufig. Vom gesamten zweiten Moment der Zielgröße entfallen $C \approx 1.37$ von $2.00$ pro Koordinate auf die unvermeidbaren Geschwindigkeitsunterschiede an den Kreuzungen.

<!--en:c6f89555-->
::: {.slide title="Rectified Flow mit unabhängigen Paaren trainieren"}

<!--en:7767e746-->
[Das Verlustplateau durch Kreuzungen erreichen]{.kicker}

<!--en:e8475b34-->
Ein mehrschichtiges Perzeptron (MLP) mit zwei versteckten Schichten, `tanh`-Aktivierung und 256 Einheiten pro Schicht trainieren. Adam mit kosinusförmigem Lernratenabfall über $10{,}000$ Minibatches aus je $2{,}048$ unabhängigen $(\mathbf{x}_0, \boldsymbol{\epsilon})$-Paaren verwenden:

<!--en:bb537e50-->
- **Warum ein Verlust von $1.4$ eine gute Anpassung bedeutet:** Nach dem CFM-Satz gilt $\mathcal{L}_{\text{CFM}}(\boldsymbol{\theta}) = \mathcal{L}_{\text{FM}}(\boldsymbol{\theta}) + C$. Der Endverlust des Netzes von $\approx 1.40$ liegt nur wenige Hundertstel über der Verlustuntergrenze $C \approx 1.37$ der Grundgesamtheit. Das bedeutet $\mathcal{L}_{\text{FM}}(\boldsymbol{\theta}) \approx 0.03$ pro Koordinate.

<!--en:12e273c9-->
::: {.slide title="Sample-Qualität bei unterschiedlicher Zahl von Euler-Schritten für zwei Halbmonde"}

<!--en:86481fc6-->
[Diskretisierungsfehler und Feldfehler]{.kicker}

<!--en:69b78d3c-->
Das trainierte Geschwindigkeitsfeld $\mathbf{u}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ rückwärts von $t = 1$ bis $t = 0$ für $2{,}048$ Rauschpunkte mit $K \in \{1, 2, 4, 8, 16, 32, 64\}$ Euler-Schritten integrieren:

<!--en:8d70307b-->
- **Kollaps nach einem Schritt ($K = 1$, $\mathcal{E}^2 \approx 0.7$):** Entspricht dem theoretischen Kollaps zum Datenmittelwert $\mathbf{0}$.

<!--en:03452278-->
- **Preis der Krümmung ($K = 16\text{–}32$, $\mathcal{E}^2 \le 0.01$):** Die gekrümmten Trajektorien der unabhängigen Paarung erfordern 16–32 Netzauswertungen pro Sample.

<!--en:eccfed89-->
::: {.slide title="Zwei Halbmonde nach 1, 2, 8 und 32 Euler-Schritten"}

<!--en:f84d987b-->
[Fehler des numerischen Verfahrens und des Netzes]{.kicker}

<!--en:8bc0d1c0-->
Die generierten Punktwolken für verschiedene Schrittzahlen machen den Diskretisierungsfehler des numerischen Verfahrens und den Approximationsfehler des Netzes getrennt sichtbar:

<!--en:370ccd58-->
Steigt die Schrittzahl von $K = 1$ auf $K = 8$, entfaltet sich die Wolke um den Mittelwert zu zwei Halbmonden. Bei $K = 32$ verschwinden die meisten verstreuten Punkte dazwischen, weil der Abbruchfehler des Euler-Verfahrens zurückgeht. Die etwas dickeren Halbmonde und die wenigen Ausreißer bei $K = 32$ bleiben auch bei mehr Schritten bestehen. Sie stammen aus dem um $\approx 0.03$ über der Untergrenze $C$ liegenden Regressionsverlust.

<!--en:bae04127-->
::: {.slide title="Reflow: Flüsse durch neue Paarung mit dem Modell begradigen"}

<!--en:8bda1188-->
[Neue Paarung für Rectified Flow]{.kicker}

<!--en:1788e907-->
**Reflow** integriert die trainierte ODE vom Rauschen $\mathbf{Z}_1 = \mathbf{X}_1 \sim p_1$ bis $t = 0$, erhält so deterministische Paare $(\mathbf{Z}_0, \mathbf{Z}_1)$ und trainiert erneut auf deren geraden Verbindungsstrecken:

<!--en:8763b38d-->
- **Randverteilungen bleiben erhalten, Kosten sinken:** Die Jensen-Ungleichung für das Zeitmittel und für die A-posteriori-Verteilung $\mathbb{E}[\mathbf{X}_1 - \mathbf{X}_0 \mid \mathbf{X}_t]$ garantiert geringere konvexe Transportkosten. Die Streckenkreuzungen verschwinden ($C \to 0$).

<!--en:4e998bd2-->
::: {.slide title="Eine Runde Reflow senkt den Verlust von 1.4 auf unter 0.001"}

<!--en:c2b6acda-->
[Streckenkreuzungen beseitigen]{.kicker}

<!--en:2bd0eb15-->
Für $8{,}192$ gekoppelte Paare $(\hat{\mathbf{x}}_0(\boldsymbol{\epsilon}), \boldsymbol{\epsilon})$ das erste Netz über $32$ Euler-Schritte integrieren. Mit den neu gepaarten Endpunkten dieselbe Architektur erneut trainieren:

<!--en:2aa9d14e-->
- **Die Verlustuntergrenze durch Kreuzungen verschwindet:** Da sich Trajektorien des deterministischen ODE-Flusses nie kreuzen, kreuzen sich auch die geraden Sehnen von $\boldsymbol{\epsilon}$ nach $\hat{\mathbf{x}}_0(\boldsymbol{\epsilon})$ in der Raumzeit fast nie. Die Untergrenze für den Trainingsverlust sinkt von $1.37$ auf unter $0.001$.

<!--en:9ba6ce66-->
::: {.slide title="Nach einer Runde Reflow leistet ein Schritt so viel wie 32 Schritte"}

<!--en:b060e29e-->
[Reflow an zwei Halbmonden gemessen]{.kicker}

<!--en:1e5e5c14-->
Vergleich von ursprünglichem Netz und Reflow-Netz bei $K \in \{1, 2, 32\}$ Euler-Schritten und anhand der Verschiebungskosten der Paare:

<!--en:1bb33926-->
- **32-fache Beschleunigung beim Test:** Ein **einziger Euler-Schritt** des Reflow-Netzes erzielt dieselbe quadrierte Energiedistanz ($\le 0.002$, die Auflösungsgrenze der Metrik) wie $32$ Schritte des ursprünglichen Modells. Dabei sinkt die mittlere quadrierte Paarlänge um etwa den Faktor $13\times$ ($\approx 4.0 \to \approx 0.3$, weit unter der theoretischen Schranke $4.00 - 2.75 = 1.25$).

<!--en:d7a8fb2c-->
::: {.slide title="Geradlinigkeit der Trajektorien durch ihren Abstand zur Sehne messen"}

<!--en:3db9fce9-->
[Geometrische Begradigung]{.kicker}

<!--en:fb969edc-->
Wie weit entfernen sich die ODE-Trajektorien mit $32$ Schritten von der geraden Strecke (Sehne) zwischen ihrem Startpunkt im Rauschen $\mathbf{Z}_1$ und dem generierten Datenendpunkt $\mathbf{Z}_0$?

<!--en:7afeeb92-->
- **Etwa 50-mal geringere Abweichung:** Die ursprünglichen Trajektorien krümmen sich nach innen zum Ursprung $\mathbf{0}$, bevor sie nach außen zu den Halbmonden abbiegen (mittlerer Abstand zur Sehne $0.25$). Nach Reflow verlaufen die Trajektorien fast geradlinig (mittlerer Abstand zur Sehne $\approx 0.005$).

<!--en:cd92c348-->
::: {.slide title="Optimaler Transport, Wasserstein-Abstand und Drehungen"}

<!--en:a240bc11-->
[Statische und dynamische Kopplungen]{.kicker}

<!--en:c50b132c-->
Eine **Kopplung** $\pi \in \Pi(p_0, p_1)$ ist eine gemeinsame Verteilung mit Randverteilungen $p_0$ und $p_1$. Der quadrierte **2-Wasserstein-Abstand** ist das Minimum der mittleren quadrierten Verschiebung über alle Kopplungen:

<!--en:af709b4f-->
- **Monotone Sortierung in 1D und Vierteldrehung in 2D:** In einer Dimension genügt es für Optimalität, die Reihenfolge ohne Kreuzungen zu erhalten ($W_2^2(\mathcal{N}(0, 1), \mathcal{N}(m, s^2)) = m^2 + (s - 1)^2 = 1$ für $m=0, s=2$ gegenüber $5$ bei unabhängigen Paaren).

<!--en:88bed06c-->
- **Warum Kreuzungsfreiheit in $\mathbb{R}^d$ ($d \ge 2$) nicht genügt:** Die Paarung von $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ mit seiner Drehung um $90^\circ$, also $\mathbf{R}\mathbf{x}_0$, ergibt gerade Strecken $((1-t)\mathbf{I} + t\mathbf{R})\mathbf{x}_0$, die **sich nie kreuzen** ($C = 0$, ein Fixpunkt von Reflow). Dennoch betragen ihre Kosten $\mathbb{E}\|\mathbf{R}\mathbf{x}_0 - \mathbf{x}_0\|^2 = 4$ bei $W_2^2 = 0$. Nach dem **Satz von Brenier** muss eine optimale Abbildung der Gradient einer konvexen Funktion $\nabla\psi$ sein (symmetrische Jacobi-Matrix). Die Drehung hat dagegen eine von null verschiedene Rotation, denn $\mathbf{R} \ne \mathbf{R}^\top$.

<!--en:95acb3f1-->
::: {.slide title="Satz zum dynamischen optimalen Transport (Benamou und Brenier)"}

<!--en:1cd6fdda-->
[Untere Schranke für die kinetische Energie]{.kicker}

<!--en:c914b350-->
Wie hängt der statische Wasserstein-Abstand $W_2^2(p_0, p_1)$ mit den stetigen Geschwindigkeitsfeldern $\mathbf{u}_t(\mathbf{x})$ zusammen, die ein Modell durch Flow-Matching lernt?

<!--en:d9071ba9-->
**Satz (Benamou und Brenier, 2000):** Unter allen zulässigen Paaren $(p_t, \mathbf{u}_t)$, die die Kontinuitätsgleichung $\partial_t p_t + \nabla\cdot(p_t\mathbf{u}_t) = 0$ von $p_0$ bis $p_1$ lösen, ist die minimale **kinetische Energie** gleich $W_2^2(p_0, p_1)$:
$$W_2^2(p_0, p_1) = \min_{(p_t, \mathbf{u}_t)} \int_0^1 \int_{\mathbb{R}^d} \|\mathbf{u}_t(\mathbf{x})\|^2\,p_t(\mathbf{x})\,d\mathbf{x}\,dt$$
Das Minimum ergibt sich, wenn jedes Paar $(\mathbf{x}_0, \mathbf{x}_1)$ der optimalen Brenier-Kopplung $\pi^\star$ eine Gerade mit konstanter Geschwindigkeit durchläuft ($C = 0$, exakt in $1$ Euler-Schritt).

<!--en:687e3afa-->
::: {.slide title="Verfahrensordnung und Wahl des Zeitgitters bei gleichem Budget"}

<!--en:f1752f58-->
[Euler und Heun für die Wahrscheinlichkeitsfluss-ODE]{.kicker}

<!--en:79c06b55-->
Beim Sampling auf gekrümmten Pfaden einer Wahrscheinlichkeitsfluss-ODE stehen nur $10$ bis $80$ Funktionsauswertungen (NFE) zur Verfügung. Schlägt **Heun** als Verfahren zweiter Ordnung ($2$ NFE/Schritt) stets **Euler** als Verfahren erster Ordnung ($1$ NFE/Schritt)?

<!--en:14aecf94-->
- **Abhängigkeit vom Gitter:** Auf einem **gleichmäßigen Gitter** schlägt Euler das Heun-Verfahren bei 10 und 20 NFE. Auf einem **quadratischen Gitter** $t_k = (1 - k/K)^2$, das nahe dem Datenende $t = 0$ dichter liegt, gewinnt Heun bei jedem Budget und senkt den Fehler bei 10 NFE auf etwa $0.034$.

<!--en:773d83c4-->
::: {.slide title="Warum Euler auf gleichmäßigen Gittern führt: Schrittfehler heben sich auf" layout="tall-output"}

<!--en:d30a4c4f-->
[Schrittfehler untersuchen]{.kicker}

<!--en:e721b421-->
Die Zerlegung des Endpunktfehlers in vorzeichenbehaftete Schrittbeiträge $c_k$, die bis $t = 0$ fortwirken, zeigt das Zusammenspiel von Gitterwahl und Verfahrensordnung:

<!--en:550e3d76-->
- **Die meiste Bewegung kommt spät:** Die Pfade legen $>90\%$ ihrer Länge unterhalb von $t = 0.5$ zurück. Nahe $t = 1$ entspricht die Verteilung fast $\mathcal{N}(0, 1)$. Bei gleich großen Schritten entfällt also oberhalb von $t = 0.5$ das halbe NFE-Budget auf $<10\%$ der Strecke.

<!--en:264a8207-->
- **Zufällige Aufhebung durch die Vorzeichen:** Die Summe der absoluten Schrittfehler $\sum_k |c_k|$ ist bei Euler auf beiden Gittern etwa $2\times$ so groß wie bei Heun oder größer. Doch auf dem gleichmäßigen Gitter heben sich $88\%$ durch entgegengesetzte Vorzeichen auf (verbleibender Anteil $0.12$). Bei dichteren Schritten nahe $t = 0$ setzt sich die Genauigkeit $O(h^2)$ von Heun durch.

<!--en:774f5804-->
::: {.slide title="Verständnisfragen zu Flow-Matching, Reflow und Transport"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:250ced7e-->
Fragen zum bedingten Flow-Matching, zu Reflow und zum optimalen Transport:

<!--en:20923ef6-->
1. **Ein Euler-Schritt bei unabhängiger Paarung:** Gegeben sei ein Datensatz in $\mathbb{R}^d$ mit Mittelwert $\boldsymbol{\mu} = (3, -1)$, unabhängig gepaart mit $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ entlang $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$. Wohin schickt ein einziger Euler-Schritt von $t = 1$ bis $t = 0$ mit der exakten marginalen Geschwindigkeit $\mathbf{u}_1(\mathbf{x})$ jeden Rauschpunkt?

<!--en:10a07107-->
2. **Optimalität von Reflow in einer Dimension:** Warum erreicht bei eindimensionalen Daten $p_0$ und gaußschem Rauschen $p_1$ schon **eine** exakte Reflow-Runde stets die exakte $W_2$-optimale Transportkopplung mit Verlustuntergrenze $C = 0$?

<!--en:57f423aa-->
3. **Fixpunkte von Reflow in zwei Dimensionen:** Ein Beispiel für eine zweidimensionale Kopplung zwischen $p_0 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ und $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ angeben, deren gerade Strecken sich nie kreuzen ($C = 0$, also ändert Reflow die Kopplung nicht), obwohl ihre Transportkosten strikt größer als $W_2^2(p_0, p_1) = 0$ sind.

<!--en:fa6dac62-->
::: {.slide title="Verständnisfragen zu Flow-Matching, Reflow und Transport (Fortsetzung)"}

<!--en:26a317d8-2-->
[Verständnisfragen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:0c1d6ab0-->
1. Bei $t = 1$ gilt $\mathbf{u}_1(\mathbf{x}) = \mathbb{E}[\boldsymbol{\epsilon} - \mathbf{x}_0 \mid \boldsymbol{\epsilon} = \mathbf{x}] = \mathbf{x} - \mathbb{E}[\mathbf{x}_0] = \mathbf{x} - \boldsymbol{\mu}$. Ein Euler-Schritt $\mathbf{x} - 1 \cdot \mathbf{u}_1(\mathbf{x}) = \boldsymbol{\mu} = (3, -1)$ bildet jeden Punkt auf den Datenmittelwert ab.

<!--en:daf4f6b1-->
2. In einer Dimension können sich die Trajektorien einer Lipschitz-stetigen ODE nicht kreuzen. Die Flussabbildung $\Phi_{1 \to 0}$ ist also streng monoton wachsend. Die eindeutige monotone Abbildung in einer Dimension paart die Quantile in sortierter Reihenfolge. Sie ist der Gradient einer konvexen Funktion und minimiert jede konvexe Transportkostenfunktion ($W_2^2$).

<!--en:652cb388-->
3. Die Paarung durch Drehung um $90^\circ$ lautet $\mathbf{x}_1 = \mathbf{R}\mathbf{x}_0$ mit $\mathbf{R} = \left(\begin{smallmatrix}0 & -1 \\ 1 & 0\end{smallmatrix}\right)$. Die Matrix $(1-t)\mathbf{I} + t\mathbf{R}$ ist für alle $t \in [0, 1]$ invertierbar (Eigenwerte $(1-t) \pm \mathrm{i}t \ne 0$). Daher kreuzen sich die Strecken nie ($C = 0$), und dennoch gilt $\mathbb{E}\|\mathbf{R}\mathbf{x}_0 - \mathbf{x}_0\|^2 = 4 > W_2^2 = 0$.

<!--en:12176db9-->
::: {.slide title="Zusammenfassung: Flow-Matching, Reflow und optimaler Transport"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:f500b488-->
| Begriff oder Satz | Mathematische Form | Bedeutung für generative Modelle und ihre Umsetzung |

<!--en:2afa6355-->
| **Marginale Geschwindigkeit** | $\mathbf{u}_t(\mathbf{x}) = \mathbb{E}\big[\mathbf{u}_t(\mathbf{x} \mid \mathbf{z}) \mid \mathbf{x}_t = \mathbf{x}\big]$ | Der A-posteriori-Mittelwert der Sample-Geschwindigkeiten erfüllt $\partial_t p_t + \nabla\cdot(p_t\mathbf{u}_t) = 0$ |

<!--en:8fcff639-->
| **CFM-Satz** | $\mathcal{L}_{\text{CFM}}(\boldsymbol{\theta}) = \mathcal{L}_{\text{FM}}(\boldsymbol{\theta}) + C$ | Trainiert $\mathbf{u}_{\boldsymbol{\theta}}$ ohne Simulation mit der Methode der kleinsten Quadrate. Die Verlustuntergrenze $C$ misst Pfadkreuzungen |

<!--en:dc2082bc-->
| **Identität zwischen Score und Geschwindigkeit** | $\mathbf{u}_t(\mathbf{x}) = \frac{\dot{\alpha}_t}{\alpha_t}\mathbf{x} - (\sigma_t\dot{\sigma}_t - \sigma_t^2\frac{\dot{\alpha}_t}{\alpha_t})\nabla\log p_t$ | Rechnet jede Diffusionsvorhersage für Rauschen oder Score ohne erneutes Training in eine Flussgeschwindigkeit um |

<!--en:249131f9-->
| **Rectified Flow und Reflow** | $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$, neue Paarung $(\mathbf{Z}_0, \mathbf{Z}_1)$ | Beseitigt Kreuzungen ($C \to 0$), senkt die Kosten $\mathbb{E}\,c(\mathbf{Z}_1 - \mathbf{Z}_0)$ und ermöglicht die Generierung mit einem Euler-Schritt |

<!--en:75e8328f-->
| **Satz von Benamou und Brenier** | $W_2^2(p_0, p_1) = \min_{(p_t, \mathbf{u}_t)} \int_0^1 \mathbb{E}_{p_t}\|\mathbf{u}_t\|^2\,dt$ | Die minimale kinetische Energie beträgt $W_2^2$. Eine geradlinig gleichförmige Bewegung gemäß der Brenier-Abbildung erreicht sie |
