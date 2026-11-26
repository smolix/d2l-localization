<!--en:a8bb76ca-->
# Entrauschung und Score

<!--en:75d7c82e-->
Minimieren wir beim Training eines Netzes den quadratischen Fehler, so lernt es den bedingten Erwartungswert der Zielgröße bei gegebener Eingabe (:numref:`sec_gan_beyond`). Addieren wir zu einem unverrauschten Datenpunkt $\mathbf{x}_0 \sim p$ isotropes gaußsches Rauschen $\sigma \boldsymbol\epsilon$, erhalten wir $\tilde{\mathbf{x}} = \mathbf{x}_0 + \sigma \boldsymbol\epsilon$. Der optimale Entrauscher (engl. denoiser) im Sinne der kleinsten Quadrate liefert daher den A-posteriori-Mittelwert (engl. posterior mean) $\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}]$. Nach *Tweedies Formel* :cite:`Robbins.1956,Miyasawa.1961,Efron.2011` bestimmt der Gradient der Log-Dichte der verrauschten Daten, $\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$, diesen Mittelwert unmittelbar. Diesen Gradienten nennen wir *Score* oder Score-Funktion. Außerdem zeigte :citet:`Vincent.2011`, dass ein Netz durch Regression auf den Score des gaußschen Störkerns den Score $\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$ lernt. Dafür muss die zugrunde liegende Datendichte $p$ nicht bekannt sein.

<!--en:15fee751-->
Wenn wir den Score anstelle der Dichte lernen, umgehen wir die schwer berechenbare Normierungskonstante energiebasierter Modelle. Trotzdem erhalten wir das Vektorfeld, das gradientenbasierte MCMC-Verfahren wie die Langevin-Dynamik zum Erzeugen von Samples benötigen. Zunächst führen wir ein zweidimensionales Gauß-Mischmodell ein. Für jeden Rauschpegel (engl. noise level) kennen wir seine Dichte, seinen Score und seinen A-posteriori-Mittelwert exakt. An diesem Beispiel überprüfen wir Tweedies Formel und untersuchen, weshalb ein auf einen einzigen Rauschpegel $\sigma$ trainiertes Score-Modell die lokale Struktur der Modi und ihre globalen Gewichte nicht zugleich erfassen kann. Abschließend entrauschen wir Bilder aus Fashion-MNIST bei einem festen Rauschpegel.

<!--en:5a419206-->
## Unser durchgängiges Beispiel

<!--en:d04fb858-->
Wir suchen eine Verteilung, für die wir alle Größen dieses Kapitels bei jedem Rauschpegel exakt berechnen können. So können wir jedes gelernte Feld und jeden Sampler mit den wahren Werten vergleichen. Dafür eignet sich ein Gauß-Mischmodell in der Ebene. Unser Modell hat drei Komponenten mit den Mittelwerten $(-2.5, -1.5)$, $(2.5, -1.5)$ und $(0, 2.5)$, den Mischgewichten $\pi_k = 0.5, 0.3, 0.2$ und derselben Standardabweichung $s = 0.5$ je Koordinate. Verglichen mit ihrer Breite liegen die Modi weit auseinander. Zwischen ihnen hat die Dichte daher tiefe Täler. An den ungleichen Gewichten lässt sich erkennen, ob ein Sampler die richtigen Anteile liefert.

<!--en:372928da-->
Der *Score* einer Dichte $p$ ist der Gradient ihres Logarithmus, $\nabla \log p(\mathbf{x})$. Dieses Vektorfeld zeigt in die Richtung, in der die Dichte am schnellsten wächst. Für das Mischmodell $p(\mathbf{x}) = \sum_k \pi_k\, \mathcal{N}(\mathbf{x}; \boldsymbol\mu_k, s^2\mathbf{I})$ lautet der Score

<!--en:f139bcdd-->
Jede Komponente zieht den Punkt zu ihrem eigenen Mittelwert. Beim Mitteln dieser Beiträge dienen die *Zugehörigkeitswahrscheinlichkeiten* (engl. responsibilities) $r_k(\mathbf{x})$ als Gewichte. Sie geben a posteriori an, mit welcher Wahrscheinlichkeit $\mathbf{x}$ aus Komponente $k$ stammt (:numref:`fig_diffusion-score-field`). Die Mischgewichte $\pi_k$ gehen nur über diese Wahrscheinlichkeiten ein. In der Nähe eines Modus liegt eine Zugehörigkeitswahrscheinlichkeit unabhängig von den Gewichten nahe bei eins. Die Gewichte verändern den Score also nur in den Tälern, wo wenige Punkte liegen. Diese unscheinbare Beobachtung entscheidet darüber, was ein Score-Modell lernen kann.

<!--en:5cf19119-->
![Der Score ist der Gradient der Log-Dichte. Links eine eindimensionale bimodale Dichte (oben) und ihr Score (unten). An beiden Modi und im Tal dazwischen ist der Score null. Rechts unser durchgängiges Beispiel. Die Schattierung zwischen den Höhenlinien stellt die Dichte dar, die Pfeile das Score-Feld. Sie zeigen zu den Modi, ihre Länge ist begrenzt.](../img/mdl-diffusion-score-field.svg)

<!--en:2642f573-->
Auch mit gaußschem Rauschen bleibt ein Gauß-Mischmodell ein Gauß-Mischmodell. Addieren wir zu einem gezogenen Punkt gaußsches Rauschen mit Standardabweichung $\sigma$, so wächst die Varianz jeder Komponente um $\sigma^2$. Aus $s^2$ wird also $s^2 + \sigma^2$. Damit gibt es für Dichte, Score und A-posteriori-Mittelwert bei jedem Rauschpegel geschlossene Formeln. Die folgende Klasse berechnet alle drei Größen. Ihr Argument `alpha` skaliert den unverrauschten Punkt und bleibt bis zum nächsten Abschnitt auf eins gesetzt.

<!--en:deb00be0-->
Im gesamten Kapitel beurteilen wir Sampler für diese Mischung nach denselben Kriterien. Wir berechnen die mittlere negative Log-Likelihood ihrer Punkte unter der wahren Dichte und drei Kennzahlen aus `summary`. Für die *Modusgewichte* zählen wir jeden Punkt beim nächstgelegenen Mittelwert. Der *Ausreißeranteil* gibt an, welcher Anteil der Punkte von jedem Mittelwert weiter als $3s = 1.5$ entfernt ist. Die *Streuung* ist der mittlere quadratische Abstand der übrigen Punkte zum jeweils nächsten Mittelwert. Da die Ausreißer unberücksichtigt bleiben, kann die Streuung höchstens $(3s)^2 = 2.25$ betragen. Erzeugt ein Sampler eine viel zu breite Verteilung, zeigt sich das vor allem am Ausreißeranteil.

<!--en:9a67eaa1-->
Neben diesen Kennzahlen können wir die zweidimensionalen Punktwolken auch visuell vergleichen. Dazu zeichnet `scatter_panels` eine oder mehrere Stichproben nebeneinander mit gleicher Achsenskalierung. Wahlweise blendet die Funktion einen Referenzdatensatz `data` in Hellgrau ein.

<!--en:b01a27a7-->
Welche Werte sollte ein perfekter Sampler ausgeben? Bis auf Stichprobenschwankungen die Gewichte $0.5$, $0.3$ und $0.2$. In zwei Dimensionen ist der quadrierte Abstand eines gaußverteilten Punktes von seinem Mittelwert exponentialverteilt mit Mittelwert $2s^2 = 0.5$. Daher erwarten wir einen Ausreißeranteil von $e^{-(3s)^2/(2s^2)} = e^{-4.5} \approx 1\%$ und für die übrigen Punkte eine Streuung knapp unter $0.5$. Wir prüfen das an 2000 gezogenen Punkten, dem Trainingsdatensatz für die zweidimensionalen Experimente dieses Abschnitts.

<!--en:866ab11a-->
Die Stichprobe liefert die vorhergesagten Werte. Das sind die Zielwerte für jeden Sampler auf diesem Mischmodell. Bei der negativen Log-Likelihood lohnt sich ein genauerer Blick. Rund $2.5$ Nat pro Punkt beträgt die Entropie der Mischung. Kein Dichtemodell erreicht auf neuen Daten im Mittel einen kleineren Wert. Die Punkte eines korrekten Samplers erreichen unter $p$ ungefähr diesen Wert. Für den Sampler ist das ein Zielwert. Ein kleinerer Wert ist nicht unbedingt besser. Häufen sich die Punkte an den Modi, sinkt der Wert, für Punkte in den Tälern steigt er.

<!--en:7c167b7d-->
## Der optimale Entrauscher

<!--en:97ee1d18-->
Für die eingangs beschriebene Regression bieten sich zwei Zielgrößen an, der unverrauschte Punkt und ein Vektor, der auf ihn zeigt. Die Identität, die ihre Minimierer verknüpft, ist das wichtigste Ergebnis dieses Abschnitts.

<!--en:c64fb05a-->
### Rauschen und Regressionsziel

<!--en:8ceacdf9-->
Wir wählen einen Rauschpegel $\sigma > 0$, ziehen einen unverrauschten Punkt $\mathbf{x}_0 \sim p$ und standardnormalverteiltes Rauschen $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ und bilden den verrauschten Punkt $\tilde{\mathbf{x}} = \mathbf{x}_0 + \sigma \boldsymbol\epsilon$. Seine Dichte $p_\sigma$ entsteht durch Glätten der Datendichte mit einer Gauß-Funktion der Breite $\sigma$. Sie ist überall positiv und glatt, selbst wenn das für $p$ nicht gilt. Wir trainieren einen Entrauscher $D$ durch Minimieren von $\mathbb{E}\|D(\tilde{\mathbf{x}}) - \mathbf{x}_0\|^2$. Wie eingangs erwähnt, minimiert der bedingte Mittelwert der Zielgröße bei gegebener Eingabe den quadratischen Fehler (:numref:`sec_gan_beyond`). Der optimale Entrauscher liefert also den *A-posteriori-Mittelwert* $\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}]$. Er mittelt über die unverrauschten Punkte, die $\tilde{\mathbf{x}}$ ergeben haben könnten, und gewichtet jeden Punkt mit der entsprechenden A-posteriori-Wahrscheinlichkeit.

<!--en:27ec3d73-->
Dieselbe Regression lässt auch ein anderes Ziel zu. Bei gegebenem unverrauschtem Punkt ist der verrauschte Punkt gaußverteilt um diesen Punkt. Der Score dieser bedingten Dichte lautet

<!--en:01bd9008-->
und zeigt als Vektor vom verrauschten Punkt zurück zum unverrauschten. Beim *Denoising-Score-Matching* passen wir ein Vektorfeld $\mathbf{s}_{\boldsymbol\theta}$ durch Regression an diesen Vektor an :cite:`Vincent.2011`:

<!--en:d6e52403-->
Der Minimierer ist wieder ein bedingter Mittelwert, diesmal der des bedingten Scores. Vincents Satz gibt an, welchem Feld er entspricht: *Der Score der verrauschten Daten ist der A-posteriori-Mittelwert des bedingten Scores*:

<!--en:55cd4860-->
Der Beweis folgt in zwei Zeilen aus der Bayes-Regel. Das Rauschen glättet die ursprüngliche Dichte, sodass $p_\sigma(\tilde{\mathbf{x}}) = \int p(\mathbf{x}_0)\, p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}_0)\, d\mathbf{x}_0$ gilt. Wir differenzieren unter dem Integral, schreiben den Gradienten des Kerns als Produkt aus dem Kern und dem Gradienten seines Logarithmus und teilen durch $p_\sigma(\tilde{\mathbf{x}})$:

<!--en:88f14be5-->
Der Bruch ergibt nach der Bayes-Regel die A-posteriori-Verteilung $p(\mathbf{x}_0 \mid \tilde{\mathbf{x}})$. Rechts steht also der A-posteriori-Mittelwert des bedingten Scores. Damit ist :eqref:`eq_diffusion-posterior-mean-score` gezeigt. Für diese beiden Zeilen mussten wir kein gaußsches Rauschen voraussetzen. Unter welchen Bedingungen wir unter dem Integral differenzieren dürfen, erläutert :numref:`sec_mdl-denoising-score-matching`.

<!--en:f74c7fc4-->
Durch Regression auf die Zielgrößen der einzelnen Paare lernen wir also den Score von $p_\sigma$, ohne ihn explizit anzugeben. Deshalb können wir in diesem Kapitel Generatoren überhaupt durch Regression trainieren. Den Score der Daten können wir nicht auswerten, den Score des selbst hinzugefügten Rauschens hingegen schon.

<!--en:bc53a216-->
### Tweedies Formel

<!--en:40d693bf-->
Die beiden Regressionen haben den unverrauschten Punkt beziehungsweise $-\boldsymbol\epsilon/\sigma$ als Zielgröße. Die Abbildung $\mathbf{x}_0 \mapsto (\mathbf{x}_0 - \tilde{\mathbf{x}})/\sigma^2$ verknüpft diese Zielgrößen und ist bei gegebenem $\tilde{\mathbf{x}}$ affin. Daher verknüpft sie auch die beiden bedingten Mittelwerte, die den jeweiligen Fehler minimieren. Auflösen nach dem A-posteriori-Mittelwert ergibt *Tweedies Formel* :cite:`Efron.2011`:

<!--en:701440a2-->
Der optimale Entrauscher macht einen Schritt der Schrittweite $\sigma^2$ in Richtung des Scores der verrauschten Dichte (:numref:`fig_mdl-dyn-tweedie`). Entrauscher, Score und Rauschvorhersage sind somit drei Darstellungen desselben A-posteriori-Mittelwerts:

<!--en:8f3284ac-->
Trainieren wir ein Netz auf eine dieser drei Ausgaben, erhalten wir daraus auch die beiden anderen.

<!--en:5bfb0b69-->
Der einfachste Fall zeigt, was dieser Schritt bewirkt. Die Daten seien gaußverteilt gemäß $\mathcal{N}(\boldsymbol\mu, v\mathbf{I})$. Dann hat $p_\sigma = \mathcal{N}(\boldsymbol\mu, (v + \sigma^2)\mathbf{I})$ den Score $-(\tilde{\mathbf{x}} - \boldsymbol\mu)/(v + \sigma^2)$, und Tweedies Formel ergibt

<!--en:b9a9329e-->
Der Entrauscher verkleinert die Abweichung des verrauschten Punkts vom Mittelwert um den Faktor $v/(v + \sigma^2)$. Die A-posteriori-Varianz je Koordinate beträgt $v\sigma^2/(v + \sigma^2)$. Für $v = s^2 = 0.25$ und $\sigma = 0.3$ beträgt der Faktor $0.74$, und der quadratische Fehler je Koordinate sinkt von $\sigma^2 = 0.09$ auf $0.066$. Abseits der Täler entspricht der A-posteriori-Mittelwert unseres Beispiels dieser Schrumpfung (engl. shrinkage) zum nächsten Mittelwert hin. Über beide Koordinaten eines Punkts sollte die Entrauschung den quadratischen Fehler also von $0.18$ auf etwa $0.13$ senken.

<!--en:c2060065-->
### Ein gelernter Score

<!--en:9935cb26-->
Erreicht ein gelernter Score die Güte des optimalen Entrauschers? Dafür brauchen wir ein Netz. In allen zweidimensionalen Experimenten dieses Kapitels verwenden wir ein kleines mehrschichtiges Perzeptron (engl. multilayer perceptron, MLP). Neben dem Punkt erhält es Sinus- und Kosinuswerte des Rauschpegels als Eingabe (ab dem nächsten Abschnitt die einer Zeit $t \in [0, 1]$ entlang eines Pfades von den Daten zum Rauschen). Hier ist der Pegel fest, diese Merkmale spielen also keine Rolle. Ab dem nächsten Abschnitt nutzen wir ein einziges Netz für alle Pegel. Damit es benachbarte Pegel unterscheiden kann, reichen die Kreisfrequenzen von 1 bis 64 Radiant je Einheit von $t$. Die schnellsten Merkmale durchlaufen auf $[0, 1]$ etwa zehn Perioden. Die Positionskodierungen aus :numref:`sec_positional-information` sind für ganzzahlige Tokenpositionen im Tausenderbereich ausgelegt und wären auf diesem Intervall nahezu konstant oder linear. Die Generatoren mit wenigen Schritten in :numref:`sec_diffusion-few-step` verwenden wieder niedrigere Frequenzen. Den Grund erläutern wir dort.

<!--en:9ffdeccd-->
Im gesamten Kapitel verwenden wir dieselbe Trainingsschleife. In jedem Schritt arbeitet Adam mit einem neuen Minibatch und neu gezogenem Rauschen. Die Schleife führt außerdem einen exponentiellen gleitenden Mittelwert (engl. exponential moving average, EMA) der Gewichte mit. Das wird ab :numref:`sec_diffusion-images` wichtig. Die übrigen Optionen stehen im Docstring.

<!--en:cc58a369-->
Für das Mischmodell minimiert ein Netz mit zwei versteckten Schichten zu je 128 Einheiten den Ausdruck :eqref:`eq_diffusion-dsm` in 2000 Schritten. In jedem Schritt verwenden wir 256 der 2000 Trainingspunkte mit neu gezogenem Rauschen. Den gelernten Score behalten wir als Funktion allein des Punkts.

<!--en:e4b2a173-->
Bei $\sigma = 0.3$ verrauschen wir 20 000 neue Punkte und vergleichen drei Schätzungen des unverrauschten Punkts: den verrauschten Punkt selbst, den exakten A-posteriori-Mittelwert und Tweedies Schritt mit dem gelernten Score.

<!--en:42e51d7d-->
Der exakte A-posteriori-Mittelwert senkt den quadratischen Fehler wie im gaußschen Fall vorhergesagt von $0.18$ auf etwa $0.13$. Tweedies Schritt mit dem gelernten Score kommt diesem Wert bis auf etwa ein Prozent nahe. Nach 2000 Schritten mit 2000 Punkten ist das Netz ein nahezu optimaler Entrauscher und liefert somit einen brauchbaren Score.

<!--en:65c52458-->
Der Score ist allerdings etwas ungenauer, als dieses Ergebnis vermuten lässt. In den Entrauscher geht er mit dem Faktor $\sigma^2 = 0.09$ ein. Im Erwartungswert übersteigt der Fehler von Tweedies Schritt den des A-posteriori-Mittelwerts um genau das $\sigma^4$-Fache des mittleren quadratischen Fehlers des Scores. Der Kreuzterm verschwindet, weil der Fehler des A-posteriori-Mittelwerts bei gegebenem $\tilde{\mathbf{x}}$ im Mittel null ist. Die Differenz von rund $0.001$ entspricht somit einem Score-Fehler $\tfrac12\mathbb{E}\|\hat{\mathbf{s}} - \nabla \log p_\sigma\|^2$ von etwa $0.06$. Ein Entrauscher kann nahezu perfekt erscheinen, obwohl sein Score sichtbar abweicht. Gerade diesem Score folgt der Sampler.

<!--en:24331c68-->
### Scores und Normierungskonstanten

<!--en:931ccc28-->
Weshalb lernen wir einen Score anstelle einer Dichte? Ein Netz mit einer skalaren Ausgabe $E_{\boldsymbol\theta}$, der *Energie*, definiert über ein *energiebasiertes Modell* eine Dichte:

<!--en:91603da5-->
Erst die Normierung macht das Lernen sinnvoll. Da die Gesamtmasse feststeht, muss die Dichte anderswo sinken, wenn sie an den Trainingspunkten steigt. Leider ist $Z_{\boldsymbol\theta}$ ein Integral über den gesamten Eingaberaum, das sich bei jeder Aktualisierung ändert. Der Gradient der Log-Likelihood enthält $\nabla_{\boldsymbol\theta} \log Z_{\boldsymbol\theta} = -\mathbb{E}_{p_{\boldsymbol\theta}}[\nabla_{\boldsymbol\theta} E_{\boldsymbol\theta}]$. Diesen Erwartungswert müssen wir durch Sampling aus dem Modell schätzen, in der Praxis mit einer Markov-Kette. Im Score fällt $Z_{\boldsymbol\theta}$ vollständig weg, weil $Z_{\boldsymbol\theta}$ nicht von $\mathbf{x}$ abhängt. Es gilt $\nabla_{\mathbf{x}} \log p_{\boldsymbol\theta} = -\nabla_{\mathbf{x}} E_{\boldsymbol\theta}$.

<!--en:ea31d3ae-->
Mit der *Fisher-Divergenz* vergleichen wir ein Feld $\mathbf{s}$ mit dem Score der Daten. Sie misst den mittleren quadratischen Abstand beider Felder an den Datenpunkten:

<!--en:5f365f80-->
Sie enthält den unbekannten Score der Daten. *Hyvärinens Identität* :cite:`Hyvarinen.2005` eliminiert ihn durch partielle Integration. Bis auf eine Konstante ist $D_{\mathrm{F}}$ gleich $\mathbb{E}_{p}[\tfrac12\|\mathbf{s}\|^2 + \nabla \cdot \mathbf{s}]$, einem Mittelwert über die Daten, der weder Normierungskonstante noch Markov-Kette benötigt (:numref:`sec_mdl-score-matching`). Sein zweiter Term, $\nabla \cdot \mathbf{s}$, ist die Spur der Jacobi-Matrix von $\mathbf{s}$. Die Berechnung kostet einen Rückwärtsdurchlauf je Eingabedimension, bei Bildern also zu viele. Auch diese Kosten entfallen bei der Entrauschung. Nach Vincents Satz entspricht :eqref:`eq_diffusion-dsm` bis auf eine Konstante der Fisher-Divergenz zu $p_\sigma$.

<!--en:1dbdba49-->
## Grenzen eines einzelnen Rauschpegels

<!--en:9f81cb71-->
Trainieren wir einen Entrauscher bei einem einzigen Rauschpegel, erhalten wir den Score einer einzigen geglätteten Verteilung, $p_\sigma$. Um aus $p$ zu ziehen, brauchen wir den Score von $p$. Zunächst stellt sich also die Frage, wie wir $\sigma$ wählen.

<!--en:4c99c160-->
### Wahl des Rauschpegels

<!--en:883be293-->
Ein niedriger Rauschpegel bringt zwei Probleme mit sich. Das erste ist statistischer Natur. Bei Daten mit glatter Dichte wie in unserem Beispiel wächst die Zielgröße $-\boldsymbol\epsilon/\sigma$ wie $1/\sigma$. Der Score, der sich durch Mitteln ergibt, bleibt dagegen beschränkt. Für kleine $\sigma$ besteht die Zielgröße daher überwiegend aus Rauschen, und das Netz kann den Score nur schlecht schätzen. Selbst mit dem exakten Score verbleibt im Verlust die Varianz der Zielgröße um ihren bedingten Mittelwert. In $d$ Dimensionen wächst diese Untergrenze wie $d/\sigma^2$, wenn $\sigma$ abnimmt (:numref:`sec_mdl-denoising-score-matching`). Ein Entrauschungsverlust, der sich deutlich über null einpendelt, deutet also noch nicht auf ein schlechtes Netz hin.

<!--en:2db2f752-->
Das zweite Problem betrifft vor allem Bilder. Man nimmt an, dass sie nahe einer niedrigdimensionalen Menge im Pixelraum liegen :cite:`Bengio.Courville.Vincent.2013`. Betrachten wir den Extremfall, dass die Daten auf einer Geraden in der Ebene liegen. Rauschen verbreitert die Gerade zu einem Band. Quer zum Band ist $p_\sigma$ gaußverteilt mit Varianz $\sigma^2$. Die Score-Komponente quer zum Band beträgt daher $-\tilde{x}_2/\sigma^2$ im vorzeichenbehafteten Abstand $\tilde{x}_2$ von der Geraden (Übung :exref:`ex_denoising-data-on-a-line`). Ihre Steigung $-1/\sigma^2$ wächst für $\sigma \to 0$ dem Betrag nach unbegrenzt. Quer zum Band stimmt die Zielgröße $-\epsilon_2/\sigma$ exakt mit dieser Score-Komponente überein. Hier liegt die Schwierigkeit also woanders. Das Netz muss quer zu den Daten ein immer steiler abfallendes Feld darstellen, während die Zielgrößen längs der Geraden überwiegend aus Rauschen bestehen.

<!--en:1b0d2c5e-->
Bei hohem Rauschpegel stellt sich das umgekehrte Problem. Die Regression ist gut konditioniert, liefert jedoch den Score von $p_\sigma$ anstelle des Scores von $p$. Für eine gaußsche Komponente $\mathcal{N}(\boldsymbol\mu, s^2\mathbf{I})$ in $d$ Dimensionen lauten die beiden Scores $-(\mathbf{x} - \boldsymbol\mu)/(s^2 + \sigma^2)$ und $-(\mathbf{x} - \boldsymbol\mu)/s^2$. Mitteln wir die quadrierte Norm ihrer Differenz über $p$, erhalten wir die Verzerrung (engl. bias)

<!--en:fddbf6a0-->
Die Verzerrung wächst zunächst wie $\sigma^4$. Für $s = 0.5$ und $d = 2$ beträgt sie $0.0004$ bei $\sigma = 0.05$, $0.28$ bei $\sigma = 0.3$ und $2.6$ bei $\sigma = 1$. Für die Mischung mit weit getrennten Komponenten ergeben sich nahezu dieselben Werte. Teilen wir durch die Größe des unverrauschten Scores, $\tfrac12 \mathbb{E}_p\|\nabla \log p\|^2 = d/(2s^2)$, so erhalten wir $(\sigma^2/(s^2 + \sigma^2))^2$. Dieser Ausdruck hängt nur von $\sigma/s$ ab. Im quadratischen Mittel weicht der geglättete Score um weniger als $10\%$ vom unverrauschten ab, solange $\sigma < s/3 \approx 0.17$ gilt. Etwa ab dieser Schwelle dürfte die Verzerrung ins Gewicht fallen.

<!--en:17233bb2-->
Zwischen diesen beiden Extremen sollte es einen optimalen Pegel geben. Wir trainieren bei sieben Pegeln von $0.05$ bis $1$ je ein Netz mit zwei verschiedenen Zufallsstartwerten. An 20 000 Testpunkten schätzen wir für jedes Netz die Fisher-Divergenz :eqref:`eq_diffusion-esm` zum unverrauschten Score und vergleichen sie mit der Verzerrung des exakten geglätteten Scores.

<!--en:1bb1199e-->
Der Fehler des gelernten Scores beschreibt eine U-Kurve. Rechts folgt er der Kurve der Verzerrung, mit einer Abweichung von höchstens etwa $0.1$ ab $\sigma = 0.5$. Die Netze lernen die geglätteten Zielgrößen gut, doch diese weichen vom gesuchten Score ab. Links steigt der Fehler trotz verschwindender Verzerrung wieder an, weil die Zielgrößen überwiegend aus Rauschen bestehen. Der zweite Zufallsstartwert verändert einige Fehler dort um die Hälfte oder mehr. Der beste Pegel liegt zwischen $\sigma = 0.1$ und $0.2$, also bei einem bis zwei Fünfteln der Komponentenbreite und nahe dem aus der Verzerrung abgeleiteten Richtwert $s/3$. Der Fehler beträgt dort etwa $0.15$ bis $0.25$. Für einen einzelnen Pegel gilt also die Faustregel: Wir messen $\sigma$ relativ zur charakteristischen Breite der Daten und wählen den kleinsten Pegel, bei dem das Netz den Score noch lernen kann.

<!--en:787bf668-->
### Unempfindlichkeit gegenüber den Modusgewichten

<!--en:da5fd3bc-->
Ein Fehler von $0.2$ in der Fisher-Divergenz klingt klein. Doch woran messen wir ihn? Vergleichen wir unser Beispiel mit derselben Mischung bei gleichen Gewichten. Mit diesen falschen Gewichten büßt das Modell pro Punkt $0.07$ Nat an Log-Likelihood ein. Das ist die Kullback-Leibler-Divergenz der beiden Mischungen. Der Score unterscheidet sich vom wahren Score jedoch nur durch die Zugehörigkeitswahrscheinlichkeiten in :eqref:`eq_diffusion-mixture-score`. Diese weichen nur in den Tälern voneinander ab, wo $p$ nahezu null ist. Die Fisher-Divergenz mittelt über $p$ und erfasst den Unterschied daher kaum.

<!--en:2e897aa0-->
Wir berechnen beide Divergenzen durch numerische Quadratur auf einem Gitter mit $721 \times 721$ Punkten. Sampling wäre hier ungünstig, weil der Integrand der Fisher-Divergenz vor allem in den Tälern beiträgt, in die nur wenige Monte-Carlo-Samples fallen. Anschließend wiederholen wir den Vergleich, nachdem wir beide Mischungen mit demselben Rauschen geglättet haben.

<!--en:2c45aac2-->
Zwischen den unverrauschten Mischungen beträgt die Fisher-Divergenz etwa $10^{-5}$. Das liegt vier Größenordnungen unter den Fehlern der trainierten Netze, die alle mindestens $0.1$ betragen. Die KL-Divergenz beträgt dagegen $0.07$ Nat. Ein Score-Modell, das wir nahe an den unverrauschten Daten trainieren, kann die Gewichte daher nicht lernen. Ein Feld mit falschen Gewichten passt ebenso gut wie eines mit den richtigen Gewichten (:numref:`sec_mdl-fisher-divergence` und Übung :exref:`ex_mdl-divergences-distances-score-blindness` in :numref:`sec_mdl-scores`). Der Score kann die Daten lokal hervorragend beschreiben, ohne etwas darüber zu verraten, wie häufig jeder Modus auftritt.

<!--en:70fc4225-->
Rauschen macht die Gewichte sichtbar. Mit wachsendem $\sigma$ breiten sich die Komponenten in die Täler aus (:numref:`fig_diffusion-noise-ladder`). Die Zugehörigkeitswahrscheinlichkeiten unterscheiden sich nun auch dort, wo $p_\sigma$ nennenswerte Masse hat. Die Fisher-Divergenz steigt auf etwa $10^{-3}$ bei $\sigma = 0.5$ und $10^{-2}$ bei $\sigma = 1$, während sich die KL-Divergenz kaum ändert. Selbst bei $\sigma = 1$ ist die Fisher-Divergenz noch um mehr als den Faktor zehn kleiner als jeder Fehler in unserer Versuchsreihe. Allein durch Rauschen lassen sich die Gewichte also noch nicht leicht lernen. Zunächst bewirkt es, dass die Ketten die Täler durchqueren können.

<!--en:2e457a0d-->
![Unser durchgängiges Beispiel nach Glättung mit gaußschem Rauschen bei wachsender Standardabweichung $\sigma$. Kreuze markieren die Mittelwerte der Komponenten. Mit wachsendem $\sigma$ füllen sich die Täler zwischen den Modi, bis bei $\sigma = 2$ nur noch eine breite Erhebung übrig ist.](../img/mdl-diffusion-noise-ladder.svg)

<!--en:55a61c67-->
### Sampling mit einem Score

<!--en:e7291aa6-->
Um Samples zu ziehen, genügt der Score. Das einfachste Verfahren dafür ist über ein Jahrhundert alt. Paul Langevin formulierte 1908 eine Gleichung dieser Art für ein Teilchen in einer Flüssigkeit. Auf das Teilchen wirkt eine Kraft, und die umgebenden Moleküle versetzen ihm zufällige Stöße. In diskreter Form beginnt die *Langevin-Dynamik* an einem beliebigen Punkt und wiederholt den Schritt

<!--en:f0f7975e-->
mit einer kleinen Schrittweite $h$. Die Drift bewegt den Punkt in Richtung steigender Log-Dichte. Das Rauschen verhindert, dass er an einem Modus hängen bleibt. Im Grenzfall kontinuierlicher Zeit ist $p$ die stationäre Verteilung, sofern $\mathbf{s} = \nabla \log p$ gilt (:numref:`sec_mdl-langevin-sampling`). Läuft die Kette lange genug, vergisst sie ihren Startpunkt. Der diskrete Schritt weicht davon etwas ab. Für eine Gauß-Verteilung mit Varianz $v$ beträgt seine stationäre Varianz $v/(1 - h/(4v))$. Für unsere Komponenten sind das bei $h = 0.05$ fünf Prozent zu viel. In der Aktualisierung kommt keine Dichte vor. Daher kann jedes Vektorfeld die Schritte bestimmen, auch ein gelerntes Feld, das kein Gradientenfeld ist.

<!--en:e6993ce0-->
Wir starten 2000 Ketten aus $\mathcal{N}(\mathbf{0}, 9\mathbf{I})$. Damit erhält jeder Modus etwa ein Drittel der Ketten. Wir führen 2000 Schritte bei $h = 0.05$ mit fünf Feldern aus. Wir verwenden den exakten Score von $p$ sowie die exakten Scores von $p_\sigma$ bei $\sigma = 1$ und $\sigma = 0.3$, die in der Ausgabe als `p_1` und `p_0.3` erscheinen. Hinzu kommen die beiden bei $\sigma = 0.3$ trainierten Netze aus der Versuchsreihe, deren Ausgaben wir wie den Score von $p$ verwenden. Das exakte Feld bei $0.3$ war das Lernziel der Netze. An ihm erkennen wir, welche Eigenschaften auf den Rauschpegel zurückgehen und welche auf die Netze selbst.

<!--en:3b3a0aa3-->
Sollten die Ketten mit dem exakten Score von $p$ am Ende die Gewichte $0.5$, $0.3$ und $0.2$ liefern? Ja, sofern 2000 Schritte ausreichen, damit eine Kette ihren Startpunkt vergisst. Die Ketten bei $\sigma = 1$ sollten Samples aus $p_\sigma$ ziehen. Dessen Komponenten haben je Koordinate die Varianz $s^2 + \sigma^2 = 1.25$, das Fünffache der Komponentenvarianz von $p$. Setzen wir in der Rechnung für den perfekten Sampler $2 \cdot 1.25$ anstelle von $2s^2$ ein, erwarten wir einen Ausreißeranteil von $e^{-2.25/2.5} \approx 41\%$ und eine Streuung nahe $1$. Die Streuung ist also nur etwa doppelt so groß wie in den Daten.

<!--en:c8a5226a-->
Der exakte Score von $p$ führt jede Kette in einen Modus mit der richtigen lokalen Streuung. Die Modusgewichte bleiben jedoch bei ihren Anfangswerten (jeweils $\approx 1/3$). In $-\log p$ liegen die Täler zwischen den Modi 10 bis 12 Nat höher als die Modi. Nach Kramers liegt die erwartete Entweichzeit daher in der Größenordnung von $10^6$ Schritten bei $h = 0.05$ (Übung :exref:`ex_denoising-escaping-mode`). In 2000 Schritten können sich die Ketten nicht über die Modi hinweg mischen. Bei $\sigma = 1$ hingegen füllt das Rauschen die Täler. Die Ketten können sich mischen und die wahren Gewichte annähernd wiedergeben ($0.5, 0.3, 0.2$). Sie ziehen ihre Samples jedoch aus der breiten Verteilung $p_1$ mit Komponentenvarianz $s^2 + \sigma^2 = 1.25$. Das ergibt einen Ausreißeranteil von $\approx 41\%$ und eine Streuung nahe $1$.

<!--en:da97d00d-->
Die bei $\sigma = 0.3$ trainierten Netze nähern den Score von $p_{0.3}$ an (Komponentenvarianz $0.34$, Ausreißeranteil $\approx 4\%$ und Streuung $\approx 0.6$). Mit dem exakten Score von $p_{0.3}$ überwinden die Ketten die Barrieren von 7 bis 9 Nat in 2000 Schritten kaum. Das Gewicht des stärksten Modus ändert sich um höchstens $0.03$. Die trainierten Netze verschieben mehr Masse zwischen den Modi, weil ihre Näherungsfehler in den dünn besetzten Tälern künstliche Driftpfade erzeugen.

<!--en:f7f7c739-->
Ein einzelner Rauschpegel erzwingt also einen grundlegenden Kompromiss. Bei kleinem $\sigma$ bleiben die Modi scharf, doch die Ketten stecken fest und die Fisher-Divergenz ist unempfindlich gegenüber den Modusgewichten. Bei großem $\sigma$ mischen sich die Ketten rasch über die Modi hinweg, ziehen ihre Samples aber aus einer stark geglätteten Dichte. In :numref:`sec_diffusion-gaussian-paths` lösen wir dieses Problem, indem wir den Rauschpegel schrittweise von grob bis fein absenken.

<!--en:8b205eab-->
## Entrauschung von Bildern

<!--en:204e861a-->
Welche Ergebnisse gelten auch jenseits der Ebene? Der Entrauschungsverlust hängt nicht von der Dimension der Daten ab und lässt sich unverändert auf Bilder anwenden. Was uns fehlt, ist der Vergleichsmaßstab. Den A-posteriori-Mittelwert der Bilder können wir nicht berechnen. Daher vergleichen wir das Netz mit dem besten berechenbaren linearen Entrauscher. Wir verwenden Fashion-MNIST (:numref:`sec_fashion_mnist`) mit Pixelwerten im Intervall $[-1, 1]$. Zwei kleine Hilfsfunktionen laden die so skalierten Bilder und zeigen sie an. Das Rauschen hat die Standardabweichung $\sigma = 0.5$. Das Netz verwendet vier $3 \times 3$-Faltungen mit 32 Kanälen und SiLU-Aktivierungen. Wir trainieren es mit derselben Schleife wie das Score-Netz für 600 Schritte. Jeder Minibatch umfasst 128 Bilder mit jeweils neu gezogenem Rauschen. Wir trainieren das Netz mit zwei Zufallsstartwerten, sodass wir jedes Ergebnis mit einem zweiten Lauf vergleichen können. Es sagt das Rauschen $\boldsymbol\epsilon$ vorher. Nach :eqref:`eq_diffusion-three-predictions` ergibt seine Ausgabe geteilt durch $-\sigma$ einen Score der verrauschten Bilder in $784$ Dimensionen, ohne dass wir je eine Dichte angeben mussten. Das entrauschte Bild ist $\tilde{\mathbf{x}} - \sigma\hat{\boldsymbol\epsilon}$.

<!--en:e276f188-->
Mit diesen Hilfsfunktionen laden wir die auf $[-1, 1]$ skalierten Bilder aus Fashion-MNIST und definieren die Zielfunktion `noise_loss` für die Rauschvorhersage bei $\sigma = 0.5$. Mit `d2l.train_regression` trainieren wir den vierschichtigen konvolutionären Entrauscher für je 600 Schritte mit zwei Zufallsstartwerten.

<!--en:54cf8736-->
Der Verlust misst den quadratischen Fehler der Rauschvorhersage je Pixel und pendelt sich bei beiden Startwerten nahe $0.12$ ein, also weit über null. Multipliziert mit $\sigma^2 = 0.25$ ergibt er den quadratischen Fehler des entrauschten Bildes, etwa $0.03$ je Pixel. Der folgende Test bestätigt das an Bildern, die das Netz noch nicht gesehen hat.

<!--en:5d429226-->
Als Referenz dient der beste *lineare* Entrauscher. Wären die Bilder gaußverteilt mit Mittelwert $\boldsymbol\mu$ und Kovarianz $\boldsymbol\Sigma$, ergäbe Tweedies Formel die oben für den gaußschen Fall hergeleitete Schrumpfung. Für jeden Eigenvektor von $\boldsymbol\Sigma$ gilt dabei ein eigener Faktor $\lambda/(\lambda + \sigma^2)$:

<!--en:c8bc8e0b-->
Dieses *Wiener-Filter* erhält die Richtungen, in denen die Bilder stärker variieren als das Rauschen, und dämpft die übrigen. Es ist der beste affine Entrauscher für jede Verteilung mit diesem Mittelwert und dieser Kovarianz (Übung :exref:`ex_denoising-linear-denoiser`). Wir schätzen $\boldsymbol\mu$ und $\boldsymbol\Sigma$ anhand der 60 000 Trainingsbilder. An 1000 Testbildern vergleichen wir den quadratischen Fehler je Pixel für die verrauschte Eingabe, das Wiener-Filter und beide Netze. Neben jedem Mittelwert gibt die Codezelle den Standardfehler über die Bilder aus. Die Zeilen der Abbildung zeigen die unverrauschten Bilder, ihre verrauschten Varianten, die Ausgabe des Filters und die des ersten Netzes.

<!--en:2296cf05-->
Die verrauschte Eingabe hat einen quadratischen Fehler je Pixel von $\sigma^2 = 0.25$. Das optimale lineare Filter (Wiener-Filter) erreicht $\approx 0.045$, beide vierschichtigen konvolutionären Netze erreichen $\approx 0.031$. Ihr Fehler ist um $30\%$ geringer, der Abstand beträgt mehr als 20 Standardfehler. Das Wiener-Filter zieht jedes Bild zu einem einzigen globalen linearen Unterraum hin. Kanten bleiben unscharf, der Hintergrund bleibt körnig. Das konvolutionäre Netz projiziert die verrauschten Punkte dagegen in Richtung der nichtlinearen Mannigfaltigkeit der Kleidungsbilder (siehe Übung :exref:`ex_denoising-data-on-a-line`). Der A-posteriori-Mittelwert mittelt jedoch über alle unverrauschten Bilder, die mit $\tilde{\mathbf{x}}$ vereinbar sind. Wo das Rauschen feine, hochfrequente Strukturen zerstört hat, bleibt daher ein glatter Mittelwert. Um aus dem Entrauscher einen Generator zu machen, der die Bildschärfe erhält, müssen wir über die Rauschpegel integrieren (:numref:`sec_diffusion-gaussian-paths` und :numref:`sec_diffusion-integration`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:bc779829-->
Minimieren wir den quadratischen Entrauschungsfehler $\mathbb{E}\|D(\tilde{\mathbf{x}}) - \mathbf{x}_0\|^2$ für $\tilde{\mathbf{x}} = \mathbf{x}_0 + \sigma\boldsymbol\epsilon$, erhalten wir den A-posteriori-Mittelwert $\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}]$. Nach Vincents Satz und Tweedies Formel gilt $\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}] = \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$. Die Vorhersagen des unverrauschten Punkts $\hat{\mathbf{x}}_0$, des Rauschens $\hat{\boldsymbol\epsilon}$ und des Scores $\hat{\mathbf{s}} = -\hat{\boldsymbol\epsilon}/\sigma$ sind somit affine Umparametrisierungen desselben bedingten Erwartungswerts. Wenn wir den Score lernen, umgehen wir sowohl die schwer berechenbare Normierungskonstante $Z_{\boldsymbol\theta}$ energiebasierter Modelle als auch die Spur der Jacobi-Matrix $\nabla \cdot \mathbf{s}$ beim expliziten Score-Matching. Ein einzelner Rauschpegel $\sigma$ erzwingt jedoch einen deutlichen Kompromiss. Bei kleinem $\sigma$ haben die Zielgrößen eine hohe Varianz. Nahe niedrigdimensionalen Datenmannigfaltigkeiten sind die Gradienten steil, die Fisher-Divergenz ist unempfindlich gegenüber den Modusgewichten und die Langevin-Ketten mischen sich nicht. Bei großem $\sigma$ mischen sich die Ketten rasch und geben die Modusgewichte wieder, ziehen die Samples aber nur aus der stark geglätteten Dichte $p_\sigma$.

<!--en:2294243a-->
## Übungen

<!--en:1ad13b34-->
1. **Lernen ohne Normierung.** Betrachten Sie das eindimensionale Modell $\tilde{p}(x) = \exp(-E(x))$ mit $E(x) = (x - \mu)^2 / (2 \tau^2)$. Passen Sie es an Daten $x_1, \ldots, x_n$ an, die mindestens zwei verschiedene Werte annehmen.
    1. Maximieren Sie $\sum_i \log \tilde{p}(x_i)$ über $\mu$ und $\tau$ ohne Normierung. Wie sieht $\tilde{p}$ abseits der Daten aus, wenn sich die Zielfunktion ihrem Supremum nähert?
    1. Wiederholen Sie die Rechnung mit dem normierten Modell $\tilde{p} / Z$. Welcher Term der Zielfunktion verhindert das Problem aus der ersten Teilaufgabe? Bestimmen Sie den Maximierer.
    1. Ein Netz zur Energieberechnung endet meist mit einer linearen Schicht mit Ausgabe-Bias $c$. Zeigen Sie, dass die normierte Dichte :eqref:`eq_diffusion-ebm` unabhängig von $c$ ist. Wie verändert Gradientenaufstieg allein auf $-\sum_i E_{\boldsymbol{\theta}}(\mathbf{x}_i)$ den Wert von $c$?
   :exlabel:`ex_denoising-learning-without-normalization`

<!--en:6e8f9333-->
1. [code] **Schätzung der Normierungskonstanten.** Die Normierungskonstante $Z = \int e^{-E(\mathbf{x})}\, d\mathbf{x}$ einer Energie lässt sich direkt mit Importance-Sampling (:numref:`sec_mdl-bayes-importance`) berechnen. Ziehen Sie $\mathbf{x}_1, \ldots, \mathbf{x}_n$ aus einer Vorschlagsverteilung $q$ und mitteln Sie die Gewichte $w_i = e^{-E(\mathbf{x}_i)} / q(\mathbf{x}_i)$, deren Erwartungswert $Z$ ist. Wählen Sie $E(\mathbf{x}) = \|\mathbf{x} - \mathbf{m}\|^2 / 2$. Die Zielverteilung ist dann bis auf die Normierung $\mathcal{N}(\mathbf{m}, \mathbf{I})$. Verwenden Sie als Vorschlagsverteilung $q = \mathcal{N}(\mathbf{0}, \mathbf{I})$.
    1. Leiten Sie $w / Z = \exp(\mathbf{m}^\top \mathbf{x} - \|\mathbf{m}\|^2 / 2)$ her und zeigen Sie mit der momenterzeugenden Funktion einer Gauß-Verteilung, dass $\mathbb{E}_q[w / Z] = 1$ und $\mathbb{E}_q[(w / Z)^2] = \exp(\|\mathbf{m}\|^2)$ gelten. Folgern Sie, dass der relative Standardfehler der Schätzung aus $n$ Samples gleich $\sqrt{(e^{\|\mathbf{m}\|^2} - 1) / n}$ ist. Wie viele Samples ergeben einen relativen Fehler von $0.1$, wenn $\|\mathbf{m}\|^2 = 20$ gilt?
    1. Sei $\mathbf{m}$ der Einsvektor in $d$ Dimensionen. Die Vorschlagsverteilung weicht also in jeder Koordinate nur um eine Standardabweichung ab. Schätzen Sie mit $n = 10^5$ die Konstante $Z$ für $d \in \{2, 4, 8, 16, 32\}$. Geben Sie das Verhältnis des Schätzwerts zum wahren Wert $Z = (2\pi)^{d/2}$ sowie die *effektive Stichprobengröße* $(\sum_i w_i)^2 / \sum_i w_i^2$ an. Vergleichen Sie beide mit den Vorhersagen der ersten Teilaufgabe (die effektive Stichprobengröße sollte nahe $n\, e^{-\|\mathbf{m}\|^2}$ liegen). Weshalb übersteigt sie diese Vorhersage bei großem $d$? Was bedeutet das für ihren Einsatz als Diagnosegröße?
    1. Ein Bild hat Hunderttausende von Koordinaten. Was folgt aus Ihren Ergebnissen für die Schätzung der Normierungskonstanten eines energiebasierten Bildmodells und für sein Training mit Maximum-Likelihood?
   :exlabel:`ex_denoising-log-normal-weight`

<!--en:da1e0253-->
1. **Daten auf einer Geraden.** Die Daten liegen auf der ersten Koordinatenachse von $\mathbb{R}^2$. Es gilt $\mathbf{x}_0 = (u, 0)$ mit $u \sim q$ für eine glatte positive Dichte $q$ auf $\mathbb{R}$, sodass $p$ auf $\mathbb{R}^2$ keine Dichte besitzt.
    1. Zeigen Sie, dass $p_\sigma(\tilde{\mathbf{x}}) = q_\sigma(\tilde{x}_1)\, \mathcal{N}(\tilde{x}_2;\, 0, \sigma^2)$ gilt. Dabei entsteht $q_\sigma$ aus $q$ durch Glättung mit $\mathcal{N}(0, \sigma^2)$. Geben Sie den Score $\nabla \log p_\sigma$ an.
    1. Zeigen Sie, dass das Quadrat der zweiten Score-Komponente den Erwartungswert $1 / \sigma^2$ unter $p_\sigma$ hat und ihre Steigung $-1 / \sigma^2$ in $\tilde{x}_2$ beträgt. Wie verhalten sich beide Größen für $\sigma \to 0$? Was bedeutet das für ein Netz, das den Score bei niedrigem Rauschpegel darstellen muss?
    1. Wenden Sie Tweedies Formel :eqref:`eq_diffusion-tweedie` an. Wie verändert der optimale Entrauscher die einzelnen Koordinaten von $\tilde{\mathbf{x}}$? Welcher geometrischen Operation entspricht das?
    1. Die Daten liegen nun auf einem $k$-dimensionalen linearen Unterraum von $\mathbb{R}^d$. Berechnen Sie den Erwartungswert der quadrierten Norm der Score-Komponente senkrecht zum Unterraum.
   :exlabel:`ex_denoising-data-on-a-line`

<!--en:4a6fae78-->
1. **Zweipunktverteilung und Untergrenze des Verlusts.** Gegeben sei $p = \tfrac12 \delta_{-1} + \tfrac12 \delta_{+1}$ in einer Dimension mit Rauschen des Pegels $\sigma$.
    1. Berechnen Sie den A-posteriori-Mittelwert $\mathbb{E}[x_0 \mid \tilde{x}]$ mit der Bayes-Regel. Zeigen Sie, dass er gleich $\tilde{x} + \sigma^2 \frac{d}{d\tilde{x}} \log p_\sigma(\tilde{x})$ ist, wie Tweedies Formel :eqref:`eq_diffusion-tweedie` verlangt.
    1. Welchen Wert gibt der optimale Entrauscher für $\tilde{x} = 0$ zurück? Weshalb ist das bei quadratischem Fehler die beste Antwort, obwohl dieser Wert bei unverrauschten Daten unmöglich ist?
    1. Schreiben Sie das Minimum von :eqref:`eq_diffusion-dsm` als $\frac{1}{2 \sigma^4}\, \mathbb{E}\big[\operatorname{tr} \textrm{Cov}(\mathbf{x}_0 \mid \tilde{\mathbf{x}})\big]$. Zeigen Sie, dass es für jede Datenverteilung positiv ist, die nicht auf einen einzigen Punkt konzentriert ist.
    1. Schreiben Sie dieses Minimum für die Zweipunktverteilung als eindimensionales Integral über $\tilde{x}$. Werten Sie es für $\sigma \in \{0.25, 0.5, 1, 2\}$ durch Quadratur aus. Multiplizieren Sie jeden Wert mit $2\sigma^2$, um ihn in Einheiten des Verlusts der Rauschvorhersage auszudrücken. Erklären Sie, weshalb diese skalierte Untergrenze sowohl für $\sigma \to 0$ als auch für $\sigma \to \infty$ verschwindet.
   :exlabel:`ex_denoising-regression-identity`

<!--en:440462d9-->
1. [code] **Der beste lineare Entrauscher.**
    1. Zeigen Sie, dass das Wiener-Filter :eqref:`eq_diffusion-wiener` den Ausdruck $\mathbb{E}\|D(\tilde{\mathbf{x}}) - \mathbf{x}_0\|^2$ unter allen affinen Abbildungen $D$ minimiert, und zwar für jede Datenverteilung mit Mittelwert $\boldsymbol{\mu}$ und Kovarianz $\boldsymbol{\Sigma}$. Zeigen Sie außerdem, dass sein quadratischer Fehler je Pixel $\frac{1}{d} \sum_i \sigma^2 \lambda_i / (\lambda_i + \sigma^2)$ beträgt. Dabei sind $\lambda_1, \ldots, \lambda_d$ die Eigenwerte von $\boldsymbol{\Sigma}$.
    1. Berechnen Sie diese Vorhersage bei $\sigma = 0.5$ aus `Sigma` und vergleichen Sie sie mit dem ausgegebenen Fehler von `x_lin`.
    1. Zählen Sie die Eigenwerte von `Sigma` oberhalb der Schwelle $\sigma^2$ für $\sigma \in \{0.25, 0.5, 1\}$ und stellen Sie einige der zugehörigen Eigenvektoren als Bilder dar. Was erhält das Filter beim jeweiligen Rauschpegel?
   :exlabel:`ex_denoising-linear-denoiser`

<!--en:10b2f440-->
1. [code] **Entweichen aus einem Modus.** ● Modellieren Sie das Tal zwischen den beiden unteren Modi unseres Beispiels in einer Dimension als gleich gewichtete Mischung aus $\mathcal{N}(-2.5, 0.25)$ und $\mathcal{N}(2.5, 0.25)$. Das Potential sei $U = -\log p$, die Potentialmulden liegen bei $a = \pm 2.5$ und die Barriere bei $b = 0$. Ist die Höhe der Barriere $\Delta U = U(b) - U(a)$ groß gegenüber eins, liefert die Kramers-Formel :cite:`Kramers.1940` die erwartete Zeit, die der Diffusionsprozess $dX = -\tfrac12 U'(X)\, dt + dW$ zum Übergang von einer Mulde in die andere benötigt. Dieser Prozess ist der Grenzfall von :eqref:`eq_diffusion-langevin` für kontinuierliche Zeit. Es gilt

   $$
   \tau \approx \frac{4 \pi}{\sqrt{U''(a)\, |U''(b)|}}\, e^{\Delta U}.
   $$

    1. Berechnen Sie $\Delta U$, $U''(a)$ und $U''(b)$. Rechnen Sie $\tau$ für die Schrittweite $h = 0.05$ in die Anzahl der Langevin-Schritte um.
    1. Modus 3 unseres Beispiels liegt etwa $4.7$ von Modus 1 (und von Modus 2) entfernt. Dieser Abstand ist kleiner als der Abstand $5$ zwischen den beiden unteren Modi. Werten Sie $U = -\log p$ unseres Beispiels entlang der Strecke zwischen den beiden betreffenden Mittelwerten aus. Lesen Sie die Höhe der Barriere ab und schätzen Sie die Übergangszeit wie zuvor. Wie viele von 2000 Ketten sollten in 2000 Schritten von Modus 1 zu Modus 3 wechseln?
    1. Wiederholen Sie die erste Teilaufgabe für $p_\sigma$ bei $\sigma = 0.3$ und $\sigma = 1$. Verstehen Sie das Ergebnis bei kleinem $\Delta U$ als Größenordnung und erklären Sie damit die Gewichte der Ketten bei $\sigma = 1$. Wie viele von 2000 Ketten sollten bei $\sigma = 0.3$ in 2000 Schritten die Barriere überwinden? Stimmt die ausgegebene Zeile für `p_0.3` damit überein? Die bei $\sigma = 0.3$ trainierten Netze verschieben mindestens doppelt so viel Masse. Was sagt das über das gelernte Feld zwischen den Modi aus?
   :exlabel:`ex_denoising-escaping-mode`

<!--en:772a15d3-->
[Diskussion](https://d2l.discourse.group/t/36818)

<!--en:71093565-->
Entrauschung und Score-Funktion<br>**A-posteriori-Mittelwert · Denoising-Score-Matching · Tweedies Formel · Unempfindlichkeit der Fisher-Divergenz gegenüber Modusgewichten · Langevin-Dynamik**

<!--en:3ef19b18-->
::: {.slide title="Wie die Entrauschung zur generativen Modellierung führt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:1120a81f-->
Lernt ein Netz, gaußsches Rauschen $\tilde{\mathbf{x}} = \mathbf{x}_0 + \sigma\boldsymbol\epsilon$ nach der Methode der kleinsten Quadrate zu entfernen, erfasst es die lokale Geometrie der Daten:

<!--en:e12220ce-->
- **Projektion auf eine nichtlineare Mannigfaltigkeit:** Bei Rauschpegel $\sigma = 0.5$ ($\textrm{MSE} = 0.25$ je Pixel) senkt ein konvolutionäres Netz mit 4 Schichten den Testfehler auf Fashion-MNIST auf **$0.031$**. Damit unterbietet es das optimale lineare $784 \times 784$-Wiener-Filter ($0.045$) um $30\%$.

<!--en:78c0e6f9-->
- **Entrauschung $\iff$ Score-Schätzung:** Nach Tweedies Formel ist der optimale Entrauscher im Sinne der kleinsten Quadrate $\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}]$ gleich $\tilde{\mathbf{x}} + \sigma^2\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$. Die Korrektur gegenüber $\tilde{\mathbf{x}}$ beträgt also das $\sigma^2$-Fache des Scores.

<!--en:351dd1ce-->
- **Die Frage bei festem Pegel:** Ein Entrauscher lernt implizit das Score-Feld $\nabla \log p_\sigma$. Können wir neue Bilder erzeugen, indem wir beim Rauschen beginnen und diesem Feld bergauf folgen?

<!--en:e6bc2db5-->
::: {.slide title="Das Score-Feld zeigt in Richtung steigender Log-Dichte"}

<!--en:6f665dee-->
[Unser Referenzbeispiel: ein zweidimensionales Gauß-Mischmodell]{.kicker}

<!--en:e9f668a6-->
Um gelernte Entrauscher und Sampler mit exakten Referenzwerten zu vergleichen, verwenden wir ein Gauß-Mischmodell mit drei Modi in $\mathbb{R}^2$. Die Mittelwerte sind $(-2.5, -1.5)$, $(2.5, -1.5)$ und $(0, 2.5)$, die Gewichte $\pi_k = (0.5, 0.3, 0.2)$ und die Breite $s = 0.5$:

<!--en:3dd2d4c6-->
- **Exaktes Score-Feld:** Der Score $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ gewichtet die rücktreibende Kraft jedes Modus mit seiner A-posteriori-Zugehörigkeitswahrscheinlichkeit $r_k(\mathbf{x}) = \pi_k \mathcal{N}(\mathbf{x}; \boldsymbol\mu_k, s^2\mathbf{I}) / p(\mathbf{x})$:
  $$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \sum_{k=1}^3 r_k(\mathbf{x})\,\frac{\boldsymbol\mu_k - \mathbf{x}}{s^2}$$

<!--en:286adea1-->
- **Gaußsche Faltung in geschlossener Form:** Isotropes Rauschen $\sigma\boldsymbol\epsilon$ erhöht die Varianz jeder Komponente von $s^2$ auf $s^2 + \sigma^2$.

<!--en:ed58121c-->
- **Wo die Modusgewichte wirken:** Nahe Modus $k$ gilt $r_k(\mathbf{x}) \approx 1$ unabhängig von $\pi_k$. Die Mischgewichte verändern den Score *nur* in den dünn besetzten Tälern zwischen den Modi.

<!--en:e9608744-->
::: {.slide title="Kennzahlen für die Mischung mit drei Modi" layout="tight"}

<!--en:56ead355-->
[Bewertung zweidimensionaler Sampler]{.kicker}

<!--en:1fa2f4e7-->
- **Modusgewichte (`weights`):** Anteil der Punkte je Modus bei Zuordnung zum nächsten Mittelwert. Ein kalibrierter Sampler trifft $[0.500, 0.300, 0.200]$ bis auf den Binomialfehler ($\pm 0.011$).

<!--en:c632a9b6-->
- **Ausreißeranteil (`stray`):** Anteil der Punkte mit einem Abstand von mehr als $3s = 1.5$ zu jedem Moduszentrum. Bei einer zweidimensionalen Gauß-Verteilung ist der quadrierte Radius exponentialverteilt. Der Anteil jenseits der Schwelle beträgt $e^{-4.5} \approx 1.1\%$.

<!--en:c42a7a46-->
- **Streuung innerhalb der Modi (`spread`):** Mittlerer quadratischer Abstand der Punkte ohne Ausreißer zum nächsten Mittelwert ($\approx 2s^2 = 0.50$, höchstens $2.25$).

<!--en:fac4bb31-->
- **Negative Log-Likelihood (`NLL`):** Die wahre Entropie der Mischung beträgt $\approx 2.47$ Nat je Punkt. Das ist der Zielwert, denn eine Konzentration auf die Moduszentren senkt die NLL künstlich.

<!--en:7c8fa86d-->
In jedem zweidimensionalen Experiment dieses Kapitels vergleichen wir die erzeugten Punktmengen mit vier Referenzwerten aus $2{,}000$ Ziehungen:

<!--en:728256bb-->
::: {.slide title="Entrauschung nach der Methode der kleinsten Quadrate liefert den A-posteriori-Mittelwert"}

<!--en:45f2af35-->
[Optimale Entrauschung]{.kicker}

<!--en:037f21fa-->
- **Orthogonale Zerlegung:** Bedingt auf $\tilde{\mathbf{x}} = \mathbf{x}_0 + \sigma\boldsymbol\epsilon$ verschwindet der Kreuzterm nach dem Gesetz des iterierten Erwartungswerts.

<!--en:398c6b50-->
- **Bayes-optimaler Entrauscher:** Für jeden Punkt $\tilde{\mathbf{x}}$ hat der quadratische Fehler genau einen Minimierer, den **A-posteriori-Mittelwert** $D^\star(\tilde{\mathbf{x}}) = \mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}]$.

<!--en:44a0d5b6-->
::: {.slide title="Denoising-Score-Matching und Vincents Identität"}

<!--en:6424cbc4-->
[Vom bedingten zum marginalen Score]{.kicker}

<!--en:17998d6f-->
- **Bekannter bedingter Score:** Für $p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_0, \sigma^2\mathbf{I})$ lautet der bedingte Score $\nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}_0) = (\mathbf{x}_0 - \tilde{\mathbf{x}})/\sigma^2 = -\boldsymbol\epsilon/\sigma$.

<!--en:43f62f46-->
- **Vincents Satz (2011):** Differenzieren von $p_\sigma(\tilde{\mathbf{x}}) = \int p(\mathbf{x}_0)p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}_0)\,d\mathbf{x}_0$ ergibt $\nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}}) = \mathbb{E}[-\boldsymbol\epsilon/\sigma \mid \tilde{\mathbf{x}}]$. Durch Regression auf $-\boldsymbol\epsilon/\sigma$ erhalten wir also exakt $\nabla\log p_\sigma$.

<!--en:ea77baab-->
::: {.slide title="Tweedies Formel und die Umrechnung zwischen den Vorhersagen"}

<!--en:1f70213d-->
[Äquivalenz von Entrauscher, Score und Rauschvorhersage]{.kicker}

<!--en:2bbae4ac-->
Der bedingte Score $(\mathbf{x}_0 - \tilde{\mathbf{x}})/\sigma^2$ ist eine affine Funktion von $\mathbf{x}_0$ bei festem $\tilde{\mathbf{x}}$. Bilden wir die bedingten Erwartungswerte, ergibt sich **Tweedies Formel**:

<!--en:6be1e3a0-->
- **Geometrische Deutung:** Der optimale Entrauscher macht einen deterministischen Schritt der Schrittweite $\sigma^2$ in Richtung des Scores der geglätteten Dichte $p_\sigma$.

<!--en:03869b06-->
- **Drei gleichwertige Parametrisierungen:** Jeder trainierte Prädiktor bestimmt unmittelbar die beiden anderen:
  $$\hat{\mathbf{x}}_0(\tilde{\mathbf{x}}) = \tilde{\mathbf{x}} + \sigma^2 \hat{\mathbf{s}}(\tilde{\mathbf{x}}) = \tilde{\mathbf{x}} - \sigma\hat{\boldsymbol\epsilon}(\tilde{\mathbf{x}})$$

<!--en:ee2459a9-->
::: {.slide title="Rechenbeispiel zur Schrumpfung bei gaußverteilten Daten"}

<!--en:a112197e-->
[Tweedies Formel in geschlossener Form]{.kicker}

<!--en:c3812658-->
- **Schrumpfung nach James und Stein bzw. Wiener:** Für $\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol\mu, v\mathbf{I})$ verkleinert Tweedies Schritt die verrauschten Abweichungen vom Moduszentrum $\boldsymbol\mu$ um den Faktor $v/(v + \sigma^2)$, das Verhältnis von Signalvarianz zu Gesamtvarianz.

<!--en:fc440a5e-->
- **Quantitativer Test ($v = s^2 = 0.25, \sigma = 0.3$):** Der Schrumpfungsfaktor beträgt $0.25/0.34 = \mathbf{0.735}$. Der mittlere quadratische Fehler je zweidimensionalem Punkt sinkt um $27\%$ von $2\sigma^2 = 0.180$ auf $0.132$.

<!--en:f2561cf5-->
::: {.slide title="Überprüfung von Tweedies Schritt mit einem trainierten Score-MLP"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:336b9541-->
Wir trainieren ein MLP mit 2 versteckten Schichten (`TimeMLP(128, 2)`) auf $J_{\mathrm{DSM}}$ bei $\sigma = 0.3$ für $2{,}000$ Adam-Schritte und prüfen Tweedies Formel an $20{,}000$ neuen Testpunkten:

<!--en:ab343583-->
**Verborgene Größe des Score-Fehlers:** Der mittlere quadratische Entrauschungsfehler übersteigt den des exakten A-posteriori-Mittelwerts um $\mathbb{E}\|\sigma^2(\hat{\mathbf{s}} - \nabla \log p_\sigma)\|^2 = \sigma^4\,\mathbb{E}\|\hat{\mathbf{s}} - \nabla \log p_\sigma\|^2$. Da $\sigma^4 = 0.0081$ gilt, verbirgt eine geringe Fehlerdifferenz von $0.001$ bei der Entrauschung einen **Score-Fehler von $0.06$ in der Fisher-Divergenz**.

<!--en:a5894893-->
::: {.slide title="Warum den Score anstelle einer energiebasierten Dichte lernen?"}

<!--en:14024a29-->
[Umgehung der Partitionsfunktion]{.kicker}

<!--en:e8c1f945-->
Ein energiebasiertes Modell (EBM) definiert eine Dichte $p_{\boldsymbol\theta}(\mathbf{x}) = \exp(-E_{\boldsymbol\theta}(\mathbf{x})) / Z_{\boldsymbol\theta}$ über ein Netz mit skalarer Ausgabe $E_{\boldsymbol\theta}(\mathbf{x})$:

<!--en:9488f6b3-->
- **Schwer berechenbare Normierungskonstante $Z_{\boldsymbol\theta}$:** Damit die gesamte Wahrscheinlichkeitsmasse $1$ beträgt, müssen wir über ganz $\mathbb{R}^d$ integrieren:
  $$Z_{\boldsymbol\theta} = \int \exp(-E_{\boldsymbol\theta}(\mathbf{x}))\,d\mathbf{x}$$

<!--en:6e5b0f41-->
- **Aufwendige Likelihood-Gradienten:** Zum Maximieren von $\log p_{\boldsymbol\theta}(\mathbf{x})$ brauchen wir $\nabla_{\boldsymbol\theta} \log Z_{\boldsymbol\theta} = -\mathbb{E}_{p_{\boldsymbol\theta}}[\nabla_{\boldsymbol\theta} E_{\boldsymbol\theta}]$. Das erfordert bei jedem Gradientenschritt aufwendiges MCMC-Sampling.

<!--en:0617697d-->
- **Invarianz des Scores:** Differenzieren wir $\log p_{\boldsymbol\theta}(\mathbf{x})$ nach $\mathbf{x}$, fällt $\log Z_{\boldsymbol\theta}$ vollständig weg. Es gilt $\nabla_{\mathbf{x}} \log p_{\boldsymbol\theta}(\mathbf{x}) = -\nabla_{\mathbf{x}} E_{\boldsymbol\theta}(\mathbf{x})$.

<!--en:0eefbf5a-->
::: {.slide title="Drei Zielfunktionen für Score-Matching im Vergleich"}

<!--en:38e40ada-->
[Von der Fisher-Divergenz zur Entrauschung]{.kicker}

<!--en:7f0c7fa7-->
- **Exakte Fisher-Divergenz $D_{\mathrm{F}}(p, \mathbf{s}_{\boldsymbol\theta}) = \frac{1}{2}\mathbb{E}_p[\|\mathbf{s}_{\boldsymbol\theta} - \nabla\log p\|^2]$:** Die ideale Zielfunktion lässt sich nicht berechnen, da der wahre Score der Daten $\nabla\log p(\mathbf{x})$ unbekannt ist.

<!--en:2cc56934-->
- **Explizites Score-Matching (Hyvärinen, 2005):** Partielle Integration eliminiert $\nabla\log p$. Die Divergenz $\nabla \cdot \mathbf{s}_{\boldsymbol\theta}$ ist jedoch die Spur der Jacobi-Matrix und erfordert $O(d)$ Rückwärtsdurchläufe.

<!--en:8c9e7cba-->
- **Denoising-Score-Matching (Vincent, 2011):** Ein schneller Regressionsdurchlauf auf das Rauschen kostet $O(1)$ in jeder Dimension $d$. Die Spur der Jacobi-Matrix entfällt, dafür entsteht eine über den Pegel $\sigma$ steuerbare Verzerrung durch Glättung.

<!--en:ce3974ee-->
::: {.slide title="Varianz und Verzerrung bei einem einzigen Rauschpegel"}

<!--en:2f10aa11-->
[Statistischer Zielkonflikt bei der Wahl von σ]{.kicker}

<!--en:d3c3c6ee-->
- **Kleines $\sigma$ (unbeschränkte Varianz der Zielgröße):** Die bedingte Zielgröße $-\boldsymbol\epsilon/\sigma$ hat die Varianz $d/\sigma^2 \to \infty$ für $\sigma \to 0$. Die Varianz des Schätzers wächst, und die Ketten bleiben in lokalen Modi gefangen.

<!--en:2adad3df-->
- **Großes $\sigma$ (Verzerrung durch Glättung):** Die Regression ist gut konditioniert, doch $\nabla\log p_\sigma$ unterschätzt den scharfen unverrauschten Score $\nabla\log p$. Die Verzerrung steigt von $0.0004$ bei $\sigma = 0.05$ auf $2.6$ bei $\sigma = 1.0$.

<!--en:2d197043-->
::: {.slide title="Problematische Gradienten nahe niedrigdimensionalen Mannigfaltigkeiten"}

<!--en:fa49430c-->
[Geometrische Schwierigkeiten bei kleinem σ]{.kicker}

<!--en:d6ef6ace-->
- **Richtungsabhängige Verbreiterung durch gaußsches Rauschen:** Liegen die Daten auf der Geraden $x_2 = 0$ in $\mathbb{R}^2$, ist die tangentiale Score-Komponente $\partial_{\tilde{x}_1}\log q_\sigma$ von der Ordnung $O(1)$. Die Normalkomponente beträgt $-\tilde{x}_2/\sigma^2$.

<!--en:6b2f9ba6-->
- **Steiler Abfall quer zur Mannigfaltigkeit:** Für $\sigma \to 0$ muss das Netz in Normalenrichtung einen unendlich steilen Abfall (Steigung $-1/\sigma^2$) darstellen und gleichzeitig tangentiales Rauschen der Ordnung $O(1/\sigma)$ herausmitteln.

<!--en:3d178259-->
::: {.slide title="Der U-förmige Verlauf des Score-Schätzfehlers in Abhängigkeit von σ" layout="tight"}

<!--en:9c65169a-->
[Empirischer Vergleich von Verzerrung und Varianz]{.kicker}

<!--en:a7878d09-->
Wir trainieren `TimeMLP`-Modelle bei sieben Rauschpegeln $\sigma \in [0.05, 1.0]$ und bestimmen ihre Fisher-Divergenz zum unverrauschten Score $\nabla \log p$. Es ergibt sich die typische U-Kurve:

<!--en:8d45cc92-->
- **Verzerrung dominiert ($\sigma \ge 0.5$):** In beiden Trainingsläufen mit unterschiedlichen Zufallsstartwerten weicht der Fehler um höchstens etwa $0.1$ von der theoretischen Kurve für die Verzerrung durch Glättung ab. Das Netz lernt $\nabla \log p_\sigma$ genau, doch es gilt $p_\sigma \neq p$.

<!--en:c4305295-->
- **Varianz dominiert ($\sigma < 0.1$):** Die Verzerrung durch Glättung verschwindet. Das Rauschen der Zielgröße von der Ordnung $1/\sigma$ lässt den Testfehler jedoch wieder stark steigen und zwischen den Startwerten deutlich schwanken.

<!--en:faa58119-->
- **Günstigster Einzelpegel:** Der Fehler erreicht sein Minimum nahe $\sigma \approx s/3 \in [0.1, 0.2]$ (Fehler etwa $0.15$ bis $0.25$).

<!--en:060fd76a-->
::: {.slide title="Warum die Fisher-Divergenz die Modusgewichte kaum erfasst"}

<!--en:6c3b1909-->
[Strukturelle Grenze lokaler Scores]{.kicker}

<!--en:9e937740-->
Wir vergleichen die wahre Mischung $\pi = (0.5, 0.3, 0.2)$ mit einer gleich gewichteten Mischung $\pi_{\textrm{eq}} = (1/3, 1/3, 1/3)$ durch zweidimensionale numerische Quadratur auf einem $721 \times 721$-Gitter:

<!--en:7cee7047-->
- **Vier Größenordnungen unter dem Fehler des Netzes:** Bei $\sigma = 0$ beträgt die KL-Divergenz $0.069$ Nat, die Fisher-Divergenz jedoch nur **$1.5 \times 10^{-5}$** (gegenüber $\ge 0.15$ für unser bestes trainiertes MLP).

<!--en:d594db06-->
- **Grund:** Nahe jedem Modus $k$ ist $\nabla \log p(\mathbf{x}) \approx (\boldsymbol\mu_k - \mathbf{x})/s^2$ unabhängig von $\pi_k$. Die Scores unterscheiden sich nur in den leeren Tälern, wo $p(\mathbf{x}) \approx 0$ gilt.

<!--en:3bd2d2ca-->
::: {.slide title="Langevin-Dynamik für das Sampling allein mit einem Score-Feld"}

<!--en:bf0c0444-->
[Score-basiertes MCMC]{.kicker}

<!--en:8a10a542-->
- **Stationäre Verteilung und Einfluss der Schrittweite:** Für $h \to 0$ und $\mathbf{s} = \nabla\log p$ hält die aufwärts gerichtete Drift dem brownschen Rauschen die Waage. Dadurch bleibt $p(\mathbf{x})$ invariant. Eine endliche Schrittweite $h$ erhöht die Varianz auf $v/(1 - h/4v)$.

<!--en:9f75ce68-->
- **Barrieren nach Kramers:** Um ein Tal der Tiefe $\Delta U \approx 10\text{--}12$ Nat zwischen den Modi zu durchqueren, braucht eine Kette im Mittel $\tau \propto e^{\Delta U} \approx 10^6$ Schritte.

<!--en:58752f33-->
::: {.slide title="Langevin-Ketten auf exakten und gelernten Score-Feldern"}

<!--en:044dfbc9-->
[Empirischer Test der Durchmischung]{.kicker}

<!--en:d876b96e-->
Wir starten $2{,}000$ Ketten aus $\mathcal{N}(\mathbf{0}, 9\mathbf{I})$ ($\approx 1/3$ je Modus) und führen auf fünf Score-Feldern je $2{,}000$ Langevin-Schritte aus ($h = 0.05$):

<!--en:1e99b2d4-->
::: {.slide title="Warum Sampling bei festem Rauschpegel scheitert"}

<!--en:2767482d-->
[Durchmischung und Schärfe]{.kicker}

<!--en:75dc3015-->
- **Exakter unverrauschter Score ($\nabla\log p$):** Die Modi bleiben scharf ($\text{spread} \approx 0.50, \text{strays} = 1\text{--}2\%$), doch Täler mit einer Tiefe von $10\text{--}12$ Nat halten die Ketten in ihren anfänglichen Anteilen von je etwa einem Drittel gefangen.

<!--en:c5a1b13a-->
- **Stark geglätteter Score ($\nabla\log p_{1.0}$):** Die Täler füllen sich, und die Gewichte nähern sich den wahren Werten $[0.5, 0.3, 0.2]$. Die Streuung innerhalb der Modi verdoppelt sich jedoch auf $\approx 0.97$ ($\text{strays} = 41\text{--}42\%$).

<!--en:bb254374-->
- **Gelernte MLP-Scores ($\sigma = 0.3$):** Barrieren von $7\text{--}9$ Nat hemmen weiterhin die Durchmischung. Die Drift zwischen den Modi entsteht vermutlich durch Näherungsfehler in den leeren Tälern.

<!--en:88f5b598-->
::: {.slide title="Entrauschung in hohen Dimensionen mit konvolutionären Netzen und Wiener-Filter"}

<!--en:3f92be53-->
[Anwendung auf Kleidungsbilder in 784 Dimensionen]{.kicker}

<!--en:ebefc0d0-->
- **Linearer Unterraum und gekrümmte Mannigfaltigkeit ($\sigma = 0.5$ auf Fashion-MNIST):** Das optimale $784 \times 784$-Wiener-Filter schrumpft jede PCA-Komponente um den Faktor $\lambda_i/(\lambda_i + \sigma^2)$ ($\textrm{MSE} = 0.045$). Unser konvolutionäres Netz mit 4 Schichten projiziert auf die nichtlineare Mannigfaltigkeit der Kleidungsbilder ($\textrm{MSE} \approx 0.031$, um mehr als 20 Standardfehler niedriger).

<!--en:eaa50c97-->
- **Grenzen der Generierung durch einen Entrauschungsschritt:** Bei starkem Rauschen mittelt $\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}]$ über alle passenden Kleidungsstücke. Das Ergebnis ist unscharf.

<!--en:44a7f100-->
::: {.slide title="Verständnisfragen zu Tweedies Formel und den Rauschpegeln"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:8e1c2299-->
Überprüfen Sie Ihre Intuition anhand dreier grundlegender Aufgaben zum Score-Matching:

<!--en:0768e674-->
1. **Vom Rauschen zum Score und zum unverrauschten Punkt:** Bei Rauschpegel $\sigma = 0.5$ sagt ein Netz das Rauschen $\hat{\boldsymbol\epsilon} = (0.6, -0.3)$ für die verrauschte Eingabe $\tilde{\mathbf{x}} = (1.0, 1.0)$ vorher. Welcher Score $\hat{\mathbf{s}}(\tilde{\mathbf{x}})$ und welcher nach Tweedies Formel entrauschte Schätzwert $\hat{\mathbf{x}}_0(\tilde{\mathbf{x}})$ folgen daraus?

<!--en:0fa7d9f6-->
2. **Verzerrung durch Glättung berechnen:** Wie groß ist für eine zweidimensionale gaußsche Komponente der Breite $s = 0.5$ ($d = 2$) die analytische Verzerrung gemäß Fisher-Divergenz $\frac{1}{2}\mathbb{E}_p\|\nabla \log p_\sigma - \nabla \log p\|^2$ bei $\sigma = 0.5$? Welchem Grenzwert nähert sie sich für $\sigma \to \infty$?

<!--en:2c1fc5a8-->
3. **Zielkonflikt bei festem Pegel:** Weshalb kann kein einzelner Rauschpegel $\sigma$ bei der Langevin-Dynamik zugleich genaue globale Modusgewichte und lokal scharf konzentrierte Samples liefern?

<!--en:557b98b6-->
**Antworten:**

<!--en:df93b0f2-->
1. **Score:** $\hat{\mathbf{s}} = -\hat{\boldsymbol\epsilon}/\sigma = \mathbf{(-1.2, 0.6)}$. **Entrauschter Punkt:** $\hat{\mathbf{x}}_0 = \tilde{\mathbf{x}} - \sigma\hat{\boldsymbol\epsilon} = \tilde{\mathbf{x}} + \sigma^2\hat{\mathbf{s}} = (1, 1) - (0.3, -0.15) = \mathbf{(0.7, 1.15)}$.

<!--en:4e961506-->
2. **Verzerrung bei $\sigma = s = 0.5$:** $\frac{d\sigma^4}{2s^2(s^2+\sigma^2)^2} = \frac{2(0.0625)}{2(0.25)(0.5)^2} = \mathbf{1.0}$. Für $\sigma \to \infty$ gilt $\nabla \log p_\sigma \to \mathbf{0}$. Die Verzerrung nähert sich daher der Hälfte der mittleren quadrierten Norm des unverrauschten Scores, $d/(2s^2) = \mathbf{4.0}$.

<!--en:2dd8e481-->
3. Bei kleinem $\sigma$ trennen Täler geringer Dichte die Modi. Die Divergenz $D_{\mathrm{F}}$ erfasst die Gewichte kaum, und nach Kramers dauert das Entweichen $\sim 10^6$ Schritte. Bei großem $\sigma$ füllen sich die Täler und die Ketten mischen sich. Im stationären Zustand folgen die Samples jedoch der breiten Dichte $p_\sigma$.

<!--en:017ce887-->
::: {.slide title="Zusammenfassung zu Entrauschung, Scores und den Grenzen eines einzelnen Rauschpegels"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:09fa3c65-->
| Begriff oder Verfahren | Mathematische Form | Verhalten und Aufwand | Rolle in generativen Diffusionsmodellen |

<!--en:70b09eb8-->
| **Optimaler Entrauscher bei quadratischem Fehler** | $D^\star(\tilde{\mathbf{x}}) = \mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}]$ | A-posteriori-Mittel über unverrauschte Punkte | Ersetzt das adversarielle Spiel durch direkte Regression |

<!--en:d2f73785-->
| **Denoising-Score-Matching** | $\frac{1}{2}\mathbb{E}\big[\|\mathbf{s}_{\boldsymbol\theta}(\mathbf{x}_0 + \sigma\boldsymbol\epsilon) + \boldsymbol\epsilon/\sigma\|^2\big]$ | Rückwärtsdurchlauf mit Aufwand $O(1)$, ohne $Z_{\boldsymbol\theta}$ oder Spur der Jacobi-Matrix | Lernt den marginalen Score der geglätteten Dichte $\nabla \log p_\sigma(\tilde{\mathbf{x}})$ |

<!--en:259e8aa8-->
| **Tweedies Formel** | $\hat{\mathbf{x}}_0 = \tilde{\mathbf{x}} + \sigma^2\hat{\mathbf{s}} = \tilde{\mathbf{x}} - \sigma\hat{\boldsymbol\epsilon}$ | Affine Abbildung verknüpft $\hat{\mathbf{x}}_0$, $\hat{\mathbf{s}}$ und $\hat{\boldsymbol\epsilon}$ | Vereint Vorhersagen für unverrauschte Daten, Rauschen und Score |

<!--en:d309a924-->
| **Kleines $\sigma$** | Zielgröße $-\boldsymbol\epsilon/\sigma$, Steigung in Normalenrichtung $-1/\sigma^2$ | Untergrenze des Verlusts $\propto d/\sigma^2$, Entweichzeit nach Kramers $\propto e^{\Delta U}$ | Lokal scharfe Modi, aber Ketten stecken fest und verfehlen die Gewichte |

<!--en:96111313-->
| **Großes $\sigma$** | Geglättete Dichte $p_\sigma = p * \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ | Verzerrung des Scores $\propto \sigma^4$, Modusvarianz $s^2 + \sigma^2$ | Rasche globale Durchmischung, aber stark geglättete Samples |

<!--en:17cab15e-->
| **Ausblick auf die Pfade in §{sec_diffusion-gaussian-paths}** | $\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\boldsymbol\epsilon$ ($t \in [0, 1]$) | Schrittweise von reinem Rauschen ($t=1$) zu Daten ($t=0$) | Verbindet Durchmischung auf grober Skala mit Schärfe auf feiner Skala |
