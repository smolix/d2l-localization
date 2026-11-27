<!--en:83fb8e63-->
# Umgebung und Verteilungsverschiebung

<!--en:54e53241-->
Alle Garantien in :numref:`chap_classification_generalization` beruhten auf einer stillschweigenden Annahme. Die Daten im Einsatz stammen aus genau derselben Verteilung wie unser Trainings- und unser Testdatensatz. In der Praxis ist diese Annahme immer wieder verletzt. Scanner und Kameras werden modernisiert, saisonale Grippewellen verändern über Nacht die Krankheitshäufigkeit, und regionale Ausdrücke wandeln sich. Schlimmer noch, der Einsatz eines Modells kann *selbst* dazu führen, dass Menschen ihr Verhalten ändern. Angenommen, eine Bank stellt anhand früherer Kredite fest, dass Personen in Oxford-Schuhen ihre Kredite häufiger zurückzahlen als Personen in Turnschuhen. Sie setzt einen Klassifikator ein, der Anträge bei Oxford-Schuhen bewilligt und bei Turnschuhen ablehnt. Innerhalb weniger Wochen erscheinen alle in Oxford-Schuhen, obwohl die zugrunde liegende Ausfallquote gleich bleibt. Das ist ein Lehrbuchbeispiel für das *Gesetz von Goodhart*. Wird ein statistisch korreliertes Merkmal zur Zielgröße für Entscheidungen, verliert es seine Aussagekraft.

<!--en:ef31390f-->
In diesem Abschnitt untersuchen wir, wie sich eine Verteilungsverschiebung (engl. distribution shift) zwischen zwei Verteilungen erkennen und korrigieren lässt. Aus der *Quellverteilung* (engl. source distribution) $p_S(\mathbf{x}, y)$ haben wir Trainingsdaten mit Labels. Aus der *Zielverteilung* (engl. target distribution) $p_T(\mathbf{x}, y)$ beobachten wir im Einsatz nur Eingaben $\mathbf{x}$ ohne Label. Wir unterscheiden drei klassische Formen der passiven Verschiebung danach, welcher Faktor der gemeinsamen Verteilung invariant bleibt. Wir leiten die *Korrektur der Kovariatenverschiebung durch Wichtigkeitsgewichtung* und die *Korrektur der Labelverteilungsverschiebung mit der Konfusionsmatrix* her und implementieren beide Verfahren. Zum Schluss betrachten wir Konzeptverschiebungen (engl. concept shift) und Rückkopplungsschleifen (engl. feedback loops).

<!--en:97a1eb6d-->
## Formen der Verteilungsverschiebung

<!--en:5723e1f6-->
Ohne *irgendeine* strukturelle Annahme über den Zusammenhang zwischen $p_S(\mathbf{x}, y)$ und $p_T(\mathbf{x}, y)$ ist eine Anpassung an eine Zieldomäne ohne Label mathematisch unmöglich. Angenommen, es gilt weiterhin $p_T(\mathbf{x}) = p_S(\mathbf{x})$, aber die Labels für Katzen und Hunde werden über Nacht unbemerkt vertauscht ($p_T(y = \textrm{cat} \mid \mathbf{x}) = p_S(y = \textrm{dog} \mid \mathbf{x})$). Kein Algorithmus kann den Tausch erkennen, wenn er nur Zielbilder ohne Label sieht.

<!--en:0a5eae36-->
Jede gemeinsame Verteilung lässt sich auf zwei Arten in Faktoren zerlegen:

<!--en:de79d1b2-->
Je nachdem, ob die bedingte Verteilung oder die Randverteilung fest bleibt und welche sich verschiebt, unterscheiden wir drei klassische Formen der Verteilungsverschiebung (:numref:`fig_mdl-clf-shift-types`):

<!--en:47945c8c-->
| Form der Verschiebung | Was sich ändert | Was fest bleibt | Typischer kausaler Zusammenhang |

<!--en:969d4dfb-->
| **Kovariatenverschiebung** | $p(\mathbf{x})$ | $p(y \mid \mathbf{x})$ | Merkmale $\mathbf{x}$ verursachen das Label $y$ |

<!--en:a0c7d4b4-->
| **Labelverteilungsverschiebung** | $p(y)$ | $p(\mathbf{x} \mid y)$ | Das Label $y$ verursacht die Merkmale $\mathbf{x}$ |

<!--en:beed80f0-->
| **Konzeptverschiebung** | $p(y \mid \mathbf{x})$ | häufig $p(\mathbf{x})$ | Die Definition des Labels ändert sich je nach Zeit oder Ort |

<!--en:a10b7c61-->
![Drei Formen der Verteilungsverschiebung für eine eindimensionale Eingabe $x$ und zwei Klassen (oben die Quelle $p_S$, unten das Ziel $p_T$, gepunktet die optimale Entscheidungsgrenze). Bei der Kovariatenverschiebung verschiebt sich $p(x)$, während $p(y \mid x)$ und die wahre Grenze fest bleiben. Bei der Labelverteilungsverschiebung ändern sich die A-priori-Wahrscheinlichkeiten der Klassen $p(y)$, während $p(x \mid y)$ fest bleibt. Dadurch wandert die Bayes-Grenze zur selteneren Klasse. Bei der Konzeptverschiebung ändert sich $p(y \mid x)$ direkt.](../img/mdl-clf-shift-types.svg)

<!--en:02b268f9-->
### Kovariatenverschiebung, Labelverteilungsverschiebung und Konzeptverschiebung

<!--en:14b9ab62-->
1. **Kovariatenverschiebung ($p(y \mid \mathbf{x})$ invariant):** Eine Kovariatenverschiebung (engl. covariate shift) liegt nahe, wenn $\mathbf{x}$ die Ursache von $y$ ist. Angenommen, wir trainieren einen Klassifikator zur Unterscheidung von Katzen und Hunden mit Fotos (:numref:`fig_cat-dog-train`) und setzen ihn auf Zeichnungen ein (:numref:`fig_cat-dog-test`). Eine gezeichnete Katze ist immer noch eine Katze ($p(y \mid \mathbf{x})$ bleibt unverändert). Die Randverteilung der Eingaben $p(\mathbf{x})$ hat sich aber in einen neuen Bereich des Pixelraums verschoben. Die wahre Bayes-Grenze hat sich nicht bewegt. Ein falsch spezifiziertes Modell oder eines mit begrenzter Kapazität erreicht nach dem Training mit den Fotos vor allem dort eine hohe Genauigkeit, wo $p_S(\mathbf{x})$ eine hohe Dichte hat. Dort, wo die Masse von $p_T(\mathbf{x})$ liegt, kann es völlig versagen.

<!--en:883103e9-->
2. **Labelverteilungsverschiebung ($p(\mathbf{x} \mid y)$ invariant):** Eine Labelverteilungsverschiebung (engl. label shift) liegt nahe, wenn $y$ die Ursache von $\mathbf{x}$ ist. Krankheiten verursachen zum Beispiel Symptome. Während einer Grippewelle steigt die Prävalenz $p(y)$ der Grippe stark an, während die Verteilung der Symptome $p(\mathbf{x} \mid y)$ bei Grippekranken unverändert bleibt. Nach dem Satz von Bayes gilt $p(y \mid \mathbf{x}) \propto p(\mathbf{x} \mid y)\,p(y)$. Verschiebt sich also die A-priori-Verteilung der Klassen, verschiebt sich auch die optimale Entscheidungsgrenze. Zur Korrektur müssen wir nur $q$ Verhältnisse von A-priori-Wahrscheinlichkeiten der Klassen schätzen. Ein hochdimensionales Dichteverhältnis über $\mathbf{x}$ brauchen wir dazu nicht.

<!--en:d27cd01c-->
3. **Konzeptverschiebung ($p(y \mid \mathbf{x})$ ändert sich):** Die Bedeutung des Labels selbst hängt vom Ort oder von der Zeit ab. Beispiele sind regionale Bezeichnungen für Erfrischungsgetränke in den Landkreisen der USA (:numref:`fig_popvssoda`), der Wandel diagnostischer Kriterien oder saisonale Änderungen der Kaufabsicht.

<!--en:9dd3d137-->
![Fotos aus der Quelldomäne für das Training zur Klassifikation von Katzen und Hunden (Bilder: Lafeez Hossain / 500px / Getty Images; ilkermetinkursova / iStock / Getty Images Plus; GlobalP / iStock / Getty Images Plus; Musthafa Aboobakuru / 500px / Getty Images).](../img/cat-dog-train.png)

<!--en:8426f49e-->
![Zeichnungen aus der Zieldomäne für die Klassifikation von Katzen und Hunden im Einsatz (Bilder: SIBAS_minich / iStock / Getty Images Plus; Ghrzuzudu / iStock / Getty Images Plus; id-work / DigitalVision Vectors / Getty Images; Yime / iStock / Getty Images Plus).](../img/cat-dog-test.png)

<!--en:77daf79b-->
![Geografische Konzeptverschiebung am Beispiel der häufigsten Bezeichnung für ein gesüßtes Getränk mit Kohlensäure („pop“, „soda“ oder „coke“) in den Landkreisen der USA (CC-BY: Alan McConchie, PopVsSoda.com).](../img/popvssoda.png)

<!--en:8bbdf031-->
## Korrektur der Kovariatenverschiebung

<!--en:2dbf84a4-->
### Wichtigkeitsgewichte

<!--en:7dbd49dc-->
Im Einsatz wollen wir das *Zielrisiko* (engl. target risk) $R_T[f] = \mathbb{E}_{(\mathbf{x}, y) \sim p_T}[l(f(\mathbf{x}), y)]$ minimieren. Bei einer Kovariatenverschiebung gilt $p_T(y \mid \mathbf{x}) = p_S(y \mid \mathbf{x}) = p(y \mid \mathbf{x})$. Der Träger der Quellverteilung muss den der Zielverteilung abdecken, das heißt, $p_S(\mathbf{x}) > 0$ gilt überall dort, wo $p_T(\mathbf{x}) > 0$ gilt. Multiplizieren und dividieren wir dann durch $p_S(\mathbf{x})$, erhalten wir das Zielrisiko als Erwartungswert bezüglich der *Quellverteilung* :cite:`Shimodaira.2000`:

<!--en:7e4604f3-->
Dabei ist $\beta(\mathbf{x}) = p_T(\mathbf{x}) / p_S(\mathbf{x})$ das *Wichtigkeitsgewicht* (engl. importance weight) der Eingabe $\mathbf{x}$. Nähern wir :eqref:`eq_covariate-shift-identity` anhand unserer $n$ Quellbeispiele mit Labels an, erhalten wir die *gewichtete empirische Risikominimierung* (engl. weighted empirical risk minimization):

<!--en:6c77d41e-->
Warum ist die *Überdeckung des Zielträgers* (engl. support overlap) unverzichtbar, also $p_S(\mathbf{x}) > 0$ überall dort, wo $p_T(\mathbf{x}) > 0$ gilt? Angenommen, ein medizinisches Start-up sammelt Blutproben von älteren Erkrankten, aber gesunde Kontrollproben ausschließlich von 20-jährigen Studierenden. Wo $p_S(\mathbf{x}) = 0$ gilt, kann keine Umgewichtung gesunde ältere Kontrollpersonen hervorbringen. Selbst bei überdecktem Zielträger können extreme Dichteverhältnisse am rechten Rand der Verteilung (:numref:`fig_mdl-clf-density-ratio`) dazu führen, dass wenige Punkte die Summe dominieren. Die *effektive Stichprobengröße* (engl. effective sample size) $n_{\textrm{eff}} = (\sum_i \beta_i)^2 / \sum_i \beta_i^2$ sinkt dann drastisch. Begrenzen wir die Gewichte gemäß $\min(\beta_i, c)$ nach oben, nehmen wir eine kleine Verzerrung in Kauf und erhalten dafür ein wesentlich größeres $n_{\textrm{eff}}$.

<!--en:9794efd0-->
![Quelldichte $p_S = \mathcal{N}(0, 1)$ und verschobene Zieldichte $p_T = \mathcal{N}(2, 1)$ (oben) sowie das daraus resultierende Wichtigkeitsgewicht $\beta(x_1) = p_T(x_1)/p_S(x_1) = e^{2x_1 - 2}$ auf logarithmischer Skala mit der Obergrenze $c = 5$ (unten).](../img/mdl-clf-density-ratio.svg)

<!--en:6d35b765-->
### Schätzung der Gewichte mit einem Domänenklassifikator

<!--en:35625807-->
Wie können wir $\beta(\mathbf{x}) = p_T(\mathbf{x}) / p_S(\mathbf{x})$ schätzen, ohne die hochdimensionalen Dichten $p_S$ und $p_T$ einzeln zu schätzen? Wir fassen die $n$ Quelleingaben (Domänenlabel $z = 0$) und $m$ Zieleingaben ohne Klassenlabel (Domänenlabel $z = 1$) zusammen. Mit binärer logistischer Regression trainieren wir einen *Domänenklassifikator* (engl. domain classifier) $P(z = 1 \mid \mathbf{x}) = \sigma(h(\mathbf{x}))$ mit Logit $h(\mathbf{x})$. Nach dem Satz von Bayes gilt:

<!--en:c38897ee-->
Der konstante Faktor $n/m$ skaliert die Zielfunktion, ohne ihren Minimierer zu verschieben. Wir lassen ihn weg und gewichten jedes Quellbeispiel mit $\beta_i = \exp(h(\mathbf{x}^{(i)}))$ (oder $\min(\exp(h(\mathbf{x}^{(i)})), c)$).

<!--en:d96af1fd-->
Der Domänenklassifikator zeigt uns auch, ob überhaupt eine Verschiebung vorliegt. Gilt $p_T(\mathbf{x}) = p_S(\mathbf{x})$, lassen sich Quell- und Zieleingaben nicht unterscheiden. Auf gemischten Eingaben, die beim Training unberücksichtigt blieben, erreicht der Domänenklassifikator nur Zufallsniveau ($1/2$ für $n = m$). Eine Genauigkeit deutlich über dem Zufallsniveau spricht also für eine Verschiebung der Eingabeverteilung. Dieses Verfahren heißt *klassifikatorbasierter Zweistichprobentest* (engl. classifier two-sample test) :cite:`Lopez-Paz.Oquab.2017`. Eine Labelverteilungsverschiebung lässt sich mit noch weniger Aufwand prüfen. Wir wenden den eingesetzten Klassifikator auf die Zieleingaben ohne Label an und vergleichen die Häufigkeit jeder vorhergesagten Klasse mit ihrer Häufigkeit auf den Validierungsdaten der Quelle. Ein System sollte im Einsatz beide Größen überwachen, denn erst wenn wir eine Verschiebung bemerken, können wir sie korrigieren.

<!--en:d6423074-->
### Ein zweidimensionales Beispiel

<!--en:75ff9b02-->
Betrachten wir ein zweidimensionales Problem, bei dem die Umgewichtung mit einem Domänenklassifikator einem linearen Klassifikator hilft. Es gelten $p_S = \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$ und $p_T = \mathcal{N}((2, 0)^\top, \mathbf{I}_2)$, und das wahre Label lautet $y = \mathbf{1}(x_2 > \frac{1}{2}x_1^2 - 1)$. Die wahre Entscheidungsgrenze ist eine nach oben geöffnete Parabel. Deshalb verläuft die beste lineare Grenze nahe $(0, 0)$ fast waagerecht und versagt nahe $(2, 0)$. Wir erzeugen $n = 1{,}000$ Quellpunkte und $1{,}000$ Zielpunkte. Wir passen den logistischen Domänenklassifikator $h(\mathbf{x})$ an, um $\beta_i = \exp(h(\mathbf{x}^{(i)}))$ zu schätzen. Dann vergleichen wir drei Varianten der logistischen Regression: ohne Gewichtung, mit Wichtigkeitsgewichten und mit nach oben begrenzten Gewichten ($c = 5$).

<!--en:acd3485d-->
Als einziges Lernverfahren verwenden wir die gewichtete logistische Regression und trainieren sie mit Gradientenabstieg. Nach :numref:`subsec_softmax_and_derivatives` ist der Gradient der Kreuzentropie bezüglich des Logits das Residuum $\sigma(\mathbf{w}^\top \mathbf{x} + b) - y$. Gewichten wir Beispiel $i$ mit $\beta_i$, multipliziert sich sein Residuum mit $\beta_i$.

<!--en:0bc3bb7a-->
Zunächst fassen wir `X_src` (Domänenlabel $z = 0$) und `X_tgt` ($z = 1$) zusammen. Damit trainieren wir unseren Domänenklassifikator `(w_h, b_h)` mithilfe von `fit_logreg`. Wir vergleichen das gelernte Logit $h(\mathbf{x})$ mit dem analytisch bestimmten logarithmischen Dichteverhältnis $2x_1 - 2$ der beiden Gauß-Verteilungen. Dann betrachten wir die daraus berechneten Wichtigkeitsgewichte `beta` der Quellbeispiele und ihre effektive Stichprobengröße `ess`. Die letzten Zeilen ziehen neue Quell- und Zieleingaben und messen, wie oft der Domänenklassifikator sie richtig zuordnet:

<!--en:48c2ce96-->
Auf neuen Eingaben unterscheidet der Domänenklassifikator Quelle und Ziel mit einer Genauigkeit von etwa $0.84$ gegenüber $0.50$ bei zufälliger Zuordnung. Diese Verschiebung ist also leicht zu erkennen. Kein Klassifikator kann hier besser abschneiden. Die beiden Gauß-Verteilungen unterscheiden sich nur im Mittelwert von $x_1$. Die beste Regel lautet $x_1 > 1$ und liegt mit Wahrscheinlichkeit $\Phi(1) \approx 0.84$ richtig. Dabei ist $\Phi$ die Verteilungsfunktion der Standardnormalverteilung.

<!--en:5a179b8b-->
Als Nächstes trainieren wir mit `fit_logreg` auf den Quelldaten mit Labels `(X_src, y_src)`. Wir verwenden drei Varianten: ohne Gewichte (`'unweighted'`), mit unveränderten Wichtigkeitsgewichten (`'weighted'`) und mit Gewichten (`'clipped'`), die wir bei $5$ begrenzen. Für jeden Klassifikator bestimmen wir die Genauigkeit auf `X_tgt` und `X_src`:

<!--en:4fdaaa71-->
Warum hilft die Umgewichtung so stark? Wir zeichnen die Punktwolken von Ziel und Quelle und skalieren die Fläche jedes Quellpunkts mit `beta`. Dazu zeichnen wir die wahre Parabel $x_2 = \frac{1}{2}x_1^2 - 1$ und die linearen Entscheidungsgrenzen der Varianten `'unweighted'` und `'weighted'`:

<!--en:4f073357-->
Auf der verschobenen Zielpunktwolke erreicht das Modell nach ungewichtetem Training nur $\sim 50\%$–$60\%$ Genauigkeit. Selbst die ständige Vorhersage der Mehrheitsklasse schneidet mit $\sim 70\%$ besser ab. Durch die Wichtigkeitsgewichtung neigt sich die lineare Entscheidungsgrenze entlang des rechten Parabelasts. So steigt die Zielgenauigkeit auf $\sim 95\%$, *ohne ein einziges* Ziellabel.

<!--en:e87f632d-->
Daraus folgt keineswegs, dass wir bei einer Kovariatenverschiebung immer umgewichten sollten. Hier hilft die Umgewichtung, weil eine Gerade einer Parabel nicht folgen kann. Das Modell ist falsch spezifiziert und muss sich entscheiden, wo es genau sein soll. Ein Modell gewinnt nichts, wenn es flexibel genug ist, $p(y \mid \mathbf{x})$ überall dort anzunähern, wo Quelldaten vorliegen. Die gewichtete und die ungewichtete Zielfunktion haben dann denselben Minimierer :cite:`Shimodaira.2000`. Tiefe Netze kommen diesem Fall nahe. Sobald sie im Training einen niedrigen Trainingsverlust erreichen, schwindet der Einfluss der Wichtigkeitsgewichte :cite:`Byrd.Lipton.2019`. Auf der Sammlung WILDS mit Verteilungsverschiebungen aus der Praxis übertreffen eigens dafür entwickelte Verfahren die gewöhnliche empirische Risikominimierung nur selten :cite:`Koh.Sagawa.Marklund.ea.2021`. In der Praxis begegnet man Kovariatenverschiebungen meist durch Training auf vielfältigeren Daten und durch Fine-Tuning mit wenigen Zielbeispielen mit Labels.

<!--en:7021f174-->
## Korrektur der Labelverteilungsverschiebung

<!--en:019d54c4-->
Bei einer Labelverteilungsverschiebung gilt $p_T(\mathbf{x} \mid y) = p_S(\mathbf{x} \mid y)$, während sich die Klassenanteile unterscheiden ($p_T(y) \neq p_S(y)$). Aus derselben Identität für die Wichtigkeitsgewichtung folgt, dass jedes Quellbeispiel der Klasse $y$ das Verhältnis der A-priori-Wahrscheinlichkeiten $\beta(y) = p_T(y) / p_S(y)$ als Gewicht erhält. Aus den Trainingslabeln kennen wir $p_S(y)$. Wir müssen also nur noch die Zielklassenanteile $\mathbf{p}_T \in \mathbb{R}^q$ aus Zieldaten *ohne Label* schätzen.

<!--en:b0619740-->
### Schätzung der Zielklassenanteile mit der Konfusionsmatrix

<!--en:4b05189b-->
Sei $\mathbf{C} \in \mathbb{R}^{q \times q}$ die spaltenweise normierte Konfusionsmatrix unseres Quellklassifikators aus :numref:`sec_softmax_scratch`. Ihre Einträge $C_{ij} = P_S(\hat{c} = i \mid y = j)$ schätzen wir anhand von Validierungsdaten der Quelle mit Labels. Da $\hat{c}$ allein von $\mathbf{x}$ abhängt und $p(\mathbf{x} \mid y)$ bei einer Labelverteilungsverschiebung invariant bleibt, gilt $P_T(\hat{c} = i \mid y = j) = C_{ij}$ auch in der Zieldomäne. Nach dem Satz der totalen Wahrscheinlichkeit erfüllt der Vektor $\boldsymbol{\mu} \in \mathbb{R}^q$ der vorhergesagten Klassenhäufigkeiten auf den Zieleingaben ohne Label die folgende Gleichung :cite:`Saerens.Latinne.Decaestecker.2002,Lipton.Wang.Smola.2018`:

<!--en:46921c63-->
Ist $\mathbf{C}$ invertierbar, können wir die unbekannten Zielklassenanteile ohne ein einziges Ziellabel bestimmen. Die Invertierbarkeit ist garantiert, wenn der Klassifikator in jeder Klasse mehr als die Hälfte der Beispiele richtig zuordnet, also $C_{jj} > 1/2$ gilt (siehe :exref:`ex_environment-and-distribution-shift-label-shift-system`). Wir lösen dazu das lineare Gleichungssystem der Größe $q \times q$ und erhalten $\hat{\mathbf{p}}_T = \mathbf{C}^{-1}\boldsymbol{\mu}$. Angenommen, die Wahrscheinlichkeiten des Quellklassifikators nähern die A-posteriori-Verteilung $p_S(y \mid \mathbf{x})$ der Quelle an. Das ist eine stärkere Anforderung als die Kalibrierung für die jeweils wahrscheinlichste Klasse aus :numref:`sec_classification`. Nach dem Satz von Bayes können wir die Vorhersagen dann sofort und ohne erneutes Training an die Zieldomäne anpassen. Dazu addieren wir die Differenz der logarithmierten A-priori-Wahrscheinlichkeiten $\log(\hat{p}_T(y) / p_S(y))$ zum Logit jeder Klasse.

<!--en:26d47f84-->
Erproben wir die Inversion der Konfusionsmatrix und die Logit-Anpassung nach Bayes an Fashion-MNIST. Zuerst trainieren wir `d2l.SoftmaxRegression` auf dem ausgeglichenen Trainingsdatensatz. Für die Validierungsdaten erfassen wir die Logits `O`, die Labels `y` und die vorhergesagten Klassen `pred = O.argmax(axis=1)`:

<!--en:6604117c-->
Nun teilen wir die $10{,}000$ Validierungsbeispiele zufällig auf. Mit $5{,}000$ Bildern bilden wir den Validierungsdatensatz `src` der Quelle und schätzen daraus die spaltenweise normierte $10 \times 10$-Konfusionsmatrix `C`. Die übrigen Bilder bilden den Vorrat `pool`. Daraus ziehen wir einen Zieldatensatz `tgt` mit ungleichen Klassenanteilen. Hemden machen rund $50\%$ aller Kleidungsstücke aus ($400$ Hemden und $45$ Stück aus jeder anderen Klasse). Dann lösen wir das System mit `p_hat = np.linalg.solve(C, mu)` anhand der Häufigkeiten `mu` der Vorhersagen auf den Zieldaten ohne Label:

<!--en:3840b3cf-->
Zum Schluss passen wir die Vorhersagen des Klassifikators an die geschätzte Zielverteilung an. Wir addieren die Differenz der logarithmierten A-priori-Wahrscheinlichkeiten `log_w = np.log(np.maximum(p_hat, 1e-6) / 0.1)` zu den Ziellogits `O[tgt]` und vergleichen die Zielgenauigkeit vor und nach der Anpassung:

<!--en:189e12b9-->
Der ursprüngliche Klassifikator sagt nur für $\sim 20\%$–$40\%$ der Zielbilder „Hemd“ vorher. Die übrigen Hemden verwechselt er mit ähnlich aussehenden Oberteilen. Trotzdem liefert $\mathbf{C}^{-1}\boldsymbol{\mu}$ einen geschätzten Hemdenanteil nahe dem wahren Wert von $0.50$. Meist beträgt die Abweichung wenige Hundertstel, gelegentlich fast ein Zehntel. Addieren wir $\log(\hat{\mathbf{p}}_T / 0.1)$ zu den Logits, steigt die Zielgenauigkeit um etwa 5 bis 15 Prozentpunkte, ohne ein einziges Ziellabel und ohne einen einzigen Gradientenschritt.

<!--en:2bef31d8-->
## Konzeptverschiebung und Rückkopplung

<!--en:d7ef2ea8-->
Wenn eine *Konzeptverschiebung* $p(y \mid \mathbf{x})$ selbst verändert, sind frühere Labels veraltet. Auch mit Wichtigkeitsgewichten lässt sich die neue Zuordnungsregel aus alten Daten nicht lernen. Wir müssen das Modell laufend anhand einiger neuer Beispiele mit aktuellen Labels überprüfen und es regelmäßig feinabstimmen oder neu trainieren.

<!--en:a75b302b-->
Noch mehr Sorgfalt ist geboten, wenn die Vorhersagen eines Modells *Handlungen auslösen, die künftige Trainingsdaten verändern* (*performative Vorhersage* (engl. performative prediction) und *Rückkopplungsschleifen*). Bei der vorausschauenden Polizeiarbeit (:numref:`fig_mdl-clf-feedback-loop`) schickt die Polizei zusätzliche Streifen in Viertel mit hoher vorhergesagter Kriminalität. Allein dadurch werden dort mehr Vorfälle registriert. Fließen diese verzerrten Festnahmedaten in die nächste Trainingsrunde ein, verstärkt sich die ursprüngliche Prognose immer weiter. Um solche Schleifen zu durchbrechen oder zu untersuchen, brauchen wir randomisierte Exploration oder unabhängig erhobene Daten. Damit gehen wir vom passiven überwachten Lernen zum *Reinforcement Learning* über (:numref:`chap_reinforcement_learning`).

<!--en:d7b0ea77-->
![Eine sich selbst verstärkende Rückkopplungsschleife in der vorausschauenden Polizeiarbeit. Die Modellvorhersagen steuern die Verteilung der Streifen. Deren Präsenz bestimmt, wo Vorfälle beobachtet werden. Die erfassten Vorfälle bilden den nächsten Trainingsdatensatz.](../img/mdl-clf-feedback-loop.svg)

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:40d9f9d1-->
Eine Anpassung an eine Einsatzumgebung ohne Label ist unmöglich, wenn keine strukturelle Invarianz die Quelle $p_S(\mathbf{x}, y)$ mit dem Ziel $p_T(\mathbf{x}, y)$ verbindet. Wir unterscheiden die Kovariatenverschiebung ($p(y \mid \mathbf{x})$ bleibt fest, $p(\mathbf{x})$ ändert sich), die Labelverteilungsverschiebung ($p(\mathbf{x} \mid y)$ bleibt fest, $p(y)$ ändert sich) und die Konzeptverschiebung ($p(y \mid \mathbf{x})$ ändert sich). Bei einer Kovariatenverschiebung mit überdecktem Zielträger richten wir falsch spezifizierte Modelle am Zielrisiko aus, indem wir die Verluste auf den Quelldaten mit $\beta(\mathbf{x}) = p_T(\mathbf{x})/p_S(\mathbf{x}) \propto \exp(h(\mathbf{x}))$ gewichten. Diese Gewichte schätzen wir mit einem logistischen Domänenklassifikator, der Quelleingaben von Zieleingaben ohne Klassenlabel unterscheidet. Bei einer Labelverteilungsverschiebung invertieren wir die Konfusionsmatrix der Quelle. Mit $\hat{\mathbf{p}}_T = \mathbf{C}^{-1}\boldsymbol{\mu}$ bestimmen wir die Zielklassenanteile aus den Vorhersagen für Zieldaten ohne Label. Kalibrierte Logits passen wir direkt an, indem wir $\log(\hat{\mathbf{p}}_T/\mathbf{p}_S)$ addieren. Verändern eine Konzeptverschiebung oder entscheidungsbedingte Rückkopplungsschleifen die Umgebung selbst, reicht passive Umgewichtung nicht aus. Systeme im Einsatz brauchen dann neue Ziellabels und unabhängige Exploration.

<!--en:2294243a-->
## Übungen

<!--en:efe39533-->
1. **Fehlende Trägerüberdeckung.** Angenommen, es gilt $p_S(\mathbf{x}) = 0$ in einem Bereich $\mathcal{A}$ des Eingaberaums, der unter der Zielverteilung die Wahrscheinlichkeit $\pi > 0$ hat. Überall sonst gilt $p_S(\mathbf{x}) > 0$. Definieren Sie $\beta(\mathbf{x}) = p_T(\mathbf{x})/p_S(\mathbf{x})$ außerhalb von $\mathcal{A}$.
    1. Zeigen Sie, dass das gewichtete Quellrisiko $\mathbb{E}_{(\mathbf{x}, y) \sim p_S}[\beta(\mathbf{x})\, l(f(\mathbf{x}), y)]$ gleich dem Zielrisiko abzüglich des Beitrags von $\mathcal{A}$ ist. Schreiben Sie den fehlenden Term als Produkt aus $\pi$ und einem bedingten Erwartungswert.
    1. Erklären Sie, warum keine Wahl der Gewichte für die Quellbeispiele den fehlenden Term rekonstruieren kann. Konstruieren Sie zwei Modelle, die außerhalb von $\mathcal{A}$ übereinstimmen und dasselbe gewichtete Quellrisiko, aber unterschiedliche Zielrisiken haben.

<!--en:56d1269d-->
1. [code] **Wenn die Umgewichtung versagt.** Der Code aus :numref:`subsec_covariate-shift-correction` (`fit_logreg`, `X_src`, `X_tgt`, `label`) setzt den Zielmittelwert auf $(2, 0)$. Variieren Sie den Zielmittelwert gemäß $(s, 0)$ für $s \in \{0, 0.5, 1, 2, 3, 4, 6\}$. Halten Sie $n = 1000$ und die Zuordnungsregel fest.
    1. Geben Sie für jedes $s$ die Genauigkeit des Domänenklassifikators $h$ auf zurückgehaltenen, zusammengefassten Quell- und Zieldaten an. Was sagt ihr Anstieg mit $s$ über die Erkennbarkeit der Verschiebung aus?
    1. Erfassen Sie für jedes $s$ die Zielgenauigkeit der ungewichteten und gewichteten Modelle sowie des Modells mit begrenzten Gewichten ($c = 5$). Erfassen Sie auch die Genauigkeit bei ständiger Vorhersage der häufigsten Zielklasse. Erklären Sie, warum die Zielgenauigkeit allein nicht zeigt, ob die Umgewichtung bei großem $s$ erfolgreich war.
   :exlabel:`ex_environment-and-distribution-shift-when-reweighting-fails`

<!--en:0c928452-->
1. **Das Gleichungssystem bei Labelverteilungsverschiebung.** Sei $\mathbf{C}$ die spaltenweise normierte Konfusionsmatrix aus :eqref:`eq_label-shift-system`.
    1. Zeigen Sie, dass $\mathbf{C}$ invertierbar ist, wenn $C_{jj} > \frac{1}{2}$ für jede Klasse $j$ gilt. Hinweis: Nehmen Sie an, dass $\mathbf{v}^\top \mathbf{C} = \mathbf{0}$ für ein $\mathbf{v} \neq \mathbf{0}$ gilt. Betrachten Sie die Spalte $j$, für die $|v_j|$ am größten ist, und nutzen Sie die Tatsache, dass ihre Einträge die Summe 1 haben.
    1. Geben Sie ein Beispiel für einen Klassifikator an, für den $\mathbf{C}$ singulär ist. Beschreiben Sie, was $\boldsymbol{\mu}$ in diesem Fall noch über $\mathbf{p}_T$ aussagt.
    1. Zeigen Sie für zwei Klassen, dass ein Fehler $\delta$ im Anteil $\mu_1$ positiver Vorhersagen den geschätzten Anteil tatsächlich positiver Fälle um $\delta/(C_{11} - C_{10})$ verändert. Verwenden Sie den Standardfehler $\sqrt{\mu_1 (1 - \mu_1)/m}$ eines aus $m$ Zieleingaben geschätzten Anteils. Bestimmen Sie damit, wie viele Zieleingaben im Beispiel zur Früherkennung nötig sind, damit der geschätzte Anteil der Erkrankten einen Standardfehler von 0.01 hat. Wiederholen Sie die Rechnung für einen schwächeren Test mit $C_{11} = 0.6$ und $C_{10} = 0.4$ in derselben Population.
   :exlabel:`ex_environment-and-distribution-shift-label-shift-system`

<!--en:1b4b48e6-->
1. [code] **Gelernte und wahre Gewichte im Vergleich.** ● Verwenden Sie erneut die Versuchsreihe aus :exref:`ex_environment-and-distribution-shift-when-reweighting-fails` mit den Zielmittelwerten $(s, 0)$.
    1. Leiten Sie das wahre logarithmische Dichteverhältnis $\log(p_T(\mathbf{x})/p_S(\mathbf{x}))$ als Funktion von $s$ her. Vergleichen Sie es mit dem gelernten $h$ für jedes $s$. Erklären Sie, wie der Domänenklassifikator bei großem $s$ eine hohe Genauigkeit erreichen kann, obwohl $h$ stark vom wahren logarithmischen Dichteverhältnis abweicht.
    1. Zeigen Sie, dass für die wahren Gewichte unter der Quellverteilung $\mathbb{E}[\beta(\mathbf{x})] = 1$ und $\mathbb{E}[\beta(\mathbf{x})^2] = e^{s^2}$ gelten. Folgern Sie, dass ihre effektive Stichprobengröße für große $n$ etwa $n e^{-s^2}$ beträgt. Werten Sie diesen Ausdruck für $n = 1000$ und $s = 1, 2, 3$ aus.
    1. Berechnen Sie für jedes $s$ die effektive Stichprobengröße der wahren und der gelernten Gewichte auf der tatsächlich gezogenen Quellstichprobe. Erklären Sie, warum bei $s = 2$ der Stichprobenwert meist deutlich größer als $1000\, e^{-4} \approx 18$ ist. Was folgt daraus, wenn Sie die Zuverlässigkeit einer gewichteten Anpassung allein anhand der Stichprobe beurteilen? Behebt eine Begrenzung bei $c = 5$ den Mangel an Beispielen?
   :exlabel:`ex_environment-and-distribution-shift-learned-true-weights`

<!--en:1ddf9dad-->
1. **Annahmen über die Verschiebung bei Daten ohne Label.** ● In einer Quelldomäne mit zwei Klassen gilt $p_S(y = 1) = p_S(y = 0) = \frac{1}{2}$. Für die skalare Eingabe gelten $x \mid y = 1 \sim \mathcal{N}(1, 1)$ und $x \mid y = 0 \sim \mathcal{N}(-1, 1)$. Daraus folgt $p_S(y = 1 \mid x) = 1/(1 + \exp(-2x))$. Zieleingaben ohne Label folgen der Mischverteilung

   $$p_T(x) = 0.8\, \mathcal{N}(x; 1, 1) + 0.2\, \mathcal{N}(x; -1, 1).$$

    1. Nehmen Sie eine Labelverteilungsverschiebung an. Geben Sie $p_T(y = 1)$ und den Schwellenwert für $x$ beim Bayes-optimalen Zielklassifikator an.
    1. Nehmen Sie stattdessen eine Kovariatenverschiebung mit demselben $p_T(x)$ an. Berechnen Sie $p_T(y = 1)$ numerisch und geben Sie den Bayes-optimalen Schwellenwert an.
    1. Erklären Sie anhand Ihrer Ergebnisse, warum keine Statistik der Zielstichprobe ohne Label entscheiden kann, welches der beiden Korrekturverfahren aus diesem Abschnitt anwendbar ist.

<!--en:df536fb8-->
[Diskussion](https://d2l.discourse.group/t/105)

<!--en:2b5f0b30-->
Umgebung und Verteilungsverschiebung: Kovariaten-, Labelverteilungs- und Konzeptverschiebung<br>**Zerlegungen der gemeinsamen Verteilung · Kovariatenverschiebung · Wichtigkeitsgewichtung mit Dichteverhältnissen · Domänenklassifikator · Matrixinversion bei Labelverteilungsverschiebung · Rückkopplungsschleifen**

<!--en:2dfa0987-->
::: {.slide title="Wenn hohe Testgenauigkeit im Einsatz einbricht"}

<!--en:94f4a38d-->
[Die Lücke zwischen Test und Einsatz]{.kicker}

<!--en:a0f00715-->
Alle statistischen Garantien in §{chap_classification_generalization} setzen voraus, dass die Daten im Einsatz aus genau derselben Verteilung $p_S(\mathbf{x}, y)$ stammen wie Trainings- und Testdaten. In der Praxis ändern sich die Verteilungen $p_T(\mathbf{x}, y)$ im Einsatz ständig:

<!--en:a0c9122d-->
- **Passive und aktive Verschiebungen:** Kameras und Klinikscanner werden modernisiert, Bildarten wechseln (Fotos $\to$ Zeichnungen), saisonale Epidemien verändern über Nacht die Prävalenz. Menschen passen sich strategisch an Entscheidungsregeln an (*Gesetz von Goodhart*: Wer Kredite für Oxford-Schuhe bewilligt, bewegt Antragstellende zum Schuhwechsel, ohne das Ausfallrisiko zu senken).

<!--en:cb8fed40-->
::: {.slide title="Warum Anpassung ohne Label Invarianzannahmen braucht"}

<!--en:09e25449-->
[Kein Transfer ohne Annahmen]{.kicker}

<!--en:d69db0cf-->
Im Einsatz liegen Beispiele aus der **Quelle** mit Labels $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n \sim p_S(\mathbf{x}, y)$ und **Zieleingaben ohne Label** $\{\tilde{\mathbf{x}}^{(j)}\}_{j=1}^m \sim p_T(\mathbf{x})$ vor:

<!--en:7899ced6-->
- **Gegenbeispiel mit gezielt vertauschten Labels:** Angenommen, es gilt $p_T(\mathbf{x}) = p_S(\mathbf{x})$, doch über Nacht vertauschen sich unbemerkt die Labels für dieselben Eingaben ($p_T(y = \text{cat} \mid \mathbf{x}) = p_S(y = \text{dog} \mid \mathbf{x})$).

<!--en:3d84c082-->
- **Mit Eingaben ohne Label nicht identifizierbar:** Wegen $p_T(\mathbf{x}) = p_S(\mathbf{x})$ kann kein Test allein anhand solcher Zieleingaben den Tausch erkennen ($100\%$ Quellgenauigkeit $\to 0\%$ Zielgenauigkeit).

<!--en:e43a2351-->
::: {.slide title="Zwei Zerlegungen definieren drei Formen der Verschiebung"}

<!--en:7405723d-->
[Einteilung der Verteilungsverschiebungen]{.kicker}

<!--en:6a97d696-->
Jede gemeinsame Verteilung $p(\mathbf{x}, y)$ lässt sich in beiden Kausalrichtungen in Faktoren zerlegen:

<!--en:a3d5fbba-->
- **Kovariatenverschiebung ($\mathbf{x} \to y$):** $p(\mathbf{x})$ verschiebt sich, während $p(y \mid \mathbf{x})$ invariant bleibt (z. B. Fotos $\to$ Zeichnungen).

<!--en:9bb1fb78-->
- **Labelverteilungsverschiebung ($y \to \mathbf{x}$):** $p(y)$ verschiebt sich, während $p(\mathbf{x} \mid y)$ invariant bleibt (z. B. Grippewelle im Winter).

<!--en:9cd2161c-->
- **Konzeptverschiebung:** Die bedingte Verteilung $p(y \mid \mathbf{x})$ selbst ändert sich mit dem Ort oder der Zeit („soda“ oder „pop“).

<!--en:bc394a6a-->
::: {.slide title="Die drei Formen der Verschiebung im geometrischen Vergleich"}

<!--en:08a4baa3-->
[Eindimensionale Eingabedichten und Bayes-optimale Grenzen]{.kicker}

<!--en:909db329-->
Gemeinsame Dichten $p(x, y)$ in Quelle (oben) und Ziel (unten). Die gepunktete Linie ist die Bayes-optimale Grenze:

<!--en:96f65616-->
- **Kovariatenverschiebung (links):** $p(x)$ verschiebt sich nach rechts, während $p(y \mid x)$ und die Bayes-Grenze fest bleiben. Ein falsch spezifiziertes Modell, das in Bereichen hoher Dichte $p_S(x)$ angepasst wurde, kann dort versagen, wo die Masse von $p_T(x)$ liegt.

<!--en:c3af0068-->
- **Labelverteilungsverschiebung (Mitte) und Konzeptverschiebung (rechts):** Ändert sich $p(y)$, wandert die Grenze gemäß $p(y \mid x) \propto p(x \mid y)p(y)$ zur selteneren Klasse. Bei einer Konzeptverschiebung ändert sich $p(y \mid x)$ direkt.

<!--en:b12f28a4-->
::: {.slide title="Kovariaten- und Labelverteilungsverschiebung im Detail"}

<!--en:4f8a2437-->
[Warum falsch spezifizierte Modelle eine Umgewichtung brauchen]{.kicker}

<!--en:28d2d95a-->
- **Kovariatenverschiebung ($p(y \mid \mathbf{x})$ fest):** Enthielte $\mathcal{F}$ die wahre bedingte Verteilung $p(y \mid \mathbf{x})$, ergäbe die Anpassung an unendlich viele Quelldaten überall ein optimales Modell. Ein beschränktes Modell (eine Gerade für eine gekrümmte Grenze) büßt dagegen dort Genauigkeit ein, wo $p_S(\mathbf{x})$ klein ist.

<!--en:62049350-->
- **Labelverteilungsverschiebung ($p(\mathbf{x} \mid y)$ fest):** Zur Anpassung genügen die **$q$ Klassenanteile** $p_T(y)$. Ein Dichteverhältnis über $\mathbb{R}^d$ ist entbehrlich.

<!--en:ff172ae8-->
::: {.slide title="Konzeptverschiebung durch Änderung der Zuordnungsregel"}

<!--en:cb46dc40-->
[Geografischer und zeitlicher Wandel]{.kicker}

<!--en:2a51ddf3-->
- **Bedeutungswandel der Labels:** Bei einer Konzeptverschiebung ändert sich die bedingte Verteilung $p(y \mid \mathbf{x})$ selbst. Identische Produktmerkmale oder Getränkebeschreibungen ergeben im Mittleren Westen der USA `"pop"`, im Nordosten `"soda"` und im Süden `"coke"`.

<!--en:6453997c-->
- **Warum Eingaben ohne Label nicht zur Korrektur reichen:** Ändert sich $p(y \mid \mathbf{x})$ bei unverändertem $p(\mathbf{x})$, sind frühere Labels veraltet. Zur Anpassung müssen laufend kleine Mengen neuer Ziellabels erhoben und das Modell kontinuierlich feinabgestimmt oder neu trainiert werden.

<!--en:8a275c23-->
::: {.slide title="Wichtigkeitsgewichtung bei Kovariatenverschiebung"}

<!--en:a81010f9-->
[Training auf Quelldaten am Zielrisiko ausrichten]{.kicker}

<!--en:ef9f61c4-->
Bei Kovariatenverschiebung ($p_T(y \mid \mathbf{x}) = p_S(y \mid \mathbf{x}) = p(y \mid \mathbf{x})$) und überdecktem Zielträger ($p_S(\mathbf{x}) > 0$ überall dort, wo $p_T(\mathbf{x}) > 0$ gilt) lässt sich das Zielrisiko als gewichteter Erwartungswert bezüglich $p_S$ schreiben (Shimodaira, 2000):

<!--en:c25c175a-->
- **Gewichtete empirische Risikominimierung:** Auf $n$ Quellbeispielen mit Labels $\frac{1}{n}\sum_{i=1}^n \beta_i\,l(f(\mathbf{x}^{(i)}), y^{(i)})$ minimieren.

<!--en:792928be-->
::: {.slide title="Trägerüberdeckung, extreme Gewichte am rechten Rand und Gewichtsbegrenzung"}

<!--en:f868944a-->
[Varianz und effektive Stichprobengröße]{.kicker}

<!--en:8cbc7d28-->
Die Wichtigkeitsgewichtung birgt zwei grundlegende statistische Risiken:

<!--en:6fff7f6c-->
- **Fehlende Trägerüberdeckung ($p_S(\mathbf{x}) = 0$, wo $p_T(\mathbf{x}) > 0$ gilt):** Besteht die Kontrollgruppe der Quelle nur aus 20-Jährigen, die Zielgruppe aber aus 70-Jährigen, ist $\beta(\mathbf{x})$ dort undefiniert. Keine Umgewichtung der 20-Jährigen liefert Daten über 70-Jährige.

<!--en:3471f16d-->
- **Extreme Gewichte und effektive Stichprobengröße ($n_{\text{eff}}$):** Für $p_S = \mathcal{N}(0,1)$ und $p_T = \mathcal{N}(2,1)$ erreicht $\beta(x_1) = e^{2x_1 - 2}$ den Wert $55$ bei $x_1 = 3$. Dadurch schrumpft die effektive Stichprobengröße:
  $$n_{\mathrm{eff}} = \frac{\big(\sum_{i=1}^n \beta_i\big)^2}{\sum_{i=1}^n \beta_i^2} \ll n$$

<!--en:e438db2a-->
- **Begrenzung der Gewichte:** Die Begrenzung gemäß $\tilde{\beta}_i = \min(\beta_i, c)$ (z. B. $c = 5$) senkt die Varianz erheblich, auf Kosten einer kleinen Verzerrung.

<!--en:136979a0-->
::: {.slide title="Schätzung des Dichteverhältnisses mit einem Domänenklassifikator"}

<!--en:93c25ad8-->
[Vermeidung hochdimensionaler Dichteschätzung]{.kicker}

<!--en:b5d2e64d-->
Anstelle separater Schätzungen von $p_S(\mathbf{x})$ und $p_T(\mathbf{x})$ in $\mathbb{R}^d$ die $n$ Quelleingaben ($z = 0$) mit $m$ Zieleingaben ohne Klassenlabel ($z = 1$) zusammenfassen und einen logistischen **Domänenklassifikator** $P(z = 1 \mid \mathbf{x}) = \sigma(h(\mathbf{x}))$ trainieren:

<!--en:28df1aa0-->
- **Direkte Formel für das Dichteverhältnis:** $\beta(\mathbf{x}) = \frac{n}{m}\exp(h(\mathbf{x})) \propto \exp(h(\mathbf{x}))$, **ohne ein einziges Ziellabel**.

<!--en:e65227a4-->
::: {.slide title="Versuchsaufbau zur zweidimensionalen Kovariatenverschiebung"}

<!--en:a50ab8b3-->
[Wahre parabelförmige Grenze bei verschobener Gauß-Verteilung]{.kicker}

<!--en:b65ccb42-->
Gegeben sind $p_S = \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$, $p_T = \mathcal{N}((2, 0)^\top, \mathbf{I}_2)$ und die wahre, invariante Regel $y = \mathbf{1}(x_2 > \frac{1}{2}x_1^2 - 1)$:

<!--en:573e3f4a-->
::: {.slide title="Training des Domänenklassifikators und Bestimmung der Wichtigkeitsgewichte"}

<!--en:a50ab8b3-2-->
[Wahre parabelförmige Grenze bei verschobener Gauß-Verteilung]{.kicker}

<!--en:748d079d-->
Aus den zusammengefassten Daten `X_src` ($z = 0$) und `X_tgt` ($z = 1$) lassen sich das logarithmische Dichteverhältnis $h(\mathbf{x})$, die Quellgewichte $\beta_i = \exp(h(\mathbf{x}^{(i)}))$ und ihre effektive Stichprobengröße schätzen:

<!--en:729b2237-->
::: {.slide title="Rekonstruktion des logarithmischen Dichteverhältnisses mit dem Domänenklassifikator"}

<!--en:a50ab8b3-3-->
[Wahre parabelförmige Grenze bei verschobener Gauß-Verteilung]{.kicker}

<!--en:3bca8e64-->
- **Logarithmisches Dichteverhältnis und Begrenzung:** Das gelernte $h(\mathbf{x}) \approx 2x_1 - 2$ stimmt mit $\log(p_T/p_S)$ überein. Die Begrenzung auf $\beta_i \le 5$ erhöht $n_{\mathrm{eff}}$ auf $\sim 200$.

<!--en:a8f9c840-->
- **Verschiebung erkannt:** Die Genauigkeit $\approx 0.84$ auf zurückgehaltenen Domänendaten erreicht die Bayes-Obergrenze $\Phi(1) \approx 0.84$ (Zufallsniveau $0.50$).

<!--en:fd881cc7-->
::: {.slide title="Logistische Regression mit Wichtigkeitsgewichten"}

<!--en:871ff5c1-->
[Empirische Risikominimierung ohne Gewichte, mit Wichtigkeitsgewichten und mit begrenzten Gewichten]{.kicker}

<!--en:71c34a45-->
Das Residuum jedes Quellbeispiels im Gradienten mit $\beta_i = \exp(h(\mathbf{x}^{(i)}))$ gewichten, und zwar in `fit_logreg`:

<!--en:8501e968-->
::: {.slide title="Höhere Zielgenauigkeit durch gewichtete logistische Regression"}

<!--en:871ff5c1-2-->
[Empirische Risikominimierung ohne Gewichte, mit Wichtigkeitsgewichten und mit begrenzten Gewichten]{.kicker}

<!--en:7de4b298-->
Die Umgewichtung erhöht die Zielgenauigkeit von $\sim 50\%$–$60\%$ auf $\sim 95\%$, ohne ein einziges Ziellabel:

<!--en:5beb1ee8-->
::: {.slide title="Wie Wichtigkeitsgewichte die lineare Entscheidungsgrenze neigen"}

<!--en:cf7f8ecb-->
[Geometrische Wirkung der Korrektur einer Kovariatenverschiebung]{.kicker}

<!--en:5a38eb1c-->
Die Quellpunktwolke (Kreisfläche $\propto \beta_i$), die verschobene Zielpunktwolke (orange) und die angepassten linearen Grenzen erklären den Anstieg der Genauigkeit um $35$ bis $45$ Prozentpunkte:

<!--en:884d1431-->
- **Ungewichtete lineare Grenze (grün):** Nahe $(0, 0)$ folgt eine fast waagerechte Gerade ($x_2 \approx -0.6$) dem Scheitel der Parabel und klassifiziert fast die Hälfte der Zielpunkte um $(2, 0)$ falsch.

<!--en:7b67bb2b-->
- **Grenze mit Wichtigkeitsgewichten (rot):** Punkte im Bereich $x_1 \le 0$ erhalten weniger Gewicht, Quellpunkte am rechten Rand ($x_1 > 1$) mehr. Dadurch neigt sich die Gerade und folgt dem rechten Parabelast, wo die Zieldaten liegen.

<!--en:4968e2d5-->
::: {.slide title="Korrektur der Labelverteilungsverschiebung mit der Konfusionsmatrix"}

<!--en:3ff023dc-->
[Schätzung der Verschiebung mit einer Blackbox (BBSE / Saerens et al.)]{.kicker}

<!--en:1a9a84b4-->
Wie lassen sich bei einer **Labelverteilungsverschiebung** ($p_T(\mathbf{x} \mid y) = p_S(\mathbf{x} \mid y)$, $p_T(y) \neq p_S(y)$) die unbekannten A-priori-Wahrscheinlichkeiten $\mathbf{p}_T \in \mathbb{R}^q$ der Zielklassen aus Zieleingaben **ohne Label** schätzen?

<!--en:dc78cf53-->
- **Invarianz der spaltenweise normierten Konfusionsmatrix $\mathbf{C}$:** Da $p(\mathbf{x} \mid y)$ invariant ist, gilt für $C_{ij} = P(\hat{c}(\mathbf{x}) = i \mid y = j)$ in Quelle und Ziel derselbe Wert.

<!--en:c8fb6fcf-->
- **Satz der totalen Wahrscheinlichkeit für die Zielvorhersagen $\boldsymbol{\mu}$:**
  $$\mu_i = P_T(\hat{c}(\mathbf{x}) = i) = \sum_{j=1}^q C_{ij}\,p_T(y = j) \iff \mathbf{C}\,\mathbf{p}_T = \boldsymbol{\mu}$$

<!--en:3576789e-->
::: {.slide title="Bestimmung der Zielklassenanteile durch Inversion von C"}

<!--en:15753c27-->
[Klinisches Rechenbeispiel und Bedingung für Invertierbarkeit]{.kicker}

<!--en:45e63f26-->
Ist $\mathbf{C}$ invertierbar (garantiert bei einer Trefferquote $C_{jj} > 1/2$ für jede Klasse), liefert die Lösung $\hat{\mathbf{p}}_T = \mathbf{C}^{-1}\boldsymbol{\mu}$ die A-priori-Wahrscheinlichkeiten der Zielklassen ohne ein einziges Ziellabel:

<!--en:d0321a42-->
- **Klinisches Rechenbeispiel (Sensitivität $90\%$, Falsch-Positiv-Rate $5\%$):** Verzeichnet eine Klinik im Winter $\mu_2 = 13.5\%$ positive Befunde, gilt:
  $$0.05(1 - p) + 0.90p = 0.135 \implies 0.85p = 0.085 \implies p = 0.10$$
  Die wahre Grippeprävalenz beträgt also $10\%$, berücksichtigt sind dabei $4.5\%$ falsch positive und $1\%$ übersehene Fälle.

<!--en:3eae04ad-->
::: {.slide title="Auflösung der Labelverteilungsverschiebung bei Fashion-MNIST"}

<!--en:e1e70e85-->
[Anstieg des Hemdenanteils auf 50% und direkte Logit-Anpassung nach Bayes]{.kicker}

<!--en:02f1364c-->
Schätzung von $\mathbf{C}$ auf $5{,}000$ Validierungsbildern mit ausgeglichenen Klassenanteilen. Test auf einem Zieldatensatz ohne Label, in dem **der Hemdenanteil von $10\%$ auf $50\%$ steigt**:

<!--en:e65a3660-->
::: {.slide title="Direkte Logit-Anpassung nach Bayes"}

<!--en:e1e70e85-2-->
[Anstieg des Hemdenanteils auf 50% und direkte Logit-Anpassung nach Bayes]{.kicker}

<!--en:33175287-->
- **Direkte Logit-Anpassung nach Bayes:** Addition von $\log\frac{\hat{p}_T(y)}{p_S(y)}$ zu jedem Klassenlogit erhöht die Zielgenauigkeit um $\sim 5\text{--}15$ Prozentpunkte, ohne einen einzigen Gradientenschritt.

<!--en:b7e348e3-->
- **Kein erneutes Training:** Die Quellgewichte bleiben fest. Nur die A-priori-Verteilung der Klassen verschiebt sich.

<!--en:a11dc798-->
::: {.slide title="Konzeptverschiebung und performative Rückkopplungsschleifen"}

<!--en:a3c2dd61-->
[Wenn das Modell seine eigene Umgebung verändert]{.kicker}

<!--en:fc31d780-->
Vorhersagen in soziotechnischen Systemen im Einsatz beschreiben die Welt und lösen zugleich **Handlungen** aus, die künftige Datenverteilungen verändern (*performative Vorhersage*):

<!--en:a471d16c-->
- **Sich selbst verstärkende Rückkopplung in der vorausschauenden Polizeiarbeit:** Hohe vorhergesagte Kriminalität im Viertel $A$ lenkt zusätzliche Streifen nach $A$. Durch die höhere Präsenz werden in $A$ mehr geringfügige Vergehen erfasst, während in Viertel $B$ keine Streifen unterwegs sind. Erneutes Training mit den protokollierten Festnahmen verstärkt die anfängliche Verzerrung.

<!--en:66418113-->
- **Rückkopplungsschleifen durchbrechen:** Passive Wichtigkeitsgewichtung kann Daten nicht korrigieren, wenn die Entscheidungen des Modells selbst bestimmte Beobachtungen ausschließen. Zur Überprüfung sind randomisierte Exploration (Banditen / RL) oder unabhängige Messungen der Ergebnisse nötig.

<!--en:231ae278-->
::: {.slide title="Verständnisfragen zur Diagnose und Korrektur von Verteilungsverschiebungen"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:bd9216d9-->
Verteilungsverschiebungen erkennen und mathematisch beschreiben:

<!--en:f09d1907-->
1. **Verschiebungen einordnen:** Welcher Fall ist eine *Kovariatenverschiebung*, *Labelverteilungsverschiebung* oder *Konzeptverschiebung*? *(a)* Eine Mautkamera an der Autobahn liefert nach einer Umrüstung kontrastreichere Bilder. *(b)* Eine sommerliche Hitzewelle verdreifacht den Anteil der Sandalenbestellungen in einem Onlinegeschäft. *(c)* Ein psychiatrisches Handbuch ändert die Diagnosekriterien für eine Erkrankung.

<!--en:1189f46d-->
2. **Verhältnis der Wichtigkeitsgewichte:** Im zweidimensionalen Beispiel mit Gauß-Verteilungen gilt $\beta(\mathbf{x}) = \exp(2x_1 - 2)$. Das Wievielfache des Gewichts erhält ein Quellpunkt bei $(3, 0)$ verglichen mit einem Quellpunkt bei $(1, 0)$? Wie groß ist für große $n$ die theoretische relative effektive Stichprobengröße $n_{\text{eff}} / n = 1 / \mathbb{E}_S[\beta^2]$ bei einer Verschiebung um $s = 2$ ($\mathbb{E}_S[\beta^2] = e^{s^2}$)?

<!--en:5e562342-->
3. **Fehlende Identifizierbarkeit ohne Kausalwissen (Übung 5):** Zieleingaben $x$ ohne Label verschieben sich zu positiven Werten hin. Kann ein statistischer Test allein anhand dieser Eingaben feststellen, ob eine *Kovariatenverschiebung* oder eine *Labelverteilungsverschiebung* vorliegt?

<!--en:667fad3d-->
**Lösungen:** **(1)** *(a)* Kovariatenverschiebung ($p(\mathbf{x})$ ändert sich, die Fahrzeugidentität bei gegebener Eingabe gemäß $p(y\mid\mathbf{x})$ bleibt gleich). *(b)* Labelverteilungsverschiebung ($p(y)$ ändert sich, die Verteilung der Produktfotos je Klasse $p(\mathbf{x}\mid y)$ bleibt gleich). *(c)* Konzeptverschiebung ($p(y\mid\mathbf{x})$ ändert sich). **(2)** $\beta(3,0) = e^4 \approx 54.6$ gegenüber $\beta(1,0) = e^0 = 1$ (Gewichtsfaktor $54.6\times$). Theoretisch ergeben sich $n_{\mathrm{eff}} \approx n e^{-s^2} = 1000 e^{-4} \approx 18.3$ effektive Beispiele. **(3)** Nein. Dieselbe Randverteilung $p_T(x)$ im Ziel kann dadurch entstehen, dass sich $p_T(x)$ bei festem $p(y\mid x)$ verschiebt (der optimale Schwellenwert bleibt fest). Ebenso kann $p_T(y=1)$ bei festem $p(x\mid y)$ steigen (der optimale Schwellenwert verschiebt sich). Die Wahl der richtigen Korrektur erfordert Fachwissen über die kausalen Zusammenhänge.

<!--en:4da51157-->
::: {.slide title="Zusammenfassung der Verteilungsverschiebungen und ihrer Korrektur"}

<!--en:27de615d-->
[Zusammenfassung des Abschnitts]{.kicker}

<!--en:1787017a-->
| Form der Verschiebung | Invarianter Faktor | Korrekturverfahren | Voraussetzung und mögliche Fehlerquelle |

<!--en:13851209-->
| **Kovariatenverschiebung** | $p(y \mid \mathbf{x})$ fest ($p(\mathbf{x})$ verschiebt sich) | Gewichtete empirische Risikominimierung mit $\beta(\mathbf{x}) = \frac{p_T(\mathbf{x})}{p_S(\mathbf{x})} \propto \exp(h(\mathbf{x}))$ aus einem logistischen Domänenklassifikator | Erfordert Überdeckung des Zielträgers ($p_S > 0$, wo $p_T > 0$ gilt). Gewichte gemäß $\min(\beta_i, c)$ begrenzen, um $n_{\mathrm{eff}}$ zu erhalten |

<!--en:699ea068-->
| **Labelverteilungsverschiebung** | $p(\mathbf{x} \mid y)$ fest ($p(y)$ verschiebt sich) | $\mathbf{C}\,\hat{\mathbf{p}}_T = \boldsymbol{\mu}$ nach den A-priori-Wahrscheinlichkeiten der Zielklassen auflösen. Mit $\hat{p}_T(y)/p_S(y)$ umgewichten oder die Logits um $+\log\frac{\hat{p}_T(y)}{p_S(y)}$ verschieben | Erfordert eine invertierbare Konfusionsmatrix $\mathbf{C}$ der Quelle (garantiert bei einer Trefferquote $C_{jj} > 1/2$ in jeder Klasse) |

<!--en:f2083461-->
| **Konzeptverschiebung** | Keine Invarianz vorausgesetzt ($p(y \mid \mathbf{x})$ ändert sich) | Laufende Überprüfung anhand von Zieldaten mit aktuellen Labels und regelmäßiges Fine-Tuning oder erneutes Training | Bei unverändertem $p(\mathbf{x})$ ist die Konzeptverschiebung anhand von Zieleingaben ohne Label nicht erkennbar |

<!--en:e8a7654e-->
| **Rückkopplungsschleifen** | Performativ: Handlungen des Modells verändern $p_T(\mathbf{x}, y)$ | Unabhängige Prüfdaten erheben, die keine Modellentscheidung filtert. Randomisierte Exploration nutzen | Erneutes Training mit bereits vom Modell gefilterten Protokolldaten erzeugt sich selbst verstärkende Schleifen, die die eigene Vorhersage bestätigen |
