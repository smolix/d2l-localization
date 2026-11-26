<!--en:c4c997a5-->
# Moderne Trainingsrezepte für konvolutionäre Netze

<!--en:ab7bab8c-->
In diesem Kapitel haben wir sorgfältig darauf geachtet, das Trainingsverfahren innerhalb jedes Abschnitts konstant zu halten. So ließen sich Unterschiede in der Genauigkeit auf die Architektur zurückführen. Selbst unter diesen kontrollierten Bedingungen änderte sich die Genauigkeit von VGG um 1.5 Prozentpunkte, allein weil wir die maximale Lernrate änderten (:numref:`sec_blocks`). Veröffentlichte Benchmarks sind oft weit weniger kontrolliert. Ein neues Netz wird mit den neuesten Verfahren zur Optimierung und Regularisierung trainiert. Zum Vergleich dient eine klassische Baseline, deren Genauigkeit aus einer Jahre zurückliegenden Veröffentlichung stammt. Welchen Anteil an einem gemeldeten Durchbruch hat das *Trainingsrezept* (engl. training recipe), also die Wahl von Optimierer, Lernratenplan, Datenaugmentierung, Regularisierung und Trainingsdauer? Wie viel trägt die Architektur selbst bei?

<!--en:ef4de680-->
ResNet-50, die tiefere Variante des ResNet-18 aus :numref:`sec_resnet`, liefert eine bemerkenswerte Antwort. :citet:`wightman2021resnet` trainierten ResNet-50 mit Verfahren aus den frühen 2020er-Jahren erneut, ohne eine einzige Schicht der Architektur von 2015 zu ändern. Die Top-1-Genauigkeit auf ImageNet stieg von 76.1% auf 80.4%, allein durch das Trainingsrezept also um 4.3 Prozentpunkte. Der Zugewinn erreicht oder übertrifft die Verbesserungen, die viele in der Zwischenzeit veröffentlichte Architekturen für sich beanspruchen. Meldet eine Arbeit einen Vorsprung von zwei Prozentpunkten gegenüber einer nach veraltetem Rezept trainierten Baseline, bleibt daher offen, ob der Fortschritt der Architektur oder dem Rezept zu verdanken ist.

<!--en:a50801f9-->
In diesem Abschnitt untersuchen wir, was sich zwischen 2015 und 2022 geändert hat. Wir implementieren jede wesentliche Zutat mit wenigen Codezeilen und trainieren dasselbe ResNet-18 nach einem klassischen und einem modernen Rezept. So können wir den Unterschied direkt beobachten. Keines dieser Verfahren ist an Faltungen gebunden. Genau dieselben Verfahren kommen beim Training moderner Transformer zum Einsatz und bilden damit eine allgemeine Grundlage für die heutige Praxis des Deep Learning.

<!--en:b7d3f81b-->
## Trainingsrezepte von 2015 bis 2022

<!--en:8981fbca-->
Schon das ursprüngliche ResNet-Rezept von 2015 war sorgfältig abgestimmt. Es kombinierte stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD) mit Momentum (:numref:`sec_efficient_cnns`), Gewichtszerfall (engl. weight decay), stufenweisen Lernratenabfall (zweimaliges Teilen durch 10) und zufällige Bildausschnitte mit horizontalen Spiegelungen :cite:`He.Zhang.Ren.ea.2016`. In den folgenden Jahren untersuchte die Forschung jeden Bestandteil dieser Pipeline erneut, zunächst einzeln und dann systematisch in zahlreichen Kombinationen :cite:`He.Zhang.Zhang.ea.2019,wightman2021resnet`. :numref:`tab_recipe_2015_2022` stellt der Baseline von 2015 ein typisches Rezept aus den frühen 2020er-Jahren gegenüber. Im Folgenden erläutern und implementieren wir jeden neuen Bestandteil.

<!--en:a9e73cbd-->
:Ein ImageNet-Rezept von 2015 und ein typisches Rezept aus den frühen 2020er-Jahren, das eine hohe Genauigkeit erzielt. Die letzte Spalte gibt keinen allgemeingültigen Standard vor. Welche Rezepte gut funktionieren, hängt weiterhin vom Modell und vom Rechenbudget ab.

<!--en:5025cb3a-->
| Zutat | Rezept von 2015 | Rezept von 2022 |

<!--en:7bd86666-->
| Optimierer | SGD mit Momentum | AdamW oder LAMB, auch SGD bleibt bei guter Abstimmung konkurrenzfähig |

<!--en:f5c33644-->
| Lernratenplan | stufenweiser Abfall (zweimal durch 10 teilen) | Kosinus-Abfall mit linearer Aufwärmphase |

<!--en:87e49180-->
| Augmentierung | zufällige Bildausschnitte, horizontale Spiegelungen | RandAugment, Mixup, CutMix, zufälliges Löschen von Bildbereichen |

<!--en:8548bff3-->
| Regularisierung | Gewichtszerfall | + Label-Smoothing, stochastische Tiefe |

<!--en:0db7158c-->
| Dauer | 90 Epochen | 300 bis 600 Epochen |

<!--en:d1275538-->
| Gewichte für die Auswertung | Gewichte am Trainingsende | exponentieller gleitender Mittelwert der Gewichte |

<!--en:bbf1305c-->
Zwei Beobachtungen sind für das Verständnis von :numref:`tab_recipe_2015_2022` wesentlich. Erstens wirken die Maßnahmen in den einzelnen Zeilen eng zusammen. Stärkere Datenaugmentierung und Regularisierung erschweren die Anpassung an das Trainingsziel und erfordern dadurch deutlich mehr Trainingsepochen. Das erklärt, warum die Trainingsdauer von 90 auf 300–600 Epochen stieg. Mit zunehmender Regularisierungsstärke muss auch die Trainingsdauer wachsen. Bei kurzem Training *verschlechtert* eine starke Augmentierung die Genauigkeit häufig gegenüber einem Rezept mit schwächerer Augmentierung.

<!--en:ace742e2-->
Zweitens zeigt die Zeile zum Optimierer, dass es inzwischen mehr Möglichkeiten gibt. Sie gibt keine allgemeingültige Wahl vor. Adaptive Optimierer wie Adam (ausführlich in :numref:`chap_optimization`) normieren den Gradienten jedes Parameters anhand eines gleitenden Schätzwerts für seinen Betrag. Dadurch ändert sich jede Koordinate ungefähr in der Größenordnung der Lernrate. Klassische Implementierungen rechneten den Gewichtszerfall vor dieser Skalierung in den Gradienten ein und verzerrten damit unbeabsichtigt den Strafterm. *AdamW* :cite:`Loshchilov.Hutter.2019` entkoppelt den Gewichtszerfall vom adaptiven Gradientenschritt. Jedes Gewicht sinkt direkt um einen Bruchteil seines Werts, der dem Produkt aus Lernrate und Zerfallskoeffizient entspricht. Mit dieser Korrektur wurden adaptive Optimierer auf Bilderkennungsbenchmarks gegenüber gut abgestimmtem SGD konkurrenzfähig. (LAMB ist mit AdamW verwandt und für große Batches ausgelegt. Das oben erwähnte ResNet-50 erreichte seine 80.4% mit LAMB nach 600 Epochen Training mit binärer Kreuzentropie als Verlust und ohne Mittelung der Gewichte. Wir benötigen LAMB hier nicht.)

<!--en:fd526baf-->
## Implementierung der Zutaten

<!--en:98edfcd0-->
Für jede moderne Zutat genügen wenige Codezeilen. Wir implementieren Label-Smoothing, Kosinuspläne mit Aufwärmphase, Mixup, stochastische Tiefe (engl. stochastic depth) und die exponentielle Mittelung der Gewichte. (RandAugment :cite:`cubuk2020randaugment` lassen wir aus. Es wendet zufällige Sequenzen gängiger Bildtransformationen mit einer abgestimmten Stärke an. In der Praxis genügt dafür ein Bibliotheksaufruf bei der Vorverarbeitung. Entscheidend ist die Abstimmung der Stärke, weniger die Implementierung.)

<!--en:b3b9e624-->
### Label-Smoothing

<!--en:c65e6b69-->
Bei der üblichen Kreuzentropie mit One-Hot-Zielwerten sinkt der Verlust für ein korrekt klassifiziertes Beispiel immer weiter, während das Logit der richtigen Klasse gegen $+\infty$ wächst. Da keine endlichen Logits die Zielfunktion minimieren, drängt das Training das Netz zu extrem sicheren, schlecht kalibrierten Vorhersagen. *Label-Smoothing* :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` wirkt dem entgegen. Es ersetzt den harten One-Hot-Zielwert durch eine geglättete Verteilung, die das Gewicht $1-\epsilon$ der richtigen Klasse $y$ zuweist und $\epsilon$ gleichmäßig auf alle $K$ Klassen verteilt. Bezeichnen wir die von der Softmax-Funktion vorhergesagten Wahrscheinlichkeiten mit $p_k$, so ergibt sich für den Verlust pro Beispiel

<!--en:5e7af1ed-->
Der erste Term ist die gewöhnliche Kreuzentropie, skaliert mit $1-\epsilon$. Der zweite bestraft jede Annäherung einer Klassenwahrscheinlichkeit an null. Der Verlust erreicht sein Minimum bei $p_y = 1-\epsilon+\epsilon/K$ und $p_k = \epsilon/K$ für $k \neq y$. Das entspricht einer wohldefinierten, endlichen Margin zwischen den Logits. Betrachten wir eine bereits sichere, korrekte Vorhersage und vergleichen wir die Kreuzentropie mit und ohne Glättung:

<!--en:5c2a9cce-->
Bei $\epsilon = 0.1$ springt der Verlust von 0.0001 auf 0.67. Für drei Klassen beträgt der kleinstmögliche geglättete Verlust etwa 0.29. Er wird erreicht, wenn das größte Logit die anderen um etwa 3.3 übertrifft. Eine Margin von 10 wird somit als übermäßige Sicherheit bestraft. Der Gradient wechselt das Vorzeichen und verringert den Abstand zwischen extremen Logits wieder auf ein moderates Maß.

<!--en:67ba5751-->
### Kosinuspläne mit Aufwärmphase

<!--en:8df91990-->
Herkömmliche Stufenpläne halten die Lernrate Dutzende Epochen lang konstant und senken sie dann abrupt um den Faktor 10. Damit sind auch die Epochen, in denen die Lernrate fällt, als Hyperparameter abzustimmen. Das heute übliche Verfahren :cite:`Loshchilov.Hutter.2016` kennen wir bereits seit AlexNet (:numref:`sec_alexnet`). In einer linearen *Aufwärmphase* (engl. warmup) steigt die Lernrate von 0 auf den Höchstwert $\eta_0$. Das dauert $t_\textrm{w}$ Schritte. Anschließend folgt ein glatter *Abfall* entlang einer halben Kosinusperiode bis auf 0 bei Schritt $T$. Die Aufwärmphase verhindert frühe Divergenz. Außerdem gibt sie den gleitenden Statistiken der Batchnormalisierung und den Momenten adaptiver Optimierer Zeit, sich vor den ersten Schritten voller Größe zu stabilisieren. Mit dem Kosinus-Abfall entfallen die abrupten Sprünge ganz. Wir plotten `step_decay` (30 Epochen, davon zwei zum Aufwärmen) und `d2l.cosine_warmup` (45 Epochen, davon drei zum Aufwärmen) bei 79 Minibatches pro Epoche. Das sind genau die Zeiträume unseres späteren Experiments in diesem Abschnitt.

<!--en:1c567062-->
### Mixup

<!--en:d8821c75-->
*Mixup* :cite:`zhang2018mixup` regularisiert einen Klassifikator, indem es ihn auf Konvexkombinationen von je zwei Trainingsbeispielen trainiert. Für jeden Minibatch zieht es einen Mischkoeffizienten $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$. Durch zufälliges Mischen ordnet es jedem Beispiel $i$ ein Partnerbeispiel $j$ zu und bildet

<!--en:b675a511-->
Dabei sind $\mathbf{y}_i$ und $\mathbf{y}_j$ One-Hot-Klassenvektoren. Somit ist $\tilde{\mathbf{y}}$ eine weiche Zielverteilung. Da die Kreuzentropie linear in der Zielverteilung ist, entspricht der Verlust für $\tilde{\mathbf{y}}$ der mit $\lambda$ gewichteten Summe der Kreuzentropien bezüglich $y_i$ und $y_j$. Diese Summe lässt sich einfacher und schneller berechnen. Wenn das Netz seine Vorhersagen zwischen je zwei Trainingsbeispielen linear interpolieren soll, prägt es sich einzelne Punkte weniger scharf ein. Bei den üblichen kleinen Werten ($\alpha \in [0.1, 0.2]$) konzentriert sich die Beta-Verteilung nahe 0 und 1. Die meisten Minibatches ändern sich daher nur wenig, einige wenige werden stark gemischt. Wir implementieren `mixup` mit einem gemeinsamen $\lambda$ pro Minibatch und speichern die Funktion in `d2l`, um sie in :numref:`sec_convnext` erneut zu verwenden.

<!--en:5a3a29de-->
Um den Effekt sichtbar zu machen, mischen wir einen Minibatch von Kleidungsstücken aus Fashion-MNIST mit $\alpha = 2$. Dadurch liegt $\lambda$ meist nahe 0.5 (beim Training ist $\alpha$ deutlich kleiner). Die obere Zeile zeigt die Originalbilder, die untere ihre Überlagerungen.

<!--en:0ac1df1d-->
Mixup erzeugt geisterhafte Doppelbelichtungen, wie sie auf realen Fotos nicht vorkommen. *CutMix* :cite:`yun2019cutmix` mischt deshalb räumlich. Es schneidet ein Rechteck aus einem Bild aus, setzt es in ein anderes ein und mischt die Labels entsprechend dem Flächenanteil des Rechtecks. So bleibt die lokale Umgebung jedes Pixels realistisch. Viele moderne Trainingsrezepte für die Bilderkennung wählen für jeden Minibatch zufällig zwischen Mixup und CutMix (siehe die Übungen unten).

<!--en:8d5fe393-->
### Stochastische Tiefe

<!--en:4faa31ef-->
Dropout (:numref:`sec_dropout`) setzt beim Training einzelne Aktivierungen zufällig auf null. *Stochastische Tiefe* :cite:`huang2016stochasticdepth` lässt ganze Residualzweige zufällig aus. Mit Wahrscheinlichkeit $p$ wird der Residualzweig eines Blocks umgangen, sodass der Block seine Eingabe unverändert weitergibt. Damit trainiert man effektiv ein Ensemble flacherer Teilnetze. Residualverbindungen ermöglichen dies, weil der Identitäts-Shortcut das Signal stets weiterleitet. Für einen Block, der $\mathbf{x} + f(\mathbf{x})$ berechnet, ersetzt die stochastische Tiefe den Zweig durch

<!--en:81025784-->
Wie beim invertierten Dropout hält die Skalierung mit $1/(1-p)$ den Erwartungswert der Zweigausgabe beim Training konstant. Bei der Auswertung entfällt daher eine erneute Skalierung. Moderne Implementierungen ziehen $b$ unabhängig für jedes Beispiel im Minibatch (unten als `drop_path` für ConvNeXt in `d2l` gespeichert). Wir leiten eine Unterklasse von `d2l.Residual` (:numref:`sec_resnet`) ab und lassen den Residualzweig unmittelbar vor der Addition der Sprungverbindung ausfallen. Verarbeiten wir im Trainingsmodus 1000 zufällige Eingaben mit diesem Block und setzen dabei $p = 0.5$, sollte etwa die Hälfte der Ausgaben exakt mit $\mathrm{ReLU}(\mathbf{x})$ übereinstimmen.

<!--en:245ba973-->
MXNet speichert den Trainingsmodus in einem globalen Flag. Beim Ausführen des Netzes liest `npx.dropout` dieses Flag auch innerhalb eines hybridisierten Netzes aus. In MXNet entspricht `drop_path` einem Dropout mit einer Ziehung pro Beispiel und benötigt daher kein Argument `training`.

<!--en:e3bef312-->
In tiefen Architekturen steigt die Ausfallwahrscheinlichkeit $p$ üblicherweise linear von 0 im ersten Block auf einen Höchstwert zwischen 0.1 und 0.5 im tiefsten Block. Die Maskierung pro Beispiel wertet weiterhin jeden Zweig aus, bevor sie die Ausgaben für die ausgelassenen Beispiele auf null setzt. Sie regularisiert also, ohne das Training zu beschleunigen. Die ursprüngliche Variante für ganze Minibatches zieht pro Block nur ein $b$ und kann ausgelassene Zweige vollständig überspringen, um Rechenaufwand zu sparen.

<!--en:51464ebd-->
### Mittelung der Gewichte

<!--en:1f45044f-->
Das Verfahren in der letzten Zeile von :numref:`tab_recipe_2015_2022` verbessert die Generalisierung bei nahezu unveränderten Kosten. Gegen Trainingsende schwanken die Parameter durch das Minibatch-Rauschen am Boden einer Senke der Verlustfunktion. Das Mittel der letzten Iterierten liegt daher näher am Zentrum der Senke als jeder einzelne Checkpoint. Die *stochastische Gewichtsmittelung* (engl. stochastic weight averaging, SWA) :cite:`Izmailov.Podoprikhin.Garipov.ea.2018` mittelt Momentaufnahmen aus der späten Trainingsphase mit gleichem Gewicht. Moderne Trainingsrezepte für die Bilderkennung verwenden häufig einen *exponentiellen gleitenden Mittelwert* (engl. exponential moving average, EMA). Er folgt derselben Rekursion wie die gleitenden Statistiken der Batchnormalisierung (:numref:`sec_batch_norm`). Nach jedem Optimiererschritt fließen in eine *Schattenkopie* $\bar{\theta}$ der trainierbaren Parameter die aktuellen Gewichte $\theta$ mit dem Abklingfaktor $\rho$ ein:

<!--en:707dbcc2-->
Die Parameter von vor $k$ Schritten gehen mit dem Gewicht $(1-\rho)\rho^k$ ein. Daraus ergibt sich ein effektives Zeitfenster von ungefähr $1/(1-\rho)$ Schritten (100 Schritte für $\rho = 0.99$). Die Optimierung aktualisiert wie gewohnt $\theta$, zur Auswertung setzen wir $\bar{\theta}$ ein. (Die gleitenden Statistiken der Batchnormalisierung sind in Puffern gespeichert und gehören nicht zu den trainierbaren Gewichten. Eine der folgenden Übungen untersucht, wie sie sich am besten mit EMA-Gewichten kombinieren lassen.) Um zu sehen, wie sich die Wahl von $\rho$ auswirkt, verfolgen wir ein einzelnes synthetisches Gewicht. Es steigt über 50 Schritte von 0 auf 1 und schwankt dann zufällig um 1.

<!--en:18a8712b-->
Bei $\rho = 0.9$ (Zeitfenster etwa 10 Schritte) hinkt das Schattengewicht dem Anstieg um etwa 10 Schritte hinterher. Anschließend dämpft es das Rauschen deutlich. Bei $\rho = 0.99$ (Zeitfenster etwa 100 Schritte) verläuft die Kurve noch glatter, folgt dem Gewicht aber so langsam, dass sie nach 200 Schritten noch ungefähr 20% unter 1 liegt. Das veranschaulicht den wichtigsten Nachteil der EMA. Ist $1/(1-\rho)$ im Vergleich zur Trainingsdauer groß, folgt die Schattenkopie den sich verbessernden Gewichten zu langsam und schneidet schlechter ab als der letzte Checkpoint. Sehr große Abklingfaktoren wie 0.999 oder 0.9999 eignen sich daher nur für lange Trainingsläufe.

<!--en:c75ec7c4-->
## Zwei Rezepte für dasselbe Netz

<!--en:2887d925-->
Setzen wir die Teile zusammen und vergleichen wir die Rezepte direkt an einer festen Architektur. Dazu bauen wir das ResNet-18 aus :numref:`sec_resnet` mit `d2l.Residual` und He-Initialisierung (:numref:`sec_numerical_stability`) nach. Wir trainieren auf einer zufälligen Teilmenge von 10 000 Fashion-MNIST-Bildern, deren Größe wir auf $96 \times 96$ anpassen. Innerhalb jedes Laufs verwenden beide Rezepte dieselben Bilder. Mit weniger Trainingsbeispielen kommt die Wirkung der Regularisierung stärker zum Tragen, und die Experimente lassen sich schnell wiederholen. Beide Rezepte berechnen die Lernrate für jeden Schritt direkt aus `lr` und `num_steps`.

<!--en:fea68a3a-->
Beide Modelle verwenden genau dieselbe Architektur und dieselben Trainingsdaten. Das Rezept für Optimierung und Regularisierung ändert sich dagegen vollständig. *Rezept A* entspricht der Praxis von 2015. Wir verwenden SGD mit Momentum 0.9 und gekoppeltem Gewichtszerfall $5 \times 10^{-4}$. Nach zwei Epochen linearer Aufwärmphase erreicht die Lernrate ihr Maximum von 0.05. Bei 60% und 85% der Trainingsdauer sinkt sie jeweils um den Faktor 10. Wir trainieren 30 Epochen lang mit dem gewöhnlichen Kreuzentropieverlust. (Die kurze Aufwärmphase von zwei Epochen ist eine kleine, neuere Maßnahme zur Stabilisierung. :citet:`Goyal.Dollar.Girshick.ea.2017` führten sie für SGD mit großen Minibatches ein. Ohne sie divergierte SGD mit Batchnormalisierung bei einer Lernrate von 0.05 in etwa einem Drittel unserer Läufe schon in Epoche 1.) Unser *kompaktes modernes Rezept* verwendet AdamW mit Gewichtszerfall 0.05, eine maximale Lernrate von 0.002 mit drei Epochen Aufwärmphase und anschließendem Kosinus-Abfall, Label-Smoothing ($\epsilon = 0.1$) und Mixup ($\alpha = 0.2$). Es trainiert 45 Epochen lang, weil moderne Regularisierung längeres Training erfordert. Damit die Laufzeit überschaubar bleibt, prüfen wir nur diese Auswahl und verzichten auf RandAugment, CutMix, stochastische Tiefe und EMA.

<!--en:f5fb9f2a-->
Der Faktor 100 zwischen den Koeffizienten für den Gewichtszerfall (0.05 gegenüber $5 \times 10^{-4}$) kann täuschen. Um welchen Bruchteil die Gewichte in einem Schritt tatsächlich schrumpfen, ergibt sich aus dem Produkt von Lernrate und Gewichtszerfall. Bei maximaler Lernrate ergibt das $0.002 \times 0.05 = 10^{-4}$ für AdamW gegenüber $0.05 \times 5 \times 10^{-4} = 2.5 \times 10^{-5}$ für SGD. Das Momentum verstärkt den Zerfall bei SGD bis auf das Zehnfache, sodass er bei SGD effektiv sogar stärker ist. Beide Rezepte berechnen den Validierungsverlust als gewöhnliche Kreuzentropie auf ungemischten Validierungsbildern. Label-Smoothing und Mixup ändern nur das Trainingsziel.

<!--en:78e3dda8-->
Zunächst trainieren wir das Netz nach Rezept A über 30 Epochen und werten es auf allen 10 000 Validierungsbildern aus.

<!--en:325c0fac-->
Als Nächstes trainieren wir dieselbe Architektur auf derselben Teilmenge der Trainingsdaten nach dem kompakten modernen Rezept über 45 Epochen.

<!--en:8a5de035-->
:numref:`tab_recipe_results` zeigt die Spannweite der Validierungsgenauigkeit über mehrere Läufe. Für jeden Lauf ziehen wir erneut 10 000 Bilder, initialisieren die Parameter zufällig und mischen die Reihenfolge der Minibatches neu.

<!--en:8c39c630-->
:Dasselbe ResNet-18 mit beiden Rezepten (10 000 Trainingsbilder aus Fashion-MNIST in der Größe $96 \times 96$, Spannweite der Validierungsgenauigkeit über mehrere Läufe).

<!--en:156e6023-->
| Rezept | Epochen | Validierungsgenauigkeit |

<!--en:ce822964-->
| A (2015): SGD, Aufwärmphase, stufenweiser Abfall | 30 | 89–90.5% |

<!--en:55920dc8-->
| Kompaktes modernes Rezept: AdamW, Kosinus-Abfall, Glättung, Mixup | 45 | 90.5–91% |

<!--en:3710cf17-->
Bei jedem paarweisen Vergleich auf derselben Teilmenge von 10 000 Bildern übertrifft `ModernRecipe` die Genauigkeit von `ClassicRecipe` um 0.5–1.5 Prozentpunkte, selbst ohne RandAugment, CutMix, stochastische Tiefe und EMA. Aus diesem Vergleich lassen sich drei Lehren für die Praxis ziehen:

<!--en:f3391242-->
1. **Vergleich ganzer Rezepte und Ablationsstudien:** Optimierer, Lernratenplan, Verlust, Augmentierung und Epochenzahl ändern sich gemeinsam. Nur wenn wir jeweils eine einzelne Zutat ändern, können wir feststellen, welche am meisten beigetragen hat (siehe :exref:`ex_training-recipes-ablation`). Die optimalen Lernraten unterscheiden sich um den Faktor $25\times$ zwischen AdamW ($0.002$) und SGD ($0.05$). Bei jedem Wechsel des Optimierers müssen wir die Lernrate daher neu abstimmen.

<!--en:6ca54307-->
2. **Trainingsverluste verschiedener Rezepte nie direkt vergleichen:** Bei `ClassicRecipe` fällt der Trainingsverlust gegen null, während der Validierungsverlust wieder auf 0.4–0.5 steigt, weil die Logits übermäßige Sicherheit ausdrücken. Bei `ModernRecipe` sinkt der Validierungsverlust unter 0.4. Der Trainingsverlust bleibt *höher* als der Validierungsverlust, weil Mixup und Label-Smoothing im Training gemischte Beispiele mit weichen Zielwerten bewerten.

<!--en:fc4d6290-->
3. **Veröffentlichte Ranglisten einordnen:** Allein das Trainingsrezept verändert die ImageNet-Genauigkeit eines unveränderten ResNet-50 um $>4$ Prozentpunkte :cite:`wightman2021resnet`. Bei einem Vorsprung von 1 Prozentpunkt gegenüber einer Baseline aus einer älteren Arbeit lassen sich die Beiträge von Rezept und Architektur daher nicht trennen. Selbst bei gleichen Rezepten erschweren Rauschen im Testdatensatz und mehrdeutige ImageNet-Bilder mit mehreren Objekten die Beurteilung :cite:`Recht.Roelofs.Schmidt.ea.2019,beyer2020imagenetreal`. Die Übertragung auf Objekterkennung oder Segmentierung prüft die Qualität des Grundnetzes oft genauer als Unterschiede von weniger als einem Prozentpunkt in der Top-1-Genauigkeit auf ImageNet.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:9f75b629-->
Zwischen 2015 und den frühen 2020er-Jahren änderten sich die Trainingsrezepte für konvolutionäre Netze in jeder Hinsicht. Hinzu kamen adaptive Optimierer mit entkoppeltem Gewichtszerfall (AdamW, LAMB), Kosinuspläne mit Aufwärmphase, RandAugment/Mixup/CutMix, Label-Smoothing, stochastische Tiefe, längere Trainingszeiten und die Mittelung der Gewichte per EMA. Eines dieser Rezepte steigerte die ImageNet-Genauigkeit eines unveränderten ResNet-50 von 76.1% auf 80.4%. Es verwendete LAMB, einen Kosinusplan, RandAugment mit Mixup und CutMix, stochastische Tiefe und 600 Epochen ohne Mittelung der Gewichte. Für jede Zutat genügen wenige Codezeilen. Schon eine kleine Auswahl verbessert zuverlässig die Genauigkeit von ResNet-18 auf `FashionMNIST10k`. In :numref:`sec_convnext` halten wir nun das moderne Rezept konstant und modernisieren Schritt für Schritt die Architektur von ResNet selbst.

<!--en:2294243a-->
## Übungen

<!--en:73308c8d-->
1. **Binäre Kreuzentropie und Mixup.** :citet:`wightman2021resnet` trainieren ImageNet-Klassifikatoren mit binärer Kreuzentropie. Sie behandeln jede der $K$ Klassen als unabhängige binäre Klassifikationsaufgabe mit Sigmoidfunktion. Damit ersetzen sie die Kreuzentropie einer Softmax-Funktion über $K$ Klassen. Erklären Sie, warum unabhängige Sigmoidausgaben gut zu Mixup und CutMix passen, wenn ein gemischtes Bild gleichzeitig Objekte aus zwei verschiedenen Klassen enthält.

<!--en:266b1851-->
1. [code] **CutMix.** CutMix :cite:`yun2019cutmix` zieht $\lambda \sim \mathrm{Beta}(1, 1) = \mathrm{Uniform}(0, 1)$ und fügt einen zufälligen rechteckigen Ausschnitt aus einem durch Mischen zugeordneten Partnerbild ein. Der angestrebte Flächenanteil beträgt $1 - \lambda$. Implementieren Sie `cutmix` als direkt einsetzbare Variante von `mixup`. Mischen Sie die One-Hot-Labels anhand des *exakten* Anteils der erhaltenen Pixel, nachdem Sie das Rechteck an den Bildrändern abgeschnitten haben. Trainieren Sie `ModernRecipe` auf `FashionMNIST10k` einmal mit CutMix anstelle von Mixup. Wählen Sie in einem zweiten Lauf für jeden Minibatch zufällig zwischen Mixup und CutMix. Vergleichen Sie die Validierungsgenauigkeit mit :numref:`tab_recipe_results`.

<!--en:7be79003-->
1. [code] **Zeitfenster der Gewichtsmittelung (EMA).** Integrieren Sie die Klasse `EMA` in `ModernRecipe`. Aktualisieren Sie dazu nach jedem Optimiererschritt die Schattengewichte und bestimmen Sie die Validierungsgenauigkeit mit den Schattenparametern. Erproben Sie die Abklingfaktoren $\rho \in \{0.9, 0.99, 0.999\}$ in :eqref:`eq_ema`. Setzen Sie den besten Wert von $\rho$ zum effektiven Zeitfenster $1/(1-\rho)$ und zur Gesamtzahl der Schritte ($45 \times 79 = 3{,}555$) in Beziehung. Erklären Sie, wie mit den gleitenden Statistiken der Batchnormalisierung bei der Auswertung mit EMA-Gewichten zu verfahren ist.

<!--en:26db2a61-->
1. [code] **Ablationsstudie mit einzelnen Zutaten.** ● In :numref:`tab_recipe_results` übertrifft `ModernRecipe` die Genauigkeit von `ClassicRecipe` um 0.5–1.5 Prozentpunkte, wenn sich alle fünf Zutaten zugleich ändern.
    1. Führen Sie `ClassicRecipe` und `ModernRecipe` jeweils fünfmal auf `FashionMNIST10k` aus. Schätzen Sie daraus die Standardabweichung der Validierungsgenauigkeit zwischen den Läufen.
    1. Gehen Sie von `ModernRecipe` aus und setzen Sie jeweils nur eine Zutat auf die Variante aus `ClassicRecipe` zurück: (i) Entfernen Sie Mixup, (ii) entfernen Sie Label-Smoothing, (iii) ersetzen Sie den Kosinusplan mit Aufwärmphase durch `step_decay`, (iv) ersetzen Sie AdamW durch SGD mit Momentum und passen Sie die maximale Lernrate auf $0.05$ an, (v) verkürzen Sie das Training von 45 auf 30 Epochen. Welche einzelnen Änderungen übersteigen die zufälligen Schwankungen zwischen Läufen?
    1. Vervollständigen Sie die $2 \times 2$-Tabelle für Optimierer (SGD oder AdamW) $\times$ Lernratenplan (`step_decay` oder `cosine_warmup`) bei aktiviertem Mixup und Label-Smoothing. Gibt es eine Wechselwirkung zwischen Optimierer und Lernratenplan?
   :exlabel:`ex_training-recipes-ablation`

<!--en:b10df53f-->
1. [code] **Stochastische Tiefe im Trainingsrezept.** ● Unser kompaktes modernes Rezept verzichtet auf stochastische Tiefe. Ergänzen Sie sie in `ResNet18`:
    1. Erweitern Sie `StochasticResidual` um Downsampling-Blöcke mit Stride 2 und $1 \times 1$-Sprungverbindungen. Bauen Sie `ResNet18` mit Ausfallwahrscheinlichkeiten, die von $0$ in Block 1 linear auf $p_{\max}$ in Block 8 steigen. Wie hängt der Erwartungswert der Anzahl aktiver Residualzweige pro Vorwärtsdurchlauf von $p_{\max}$ ab?
    1. Testen Sie $p_{\max} \in \{0.1, 0.2, 0.5\}$ mit `ModernRecipe` auf `FashionMNIST10k` und vergleichen Sie die Ergebnisse mit $p_{\max} = 0$. Warum profitiert ein flaches Netz mit 8 Blöcken weniger von großen Werten für $p_{\max}$ als ein Netz mit 50 oder 100 Blöcken?
    1. Implementieren Sie die Minibatch-Variante von :citet:`huang2016stochasticdepth`. Sie zieht für jeden Block eine einzige Bernoulli-Variable $b$ und überspringt für $b = 0$ die Berechnung des Residualzweigs vollständig. Vergleichen Sie bei $p_{\max} = 0.5$ den theoretischen Anteil übersprungener Zweige mit der gemessenen relativen Verkürzung der Trainingszeit.

<!--en:1bf530dd-->
[Diskussionen](https://d2l.discourse.group/t/36761)

<!--en:8e1d1a79-->
Moderne Trainingsrezepte für konvolutionäre Netze<br>**AdamW · Kosinusplan mit Aufwärmphase · Label-Smoothing · Mixup und CutMix · stochastische Tiefe · EMA der Gewichte**

<!--en:967ae85e-->
::: {.slide title="Warum Fortschritte bei der Architektur oft auf das Trainingsrezept zurückgehen"}

<!--en:41a9af55-->
[Motivation und Störfaktoren in Benchmarks]{.kicker}

<!--en:38eafcaf-->
- **ResNet-50 Strikes Back (Wightman et al., 2021):** Ein neues Trainingsrezept steigert die ImageNet-Genauigkeit eines *unveränderten* ResNet-50 von 2015 **von $76.1\%$ auf $80.4\%$** (+4.3 Prozentpunkte). Es nutzt LAMB, binäre Kreuzentropie als Verlust, RandAugment/Mixup/CutMix und 600 Epochen.

<!--en:ea9cc06f-->
- **Wissenschaftliche Grundregel:** Architekturen immer mit denselben Trainingsrezepten vergleichen. Bei einem Gewinn von 1–2 Prozentpunkten gegenüber einer nicht neu trainierten Baseline lassen sich die Beiträge von Rezept und Architektur nicht trennen.

<!--en:47d63efd-->
::: {.slide title="Trainingsrezepte für die Bilderkennung 2015 und 2022"}

<!--en:12640fed-->
[Zutaten moderner Trainingsrezepte]{.kicker}

<!--en:620a13b1-->
- **Daten und Vorwärtsdurchlauf:** RandAugment + Mixup + CutMix + stochastische Tiefe (`drop_path`).

<!--en:c48ef72c-->
- **Zielfunktion und Optimierer:** Label-Smoothing ($\epsilon = 0.1$) + AdamW oder LAMB (entkoppelter Gewichtszerfall) + lineare Aufwärmphase und Abfall entlang einer halben Kosinusperiode über insgesamt 300–600 Epochen.

<!--en:eff40975-->
- **Checkpoint:** Häufig eine Schattenkopie mit dem exponentiellen gleitenden Mittelwert (EMA) $\bar{\boldsymbol{\theta}}_T$. Der Lauf mit $80.4\%$ für ResNet-50 kam ohne sie aus.

<!--en:b09bebf5-->
::: {.slide title="Gewichtszerfall bei AdamW und SGD: entkoppelt und gekoppelt"}

<!--en:9d99e476-->
[Funktionsweise der Optimierer]{.kicker}

<!--en:cf6dabdc-->
- **Warum L2 bei Adam versagt:** Die Division von $\lambda \boldsymbol{\theta}_t$ durch $\sqrt{\hat{\mathbf{v}}_t}$ verkleinert Gewichte mit großem Gradienten *weniger*.

<!--en:e1d8c97f-->
- **Vergleich zwischen Optimierern:** Maßgeblich ist die relative Abnahme $\eta_0 \lambda$ pro Schritt ($0.002 \times 0.05 = 10^{-4}$ bei AdamW).

<!--en:ae58226a-->
::: {.slide title="Wie Label-Smoothing die optimale Logit-Margin begrenzt"}

<!--en:f2895473-->
[Regularisierung über den Verlust]{.kicker}

<!--en:4066a510-->
Bei gewöhnlicher One-Hot-Kreuzentropie ($-\log p_y$) gilt für das Logit der richtigen Klasse $z_y \to +\infty$, wenn der Verlust gegen null geht. Das führt zu übermäßig sicheren, schlecht kalibrierten Vorhersagen:

<!--en:0c324d55-->
- **Geglättete Zielverteilung:** *Label-Smoothing* (Szegedy et al., 2016) weist die Wahrscheinlichkeit $1 - \epsilon$ der richtigen Klasse $y$ zu und verteilt $\epsilon$ gleichmäßig auf alle $K$ Klassen (je $\epsilon / K$):
  $$l_\epsilon(\mathbf{p}, y) = -(1 - \epsilon)\log p_y - \frac{\epsilon}{K} \sum_{k=1}^K \log p_k$$

<!--en:c63fe72c-->
- **Endlicher optimaler Logit-Abstand:** Der Verlust ist minimal bei $p_y^* = 1 - \epsilon + \epsilon/K$ und $p_k^* = \epsilon/K$ ($k \neq y$). Dies entspricht genau der endlichen Logit-Margin $z_y^* - z_k^* = \log\bigl(1 + \frac{K(1-\epsilon)}{\epsilon}\bigr)$ (z. B. $\approx 3.33$ für $K=3, \epsilon=0.1$).

<!--en:20733a37-->
- **Strafe für übermäßige Sicherheit:** Bei den Logits $(10, 0, 0)$ erhöht $\epsilon=0.1$ den Verlust von $0.0001$ auf $0.67$ und führt extreme Logits wieder zusammen:

<!--en:a4e33015-->
::: {.slide title="Stufenweiser Abfall und Kosinus-Abfall mit linearer Aufwärmphase"}

<!--en:3f8bdcbc-->
[Lernratenpläne]{.kicker}

<!--en:f56fcb4e-->
Der Vergleich unseres Stufenplans über 30 Epochen mit dem Kosinusplan mit Aufwärmphase über 45 Epochen zeigt, warum moderne Rezepte auf abrupte Lernratensprünge verzichten:

<!--en:cda8041a-->
- **Aufwärmphase ($0 \le t < t_{\textrm{w}}$):** Die Lernrate steigt linear gemäß $\eta_t = \eta_0 \cdot (t / t_{\textrm{w}})$. Die gleitenden Statistiken der Batchnormalisierung und die zweiten Momente $\hat{\mathbf{v}}_t$ von AdamW können sich vor den ersten Schritten voller Größe stabilisieren.

<!--en:c2642ee2-->
- **Abfall entlang einer halben Kosinusperiode ($t_{\textrm{w}} \le t \le T$):**
  $$\eta_t = \frac{\eta_0}{2}\left(1 + \cos\frac{\pi(t - t_{\textrm{w}})}{T - t_{\textrm{w}}}\right)$$
  Die Lernrate sinkt glatt auf $0$ bei Schritt $T$. Damit entfällt die Wahl der Epochen für den stufenweisen Abfall.

<!--en:dc7cd99a-->
::: {.slide title="Wie Mixup und CutMix Beispiele und Labels mischen"}

<!--en:3593950d-->
[Datenaugmentierung]{.kicker}

<!--en:f5eb25af-->
- **Konvexkombinationen ($\lambda \sim \mathrm{Beta}(\alpha, \alpha)$):** Mixup überlagert ganze Bilder. CutMix fügt einen rechteckigen Ausschnitt mit Flächenanteil $1-\lambda$ ein und mischt die Labels nach Flächenanteil.

<!--en:f8066232-->
::: {.slide title="Mixup im Bild und im Vergleich mit CutMix"}

<!--en:71b43dec-->
[Geometrie der Augmentierung]{.kicker}

<!--en:e1585c10-->
Beim Training konzentriert $\alpha = 0.2$ den Mischkoeffizienten $\lambda$ nahe $0$ und $1$. Für die Darstellung wählen wir $\alpha = 2.0$ ($\lambda \approx 0.5$), damit die überlagerten Kleidungsstücke deutlich erkennbar sind:

<!--en:61e293f8-->
- **Lineare Entscheidungsgrenzen:** Mixup zwingt die Vorhersagen des Netzes zu linearen Übergängen zwischen Klassen und verhindert starke Schwankungen außerhalb der Mannigfaltigkeit der Trainingsdaten.

<!--en:5bb91583-->
- **CutMix (Yun et al., 2019):** Da die gewichtete Überlagerung ganzer Bilder geisterhafte Doppelbelichtungen erzeugt, setzt CutMix ein Rechteck mit Flächenanteil $1 - \lambda$ aus $\mathbf{x}_j$ in $\mathbf{x}_i$ ein. Es mischt die Labels nach dem exakten Flächenanteil und erhält die lokale Pixelstatistik.

<!--en:67a18c6f-->
::: {.slide title="Stochastische Tiefe durch Auslassen ganzer Residualzweige"}

<!--en:f8bf7148-->
[Strukturelle Regularisierung]{.kicker}

<!--en:a5b18102-->
- **Implizites Ensemble unterschiedlich tiefer Netze:** Für jedes Beispiel entfallen ganze Residualzweige nach dem linearen Plan $p_\ell = (\ell/L)p_{\max}$.

<!--en:5cbb04ea-->
- **Erhaltung des Erwartungswerts:** Die Skalierung mit $1/(1-p_\ell)$ sichert $\mathbb{E}[\mathbf{x}_{\ell+1}] = \mathbf{x}_\ell + f(\mathbf{x}_\ell)$. Für die Auswertung gilt daher $b \equiv 1$.

<!--en:b313c488-->
::: {.slide title="Implementierung und Überprüfung von `drop_path`"}

<!--en:122bdc92-->
[Code zur stochastischen Tiefe]{.kicker}

<!--en:0e46f917-->
::: {.slide title="Implementierung und Überprüfung von `drop_path` (Fortsetzung)"}

<!--en:122bdc92-2-->
[Code zur stochastischen Tiefe]{.kicker}

<!--en:596e3671-->
Wir ziehen eine Bernoulli-Maske pro Beispiel (`shape = (B, 1, 1, 1)`) und verarbeiten 1000 Eingaben mit `StochasticResidual(p=0.5)`. Wie erwartet umgeht die Hälfte der Beispiele den Zweig. Die entsprechenden Ausgaben stimmen exakt mit $\textrm{ReLU}(\mathbf{X})$ überein:

<!--en:5de347cc-->
::: {.slide title="Exponentieller gleitender Mittelwert (EMA) der Modellgewichte"}

<!--en:9570a38b-->
[Mittelung der Iterierten]{.kicker}

<!--en:5bd29606-->
Nahe der Konvergenz lässt das Minibatch-Rauschen die Iterierten $\boldsymbol{\theta}_t$ von SGD und AdamW in einem flachen Tal der Verlustfunktion hin und her schwanken. Eine nicht trainierbare Schattenkopie $\bar{\boldsymbol{\theta}}_t$ mittelt diese Schwankungen aus:

<!--en:b95c5816-->
- **EMA-Aktualisierung nach jedem Optimiererschritt:**
  $$\bar{\boldsymbol{\theta}}_t \leftarrow \rho\,\bar{\boldsymbol{\theta}}_{t-1} + (1 - \rho)\,\boldsymbol{\theta}_t$$

<!--en:d2c9e65f-->
- **Effektives Zeitfenster $1/(1-\rho)$:** Die Iterierte $\boldsymbol{\theta}_{t-k}$ geht mit dem Gewicht $(1-\rho)\rho^k$ ein.

<!--en:c8a46f64-->
- **Fallstrick bei langen Zeitfenstern:** Bei $\rho = 0.99$ (Zeitfenster $100$ Schritte) liegt die Schattenkopie selbst nach 200 Schritten noch $20\%$ unter dem Endwert eines 50 Schritte langen Anstiegs. Stets $1/(1-\rho) \ll T$ wählen (z. B. $\rho = 0.9999$ nur bei Läufen mit $>50{,}000$ Schritten).

<!--en:9b9ffadb-->
::: {.slide title="Kontrolliertes Experiment: ein ResNet-18, zwei Trainingsrezepte"}

<!--en:b2dd487e-->
[Direkter Vergleich]{.kicker}

<!--en:5ff67b96-->
- **Dasselbe ResNet-18 mit 11.2 Mio. Parametern auf `FashionMNIST10k`:** Vergleich von `ClassicRecipe` (30 Epochen, SGD mit $\eta_0=0.05$, stufenweiser Abfall, One-Hot-Kreuzentropie) mit `ModernRecipe` (45 Epochen, AdamW mit $\eta_0=0.002$, Kosinusplan mit Aufwärmphase, Mixup mit $0.2$, Label-Smoothing mit $0.1$).

<!--en:a31b360e-->
- **Ergebnis:** `ModernRecipe` gewinnt jeden paarweisen Vergleich mit **+0.5 bis +1.5 Prozentpunkten** ($90.5\text{--}91.0\%$ gegenüber $89.0\text{--}90.5\%$).

<!--en:c4ffaa83-->
::: {.slide title="Implementierung klassischer und moderner Trainingsschritte" except="tensorflow"}

<!--en:bf90e49c-->
[Implementierung der Rezepte]{.kicker}

<!--en:1d35e97e-->
`ClassicRecipe` und `ModernRecipe` verwenden dasselbe `ResNet18` und unterscheiden sich nur in zwei Methoden:

<!--en:34f62700-->
::: {.slide title="Implementierung der Rezepte: Lernratenplan und gekoppelter Gewichtszerfall" only="tensorflow"}

<!--en:bf90e49c-2-->
[Implementierung der Rezepte]{.kicker}

<!--en:1d35e97e-2-->
`ClassicRecipe` und `ModernRecipe` verwenden dasselbe `ResNet18` und unterscheiden sich nur in zwei Methoden:

<!--en:399c317b-->
::: {.slide title="Implementierung klassischer und moderner Trainingsschritte" only="tensorflow"}

<!--en:9d4a1416-->
::: {.slide title="Verlauf der Baseline: 30 Epochen mit dem klassischen Rezept A"}

<!--en:187bcb15-->
[Trainingskurve des klassischen Rezepts]{.kicker}

<!--en:6636b2c2-->
Bei `ClassicRecipe` (30 Epochen, ungeglättete Kreuzentropie ohne Mixup) fällt der Trainingsverlust gegen null. Der Validierungsverlust steigt wieder auf $0.4\text{--}0.5$, weil die Logits übermäßige Sicherheit ausdrücken:

<!--en:7fc5bb6f-->
::: {.slide title="Fallstrick bei der Diagnose: Trainingsverluste nie direkt vergleichen"}

<!--en:7d41d3bb-->
[Trainingskurven im Vergleich]{.kicker}

<!--en:596a47ef-->
Die Trainings- und Validierungskurven von `ModernRecipe` zeigen einen Fallstrick, der in der Praxis häufig zu Fehlschlüssen führt:

<!--en:cf9543d0-->
- **Warum `train_loss > val_loss` bei `ModernRecipe` gilt:** `train_loss` bewertet mit Mixup gemischte Bilder anhand geglätteter Zielwerte. Der theoretisch kleinstmögliche Verlust liegt dabei deutlich über $0$. Dagegen ist `val_loss` die gewöhnliche Kreuzentropie auf unveränderten, ungemischten Validierungsbildern.

<!--en:26cbd97e-->
- **Bessere Kalibrierung:** Bei `ClassicRecipe` erreicht der Trainingsverlust $\approx 0$. Übermäßig sichere Vorhersagen treiben den Validierungsverlust wieder auf $0.4\text{--}0.5$. Mit `ModernRecipe` bleibt der Validierungsverlust unter $0.4$.

<!--en:7dbd1c20-->
::: {.slide title="Ranglisten einordnen: Ablationsstudien und Verteilungsverschiebung"}

<!--en:ca3f2751-->
[Methodik für Benchmarks]{.kicker}

<!--en:4178ac9b-->
- **1. Lernrate beim Wechsel des Optimierers immer neu abstimmen:** Bei AdamW ($\eta_0 = 0.002$) und SGD ($\eta_0 = 0.05$) unterscheiden sich die Lernraten um den Faktor $25\times$.

<!--en:6dcbe7d2-->
- **2. Verteilungsverschiebung bei ImageNet-V2 (Recht et al., 2019):** Die Genauigkeit aller Modelle sinkt um $11\text{--}14\%$. Die Beziehung zwischen alten und neuen Genauigkeiten ist linear, die Rangfolge bleibt erhalten.

<!--en:d3cba3a4-->
- **3. Übertragung auf Folgeaufgaben:** Nähert sich der Top-1-Fehler auf ImageNet seiner durch Mehrfachlabels bedingten Untergrenze, prüfen wir die Grundnetze mit Objekterkennung und Segmentierung auf COCO.

<!--en:c5697f19-->
::: {.slide title="Verständnisfragen zu geglätteten Logits, EMA und Gewichtszerfall"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:e2a1fdec-->
Prüfen Sie mit diesen drei Rechnungen, wie gut Sie die Wirkungsweise moderner Trainingsrezepte verstehen:

<!--en:e442ab94-->
1. **Optimaler Logit-Abstand bei Label-Smoothing:** Betrachten Sie $K = 10$ Klassen und $\epsilon = 0.1$. Welche Wahrscheinlichkeit $p_y^*$ weist Label-Smoothing der richtigen Klasse zu? Welche Wahrscheinlichkeit $p_k^*$ erhält jede der 9 falschen Klassen, und wie groß ist der optimale Logit-Abstand $z_y^* - z_k^* = \log(p_y^* / p_k^*)$?

<!--en:ffe9ea88-->
2. **EMA-Zeitfenster und Epochenzahl:** Unser Lauf auf `FashionMNIST10k` dauert 45 Epochen ($79$ Schritte pro Epoche, insgesamt $T = 3{,}555$ Schritte). Über wie viele Epochen mittelt eine EMA mit $\rho = 0.999$ effektiv, und warum kann das die Validierungsgenauigkeit am Trainingsende verschlechtern?

<!--en:d6f3adc5-->
3. **EMA und Batchnormalisierung:** Sie setzen zur Auswertung die EMA-Schattengewichte $\bar{\boldsymbol{\theta}}$ ein. Warum passen die gleitenden Statistiken $(\hat{\mu}, \hat{\sigma}^2)$ der Batchnormalisierung möglicherweise nicht ganz zu diesen Gewichten? Wie lässt sich das beheben?

<!--en:557b98b6-->
**Antworten:**

<!--en:fa7dfd6c-->
1. Es gilt $p_y^* = 1 - 0.1 + 0.1/10 = 0.91$ und $p_k^* = 0.1/10 = 0.01$ für jede der 9 anderen Klassen. Der optimale Logit-Abstand beträgt $\log(0.91 / 0.01) = \log(91) \approx \mathbf{4.51}$ (gegenüber $\infty$ ohne Glättung).

<!--en:7e314645-->
2. Das effektive Zeitfenster beträgt $1/(1 - 0.999) = 1{,}000$ Schritte $\approx \mathbf{12.7\text{ epochs}}$ (etwas mehr als ein Viertel des gesamten Laufs über 45 Epochen). Dadurch verschieben die frühen, noch unzureichend trainierten Gewichte den Mittelwert in ihre Richtung.

<!--en:3b92e61b-->
3. Die Batchnormalisierung hat `moving_mean` und `moving_var` mit den jeweils aktuellen Gewichten $\boldsymbol{\theta}_t$ und $\rho_{\text{BN}} = 0.9$ (Zeitfenster von 10 Schritten) berechnet. Die geglätteten Gewichte $\bar{\boldsymbol{\theta}}_t$ ergeben andere Statistiken. Führen Sie entweder EMA-Schattenkopien der Puffer der Batchnormalisierung mit oder durchlaufen Sie die Trainingsbatches einmal vorwärts mit festen Gewichten $\bar{\boldsymbol{\theta}}$, um die Statistiken neu zu berechnen.

<!--en:489f0ab3-->
::: {.slide title="Zusammenfassung: moderne Trainingsrezepte"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:08d2d255-->
| Zutat | Mathematische Formulierung | Wirkung und Wahl der Hyperparameter |

<!--en:39bd2fd3-->
| **AdamW** | $\boldsymbol{\theta}_{t+1} = (1 - \eta_t \lambda)\boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ | Entkoppelt die relative Gewichtsabnahme $\eta_t \lambda$ von der adaptiven Gradientennormierung. Basislernrate um den Faktor $\sim 25\times$ kleiner als bei SGD |

<!--en:d9fa35ad-->
| **Kosinusplan mit Aufwärmphase** | Linearer Anstieg $0 \to \eta_0$ über $t_{\textrm{w}}$ Schritte, dann $\frac{\eta_0}{2}(1 + \cos\frac{\pi(t-t_{\textrm{w}})}{T-t_{\textrm{w}}})$ | Verhindert frühe Divergenz bei Batchnormalisierung und Adam. Die Abstimmung der Zeitpunkte für Lernratensprünge entfällt |

<!--en:8b1ff5e7-->
| **Label-Smoothing** | $l_\epsilon = -(1-\epsilon)\log p_y - \frac{\epsilon}{K}\sum_k \log p_k$ | Begrenzt den optimalen Logit-Abstand auf $\log(1 + K(1-\epsilon)/\epsilon)$ und verbessert die Kalibrierung |

<!--en:82cac13a-->
| **Mixup / CutMix** | $\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j$, $\tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1-\lambda)\mathbf{y}_j$ | Regularisiert Übergänge zwischen Klassen und erfordert längeres Training (300–600 Epochen auf ImageNet) |

<!--en:6fb2a641-->
| **Stochastische Tiefe** | $\mathbf{x} + \frac{b}{1-p}f(\mathbf{x})$ mit $b \sim \mathrm{Bernoulli}(1-p)$ pro Beispiel | Lässt beim Training ganze Residualzweige aus. Mit zunehmender Tiefe steigt $p_\ell$ linear von $0$ auf $p_{\max}$ |

<!--en:23b98033-->
| **EMA der Gewichte** | $\bar{\boldsymbol{\theta}}_t \leftarrow \rho \bar{\boldsymbol{\theta}}_{t-1} + (1-\rho)\boldsymbol{\theta}_t$ | Dämpft Schwankungen gegen Trainingsende ohne zusätzliche Inferenzkosten. Zeitfenster so wählen, dass $1/(1-\rho) \ll T$ gilt |
