<!--en:ed404131-->
# Bedingte Generierung

<!--en:e739237a-->
Die Generatoren in :numref:`sec_basic_gan` und :numref:`sec_dcgan` sind *unbedingt* (engl. unconditional). Sie ziehen zufällige Samples aus der Randverteilung der Daten $p(x)$. Eine gewünschte Klasse, ein Textprompt oder ein Eingabebild lässt sich dabei nicht vorgeben. Kurz nach der ursprünglichen Arbeit über GANs führten :citet:`Mirza.Osindero.2014` *bedingte GANs* (engl. conditional GANs, cGANs) ein. Dabei erhalten sowohl der Generator als auch der Diskriminator eine Bedingungsvariable $c$ als Eingabe. Wir schreiben dafür $G(z, c)$ und $D(x, c)$.

<!--en:2b358c20-->
In diesem Abschnitt gehen wir drei Fragen nach:

<!--en:1b4716f0-->
1. Welche Zielfunktion optimiert die bedingte Formulierung, und welche Größe schätzt der optimale Kritiker?

<!--en:2485cd2c-->
2. Wie lässt sich die Bedingungsvariable $c$ in die Architektur des Generators und des Diskriminators einbinden?

<!--en:6a67092f-->
3. Wie beurteilen wir bei einem bedingten Generator getrennt, wie gut seine Samples die Vorgabe erfüllen, wie realistisch sie sind und wie vielfältig sie innerhalb jeder Klasse sind?

<!--en:b3c7601a-->
Wir veranschaulichen diese Konzepte, indem wir einen klassenbedingten Generator mit den 10 Klassen von CIFAR-10 trainieren und auswerten.

<!--en:0ce06c5c-->
## Die bedingte Zielfunktion

<!--en:4f64c329-->
### Der optimale bedingte Kritiker

<!--en:54935a72-->
Wir ziehen Paare aus Datenpunkt und Bedingung $(x, c) \sim p(x, c)$ aus der gemeinsamen Datenverteilung. Dabei kann $c$ ein diskretes Klassenlabel, eine Texteinbettung oder ein Bild sein. Damit die Randverteilung der Bedingung $p(c)$ mit derjenigen der echten Daten übereinstimmt, ziehen wir $c \sim p(c)$ aus den echten Daten und erzeugen $x' = G(z, c) \sim q(\cdot \mid c)$. Daraus ergibt sich die gemeinsame Verteilung der generierten Paare $q(x, c) = p(c)\,q(x \mid c)$. Als bedingte Zielfunktion verwenden wir den klassischen binären logarithmischen Verlust (:numref:`sec_basic_gan`), angewandt auf die Paare $(x, c)$:

<!--en:20a56e03-->
Die punktweise Optimierung über $D(x, c)$ verläuft auf dem Produktraum $\mathcal{X} \times \mathcal{C}$ genauso wie die Herleitung im unbedingten Fall. Der optimale bedingte Kritiker ist daher das logarithmierte Verhältnis der gemeinsamen Dichten. Die gemeinsame Randverteilung $p(c)$ kürzt sich heraus:

<!--en:f356be1b-->
Für jede Bedingung $c$ berechnet der optimale Kritiker also das logarithmierte Dichteverhältnis der bedingten Verteilungen $p(\cdot \mid c)$ und $q(\cdot \mid c)$. Setzen wir $D^\star(x, c)$ wieder in :eqref:`eq_gan_cond_V` ein, erhalten wir einen Ausdruck mit der erwarteten bedingten Jensen-Shannon-Divergenz:

<!--en:c0b2e78e-->
Dieser Ausdruck erreicht sein globales Minimum $-2\log 2$ genau dann, wenn $q(\cdot \mid c) = p(\cdot \mid c)$ für jede Bedingung $c$ mit $p(c) > 0$ gilt.

<!--en:6c81ef3a-->
### Übereinstimmung der bedingten Verteilungen und der Randverteilungen

<!--en:79313072-->
Die Forderung, dass die bedingten Verteilungen für jede Bedingung übereinstimmen ($q(\cdot \mid c) = p(\cdot \mid c)$), ist strikt stärker als die Forderung nach Übereinstimmung der Randverteilungen der Bilder ($q(x) = p(x)$). Angenommen, ein CIFAR-10-Generator erzeugt realistische Bilder, verschiebt aber alle Klassenlabel zyklisch ($q(\cdot \mid c) = p(\cdot \mid c + 1)$). Da alle 10 Klassen dieselbe A-priori-Wahrscheinlichkeit $p(c) = 1/10$ haben, stimmt seine Randverteilung der Bilder $q(x)$ exakt mit $p(x)$ überein. Die unbedingte Divergenz erreicht also ihr Minimum $\mathrm{JS}(p(x), q(x)) = 0$. Die bedingte Zielfunktion :eqref:`eq_gan_cond_value` vergleicht dagegen jede bedingte Verteilung mit der einer falschen Klasse ($\mathrm{JS}(p(\cdot \mid c), p(\cdot \mid c+1)) \approx \log 2$). Die bedingte Divergenz ist deshalb fast maximal (Aufgabe :exref:`ex_conditional-verify-decomposition-eqref-eq`).

<!--en:8e6a9bfe-->
Andererseits bestimmt für jede Bedingung $c$ ihre A-priori-Wahrscheinlichkeit $p(c)$ das Gewicht in :eqref:`eq_gan_cond_value`. Eine seltene Bedingung mit Wahrscheinlichkeit $p(c) = \epsilon$ trägt höchstens $2\epsilon \log 2$ zur Zielfunktion bei. Bei unausgewogenen Datensätzen müssen wir die Klassen beim Sampling daher oft gleich gewichten, damit der Generator seltene Klassen nicht ignoriert.

<!--en:e71cd5ff-->
## Architekturen für die Konditionierung

<!--en:b4a755be-->
### Drei Verfahren zur Konditionierung

<!--en:1489fc3c-->
:numref:`fig_gan_conditioning` zeigt die drei wichtigsten Möglichkeiten, einem Generator oder Diskriminator eine Bedingung $c$ zu übergeben:

<!--en:2cac0d6b-->
1. Bei der *Verkettung* (engl. concatenation) :cite:`Mirza.Osindero.2014` betten wir $c$ in einen Vektor $e_c$ ein. Diesen verketten wir mit dem latenten Vektor $z$ oder erweitern ihn durch Broadcasting und verketten ihn mit den räumlichen Merkmalskarten.

<!--en:f9729191-->
2. Die *Merkmalsmodulation* (engl. feature modulation) skaliert und verschiebt jeden Kanal einer intermediären Merkmalskarte $h$ mithilfe erlernter Funktionen von $c$ gemäß $a(c) \odot h + b(c)$. Normalisieren wir $h$ zuerst, erhalten wir die *bedingte Batchnormalisierung* :cite:`De-Vries.Strub.Mary.ea.2017,Brock.Donahue.Simonyan.2019` (oder AdaIN / FiLM). So fließt die Bedingung bei jeder Auflösung in den Generator ein.

<!--en:6a3fc184-->
3. Der *Projektionskopf des Diskriminators* :cite:`Miyato.Koyama.2018` bildet das innere Produkt einer Klasseneinbettung $e_c$ mit dem durch Pooling gewonnenen Merkmalsvektor $\phi(x)$ des Diskriminators und addiert einen unbedingten Score $\psi_0(x)$.

<!--en:bd62b65d-->
![Drei Verfahren zur Konditionierung von GANs. Links hängt die Verkettung eine Einbettung $e_c$ an die Eingabe oder Merkmalskarte an. In der Mitte skaliert und verschiebt die Merkmalsmodulation jeden Kanal abhängig von der Bedingung gemäß $a(c) \odot h + b(c)$. Rechts berechnet der Projektionskopf des Diskriminators $e_c^\top \phi(x) + \psi_0(x)$.](../img/mdl-gan-conditioning.svg)

<!--en:fb0a902f-->
Wir implementieren nun `Modulate` mit Einbettungstabellen für die Skalierung und Verschiebung je Klasse. Wir initialisieren sie mit $a(c) = 1$ und $b(c) = 0$ (Identitätsabbildung) und überprüfen, dass ein Gradientenschritt für Klasse 3 nur Zeile 3 der Einbettungstabellen aktualisiert.

<!--en:2a668a57-->
Bevor wir die Schicht ausführen, können wir zwei Vorhersagen treffen. Bei der Initialisierung muss die Ausgabe für jede Klasse mit der Eingabe übereinstimmen. Ein Gradientenschritt für einen Batch, der ausschließlich Klasse 3 anfordert (Katze in der unten verwendeten CIFAR-10-Reihenfolge), muss in jeder Tabelle genau Zeile 3 ändern. Denn beim Nachschlagen einer Einbettung fließt der Gradient nur zu den gelesenen Zeilen zurück. Zur Überprüfung machen wir einen solchen Schritt in Richtung des einfachen Ziels $2h + 1$.

<!--en:d4edd026-->
### Herleitung des Projektionskopfs mit dem Satz von Bayes

<!--en:d143b536-->
Wenden wir den Satz von Bayes $p(x \mid c) = p(c \mid x)p(x)/p(c)$ auf Zähler und Nenner von :eqref:`eq_gan_cond_dstar` an (mit gemeinsamer A-priori-Verteilung $p(c)$), so zerfällt der optimale bedingte Kritiker in zwei Terme:

<!--en:aaed50b8-->
Der erste Term misst die *Klassenübereinstimmung* (engl. class alignment) anhand des logarithmierten Verhältnisses der A-posteriori-Wahrscheinlichkeiten der Klasse bei $x$ für echte und generierte Daten. Der zweite Term misst, *wie realistisch das Sample unabhängig von der Klasse ist*. Wir modellieren beide A-posteriori-Verteilungen als Softmax-Klassifikatoren mit gemeinsamer Merkmalsdarstellung $\phi(x)$, nämlich $p(c \mid x) \propto \exp(u_c^\top \phi(x))$ und $q(c \mid x) \propto \exp(w_c^\top \phi(x))$. Ihr logarithmiertes Verhältnis ist dann $(u_c - w_c)^\top \phi(x)$ zuzüglich der von $c$ unabhängigen Differenz der logarithmierten Zustandssummen. Wir fassen diese Differenz und $\log(p(x)/q(x))$ in einem einzigen skalaren Ausgabekopf $\psi_0(x)$ zusammen und setzen $e_c = u_c - w_c$. So erhalten wir den *Projektionsdiskriminator* von :citet:`Miyato.Koyama.2018` (Aufgabe :exref:`ex_conditional-projection-derivation-modeled-both`, zum Vergleich siehe das GAN mit Hilfsklassifikator in Aufgabe :exref:`ex_conditional-auxiliary-classifier-bias`):

<!--en:c8f97d50-->
## Klassenbedingte Generierung mit CIFAR-10

<!--en:5170f39f-->
Wir trainieren nun einen klassenbedingten Generator mit den 50 000 mit Klassenlabeln versehenen Trainingsbildern der Größe $32 \times 32$ aus CIFAR-10 (5000 je Klasse, geladen mit `d2l.load_cifar10_arrays`).

<!--en:51e47f10-->
### Bedingter Generator und Projektionskritiker

<!--en:07c14769-->
Wir erweitern `d2l.Generator` und `d2l.Discriminator` (siehe :numref:`sec_dcgan`) für die Auflösung $32 \times 32$:

<!--en:f811f55b-->
- `CondGenerator` erweitert eine 64-dimensionale Klasseneinbettung mit Varianz eins durch Broadcasting auf $4 \times 4 \times 64$ und übergibt sie dem Generator über die Eingabe `extra`. Aufgabe :exref:`ex_conditional-modulated-generator` vergleicht diese Verkettung an der Eingabe mit `Modulate`-Schichten in jedem Block.

<!--en:6221a3fa-->
- `ProjectionCritic` summiert die $4 \times 4$-Merkmalskarte `self.phi(x)` über die räumlichen Positionen und erhält so $\phi(x)$. Anschließend berechnet der Kritiker $e_c^\top \phi(x) + \psi_0(\phi(x))$ (:eqref:`eq_gan_cond_projection`). Mit der Initialisierung $e_c = 0$ beginnt er als unbedingter Diskriminator.

<!--en:6a749dc1-->
Die Standardinitialisierung von `nnx.Embed` zieht Einträge, deren Standardabweichung dem Kehrwert der Quadratwurzel der Einbettungsbreite entspricht, hier also einem Achtel. Deshalb gibt der Code die Varianz eins ausdrücklich vor.

<!--en:a72af257-->
`ProjectionCritic` implementiert :eqref:`eq_gan_cond_projection`. Die Methode `phi` der Basisklasse liefert die $4 \times 4$-Merkmalskarte nach der Faltung, die die Merkmale mischt. Die Summe über alle Positionen ergibt $\phi(x)$. Eine lineare Ausgabeschicht berechnet aus $\phi(x)$ den Wert $\psi_0(x)$. Sie ersetzt den Ausgabekopf der Basisklasse, der die zu einem Vektor umgeformte Merkmalskarte einliest. Eine Tabelle enthält für jede Klasse eine Einbettung $e_c$. Anfangs sind alle Einträge null. Der Kritiker beginnt also als unbedingter Kritiker und lernt den vom Label abhängigen Term aus den Daten.

<!--en:9d02a473-->
### Training mit Paaren derselben Klasse

<!--en:5d6ec6c9-->
Wir trainieren mit `d2l.train_r3gan` (:numref:`sec_dcgan`) über 15 000 Schritte. Die Batchgröße beträgt 128, das Gewicht der Gradientenstrafe ist $\gamma = 0.5$. In jedem Minibatch geben wir für die generierten Bilder genau die Klassenlabel $y$ des Minibatchs mit den echten Bildern vor. Der Verlust der relativistischen Paarbildung $\log \sigma(D(x, c) - D(x', c))$ vergleicht daher stets ein echtes und ein generiertes Bild *derselben* Klasse $c$. Da alle Vergleiche innerhalb einer Klasse bleiben, ist $D(x, c)$ nur bis auf eine unabhängige additive Konstante $\zeta_c$ je Klasse bestimmt (Aufgabe :exref:`ex_conditional-critic-classifier`). Die protokollierte Genauigkeit des paarweisen Rankings $\hat P(D(x, c) > D(x', c))$ auf 512 zurückgehaltenen Testpaaren ist gegenüber diesen klassenweisen Verschiebungen invariant.

<!--en:d32ff5a8-->
Wir mitteln die protokollierten Werte über die letzten 5000 Schritte, die Genauigkeit des paarweisen Rankings zusätzlich über die ersten 2000.

<!--en:27fd4e53-->
An den Verläufen von `loss_D`, `loss_G` und der auf zurückgehaltenen Daten gemessenen Genauigkeit `accuracy` über alle 15 000 Schritte sehen wir, wie schnell sich das Spiel zwischen Projektionskritiker und bedingtem Generator stabilisiert:

<!--en:4d181d02-->
Nach einer kurzen Einschwingphase stabilisiert sich der Verlust der Paarbildung beim Kritiker nahe $0.52$ (unter $\log 2 \approx 0.69$). Die Genauigkeit des paarweisen Rankings auf zurückgehaltenen Paaren derselben Klasse pendelt sich bei etwa $0.80$ ein.

<!--en:99eed541-->
### Samples für vorgegebene Klassen

<!--en:04f38785-->
Wir erzeugen nun 9 Samples für jede der 10 CIFAR-10-Klassen mit `ema_G`.

<!--en:48639466-->
Jedes $3 \times 3$-Bildraster zeigt typische Merkmale der vorgegebenen Klasse (z. B. blauen Himmel bei Flugzeugen, Wasser bei Schiffen, grünes Laub bei Fröschen und Hirschen sowie Straßen im Hintergrund bei Autos und Lastwagen). Pose und Farbe variieren zwischen den 9 zufällig gezogenen latenten Vektoren.

<!--en:cb526ed2-->
## Auswertung eines bedingten Generators

<!--en:64c5d380-->
Ein bedingter Generator kann in drei verschiedenen Punkten versagen:

<!--en:7056caee-->
1. *Falsche Klasse* (geringe Übereinstimmung mit der Vorgabe),

<!--en:d2f76664-->
2. *unrealistische Bilder* (geringe Bildtreue) oder

<!--en:f6058389-->
3. *mangelnde Vielfalt innerhalb der Klassen* (Modusausfall innerhalb jeder Klasse).

<!--en:78fb2bb0-->
Wie :numref:`tab_gan_cond_failures` zeigt, erfasst keine einzelne Metrik alle drei Fehlerarten. Wir kombinieren deshalb einen *Prüfklassifikator* (engl. referee classifier) zur Messung der Klassenübereinstimmung mit marginalen Merkmalsabständen (FD und $\mathrm{MMD}^2$). Hinzu kommen *klassenweise* bestimmte $K$-NN-Präzision und -Trefferquote.

<!--en:facee8cb-->
:Welche Metriken welche Fehler bei der bedingten Generierung erkennen.

<!--en:d36f18a9-->
| Fehlerart | Klassenübereinstimmung | marginale FD, $\mathrm{MMD}^2$ | klassenweise Präzision | klassenweise Trefferquote |

<!--en:230c9e5b-->
| falsche Klasse | erkannt | unerkannt | erkannt | erkannt |

<!--en:1c44a041-->
| geringe Bildtreue | teilweise erkannt | erkannt | erkannt | teilweise erkannt |

<!--en:a8524791-->
| geringe Vielfalt innerhalb der Klassen | unerkannt (evtl. sogar belohnt) | erkannt | unerkannt (evtl. sogar belohnt) | erkannt |

<!--en:bab203cd-->
### Prüfklassifikator und Klassenübereinstimmung

<!--en:1eb61772-->
Wir trainieren den CIFAR-10-Klassifikator `d2l.FeatureCNN` mit 3 Blöcken (:numref:`sec_dcgan`) über 15 Epochen als unabhängigen Prüfklassifikator. Er erreicht eine Testgenauigkeit von $\sim 78\%$. Wir nutzen sowohl seine Klassenvorhersagen als auch seine vorletzte Schicht mit 128 Merkmalen.

<!--en:917af285-->
Wir erzeugen 500 Bilder je Klasse mit `ema_G`. Für jede Klasse vergleichen wir die Klassifikationsgenauigkeit des Prüfklassifikators auf generierten Bildern (*Klassenübereinstimmung*) mit seiner Genauigkeit auf echten Testbildern.

<!--en:20f4dc22-->
Wir stellen `align` und die Genauigkeit `cls_acc` des Prüfklassifikators auf echten Testbildern für jede der 10 Klassen tabellarisch gegenüber. So sehen wir, wie gut `ema_G` das vorgegebene Klassenlabel einhält:

<!--en:7c9f67de-->
Die mittlere Klassenübereinstimmung über die 10 Klassen beträgt $0.72\text{--}0.74$. Sie liegt nur $4\text{--}6$ Prozentpunkte unter der Genauigkeit des Prüfklassifikators auf echten CIFAR-10-Testbildern ($0.78$).

<!--en:3ae56a00-->
### Bildtreue und Vielfalt innerhalb der Klassen

<!--en:d855dd35-->
Wir teilen die 1000 Testbilder je Klasse in zwei Gruppen: 500 Referenzbilder (`f_real`) und 500 weitere echte Bilder als Vergleichsgruppe (`f_real2`). Wir berechnen die marginale FD und $\mathrm{MMD}^2$ (`d2l.fid_score`, `d2l.kid_score`) sowie die klassenweise $K$-NN-Präzision und -Trefferquote (`d2l.precision_recall` mit $K = 3$).

<!--en:985d3d63-->
Wir berechnen nun den marginalen Merkmalsabstand (`d2l.fid_score`) und die quadrierte MMD (`d2l.kid_score`) jeweils zum Referenzdatensatz `f_real`. Dabei vergleichen wir die echte Vergleichsgruppe `f_real2` mit beiden generierten Stichproben (`f_gen`, `f_gen2`). Außerdem prüfen wir den Anteil der rohen Pixelwerte außerhalb des zulässigen Bereichs (im Code `out_of_range`):

<!--en:08ac7b78-->
Gemessen an der FD liegen die generierten Bilder zehn- bis zwölfmal so weit von der Referenz entfernt wie die echte Vergleichsgruppe. Eine zweite Ziehung verändert die FD um weniger als ein Zehntel. Der Abstand ist also eine Eigenschaft des Generators und kaum von der Ziehung abhängig. Bei $\mathrm{MMD}^2$ liegt der Vergleichswert für echte Bilder innerhalb der Zufallsschwankungen bei null, der Wert für generierte Bilder bei etwa 0.11. Vor der Begrenzung liegt etwa einer von hundert Rohwerten außerhalb von $[-1, 1]$. Die Begrenzung verändert also nur wenige Pixel und verdeckt entsprechend wenig.

<!--en:fc910fb2-->
Die Abstände fassen alle Unterschiede in einer Zahl zusammen. Klassenweise Präzision und Trefferquote unterscheiden zwei Fälle: generierte Bilder, die keinem echten Bild ihrer Klasse ähneln, und echte Bilder, die keinem generierten Bild ähneln. Um jeden Merkmalsvektor legen wir eine Kugel, die bis zu seinem drittnächsten Nachbarn in der eigenen Menge reicht (:numref:`subsec_gan_precision_recall`). Die *Präzision* ist der Anteil der generierten Bilder der Klasse $c$, deren Merkmalsvektor in mindestens einer Kugel eines echten Bildes der Klasse $c$ liegt. Die *Trefferquote* ist der Anteil der echten Bilder der Klasse $c$, deren Merkmalsvektor in mindestens einer Kugel eines generierten Bildes liegt. Die Codezelle berechnet beide Maße für jede Klasse im Vergleich zur Referenz und gibt die Mittelwerte über alle Klassen sowie die jeweils schlechteste Klasse aus. Zum Schluss berechnet sie die Merkmalsvarianz innerhalb der Klassen. Dazu mittelt sie die innerhalb jeder Klasse berechneten Varianzen über Merkmale und Klassen.

<!--en:7d35c04e-->
Die klassenweise Präzision ist für generierte Bilder ($0.90$) sogar *höher* als für die echte Vergleichsgruppe ($0.85$). Die generierten Merkmalsvektoren liegen also weit im Inneren der Cluster der echten Klassen. Die klassenweise Trefferquote ist dagegen deutlich niedriger ($0.58$ gegenüber $0.83$), und die Merkmalsvarianz innerhalb der Klassen fällt um etwa 20% geringer aus. Der Generator konzentriert sich also auf die Umgebung der Modi jeder Klasse und deckt ihre Vielfalt nur teilweise ab.

<!--en:5d356e2d-->
### Prototypen nahe dem Klassenschwerpunkt als Baseline

<!--en:7f22bbd9-->
Warum können hohe Klassenübereinstimmung und hohe Präzision einen Mangel an Vielfalt verdecken? Für eine *Prototypen-Baseline* wählen wir aus der echten Vergleichsgruppe je Klasse die 100 Bilder aus, deren Merkmalsvektoren dem Klassenschwerpunkt am nächsten liegen. Wir vergleichen sie mit je 100 zufälligen Bildern aus der echten Vergleichsgruppe und 100 generierten Bildern pro Klasse.

<!--en:230c5340-->
Wie nach :numref:`tab_gan_cond_failures` zu erwarten, erreichen die Prototypen nahe dem Schwerpunkt eine fast perfekte Klassenübereinstimmung (etwa $0.9$, höher als bei zufälligen echten Bildern) und Präzision (etwa $0.97$). Ihre Trefferquote fällt jedoch erheblich geringer aus (unter $0.4$ gegenüber etwa $0.85$), und ihre FD ist schlechter (etwa $3$ gegenüber $0.43$), weil die Auswahl die Kovarianz ihrer Merkmale künstlich verringert. BigGAN nutzt diesen Zielkonflikt mit dem *Truncation-Trick* gezielt aus: Das Modell beschränkt die Verteilung der latenten Vektoren $z$ auf einen Bereich um null und erhöht so Präzision und Klassenübereinstimmung auf Kosten der Trefferquote (Aufgabe :exref:`ex_conditional-biggan-s-truncation-trick`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:52809c78-->
Konditionieren wir sowohl den Generator $G(z, c)$ als auch den Kritiker $D(x, c)$ auf $c \sim p(c)$, erhalten wir für jede Bedingung den optimalen Kritiker $D^\star(x, c) = \log(p(x \mid c)/q(x \mid c))$ und minimieren die erwartete bedingte Divergenz im Ausdruck $\mathbb{E}_{c \sim p(c)}[2\,\mathrm{JS}(p(\cdot \mid c), q(\cdot \mid c))] - 2\log 2$. Mit dem Satz von Bayes zerlegen wir $D^\star(x, c)$ in einen Term für die Klassenübereinstimmung $\log(p(c \mid x)/q(c \mid x))$ und einen Term für den klassenunabhängigen Realismus $\log(p(x)/q(x))$. Daraus ergibt sich der Projektionskopf des Diskriminators $D(x, c) = e_c^\top \phi(x) + \psi_0(x)$ :cite:`Miyato.Koyama.2018`. Bei der Auswertung eines bedingten Generators müssen wir drei Aspekte getrennt betrachten: die Übereinstimmung mit der Vorgabe (unabhängiger Prüfklassifikator), die Bildtreue (klassenweise Präzision) und die Vielfalt innerhalb der Klassen (klassenweise Trefferquote und Merkmalsabstände). Gibt ein Generator nur besonders typische Klassenprototypen aus, kann er trotz des Verlusts der Vielfalt innerhalb der Klassen eine hohe Klassenübereinstimmung und eine hohe Präzision erreichen. Ohne ein Maß für die Trefferquote bleibt das verborgen.

<!--en:2294243a-->
## Übungen

<!--en:db791930-->
1. [code] **Zwei bedingte Gauß-Verteilungen mit bekannten Dichten.** Die beiden Werte von $c \in \{1, 2\}$ seien gleich wahrscheinlich, mit $p(\cdot \mid 1) = \mathcal{N}(-2, 1)$ und $p(\cdot \mid 2) = \mathcal{N}(2, 1)$. Ferner gelte $q(\cdot \mid 1) = \mathcal{N}(0, 1)$ und $q(\cdot \mid 2) = \mathcal{N}(2, 1)$.
    1. Berechnen Sie den theoretischen Wert :eqref:`eq_gan_cond_value` durch eindimensionale numerische Quadratur.
    1. Trainieren Sie einen kleinen MLP-Kritiker mit $(x, \text{one\_hot}(c))$, um :eqref:`eq_gan_cond_V` zu maximieren. Prüfen Sie, ob der Zielfunktionswert nach der Konvergenz mit dem Wert aus (a) übereinstimmt. Plotten Sie $D(x, c)$ für beide Klassen im Vergleich zum exakten logarithmierten Dichteverhältnis :eqref:`eq_gan_cond_dstar`.
    1. Nehmen Sie nun an, der Generator vertauscht die beiden Klassen: $q(\cdot \mid 1) = \mathcal{N}(2, 1)$ und $q(\cdot \mid 2) = \mathcal{N}(-2, 1)$. Berechnen Sie sowohl den bedingten Wert :eqref:`eq_gan_cond_value` als auch den unbedingten Wert für die Randverteilungen (ohne Berücksichtigung von $c$). Überprüfen Sie beide durch Training.
   :exlabel:`ex_conditional-verify-decomposition-eqref-eq`

<!--en:7d7c767b-->
1. **Ausdrucksstärke des Projektionskopfs im Diskriminator.** Bei der Herleitung von :eqref:`eq_gan_cond_projection` haben wir eine gemeinsame Merkmalsabbildung $\phi(x)$ für $p(c \mid x)$ und $q(c \mid x)$ vorausgesetzt.
    1. Angenommen, $p(c \mid x) \propto \exp(u_c^\top \phi_p(x))$ und $q(c \mid x) \propto \exp(w_c^\top \phi_q(x))$ verwenden verschiedene Merkmalsabbildungen. Zeigen Sie, dass der resultierende optimale Kritiker für den verketteten Merkmalsvektor weiterhin die Projektionsform :eqref:`eq_gan_cond_projection` hat. Verwenden Sie dazu den Vektor $(\phi_p(x), \phi_q(x))$.
    1. Zeigen Sie für $K$ diskrete Klassen, dass sich *jede* Funktion $D(x, c)$ in der Form $e_c^\top \phi(x) + \psi_0(x)$ schreiben lässt, sofern $\dim(\phi(x)) \geq K$ gilt.
   :exlabel:`ex_conditional-projection-derivation-modeled-both`

<!--en:ef429280-->
1. [short-code] **Der Projektionskritiker als Klassifikator.** Der trainierte Projektionskritiker `net_D` weist jedem Bild zehn Scores zu, nämlich $S_{ic} = D(x_i, c)$ für das Bild $x_i$.
    1. Drücken Sie mithilfe von :eqref:`eq_gan_cond_bayes` die Differenz $D^\star(x, c) - D^\star(x, c')$ durch $p(c \mid x)$ und $q(c \mid x)$ aus. Warum unterscheidet sich $\arg\max_c D^\star(x, c)$ vom Bayes-Klassifikator $\arg\max_c p(c \mid x)$?
    1. Da die Paarbildung innerhalb einer Klasse eine beliebige klassenweise Verschiebung $\zeta_c$ in $D(x, c)$ zulässt, schätzen Sie $\hat\zeta_c$ als mittleren Score des Kritikers auf generierten Bildern der Klasse $c$. Vergleichen Sie die Klassifikationsgenauigkeit auf `test_X` für die unkorrigierte Entscheidung $\arg\max_c S_{ic}$ und die um die Verschiebung korrigierte Entscheidung $\arg\max_c (S_{ic} - \hat\zeta_c)$.
    1. Warum muss sich die Genauigkeit des korrigierten Kritikers der Genauigkeit beim zufälligen Raten ($1/10$) nähern, wenn $q(\cdot \mid c) \to p(\cdot \mid c)$ gilt?
   :exlabel:`ex_conditional-critic-classifier`

<!--en:26eefb5b-->
1. [code] **Modulierter Generator und Verkettung an der Eingabe im Vergleich.** ● Bauen Sie einen bedingten Generator, der die Klasseninformation nach jedem Faltungsblock über eine `Modulate`-Schicht erhält statt durch Verkettung an der Eingabe.
    1. Bei der Initialisierung ($a(c) = 1, b(c) = 0$) ist die Ausgabe des modulierten Generators unabhängig von $c$. Verfolgen Sie, wie beim ersten Generatorschritt ausgehend vom Projektionskritiker von null verschiedene Gradienten bezüglich $a(c)$ und $b(c)$ entstehen.
    1. Trainieren Sie den modulierten Generator und `CondGenerator` mit jeweils zwei Zufallsstartwerten für je 4000 Schritte. Vergleichen Sie die mittlere Klassenübereinstimmung und die klassenweise Trefferquote.
   :exlabel:`ex_conditional-modulated-generator`

<!--en:a6b90c4f-->
1. [code] **Der Truncation-Trick und der Zielkonflikt zwischen Präzision und Trefferquote.** Beim Truncation-Trick :cite:`Brock.Donahue.Simonyan.2019` ziehen wir jede latente Koordinate aus einer auf $[-\tau, \tau]$ gestutzten Standardnormalverteilung.
    1. Berechnen Sie für einen 100-dimensionalen latenten Vektor und $\tau \in \{0.5, 1, 2\}$ den Erwartungswert der quadrierten Norm $\mathbb{E}[\|z\|^2]$ beim Ziehen aus der gestutzten Verteilung. Vergleichen Sie ihn mit dem Wert $\mathbb{E}[\|z\|^2] = 100$ beim Training.
    1. Werten Sie `ema_G` mit der gestutzten Verteilung für $\tau \in \{0.5, 1, 2, \infty\}$ aus. Geben Sie die Klassenübereinstimmung, die klassenweise Präzision und Trefferquote sowie die FD gegenüber `f_real` an. Prüfen Sie, ob bei abnehmendem $\tau$ die Trefferquote zugunsten höherer Klassenübereinstimmung und Präzision sinkt und sich der Generator so den Prototypen nahe dem Klassenschwerpunkt annähert.

   *Nach Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), Aufgabe 15.6.*
   :exlabel:`ex_conditional-biggan-s-truncation-trick`

<!--en:42930edb-->
1. **Verzerrung beim Auxiliary Classifier GAN (AC-GAN).** ● Ein AC-GAN :cite:`Odena.Olah.Shlens.2017` trainiert einen Hilfsklassifikator $C(c \mid x)$ und addiert die Kreuzentropie $-\mathbb{E}_{c \sim p(c), x \sim q(\cdot \mid c)}[\log C(c \mid x)]$ zum Generatorverlust.
    1. Zeigen Sie, dass dieser zusätzliche Generatorverlust gleich folgendem Ausdruck ist:

       $$\mathbb{E}_{x \sim q}\Big[H\big(q(\cdot \mid x)\big) + D_{\textrm{KL}}\big(q(\cdot \mid x) \,\|\, C(\cdot \mid x)\big)\Big].$$

    1. Nehmen Sie an, dass $C(c \mid x) = p(c \mid x)$ gilt und sich Klassen überlappen ($H(p(\cdot \mid x)) > 0$). Erklären Sie, warum die Minimierung dieses Terms den Generator dazu bewegt, Überlappungsbereiche der Datenverteilung zu meiden. Warum verringert das die Vielfalt innerhalb der Klassen im Vergleich zum Projektionsdiskriminator (:eqref:`eq_gan_cond_bayes`)?
   :exlabel:`ex_conditional-auxiliary-classifier-bias`

<!--en:042792e0-->
[Diskussion](https://d2l.discourse.group/t/36813)

<!--en:909a12ea-->
Bedingte Generierung und Projektionsdiskriminatoren<br>**Bedingtes Minimax-Spiel · JS-Divergenz je Bedingung · Merkmalsmodulation · Projektionskopf nach Miyato und Koyama · Bewertung von Klassenübereinstimmung und Vielfalt**

<!--en:fb5c23a8-->
::: {.slide title="Warum sich bei unbedingten GANs keine Klasse vorgeben lässt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:bfaa8ce0-->
Unbedingte Generatoren ziehen zufällige Samples aus der Randverteilung der Bilder $p(x)$. Es gibt keine Möglichkeit, eine bestimmte Klasse $c$ (z. B. *„Vogel“* oder *„Schiff“*), eine Bildbeschreibung oder ein Ausgangsbild vorzugeben:

<!--en:6bf88dcc-->
- **Übereinstimmung der Randverteilungen $\neq$ Erfüllung der Bedingung:** Ein Generator für die 10 CIFAR-10-Klassen erzeuge realistische Bilder, verschiebe aber jedes Label zyklisch: $q(x \mid c) = p(x \mid c + 1 \bmod 10)$.

<!--en:3611b1b4-->
- Da alle $10$ Klassen dieselbe A-priori-Wahrscheinlichkeit $p(c) = \frac{1}{10}$ haben, stimmt seine Randverteilung $q(x) = \sum_c p(c)q(x \mid c) = p(x)$ **exakt** mit der Datenverteilung überein ($\mathrm{JS}(p(x), q(x)) = 0$), obwohl er bei jeder Klassenvorgabe das falsche Objekt liefert.

<!--en:2cd9bbb1-->
- **Lösung (Mirza & Osindero, 2014):** Die Bedingung $c$ erhalten **beide**, der Generator $G(z, c)$ und der Diskriminator $D(x, c)$.

<!--en:bd9fbf4b-->
::: {.slide title="Das bedingte Minimax-Spiel für Paare (x, c)"}

<!--en:9d5e7bff-->
[Formulierung und optimaler Kritiker]{.kicker}

<!--en:5bccc0d5-->
- **Gemeinsame Verteilungen echter und generierter Paare:** Echte Paare $(x, c) \sim p(x, c)$ mit generierten Paaren vergleichen. Für letztere $c \sim p(c)$ aus der echten Randverteilung ziehen und $x' = G(z, c) \sim q(\cdot \mid c)$ erzeugen.

<!--en:403e39c7-->
- **Die gemeinsame A-priori-Verteilung $p(c)$ kürzt sich heraus:** Bei punktweiser Optimierung auf $\mathcal{X} \times \mathcal{C}$ entfällt der Faktor $p(c)$ in Zähler und Nenner. Übrig bleibt für jede Bedingung das exakte logarithmierte Dichteverhältnis.

<!--en:2e5f5ca0-->
::: {.slide title="Erwartete bedingte JS-Divergenz und seltene Klassen"}

<!--en:78372514-->
[Zielfunktionswert für die Grundgesamtheit und Klassenungleichgewicht]{.kicker}

<!--en:f6117c1e-->
- **Unabhängiges Spiel für jede Bedingung $c$:** Das globale Minimum $-2\log 2$ ist genau dann erreicht, wenn $q(\cdot \mid c) = p(\cdot \mid c)$ für jedes $c$ mit $p(c) > 0$ gilt. Eine zyklische Verschiebung der Label fällt sofort auf.

<!--en:31399d60-->
- **Problem bei seltenen Klassen:** Da jede Bedingung $c$ mit $p(c)$ gewichtet wird, trägt eine seltene Klasse mit $p(c) = 10^{-3}$ höchstens $0.0014$ Nat bei. Daher bietet sich klassenbalanciertes Sampling für Minibatches an.

<!--en:85134e03-->
::: {.slide title="Drei Verfahren zur Konditionierung in der Netzarchitektur"}

<!--en:cbef823d-->
[Konditionierung von Generator und Kritiker]{.kicker}

<!--en:7a360cf7-->
1. **Verkettung (Mirza & Osindero, 2014):** Die Einbettung $c \mapsto e_c$ mit dem latenten Vektor $z$ verketten oder durch Broadcasting über die räumlichen Merkmalskarten erweitern (z. B. $4 \times 4 \times d_e$).

<!--en:102e4b8e-->
2. **Merkmalsmodulation (bedingte Batchnormalisierung / AdaIN / FiLM):** Nach Faltungen im gesamten Generator jeden Kanal abhängig von der Bedingung gemäß $a(c) \odot h + b(c)$ skalieren und verschieben.

<!--en:d6aed7b0-->
3. **Projektionskopf des Diskriminators (Miyato & Koyama, 2018):** Der Score $e_c^\top \phi(x) + \psi_0(x)$ enthält das innere Produkt aus der Klasseneinbettung $e_c$ und dem durch Pooling gewonnenen Merkmalsvektor $\phi(x)$ des Kritikers.

<!--en:f3f7d2da-->
::: {.slide title="Schicht zur kanalweisen Merkmalsmodulation"}

<!--en:c3a5cd44-->
[Identität als Startwert und dünn besetzte Gradienten]{.kicker}

<!--en:9f6973f1-->
`Modulate` speichert für jede Klasse Skalierung $a(c)$ und Verschiebung $b(c)$ in Einbettungstabellen. Die Anfangswerte ergeben die Identitätsabbildung $a(c) = \mathbf{1}, b(c) = \mathbf{0}$. Beim Nachschlagen fließen Gradienten nur zu den im Minibatch gelesenen Zeilen zurück. Ein Schritt für Klasse $3$ aktualisiert also nur Zeile $3$:

<!--en:d2f8e103-->
::: {.slide title="Herleitung des Projektionskopfs mit dem Satz von Bayes"}

<!--en:ce3837d9-->
[Zerlegung nach Miyato und Koyama]{.kicker}

<!--en:9f4a3cb5-->
- **Additive Zerlegung:** Nach dem Satz von Bayes ist das bedingte logarithmierte Dichteverhältnis die Summe zweier Terme: **Klassenübereinstimmung** $\log(p(c \mid x)/q(c \mid x))$ und **klassenunabhängiger Realismus** $\log(p(x)/q(x))$.

<!--en:4ea7dc66-->
- **Ausgabekopf mit innerem Produkt (Miyato & Koyama, 2018):** Bei einer Modellierung von $p(c \mid x)$ und $q(c \mid x)$ durch Softmax-Klassifikatoren auf den Merkmalen $\phi(x)$ des Kritikers ist der klassenabhängige Teil ihres logarithmierten Verhältnisses linear in $\phi(x)$ mit Klasseneinbettung $e_c = u_c - w_c$.

<!--en:0d6fb44b-->
::: {.slide title="Projektionsdiskriminator und Hilfsklassifikator im Vergleich"}

<!--en:18c987fe-->
[Warum keine niedrige Klassenentropie erzwingen?]{.kicker}

<!--en:a9c3c324-->
- **Verzerrung durch Entropiereduktion beim AC-GAN (Odena et al., 2017):** Die Minimierung von $H(q(\cdot \mid x'))$ bestraft den Generator für Samples in Bereichen, in denen sich die echten Klassen überlappen ($H(p(\cdot \mid x)) > 0$). Die Vielfalt innerhalb der Klassen bricht ein, und es bleiben überzeichnete, besonders typische Prototypen.

<!--en:1e6c7fcd-->
- **Unverzerrter Projektionskritiker:** Er schätzt $\log(p(c \mid x)/q(c \mid x))$. Dieser Term verschwindet bei $q(\cdot \mid c) = p(\cdot \mid c)$, ohne die bedingte Entropie auf null zu zwingen.

<!--en:3c373d43-->
::: {.slide title="Implementierung von CondGenerator und ProjectionCritic"}

<!--en:2dc21d1a-->
[Grundnetz für CIFAR-10 mit 32×32 Pixeln]{.kicker}

<!--en:79229161-->
::: {.slide title="Implementierung von CondGenerator und ProjectionCritic (Fortsetzung)"}

<!--en:2dc21d1a-2-->
[Grundnetz für CIFAR-10 mit 32×32 Pixeln]{.kicker}

<!--en:0ddda4d5-->
Kontrolle der Formen: Der Generator bildet 100-dimensionale latente Vektoren und Label auf RGB-Bilder der Größe $32 \times 32$ ab. Der Kritiker liefert einen Score je Bild:

<!--en:085ecbfa-->
::: {.slide title="Relativistische Paarbildung innerhalb jeder CIFAR-10-Klasse"}

<!--en:0dd06fbe-->
[Trainingsablauf und Verschiebungsinvarianz]{.kicker}

<!--en:bc23c4d9-->
`CondGenerator` und `ProjectionCritic` auf $50{,}000$ CIFAR-10-Bildern ($32 \times 32$) über $15{,}000$ Schritte mit `train_r3gan` trainieren (Batchgröße $128$, $\gamma = 0.5$). Jedes echte Bild erhält ein generiertes Bild **derselben Klasse $c$** als Partner:

<!--en:2d66ed17-->
- **Verlust bei Paarbildung innerhalb einer Klasse:** $-\mathbb{E}_{c}[\log \sigma(D(x, c) - D(x', c))]$ stabilisiert sich nahe $0.52$ (unter $\log 2 \approx 0.69$). Die Ranking-Genauigkeit auf zurückgehaltenen Paaren derselben Klasse pendelt sich bei etwa $0.80$ ein.

<!--en:a8e9e47a-->
- **Verschiebungsinvarianz je Klasse:** Da Vergleiche nur innerhalb jeder Klasse $c$ stattfinden, ändert die Verschiebung $D(x, c) \mapsto D(x, c) + \zeta_c$ um eine beliebige klassenabhängige Konstante $\zeta_c$ den Verlust der Paarbildung nicht.

<!--en:8ac61b3d-->
::: {.slide title="CIFAR-10-Samples für vorgegebene Klassen"}

<!--en:251414b2-->
[Bilder für 10 vorgegebene Klassen]{.kicker}

<!--en:f92e0b4b-->
$9$ zufällige latente Vektoren für jede der $10$ vorgegebenen CIFAR-10-Klassen ziehen und daraus mit dem EMA-Generator `ema_G` Bilder erzeugen:

<!--en:e5968fe0-->
::: {.slide title="Drei unabhängige Fehlerarten bei der Auswertung"}

<!--en:f7ca5205-->
[Einteilung der Bewertungskriterien]{.kicker}

<!--en:fa42de42-->
- **Kriterium 1: falsche Klasse ($q(\cdot \mid c) = p(\cdot \mid c')$):** Ein unabhängiger **Prüfklassifikator** erkennt den Fehler sofort. Die marginale FD und $\mathrm{MMD}^2$ erkennen ihn überhaupt nicht, wenn die Randverteilungen der Mischungen übereinstimmen.

<!--en:b32225f8-->
- **Kriterium 2: geringe Bildtreue (Unschärfe/Artefakte):** Das zeigen die **marginale FD** und die **klassenweise $K$-NN-Präzision**.

<!--en:8e3ffc59-->
- **Kriterium 3: geringe Vielfalt innerhalb der Klasse (Kollaps auf Prototypen):** Die Genauigkeit des Prüfklassifikators und die Präzision *belohnen* besonders typische Prototypen sogar. Erst die **klassenweise $K$-NN-Trefferquote** deckt einen Modusausfall innerhalb der Klasse $c$ auf.

<!--en:69c92dd6-->
::: {.slide title="Messung der Klassenübereinstimmung mit einem Prüfklassifikator (CNN)" layout="tall-output"}

<!--en:5c465d26-->
[Kriterium 1: Übereinstimmung mit der Vorgabe]{.kicker}

<!--en:fcb77a4b-->
Der CIFAR-10-Klassifikator `FeatureCNN` mit 3 Blöcken dient nach einem Training mit horizontalen Spiegelungen und Verschiebungen um 4 Pixel über $15$ Epochen als unabhängiger Prüfklassifikator ($78\%$ Testgenauigkeit). Seine Genauigkeit auf $500$ generierten Bildern je Klasse ergibt:

<!--en:a484acf4-->
- **Nahe am Ergebnis für echte Fotos:** Die mittlere Klassenübereinstimmung über die $10$ Klassen liegt bei **$0.72\text{--}0.74$**, nur $4\text{--}6$ Prozentpunkte unter der Genauigkeit des Prüfklassifikators auf echten CIFAR-10-Testbildern ($0.78$).

<!--en:1c353cef-->
- Bei echten wie bei generierten Bildern erkennt der Klassifikator *Katzen* ($0.51\text{--}0.58$) am schlechtesten und *Autos, Schiffe und Lastwagen* ($\ge 0.79$) am besten.

<!--en:c71b18e0-->
::: {.slide title="Marginale Merkmalsabstände im Vergleich zu echten Testbildern"}

<!--en:c9b0e3eb-->
[Kriterium 2: Übereinstimmung der Randverteilungen]{.kicker}

<!--en:9614ffd7-->
Die $10{,}000$ CIFAR-10-Testbilder liefern $5{,}000$ Referenzbilder (`f_real`, $500$ je Klasse) und $5{,}000$ echte Bilder als Vergleichsgruppe (`f_real2`, $500$ je Klasse). Damit lassen sich die $5{,}000$ generierten Bilder im 128-dimensionalen Merkmalsraum des Prüfklassifikators bewerten:

<!--en:1742ec37-->
- Die generierten Bilder erreichen $\mathrm{FD} \approx 1.6\text{--}2.0$ ($10\text{--}12\times$ so viel wie der Vergleichswert von $\approx 0.15$ für eine endliche Stichprobe echter Bilder) und $\mathrm{MMD}^2 \approx 0.11$. Eine zweite Ziehung latenter Vektoren verändert die FD um $< 0.1$. Vor der Begrenzung liegen weniger als $1\%$ der rohen Ausgabewerte des Generators außerhalb von $[-1, 1]$.

<!--en:4d55e56d-->
::: {.slide title="Klassenweise Präzision und Trefferquote auf CIFAR-10"}

<!--en:b25ff3f4-->
[Kriterium 3: Abdeckung innerhalb der Klassen]{.kicker}

<!--en:c2033cd1-->
Die 3-NN-Präzision und -Trefferquote **getrennt für jede Klasse $c$** anhand der $500$ Referenzbilder aus dem Testdatensatz der Klasse $c$ berechnen. So lassen sich der Realismus einzelner Samples und die Vielfalt innerhalb der Klasse unterscheiden:

<!--en:934b30f2-->
- **Höhere Präzision als in der echten Vergleichsgruppe ($0.90$ gegenüber $0.85$):** Die generierten Merkmalsvektoren liegen tief im dicht besetzten Kern jedes echten Klassenclusters.

<!--en:c4c4d1ad-->
- **Geringere Trefferquote ($0.58$ gegenüber $0.83$) und um $20\%$ geringere Varianz innerhalb der Klassen:** Rund $42\%$ der echten Testbilder haben keinen generierten Nachbarn in ihrer 3-NN-Kugel. Der Generator konzentriert sich auf die Umgebung typischer Klassenmodi und deckt die Vielfalt jeder Klasse nur teilweise ab.

<!--en:7edc716d-->
::: {.slide title="Die Prototypenfalle: Klassenübereinstimmung belohnt den Kollaps"}

<!--en:5c23d9ff-->
[Prototypen nahe dem Klassenschwerpunkt als Baseline]{.kicker}

<!--en:43892040-->
Hohe Klassenübereinstimmung und hohe Präzision können einen Zusammenbruch der Vielfalt verdecken. Zum Nachweis dient eine **Prototypen-Baseline** aus den $100$ Bildern je Klasse aus der echten Vergleichsgruppe, deren Merkmalsvektoren dem Klassenschwerpunkt am nächsten liegen:

<!--en:11111af9-->
**Das Prototypenparadox:** Die Auswahl der $100$ typischsten echten Bilder je Klasse erhöht die **Klassenübereinstimmung** auf $\approx 0.9$ (gegenüber $\approx 0.8$ bei zufälligen echten Bildern) und die **Präzision** auf $\approx 0.97$. Zugleich bricht die **Trefferquote** ein ($\approx 0.85 \to \approx 0.37$), und die **FD** verschlechtert sich um das Siebenfache ($0.43 \to \approx 3$). Der *Truncation-Trick* von BigGAN nutzt diesen Zielkonflikt gezielt durch die Beschränkung $z \in [-\tau, \tau]$.

<!--en:ad79f5b3-->
::: {.slide title="Verständnisfragen: bedingte Kritiker und Metriken"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:3d0bf752-->
Fragen zu Architekturen und Auswertung bedingter GANs:

<!--en:31ef847f-->
1. **Scores des Kritikers über Klassen hinweg vergleichen:** Nach dem RpGAN-Training mit Paaren derselben CIFAR-10-Klasse vergibt der Projektionskritiker im Mittel den Score $+4.2$ für echte Katzen und $-1.5$ für echte Lastwagen. Hält er Katzen also für realistischer als Lastwagen?

<!--en:4d23db7f-->
2. **Auswirkung der Stutzung im latenten Raum ($\tau \downarrow$):** Beim Truncation-Trick von BigGAN stammt jede latente Koordinate aus einer auf $[-\tau, \tau]$ gestutzten Standardnormalverteilung. Wie verändern sich Klassenübereinstimmung, klassenweise Präzision, klassenweise Trefferquote und FD, wenn $\tau$ von $\infty$ in Richtung $0.5$ sinkt?

<!--en:557b98b6-->
**Lösungen:**

<!--en:bb8aa35a-->
1. **Nein.** Die Paarbildung innerhalb jeder Klasse wertet nur Differenzen $D(x, c) - D(x', c)$ innerhalb derselben Klasse $c$ aus. Eine beliebige additive Konstante $\zeta_c$ je Klasse ändert daher die Zielfunktion nicht. Den Scores verschiedener Klassen fehlt ein gemeinsamer Maßstab.

<!--en:60d05002-->
2. Die Stutzung von $z$ rückt die generierten Samples näher an den dicht besetzten Schwerpunkt jeder Klasse. Das entspricht der **Prototypen-Baseline mit den Bildern nahe dem Schwerpunkt**: **Klassenübereinstimmung und Präzision steigen**, die **klassenweise Trefferquote sinkt**, und die **FD verschlechtert sich (steigt)**.

<!--en:11d1870d-->
::: {.slide title="Verständnisfragen: bedingte Kritiker und Metriken (Fortsetzung)"}

<!--en:929b7671-2-->
[Wissen überprüfen]{.kicker}

<!--en:48824d90-->
3. **Nachträglicher Einsatz des Kritikers als Klassifikator:** Warum muss die Klassifikationsgenauigkeit eines um die Verschiebung korrigierten optimalen bedingten Kritikers $D^\star(x, c) = \log(p(x \mid c)/q(x \mid c))$ auf das Niveau zufälligen Ratens ($1/10$) sinken, wenn der Generator konvergiert ($q(\cdot \mid c) \to p(\cdot \mid c)$)?

<!--en:23db34b8-->
**Lösung:**

<!--en:4e490368-->
3. Nach dem Satz von Bayes gilt $D^\star(x, c) - D^\star(x, c')$ $= \log \frac{p(c \mid x)}{p(c' \mid x)} - \log \frac{q(c \mid x)}{q(c' \mid x)}$. Für $q(\cdot \mid c) \to p(\cdot \mid c)$ nähert sich die generierte A-posteriori-Verteilung der wahren an. Dann gilt $D^\star(x, c) \to 0$ für jede Klasse $c$.

<!--en:1c6494ff-->
::: {.slide title="Zusammenfassung: bedingte GANs und ihre Auswertung"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:5f6d474b-->
| Komponente oder Metrik | Mathematische Form | Wichtige Eigenschaft oder blinder Fleck | Rolle bei bedingten GANs |

<!--en:ee7b7bc0-->
| **Bedingtes Spiel mit logarithmischem Verlust** | $\max_D V = \mathbb{E}_c[2\,\mathrm{JS}(p(\cdot \mid c), q(\cdot \mid c))] - 2\log 2$ | Optimaler Kritiker $D^\star(x, c) = \log \frac{p(x \mid c)}{q(x \mid c)}$ | Abgleich für jede Bedingung: $q(\cdot \mid c) = p(\cdot \mid c)$ |

<!--en:3c33657f-->
| **Schicht zur Merkmalsmodulation** | $\mathrm{Modulate}(h, c) = a(c) \odot h + b(c)$ | Anfangswerte $a = \mathbf{1}, b = \mathbf{0}$ (Identität), Gradienten nur in einzelnen Zeilen | Klasse, Bildbeschreibung oder Zeitschritt als Eingabe für jeden Generatorblock |

<!--en:807e8f79-->
| **Projektionskopf des Kritikers** | $D(x, c) = e_c^\top \phi(x) + \psi_0(x)$, $\;e_c = u_c - w_c$ | Herleitung aus $\log\frac{p(c \mid x)}{q(c \mid x)} + \log\frac{p(x)}{q(x)}$ | Unverzerrter bedingter Kritiker (vermeidet die Entropieverzerrung des AC-GAN) |

<!--en:f0a4cc2e-->
| **RpGAN-Paarbildung innerhalb der Klassen** | $-\mathbb{E}_c[\log\sigma(D(x, c) - D(x', c))]$ | Invariant gegenüber klassenweisen Verschiebungen $D(x, c) + \zeta_c$ | Kontrolle der paarweisen Genauigkeit $\hat{P}(D(x,c) > D(x',c))$ auf zurückgehaltenen Paaren derselben Klasse |

<!--en:de48a264-->
| **Klassenübereinstimmung laut Prüfklassifikator und Präzision** | $\hat{P}(\text{referee}(G(z, c)) = c)$ und Abdeckung durch 3-NN-Kugeln echter Bilder | Erkennt keinen Kollaps innerhalb der Klasse (belohnt Prototypen) | Misst Erfüllung der Klassenvorgabe und Realismus je Sample |

<!--en:8f74ecf3-->
| **Klassenweise 3-NN-Trefferquote und FD** | Anteil der echten Testbilder der Klasse $c$ in Kugeln generierter Bilder | Abhängig von der Stichprobengröße $n$ | Erkennt Modusausfall innerhalb der Klasse und Kollaps auf Prototypen |
