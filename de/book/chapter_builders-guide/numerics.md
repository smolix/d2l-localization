<!--en:e2043a49-->
# Zahlenformate

<!--en:ccc19a83-->
Bisher haben wir alle Tensoren unserer Modelle als 32-Bit-Gleitkommazahlen (fp32) dargestellt. Brauchen neuronale Netze tatsächlich so viele Bits? Halbieren wir die Bitbreite, gewinnen wir sofort bei Speicherbedarf und Durchsatz. Das Modell mit 8 Milliarden Parametern, das uns durch dieses Kapitel begleitet, braucht für seine Gewichte bei 32 Bit Genauigkeit 32 GB, bei 16 Bit nur 16 GB. Moderne GPUs führen Matrizenmultiplikationen mit 16 Bit mindestens doppelt so schnell aus wie mit 32 Bit. Können wir mit 16 Bit trainieren und trotzdem dasselbe Modell erhalten?

<!--en:389dab82-->
Ohne weitere Vorkehrungen gelingt das nicht. 16-Bit-Formate haben ein merklich gröberes Zahlenraster. Dadurch können zwei Probleme auftreten, die zur Laufzeit keine Fehlermeldung auslösen. In fp16 führt schon $300^2$ zu einem Überlauf auf `inf`. In der Folge treten beim Training überall `inf` und NaN auf. Addieren wir in bf16 $0.001$ zu einem Gewicht von $1$, rundet das Ergebnis wieder auf $1$. Das Lernen kommt unbemerkt zum Stillstand, obwohl die Gradienten unauffällig aussehen.

<!--en:facb2fdc-->
In diesem Abschnitt zeigen wir, wie sich beide Probleme vermeiden lassen, damit das Training mit 16 Bit zuverlässig funktioniert. Jeder Tensor hat einen *Datentyp* (engl. dtype), der sein Zahlenformat festlegt. Jedes Format wägt den *dynamischen Bereich* (engl. dynamic range) gegen die *Genauigkeit* (engl. precision) ab. Ersterer bestimmt, wie groß oder klein die darstellbaren Zahlen sein können, Letztere, wie eng benachbarte Zahlen beieinanderliegen. Beim *Training mit gemischter Genauigkeit* (engl. mixed-precision training) behalten wir die Hauptgewichte (engl. master weights) in fp32. So summieren sich auch winzige Gradientenupdates zuverlässig. Rechenintensive Matrizenmultiplikationen führen wir mit 16 Bit aus, und auch die Aktivierungen speichern wir mit 16 Bit. Bei fp16 verhindert *Loss-Scaling*, dass kleine Gradienten durch Unterlauf zu null werden. In :numref:`sec_parameters` haben wir die Bytes pro Parameter gezählt, hier steuern wir die Anzahl der Bytes pro Zahl. Der zweite Unterabschnitt zur automatischen Typumwandlung und zur Umwandlung ganzer Modelle dient zum Nachschlagen. Beim ersten Lesen können Sie ihn überfliegen. Alle Beispiele laufen auf CPU oder GPU. Einen Geschwindigkeitsgewinn durch 16 Bit erzielen wir erst mit den Matrixrecheneinheiten einer GPU. Wir messen ihn in :numref:`sec_use_gpu`.

<!--en:a88f1a64-->
## Gleitkommaformate

<!--en:86cc19a8-->
### Wertebereich und Genauigkeit

<!--en:cadfff4a-->
Schon beim Quadrieren von 300 in den beiden 16-Bit-Formaten zeigen sich zwei unterschiedliche Folgen geringer Genauigkeit. Der Wert 300 ist in einem neuronalen Netz keineswegs ungewöhnlich. Ein unnormalisiertes Logit oder eine Teilsumme bei der Varianzberechnung kann diese Größe leicht erreichen.

<!--en:1b3a70ab-->
MXNet unterstützt bf16 nur teilweise und kann keine bf16-Gewichte trainieren. Daher arbeiten wir hier mit fp16 und quadrieren 300 nur in fp16. In bf16 wäre das Ergebnis 90 112.

<!--en:03799ea0-->
In fp16 (*halbe Genauigkeit*) führt $300^2$ direkt zu einem Überlauf auf `inf`, weil der größte endliche Wert nur 65 504 beträgt. In bf16 (*brain floating point*) bleibt das Ergebnis endlich (90 112 statt 90 000), doch schon in der dritten signifikanten Stelle tritt ein Rundungsfehler auf. Beide Formate kommen mit 16 Bit aus, teilen die Bits aber unterschiedlich auf. Jede normale Gleitkommazahl speichert ein Vorzeichenbit $s$, einen Exponenten $e$ und eine Mantisse mit $t$ Bit. Die Mantisse kodiert einen Bruchteil $f \in [0, 1)$. Die so dargestellte Zahl ist

<!--en:1bde83c0-->
Die Exponentenbits bestimmen den dynamischen Bereich, also den größten und kleinsten darstellbaren Betrag $|x|$. Die Mantissenbits bestimmen die relative Genauigkeit. Zwischen $1$ und $2$ haben benachbarte Zahlen den Abstand $2^{-t}$, zwischen $2^e$ und $2^{e+1}$ den Abstand $2^{e-t}$. Der relative Abstand bleibt damit über alle Größenordnungen hinweg gleichmäßig. Unsere drei Formate teilen ihre Bits wie folgt auf:

<!--en:1aa87e21-->
| Format | Vorzeichen | Exponent | Mantisse |

<!--en:c0692661-->
| fp32   | 1    | 8        | 23       |

<!--en:43191e09-->
| bf16   | 1    | 8        | 7        |

<!--en:3201724b-->
| fp16   | 1    | 5        | 10       |

<!--en:d29aceba-->
Hier wird der Zielkonflikt sichtbar. bf16 behält alle 8 Exponentenbits von fp32 bei und verkürzt die Mantisse auf 7 Bit. Damit deckt bf16 bei gröberer Auflösung den vollen dynamischen Bereich von fp32 ab. fp16 setzt den Schwerpunkt anders. Mit 10 Mantissenbits und nur 5 Exponentenbits bietet es feinere Schritte, aber einen viel engeren Wertebereich. Beide Ergebnisse von oben lassen sich von Hand nachrechnen. Bei 5 Exponentenbits durchläuft $e$ das Intervall $[-14, 15]$. Damit liegt die Obergrenze für fp16 bei $2^{15}(2 - 2^{-10}) = 65504$. Mit 7 Mantissenbits haben bf16-Werte zwischen $2^{16}$ und $2^{17}$ den Abstand $2^9 = 512$. Also liegt 90 000 zwischen $175 \cdot 512 = 89600$ und $176 \cdot 512 = 90112$ und rundet auf den näheren Wert 90 112.

<!--en:149abb7b-->
:numref:`fig_bg_float-formats` stellt die Bitaufteilungen nebeneinander dar (mehr zur IEEE-Gleitkommaarithmetik in :numref:`sec_mdl-numerical-stability-conditioning`).

<!--en:69f19240-->
![Bitaufteilung der Formate aus der Tabelle. Die Balken sind am Vorzeichenbit ausgerichtet. Ein längerer Balken steht für mehr Bits. bf16 übernimmt den 8-Bit-Exponenten von fp32. fp16 und bf16 teilen dieselben 16 Bits mit entgegengesetzter Gewichtung auf.](../img/bg-float-formats.svg)

<!--en:0b37d631-->
Mit `finfo` können wir die exakten numerischen Grenzen jedes Formats abfragen. Wir betrachten drei Felder: `max` (größter endlicher Wert), `tiny` (kleinster positiver *normaler* Wert) und `eps` (der Abstand $2^{-t}$ direkt oberhalb von 1). Unterhalb von `tiny` hat der Exponent seine Untergrenze erreicht. Dort fällt die Genauigkeit bis zur Null allmählich ab, denn die *subnormalen* Zahlen haben einen festen Abstand. In fp16 sind sie Vielfache von $2^{-24}$. Ein Wert nahe $3 \times 10^{-5}$ behält daher noch 9 signifikante Bits, ein Wert nahe $10^{-7}$ nur noch ein einziges Bit.

<!--en:8f3901aa-->
`mnp.finfo` bezeichnet den kleinsten normalen Wert als `smallest_normal` statt als `tiny`. Für bf16, das in der Codezelle fehlt, lauten die drei Werte `max` $3.39 \times 10^{38}$, `tiny` $1.18 \times 10^{-38}$ und `eps` $2^{-7} \approx 0.0078$.

<!--en:b1df3d91-->
TensorFlow hat für bf16 kein eigenes `finfo`. Die Zahlen stammen aus `ml_dtypes`, dem kleinen Paket, das den von TensorFlow verwendeten bf16-Typ definiert.

<!--en:41b1e754-->
Da bf16 denselben 8-Bit-Exponenten wie fp32 verwendet, ist `tiny` für beide Formate gleich und `max` fast gleich. Sein `eps` von $0.0078$ entspricht einer Genauigkeit von zwei bis drei Dezimalstellen. fp16 löst drei bis vier Dezimalstellen auf, läuft aber oberhalb von 65 504 über. Unterhalb von $6.1 \times 10^{-5}$ beginnt der subnormale Bereich. Unterhalb von $2^{-24} \approx 6.0 \times 10^{-8}$ ist keine positive Zahl darstellbar. Welche Einschränkung wiegt beim Deep Learning schwerer? Aktivierungen und Gradienten erstrecken sich regelmäßig über viele Größenordnungen. Ein zu kleiner Wertebereich hat gravierende Folgen, weil Zahlen zu `inf` oder null werden. Eine etwas gröbere Auflösung führt nur zu leichtem Rundungsrauschen. Deshalb ist bf16 auf moderner Hardware das übliche 16-Bit-Format für das Training. fp16 kommt vor allem auf älteren GPUs und bei der Inferenz zum Einsatz.

<!--en:b1be5b05-->
### Rundung kleiner Updates

<!--en:89914b9b-->
Bei Parameterupdates wirkt sich das grobe Raster von bf16 *tatsächlich* nachteilig aus. Ein Schritt des Gradientenabstiegs ändert ein Gewicht um $\eta g$, oft nur um $10^{-3}$ seines Betrags oder weniger. Runden wir auf die nächstgelegene darstellbare Zahl, geht jeder Schritt verloren, der kleiner als der halbe lokale Rasterabstand ist. Im Folgenden addieren wir in jedem Format $10^{-3}$ und $10^{-4}$ jeweils zu einem Gewicht von $1$.

<!--en:a6e8a227-->
Für bf16, das in der Codezelle fehlt, ergeben beide Summen `1.0000000`.

<!--en:cfd77475-->
In fp32 kommen beide Änderungen beim Gewicht an. In bf16 beträgt der Abstand nahe 1 dagegen $2^{-7} \approx 0.0078$. Beide Summen runden wieder auf 1, das Gewicht bleibt unverändert. In fp16 (Abstand $2^{-10} \approx 0.00098$) bleibt der Schritt von $10^{-3}$ erhalten. Er rundet auf einen Rasterschritt, während $10^{-4}$ verschwindet. :numref:`fig_bg_update-swamping` zeigt beide Raster im selben Maßstab. Ein Update wirkt sich nur aus, wenn es das Gewicht um mehr als einen halben Rasterabstand verschiebt. Speichern wir Gewichte direkt mit 16 Bit, kann die Optimierung deshalb selbst bei lauter korrekten Gradienten zum Stillstand kommen. Deshalb behalten wir beim Training Hauptgewichte in fp32.

<!--en:4b821142-->
![Das bf16-Raster (oben) und das fp16-Raster (unten) nahe einem Gewicht von 1 im selben Maßstab. Im schattierten Intervall liegen die Werte, die wieder auf 1 runden. Es reicht auf jeder Seite bis zum halben Rasterabstand. Unterhalb von 1 ist der Abstand halb so groß wie oberhalb. In bf16 liegt das Update $1 + 10^{-3}$ im Intervall und geht verloren. In fp16 liegt es außerhalb und rundet auf den nächsten Rasterpunkt $1 + 2^{-10}$.](../img/bg-update-swamping.svg)

<!--en:5ed0c3f8-->
## Automatische Typumwandlung und Umwandlung ganzer Modelle

<!--en:e2b34395-->
### Automatische Typumwandlung

<!--en:c3d1eb2d-->
Welchen Datentyp hat das Ergebnis, wenn ein Ausdruck Operanden unterschiedlicher Typen kombiniert? Wir addieren einen fp16-Tensor zunächst zu einem fp32-Tensor, dann zu einer Python-Gleitkommazahl und schließlich zu einem ganzzahligen Tensor.

<!--en:631308e1-->
Bei zwei Gleitkommatensoren gilt das breitere Format (bei fp16 + bf16 ergibt sich fp32). So kann ein einzelner versehentlich eingesetzter fp32-Tensor unbemerkt eine ganze nachfolgende 16-Bit-Berechnung auf fp32 umstellen. Gleitkommaliterale in Python und ganzzahlige Tensoren ändern den Typ eines Gleitkommatensors nicht.

<!--en:1653e4aa-->
TensorFlow wandelt unterschiedliche Tensortypen nie automatisch ineinander um. Wer fp16 und fp32 kombiniert, erhält einen `InvalidArgumentError`, sofern die Typen nicht explizit mit `tf.cast` angeglichen wurden.

<!--en:cb7a9569-->
### Umwandlung eines Modells

<!--en:67db367f-->
Da Parameter selbst Tensoren sind, können wir ein ganzes Modell auf 16 Bit umstellen. In diesem Abschnitt verwenden wir durchgehend unser Fashion-MNIST-MLP aus :numref:`sec_mlp-implementation`. Es hat 784 Eingaben, eine versteckte ReLU-Schicht mit 256 Einheiten und 10 Ausgaben. Die Gewichte folgen der Verteilung $\mathcal{N}(0, 0.01^2)$, die Bias-Terme sind null. Insgesamt belegen die 203 530 Parameter in fp32 genau 814 120 Bytes.

<!--en:78dbf419-->
`net.cast('float16')` wandelt alle Parameter im Baum der Blöcke um. Die Eingaben müssen wir in denselben Typ umwandeln.

<!--en:b5cae599-->
`net.to(dtype)` (oder `net.bfloat16()`) wandelt alle Parameter und Puffer um. Die Schichten weisen Eingaben zurück, deren Datentyp von dem ihrer Gewichte abweicht.

<!--en:746729d6-->
Mit `dtype='bfloat16'` legen wir beim Erzeugen jeder Keras-Schicht sowohl `variable_dtype` als auch `compute_dtype` fest. Beim Aufruf wandelt die Schicht ihre Eingaben automatisch um.

<!--en:0ffd0454-->
Schichten in Flax NNX unterscheiden zwischen dem Datentyp beim Speichern (`param_dtype`, standardmäßig fp32) und beim Rechnen (`dtype`). Wenden wir `.astype(jnp.bfloat16)` auf alle Parameter vom Typ `nnx.Param` an, wandeln wir die gespeicherten Gewichte um.

<!--en:2712a7d2-->
Durch die Umwandlung halbiert sich der Parameterspeicher sofort. Ob ein 16-Bit-Modell eine fp32-Eingabe automatisch umwandelt oder wegen des abweichenden Typs zurückweist, hängt vom Framework ab.

<!--en:1e386f60-->
Für die *Inferenz* halbiert die Umwandlung eines trainierten Modells auf 16 Bit sofort den Speicherbedarf seiner Gewichte. Dabei müssen wir prüfen, ob die Genauigkeit für die jeweilige Aufgabe erhalten bleibt. Beim *Training* mit reinen 16-Bit-Gewichten gehen dagegen kleine Updates wie eben beschrieben durch Rundung verloren. Das prüfen wir im folgenden Experiment.

<!--en:cc92e612-->
## Training mit gemischter Genauigkeit

<!--en:edcc5a6f-->
### Hauptgewichte und Berechnungen mit 16 Bit

<!--en:b73cffcb-->
Wir stehen scheinbar vor einem Dilemma. Zuverlässige Gewichtsänderungen erfordern 32 Bit beim Speichern, die Hardware rechnet aber mit 16 Bit schneller. *Training mit gemischter Genauigkeit* löst diesen Konflikt, indem es die Speicherung der Parameter von der Berechnung trennt :cite:`Micikevicius.Narang.Alben.ea.2018`. Wir behalten *Hauptgewichte* in fp32, damit sich auch kleine Updates zuverlässig summieren. Für die rechenintensiven Matrizenmultiplikationen und die Aktivierungen im Vorwärts- und Rückwärtsdurchlauf wandeln wir die Gewichte auf 16 Bit um. Numerisch empfindliche Reduktionen, etwa die Berechnung des Verlusts, führen wir weiterhin in fp32 aus. :numref:`fig_bg_amp-loop` zeigt einen Trainingsschritt im Überblick.

<!--en:5dd1d066-->
![Ein Trainingsschritt mit gemischter Genauigkeit. Der Vorwärtsdurchlauf verwendet die in bf16 umgewandelten fp32-Hauptgewichte und liefert bf16-Aktivierungen. Der Verlust wird in fp32 berechnet. Der Rückwärtsdurchlauf liefert die Gradienten, mit denen der Optimierer die fp32-Hauptgewichte aktualisiert. Bei fp16 wird zusätzlich der Verlust vor dem Rückwärtsdurchlauf hochskaliert. Vor dem Update werden die Gradienten wieder herunterskaliert.](../img/bg-amp-loop.svg)

<!--en:b02a9f42-->
Wir wandeln das Netz in fp16 um (`net.cast('float16')`) und übergeben dem Optimierer `multi_precision=True`. So führt er intern Hauptgewichte in fp32.

<!--en:d7722fe6-->
`torch.autocast` belässt die Parameter in fp32 und stellt geeignete Operationen (Matrizenmultiplikationen, Faltungen) automatisch auf 16 Bit um. Reduktionen und Verlustberechnungen verbleiben in fp32.

<!--en:26e4ff12-->
Erzeugen wir ein Modell mit der globalen Einstellung `'mixed_bfloat16'`, erhält jede Schicht fp32 als `variable_dtype` und bf16 als `compute_dtype`.

<!--en:36fb72b5-->
Erzeugen wir NNX-Schichten mit `dtype=jnp.bfloat16` und belassen `param_dtype` beim Standardwert fp32, speichern sie die Hauptgewichte in fp32 und rechnen beim Vorwärtsdurchlauf in bf16.

<!--en:14102884-->
### Training in fp32 und mit 16 Bit

<!--en:2e2724c9-->
Wie stark beeinflusst 16-Bit-Arithmetik die Optimierung? Was geht schief, wenn wir auf die Hauptkopie in fp32 verzichten? Wir trainieren unser MLP dreimal mit einfachem SGD auf Fashion-MNIST. Jeder Minibatch umfasst 256 Bilder. Einmal verwenden wir durchgehend fp32, einmal reine bf16-Gewichte (ohne fp32-Hauptkopie) und einmal gemischte Genauigkeit. Alle drei Läufe beginnen mit denselben zufälligen Gewichten und verarbeiten genau dieselben Minibatches in derselben Reihenfolge.

<!--en:0f6313cc-->
Für die Lernrate wählen wir $\eta = 0.01$, ein Zehntel des Werts aus :numref:`sec_mlp-implementation`. Wir trainieren über 1000 Schritte (etwa vier Epochen). Unsere Betrachtung der Rundung erklärt den großen Einfluss von $\eta$. Ein Update $\eta g$ geht verloren, sobald sein Betrag unter den halben lokalen bf16-Rasterabstand fällt. Bei kleinerer Lernrate verschwinden also mehr Updates. Eine kleine Lernrate ist kein künstlicher Sonderfall. Jeder Trainingslauf mit abklingender Lernrate erreicht diesen Bereich in der Praxis. Unmittelbar vor dem Training messen wir für beide Lernraten, wie viel verloren geht.

<!--en:15c4aa5e-->
Wir laden die Trainingsminibatches und 1024 Bilder aus den Validierungsdaten vorab in den Speicher. Dann führt `train` insgesamt $1{,}000$ SGD-Schritte mit $\eta = 0.01$ aus und gibt den Verlust jedes Schritts sowie die abschließende Validierungsgenauigkeit zurück.

<!--en:630d971c-->
Ein Detail der Implementierung von `train` bewährt sich unabhängig von der gewählten Genauigkeit. Während der Schleife lassen wir die Verlusttensoren der einzelnen Schritte auf dem Gerät. Erst am Ende wandeln wir sie auf einmal in Python-Gleitkommazahlen um. So entfällt die Wartezeit für die Synchronisierung von CPU und GPU bei jedem einzelnen Schritt (:numref:`sec_use_gpu`).

<!--en:826ab51d-->
Da MXNet bf16-Gewichte nicht direkt trainieren kann, rundet `bf16_round` die fp32-Gewichte nach jedem Schritt auf die 7-Bit-Mantisse von bf16. Beim Training mit gemischter Genauigkeit verwenden wir fp16 mit `multi_precision=True`.

<!--en:3c56a48e-->
Vor dem Training überprüfen wir unsere Überlegung zur Rundung am ersten Minibatch. Wir betrachten die von null verschiedenen Updates $\eta g$ der ersten Schicht bei der Initialisierung. Welcher Anteil geht beim Speichern in bf16 für $\eta = 0.01$ und für $\eta = 0.1$ durch Rundung verloren? Welchen Anteil an der Summe aller Updatebeträge haben die verlorenen Updates?

<!--en:e3227b36-->
Bei $\eta = 0.01$ verschwinden etwa neun von zehn Updates durch Rundung. Sie machen rund vier Fünftel der Summe aller Updatebeträge aus („mass“ in der Ausgabe). Der größte Teil des Gradientenschritts erreicht die Gewichte also nie. Bei $\eta = 0.1$ geht noch immer etwa die Hälfte der Updates verloren. Es sind jedoch die kleinen Updates, die nur rund 15% des Gesamtbetrags ausmachen. Der Gradientenschritt kommt dann zum größten Teil bei den Gewichten an, und das Problem bleibt verborgen, bis die Lernrate sinkt. Nun trainieren wir alle drei Varianten mit $\eta = 0.01$.

<!--en:a0e135f0-->
Die Verlustkurven für fp32 und gemischte Genauigkeit liegen fast genau übereinander. Die Validierungsgenauigkeiten weichen höchstens um etwa einen Prozentpunkt voneinander ab. Das Modell mit reinen bf16-Gewichten fällt dagegen schon innerhalb der ersten hundert Schritte zurück. Nach 1000 Schritten liegt sein Verlust noch bei etwa 1.15 gegenüber 0.85, seine Validierungsgenauigkeit bei etwa 64% gegenüber 68%. Wie unser Experiment mit Skalaren erwarten ließ, gehen die meisten Updates der ersten Schicht durch Rundung verloren, wenn wir Gewichte direkt in bf16 speichern. Bei gemischter Genauigkeit kommen die Updates zuverlässig bei den fp32-Hauptgewichten an. Keiner der drei Läufe reicht bis zur Konvergenz. Für den direkten Vergleich zählen die identischen Daten und die gleiche Schrittzahl.

<!--en:24829a89-->
Zwei Einschränkungen sind für die Beurteilung der gemischten Genauigkeit wichtig. Erstens verkürzt sich die Laufzeit bei einem winzigen MLP auf der CPU nicht. Der Geschwindigkeitsgewinn bei 16 Bit entsteht, wenn spezialisierte Tensor Cores auf der GPU große Matrizen multiplizieren. Das messen wir in :numref:`subsec_async_dispatch`. Zweitens senkt gemischte Genauigkeit den statischen Mindestspeicherbedarf pro Parameter aus :numref:`sec_parameters` *nicht*, denn Hauptgewichte und Momente des Optimierers bleiben in fp32. Die eingangs genannten 16 GB gelten für ein Modell mit 8 Milliarden Parametern, das wir für die *Inferenz* umgewandelt haben. Beim *Training* spart gemischte Genauigkeit Speicher für Aktivierungen (mit 16 Bit brauchen sie halb so viel Platz) und Rechenzeit für Matrizenmultiplikationen.

<!--en:734dcac4-->
In MXNet haben wir beim Training mit gemischter Genauigkeit oben in fp16 statt in bf16 gerechnet und auf Loss-Scaling verzichtet. Im nächsten Unterabschnitt messen wir, wie sich fp16 auf die Gradienten dieses Netzes auswirkt, und klären, wann eine Skalierung nötig ist.

<!--en:f22f8c6a-->
### Loss-Scaling für fp16

<!--en:d25bafc9-->
Bei bf16 genügen die Hauptgewichte. Bei fp16 birgt der enge Exponentenbereich am unteren Ende eine weitere Gefahr. Kleine Gradienten erleiden einen *Unterlauf* (engl. underflow). Beträge unter $2^{-14} \approx 6.1 \times 10^{-5}$ fallen in den subnormalen Bereich und verlieren signifikante Bits. Alles unter der Hälfte der kleinsten positiven subnormalen Zahl ($\approx 3 \times 10^{-8}$) rundet direkt auf null. Ein Gradient von $10^{-8}$ verschwindet in fp16. Multiplizieren wir ihn jedoch mit $2^{16}$, liegt er sicher im normalen Bereich.

<!--en:ca471153-->
Wie häufig sind winzige Gradienten in der Praxis? Wir berechnen für den ersten Minibatch die fp32-Gradienten der ersten Schicht unseres MLP. Dann messen wir, welcher Anteil der von null verschiedenen Einträge in fp16 subnormal oder null wird, einmal ohne Skalierung und einmal nach Skalierung mit $2^{16}$.

<!--en:216f4fab-->
Etwa ein Drittel der von null verschiedenen Gradienten der ersten Schicht fällt in den subnormalen Bereich von fp16. Etwa zwei von tausend werden durch Unterlauf sogar zu null. Bei diesem kleinen MLP sind die Folgen gering, weil nur die allerkleinsten Einträge verschwinden. Es ist sogar schwierig, ein kleines Beispielmodell zu konstruieren, bei dem fp16 völlig versagt und Loss-Scaling das Training rettet. Schwerwiegender Unterlauf trifft große, tiefe Modelle, bei denen sich der Verlust als Mittelwert über viele Token oder Pixel ergibt und die Gradienten über Dutzende von Schichten abklingen. :citet:`Micikevicius.Narang.Alben.ea.2018` konnten mehrere ImageNet-Klassifikatoren problemlos in fp16 ohne Skalierung trainieren. Ein Objektdetektor divergierte jedoch, und ein großes rekurrentes Sprachmodell erreichte die Genauigkeit von fp32 erst mit Loss-Scaling.

<!--en:b7601f70-->
Multiplizieren wir die Gradienten mit $2^{16}$, bleibt kein einziger Eintrag subnormal. Alle Werte bleiben dabei deutlich unter der fp16-Überlaufgrenze von 65 504. Darauf beruht *Loss-Scaling*. Vor dem Rückwärtsdurchlauf multiplizieren wir den skalaren Verlust mit einer großen Konstanten $S$. Wegen der Linearität der Ableitung wächst bei der Backpropagation auch jeder Zwischengradient um den Faktor $S$. Vor dem Optimierungsschritt teilen wir die Gradienten in fp32 wieder durch $S$. Dort lassen sich auch winzige Werte genau darstellen. Mathematisch bleibt das Parameterupdate dadurch unverändert, während die fp16-Zwischengradienten oberhalb des Unterlaufbereichs bleiben.

<!--en:03953dc7-->
Ein fester Wert für $S$ ist unzuverlässig. Ist er zu klein, kommt es bei Gradienten zum Unterlauf. Ist er zu groß, laufen große Gradienten auf `inf` über. *Dynamisches Loss-Scaling* automatisiert die Wahl des Faktors. Wir beginnen mit einem großen $S$ und prüfen die Gradienten bei jedem Schritt. Sobald `inf` oder NaN auftritt, überspringen wir das Update des Optimierers und halbieren $S$. Nach mehreren fehlerfreien Schritten in Folge verdoppeln wir $S$, um nahe der Obergrenze von fp16 zu bleiben. :numref:`fig_bg_loss-scaling` zeigt einen Schritt dieser Schleife.

<!--en:a1d50e83-->
![Ein Schritt beim dynamischen Loss-Scaling. Nach Multiplikation des Verlusts mit $S$ läuft der Rückwärtsdurchlauf in fp16. Anschließend werden die Gradienten in fp32 durch $S$ geteilt. Sind alle endlich, führt der Optimierer seinen Schritt aus. Nach mehreren solchen Schritten in Folge verdoppelt sich $S$. Ist ein Gradient `inf` oder NaN, entfällt der Schritt und $S$ halbiert sich.](../img/bg-loss-scaling.svg)

<!--en:ecadaad3-->
Der `Trainer` von Gluon unterstützt kein automatisches Loss-Scaling. Multiplizieren Sie beim Training in fp16 den Verlust vor dem Rückwärtsdurchlauf mit $S$ und teilen Sie die Gradienten vor dem Update des Optimierers durch $S$.

<!--en:e7b1f7ed-->
Optax unterstützt kein automatisches Loss-Scaling (JAX ist vor allem auf TPUs und moderne GPUs mit nativer bf16-Unterstützung ausgelegt). Multiplizieren Sie beim Training in fp16 den Verlust vor dem Rückwärtsdurchlauf mit $S$ und teilen Sie die Gradienten vor dem Update des Optimierers durch $S$.

<!--en:06e204d0-->
`torch.amp.GradScaler` automatisiert das dynamische Loss-Scaling beim Einsatz von fp16 mit `autocast`:

<!--en:47632e7d-->
`tf.keras.mixed_precision.LossScaleOptimizer` umschließt einen Optimierer. Unter `'mixed_float16'` skaliert er den Verlust, macht die Skalierung der Gradienten rückgängig und prüft sie:

<!--en:e2449b05-->
Auf moderner Hardware mit nativer bf16-Unterstützung (neuere GPUs, TPUs und Server-CPUs) ist daher gemischte Genauigkeit mit bf16 ohne Loss-Scaling der Standard. Auf älteren GPUs bleibt fp16 mit dynamischem Loss-Scaling unverzichtbar.

<!--en:b84ad983-->
Eine weitere Hardwarefunktion beschleunigt fp32-Modelle, ohne den Datentyp eines Tensors zu ändern. Seit der Ampere-Architektur von NVIDIA können Tensor Cores fp32-Matrizenmultiplikationen im *tf32*-Modus ausführen. Die Eingaben werden auf die 10-Bit-Mantisse von fp16 gekürzt, behalten aber den 8-Bit-Exponenten von fp32. Bei inneren Produkten werden die Summanden mit voller fp32-Genauigkeit aufsummiert. Das beschleunigt die Berechnung erheblich, und das Ergebnis stimmt in drei bis vier signifikanten Stellen mit einer reinen fp32-Rechnung überein.

<!--en:28a4867a-->
MXNet aktiviert tf32 auf unterstützten GPUs standardmäßig (steuerbar über `NVIDIA_TF32_OVERRIDE=0`).

<!--en:94513602-->
TensorFlow aktiviert tf32 auf unterstützten GPUs standardmäßig (steuerbar über `tf.config.experimental.enable_tensor_float_32_execution(False)`).

<!--en:19618544-->
JAX aktiviert tf32 auf unterstützten GPUs standardmäßig (steuerbar über `jax.default_matmul_precision('float32')`).

<!--en:c220008e-->
PyTorch verwendet standardmäßig reine fp32-Matrizenmultiplikationen. Mit `torch.set_float32_matmul_precision('high')` (:numref:`sec_use_gpu`) lässt sich tf32 aktivieren.

<!--en:c276086f-->
## Numerische Fehlerquellen

<!--en:eee5ba43-->
Die allermeisten numerischen Fehler in 16-Bit-Code lassen sich auf drei Ursachen zurückführen. Ein früher Überlauf macht sich erst später als NaN bemerkbar. Eine Exponentialfunktion ohne vorherige Verschiebung läuft über, bevor der abschließende Logarithmus den Wert wieder verkleinern kann. Oder eine lange Summe wächst nicht mehr, sobald die einzelnen Summanden unter die Rundungsschwelle des Akkumulators fallen.

<!--en:36f7cf6d-->
### Vom Überlauf zu NaN

<!--en:53cc5294-->
Ein Verlust mit dem Wert NaN ist selten die eigentliche Ursache. Meist geht er auf einen früheren Überlauf auf `inf` zurück, denn unbestimmte Ausdrücke wie `inf - inf` oder `inf / inf` ergeben NaN.

<!--en:2bc98b26-->
Der erste Überlauf kann schon mehrere Schichten früher auftreten, lange bevor NaN den Verlust erreicht. Sobald ein einziges Parameterupdate ein NaN enthält, sind alle nachfolgenden Schritte betroffen. Wird der Verlust zu NaN, prüfen Sie zuerst die Wertebereiche. Rechnet eine Operation in fp16 nahe der Obergrenze von 65 504? Erst danach sollten Sie nach einem Fehler in der Lernrate oder der Architektur suchen.

<!--en:226bcd8d-->
Um bei der Fehlersuche sofort bei der ersten Operation anzuhalten, die `inf` oder NaN liefert, aktivieren Sie `tf.debugging.enable_check_numerics()`.

<!--en:df7541ec-->
Für `inf` ist eine gesonderte Prüfung nötig. Um bei der Fehlersuche bei der ersten Operation anzuhalten, die NaN liefert, aktivieren Sie `jax.config.update('jax_debug_nans', True)`.

<!--en:d01706d3-->
### Log-Sum-Exp

<!--en:16493783-->
Für die Kreuzentropie mit Softmax-Funktion müssen wir $\log \sum_i \exp(x_i)$ berechnen. Werten wir den Ausdruck direkt aus, kann es in fp16 selbst bei einem kleinen Endergebnis zum Überlauf kommen.

<!--en:69bb28de-->
Das tatsächliche Ergebnis beträgt nur 12.4, doch der Zwischenterm $e^{12} \approx 163{,}000$ übersteigt die fp16-Grenze von 65 504 deutlich. Abhilfe schafft genau die Verschiebung aus :numref:`subsec_softmax-implementation-revisited`. Vor dem Exponenzieren ziehen wir den größten Eintrag $m = \max_i x_i$ ab und addieren $m$ anschließend wieder:

<!--en:4e5bdd13-->
Jeder Exponent auf der rechten Seite ist kleiner oder gleich null. Ein Überlauf ist damit ausgeschlossen. Der größte Summand ist genau $e^0 = 1$, sodass auch kein Unterlauf zu $\log 0$ führen kann. In der Praxis müssen Sie diese Verschiebung selten selbst programmieren. Die eingebauten Routinen für Log-Softmax und Kreuzentropie wenden sie automatisch an.

<!--en:a770a33b-->
### Akkumulation in fp32

<!--en:00adeb21-->
Bei langen Summen in 16 Bit kann die Zwischensumme unbemerkt aufhören zu wachsen. Sobald sie groß genug ist, sind die kleinen Summanden jeweils kleiner als der halbe lokale Rasterabstand. Ihr Beitrag verschwindet beim Runden. Wir addieren im Folgenden $2^{-10}$ (in fp16 *exakt* darstellbar) jeweils 4096-mal zu einem fp16-Akkumulator und zum Vergleich zu einem fp32-Akkumulator.

<!--en:4a20d599-->
Die exakte Summe beträgt 4. Der fp32-Akkumulator erreicht sie genau. Der fp16-Akkumulator bleibt nach 2048 Schritten bei 2 stehen. Zwischen 2 und 4 haben benachbarte fp16-Werte den Abstand $2^{-9}$. Daher liegt $2 + 2^{-10}$ genau in der Mitte. Beim Runden auf den nächsten Wert mit gerader Endziffer ergibt sich jedes Mal wieder 2. Dasselbe Risiko besteht beim Mitteln über einen Batch, beim Aufsummieren der Verluste einer Epoche und bei der Varianzberechnung in Normalisierungsschichten. Deshalb verwenden Verfahren mit gemischter Genauigkeit für Reduktionen und Normalisierungen fp32. Auch bei eigenen Reduktionen über 16-Bit-Tensoren sollten Sie in fp32 akkumulieren.

<!--en:d7f562b9-->
Alle drei Fehler haben dieselbe Ursache. Ein Zwischenwert überschreitet den Wertebereich oder erfordert mehr Genauigkeit, als ein 16-Bit-Format bietet, selbst wenn das Endergebnis problemlos hineinpasst. Beachten Sie auch, dass gemischte Genauigkeit und tf32 konstruktionsbedingt die niederwertigsten Bits der Rechenergebnisse ändern. Mit reinem fp32 stimmen sie daher nicht durchgängig bitgenau überein. Wann sich Läufe bitgenau wiederholen *lassen* und wie das gelingt, behandelt :numref:`sec_repro`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:1306b986-->
Bei Gleitkommaformaten besteht ein Zielkonflikt zwischen dynamischem Bereich (Exponentenbits) und Genauigkeit (Mantissenbits). fp16 bietet feinere lokale Schritte, aber einen engen Wertebereich. Oberhalb von 65 504 kommt es zum Überlauf, bei winzigen Gradienten zum Unterlauf. bf16 behält den vollen 8-Bit-Exponenten von fp32, erreicht aber nur zwei bis drei Dezimalstellen Genauigkeit. Speichern wir Gewichte direkt mit 16 Bit, gehen kleine Gradientenupdates durch Rundung verloren und das Training gerät ins Stocken. Beim Training mit gemischter Genauigkeit behalten wir deshalb fp32-Hauptgewichte, multiplizieren Matrizen mit 16 Bit und speichern Aktivierungen mit 16 Bit. Unser Fashion-MNIST-MLP erreicht damit die Genauigkeit von fp32, während reine bf16-Gewichte zurückbleiben. Bei fp16 hebt dynamisches Loss-Scaling die Gradienten während der Backpropagation aus dem subnormalen Bereich und verhindert Unterlauf. Bei bf16 entfällt dieser Schritt. Schützen Sie schließlich Code mit geringer Genauigkeit vor NaN-Werten infolge eines Überlaufs. Verwenden Sie dafür Log-Sum-Exp mit Verschiebung und bei langen Reduktionen und Normalisierungen fp32-Akkumulatoren.

<!--en:2294243a-->
## Übungen

<!--en:cbbfb276-->
1. **Speicherbedarf bei gemischter Genauigkeit.** Beim Training mit Adam in fp32 belegt jeder Parameter 16 Bytes (:numref:`sec_parameters`). Nehmen Sie an, `train` verwende Adam anstelle von SGD.
    1. In welchem Datentyp speichert Ihr Framework beim Training mit gemischter Genauigkeit die Gewichte, die Gradienten und die gleitenden Mittelwerte von Adam zwischen den Schritten? Sinkt durch gemischte Genauigkeit die Anzahl der Bytes pro Parameter?
    1. Zählen Sie die Eingaben jeder linearen Schicht und die Logits als gespeicherte Aktivierungen. Rechnen Sie für das MLP von `train` die Anzahl pro Beispiel in Bytes um, einmal für fp32 und einmal für das 16-Bit-Format, das `train` verwendet. Bei welcher Batchgröße entspricht die Speicherersparnis bei den Aktivierungen dem Speicherbedarf des dauerhaft gespeicherten Trainingszustands dieses MLP?

<!--en:1619a51c-->
2. **Loss-Scaling bei Überlauf.** Beim dynamischen Loss-Scaling (:numref:`subsec_loss_scaling`) entfällt ein Schritt, wenn seine zurückskalierten Gradienten `inf` oder NaN enthalten. Der Skalierungsfaktor halbiert sich dann. Nach $N$ fehlerfreien Schritten in Folge verdoppelt er sich.
    1. Bei einem übersprungenen Schritt müssen die gleitenden Mittelwerte von Adam unverändert bleiben. Erklären Sie, was schiefginge, wenn der Optimierer sie mit dem nicht endlichen Gradienten aktualisieren würde. Was ginge schief, wenn er zuvor die nicht endlichen Einträge durch Nullen ersetzte?
    1. Erklären Sie, warum der Skalierungsfaktor nach einer Folge fehlerfreier Schritte wieder wachsen muss. Warum reicht es nicht, ihn immer nur zu verkleinern?
    1. Nehmen Sie an, dass bei einem Skalierungsfaktor oberhalb eines festen Schwellenwerts jeder Schritt zu einem Überlauf führt und unterhalb davon kein einziger. Welcher Anteil der Schritte entfällt in Abhängigkeit von $N$, nachdem sich der Skalierungsfaktor eingependelt hat? Welchen Nachteil hat ein größeres $N$, wenn die Annahme eines festen Schwellenwerts entfällt?

<!--en:75c0bf01-->
3. [code] **Normalisierung in fp16.** Verfahren mit gemischter Genauigkeit führen Normalisierungsschichten in fp32 aus. Um den Grund dafür zu verstehen, betrachten Sie RMSNorm mit einem Verstärkungsfaktor von eins (in :numref:`sec_custom_layer` ist RMSNorm als Modul implementiert). RMSNorm teilt eine Eingabezeile $\mathbf{x}$ der Breite $d$ durch ihren quadratischen Mittelwert mit einem kleinen Zusatz unter der Wurzel:

   $$\frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}},$$

   Dabei schützt die kleine Konstante $\epsilon$ vor einer Division durch null. Führen Sie die Berechnung vollständig in fp16 aus, und zwar für einen Batch aus 64 Zeilen der Breite $d = 1024$, deren Einträge eine Standardabweichung von etwa 100 haben.
    1. Welche Zwischengröße wird zuerst zu `inf`, und was gibt die Schicht für die betroffenen Zeilen zurück?
    1. Nehmen Sie an, die Quadrate seien exakt. Ab welcher Standardabweichung der Eingaben übersteigt ihr Mittelwert das fp16-Maximum? Ab welcher Standardabweichung übersteigt die Summe über die $d$ Einträge diese Grenze?

<!--en:d15472d0-->
4. **Das Format e4m3fn.** ● Das fp8-Format `float8_e4m3fn` besitzt ein Vorzeichenbit, 4 Exponentenbits mit einem Exponentenoffset von 7 und 3 Mantissenbits. Leiten Sie `tiny` und `eps` her und vergleichen Sie beide mit den Werten von fp16 und bf16. Erklären Sie den Namen `e4m3fn`. Ein Format mit derselben Bitaufteilung würde nach IEEE-Konvention das Exponentenmuster aus lauter Einsen für `inf` und NaN reservieren. Berechnen Sie den größten endlichen Wert dieses Formats und erklären Sie, wie `float8_e4m3fn` stattdessen einen größten endlichen Wert von 448 erreicht.

<!--en:1cacf75c-->
4. [code] **Das Format e4m3fn.** ● Geben Sie `torch.finfo(torch.float8_e4m3fn)` aus und vergleichen Sie `max`, `tiny` und `eps` mit den entsprechenden Werten von fp16 und bf16. Erklären Sie den Namen `e4m3fn`. Ein Format mit einem 4-Bit-Exponenten und einer 3-Bit-Mantisse würde nach IEEE-Konvention das Exponentenmuster aus lauter Einsen für `inf` und NaN reservieren. Berechnen Sie den größten endlichen Wert dieses Formats und erklären Sie, wie `e4m3fn` stattdessen einen `max`-Wert von 448 erreicht.

<!--en:7cbee749-->
4. [code] **Das Format e4m3fn.** ● Geben Sie `ml_dtypes.finfo(ml_dtypes.float8_e4m3fn)` aus und vergleichen Sie `max`, `tiny` und `eps` mit den entsprechenden Werten von fp16 und bf16. Erklären Sie den Namen `e4m3fn`. Ein Format mit einem 4-Bit-Exponenten und einer 3-Bit-Mantisse würde nach IEEE-Konvention das Exponentenmuster aus lauter Einsen für `inf` und NaN reservieren. Berechnen Sie den größten endlichen Wert dieses Formats und erklären Sie, wie `e4m3fn` stattdessen einen `max`-Wert von 448 erreicht.

<!--en:3e379431-->
4. [code] **Das Format e4m3fn.** ● Geben Sie `jnp.finfo(jnp.float8_e4m3fn)` aus und vergleichen Sie `max`, `tiny` und `eps` mit den entsprechenden Werten von fp16 und bf16. Erklären Sie den Namen `e4m3fn`. Ein Format mit einem 4-Bit-Exponenten und einer 3-Bit-Mantisse würde nach IEEE-Konvention das Exponentenmuster aus lauter Einsen für `inf` und NaN reservieren. Berechnen Sie den größten endlichen Wert dieses Formats und erklären Sie, wie `e4m3fn` stattdessen einen `max`-Wert von 448 erreicht.

<!--en:09740702-->
[Diskussion](https://d2l.discourse.group/t/36756)

<!--en:f531cb1a-->
Numerische Genauigkeit und Training mit gemischter Genauigkeit<br>**fp32, bf16 und fp16 im Vergleich · Rundung von Updates · fp32-Hauptgewichte · dynamisches Loss-Scaling · numerische Risiken**

<!--en:0fac2724-->
::: {.slide title="16-Bit-Formate halbieren den Speicherbedarf und bergen zwei Risiken"}

<!--en:41974952-->
[Der Preis geringer Genauigkeit]{.kicker}

<!--en:8711c691-->
- **Überlauf in `fp16`:** Beim Quadrieren von $300.0$ kommt es zum Überlauf auf `inf` ($\max = 65{,}504$).

<!--en:7016a4d2-->
- **Rundung in `bf16`:** In `bf16` bleibt $300^2$ endlich ($90{,}112$). Doch $1.0 + 10^{-3}$ rundet wieder auf $1.0$. Bei direkten 16-Bit-Updates bleibt das Gewicht unverändert.

<!--en:df2e5164-->
::: {.slide title="Aufbau einer Gleitkommazahl: Wertebereich und Genauigkeit"}

<!--en:ffea472c-->
[Exponentenbits und Mantissenbits]{.kicker}

<!--en:8542ff32-->
- **Die Exponentenbits bestimmen den dynamischen Bereich:** Mit 8 Exponentenbits decken `fp32` und `bf16` denselben Bereich von $\sim 10^{\pm 38}$ ab. Mit 5 Exponentenbits reicht `fp16` nur bis $65{,}504$.

<!--en:8bd06156-->
- **Die Mantissenbits ($t$) bestimmen die relative Genauigkeit:** In jeder Oktave $[2^e, 2^{e+1})$ haben benachbarte Zahlen den Abstand $2^{e-t}$ ($\text{eps} = 2^{-t}$).

<!--en:d4e43903-->
::: {.slide title="Bitaufteilung von fp32, bf16 und fp16 im Vergleich"}

<!--en:8b8d45c5-->
[Zwei Aufteilungen für 16 Bit]{.kicker}

<!--en:e282682a-->
- **`bf16` verkürzt `fp32`:** Mit allen 8 Exponentenbits behält `bf16` den dynamischen Bereich von `fp32` vollständig bei. Überlauf und Unterlauf sind fast so selten wie beim Training mit 32 Bit.

<!--en:3c2484d6-->
- **`fp16` bevorzugt Mantissenbits:** Mit 10 Mantissenbits sind die Schritte $8\times$ feiner als in `bf16`. Mit 5 Exponentenbits kommt es oberhalb von $65{,}504$ zum Überlauf. Unterhalb von $6.1 \times 10^{-5}$ beginnt der subnormale Bereich.

<!--en:8b809e20-->
::: {.slide title="Numerische Grenzen mit finfo abfragen"}

<!--en:4f57abfc-->
[Exakte Grenzen der Formate]{.kicker}

<!--en:f2f6d721-->
Direkt im Code lassen sich `max` (größter endlicher Wert), `tiny` (kleinste positive normale Zahl) und `eps` (Abstand $2^{-t}$ bei $1.0$) abfragen:

<!--en:553ffcb8-->
- **Warum der Wertebereich beim Deep Learning wichtiger ist als die Genauigkeit:** Aktivierungen und Gradienten erstrecken sich über viele Schichten hinweg über viele Größenordnungen. Außerhalb des darstellbaren Bereichs werden Werte zu `inf` oder $0$. Eine kürzere Mantisse erzeugt nur leichtes Quantisierungsrauschen.

<!--en:da9caba3-->
::: {.slide title="Warum 16-Bit-Gewichte das Training ausbremsen: Rundung kleiner Updates"}

<!--en:1500c236-->
[Wenn kleine Parameterupdates durch Rundung verschwinden]{.kicker}

<!--en:f6099678-->
- **Faustregel für Hauptgewichte:** Unmittelbar oberhalb von $w = 1.0$ beträgt der Rasterabstand in `bf16` genau $2^{-7} \approx 0.0078$. Eine Erhöhung mit $|\Delta w| < 0.0039$ rundet wieder auf $1.0$, sofern die Updates nicht in `fp32` aufsummiert werden.

<!--en:1f640abd-->
::: {.slide title="Regeln für die automatische Typumwandlung bei unterschiedlichen Operanden" except="tensorflow"}

<!--en:c7c4bdbe-->
[Rechnen mit unterschiedlichen Datentypen]{.kicker}

<!--en:afcbf4d4-->
- Bei unterschiedlichen Gleitkommabreiten gilt das breitere Format (`fp16 + fp32 -> fp32`, `fp16 + bf16 -> fp32`). Schon ein einziger versehentlich verwendeter `fp32`-Puffer stellt einen 16-Bit-Durchlauf auf fp32 um:

<!--en:5fcfa4be-->
::: {.slide title="Explizite Typprüfung in TensorFlow" only="tensorflow"}

<!--en:c7c4bdbe-2-->
[Rechnen mit unterschiedlichen Datentypen]{.kicker}

<!--en:a4aab688-->
- TensorFlow lehnt implizite Umwandlungen zwischen Tensortypen ab (`InvalidArgumentError`). Die Umwandlung muss explizit mit `tf.cast` geschehen:

<!--en:12bf018f-->
::: {.slide title="Umwandlung auf 16 Bit halbiert den Speicherbedarf der Gewichte bei der Inferenz"}

<!--en:3189d5d0-->
[Typumwandlung für das gesamte Modell]{.kicker}

<!--en:6d9a7fc1-->
- Die Umwandlung aller Parameter im Zustandsbaum auf 16 Bit halbiert den Speicherbedarf der Gewichte (`814,120 B -> 407,060 B`).

<!--en:2925cdc0-->
::: {.slide title="Alle Parameter auf 16 Bit umwandeln"}

<!--en:3189d5d0-2-->
[Typumwandlung für das gesamte Modell]{.kicker}

<!--en:a0480104-->
::: {.slide title="Gemischte Genauigkeit mit fp32-Hauptgewichten und 16-Bit-Arithmetik"}

<!--en:5effde73-->
[Die AMP-Trainingsschleife]{.kicker}

<!--en:ebbff779-->
**Training mit gemischter Genauigkeit** (Micikevicius et al., 2018) trennt die Genauigkeit beim Rechnen von der Genauigkeit beim Speichern:

<!--en:3d0aa97c-->
1. **Zum Rechnen umwandeln:** Die `fp32`-Hauptgewichte für Matrizenmultiplikationen im Vorwärts- und Rückwärtsdurchlauf in `bf16` (oder `fp16`) umwandeln. Auch die Aktivierungen in diesem Format speichern.

<!--en:825ea479-->
2. **Reduktionen in `fp32`:** Verlustfunktionen, Softmax-Funktionen und Schichtnormalisierungen in `fp32` berechnen.

<!--en:aa571651-->
3. **Updates in `fp32`:** Gradienten (und bei Adam die `fp32`-Momente $\mathbf{m}, \mathbf{v}$) direkt auf die `fp32`-Hauptgewichte anwenden.

<!--en:4a26b7c3-->
::: {.slide title="Gemischte Genauigkeit in vier Codezeilen aktivieren"}

<!--en:c5230a82-->
[Automatische Umwandlung und Genauigkeitseinstellungen der Frameworks]{.kicker}

<!--en:7bfb304a-->
Eine Einstellung für die Genauigkeit oder ein Kontextmanager belässt die Gewichte in `fp32`, während die Aktivierungen mit 16 Bit ausgegeben werden:

<!--en:cc2647a3-->
::: {.slide title="Reine bf16-Gewichte bremsen das Training, gemischte Genauigkeit hält mit fp32 Schritt"}

<!--en:80ef6c1b-->
[Trainingsvergleich auf Fashion-MNIST]{.kicker}

<!--en:a56529a7-->
Das Training identischer Fashion-MNIST-MLPs über $1{,}000$ SGD-Schritte mit $\eta = 0.01$ bei drei verschiedenen Genauigkeitseinstellungen zeigt, warum `fp32`-Hauptgewichte unverzichtbar sind:

<!--en:491c1167-->
- **90% der `bf16`-Updates gehen verloren:** Bei reinen `bf16`-Gewichten bleiben etwa $90\%$ der Updates in der ersten Schicht unter dem halben lokalen `bf16`-Rasterabstand. Der Verlust liegt noch bei $\sim 1.15$ nach $1{,}000$ Schritten. Mit gemischter Genauigkeit verläuft die Verlustkurve fast deckungsgleich mit der für `fp32` ($\sim 0.85$).

<!--en:aa5d80c8-->
::: {.slide title="Warum der statische Mindestspeicherbedarf bei gemischter Genauigkeit 128 GB beträgt"}

<!--en:a8221560-->
[Speicherbedarf bei Training und Inferenz]{.kicker}

<!--en:d5d18fe7-->
- **Der statische Mindestbedarf beim Training bleibt bei `128 GB`:** Gemischte Genauigkeit erfordert `fp32`-Hauptgewichte (`32 GB`), `fp32`-Gradienten (`32 GB`) und die `fp32`-Momente $\mathbf{m}, \mathbf{v}$ von Adam (`64 GB`). Die 16-Bit-Werte für den Vorwärtsdurchlauf entstehen durch Umwandlung.

<!--en:9ee3131d-->
- **Vorteile gemischter Genauigkeit:** Sie halbiert den dynamischen **Aktivierungsspeicher im Vorwärtsdurchlauf** und verdoppelt den Durchsatz der Tensor Cores bei Matrizenmultiplikationen.

<!--en:5123d8cd-->
::: {.slide title="Unterlauf von Gradienten in fp16 messen"}

<!--en:b7a98ba1-->
[Warum fp16 Loss-Scaling benötigt]{.kicker}

<!--en:c6973b6f-->
::: {.slide title="Unterlauf von Gradienten in fp16 und dynamisches Loss-Scaling"}

<!--en:b7a98ba1-2-->
[Warum fp16 Loss-Scaling benötigt]{.kicker}

<!--en:cce47e23-->
- **Ein Drittel der Gradienten wird subnormal:** Ohne Skalierung (`scale 1`) fällt etwa ein Drittel der von null verschiedenen Gradienten der ersten Schicht unter $2^{-14}$. Etwa $\sim 0.1\text{--}0.2\%$ runden auf null.

<!--en:8fe130bd-->
- **Loss-Scaling ($S = 2^{16}$):** Nach Multiplikation des Verlusts mit $S$ vor der Backpropagation liegen alle diese Gradienten sicher im normalen `fp16`-Bereich.

<!--en:fae3259d-->
::: {.slide title="Automatisches dynamisches Loss-Scaling in der Praxis" only="pytorch,tensorflow"}

<!--en:fa9956b9-->
[GradScaler und LossScaleOptimizer]{.kicker}

<!--en:f35fd1f7-->
- **Dynamisches Loss-Scaling** beginnt mit einem großen Skalierungsfaktor $S$. Es überspringt den Optimierungsschritt und halbiert $S$, sobald ein Gradient `inf` oder `NaN` ist. Nach mehreren fehlerfreien Schritten in Folge verdoppelt es $S$:

<!--en:a10ccbab-->
::: {.slide title="Wann eignen sich bf16, fp16 und TF32?"}

<!--en:3a5b3b6f-->
[Wahl der Genauigkeit je nach Hardware]{.kicker}

<!--en:cd9709ea-->
- **Gemischte Genauigkeit mit `bfloat16` (Standard auf Ampere, Hopper und TPUs):** `fp32`-Hauptgewichte und Berechnungen in `bf16`, **kein Loss-Scaling nötig**.

<!--en:6061c698-->
- **Gemischte Genauigkeit mit `float16` und `GradScaler` (Volta- und Turing-GPUs):** Verhindert den Unterlauf von Gradienten bei 5 Exponentenbits.

<!--en:297bb0cd-->
- **Rechenmodus TensorFloat-32 (`TF32`):** Arbeitet mit `fp32`-Tensoren und rundet die Eingaben für Matrizenmultiplikationen intern auf 10 Mantissenbits.

<!--en:0492b1a7-->
::: {.slide title="Numerische Risiken: Vom Überlauf zu NaN und Log-Sum-Exp"}

<!--en:e0801be8-->
[Innerhalb des 16-Bit-Wertebereichs bleiben]{.kicker}

<!--en:cbb4411b-->
- **Log-Sum-Exp mit Verschiebung:** Das Abziehen von $m = \max_i x_i$ vor dem Exponenzieren verhindert einen Überlauf wie bei $e^{12} \approx 163{,}000 > 65{,}504$ und damit `inf` und nachfolgende `NaN`-Werte.

<!--en:cf57fcb3-->
::: {.slide title="Numerisches Risiko: Laufende Summen wachsen mit 16 Bit nicht mehr"}

<!--en:e5c2ed8c-->
[Reduktionen in fp32 akkumulieren]{.kicker}

<!--en:60c4ba08-->
- Zwischen $2$ und $4$ haben benachbarte `fp16`-Zahlen den Abstand $2^{-9}$. Daher rundet $2 + 2^{-10}$ jedes Mal wieder auf $2.0$. Summen und Normalisierungen stets in `fp32` akkumulieren.

<!--en:07f63548-->
::: {.slide title="Verständnisfragen zu Gleitkommaformaten und gemischter Genauigkeit"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:01f25877-->
Drei Fragen zur numerischen Genauigkeit von Hand lösen:

<!--en:27d0c9af-->
1. Welchen Abstand haben benachbarte darstellbare Zahlen in `bfloat16` (7 Mantissenbits) im Intervall $[512, 1024)$, und welchen Wert ergibt `bf16(1000.0) + bf16(1.0)`?

<!--en:961ae03f-->
2. Ein Gradienteneintrag hat den Betrag $g = 2 \times 10^{-6}$. Bei der Rechnung ohne Skalierung in `fp16` gelten die Grenzen `tiny` $= 6.1 \times 10^{-5}$ und kleinste positive subnormale Zahl $= 6.0 \times 10^{-8}$. Ist $g$ normal, subnormal oder null? Was ändert sich bei Skalierung mit $S = 2^{16}$?

<!--en:7ef24957-->
3. Wenn beim dynamischen Loss-Scaling ein `inf`-Gradient in Schritt $t$ auftritt, warum muss die Aktualisierung der Adam-Momente $\mathbf{m}$ und $\mathbf{v}$ entfallen? Warum darf man den `inf`-Eintrag nicht einfach auf null setzen?

<!--en:557b98b6-->
**Lösungen:**

<!--en:c4dd3a33-->
1. **Der Abstand beträgt $2^{9 - 7} = 4.0$, also gilt $1000.0 + 1.0 = 1000.0$:** Wegen $512 = 2^9$ haben benachbarte `bf16`-Werte den Abstand $4$. Eine Erhöhung um $1$ ist kleiner als ein halber Rasterschritt ($2.0$) und geht verloren.

<!--en:fc70e501-->
2. **Ohne Skalierung subnormal (noch $\sim 5$ Bit Genauigkeit), mit Skalierung normal:** $S \cdot g = 65{,}536 \times 2 \times 10^{-6} \approx 0.131$. Damit steht wieder die volle Genauigkeit der 10 Mantissenbits von `fp16` zur Verfügung.

<!--en:97f83244-->
3. **Ein Update von $\mathbf{m}, \mathbf{v}$ mit `inf` macht Adam durch `NaN`-Werte dauerhaft unbrauchbar.** Ersetzt man dagegen `inf` durch $0$, verfälscht das die gleitenden Momente der Parameter mit den größten Gradienten.

<!--en:8ca49a5e-->
::: {.slide title="Zusammenfassung: Zahlenformate und gemischte Genauigkeit"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:a2e07a05-->
- **Eigenschaften der Formate:** `fp32` ($1+8+23$) dient für Hauptgewichte, Adam-Momente und Reduktionen. `bf16` ($1+8+7$) deckt den Wertebereich von `fp32` ohne Loss-Scaling ab. `fp16` ($1+5+10$) erfordert dynamisches Loss-Scaling mit $S \cdot L$.

<!--en:bcd05c66-->
- **Gemischte Genauigkeit (AMP):** Für Matrizenmultiplikationen und Aktivierungen werden die `fp32`-Hauptgewichte auf 16 Bit umgewandelt. Damit entspricht die Konvergenz der von `fp32` bei halbem Aktivierungsspeicher.

<!--en:d0162d32-->
- **Stabile Reduktionen:** `logsumexp` mit Verschiebung und `fp32`-Akkumulatoren verhindern Überlauf und ein vorzeitiges Stehenbleiben der laufenden Summe.
