<!--en:780fdc3f-->
# Reproduzierbarkeit und Modellinspektion

<!--en:8d1c4c05-->
Angenommen, Sie ändern eine Zeile in einem Trainingsskript, und die Validierungsgenauigkeit auf Fashion-MNIST steigt um 0.3 Prozentpunkte. Um den Erfolg zu beurteilen, müssen Sie wissen, wie stark die Ergebnisse zweier Läufe des *unveränderten* Skripts von sich aus schwanken. In :numref:`subsec_seed_spread` trainieren wir eine Softmax-Regression fünfmal und ändern dabei nur den Zufallsstartwert (engl. random seed). Allein durch andere Anfangsgewichte und eine andere Mischreihenfolge der Batches liegen die endgültigen Genauigkeiten einen bis vier Prozentpunkte auseinander. Darin geht ein Gewinn von 0.3 Punkten völlig unter.

<!--en:413b2575-->
Bei der Fehlersuche nach einem Trainingsabbruch stellt sich die umgekehrte Frage. Wenn der Verlust nach Stunden des Trainings explodiert und NaN wird, möchten wir die Schritte bis zum Absturz exakt reproduzieren. Dann können wir im laufenden Netz ermitteln, welche Schicht zuerst einen nicht endlichen Wert ausgegeben hat.

<!--en:79b67299-->
Hier behandeln wir beide Fragen zusammen. Zunächst machen wir Trainingsläufe wiederholbar. Dazu kontrollieren wir Pseudozufallszahlengeneratoren über globale Startwerte, private Generatorobjekte und Datenlader mit mehreren Arbeitsprozessen. Anschließend untersuchen wir eine Besonderheit der GPU: Bei parallelen atomaren Reduktionen ist die Gleitkommaaddition nicht assoziativ. Im zweiten Teil untersuchen wir Modelle im laufenden Betrieb, ohne ihren Code zu ändern. Mit *Hooks* oder ihren Entsprechungen im jeweiligen Framework protokollieren wir die Größenordnung der Aktivierungen, finden die erste Schicht mit NaN und überwachen Gradientennormen. Beides gehört zusammen: Interne Messwerte zweier Codeversionen lassen sich nur vergleichen, wenn sich zwischen den Läufen sonst nichts ändert. Beim ersten Lesen können Sie die Unterabschnitte über Generatorobjekte und Arbeitsprozesse des Datenladers überfliegen. Kommen Sie darauf zurück, sobald Ihre Datenpipelines zufällige Datenaugmentierung einsetzen.

<!--en:b067c583-->
## Startwerte und Zufallszahlengeneratoren

<!--en:0127875f-->
Der Zufall spielt beim Training an mehr Stellen eine Rolle, als viele in der Praxis vermuten: bei der Gewichtsinitialisierung (:numref:`sec_init_param`), bei den Dropout-Masken in jedem Schritt, beim Mischen der Minibatches in jeder Epoche, bei der zufälligen Datenaugmentierung (Ausschnitte, Spiegelungen, Farbjitter) und in den parallelen Arbeitsprozessen des Datenladers. Jede Zufallszahl stammt aus einem *Pseudozufallszahlengenerator* (engl. pseudorandom number generator, PRNG). Dieser Algorithmus ist deterministisch: Sein anfänglicher *Startwert* legt die gesamte Zahlenfolge fest. Setzen wir für jeden Generator in der Pipeline einen Startwert, liegt damit jede einzelne Ziehung fest.

<!--en:2ba98bfe-->
### Globale Startwerte

<!--en:9d73e200-->
Deep-Learning-Frameworks beziehen Zufallszahlen entweder aus einem gemeinsamen globalen Generator oder verlangen, dass wir den Generatorzustand explizit als Argument übergeben. Wir zeigen das Setzen des Startwerts an einer Trainingsschleife mit fünf Schritten. Alle Gewichte und synthetischen Eingaben stammen aus derselben Quelle mit festem Startwert.

<!--en:cf3ba6fc-->
`mnp.random.seed(k)` setzt den Startwert der globalen CPU- und GPU-Generatoren des Frameworks. Verwenden wir auch `random` aus Python und `np.random` aus NumPy, müssen wir deren Startwerte separat setzen.

<!--en:53506c71-->
`torch.manual_seed(k)` setzt den Startwert der globalen CPU- und GPU-Generatoren des Frameworks. Verwenden wir auch `random` aus Python und `np.random` aus NumPy, müssen wir deren Startwerte separat setzen.

<!--en:68f19a14-->
`tf.keras.utils.set_random_seed(k)` setzt mit einem Aufruf die Startwerte von `random` aus Python und der globalen Zufallszahlengeneratoren von NumPy und TensorFlow.

<!--en:7fd5ff02-->
JAX hat keinen globalen Zufallszahlengenerator. Jede Zufallsfunktion erwartet einen expliziten *Schlüssel* (`jax.random.key(seed)`). Mit `jax.random.split` lassen sich daraus unabhängige Zufallsfolgen ableiten.

<!--en:057b4195-->
Beide Läufe mit `seed=0` stimmen *Bit für Bit* überein: Jedes Gewicht und der endgültige Verlust sind identisch. Identische Anfangsgewichte und Daten ergeben bei deterministischer Arithmetik identische Gradienten. Die Reihenfolge der Rechenoperationen untersuchen wir in :numref:`subsec_deterministic_arithmetic`. Ändern wir den Startwert auf `1`, erhalten wir eine andere Trajektorie und einen anderen Verlust.

<!--en:9c977d4e-->
Den TensorFlow-Lauf führen wir auf der CPU aus. Auf einer GPU ergibt derselbe Code mit demselben Startwert gleiche Verluste. In manchen Läufen unterscheiden sich die Gewichte jedoch in den letzten Bits.

<!--en:6090b59f-->
### Generatorobjekte

<!--en:793e7e1e-->
Ein einziger globaler PRNG koppelt alle Codeteile, die daraus Zufallszahlen ziehen. Schon eine zusätzliche Schicht oder die vorübergehende Ausgabe einer weiteren Zufallszahl bei der Fehlersuche verschiebt die gesamte nachfolgende Zahlenfolge. Unbemerkt ändern sich zugleich die Aufteilung in Trainings- und Validierungsdaten, die Gewichtsinitialisierungen und die Dropout-Masken. Ein privates *Generatorobjekt* isoliert eine Zufallsfolge durch einen eigenen Startwert. Ziehungen an anderer Stelle können sie dann nicht beeinflussen.

<!--en:e1f6826f-->
MXNet bietet keine privaten Generatorobjekte. Deshalb setzen wir den Startwert unmittelbar vor den Ziehungen erneut, deren Ergebnisse gleich bleiben sollen:

<!--en:d9fa398e-->
`torch.Generator().manual_seed(42)` erzeugt eine isolierte Pseudozufallsfolge. Ziehungen aus dem globalen Generator beeinflussen sie nicht:

<!--en:30e176b6-->
`tf.random.Generator.from_seed(42)` erzeugt eine isolierte Pseudozufallsfolge. Ziehungen aus dem globalen Generator beeinflussen sie nicht:

<!--en:1987db8e-->
Jeder JAX-Schlüssel ist bereits ein isolierter Wert ohne veränderlichen Zustand. Ziehungen mit `key_other` beeinflussen `key_split` nie:

<!--en:819be5c3-->
### Arbeitsprozesse des Datenladers

<!--en:d0fdb1b0-->
Bereiten Datenlader Minibatches in parallelen *Arbeitsprozessen* vor, kann die zufällige Datenaugmentierung auf zweierlei Weise unbemerkt fehlschlagen. Erzeugt das Betriebssystem die Prozesse mit `fork` als Kopien des Hauptprozesses, erben alle denselben Generatorzustand. Ohne neue Startwerte erzeugen sie daher identische „zufällige“ Augmentierungen. Initialisiert sich dagegen jeder Arbeitsprozess aus der Entropiequelle des Betriebssystems, stimmen keine zwei Läufe überein.

<!--en:4c5a77d1-->
Der `DataLoader` von Gluon erzeugt Arbeitsprozesse mit identischen Kopien des globalen Generatorzustands von NumPy und akzeptiert kein `worker_init_fn`. Damit die Prozesse unterschiedliche Zahlen ziehen, initialisieren wir pro Beispiel einen Generator mit `np.random.default_rng([seed, i])` innerhalb von `__getitem__`. Mit `np.random.seed` legen wir den Startwert für die Mischreihenfolge des Datenladers fest:

<!--en:40c42393-->
Der `DataLoader` initialisiert in jedem Arbeitsprozess die Generatoren von PyTorch, von `random` und den globalen NumPy-Generator anhand von `base_seed + worker_id`. Ein privates `np.random.default_rng()`-Attribut des `Dataset` wird jedoch in alle per Fork erzeugten Prozesse identisch kopiert. Deshalb setzen wir seinen Startwert über `worker_init_fn` mit `torch.utils.data.get_worker_info().seed` neu:

<!--en:ced52432-->
`tf.data` nutzt parallele Threads innerhalb eines Prozesses und akzeptiert explizite Startwerte (`Dataset.shuffle(buffer_size, seed=0)`). So entstehen von vornherein keine Kopien eines Generatorzustands durch einen Fork:

<!--en:366eae85-->
In JAX erhalten die Arbeitsprozesse explizite Schlüssel, die wir mit `jax.random.split(key, num_workers)` und `jax.random.fold_in(key, epoch)` ableiten. Nur wenn zwei Prozesse denselben Schlüssel bekommen, können sie versehentlich denselben Zustand nutzen.

<!--en:47ae25b0-->
Ohne erneute Initialisierung pro Arbeitsprozess oder Beispiel geben beide Prozesse in der ersten Zeile identische „Zufallszahlen“ aus. Mit den richtigen Startwerten ergeben sich in der zweiten Zeile unterschiedliche, reproduzierbare Ziehungen.

<!--en:433648de-->
### Zufall als Wert

<!--en:abb71f5b-->
Alle oben beschriebenen Fehler entstehen, weil Arbeitsprozesse einen versteckten, veränderlichen Generatorzustand implizit kopieren oder teilen. Funktionale Schnittstellen für Zufallszahlen vermeiden diesen versteckten Zustand, indem sie Zufall als expliziten Wert behandeln. Jede Zufallsfunktion erhält einen Schlüssel. Zwei Aufrufe mit demselben Schlüssel liefern identische Zahlen. Unabhängige Zufallsfolgen entstehen durch explizites Aufteilen der Schlüssel. Ein Schlüssel lässt sich zwar versehentlich mehrfach verwenden, doch das ist direkt im Code sichtbar. Dafür müssen wir Schlüssel an jede stochastische Funktion weiterreichen. Ein globaler Startwert ist eine implizite Kurzform für dieselbe Vereinbarung. Private Generatorobjekte erlauben es, einzelne Zufallsfolgen bei Bedarf wieder explizit zu isolieren.

<!--en:79286e64-->
So funktioniert `jax.random` in JAX.

<!--en:5bfdcc53-->
### Streuung bei unterschiedlichen Startwerten

<!--en:b09ee08c-->
Nun messen wir die eingangs erwähnte Varianz zwischen Läufen mit unterschiedlichen Startwerten. Wir trainieren eine Softmax-Regression (:numref:`sec_softmax_concise`) auf Fashion-MNIST jeweils drei Epochen lang auf einer GPU. Für jeden der fünf Startwerte (0 bis 4) initialisieren wir alle beteiligten Generatoren. Jeder Lauf liefert die endgültige Validierungsgenauigkeit und die Gewichte.

<!--en:0eba80b7-->
Alle fünf Läufe verwenden denselben Code und dieselben Hyperparameter. Dennoch liegen ihre endgültigen Genauigkeiten bei rund 81% um einen bis vier Prozentpunkte auseinander. Das bestätigt, dass ein Anstieg um 0.3 Punkte in einem einzelnen Lauf reines Rauschen ist. Für einen aussagekräftigen empirischen Vergleich geben wir stets Mittelwert und Streuung über mehrere Startwerte an.

<!--en:dfccefef-->
Um zu prüfen, ob allein der Startwert die Unterschiede verursacht hat, wiederholen wir den Lauf mit `seed=0`. Wir vergleichen sowohl die Genauigkeit als auch jedes einzelne Gewicht mit dem ersten Lauf:

<!--en:22f03a31-->
Die Validierungsgenauigkeit stimmt exakt mit der des ersten Laufs überein. Ob auch jedes einzelne Gewicht bis auf das letzte Bit übereinstimmt, hängt davon ab, ob die GPU Gleitkommazahlen in einer festen Reihenfolge summiert. Startwerte allein legen diese Reihenfolge nicht fest (:numref:`subsec_deterministic_arithmetic`).

<!--en:a3b12252-->
Auf einer GPU summiert TensorFlow bei der Reduktion für den Bias-Gradienten die Terme in nicht deterministischer Thread-Reihenfolge. Zwei GPU-Läufe mit identischem Startwert können sich daher in den niederwertigsten Bits um $\sim 10^{-6}$ unterscheiden.

<!--en:b5c89c5b-->
## Deterministische Arithmetik

<!--en:d65803ff-->
Startwerte legen fest, *welche* Zufallszahlen wir in einem Lauf ziehen. Die Gleitkommaarithmetik bestimmt, *wie* wir sie miteinander verrechnen. Da jede Gleitkommaaddition auf den nächsten darstellbaren Wert rundet, ist sie *nicht* assoziativ: Ändern wir die Reihenfolge der Summation, kann sich das Ergebnis ändern.

<!--en:8521042d-->
Drei float32-Zahlen machen den Effekt deutlich. In der Nähe von $10^8$ beträgt der Abstand zwischen benachbarten float32-Zahlen 8 (:numref:`sec_numerics`). Daher ergibt $-10^8 + 1$ gerundet wieder $-10^8$:

<!--en:4b435b31-->
Allein durch Verschieben der Klammern ändert sich das Ergebnis von `1.0` auf `0.0`. Bei langen Reduktionen sind die Unterschiede kleiner, treten aber häufig auf: Summieren wir eine Million Gleitkommazahlen einmal vorwärts und einmal rückwärts, unterscheiden sich die Ergebnisse regelmäßig in den niederwertigsten Bits.

<!--en:53fd63a2-->
Auf einer CPU mit fester Thread-Anzahl bleibt die Reihenfolge der Reduktion meist gleich. Läufe mit festem Startwert lassen sich dann exakt wiederholen. Auf einer GPU arbeiten Tausende Threads parallel. Mehrere schnelle Kernel (etwa für Einbettungsgradienten, Scatter-Add, bestimmte Rückwärtsdurchläufe von Faltungen und Reduktionen) summieren Teilergebnisse durch *atomare Additionen*. Deren Reihenfolge hängt davon ab, wann die Threads fertig werden. Selbst auf derselben GPU können sich zwei Läufe mit identischem Startwert daher in den niederwertigsten Bits unterscheiden. Chaotische Gradientenschritte können diese Unterschiede mit der Zeit so verstärken, dass sichtbar verschiedene Verlustkurven entstehen. Jedes Framework bietet einen deterministischen Modus oder einen alternativen Kernel mit fester Summationsreihenfolge, meist auf Kosten der Geschwindigkeit. Wir beobachten den Effekt direkt bei einer *Scatter-Add*-Operation auf der GPU: Zweimal hintereinander summieren wir eine Million Zufallszahlen in zehn Gruppen.

<!--en:b25f296d-->
`MXNET_ENFORCE_DETERMINISM=1` schränkt die Wahl der cuDNN-Algorithmen für Faltungen ein. `npx.index_add` bleibt auf der GPU jedoch nicht deterministisch. Formulieren wir Scatter-Add als Matrizenmultiplikation mit One-Hot-Kodierung, verwendet cuBLAS eine Reduktion mit fester Reihenfolge:

<!--en:2fa8700c-->
`torch.use_deterministic_algorithms(True)` stellt Operationen wie `index_add_` auf deterministische Kernel um oder löst einen `RuntimeError` aus, wenn kein solcher Kernel existiert:

<!--en:382a8f91-->
`tf.config.experimental.enable_op_determinism()` erzwingt für den restlichen Prozess deterministische GPU-Kernel (etwa `tf.math.unsorted_segment_sum`):

<!--en:3ca14d76-->
Übergeben wir `compiler_options={'xla_gpu_deterministic_ops': True}` an `jax.jit` oder setzen wir `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, erzwingt XLA deterministische Reduktionen auf der GPU:

<!--en:bd0d194f-->
Ohne deterministischen Modus liefern zwei aufeinanderfolgende Scatter-Add-Operationen auf exakt denselben GPU-Daten leicht unterschiedliche Ergebnisse. Bei Summen von einigen Hundert beträgt der Unterschied einige Tausendstel. Mit einem deterministischen Kernel stimmen beide Aufrufe Bit für Bit überein.

<!--en:eb43d523-->
Bei aktiviertem Schalter benötigen Zufallsoperationen einen Startwert. Denn auch eine Operation, die ihren Startwert selbst vom Betriebssystem bezieht, ist nicht deterministisch. Das lässt sich zeigen, indem wir den globalen Startwert löschen:

<!--en:b227241b-->
Bitgenaue Reproduzierbarkeit gilt nur auf ein und derselben Maschine mit derselben Software- und Hardwarekonfiguration. Kein Framework garantiert identische Bits bei einem Versionswechsel der Bibliothek, auf verschiedenen GPU-Architekturen oder beim Wechsel zwischen CPU und GPU. Bei der *Fehlersuche* ist bitgenauer Determinismus besonders wertvoll: Durch wiederholtes Halbieren des Suchbereichs finden wir genau den Schritt, ab dem zwei Läufe auseinanderlaufen. Wissenschaftliche Reproduzierbarkeit ist zugleich weniger streng und anspruchsvoller. Eine echte Verbesserung („Diese Architekturänderung erhöht die Genauigkeit“) muss sich bei mehreren Startwerten zuverlässig zeigen. Wir geben Mittelwert und Streuung wie in :numref:`subsec_seed_spread` an (rund 81% mit einer Spannweite von 1 bis 4 Punkten). Wie bei 16-Bit-Formaten (:numref:`sec_numerics`) gilt: Kehrt ein anderer Startwert oder ein Rundungsunterschied im letzten Bit eine Aussage um, brauchen wir stärkere Belege.

<!--en:7a4c3dc7-->
## Modellinspektion mit Hooks

<!--en:1dab14a0-->
Lässt sich ein Fehler zuverlässig reproduzieren, steht fest, dass er existiert. Offen bleibt, *wo* im Netz er entsteht. Ausgabeanweisungen in die `forward`-Methode jeder Schicht einzufügen ist umständlich. Bei Modellen aus fremden Bibliotheken ist das kaum praktikabel, wenn wir keine eigene Version des Quellcodes pflegen möchten. Ein *Hook* löst das Problem: Wir registrieren von außerhalb des Modells eine Rückruffunktion. Das Framework ruft sie automatisch vor oder nach der Ausführung einer Schicht auf.

<!--en:94e3a019-->
Mit `block.register_forward_hook(f)` läuft `f(block, inputs, output)` innerhalb von `__call__` nach `forward`. Eine mit `register_forward_pre_hook` registrierte Funktion läuft davor.

<!--en:85a52851-->
Mit `module.register_forward_hook(f)` läuft `f(module, inputs, output)` innerhalb von `__call__` nach `forward`. Gibt die Funktion einen neuen Tensor zurück, ersetzt dieser die Ausgabe.

<!--en:0059043d-->
Keras bietet keine externe Schnittstelle für Hooks. Wir erstellen entweder ein zweites `tf.keras.Model`, das Zwischentensoren eines funktionalen Graphen symbolisch zugänglich macht, oder protokollieren Tensoren in einer eigenen `call`-Methode.

<!--en:93fd9ace-->
`nnx.capture(net, nnx.Intermediate, method_outputs=nnx.Intermediate)(X)` erfasst in einem einzigen funktionalen Aufruf den Rückgabewert jedes Untermoduls, ohne den Modellcode zu ändern.

<!--en:110a38bd-->
![Die Pipeline des Wrappers `__call__`: Die Eingabe durchläuft zuerst die vorgeschalteten Hooks, dann die Vorwärtsberechnung und schließlich die Hooks vor der Ausgabe. Die beiden Hook-Stufen sind orange gestrichelt, die Vorwärtsberechnung ist blau durchgezogen. Ein seitlicher Pfeil führt von der letzten Hook-Stufe zu einem Beobachter, der die Ausgabe erfassen, prüfen oder ändern kann.](../img/bg-hooks.svg)

<!--en:eadf3c9b-->
Für unsere Beispiele setzen wir acht Residualblöcke der Breite 64 zwischen eine Schicht mit 20 Eingaben und einen Ausgabekopf mit 10 Ausgaben (:numref:`sec_model_construction`). Jeder Block berechnet $\mathbf{h} = \mathbf{x} + \textrm{body}(\mathbf{x})$ mit Xavier-initialisierten Gewichten (:numref:`subsec_xavier`) und Bias-Termen von null. Wir verzichten hier bewusst auf einen festen Startwert, damit jeder Lauf neue zufällige Gewichte zieht. Deshalb geben wir die folgenden Messwerte als typische Bereiche an.

<!--en:6c29c95f-->
Nach den Läufen mit festem Startwert befindet sich der globale MXNet-Generator in einem festen Zustand. Damit stünden auch die Gewichte dieses Modells fest. Die Zelle initialisiert den Generator daher zunächst neu mit Entropie aus dem Betriebssystem.

<!--en:9337b795-->
Nach den Läufen mit festem Startwert befindet sich der globale PyTorch-Generator in einem festen Zustand. Damit stünden auch die Gewichte dieses Modells fest. Die Zelle initialisiert den Generator daher zunächst mit `torch.seed()` neu und bezieht dafür Entropie aus dem Betriebssystem.

<!--en:ea90c59f-->
### Aktivierungsstatistik

<!--en:17ec0ccf-->
In :numref:`sec_init_param` haben wir die Blöcke in einer selbst geschriebenen Schleife durchlaufen, um die Standardabweichung der Aktivierungen in Abhängigkeit von der Tiefe zu messen. Mit Hooks genügen wenige Zeilen, um dasselbe Profil am unveränderten Modell zu erheben. Wir verwenden einen Minibatch mit 256 standardnormalverteilten Eingaben:

<!--en:1972d24e-->
Die Standardabweichung der Aktivierungen steigt mit der Tiefe stetig an: von etwa 0.7 direkt nach der Eingabeschicht auf etwa 3–4.5 nach dem achten Block. Das stimmt gut mit der Theorie überein. Bei der Xavier-Initialisierung hat die $20 \to 64$-Eingabeschicht eine Gewichtsvarianz von $2/84$. Daraus ergibt sich eine Ausgabevarianz von $20 \cdot 2/84 \approx 0.48$ (Standardabweichung 0.69). Jeder Residualblock addiert zur Eingabe die damit annähernd unkorrelierte Ausgabe seines Residualzweigs. Deren Varianz beträgt die Hälfte der Eingabevarianz: Die erste lineare Schicht erhält die Varianz, die ReLU halbiert den Mittelwert der Quadrate, und die zweite lineare Schicht erhält diesen Wert. Damit wächst die Gesamtvarianz pro Block um den Faktor $1.5$ und die Standardabweichung um den Faktor $\sqrt{1.5} \approx 1.22$. Über acht Blöcke ergibt sich $0.69 \times 1.5^4 \approx 3.5$, genau in der Mitte unserer Messwerte. Das ist dieselbe Analyse des Varianzwachstums wie in :numref:`sec_init_param`, hier mit dem Faktor $1.5$ anstelle von $2$ bei He-Initialisierung. Sie zeigt, warum tiefe Residualzweige davon profitieren, ihre Ausgabeprojektionen herunterzuskalieren.

<!--en:bed2233b-->
Zwei Regeln verhindern Speicherlecks bei Hooks. Trennen Sie jeden im Hook gespeicherten Tensor mit `.detach()` vom Rechengraphen. Andernfalls bleibt der gesamte Autograd-Graph über die Schritte hinweg im Speicher. Entfernen Sie nach der Untersuchung den Hook stets über sein Handle (`h.detach()`).

<!--en:9f53bf92-->
Zwei Regeln verhindern Speicherlecks bei Hooks. Trennen Sie jeden im Hook gespeicherten Tensor mit `.detach()` vom Rechengraphen. Andernfalls bleibt der gesamte Autograd-Graph über die Schritte hinweg im Speicher. Entfernen Sie nach der Untersuchung den Hook stets über sein Handle (`h.remove()`).

<!--en:839d6c17-->
`probe(X)` hinterlegt keinen dauerhaften Zustand im Modell. Daher müssen wir anschließend nichts abtrennen oder entfernen.

<!--en:d7b15839-->
`nnx.capture` hinterlegt keinen dauerhaften Zustand im Modell. Daher müssen wir anschließend nichts abtrennen oder entfernen.

<!--en:9070aa22-->
### Suche nach dem ersten NaN

<!--en:2686433e-->
Wenn der Trainingsverlust bei Schritt 40 000 plötzlich NaN wird, lautet die erste Frage: *Welche Schicht hat zuerst einen nicht endlichen Wert ausgegeben?* Dazu registrieren wir an jeder Blattschicht einen Hook, der die Ausgabe auf endliche Werte prüft. Da die Hooks in der Reihenfolge der Vorwärtsberechnung laufen, genügt ein einziger Vorwärtsdurchlauf, um die Schicht zu finden. Für einen Test setzen wir absichtlich ein Gewicht des dritten Residualblocks (Index 3) auf NaN:

<!--en:bfe1cfc8-->
Der Hook identifiziert sofort die fehlerhafte Schicht in Block 3, ohne dass wir eine einzige Ausgabeanweisung in die Schichtklassen einfügen müssen. Ein absichtlich gesetztes NaN macht die Demonstration besonders einfach. Bei einem echten Trainingsabbruch zeigt uns die Suche das erste *Symptom*. Die Ursache kann mehrere Schichten früher liegen, etwa wenn Aktivierungen unkontrolliert wachsen, bis ein Überlauf auftritt.

<!--en:d4ae7e59-->
### Untersuchung der Gradienten

<!--en:7ae18d83-->
Den Rückwärtsdurchlauf können wir ebenso genau überwachen. Die Aktivierungen sind im Vorwärtsdurchlauf über unsere acht Residualblöcke gewachsen. Wachsen die Gradienten in umgekehrter Richtung? Als Verlust verwenden wir den Mittelwert der quadrierten Ausgaben für 256 standardnormalverteilte Eingaben. Wir protokollieren die Gradientennorm an der Ausgabe jedes Residualblocks sowie die Normen der Gewichts- und Bias-Gradienten der ersten linearen Schicht in Block 3.

<!--en:e54545bf-->
Um die Gradienten der Zwischenausgaben ohne Rückwärts-Hooks zu untersuchen, addieren wir zur Ausgabe jedes Blocks einen Nulltensor `Z` und differenzieren nach `Z`:

<!--en:d9dc6213-->
`module.register_full_backward_hook(f)` ruft `f(module, grad_input, grad_output)` während `backward()` auf. Die Hooks laufen in umgekehrter topologischer Reihenfolge, vom letzten bis zum ersten Block:

<!--en:762f5296-->
`tape.gradient(loss, [acts[1:], net.trainable_variables])` berechnet in einem Aufruf die Gradienten sowohl nach den Zwischenausgaben des funktionalen Modells als auch nach den trainierbaren Variablen:

<!--en:32df3dcf-->
Vom achten Block zurück zum ersten wächst die Norm des Ausgabegradienten je Block ungefähr um den Faktor $1.2\times$ bis $1.4\times$, insgesamt auf das Fünf- bis Sechsfache. Das entspricht dem Wachstum um den Faktor $1.22\times$ im Vorwärtsdurchlauf und hat denselben Grund: Im Rückwärtsdurchlauf addiert jeder Residualblock den Gradienten des Residualzweigs zum Gradienten der Skip-Verbindung. Die Gradientennorm eines Parameters sagt für sich genommen wenig aus. Vergleichen wir die Normen jedoch über die Tiefe oder verfolgen wir sie über mehrere Trainingsschritte, erkennen wir sofort, wenn Gradienten verschwinden oder explodieren (siehe Übungen).

<!--en:3580cd70-->
Beachten Sie die Reichweite dieser Diagnosewerkzeuge: Sie können auf verschiedenen Maschinen keine bitgenaue Übereinstimmung erzwingen. Auch kann ein einzelner Lauf nicht zeigen, ob ein gemessener Gewinn echt ist. Dazu vergleichen wir Verteilungen über mehrere Startwerte. Zusammen mit Checkpoints, an denen sich ein Lauf fortsetzen lässt (:numref:`sec_read_write`), haben Sie nun alle Werkzeuge, um einen fehlgeschlagenen Lauf bis zum Absturz zu wiederholen. Mit einem Diagnose-Hook können Sie dann die fehlerhafte Schicht genau bestimmen.

<!--en:f37062da-->
Das Sprachmodell mit 8 Milliarden Parametern vom Kapitelanfang benötigt in fp32 weiterhin 32 GB. Wie Frameworks damit umgehen, können wir jetzt in jedem Schritt nachvollziehen. Wir bauen Architekturen aus Konfigurationen auf, prüfen Parameter und Puffer, initialisieren tiefe Schichtfolgen, definieren eigene Schichten und Gradienten, trainieren mit gemischter Genauigkeit, speichern ausfallsichere Checkpoints, verwalten GPU-Speicher und untersuchen laufende Modelle. All das geschieht auf demselben Modulbaum. Damit sind wir gerüstet für Architekturen, deren Aufbau die Struktur realer Daten berücksichtigt. Den Anfang machen konvolutionäre neuronale Netze in :numref:`chap_cnn`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:36926a5c-->
Jede zufällige Entscheidung beim Training (Initialisierung, Mischen, Dropout und Datenaugmentierung) geht auf Pseudozufallszahlengeneratoren zurück, die wir über Startwerte steuern. Private Generatorobjekte oder explizite Schlüssel verhindern, dass unabhängiger Code eine Zufallsfolge verändert. Die Gleitkommaaddition ist nicht assoziativ, und bei parallelen atomaren Operationen auf der GPU ist die Reihenfolge der Summation über die Threads beliebig. Bitgenaue Wiederholbarkeit erfordert daher auch deterministische Kernel und gilt nur auf derselben Maschine mit derselben Softwarekonfiguration. Bitgenauer Determinismus hilft bei der Fehlersuche. Wissenschaftliche Schlussfolgerungen hingegen müssen über mehrere Startwerte Bestand haben. Mit Vorwärts- und Rückwärts-Hooks oder ihren Entsprechungen im jeweiligen Framework können wir schließlich am unveränderten Modell die Größenordnung der Aktivierungen und die Gradientennormen untersuchen sowie die Stelle finden, an der zuerst ein NaN auftritt.

<!--en:2294243a-->
## Übungen

<!--en:3d307450-->
1. [code] **Varianzanteile im Residualblock.** Es seien $\mathbf{x}_k$, $\mathbf{b}_k$ und $\mathbf{h}_k$ die Eingabe, die Ausgabe des Residualzweigs und die Ausgabe von Block $k$ von `net`. Bestimmen Sie jede Varianz über alle Einträge. Erfassen Sie für einen Batch in jedem `ResidualBlock` die Varianz von $\mathbf{x}_k$ mit `register_forward_pre_hook`. Der vorgeschaltete Hook erhält die Argumente als Tupel, das Sie entpacken müssen. Erfassen Sie die Varianz von $\mathbf{h}_k$ mit `register_forward_hook` und die von $\mathbf{b}_k$ mit einem Vorwärts-Hook an seinem `body`. Die Standardabweichungen in der Tabelle in :numref:`subsec_activation_stats` wachsen, weil jeder Block die Ausgabe seines Residualzweigs zur Eingabe addiert. Messen Sie für jeden Block, wie genau die Gleichung

   $$\textrm{Var}[\mathbf{h}_k] = \textrm{Var}[\mathbf{x}_k] + \textrm{Var}[\mathbf{b}_k]$$

   gilt, und nennen Sie den Term, der in ihr fehlt.

<!--en:25edcc03-->
2. [code] **Startwerte beim Laden von Daten.** ● Erweitern Sie `train_once` so, dass die Funktion ihre Daten über Gluons `DataLoader` mit `shuffle=True` und `num_workers=4` lädt. Behalten Sie den einmaligen Aufruf von `mnp.random.seed(seed)` am Anfang bei. MXNet bietet keine Generatorobjekte. Daher verwendet der Datensatz NumPy: Er addiert in `__getitem__` zu jedem Beispiel Rauschen aus einem `np.random.default_rng()`-Objekt, das im Konstruktor des Datensatzes erzeugt wurde.
    1. Sagen Sie vorher, ob zwei Läufe mit demselben Startwert in der Mischreihenfolge und im Rauschen übereinstimmen. Ziehen die vier Arbeitsprozesse dasselbe Rauschen? Begründen Sie jede Vorhersage anhand der jeweiligen Zufallsquelle und überprüfen Sie alle drei.
    1. Leiten Sie wie in diesem Abschnitt den Startwert für das Rauschen aus dem Index des Beispiels ab. Legen Sie die Mischreihenfolge fest und prüfen Sie, ob zwei Läufe nun übereinstimmen. Jedes Beispiel erhält dann in jeder Epoche dasselbe Rauschen, was dem Zweck der Augmentierung zuwiderläuft. Wählen Sie die Startwerte so, dass das Rauschen von Epoche zu Epoche variiert und von Lauf zu Lauf reproduzierbar bleibt.

<!--en:e3708223-->
3. [code] **Divergenz beim Training erkennen.** ● Erweitern Sie die Prüfung auf endliche Werte aus :numref:`subsec_nan_finder` so, dass sie jeden Trainingsschritt überwacht. Erstellen Sie `net` neu und trainieren Sie das Modell mit SGD und Lernrate 1. Minimieren Sie den mittleren quadratischen Fehler zwischen der Ausgabe und standardnormalverteilten Zielwerten. Verwenden Sie einen festen Batch mit 256 standardnormalverteilten Eingaben. Geben Sie den ersten Schritt an, in dem eine Schicht einen nicht endlichen Wert ausgibt, und nennen Sie die erste solche Schicht in der Reihenfolge der Netztiefe. Prüfen Sie auch den Verlust in jedem Schritt. Geben Sie an, ob er bereits einen nicht endlichen Wert annimmt, während noch alle Schichtausgaben endlich sind.

<!--en:c6c3cc80-->
4. [code] **Gradientennormen.** ● Erstellen Sie `net` neu. Multiplizieren Sie die Gewichtsmatrix jeder `Dense`-Schicht in den Residualblöcken mit 3. Verwenden Sie den Mittelwert der quadrierten Ausgaben für einen festen Batch als Verlust.
    1. Berechnen Sie wie in diesem Abschnitt für jeden Block die Norm seines Ausgabegradienten. Bestimmen Sie die Norm seines Parametergradienten anhand von `net.collect_params()`. Stellen Sie beide Arten von Normen grafisch in Abhängigkeit von der Tiefe dar.
    1. Drücken Sie den Gradienten der Gewichtsmatrix einer linearen Schicht durch deren Eingabe und ihren Ausgabegradienten aus. Erklären Sie damit, warum sich die beiden Tiefenprofile unterscheiden. Welche Normen sollten Sie überwachen, um ein explodierendes Signal im Rückwärtsdurchlauf zu erkennen?
   :exlabel:`ex_reproducibility-inspection-gradient-norms`

<!--en:495717cf-->
1. [code] **Varianzanteile im Residualblock.** Es seien $\mathbf{x}_k$, $\mathbf{b}_k$ und $\mathbf{h}_k$ die Eingabe, die Ausgabe des Residualzweigs und die Ausgabe von Block $k$. Bestimmen Sie jede Varianz über alle Einträge. Erfassen Sie für einen Batch in jedem `ResidualBlock` die Varianz von $\mathbf{x}_k$ mit `register_forward_pre_hook`. Der vorgeschaltete Hook erhält die Argumente als Tupel, das Sie entpacken müssen. Erfassen Sie die Varianz von $\mathbf{h}_k$ mit einem Vorwärts-Hook am Block und die von $\mathbf{b}_k$ mit einem Vorwärts-Hook an seinem `body`. Die Standardabweichungen in der Tabelle in :numref:`subsec_activation_stats` wachsen, weil jeder Block die Ausgabe seines Residualzweigs zur Eingabe addiert. Messen Sie für jeden Block, wie genau die Gleichung

   $$\textrm{Var}[\mathbf{h}_k] = \textrm{Var}[\mathbf{x}_k] + \textrm{Var}[\mathbf{b}_k]$$

   gilt, und nennen Sie den Term, der in ihr fehlt.

<!--en:3bf3b774-->
2. [code] **Startwerte beim Laden von Daten.** ● Erweitern Sie `train_once` so, dass die Funktion ihre Daten über einen `DataLoader` mit `shuffle=True` und `num_workers=4` lädt. Behalten Sie den einmaligen Aufruf von `torch.manual_seed(seed)` am Anfang bei. Der Datensatz addiert in `__getitem__` zu jedem Beispiel Rauschen aus einem `np.random.default_rng()`-Objekt, das im Konstruktor des Datensatzes erzeugt wurde. Führen Sie den Code als eigenständiges Skript aus. Arbeitsprozesse, die mit `spawn` starten (der Voreinstellung unter macOS und Windows), können eine in einem Notebook definierte Datensatzklasse nicht importieren.
    1. Sagen Sie vorher, ob zwei Läufe mit demselben Startwert in der Mischreihenfolge und im Rauschen übereinstimmen. Ziehen die vier Arbeitsprozesse dasselbe Rauschen? Begründen Sie jede Vorhersage anhand der jeweiligen Zufallsquelle und überprüfen Sie alle drei.
    1. Ergänzen Sie den Datenlader um `seed_worker` aus diesem Abschnitt und einen `generator` mit festem Startwert. Prüfen Sie, ob zwei Läufe nun übereinstimmen und die Arbeitsprozesse unterschiedliches Rauschen ziehen. Erhält ein Beispiel in der zweiten Epoche dasselbe Rauschen wie in der ersten? Begründen Sie Ihre Antwort damit, wann der Datenlader seinen Basisstartwert zieht und die Initialisierungsfunktion aufruft.

    *Nach der PyTorch-Dokumentation zum [Laden von Daten mit mehreren Prozessen](https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading).*

<!--en:e3708223-2-->
3. [code] **Divergenz beim Training erkennen.** ● Erweitern Sie die Prüfung auf endliche Werte aus :numref:`subsec_nan_finder` so, dass sie jeden Trainingsschritt überwacht. Erstellen Sie `net` neu und trainieren Sie das Modell mit SGD und Lernrate 1. Minimieren Sie den mittleren quadratischen Fehler zwischen der Ausgabe und standardnormalverteilten Zielwerten. Verwenden Sie einen festen Batch mit 256 standardnormalverteilten Eingaben. Geben Sie den ersten Schritt an, in dem eine Schicht einen nicht endlichen Wert ausgibt, und nennen Sie die erste solche Schicht in der Reihenfolge der Netztiefe. Prüfen Sie auch den Verlust in jedem Schritt. Geben Sie an, ob er bereits einen nicht endlichen Wert annimmt, während noch alle Schichtausgaben endlich sind.

<!--en:3edb5371-->
4. [code] **Gradientennormen.** ● Erstellen Sie `net` neu. Multiplizieren Sie die Gewichtsmatrix jeder linearen Schicht in den Residualblöcken mit 3. Verwenden Sie den Mittelwert der quadrierten Ausgaben für einen festen Batch als Verlust.
    1. Erfassen Sie wie in diesem Abschnitt für jeden Block die Norm seines Ausgabegradienten. Berechnen Sie nach dem Rückwärtsdurchlauf die Norm seines Parametergradienten. Stellen Sie beide Arten von Normen grafisch in Abhängigkeit von der Tiefe dar.
    1. Drücken Sie den Gradienten der Gewichtsmatrix einer linearen Schicht durch deren Eingabe und ihren Ausgabegradienten aus. Erklären Sie damit, warum sich die beiden Tiefenprofile unterscheiden. Welche Normen sollten Sie überwachen, um ein explodierendes Signal im Rückwärtsdurchlauf zu erkennen?
   :exlabel:`ex_reproducibility-inspection-gradient-norms`

<!--en:1ef20436-->
1. [code] **Varianzanteile im Residualblock.** Überschreiben Sie `call` in `ResidualBlock` so, dass die Methode drei Varianzen speichert: die der Eingabe, die der Ausgabe des Residualzweigs und die der Blockausgabe. Führen Sie ein neu erstelltes `Checked`-Modell auf einem Batch aus. Es seien $\mathbf{x}_k$, $\mathbf{b}_k$ und $\mathbf{h}_k$ diese drei Tensoren für Block $k$. Bestimmen Sie jede Varianz über alle Einträge. Die Standardabweichungen in der Tabelle in :numref:`subsec_activation_stats` wachsen, weil jeder Block die Ausgabe seines Residualzweigs zur Eingabe addiert. Messen Sie für jeden Block, wie genau die Gleichung

   $$\textrm{Var}[\mathbf{h}_k] = \textrm{Var}[\mathbf{x}_k] + \textrm{Var}[\mathbf{b}_k]$$

   gilt, und nennen Sie den Term, der in ihr fehlt.

<!--en:46d521a5-->
2. [code] **Startwerte beim Laden von Daten.** ● Arbeiten Sie in einem neuen Prozess. In diesem Abschnitt wurde `enable_op_determinism` bereits aufgerufen, und das lässt sich nicht rückgängig machen. Bauen Sie eine `tf.data`-Pipeline für `Dataset.range(64)` auf. Mischen Sie die Daten mit `shuffle(64, seed=0)` und addieren Sie in einem `map`-Aufruf mit `num_parallel_calls=4` zu jedem Element Rauschen. Beginnen Sie jeden Lauf mit `tf.keras.utils.set_random_seed(0)`.
    1. Ziehen Sie das Rauschen über `tf.numpy_function` aus dem globalen Generator von NumPy. Sagen Sie vorher, ob bei der Voreinstellung `deterministic=True` zwei Läufe zu demselben Element dasselbe Rauschen addieren. Begründen Sie Ihre Vorhersage anhand der Zufallsquelle. Überprüfen Sie sie in fünf Läufen.
    1. Wiederholen Sie die fünf Läufe zweimal: einmal mit einem Aufruf von `tf.config.experimental.enable_op_determinism()` beim Programmstart und einmal ohne diesen Aufruf. Ziehen Sie im zweiten Fall das Rauschen mit `tf.random.stateless_normal` und leiten Sie den Startwert aus dem Wert des Elements ab. Erklären Sie jedes Ergebnis.

<!--en:8f0cfda3-->
3. [code] **Divergenz beim Training erkennen.** ● Erweitern Sie die Prüfung auf endliche Werte aus :numref:`subsec_nan_finder` so, dass sie jeden Trainingsschritt überwacht. Erstellen Sie das `Checked`-Modell neu und trainieren Sie das Modell mit SGD und Lernrate 1. Minimieren Sie den mittleren quadratischen Fehler zwischen der Ausgabe und standardnormalverteilten Zielwerten. Verwenden Sie einen festen Batch mit 256 standardnormalverteilten Eingaben. Geben Sie den ersten Schritt an, in dem eine Schicht einen nicht endlichen Wert ausgibt, und nennen Sie die erste solche Schicht in der Reihenfolge der Netztiefe. Prüfen Sie auch den Verlust in jedem Schritt. Geben Sie an, ob er bereits einen nicht endlichen Wert annimmt, während noch alle Schichtausgaben endlich sind.

<!--en:8dcf6dd6-->
4. [code] **Gradientennormen.** ● Erstellen Sie `net` neu. Multiplizieren Sie die Gewichtsmatrix jeder `Dense`-Schicht in den Residualblöcken mit 3. Verwenden Sie den Mittelwert der quadrierten Ausgaben für einen festen Batch als Verlust.
    1. Berechnen Sie wie in diesem Abschnitt für jeden Block die Norm seines Ausgabegradienten. Bestimmen Sie die Norm seines Parametergradienten aus den Gradienten der trainierbaren Variablen. Stellen Sie beide Arten von Normen grafisch in Abhängigkeit von der Tiefe dar.
    1. Drücken Sie den Gradienten der Gewichtsmatrix einer linearen Schicht durch deren Eingabe und ihren Ausgabegradienten aus. Erklären Sie damit, warum sich die beiden Tiefenprofile unterscheiden. Welche Normen sollten Sie überwachen, um ein explodierendes Signal im Rückwärtsdurchlauf zu erkennen?
   :exlabel:`ex_reproducibility-inspection-gradient-norms`

<!--en:5da3ef2e-->
1. [code] **Varianzanteile im Residualblock.** Es seien $\mathbf{x}_k$, $\mathbf{b}_k$ und $\mathbf{h}_k$ die Eingabe, die Ausgabe des Residualzweigs und die Ausgabe von Block $k$. Bestimmen Sie jede Varianz über alle Einträge. Erfassen Sie die Ausgaben aller Schichten und Residualzweige durch einen einzigen Aufruf von `nnx.capture` mit `method_outputs=nnx.Intermediate`. Die Eingabe von Block $k$ ist die Ausgabe der vorhergehenden Schicht. Die Standardabweichungen in der Tabelle in :numref:`subsec_activation_stats` wachsen, weil jeder Block die Ausgabe seines Residualzweigs zur Eingabe addiert. Messen Sie für jeden Block, wie genau die Gleichung

   $$\textrm{Var}[\mathbf{h}_k] = \textrm{Var}[\mathbf{x}_k] + \textrm{Var}[\mathbf{b}_k]$$

   gilt, und nennen Sie den Term, der in ihr fehlt.

<!--en:58d803dc-->
2. [code] **Startwerte beim Laden von Daten.** ● Simulieren Sie in reinem Python eine Eingabepipeline mit vier Arbeitsprozessen. Teilen Sie einen Datensatz mit 64 Beispielen und je 20 Merkmalen in vier Teile auf, einen pro Prozess. Jeder Arbeitsprozess soll mit `jax.random.normal` gaußsches Rauschen zu seinen Beispielen addieren.
    1. Leiten Sie die Schlüssel für das Rauschen aus einem Basisschlüssel ab. Teilen Sie ihn dazu mit `jax.random.split(key, 4)` in je einen Schlüssel pro Arbeitsprozess auf und beziehen Sie mit `jax.random.fold_in` die Epochennummer in jeden Schlüssel ein. Prüfen Sie, ob zwei Läufe mit demselben Basisschlüssel identisches Rauschen erzeugen. Reduzieren Sie die Anzahl der Arbeitsprozesse auf zwei und prüfen Sie, ob jedes Beispiel weiterhin dasselbe Rauschen erhält.
    1. Leiten Sie nun den Schlüssel jedes Beispiels anders ab: Beziehen Sie zuerst die Epochennummer und dann den Index des Beispiels in den Basisschlüssel ein. Wiederholen Sie beide Prüfungen. Erklären Sie, bei welcher Ableitung das Rauschen eines Beispiels unabhängig davon ist, wie die Beispiele auf die Arbeitsprozesse verteilt sind.

<!--en:e3708223-3-->
3. [code] **Divergenz beim Training erkennen.** ● Erweitern Sie die Prüfung auf endliche Werte aus :numref:`subsec_nan_finder` so, dass sie jeden Trainingsschritt überwacht. Erstellen Sie `net` neu und trainieren Sie das Modell mit SGD und Lernrate 1. Minimieren Sie den mittleren quadratischen Fehler zwischen der Ausgabe und standardnormalverteilten Zielwerten. Verwenden Sie einen festen Batch mit 256 standardnormalverteilten Eingaben. Geben Sie den ersten Schritt an, in dem eine Schicht einen nicht endlichen Wert ausgibt, und nennen Sie die erste solche Schicht in der Reihenfolge der Netztiefe. Prüfen Sie auch den Verlust in jedem Schritt. Geben Sie an, ob er bereits einen nicht endlichen Wert annimmt, während noch alle Schichtausgaben endlich sind.

<!--en:3864c825-->
4. [code] **Gradientennormen.** ● Erstellen Sie `net` neu. Multiplizieren Sie die Gewichtsmatrix jeder linearen Schicht in den Residualblöcken mit 3. Verwenden Sie den Mittelwert der quadrierten Ausgaben für einen festen Batch als Verlust.
    1. Berechnen Sie wie in diesem Abschnitt für jeden Block die Norm seines Ausgabegradienten. Bestimmen Sie die Norm seines Parametergradienten aus der Baumstruktur, die `nnx.grad` zurückgibt. Stellen Sie beide Arten von Normen grafisch in Abhängigkeit von der Tiefe dar.
    1. Drücken Sie den Gradienten der Gewichtsmatrix einer linearen Schicht durch deren Eingabe und ihren Ausgabegradienten aus. Erklären Sie damit, warum sich die beiden Tiefenprofile unterscheiden. Welche Normen sollten Sie überwachen, um ein explodierendes Signal im Rückwärtsdurchlauf zu erkennen?
   :exlabel:`ex_reproducibility-inspection-gradient-norms`

<!--en:62210101-->
[Diskussion](https://d2l.discourse.group/t/36759)

<!--en:85e9f461-->
Reproduzierbarkeit, Fehlersuche und Modellinspektion<br>**PRNG-Startwerte und Schlüssel · Arbeitsprozesse des Datenladers · nicht assoziative atomare GPU-Additionen · Vorwärts- und Rückwärts-Hooks**

<!--en:8be70322-->
::: {.slide title="Warum Reproduzierbarkeit und Modellinspektion zusammengehören"}

<!--en:bd549f17-->
[Wissenschaftliche Aussagekraft und Fehlersuche]{.kicker}

<!--en:4f5024a3-->
- **Ist ein Genauigkeitsgewinn von 0.3 Punkten real oder zufälliges Rauschen?** Fünf Läufe mit *genau demselben* Trainingsskript für Fashion-MNIST unterscheiden sich nur im Zufallsstartwert. Ihre Genauigkeiten liegen $1\text{--}4$ Prozentpunkte auseinander.

<!--en:fdc66a5a-->
- **Warum wurde der Verlust bei Schritt 40 000 zu `NaN`?** Bei späten Abstürzen müssen wir die Trajektorie exakt reproduzieren und Aktivierungen und Gradienten der Schichten mit Hooks untersuchen, ohne das Modell zu ändern.

<!--en:004f2dfe-->
::: {.slide title="Zufallsquellen in einer Deep-Learning-Pipeline"}

<!--en:26be0852-->
[Fünf Quellen des Zufalls]{.kicker}

<!--en:0f1d1f3b-->
- **Zufall in Modell und Schichten:** Gewichtsinitialisierung ($\mathbf{W}_0$) und stochastische Masken im Vorwärtsdurchlauf (Dropout $\boldsymbol{\xi}_j$).

<!--en:cd7f5dd4-->
- **Zufall bei Daten und Arbeitsprozessen:** Mischen der Minibatches pro Epoche, zufällige Datenaugmentierungen und per Fork erzeugte Arbeitsprozesse des Datenladers. Ohne eigene Startwerte erben diese identische Kopien des NumPy-Generatorzustands.

<!--en:56c6fd6b-->
::: {.slide title="Mit festen Startwerten eine Trainingstrajektorie bitgenau wiederholen"}

<!--en:82b483f9-->
[Globale Startwerte und explizite Schlüssel]{.kicker}

<!--en:3fce1fb4-->
Setzen wir den Startwert jedes PRNG, den die Schleife verwendet, liefern zwei Läufe mit `seed=0` bis auf das letzte Bit identische Verluste und Gewichte:

<!--en:466a80ed-->
- **Geltungsbereich beachten:** Ein Framework-Startwert setzt die Startwerte von `random` aus Python und `np.random` aus NumPy **nicht** automatisch. Nutzt Ihr Datensatz NumPy, setzen Sie auch dort explizit den Startwert.

<!--en:b5aa835d-->
::: {.slide title="Private Generatorobjekte entkoppeln Zufallsfolgen"}

<!--en:f364e56d-->
[Zufallsfolgen isolieren]{.kicker}

<!--en:6f3d7085-->
- Ein privates **Generatorobjekt** oder ein abgeleiteter Schlüssel isoliert eine Zufallsfolge. Ein zusätzlicher Aufruf zur Fehlersuche verändert dann weder nachfolgende Gewichte noch Batches:

<!--en:58685406-->
::: {.slide title="Der Fork-Fehler bei Datenladern mit mehreren Arbeitsprozessen" only="pytorch,mxnet"}

<!--en:b4aa8ff4-->
[Warum Arbeitsprozesse nach einem Fork dieselben Zufallszahlen ziehen]{.kicker}

<!--en:7301e811-->
Erzeugt ein `DataLoader` mit `num_workers=2` Prozesse über den Betriebssystemaufruf `fork`, erbt jeder Prozess eine identische Kopie jedes `np.random.default_rng()`-Attributs des Datensatzes:

<!--en:afbee469-->
- **Identische Batches in beiden Arbeitsprozessen (Zeile 1):** Ohne neue Startwerte erzeugen Prozess 0 und Prozess 1 genau dieselben „Zufallszahlen“. Die beiden Batches in der ersten Zeile sind daher identisch.

<!--en:6c6a6905-->
- **Abhilfe (Zeile 2):** Setzen Sie den Startwert jedes Generators in `worker_init_fn` mit `get_worker_info().seed` neu. Alternativ setzen Sie ihn pro Beispiel anhand des Index `[base_seed, i]`.

<!--en:28be3484-->
::: {.slide title="Deterministisches Mischen in Datenpipelines mit mehreren Threads" only="tensorflow"}

<!--en:09dc5457-->
[Startwerte in tf.data-Pipelines]{.kicker}

<!--en:a3ebe9f9-->
- In TensorFlow nutzt `tf.data` parallele Arbeitsthreads innerhalb eines Prozesses und akzeptiert einen expliziten Startwert `seed=` beim Aufruf von `.shuffle(buffer_size, seed=0)`:

<!--en:d5aafd68-->
::: {.slide title="Explizite PRNG-Schlüssel vermeiden versteckte Zustände in Arbeitsprozessen" only="jax"}

<!--en:a41ade9b-->
[Zufall als reiner Wert]{.kicker}

<!--en:16ad29af-->
- **Zustandslose Schlüssel in funktionalem Code:** In JAX ist `jax.random.normal(key, shape)` eine reine, deterministische Funktion von `key`.

<!--en:e9d84f5c-->
- **Unabhängige Zufallsfolgen pro Arbeitsprozess:** Leiten wir mit `jax.random.split(base_key, num_workers)` oder `jax.random.fold_in(base_key, idx)` Schlüssel aus einem Basisschlüssel ab, können keine unbeabsichtigten Kopien einer Zufallsfolge entstehen.

<!--en:a26ddd2f-->
::: {.slide title="Varianz bei fünf Läufen mit unterschiedlichen Startwerten"}

<!--en:866c5107-->
[Statistische Reproduzierbarkeit]{.kicker}

<!--en:fa09763d-->
Trainieren wir eine Softmax-Regression auf Fashion-MNIST für 3 Epochen mit fünf Startwerten (`0..4`), zeigt sich bei gleichem Code und gleichen Hyperparametern die natürliche Varianz der stochastischen Optimierung:

<!--en:ace2554f-->
- **Eine Differenz bei nur einem Startwert ist wenig aussagekräftig:** Die endgültigen Validierungsgenauigkeiten streuen je nach Startwert um $1\text{--}4$ Prozentpunkte. Geben Sie für mehrere Startwerte stets **Mittelwert $\pm$ Standardabweichung** an.

<!--en:f97b51cf-->
::: {.slide title="Warum die Gleitkommaaddition nicht assoziativ ist"}

<!--en:227aefdc-->
[Eine Besonderheit der Arithmetik]{.kicker}

<!--en:41b09298-->
- Jede Gleitkommaaddition rundet auf den nächsten darstellbaren Wert. Daher gilt im Beispiel $(a + b) + c \neq a + (b + c)$:

<!--en:195b0ccd-->
- **Verlust kleiner Summanden bei $10^8$:** Nahe $10^8$ haben benachbarte `float32`-Zahlen den Abstand $8.0$. Daher ergibt sich $10^8 + (-10^8 + 1.0) = 0.0$, während $(10^8 - 10^8) + 1.0 = 1.0$ gilt.

<!--en:e40de096-->
::: {.slide title="Atomare GPU-Reduktionen und Schalter für deterministische Kernel"}

<!--en:7bf655ce-->
[Warum GPU-Läufe trotz festem Startwert auseinanderlaufen]{.kicker}

<!--en:81fab9ba-->
Auf einer GPU addieren Tausende Threads Teilsummen in gemeinsamen Speicherplätzen, etwa bei `scatter_add`, `index_add_`, Rückwärtsdurchläufen von Einbettungen und Faltungen. Diese **atomaren Additionen** erfolgen in der Reihenfolge, in der die Threads fertig werden:

<!--en:e52aa650-->
- **Bitgenauen GPU-Determinismus aktivieren:** Deterministische Kernel erzwingen einen festen Reduktionsbaum. Aufeinanderfolgende GPU-Läufe stimmen dann Bit für Bit überein.

<!--en:8664cc57-->
::: {.slide title="Bitgenauer Determinismus und wissenschaftliche Reproduzierbarkeit"}

<!--en:ce624784-->
[Geltungsbereich bitgenauer Wiederholbarkeit]{.kicker}

<!--en:51ff27ba-->
- **Bitgenauer Determinismus dient der lokalen Fehlersuche:** Auf **derselben Maschine mit derselben GPU-Architektur und Bibliotheksversion** garantiert er identische Bits. Durch Halbieren des Suchbereichs finden wir genau den Schritt, in dem ein Fehler oder `NaN` auftritt.

<!--en:9e71e198-->
- **Wissenschaftliche Reproduzierbarkeit ist statistisch:** Echte Modellverbesserungen müssen über mehrere Zufallsstartwerte ($N \ge 5$) Bestand haben, auch wenn andere Hardware Gleitkommaoperationen geringfügig anders klammert.

<!--en:c48b7b58-->
::: {.slide title="Modellinspektion mit Hooks ohne Eingriff in den Code"}

<!--en:e00a8fa3-->
[Schichten von außen beobachten]{.kicker}

<!--en:45516778-->
Bei der Fehlersuche in einem tiefen Netz oder einem unveränderlichen Modell aus einer fremden Bibliothek ist es umständlich, jede `forward`-Methode um `print`-Anweisungen zu ergänzen. **Hooks** greifen über `__call__` von außen in den Ablauf jedes Untermoduls ein:

<!--en:471c5590-->
- **Pipeline um `forward`:** Der Aufruf `net(X)` startet die registrierten vorgeschalteten Hooks, führt `forward(X)` aus und übergibt `(module, inputs, output)` an die registrierten Vorwärts-Hooks.

<!--en:a372e4ec-->
::: {.slide title="Aufbau eines Modells mit 8 Residualblöcken zur Inspektion"}

<!--en:e829c7c5-->
[Residualblöcke mit Xavier-Initialisierung ohne Skalierung]{.kicker}

<!--en:eb11f34b-->
- Wir bauen ein MLP mit 8 Residualblöcken ($20 \to 64 \to 8\text{ blocks of width } 64 \to 10$). Die Gewichte initialisieren wir nach Xavier mit Gleichverteilung, die Bias-Terme mit null.

<!--en:5fa35b14-->
::: {.slide title="Die 8 Residualblöcke im Code"}

<!--en:e829c7c5-2-->
[Residualblöcke mit Xavier-Initialisierung ohne Skalierung]{.kicker}

<!--en:3a3c44c9-->
::: {.slide title="Standardabweichung der Aktivierungen in Abhängigkeit von der Tiefe"}

<!--en:c53f9012-->
[Diagnose mit Vorwärts-Hooks]{.kicker}

<!--en:2bbaf4b9-->
Ein Vorwärts-Hook an jedem Modul der obersten Ebene erfasst das Wachstum der Aktivierungen über alle 10 Schichten. Dafür genügt ein Vorwärtsdurchlauf mit 256 standardnormalverteilten Eingaben:

<!--en:db1b7ed0-->
- **Stetiges Wachstum über die Blöcke:** Die Standardabweichung der Ausgaben beträgt zunächst $\approx 0.7$ nach der $20 \to 64$-Eingabeschicht und steigt bis nach dem achten Residualblock stetig auf $\approx 3.2\text{--}4.5$.

<!--en:02967725-->
- **Keine Änderung am Modell:** Erfassen wir Zwischenausgaben von außen, können wir die Größenordnung der Ausgaben jedes Blocks protokollieren, ohne die Schichtdefinitionen zu ändern.

<!--en:6f6cf73d-->
::: {.slide title="Warum die Standardabweichung bei Xavier-Initialisierung pro Residualblock um den Faktor √1.5 ≈ 1.22 wächst"}

<!--en:3d6d2933-->
[Analytische Überprüfung der Hook-Messwerte]{.kicker}

<!--en:cdfc798c-->
- **Warum der Residualzweig $\frac{1}{2}\text{Var}[\mathbf{x}_k]$ beiträgt:** Die erste $64 \to 64$-Schicht mit Xavier-Initialisierung erhält die Varianz ($1\times$), die ReLU halbiert den Mittelwert der Quadrate ($\frac{1}{2}\times$), und die zweite Xavier-initialisierte Schicht erhält diese Hälfte ($\frac{1}{2}\times$).

<!--en:8f638193-->
- **Genaue Übereinstimmung mit den Hook-Messwerten:** Bei einem Anfangswert von $\approx 0.69$ und einem Faktor von $\sqrt{1.5} \approx 1.225$ pro Block erwarten wir nach 8 Blöcken $0.69 \times 1.5^4 \approx 3.5$.

<!--en:37f9a4f5-->
::: {.slide title="Zwei Regeln für Hooks ohne Speicherlecks" only="pytorch,mxnet"}

<!--en:895e3c36-->
[Speicherlecks bei der automatischen Differentiation vermeiden]{.kicker}

<!--en:7b218dd3-->
1. **Gespeicherte Tensoren immer mit `.detach()` abtrennen (oder Skalare mit `.item()` extrahieren):** Hängen wir einen noch mit dem Graphen verbundenen Aktivierungstensor `output` direkt an eine Python-Liste an, bleibt der gesamte Rechengraph für den Rückwärtsdurchlauf über mehrere Schritte im GPU-Speicher.

<!--en:1bc33552-->
2. **Hooks nach Gebrauch stets über ihre Handles entfernen (`h.remove()` / `h.detach()`):** Melden Sie die Rückruffunktionen in einem `finally`-Block ab, damit sie bei späteren Vorwärtsdurchläufen nicht mehr ausgeführt werden.

<!--en:050aae61-->
::: {.slide title="Die erste Schicht mit NaN in einem Vorwärtsdurchlauf finden"}

<!--en:fc2b1478-->
[NaN automatisch erkennen]{.kicker}

<!--en:79a88a35-->
Vorwärts-Hooks laufen genau in der Ausführungsreihenfolge der Schichten. Prüfen wir jede Blattschicht auf endliche Werte (`isfinite(output).all()`), hält die Ausführung sofort an und nennt die erste Schicht, die `inf` oder `NaN` ausgibt:

<!--en:b284d278-->
::: {.slide title="Gradientennormen im Rückwärtsdurchlauf in Abhängigkeit von der Tiefe"}

<!--en:a287f550-->
[Rückwärts-Hooks und Gradientenfluss]{.kicker}

<!--en:1cf206ab-->
Untersuchen wir während der Backpropagation die Gradientennorm an der Ausgabe jedes Residualblocks, sehen wir das Spiegelbild des Aktivierungswachstums im Vorwärtsdurchlauf:

<!--en:5e0d0375-->
- **Verstärkung im Rückwärtsdurchlauf ($\times 1.2\text{--}1.4$ pro Block):** Jeder Residualblock addiert im Rückwärtsdurchlauf die Jacobi-Matrix des Residualzweigs zum Beitrag der Skip-Verbindung. Von Block 8 zurück bis Block 1 wachsen die Normen der Ausgabegradienten dadurch um den Faktor $5\text{--}6\times$.

<!--en:0ee8c46f-->
::: {.slide title="Verständnisfragen zu Determinismus, Varianzwachstum und Hooks"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:50a73e0a-->
Prüfen Sie Ihre Diagnosefähigkeiten anhand dieser drei Fragen:

<!--en:1daec1f8-->
1. Ein Trainingsskript initialisiert den PRNG des Frameworks zu Beginn mit `0`. Der Datensatz addiert in `__getitem__` gaußsches Rauschen. Dafür verwendet er `self.rng = np.random.default_rng()` aus seinem Konstruktor `__init__`. Stimmen zwei Läufe mit je einem Prozess überein? Was geschieht bei 4 Arbeitsprozessen des Datenladers mit `fork`?

<!--en:bcceb214-->
2. In unseren Xavier-initialisierten Residualblöcken wächst die Standardabweichung der Aktivierungen pro Block um den Faktor $\sqrt{1.5} \approx 1.225$. Nach wie vielen Blöcken $k$ ist sie mehr als $100\times$ so groß wie direkt nach der Eingabeschicht?

<!--en:86d9f588-->
3. Was geschieht beim Training mit dem GPU-Speicher, wenn ein Vorwärts-Hook in jedem Schritt `history.append(output)` ausführt, ohne `.detach()` aufzurufen?

<!--en:557b98b6-->
**Antworten:**

<!--en:9de27530-->
1. **Läufe mit einem Prozess unterscheiden sich**, weil `np.random.default_rng()` jedes Mal Entropie vom Betriebssystem bezieht und den Startwert des Frameworks ignoriert. **Bei 4 per Fork erzeugten Arbeitsprozessen erben alle 4 identische Kopien** von `self.rng` und addieren identisches Rauschen.

<!--en:b49f938b-->
2. **$1.5^{k/2} \ge 100 \implies k \ge \frac{2 \ln 100}{\ln 1.5} \approx 22.7$ (23 Blöcke).**

<!--en:ac6ecb24-->
3. **Absturz wegen Speichermangels (OOM):** Jeder Tensor `output` hält seinen gesamten Autograd-Graphen samt Zwischenaktivierungen dauerhaft in `history`.

<!--en:777330a2-->
::: {.slide title="Zusammenfassung zu Reproduzierbarkeit, Fehlersuche und Modellinspektion"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:27af68a7-->
- **PRNG-Startwerte und Statistik:** Setzen Sie die Startwerte für das Framework, NumPy, das Python-Modul `random` und die Arbeitsprozesse des Datenladers. Geben Sie Mittelwert $\pm$ Standardabweichung über $N \ge 5$ Startwerte an.

<!--en:84835d37-->
- **Bitgenauer Determinismus:** Aktivieren Sie zum Eingrenzen von Fehlern deterministische GPU-Kernel. Sie ersetzen nicht assoziative parallele atomare Additionen durch Reduktionen mit fester Reihenfolge.

<!--en:3ea0da55-->
- **Hooks während der Ausführung:** Erfassen Sie Aktivierungen im Vorwärtsdurchlauf und Gradienten im Rückwärtsdurchlauf von außen. So messen Sie deren Größenordnung und finden die erste Schicht mit `NaN`. Trennen Sie gespeicherte Tensoren vom Graphen und entfernen Sie die Hooks anschließend über ihre Handles.
