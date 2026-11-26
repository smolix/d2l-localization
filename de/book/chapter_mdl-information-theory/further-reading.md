<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:cb8b64be-->
In den Hauptabschnitten dieses Kapitels beschränken wir uns auf Ergebnisse, die wir sauber beweisen und an kleinen Beispielen überprüfen können. Die folgenden kurzen Beiträge knüpfen jeweils an einen dieser Abschnitte an und behandeln verwandte Ideen, die dort keinen Platz fanden. Wenn Sie wenig Zeit haben, beginnen Sie mit den drei Themen, die Ihnen im Alltag am ehesten begegnen: der Kalibrierung von Modellen, der Konstanten für den irreduziblen Verlust in Skalierungsgesetzen (engl. scaling laws) und dem Zielkonflikt zwischen Verzerrung und Varianz bei Schätzern der gegenseitigen Information (engl. mutual information). Auch die Bits-Back-Kodierung lohnt einen Blick, schon wegen der raffinierten Idee dahinter. Kanalkapazität (engl. channel capacity) und Informationsgeometrie können warten. Kein späterer Abschnitt des Buches setzt diese Seite voraus.

<!--en:198fec87-->
## Entropie und Verluste

<!--en:50387741-->
### Kalibrierung nach dem Training

<!--en:b874e174-->
Minimieren wir den logarithmischen Verlust über die Grundgesamtheit, so erhalten wir im Prinzip die wahren bedingten Klassenwahrscheinlichkeiten (:numref:`subsec_mdl-irreducible-loss`). Diese sind automatisch kalibriert. Trainierte neuronale Netze erreichen dieses Optimum jedoch selten. Wie :citet:`Guo.Pleiss.Sun.Weinberger.2017` in einer viel beachteten Arbeit feststellten, neigen moderne Netze zu einem Fehler in eine bestimmte Richtung: Ihre Konfidenz ist überhöht. Eine wesentliche Ursache sind One-Hot-Trainingslabel. Sie belohnen es, wenn das größte Logit den übrigen um einen beliebig großen Betrag vorauseilt (:numref:`sec_mdl-information_theory`). Sobald ein Netz den Trainingsdatensatz fehlerfrei klassifiziert, kann es den Verlust nur noch senken, indem es die Logits immer weiter auseinanderschiebt. Gegen Ende des Trainings beobachten die Autoren auf zurückgehaltenen Daten einen Anstieg des logarithmischen Verlusts bei gleichbleibendem Klassifikationsfehler. Die Überanpassung (engl. overfitting) betrifft die geschätzten Wahrscheinlichkeiten, nicht die Entscheidungen. Die Temperaturskalierung (engl. temperature scaling, :numref:`sec_classification`) behebt das Problem weitgehend mit einem einzigen Skalar, den man auf einem separaten Validierungsdatensatz anpasst. Es ist dieselbe Softmax-Temperatur wie bei der Destillation. Hier schwächt sie Vorhersagen mit überhöhter Konfidenz ab.

<!--en:231481ef-->
Wie viel lässt sich mit einer solchen nachträglichen Kalibrierung erreichen? Mit $\bar{\mathbf{p}}(\mathbf{q})$ bezeichnen wir die empirischen Klassenhäufigkeiten unter allen Eingaben, für die das Modell den Wahrscheinlichkeitsvektor $\mathbf{q}$ ausgibt. Jede Kreuzentropie ist die Summe aus einer Entropie und einer KL-Divergenz. Wenden wir diese Identität bedingt auf jeden vorhergesagten Vektor $\mathbf{q}$ an, so zerfällt der logarithmische Verlust auf zurückgehaltenen Daten in zwei Terme:

<!--en:9284eb3f-->
Der erste Term misst die Fehlkalibrierung in Nat. Er verschwindet genau dann, wenn Ereignisse tatsächlich so häufig eintreten, wie es die vorhergesagten Wahrscheinlichkeiten angeben. Der zweite Term misst die Unsicherheit über das wahre Label, die selbst dann bleibt, wenn wir die Vorhersage des Modells kennen. Durch nachträgliches Umrechnen der vorhergesagten Wahrscheinlichkeiten lässt sich der erste Term gegen null treiben. Den zweiten kann keine solche Abbildung verringern. Die Temperaturskalierung ist eine Familie solcher Abbildungen mit nur einem Parameter. Das macht sie besonders einfach. Allerdings reicht ein einziger Skalar nicht aus, wenn die Konfidenz bei manchen Klassen zu hoch und bei anderen zu niedrig ist. Allgemeiner lässt sich jede echte Bewertungsregel analog in einen Divergenzterm und einen Entropieterm zerlegen :cite:`Gneiting.Raftery.2007`.

<!--en:e798f114-->
### Entropie relativ zu einem Referenzmaß

<!--en:dd615881-->
In :numref:`subsec_mdl-differential-entropy` sind wir auf eine unangenehme Eigenschaft der differentiellen Entropie gestoßen: Beim Wechsel von Metern zu Zentimetern steigt sie um $\ln 100$. Die KL-Divergenz bleibt dagegen gleich, weil sich beide Dichten im Quotienten um denselben Faktor ändern. Auch die Entropie lässt sich koordinateninvariant definieren. Dazu messen wir sie relativ zu einer Referenzdichte $m$ und setzen $-\int p \log(p/m)\,dx = -D_{\textrm{KL}}(p\|m)$. Die gewöhnliche differentielle Entropie $h$ ist der Spezialfall $m = 1$ in den jeweils gewählten Einheiten. Eine konstante Dichte von eins pro Meter beschreibt ein anderes Maß als eine Dichte von eins pro Zentimeter. Hinter der Verschiebung um $\ln 100$ steckt also ein Wechsel des Referenzmaßes. :citet:`Jaynes.1968` leitet diese relative Form als Grenzwert diskreter Entropien auf immer feineren Gittern her. Damit lässt sich das Prinzip der maximalen Entropie natürlich formulieren: „Bleibe so nahe am Referenzmaß, wie es die Nebenbedingungen erlauben.“ Bis auf das Vorzeichen ist der KL-Strafterm eines Variationsautoencoders (:numref:`subsec_mdl-variational-bottleneck`) eine solche Entropie. Die A-priori-Verteilung übernimmt die Rolle von $m$. Deshalb bleibt der Term bei glatten Umparametrisierungen des latenten Raums invariant.

<!--en:0b7a4e8f-->
## Codes und Kompression

<!--en:ee8c3a56-->
### Kompression ohne Modell

<!--en:ca28237f-->
Jeder Kompressor in :numref:`sec_mdl-coding-compression` sagt das nächste Symbol mit einem expliziten probabilistischen Modell vorher. Die Verfahren der Lempel-Ziv-Familie kommen ganz ohne diese Voraussetzung aus. LZ77 :cite:`Ziv.Lempel.1977` ersetzt jede wiederholt auftretende Zeichenfolge durch einen kurzen Verweis auf ein früheres Vorkommen. LZ78 :cite:`Ziv.Lempel.1978` baut beim Lesen der Datei schrittweise ein Wörterbuch aus Zeichenfolgen auf. Bemerkenswerterweise sind beide Algorithmen *universell*: Für jede stationäre ergodische Quelle nähert sich ihre mittlere Codelänge mit wachsender Datei der wahren Entropierate, ganz ohne Vorwissen über die Quelle. Bei einer solchen Quelle lassen sich die langfristigen statistischen Eigenschaften grob gesagt an einer einzigen langen Stichprobe ablesen. In der Praxis ist die Konvergenz gegen diesen Grenzwert allerdings langsam. Für *The Time Machine* benötigt `zlib`, das LZ77 mit der Huffman-Kodierung kombiniert, $2.84$ Bit pro Zeichen. Das einfache Zählmodell aus :numref:`subsec_mdl-lm-compressor` sagt jedes Zeichen aus den drei vorhergehenden Zeichen vorher und kommt bereits mit $2.55$ aus.

<!--en:7e225c8f-->
### Bits-Back-Kodierung

<!--en:1bdf9c43-->
Ein latentes Variablenmodell wie ein Variationsautoencoder kann den arithmetischen Kodierer aus :numref:`subsec_mdl-arithmetic-coding` nicht direkt steuern. Der Kodierer braucht die exakte Randwahrscheinlichkeit $p(x)$. Das Modell liefert uns dagegen nur die ELBO (:numref:`sec_mdl-latent-em-elbo`), eine untere Schranke für $\log p(x)$, die auf einer Näherung $q(z \mid x)$ der A-posteriori-Verteilung beruht. Überraschenderweise entspricht diese Schranke dennoch einer erreichbaren Dateilänge. Dahinter steckt eine raffinierte Dualität: Einen komprimierten Bitstrom zu dekodieren ist mathematisch dasselbe wie Sampling. Speisen wir zufällige Bits in einen für $q(z \mid x)$ ausgelegten Decoder ein, so verbraucht er ungefähr $\log_2 1/q(z \mid x)$ Bit und gibt als latentes Sample einen Wert $z$ aus der Verteilung $q$ aus.

<!--en:aedd0e83-->
Die *Bits-Back-Kodierung* macht aus dieser Beobachtung ein praktisches Verfahren. Nehmen wir an, der Sender hat bereits eine komprimierte Datei mit früheren Datenpunkten. Gute Kompression beseitigt Redundanz. Deshalb lassen sich die Bits am Dateiende dem Anschein nach nicht von zufälligen Münzwürfen unterscheiden. Um einen neuen Datenpunkt $x$ zu kodieren, leiht sich der Sender $\log_2 1/q(z \mid x)$ Bit vom Dateiende und „dekodiert“ sie mit $q(z \mid x)$. So zieht er einen latenten Wert $z$. Anschließend kodiert der Sender $x$ mit $p(x \mid z)$ und $z$ mit der A-priori-Verteilung $p(z)$ und hängt beides an die Datei an. Um $x$ zu entpacken, kehrt der Empfänger die Schritte um. Er dekodiert $z$ mit der A-priori-Verteilung und $x$ mit $p(x \mid z)$. Dann *kodiert* er $z$ mit $q(z \mid x)$ *erneut* und setzt damit die geliehenen Bits an ihre ursprüngliche Stelle zurück. Insgesamt wächst die Datei um $\log_2 1/p(x \mid z) + \log_2 1/p(z)$ Bit und schrumpft um $\log_2 1/q(z \mid x)$ Bit. Ihr erwarteter Nettozuwachs unter $q$ ist also die negative ELBO. Der Empfänger muss die Schritte des Senders vom letzten bis zum ersten rückgängig machen. Der Kodierer muss sich daher wie ein Stapelspeicher verhalten. Die Idee der Bits-Back-Kodierung geht auf :citet:`Wallace.1990` und :citet:`Hinton.Van-Camp.1993` zurück. Den ersten funktionsfähigen Kodierer nach diesem Prinzip bauten :citet:`Frey.Hinton.1997` auf der klassischen arithmetischen Kodierung auf, die sich wie eine Warteschlange verhält. Asymmetrische Zahlensysteme :cite:`Duda.2013` arbeiten von Haus aus nach dem Stapelprinzip. Damit zeigten :citet:`Townsend.Bird.Barber.2019`, dass ein kleiner Variationsautoencoder MNIST mit Raten nahe seiner negativen ELBO komprimiert.

<!--en:71f3fe1f-->
### Beschreibungslänge über zweiteilige Codes hinaus

<!--en:90b82ad3-->
Der zweiteilige Code aus :numref:`subsec_mdl-mdl` kodiert die Modellparameter explizit mit einer gewählten Genauigkeit. Die gesamte Beschreibungslänge hängt deshalb davon ab, wie fein wir den Parameterraum diskretisieren. Bei einem *präquenziellen* Code (engl. prequential code) entfällt die Übertragung von Parametern ganz. Wie beim adaptiven Zählmodell in :numref:`subsec_mdl-lm-compressor` kodieren wir jeden Datenpunkt mit dem Modell, das wir an die bisherigen Punkte angepasst haben. Die Kosten für das Erlernen des Modells gehen damit direkt in die Codelänge der Daten ein. :citet:`Blier.Ollivier.2018` komprimierten die Trainingslabel tiefer neuronaler Netze mit präquenzieller Kodierung. Ihre Codes waren deutlich kürzer als zweiteilige Codes und auch als variationelle Codes, die die Gewichte des Netzes mit Bits-Back-Kodierung komprimieren. Selbst ein stark überparametrisiertes Netz komprimiert bei dieser Messung seine Label. :citet:`Grunwald.2007` behandelt sowohl präquenzielle Codes als auch die normierte Maximum-Likelihood. Deren Code minimiert den maximalen Regret gegenüber dem Modell der Klasse, das sich im Nachhinein als das beste erweist. Denken wir diesen Kompressionsansatz bis zum Ende, gelangen wir zur Kolmogorov-Komplexität, der Länge des kürzesten Computerprogramms, das die Daten ausgibt. Sie vereint all diese Ideen, ist aber bekanntlich nicht berechenbar :cite:`Li.Vitanyi.2019`.

<!--en:aa0d5d2d-->
### Die Konstante eines Skalierungsgesetzes

<!--en:8c0f0f75-->
In :numref:`sec_scaling-laws` beschreiben wir den Verlust von Sprachmodellen auf zurückgehaltenen Daten über viele Größenordnungen hinweg als Summe aus einer Konstanten $E$ und zwei Potenztermen. Der eine nimmt mit der Anzahl der Parameter ab, der andere mit der Anzahl der Trainingstoken :cite:`hoffmann2022training`. Es liegt nahe, $E$ als die wahre Entropie natürlichsprachlicher Texte zu deuten. Im besten Fall entspricht $E$ tatsächlich der Entropierate pro Token (:numref:`subsec_mdl-entropy-rate`), die kein erwarteter Verlust unterschreiten kann. In der Praxis ist $E$ allerdings nur eine angepasste Regressionskonstante. Wir extrapolieren sie aus Verlustwerten, die alle deutlich höher liegen. Eine direkte Messung der Entropierate ist das nicht. In einer früheren Arbeit passten :citet:`kaplan2020scaling` ähnliche Skalierungskurven ganz ohne additive Konstante an. Bei hinreichend großen Modellen und Datenmengen muss diese Näherung versagen, weil der Verlust die Entropieuntergrenze nicht unterschreiten kann. Außerdem gilt $E$ pro Token. Konstanten, die mit verschiedenen Tokenizern geschätzt wurden, lassen sich erst dann direkt vergleichen, wenn beide in Bit pro Byte umgerechnet sind (:numref:`subsec_bits-per-byte`).

<!--en:91d5bbee-->
## Vergleich von Verteilungen

<!--en:41aa6e51-->
### Rényi-Divergenzen und Datenschutz

<!--en:e2855dc6-->
Wie viel gibt ein randomisierter Algorithmus über einen einzelnen Trainingseintrag preis, wenn wir ihn wiederholt ausführen? Der differentielle Datenschutz (engl. differential privacy) erfasst diese Preisgabe als Divergenz zwischen den Ausgabeverteilungen des Algorithmus für zwei Datensätze, die sich nur in diesem einen Eintrag unterscheiden. Beim stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD) mit differentiellem Datenschutz kommt in jedem Gradientenschritt Rauschen hinzu, und das über Tausende von Iterationen. Wir brauchen deshalb eine Divergenz, die sich über unabhängige Schritte leicht zusammensetzen lässt. Die Rényi-Divergenz der Ordnung $\alpha$ :cite:`Renyi.1961`,

<!--en:09037f16-->
hat diese Eigenschaft. Der Logarithmus steht *außerhalb* der Summe. Bei einer f-Divergenz wirkt die konvexe Funktion dagegen innerhalb der Summe. Für unabhängige Koordinaten zerfällt die Summe in ein Produkt, das der Logarithmus wieder in eine Summe überführt. Daher ist $D_\alpha$ über unabhängige Schritte additiv, genau wie die KL-Divergenz, anders als $\chi^2$ oder der Totalvariationsabstand. Der differentielle Datenschutz nach Rényi :cite:`Mironov.2017` beschränkt $D_\alpha$ für einen einzelnen verrauschten Schritt und summiert diese Schranke über den gesamten Trainingslauf. Die KL-Divergenz ist ebenfalls additiv, beschränkt das logarithmierte Dichteverhältnis $\log(p(x)/q(x))$ aber nur im Mittel. Dagegen wächst $D_\alpha$ monoton mit $\alpha$, von der KL-Divergenz für $\alpha \to 1$ bis zum größten logarithmierten Dichteverhältnis für $\alpha \to \infty$. Eine größere Ordnung $\alpha$ schützt daher auch vor seltenen Ausgaben, die besonders viel preisgeben. :citet:`Van-Erven.Harremoes.2014` geben einen umfassenden Überblick über diese Grenzwerte und Ungleichungen. Aufgabe :exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`d in :numref:`sec_mdl-divergences-distances` stellt den Zusammenhang zwischen $D_\alpha$ und den $\alpha$-Divergenzen her.

<!--en:61740827-->
### Informationsgeometrie

<!--en:ca692b58-->
In :numref:`sec_mdl-divergences-distances` haben wir mit einer kurzen Taylor-Entwicklung gezeigt, dass jede glatte f-Divergenz zwischen $q_{\boldsymbol\theta + \mathbf{h}}$ und $q_{\boldsymbol\theta}$ bis zur zweiten Ordnung durch die quadratische Form $\tfrac12 f''(1)\, \mathbf{h}^\top \mathbf{I}(\boldsymbol\theta)\, \mathbf{h}$ mit der Fisher-Informationsmatrix gegeben ist. Das mag wie ein algebraischer Zufall aussehen. Der Satz von Chentsov zeigt jedoch, dass es gar nicht anders sein kann. Eine positiv definite quadratische Form weist jedem infinitesimalen Schritt $\mathbf{h}$ eine Länge zu. So wird der Raum der Verteilungen zu einer riemannschen Mannigfaltigkeit. Sie misst lokale Schritte, während die Metriken aus :numref:`sec_mdl-what-is-a-divergence` globale Abstände zwischen zwei Verteilungen messen. Für Verteilungen auf endlichen Mengen bewies Chentsov, dass die Fisher-Information bis auf eine positive Konstante die *einzige* riemannsche Metrik ist, die beim Ersetzen der Rohdaten durch eine suffiziente Statistik invariant bleibt :cite:`Cencov.1982`. Jede f-Divergenz besitzt diese Invarianz. Aus einer suffizienten Statistik lassen sich auch ohne Kenntnis von $\boldsymbol\theta$ erneut Daten ziehen. Deshalb gilt die Datenverarbeitungsungleichung aus :numref:`subsec_mdl-f-div-data-processing` in beiden Richtungen und erzwingt Gleichheit. Die lokale quadratische Näherung jeder glatten f-Divergenz muss also ein Vielfaches der Fisher-Information sein. Messen wir die Länge von Gradientenschritten in dieser Geometrie, so erhalten wir den natürlichen Gradienten. K-FAC approximiert ihn mit einem Kronecker-Produkt pro Schicht (:numref:`subsec_mdl-preconditioning-ladder`). Die KL-Trust-Region aus :numref:`sec_ppo` ist bis zur zweiten Ordnung eine Kugel in genau dieser Metrik. :citet:`Amari.2016` entwickelt diese geometrische Sichtweise im Detail und zeigt, wie sich die Maximum-Likelihood-Schätzung und beide Schritte des EM-Algorithmus (:numref:`sec_mdl-latent-em-elbo`) als orthogonale Projektionen verstehen lassen.

<!--en:e66547a1-->
### Sinkhorn-Divergenzen und Sliced-Wasserstein-Abstand

<!--en:ae9603d6-->
Wer einen Abstand aus dem optimalen Transport als Verlust für hochdimensionale Minibatches verwenden will, stößt auf zwei Hindernisse: den Rechenaufwand und die statistische Verzerrung. Entropische Regularisierung (:numref:`subsec_mdl-sinkhorn`) löst das Rechenproblem, führt aber zu einer subtilen Verzerrung. Die regularisierten Kosten für den Transport einer Verteilung zu sich selbst sind von null verschieden. Ein Modell, das diese Kosten minimiert, neigt daher zu einer engeren Verteilung als der Datenverteilung. Die Sinkhorn-Divergenz aus Aufgabe :exref:`ex_mdl-divergences-distances-sinkhorn-divergences` in :numref:`sec_mdl-test-functions-transport` behebt das Problem, indem sie den Mittelwert der beiden Kosten für den Transport zu sich selbst abzieht. Über die Aufgabe hinaus gilt ein Satz, der die Sinkhorn-Divergenz als verlässliches Trainingsziel ausweist. Auf einem beschränkten Gebiet mit dem euklidischen Abstand oder seinem Quadrat als Grundkosten ist sie nichtnegativ. Sie ist genau dann null, wenn die beiden Verteilungen übereinstimmen. Sie konvergiert genau dann gegen null, wenn die Modellverteilung schwach gegen die Datenverteilung konvergiert :cite:`Feydy.Sejourne.Vialard.ea.2019`.

<!--en:04881087-->
Der *Sliced*-Wasserstein-Abstand verfolgt einen ganz anderen Ansatz. Projizieren wir zwei hochdimensionale Stichproben auf dieselbe Gerade, genügt es für den optimalen Transport, die projizierten Werte zu sortieren (:numref:`sec_mdl-optimal-transport`). Mitteln wir diese eindimensionalen Transportkosten über viele zufällige Richtungen, erhalten wir den Sliced-Wasserstein-Abstand :cite:`Bonneel.Rabin.Peyre.ea.2015`. In hohen Dimensionen liegt eine zufällige Richtung allerdings selten in dem Unterraum, in dem sich die Verteilungen tatsächlich unterscheiden. Um diese Schwierigkeit zu umgehen, trainieren :citet:`Deshpande.Zhang.Schwing.2018` einen Diskriminator darauf, aussagekräftige Merkmale zu extrahieren. Den Sliced-Wasserstein-Abstand berechnen sie dann in diesem erlernten Merkmalsraum statt auf den rohen Pixelwerten.

<!--en:1f2dd48b-->
### Lipschitz-stetige Kritiker in Wasserstein-GANs

<!--en:295af25a-->
Das duale Problem von Kantorovich und Rubinstein in :numref:`sec_mdl-optimal-transport` optimiert über 1-Lipschitz-stetige Funktionen. Ein gewöhnliches neuronales Netz erfüllt diese Nebenbedingung nicht von sich aus. :numref:`sec_gan_objectives` vergleicht die drei wichtigsten Möglichkeiten, sie durchzusetzen: Begrenzung der Gewichte, spektrale Normalisierung und Gradientenstrafterme. :numref:`sec_gan_convergence` untersucht die Gradientenstrafterme im Detail. Der Strafterm von :citet:`Gulrajani.Ahmed.Arjovsky.ea.2017` drängt die Gradientennorm des Kritikers an zufällig interpolierten Punkten zwischen echten und erzeugten Beispielen in Richtung eins. Darin spiegelt sich eine geometrische Eigenschaft wider: Ein optimaler Kritiker hat entlang der Transportstrahlen, die einander zugeordnete Punkte verbinden, die Steigung eins (Aufgabe :exref:`ex_mdl-test-functions-transport-optimal-critics`d in :numref:`sec_mdl-test-functions-transport`).

<!--en:7f1cebf5-->
### Variationeller Gradientenabstieg nach Stein

<!--en:bb75a17e-->
Um Stichproben aus einer Verteilung $Q$ zu ziehen, von der nur der Score $s_Q$ bekannt ist, braucht man in der Regel eine lange Irrfahrt (:numref:`sec_mdl-scores`). In :numref:`sec_mdl-ksd` haben wir gesehen, dass die Kernel-Stein-Diskrepanz misst, wie weit die Verteilung $P$ einer Stichprobe von einem solchen Modell $Q$ entfernt ist. Der variationelle Gradientenabstieg nach Stein (engl. Stein variational gradient descent) :cite:`Liu.Wang.2016` macht aus dieser Diskrepanz einen deterministischen Algorithmus, der eine Menge von Partikeln gezielt in Richtung $Q$ transportiert. Wir beginnen mit einer beliebigen Punktwolke mit Verteilung $P$. Jedes Partikel $y$ bewegt sich in kleinen Schritten entlang des Geschwindigkeitsfeldes

<!--en:c2cab908-->
Die beiden Terme im Erwartungswert ergänzen einander. Der erste mittelt die Scores in der Nähe von $y$ mit Gewichten aus einem Kernel. So steuert er die Partikel bergauf in Gebiete höherer Wahrscheinlichkeitsdichte. Der zweite Term ist eine abstoßende Kraft, die $y$ von seinen Nachbarn wegdrückt und die Rolle des Rauschens in der Langevin-Dynamik übernimmt. Ohne diese Kraft würden sich alle Partikel an einem einzigen Modus sammeln. Normieren wir dieses Geschwindigkeitsfeld auf die Einheitskugel des Hilbertraums mit reproduzierendem Kern aus :numref:`sec_mdl-ksd`, erhalten wir die Richtung des steilsten Abstiegs für $D_{\textrm{KL}}(P\|Q)$. Die KL-Divergenz sinkt dabei mit der Rate $\|v\| = \mathrm{KSD}(P, Q)$. Bemerkenswerterweise dient derselbe Stein-Operator sowohl als Anpassungstest als auch als Vorschrift für den Partikeltransport. Die Partikel brauchen zwar weiterhin viele kleine Schritte, doch die Aktualisierungen sind deterministisch und die Partikel verteilen sich im Zusammenspiel. Deshalb eignet sich das Verfahren gut, um eine A-posteriori-Verteilung mit wenigen Dutzend Partikeln zu approximieren, etwa die Verteilung der Gewichte eines kleinen neuronalen Netzes in der Originalarbeit.

<!--en:0e8ee723-->
## Gegenseitige Information

<!--en:c8b9966a-->
### Codes nahe der Kanalkapazität

<!--en:5b7fc8fe-->
Shannons Satz zur Kanalkodierung (:numref:`subsec_mdl-channel-capacity`) garantiert, dass zuverlässige Kommunikation bei jeder Rate unterhalb der Kanalkapazität möglich ist. Sein Beweis beruht jedoch auf zufälligen Codebüchern, deren Dekodierung exponentiell viel Zeit benötigt. Vom Existenzbeweis bis zur Umsetzung in Hardware vergingen Jahrzehnte. Low-Density-Parity-Check-Codes (LDPC-Codes) lassen sich effizient dekodieren, indem man Nachrichten entlang eines dünn besetzten Faktorgraphen weitergibt. In der Praxis kommen sie der Kanalkapazität bemerkenswert nahe :cite:`Richardson.Urbanke.2008`. Polar-Codes :cite:`Arikan.2009` gehen noch einen Schritt weiter. Sie sind die erste explizite Codefamilie, die auf symmetrischen Kanälen mit binärer Eingabe nachweislich die Kanalkapazität erreicht. Sowohl die Kodierung als auch die Dekodierung benötigen $O(n \log n)$ Operationen bei Blocklänge $n$. Moderne 5G-Mobilfunkstandards nutzen beide Familien.

<!--en:4cf7227a-->
### Erlernte verlustbehaftete Kompression

<!--en:3a5bd207-->
In :numref:`subsec_mdl-rate-distortion` haben wir die exakte Rate-Distortion-Kurve für eine gaußsche Quelle hergeleitet. Für natürliche Fotografien ist keine Rate-Distortion-Funktion in geschlossener Form bekannt. Jeder erlernte Bildcodec erreicht daher einen praktischen Betriebspunkt irgendwo oberhalb dieser unbekannten Grenzkurve. :citet:`Balle.Laparra.Simoncelli.2017` trainieren drei Komponenten gemeinsam: einen Encoder, einen Decoder und ein *Entropiemodell* (engl. entropy model), das eine erlernte Wahrscheinlichkeitsverteilung über der quantisierten latenten Repräsentation definiert. Da die Rundungsfunktion fast überall den Gradienten null hat, ersetzen sie die Quantisierung beim Training durch additives, gleichverteiltes Rauschen. Sie minimieren $R + \lambda D$. Die Rate $R$ ist die Codelänge der latenten Repräsentation bei arithmetischer Kodierung mit dem Entropiemodell (:numref:`sec_mdl-coding-compression`). Die Größe $D$ misst die Rekonstruktionsverzerrung des dekodierten Bildes. Variiert man $\lambda$, erhält man die Rate-Distortion-Kurve des Codecs, indem man jeweils den Punkt mit der Steigung $-\lambda$ auswählt. In einer Folgearbeit verwenden :citet:`Balle.Minnen.Singh.ea.2018` eine zweite, kleinere latente Repräsentation, den *Hyperprior*, als Bedingung für das Entropiemodell. Er erfasst räumliche Unterschiede in der Bildkomplexität und senkt die Bitrate bei gleicher visueller Qualität erheblich.

<!--en:4071a501-->
## Schätzung der gegenseitigen Information

<!--en:9164835c-->
### Obere Schranken

<!--en:d22a5b6d-->
Jeder variationelle Schätzer aus :numref:`sec_mdl-mi-estimation` liefert eine *untere* Schranke für die gegenseitige Information. Das ist nützlich, wenn wir die gemeinsame Information maximieren wollen. Es liefert aber keinen Nachweis dafür, dass eine erlernte Repräsentation nur *wenig* Information über ein sensibles oder geschütztes Attribut enthält, das sie verwerfen soll. Für eine echte obere Schranke brauchen wir die bedingte Dichte $p(z \mid x)$, die ein stochastischer Encoder bereitstellt. Der variationelle Informationsflaschenhals (engl. variational information bottleneck, :numref:`subsec_mdl-variational-bottleneck`) liefert eine solche obere Schranke mit einer festen Näherung $r(z)$ für die Randverteilung. Die kontrastive obere Schranke für das logarithmierte Verhältnis (CLUB) :cite:`Cheng.Hao.Dai.ea.2020` kommt ganz ohne $r(z)$ aus. Sie mittelt $\log p(z \mid x)$ über zusammengehörige Paare eines Minibatchs und zieht davon den Mittelwert über falsch zugeordnete Paare ab. Ist $p(z \mid x)$ die wahre Dichte des Encoders, so ist CLUB im Erwartungswert eine obere Schranke. Für eine Repräsentation und ein geschütztes Attribut liegt jedoch keine analytische bedingte Dichte vor. An ihrer Stelle muss man ein neuronales Netz anpassen. Die obere Schranke gilt dann nur, wenn die angepasste bedingte Dichte im Sinne der KL-Divergenz näher an der wahren gemeinsamen Verteilung liegt als am Produkt der Randverteilungen. Um das zu überprüfen, bräuchten wir genau die Dichte, die uns von Anfang an fehlte.

<!--en:5c8575b1-->
### Verzerrung und Varianz der Schranken

<!--en:c8d02af2-->
Selbst mit dem exakt optimalen Kritiker besteht bei variationellen Schranken ein ausgeprägter Zielkonflikt. NWJ ist erwartungstreu, hat aber eine hohe Varianz. InfoNCE hat eine geringe Varianz, ist aber durch $\log N$ nach oben beschränkt (:numref:`subsec_mdl-exact-critics`). :citet:`Poole.Oord.Alemi.ea.2019` führen einen einzigen Mischparameter ein, mit dem man stetig zwischen den beiden Schranken interpolieren und so zwischen Verzerrung und Varianz abwägen kann. Schwerer noch wiegt das Ergebnis von :citet:`Song.Ermon.2020b`: Das explosionsartige Wachstum der Varianz ist ein grundsätzliches Problem und keine Besonderheit einer einzelnen Zielfunktion. Sowohl bei der NWJ- als auch bei der Donsker-Varadhan-Schranke wächst die Varianz des Schätzers exponentiell mit der wahren gegenseitigen Information, selbst bei einem exakten Kritiker. Als Abhilfe schlagen die Autoren vor, das Dichteverhältnis des Kritikers zu begrenzen. So verringern sie die Varianz auf Kosten einer Verzerrung. Insgesamt verschieben alle verfügbaren Verfahren den Fehler nur zwischen Varianz und Verzerrung, ohne ihn zu beseitigen.

<!--en:6ab2338b-->
Da die wahre gegenseitige Information bei realen Daten selten bekannt ist, schlagen dieselben Autoren Konsistenzprüfungen vor, die ohne bekannten Vergleichswert auskommen. Ergänzen wir eine Variable um den Wert einer deterministischen Funktion dieser Variablen, sollte ein zuverlässiger Schätzer denselben Wert liefern. Ersetzen wir ein einzelnes Paar $(X, Y)$ durch zwei unabhängige Kopien dieses Paars, sollte sich der Schätzwert verdoppeln. Auf Bilddatensätzen bestehen die untersuchten Standardschätzer diese grundlegenden Tests nicht. Als Faustregel für die Praxis gilt: Vergleichen Sie einen angegebenen Schätzwert von mehreren Nat für die gegenseitige Information stets mit dem Logarithmus der Batchgröße. Vertrauen Sie einem Schätzer auf Ihren eigenen Daten erst, wenn er diese Konsistenzprüfungen auch dort besteht.

<!--en:161096d4-->
### Verrauschte Netze aus Sicht des Informationsflaschenhalses

<!--en:9cf578d6-->
Die Debatte über Kompression aus :numref:`subsec_mdl-compression-debate` begann mit dem Vorschlag von :citet:`Tishby.Zaslavsky.2015`, jede Schicht eines neuronalen Netzes zusammen mit der Grenzkurve des Informationsflaschenhalses darzustellen (:numref:`sec_mdl-information-bottleneck`). Die Kurve gibt das maximale $I(Y; Z)$ an, das sich bei gegebenem $I(X; Z)$ erreichen lässt. Wie wir dort gesehen haben, stößt die Idee auf ein mathematisches Problem. In einem deterministischen Netz mit kontinuierlichen Eingaben ist $I(X; Z)$ formal unendlich. Es gibt also keine endliche Kompressionskurve, deren Verlauf man verfolgen könnte. :citet:`Goldfeld.Berg.Greenewald.ea.2019` geben der Frage eine wohldefinierte Form, indem sie die versteckten Aktivierungen mit schwachem gaußschem Rauschen überlagern. Dadurch wird $I(X; Z)$ endlich. In diesem verrauschten Modell kann $I(X; Z)$ während des Trainings tatsächlich abnehmen. Die Autoren führen den Rückgang auf die Bildung geometrischer Gruppen zurück: Versteckte Repräsentationen aus derselben Klasse rücken näher zusammen. Nach ihrer Auffassung war es genau diese Gruppenbildung, die frühere Schätzungen der gegenseitigen Information an deterministischen Netzen durch Einteilung in Intervalle erfasst hatten.

<!--en:e8db4f80-->
## Allgemeine Literatur und Lehrmaterialien zur Informationstheorie

<!--en:f3e47eb0-->
**Bücher**

<!--en:5d2dbe48-->
- [Elements of Information Theory – Cover & Thomas](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X). Dieses klassische Nachschlagewerk für Fortgeschrittene leitet Entropie, KL-Divergenz (hier relative Entropie genannt), gegenseitige Information und die zugehörigen Kettenregeln aus den Grundprinzipien her. Es behandelt auch Quellenkodierung, Kanalkapazität und Rate-Distortion-Theorie und ergänzt :numref:`sec_mdl-information_theory`, :numref:`sec_mdl-coding-compression` und :numref:`sec_mdl-mutual-information`.

<!--en:1c9c53df-->
- [Information Theory, Inference, and Learning Algorithms – David MacKay](https://www.inference.org.uk/mackay/itila/). Dieses online frei verfügbare Buch verbindet durchgehend Informationstheorie und maschinelles Lernen. Die Kapitel über Symbolcodes und Stromcodes behandeln unter anderem die arithmetische Kodierung und ergänzen :numref:`sec_mdl-coding-compression`.

<!--en:1fa9ddc6-->
- [Information Theory: From Coding to Learning – Polyanskiy & Wu](https://www.cambridge.org/9781108832908). Dieses moderne Lehrbuch für Fortgeschrittene ist auf Statistik und maschinelles Lernen ausgerichtet. Die Kapitel über f-Divergenzen, variationelle Darstellungen und Datenverarbeitung ergänzen :numref:`sec_mdl-divergences-distances` und :numref:`sec_mdl-mutual-information`. Das [Skript zur MIT-Vorlesung 6.441](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/), aus dem das Buch hervorging, ist frei verfügbar.

<!--en:4940a6ad-->
- [Computational Optimal Transport – Peyré & Cuturi](https://optimaltransport.github.io/book/). Das Buch behandelt Wasserstein-Abstände, entropische Regularisierung und Sinkhorn-Iterationen für die Datenwissenschaft und ergänzt damit den Teil über Transport in :numref:`sec_mdl-test-functions-transport`. Der vollständige Text ist auch auf [arXiv](https://arxiv.org/abs/1803.00567) verfügbar.

<!--en:ae356d1b-->
- [The Minimum Description Length Principle – Peter Grünwald](https://mitpress.mit.edu/9780262072816/the-minimum-description-length-principle/). Das Buch entwickelt statistisches Lernen als Datenkompression mithilfe zweiteiliger und universeller Codes und vertieft damit :numref:`subsec_mdl-mdl`. Die Grundideen stellt auch ein [frei verfügbares Tutorial](https://arxiv.org/abs/math/0406077) vor.

<!--en:b4a79945-->
**Kurse, Tutorials und Software**

<!--en:912d6a39-->
- [Stanford EE376A Information Theory – Tsachy Weissman](https://web.stanford.edu/class/ee376a/). Zu diesem gut zugänglichen Kurs gibt es ein öffentliches, von Studierenden verfasstes [Vorlesungsskript](https://web.stanford.edu/class/ee376a/files/scribes/lecture_notes.pdf) zu Entropie, Präfixcodes und typischen Mengen. Es ergänzt :numref:`sec_mdl-information_theory` und :numref:`sec_mdl-coding-compression`.

<!--en:268070ac-->
- [Visual Information Theory – Christopher Olah](https://colah.github.io/posts/2015-09-Visual-Information/). Diese anschauliche visuelle Einführung führt von den Codewortlängen zur Entropie, zur Kreuzentropie und zur gegenseitigen Information. Sie ergänzt :numref:`sec_mdl-information_theory`, :numref:`sec_mdl-coding-compression` und :numref:`sec_mdl-mutual-information`.

<!--en:ce625855-->
- [From GAN to WGAN – Lilian Weng](https://lilianweng.github.io/posts/2017-08-20-gan/). Diese verständliche Einführung vergleicht KL-, Jensen-Shannon- und Wasserstein-Verluste für generative Modelle und ergänzt :numref:`sec_mdl-divergences-distances` und :numref:`sec_mdl-test-functions-transport`.

<!--en:bc259aa4-->
- [POT: Python Optimal Transport – Flamary et al.](https://pythonot.github.io/). Diese quelloffene Python-Bibliothek bietet Lösungsverfahren sowohl für den exakten optimalen Transport als auch für entropisch regularisierte Sinkhorn-Iterationen. Damit können Sie die Abstände aus :numref:`sec_mdl-optimal-transport` auf Ihren eigenen Datensätzen berechnen.

<!--en:13e6eab6-->
- [Generative Modeling by Estimating Gradients of the Data Distribution – Yang Song](https://yang-song.net/blog/2021/score/). Dieses gut zugängliche Tutorial erklärt, warum Score-Funktionen ohne Normierungskonstanten auskommen und wie Score-Matching sie an Daten anpasst (siehe :numref:`sec_mdl-scores`). Anschließend behandelt es Sampling und Diffusionsmodelle.

<!--en:d63f4e40-->
**Grundlegende Arbeiten**

<!--en:ab57dafd-->
- [Language Modeling Is Compression – Delétang et al.](https://arxiv.org/abs/2309.10668). Die Arbeit kombiniert einen arithmetischen Kodierer mit den Wahrscheinlichkeiten, die ein Sprachmodell für das nächste Token vorhersagt. Genau dieses Verfahren zeigt :numref:`subsec_mdl-lm-compressor`.

<!--en:6a1067d6-->
- [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization – Nowozin et al.](https://arxiv.org/abs/1606.00709). Die Arbeit zeigt, wie in :numref:`sec_mdl-f-gan-dual` hergeleitet, dass jede f-Divergenz über eine untere Schranke aus ihrer Fenchel-Konjugierten eine Zielfunktion für adversariales Training liefert.

<!--en:ebed8555-->
- [A Kernel Two-Sample Test – Gretton et al.](https://jmlr.org/papers/v13/gretton12a.html). Die Arbeit führt die Maximum Mean Discrepancy, ihren erwartungstreuen Zweistichprobenschätzer und den daraus folgenden Hypothesentest aus :numref:`sec_mdl-ipm-mmd` ein.

<!--en:ac6b4748-->
- [Wasserstein GAN – Arjovsky et al.](https://arxiv.org/abs/1701.07875). Die Arbeit ersetzt die Jensen-Shannon-Divergenz durch den Wasserstein-1-Abstand. So bleiben Gradienten auch dann aussagekräftig, wenn sich die Träger der Verteilungen nicht überlappen. Sie ergänzt :numref:`sec_mdl-optimal-transport`.

<!--en:d6f3fa17-->
- [Estimation of Non-Normalized Statistical Models by Score Matching – Aapo Hyvärinen](https://jmlr.org/papers/v6/hyvarinen05a.html). Die Arbeit passt unnormierte Modelle an, indem sie den Gradienten der Log-Dichte an den entsprechenden Gradienten der Daten angleicht. Den unbekannten Score der Daten eliminiert sie durch partielle Integration. Das sind die beiden entscheidenden Schritte in :numref:`sec_mdl-scores`.

<!--en:58989f40-->
- [A Kernelized Stein Discrepancy for Goodness-of-fit Tests – Liu, Lee & Jordan](https://proceedings.mlr.press/v48/liub16.html). Die Arbeit prüft durch paarweise Auswertung eines Kernels, ob eine Stichprobe aus einem Modell stammt, von dem nur der Score bekannt ist. Dieses Verfahren behandelt :numref:`sec_mdl-ksd`.

<!--en:9c21094d-->
- [The Information Bottleneck Method – Tishby, Pereira & Bialek](https://arxiv.org/abs/physics/0004057). Die Arbeit formuliert die Frage, wie man $X$ komprimiert und dabei Information über $Y$ erhält. Diesen Ansatz untersuchen wir in :numref:`sec_mdl-information-bottleneck`.

<!--en:a3fdafab-->
- [Estimating Mutual Information – Kraskov, Stögbauer & Grassberger](https://arxiv.org/abs/cond-mat/0305641). Die Arbeit schätzt die gegenseitige Information kontinuierlicher Variablen aus Abständen zu nächsten Nachbarn. Auf diesem Schätzer aus :numref:`subsec_mdl-ksg` beruht [`mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) in scikit-learn.

<!--en:9fed735f-->
- [On Variational Bounds of Mutual Information – Poole et al.](https://arxiv.org/abs/1905.06922). Die Arbeit fasst die Schranken von Barber und Agakov, Donsker und Varadhan sowie die NWJ- und InfoNCE-Schranken in einem gemeinsamen Rahmen zusammen. Sie untersucht den jeweiligen Zielkonflikt zwischen Verzerrung und Varianz und ergänzt damit :numref:`sec_mdl-mi-variational-bounds`.

<!--en:e06d1664-->
- [Representation Learning with Contrastive Predictive Coding – van den Oord et al.](https://arxiv.org/abs/1807.03748). Die Arbeit führt den InfoNCE-Verlust als untere Schranke für die gegenseitige Information beim selbstüberwachten Repräsentationslernen ein. Diesen Ansatz entwickeln wir in :numref:`sec_mdl-infonce`.

<!--en:e67f7d89-->
[Diskussion](https://d2l.discourse.group/t/36867)
