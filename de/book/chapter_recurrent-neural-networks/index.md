<!--en:37fbf7fc-->
# Sequenzmodelle

<!--en:1940ee51-->
Nach zwei Kapiteln über Bilder wenden wir uns zum Abschluss dieses Teils des Buches sequenziellen Daten zu. Alle bisherigen Modelle erwarten eine Eingabe mit fester Form, etwa einen Merkmalsvektor aus einer Tabelle oder ein Raster aus $28 \times 28$ Pixeln. Jedes Beispiel verarbeiten sie unabhängig von den anderen. Bei Sätzen, Tonaufnahmen, Videos oder Datenströmen aus Sensoren ist das anders. Ihre Länge variiert von Beispiel zu Beispiel, und die einzelnen Elemente hängen stark voneinander und von ihrer Reihenfolge ab. Betrachten wir den ersten Satz aus *The Time Machine* von H. G. Wells. Die Novelle dient uns im ganzen Kapitel als Datensatz. „The Time Traveller (for so it will be convenient to speak of him) was expounding a recondite matter to us.“ Um das letzte Wort zu erraten, müssen wir die neunzehn Wörter davor in ihrer Reihenfolge lesen. Wie kann ein einziges Modell diese Vorhersage treffen, wenn der vorausgehende Text beliebig lang sein kann?

<!--en:7115d391-->
Claude Shannon machte aus dieser Frage ein Ratespiel :cite:`Shannon.1951`. Er ließ Versuchspersonen eine Textpassage Zeichen für Zeichen erraten und schätzte so, wie gut sich gedrucktes Englisch vorhersagen lässt. Ein *Sprachmodell* (engl. language model) spielt dasselbe Spiel. Anhand der bisherigen Token weist es dem nächsten *Token* eine Wahrscheinlichkeit zu, ob es nun ein Zeichen, ein Subwort oder ein ganzes Wort ist. Es kann neuen Text erzeugen, indem es jedes gewählte Token als nächste Eingabe verwendet.

<!--en:82f17362-->
Mit jedem Schritt wächst die bereits gelesene Sequenz. Betrachtet ein Modell nur die letzten Token in einem Fenster fester Größe, geht alles davor verloren. Ein *rekurrentes neuronales Netz* (engl. recurrent neural network, RNN) führt dagegen einen *versteckten Zustand* (engl. hidden state) fester Größe mit. Er fasst die bisher gelesene Sequenz zusammen. Nach jedem Token aktualisiert das Netz ihn mit denselben Parametern. Moderne große Sprachmodelle ersetzen die Rekurrenz teils durch Attention (:numref:`chap_attention`), teils durch strukturierte Zustandsraummodelle (:numref:`chap_modern_rnn`). Auch dort gelten die grundlegenden Verfahren dieses Kapitels unverändert weiter. Dazu gehören die Subwort-Tokenisierung, die Bewertung anhand der Perplexität und der Bit pro Byte sowie das autoregressive Dekodieren.

<!--en:72abfdcf-->
Zunächst formulieren wir die autoregressive Vorhersage von Sequenzen. Wir untersuchen, warum wiederholte Vorhersagen über mehrere Schritte viel schneller an Genauigkeit verlieren als Vorhersagen für einen einzelnen Schritt (:numref:`sec_sequence`). Dann wandeln wir Rohtext mit Byte-Pair-Encoding auf Byteebene in Sequenzen von Subwort-Token um (:numref:`sec_text-sequence`). Wir entwickeln $n$-Gramm-Sprachmodelle auf der Grundlage von Häufigkeiten und führen zwei Bewertungsmaße ein, die Perplexität und die Anzahl der Bit pro Byte (:numref:`sec_language-model`). Anschließend formulieren wir Sprachmodelle auf der Basis rekurrenter neuronaler Netze und implementieren sie sowohl von Grund auf als auch mit High-Level-APIs (:numref:`sec_rnn`). Wir stabilisieren das Training durch Gradienten-Clipping und vergleichen die erlernten Repräsentationen mit $n$-Grammen nach Kneser–Ney. In :numref:`sec_bptt` leiten wir die Backpropagation durch die Zeit (engl. backpropagation through time, BPTT) her. Sie erklärt, warum Gradienten in einfachen RNNs verschwinden oder explodieren. In :numref:`sec_decoding` untersuchen wir die gierige Suche (engl. greedy search), die Beam-Search und Sampling-Verfahren mit abgeschnittener Verteilung zur Texterzeugung.

<!--en:a013cd3a-->
Sequenzmodelle und rekurrente neuronale Netze<br>**Autoregression · Subwort-Tokenisierung mit BPE · Sprachmodelle · Perplexität und Bit pro Byte · RNNs · BPTT · Dekodieren**

<!--en:15ff545e-->
::: {.slide title="Warum Feedforward-Modelle mit fester Eingabeform bei Sequenzen scheitern"}

<!--en:25d9aa8a-->
[Motivation und Kernproblem]{.kicker}

<!--en:25f13b36-->
- **Unbegrenzter kausaler Kontext:** Um den 20 Wörter langen ersten Satz aus *The Time Machine* von H. G. Wells zu vervollständigen, müssen wir die neunzehn vorangehenden Wörter in ihrer genauen zeitlichen Reihenfolge berücksichtigen.

<!--en:80079357-->
- **Kombinatorische Explosion der Kontexte:** Für 20 aufeinanderfolgende Token aus einem Vokabular von 1024 Token gibt es $1024^{20} \approx 10^{60}$ mögliche Kontexte. Selbst bei Folgen aus nur 3 Wörtern ($n = 3$) kommen $87\%$ der Tripel aus den zurückgehaltenen Daten im Training nie vor.

<!--en:c4c68f58-->
- **Kompression bei begrenztem Zustand:** Ein Sequenzmodell muss ein beliebig langes Präfix $x_{<t} = (x_1, \ldots, x_{t-1})$ auf eine Repräsentation fester Größe verdichten, bei konstanter Parameteranzahl und einem Rechenaufwand von $O(1)$ pro Schritt.

<!--en:281e28aa-->
::: {.slide title="Überblick über Kapitel §{num:chap_rnn}: Von Zeitreihen zum Dekodieren"}

<!--en:b514d0e1-->
[Aufbau der Architekturen]{.kicker}

<!--en:4701a887-->
- **Daten und Repräsentation (§§{sec_sequence}–§{num:sec_text-sequence}):** Wir formulieren die Autoregression mit der Kettenregel (Fenster von $m$ Schritten oder latenter Zustand $\mathbf{h}_t$). Wir zerlegen rohe UTF-8-Zeichenketten durch Vortokenisierung und Byte-Pair-Encoding (BPE) in Subwort-IDs.

<!--en:829fa6e4-->
- **Probabilistische Bewertung und RNNs (§§{sec_language-model}–§{num:sec_rnn}):** Wir bewerten $n$-Gramme nach Kneser–Ney unabhängig vom Tokenizer in Bit pro Byte ($\textrm{bpb}$). Dann ersetzen wir die Häufigkeitstabellen durch eine rekurrente $\tanh$-Zelle mit geteilten Gewichten und stabilisieren sie durch globales Gradienten-Clipping.

<!--en:ce722e62-->
- **Zuordnung von Fehlerbeiträgen und Dekodieren (§§{sec_bptt}–§{num:sec_decoding}):** Wir leiten die Backpropagation durch die Zeit (BPTT) her und erklären damit, warum Gradienten verschwinden oder explodieren. Wir vergleichen die Beam-Search mit Sampling aus abgeschnittenen Verteilungen ($\text{top-}k$, $\text{top-}p$, $\text{min-}p$).

<!--en:a94c4ee8-->
::: {.slide title="Quantitative Benchmarks für die gesamte Verarbeitungskette am Beispiel der Novelle von Wells"}

<!--en:4ed205ed-->
[Messergebnisse]{.kicker}

<!--en:de6cf1d0-->
- **Drift bei wiederholter Vorhersage und Subwort-Kompression (§§{sec_sequence}–§{num:sec_text-sequence}):** Ein lineares AR($4$)-Modell erzielt auf einer verrauschten Sinuskurve einen mittleren quadratischen Fehler von $0.05$ für 1 Schritt. Über 64 Schritte steigt er auf das Fünffache ($5\times$), also auf $0.25$. BPE-1024 komprimiert *The Time Machine* auf $2.72$ Byte pro Token, ohne ein einziges Token außerhalb des Vokabulars.

<!--en:ae1ca8a4-->
- **Häufigkeitsbasierte und neuronale Sprachmodelle (§§{sec_language-model}–§{num:sec_rnn}):** Kneser–Ney senkt die Perplexität von Worttrigrammen ($2{,}481 \to 300$) und erzielt mit BPE-1024 auf kleingeschriebenem Text $1.90$ bpb. Auf unverändertem Text erreicht `RNNLMScratch` (Zustand von $512$ Byte, $89\%$ der Parameter in den Vokabularschichten) mit Gradienten-Clipping $2.12$ bpb und liegt damit praktisch gleichauf mit einem Kneser–Ney-Trigrammmodell ($2.09$).

<!--en:fc45e067-->
- **Reichweite des Gradienten und Texterzeugung (§§{sec_bptt}–§{num:sec_decoding}):** Trotz $\rho(\mathbf{W}_{\textrm{hh}}) \approx 1.80$ lässt die Kontraktion durch $\tanh'$ ($\approx 0.27$) die Gradienten auf unter $1\%$ sinken, und das innerhalb von $8$ Schritten. Mit Sampling aus abgeschnittenen Verteilungen erreicht `distinct-3` wieder $\approx 0.99$.

<!--en:d709b8c3-->
::: {.slide title="Verständnisfragen zum Entwurf eines End-to-End-Sprachmodells"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:8c949491-->
Prüfen Sie vor dem Einstieg in §{sec_sequence}, wie die Komponenten der Pipeline zur Sequenzmodellierung zusammenspielen:

<!--en:1974da16-->
1. **Tokenizer und Perplexität:** Modell A verwendet ein Vokabular von 27 Zeichen und erreicht auf den Testdaten eine Perplexität von $\textrm{PPL} = 6.5$. Modell B verwendet ein BPE-Vokabular mit 1024 Token ($2.78$ Byte pro Token) und erreicht $\textrm{PPL} = 60.0$. Welches Modell komprimiert den Testtext auf weniger Bit pro Byte?

<!--en:d54e0235-->
2. **Zustandsspeicher und Sequenzlänge:** Wie viele Byte Speicher für den versteckten Zustand benötigt ein einschichtiges `float32`-RNN mit $h = 128$ Einheiten für ein Fenster von 32 Token und wie viele für ein Buch mit 1 000 000 Token?

<!--en:37fb2086-->
3. **Unterschied zwischen Training und Inferenz:** Warum kann ein autoregressives Modell beim Training mit Teacher-Forcing einen niedrigen Kreuzentropieverlust für 1 Schritt erreichen und bei der freien Texterzeugung dennoch in Wiederholungen verfallen oder vom Thema abkommen?

<!--en:557b98b6-->
**Antworten:**

<!--en:a024acd9-->
1. **Modell B komprimiert besser:** Das Zeichenmodell A (1 Byte pro Zeichen) benötigt $\log_2(6.5) / 1.0 \approx 2.70$ Bit pro Byte, das BPE-Modell B nur $\log_2(60.0) / 2.78 \approx 5.91 / 2.78 \approx 2.13$ Bit pro Byte. Bei unterschiedlich feiner Tokenisierung lassen sich die Perplexitätswerte nicht direkt vergleichen.

<!--en:d70b9865-->
2. **In beiden Fällen $512$ Byte:** Der Speicherbedarf beträgt $128 \times 4\text{ bytes} = 512\text{ bytes}$ pro Sequenz, unabhängig von ihrer Länge $T$. Bei jedem Schritt überschreibt $\mathbf{h}_t$ den Zustand $\mathbf{h}_{t-1}$ am selben Speicherort.

<!--en:41617665-->
3. **Exposure-Bias und Fehlerfortpflanzung bei wiederholter Vorhersage:** Beim Teacher-Forcing erhält das Modell stets die Präfixe $x_{<t}$ aus der Ground Truth. Bei der Texterzeugung gibt es die selbst gezogenen Token $\hat{x}_t$ als Eingabe zurück. Schon ein einzelnes sehr unwahrscheinliches Token am Rand der Verteilung oder ein deterministischer Zyklus führt den Zustand $\mathbf{h}_t$ aus der Trainingsverteilung heraus.

<!--en:e6e88fe1-->
::: {.slide title="Die wichtigsten Schritte der Sequenzmodellierung"}

<!--en:1038e504-->
[Zusammenfassung des Kapitels]{.kicker}

<!--en:91bc384f-->
- **Universelle Subwörter und vergleichbare Bewertung (§§{sec_text-sequence}–§{num:sec_language-model}):** BPE auf Byteebene mit Vortokenisierung kommt ohne `<unk>`-Token aus und reserviert IDs oberhalb von $|\mathcal{V}_{\textrm{BPE}}|$ für spezielle Token. Bit pro Byte ($\textrm{bpb}$) liefern für jede Tokenisierung denselben Maßstab für die verlustfreie Kompression.

<!--en:8f4729de-->
- **Rekurrenz mit konstantem Speicherbedarf und BPTT (§§{sec_rnn}–§{num:sec_bptt}):** Mit denselben Gewichten aktualisieren wir den Zustand $\mathbf{H}_t \in \mathbb{R}^{b \times h}$ fester Größe in $O(1)$ pro Schritt. Verkürzte BPTT begrenzt den Speicherbedarf auf $O(T_0 bh)$. Das Begrenzen der globalen Norm verhindert, dass Gradientenschritte explodieren.

<!--en:c62b15e0-->
- **Kontrolliertes autoregressives Dekodieren (§{sec_decoding}):** Bei Aufgaben mit Nebenbedingungen verwenden wir die längennormierte Beam-Search. Bei der freien Texterzeugung verhindert die Kombination aus Softmax-Temperatur und Abschneiden des Verteilungsrands mit $\text{top-}p / \text{min-}p$ Wiederholungsschleifen.
