<!--en:e79c11de-->
# Zustandsraummodelle

<!--en:392d9d59-->
Die rekurrenten Netze aus :numref:`chap_rnn` lesen einen Text Token für Token und halten das Gelesene in einem *Zustand* fester Größe fest. Das millionste Token zu erzeugen kostet daher genauso viel wie das zehnte. Die Transformer aus :numref:`chap_transformers` behalten dagegen jedes Token. Um eine Frage zu einem langen Bericht zu beantworten, speichern sie für jedes Token des Berichts einen Schlüssel und einen Wert (:numref:`sec_kv-cache`). Mit der Kontextlänge wachsen also sowohl der Speicherbedarf als auch der Rechenaufwand pro Token. Doch in :numref:`sec_gpt` fiel ein einfaches rekurrentes Netz von der Größe unseres GPT schon nach den ersten wenigen Byte Kontext zurück und holte den Rückstand nie auf. Kann ein besseres rekurrentes Netz diese Lücke schließen? Ab welcher Kontextlänge fallen die steigenden Kosten der Attention ins Gewicht?

<!--en:3f113345-->
Dem stehen zwei Hindernisse im Weg. Klassische rekurrente Netze aktualisieren ihren Zustand nichtlinear. Deshalb müssen die Schritte beim Training nacheinander ablaufen, und ein Großteil der GPU bleibt ungenutzt. Außerdem muss ein Zustand fester Größe Informationen vergessen. Das Netz muss also auswählen, was es behält. Drei Ansätze liefern Antworten. Der erste nutzt Gates aus den 1990er-Jahren :cite:`Hochreiter.Schmidhuber.1997,Gers.Schmidhuber.Cummins.2000`. Der zweite greift auf lineare dynamische Systeme aus der Regelungstechnik zurück :cite:`Kalman.1960`. Als trainierbare Schichten fanden sie Eingang ins Deep Learning :cite:`Gu.Dao.Ermon.ea.2020,Gu.Goel.Re.2022`. Der dritte Ansatz verwendet Attention ohne Softmax-Funktion :cite:`Katharopoulos.Vyas.Pappas.ea.2020`. Die drei Ansätze kommen in rekurrenten Schichten zusammen, die sich parallel trainieren lassen und eine Form der linearen Attention berechnen. Jede rekurrente Schicht in diesem Kapitel hat einen Zustand, eine *Aktualisierungsregel* (engl. update), die jedes Token in den Zustand einbezieht, und eine *Auslesefunktion* (engl. readout), die daraus die Ausgabe berechnet. In der Regelungstechnik heißt ein solches System *Zustandsraummodell* (engl. state space model). Im Kapiteltitel verwenden wir den Begriff in diesem weiten Sinn. In :numref:`sec_ssm` bezeichnet er im engeren Sinn ein lineares dynamisches System in kontinuierlicher Zeit.

<!--en:f74c7bba-->
Wir beginnen mit Gates, mit denen ein Netz lernt, was es behalten soll. Dann betrachten wir die Encoder-Decoder-Architektur von 2014. Ihr einziger Zustandsvektor bildet einen Flaschenhals, dessen Wirkung wir messen können. Drei Änderungen sorgen dafür, dass das Netz seinen Zustand schnell und selektiv aktualisiert. Erstens lässt sich eine lineare Aktualisierungsregel für alle Positionen parallel auswerten. Ihre Laufzeit vergleichen wir mit der von Attention (:numref:`subsec_linear-recurrence-cost`). Zweitens diskretisieren wir Abklingvorgänge, die in kontinuierlicher Zeit auf mehreren Zeitskalen ablaufen. Drittens bestimmt dank eingabeabhängiger Schrittweiten jedes Token, wie viel es schreibt und wie viel es vergisst. Erweitern wir den Zustand von einem Vektor zu einer Matrix, erhalten wir einen assoziativen Speicher für Schlüssel-Wert-Paare. Seine Kapazität können wir berechnen und seine Schreibregel mit der Delta-Regel präziser machen. Fügen wir schließlich einige Schichten mit vollständiger Attention zwischen die rekurrenten Schichten ein, wird ein exakter Abruf wieder möglich. Kein Zustand fester Größe allein leistet das. Mehrere produktiv eingesetzte Sprachmodelle nutzen diese Hybridarchitektur :cite:`Lieber.Lenz.Bata.ea.2024,Qwen.Team.2025`.

<!--en:dfa95979-->
Bei der ersten Lektüre können Sie :numref:`sec_seq2seq`, die Herleitung in kontinuierlicher Zeit in :numref:`sec_ssm` und die blockweise arbeitenden Algorithmen aus :numref:`sec_matrix-state` und :numref:`sec_deltanet` überspringen. Wir prüfen die Mechanismen mit kleinen synthetischen Aufgaben, die jeweils eine einzelne Gedächtnisleistung testen. Mit dem bytebasierten Textbenchmark aus :numref:`sec_gpt` prüfen wir, wie gut jeder Sequenzmischer Sprache modelliert.

<!--en:23491531-->
Zustandsraummodelle und moderne rekurrente Architekturen<br>**Rekurrenzen mit Gates · parallele assoziative Scans · strukturierte und selektive SSMs · Matrixspeicher · Delta-Regel · Hybridmodelle**

<!--en:6c404a39-->
::: {.slide title="Das Dilemma bei langem Kontext: KV-Cache oder Zustand fester Größe?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:083abb07-->
- **Kausale Attention (KV-Cache der Größe $O(T)$):** Benötigt 4 KB pro Token ($8$ KV-Köpfe $\times 128$ fp16-Werte). Der Speicherbedarf wächst auf **$512\text{ MB}$ pro Schicht** bei $T = 128\text{K}$. Die Berechnung der Scores kostet $O(T^2 d)$ FLOPs.

<!--en:c82e955b-->
- **Rekurrenter Zustand (Speicherbedarf $O(1)$):** Verdichtet die Vorgeschichte auf konstant **$2\text{ MB}$** (Mamba-2 bei $d=4{,}096$). Bei $T = 512$ Token ist der KV-Cache ebenso groß.

<!--en:89f54574-->
- **Leitfrage:** Lässt sich ein Modell mit einem Zustand der Größe $O(1)$ auf GPUs parallel trainieren und ruft es Informationen ebenso zuverlässig ab wie ein Transformer?

<!--en:3e921c00-->
::: {.slide title="Zwei Hindernisse bei der Modernisierung rekurrenter Netze"}

<!--en:c93f079a-->
[Zentrale Herausforderungen]{.kicker}

<!--en:7a0f3109-->
- **Hindernis 1 – sequenzielle Tiefe $O(T)$:** Die Nichtlinearität $\tanh(\mathbf{W}_{\textrm{hh}}\mathbf{h}_{t-1})$ verhindert GPU-Parallelität. Affine Übergänge (§{sec_linear-recurrence}–§{num:sec_mamba}) benötigen nur $O(\log_2 T)$ Runden eines parallelen Scans.

<!--en:d6715935-->
- **Hindernis 2 – Informationsverlust im Vektor fester Größe:** In einem eindimensionalen Array $\mathbf{h}_t \in \mathbb{R}^d$ verschwimmen gespeicherte Fakten. Schrittweiten $\Delta_t$ für mehrere Zeitskalen, Matrixzustände $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ und die Delta-Regel (§{sec_ssm}–§{num:sec_deltanet}) ermöglichen die Adressierung von Werten über Schlüssel.

<!--en:3508a84f-->
- **Gemeinsame Sicht im Zustandsraum:** Jeder Sequenzmischer hat einen **Zustand**, eine **Aktualisierungsregel zum Schreiben** und eine **Auslesefunktion**.

<!--en:ae20dcc7-->
::: {.slide title="Entwicklung der Architekturen in Kapitel §{num:chap_modern_rnn}"}

<!--en:99e0030a-->
[Kapitelüberblick]{.kicker}

<!--en:11633585-->
- **Von Gates zu linearen Scans (§{sec_lstm}–§{num:sec_mamba}):** Von sequenziellen Gates in LSTM und GRU und dem Flaschenhals in Seq2Seq-Modellen zu parallelen Scans in `minGRU`, mehreren Zeitskalen in `S4D` und dem selektiven Modell `Mamba`.

<!--en:2157051e-->
- **Vom Vektor zum Matrixspeicher (§{sec_matrix-state}–§{num:sec_deltanet}):** Äußere Produkte erweitern den Zustand zu einer Matrix für Schlüssel-Wert-Paare (`GLA`). Auslesen vor jedem Schreibvorgang ermöglicht eine gezielte Online-Fehlerkorrektur (`GatedDeltaNet`).

<!--en:be69c8fa-->
- **Hybridmodelle mit einzelnen Attention-Schichten (§{sec_hybrids}):** Auf je 3 rekurrente Schichten kommt 1 Schicht mit vollständiger Attention. Damit entfällt die durch die Welch-Schranke bedingte Begrenzung des Abrufs bei überlappenden Schlüsseln.

<!--en:08af958d-->
::: {.slide title="Synthetische Belastungstests und Sprachbenchmark"}

<!--en:48261854-->
[Empirische Methoden]{.kicker}

<!--en:37e19683-->
- **Fünf gezielte Tests der Speicherfähigkeit:** verzögerter Abruf (Halbwertszeit), Umkehrung einer Zeichenfolge (Bit pro Gleitkommazahl), selektives Kopieren (tokenabhängiges $\Delta_t$), assoziativer Abruf (Ranggrenze nach Welch) und Neuzuordnung von Schlüsseln (Löschen mit der Delta-Regel).

<!--en:d775306c-->
- **Einheitlicher Sprachbenchmark (`SeqLM`):** Wir testen jeden Sequenzmischer in derselben Folge von 4 Residualblöcken mit Pre-Norm auf den Bytefolgen von WikiText-103.

<!--en:5d26d31c-->
::: {.slide title="Lesewege durch Kapitel §{num:chap_modern_rnn}"}

<!--en:123f19cf-->
[Lesehinweise]{.kicker}

<!--en:778f6c8d-->
- **Die wichtigsten Architekturen:** Folgen Sie **§{sec_lstm} (Gates)** $\to$ **§{sec_linear-recurrence} (lineare Rekurrenzen und parallele Scans)** $\to$ **§{sec_mamba} (selektive SSMs / Mamba)** $\to$ **§{sec_matrix-state}–§{num:sec_deltanet} (Matrixspeicher und DeltaNet)** $\to$ **§{sec_hybrids} (Hybridarchitekturen)**.

<!--en:09476881-->
- **Historische und informationstheoretische Grundlagen (§{sec_seq2seq}):** An der Encoder-Decoder-Architektur von 2014 messen wir mit Übersetzungsmetriken (`chrF`, `BLEU`), wie schnell ein einzelner Zustandsvektor bei wachsender Eingabelänge seine Kapazitätsgrenze erreicht.

<!--en:303c4c1a-->
- **Mathematische Herleitungen für die zweite Lektüre:** Diskretisierung eines Systems in kontinuierlicher Zeit mit einem Halteglied nullter Ordnung in **§{sec_ssm}** und blockweise Algorithmen für blockweise semiseparable Matrizen in **§{sec_matrix-state}–§{num:sec_deltanet}**.

<!--en:8560637e-->
- **Kontrollierte Vergleiche im Code:** Wir setzen jeden Sequenzmischer in dasselbe residuale `SeqLM`-Basisnetz mit Pre-Norm und LeCun-Initialisierung ein. Unterschiede gehen damit allein auf den Sequenzmischer zurück.

<!--en:8f6fc707-->
**Als Nächstes (§{sec_lstm} Rekurrente Netze mit Gates):** Wie Sigmoid-Gates für jedes Element einen eigenen Abklingpfad lernen und warum die Initialisierung des Gate-Bias darüber entscheidet, ob das Netz beim Training ein Langzeitgedächtnis entwickelt.
