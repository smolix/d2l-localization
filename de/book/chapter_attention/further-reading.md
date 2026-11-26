<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:7ca3a29b-->
In den vorangegangenen Abschnitten haben wir für jeden Bestandteil einer Schicht eine typische Lösung entwickelt. Dazu gehören das Nachschlagen mit der Softmax-Funktion (:numref:`sec_queries-keys-values`), skalierte Skalarprodukt-Scores und Maskierung (:numref:`sec_attention-scoring-functions`), Self-Attention mit mehreren Köpfen oder gruppierten Abfragen (:numref:`sec_multihead-attention`), Positionskodierungen (:numref:`sec_positional-information`), die kachelweise Online-Softmax-Berechnung (:numref:`sec_attention-at-scale`), Attention mit gleitendem Fenster und lineare Attention (:numref:`sec_windowed-linear-attention`) sowie Schaltkreise zur mechanistischen Erklärung (engl. circuits) (:numref:`sec_what-attention-computes`). Dieser Abschnitt gibt einen Überblick über eng verwandte Varianten, architektonische Verfeinerungen in modernen Modellen mit offenen Gewichten und die Primärliteratur dazu.

<!--en:f9243964-->
## Nachschlagen und Normierung

<!--en:eacf96f0-->
### Attention als assoziativer Speicher

<!--en:7ac2ec98-->
:citet:`Ramsauer.Schafl.Lehner.ea.2021` stellen eine Verbindung zwischen Softmax-Attention und klassischen assoziativen Speichern her. Sie zeigen, dass ein Schritt der Softmax-Attention genau einem Update eines kontinuierlichen *modernen Hopfield-Netzes* entspricht. Seine gespeicherten Muster sind die Schlüssel-Wert-Zeilen. Die inverse Temperatur beträgt bei der Kernel-Regression $1/\sigma^2$ und bei der Skalarprodukt-Attention $1/\sqrt{d}$. Sie bestimmt, welche Fixpunkte (engl. fixed points) auftreten. Bei hoher inverser Temperatur sind es einzelne gespeicherte Muster, bei niedriger metastabile Mittelwerte über mehrere Muster. Aus dieser Sicht beobachten die Autoren, dass frühe Schichten vortrainierter Encoder wie BERT vorwiegend über viele Muster mitteln. Tiefere Schichten rufen dagegen gezielt einzelne Muster ab. Adaptive Kernel-Bandbreiten :cite:`Silverman86` entsprechen Softmax-Temperaturen, die von der Abfrage oder vom Kopf abhängen.

<!--en:43012c03-->
### Monotones Alignment

<!--en:f1a68df5-->
Ein Jahr bevor :citet:`Bahdanau.Cho.Bengio.2014` ein inhaltsbasiertes Alignment ohne Nebenbedingungen für die maschinelle Übersetzung einführten, stellte :citet:`Graves.2013` einen differenzierbaren *monotonen*, positionsbasierten Attention-Mechanismus zur Handschriftsynthese vor. Der Mechanismus parametrisiert eine Mischung aus Gauß-Fenstern, deren mittlere Positionen mit jedem Stiftstrich monoton vorrücken. Dieser induktive Bias eignet sich für gesprochene Sprache und Handschrift (:cite:`rabiner1993fundamentals`). Für Übersetzungen, bei denen sich die Reihenfolge von Wörtern umkehrt, ist er jedoch zu restriktiv (:numref:`subsec_learned-alignment`).

<!--en:eeca9527-->
### Gewichte, deren Summe von eins abweichen darf

<!--en:2a2284eb-->
Da die Gewichte jeder Softmax-Zeile in der Summe eins ergeben müssen, legt eine Abfrage ohne relevante Schlüssel ihre Wahrscheinlichkeitsmasse oft auf das erste Token. Es ist der *Attention-Sink* aus :numref:`subsec_cross-attention-alignment` und :numref:`sec_what-attention-computes` (siehe auch :numref:`subsec_kv-cache-sinks`). :citet:`Sun.Chen.Kolter.ea.2024` bringen Attention-Sinks mit extremen Ausreißern unter den Aktivierungen im Residualstrom (engl. residual stream) in Verbindung. Diese Ausreißer wirken als implizite Bias-Terme. :citet:`Barbero.Arroyo.Gu.ea.2025` argumentieren, dass Attention-Sinks in tiefen Transformern mit langem Kontext den Kollaps der Repräsentationen verhindern, weil sie ein Ziel ohne Wirkung bieten.

<!--en:ce9a76df-->
Mehrere neuere Ansätze lockern die Nebenbedingung, dass die Summe eins ergeben muss. So können Köpfe inaktiv bleiben, ohne ein Eingabetoken als Attention-Sink zu benötigen.

<!--en:09c2859d-->
* **Sigmoid-Attention** :cite:`Ramapuram.Danieli.Dhekane.ea.2025` ersetzt die zeilenweise Softmax-Funktion durch eine elementweise logistische Sigmoidfunktion mit einer Verschiebung um $-\log n$. Die Gewichte liegen dadurch bei der Initialisierung nahe bei $1/(n + 1)$.

<!--en:749a28e7-->
* **Attention mit Gatter** (engl. gated attention) :cite:`Qiu.Wang.Zheng.ea.2025` kommt in Qwen3-Next :cite:`Qwen.Team.2025` zum Einsatz. Sie multipliziert die Softmax-Ausgabe jedes Kopfes mit dem Wert eines eingabeabhängigen Sigmoid-Gatters.

<!--en:20df9920-->
* Bei **lernbaren Sink-Logits**, wie sie `gpt-oss` :cite:`OpenAI.2025` verwendet, geht ein zusätzlicher trainierbarer Score in den Nenner der Softmax-Funktion ein. Er gehört zu einem virtuellen Schlüssel ohne zugehörigen Wertevektor.

<!--en:8a6f1e90-->
* **Differentielle Attention** :cite:`Ye.Dong.Xia.ea.2025` bildet pro Kopf die Differenz zweier Softmax-Verteilungen. Dadurch hebt sich das gemeinsame Hintergrundgewicht auf irrelevanten Schlüsseln auf.

<!--en:c599fa90-->
## Self-Attention und Attention-Köpfe

<!--en:56792267-->
### Ursprünge und Redundanz der Köpfe

<!--en:98988fa9-->
Self-Attention entwickelte sich aus Rekurrenzen mit *Intra-Attention* :cite:`Cheng.Dong.Lapata.2016`, aus zerlegbarer Attention ohne Rekurrenz für die natürlichsprachliche Inferenz :cite:`Parikh.Tackstrom.Das.ea.2016` und aus mehrzeiligen Satzeinbettungen mit Straftermen für mangelnde Vielfalt :cite:`Lin.Feng.Santos.ea.2017`. :citet:`Vaswani.Shazeer.Parmar.ea.2017` vereinten diese Ansätze im Transformer zur Multi-Head-Self-Attention mit Projektionen.

<!--en:c8e64bf8-->
Mehrere Köpfe sind nötig, um den Flaschenhals einer einzigen Mischung zu überwinden (:numref:`subsec_one-mixture-bottleneck`). Dennoch erweisen sich bei trainierten Modellen im Test oft viele Köpfe als redundant. :citet:`Michel.Levy.Neubig.2019` und :citet:`Voita.Talbot.Moiseev.ea.2019` zeigten, dass sich ein großer Teil der Köpfe in BERT und in Modellen für die neuronale maschinelle Übersetzung (NMT) nachträglich entfernen lässt. Die Güte sinkt dabei nur geringfügig. Übrig bleibt ein Kern von Köpfen, die auf Positionen, Syntax und seltene Wörter spezialisiert sind. Moderne Architekturen sparen durch Grouped-Query-Attention schon vor dem Training Parameter und Cache-Speicher (:numref:`subsec_kv-head-sharing`, :numref:`sec_kv-cache-reduction`).

<!--en:d5ac1b26-->
## Positionen

<!--en:1d6cd3d9-->
### Gelernte relative Positionen

<!--en:b7c78f35-->
Vor RoPE gab es verschiedene Möglichkeiten, den relativen Versatz einzubeziehen. Ein Ansatz lernte für jeden Versatz eine eigene Schlüsseleinbettung :cite:`shaw2018self`. Der Music Transformer nutzte relative Logits mit zeilenweise verschobenen Indizes :cite:`huang2018music`. T5 verwendete skalare Bias-Terme für logarithmisch eingeteilte Abstandsklassen :cite:`raffel2020exploring`. ALiBi ersetzte die gelernten Werte der Abstandsklassen durch feste lineare Steigungen. Bei RoPE sorgen parameterfreie Drehungen der Koordinaten dafür, dass der Einfluss der Positionen auf den Score ausschließlich von ihrem relativen Versatz abhängt.

<!--en:fe39c899-->
### RoPE in trainierten Modellen

<!--en:ac7d606f-->
Für längere Kontexte haben Modelle mit offenen Gewichten die RoPE-Basisfrequenz $b$ schrittweise erhöht. Sie stieg von $b = 10^4$ in Llama 2 auf $5 \times 10^5$ in Llama 2 Long :cite:`Xiong.Liu.Molybog.ea.2024` und Llama 3 :cite:`Grattafiori.Dubey.Jauhri.ea.2024`. In den globalen Schichten von Gemma 3 beträgt sie $10^6$ :cite:`Gemma.Team.2025`. Mit größerem $b$ schwingen mehr Koordinatenpaare so langsam, dass ihre Wellenlänge das Trainingsfenster übersteigt (:numref:`subsec_pos-context-extension`). :citet:`Barbero.Vitvitskyi.Perivolaropoulos.ea.2025` zeigen, dass trainierte Modelle hochfrequente RoPE-Paare für positionsabhängige Köpfe nutzen, etwa für Köpfe, die auf das vorherige Token zugreifen. Niederfrequente Paare dienen als nahezu positionsinvariante semantische Kanäle. Darauf beruht *p-RoPE*. Es setzt die niedrigsten Frequenzen ganz auf null, damit die semantischen Kanäle selbst bei extremen Abständen nicht driften. Neben YaRN :cite:`Peng.Quesnelle.Fan.ea.2024` ermöglicht auch *LongRoPE* :cite:`Ding.Zhang.Zhang.ea.2024` eine Kontexterweiterung nach dem Training. Eine evolutionäre Suche bestimmt für jede Dimension einen eigenen Skalierungsfaktor und erweitert so das Kontextfenster auf bis zu 2 Millionen Token.

<!--en:7bbb09f7-->
### Generalisierung auf andere Sequenzlängen und Nutzung langer Kontexte

<!--en:0b71a641-->
:citet:`Chi.Fan.Rudnicky.ea.2023` untersuchen das effektive rezeptive Feld von ALiBi. Ihre Analyse bestätigt unseren Befund aus :numref:`subsec_pos-experiment`. Bleibt die Perplexität bei der Extrapolation konstant, spiegelt das die Beschränkung auf lokale Fenster wider und belegt keinen Abruf über große Entfernungen. Für kausale Modelle ohne Positionskodierung (NoPE, :cite:`Haviv.Ram.Press.ea.2022`) zeigen :citet:`Wang.Ji.Wu.ea.2024`, dass die Entropie der Attention mit der Sequenzlänge wächst. Eine Temperaturskalierung pro Kopf wirkt diesem Anstieg entgegen. Llama 4 nutzt diesen Mechanismus bei der Inferenz in seinen globalen NoPE-Schichten :cite:`Meta.2025`. Auch eine insgesamt niedrige Perplexität auf langen Dokumenten garantiert keinen zuverlässigen Abruf über das gesamte Fenster. :citet:`Liu.Lin.Hewitt.ea.2024` zeigen ein Phänomen, das sie als *„lost in the middle“* bezeichnen. Die Genauigkeit sinkt deutlich, wenn die relevanten Fakten in der Mitte eines langen Kontexts stehen und nicht in der Nähe des Anfangs oder des Endes (siehe :numref:`sec_lm_long_context`).

<!--en:a4651e26-->
## Kosten

<!--en:b1ef82f1-->
### Frühe Näherungsverfahren mit linearer Komplexität

<!--en:8871cfcf-->
Vor FlashAttention näherten zahlreiche Architekturen die $n \times n$-Attention-Matrix an. *Performer* :cite:`Choromanski.Likhosherstov.Dohan.ea.2021` nutzte positive orthogonale zufällige Merkmale, um den Exponentialkernel anzunähern. *Linformer* :cite:`Wang.Li.Khabsa.ea.2020` verwendete Projektionen der Sequenz mit niedrigem Rang. *Reformer* :cite:`Kitaev.Kaiser.Levskaya.2020` nutzte lokalitätssensitives Hashing. *Longformer* :cite:`beltagy2020longformer` und *Sparse Transformer* :cite:`child2019generating` kombinierten lokale Fenster mit Token in regelmäßigen Abständen oder mit globalen Token. Einen Überblick geben :citet:`Tay.Dehghani.Bahri.ea.2020`.

<!--en:d25e4a5d-->
### Fenster und inhaltsabhängige Sparsität in veröffentlichten Modellen

<!--en:b8f5bee0-->
Mistral 7B :cite:`Jiang.Sablayrolles.Mensch.ea.2023` nutzte in jeder Schicht ein gleitendes Fenster. Eine spätere Version verzichtete darauf :cite:`Mistral.AI.2023`. Nachfolgende Architekturen kombinieren lokale Fenster in regelmäßigen Abständen mit globalen Schichten. Das Verhältnis von lokalen zu globalen Schichten beträgt 1:1 in Gemma 2 :cite:`Gemma.Team.2024`, 5:1 in Gemma 3 ($w = 1{,}024$, :cite:`Gemma.Team.2025`) und 1:1 bei $w = 128$ in `gpt-oss` :cite:`OpenAI.2025`. Llama 4 kombiniert blockweise arbeitende lokale Schichten mit globalen NoPE-Schichten :cite:`Meta.2025`.

<!--en:c64b425b-->
Neben festen Positionsfenstern kommt beim Training und beim Dekodieren mit langen Kontexten inzwischen auch *inhaltsbasierte Sparse Attention* zum Einsatz, die auf die Hardware abgestimmt ist (:numref:`sec_lm_sparse_attention`). Dazu gehören *Native Sparse Attention* (NSA, :cite:`Yuan.Gao.Dai.ea.2025`), *Mixture of Block Attention* (MoBA, :cite:`Lu.Jiang.Liu.ea.2025`) und *DeepSeek Sparse Attention* :cite:`Liu.Mei.Lin.ea.2025`.

<!--en:97c18b6a-->
### FlashAttention-2, FlashAttention-3 und FlexAttention

<!--en:dac281b1-->
Aufbauend auf FlashAttention :cite:`Dao.Fu.Ermon.ea.2022` parallelisiert *FlashAttention-2* :cite:`Dao.2024` die Berechnung entlang der Abfragesequenz und normiert erst am Ende der Schleife über die Schlüssel. Auf Hopper-GPUs überlappt *FlashAttention-3* :cite:`Shah.Bikshandi.Zhang.ea.2024` asynchrone Matrizenmultiplikationen auf Tensor Cores mit der Berechnung der Softmax-Exponentialfunktion auf eigens dafür vorgesehenen Warps. Inkohärente orthogonale Vorrotationen ermöglichen dabei die Verwendung von FP8. *FlexAttention* :cite:`Dong.Feng.Guessous.ea.2025` kompiliert beliebige Closures zur Änderung der Scores und Blockmasken zu fusionierten Triton-Kerneln. Parallele Kernel für das Dekodieren mit KV-Cache (:numref:`sec_lm_decode_kernels`) teilen die Schlüsselachse auf und führen die Teilzustände $(m_B, s_B, \mathbf{o}_B)$ mithilfe des Online-Softmax-Algorithmus zusammen.

<!--en:9352878e-->
### Rekurrenzen mit Gattern und hybride Architekturen

<!--en:aa39988b-->
Moderne Weiterentwicklungen der linearen Attention nutzen zwei Verfahren aus :numref:`chap_modern_rnn`, um die Speicherkapazität zu verbessern. Das sind (1) datenabhängige Vergessensgatter (engl. forget gates) wie bei *Gated Linear Attention* (:cite:`Yang.Wang.Shen.ea.2024`, :numref:`sec_matrix-state`) und (2) fehlerkorrigierende Delta-Updates wie bei *DeltaNet* (:cite:`Schlag.Irie.Schmidhuber.2021,Yang.Wang.Zhang.ea.2024`) und *Gated DeltaNet* (:cite:`Yang.Kautz.Hatamizadeh.2024`, :numref:`sec_deltanet`). In der Praxis kombinieren Architekturen diese rekurrenten Schichten in regelmäßigen Abständen mit Schichten, die volle Softmax-Attention verwenden (typischerweise 1 von 4 bis 1 von 8 Schichten). Dadurch bleibt der exakte assoziative Abruf erhalten :cite:`Qwen.Team.2025,Kimi.Team.2025b,MiniMax.2025` (:numref:`sec_hybrids`).

<!--en:8aefd574-->
## Schaltkreise in großen Modellen

<!--en:8e9f08c5-->
### Activation-Patching, Superposition und Erklärung

<!--en:38cb8d5a-->
Bei der Ablation ersetzt man Aktivierungen durch null oder durch ihren Mittelwert (:numref:`subsec_head-ablation`). *Activation-Patching* geht weiter und setzt Aktivierungen aus einem ungestörten Durchlauf in einen gestörten Durchlauf ein. Damit lassen sich kausale Pfade isolieren :cite:`Meng.Bau.Andonian.ea.2022,Wang.Variengien.Conmy.ea.2023`. Mit ACDC :cite:`Conmy.MavorParker.Lynch.ea.2023` oder *Attribution-Patching* erster Ordnung :cite:`Nanda.2023,Kramar.Lieberum.Shah.ea.2024,Zhang.Nanda.2024,Heimersheim.Nanda.2024` lässt sich das Verfahren automatisieren. Gibt es mehr Merkmale als Dimensionen im Residualstrom (*Superposition*, :cite:`Elhage.Hume.Olsson.ea.2022`), zerlegen *dünn besetzte Autoencoder* (engl. sparse autoencoders) und Attributionsgraphen polysemantische Richtungen in monosemantische Merkmale :cite:`Bricken.Templeton.Batson.ea.2023,Cunningham.Ewart.Riggs.ea.2024,Ameisen.Lindsey.Pearce.ea.2025` (:numref:`sec_lm_interpretability`). Mit solchen gezielten Eingriffen lässt sich auch die klassische Streitfrage zwischen :citet:`Jain.Wallace.2019` und :citet:`Wiegreffe.Pinter.2019` klären, ob Attention-Gewichte die Entscheidungen eines Modells getreu erklären.

<!--en:04fc81e2-->
### In-Context-Lernen als Algorithmus

<!--en:6777c59d-->
Neben dem Kopieren durch Induktionsköpfe (engl. induction heads) :cite:`Olsson.Elhage.Nanda.ea.2022` können Transformer auch Rechenverfahren direkt im Vorwärtsdurchlauf umsetzen. Trainiert man sie auf Familien von Regressionsaufgaben mit Beispielen im Kontext, können sie Gradientenabstieg (engl. gradient descent) und Ridge-Regression ausführen sowie Probleme der kleinsten Quadrate lösen :cite:`Garg.Tsipras.Liang.ea.2022,Oswald.Niklasson.Randazzo.ea.2023,Akyurek.Schuurmans.Andreas.ea.2023`. Inwieweit ein allgemeines Pre-Training auf Texten einen impliziten Gradientenabstieg hervorbringt, ist jedoch umstritten :cite:`Shen.Mishra.Khashabi.2024`.

<!--en:60b0333c-->
## Empfohlene Quellen

<!--en:e8dd6d7a-->
* **Visuelle Einführungen:** 3Blue1Brown, [*Attention in Transformern Schritt für Schritt*](https://www.3blue1brown.com/lessons/attention) (2024), Jay Alammar, [*Der Transformer in Bildern*](https://jalammar.github.io/illustrated-transformer/) (2018), Peter Bloem, [*Transformer von Grund auf*](https://peterbloem.nl/blog/transformers) (2019), und Lilian Weng, [*Die Transformer-Familie, Version 2.0*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) (2023).

<!--en:429e8694-->
* **Monografien zu Grundlagen und Schaltkreisen:** Elhage et al., [*Ein mathematischer Rahmen für Transformer-Schaltkreise*](https://transformer-circuits.pub/2021/framework/index.html) (2021), Olsson et al., [*In-Context-Lernen und Induktionsköpfe*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (2022), Weiss, Goldberg und Yahav, [*Denken wie ein Transformer*](https://arxiv.org/abs/2106.06981) (2021), sowie die begleitenden [*Transformer-Rätsel*](https://github.com/srush/Transformer-Puzzles) von Sasha Rush.

<!--en:8beae255-->
[Diskussion](https://d2l.discourse.group/t/36852)
