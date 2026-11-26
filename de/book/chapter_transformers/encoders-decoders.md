<!--en:d7b95771-->
# Encoder, Decoder und Cross-Attention

<!--en:f37e8b4d-->
Bisher war jeder unserer Transformer-Stapel ein **kausaler Decoder** (:numref:`sec_gpt`). Eine Dreiecksmaske erlaubt Token $i$ nur den Zugriff auf die Präfixpositionen $j \le i$. So liefert ein einziger Vorwärtsdurchlauf (engl. forward pass) parallel $T$ Vorhersagen des jeweils nächsten Tokens (engl. next-token prediction). Wenn wir Token einzeln erzeugen, ist die kausale Maskierung unverzichtbar, denn bei der Inferenz existieren die künftigen Token noch nicht. Bei anderen Aufgaben liegt die *gesamte* Eingabesequenz bereits vor, etwa wenn wir ein Dokument für das Retrieval einbetten, Token klassifizieren, einen Ausgangssatz für die Übersetzung kodieren oder ein Raster aus Bild-Patches kodieren (:numref:`sec_vision-transformer`). Verwehren wir Position $i$ den Zugriff auf die Positionen $j > i$, geht die Hälfte des umgebenden Kontexts verloren.

<!--en:6c96e267-->
Welche Token liegen zu Beginn der Berechnung vor? Gehören Eingabe und Ausgabe zur selben Sequenz? Je nach Antwort lässt sich derselbe `TransformerBlock` (:numref:`sec_transformer-block`) auf drei grundlegende Arten verschalten (:numref:`fig_three-wirings`):

<!--en:bff9b4da-->
1. **Reiner Encoder (bidirektionale Self-Attention):** Jedes Token greift auf jedes Token zu (`is_causal=False`). So entsteht in einem Durchlauf eine kontextabhängige Repräsentation der gesamten Eingabesequenz (BERT :cite:`Devlin.Chang.Lee.ea.2018`, RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`, ViT :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`).

<!--en:cf4590ca-->
2. **Reiner Decoder (kausale Self-Attention):** Jedes Token greift nur auf sich selbst und frühere Positionen zu (`is_causal=True`). Derselbe einheitlich aufgebaute Stapel liest den Prompt und generiert autoregressiv weitere Token (GPT, Llama 3, Qwen3).

<!--en:5eacca8b-->
3. **Encoder-Decoder (bidirektionaler Encoder + kausaler Decoder mit Cross-Attention):** Ein bidirektionaler Encoder bildet die Quellsequenz $\mathbf{X}_{\textrm{src}}$ auf kontextabhängige Repräsentationen $\mathbf{H}_{\textrm{enc}}$ ab. Ein kausaler Decoder erzeugt die Zielsequenz $\mathbf{Y}_{\textrm{tgt}}$ Token für Token und fragt dabei $\mathbf{H}_{\textrm{enc}}$ über **Cross-Attention** ab (der ursprüngliche Transformer von 2017 :cite:`Vaswani.Shazeer.Parmar.ea.2017`, T5 :cite:`raffel2020exploring`, Whisper :cite:`radford2023whisper`).

<!--en:222e20ca-->
## Architekturen mit reinem Encoder, reinem Decoder und Encoder-Decoder

<!--en:62475b2c-->
![Die drei Verschaltungen und ihre Attention-Muster. Die Abfrageposition nimmt nach unten zu, die Schlüsselposition nach rechts. Eine ausgefüllte Zelle zeigt an, dass die Abfrage auf diesen Schlüssel zugreifen darf. Beim reinen Encoder greift jedes Token auf jedes Token zu. Beim reinen Decoder greift jedes Token nur auf sich selbst und seine Vorgänger zu. Beim Encoder-Decoder greift die Quellsequenz bidirektional auf sich selbst zu. Jedes Zieltoken greift über Cross-Attention auf die gesamte Quelle und über kausale Self-Attention auf seine Vorgänger zu. Ein Detail ist im dritten Teilbild nicht zu sehen. Die Zeilen für Quelle und Ziel gehören zu zwei getrennten Stapeln, und jeder Decoder-Block liest die Ausgabe des letzten Encoder-Blocks.](../img/mdl-transformers-three-wirings.svg)

<!--en:7f349cdc-->
## Bidirektionaler Kontext und Vorhersage maskierter Token

<!--en:523f954c-->
### Ein bidirektionaler Encoder

<!--en:07f8da1a-->
Entfernen wir die kausale Maske aus `d2l.TransformerBlock`, wird aus einem kausalen Stapel ein **bidirektionaler Transformer-Encoder** (`TransformerEncoder`). Er erhält einen Batch von Token-IDs und optional eine Padding-Maske. Daraus berechnet er kontextabhängige Repräsentationen der Form $(B, n, d)$:

<!--en:9ad2ca11-->
### Das Trainingsziel der maskierten Sprachmodellierung (MLM)

<!--en:949e9a85-->
Warum können wir einen bidirektionalen Stapel nicht wie in :numref:`sec_gpt` auf die Vorhersage des nächsten Tokens trainieren? Das Ziel an Position $t$ ist das Token $x_{t+1}$. Genau dieses Token steht als Eingabe an Position $t+1$. Ohne Maske kann Position $t$ schon im ersten Block auf Position $t+1$ zugreifen und dort ihr eigenes Ziel ablesen (:numref:`fig_mdl-transformers-leakage`). Für die Vorhersage des nächsten Tokens genügt es also, die Eingabe zu kopieren.

<!--en:c1768a8b-->
![Vorhersage des nächsten Tokens in einem Block. Ohne Maske greift Position 3 auf jede Eingabe zu, auch auf ihr eigenes Ziel $x_4$. Mit der kausalen Maske liest sie nur die Positionen 1 bis 3.](../img/mdl-transformers-leakage.svg)

<!--en:4d28580d-->
Die **maskierte Sprachmodellierung** (engl. masked language modeling, MLM) wurde mit **BERT** eingeführt :cite:`Devlin.Chang.Lee.ea.2018`. Sie ersetzt die Vorhersage des nächsten Tokens durch eine Lückentextaufgabe. Wir wählen zufällig eine Teilmenge der Positionen $M \subset \{1, \dots, n\}$ aus (typischerweise $p = 15\%$ der Token). Jedes $x_i$ ($i \in M$) ersetzen wir durch die eigens vorgesehene Token-ID für `<mask>`. So entsteht die verfälschte Eingabe $\tilde{\mathbf{x}}$. Wir minimieren die Kreuzentropie bei der Rekonstruktion der ursprünglichen Token *nur an den maskierten Positionen* $i \in M$:

<!--en:d0503cbc-->
Wir trainieren ein bytebasiertes `MaskedLM` mit 4 Blöcken ($d = 128, H = 4, L = 4$) auf genau demselben Bytestrom aus WikiText-103 wie in :numref:`sec_gpt`. Die Token-ID `MASK = len(data.vocab)` ($256$) reservieren wir für `<mask>`:

<!--en:d6e9ceb4-->
### Verlust in Abhängigkeit vom verfügbaren Kontext

<!--en:46e1dbc7-->
In :numref:`subsec_gpt-rnn-baseline` nahm der Validierungsverlust eines kausalen Sprachmodells von Position zu Position monoton ab, von links ($t = 1$, kein Kontext) nach rechts ($t = T$, vollständiger Präfixkontext). Wie sieht `loss_by_position` für ein bidirektionales `MaskedLM` aus, wenn wir den Rekonstruktionsverlust an jeder Fensterposition $t \in \{0, \dots, n-1\}$ bestimmen?

<!--en:45be31d4-->
Für den bidirektionalen Encoder ergibt sich eine symmetrische **Badewannenkurve (U-Form)**. An jeder Position im Inneren steht Kontext von links und rechts zur Verfügung (etwa 1.5–1.6 Nat/Byte). Nur an den beiden Fensterrändern steigt der Verlust ($t \approx 0$ und $t \approx n-1$, etwa 2.4 Nat/Byte), weil der Kontext dort auf einer Seite fehlt. Werten wir `model` an einem maskierten Satz aus, sehen wir, wie der rechte Kontext fehlende Bytes im Satzinneren eindeutig bestimmt:

<!--en:6be38e71-->
## Cross-Attention in einem Encoder-Decoder

<!--en:dcb7753f-->
### Eine Sequenz-zu-Sequenz-Aufgabe mit bekanntem Alignment

<!--en:2fed3239-->
Um einen **Encoder-Decoder** (:numref:`fig_three-wirings`, rechts) an einer Sequenz-zu-Sequenz-Aufgabe mit eindeutiger Zuordnung der Positionen zu untersuchen, trainieren wir ihn auf die Umkehrung von Zeichenketten. Das Alphabet umfasst $V = 16$ Buchstaben (`a` bis `p`), die Länge beträgt $T = 12$. Zur Quellzeichenkette $\mathbf{x} = (x_1, \dots, x_T)$ erhält der Decoder die Eingabe $\textrm{dec\_in} = (\texttt{<bos>}, x_T, x_{T-1}, \dots, x_2)$ und sagt die umgekehrte Zeichenkette $\mathbf{y} = (x_T, x_{T-1}, \dots, x_1)$ vorher:

<!--en:f7ac59b4-->
### Der Decoder-Block und `EncoderDecoder`

<!--en:d933bcc4-->
Ein `DecoderBlock` fügt zwischen der kausalen Self-Attention und dem SwiGLU-FFN eine Teilschicht für **Cross-Attention** ein (`d2l.MultiHeadAttention`). Bei der Cross-Attention **stammen die Abfragen vom Decoder** ($\mathbf{Q} = \textrm{Norm}(\mathbf{Y})\mathbf{W}_q \in \mathbb{R}^{B \times T_{\textrm{dec}} \times d}$). **Schlüssel und Werte stammen aus der obersten Encoder-Schicht** ($\mathbf{K} = \mathbf{H}_{\textrm{enc}}\mathbf{W}_k \in \mathbb{R}^{B \times T_{\textrm{src}} \times d}$, $\mathbf{V} = \mathbf{H}_{\textrm{enc}}\mathbf{W}_v \in \mathbb{R}^{B \times T_{\textrm{src}} \times d}$). Da die gesamte kodierte Quelle $\mathbf{H}_{\textrm{enc}}$ bereits vor der Dekodierung vorliegt, verwendet die Cross-Attention **keine kausale Maske**:

<!--en:6f59ba51-->
`EncoderDecoder` kombiniert einen `TransformerEncoder` mit einem Stapel aus `DecoderBlock`-Schichten und einem linearen Ausgabekopf. Die Funktion `decode(dec_in, enc)` erhält die vorberechnete Encoder-Ausgabe `enc`. Bei der autoregressiven Generierung verarbeitet der Encoder die Quellsequenz also **nur einmal**, während `decode` die Zieltoken Schritt für Schritt erzeugt:

<!--en:8c8eda23-->
### Training und autoregressives Dekodieren

<!--en:bc142ac2-->
Wir trainieren `EncoderDecoder` für 300 Schritte mit zufälligen Batches von je 128 Zeichenketten. Bei der anschließenden autoregressiven Greedy-Dekodierung (`reverse`) kehrt das Modell fast jede Zeichenkette vollständig korrekt um:

<!--en:baf87237-->
Bei der Inferenz kodiert `reverse(model, src)` die Quellsequenz einmal. Ausgehend von `<bos>` wählt die Funktion in jedem Schritt das wahrscheinlichste Zieltoken. Von $1{,}000$ zufälligen Zeichenketten, die beim Training unberücksichtigt blieben, kehrt sie fast alle vollständig korrekt um:

<!--en:46139c86-->
### Interpretation des Cross-Attention-Alignments

<!--en:2d8e016e-->
Bei der Umkehrung einer Zeichenkette gibt der Decoder im Zielschritt $t$ das Quelltoken $T - 1 - t$ aus. Müssen sich die Gewichte in der Cross-Attention-Matrix deshalb auf der Gegendiagonalen $(t, T - 1 - t)$ konzentrieren? Vergleichen wir unser trainiertes Modell `EncoderDecoder` mit je einem Encoder- und Decoder-Block (Variante `with`, mit Self-Attention im Encoder) mit einem ansonsten identischen Modell. Dessen Encoder enthält nur seine Einbettungen und keine Blöcke (Variante `without`, sodass Encoder-Position $j$ nur Information über Quelltoken $j$ enthalten kann):

<!--en:194e930b-->
Wir trainieren beide Varianten, `with` und `without`, mit jeweils sechs unabhängigen Zufallsstartwerten. Für jede Gruppe zeigen wir die Cross-Attention-Heatmap, bei der die Gewichte am wenigsten auf der Gegendiagonalen liegen. Der Vergleich zeigt den Unterschied zwischen direktem Zugriff über die Position und der Weiterleitung von Information über mehrere Schichten:

<!--en:2537faaf-->
Ohne Self-Attention im Encoder (`without`) *muss* die Cross-Attention direkt auf die Gegendiagonale $(t, T - 1 - t)$ zugreifen, um Quelltoken $T - 1 - t$ abzurufen. Mit bidirektionaler Self-Attention (`with`) können die Encoder-Schichten die Quellinformation bereits zwischen den Positionen weiterleiten oder neu anordnen, bevor die Cross-Attention $\mathbf{H}_{\textrm{enc}}$ liest. Das illustriert eine bekannte Beobachtung :cite:`Jain.Wallace.2019`. Rohe Attention-Gewichte in einem mehrschichtigen Netz spiegeln die Weiterleitung zwischen Schichten wider. Aus ihnen lässt sich der Beitrag eines Tokens zu einem anderen nicht direkt ablesen.

<!--en:3877c146-->
## Cross-Attention als Schnittstelle für latente Flaschenhälse fester Größe

<!--en:a503eba4-->
Bei der Cross-Attention müssen Abfragen $\mathbf{Q} \in \mathbb{R}^{B \times M \times d}$ und Schlüssel-Wert-Paare $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times N \times d}$ nur in der Merkmalsbreite $d$ übereinstimmen. Dadurch lässt sich die intern verarbeitete Sequenzlänge $M$ von der Länge $N$ der Roheingabe entkoppeln (:numref:`fig_latent-bottleneck`).

<!--en:ba788cf6-->
![Ein latenter Flaschenhals mit Eingabelänge $N$, $M$ gelernten latenten Vektoren, Breite $d$ und $L$ Blöcken für die latenten Vektoren. Jeder latente Vektor liest alle Eingaben (schattiertes Band). Für einen Vektor sind alle Lesezugriffe eingezeichnet. Beim einmaligen Lesen über Cross-Attention kostet das Mischen $O(MNd)$. Jede anschließende Self-Attention-Schicht mischt die latenten Vektoren mit Kosten von $O(M^2d)$, über den gesamten Stapel also $O(LM^2d)$. Diese Terme setzen eine feste Batchgröße, Kopfanzahl und Breite voraus. Projektionen und FFNs sind nicht eingerechnet.](../img/mdl-transformers-latent-bottleneck.svg)

<!--en:4a3c302e-->
Im **Perceiver** :cite:`Jaegle.Gimeno.Brock.ea.2021` und in multimodalen Resamplern für Bild und Sprache :cite:`alayrac2022flamingo` greift ein gelerntes Array aus $M \ll N$ latenten Abfragevektoren $\mathbf{Z}_0 \in \mathbb{R}^{M \times d}$ über Cross-Attention auf eine Eingabe aus $N$ Token $\mathbf{X} \in \mathbb{R}^{B \times N \times d}$ zu. Dafür sind $O(M N d)$ Operationen nötig. Anschließend verarbeiten $L$ Self-Attention-Blöcke das kompakte latente Array der Länge $M$ mit $O(L M^2 d)$ statt $O(L N^2 d)$ Operationen:

<!--en:8f2fe29d-->
## Wahl zwischen den drei Verschaltungen

<!--en:6c2d164e-->
* Ein **reiner Encoder (`is_causal=False`)** gibt jedem Token in einem Durchlauf den vollständigen bidirektionalen Kontext. Er ist daher die übliche Basis für dichte Texteinbettungen, die Neusortierung von Suchergebnissen, die Tokenklassifikation und nicht autoregressive Modelle zur Bildverarbeitung (:numref:`sec_vision-transformer`).

<!--en:0c7e1bbf-->
* Ein **reiner Decoder (`is_causal=True`)** nutzt bei der Vorhersage des nächsten Tokens $100\%$ der Token einer Sequenz als Trainingsziele (gegenüber $15\%$ bei MLM). Für Prompts und generierte Fortsetzungen verwendet er denselben Stapel mit denselben Parametern und demselben KV-Cache (:numref:`sec_kv-cache`). Damit ist er die vorherrschende Architektur für universelle Sprachmodelle.

<!--en:b699e535-->
* Ein **Encoder-Decoder (mit Cross-Attention)** eignet sich besonders, wenn sich Eingabe und Ausgabe in ihrer Modalität oder Länge unterscheiden. So wandelt Whisper Sprachspektrogramme in Text um :cite:`radford2023whisper`. Hochaufgelöste Bild- oder Sensordaten lassen sich über eine latente Cross-Attention-Schnittstelle fester Größe komprimieren (:numref:`fig_latent-bottleneck`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:aca49938-->
Derselbe `TransformerBlock` lässt sich als **bidirektionaler Encoder** (`is_causal=False`), als **kausaler Decoder** (`is_causal=True`) oder als **Encoder-Decoder mit Cross-Attention** (:numref:`fig_three-wirings`) verschalten. Bei der Vorhersage des nächsten Tokens könnte jede Position über unmaskierte Self-Attention ihr eigenes Ziel lesen. Deshalb trainieren wir bidirektionale Encoder mit **maskierter Sprachmodellierung (MLM)** (:eqref:`eq_mlm`). Über die Fensterpositionen ergibt sich dabei für `loss_by_position` eine symmetrische U-Kurve. In einem **Encoder-Decoder** kombiniert jeder `DecoderBlock` kausale Self-Attention auf der Zielsequenz mit **Cross-Attention** und einem positionsweisen FFN. Bei der Cross-Attention greifen die Abfragen des Decoders auf Schlüssel und Werte des Encoders zu. Der Encoder verarbeitet die Quelle daher nur einmal, während der Decoder autoregressiv fortschreitet. Allgemeiner kann Cross-Attention mit $M \ll N$ gelernten Abfragevektoren (`PerceiverEncoder`) eine Eingabe mit $N$ Token auf einen latenten Flaschenhals mit $M$ Token komprimieren. Die Laufzeit beträgt $O(M N d + L M^2 d)$ (:numref:`fig_latent-bottleneck`).

<!--en:2294243a-->
## Übungen

<!--en:2415ae1a-->
1. **Die 80 / 10 / 10-Regel zur Maskierung in BERT.** `mask_tokens` ersetzt das Token an jeder ausgewählten Position ($15\%$ der Bytes) durch `<mask>`. Da `<mask>` beim Fine-Tuning und bei der Inferenz nie vorkommt, gehen :citet:`Devlin.Chang.Lee.ea.2018` anders vor. Sie ersetzen die ausgewählten Token durch `<mask>` in $80\%$ der Fälle und durch ein gleichverteilt aus dem Vokabular gezogenes Token in $10\%$ der Fälle. In $10\%$ der Fälle behalten sie das ursprüngliche Token bei. Implementieren Sie diese 80 / 10 / 10-Regel in `mask_tokens`. Erklären Sie, warum die beiden Anteile von $10\%$ unmaskierten und $10\%$ zufällig ersetzten Token den Encoder zwingen, sowohl für *unmaskierte* Token als auch für `<mask>`-Token präzise Repräsentationen zu bilden.
   :exlabel:`ex_encdec-masking-recipe`

<!--en:bf5e822e-->
2. **Padding-Masken für Sequenz-zu-Sequenz-Batches mit unterschiedlichen Längen.** Ein Batch enthält Quellzeichenketten der Längen $(s_1, \dots, s_B) \le S$ und Zielzeichenketten der Längen $(t_1, \dots, t_B) \le T_{\textrm{dec}}$. Dafür sind drei boolesche Masken nötig: (a) eine Padding-Maske der Form $(B, 1, S, S)$ für den Encoder, (b) eine kombinierte Kausal- und Padding-Maske der Form $(B, 1, T_{\textrm{dec}}, T_{\textrm{dec}})$ für den Decoder und (c) eine rechteckige Padding-Maske der Form $(B, 1, T_{\textrm{dec}}, S)$ für die Quellpositionen in der Cross-Attention. Schreiben Sie eine Hilfsfunktion `make_seq2seq_masks(src_len, tgt_len, S, T_dec)`, die alle drei Masken erstellt. Prüfen Sie, dass das Anhängen von Padding-Token an `src` in `EncoderDecoder` die Decoder-Logits an den nicht aufgefüllten Positionen unverändert lässt.
   :exlabel:`ex_encdec-padding`

<!--en:546c2212-->
3. **Variation des Maskierungsanteils $p$ bei MLM.** Trainieren Sie `MaskedLM` für jeweils 2000 Schritte mit den Maskierungsanteilen $p \in \{0.05, 0.15, 0.30, 0.50\}$. Werten Sie alle vier Modelle bei $p = 0.15$ auf demselben Validierungsdatensatz `masked.val_dataloader()` aus. Wie beeinflusst $p$ die Anzahl der überwachten Gradientensignale pro Batch und die Menge an unverfälschtem Kontext, der um jede maskierte Position verfügbar ist?

<!--en:694750dc-->
4. **Vorwärts- und Rückwärtsrichtung der Zeit in kausalen und bidirektionalen Modellen.** Ein kausales Sprachmodell verarbeitet Text von links nach rechts ($x_1 \to x_n$). Kehren Sie jedes Bytefenster in `WikiText103` um, sodass ein kausales Modell von rechts nach links vorhersagt ($x_n \to x_1$). Trainieren Sie es für 1000 Schritte. Vergleichen Sie seinen mittleren Validierungsverlust pro Byte und seine Kurve für `loss_by_position` mit den Ergebnissen des kausalen Modells mit Leserichtung von links nach rechts und des bidirektionalen `MaskedLM`. Warum ist die gesamte gemeinsame Log-Likelihood $\sum_{t=1}^n \log P(x_t \mid \textrm{past})$ theoretisch invariant gegenüber Zeitumkehr? Woher stammt der Vorteil des bidirektionalen Modells gegenüber beiden einseitigen Leserichtungen?
   :exlabel:`ex_encdec-times-arrow`

<!--en:3b90a346-->
5. **Inhalts- und positionsbasiertes Alignment bei der Cross-Attention.** In unserem Experiment zur Umkehrung von Zeichenketten kann jeder Buchstabe in einer zufälligen Zeichenkette mehrfach vorkommen. Um gleiche Buchstaben zu unterscheiden, brauchen wir daher ihre Position. Ändern Sie `sample_batch` so, dass jede Zeichenkette der Länge 12 eine zufällige Permutation von 12 *verschiedenen* Buchstaben ist. Trainieren Sie `EncoderDecoder` ohne den Positionsterm im Vorwärtsdurchlauf des Encoders. Warum scheitert ein Encoder mit 1 Schicht ohne Positionseinbettungen an der Umkehrung, selbst wenn alle 12 Buchstaben verschieden sind? Warum gelingt ihm dagegen das Sortieren (Ausgabe der Eingabebuchstaben in alphabetischer Reihenfolge `a..p`) ganz ohne Positionseinbettungen im Encoder?
   :exlabel:`ex_encdec-depth-heads`

<!--en:a50359ab-->
6. **Perceiver IO zur Umkehrung von Sequenzen.** Perceiver IO :cite:`Jaegle.Borgeaud.Alayrac.ea.2022` kombiniert einen `PerceiverEncoder` ($M$ gelernte latente Abfragen greifen auf die Eingabe mit $N$ Token zu) mit einem **Decoder** mit Cross-Attention. Darin greifen $N_{\textrm{out}}$ Positionseinbettungen des Ziels $\mathbf{Q}_{\textrm{out}} \in \mathbb{R}^{N_{\textrm{out}} \times d}$ über Cross-Attention auf die $M$ latenten Vektoren $\mathbf{Z}^{(L)} \in \mathbb{R}^{B \times M \times d}$ zu. Bauen Sie einen `PerceiverAutoencoder`, der eine Zeichenkette aus 12 Buchstaben in $M \in \{2, 4, 8, 12\}$ latente Vektoren ($d = 128$) kodiert und die umgekehrte Zeichenkette in einem einzigen parallelen Durchlauf über Cross-Attention ausliest. Wie hängt die Genauigkeit der Umkehrung von der Größe $M$ des Flaschenhalses ab?
   :exlabel:`ex_encdec-perceiver-reversal`

<!--en:34839f7f-->
[Diskussionen](https://d2l.discourse.group/t/36777)

<!--en:6048b93f-->
Bidirektionale Encoder, Encoder-Decoder und Cross-Attention<br>**drei Blockverschaltungen · maskierte Sprachmodellierung · Cross-Attention für Sequenz-zu-Sequenz-Aufgaben · latente Flaschenhälse im Perceiver**

<!--en:8cdd20b6-->
::: {.slide title="Nachteile kausaler Maskierung bei vollständig bekannter Eingabe"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:4999b5af-->
- **Eindeutige Ergänzung im Lückentext:** In `Homarus gammarus, known as _he European lobster` lässt das linke Präfix `known as ` Hunderte Ergänzungen zu. Mit dem rechten Kontext `he European` ist nur `t` für das fehlende Byte plausibel (Wahrscheinlichkeit $96\text{--}97\%$).

<!--en:37481860-->
- **Gewinn durch beidseitigen Kontext:** In Bytefenstern aus WikiText-103 sinkt der Verlust bei der Rekonstruktion eines maskierten Bytes von **$\sim 2.4\text{ nats}$** am Rand mit einseitigem Kontext auf **$\approx 1.55\text{ nats}$** im Inneren mit beidseitigem Kontext.

<!--en:9dbc3829-->
::: {.slide title="Ein Transformer-Block, drei grundlegende Verschaltungen"}

<!--en:161e9166-->
[Architekturtypen]{.kicker}

<!--en:e2a20ee6-->
- **1. Reiner Encoder (`is_causal=False`):** Jedes Token greift in allen $L$ Schichten in einem einzigen bidirektionalen Durchlauf auf jedes Token zu (BERT, RoBERTa, ViT).

<!--en:38126379-->
- **2. Reiner Decoder (`is_causal=True`):** Jedes Token greift nur auf Positionen $j \le i$ zu. Derselbe einheitlich aufgebaute Stapel liest den Prompt und generiert die Fortsetzung (GPT, Llama 3, Qwen3).

<!--en:afac055b-->
- **3. Encoder-Decoder:** Ein bidirektionaler Encoder berechnet die Abbildung $\mathbf{X}_{\text{src}} \to \mathbf{H}_{\text{enc}}$ für die Quelle. Ein kausaler Decoder generiert $\mathbf{Y}_{\text{tgt}}$ und liest dabei $\mathbf{H}_{\text{enc}}$ über **Cross-Attention** (Transformer von 2017, T5, Whisper).

<!--en:e4c10e20-->
::: {.slide title="Bidirektionaler Encoder ohne kausale Maske" only="pytorch"}

<!--en:0cc5f3e5-->
[Implementierung des Encoders]{.kicker}

<!--en:5f2336b2-->
Ohne die Dreiecksmaske in `d2l.TransformerBlock` wird aus einem kausalen Stapel ein `TransformerEncoder`. Er bildet einen Batch der Form $(B, n)$ aus Token-IDs auf bidirektionale kontextabhängige Repräsentationen in $\mathbb{R}^{B \times n \times d}$ ab:

<!--en:97508c8c-->
Jeder Ausgabevektor $\mathbf{h}_i^{(L)} \in \mathbb{R}^d$ fasst die gesamte Sequenz aus Sicht von Position $i$ zusammen. Pooling über die Positionen (Mittelwertbildung oder Auslesen an Index 0) liefert pro Dokument eine einzige Einbettung fester Breite.

<!--en:0f63717e-->
::: {.slide title="Bidirektionaler Encoder ohne kausale Maske" only="jax"}

<!--en:0cc5f3e5-2-->
[Implementierung des Encoders]{.kicker}

<!--en:5f2336b2-2-->
Ohne die Dreiecksmaske in `d2l.TransformerBlock` wird aus einem kausalen Stapel ein `TransformerEncoder`. Er bildet einen Batch der Form $(B, n)$ aus Token-IDs auf bidirektionale kontextabhängige Repräsentationen in $\mathbb{R}^{B \times n \times d}$ ab:

<!--en:97508c8c-2-->
Jeder Ausgabevektor $\mathbf{h}_i^{(L)} \in \mathbb{R}^d$ fasst die gesamte Sequenz aus Sicht von Position $i$ zusammen. Pooling über die Positionen (Mittelwertbildung oder Auslesen an Index 0) liefert pro Dokument eine einzige Einbettung fester Breite.

<!--en:caa34b13-->
::: {.slide title="Warum das nächste Token als Trainingsziel für Encoder ungeeignet ist"}

<!--en:121d0f94-->
[Unbeabsichtigter Zugriff auf das Ziel]{.kicker}

<!--en:1c3bfe33-->
- **Direkter Zugriff in Schicht 1:** Abfrage $t$ greift direkt auf Schlüssel $t+1$ zu und kopiert so die Zieleinbettung $\mathbf{e}_{x_{t+1}}$ nach $\mathbf{h}_t^{(1)}$.

<!--en:d918912f-->
- **Indirekter Zugriff in Schicht 2:** Selbst bei maskiertem Eintrag $(t, t+1)$ greift Abfrage $t$ in Schicht 2 auf Position $t+2$ (oder $t-1$) zu. Diese hat $x_{t+1}$ bereits in Schicht 1 gelesen.

<!--en:ba91b82b-->
- **Regel:** Jedes Zieltoken $x_i$ ($i \in \mathcal{M}$) schon in der Eingabe vor Schicht 1 durch `<mask>` ersetzen.

<!--en:938f758a-->
::: {.slide title="Maskierte Sprachmodellierung (MLM)"}

<!--en:270c247e-->
[Bidirektionales Pre-Training]{.kicker}

<!--en:ad1ada60-->
Mit **BERT** (Devlin et al., 2018) wurde die **maskierte Sprachmodellierung** eingeführt. Sie wählt zufällig eine Teilmenge der Positionen $\mathcal{M} \subset \{1, \dots, n\}$ aus (mit Wahrscheinlichkeit $p = 0.15$) und ersetzt jedes $x_i$ ($i \in \mathcal{M}$) durch die spezielle Token-ID für `<mask>` in $\tilde{\mathbf{x}}$. Die Kreuzentropie wird *nur* für die maskierte Teilmenge $\mathcal{M}$ berechnet:

<!--en:89fea7c4-->
- **Maskierung der Labels (`-100`):** Unmaskierte Positionen ($i \notin \mathcal{M}$, $85\%$ der Token) erhalten das Ziellabel `-100`. So liefern sie Kontext, ohne dass triviales Kopieren zum Verlust beiträgt.

<!--en:8d8f3462-->
- **Abwägung bei der Dateneffizienz:** Pro Trainingsfenster liefern nur $15\%$ der Token ein Trainingssignal (etwa $10$ von $64$ Bytes), bei einem kausalen GPT dagegen $100\%$ der Token.

<!--en:ac3a6af9-->
::: {.slide title="U-förmige Verlustkurve über die Fensterpositionen"}

<!--en:68ad9631-->
[Nutzen des beidseitigen Kontexts]{.kicker}

<!--en:0eebc105-->
Die Auswertung des trainierten bytebasierten `MaskedLM` mit 4 Blöcken an den Fensterpositionen $t \in \{0, \dots, 63\}$ zeigt einen deutlichen Unterschied zur monoton fallenden Kurve eines kausalen Modells:

<!--en:0b203097-->
- **Symmetrische Badewannenkurve (U-Form):** An den inneren Positionen ($t \in [16, 47]$) ist linker und rechter Kontext verfügbar. Der mittlere Verlust bleibt dort bei **$\approx 1.55\text{ nats/byte}$** ($\approx 53\%$ Genauigkeit bei der bytegenauen Vorhersage).

<!--en:f174e77c-->
- **Zusätzlicher Verlust bei $t = 0$ und $t = 63$:** An beiden Fensterrändern fehlt Kontext auf einer Seite. Dort steigt der Verlust auf **$2.36\text{--}2.44\text{ nats/byte}$**. Die zweite Kontextseite bringt also einen Vorteil von $\sim 0.85\text{-nat}$.

<!--en:3426b424-->
::: {.slide title="Eindeutige Ergänzung von Lücken durch rechten Kontext"}

<!--en:4c18e4bc-->
[Qualitative Untersuchung]{.kicker}

<!--en:c0941cc5-->
- Das Maskieren einzelner Bytes in `'Homarus gammarus , known as the European lobster'` zeigt, wie bidirektionale Attention die Hinweise zur Schreibweise von links und rechts miteinander verbindet:

<!--en:44b657f1-->
- An Position 28 (`known as _he European`) beträgt die vom Encoder geschätzte Wahrscheinlichkeit $96\text{--}97\%$ für `'t'`, an Position 22 (`kno_n as`) beträgt sie $47\text{--}66\%$ für `'w'`.

<!--en:2fbf75f5-->
::: {.slide title="Eindeutige Ergänzung von Lücken durch rechten Kontext (Fortsetzung)"}

<!--en:4c18e4bc-2-->
[Qualitative Untersuchung]{.kicker}

<!--en:e74d86d3-->
Der Test maskiert jeweils ein Byte und gibt die drei wahrscheinlichsten Vorhersagen des Encoders dafür aus:

<!--en:d1b9c804-->
::: {.slide title="Sequenz-zu-Sequenz-Alignment bei der Umkehrung von Zeichenketten"}

<!--en:c119ee6c-->
[Benchmark für Encoder-Decoder]{.kicker}

<!--en:943ce95e-->
- **Decoder-Eingabe beim Teacher Forcing (`dec_in`):** `<bos>` (`id = 16`) voranstellen und die umgekehrte Zielsequenz um eine Position nach rechts verschieben. Mit `dec_in` $= (\texttt{<bos>}, x_T, \dots, x_2)$ sagt der Decoder `tgt` $= (x_T, \dots, x_1)$ vorher.

<!--en:905a8226-->
- **Exaktes Ground-Truth-Alignment:** Für die Ausgabe an Zielposition $t$ muss der Decoder Quelltoken $x_{T - 1 - t}$ entlang der **Gegendiagonalen** $(t, T - 1 - t)$ abrufen.

<!--en:a3291564-->
::: {.slide title="Sequenz-zu-Sequenz-Alignment bei der Umkehrung von Zeichenketten (Fortsetzung)"}

<!--en:c119ee6c-2-->
[Benchmark für Encoder-Decoder]{.kicker}

<!--en:954cf6d9-->
Das Ziehen eines Batches liefert die Quellsequenz, ihre Umkehrung als Ziel und die verschobene Decoder-Eingabe:

<!--en:7733238a-->
::: {.slide title="Der Decoder-Block mit zusätzlicher Cross-Attention-Teilschicht"}

<!--en:f48e9797-->
[Aufbau des Decoder-Blocks]{.kicker}

<!--en:3d11f1ea-->
- Jeder `DecoderBlock` enthält **drei** residuale Teilschichten mit vorgeschalteter Normalisierung. Sie verarbeiten den Zielstrom $\mathbf{X} \in \mathbb{R}^{B \times T_{\text{dec}} \times d}$ bei gegebener kodierter Quelle $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times T_{\text{src}} \times d}$:
  $$\mathbf{U} = \mathbf{X} + \mathrm{SelfAttn}_{\mathrm{causal}}(\mathrm{Norm}_1(\mathbf{X}))$$
  $$\mathbf{V} = \mathbf{U} + \mathrm{CrossAttn}(\mathrm{Norm}_2(\mathbf{U}), \mathbf{H}_{\mathrm{enc}}, \mathbf{H}_{\mathrm{enc}})$$
  $$\mathbf{Y} = \mathbf{V} + \mathrm{FFN}(\mathrm{Norm}_3(\mathbf{V}))$$

<!--en:d1eac44d-->
::: {.slide title="Implementierung von DecoderBlock" only="pytorch"}

<!--en:ce0f4436-->
[Implementierung]{.kicker}

<!--en:89cbcf40-->
`self_attention` wendet eine kausale Maske auf das Zielpräfix an. Dagegen greift `cross_attention` auf die gesamte kodierte Quelle `enc` zu (nur mit der optionalen Padding-Maske `cross_mask` für die Quelle):

<!--en:63347537-->
Bei der autoregressiven Inferenz hängen `enc = self.encoder(src)` und die zugehörigen Schlüssel- und Wertprojektionen der Cross-Attention nur von der Quellsequenz ab. Daher genügt es, sie **einmal pro Prompt** zu berechnen und bei jedem Dekodierschritt wiederzuverwenden.

<!--en:cbcbbb49-->
::: {.slide title="Implementierung von DecoderBlock" only="jax"}

<!--en:ce0f4436-2-->
[Implementierung]{.kicker}

<!--en:89cbcf40-2-->
`self_attention` wendet eine kausale Maske auf das Zielpräfix an. Dagegen greift `cross_attention` auf die gesamte kodierte Quelle `enc` zu (nur mit der optionalen Padding-Maske `cross_mask` für die Quelle):

<!--en:63347537-2-->
Bei der autoregressiven Inferenz hängen `enc = self.encoder(src)` und die zugehörigen Schlüssel- und Wertprojektionen der Cross-Attention nur von der Quellsequenz ab. Daher genügt es, sie **einmal pro Prompt** zu berechnen und bei jedem Dekodierschritt wiederzuverwenden.

<!--en:c4cc0fbc-->
::: {.slide title="Training und Greedy-Dekodierung bei der Umkehrung von Zeichenketten"}

<!--en:7c2cac5e-->
[Sequenz-zu-Sequenz von Anfang bis Ende]{.kicker}

<!--en:01bb59df-->
Beim Training eines `EncoderDecoder` mit 1 Block auf jeweils neu gezogenen Batches von $128$ Zeichenketten sinkt die Kreuzentropie von $\ln 16 \approx 2.77\text{ nats}$ auf $0.002\text{ nats}$ in $300$ Schritten. Das Modell kehrt $100\%$ von $1{,}000$ zuvor ungesehenen Testzeichenketten vollständig korrekt um:

<!--en:0b5b3852-->
::: {.slide title="Kombination der Masken in den drei Attention-Schichten"}

<!--en:1c82926c-->
[Maskierungsregeln]{.kicker}

<!--en:ba807f98-->
- **1. Self-Attention im Encoder $(S \times S)$:** Padding-Maske für die Quelle ($j < s_b$)

<!--en:ca4a7f7f-->
- **2. Self-Attention im Decoder $(T_{\text{dec}} \times T_{\text{dec}})$:** Kausale untere Dreiecksmaske $\land$ Padding-Maske für das Ziel

<!--en:88ab4966-->
- **3. Cross-Attention $(T_{\text{dec}} \times S)$:** **Nur** die Padding-Maske für die Quelle verwenden. Eine kausale Maske ist hier ausgeschlossen, denn Zieltoken $t = 0$ muss sofort auf Quelltoken $S - 1$ zugreifen können.

<!--en:47901212-->
::: {.slide title="Zeigen Cross-Attention-Gewichte den Beitrag einzelner Token?"}

<!--en:9b60c60d-->
[Grenzen der Interpretierbarkeit]{.kicker}

<!--en:1168a659-->
- **Ohne Self-Attention im Encoder (`without`, rechte Heatmap):** Encoder-Position $j$ enthält nur Token $x_j$. Die Cross-Attention ist damit auf die exakte Gegendiagonale $(t, T-1-t)$ festgelegt ($100\%$ der Zeilen).

<!--en:8289abf2-->
- **Mit Self-Attention im Encoder (`with`, linke Heatmap):** Beide Modelle erreichen bei der Umkehrung eine Genauigkeit von $99.9\text{--}100\%$. Dennoch folgt die Cross-Attention teils in nur $15\%\text{--}17\%$ der Zeilen der Gegendiagonalen. Die Self-Attention im Encoder kann Token bereits zwischen Positionen vertauschen oder zusammenführen.

<!--en:8baa5017-->
::: {.slide title="Cross-Attention als latenter Flaschenhals fester Größe"}

<!--en:05922e76-->
[Perceiver-Architektur]{.kicker}

<!--en:44e68c32-->
- **Entkopplung des Rechenaufwands von der Eingabelänge $N$:** Bei `CrossAttn` bestimmt die Anzahl der **Abfragen $M$** die Länge der Ausgabesequenz, unabhängig von der Anzahl der **Schlüssel-Wert-Paare $N$**.

<!--en:15e90338-->
- **Perceiver (Jaegle et al., 2021):** $M \ll N$ gelernte latente Abfragevektoren $\mathbf{Z}_0 \in \mathbb{R}^{M \times d}$ greifen über Cross-Attention auf eine Eingabe mit $N$ Token zu. Das erfordert **$O(M N d)$** Zeit. Anschließend verarbeiten $L$ Self-Attention-Blöcke das kurze latente Array in **$O(L M^2 d)$** Zeit. Damit entfällt der Engpass von $O(L N^2 d)$.

<!--en:ffd2b380-->
::: {.slide title="PerceiverEncoder in zehn Zeilen" only="pytorch"}

<!--en:cdeef85c-->
[Code für den latenten Flaschenhals]{.kicker}

<!--en:c99ec4e2-->
::: {.slide title="PerceiverEncoder in zehn Zeilen" only="jax"}

<!--en:cdeef85c-2-->
[Code für den latenten Flaschenhals]{.kicker}

<!--en:9290867c-->
::: {.slide title="Wahl der Verschaltung in modernen Systemen"}

<!--en:79da16e4-->
[Entscheidungshilfe zur Architektur]{.kicker}

<!--en:1208c7fa-->
- **Reiner Encoder:** Vollständiger beidseitiger Kontext in einem parallelen Durchlauf (Einbettungen, Reranker, ViT)

<!--en:57efd542-->
- **Reiner Decoder:** Ein gemeinsamer Stapel mit KV-Cache bewertet pro Durchlauf $100\%$ der Token (universelle große Sprachmodelle, Code, Chat).

<!--en:7a411dc2-->
- **Encoder-Decoder:** Kodiert ein langes Quellsignal einmal und dekodiert kompakte Zielsequenzen über Cross-Attention (Spracherkennung mit Whisper, Perceiver-Resampler).

<!--en:089fe715-->
::: {.slide title="Verständnisfragen zu Maskierung, Komplexität und Verschaltung"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:6d48d68f-->
Drei Fragen zur Architektur beantworten:

<!--en:64f43024-->
1. **Anzahl der Scores im Perceiver:** Ein Audio- oder Bildarray umfasst $N = 8{,}192$ Token. Ein `PerceiverEncoder` verwendet $M = 64$ gelernte latente Vektoren und $L = 6$ Self-Attention-Blöcke für diese Vektoren. Die Anzahl der Positionspaare mit Attention-Score pro Kopf im Perceiver und in einem Encoder mit 6 Schichten und vollständiger Self-Attention vergleichen.

<!--en:2c2052fc-->
2. **Zwischenspeichern bei der Cross-Attention:** Aus einer kodierten Quelle sollen autoregressiv $T_{\text{dec}} = 100$ Zieltoken dekodiert werden. Die Quelle hat die Länge $T_{\text{src}} = 1{,}000$. Wie oft werden die Schlüssel- und Wertprojektionen $\mathbf{H}_{\text{enc}}\mathbf{W}_k, \mathbf{H}_{\text{enc}}\mathbf{W}_v$ für die Cross-Attention berechnet?

<!--en:8ef5f784-->
3. **Wahl der Architektur:** Für eine Nächste-Nachbarn-Suche sollen $50\text{ million}$ Produktbeschreibungen in $768$-dimensionale Vektoren eingebettet werden. Welche Verschaltung eignet sich und warum?

<!--en:557b98b6-->
**Lösungen:**

<!--en:2fc3049d-->
1. **Vollständiger Encoder:** $6 \times 8192^2 \approx \mathbf{4.03 \times 10^8}$ Positionspaare pro Kopf. **Perceiver:** $64 \times 8192 + 6 \times 64^2 = 524{,}288 + 24{,}576 \approx \mathbf{5.49 \times 10^5}$ Positionspaare pro Kopf (**um den Faktor $734\times$ weniger**).

<!--en:7eb3676e-->
2. **Genau einmal** (unmittelbar nach dem Encoder-Durchlauf), da sich $\mathbf{H}_{\text{enc}}$ bei der Generierung der Zieltoken nie ändert.

<!--en:cad42121-->
3. **Reiner Encoder:** Jedes Token nutzt den vollständigen bidirektionalen Kontext. Ein einziger paralleler Vorwärtsdurchlauf liefert die Einbettung jeder Beschreibung ohne autoregressive Dekodierschleife.

<!--en:80e112eb-->
::: {.slide title="Zusammenfassung zu Encodern, Decodern und Cross-Attention"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:d63ef6ea-->
- **Bidirektionaler Encoder + MLM:** Unmaskierte Self-Attention liest den gesamten linken und rechten Kontext. Das Ersetzen von $15\%$ der Token durch `<mask>` verhindert, dass das Modell seine Ziele ablesen kann.

<!--en:5ec3ebfa-->
- **Cross-Attention:** Verbindet die getrennten Stapel für Quelle und Ziel ($\mathbf{Q}$ aus dem Ziel, $\mathbf{K}, \mathbf{V}$ aus $\mathbf{H}_{\text{enc}}$).

<!--en:7a591657-->
- **Latenter Flaschenhals im Perceiver:** $M \ll N$ gelernte Abfragen senken die Kosten der Attention von $O(L N^2 d)$ auf $O(M N d + L M^2 d)$.
