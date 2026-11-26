<!--en:1581b396-->
# Hybride Architekturen

<!--en:533910c4-->
Von Gattern über selektive Dynamik bis hin zu fehlerkorrigierenden Delta-Regeln hat jedes Modell in diesem Kapitel verbessert, *wie* es einen Zustand fester Größe (engl. fixed-size state) aktualisiert. Keines kann jedoch mehr Informationen speichern, als die Größe seines Zustands zulässt. Fügt jemand eine Tabelle mit 64 Produktcodes und Preisen in einen Prompt ein und fragt nach 20 davon, kann ein Transformer auf seinen gesamten KV-Cache zugreifen und genau die gesuchten Token abrufen. Ein rein rekurrentes Modell verfügt nur über das, was es in seinem Zustand fester Größe verdichtet hat, bevor es die abgefragten Einträge kannte. Beim Abruf aus dem Kontext (engl. in-context retrieval) stoßen Sequenzmodelle mit fester Zustandsgröße an eine harte Grenze, etwa beim Few-Shot-Prompting, bei der retrievalgestützten Generierung und beim Bearbeiten von Code.

<!--en:0346fd25-->
Rekurrente Sprachmodelle im praktischen Einsatz lösen das Problem pragmatisch. Zwischen die rekurrenten Blöcke schieben sie einige wenige Schichten mit vollständiger Attention. Wir leiten zunächst zwei grundlegende Schranken für den Abruf aus einem Zustand fester Größe her. Die eine ergibt sich aus der Anzahl der Bits, die andere ist die geometrische untere Schranke von Welch für die Überlappung der Schlüssel. Wir vergleichen reines `GatedDeltaNet`, reine Attention und ein Hybridmodell `GA` beim assoziativen Abruf mit mehreren Abfragen (engl. multi-query associative recall) und auf WikiText-103 auf Byte-Ebene. Dann untersuchen wir, was sich ändert, wenn die kurze tiefenweise Faltung (engl. depthwise convolution) entfällt, die benachbarte Schlüssel- und Werttoken verknüpft. Wir beziffern die Einsparungen beim KV-Cache während der Inferenz für eine Kontextlänge von 256K Token und vergleichen alle Sequenzspeicher dieses Kapitels.

<!--en:ddc7e40e-->
## Grenzen eines Zustands fester Größe

<!--en:91d6275e-->
### Bits zählen

<!--en:d2b37c26-->
Nach dem Schubfachprinzip hat ein Zustand aus $M$ Zahlen mit je $b$ Bit höchstens $2^{Mb}$ verschiedene Konfigurationen. Um alle $V^k$ möglichen Sequenzen aus $k$ Token eines Vokabulars der Größe $V$ unterscheiden und wiedergeben zu können, muss der Zustand folgende Bedingung erfüllen :cite:`Jelassi.Brandfonbrener.Kakade.ea.2024`:

<!--en:b2c5104b-->
Die maximale Anzahl von Token, die sich aus einem Zustand fester Größe abrufen lassen, wächst also linear mit der Zustandsgröße $M$ und ist völlig unabhängig von der Kontextlänge $T$. Der KV-Cache eines Transformers hingegen wächst linear mit $T$ und erreicht keine Sättigung.

<!--en:515f812f-->
In der Praxis stoßen lineare Matrixspeicher an eine geometrische Grenze, lange bevor sie ihre Speicherkapazität in Bit ausschöpfen. Im folgenden Test zum assoziativen Abruf speichert eine Sequenz bis zu 64 Paare. Die Schlüssel stammen aus 128 Symbolen, die Werte aus 4 Symbolen. Das entspricht höchstens $128 \times 2 = 256$ Bit für die Wertetabelle oder $64 \times 2 = 128$ Bit für die 64 Werte. Zwei `GatedDeltaNet`-Schichten mit je 8 Köpfen und $4 \times 4$-Zuständen im Format float32 fassen dagegen $256 \times 32 = 8{,}192$ Bit, also mehr als $30\times$ so viele wie nötig. Warum verliert ein Matrixspeicher Paare, lange bevor er alle seine Bits ausschöpft?

<!--en:30455f5b-->
### Überlappende Schlüssel

<!--en:f3d79c22-->
Ein Matrixkopf $\mathbf{S} \in \mathbb{R}^{d_v \times d_k}$ lässt sich additiv aktualisieren (:numref:`sec_matrix-state`) oder mit der Delta-Regel in Richtung der Lösung nach der Methode der kleinsten Quadrate (:numref:`sec_deltanet`). In beiden Fällen setzt das exakte lineare Auslesen $\mathbf{S}\mathbf{k}_j = \mathbf{v}_j$ für beliebige Werte voraus, dass die gespeicherten Schlüssel linear unabhängig sind. In $\mathbb{R}^{d_k}$ gibt es höchstens $d_k$ linear unabhängige Vektoren. Sobald die Anzahl der gespeicherten Schlüssel $m$ die Kopfdimension $d_k$ übersteigt, *müssen* sich die Schlüssel überlappen, wie auch immer das Training sie wählt.

<!--en:7ccf6cc0-->
**Proposition (Welch-Schranke).** Seien $\mathbf{k}_1, \ldots, \mathbf{k}_m$ Einheitsvektoren in $\mathbb{R}^{d_k}$ mit $m > d_k$. Dann gilt für den Mittelwert der quadrierten inneren Produkte über alle Paare verschiedener Indizes $i \neq j$ folgende untere Schranke :cite:`Welch.1974`:

<!--en:3abeca33-->
**Beweis.** Sei $\mathbf{G} = \sum_{i=1}^m \mathbf{k}_i \mathbf{k}_i^\top \in \mathbb{R}^{d_k \times d_k}$. Die Spur dieser Matrix beträgt $\sum_{i=1}^m \|\mathbf{k}_i\|^2 = m$, also ist die Summe ihrer $d_k$ Eigenwerte gleich $m$. Nach der Cauchy-Schwarz-Ungleichung ist die Summe der quadrierten Eigenwerte, also $\|\mathbf{G}\|_F^2 = \sum_{i,j} (\mathbf{k}_i^\top \mathbf{k}_j)^2$, mindestens $m^2 / d_k$. Ziehen wir die $m$ Diagonalterme $(\mathbf{k}_i^\top \mathbf{k}_i)^2 = 1$ ab, bleibt mindestens $m^2 / d_k - m = m(m - d_k)/d_k$ für die Summe über die $m(m - 1)$ Paare außerhalb der Diagonale. $\blacksquare$

<!--en:832c8baa-->
Bei einer Kopfdimension von $d_k = 4$ haben $m = 64$ Schlüssel der Norm eins eine mittlere quadratische Überlappung von mindestens $60 / (4 \times 63) \approx 0.24$. Das unterscheidet sich kaum vom Wert für zufällige Einheitsvektoren ($1/d_k = 0.25$). Erhöhen wir die Kopfbreite auf $d_k = 16$, sinkt die Welch-Schranke auf $48 / (16 \times 63) \approx 0.05$.

<!--en:77e4b707-->
Daraus ergeben sich zwei Vorhersagen, die wir beim assoziativen Abruf mit mehreren Abfragen überprüfen können:

<!--en:26dc9108-->
1. Ein reiner Stapel aus `GatedDeltaNet`-Blöcken ruft weniger zuverlässig ab, sobald die Anzahl der gespeicherten Paare $n$ die Kapazität seines Zustands übersteigt. Vervierfachen wir die Anzahl der Zustandseinträge pro Schicht (von $8 \times 4^2 = 128$ auf $2 \times 16^2 = 512$ bei fester Breite $d = 32$), setzt der Rückgang erst bei ungefähr $4\times$ so vielen Paaren ein.

<!--en:58a79148-->
2. Ersetzen wir nur einen der beiden rekurrenten Blöcke durch vollständige kausale Attention (`GA`), verschwindet der kapazitätsbedingte Rückgang ganz, weil die Attention-Schicht jedes Token in ihrem Cache behält.

<!--en:cd1044c9-->
## Abruf mit einer Attention-Schicht

<!--en:df9a25d5-->
### Vier Modelle

<!--en:4d445b3a-->
Wir testen vier `d2l.SeqLM`-Modelle mit je zwei Blöcken ($d = 32$) beim assoziativen Abruf mit mehreren Abfragen (`num_keys=128`, `num_values=4` und $n \in \{4, 8, 16, 32, 64\}$ gespeicherte Paare):

<!--en:f339b3bb-->
- **`Gated DeltaNet` (`GG`):** 8 Köpfe der Größe 4 ($128$ Zustandseinträge pro Schicht).

<!--en:16ef808a-->
- **`Gated DeltaNet, 4x state` (`GG, num_heads=2`):** 2 Köpfe der Größe 16 ($512$ Zustandseinträge pro Schicht bei gleicher Parameteranzahl).

<!--en:febf868e-->
- **`attention` (`AA`):** kausale Attention mit 2 Köpfen (`d2l.AttentionMixer`), derselben tiefenweisen kausalen Faltung der Breite 4 auf $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ und ohne Positionskodierung (`rope=False`).

<!--en:5c822ac4-->
- **`hybrid` (`GA`):** `GatedDeltaNet` im ersten Block, gefolgt von `AttentionMixer` im zweiten.

<!--en:214b301b-->
### Abruf bei steigender Paaranzahl

<!--en:dfe57736-->
Für jede Architektur verwenden wir zwei Zufallsstartwerte. In jedem Lauf trainieren wir 2000 Schritte mit `d2l.train_recall` und messen die Genauigkeit für jede Paaranzahl.

<!--en:04afeb82-->
Für jede Variante mitteln wir die Abrufkurven über beide Läufe und tragen die Genauigkeit gegen die Anzahl der gespeicherten Paare $n \in \{4, 8, 16, 32, 64\}$ auf:

<!--en:a5d956ab-->
Beide Vorhersagen treffen zu:

<!--en:aa25a4c0-->
1. **Zustandskapazität von `GatedDeltaNet`:** Bei 128 Zustandseinträgen pro Schicht (`GG`) sinkt die Abrufgenauigkeit von etwa $1.00$ bei 4 Paaren auf einen Wert zwischen $0.6$ und $0.7$ bei 32 Paaren und auf etwa $0.5$ bei 64 Paaren. Vervierfachen wir den Zustand auf 512 Einträge pro Schicht (`Gated DeltaNet, 4x state`), bleibt die Genauigkeit bis einschließlich 32 Paaren nahe $1.00$. Auch bei 64 Paaren ruft das Modell die meisten Paare korrekt ab. Je nach Lauf liegt die Genauigkeit zwischen etwa $0.8$ und $1.0$.

<!--en:30b597ae-->
2. **Eine Attention-Schicht ermöglicht wieder exakten Abruf:** Sowohl `attention` (`AA`) als auch das Hybridmodell `hybrid` (`GA`) erreichen in jedem Lauf bei jeder Paaranzahl bis 64 eine Abrufgenauigkeit von $1.00$. Das entspricht den Ergebnissen von :citet:`Waleffe.Byeon.Riach.ea.2024` in größeren Experimenten.

<!--en:564d270c-->
### Die kurze Faltung

<!--en:b22823a2-->
Warum haben wir unseren rekurrenten Mischern und Attention-Mischern eine tiefenweise kausale Faltung der Breite 4 vorangestellt? Beim assoziativen Abruf kommt jeder Wert an der Tokenposition $t$ unmittelbar *nach* seinem Schlüssel an Position $t - 1$. Ohne lokale Mischung der Token (`conv_width=1`) sieht eine einzelne Schicht mit Matrixzustand im Schritt $t$ nur das Werttoken. Sie kann es daher nicht mit dem vorhergehenden Schlüsseltoken verknüpfen. Prüfen wir, was ohne die Faltung geschieht, indem wir `conv_width=1` in `GG` setzen:

<!--en:667dd67d-->
Entfernen wir die kausale Faltung der Breite 4, sinkt die Abrufgenauigkeit bei jeder Paaranzahl um rund $10$ bis $35$ Prozentpunkte. Selbst bei 4 Paaren liegt sie dann nur noch zwischen $0.8$ und $0.9$.

<!--en:a2021b28-->
## Sprachmodellierung bei gleicher Modellgröße

<!--en:13387905-->
Wie schneiden ein reines `GatedDeltaNet` (`GGGG`), ein 3:1-Hybridmodell (`GGGA`) und reine kausale `attention` (`AAAA`) in unserem Test auf WikiText-103 auf Byte-Ebene ab (`d2l.SeqLM`, 4 Blöcke der Breite $d = 256$)? Wir prüfen das zunächst mit den in diesem Kapitel verwendeten Fenstern von 128 Byte und dann mit Fenstern von 1024 Byte. Zeigt sich bei längerem Kontext ein Unterschied zwischen einem Zustand fester Größe und einem wachsenden Cache? In beiden Fällen trainieren wir 8000 Schritte lang, also länger als die 1000 Schritte der vorigen Abschnitte. Ein Modell lernt erst spät im Training, weit zurückliegenden Kontext zu nutzen. Die folgenden Verluste liegen daher unter denen der früheren Tabellen zur Sprachmodellierung und lassen sich nicht mit ihnen vergleichen.

<!--en:ce088511-->
### Kurze Fenster

<!--en:48587b4d-->
Zuerst laden wir den Datensatz `WikiText103` mit Fenstern von 128 Byte:

<!--en:e4be558e-->
Für einen fairen Vergleich brauchen wir eine weitere Variante. Mamba, lineare Attention mit Gate und Gated DeltaNet mischen benachbarte Token vor der Rekurrenz mit einer kausalen Faltung der Breite 4. Der Attention-Mischer in unseren bisherigen Textversuchen kam ohne Faltung aus. Ein Vorsprung von `GatedDeltaNet` gegenüber `attention` könnte also auf die Faltung zurückgehen statt auf den Speicher. Wie im Abrufexperiment stellen wir der Attention in einer vierten Variante dieselbe Faltung voran.

<!--en:52d2eb8f-->
Dazu erzeugt `text_model(layout, d)` ein `d2l.SeqLM`-Modell mit vier Blöcken. Die vier Buchstaben in der Zeichenkette `layout` geben die Mischer an: `'G'` (`GatedDeltaNet`, 4 Köpfe), `'A'` (`AttentionMixer`, 8 Köpfe), `'C'` (dieselbe Attention nach einer kausalen Faltung der Breite 4) und `'R'` (eine `GRU` für die längeren Fenster weiter unten):

<!--en:d3490b2b-->
Wir trainieren `'GGGG'`, `'GGGA'`, `'AAAA'` und `'CCCC'` bei Breite $d = 256$ mit `d2l.text_table` in je zwei Läufen für jeweils 8000 Schritte:

<!--en:83039a31-->
Bei Fenstern von 128 Byte liegen drei der vier Varianten am Ende fast gleichauf. `Gated DeltaNet` erreicht etwa $1.16$ Nat pro Byte, das Hybridmodell `GGGA` und Attention mit Faltung jeweils etwa $1.15$. Nur `attention` ohne Faltung liegt mit einem Rückstand von etwa $0.03$ bei $1.18$. Bei dieser Fensterlänge bestimmt die lokale Syntax den Verlust. Nach den ersten 16 Byte eines Fensters ändert er sich kaum noch (:numref:`subsec_deep-rnn`). Die Tabelle zeigt also vor allem, wie gut ein Mischer benachbarte Bytes kombiniert.

<!--en:43b6a5d3-->
Die Faltung erklärt somit den gesamten Rückstand der reinen Attention. Mit Faltung liegt Attention gleichauf mit dem Hybridmodell. Gegenüber `Gated DeltaNet` beträgt ihr Vorsprung höchstens $0.01$. Das entspricht etwa dem Unterschied zwischen zwei Läufen derselben Variante, sodass wir daraus keine Rangfolge ableiten. Bei kurzen Fenstern hat weder der Matrixspeicher noch der Cache einen Vorteil, der über die lokale Mischung hinausgeht. Auch die einzelne Attention-Schicht des Hybridmodells bringt keinen messbaren Gewinn. Sie ist dort wichtig, wo wir weit zurückliegende Inhalte exakt abrufen müssen (Übung :exref:`ex_hybrids-recall-in-text`). Spielt sie auch bei echten Texten eine Rolle, sobald die Fenster lang genug sind?

<!--en:d343be28-->
### Lange Fenster

<!--en:ebd23841-->
Fenster von 128 Byte können weder den Nachteil eines Zustands fester Größe gegenüber einem wachsenden Cache zeigen noch den eines Vektorzustands gegenüber einem Matrixspeicher. Deshalb trainieren wir vier Mischer mit Fenstern von 1024 Byte: einen Stapel aus `GRU`-Schichten (`'RRRR'`), `Gated DeltaNet`, das Hybridmodell `GGGA` und Attention mit Faltung. Wir verringern die Batchgröße von 64 auf 8. So sieht das Modell in jedem Schritt gleich viele Bytes wie zuvor und in 8000 Schritten dieselbe Textmenge.

<!--en:70047655-->
Der Mittelwert über ein Fenster verdeckt, an welchen Positionen das jeweilige Modell gewinnt. Nutzt es den gesamten Kontext, verbessert es sich mit fortschreitender Position weiter. Fasst es nur die jüngsten Bytes zusammen, hört es früh auf, sich zu verbessern. Um das zu unterscheiden, mitteln wir den Validierungsverlust beider Läufe über Positionsbereiche $[a, 2a)$. Ihre Länge verdoppelt sich jeweils, von Byte 16–31 bis Byte 512–1023:

<!--en:1ca8c683-->
Längere Fenster helfen jedem Modell, das wir mit beiden Fensterlängen trainiert haben. `Gated DeltaNet` erreicht am Ende Werte zwischen $1.11$ und $1.12$, das Hybridmodell zwischen $1.07$ und $1.08$, gegenüber $1.16$ und $1.15$ bei Fenstern von 128 Byte. Attention mit Faltung erreicht in den meisten Läufen etwa $1.07$. Etwa jeder fünfte Lauf endet dagegen nahe $1.12$. Den Grund sehen wir weiter unten. Die `GRU` erreicht $1.09$. Sie ist mit Abstand die langsamste Variante. Pro Schritt braucht sie etwa doppelt so lange wie die nächstlangsamere Variante oder noch länger, weil sie als einzige das Fenster Schritt für Schritt verarbeitet.

<!--en:481c9327-->
Die drei Varianten, die bei kurzen Fenstern gleichauf lagen, unterscheiden sich nun. Die Spalten der Tabelle zeigen, wo ihre Ergebnisse auseinandergehen. Von den ersten 128 Byte eines Fensters bis zu den letzten 512 sinkt der Verlust der `GRU` um etwa $0.07$ und der von `Gated DeltaNet` um etwa $0.08$. Das Hybridmodell gewinnt etwa $0.13$, Attention in den meisten Läufen etwa $0.16$. Die Abbildung zeigt genauer, wo der Gewinn einsetzt. Bis Byte 127 halten die Zustände fester Größe mit: Die `GRU` hat in jedem Bereich den niedrigsten Verlust, Attention in den ersten Bereichen den höchsten. Das Hybridmodell lässt sich von reinem `Gated DeltaNet` nicht unterscheiden. Nach Byte 128 gewinnen die `GRU` und `Gated DeltaNet` bis zum letzten Bereich nur noch etwa $0.03$. Der Verlust des Hybridmodells sinkt weiter. Ab Byte 128 liegt er unter dem von `Gated DeltaNet` und ab Byte 256 auch unter dem der `GRU`. Bei Byte 512–1023 beträgt der Vorsprung gegenüber `Gated DeltaNet` etwa $0.05$ und gegenüber der `GRU` etwa $0.02$. Diesen gesamten Vorsprung verdankt das Hybridmodell der einen Attention-Schicht in seinem vierten Block.

<!--en:7de36247-->
Bei Attention sinkt der Verlust am stärksten, allerdings nicht in jedem Lauf. Meist erreicht sie am Ende bei Byte 512–1023 einen Verlust von $1.03$ bis $1.04$. Damit liegt sie etwa $0.02$ vor dem Hybridmodell und schneidet von den vier Modellen am besten ab. Etwa jeder fünfte Lauf bleibt zurück. Er endet bei insgesamt etwa $1.12$ und im letzten Bereich bei etwa $1.10$, also nicht besser als `Gated DeltaNet`. Die Abbildung zeigt den Mittelwert der beiden Läufe aus der Tabelle. Bleibt einer davon zurück, endet die Attention-Kurve daher oberhalb der Kurve des Hybridmodells.

<!--en:4d86a6c5-->
Die beiden Arten von Läufen unterscheiden sich durch ein einziges Ereignis. In den ersten rund tausend Schritten verläuft der Trainingsverlust in allen Läufen gleich. In den meisten fällt er dann zwischen Schritt 1000 und 5000 plötzlich um etwa $0.05$. Bei einem zurückbleibenden Lauf nimmt er gleichmäßig ab, ohne diesen Sprung. Der Sprung markiert den Moment, in dem Attention das Kopieren lernt. Text enthält Wiederholungen: Ein Name, ein Datum oder eine Wendung kommt oft innerhalb eines Kilobytes erneut vor. Findet ein Modell eine frühere Stelle mit den gerade gelesenen Bytes und sagt das dort folgende Byte vorher, gewinnt es bei jeder solchen Wiederholung. :citet:`Olsson.Elhage.Nanda.ea.2022` nennen diese Schaltung einen *Induktionskopf*. Wir kennen ihn aus :numref:`sec_what-attention-computes`. Seine Wirkung lässt sich direkt überprüfen. Dazu betrachten wir Fenster, die denselben Abschnitt von 512 Byte zweimal enthalten. Nach dem Sprung beträgt der Verlust für die zweite Kopie nur das $0.6$- bis $0.8$-Fache des Verlusts für die erste. Bei einem zurückbleibenden Lauf ist der Verlust für beide Kopien über alle 8000 Schritte fast gleich. In den ersten 128 Byte eines Fensters gibt es wenig zu kopieren. Dort lassen sich die beiden Arten von Läufen nicht unterscheiden.

<!--en:4ae70aed-->
Ein zurückbleibender Lauf ist stabil. Sein Verlust divergiert nicht, und seine Gradienten sind so groß wie die der anderen Läufe. Er ist lediglich spät dran. In welchem Schritt sich die Schaltung bildet, hängt vom Zufallsstartwert ab. Nach etwa 6000 Schritten ist die Lernrate im Kosinusplan nur noch gering. Hat sich die Schaltung bis dahin nicht gebildet, bildet sie sich gar nicht mehr. Die kurze Faltung hilft hier wie bei der Abrufaufgabe: Ohne sie bleibt der Sprung in den meisten Läufen bis Schritt 5000 aus. Wir haben auch andere Lernraten, den Verzicht auf Gewichtszerfall und eine andere Anzahl von Attention-Köpfen erprobt. Nur in einem Fall blieben die Sprünge seltener aus, in keinem traten sie immer auf. Solche plötzlichen Rückgänge sind typisch, wenn ein Netz eine Schaltung aus Teilen bilden muss, die für sich genommen wenig nützen. Deshalb lohnt es sich, zwei Architekturen mit mehreren Zufallsstartwerten zu trainieren, bevor wir sie nach ihrer Güte ordnen.

<!--en:eafdd120-->
Bei langem Kontext und ausreichend Training zeigt sich im Text also eine schwächere Form dessen, was die Abrufaufgabe gezeigt hat. Der wachsende Cache gewinnt am meisten durch weit zurückliegende Bytes. Eine einzelne Attention-Schicht neben drei rekurrenten Schichten erzielt einen guten Teil dieses Gewinns. Ein Zustand fester Größe allein gewinnt am wenigsten. In Fenstern von 128 Byte fehlen die weit zurückliegenden Bytes. Deshalb kann die Tabelle für kurze Fenster diesen Unterschied nicht zeigen.

<!--en:2f913eb3-->
## Speicherbedarf bei der Inferenz

<!--en:e1f1db72-->
Bei der autoregressiven Generierung bestimmt der Speicherbedarf pro Sequenz sowohl die maximale Batchgröße im Betrieb als auch die durch die Speicherbandbreite begrenzte Dekodiergeschwindigkeit:

<!--en:15134214-->
- **Schicht mit Grouped-Query-Attention** ($n_\textrm{kv} = 8$ Schlüssel-Wert-Köpfe der Dimension $d_\textrm{head} = 128$, Gleitkommazahlen mit 16 Bit): Sie speichert $2 \times n_\textrm{kv} \times d_\textrm{head} \times 2 = 4\textrm{ KB}$ *pro Token*. Das ergibt $512\textrm{ MB}$ pro Schicht bei $T = 128\textrm{K}$ Token.

<!--en:430070c6-->
- **Rekurrente Schicht oder Schicht mit Matrixzustand** (z. B. Mamba-2 mit der Breite $4{,}096$, mit $8{,}192$ Kanälen und der Zustandsdimension $128$ bei Gleitkommazahlen mit 16 Bit): Sie belegt stets $8{,}192 \times 128 \times 2 = 2\textrm{ MB}$, *unabhängig von der Kontextlänge*. Der KV-Cache der Attention-Schicht übersteigt diese Größe schon nach $512$ Token.

<!--en:b79cac3d-->
In einer hybriden Architektur wächst der KV-Cache nur in den Attention-Schichten (:numref:`fig_hy-cache`). Ein Modell mit 32 Schichten, davon 4 Attention-Schichten (jede achte Schicht, wie bei Jamba :cite:`Lieber.Lenz.Bata.ea.2024`), senkt den Cachebedarf bei 256K Token von $32\textrm{ GB}$ auf $4\textrm{ GB}$. 3:1-Hybridmodelle mit Gated DeltaNet wie Qwen3-Next :cite:`Qwen.Team.2025` und Kimi Linear :cite:`Kimi.Team.2025b` verkleinern den KV-Cache um den Faktor $4\times$.

<!--en:13c29e5e-->
![Speicherbedarf pro Sequenz bei der Generierung in Abhängigkeit von der Kontextlänge. Das Modell hat 32 Schichten mit Grouped-Query-Attention (8 Schlüssel-Wert-Köpfe der Dimension 128) oder rekurrenten Zuständen wie in Mamba-2 und verwendet 16-Bit-Zahlen. Bei 256K Token erreicht der Cache des Transformers 32 GB. Ein rein rekurrentes Modell bleibt bei jeder Länge bei 64 MB. Ein Hybridmodell mit 4 Attention-Schichten benötigt ein Achtel des KV-Caches des Transformers.](../img/mdl-modernrnn-hybrid-cache.svg)

<!--en:a459749c-->
## Die Speicher dieses Kapitels im Vergleich

<!--en:26f760e5-->
:numref:`tab_hy-memories` vergleicht alle in diesem Kapitel entwickelten Sequenzmischer miteinander sowie mit kausaler Attention und dem Hybridmodell. Dabei bezeichnet $d$ die Modellbreite, $N$ die Zustandsdimension, $H = 2d$ die Anzahl der Kanäle nach der Erweiterung in Mamba, $h$ die Anzahl der Köpfe und $T$ die Sequenzlänge.

<!--en:2a9162ec-->
:Die Speicher dieses Kapitels. Die Spalte „Text“ zeigt den Validierungsverlust in Nat pro Byte auf WikiText-103 auf Byte-Ebene, gemessen in Fenstern von 128 Byte mit zwei Läufen pro Modell. Die Parameteranzahl steht in Klammern. Die Modelle dieses Abschnitts haben 8000 Trainingsschritte durchlaufen, die übrigen 1000. Nur Verluste bei gleicher Schrittzahl sind vergleichbar. Bei Fenstern von 1024 Byte führt nach 8000 Schritten in den meisten Läufen Attention mit Faltung vor dem Hybridmodell, der GRU und Gated DeltaNet.

<!--en:a5ea812f-->
| Speicher | Zustand pro Schicht bei der Generierung | Training | Synthetische Aufgabe | Text |

<!--en:88b2912a-->
| LSTM, GRU | $2d$, $d$ | Schritt für Schritt | Gatter mit Startwert nahe eins behalten ein Symbol 200 Schritte lang, bei einem Startwert von 0.5 verlieren sie es spätestens nach 50 Schritten | 1.29 bis 1.30 (4.3 Mio., 3.8 Mio.) |

<!--en:0ae789be-->
| minGRU | $d$ | Scan | nach 100 Schritten auf Zufallsniveau, mit den Zeitskalen von S4D gelingt der Abruf noch nach 1000 Schritten | 1.34 bis 1.35 (2.8 Mio.) |

<!--en:25e70d9e-->
| S4D | $dN$ | Faltung oder Scan | Abruf nach 1000 Schritten, beim selektiven Kopieren höchstens jeder fünfte Wert richtig | nicht trainiert |

<!--en:9a79917b-->
| Mamba | $HN$ | Scan | 0.98 oder mehr beim selektiven Kopieren | 1.30 (4.0 Mio.) |

<!--en:c50f8233-->
| Lineare Attention mit Gate | $d^2/h$ | blockweise | bei 128 Einträgen etwa die Hälfte von 64 Paaren, bei 512 mehr | nicht trainiert |

<!--en:b595eddd-->
| Gated DeltaNet | $d^2/h$ | blockweises Lösen eines Dreieckssystems | bei neu belegten Schlüsseln mindestens 0.95 gegenüber etwa 0.9 bei linearer Attention mit Gate | 1.16 nach 8000 Schritten (3.6 Mio.) |

<!--en:d7787a68-->
| Attention | $2dT$ | fusionierte Attention über alle Positionen | jedes Paar bei jeder Paaranzahl | 1.38 bis 1.39, nach 8000 Schritten 1.18, mit Faltung der Breite 4 dann 1.15 (3.3 Mio.) |

<!--en:932de43b-->
| Hybridmodell | rekurrente Zustände plus $2dT$ pro Attention-Schicht | beides | jedes Paar, mit Attention in einem von zwei Blöcken | 1.15 nach 8000 Schritten (3.5 Mio.) |

<!--en:4dd42c83-->
Aus dem Vergleich der Spalten ergeben sich folgende Abwägungen für den Entwurf:

<!--en:bf16ff18-->
- Für **stromsparende Verarbeitung von Datenströmen oder kontinuierliche Verarbeitung von Sensor- und Audiodaten** eignen sich klassische Rekurrenzen mit Gattern oder selektive Zustandsraummodelle, sofern kein exakter Abruf beliebiger früherer Token nötig ist. Sie kommen mit einem kompakten Zustand der Größe $O(1)$ aus.

<!--en:b787eadd-->
- Für **Basismodelle mit langem Kontext** kombinieren wir blockparallel berechnete Matrixrekurrenzen mit Delta-Regel (`GatedDeltaNet`) oder selektive Zustandsraummodelle (`Mamba-2`) mit wenigen Schichten mit vollständiger Attention (Verhältnis $1{:}3$ bis $1{:}7$). So verbinden wir paralleles Training in linearer Zeit mit einem um den Faktor $4\times$–$8\times$ kleineren KV-Cache bei der Inferenz und exaktem inhaltsadressiertem Abruf.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ca8a0f99-->
Ein Zustand fester Größe aus $M$ Zahlen mit je $b$ Bit kann höchstens $2^{Mb}$ verschiedene Vorgeschichten unterscheiden (es muss $Mb \ge k \log_2 V$ gelten, um $k$ Token aus einem Vokabular $V$ abzurufen). Bei Schichten mit Matrixzustand begrenzt die Geometrie die Kapazität schon früher. Nach der Welch-Schranke haben $m > d_k$ Schlüssel der Norm eins in $\mathbb{R}^{d_k}$ eine mittlere quadratische Überlappung von mindestens $(m - d_k)/(d_k(m - 1))$. Daher stören sich die Schlüssel sowohl bei additiven Aktualisierungen als auch bei der Delta-Regel. Beim assoziativen Abruf mit mehreren Abfragen lässt ein reines `GatedDeltaNet`-Modell nach, sobald die Paaranzahl die Kapazität seiner Zustandseinträge übersteigt. Viermal so viele Zustandseinträge erhöhen die Kapazität im gleichen Verhältnis. Ersetzen wir einen einzigen rekurrenten Block durch vollständige Attention (`GA`), erreichen wir bei jeder Paaranzahl wieder $100\%$ Abrufgenauigkeit. Eine kausale Faltung der Breite 4 vor dem Mischer ist entscheidend, damit eine einzige Schicht benachbarte Schlüssel- und Werttoken verknüpfen kann. Bei Text auf Byte-Ebene fallen die Unterschiede geringer aus und hängen von der Fensterlänge ab. Bei Fenstern von 128 Byte liegen `Gated DeltaNet`, das Hybridmodell und Attention mit Faltung nach 8000 Schritten höchstens etwa $0.01$ Nat pro Byte auseinander. Der Rückstand der reinen Attention von etwa $0.03$ verschwindet, sobald sie dieselbe Faltung erhält. Bei Fenstern von 1024 Byte zeigen sich die Unterschiede nach Byte 128: In den meisten Läufen gewinnt Attention am meisten durch weit zurückliegenden Kontext, gefolgt vom Hybridmodell. Reines `Gated DeltaNet` gewinnt am wenigsten. Bei der Generierung brauchen im Hybridmodell nur die Attention-Schichten einen wachsenden KV-Cache. Der dafür benötigte Speicher bei der Inferenz sinkt proportional mit dem Anteil der Attention-Schichten.

<!--en:2294243a-->
## Übungen

<!--en:4cd5dd58-->
1. **Die Schranke aus der Bitzahl.** Ein Modell liest eine Folge von $k$ Token aus einem Vokabular der Größe $V$ und soll sie anschließend wiedergeben.
    1. Beweisen Sie :eqref:`eq_hy-counting` für ein Modell, dessen Zustand zwischen Lesen und Schreiben aus $M$ Zahlen mit je $b$ Bit besteht und dessen Ausgabe eine deterministische Funktion dieses Zustands ist.
    1. Nehmen Sie nun an, dass das Modell mit einer Wahrscheinlichkeit von höchstens $\epsilon$ eine falsche Folge ausgeben darf, wenn die Folge gleichverteilt gezogen wird. Zeigen Sie, dass $Mb \ge k \log_2 V + \log_2(1 - \epsilon)$ gilt. (Hinweis: Zählen Sie die Folgen, die sich aus jedem Zustand korrekt dekodieren lassen.)
    1. Wie viele Token eines Vokabulars der Größe 50 000 kann ein Modell mit 48 Schichten höchstens kopieren, wenn jede Schicht einen Mamba-2-Zustand von 2 MB in 16-Bit-Zahlen speichert? Warum kommt in der Praxis kein Modell dieser Größe an die Schranke heran?

<!--en:5b29ceba-->
1. **Speicher für viele Nutzer.** Auf einer GPU mit 80 GB läuft ein Modell mit 32 Schichten und den Cachegrößen aus :numref:`fig_hy-cache`. Die Gewichte belegen 16 GB des GPU-Speichers.
    1. Wie viele Sequenzen mit je 128K Token kann die GPU gleichzeitig verarbeiten, wenn auf ihr der Transformer, das Hybridmodell mit 4 Attention-Schichten oder das rekurrente Modell läuft?
    1. Sowohl Grouped-Query-Attention mit weniger Schlüssel-Wert-Köpfen als auch 8-Bit-Zahlen für den Cache halbieren jeweils den Cachebedarf. Wenden Sie beide Änderungen auf den Transformer und auf das Hybridmodell an und berechnen Sie die Anzahlen erneut. Welche der beiden Änderungen könnte auch die rekurrenten Zustände des Hybridmodells verkleinern? Welchen Anteil am Speicherbedarf des Hybridmodells haben diese Zustände nach beiden Änderungen?

<!--en:f1e0b0a0-->
1. **Überlappung und Training.**
    1. Um wie viel besser als zufällige Schlüssel können die Schlüssel eines Kopfes durch Training werden? Teilen Sie die Schranke :eqref:`eq_hy-welch` durch $1/d_k$, die mittlere quadratische Überlappung zufälliger Schlüssel der Norm eins. Berechnen Sie das Verhältnis für 64 Schlüssel in den Dimensionen 4, 8, 16 und 32. Bei welchen dieser Köpfe können gelernte Schlüssel deutlich besser sein als zufällige?
    1. [code] Trainieren Sie das kleine Modell mit Gated DeltaNet aus der Versuchsreihe mit 4000 und 8000 Aktualisierungen (`num_updates` von `d2l.train_recall`). Nähert sich seine Abrufgenauigkeit bei 32 und 64 Paaren der des Hybridmodells? Was sagt das Ergebnis darüber aus, ob es dem kleinen Modell an Training oder an Kapazität fehlt?
    1. [code] Ziehen Sie die Schlüssel einmal aus 64 und einmal aus 512 möglichen Schlüsseln (`num_keys`, mit entsprechend angepasstem Vokabular von `recall_model`). Eine Sequenz speichert weiterhin höchstens 64 Paare. Warum kann die Anzahl möglicher Schlüssel die Abrufgenauigkeit überhaupt beeinflussen?

<!--en:98942848-->
1. [code] **Wie viel Attention und an welcher Stelle?**
    1. Trainieren Sie das Hybridmodell `AG` mit Attention im ersten Block und vergleichen Sie seine Abrufgenauigkeit mit der von `GA`. Spielt die Position der Attention-Schicht eine Rolle? Überlegen Sie, welche Schicht jeden Schlüssel mit dem darauffolgenden Wert verknüpfen kann.
    1. Trainieren Sie die Modelle `GGGG`, `GGGA` und `GAGA` mit je vier Blöcken und gleicher Breite. Verschieben zwei zusätzliche rekurrente Schichten die Abrufkurve ebenso weit wie ein viermal so großer Zustand pro Schicht? Gelingt es mit einer Attention-Schicht unter vier Schichten weiterhin, jedes Paar abzurufen?

<!--en:4eaa1cad-->
1. [code] **Abruf im Text.** Markieren Sie in den Validierungsfenstern von WikiText-103 auf Byte-Ebene alle Positionen, deren Byte eine Folge von mindestens acht Bytes abschließt, die im selben Fenster schon früher vorkam. :exlabel:`ex_hybrids-recall-in-text`
    1. Vergleichen Sie für die trainierten Modelle mit Gated DeltaNet, hybrider Architektur und Attention (`trained`) den Verlust an den markierten Positionen mit dem an allen anderen Positionen. Zeigt sich der Vorteil von Attention beim Abruf dort, wo sich Text wiederholt?
    1. Wiederholen Sie den Versuch mit Fenstern von 512 Byte (`num_steps=512` und weniger Fenstern pro Schritt). Wie verändert sich der Anteil markierter Positionen und damit der Unterschied zwischen den Modellen?

<!--en:4e181a9c-->
1. [code] **Wie sich die Schaltung zum Kopieren bildet.** Erstellen Sie Validierungsfenster mit 1024 Byte, deren zweite Hälfte die erste wiederholt. Trainieren Sie Attention mit Faltung (`'CCCC'`) auf gewöhnlichen Fenstern von 1024 Byte. Erfassen Sie alle 250 Schritte den Trainingsverlust und das Verhältnis des Verlusts auf der zweiten Hälfte der Fenster mit Wiederholung zum Verlust auf der ersten Hälfte.
    1. In welchem Schritt fällt das Verhältnis unter 1? Sinkt der Trainingsverlust im selben Schritt?
    1. Wiederholen Sie den Versuch mit mehreren Zufallsstartwerten. Wie stark variiert der Schritt, in dem der Verlust sprunghaft sinkt? Wie verläuft ein Lauf, bei dem dieser Sprung innerhalb von 8000 Schritten ausbleibt?

<!--en:8eef31cf-->
[Diskussion](https://d2l.discourse.group/t/36785)

<!--en:f2eef1a1-->
Hybride Architekturen aus Rekurrenz und globaler Attention<br>**Kapazitätsschranken aus Informationstheorie und Geometrie nach Welch · Hybridmodelle aus Gated DeltaNet und Attention · Verknüpfung durch lokale Faltung · Wachstum des KV-Caches**

<!--en:2ecef890-->
::: {.slide title="Warum jeder Zustand fester Größe den Abruf aus dem Kontext begrenzt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:72330d61-->
- **Verdichtung vor der Abfrage:** Ein rekurrenter Zustand fester Größe $\mathbf{S}_t$ muss alle $64$ Paare im Prompt verdichten, *bevor* bekannt ist, welche $20$ Schlüssel abgefragt werden.

<!--en:f385eadb-->
- **Abruf durch Attention nach der Abfrage:** Vollständige Attention hält die ursprüngliche KV-Tabelle für den exakten Direktzugriff vor. Die Speicherkosten betragen $O(T)$.

<!--en:64d6b809-->
- **Prinzip des Hybridmodells:** Auf je 3–7 rekurrente Schichten kommt 1 Attention-Schicht.

<!--en:477dfd88-->
::: {.slide title="Grenze 1: Die informationstheoretische Schranke aus der Bitzahl"}

<!--en:9225a70c-->
[Kapazitätsgrenze nach dem Schubfachprinzip]{.kicker}

<!--en:84e1b112-->
- **Reichlich Bits verfügbar:** Unser zweischichtiges `GG`-Modell ($8$ Köpfe mit je $4 \times 4$ Einträgen) speichert **$8{,}192$ Bit im Format `float32`**. Für $64$ Antworten mit je vier möglichen Werten reichen **$128$ Bit** ($>30\times$ weniger).

<!--en:81f3e58b-->
- **Flaschenhals der linearen Algebra:** Matrixspeicher erreichen Rang $d_k$, lange bevor sie die verfügbaren Bits der Gleitkommazahlen ausschöpfen.

<!--en:af9ab242-->
::: {.slide title="Grenze 2: Lineare Unabhängigkeit und Welch-Schranke für die Überlappung"}

<!--en:67ff6741-->
[Geometrische Kapazitätsgrenze]{.kicker}

<!--en:43fc1db2-->
- **Beschränkter Rang $d_k$:** Die Matrix $\mathbf{G} = \sum_{i=1}^m \mathbf{k}_i \mathbf{k}_i^\top \in \mathbb{R}^{d_k \times d_k}$ hat die Spur $m$ und den Rang $\le d_k$. Nach Cauchy-Schwarz gilt daher $\|\mathbf{G}\|_F^2 \ge m^2 / d_k$.

<!--en:ed7abb7a-->
- **Allgemeine untere Schranke:** Sie gilt für *jedes* lineare Auslesen eines Matrixspeichers $\mathbf{y} = \mathbf{S}\mathbf{q}$ (`GLA`, `DeltaNet`, `Longhorn`).

<!--en:50d16e95-->
::: {.slide title="Was die Welch-Schranke für gelernte Schlüsseleinbettungen vorhersagt"}

<!--en:76f3550d-->
[Von der Theorie zum Experiment]{.kicker}

<!--en:4e528ce1-->
- **Bei $d_k = 4, m = 64$:** Die untere Schranke von Welch ($0.2381$) beträgt **$95.2\%$** der Überlappung zufälliger Schlüssel ($1/4 = 0.2500$). Dem Gradientenabstieg bleibt kaum Spielraum, Störbeiträge zu verringern.

<!--en:9f7dd6d6-->
- **Bei $d_k = 16, m = 64$:** Die Welch-Schranke sinkt um den Faktor $5\times$ auf **$0.0476$**. Die erwartete Kapazität steigt auf $4\times$ ihren Ausgangswert, wenn sich die Kopfbreite bei fester Modellbreite $d = 32$ vervierfacht.

<!--en:f2fe10a9-->
::: {.slide title="Vier Architekturen mit je zwei Blöcken beim assoziativen Abruf"}

<!--en:9e700cb8-->
[Kontrollierter Versuchsaufbau]{.kicker}

<!--en:e9d725ef-->
- **Vier vergleichbare Modelle mit je zwei Blöcken ($d = 32$):**
  1. `GG` ($8$ Köpfe, zusammen: Kopfanzahl mal $4 \times 4 = 128$ Gleitkommazahlen pro Schicht).
  2. `GG, 4x state` ($2$ Köpfe, zusammen: Kopfanzahl mal $16 \times 16 = 512$ Gleitkommazahlen pro Schicht, gleiche Parameteranzahl).
  3. `AA` ($2$ Schichten mit vollständiger kausaler Attention).
  4. Hybridmodell `GA` (`GatedDeltaNet` $\to$ `Attention`, $50\%$ des KV-Caches).

<!--en:de42743a-->
::: {.slide title="Assoziativer Abruf mit mehreren Abfragen: 4 bis 64 Schlüssel-Wert-Paare"}

<!--en:64d6cfe8-->
[Abrufgenauigkeit nach Paaranzahl]{.kicker}

<!--en:a2c88975-->
Mittlere Abrufgenauigkeit über zwei Trainingsläufe ($2{,}000$ AdamW-Aktualisierungen, zyklischer Wechsel zwischen den Paaranzahlen $n \in \{4, 8, 16, 32, 64\}$, Zufallsniveau $= 0.25$):

<!--en:6b690235-->
::: {.slide title="Eine Attention-Schicht beseitigt die Kapazitätsgrenze, der vierfache Zustand verschiebt sie"}

<!--en:c77cc3d1-->
[Auswertung der Abrufkurven]{.kicker}

<!--en:82085b9d-->
- **Vierfacher Zustand (`GG` gegenüber `GG, 4x state`):** Bei gleicher Parameteranzahl steigt die Zahl der Zustandseinträge von $128$ auf $512$ Gleitkommazahlen pro Schicht. Damit steigt die Abrufgenauigkeit bei 64 Paaren von **$\approx 0.5$ auf einen Wert zwischen $0.8$ und $1.0$**.

<!--en:402264f7-->
- **Wirkung der kurzen Faltung:** Benachbarte Token $(k_i, v_i)$ benötigen die kausale Faltung der Breite 4, damit im Schritt $t$ das äußere Produkt $\mathbf{v}_t \mathbf{k}_t^\top$ beide Token verknüpft.

<!--en:7a04e18f-->
::: {.slide title="Warum die kausale Faltung der Breite 4 entscheidend ist"}

<!--en:6a14c4c2-->
[Ablation der lokalen Verknüpfung von Token]{.kicker}

<!--en:72ec7701-->
Was geschieht, wenn die tiefenweise kausale Faltung der Breite $4$ vor `GatedDeltaNet` durch die Einstellung `conv_width=1` entfällt?

<!--en:f7fcc268-->
- **Warum `conv_width=1` schon bei $4$ Paaren scheitert:** In unserem Sequenzformat kommt das Schlüsseltoken $k_i$ im Schritt $t - 1$ und sein Werttoken $v_i$ im Schritt $t$. Ohne Mischung im lokalen rezeptiven Feld (`conv_width=1`) sehen die linearen Projektionen im Schritt $t$, also $(\mathbf{k}_t, \mathbf{v}_t)$, nur das Werttoken $v_i$. Innerhalb einer einzelnen Schicht können sie es nicht mit dem vorhergehenden Schlüsseltoken $k_i$ verknüpfen.

<!--en:fd076d67-->
- **Die kausale Faltung der Breite 4 verbindet benachbarte Token:** Die Mischung von $4$ aufeinanderfolgenden Schritten in $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ ermöglicht es, im Schritt $t$ den Schlüssel $\mathbf{k}_t$ aus Position $t-1$ und den Wert $\mathbf{v}_t$ aus Position $t$ zu gewinnen und beide mit einem einzigen äußeren Produkt in die Matrix zu schreiben.

<!--en:cf8d94f5-->
::: {.slide title="WikiText-103 auf Byte-Ebene: GGGG, Hybridmodell GGGA und Attention"}

<!--en:a94e4287-->
[Vergleich zur Sprachmodellierung]{.kicker}

<!--en:29631af3-->
Vergleich von vier Modellen mit 4 Blöcken der Breite 256: `Gated DeltaNet` (`GGGG`), das 3:1-Hybridmodell `hybrid` (`GGGA`, Attention in Block 4), reine kausale `attention` (`AAAA`) und Attention nach derselben kausalen Faltung der Breite 4 (`CCCC`). Training auf Fenstern von 128 Byte aus WikiText-103 über $8{,}000$ Schritte:

<!--en:9802655d-->
::: {.slide title="Verlust nach Position in Fenstern von 1024 Byte"}

<!--en:57295c4b-->
[Längerer Kontext macht die Unterschiede zwischen den Speichern sichtbar]{.kicker}

<!--en:d236f68a-->
- **Versuchsaufbau:** Fenster in Batches zu $8 \times 1{,}024$ Byte für $8{,}000$ Schritte (gleiche Byteanzahl pro Schritt). Mittelwert aus zwei Läufen je Positionsbereich $[a, 2a)$.

<!--en:5929cb61-->
- **Zustände fester Größe gewinnen kaum noch:** Die `GRU` führt bis Byte $127$. Sowohl `GRU` als auch `GGGG` gewinnen nur $\approx 0.03$ nach Byte $128$.

<!--en:c923c49f-->
- **Der Verlust des Hybridmodells sinkt weiter:** Anfangs liegt es gleichauf mit `GGGG`. Später beträgt sein Vorsprung $\approx 0.05$ gegenüber diesem Modell und gegenüber der `GRU` $\approx 0.02$, gemessen bei Byte $512$–$1{,}023$.

<!--en:ad38d092-->
- **Attention:** Im letzten Bereich erreicht sie $1.03$–$1.04$, sobald sich ein Induktionskopf bildet (etwa jeder fünfte Lauf bleibt nahe $1.10$ und hebt den Mittelwert).

<!--en:f1f12e79-->
::: {.slide title="Kurze Fenster verdecken Schwächen beim Abruf"}

<!--en:18254037-->
[Aussagekraft des Vergleichs]{.kicker}

<!--en:7f1ade24-->
- **Verlust bei kurzen Fenstern (`T = 128`, $8{,}000$ Schritte):** `GGGG` ($\approx 1.16$ Nat/Byte), `GGGA` und Attention mit Faltung (beide $\approx 1.15$) liegen fast gleichauf. Reine Attention (`AAAA`) liegt nur wegen der fehlenden lokalen Mischung bei $1.18$ zurück. Nach Byte 16 ändert sich der Verlust kaum noch und sagt nichts über den Abruf aus.

<!--en:f8393b56-->
- **Abruf über große Entfernungen (`m = 64` Paare, 2 Blöcke):** `GG` ($d_k = 4$) erreicht höchstens $\approx 50\%$, das Hybridmodell `GA` dagegen **$100\%$ Abrufgenauigkeit**. Bei Fenstern von $1{,}024$ Byte liegt das 3:1-Modell `GGGA` (**$1.07$–$1.08$ Nat/Byte**) vor `GGGG` ($1.11$–$1.12$) und braucht nur $25\%$ des KV-Caches.

<!--en:0b21d20b-->
::: {.slide title="Speicherbedarf bei der Inferenz: GQA-Cache und Zustand fester Größe"}

<!--en:b3ae404e-->
[Speicherbedarf im Betrieb und bei der Generierung]{.kicker}

<!--en:a571b209-->
Beim autoregressiven Dekodieren im großen Maßstab ($16$ Bit, `float16` / `bfloat16`) bestimmt der Speicherbedarf pro Sequenz die maximale Batchgröße und die durch die Speicherbandbreite bedingte Latenz:

<!--en:7970f877-->
| Schichttyp (Breite $d = 4{,}096$, Gleitkommazahlen mit $16$ Bit) | Speicherbedarf pro Schicht | Bei $T = 512$ | Bei $T = 128\text{K}$ | Bei $T = 256\text{K}$ |

<!--en:325f8875-->
| **Grouped-Query-Attention ($8$ KV-Köpfe $\times 128$)** | $2 \times 8 \times 128 \times 2\text{ B} \times T = 4\text{ KB} \times T$ | $2\text{ MB}$ | **$512\text{ MB}$** | **$1{,}024\text{ MB}$** |

<!--en:8a85ab9d-->
| **Mamba-2 / Gated DeltaNet (Zustand der Größe $8{,}192 \times 128$)** | $8{,}192 \times 128 \times 2\text{ B} = \mathbf{2\text{ MB}}$ | $2\text{ MB}$ | **$2\text{ MB}$** | **$2\text{ MB}$** |

<!--en:3011f202-->
- **Gleichstand bei 512 Token:** Bei $T = 128\text{K}$ Token ist der KV-Cache einer einzigen GQA-Schicht **$256\times$ so groß** wie ein rekurrenter Matrixzustand ($512\text{ MB}$ gegenüber $2\text{ MB}$).

<!--en:78405c97-->
::: {.slide title="Einsparungen beim KV-Cache in modernen Hybridarchitekturen"}

<!--en:63291912-->
[Speicherbedarf bei 32 Schichten]{.kicker}

<!--en:98786d71-->
- **Transformer mit 32 Schichten bei $256\text{K}$ Token:** $32 \times 1\text{ GB} = \mathbf{32\text{ GB}}$ KV-Cache *pro Sequenz*.

<!--en:ef94f472-->
- **Hybridmodell mit jeder achten Schicht als Attention-Schicht (Jamba, $4$ Attention- und $28$ Mamba-Schichten):** $4 \times 1\text{ GB} + 28 \times 2\text{ MB} \approx \mathbf{4.06\text{ GB}}$ (**Reduktion um den Faktor $8\times$**).

<!--en:66a9717b-->
- **Hybridmodell mit jeder vierten Schicht als Attention-Schicht (Qwen3-Next und Kimi Linear, Gated DeltaNet und Attention im Verhältnis 3:1):** Verkleinert den KV-Cache um den Faktor **$4\times$** bei weiterhin vollständigem Abruf aus dem Kontext.

<!--en:2f3fd5d4-->
::: {.slide title="Sequenzmischer aus Kapitel §{num:chap_modern_rnn} im Vergleich (1/2)"}

<!--en:9f67ec88-->
[Architekturen im Überblick]{.kicker}

<!--en:3a4b938d-->
| Sequenzmischer | Zustand pro Schicht bei der Generierung | Methode für paralleles Training | Ergebnis auf synthetischen Daten | WikiText-103 ($d=256$) |

<!--en:9e15ea00-->
| **LSTM / GRU (§{sec_lstm})** | $2d$ / $d$ | Sequenziell ($T$ Schritte) | $1.00$ beim Abruf nach $T=200$ Schritten mit Gate-Bias $+5$ für den Erhalt | **$1.29$–$1.30$** ($4.3\text{M}$ / $3.8\text{M}$) |

<!--en:234f6699-->
| **minGRU (§{sec_linear-recurrence})** | $d$ | Assoziativer Scan | $1.00$ beim Abruf nach $T=1000$ Schritten mit den Zeitskalen von S4D | $1.34$–$1.35$ ($2.8\text{M}$) |

<!--en:8d6f322e-->
| **S4D (§{sec_ssm})** | $dN$ | FFT-Faltung oder Scan | $1.00$ bei fester Verzögerung $T=1000$, $\le 0.20$ beim selektiven Kopieren | – |

<!--en:b8959ea8-->
| **Mamba (§{sec_mamba})** | $HN$ ($H=2d$) | Scan und adjungierter Scan | **$0.98+$** beim selektiven Kopieren über $1{,}024$ Schritte | **$1.30$** ($4.0\text{M}$) |

<!--en:7b2a412f-->
::: {.slide title="Sequenzmischer aus Kapitel §{num:chap_modern_rnn} im Vergleich (2/2)"}

<!--en:9f67ec88-2-->
[Architekturen im Überblick]{.kicker}

<!--en:3a4b938d-2-->
| Sequenzmischer | Zustand pro Schicht bei der Generierung | Methode für paralleles Training | Ergebnis auf synthetischen Daten | WikiText-103 ($d=256$) |

<!--en:70bdef49-->
| **Lineare Attention mit Gate (§{sec_matrix-state})** | $d^2 / h$ | Blockweise GEMM und Scan | Abrufkapazität $\propto d^2/h$, überlagert Werte bei Neubelegung von Schlüsseln | – |

<!--en:c68b89c5-->
| **Gated DeltaNet (§{sec_deltanet})** | $d^2 / h$ | Blockweises Lösen und Scan | **$0.96+$** bei kurz darauf neu belegten Schlüsseln, Sättigung bei $n \gg d_k$ | $1.16$ nach $8{,}000$ Schritten ($3.6\text{M}$) |

<!--en:b23b6c83-->
| **Kausale Attention** | $2dT$ | FlashAttention / fusionierte GEMMs | **$1.00$** bei jeder Paaranzahl, Rechenaufwand $O(T^2)$ und Cache $O(T)$ | $1.38$–$1.39$, nach $8{,}000$ Schritten $1.18$, mit Faltung $1.15$ ($3.3\text{M}$) |

<!--en:bb3f09ac-->
| **Hybridmodell (`GA` / `GGGA`, §{sec_hybrids})** | Zustände fester Größe, vereinzelt zusätzlich $2dT$ | Blockweise Berechnung und Attention | **$1.00$** bei jeder Paaranzahl mit einem um den Faktor $4\times$–$8\times$ kleineren Cache | $1.15$ nach $8{,}000$ Schritten ($3.5\text{M}$) |

<!--en:3d44b09b-->
::: {.slide title="Welche Sequenzarchitektur eignet sich wann?"}

<!--en:da2e254b-->
[Richtlinien für den Entwurf]{.kicker}

<!--en:91c3daab-->
- **Unbegrenzte Datenströme und Verarbeitung auf Endgeräten (`Mamba` / `S4D` / `GRU`):** Besonders geeignet für Audio, Robotik und Sensordatenströme, wenn Speicherbedarf und Latenz pro Schritt strikt konstant bleiben müssen ($O(1)$).

<!--en:827315dd-->
- **Basismodelle mit langem Kontext (3:1- oder 7:1-Hybridmodell):** Blockweise berechnete `GatedDeltaNet`- oder `Mamba-2`-Schichten im Wechsel mit vereinzelten GQA-Schichten ermöglichen schnellen Prefill, um den Faktor $4\times$–$8\times$ kleinere KV-Caches und $100\%$ exakten Abruf.

<!--en:448e0ba9-->
::: {.slide title="Verständnisfragen zur Welch-Schranke und zum Speicher im Hybridmodell"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:8bf0cb0e-->
Zum Abschluss drei Rechen- und Verständnisaufgaben lösen:

<!--en:57e6de21-->
1. Welche untere Schranke liefert die Welch-Schranke für das mittlere quadrierte innere Produkt von $m = 33$ Schlüsseln der Norm eins in einem Kopf der Dimension $d_k = 8$? Wie groß ist der entsprechende Wert für zufällige Schlüssel der Norm eins?

<!--en:b94f69bc-->
2. Eine GPU mit $80\text{ GB}$ Speicher hält $16\text{ GB}$ Modellgewichte. Damit bleiben $64\text{ GB}$ für Inferenzcaches. Bei $T = 128\text{K}$ Token braucht jede GQA-Schicht $512\text{ MB}$ und jede rekurrente Schicht $2\text{ MB}$. Wie viele Sequenzen mit je $128\text{K}$ Token kann ein Transformer mit 32 Schichten gleichzeitig verarbeiten, wie viele ein Hybridmodell mit 32 Schichten, davon $4$ Attention-Schichten?

<!--en:001c263c-->
3. Warum scheitert ein einschichtiger Matrixspeicher mit `conv_width=1`, wenn Schlüssel und Werte in aufeinanderfolgenden Token $(t-1, t)$ eintreffen?

<!--en:557b98b6-->
**Lösungen:**

<!--en:02cabb8f-->
1. Die Welch-Schranke beträgt $\frac{33 - 8}{8 \times (33 - 1)} = \frac{25}{256} \approx \mathbf{0.0977}$ (gegenüber $1/8 = \mathbf{0.1250}$ für zufällige Schlüssel der Norm eins).

<!--en:28816659-->
2. **Transformer mit 32 Schichten:** $32 \times 0.5\text{ GB} = 16\text{ GB/seq} \implies 64 / 16 = \mathbf{4\text{ sequences}}$. **Hybridmodell mit 32 Schichten ($4$ Attention-Schichten und $28$ rekurrente Schichten):** $4 \times 512\text{ MB} + 28 \times 2\text{ MB} \approx 2.10\text{ GB/seq} \implies \lfloor 64 / 2.10 \rfloor = \mathbf{30\text{ sequences}}$ ($7.5\times$ so viele Sequenzen gleichzeitig).

<!--en:519d77a9-->
3. Bei `conv_width=1` hängen $\mathbf{k}_t$ und $\mathbf{v}_t$ im Schritt $t$ nur von $x_t$ ab, dem Werttoken. Deshalb sieht $\mathbf{v}_t \mathbf{k}_t^\top$ das Schlüsseltoken $x_{t-1}$ nie. Eine kausale Faltung der Breite 4 führt $(x_{t-1}, x_t)$ im selben Schritt zusammen.

<!--en:dee6e157-->
::: {.slide title="Zusammenfassung: Hybride Architekturen und Grenzen eines Zustands fester Größe"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:4b65ddb8-->
| Prinzip / Komponente | Mathematisches / architektonisches Ergebnis | Praktische Bedeutung |

<!--en:dd6ef74c-->
| **Schranke aus der Bitzahl** | $M b \ge k \log_2 V \implies k_{\max} = O(M)$ | Die Abrufkapazität eines Zustands fester Größe ist unabhängig von der Kontextlänge $T$ |

<!--en:02bb44ae-->
| **Welch-Schranke für die Überlappung** | $\operatorname{mean}_{i \neq j}(\mathbf{k}_i^\top \mathbf{k}_j)^2 \ge \frac{m - d_k}{d_k(m - 1)}$ für $m > d_k$ | Das lineare Auslesen der Matrix wird ungenauer, sobald die Anzahl der gespeicherten Schlüssel die Kopfbreite $d_k$ übersteigt |

<!--en:dbca6bfa-->
| **Skalierung der Zustandsbreite** | $h$ Köpfe der Größe $d_h \times d_h$ fassen $d \cdot d_h$ Einträge | Eine vierfache Kopfdimension $d_h$ bei fester Breite $d$ vervierfacht die Abrufkapazität für Paare |

<!--en:1f9b5645-->
| **Kurze kausale Faltung** | Tiefenweise kausale `Conv1d` der Breite 4 vor QKV | Verknüpft benachbarte Schlüssel- und Werttoken bei einem einzigen Schreibzugriff auf die Matrix |

<!--en:ec37e5bd-->
| **Hybridmodell aus Rekurrenz und Attention** | `GatedDeltaNet` / `Mamba-2` und GQA im Verhältnis 3:1 oder 7:1 | Ermöglicht wieder $100\%$ Genauigkeit beim assoziativen Abruf und verkleinert den KV-Cache bei $128\text{K}$–$256\text{K}$ Token um den Faktor $4\times$–$8\times$ |
