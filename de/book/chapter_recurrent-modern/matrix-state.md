<!--en:17f19eff-->
# Rekurrenzen mit Matrixzustand

<!--en:1408ca56-->
Die selektiven Zustandsraummodelle aus :numref:`sec_mamba` entscheiden für jedes Token, was sie wie lange behalten. Wie viele verschiedene Zuordnungen ein Zustand fester Größe speichern und wieder abrufen kann, ist dabei noch offen. Die Sprachmodellierung erfordert häufig einen *assoziativen Abruf* (engl. associative recall). Dabei binden wir einen Schlüssel an einen Wert (`rate = 0.05` oder *„Mira ist die Schiffsingenieurin“*) und rufen den Wert Hunderte von Token später ab, wenn der Schlüssel erneut erscheint. Transformer speichern dazu alle bisherigen Schlüssel und Werte in einem wachsenden KV-Cache und vergleichen die Abfrage mit allen gespeicherten Schlüsseln. Kann auch ein rekurrenter Zustand fester Größe Schlüssel-Wert-Paare speichern und wieder abrufen? Wie skaliert seine Kapazität mit der Zustandsgröße?

<!--en:7f17cf58-->
Ja, wenn der Zustand eine *Matrix* $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ ist. Zum Schreiben addieren wir äußere Produkte (engl. outer products) $\mathbf{v}_t \mathbf{k}_t^\top$, zur Abfrage verwenden wir die Matrix-Vektor-Multiplikation $\mathbf{S}_t \mathbf{q}_t$. Zwei Beispiele für diese Struktur kennen wir bereits. Das eine ist die unnormierte lineare Attention (:numref:`sec_windowed-linear-attention`), das andere die $H \times N$-Matrix von Mamba mit einer Zeile pro Kanal und einer Spalte pro Zustandskomponente (:numref:`sec_mamba`). Für Matrixspeicher, die äußere Produkte addieren, leiten wir das exakte Skalierungsgesetz für Abruffehler und Kapazität her. Wir führen einen eingabeabhängigen skalaren Abklingfaktor ein, damit der Speicher veraltete Paare vergisst. Anschließend zeigen wir die *Zustandsraumdualität* (engl. state space duality) zwischen Matrixrekurrenzen mit Abklingen und kausal maskierter linearer Attention. Wir implementieren den hardwareeffizienten Algorithmus für blockweises Training (`chunked_linear_attention`) und prüfen, ob trainierte `GatedLinearAttention`-Schichten beim assoziativen Abruf mit mehreren Abfragen dem theoretischen Kapazitätsgesetz folgen.

<!--en:0c392e24-->
## Ein assoziativer Speicher

<!--en:73238bd4-->
### Paare schreiben und lesen

<!--en:54eae992-->
Der einfachste assoziative Speicher speichert Paare als Summe äußerer Produkte. Betrachten wir ein Beispiel in $\mathbb{R}^2$. Wir binden die orthogonalen Schlüssel $\mathbf{k}_1 = (1, 0)^\top$ und $\mathbf{k}_2 = (0, 1)^\top$ mit Einheitsnorm an die skalaren Werte $v_1 = 3$ und $v_2 = 5$ und speichern die Paare im Zeilenvektor $\mathbf{S} = v_1 \mathbf{k}_1^\top + v_2 \mathbf{k}_2^\top = (3, 5)$. Multiplizieren wir ihn mit einem der Schlüssel, erhalten wir exakt den zugehörigen Wert: $\mathbf{S}\mathbf{k}_1 = 3$ und $\mathbf{S}\mathbf{k}_2 = 5$. Fügen wir jedoch ein drittes Paar aus $\mathbf{k}_3 = (1, 1)^\top / \sqrt{2}$ und $v_3 = 7$ hinzu, so überlappt $\mathbf{k}_3$ mit $\mathbf{k}_1$ und $\mathbf{k}_2$. Die Matrix $\mathbf{S}$ wird zu $(3 + 7/\sqrt{2}, 5 + 7/\sqrt{2}) \approx (7.95, 9.95)$, und die Abfrage mit $\mathbf{k}_1$ liefert $7.95$ statt $3$. In zwei Schlüsseldimensionen lassen sich höchstens zwei orthogonale Schlüssel ohne gegenseitige Störbeiträge speichern.

<!--en:d9f524c8-->
Im allgemeinen Fall haben wir Schlüsselvektoren $\mathbf{k}_t \in \mathbb{R}^{d_k}$, Wertvektoren $\mathbf{v}_t \in \mathbb{R}^{d_v}$ und Abfragevektoren $\mathbf{q}_t \in \mathbb{R}^{d_k}$. Der Zustand ist eine Matrix $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$, die wir in jedem Schritt wie folgt aktualisieren und auslesen:

<!--en:47fc06ed-->
Bei jedem Schreibvorgang addieren wir eine Rang-eins-Matrix $\mathbf{v}_t \mathbf{k}_t^\top$, die die Richtung $\mathbf{k}_t$ auf $\mathbf{v}_t$ abbildet. Dabei sind $(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t)$ gelernte lineare Projektionen der Tokendarstellung.

<!--en:c9c1690f-->
### Lineare Attention

<!--en:22ff05d5-->
Entrollen wir :eqref:`eq_ms-recurrence` ausgehend von $\mathbf{S}_0 = \mathbf{0}$, so sehen wir, dass eine additive Matrixrekurrenz unnormierte kausale lineare Attention berechnet (:numref:`sec_windowed-linear-attention`, :citet:`Katharopoulos.Vyas.Pappas.ea.2020`):

<!--en:6090a26e-->
Werten wir das Produkt von rechts aus, so erhält für jede Ausgabe $\mathbf{y}_t$ jeder bisherige Wert $\mathbf{v}_s$ ein Gewicht, das seiner Schlüsselähnlichkeit zur Abfrage entspricht, dem Skalarprodukt $\mathbf{k}_s^\top \mathbf{q}_t$. Werten wir von links aus, so führen wir die Summe der äußeren Produkte im rekurrenten Zustand $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ fester Größe mit.

<!--en:08d441d6-->
### Interferenz

<!--en:bec15793-->
Wie viele Paare kann $\mathbf{S} = \sum_{i=1}^n \mathbf{v}_i \mathbf{k}_i^\top$ speichern, bevor die Störbeiträge das Signal überlagern? Bei einer Abfrage mit einem gespeicherten Schlüssel $\mathbf{k}_j$ mit Einheitsnorm ($\|\mathbf{k}_j\| = 1$) können wir den Zielwert und die Interferenz der übrigen $n - 1$ Paare getrennt angeben:

<!--en:59a81a77-->
Fehlerfreies Auslesen setzt paarweise orthogonale Schlüssel voraus. Im Raum $\mathbb{R}^{d_k}$ gibt es davon höchstens $d_k$. Für zufällige Schlüssel mit Einheitsnorm können wir die erwartete Interferenz exakt berechnen.

<!--en:85b1d5a3-->
**Proposition.** Die Schlüssel $\mathbf{k}_1, \ldots, \mathbf{k}_n$ seien unabhängig und gleichverteilt auf der Einheitssphäre in $\mathbb{R}^{d_k}$. Ferner seien $\mathbf{v}_1, \ldots, \mathbf{v}_n \in \mathbb{R}^{d_v}$ unabhängig von den Schlüsseln gewählte Einheitsvektoren. Dann gilt für den erwarteten quadratischen Abruffehler an jedem gespeicherten Schlüssel:

<!--en:e5f1234f-->
**Beweis.** Multiplizieren wir das Normquadrat von $\sum_{i \neq j} (\mathbf{k}_i^\top \mathbf{k}_j)\mathbf{v}_i$ aus, so verschwinden die gemischten Terme mit $i \neq l$ im Erwartungswert. Denn bedingt auf $\mathbf{k}_j$ sind $\mathbf{k}_i^\top \mathbf{k}_j$ und $\mathbf{k}_l^\top \mathbf{k}_j$ unabhängig und haben Mittelwert null. Wegen $\|\mathbf{v}_i\|^2 = 1$ bleiben $n - 1$ Diagonalterme $\mathbb{E}[(\mathbf{k}_i^\top \mathbf{k}_j)^2]$ übrig. Da die Einheitssphäre in $\mathbb{R}^{d_k}$ rotationssymmetrisch ist, hat das Quadrat jeder Koordinate von $\mathbf{k}_i$ den Erwartungswert $1/d_k$. Somit gilt $\mathbb{E}[(\mathbf{k}_i^\top \mathbf{k}_j)^2] = 1/d_k$. Die Summe über die $n - 1$ störenden Paare ergibt $(n - 1)/d_k$. $\blacksquare$

<!--en:fde9db7b-->
Der Abruffehler hängt vom Belegungsgrad $(n - 1)/d_k$ ab. *Wann* ein Paar geschrieben wurde, spielt keine Rolle. Ein additiver Speicher verliert Paare, wenn sich zu viele den Speicher teilen, unabhängig von ihrem Alter.

<!--en:696b0e5a-->
Um :eqref:`eq_ms-capacity-law` empirisch in NumPy zu prüfen, schreibt `matrix_memory` zunächst $n$ Paare aus Schlüsseln mit Einheitsnorm und Werten aus einem Codebuch in $\mathbf{S} = \sum_{i=1}^n \mathbf{v}_i \mathbf{k}_i^\top$. Wahlweise erzeugt eine gemeinsame Richtung eine Korrelation mit Parameter $\rho$. Die Funktion liefert den mittleren quadratischen Abruffehler und die Trefferquote bei der Suche nach dem nächsten Nachbarn in einem Codebuch aus 256 Einheitsvektoren.

<!--en:65fd7061-->
Wir werten `matrix_memory` für verschiedene Anzahlen gespeicherter Paare $n \in \{8, 16, \ldots, 512\}$ aus, einmal für unabhängige zufällige Schlüssel ($d_k \in \{32, 64, 128\}, \rho = 0$) und einmal für korrelierte Schlüssel ($d_k = 64, \rho = 0.5$). Die quadratischen Lesefehler vergleichen wir im Diagramm mit der theoretischen Geraden $(n - 1)/d_k$:

<!--en:c49f4300-->
In Diagramm und Tabelle fallen zwei Eigenschaften auf:

<!--en:21174fba-->
1. **Die Abrufkapazität skaliert mit der Gesamtzahl der Zustandseinträge $d_k d_v$:** Die Kurven für den quadratischen Fehler liegen direkt auf der Geraden $(n - 1)/d_k$. Für $n = d_k = d_v = 64$ beträgt der gesamte quadratische Fehler $\approx 1$ und ist damit so groß wie das Einheitssignal $\|\mathbf{v}_j\|^2 = 1$. Dennoch liegt die Trefferquote bei der Suche nach dem nächsten Nachbarn noch bei $1.00$ ($96\%$ für $n = 128$). Warum? Der Interferenzvektor verteilt sich isotrop über alle $d_v$ Wertdimensionen. Seine Projektion auf jeden einzelnen Kandidatenvektor im Codebuch hat daher die Varianz $\approx n / (d_k d_v)$. Für $d_v = d_k$ vervierfacht eine Verdopplung von $d_k$ sowohl die Zahl der Zustandseinträge $d_k d_v$ als auch die Zahl der Paare, die sich bei gegebener Genauigkeit abrufen lassen ($96\%$ bei $n = 32$ für $d_k = 32$ und bei $n = 128$ für $d_k = 64$ sowie $94\%$ bei $n = 512$ für $d_k = 128$). In Übung :exref:`ex_ms-value-width` variieren wir $d_k$ und $d_v$ unabhängig voneinander.

<!--en:af16c627-->
2. **Korrelierte Schlüssel setzen dem Fehler eine feste Untergrenze:** Haben die Schlüssel eine gemeinsame Richtung mit Gewicht $\rho$, so beträgt ihre paarweise Überlappung im Mittel $\approx \rho^2$. Für den erwarteten quadratischen Fehler ergibt sich

   $$
   \mathbb{E}\,\big\|\mathbf{S}\mathbf{k}_j - \mathbf{v}_j\big\|^2 \approx
   (n - 1)\Big(\rho^4 + \frac{(1 - \rho^2)^2}{d_k}\Big)
   $$

   bei Werten mit Mittelwert null (hinzu kommt ein in $n$ quadratischer Term aus dem Stichprobenmittelwert des endlichen Codebuchs, siehe Übung :exref:`ex_ms-correlated-keys`). Weil $\rho^4$ mit wachsendem $d_k$ nicht abnimmt, sinkt bei korrelierten Schlüsseln ($\rho = 0.5$) die Trefferquote für $n = d_k = 64$ von $1.00$ auf $0.49$.

<!--en:f74a1750-->
## Abklingen

<!--en:707cfc85-->
### Vergessen alter Paare

<!--en:fa3dd979-->
Ohne Abklingen bleibt jedes einmal geschriebene Paar mit vollem Gewicht in $\mathbf{S}_t$. Die Interferenz wächst daher mit der Sequenzlänge unbegrenzt. Multiplizieren wir den Zustand vor jedem Schreibvorgang mit einem skalaren Abklingfaktor $a_t \in (0, 1)$, so verlieren ältere Paare allmählich an Gewicht:

<!--en:8c2f15ff-->
Bei konstantem Abklingfaktor $a$ hat ein vor $\ell$ Schritten geschriebenes Paar das Gewicht $a^\ell$. Die Gewichte summieren sich zu $1/(1 - a)$, was einem effektiven Fenster von $1/(1 - a)$ Paaren entspricht. In die Varianz der Interferenz gehen die quadrierten Gewichte ein. Ihre Summe beträgt $1/(1 - a^2)$. Damit ist die erwartete Interferenz im stationären Zustand unabhängig von der gesamten Sequenzlänge durch $1 / ((1 - a^2) d_k)$ nach oben beschränkt (Übung :exref:`ex_ms-sliding-window`). Berechnen wir $a_t$ aus dem Eingabetoken, kann jeder Kopf seinen Zustand über eine Passage hinweg beibehalten und bei einem Themenwechsel oder am Ende eines Teilsatzes löschen. (RetNet verwendet feste Abklingfaktoren pro Kopf :cite:`Sun.Dong.Huang.ea.2023`, Mamba-2 eingabeabhängige skalare Abklingfaktoren pro Kopf :cite:`Dao.Gu.2024`. Gated Linear Attention verwendet einen Vektor von Abklingfaktoren mit einem Faktor pro Schlüsselkoordinate: $\mathbf{S}_{t-1}\textrm{diag}(\boldsymbol{\alpha}_t)$ :cite:`Yang.Wang.Shen.ea.2024`. Wir verwenden einen eingabeabhängigen skalaren Abklingfaktor pro Kopf.)

<!--en:f49f0942-->
### Selektive Zustandsraummodelle als Matrixspeicher

<!--en:524ea3a2-->
Wir schreiben die $H$ Kanalzustände $\mathbf{h}_t^{(c)} \in \mathbb{R}^N$ des selektiven Zustandsraummodells von Mamba (:numref:`sec_mamba`, ohne die Skip-Verbindung $D_c x_{t,c}$) als Zeilen einer Matrix $\mathbf{S}_t \in \mathbb{R}^{H \times N}$. Damit erkennen wir, dass Mamba bereits eine Rekurrenz mit Matrixzustand und Abklingen ist:

<!--en:1f0370bd-->
Der Vergleich mit :eqref:`eq_ms-decay` liefert den Wertvektor $\mathbf{v}_t = \boldsymbol{\Delta}_t \odot \mathbf{x}_t \in \mathbb{R}^H$, den Schlüsselvektor $\mathbf{k}_t = \mathbf{B}_t \in \mathbb{R}^N$ und den Abfragevektor $\mathbf{q}_t = \mathbf{C}_t \in \mathbb{R}^N$. Der einzige Unterschied besteht darin, dass die Einträge der Abklingmatrix $\mathbf{E}_t \in \mathbb{R}^{H \times N}$ bei Mamba sowohl vom Kanal als auch von der Zustandskomponente abhängen. Beschränken wir $\mathbf{E}_t$ auf einen einzigen Skalar $a_t$ pro Kopf, erhalten wir Mamba-2 :cite:`Dao.Gu.2024` (Übung :exref:`ex_ms-mamba`). Diese Einschränkung ermöglicht die folgende duale Form mit Matrizenmultiplikation.

<!--en:1d563f52-->
## Die duale Form

<!--en:7bab332e-->
### Entrollen der Rekurrenz mit Abklingen

<!--en:e662ade6-->
Entrollen wir :eqref:`eq_ms-decay` ausgehend von $\mathbf{S}_0 = \mathbf{0}$, so sehen wir, mit welchem Gewicht ein in Schritt $s$ geschriebenes Paar in Schritt $t \ge s$ ankommt. Es ist um das Produkt der bis dahin wirksamen Abklingfaktoren $L_{ts} = a_{s+1} a_{s+2} \cdots a_t$ abgeschwächt:

<!--en:aebf667d-->
Schreiben wir $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t, \mathbf{y}_t$ als Zeilen von $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{Y}$ und setzen $L_{ts} = 0$ für $s > t$, erhalten wir die *duale Form*:

<!--en:5922a188-->
:eqref:`eq_ms-semiseparable` beschreibt kausale lineare Attention mit einer *gelernten, eingabeabhängigen Abklingmaske* $\mathbf{L}$. Diese Entsprechung bezeichnen :citet:`Dao.Gu.2024` als *Zustandsraumdualität* (SSD). Werten wir :eqref:`eq_ms-semiseparable` schrittweise als Rekurrenz aus, benötigen wir $O(T d_v d_k)$ Rechenaufwand und $O(d_v d_k)$ Speicherplatz. Verwenden wir die volle $T \times T$-Matrix $\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top$, können wir parallele Matrizenmultiplikationen einsetzen. Der Rechenaufwand beträgt dann $O(T^2(d_k + d_v))$, der Speicherbedarf $O(T^2)$. Schreiben wir beide Formen explizit auf, zunächst die sequenzielle Rekurrenz `linear_attention_recurrent`:

<!--en:be53f146-->
Als Nächstes bildet `decay_mask(log_a)` über kumulative Summen die untere Dreiecksmatrix $L_{ts} = \exp(\sum_{r=s+1}^t \log a_r)$ der Abklingfaktoren. Damit kann `linear_attention_dual` die Ausgabe $\mathbf{Y} = (\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ für alle Positionen zugleich berechnen:

<!--en:5dbd7fcc-->
### Blockweise Berechnung

<!--en:d5b3a7c5-->
Für das Training auf der GPU haben beide Formen Nachteile. Die Rekurrenz auf Tokenebene besteht aus $T$ winzigen sequenziellen Schritten. Die volle duale Form legt für jeden Kopf eine $T \times T$-Matrix im Speicher an. Die *blockweise Berechnung* (engl. chunking) verbindet die Vorteile beider Formen. Dazu zerlegen wir die Sequenz in $T / m$ Blöcke der Länge $m$ (:numref:`fig_ms-semiseparable`). Jeder blockübergreifende Abklingfaktor $L_{ts}$ zerfällt in drei Faktoren: (Abklingen vom Schlüssel bis zu seinem Blockende) $\times$ (Abklingen in den dazwischenliegenden Blöcken) $\times$ (Abklingen vom Blockanfang bis zur Abfrage). Deshalb hat jeder Block unterhalb der Diagonalen von $\mathbf{L}$ Rang $1$. In $\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top$ haben diese Blöcke höchstens Rang $d_k$. Die $d_v \times d_k$-Zustandsmatrix, die wir über die Blockgrenzen hinweg weiterreichen, fasst ihre Beiträge vollständig zusammen (Übung :exref:`ex_ms-rank-one`).

<!--en:3d0629df-->
![Die Maske $\mathbf{L}$ aus :eqref:`eq_ms-semiseparable` für zufällige Abklingfaktoren, in Blöcke unterteilt. Die Diagonalblöcke (blau umrandet) sind kleine Produkte der maskierten Attention, die sich parallel berechnen lassen. Jeder Block unterhalb der Diagonalen (orange umrandet) hat Rang eins in $\mathbf{L}$ und höchstens Rang $d_k$ in $\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top$. Sein Beitrag steckt im Zustand, der von Block zu Block weitergegeben wird (Pfeile). Je mehr Abklingfaktoren zusammenwirken, desto blasser die Schattierung.](../img/mdl-modernrnn-semiseparable.svg)

<!--en:676da4f0-->
Wir nummerieren die Positionen in jedem Block von $1$ bis $m$. Der Algorithmus `chunked_linear_attention` arbeitet in drei Schritten, jeweils im Batch:

<!--en:3047b8ac-->
1. **Innerhalb der Blöcke (Diagonalblöcke):** Wir werten die duale Form :eqref:`eq_ms-semiseparable` für alle Blöcke der Länge $m$ parallel aus. Jeder Block verwendet seine eigene $m \times m$-Abklingmaske.

<!--en:924f9621-->
2. **Über Blockgrenzen hinweg (Zustände am Blockende):** Sei $\gamma^{(c)} = a_1 \cdots a_m$ der gesamte Abklingfaktor für Block $c$. Der Zustand am Ende von Block $c$ erfüllt auf Blockebene die lineare Rekurrenz

   $$\mathbf{S}^{(c)} = \gamma^{(c)}\, \mathbf{S}^{(c-1)} + \sum_{t=1}^{m}
   L_{mt}\, \mathbf{v}_t \mathbf{k}_t^\top,$$

   die `d2l.linear_scan` über alle $T / m$ Blöcke in $\lceil \log_2(T/m) \rceil$ Runden auswertet.

<!--en:1260dfe1-->
3. **Auslesen des vorherigen Zustands (Blöcke unterhalb der Diagonalen):** Wir addieren $(a_1 \cdots a_t)\,\mathbf{S}^{(c-1)}\mathbf{q}_t$ an jeder Position $t$ in Block $c$.

<!--en:f7c00f01-->
Fast alle Gleitkommaoperationen entfallen nun auf Multiplikationen dicht besetzter Matrizen. Der maximale Arbeitsspeicherbedarf sinkt von $O(T^2)$ auf $O(Tm + (T/m) d_v d_k)$.

<!--en:be75fd21-->
Prüfen wir an einer Sequenz, ob die sequenzielle Rekurrenz, die volle duale Form und der blockweise Algorithmus für die Blockgrößen $m \in \{16, 64, 256\}$ bis auf float32-Genauigkeit übereinstimmen. Die Sequenz hat die Länge $T = 512$.

<!--en:969fb426-->
Alle Formen stimmen bis auf einen relativen Fehler von $\sim 10^{-6}$ überein (Übung :exref:`ex_ms-operations` untersucht die optimale Blockgröße $m$).

<!--en:03ba5c53-->
## Eine Schicht für Gated Linear Attention

<!--en:2cdc020d-->
### Aufbau der Schicht

<!--en:43fb5364-->
`GatedLinearAttention` fasst blockweise lineare Attention mit mehreren Köpfen zu einem Sequenzmischer zusammen. Für jeden Kopf $h$ berechnet die Schicht einen eingabeabhängigen skalaren Abklingfaktor $a_{t,h} = \sigma(\mathbf{w}_h^\top \mathbf{x}_t + b_h)$. Wir initialisieren die Bias-Terme mit `init_decays`. Dadurch verteilen sich die anfänglichen Zeitskalen $\tau_h$ ($a_{t,h} \approx e^{-1/\tau_h}$) über die Köpfe hinweg auf logarithmischer Skala gleichmäßig über den Bereich von $[4, 256]$ Schritten.

<!--en:500c3b18-->
Wir kapseln die Aktualisierung des Speichers in einer eigenen Methode `memory`. So können wir in :numref:`sec_deltanet` eine Unterklasse von `GatedLinearAttention` ableiten und die Aktualisierungsregel durch die Delta-Regel ersetzen, ohne den übrigen Block zu duplizieren. Die Funktionen `heads_to_batch` und `batch_to_heads` fassen die Kopfachse mit der Batchachse zusammen und trennen die beiden Achsen wieder.

<!--en:1743c351-->
Der Vorwärtsdurchlauf umfasst die folgenden Schritte (:numref:`fig_ms-gla-layer`):

<!--en:6024ab52-->
1. Eine tiefenweise kausale Faltung der Breite 4 und SiLU kombinieren benachbarte Token zu $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$. So kann sich ein Werttoken an Position $t$ innerhalb einer einzigen Schicht direkt an das Schlüsseltoken an Position $t - 1$ binden (in :numref:`sec_hybrids` untersuchen wir den Effekt dieser Faltung, indem wir sie weglassen).

<!--en:906fcbd5-->
2. Wir normieren Abfragen und Schlüssel mit der $\ell_2$-Norm auf Länge eins, wie in :eqref:`eq_ms-capacity-law` vorausgesetzt.

<!--en:69cd6401-->
3. Für jeden Kopf normieren wir den ausgelesenen Vektor mit RMSNorm, multiplizieren ihn mit dem Gatterwert $\textrm{SiLU}(\mathbf{W}_g \mathbf{x}_t)$ und projizieren ihn zurück auf die Breite $d$ :cite:`Yang.Wang.Shen.ea.2024`.

<!--en:a2fecba2-->
![Die Schicht für Gated Linear Attention. Eine Projektion und eine kurze kausale Faltung liefern die Abfragen, Schlüssel und Werte aller Köpfe. Eine zweite Projektion liefert für jeden Kopf den Abklingfaktor $a_t$. Die Berechnung des Matrixspeichers geschieht blockweise. Nach dem Auslesen wird der Vektor pro Kopf mit RMSNorm normiert, mit einem aus der Eingabe berechneten Gatterwert multipliziert und auf die Breite $d$ zurückprojiziert.](../img/mdl-modernrnn-gla-layer.svg)

<!--en:a9dfdd80-->
### Assoziativer Abruf

<!--en:3520e417-->
Belegt ein trainiertes neuronales Netz seinen Matrixzustand gemäß dem Kapazitätsgesetz $O(d_k d_v)$, das wir für zufällige Schlüssel hergeleitet haben? Das prüfen wir mit einem synthetischen Benchmark für den *assoziativen Abruf mit mehreren Abfragen* (engl. multi-query associative recall) :cite:`Arora.Eyuboglu.Timalsina.ea.2024`. Jede Sequenz beginnt mit $n$ Schlüssel-Wert-Paaren. Jedes Paar besteht aus einem eigenen Schlüsseltoken und einem nachfolgenden Werttoken aus einer von 4 Klassen. Optional folgen Fülltoken. Am Schluss erscheinen alle $n$ Schlüssel erneut in zufälliger Reihenfolge als Abfragen. (Mit den Optionen `num_rebound` und `gap` lässt sich einem Teil der Schlüssel ein neuer Wert zuweisen. Wir verwenden sie in :numref:`sec_deltanet`.)

<!--en:2313cf0b-->
Betrachten wir eine kurze Sequenz mit 4 Paaren, darunter 1 erneut gebundener Schlüssel (die Token `0..3` sind Werte, `4..515` sind Schlüssel und `516` ist das Fülltoken):

<!--en:1110dfe7-->
Wir speichern die Trainingsfunktion `train_recall`. Sie trainiert ein Modell mit AdamW über 2000 Schritte (maximale Lernrate $0.01$, Kosinusplan mit Aufwärmphase). Dabei durchläuft sie zyklisch die Paaranzahlen $n \in \{4, 8, 16, 32, 64\}$ und misst für jede Paaranzahl die Trefferquote an 1024 neuen Sequenzen.

<!--en:d6b11775-->
Wir variieren die Kapazität des Matrixzustands, *ohne* die Modellbreite oder die Parameteranzahl zu ändern. Dazu verwenden wir ein `SeqLM` mit zwei Blöcken und fester Breite $d = 32$. In `memory_model(num_heads)` teilen wir die Breite auf $h \in \{8, 4, 2\}$ Köpfe der Dimension $e = d / h \in \{4, 8, 16\}$ auf:

<!--en:385f4e66-->
Jeder Kopf führt eine $e \times e$-Zustandsmatrix. Damit skaliert die Zahl der Zustandseinträge pro Schicht gemäß $h \cdot e^2 = d \cdot e \in \{128, 256, 512\}$, während die gesamte Parameteranzahl nahezu gleich bleibt (etwa $37{,}000$). Für jede Kopfkonfiguration starten wir zwei Trainingsläufe mit `train_recall`:

<!--en:83e54df0-->
Diagramm und Tabelle zeigen die mittlere Abrufgenauigkeit in Abhängigkeit von der Paaranzahl $n \in \{4, 8, 16, 32, 64\}$. Daran sehen wir, wie die Abrufkapazität mit der Gesamtzahl der Zustandseinträge skaliert:

<!--en:18be78bc-->
Alle drei Konfigurationen bewältigen 4 Paare ($\ge 98\%$). Mit zunehmender Paaranzahl zeigt sich, dass die Abrufkapazität direkt mit der Zahl der Zustandseinträge skaliert. Verdoppeln wir die Kapazität von 128 Einträgen ($8 \times 4^2$) auf 256 Einträge ($4 \times 8^2$), so verdoppelt sich etwa die Zahl der bei gegebener Genauigkeit abrufbaren Paare (bei 64 Paaren steigt die Trefferquote von $\sim 0.50$ auf $\sim 0.75$). Mit 512 Einträgen ($2 \times 16^2$) ist die Trefferquote für alle Paaranzahlen $\ge 16$ höher als mit 128 Einträgen.

<!--en:25d350da-->
Jeder additive Matrixspeicher hat eine strukturelle Schwäche. Beim Schreiben wird $\mathbf{v}_t \mathbf{k}_t^\top$ zum bereits unter $\mathbf{k}_t$ gespeicherten Wert *addiert*. Binden wir einen Schlüssel erneut, diesmal an einen anderen Wert, liefert die Abfrage eine Überlagerung des alten und des neuen Werts. Wie der Speicher lernt, den Wert eines Schlüssels zu *überschreiben*, ist Gegenstand von :numref:`sec_deltanet`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:a435fbad-->
Ein Matrixzustand $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ speichert Schlüssel-Wert-Paare, indem er äußere Produkte $\mathbf{v}_t \mathbf{k}_t^\top$ addiert. Lineare Abfragen $\mathbf{S}_t \mathbf{q}_t$ lesen die Werte wieder aus. Entrollt entspricht diese Rekurrenz kausaler linearer Attention. Das selektive Zustandsraummodell von Mamba ist ein Beispiel dafür, mit Schlüssel $\mathbf{B}_t$ und Abfrage $\mathbf{C}_t$. Bei zufälligen Schlüsseln mit Einheitsnorm führen die paarweisen Störbeiträge zu einem erwarteten quadratischen Abruffehler von $(n - 1)/d_k$ nach $n$ Schreibvorgängen. Die Abrufkapazität bei der Suche nach dem nächsten Nachbarn skaliert dagegen mit der Gesamtzahl der Zustandseinträge $d_k d_v$. Ein eingabeabhängiger skalarer Abklingfaktor $a_t \in (0, 1)$ pro Kopf schwächt alte Paare ab und führt zur *Zustandsraumdualität* mit kausal maskierter linearer Attention $\mathbf{Y} = (\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top)\mathbf{V}$. Die blockweise Auswertung dieser dualen Form verbindet parallele Matrizenmultiplikationen innerhalb der Blöcke mit einem Scan über die Blöcke hinweg. Sie benötigt $O(T)$ Speicherplatz. Beim assoziativen Abruf mit mehreren Abfragen zeigt sich bei fester Parameteranzahl: Verdoppeln wir die Zahl der Zustandseinträge einer `GatedLinearAttention`-Schicht, kann sie etwa doppelt so viele Paare abrufen. Rein additive Schreibvorgänge können den Wert eines bereits gebundenen Schlüssels jedoch weiterhin nicht überschreiben.

<!--en:2294243a-->
## Übungen

<!--en:17cf6afe-->
1. **Mamba als Matrixspeicher.** In der Form
   $\mathbf{S}_t = \mathbf{E}_t \odot \mathbf{S}_{t-1} +
   (\boldsymbol{\Delta}_t \odot \mathbf{x}_t)\, \mathbf{B}_t^\top$ mit
   $\mathbf{E}_t = \exp(\textrm{diag}(\boldsymbol{\Delta}_t)\,
   \boldsymbol{\Lambda})$ ergibt sich der eintragsweise Logarithmus der Abklingmatrix aus der Matrix der Raten, indem jede Zeile $c$ mit der Schrittweite $\Delta_{t,c}$ skaliert wird.
    1. Unter welchen Bedingungen an $\boldsymbol{\Delta}_t$ und an die Raten $\boldsymbol{\Lambda}$ ist $\mathbf{E}_t$ ein skalares Vielfaches der Einsmatrix, sodass die Aktualisierung die Form :eqref:`eq_ms-decay` hat? Welche dieser Bedingungen setzt Mamba-2 voraus, und welche Größen bleiben eingabeabhängig?
    1. Bestimmen Sie für jede Form des Abklingens in diesem Abschnitt (RetNet, Mamba-2, Gated Linear Attention, Mamba), wie viele eingabeabhängige Zahlen pro Token die Abklingfaktoren eines Kopfes mit $d_v \times d_k$ Einträgen bestimmen.
   :exlabel:`ex_ms-mamba`

<!--en:00d905e4-->
1. **Blöcke vom Rang eins.** Seien $p_t = a_1 a_2 \cdots a_t$ die Produkte der Abklingfaktoren in :eqref:`eq_ms-semiseparable`.
    1. Drücken Sie $L_{ts}$ durch die $p_t$ aus. Zeigen Sie, dass jede Teilmatrix von $\mathbf{L}$ Rang eins hat, wenn jeder ihrer Zeilenindizes mindestens so groß ist wie jeder ihrer Spaltenindizes.
    1. Leiten Sie daraus ab, dass jeder Block von $\mathbf{L} \odot
       \mathbf{Q}\mathbf{K}^\top$, der einen Sequenzblock mit einem früheren koppelt, höchstens Rang $d_k$ hat. Geben Sie an, an welchen Stellen `chunked_linear_attention` die Abklingfaktoren eines solchen Blocks auf der Abfrageseite und auf der Schlüsselseite anwendet.
    1. Bei einem Abklingfaktor je Schlüsselkoordinate gilt $\mathbf{S}_t = \mathbf{S}_{t-1}
       \textrm{diag}(\boldsymbol{\alpha}_t) + \mathbf{v}_t
       \mathbf{k}_t^\top$. Dann gibt es keine einzelne Maske $\mathbf{L}$, mit der sich $\mathbf{Q}\mathbf{K}^\top$ wie zuvor multiplizieren lässt. Geben Sie den durch das Abklingen gewichteten Score für Abfrage $t$ und Schlüssel $s$ an. Zeigen Sie, dass ein Block, der zwei Sequenzblöcke koppelt, weiterhin höchstens Rang $d_k$ hat.
   :exlabel:`ex_ms-rank-one`

<!--en:1b22bd61-->
1. **Anzahl der Rechenoperationen.** Zählen Sie eine Multiplikation mit anschließender Addition als zwei Gleitkommaoperationen. Setzen Sie $d_k = d_v = d$ und berücksichtigen Sie nur die führenden Terme.
    1. Bestimmen Sie die Anzahl der Operationen für die Rekurrenz, die duale Form und die blockweise Form bei Sequenzlänge $T$ und Blockgröße $m$. Schlüsseln Sie die Anzahl für die blockweise Form nach den drei Teilen des Algorithmus auf.
    1. Zeigen Sie, dass der Aufwand der blockweisen Form höchstens ein kleines konstantes Vielfaches des Aufwands der Rekurrenz beträgt, solange $m \le d$ gilt. Berechnen Sie das Verhältnis der Operationszahlen von blockweiser und dualer Form für $T = 4096$, $d = 64$ und $m = 256$.
    1. Modellieren Sie die Laufzeit eines Verfahrens als Anzahl seiner Operationen geteilt durch den Durchsatz $P$, zuzüglich einer festen Zeit $\tau$ pro sequenziellem Schritt. Die Rekurrenz benötigt $T$ Schritte, die blockweise Form $T/m$. Bestimmen Sie die Blockgröße, die die Laufzeit der blockweisen Form minimiert. Berechnen Sie diese Größe für $T = 4096$, $d = 64$, $P = 10^{13}$ Operationen pro Sekunde und $\tau = 5\,\mu\textrm{s}$. Um welchen Faktor ist die blockweise Form laut Modell schneller als die Rekurrenz?
   :exlabel:`ex_ms-operations`

<!--en:5f715f8d-->
1. **Korrelierte Schlüssel.** ● Die Schlüssel mit $\rho > 0$ in `matrix_memory` entstehen durch Normieren von $\sqrt{1-\rho^2}\, \mathbf{g}_i + \rho\,
   \mathbf{c}$. Dabei sind $\mathbf{g}_i$ unabhängige zufällige Einheitsvektoren und $\mathbf{c}$ ist ein gemeinsamer Einheitsvektor. Die Formel im Text modelliert die Überlappung zweier verschiedener Schlüssel als $\rho^2 + \epsilon_{ij}$. Dabei sind die $\epsilon_{ij}$ unabhängig mit Mittelwert null und Varianz $(1-\rho^2)^2/d_k$. Die Formel setzt außerdem Werte mit Mittelwert null voraus.
    1. Die Werte von `matrix_memory` stammen aus einem Codebuch mit 256 zufälligen Vektoren. Deren Mittelwert $\bar{\mathbf{v}}$ ist nicht exakt null. Zeigen Sie, dass dadurch zum erwarteten quadratischen Fehler ein Term proportional zu $(n-1)(n-2)\,
       \|\bar{\mathbf{v}}\|^2$ hinzukommt. Vergleichen Sie Ihre vollständige Vorhersage mit den gemessenen Fehlern.
    1. Zentrieren Sie das Codebuch, indem Sie seinen Mittelwert abziehen und anschließend jeden Eintrag erneut normieren. Prüfen Sie, ob die Krümmung der Kurve nach oben verschwindet.
    1. Messen Sie den Fehler und die Trefferquote für $\rho \in \{0, 0.25, 0.5,
       0.75\}$ bei $d_k = 64$ und $d_k = 256$. Für welche $\rho$ bringt der breitere Schlüssel noch einen Vorteil, und wann nähert sich der Fehler dem Wert $(n-1)
       \rho^4$?
   :exlabel:`ex_ms-correlated-keys`

<!--en:711ec0c7-->
1. [code] **Schlüssel und Werte unterschiedlicher Breite.** ● Geben Sie `matrix_memory` eine eigene Wertbreite `d_v`, sodass das Codebuch 256 zufällige Einheitsvektoren in $\mathbb{R}^{d_v}$ enthält.
    1. Halten Sie $n/(d_k d_v) = 1/32$ fest und messen Sie die Trefferquote für $(d_k, d_v) = (64,
       64)$, $(256, 16)$ und $(16, 256)$. Hängt die Trefferquote allein vom Produkt $d_k d_v$ ab?
    1. Zeigen Sie, dass der erwartete quadratische Lesefehler unabhängig von $d_v$ ist und Ihre Messwerte deshalb nicht erklären kann.
    1. Jeder Codebucheintrag ist unter etwa $n/256$ Schlüsseln gespeichert. Zeigen Sie, dass deren Überlappungen mit dem zur Abfrage verwendeten Schlüssel in Richtung dieses Eintrags eine Komponente mit einer Varianz von etwa $n/(256\, d_k)$ beitragen. Erklären Sie damit, für welche Matrixformen das Produktgesetz versagt.
   :exlabel:`ex_ms-value-width`

<!--en:2c2e8660-->
1. [code] **Abklingen als gleitendes Fenster.** ● Ergänzen Sie einen konstanten Abklingfaktor $a$ in `matrix_memory`, sodass ein Paar, das $\ell$ Schritte vor dem Ende geschrieben wurde, mit Gewicht $a^\ell$ in den endgültigen Speicherzustand eingeht.
    1. Messen Sie bei $d_k = 64$ und $n = 256$ die Trefferquote getrennt für das älteste und das jüngste Viertel der Paare, jeweils für $a \in \{1, 0.99,
       0.95, 0.9\}$. Was gewinnen die jüngsten Paare durch das Abklingen, und was verlieren die ältesten?
    1. Zeigen Sie wie im Beweis der Proposition, dass die erwartete quadratische Interferenz an jedem Schlüssel höchstens $1 / ((1-a^2)\, d_k)$ beträgt. Zeigen Sie, dass die Zahl der Paare, deren Signalquadrat $a^{2\ell}$ diese Schranke übersteigt, höchstens etwa $d_k / e$ beträgt, unabhängig von der Wahl von $a < 1$.
   :exlabel:`ex_ms-sliding-window`

<!--en:8afd5e9c-->
[Diskussion](https://d2l.discourse.group/t/36783)

<!--en:4bc4427d-->
Rekurrenzen mit Matrixzustand und Gated Linear Attention<br>**Schlüssel-Wert-Speicher mit äußeren Produkten · Gesetze für Kapazität und Interferenz · Zustandsraumdualität · blockweises Training**

<!--en:6488f651-->
::: {.slide title="Von Vektorzuständen zu assoziativen Schlüssel-Wert-Matrizen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:02923d0a-->
- **Assoziativer Abruf:** In Sprache und Code wird immer wieder ein Schlüssel an einen Wert gebunden (`rate = 0.05`) und dieser Hunderte von Token später abgerufen.

<!--en:bc56dae6-->
- **Matrixzustand aus äußeren Produkten:** Der Übergang von $\mathbf{h}_t \in \mathbb{R}^d$ zu $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ erlaubt es, äußere Produkte vom Rang 1 unter ihrem Schlüssel zu speichern und über $\mathbf{S}_t \mathbf{q}_t$ abzufragen.

<!--en:11907a2c-->
::: {.slide title="Schreiben mit äußeren Produkten und Störbeiträge in Dimension 2"}

<!--en:9914b045-->
[Geometrische Anschauung]{.kicker}

<!--en:fd786e2a-->
- **Exakt für bis zu $d_k$ orthogonale Schlüssel:** In $\mathbb{R}^2$ erzeugen die orthogonalen Schlüssel $\mathbf{k}_1 = (1, 0)^\top, \mathbf{k}_2 = (0, 1)^\top$ keine Interferenz ($\mathbf{k}_1^\top \mathbf{k}_2 = 0$).

<!--en:2808fd64-->
- **Störbeiträge, sobald $n > d_k$ gilt:** Ein dritter Schlüssel $\mathbf{k}_3 = (1, 1)^\top / \sqrt{2}$ mit $v_3 = 7$ trägt zusätzlich $7/\sqrt{2}$ zum ausgelesenen Wert $\mathbf{S}\mathbf{k}_1 \approx 7.95$ bei.

<!--en:388564bc-->
::: {.slide title="Die Matrixrekurrenz berechnet unnormierte lineare Attention"}

<!--en:6c58586f-->
[Links- und Rechtsklammerung]{.kicker}

<!--en:351162c9-->
- **Linksklammerung (rekurrent):** Summiert die Beiträge in $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ mit $O(d_v d_k)$ Zustandseinträgen und $O(d_v d_k)$ Rechenaufwand pro dekodiertem Token.

<!--en:208b5b2c-->
- **Rechtsklammerung (Attention):** Berechnet die Skalarprodukte $\mathbf{k}_s^\top \mathbf{q}_t$ für mehrere Blöcke parallel durch Matrizenmultiplikationen auf Tensor Cores.

<!--en:8eec64e8-->
::: {.slide title="Exaktes Fehlergesetz für zufällige Schlüssel mit Einheitsnorm"}

<!--en:58524fa2-->
[Signal und Interferenz]{.kicker}

<!--en:10085186-->
- **Entscheidend ist die Belegung, nicht das Alter:** Für zufällige Schlüssel auf der Einheitssphäre in $\mathbb{R}^{d_k}$ beträgt der erwartete quadratische Abruffehler für jedes gespeicherte Paar $(n - 1)/d_k$.

<!--en:e0bc942f-->
- **Eindimensionale Projektion im Codebuch:** In Richtung jedes falschen Codebuchvektors $\mathbf{c} \in \mathbb{R}^{d_v}$ sinkt die Varianz auf $(n - 1)/(d_k d_v)$.

<!--en:41026ef0-->
::: {.slide title="Gemessener Fehler und Trefferquote im Codebuch bei n Paaren"}

<!--en:fe59451a-->
[Theoretische und gemessene Kapazität]{.kicker}

<!--en:da1bc446-->
Gemessener quadratischer Fehler und Trefferquote beim nächsten Nachbarn (Codebuch mit 256 Vektoren) im Vergleich zur Geraden $(n - 1)/d_k$:

<!--en:65db5244-->
::: {.slide title="Warum die Abrufkapazität beim nächsten Nachbarn mit der Eintragszahl dk dv skaliert"}

<!--en:7d5a514d-->
[Isotrope Fehlerprojektion]{.kicker}

<!--en:fd43d6bc-->
Ein Blick auf die Tabellenzeile für $d_k = d_v = 64$. Bei $n = 64$ Paaren beträgt der quadratische Fehler $(64-1)/64 \approx 1.0$ und ist damit so groß wie das Einheitssignal $\|\mathbf{v}_j\|^2 = 1$. Dennoch beträgt die Trefferquote beim nächsten Nachbarn **$1.00$** ($0.96$ bei $n = 128$). Warum?

<!--en:1684dd86-->
- **Isotrope Verteilung über $d_v$ Wertdimensionen:** Der gesamte Interferenzvektor hat das Normquadrat $\approx n / d_k$, zeigt aber in eine zufällige Richtung in $\mathbb{R}^{d_v}$.

<!--en:78032b1c-->
- **Eindimensionale Projektion auf einen beliebigen falschen Codebuchvektor $\mathbf{c} \in \mathbb{R}^{d_v}$:** Nur die Komponente in Richtung $\mathbf{c}$ konkurriert mit dem richtigen Wert $\mathbf{v}_j$. Diese eindimensionale Projektion hat die Varianz
  $$\textrm{Var}\big(\mathbf{c}^\top (\mathbf{S}\mathbf{k}_j - \mathbf{v}_j)\big) = \frac{n - 1}{d_k\, d_v}$$

<!--en:9fdd3f65-->
- **Skalierung mit den $O(d_k d_v)$ Einträgen:** Für $d_v = d_k$ **vervierfacht** eine Verdopplung von $d_k$ die Zahl der Zustandseinträge $d_k^2$ und **vervierfacht** die Zahl der Paare, die sich mit einer Genauigkeit von $\sim 95\%$ abrufen lassen ($32 \to 128 \to 512$ Paare).

<!--en:c451bfd7-->
::: {.slide title="Fehleruntergrenze durch korrelierte Schlüssel"}

<!--en:5ea86018-->
[Anisotrope Interferenz]{.kicker}

<!--en:514aa53b-->
- **Bleibende Überlappung:** Haben die Schlüssel eine gemeinsame Richtung $\mathbf{c}$ mit Gewicht $\rho > 0$, so bleibt die Untergrenze $(n - 1)\rho^4$ auch für $d_k \to \infty$ bestehen.

<!--en:6b04a845-->
- **Einbruch der gemessenen Trefferquote:** Bei $d_k = 64, \rho = 0.5$ sinkt die Trefferquote für 64 Paare von $1.00$ auf $0.49$.

<!--en:1bf73291-->
::: {.slide title="Eingabeabhängiges Abklingen begrenzt die Interferenz im Speicher"}

<!--en:3fc613cb-->
[Veraltete Paare vergessen]{.kicker}

<!--en:aa7c9867-->
- **Effektives gleitendes Fenster:** Bei konstantem Abklingfaktor $a$ hat ein vor $\ell$ Schritten geschriebenes Paar das Gewicht $a^\ell$. Die Summe der Gewichte beträgt $\frac{1}{1 - a}$ ($10$ Paare bei $a = 0.9$, $100$ Paare bei $a = 0.99$).

<!--en:88fafa97-->
- **Beschränkte Interferenz im stationären Zustand:** Die Summe der quadrierten Gewichte beträgt $\frac{1}{1 - a^2}$. Damit ist die erwartete quadratische Interferenz für jede Länge $T$ durch $\frac{1}{(1 - a^2)d_k}$ nach oben beschränkt.

<!--en:2259d6f9-->
::: {.slide title="Mamba und Mamba-2 als Rekurrenzen mit Matrixzustand und Abklingen"}

<!--en:aed72d15-->
[Zustandsraummodelle und lineare Attention]{.kicker}

<!--en:b2b5f05e-->
Die Anordnung der $H$ Kanalzustände $\mathbf{h}_t^{(c)} \in \mathbb{R}^N$ des selektiven Zustandsraummodells von Mamba (§{sec_mamba}) als Zeilen einer Matrix $\mathbf{S}_t \in \mathbb{R}^{H \times N}$ zeigt genau dieselbe Speicherstruktur aus äußeren Produkten:

<!--en:aa44cf48-->
| Architektur | Wert $\mathbf{v}_t$ | Schlüssel $\mathbf{k}_t$ und Abfrage $\mathbf{q}_t$ | Abklingfaktoren für $\mathbf{S}_{t-1} \in \mathbb{R}^{d_v \times d_k}$ |

<!--en:f15cde86-->
| **Lineare Attention** | $\mathbf{v}_t \in \mathbb{R}^{d_v}$ | $\mathbf{k}_t, \mathbf{q}_t \in \mathbb{R}^{d_k}$ | Kein Abklingen ($a_t = 1$) |

<!--en:01458323-->
| **RetNet** | $\mathbf{v}_t \in \mathbb{R}^{d_v}$ | $\mathbf{k}_t, \mathbf{q}_t \in \mathbb{R}^{d_k}$ | Fester Skalar $a \in (0, 1)$ pro Kopf |

<!--en:086dc0b2-->
| **Mamba-2 / skalare GLA** | $\mathbf{v}_t \in \mathbb{R}^{d_v}$ | $\mathbf{k}_t, \mathbf{q}_t \in \mathbb{R}^{d_k}$ | Eingabeabhängiger Skalar $a_t \in (0, 1)$ pro Kopf |

<!--en:9588f14f-->
| **GLA mit vektoriellem Abklingen** | $\mathbf{v}_t \in \mathbb{R}^{d_v}$ | $\mathbf{k}_t, \mathbf{q}_t \in \mathbb{R}^{d_k}$ | Pro Schlüsselkoordinate ein Faktor: $\mathbf{S}_{t-1}\textrm{diag}(\boldsymbol{\alpha}_t)$ ($\boldsymbol{\alpha}_t \in (0, 1)^{d_k}$) |

<!--en:2e9a9697-->
| **Mamba-1 (§{sec_mamba})** | $\boldsymbol{\Delta}_t \odot \mathbf{x}_t \in \mathbb{R}^H$ | $\mathbf{B}_t, \mathbf{C}_t \in \mathbb{R}^N$ | Volle Matrix $\mathbf{E}_t = \exp(\textrm{diag}(\boldsymbol{\Delta}_t)\boldsymbol{\Lambda})$ mit einem Faktor pro Eintrag |

<!--en:4ebdb208-->
::: {.slide title="Zustandsraumdualität (SSD): Rekurrenz und maskierte duale Form"}

<!--en:37aa3886-->
[Primale und duale Form]{.kicker}

<!--en:7a0e571c-->
- **Zustandsraumdualität (Dao und Gu, 2024):** Mit Aufwand $O(T d_v d_k)$ berechnet die primale Rekurrenz $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ dieselbe Ausgabe wie die duale Form mit Aufwand $O(T^2 d)$: Die kausal maskierte lineare Attention liefert exakt $\mathbf{Y}$ mit $L_{ts} = \prod_{r=s+1}^t a_r$.

<!--en:5d9bdb44-->
::: {.slide title="Blockweise semiseparable Struktur und blockweises Training"}

<!--en:a1391d3c-->
[Hardwareeffiziente Blockaufteilung]{.kicker}

<!--en:3e604d70-->
Die Aufteilung von $T$ Token in $T/m$ Blöcke der Länge $m$ verbindet Matrizenmultiplikationen auf Tensor Cores mit linearer Skalierung:

<!--en:13cebff9-->
1. **Innerhalb der Blöcke (blaue $m \times m$-Diagonalblöcke):** $(\mathbf{L}_{\text{diag}} \odot \mathbf{Q}_c \mathbf{K}_c^\top)\mathbf{V}_c$ für alle Blöcke parallel durch Matrizenmultiplikation im Batch berechnen.

<!--en:ca65f455-->
2. **Scan über die Blöcke (orangefarbene Blöcke unterhalb der Diagonalen):** Da sich $L_{ts}$ an Blockgrenzen faktorisieren lässt, hat jeder Block unterhalb der Diagonalen von $\mathbf{L}$ **Rang $1$** (Rang $\le d_k$ im entsprechenden Block von $\mathbf{L}\odot\mathbf{Q}\mathbf{K}^\top$). Der Scan über die $d_v \times d_k$-Zustandsmatrizen $\mathbf{S}^{(c)} = \gamma^{(c)}\mathbf{S}^{(c-1)} + \text{writes}^{(c)}$ am Ende der $T/m$ Blöcke benötigt $\lceil\log_2(T/m)\rceil$ Runden.

<!--en:5e3e535d-->
3. **Auslesen des vorherigen Zustands:** An jeder Position im Block $(a_1\cdots a_t)\mathbf{S}^{(c-1)}\mathbf{q}_t$ addieren.

<!--en:cc29b9e4-->
::: {.slide title="Numerische Äquivalenz der rekurrenten, dualen und blockweisen Form"}

<!--en:1b718c47-->
[Numerische Überprüfung]{.kicker}

<!--en:6b4b4ab2-->
Vergleich der sequenziellen Berechnung mit `linear_attention_recurrent`, der vollen $T \times T$-Form mit `linear_attention_dual` und der blockweisen Berechnung mit `chunked_linear_attention` ($m \in \{16, 64, 256\}$) für $T = 512, d_k = d_v = 64$:

<!--en:0af3e80b-->
Alle drei Algorithmen stimmen bis auf einen relativen Fehler von $\sim 10^{-6}$ überein. Die blockweise Berechnung senkt den maximalen Arbeitsspeicherbedarf von $O(T^2)$ auf $O(Tm + (T/m)d_v d_k)$.

<!--en:3f70b439-->
::: {.slide title="Der Sequenzmischer Gated Linear Attention (GLA)"}

<!--en:dc7d6db5-->
[Schichtaufbau]{.kicker}

<!--en:5ea195b5-->
- **Kausale Faltung der Breite 4 und SiLU:** Binden benachbarte Schlüssel- und Werttoken $(t-1, t)$, bevor das äußere Produkt geschrieben wird.

<!--en:956f11f4-->
- **$\ell_2$-normierte $\mathbf{Q}, \mathbf{K}$ und logarithmisch gleichmäßig verteilte Abklingfaktoren:** Die Schlüssel haben Einheitsnorm. Die anfänglichen Zeithorizonte der Köpfe decken den Bereich $\tau_h \in [4, 256]$ ab.

<!--en:354fabcf-->
- **RMSNorm und SiLU-Gatter pro Kopf:** Stabilisieren die Größenordnung der ausgelesenen Werte vor der Ausgabeprojektion $\mathbf{W}_o$.

<!--en:8532f0d7-->
::: {.slide title="Versuchsaufbau für den assoziativen Abruf mit mehreren Abfragen"}

<!--en:00ffc4b0-->
[Synthetischer Belastungstest]{.kicker}

<!--en:806ac4e2-->
Folgt ein trainiertes neuronales Netz dem Kapazitätsgesetz $O(d_k d_v)$ für zufällige Schlüssel mit Einheitsnorm? Als Test dient der **assoziative Abruf mit mehreren Abfragen** (Arora et al., 2024):

<!--en:a4f0b65c-->
- **Sequenzformat:** Auf $n$ Schlüssel-Wert-Paare (verschiedene Schlüsseltoken aus $128$ möglichen Schlüsseln, jeweils gefolgt von einem Werttoken aus $4$ Klassen) folgen Fülltoken und schließlich erneut alle $n$ Schlüssel in zufälliger Reihenfolge als Abfragen.

<!--en:a66ce3e5-->
::: {.slide title="Variable Zahl von Zustandseinträgen bei konstanter Parameteranzahl"}

<!--en:a69fdbcb-->
[Versuchsaufbau und Ergebnisse]{.kicker}

<!--en:7809603f-->
Ein `SeqLM` mit 2 Blöcken hat die feste Breite $d = 32$ (etwa $37{,}000$ Parameter). Die Breite verteilt sich auf $h \in \{8, 4, 2\}$ Köpfe der Dimension $d_h = 32/h \in \{4, 8, 16\}$. Jeder Kopf speichert eine $d_h \times d_h$-Matrix. Damit skaliert die Zahl der Zustandseinträge pro Schicht gemäß $h \cdot d_h^2 \in \{128, 256, 512\}$, bei **nahezu identischer Parameteranzahl**:

<!--en:da2dcc75-->
::: {.slide title="Trainierte GLA folgt dem Kapazitätsgesetz bis zur erneuten Bindung"}

<!--en:691c7277-->
[Abrufgenauigkeit und nächster Schritt]{.kicker}

<!--en:5645bf42-->
Die Kurven für den Abruf mit mehreren Abfragen bestätigen das theoretische Skalierungsgesetz und zeigen eine verbleibende Schwäche:

<!--en:9f5fb2eb-->
- **Die Kapazität skaliert mit der Zahl der Zustandseinträge $h \cdot d_h^2$:** Bei $4$ Paaren erreichen alle drei Kopfkonfigurationen eine Genauigkeit von $\ge 98\%$. Bei $64$ Paaren erhöht eine Verdopplung der Zahl der Zustandseinträge von $128$ ($8 \times 4^2$) auf $256$ ($4 \times 8^2$) die Trefferquote von $\sim 0.50$ auf $\sim 0.75$. Bei fester Genauigkeit lassen sich ungefähr doppelt so viele Paare abrufen, ohne einen einzigen zusätzlichen trainierbaren Parameter.

<!--en:1102ea04-->
- **Additives Schreiben verhindert das Überschreiben:** Da $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ äußere Produkte nur *addiert*, liefert ein bereits vorhandener Schlüssel $\mathbf{k}$ nach der Bindung an einen neuen Wert $\mathbf{v}^{\text{new}}$ eine Überlagerung des alten und des neuen Werts. Der Matrixzustand soll lernen, **vor dem Schreiben zu lesen**. Das ist das Ziel von §{sec_deltanet}.

<!--en:3245cd82-->
::: {.slide title="Verständnisfragen zu Kapazität, Speicherbedarf und Dualität"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:aa6b5a39-->
Die folgenden drei Fragen bearbeiten:

<!--en:5916cec1-->
1. Bei Breite $d = 256$ die Zahl der Zustandseinträge pro Schicht vergleichen: eine `GatedLinearAttention`-Schicht mit 8 Köpfen ($d_h = 32$) und eine Schicht mit 4 Köpfen ($d_h = 64$). Welche speichert mehr Schlüssel-Wert-Paare und um welchen Faktor?

<!--en:793b0633-->
2. Ein Kopf arbeitet mit $T = 4{,}096$ und $d_k = d_v = 64$ in `float32` ($4$ Byte). Wie viel Speicherplatz belegt die volle duale $T \times T$-Maske $\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top$ im Vergleich zur blockweisen Auswertung mit Blockgröße $m = 64$?

<!--en:6a3e45e9-->
3. Bei konstantem Abklingfaktor $a = 0.95$: Wie groß ist die effektive Anzahl $1/(1 - a)$ der erhaltenen früheren Paare, und wie groß ist die maximale erwartete quadratische Interferenz $1/((1 - a^2)d_k)$ bei $d_k = 64$?

<!--en:557b98b6-->
**Antworten:**

<!--en:803a563c-->
1. $8$ Köpfe der Größe $32 \times 32$ speichern $8 \times 1{,}024 = \mathbf{8{,}192\text{ entries}}$. Für $4$ Köpfe der Größe $64 \times 64$ sind es $4 \times 4{,}096 = \mathbf{16{,}384\text{ entries}}$, also **doppelt so viele Zustandseinträge und eine um den Faktor $\sim 2\times$ höhere Paarkapazität** bei exakt gleicher Parameteranzahl in $W_{qkv}$.

<!--en:3b09942a-->
2. Volle duale Matrix: $4{,}096^2 \times 4\text{ B} = \mathbf{64\text{ MB}}$. Blockweise ($64$ Blöcke mit Masken der Größe $64 \times 64$ und $64$ Blockzustände der Größe $64 \times 64$): $2 \times 64 \times 64^2 \times 4\text{ B} = \mathbf{2\text{ MB}}$ (um den Faktor $32\times$ kleiner, linear in $T$).

<!--en:056b13fc-->
3. Effektive Fenstergröße: $1/(1 - 0.95) = \mathbf{20\text{ pairs}}$. Obergrenze der Interferenz im stationären Zustand: $1 / ((1 - 0.95^2) \times 64) \approx 1 / (0.0975 \times 64) \approx \mathbf{0.16}$.

<!--en:633073dc-->
::: {.slide title="Zusammenfassung: Rekurrenzen mit Matrixzustand und lineare Attention"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:155c8c60-->
| Eigenschaft oder Mechanismus | Mathematische Form | Skalierung und Rolle im Modell |

<!--en:64c7f360-->
| **Speicher mit äußeren Produkten** | $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top \in \mathbb{R}^{d_v \times d_k}$, $\mathbf{y}_t = \mathbf{S}_t \mathbf{q}_t$ | Äquivalent zur kausalen linearen Attention $\sum_{s \le t} (\mathbf{k}_s^\top \mathbf{q}_t)\mathbf{v}_s$ |

<!--en:cd63b668-->
| **Interferenz bei zufälligen Schlüsseln** | $\mathbb{E}\|\mathbf{S}\mathbf{k}_j - \mathbf{v}_j\|^2 = (n - 1)/d_k$ | Die Projektion des isotropen Fehlers auf einen Codebuchvektor hat Varianz $(n-1)/(d_k d_v)$ |

<!--en:583391e8-->
| **Skalierung der Abrufkapazität** | $n_{\max} \propto d_k d_v$ pro Kopf ($d^2 / h$ pro Schicht) | Weniger, breitere Köpfe speichern bei gleicher Parameteranzahl proportional mehr Paare |

<!--en:e7a535fd-->
| **Zustandsraumdualität (SSD)** | $\mathbf{Y} = (\mathbf{L} \odot \mathbf{Q}\mathbf{K}^\top)\mathbf{V}$, $L_{ts} = \prod_{r=s+1}^t a_r$ | Verbindet die Matrixrekurrenz mit Abklingen und die maskierte lineare Attention |

<!--en:de2f4c6a-->
| **Blockweises Training** | Matrizenmultiplikationen in Diagonalblöcken der Länge $m$ und Scan über $(T/m)$ Schritte | Abklingblöcke vom Rang 1 unterhalb der Diagonalen ermöglichen $O(T)$ Speicherbedarf und die hohe Rechengeschwindigkeit der Tensor Cores |
