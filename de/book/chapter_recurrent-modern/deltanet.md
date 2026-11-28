<!--en:a7514c01-->
# Die Delta-Regel

<!--en:8c450eec-->
Die additiven Matrixspeicher (engl. matrix memories) aus :numref:`sec_matrix-state` speichern jedes neue Paar, indem sie $\mathbf{v}_t \mathbf{k}_t^\top$ zum bisherigen Inhalt von $\mathbf{S}_{t-1}$ addieren. Das funktioniert, solange jeder Schlüssel nur einmal einen Wert erhält. In Texten und Programmen werden Schlüssel jedoch häufig *neu belegt* (engl. rebind). Ein Programm setzt `rate = 0.05` und später `rate = 0.07`. In einer Geschichte zieht eine Figur von Paris nach Rom. Wenn wir additiv schreiben und einen zweimal belegten Schlüssel abfragen, erhalten wir die Summe aus altem und neuem Wert. Bei wiederholten Aktualisierungen überlagern immer mehr veraltete Werte den jüngsten. Wie kann ein Matrixspeicher fester Größe den Wert unter einem Schlüssel überschreiben?

<!--en:dbcf1af4-->
Die Lösung lautet: *vor dem Schreiben lesen*. Wir berechnen die aktuelle Vorhersage des Speichers für $\mathbf{k}_t$ ($\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}\mathbf{k}_t$) und schreiben nur den Vorhersagefehler $\beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$. Mit dieser *Delta-Regel* :cite:`Widrow.Hoff.1960,Schlag.Irie.Schmidhuber.2021` führen wir während des Vorwärtsdurchlaufs einen Online-Schritt des Gradientenabstiegs für den quadratischen Abrufverlust (engl. recall loss) $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ aus. Wir leiten die Geometrie der Projektion her und messen an zufälligen und an korrelierten Schlüsseln, wie viel die Regel bei der Neubelegung gewinnt und wie viel Signal dabei verloren geht. In `GatedDeltaNet` :cite:`Yang.Kautz.Hatamizadeh.2024` kombinieren wir sie mit einem skalaren Abklingfaktor und lösen zur Auswertung blockweise Dreieckssysteme :cite:`Yang.Wang.Zhang.ea.2024`. Zum Schluss betrachten wir Sequenzspeicher einheitlich als Verfahren zur Online-Regression.

<!--en:addf4128-->
## Additives Schreiben

<!--en:eddc9cf6-->
### Ein zweimal belegter Schlüssel

<!--en:a29e2b5a-->
Untersuchen wir zunächst, wie sich additives Schreiben bei der Neubelegung eines Schlüssels auswirkt. Schreiben wir unter einem auf Eins normierten Schlüssel $\mathbf{k}$ insgesamt $R$-mal die Werte $\mathbf{v}^{(1)}, \ldots, \mathbf{v}^{(R)}$, so liefert ein additiver Speicher ihre ungewichtete Summe:

<!--en:cff7d725-->
Hinzu kommen die Störbeiträge $\sum_i (\mathbf{k}_i^\top \mathbf{k})\mathbf{v}_i$ der anderen Schlüssel. Ein skalarer Abklingfaktor (engl. scalar decay) $a_t \in (0, 1)$ schwächt ältere Werte gegenüber neueren ab. Er behandelt jedoch *alle* gleich alten Schlüssel gleich. Einmal gespeicherte Fakten verblassen ebenso schnell wie veraltete Werte neu belegter Schlüssel (Übung :exref:`ex_dn-decay`).

<!--en:f3e9a211-->
Um die Neubelegung unabhängig von Störbeiträgen anderer Schlüssel zu untersuchen, testet `overwrite_recall` 8 orthonormale Schlüssel in $\mathbb{R}^{64}$. Die Funktion schreibt unter jedem Schlüssel $R \in \{1, 2, 4, 8\}$-mal einen Wert aus einem Codebuch mit 32 Einheitsvektoren. Sie misst, wie oft der nächste Nachbar von $\mathbf{S}\mathbf{k}$ im Codebuch dem jüngsten Wert $\mathbf{v}^{(R)}$ entspricht.

<!--en:b915f5a8-->
Obwohl sich die 8 orthonormalen Schlüssel in $\mathbb{R}^{64}$ gegenseitig gar nicht stören, fällt die Trefferquote beim additiven Schreiben von $1.00$ bei $R = 1$ auf $0.51$ bei $R = 2$ (wie bei einem Münzwurf zwischen den beiden summierten Werten) und auf $0.21$ bei $R = 8$ (Übung :exref:`ex_dn-order`).

<!--en:8acc7bc5-->
## Schreibvorgänge als Gradientenschritte

<!--en:bb340002-->
### Die Delta-Regel

<!--en:f57461fa-->
Wir möchten den Wert unter $\mathbf{k}_t$ ersetzen und dabei andere Richtungen möglichst wenig stören. Dazu messen wir den quadratischen Vorhersagefehler des Speichers für das neue Paar $(\mathbf{k}_t, \mathbf{v}_t)$:

<!--en:1e62642d-->
Dann gehen wir einen Schritt des Gradientenabstiegs $\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \nabla_{\mathbf{S}} \ell_t(\mathbf{S}_{t-1})$ mit der Schrittweite (*Schreibstärke*, engl. write strength) $\beta_t$:

<!--en:d9cace69-->
Das ist die *Delta-Regel* :cite:`Widrow.Hoff.1960,Schlag.Irie.Schmidhuber.2021`: Wir lesen die aktuelle Vorhersage $\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}\mathbf{k}_t$ und addieren die Rang-eins-Korrektur $\beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$. Dieser Gradientenschritt geschieht *innerhalb des Vorwärtsdurchlaufs*, für jedes Token sowohl beim Training als auch bei der Inferenz. Für einen normierten Schlüssel $\|\mathbf{k}_t\| = 1$ liefert eine Abfrage von $\mathbf{k}_t$ unmittelbar nach :eqref:`eq_dn-delta` den Wert $\mathbf{S}_t \mathbf{k}_t = (1 - \beta_t)\hat{\mathbf{v}}_t + \beta_t \mathbf{v}_t$. Bei $\beta_t = 1$ ersetzt der neue Wert den alten exakt, bei $\beta_t \in (0, 1)$ entsteht eine Konvexkombination. (Additives Schreiben mit $\mathbf{v}_t \mathbf{k}_t^\top$ entspricht dagegen einem Schritt des Gradientenaufstiegs für das Skalarprodukt $\mathbf{v}_t^\top \mathbf{S}\mathbf{k}_t$, das keinen Sättigungspunkt hat.)

<!--en:76991b4b-->
### Der Zustandsübergang

<!--en:1fe3a9c3-->
Klammern wir $\mathbf{S}_{t-1}$ in :eqref:`eq_dn-delta` aus, erkennen wir die Übergangsmatrix, die auf den bisherigen Speicher wirkt:

<!--en:32eb5d1a-->
Für einen auf Eins normierten Schlüssel $\mathbf{k}_t$ lässt die symmetrische Matrix $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ jeden zu $\mathbf{k}_t$ orthogonalen Unterraum punktweise fest (Eigenwert $1$). Die Richtung $\mathbf{k}_t$ skaliert sie mit dem Eigenwert $1 - \beta_t$ (:numref:`fig_dn-geometry`):

<!--en:a57359f2-->
- $\beta_t = 0$: Identität (kein Schreibvorgang).

<!--en:1f04d7b4-->
- $\beta_t \in (0, 1)$: teilweise Kontraktion entlang $\mathbf{k}_t$.

<!--en:fe753b0e-->
- $\beta_t = 1$: orthogonale Projektion $\mathbf{I} - \mathbf{k}_t \mathbf{k}_t^\top$. Sie löscht die Komponente entlang $\mathbf{k}_t$ vollständig, bevor wir $\mathbf{v}_t \mathbf{k}_t^\top$ schreiben.

<!--en:e62cd025-->
- $\beta_t = 2$: Householder-Spiegelung $\mathbf{I} - 2\mathbf{k}_t \mathbf{k}_t^\top$ (Eigenwert $-1$). Sie kehrt das Vorzeichen entlang $\mathbf{k}_t$ um.

<!--en:c79aec36-->
![Der Zustandsübergang $\mathbf{I} - \beta\mathbf{k}\mathbf{k}^\top$ wirkt auf einen Vektor $\mathbf{s}$ in der Ebene, die ein normierter Schlüssel $\mathbf{k}$ und eine dazu orthogonale Richtung aufspannen. Die Komponente entlang $\mathbf{k}$ wird mit $1 - \beta$ multipliziert. Bei $\beta = 0$ bleibt sie erhalten, bei $\beta = 0.5$ halbiert sie sich, bei $\beta = 1$ verschwindet sie und bei $\beta = 2$ kehrt sich ihr Vorzeichen um. Die orthogonale Komponente bleibt unverändert.](../img/mdl-modernrnn-delta-geometry.svg)

<!--en:aec4f32b-->
Aus dem Eigenwert $1 - \beta_t\|\mathbf{k}_t\|^2$ folgen zwei Eigenschaften:

<!--en:bcc7b4af-->
1. **Schlüssel mit Norm eins verhindern Instabilität:** Für $\|\mathbf{k}_t\| = 1$ und $\beta_t \in [0, 2]$ liegen alle Eigenwerte der Übergangsmatrix in $[-1, 1]$. Die homogene Rekursion lässt Speicherzustände daher nie geometrisch anwachsen.

<!--en:cbc20745-->
2. **Negative Eigenwerte ermöglichen das Zählen modulo 2 (Parität):** Für $\beta_t \in [0, 1]$ sind alle Eigenwerte der Übergangsmatrix nichtnegativ. Bei endlicher Rechengenauigkeit lässt sich die Parität damit nicht für beliebige Sequenzlängen verfolgen :cite:`Grazzi.Siems.Franke.ea.2025`. Erweitern wir den Wertebereich von $\beta_t$ bis $2$, werden Spiegelungen möglich (Eigenwert $-1$). Sie schalten den gespeicherten Zustand bei jedem gezählten Token um und berechnen so die Parität für jede Länge exakt. In :numref:`subsec_dn-state-tracking` testen wir, ob eine trainierte Schicht diese Lösung findet.

<!--en:c0e1690e-->
### Der Preis der Korrektur

<!--en:1d613e91-->
Bei orthonormalen Schlüsseln löscht $\beta_t = 1$ veraltete Werte neu belegter Schlüssel, ohne andere Schlüssel zu beeinflussen. Was geschieht, wenn $n > d_k$ gilt oder die Schlüssel zufällige Einheitsvektoren sind, sodass verschiedene Schlüssel $\mathbf{k}_i$ und $\mathbf{k}_t$ eine Überlappung $\mathbf{k}_t^\top \mathbf{k}_i$ ungleich null haben? Jeder spätere Delta-Schritt entlang $\mathbf{k}_t$ entfernt durch Projektion die Komponente in Richtung $\mathbf{k}_t$ aus $\mathbf{v}_i \mathbf{k}_i^\top$. Dadurch schrumpft das gespeicherte Signal von Schlüssel $i$ pro Schritt um den Faktor $1 - (\mathbf{k}_t^\top \mathbf{k}_i)^2 \approx 1 - 1/d_k$. Nach $m$ weiteren Schreibvorgängen bleibt vom Signal eines alten Paares nur der Anteil $(1 - 1/d_k)^m \approx e^{-m/d_k}$.

<!--en:7e6fe857-->
Wann hilft gezieltes Löschen, wann schadet es? Wir implementieren mit `delta_rule(beta)` die Online-Schreibvorgänge der Delta-Regel. Mit `correction_cost` messen wir bei der Breite $d = 64$ den quadratischen Lesefehler und die Trefferquote beim Abruf über den nächsten Nachbarn. Dabei variieren wir die Anzahl der Schreibrunden und die Korrelation der Schlüssel:

<!--en:1441e30b-->
Wir vergleichen vier Schreibregeln: `additive`, `delta, β = 1`, `delta, β = 0.5` und das Offline-Verfahren `least squares`, bei dem `np.linalg.lstsq` den Fehler $\sum_{s} \|\mathbf{S}\mathbf{k}_s - \mathbf{v}_s\|^2$ für alle Paare gleichzeitig minimiert. Wir betrachten drei Fälle: (1) 8 orthonormale Schlüssel, die wir je 4-mal belegen, (2) $n \in \{32, 64, 128, 256\}$ unabhängige zufällige Schlüssel mit Norm eins, die wir je einmal belegen, und (3) 64 korrelierte Schlüssel ($\rho = 0.5$), die wir ebenfalls je einmal belegen:

<!--en:d27535cd-->
Die Tabelle zeigt die Vor- und Nachteile der Online-Projektion:

<!--en:9033da41-->
- **Neubelegung (`8 orthonormal keys, 4 writes`):** `delta` erreicht eine Trefferquote von $1.00$ sowohl bei $\beta = 1$ als auch bei $\beta = 0.5$ (quadratischer Fehler $0.00$ beziehungsweise $0.33$). Dagegen gewichten `additive` und die ungewichtete Methode `least squares` alle 4 Werte eines Schlüssels gleich und erreichen nur $0.28$.

<!--en:359c6209-->
- **Korrelierte Schlüssel (`64 keys, ρ = 0.5`):** Weil alle Schlüssel eine gemeinsame Richtung haben, summiert `additive` jeden Wert auf dieser Komponente auf (quadratischer Fehler $5.82$, Trefferquote $0.54$). Dagegen zieht `delta` vor dem Schreiben die Vorhersage entlang der gemeinsamen Richtung ab (Trefferquote $0.77$ bei $\beta = 1$ und $0.89$ bei $\beta = 0.5$).

<!--en:ed75b2fa-->
- **Viele unabhängige Schlüssel auf engem Raum (`128 keys`, `256 keys`):** Hier hat `delta, β = 1` einen kleineren quadratischen Fehler als `additive` ($1.06$ gegenüber $1.99$ bei $n = 128$), aber eine *niedrigere* Trefferquote beim Abruf über den nächsten Nachbarn ($0.53$ gegenüber $0.96$). Warum? Wenn die ausgelesenen alten Werte gegen $\mathbf{0}$ schrumpfen, sinkt ihr quadratischer Fehler in Richtung $\|\mathbf{0} - \mathbf{v}_j\|^2 = 1$. Zugleich schwächt sich das Signal früher Paare gegenüber dem späterer Paare ab. Ein kleinerer Schritt $\beta = 0.5$ senkt den quadratischen Fehler weiter ($0.67$) und erhöht die Trefferquote deutlich ($0.82$). Es liegt also nahe, $\beta_t$ als eingabeabhängiges Gate zu lernen. Das Netz kann dann mit $\beta_t \approx 1$ einen Schlüssel neu belegen und mit kleinerem $\beta_t$ Fakten sammeln.

<!--en:e86329ae-->
## Gated DeltaNet

<!--en:8fbf6ef5-->
### Abklingen und Korrektur

<!--en:dfe58852-->
Die Delta-Regel verändert $\mathbf{S}_{t-1}$ nur entlang des aktuellen Schlüssels $\mathbf{k}_t$. In Richtungen, die zu allen jüngeren Schlüsseln orthogonal sind, klingt der Speicher daher nie von selbst ab. *Gated DeltaNet* :cite:`Yang.Kautz.Hatamizadeh.2024` kombiniert einen eingabeabhängigen skalaren Abklingfaktor $a_t \in (0, 1)$ für globales Vergessen mit der gerichteten Rang-eins-Korrektur der Delta-Regel zum gezielten Überschreiben:

<!--en:5e0c0ee7-->
Die beiden Gates erfüllen unterschiedliche Aufgaben. Je nach Einstellung kann die Schicht ein neues Paar auf drei Arten behandeln (:numref:`fig_dn-gate-modes`). Liegt $a_t$ nahe null, leert die Schicht ihren Speicher und behält nur das neue Paar. Das ist bei einem Themenwechsel sinnvoll. Bei $a_t \approx 1$ und $\beta_t = 1$ ersetzt sie den Wert unter dem aktuellen Schlüssel und lässt orthogonale Schlüssel unberührt, wie im Versuch zur Neubelegung. Bei $a_t \approx 1$ und kleinem $\beta_t$ verschiebt sich der Auslesewert nur ein Stück in Richtung des neuen Werts. Schreiben wir mehrmals unter ähnlichen Schlüsseln, entsteht so ein Mittelwert, statt dass sich der letzte Wert durchsetzt.

<!--en:ea903058-->
![Ein Schritt von Gated DeltaNet auf einem $2 \times 2$-Speicher bei drei Einstellungen der Gates. Die Pfeile zeigen die Auslesewerte entlang des aktuellen Schlüssels $\mathbf{k}_t$ (blau) und eines orthogonalen Schlüssels (violett), jeweils vor dem Schritt (gestrichelt) und danach (durchgezogen). Ein Abklingfaktor $a_t$ nahe null leert den Speicher, sodass nur das neue Paar bleibt. Bei $a_t \approx 1$ ersetzt die Schreibstärke $\beta_t = 1$ den Wert unter $\mathbf{k}_t$ durch $\mathbf{v}_t$ und lässt den anderen Schlüssel unberührt. Ein kleines $\beta_t$ verschiebt den Auslesewert nur ein Stück in Richtung $\mathbf{v}_t$.](../img/mdl-modernrnn-gated-delta-modes.svg)

<!--en:5dbd7fcc-->
### Blockweise Berechnung

<!--en:1e26dab2-->
Da die Übergangsmatrix $a_t(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$ eine $d_k \times d_k$-Matrix ist und kein Skalar, müsste ein einfacher Scan über die Token $d_k \times d_k$-Matrizen multiplizieren. Das kostet $O(T d_k^3)$ Operationen. :citet:`Yang.Wang.Zhang.ea.2024` zeigten, wie wir :eqref:`eq_dn-gated` mit der *WY-Darstellung* in Blöcken der Größe $m$ auswerten können. Dazu schreiben wir :eqref:`eq_dn-gated` als additive Aktualisierung mit Abklingen $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{u}_t \mathbf{k}_t^\top$ und dem *effektiven korrigierten Wert* $\mathbf{u}_t = \beta_t(\mathbf{v}_t - a_t \mathbf{S}_{t-1}\mathbf{k}_t)$. Betrachten wir einen Block der Länge $m$ mit Anfangszustand $\mathbf{S}_0$. Durch Aufrollen der Rekursion erhalten wir $a_t \mathbf{S}_{t-1}\mathbf{k}_t = \gamma_t \mathbf{S}_0 \mathbf{k}_t + \sum_{s < t} L_{ts}(\mathbf{k}_s^\top \mathbf{k}_t)\mathbf{u}_s$ mit $L_{ts} = \prod_{r=s+1}^t a_r$ und $\gamma_t = \prod_{r=1}^t a_r$. Daraus ergibt sich ein unteres lineares $m \times m$-Dreieckssystem mit Einsen auf der Diagonale für die korrigierten Werte $\mathbf{u}_1, \ldots, \mathbf{u}_m$ des Blocks:

<!--en:03c87c24-->
Die rechte Seite ist affin in $\mathbf{S}_0$. Daher löst `delta_rule_writes` das untere $m \times m$-Dreieckssystem mit der Matrix $\mathbf{M} = \mathbf{I} + \textrm{diag}(\boldsymbol{\beta})\,\textrm{tril}_{-1}(\mathbf{L} \odot \mathbf{K}\mathbf{K}^\top)$ und Einsen auf der Diagonale für zwei rechte Seiten parallel über alle Blöcke und Köpfe. Die beiden Lösungen sind $\mathbf{W} = \mathbf{M}^{-1}(\boldsymbol{\beta} \odot \mathbf{V})$ und $\mathbf{Z} = \mathbf{M}^{-1}((\boldsymbol{\beta} \odot \boldsymbol{\gamma}) \odot \mathbf{K})$. Damit ergibt sich $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$, sobald $\mathbf{S}_0$ bekannt ist.

<!--en:38c5a861-->
Setzen wir $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ in den Zustand am Blockende $\mathbf{S}_m = \gamma_m \mathbf{S}_0 + \mathbf{U}^\top \mathbf{D}\mathbf{K}$ ein. Dabei ist $\mathbf{D}$ die Diagonalmatrix der Abklingfaktoren bis zum Blockende. So erhalten wir für den Übergang zwischen den $T / m$ Blöcken eine affine Matrixrekursion:

<!--en:82ede4ce-->
Der Scan von $(\mathbf{A}, \mathbf{W}^\top \mathbf{D}\mathbf{K})$ über die $T / m$ Blöcke kostet nur $(T/m) d_k^3$ Operationen. Das entspricht dem Aufwand $O(T d_k^2)$ für die Matrixprodukte innerhalb der Blöcke, wenn $m \approx d_k$ gilt. Anschließend erhalten wir $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ und alle Ausgaben $\mathbf{Y}$ mit einem gebündelten Matrixprodukt.

<!--en:90e5770f-->
Um `chunked_delta_rule` zu überprüfen, implementieren wir zunächst die Referenzversion `delta_rule_recurrent`, die ein Token nach dem anderen verarbeitet:

<!--en:1e07fc2a-->
Nun vergleichen wir `chunked_delta_rule` für Blockgrößen $m \in \{16, 64\}$ mit `delta_rule_recurrent`. Wir verwenden eine zufällige Sequenz der Länge $T = 256$ mit auf Eins normierten Schlüsseln:

<!--en:1a831ded-->
Der blockweise Algorithmus stimmt bis auf einen relativen Fehler von $\sim 10^{-7}$ mit der Rekursion über einzelne Token überein.

<!--en:2cdc020d-->
### Die Schicht

<!--en:f9495a07-->
`GatedDeltaNet` erbt von `d2l.GatedLinearAttention` (:numref:`sec_matrix-state`). Die Klasse ergänzt nur die Projektion `W_beta` für die Schreibstärke ($\beta_{t,h} = \sigma(\mathbf{w}_{\beta,h}^\top \mathbf{x}_t + b_{\beta,h}) \in (0, 1)$ pro Kopf) und überschreibt `memory`, um `chunked_delta_rule` aufzurufen. Alle anderen Komponenten sind identisch: die kausale Faltung der Breite 4, die $\ell_2$-normierten Abfragen und Schlüssel, der skalare Abklingfaktor pro Kopf, die RMS-Normalisierung pro Kopf und das SiLU-Ausgabegatter.

<!--en:08491a83-->
### Neubelegung

<!--en:b55ef495-->
Kann eine trainierte `GatedDeltaNet`-Schicht Schlüssel dort neu belegen, wo `GatedLinearAttention` Schwierigkeiten hat? Wir testen ein `SeqLM` mit einem Block ($d = 64$, 4 Köpfe der Dimension 16, anfängliche Zeitskalen des Abklingens $[256, 2048]$) an der Aufgabe `d2l.associative_recall`. Wir ziehen 16 von 128 Schlüsseln und wählen die Werte aus 16 Möglichkeiten. 8 der 16 Schlüssel erhalten schon 2 Schreibvorgänge nach ihrer ersten Belegung einen neuen Wert (`num_rebound=8, gap=2`). Bei diesem kurzen Abstand sind beide Belegungen fast gleich alt. Durch globales, altersabhängiges Abklingen lässt sich der neue Wert daher kaum vom veralteten trennen, ohne auch einmal belegte Schlüssel zu vergessen. (Wir verwenden einen einzigen Block, weil ein additives Modell mit zwei Blöcken über die Tiefe einen Delta-Schritt annähern kann, siehe Übung :exref:`ex_dn-rebinding`.)

<!--en:5a3d9dae-->
Mit `recall_by_kind` messen wir die Genauigkeit für neu belegte Schlüssel (`rebound`) und für nur einmal belegte Schlüssel (`~rebound`) getrennt. Wir trainieren `GatedLinearAttention` und `GatedDeltaNet` jeweils mit zwei Zufallsstartwerten für die Aufgabe zur Neubelegung mit 16 Paaren:

<!--en:f35b0ad3-->
Der zusätzliche Kopf `W_beta` mit 260 Parametern erhöht die Trefferquote für neu belegte Schlüssel von etwa $0.9$ (`gated linear attention`) auf $0.96$ oder mehr (`Gated DeltaNet`). Damit liegen neu belegte Schlüssel nur noch etwa $2$ Prozentpunkte hinter den einmal belegten.

<!--en:7df486a6-->
### Zustandsverfolgung

<!--en:74e16226-->
In allen bisherigen Versuchen sollte ein Speicher etwas ablegen und es später wieder ausgeben. Eine andere Aufgabe ist es, einen Zustand zu *verfolgen*, den jedes Token ändern kann: Unter welchem Becher liegt die Kugel, nachdem wir die Becher mehrmals vertauscht haben? Ist ein Zitat noch offen? Oder im einfachsten Fall: Ist die Anzahl der bisher gelesenen Einsen gerade oder ungerade? Für diese Parität genügt ein einziges Zustandsbit. Eine 1 schaltet es um, eine 0 lässt es unverändert. Nichts muss lange gespeichert bleiben, aber wir dürfen auch nichts überspringen. Ein einziges übersehenes Bit macht alle späteren Antworten falsch.

<!--en:5c22d934-->
Eine GRU löst diese Aufgabe wie ein Programm. Ihre Gatter kennen den vorigen Zustand $\mathbf{h}_{t-1}$. So lässt sich in einem Schritt die Anweisung „Bei Eingabe 1 umschalten“ ausführen. Die linearen Rekursionen dieses Kapitels verzichten zugunsten des parallelen Trainings auf diesen Zugriff: Ihre Abklingfaktoren und Schreibvorgänge hängen allein von der Eingabe ab (:numref:`sec_linear-recurrence`). Um ein gespeichertes Bit umzuschalten, muss eine solche Rekursion eine Zustandskoordinate mit einer negativen Zahl multiplizieren. Der Abklingfaktor $\mathbf{z}_t \in (0, 1)$ einer minGRU kann eine Koordinate abschwächen, aber nie ihr Vorzeichen ändern. Dieselbe Einschränkung gilt für die Delta-Regel, solange die Sigmoidfunktion die Schreibstärke bestimmt. Der Zustandsübergang :eqref:`eq_dn-transition` hat dann entlang des Schlüssels den Eigenwert $1 - \beta_t \in (0, 1)$. Mit $\beta_t \in (0, 2)$ kann der Eigenwert beliebig nahe an $-1$ herankommen. Angenommen, die Schicht schreibt bei jeder 1 den Wert $\mathbf{v}$ mit $\beta_t = 2$ unter denselben Schlüssel $\mathbf{k}$ und bei einer 0 gar nichts. Dann ändert jede 1 den Auslesewert gemäß $\mathbf{S}_t\mathbf{k} = 2\mathbf{v} - \mathbf{S}_{t-1}\mathbf{k}$. Ist der Speicher anfangs leer, wechselt der Auslesewert zwischen $2\mathbf{v}$ und $\mathbf{0}$ und gibt damit die Parität an.

<!--en:1841b126-->
Eine solche Konstruktion gibt es also. Findet das Training sie auch? Wir trainieren ein `d2l.SeqLM` mit zwei Blöcken der Breite 64, also mit derselben Tiefe wie :citet:`Grazzi.Siems.Franke.ea.2025` bei dieser Aufgabe. Für zufällige Folgen aus 64 Bits soll das Modell nach jedem Schritt die Parität der bisher gelesenen Bits ausgeben. Anschließend testen wir es an Folgen aus 256 Bits. So unterscheiden wir ein Modell, das die Regel gelernt hat, von einem, das sich nur an die Trainingslängen angepasst hat. Wir vergleichen vier Sequenzmischer: eine GRU, eine minGRU, `GatedDeltaNet` und `GatedDeltaNetReflect`. Die letzte Variante verdoppelt den Wert der Sigmoidfunktion für die Schreibstärke und ist ansonsten dieselbe Schicht. Trifft unsere Überlegung zu, sollten die GRU und die Schicht mit $\beta_t \in (0, 2)$ an jeder Position richtig liegen, auch jenseits der Trainingslänge. Die minGRU und die Delta-Regel mit einfacher Sigmoidfunktion sollten scheitern. Bei längeren Folgen als im Training sollte ihre Genauigkeit auf das Zufallsniveau $0.5$ fallen.

<!--en:504dc123-->
Wir trainieren jeden Sequenzmischer zweimal und geben die Genauigkeit getrennt für die Positionen 1–64 (die Trainingslängen) und 65–256 an:

<!--en:89905f1c-->
In beiden Läufen liefert die GRU an jeder Position die richtige Antwort, bei den Trainingslängen und darüber hinaus. Der minGRU und der Delta-Regel mit einfacher Sigmoidfunktion gelingt das nicht. Auf den ersten 64 Positionen liegen sie über Zufallsniveau: die minGRU knapp über $0.5$ und Gated DeltaNet bei etwa $0.75$ bis $0.8$. Die Regel hat keines der beiden Modelle gelernt. Jenseits von Position 64 treffen sie mit einer Genauigkeit von $0.50$, also wie bei einem Münzwurf. Verdoppeln wir die Schreibstärke, ändert sich das Bild. `GatedDeltaNetReflect` erreicht bei den Trainingslängen $1.00$ und hält diese Genauigkeit bis zu 256 Bits. Die beiden Läufe sind typisch: Von rund fünfzig weiteren Läufen generalisierten alle bis auf einen ebenso gut. Selbst die Ausnahme erreichte jenseits von Position 64 noch etwa $0.9$. Die Linearität ist also kein Hindernis. Die linearen Rekursionen verzichten auf eine zustandsabhängige Aktualisierung. Ob ein Matrixspeicher einen Zustand trotzdem verfolgen kann, hängt vom Wertebereich der Eigenwerte seines Übergangs ab. Die minGRU hat weder einen negativen Eigenwert noch ein Gatter mit Zugriff auf ihren Zustand. Auch ein zweiter Block liefert das fehlende Vorzeichen nicht, und das Modell scheitert.

<!--en:c4603589-->
Die Spiegelung reagiert allerdings empfindlich auf Abweichungen. Die verdoppelte Sigmoidfunktion nähert sich $\beta_t = 2$ nur an. Bei jedem Umschalten schrumpft daher auch das gespeicherte Bit ein wenig. Bleibt die Schreibstärke in einem Lauf unter 2, sinkt die Genauigkeit bei langen Bitfolgen (Übung :exref:`ex_dn-parity`). Schwierigere Aufgaben der Zustandsverfolgung, etwa mehrere Becher beim Vertauschen im Blick zu behalten, erfordern mehr als eine Spiegelung pro Token.

<!--en:31afd808-->
## Speicher als Regression

<!--en:e4d09b7e-->
Aus einer übergeordneten Perspektive lassen sich alle Matrixspeicher dieses Kapitels ebenso wie die Softmax-Attention als Regression verstehen: Während sie eine Sequenz lesen, passen sie ein Modell an, das aus den Schlüsseln $\mathbf{K}_t \in \mathbb{R}^{t \times d_k}$ die Werte $\mathbf{V}_t \in \mathbb{R}^{t \times d_v}$ vorhersagt :cite:`Wang.Shi.Fox.2025`. Die exakte Lösung der linearen Regression nach der Methode der kleinsten Quadrate minimiert $\sum_{s \le t} \|\mathbf{S}\mathbf{k}_s - \mathbf{v}_s\|^2$ und lautet:

<!--en:4251c516-->
Dabei ersetzen wir $(\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$ durch die Pseudoinverse, wenn $t < d_k$ gilt. Betrachten wir die beiden Faktoren in :eqref:`eq_dn-least-squares` (:numref:`tab_dn-regression`):

<!--en:7943d2dd-->
- **Lineare Attention** behält nur den ersten Faktor, das unnormierte Kreuzmoment (engl. cross-moment) $\mathbf{V}_t^\top \mathbf{K}_t = \sum_{s \le t} \mathbf{v}_s \mathbf{k}_s^\top$. Sie lässt die inverse Gram-Matrix $(\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$ weg, die überlappende Schlüssel dekorreliert.

<!--en:af53bcef-->
- **Die Delta-Regel** geht online für jedes Token einen stochastischen Gradientenschritt in Richtung :eqref:`eq_dn-least-squares`. **Gated DeltaNet** ergänzt vor jedem Schritt einen Gewichtszerfall ($a_t < 1$). Ersetzen wir den expliziten Gradientenschritt durch den proximalen impliziten Schritt $\arg\min_{\mathbf{S}} \frac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|^2 + \frac{\beta_t}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$, erhalten wir die normierte Schrittweite $\beta_t / (1 + \beta_t\|\mathbf{k}_t\|^2)$ von Longhorn :cite:`Liu.Wang.Wu.ea.2024` (Übung :exref:`ex_dn-implicit`). Führen wir dagegen $(\mathbf{K}_t^\top \mathbf{K}_t + \lambda\mathbf{I})^{-1}$ explizit mit, erhalten wir die rekursive Methode der kleinsten Quadrate (Übung :exref:`ex_dn-rls`).

<!--en:83f55fe6-->
- **Softmax-Attention** berechnet bei jeder Abfrage eine *nichtparametrische* Nadaraya-Watson Kernel-Regression (:numref:`sec_attention-pooling`) über alle gespeicherten Paare. Sie kommt ohne parametrische Kompression aus, benötigt dafür aber einen KV-Cache der Größe $O(t)$.

<!--en:ab92c337-->
Auch die exakte Lösung lässt sich mit jedem neuen Paar aktualisieren. Jedes Paar ändert $\mathbf{K}_t^\top \mathbf{K}_t$ nur um den Rang-eins-Term $\mathbf{k}_t\mathbf{k}_t^\top$. Mit der Sherman-Morrison-Formel (:numref:`subsec_mdl-block-matrices`) aktualisieren wir daher die Inverse, ohne erneut eine Matrix zu invertieren. Das ist die rekursive Methode der kleinsten Quadrate. Sie führt neben dem Speicher einen zweiten Zustand der Größe $d_k \times d_k$ mit (Übung :exref:`ex_dn-rls`).

<!--en:c87f7ae0-->
:Speicher als Regressionsverfahren für die bisher gelesenen Paare.

<!--en:99d76f8d-->
| Speicher | Was er schätzt | Verfahren | Zustand |

<!--en:e898c6e8-->
| Softmax-Attention | Mit einem Kernel gewichteter Mittelwert aller Werte | Berechnung bei der Abfrage | Alle Paare |

<!--en:496a3722-->
| Lineare Attention | $\mathbf{V}^\top\mathbf{K}$, kleinste Quadrate ohne Korrektur für die Schlüssel | Eine Addition pro Paar | $d_v \times d_k$ |

<!--en:bb77d212-->
| Lineare Attention mit Gate | Wie oben, mit höherem Gewicht für jüngere Paare | Eine Addition pro Paar | $d_v \times d_k$ |

<!--en:1e980258-->
| Delta-Regel | Näherungslösung nach der Methode der kleinsten Quadrate | Ein Gradientenschritt pro Paar | $d_v \times d_k$ |

<!--en:c5cb210f-->
| Gated DeltaNet | Näherungslösung nach der Methode der kleinsten Quadrate mit Gewichtszerfall | Ein Gradientenschritt pro Paar | $d_v \times d_k$ |

<!--en:de31b8ab-->
In all diesen Architekturen lernen die Gewichte auf zwei Zeitskalen. Die *schnellen Gewichte* (engl. fast weights) des Zustands $\mathbf{S}_t$ passen sich während des Vorwärtsdurchlaufs an die Schlüssel-Wert-Zuordnungen im aktuellen Kontext an. Die *langsamen Gewichte* (engl. slow weights) des Netzes ($\mathbf{W}_{qkv}, \mathbf{W}_a, \mathbf{W}_\beta, \ldots$) trainieren wir sequenzübergreifend durch Backpropagation. Sie sollen Schlüssel, Werte, Abfragen und Schrittweiten liefern, mit denen die Online-Anpassung gelingt. Selbst mit einer optimalen Schreibregel hat eine $d_v \times d_k$-Matrix jedoch endlichen Rang und kann nur endlich viele Bits speichern. In :numref:`sec_hybrids` messen wir diese feste Grenze und zeigen, wie sie sich anheben lässt, indem wir vereinzelt Schichten mit vollständiger Attention einschieben.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:644547ce-->
Additive Matrixspeicher überlagern alle Schreibvorgänge unter demselben Schlüssel. Fragen wir einen neu belegten Schlüssel ab, erhalten wir daher die Summe aus alten und neuen Werten. Die Delta-Regel liest vor dem Schreiben die aktuelle Vorhersage $\mathbf{S}_{t-1}\mathbf{k}_t$ und addiert nur die Fehlerkorrektur $\beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$. Das entspricht einem Online-Schritt des Gradientenabstiegs für $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$. Der zugehörige Zustandsübergang $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ kontrahiert den Speicher gezielt entlang $\mathbf{k}_t$. Gezieltes Löschen ist dem additiven Schreiben überlegen, wenn Schlüssel neu belegt werden oder korreliert sind. Starkes Löschen ($\beta_t = 1$) kann jedoch frühe Signale abschwächen, wenn viele zufällige Schlüssel überlappen. `GatedDeltaNet` kombiniert eine eingabeabhängige Schreibstärke $\beta_t$ mit einem skalaren Abklingfaktor $a_t$ pro Kopf. Zur parallelen blockweisen Auswertung der Rekursion löst das Verfahren untere Dreieckssysteme (`delta_rule_writes`) und verwendet einen Scan über die Blöcke (`chunked_delta_rule`). Beim assoziativen Abruf liefert es so zuverlässig den aktuellen Wert neu belegter Schlüssel.

<!--en:2294243a-->
## Übungen

<!--en:abdd1c3b-->
1. **Die Reihenfolge spielt keine Rolle.** Zeigen Sie, dass der Inhalt eines additiven Speichers nach einer Folge von Schreibvorgängen unabhängig von deren Reihenfolge ist. Folgern Sie daraus, dass selbst die raffinierteste Leseregel allein aus einem additiven Speicher den jüngsten von $R$ Werten unter demselben Schlüssel nicht ermitteln kann. Berechnen Sie die Trefferquote der besten Regel, wenn die $R$ Werte verschieden sind. Wie ändert sich die gemessene Trefferquote nach acht Schreibvorgängen, wenn das Codebuch von 32 auf 1024 Werte wächst, und warum?
   :exlabel:`ex_dn-order`

<!--en:00fc44cb-->
1. [code] **Abklingen statt Korrektur.** Ergänzen Sie beim additiven Schreiben in `overwrite_recall` einen konstanten Abklingfaktor: $\mathbf{S} \leftarrow a\mathbf{S} +
   \mathbf{v}\mathbf{k}^\top$.
    1. Bestimmen Sie für $R = 2, 4, 8$ den größten Abklingfaktor $a$, bei dem der Abruf in mindestens 95% der Fälle den jüngsten Wert liefert. Warum hängt die Antwort davon ab, wie viele Schreibvorgänge zwischen zwei Belegungen desselben Schlüssels liegen? Warum schadet ein kleinerer Abklingfaktor in diesem Test nie?
    1. Ersetzen Sie nun die orthonormalen Schlüssel durch 16 zufällige Einheitsvektoren der Dimension 64. Schreiben Sie wie zuvor unter 8 dieser Schlüssel $R$-mal und unter den übrigen 8 nur einmal zu Beginn. Messen Sie am Ende die Trefferquote für beide Gruppen. Welche Schreibregel eignet sich für beide?
   :exlabel:`ex_dn-decay`

<!--en:e3c84f91-->
1. **Der implizite Schritt.** Zeigen Sie, dass $\tfrac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|^2 +
   \tfrac{\beta}{2}\|\mathbf{S}\mathbf{k} - \mathbf{v}\|^2$ sein Minimum bei $\mathbf{S}_{t-1} - \frac{\beta}{1 + \beta\|\mathbf{k}\|^2}
   (\mathbf{S}_{t-1}\mathbf{k} - \mathbf{v})\mathbf{k}^\top$ annimmt. (Hinweis: Setzen Sie den Gradienten gleich null und multiplizieren Sie das Ergebnis mit $\mathbf{k}$, um zuerst $\mathbf{S}\mathbf{k}$ zu bestimmen.) Geben Sie die Eigenwerte des Zustandsübergangs an und zeigen Sie, dass der Schritt nur im Grenzfall $\beta \to \infty$ einen exakten Abruf des neuen Paares ermöglicht.
   :exlabel:`ex_dn-implicit`

<!--en:8eea3d19-->
1. [code] **Rekursive Methode der kleinsten Quadrate.** Eine regularisierte Lösung nach der Methode der kleinsten Quadrate nähert sich :eqref:`eq_dn-least-squares` für $\lambda \to 0$. Sie lässt sich mit jedem neuen Paar aktualisieren, wenn der Speicher zusätzlich $\mathbf{P}_t =
   (\mathbf{K}_t^\top \mathbf{K}_t + \lambda\mathbf{I})^{-1}$ mitführt.
    1. Leiten Sie mit der Sherman-Morrison-Formel aus :numref:`subsec_mdl-block-matrices` die Aktualisierungen von $\mathbf{P}_t$ und $\mathbf{S}_t$ für ein neues Paar her. Zeigen Sie, dass die Aktualisierung von $\mathbf{S}_t$ einer Delta-Regel entspricht, die entlang $\mathbf{P}_t\mathbf{k}_t$ statt entlang $\mathbf{k}_t$ schreibt. Wie viele Zahlen umfasst der Zustand nun im Vergleich zu $d_v d_k$ bei der Delta-Regel?
    1. Implementieren Sie das Verfahren und vergleichen Sie es im Versuch zum Überschreiben mit der Delta-Regel. Warum liefert die rekursive Methode der kleinsten Quadrate, die alle Paare bestmöglich annähert, so selten den jüngsten Wert? Wie ließe sich das beheben?
   :exlabel:`ex_dn-rls`

<!--en:78302446-->
1. [code] **Parität.** ● Die Schreibstärke von `GatedDeltaNet` ist durch eine Sigmoidfunktion gegeben. Daher liegt der Eigenwert $1 - \beta_t$ des Zustandsübergangs entlang des Schlüssels in $(0, 1)$. Verwenden wir $2\sigma(\cdot)$ anstelle von $\sigma(\cdot)$, liegt er in $(-1, 1)$.
    1. Auch dann kann sich eine trainierte Schicht $\beta_t = 2$ nur annähern. Setzen Sie in der Konstruktion für die Parität aus :numref:`subsec_dn-state-tracking` bei jedem gezählten Token die Schreibstärke auf $\beta = 2 - \epsilon$. Zeigen Sie, dass nach $n$ gezählten Token der Auslesewert $\big(1 - (\epsilon - 1)^n\big)\mathbf{v}$ beträgt. Wie schnell schrumpft die Differenz zwischen den Auslesewerten nach einer ungeraden und einer geraden Anzahl? Nach wie vielen gezählten Token liegt sie für $\epsilon = 0.01$ unter 1% von $\|\mathbf{v}\|$? Was bedeutet das für das Auslesen mit endlicher Rechengenauigkeit?
    1. In :numref:`subsec_dn-state-tracking` haben wir die Modelle bei der vierfachen Trainingslänge getestet. Trainieren Sie `Gated DeltaNet, 2σ` erneut und testen Sie es an Folgen von 1024 und 4096 Bits. Ab welcher Länge sinkt die Genauigkeit? Passt diese Länge zu Ihrer Antwort auf den ersten Teil?
   :exlabel:`ex_dn-parity`

<!--en:23506d47-->
1. [code] **Andere Wege zur Neubelegung.** Im Versuch zur Neubelegung haben wir drei Bedingungen festgelegt, sodass nur die Schreibregel einen Wert korrigieren konnte. Lockern Sie die Bedingungen einzeln und erklären Sie jedes Ergebnis.
    1. Schalten Sie das Abklingen bei der linearen Attention mit Gate aus. Überschreiben Sie dazu die Methode `decays`, die $\log a_t$ zurückgibt, sodass sie Nullen in derselben Form liefert und stets $a_t = 1$ gilt. Trainieren Sie das Modell erneut. Wie ändert sich die Trefferquote für neu belegte und für einmal belegte Schlüssel?
    1. Schalten Sie das Abklingen wieder ein und lassen Sie jede Neubelegung an einer beliebigen späteren Position zu (`gap=None` in `task`). Schrumpft der Unterschied zwischen den beiden Schichten? Begründen Sie Ihre Antwort anhand des Alters der beiden Werte.
    1. Trainieren Sie beide Schichttypen mit je zwei Blöcken statt einem. Wie können zwei additive Schichten gemeinsam einen Wert korrigieren?
   :exlabel:`ex_dn-rebinding`

<!--en:e74aae10-->
[Diskussionen](https://d2l.discourse.group/t/36784)

<!--en:6f6e76f7-->
Die Delta-Regel und Gated DeltaNet<br>**Fehlerkorrektur durch Lesen vor dem Schreiben · Geometrie der Householder-Transformation · blockweises Lösen von Dreieckssystemen in WY-Form · Speicher als Online-Regression**

<!--en:149202ac-->
::: {.slide title="Warum additive Matrixspeicher bei neu belegten Schlüsseln versagen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:6ba14efa-->
Programme und Erzählungen ändern laufend ihren Zustand (`rate = 0.05` $\to$ `rate = 0.07` oder eine Figur, die von Paris nach Rom reist). Schreiben wir unter einem normierten Schlüssel $\mathbf{k}$ insgesamt $R$-mal die Werte $\mathbf{v}^{(1)}, \ldots, \mathbf{v}^{(R)}$, liefert ein additiver Speicher deren ungewichtete Überlagerung:

<!--en:cb551021-->
Wir testen $8$ orthonormale Schlüssel in $\mathbb{R}^{64}$ (keine gegenseitigen Störbeiträge) und schreiben unter jedem Schlüssel $R \in \{1, 2, 4, 8\}$-mal:

<!--en:aa850fde-->
Selbst bei $8\times$ so vielen Dimensionen wie Schlüsseln fällt die additive Trefferquote für den jüngsten Wert auf Münzwurfniveau ($0.51$) bei $R = 2$ und auf $0.21$ bei $R = 8$. Globales Abklingen mit $a_t < 1$ schwächt alte Werte nur dadurch ab, dass *alle* Schlüssel gleichermaßen altern.

<!--en:046c7199-->
::: {.slide title="Die Delta-Regel: ein Online-Gradientenschritt für den Abrufverlust"}

<!--en:f42cfda6-->
[Vor dem Schreiben lesen]{.kicker}

<!--en:6c14dae3-->
- **Vor dem Schreiben lesen:** Erst die Vorhersage $\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}\mathbf{k}_t$ lesen, dann das skalierte Residuum $\beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$ schreiben.

<!--en:6002e646-->
- **Exakter Ersatz bei $\beta_t = 1$:** Für $\|\mathbf{k}_t\| = 1$ gilt nach einem Schritt $\mathbf{S}_t \mathbf{k}_t = \mathbf{v}_t$.

<!--en:e1d289da-->
::: {.slide title="Projektionsgeometrie der Übergangsmatrix der Delta-Regel"}

<!--en:7e4ea30e-->
[Richtungsabhängiges Löschen]{.kicker}

<!--en:30cb410d-->
Klammern wir $\mathbf{S}_{t-1}$ bei der Aktualisierung nach der Delta-Regel aus, erhalten wir die Übergangsmatrix, mit der wir von rechts multiplizieren:

<!--en:12dd69bc-->
Für einen normierten Schlüssel $\|\mathbf{k}_t\| = 1$ hat $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ den Eigenwert **$1$** auf dem $(d_k - 1)$-dimensionalen Unterraum $\mathbf{k}_t^\perp$ und den Eigenwert **$1 - \beta_t$** entlang $\mathbf{k}_t$:

<!--en:1e237e32-->
- **$\beta_t = 0$:** Identität $\mathbf{I}$ (der Speicher bleibt unverändert).

<!--en:c73bdf92-->
- **$\beta_t \in (0, 1)$:** Teilweise Kontraktion entlang $\mathbf{k}_t$.

<!--en:c91fd284-->
- **$\beta_t = 1$:** Orthogonale Projektion $\mathbf{I} - \mathbf{k}_t\mathbf{k}_t^\top$ (löscht die Komponente entlang $\mathbf{k}_t$ vollständig, bevor wir $\mathbf{v}_t \mathbf{k}_t^\top$ schreiben).

<!--en:37a8eb53-->
- **$\beta_t = 2$:** Householder-Spiegelung $\mathbf{I} - 2\mathbf{k}_t\mathbf{k}_t^\top$ (Eigenwert $-1$, Vorzeichenwechsel entlang $\mathbf{k}_t$).

<!--en:c8f2b5b0-->
::: {.slide title="Stabilität durch normierte Schlüssel und Parität durch Householder-Spiegelungen"}

<!--en:add1fec0-->
[Spektrale Eigenschaften]{.kicker}

<!--en:560229c2-->
- **$\ell_2$-Normierung der Schlüssel:** Entlang $\mathbf{k}_t$ bleibt der Eigenwert $\lambda_{\mathbf{k}} = 1 - \beta_t\|\mathbf{k}_t\|^2$ in $[-1, 1]$, wenn für $\beta_t \in [0, 2]$ die Normierung $\|\mathbf{k}_t\| = 1$ gilt.

<!--en:c175820e-->
- **Negative Eigenwerte ermöglichen Parität:** Mit positiven Abklingfaktoren $a_t > 0$ lässt sich nicht modulo $2$ zählen. Bei $\beta_t = 2$ schaltet die Householder-Spiegelung ($\lambda_{\mathbf{k}} = -1$) den Auslesewert um: $\mathbf{0} \leftrightarrow 2\mathbf{v}$.

<!--en:37abad06-->
::: {.slide title="Der verborgene Preis der Korrektur: Signalverlust"}

<!--en:c035ecef-->
[Viele zufällige Schlüssel auf engem Raum]{.kicker}

<!--en:047642b0-->
- **Kontraktion anderer Signale:** Jeder spätere zufällige Schlüssel mit Norm eins $\mathbf{k}_s$ hat im Mittel die quadrierte Überlappung $1/d_k$ mit $\mathbf{k}_i$. Dadurch schrumpft $\mathbf{v}_i$ pro Schritt um den Faktor $1 - 1/d_k$.

<!--en:3068966a-->
- **Abklingen bei hoher Schlüsseldichte:** Nach $m = 2d_k$ zufälligen Schreibvorgängen mit $\beta = 1$ verbleibt von frühen Signalen der Anteil $e^{-2} \approx 13.5\%$.

<!--en:d926a87d-->
::: {.slide title="Vier Schreibregeln im Vergleich für drei Fälle" layout="code"}

<!--en:b8ef1134-->
[Additives Schreiben, Delta-Regel und kleinste Quadrate]{.kicker}

<!--en:c4fb8137-->
Wir vergleichen `additive`, `delta, β = 1`, `delta, β = 0.5` und das Offline-Verfahren `least squares` ($\mathbf{S}^\star = \mathbf{V}^\top \mathbf{K}(\mathbf{K}^\top \mathbf{K})^+$, Anpassung an alle Paare gleichzeitig) bei der Breite $d = 64$:

<!--en:f14e0e5a-->
::: {.slide title="Wann gezieltes Löschen hilft und warum das Netz βt lernen muss"}

<!--en:2cc16424-->
[Auswertung der Tabelle zu den Schreibregeln]{.kicker}

<!--en:ab32d595-->
Der Benchmark zeigt drei Fälle, die für ein lernbares, eingabeabhängiges Gate $\beta_t \in (0, 1)$ sprechen:

<!--en:ff529376-->
1. **Neubelegung (`8 orthonormal keys, 4 writes`):** `delta` ($\beta = 1$ oder $0.5$) erreicht für den jüngsten Wert eine **Trefferquote von $1.00$**. Dagegen gewichten `additive` und die ungewichtete Methode `least squares` alle $4$ Werte pro Schlüssel gleich ($0.28$).

<!--en:de46967a-->
2. **Korrelierte Schlüssel (`64 keys, ρ = 0.5`):** `additive` summiert jeden Wert auf der gemeinsamen Schlüsselrichtung auf (Fehler $5.82$, Trefferquote $0.54$). Dagegen zieht `delta` die Vorhersage in dieser Richtung vor dem Schreiben ab (Trefferquote **$0.89$** bei $\beta = 0.5$).

<!--en:ba406d64-->
3. **Viele unabhängige Schlüssel auf engem Raum (`128 keys`, `256 keys`):** Bei $n = 128$ hat `delta, β = 1` einen *kleineren* quadratischen Fehler als `additive` ($1.06$ gegenüber $1.99$), aber eine *niedrigere* Trefferquote ($0.53$ gegenüber $0.96$). Schrumpfen alte Auslesewerte gegen $\mathbf{0}$, begrenzt das den quadratischen Fehler auf etwa $\|\mathbf{0} - \mathbf{v}_j\|^2 = 1$ und schwächt frühe Signale ab. Mit der halben Schrittweite $\beta = 0.5$ erreichen wir wieder eine **Trefferquote von $0.82$** bei einem **quadratischen Fehler von $0.67$**.

<!--en:2141b3e2-->
::: {.slide title="Gated DeltaNet: globales Abklingen und gezieltes Überschreiben"}

<!--en:3792beea-->
[Die Rekursion von Gated DeltaNet]{.kicker}

<!--en:954507c4-->
- **Drei Betriebsarten (Yang et al., 2024):**
  1. **Globales Löschen ($a_t \to 0$):** Leert die gesamte Matrix bei einem Themenwechsel.
  2. **Gezielte Neubelegung ($a_t \approx 1, \beta_t \to 1$):** Überschreibt den Wert unter $\mathbf{k}_t$ und erhält $\mathbf{k}_t^\perp$.
  3. **Behutsames Sammeln ($a_t \approx 1, \beta_t \approx 0.2$):** Assoziatives Schreiben mit geringem Signalverlust.

<!--en:fdc094b9-->
::: {.slide title="Blockweise Auswertung mit unteren Dreieckssystemen (WY-Form)"}

<!--en:39be9e78-->
[Dreieckssystem innerhalb eines Blocks]{.kicker}

<!--en:5ced0d7e-->
- **WY-Darstellung:** Schreibt die Delta-Rekursion als $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{u}_t \mathbf{k}_t^\top$ mit den korrigierten Werten $\mathbf{u}_t$.

<!--en:55dac04e-->
- **Paralleles Lösen von $m \times m$-Dreieckssystemen:** Wir lösen $\mathbf{M}\mathbf{W} = \boldsymbol{\beta} \odot \mathbf{V}$ und $\mathbf{M}\mathbf{Z} = (\boldsymbol{\beta} \odot \boldsymbol{\gamma}) \odot \mathbf{K}$ für alle Blöcke parallel und erhalten $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$.

<!--en:fbc50197-->
::: {.slide title="Matrix-Scan über die Blöcke und numerische Überprüfung"}

<!--en:eceb3ce9-->
[Überprüfung des blockweisen Algorithmus]{.kicker}

<!--en:b8891f38-->
Setzen wir $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ in den Zustand am Blockende ein, erhalten wir eine affine Matrixrekursion über die $T/m$ Blöcke:

<!--en:7941f9e5-->
- **Gleicher Rechenaufwand:** Die Verkettung von $\mathbf{A}^{(c)}$ über $T/m$ Blöcke kostet $(T/m)d_k^3$ FLOPs. Das entspricht dem Aufwand $O(T d_k^2)$ für die Matrixmultiplikationen (GEMMs) innerhalb der Blöcke, wenn für die Blockgröße $m \approx d_k$ gilt.

<!--en:a0132ae6-->
::: {.slide title="GatedDeltaNet als direkt einsetzbare GLA-Unterklasse"}

<!--en:597bc086-->
[Objektorientierter Entwurf]{.kicker}

<!--en:9c8d1db4-->
Da `GatedLinearAttention` (§{sec_matrix-state}) die Zustandsrekursion in `memory(X, Q, K, V, log_a)` kapselt, ergänzt `GatedDeltaNet` nur eine lineare Projektion `W_beta` ($d \to h$) und überschreibt `memory`:

<!--en:c98a55d1-->
Alle übrigen Komponenten erbt die Klasse unverändert: kausale Faltung der Breite 4, $\ell_2$-normierte Abfragen und Schlüssel, logarithmisch gleichverteilte Initialisierung der Abklingfaktoren der Köpfe, RMSNorm pro Kopf und SiLU-Ausgabegatter.

<!--en:383f1923-->
::: {.slide title="Versuch zur Neubelegung: Überschreiben in kurzem Abstand"}

<!--en:edf40355-->
[Ergebnisse zur Neubelegung]{.kicker}

<!--en:c17b2812-->
Wir vergleichen 1 Block `GatedLinearAttention` mit 1 Block `GatedDeltaNet` ($d = 64$, $4$ Köpfe der Dimension $16 \times 16$). Die Aufgabe ist der assoziative Abruf für $16$ Paare. Dabei erhalten $8$ der $16$ Schlüssel bereits **zwei Schreibvorgänge nach der ersten Belegung (`gap=2`)** einen neuen Wert:

<!--en:81718780-->
- **Warum `gap=2` die Wirkung der Delta-Regel isoliert:** Erfolgt das Überschreiben schon $2$ Paare nach der ersten Belegung, sind beide Belegungen fast gleich alt. Globales Abklingen mit $a_t$ kann den alten Wert kaum abschwächen, ohne zugleich zuvor einmal belegte Schlüssel zu löschen.

<!--en:f18a6bdc-->
- **Zuverlässiges Überschreiben:** Die zusätzlichen $260$ Parameter der Projektion `W_beta` erhöhen die Genauigkeit für neu belegte Schlüssel in den Läufen auf **$0.96$ oder mehr**. Das übertrifft `GatedLinearAttention` (etwa $0.9$) und liegt nur etwa $2$ Prozentpunkte unter der Genauigkeit für einmal belegte Schlüssel.

<!--en:667d2e70-->
::: {.slide title="Warum wir 1 Block testen: Tiefe kann einen Delta-Schritt annähern"}

<!--en:ed9152e6-->
[Zusammenspiel mehrerer Schichten]{.kicker}

<!--en:20632a06-->
- **Zwei additive Schichten bilden einen Delta-Schritt nach:** Block 1 (`GLA`) kann $\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}^{(1)}\mathbf{k}_t$ abrufen und in den Residualstrom einspeisen, sodass Block 2 (`GLA`) den Term $(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$ schreibt.

<!--en:4cb86c2b-->
- **Vorteil der Delta-Regel innerhalb der Schicht:** `GatedDeltaNet` liest bei beliebig vielen Neubelegungen $R$ in *jeder einzelnen Schicht* vor dem Schreiben. Tiefere Schichten stehen damit für Schlussfolgerungen zur Verfügung.

<!--en:ac5f68bf-->
::: {.slide title="Der implizite proximale Delta-Schritt (Longhorn)"}

<!--en:63b3626e-->
[Unbedingte Stabilität ohne Normierung der Schlüssel]{.kicker}

<!--en:a7ae83c3-->
- **Implizite proximale Zielfunktion (Liu et al., 2024):** $\frac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|_F^2 + \frac{\beta_t}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ nimmt sein Minimum beim *neuen* Zustand $\mathbf{S}_t$ an.

<!--en:6d44d6ad-->
- **Unbedingte Stabilität:** Der Eigenwert in Schlüsselrichtung beträgt $\frac{1}{1 + \beta_t\|\mathbf{k}_t\|^2} \in (0, 1)$ für beliebige $\|\mathbf{k}_t\|$ und $\beta_t > 0$.

<!--en:56b68b56-->
::: {.slide title="Sequenzmischer als Verfahren zur Online-Regression"}

<!--en:f8ac5a25-->
[Regression zur Testzeit]{.kicker}

<!--en:a7c172a4-->
- **Schnelle Gewichte ($\mathbf{S}_t$):** Lösen im Vorwärtsdurchlauf eine lineare Regression $\mathbf{S}_t \mathbf{k}_s \approx \mathbf{v}_s$ im Kontext der Token $1\ldots t$.

<!--en:b0062406-->
- **Langsame Gewichte ($\theta$):** AdamW trainiert sie in der äußeren Schleife, sodass sie $(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t, a_t, \beta_t)$ formen.

<!--en:337a5401-->
::: {.slide title="Sequenzspeicher nach Regressionsverfahren"}

<!--en:de5cfb1c-->
[Regressionsverfahren im Überblick]{.kicker}

<!--en:c760839b-->
| Sequenzspeicher | Regressionsziel oder Schätzer im Kontext | Online-Aktualisierung pro Tokenpaar $(\mathbf{k}_t, \mathbf{v}_t)$ | Mitgeführter Zustand |

<!--en:f2a5d482-->
| **Lineare Attention** | Unnormiertes Kreuzmoment $\mathbf{V}_t^\top \mathbf{K}_t$ (ohne $(\mathbf{K}_t^\top\mathbf{K}_t)^{-1}$) | $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | $d_v \times d_k$ |

<!--en:b7ec41c0-->
| **Lineare Attention mit Gate** | Exponentiell gewichtetes Kreuzmoment | $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | $d_v \times d_k$ |

<!--en:0881a85e-->
| **DeltaNet** | Online-SGD für $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ | $\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$ | $d_v \times d_k$ |

<!--en:df73f627-->
| **Gated DeltaNet** | Online-SGD mit Gewichtszerfall $a_t \in (0, 1)$ | $\mathbf{S}_t = a_t\mathbf{S}_{t-1}(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top) + \beta_t\mathbf{v}_t\mathbf{k}_t^\top$ | $d_v \times d_k$ |

<!--en:392b9550-->
| **Longhorn (impliziter Schritt)** | Proximaler Schritt $\min_{\mathbf{S}} \frac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|^2 + \frac{\beta_t}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ | Delta-Regel mit normierter Schrittweite $\frac{\beta_t}{1 + \beta_t\|\mathbf{k}_t\|^2}$ | $d_v \times d_k$ |

<!--en:5b253289-->
| **Rekursive kleinste Quadrate** | Exakte Ridge-Regression $\mathbf{V}_t^\top\mathbf{K}_t(\mathbf{K}_t^\top\mathbf{K}_t + \lambda\mathbf{I})^{-1}$ | Aktualisierung nach Sherman und Morrison entlang $\mathbf{P}_t\mathbf{k}_t$ | $d_v d_k + d_k^2$ |

<!--en:e3054842-->
| **Softmax-Attention** | Nichtparametrische Nadaraya-Watson Kernel-Regression | Hängt $(\mathbf{k}_t, \mathbf{v}_t)$ an den Cache an und wertet ihn bei der Abfrage aus | $t \times (d_k + d_v)$ |

<!--en:d9a85aa2-->
::: {.slide title="Verständnisfragen zu Delta-Übergängen und blockweisem Lösen"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:b1fef131-->
Prüfen Sie Ihr Verständnis der Delta-Regel anhand dieser drei Fragen:

<!--en:88f4dcb5-->
1. Angenommen, es gilt $\mathbf{S}_{t-1}\mathbf{k}_t = \hat{\mathbf{v}}_t$. Wir wenden die Delta-Regel mit einem normierten Schlüssel $\|\mathbf{k}_t\| = 1$ und der Schreibstärke $\beta_t = 0.5$ an. Was liefert eine sofortige Abfrage $\mathbf{S}_t \mathbf{k}_t$, und was geschieht mit $\mathbf{S}_t \mathbf{q}$ für eine beliebige Abfrage $\mathbf{q} \perp \mathbf{k}_t$?

<!--en:a8e0db03-->
2. Warum ersetzt der implizite proximale Schritt (Longhorn) $\beta_t$ durch $\tilde{\beta}_t = \beta_t / (1 + \beta_t\|\mathbf{k}_t\|^2)$, und warum kann ein impliziter Schritt die Parität nicht exakt berechnen?

<!--en:7e6824e2-->
3. Warum ist in `chunked_delta_rule` die Systemmatrix $\mathbf{M} = \mathbf{I} + \textrm{diag}(\boldsymbol{\beta})\,\textrm{tril}_{-1}(\mathbf{L} \odot \mathbf{K}\mathbf{K}^\top)$ innerhalb eines Blocks eine untere Dreiecksmatrix mit Einsen auf der Diagonale?

<!--en:557b98b6-->
**Antworten:**

<!--en:cf367c97-->
1. $\mathbf{S}_t \mathbf{k}_t = (1 - 0.5)\hat{\mathbf{v}}_t + 0.5\mathbf{v}_t = \mathbf{\tfrac{1}{2}(\hat{\mathbf{v}}_t + \mathbf{v}_t)}$ (der Mittelwert beider Werte). Für $\mathbf{q} \perp \mathbf{k}_t$ gilt $\mathbf{k}_t^\top \mathbf{q} = 0$ und damit $\mathbf{S}_t \mathbf{q} = \mathbf{\mathbf{S}_{t-1}\mathbf{q}}$ (**völlig unverändert**).

<!--en:4117543a-->
2. Beim Lösen von $\mathbf{S} - \mathbf{S}_{t-1} + \beta_t(\mathbf{S}\mathbf{k}_t - \mathbf{v}_t)\mathbf{k}_t^\top = \mathbf{0}$ verwenden wir die Vorhersage $\mathbf{S}\mathbf{k}_t$ *nach der Aktualisierung*. Daraus ergibt sich in Schlüsselrichtung der Eigenwert $1 - \tilde{\beta}_t\|\mathbf{k}_t\|^2 = \frac{1}{1 + \beta_t\|\mathbf{k}_t\|^2} \in (0, 1)$ für alle $\beta_t > 0$. Das verhindert auch ohne Normierung der Schlüssel ein Überschwingen, schließt aber negative Eigenwerte ($\lambda = -1$) aus, wie sie die Spiegelung für die Parität erfordert.

<!--en:a5193fad-->
3. Wegen der Kausalität hängt der korrigierte Wert $\mathbf{u}_t$ im Schritt $t$ nur von früheren Schreibvorgängen $s < t$ im selben Block ab. ($\textrm{tril}_{-1}$ hat auf und oberhalb der Diagonale Nullen, daher hat $\mathbf{M}$ auf der Diagonale jeweils den Eintrag $1$.)

<!--en:8c964813-->
::: {.slide title="Zusammenfassung: die Delta-Regel und Gated DeltaNet"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:ed4ad812-->
| Konzept oder Komponente | Mathematische Form | Wirkung und Grenzen |

<!--en:dc754dcf-->
| **Die Delta-Regel** | $\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$ | Online-Gradientenschritt für $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$, liest vor dem Schreiben |

<!--en:41170557-->
| **Geometrie des Übergangs** | $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ ($\|\mathbf{k}_t\| = 1$) | Eigenwert $1$ auf $\mathbf{k}_t^\perp$ und $1 - \beta_t$ entlang $\mathbf{k}_t$ (Projektion bei $\beta=1$, Spiegelung bei $\beta=2$) |

<!--en:7ec0b203-->
| **Signalverlust** | $(1 - 1/d_k)^m \approx e^{-m/d_k}$ nach $m$ zufälligen Schreibvorgängen | Volle Schreibstärke $\beta_t = 1$ schwächt frühe, überlappende Schlüssel. Ein gelerntes $\beta_t \in (0, 1)$ wägt Überschreiben und Signalerhalt ab. |

<!--en:7b244eb5-->
| **Gated DeltaNet** | $\mathbf{S}_t = a_t\mathbf{S}_{t-1}(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top) + \beta_t\mathbf{v}_t\mathbf{k}_t^\top$ | Kombiniert globales Abklingen pro Kopf $a_t$ mit gezieltem Überschreiben $\beta_t$ (Trefferquote $0.96+$ bei Neubelegung) |

<!--en:7bb2223d-->
| **Blockweiser WY-Algorithmus** | Dreieckssysteme für $\mathbf{M}\mathbf{W}, \mathbf{M}\mathbf{Z}$ und Scan über die Blöcke | Wertet $d_k \times d_k$-Übergangsmatrizen mit Aufwand $O(T d_k^2)$ durch parallele Matrixmultiplikationen auf Tensor Cores aus |
