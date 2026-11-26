<!--en:6a1d8dc7-->
# Rekurrente Netze mit Gattern

<!--en:9be19792-->
Das einfache rekurrente Netz aus :numref:`sec_rnn-scratch` erzeugt erkennbare Wörter und Satzzeichen. Über mehr als ein paar Wörter hinweg gerät seine Grammatik jedoch durcheinander. Welches Verb folgt zum Beispiel auf *„the keys to the cabinet in the hall“*? Um *„are“* statt *„is“* zu wählen, muss sich das Netz daran erinnern, dass *„keys“* 30 Zeichen zuvor im Plural steht. Ein einfaches RNN überschreibt bei jedem Schritt seinen gesamten Zustand mit derselben rekurrenten Matrix und derselben $\tanh$-Nichtlinearität. Dadurch klingen sowohl das Signal im Vorwärtsdurchlauf als auch der Gradient im Rückwärtsdurchlauf über einige Dutzend Schritte rasch ab. Wie kann ein rekurrentes Netz das Gelesene über Hunderte von Schritten behalten?

<!--en:627e119b-->
Die Lösung aus den späten 1990er-Jahren lässt das Netz für jede Einheit bei jedem Schritt entscheiden, welchen Anteil ihres Zustands sie behält. Dazu dient ein *Gatter* (engl. gate), ein Vektor mit Werten in $(0, 1)$, den das Netz aus der aktuellen Eingabe und dem vorherigen Zustand berechnet. Bleibt das Gatter einer Einheit nahe bei $1$, gibt sie sowohl den gespeicherten Inhalt als auch den rückwärts fließenden Gradienten praktisch unverändert über diesen Schritt weiter. Wir bauen die beiden klassischen Architekturen mit Gattern auf, das Long Short-Term Memory (LSTM) und die Gated Recurrent Unit (GRU). Wir testen sie an einer synthetischen Aufgabe zum verzögerten Abruf (engl. delayed recall) und am WikiText-103-Benchmark auf Byte-Ebene aus :numref:`chap_transformers`. Mit Gattern kann das Netz Informationen langfristig behalten. Da die Gatter aber vom vorherigen Zustand abhängen, muss das Training weiterhin Schritt für Schritt ablaufen. :numref:`sec_linear-recurrence` zeigt, wie sich dieser Flaschenhals beseitigen lässt.

<!--en:7ccd7d65-->
In diesem Kapitel schreiben wir die Gleichungen für eine einzelne Sequenz mit Spaltenvektoren. Der Code verarbeitet dagegen Minibatches von Sequenzen mit Zeilenvektoren.

<!--en:b19401fc-->
## Gatter

<!--en:609da68f-->
### Verschwindende Gradienten

<!--en:3c9e608e-->
Erinnern wir uns an :numref:`subsec_bptt_analysis`: Wie breiten sich Gradienten in einem einfachen RNN mit verstecktem Zustand (engl. hidden state) $\mathbf{h}_t \in \mathbb{R}^d$ aus?

<!--en:eb3d6908-->
Eine Störung von $\mathbf{h}_{t-k}$ erreicht $\mathbf{h}_t$ multipliziert mit dem Produkt der $k$ Jacobi-Matrizen $\mathbf{J}_s = \textrm{diag}(1 - \mathbf{h}_s^2)\,\mathbf{W}_\textrm{hh}$ der einzelnen Schritte:

<!--en:4eb7b466-->
Für eine einzelne skalare Einheit lautet jeder Faktor $w(1 - h_s^2)$ und hat höchstens den Betrag $|w|$. Für $|w| < 1$ schrumpft das Produkt mindestens so schnell wie $|w|^k$ (bei $w = 0.9$ auf etwa $4\%$ nach 30 Schritten und auf $\le 0.003\%$ nach 100 Schritten).

<!--en:71d5728c-->
Ist bei einer Matrix $\mathbf{W}_\textrm{hh}$ der größte Singulärwert kleiner als $1$, so verkürzt jeder Faktor jeden von null verschiedenen Vektor und das Produkt muss gegen null gehen. Über viele Schritte skalieren Matrixpotenzen wie $\rho^k$. Dabei erfüllt der *Spektralradius* $\rho$ (der größte Betrag eines Eigenwerts) die Ungleichung $\rho \le \|\mathbf{W}_\textrm{hh}\|_2$. Gilt $\rho \approx 1$ und bleibt der Zustand so klein, dass $\tanh$ nahezu linear ist, können Komponenten entlang der führenden Eigenvektoren viele Schritte überdauern. Diesen Grenzfall beobachten wir später in unseren Experimenten. Abseits des Grenzfalls $\rho \approx 1$ explodieren die Gradienten entweder oder sie verschwinden. Explodierende Gradienten lassen sich durch Gradienten-Clipping begrenzen. Verschwundene Gradienten kann keine nachträgliche Skalierung wiederherstellen.

<!--en:f1168068-->
### Additives Gedächtnis

<!--en:a28f9280-->
Die einfachste Rekursion mit Ableitung eins ist die laufende Summe $c_t = c_{t-1} + x_t$. Für sie gilt $\partial c_t / \partial c_{t-k} = 1$ bei jedem zeitlichen Abstand $k$. Eine ungewichtete Summe vergisst allerdings nie: Alle bisherigen Eingaben überlagern sich mit gleichem Gewicht, und die Varianz des Zustands wächst mit der Sequenzlänge.

<!--en:e095d906-->
### Gatter für erlerntes Abklingen

<!--en:a7a5ccc1-->
Damit das Netz relevante Eingaben behält und irrelevante verwirft, wählt es bei jedem Schritt den Anteil $f_t \in (0, 1)$ des alten Inhalts, den es behält, und den Eingabeanteil $i_t \in (0, 1)$, mit dem es den vorgeschlagenen neuen Inhalt $\tilde{c}_t$ multipliziert:

<!--en:d2da23ff-->
Entlang des direkten Speicherpfads ergibt sich die Ableitung über $k$ Schritte als Produkt der jeweils beibehaltenen Anteile: $\partial c_t / \partial c_{t-k} = \prod_{s=t-k+1}^t f_s$. Bei konstantem Anteil $f$ klingt ein gespeicherter Wert wie $f^k$ ab. Seine *Halbwertszeit* (engl. half-life) beträgt $\ln 2 / \ln(1/f)$ Schritte, also rund 7 Schritte bei $f = 0.9$, 69 bei $f = 0.99$ und 693 bei $f = 0.999$. Für $f_t \to 1$ bleiben der Wert und sein Gradient beliebig lange erhalten. Für $f_t \to 0$ leert sich die Zelle in einem einzigen Schritt.

<!--en:54b03ac8-->
In einer Schicht der Breite $d$ wird das Gedächtnis zu einem Vektor $\mathbf{c}_t \in \mathbb{R}^d$. Hinzu kommt ein nach außen sichtbarer versteckter Zustand $\mathbf{h}_t \in \mathbb{R}^d$. Jedes *Gatter* berechnet einen Vektor in $(0, 1)^d$ mit einer Schicht, die die Sigmoidfunktion verwendet. Sie verarbeitet die aktuelle Eingabe $\mathbf{x}_t$ und die vorherige Ausgabe $\mathbf{h}_{t-1}$:

<!--en:06da2577-->
Ein einfaches RNN multipliziert bei jedem Schritt den gesamten Zustand mit einer dicht besetzten Matrix $\mathbf{W}_\textrm{hh}$ und wendet $\tanh$ an. Der Speicher mit Gattern aktualisiert dagegen jede Koordinate durch elementweise Multiplikation mit ihrem eigenen erlernten Faktor $f_{t,j} \in (0, 1)$. Wegen $f_{t,j} \le 1$ können die Gradienten entlang des direkten Speicherpfads nicht explodieren. Bleibt $f_{t,j}$ nahe bei $1$, verschwinden sie auch nicht.

<!--en:dba65260-->
Für die Praxis sind noch zwei Punkte wichtig. Erstens kann bei $f_t \approx 1$ der gespeicherte Wert $c_t$ durch wiederholtes Schreiben weiter wachsen. Daher wendet das LSTM die Funktion $\tanh$ an, wenn es $\mathbf{c}_t$ nach $\mathbf{h}_t$ ausliest. Zweitens starten Gatter mit Sigmoidfunktion nahe bei $\sigma(0) = 1/2$, wenn wir die Bias-Terme mit null initialisieren. Das entspricht einer Halbwertszeit von nur einem Schritt. Ein Netz mit Gattern kann also Informationen lange behalten, hat diese Eigenschaft aber in der Standardeinstellung noch nicht zu Beginn.

<!--en:0deed749-->
## Long Short-Term Memory

<!--en:bb7e7fd6-->
### Die Speicherzelle

<!--en:ad235184-->
Das *Long Short-Term Memory* (LSTM) von :citet:`Hochreiter.Schmidhuber.1997`, ergänzt um das Vergessensgatter (engl. forget gate) von :citet:`Gers.Schmidhuber.Cummins.2000`, baut auf :eqref:`eq_lstm-gated-sum` eine rekurrente Schicht auf. Sie führt zwei Vektoren der Breite $d$ mit. Die interne *Speicherzelle* (engl. memory cell) $\mathbf{c}_t$ ändert sich ausschließlich durch elementweise Gewichtung mit Gattern und Addition. Der nach außen sichtbare *versteckte Zustand* $\mathbf{h}_t$ gibt den nachfolgenden Schichten und den Gattern des nächsten Schritts eine durch ein Gatter gewichtete, im Wertebereich begrenzte Darstellung von $\mathbf{c}_t$.

<!--en:f0a5121d-->
Drei Gatter mit Sigmoidfunktion steuern den Zugriff auf die Zelle: das *Eingabegatter* (engl. input gate) $\mathbf{i}_t$, das *Vergessensgatter* $\mathbf{f}_t$ und das *Ausgabegatter* (engl. output gate) $\mathbf{o}_t$:

<!--en:999ac456-->
Eine vierte Projektion mit $\tanh$-Aktivierung liefert am *Eingabeknoten* (engl. input node) den Kandidaten $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_\textrm{xc} \mathbf{x}_t + \mathbf{W}_\textrm{hc} \mathbf{h}_{t-1} + \mathbf{b}_\textrm{c})$. Zelle und versteckter Zustand aktualisieren sich dann mithilfe elementweiser Multiplikation ($\odot$):

<!--en:3702a3c0-->
Die Trennung von $\mathbf{c}_t$ und $\mathbf{h}_t$ erlaubt es einer Einheit, Informationen bis zum Abruf intern zu behalten. Gilt zum Beispiel $c_{t-1} = 0.8$ mit $f_t = 0.95$, $i_t = 0.1$, $\tilde{c}_t = -0.5$ und $o_t = 0.2$, so hält die Zelle den Wert $c_t = 0.71$, gibt aber nur $h_t = 0.2\tanh(0.71) = 0.12$ aus (:numref:`fig_lstm_cell`).

<!--en:625da204-->
![Die LSTM-Speicherzelle. Die vorherige Ausgabe $\mathbf{h}_{t-1}$ und die Eingabe $\mathbf{x}_t$ speisen vier Schichten: drei Gatter mit Sigmoidfunktion und einen Eingabeknoten mit $\tanh$. Die Zelle $\mathbf{c}_t$ aktualisiert ihren Inhalt additiv, wobei das Vergessensgatter den alten Inhalt skaliert. Das Ausgabegatter steuert, was der Rest des Netzes sieht.](../img/mdl-modernrnn-lstm-cell.svg)

<!--en:4d172920-->
Halten wir die Gatter fest, ergibt sich für die direkte Jacobi-Matrix von Zelle zu Zelle $\partial \mathbf{c}_t / \partial \mathbf{c}_{t-1}$ der Ausdruck $\textrm{diag}(\mathbf{f}_t)$ (oberer Pfad in :numref:`fig_lstm_cell`). Die vollständige Ableitung enthält auch indirekte Terme über $\mathbf{h}_{t-1}$ und die Gatter. Der direkte Zellpfad leitet den Gradienten praktisch ungeschwächt weiter, solange $\mathbf{f}_t$ nahe bei $1$ bleibt.

<!--en:869d223f-->
### Implementierung

<!--en:4ca59599-->
Da $\mathbf{i}_t$, $\mathbf{f}_t$, $\tilde{\mathbf{c}}_t$ und $\mathbf{o}_t$ alle dieselben Eingaben $(\mathbf{x}_t, \mathbf{h}_{t-1})$ verwenden, fassen wir ihre Gewichte in zwei Matrizen $\mathbf{W}_\textrm{x} \in \mathbb{R}^{d_\textrm{in} \times 4d}$ und $\mathbf{W}_\textrm{h} \in \mathbb{R}^{d \times 4d}$ zusammen. So genügt pro Eingabe eine einzige Matrizenmultiplikation. Das Ergebnis teilen wir in vier Blöcke.

<!--en:a75d2644-->
Um zu prüfen, ob `LSTMScratch` genau dieselbe Rekursion berechnet wie die LSTM-Schicht des Frameworks, kopieren wir die Gewichte der Bibliothek in unsere Klasse. Anschließend vergleichen wir die Ausgaben für einen zufälligen Batch.

<!--en:3ac15281-->
Die Bibliothek verwendet die fusionierte Schicht `nn.LSTM`.

<!--en:3797f97c-->
Die Bibliothek verwendet die Schicht `nnx.LSTMCell`.

<!--en:43ceb982-->
Die Ausgaben stimmen bis auf Rundungsfehler im Format float32 ($\sim 10^{-7}$) überein. Daher verwenden wir im Folgenden die schnelleren fusionierten Schichten der Bibliothek.

<!--en:dbaace24-->
## Gated Recurrent Units

<!--en:a636cbdc-->
Braucht eine Schicht mit Gattern tatsächlich zwei Zustandsvektoren und drei Gatter? Die *Gated Recurrent Unit* (GRU) von :citet:`Cho.Van-Merrienboer.Gulcehre.ea.2014` fasst die Zelle und den versteckten Zustand in einem einzigen Vektor $\mathbf{h}_t$ zusammen. Ihn steuern nur zwei Gatter: ein *Reset-Gatter* $\mathbf{r}_t$ und ein *Update-Gatter* $\mathbf{z}_t$. Dennoch erreicht die GRU bei vielen Sequenzaufgaben die gleiche Güte wie das LSTM :cite:`Chung.Gulcehre.Cho.ea.2014`:

<!--en:dbe59dd3-->
Das Reset-Gatter bestimmt, wie viel von $\mathbf{h}_{t-1}$ in den vorgeschlagenen *Kandidatenzustand* (engl. candidate state) eingeht:

<!--en:2cd76dc6-->
Das Update-Gatter $\mathbf{z}_t$ bildet dann eine Konvexkombination aus dem alten Zustand und dem Kandidaten:

<!--en:db300ba9-->
:eqref:`gru_H` entspricht der Summe mit Gattern aus :eqref:`eq_lstm-gated-sum`, wenn Vergessens- und Eingabegatter gekoppelt sind ($f_t = z_t, i_t = 1 - z_t$). Da jede Koordinate eine Konvexkombination zweier Werte in $[-1, 1]$ ist, bleibt $\mathbf{h}_t$ auf $[-1, 1]$ beschränkt, ganz ohne zusätzliche $\tanh$-Funktion an der Ausgabe (:numref:`fig_gru_cell`). Eine GRU-Schicht benötigt insgesamt $6d^2$ rekurrente Gewichte und Eingabegewichte, eine LSTM-Schicht dagegen $8d^2$. (Fusionierte GRU-Schichten in Bibliotheken wenden $\mathbf{r}_t$ erst *nach* der Multiplikation mit $\mathbf{W}_\textrm{hh}$ an. So lassen sich alle rekurrenten Matrixprodukte in einem Aufruf berechnen, siehe die Übungen.)

<!--en:fc2f9dbe-->
![Die GRU-Zelle. Das Reset-Gatter filtert den Beitrag des vorherigen Zustands zum Kandidaten $\tilde{\mathbf{h}}_t$. Das Update-Gatter mischt den alten Zustand und den Kandidaten.](../img/mdl-modernrnn-gru-cell.svg)

<!--en:7e09b2af-->
## Training von Netzen mit Gattern

<!--en:ed16f781-->
### Verzögerter Abruf

<!--en:3854aee8-->
Die Gleichungen mit Gattern *erlauben* es, Informationen über Hunderte von Schritten zu speichern. Findet der Gradientenabstieg solche Lösungen tatsächlich? Wir testen das Speichervermögen direkt an einer synthetischen Aufgabe zum *verzögerten Abruf*. Eine Sequenz beginnt mit einem von 8 Zielsymbolen (0–7). Es folgen $T - 1$ zufällige Fülltoken aus 8 anderen Symbolen (8–15). Im Schritt $T$ muss das Modell das Anfangssymbol klassifizieren (Genauigkeit bei zufälligem Raten: $1/8 = 0.125$).

<!--en:670e68f9-->
Unser Testmodell (`RecallModel`) bettet Token in 32 Dimensionen ein, verarbeitet sie mit einer einzigen rekurrenten Schicht der Breite 64 und wendet im letzten Schritt einen linearen Klassifikator an. In diesem Kapitel initialisieren wir alle Gewichtsmatrizen in linearen Schichten, Faltungsschichten und rekurrenten Schichten nach LeCun mit einer Normalverteilung (Varianz $1 / n_\textrm{in}$). Die Bias-Terme setzen wir auf null (:numref:`sec_numerical_stability`).

<!--en:e19f0d4a-->
`init_lecun` initialisiert die Gewichte linearer Module, Faltungsmodule und rekurrenter Module nach LeCun mit einer Normalverteilung und setzt die Bias-Terme auf null. `Recurrent` kapselt die rekurrenten Module von PyTorch und gibt nur den Tensor mit der Ausgabesequenz zurück.

<!--en:bd91aa99-->
Flax verwendet für lineare Schichten und rekurrente Eingabeprojektionen standardmäßig die Normalverteilung nach LeCun, für rekurrente Gewichtsmatrizen dagegen orthogonale Matrizen. Wir übergeben `recurrent_kernel_init`, um auch dort die Normalverteilung nach LeCun zu verwenden.

<!--en:e7e1a643-->
Wir vergleichen sechs rekurrente Konfigurationen in drei Paaren:

<!--en:c7260d18-->
1. **Einfache RNNs:** ein RNN mit normalverteilten rekurrenten Gewichten nach LeCun ($\sigma = 1/\sqrt{64} = 1/8$) und als Kontrolle eines mit gleichverteilten Gewichten auf $[-1/8, 1/8]$ (Varianz $1/3$ derjenigen bei der LeCun-Initialisierung).

<!--en:529aca41-->
2. **LSTM und GRU mit Standardinitialisierung:** Alle Gatter haben einen Bias von null und starten daher nahe bei $\sigma(0) = 1/2$ (Halbwertszeit: 1 Schritt).

<!--en:517ea473-->
3. **LSTM und GRU mit auf Erhalt eingestellten Gattern (`init_keep_gate`):** Den Bias des LSTM-Vergessensgatters und den des GRU-Update-Gatters initialisieren wir mit $+5$ ($\sigma(5) = 0.993$, Halbwertszeit $\approx 100$ Schritte). Den Bias des LSTM-Eingabegatters setzen wir auf $-5$ ($\sigma(-5) = 1 - \sigma(5) = 0.007$). Die Kopplung $b_\textrm{i} = -b_\textrm{f}$ begrenzt den Betrag des LSTM-Zellzustands auch bei langen Folgen von Fülltoken auf $\sigma(-5)/(1 - \sigma(5)) = 1$. Dadurch gerät die $\tanh$-Funktion an der Ausgabe nie in Sättigung. Das entspricht der eingebauten Kopplung über $1 - \mathbf{z}_t$ in der GRU (siehe :citet:`Jozefowicz.Zaremba.Sutskever.2015` und die *Chrono-Initialisierung* von :citet:`Tallec.Ollivier.2018`).

<!--en:244c051e-->
Die Gleichverteilung im Kontrollversuch entspricht der Standardinitialisierung von `nn.RNN` in PyTorch.

<!--en:4c82f655-->
Was sagt eine einfache Rechnung für Netze mit Gattern und der üblichen Initialisierung aller Bias-Terme mit null voraus? Startet jedes Gatter bei $1/2$, so schwächt der direkte Pfad zwischen Schritt $1$ und Schritt $T$ das Signal in beiden Richtungen um den Faktor $2^{-(T-1)}$ ab. Das ergibt rund $10^{-6}$ bei $T = 20$ (in float32 noch auflösbar), aber nur $10^{-15}$ bei $T = 50$. Im Zustand bei Schritt $T$ addiert sich der Beitrag des Symbols zu den Beiträgen der letzten Schritte, die von der Größenordnung $O(1)$ sind. Was unterhalb der relativen Genauigkeit von float32 (etwa $\sim 10^{-7}$) liegt, geht schon vor dem Auslesen durch Rundung verloren. Der Gradient, der zur Einbettung des Symbols zurückfließt, ist ebenso klein. Wir erwarten daher, dass LSTM und GRU mit Standardinitialisierung die Aufgabe bei $T = 20$ lösen und spätestens bei $T = 50$ scheitern. Die auf Erhalt initialisierten Varianten sollten dagegen bei allen Verzögerungen bis $T = 200$ erfolgreich sein.

<!--en:3b8b6a72-->
Wir speichern die Hilfsfunktionen für das Training `train_queries`, `query_accuracy` und `train_delayed_recall`, um sie bei allen synthetischen Abrufaufgaben in diesem Kapitel wiederzuverwenden.

<!--en:2690f6f9-->
Prüfen wir vor dem Training auch den Spektralradius und den größten Singulärwert der beiden unterschiedlich initialisierten $64 \times 64$-Matrizen des einfachen RNN:

<!--en:26989872-->
Die normalverteilte Matrix nach LeCun hat den Spektralradius $1.02$. Sie liegt damit genau im Grenzbereich $\rho \approx 1$, in dem die führenden Eigenmoden bei kleinen Zuständen weder verschwinden noch explodieren. Die gleichverteilte Matrix mit Standardskalierung hat dagegen den Spektralradius $0.62$ und verkleinert Zustände pro Schritt um den Faktor $\sim 0.6$. Nun trainieren wir alle sechs Modelle mit je zwei Zufallsstartwerten für die Verzögerungen $T \in \{20, 50, 100, 200\}$.

<!--en:110da650-->
Die Ergebnisse bestätigen die Analyse für die GRU sowie für alle Verzögerungen $T \ge 50$. Für das LSTM müssen wir die Vorhersage in einem Punkt korrigieren:

<!--en:b5692cd5-->
- **LSTM und GRU mit Standardinitialisierung oder auf Erhalt eingestellten Gattern:** Bei einem Gatter-Bias von null löst die GRU die Aufgabe für $T = 20$. Beide Zellen fallen auf Zufallsniveau ($0.12$), sobald $T \ge 50$ gilt. Das LSTM mit Standardinitialisierung ist allerdings schon bei $T = 20$ unzuverlässig, obwohl die Rechnung einen Erfolg vorhersagte. Diese Abschätzung berücksichtigte nur die Vergessensgatter. Wir haben die Ursache nicht isoliert. Wahrscheinlich liegt es daran, dass das Symbol im LSTM auch durch ein halb offenes Eingabegatter in die Zelle gelangen und sie durch ein halb offenes Ausgabegatter und eine $\tanh$-Funktion wieder verlassen muss. Damit liegt $T = 20$ an der Grenze dessen, was das Training noch erschließen kann. Mit `init_keep_gate` erreichen beide Modelle eine Genauigkeit von $1.00$ in jedem Lauf und bei jeder Verzögerung bis $T = 200$.

<!--en:eb3c656b-->
- **Einfache RNNs im Grenzbereich und im kontrahierenden Bereich:** Das nach LeCun normalverteilte RNN startet mit Spektralradius $\rho \approx 1$. Bei manchen Zufallsstartwerten behält es das Symbol überraschend lange (sogar bis $T = 100$), bei anderen gelingt das nicht. Das RNN mit Spektralradius $\rho \approx 0.62$ und Standardskalierung löst die Aufgabe bis $T = 50$ nur gelegentlich und scheitert ab $T = 100$ bei jedem Zufallsstartwert. Mit passend initialisierten Bias-Termen steuern Gatter den Speicher jeder Einheit zuverlässig. Beim einfachen RNN hängt der Erhalt der Information vom zufälligen Spektrum der Matrix ab.

<!--en:7c259aa7-->
### Netze mit Gattern für Text

<!--en:ca2dfd8c-->
In :numref:`subsec_gpt-rnn-baseline` fiel ein einzelnes breites, einfaches RNN nach den ersten 16 Kontextbytes hinter ein GPT mit mehreren Blöcken zurück. Um die Sequenzmischer fair zu vergleichen, müssen wir den Einfluss der Rekurrenz von dem der Anzahl der Residualblöcke und der positionsweisen MLPs trennen. Dazu setzen wir in diesem Kapitel jeden Mischer in dieselbe Residualarchitektur mit vorgelagerter Normalisierung (`SeqLM`) ein. Ihr Aufbau entspricht :numref:`sec_transformer-block`: Auf eine Tokeneinbettung folgt ein Stapel von Blöcken. Jeder Block berechnet zunächst `H = H + mixer(norm(H))` und anschließend mit einem zweischichtigen GELU-MLP `H = H + mlp(H)`. Den Abschluss bilden eine Schichtnormalisierung und ein linearer Ausgabekopf (:numref:`fig_seqlm_backbone`).

<!--en:5d939266-->
![Die gemeinsame Grundarchitektur `SeqLM`. Jeder Block addiert erst die Ausgabe eines Mischers und dann die eines MLP zum Residualstrom. Beiden geht jeweils eine eigene Schichtnormalisierung voraus. Von Modell zu Modell wechselt in diesem Kapitel nur der Mischer. Die Sprachmodelle bestehen aus $L = 4$ aufeinandergestapelten Blöcken.](../img/mdl-modernrnn-seqlm-backbone.svg)

<!--en:5155b5ae-->
Als Attention-Baseline in `SeqLM` dient `AttentionMixer`. Dieses Modul kapselt kausale Multi-Head-Attention mit RoPE (:numref:`sec_gpt`, :numref:`sec_positional-information`), eine Initialisierung mit der Normalverteilung nach LeCun und eine optionale kanalweise kausale Faltung über $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ (später in :numref:`sec_hybrids` verwendet).

<!--en:5b9d52b2-->
Außerdem speichern wir drei Hilfsfunktionen für die Auswertung im gesamten Kapitel. Die erste, `loss_by_position`, berechnet an jeder Position des Kontextfensters die mittlere Kreuzentropie in Nat auf den Validierungsdaten:

<!--en:cf6ffb78-->
Mit `train_text` trainieren wir ein Modell für 1000 AdamW-Schritte (Aufwärmphase von 100 Schritten bis $2 \times 10^{-3}$, danach kosinusförmiges Abklingen der Lernrate). Die Funktion `text_table` führt jede Konfiguration mit zwei Zufallsstartwerten aus. Sie gibt die Parameteranzahl, den Validierungsverlust insgesamt und für die Positionsbereiche `0-15`, `16-63` und `64-127` sowie die Millisekunden pro Schritt aus:

<!--en:8d04044e-->
Nun vergleichen wir unsere rekurrenten Mischer und den Attention-Mischer auf WikiText-103 auf Byte-Ebene (`d2l.WikiText103`, Fenster von 128 Byte, Batchgröße 64):

<!--en:60d57e3b-->
Jedes `text_model` besteht aus 4 Residualblöcken der Breite $d = 256$. Wir vergleichen ein einfaches `RNN`, ein `LSTM`, eine `GRU` und kausale `attention` mit 8 Köpfen. Hinzu kommt ein einfaches RNN der Breite $d = 302$, das so auf dieselbe Parameteranzahl wie der GRU-Stapel kommt ($3.81\textrm{M}$ Parameter):

<!--en:c908f717-->
Ein Aufruf von `text_table` trainiert jede Variante zweimal und gibt die Ergebnisse nebeneinander aus:

<!--en:a8188d4f-->
Aus der Tabelle ergeben sich drei Befunde:

<!--en:42e577bd-->
1. **Residualblöcke und Gatter:** Schon 4 schmale RNN-Schichten mit $\tanh$ in einer Residualarchitektur mit vorgelagerter Normalisierung und MLPs erreichen $\sim 1.35$ Nat/Byte. Damit liegen sie gleichauf mit dem SwiGLU-GPT aus :numref:`sec_gpt` mit 6 Blöcken. Ersetzen wir das einfache RNN als Mischer durch ein LSTM oder eine GRU, sinkt der Validierungsverlust um weitere $0.05$ Nat/Byte (auf $\sim 1.30$). Verbreitern wir das einfache RNN auf die Parameteranzahl der GRU (`RNN, d=302`), schließt sich weniger als ein Drittel dieser Lücke (etwa $1.33$).

<!--en:888f5c11-->
2. **Nutzung von kurzem Kontext:** Bei allen Modellen sinkt der Verlust nach 1000 Trainingsschritten mit Fenstern von 128 Byte vom Bereich `0-15` zum Bereich `16-63` deutlich. Jenseits von Byte 16 ändert er sich um weniger als $0.005$. Die Gatter gewinnen ihren Vorsprung hier durch bessere Nutzung der benachbarten Kontextbytes, ohne dafür Informationen über 100 Schritte speichern zu müssen. Bei diesem kleinen Trainingsbudget liegt der Attention-Stapel (etwa $1.38$) schon in den ersten 16 Byte hinter den rekurrenten Mischern.

<!--en:c62c5daf-->
3. **Zeit pro Schritt:** Trotz fusionierter Bibliotheksschleifen brauchen LSTM und GRU pro Trainingsschritt $1.4\times$–$3\times$ so lange wie das einfache RNN. Der Grund ist der Aufwand innerhalb eines Schritts: drei oder vier Produkte für die Gatter und den Kandidaten, wo das einfache RNN nur eines berechnet. Die sequenzielle Auswertung erklärt den Unterschied nicht, denn auch das einfache RNN wartet auf den versteckten Zustand $\mathbf{h}_{t-1}$ des vorherigen Schritts. Bei 128 Schritten dominiert diese Abhängigkeit den Aufwand noch nicht und ist daher in der Tabelle nicht erkennbar. Der Vergleich verschiedener Sequenzlängen in :numref:`subsec_linear-recurrence-cost` isoliert ihren Einfluss: Dort steigt die Zeit pro GRU-Schritt zwischen 512 und 16 384 Token auf mehr als das Fünfzehnfache.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:293efa7e-->
In einem einfachen rekurrenten Netz durchlaufen Signale und Gradienten ein Produkt von Jacobi-Matrizen der einzelnen Schritte, die alle von derselben Matrix $\mathbf{W}_\textrm{hh}$ abhängen. Über lange Zeiträume bleiben sie nur erhalten, wenn der Spektralradius nahe bei $1$ liegt. Rekurrente Netze mit Gattern führen einen elementweise additiven Speicherpfad $c_t = f_t c_{t-1} + i_t \tilde{c}_t$ ein, den erlernte Gatter mit Sigmoidfunktion steuern. Das LSTM trennt die interne Speicherzelle $\mathbf{c}_t$, auf die Eingabe-, Vergessens- und Ausgabegatter den Zugriff steuern, vom nach außen sichtbaren Zustand $\mathbf{h}_t$. Die GRU koppelt Vergessens- und Eingabegatter ($1 - \mathbf{z}_t$) und filtert den rekurrenten Zustand mit einem Reset-Gatter $\mathbf{r}_t$. Dafür genügt ihr ein einziger Zustandsvektor. Beim synthetischen verzögerten Abruf entscheidet die Initialisierung der Vergessens- und Update-Gatter nahe bei $1$ (`init_keep_gate`) über den Erfolg: Statt einer Genauigkeit auf Zufallsniveau bei $T \ge 50$ erreichen die Modelle $100\%$ korrekte Abrufe bei $T = 200$. Auf WikiText-103 auf Byte-Ebene erzielen LSTM- und GRU-Mischer in einer Residualarchitektur (`SeqLM`) einen um $0.03$–$0.05$ Nat/Byte niedrigeren Verlust als einfache RNNs gleicher Breite oder gleicher Parameteranzahl. Da ihre Gatter aber von $\mathbf{h}_{t-1}$ abhängen, erfordert das Training weiterhin $T$ sequenzielle Schritte.

<!--en:2294243a-->
## Übungen

<!--en:ba88212b-->
1. **Einen Wert speichern.** Angenommen, das Zeichen in Schritt $s$ bestimmt die Vorhersage in einem viel späteren Schritt $t$. Eine LSTM-Einheit soll es von $s$ bis $t$ speichern.
    1. Geben Sie für jedes Gatter dieser Einheit in :eqref:`lstm_gates` an, welchen Wert es im Schritt $s$, in den Schritten strikt zwischen $s$ und $t$ sowie im Schritt $t$ annehmen sollte. Geben Sie anschließend die entsprechenden Werte für das Update-Gatter einer GRU-Einheit in :eqref:`gru_H` an.
    1. Ein Gatter mit Sigmoidfunktion erreicht nie den Wert 1. Angenommen, die Präaktivierung des Vergessensgatters bleibt konstant bei $b$, sodass sein Wert in jedem Schritt $\sigma(b)$ beträgt. Welcher Anteil des gespeicherten Werts bleibt entlang des direkten Pfads über $k$ Schritte erhalten? Wie groß muss $b$ sein, damit nach 1000 Schritten noch 90% des Werts erhalten sind?
    1. Zeigen Sie, dass ein Bias des Vergessensgatters von $\log(T - 1)$ bei Gewichten von null den anfänglichen Gatterwert $1 - 1/T$ ergibt. Für große $T$ sinkt ein gespeicherter Wert also um etwa den Faktor $e$ über $T$ Schritte. Die *Chrono-Initialisierung* von :citet:`Tallec.Ollivier.2018` wählt den Bias des Vergessensgatters jeder Einheit als $\log u$, wobei $u$ auf $[1, T_\textrm{max} - 1]$ gleichverteilt ist, und setzt $\mathbf{b}_\textrm{i} =
       -\mathbf{b}_\textrm{f}$. Zeigen Sie, dass bei Gewichten von null dann $\mathbf{f}_t + \mathbf{i}_t = 1$ gilt. Vergleichen Sie die resultierende Aktualisierung der Zelle mit :eqref:`gru_H`.

<!--en:36eb5965-->
1. **Den Wertebereich der Ausgabe begrenzen.** Der Wert am Eingabeknoten $\tilde{\mathbf{c}}_t$ wird bereits durch eine $\tanh$-Funktion begrenzt. Dennoch wendet :eqref:`lstm_update` vor dem Ausgabegatter erneut $\tanh$ auf $\mathbf{c}_t$ an.
    1. Gehen Sie von $\mathbf{c}_0 = \mathbf{0}$ aus und zeigen Sie, dass jeder Eintrag von $\mathbf{c}_t$ im Betrag kleiner als $t$ ist. Nehmen Sie weiter an, dass alle Werte der Vergessensgatter höchstens $f_\textrm{max} < 1$ betragen und alle Werte der Eingabegatter höchstens $i_\textrm{max}$. Zeigen Sie, dass dann jeder Eintrag im Betrag unter $i_\textrm{max} / (1 - f_\textrm{max})$ bleibt, unabhängig von $t$.
    1. Angenommen, die Ausgabe wäre stattdessen $\mathbf{h}_t = \mathbf{o}_t \odot
       \mathbf{c}_t$. Was geschieht nach vielen Schritten der Akkumulation im nächsten Schritt mit den Präaktivierungen der Gatter? Wie wirkt sich das auf die Gradienten aus, die durch die Gatter fließen?

<!--en:cca46e2b-->
1. **Zurücksetzen vor oder nach dem Matrixprodukt.** Die fusionierten GRU-Schichten in Deep-Learning-Frameworks wenden das Reset-Gatter nach dem rekurrenten Matrixprodukt an. Abgesehen von der Position der Bias-Terme berechnen sie den Kandidaten
   $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_\textrm{xh} \mathbf{x}_t + \mathbf{r}_t \odot (\mathbf{W}_\textrm{hh} \mathbf{h}_{t-1}) + \mathbf{b}_\textrm{h})$$
   anstelle von :eqref:`gru_tilde_H`.
    1. Zeigen Sie, dass die beiden Kandidaten für alle $\mathbf{r}_t$ und $\mathbf{h}_{t-1}$ genau dann übereinstimmen, wenn $\mathbf{W}_\textrm{hh}$ diagonal ist. Für welche Werte des Reset-Gatters stimmen sie auch bei einer allgemeinen Matrix $\mathbf{W}_\textrm{hh}$ überein?
    1. Die Eingabeprodukte $\mathbf{W}_{\textrm{x}\cdot} \mathbf{x}_t$ aller Schritte lassen sich vor Beginn der Rekursion als ein einziges großes Produkt berechnen. Wie viele Matrixprodukte mit dem vorherigen Zustand muss jeder Schritt bei den beiden Formen des Kandidaten jeweils nacheinander ausführen? Warum verwenden fusionierte GRU-Kernel die oben angegebene Form?

<!--en:fc179a87-->
4. [code] **Eine GRU von Grund auf implementieren.** Schreiben Sie eine Klasse `GRUScratch`, indem Sie `LSTMScratch` anpassen. Verwenden Sie drei Köpfe gemäß :eqref:`gru_gates` und :eqref:`gru_tilde_H`, die Aktualisierung aus :eqref:`gru_H` und einen Zustand, der allein aus $\mathbf{h}_t$ besteht.
    1. Kopieren Sie wie beim LSTM die Gewichte der GRU-Schicht aus der Bibliothek in Ihre Schicht und vergleichen Sie die Ausgaben. Sie stimmen nicht überein. Ändern Sie anhand der vorherigen Aufgabe eine Zeile Ihrer Schicht so, dass die Ausgaben übereinstimmen. (Behalten Sie außerdem das letzte Drittel von `bias_hh_l0`, den rekurrenten Bias des Kandidaten, als separaten Vektor innerhalb des Reset-Gatters bei oder setzen Sie es vor dem Kopieren auf null.)
    1. Entfernen Sie das Reset-Gatter, indem Sie $\mathbf{r}_t = \mathbf{1}$ festsetzen. Entfernen Sie in einer zweiten Schicht das Update-Gatter, indem Sie $\mathbf{z}_t =
       \mathbf{0}$ festsetzen. Sagen Sie vor dem Training anhand von :eqref:`gru_H` voraus, welcher Eingriff den verzögerten Abruf stärker beeinträchtigt. Führen Sie anschließend das Experiment aus :numref:`subsec_delayed-recall` aus und prüfen Sie Ihre Vorhersage.

<!--en:e6ee5b3c-->
4. [code] **Eine GRU von Grund auf implementieren.** Schreiben Sie eine Klasse `GRUScratch`, indem Sie `LSTMScratch` anpassen. Verwenden Sie drei Köpfe gemäß :eqref:`gru_gates` und :eqref:`gru_tilde_H`, die Aktualisierung aus :eqref:`gru_H` und einen Zustand, der allein aus $\mathbf{h}_t$ besteht.
    1. Kopieren Sie wie beim LSTM die Gewichte der GRU-Schicht aus der Bibliothek in Ihre Schicht und vergleichen Sie die Ausgaben. Sie stimmen nicht überein. Ändern Sie anhand der vorherigen Aufgabe eine Zeile Ihrer Schicht so, dass die Ausgaben übereinstimmen.
    1. Entfernen Sie das Reset-Gatter, indem Sie $\mathbf{r}_t = \mathbf{1}$ festsetzen. Entfernen Sie in einer zweiten Schicht das Update-Gatter, indem Sie $\mathbf{z}_t =
       \mathbf{0}$ festsetzen. Sagen Sie vor dem Training anhand von :eqref:`gru_H` voraus, welcher Eingriff den verzögerten Abruf stärker beeinträchtigt. Führen Sie anschließend das Experiment aus :numref:`subsec_delayed-recall` aus und prüfen Sie Ihre Vorhersage.

<!--en:f4fc336c-->
5. [code] **Weitere Pfade durch die Zelle.** ● Betrachten Sie ein LSTM mit einer Eingabe und einer versteckten Einheit. Damit ist jede Größe in :eqref:`lstm_gates` und :eqref:`lstm_update` ein Skalar. Bezeichnen Sie die rekurrenten Gewichte des Vergessensgatters, des Eingabegatters und des Eingabeknotens in dieser Reihenfolge mit $w_\textrm{hf}$, $w_\textrm{hi}$ und $w_\textrm{hc}$.
    1. Das Ausgabegatter $o_{t-1}$ hängt von $h_{t-2}$ ab, aber nicht von $c_{t-1}$. Leiten Sie mithilfe von $h_{t-1} = o_{t-1} \tanh(c_{t-1})$ die totale Ableitung $\textrm{d}c_t / \textrm{d}c_{t-1}$ her. Schreiben Sie sie als Summe aus dem Wert $f_t$ des direkten Pfads und einem Korrekturterm. Zerlegen Sie den Korrekturterm in zwei Faktoren: einen Term aus Schritt $t$ und die Ableitung $\textrm{d}h_{t-1} / \textrm{d}c_{t-1}$.
    1. Für welche Werte der Gatter, von $\tilde{c}_t$ und von $c_{t-1}$ verschwindet der Korrekturterm? Welche dieser Bedingungen kann das Netz über seine Gatter einstellen, während die Zelle einen Wert speichert?
    1. Trainieren Sie ein LSTM auf verzögerten Abruf bei einer Verzögerung, bei der es die Aufgabe löst. Verwenden Sie automatisches Differenzieren, um für eine Testsequenz die Norm des Gradienten des Verlusts bezüglich der Zelle $\mathbf{c}_{T-k}$ für jedes $k$ zu berechnen. Vergleichen Sie diese Norm mit der Norm des Gradienten, den allein der direkte Pfad überträgt. Dieser Pfad multipliziert den Gradienten bei $\mathbf{c}_T$ elementweise mit dem Produkt der Werte des Vergessensgatters über die letzten $k$ Schritte. Wo gehen die beiden Kurven auseinander, und welche liegt höher?

<!--en:7e0201f4-->
1. [code] **Informationsleck bei bidirektionaler Verarbeitung.** ● Eine rekurrente Schicht kann die Sequenz auch rückwärts lesen. Ein *bidirektionales* Netz :cite:`Schuster.Paliwal.1997` lässt eine zweite Schicht vom letzten Schritt zum ersten laufen und verkettet die Ausgaben beider Schichten. So hängt die Darstellung jeder Position von der gesamten Sequenz ab. Das hilft, wenn die ganze Eingabe vorliegt, etwa beim Annotieren der Wörter eines Satzes oder beim Kodieren eines Satzes für die Übersetzung. Bei einem Sprachmodell verhält es sich anders.
    1. Bauen Sie einen bidirektionalen Mischer: Lassen Sie ein zweites LSTM die umgekehrte Sequenz verarbeiten, kehren Sie die Reihenfolge seiner Ausgaben um und verketten Sie diese mit den Ausgaben eines vorwärts laufenden LSTM. Projizieren Sie das Ergebnis zurück auf die Breite $d$. Trainieren Sie `SeqLM` mit solchen Mischern anstelle der LSTMs aus :numref:`subsec_deep-rnn`. Verwenden Sie dazu `train_text` und vergleichen Sie den Validierungsverlust mit dem des Vorwärtsmodells. Erklären Sie den erhaltenen Wert: Welche Eingabe liest die rückwärts laufende Schicht an Position $t$? Was könnte das Modell beim Generieren tun, wenn die nachfolgenden Bytes noch gar nicht existieren?
    1. Ein bidirektionaler Prädiktor ohne Informationsleck sagt das Byte $x_t$ voraus, indem er den Vorwärtszustand nach Schritt $t-1$ mit dem Rückwärtszustand nach Schritt $t+1$ kombiniert. Er modelliert also die Wahrscheinlichkeit von $x_t$ gegeben alle anderen Bytes im Fenster. Zeigen Sie anhand einer Sequenz aus zwei binären Token, dass das Produkt dieser bedingten Wahrscheinlichkeiten über $t$ in der Summe über alle Sequenzen nicht unbedingt eins ergibt. Was folgt daraus, wenn man mit diesem Produkt ganze Sequenzen bewerten oder eine Perplexität angeben möchte?

<!--en:922f4127-->
[Diskussion](https://d2l.discourse.group/t/1057)

<!--en:9ae4b930-->
[Diskussion](https://d2l.discourse.group/t/18016)

<!--en:ebf0985c-->
Rekurrente Netze mit Gattern: LSTM und GRU<br>**verschwindende Gradienten · Vergessensgates mit Sigmoidfunktion · Speicherzellen · Initialisierung des Gatter-Bias · Sequenzmischer in Residualblöcken**

<!--en:108647b1-->
::: {.slide title="Warum einfache RNNs grammatische Bezüge über größere Abstände verlieren"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:7103ba49-->
- **Überschreiben des gesamten Zustands:** Jeder Schritt multipliziert $\mathbf{h}_{t-1}$ mit $\mathbf{W}_\textrm{hh}$ und begrenzt das Ergebnis mit $\tanh$.

<!--en:3bb174bc-->
- **Geometrisches Abklingen:** Bei $w = 0.9$ schrumpfen Störungen auf $0.9^{30} \approx 4\%$ nach $30$ Schritten und auf $0.9^{100} \approx 0.003\%$ nach $100$ Schritten.

<!--en:8520e448-->
::: {.slide title="Produkte von Jacobi-Matrizen und der Grenzfall beim Spektralradius"}

<!--en:3f6eb9b2-->
[Verschwindende und explodierende Gradienten]{.kicker}

<!--en:a4717ead-->
- **Kontraktion und Expansion:** Wegen $|1 - h_s^2| \le 1$ skalieren die Potenzen wie $\rho^k$ ($\rho = \max_i |\lambda_i(\mathbf{W}_\textrm{hh})|$). Für $\rho < 1$ klingen sie ab, für $\rho > 1$ explodieren sie.

<!--en:0a9d68d6-->
- **Unterschiedliche Folgen:** Gradienten-Clipping begrenzt explodierende Gradienten. Sind Gradienten über lange Zeitabstände durch Unterlauf verloren gegangen, kann es sie nicht wiederherstellen.

<!--en:0cfd9f2e-->
::: {.slide title="Von unbeschränkten laufenden Summen zu erlernten Vergessensgates"}

<!--en:1eb1f3ff-->
[Funktionsweise der Gatter]{.kicker}

<!--en:49379e14-->
- **Direkter Pfad für jede Komponente:** Entlang des direkten Speicherpfads gilt $\partial c_t / \partial c_{t-k} = \prod_{s=t-k+1}^t f_s \le 1$, ganz ohne Matrixprodukt mit $\mathbf{W}_\textrm{hh}$.

<!--en:9ff7eb72-->
- **Einstellbarer Zeithorizont:** Bei $f_t \approx 0.99$–$0.999$ verlängert sich die Halbwertszeit des Speichers von $1$ Schritt auf $69$–$693$ Schritte.

<!--en:62e78e80-->
::: {.slide title="Die Architektur des Long Short-Term Memory (LSTM)"}

<!--en:3c66796d-->
[Funktionsweise des LSTM]{.kicker}

<!--en:fdb4f04c-->
Das LSTM (Hochreiter und Schmidhuber, 1997; Gers et al., 2000) führt eine interne **Speicherzelle** $\mathbf{c}_t \in \mathbb{R}^d$ und einen nach außen sichtbaren **versteckten Zustand** $\mathbf{h}_t \in \mathbb{R}^d$ mit:

<!--en:4af7bd2e-->
- **Drei Gatter mit Sigmoidfunktion und Werten in $(0, 1)^d$:** Eingabegatter $\mathbf{i}_t$, Vergessensgatter $\mathbf{f}_t$ und Ausgabegatter $\mathbf{o}_t$ berechnen jeweils $\sigma(\mathbf{W}_{\textrm{x}\cdot}\mathbf{x}_t + \mathbf{W}_{\textrm{h}\cdot}\mathbf{h}_{t-1} + \mathbf{b}_\cdot)$.

<!--en:a58daf49-->
- **Kandidat am Eingabeknoten:** $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_\textrm{xc}\mathbf{x}_t + \mathbf{W}_\textrm{hc}\mathbf{h}_{t-1} + \mathbf{b}_\textrm{c}) \in (-1, 1)^d$.

<!--en:05d8fa3b-->
- **Additive Aktualisierung und Auslesen über ein Gatter:**
  $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
  mit dem nach außen sichtbaren Zustand $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$.

<!--en:7633d7f4-->
::: {.slide title="Trennung von internem Speicher und sichtbarer Ausgabe"}

<!--en:e37170a7-->
[Internes Speichern und direkter Gradientenpfad]{.kicker}

<!--en:fded6141-->
- **Intern speichern bis zum Abruf:** Bei $f_t = 0.95, i_t = 0.1$ bleibt $c_t = 0.71$ erhalten. Das weitgehend geschlossene Ausgabegatter ($o_t = 0.2$) gibt nur einen kleinen Wert an nachfolgende Schichten weiter ($h_t = 0.12$).

<!--en:50056636-->
- **Begrenzter Betrag beim Auslesen:** Durch $\tanh(\mathbf{c}_t)$ gilt $\mathbf{h}_t \in (-1, 1)^d$, selbst wenn wiederholtes Schreiben zu $|c_t| > 1$ führt.

<!--en:fb0eeaa3-->
- **Direkte Jacobi-Matrix der Zelle:** Entlang des oberen Zellpfads gilt $\partial \mathbf{c}_t / \partial \mathbf{c}_{t-1} = \textrm{diag}(\mathbf{f}_t)$.

<!--en:ed1edc1f-->
::: {.slide title="Gemeinsame Matrixprojektion für 4 Zweige und Ergebnisprüfung" only="pytorch"}

<!--en:61aefc63-->
[LSTM-Implementierung]{.kicker}

<!--en:f933c041-->
Da $\mathbf{i}_t$, $\mathbf{f}_t$, $\tilde{\mathbf{c}}_t$ und $\mathbf{o}_t$ dieselben Eingaben $(\mathbf{x}_t, \mathbf{h}_{t-1})$ lesen, genügt es, ihre Gewichte in $\mathbf{W}_\textrm{x} \in \mathbb{R}^{d_\textrm{in} \times 4d}$ und $\mathbf{W}_\textrm{h} \in \mathbb{R}^{d \times 4d}$ zusammenzufassen. So berechnet ein Matrixprodukt für den gesamten Batch pro Schritt alle vier Präaktivierungen:

<!--en:adcf8cc0-->
Die selbst implementierte Schleife und die fusionierte Schicht `nn.LSTM` stimmen bis auf `float32`-Rundungsfehler ($\sim 10^{-7}$) überein. Daher kommen beim Training die fusionierten Kernel der Bibliothek zum Einsatz.

<!--en:2ef06a9a-->
::: {.slide title="Gemeinsame Matrixprojektion für 4 Zweige und Ergebnisprüfung" only="jax"}

<!--en:61aefc63-2-->
[LSTM-Implementierung]{.kicker}

<!--en:f933c041-2-->
Da $\mathbf{i}_t$, $\mathbf{f}_t$, $\tilde{\mathbf{c}}_t$ und $\mathbf{o}_t$ dieselben Eingaben $(\mathbf{x}_t, \mathbf{h}_{t-1})$ lesen, genügt es, ihre Gewichte in $\mathbf{W}_\textrm{x} \in \mathbb{R}^{d_\textrm{in} \times 4d}$ und $\mathbf{W}_\textrm{h} \in \mathbb{R}^{d \times 4d}$ zusammenzufassen. So berechnet ein Matrixprodukt für den gesamten Batch pro Schritt alle vier Präaktivierungen:

<!--en:0b6a34a6-->
Die selbst implementierte Schleife und `nnx.LSTMCell` stimmen bis auf `float32`-Rundungsfehler ($\sim 10^{-7}$) überein. Daher kommen beim Training die fusionierten Kernel der Bibliothek zum Einsatz.

<!--en:7a6217cd-->
::: {.slide title="Gated Recurrent Units (GRU): gekoppeltes Zurücksetzen und Aktualisieren"}

<!--en:72ee2d92-->
[GRU-Architektur]{.kicker}

<!--en:9987a487-->
Die GRU (Cho et al., 2014) fasst Zelle und versteckten Zustand in einem einzigen Vektor $\mathbf{h}_t \in \mathbb{R}^d$ zusammen. Zwei Gatter steuern ihn:

<!--en:9a4bdf79-->
- **Reset-Gatter $\mathbf{r}_t$ und Update-Gatter $\mathbf{z}_t$:** Beide berechnen ihren Wert mit $\sigma(\mathbf{W}_{\textrm{x}\cdot}\mathbf{x}_t + \mathbf{W}_{\textrm{h}\cdot}\mathbf{h}_{t-1} + \mathbf{b}_\cdot)$.

<!--en:981a5fec-->
- **Kandidat mit Reset-Gatter:** $\mathbf{r}_t$ steuert, wie viel vom vorherigen Zustand in den nichtlinearen Kandidaten eingeht:
  $$\tilde{\mathbf{h}}_t = \tanh\big(\mathbf{W}_\textrm{xh}\mathbf{x}_t + \mathbf{W}_\textrm{hh}(\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_\textrm{h}\big)$$

<!--en:0c562117-->
- **Konvexe Interpolation ($i_t = 1 - f_t$):**
  $$\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$$

<!--en:dbe0dff9-->
::: {.slide title="LSTM und GRU: Unterschiede in Aufbau und Rechenaufwand"}

<!--en:61810873-->
[Vergleich der Architekturen]{.kicker}

<!--en:1c126ac6-->
- **Zustand und Anzahl der Gatter:** Das LSTM führt zwei Vektoren $(\mathbf{c}_t, \mathbf{h}_t)$ mit, die 3 unabhängige Gatter steuern ($8d^2$ Parameter). Die GRU führt einen Vektor $\mathbf{h}_t$ mit 2 Gattern mit ($6d^2$ Parameter, also $25\%$ weniger).

<!--en:ebb04294-->
- **Begrenzung des Zustands:** Das LSTM begrenzt $\mathbf{h}_t$ beim Auslesen durch $\tanh(\mathbf{c}_t)$. Bei der GRU sorgt die Konvexkombination $\mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$ automatisch dafür, dass $\mathbf{h}_t \in [-1, 1]^d$ gilt.

<!--en:cb528d8a-->
::: {.slide title="Synthetischer Belastungstest: verzögerter Abruf"}

<!--en:5e7332f1-->
[Versuchsaufbau]{.kicker}

<!--en:2f678452-->
Die Gleichungen mit Gattern *erlauben* langfristiges Speichern. Findet der Gradientenabstieg solche Lösungen? Als Test für das Speichervermögen dient eine synthetische Aufgabe zum **verzögerten Abruf**:

<!--en:56caead4-->
- **Aufgabe:** In Schritt $1$ erscheint ein Zielsymbol aus $\{0, \ldots, 7\}$, gefolgt von $T - 1$ zufälligen Fülltoken aus $\{8, \ldots, 15\}$. In Schritt $T$ muss das Netz das Anfangssymbol klassifizieren (Genauigkeit bei zufälligem Raten $= 1/8 = 0.125$).

<!--en:44e65bed-->
- **Architektur (`RecallModel`):** Tokeneinbettung ($d_\textrm{in} = 32$) $\to$ eine rekurrente Schicht ($d = 64$, normalverteilte Gewichte nach LeCun) $\to$ linearer Klassifikator in Schritt $T$. Training über $1{,}500$ Adam-Schritte ($\text{lr} = 3 \times 10^{-3}$, Batchgröße $128$).

<!--en:e28e7fc3-->
::: {.slide title="Gatter mit Bias null starten mit einer Halbwertszeit von nur 1 Schritt"}

<!--en:4f4ec00e-->
[Risiko bei der Initialisierung]{.kicker}

<!--en:60de47a1-->
- **Anfängliche Halbwertszeit von einem Schritt:** Bei $\mathbf{b}_\textrm{f} = \mathbf{0}$ starten die Gatter bei $\sigma(0) = 1/2$. Der direkte Gradient von Schritt $T$ bis Schritt $1$ wird um den Faktor $2^{-(T-1)}$ abgeschwächt.

<!--en:44b8bb72-->
- **Rundungsverlust bei $T \ge 50$:** Der Beitrag $2^{-19} \approx 1.9 \times 10^{-6}$ bei $T=20$ bleibt in `float32` erhalten. Dagegen geht $2^{-49} \approx 1.8 \times 10^{-15}$ bei $T=50$ durch Rundung verloren, wenn er sich im Zustand zu den Beiträgen der Größenordnung $O(1)$ aus den letzten Schritten addiert.

<!--en:7b1b562c-->
::: {.slide title="Initialisierung auf Erhalt und Begrenzung des Zellzustands"}

<!--en:e33612ff-->
[Chrono-Initialisierung und Gatter-Bias]{.kicker}

<!--en:7c18aee3-->
- **Anfangshorizont von 100 Schritten:** Mit $b_\textrm{f} = +5$ starten die Gatter bei $\sigma(5) \approx 0.9933$ ($\tau_{1/2} \approx 103$ Schritte). Damit bleiben $0.9933^{50} \approx 71\%$ des Gradienten über $50$ Schritte erhalten.

<!--en:d27dfe23-->
- **Gekoppelter Eingabe-Bias ($b_\textrm{i} = -5$):** Bei $b_\textrm{i} = 0$ können LSTM-Zellen bis zur Schranke $|c_t| \le 0.5 / 0.0067 \approx 75$ wachsen. Durch $b_\textrm{i} = -b_\textrm{f} = -5$ gilt die Kopplung $i_t = 1 - f_t$ und damit $|c_t| \le 1$.

<!--en:a4fea71a-->
::: {.slide title="Initialisierung einfacher RNNs: Grenzfall und Kontraktion"}

<!--en:fa72b027-->
[Analyse des Spektralradius]{.kicker}

<!--en:ea14933f-->
Vergleich vor dem Training: zwei Initialisierungen der rekurrenten $64 \times 64$-Matrix $\mathbf{W}_\textrm{hh}$ eines einfachen RNN:

<!--en:2432fe21-->
- **Normalverteilung nach LeCun ($\mathcal{N}(0, 1/64)$):** Nach dem Kreisgesetz der Zufallsmatrizentheorie füllen die Eigenwerte die Einheitskreisscheibe ($\rho \approx 1.02$), obwohl der größte Singulärwert $\approx 2.0$ beträgt. Kleine Zustände entlang der führenden Eigenmoden klingen weder ab noch explodieren sie.

<!--en:445b54d6-->
- **Standard-Gleichverteilung ($\mathcal{U}[-1/8, 1/8]$):** Ihre Varianz beträgt $1/3$ der Varianz bei der LeCun-Initialisierung. Das ergibt den Spektralradius $\rho \approx 0.62$ und schwächt jedes Signal um den Faktor $\sim 0.6^k$ ab.

<!--en:2ea6e4f5-->
::: {.slide title="Verzögerter Abruf über 20 bis 200 Zeitschritte"}

<!--en:dfdf448d-->
[Messergebnisse]{.kicker}

<!--en:f3e35938-->
Genauigkeit bei zwei Zufallsstartwerten und Verzögerungen $T \in \{20, 50, 100, 200\}$ (Zufallsniveau $= 0.125$):

<!--en:7646dc3b-->
Mit einem anfänglichen Bias von $+5$ für Vergessens- und Update-Gatter (`keep gate`) steigt die Genauigkeit vom Zufallsniveau ($0.12$) bei $T \ge 50$ auf $1.00$ für jeden Zufallsstartwert und jede Verzögerung bis $T = 200$.

<!--en:b248e9d2-->
::: {.slide title="Warum die Initialisierung der Gatter über den langfristigen Abruf entscheidet"}

<!--en:c77cc3d1-->
[Auswertung der Abrufversuche]{.kicker}

<!--en:6fada77c-->
Der Vergleich der drei Modellpaare in der Tabelle zum verzögerten Abruf trennt den Einfluss der Architektur von dem der Initialisierung:

<!--en:f02ff73c-->
- **LSTM und GRU mit Bias null:** Die GRU löst die Aufgabe bei $T = 20$ ($1.00$), das LSTM ist dort unzuverlässig ($0.2$–$1.00$). Beide fallen auf Zufallsniveau ($0.12$–$0.14$) bei $T \in \{50, 100, 200\}$, weil ein Beitrag von $2^{-49}$ in `float32` durch Rundung verloren geht.

<!--en:8b469768-->
- **LSTM und GRU mit auf Erhalt eingestellten Gattern:** Architektur und Parameteranzahl bleiben gleich. Mit einem Anfangswert von $\sigma(5) = 0.993$ erreichen sie jedoch **eine Genauigkeit von $1.00$ in jedem Lauf** von $T = 20$ bis $T = 200$.

<!--en:29f7f6f1-->
- **Einfache RNNs im Grenzbereich und bei Kontraktion:** Mit $\rho \approx 1.02$ behält das LeCun-RNN das Symbol gelegentlich bis $T = 100$ (große Varianz zwischen Zufallsstartwerten). Mit $\rho \approx 0.62$ ist das gleichverteilte RNN bis $T = 50$ nur gelegentlich erfolgreich und scheitert ab $T = 100$ immer.

<!--en:143507d0-->
Gatter ersetzen die empfindliche Abstimmung der Eigenwerte der gesamten Matrix durch **erlerntes Abklingen für jede Einheit**, sofern die anfänglichen Bias-Terme den Gradienten über genügend Schritte erhalten.

<!--en:671fc43b-->
::: {.slide title="Vergleichbare Sprachmodelle durch gemeinsame Residualblöcke"}

<!--en:7a563f94-->
[Die Grundarchitektur von SeqLM]{.kicker}

<!--en:16332701-->
- **Einheitlicher Aufbau:** Byte-Einbettung $\to$ $4$ Residualblöcke mit vorgelagerter Normalisierung ($d = 256$, GELU-MLP der Breite $4d$) $\to$ LayerNorm und Ausgabekopf für Fenster von 128 Byte aus WikiText-103.

<!--en:c0c2dc4d-->
- **Vergleich der Mischer:** Nur `mixer(·)` ändert sich von §§{sec_lstm} bis §{num:sec_hybrids}.

<!--en:99503069-->
::: {.slide title="WikiText-103 auf Byte-Ebene: Mischer im direkten Vergleich"}

<!--en:a94e4287-->
[Benchmark für Sprachmodellierung]{.kicker}

<!--en:71a0f821-->
Zwei Läufe pro Konfiguration liefern Parameteranzahl, gesamten Validierungsverlust (Nat/Byte), Verlust nach Positionsbereich im Fenster (`0-15`, `16-63`, `64-127`) und Zeit pro Trainingsschritt:

<!--en:376d2625-->
::: {.slide title="WikiText-103: Ergebnisse und Zeit pro Schritt"}

<!--en:8d68a0f4-->
[Modellgüte und Zeitaufwand]{.kicker}

<!--en:5b9a50d9-->
Der WikiText-103-Benchmark liefert drei Erkenntnisse zu Modellierung und Rechenaufwand:

<!--en:554bcf87-->
1. **Residualblöcke schließen die Lücke des einfachen RNN:** Schon $4$ schmale RNN-Schichten mit $\tanh$ in `SeqLM` erreichen $\sim 1.35$ Nat/Byte. Damit liegen sie gleichauf mit dem SwiGLU-GPT mit 6 Blöcken aus §{num:chap_transformers} und vor dem GELU-Stapel mit 4 Blöcken und `attention` ($1.38$) bei einem Budget von $1{,}000$ Schritten.

<!--en:2fbf567c-->
2. **Gatter bringen mehr als zusätzliche Parameter:** Ein `LSTM` oder eine `GRU` anstelle des einfachen RNN senkt den Verlust um $0.05$ Nat/Byte auf $\sim 1.30$. Bei einer Verbreiterung des einfachen RNN auf die $3.81\text{M}$ Parameter der GRU (`RNN, d=302`) schließt sich weniger als ein Drittel der Lücke ($\sim 1.33$).

<!--en:f849879b-->
3. **Zeit pro Schritt:** `LSTM` und `GRU` brauchen **$1.4\times$–$3\times$ so lange pro Schritt** wie das einfache RNN, da sie jeweils drei oder vier Produkte anstelle eines einzigen berechnen. Alle drei warten auf $\mathbf{h}_{t-1}$. Der Vergleich verschiedener Sequenzlängen in §{sec_linear-recurrence} isoliert diesen Aufwand.

<!--en:7f673778-->
::: {.slide title="Verständnisfragen: Halbwertszeiten der Gatter und Sättigungsgrenzen"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:38a668a3-->
Drei Fragen zum Entwurf beantworten:

<!--en:a945c0d7-->
1. Ein LSTM-Vergessensgatter bleibt bei $f_t = 0.99$. Wie viele Schritte beträgt die Halbwertszeit des Speichers, und welcher Anteil des direkten Zellgradienten bleibt nach $k = 100$ Schritten erhalten?

<!--en:34910dc4-->
2. Warum setzt `init_keep_gate` beim LSTM den Bias des Eingabegatters auf $b_\textrm{i} = -5$ und zugleich $b_\textrm{f} = +5$, während die GRU nur $b_\textrm{z} = +5$ benötigt?

<!--en:0aa2aab7-->
3. Warum kann Gradienten-Clipping ein RNN mit Spektralradius $\rho = 1.3$ stabilisieren, aber einem LSTM mit Bias null beim verzögerten Abruf über $100$ Schritte nicht helfen?

<!--en:557b98b6-->
**Lösungen:**

<!--en:7b6e0fba-->
1. Die Halbwertszeit beträgt $\tau_{1/2} = \ln 2 / \ln(1/0.99) \approx \mathbf{69\text{ steps}}$. Nach $100$ Schritten bleiben $0.99^{100} \approx \mathbf{36.6\%}$ des direkten Gradienten erhalten.

<!--en:16a6fdb9-->
2. Die GRU koppelt ihr Eingabegewicht automatisch an $1 - z_t = 1 - \sigma(5) = \sigma(-5)$. Lässt man beim LSTM $b_\textrm{i} = 0$ ($i_t = 0.5$), können Zellzustände bis auf $0.5 / (1 - 0.9933) \approx 75$ wachsen. Dadurch gerät $\tanh(c_t)$ in Sättigung.

<!--en:b02d74d4-->
3. Gradienten-Clipping skaliert zu große Gradienten nach unten. Dagegen geht $2^{-99} \approx 1.6 \times 10^{-30}$ in `float32` beim Addieren zu den Gradienten der letzten Schritte durch Rundung ganz verloren. Damit geht auch die Richtungsinformation verloren.

<!--en:850eef61-->
::: {.slide title="Zusammenfassung: einfache rekurrente Architekturen und Architekturen mit Gattern"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:d9461fc0-->
| Architektur / Konfiguration | Direkter Speicherpfad | Rekurrente Parameter ($d_\textrm{in}=d$) | Abrufgenauigkeit bei $T=200$ | WikiText-103 (Nat/Byte) | Trainingstiefe |

<!--en:05f7c249-->
| **Einfaches RNN (Gleichverteilung)** | $\tanh(\mathbf{W}_\textrm{hh}\mathbf{h}_{t-1} + \cdots)$, $\rho \approx 0.62$ | $2d^2$ | $0.12$ (Zufallsniveau) | – | $T$ sequenzielle Schritte |

<!--en:448fd2e5-->
| **Einfaches RNN (Normalverteilung nach LeCun)** | $\tanh(\mathbf{W}_\textrm{hh}\mathbf{h}_{t-1} + \cdots)$, $\rho \approx 1.02$ | $2d^2$ | $0.12$–$0.24$ (bis $1.00$ bei $T=100$) | $1.346$–$1.349$ ($1.332$–$1.337$ bei $d=302$) | $T$ sequenzielle Schritte |

<!--en:9bf0431c-->
| **LSTM (Gatter-Bias null)** | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$, $f_0 = 0.5$ | $8d^2$ | $0.12$ (Zufallsniveau bei $T \ge 50$) | **$1.292$–$1.296$** | $T$ sequenzielle Schritte |

<!--en:10618da9-->
| **GRU (Gatter-Bias null)** | $\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1-\mathbf{z}_t)\odot \tilde{\mathbf{h}}_t$ | $6d^2$ | $0.12$–$0.13$ (Zufallsniveau bei $T \ge 50$) | **$1.297$–$1.300$** | $T$ sequenzielle Schritte |

<!--en:47cfde43-->
| **LSTM / GRU (Gatter auf Erhalt, $+5$)** | Anfänglicher Gatterwert $\sigma(5) = 0.993$ ($\tau_{1/2} \approx 103$) | $8d^2$ / $6d^2$ | **$1.00$** (alle Zufallsstartwerte) | – | $T$ sequenzielle Schritte |
