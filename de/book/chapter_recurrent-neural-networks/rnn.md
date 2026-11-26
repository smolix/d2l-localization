<!--en:2e3dfeff-->
# Rekurrente neuronale Netze

<!--en:9010c297-->
Die häufigkeitsbasierten Sprachmodelle (engl. count-based language models) aus :numref:`sec_language-model` sagen jedes Token anhand eines kurzen Fensters unmittelbar vorausgehender Token vorher. Bei einem Trigrammmodell hängt die Vorhersage nur von den beiden vorangegangenen Token ab. Was im Dokument weiter zurückliegt, hat keinen Einfluss. Ein größeres Fenster stößt schnell an Grenzen. Eine Häufigkeitstabelle für 20 BPE-Token müsste $1024^{20} \approx
10^{60}$ Kontexte erfassen. Selbst ein neuronales Netz, das in einem festen Fenster $m$ Tokeneinbettungen verkettet und verarbeitet :cite:`Bengio.Ducharme.Vincent.ea.2003`, sieht kein Token, das $m + 1$ Schritte zurückliegt. Zugleich wächst seine Parameteranzahl linear mit $m$.

<!--en:c144152d-->
In :numref:`sec_sequence` haben wir einen grundlegend anderen Ansatz kennengelernt. Ein *latentes* autoregressives Modell verdichtet die gesamte Vorgeschichte zu einem Vektor fester Größe, dem *versteckten Zustand* (engl. hidden state) $\mathbf{h}_{t-1}$. Aus dieser Zusammenfassung sagt es das nächste Token vorher:

<!--en:41496a8b-->
Bei dieser Rekursion bleibt der Rechenaufwand pro Schritt konstant, unabhängig von der Länge des Präfixes. Ein *rekurrentes neuronales Netz* (engl. recurrent neural network, RNN) parametrisiert die Übergangsfunktion $g$ durch eine Schicht, die in allen Zeitschritten dieselben Gewichte verwendet :cite:`elman1990finding`. (Der *versteckte Zustand* wird von Zeitschritt zu Zeitschritt weitergegeben. Eine *versteckte Schicht* eines Feedforward-Netzes liegt innerhalb eines einzelnen Schritts zwischen Eingabe und Ausgabe.) In diesem Abschnitt formulieren wir das RNN-Sprachmodell und bestimmen seine Parameteranzahl und seinen Rechenaufwand. Wir implementieren es für *The Time Machine* sowohl von Grund auf als auch mit den vorgefertigten Schichten der Frameworks.

<!--en:f018dd4c-->
## Rekurrente neuronale Netze mit versteckten Zuständen

<!--en:12c00de7-->
Ein RNN lässt sich am leichtesten verstehen, wenn wir einem MLP mit einer einzigen versteckten Schicht (:numref:`sec_mlp`) eine rekurrente Verbindung hinzufügen. Betrachten wir einen Minibatch aus $b$ Beispielen mit je $d$ Merkmalen, dargestellt durch $\mathbf{X} \in \mathbb{R}^{b \times d}$. (In diesem Kapitel steht $b$ für die Batchgröße, weil $n$ die Ordnung eines $n$-Gramms bezeichnet.) Eine gewöhnliche versteckte Schicht mit der Aktivierungsfunktion $\phi$ berechnet

<!--en:cf3f1e54-->
mit $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$ und $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$. Darauf folgt die Ausgabeprojektion (engl. output projection) $\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} +
\mathbf{b}_\textrm{q}$ auf $q$ Logits.

<!--en:84910a69-->
Nehmen wir nun an, wir erhalten eine Sequenz von Minibatches $\mathbf{X}_t \in \mathbb{R}^{b \times d}$, indiziert durch den Zeitschritt $t$. Damit die versteckte Darstellung Informationen aus früheren Schritten behält, führen wir den vorherigen versteckten Zustand $\mathbf{H}_{t-1} \in \mathbb{R}^{b \times h}$ über eine rekurrente Gewichtsmatrix $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ in die Schicht zurück:

<!--en:fffc2b34-->
In jedem Schritt verwenden wir dieselben fünf Parametertensoren ($\mathbf{W}_{\textrm{xh}}$, $\mathbf{W}_{\textrm{hh}}$, $\mathbf{b}_\textrm{h}$ in der rekurrenten Schicht und $\mathbf{W}_{\textrm{hq}}$, $\mathbf{b}_\textrm{q}$ in der Ausgabeschicht). Wir beginnen mit $\mathbf{H}_0 = \mathbf{0}$. Damit ist die Parameteranzahl völlig unabhängig von der Sequenzlänge (:numref:`fig_rnn`). Entrollen wir ein RNN über $T$ Schritte, erhalten wir ein Feedforward-Netz mit $T$ Schichten, die alle dieselben Gewichte verwenden.

<!--en:f8fdb8bc-->
![Ein RNN mit einem versteckten Zustand, entrollt über drei aufeinanderfolgende Zeitschritte. Jeder Schritt verwendet dieselben Gewichte.](../img/mdl-rnn-unfolded.svg)

<!--en:fbe99ae0-->
Die beiden Matrixprodukte in :eqref:`rnn_h_with_state` lassen sich als ein einziges Produkt schreiben. Dazu verketten wir die Eingaben horizontal und ordnen die Gewichtsmatrizen vertikal übereinander an:

<!--en:cc9a3656-->
Bibliotheken berechnen die Produkte dennoch getrennt. Beim Training sind alle Eingaben $\mathbf{X}_1, \ldots, \mathbf{X}_T$ im Voraus bekannt. Daher lässt sich $\mathbf{X}_t \mathbf{W}_{\textrm{xh}}$ für alle $T$ Schritte gemeinsam außerhalb der sequenziellen Schleife über die Zeit berechnen (Übung :exref:`ex_rnn-input-product`).

<!--en:6077e457-->
Mit einem rekurrenten Zustand lassen sich außerdem Aufgaben lösen, an denen *jedes* Modell mit festem Fenster scheitert. Ein Beispiel ist die Parität eines Bitstroms: In jedem Schritt soll das Modell angeben, ob die Anzahl der bisher gelesenen Einsen gerade oder ungerade ist. Kippt man ein Bit vor einem beliebigen festen Fenster der Länge $m$, kehrt sich die tatsächliche Parität um, obwohl die letzten $m$ Bits gleich bleiben. Ein rekurrenter Zustand mit zwei Einheiten kann die Parität dagegen über beliebige Sequenzlängen verfolgen.

<!--en:b147e31d-->
### Konstanter Speicherbedarf pro Schritt

<!--en:429d7ff0-->
Um $\mathbf{H}_t$ zu berechnen, benötigen wir nur $\mathbf{H}_{t-1}$ und $\mathbf{X}_t$. Speicherbedarf und Rechenaufwand pro Schritt bleiben also unabhängig von der Sequenzlänge konstant. Bei $h = 128$ versteckten Einheiten vom Typ `float32` belegt der Zustand einer Sequenz nur 512 Byte, ob das Modell zehn Token oder eine Million gelesen hat. Dieser feste Speicherplatz ist zugleich ein *verlustbehafteter* Flaschenhals. Jedes Detail, das vom Präfix erhalten bleibt, muss in diese 512 Byte passen. *Attention* (:numref:`chap_attention`) hingegen behält die Darstellungen aller bisherigen Token, wobei der Aufwand mit der Sequenzlänge wächst.

<!--en:4279b8d5-->
### RNN-Sprachmodelle und Teacher-Forcing

<!--en:3389c387-->
Ein RNN-Sprachmodell liest in jedem Schritt eine Token-ID. Für das nächste Token sagt es eine Wahrscheinlichkeitsverteilung über alle $q = |\mathcal{V}|$ Vokabulareinträge vorher (:numref:`fig_rnn_train`). Im Schritt $t$ wählen die $b$ Token-IDs des Minibatchs die entsprechenden Zeilen einer erlernten Einbettungstabelle $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ aus. So erhalten wir $\mathbf{X}_t \in \mathbb{R}^{b \times d}$. Mathematisch entspricht das der Multiplikation einer One-Hot-Matrix der Form $b \times |\mathcal{V}|$ mit $\mathbf{E}$, ohne die Nullen speichern oder mit ihnen multiplizieren zu müssen. Die rekurrente Schicht aktualisiert $\mathbf{H}_t$. Dabei definiert $\textrm{softmax}(\mathbf{O}_t)$ die bedingte Wahrscheinlichkeit $P(x_{t+1} \mid x_{\leq t})$ in der Zerlegung nach der Kettenregel $P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t})$ (:numref:`sec_language-model`).

<!--en:b0e9b03b-->
![Ein RNN-Sprachmodell beim Training mit Teacher-Forcing auf BPE-Token aus *The Time Machine*. Die Zielsequenz ist gegenüber der Eingabe um ein Token vorversetzt. So lernt das Modell in jedem Schritt, das nächste Token vorherzusagen.](../img/mdl-rnn-lm-shift.svg)

<!--en:203596c9-->
Wie verteilen sich Parameter und FLOPs? Für unser BPE-Vokabular mit $|\mathcal{V}| = 1{,}027$ IDs (1024 BPE-Token und drei Sondertoken), die Einbettungsbreite $d = 64$ und die Größe $h = 128$ des versteckten Zustands ergibt sich:

<!--en:807410e2-->
| Bestandteil | Form | Parameter |

<!--en:9e14a303-->
| Einbettungstabelle $\mathbf{E}$ | $1{,}027 \times 64$ | 65 728 |

<!--en:d1984a18-->
| Eingabegewicht $\mathbf{W}_\textrm{xh}$ | $64 \times 128$ | 8192 |

<!--en:85ba896f-->
| Rekurrentes Gewicht $\mathbf{W}_\textrm{hh}$ | $128 \times 128$ | 16 384 |

<!--en:2d30135b-->
| Bias der versteckten Schicht $\mathbf{b}_\textrm{h}$ | $128$ | 128 |

<!--en:ce6928fb-->
| Ausgabegewicht $\mathbf{W}_\textrm{hq}$ | $128 \times 1{,}027$ | 131 456 |

<!--en:6f1afec4-->
| Bias der Ausgabeschicht $\mathbf{b}_\textrm{q}$ | $1{,}027$ | 1027 |

<!--en:d15b68cc-->
| Gesamt | | 222 915 |

<!--en:0b77848c-->
Auf die rekurrente Schicht entfallen nur 24 704 Parameter (11% des Modells) und $24{,}576$ Multiplikationsadditionen pro Token. Die Einbettungs- und Ausgabeschicht, deren Größe vom Vokabular abhängt, enthalten dagegen 89% der Parameter. Allein die Ausgabeprojektion benötigt $128 \cdot 1{,}027 = 131{,}456$ Multiplikationsadditionen.

<!--en:7d5acca7-->
Beim Training erhält das Modell in jedem Schritt das *Ground-Truth*-Token aus dem Korpus, unabhängig von seiner Vorhersage im vorigen Schritt. Dieses Verfahren heißt *Teacher-Forcing*. Da alle Ground-Truth-Eingaben eines Fensters im Voraus bekannt sind, lässt sich der Kreuzentropieverlust (engl. cross-entropy loss) des gesamten Fensters in einem Vorwärtsdurchlauf berechnen. Dieser Verlust ist der Logarithmus der Perplexität. Bei der Generierung dient dagegen jedes durch Sampling oder gieriges Dekodieren ausgewählte Token als nächste Eingabe. So wirken sich eigene Fehler auf die rekursive Vorhersage aus, wie bei unserer Mehrschrittvorhersage der Sinuskurve in :numref:`sec_sequence`.

<!--en:6e98ae08-->
## Ein RNN-Sprachmodell von Grund auf

<!--en:8dfc2025-->
Wir implementieren nun die drei Bestandteile dieses Sprachmodells von Grund auf: die rekurrente Zelle, das Nachschlagen der Einbettungen und die Ausgabeschicht.

<!--en:a5af1e41-->
### Die rekurrente Zelle

<!--en:d8506adb-->
Die Klasse `RNNScratch` speichert die Parameter aus :eqref:`rnn_h_with_state` mit $\phi = \tanh$. Wir ziehen $\mathbf{W}_{\textrm{xh}}$ und $\mathbf{W}_{\textrm{hh}}$ aus $\mathcal{N}(0, \sigma^2)$ mit $\sigma = 0.01$ und initialisieren $\mathbf{b}_\textrm{h}$ mit null. Bei dieser Wahl beträgt der Spektralradius von $\mathbf{W}_{\textrm{hh}}$ etwa $\sigma\sqrt{h} \approx 0.11$. In einer untrainierten Zelle klingt der Einfluss des vorherigen Zustands daher schnell ab. Wo größere rekurrente Gewichte nötig sind, können sie beim Training wachsen.

<!--en:cc37ae95-->
Jede Gewichtsmatrix erhält ein eigenes Objekt zur Initialisierung. Ein Keras-Initialisierer ohne vorgegebenen Startwert für den Zufallsgenerator gibt bei jedem Aufruf dieselben Werte zurück. Mit einem gemeinsamen Initialisierer wären die ersten Zeilen von $\mathbf{W}_{\textrm{hh}}$ daher eine Kopie von $\mathbf{W}_{\textrm{xh}}$.

<!--en:5dff0e62-->
Der Vorwärtsdurchlauf iteriert über die erste Achse von `inputs` mit der Form (`num_steps`, `batch_size`, `num_inputs`), also über die Zeit. Ist kein versteckter Zustand vorgegeben, beginnt er mit Nullen. Er gibt die Liste der versteckten Zustände aller Schritte zusammen mit dem Endzustand zurück.

<!--en:243cb09c-->
Die MXNet-Version liest die Parameterwerte mit `data()` aus, sowohl hier als auch im folgenden Sprachmodell.

<!--en:e8fdd30b-->
Beim Kompilieren des Trainingsschritts wird die Python-Schleife entrollt. Dabei entsteht für jeden Zeitschritt eine Kopie der Zelle, sodass die Kompilierzeit mit der Schrittzahl wächst. Die kompakte Implementierung am Ende dieses Abschnitts verwendet `nnx.RNN`. Damit läuft die Zelle unter `jax.lax.scan` und wird nur einmal kompiliert.

<!--en:90fb5dcf-->
Um die Dimensionen der rekurrenten Zelle zu prüfen, führen wir zunächst einen Vorwärtsdurchlauf von `RNNScratch` mit einem Test-Minibatch `X` der Form `(num_steps, batch_size, num_inputs)` aus.

<!--en:0690ee4a-->
Wir definieren die Hilfsfunktion `check_shape`, die wir auch in späteren Kapiteln verwenden. Damit prüfen wir, ob das Stapeln von `outputs` die Form `(num_steps, batch_size, num_hiddens)` ergibt und ob `state` die Form `(batch_size, num_hiddens)` hat.

<!--en:6cc5196a-->
### Einbettungs- und Ausgabeschicht

<!--en:4bf033dd-->
`RNNLMScratch` ergänzt eine rekurrente Zelle `rnn` um eine Einbettungstabelle `E` und eine Ausgabeprojektion (`W_hq`, `b_q`). Für Einbettungen und versteckte Zustände lassen sich zudem Dropout-Raten angeben. Wir initialisieren die Zeilen der Einbettungstabelle aus $\mathcal{N}(0, 1)$, passend zur Einheitsskala der One-Hot-Eingaben. Für die Ausgabegewichte verwenden wir die Skala $\sigma = 0.01$ und setzen den Bias auf null. Dadurch erzeugt ein untrainiertes Modell Logits nahe null, und seine anfängliche Perplexität liegt nahe bei $|\mathcal{V}|$. Von `d2l.Classifier` erben wir den mittleren Kreuzentropieverlust. Wir überschreiben `validation_step`, um diesen Verlust direkt zu protokollieren. Seine Exponentialfunktion liefert die Validierungsperplexität.

<!--en:3dc8aab1-->
Der Datenlader liefert Minibatches der Form (`batch_size`, `num_steps`). Daher transponiert `embedding` die Token-IDs vor dem Zugriff auf `E`. Das Ergebnis ist ein Tensor der Form (`num_steps`, `batch_size`, `num_inputs`) mit der Zeit als erster Achse.

<!--en:53e82fb3-->
`output_layer` stapelt die Zustände der einzelnen Schritte zu einem Tensor der Form (`batch_size`, `num_steps`, `num_hiddens`) und berechnet die gemeinsame Ausgabeprojektion in einem einzigen gebündelten Matrixprodukt. `forward` führt nacheinander das Nachschlagen der Einbettungen, das Dropout, die rekurrente Zelle und die Ausgabeprojektion aus.

<!--en:5a7fd894-->
Wir übergeben einen Test-Minibatch mit Tokenindizes der Form `(batch_size, num_steps)` an `RNNLMScratch`. Damit bestätigen wir, dass der zurückgegebene Tensor `outputs` der Logits die Form `(batch_size, num_steps, vocab_size)` hat.

<!--en:603cdb6a-->
## Gradienten-Clipping

<!--en:22d48032-->
Ein entrolltes RNN wendet die Jacobi-Matrix desselben Zustandsübergangs über $T$ Schritte an. Daher können Gradienten mit $T$ exponentiell schrumpfen oder wachsen (:numref:`sec_bptt`). Eine plötzliche Gradientenspitze verschiebt die Parameter um den euklidischen Abstand $\eta\|\mathbf{g}\|$. Der Schritt führt aus dem lokalen Bereich hinaus, in dem die Taylor-Näherung erster Ordnung gilt, und kann Dutzende von Aktualisierungen auf einmal zunichtemachen. Statt $\eta$ generell zu verkleinern, begrenzt *Gradienten-Clipping* die globale Gradientennorm auf einen Schwellenwert $\theta$. Dazu berechnen wir die Norm gemeinsam über alle Modellparameter und projizieren $\mathbf{g}$ auf die Kugel mit Radius $\theta$:

<!--en:39f3c49c-->
Ist der Gradient länger als $\theta$, wird er unter Beibehaltung seiner Richtung auf die Länge $\theta$ verkürzt. Ein kürzerer Gradient bleibt unverändert (:numref:`fig_grad-clipping`).

<!--en:4b7d5c77-->
![Gradienten-Clipping als Projektion auf die Kugel mit Radius $\theta$. Der lange Gradient $\mathbf{g}$ wird entlang seiner Richtung auf die Länge $\theta$ verkürzt. Der Gradient $\mathbf{g}'$ liegt im Inneren der Kugel und bleibt unverändert.](../img/mdl-rnn-grad-clipping.svg)

<!--en:456e09f6-->
Ist `gradient_clip_val` positiv, wendet `d2l.Trainer` (:numref:`sec_oo-design`) die Projektion :eqref:`eq_grad_clipping` unmittelbar nach der Backpropagation und vor dem Schritt des Optimierers an.

<!--en:3b9ad0e3-->
Die folgende Methode skaliert den Gradienten jedes Parameters direkt an seinem Speicherort.

<!--en:e02a2032-->
Die folgende Methode gibt die gegebenenfalls skalierten Gradienten zurück. Der kompilierte Trainingsschritt übergibt sie dem Optimierer.

<!--en:d1945d90-->
Der JAX-Trainer aktualisiert die Parameter in seinem kompilierten Trainingsschritt mit einer optax-Transformation. Deshalb findet auch das Clipping dort statt. `prepare_optimizer` macht aus `clip_gradients` eine optax-Transformation und schaltet sie dem Optimierer des Modells vor.

<!--en:e1f67b84-->
## Training und Auswertung

<!--en:cfd7668e-->
`d2l.TimeMachine` (:numref:`sec_language-model`) kodiert die 65 741 BPE-Token der Novelle und liefert Minibatches mit je $T = 32$ überlappenden Fenstern. Davon dienen 50 000 Fenster zum Training und 5000 zur Validierung. Dazwischen liegen 32 ausgelassene Fenster, damit kein Validierungstoken als Trainingsziel vorkommt. Da jedes Fenster mit $\mathbf{H}_0 = \mathbf{0}$ beginnt, ist die Backpropagation auf 32 Schritte *verkürzt* (:numref:`sec_bptt`).

<!--en:f721ac6e-->
### Ein einfaches Modell und die Rolle des Gradienten-Clippings

<!--en:c57174fa-->
Zunächst erzeugen wir eine Instanz unseres Basismodells mit 222 915 Parametern ($d = 64$, $h = 128$, ohne Dropout). Für den stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD) verwenden wir die Lernrate $\eta = 4$.

<!--en:ee526c68-->
Vor dem Training speichern wir eine NumPy-Kopie der zufällig initialisierten Einbettungsmatrix `model.E` in `E_init`. So können wir messen, wie weit SGD ihre Zeilen verschiebt.

<!--en:5d567f25-->
Als Nächstes bestimmen wir die anfängliche Validierungsperplexität `ppl_init` des untrainierten Modells. Dann trainieren wir `model` über 10 Epochen mit Gradienten-Clipping beim Schwellenwert `gradient_clip_val=1`.

<!--en:949ad032-->
Wenden wir die Exponentialfunktion auf den abschließenden Kreuzentropieverlust `trainer.history['val_loss'][-1]` der Validierung an, erhalten wir die Validierungsperplexität `ppl_plain` des trainierten Basismodells.

<!--en:24a5e874-->
Anfangs liegt die Validierungsperplexität nahe bei $|\mathcal{V}| = 1{,}027$. Nach zehn Epochen erreicht das Modell mit Gradienten-Clipping eine Validierungsperplexität von etwa 85 bis 100, obwohl der Verlust in der ersten Epoche oft deutlich über den Anfangswert springt. Die Trainingsperplexität sinkt auf weniger als die Hälfte der erreichten Validierungsperplexität, da bei 222 915 Parametern und 50 000 Trainingstoken die Überanpassung (engl. overfitting) einsetzt.

<!--en:ec1de8a8-->
Um zu sehen, warum Gradienten-Clipping bei $\eta = 4$ entscheidend ist, definieren wir die Hilfsfunktion `unclipped_model()`. Sie erzeugt eine neue Instanz des Basismodells mit genau derselben Architektur und Lernrate.

<!--en:f550ae6b-->
Wir trainieren nun drei unabhängige Instanzen ohne festen Zufallsstartwert und ohne Gradienten-Clipping. Für jeden Lauf zeichnen wir die Validierungsperplexität nach jeder Epoche in `ppl_runs` auf und geben den Höchstwert und den Endwert aus.

<!--en:f22ae1c6-->
Tragen wir die drei Verläufe ohne Clipping aus `ppl_runs` zusammen mit dem Verlauf `ppl_clipped` des Basismodells mit Clipping auf einer logarithmischen Achse auf, wird die Instabilität deutlich sichtbar.

<!--en:0a3bc933-->
Ohne Clipping führen Gradientenspitzen zu heftigen Sprüngen der Validierungsperplexität, mitunter weit über den Wert des untrainierten Basismodells von 1027 hinaus. In den meisten Läufen ist die Perplexität am Ende mehr als doppelt so hoch wie beim Modell mit Clipping. Um dessen Validierungsperplexität in Bit pro Byte umzurechnen (:numref:`subsec_bits-per-byte`), teilen wir $\log_2 \textrm{ppl}$ durch die mittlere Anzahl von UTF-8-Bytes pro BPE-Token ($\approx 2.8$):

<!--en:f07d3772-->
### Einbettungen bei SGD und ein breiteres Modell mit Dropout

<!--en:a1eb4ecc-->
Wie weit hat sich jede Zeile von $\mathbf{E}$ im Laufe der zehn SGD-Epochen verschoben? Das verrät uns, wo ein zweiter Flaschenhals liegt. Zunächst lesen wir die trainierte Einbettungstabelle in ein NumPy-Array `E_final` ein.

<!--en:1fa5b557-->
Wir berechnen für jede Zeile des Vokabulars die relative euklidische Verschiebung `change` zwischen `E_final` und `E_init`. Daran sehen wir, wie wenig sich die einzelnen Tokenvektoren bewegt haben.

<!--en:4e7ea808-->
Zeile $i$ erhält nur von den Positionen einen Gradientenbeitrag, an denen Token $i$ vorkommt. Der Verlust mittelt jedoch über alle $1{,}024 \times 32 = 32{,}768$ Token eines Minibatchs. Bei einer einheitlichen SGD-Lernrate erhalten seltene und mäßig häufige Token deshalb sehr kleine Gradienten. Dadurch bleibt $\mathbf{E}$ nahezu unverändert. Adaptive Optimierer wie Adam (:numref:`sec_adam`) passen die Schrittweiten für jede Koordinate an und verschieben die Einbettungen dadurch wesentlich weiter. Das untersuchen wir in den Übungen. Wir können das Problem auch direkt mit SGD angehen. Erhöhen wir die Einbettungsbreite von $d = 64$ auf $d = 512$, erhält die dicht aktualisierte Eingabematrix $\mathbf{W}_{\textrm{xh}}$ eine wesentlich reichhaltigere Basis zufälliger Merkmale. Zusätzlich begrenzt Dropout die Überanpassung (0.1 für Einbettungen, 0.5 für versteckte Zustände).

<!--en:46865f07-->
Wir trainieren dieses breitere, regularisierte Modell über 10 Epochen mit derselben Lernrate `lr=4` und demselben Clipping-Schwellenwert `gradient_clip_val=1`.

<!--en:58e26265-->
Aus dem abschließenden Validierungsverlust berechnen wir sowohl die Validierungsperplexität `ppl_scratch` als auch die entsprechende Kompressionsrate in Bit pro Byte.

<!--en:dc7aad2b-->
Die Validierungsperplexität sinkt von 85–100 auf rund 60. Damit sinkt die Anzahl der Bit pro Byte von 2.3–2.4 auf etwa 2.1.

<!--en:62e01890-->
### Vergleich mit Kneser–Ney und effektive Kontextlänge

<!--en:a77bfe1a-->
Wie schneiden 2.1 Bit pro Byte im Vergleich zur interpolierten Kneser–Ney-Glättung ab (:numref:`subsec_kneser-ney`)? Die $1.9$ Bit pro Byte, die das Kneser–Ney-Trigramm in :numref:`sec_language-model` erreicht hat, sind kein geeigneter Maßstab. Dort bestand der Text nur aus Kleinbuchstaben und Leerzeichen. In diesem Abschnitt modellieren wir dagegen den Originaltext der Novelle mit Großbuchstaben, Satzzeichen und Zeilenumbrüchen. Deshalb passen wir Kneser–Ney-$n$-Gramme ($n = 2, 3, 4$) an genau dieselben 50 000 BPE-Trainingstoken an und werten sie auf `val[32:]` aus. Dabei bewerten wir jedes Token als letztes Ziel seines Validierungsfensters von 32 Token.

<!--en:dff26740-->
Um unser trainiertes RNN auf genau denselben zurückgehaltenen Token `val[32:]` auszuwerten, wählen wir den Slice `[start:stop]` der Validierungsfenster. Ihre letzten Zieltoken decken zusammen `val[32:]` ab.

<!--en:576dfbc0-->
Wir führen `model` im Auswertungsmodus auf `data.X[start:stop]` aus, jeweils in Batches aus 1024 Fenstern. So erhalten wir die Matrix `logp` mit den logarithmierten Wahrscheinlichkeiten der Zieltoken an allen 32 Fensterpositionen.

<!--en:dedd5c8c-->
Das RNN und das Trigramm erreichen nahezu denselben Mittelwert. Treffen sie auch dieselben Vorhersagen? Die letzte Spalte von `logp` bewertet dieselben Token `val[32:]` wie `lp_kn`. Damit können wir beide Modelle Token für Token vergleichen. Wir bewerten auch ihre gleich gewichtete Mischung $0.5\, P_\textrm{RNN} + 0.5\, P_\textrm{KN}$. So kombiniert man üblicherweise ein neuronales mit einem häufigkeitsbasierten Sprachmodell :cite:`Mikolov.Kombrink.Burget.ea.2011`.

<!--en:5cbd35b8-->
Im Mittel liegen die Modelle gleichauf, doch bei einzelnen Token unterscheiden sie sich. Bei etwa der Hälfte der Token weist das RNN dem beobachteten Token die höhere Wahrscheinlichkeit zu, bei der anderen Hälfte das Trigramm. Die Mischung liegt etwa 0.06 Bit pro Byte unter dem Trigramm und schneidet damit besser ab als jedes der beiden Modelle für sich.

<!--en:7b751c7f-->
Eine Mischung kann nur dort gewinnen, wo die Einzelmodelle voneinander abweichen. Die beiden Modelle haben also teilweise Unterschiedliches über den Text gelernt. Umso erstaunlicher ist der Gleichstand. Warum kommt ein rekurrentes Netz mit 32 Token Vorgeschichte auf fast dieselben $\approx 2.1$ Bit pro Byte wie ein Kneser–Ney-Trigramm, das nur zwei Token zurückblickt? Um zu sehen, wie viel Kontext das trainierte RNN tatsächlich nutzt, tragen wir für jede Fensterposition $1, \ldots, 32$ seinen Mittelwert in Bit pro Byte auf den Validierungsdaten auf. Als Vergleich dient das Kneser–Ney-Trigramm.

<!--en:4c72dc90-->
Der Kurvenverlauf erklärt den Gleichstand. Fast der gesamte Gewinn stellt sich mit dem zweiten Kontexttoken ein. Es senkt den Verlust um etwa 0.15, von rund 2.3 auf etwa 2.14 Bit pro Byte. Das dritte Token bringt noch ein bis zwei Hundertstel. Von Position 4 bis 32 verläuft die Kurve flach. Obwohl der versteckte Zustand im Vorwärtsdurchlauf von allen 32 Token abhängen *kann*, nutzt das trainierte Modell nur die letzten zwei oder drei.

<!--en:d949a1e6-->
Warum so wenige? Eine Erklärung liegt in der Optimierung. Der Gradient, durch den die Zelle lernen könnte, ein weit zurückliegendes Token zu nutzen, schrumpft auf dem Rückweg mit jedem Schritt. Das messen wir in :numref:`sec_bptt`. Die andere Erklärung liegt in den Daten. Möglicherweise lässt sich bei 50 000 Trainingstoken aus Token, die mehr als zwei oder drei Positionen zurückliegen, nur wenig über das nächste Token lernen. Die oben ausgegebenen Ergebnisse der Kneser–Ney-Modelle sprechen für die zweite Erklärung. Diese Modelle beruhen auf Häufigkeiten, kommen also ohne Gradienten aus, die verschwinden könnten. Dennoch zeigt sich die Sättigung an derselben Stelle. Das Trigramm gewinnt gegenüber dem Bigramm 0.15 Bit pro Byte, das 4-Gramm gegenüber dem Trigramm nur 0.01. Unser Experiment kann die beiden Erklärungen nicht auseinanderhalten. Dazu müssten wir ein Modell auf denselben Token trainieren, dessen Gradienten auch über große zeitliche Abstände erhalten bleiben. Solche Modelle entwickeln wir in :numref:`chap_modern_rnn`.

<!--en:6aa8a5c8-->
## Textgenerierung

<!--en:b410e663-->
Nach dem Training erzeugt unser Sprachmodell mit `predict` autoregressiv Text. Zunächst verarbeitet es die Token des vorgegebenen Präfixes, um den versteckten Zustand aufzubauen. Dann erzeugt es `num_tokens` weitere Token. Dabei dient jedes gewählte Token als nächste Eingabe. Bei `temperature=0.0` wählt das Modell mit Argmax das wahrscheinlichste Token. Andernfalls zieht es ein Token aus der Verteilung, die die Softmax-Funktion aus den temperaturskalierten Logits berechnet (siehe :numref:`sec_decoding`).

<!--en:d510cc22-->
Testen wir `model.predict` mit gierigem Dekodieren (engl. greedy decoding, `temperature=0.0`). Dazu erzeugen wir eine Fortsetzung von 50 Token für den Prompt `'The Time Traveller'`.

<!--en:d85ce0df-->
Gieriges Dekodieren ist deterministisch. Kehrt der versteckte Zustand zu einer gerade erzeugten Wortfolge zurück, wiederholt sich diese daher endlos. In :numref:`sec_decoding` untersuchen und beheben wir dieses Problem.

<!--en:d52bb9d8-->
## Kompakte Implementierung

<!--en:682f888a-->
Die rekurrenten Schichten der Frameworks fassen den Rekursionsschritt und die Schleife über die Zeit in optimierten hardwarenahen Kerneln zusammen. Das folgende Modul `RNN` verwendet dieselbe Schnittstelle und Initialisierung wie `RNNScratch`. Die Klasse `RNNLM` leitet sich von `RNNLMScratch` ab und setzt die eingebauten Einbettungs- und linearen Schichten ein.

<!--en:609d0f40-->
Die rekurrente Schicht von Gluon verwendet zwei Bias-Vektoren, einen für den Eingabeterm und einen für den rekurrenten Term. Ihre Summe entspricht $\mathbf{b}_\textrm{h}$, und beide beginnen bei null. Die Standardaktivierung in Gluon ist ReLU, daher geben wir tanh ausdrücklich an.

<!--en:ad602df2-->
`nn.RNN` verwendet zwei Bias-Vektoren, einen für den Eingabeterm und einen für den rekurrenten Term. Ihre Summe entspricht $\mathbf{b}_\textrm{h}$, und beide beginnen bei null.

<!--en:db30b25a-->
`SimpleRNN` erwartet Eingaben mit der Batchachse an erster Stelle. Deshalb transponiert `forward` sowohl die Eingabe als auch die Ausgabe. Wie bei `RNNScratch` erhält jede Gewichtsmatrix ein eigenes Objekt zur Initialisierung.

<!--en:33109d39-->
Flax NNX stellt mit `SimpleCell` eine einfache rekurrente Zelle bereit. Sie verarbeitet die Eingabe eines Zeitschritts und einen explizit übergebenen versteckten Zustand. `nnx.RNN` wendet sie nacheinander auf die gesamte Eingabesequenz an. Nur die Eingabeprojektion enthält einen Bias, die rekurrente Projektion kommt ohne ihn aus. Die Zelle hat also wie `RNNScratch` genau einen Bias-Vektor.

<!--en:2a748ce1-->
`nn.Embedding` zieht seine Zeilen standardmäßig aus $\mathcal{N}(0, 1)$. Die Gewichte von `nn.Linear` setzen wir explizit.

<!--en:0cf1b81e-->
Um unsere Implementierung mit Framework-Schichten zu überprüfen, erzeugen wir Instanzen von `RNN` und `RNNLM` mit denselben Hyperparametern ($d = 512$, $h = 128$ und Dropout).

<!--en:814872c1-->
Wir trainieren `RNNLM` nun über 10 Epochen mit `gradient_clip_val=1`. Der zusammengefasste rekurrente Kernel benötigt nur einen Bruchteil der Laufzeit unserer eigenen Implementierung.

<!--en:c5c17e00-->
Zum Schluss geben wir die Validierungsperplexität `ppl_concise` der kompakten Implementierung aus. Dazu kommt eine durch gieriges Dekodieren erzeugte Fortsetzung von 30 Token für `'The Time Traveller'`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3a785703-->
Ein rekurrentes neuronales Netz aktualisiert einen versteckten Zustand fester Größe anhand der aktuellen Eingabe und des vorherigen Zustands. Dazu verwendet es in allen Zeitschritten dieselben Gewichte. Speicherbedarf und Rechenaufwand pro Schritt bleiben konstant, doch bei der Verdichtung der Vorgeschichte zum Zustandsvektor gehen Informationen verloren. Ein RNN-Sprachmodell kombiniert eine Einbettungstabelle, eine rekurrente Zelle mit $\tanh$-Aktivierung und eine Ausgabeprojektion in Vokabulargröße. Auf Einbettungstabelle und Ausgabeprojektion entfallen 89% der Parameter unseres Basismodells. Beim Training mit Teacher-Forcing minimieren wir die Kreuzentropie pro Token. Globales Gradienten-Clipping (:eqref:`eq_grad_clipping`) verhindert, dass explodierende Gradienten das Training bei hohen Lernraten destabilisieren. Auf *The Time Machine* erreicht das mit SGD trainierte RNN dank einer breiteren Einbettung mit Dropout $\approx 2.1$ Bit pro Byte. Damit liegt es gleichauf mit einem interpolierten Kneser–Ney-Trigramm. Das trainierte einfache RNN nutzt effektiv nur die letzten zwei oder drei Kontexttoken. In :numref:`sec_bptt` messen wir, wie weit seine Gradienten reichen.

<!--en:2294243a-->
## Übungen

<!--en:fe189bae-->
1. **Das Eingabeprodukt getrennt berechnen.** Seien $\mathbf{X}_t \in \mathbb{R}^{b \times d}$ die Eingabe und $\mathbf{H}_{t-1} \in \mathbb{R}^{b \times h}$ der Zustand in :eqref:`rnn_h_with_state`.
    1. Beim Training sind alle Eingaben $\mathbf{X}_1, \ldots, \mathbf{X}_T$ eines Fensters bekannt, bevor der erste Zeitschritt berechnet wird. Welches der beiden Produkte $\mathbf{X}_t \mathbf{W}_{\textrm{xh}}$ und $\mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ lässt sich für alle $T$ Schritte in einer einzigen Matrizenmultiplikation berechnen? Welche Form haben seine Faktoren? Welches Produkt muss innerhalb der sequenziellen Schleife über die Zeitschritte bleiben und warum?
    1. Vergleichen Sie den Rechenaufwand, der pro Schritt in der Schleife verbleibt, mit dem Aufwand pro Schritt bei der verketteten Form $[\mathbf{X}_t, \mathbf{H}_{t-1}]$. Bei der Generierung ist die Eingabe im Schritt $t$ das im Schritt $t-1$ erzeugte Token. Bleibt der Vorteil der getrennten Berechnung dabei erhalten?
   :exlabel:`ex_rnn-input-product`

<!--en:f74cda98-->
1. **Exposure-Bias.** Beim Teacher-Forcing (:numref:`subsec_teacher-forcing`) hängt jeder Trainingsschritt vom tatsächlichen Präfix ab, bei der Generierung dagegen von den bisherigen Ausgaben des Modells. Diese Diskrepanz heißt *Exposure-Bias* :cite:`Ranzato.Chopra.Auli.ea.2016`.
    1. Nehmen Sie an, dass bei der Generierung jedes Token aus der vom Modell vorhergesagten Verteilung für das nächste Token gezogen wird. Solange das Präfix korrekt ist, weist das Modell dem korrekten nächsten Token die Wahrscheinlichkeit $1 - \epsilon$ zu. Mit welcher Wahrscheinlichkeit enthalten $T$ generierte Token keinen Fehler? Berechnen Sie den Wert für $\epsilon = 0.01$ und $T = 100$.
    1. Beim *Scheduled Sampling* :cite:`Bengio.Vinyals.Jaitly.ea.2015` erhält das Modell bereits im Training seine eigenen Ausgaben als Eingabe. In jedem Schritt wird mit einer gewissen Wahrscheinlichkeit das tatsächliche Token durch ein Token ersetzt, das aus der Vorhersage des vorigen Schritts gezogen wurde. Welche in :numref:`subsec_teacher-forcing` genutzte Eigenschaft des Teacher-Forcings geht dadurch verloren? Wie verändert das die Berechnung für ein Trainingsfenster?

<!--en:33a7f09f-->
1. **Ein Fenster innerhalb einer Rekursion.** Das neuronale $n$-Gramm liest ein festes Fenster aus den letzten $m$ Eingabeeinbettungen. Wählen Sie $\phi$ als Identität und setzen Sie $\mathbf{b}_\textrm{h} = \mathbf{0}$ in :eqref:`rnn_h_with_state`. Der Zustand soll $h = md$ Einheiten haben.
    1. Wählen Sie $\mathbf{W}_{\textrm{xh}}$ und $\mathbf{W}_{\textrm{hh}}$ so, dass der Zustand, ausgehend von $\mathbf{H}_0 = \mathbf{0}$, die Verkettung $\mathbf{H}_t = [\mathbf{X}_t, \mathbf{X}_{t-1}, \ldots, \mathbf{X}_{t-m+1}]$ für jedes $t \geq m$ ist.
    1. Zeigen Sie, dass $\mathbf{W}_{\textrm{hh}}^m = \mathbf{0}$ gilt. Erklären Sie, warum damit keine Eingabe, die mehr als $m$ Schritte zurückliegt, Einfluss auf $\mathbf{H}_t$ hat.
    1. Wie viele Zustandseinheiten erfordert diese Konstruktion für $d = 256$ und $m = 20$? Erklären Sie, was ein RNN mit nur 256 Einheiten im Zustand tun muss, wenn es das Fenster nicht vollständig speichern kann.

<!--en:39513a7a-->
1. [code] **Nachschlagen von Einbettungen als Matrixprodukt.** Sei $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ eine Einbettungstabelle, und enthalte $\mathbf{x}$ die Token-IDs eines Minibatchs aus $b$ Token.
    1. Berechnen Sie die Eingaben auf zwei Arten. Wählen Sie zunächst aus $\mathbf{E}$ die durch $\mathbf{x}$ indizierten Zeilen aus. Multiplizieren Sie dann die One-Hot-Matrix der Form $b \times |\mathcal{V}|$ für $\mathbf{x}$ mit $\mathbf{E}$. Prüfen Sie, ob die Ergebnisse übereinstimmen. Messen Sie die Laufzeiten beider Berechnungen für $|\mathcal{V}| \in \{10^3, 10^4, 10^5\}$ bei $d = b = 256$. Geben Sie an, wie die jeweilige Laufzeit mit $|\mathcal{V}|$ wächst.
    1. Leiten Sie die Summe aller Einträge der ausgewählten Zeilen durch automatisches Differenzieren nach $\mathbf{E}$ ab. Welche Zeilen des Gradienten sind ungleich null? Was enthält eine Zeile, deren Token mehrfach im Minibatch vorkommt? Schätzen Sie für $|\mathcal{V}| = 10^4$ und $b = 256$ gleichverteilt gezogene IDs den Anteil der Zeilen mit einem Gradienten ungleich null.

<!--en:4d502de9-->
1. **Parität jenseits jedes festen Fensters.** ● Sei $x_1, x_2, \ldots$ eine Bitfolge. Die gewünschte Ausgabe im Schritt $t$ sei die Parität von $x_1 + \cdots + x_t$.
    1. Zeigen Sie, dass kein Modell, das nur die letzten $m$ Eingaben liest, in jedem Schritt einer Sequenz der Länge größer als $m$ die richtige Ausgabe erzeugen kann. Dies soll für jedes endliche $m$ gelten.
    1. Sei $\phi$ eine Schwellwertaktivierung. Konstruieren Sie ein RNN der Form :eqref:`rnn_h_with_state` mit zwei versteckten Einheiten, das bei jeder Sequenzlänge in jedem Schritt die Parität ausgibt. Geben Sie die Gewichte an und zeigen Sie, dass eine versteckte Einheit nicht ausreicht.
    1. Wie viele Bit an Information über $x_1, \ldots, x_t$ muss der versteckte Zustand bewahren, damit alle späteren Ausgaben korrekt sind? Setzen Sie das in Beziehung zur Beschreibung des Zustands als verlustbehaftete Zusammenfassung fester Größe in :numref:`subsec_rnn-constant-memory`.

<!--en:b449ffe5-->
1. [code] **Ablationsstudie zum Gradienten-Clipping.** `d2l.Trainer` wendet die Projektion :eqref:`eq_grad_clipping` an, wenn `gradient_clip_val` positiv ist.
    1. Zeichnen Sie bei jeder Aktualisierung in der ersten Epoche die Gradientennorm vor dem Clipping auf und stellen Sie den Verlauf grafisch dar. Wie oft greift das Clipping ein und wann?
    1. Bei `lr=4` traten in den meisten Läufen dieses Abschnitts ohne Clipping Sprünge auf. Halbieren Sie die Lernrate so lange, bis in drei aufeinanderfolgenden Trainingsläufen ohne Clipping keine Sprünge der Validierungsperplexität mehr auftreten. Wie hoch ist ihre Validierungsperplexität nach zehn Epochen im Vergleich zum Lauf mit Clipping bei `lr=4`? Was ermöglicht das Clipping also?
    1. Ersetzen Sie $\tanh$ durch ReLU in `RNNScratch` und trainieren Sie erneut, mit und ohne Clipping. Ist Clipping weiterhin nötig? Erklären Sie den Unterschied zur $\tanh$-Zelle anhand des Wertebereichs des versteckten Zustands.

<!--en:cc145ae6-->
1. [code] **Nutzbare Vorgeschichte.** Jedes Trainingsfenster beginnt mit dem Nullzustand. Das Modell lernt also nie, mehr als `num_steps` Kontexttoken zu nutzen. Der Verlust nach Fensterposition hat gezeigt, dass es tatsächlich viel weniger nutzt. `predict` hingegen führt den Zustand über das ganze Präfix und jedes erzeugte Token hinweg fort. Werten Sie das trainierte Modell auf 512 aufeinanderfolgenden Validierungstoken auf zwei Arten aus. Setzen Sie den Zustand einmal wie im Training alle 32 Token auf null zurück und führen Sie ihn einmal über alle 512 Token fort. Vergleichen Sie die mittlere Kreuzentropie beider Varianten. Hilft der fortgeführte Zustand, schadet er oder macht er keinen Unterschied? Was sagt das über die Zustände aus, die das Modell jenseits seiner Trainingsfenster erreicht?

<!--en:afee1e6d-->
1. [code] **Adam.** Trainieren Sie das Modell mit der breiteren Einbettung mit dem Adam-Optimierer aus :numref:`sec_adam` anstelle von SGD. Leiten Sie dazu eine Unterklasse von `RNNLMScratch` ab. Überschreiben Sie `configure_optimizers` so, dass die Methode Adam mit der Lernrate 0.003 zurückgibt, und belassen Sie den Clipping-Schwellenwert bei 1.
    1. Bewerten Sie mit `emb_dropout=0.2` die Token `val[32:]` wie im obigen Vergleich. Vergleichen Sie die Bitzahl pro Byte mit den Werten des SGD-Modells, des Trigramms und des 4-Gramms nach Kneser–Ney sowie der Mischung aus RNN und Trigramm bei $\lambda = 0.5$.
    1. Zeichnen Sie nach jeder Epoche den Validierungsverlust auf, einmal mit `emb_dropout=0.1` und einmal mit 0.2. In welcher Epoche ist er am niedrigsten und was passiert danach?
    1. Messen Sie erneut, wie weit sich die Zeilen der Einbettungstabelle beim Training verschieben. Warum verschiebt Adam sie weiter als SGD?

<!--en:03c85530-->
1. [code] **Vokabulargröße und Bit pro Byte.** ● Trainieren Sie das Modell mit der breiteren Einbettung erneut. Verwenden Sie für `vocab_size` die Werte 512, 1024, 2048 und 4096 in `d2l.TimeMachine`. Die Anzahl der Fenster hängt von der Vokabulargröße ab. Teilen Sie die Daten daher nach Anteilen auf. Erstellen Sie jeden Datensatz zunächst, um `n = len(data.X)` abzulesen, und erstellen Sie ihn dann erneut mit `num_train=int(0.8 * n)` und `num_val=n - int(0.8 * n)`. (Wenn Sie `data.num_train` erst nachträglich setzen, überspringen Sie den Schritt, der Validierungstoken von den Trainingszielen ausschließt.) So nutzen alle Modelle für Training und Validierung ungefähr denselben Textabschnitt. Da ihre Perplexitäten pro Token nicht vergleichbar sind, vergleichen Sie die Modelle anhand der Bitzahl pro Byte.
    1. Bei welcher Vokabulargröße ist die Bitzahl pro Byte am niedrigsten?
    1. Wie ändern sich die Parameteranzahl der Einbettungstabelle und der Ausgabeschicht sowie die Korpuslänge in Token mit der Vokabulargröße? Welcher dieser Effekte erklärt den gemessenen Trend am besten?

<!--en:3b35e499-->
1. [code] **Generalisierung über Buchgrenzen hinweg.** ● Leiten Sie eine Unterklasse von `d2l.TimeMachine` ab. Überschreiben Sie `_download` so, dass die Methode den Text eines anderen Romans von H. G. Wells zurückgibt, etwa *The War of the Worlds* (E-Book 36 im Project Gutenberg). Trainieren Sie den Tokenizer und das Modell auf diesem Text.
    1. Werten Sie das Modell auf einem zurückgehaltenen Textabschnitt desselben Romans und auf *The Time Machine* aus. Kodieren Sie beide Texte mit dem Tokenizer des neuen Romans. Geben Sie für beide die Perplexität und die Bitzahl pro Byte an.
    1. Bezeichnen Sie den Überraschungswert pro Token mit $s = \log_2(\textrm{ppl})$ und die Anzahl der Bytes pro Token eines ausgewerteten Texts mit $b$. Die Anzahl der Bit pro Byte ist dann $s / b$. Das Verhältnis der Bitzahl pro Byte für *The Time Machine* (TM) zu der für den zurückgehaltenen Text des neuen Romans (own) lässt sich wie folgt zerlegen:
       $$\frac{s_\textrm{TM}}{s_\textrm{own}} \cdot \frac{b_\textrm{own}}{b_\textrm{TM}}.$$
       Der erste Faktor spiegelt die Vorhersagen des Modells wider, der zweite die Kompression durch den Tokenizer. Messen Sie beide Faktoren. Welcher liegt weiter von eins entfernt?

<!--en:f1f28c08-->
[Diskussion](https://d2l.discourse.group/t/336)

<!--en:f55f9823-->
[Diskussion](https://d2l.discourse.group/t/486)

<!--en:a25c74e6-->
[Diskussion](https://d2l.discourse.group/t/1052)

<!--en:63d2c5d7-->
[Diskussion](https://d2l.discourse.group/t/18014)

<!--en:6096a66f-->
Rekurrente neuronale Netze<br>**Rekursion des versteckten Zustands · Gewichtsteilung · konstanter Speicherbedarf · Teacher-Forcing · Gradienten-Clipping · Vergleich mit Kneser–Ney · Framework-APIs**

<!--en:c90cea75-->
::: {.slide title="Warum neuronale Modelle mit festem Fenster nur begrenzt zurückblicken"}

<!--en:726403da-->
[Motivation und latenter autoregressiver Zustand]{.kicker}

<!--en:9ad13ba0-->
- **Lineares Parameterwachstum und feste Grenze:** Bei der Verkettung von $m$ Einbettungen der Breite $d$ kommen pro zusätzlichem Zeitschritt im Fenster $d \times h$ Gewichte hinzu ($+8{,}192$ bei $d=64, h=128$). Token $t - m - 1$ bleibt außer Sichtweite.

<!--en:a79209c3-->
- **Rekurrenter Zustand mit geteilten Gewichten:** Ein **RNN** verdichtet das Präfix zu einem Vektor fester Größe $\mathbf{h}_t \in \mathbb{R}^h$ und aktualisiert ihn in jedem Schritt mit denselben Gewichten. Parameteranzahl und Rechenaufwand pro Schritt bleiben in $O(1)$, unabhängig von der Länge $T$.

<!--en:1a8e05db-->
::: {.slide title="Von der versteckten Schicht eines MLP zum rekurrenten Zustand"}

<!--en:bac143a7-->
[Formulierung der rekurrenten Schicht]{.kicker}

<!--en:6cfe2fa3-->
- **Minibatch-Konventionen:** Batchgröße $b$, Einbettungsbreite $d$, Größe des versteckten Zustands $h$, Vokabulargröße $q = |\mathcal{V}|$ und Nichtlinearität $\phi = \tanh$.

<!--en:53298d65-->
- **Fünf geteilte Parametertensoren:** $(\mathbf{W}_{\textrm{xh}}, \mathbf{W}_{\textrm{hh}}, \mathbf{b}_{\textrm{h}}, \mathbf{W}_{\textrm{hq}}, \mathbf{b}_{\textrm{q}})$ kommen in jedem Zeitschritt zum Einsatz, ausgehend von $\mathbf{H}_0 = \mathbf{0}$.

<!--en:c1c8cb39-->
::: {.slide title="Ein RNN mit geteilten Gewichten über die Zeit entrollen"}

<!--en:73cb4c60-->
[Zeitliche Entrollung und Fortbestehen des Zustands]{.kicker}

<!--en:6b010e92-->
Über $T$ Zeitschritte entrollt ergibt ein RNN ein Feedforward-Netz mit $T$ Schichten. Alle Schichten verwenden dieselben Gewichte $(\mathbf{W}_{\textrm{xh}}, \mathbf{W}_{\textrm{hh}}, \mathbf{b}_{\textrm{h}})$:

<!--en:4cdc5a39-->
- **Eindimensionale skalare Impulsantwort:** Gegeben sei $h_t = \tanh(x_t + 0.5\,h_{t-1})$ mit $h_0 = 0$ und dem Impuls $(x_1, x_2, x_3) = (1, 0, 0)$:
  - $t = 1$: $h_1 = \tanh(1 + 0) = 0.76$
  - $t = 2$: $h_2 = \tanh(0 + 0.5 \times 0.76) = 0.36$
  - $t = 3$: $h_3 = \tanh(0 + 0.5 \times 0.36) = 0.18$

<!--en:ab60d366-->
- **Gedächtnis des rekurrenten Zustands:** Auch wenn die Eingabe wieder auf $0$ fällt, wirkt der Anfangsimpuls fort. Wie schnell er abklingt, hängt von $\mathbf{W}_{\textrm{hh}}$ ab.

<!--en:49847cd2-->
::: {.slide title="Verkettete Eingaben und gebündelte Eingabeprojektionen"}

<!--en:951915fb-->
[Lineare Algebra und Systemoptimierung]{.kicker}

<!--en:63e0fdcd-->
- **Warum die beiden Produkte getrennt bleiben:** Beim Training mit Teacher-Forcing sind alle $T$ Eingaben $\mathbf{X}_1, \ldots, \mathbf{X}_T$ des Fensters im Voraus bekannt.

<!--en:9f36cd84-->
- **$\mathbf{X}_{1:T}\mathbf{W}_{\textrm{xh}}$ vor der Schleife berechnen:** Frameworks bündeln alle $T$ Eingabeprojektionen in einer einzigen Matrizenmultiplikation (GEMM) der Form $(Tb \times d) \times (d \times h)$ auf der GPU. In der sequenziellen Schleife über $t$ bleibt nur $\mathbf{H}_{t-1}\mathbf{W}_{\textrm{hh}}$.

<!--en:d3a520fb-->
::: {.slide title="Ausdrucksstärke und der Flaschenhals eines Zustands mit 512 Byte"}

<!--en:777f701a-->
[Theoretische Möglichkeiten und Speichergrenze]{.kicker}

<!--en:9f931c92-->
Ersetzt ein rekursiv berechneter versteckter Zustand ein endliches Fenster, entsteht ein grundlegender Zielkonflikt zwischen **unbegrenzter zeitlicher Reichweite** und **begrenzter Informationskapazität des Zustands**:

<!--en:3843e6a1-->
- **Parität als unlösbare Aufgabe für jedes endliche Fenster der Länge $m$:** Ein Modell soll in jedem Schritt $t$ ausgeben, ob die bisherige Bitsumme $x_1 + \cdots + x_t$ gerade oder ungerade ist. Kippt man Bit $x_{t-m}$, kehrt sich die tatsächliche Parität im Schritt $t$ um. Die letzten $m$ Bits $(x_{t-m+1}, \ldots, x_t)$ bleiben gleich. Damit scheitert *jedes* Modell mit Fensterlänge $m$ an Sequenzen der Länge $T > m$. Ein RNN mit 2 Schwellwerteinheiten bestimmt die Parität für beliebiges $T$.

<!--en:3e085b58-->
- **Konstanter Speicherbedarf von 512 Byte:** Da $\mathbf{H}_t$ nur von $(\mathbf{H}_{t-1}, \mathbf{X}_t)$ abhängt, lassen sich ältere Aktivierungen bei der Inferenz verwerfen. Bei $h = 128$ Einheiten vom Typ `float32` belegt der gesamte Zustand einer Sequenz **$128 \times 4 = 512$ Byte**, ob das Präfix 10 oder $10^6$ Token umfasst.

<!--en:a36e095e-->
- **Verlustbehaftete Kompression als Flaschenhals:** Jedes Detail, das aus der Vergangenheit erhalten bleibt, muss in diese 512 Byte passen. Neue Informationen überschreiben zwangsläufig ältere Zustandskomponenten. Bei Attention wächst der KV-Cache dagegen wie $O(T)$.

<!--en:932be8f3-->
::: {.slide title="Architektur eines RNN-Sprachmodells und Teacher-Forcing"}

<!--en:0f5dca6b-->
[Vom Eingabetoken zur Vorhersage]{.kicker}

<!--en:46e1b6a8-->
In einem RNN-Sprachmodell liegt die rekurrente Zelle zwischen einer Einbettungstabelle $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ für die Eingabe und einer Ausgabeschicht $(\mathbf{W}_{\textrm{hq}}, \mathbf{b}_{\textrm{q}})$ in Vokabulargröße:

<!--en:7b93c8b3-->
- **Schritte des Vorwärtsdurchlaufs:**
  1. **Einbettungen nachschlagen:** Die Token-IDs $\mathbf{x}_t \in \{0, \ldots, |\mathcal{V}|-1\}^b$ wählen die Zeilen $\mathbf{X}_t = \mathbf{E}[\mathbf{x}_t] \in \mathbb{R}^{b \times d}$ aus. Das entspricht der One-Hot-Multiplikation $\textrm{one\_hot}(\mathbf{x}_t)\mathbf{E}$, benötigt aber nur $O(1)$ Zeit und keine Multiplikationen mit null.
  2. **Rekurrenten Zustand aktualisieren:** $\mathbf{H}_t = \tanh(\mathbf{X}_t\mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1}\mathbf{W}_{\textrm{hh}} + \mathbf{b}_{\textrm{h}}) \in \mathbb{R}^{b \times h}$.
  3. **Verteilung für das nächste Token:** $\hat{\mathbf{P}}_{t+1} = \textrm{softmax}(\mathbf{H}_t\mathbf{W}_{\textrm{hq}} + \mathbf{b}_{\textrm{q}}) \in \mathbb{R}^{b \times |\mathcal{V}|}$.

<!--en:517d60a2-->
- **Teacher-Forcing:** Beim Training erhält das Modell im Schritt $t$ stets das **Ground-Truth-Token** $x_t$ aus dem Korpus. So lassen sich alle $T$ Kreuzentropieverluste eines Fensters in einem Durchlauf berechnen.

<!--en:31bbf4e0-->
::: {.slide title="Wie verteilen sich Parameter und FLOPs?"}

<!--en:e96758c3-->
[Parameteranzahl und Rechenaufwand]{.kicker}

<!--en:19362e71-->
- **Basismodell ($|\mathcal{V}| = 1{,}027, d = 64, h = 128$):** Insgesamt $222{,}915$ Parameter und $156{,}032$ Multiplikationsadditionen (MACs) pro Token.

<!--en:6698858e-->
- **Die rekurrente Zelle ist klein ($11.1\%$ der Parameter):** Auf $\mathbf{W}_{\textrm{xh}}$ ($8{,}192$) und $\mathbf{W}_{\textrm{hh}} + \mathbf{b}_{\textrm{h}}$ ($16{,}512$) entfallen nur $24{,}704$ Parameter und $15.8\%$ des Rechenaufwands.

<!--en:7e7dfc01-->
- **Die vom Vokabular abhängigen Schichten dominieren ($88.9\%$ der Parameter):** Auf die Einbettung $\mathbf{E}$ ($65{,}728$ Parameter, Zeilenzugriff in $O(1)$) und die Ausgabeprojektion $\mathbf{W}_{\textrm{hq}} + \mathbf{b}_{\textrm{q}}$ ($132{,}483$ Parameter, $84.2\%$ der FLOPs) entfällt der größte Teil des Aufwands.

<!--en:ffed11ca-->
::: {.slide title="Implementierung der rekurrenten Zelle von Grund auf"}

<!--en:c8bf2621-->
[Eigene Implementierung]{.kicker}

<!--en:9115e713-->
`RNNScratch` initialisiert $\mathbf{W}_{\textrm{xh}}, \mathbf{W}_{\textrm{hh}} \sim \mathcal{N}(0, \sigma^2)$ mit $\sigma = 0.01$. Der anfängliche Spektralradius $\rho(\mathbf{W}_{\textrm{hh}}) \approx \sigma\sqrt{h} \approx 0.11$ sorgt dafür, dass der Zustand anfangs stabil abklingt. Die Schleife durchläuft die erste Achse, also die Zeitachse von `inputs` mit der Form `(num_steps, batch_size, num_inputs)`:

<!--en:70f8ce36-->
::: {.slide title="Implementierung der rekurrenten Zelle von Grund auf (Fortsetzung)"}

<!--en:c8bf2621-2-->
[Eigene Implementierung]{.kicker}

<!--en:e10119c5-->
::: {.slide title="Nachschlagen der Einbettungen und gebündelte Ausgabeprojektion"}

<!--en:e1ca47f7-->
[Sprachmodell von Grund auf]{.kicker}

<!--en:d7067204-->
`RNNLMScratch` transponiert die Token-IDs `X` der Form `(b, T)` so, dass die Zeitachse an erster Stelle steht (`(T, b)`). Dann schlägt es die Einbettungen in `self.E` nach ($\mathcal{N}(0, 1)$), führt `self.rnn` aus und projiziert alle gestapelten versteckten Zustände `(b, T, h)` in einem gebündelten Matrixprodukt auf die Logits `(b, T, |V|)`:

<!--en:21d77543-->
::: {.slide title="Nachschlagen der Einbettungen und gebündelte Ausgabeprojektion (Fortsetzung)"}

<!--en:e1ca47f7-2-->
[Sprachmodell von Grund auf]{.kicker}

<!--en:c9f625f4-->
**Plausibilitätsprüfung der Anfangsperplexität:** Wegen $\mathbf{W}_{\textrm{hq}} \sim \mathcal{N}(0, 0.01^2)$ und $\mathbf{b}_{\textrm{q}} = \mathbf{0}$ liefert ein untrainiertes Modell Logits nahe null. Daraus folgen annähernd gleiche Wahrscheinlichkeiten $1/|\mathcal{V}|$, eine Kreuzentropie von $\approx \log(1{,}027) = 6.93$ Nat und eine Anfangsperplexität von $\approx 1{,}027$.

<!--en:0b186194-->
::: {.slide title="Clipping der globalen Gradientennorm stabilisiert die Aktualisierungen"}

<!--en:757b4dd3-->
[Absicherung der Optimierung]{.kicker}

<!--en:2d773ca4-->
- **Radiale Projektion auf eine Kugel:** Sie begrenzt die globale euklidische Gradientennorm $\|\mathbf{g}\|$ auf $\theta$ und erhält die Richtung der Aktualisierung. So treiben seltene Spitzen der Jacobi-Matrix die Gewichte nicht aus dem Einzugsgebiet des Minimums.

<!--en:5aa28271-->
::: {.slide title="Training des einfachen RNN-Basismodells mit Clipping"}

<!--en:8bdc3a27-->
[Training des Basismodells und Auswertung in Bit pro Byte]{.kicker}

<!--en:1acf68fb-->
Das Basismodell mit 222 915 Parametern ($d = 64, h = 128$, ohne Dropout) sieht beim Training $50{,}000$ überlappende Fenster zu je 32 Token (`batch_size = 1024`). Es durchläuft 10 SGD-Epochen bei $\eta = 4$ mit globalem Gradienten-Clipping bei $\theta = 1$:

<!--en:0f733eff-->
- **An der Fenstergrenze abgeschnittene Zustände:** Jedes Fenster mit 32 Token beginnt mit $\mathbf{H}_0 = \mathbf{0}$. Das begrenzt die Backpropagation auf 32 Schritte.

<!--en:16056a85-->
- **Konvergenz:** Nach einer ersten Epoche mit starken Schwankungen sinkt die Validierungsperplexität von $1{,}027$ auf etwa $85$–$100$. Eine Perplexität von $95$ entspricht $\log_2(95) / 2.78 \approx 2.36$ Bit pro Byte. Der Trainingsverlust fällt schneller, da sich die 223 000 Parameter zu stark an die 50 000 Token anpassen.

<!--en:dd764c09-->
::: {.slide title="Ablationsstudie zur Instabilität ohne Clipping"}

<!--en:041e3e39-->
[Warum Gradienten-Clipping entscheidend ist]{.kicker}

<!--en:fcd810ea-->
Um den Effekt des Gradienten-Clippings zu isolieren, durchlaufen drei neue Instanzen desselben Modells das Training bei $\eta = 4$ ohne Clipping. Die Grafik vergleicht ihre Validierungsperplexität mit der des Basismodells mit Clipping:

<!--en:7da81cf4-->
- **Heftige Ausschläge:** Ohne Clipping bei $\theta = 1$ treiben explodierende Gradienten die Parameter aus dem Einzugsgebiet des Minimums. In einigen Läufen steigt die Validierungsperplexität weit über den Ausgangswert $1{,}027$ des untrainierten Modells.

<!--en:da3e44c7-->
- **Unzuverlässige Konvergenz:** Die meisten Läufe ohne Clipping bei $\eta = 4$ enden mit mehr als der doppelten Perplexität des Modells mit Clipping.

<!--en:0d4aa0ef-->
::: {.slide title="Warum SGD selten aktualisierte Einbettungen kaum verschiebt"}

<!--en:d4759d1a-->
[Der Flaschenhals beim Lernen von Einbettungen mit SGD]{.kicker}

<!--en:03bb7cbf-->
Der relative euklidische Abstand $\|\mathbf{E}_{i,\textrm{final}} - \mathbf{E}_{i,\textrm{init}}\| / \|\mathbf{E}_{i,\textrm{init}}\|$ für alle 1027 Zeilen von $\mathbf{E}$ nach 10 SGD-Epochen zeigt einen deutlichen Flaschenhals der Optimierung:

<!--en:988550c2-->
- **Spärliche und dichte Gradientenakkumulation:** Die dichten Gewichte $(\mathbf{W}_{\textrm{xh}}, \mathbf{W}_{\textrm{hh}}, \mathbf{W}_{\textrm{hq}})$ erhalten Gradientenbeiträge von allen $b \times T = 1{,}024 \times 32 = 32{,}768$ Tokenpositionen eines Minibatchs. Einbettungszeile $\mathbf{E}_i$ erhält Beiträge *nur* von den wenigen Positionen, an denen Token $i$ vorkommt.

<!--en:2d97b70a-->
- **Warum SGD mit einheitlicher Schrittweite bei $\mathbf{E}$ kaum vorankommt:** Nach der Division durch $32{,}768$ bleiben für seltene und mäßig häufige Token nur winzige Aktualisierungen. Im Mittel verschiebt sich eine Zeile in 10 Epochen um etwa $4\%$ ihrer Länge. Damit wirkt $\mathbf{E}$ weitgehend wie eine feste Zufallsprojektion. Adaptive Optimierer wie Adam nutzen die Varianz jeder Koordinate zur Skalierung und verändern $\mathbf{E}$ rasch.

<!--en:f86eaf3c-->
- **Abhilfe bei SGD:** Die zufällige Einbettungsbasis verbreitern ($d = 64 \to 512$), damit $\mathbf{W}_{\textrm{xh}}$ auf $8\times$ so viele Merkmale zugreifen kann. Dropout (`emb_dropout = 0.1`, `dropout = 0.5`) begrenzt die Überanpassung.

<!--en:fdbb9a50-->
::: {.slide title="Breitere Einbettungen mit Dropout erreichen rund 2.1 Bit pro Byte"}

<!--en:7263ceb7-->
[Breiteres Modell mit Regularisierung]{.kicker}

<!--en:56062106-->
Das Training von `RNNLMScratch` mit $d = 512$, $h = 128$, `emb_dropout = 0.1` und `dropout = 0.5` ($740{,}355$ Parameter) senkt die Validierungsperplexität um ein Drittel. Die SGD-Einstellungen bleiben gleich ($\eta = 4, \theta = 1$, 10 Epochen):

<!--en:8b74c607-->
- **Weniger Überanpassung:** Anders als beim unregularisierten Basismodell mit $d = 64$ liegen Trainings- und Validierungskurve dank Dropout nahe beieinander. Nach 10 Epochen beträgt der Verlustunterschied etwa $0.3$ Nat statt etwa $1$.

<!--en:6a0ce83a-->
- **Verbesserung bei der Validierung:** Die Validierungsperplexität fällt von $85\text{--}100$ auf $\sim 60$. Damit sinkt die Anzahl der Bit pro Byte von $2.3\text{--}2.4$ auf $\mathbf{\approx 2.1\text{ bpb}}$.

<!--en:f5b49679-->
::: {.slide title="Vergleich des RNN mit Kneser–Ney-n-Grammen"}

<!--en:eb709ea0-->
[Neuronale und klassische häufigkeitsbasierte Modelle]{.kicker}

<!--en:7cd6c4d9-->
Wie schneidet das trainierte RNN ($\approx 2.1$ bpb) gegen interpolierte Kneser–Ney-$n$-Gramme ($n = 2, 3, 4$) ab, die auf genau denselben 50 000 BPE-Token trainiert und auf `val[32:]` ausgewertet wurden?

<!--en:9bbd125c-->
- **Nahezu Gleichstand bei $\approx 2.1$ bpb:** Das RNN mit einem Fenster von 32 Token liegt fast gleichauf mit dem Kneser–Ney-Trigramm ($2.09$ bpb) und dem 4-Gramm ($2.08$ bpb), die nur 2 bzw. 3 Token zurückblicken.

<!--en:869ade2a-->
- **Komplementäre Fehler (Mikolov et al., 2011):** Trotz fast identischer mittlerer Bitzahl pro Byte schneiden das RNN (kontinuierliche semantische Darstellungen) und Kneser–Ney (exakte Wortfolgen) bei jeweils $\approx 50\%$ der Validierungstoken besser ab als das andere Modell. Eine $50/50$-Mischung $0.5\,P_{\textrm{RNN}} + 0.5\,P_{\textrm{KN}}$ gewinnt gegenüber dem Trigramm allein $\approx 0.06$ bpb.

<!--en:3429193f-->
::: {.slide title="Die effektive Kontextlänge des RNN"}

<!--en:73a61aa9-->
[Warum das RNN mit dem Trigramm gleichauf liegt]{.kicker}

<!--en:e5c35f8b-->
Warum übertrifft ein rekurrentes Netz mit 32 Token Vorgeschichte kein 3-Gramm? Die Grafik zeigt die Bit pro Byte auf den Validierungsdaten für jede Fensterposition $k \in \{1, \ldots, 32\}$ einzeln. An Position $k$ ist die Vorhersage durch $k$ Kontexttoken bedingt:

<!--en:583234c1-->
- **Starker Rückgang bei $k = 1 \to 2$, dann flacher Verlauf:** Der Validierungsverlust sinkt von $\approx 2.3$ bpb bei 1 Kontexttoken auf $\approx 2.14$ bpb bei 2 Token. Bei 3 Token fällt er um weitere $0.01$–$0.02$ und **bleibt von Position 4 bis Position 32 bei $\approx 2.12$ bpb**.

<!--en:f6962e48-->
- **Zwei mögliche Ursachen, die der Versuch nicht trennt:** *Optimierung*: Gradienten schwinden auf dem Rückweg mit jedem Schritt (§{sec_bptt}). *Daten*: Auch die gradientenfreie Kneser–Ney-Schätzung erreicht an derselben Stelle die Sättigung ($2.24 \to 2.09 \to 2.08$ bpb für $n = 2, 3, 4$). Vielleicht enthalten 50 000 Token nur wenig Information über längere Abstände.

<!--en:16b90f70-->
::: {.slide title="Autoregressive Textgenerierung und Wiederholungen beim gierigen Dekodieren"}

<!--en:7ef1b2c5-->
[Inferenz und Verlauf der rekursiven Vorhersage]{.kicker}

<!--en:fda9e23c-->
Die Methode `predict` durchläuft zunächst **eine Schleife über das Präfix**. Sie verarbeitet die Token des Prompts nacheinander, um $\mathbf{H}_t$ aufzubauen. In der anschließenden **Schleife zur autoregressiven Generierung** dient jedes neu gewählte Token als Eingabe des nächsten Schritts:

<!--en:59633459-->
::: {.slide title="Autoregressive Textgenerierung und Wiederholungen beim gierigen Dekodieren (Fortsetzung)"}

<!--en:7ef1b2c5-2-->
[Inferenz und Verlauf der rekursiven Vorhersage]{.kicker}

<!--en:ed338989-->
**Warum gieriges Dekodieren (`temperature=0.0`) in eine Schleife gerät:** Sowohl die rekurrente Aktualisierung als auch $\operatorname*{argmax}_i o_i$ sind deterministisch. Sobald der versteckte Zustand $\mathbf{H}_t$ wieder in die Nähe eines Zustands zu einer früheren Wortfolge gelangt, gerät die Trajektorie in einen endlosen periodischen Zyklus (§{sec_decoding}).

<!--en:1bfd8d04-->
::: {.slide title="Kompakte Implementierung mit RNN-Kerneln der Frameworks"}

<!--en:c8e3e606-->
[Höhere Programmierschnittstellen der Frameworks]{.kicker}

<!--en:951e58f9-->
Die eingebaute RNN-Schicht des Frameworks ersetzt die Python-Schleife `for X in inputs:` durch einen einzigen zusammengefassten Kernel:

<!--en:aca3e9b6-->
::: {.slide title="Kompakte Implementierung mit RNN-Kerneln der Frameworks (Fortsetzung)"}

<!--en:c8e3e606-2-->
[Höhere Programmierschnittstellen der Frameworks]{.kicker}

<!--en:6239515a-->
`RNNLM` verwendet die eingebaute Einbettungsschicht und die eingebaute vollständig verbundene Schicht:

<!--en:734a4434-->
::: {.slide title="Training und Perplexität der kompakten RNNLM-Implementierung"}

<!--en:748522a6-->
[Gleiche Ergebnisse bei kürzerer Laufzeit]{.kicker}

<!--en:c98e6cbc-->
Dieselbe Gewichtsinitialisierung ($\mathcal{N}(0, 0.01^2)$ für Zellen- und Ausgabegewichte, $\mathcal{N}(0, 1)$ für Einbettungen) reproduziert die Validierungsperplexität ($\sim 60$) und die Wiederholungen beim gierigen Dekodieren. Die Laufzeit beträgt nur einen Bruchteil der Laufzeit der Python-Schleife:

<!--en:371888f4-->
::: {.slide title="Verständnisfragen zu Formen, Skalierung und Kontextlänge"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:4c864519-->
Wie hängen Parameteranzahl, gebündelte Projektionen und die Kombination mehrerer Modelle mit den Eigenschaften eines RNN zusammen?

<!--en:c7ca82ea-->
1. **Verdopplung der Dimension des versteckten Zustands ($h: 128 \to 256$):** Die Dimension $h$ steigt von $128$ auf $256$, während $d = 64$ und $|\mathcal{V}| = 1{,}027$ gleich bleiben. Um welchen Faktor ($1\times, 2\times, 4\times$) ändert sich jeweils die Parameteranzahl von (a) $\mathbf{E}$, (b) $\mathbf{W}_{\textrm{xh}}$ und $\mathbf{W}_{\textrm{hq}}$ sowie (c) $\mathbf{W}_{\textrm{hh}}$?

<!--en:c37d0441-->
2. **Gebündelte Eingabeprojektion bei Training und Inferenz:** Beim Training mit Teacher-Forcing auf einem Fenster von $T = 32$ Schritten lässt sich $\mathbf{X}_{1:T}\mathbf{W}_{\textrm{xh}}$ für alle 32 Schritte in einem einzigen Matrixprodukt vor der Zeitschleife berechnen. Warum ist diese Optimierung bei 32 Schritten autoregressiver Textgenerierung *nicht* möglich?

<!--en:197be394-->
3. **Warum eine 50/50-Mischung der Wahrscheinlichkeiten den Überraschungswert begrenzt:** Für Token $A$ gelten $P_{\textrm{RNN}}(A) = 0.50$ und $P_{\textrm{KN}}(A) = 0.02$, für Token $B$ dagegen $P_{\textrm{RNN}}(B) = 0.02$ und $P_{\textrm{KN}}(B) = 0.50$. Den gesamten Überraschungswert $-\log_2 P(A) - \log_2 P(B)$ in Bit für jedes Einzelmodell und für die lineare Mischung $P_{\textrm{mix}} = \frac{1}{2}P_{\textrm{RNN}} + \frac{1}{2}P_{\textrm{KN}}$ vergleichen.

<!--en:3e03515d-->
::: {.slide title="Verständnisfragen zu Formen, Skalierung und Kontextlänge (Lösungen)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:5d07996c-->
1. **(a) $1\times$** ($\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ ist unabhängig von $h$), **(b) $2\times$** ($\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$ und $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times |\mathcal{V}|}$ wachsen linear mit $h$), **(c) $4\times$** ($\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ wächst proportional zu $h^2$: $16{,}384 \to 65{,}536$).

<!--en:41ccc705-->
2. **Sequenzielle kausale Abhängigkeit:** Bei der Generierung steht das Eingabetoken $x_{t+1}$ erst fest, wenn Schritt $t$ abgeschlossen ist. Dazu muss das Modell $\mathbf{H}_t$ berechnen, durch Projektion $\mathbf{O}_t$ bestimmen und schließlich $x_{t+1}$ ziehen.

<!--en:201bf774-->
3. **Einzelmodelle:** $-\log_2(0.50) - \log_2(0.02) \approx 1.00 + 5.64 = 6.64\text{ bits}$ für jedes Modell. **Mischung:** $P_{\textrm{mix}}(A) = P_{\textrm{mix}}(B) = 0.26$ mit einem Gesamtaufwand von $2 \times (-\log_2 0.26) \approx 3.89\text{ bits}$. Das spart $2.75$ Bit, denn wegen $\frac{1}{2}(p+q) \ge \frac{1}{2}\max(p,q)$ benötigt die Mischung pro Token höchstens $1$ Bit mehr als das jeweils bessere Modell.

<!--en:a7226355-->
::: {.slide title="Zusammenfassung der RNN-Sprachmodelle"}

<!--en:408fcbdb-->
[Ergebnisse des Abschnitts]{.kicker}

<!--en:85e4e35f-->
| Bestandteil oder Verfahren | Mathematische Darstellung | Formen ($b, T, d, h, \lvert\mathcal{V}\rvert$) | Rechenaufwand und statistische Eigenschaften |

<!--en:6420688d-->
| **Nachschlagen der Einbettungen** | $\mathbf{X}_t = \mathbf{E}[\mathbf{x}_t] = \textrm{one\_hot}(\mathbf{x}_t)\mathbf{E}$ | $\mathbf{E}: (\lvert\mathcal{V}\rvert, d)$, $\mathbf{X}: (T, b, d)$ | Zeilenzugriff in $O(1)$. Spärliche Gradienten verschieben die Zeilen seltener Token bei SGD nur langsam |

<!--en:a53d8398-->
| **Aktualisierung des rekurrenten Zustands** | $\mathbf{H}_t = \tanh(\mathbf{X}_t\mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1}\mathbf{W}_{\textrm{hh}} + \mathbf{b}_{\textrm{h}})$ | $\mathbf{W}_{\textrm{xh}}: (d, h)$, $\mathbf{W}_{\textrm{hh}}: (h, h)$, $\mathbf{H}_t: (b, h)$ | $11\%$ der Parameter des Basismodells. Der Zustand belegt $512$ Byte pro Sequenz, unabhängig von der Länge $T$ |

<!--en:d2353111-->
| **Ausgabeprojektion** | $\mathbf{O}_t = \mathbf{H}_t\mathbf{W}_{\textrm{hq}} + \mathbf{b}_{\textrm{q}}$, $\hat{\mathbf{P}}_{t+1} = \textrm{softmax}(\mathbf{O}_t)$ | $\mathbf{W}_{\textrm{hq}}: (h, \lvert\mathcal{V}\rvert)$, $\mathbf{O}: (b, T, \lvert\mathcal{V}\rvert)$ | $59\%$ der Parameter des Basismodells und $84\%$ der Multiplikationsadditionen pro Token skalieren mit $\lvert\mathcal{V}\rvert$ |

<!--en:fc9ff7c6-->
| **Globales Gradienten-Clipping** | $\mathbf{g} \leftarrow \min(1, \theta / \|\mathbf{g}\|)\mathbf{g}$ | Globale Norm $\|\mathbf{g}\|$ über alle $\Theta$ | Verhindert bei $\eta = 4$, dass Spitzen der Jacobi-Matrix die Gewichte durch zu große Schritte unbrauchbar machen |

<!--en:89b1cd36-->
| **Effektive Kontextlänge** | Verlust an Fensterposition $k \in \{1, \ldots, 32\}$ | Konstant bei $\approx 2.12$ bpb für $k \ge 4$ | Das einfache $\tanh$-RNN liegt gleichauf mit dem Kneser–Ney-3-Gramm. Es nutzt nur die letzten 2–3 Token (schwindende Gradienten, begrenzte Datenmenge oder beides) |
