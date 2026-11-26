<!--en:075ae896-->
# Lineare Rekurrenzen und parallele Scans

<!--en:4ac014eb-->
Die Netze mit Gattern (engl. gates) aus :numref:`sec_lstm` können Informationen über Hunderte von Schritten bewahren. Dennoch verarbeiten sie eine Sequenz Token für Token, weil die Gatter und der Kandidat für den neuen Zustand in jedem Schritt vom vorherigen Zustand $\mathbf{h}_{t-1}$ abhängen. Bei einer Sequenz von $4{,}096$ Token führt eine GRU $4{,}096$ Aktualisierungen nacheinander aus. Dadurch bleibt ein Großteil der GPU beim Training ungenutzt. Ein Transformer verarbeitet dagegen alle Positionen parallel mit großen Matrizenmultiplikationen. Kann ein rekurrentes Netz bei der autoregressiven Generierung weiterhin mit $O(1)$ Speicher für den Zustand auskommen und beim Training alle Positionen parallel verarbeiten?

<!--en:67051a91-->
Ja, sobald die Gatter und der Kandidat nur von der aktuellen Eingabe $\mathbf{x}_t$ abhängen. Dann ist die Rekurrenz *linear* (genauer gesagt *affin*) in $\mathbf{h}_{t-1}$. Affine Aktualisierungen lassen sich wie bei der Berechnung von Präfixsummen assoziativ verknüpfen :cite:`Hillis.Steele.1986`. Damit lassen sich alle $T$ Zustände in $\lceil \log_2 T \rceil$ parallelen Runden berechnen. Wir beginnen mit einem skalaren Abklingfaktor und leiten die minimale GRU (`MinGRU`) von :citet:`Feng.Tung.Ahmed.ea.2024` her. Anschließend implementieren wir den parallelen assoziativen Scan und werten `MinGRU` auf WikiText-103 auf Byte-Ebene aus. Für Sequenzen mit bis zu $16{,}384$ Token vergleichen wir, wie die Laufzeiten des Scans, der sequenziellen GRU und der kausalen Attention mit der Länge wachsen.

<!--en:553186e2-->
## Lineare Rekurrenzen

<!--en:2cb474b8-->
### Das Abklingen einer einzelnen Zahl

<!--en:eb03fda8-->
Beginnen wir mit einem skalaren Zustand $h_t$. Er behält einen festen Anteil $a \in (0, 1)$ seines vorherigen Werts bei und addiert die Eingabe $x_t$:

<!--en:2b4858d8-->
Durch wiederholtes Einsetzen in :eqref:`eq_lr-scalar` erhalten wir $h_t$ direkt als Funktion der bisherigen Eingaben:

<!--en:734ee400-->
Jeder Zustand $h_t$ ist eine exponentiell gewichtete gleitende Summe der bisherigen Eingaben. Die Halbwertszeit beträgt $\ln 2 / \ln(1/a)$ Schritte ($\approx 7$ Schritte bei $a = 0.9$ und $\approx 693$ Schritte bei $a = 0.999$). Der Gradient $\partial h_t / \partial x_s = a^{t-s}$ klingt mit genau derselben Rate ab.

<!--en:c91506e0-->
### Eingabeabhängiges Abklingen

<!--en:3f17df55-->
Verallgemeinern wir den Zustand zu einem Vektor $\mathbf{h}_t \in \mathbb{R}^d$ mit eingabeabhängigen Abklingfaktoren $\mathbf{a}_t \in (0, 1)^d$ für die einzelnen Komponenten und einem *Schreibvektor* (engl. write vector) $\mathbf{b}_t \in \mathbb{R}^d$. Das ergibt die affine Rekurrenz:

<!--en:46afd3b3-->
Entwickeln wir :eqref:`eq_affine_recurrence` durch wiederholtes Einsetzen, so tritt an die Stelle der Potenz $a^{t-s}$ das kumulative Produkt der dazwischenliegenden Abklingfaktoren:

<!--en:87427059-->
Da $\mathbf{a}_t$ und $\mathbf{b}_t$ nur von $\mathbf{x}_t$ abhängen, können wir alle $T$ Paare aus Abkling- und Schreibvektoren für die gesamte Sequenz parallel berechnen, bevor wir die Rekurrenz auswerten. Da außerdem stets $\mathbf{a}_r \in (0, 1)^d$ gilt, lassen sich frühere Schreibbeiträge gezielt erhalten ($\mathbf{a}_r \approx \mathbf{1}$) oder löschen ($\mathbf{a}_r \approx \mathbf{0}$), ohne dass sie je unbeschränkt anwachsen.

<!--en:f9fd904b-->
### Die minimale GRU

<!--en:e67efe9d-->
Betrachten wir die drei Stellen, an denen der vorherige Zustand $\mathbf{h}_{t-1}$ in die Standard-GRU aus :numref:`sec_gru` eingeht:

<!--en:4f6d73de-->
Anschließend gilt $\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$. Streichen wir die drei markierten rekurrenten Abhängigkeiten, entfällt das Reset-Gatter $\mathbf{r}_t$ ganz. So erhalten wir die *minimale GRU* (*minGRU*) von :citet:`Feng.Tung.Ahmed.ea.2024`:

<!--en:cd1cdd19-->
Dies ist eine affine Rekurrenz gemäß :eqref:`eq_affine_recurrence` mit Abklingvektor $\mathbf{a}_t = \mathbf{z}_t$ und Schreibvektor $\mathbf{b}_t = (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$. Die Parameteranzahl der Schicht sinkt von $6d^2$ auf $2d^2$. Das Gatter einer einzelnen minGRU-Schicht hat keinen Zugriff auf den bereits in $\mathbf{h}_{t-1}$ gespeicherten Inhalt. Stapeln wir mehrere minGRU-Blöcke mit positionsweise angewandten MLPs, gewinnen wir diese Fähigkeit teilweise zurück. Die Gatter jeder Schicht sehen die Zustände der darunterliegenden Schicht, und wir können weiterhin alle Zeitschritte parallel auswerten. Damit erhalten wir pro Schicht eine Runde zustandsabhängiger Gattersteuerung, bei der GRU dagegen eine pro Zeitschritt. :numref:`subsec_dn-state-tracking` untersucht empirisch, wo die Grenzen einer festen Rundenzahl liegen.

<!--en:ab53b684-->
## Parallele Scans

<!--en:faccc20b-->
### Verkettung affiner Schritte

<!--en:527f381a-->
Die direkte Auswertung der entwickelten Summe :eqref:`eq_lr-unrolled-gated` für alle $t = 1, \ldots, T$ würde $O(T^2)$ Operationen erfordern. Wir nutzen stattdessen die Assoziativität der Funktionenverkettung. Für Schritt $t$ stellen wir die affine Abbildung $\mathbf{h} \mapsto \mathbf{a}_t \odot \mathbf{h} + \mathbf{b}_t$ durch das Paar $(\mathbf{a}_t, \mathbf{b}_t)$ dar. Wenden wir zuerst Schritt 1 und dann Schritt 2 an, ergibt sich $\mathbf{a}_2 \odot (\mathbf{a}_1 \odot \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2$. Auch dies ist eine affine Abbildung mit den Koeffizienten:

<!--en:9b2f721a-->
Die Funktionenverkettung ist assoziativ, und die Verkettung eines beliebigen Blocks von Schritten ergibt wieder ein einziges Paar $(\mathbf{a}, \mathbf{b})$ derselben Größe. Somit genügt für die Berechnung von $\mathbf{h}_1, \ldots, \mathbf{h}_T$ aus $\mathbf{h}_0 = \mathbf{0}$ ein assoziativer Präfix-*Scan* mit der Operation $\bullet$. Den Zustand $\mathbf{h}_t$ lesen wir aus der zweiten Komponente des $t$-ten Präfixes ab.

<!--en:048b650e-->
### Verdopplung des Strides

<!--en:9edab3e0-->
:numref:`fig_scan_tree` zeigt das parallele Verdopplungsverfahren von :citet:`Hillis.Steele.1986` für $T = 8$ Schritte. In Runde $r = 0, 1, \ldots, \lceil \log_2 T \rceil - 1$ verknüpft jede Position ihren Wert mit dem Wert, der um den Stride $2^r$ vor ihr liegt, sofern diese frühere Position existiert. Nach $\lceil \log_2 T \rceil$ Runden enthält jede Position ihr vollständiges Präfix.

<!--en:ea68843a-->
![Ein paralleler Scan über acht Schritte. Jedes Kästchen zeigt, welche Schritte sein Wert zusammenfasst. So steht $(1{:}4)$ für die Schritte 1 bis 4. Die blauen Pfeile verknüpfen Werte in einem festen Abstand, der sich von Runde zu Runde verdoppelt. Gestrichelte Pfeile kopieren einen Wert unverändert. Türkise Kästchen enthalten vollständige Präfixe. Nach $\log_2 8 = 3$ Runden gilt das für alle Positionen.](../img/mdl-modernrnn-scan-tree.svg)

<!--en:c677c2e4-->
Das Verdopplungsverfahren von Hillis und Steele benötigt $O(T \log_2 T)$ Verknüpfungen in $\lceil \log_2 T \rceil$ parallelen Runden. Das *arbeitseffiziente* Baumverfahren von :citet:`Blelloch.1990` durchläuft den Baum erst aufwärts und dann abwärts. Es benötigt $\sim 2T$ Verknüpfungen in $2\lceil \log_2 T \rceil$ Runden mit je nach Baumebene unterschiedlich vielen aktiven Positionen.

<!--en:507bf689-->
Wir implementieren das kompakte Verdopplungsverfahren als `associative_scan`.

<!--en:e4af7dfc-->
Wir verwenden die in XLA integrierte arbeitseffiziente Funktion `jax.lax.associative_scan`.

<!--en:869d223f-->
### Implementierung

<!--en:f2035749-->
`linear_scan` berechnet alle Zustände $\mathbf{h}_1, \ldots, \mathbf{h}_T$ aus :eqref:`eq_affine_recurrence` entlang der Zeitachse (Achse 1). Einen optionalen Anfangszustand $\mathbf{h}_0$ berücksichtigt die Funktion bereits im ersten Schreibvektor gemäß $\mathbf{b}_1 \leftarrow \mathbf{a}_1 \odot \mathbf{h}_0 + \mathbf{b}_1$.

<!--en:5701493e-->
Vergleichen wir `linear_scan` mit einer Schleife, die jeden Schritt einzeln ausführt. Als Länge wählen wir $T = 100$, also keine Zweierpotenz.

<!--en:1ecba678-->
Die Ergebnisse des parallelen Scans und der sequenziellen Schleife stimmen bis auf die Rundungstoleranz von float32 überein ($< 10^{-6}$).

<!--en:f8151318-->
## Ein lineares rekurrentes Sprachmodell

<!--en:ac4e0e12-->
### Die minGRU als Textmodell

<!--en:f66dc32e-->
Wir wenden `linear_scan` auf zwei lineare Projektionen an ($\mathbf{W}_\textrm{z}$ und $\mathbf{W}_\textrm{h}$). Damit implementieren wir `MinGRU` gemäß :eqref:`eq_mingru` als direkt austauschbaren Sequenzmischer für `d2l.SeqLM`.

<!--en:c39b4c56-->
Wie viel Modellgüte geht auf WikiText-103 auf Byte-Ebene verloren, wenn wir $\mathbf{h}_{t-1}$ aus den Gattern entfernen? Wir übernehmen den Versuchsaufbau aus :numref:`subsec_deep-rnn` unverändert (4 Blöcke in `d2l.SeqLM`, Fenster von 128 Byte, 1000 Schritte). Zuerst definieren wir `text_model` und legen in `arms` die drei Vergleichsvarianten fest: `GRU` ($d = 256$, $3.81\textrm{M}$ Parameter), `minGRU` mit gleicher Breite ($d = 256$, $2.76\textrm{M}$ Parameter) und `minGRU` mit größerer Breite für dieselbe Parameteranzahl (`minGRU, d=302`, $3.81\textrm{M}$ Parameter):

<!--en:516497af-->
Wir laden den in Fenster von 128 Byte unterteilten Datensatz `WikiText103` und rufen `d2l.text_table` für alle drei Varianten auf. Das erste trainierte `minGRU`-Modell behalten wir für die weitere Untersuchung:

<!--en:78a5f29d-->
Bei gleicher Breite ($d = 256$) liegt `minGRU` am Ende um $\sim 0.05$ Nat pro Byte hinter `GRU` ($1.35$ gegenüber $1.30$). Bei gleicher Parameteranzahl (`minGRU, d=302`) schließt sich die Lücke um etwa zwei Fünftel ($\sim 1.33$). Damit liegt das Modell gleichauf mit dem einfachen $\tanh$-RNN aus :numref:`subsec_deep-rnn` und übertrifft die Attention-Baseline ($1.38$). Für das Training nutzt es einen parallelen Scan anstelle einer sequenziellen Schleife.

<!--en:6f13ac1c-->
### Einblick in das trainierte Modell

<!--en:cd6d0c1b-->
Wie stark stützt sich `minGRU` auf seinen rekurrenten Zustand? Wir werten das trainierte Modell der Breite $d = 256$ aus und setzen vor jedem Schritt seinen Zustand auf null (`StateReset`). Damit hängt $\mathbf{h}_t = \mathbf{b}_t$ nur vom aktuellen Byte ab.

<!--en:83eff573-->
Das Zurücksetzen des Zustands erhöht den Validierungsverlust von $1.35$ auf $2.48$ Nat pro Byte. Das entspricht nahezu den $2.45$ Nat pro Byte der empirischen Byte-Bigramm-Baseline aus :numref:`sec_gpt`. Der im Zustand gespeicherte Kontext senkt den Verlust also um mehr als $1.1$ Nat pro Byte.

<!--en:5bfe6149-->
Wir können auch untersuchen, wie sich die Abklingfaktoren $\mathbf{z}_t$ der Update-Gatter im ersten Block vor und nach dem Training verteilen. Mit der zusätzlichen Methode `first_mixer_input` in `d2l.SeqLM` können wir die normalisierten Tokeneinbettungen auslesen, die der erste Sequenzmischer als Eingabe erhält.

<!--en:4eaa1a18-->
Wir wenden `first_gates` auf einen Validierungsminibatch an, einmal mit einer untrainierten `minGRU` und einmal mit unserem trainierten Modell `model`. Die Tabelle zeigt, welcher Anteil der Update-Gatter jeweils über $0.5$, $0.9$ und $0.99$ liegt:

<!--en:09683d25-->
Durch das Training steigt der Anteil der Gatter mit einer Halbwertszeit von $\ge 7$ Byte ($g > 0.9$) auf mehr als das Zehnfache (von $\sim 1\%$ auf etwa $18\%$). Das entspricht den Zeitskalen auf Wortebene, die bei Fenstern von 128 Byte relevant sind. Bei einer Initialisierung mit Bias null erreichen hingegen fast keine Gatter $g > 0.99$ (Halbwertszeit 69 Byte).

<!--en:bb396fe5-->
## Kosten im Vergleich zur Attention

<!--en:3ea1e294-->
### Rechenaufwand, Tiefe und Speicherbedarf

<!--en:7a20e199-->
:numref:`tab_linear-recurrence-cost` vergleicht für eine Schicht der Breite $d$ und eine Sequenz aus $T$ Token drei Größen: den asymptotischen Rechenaufwand beim Training, die sequenzielle Tiefe und den Speicherbedarf für den Zustand bei der autoregressiven Generierung.

<!--en:c098b47f-->
:Kosten einer Schicht der Breite $d$ für $T$ Token bis auf konstante Faktoren.

<!--en:ab2e7ff7-->
| Schicht | Rechenaufwand beim Training | Schritte nacheinander | Speicherbedarf bei der Generierung |

<!--en:6c6b5287-->
| GRU, Schritt für Schritt | $T d^2$ | $T$ | $d$ |

<!--en:bdf03237-->
| minGRU, Verdopplungsscan | $T d^2 + T d \log_2 T$ | $\log_2 T$ | $d$ |

<!--en:9229226f-->
| minGRU, arbeitseffizienter Scan | $T d^2 + T d$ | $2 \log_2 T$ | $d$ |

<!--en:884ea703-->
| Kausale Attention | $T d^2 + T^2 d$ | wenige Matrixprodukte | $2 T d$ |

<!--en:65d996a9-->
Sowohl `GRU` als auch `minGRU` benötigen $O(Td^2)$ Rechenoperationen für die Projektionen. Beim Dekodieren führen beide einen Zustandsvektor der Größe $O(d)$ mit. Die `minGRU` verringert jedoch die sequenzielle Tiefe beim Training von $T$ Matrix-Vektor-Produkten auf $O(\log_2 T)$ Scan-Runden mit elementweisen Operationen. Kausale Attention hat beim Training eine sequenzielle Tiefe von $O(1)$. Der Aufwand für die Attention-Scores und das Mischen der Werte wächst jedoch quadratisch mit $O(T^2 d)$ und übersteigt die $8Td^2$ FLOPs für die Projektionen ab etwa $T \approx 4d$. Der KV-Cache für die Inferenz wächst linear mit $2Td$.

<!--en:709e074b-->
### Gemessene Kosten

<!--en:388695b3-->
Wo schneiden sich die Kurven dieser Skalierungsgesetze auf realer Hardware? Dazu messen wir die Zeit für einen Vorwärts- und einen Rückwärtsdurchlauf in float16 bei einer Breite von $d = 256$. Wir halten die Tokenanzahl pro Batch bei $B \times T = 65{,}536$ und variieren die Sequenzlänge $T$ von $512$ bis $16{,}384$. Bei fester Tokenanzahl verläuft die Kurve einer Schicht mit streng linearen Kosten in $T$ waagerecht. Dominieren dagegen $O(T)$ sequenzielle Schritte oder $O(T^2)$ paarweise Operationen, verdoppelt sich die Laufzeit mit jeder Verdopplung von $T$.

<!--en:83efca2c-->
Wir rufen `time_call` für die Sequenzlängen $T \in \{512, 1024, 2048, 4096, 8192, 16384\}$ auf und vergleichen die gemessenen Laufzeiten von `GRU`, `minGRU` und `attention` in einem doppelt logarithmischen Diagramm:

<!--en:3024f70d-->
Die gemessenen Kurven entsprechen den Vorhersagen aus :numref:`tab_linear-recurrence-cost`:

<!--en:d00847cc-->
- **`GRU`:** Bei jeder Länge am langsamsten. Die Laufzeit steigt um mehr als den Faktor $15\times$, wenn die Länge von $T = 512$ auf $16{,}384$ wächst, da sich die $T$ aufeinanderfolgenden Schritte nicht über die Zeit parallelisieren lassen.

<!--en:a2601424-->
- **`minGRU`:** Die Laufzeit bleibt nahezu konstant, obwohl die Sequenzlänge auf das 32-Fache steigt.

<!--en:9779167c-->
- **`attention`:** Bis $T \approx 2{,}048$ bleibt die Laufzeit nahezu konstant, da die Projektionen mit Aufwand $O(Td^2)$ und die Speichertransfers dominieren. Danach nähert sich der Anstieg dem Faktor $2\times$ je Verdopplung der Länge, sobald der Attention-Term mit Aufwand $O(T^2 d)$ überwiegt.

<!--en:76e47856-->
Die Laufzeit des Verdopplungsscans der `minGRU` wächst um weniger als $50\%$. Bei kurzen Sequenzen ist `attention` schneller. Bei etwa $T = 4{,}096$ schneiden sich die Kurven, und bei langen Sequenzen braucht die Attention rund $2.5\times$ so lange wie der Scan (bei $T = 16{,}384$).

<!--en:96784ed1-->
Die Laufzeit des arbeitseffizienten Scans der `minGRU` bleibt konstant. Er ist bei jeder Länge schneller als `attention`, etwa um den Faktor $14\times$ bei $T = 16{,}384$.

<!--en:cff0613c-->
Der `minGRU` fehlt noch ein systematischer Weg, Speicher für unterschiedliche Zeitskalen zu initialisieren und zu steuern. Dieser Frage geht :numref:`sec_ssm` anhand von Zustandsraummodellen in kontinuierlicher Zeit nach.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:2cde00c0-->
Entfernen wir den vorherigen Zustand $\mathbf{h}_{t-1}$ aus den Gattern einer rekurrenten Schicht, ergibt sich eine affine Rekurrenz $\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$. Ihre Schritte lassen sich durch $(\mathbf{a}_1, \mathbf{b}_1) \bullet (\mathbf{a}_2, \mathbf{b}_2) = (\mathbf{a}_1 \odot \mathbf{a}_2, \mathbf{a}_2 \odot \mathbf{b}_1 + \mathbf{b}_2)$ assoziativ verknüpfen. Beim Training berechnet ein assoziativer Scan alle $T$ Zustände in $O(\log_2 T)$ parallelen Runden. Beim Dekodieren genügen weiterhin sequenzielle Aktualisierungen mit $O(1)$ Speicher für den Zustand. Auf die GRU angewandt ergibt diese Vereinfachung die `MinGRU`. Auf WikiText-103 auf Byte-Ebene erreicht sie die Güte eines einfachen RNN mit gleicher Parameteranzahl (etwa $0.03$ Nat pro Byte hinter einer vollständigen GRU). Bis zu einer Länge von $16{,}384$ Token bleibt die Trainingszeit pro Token nahezu konstant. Bei langem Kontext ist das Training damit um mehr als eine Größenordnung schneller als bei einer sequenziellen GRU und um ein Mehrfaches schneller als bei quadratischer Attention.

<!--en:2294243a-->
## Übungen

<!--en:c6872e78-->
1. **Abgeschlossenheit unter Verkettung.** Ein paralleler Scan erfordert mehr als Assoziativität. Die Verkettung einer beliebigen Folge von Schritten muss sich durch ebenso viele Zahlen beschreiben lassen wie ein einzelner Schritt. Nur so kostet jede Verknüpfung gleich viel.
    1. Verallgemeinern Sie :eqref:`eq_scan_combine` auf die Rekurrenz $\mathbf{h}_t = \mathbf{A}_t \mathbf{h}_{t-1} + \mathbf{b}_t$ mit dicht besetzten Matrizen $\mathbf{A}_t \in \mathbb{R}^{d \times d}$. Achten Sie bei jedem Produkt auf die Reihenfolge der Faktoren. Zählen Sie die Multiplikationsadditionen für eine Verknüpfung und für einen sequenziellen Schritt. Vergleichen Sie den Gesamtaufwand des Verdopplungsscans (etwa $T \log_2 T$ Verknüpfungen) mit dem der Schleife bei $T = 4096$ und $d = 64$. Was legt dieser Vergleich für dicht besetzte Übergangsmatrizen nahe?
    1. Betrachten Sie den skalaren Schritt $h \mapsto \tanh(w_t h + c_t)$. Zeigen Sie, dass sich die Verkettung zweier Schritte mit $w_1, w_2 \neq 0$ für keine Wahl der Parameter in der Form $h \mapsto \tanh(w h + c)$ mit $w$ und $c$ schreiben lässt. Was müsste ein Scan speichern, um die Verkettung von $k$ Schritten darzustellen?

<!--en:2ffdb390-->
1. [code] **Der direkte Weg.** Bei skalaren Abklingfaktoren lässt sich die entwickelte Summe :eqref:`eq_lr-unrolled-gated` ohne Scan berechnen. Dazu genügen die kumulativen Produkte $P_t = a_1 a_2 \cdots a_t$ und eine kumulative Summe.
    1. Zeigen Sie, dass $h_t = P_t \sum_{s=1}^{t} b_s / P_s$ gilt. Implementieren Sie die Formel mit den Bibliotheksfunktionen `cumprod` und `cumsum`. Vergleichen Sie das Ergebnis mit `linear_scan` für zufällig gezogene Abklingfaktoren nahe 0.9 in 32-Bit-Gleitkommaarithmetik. Ab welcher Länge versagt die direkte Berechnung? Warum funktioniert der Scan dort weiterhin?
    1. :citet:`Feng.Tung.Ahmed.ea.2024` beheben das Problem, indem sie mit den Logarithmen der Abklingfaktoren und der Schreibbeiträge rechnen. An die Stelle der Summe tritt ein kumulatives Log-Sum-Exp. Implementieren Sie diese Korrektur. Welche Bedingung müssen die Vorzeichen der Schreibbeiträge erfüllen, und wie stellen die Autoren das sicher?

<!--en:98fa5295-->
1. [code] **Nichtlinearität des Kandidaten.** `MinGRU` behält die $\tanh$-Funktion für den Kandidaten $\tilde{\mathbf{h}}_t$ bei. Dagegen verwenden :citet:`Feng.Tung.Ahmed.ea.2024` einen linearen Kandidaten.
    1. Gehen Sie von $\mathbf{h}_0 = \mathbf{0}$ aus und schreiben Sie jede Komponente von $\mathbf{h}_t$ als gewichtete Summe der Kandidaten $\tilde{\mathbf{h}}_1, \ldots, \tilde{\mathbf{h}}_t$. Zeigen Sie, dass die Gewichte nichtnegativ sind und ihre Summe höchstens eins beträgt. Leiten Sie für $|\mathbf{h}_t|$ eine Schranke her, die bei jeder Länge gilt, jeweils mit und ohne $\tanh$-Funktion.
    1. Entfernen Sie die $\tanh$-Funktion und trainieren Sie das Sprachmodell dieses Abschnitts zweimal mit `d2l.train_text`. Vergleichen Sie seinen Validierungsverlust mit dem der beiden Trainingsläufe von `MinGRU`. Ist der Unterschied größer als die Streuung zwischen den Läufen? Setzen Sie Ihre Beobachtung zu Ihrer Schranke in Beziehung.

<!--en:8351519d-->
1. [code] **Weitergabe des Zustands.** `linear_scan` akzeptiert einen Anfangszustand, sodass sich eine lange Sequenz abschnittsweise verarbeiten lässt.
    1. Teilen Sie eine Sequenz der Länge $T$ in Abschnitte der Länge $\tau$. Führen Sie den Scan für jeden Abschnitt mit dem Endzustand des vorherigen Abschnitts als Anfangszustand aus. Prüfen Sie, ob die aneinandergereihten Zustände mit denen eines einzigen Aufrufs für die gesamte Sequenz übereinstimmen.
    1. Stoppen Sie nun den Gradientenfluss durch den weitergegebenen Zustand an jeder Abschnittsgrenze. Die Zustände ändern sich dadurch nicht. Welche Beiträge zum Gradienten des Verlusts in Schritt $t$ entfallen? Welche Abhängigkeiten kann das trainierte Modell bei der Textgenerierung weiterhin nutzen? Vergleichen Sie dies mit der abgeschnittenen Backpropagation durch die Zeit (:numref:`sec_bptt`).

<!--en:23b396e6-->
1. [code] **Arbeitseffiziente Scans.** ● Ein rekursives Verfahren benötigt weniger Rechenarbeit als das Verdopplungsverfahren. Verknüpfen Sie zunächst benachbarte Positionen paarweise. Das halbiert die Sequenzlänge. Ein rekursiver Scan über die halb so lange Sequenz liefert jedes zweite Präfix. Jedes übrige Präfix erhalten Sie mit einer weiteren Verknüpfung.
    1. Zählen Sie für $T = 2^k$ bei beiden Verfahren die Verknüpfungen und die Verknüpfungsrunden. Vergleichen Sie beides mit den $T - 1$ Schritten der Schleife bei $T = 16384$.
    1. Implementieren Sie beide Verfahren mit Slices und prüfen Sie die Ergebnisse anhand von `sequential_scan`. Messen Sie die Laufzeiten bei den Sequenzlängen aus :numref:`subsec_linear-recurrence-cost`. Entspricht das Verhältnis der Laufzeiten dem Verhältnis der Verknüpfungszahlen? Erklären Sie, warum die Verhältnisse bei kurzen Sequenzen voneinander abweichen können.

<!--en:7b3a7b51-->
[Diskussion](https://d2l.discourse.group/t/36885)

<!--en:7ec304cd-->
Lineare Rekurrenzen und parallele assoziative Scans<br>**Affine Zustandsübergänge · minimale GRU (minGRU) · assoziative Präfix-Scans · Komplexität im Vergleich zur Attention**

<!--en:1a0384ee-->
::: {.slide title="Wie nichtlineare Rekurrenz die GPU-Parallelität begrenzt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:b0232b52-->
- **Sequenzieller Flaschenhals:** Bei $T = 4{,}096$ Token führt eine nichtlineare GRU $4{,}096$ Schritte nacheinander aus. Schritt $t$ wartet auf $\mathbf{h}_{t-1}$ als Eingabe für $\tanh(\cdot)$.

<!--en:807e44a0-->
- **Linearisierung:** Ohne $\mathbf{h}_{t-1}$ als Gattereingabe wird der Übergang affin ($\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$).

<!--en:9055e57b-->
- **Logarithmische Tiefe:** Affine Übergänge lassen sich in $\log_2 4{,}096 = 12$ parallelen Runden assoziativ verketten.

<!--en:81c26246-->
::: {.slide title="Die skalare lineare Rekurrenz als exponentiell gewichtete gleitende Summe"}

<!--en:1fc1aaf9-->
[Grundlagen der linearen Rekurrenz]{.kicker}

<!--en:c805daee-->
- **Impulsantwort:** Ein Einheitsimpuls bei $s=1$ klingt geometrisch gemäß $a^{t-1}$ ab. Seine Halbwertszeit beträgt $\tau_{1/2} = \ln 2 / \ln(1/a)$.

<!--en:37c7dca9-->
- **Symmetrie von Vorwärts- und Rückwärtsdurchlauf:** Die Sensitivität $\partial h_t / \partial x_s = a^{t-s}$ im Rückwärtsdurchlauf entspricht genau dem Gewicht, mit dem die Eingabe im Vorwärtsdurchlauf erhalten bleibt.

<!--en:da9ffc0e-->
::: {.slide title="Eingabeabhängige affine Vektorrekurrenzen"}

<!--en:ea2761b6-->
[Zeitlich veränderliche Abklingfaktoren]{.kicker}

<!--en:1c2fa51f-->
- **Parallele Vorberechnung:** Da $\mathbf{a}_t, \mathbf{b}_t$ nur von $\mathbf{x}_t$ abhängen, genügt eine einzige Batch-Matrizenmultiplikation (GEMM) für alle $T$ Gatter- und Schreibvektoren.

<!--en:0ad40049-->
- **Selektive Taktsteuerung:** Bei aussagekräftigen Token gilt $a_{t,i} \approx 0$, sodass neue Fakten in den Speicher gelangen. Bei Fülltoken gilt $a_{t,i} \approx 1$, sodass der Speicherinhalt ohne Abklingen erhalten bleibt.

<!--en:c6648962-->
::: {.slide title="Von der Standard-GRU zur minimalen GRU (minGRU)"}

<!--en:aa03a86b-->
[Architektur der minGRU]{.kicker}

<!--en:3e07d0a3-->
- **Wegfall rekurrenter Eingaben:** Aus der Standard-GRU entfallen $\mathbf{W}_{\textrm{hz}}\mathbf{h}_{t-1}$, $\mathbf{W}_{\textrm{hr}}\mathbf{h}_{t-1}$ und $\mathbf{W}_{\textrm{hh}}(\mathbf{r}_t \odot \mathbf{h}_{t-1})$ (Feng et al., 2024).

<!--en:6d936477-->
- **Kompakt und für Scans geeignet:** Die Parameteranzahl sinkt um den Faktor $3\times$ ($6d^2 \to 2d^2$). Der Übergang hat direkt die Form $\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$.

<!--en:e46e2bbc-->
::: {.slide title="Was einer einzelnen linearen Schicht fehlt und was Tiefe ausgleicht"}

<!--en:141be7ae-->
[Grenzen der Ausdrucksstärke]{.kicker}

<!--en:049372e0-->
- **Eine Schicht sieht den Speicherinhalt nicht:** Da $\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{z}\mathbf{x}_t + \mathbf{b}_\textrm{z})$ keinen Zugriff auf $\mathbf{h}_{t-1}$ hat, erkennt eine `minGRU` mit 1 Schicht keinen bereits belegten Speicherplatz und überschreibt ihn bei späteren Ziffern.

<!--en:db3a8eda-->
- **Eine Runde zustandsabhängiger Steuerung pro Schicht:** Bei 2 Blöcken fasst der Zustand $\mathbf{h}_t^{(1)}$ von Schicht 1 die Vorgeschichte zusammen. So kann das Gatter $\mathbf{z}_t^{(2)}$ von Schicht 2 den Speicherplatz sperren. Das Training mit parallelem Scan bleibt möglich.

<!--en:090aa603-->
::: {.slide title="Assoziative Verkettung affiner Zustandsübergänge"}

<!--en:a33813a5-->
[Algebra paralleler Scans]{.kicker}

<!--en:0cdb1536-->
- **Abgeschlossenheit unter Verkettung:** Die Verkettung zweier affiner Schritte $f_2(f_1(\mathbf{h}))$ ergibt ein einziges affines Paar $(\mathbf{A}, \mathbf{B})$ der Breite $d$ und erfordert $O(d)$ elementweise Operationen.

<!--en:802bff53-->
- **Assoziativ, aber nicht kommutativ:** Die Klammerung spielt keine Rolle. Damit lassen sich die Zeitschritte parallel in einer Baumstruktur zusammenfassen.

<!--en:62fe75bb-->
::: {.slide title="Der parallele Verdopplungsscan nach Hillis und Steele"}

<!--en:20cd62bf-->
[Paralleler Präfix-Scan]{.kicker}

<!--en:bfe69a5b-->
- **Verdopplung des Strides (Hillis & Steele, 1986):** In Runde $r = 0, 1, \ldots, \lceil \log_2 T \rceil - 1$ verknüpfen alle Positionen $t > 2^r$ ihre Werte parallel mit denen an Position $t - 2^r$:
  $$\text{stride } s = 1 \to 2 \to 4 \to \cdots \to 2^{\lceil\log_2 T\rceil - 1}$$

<!--en:d4230553-->
- **Verdopplung und arbeitseffiziente Bäume:** Hillis und Steele benötigen $O(T \log_2 T)$ Verknüpfungen in $\lceil \log_2 T \rceil$ Runden. Blellochs Baumverfahren mit Aufwärts- und Abwärtsdurchlauf (Blelloch, 1990) benötigt $\sim 2T$ Verknüpfungen in $2\lceil \log_2 T \rceil$ Runden.

<!--en:5f503ad5-->
::: {.slide title="Implementierung von associative_scan und linear_scan" only="pytorch"}

<!--en:48bd9be4-->
[Aufbau des Scan-Codes]{.kicker}

<!--en:6c06d3df-->
`associative_scan` bildet die Slices `earlier = x[:, :-s]` und `later = x[:, s:]` mit jeweils verdoppeltem Stride $s = 1, 2, 4, \ldots$:

<!--en:4dcbf897-->
- **Anfangszustände (`h0`):** `linear_scan(a, b, h0)` berücksichtigt einen von null verschiedenen Anfangszustand $\mathbf{h}_0$ schon im ersten Schreibvektor ($\mathbf{b}_1 \leftarrow \mathbf{a}_1 \odot \mathbf{h}_0 + \mathbf{b}_1$), bevor der Präfix-Scan beginnt. So lässt sich der Zustand von Abschnitt zu Abschnitt weitergeben.

<!--en:858f2de0-->
::: {.slide title="Implementierung von linear_scan mit jax.lax.associative_scan" only="jax"}

<!--en:48bd9be4-2-->
[Aufbau des Scan-Codes]{.kicker}

<!--en:9f7af8cc-->
`linear_scan` übergibt die affine Verknüpfungsfunktion `combine` an die arbeitseffiziente XLA-Funktion `jax.lax.associative_scan`:

<!--en:4dcbf897-2-->
- **Anfangszustände (`h0`):** `linear_scan(a, b, h0)` berücksichtigt einen von null verschiedenen Anfangszustand $\mathbf{h}_0$ schon im ersten Schreibvektor ($\mathbf{b}_1 \leftarrow \mathbf{a}_1 \odot \mathbf{h}_0 + \mathbf{b}_1$), bevor der Präfix-Scan beginnt. So lässt sich der Zustand von Abschnitt zu Abschnitt weitergeben.

<!--en:cb638701-->
::: {.slide title="Vergleich des parallelen Scans mit der sequenziellen Berechnung"}

<!--en:eb538359-->
[Implementierung des Scans]{.kicker}

<!--en:c7a6206d-->
`linear_scan(a, b, h0)` berücksichtigt einen optionalen Anfangszustand $\mathbf{h}_0$ im ersten Schreibvektor ($\mathbf{b}_1 \leftarrow \mathbf{a}_1 \odot \mathbf{h}_0 + \mathbf{b}_1$) und führt den assoziativen Präfix-Scan entlang der Zeitachse aus:

<!--en:acd7ca6f-->
Selbst bei einer Länge, die keine Zweierpotenz ist ($T = 100$), stimmen paralleler Scan und sequenzielle Schleife bis auf den Rundungsfehler von `float32` überein ($\sim 10^{-7}$).

<!--en:4f904f63-->
::: {.slide title="Auswertung der minGRU auf WikiText-103 auf Byte-Ebene"}

<!--en:a94e4287-->
[Benchmark für Sprachmodelle]{.kicker}

<!--en:794a17d9-->
Der Benchmark mit 4 Blöcken in `SeqLM` (Fenster von $128$ Byte, $1{,}000$ AdamW-Schritte) vergleicht `GRU` ($d = 256$), `minGRU` ($d = 256$) und eine Variante mit angeglichener Parameteranzahl, `minGRU, d=302` ($3.81\text{M}$ Parameter):

<!--en:0b15c636-->
Bei gleicher Breite ($2.76\text{M}$ gegenüber $3.81\text{M}$ Parametern) liegt `minGRU` hinter `GRU`, und zwar um $\sim 0.05$ Nat pro Byte ($1.35$ gegenüber $1.30$). Bei gleicher Parameteranzahl (`minGRU, d=302`) sinkt der Abstand auf $\sim 0.03$ Nat pro Byte. Damit übertrifft das Modell die Attention-Baseline ($1.38$).

<!--en:a0640efe-->
::: {.slide title="Ablationsstudie zum Nutzen des Zustands durch Zurücksetzen"}

<!--en:96625095-->
[Einblick in das trainierte Modell]{.kicker}

<!--en:161af61f-->
Wie viel verdankt `minGRU` bei der Vorhersage dem Kontext in $\mathbf{h}_{t-1}$ und wie viel den Merkmalen des aktuellen Tokens im Residualstrom? Im trainierten Modell gilt versuchsweise bei jedem Schritt $\mathbf{h}_{t-1} = \mathbf{0}$ (`StateReset`, also $\mathbf{h}_t = \mathbf{b}_t$):

<!--en:6f59a5a9-->
- **Über 1.1 Nat pro Byte durch den rekurrenten Speicher:** Bei auf null gesetztem Zustand steigt der Validierungsverlust von **$1.35$** auf **$2.48$ Nat pro Byte**. Damit fällt das Modell auf das Niveau der empirischen Byte-Bigramm-Baseline zurück ($2.45$ Nat pro Byte).

<!--en:5a21db35-->
::: {.slide title="Wie das Training die Verteilung der Halbwertszeiten der Gatter verändert"}

<!--en:600183bc-->
[Zeitskalen der Gatter]{.kicker}

<!--en:7744a306-->
Die Update-Gatter $\mathbf{z}_t \in (0, 1)^d$ des ersten `minGRU`-Blocks auf einem Validierungsbatch vor und nach $1{,}000$ Trainingsschritten:

<!--en:52dd5563-->
- **Speicherung auf der Zeitskala von Wörtern:** Bei der Initialisierung ($\mathbf{b}_\textrm{z} = \mathbf{0}$) liegen nur $1\%$–$2\%$ der Gatterwerte über $0.9$ (Halbwertszeit $\ge 7$ Byte). Durch das Training steigt der Anteil auf mehr als das Zehnfache und erreicht $\sim 18\%$.

<!--en:e22d508e-->
- **Warum kaum ein Gatter über $0.99$ liegt (Halbwertszeit $69$ Byte):** In Fenstern von 128 Byte bestimmen vor allem die benachbarten Wörter und Wortgruppen den Verlust. Mit einem Bias von null liegen die Anfangswerte weit vom Bereich $0.99$–$0.999$ entfernt. §{sec_ssm} behandelt diese Einschränkung ausdrücklich.

<!--en:2eadb055-->
::: {.slide title="Asymptotische Komplexität von GRU, Scan und Attention"}

<!--en:5c137d8c-->
[Rechenaufwand, Tiefe und Speicherbedarf]{.kicker}

<!--en:dbe2df99-->
- **Sequenzielle `GRU`:** $\Theta(T d^2)$ FLOPs in $\Theta(T)$ aufeinanderfolgenden Schritten. Der Zustand bei der Generierung benötigt $\Theta(d)$ Speicher.

<!--en:3a2801f0-->
- **Paralleler `minGRU`-Scan:** $\Theta(T d^2)$ FLOPs für Projektionen bei einer Tiefe von $\Theta(\log_2 T)$. Der Speicherbedarf für den Zustand bei der Inferenz bleibt bei $\Theta(d)$.

<!--en:3e5f4fff-->
- **Schwelle bei kausaler Attention:** Die paarweisen Attention-Scores ($2T^2 d$) erfordern mehr Rechenarbeit als die Projektionen ($8Td^2$), sobald $\mathbf{T > 4d}$ gilt.

<!--en:587b593e-->
::: {.slide title="Gemessene Laufzeiten bis zu einer Sequenzlänge von 16 384"}

<!--en:0240f3fd-->
[Hardware-Benchmark]{.kicker}

<!--en:22e1273b-->
Laufzeit für Vorwärts- und Rückwärtsdurchlauf (ms) in `float16` bei Breite $d = 256$ und konstant $B \times T = 65{,}536$ Token pro Batch. Die Sequenzlänge $T$ variiert von $512$ bis $16{,}384$:

<!--en:65e987c8-->
::: {.slide title="Speichertransfers beim Verdopplungsscan im Vergleich zur quadratischen Attention" only="pytorch"}

<!--en:430cb597-->
[Skalierung in PyTorch]{.kicker}

<!--en:01a6b062-->
Bei konstant $B \times T = 65{,}536$ Token pro Batch verläuft die Laufzeitkurve einer idealen Schicht mit linearem Aufwand waagerecht:

<!--en:437c2a0c-->
- **Sequenzielle `GRU` (Tiefe $O(T)$):** Bei jeder Länge am langsamsten. Die Laufzeit steigt um den Faktor $>15\times$ von $T = 512$ bis $16{,}384$. Jede Verdopplung von $T$ verdoppelt die Zahl voneinander abhängiger Kernel-Schritte und halbiert die Parallelität $B$ innerhalb des Batchs.

<!--en:b207bae8-->
- **Kompilierte `minGRU` (Verdopplungsscan):** Benötigt $\lceil \log_2 T \rceil$ Speicherdurchläufe (von $9$ Runden bei $T = 512$ bis $14$ Runden bei $T = 16{,}384$). Die Laufzeit wächst um weniger als $50\%$, obwohl die Sequenz um den Faktor $32\times$ länger wird.

<!--en:8820fb37-->
- **Kausale `attention` (Aufwand $O(T^2 d)$):** Bis $T = 2{,}048$ bleibt die Laufzeit nahezu konstant, da Projektionen und fixer Zusatzaufwand dominieren. Bei etwa $T = 4{,}096$ schneidet die Kurve die des Verdopplungsscans. Danach nähert sich der Anstieg dem Faktor $2\times$ je Verdopplung der Länge. Gegenüber der `minGRU` beträgt der Laufzeitfaktor **$2.5\times$ bei $T = 16{,}384$**.

<!--en:aa6883e8-->
::: {.slide title="Arbeitseffizienter XLA-Scan im Vergleich zur fusionierten cuDNN-Attention" only="jax"}

<!--en:bb24bb81-->
[Skalierung in JAX]{.kicker}

<!--en:01a6b062-2-->
Bei konstant $B \times T = 65{,}536$ Token pro Batch verläuft die Laufzeitkurve einer idealen Schicht mit linearem Aufwand waagerecht:

<!--en:054a52bf-->
- **Sequenzielle `GRU` (Tiefe $O(T)$):** Bei jeder Länge am langsamsten. Die Laufzeit steigt um den Faktor $>15\times$ von $T = 512$ bis $16{,}384$. Jede Verdopplung von $T$ verdoppelt die Zahl sequenzieller Scan-Schritte und halbiert die Batchgröße $B$.

<!--en:4fa8a54c-->
- **Arbeitseffiziente `minGRU` (`jax.lax.associative_scan`):** Benötigt $\sim 2T$ elementweise Verknüpfungen für jede Sequenzlänge $T$. Die Laufzeit **bleibt konstant ($\sim 1.2\text{ ms}$), obwohl die Sequenz um den Faktor $32\times$ länger wird**. Der Scan ist bei jeder Länge schneller als Attention.

<!--en:facb1f20-->
- **cuDNN-Funktion `FusedAttention` (Aufwand $O(T^2 d)$):** Bis $T = 2{,}048$ bleibt die Laufzeit nahezu konstant. Sobald das quadratische paarweise Mischen dominiert, nähert sich der Anstieg dem Faktor $2\times$ je Verdopplung der Länge. Gegenüber der `minGRU` beträgt der Laufzeitfaktor **etwa $14\times$ bei $T = 16{,}384$**.

<!--en:54e3744e-->
::: {.slide title="Verständnisfragen zu Scan-Tiefe, Abgeschlossenheit und Laufzeitvergleich"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:bac06233-->
Drei Fragen zu linearen Rekurrenzen und parallelen Scans beantworten:

<!--en:403d9a05-->
1. Wie viele parallele Runden benötigt der Verdopplungsscan nach Hillis und Steele bei einer Sequenz der Länge $T = 16{,}384$ im Vergleich zu einer sequenziellen Schleife? Was ergibt die binäre Verknüpfung $(0.5, 1) \bullet (0.5, 2)$?

<!--en:dfa67805-->
2. Warum wird ein Scan für den Zustandsübergang $\mathbf{h}_t = \mathbf{A}_t \mathbf{h}_{t-1} + \mathbf{b}_t$ mit dicht besetzter Matrix ($\mathbf{A}_t \in \mathbb{R}^{d \times d}$) bei $d = 256$ zu teuer, obwohl die Matrizenmultiplikation assoziativ ist?

<!--en:5918dffe-->
3. In unserem Benchmark mit konstanter Tokenanzahl ($B \times T = 65{,}536$, $d = 256$) verdoppelt sich die Laufzeit der `GRU` mit jeder Verdopplung von $T$. Warum ist das so, obwohl die Gesamtzahl der FLOPs $B T d^2$ konstant bleibt?

<!--en:557b98b6-->
**Antworten:**

<!--en:efc0d2c6-->
1. $\log_2 16{,}384 = \mathbf{14\text{ rounds}}$ gegenüber $16{,}383$ sequenziellen Schritten. Die Verknüpfung ergibt $(0.5, 1) \bullet (0.5, 2) = (0.5 \times 0.5,\; 0.5 \times 1 + 2) = \mathbf{(0.25, 2.5)}$.

<!--en:42982fcb-->
2. Die Verkettung zweier dicht besetzter Übergänge erfordert das Matrixprodukt $\mathbf{A}_2 \mathbf{A}_1$ mit $O(d^3)$ FLOPs anstelle von $O(d)$ elementweisen Multiplikationen ($d^2 = 65{,}536\times$ so viel Arbeit pro Verknüpfung).

<!--en:86ea1fb2-->
3. Jeder Schritt der `GRU` wartet auf $\mathbf{h}_{t-1}$. Für Kernel-Ausführung und Synchronisierung fällt pro Schritt eine feste Latenz an. Verdoppelt sich $T$, verdoppelt sich daher die Latenz auf dem sequenziellen kritischen Pfad.

<!--en:a87e6336-->
::: {.slide title="Zusammenfassung zu linearen Rekurrenzen und assoziativen Scans"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:635ac7f1-->
| Verfahren oder Modell | Mathematische Definition | Eigenschaften für Skalierung und Modellierung |

<!--en:6918e6c9-->
| **Affine Vektorrekurrenz** | $\mathbf{h}_t = \mathbf{a}_t \odot \mathbf{h}_{t-1} + \mathbf{b}_t$ | Ergibt durch wiederholtes Einsetzen $\sum_{s \le t} (\prod_{r=s+1}^t \mathbf{a}_r) \odot \mathbf{b}_s$. Im Vorwärtsdurchlauf klingen die Beiträge genauso schnell ab wie die Gradienten im Rückwärtsdurchlauf. |

<!--en:4c1e8020-->
| **Minimale GRU (`MinGRU`)** | $\mathbf{z}_t = \sigma(\mathbf{W}_\textrm{z}\mathbf{x}_t + \mathbf{b}_\textrm{z})$, $\mathbf{b}_t = (1-\mathbf{z}_t)\odot\tilde{\mathbf{h}}_t$ | Gatter ohne $\mathbf{h}_{t-1}$ ($2d^2$ Parameter). Bei gleicher Modellgröße um $0.03$ Nat pro Byte schlechter als die GRU. |

<!--en:36fa3d1f-->
| **Assoziative Verknüpfung ($\bullet$)** | $(\mathbf{a}_1, \mathbf{b}_1) \bullet (\mathbf{a}_2, \mathbf{b}_2) = (\mathbf{a}_1 \odot \mathbf{a}_2, \mathbf{a}_2 \odot \mathbf{b}_1 + \mathbf{b}_2)$ | Fasst jeden Block von Schritten zu einem Paar der Breite $d$ zusammen, mit Aufwand $O(d)$. |

<!--en:031dfb5f-->
| **Verdopplungsscan und Baumscan** | $\lceil \log_2 T \rceil$ gegenüber $2\lceil \log_2 T \rceil$ parallelen Runden | $O(T \log T)$ gegenüber $O(T)$ Verknüpfungen. Nahezu konstante gemessene Laufzeit bis $T = 16{,}384$. |

<!--en:d24f1f6b-->
| **Schwelle bei Attention** | $8Td^2$ FLOPs für Projektionen gegenüber $2T^2 d$ FLOPs für Attention-Scores | Die quadratischen Attention-Kosten dominieren ab $T \gtrsim 4d$. Pro Token wachsen sie linear mit $T$. |
