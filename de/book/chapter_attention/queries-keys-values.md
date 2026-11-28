<!--en:af0fe3df-->
# Abfragen, Schlüssel und Werte

<!--en:c52a8f04-->
Ein rekurrentes neuronales Netz (engl. recurrent neural network, RNN) (:numref:`sec_rnn`) verdichtet alles bisher Gelesene zu einem Zustandsvektor fester Größe. Welche Einzelheiten es für eine spätere Vorhersage braucht, weiß es zu diesem Zeitpunkt noch nicht. Liest etwa ein rekurrentes Modell mit einem versteckten Zustand der Dimension 768 eine Folge von 1024 Token, muss es alle 1024 Positionen in 768 Zahlen zusammenfassen.

<!--en:c1071ac6-->
Eine Datenbank vermeidet diese vorzeitige Verdichtung. Sie speichert jeden Datensatz unverändert und entscheidet erst bei einer Abfrage (engl. query), welche Daten sie abruft. Das exakte Nachschlagen in einer Datenbank ist jedoch unstetig. Betrachten wir die Paare aus Schlüssel (engl. key) und Wert (engl. value) $\mathcal{D}=\{(2,10),(5,20)\}$. Die exakte Abfrage $q=5$ liefert $20$, während $q=4$ zu keinem Schlüssel passt. Für das Training mit Gradientenabstieg ersetzen wir das harte Nachschlagen durch *weiches* Nachschlagen (engl. soft lookup). Dabei erhält jeder Schlüssel einen Score, etwa das negative Abstandsquadrat $a(q,k)=-(q-k)^2$. Die Softmax-Funktion rechnet die Scores in Gewichte für eine Konvexkombination um. Bei $q=4$ erhalten die Schlüssel $2$ und $5$ die Scores $-4$ und $-1$. Daraus ergeben sich die Gewichte $e^{-4}/(e^{-4}+e^{-1}) \approx 0.05$ und $0.95$ und der gewichtete Wert $0.05 \cdot 10 + 0.95 \cdot 20 \approx 19.5$. Die Ausgabe hängt glatt von der Abfrage ab. Das Verfahren funktioniert für beliebig viele gespeicherte Datensätze und kommt vor der Abfrage ohne Flaschenhals fester Größe aus.

<!--en:00de58de-->
Dieses differenzierbare Nachschlagen in Schlüssel-Wert-Paaren ist der *Attention-Mechanismus* :cite:`Bahdanau.Cho.Bengio.2014`. Zunächst formulieren wir das Attention-Pooling für Abfragen, Schlüssel und Werte in Vektorform. Dann untersuchen wir seinen klassischen nichtparametrischen Vorläufer, die Nadaraya-Watson Kernel-Regression (engl. Nadaraya–Watson kernel regression). Dort bestimmt ein einziger Parameter, die Bandbreite (engl. bandwidth), die Schärfe des Nachschlagens.

<!--en:347f47f9-->
## Attention als differenzierbares Nachschlagen

<!--en:1a70b45a-->
Sei $\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1),
\ldots, (\mathbf{k}_m, \mathbf{v}_m)\}$ eine Datenbank mit $m$ Schlüssel-Wert-Paaren, deren Schlüssel $\mathbf{k}_i \in \mathbb{R}^{d}$ und deren Werte $\mathbf{v}_i \in
\mathbb{R}^{d_v}$ sind. Für eine Abfrage $\mathbf{q} \in \mathbb{R}^{d}$ liefert das *Attention-Pooling* eine gewichtete Summe der Werte:

<!--en:7cf1aaaf-->
Dabei gilt $\alpha(\mathbf{q},\mathbf{k}_i) \ge 0$ und $\sum_{i=1}^m
\alpha(\mathbf{q},\mathbf{k}_i) = 1$. Die Ausgabe ist somit eine Konvexkombination der Werte. Sie interpoliert zwischen exaktem Nachschlagen (One-Hot-Gewichtsvektor) und Average-Pooling (einheitliche Gewichte $1/m$).

<!--en:85f808ce-->
Um beliebige reelle Kompatibilitäts-Scores $a(\mathbf{q},
\mathbf{k}_i)$ in Gewichte für eine Konvexkombination umzurechnen, verwenden wir die Softmax-Funktion (:numref:`sec_softmax`):

<!--en:28d893ff-->
Wegen $\alpha_i = 1 / \big(1 + \sum_{j \neq i} e^{a_j - a_i}\big)$ hängen die Gewichte nur von den Score-Differenzen $a_j - a_i$ ab. Sie ändern sich also nicht, wenn wir zu den Scores aller Schlüssel dieselbe Konstante addieren. Sowohl die Softmax-Funktion als auch die Linearkombination sind nach $\mathbf{q}$, $\mathbf{k}_i$ und $\mathbf{v}_i$ differenzierbar.

<!--en:4ba4e9fd-->
Als konkretes Vektorbeispiel betrachten wir $a(\mathbf{q}, \mathbf{k}) = -\|\mathbf{q}

<!--en:884264f4-->
- \mathbf{k}\|^2$ mit der Abfrage $\mathbf{q} = (1, 1)$, den Schlüsseln $\mathbf{k}_1 = (1,
1)$, $\mathbf{k}_2 = (2, 1)$, $\mathbf{k}_3 = (0, 0)$ und den Werten $\mathbf{v}_1 = (1, 3)$, $\mathbf{v}_2 = (0, 0)$, $\mathbf{v}_3 = (4, 1)$. Die Scores betragen $0$, $-1$ und $-2$. Daraus ergeben sich die Softmax-Gewichte $(1, e^{-1}, e^{-2}) / (1 + e^{-1} + e^{-2}) \approx (0.67, 0.24, 0.09)$ und die Ausgabe

<!--en:b5e43445-->
Zwei Eigenschaften lassen sich an :numref:`fig_qkv` ablesen. Erstens stimmt $\mathbf{k}_1$ exakt mit $\mathbf{q}$ überein, erhält aber nur zwei Drittel des Gewichts. Multiplizieren wir die Scores mit $10$, ergeben sich $0, -10, -20$, und $99.995\%$ des Gewichts entfallen auf $\mathbf{k}_1$. Die Größenordnung der Scores bestimmt also, wie stark die Softmax-Funktion einen einzelnen Schlüssel bevorzugt. Zweitens trägt $\mathbf{v}_2 = (0, 0)$ zur Summe null bei. Sein Gewicht von $0.24$ verringert dennoch den Anteil der anderen Werte und zieht so die Ausgabe zum Ursprung.

<!--en:94d3274c-->
![Weiches Nachschlagen für die Abfrage $\mathbf{q} = (1, 1)$ in drei Schlüssel-Wert-Paaren mit den Schlüsseln $(1, 1)$, $(2, 1)$, $(0, 0)$ und den Werten $(1, 3)$, $(0, 0)$, $(4, 1)$. Aus den Scores $s_i = -\|\mathbf{q} - \mathbf{k}_i\|^2$ entstehen Softmax-Gewichte $\alpha_i$ mit Summe eins (links). Die Ausgabe $\mathbf{o} \approx (1.0, 2.1)$ ist die gewichtete Summe der Werte. Sie liegt im Inneren des Dreiecks, dessen Ecken die Werte bilden (rechts).](../img/mdl-attention-soft-lookup.svg)

<!--en:34483fb8-->
Die Attention-Regel :eqref:`eq_attention_pooling` gilt unverändert für jede Anzahl $m$ von Datensätzen. So kann eine einzige Schicht beliebig lange Sequenzen verarbeiten, ohne sie zu einem Zustand fester Größe zu verdichten. Die Kosten verlagern sich auf den Speicher. Eine rekurrente Schicht der Breite 768 hält 768 Zahlen im Speicher. Eine Attention-Schicht von GPT-2 small mit 12 Köpfen ($d = 64$ pro Kopf) speichert bei 1024 Token pro Kopf $2 \times 1{,}024 \times 64 = 131{,}072$ Einträge für Schlüssel und Werte. Über alle 12 Köpfe sind das rund $1.6 \times 10^6$ Zahlen.

<!--en:f957e553-->
## Attention-Pooling mit festen Kerneln

<!--en:265b8c11-->
Lange vor dem Deep Learning verwendete die *Nadaraya-Watson Kernel-Regression* :cite:`Nadaraya.1964,Watson.1964` bereits :eqref:`eq_attention_pooling` für die nichtparametrische Regression. Die Trainingseingaben dienen als Schlüssel, die Trainingslabel als Werte. Ein fester Ähnlichkeitskernel liefert die Scores.

<!--en:d880db0d-->
### Kernel und der Nadaraya-Watson-Schätzer

<!--en:bf1acd80-->
Betrachten wir einen nichtnegativen Ähnlichkeitskernel $K(\mathbf{q}, \mathbf{k})$, etwa den Gauß-Kernel mit *Bandbreite* $\sigma > 0$:

<!--en:1797c97f-->
Durch Normalisierung über alle Schlüssel erhalten wir den *Nadaraya-Watson-Schätzer*:

<!--en:515d2a21-->
Für jeden strikt positiven Kernel entspricht :eqref:`eq_nadaraya-watson` der Softmax-Attention aus :eqref:`eq_softmax_attention` mit dem Score $a(\mathbf{q}, \mathbf{k})
= \log K(\mathbf{q}, \mathbf{k})$. Für den Gauß-Kernel gilt $a(\mathbf{q},
\mathbf{k}) = -\|\mathbf{q} - \mathbf{k}\|^2 / (2\sigma^2)$. Bei $2\sigma^2 = 1$ erhalten wir unser Eingangsbeispiel. Unter Regularitätsbedingungen an die Dichte und die Zielfunktion gilt für $m \to \infty$ und $\sigma \to 0$ bei geeigneter Rate: $f(\mathbf{q})$ konvergiert gegen den bedingten Erwartungswert $\mathbb{E}[y \mid \mathbf{q}]$ :cite:`mack1982weak`.

<!--en:85398a97-->
### Die Bandbreite

<!--en:db1f1f8b-->
Wie die Bandbreite $\sigma$ den Schätzer beeinflusst, untersuchen wir anhand von $n = 40$ verrauschten Trainingsbeobachtungen. Wir ziehen sie gemäß $y = 2\sin(x) + x + \epsilon$ mit $\epsilon
\sim \mathcal{N}(0, 1)$ auf $[0, 5)$ und berechnen die Vorhersagen auf einem engmaschigen Gitter.

<!--en:e296f2df-->
Die Implementierung berechnet alle paarweisen Abstände zwischen Abfragen und Schlüsseln und wertet den Gauß-Kernel aus. Sie normalisiert die resultierende Matrix für jede Abfrage zeilenweise und multipliziert sie mit `y_train`.

<!--en:a5e6c63a-->
Für $\sigma \to 0$ konzentriert sich das gesamte Gewicht auf den nächstgelegenen Trainingspunkt (Interpolation mit dem nächsten Nachbarn). Für $\sigma \to \infty$ nähern sich alle Gewichte $1/n$, sodass die Ausgabe gegen den globalen Stichprobenmittelwert strebt. Im Folgenden zeigen wir die Kurven für $\sigma \in \{0.1, 0.5,
2.0\}$.

<!--en:7b82baf1-->
Bei kleiner Bandbreite ($\sigma = 0.1$) passt sich der Schätzer zu stark an einzelne verrauschte Label an. Bei großer Bandbreite ($\sigma = 2.0$) glättet er so stark, dass sich die Kurve einer horizontalen Geraden nähert. Beim mittleren Wert $\sigma = 0.5$ folgt die Schätzkurve dem wahren Verlauf genau, außer an den Rändern, wo alle Nachbarn auf einer Seite liegen. In der Praxis lässt sich $\sigma$ durch Leave-one-out-Kreuzvalidierung wählen. Dazu setzen wir die Diagonale der Gewichtsmatrix auf null und normalisieren die Zeilen erneut (Übung :exref:`ex_queries-keys-values-learned-metric`).

<!--en:14809aef-->
Allgemeiner wirkt $1/\sigma^2$ als inverse Temperatur. Mit ihr multiplizieren wir die ursprünglichen Scores $-\tfrac{1}{2}(q - k)^2$ vor Anwendung der Softmax-Funktion. Skalieren wir die Scores nach oben, vergrößern sich ihre paarweisen Abstände, und das Nachschlagen nähert sich einer One-Hot-Auswahl. Skalieren wir sie nach unten, schrumpfen die Abstände, und die Softmax-Gewichte nähern sich denen einer gleichmäßigen Mittelung. Dieselbe Temperaturskalierung begegnet uns bei der Skalarprodukt-Attention (:numref:`sec_attention-scoring-functions`).

<!--en:45db0849-->
## Attention-Gewichte als Diagnosemittel

<!--en:3e3b98d5-->
Jede Zeile der Attention-Matrix $\mathbf{A}$ ist eine Wahrscheinlichkeitsverteilung über die Schlüssel. Stellen wir $\mathbf{A}$ als Heatmap dar, erkennen wir, welche Schlüssel die einzelnen Abfragen heranziehen.

<!--en:d3a79b80-->
Bei $\sigma = 0.1$ konzentriert sich das Gewicht in einem schmalen diagonalen Band um die jeweils nächstgelegenen Trainingseingaben. Bei $\sigma = 2.0$ verteilt es sich breit über den Datensatz. Kein Gewicht übersteigt etwa das Doppelte des einheitlichen Gewichts $1/40 = 0.025$. Die größten Gewichte von etwa $0.05$ treten bei Abfragen an den Rändern auf. Deren weit entfernte Schlüssel erhalten nur etwa $0.002$.

<!--en:48cfc3c4-->
Bei der Betrachtung von Attention-Heatmaps in tiefen Netzen sind zwei Einschränkungen zu beachten:

<!--en:12500eef-->
1. **Gewichte zeigen, wohin eine Schicht blickt, aber nicht, was sie überträgt:** Ein Schlüssel kann selbst dann ein großes Attention-Gewicht erhalten, wenn sein Wertvektor null ist (wie bei $\mathbf{v}_2 = (0, 0)$ in unserem Vektorbeispiel) oder im Nullraum nachgelagerter Projektionen liegt. Die kausale Rolle eines Kopfes lässt sich nur durch Experimente mit gezielten Eingriffen überprüfen (:numref:`sec_what-attention-computes`).

<!--en:b73e3f57-->
2. **Feste Kernel geben eine unveränderliche Ähnlichkeitsmetrik vor:** In :eqref:`eq_nadaraya-watson` verwendet jede Abfrage denselben euklidischen Abstand im Eingaberaum. Bei der Modellierung von Sequenzen hängt die Relevanz jedoch vom Kontext und von der Aufgabe ab. Ein Verb kann etwa sein syntaktisches Subjekt benötigen, obwohl mehrere Token dazwischenliegen. Lernende Attention-Schichten projizieren die Eingaben daher in trainierbare Räume für Abfragen, Schlüssel und Werte, bevor sie die Scores berechnen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:940d33e8-->
Attention-Pooling berechnet eine Konvexkombination von Wertvektoren $\sum_i \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i$. Ihre Softmax-Gewichte $\alpha(\mathbf{q}, \mathbf{k}_i)$ ermöglichen einen glatten Übergang zwischen exaktem Nachschlagen in einem Wörterbuch und Average-Pooling mit einheitlichen Gewichten. Anders als ein rekurrenter Zustand fester Größe ermöglicht Attention direkten Zugriff auf alle gespeicherten Schlüssel-Wert-Paare. Die Regel dafür ist unabhängig von der Anzahl $m$ der Paare, der Arbeitsspeicherbedarf wächst linear mit $m$. Die Nadaraya-Watson Kernel-Regression entspricht einem Attention-Pooling mit festem euklidischem Ähnlichkeitskernel. Seine Bandbreite $\sigma$ wirkt als Softmax-Temperatur und steuert den Zielkonflikt zwischen Verzerrung und Varianz. Attention-Heatmaps veranschaulichen das Alignment zwischen Abfragen und Schlüsseln. Um über einen festen Abstand im Eingaberaum hinauszugehen, brauchen wir trainierbare Projektionen und erlernte Score-Funktionen.

<!--en:2294243a-->
## Übungen

<!--en:269b9cdf-->
1. **Mittelwert zweier Werte.** Die Schlüssel $\mathbf{k}_1, \ldots,
   \mathbf{k}_m$ seien orthonormal. Der Score sei $a(\mathbf{q},
   \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$ mit den Gewichten aus :eqref:`eq_softmax_attention`.
    1. Betrachten Sie die Abfrage $\mathbf{q} = c\,(\mathbf{k}_a + \mathbf{k}_b)$ mit $c > 0$. Berechnen Sie alle $m$ Gewichte und zeigen Sie, dass die Ausgabe gegen $\tfrac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$ strebt, wenn $c \to \infty$.
    1. Wie groß muss $c$ sein, damit die beiden ausgewählten Schlüssel für $m = 10$ mindestens 99% des Gesamtgewichts erhalten?
    1. Die Schlüssel seien nun orthogonal mit $\|\mathbf{k}_b\| =
       2\,\|\mathbf{k}_a\|$. Finden Sie eine Abfrage, die wieder über $\mathbf{v}_a$ und $\mathbf{v}_b$ mittelt. Geben Sie an, was die Abfrage $\mathbf{q} = c\,(\mathbf{k}_a + \mathbf{k}_b)$ für $c \to \infty$ stattdessen liefert.

    *Nach Stanford CS224N, [Aufgabenblatt 4](https://web.stanford.edu/class/cs224n/assignments_w25/a4.pdf), Aufgaben 1(b) und 1(c).*
   :exlabel:`ex_queries-keys-values-average-two-values`

<!--en:60f4a429-->
1. [code] **Effektive Anzahl der Schlüssel.** Für die Gewichte $\alpha_i(q)$ einer Abfrage definieren wir die effektive Anzahl der Schlüssel als $n_{\textrm{eff}}(q) = 1 / \sum_i \alpha_i(q)^2$.
    1. Die Werte seien $y_i = g(x_i) + \epsilon_i$ mit unabhängigem Rauschen der Varianz $s^2$. Halten Sie die Schlüssel fest. Zeigen Sie, dass die Schätzung aus :eqref:`eq_nadaraya-watson` an der Stelle $q$ die Varianz $s^2 / n_{\textrm{eff}}(q)$ hat.
    1. Berechnen Sie $n_{\textrm{eff}}$ für jede Abfrage in `x_val` aus den Gewichten, die `nadaraya_watson` zurückgibt. Wiederholen Sie dies für jede Bandbreite in `sigmas` und plotten Sie die drei Kurven. Ziehen Sie dann das Rauschen in `y_train` bei festem `x_train` insgesamt 200-mal neu. Vergleichen Sie die Standardabweichung der Schätzung für jede Abfrage mit $1 / \sqrt{n_{\textrm{eff}}(q)}$.

<!--en:1878df0a-->
1. [code] **Der Gradient der Attention.** ● Es gelte $a(\mathbf{q},
   \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$ und $\mathbf{k}_i =
   \mathbf{v}_i$ für $i = 1, \ldots, m$. Bezeichnen Sie mit $p(\mathbf{k}_i;
   \mathbf{q})$ die Verteilung über die Schlüssel, die sich aus der Softmax-Normalisierung :eqref:`eq_softmax_attention` ergibt.
    1. Beweisen Sie die Identität
       $$\nabla_{\mathbf{q}} \mathop{\textrm{Attention}}(\mathbf{q}, \mathcal{D}) = \textrm{Cov}_{p(\mathbf{k}_i; \mathbf{q})}[\mathbf{k}_i].$$
    1. Folgern Sie, dass diese Jacobi-Matrix positiv semidefinit ist. Bestimmen Sie, für welche Datenbanken sie die Nullmatrix ist.
    1. Sei $\mathbf{q} = c\,\mathbf{u}$ für eine feste Richtung $\mathbf{u}$, in der genau ein Schlüssel den höchsten Score $\mathbf{u}^\top \mathbf{k}_i$ erreicht. Bestimmen Sie den Grenzwert der Jacobi-Matrix für $c \to \infty$. Was bedeutet er für das Training der Abfrage mit Gradientenabstieg?
    1. Überprüfen Sie die Identität und den Grenzwert durch automatisches Differenzieren an einer zufällig erzeugten Datenbank mit $m = 8$ und $d = 4$.

<!--en:c06772d1-->
1. [code] **Weiches Nachschlagen als Retrieval.** ● Erstellen Sie eine Datenbank mit $m$ Schlüsseln. Ziehen Sie diese gleichverteilt von der Einheitssphäre in $\mathbb{R}^{16}$. Verwenden Sie als Wert $\mathbf{v}_i$ den One-Hot-Vektor für den Index $i$. Gewichten Sie die Werte mit dem Gauß-Kernel der Bandbreite $\sigma$ und normalisieren Sie wie in :eqref:`eq_nadaraya-watson`. Die Ausgabe ist dann der Gewichtsvektor. Verwenden Sie als Abfrage eine verrauschte Kopie $\mathbf{q} = \mathbf{k}_i +
   \boldsymbol{\epsilon}$ eines gespeicherten Schlüssels mit $\boldsymbol{\epsilon} \sim
   \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})$.
    1. Zeigen Sie, dass für jede Bandbreite $\sigma > 0$ derjenige Schlüssel das größte Gewicht erhält, der der Abfrage am nächsten liegt. Die Bandbreite bestimmt also nicht, welcher Index gewinnt. Sie beeinflusst aber, wie viel Gewicht auf den Gewinner entfällt.
    1. Ohne Rauschen erhält der richtige Schlüssel das Gewicht
       $$\alpha_i = \frac{1}{1 + \sum_{j \neq i} \exp\big(-\|\mathbf{k}_i - \mathbf{k}_j\|^2 / 2\sigma^2\big)}.$$
       Verwenden Sie für zufällige Einheitsvektoren die Näherung $\mathbf{k}_i^\top \mathbf{k}_j \approx 0$, sodass $\|\mathbf{k}_i - \mathbf{k}_j\|^2 \approx 2$ gilt. Leiten Sie die Bandbreite her, bei der $\alpha_i$ auf $0.9$ fällt, und zwar für $m = 20$ und für $m = 200$. Wie muss $\sigma$ mit wachsendem $m$ schrumpfen?
    1. Damit die nächste Schicht die Ausgabe nutzen kann, muss der größte Teil des Gewichts auf einen einzigen Schlüssel entfallen. Der größte Einzelanteil allein genügt nicht. Werten Sie den Abruf als erfolgreich, wenn $\alpha_i \geq 0.5$ gilt. Variieren Sie $\tau$ von $0$ bis $1$ für $m = 20$ und $m = 200$. Verwenden Sie für jedes $m$ die Bandbreite aus (b) und plotten Sie die Abrufgenauigkeit in Abhängigkeit von $\tau$. Berechnen Sie die Gewichte mit der Softmax-Funktion aus :eqref:`eq_softmax_attention` für die Scores $-\|\mathbf{q} - \mathbf{k}_j\|^2 / 2\sigma^2$. Erklären Sie, warum ein größeres $m$ bei festem $\tau$ die Genauigkeit senkt, obwohl der Abstand von der Abfrage zum richtigen Schlüssel gleich bleibt.

<!--en:c68ac9ce-->
1. [extended] **Eine erlernte Metrik.** ● Die Kernel-Regression mit erlerntem Abstand :cite:`Weinberger.Tesauro.2007` ersetzt das feste Ähnlichkeitsmaß aus :eqref:`eq_nadaraya-watson` durch ein erlerntes. Ziehen Sie $n = 200$ Eingaben $\mathbf{x}_i$ gleichverteilt aus $[0, 5]^5$ und erzeugen Sie die Label $y_i =
   2\sin(x_{i1}) + x_{i1} + \epsilon_i$ mit standardnormalverteiltem Rauschen. Damit ist nur die erste Koordinate relevant. Berechnen Sie den Score einer Abfrage für einen Schlüssel mit
   $$a(\mathbf{q}, \mathbf{k}) = -\tfrac{1}{2} \|\mathbf{W}(\mathbf{q} - \mathbf{k})\|^2, \qquad \mathbf{W} = \textrm{diag}(w_1, \ldots, w_5),$$
   normalisieren Sie wie in :eqref:`eq_softmax_attention` und passen Sie $\mathbf{W}$ durch Minimierung des Leave-one-out-Fehlers an. Dabei bleibt bei der Schätzung an der Stelle $\mathbf{x}_i$ das Paar $(\mathbf{x}_i, y_i)$ unberücksichtigt. Messen Sie den Testfehler auf $1000$ neuen Eingaben anhand der rauschfreien Funktion.
    1. Passen Sie den isotropen Kernel $w_1 = \cdots = w_5 = 1/\sigma$ durch eine Gittersuche über $\sigma$ an.
    1. Passen Sie ausgehend von der isotropen Lösung die fünf Skalenfaktoren mit Gradientenabstieg an. Vergleichen Sie die erlernten Skalenfaktoren mit der Relevanz der jeweiligen Koordinate. Vergleichen Sie den Testfehler mit dem des isotropen Kernels und dem eines Orakels, das nur die erste Koordinate bei optimaler Bandbreite verwendet.
    1. Vergleichen Sie die Leave-one-out-Fehler der erlernten Metrik und der Orakelmetrik mit ihren Testfehlern. Wiederholen Sie den Vergleich bei $n = 1000$. Was sagt das Ergebnis darüber aus, wie gut sich ein Ähnlichkeitsmaß aus einer endlichen Stichprobe lernen lässt?
    1. Eine allgemeine Matrix $\mathbf{W} \in \mathbb{R}^{5 \times 5}$ bildet Abfragen und Schlüssel durch dieselbe lineare Abbildung ab. Erlernte Attention verwendet dagegen zwei separate Matrizen $\mathbf{W}_q$ und $\mathbf{W}_k$. Nennen Sie zwei Eigenschaften des Scores $a$, die eine gemeinsame Abbildung garantiert und die bei separaten Abbildungen verloren gehen.
   :exlabel:`ex_queries-keys-values-learned-metric`

<!--en:afed111e-->
[Diskussion](https://d2l.discourse.group/t/36768)

<!--en:8f9a1575-->
Abfragen, Schlüssel und Werte: von der nichtparametrischen Glättung zur Attention<br>**Differenzierbares Nachschlagen · Konvexkombinationen · Nadaraya-Watson Kernel-Regression · Bandbreite als Temperatur**

<!--en:19722da5-->
::: {.slide title="Warum rekurrente Zustände fester Größe frühen Kontext verlieren"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:ba2c0c99-->
- **Kapazitätsunterschied an einem Beispiel:** Liest ein Modell der Größe von GPT-2 small ($d = 768$) eine Folge von $n = 1{,}024$ Token, enthalten die Tokendarstellungen $1{,}024 \times 768 = 786{,}432$ Zahlen. Ein rekurrenter Zustand speichert nur $768$ Zahlen.

<!--en:e16a9f13-->
- **Vorzeitige Zusammenfassung:** Ein RNN schreibt $\mathbf{h}_t$ in Schritt $t$, ohne zu wissen, ob es das Token $t$ noch $900$ Schritte später braucht.

<!--en:0c1cf627-->
- **Alternative Datenbank:** Alle $m$ Datensätze belegen unverändert $O(md)$ Speicher. Die relevanten Werte werden erst abgerufen, wenn eine **Abfrage** eintrifft.

<!--en:3cf4ad1d-->
::: {.slide title="Warum sich exaktes Nachschlagen in einer Datenbank nicht trainieren lässt"}

<!--en:aaa3d47e-->
[Vom harten zum weichen Nachschlagen]{.kicker}

<!--en:71d62326-->
- **Hartes Nachschlagen ist unstetig:** Für $\mathcal{D} = \{(2, 10), (5, 20)\}$ liefert die exakte Abfrage $q = 5$ den Wert $20$. Für $q = 4$ gibt es keinen Treffer (bei Wahl des nächsten Schlüssels springt die Ausgabe am Mittelpunkt $q = 3.5$).

<!--en:f70c40a8-->
- **Gradient fast überall null:** Eine stückweise konstante Treppenfunktion ist zwischen ihren Sprüngen flach. Ihre Ableitung nach Abfragen und Schlüsseln ist daher fast überall null.

<!--en:6e412a61-->
- **Differenzierbare Relaxation:** *Jeder* Schlüssel erhält einen Score für die Abfrage. Die Softmax-Funktion liefert Gewichte für einen glatt variierenden Mittelwert aller gespeicherten Werte.

<!--en:2632b703-->
::: {.slide title="Rechenbeispiel: weiches Nachschlagen mit zwei skalaren Datensätzen"}

<!--en:a94b2d66-->
[Weiches Nachschlagen mit Skalaren]{.kicker}

<!--en:d404a4e6-->
- **Negatives Abstandsquadrat als Score:** Für die Abfrage $q = 4$ erhalten die Schlüssel $k_1 = 2$ und $k_2 = 5$ die Scores $a_1 = -(4-2)^2 = -4$ und $a_2 = -(4-5)^2 = -1$.

<!--en:18b31f9b-->
- **Normalisierung mit der Softmax-Funktion:** Exponenzieren und Teilen durch $e^{-4} + e^{-1}$ ergibt die Gewichte $\alpha_1 \approx 0.047$ und $\alpha_2 \approx 0.953$.

<!--en:bb71774c-->
- **Konvexkombination der Werte:**
  $$\textrm{Attention}(q=4, \mathcal{D}) = 0.047(10) + 0.953(20) \approx 19.53$$

<!--en:76aca579-->
::: {.slide title="Attention-Pooling für vektorwertige Datensätze"}

<!--en:6f14e397-->
[Mathematische Formulierung]{.kicker}

<!--en:b478bf3a-->
- **Attention-Pooling:** Für $m$ Datensätze $\mathcal{D} = \{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^m$ mit $\mathbf{q}, \mathbf{k}_i \in \mathbb{R}^d$ und $\mathbf{v}_i \in \mathbb{R}^{d_v}$ gilt:
  $$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i)\,\mathbf{v}_i \in \mathbb{R}^{d_v}$$

<!--en:662164db-->
- **Simplex-Nebenbedingung:** Es gilt $\alpha(\mathbf{q}, \mathbf{k}_i) \ge 0$ und $\sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) = 1$. Damit liegt die Ausgabe in der konvexen Hülle von $\{\mathbf{v}_i\}$.

<!--en:cc17b57a-->
- **Grenzfälle der Interpolation:** One-Hot-Gewichte $\mathbf{e}_{i^*}$ ergeben exaktes Nachschlagen, einheitliche Gewichte $1/m$ ergeben Average-Pooling.

<!--en:5f49820d-->
::: {.slide title="Von Kompatibilitäts-Scores zu Softmax-Gewichten"}

<!--en:50e125bc-->
[Softmax-Normalisierung]{.kicker}

<!--en:e5d08fee-->
- **Softmax-Funktion:** Beliebige reelle Kompatibilitäts-Scores $a(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$ ergeben einen Gewichtsvektor auf dem Simplex:
  $$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))}$$

<!--en:075e24ce-->
- **Verschiebungsinvarianz:** Eine zu allen Scores addierte schlüsselunabhängige Konstante $c(\mathbf{q})$ kürzt sich heraus. Nur die Differenzen $a_j - a_i$ bestimmen die Gewichte.

<!--en:1e29dadc-->
- **Numerische Stabilität:** Nach Subtraktion von $M = \max_j a_j$ sind alle Exponenten höchstens $0$. Das verhindert einen Überlauf oder den Ausdruck $0/0$ infolge eines Unterlaufs.

<!--en:4c94dd5e-->
::: {.slide title="Rechenbeispiel: weiches Nachschlagen mit Vektoren in der Ebene"}

<!--en:3a19eee9-->
[Nachschlagen mit zweidimensionalen Vektoren]{.kicker}

<!--en:f017ce4e-->
- **Abfrage $\mathbf{q} = (1, 1)$ bei 3 Datensätzen:** Mit $s_i = -\|\mathbf{q} - \mathbf{k}_i\|^2$ erhalten die Schlüssel $(1,1)$, $(2,1)$ und $(0,0)$ die Scores $0, -1, -2$ und damit die Gewichte $(0.67, 0.24, 0.09)$.

<!--en:69cf01b8-->
- **Begrenztes Gewicht für den Treffer:** Selbst bei exakter Übereinstimmung $\mathbf{k}_1 = \mathbf{q}$ entfallen nur $67\%$ des Gewichts auf diesen Schlüssel, solange die Scores unverändert bleiben.

<!--en:7133a94c-->
- **Ein Nullvektor schwächt die anderen Beiträge:** Datensatz 2 hat den Wert $\mathbf{v}_2 = (0, 0)$. Er trägt null zur Summe bei, bindet aber $24\%$ des Gewichts und zieht $\mathbf{o} \approx (1.03, 2.09)$ zum Ursprung.

<!--en:f8207474-->
::: {.slide title="Geometrische Sicht: konvexe Hülle der gespeicherten Werte"}

<!--en:5931bd17-->
[Geometrie der Attention]{.kicker}

<!--en:df99adcc-->
- **Die Ausgabe bleibt in der konvexen Hülle:** Wegen $\alpha_i \ge 0$ und $\sum_{i=1}^m \alpha_i = 1$ liegt die Ausgabe $\mathbf{o}$ stets in der konvexen Hülle von $\{\mathbf{v}_1, \ldots, \mathbf{v}_m\}$ (dem Dreieck mit den Ecken $(1,3)$, $(0,0)$ und $(4,1)$).

<!--en:796dea6f-->
- **Skalieren verstärkt das größte Gewicht:** Die Multiplikation aller Scores mit $10$ ergibt $0, -10, -20$. Damit entfallen $99.995\%$ des Gewichts auf $\mathbf{k}_1$, und $\mathbf{o}$ rückt fast bis an den Eckpunkt $\mathbf{v}_1 = (1, 3)$.

<!--en:c6e49d00-->
- **Keine Extrapolation:** Softmax-Attention interpoliert zwischen gespeicherten Werten. Die Ausgabe kann nie außerhalb ihrer konvexen Hülle liegen.

<!--en:17b93f41-->
![Weiches Nachschlagen bildet Scores auf Gewichte im Simplex ab. Die Ausgabe liegt im Inneren des Dreiecks, das die Werte bilden.](../img/mdl-attention-soft-lookup.svg){.nostretch width=100%}

<!--en:06a0b949-->
::: {.slide title="Unabhängig von der Länge, linear im Speicherbedarf"}

<!--en:d94bd695-->
[Speicherbedarf und Zugriff]{.kicker}

<!--en:7bb3bd85-->
- **Längenunabhängige Regel:** Kein Parameter des Attention-Pooling-Operators $\sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i$ hängt von der Kontextlänge $m$ ab.

<!--en:eab48098-->
- **Linearer Speicherbedarf für Schlüssel und Werte:** Bei $m = 1{,}024$ speichert in GPT-2 small ($d = 768, H = 12, d_h = 64$) jeder Kopf $2 \times 1{,}024 \times 64 = 131{,}072$ Gleitkommazahlen ($1.57\text{M}$ bei $12$ Köpfen gegenüber $768$ in einem RNN).

<!--en:37b42a74-->
- **Direkter Zugriff in einem Schritt:** Token $m$ bewertet Token $1$ direkt über $\alpha(\mathbf{q}_m, \mathbf{k}_1)$, ohne rekurrente Verdichtung.

<!--en:c1dcc233-->
::: {.slide title="Klassischer Vorläufer: Nadaraya-Watson-Regression"}

<!--en:1e32c07a-->
[Nichtparametrische Statistik (1964)]{.kicker}

<!--en:11a552e8-->
Lange vor dem Deep Learning formulierten Nadaraya (1964) und Watson (1964) die nichtparametrische Regression als Attention-Pooling über einem Trainingsdatensatz $\{(x_i, y_i)\}_{i=1}^n$:

<!--en:9bcfebbb-->
- **Zuordnung zur Datenbank:** Die Trainingseingaben $x_i$ dienen als **Schlüssel**, die Trainingslabel $y_i$ als **Werte** und die Teststelle $q$ als **Abfrage**.

<!--en:6654a09a-->
- **Äquivalenz zur Softmax-Attention:** Für jeden strikt positiven Kernel $K(q, k) > 0$ entspricht die Normalisierung von $K(q, x_i)$ über alle Schlüssel der Softmax-Attention mit dem Score $a(q, k) = \log K(q, k) = -\|q - k\|^2 / (2\sigma^2)$ (letzte Gleichheit für den Gauß-Kernel).

<!--en:74878e44-->
- **Statistische Konsistenz:** Für $n \to \infty$ und eine mit geeigneter Rate schrumpfende Bandbreite $\sigma \to 0$ konvergiert $f(q)$ gegen den wahren bedingten Erwartungswert $\mathbb{E}[y \mid x = q]$.

<!--en:ea41f332-->
::: {.slide title="Synthetischer Testfall und vektorisierte Implementierung"}

<!--en:a4fbb6b9-->
[Implementierung im Code]{.kicker}

<!--en:4a322a49-->
$n = 40$ verrauschte Beobachtungen gemäß $y = 2\sin(x) + x + \epsilon$ mit $\epsilon \sim \mathcal{N}(0, 1)$ auf $[0, 5)$ ziehen. Alle Abstände zwischen Abfragen und Schlüsseln per Broadcasting berechnen:

<!--en:5bdf5176-->
- **Abstandsmatrix per Broadcasting:** `x_val[:, None] - x_train[None, :]` wendet Broadcasting auf die Formen `(50, 1)` und `(1, 40)` an. Das ergibt die Matrix der Form `(50, 40)` mit den paarweisen Differenzen $q - k_i$.

<!--en:5af0a447-->
- **Zeilenweise Normalisierung:** `k / k.sum(1, keepdims=True)` normalisiert die Zeile jeder Abfrage über alle $40$ Schlüssel. Mit `attention_w @ y_train` lassen sich alle $50$ Vorhersagen in einem Matrix-Vektor-Produkt berechnen.

<!--en:eff5268f-->
::: {.slide title="Wie die Bandbreite den Zielkonflikt zwischen Verzerrung und Varianz bestimmt"}

<!--en:76c7c8c5-->
[Der Schätzer im Diagramm]{.kicker}

<!--en:9a99010a-->
- **Kleine Bandbreite ($\sigma = 0.1$):** Geringe Verzerrung, hohe Varianz. Das Gewicht konzentriert sich auf den nächsten Trainingspunkt ($\to$ Interpolation mit dem nächsten Nachbarn für $\sigma \to 0$). Der Schätzer folgt einzelnen Rauschspitzen.

<!--en:b23941eb-->
- **Mittlere Bandbreite ($\sigma = 0.5$):** Im Inneren folgt der Schätzer der wahren Kurve $2\sin(x) + x$ genau. Nur an den Rändern bleibt er zurück, weil dort alle Nachbarn auf einer Seite liegen.

<!--en:95e1c9f2-->
- **Große Bandbreite ($\sigma = 2.0$):** Hohe Verzerrung, geringe Varianz. Die Gewichte nähern sich $1/n$. Der Schätzer glättet zu stark und nähert sich dem globalen Stichprobenmittelwert $\bar{y}$.

<!--en:be3c1440-->
::: {.slide title="Bandbreite und inverse Softmax-Temperatur"}

<!--en:7b16e109-->
[Steuerung der Schärfe]{.kicker}

<!--en:0de8192f-->
- **Inverse Varianz als inverse Temperatur ($\beta = 1/\sigma^2$):**
  $$\alpha_i(q) = \mathrm{softmax}_i\!\left(\frac{1}{\sigma^2} \cdot \left(-\frac{1}{2}\|q - k_i\|^2\right)\right)$$

<!--en:b5341294-->
- **Niedrige Temperatur ($\sigma \to 0$, großes $1/\sigma^2$):** Die paarweisen Score-Differenzen wachsen ($\Delta a / \sigma^2 \to \infty$). Die Softmax-Funktion nähert sich einer **One-Hot-Auswahl des Maximums**.

<!--en:7f9e7b88-->
- **Hohe Temperatur ($\sigma \to \infty$, kleines $1/\sigma^2$):** Die Score-Differenzen schrumpfen gegen null. Die Gewichte nähern sich denen einer Mittelung mit **einheitlichem Gewicht $1/n$** (der Übergang $\sigma: 2.0 \to 0.1$ vergrößert die Score-Differenzen um den Faktor $400\times$).

<!--en:69afbca0-->
::: {.slide title="Effektive Anzahl der Schlüssel und Varianz des Schätzers"}

<!--en:94526b3d-->
[Statistische Diagnostik]{.kicker}

<!--en:e66a440c-->
Wie viele Schlüssel tragen effektiv zur Vorhersage für eine Abfrage bei? Für $y_i = g(x_i) + \epsilon_i$ mit unabhängigem Beobachtungsrauschen der Varianz $\textrm{Var}(\epsilon_i) = s^2$ hat der Attention-Schätzer folgende bedingte Varianz:

<!--en:2bcca72a-->
- **Effektive Stichprobengröße $n_{\textrm{eff}}(q) \in [1, n]$:** Sie gibt an, über wie viele Trainingswerte die Abfrage $q$ effektiv mittelt.

<!--en:cb736376-->
- **One-Hot-Grenzfall ($\sigma \to 0$):** $\sum_i \alpha_i^2 = 1$ $\to$ $n_{\textrm{eff}}(q) = 1$. Die volle Rauschvarianz $s^2$ einer einzelnen Beobachtung bleibt erhalten.

<!--en:bc572e8a-->
- **Grenzfall gleicher Gewichte ($\sigma \to \infty$):** $\alpha_i = 1/n$ $\to$ $\sum_i \alpha_i^2 = 1/n$ $\to$ $n_{\textrm{eff}}(q) = n$. Die Rauschvarianz sinkt auf $s^2/n$.

<!--en:fec6abe7-->
::: {.slide title="Heatmaps der Attention-Gewichte"}

<!--en:9b77dc58-->
[Visuelle Diagnostik]{.kicker}

<!--en:7eb06795-->
- **Zeilenstochastische Matrix:** Jede Zeile von $\mathbf{A} \in \mathbb{R}^{n_q \times n_k}$ ist die Wahrscheinlichkeitsverteilung einer Abfrage über die Schlüssel.

<!--en:f2c95b23-->
- **Diagonales Band bei $\sigma = 0.1$:** Bei sortierten Abfragen und Schlüsseln entsteht ein schmaler diagonaler Streifen. Jede Abfrage zieht nur $2$–$3$ benachbarte Schlüssel heran.

<!--en:97515ffb-->
- **Breite Verteilung bei $\sigma = 2.0$:** Die Wahrscheinlichkeitsmasse verteilt sich auf fast alle $40$ Schlüssel. Die Gewichte betragen höchstens etwa das Doppelte des einheitlichen Gewichts $1/40 = 0.025$ (Maximum $\approx 0.05$ an den Rändern).

<!--en:7316f363-->
::: {.slide title="Fallstrick 1: Abfragen außerhalb des Trainingsbereichs und Unterlauf"}

<!--en:48ee3cb6-->
[Numerische und statistische Fallstricke]{.kicker}

<!--en:67310bb4-->
- **Begrenzung auf die konvexe Hülle:** Für $q \to \infty$ jenseits von $[0, 5)$ dominiert der nächstgelegene Randschlüssel $x_{\max}$ alle anderen. Damit gilt $f(q) \to y_{\max}$. Diese horizontale Asymptote kann keinem linearen Trend folgen.

<!--en:703af4ae-->
- **Float32-Unterlauf führt zu `NaN`:** Bei $q = 15, \sigma = 0.5$ ist der Exponent $-(q - x_i)^2/(2\sigma^2) < -200$. In `float32` kommt es zum Unterlauf ($\exp(-200) \to 0.0$), sodass `k / k.sum()` den Ausdruck `0.0 / 0.0 = NaN` auswertet.

<!--en:dc5c02ee-->
- **Abhilfe:** Die logarithmierten Scores $-(q - k_i)^2/(2\sigma^2)$ direkt an eine `softmax`-Funktion übergeben, die zuerst das Maximum subtrahiert.

<!--en:04547451-->
::: {.slide title="Fallstrick 2: Grenzen fester Kernel und möglicher Fehlschluss bei Heatmaps"}

<!--en:b088e25c-->
[Interpretierbarkeit und Grenzen der Modellierung]{.kicker}

<!--en:061bb831-->
Vor dem Aufbau tiefer Attention-Schichten sind zwei grundlegende Einschränkungen des Poolings mit festen Kerneln zu beachten:

<!--en:79b7426d-->
- **1. Attention-Gewichte zeigen, *wohin* eine Schicht blickt, aber nicht, *was* sie überträgt:** Im zweidimensionalen Vektorbeispiel erhielt der Schlüssel $\mathbf{k}_2 = (2, 1)$ einen Anteil von $24\%$ am Gesamtgewicht, obwohl sein Wert $\mathbf{v}_2 = (0, 0)$ nichts übertrug. In tiefen Netzen hat selbst ein stark gewichtetes Token keinen kausalen Effekt, wenn sein Wert im Nullraum nachgelagerter Projektionen liegt.

<!--en:55652c32-->
- **2. Feste Metriken im Eingaberaum erfassen weder syntaktische noch aufgabenspezifische Relevanz:** Der Nadaraya-Watson-Schätzer misst den quadrierten euklidischen Abstand $\|q - k\|^2$ in den ursprünglichen Eingabekoordinaten mit einer einzigen isotropen Bandbreite $\sigma$.

<!--en:5734ac5e-->
- **Warum tiefe Netze Projektionen lernen:** Bei der Sprach- oder Bildverarbeitung bestimmt die Aufgabe, was als ähnlich gilt. Das erfordert **erlernte Projektionen für Abfragen, Schlüssel und Werte** ($\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$), etwa wenn ein Verb sein Subjekt oder ein Pronomen seinen Bezugsausdruck finden soll.

<!--en:254c6e82-->
::: {.slide title="Verständnisfragen: weiches Nachschlagen, Bandbreite und Skalierung"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:793a3cfa-->
Die folgenden drei Fragen lassen sich mit den Gleichungen der Softmax-Attention beantworten:

<!--en:6c0b2326-->
1. **Halbierung der Bandbreite:** Bei Bandbreite $\sigma$ erhalten zwei Schlüssel Attention-Gewichte im Verhältnis $\alpha_1 / \alpha_2 = 2 : 1$. Die Bandbreite sinkt auf $\sigma / 2$. Wie lautet das neue Gewichtsverhältnis $\alpha_1' / \alpha_2'$?

<!--en:1047c934-->
2. **Mittelung bei orthonormalen Schlüsseln:** Die Schlüssel $\mathbf{k}_1, \ldots, \mathbf{k}_m$ seien orthonormal, der Skalarprodukt-Score sei $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$. Was geschieht mit der Attention-Ausgabe für die Abfrage $\mathbf{q} = c\,(\mathbf{k}_1 + \mathbf{k}_2)$, wenn $c \to \infty$?

<!--en:9657b2e1-->
3. **Verschwindender Gradient der Abfrage:** Für $c \to \infty$ konvergiere $\boldsymbol{\alpha}$ gegen einen One-Hot-Vektor mit einem einzigen siegreichen Schlüssel. Was geschieht mit dem Gradienten $\nabla_{\mathbf{q}} \textrm{Attention}(\mathbf{q}, \mathcal{D})$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:70cf0cc0-->
1. **$16 : 1$.** Eine Halbierung von $\sigma$ multipliziert $1/\sigma^2$ mit $4$. Die Score-Differenz $\Delta$ wächst auf $4\Delta$, und es gilt $e^{4\Delta} = (e^\Delta)^4 = 2^4 = 16$.

<!--en:81265ac4-->
2. **Die Ausgabe konvergiert gegen $\frac{1}{2}(\mathbf{v}_1 + \mathbf{v}_2)$.** Die Schlüssel $1$ und $2$ erzielen denselben Score $c$, alle übrigen den Score $0$. Damit gilt für jedes der beiden Gewichte $e^c / (2e^c + m - 2) \to 1/2$.

<!--en:6324fae3-->
3. **Er verschwindet ($\to \mathbf{0}$).** Sobald ein einzelner Schlüssel das Gewicht $1$ trägt, verändern kleine Störungen von $\mathbf{q}$ die Gewichte kaum noch. Die Softmax-Funktion ist gesättigt, ihre Jacobi-Matrix nähert sich null.

<!--en:7e5ca085-->
::: {.slide title="Zusammenfassung: differenzierbares Nachschlagen in Schlüssel-Wert-Paaren"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:33942d70-->
- **Weiches Pooling durch Konvexkombination:** $\textrm{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i)\,\mathbf{v}_i$ mit $\boldsymbol{\alpha} \in \Delta^{m-1}$.

<!--en:3ab4e260-->
- **Temperatur und effektive Anzahl der Schlüssel:** Die Bandbreite $\sigma$ steuert den Übergang vom nächsten Nachbarn ($\sigma \to 0, n_{\textrm{eff}} \to 1$) zum Mittelwert mit einheitlichen Gewichten ($\sigma \to \infty, n_{\textrm{eff}} \to m$).

<!--en:646fd4d6-->
- **Numerische Stabilität durch Subtraktion des Maximums:** Die Subtraktion von $m(\mathbf{q}) = \max_j a(\mathbf{q}, \mathbf{k}_j)$ verhindert einen Unterlauf mit dem Ergebnis `0/0 = NaN`.
