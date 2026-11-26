<!--en:dc36ac7b-->
# Arbeiten mit Sequenzen

<!--en:61354138-->
Liest ein E-Mail-Programm den Satzanfang „Vielen Dank für Ihre“, schlägt es naheliegende Fortsetzungen wie „Hilfe“ oder „Zeit“ vor. Ähnlich nutzt ein Wetterdienst, der stündlich die Temperatur misst, die letzten Messwerte für die Vorhersage des nächsten Werts. In beiden Fällen kommt es für eine gute Vorhersage sowohl auf das bisher Beobachtete als auch auf dessen Reihenfolge an. Stellen wir die Wörter zu „Ihre vielen für Dank“ um, ändert sich die Fortsetzung völlig. Ebenso legt eine fallende Temperaturkurve eine ganz andere Prognose nahe als eine steigende. Keines der Modelle aus den bisherigen Kapiteln kann diese Struktur direkt erfassen. Sie bilden eine einzelne Eingabe fester Form, einen Vektor $\mathbf{x} \in \mathbb{R}^d$, auf eine Ausgabe ab und behandeln jedes Beispiel als unabhängig von den übrigen. Sätze, Tonaufnahmen und Folgen von Sensormessungen verletzen beide Annahmen. Ihre Länge variiert von Beispiel zu Beispiel, und ihre Einträge hängen stark voneinander ab.

<!--en:8b55dc8b-->
In diesem Abschnitt führen wir die Sequenzmodellierung auf ein Problem zurück, das wir bereits lösen können, nämlich die Regression. Wir sagen jeden Eintrag aus seinen Vorgängern vorher. Dafür verdichten wir die wachsende Vorgeschichte zu einer Eingabe fester Größe. Entweder behalten wir ein Fenster mit den jüngsten Beobachtungen bei, oder wir aktualisieren laufend eine Zusammenfassung aller bisherigen Beobachtungen. Wir erproben diese Ideen an einem Beispiel, für das wir die Ground Truth genau kennen. Dazu passen wir ein lineares autoregressives Modell mit festem Fenster an eine verrauschte Sinuswelle an. Wie wir sehen, lässt sich der jeweils nächste Schritt leicht vorhersagen. Deutlich schwieriger ist die Vorhersage über viele Schritte, wenn das Modell seine eigenen Vorhersagen als Eingabe erhält. Im weiteren Verlauf des Kapitels übertragen wir dieselben Prinzipien auf Text.

<!--en:f65a78d3-->
## Sequenzielle Daten

<!--en:c68629c6-->
Eine *Sequenz* ist eine geordnete Liste von Merkmalsvektoren $\mathbf{x}_1, \ldots, \mathbf{x}_T$ oder, im skalaren Fall, von Werten $x_1, \ldots, x_T$. Der Index $t \in \mathbb{Z}^+$ bezeichnet den *Zeitschritt* (engl. time step). Die Länge $T$ kann von Sequenz zu Sequenz variieren. Beispiele sind uns aus dem Alltag vertraut. Text ist eine Sequenz von Zeichen, Subwörtern oder Wörtern. Sprache und Musik lassen sich als Sequenzen von Abtastwerten einer Schallwelle oder von Zeitfenstern eines Spektrogramms darstellen. Bei Zeitreihen wie Aktienkursen, Sensormesswerten und Vitalwerten von Patienten kommt jeweils ein Messwert nach dem anderen hinzu.

<!--en:a982c6e3-->
Wir nehmen weiterhin an, dass wir *ganze* Sequenzen unabhängig aus derselben zugrunde liegenden Verteilung über Sequenzen ziehen, etwa vollständige Dokumente oder ganze Krankenhausaufenthalte. Für die Einträge *innerhalb* einer Sequenz können wir Unabhängigkeit jedoch keineswegs voraussetzen. Welche Medikamente ein Patient am zehnten Tag eines Krankenhausaufenthalts erhält, hängt stark vom Verlauf der vorherigen neun Tage ab. Gerade diese Abhängigkeit macht die Vergangenheit aufschlussreich. Wären aufeinanderfolgende Einträge unabhängig, würden frühere Beobachtungen nichts über den nächsten Eintrag verraten. Auch müssen die Einträge an verschiedenen Positionen einer Sequenz nicht dieselbe Randverteilung haben. Der Schlussabsatz eines Dokuments unterscheidet sich meist deutlich vom ersten Satz. Der Zustand eines Patienten kann sich im Laufe der Behandlung verbessern oder verschlechtern.

<!--en:894ca335-->
Vorhersageaufgaben für Sequenzen haben unterschiedliche Formen. Wir können einer ganzen Sequenz ein einziges Label zuweisen, etwa die Stimmung einer Filmkritik klassifizieren. Wir können aus einer festen Eingabe eine Sequenz erzeugen, etwa eine Bildbeschreibung, oder eine Eingabesequenz auf eine Ausgabesequenz abbilden, etwa beim Übersetzen eines Satzes oder beim Transkribieren gesprochener Sprache. Bevor wir uns mit Zielgrößen für das überwachte Lernen befassen, lohnt es sich jedoch, die Sequenz selbst zu modellieren. Dazu schätzen wir ihre gemeinsame Wahrscheinlichkeit $P(x_1, \ldots, x_T)$. Für die Schätzung dieser Verteilung brauchen wir keine externen Label, denn jeder Eintrag dient als Ziel für die vorangehenden Einträge. Ein Modell, das Sequenzen Wahrscheinlichkeiten zuweist, kann außerdem neue Sequenzen erzeugen und konkurrierende Kandidaten nach ihrer Wahrscheinlichkeit ordnen. So kann es etwa von zwei Transkriptionen derselben Tonaufnahme die plausiblere auswählen.

<!--en:e481b0b6-->
## Autoregressive Modelle

<!--en:578125aa-->
Betrachten wir eine Zahlenreihe, etwa einen Aktienindex oder stündliche Temperaturmesswerte. Wer entscheiden möchte, wie auf den nächsten Messwert zu reagieren ist, interessiert sich vor allem für die bedingte Verteilung dieses Werts bei gegebenen bisherigen Beobachtungen:

<!--en:e8ba4994-->
Das Präfix $x_1, \ldots, x_{t-1}$ kommt in diesem Kapitel häufig vor. Wir kürzen es mit $x_{<t}$ ab. Selbst wenn sich die vollständige bedingte Verteilung nur schwer schätzen lässt, liefert ihr bedingter Erwartungswert $\mathbb{E}[x_t \mid x_{<t}]$ bereits eine nützliche Zusammenfassung. Wir könnten ihn etwa mit linearer Regression schätzen (:numref:`sec_linear_regression`). Ein Modell, das ein Signal durch Regression auf seine eigenen früheren Werte beschreibt, heißt *autoregressives Modell*. Sowohl die Merkmale als auch die Zielgröße stammen aus derselben Reihe.

<!--en:63696e68-->
Dabei stoßen wir sofort auf eine Schwierigkeit. Die Zahl der Eingaben in $x_{<t}$, auf die wir bedingen, wächst mit $t$. Damit liefert jeder Zeitschritt eine andere Anzahl von Merkmalen. Ein Modell mit fester Eingabegröße kann sie nicht direkt verarbeiten. Zwei allgemeine Strategien lösen dieses Problem, indem sie die Vorgeschichte zu einer Darstellung fester Größe verdichten (:numref:`fig_ar-vs-latent`). Beide bleiben autoregressiv. Sie unterscheiden sich nur darin, wie viel sie von der Vergangenheit bewahren.

<!--en:2b317836-->
### Feste Fenster

<!--en:fb800885-->
Bei der ersten Strategie bedingen wir nur auf die $m$ jüngsten Beobachtungen $x_{t-m}, \ldots, x_{t-1}$ und verwerfen alle früheren Beobachtungen. Für jeden Zeitpunkt $t > m$ hat das Trainingsbeispiel genau $m$ Merkmale. Damit können wir jede lineare oder tiefe Architektur mit fester Eingabegröße direkt einsetzen. Dies ist die klassische Form der Autoregression. Wir nennen $m$ die *Fensterlänge* (engl. window length). Feste Fenster lassen sich leicht implementieren und über Zeitschritte hinweg parallel verarbeiten, da jede Vorhersage nur von ihrem eigenen Ausschnitt der Reihe abhängt. Ihr Nachteil ist die starre zeitliche Grenze. Was mehr als $m$ Schritte zurückliegt, ist für das Modell vollständig unsichtbar.

<!--en:0be973ac-->
### Latente Zusammenfassungen

<!--en:632e477a-->
Die zweite Strategie führt eine Zusammenfassung $\mathbf{h}_t$ fester Größe über alle Beobachtungen $x_1, \ldots, x_t$ mit. Bei jedem neuen Wert aktualisieren wir sie rekursiv nach $\mathbf{h}_t = g(\mathbf{h}_{t-1}, x_t)$. Aus dieser Zusammenfassung sagen wir mit $\hat{x}_{t+1} = f(\mathbf{h}_t)$ den nächsten Eintrag vorher. Da wir $\mathbf{h}_t$ nie direkt beobachten, heißt ein solches Modell *latentes autoregressives Modell*. Der Zustandsvektor $\mathbf{h}_t$ benötigt unabhängig von der Länge der Sequenz immer gleich viel Speicher. So kann das Modell eine beliebig lange Vorgeschichte auf begrenztem Raum zusammenfassen. Rekurrente neuronale Netze (:numref:`sec_rnn`) sind latente autoregressive Modelle mit einer erlernten Übergangsfunktion $g$. Dasselbe gilt für die Zustandsraummodelle aus :numref:`chap_modern_rnn`. Was soll der versteckte Zustand speichern, und wie lässt er sich zuverlässig aktualisieren? Diese Fragen stehen beim Entwurf der Modelle in jenen Kapiteln im Mittelpunkt.

<!--en:3e444837-->
![Zwei Möglichkeiten, einem Sequenzmodell eine Eingabe fester Größe zu geben. (Links) Ein festes Fenster berücksichtigt die letzten $m$ Beobachtungen und verwirft den Rest. (Rechts) Eine latente Zusammenfassung führt einen Zustand $\mathbf{h}_t$ mit, der die gesamte Vergangenheit auf festem Raum zusammenfasst.](../img/mdl-rnn-ar-vs-latent.svg)

<!--en:56544633-->
### Von bedingten Verteilungen zu Sequenzen

<!--en:b2ebf13c-->
Unabhängig von der gewählten Darstellung legen die bedingten Verteilungen für jeweils einen Schritt das Wahrscheinlichkeitsmodell der gesamten Sequenz vollständig fest. Nach der *Kettenregel der Wahrscheinlichkeit* (engl. chain rule of probability) zerfällt jede gemeinsame Verteilung in ein Produkt bedingter Verteilungen. Zu jedem Zeitschritt gehört ein Faktor, der auf alle vorausgehenden Einträge bedingt ist:

<!--en:da6c1542-->
Für $t = 1$ ist das Präfix $x_{<1}$ leer, und der erste Faktor lautet $P(x_1)$. Diese Identität gilt exakt für jede gemeinsame Verteilung über Sequenzen. Daher ist es mathematisch gleichwertig, den nächsten Eintrag oder die gesamte Sequenz zu modellieren. Für eine Wortfolge aus drei Wörtern gilt zum Beispiel

<!--en:db4e2c4c-->
Weist ein Modell diesen drei Faktoren die Wahrscheinlichkeiten $0.05$, $0.01$ und $0.2$ zu, so erhält der Satz die gemeinsame Wahrscheinlichkeit $10^{-4}$. In der Praxis bewerten wir eine Sequenz, indem wir die Logarithmen der bedingten Wahrscheinlichkeiten addieren. So vermeiden wir den numerischen Unterlauf, der bei langen Sequenzen durch direkte Multiplikation entstehen kann. Um eine neue Sequenz zu erzeugen, kehren wir das Vorgehen um. Wir ziehen $x_1$, dann ziehen wir $x_2$ bedingt auf $x_1$. Jeden neu gezogenen Wert verwenden wir für die nächste bedingte Verteilung.

<!--en:592d8204-->
Mit :eqref:`eq_chain_rule` haben wir stillschweigend eine Reihenfolge gewählt. Wir haben die Sequenz von links nach rechts faktorisiert. Grundsätzlich liefert jede Permutation der Indizes eine mathematisch gültige Faktorisierung. Bei kausalen Zusammenhängen lässt sich die Vorwärtsrichtung jedoch meist viel leichter modellieren. Eine Wirkung aus ihren Ursachen vorherzusagen ist in der Regel einfacher, als den Prozess umzukehren :cite:`Peters.Janzing.Scholkopf.2017`. Da wir auch in der Praxis aus der Vergangenheit die Zukunft vorhersagen möchten, faktorisieren wir Sequenzen in diesem Kapitel durchgehend von links nach rechts.

<!--en:1a1472de-->
Sind die Einträge der Sequenz diskrete Token, so ist jede bedingte Verteilung ein Klassifikator über dem Vokabular. Die Schätzung von $P(x_1, \ldots, x_T)$ heißt dann *Sprachmodellierung* (engl. language modeling, siehe :numref:`sec_language-model`). Für den Rest dieses Abschnitts bleiben wir bei kontinuierlichen Werten. So können wir uns darauf konzentrieren, wie die autoregressive Vorhersage funktioniert.

<!--en:cd034ab6-->
## Markov-Modelle und Stationarität

<!--en:12aecfa3-->
Mit einem festen Fenster setzen wir stillschweigend voraus, dass wir ältere Beobachtungen vernachlässigen dürfen. Passen wir ein einziges Modell für alle Zeitschritte an, nehmen wir außerdem an, dass sich die zugrunde liegende Vorhersageregel im Laufe der Zeit nicht verschiebt. Beide Annahmen sind klassische Konzepte der Wahrscheinlichkeitstheorie und der Zeitreihenanalyse.

<!--en:435b801b-->
### Die Markov-Bedingung

<!--en:455884ce-->
Reichen die $m$ jüngsten Schritte für eine ebenso gute Vorhersage aus wie die gesamte Vorgeschichte, so erfüllt die Sequenz eine *Markov-Bedingung*. Bei gegebener jüngerer Vergangenheit ist die Zukunft bedingt unabhängig von der ferneren Vergangenheit. Für $m = 1$ ist der Prozess ein Markov-Modell *erster Ordnung*, allgemein ein Markov-Modell $m^{\textrm{th}}$ Ordnung. Unter der Bedingung erster Ordnung hängt der jeweilige Eintrag in jedem Faktor von :eqref:`eq_chain_rule` nur von seinem unmittelbaren Vorgänger ab. Die gemeinsame Wahrscheinlichkeit vereinfacht sich damit zu einem Produkt über benachbarte Paare:

<!--en:0faf688e-->
Reale Sequenzen erfüllen eine Markov-Bedingung nur selten exakt. Bei natürlichsprachlichen Texten liefert ein längerer Kontext immer wieder nützliche Informationen, auch wenn jedes weitere Token weniger dazu beiträgt. Dennoch ist eine Markov-Bedingung $m^{\textrm{th}}$ Ordnung in der Praxis oft nötig. Mit dieser Annahme bleibt der Rechenaufwand beherrschbar, und das Modell lässt sich aus endlich vielen Daten schätzen. Selbst moderne große Sprachmodelle arbeiten mit einem begrenzten Kontextfenster statt mit einer unendlichen Vorgeschichte. Für diskrete Sequenzen schätzt das einfachste Markov-Modell $P(x_t \mid x_{t-1})$ direkt, indem es zählt, wie oft jeder Nachfolger auf den jeweiligen Kontext folgt. Genau auf diesem Prinzip beruhen die $n$-Gramm-Sprachmodelle, die wir in :numref:`sec_language-model` entwickeln.

<!--en:93449f23-->
### Stationarität

<!--en:26549769-->
Wenn wir ein einziges Vorhersagemodell für alle Zeitschritte anpassen, treffen wir eine zweite Annahme. Die *Dynamik*, nach der jeder Eintrag aus seinen Vorgängern entsteht, bleibt über die Zeit gleich. Prozesse mit zeitinvarianter Dynamik heißen *zeithomogen* (engl. time-homogeneous). Im maschinellen Lernen ist die weniger präzise Bezeichnung *stationär* weitaus üblicher. Die beobachteten Werte selbst können durchaus stark schwanken. Konstant bleibt die bedingte Verteilung $P(x_t \mid x_{t-m}, \ldots, x_{t-1})$. Eine feste Übergangsregel bedeutet insbesondere nicht, dass die Einträge identisch verteilt sind. Das Ende eines Dokuments unterscheidet sich weiterhin von seinem Anfang, weil dieselbe Regel auf einen anderen Kontext wirkt. Die Stationarität rechtfertigt es, Beispiele aus verschiedenen Teilen einer Sequenz oder aus mehreren Sequenzen zusammenzuführen. Dazu verschieben wir ein Fenster entlang der jeweiligen Reihe und bilden so einen gemeinsamen Trainingsdatensatz.

<!--en:53e956e5-->
## Vorhersage einer verrauschten Sinuswelle

<!--en:b18f8a64-->
Erproben wir diese Konzepte an einem Beispiel, für das wir die Ground Truth genau kennen. Dazu passen wir ein einfaches autoregressives Modell an synthetische Daten an und vergleichen Vorhersagen für den nächsten Schritt mit rekursiven Vorhersagen über mehrere Schritte.

<!--en:b283db49-->
### Daten und Modell

<!--en:0bd5863f-->
Wir erzeugen eine Zeitreihe mit $T = 1000$ Zeitschritten aus einer verrauschten Sinuswelle $x_t = \sin(0.01\, t) + \epsilon_t$. Das additive Rauschen $\epsilon_t \sim \mathcal{N}(0, 0.2^2)$ ziehen wir bei jedem Schritt unabhängig. Bei dieser Frequenz erstreckt sich eine volle Periode über $2\pi / 0.01 \approx 628$ Schritte.

<!--en:78791433-->
Erzeugen wir eine Instanz von `Data` und tragen `data.x` gegen `data.time` auf, sehen wir etwa anderthalb Perioden der verrauschten Schwingung.

<!--en:48f794ca-->
Um aus dieser Reihe Beispiele für das überwachte Training zu gewinnen, verwenden wir ein festes Fenster der Länge $m$. Jedes Beispiel besteht aus dem Merkmalsvektor $(x_{t-m}, \ldots, x_{t-1})$ und dem zugehörigen Label $y = x_t$. Insgesamt erhalten wir $T - m$ Beispiele. Die ersten $m$ Einträge lassen wir weg, da ihre Vorgeschichte zu kurz ist. Eine gängige Alternative ist Zero-Padding. Die Methode `lag_features` erstellt die gesamte Merkmalsmatrix auf einmal, mit einer Spalte pro Zeitversatz. Der Datenlader verwendet die ersten 600 Beispiele für das Training. Sie umfassen knapp eine volle Periode der Sinuswelle. :numref:`fig_lag-matrix` zeigt die resultierende Tabelle für $m = 4$ und $T = 1000$.

<!--en:536cd0d5-->
![Zeitversetzte Merkmale und Label für $m = 4$ und $T = 1000$. Zeile $i$ enthält die vier Werte $(x_i, \ldots, x_{i+3})$ und das zugehörige Label $x_{i+4}$. Die 996 Zeilen enden daher bei $x_{1000}$. Die ersten 600 Zeilen dienen dem Training, die übrigen 396 der Validierung.](../img/mdl-rnn-lag-matrix.svg)

<!--en:9540c9ec-->
Für die Vorhersage verwenden wir eine gewöhnliche lineare Regression mit den $m = 4$ zeitversetzten Merkmalen. Ein einfacheres autoregressives Modell ist kaum denkbar.

<!--en:34723083-->
### Einschrittvorhersage

<!--en:6ce4ef32-->
Wir beginnen mit dem einfachen Fall. Wir berechnen die Vorhersage $\hat{x}_t$ aus den *beobachteten* $m$ vorangehenden Werten, und zwar für jeden Zeitschritt $t > m$.

<!--en:2049d6a6-->
Wir zeichnen die Einschrittvorhersagen `onestep_preds` zusammen mit den tatsächlichen Zielwerten `labels` für alle $T - m = 996$ Zeitschritte.

<!--en:69295ffa-->
Die Einschrittvorhersagen folgen der zugrunde liegenden Welle sehr genau, auch über die ersten 600 Trainingsfenster hinaus, wenn wir das Modell an neuen Daten auswerten.

<!--en:679aecbc-->
### Rekursive Mehrschrittvorhersage

<!--en:4e234952-->
Mehrere Schritte in die Zukunft vorherzusagen ist erheblich schwieriger. Nehmen wir an, die Beobachtungen enden bei Zeitschritt $t$ und wir möchten $\hat{x}_{t+5}$ vorhersagen. Das Modell für die Einschrittvorhersage können wir nicht direkt anwenden, denn $x_{t+1}, \ldots, x_{t+4}$ haben wir nicht beobachtet. Stattdessen geben wir dem Modell seine eigenen Vorhersagen als Eingabe zurück, als wären sie tatsächliche Beobachtungen. So gehen wir jeweils einen Zeitschritt weiter:

<!--en:f16e4a8a-->
Vom fünften Schritt an stützt sich das Modell ausschließlich auf seine eigenen früheren Vorhersagen. Wir nennen $\hat{x}_{t+k}$ eine *$k$-Schritt-Vorhersage*. Das Verfahren, Einschrittvorhersagen auf diese Weise zu verketten, heißt *rekursive Vorhersage* (engl. rollout). Der folgende Code berechnet die rekursive Vorhersage des linearen Modells von $t = 604$, dem letzten für das Training verwendeten Zeitpunkt, bis zum Ende der Reihe.

<!--en:4b2d52cc-->
In JAX übernimmt `jax.lax.scan` die Schleife über die Zeitschritte. Die Operation reicht das Fenster von einem Schritt zum nächsten weiter und fasst die Vorhersagen entlang einer neuen Achse zusammen.

<!--en:bcdfab8c-->
Die rekursive Vorhersage kann der Schwingung überhaupt nicht folgen. Sie beginnt etwa auf dem Niveau des letzten beobachteten Fensters und nähert sich allmählich einem konstanten Wert, während die tatsächliche Welle weiter steigt und fällt. Wie kann ein Modell mit so kleinem Einschrittfehler derart versagen? Ein Blick auf die erlernten Gewichte erklärt es unmittelbar.

<!--en:2f3d04a5-->
Alle vier Gewichte liegen nahe bei $1/4$. Das Modell hat also gelernt, den Mittelwert der vier letzten Beobachtungen als nächste Beobachtung vorherzusagen. Für die Einschrittvorhersage ist das sinnvoll. Die Sinuswelle ändert sich pro Schritt um höchstens $0.01$, während das additive Rauschen eine Standardabweichung von $0.2$ hat. Mitteln wir vier verrauschte Messwerte, sinkt die Rauschvarianz um den Faktor vier (von $0.04$ auf $0.01$). Dass der Mittelwert der langsam veränderlichen Kurve um ein oder zwei Schritte hinterherläuft, führt nur zu einer vernachlässigbaren Verzerrung. Erhält das Modell jedoch seine eigenen Ausgaben als Eingaben, kann ein gleitender Mittelwert nur das aktuelle Niveau fortschreiben. Da die Summe der Gewichte knapp unter eins liegt, nähert sich dieses Niveau langsam dem Fixpunkt $b / (1 - \sum_i w_i)$ der rekursiven Vorhersage. Dabei ist $b$ der Bias. Der Fixpunkt ist ein Quotient aus zwei kleinen Zahlen, weshalb die genaue Asymptote von Lauf zu Lauf variiert. Das Modell hat die Krümmung der Sinuswelle nicht erlernt, weil das Trainingsziel für einen Schritt davon kaum profitiert. Bei langen Vorhersagehorizonten macht das den größten Teil des Fehlers aus. Die Ursache liegt allerdings nicht in der Linearität. Ohne Rauschen erfüllt eine Sinuswelle die exakte Rekursion zweiter Ordnung $x_t = 2\cos(0.01)\, x_{t-1} - x_{t-2}$. Zwei zeitversetzte Werte mit Gewichten nahe $2$ und $-1$ würden die Schwingung also unbegrenzt fortsetzen. Bei verrauschten Daten würden diese Gewichte die Rauschvarianz jedoch verfünffachen. Minimieren wir den quadratischen Einschrittfehler, begünstigt das daher die Mittelung (siehe :exref:`ex_sequence-window-length`).

<!--en:085ddbf8-->
### Fehlerwachstum mit dem Vorhersagehorizont

<!--en:405b0864-->
Wie schnell verschlechtert sich die Vorhersage mit wachsendem Horizont $k$? Dazu berechnen wir $k$-Schritt-Vorhersagen gleichzeitig für jede gültige Startposition der Reihe. Die Hilfsfunktion `k_step_pred` verwaltet eine Liste von Spalten. Jede Spalte enthält einen Eintrag pro Startposition. Die Liste beginnt mit den $m$ Spalten für die beobachteten zeitversetzten Werte. In jeder Runde berechnet die Funktion aus den letzten $m$ Spalten die nächste Vorhersagespalte und hängt sie an. Sie gibt die Liste der Vorhersagespalten zurück. Das Element mit Index $(j-1)^{\textrm{th}}$ enthält die $j$-Schritt-Vorhersagen.

<!--en:abe2f551-->
Mit `k_step_pred` berechnen wir die rekursiven Vorhersagen des trainierten Modells für bis zu $k = 64$ Schritte. Dabei beginnen wir bei jedem gültigen Startfenster in `data.x`.

<!--en:f731a3f8-->
Wir zeichnen nun für die Horizonte $k \in \{1, 4, 16, 64\}$ die Vorhersagekurven aus `preds`. So sehen wir, wie sich die Fehler mit wachsendem Horizont summieren.

<!--en:a76ee624-->
Die Vorhersagen für 1 und 4 Schritte folgen der Reihe eng, während die Vorhersage für 16 Schritte merklich hinterherläuft. Die Vorhersage für 64 Schritte sieht aus wie eine um etwa 64 Schritte verschobene Kopie der Welle mit sichtbar gedämpfter Amplitude. Wie stark die Amplitude abnimmt, hängt davon ab, wie weit die Summe der Gewichte unter eins liegt. Wir können die Verschlechterung anhand des mittleren quadratischen Fehlers gegenüber den beobachteten Werten messen. Als Baseline verwenden wir die Varianz der Reihe. Sie entspricht dem Fehler einer konstanten Vorhersage, die stets den Mittelwert liefert.

<!--en:53c3df48-->
Für einen Schritt liegt der mittlere quadratische Fehler nahe bei $0.05$. Er setzt sich aus der irreduziblen Rauschvarianz $0.2^2 = 0.04$ und der Rauschvarianz $0.01$ zusammen, die im Mittelwert aus vier Beobachtungen verbleibt. Mit wachsendem $k$ steigt der Fehler fast ausschließlich durch die Phasenverzögerung. Eine $k$-Schritt-Vorhersage gibt ungefähr das Niveau der Reihe $k$ Schritte zuvor wieder. Der zusätzliche Fehler entspricht daher der Änderung der Sinuswelle über $k$ Schritte. Mitteln wir das Quadrat dieser Änderung über die Phase $a$ der Welle, erhalten wir

<!--en:7fbc9091-->
Dies ergibt $0.001$ für $k = 4$, $0.013$ für $k = 16$ und $0.20$ für $k = 64$. Addieren wir den Grundfehler von $0.05$, erwarten wir Fehler von etwa $0.05$, $0.05$, $0.065$ und $0.25$. Das stimmt gut mit unseren Messungen überein. Bei 64 Schritten beträgt der Fehler das Vier- bis Sechsfache des Einschrittfehlers und bereits rund die Hälfte der Varianz der gesamten Reihe.

<!--en:77e3794b-->
Ein wichtiges Detail ist, dass dieser Fehler an keiner Stelle durch eine unkontrollierte Anhäufung von Fehlern entsteht. Das mittelnde Modell dämpft alles, was es als Eingabe zurückerhält. Seine rekursive Vorhersage nähert sich daher stetig einer Konstanten, statt zu divergieren. Die $k$-Schritt-Vorhersage verfehlt ihr Ziel, weil die tatsächliche Welle inzwischen weitergelaufen ist. Kleine Fehler haben sich hier nicht exponentiell verstärkt. Allgemein lernt ein Modell mit einem reinen Einschrittziel nur das, was für den nächsten Schritt nützlich ist. Wie gut seine rekursive Vorhersage über viele Schritte ausfällt, hängt davon ab, ob es die zugrunde liegende Dynamik erfasst. Ein geringer Einschrittverlust garantiert das nicht.

<!--en:519dd121-->
Bei rekursiven Mehrschrittvorhersagen gibt es noch eine zweite, bekanntere Fehlerquelle, die bei diesem dämpfenden linearen Modell nicht auftritt. Jeder Vorhersagefehler wird Teil der Eingabe für alle folgenden Schritte. Verstärkt ein Modell Störungen, statt sie zu dämpfen, kann es sich immer weiter von der Datenverteilung entfernen. Bei der Sprachmodellierung genügen wenige ungünstig gezogene Token, damit ein Generator zu unnatürlichen Formulierungen übergeht, wie sie im Training nie vorkamen. Er kann dadurch vom Thema abkommen oder in Wiederholungsschleifen geraten. Aus ähnlichem Grund verlieren erlernte physikalische Simulatoren auf langen Trajektorien ihre Stabilität. Zwei allgemeine Strategien helfen, dieser Abweichung entgegenzuwirken. Beim Training können wir dem Modell neben tatsächlich beobachteten Verläufen auch eigene Vorhersagen vorlegen. Damit liegen fehlerhafte Präfixe nicht mehr außerhalb der Trainingsverteilung. Bei der Inferenz wandelt eine Dekodierregel jede vorhergesagte Verteilung in eine konkrete Ausgabe um. Die Regel lässt sich so gestalten, dass sie unwahrscheinliche Ausgaben unterdrückt, die eine rekursive Vorhersage aus dem Tritt bringen. Für Text kommen wir in :numref:`sec_decoding` darauf zurück. Zunächst müssen wir jedoch Rohtext in eine Darstellung umwandeln, die Sequenzmodelle verarbeiten können.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:042a4f01-->
Autoregressive Modelle führen die Sequenzmodellierung auf eine überwachte Vorhersageaufgabe zurück. Sie schätzen jeden Eintrag aus seinen Vorgängern. Dazu verwenden sie entweder ein festes Fenster mit den $m$ jüngsten Beobachtungen (das Prinzip der $n$-Gramme) oder einen latenten Zustand begrenzter Größe, der die gesamte Vergangenheit zusammenfasst (das Prinzip rekurrenter Netze). Multiplizieren wir nach der Kettenregel der Wahrscheinlichkeit die bedingten Verteilungen für jeweils einen Schritt, erhalten wir ein vollständiges probabilistisches Modell der Sequenz. Damit können wir Sequenzen sowohl bewerten als auch erzeugen. Ein einziges Modell für alle Zeitschritte anzupassen setzt Stationarität voraus. Die bedingte Regel, nach der jeder Eintrag aus seiner Vorgeschichte entsteht, muss also über die Zeit gleich bleiben. Schließlich garantiert eine hohe Genauigkeit für einen Schritt noch keine guten Mehrschrittvorhersagen, da das Einschritttraining nur kurzfristig nützliche Merkmale begünstigt. Bei unserer verrauschten Sinuswelle nähert sich das mittelnde lineare Modell für einen Schritt der rauschbedingten Fehleruntergrenze. Für 64 Schritte ist sein Fehler jedoch vier- bis sechsmal so groß. Dabei ist die zunehmende Drift noch gar nicht berücksichtigt, die bei anderen Modellen entsteht, wenn sich ihre Fehler bei der rekursiven Vorhersage verstärken.

<!--en:2294243a-->
## Übungen

<!--en:e17a120d-->
1. [code] **Fensterlänge.** `Data` liefert `m=4` zeitversetzte Merkmale, an die `d2l.LinearRegression` ein Modell anpasst.
    1. Erhöhen Sie `m` und trainieren Sie das Modell erneut. Ab wie vielen vergangenen Beobachtungen sinkt der Validierungsfehler nicht mehr?
    1. Entfernen Sie den Rauschterm in `Data` und passen Sie das Modell mit `m=2` an. Vergleichen Sie die beiden Gewichte mit $2\cos(0.01)$ und $-1$ und berechnen Sie die rekursive Vorhersage. Fügen Sie das Rauschen anschließend mit Standardabweichungen zwischen $0.001$ und $0.2$ wieder hinzu. Ab welchem Rauschpegel gehen die Gewichte von der Nachbildung der Krümmung zur Mittelung über?
   :exlabel:`ex_sequence-window-length`

<!--en:c233aa95-->
1. [code] **Nichtlineares autoregressives Modell.** Ersetzen Sie `d2l.LinearRegression` durch ein kleines MLP mit denselben vier zeitversetzten Merkmalen. Trainieren Sie es erneut, gegebenenfalls über mehr Epochen. Wie verändern sich die Einschritt- und die Mehrschrittvorhersagen?

<!--en:738ab061-->
1. [code] **Fehlerwachstum mit dem Vorhersagehorizont.** Für jedes $k$ enthält der letzte Eintrag der von `k_step_pred(model, data.x, data.m, k)` zurückgegebenen Liste die $k$-Schritt-Vorhersagen für `data.x[data.m+k-1:]`.
    1. Tragen Sie für $k \in \{1, 2, 4, \ldots, 256\}$ den mittleren quadratischen $k$-Schritt-Fehler gegen $k$ auf. Skalieren Sie die horizontale Achse logarithmisch. Wo biegt die Kurve nach oben ab, und wie hängt das mit der Dämpfung der Vorhersagen für 16 und 64 Schritte zusammen?
    1. Finden Sie das kleinste $k$, bei dem der mittlere quadratische $k$-Schritt-Fehler die Varianz der Reihe übersteigt. Welchen Vorteil bietet eine Vorhersage jenseits dieses Horizonts gegenüber der konstanten Vorhersage?

<!--en:67a27f26-->
1. [code] **Zeitversetzte Werte mit größeren Abständen und Blockmittelwerte.** ● Behalten Sie für `d2l.LinearRegression` vier Merkmale bei, lassen Sie diese jedoch weiter zurückreichen. Verwenden Sie einmal die zeitversetzten Werte $x_{t-1}, x_{t-2}, x_{t-4}, x_{t-8}$ und einmal die vier Blockmittelwerte $\frac{1}{4}\sum_{i=0}^{3} x_{t-4j+i}$ für $j = 1, \ldots, 4$. Vergleichen Sie die Validierungsfehler mit denen für $m = 4$ und $m = 16$. Welche Konstruktion hilft und warum?

<!--en:49cb9c6e-->
1. [code] **Regimewechsel.** ● Ersetzen Sie die einzelne Sinuswelle in `Data` durch eine Reihe, die alle 200 Zeitschritte zwischen $\sin(0.01\,t)$ und $\sin(0.05\,t)$ wechselt. Behalten Sie das additive Rauschen bei. Beide Regime kommen dann sowohl in den Trainings- als auch in den Validierungsbeispielen vor. Die Reihe mit Regimewechsel verletzt die Stationaritätsannahme aus :numref:`subsec_stationarity`. Kann ein einziges lineares Modell mit $m = 4$ dennoch beide Regime gut beschreiben? Warum? Geben Sie den Trainings- und den Validierungsfehler für jedes Regime getrennt an. Bestimmen Sie die Stellen der Reihe, an denen die Anpassung am schlechtesten ist.

<!--en:c2442f26-->
[Diskussion](https://d2l.discourse.group/t/113)

<!--en:5e820f28-->
[Diskussion](https://d2l.discourse.group/t/114)

<!--en:215d9bd1-->
[Diskussion](https://d2l.discourse.group/t/1048)

<!--en:5fe44a9a-->
[Diskussion](https://d2l.discourse.group/t/18010)

<!--en:3b45919c-->
Arbeiten mit Sequenzen<br>**Autoregression · feste Fenster und latenter Zustand · Kettenregel · Markov-Modelle · Stationarität · Fehler bei rekursiven Mehrschrittvorhersagen**

<!--en:e638d5a8-->
::: {.slide title="Warum Sequenzdaten die i.i.d.-Annahme verletzen"}

<!--en:687ed302-->
[Motivation und Grundidee]{.kicker}

<!--en:98feba81-->
- **Variable Länge und feste zeitliche Reihenfolge:** Bei Dokumenten, Sprachsignalen oder Vitalwerten variiert die Länge $T$ von Beispiel zu Beispiel. Eine Umstellung (*„Vielen Dank für Ihre“* $\to$ *„Ihre vielen für Dank“*) zerstört den Sinn.

<!--en:bdd78ab7-->
- **Statistische Abhängigkeit innerhalb der Sequenz:** Verschiedene Trajektorien können unabhängig gezogen sein. Aufeinanderfolgende Einträge $x_t$ *innerhalb* einer Sequenz hängen jedoch stark von ihrem Präfix $x_{<t} = (x_1, \ldots, x_{t-1})$ ab.

<!--en:d1c851ad-->
- **Fehler bei längerfristigen Vorhersagen:** Für $x_t = \sin(0.01t) + \epsilon_t$ erzielt ein lineares Modell mit 4 vergangenen Werten einen nahezu optimalen mittleren quadratischen Fehler (MSE) für 1 Schritt ($\approx 0.05$). Bei der rekursiven Vorhersage über 64 Schritte versagt es jedoch (MSE $\approx 0.25$).

<!--en:c7598575-->
::: {.slide title="Aufgaben beim Lernen mit Sequenzen"}

<!--en:b245e065-->
[Problemstellung]{.kicker}

<!--en:830718db-->
- **Überwachte Abbildungen von Sequenzen:** Eine Sequenz variabler Länge lässt sich einer Kategorie zuordnen (**Sequenz-zu-Label**), eine feste Eingabe zu einer Sequenz erweitern (**feste Eingabe zu Sequenz**) oder eine Eingabesequenz $\mathbf{x}_{1:T}$ in eine Zielsequenz $\mathbf{y}_{1:T'}$ ohne positionsweise Zuordnung überführen (**Sequenz-zu-Sequenz**).

<!--en:332639bb-->
- **Unüberwachte Dichteschätzung:** Die Modellierung der gemeinsamen Wahrscheinlichkeit $P(x_1, \ldots, x_T)$ erfordert **keinerlei externe Label**, da jeder Eintrag $x_t$ als Ziel beim überwachten Lernen mit seinem eigenen Präfix $x_{<t}$ dient.

<!--en:788e499d-->
- **Zwei Anwendungen:** Mit einer gemeinsamen Verteilung über Sequenzen lassen sich Kandidaten nach ihrer Likelihood ordnen (z. B. Transkriptionen bei der automatischen Spracherkennung neu bewerten) und Sequenzen durch Ancestral Sampling erzeugen.

<!--en:710ee506-->
::: {.slide title="Autoregressive Vorhersage bei wachsenden Präfixen"}

<!--en:2a007293-->
[Autoregressive Modelle]{.kicker}

<!--en:950f935f-->
- **Warum „autoregressiv“:** Die bedingenden Merkmale $x_{<t}$ und das Regressionsziel $x_t$ stammen aus *ein und derselben* Reihe.

<!--en:6de9991f-->
- **Wechselnde Dimension:** Bei $t = 2$ enthält $x_{<2}$ genau 1 Merkmal, bei $t = 500$ enthält $x_{<500}$ schon 499 Merkmale. Modelle mit fester Eingabegröße erfordern eine konstante Merkmalsdimension.

<!--en:aabfae73-->
- **Zwei Strategien zur Verdichtung:** Das Präfix $x_{<t}$ auf ein festes Fenster der $m$ jüngsten Schritte **kürzen** oder einen **rekursiven latenten Zustand** $\mathbf{h}_{t-1}$ aktualisieren.

<!--en:db51337e-->
::: {.slide title="Feste Fenster und latenter autoregressiver Zustand"}

<!--en:8b8c5735-->
[Zwei Strategien zur Verdichtung]{.kicker}

<!--en:4857d85a-->
Beide Strategien führen die Sequenzvorhersage auf eine Vorhersage mit fester Eingabegröße zurück. Sie begrenzen den Speicherbedarf auf unterschiedliche Weise:

<!--en:3da4c77d-->
- **Strategie 1: festes Fenster über $m$ Schritte:** Nur auf die $m$ jüngsten Beobachtungen $(x_{t-m}, \ldots, x_{t-1})$ bedingen und $x_{\le t-m-1}$ verwerfen:
  $$\hat{x}_t = f(x_{t-m}, \ldots, x_{t-1})$$
  Das Verfahren ist einfach und über alle $t$ parallelisierbar. Sein **Gedächtnis reicht genau** $m$ Schritte zurück. Darauf beruhen $n$-Gramm-Modelle.

<!--en:d0dbbf59-->
- **Strategie 2: latente autoregressive Zusammenfassung:** Einen Vektor $\mathbf{h}_t \in \mathbb{R}^h$ fester Dimension bei jedem Schritt rekursiv aktualisieren:
  $$\mathbf{h}_t = g(\mathbf{h}_{t-1}, x_t)$$
  Aus dem Zustand lässt sich die Vorhersage $\hat{x}_{t+1} = f(\mathbf{h}_t)$ ablesen. So lässt sich eine beliebig lange Vorgeschichte mit Speicherbedarf $O(1)$ zusammenfassen. Darauf beruhen RNNs und Zustandsraummodelle.

<!--en:bf490b9c-->
::: {.slide title="Faktorisierung gemeinsamer Sequenzwahrscheinlichkeiten mit der Kettenregel"}

<!--en:bbdb2912-->
[Probabilistische Grundlage]{.kicker}

<!--en:fe68b496-->
- **Exakte Äquivalenz:** Die bedingten Verteilungen $P(x_t \mid x_{<t})$ für jeweils einen Schritt legen die gemeinsame Wahrscheinlichkeit der gesamten Sequenz vollständig fest.

<!--en:3e497fd3-->
- **Bewertung mit Logarithmen:** Das Produkt hunderter Wahrscheinlichkeiten aus $(0, 1)$ kann in `float32` durch Unterlauf zu $0.0$ werden. Abhilfe schafft die Summe der logarithmierten Wahrscheinlichkeiten $\sum_{t=1}^T \log P(x_t \mid x_{<t})$.

<!--en:b64a6cf9-->
- **Kausale Reihenfolge von links nach rechts:** Die Faktorisierung in zeitlicher Reihenfolge entspricht der physikalischen Kausalität und ermöglicht schrittweises Ancestral Sampling.

<!--en:a3ae875f-->
::: {.slide title="Die Markov-Bedingung und das begrenzte Gedächtnis"}

<!--en:1964c03e-->
[Statistische Annahmen]{.kicker}

<!--en:e4ff6f51-->
- **Markov-Bedingung $m^\textrm{th}$ Ordnung:** Der künftige Wert $x_t$ ist bedingt unabhängig von der fernen Vergangenheit $x_{\le t-m-1}$, wenn die $m$ jüngsten Beobachtungen gegeben sind.

<!--en:9d2ab183-->
- **Modell erster Ordnung ($m = 1$):** Jeder Eintrag hängt ausschließlich von seinem unmittelbaren Vorgänger ab. Es gilt $P(x_{1:T}) = P(x_1)\prod_{t=2}^T P(x_t \mid x_{t-1})$.

<!--en:feea4588-->
- **Näherung und Berechenbarkeit:** Natürliche Sprache erfüllt die Markov-Bedingung nie exakt. Begrenzte Fenster machen Häufigkeitstabellen und Modelle mit endlichem Kontext jedoch praktisch handhabbar.

<!--en:ab249fe0-->
::: {.slide title="Stationarität und zeithomogene Übergangsdynamik"}

<!--en:1964c03e-2-->
[Statistische Annahmen]{.kicker}

<!--en:051b6129-->
- **Zeitinvariante bedingte Verteilung:** *Ein einziges* Vorhersagemodell $f$ für alle Schritte $t \in \{m+1, \ldots, T\}$ zu verwenden, setzt voraus, dass sich die Übergangsverteilung $P(x_t \mid x_{t-m:t-1})$ mit $t$ nicht verändert.

<!--en:66ff44fe-->
- **Stationäre Dynamik bei veränderlichen Werten:** Eine feste Übergangsregel erzwingt *keine* konstante Randverteilung $P(x_t)$. Der Schlussabsatz eines Romans unterscheidet sich vom ersten Satz, weil dieselbe Regel auf einen anderen Kontext wirkt.

<!--en:9b988ba7-->
- **Gleitende Fenster als gemeinsame Datengrundlage:** Stationarität rechtfertigt es, $T - m$ Trainingspaare aus einer einzigen Trajektorie der Länge $T$ zu gewinnen.

<!--en:f6b56d59-->
::: {.slide title="Synthetischer Benchmark mit einer verrauschten Sinuswelle"}

<!--en:49abe507-->
[Kontrolliertes Experiment mit bekannter Ground Truth]{.kicker}

<!--en:fea9fcac-->
Für einen Test der Autoregression mit festem Fenster und analytisch bekannter Dynamik dienen $T = 1000$ Schritte einer verrauschten Sinuswelle mit einer Periode von $2\pi / 0.01 \approx 628$ Schritten:

<!--en:4a2054ca-->
- **Datenerzeugender Prozess:**
  $$x_t = \sin(0.01\,t) + \epsilon_t, \qquad \epsilon_t \stackrel{\textrm{i.i.d.}}{\sim} \mathcal{N}(0, 0.2^2)$$

<!--en:a840cb28-->
- **Größenordnung von Signal und Rauschen:** Die rauschfreie Welle ändert sich pro Schritt um höchstens $0.01$ ($|\cos(0.01t)| \cdot 0.01 \le 0.01$). Das additive weiße Rauschen hat die Standardabweichung $\sigma = 0.2$ ($20\times$ so groß wie die maximale Signaländerung pro Schritt) und die irreduzible Varianz $\sigma^2 = 0.04$.

<!--en:7594587e-->
- **Aufteilung für Training und Validierung:** Die ersten `num_train = 600` Fenster (etwa 1 volle Periode) bilden den Trainingsdatensatz. Die übrigen Fenster dienen der Prüfung an neuen Daten.

<!--en:052fda9f-->
::: {.slide title="Von der Zeitreihe zu gleitenden Fenstern"}

<!--en:f04c6d25-->
[Merkmale für das überwachte Lernen]{.kicker}

<!--en:da2c398e-->
- **Vektorisierte Konstruktion zeitversetzter Merkmale:** Vier gegeneinander verschobene Ausschnitte aus `self.x` ergeben die Merkmalsmatrix der Größe $(T - m) \times m = 996 \times 4$. Dazu gehört die Zielspalte $\mathbf{y} \in \mathbb{R}^{996 \times 1}$.

<!--en:3aad966d-->
- **Chronologische Aufteilung:** Die Zeilen $1\text{--}600$ ($t = 5\text{--}604$) dienen dem Training, die Zeilen $601\text{--}996$ ($t = 605\text{--}1000$) der Auswertung auf neuen Daten.

<!--en:b9f0c7f3-->
::: {.slide title="Training eines linearen autoregressiven Modells mit vier zeitversetzten Werten"}

<!--en:2a3892d1-->
[Modell und Auswertung für einen Schritt]{.kicker}

<!--en:523c29c2-->
Eine gewöhnliche lineare Regression $\hat{x}_t = w_1 x_{t-4} + w_2 x_{t-3} + w_3 x_{t-2} + w_4 x_{t-1} + b$ minimiert den mittleren quadratischen Einschrittfehler über 5 Epochen mit Minibatch-SGD ($\eta = 0.01$, Batchgröße $16$):

<!--en:3a05e635-->
**Auswertung der Einschrittvorhersage:** Bei `onestep_preds = model(features)` beruht jede Vorhersage $\hat{x}_t$ auch im Validierungsbereich $t > 604$ auf den vier **tatsächlich beobachteten** Werten $(x_{t-4}, x_{t-3}, x_{t-2}, x_{t-1})$ aus dem Datensatz.

<!--en:91b01761-->
::: {.slide title="Einschrittvorhersagen folgen der Ground Truth"}

<!--en:2677fb3c-->
[Ergebnisse der Einschrittvorhersage]{.kicker}

<!--en:39032e4e-->
Mit den vier tatsächlich vorangehenden Messwerten als Eingabe in jedem Schritt folgt das lineare Modell der zugrunde liegenden Sinuswelle sowohl in den Trainings- als auch in den Validierungsdaten:

<!--en:a8a43f4c-->
- **Nahezu optimale Entrauschung für 1 Schritt:** Über die gesamte Trajektorie $t \in [5, 1000]$ glätten die Einschrittvorhersagen (magenta gestrichelte Kurve) die gaußschen Schwankungen mit $\sigma = 0.2$ und folgen der tatsächlichen Phase der Welle.

<!--en:f92da609-->
- **Generalisierung über $t = 604$ hinaus:** Die Übergangsdynamik der Sinuswelle ist stationär. Daher gilt die an den ersten 600 Fenstern angepasste lineare Regel auch für die Schritte $605\text{--}1000$, sofern die Eingabefenster tatsächlich beobachtete Werte enthalten.

<!--en:77ea5e75-->
::: {.slide title="Rekursive Mehrschrittvorhersage durch Rückführung der Vorhersagen"}

<!--en:07de0500-->
[Funktionsweise der rekursiven Vorhersage]{.kicker}

<!--en:220bbe27-->
Enden die Beobachtungen bei $t = 604$, erfordert die Vorhersage von $\hat{x}_{605}, \ldots, \hat{x}_{1000}$ eine **rekursive Vorhersage** mit dem autoregressiven Modell. Jeder vorhergesagte Wert $\hat{x}_{t+k}$ geht in das Eingabefenster für die folgenden Schritte ein:

<!--en:af5894b8-->
- **Vorhersage aus eigenen Ausgaben:** Ab Schritt $t + m + 1$ ($k = 5$ für $m = 4$) enthält das Fenster keine einzige tatsächliche Beobachtung mehr. Das Modell arbeitet ausschließlich mit seinen eigenen Ausgaben.

<!--en:2bcd2a8d-->
::: {.slide title="Scheitern der rekursiven Vorhersage nach Schritt 604"}

<!--en:075d484e-->
[Ergebnisse der rekursiven Mehrschrittvorhersage]{.kicker}

<!--en:9fbe045a-->
Trotz seines geringen Validierungsfehlers für 1 Schritt kann dasselbe lineare Modell die Sinuswelle ab $t = 604$ bei rekursiver Vorhersage ohne neue Beobachtungen überhaupt nicht mehr wiedergeben:

<!--en:0651c91c-->
- **Keine Schwingung:** Die rekursive Vorhersage (magenta) beginnt auf dem Niveau des letzten beobachteten Fensters $(x_{601}, x_{602}, x_{603}, x_{604})$. Den Richtungswechseln der Welle folgt sie nicht.

<!--en:cef9d610-->
- **Langsamer exponentieller Abfall auf einen Fixpunkt:** Anstelle einer Schwingung mit Periode $628$ flacht die Trajektorie sofort ab und nähert sich monoton einer konstanten Asymptote. Die tatsächliche Reihe durchläuft währenddessen eine weitere halbe Periode.

<!--en:c02ec21d-->
::: {.slide title="Warum der Einschrittfehler einen gleitenden Mittelwert begünstigt"}

<!--en:af04a726-->
[Statistische Erklärung für das Scheitern]{.kicker}

<!--en:dccbba4f-->
- **Mittelung des Rauschens senkt den Einschritt-MSE:** Gewichte $w_i \approx 0.25$ verringern die Varianz des weißen Rauschens auf ein Viertel ($\sigma^2/4 = 0.01$). Die Verzerrung durch den Phasenversatz für 1 Schritt ist vernachlässigbar ($\sim 10^{-4}$).

<!--en:4781faeb-->
- **Warum der Einschritt-MSE gegen den wahren Oszillator spricht:** Die für die rauschfreie Welle exakte Rekursion $x_t = 2\cos(0.01)x_{t-1} - x_{t-2}$ verstärkt die Rauschvarianz um den Faktor $5\times$ ($5\sigma^2 = 0.20$).

<!--en:552b06d7-->
- **Konvergenz gegen einen Fixpunkt:** Erhält ein gleitender Mittelwert seine eigenen Ausgaben als Eingabe, kann er keine Schwingung fortsetzen. Er klingt gegen $x^* = b / (1 - \sum w_i)$ ab.

<!--en:193ff8de-->
::: {.slide title="Vektorisierte rekursive k-Schritt-Vorhersage für alle Startpositionen"}

<!--en:2e30c887-->
[Vorhersagegüte bei wachsendem Horizont]{.kicker}

<!--en:b7c339bf-->
Um die Vorhersagegüte für die Horizonte $k \in \{1, 4, 16, 64\}$ zu messen, berechnet `k_step_pred` die rekursive Vorhersage für $k$ Schritte gleichzeitig ab jedem gültigen Startindex der Reihe:

<!--en:618dd095-->
- **Phasenverzögerung und Amplitudendämpfung:** Bei $k = 1$ und $k = 4$ folgen die Vorhersagen der Welle eng. Bei $k = 16$ und $k = 64$ ist die Vorhersage eine sichtbar gedämpfte, um $\approx k$ Zeitschritte verzögerte Kopie der Reihe.

<!--en:5197faa4-->
::: {.slide title="Fehlerwachstum durch Phasenverzögerung beim Horizont k"}

<!--en:a6dd649e-->
[Quantitative Analyse des Vorhersagehorizonts]{.kicker}

<!--en:735308f8-->
Der gleitende Mittelwert der 4 zeitversetzten Werte sagt ungefähr das Niveau der Reihe $k$ Schritte zuvor vorher ($\hat{x}_{t+k} \approx \sin(0.01\,t)$). Sein mittlerer quadratischer $k$-Schritt-Fehler entspricht daher dem rauschbedingten Einschrittfehler ($0.05$) zuzüglich der mittleren quadratischen Änderung der Sinuswelle über $k$ Schritte:

<!--en:9f022884-->
**Zwei Fehlerquellen bei rekursiven Vorhersagen:**

<!--en:039a7e74-->
1. **Zu kurzer Horizont des Trainingsziels (hier zu beobachten):** Das mittelnde Modell *dämpft* Störungen ($\sum w_i < 1$), daher wachsen Fehler nicht ungebremst. Das Training für 1 Schritt bot keinen Anreiz, die bei $k = 64$ benötigte Krümmung zu lernen.

<!--en:8d6e4f7c-->
2. **Zunehmende Drift bei rekursiver Vorhersage (Exposure-Bias):** Bei Modellen, die Störungen *verstärken* (etwa autoregressiven Textgeneratoren oder physikalischen Simulatoren), verschiebt jeder Vorhersagefehler die Eingabeverteilung in Schritt $t+1$. Bei langen rekursiven Vorhersagen verstärken sich die Fehler dadurch rasch.

<!--en:823b1d97-->
::: {.slide title="Verständnisfragen zu Autoregression und rekursiver Vorhersage"}

<!--en:929b7671-->
[Wissenskontrolle]{.kicker}

<!--en:e2b5af37-->
Die folgenden drei Rechnungen prüfen das Verständnis der autoregressiven Faktorisierung und der Dynamik rekursiver Vorhersagen:

<!--en:e3b85cfc-->
1. **Kettenregel und Markov-Faktorisierung:** Die exakte Faktorisierung von $P(x_1, x_2, x_3, x_4)$ für 4 Schritte angeben, und zwar (a) nach der allgemeinen Kettenregel und (b) für ein Markov-Modell $2^\textrm{nd}$ Ordnung ($m = 2$). Welcher Faktor ändert sich?

<!--en:5d5545be-->
2. **Exakter Grenzwert des rekursiven gleitenden Mittelwerts:** Ein lineares Modell mit 4 zeitversetzten Werten hat die exakten Gewichte $w_1 = w_2 = w_3 = w_4 = 1/4$ und den Bias $b = 0$. Die letzten 4 Beobachtungen sind $(0.2, 0.4, 0.6, 0.8)$. Wie groß ist die Einschrittvorhersage $\hat{x}_{t+1}$? Gegen welchen exakten Wert konvergiert die rekursive Vorhersage bei unendlich vielen Schritten?

<!--en:b0efc6ac-->
3. **Abwägung zwischen Rauschen und Krümmung:** Die Standardabweichung des Rauschens sinkt von $\sigma = 0.2$ auf $\sigma = 0$. Das Modell verwendet $m = 2$ zeitversetzte Werte. Welche Gewichte $(w_1, w_2)$ lernt die lineare Regression für $(x_{t-2}, x_{t-1})$, und wie verhält sich die rekursive Vorhersage über 400 Schritte?

<!--en:557b98b6-->
**Lösungen:**

<!--en:fbcd49f1-->
1. **(a)** $P(x_1)\,P(x_2 \mid x_1)\,P(x_3 \mid x_1, x_2)\,P(x_4 \mid x_1, x_2, x_3)$. **(b)** Nur der vierte Faktor ändert sich. Ohne $x_1$ lautet er $P(x_4 \mid x_2, x_3)$.

<!--en:4a1b7476-->
2. **$\hat{x}_{t+1} = 0.5$** (damit kehrt sich der Aufwärtstrend sofort um). Der Ausdruck $x_{t-3} + 2x_{t-2} + 3x_{t-1} + 4\hat{x}_t$ bleibt unter der Rekursion $\hat{x}_t = \frac{1}{4}\sum_{i=1}^4 x_{t-i}$ konstant. Daher konvergiert die rekursive Vorhersage gegen $(1(0.2) + 2(0.4) + 3(0.6) + 4(0.8)) / 10 = 6.0 / 10 = 0.60$.

<!--en:2c0b1dc6-->
3. **Exakte Gewichte des Oszillators:** Bei $\sigma = 0$ gelten $w_1 = -1$ (für $x_{t-2}$) und $w_2 = 2\cos(0.01) \approx 1.9999$ (für $x_{t-1}$). Der Einschrittfehler ist null, und die rekursive Vorhersage setzt die Schwingung unbegrenzt exakt fort.

<!--en:292a1eaf-->
::: {.slide title="Grundbegriffe der Sequenzmodellierung"}

<!--en:408fcbdb-->
[Das Wichtigste im Überblick]{.kicker}

<!--en:aab36ff3-->
| Konzept | Mathematische Form | Eigenschaft / Komplexität | Schwierigkeit in der Praxis |

<!--en:42240681-->
| **Faktorisierung nach der Kettenregel** | $P(x_{1:T}) = \prod_{t=1}^T P(x_t \mid x_{<t})$ | Exakt für jede gemeinsame Verteilung, Bewertung mit $\sum_t \log P(x_t \mid x_{<t})$ | Unterlauf beim direkten Produkt $\prod_t P(x_t \mid x_{<t})$ in `float32` für lange Sequenzen |

<!--en:6d50482b-->
| **Markov-Autoregression mit festem Fenster** | $P(x_t \mid x_{<t}) = P(x_t \mid x_{t-m:t-1})$ | Feste Eingabebreite $m$, parallelisierbar über alle $T - m$ Fenster | Keine Erfassung von Abhängigkeiten, die mehr als $m$ Schritte zurückreichen |

<!--en:7d934a30-->
| **Latenter autoregressiver Zustand** | $\mathbf{h}_t = g(\mathbf{h}_{t-1}, x_t)$, $\hat{x}_{t+1} = f(\mathbf{h}_t)$ | Speicherbedarf $O(1)$ bei Zustandsdimension $h$ für ein beliebig langes Präfix $x_{\le t}$ | Aufeinanderfolgende Aktualisierungen über $t$, Informationsverlust durch den Flaschenhals im Zustand |

<!--en:de4e056b-->
| **Stationarität (Zeithomogenität)** | $P(x_t \mid x_{t-m:t-1})$ unabhängig von $t$ | Gemeinsamer Datensatz aus den gleitenden Fenstern aller $t$ | Bei Regimewechseln mit zeitlich veränderlicher Übergangsdynamik verletzt |

<!--en:db60c7d2-->
| **Vorhersage über 1 Schritt und rekursiv über $k$ Schritte** | $\hat{x}_{t+k} = f(\hat{x}_{t+k-m}, \ldots, \hat{x}_{t+k-1})$ | Nach $k > m$ Schritten ausschließlich selbst erzeugte Eingaben | Geringer Einschrittverlust begünstigt lokale Rauschglättung gegenüber längerfristiger Dynamik |
