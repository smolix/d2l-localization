<!--en:b65ef9b9-->
# Sprachmodelle

<!--en:ea0d00aa-->
Eine Smartphone-Tastatur schlägt das nächste Wort vor. Ein Spracherkennungssystem entscheidet, ob jemand „to recognize speech“ oder „to wreck a nice beach“ gesagt hat. Beide müssen einer Textpassage einen numerischen Score zuweisen, der angibt, wie plausibel sie ist. Die beiden englischen Wendungen klingen fast gleich. Allein nach dem Klang lassen sie sich nicht unterscheiden. Die Mehrdeutigkeit löst sich auf, wenn wir wissen, dass die erste Wendung im Englischen geläufig ist und die zweite abwegig klingt. Diese Einschätzung liefert ein *Sprachmodell* (engl. language model). Für eine Sequenz von Token $x_1, \ldots, x_T$ aus einem Vokabular $\mathcal{V}$, etwa den BPE-Token aus :numref:`sec_text-sequence`, schätzt es die gemeinsame Wahrscheinlichkeit $P(x_1, \ldots, x_T)$.

<!--en:cf629f3b-->
Auf den ersten Blick erscheint es aussichtslos, jeder möglichen Sequenz eine Wahrscheinlichkeit zuzuweisen. Schon mit unserem bescheidenen Vokabular von 1024 Token gibt es $1024^{10} \approx 10^{30}$ Sequenzen der Länge zehn. Glücklicherweise zerlegt die Kettenregel der Wahrscheinlichkeitsrechnung (:numref:`sec_sequence`) diese schwer handhabbare gemeinsame Verteilung exakt in ein Produkt bedingter Wahrscheinlichkeiten. Jeder Faktor gibt die Wahrscheinlichkeit des nächsten Tokens unter dem bisherigen Präfix $x_{<t} = x_1, \ldots, x_{t-1}$ an:

<!--en:941cbeb7-->
Ein Sprachmodell sagt also im Kern das nächste Token vorher. Mit denselben bedingten Verteilungen können wir eine vorliegende Sequenz *bewerten*, indem wir die Wahrscheinlichkeiten multiplizieren oder ihre Logarithmen addieren. Wir können auch neuen Text *generieren*, indem wir zunächst $x_1$ ziehen, dann $x_2$ bei gegebenem $x_1$ und so weiter. Erstaunlich viele Aufgaben lassen sich unmittelbar als Textfortsetzung formulieren. Wer den Prompt „Die Novelle ‚Die Zeitmaschine‘ stammt aus der Feder von“ vervollständigt, beantwortet eine Wissensfrage. Genau auf diesem Prinzip beruhen moderne große Sprachmodelle (:numref:`sec_gpt`).

<!--en:1bba1592-->
Wir beginnen mit dem einfachsten Schätzer für diese bedingten Wahrscheinlichkeiten. Wir zählen die Vorkommen in einem Korpus und untersuchen, wo dieser Ansatz an seine Grenzen stößt. Um Verbesserungen objektiv zu bewerten, führen wir die Perplexität ein und vergleichen zwei klassische Glättungsverfahren. Anschließend definieren wir das Maß Bit pro Byte, mit dem wir Modelle mit unterschiedlichen Tokenizern fair vergleichen können. Zum Schluss bauen wir den Minibatch-Datenlader, der die neuronalen Sprachmodelle der nächsten Abschnitte mit tokenisierten Sequenzen versorgt.

<!--en:39db4a36-->
## $n$-Gramm-Sprachmodelle

<!--en:a76b6357-->
### Markov-Modelle und Häufigkeiten

<!--en:60ddaf47-->
Jeder Faktor in :eqref:`eq_lm_factorization` ist durch das gesamte Präfix $x_{<t}$ bedingt. Daher wächst die Zahl verschiedener Kontexte mit $t$ unbegrenzt. Mit der Markov-Annahme (:numref:`subsec_markov-models`) beschränken wir den Kontext auf die letzten $n-1$ Token:

<!--en:c737b2b1-->
Diese Annahme ist grob, denn ein Satz vergisst sein Subjekt kaum nach zwei Wörtern. Dafür begrenzt sie die Zahl der möglichen Kontexte auf einen endlichen Wert. Ein Modell ohne Kontext ($n=1$) heißt *Unigrammmodell*, eines mit einem vorangehenden Token ($n=2$) *Bigrammmodell* und eines mit zwei vorangehenden Token ($n=3$) *Trigrammmodell*. Bei der Maximum-Likelihood-Schätzung ergibt sich jede bedingte Wahrscheinlichkeit als Verhältnis zweier absoluter Häufigkeiten im Korpus. Bezeichnet $N(\cdot)$ die Anzahl der Vorkommen eines Tokentupels im Trainingskorpus, so erhalten wir

<!--en:546f719e-->
also den empirischen Anteil der Fälle, in denen auf den Kontext $(x_{t-n+1}, \ldots, x_{t-1})$ das Token $x_t$ folgte. Ein $n$-Gramm-Sprachmodell ist eine Nachschlagetabelle für diese bedingten relativen Häufigkeiten.

<!--en:b9080919-->
Schätzen wir die Tabellen für Unigramme, Bigramme und Trigramme anhand von *The Time Machine*. Die neuronalen Sprachmodelle später in diesem Kapitel arbeiten mit BPE-Token. Hier zerlegen wir die Novelle in Wörter, damit wir die generierten Passagen leicht lesen können. Wie in :numref:`sec_text-sequence` normalisieren wir den Text so, dass nur Kleinbuchstaben übrig bleiben. Wir trennen die Wörter an Leerraumzeichen und halten die letzten 10% des Textes für die Auswertung zurück. Das Vokabular umfasst alle Wörter aus dem Trainingsteil. Jedes Wort im zurückgehaltenen Text, das im Training fehlt, bilden wir auf das unbekannte Token `<unk>` ab.

<!--en:933be907-->
Die folgende Klasse `NGramLM` speichert für jeden beobachteten Kontext aus $(n-1)$ Token, wie oft jedes nachfolgende Token im Training vorkam. Das genügt, um :eqref:`eq_ngram_mle` auszuwerten. Das Attribut `alpha` brauchen wir in Kürze für die Glättung. Die Methode `sample` verlängert ein gegebenes Präfix, indem sie wiederholt das nächste Token zieht. Seine Wahrscheinlichkeit ist proportional zu seiner Häufigkeit nach den letzten $n-1$ Token.

<!--en:489e2b04-->
Was erfassen diese Tabellen? Fragen wir das Trigrammmodell nach dem Kontext `"the time"`, der in einer Novelle über Zeitreisen häufig vorkommt.

<!--en:188a391a-->
Im Trainingsteil kommt `"the time"` 87-mal vor. In rund der Hälfte der Fälle folgt `"traveller"` und in rund einem Drittel `"machine"`. Die Tabelle erfasst also mühelos die zentralen Wendungen des Romans. Solche häufigen Kontexte sind jedoch die Ausnahme. Für mehr als vier von fünf beobachteten Wortpaaren enthält die Trainingstabelle nur eine einzige Fortsetzung.

<!--en:4732c6df-->
### Sampling aus $n$-Gramm-Modellen

<!--en:174fadf5-->
Beim Generieren von Text wenden wir :eqref:`eq_lm_factorization` direkt an. Wir ziehen das nächste Token aus der bedingten Verteilung für das bisherige Präfix, hängen es an und wiederholen den Vorgang. Flüssig klingender Text kann leicht verbergen, dass ein Modell Passagen wortwörtlich auswendig gelernt hat. Deshalb untersuchen wir unsere Samples mit der Diagnosefunktion `longest_copy`. Sie misst die Länge der längsten zusammenhängenden Wortfolge im generierten Text, die wörtlich im Trainingsdatensatz vorkommt.

<!--en:85013ca3-->
Wir geben jedem Modell das Präfix `"the time"` vor und generieren 25 Wörter. Für den Zufallszahlengenerator geben wir keinen Startwert vor, damit jeder Durchlauf neue Samples liefert.

<!--en:37b1df53-->
Das Unigrammmodell gibt Wörter entsprechend ihrer Häufigkeit im Korpus aus, ohne grammatischen Zusammenhang. Das Bigrammmodell reiht lokal plausible Wortpaare aneinander, verliert aber jeden roten Faden, weil jedes Wort nur von seinem unmittelbaren Vorgänger abhängt. Das Sample des Trigrammmodells klingt dagegen schon verblüffend nach H. G. Wells.

<!--en:816a4a7f-->
Bei einem Korpus von 30 000 Wörtern klingt das Trigrammmodell vor allem deshalb nach Wells, weil vieles davon tatsächlich von Wells *stammt*. Die längste kopierte Wortfolge zeigt das sofort. Beim Unigrammmodell umfasst sie zwei oder drei Wörter, beim Bigrammmodell eine Handvoll und beim Trigrammmodell meist sieben bis zwölf aufeinanderfolgende Wörter. Trifft das Modell beim Sampling auf ein Wortpaar mit nur einem Nachfolger im Training, ist die Fortsetzung fest vorgegeben. So gibt es ganze Teilsätze der Novelle wörtlich wieder. Es setzt auswendig gelernte Fragmente zusammen, anstatt zu generalisieren. (Auf der Grundlage von Milliarden von Wörtern erfassen $n$-Gramm-Tabellen durchaus vielfältige lokale Satzstrukturen des Englischen. Deshalb kamen sie jahrzehntelang in der Spracherkennung und der statistischen maschinellen Übersetzung zum Einsatz.)

<!--en:301e40d4-->
### Sparsität der Häufigkeitstabellen

<!--en:7f06e922-->
Das Auswendiglernen ist das sichtbare Symptom eines tiefer liegenden statistischen Engpasses. Die Zahl der *möglichen* $n$-Gramme beträgt $|\mathcal{V}|^n$ und wächst exponentiell mit $n$. Die Zahl der tatsächlich im Korpus vorkommenden $n$-Gramme wächst hingegen nur linear mit der Korpuslänge. Die daraus entstehende Sparsität lässt sich direkt messen. Wir bestimmen dazu den Anteil der $n$-Gramme im zurückgehaltenen Text, die im Training kein einziges Mal vorkamen.

<!--en:2abafb50-->
Rund jedes zwölfte Wort im zurückgehaltenen Text kam im Training nie vor. Bei den Wortpaaren betrifft das die Hälfte, bei den Worttripeln fast neun von zehn. Der Maximum-Likelihood-Schätzer :eqref:`eq_ngram_mle` weist jedem unbeobachteten $n$-Gramm die Wahrscheinlichkeit null zu. Ein einziger Nullfaktor in :eqref:`eq_lm_factorization` reicht also aus, damit eine ganze sprachlich einwandfreie englische Passage die Wahrscheinlichkeit null erhält.

<!--en:d49ea02e-->
### Worthäufigkeit und das Zipf-Gesetz

<!--en:597eb530-->
Warum werden die Häufigkeitstabellen so schnell dünn besetzt? Der Grund ist die starke Schiefe der Verteilung von Worthäufigkeiten in natürlicher Sprache. Mit der Klasse `Vocab` aus :numref:`sec_text-sequence`, die Token nach absteigender Häufigkeit ordnet, betrachten wir die zehn häufigsten Wörter in *The Time Machine*.

<!--en:3b9c2559-->
Diese zehn häufigsten Wörter verraten nichts über Zeitreisen. Es sind Artikel, Pronomen und Präpositionen (oft *Stoppwörter* genannt). Fast jeder viktorianische Roman ergäbe eine nahezu identische Liste. Auffällig ist, wie steil die Häufigkeiten abfallen. Schon das zehnthäufigste Wort kommt weniger als ein Fünftel so oft vor wie das erste. Tragen wir die Häufigkeit gegen den Rang auf doppelt logarithmischen Achsen auf, ergibt sich jenseits der ersten Ränge nahezu eine Gerade. Das ist charakteristisch für ein Potenzgesetz. Genauer gesagt folgt die Häufigkeit $n_i$ des Wortes auf dem $i^\textrm{th}$ Rang dem *Zipf-Gesetz*:

<!--en:7713d3dc-->
Dabei ist $s$ ein Exponent und $c$ eine Konstante. Die folgende Hilfsfunktion `zipf_fit` schätzt $s$ und $c$, indem sie in doppelt logarithmischer Darstellung eine Gerade an die Ränge 10 bis 1000 anpasst.

<!--en:ee26fdd4-->
Wir wenden `zipf_fit` auf die Worthäufigkeiten `freqs` aus `vocab.token_freqs` an. Dann geben wir den geschätzten Exponenten $s$ aus und zeichnen die Gerade des Potenzgesetzes über die empirischen Häufigkeiten.

<!--en:1d29f40e-->
Der geschätzte Exponent liegt nahe bei 1, wie Zipf es für verschiedene natürliche Sprachen beobachtet hat. Verteilungen nach einem Potenzgesetz haben einen schweren Rand (engl. heavy tail). Jedes einzelne seltene Wort kommt zwar nur selten vor, doch es gibt so viele verschiedene seltene Wörter, dass sie zusammen einen beträchtlichen Teil jedes Textes ausmachen. Verhalten sich Bigramme und Trigramme ähnlich? Wir berechnen ihre Häufigkeitskurven, schätzen ihre Zipf-Exponenten und bestimmen den Anteil verschiedener $n$-Gramme, die nur einmal vorkommen. Anschließend zeichnen wir alle drei Kurven in ein Diagramm.

<!--en:1e359bbf-->
Auch Wortpaare und Worttripel folgen Potenzgesetzen, allerdings mit kleineren Exponenten (rund 0.7 und 0.5). Für häufigkeitsbasierte Modelle ist entscheidend, dass der Rand mit wachsendem $n$ erheblich schwerer wird. Die Hälfte aller verschiedenen Wörter kommt nur einmal vor. Bei den verschiedenen Trigrammen trifft das auf mehr als neun von zehn zu. Ihre geschätzte Wahrscheinlichkeit beruht also auf einer einzigen Beobachtung. Dieser schwere Rand erklärt, warum so viele Trigramme im zurückgehaltenen Text neu sind. Ein Modell, das jeden Kontext unabhängig schätzt, benötigt mit zunehmender Kontextlänge exponentiell mehr Daten.

<!--en:452c1ec1-->
## Perplexität und Glättung

<!--en:0d595c81-->
### Perplexität

<!--en:4780f5ab-->
Die Häufigkeitstabellen erwiesen sich als äußerst dünn besetzt, obwohl unsere generierten Samples das nicht erkennen ließen. Bevor wir die Tabellen verbessern, brauchen wir ein fundiertes quantitatives Maß. Nur so können wir beurteilen, ob eine Änderung die Generalisierung tatsächlich verbessert. Es liegt nahe, die gemeinsame Wahrscheinlichkeit des zurückgehaltenen Textes zu bestimmen. Die Wahrscheinlichkeit einer Sequenz nimmt jedoch mit der Dokumentlänge exponentiell ab. Jedes Modell weist Tolstois *Krieg und Frieden* eine weitaus geringere Wahrscheinlichkeit zu als Saint-Exupérys *Der kleine Prinz*, unabhängig von der Modellgüte. Wir brauchen einen Mittelwert pro Token.

<!--en:1192a422-->
Genau das liefert die Informationstheorie. Wie wir in :numref:`subsec_info_theory_basics` gesehen haben, entspricht der Kreuzentropieverlust eines probabilistischen Klassifikators dem *Überraschungswert* (engl. surprisal) des wahren Labels, also dem negativen Logarithmus der ihm zugewiesenen Wahrscheinlichkeit. Da ein Sprachmodell an jeder Position das nächste Token klassifiziert, bewerten wir es anhand des mittleren Überraschungswerts über alle $T$ Token der Sequenz:

<!--en:3af94cfa-->
Dabei ist $P$ die prädiktive Verteilung des Modells und $x_t$ das tatsächlich beobachtete Token im Schritt $t$. Da wir bei dieser Mittelung durch die Länge teilen, werden Dokumente unterschiedlicher Länge vergleichbar. In der Verarbeitung natürlicher Sprache gibt man üblicherweise den Exponentialwert von :eqref:`eq_avg_ce_for_lm` an. Er heißt *Perplexität* :cite:`Jelinek.Mercer.Bahl.ea.1977`:

<!--en:0bcc4e89-->
Gleichwertig lässt sich die Perplexität als geometrisches Mittel der Kehrwerte der Wahrscheinlichkeiten ausdrücken, die das Modell den beobachteten Token zuweist. Weist es jedem Zieltoken die Wahrscheinlichkeit $1/4$ zu, so beträgt seine Perplexität 4. Die Unsicherheit ist dann genauso groß wie bei der Wahl zwischen vier gleich wahrscheinlichen Kandidaten. Die Perplexität lässt sich daher als *effektiver Verzweigungsfaktor* (engl. effective branching factor) des Modells pro Token verstehen. Drei Grenzfälle veranschaulichen die Skala:

<!--en:c12f668a-->
* Ein Modell, das dem tatsächlichen Token immer die Wahrscheinlichkeit 1 zuweist, erreicht die kleinstmögliche Perplexität von 1.

<!--en:d7d203d9-->
* Weist ein Modell auch nur einem beobachteten Token die Wahrscheinlichkeit 0 zu, ist seine Perplexität unendlich.

<!--en:f8024e04-->
* Ein Baseline-Modell mit Gleichverteilung über das Vokabular erreicht die Perplexität $|\mathcal{V}|$. Damit ist eine Obergrenze gesetzt, die jedes brauchbare Modell unterbieten muss.

<!--en:b37ad5bc-->
Unsere ungeglätteten Maximum-Likelihood-Modelle für $n$-Gramme fallen sofort unter den zweiten Fall. Da sie jedem unbeobachteten $n$-Gramm die Wahrscheinlichkeit null zuweisen, treibt ein einziges neues Trigramm im zurückgehaltenen Text die Perplexität ins Unendliche. Um häufigkeitsbasierte Modelle sinnvoll zu bewerten und zu vergleichen, müssen wir zunächst die Häufigkeiten *glätten* (engl. smoothing), damit keine Nullwerte übrig bleiben.

<!--en:07b230ca-->
### Laplace-Glättung

<!--en:731a7653-->
Das klassische Verfahren ist die *Laplace-Glättung* (auch Add-$\alpha$-Glättung). Wir addieren zu jeder absoluten Häufigkeit eine positive Konstante $\alpha > 0$ und zum Nenner $\alpha |\mathcal{V}|$, damit die resultierende Verteilung normiert bleibt:

<!--en:77add483-->
Jedes unbeobachtete $n$-Gramm erhält nun eine kleine positive Wahrscheinlichkeit. Für einen gänzlich unbeobachteten Kontext ergibt sich die Gleichverteilung $1/|\mathcal{V}|$. Die Idee ist zwei Jahrhunderte älter als die Sprachmodellierung. Laplace führte sie mit $\alpha = 1$ ein, um die Wahrscheinlichkeit dafür zu schätzen, dass die Sonne morgen aufgeht, nachdem sie bislang jeden Tag aufgegangen ist.

<!--en:9e5c7f0e-->
Ergänzen wir zunächst eine Methode `prob` mit Laplace-Glättung in `NGramLM`. Sie implementiert :eqref:`eq_laplace_smoothing` mit der im Modell gespeicherten Glättungskonstanten `alpha`.

<!--en:b62afb89-->
Mithilfe von `prob` summiert `eval_nll` die negative Log-Likelihood über eine Tokensequenz. So können wir die Perplexität auf den zurückgehaltenen Daten `words_val` für jede $n$-Gramm-Ordnung berechnen.

<!--en:b397eda4-->
Mit Laplace-Glättung erzielt ausgerechnet das Trigrammmodell, dessen Samples am überzeugendsten klangen, auf dem zurückgehaltenen Text die mit Abstand *schlechteste* Perplexität. Das Bigrammmodell schneidet kaum besser ab als das Unigrammmodell. Beides erklärt sich durch die Sparsität der Häufigkeitstabellen. Wenn 87% der zurückgehaltenen Trigramme im Training fehlen, laufen die allermeisten Trigrammvorhersagen nach der Glättung auf nahezu gleichverteiltes Raten hinaus. Auch eine Anpassung von $\alpha$ hilft kaum. Selbst mit dem optimalen $\alpha$ aus Aufgabe :exref:`ex_language-model-smoothing-constant` liegt die Trigrammperplexität noch über 2400. Flüssig klingende Samples haben uns also über die Generalisierung getäuscht.

<!--en:a909f686-->
Diese Zahlen zeigen zwei grundlegende Schwächen der Laplace-Glättung. Erstens verteilt sie die Wahrscheinlichkeitsmasse gleichmäßig auf alle unbeobachteten Nachfolger, unabhängig davon, wie plausibel das jeweilige Wort überhaupt ist. Zweitens greift sie nicht auf kürzere Kontexte zurück. Ein unbeobachtetes Trigramm erhält dieselbe pauschale Schätzung, ob seine letzten beiden Wörter nun ein häufiges Bigramm oder ein unplausibles Paar bilden.

<!--en:a56a15c9-->
### Kneser-Ney-Glättung

<!--en:e612c040-->
Die Kneser-Ney-Glättung :cite:`Kneser.Ney.1995` in der interpolierten Fassung von :citet:`Chen.Goodman.1999` behebt beide Schwächen. Zwei Jahrzehnte lang lieferte sie das beste häufigkeitsbasierte Sprachmodell. Dabei bezeichne $u$ einen Kontext aus $n-1$ Token und $w$ das nächste Token. Mit $u'$ bezeichnen wir den kürzeren Kontext, der entsteht, wenn wir das älteste Token von $u$ weglassen. Das Verfahren verbindet drei Ideen.

<!--en:69ac1cd8-->
Die erste Idee ist die *absolute Diskontierung* (engl. absolute discounting). Empirische Häufigkeiten überschätzen systematisch die wahren Häufigkeiten seltener Ereignisse. Ein im Training einmal beobachtetes $n$-Gramm kommt in einer neuen Stichprobe gleicher Größe im Mittel weniger als einmal vor. Der Grund ist der Selektionsbias bei der Auswahl aus der riesigen Menge seltener $n$-Gramme. Ziehen wir von jeder positiven Häufigkeit einen festen Betrag $0 < D < 1$ ab, korrigieren wir diese Verzerrung nach oben. Bei kleinen Häufigkeiten wirkt die Korrektur am stärksten, also gerade dort, wo die Verzerrung am größten ist. Dadurch wird die Wahrscheinlichkeitsmasse $D\,S(u)/N(u)$ im Kontext $u$ frei. Dabei zählt $S(u)$ die verschiedenen Tokentypen, die im Training unmittelbar auf $u$ folgen.

<!--en:ff6c7e41-->
Die zweite Idee ist die *Interpolation*. Wir verteilen die frei gewordene Masse gemäß der Schätzung $P_\textrm{KN}(w \mid u')$ der nächstniedrigeren Ordnung. Diese ist rekursiv auf genau dieselbe Weise definiert, bis hinunter zur Gleichverteilung $1/|\mathcal{V}|$:

<!--en:507c0a9a-->
Bei dieser Rekursion richtet sich die Vorhersage für einen häufig beobachteten Kontext hauptsächlich nach seinen eigenen empirischen Häufigkeiten. Bei einem Kontext mit vielen verschiedenen Nachfolgern erhält der kürzere Kontext mehr Gewicht. Das entspricht der Vorstellung, dass ein Kontext mit vielen bekannten Fortsetzungen eher auch neue zulässt. Kam ein Kontext $u$ im Training nie vor, geht die gesamte Wahrscheinlichkeitsmasse direkt an die niedrigere Ordnung über: $P_\textrm{KN}(w \mid u) = P_\textrm{KN}(w \mid u')$.

<!--en:01548614-->
Die dritte Idee ändert, *was* die Modelle niedrigerer Ordnung zählen. Ihre Verteilung dient vor allem dazu, Wörter zu bewerten, die im längeren Kontext noch nie aufgetreten sind. Sie soll daher erfassen, wie leicht ein Wort in *neuen* Kontexten vorkommt. Seine bloße Häufigkeit im Korpus sagt darüber wenig aus. In unserem Trainingstext kommt `"traveller"` 49-mal vor, aber nur nach zwei verschiedenen Wörtern: 48-mal nach `"time"` und einmal nach `"real"`. Dagegen kommt `"said"` 76-mal nach insgesamt 48 verschiedenen Vorgängern vor. Anhand der absoluten Unigrammhäufigkeit würden wir `"traveller"` in einem unbekannten Kontext fälschlich fast ebenso leicht vorhersagen wie `"said"`. Zählen wir dagegen die verschiedenen Vorgänger, erkennen wir, dass `"traveller"` selten ohne `"time"` vorkommt. (Das klassische Beispiel aus der Sprachverarbeitung ist `"Francisco"`. In Nachrichtentexten kommt es häufig vor, aber fast ausschließlich nach `"San"`.) Daher ersetzt das Kneser-Ney-Verfahren $N(u, w)$ auf jeder Stufe unterhalb der höchsten Ordnung durch die *Fortsetzungshäufigkeit* (engl. continuation count). Sie zählt die verschiedenen Token, die im Training vor $(u, w)$ stehen. Aus diesen Fortsetzungshäufigkeiten berechnen wir analog $N(u)$ und $S(u)$.

<!--en:6bda739e-->
Für den Diskontierungsparameter $D$ gibt es sogar einen Schätzer in geschlossener Form. :citet:`Ney.Essen.Kneser.1994` leiteten für jede Ordnung die Formel $D = c_1 / (c_1 + 2 c_2)$ her. Dabei zählen $c_1$ und $c_2$ die verschiedenen $n$-Gramme dieser Ordnung, die genau einmal beziehungsweise zweimal vorkommen. Unterhalb der höchsten Ordnung verwenden wir die Fortsetzungshäufigkeiten.

<!--en:ee9a8339-->
Die folgende Funktion `kneser_ney` implementiert :eqref:`eq_kneser_ney` vollständig. Sie zählt die $n$-Gramme der höchsten Ordnung und leitet die Fortsetzungshäufigkeiten jeder niedrigeren Ordnung aus den verschiedenen Tupeln der nächsthöheren Ordnung ab. Dann berechnet sie $N(u)$, $S(u)$ und $D$ und wertet die Rekursion für eine Tokensequenz aus. Sie gibt für jedes Token die Log-Wahrscheinlichkeit (in Nat) unter seinem verfügbaren Präfix zurück.

<!--en:4de59dac-->
Bevor wir den gesamten Validierungsteil bewerten, betrachten wir die Kneser-Ney-Glättung für den bekannten Kontext `"the time"`. Wir geben den Diskontierungsbetrag $D$ für Trigramme aus und vergleichen die Wahrscheinlichkeiten, die Kneser-Ney- und Laplace-Glättung drei möglichen Fortsetzungen zuweisen. Das Wort `"traveller"` folgte im Training in 47 von 87 Fällen auf `"the time"`. Die beiden anderen Wörter, `"was"` und `"filby"`, folgten dort nie.

<!--en:2a2d8957-->
Der Diskontierungsbetrag für Trigramme beträgt rund $0.9$. Damit ziehen wir von jedem beobachteten Trigramm fast ein ganzes Vorkommen ab. Das verdeutlicht, wie unsicher Schätzungen auf der Grundlage einer einzigen Beobachtung sind. Bei einem häufigen Trigramm mit 47 Vorkommen fällt der Abzug von $0.9$ kaum ins Gewicht. Die Wahrscheinlichkeit für `"traveller"` bleibt nahe bei der empirischen Häufigkeit $47/87$, während die Laplace-Glättung fast ein Drittel dieser Wahrscheinlichkeitsmasse über das Vokabular verstreut. Bei unbeobachteten Fortsetzungen tritt der Unterschied noch deutlicher hervor. Die Laplace-Glättung behandelt `"was"` und `"filby"` gleich. Die Kneser-Ney-Glättung erkennt dagegen, dass `"was"` auf viele verschiedene Kontexte folgt, während `"filby"` nur in sehr wenigen vorkommt. Ihre Wahrscheinlichkeiten unterscheiden sich dadurch um mehr als eine Größenordnung. Messen wir den Gewinn auf dem zurückgehaltenen Text.

<!--en:de24ffdd-->
Mit der Kneser-Ney-Glättung kehrt sich die Rangfolge vollständig um. Schon ihr Unigrammmodell übertrifft das mit Laplace-Glättung, und ihr Bigrammmodell halbiert die Perplexität in etwa. Ihr Trigrammmodell übertrifft alle anderen häufigkeitsbasierten Modelle. Für unbeobachtete Trigramme geht es stufenlos auf gut kalibrierte Schätzungen aus Bigramm- und Unigrammfortsetzungen über, anstatt gleichverteilt zu raten. Allerdings verbessert sich das Trigrammmodell gegenüber dem Bigrammmodell nur wenig. Die meisten zurückgehaltenen Trigramme sind weiterhin unbekannt, sodass das Modell an den meisten Positionen auf Bigramme zurückgreift. Entscheidend für ein Sprachmodell ist eine niedrige Perplexität auf *unbekanntem* Text. Die Perplexität dient uns daher als wichtigster Maßstab für die folgenden neuronalen Modelle.

<!--en:902dd94e-->
Eine grundlegende Einschränkung teilt die Kneser-Ney-Glättung mit allen häufigkeitsbasierten Verfahren. Glättung verteilt Wahrscheinlichkeitsmasse auf bislang unbesetzte Einträge, kann aber nicht erschließen, welche neuen Kombinationen semantisch plausibel sind. In einer Häufigkeitstabelle ist jedes Token ein einzelnes Symbol ohne Bezug zu den anderen. Die Tabelle kann daher nicht erkennen, dass `"cat"` und `"feline"` syntaktische und semantische Rollen teilen. Ebenso wenig erkennt sie, dass `"I want to eat, grandma"` harmlos klingt, während `"I want to eat grandma"` Anlass zur Sorge gibt. Um statistische Information über verwandte Wörter und Kontexte hinweg zu nutzen, brauchen wir gelernte kontinuierliche Darstellungen. Genau deshalb wenden wir uns in :numref:`sec_rnn` neuronalen Netzen zu.

<!--en:61bb8a82-->
### Bit pro Byte

<!--en:2734e2e1-->
Die Perplexität hat eine entscheidende Einschränkung. Sie hängt von der Tokenisierung ab. Ein Zeichenmodell wählt in jedem Schritt unter 27 Symbolen, ein Wortmodell dagegen unter Tausenden. Die Perplexitäten liegen damit auf völlig verschiedenen Skalen und lassen sich nicht direkt vergleichen. Die Tokenisierung ist aber eine interne Entwurfsentscheidung (:numref:`sec_text-sequence`). Ein faires Gütemaß muss von ihr unabhängig sein. Dazu genügt es, den Nenner zu ersetzen. Wir teilen den gesamten Überraschungswert durch die Anzahl der *Bytes* im ausgewerteten Text. Anders als die Tokenzahl hängt diese Größe allein vom Text selbst ab. Messen wir den Überraschungswert in Bit (mit Logarithmen zur Basis 2), erhalten wir *Bit pro Byte* (bpb):

<!--en:32a6a3d4-->
Dabei ist $B$ die Länge des ausgewerteten Textes in UTF-8-Bytes. Bit pro Byte hat eine unmittelbare praktische Bedeutung als Maß für die verlustfreie Kompression. In Verbindung mit einem arithmetischen Kodierer komprimiert ein Modell mit $\textrm{bpb}$ Bit pro Byte den Text auf den Anteil $\textrm{bpb}/8$ seiner ursprünglichen Größe in Bytes.

<!--en:577da165-->
Um Perplexität und Bit pro Byte direkt zu vergleichen, passen wir Trigrammmodelle an denselben normalisierten Trainingstext an, zunächst mit Laplace-Glättung und dann mit Kneser-Ney-Glättung. Wir verwenden drei Tokenisierungen: Zeichen, Wörter und ein BPE-Vokabular mit 1024 Token. Letzteres trainieren wir mit dem Tokenizer aus :numref:`sec_text-sequence` auf dem normalisierten Trainingsteil. Für einen fairen Vergleich müssen wir ein Detail beachten. Sagt das Wortmodell `<unk>` vorher, ist noch offen, *welches* Wort außerhalb des Vokabulars vorkam. Zur verlustfreien Kodierung des zurückgehaltenen Textes muss das Wortmodell jedes unbekannte Wort buchstabieren. Deshalb addieren wir $\log_2 27$ Bit für jeden Buchstaben und für das abschließende Leerzeichen. Das entspricht den Kosten für die Kodierung eines Zeichens bei Gleichverteilung über die 27 normalisierten Symbole.

<!--en:70815cf7-->
Die Perplexitäten der drei Modelle mit Laplace-Glättung unterscheiden sich um mehr als zwei Größenordnungen. In Bit pro Byte liegen sie dagegen weniger als ein Drittel Bit auseinander. Das Zeichenmodell, das nach der Perplexität zu urteilen weit vorn lag, gewinnt nicht mehr. Bei fester Markov-Ordnung $n$ decken gröbere Token längere Textabschnitte ab. Zwei Wörter Kontext reichen rund zehn Zeichen zurück, zwei Zeichen dagegen kaum eine Silbe. Das Wortmodell läge vorn, wären da nicht die Wörter außerhalb des Vokabulars. Sie zu buchstabieren kostet zusätzlich rund $0.6$ Bit pro Byte und lässt das Modell bei Laplace-Glättung auf den letzten Platz fallen. Bei BPE auf Byte-Ebene entfallen diese Zusatzkosten vollständig. Das ergibt die stärkste Kompression und liefert einen weiteren Grund für die Subwort-Tokenisierung in modernen Sprachmodellen.

<!--en:9cbe8956-->
In der letzten Spalte spart die Kneser-Ney-Glättung sowohl bei BPE- als auch bei Worttrigrammen mehr als ein halbes Bit pro Byte ein. Bei Zeichentrigrammen ändert sich dagegen kaum etwas. Bei nur 27 Symbolen gibt es sehr wenige unbeobachtete Zeichentrigramme, sodass die Glättung wenig bewirkt. Mit rund $1.9$ Bit pro Byte ist das BPE-Trigrammmodell mit Kneser-Ney-Glättung unsere beste häufigkeitsbasierte Baseline.

<!--en:426b3e15-->
Bit pro Byte ist daher das Standardmaß für den Vergleich moderner Sprachmodelle mit unterschiedlichen Vokabularen, etwa eines Modells mit 50 000 Token und eines mit 200 000 Token. Drei Vergleichswerte helfen bei der Einordnung. In seinen klassischen Vorhersageexperimenten mit Menschen schätzte Shannon die Entropie gedruckter englischer Texte auf rund 1 Bit pro Zeichen :cite:`Shannon.1951`. Moderne neuronale Sprachmodelle komprimieren vielfältige Texte aus dem Web auf deutlich unter 1 Bit pro Byte. Die häufigkeitsbasierten Modelle dieses Abschnitts liegen zwischen rund $1.9$ und $2.8$ Bit pro Byte.

<!--en:5a5d5b73-->
## Aufteilung von Sequenzen

<!--en:ee371335-->
Die neuronalen Sprachmodelle der nächsten Abschnitte verarbeiten Minibatches aus gleich langen Teilsequenzen von BPE-Token. Die Token liefert der Tokenizer aus :numref:`sec_text-sequence`. Bauen wir nun die zugehörige Datenpipeline auf.

<!--en:004ac7b6-->
Stellen wir uns das tokenisierte Korpus als einen durchgehenden Strom von Token-IDs $x_1, x_2, x_3, \ldots$ vor. Ein Eingabefenster der Länge $T$ mit Startindex $t$ enthält $x_t, \ldots, x_{t+T-1}$. Das zugehörige Ziel ist dasselbe Fenster, um ein Token nach rechts verschoben: $x_{t+1}, \ldots, x_{t+T}$. Das Modell durchläuft das Eingabefenster und sagt an jeder Position das nächste Token vorher. Jedes Fenster liefert also $T$ Zielwerte für die überwachte Vorhersage des nächsten Tokens. Bis auf die letzten $T$ Token ist jede Position im Korpus ein gültiger Fensteranfang. Benachbarte Fenster überlappen sich bis auf ein Token (:numref:`fig_lang_model_data`). Im Training ziehen wir Minibatches dieser Fenster nach dem Zufallsprinzip. So erhalten wir die Durchmischung, die der stochastische Gradientenabstieg (engl. stochastic gradient descent, SGD) benötigt.

<!--en:ba53549f-->
![Die ersten beiden Eingabefenster der Länge $T = 5$ über den BPE-Token am Anfang von *The Time Machine* (␣ kennzeichnet ein führendes Leerzeichen). Jedes Zielfenster beginnt ein Token später als sein Eingabefenster. Aufeinanderfolgende Fenster überlappen sich bis auf ein Token. Ein Minibatch fasst zufällig gezogene Fenster zusammen.](../img/mdl-rnn-partitioning.svg)

<!--en:1025ad20-->
Der folgende Konstruktor von `TimeMachine` setzt die Pipeline zusammen. Er lädt die Novelle herunter, trainiert den BPE-Tokenizer aus :numref:`sec_text-sequence` und kodiert den Text als ganzzahlige IDs. Alle überlappenden Fenster der Länge `num_steps` fasst er zu zwei aufeinander abgestimmten Arrays für die Eingaben (`X`) und Ziele (`Y`) zusammen. Der trainierte Tokenizer steht sowohl in `data.tokenizer` als auch in `data.vocab`. So können wir IDs wieder zu Text dekodieren und bei jeder Tokenisierungsart einheitlich `len(data.vocab)` abfragen.

<!--en:0013d7d6-->
MXNet-Arrays enthalten standardmäßig float32-Zahlen. Deshalb fordert der MXNet-Konstruktor ausdrücklich ganzzahlige Token-IDs an.

<!--en:c72f1753-->
Wir reservieren die ersten `num_train` Fenster für das Training und `num_val` spätere Fenster für die Validierung. Benachbarte Fenster überlappen sich stark. Die Fenster unmittelbar nach dem Trainingsteil würden daher Token enthalten, die bereits als Trainingsziele vorkamen. Als Puffer lassen wir an der Grenze `num_steps` Fenster aus. Das verhindert Datenleckage (engl. data leakage) zwischen Training und Validierung. Bei der Validierung durchlaufen wir die Minibatches der Reihe nach, ohne sie zu mischen.

<!--en:cdd0d6f8-->
Jeder Minibatch besteht aus zwei ganzzahligen Tensoren der Form (`batch_size`, `num_steps`). Dekodieren wir eine Beispielzeile, sehen wir, dass `X` die Prosa von Wells enthält und die Sequenz in `Y` genau ein Token später beginnt.

<!--en:11e057a8-->
Ein Fenster aus BPE-Token liefert weit mehr Kontext, als eine Häufigkeitstabelle nutzen kann. Unseren Worttabellen fehlten schon bei einem Kontext von zwei Wörtern die nötigen Daten. Im nächsten Abschnitt ersetzen wir die Nachschlagetabellen durch ein neuronales Netz, das ein beliebig langes Präfix in einem gelernten versteckten Zustand verdichtet.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:12388241-->
Ein Sprachmodell weist Tokensequenzen Wahrscheinlichkeiten zu. Die Kettenregel der Wahrscheinlichkeitsrechnung führt diese Aufgabe auf die Vorhersage des nächsten Tokens zurück. So kann dasselbe Modell Text bewerten und generieren. Unter der Markov-Annahme schätzen $n$-Gramm-Modelle die Wahrscheinlichkeiten des nächsten Tokens direkt aus den Häufigkeiten im Korpus. Sie sind leicht aufzubauen und können lokal flüssige Samples erzeugen. Allerdings wächst der Raum möglicher Kontexte exponentiell, und die Häufigkeitsverteilungen haben nach dem Zipf-Gesetz schwere Ränder. Die meisten beobachteten $n$-Gramme kommen daher nur einmal vor. Die Modelle lernen auswendig, anstatt zu generalisieren. Die Perplexität ist der Exponentialwert der mittleren Kreuzentropie pro Token. Sie misst den effektiven Verzweigungsfaktor eines Modells und zeigt, warum Trigramme mit Laplace-Glättung auf zurückgehaltenem Text schlecht abschneiden. Die interpolierte Kneser-Ney-Glättung diskontiert beobachtete Häufigkeiten und verteilt die frei gewordene Masse entsprechend den Fortsetzungshäufigkeiten niedrigerer Ordnung. Das liefert die besten häufigkeitsbasierten Modelle. Doch auch sie können nicht über semantisch verwandte Token hinweg generalisieren. Da die Perplexität von der Granularität des Vokabulars abhängt, verwenden wir für faire Modellvergleiche Bit pro Byte als tokenisierungsunabhängiges Kompressionsmaß. Für die neuronalen Sprachmodelle der nächsten Abschnitte zerlegen wir schließlich das BPE-tokenisierte Korpus in überlappende Paare von Eingabe- und Zielfenstern fester Länge. Eine Lücke vor dem Validierungsteil verhindert Datenleckage.

<!--en:2294243a-->
## Übungen

<!--en:30a1fcba-->
1. **Dichte und dünn besetzte Speicherung.** Betrachten Sie ein 5-Gramm-Modell mit einem Vokabular von $|\mathcal{V}| = 50{,}000$ Worttoken.
    1. Wie viele Einträge enthält eine dicht besetzte Tabelle der bedingten Wahrscheinlichkeiten :eqref:`eq_ngram_mle`?
    1. Nehmen Sie an, Sie speichern nur die 5-Gramme, die in einem Korpus von einer Billion Token vorkommen, mit jeweils 16 Bytes pro Eintrag. Wie viel Speicher benötigen Sie höchstens? Welchen Anteil aller möglichen 5-Gramme decken Sie damit ab?
    1. Das Wortvokabular aus :numref:`subsec_markov-models-and-n-grams` hat $|\mathcal{V}| = 4{,}316$ Einträge. Der dortige Code gibt die Anzahl verschiedener Worttrigramme im Trainingstext aus. Welcher Anteil aller möglichen Trigramme kommt im Training vor? Welcher Anteil der zurückgehaltenen Trigramme kommt dort vor? Schätzt die in (b) berechnete Abdeckung also gut, wie oft eine Abfrage für neuen Text das gesuchte 5-Gramm in der Tabelle findet?

<!--en:fcd0f1f2-->
1. [code] **Glättungskonstante.** Variieren Sie $\alpha$ in :eqref:`eq_laplace_smoothing` über die 19 Werte $\alpha = 10^{k/4}$ für $k = -16, -15, \ldots, 2$, jeweils für das Unigramm-, Bigramm- und Trigrammmodell auf Wortebene. Tragen Sie die Perplexität auf zurückgehaltenem Text gegen $\alpha$ auf einer logarithmischen Achse auf.
    1. Wie verändert sich das beste $\alpha$ mit $n$ und warum?
    1. Ordnen Sie die drei Modelle, jedes mit seinem besten $\alpha$, nach der Perplexität auf zurückgehaltenem Text. Wie unterscheidet sich diese Rangfolge von der Ausgabe in :numref:`subsec_perplexity` mit dem Standardwert $\alpha = 0.01$ von `NGramLM`?
   :exlabel:`ex_language-model-smoothing-constant`

<!--en:1e5fc863-->
1. **Perplexität bei schiefer Verteilung.** In einem Trainingskorpus mit zehn Symbolen entfallen 91% der Positionen auf ein Symbol und jeweils 1% auf jedes der neun übrigen.
    1. Berechnen Sie die Perplexität des aus diesem Korpus geschätzten Unigrammmodells auf einem langen Testtext aus derselben Verteilung. Wie verhält sie sich zur Vokabulargröße von zehn?
    1. Das Unigrammmodell auf Wortebene aus :numref:`subsec_perplexity` wählt unter Tausenden von Worttypen. Lesen Sie seine Perplexität auf zurückgehaltenem Text als effektive Anzahl gleich wahrscheinlicher Wahlmöglichkeiten ab. Erklären Sie mit dem Zipf-Gesetz :eqref:`eq_zipf_law`, warum diese Zahl weit unter $|\mathcal{V}|$ liegt.

    *Nach Daniel Jurafsky und James H. Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), Entwurf der 3. Auflage, Aufgabe 3.12.*

<!--en:ab398937-->
1. **Überlappende Fenster.** `d2l.TimeMachine` bildet für jede mögliche Startposition im Korpus eine Zeile aus `num_steps + 1` aufeinanderfolgenden Token. Die ersten `num_steps` Token einer Zeile ergeben eine Zeile von `X`, die letzten `num_steps` Token die zugehörige Zeile von `Y`. Die ersten `num_train` Zeilen dienen dem Training, die nächsten `num_steps` Zeilen entfallen und die darauffolgenden `num_val` Zeilen dienen der Validierung (:numref:`subsec_partitioning-seqs`). Verwenden Sie `num_steps=32`, `num_train=10000` und `num_val=5000`.
    1. Welche Korpuspositionen erfassen die Trainingszeilen und welche die Validierungszeilen, wenn Sie Eingabe- und Zieltoken zusammen betrachten? Wie viele Positionen wären beiden gemeinsam, wenn keine Zeilen entfielen? Warum würde das die Validierungsgüte zu hoch erscheinen lassen?
    1. Wie oft pro Epoche dient ein Token im Inneren des Trainingsbereichs als Ziel?
    1. Wie viele Fenster ergeben sich, wenn Sie das gesamte Korpus von etwa 66 000 Token in überlappungsfreie Fenster zu je 32 Token aufteilen? Vergleichen Sie diese Aufteilung mit den 10 000 überlappenden Trainingszeilen hinsichtlich der Fensterzahl und der Anzahl verschiedener erfasster Korpuspositionen.

<!--en:79f73761-->
1. [code] **Normierung über verschiedene Längen.** ● `NGramLM` hat kein Sequenzendtoken. Trainieren Sie ein Bigrammmodell der Klasse `NGramLM` mit $\alpha = 1$ und `vocab_size=3` auf einer kurzen Zeichenkette aus einem Vokabular von drei Symbolen. Summieren Sie mit `prob` und der Kettenregel :eqref:`eq_lm_factorization` die Wahrscheinlichkeiten aller Sequenzen der Länge 2 und getrennt davon die aller Sequenzen der Länge 3.
    1. Welchen Wert hat jede Summe? Was sagen die beiden Summen über das Modell als Verteilung über Sequenzen aus, deren Länge nicht vorab feststeht?
    1. Erweitern Sie das Vokabular um ein Sequenzendtoken, sodass `vocab_size=4` gilt, und hängen Sie es an die Trainingszeichenkette an. Betrachten Sie eine Sequenz als vollständig, sobald das Modell dieses Token ausgibt. Zeigen Sie, dass die Summe der Wahrscheinlichkeiten aller vollständigen Sequenzen nun genau eins beträgt. Welche Eigenschaft des Modells schließt Wahrscheinlichkeitsmasse auf Sequenzen aus, die nie enden?

    *Nach Daniel Jurafsky und James H. Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), Entwurf der 3. Auflage, Aufgabe 3.5.*

<!--en:8242db42-->
1. [code] **Beobachtete und unbeobachtete Trigramme.** ● Teilen Sie die Positionen im zurückgehaltenen Text ab dem dritten Wort in zwei Gruppen ein. Entscheidend ist, ob das an der jeweiligen Position endende Worttrigramm im Training vorkam. Verwenden Sie das Bigramm- und das Trigrammmodell auf Wortebene, jeweils mit dem besten $\alpha$ aus der Versuchsreihe in Aufgabe :exref:`ex_language-model-smoothing-constant`. Berechnen Sie die mittlere negative Log-Likelihood jedes Modells für jede Gruppe. In welcher Gruppe ist das Trigrammmodell besser? Wie nutzt die Kneser-Ney-Glättung :eqref:`eq_kneser_ney` dies aus?

<!--en:d7dcec1c-->
[Diskussion](https://d2l.discourse.group/t/117)

<!--en:7c88790f-->
[Diskussion](https://d2l.discourse.group/t/118)

<!--en:561d9376-->
[Diskussion](https://d2l.discourse.group/t/1049)

<!--en:08430c7a-->
[Diskussion](https://d2l.discourse.group/t/18012)

<!--en:80d3c89f-->
Sprachmodelle, Perplexität und Bit pro Byte<br>**Zerlegung mit der Kettenregel · Maximum-Likelihood-Schätzung für n-Gramme · Zipf-Gesetz · Perplexität · Laplace- und Kneser-Ney-Glättung · Bit pro Byte · gleitende Fenster**

<!--en:eddb29b8-->
::: {.slide title="Das Grundproblem der statistischen Sprachmodellierung"}

<!--en:412fafa0-->
[Motivation und Kettenregel]{.kicker}

<!--en:06fe742c-->
- **Die $10^{30}$ möglichen Sequenzen beherrschen:** Bei 1024 Token im Vokabular gibt es $1024^{10} \approx 10^{30}$ Sequenzen der Länge 10. Die Kettenregel führt die Schätzung der gemeinsamen Dichte auf die Klassifikation des nächsten Tokens in jedem Schritt $t$ zurück.

<!--en:3a388259-->
- **Eine Verteilung, zwei Anwendungen:** Die bedingten Verteilungen $P(x_t \mid x_{<t})$ **bewerten** mögliche Passagen (z. B. zur Unterscheidung gleich klingender Ausdrücke in der automatischen Spracherkennung) und **generieren** durch Ancestral Sampling von links nach rechts neuen Text.

<!--en:1a7782c6-->
::: {.slide title="Markov-Annahme und Maximum-Likelihood-Schätzung für n-Gramme"}

<!--en:52a1f49e-->
[Häufigkeitsbasierte Sprachmodelle]{.kicker}

<!--en:4068f894-->
Da das vollständige Präfix $x_{<t}$ unbegrenzt wächst, verwendet ein **$n$-Gramm-Sprachmodell** eine Markov-Näherung der Ordnung $(n - 1)^\textrm{th}$. Es schätzt jede bedingte Wahrscheinlichkeit aus dem empirischen Verhältnis der absoluten Häufigkeiten $N(\cdot)$ im Trainingskorpus:

<!--en:f991a26f-->
- **Ordnungen $n = 1, 2, 3$:** Unigramm ($n = 1$, kein Kontext), Bigramm ($n = 2$, 1 vorangehendes Token) und Trigramm ($n = 3$, 2 vorangehende Token).

<!--en:083c66bf-->
- **Lokale Wendungen und erzwungene Fortsetzungen:** Nach `"the time"` ($N = 87$) weist das Worttrigramm die Wahrscheinlichkeit $0.54$ dem Wort `"traveller"` und $0.31$ dem Wort `"machine"` zu. Doch **bei $84\%$ aller im Training beobachteten Wortpaare kommt nur eine einzige Fortsetzung vor**.

<!--en:72b9e203-->
::: {.slide title="Häufigkeitsbasierte n-Gramm-Modelle aufbauen und daraus ziehen"}

<!--en:c8bf2621-->
[Implementierung von Grund auf]{.kicker}

<!--en:f6ad2b70-->
`NGramLM` ordnet jedem beobachteten Kontext aus $(n - 1)$ Token einen `Counter` mit den Häufigkeiten seiner Nachfolger zu:

<!--en:8a28df16-->
::: {.slide title="Warum n-Gramme höherer Ordnung täuschen können"}

<!--en:d03584e4-->
[Samples beurteilen und Generalisierung prüfen]{.kicker}

<!--en:5a8f9715-->
Zeugen flüssig klingende Samples von Generalisierung oder sind sie wörtliche Kopien? `longest_copy` misst die Länge der längsten zusammenhängenden Wortfolge, die direkt aus dem Trainingstext stammt:

<!--en:dfa57feb-->
**Warum Samples täuschen können:** Das Unigrammmodell erzeugt Wortsalat, das Bigrammmodell verkettet lokal plausible Paare und das Trigrammmodell klingt verblüffend nach H. G. Wells. **Es kopiert nämlich 7 bis 12 aufeinanderfolgende Wörter wörtlich**, sobald es beim Sampling auf eines der $84\%$ Wortpaare mit nur einer Fortsetzung im Training trifft.

<!--en:53e06b39-->
::: {.slide title="Grenzen durch Sparsität bei exponentiell wachsendem Kontextraum"}

<!--en:160996a8-->
[Statistischer Engpass bei Häufigkeitstabellen]{.kicker}

<!--en:8ad9398e-->
Warum lernen $n$-Gramme höherer Ordnung den Trainingstext auswendig und versagen bei zurückgehaltenem Text? Die Zahl der *möglichen* $n$-Gramme wächst exponentiell gemäß $|\mathcal{V}|^n$, der Korpusumfang dagegen nur linear:

<!--en:57bbbada-->
- **Nullhäufigkeiten im zurückgehaltenen Text:** Von den letzten $10\%$ von *The Time Machine* kamen $8.4\%$ der Unigramme (auf `<unk>` abgebildet), $51.5\%$ der Wortbigramme und **$87.0\%$ der Worttrigramme** im Trainingsteil nie vor.

<!--en:ed4c65be-->
- **Speicherbedarf:** Anstelle eines dicht besetzten Arrays der Größe $|\mathcal{V}|^n$ ($4316^3 \approx 8 \times 10^{10}$ Einträge) speichert eine verschachtelte Hashtabelle `self.cont[ctx][token]` nur die $n$-Gramme mit positiver Häufigkeit im Training.

<!--en:1370743d-->
- **Wahrscheinlichkeit null für die ganze Passage:** Bei der Maximum-Likelihood-Schätzung ($\hat{P} = N(\text{trigram}) / N(\text{bigram})$) erhält jedes unbeobachtete $n$-Gramm die Wahrscheinlichkeit $0$. Schon ein neues Worttripel ergibt damit $\prod_{t=1}^T \hat{P}(x_t \mid x_{<t}) = 0$ für die gesamte Validierungspassage.

<!--en:4357e78e-->
::: {.slide title="Das Zipf-Potenzgesetz für Worthäufigkeiten"}

<!--en:05986183-->
[Statistik natürlicher Sprache]{.kicker}

<!--en:5e5f833c-->
Häufigkeitstabellen werden so schnell dünn besetzt, weil die Worthäufigkeiten in natürlicher Sprache extrem schief verteilt sind. Über die Ränge $i = 1, 2, \ldots, |\mathcal{V}|$ folgen sie dem **Zipf-Potenzgesetz**:

<!--en:d5fef681-->
- **Stoppwörter dominieren:** Die 10 häufigsten Wörter (*the, I, and, of, a, to, was, in, that, my*) sind Funktionswörter. Das Wort auf Rang 10 (`'my'`, $440$) kommt bereits $5\times$ seltener vor als das auf Rang 1 (`'the'`, $2{,}261$).

<!--en:8c864e35-->
- **Gerade im doppelt logarithmischen Diagramm:** Die Anpassung über die Ränge $10\text{--}1000$ liefert den Zipf-Exponenten $s = 1.04 \approx 1$.

<!--en:a218c335-->
::: {.slide title="Von Unigrammen zu Trigrammen nimmt der Anteil einmaliger Vorkommen zu"}

<!--en:6fff9261-->
[Sparsität nach dem Zipf-Gesetz]{.kicker}

<!--en:4fc832a4-->
Die Rang-Häufigkeits-Kurven für Wortunigramme, -bigramme und -trigramme im selben doppelt logarithmischen Diagramm zeigen, wie die Ränder der Potenzgesetze abflachen. Dort überwiegen zunehmend einmalige Vorkommen ($N = 1$), wenn $n$ wächst:

<!--en:172bddaa-->
- **Abnehmende Zipf-Exponenten:** Der Betrag der Steigung sinkt von $s = 1.04$ (Unigramme) auf $s = 0.69$ (Bigramme) und $s = 0.47$ (Trigramme).

<!--en:272dd8a1-->
- **Immer mehr einmalige Vorkommen:** **$52\%$ der verschiedenen Wörter**, **$82\%$ der verschiedenen Bigramme** und **$95\%$ der verschiedenen Trigramme** in *The Time Machine* kommen im gesamten Korpus nur *einmal* vor.

<!--en:10bd0129-->
- **Folge:** Die unabhängige Schätzung für jedes Kontexttupel erfordert mit jedem zusätzlichen Kontexttoken exponentiell größere Korpora.

<!--en:97d19d6b-->
::: {.slide title="Perplexität als effektiver Verzweigungsfaktor pro Token"}

<!--en:d2d03114-->
[Quantitatives Gütemaß]{.kicker}

<!--en:97c865eb-->
- **Effektiver Verzweigungsfaktor:** Erhält jedes Zieltoken die Wahrscheinlichkeit $1/k$, so gilt $\textrm{PPL} = k$. Die Unsicherheit entspricht in jedem Schritt der eines Wurfs mit einem fairen $k$-seitigen Würfel.

<!--en:66f0631d-->
- **Kalibrierung der Skala:** Sie reicht von $\textrm{PPL} = 1$ (deterministisches Orakel) über $|\mathcal{V}|$ (gleichverteiltes Raten) bis $\infty$, sobald ein Testtoken die Wahrscheinlichkeit $0$ erhält.

<!--en:8d42b5a7-->
::: {.slide title="Laplace-Glättung (Add-α) und ihre Grenzen"}

<!--en:7a28ba67-->
[Klassische Glättung von Häufigkeiten]{.kicker}

<!--en:379af587-->
Die **Laplace-Glättung (Add-$\alpha$)** beseitigt Nullwahrscheinlichkeiten. Sie addiert die Pseudohäufigkeit $\alpha > 0$ zu jedem $n$-Gramm-Zähler und $\alpha|\mathcal{V}|$ zum jeweiligen Kontextnenner:

<!--en:b3eb89a2-->
- **Flüssiger Text trotz hoher Perplexität:** Bei Laplace-Glättung ($\alpha = 0.01$) liefert das Trigrammmodell die Samples, die am ehesten nach Wells klingen. Dennoch hat es die **schlechteste Perplexität auf zurückgehaltenem Text ($2{,}481.4$ gegenüber $753.6$ beim Bigrammmodell)**.

<!--en:df1412c7-->
- **Warum die Laplace-Glättung bei $n = 3$ versagt:** $87\%$ der Validierungstrigramme fehlen im Training. Deshalb liefert die Laplace-Glättung an den meisten Testpositionen eine Vorhersage nahe der Gleichverteilung. Ist der Kontext aus zwei Wörtern selbst unbekannt, gilt genau $1/|\mathcal{V}|$. Das trifft auf etwa die Hälfte der Testpositionen zu. Dabei nutzt die Glättung weder die allgemeine Worthäufigkeit noch den beobachteten Bigrammkontext $(x_{t-1})$ für den Rückgriff auf kürzere Kontexte.

<!--en:f4873a7b-->
::: {.slide title="Interpolierte Kneser-Ney-Glättung mit Diskontierung und Rückgriff auf kürzere Kontexte"}

<!--en:7a929a77-->
[Glättung für die besten häufigkeitsbasierten Modelle]{.kicker}

<!--en:1cbf0dc6-->
- **Absolute Diskontierung ($D \approx 0.93$):** Von jeder positiven Häufigkeit wird $D = c_1 / (c_1 + 2c_2)$ abgezogen. Das korrigiert den Selektionsbias bei einmaligen Vorkommen und setzt die Masse $D\,S(u)/N(u)$ frei.

<!--en:9f5c2584-->
- **Rekursive Interpolation:** Die frei gewordene Masse geht an den kürzeren Suffixkontext $u' = u_{2:}$, bis hinunter zur Gleichverteilung $1/|\mathcal{V}|$.

<!--en:89be6617-->
- **Fortsetzungshäufigkeiten bei niedrigeren Ordnungen:** Entscheidend ist, wie viele *verschiedene Vorgänger* vor $w$ stehen. So erhält das vielseitige `"said"` mehr Gewicht als das kontextgebundene `"traveller"`.

<!--en:0ce350cf-->
::: {.slide title="Implementierung der interpolierten Kneser-Ney-Glättung in Python"}

<!--en:c8bf2621-2-->
[Implementierung von Grund auf]{.kicker}

<!--en:08f97d4a-->
`kneser_ney` leitet Fortsetzungshäufigkeiten ab, schätzt $D_k = c_1 / (c_1 + 2c_2)$ und wertet die Rekursion aus:

<!--en:e3eb5f91-->
::: {.slide title="Kneser-Ney- und Laplace-Glättung bei bekannten und unbekannten Wörtern"}

<!--en:18c4776a-->
[Empirischer Vergleich der Glättungsverfahren]{.kicker}

<!--en:de8dbc27-->
Der Vergleich von Kneser-Ney- und Laplace-Glättung ($\alpha = 0.01$) nach dem Kontext `"the time"` ($N = 87$) und auf dem gesamten Validierungsteil zeigt, warum die Kneser-Ney-Glättung zwei Jahrzehnte lang die statistische Sprachverarbeitung dominierte:

<!--en:e6e980b2-->
- **Häufige $n$-Gramme erhalten:** Bei der bekannten Fortsetzung `"traveller"` ($N = 47$) bleibt die Schätzung nach Abzug von $D \approx 0.93$ nahe an der empirischen Häufigkeit: $P_{\textrm{KN}} = 0.53 \approx 47/87$. Die Laplace-Glättung verteilt ein Drittel der Masse ($\hat{P}_\alpha = 0.36$) auf $4{,}316 \times 0.01 = 43.16$ fiktive Vorkommen.

<!--en:c4fab331-->
- **Unbekannte Fortsetzungen unterscheiden:** Die Laplace-Glättung weist den unbeobachteten Fortsetzungen `"was"` und `"filby"` dieselbe Wahrscheinlichkeit zu ($7.7 \times 10^{-5}$). Bei der Kneser-Ney-Glättung ist die Wahrscheinlichkeit des vielseitigen `"was"` **$16\times$ so hoch** wie die des kontextgebundenen `"filby"`. Die Trigrammperplexität auf zurückgehaltenem Text sinkt auf ein Achtel (Verbesserung: $8\times$) ($2{,}481.4 \to 300.1$).

<!--en:36491e05-->
::: {.slide title="Warum Perplexität keinen Vergleich verschiedener Tokenizer erlaubt"}

<!--en:fdbd278b-->
[Tokenizerunabhängige Auswertung]{.kicker}

<!--en:d3c5cfe4-->
- **Unterschiedliche Skalen:** Bei der Perplexität steht die Tokenzahl $T$ im Nenner. Daher liegen Zeichenmodell ($\textrm{PPL} \approx 6.4$), BPE-1024-Modell ($\textrm{PPL} \approx 63$) und Wortmodell ($\textrm{PPL} \approx 300$) auf unterschiedlichen Skalen.

<!--en:6cebe169-->
- **Bit pro Byte ($\textrm{bpb}$):** Aus dem gesamten Überraschungswert zur Basis 2, geteilt durch die unveränderliche Textlänge $B$ in UTF-8-Bytes, ergibt sich die genaue Rate der verlustfreien Kompression ($\textrm{bpb}/8$).

<!--en:96e41484-->
::: {.slide title="Zeichen, BPE und Wörter im Vergleich in Bit pro Byte"}

<!--en:48f83f93-->
[Vergleich verschiedener Tokenisierungsgranularitäten]{.kicker}

<!--en:9c05b93b-->
Derselbe zurückgehaltene Text liefert einen gemeinsamen Maßstab für die Trigrammmodelle für Zeichen, BPE-1024 und Wörter. Beim Wortmodell kommen $\log_2 27$ Bit pro Buchstabe hinzu, um jedes `<unk>`-Token auszuschreiben:

<!--en:284cbfea-->
- **Vom Faktor $400\times$ bei der Perplexität zu $0.2$ bpb Differenz:** Bei Laplace-Glättung reicht die Perplexität von $6.4$ bis $2{,}481.4$, die Werte in Bit pro Byte liegen dagegen im Bereich $2.58\text{--}2.81$ bpb.

<!--en:7515a26d-->
- **Warum BPE-1024 gewinnt ($1.90$ bpb mit Kneser-Ney):** Zwei BPE-Token umfassen $\sim 6$ Zeichen (gegenüber 2 beim Zeichentrigramm). Dabei entstehen **keine Zusatzkosten für das Ausschreiben von `<unk>`** (beim Worttrigramm sind das $\approx 0.6$ bpb).

<!--en:3b75e339-->
::: {.slide title="Gleitende Eingabe- und Zielfenster für das Training neuronaler Modelle"}

<!--en:67c7b5cb-->
[Aufteilung von Sequenzdaten]{.kicker}

<!--en:53e7a41d-->
Für das Training der neuronalen Sprachmodelle aus §{sec_rnn}–§{sec_decoding} mit Minibatch-SGD wird das BPE-kodierte Korpus $x_1, x_2, \ldots$ in überlappende Fenster der Länge $T = \texttt{num\_steps}$ zerlegt:

<!--en:db8bb7a7-->
- **Eingabe und um einen Schritt verschobenes Ziel:** Ein Fenster ab Index $i$ ergibt die Eingabezeile $\mathbf{X}_{i,:} = (x_i, \ldots, x_{i+T-1})$ und die Zielzeile $\mathbf{Y}_{i,:} = (x_{i+1}, \ldots, x_{i+T})$.

<!--en:67ddad03-->
- **Ein Ziel an jeder Position ($T$ Ziele pro Zeile):** In jedem Schritt $t \in \{1, \ldots, T\}$ im Fenster sagt das Modell das nächste Token $x_{i+t}$ aus dem Präfix $(x_i, \ldots, x_{i+t-1})$ vorher.

<!--en:451aa98f-->
- **Zufällige Auswahl der Minibatches:** Aus den gemischten Startindizes der Fenster entstehen Minibatches der Form `(batch_size, num_steps)`. Das verringert die Korrelation der Gradienten aufeinanderfolgender SGD-Updates.

<!--en:e3911ef1-->
::: {.slide title="Datenleckage zwischen Trainings- und Validierungsfenstern verhindern"}

<!--en:dee0267b-->
[Datenpipeline mit Schutz vor Datenleckage]{.kicker}

<!--en:3e70d648-->
::: {.slide title="Datenleckage zwischen Trainings- und Validierungsfenstern verhindern (Fortsetzung)"}

<!--en:dee0267b-2-->
[Datenpipeline mit Schutz vor Datenleckage]{.kicker}

<!--en:2e8c089b-->
- **Warum `num_steps` Fenster an der Grenze entfallen:** Benachbarte Fenster der Länge $T$ überlappen sich in $T - 1$ Token. Das Auslassen von `windows[num_train : num_train + num_steps]` verhindert, dass eine Korpusposition aus den Trainingszielen in der Validierung vorkommt.

<!--en:4f346b10-->
::: {.slide title="Minibatches aus dem TimeMachine-Datenlader untersuchen"}

<!--en:80c2718c-->
[Tensorformen und Ausrichtung prüfen]{.kicker}

<!--en:e5e99b0c-->
Jeder Minibatch aus `data.train_dataloader()` liefert zwei ganzzahlige ID-Tensoren `(X, Y)` der Form `(batch_size, num_steps)`. Jede Zeile von `Y` entspricht der um eine Spalte nach links verschobenen Zeile von `X` mit dem nächsten Token am Ende:

<!--en:154381af-->
::: {.slide title="Verständnisfragen zu Glättung, Perplexität und Datenleckage"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:b5401de3-->
Drei Rechenfragen zur Glättung, zu Bit pro Byte und zur Aufteilung in Fenster:

<!--en:507a57ec-->
1. **Verdünnung durch Laplace-Pseudohäufigkeiten:** In *The Time Machine* kommt `"the time"` insgesamt $N(u) = 87$ Mal vor. Darauf folgt `"traveller"` in $N(u, w) = 47$ Fällen ($|\mathcal{V}| = 4{,}316$). Die mit dem Laplace-Verfahren geglättete Wahrscheinlichkeit $\hat{P}_\alpha(\textrm{traveller} \mid \textrm{the time})$ für $\alpha = 0.01$ und für $\alpha = 1.0$ berechnen. Warum fällt der Wert bei $\alpha = 1.0$ so stark ab?

<!--en:1cbdf2ab-->
2. **Von Perplexität zu Bit pro Byte:** Ein neuronales BPE-Sprachmodell erreicht auf dem Validierungstext eine Perplexität von $\textrm{PPL} = 64.0$. Jedes BPE-Token umfasst dort im Mittel $2.80$ UTF-8-Bytes. Wie viele Bit pro Byte ($\textrm{bpb}$) ergibt das? Auf welchen Prozentsatz der ursprünglichen Größe in UTF-8-Bytes könnte ein arithmetischer Kodierer den Text komprimieren?

<!--en:b4347abb-->
3. **Fortsetzungshäufigkeiten bei der Kneser-Ney-Glättung:** Das Wort `"Francisco"` hat in Nachrichtenkorpora eine hohe absolute Häufigkeit, steht aber fast ausschließlich nach `"San"`. Seine Maximum-Likelihood-Wahrscheinlichkeit im Unigrammmodell mit seiner Kneser-Ney-Fortsetzungswahrscheinlichkeit auf Unigrammebene nach einem unbekannten Bigrammkontext wie `"ate ___"` vergleichen.

<!--en:b845d8e2-->
::: {.slide title="Verständnisfragen zu Glättung, Perplexität und Datenleckage (Lösungen)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:34b62cec-->
1. **$\alpha = 0.01$:** $(47 + 0.01) / (87 + 43.16) = 47.01 / 130.16 \approx 0.361$. **$\alpha = 1.0$:** $(47 + 1) / (87 + 4316) = 48 / 4403 \approx 0.0109$. Die $4{,}316$ gleichmäßig verteilten Pseudohäufigkeiten überwiegen die $87$ tatsächlichen Beobachtungen um den Faktor $50\times$.

<!--en:35caf3b6-->
2. **$\textrm{bpb} = \log_2(64.0) / 2.80 = 6.00 / 2.80 \approx 2.14\text{ bits/byte}$**. Das entspricht einer verlustfreien Kompression auf $2.14 / 8 \approx 26.8\%$ der ursprünglichen Größe in Bytes.

<!--en:6b05e7e2-->
3. **Maximum-Likelihood-Schätzung für Unigramme und Fortsetzungshäufigkeit:** Die Unigrammschätzung weist `"Francisco"` wegen seiner Gesamthäufigkeit eine hohe Wahrscheinlichkeit zu. Die Kneser-Ney-Glättung zählt, wie viele *verschiedene Vorgänger* vor `"Francisco"` stehen ($\approx 1$, `"San"`). Daher erhält das Wort nach `"ate"` eine Wahrscheinlichkeit nahe null.

<!--en:9f7c66ab-->
::: {.slide title="Zusammenfassung zur Sprachmodellierung und ihren Gütemaßen"}

<!--en:408fcbdb-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:846a9ff4-->
| Modell / Maß | Mathematische Definition | Stärke | Grundlegende Einschränkung |

<!--en:37da18ea-->
| **Maximum-Likelihood-Schätzung für $n$-Gramme** | $\hat{P}(x_t \mid u) = \frac{N(u, x_t)}{N(u)}$ für $\|u\| = n - 1$ | Schnelles Zählen in einem Durchlauf, erfasst häufige lokale Wendungen | Für $87\%$ der zurückgehaltenen Worttrigramme gilt $N = 0 \implies \textrm{PPL} = \infty$ |

<!--en:fddeee29-->
| **Laplace-Glättung (Add-$\alpha$)** | $\hat{P}_\alpha(x_t \mid u) = \frac{N(u, x_t) + \alpha}{N(u) + \alpha \lvert\mathcal{V}\rvert}$ | Garantiert überall $\hat{P}_\alpha > 0$ | Schwächt beobachtete Häufigkeiten ab, greift bei unbekannten Kontexten auf die Gleichverteilung $1/\lvert\mathcal{V}\rvert$ zurück |

<!--en:e19b430c-->
| **Interpolierte Kneser-Ney-Glättung** | $\frac{\max(N(u,w)-D, 0)}{N(u)} + \frac{D\,S(u)}{N(u)} P_{\textrm{KN}}(w \mid u')$ | Diskontiert einmalige Vorkommen ($D \approx 0.9$), greift auf Fortsetzungshäufigkeiten zurück | Behandelt Token als diskrete orthogonale Symbole (`"cat"` $\perp$ `"feline"`) |

<!--en:fa707bc4-->
| **Perplexität ($\textrm{PPL}$)** | $\exp\bigl(-\frac{1}{T}\sum_{t=1}^T \log P(x_t \mid x_{<t})\bigr)$ | Längennormierter effektiver Verzweigungsfaktor pro Token | Skala hängt von Größe $\lvert\mathcal{V}\rvert$ und Granularität des Tokenizer-Vokabulars ab |

<!--en:2924c471-->
| **Bit pro Byte ($\textrm{bpb}$)** | $\frac{1}{B}\sum_{t=1}^T -\log_2 P(x_t \mid x_{<t}) = \frac{T}{B}\log_2 \textrm{PPL}$ | Tokenizerunabhängige Rate der verlustfreien Kompression ($\textrm{bpb}/8$) | Bei einem Tokenizer mit `<unk>` müssen die Kosten für das Ausschreiben unbekannter Wörter explizit berücksichtigt werden |
