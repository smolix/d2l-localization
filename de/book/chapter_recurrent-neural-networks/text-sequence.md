<!--en:42004d3c-->
# Vom Text zu Token

<!--en:e4bd986c-->
Die meisten Sequenzen, die wir in diesem Kapitel modellieren, sind Texte. Damit stellt sich ein Problem, das bei den synthetischen Zahlenreihen aus :numref:`sec_sequence` nicht auftrat. Neuronale Netze verarbeiten Zahlenvektoren, Text liegt zunächst als Zeichenkette vor. Bevor wir ein Sprachmodell trainieren können, müssen wir entscheiden, wie wir eine Zeichenkette in *Token* zerlegen, also in die kleinsten Einheiten, die uns als Beobachtungen $x_t$ dienen. Außerdem müssen wir jedem Token einen numerischen Index zuordnen. Diese scheinbar nebensächliche Entscheidung bei der Vorverarbeitung hat weitreichende Folgen. Sie bestimmt die Größe der Einbettungs- und Ausgabeschichten des Modells, die Sequenzlänge jedes gelesenen Dokuments und sogar die Schwierigkeit von Aufgaben wie dem Rechnen. Zerlegen wir *The Time Machine* in einzelne Zeichen, muss ein Modell rund 170 000 Zeitschritte durchlaufen. Zerlegen wir den Text in ganze Wörter, brauchen wir für jedes der rund 4600 verschiedenen Wörter eine Ausgabeklasse. Trotzdem scheitert das Modell an jedem Wort, das beim Training nicht vorkam. Deshalb bringt jedes moderne Sprachmodell einen eigenen Tokenizer mit. Wir betrachten ihn daher als festen Bestandteil des Modells.

<!--en:0cf745c7-->
Wir vergleichen zunächst die Tokenisierung auf Zeichenebene mit der auf Wortebene. Sie markieren die beiden Extreme im Zielkonflikt zwischen Vokabulargröße und Sequenzlänge. Dann gehen wir eine Ebene tiefer und verwenden die einzelnen Bytes als universelles Grundalphabet. Mit *Byte-Pair-Encoding* (BPE) lernen wir direkt aus den Daten ein Vokabular aus Subwörtern. Auf diesem Verfahren beruht die GPT-Modellfamilie :cite:`Sennrich.Haddow.Birch.2015,Radford.Wu.Child.ea.2019`. Unsere eigene Implementierung umfasst nur wenige Dutzend Zeilen Python und liefert doch Token für Token dieselben Ergebnisse wie der produktiv eingesetzte Tokenizer von GPT-2.

<!--en:ffe055fe-->
## Einlesen des Datensatzes

<!--en:90bc618b-->
Als Korpus verwenden wir in diesem Kapitel die Novelle [The Time Machine](http://www.gutenberg.org/ebooks/35) von H. G. Wells mit etwas mehr als 30 000 Wörtern. Korpora für das Pre-Training umfassen in der Praxis Billionen von Token. Doch schon unser kleiner Datensatz zeigt die statistischen Eigenschaften natürlicher Sprache und lässt sich in Sekunden tokenisieren. Die folgende Methode `_download` lädt das Buch herunter und liest den Rohtext als eine einzige Zeichenkette ein.

<!--en:66eff95e-->
Klassische Verfahren der Sprachverarbeitung normalisierten Rohtext sehr stark. Sie entfernten Satzzeichen, wandelten alle Buchstaben in Kleinbuchstaben um und zerlegten erst dann die bereinigte Zeichenkette in Token. Für den folgenden Vergleich zwischen Zeichen und Wörtern behalten wir eine einfache Normalisierung bei. Auch spätere Abschnitte greifen noch auf diese Verfahren zurück. Eine solche Vorverarbeitung ist allerdings grob. Sie verwirft wertvolle Informationen (die Großschreibung unterscheidet „Bill“ von „bill“) und legt von vornherein fest, welche Zeichen zählen. Solche Annahmen können leicht scheitern. Der Tokenizer auf Byteebene, den wir später in diesem Abschnitt entwickeln, kommt ganz ohne Normalisierung aus.

<!--en:01d7b970-->
## Zeichen, Wörter und Bytes

<!--en:34f9ac63-->
### Zeichen und Wörter

<!--en:a1fe1c54-->
Ein Sequenzmodell verarbeitet pro Zeitschritt ein Token. Als Token bieten sich daher einzelne Zeichen oder ganze Wörter an. Betrachten wir beide Möglichkeiten und beginnen wir mit den Zeichen.

<!--en:a951d158-->
Trennen wir den Text an Leerraumzeichen, erhalten wir Wörter als Token. Für beide Verfahren zählen wir die *verschiedenen* Tokentypen und alle Token im Korpus. Die erste Zahl bestimmt die Vokabulargröße, die zweite die Sequenzlänge.

<!--en:0726a1b2-->
Für die Tokenisierung unseres normalisierten Textes auf Zeichenebene genügen 27 Symbole. Ein unbekannter Buchstabe kann dabei nie auftreten. Dafür erstreckt sich die Novelle über mehr als 170 000 Zeitschritte. Weitreichende Abhängigkeiten wie die grammatische Übereinstimmung zwischen Subjekt und Verb müssen so Dutzende von Schritten überbrücken. Auf Wortebene brauchen wir für denselben Text nur etwa ein Fünftel so viele Token, und jedes trägt deutlich mehr Bedeutung. Doch schon diese kurze Novelle enthält Tausende verschiedener Worttypen, und das Vokabular wächst mit dem Korpus weiter. Jeder Eigenname, jeder Tippfehler und jede Wortform („traveller“, „travellers“, „travelled“) belegt einen eigenen Eintrag. Die meisten Einträge kommen nur ein- oder zweimal vor (:numref:`sec_language-model`). Schlimmer noch, jedes Wort, das im Trainingsdatensatz fehlt, liegt beim Testen *außerhalb des Vokabulars* (engl. out of vocabulary, OOV). :numref:`fig_granularity_spectrum` veranschaulicht den grundlegenden Zielkonflikt. Je größer das Vokabular, desto kürzer die Sequenzen. Wo auf diesem Spektrum soll unser Tokenizer liegen? Mit einem Subwort-Tokenizer können wir diesen Punkt frei wählen. Im weiteren Verlauf des Abschnitts entwickeln wir einen solchen Tokenizer.

<!--en:1e311f6c-->
![Der Zielkonflikt bei unterschiedlich feiner Tokenisierung: Je größer das Vokabular, desto weniger Token braucht es für denselben Text. Bei einzelnen Bytes deckt das Vokabular jede Eingabe ab, bei einer festen Wortliste können Wörter fehlen. Byte-Pair-Encoding vermittelt zwischen den Extremen.](../img/mdl-rnn-granularity-spectrum.svg)

<!--en:8f882c15-->
### Von Token zu Indizes

<!--en:b40f45a7-->
Bevor wir zu Subwörtern übergehen, vervollständigen wir das klassische Verfahren, das auch in späteren Abschnitten des Buches zum Einsatz kommt. Neuronale Netze verarbeiten ganzzahlige Indizes anstelle von Zeichenketten. Deshalb verwenden beide Verfahren ein Objekt für das *Vokabular*, das jedem unterschiedlichen Token einen Index zuweist, vom häufigsten bis zum seltensten Token. Unbekannte Token bildet es auf den reservierten Eintrag `<unk>` ab. Damit das Vokabular kompakt bleibt, können wir auch seltene Token entfernen, die weniger als `min_freq` Mal vorkommen. Für sie gilt dann ebenfalls `<unk>`.

<!--en:98b95586-->
Um `Vocab` zu testen, erstellen wir aus den Zeichen in `tokens` ein Vokabular. Wir bilden die ersten zehn Zeichen auf ganzzahlige Indizes (`indices`) ab und wandeln sie mit `vocab.to_tokens` wieder in Zeichen um.

<!--en:30f6b85e-->
Die Hilfsmethode `build` bündelt das gesamte klassische Verfahren in einem Aufruf. Sie übernimmt Vorverarbeitung, Tokenisierung und Indexierung und gibt das Korpus als Folge ganzzahliger Indizes zusammen mit dem Vokabular zurück.

<!--en:84fad0b7-->
### Text als Bytefolge

<!--en:bfad48b9-->
Um die Tokengröße zwischen Zeichen und Wörtern frei wählen zu können, brauchen wir zunächst ein Grundalphabet, das jede mögliche Zeichenkette darstellen kann. Zeichen selbst eignen sich dafür überraschend schlecht. Unicode definiert rund 150 000 Zeichen, darunter Zeichen verschiedener Schriftsysteme, diakritische Zeichen, mathematische Symbole und Emojis. Sobald wir über einfachen englischen Text hinausgehen, ist ein Zeichenvokabular also keineswegs mehr klein. Moderne Tokenizer setzen eine Ebene tiefer an. Eine Zeichenkodierung stellt jede auf einem Computer gespeicherte Zeichenkette letztlich als Folge von *Bytes* dar. Fast überall kommt dafür UTF-8 zum Einsatz. Ein Byte kann nur $2^8 = 256$ Werte annehmen. Ein Grundvokabular auf Byteebene enthält also nur 256 Symbole, und eine Eingabe kann *nie* außerhalb des Vokabulars liegen. Jede Zeichenkette jeder Sprache lässt sich vollständig in Bytes zerlegen, einschließlich Satzzeichen, Code und Tippfehlern.

<!--en:0550ea91-->
UTF-8 erreicht diese universelle Abdeckung durch eine Kodierung variabler Länge. Für ein gewöhnliches ASCII-Zeichen genügt ein Byte, für Zeichen seltenerer Schriftsysteme und für Symbole braucht UTF-8 bis zu vier Bytes.

<!--en:50447afd-->
Der Preis für dieses universelle Alphabet sind lange Sequenzen. Unsere Novelle umfasst fast 180 000 Bytes, also mehr als fünf Token pro Wort. Buchstaben mit Akzent wie „é“ brauchen zwei Bytes, chinesische Schriftzeichen drei und Emojis vier. Bei nicht englischen Texten fallen reine Bytefolgen daher noch länger aus. Alle folgenden Verfahren behalten die universelle Abdeckung der Bytes bei und komprimieren zugleich die Sequenzen auf eine handhabbare Länge.

<!--en:0740b647-->
## Byte-Pair-Encoding

<!--en:f151afb2-->
### Die Verschmelzungsregel

<!--en:bc4f8f56-->
Byte-Pair-Encoding (BPE) entstand als Kompressionsverfahren. Man ersetzt das häufigste Paar benachbarter Bytes durch ein neues Symbol und wiederholt diesen Schritt :cite:`Gage.1994`. :citet:`Sennrich.Haddow.Birch.2015` wandten dieselbe Regel auf die Zeichen innerhalb von Wörtern an, um Subwörter für die maschinelle Übersetzung zu gewinnen. GPT-2 kehrte zu den einzelnen Bytes zurück :cite:`Radford.Wu.Child.ea.2019`. Diese Variante auf Byteebene implementieren wir hier. Sie beginnt mit 256 Symbolen für die einzelnen Bytes und erweitert ihr Vokabular nach einem gierigen Verfahren. In jedem Schritt zählt sie jedes *Paar* benachbarter Token im Trainingskorpus und verschmilzt das häufigste Paar zu einem neuen Token. Das wiederholt sie, bis das Vokabular die gewünschte Größe erreicht. Jede Verschmelzung verkürzt das Korpus um genau so viele Token, wie sie Paare ersetzt. BPE arbeitet also als gieriger Kompressionsalgorithmus. Das gelernte Vokabular bildet ein Wörterbuch der Fragmente, die im Korpus am häufigsten wiederkehren.

<!--en:3b03658c-->
Verfolgen wir den Algorithmus von Hand, um zu sehen, wie er arbeitet. Betrachten wir ein kleines Beispielkorpus mit insgesamt acht Wörtern, darunter fünf verschiedene:

<!--en:cd91f443-->
Das ursprüngliche Korpus ist 32 Bytes lang. Das häufigste Paar benachbarter Bytes ist (`u`, `g`). Es kommt sechsmal vor (in „hug“, „pug“ und „hugs“). Die erste Verschmelzung ergibt also das neue Token `ug`. Sobald wir überall `ug` eingesetzt haben, ist (`h`, `ug`) das häufigste benachbarte Paar. So geht das Verfahren weiter:

<!--en:4f534139-->
| Schritt | häufigstes Paar | Häufigkeit | neues Token | Korpuslänge |

<!--en:1065ebcb-->
| 0 | (Ausgangspunkt: einzelne Bytes) | | | 32 |

<!--en:b22df729-->
| 2 | `h`, `ug` | 4 (Gleichstand) | `hug` | 22 |

<!--en:100f36fb-->
| 3 | `hug`, `' '` | 3 (Gleichstand) | `'hug '` | 19 |

<!--en:3617e964-->
In den Schritten 2 und 3 gibt es Gleichstände. Das Paar (`ug`, `' '`) kommt in Schritt 2 ebenfalls viermal vor. Die Paare (`' '`, `p`) und (`' '`, `hug`) finden sich ebenfalls je dreimal in Schritt

<!--en:a5f32a6e-->
3. Bei Gleichstand wählen wir wie üblich das Paar, das im Korpus zuerst vorkommt.

<!--en:3b721bca-->
An diesem Ablauf erkennen wir zwei wichtige Eigenschaften. Erstens bauen die Verschmelzungen hierarchisch aufeinander auf. So entsteht `hug` aus `h` und `ug`. Das gelernte Vokabular definiert damit einen *Verschmelzungsbaum* (engl. merge tree) über dem Bytealphabet (:numref:`fig_merge_tree`). Wenden wir die gelernten Verschmelzungen erneut an, können wir beliebigen neuen Text zerlegen. „hugs“ zerfällt zum Beispiel in `hug` und `s`, während das seltene Wort „bun“ aus drei einzelnen Bytes bestehen bleibt. Für jede Teilzeichenkette, auf die keine gelernte Verschmelzung zutrifft, bleiben die einzelnen Bytes erhalten. Zweitens hat Schritt 3 ein Wort mit seinem *nachgestellten Leerzeichen* verschmolzen. Für sich genommen stört das Leerzeichen kaum. Die nächste Verschmelzung kann aber auch den Anfang des folgenden Wortes anhängen. Ohne Einschränkungen erzeugt BPE in einem echten Korpus leicht Token aus mehreren Wörtern. Dieses Problem beheben wir in Kürze.

<!--en:45609f10-->
![Die ersten drei am Beispielkorpus gelernten Verschmelzungen bilden einen Baum über dem Bytealphabet. Für neuen Text nutzen wir denselben Baum: Aus „hugs“ werden zwei Token, unbekannte Fragmente bleiben als einzelne Bytes erhalten.](../img/mdl-rnn-merge-tree.svg)

<!--en:869d223f-->
### Implementierung

<!--en:a3e669f7-->
Wir implementieren BPE in drei kurzen Teilen. Die Klasse `BPETokenizer` speichert den Zustand des Tokenizers. Die Token-IDs von 0 bis 255 stehen für einzelne Bytes. Jede gelernte Verschmelzung erhält die nächste freie ganzzahlige ID. Damit geben die IDs zugleich die *Ränge* der Verschmelzungen an (eine kleinere ID steht für eine frühere Verschmelzung). Das Wörterbuch `merges` ordnet jedem verschmolzenen ID-Paar seine neue ID zu, während `vocab` jeder ID die zugehörige Bytefolge zuordnet. Die Argumente `pattern` und `specials` sowie `byte_ids` brauchen wir erst später in diesem Abschnitt. Außerdem importieren wir das Paket `regex`. Es erweitert das Python-Standardmodul `re` um Unicode-Zeichenklassen (etwa `\p{L}` für einen beliebigen Buchstaben), die wir für die Vortokenisierung benötigen.

<!--en:13755e1e-->
Training und Kodierung beruhen auf derselben Grundoperation. Wo immer ein bestimmtes Paar benachbarter IDs vorkommt, ersetzen wir es durch die ID des verschmolzenen Tokens. Die Hilfsfunktion `_merge` läuft von links nach rechts. Sie schreibt `new_id` und rückt zwei Stellen weiter, wenn sie `pair` findet. Andernfalls kopiert sie eine ID. In der Zeichenkette `aaa` ersetzt sie das Paar (`a`, `a`) also einmal links und vermeidet Überlappungen.

<!--en:83cbadc7-->
Das Training wiederholt dieselben Schritte wie unsere Rechnung von Hand. Wir zählen alle benachbarten Paare, verschmelzen überall das häufigste und speichern die zugehörige Regel. Bei Gleichstand gewinnt wieder das zuerst gefundene Paar. Genau so verhält sich die Python-Funktion `max`. Eine praktische Optimierung besteht darin, identische Textstücke nur einmal zu speichern. Die Häufigkeit jedes Paars gewichten wir dann damit, wie oft das Textstück im Korpus vorkommt. Ohne Vortokenisierung bildet das gesamte Korpus ein einziges Textstück mit Gewicht 1. Zerlegt die Vortokenisierung den Text in Wörter, verarbeiten wir ein häufiges Wort nur einmal statt Tausende Male. BPE-Trainingsverfahren im produktiven Einsatz gehen weiter und aktualisieren die Häufigkeiten nur in der Umgebung jeder Verschmelzung. Der Übersichtlichkeit halber zählen wir hier nach jedem Schritt neu, auch wenn das langsamer ist.

<!--en:99666d17-->
Trainieren wir `BPETokenizer` auf unserem Beispielkorpus für drei Verschmelzungen (`vocab_size=256 + 3`) und sehen uns `toy_tok.merges` an, finden wir genau die drei von Hand ermittelten Schritte wieder:

<!--en:1e377853-->
### Kodierung und Dekodierung

<!--en:42f1fc34-->
Das Training liefert eine Verschmelzungstabelle. Um neuen Text zu *kodieren*, beginnen wir mit seinen UTF-8-Bytes. Wir verschmelzen jeweils das benachbarte Paar mit dem niedrigsten Verschmelzungsrang, bis kein benachbartes Paar mehr in der Tabelle steht. Für die Kodierung zählen allein die Verschmelzungsränge. Die Häufigkeiten im Korpus spielen keine Rolle mehr. Das Ergebnis hängt also nur von der gelernten Tabelle ab: **Die Verschmelzungsränge legen den Tokenizer vollständig fest**. Die Wahl des niedrigsten Rangs bildet das Training genau nach. Denn als das Training eine Verschmelzung lernte, waren alle früheren Verschmelzungen im Text bereits angewandt. In den Übungen fassen wir diese Eigenschaft formal. Die Dekodierung ist noch einfacher. Jede Token-ID entspricht direkt einer Bytefolge. Wir hängen die Bytes aneinander und dekodieren sie als UTF-8. Mit `errors='replace'` erhalten unvollständige oder ungültige Bytefolgen aus einem generativen Modell ein Ersatzzeichen, ohne eine Ausnahme auszulösen.

<!--en:042232d7-->
Bei unserem Beispiel-Tokenizer liefert die Kodierung dasselbe Ergebnis wie die Rechnung von Hand. Die Dekodierung stellt die ursprüngliche Zeichenkette exakt wieder her.

<!--en:c8647434-->
### Training mit The Time Machine

<!--en:7bc2c991-->
Um zu sehen, was BPE aus literarischer Prosa lernt, trainieren wir es auf *The Time Machine*. Tokenlisten und generierte Textpassagen sind oft breiter als eine Druckseite. Wir definieren deshalb zunächst eine kleine Hilfsfunktion für die Ausgabe, die lange Zeilen passend umbricht. Wir verwenden sie hier und im weiteren Verlauf des Buches.

<!--en:ad3d29b1-->
Wir lernen nun ein Vokabular mit 1024 Token direkt aus dem unnormalisierten Rohtext der Novelle. Die ersten Verschmelzungen erfassen häufige Buchstabenpaare, die letzten setzen ganze Wörter aus der Prosa von Wells zusammen. Ein Token im folgenden kodierten Textbeispiel enthält allerdings zwei Wörter samt Leerzeichen: `"Time Traveller "`. Wie es dazu kommt, untersuchen wir als Nächstes.

<!--en:892969d4-->
Wie viel kürzer wird eine Sequenz, wenn wir das Vokabular vergrößern? Die Nummerierung entspricht genau der Reihenfolge, in der die Verschmelzungen gelernt wurden. Deshalb ist jedes kleinere BPE-Vokabular ein *Präfix* eines größeren, das auf demselben Korpus trainiert wurde. Behalten wir nur Verschmelzungen mit IDs strikt unter $|\mathcal{V}|$, erhalten wir exakt denselben Tokenizer wie bei einem Trainingsabbruch bei Vokabulargröße $|\mathcal{V}|$. Mit einem einzigen Trainingslauf können wir also Vokabulargrößen von 256 bis 4096 untersuchen. Im Folgenden messen wir die Kompression in Bytes pro Token an den ersten 10 000 Zeichen der Novelle (in der Übung :exref:`ex_text-sequence-pretok-ablation` untersuchen wir Text, der beim Training unberücksichtigt blieb).

<!--en:6b224def-->
Jede Verdopplung des Vokabulars verbessert die Kompression um nahezu denselben Betrag. Die Zahl der Bytes pro Token wächst also annähernd logarithmisch mit der Vokabulargröße. Das folgt direkt aus der gierigen Verschmelzungsregel. Frühe Verschmelzungen wie `"e "`, `"th"` und `"d "` ersetzen jeweils Tausende von Paaren, die 4000. Verschmelzung nur noch eine Handvoll. Diese Kurve sollten wir allerdings nicht blind auf produktiv eingesetzte Vokabulare mit 50 000 bis 200 000 Token extrapolieren. Das obere Ende der Kurve fällt zu günstig aus, weil wir auf demselben Text messen, mit dem wir den Tokenizer trainiert haben. Ein Vokabular mit 4096 Token hat sich bereits Wörter und Wendungen eingeprägt, die für *The Time Machine* charakteristisch sind. Bei allgemeiner englischer Prosa erreichen produktiv eingesetzte Vokabulare typischerweise vier bis fünf Bytes pro Token, also rund ein Token pro Wort. Das prüfen wir weiter unten nach. Die zusätzliche Kapazität dieser Vokabulare kommt selteneren Wörtern, Quellcode und bei mehrsprachigen Vokabularen wie `o200k_base` auch anderen Sprachen zugute.

<!--en:9a635a30-->
### Vortokenisierung

<!--en:dfebb894-->
Die Verbindung `"Time Traveller "` ist kein Einzelfall. Neben echten Wortfragmenten sammelt unser Tokenizer ohne weitere Einschränkungen Dutzende von Token an, die über Wortgrenzen hinweg reichen:

<!--en:d2f34409-->
Verschmelzungen mehrerer Wörter wie `"of the "` verbessern zwar die Kompression des Trainingstexts, sind aber nur begrenzt brauchbar. Sie belegen Plätze im Vokabular mit festen Wendungen und verteilen zugleich die statistische Information über deren Bestandteile auf viele Token. So steckt `"of"` nun in zahlreichen verschiedenen Token. Deshalb verwendet jeder produktiv eingesetzte BPE-Tokenizer vor dem Verschmelzen eine *Vortokenisierung* (engl. pre-tokenization). Ein regulärer Ausdruck zerlegt die Eingabe zunächst in Textstücke. Wir trainieren BPE ausschließlich *innerhalb* der einzelnen Textstücke und wenden es auch nur dort an. Verschmelzungen können daher keine Grenze zwischen Textstücken überschreiten. Das Muster von GPT-2 :cite:`Radford.Wu.Child.ea.2019` ist unter `BPETokenizer.GPT2_PATTERN` gespeichert und besteht aus fünf Zweigen:

<!--en:9c18fa5f-->
Das Muster setzt drei bewusst gewählte Konventionen um. Erstens behält jedes Wort sein *vorangestelltes Leerzeichen* (`" the"` anstelle von `"the "`). So bleibt der Abstand mit dem Wort verbunden. Zweitens bilden Buchstaben, Ziffern und Satzzeichen getrennte Textstücke, damit Zahlen und Symbole nie mit Wörtern verschmelzen. Drittens trennt das Muster häufige englische Kontraktionsendungen als eigene Textstücke ab.

<!--en:d9fa855a-->
Um die Vortokenisierung einzuschalten, übergeben wir lediglich `pattern` an `BPETokenizer`. Den gesamten Ablauf vom Rohtext bis zu den Token-IDs zeigt :numref:`fig_pretok_pipeline`. Anders als beim unbeschränkten Trainingslauf verschwinden nun alle Verbindungen über Wortgrenzen hinweg. Die frei gewordenen Plätze im Vokabular stehen für ganze Wörter mit vorangestelltem Leerzeichen zur Verfügung. Die ersten Verschmelzungen verbinden jetzt häufige Buchstaben mit vorangestellten Leerzeichen (`" t"`, `" a"`) oder fassen häufige Buchstabenpaare zusammen (`"he"`, `"in"`).

<!--en:024dd1ef-->
![Vortokenisierung am Beispiel eines Satzes. Der reguläre Ausdruck zerlegt den Text zunächst in Wörter mit vorangestelltem Leerzeichen, Ziffernfolgen und Satzzeichen. BPE verschmilzt Paare nur innerhalb der so entstandenen Textstücke und überschreitet nie deren Grenzen.](../img/mdl-rnn-pretokenization-pipeline.svg)

<!--en:0d1dd5e3-->
## Sondertoken

<!--en:613a6f50-->
Beim Training und beim Betrieb eines Modells brauchen wir außerdem einige Steuertoken. Ihre Bedeutung liegt in der *Infrastruktur*, nicht im Text selbst. Mit `<pad>` füllen wir kürzere Sequenzen auf eine gemeinsame Länge im Batch auf. `<bos>` kennzeichnet den Anfang einer Sequenz. Mit `<eos>` kann ein Modell das Ende der Generierung signalisieren (die Decoder aus :numref:`sec_decoding` verwenden es als Stoppsignal). Unser `BPETokenizer` vergibt die IDs dieser Sondertoken *oberhalb* des BPE-Bereichs. Diese feste Regel sorgt dafür, dass gewöhnlicher Text beim Kodieren nie versehentlich ein Sondertoken ergibt. Andernfalls könnte jedes Dokument oder jede Nachricht mit der wörtlichen Zeichenfolge `"<eos>"` ein Steuersignal in die Eingabe des Modells einschleusen. Das ist ein klassischer Angriffsweg für *Prompt-Injection*, bei der das System nicht vertrauenswürdigen Benutzertext als Systemanweisung interpretiert. Unsere Methode `encode` behandelt die Zeichenfolgen für Sondertoken daher als gewöhnliche Bytes, außer wenn beim Aufruf ausdrücklich `allow_special=True` gesetzt ist. Eine entsprechende Prüfung bietet `tiktoken` mit `allowed_special` (standardmäßig löst sie einen Fehler aus).

<!--en:3e924bc0-->
## Tokenizer im praktischen Einsatz

<!--en:13fb945d-->
### Die Rangtabelle eines produktiv eingesetzten Tokenizers

<!--en:e91f65e7-->
Die meisten weitverbreiteten Sprachmodelle, darunter die GPT-Familie und Llama ab Version 3, verwenden BPE-Tokenizer auf Byteebene. Sie sind genauso aufgebaut wie unserer, wurden aber auf weitaus größeren Korpora trainiert. Der Tokenizer von GPT-2 :cite:`Radford.Wu.Child.ea.2019` lässt sich kompakt speichern. Eine Nachschlagetabelle ordnet der Bytefolge jedes Tokens seinen Verschmelzungsrang zu. Hinzu kommen der reguläre Ausdruck für die Vortokenisierung und das Verzeichnis der Sondertoken. Ein neuronales Netz enthält der Tokenizer nicht. (Andere Subwortverfahren wie WordPiece oder Unigram speichern zusätzlich Regeln zur Textnormalisierung oder Tokenwahrscheinlichkeiten für die Wahl zwischen möglichen Zerlegungen.) Die Bibliothek `tiktoken` von OpenAI liefert die Rangtabelle von GPT-2 mit. Wir betrachten Einträge unmittelbar beiderseits der Grenze zwischen einzelnen Bytes und gelernten Verschmelzungen. Der führende Unterstrich in `_mergeable_ranks` kennzeichnet ein internes Attribut. Sein Name kann sich mit neuen Versionen der Bibliothek ändern.

<!--en:0c9ba6a8-->
Die Ränge 0 bis 255 entsprechen den einzelnen Bytes in permutierter Reihenfolge (GPT-2 führt die druckbaren ASCII-Zeichen zuerst auf). Ab Rang 256 folgen die gelernten Verschmelzungen in der Reihenfolge des Trainings, genau wie bei unserem `BPETokenizer`. Bemerkenswerterweise sind die ersten vier Verschmelzungen, die GPT-2 aus 40 Gigabyte Webtext gelernt hat (`" t"`, `" a"`, `"he"` und `"in"`), identisch mit den ersten vier, die `bpe` aus einer einzigen Novelle des 19. Jahrhunderts gelernt hat. Das zeigt, wie stabil die häufigsten englischen Zeichenmuster sind.

<!--en:587543e0-->
### Überprüfung unserer Implementierung

<!--en:74a96cb3-->
Da die Rangtabelle einen BPE-Tokenizer vollständig festlegt, können wir die veröffentlichten Ränge von GPT-2 direkt in unseren eigenen Encoder `BPETokenizer` laden. Jeder Rang wird zur Token-ID. Die 256 Einträge für einzelne Bytes legen die Permutation der Bytes fest. Die veröffentlichte Tabelle enthält für jedes Token nur die resultierende Bytefolge. Das Paar, aus dem das Token entstanden ist, müssen wir rekonstruieren. Dazu wenden wir auf seine Bytes alle Verschmelzungen mit niedrigerem Rang an, bis genau zwei Teile übrig sind. Die Einzelheiten der Rekonstruktion im folgenden Code können Sie überfliegen. Entscheidend ist die anschließende Zelle zur Überprüfung.

<!--en:0b549eb7-->
Für einen Testabsatz mit Prosa, Zahlen, Buchstaben mit Akzent, Währungssymbolen und einem Emoji liefert unser Encoder Token für Token dasselbe Ergebnis wie `tiktoken`. Das wird bei jedem Erstellen dieses Buches automatisch geprüft.

<!--en:46a264d8-->
Wie gut komprimieren große, produktiv eingesetzte Vokabulare die Novelle *The Time Machine*? Wir vergleichen unseren Tokenizer mit 1024 Token mit dem Vokabular von GPT-2 mit 50 257 Token und mit `o200k_base`, dem Vokabular von GPT-4o mit rund 200 000 Token.

<!--en:8fb08157-->
Obwohl unser Vokabular um den Faktor 50 bis 200 kleiner ist, erreicht es rund zwei Drittel ihrer Kompressionsrate. Denn wir haben es eigens auf dieser Novelle trainiert. Wie jedes statistische Modell ist ein Tokenizer ein Kompressionsmodell seiner Trainingsverteilung. Am besten komprimiert er Texte, die seinem Trainingskorpus ähneln.

<!--en:08b3018d-->
### Fertilität, Ziffern und Glitch-Token

<!--en:d678a6cb-->
Wie viele Token ein Tokenizer in einer Sprache im Mittel pro Wort braucht, bezeichnet man als seine *Fertilität* (engl. fertility). Der Tokenizer von GPT-2 wurde vorwiegend auf englischen Webtexten trainiert. Entsprechend stark unterscheidet sich die Tokenzahl für ein und denselben Satz in drei Sprachen:

<!--en:643966f8-->
Beim Tokenizer von GPT-2 braucht ein griechisches Wort rund sieben Token, ein englisches nur etwa eines. Jeder griechische Buchstabe benötigt zwei UTF-8-Bytes, und der Tokenizer hat nur wenige griechische Verschmelzungen gelernt. In ein Kontextfenster mit 1024 Token passen deshalb etwa 900 englische, aber nur etwa 140 griechische Wörter. Für Nutzer anderer Sprachen als Englisch wird die Inferenz dadurch langsamer und teurer. Das mehrsprachige Vokabular `o200k_base` senkt die Fertilität für Griechisch hingegen auf rund zwei Token pro Wort.

<!--en:acbdc9fa-->
An Zahlen sehen wir, wie selbst kleine Details der Tokenisierung verändern, was ein Modell lernen muss. GPT-2 zerlegt Ziffernfolgen dort, wo BPE beim Training häufige Fragmente gefunden hat. Neuere Tokenizer begrenzen Zifferngruppen schon bei der Vortokenisierung auf drei Zeichen, damit die Zerlegung gleichmäßiger ausfällt:

<!--en:9877d26f-->
Trotzdem entspricht keine der beiden Zerlegungen den Spalten beim schriftlichen Rechnen. Dort ordnet man die Ziffern von rechts nach links den Einern, Zehnern und Hundertern zu. Bilden wir Gruppen von höchstens drei Ziffern von links nach rechts, zerfällt `"12345678"` in `"123"`, `"456"` und `"78"`. Das Sprachmodell muss Überträge dann über Grenzen hinweg verarbeiten, die nicht mit den Stellenwerten übereinstimmen. Diese Diskrepanz trägt bekanntlich zu Rechenfehlern bei Sprachmodellen bei. Sie erklärt mit, warum übliche Tausendertrennzeichen (`"12,345,678"`) das Rechnen erleichtern können.

<!--en:f40d0a75-->
Wie zuverlässig ein Modell mit einem Vokabulareintrag umgeht, hängt schließlich davon ab, wie oft es ihm im Training begegnet ist. Frühe Vokabulare von GPT-2 und GPT-3 enthielten Token wie `" SolidGoldMagikarp"`. Dieser Reddit-Benutzername kam im Trainingskorpus des *Tokenizers* so häufig vor, dass er eine eigene Verschmelzung erhielt. In den Daten für das Pre-Training des *Sprachmodells* fehlte er dagegen fast völlig. Ein solches *Glitch-Token* kann bei der Kodierung eines Prompts entstehen, obwohl sein Einbettungsvektor durch Gradienten kaum aktualisiert wurde. Auf die Aufforderung, es zu wiederholen, antworteten frühe Modelle oft ausweichend oder mit Wörtern ohne Bezug dazu.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:59f62e29-->
Ein Tokenizer übersetzt Rohtext in die Folgen ganzer Zahlen, die neuronale Netze verarbeiten. Seine Vokabulargröße bestimmt den Zielkonflikt zwischen der Sequenzlänge und der Größe der Ausgabeschicht. Feste Wortvokabulare scheitern außerdem an unbekannten Wörtern. BPE auf Byteebene beginnt mit den 256 Symbolen für einzelne Bytes. Damit liegt keine Eingabe außerhalb des Vokabulars. BPE verschmilzt jeweils das häufigste benachbarte Tokenpaar, bis das Vokabular die gewünschte Größe erreicht. Bei der Kodierung gelten die gelernten Verschmelzungen streng in ihrer Rangfolge. Daher definiert die Verschmelzungstabelle allein den Tokenizer. In der Praxis verhindert ein regulärer Ausdruck bei der Vortokenisierung, dass Verschmelzungen Wortgrenzen überschreiten. Steuertoken erhalten IDs oberhalb des BPE-Bereichs, damit sie nicht aus gewöhnlichem Text entstehen können. Der produktiv eingesetzte Tokenizer von GPT-2 folgt genau diesem Aufbau in größerem Maßstab. Unsere Implementierung liefert Token für Token dieselben Ergebnisse. Die je nach Sprache unterschiedliche Fertilität, die Besonderheiten bei der Zerlegung von Zahlen und die Glitch-Token ergeben sich direkt aus den Trainingsdaten und den Verschmelzungsregeln. Im nächsten Abschnitt modellieren wir die Tokensequenzen, die ein Tokenizer erzeugt.

<!--en:2294243a-->
## Übungen

<!--en:055c6be0-->
1. **Kodierung nach Rang.** Ein `BPETokenizer` mit `vocab_size=258` habe genau zwei Verschmelzungen gelernt: zuerst (`b`, `c`) mit ID 256, dann (`a`, `b`) mit ID 257. Die *Rangregel* von `_encode_chunk` wählt stets die anwendbare Verschmelzung mit dem niedrigsten Rang. Die Regel *linkes Paar zuerst* verschmilzt stets das am weitesten links stehende benachbarte Paar, das in der Verschmelzungstabelle vorkommt.
    1. Zerlegen Sie die Zeichenkette „abc“ nach jeder der beiden Regeln.
    1. Geben Sie an, wie jedes Vorkommen von „abc“ im Trainingskorpus am Ende des Trainings zerlegt war. Welche Regel stimmt damit überein?

<!--en:01946aa2-->
1. [code] **Sondertoken kodieren und wieder dekodieren.** In :numref:`subsec_special-tokens` liegen die Sondertoken oberhalb des BPE-ID-Bereichs, damit sie nie aus gewöhnlichem Text entstehen können. Erstellen Sie einen `BPETokenizer`, dessen `specials` die drei Standardtoken um ein Dokumenttrennzeichen `<sep>` ergänzen. Trainieren Sie ihn auf `raw_text`. Die Zeichenkette `x` bestehe aus zwei kurzen Dokumenten, die durch `<sep>` verbunden sind.
    1. Kodieren Sie `x` einmal mit und einmal ohne `allow_special=True`. Prüfen Sie, ob beide ID-Sequenzen beim Dekodieren wieder `x` ergeben.
    1. Kodieren Sie den dekodierten Text erneut, einmal mit `allow_special=True` und einmal ohne diese Option. Welche der beiden ID-Sequenzen aus der vorherigen Teilaufgabe erhalten Sie jeweils? Erläutern Sie, was das für ein Verfahren bedeutet, das die Ausgabe eines Modells dekodiert und den Text später wieder als Eingabe verwendet.
    1. `tiktoken` löst einen Fehler aus, wenn die Zeichenfolge eines Sondertokens in einem Text erscheint, der sie nicht enthalten darf. Dagegen behandelt `encode` die Zeichenfolge als gewöhnliche Bytes. Nennen Sie für jedes der beiden Verfahren eine Situation, in der es vorzuziehen ist.

    *Nach Andrej Karpathy, [minbpe](https://github.com/karpathy/minbpe/blob/master/exercise.md), Schritt 4.*

<!--en:201710d9-->
1. [code] **Gruppierung von Ziffern.** Tokenisieren Sie die Additionsprompts „1234+5678=“, „1234+567=“ und „567+678=“ sowie ihre Summen „6912“, „1801“ und „1245“ mit den Kodierungen `gpt2` und `o200k_base` aus `tiktoken`. Zeichnen Sie für jede Zahl die Tokengrenzen ein.
    1. Geben Sie für jedes Zifferntoken an, welche Stellenwerte (Einer, Zehner, Hunderter, Tausender) es bei der Begrenzung auf drei Ziffern in `o200k_base` umfasst. Decken die Token an entsprechenden Positionen der beiden Summanden und der Summe dieselben Stellenwerte ab? Beantworten Sie das für jeden Prompt. Erklären Sie, was bei einem Modell schiefgeht, das Token für Token addiert, wenn die Stellenwerte nicht übereinstimmen.
    1. Schlagen Sie eine Regel für die Vortokenisierung vor, die Token an den Stellenwerten ausrichtet, und nennen Sie einen Nachteil Ihrer Regel.

<!--en:3f0c9434-->
1. **Glitch-Token.** Nach :numref:`subsec_fertility-digits-glitch` entsteht ein Glitch-Token aus einer Zeichenkette, die im Trainingskorpus des Tokenizers häufig, im Trainingskorpus des Sprachmodells aber selten vorkam.
    1. Warum macht die Verschmelzungsregel von `BPETokenizer.train` aus einer solchen Zeichenkette ein einziges Token, statt sie in mehreren Fragmenten zu belassen? Welche Rolle spielt die Vortokenisierung dabei, wo das Token beginnt und endet?
    1. Entwerfen Sie einen Test, der anhand der trainierten Einbettungstabelle und des Trainingskorpus des Modells Glitch-Token kennzeichnet. Geben Sie an, was jede der beiden Eingaben dazu beiträgt.

<!--en:8a0696c5-->
1. [code] **Ablation der Vortokenisierung.** ● Teilen Sie `raw_text` in die ersten 90% und die letzten 10% auf. Trainieren Sie auf dem ersten Teil zwei Tokenizer mit je 4096 Token, die sich nur im Muster für die Vortokenisierung unterscheiden. Trainieren Sie einen wie `big` ohne Muster und den anderen mit `BPETokenizer.GPT2_PATTERN`.
    1. Listen Sie für jeden Tokenizer die 20 längsten gelernten Token auf und markieren Sie diejenigen, die mehr als ein Wort umfassen.
    1. Messen Sie, welcher der beiden Tokenizer den ersten Teil besser komprimiert. Erklären Sie das Ergebnis anhand der Verschmelzungsregel.
    1. Messen Sie für beide Tokenizer die Bytes pro Token auf den letzten 10%. Welcher Tokenizer büßt auf diesem zurückgehaltenen Text den größeren Anteil seiner Kompressionsrate ein, und welcher komprimiert ihn besser? Begründen Sie beide Antworten anhand der aufgelisteten Token.
   :exlabel:`ex_text-sequence-pretok-ablation`

<!--en:a79a5e6c-->
[Diskussion](https://d2l.discourse.group/t/115)

<!--en:72fc827f-->
[Diskussion](https://d2l.discourse.group/t/3857)

<!--en:2e3bf10e-->
[Diskussion](https://d2l.discourse.group/t/3858)

<!--en:68c9fff8-->
[Diskussion](https://d2l.discourse.group/t/18011)

<!--en:b880710c-->
Vom Text zu Token<br>**Zeichen und Wörter · UTF-8-Bytes · Byte-Pair-Encoding (BPE) · Verschmelzungsbäume · Vortokenisierung mit regulären Ausdrücken · Sondertoken · Tokenizer im praktischen Einsatz**

<!--en:c5df1d38-->
::: {.slide title="Warum die Tokenisierung zum Modell gehört"}

<!--en:687ed302-->
[Motivation und Grundidee]{.kicker}

<!--en:6c0bcf33-->
- **Parameter für Einbettung und Ausgabe:** Jedes Token in $\mathcal{V}$ belegt eine Zeile in $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ und eine Spalte in $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times |\mathcal{V}|}$ ($89\%$ der Parameter unseres RNN aus §{sec_rnn}).

<!--en:765544b8-->
- **Effektiver Kontexthorizont ($T$):** Zerlegt man *The Time Machine* in Zeichen, ergeben sich $173{,}428$ Zeitschritte. Die Übereinstimmung zwischen Subjekt und Verb muss Dutzende rekurrenter Aktualisierungen überbrücken.

<!--en:4625272f-->
- **Robustheit bei offenem Vokabular:** Die Zerlegung in $32{,}775$ ganze Wörter ($4{,}579$ Typen) scheitert beim Testen an jedem unbekannten Wort (*außerhalb des Vokabulars*, OOV). BPE mit Subwörtern vermittelt zwischen den Extremen und vermeidet OOV-Fälle vollständig.

<!--en:43257b51-->
::: {.slide title="The Time Machine einlesen und normalisieren"}

<!--en:ddf983e8-->
[Verarbeitung des Korpus]{.kicker}

<!--en:f0ecf98e-->
Als Korpus dient uns *The Time Machine* von H. G. Wells (rund 30 000 Wörter, 178 979 UTF-8-Bytes). Klassische Verfahren auf Zeichen- oder Wortebene normalisieren den Text stark. Unser BPE-Tokenizer auf Byteebene verarbeitet `raw_text` verlustfrei:

<!--en:87e5f770-->
::: {.slide title="The Time Machine einlesen und normalisieren (Fortsetzung)"}

<!--en:ddf983e8-2-->
[Verarbeitung des Korpus]{.kicker}

<!--en:25194be8-->
**Was bei starker Normalisierung verloren geht:** Wer alle Zeichen außer ASCII-Buchstaben entfernt und alles kleinschreibt (`re.sub('[^A-Za-z]+', ' ', text).lower()`), verliert Satzzeichen, Zahlen, Codesyntax und die Unterscheidung von Eigennamen (*„Bill“* und *„bill“*).

<!--en:be62a7e9-->
::: {.slide title="Der Zielkonflikt zwischen Zeichen- und Wortebene"}

<!--en:b4667169-->
[Zwei Enden des Spektrums]{.kicker}

<!--en:07ad7c79-->
Die Tokenisierung auf Zeichen- und auf Wortebene markiert die beiden Extreme des Zielkonflikts zwischen Vokabulargröße $|\mathcal{V}|$ und Sequenzlänge $T$ des Korpus:

<!--en:4a151020-->
- **Zeichen ($|\mathcal{V}| = 27$, $T = 173{,}428$):** Sehr kleine Ausgabeschicht und keine unbekannten ASCII-Buchstaben, dafür um den Faktor $5.3\times$ längere Sequenzen, deren einzelne Token kaum Bedeutung tragen.

<!--en:2ee31384-->
- **Wörter ($|\mathcal{V}| = 4{,}579$, $T = 32{,}775$):** Kurze Sequenzen mit bedeutungsreichen Token. Wortformen (*„traveller“*, *„travellers“*, *„travelled“*) verteilen die Häufigkeiten auf mehrere Einträge. Unbekannte Wörter liegen beim Testen außerhalb des Vokabulars (OOV).

<!--en:9435dd87-->
::: {.slide title="Klassische Indexierung mit Vocab und `<unk>` als Ersatztoken"}

<!--en:dd7219fe-->
[Klassische Vokabulare aus Wörtern und Zeichen]{.kicker}

<!--en:3bf46470-->
- **Zusammenfassung unter `<unk>`:** Jedes seltene (`< min_freq`) oder unbekannte Wort erhält den Index `0 (<unk>)`. Dadurch gehen Wortstämme verloren, und bei der Auswertung fallen explizit Kosten für die buchstabenweise Darstellung an (§{sec_language-model}).

<!--en:004dc7d1-->
::: {.slide title="UTF-8-Bytes als universelles Grundalphabet mit 256 Symbolen"}

<!--en:9629dbf7-->
[Text als Bytefolge]{.kicker}

<!--en:60b487f9-->
- **Garantiert keine OOV-Fälle:** Ein Byte kann $2^8 = 256$ Werte annehmen. Jede Zeichenkette in jeder Schrift, auch Code oder Emojis, lässt sich daher verlustfrei in IDs im Bereich `0..255` zerlegen.

<!--en:6bfa9556-->
- **Variable Bytezahl:** UTF-8 benötigt 1 Byte für ASCII-Zeichen, 2 für lateinische Buchstaben mit diakritischen Zeichen oder griechische Buchstaben, 3 für CJK-Schriftzeichen und 4 für Emojis ($178{,}979$ Bytes für die gesamte Novelle).

<!--en:49380f49-->
::: {.slide title="Byte-Pair-Encoding (BPE) durch gieriges Verschmelzen von Paaren"}

<!--en:8c1c33a5-->
[Der BPE-Algorithmus]{.kicker}

<!--en:015fa25a-->
- **Schrittweise gierige Kompression (Sennrich et al., 2016):** Wir beginnen mit den 256 Token für einzelne Bytes (`0..255`). In jedem Schritt zählen wir alle benachbarten Tokenpaare und verschmelzen das häufigste Paar zu einem Token mit neuer ID (`256, 257, ...`), bis das Vokabular die Größe $|\mathcal{V}|$ erreicht.

<!--en:bb1a1b0b-->
- **Kürzeres Korpus:** Ersetzen wir ein Paar mit der Häufigkeit $c$ überall, verkürzt sich die Trainingssequenz um $c$ Token (abzüglich Selbstüberlappungen in Folgen wie `'aaa'`).

<!--en:6c684c8a-->
- **Token-IDs als Verschmelzungsränge:** Da wir die IDs fortlaufend vergeben (`256, 257, ...`), hält die ganzzahlige ID eines Tokens genau fest, an welcher Stelle seine Verschmelzung gelernt wurde.

<!--en:9ec3102d-->
::: {.slide title="Hierarchische Verschmelzungsbäume und Rückgriff auf Bytes"}

<!--en:82d8ec4e-->
[Generalisierung mit Subwörtern]{.kicker}

<!--en:29e155ff-->
Spätere Verschmelzungen verbinden Token aus früheren Verschmelzungen (`'h'` + `'ug'` $\to$ `'hug'`). Ein trainiertes BPE-Vokabular definiert damit binäre **Verschmelzungsbäume**, die auf den 256 einzelnen Bytes aufbauen:

<!--en:0682a310-->
- **Morphologischer Aufbau:** Auch bei neuen oder flektierten Wörtern greifen die gelernten Regeln. Für `"hugs"` ergeben Verschmelzung 1 (`u`+`g` $\to$ `ug`) und Verschmelzung 2 (`h`+`ug` $\to$ `hug`) die Zerlegung in Stamm und Suffix `[b'hug', b's']`.

<!--en:121714ff-->
- **Rückgriff auf Bytes:** Auf das seltene Wort `"bun"` passt keine gelernte Verschmelzung. Es bleibt als Folge dreier einzelner Bytetoken `[b'b', b'u', b'n']` erhalten und fällt nicht auf `<unk>` zurück.

<!--en:0ad0d1d2-->
::: {.slide title="Implementierung der paarweisen Verschmelzung von links nach rechts"}

<!--en:c8bf2621-->
[Eigene Implementierung]{.kicker}

<!--en:c737813e-->
Training und Kodierung mit BPE verwenden dieselbe Grundoperation `_merge(ids, pair, new_id)`:

<!--en:42fa0a88-->
- **Von links nach rechts ohne Überlappung:** Gilt `(ids[i], ids[i + 1]) == pair`, hängen wir `new_id` an und rücken mit `i += 2` weiter. In `[a, a, a]` ergibt die Verschmelzung von `(a, a)` somit `[aa, a]`. Die Paare überlappen sich nicht.

<!--en:1b4a0f4f-->
- **Zustand in `BPETokenizer`:** `self.vocab` ordnet jeder ID im Bereich `0..vocab_size-1` die ursprüngliche Folge vom Typ `bytes` zu (`vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]`). Das Wörterbuch `self.merges` speichert die Zuordnung `(id_a, id_b) -> new_id`.

<!--en:b840324c-->
::: {.slide title="BPE-Training mit nach Häufigkeit gewichteten Textstücken"}

<!--en:c8bf2621-2-->
[Eigene Implementierung]{.kicker}

<!--en:fb8b38f2-->
Um `vocab_size - 256` Verschmelzungen zu lernen, zählen wir die verschiedenen Textstücke nach der Vortokenisierung einmal mit `collections.Counter`. Jedes benachbarte Paar in einem Textstück gewichten wir mit der Häufigkeit `w` dieses Textstücks im Korpus:

<!--en:f9185cbd-->
**Warum identische Textstücke nur einmal speichern?** Nach der Vortokenisierung mit einem regulären Ausdruck kommt ein häufiges Wort wie `" the"` Tausende Male als eigenes Textstück vor. Gewichten wir die Paarhäufigkeiten mit der Häufigkeit `w` des Textstücks, verarbeiten wir `" the"` pro Iteration nur einmal statt Tausende Male.

<!--en:be5e8c60-->
::: {.slide title="Neuen Text in der Rangfolge der Verschmelzungen kodieren"}

<!--en:c7da86fb-->
[Kodierung und Dekodierung]{.kicker}

<!--en:6921f43a-->
::: {.slide title="Neuen Text in der Rangfolge der Verschmelzungen kodieren (Fortsetzung)"}

<!--en:c7da86fb-2-->
[Kodierung und Dekodierung]{.kicker}

<!--en:fa0e51df-->
Für ein beliebiges Stück der Bytefolge verschmilzt `_encode_chunk` jeweils das benachbarte Paar mit dem **niedrigsten Verschmelzungsrang** (kleinste `new_id`), bis kein benachbartes Paar mehr in `self.merges` steht:

<!--en:c7394c96-->
- **Die Verschmelzungstabelle bestimmt den Tokenizer:** Die Kodierung verwendet ausschließlich Verschmelzungsränge. Häufigkeiten im Korpus braucht sie nicht.

<!--en:28a64250-->
- **Warum die Rangfolge zwingend ist:** Als das Training Verschmelzung $r$ lernte, waren alle Verschmelzungen mit Rang $< r$ in den Sequenzen *bereits* ausgeschöpft. Das Paar mit dem niedrigsten Rang zuerst zu verschmelzen bildet das Training exakt nach.

<!--en:a658ca52-->
::: {.slide title="Was unbeschränktes BPE aus der Novelle von Wells lernt"}

<!--en:27c2d927-->
[Beobachtungen an literarischer Prosa]{.kicker}

<!--en:e3ed9e4d-->
Ein BPE-Vokabular mit 1024 Token ($1024 - 256 = 768$ Verschmelzungen), direkt auf dem Rohtext von *The Time Machine* trainiert, zeigt eine klare Entwicklung von häufigen Bigrammen zu textspezifischen Wörtern:

<!--en:215f957a-->
- **Erste Verschmelzungen (`256..263`):** Häufige englische Buchstabenpaare und Wortenden mit nachgestelltem Leerzeichen (`b'e '`, `b'th'`, `b't '`, `b'in'`, `b'd '`, `b'er'`).

<!--en:03b4d7c7-->
- **Letzte Verschmelzungen (`1016..1023`):** Wortstämme und ganze Wörter mit nachgestelltem Leerzeichen (`b'ering '`, `b'door'`, `b'after '`).

<!--en:8fb6d105-->
- **Das Problem:** Das Token `[b'Time Traveller ']` vereint zwei eigenständige Wörter und ein nachgestelltes Leerzeichen in einem einzigen Vokabulareintrag.

<!--en:eebc07b5-->
::: {.slide title="Logarithmischer Zusammenhang zwischen Kompression und Vokabulargröße"}

<!--en:9401b14d-->
[Kompression und Vokabulargröße]{.kicker}

<!--en:e577d933-->
BPE vergibt die IDs fortlaufend in der Reihenfolge des Trainings. Behalten wir aus einem Lauf mit 4096 Token nur die Verschmelzungen mit ID $< |\mathcal{V}|$, erhalten wir exakt den Tokenizer der Größe $|\mathcal{V}|$:

<!--en:b73e8d96-->
- **Kompressionsgewinn bei Verdopplung:** Tragen wir die Zahl der UTF-8-Bytes pro Token gegen $\log |\mathcal{V}|$ für den Bereich $256 \to 4{,}096$ auf, ergibt sich fast eine Gerade. Jede Verdopplung von $|\mathcal{V}|$ bringt etwa $0.7$–$1$ Byte mehr pro Token.

<!--en:afddaa28-->
- **Sinkende Häufigkeiten:** Verschmelzung 256 fasst Tausende Vorkommen von `b'e '` zusammen, Verschmelzung 4095 nur noch eine Handvoll.

<!--en:136b7699-->
- **Überanpassung an den Trainingstext:** Die Auswertung auf dem Trainingstext begünstigt große Werte von $|\mathcal{V}|$, weil sich das Vokabular mit 4096 Token für die Novelle typische Wendungen einprägt.

<!--en:15b5ac8f-->
::: {.slide title="Warum unbeschränktes BPE Wortgrenzen überschreitet"}

<!--en:46fc583e-->
[Warum wir vortokenisieren]{.kicker}

<!--en:17deb797-->
Ohne Beschränkung auf Wortgrenzen reichen 78 der 768 gelernten Verschmelzungen in unserem Vokabular mit 1024 Token über ein Leerzeichen zwischen Wörtern hinweg:

<!--en:d8434300-->
- **Warum Token aus mehreren Wörtern die Generalisierung erschweren:** Die Einträge für `'of the '`, `'in the '` und `'to the '` stehen nicht für Inhaltswörter zur Verfügung. Je nach Folgewort verteilt sich `"of"` auf Dutzende verschiedener Token-IDs.

<!--en:b76b3b77-->
::: {.slide title="Warum unbeschränktes BPE Wortgrenzen überschreitet (Fortsetzung)"}

<!--en:46fc583e-2-->
[Warum wir vortokenisieren]{.kicker}

<!--en:ab94b41d-->
- **Der reguläre Ausdruck von GPT-2 mit 5 Zweigen (`GPT2_PATTERN`):** Er trennt Kontraktionsendungen ab (`'s`, `'ve`, `'ll`), bindet jedes **vorangestellte Leerzeichen** an die folgende Buchstabenfolge (` ?\p{L}+`) oder Ziffernfolge (` ?\p{N}+`) und isoliert Satzzeichen (` ?[^\s\p{L}\p{N}]+`). BPE darf nur *innerhalb* jedes Textstücks verschmelzen.

<!--en:d86510d0-->
::: {.slide title="Vortokenisierung mit regulären Ausdrücken bei produktiv eingesetztem BPE"}

<!--en:ad007263-->
[Ergebnisse mit Vortokenisierung]{.kicker}

<!--en:6abd8e1e-->
Übergeben wir `pattern=BPETokenizer.GPT2_PATTERN`, verschwinden alle Verbindungen über Wortgrenzen hinweg. Die frei gewordenen Vokabulareinträge stehen für ganze Wörter mit vorangestelltem Leerzeichen zur Verfügung:

<!--en:d6fbb701-->
- **Keine Token über Wortgrenzen hinweg:** Die Zahl der grenzüberschreitenden Token sinkt (`len(crossers)`: $78 \to 0$). An die Stelle dieser Token treten ganze Wörter wie `' Traveller'`, `' Machine'` und `' Morlocks'`.

<!--en:333fda5b-->
- **Vorangestelltes Leerzeichen:** Die ersten Verschmelzungen ergeben `b' t'`, `b' a'`, `b'he'`, `b'in'`. Der Wortanfang enthält damit das vorausgehende Leerzeichen. Beim Dekodieren braucht man keine Heuristik, um Leerzeichen einzufügen.

<!--en:75138c95-->
::: {.slide title="Steuertoken und Schutz vor Prompt-Injection"}

<!--en:9df4157a-->
[Infrastruktur und Sicherheitsgarantien]{.kicker}

<!--en:8d2eebaa-->
- **Eigener ID-Bereich oberhalb der BPE-Verschmelzungen:** Sondertoken (`<pad>`, `<bos>`, `<eos>`) erhalten IDs `1024..1026`. Gewöhnliche Byteverschmelzungen können daher nie eine Steuer-ID erzeugen.

<!--en:43015c2e-->
- **Explizite Freigabe mit `allow_special`:** Standardmäßig (`allow_special=False`) wird `"<bos>"` als Folge von 5 gewöhnlichen ASCII-Bytes `[60, 98, 111, 115, 62]` kodiert. Dadurch lassen sich Steuertoken nicht über den Prompt einschleusen.

<!--en:8f43cf9f-->
::: {.slide title="Die Rangtabelle des GPT-2-Tokenizers in tiktoken"}

<!--en:d90a67d3-->
[Abgleich mit dem produktiv eingesetzten Tokenizer]{.kicker}

<!--en:498776e8-->
Produktiv eingesetzte BPE-Tokenizer auf Byteebene (GPT-2/3/4, Llama 3) speichern keine neuronalen Gewichte. Es genügen ein Wörterbuch mit der Zuordnung Bytefolge $\to$ Rang, der reguläre Ausdruck für die Vortokenisierung und die Zuordnung der Sondertoken:

<!--en:f9b081ab-->
- **Stabile Häufigkeiten im Englischen:** Die ersten vier Verschmelzungen von GPT-2 aus 40 GB WebText (`b' t'`, `b' a'`, `b'he'`, `b'in'` mit den Rängen `256..259`) sind **identisch** mit den ersten vier, die unser Tokenizer aus einer einzigen Novelle von 1895 gelernt hat.

<!--en:c10788fa-->
- **Exakte Übereinstimmung:** Laden wir die 50 257 Vokabulareinträge von GPT-2 mit `BPETokenizer.from_tiktoken('gpt2')`, erhalten wir für Prosa, Dezimalzahlen, diakritische Zeichen, Währungssymbole und Emojis Token für Token dasselbe Ergebnis wie `tiktoken`.

<!--en:4de4ee11-->
::: {.slide title="Kompressionsraten von spezialisierten und allgemeinen Tokenizern"}

<!--en:1cd90361-->
[Fertilität im Vergleich zwischen Tokenizern und Sprachen]{.kicker}

<!--en:5769150b-->
Wir vergleichen unseren Novellen-Tokenizer mit 1024 Token mit den Vokabularen von OpenAI, `gpt2` mit rund 50 000 und `o200k_base` (GPT-4o) mit rund 200 000 Token. So sehen wir, wie gut die Kompression im eigenen Anwendungsgebiet ausfällt und wie sich die **Fertilität** (Token pro Wort) je nach Sprache unterscheidet:

<!--en:c1a4e08a-->
- **Domänenanpassung:** Unser spezialisiertes BPE mit rund 1000 Token erreicht $2.72$ Bytes pro Token. Das sind rund zwei Drittel der Kompressionsrate der um die Faktoren $50\times$ und $200\times$ größeren Vokabulare ($4.06$ bzw. $4.20$ Bytes pro Token).

<!--en:69eb5960-->
- **Preis hoher Fertilität:** Mit dem auf Englisch ausgerichteten Tokenizer `gpt2` benötigt Griechisch **$7.33$ Token pro Wort**, Englisch nur $1.14$. In ein festes Kontextfenster passt damit um den Faktor $6.4\times$ weniger Text, und die Inferenzkosten steigen um den Faktor $6.4\times$. Das mehrsprachige Vokabular `o200k_base` senkt die griechische Fertilität auf $1.89$.

<!--en:f030f707-->
::: {.slide title="Stolpersteine bei Stellenwerten und Glitch-Token"}

<!--en:d4bc3e62-->
[Probleme im praktischen Einsatz]{.kicker}

<!--en:474a4782-->
Zwei bekannte Eigenheiten moderner großer Sprachmodelle lassen sich direkt auf Entscheidungen bei der Subwort-Tokenisierung zurückführen:

<!--en:dfb80c58-->
- **Rechnen mit Zifferngruppen von links nach rechts:** `o200k_base` teilt Ziffernfolgen von links in Gruppen von höchstens 3 Zeichen und zerlegt `"12345678"` so in `['123', '456', '78']`. Das passt nicht zur Gruppierung nach Stellenwert von rechts (`12,345,678`). Überträge liegen daher je nach Zahl an anderen Stellen relativ zu den Tokengrenzen.

<!--en:373db615-->
- **Glitch-Token (`' SolidGoldMagikarp'`):** Eine Zeichenkette, etwa ein häufig auftretender Reddit-Benutzername, kann in den gesammelten Trainingsdaten des *Tokenizers* oft vorkommen, aber aus dem Korpus für das Pre-Training des *Sprachmodells* herausgefiltert werden. Sie erhält dann eine eigene Token-ID (`43453`), deren Zeile $\mathbf{E}_{43453}$ in der Einbettungstabelle beim Testen noch die zufälligen Anfangswerte enthält.

<!--en:8e30207a-->
::: {.slide title="Verständnisfragen zu BPE-Rängen, regulären Ausdrücken und Fertilität"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:1d1aa995-->
Wie funktionieren die BPE-Kodierung und die Vortokenisierung in der Praxis?

<!--en:1b186310-->
1. **Rangfolge oder Position:** Ein BPE-Tokenizer lernt zuerst `('b', 'c') -> 256` (Rang 256), dann `('a', 'b') -> 257` (Rang 257). Wie zerfällt `"abc"`, wenn (a) wie vorgesehen der niedrigste Rang entscheidet oder (b) immer das am weitesten links stehende anwendbare Paar gewählt wird? Welche Zerlegung entspricht dem Trainingsende?

<!--en:707f2b15-->
2. **Vortokenisierung in GPT-2:** In welche Textstücke genau zerlegt `BPETokenizer.GPT2_PATTERN` den Satz `"I've got 1234 cats!"` vor den BPE-Verschmelzungen?

<!--en:f1ba4425-->
3. **Kapazität des Kontextfensters:** Ein großes Sprachmodell hat ein Kontextfenster für 1024 Token und verwendet den Tokenizer `gpt2` (Fertilität für Englisch $1.14$ Token pro Wort, für Griechisch $7.33$ Token pro Wort). Wie viele englische bzw. griechische Wörter passen ungefähr in ein Kontextfenster?

<!--en:557b98b6-->
**Antworten:**

<!--en:a9d9cabd-->
1. **(a) Niedrigster Rang zuerst:** Zuerst verschmilzt `('b', 'c')` (Rang 256 < 257). Das ergibt `['a', 'bc']` (`[97, 256]`). So war es auch beim Training: Jedes `'bc'` war bereits verschmolzen, bevor `('a', 'b')` erstmals gezählt wurde. **(b) Linkes Paar zuerst:** Diese Regel verschmilzt fälschlich zuerst `('a', 'b')` und liefert `['ab', 'c']`.

<!--en:0ae1eea9-->
2. **Sechs Textstücke:** `['I', "'ve", ' got', ' 1234', ' cats', '!']` (Kontraktionsendung abgetrennt, vorangestelltes Leerzeichen an Wort bzw. Ziffernfolge gebunden, abschließendes Satzzeichen isoliert).

<!--en:92e74664-->
3. **Rund 898 englische oder 140 griechische Wörter:** $1024 / 1.14 \approx 898$ Wörter im Englischen gegenüber $1024 / 7.33 \approx 140$ Wörtern im Griechischen. Die effektive Kontextlänge sinkt um den Faktor $6.4\times$.

<!--en:af43e260-->
::: {.slide title="Tokenisierungsverfahren im Vergleich"}

<!--en:408fcbdb-->
[Ergebnisse des Abschnitts]{.kicker}

<!--en:6eb2bee8-->
| Tokenisierung | Grundalphabet | Vokabulargröße $|\mathcal{V}|$ (*Time Machine*) | Sequenzlänge $T$ | Verhalten außerhalb des Vokabulars (OOV) | Zielkonflikt beim Modellentwurf |

<!--en:01233e9d-->
| **Zeichenebene** | Normalisierte ASCII-Zeichen / Unicode-Zeichen | $27$ (normalisiert) bis $\sim 150\textrm{k}$ (Unicode) | $173{,}428$ | Keine OOV-Fälle bei ASCII-Buchstaben, unbekannte Unicode-Schriften fehlen | Sehr kleine Schicht für die Softmax-Funktion, aber um den Faktor $5\times$ längere Sequenzen mit wenig Bedeutung pro Token |

<!--en:cc6d03f5-->
| **Wortebene** | An Leerraumzeichen getrennte Worttypen | $4{,}579 + \texttt{<unk>}$ (wächst mit dem Korpus unbegrenzt) | $32{,}775$ | Alle unbekannten Wörter werden zu `<unk>` | Kurze Sequenzen, aber sehr große Schicht für die Softmax-Funktion, viele unbekannte Wörter und spärliche Daten für einzelne Wortformen |

<!--en:791b42e3-->
| **Einzelne UTF-8-Bytes** | Einzelbytes `0..255` | $256$ (fest, universell) | $178{,}979$ | **Keine OOV-Fälle** für beliebige Zeichenketten in jeder Sprache | Universelle Abdeckung, aber um den Faktor $5.5\times$ längere Sequenzen (für Nicht-ASCII-Zeichen nochmals um den Faktor $2\text{--}4\times$ länger) |

<!--en:4b1d8442-->
::: {.slide title="Tokenisierungsverfahren im Vergleich (Fortsetzung)"}

<!--en:408fcbdb-2-->
[Ergebnisse des Abschnitts]{.kicker}

<!--en:6eb2bee8-2-->
| Tokenisierung | Grundalphabet | Vokabulargröße $|\mathcal{V}|$ (*Time Machine*) | Sequenzlänge $T$ | Verhalten außerhalb des Vokabulars (OOV) | Zielkonflikt beim Modellentwurf |

<!--en:fcd4993c-->
| **Unbeschränktes BPE** | UTF-8-Bytes + gierige Paarverschmelzungen | Frei gewählter Hyperparameter (z. B. $1{,}024$) | $\sim 62{,}000$ | **Keine OOV-Fälle** (unbekannte Stämme bleiben als Bytes erhalten) | Verschmilzt auch über Leerzeichen hinweg (`'of the '`) und verteilt so die Statistik einzelner Wörter auf mehrere Token |

<!--en:913d123b-->
| **BPE mit Vortokenisierung (GPT-2 / Llama 3)** | Textstücke aus regulärem Ausdruck + BPE auf Byteebene + Sondertoken mit hohen IDs | $1{,}024 + 3$ (hier) oder $50\textrm{k}\text{--}200\textrm{k}$ (in der Praxis) | $65{,}741$ ($2.72$ Bytes/Token) | **Keine OOV-Fälle** + gegen Prompt-Injection geschützte Sondertoken | Industriestandard als Kompromiss zwischen Sequenzkompression, Kosten der Softmax-Funktion und offenem Vokabular |
