<!--en:1cfabb9f-->
# Einführung

<!--en:e76b887f-->
Bis vor Kurzem bestand fast jedes Computerprogramm, das Ihnen im Alltag begegnete, aus festen Regeln, die sein Verhalten vorgaben. Angenommen, wir möchten einen Onlineshop programmieren. Nach ein paar Stunden am Whiteboard haben wir vielleicht einen ersten brauchbaren Entwurf. Die Kundschaft bedient den Shop über einen Browser oder eine mobile App. Eine Datenbank verwaltet Lagerbestände, Warenkörbe und frühere Bestellungen. Im Herzen des Systems legt die *Geschäftslogik* fest, welche Aktion auf jedes vorhergesehene Ereignis folgt. Legt jemand per Mausklick einen Artikel in den Warenkorb, fügt das Programm eine Zeile ein, die die Benutzer-ID mit der Produkt-ID verknüpft. Wir spielen Grenzfälle durch (was passiert, wenn jemand mit leerem Warenkorb zur Kasse geht?), beheben beim Testen einige Fehler und nehmen das System in Betrieb, *bevor* wir auch nur einen echten Kunden gesehen haben. Wenn Sie explizite Regeln angeben können, die in $100\%$ der Fälle funktionieren, brauchen Sie sich über maschinelles Lernen keine Gedanken zu machen.

<!--en:55cc3f8d-->
Zum Glück für die Forschung zum maschinellen Lernen lassen sich viele Aufgaben, die wir gern automatisieren würden, auch mit menschlichem Einfallsreichtum schwer in Regeln fassen. Stellen Sie sich vor, Sie stehen wieder am Whiteboard, zusammen mit den klügsten Programmierern, die Sie kennen. Wie würden Sie eine der folgenden Aufgaben lösen?

<!--en:3ec5ac63-->
* Schreiben Sie ein Programm, das aus geografischen Informationen, Satellitenbildern und den Beobachtungen eines zurückliegenden Zeitfensters das Wetter für morgen vorhersagt.

<!--en:5996621b-->
* Schreiben Sie ein Programm, das eine frei formulierte Frage entgegennimmt und richtig beantwortet. Erfordert die Antwort eine Berechnung, soll es dafür Code schreiben und ausführen.

<!--en:9f64000b-->
* Schreiben Sie ein Programm, das in einer Straßenszene jede Person und jedes Fahrzeug erkennt und ihre Umrisse nachzeichnet.

<!--en:733eb713-->
* Schreiben Sie ein Programm, das aus Millionen von Büchern oder Musikstücken solche empfiehlt, die einem Nutzer gefallen und auf die er allein nie stoßen würde.

<!--en:6f5db7ca-->
* Schreiben Sie ein Programm, das allein aus der Aminosäuresequenz eines Proteins vorhersagt, zu welcher dreidimensionalen Struktur es sich faltet.

<!--en:f002efdf-->
Selbst die besten Programmierer hätten bei jeder dieser Aufgaben Mühe, die Regeln von Grund auf zu formulieren. Manchmal verändert sich das zugrunde liegende Muster mit der Zeit. Dann muss sich das System laufend anpassen, um erfolgreich zu bleiben. Häufiger hängt der Zusammenhang zwischen Eingaben und Ausgaben von Millionen feiner Wechselwirkungen ab, sei es zwischen rohen Pixelwerten und Objektgrenzen oder zwischen Aminosäuresequenzen und räumlichen Faltungsstrukturen. Ihre Regeln können wir nicht bewusst benennen, selbst wenn unsere eigenen Augen und Ohren die Aufgabe mühelos lösen.

<!--en:6691fdac-->
Das *maschinelle Lernen* (engl. machine learning) untersucht Algorithmen, die eine Aufgabe mit zunehmender Erfahrung besser lösen. Die Erfahrung stammt meist aus Beobachtungsdaten oder aus der Interaktion mit einer Umgebung. Ein regelbasierter Shop verhält sich beim millionsten Kunden genauso wie beim ersten, bis jemand den Code umschreibt. Ein maschinelles Lernsystem dagegen verfeinert sein Verhalten anhand von Daten. Dieses Buch vermittelt die Grundlagen und die Praxis des *Deep Learning*. Zu dieser Familie gehören Lernverfahren mit mehreren Schichten, die heute Fortschritte bei der Verarbeitung von Text, Bildern und gesprochener Sprache, in der Wissenschaft und bei der Entscheidungsfindung ermöglichen.

<!--en:c49056db-->
## Ein Beispiel zum Einstieg

<!--en:b6eafc55-->
Stellen Sie sich vor, Sie steigen ins Auto, rufen *„Hey Siri“* und fragen nach dem Weg zum nächsten Café. Innerhalb von ein bis zwei Sekunden wacht das Telefon auf, verschriftlicht Ihre Anfrage, wählt eine Route und schätzt Ihre Ankunftszeit. Schon an diesem kurzen Austausch sind mehrere Modelle des maschinellen Lernens beteiligt.

<!--en:212742dd-->
Betrachten wir allein den ersten Schritt, die Reaktion auf ein *Aktivierungswort* (engl. wake word) wie *„Alexa“*, *„OK Google“* oder *„Hey Siri“* (:numref:`fig_wake_word`). Stellen Sie sich vor, Sie sitzen in einem ruhigen Zimmer, ausgestattet nur mit einem Computer und einem Codeeditor. Sie möchten die Erkennung eines Aktivierungsworts von Grund auf programmieren. Pro Sekunde zeichnet ein Telefonmikrofon etwa $16{,}000$ Zahlen auf (bei CD-Audio sind es $44{,}100$). Jede Zahl misst die Amplitude des Luftdrucks zu einem Zeitpunkt. Welche explizite Regel könnte ein Feld von $16{,}000$ Rohwerten aus einer Sekunde zuverlässig auf $\{\textrm{yes}, \textrm{no}\}$ abbilden, bei unterschiedlichen Stimmen, Akzenten, Räumen und Verkehrslärm im Hintergrund? Wenn Sie hier nicht weiterkommen, seien Sie unbesorgt. Auch sonst weiß niemand, wie man diese Regel von Hand formuliert.

<!--en:67d44c1b-->
![Die beiden Modelle hinter einer gesprochenen Anfrage. Ein kleines Modell auf dem Telefon lauscht ständig auf das Aktivierungswort und antwortet mit Ja oder Nein. Sobald es das Wort erkennt, bearbeitet ein großes Assistenzmodell in einem Rechenzentrum die eigentliche Anfrage. Beide lernen nach demselben Verfahren.](../img/wake-word.svg)

<!--en:75046faf-->
Der entscheidende Gedanke ist folgender. Wir können zwar keine Regel angeben, die Schallwellen auf Wörter abbildet, doch Beispiele können wir mühelos selbst mit einem Label versehen. Wir können also einen *Datensatz* (engl. dataset) aus Tausenden kurzer Tonaufnahmen sammeln. Zu jeder Aufnahme gibt ein Label an, ob sie das Aktivierungswort enthält. Statt die Erkennungsregeln fest einzuprogrammieren, schreiben wir ein flexibles Programm. Veränderliche numerische *Parameter* bestimmen, wie es Eingaben auf Ausgaben abbildet. Wir können sie uns als Drehknöpfe vorstellen. Sind die Parameter festgelegt, nennen wir das Programm ein *Modell*. Die Menge aller Abbildungen, die wir durch Drehen an den Knöpfen erreichen können, heißt *Modellfamilie* (engl. model family). Das übergeordnete Programm, das die Knöpfe anhand von Daten einstellt, heißt *Lernalgorithmus* (engl. learning algorithm).

<!--en:8a6ee67e-->
Eine gut gewählte Modellfamilie könnte bei einer Einstellung der Knöpfe *„Alexa“* erkennen, bei einer anderen *„Apricot“*. Bei einer zufälligen Einstellung erkennt sie natürlich keines von beiden. Beim *Training* nähern wir uns mithilfe von Daten schrittweise einer brauchbaren Einstellung der Knöpfe (:numref:`fig_ml_loop`).

<!--en:00ab0fb7-->
1. Wir initialisieren die Parameter des Modells zufällig, sodass es anfangs nichts weiß.

<!--en:af8386a3-->
1. Wir ziehen Beispiele aus dem Datensatz (kurze Tonaufnahmen mit Labels aus $\{\textrm{yes}, \textrm{no}\}$).

<!--en:3d939e9d-->
1. Wir ändern die Parameter geringfügig, damit die Vorhersagen des Modells besser mit den Labels dieser Beispiele übereinstimmen.

<!--en:89b61ac8-->
1. Wir wiederholen die Schritte 2 und 3, bis sich die Güte nicht mehr verbessert.

<!--en:3d381678-->
![Ein typischer Trainingsablauf.](../img/ml-loop.svg)

<!--en:cb8cdf62-->
Dieses Vorgehen lässt sich als *Programmieren mit Daten* verstehen. Werfen wir noch einmal einen Blick auf :numref:`fig_wake_word`. Bemerkenswerterweise steckt hinter beiden Enden der gesprochenen Anfrage genau dasselbe Verfahren. Auf dem Telefon läuft mit einer Leistung von wenigen Milliwatt ständig die winzige Aktivierungsworterkennung (mit etwa hunderttausend Parametern). Im Rechenzentrum kann der Dialogassistent (mit Hunderten Milliarden Parametern) offene Fragen beantworten, zwischen Sprachen übersetzen oder Fehler in Software finden und beheben.

<!--en:a48d4684-->
## Grundbausteine

<!--en:d1d8ad46-->
Sagen wir wie bei der Erkennung von Aktivierungswörtern bekannte Labels aus Eingaben vorher, sprechen wir von *überwachtem Lernen* (engl. supervised learning). Vier Grundbausteine begegnen uns beim überwachten Lernen ebenso wie bei allen anderen Lernverfahren in diesem Buch.

<!--en:d284bd84-->
1. **Daten**, aus denen das Modell lernt.

<!--en:afe3a480-->
1. Ein **Modell**, das festlegt, wie aus Eingaben Ausgaben entstehen.

<!--en:51479f24-->
1. Eine **Zielfunktion** (engl. objective function), die misst, wie gut das Modell seine Aufgabe löst.

<!--en:12867292-->
1. Ein **Optimierungsalgorithmus**, der die Modellparameter so anpasst, dass sich der Wert der Zielfunktion verbessert.

<!--en:3e600d96-->
### Daten

<!--en:3140127f-->
Ein Datensatz ist eine Sammlung von *Beispielen* (auch *Datenpunkte*, *Instanzen* oder *Stichprobenwerte* genannt). Jedes Beispiel hat numerische Eigenschaften, die *Merkmale* (engl. features, auch *Eingaben* oder *Kovariaten*) heißen. Bei überwachten Aufgaben gehört zu jedem Beispiel außerdem ein vorgegebener Zielwert, das *Label*.

<!--en:29b09686-->
Manchmal hat jedes Beispiel gleich viele Merkmale. Sie bilden dann einen Vektor fester Länge. Diese Länge heißt *Dimension* der Eingabe. Ein Farbfoto der Größe $200 \times 200$ enthält $200 \times 200 \times 3 = 120{,}000$ Helligkeitswerte in den Kanälen Rot, Grün und Blau. Bei der Aufnahme in ein Krankenhaus lässt sich etwa ein Vektor fester Länge aus Vitalwerten, Laborwerten und Diagnosen erfassen, um das Überleben nach 30 Tagen vorherzusagen. Vektoren fester Länge sind praktisch, doch vieles in der Welt lässt sich nur schwer in ein festes Schema pressen. Fotos im Internet gibt es in allen Auflösungen und Seitenverhältnissen. Kundenrezensionen reichen von zwei Wörtern (*„Es stinkt!“*) bis zu mehreren Seiten. In der Praxis zerlegen wir Text in diskrete *Token* (Wörter oder Wortteile) und stellen jedes Dokument als Sequenz von Token-IDs variabler Länge dar. Eine besondere Stärke des Deep Learning liegt darin, wie selbstverständlich moderne Architekturen Bilder unterschiedlicher Größe sowie Tonaufnahmen und Texte unterschiedlicher Länge direkt verarbeiten.

<!--en:499e4aaa-->
Bei Daten zählen Menge und Qualität. Ausdrucksstarke tiefe Netze profitieren von großen Datenmengen. Moderne Sprachmodelle lernen aus Textkorpora in der Größenordnung von zehn Billionen ($10^{13}$) Token. Bei solchen Mengen können die Modelle vielfältige Strukturen lernen, statt auf starre vereinfachende Annahmen angewiesen zu sein. Doch auch die größte Datenmenge hilft nicht, wenn die Daten fehlerhaft oder nicht repräsentativ sind (*garbage in, garbage out*). Ein Klassifikator für Hautveränderungen, der ausschließlich auf heller Haut trainiert wurde, versagt bei dunkler Haut mit gefährlichen Folgen. Ein Programm zur Vorauswahl von Bewerbungen, das aus verzerrten früheren Einstellungsentscheidungen lernt, automatisiert frühere Vorurteile, ohne dass jemand dies bei der Entwicklung beabsichtigt hätte. Was ein Datensatz abdeckt und wen er auslässt, sollten wir bereits beim Entwurf des Systems fragen.

<!--en:768d2c33-->
### Modelle

<!--en:e6dd3c3e-->
Ein *Modell* ist eine parametrisierte Rechenvorschrift, die Eingaben eines Typs auf Vorhersagen eines anderen Typs abbildet. Deep-Learning-Modelle verketten von unten nach oben viele parametrisierte Schichten aus linearen und nichtlinearen Transformationen. Daher stammt das Wort *tief* in Deep Learning. Die Größenordnung von Modellen reicht von zwei Zahlen, die eine Gerade festlegen, bis zu Hunderten Milliarden *Gewichten* (engl. weights) in einem der leistungsfähigsten Sprachmodelle.

<!--en:b87496fa-->
### Zielfunktionen

<!--en:98becc78-->
Um „aus Erfahrung besser werden“ mathematisch zu fassen, brauchen wir ein skalares Fehlermaß. Wir nennen es *Zielfunktion* oder, da nach üblicher Konvention kleinere Werte besser sind, *Verlustfunktion* (engl. loss function). Bei stetigen numerischen Zielwerten verwenden wir häufig den *quadratischen Fehler* (engl. squared error), also das Quadrat der Differenz zwischen Vorhersage und wahrem Wert. Bei der Klassifikation interessiert uns letztlich oft die Fehlerrate, der Anteil falsch klassifizierter Beispiele. Sie ändert sich sprunghaft mit den Parametern und liefert daher kein glattes Gefälle, dem wir folgen könnten. Stattdessen minimieren wir einen glatten *Ersatzverlust* (engl. surrogate loss), meist die *Kreuzentropie* (engl. cross-entropy). Sie bestraft das Modell, wenn es der richtigen Klasse eine niedrige Wahrscheinlichkeit zuweist (:numref:`chap_mdl-information-theory`).

<!--en:c7585c80-->
Selbst wenn der Verlust auf den Beispielen, mit denen wir die Drehknöpfe einstellen, auf null sinkt, garantiert das keine Genauigkeit auf neuen Eingaben. Uns interessiert die *Generalisierung* (engl. generalization) auf bislang ungesehene Beispiele. Wer die Übungsklausur vom Vorjahr auswendig lernt, kann in einer Abschlussprüfung mit neuen Fragen dennoch durchfallen. Ebenso kann ein Modell mit hoher Kapazität Besonderheiten seiner Trainingsbeispiele auswendig lernen. Dieses Problem heißt *Überanpassung* (engl. overfitting). Wir teilen die Daten deshalb in einen *Trainingsdatensatz* (engl. training set), mit dem wir die Parameter anpassen, und einen zurückgehaltenen *Testdatensatz* (engl. test set). Letzterer dient ausschließlich dazu, die Generalisierung des gelernten Modells zu messen. In der Praxis kommt ein *Validierungsdatensatz* (engl. validation set) für die Modellauswahl hinzu.

<!--en:3167aa86-->
### Optimierungsalgorithmen

<!--en:12a2d45d-->
Sind Daten, Modellfamilie und Verlustfunktion gegeben, brauchen wir einen Algorithmus, der gute Parameterwerte sucht. Fast alle Modelle in diesem Buch lernen mit Varianten des *Gradientenabstiegs* (engl. gradient descent). Bei jedem Schritt berechnen wir, wie sich der Verlust bei einer winzigen Änderung jedes einzelnen Parameters ändern würde, und gehen in Richtung des Gefälles. Diese Berechnung bei jedem Schritt über Millionen oder Billionen Trainingsbeispiele zu mitteln wäre viel zu langsam. Daher schätzen wir den Gradienten bei jeder Aktualisierung anhand eines kleinen, zufällig gezogenen *Minibatchs* von Beispielen. Dieser Algorithmus heißt *stochastischer Gradientenabstieg* (engl. stochastic gradient descent, SGD). Seine modernen adaptiven Varianten (:numref:`chap_optimization`) trainieren jedes tiefe Netz, dem wir begegnen. Nach Abschluss des Trainings bleiben die Gewichte fest. Das Modell dann auf neuen Eingaben vorwärts auszuwerten heißt *Inferenz*.

<!--en:0670d7f4-->
## Arten von Aufgaben im maschinellen Lernen

<!--en:9ea154fa-->
Mit diesen vier Bausteinen können wir nun untersuchen, wie sich die wichtigsten Aufgaben formulieren lassen, denen Sie im Buch begegnen.

<!--en:5866c371-->
### Überwachtes Lernen

<!--en:aa1746c8-->
Beim überwachten Lernen erhält ein Lernalgorithmus einen Datensatz aus Paaren von Merkmalen und Labels. Daraus gewinnt er ein Modell, das neuen Eingaben vorhergesagte Labels zuordnet (:numref:`fig_supervised_learning`). Probabilistisch formuliert schätzt das Modell die bedingte Verteilung des Labels bei gegebenen Merkmalen. Erstaunlich viele praktische Entscheidungen folgen diesem Muster, von der Tumordiagnose anhand von CT-Aufnahmen bis zur Vorhersage des Strombedarfs in der nächsten Stunde.

<!--en:3ff881f1-->
![Überwachtes Lernen.](../img/supervised-learning.svg)

<!--en:03dbe744-->
#### Regression

<!--en:be771d9b-->
Ist der Zielwert eine stetige Zahl und lautet die Frage *wie viel?* oder *wie viele?*, sprechen wir von *Regression*. Betrachten wir die Schätzung des Verkaufspreises einer Immobilie anhand von `[square footage, bedrooms, bathrooms, walk-minutes to downtown]`. Die Merkmale könnten in Manhattan etwa `[600, 1, 1, 60]` betragen, in Pittsburgh dagegen `[3000, 4, 3, 10]`. Nach demselben Muster lassen sich die Dauer eines Krankenhausaufenthalts, die Dauer einer Operation oder die Niederschlagsmenge in sechs Stunden vorhersagen. Da nicht erfasste Faktoren immer für Streuung sorgen, treffen Vorhersagen selten genau den Zielwert. Wie wir noch sehen, entspricht die Minimierung des quadratischen Fehlers einem Modell, das diese Residuen als gaußsches Rauschen auffasst.

<!--en:d01a300d-->
#### Klassifikation

<!--en:09a4e4fd-->
Wählt der Zielwert aus einer diskreten Menge von Kategorien und lautet die Frage *welche davon?*, sprechen wir von *Klassifikation*. Bei zwei Klassen (`{cat, dog}` oder `{benign, malignant}`) liegt *binäre Klassifikation* vor, bei mehr Klassen (etwa den Ziffern `{0, ..., 9}`) *Mehrklassenklassifikation*. Klassifikatoren geben statt eines starren, eindeutigen Labels eine Wahrscheinlichkeitsverteilung über die Klassen aus. Sie drückt die kalibrierte Sicherheit des Modells aus ($90\%$ Katze, $10\%$ Hund). Mit der Kreuzentropie als glatter Verlustfunktion lässt sich der Klassifikator trainieren.

<!--en:6064a337-->
Von Wahrscheinlichkeiten zu Entscheidungen ist es allerdings noch ein Schritt, denn unterschiedliche Fehler können sehr unterschiedliche Kosten haben. Angenommen, Sie entdecken einen Pilz in Ihrem Garten (:numref:`fig_death_cap`). Ein Bildklassifikator schätzt die Wahrscheinlichkeit, dass es sich um einen tödlich giftigen Grünen Knollenblätterpilz handelt, auf $0.2$ (also mit $80\%$ Wahrscheinlichkeit ungefährlich).

<!--en:5d69aee3-->
![Grüner Knollenblätterpilz – nicht essen!](../img/death-cap.jpg)

<!--en:9eaef2d4-->
Wer die wahrscheinlichste Klasse wählt, bekommt den Rat *„Essen Sie ihn!“*. Es wäre töricht, darauf zu hören. Ein köstliches Abendessen ist ein Risiko von $20\%$ für ein Leberversagen nicht wert. Der erwartete Verlust beim Essen ($0.2 \times \infty + 0.8 \times 0 = \infty$) übersteigt bei Weitem den kleinen Verlust, einen guten Pilz wegzuwerfen ($0.8$). Rationale Entscheidungen gewichten vorhergesagte Wahrscheinlichkeiten mit den asymmetrischen Kosten der Handlungen. Hier rettet Ihnen diese Vorsicht das Leben, denn :numref:`fig_death_cap` zeigt tatsächlich einen Grünen Knollenblätterpilz.

<!--en:3987a4fa-->
Die Mehrklassenklassifikation funktioniert auch bei riesigen Labelmengen. Im Kern ist ein modernes *Sprachmodell* (engl. language model) ein Mehrklassenklassifikator über einem Vokabular von ungefähr $100{,}000$ Token. Es beantwortet immer wieder dieselbe Frage: *Welches Token kommt nach dem bisherigen Text als Nächstes?* (:numref:`fig_next_token`). Wir ziehen ein Token aus dieser Verteilung, hängen es an den Prompt an und wiederholen den Vorgang. So entstehen Token für Token zusammenhängende Texte, Gleichungen oder Code.

<!--en:5d3e8610-->
![Vorhersage des nächsten Tokens als Klassifikation. Anhand des bisherigen Textes weist das Modell jedem Token seines Vokabulars eine Wahrscheinlichkeit zu. Von etwa 100 000 Klassen sind die fünf wahrscheinlichsten dargestellt.](../img/intro-next-token.svg)

<!--en:04d35b6f-->
#### Erkennung, Segmentierung und Verschlagwortung

<!--en:453454ed-->
Die Klassifikation mit einem einzigen Label setzt voraus, dass eine Kategorie die gesamte Eingabe beschreibt. Selbst ein fast perfekter Klassifikator für Katzen und Hunde bekommt Schwierigkeiten mit einer Illustration der *Bremer Stadtmusikanten* (:numref:`fig_stackedanimals`). Ein Hahn sitzt auf einer Katze, die auf einem Hund steht, der wiederum auf einem Esel steht.

<!--en:a502972b-->
![Ein Esel, ein Hund, eine Katze und ein Hahn.](../img/stackedanimals.png)

<!--en:330dde0a-->
Welche Ausgabe wünschen wir uns bei einer komplexen Szene? Interessiert uns nur, *welche* Kategorien vorkommen, unabhängig von ihrem Ort, sagt die *Mehrlabelklassifikation* (engl. multi-label classification, auch *Verschlagwortung*, engl. tagging) für jedes mögliche Label unabhängig die Wahrscheinlichkeit einer binären Entscheidung vorher (`donkey`, `dog`, `cat`, `rooster`). Beim maschinellen Sehen (:numref:`chap_cv`) interessiert uns meist auch die räumliche Struktur. Die *Objekterkennung* (engl. object detection) sagt einen Begrenzungsrahmen und eine Kategorie für jedes einzelne Tier vorher. Die *Instanzsegmentierung* (engl. instance segmentation) zeichnet pixelgenaue Masken, auch dort, wo sich zwei Tiere überlappen. Multimodale Bild-Sprach-Modelle beschreiben die Anordnung direkt in natürlicher Sprache.

<!--en:ab022012-->
#### Suche, Ranking und Empfehlung

<!--en:4bdb3de1-->
Bei der Suche und beim Retrieval bewerten wir Kandidaten anhand einer Abfrage und geben die relevantesten zuerst zurück. So ordnen wir Webseiten nach Relevanz oder suchen in einer Dokumentensammlung Belegstellen, auf die ein Sprachmodell seine Antworten stützen kann (:numref:`chap_lm_encoders`). *Empfehlungssysteme* (engl. recommender systems) ergänzen die *Personalisierung*. Sie wählen aus Millionen von Filmen, Liedern oder Produkten passend zum individuellen Geschmack aus. Dafür nutzen sie sowohl explizites Feedback (Sternebewertungen) als auch implizite Signale (Klicks, Wiedergabedauer, übersprungene Titel).

<!--en:addcbf98-->
Wer Empfehlungen allein auf naive Vorhersagen stützt, kann leicht etwas übersehen. Das Feedback ist *zensiert*, weil Menschen Filme, die sie lieben oder hassen, viel häufiger bewerten als mittelmäßige. Schwerer wiegt, dass die Empfehlungen selbst verändern, was die Nutzer als Nächstes sehen. So entstehen *Rückkopplungsschleifen* (engl. feedback loops). Beliebte Inhalte sammeln schon deshalb Klicks, weil sie angezeigt wurden. Nach erneutem Training empfiehlt das Modell sie daraufhin noch häufiger.

<!--en:ba4ada25-->
#### Lernen mit Sequenzen

<!--en:13f22608-->
Regression und Bildklassifikation behandeln aufeinanderfolgende Eingaben als unabhängig. Sobald das Modell ein Beispiel bewertet hat, vergisst es dieses wieder. Gesprochene Sprache, Text, Video und die fortlaufend gemessenen Vitalwerte auf einer Intensivstation sind jedoch *Sequenzen variabler Länge*. Ihre Bedeutung hängt von der zeitlichen Reihenfolge und vom weiter zurückliegenden Kontext ab. *Sequenz-zu-Sequenz*-Aufgaben bilden eine Eingabesequenz auf eine Ausgabesequenz ab. Länge und Zuordnung der Positionen können sich dabei stark unterscheiden.

<!--en:6ca186f5-->
* **Automatische Spracherkennung und Sprachsynthese.** Auf nur wenige Schriftzeichen kommen bei einer gesprochenen Äußerung (:numref:`fig_speech`) Zehntausende Audioabtastwerte pro Sekunde. Es gibt keine Eins-zu-eins-Zuordnung zwischen Abtastwerten und Buchstaben. Bei der Sprachsynthese verläuft die Abbildung in umgekehrter Richtung.

<!--en:154c83e4-->
![Eine Tonaufnahme und ihre Transkription. Wie bei einer Audio-CD wurde das Signal 44 100-mal pro Sekunde abgetastet. Jedes gesprochene Wort umfasst Tausende Amplitudenwerte. Daher gibt es keine Eins-zu-eins-Zuordnung zwischen Eingabe und Ausgabe.](../img/intro-speech.svg)

<!--en:68973f62-->
* **Maschinelle Übersetzung und Sprachaufgaben.** Bei der Übersetzung zwischen Sprachen ändert sich die Reihenfolge von Teilsätzen nichtmonoton. Im Deutschen etwa steht das Verb bekanntlich oft am Ende des Teilsatzes:

<!--en:fcf027e4-->
Noch vor einem Jahrzehnt brauchten Transkription, Übersetzung, Zusammenfassung und Dialog jeweils eine eigens entwickelte Pipeline. Mit modernen Sequenzarchitekturen (:numref:`chap_rnn` bis :numref:`chap_modern_rnn`) kann ein einziges vortrainiertes Modell alle diese Aufgaben lösen.

<!--en:ea7fbd91-->
### Unüberwachtes und selbstüberwachtes Lernen

<!--en:2eca5a1a-->
Von Menschen vergebene Labels sind teuer und aufgabenspezifisch. Rohtexte, Bilder, Tonaufnahmen und wissenschaftliche Messwerte gibt es dagegen im Überfluss. Beim *unüberwachten Lernen* (engl. unsupervised learning) fragen wir, welche Strukturen sich allein aus Daten ohne Labels gewinnen lassen.

<!--en:cd243ed4-->
* **Clusteranalyse und Schätzung von Unterräumen:** Wir gruppieren Fotos oder Verhaltensweisen von Kunden ohne vorgegebene Kategorien zu in sich ähnlichen Gruppen. Oder wir suchen niedrigdimensionale Koordinaten (*Hauptkomponentenanalyse* (engl. principal component analysis) und nichtlineare Mannigfaltigkeiten), die die wesentlichen Freiheitsgrade hochdimensionaler Beobachtungen erfassen.

<!--en:42c30caf-->
* **Repräsentationslernen** (engl. representation learning): Wir betten diskrete Symbole oder multimodale Objekte in einen euklidischen Raum ein. So werden semantische Beziehungen zu geometrischen Beziehungen. Ein klassisches Beispiel sind Vektoranalogien wie `"Rome"` $-$ `"Italy"` $+$ `"France"` $\approx$ `"Paris"`.

<!--en:0a544d15-->
* **Generative Modellierung:** Ein *diskriminatives* Modell bildet hochdimensionale Daten auf ein Label ab (ein Foto hinein, `"cat"` heraus). Ein *generatives* Modell lernt die Datenverteilung so gut, dass es realistische neue Samples erzeugen kann (:numref:`fig_discriminative_generative`). Erste Durchbrüche gelangen mit *Variationsautoencodern* :cite:`Kingma.Welling.2014,rezende2014stochastic`, *Generative Adversarial Networks* :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` und *normalisierenden Flüssen* :cite:`dinh2014nice,dinh2017density`. Heute erzeugen *Diffusionsmodelle* und Flow-Matching-Modelle :cite:`sohl2015deep,song2019generative,ho2020denoising,song2021score` hochauflösende Bilder, Audio, Video und Molekülstrukturen. Sie lernen, zufälliges Rauschen schrittweise in strukturierte Daten umzuwandeln.

<!--en:012ecf64-->
![Ein diskriminatives Modell bildet Daten auf ein Label ab, ein generatives Modell eine Beschreibung auf Daten.](../img/intro-generative.svg)

<!--en:33fede0b-->
Die weitreichendsten Folgen hatte das *selbstüberwachte Lernen* (engl. self-supervised learning). Es gewinnt die Zielwerte für das Training direkt aus Daten ohne Labels. Beispiele sind die Vorhersage des nächsten Tokens in Texten aus dem Internet, die Ergänzung maskierter Wörter :cite:`Devlin.Chang.Lee.ea.2018` und die Rekonstruktion maskierter Bild-Patches :cite:`he2022masked`. Um bei Billionen von Wörtern jedes nächste Token vorherzusagen, muss sich ein einziges Netz Syntax, Weltwissen, die Struktur von Code und Schlussmuster aneignen, ohne auch nur ein von Hand vergebenes Label. Das einmalige Training eines solchen vielseitigen *Basismodells* (engl. base model) heißt *Pre-Training*. Anschließend lässt sich das Modell durch überwachtes *Fine-Tuning*, die Ausrichtung an menschlichen Präferenzen und den Zugriff auf Werkzeuge anpassen. Aus derselben vortrainierten Grundarchitektur wird so ein Klassifikator, ein Assistent oder ein autonomer Programmieragent (:numref:`fig_pretrain_adapt`).

<!--en:ed1bdfea-->
![Einmal vortrainieren, vielfach anpassen. Ein einzelnes Modell lernt beim Pre-Training auf Texten aus dem Internet, das nächste Token vorherzusagen, ganz ohne Labels. Fine-Tuning mit gelabelten Beispielen, Lernen aus menschlichen Präferenzen und der Zugriff auf Werkzeuge machen aus dem Basismodell einen Klassifikator, einen Assistenten oder einen Agenten.](../img/intro-pretrain-adapt.svg)

<!--en:000d2863-->
### Interaktion mit einer Umgebung

<!--en:d74bea3c-->
Überwachtes und selbstüberwachtes Pre-Training laufen *offline* ab. Wir sammeln einmal Daten aus der Welt, danach findet das Lernen hinter verschlossenen Türen statt (:numref:`fig_data_collection`).

<!--en:93f15e40-->
![Erhebung von Daten aus einer Umgebung für das überwachte Lernen.](../img/data-collection.svg)

<!--en:7f91dc0a-->
Sobald wir ein Modell in der Welt einsetzen, löst sich diese saubere Trennung auf. Umgebungen verändern sich mit der Zeit von selbst, und sowohl Angreifer als auch Nutzer reagieren auf das Modell. Wer Spam verschickt, formuliert Nachrichten gezielt so, dass sie den Spamfilter von gestern umgehen. Dadurch entsteht eine *Verteilungsverschiebung* (engl. distribution shift) zwischen Trainingsdaten und Daten im Einsatz.

<!--en:9debc782-->
### Bestärkendes Lernen und Agenten

<!--en:7f5906fd-->
Muss ein System *Aktionen* wählen, deren Folgen den künftigen Zustand der Welt verändern, gelangen wir zum *bestärkenden Lernen* (engl. reinforcement learning) (:numref:`fig_rl-environment`). In jedem Schritt erhält ein *Agent* eine *Beobachtung*. Er wählt gemäß seiner *Strategie* (engl. policy) eine *Aktion* und erhält eine skalare *Belohnung* (engl. reward), während die Umgebung in einen neuen Zustand übergeht.

<!--en:0ead0ea0-->
![Interaktion mit einer Umgebung beim bestärkenden Lernen.](../img/rl-environment.svg)

<!--en:a6f0d5a1-->
Tiefes bestärkendes Lernen erregte erstmals öffentliche Aufmerksamkeit, als Systeme allein anhand von Pixeln Atari-Spiele meisterten :cite:`mnih2015human` und den menschlichen Weltmeister im Go besiegten :cite:`Silver.Huang.Maddison.ea.2016`. Heute spielt es auch im Post-Training von Sprachmodellen eine zentrale Rolle. Mögliche Antworten bewerten entweder gelernte Modelle menschlicher Präferenzen :cite:`Christiano.Leike.Brown.ea.2017,ouyang2022training` oder automatische Prüfer, die Softwaretests ausführen oder mathematische Beweise prüfen :cite:`DeepSeekAI.2025`. Mit externen *Werkzeugen* kann ein Sprachmodell etwa Befehle in einer Shell ausführen, Dateien lesen oder Suchmaschinen abfragen. Wechselt es dabei über mehrere Runden zwischen Beobachtung und Aktion, wird es zum autonomen *Agenten* (:numref:`chap_lm_agents`).

<!--en:00bf9ce3-->
Beim bestärkenden Lernen stellen sich drei Herausforderungen, die es beim gewöhnlichen überwachten Lernen nicht gibt. Die erste ist die *Zuordnung des Erfolgs* (engl. credit assignment). Wenn eine Schachpartie nach 60 Zügen gewonnen oder verloren ist, welchem früheren Zug gebührt das Verdienst oder die Schuld? Zweitens besteht *partielle Beobachtbarkeit* (engl. partial observability). Teile des wahren Weltzustands können dem Beobachter verborgen bleiben. Drittens stehen *Exploitation* (mit der besten bekannten Strategie Belohnungen erzielen) und *Exploration* (unbekannte, möglicherweise bessere Aktionen ausprobieren) in einem Spannungsverhältnis. Nützliche Sonderfälle ergeben sich, wenn der gesamte Zustand beobachtbar ist (*Markov-Entscheidungsprozesse*), wenn Aktionen künftige Zustände nicht verändern (*kontextuelle Banditen*) oder wenn es überhaupt keinen Zustand gibt (*mehrarmige Banditen*).

<!--en:44c050db-->
## Der Weg zum Deep Learning

<!--en:ddfa0db6-->
Die heutigen Architekturen sind jung, doch der Gedanke, mit mathematischen Mitteln aus verrauschten Daten zu lernen, hat in der Wissenschaft eine lange Geschichte. Zu ihr gehören Bernoullis Wahrscheinlichkeitsrechnung, die Methode der kleinsten Quadrate von Legendre und Gauß :cite:`Legendre.1805`, Shannons Informationstheorie, Turings Formulierung maschineller Intelligenz von 1950 :cite:`Turing.1950` und Hebbs Hypothese zur synaptischen Verstärkung :cite:`Hebb.1949`. Frühe *neuronale Netze* (engl. neural networks) orientierten sich lose an biologischen Schaltungen :cite:`McCulloch.Pitts.1943`. Sie begründeten zwei Prinzipien, die bis heute gelten. In aufeinanderfolgenden *Schichten* (engl. layers) wechseln lineare Transformationen mit elementweisen Nichtlinearitäten. Mithilfe der Kettenregel differenziert man die gesamte Folge von einem Ende zum anderen. Dieses Verfahren heißt *Backpropagation* :cite:`Rumelhart.Hinton.Williams.ea.1988`.

<!--en:cf1c96dc-->
Zwischen etwa 1995 und 2005 setzten sich jedoch Kernel-Methoden, Entscheidungsbäume und grafische Modelle weitgehend gegen neuronale Netze durch. Sie boten stärkere Garantien und ließen sich auf kleinen Datensätzen zuverlässig trainieren. Fishers 150 Schwertlilien aus dem Jahr 1936 tauchten weiterhin regelmäßig in Fachartikeln auf, und 60 000 handgeschriebene Ziffern aus MNIST galten als „groß“. Zwei Entwicklungen kehrten das Urteil um. Über mehrere Jahrzehnte wuchs die Menge digitaler Daten aus dem Web und von preiswerten Sensoren rasant. Zugleich ermöglichten ursprünglich für Videospiele entwickelte GPUs parallele Berechnungen (:numref:`tab_intro_decade`). Die Rechenleistung wuchs viel schneller als Speicherbandbreite und Speicherkapazität. Neuronale Netze mit vielen Parametern, die über viele Durchläufe mit Minibatch-SGD lernten, nutzten diese Hardware besonders gut aus. So lebten Grundideen konvolutionärer Netze :cite:`LeCun.Bottou.Bengio.ea.1998`, rekurrenter Netze mit Toren :cite:`Hochreiter.Schmidhuber.1997` und des Q-Learning :cite:`Watkins.Dayan.1992` in weit größerem Maßstab wieder auf.

<!--en:f067db93-->
:Datensätze, die größten darauf trainierten Modelle und der Rechenaufwand eines Trainingslaufs, jeweils nach Größenordnung. Die Datensatzgröße ist in Beispielen (Blüten, Ziffern, Bilder) oder Token angegeben.

<!--en:fcb90698-->
|Jahr|Datensatz|Größtes trainiertes Modell (Parameter)|Rechenaufwand pro Trainingslauf (Gleitkommaoperationen)|

<!--en:08c499e6-->
|1970|Iris, $10^2$ Blüten|$10^1$|$10^5$|

<!--en:5c703224-->
|1998|MNIST, $10^5$ handgeschriebene Ziffern|$10^5$|$10^{13}$|

<!--en:f280313d-->
|2012|ImageNet, $10^6$ Bilder mit Labels|$10^8$|$10^{18}$|

<!--en:f9bbba85-->
|2020|Webtexte, $10^{11}$ Token|$10^{11}$|$10^{23}$|

<!--en:6470e9b6-->
|2025|Webtexte, $10^{13}$ Token|$10^{12}$|$10^{26}$|

<!--en:e62ce758-->
Größe allein genügte nicht. Erst mehrere Durchbrüche bei Algorithmen und Systemen ermöglichten das moderne Deep Learning.

<!--en:b8f6c47f-->
* **Durchgängig gelernte visuelle Hierarchien.** Im Jahr 2012 senkte AlexNet, ein konvolutionäres Netz mit acht Schichten, die Top-5-Fehlerrate auf ImageNet von etwa $26\%$ auf $15\%$ :cite:`Krizhevsky.Sutskever.Hinton.2012`. Zum Training dienten zwei GPUs für Videospiele. Bald darauf sank die Fehlerrate unter $3\%$ :cite:`Hu.Shen.Sun.2018`. Direkt anhand von Pixeln gelernte Filter übertrafen die Ergebnisse jahrzehntelanger Handarbeit an Pipelines für maschinelles Sehen. Dazu trugen Regularisierungsverfahren wie *Dropout* bei :cite:`Bishop.1995,Srivastava.Hinton.Krizhevsky.ea.2014`.

<!--en:35007a73-->
* **Attention und der Transformer.** Mit *Attention-Mechanismen* :cite:`Bahdanau.Cho.Bengio.2014` müssen Sequenzmodelle beliebig lange Eingaben nicht mehr in einen einzigen Vektor fester Größe pressen. Stattdessen lernen sie, Informationen anhand ihres Inhalts nachzuschlagen. Der ganz auf Attention aufgebaute *Transformer* :cite:`Vaswani.Shazeer.Parmar.ea.2017` vereinheitlichte die Modellierung von Text :cite:`Devlin.Chang.Lee.ea.2018,brown2020language`, Bildern :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`, gesprochener Sprache :cite:`gulati2020conformer` und Entscheidungsprozessen :cite:`chen2021decision`.

<!--en:883d4d1b-->
* **Foundation Models, Ausrichtung und Schlussfolgern.** Beim Pre-Training auf Korpora in der Größenordnung des Webs lernen Sprachmodelle, das nächste Token vorherzusagen :cite:`brown2020language,hoffmann2022training`. Anschließend richtet man sie an Präferenzen aus :cite:`Christiano.Leike.Brown.ea.2017,ouyang2022training` und trainiert sie durch bestärkendes Lernen mit Prüfern für Code und Mathematik :cite:`DeepSeekAI.2025`. So wurden aus Sprachmodellen vielseitig einsetzbare Assistenten und Agenten, die in mehreren Schritten Schlussfolgerungen ziehen.

<!--en:bad8b7bc-->
* **Synthese mit tiefen generativen Modellen.** Adversariales Training :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` und schrittweise Entrauschung mit *Diffusionsmodellen* :cite:`sohl2015deep,ho2020denoising,song2021score` haben es zur Routine gemacht, aus Textbeschreibungen fotorealistische Bilder und Videos sowie realistische Tonaufnahmen zu erzeugen :cite:`Rombach.Blattmann.Lorenz.ea.2022,ramesh2022hierarchical,saharia2022photorealistic`.

<!--en:d90bc2f3-->
* **Verteilte Systeme und offene Frameworks.** Verteilte Trainingsalgorithmen koordinieren heute Tausende von Beschleunigern in einem einzigen Lauf. Imperative Frameworks nach dem Vorbild von NumPy unterstützen automatisches Differenzieren und Kompilierung (PyTorch, JAX und TensorFlow). Wofür Promovierende an der Carnegie Mellon University 2014 eine Woche brauchten, genügen damit heute sechs gut lesbare Zeilen Python.

<!--en:1aadffc1-->
## Was Deep Learning ausmacht

<!--en:c71bd424-->
Was Deep Learning letztlich von klassischen Pipelines unterscheidet, ist das *durchgängige Repräsentationslernen* (engl. end-to-end representation learning). Wer vor 2012 maschinelles Lernen auf Bilder oder Sprache anwandte, verbrachte Jahre damit, von Hand Verfahren zur Merkmalsextraktion für das jeweilige Fachgebiet zu entwickeln. Beispiele sind der Canny-Kantendetektor :cite:`Canny.1987` und SIFT-Deskriptoren :cite:`Lowe.2004`. Ihre fest vorgegebenen Ausgaben dienten einem flachen Klassifikator als Eingabe. Deep Learning ersetzt die gesamte Pipeline durch differenzierbare Schichten, die gemeinsam auf das endgültige Ziel abgestimmt werden. Mit der fachspezifischen Konstruktion von Merkmalen fielen auch die Grenzen zwischen Fachgebieten. Varianten derselben Grundarchitekturen übersetzen heute gesprochene Sprache :cite:`Xiong.Wu.Alleva.ea.2018`, berechnen in Sekunden globale Wettervorhersagen für zehn Tage und sagen dreidimensionale Strukturen für mehr als zweihundert Millionen katalogisierte Proteine vorher. Für diesen letzten Durchbruch wurde die Hälfte des Chemienobelpreises 2024 vergeben.

<!--en:95e8fd9f-->
Ebenso bemerkenswert ist die *vorhersagbare Skalierung* (:numref:`fig_scaling_schematic`). Wachsen Modellgröße, Datensatzgröße und Rechenaufwand fürs Training gemeinsam, folgt der jeweils erreichbare Kreuzentropieverlust einem glatten Potenzgesetz, das sich auf logarithmischen Achsen ablesen lässt :cite:`kaplan2020scaling,hoffmann2022training`. Größere Modelle zu trainieren wurde so von einem Schuss ins Blaue zu einer quantitativ planbaren Ingenieuraufgabe. Darin zeigt sich, was Rich Sutton (2019) die *bittere Lektion* (engl. bitter lesson) nannte. Allgemeine Lern- und Suchverfahren, die mit der Rechenleistung skalieren, überholen regelmäßig Ansätze, die auf von Hand programmierten menschlichen Heuristiken beruhen. Natürlich ergibt sich eine skalierbare Architektur nicht von selbst. Wie gestaltet man Architekturen, die Speicher und Rechenleistung effizient nutzen, und wie wählt man Ziele, die verlässliches Verhalten fördern? Diese Fragen motivieren die Teile des Buches zu fortgeschrittenen Themen und Sprachmodellen.

<!--en:93f9240c-->
![Schema eines Skalierungsgesetzes. Auf logarithmischen Achsen ist der Verlust als Funktion des Rechenaufwands für Modelle zunehmender Größe aufgetragen. Die Kurve jedes Modells flacht ab. Die Hüllkurve der Kurvenschar, also das beste Modell für jedes Budget, ist eine Gerade. Da es sich um eine Skizze handelt, zeigen die Achsen nur Zehnerpotenzen. Messkurven finden sich in :numref:`sec_scaling-laws`.](../img/intro-scaling.svg)

<!--en:b1396687-->
Der Einsatz in der Praxis verlangt dennoch Bescheidenheit. Autonome Fahrzeuge sind bereits in mehreren Städten kommerziell im Einsatz. Überall vollständig autonom zu fahren bleibt jedoch wegen des *langen Schwanzes* der Verteilung (engl. long tail) schwierig. Manche Verkehrssituationen treten nur einmal unter einer Million auf, ohne dass die Trainingsaufzeichnungen einen vergleichbaren Fall enthalten. Allgemeiner gilt, dass KI-Systeme weder Magie sind noch im Begriff stehen, ein Bewusstsein zu entwickeln. Es sind statistische Modelle, die von Menschen gewählte Ziele anhand begrenzter, von Menschen gesammelter Daten optimieren. Wenn Modelle Kreditanträge prüfen, medizinisches Fachpersonal unterstützen oder geistige Routinearbeit automatisieren, zählen konkrete Fragen: *Wer hat das Ziel gewählt? Was deckten die Trainingsdaten ab? Wie verhält sich das System bei einer Eingabe außerhalb seines Erfahrungshorizonts?*

<!--en:00317141-->
Von Anfang an hat der offene Austausch den Fortschritt im Deep Learning beschleunigt. Architekturen werden veröffentlicht, quelloffene Bibliotheken und Modellgewichte freigegeben. Forschung und Industrie teilen Code, mit dem sich Ergebnisse nachvollziehen lassen. In diesem Sinne verbindet jedes Kapitel dieses Buches die mathematischen Grundlagen direkt mit ausführbaren Notebooks. Fangen wir an.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:0fb64fe8-->
Mit maschinellem Lernen entstehen Programme, deren Verhalten sich durch Daten und Interaktion verbessert, ohne dass wir die Regeln von Hand schreiben. Jede Aufgabenformulierung verbindet vier Bausteine: Daten, ein parametrisiertes Modell, eine Zielfunktion (Verlustfunktion) und einen Optimierungsalgorithmus, fast immer eine Minibatch-Variante des stochastischen Gradientenabstiegs. An zurückgehaltenen Daten prüfen wir die Generalisierung. Die Aufgaben reichen vom überwachten Lernen (Regression, Klassifikation, strukturierte räumliche Wahrnehmung, Ranking und Sequenztransduktion) über unüberwachtes und selbstüberwachtes Pre-Training bis zur generativen Modellierung und zum bestärkenden Lernen. Deep Learning zeichnet sich dabei dadurch aus, dass es mehrschichtige Repräsentationen *durchgängig* aus Roheingaben lernt. Datensätze in der Größenordnung des Webs, parallele Berechnungen auf GPUs, vorhersagbare Skalierungsgesetze und quelloffene Frameworks für automatisches Differenzieren haben diese Entwicklung ermöglicht. Heute beruht moderne KI für Sprache, maschinelles Sehen, Wissenschaft und autonome Agenten auf einer gemeinsamen Familie von Architekturen.

<!--en:2294243a-->
## Übungen

<!--en:f0b0f7cd-->
1. **Regeln oder Lernen.** Welche Teile des Codes, den Sie gerade schreiben, ließen sich „lernen“? Wo könnte ein Lernverfahren Entwurfsentscheidungen in Ihrem Code automatisch treffen und so sein Verhalten verbessern? Enthält Ihr Code heuristische Entscheidungen, etwa einen von Hand eingestellten Schwellenwert oder eine Vorrangregel?
    1. Formulieren Sie eine solche Entscheidung als Abbildung von Eingaben auf Ausgaben. Beschreiben Sie eine Eingabe, bei der die Heuristik versagt.
    1. Welche Daten und Labels könnten Sie sammeln, um das gewünschte Verhalten zu lernen? Welche Zielfunktion würden Sie beim Training minimieren, und wie würden Sie das gelernte Modell im Vergleich zur Heuristik bewerten?

<!--en:65eafc15-->
1. **Aufgaben formulieren.** Ordnen Sie jede der folgenden Aufgaben einer Formulierung aus diesem Kapitel zu (*Regression*, *Klassifikation*, *Mehrlabelklassifikation*, *Objekterkennung*, *Sequenz-zu-Sequenz-Lernen*, *Clusteranalyse* oder *bestärkendes Lernen*). Geben Sie an, ob das Modell aus einem Label, einer Belohnung oder keinem von beiden lernt.
    1. Den Wiederverkaufspreis eines Gebrauchtwagens aus Alter, Kilometerstand und Modell schätzen.
    1. Jedes Verkehrsschild im Einzelbild einer Fahrzeugkamera mit einem Begrenzungsrahmen umschließen und mit einem Klassenlabel versehen.
    1. Jeder eingehenden E-Mail an den Kundendienst aus einer festen Liste alle Produktbereiche zuordnen, die sie betrifft.
    1. Einen Nachrichtenartikel in einem Absatz zusammenfassen.
    1. Gene anhand ihrer Expressionsmuster über viele Experimente hinweg gruppieren, ohne dass vorab Gruppen festgelegt sind.
    1. Die Heizung eines Gebäudes alle zehn Minuten so einstellen, dass der Energieverbrauch sinkt und die Raumtemperaturen angenehm bleiben. Zum Lernen dienen ausschließlich die Messwerte zum Energieverbrauch und die Beschwerden über das Raumklima nach jeder Anpassung.

<!--en:b8824945-->
1. **Entscheidungsschwelle.** Ein Klassifikator gibt die Wahrscheinlichkeit $p$ dafür aus, dass ein Pilz ein Grüner Knollenblätterpilz ist. Wer einen Grünen Knollenblätterpilz isst, erleidet einen Verlust von $L > 0$. Wer einen essbaren Pilz wegwirft, erleidet einen Verlust von $1$. Einen essbaren Pilz zu essen oder einen Grünen Knollenblätterpilz wegzuwerfen verursacht keinen Verlust.
    1. Leiten Sie in Abhängigkeit von $L$ den Schwellenwert für $p$ her, oberhalb dessen das Wegwerfen des Pilzes einen geringeren erwarteten Verlust hat als das Essen.
    1. Berechnen Sie den Schwellenwert für $L = 1$ und für $L = 1000$. Vergleichen Sie beide Schwellenwerte mit der Regel, die jeweils wahrscheinlichere Klasse zu wählen.

<!--en:2ae58ead-->
1. **Algorithmen, Daten und Rechenaufwand.** Verwenden Sie :numref:`tab_intro_decade`. Dabei bezeichnet $n$ die Datensatzgröße (Beispiele oder Token), $p$ die Parameteranzahl des größten Modells und $c$ den Rechenaufwand eines Trainingslaufs in Gleitkommaoperationen.
    1. Berechnen Sie für jede Zeile die Anzahl der Parameter pro Datenpunkt $p/n$ und den Rechenaufwand pro Parameter $c/p$. Beschreiben Sie, wie sich diese Verhältnisse von 1970 bis 2025 entwickelt haben.
    1. Ein dicht besetzter Transformer, der jedes Token einmal verarbeitet, benötigt ungefähr $6p$ Operationen pro Token ($2p$ im Vorwärtsdurchlauf und $4p$ im Rückwärtsdurchlauf), also etwa $6pn$ Operationen pro Durchlauf durch die Daten. Überprüfen Sie die Schätzung anhand der Zeilen für 2020 und 2025. Erklären Sie, warum sie den Wert in der ImageNet-Zeile von 2012 um mehrere Größenordnungen unterschätzt. Dort durchlief ein konvolutionäres Netz die Bilder beim Training etwa 90-mal.
    1. Ein Verfahren, das alle Paare von Trainingsbeispielen vergleicht (wie es exakte Kernel-Methoden tun), speichert eine $n \times n$-Matrix und benötigt in der Größenordnung von $n^2$ Operationen. Berechnen Sie für jede Zeile den Speicherbedarf einer $n \times n$-Matrix mit 4 Byte pro Eintrag. Bestimmen Sie die erste Zeile, für die ein Terabyte ($10^{12}$ Byte) nicht mehr ausreicht. Erklären Sie, warum diese Entwicklungen tiefe neuronale Netze begünstigten.

<!--en:e37e9ae4-->
1. **End-to-End-Training.** Betrachten Sie ein Fahrerassistenzsystem mit drei Stufen. Zuerst ordnet ein Wahrnehmungsnetz jedem Kamerapixel eines der Labels Fahrzeug, Fußgänger oder Hintergrund zu. Als Maß für die Güte dient der Anteil falsch beschrifteter Pixel. Es folgen ein von Hand abgestimmter Bewegungsplaner und ein ebenso abgestimmter Regler.
    1. Beschreiben Sie einen Fehler des Wahrnehmungsnetzes bei der Pixelbeschriftung, der den Gesamtanteil falsch beschrifteter Pixel kaum verändert, den Planer aber zur Wahl einer gefährlichen Trajektorie veranlasst.
    1. Angenommen, Planer und Regler wären ebenfalls neuronale Netze und würden gemeinsam mit dem Wahrnehmungsnetz durchgängig trainiert. Welche Daten und welche einzelne Zielfunktion wären nötig? Warum könnte man bei der Entwicklung dennoch getrennte Module bevorzugen?

<!--en:5d56df07-->
1. **Vorhersage des nächsten Tokens und Pre-Training.** Betrachten Sie ein auf Webtexten vortrainiertes Sprachmodell mit einem Vokabular von $100{,}000$ Token.
    1. Beschreiben Sie Aufgabenformulierung, Eingabe, Ausgabe und Verlustfunktion mit den Begriffen dieses Kapitels. Erklären Sie, warum keine Labels von Hand vergeben werden mussten.
    1. Nennen Sie eine Aufgabe, die das Modell allein nach dem Pre-Training lösen kann, und eine, die anschließendes Fine-Tuning oder die Ausrichtung an Präferenzen erfordert.

<!--en:7a69672a-->
1. **Rückkopplungsschleifen in Empfehlungssystemen.** Angenommen, ein Empfehlungssystem lernt anhand der Klicks des Vorjahres und wird dann ein Jahr lang eingesetzt. Anschließend wird es anhand der im Einsatz gesammelten Klicks erneut trainiert.
    1. Erklären Sie, wie sich die neuen Trainingsdaten von einer unverzerrten Stichprobe der Nutzerpräferenzen unterscheiden. Nennen Sie je einen Inhalt, dessen Beliebtheit die Daten überschätzen beziehungsweise unterschätzen.
    1. Schlagen Sie für das Einsatzjahr ein Onlineexperiment vor, mit dem sich diese Rückkopplungsschleife nachweisen und messen lässt.

<!--en:50fa08d5-->
[Diskussion](https://d2l.discourse.group/t/22)
