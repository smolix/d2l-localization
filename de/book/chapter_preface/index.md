<!--en:3c0e38c7-->
# Vorwort

<!--en:468f88d3-->
## Vorwort zur zweiten Auflage

<!--en:4cd42ef5-->
Im Rückblick scheint die erste Auflage aus einer anderen Welt zu stammen. Wir schrieben Texte und Code von Hand und zeichneten die Abbildungen selbst. Meine Mitautoren wachten mit Eifer über die Einhaltung der Stilregeln. Für jedes neue Framework brauchten wir Fachleute, die ihr Handwerk verstanden: Anirudh Dagar für PyTorch und JAX, Yuan Tang für TensorFlow. Das Buch wuchs auf über 1000 Seiten. Auf GitHub und Discourse gingen Hunderte von Issues, Pull Requests und Kommentaren ein. Alles auf dem neuesten Stand zu halten, wurde zur Vollzeitbeschäftigung. Als wir Autoren schließlich alle unserer Wege gingen, blieb das Buch ein halbes Jahrzehnt liegen. Viele seiner Grundideen haben sich bewährt: gut zugänglicher Code, Lernen am Beispiel und Notebooks als Format. Sie sind so frisch wie 2020. Etwa die Hälfte des Inhalts hat die Zeit gut überdauert. In anderen Bereichen stehen inzwischen weitaus spannendere Fragen, Verfahren und Anwendungen im Vordergrund. Vom Code hat sich außer dem Format der D2L-Bibliothek wenig bewährt, so gewaltig waren die Fortschritte bei der Hardware und den zugehörigen Softwarewerkzeugen.

<!--en:5da87d86-->
Zwei Ereignisse im Jahr 2026 brachten mich dazu, das Projekt wieder aufzunehmen und zu sehen, ob sich D2L „renovieren“ ließe. Erstens fragte ein griechischer Verlag an, ob ich einer Übersetzung und einem Druck auf Griechisch zustimmen würde. Selbst fünf Jahre nach unserer letzten Überarbeitung hatte das Buch also noch so viel zu bieten, dass jemand eine ganz neue Ausgabe drucken wollte. Zweitens haben wir alle erlebt, wie rasch sich KI-Agenten entwickelt haben. Die Werkzeuge von Anthropic, OpenAI und Google erleichtern umfassende Überarbeitungen und Korrekturen, die früher unbezahlbar gewesen wären. Mit ihrer Hilfe konnte ich den Code an die aktuellen Frameworks anpassen und die offenen Issues und Pull Requests auf GitHub einarbeiten. Ich konnte die Fehlermeldungen und Kommentare auf Discourse durchgehen und Unstimmigkeiten in Code, Sprache, Experimenten und Mathematik aufspüren und beheben. Auch die Werkzeuge für die Veröffentlichung sind auf dem neuesten Stand: Das Buch nutzt jetzt [Quarto](https://quarto.org) und [uv](https://docs.astral.sh/uv/). Die Diskussionsforen laufen auf den eigenen Servern von [Discourse](https://d2l.discourse.group). Mithilfe der Agenten konnte ich auch sicherstellen, dass alle Frameworks bei denselben Aufgaben vergleichbar abschneiden. Leistungseinbußen durch ungeschickt eingesetzte JIT-Kompilierung, Unterschiede in der Initialisierung oder im Optimierungsplan und der Rückgriff auf „magische“ Zufallsstartwerte gehören damit der Vergangenheit an. Diesen Luxus hätten wir uns früher nicht leisten können. Ich glaube, dass das Buch dadurch als Lehrwerk deutlich gewinnt: Es zeigt durchgängig, wie sich das jeweilige Framework am besten einsetzen lässt.

<!--en:5e8f780f-->
Neben der Infrastruktur hat sich auch der Inhalt erheblich weiterentwickelt. Mit über 3000 Seiten im PDF sprengt das Buch den Rahmen eines einzelnen Bandes. Es gliedert sich nun in eigenständige Teile. Der Teil [Grundlagen](../chapter_basics/index.qmd) erarbeitet die Grundlagen des Deep Learning und ist als eigenständige Begleitlektüre für einen Einführungskurs gedacht. Der Teil [Fortgeschrittene Themen](../chapter_advanced/index.qmd) führt in vielen Bereichen nahe an den Stand der Forschung heran. Er behandelt [Optimierung](../chapter_optimization/index.qmd), [Zustandsraummodelle](../chapter_recurrent-modern/index.qmd) (engl. state space models), [Transformer](../chapter_transformers/index.qmd), [bestärkendes Lernen](../chapter_reinforcement-learning/index.qmd) (engl. reinforcement learning), [GANs](../chapter_generative-adversarial-networks/index.qmd) und [Diffusionsmodelle](../chapter_diffusion-models/index.qmd), ohne ein einzelnes Gebiet zu sehr zu vertiefen. Angesichts der Fülle neuer Erkenntnisse auf jedem dieser Gebiete wäre das unmöglich. Der Teil [Mathematik für Deep Learning](../chapter_mathematics-for-deep-learning/index.qmd) ist deutlich gewachsen. Er behandelt [Diffusion und Flüsse](../chapter_mdl-dynamics/index.qmd), [Informationstheorie](../chapter_mdl-information-theory/index.qmd) und [Optimierung](../chapter_mdl-optimization/index.qmd) ausführlicher. Er eignet sich als Lehrbuch für einen mathematischen Einführungskurs auf diesem Gebiet. Er vermittelt das Hintergrundwissen, das Forschungsartikel (und manche Kapitel dieses Buches) häufig voraussetzen. Ich danke Rasool Fakoor für seine Kapitel über bestärkendes Lernen und Diffusion.

<!--en:e3fd8b95-->
Es bleibt noch viel zu tun. Der Teil [Sprachmodelle](../chapter_language-models/index.qmd) (engl. language models) ist noch nicht geschrieben. Auch Computer Vision, Grafik, Audio, gesprochene Sprache und halbüberwachtes Lernen kommen bislang nicht zur Sprache. Eine ausführliche Behandlung von Agenten steht ebenfalls noch aus. Mit etwas Zeit, reichlich Token und viel redaktioneller Arbeit werden mir Letztere (die Agenten) helfen, Ersteres (das Buch) zu vollenden. Damit dürfte es auch leichter fallen, das Buch auf dem neuesten Stand zu halten, wenn die Forschung neue Erkenntnisse gewinnt und neue Verfahren entwickelt.

<!--en:12544c61-->
*Alex Smola, November 2026*

<!--en:f8105d57-->
## Vorwort zur ersten Auflage

<!--en:d55e23c5-->
Noch vor wenigen Jahren beschäftigten Großunternehmen und Start-ups keine Heerscharen von Forschenden, die mit Deep Learning intelligente Produkte und Dienste entwickelten. Als wir auf diesem Gebiet anfingen, machte maschinelles Lernen (engl. machine learning) keine Schlagzeilen in den Tageszeitungen. Unsere Eltern hatten keine Vorstellung davon, was maschinelles Lernen war, geschweige denn, warum wir es einer Laufbahn in der Medizin oder den Rechtswissenschaften vorziehen sollten. In der Wissenschaft war maschinelles Lernen ein Gebiet der freien Grundlagenforschung. In der Industrie spielte es nur bei wenigen praktischen Anwendungen eine Rolle, darunter Spracherkennung und Computer Vision. Viele dieser Anwendungen erforderten so viel Fachwissen, dass man sie oft als ganz eigene Gebiete betrachtete, in denen maschinelles Lernen nur einen kleinen Beitrag leistete. Neuronale Netze (engl. neural networks), die Vorläufer der Deep-Learning-Verfahren in diesem Buch, galten damals weithin als überholt.

<!--en:477fd141-->
Innerhalb weniger Jahre hat Deep Learning dann die Welt überrascht. Es trieb den Fortschritt auf so unterschiedlichen Gebieten wie Computer Vision, der Verarbeitung natürlicher Sprache, der automatischen Spracherkennung, dem bestärkenden Lernen und der biomedizinischen Informatik rasant voran. Die Erfolge bei so vielen praktisch bedeutsamen Aufgaben gaben sogar der theoretischen Forschung zum maschinellen Lernen und der Statistik neue Impulse. Dank dieser Fortschritte können wir Autos bauen, die selbstständiger fahren als je zuvor (wenn auch weniger selbstständig, als manche Unternehmen uns glauben machen wollen). Wir können Dialogsysteme bauen, die mit gezielten Rückfragen Fehler im Code finden und beheben. Und wir können Softwareagenten entwickeln, die in Brettspielen wie Go die besten Spieler der Welt schlagen. Das schien einst noch Jahrzehnte entfernt. Schon heute prägen diese Werkzeuge Industrie und Gesellschaft in immer mehr Bereichen. Sie verändern, wie Filme entstehen und wie Krankheiten diagnostiziert werden. Auch in der Grundlagenforschung spielen sie eine wachsende Rolle, von der Astrophysik über die Klimamodellierung und Wettervorhersage bis zur Biomedizin.

<!--en:7fd586dc-->
*Aston Zhang, Zach Lipton, Mu Li und Alex Smola, 2021*

<!--en:c9dfe567-->
## Über dieses Buch

<!--en:6e4bdc5c-->
Mit diesem Buch möchten wir den Einstieg in Deep Learning erleichtern. Deshalb vermitteln wir die *Konzepte*, ihren *Zusammenhang* und den *Code* an einem Ort.

<!--en:bc1c9331-->
### Code, Mathematik und HTML in einem Medium

<!--en:67f95cfd-->
Eine Technik der Informatik kann ihre volle Wirkung erst entfalten, wenn man sie gut versteht, sie gut dokumentiert ist und ausgereifte, gut gepflegte Werkzeuge bereitstehen. Eine klare Darstellung der wesentlichen Ideen erleichtert den Einstieg in die Praxis. Ausgereifte Bibliotheken sollten häufige Aufgaben automatisieren. Vorbildlicher Beispielcode sollte es leicht machen, Standardmodelle für die eigenen Zwecke zu verändern, einzusetzen und zu erweitern.

<!--en:d078b25b-->
Eine Parallele bieten dynamische Webanwendungen. Unternehmen wie Amazon entwickelten schon in den 1990er-Jahren erfolgreiche datenbankgestützte Webanwendungen. Doch erst im folgenden Jahrzehnt entstanden leistungsfähige, gut dokumentierte Webframeworks. Damit konnten kreative Unternehmer die Möglichkeiten dieser Technik auf breiter Basis ausschöpfen.

<!--en:765e3149-->
Die Möglichkeiten des Deep Learning auszuschöpfen ist besonders anspruchsvoll, denn schon eine einzelne Anwendung vereint mehrere Fachgebiete. Wer Deep Learning einsetzt, muss zugleich verstehen, (i) warum man ein Problem auf eine bestimmte Weise formuliert, (ii) welche mathematische Form ein Modell hat und (iii) mit welchen Optimierungsalgorithmen man es an Daten anpasst. Hinzu kommen (iv) die statistischen Prinzipien, nach denen wir beurteilen, ob ein Modell auf bisher unbekannte Daten generalisieren dürfte, und praktische Verfahren, mit denen wir das überprüfen. Schließlich brauchen wir (v) technische Verfahren für ein effizientes Training, mit denen wir die Fallstricke der numerischen Berechnung umgehen und die vorhandene Hardware ausschöpfen. Kritisches Denken für die Formulierung von Problemen, Mathematik für ihre Lösung und Softwarewerkzeuge für die Umsetzung an einem Ort zu vermitteln ist eine beträchtliche Herausforderung. Mit diesem Buch möchten wir diese Gebiete zusammenführen und den Einstieg in die Praxis ermöglichen.

<!--en:d3a46cea-->
Als wir dieses Buch in Angriff nahmen, gab es kein Lehrwerk, das zugleich (i) aktuell blieb, (ii) die Breite des modernen maschinellen Lernens mit genügend fachlicher Tiefe behandelte und (iii) die Genauigkeit eines Lehrbuchs mit dem klaren, lauffähigen Code einer praktischen Anleitung verband. In Blogbeiträgen und GitHub-Repositories fanden wir verstreut zahlreiche Codebeispiele. Sie zeigten, wie man ein bestimmtes Deep-Learning-Framework nutzt (etwa für einfache Matrixberechnungen in TensorFlow) oder einzelne Architekturen wie LeNet, AlexNet oder ResNet implementiert. Solche Beispiele zeigten meist, *wie* man ein Verfahren umsetzt. Offen blieb, *warum* man sich beim Entwurf des Algorithmus so entschieden hatte. Interaktive Artikel, etwa auf [Distill](http://distill.pub) oder in persönlichen Blogs, machten einzelne Themen anschaulich. Sie erfassten aber nur kleine Ausschnitte des Fachs, und oft fehlte der zugehörige Code. Die damals verfügbaren Deep-Learning-Lehrbücher stellten wiederum keine Verbindung zwischen ihren mathematischen Beschreibungen und lauffähigen Implementierungen her. Das galt auch für den umfassenden Überblick über die Grundlagen von :citet:`Goodfellow.Bengio.Courville.2016`. Vielen blieb so unklar, wie sie von der Theorie zum Code gelangen sollten. Zudem waren zu viele Lernangebote nur über kostenpflichtige Kurse kommerzieller Anbieter zugänglich.

<!--en:77f09e2d-->
Wir wollten ein Lehrwerk schaffen, das (i) allen kostenlos zur Verfügung steht und (ii) genügend fachliche Tiefe für den Einstieg in die angewandte Forschung zum maschinellen Lernen bietet. Es sollte (iii) anhand von lauffähigem Code zeigen, *wie* man Probleme in der Praxis löst, und (iv) sowohl uns als auch der Community ermöglichen, es rasch zu aktualisieren. Ergänzend sollte (v) ein [Diskussionsforum](https://d2l.discourse.group/c/5) den direkten Austausch über technische Einzelheiten und offene Fragen ermöglichen.

<!--en:2fdf61b3-->
Diese Ziele standen oft im Widerspruch zueinander. Gleichungen, Sätze und Literaturverweise lassen sich am besten in LaTeX schreiben und setzen. Für Code bietet sich Python an, für Webseiten sind HTML und JavaScript die natürlichen Formate. Außerdem sollte der Inhalt gleichermaßen als ausführbares Notebook, als gedrucktes Buch, als PDF zum Herunterladen und als interaktive Website zugänglich sein. Da kein bestehendes Publikationsverfahren alle Anforderungen erfüllte, stellten wir ein eigenes zusammen (:numref:`sec_how_to_contribute`). Auf GitHub verwalten wir die Quelldateien und koordinieren die Beiträge der Community. Markdown ist das gemeinsame Quellformat für Text, Gleichungen und Code. Daraus entstehen automatisch Jupyter-Notebooks. Quarto erzeugt die Ausgabeformate, Discourse dient als Diskussionsplattform. Kein System ist perfekt, doch diese Werkzeuge bringen die unterschiedlichen Anforderungen gut in Einklang. Unseres Wissens war *Dive into Deep Learning* das erste Buch, das auf diese Weise entstand.

<!--en:27b0d331-->
### Lernen durch Ausprobieren

<!--en:cdf06e42-->
Viele Lehrbücher führen Konzepte streng nacheinander ein und behandeln jedes Thema erschöpfend, bevor sie zum nächsten übergehen. So geht das klassische Lehrbuch von :citet:`Bishop.2006` bei jedem Thema derart ins Detail, dass schon der Weg bis zum Kapitel über lineare Regression einiges an Arbeit erfordert. Fachleute schätzen das Buch gerade wegen seiner Gründlichkeit. Als Einführung für Neulinge ist es dadurch allerdings weniger zugänglich.

<!--en:6dc8f980-->
In diesem Buch führen wir die meisten Konzepte *genau dann ein, wenn sie gebraucht werden*. Sie begegnen einer neuen Idee also in dem Moment, in dem Sie sie für ein konkretes Ziel in der Praxis benötigen. Ein kurzes Kapitel am Anfang behandelt wichtige Grundlagen wie lineare Algebra und Wahrscheinlichkeitsrechnung. Zum Vertiefen steht ein umfassender mathematischer Anhang bereit. Vor allem aber sollen Sie erleben, wie befriedigend es ist, das erste funktionierende Modell zu trainieren, bevor Sie sich mit der abstrakteren Theorie befassen.

<!--en:2c6f7130-->
Einige einführende Notebooks vermitteln die mathematischen Grundlagen im Schnelldurchgang. Danach führt jedes Kapitel eine überschaubare Zahl neuer Konzepte anhand eigenständiger, lauffähiger Beispiele mit realen Datensätzen ein. Diese Vorgehensweise erschwerte die Gliederung. Manche Modelle passen inhaltlich gut in ein gemeinsames Notebook, andere Ideen vermittelt man am besten, indem man mehrere Modelle nacheinander ausführt. Trotzdem hat die Regel *ein lauffähiges Beispiel pro Notebook* einen großen praktischen Vorteil: Sie können für eigene Forschungs- oder Entwicklungsprojekte direkt auf unserem Code aufbauen. Kopieren Sie ein Notebook und passen Sie es an.

<!--en:9e130586-->
Im gesamten Buch verbinden wir lauffähigen Code nach Bedarf mit theoretischen Erläuterungen. Im Zweifelsfall geben wir Ihnen ein Werkzeug schon an die Hand, bevor wir jede Einzelheit erklärt haben. Die Hintergründe vertiefen wir, sobald Sie das Werkzeug in Aktion erlebt haben. So trainieren wir schon die ersten Modelle mit dem *stochastischen Gradientenabstieg* (engl. stochastic gradient descent), lange bevor wir untersuchen, warum er so wirksam ist. Auf diese Weise erhalten Sie rasch die Werkzeuge, die Sie in der Praxis zur Lösung von Problemen brauchen. Dafür bitten wir Sie nur, uns bei einigen didaktischen Entscheidungen zunächst zu vertrauen.

<!--en:9087e740-->
Die zentralen Konzepte des Deep Learning erarbeiten wir auch von Grund auf. Oft betrachten wir Einzelheiten der Modelle, die moderne Deep-Learning-Frameworks normalerweise verbergen. Das gilt besonders für die Grundlagenkapitel. Dort sollen Sie jeden Vorgang in einer Schicht (engl. layer), einer Verlustfunktion (engl. loss function) oder einem Optimierer nachvollziehen können. Dazu stellen wir regelmäßig zwei Implementierungen nebeneinander. Die erste bauen wir ausschließlich aus elementaren Tensoroperationen und automatischem Differenzieren auf. Die zweite fällt kürzer aus und nutzt die übergeordneten Programmierschnittstellen eines Deep-Learning-Frameworks. Sobald wir eine Komponente von Grund auf erklärt haben, greifen wir in späteren Kapiteln auf die fertige Implementierung des Frameworks zurück.

<!--en:02bb71cd-->
### Inhalt und Aufbau

<!--en:ef1db4b0-->
Das Buch beginnt mit drei kurzen Einführungskapiteln. :numref:`chap_installation` führt durch die Einrichtung einer Umgebung, in der Sie den Code ausführen können. :numref:`chap_notation` fasst unsere mathematischen Konventionen zusammen. :numref:`chap_introduction` gibt einen Überblick über Deep Learning. Der Rest des Buches gliedert sich in sechs Teile. Den Aufbau der beiden zentralen Teile „Grundlagen“ und „Fortgeschrittene Themen“ zeigt :numref:`fig_book_org`.

<!--en:eb0b5fab-->
![Aufbau des Buches. Ein Pfeil von einem Kapitel zu einem anderen bedeutet, dass das zweite auf dem ersten aufbaut. Der Pfeil von den Grundlagen zu den fortgeschrittenen Themen zeigt, dass jedes fortgeschrittene Kapitel auf den Grundlagen aufbaut. Die Teile „Sprachmodelle“, „Bildmodelle“ und „Archiv“ sind nicht dargestellt.](../img/front-book-org.svg)

<!--en:8d86b3f8-->
* **Grundlagen**. In :numref:`chap_preliminaries` erarbeiten wir zügig das praktische Rüstzeug für Deep Learning. Wir speichern und verarbeiten Daten als Tensoren, berechnen Ableitungen automatisch und lernen, mit Wahrscheinlichkeiten und Unsicherheit umzugehen. In :numref:`chap_regression` und :numref:`chap_classification` passen wir lineare Modelle an Daten an. Dabei begegnen uns die drei Grundbausteine fast aller Verfahren in diesem Buch: ein parametrisiertes Modell, eine Verlustfunktion, die seine Fehler misst, und ein Optimierungsalgorithmus, der den Verlust verringert. In :numref:`chap_perceptrons` fügen wir versteckte Schichten (engl. hidden layers) hinzu und erhalten so tiefe Netze. Wir fragen, warum Modelle mit so vielen Parametern dennoch auf neue Daten generalisieren können. :numref:`chap_computation` behandelt dann die Funktionsweise von Deep-Learning-Software. Wir setzen Modelle aus wiederverwendbaren Modulen zusammen, initialisieren und speichern sie und beschleunigen sie auf GPUs. In den letzten drei Kapiteln dieses Teils geht die Struktur der Daten direkt in die Architektur ein. Die konvolutionären neuronalen Netze (engl. convolutional neural networks, CNNs) in :numref:`chap_cnn` und :numref:`chap_modern_cnn` nutzen die räumlichen Symmetrien von Bildern. Die rekurrenten Sequenzmodelle in :numref:`chap_rnn` erfassen die Reihenfolge in Texten und Zeitreihen. Die ersten fünf Kapitel lesen Sie am besten der Reihe nach. Danach lassen sich die Kapitel über Bilder und Sequenzen unabhängig voneinander lesen.

<!--en:bd4cacde-->
* **Fortgeschrittene Themen**. Auf dem Stoff des Grundlagenteils bauen wir die Architekturen und Trainingsverfahren auf, mit denen die aktuelle Forschung arbeitet. Diese Kapitel lassen sich in verschiedener Reihenfolge lesen. Die Pfeile in :numref:`fig_book_org` zeigen, welche Kapitel aufeinander aufbauen. Im Grundlagenteil behandeln wir Optimierer, Lernratenpläne und Batchgrößen vorwiegend als praktische Rezepte. :numref:`chap_optimization` untersucht, warum sie funktionieren und wie man sie für immer größere Modelle anpassen muss. :numref:`chap_attention` führt *Attention*-Mechanismen ein. Damit kann jede Position einer Sequenz direkt Informationen von jeder anderen Position abrufen. In :numref:`chap_transformers` schichten wir diese Bausteine zu Transformer-Architekturen, auf denen moderne große Sprachmodelle beruhen. :numref:`chap_modern_rnn` greift die Rekurrenz anhand von Zustandsraummodellen wieder auf. Sie arbeiten mit einem Zustand fester Größe und lassen sich dennoch über alle Positionen einer Sequenz parallel trainieren. :numref:`chap_performance` untersucht am durchgehenden Beispiel eines Transformers, was die Geschwindigkeit eines Trainingsschritts auf einer oder mehreren GPUs begrenzt. In :numref:`chap_reinforcement_learning` und :numref:`chap_deep_rl` wechseln wir vom überwachten Lernen zu Agenten, die mit ihrer Umgebung interagieren und aus Belohnungen lernen. Wir beginnen mit tabellarisch dargestellten Problemen, die klein genug für eine exakte Lösung sind. Anschließend gehen wir mit neuronalen Netzen zu größeren Problemen über. Schließlich behandeln :numref:`chap_gans` und :numref:`chap_diffusion` die generative Modellierung. Neue Daten entstehen entweder durch ein Minimax-Spiel zwischen einem Generator und einem Diskriminator oder indem ein Modell lernt, eine schrittweise Verfälschung der Daten iterativ rückgängig zu machen.

<!--en:b2703d68-->
* **Sprachmodelle und Bildmodelle**. Der Teil „Sprachmodelle“ ist in Vorbereitung und erweitert die Verfahren aus „Fortgeschrittene Themen“ auf moderne große Sprachmodelle. Er behandelt Encoder, Einbettungen (engl. embeddings) und Informationsabruf (engl. retrieval) (:numref:`chap_lm_encoders`), Pre-Training (:numref:`chap_lm_pretraining`), Fine-Tuning (:numref:`chap_lm_finetuning`), die Ausrichtung an Präferenzen und überprüfbaren Belohnungen (:numref:`chap_lm_rl`), Agenten (:numref:`chap_lm_agents`) und den effizienten Betrieb von Modellen (:numref:`chap_lm_serving`). Der Teil „Bildmodelle“ (:numref:`chap_cv`) behandelt Objekterkennung, semantische Segmentierung und neuronale Stilübertragung. Er stammt aus der ersten Auflage und soll demnächst grundlegend überarbeitet werden.

<!--en:2f126fdd-->
* **Mathematik für Deep Learning**. Dieser Anhang entwickelt die mathematischen Grundlagen des Haupttextes. Die Hauptkapitel nennen jedes mathematische Ergebnis und erläutern die Idee dahinter. Für vollständige Beweise, Herleitungen und vertiefende Zusammenhänge verweisen sie auf den Anhang. :numref:`chap_mdl-linear-algebra` behandelt geometrische Projektionen, Eigenwertzerlegungen und die Singulärwertzerlegung. :numref:`chap_mdl-calculus` erarbeitet Ableitungen von Funktionen mehrerer Variablen, Gradienten und die Funktionsweise des automatischen Differenzierens. :numref:`chap_mdl-optimization` liefert theoretische Garantien für das in :numref:`chap_optimization` beobachtete Verhalten und erklärt etwa, warum die Krümmung der Lernrate eine obere Schranke setzt. :numref:`chap_mdl-probability-statistics` behandelt statistische Schätzung, bayessche Inferenz und Generalisierungsschranken. :numref:`chap_mdl-information-theory` erklärt, was die Kreuzentropie misst und wie sich Wahrscheinlichkeitsverteilungen mit Divergenzen und optimalem Transport vergleichen lassen. :numref:`chap_mdl-dynamics` entwickelt die Differentialgleichungen und stochastischen Prozesse, auf denen Diffusionsmodelle und flussbasierte generative Modelle beruhen. Wer mit einem Thema schon vertraut ist, kann das entsprechende Kapitel überspringen oder zum Nachschlagen nutzen. Allen anderen dient der Anhang als eigenständiges einführendes Lehrbuch.

<!--en:344d1818-->
* **Werkzeuge für Deep Learning**. :numref:`chap_appendix_tools` behandelt praktische Aufgaben der Softwareentwicklung: die Notebooks des Buches lokal oder in Cloud-Umgebungen ausführen, GPU-Hardware auswählen und bereitstellen, verteiltes Training skalieren, Modelle im Produktivbetrieb bereitstellen und Beiträge zu diesem Buch leisten. Dort findet sich auch die API-Referenz für die Bibliothek `d2l` (:numref:`sec_d2l`).

<!--en:3c985ab8-->
* **Archiv**. Der letzte Teil bewahrt Kapitel aus der ersten Auflage, die wir in dieser Auflage nicht mehr erweitern. Sie behandeln Gauß-Prozesse, Hyperparameteroptimierung und Empfehlungssysteme (:numref:`chap_gp` bis :numref:`chap_recsys`).

<!--en:96104f9f-->
### Code

<!--en:b3a3b264-->
Die meisten Abschnitte dieses Buches enthalten ausführbaren Code. Viele grundlegende Zusammenhänge im Deep Learning versteht man unserer Erfahrung nach am besten durch eigene Experimente: Man ändert den Code ein wenig und beobachtet, wie sich die Ergebnisse verändern. Im Idealfall würde uns eine übergreifende mathematische Theorie im Voraus sagen, was jede Änderung an der Architektur oder an einem Hyperparameter bewirkt. In der Praxis müssen wir der formalen Theorie aber oft vorgreifen. Trotz rascher Fortschritte fehlt für viele Verfahren weiterhin eine vollständige theoretische Erklärung ihrer Wirksamkeit. Stark nichtlineare Netze mathematisch zu beschreiben ist schwierig. Die Generalisierung hängt oft von strukturellen Eigenschaften realer Daten ab, die sich einer einfachen Formalisierung entziehen. Auch bei modernen Architekturen hat die systematische theoretische Forschung noch nicht mit den empirischen Entdeckungen Schritt gehalten. Wir freuen uns darauf, in künftigen Auflagen dieses Buches tiefere formale Einsichten aufzugreifen, sobald die Theorie des Deep Learning weiter fortgeschritten ist.

<!--en:c86f10a2-->
Unser Code steht für bis zu vier Deep-Learning-Frameworks bereit, jedes auf einer eigenen interaktiven Registerkarte. Für den Teil „Grundlagen“ und den Anhang „Mathematik für Deep Learning“ gibt es Implementierungen in PyTorch, JAX, TensorFlow und MXNet. Der Teil „Fortgeschrittene Themen“ verwendet PyTorch und JAX. :numref:`chap_installation` hilft bei der Wahl des Frameworks und beim Einrichten einer Arbeitsumgebung. Jedes Notebook beginnt mit einer Standardzelle für gemeinsame Importe, die wir in :numref:`sec_preamble` erklären. Damit der Text übersichtlich bleibt, zeigen wir die einleitende Zelle nur an dieser einen Stelle. In den folgenden Abschnitten können wir direkt mit dem Code zum jeweiligen Thema beginnen.

<!--en:bb3d4b26-->
Damit wir denselben Standardcode nicht in jedem Kapitel wiederholen müssen, sammeln wir wiederverwendbare Funktionen und Klassen in der schlanken Bibliothek `d2l`. Codeblöcke mit der Markierung `#@save`, etwa Hilfsfunktionen, Basisklassen oder Standardimporte, definieren und erklären wir einmal im Text. Gleichzeitig speichern wir sie im Paket `d2l`, sodass spätere Kapitel sie direkt über `d2l` aufrufen können. Eine vollständige Liste dieser Funktionen und Klassen finden Sie in :numref:`sec_d2l`. Der vom Framework unabhängige Kern der Bibliothek `d2l` benötigt nur die folgenden Python-Standardimporte:

<!--en:4b9cff6b-->
Sie lesen die MXNet-Ausgabe dieses Buches. Für den Teil „Grundlagen“ und den Anhang „Mathematik für Deep Learning“ stehen MXNet-Implementierungen zur Verfügung. Der Teil „Fortgeschrittene Themen“ beginnt mit :numref:`chap_optimization` und verwendet PyTorch und JAX. Wechseln Sie für diese Kapitel auf eine der beiden Registerkarten. Wir führen den MXNet-Code mit der Version von MXNet 2.0 aus, die die Umgebungskonfiguration in :numref:`chap_installation` festlegt. Bei Versionskonflikten verwenden Sie bitte diese Umgebung. Das MXNet-Backend der Bibliothek `d2l` ergänzt folgende Importe:

<!--en:d3fadea1-->
Sie lesen die PyTorch-Ausgabe dieses Buches. Sie enthält Implementierungen für jedes Kapitel. Wir führen den Code mit der PyTorch-Version aus, die die Umgebungskonfiguration in :numref:`chap_installation` festlegt. Deep-Learning-Frameworks entwickeln sich rasch. Bei Problemen mit einer neueren Version empfehlen wir daher die dort angegebene Umgebung. Das PyTorch-Backend der Bibliothek `d2l` ergänzt folgende Importe:

<!--en:6a407cc8-->
Sie lesen die TensorFlow-Ausgabe dieses Buches. Für den Teil „Grundlagen“ und den Anhang „Mathematik für Deep Learning“ stehen TensorFlow-Implementierungen zur Verfügung. Der Teil „Fortgeschrittene Themen“ beginnt mit :numref:`chap_optimization` und verwendet PyTorch und JAX. Wechseln Sie für diese Kapitel auf eine der beiden Registerkarten. Wir führen den Code mit der TensorFlow-Version aus, die die Umgebungskonfiguration in :numref:`chap_installation` festlegt. Deep-Learning-Frameworks entwickeln sich rasch. Bei Problemen mit einer neueren Version empfehlen wir daher die dort angegebene Umgebung. Das TensorFlow-Backend der Bibliothek `d2l` ergänzt folgende Importe:

<!--en:25e699d6-->
Sie lesen die JAX-Ausgabe dieses Buches. Sie enthält Implementierungen für die Teile „Grundlagen“ und „Fortgeschrittene Themen“ sowie für den Anhang „Mathematik für Deep Learning“. JAX transformiert reine numerische Python-Funktionen: Es differenziert, kompiliert und vektorisiert sie automatisch. Im gesamten Buch definieren wir neuronale Netze mit dem Flax-Modul `nnx` und trainieren sie mit Optimierern aus Optax. Wir führen den Code mit den Versionen der JAX-Bibliotheken aus, die die Umgebungskonfiguration in :numref:`chap_installation` festlegt. Diese Bibliotheken entwickeln sich rasch. Bei Problemen mit einer neueren Version empfehlen wir daher die dort angegebene Umgebung. Das JAX-Backend der Bibliothek `d2l` ergänzt folgende Importe:

<!--en:3b7e757c-->
### Zielgruppe

<!--en:3d43c720-->
Dieses Buch richtet sich an Studierende im Grund- und Aufbaustudium sowie an Fachleute aus Softwareentwicklung und Forschung, die modernes Deep Learning gründlich verstehen und praktisch anwenden möchten. Da wir jedes Konzept von Grund auf entwickeln, setzen wir keine Vorkenntnisse in Deep Learning oder maschinellem Lernen voraus. Um die Verfahren des Deep Learning gründlich zu erklären, brauchen wir sowohl Mathematik als auch Programmierung. Wir setzen aber nur Grundkenntnisse in linearer Algebra, Analysis, Wahrscheinlichkeitsrechnung und Python voraus. Möchten Sie Ihre mathematischen Vorkenntnisse auffrischen oder tiefer in die Theorie einsteigen? Der Anhang [Mathematik für Deep Learning](../chapter_mathematics-for-deep-learning/index.qmd) deckt fast die gesamte Mathematik dieses Buches ab. In den Hauptkapiteln geben wir einer anschaulichen Darstellung und den Grundideen stets Vorrang vor größtmöglicher mathematischer Allgemeinheit.

<!--en:ab2899ee-->
Wer die mathematischen Grundlagen noch weiter vertiefen möchte, dem empfehlen wir einige Klassiker. *Linear Analysis* von :citet:`Bollobas.1999` behandelt lineare Algebra und Funktionalanalysis ausführlich. *All of Statistics* :cite:`Wasserman.2013` bietet eine knappe und klare Einführung in die statistische Inferenz. Joe Blitzsteins [Lehrbuch](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918) und sein [Kurs](https://projects.iq.harvard.edu/stat110/home) zur Wahrscheinlichkeitsrechnung sind didaktische Glanzstücke. Wenn Python für Sie neu ist, empfehlen wir vor dem Einstieg in den Code ein interaktives [Python-Tutorial](http://learnpython.org/).

<!--en:334b5836-->
### Notebooks, Website, GitHub und Forum

<!--en:1cb5fffd-->
Alle ausführbaren Notebooks zu diesem Buch können Sie von der [D2L.ai-Website](https://d2l.ai) und aus unserem [GitHub-Repository](https://github.com/smolix/d2l-neu) herunterladen. Unter [d2l.discourse.group](https://d2l.discourse.group/c/5) betreiben wir außerdem ein reges Diskussionsforum. Haben Sie Fragen zu einem Abschnitt des Buches, finden Sie am Ende des Notebooks einen direkten Link zur zugehörigen Diskussion.

<!--en:37aeb2c1-->
## Danksagung

<!--en:2ea4ef65-->
Wir danken den Hunderten von Menschen, die an der englischen und der chinesischen Ausgabe mitgewirkt haben. Sie haben den Inhalt verbessert und uns wertvolle Rückmeldungen gegeben. Ursprünglich verwendeten wir für den Code in diesem Buch hauptsächlich MXNet. Anirudh Dagar übertrug einen Großteil des früheren MXNet-Codes nach PyTorch, Yuan Tang nach TensorFlow. Beiden gilt unser Dank. Seit Juli 2021 haben wir das Buch neu konzipiert und den Code in PyTorch, MXNet und TensorFlow neu geschrieben. Dabei haben wir PyTorch als wichtigstes Framework gewählt. Wir danken Anirudh Dagar dafür, dass er einen Großteil des neueren PyTorch-Codes nach JAX übertragen hat. Gaosheng Wu, Liujun Hu, Ge Zhang und Jiehang Xie von Baidu danken wir dafür, dass sie einen Großteil des neueren PyTorch-Codes für die chinesische Ausgabe nach PaddlePaddle übertragen haben. Shuai Zhang danken wir dafür, dass er die LaTeX-Vorlage des Verlags in die PDF-Erstellung eingebunden hat.

<!--en:5a5a5764-->
Allen, die auf GitHub zur ersten Auflage beigetragen haben, danken wir dafür, dass sie das Buch für alle verbessert haben. Hier stehen ihre GitHub-Kennungen oder Namen in beliebiger Reihenfolge:
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem,
Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh,
pggPL, RohanKarthikeyan, Ryan Choi und Likun Lei.

<!--en:0bec82b4-->
Wir danken Amazon Web Services, insbesondere Wen-Ming Ye, George Karypis, Swami Sivasubramanian, Peter DeSantis, Adam Selipsky und Andrew Jassy, für ihre großzügige Unterstützung beim Schreiben dieses Buches. Ohne die Zeit und die Mittel, die uns zur Verfügung standen, ohne die Gespräche im Kollegenkreis und ohne den stetigen Zuspruch wäre dieses Buch nie entstanden. Cambridge University Press hat uns bei der Vorbereitung der Veröffentlichung hervorragend unterstützt. Wir danken unserem Lektor David Tranah für seine Hilfe und seine professionelle Arbeit.

<!--en:dc84a00f-->
### Mitwirkende an der zweiten Auflage

<!--en:3945e89f-->
Wir danken auch den vielen Mitgliedern der Community, die in den vergangenen vier Jahren auf GitHub und im Diskussionsforum auf Fehler hingewiesen und präzisere mathematische oder verständlichere didaktische Darstellungen vorgeschlagen haben. Andere haben defekte Links und Fehler im Code behoben oder Pull Requests eingereicht. Dazu gehören Abhishek Gupta (\@abhishek-gupta2707), Akshaya Bali (\@akshayabali), Alankrit (\@alankrit02), Alexhead (\@Alexhead), Amirhossein (\@amirhosein-mesbah), Andrey (\@v4lproik), Andrii Oriekhov (\@andriyor), Ankit Kumar Singh (\@ankitkumarsingh19091995), Ashwinpathak20 (\@ashwinpathak20), Avinash (\@avinashscs), Baptiste (\@baptiste-pasquier), Basti564 (\@Basti564), Berke (\@berkeoral), Bhavesh Laddagiri (\@bhaveshladdagiri), Bjoernenz (\@bjoernenz), Brian Bartoldson (\@bartoldson), Catalin (\@catalin-sipos), Chaitanya (\@chaitanyab2001), Cuong V. Nguyen (\@ncv), Daniel (\@dakenf), Daniel (\@dani2442), Danishjeet Singh (\@danishjeet), dark_master (\@Dark_Master), Dd3607 (\@Dd3607), Denis Kazakov (\@Denis-Kazakov), Dilip (\@DilipSCU), doc_karl (\@Doc_Karl), DomonkosNagy (\@DomonkosNagy), Dongryeol Lee (\@dongryeol-lee), Dr.rer.nat (\@Dr.rer.nat), Duc (\@nguyenduc1102), Ebad Sayed (\@ebadsayed), Elvis (\@elvisesp), Emmett (\@emmett9001), EnesBerkSak (\@EnesBerkSak), Eurus (\@Eurus-Holmes), Evan (\@evanmiller), Evgenii (\@kaiser179), Fardin (\@fardinabbasi), Fenilgandhi (\@fenilgandhi), Fredrick (\@FriskyFred), Fzyzcjy (\@fzyzcjy), Georgy (\@goshik92), Glenn (\@glennmatlin), GoodMan (\@GoodMan666), Hannes (\@hannesdornbusch), Haocong Wang (\@HaocongWang), HappyB1rd (\@HappyB1rd), Harsh (\@Harsh8604), HenryJia (\@HenryJia), Himanshu (\@himanshurawlani), HyeongjinKim (\@HyeongjinKim), Inaki (\@InakiLPP), James (\@james-Howard), Jayanth (\@jayanthjayanth), Jcbl1 (\@jcbl1), Jiawei He (\@h-jiawei), Jinyang (\@jinyangyuan), Joe (\@joekakone), Jon (\@jontonsoup4), Jonathan (\@JonathanRayner), Juanlu (\@Juanlu001), Juha (\@kamiju), kishan_kc (\@Kishan_KC), L. Jia (\@L.Jia), Legin (\@Legin2022), Lieven (\@lievengovaerts), Lin (\@Lin13579), Lockon (\@Lockon007), Lorenzoh (\@lorenzoh), Luohao (\@luohao-brian), Mahmood (\@mahmood-taghavi), Manimaran (\@Manimaran-T), Marc (\@MarcBresson), Mark (\@Mark1001), Martin (\@mjirik), Matthieu (\@matthieubulte), Mhamed (\@mhamedbenhadj), Michael (\@michael-fonder), Mieszko (\@Mieszko4), Minato (\@Minato-namikaze), Moises (\@moises-b2), MrDiver (\@MrDiver), Noam (\@noamgot), Notplus (\@notplus), OvO (\@OvO), Pan (\@pan-x-c), Phumtawan (\@Phumtawan), Piotrek (\@piotrek-k), prajwal_pitlehra (\@Prajwal_Pitlehra), Qaq (\@qaq), Ranis (\@ranisalt), Robert (\@robert-abc), Saketh (\@sakethram), Sam (\@SamFancher), Satvik (\@Satvik-Garg), Sepehr (\@sepehr-gh), Sepehr (\@sepehrnoey), Shichao (\@shichaoy), Shiva (\@shiva-sankaran), Smsxgz (\@smsxgz), Sravan (\@sravan-s), Stest2012 (\@stest2012), Tirth (\@Tirth27), Tushar (\@tusharmaurya), Verify (\@verify-code), Victor (\@victor-iyi), Wj (\@wj-Mcat), Xinyu (\@xinyu-dev) und Yibo (\@yibotongxue).

<!--en:dcd4d01d-->
Den folgenden Leserinnen und Lesern danken wir für ihre rege Beteiligung, ihre Rückmeldungen und die fachlichen Diskussionen im Forum und im Repository:

<!--en:51702d68-->
Amritansh (\@Amritansh), Angad (\@Angad_Sandhu), anh_tuan_bui (\@Anh_Tuan_Bui), arijit_das (\@Arijit_Das), Artyom (\@artyompal), carlos_arriaga (\@Carlos_Arriaga), Daniel, David (\@d_c), Fan (\@Fan_Gu), Hanker (\@Hanker), Igor (\@igormorgado), Jeet (\@Jeet), Jonathan (\@jonathan-code-r), mahmoud_taha (\@Mahmoud_Taha), Marios (\@marios_g), Mark (\@Mark_Hall), max_ah (\@Max_Ah), Maxe (\@Maxe), Mugen (\@Mugen), Nishant (\@Nishant_Thakur), Porter (\@Porter), Raj (\@Raj), Rling (\@Rling) und Sansui (\@Sansui).

<!--en:dda32b61-->
[Diskussion](https://d2l.discourse.group/t/18)

<!--en:db81ad28-->
[Diskussion](https://d2l.discourse.group/t/20)

<!--en:dedd2dcd-->
[Diskussion](https://d2l.discourse.group/t/186)

<!--en:ef35fe23-->
[Diskussion](https://d2l.discourse.group/t/17963)

<!--en:473ac740-->
::: {.slide title="Die Codebasis von D2L"}

<!--en:e6227312-->
Die D2L-Notebooks verwenden gemeinsame Importe und Hilfsfunktionen aus dem Paket `d2l`. Mit `#@save` markierte Blöcke werden dorthin exportiert. So können sich spätere Notebooks auf die Modelle konzentrieren, ohne den Code zur Einrichtung zu wiederholen. Am Anfang der Bibliothek stehen nur Importe und die Bindung des Moduls an den Namen `d2l`:

<!--en:2cd285c2-->
::: {.slide title="Importe je nach Framework"}

<!--en:c01956d1-->
Jede Backend-Registerkarte importiert die im Buch verwendete Tensorbibliothek, die Schichten für neuronale Netze, Datentransformationen und Hilfsfunktionen. Der öffentliche Namensraum `d2l` bleibt gleich, die Implementierung wechselt mit der Registerkarte.

<!--en:6a7286e9-->
::: {.slide title="Hinweise zu den folgenden Notebooks"}

<!--en:7b48f6d3-->
- `#@save` nimmt den Code in die wiederverwendbare Bibliothek `d2l` auf.

<!--en:761e50f3-->
- `#@tab` markiert Backend-spezifischen Code für MXNet, PyTorch, TensorFlow oder JAX.

<!--en:1af7d3e0-->
- Die meisten Kapitel behandeln Algorithmen. Dieses Vorwort erklärt, woher die gemeinsamen Namen und Importe stammen.
