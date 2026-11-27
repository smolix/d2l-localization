<!--en:8265ce9c-->
# Ökosystem

<!--en:e4002bc3-->
Die meisten Modelle in diesem Buch haben wir von Grund auf mit Fashion-MNIST oder mit *The Time Machine* trainiert. So blieb jeder Schritt nachvollziehbar. In der Praxis beginnt ein Projekt selten so. Meist baut es auf der Arbeit anderer auf, etwa auf dem Checkpoint eines vortrainierten Modells, einem öffentlichen Datensatz oder dem Code zu einer Forschungsarbeit. Angenommen, Sie suchen ein Modell, das die Supportanfragen eines kleinen Unternehmens sortiert. Eine Suche nach Textklassifikatoren auf dem Hugging Face Hub liefert mehr als hunderttausend Kandidaten. Zu jedem sehen Sie die Anzahl der Downloads, zu vielen auch eine Modellkarte (engl. model card) und einen Benchmark-Score. Welches Modell sollten Sie zuerst ausprobieren? Erreicht es bei Ihren Anfragen denselben Score? Und laden Sie nächstes Jahr noch dasselbe Modell, das Sie heute evaluiert haben?

<!--en:8afbf452-->
Hier geht es weniger um das Training als um zwei Fragen: Was belegen die Messwerte, und welches Modell haben wir vor uns? Dazu behandeln wir externe Artefakte so, wie wir in der Softwareentwicklung mit Abhängigkeiten umgehen. Wir suchen sie dort, wo die Fachgemeinschaft sie bereitstellt. Wir beurteilen sie anhand von Messungen für unsere eigene Aufgabe und verlassen uns nicht allein auf Ranglisten (engl. leaderboards). Außerdem halten wir fest, welche Version wir unter welcher Lizenz verwendet haben. Ohne diese Angaben lässt sich ein Ergebnis nicht reproduzieren. Selbst ein im guten Glauben heruntergeladener Checkpoint kann Code auf Ihrem Rechner ausführen, wenn Sie das Dateiformat nicht zuvor prüfen. Dieser Abschnitt folgt auf die Hardwareabschnitte, weil die Wahl des Modells und die des Rechners einander einschränken. Wir sehen uns zuerst an, wo wir Modelle, Datensätze und Forschungsarbeiten finden. Dann untersuchen wir, wie man Benchmarks und Ranglisten liest und welche Quellen einen in der Praxis auf dem Laufenden halten. Schließlich klären wir, wie wir die Revision heruntergeladener Artefakte festschreiben, die Artefakte prüfen und mit ihren Lizenzen umgehen.

<!--en:ab206d43-->
## Quellen für Modelle, Daten und Forschungsarbeiten

<!--en:768d2c33-->
### Modelle

<!--en:fe38dc73-->
Auf dem [Hugging Face Hub](https://huggingface.co/models) waren im November 2026 mehr als drei Millionen öffentliche Modelle verzeichnet. Jedes Modell liegt in einem Git-Repository mit einer Modellkarte und Schnittstellen zu den Bibliotheken, die wir in diesem Buch verwenden. Wie beliebt ein Modell ist, sagt wenig über seine Eignung aus. Suchen Sie daher nach Aufgabe, prüfen Sie Aktualität und Downloadzahlen und lesen Sie die Modellkarte. Weitere Quellen ergänzen den Hub:

<!--en:b11284eb-->
* [ModelScope](https://modelscope.cn/), der Hub von Alibaba, beherbergt einen großen Teil der offenen Modelle aus dem chinesischen Ökosystem. Im November 2026 zählte die Plattform mehr als 280 000 Modelle.

<!--en:50c40048-->
* [Kaggle Models](https://www.kaggle.com/models) bietet eine Auswahl an Modellgewichten, die direkt in Kaggles Infrastruktur für Wettbewerbe und Notebooks eingebunden sind (:numref:`sec_hosted_notebooks`).

<!--en:30e9ead0-->
* Die [Ollama-Bibliothek](https://ollama.com/library) bietet eine überschaubare Auswahl von Modellen, die für eine lokale Laufzeitumgebung gepackt sind (:numref:`sec_model_serving`).

<!--en:d3a7b0b8-->
* [`timm`](https://huggingface.co/timm) bietet eine breite Auswahl an Basisnetzen für die Bildverarbeitung. Auf [Civitai](https://civitai.com/) stellt die Community Checkpoints zur Bilderzeugung und LoRA-Adapter bereit.

<!--en:0194f28a-->
### Datensätze

<!--en:a63e5321-->
[Hugging Face Datasets](https://huggingface.co/datasets) (mehr als eine Million öffentliche Datensätze im November 2026) und [Kaggle](https://www.kaggle.com/datasets) decken die meisten Aufgaben des überwachten Lernens ab. Textkorpora in der für das Pre-Training benötigten Größe gehen auf die Rohdaten von [Common Crawl](https://commoncrawl.org/) zurück, aus denen durch Filtern weitere Korpora entstanden. [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) umfasste bei seiner Veröffentlichung 2024 bereits 15 Billionen Token. Mit neueren Webcrawls wuchs der Bestand in späteren Versionen auf über 18.5 Billionen Token. Das mehrsprachige FineWeb2 wendet dieselbe Verarbeitungskette auf mehr als 1000 Sprachen an. FineWeb dient heute vielfach als offene Baseline für das Pre-Training. Diese Rolle hatte zuvor C4, das 2019 zusammen mit T5 erschien. Für Bild-Text-Paare bietet [DataComp](https://www.datacomp.ai/) eine Alternative zu LAION. Der Schwerpunkt liegt auf dem kontrollierten Aufbau und der Evaluation von Datensätzen. Ältere und sehr große Korpora sind mitunter nur über [Academic Torrents](https://academictorrents.com/) verfügbar. Unabhängig von der Quelle haben Datensätze wie Modelle Versionen, Lizenzen und dokumentierte Schwachstellen. Lesen Sie die Dokumentation eines Datensatzes, bevor Sie ihn übernehmen.

<!--en:8e2f2fc8-->
### Forschungsarbeiten und Code

<!--en:e2a51fef-->
Neue Forschungsarbeiten erscheinen zuerst auf [arXiv](https://arxiv.org/list/cs.LG/recent). [Hugging Face Papers](https://huggingface.co/papers) stellt täglich eine Auswahl für die Startseite zusammen und verknüpft die Arbeiten mit dem zugehörigen Code. Diese Aufgabe erfüllte jahrelang Papers with Code, ein Verzeichnis, das zu jeder Arbeit Code und Benchmarks aufführte. Als der Dienst im Juli 2025 eingestellt wurde, übernahm Hugging Face Papers diese Rolle. Die frühere Adresse leitet jetzt dorthin weiter. Der historische Datenbestand ist als [Archiv](https://github.com/paperswithcode/paperswithcode-data) auf dem Stand der Einstellung erhalten. Das Beispiel zeigt, weshalb ein Projekt nicht von einem einzigen externen Verzeichnis abhängen sollte. [Semantic Scholar](https://www.semanticscholar.org/) und [alphaXiv](https://www.alphaxiv.org/) helfen bei der Suche und Diskussion. Veröffentlichter Code liegt meist auf GitHub. Dort liefern frühere Fehlermeldungen und Diskussionen, aktuelle Commits und Berichte über reproduzierte Ergebnisse weitere Anhaltspunkte für die Beurteilung einer Arbeit.

<!--en:6e3024d2-->
## Benchmarks und Ranglisten

<!--en:91ca44af-->
Eine Rangliste ordnet Modelle nach einer Messgröße. Für ein Projekt zählt eine andere: Wie gut löst das Modell die eigene Aufgabe? Auch ein statischer Benchmark verliert mit der Zeit an Aussagekraft. Modelle erreichen die höchsten möglichen Scores, oder Testfragen gelangen in die Trainingsdaten. Zwei eingestellte Benchmarks veranschaulichen diese Probleme. Das Open LLM Leaderboard prüfte mehr als zehntausend offene Modelle mit festen Benchmarks. 2024 ersetzte es die Benchmarks, weil die Modelle deren Obergrenze erreicht hatten. 2025 wurde die Rangliste eingestellt. Im Februar 2026 stellte OpenAI die Berichterstattung über SWE-bench Verified ein, eine Sammlung echter GitHub-Issues für Programmieragenten. Als Gründe nannte OpenAI die Kontamination der Trainingsdaten und Tests, die korrekte Fehlerbehebungen zurückweisen. OpenAI empfahl den schwierigeren [SWE-bench Pro](https://labs.scale.com/leaderboard/swe_bench_pro). Für die Modellauswahl ziehen wir daher unterschiedliche Arten von Messungen heran:

<!--en:d3bacae0-->
* [Arena](https://arena.ai/), früher LMArena und Chatbot Arena, ordnet Modelle nach den Präferenzen von Menschen in verdeckten Paarvergleichen. Was die Nutzer der Website im Durchschnitt bevorzugen, kann von den Anforderungen der Zielaufgabe abweichen.

<!--en:42dca9b9-->
* [LiveBench](https://livebench.ai/) begrenzt die Kontamination, indem es regelmäßig neue Fragen veröffentlicht und die neuesten zunächst zurückhält.

<!--en:d1744061-->
* Benchmarks wie [SWE-bench](https://www.swebench.com/) für Programmieragenten prüfen jeweils eine bestimmte Fähigkeit. Für Mathematik, den Umgang mit langem Kontext und Sicherheit gibt es eigene Benchmarks.

<!--en:9c20ed5c-->
* [Artificial Analysis](https://artificialanalysis.ai/) vergleicht Qualität, Preis und Latenz über Modelle und Anbieter hinweg.

<!--en:6a48e4e7-->
* Das [Ranking von OpenRouter](https://openrouter.ai/rankings) ordnet Modelle nach der Anzahl der Token, die Kunden an sie senden. Diese im tatsächlichen Gebrauch erkennbare Präferenz ist besonders aufschlussreich, wenn sie von den Qualitätsranglisten abweicht.

<!--en:84a5c2af-->
Stellen Sie nach der Vorauswahl einen kleinen Evaluationsdatensatz aus Ihrer Zielaufgabe zusammen und prüfen Sie jeden Kandidaten daran. Bei der Zielaufgabe unterscheiden sich Modelle auf benachbarten Rangplätzen oft stärker, als ihre Scores auf der Rangliste vermuten lassen. Auch ihre Reihenfolge kann sich ändern. Die in diesem Buch verwendeten Evaluationsverfahren gelten für die *Auswahl* von Modellen ebenso wie für ihr Training: zurückgehaltene Daten, aussagekräftige Baselines und Fehleranalyse.

<!--en:461bb693-->
## Informationsquellen

<!--en:6ec61dba-->
Täglich erscheinen neue Modelle, Forschungsarbeiten und Werkzeuge. Niemand kann jede Ankündigung lesen. Eine kleine, bewusst getroffene Auswahl an Quellen lässt sich leichter beurteilen als ein unablässiger Nachrichtenstrom. Mit Stand vom November 2026 bietet sich etwa folgende Auswahl an:

<!--en:c6cd78cc-->
* [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/), eine Community, die über neue Modelle mit offenen Gewichten, Quantisierungen, Hardwaremessungen und Versuche zur Reproduktion von Ergebnissen berichtet.

<!--en:66bd6769-->
* Die tägliche Auswahl von Hugging Face Papers anstelle der Hunderte von Arbeiten, die täglich bei arXiv eingehen.

<!--en:288c3869-->
* Ein Newsletter, etwa [*The Batch*](https://www.deeplearning.ai/the-batch/) von Andrew Ng, [*Import AI*](https://importai.substack.com/) von Jack Clark oder [*Ahead of AI*](https://magazine.sebastianraschka.com/) von Sebastian Raschka.

<!--en:8772f5e3-->
* Die Vorträge und die Community von [GPU MODE](https://github.com/gpu-mode/lectures) zu Systemfragen sowie die [EleutherAI-Community](https://www.eleuther.ai/community) zur offenen Forschung.

<!--en:2dce62e7-->
Forschungslabore kündigen Ergebnisse in sozialen Medien an. Nutzen Sie diese, um von den Ergebnissen zu erfahren. Zitieren Sie die Forschungsarbeit oder die Modellkarte. Die Einleitung zu diesem Kapitel (:numref:`chap_appendix_tools`) nennt ausführlichere Quellen.

<!--en:c3787fe1-->
## Umgang mit externen Artefakten

<!--en:42c34af9-->
### Festgeschriebene Revisionen

<!--en:5dde2c97-->
Ein Modell umfasst mehr als seine Gewichte. Ob sich ein Ergebnis reproduzieren lässt, hängt auch vom Tokenizer, der Vorverarbeitung, der Konfiguration, der Lizenz und der Revision ab (:numref:`fig_tools_model_artifact`).

<!--en:a375b7c0-->
![Ein wiederverwendbares Modellartefakt verbindet Konfiguration, Gewichte, Vorverarbeitung und Dokumentation mit einer unveränderlichen Revision.](../img/tools-ecosystem-artifact.svg)

<!--en:45b964ab-->
Die Repositories auf dem Hub sind Git-Repositories. Der Branch `main` verweist auf einen neuen Stand, sobald der Eigentümer einen Commit auf diesen Branch überträgt. Um morgen dieselben Dateien zu laden, laden Sie den Stand eines bestimmten Commits herunter:

<!--en:8b3d1c8c-->
Eine kleine Manifestdatei hält ausdrücklich fest, um welches Modell es sich handelt. Ergänzen Sie für den Produktivbetrieb die Hashwerte der Dateien, die Bibliotheksversionen und ein Evaluationsprotokoll:

<!--en:72f399f3-->
Auch bei *abgeleiteten* Artefakten müssen wir die Revision festschreiben. Ein LoRA-Adapter ohne Angabe der Revision seines Basismodells ist unvollständig. Bei der Umwandlung in ein quantisiertes Format (GGUF oder AWQ, siehe :numref:`sec_model_serving`) entsteht ein neues Artefakt. Wir müssen prüfen, wie genau es das numerische Verhalten des Originals wiedergibt. Das Feld `parent` von `Artifact` hält fest, aus welchem Artefakt ein anderes abgeleitet wurde.

<!--en:3800e5b4-->
### Sicherheit und Lizenzen

<!--en:0adddbdb-->
Ein heruntergeladenes Modell wird Teil Ihrer Softwarelieferkette (engl. software supply chain). Dabei stellen sich dieselben Sicherheits- und Lizenzfragen wie bei jeder Bibliothek eines Drittanbieters:

<!--en:cdae7bcc-->
* Bevorzugen Sie **safetensors**, ein Containerformat, das ausschließlich Tensoren enthält. Ältere Checkpoints im Pickle-Format können beim Laden beliebigen Code ausführen. Der Hub durchsucht solche Dateien nach bekannten Schadmustern. Das verringert das Risiko, beseitigt es aber nicht.

<!--en:99d287cc-->
* `trust_remote_code=True` führt Python-Code aus dem Repository auf Ihrem Rechner aus. Lesen Sie den Code und schreiben Sie seine Revision fest. Verwenden Sie diese Option nur, wenn die Architektur sie erfordert.

<!--en:22d9c135-->
* „Offen“ umfasst sehr unterschiedliche Lizenzbedingungen. Manche Gewichte stehen unter freizügigen Lizenzen wie Apache oder MIT. Andere Lizenzen regeln die zulässige Nutzung und setzen Schwellenwerte für den kommerziellen Einsatz. Manche Modelle sind nur für die Forschung freigegeben, bei anderen muss man die Bedingungen für jedes Nutzerkonto akzeptieren, bevor man Zugriff erhält. Halten Sie die Lizenz der festgeschriebenen Revision fest, denn ein Repository kann seine Lizenz von einer Revision zur nächsten ändern.

<!--en:79328723-->
* Hub-Clients speichern jede heruntergeladene Datei zwischen. Einige Dutzend bis mehrere Hundert Gigabyte in `~/.cache/huggingface` sind keine Seltenheit. Mit `hf cache ls` listen Sie den Inhalt des Caches auf, mit `hf cache rm` löschen Sie ein Repository oder eine Revision. Der Befehl `hf cache prune` entfernt Revisionen, auf die kein Branch oder Tag mehr verweist, sowie unvollständige Downloads.

<!--en:cdabafa9-->
So wird aus einem Modell, das Sie auf einem Hub finden, eine Softwareabhängigkeit, deren Wahl Sie begründen, deren Stand Sie reproduzieren und deren Herkunft Sie prüfen können. Ob das Modell für Ihre Aufgabe gut genug ist, klärt nur eine Evaluation mit Ihren eigenen Daten. Die folgenden Aufgaben zeigen, wie groß der Evaluationsdatensatz sein muss, um zwei Modelle unterscheiden zu können.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:74114123-->
Vortrainierte Modelle, Datensätze und Forschungsarbeiten findet man vor allem auf dem Hugging Face Hub, auf ergänzenden Plattformen und auf arXiv. Ranglisten nach menschlichen Präferenzen, kontaminationsresistente Benchmarks, Tests einzelner Fähigkeiten sowie Preis- und Latenzdaten messen jeweils etwas anderes. Benchmarks werden eingestellt, wenn Modelle die erreichbare Höchstleistung erzielen oder die Aufgaben auswendig lernen. Deshalb entscheidet keine Rangliste allein über die Modellauswahl. Grundlage der Auswahl ist eine kleine Evaluation mit Beispielen aus der eigenen Aufgabe und mit Fehlerbalken. Jedes externe Artefakt ist eine Abhängigkeit in der Softwarelieferkette. Schreiben Sie seine Revision fest, bevorzugen Sie safetensors, prüfen Sie Code aus fremden Repositories und dokumentieren Sie die Lizenz sowie das Ursprungsartefakt.

<!--en:2294243a-->
## Aufgaben

<!--en:e809b666-->
1. **Punktabstände in Arena.** Arena (früher LMArena) passt das Bradley-Terry-Modell aus :numref:`sec_regularized` an die Stimmen aus Paarvergleichen an :cite:`Chiang.Zheng.Sheng.ea.2024` und gibt die Scores auf der Elo-Skala an. Lassen wir Unentschieden außer Betracht, so schlägt Modell A das Modell B mit der Wahrscheinlichkeit
   $$P(A \succ B) = \frac{1}{1 + 10^{-(R_A - R_B)/400}},$$
   wobei $R_A$ und $R_B$ die angegebenen Scores sind.
    1. Berechnen Sie diese Wahrscheinlichkeit für Punktabstände $R_A - R_B$ von 10, 20, 50 und 100 Punkten.
    1. Linearisieren Sie $P(A \succ B)$ um $R_A = R_B$. Zeigen Sie, dass der geschätzte Abstand $R_A - R_B$ bei $v$ unabhängigen Stimmen für zwei ähnlich starke Modelle einen Standardfehler von etwa $347/\sqrt{v}$ Punkten hat. Wie viele Stimmen sind nötig, damit ein Abstand von 20 Punkten zwei Standardfehler von null entfernt liegt?
    1. Über alle Stimmen hinweg führt Modell A mit 20 Punkten vor Modell B. Auf Prompts, die Ihrer Aufgabe ähneln, entfallen 10% der Stimmen. Davon gewinnt B 60%. Wie hoch ist die Gewinnrate von A bei den übrigen Stimmen? Was bedeutet das Ergebnis für die Wahl zwischen A und B?
   :exlabel:`ex_software-ecosystem-rating-gaps`

<!--en:9ecaecba-->
1. **Was eine Revision festlegt.** Auf ein Hugging-Face-Repository lässt sich über den Branch `main`, über ein Tag wie `v1.0`, über eine Pull-Request-Referenz wie `refs/pr/3` oder über einen 40-stelligen Commit-Hash zugreifen.
    1. Welche dieser Kennungen verweisen immer auf dieselben Dateien, welche können morgen auf andere Dateien verweisen? Wer kann die jeweilige Referenz verschieben? Datensatz-Repositories lassen sich auf dieselbe Weise ansprechen. In `HuggingFaceFW/fineweb` liegt jede frühere Version in einem gleichnamigen Branch, etwa `v1.3.0`. Welche Kennung dieses Datensatzes sollte die Manifestdatei für das Training enthalten?
    1. Ein festgeschriebener Commit eines Repositorys enthält eine `config.json`. Ihr Eintrag `auto_map` lautet `"AutoModelForCausalLM": "other-org/shared-code--modeling.MyModel"`. Mit `trust_remote_code=True` importiert `transformers` dann `modeling.py` aus dem Repository `other-org/shared-code`. Lesen Sie `get_class_from_dynamic_module` in `transformers/dynamic_module_utils.py`. Welche Revision von `modeling.py` läuft, wenn Sie nur `revision` mit Ihrem Commit übergeben? Was muss daher in der Manifestdatei stehen?
    1. Nennen Sie drei Eingaben außerhalb aller beteiligten Repositories, die die Ausgaben des geladenen Modells auch dann verändern, wenn jedes Repository auf einen Commit festgeschrieben ist.

<!--en:da79bc8c-->
1. [code] **Codeausführung beim Laden.** Das Python-Modul `pickle` rekonstruiert ein Objekt, indem es das von dessen Methode `__reduce__` zurückgegebene aufrufbare Objekt mit den ebenfalls zurückgegebenen Argumenten aufruft.
    1. Betrachten Sie eine Klasse, deren Methode `__reduce__` den Wert `(print, ("code ran during load",))` zurückgibt. Mit `torch.save` wurde ein Wörterbuch gespeichert, das eine Instanz der Klasse und einen Tensor enthält. Erklären Sie für `torch.load` anhand der Dokumentation, was beim Laden mit den Standardargumenten und was mit `weights_only=False` geschieht. Welche Datei im Repository `openai-community/gpt2`, das auch `model.safetensors`, `tf_model.h5` und `flax_model.msgpack` enthält, wird mit demselben Pickle-Mechanismus gelesen?
    1. Mit `weights_only=True` interpretiert `torch.load` weiterhin die Folge von Pickle-Opcodes, lässt aber nur bestimmte Klassen und Funktionen zu. In PyTorch 2.5.1 und früheren Versionen ließ sich diese Einschränkung umgehen, sodass beim Laden einer präparierten Datei beliebiger Code ausgeführt wurde (CVE-2025-32434). Warum benötigt ein Ladeprogramm für safetensors keine solche Positivliste? Welche Risiken bleiben, wenn Sie ein Modell im safetensors-Format aus einem Repository laden, dem Sie nicht vertrauen?

<!--en:9800612e-->
1. [extended] **Aufgabenspezifische Modellauswahl.** ● Wählen Sie eine Aufgabe, die Sie gut kennen. Ein Beispiel wäre das Beantworten von Supportfragen in Ihrem Fachgebiet oder das Extrahieren von Feldern aus Dokumenten, mit denen Sie arbeiten.
    1. Treffen Sie eine Vorauswahl von drei Modellen. Nutzen Sie dazu Arena oder LiveBench, einen Benchmark für die bei Ihrer Aufgabe geforderte Fähigkeit sowie Artificial Analysis. Wo unterscheiden sich die Rangfolgen? Erklären Sie jede Abweichung anhand dessen, was die jeweiligen Ranglisten messen.
    1. Stellen Sie einen Evaluationsdatensatz für Ihre Aufgabe zusammen. Bestimmen Sie seinen Umfang mit der Standardfehleranalyse für Paarvergleiche aus :numref:`subsec_mdl-paired-comparisons`. Legen Sie dabei den kleinsten Unterschied in der Genauigkeit zugrunde, der Ihre Wahl ändern würde. Ergänzen Sie eine Zufallsbaseline und eine einfache heuristische Baseline, etwa die häufigste Antwort oder die Suche nach Schlüsselwörtern.
    1. Werten Sie die ausgewählten Modelle und die Baselines auf dem Datensatz aus. Geben Sie jeden Score mit seinem Standardfehler an und jede Differenz zwischen zwei Modellen mit ihrem gepaarten Standardfehler. Welche Unterschiede lassen sich mit diesen Daten statistisch nachweisen? Stimmt Ihre Rangfolge mit den Ranglisten überein?

    *Nach Chip Huyen,
    [Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html),
    „Modellauswahl“.*
   :exlabel:`ex_software-ecosystem-pick-task-you-care`

<!--en:994bc08b-->
1. [code] **Ein eigener Tensorcontainer.** ● Eine safetensors-Datei besteht aus einer vorzeichenlosen 8-Byte-Ganzzahl $N$ in Little-Endian-Bytefolge, einem $N$ Byte langen JSON-Header in UTF-8 und einem Bytepuffer. Der Header ordnet jedem Tensornamen den Datentyp `dtype` (etwa `"F32"` oder `"I64"`), die Form `shape` und die Bytepositionen `data_offsets` zu. Letztere bilden ein Paar `[begin, end)` relativ zum Pufferanfang. Ein optionaler Schlüssel `__metadata__` enthält eine Zuordnung von Zeichenketten zu Zeichenketten. Die Tensoren sind zeilenweise in Little-Endian-Bytefolge gespeichert.
    1. Schreiben Sie `save_tensors(path, tensors, metadata)` und `load_tensors(path)` für Wörterbücher mit NumPy-Arrays. Verwenden Sie nur `json`, `struct` und `numpy`. Prüfen Sie, ob Speichern und anschließendes Laden dieselben Arrays ergibt. Falls das Paket `safetensors` installiert ist, prüfen Sie auch, ob `safetensors.numpy.load_file` Ihre Dateien liest.
    1. Ergänzen Sie `load_tensors` um ein Argument, mit dem die Funktion einen einzelnen Tensor anhand seines Namens über `np.memmap` lädt, ohne den übrigen Puffer zu lesen.
    1. Lesen Sie nur den Header von `model.safetensors` aus der Revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` von `openai-community/gpt2`. Rufen Sie dazu mit zwei HTTP-Anfragen gezielt die benötigten Bytebereiche ab. Berechnen Sie aus den Tensorformen die Anzahl der gespeicherten Werte. Weshalb übersteigt sie die 124 Millionen Parameter von GPT-2 small?
    1. Listen Sie die Prüfungen auf, die ein Ladeprogramm vornehmen muss, bevor es einem Header aus unbekannter Quelle vertraut. Erstellen Sie für jede Prüfung eine fehlerhafte Datei, die bei dieser Prüfung durchfällt. Vergewissern Sie sich, dass Ihr Ladeprogramm alle diese Dateien zurückweist.

<!--en:22e59a86-->
[Diskussionen](https://d2l.discourse.group/t/36845)

<!--en:76552394-->
Ökosystem: vortrainierte Modelle, Datensätze, Ranglisten und Sicherheit der Softwarelieferkette<br>**Modell-Hubs · FineWeb und DataComp · Benchmark-Kontamination · Arena-Elo · Festschreiben von Commits · safetensors und pickle im Vergleich · Lizenzierung**

<!--en:5722cf67-->
::: {.slide title="Externe Artefakte sind Abhängigkeiten in der Softwarelieferkette"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:14a8fba9-->
- **1. Suche und Auswahl für den Einsatzbereich:** Treffen Sie unter mehr als 3 Mio. öffentlichen Checkpoints eine Vorauswahl. Prüfen Sie, ob die Modelle ihre Ranglisten-Scores auf Ihren Fachdaten erreichen.

<!--en:fccf1f95-->
- **2. Bitweise Identität und Reproduzierbarkeit:** Schreiben Sie den exakten 40-stelligen SHA-Hash des Git-Commits für Gewichte, Konfiguration und Tokenizer fest. So bleiben die Läufe auch sechs Monate später identisch.

<!--en:e243832e-->
- **3. Sicherheit und Einhaltung der Lizenz:** Verhindern Sie mit `safetensors` die Ausführung fremden Codes durch Pickle-Exploits. Prüfen Sie vor der Bereitstellung die Lizenzbedingungen des Commits.

<!--en:4a283476-->
::: {.slide title="Wo die Community Modelle und Gewichte bereitstellt"}

<!--en:a1e47a6f-->
[Modell-Repositories]{.kicker}

<!--en:66042de8-->
- **Zentrale und weltweit genutzte Hubs:** Hugging Face Hub ($3\text{M+}$ Git-Repositories mit `safetensors` und Modellkarten) und ModelScope ($280\text{K+}$ Modelle).

<!--en:ba0384eb-->
- **Kataloge für gehostete und lokale Inferenz:** Modelle von Kaggle Models lassen sich ohne Download direkt unter `/kaggle/input` einbinden. Ollama bündelt quantisierte GGUF-Gewichte mit Chatvorlagen.

<!--en:42d5fa29-->
- **Fachspezifische Sammlungen:** `timm` vereinheitlicht Basisnetze für die Bildverarbeitung mit PyTorch. Civitai stellt Adapter für Diffusionsmodelle bereit.

<!--en:5011ce52-->
::: {.slide title="Offene Datensätze und die Herkunft von Daten für das Pre-Training im Webmaßstab"}

<!--en:0cfcd846-->
[Daten-Repositories]{.kicker}

<!--en:97d1ae16-->
- **Herkunft großer Textkorpora für das Pre-Training:** WARC-Roharchive von Common Crawl $\to$ gefiltertes **C4** (T5, 2019) $\to$ **FineWeb** ($15\text{T}\to 18.5\text{T+}$ Token nach Deduplizierung) und **FineWeb2** ($1{,}000+$ Sprachen).

<!--en:325f6874-->
- **Korpora für überwachtes und multimodales Lernen:** HF/Kaggle Datasets unterstützen Streaming mit Arrow. **DataComp** stellt überprüfbare Bild-Text-Paare zusammen. Academic Torrents verteilt mehrere Terabyte große Archive.

<!--en:a8bd8adf-->
::: {.slide title="Forschungsarbeiten, Code und Reproduktionen finden"}

<!--en:dc34dc41-->
[Recherche]{.kicker}

<!--en:f5cd3ea0-->
- **Erste Anlaufstellen:** **arXiv** (`cs.LG`, `cs.CL`, `cs.CV`), **Hugging Face Papers** (verknüpft Preprints direkt mit Gewichten, Datensätzen und Code auf GitHub) und **Semantic Scholar / alphaXiv**.

<!--en:77cfe49d-->
- **Lehre aus dem Ende von Papers with Code (Juli 2025):** Zentrale Benchmark-Verzeichnisse können ihren Betrieb einstellen. Weder der Build noch die Evaluationspipeline darf davon abhängen, dass eine einzelne externe Website erreichbar bleibt.

<!--en:e37abf3d-->
::: {.slide title="Warum statische Ranglisten altern und ihre Aussagekraft verlieren"}

<!--en:594ccb1e-->
[Grenzen von Benchmarks]{.kicker}

<!--en:70a153c2-->
- **Drei Ursachen (Goodharts Gesetz):** Testdaten gelangen über Webdaten ins Training. Die Auswahl unter Tausenden von Checkpoint-Kombinationen führt zur Überanpassung. Fehlerhafte Modultests setzen der messbaren Leistung eine Obergrenze.

<!--en:c6e0f98e-->
- **Zwei bekannte Fälle:** Das statische **Open LLM Leaderboard** wurde 2025 eingestellt, als die Modelle bei den Aufgaben die Obergrenze erreichten. Auf **SWE-bench Verified** folgte im Februar 2026 der gegen Kontamination robustere **SWE-bench Pro**.

<!--en:25e20a8e-->
::: {.slide title="Fünf einander ergänzende Kriterien zur Modellbeurteilung"}

<!--en:0db7f655-->
[Beurteilung der Kandidaten]{.kicker}

<!--en:4a924f3b-->
- **Menschliche Präferenzen und dynamische Tests:** **Arena** (`arena.ai`) misst Präferenzen in verdeckten Paarvergleichen (Elo). **LiveBench** wechselt die zurückgehaltenen Fragen mit objektiven Antworten aus, um Kontamination zu verhindern.

<!--en:1bd88520-->
- **Fähigkeiten, Kosten und Nutzung:** Kombinieren Sie fachspezifische Testreihen (SWE-bench Pro), **Artificial Analysis** (Qualität im Verhältnis zu Token/s und $\$/\text{M}$ Token) und das Tokenvolumen im Produktivbetrieb bei **OpenRouter**.

<!--en:7d746d8f-->
::: {.slide title="Statistische Auflösung der Paarvergleiche in Arena"}

<!--en:562533ca-->
[Elo-Punkte im Bradley-Terry-Modell]{.kicker}

<!--en:4e0e18dd-->
- **Erforderliche Stichprobengröße:** Ein Vorsprung von $+20$ Elo-Punkten ($52.9\%$ Gewinnrate) erfordert **$v \ge (347/10)^2 \approx 1{,}204$ Stimmen im direkten Vergleich**, um überhaupt $2\,\text{SE}$ zu erreichen.

<!--en:0d249d00-->
- **Rangumkehr in Teilgebieten:** Ein Modell mit einem Gesamtvorsprung von $+20$ Elo-Punkten kann im Verhältnis **$40\%\text{--}60\%$** verlieren, wenn man nur die $10\%$ der Prompts betrachtet, die Ihrer Aufgabe entsprechen.

<!--en:5c9dca94-->
::: {.slide title="Evaluation für die Zielaufgabe"}

<!--en:0275ea91-->
[Auswahl für den eigenen Einsatzbereich]{.kicker}

<!--en:142e625a-->
- **1. 3–4 Kandidaten auswählen:** Filtern Sie öffentliche Ranglisten nach Lizenz, Parameteranzahl und Obergrenzen für Grafikspeicher und Latenz.

<!--en:9ec971e6-->
- **2. Umfang des Evaluationsdatensatzes für Paarvergleiche bestimmen:** Ziehen Sie Eingaben aus realen Arbeitsabläufen. Bestimmen Sie $n$ über den gepaarten Standardfehler $\text{SE} = \sqrt{(p_{10} + p_{01})/n}$ und nehmen Sie triviale sowie schnelle heuristische Baselines auf.

<!--en:5e6e3436-->
- **3. Abweichende Paare prüfen:** Untersuchen Sie vor der Auslieferung Beispiele, für deren Ergebnisse gilt: Modell A $\neq$ Modell B.

<!--en:68c1aca9-->
::: {.slide title="Bestandteile eines reproduzierbaren Modellartefakts"}

<!--en:e92caa5b-->
[Identität eines Artefakts]{.kicker}

<!--en:178563f0-->
Ein „Modell“ umfasst mehr als eine Datei mit Gewichtstensoren. Um seine Vorhersagen zu reproduzieren, müssen fünf zusammengehörige Bestandteile auf eine unveränderliche Revision festgelegt sein:

<!--en:490035e7-->
- **1. Konfiguration (`config.json`, `generation_config.json`):** Hyperparameter der Architektur, RoPE-Skalierung sowie Vorgaben für Sampling und Stopptoken.

<!--en:33d6d0b8-->
- **2. Gewichte (`*.safetensors`):** Tensorparameter in einem genau festgelegten numerischen Datentyp oder Quantisierungsformat.

<!--en:da602b6f-->
- **3. Vorverarbeitung und Tokenizer (`tokenizer.json`, Chatvorlage):** Verschmelzungsregeln des Vokabulars, IDs der Sondertoken und Formatierung der System- und Nutzerrollen.

<!--en:e544f7a4-->
- **4. Dokumentation und Lizenz (`README.md`, `LICENSE`):** Vorgesehene Verwendung, Evaluationsergebnisse und rechtliche Bedingungen der jeweiligen Revision.

<!--en:ab1c100a-->
- **5. Unveränderliche Revision (40-stelliger Git-SHA-Hash):** Legt den Stand aller vier Bestandteile zu einem bestimmten Zeitpunkt fest.

<!--en:08d7f53a-->
::: {.slide title="Unveränderliche Revisionen festschreiben und in einer Manifestdatei erfassen" layout="tight"}

<!--en:f61edcd4-->
[Abhängigkeiten festschreiben]{.kicker}

<!--en:23dce963-->
Hub-Repositories sind Git-Repositories. Der Branch `main`, ein Versionstag (`v1.0`) oder eine Pull-Request-Referenz (`refs/pr/3`) können morgen auf einen anderen Commit zeigen. Der Eigentümer kann sie verschieben oder per Force-Push überschreiben.

<!--en:6f9f2502-->
- **Den 40-stelligen SHA-Hash des Commits festschreiben:**

<!--en:918d9582-->
- **Herkunft mit `parent` verfolgen:** Bei einem LoRA-Adapter oder einem quantisierten GGUF/AWQ-Export fehlt eine entscheidende Angabe, wenn unter `parent` nicht der genaue SHA-Hash des Commits des zugrunde liegenden Basismodells steht.

<!--en:9c7befea-->
::: {.slide title="Unveränderliche Revisionen festschreiben und in einer Manifestdatei erfassen (Code)" layout="tight"}

<!--en:f61edcd4-2-->
[Abhängigkeiten festschreiben]{.kicker}

<!--en:f80d8a47-->
::: {.slide title="Checkpoint-Sicherheit: Pickle-Exploits und das safetensors-Format"}

<!--en:ad78bed4-->
[Sichere Deserialisierung]{.kicker}

<!--en:10ed3fea-->
- **Warum ältere `pickle`-Checkpoints (`.bin` / `.pt`) gefährlich sind:** `pickle` arbeitet als stapelbasierte virtuelle Maschine. Über das Protokoll von `__reduce__` kann es beim Laden beliebige aufrufbare Python-Objekte aufrufen. Beim Laden (z. B. mit `torch.load(..., weights_only=False)`) kann daher beliebiger Code laufen. Selbst die Positivliste von `weights_only=True` ließ sich umgehen (**CVE-2025-32434**).

<!--en:46655f28-->
- **Wie das Format `safetensors` die Codeausführung ausschließt:** Auf 8 Byte für die Headerlänge folgen ein reiner UTF-8-JSON-Header (`dtype`, `shape`, Bytepositionen `[begin, end)`) und die zusammenhängenden Tensorbytes. Es gibt keine ausführbaren Opcodes. Mit `mmap` lassen sich die Daten sofort ohne Kopie einblenden.

<!--en:d67db5d6-->
::: {.slide title="Prüfung von externem Code, Lizenzen und lokalem Speicherverbrauch"}

<!--en:16511db8-->
[Umgang mit der Softwarelieferkette]{.kicker}

<!--en:226c3219-->
- **1. `trust_remote_code=True` und `auto_map`:** Damit laufen die `.py`-Dateien des Repositorys. Verweist `config.json` auf Code in einem externen Repository, schreiben Sie **beide** Commit-Hashes fest.

<!--en:61463762-->
- **2. Lizenz des festgeschriebenen Commits:** Unterscheiden Sie Apache-2.0/MIT von eigenen Nutzungsbeschränkungen (Obergrenzen für monatlich aktive Nutzer) und von Gewichten unter CC-BY-NC, die nur nichtkommerziell genutzt werden dürfen.

<!--en:bd5b8c10-->
- **3. Cache bereinigen:** Prüfen und entfernen Sie veraltete Datenobjekte in `~/.cache/huggingface` mit `hf cache ls`, `hf cache rm` und `hf cache prune`.

<!--en:d116609d-->
::: {.slide title="Auswahl aussagekräftiger Informationsquellen"}

<!--en:36b7e254-->
[Auf dem Laufenden bleiben]{.kicker}

<!--en:17dcb972-->
Täglich erscheinen Hunderte von Preprints und Modellen. Statt fortwährend soziale Medien zu durchforsten, nutzen Sie eine kleine Auswahl überprüfbarer Primärquellen:

<!--en:8f84d772-->
- **Tägliche und wöchentliche Forschungsauswahl:**
  - **Hugging Face Papers:** Die Community wählt täglich Preprints aus und verlinkt direkt auf Code und Gewichte.
  - **Ein technischer Newsletter:** z. B. *Ahead of AI* von Sebastian Raschka, *Import AI* von Jack Clark oder *The Batch* von Andrew Ng.

<!--en:6a63e0dc-->
- **Systeme, Kernel und empirische Überprüfung:**
  - **Diskussionen auf `r/LocalLLaMA` und zu `llama.cpp`:** Quantisierungstests aus der Praxis, Durchsatzmessungen und Reproduktionsberichte.
  - **GPU MODE und EleutherAI:** Ausführliche Vorträge zu Kerneln, Rückblicke auf verteilte Trainingsläufe und offene Forschung zum Pre-Training.

<!--en:43c8a806-->
- **Zitierregel:** Ankündigungen weisen auf neue Ergebnisse hin. Prüfen und zitieren Sie stets die Originalarbeit, das Datenblatt oder die Modellkarte.

<!--en:c041d63c-->
::: {.slide title="Verständnisfragen zu Ranglisten, Revisionen und Sicherheit"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:e05a9405-->
Prüfen Sie anhand dieser drei Fragen, wie Sie die Softwarelieferkette und Evaluationsergebnisse beurteilen:

<!--en:d9f19391-->
1. Zwei Modelle liegen auf der Arena-Rangliste um $15$ Elo-Punkte auseinander. Der Abstand beruht auf $v = 400$ Stimmen im direkten Vergleich. Ist dieser Unterschied statistisch signifikant?

<!--en:b7e96963-->
2. Jedes Repository in Ihrer Pipeline ist auf einen 40-stelligen SHA-Hash eines Commits festgeschrieben und lädt `model.safetensors`. Nennen Sie drei Faktoren *außerhalb* der Hub-Repositories, die auf verschiedenen Rechnern dennoch zu unterschiedlichen Modellausgaben führen können.

<!--en:19f62a62-->
3. Sie untersuchen den `safetensors`-Header von `openai-community/gpt2` (GPT-2 small, $124\text{M}$ Parameter). Aus den Tensorformen in der Datei errechnen Sie insgesamt $\sim 137\text{M}$ Werte. Warum übersteigt die Summe aus dem Header $124\text{M}$?

<!--en:557b98b6-->
**Antworten:**

<!--en:5f20bca3-->
1. **Nein.** Bei $v = 400$ Stimmen beträgt der Standardfehler $\text{SE} \approx 347 / \sqrt{400} = 17.35$ Elo-Punkte. Ein Abstand von $15$ Punkten liegt weniger als $1\,\text{SE}$ von null entfernt, also innerhalb der üblichen Stichprobenschwankungen.

<!--en:8e32f93a-->
2. (i) **Versionen der Inferenzsoftware und der Kernel sowie hardwareabhängige Rundung bei Gleitkommaoperationen** (CUDA-/cuBLAS-/vLLM-Version, GPU-Architektur, geänderte Reduktionsreihenfolge durch die Batchgröße), (ii) **Systemprompt, Einbettung in eine Chatvorlage, Systemzeit zur Laufzeit oder Werkzeugausgaben**, (iii) **Sampling mit einem Pseudozufallszahlengenerator ohne festen Startwert oder mit hardwareabhängigem Verhalten** (`temperature > 0`).

<!--en:18602965-->
3. Die Datei enthält auch Puffer, die keine Parameter sind: die Masken für die kausale Attention `attn.bias` mit der Form $1 \times 1 \times 1024 \times 1024$ in jeder der 12 Schichten ($12 \times 1024^2 \approx 12.6\text{M}$ Werte). Zusammen mit den 124.4 Mio. Parametern ergibt das genau die Summe aus dem Header.

<!--en:72fb60e6-->
::: {.slide title="Zusammenfassung: Orientierung im Open-Source-Ökosystem"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:997bdde2-->
| Bereich des Ökosystems | Häufiger Fehler | Bewährtes Vorgehen |

<!--en:dacb4f13-->
| **Suche nach Modellen und Datensätzen** | Nach der Gesamtzahl der Downloads oder der Begeisterung in sozialen Medien auswählen | Nach Aufgabe, Lizenz, Aktualität und Eignung der Architektur filtern, Modell- und Datensatzkarten lesen |

<!--en:c705f39c-->
| **Auslegung von Ranglisten** | Eine einzelne statische Rangliste als verlässlichen Maßstab für jede Aufgabe ansehen | Arena, LiveBench, fachspezifische Tests, Artificial Analysis und das Nutzungsvolumen bei OpenRouter miteinander abgleichen |

<!--en:3565369e-->
| **Modellauswahl** | Den Erstplatzierten ohne Tests im eigenen Einsatzbereich bereitstellen | Einen aufgabenspezifischen Evaluationsdatensatz anhand gepaarter Standardfehler dimensionieren und die Modelle mit einfachen Baselines vergleichen |

<!--en:185c98b6-->
| **Reproduzierbarkeit des Versionsstands** | `revision="main"` oder verschiebbare Versionstags laden | Unveränderliche 40-stellige SHA-Hashes der Git-Commits festschreiben und die Herkunft mit `parent` in einer unveränderlichen Manifestdatei dokumentieren |

<!--en:b316cb0c-->
| **Checkpoint-Sicherheit** | Pickle-Dateien (`.bin` / `.pt`) aus nicht vertrauenswürdigen Quellen oder ungeprüften externen Code laden | `safetensors` vorschreiben, alle mit `trust_remote_code=True` geladenen Module in sämtlichen referenzierten Repositories prüfen und ihre Revisionen festschreiben |

<!--en:aca7f132-->
| **Lizenzierung und Speicherplatz** | Bei „offenen Gewichten“ Apache-2.0 voraussetzen oder die Festplatte mit zwischengespeicherten Datenobjekten füllen | Die Lizenz des festgeschriebenen Commits dokumentieren und `~/.cache/huggingface` mit `hf cache prune` bereinigen |
