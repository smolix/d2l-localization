<!--en:6d566a4e-->
# Bereitstellung von Modellen

<!--en:d8564eaf-->
Die Sprachmodelle in diesem Buch haben Text in einem Notebook erzeugt, Prompt für Prompt, jeweils für die Person, die die Zelle ausgeführt hat. Nehmen wir nun an, hundert Personen aus Ihrem Kollegenkreis schicken Anfragen an dasselbe Sprachmodell, wann immer sie möchten. Beantwortet die GPU die Anfragen nacheinander und braucht dafür jeweils zwei Sekunden, wartet die letzte Person mehr als drei Minuten. Was muss sich ändern, wenn wir ein Modell als Dienst bereitstellen wollen?

<!--en:4b845d92-->
Bei der Bereitstellung als Dienst (engl. model serving) bearbeitet ein Beschleuniger viele Anfragen gleichzeitig. Um ein Token zu erzeugen, liest er die Gewichte und den Cache für Schlüssel und Werte (KV-Cache) aus dem Speicher (:numref:`sec_hardware_buyers`). Alle Anfragen nutzen dieselben Gewichte. Verarbeitet eine Inferenz-Engine viele Anfragen im selben Schritt, liest sie die Gewichte für alle zusammen nur einmal, sofern der Cache jeder aktiven Anfrage in den Speicher passt. Wir beginnen mit den Latenzen, die bei der Nutzung auffallen. Dann stellen wir lokale und serverseitige Inferenz-Engines mit Stand November 2026 vor und erklären, welche Verfahren sie schnell machen.

<!--en:8e52e143-->
## Lastprofile und Latenzmaße

<!--en:94a4c9ff-->
Wer einen Chatassistenten nutzt, bemerkt zwei Verzögerungen: Wie lange dauert es, bis das erste Wort erscheint, und wie schnell folgen die weiteren Wörter? Die erste Größe ist die **Zeit bis zum ersten Token** (engl. time to first token, TTFT), gemessen ab Eingang der Anfrage. Auf einem unbelasteten Server bestimmt vor allem der *Prefill* diese Zeit. Er verarbeitet alle Token des Prompts parallel und ist rechenbegrenzt. Die zweite Größe ist die **Zeit pro Ausgabetoken** (engl. time per output token, TPOT). Sie hängt vor allem von der *Dekodierung* ab, die pro Sequenz und Schritt ein Token erzeugt und durch die Speicherbandbreite begrenzt ist. Zwischen beiden Latenzen und dem Durchsatz, also der Gesamtzahl der Token pro Sekunde über alle Nutzer hinweg, besteht ein Zielkonflikt.

<!--en:776644ad-->
![Der Prefill verarbeitet den Prompt parallel und ist oft rechenintensiv. Die autoregressive Dekodierung erzeugt pro Sequenz jeweils ein Token und beansprucht vor allem die Speicherbandbreite.](../img/tools-prefill-decode.svg)

<!--en:bb6f6412-->
Welche Größen zählen, hängt vom Lastprofil ab. Danach richtet sich auch die Wahl des Werkzeugs:

<!--en:112f3221-->
:Lastprofile für Inferenzdienste und geeignete Werkzeuge für den Einstieg (Stand November 2026)

<!--en:8d76a1b1-->
| Lastprofil | Optimierungsziel | Einstieg |

<!--en:763f96de-->
| Persönlicher Assistent auf dem eigenen Rechner | einfache Bedienung, Datenschutz | Ollama / LM Studio (llama.cpp), auf Macs auch MLX |

<!--en:4f415ada-->
| Stapelweise Offline-Bewertung | Durchsatz pro Dollar | Offline-Modus von vLLM |

<!--en:10d53be5-->
| API für eine einzelne Anwendung | einfacher Betrieb | vLLM hinter einem kleinen Proxy mit Authentifizierung |

<!--en:72e8b470-->
| Dienst für mehrere Nutzer oder Agenten | Goodput bei vorgegebenem SLO | vLLM oder SGLang, bei großem Umfang zusätzlich NVIDIA Dynamo |

<!--en:1776ee70-->
| Maximaler Durchsatz mit einem festen Bestand an NVIDIA-GPUs | Durchsatz pro GPU | TensorRT-LLM |

<!--en:179436d5-->
Ein Service Level Objective (SLO) gibt ein Latenzziel vor, etwa eine TTFT unter 1 s. Im Produktivbetrieb zählt der *Goodput*, die Zahl der pro Zeiteinheit abgeschlossenen Anfragen, die ihr Ziel eingehalten haben. Bei Überlast kann der reine Durchsatz steigen, während der Goodput einbricht: Anfragen warten so lange in der Warteschlange, bis sie ihr Ziel verfehlen.

<!--en:f9145fc0-->
## Lokale Bereitstellung

<!--en:8aed1fd9-->
Nutzt eine einzelne Person einen Laptop oder eine Workstation, stellen sich zwei Fragen: Passt das Modell in den Speicher und wie schnell antwortet es? Viele lokale Inferenzwerkzeuge verwenden **llama.cpp** und dessen Dateiformat **GGUF**. In den Repositorys der Community finden sich häufig GGUF-Versionen von Modellen mit öffentlich zugänglichen Gewichten. Der Zugang zu llama.cpp erfolgt oft über eines der folgenden Werkzeuge:

<!--en:aff32ff0-->
* **Ollama** baut auf llama.cpp auf, auf Apple Silicon auch auf MLX. Der Befehl `ollama run qwen3:8b` lädt ein Modell herunter, speichert es lokal zwischen und stellt es über eine OpenAI-kompatible API unter `http://localhost:11434/v1` bereit.

<!--en:3137bd3e-->
* **LM Studio** bietet dieselben Inferenz-Engines über eine grafische Oberfläche an. Dort lässt sich einstellen, welcher Anteil des Modells auf die GPU ausgelagert wird.

<!--en:ea40b5fd-->
* **MLX** ist Apples Framework für Arrays. Mit dem Paket `mlx-lm` lassen sich Modelle auf Apple Silicon ausführen und feinabstimmen. Die Organisation `mlx-community` auf Hugging Face veröffentlicht Tausende konvertierte Modelle.

<!--en:7bed0d9f-->
Zu den gängigen GGUF-Quantisierungen zählen `Q4_K_M` (etwa 4.9 Bit pro Gewicht), `Q5_K_M` (etwa 5.7) und `Q8_0` (etwa 8.5). Für die 8.2 Milliarden Parameter von Qwen3-8B ergeben sich mit `Q4_K_M` insgesamt $8.2 \times 10^9 \times 4.9 / 8
\approx 5.0$ GB. Das liegt nahe an den 5.2 GB, die Ollama für `qwen3:8b` herunterlädt. Die 4.9 Bit wurden an Llama 3.1 8B gemessen. Bei `Q4_K_M` behält ein modellabhängiger Anteil der Tensoren eine höhere Genauigkeit. Der Cache für einen vollen Kontext von 32 768 Token in bf16 benötigt zusätzlich fast ebenso viel Speicher. Formate mit mehr Bit erhalten die Modellgüte meist besser. Messen Sie den Effekt für Ihre eigene Aufgabe. Die Dekodiergeschwindigkeit ist höchstens so groß wie die Speicherbandbreite geteilt durch die Anzahl der pro Token gelesenen Bytes. Jedes zusätzliche Bit kostet also Speicher und Geschwindigkeit. Eine gängige Faustregel lautet: Wählen Sie das größte Modell, dessen quantisierte Gewichte neben dem Cache in den Speicher passen. Wählen Sie dann das Format mit der höchsten Bitbreite, das noch hineinpasst.

<!--en:ba7a933e-->
## Inferenz-Engines für Server

<!--en:2dd63b8c-->
### Open-Source-Engines und Engines von NVIDIA

<!--en:518ac29a-->
Ein gemeinsam genutzter Server braucht eine Inferenz-Engine, die mehrere Anfragen zusammen einplant. Für einen quelloffenen GPU-Server bietet **vLLM** kontinuierliches Batching, seitenweise Verwaltung des KV-Caches, Präfix-Caching, Tensorparallelität und Unterstützung für quantisierte Modelle. Ein einziger Befehl startet den OpenAI-kompatiblen Server mit diesen Funktionen:

<!--en:4c2eaadf-->
**SGLang** kommt vLLM am nächsten. Das Verfahren *RadixAttention* :cite:`Zheng.Yin.Xie.ea.2024` speichert Schlüssel und Werte für alle Anfragen gemeinsam in einem Radixbaum. Das hilft, wenn viele Anfragen lange Präfixe gemeinsam haben, etwa in Chats mit demselben Systemprompt, beim Retrieval aus denselben Dokumenten oder bei baumartig verzweigten Agentenaufrufen. Beide Projekte entwickeln sich rasch weiter und unterstützen mehrere Backends für Beschleuniger. Messen Sie vor der Auswahl die Leistung beider Engines unter realistischen Bedingungen: Verwenden Sie die in Ihrer Anwendung beobachteten Verteilungen der Promptlänge und der Zahl gleichzeitiger Anfragen.

<!--en:469cf17d-->
Die Werkzeuge von NVIDIA nehmen für hardwarespezifische Optimierungen eine geringere Portabilität in Kauf. **TensorRT-LLM** führt Modelle auf PyTorch mit Kerneln aus, die NVIDIA optimiert hat. Das kann bei höherem Einrichtungsaufwand den Durchsatz im Dauerbetrieb steigern. **Dynamo** koordiniert viele GPUs und Rechenknoten mit vLLM, SGLang oder TensorRT-LLM als Backend. Prefill und Dekodierung laufen auf getrennten GPU-Gruppen. Dynamo verteilt die Anfragen danach, wie stark die einzelnen Worker ausgelastet sind und wie viel vom jeweiligen Präfix sie bereits im Cache halten. Diese Werkzeuge lohnen sich für ganze Rechenzentren. Bei wenigen GPUs ist vLLM oder SGLang allein einfacher.

<!--en:70576e59-->
### Eine gemeinsame Client-API

<!--en:31d9815b-->
Beim Wechsel vom Laptop auf einen Server sollte man die Anwendung nicht umschreiben müssen. Viele dieser Werkzeuge implementieren die Chat-Completions-API von OpenAI. Mit demselben Clientcode lässt sich daher eine lokale Ollama-Instanz, der eigene vLLM-Server oder ein kommerzieller Anbieter ansprechen. Dazu genügen eine andere Basis-URL und ein anderer Modellname:

<!--en:9bc909dd-->
Server, die dieselben Anfragen akzeptieren, können dennoch unterschiedlich antworten. Unterschiede gibt es bei der Tokenisierung, den Voreinstellungen für das Sampling, der Unterstützung strukturierter Ausgaben, dem Streaming-Verhalten und der Zählung der Token. Halten Sie einen kleinen Konformitätstest bereit und führen Sie ihn vor jedem Wechsel der Engine erneut aus.

<!--en:d7830a37-->
## Verfahren der Inferenz-Engines

<!--en:fd3ff5d4-->
Vier miteinander kombinierbare Verfahren erklären einen großen Teil des Geschwindigkeitsgewinns dieser Engines.

<!--en:a5e13069-->
**Kontinuierliches Batching.** Bei einem statischen Batch muss auch die langsamste Anfrage abgeschlossen sein, bevor der nächste Batch beginnen kann. Kontinuierliches Batching entfernt zwischen den Dekodierschritten fertige Anfragen und nimmt neue auf. So bleibt der Batch nahezu voll:

<!--en:62053be5-->
![Zwei Plätze im Batch bedienen Anfragen mit 5, 2, 7 und 3 Ausgabetoken wie im nachfolgenden einfachen Scheduler. Statisches Batching füllt jeden Batch auf die Länge der längsten Anfrage auf. Kontinuierliches Batching besetzt einen frei gewordenen Platz im nächsten Schritt neu und benötigt 9 statt 12 Schritte.](../img/tools-continuous-batching.svg)

<!--en:08e1a195-->
Der folgende einfache Scheduler zeigt das Prinzip in wenigen Zeilen. In Aufgabe :exref:`ex_model-serving-arrivals` kommen Ankunftszeiten und ein Speicherbudget hinzu:

<!--en:142f2586-->
**Seitenweise verwalteter KV-Cache.** Der KV-Cache jeder aktiven Sequenz wächst mit ihrer Länge. PagedAttention :cite:`Kwon.Li.Zhuang.ea.2023` in vLLM teilt dem Cache wie bei virtuellem Speicher Blöcke fester Größe zu. Damit verschwindet die Fragmentierung, die früher die Batchgröße begrenzte, fast vollständig. Für die Kapazitätsplanung heißt das: In den GPU-Speicher müssen die Gewichte *und zusätzlich* die Caches aller gleichzeitig aktiven Sequenzen passen. Reicht der Speicher für eine einzelne Anfrage, kann er bei vielen gleichzeitigen Anfragen dennoch zu klein sein.

<!--en:ce8ea209-->
**Präfix-Caching.** Haben zwei Anfragen dasselbe tokenisierte Präfix, etwa denselben Systemprompt oder dasselbe Dokument, genügt es, die Schlüssel und Werte des Präfixes einmal zu berechnen. Anschließend lassen sie sich aus dem Cache wiederverwenden:

<!--en:fd767b0c-->
![Ein Präfix-Cache verwendet den KV-Zustand nur dann wieder, wenn die tokenisierten Präfixe übereinstimmen. Anschließend wird für jede Anfrage ihr eigenes Suffix berechnet.](../img/tools-prefix-cache.svg)

<!--en:a9671e2b-->
Einige kommerzielle APIs berechnen für zwischengespeicherte Eingabetoken weniger. Die Regeln unterscheiden sich je nach Anbieter. Stehen gemeinsame, unveränderliche Inhalte vor dem variablen Teil eines Prompts, spart das in diesem Fall sowohl Geld als auch Prefill-Zeit.

<!--en:761e9843-->
**Spekulative Dekodierung und Quantisierung.** Ein kleines Entwurfsmodell schlägt mehrere Token vor. Das Zielmodell prüft sie in einem einzigen parallelen Durchlauf. Das beschleunigt die Dekodierung, wenn die meisten Vorschläge angenommen werden, und erhält die Zielverteilung. Durch Quantisierung muss die Engine bei der Dekodierung weniger Bytes lesen. Weniger Bytes pro Gewicht ergeben mehr Token pro Sekunde, bis bei den geringsten Bitbreiten die Güte leidet. Neuere Tensorkerne von NVIDIA unterstützen FP8 (ab den Generationen Hopper und Ada) und NVFP4 (ab Blackwell). vLLM und SGLang laden auch mit AWQ auf 4 Bit quantisierte Gewichte. Lokal erfüllen GGUF-Formate denselben Zweck.

<!--en:b10e1541-->
## Betrieb

<!--en:f5398fb9-->
Eine Inferenz-Engine, die auf einem Port Anfragen entgegennimmt, braucht für den gemeinsamen Betrieb noch weitere Vorkehrungen. Schalten Sie einen Proxy für Authentifizierung, TLS und Ratenbegrenzung vor die Engine. Halten Sie deren eigenen Port von außen unzugänglich. Die Zulassungssteuerung sollte Anfragen abweisen oder in eine Warteschlange einreihen, *bevor* der Speicher erschöpft ist. Bei einem Speicherfehler ist es dafür zu spät. Leiten Sie Abbrüche vom Client an die Engine weiter, damit aufgegebene Anfragen keine weitere Rechenleistung verbrauchen. Messen Sie für TTFT, TPOT und Ende-zu-Ende-Latenz das 50. und 95. Perzentil (p50 und p95) bei realistischen Ankunftsmustern. Mittelwerte aus einem geschlossenen Lasttest, bei dem jeder Client vor der nächsten Anfrage auf die Antwort wartet, reichen dafür nicht aus. Protokollieren Sie standardmäßig keine Prompts im Wortlaut. Schreiben Sie die Versionen von Modell, Tokenizer, Quantisierung und Engine gemeinsam fest. Ein schnellerer Server, der anders antwortet, ist ein anderes System. Machen Sie die Freigabe einer neuen Version daher sowohl von der Güte bei der jeweiligen Aufgabe als auch von der Latenz abhängig.

<!--en:e0a2b9d7-->
Sie können nun abschätzen, ob ein Modell und die Caches seiner Nutzer auf eine Karte passen und welche Inferenz-Engine Sie zuerst ausprobieren sollten. Offen bleibt, welche Anfragen ein Server unter Last zulassen und in welcher Reihenfolge er sie bearbeiten sollte. Das untersuchen Sie mit den Simulatoren in den Übungen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:bcb43b23-->
Die Latenz eines Inferenzdienstes hat zwei Komponenten. Der rechenbegrenzte Prefill bestimmt die Zeit bis zum ersten Token (TTFT), die durch die Speicherbandbreite begrenzte Dekodierung die Zeit pro Ausgabetoken (TPOT). Wählen Sie die Inferenz-Engine passend zum Lastprofil: lokal Ollama, LM Studio oder MLX, für gemeinsam genutzte Dienste vLLM oder SGLang und für große Bestände an NVIDIA-GPUs TensorRT-LLM mit Dynamo. Zur Quantisierung eignen sich lokal GGUF-Formate und auf Servern AWQ, FP8 oder NVFP4. Jedes bei den Gewichten eingesparte Byte beschleunigt unmittelbar die Tokenerzeugung, bis bei den geringsten Bitbreiten die Güte leidet. Moderne Engines kombinieren kontinuierliches Batching, seitenweise verwaltete KV-Caches, Präfix-Caching und spekulative Dekodierung. Wie viel vom Zustand eines Präfixes wiederverwendbar ist, hängt davon ab, wie der Client den Prompt anordnet. Mit der OpenAI-kompatiblen Chat-Completions-API lässt sich derselbe Clientcode für lokale Server und Cloudserver nutzen. Kleine Unterschiede im Verhalten rechtfertigen dennoch einen kurzen Konformitätstest vor dem Wechsel der Engine. Neben den Modellgewichten müssen die KV-Caches aller gleichzeitig aktiven Nutzer in den Speicher passen. Im Produktivbetrieb muss der Dienst die Zulassung steuern, Abbrüche vom Client weiterleiten und die hohen Perzentile der Latenz überwachen. Außerdem sind die Versionen von Modell und Engine gemeinsam festzuschreiben.

<!--en:2294243a-->
## Übungen

<!--en:6088e58b-->
1. **Grenzen der Batchgröße.** Der Befehl `vllm serve Qwen/Qwen3-8B` stellt ein Modell mit $L = 36$ Schichten und $H_\textrm{kv} = 8$ Schlüssel-Wert-Köpfen der Dimension $d_h = 128$ bereit. Seine bf16-Gewichte belegen $B_w = 16.4$ GB. Auch der Cache verwendet bf16. Stellen Sie das Modell auf einer H100 SXM bereit. Ihre Speicherbandbreite beträgt $\beta = 3.35$ TB/s, ihre Rechenleistung für dichte bf16-Operationen 989 TFLOP/s (:numref:`tab_gpu_specs`). Ein Dekodierschritt liest die Gewichte einmal für den gesamten Batch und den Cache jeder aktiven Sequenz. Für $b$ Sequenzen mit jeweils $n$ zwischengespeicherten Token beträgt die Dauer eines Schritts und damit die Zeit pro Ausgabetoken jeder Sequenz ungefähr

   $$t(b) = \frac{B_w + b\, n\, B_\textrm{kv}}{\beta},$$

   wobei $B_\textrm{kv}$ die Cachegröße pro Token bezeichnet.
    1. Jede Schicht speichert pro Schlüssel-Wert-Kopf und Token je einen Schlüssel- und einen Wertvektor. Daher gilt $B_\textrm{kv} = 2 \cdot L \cdot H_\textrm{kv} \cdot
       d_h \cdot (\textrm{bytes per element})$ (:numref:`sec_kv-cache`). Berechnen Sie $B_\textrm{kv}$. Bestimmen Sie die TPOT und den Durchsatz $b / t(b)$ bei $n = 2048$ für $b \in \{1, 8, 64, 256\}$. Welchem Wert nähert sich der Durchsatz, wenn $b$ wächst? Welcher Term von $t(b)$ bestimmt diesen Wert? Rechnen Sie bei den 8.2 Milliarden Parametern mit 2 FLOPs pro Parameter und Token. Prüfen Sie anhand der bf16-Rechenleistung, ob der Datentransfer den Schritt bei $b = 256$ weiterhin begrenzt.
    1. Ein interaktiver Dienst verspricht eine TPOT von höchstens 20 ms. Wie groß darf sein Batch höchstens sein? Welche Batchgröße wäre für eine Offline-Bewertung aus :numref:`tab_serving_workloads` sinnvoll und warum?
    1. Der Engine stehen auf der Karte 72 GB für Gewichte und Cache zur Verfügung. Die Sequenzen enthalten im Mittel 2048 Token. Vor der Einführung seitenweise verwalteter Caches reservierten Server für jede Anfrage einen zusammenhängenden Bereich für deren maximale Länge. Wie viele Sequenzen passen in den Speicher, wenn für jede die mit `--max-model-len` festgelegten 32 768 Token reserviert werden? Welcher Anteil des reservierten Caches enthält dann Tokenzustände? :citet:`Kwon.Li.Zhuang.ea.2023` maßen in solchen Systemen 20.4% bis 38.2%. Warum ist Ihr Anteil kleiner? Wie viele Sequenzen passen in den Speicher, wenn der Cache in Blöcken zu je 16 Token zugeteilt wird? Geben Sie eine obere Schranke für den Speicherplatz an, der in teilweise gefüllten Blöcken ungenutzt bleibt.
    1. Was begrenzt bei den beiden Zuteilungsverfahren jeweils die Batchgröße des interaktiven Dienstes: der Speicher oder das TPOT-Ziel?
   :exlabel:`ex_model-serving-batch-limits`

<!--en:20d06a0f-->
1. [code] **Konformitätstest.** :numref:`subsec_model-serving-contract` empfiehlt einen kleinen Konformitätstest vor dem Wechsel der Engine.
    1. Schreiben Sie eine Funktion `conformance(base_url, model)` mit dem `openai`-Client aus :numref:`subsec_model-serving-contract`. Sie soll vier Prüfungen ausführen und melden, welche davon bestanden wurden. Jede Prüfung verlangt außerdem, dass `content` nicht leer ist. Erstens müssen zwei identische Anfragen bei Softmax-Temperatur 0 denselben Text zurückgeben. Zweitens muss eine Anfrage mit `max_tokens=8` beim Abschluss für `finish_reason` den Wert `"length"` liefern und in `usage` höchstens 8 Ausgabetoken melden. Drittens soll eine Anfrage das Modell auffordern, von 1 bis 20 zu zählen. Mit `stop=["7"]` muss der zurückgegebene Text frei von der Ziffer 7 sein. Viertens soll eine Streaming-Anfrage bei Softmax-Temperatur 0 mit `stream_options={"include_usage": True}` einzelne Abschnitte liefern. Ihr zusammengesetzter Inhalt und die abschließende Angabe `usage` müssen mit denen derselben Anfrage ohne Streaming übereinstimmen. Führen Sie den Test mit Ollama und einem kleinen, auf einer CPU lauffähigen Modell wie `qwen3:0.6b` aus. Testen Sie auch einen zweiten OpenAI-kompatiblen Server Ihrer Wahl. Welche Prüfungen schlagen fehl und warum?
    1. Auf einem vLLM-Server gelingt die erste Prüfung, solange der Server sonst untätig ist. Sie schlägt gelegentlich fehl, wenn andere Clients ebenfalls Anfragen senden. Bei Softmax-Temperatur 0 findet kein Sampling statt. Erklären Sie den Fehlschlag.
    1. Beide Server laden dasselbe Modell mit demselben Tokenizer und erhalten dieselben Nachrichten. Dennoch unterscheiden sich die gemeldeten Werte von `usage.prompt_tokens` um einige Token. Nennen Sie zwei mögliche Ursachen. Warum spielt dieser Unterschied für einen Client auch unabhängig von der Abrechnung eine Rolle?

<!--en:269bd065-->
1. [code] **Präfix-Cache für einen Agenten.** Ein Programmieragent ruft ein Modell 30-mal auf. Der erste Prompt umfasst einen Systemprompt und Kontext aus einem Repository mit insgesamt 20 000 Token. Jeder weitere Prompt wiederholt den vorherigen und hängt 1000 Token an: die letzte Antwort des Modells und das Ergebnis eines Werkzeugaufrufs.
    1. Für wie viele Prompttoken muss die Engine bei den 30 Aufrufen den Prefill ohne Präfix-Caching berechnen? Wie viele sind es mit einem Cache, der alle früheren Prompts behält? Ein Anbieter berechnet für zwischengespeicherte Token ein Zehntel des üblichen Eingabepreises. Welcher Anteil der Eingabekosten ohne Cache bleibt übrig?
    1. vLLM speichert nur volle Blöcke. Der Schlüssel jedes Blocks ist ein Hash aus dem Schlüssel des vorherigen Blocks und den Token-IDs des Blocks selbst. Implementieren Sie dieses Verfahren für Blöcke von 16 Token als Funktion `prefill_tokens(cache, tokens)`. Sie soll die Zahl der zu berechnenden Token zurückgeben und die neuen vollen Blöcke zur Menge `cache` hinzufügen. Simulieren Sie den Agenten mit zufälligen Token-IDs und vergleichen Sie das Ergebnis mit Ihrer Rechnung aus der ersten Teilaufgabe. Warum geht der Schlüssel des vorherigen Blocks in den Schlüssel eines Blocks ein?
    1. Das Agentenframework schreibt die aktuelle Uhrzeit sekundengenau in die erste Zeile des Systemprompts. Wiederholen Sie die Simulation. Wo sollte der Zeitstempel stattdessen stehen?
    1. Eine Retrieval-Anwendung setzt hinter einen gemeinsamen Systemprompt drei von 50 Dokumenten, geordnet nach Relevanz. Zwei Anfragen rufen dieselben drei Dokumente in unterschiedlicher Reihenfolge ab. Welche Blöcke verwendet die zweite Anfrage wieder? Schlagen Sie eine Reihenfolge vor, die die Wiederverwendung erhöht. Welchen Vorteil geben Sie dafür auf?

<!--en:59bb69c7-->
1. [code] **Ankunftszeiten und Speicherbudget.** ● Erweitern Sie die Zelle, die `timeline` aufbaut, zu einem Simulator, dessen Zeit in Dekodierschritten fortschreitet. Anfrage $i$ trifft in Schritt $a_i$ ein und hat einen Prompt mit $P_i$ Token. Ab dem Schritt ihrer Zulassung erzeugt sie insgesamt $T_i$ Token, eines pro Schritt. Der Server verarbeitet höchstens 64 Sequenzen und fasst höchstens 32 768 Token im Cache. Eine Sequenz umfasst ihren Prompt und die bisher erzeugten Token. Erzeugen Sie die Ankünfte nach einem Poisson-Prozess mit $\lambda$ Anfragen pro Schritt. Ziehen Sie $P_i$ gleichverteilt zwischen 100 und 1000. Setzen Sie $T_i$ auf eins plus eine exponentialverteilte Zufallsvariable mit Mittelwert 200, runden Sie ab und begrenzen Sie das Ergebnis auf `max_tokens = 1024`. Messen Sie die Zeit bis zum ersten Token in Schritten von der Ankunft bis zum ersten erzeugten Token.
    1. Lassen Sie Anfragen in der Reihenfolge ihrer Ankunft zu, sobald der Cache den Prompt zuzüglich `max_tokens` aufnehmen kann. Mehr kann eine Anfrage nie benötigen. Weisen Sie eine eintreffende Anfrage ab, wenn bereits 50 Anfragen warten. Geben Sie für $\lambda \in \{0.05, 0.1, 0.15, 0.2\}$ die Zahl der pro Schritt erzeugten Token, die TTFT als p50 und p95 sowie den Anteil abgewiesener Anfragen an.
    1. Lassen Sie eine Anfrage nun bereits dann zu, wenn der Cache ihren Prompt und ein weiteres Token aufnehmen kann. Übersteigt der Bedarf der aktiven Sequenzen die Cachekapazität, unterbrechen Sie die zuletzt zugelassene Sequenz. Geben Sie ihren Cache frei und stellen Sie die Anfrage wieder an den Anfang der Warteschlange, sodass der Cache bei erneuter Zulassung neu berechnet wird. Wiederholen Sie die Messungen. Der Simulator setzt für den Prefill keine Zeit an. Welche Ressource beansprucht die Neuberechnung auf einem echten Server? Welche Ressource würde stattdessen die Auslagerung des Caches in den CPU-Speicher beanspruchen? Wie ändert sich der Vergleich mit der ersten Teilaufgabe, wenn Sie die Prefill-Zeit einrechnen?
    1. Ordnen Sie die Warteschlange nach der Anzahl der verbleibenden Token, die kürzeste Restlänge zuerst. Führen Sie beide Zulassungsregeln bei $\lambda = 0.2$ aus. Vergleichen Sie diese Reihenfolge mit der Ankunftsreihenfolge anhand des 95. Perzentils der TTFT: einmal für alle Anfragen und einmal für die 5% der Anfragen mit den längsten Ausgaben.
    1. Ein Server kennt `max_tokens`, aber nicht $T_i$. Wonach könnte er die Warteschlange stattdessen ordnen? Welcher Anteil des Gewinns aus der dritten Teilaufgabe bliebe Ihrer Einschätzung nach erhalten?
   :exlabel:`ex_model-serving-arrivals`

<!--en:da8f655b-->
1. **Spekulatives Sampling.** ● Bei der spekulativen Dekodierung nach :citet:`Leviathan.Kalman.Matias.2023` schlägt ein Entwurfsmodell mit der Verteilung $q$ für das nächste Token ein Token $x \sim q$ vor. Das Zielmodell mit der Verteilung $p$ für das nächste Token nimmt es mit Wahrscheinlichkeit $\min(1, p(x) / q(x))$ an. Nach einer Ablehnung wird das Token stattdessen aus der Restverteilung gezogen:

   $$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}.$$

    1. Zeigen Sie, dass das ausgegebene Token gemäß $p$ verteilt ist und die Annahmewahrscheinlichkeit $\beta = \sum_x \min(p(x), q(x))$ beträgt.
    1. In jeder Runde schlägt das Entwurfsmodell nacheinander $\gamma$ Token vor. Das Zielmodell bewertet sie alle in einem Durchlauf. Es gibt die angenommenen Token bis zur ersten Ablehnung aus, danach ein eigenes Token. Dieses zieht es nach einer Ablehnung aus $p'$ oder aus $p$, wenn es alle $\gamma$ Token angenommen hat. Sei $\alpha$ der Mittelwert der Annahmewahrscheinlichkeit $\beta$ aus der ersten Teilaufgabe über die Positionen einer Sequenz. Nehmen Sie an, dass jedes vorgeschlagene Token unabhängig mit Wahrscheinlichkeit $\alpha$ angenommen wird. Zeigen Sie, dass der Erwartungswert der Zahl der pro Runde ausgegebenen Token $(1 - \alpha^{\gamma + 1}) / (1 - \alpha)$ beträgt. Ein Schritt des Entwurfsmodells benötigt den Bruchteil $c$ der Zeit eines Zielmodellschritts. Der Bewertungsdurchlauf dauert einen Zielmodellschritt. Leiten Sie die erwartete Beschleunigung gegenüber der gewöhnlichen Dekodierung her. Bestimmen Sie das beste $\gamma$ für $c = 0.05$ und $\alpha \in \{0.6, 0.8, 0.9\}$.
    1. Nur wenn der Datentransfer den Bewertungsdurchlauf begrenzt, dauert dieser lediglich einen Zielmodellschritt. Ein Server dekodiert $b$ Sequenzen gemeinsam. Für jede Sequenz bewertet er $\gamma + 1$ Positionen pro Runde. Vernachlässigen Sie den Cache, speichern Sie die Gewichte in bf16 und rechnen Sie mit 2 FLOPs pro Parameter und Token. Ab welcher Batchgröße $b$ wird der Durchlauf auf der H100 aus Aufgabe :exref:`ex_model-serving-batch-limits` bei $\gamma = 4$ rechenbegrenzt? Was folgt daraus für die spekulative Dekodierung auf einem ausgelasteten Server im Vergleich zur Nutzung durch eine einzelne Person?

<!--en:5335d342-->
1. [extended] **Goodput unter Last.** ●● Bauen Sie Ihren Simulator aus Aufgabe :exref:`ex_model-serving-arrivals` so um, dass seine Uhr in Sekunden statt in Dekodierschritten läuft. Betrachten Sie Qwen3-8B auf der H100 aus Aufgabe :exref:`ex_model-serving-batch-limits` mit 72 GB für Gewichte und einen seitenweise verwalteten Cache. Ein Schritt verarbeitet $p$ Prompttoken sowie je ein Token für jede der $b$ zu dekodierenden Sequenzen, die zusammen $K$ Token im Cache halten. Bei $N$ Parametern und einer Rechenleistung $F$ für dichte bf16-Operationen ergibt sich mit der Notation aus Aufgabe :exref:`ex_model-serving-batch-limits` die Schrittdauer

   $$t_\textrm{step} = \max\left(\frac{2N(p + b)}{F}, \frac{B_w + K B_\textrm{kv}}{\beta}\right),$$

   Dabei bleiben die FLOPs für die Attention-Scores unberücksichtigt. Anfragen treffen gemäß einem Poisson-Prozess mit $\lambda$ Anfragen pro Sekunde ein. Die Promptlängen sind zwischen 500 und 4000 Token gleichverteilt. Die Ausgabelänge ergibt sich aus eins plus einer exponentialverteilten Zufallsvariable mit Mittelwert 250. Runden Sie ab und begrenzen Sie die Länge auf 2048. Eine Anfrage zählt zum Goodput, wenn ihre TTFT höchstens 1 s und ihre TPOT höchstens 50 ms beträgt. Die TPOT ist hier die mittlere Zeit zwischen ihren Ausgabetoken.
    1. Verarbeiten Sie jeden zugelassenen Prompt beim Prefill vollständig in einem Schritt. Schätzen Sie vor der Simulation anhand von $t_\textrm{step}$ die Sättigungsrate. Variieren Sie dann $\lambda$ von schwacher Last bis weit über die Sättigung hinaus. Stellen Sie Durchsatz und Goodput jeweils in Anfragen pro Sekunde grafisch dar. Bei welcher Rate erreicht der Goodput sein Maximum? Wie verhält sich diese Rate zu Ihrer Schätzung?
    1. Ein vollständiger Prefill für einen langen Prompt verzögert das nächste Token jeder gerade dekodierten Sequenz. Implementieren Sie den blockweisen Prefill :cite:`Agrawal.Kedia.Panwar.ea.2024`. Er begrenzt die Gesamtzahl der Token pro Schritt einschließlich der Dekodiertoken auf ein festes Budget und verteilt einen Prompt auf mehrere Schritte. Wiederholen Sie die Versuchsreihe für Budgets von 512, 2048 und 8192 Token. Welches Budget maximiert den Goodput? Wie hängt das Ergebnis von den beiden SLOs ab?
    1. Ergänzen Sie eine Zulassungssteuerung, die eine Anfrage abweist, sobald ihr TTFT-Ziel nicht mehr zu erreichen ist. Wie ändert sich dadurch der Goodput jenseits der Sättigung? Erklären Sie, warum der Goodput ohne diese Steuerung einbricht, während der Durchsatz hoch bleibt.
    1. Lassen Sie jeden Client seine Anfrage nach 10 s aufgeben und einmal erneut senden. Vergleichen Sie den Goodput für zwei Fälle: Der Server erzeugt für aufgegebene Anfragen weiterhin Token oder er bricht die Verarbeitung ab.
    1. Geben Sie die höchste Anfragerate an, bei der mindestens 90% aller eintreffenden Anfragen beide SLOs einhalten. Zählen Sie abgewiesene Anfragen als Zielverfehlungen und nennen Sie die Konfiguration, die diese Rate erreicht.

<!--en:75a07f82-->
[Diskussion](https://d2l.discourse.group/t/36847)

<!--en:fbd44802-->
Bereitstellung von Modellen: Prefill und Dekodierung, kontinuierliches Batching, PagedAttention und SLOs<br>**TTFT und TPOT · Goodput · GGUF und AWQ/FP8 · vLLM und SGLang · kontinuierliches Batching · PagedAttention · Präfix-Caching · spekulative Dekodierung**

<!--en:925423d1-->
::: {.slide title="Vom Notebook für eine Person zum Dienst für viele"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:bbb39ad7-->
- **Warum serielle Verarbeitung an Grenzen stößt:** Bei 100 Anfragen von je zwei Sekunden wartet die 100. Person **über 3 Minuten**. Die GPU liest die Gewichte 100-mal aus dem Speicher (ALU zu $98\%$ untätig).

<!--en:f16e4aae-->
- **Was dynamisches Batching bei der Dekodierung bewirkt:** Bei $b$ gleichzeitigen Nutzern liest die GPU die $16.4\text{ GB}$ Modellgewichte **einmal pro Schritt** für alle $b$ Nutzer zusammen. Der Durchsatz steigt um den Faktor $30\text{--}50\times$.

<!--en:ce006152-->
::: {.slide title="Zwei Phasen der Generierung: Prefill und Dekodierung"}

<!--en:0a9c33e3-->
[Aufbau der Arbeitslast]{.kicker}

<!--en:5f8df57e-->
Jede Anfrage zur Textgenerierung durchläuft zwei Phasen mit unterschiedlichen Hardwareanforderungen. Beiden entspricht jeweils eine spürbare Latenz:

<!--en:0bd5a4e7-->
- **1. Prefill des Prompts $\to$ Zeit bis zum ersten Token (TTFT):**
  - Verarbeitet alle $p$ Token des Prompts parallel und füllt damit erstmals den KV-Cache.
  - Hohe arithmetische Intensität (**rechenbegrenzt**). Bestimmt die Wartezeit bis zum ersten Wort.

<!--en:79e9d361-->
- **2. Autoregressive Dekodierung $\to$ Zeit pro Ausgabetoken (TPOT):**
  - Erzeugt pro aktiver Sequenz und Schritt ein neues Token.
  - Geringe arithmetische Intensität (**durch die Speicherbandbreite begrenzt**). Bestimmt das Lesetempo beim Streaming ($\text{tok/s} = 1/\text{TPOT}$).

<!--en:4e0e602c-->
::: {.slide title="Latenzziele, reiner Durchsatz und Goodput"}

<!--en:bb40f9ca-->
[Kennzahlen im Produktivbetrieb]{.kicker}

<!--en:cc5cde65-->
- **Service Level Objectives (SLOs):** Begrenzen die Zeit bis zum ersten Token (z. B. $\text{TTFT} \le 1.0\text{ s}$) und die Zeit pro Ausgabetoken ($\text{TPOT} \le 50\text{ ms}$, also $\ge 20\text{ tok/s}$ pro Nutzer).

<!--en:b3ee6d35-->
- **Goodput und reiner Durchsatz:** Bei Überlast wächst die Warteschlange rapide. Der **Goodput (SLO-konforme Anfragen/s) fällt gegen null**, auch wenn der reine Tokendurchsatz nahe $100\%$ des Spitzenwerts bleibt.

<!--en:f211de96-->
::: {.slide title="Inferenzsysteme nach Lastprofil (November 2026)"}

<!--en:ec4fd786-->
[Inferenz-Engines im Überblick]{.kicker}

<!--en:8221a62b-->
- **Lokaler Betrieb und Offline-Batching:** **Ollama / LM Studio** (`llama.cpp` mit GGUF) oder **MLX** schützen bei lokaler Einzelnutzung die Daten. Der **Offline-Modus von vLLM** maximiert bei der Bewertung von Datensätzen die Zahl der Token pro Dollar.

<!--en:ef593be1-->
- **Online-APIs und GPU-Bestände im Rechenzentrum:** **vLLM** oder **SGLang** (Präfixbaum von RadixAttention) eignen sich für Mehrbenutzerdienste mit geringer Latenz, **NVIDIA Dynamo + TensorRT-LLM** für die Verteilung der Inferenzphasen auf mehrere Knoten.

<!--en:32def55e-->
::: {.slide title="Lokale Inferenz-Engines und GGUF-Quantisierung"}

<!--en:faa801e1-->
[Inferenz für einzelne Nutzer]{.kicker}

<!--en:472fe4f7-->
- **Speicherbedarf von Qwen3-8B ($8.2\text{B}$) in GGUF:** `Q4_K_M` ($\sim 4.9\text{ bits/w}$) benötigt $\sim 5.2\text{ GB}$, `Q5_K_M` benötigt $\sim 5.8\text{ GB}$ und `Q8_0` benötigt $\sim 8.7\text{ GB}$.

<!--en:73b8563b-->
- **Kapazität, Geschwindigkeit und KV-Cache:** Mit 4-Bit-Gewichten läuft die Dekodierung bei Batchgröße 1 fast doppelt so schnell wie mit 8-Bit-Gewichten. Auf einer Karte mit 16 GB bleibt Platz für einen BF16-KV-Cache für 32K Token ($\sim 4.8\text{ GB}$).

<!--en:1ee27c3c-->
::: {.slide title="Rechenmodell für die Dekodierung im Batch und die TPOT-Grenze"}

<!--en:240528a6-->
[Roofline bei mehreren Nutzern]{.kicker}

<!--en:9ba668f9-->
- **Qwen3-8B (BF16) auf einer H100 ($\beta = 3.35\text{ TB/s}$, $n = 2{,}048$):** Bei größerem Batch ($b$: $1 \to 64$) wächst die TPOT nur um den Faktor $2.1\times$ ($5.0 \to 10.7\text{ ms}$), der Durchsatz hingegen um den Faktor **$30\times$** ($201 \to 6{,}000\text{ tok/s}$).

<!--en:fb596ed5-->
- **Grenzen durch SLO und KV-Cache:** Ein TPOT-Ziel von $20\text{ ms}$ begrenzt die Batchgröße auf $b_{\max} = 167$ ($8{,}350\text{ tok/s}$). Damit nähert sich der Durchsatz der durch den KV-Cache gesetzten Grenze $\beta/(n B_{\text{kv}}) \approx 11{,}090\text{ tok/s}$.

<!--en:3e94836c-->
::: {.slide title="Inferenz-Engines für Server: vLLM, SGLang, TensorRT-LLM, Dynamo"}

<!--en:1951c28f-->
[Architekturen für mehrere Nutzer]{.kicker}

<!--en:b3d9f40a-->
- **vLLM und SGLang:** vLLM bietet standardmäßig PagedAttention, kontinuierliches Batching und eine OpenAI-kompatible Bereitstellung (`vllm serve`). **RadixAttention** in SGLang speichert KV-Caches für Agenten mit mehreren Dialogrunden in einem Radixbaum.

<!--en:b040a898-->
- **TensorRT-LLM und NVIDIA Dynamo:** TensorRT-LLM kompiliert hardwareoptimierte Kernel für FP8/NVFP4. Dynamo verteilt **Prefill und Dekodierung auf getrennte GPU-Gruppen** und berücksichtigt beim Weiterleiten der Anfragen deren Präfixe.

<!--en:56561445-->
::: {.slide title="Schnittstellenvertrag der OpenAI-API und Konformitätstests"}

<!--en:64d679f3-->
[Portabler Clientcode]{.kicker}

<!--en:8c938eaf-->
Lokale und serverseitige Engines implementieren das Chat-Completions-Schema von OpenAI. Für den Wechsel von Ollama auf dem Laptop zu einem vLLM-Cluster in der Cloud muss die Clientanwendung daher nur `base_url` und `model` ändern:

<!--en:2297e73e-->
- **Warum sich Server trotz identischer API-Schemata unterschiedlich verhalten:**
  - Unterschiedliche Leerraumzeichen in Jinja-Chatvorlagen und unterschiedliche Token für den Sequenzanfang verändern `usage.prompt_tokens` und verbrauchen unterschiedlich viel vom Kontextbudget.
  - Je nach Server unterscheiden sich die Übertragung der Nutzungsdaten beim Streaming (`stream_options`), das Abschneiden bei `stop`-Zeichenketten und die Grammatikverarbeitung für strukturierte Ausgaben.
  - **Nichtdeterminismus durch die Batchgröße:** Selbst bei `temperature=0` fasst ein ausgelasteter vLLM-Server Anfragen zu Batches unterschiedlicher Form zusammen. Dadurch ändert sich die Reihenfolge der Gleitkommareduktionen so weit, dass bei fast gleichen Werten ein anderes Token das Maximum erreichen kann.

<!--en:651358c8-->
::: {.slide title="Kontinuierliches Batching vermeidet Leerlauf durch statisches Padding"}

<!--en:315be2e8-->
[Verfahren 1]{.kicker}

<!--en:beca1164-->
Beim statischen Batching bleiben $b$ Anfragen zusammen, bis die längste Ausgabe fertig ist. Enden kürzere Antworten früher, bleiben GPU-Plätze ungenutzt:

<!--en:e2374378-->
- **Planung pro Iteration (kontinuierliches Batching):**
  - Am Ende *jedes einzelnen Dekodierschritts* prüft die Engine, welche Sequenzen `<eos>` ausgegeben oder `max_tokens` erreicht haben.
  - Fertige Sequenzen scheiden sofort aus und geben ihre KV-Cache-Blöcke frei. Schon im nächsten Schritt rücken wartende Anfragen in den Batch nach.

<!--en:f04035e4-->
- **Ergebnis für Anfragen mit `[5, 2, 7, 3]` Ausgabetoken auf 2 Plätzen:** Kontinuierliches Batching schließt alle vier Anfragen in **9 statt 12 Dekodierschritten** ab ($25\%$ weniger Schritte, Auslastung der Plätze von $71\%$ auf $94\%$ gestiegen).

<!--en:9c345563-->
::: {.slide title="Simulation eines Schedulers für kontinuierliches Batching"}

<!--en:86fa69b7-->
[Ausführbarer Scheduler]{.kicker}

<!--en:956d53cb-->
Zehn Zeilen Python bilden die wesentlichen Zustandsübergänge einer Schleife für kontinuierliches Batching ab, die Anfragen in jeder Iteration neu einplant:

<!--en:24d8be3d-->
- **Neubelegung der Plätze Schritt für Schritt:**
  - Schritt 1 beginnt mit `(5, 2)`. Nach 2 Schritten ist die Anfrage mit 2 Token fertig (übrig bleibt `(3,)`). Die Anfrage mit 7 Token übernimmt sofort Platz 2 (`(3, 7)`).
  - In Schritt 5 endet die erste Anfrage. Die letzte Anfrage mit 3 Token kommt zur Anfrage mit ursprünglich 7 Token hinzu, von denen noch 4 fehlen (`(4, 3)`). Nur im letzten Schritt ist ein einziger Platz belegt.

<!--en:0a78e147-->
::: {.slide title="PagedAttention und Kapazitätsplanung für den KV-Cache"}

<!--en:993f397d-->
[Verfahren 2]{.kicker}

<!--en:e64996d6-->
- **Speicherverschwendung bei zusammenhängender Reservierung (`max_len=32,768`):** Pro Sequenz werden $4.83\text{ GB/seq}$ von $55.6\text{ GB}$ freiem Grafikspeicher der H100 reserviert. So finden nur **$b = 11$ Sequenzen** Platz. Dabei bleiben $93.7\%$ des reservierten KV-Speichers ungenutzt, wenn $n = 2{,}048$ gilt.

<!--en:065fdfad-->
- **PagedAttention (virtuelle Blöcke mit je 16 Token):** Reserviert nach Bedarf Blöcke von $2.36\text{ MB}$ (interner Verschnitt $< 0.7\%$). Die Kapazität für gleichzeitige Sequenzen wächst **um den Faktor $16.7\times$ auf $b = 184$ Sequenzen**. Damit greift zuerst das TPOT-Ziel von $20\text{ ms}$ ($b = 167$).

<!--en:ae76088e-->
::: {.slide title="Präfix-Caching und Promptgestaltung für mehr Cachetreffer"}

<!--en:8cc6044e-->
[Verfahren 3]{.kicker}

<!--en:724274b8-->
Beginnen Anfragen mit derselben Tokenfolge (Systemanweisungen, Few-Shot-Beispiele oder Kontext aus einem Repository), verwendet das **Präfix-Caching** die vorberechneten KV-Blöcke ohne erneuten Prefill:

<!--en:d13f2600-->
- **Verkettete Block-Hashes:** Jeder Block mit 16 Token erhält den Schlüssel $\text{hash}(\text{parent\_block\_key}, \text{token\_ids})$, denn bei kausaler Self-Attention hängt die Repräsentation jedes Tokens von allen vorhergehenden Token ab.

<!--en:adcaf5c9-->
- **Programmieragent über 30 Runden ($20\text{K}$ Ausgangskontext + $1\text{K}$ pro Runde):**
  - Ohne Cache verarbeitet der Prefill $\sum_{k=0}^{29} (20 + k)\text{K} = \mathbf{1{,}035{,}000\text{ tokens}}$.
  - Mit Präfix-Caching sind es nur $20\text{K} + 29 \times 1\text{K} = \mathbf{49{,}000\text{ tokens}}$ (**um den Faktor $21\times$ geringerer Prefill-Aufwand**).

<!--en:ec27b42a-->
- **Regel für die Promptgestaltung:** Setzen Sie keinen sekundengenauen Zeitstempel in Zeile 1 eines Systemprompts. Er macht die Hashkette ab Block 0 ungültig. Veränderliche Metadaten gehören ans *Ende* des Prompts.

<!--en:a89d9df7-->
::: {.slide title="Spekulative Dekodierung und Quantisierung auf dem Server"}

<!--en:8f14c006-->
[Verfahren 4]{.kicker}

<!--en:99ef6cb8-->
- **Entwurf und parallele Prüfung:** Ein kleines Entwurfsmodell $q$ (Kosten $c \ll 1$) schlägt $\gamma$ Token vor. Ein Durchlauf des Zielmodells $p$ wertet alle $\gamma+1$ Positionen parallel aus und erhält so die Zielverteilung exakt (Beschleunigung um den Faktor $2.80\times$ bei $\alpha=0.8, \gamma=4, c=0.05$).

<!--en:82cb739c-->
- **Einschränkung bei ausgelasteten Servern:** Die parallele Prüfung kostet nur bei speicherbegrenzter Ausführung keine zusätzliche Zeit. Auf der H100 ($I^* \approx 295$) ist jeder Batch mit $b > 295/(\gamma+1) = 59$ rechenbegrenzt.

<!--en:cd977368-->
::: {.slide title="Blockweiser Prefill, Zulassungssteuerung und Betrieb"}

<!--en:804dd38b-->
[Zuverlässigkeit im Produktivbetrieb]{.kicker}

<!--en:b1542776-->
- **Blockweiser Prefill:** Verteilt lange 16K-Prompts auf Schritte mit festem Tokenbudget (z. B. $2{,}048$ Token/Schritt). Im selben Schritt werden auch die Token der aktiven Dekodierung verarbeitet. So verursachen Prefill-Lastspitzen keine TPOT-Spitzen.

<!--en:b4716b67-->
- **Zulassung, Abbruch und gemeinsame Versionsbindung:** Weisen Sie Anfragen frühzeitig ab, wenn die Warteschlange für das TTFT-Ziel zu lang ist. Brechen Sie unterbrochene HTTP-Streams ab. Schalten Sie einen Proxy für TLS, Authentifizierung und Ratenbegrenzung vor und schreiben Sie `(model_sha, tokenizer, quant, engine)` gemeinsam fest.

<!--en:d16e92a8-->
::: {.slide title="Verständnisfragen zu KV-Cache, Batching und Präfixwiederverwendung"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:a2fc0fd0-->
Prüfen Sie anhand der drei Szenarien, wie gut Sie Inferenzsysteme durchrechnen können:

<!--en:9a741ecd-->
1. Bei einem Dienst für retrievalgestützte Generierung (RAG) folgen auf einen gemeinsamen Systemprompt mit 2000 Token je 3 abgerufene Dokumente aus einem Korpus von 50 Dokumenten. Anfrage 1 ruft `[Doc A, Doc B, Doc C]` ab, Anfrage 2 dagegen `[Doc B, Doc A, Doc C]`. Welche KV-Blöcke verwendet Anfrage 2 beim Präfix-Caching mit Block-Hashes wieder? Wie lässt sich die Zahl der Cachetreffer erhöhen?

<!--en:040a11d1-->
2. Unsere H100 stellt Qwen3-8B bereit ($n = 2{,}048$ Token mittlere Kontextlänge, $72\text{ GB}$ Speicherbudget). Der Dienst verlangt $\text{TPOT} \le 20\text{ ms}$ ($b_{\max,\text{SLO}} = 167$). Welche Grenze bestimmt die Batchgröße bei zusammenhängender Reservierung für `max_model_len=32768` ($b_{\max,\text{mem}} = 11$), welche bei PagedAttention ($b_{\max,\text{mem}} = 184$)?

<!--en:557b98b6-->
**Antworten:**

<!--en:e857af46-->
1. Anfrage 2 verwendet **nur den Systemprompt mit 2000 Token** wieder. Da an Position 2001 `Doc B` anstelle von `Doc A` beginnt, ändern sich ab Token 2001 alle Block-Hashes. Eine **kanonische Reihenfolge nach Dokument-ID** (`[Doc A, Doc B, Doc C]`) ermöglicht die vollständige Wiederverwendung des gemeinsamen Präfixes, wenn die Teilmengen mit denselben Dokumenten beginnen. Dafür stehen die relevantesten Dokumente nicht mehr bevorzugt am Anfang.

<!--en:609faaad-->
2. Bei **zusammenhängender Reservierung** begrenzt zuerst **die Speicherkapazität** die Batchgröße ($11 \ll 167$, die GPU bleibt unterausgelastet). Bei **PagedAttention** fasst der Speicher $184$ Sequenzen. Somit greift zuerst das **Latenzziel für die TPOT von $20\text{ ms}$**, und zwar bei $b = 167$.

<!--en:a2cd1cc1-->
::: {.slide title="Verständnisfragen zu KV-Cache, Batching und Präfixwiederverwendung (Fortsetzung)"}

<!--en:929b7671-2-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:4b0228ef-->
3. Welche Beschleunigung gegenüber gewöhnlicher Dekodierung mit Batchgröße 1 ist bei spekulativer Dekodierung mit Entwurfskosten $c = 0.05$, Annahmerate $\alpha = 0.8$ und $\gamma = 4$ Entwurfstoken zu erwarten?

<!--en:23db34b8-->
**Antwort:**

<!--en:19f1302a-->
3. Erwartete Anzahl an Token pro Runde $= (1 - 0.8^5) / (1 - 0.8)$ $= 0.67232 / 0.2 = \mathbf{3.362\text{ tokens}}$ bei einem Zeitbedarf von $1 + 4(0.05) = \mathbf{1.20\text{ target steps}}$. Das ergibt eine **Beschleunigung um den Faktor $\mathbf{2.80\times}$**.

<!--en:40b7a0d2-->
::: {.slide title="Zusammenfassung: Systeme zur Bereitstellung von Modellen"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:9096ab99-->
| Komponente | Physikalisches oder algorithmisches Prinzip | Regel für den Produktivbetrieb |

<!--en:986a80e3-->
| **Prefill und Dekodierung** | Der parallele, rechenbegrenzte Prefill bestimmt die TTFT. Die serielle, bandbreitenbegrenzte Dekodierung bestimmt die TPOT. | Prefill mit Tensorkernen und Präfix-Caching optimieren, Dekodierung mit Batching und Quantisierung |

<!--en:6c5590a7-->
| **Skalierung der Dekodierung im Batch** | $t(b) = (B_w + b \, n \, B_{\text{kv}}) / \beta$ | Batching verteilt die Kosten für das Lesen der Gewichte $B_w$ auf mehrere Anfragen, bis der Bandbreitenbedarf des KV-Caches $b \, n \, B_{\text{kv}}$ oder das TPOT-Ziel $b$ begrenzt |

<!--en:09713019-->
| **Kontinuierliches Batching** | Entfernt in jedem Dekodierschritt fertige Sequenzen und besetzt freie Plätze neu | Vermeidet Leerlauf durch statisches Padding und verbessert die Auslastung der GPU-Plätze |

<!--en:e99d33b4-->
| **PagedAttention** | Weist dem KV-Cache nach Bedarf virtuelle Blöcke von je 16 Token zu | Vermeidet den bei zusammenhängender Reservierung verschwendeten Speicheranteil von $\sim 90\%+$ und erhöht die Kapazität für gleichzeitige Sequenzen um einen Faktor $> 10\times$ |

<!--en:0f1b10f1-->
| **Präfix-Caching** | Verwendet KV-Blöcke bei übereinstimmenden verketteten Block-Hashes wieder | Statische Systemprompts und kanonisch geordneten Kontext *vor* veränderliche Zeitstempel oder Nutzeranfragen setzen |

<!--en:f5d4a93f-->
| **Goodput und Zulassung** | Bei Überlast wächst die Warteschlange. Der Goodput bricht ein, während der reine Durchsatz hoch bleibt. | Blockweisen Prefill nutzen, frühzeitig anhand der TTFT über die Zulassung entscheiden und Abbrüche vom Client weiterleiten |
