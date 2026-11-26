<!--en:f5b9081b-->
# Berechnungen

<!--en:d5436961-->
Bisher haben wir den Frameworks überlassen, wie sie Modelle aufbauen, Parameter verwalten, Gewichte initialisieren und Berechnungen auf der Hardware ausführen. Der Schritt vom Lehrbuchprototyp zum großen Produktionssystem erfordert einen Blick ins Innere. Betrachten wir ein Sprachmodell mit 8 Milliarden Parametern. Beim Training mit Adam in 32-Bit-Gleitkommaarithmetik belegen Parameter, Gradienten und Optimiererzustand bereits 128 GB, noch bevor wir eine einzige Aktivierung speichern. Das Training läuft auf Tausenden von GPUs, und alle paar Stunden unterbricht ein Hardwareausfall den Lauf. Bei dieser Größenordnung müssen wir verstehen, wie Frameworks Modelle darstellen, initialisieren, für Berechnungen nutzen und dauerhaft speichern.

<!--en:6ddf6de1-->
Das gesamte Kapitel baut auf einer Datenstruktur auf. Ein neuronales Netz ist ein *Baum aus Modulen* (engl. tree of modules). In den Blättern liegt sein *Zustand* (engl. state). Dazu gehören trainierbare Parameter ebenso wie feste Puffer (engl. buffers). Jeder Parameter und jeder Puffer hat einen eindeutigen Pfad von der Wurzel aus. In der ersten Kapitelhälfte entwickeln wir diese Abstraktion und zeigen, wie Frameworks den Baum bei ihren Operationen durchlaufen. Wir beginnen mit Modulhierarchien, der Registrierung in Containern, Skip-Verbindungen in Residualblöcken, der verzögerten Bestimmung von Tensorformen und dem Aufbau von Modellen anhand einer Konfiguration (:numref:`sec_model_construction`). Dann betrachten wir den Zustandsbaum genauer und unterscheiden trainierbare Parameter von nicht trainierbaren Puffern. Wir leiten her, warum Adam beim Training in fp32 mindestens 16 Byte pro Parameter benötigt, zählen Parameter, ohne Speicher zu reservieren, und nutzen Gewichte gemeinsam oder frieren sie ein (:numref:`sec_parameters`). Von Varianzregeln für einzelne Schichten gehen wir zur Initialisierung ganzer Modulbäume und ihrer Residualzweige über. Wir zeigen, warum die Varianz der Aktivierungen bei unskalierten Residualzweigen um den Faktor $(1 + g^2)^N$ wächst. Mit dem Faktor $1/\sqrt{N}$ oder einer auf null gesetzten Ausgabeprojektion lassen sich tiefe Netze stabilisieren (:numref:`sec_init_param`). Anschließend erweitern wir das Framework um zustandslose Schichten, Schichten mit Parametern (`RMSNorm`) und Schichten mit festen Puffern (`RandomFeatures`). Mit dem Straight-Through-Schätzer definieren wir eigene Regeln für den Rückwärtsdurchlauf (:numref:`sec_custom_layer`).

<!--en:9959aaf9-->
In der zweiten Kapitelhälfte wenden wir uns der Zahlendarstellung, dem dauerhaften Speichern, der Ausführung auf der Hardware und der Diagnose zu. Wir vergleichen fp32, fp16 und bf16 und erklären, warum kleine Updates bei 16-Bit-Gewichten durch Rundung verloren gehen. Beim Training mit gemischter Genauigkeit (engl. mixed precision) bleiben die Hauptgewichte (engl. master weights) in fp32. Die Berechnungen laufen mit 16 Bit, bei fp16 kommt Loss-Scaling hinzu (:numref:`sec_numerics`). Anschließend vergleichen wir unsichere `pickle`-Archive mit `safetensors`-Dateien ohne ausführbaren Code. Wir erstellen atomare, fünfteilige Checkpoints, mit denen sich das Training bitgenau fortsetzen lässt. Beim Laden vortrainierter Basisnetze prüfen wir, welche Schlüssel fehlen oder unerwartet hinzukommen (:numref:`sec_read_write`). Schließlich übertragen wir Tensoren und Modelle auf GPUs und messen die Ausführungszeit von Befehlen in asynchronen Warteschlangen präzise. Wir untersuchen die GPU-Speicherbelegung und das Aktivierungs-Checkpointing mit einem Speicherbedarf von $O(\sqrt{K})$ (:numref:`sec_use_gpu`). Für reproduzierbare Läufe legen wir die Startwerte der Pseudozufallsgeneratoren fest und sorgen für deterministische GPU-Berechnungen. Dabei beobachten wir den Vorwärts- und Rückwärtsdurchlauf mit Hooks, ohne in die Berechnung einzugreifen (:numref:`sec_repro`).

<!--en:5dd576ec-->
## Materialien und weiterführende Literatur {.unnumbered}

<!--en:a64691d2-->
Die folgenden Materialien ergänzen die einzelnen Abschnitte dieses Kapitels. Bei jedem Eintrag ist der zugehörige Abschnitt angegeben. Soweit nicht anders vermerkt, sind alle Materialien online frei zugänglich.

<!--en:f3e47eb0-->
**Bücher**

<!--en:d3974bde-->
- [Deep Learning mit PyTorch (Stevens, Antiga und Viehmann)](https://web.archive.org/web/20211012030609/https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf): Freies PDF des PyTorch-Teams (archivierte Fassung). Teil 1 behandelt Tensoren, Speicher, automatisches Differenzieren und `nn.Module` in derselben Tiefe wie :numref:`sec_model_construction` und :numref:`sec_parameters`.

<!--en:d790fd9e-->
- [Systeme für maschinelles Lernen (Vijay Janapa Reddi)](https://mlsysbook.ai/): Das frei zugängliche Onlinebuch ordnet dieses Kapitel aus Systemsicht ein. Es behandelt Frameworks, Datenpipelines, Trainingsinfrastruktur und die effiziente Bereitstellung von Modellen (:numref:`sec_use_gpu`, :numref:`sec_repro`).

<!--en:51311a3d-->
**Kurse und Videovorlesungen**

<!--en:c7ced7f4-->
- [CMU 10-414/714: Deep-Learning-Systeme (Chen und Kolter)](https://dlsyscourse.org/): In den frei zugänglichen Vorlesungen und Aufgaben entwickeln Studierende „needle“, ein kleines Framework mit automatischem Differenzieren, Modulen, Initialisierung und GPU-Unterstützung (:numref:`sec_model_construction`, :numref:`sec_init_param`, :numref:`sec_use_gpu`).

<!--en:7dee2f99-->
- [Neuronale Netze von Grund auf (Andrej Karpathy)](https://karpathy.ai/zero-to-hero.html): Die kostenlose Videoreihe zeigt, wie sich automatisches Differenzieren, Module und Trainingsschleifen in reinem Python von Grund auf implementieren lassen (:numref:`sec_model_construction`, :numref:`sec_custom_layer`).

<!--en:b066c45d-->
- [fast.ai, Teil 2: Grundlagen des Deep Learning (Howard et al.)](https://course.fast.ai/Lessons/part2.html): In diesem kostenlosen Kurs entsteht ausgehend von Tensoroperationen ein Trainingsframework mit Modulen, Initialisierung und Berechnungen mit gemischter Genauigkeit (:numref:`sec_model_construction`, :numref:`sec_init_param`, :numref:`sec_numerics`).

<!--en:b24c3731-->
**Anleitungen, Notizen und interaktive Materialien**

<!--en:84cbfaef-->
- [PyTorch von innen (Edward Yang)](http://blog.ezyang.com/2019/05/pytorch-internals/): Der Beitrag erklärt, wie PyTorch intern Tensoren, Strides, die Auswahl der passenden Implementierung und automatisches Differenzieren handhabt. Er geht eine Ebene tiefer als :numref:`sec_model_construction`.

<!--en:565210e5-->
- [JAX und seine Fallstricke](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html): Anhand praktischer Fallstricke erläutert die Anleitung den funktionalen Ansatz von JAX. Sie behandelt reine Funktionen, explizite PRNG-Schlüssel und die Einschränkungen der JIT-Kompilierung (:numref:`sec_parameters`, :numref:`sec_repro`).

<!--en:599e8531-->
- [Training mit gemischter Genauigkeit (NVIDIA)](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html): Der Leitfaden des Herstellers behandelt die Arithmetik in fp16 und bf16 sowie das Loss-Scaling. Die wesentlichen Punkte fasst :numref:`sec_numerics` zusammen.

<!--en:57c07d87-->
- [Was man in der Informatik über Gleitkommaarithmetik wissen sollte (David Goldberg)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html): Das Standardwerk erklärt die Grundlagen aller Rundungs- und Überlaufprobleme aus :numref:`sec_numerics`.

<!--en:3dcc9c62-->
- [Was Deep Learning schnell macht – von Grund auf erklärt (Horace He)](https://horace.io/brrr_intro.html): Der Beitrag erklärt Rechenleistung, Speicherbandbreite und den Zusatzaufwand durch das Framework anhand grundlegender Prinzipien (:numref:`sec_use_gpu`).

<!--en:d8f41a16-->
- [safetensors (Hugging Face)](https://huggingface.co/docs/safetensors/index): Die Dokumentation beschreibt das Dateiformat zum Austausch von Gewichten, das beim Laden keinen Code ausführt. Sie erklärt, warum es Checkpoints auf Pickle-Basis abgelöst hat (:numref:`sec_read_write`).

<!--en:00a668af-->
- [Reproduzierbarkeit (PyTorch-Notizen)](https://pytorch.org/docs/stable/notes/randomness.html): Als Ergänzung zu :numref:`sec_repro` erläutern die Notizen die Optionen für deterministische Berechnungen, ihre Laufzeitkosten und ihre Grenzen.

<!--en:d63f4e40-->
**Grundlegende Veröffentlichungen**

<!--en:6dad5c08-->
- [Automatisches Differenzieren im maschinellen Lernen: ein Überblick (Baydin, Pearlmutter, Radul und Siskind, 2018, *JMLR*)](https://jmlr.org/papers/v18/17-468.html): Das maßgebliche Übersichtswerk behandelt automatisches Differenzieren im Vorwärts- und Rückwärtsmodus. Auf diesem Verfahren beruhen alle selbst definierten Gradienten in :numref:`sec_custom_layer`.

<!--en:f8791fdd-->
- [Training mit gemischter Genauigkeit (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740): Aus dieser Arbeit stammt das Verfahren aus :numref:`sec_numerics`, das fp16, Hauptgewichte und Loss-Scaling kombiniert.

<!--en:303a71d3-->
- [PyTorch: eine leistungsfähige Deep-Learning-Bibliothek mit imperativem Programmierstil (Paszke et al., 2019)](https://arxiv.org/abs/1912.01703): Die Arbeit erläutert die Entwurfsentscheidungen hinter dem imperativen Stil von :numref:`sec_model_construction` und :numref:`sec_use_gpu`. Dazu gehören die sofortige Ausführung, automatisches Differenzieren und eine Speicherverwaltung, die freigegebene Speicherblöcke zur Wiederverwendung vorhält.

<!--en:8c19e674-->
Berechnungen im Deep Learning: vom Prototypcode zum Produktionssystem<br>**Modulbäume · Parameterzustand · Initialisierung · eigene Schichten · Numerik · Checkpoints · GPUs · Reproduzierbarkeit**

<!--en:7ca5d243-->
::: {.slide title="Vom MLP mit 4 Arrays zum Basismodell mit 8 Mrd. Parametern"}

<!--en:961550ab-->
[Anforderungen an das System]{.kicker}

<!--en:f970f620-->
- **Speicherbedarf (mindestens `128 GB`):** 8 Mrd. Gewichte belegen `16 GB` in `bfloat16`. Beim Training mit Adam in `float32` benötigen Parameter, Gradienten und die Momentschätzungen des Optimierers schon vor der ersten Aktivierung `128 GB`.

<!--en:c3e2c46f-->
- **Numerische Stabilität und Genauigkeit:** Über 32 Blöcke verstärken unskalierte Residualzweige die Varianz der Aktivierungen um den Faktor $2^{32}$. Bei 16-Bit-Gewichten gehen kleine Updates unbemerkt durch Rundung verloren.

<!--en:075308b7-->
- **Fehlertoleranz, Sicherheit und asynchrone GPUs:** Atomare Checkpoints (`96 GB`), `safetensors` und das Risiko der Codeausführung aus der Ferne bei `pickle`, synchronisierte GPU-Zeitmessung und Schicht-Hooks ohne Eingriff in die Berechnung

<!--en:798ec937-->
::: {.slide title="Ein Modell als Baum aus Modulen"}

<!--en:fc6976eb-->
[Grundvorstellung]{.kicker}

<!--en:127806f1-->
- **Adressierung über Pfade:** Der Pfad vom Wurzelmodul aus bezeichnet jeden Parameter und jeden nicht trainierbaren Puffer eindeutig (z. B. Block 1, Schicht 0, Gewicht).

<!--en:c82ab8fd-->
- **Einheitlicher Baumdurchlauf:** Beim Einrichten des Optimierers, beim Initialisieren, beim Serialisieren mit `safetensors`, beim Umwandeln von Datentypen, beim Übertragen auf GPUs und bei Hooks werden die registrierten Kindmodule rekursiv durchlaufen.

<!--en:d30ce925-->
- **Registrierung als stille Voraussetzung:** Schichten und Tensoren, die außerhalb des Modulregisters in gewöhnlichen Python-Attributen oder Listen liegen, bleiben für Optimierer, Checkpoints und GPUs unsichtbar.

<!--en:347e847f-->
::: {.slide title="Kapitel §{num:chap_computation} im Überblick: Architektur, Zustand und Ausführung"}

<!--en:8b8a96b7-->
[Architektur und System im Überblick]{.kicker}

<!--en:e20c1d9c-->
- **Modellstruktur und Zustand (§§{sec_model_construction}–§{num:sec_custom_layer}):** Modulbäume und Registrierung über `__setattr__`, mindestens $16\text{ B/param}$ beim Training in `fp32` mit Adam, Gewichtsbindung, Skalierung der Varianz in Residualnetzen mit $(1+g^2)^N$ ($g = 1/\sqrt{N}$), `RMSNorm`, feste Puffer und der Straight-Through-Schätzer

<!--en:392bb4e2-->
- **Genauigkeit, Ein- und Ausgabe sowie Hardware (§§{sec_numerics}–§{num:sec_repro}):** Hauptgewichte in `fp32` und Berechnungen in `bf16`/`fp16`, Laden ohne Kopieren mit `safetensors` und 5-teilige atomare Checkpoints, asynchrone GPU-Warteschlangen, Aktivierungs-Checkpointing mit $O(\sqrt{K})$ Speicherbedarf, PRNG-Startwerte und Hooks für den Vorwärts- und Rückwärtsdurchlauf

<!--en:5b6075a6-->
::: {.slide title="Wiederkehrende Benchmarks im Kapitel"}

<!--en:418d9f93-->
[Vergleichsgrößen]{.kicker}

<!--en:84dc51a2-->
- **Parameterzahl und Speicherbedarf (§§{sec_model_construction}–§{num:sec_parameters}, §{sec_numerics}):** Kleines Residual-MLP mit 4 Blöcken ($729{,}866$ Parameter) im Vergleich zu Llama 3 8B ($291$ Tensoren, `16 GB` für die Inferenz in `bf16`, mindestens `128 GB` für das Training in `fp32` mit Adam)

<!--en:18c93e45-->
- **Signalstabilität sowie Ein- und Ausgabe (§{sec_init_param}, §{sec_read_write}):** Mit $g = 1/\sqrt{N}$ wächst die Standardabweichung über 32 Blöcke höchstens um den Faktor $\sqrt{e} \approx 1.65\times$ (ohne Skalierung um $65{,}536\times$). Mit 5-teiligen Checkpoints von je `96 GB` lässt sich das Training bitgenau fortsetzen.

<!--en:7b14e43e-->
- **Aktivierungsspeicher (§{sec_use_gpu}):** Bei $K = 16$ Blöcken senken Segmente der Länge $s = \sqrt{K} = 4$ den maximalen Speicherbedarf pro Trainingsschritt: $1.1\text{ GiB} \to 0.45\text{ GiB}$.

<!--en:15d021a2-->
::: {.slide title="Wegweiser durch das Kapitel"}

<!--en:72fdbd10-->
[Hinweise für Studium und Praxis]{.kicker}

<!--en:5ba38583-->
Jeder Abschnitt verbindet ausführbaren Code für mehrere Frameworks mit Diagnoseverfahren, die unbemerkte Fehler beim Training verhindern:

<!--en:93729ff5-->
- **Grundlagen für alle:** **§{sec_model_construction}–§{sec_init_param}** lesen, um die Modulregistrierung, die Speicherbilanz für Parameter und Puffer und die Varianzskalierung in Residualnetzen zu verstehen. Anschließend **§{sec_numerics}–§{sec_use_gpu}** zu gemischter Genauigkeit, sicherem Checkpointing und GPU-Speicherverwaltung lesen.

<!--en:39e92d3c-->
- **Besonderheiten der Frameworks:** Genau darauf achten, wie das jeweilige Framework Listencontainer, die verzögerte Bestimmung von Tensorformen, den PRNG-Zustand und die Graphkompilierung handhabt.

<!--en:4b6f6be4-->
- **Diagnose von Anfang an:** Vor dem Start eines Trainingslaufs den Speicherbedarf für Parameter und Aktivierungen in Byte bestimmen. Die anfänglichen Standardabweichungen der Aktivierungen mit einem einzigen Vorwärtsdurchlauf prüfen. Bei jedem Laden externer Gewichte auf fehlende oder unerwartete Schlüssel achten.
