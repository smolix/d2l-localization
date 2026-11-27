<!--en:8fb4d2dc-->
# Werkzeuge für Deep Learning

<!--en:52543bfb-->
Die meisten Leserinnen und Leser führen den Code aus diesem Buch zunächst auf einem Laptop oder in einem Browsertab aus. Sie öffnen ein Notebook, führen die Zellen aus, ändern die Lernrate und führen die Zellen erneut aus. Früher oder später stoßen sie an eine Grenze. Ein Modell braucht mehr Speicher, als der Laptop hat, ein Trainingslauf würde eine Woche dauern, oder ein offenes Sprachmodell, das sie für ein Wochenendprojekt heruntergeladen haben, schreibt drei Wörter pro Sekunde. In jedem dieser Fälle scheint es um die Wahl des richtigen Produkts zu gehen. Dahinter stehen immer wieder dieselben Fragen: Passt die Aufgabe in den Speicher? Was begrenzt ihre Geschwindigkeit? Und was kostet es, ein fertiges Ergebnis zu erhalten?

<!--en:35dc7d83-->
Produkte ändern sich innerhalb weniger Monate, doch die Antworten beruhen auf einigen langlebigeren Regeln. Der Speicher entscheidet, was überhaupt laufen kann. Bevor wir also einen Lauf beschleunigen, prüfen wir, ob der Speicher dafür ausreicht. Wenn ein Sprachmodell Text erzeugt, liest es für jedes neue Token jedes benötigte Gewicht einmal aus dem Speicher. Die Speicherbandbreite bestimmt also, wie schnell das Modell Text erzeugt. Der Rechner mit dem niedrigsten Stundenpreis kann das Experiment verteuern, wenn er für die Aufgabe länger braucht oder der Lauf vorzeitig abbricht. Deshalb vergleichen wir die Kosten pro fertigem Ergebnis. Lässt sich ein Ergebnis anhand der Aufzeichnungen nicht reproduzieren, können wir es weder überprüfen noch darauf aufbauen. Reproduzierbarkeit gehört also von der ersten Zelle an zum Arbeitsablauf. In :numref:`chap_performance` begrenzten die Rechenleistung, die Speicherbandbreite oder der Aufwand für das Anstoßen von Operationen die Geschwindigkeit eines Programms. Mit diesen Regeln wenden wir das dortige Modell an, wenn wir entscheiden, was wir ausführen, mieten, kaufen oder in Betrieb nehmen.

<!--en:f8af5c3c-->
Das Kapitel führt vom einzelnen interaktiven Notebook bis zum Training auf mehreren Rechnern und zum produktiven Inferenzbetrieb. Wir beginnen dort, wo wir unseren Code im Alltag ausführen. Wir verwalten den Zustand des Jupyter-Kernels und lokale `uv`-Umgebungen so, dass die Ergebnisse nicht davon abhängen, in welcher Reihenfolge Zellen zuvor ausgeführt wurden, ohne dass dies im Notebook ersichtlich ist (:numref:`sec_interactive_development`). Auf Colab und Kaggle führen wir Notebooks ganz ohne lokale Einrichtung aus (:numref:`sec_hosted_notebooks`). Bei gemieteten Cloud-Beschleunigern berechnen wir die Gesamtkosten pro fertigem Ergebnis (:numref:`sec_cloud_instances`). Danach wenden wir uns der Hardware und externen Artefakten zu. Wir entscheiden anhand von Speicherkapazität, Bandbreite und Grenzen der Rechenleistung, welche Hardware wir kaufen (:numref:`sec_hardware_buyers`). Auf öffentlichen Plattformen finden und bewerten wir vortrainierte Modelle und Datensätze, legen ihre Versionen fest und klären ihre Lizenzierung (:numref:`sec_software_ecosystem`). Zum Schluss beziehen wir mehrere Rechner und Mitwirkende ein. Wir verteilen das Training auf mehrere Beschleuniger (:numref:`sec_training_systems`), bedienen mehrere Nutzer gleichzeitig unter Einhaltung der Latenzvorgaben (:numref:`sec_model_serving`) und sehen uns an, wie dieses Buch aus einer gemeinsamen Markdown-Quelle entsteht und gepflegt wird (:numref:`sec_developers_guide`).

<!--en:7da7f1cb-->
Preise, Produktnamen, Kontingente und Bibliotheksversionen ändern sich schnell. Die jeweiligen Abschnitte nennen den Stand jeder Angabe. Prüfen Sie vor der Verwendung, ob sie noch aktuell ist.

<!--en:604fe18a-->
Die automatisch erzeugte Dokumentation der Hilfsfunktionen und der `d2l`-API bleibt als durchsuchbare HTML-Referenz verfügbar. Sie gehört weder zum Lehrstoff dieses Kapitels noch zur PDF-Ausgabe.

<!--en:5dd576ec-->
## Materialien und weiterführende Literatur {.unnumbered}

<!--en:71077584-->
Die folgenden Materialien begleiten die Abschnitte dieses Kapitels von der Arbeit auf einem einzelnen Rechner bis zum Training und Inferenzbetrieb in großem Maßstab. Alle sind online frei zugänglich.

<!--en:1a925603-->
**Systeme und Skalierung**

<!--en:3474c6d3-->
- [How to Scale Your Model — Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/): Dieses Onlinebuch zeigt, wie sich die Rechenleistung vor der ersten Messung abschätzen lässt. Es behandelt Rooflines, den Aufbau von TPUs und GPUs sowie Parallelität und rechnet den Aufwand der Inferenz an Zahlenbeispielen vor. Es ergänzt :numref:`sec_hardware_buyers`, :numref:`sec_training_systems` und :numref:`sec_model_serving`.

<!--en:783b25e3-->
- [The Ultra-Scale Playbook — Hugging Face (2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook): Dieser Leitfaden zum Training auf GPU-Clustern stützt sich auf Messungen aus über 4000 Läufen auf bis zu 512 GPUs. Er ergänzt :numref:`sec_training_systems` um detaillierte Speicherbilanzen und zeigt, welche Vor- und Nachteile verschiedene Formen der Parallelität haben.

<!--en:1cb373fe-->
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/): Mit Vorlesungen auf YouTube und Aufgaben auf GitHub entwickeln Sie ein Sprachmodell von Grund auf, von der Tokenisierung bis zum verteilten Training und zur Inferenz. Der Kurs begleitet :numref:`sec_training_systems` und :numref:`sec_model_serving`.

<!--en:98115f0f-->
- [Vorlesungen von GPU MODE](https://github.com/gpu-mode/lectures): Videos, Folien und Notebooks von den CUDA-Grundlagen bis zu FlashAttention und Quantisierung für alle, die selbst die Kernel schreiben möchten, deren Leistungsgrenzen wir in :numref:`sec_hardware_buyers` abschätzen.

<!--en:38908ea7-->
**Die Laufzeit verstehen**

<!--en:a4fc976e-->
- [Making Deep Learning Go Brrrr From First Principles — Horace He (2022)](https://horace.io/brrr_intro.html): Dieser kurze Essay schlüsselt die Laufzeit eines Programms danach auf, wie viel Zeit Rechenoperationen, bandbreitenbegrenzte Speicherzugriffe und Verwaltungsaufwand beanspruchen. Auf dieser Einteilung beruht :numref:`sec_hardware_buyers`.

<!--en:46a39da1-->
- [Transformer Inference Arithmetic — kipply (2022)](https://kipp.ly/p/transformer-inference-arithmetic): Der Beitrag rechnet von Hand vor, wie groß der KV-Cache ist und wie die Bandbreite die Geschwindigkeit der Dekodierung begrenzt. Darauf beruhen die Abschätzungen in :numref:`sec_hardware_buyers` und :numref:`sec_model_serving`.

<!--en:c18a5abb-->
- [Accelerating Generative AI with PyTorch II: GPT, Fast — PyTorch-Team (2023)](https://pytorch.org/blog/accelerating-generative-ai-2/): Der Beitrag und der zugehörige [gpt-fast](https://github.com/meta-pytorch/gpt-fast)-Code zeigen Schritt für Schritt, wie sich die Inferenz in weniger als 1000 Zeilen PyTorch-Code fast um den Faktor 10 beschleunigen lässt. Zum Einsatz kommen Kompilierung, Quantisierung, spekulative Dekodierung und Tensorparallelität.

<!--en:e6855df9-->
- [Which GPU for Deep Learning? — Tim Dettmers (2023)](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/): Der Beitrag wurde zuletzt im Januar 2023 aktualisiert. Seine Hardwarebeispiele sind daher veraltet, doch sein Verfahren, eine GPU passend zur Arbeitslast auszuwählen, ergänzt :numref:`sec_hardware_buyers`.

<!--en:5d8b025f-->
**Übersichtsarbeiten**

<!--en:5f8b0eff-->
- [A Survey on Efficient Inference for Large Language Models — Zhou et al. (2024)](https://arxiv.org/abs/2404.14294): Die Arbeit ordnet Optimierungen für die Inferenz nach Daten-, Modell- und Systemebene und vergleicht repräsentative Verfahren in Experimenten. Sie liefert Hintergrundwissen zu :numref:`sec_model_serving`.

<!--en:774f6e59-->
- [A Survey on Large Language Model Acceleration Based on KV Cache Management — Li et al. (2024)](https://arxiv.org/abs/2412.19442): Die Arbeit behandelt Verfahren zum Speichern, Komprimieren und Verdrängen von Einträgen im KV-Cache, den :numref:`sec_model_serving` einführt, und enthält eine Auswahl an Fachartikeln.

<!--en:eeeb051a-->
**Auf dem Laufenden bleiben**

<!--en:bd3915f4-->
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) und die [Diskussionen zu llama.cpp](https://github.com/ggml-org/llama.cpp/discussions) bieten Berichte zu Neuveröffentlichungen, Ergebnisse zur Quantisierung, Hardwaremessungen und Versuche, Ergebnisse zu reproduzieren. Wie man solche Quellen einordnet, beschreibt :numref:`sec_software_ecosystem`.

<!--en:0b9f61a6-->
- [Pro Git — Chacon und Straub](https://git-scm.com/book/en/v2): Die Kapitel 1–3 und 6 dieses frei zugänglichen Buchs behandeln den Umgang mit Git und Pull Requests, den :numref:`sec_developers_guide` voraussetzt.

<!--en:e76fff19-->
Werkzeuge für Deep Learning: von lokalen Notebooks bis zu verteilten Clustern und zum Inferenzbetrieb<br>**reproduzierbare Kernel · gehostete Laufzeitumgebungen · Cloud-Kosten · Hardware-Rooflines · Artefakte mit offenen Gewichten · verteiltes Training · Inferenz-Engines**

<!--en:c169e67c-->
::: {.slide title="Vier Fragen zur Wahl der Werkzeuge"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:6a02baab-->
- **1. Reicht der Speicher?** Beim Training liegen Parameter, Gradienten, die Momente des Optimierers (zusammen $16N$) und Aktivierungen im VRAM. Geht der Speicher aus, bricht der Lauf ab, bevor die FLOP/s eine Rolle spielen.

<!--en:d07db932-->
- **2. Was begrenzt den Durchsatz?** Training und Prompt-Prefill sind rechenbegrenzt. Beim autoregressiven Dekodieren mit Batchgröße 1 begrenzt die Speicherbandbreite den Durchsatz.

<!--en:63b97272-->
- **3. Kosten und Reproduzierbarkeit:** Minimieren Sie die Gesamtkosten pro fertigem Ergebnis, einschließlich Einrichtung, ausgehendem Datenverkehr und Wiederanlauf nach einer Spot-Unterbrechung. Legen Sie Umgebungen und SHA-Hashes ab der ersten Zelle fest.

<!--en:517c847d-->
::: {.slide title="Drei grundlegende Gleichungen zur Dimensionierung von Systemen"}

<!--en:b2d9898b-->
[Quantitative Regeln]{.kicker}

<!--en:7fd2b1d8-->
- **Trainingszustand und Rechenaufwand (§{sec_hosted_notebooks}, §{sec_hardware_buyers}, §{sec_training_systems}):** Adam benötigt bei gemischter Genauigkeit $16\text{ B/param}$ an Speicher. Der Rechenaufwand beträgt $C \approx 6ND\text{ FLOPs}$ für $N$ Parameter und $D$ Token.

<!--en:50b9507a-->
- **Dekodierung und Checkpoint-Intervalle (§{sec_hosted_notebooks}–§{num:sec_model_serving}):** Die Dekodierung mit Batchgröße 1 liest $B_w + B_{\text{kv}}$ mit der Bandbreite $\beta$. Nach der Young-Formel liegt das optimale Intervall zwischen Checkpoints bei $\tau^* = \sqrt{2\delta M}$.

<!--en:d694597f-->
::: {.slide title="Kapitel §{num:chap_appendix_tools} im Überblick: vom Laptop zum Produktivbetrieb"}

<!--en:00be04d1-->
[Kapitelaufbau]{.kicker}

<!--en:ad3d532b-->
- **§{sec_interactive_development}–§{num:sec_cloud_instances} Notebooks, gehostete VMs und Cloud:** Den Kernelzustand verwalten, Versionen mit `uv`-Lockdateien festschreiben, Daten auf Colab und Kaggle dauerhaft speichern und Cloud-Kosten pro fertigem Ergebnis berechnen.

<!--en:46b2f579-->
- **§{sec_hardware_buyers}–§{num:sec_software_ecosystem} Hardware und Softwareumfeld:** Kapazitäts- und Bandbreitengrenzen im Roofline-Modell, gemeinsamer Speicher für MoE, Auflösungsvermögen der Arena-Elo-Wertung und feste Versionen mit `safetensors` und SHA-Hashes.

<!--en:931e7f99-->
- **§{sec_training_systems}–§{num:sec_utils} Skalierung, Inferenzbetrieb und D2L-Interna:** DDP/FSDP2, 3D-Parallelität, kontinuierliches Batching, PagedAttention und der Build von D2L aus einer gemeinsamen Quelle.

<!--en:8142e7ce-->
::: {.slide title="Durchgerechnete Beispiele und Benchmarks im Kapitel"}

<!--en:05adf916-->
[Beispiele im Zusammenhang]{.kicker}

<!--en:06cebc8a-->
- **Reproduzierbare Umgebungen und Kosten (§{sec_interactive_development}–§{num:sec_cloud_instances}):** `uv pip sync`, Fingerabdrücke von Umgebungen und Cloud-Kostenmodelle für 4090, A100 und H100, die auch die Arbeitszeit der Ingenieure berücksichtigen.

<!--en:f26e36ed-->
- **Dimensionierung und Überprüfung (§{sec_hardware_buyers}–§{num:sec_software_ecosystem}):** Trainingszustand für 7 Milliarden Parameter ($78\text{ GiB}$ beim vollständigen Training gegenüber $\sim 4\text{ GiB}$ mit QLoRA), Leistungsgrenzen der Dekodierung, Bradley-Terry-Standardfehler ($347/\sqrt{v}$) und festgeschriebene Manifestdateien.

<!--en:e6bc6299-->
- **Verteiltes Training, Inferenzbetrieb und Build (§{sec_training_systems}–§{num:sec_developers_guide}):** FSDP2 auf 8 GPUs ($22\text{ GiB}$ gegenüber $95\text{ GiB}$ bei DDP), Simulatoren für kontinuierliches Batching und Hashwerte für die Ausgaben von Notebook-Präfixen.

<!--en:5ec7fbd8-->
::: {.slide title="Lesewege durch das Kapitel und Quellen für aktuelle Entwicklungen"}

<!--en:123f19cf-->
[Lesehilfe]{.kicker}

<!--en:33b7fe08-->
- **Schnelle Einrichtung (§{sec_interactive_development}–§{num:sec_hosted_notebooks}):** Beginnen Sie hier, um die Notebooks des Buchs lokal mit `uv` oder auf Colab und Kaggle auszuführen und Ihre Arbeit auch dann zu sichern, wenn eine Sitzung endet.

<!--en:50ebd20b-->
- **Infrastrukturentscheidungen (§{sec_cloud_instances}–§{num:sec_software_ecosystem}):** Lesen Sie diese Abschnitte, bevor Sie Cloud-GPUs mieten, eine Workstation bauen oder einen Checkpoint mit offenen Gewichten von Hugging Face übernehmen.

<!--en:ecb18b1a-->
- **Skalierung und Mitarbeit (§{sec_training_systems}–§{num:sec_utils}):** Für das Training auf mehreren GPUs und den Inferenzbetrieb großer Sprachmodelle lesen Sie §{sec_training_systems}–§{num:sec_model_serving} zusammen mit Kapitel §{num:chap_performance} (*Rechenleistung*). §{sec_developers_guide} erläutert, wie Sie zu D2L beitragen, und §{sec_utils} beschreibt die Hilfsfunktionen von `d2l`.

<!--en:c10e68dc-->
- **Begleitliteratur:** Ergänzend zum Kapitel eignen sich *How to Scale Your Model* von DeepMind, das *Ultra-Scale Playbook* von Hugging Face und der Kurs Stanford CS336.

<!--en:05480b67-->
**Als Nächstes (§{sec_interactive_development} Notebooks):** Wie der Jupyter-Kernel Zellen ausführt, warum ein versteckter Zustand die Reproduzierbarkeit verhindert und wie wir mit `uv` deterministische lokale Umgebungen einrichten.
