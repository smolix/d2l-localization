<!--en:e436e849-->
# Informationstheorie und Divergenzen

<!--en:6ef8edf1-->
Die meisten Modelle in diesem Buch trainieren wir, indem wir den Kreuzentropieverlust (engl. cross-entropy loss) minimieren. Unser Ziel ist zunächst informell: Das Modell soll beobachteten Ergebnissen eine hohe Wahrscheinlichkeit zuweisen. Die Kreuzentropie macht daraus eine glatte Zielfunktion für den Gradientenabstieg. Doch was misst der Wert dieser Verlustfunktion? Wenn ein Sprachmodell 1.2 Nat pro Zeichen erreicht, wie weit könnte dieser Wert noch sinken? Und was vergleichen wir, wenn wir die Kreuzentropie durch den KL-Strafterm eines Variationsautoencoders oder durch den Kritiker eines generativen adversarialen Netzes ersetzen?

<!--en:666cb776-->
Claude Shannon beantwortete diese Fragen 1948, als er untersuchte, wie effizient sich eine Nachricht über eine verrauschte Leitung übertragen lässt. Sein Maß für die Unvorhersagbarkeit, die *Entropie*, gibt eine Untergrenze an, die kein Modell mit seiner Kreuzentropie unterschreiten kann (:numref:`subsec_info_theory_basics`). Daraus ergeben sich drei Konsequenzen für das gesamte Deep Learning. Erstens ist die Kreuzentropie um die *KL-Divergenz* von der Datenverteilung zur Modellverteilung größer als die Entropie der Daten. Diese Differenz verschwindet nur, wenn beide Verteilungen übereinstimmen. Zweitens entspricht der Gesamtverlust eines Modells in Bit bis auf höchstens zwei zusätzliche Bit genau der Größe einer komprimierten Datei. Damit hat die Perplexität eines Sprachmodells (:numref:`subsec_perplexity`) eine unmittelbar messbare Bedeutung. Drittens liefern trainierte Kritiker nur untere Schranken für Divergenzen und für die gegenseitige Information (engl. mutual information). Beim kontrastiven Lernen lässt sich durch die Identifikation des passenden Partners unter $N$ Kandidaten ein Informationsgehalt von höchstens $\log N$ Nat nachweisen. Schwache adversariale Kritiker (:numref:`sec_gan_objectives`) können weit auseinanderliegende Verteilungen ähnlich erscheinen lassen.

<!--en:75196a36-->
Das Kapitel gliedert sich in drei Teile. Im ersten Teil leiten wir Entropie und Kreuzentropie her (:numref:`sec_mdl-information_theory`) und zeigen, wie der logarithmische Verlust (engl. log-loss) unmittelbar mit verlustfreier Kompression und Perplexität zusammenhängt (:numref:`sec_mdl-coding-compression`). Der zweite Teil behandelt eine grundlegende Schwäche von Divergenzen, die auf Dichteverhältnissen beruhen (:numref:`sec_mdl-divergences-distances`). Wenn die Samples des Modells und die Daten aus disjunkten Bereichen stammen, wird die KL-Divergenz unendlich. Beschränkte $f$-Divergenzen erreichen dann ihren Sättigungswert und ihr Gradient ist null. Vergleichen wir dagegen Erwartungswerte glatter Testfunktionen oder ziehen wir die Kosten des optimalen Transports heran (:numref:`sec_mdl-test-functions-transport`), erhalten wir wieder aussagekräftige Gradienten. Durch den Abgleich der Gradienten der Log-Dichte (:numref:`sec_mdl-scores`) können wir auch unnormierte Modelle vergleichen, ohne deren schwer berechenbare Partitionsfunktionen auszuwerten. Im dritten Teil untersuchen wir Paare von Zufallsvariablen. Hier geht es um gegenseitige Information, Datenverarbeitung, die Fano-Ungleichung und den Informationsflaschenhals (engl. information bottleneck) (:numref:`sec_mdl-mutual-information`). Zum Abschluss behandeln wir die Schätzung aus Stichproben und InfoNCE (:numref:`sec_mdl-mi-estimation`). Diese beiden letzten Abschnitte können Sie direkt nach :numref:`sec_mdl-information_theory` lesen, wenn Sie sich Transport und Scores für :numref:`chap_gans` und :numref:`chap_diffusion` aufheben möchten.

<!--en:6168ab04-->
Wir setzen die Jensen-Ungleichung (:numref:`chap_mdl-optimization`), bedingte Verteilungen und die Maximum-Likelihood-Methode (:numref:`chap_mdl-probability-statistics`) als bekannt voraus. Einen Überblick über fehlerkorrigierende Codes, Kolmogorov-Komplexität und Informationsgeometrie bietet :numref:`sec_mdl-information-theory-further-reading`.

<!--en:bb63df72-->
Informationstheorie und Divergenzen<br>**Entropie · Kreuzentropie · Quellenkodierung · f-Divergenzen · optimaler Transport · Scores · gegenseitige Information**

<!--en:a1b635eb-->
::: {.slide title="Warum trainieren wir Modelle mit Kreuzentropie?"}

<!--en:d19f72f6-->
[Leitfragen zu Kapitel §{num:chap_mdl-information-theory}]{.kicker}

<!--en:4220ee9e-->
- **Was misst der Verlustwert?** Jeder Kreuzentropieverlust ist die Summe aus der unvermeidbaren Entropie der Daten und einer nichtnegativen Divergenz: $\textrm{CE}(P, Q) = H(P) + D_{\textrm{KL}}(P \| Q) \ge H(P)$.

<!--en:f8366967-->
- **Warum entspricht der logarithmische Verlust einer Dateigröße?** Arithmetische Kodierung komprimiert $n$ Symbole unter dem Modell $q(x_i \mid x_{<i})$ auf höchstens $n\,\textrm{CE}_2(P, Q) + 2$ Bit.

<!--en:0ba6ec15-->
- **Was tun, wenn die KL-Divergenz unendlich wird?** Optimaler Transport ($W_1$), Kernel-MMD und Score-Matching vergleichen Verteilungen allein anhand von Stichproben oder Gradienten der Log-Dichte.

<!--en:7eb76565-->
::: {.slide title="Drei grundlegende Sätze für die Praxis"}

<!--en:787799c2-->
[Mathematische Grundlagen]{.kicker}

<!--en:5e149f89-->
- **Gibbs-Ungleichung (§{sec_mdl-information_theory}):** Es gilt $D_{\textrm{KL}}(P \| Q) \ge 0$ ($= 0 \iff P = Q$). Daher kann kein Klassifikator die bedingte Entropie der Labels $H(Y \mid X)$ unterschreiten.

<!--en:5c3a99c4-->
- **Quellenkodierung (§{sec_mdl-coding-compression}):** Mit der Kraft-Ungleichung ($\sum_i 2^{-\ell_i} \le 1$) und arithmetischer Kodierung lässt sich jedes kausale LLM zur verlustfreien Kompression nutzen.

<!--en:efa94483-->
- **Variationelle Kritiker (§§{sec_mdl-divergences-distances}–§{num:sec_mdl-mi-estimation}):** $f$-GAN und InfoNCE liefern aus Stichproben untere Schranken für Divergenzen und gegenseitige Information. Die InfoNCE-Schranke beträgt höchstens $\log N$ Nat bei Batchgröße $N$.

<!--en:6045bd7e-->
::: {.slide title="Kapitel §{num:chap_mdl-information-theory} im Überblick: Sieben zusammenhängende Abschnitte"}

<!--en:00be04d1-->
[Aufbau des Kapitels]{.kicker}

<!--en:466fc526-->
- **Teil 1: Logarithmischer Verlust, Kodierung und Perplexität (§§{sec_mdl-information_theory}–§{num:sec_mdl-coding-compression}):** Überraschungswert, Shannon-Entropie, untere Schranke nach Gibbs, Label-Smoothing, Destillation, Kraft-Ungleichung, arithmetische Kodierung, MDL, AEP und Perplexität

<!--en:c2ba23fe-->
- **Teil 2: Divergenzen, Transport und Scores (§§{sec_mdl-divergences-distances}–§{num:sec_mdl-scores}):** $f$-Divergenzen, Pinsker-Ungleichung, $f$-GAN, IPMs, Kernel-MMD, Wasserstein-Distanz $W_1$, optimaler Transport nach Sinkhorn, Stein-Identität, Score-Matching und KSD

<!--en:69a456db-->
- **Teil 3: Abhängigkeit und kontrastive Schranken (§§{sec_mdl-mutual-information}–§{num:sec_mdl-mi-estimation}):** Gegenseitige Information, Datenverarbeitungsungleichung (DPI), Fano-Ungleichung, Informationsflaschenhals, nächste Nachbarn im KSG-Schätzer, obere Schranke nach McAllester und Stratos und InfoNCE

<!--en:c98d44c1-->
::: {.slide title="Wie lassen sich Verteilungen je nach Datenzugang vergleichen?"}

<!--en:34675d55-->
[Entscheidungshilfe für die Praxis]{.kicker}

<!--en:03a980b7-->
- **Normierte Dichte $q_{\boldsymbol{\theta}}(\mathbf{x}) > 0$:** Die **Vorwärts-KL** $D_{\textrm{KL}}(P \| Q_{\boldsymbol{\theta}})$ mit der Maximum-Likelihood-Methode minimieren (deckt die Modi ab).

<!--en:98fc89e4-->
- **Unnormierte Zieldichte $\tilde{p}(\mathbf{z}, \mathbf{x})$ mit berechenbarem $q_{\boldsymbol{\theta}}$:** Die **Rückwärts-KL** $D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \| P)$ durch variationelle Inferenz mithilfe der ELBO minimieren (sucht einzelne Modi).

<!--en:c8340f17-->
- **Impliziter Generator oder EBM:** Liefert der Generator nur Samples, **Kernel-MMD** oder die **Wasserstein-Distanz $W_1$** verwenden. Lässt sich nur der Score $-\nabla_{\mathbf{x}}E_{\boldsymbol{\theta}}$ berechnen, eignen sich die **Fisher-Divergenz** oder die **KSD**.

<!--en:0abd062d-->
::: {.slide title="Durchgängige Benchmarks und Lesewege"}

<!--en:2aa2f8db-->
[Orientierung im Kapitel]{.kicker}

<!--en:8e50e6b0-->
Konkrete, reproduzierbare Benchmarks prüfen jede theoretische Schranke in diesem Kapitel:

<!--en:7076359f-->
- **Diskrete Benchmarks:** Die dyadische Quelle $(\tfrac12, \tfrac14, \tfrac18, \tfrac18)$, das asymmetrische Paar $P = (0.6, 0.3, 0.1)$ und $Q = (0.2, 0.5, 0.3)$, das gekippte Bit ($\delta = 0.1$) und die zeichenweise Kompression von *The Time Machine*

<!--en:3fb67706-->
- **Stetige Benchmarks:** Gauß-Verteilungen $\mathcal{N}(0, 1)$ und $\mathcal{N}(d, \sigma^2)$ mit verschobenem Mittelwert und skalierter Varianz im Vergleich, bimodale Gauß-Mischungen (Mischungsverhältnis $70/30$) und ein Paar gemeinsam normalverteilter Zufallsvariablen mit Korrelation $\rho = 0.9$ und $\rho = 0.99$

<!--en:832e47e3-->
- **Drei Lesewege:**
  1. **Logarithmischer Verlust und Sprachmodelle (§§{sec_mdl-information_theory}–§{num:sec_mdl-coding-compression}):** Für überwachtes Lernen und Perplexität beide Abschnitte der Reihe nach lesen.
  2. **Verteilungsvergleiche in der generativen Modellierung (§§{sec_mdl-divergences-distances}–§{num:sec_mdl-scores}):** Grundlagen für GANs, Flow-Matching und Diffusionsmodelle
  3. **Abhängigkeit und kontrastives Lernen (§§{sec_mdl-mutual-information}–§{num:sec_mdl-mi-estimation}):** Baut nur auf §{sec_mdl-information_theory} auf. Für VAEs, SimCLR und CLIP direkt hier weiterlesen.
