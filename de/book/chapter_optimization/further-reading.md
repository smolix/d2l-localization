<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:c6951c42-->
Dieser Abschnitt stellt grundlegende Lehrbücher, Übersichtsarbeiten und neuere Forschungsarbeiten vor, die die Methoden dieses Kapitels erweitern. Die mathematischen Beweise für die im Text genannten Konvergenzraten finden sich in :numref:`chap_mdl-optimization`. Weitere Literatur zur Theorie nennt :numref:`sec_mdl-optimization-further-reading`. Die automatisierte Hyperparameteroptimierung behandeln wir in :numref:`chap_hyperopt`, Systeme für das verteilte Training auf mehreren Geräten in :numref:`chap_performance`.

<!--en:7e6a74bc-->
## Lehrbücher, Übersichtsarbeiten und Praxisleitfäden

<!--en:af678677-->
* **Konvexe und numerische Optimierung:** Das Buch von :citet:`Boyd.Vandenberghe.2004` (*[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)*) ist das Standardwerk zur Konditionierung, zum Gradientenabstieg, zum Newton-Verfahren (:numref:`sec_gd`) und zum steilsten Abstieg in allgemeinen Normen (:numref:`sec_muon`). :citet:`Nocedal.Wright.2006` behandeln in *Numerical Optimization* ausführlich Newton-Verfahren, Quasi-Newton-Verfahren, Liniensuche und Trust-Region-Verfahren.

<!--en:06a2b213-->
* **Stochastische Optimierung:** :citet:`Bottou.Curtis.Nocedal.2018` (*[Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)*) geben einen Überblick über die Kugel um das Minimum (engl. noise ball), in der die Iterierten bei stochastischen Gradienten schwanken, über abnehmende Schrittweiten (:numref:`sec_sgd`) und über Varianzreduktion.

<!--en:982305cd-->
* **Interaktive Einführungen und Praxisleitfäden:** Gabriel Goh veranschaulicht in *[Why Momentum Really Works](https://distill.pub/2017/momentum/)* (Distill, 2017) mit interaktiven Grafiken die kritische Dämpfung und das Beschleunigungsgesetz mit dem Faktor $\sqrt{\kappa}$ (:numref:`sec_momentum`). Jeremy Bernsteins Beitrag *[Deriving Muon](https://jeremybernste.in/writing/deriving-muon)* und die Arbeit von :citet:`Bernstein.Newhouse.2024` geben eine einheitliche Darstellung des steilsten Abstiegs für unterschiedliche Normkugeln (:numref:`sec_muon`). Der *[Practitioner's Guide to Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/)* von EleutherAI beschreibt die Koordinatenprüfungen für $\mu\text{P}$ im Detail (:numref:`sec_scaling`). Das *[Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)* von Google Research :cite:`Godbole.Dahl.Gilmer.ea.2023` unterscheidet formal zwischen den wissenschaftlich untersuchten Hyperparametern, den Störhyperparametern und den fest vorgegebenen Hyperparametern. Diese Unterscheidung verwenden wir durchgehend in :numref:`sec_test_models` und :numref:`sec_practice`.

<!--en:efa04b17-->
## Die Verlustlandschaft

<!--en:d26423de-->
### Visualisierung hochdimensionaler Verlustlandschaften und der Verbindungen zwischen Minima

<!--en:522ac3f4-->
Niedrigdimensionale Schnitte durch die Verlustlandschaft eines neuronalen Netzes können täuschen. Um die Skaleninvarianzen aufgrund positiver Homogenität auszuschalten, muss man die Störrichtungen für jeden Filter einzeln normieren. Mit den skalierten Gewichten $c\,\mathbf{W}_1$ und $c^{-1}\mathbf{W}_2$ berechnet das ReLU-Netz dieselbe Funktion. Anhand solcher filternormierten Schnitte zeigten :citet:`Li.Xu.Taylor.ea.2018`, dass Residualverbindungen (:numref:`sec_resnet`) zerklüftete, nichtkonvexe Flächen in glatte Becken verwandeln.

<!--en:2382e906-->
Bei linearer Interpolation zwischen zwei unabhängig trainierten Netzen muss man eine Barriere mit hohem Verlust überwinden. Dennoch entdeckten :citet:`Garipov.Izmailov.Podoprikhin.ea.2018` und :citet:`Draxler.Veschgini.Salmhofer.ea.2018` die **Verbindung zwischen Minima** (engl. mode connectivity). Ein einfacher, stückweise linearer Pfad mit nur einem Knick verbindet unabhängig gefundene Minima. Entlang des gesamten Pfads bleiben Trainings- und Testverlust niedrig. :citet:`Entezari.Sedghi.Saukh.ea.2022` und :citet:`Ainsworth.Hayase.Srinivasa.2023` zeigten, dass die Barriere auf der Geraden zwischen breiten Netzen fast verschwindet, wenn man ihre versteckten Einheiten mithilfe der Permutationssymmetrie (:numref:`sec_gd`) aufeinander ausrichtet.

<!--en:47554bb1-->
### Flache Minima und Sharpness-Aware Minimization (SAM)

<!--en:d8f23678-->
Dem Verfahren **Sharpness-Aware Minimization (SAM)** :cite:`Foret.Kleiner.Mobahi.ea.2021` liegt die Hypothese zugrunde, dass flachere Minima besser generalisieren. SAM minimiert den größten Verlust innerhalb einer $\ell_2$-Kugel um die aktuellen Gewichte. Dazu stört SAM die Gewichte zunächst mit einem Schritt in Richtung des Gradientenanstiegs und berechnet dort den Gradienten für den Abstieg. Damit steigen die Kosten der Gradientenberechnung pro Aktualisierung auf $2\times$ die Kosten beim gewöhnlichen Abstieg. Die direkt aus der Hesse-Matrix bestimmte Schärfe ist allerdings gegenüber schichtweisen Umparametrisierungen nicht invariant :cite:`Dinh.Pascanu.Bengio.ea.2017`.

<!--en:afb51542-->
## Klassische Verfahren zweiter Ordnung und Verfahren mit Varianzreduktion

<!--en:31c92eb8-->
### Quasi-Newton-Verfahren, Liniensuche und Trust-Region-Verfahren

<!--en:e5c00a49-->
**Quasi-Newton-Verfahren** approximieren die inverse Hesse-Matrix $\mathbf{H}^{-1}$ anhand aufeinanderfolgender Sekantenpaare $(\mathbf{x}_{t+1} - \mathbf{x}_t, \mathbf{g}_{t+1} - \mathbf{g}_t)$. Dazu benötigen sie nur erste Ableitungen. Die speichersparende BFGS-Variante (**L-BFGS**, :citet:`Liu.Nocedal.1989`) speichert die $m$ jüngsten Paare und benötigt dafür $O(md)$ Speicher. Für deterministische glatte Probleme ist sie das Verfahren der Wahl (:numref:`subsec_mdl-quasi-newton`), reagiert aber empfindlich auf Differenzen zwischen Minibatch-Gradienten. Auch die **Liniensuche** und die **Trust-Region-Verfahren** eignen sich besonders für die deterministische Optimierung. Die Liniensuche wählt die Schrittweite $\eta_t$ dynamisch und prüft mit der Armijo-Bedingung, ob die Zielfunktion hinreichend abnimmt. Trust-Region-Verfahren behandeln wir in :numref:`subsec_mdl-trust-region`. Eine entsprechende Analyse von Muon findet sich bei :citet:`Kovalev.2025`.

<!--en:332f8348-->
### Hesse-freie und sattelfreie Newton-Verfahren

<!--en:77d9f5b1-->
Verfahren zweiter Ordnung lassen sich auf tiefe Netze anwenden, ohne die $d \times d$ große Hesse-Matrix explizit zu bilden. Die **Hesse-freie Optimierung** :cite:`Martens.2010` löst $\mathbf{H}\mathbf{d} = -\mathbf{g}$ näherungsweise mit dem Verfahren der konjugierten Gradienten. Dazu nutzt sie Hesse-Vektor-Produkte (:numref:`subsec_mdl-hvp`) mit der positiv semidefiniten Gauß-Newton-Matrix. **Saddle-Free Newton** :cite:`Dauphin.Pascanu.Gulcehre.ea.2014` verwendet $|\mathbf{H}|^{-1}$ zur Vorkonditionierung auf einem niedrigdimensionalen Krylov-Unterraum. In Richtungen negativer Krümmung bewegen sich die Iterierten dadurch vom Sattelpunkt weg statt auf ihn zu.

<!--en:c6c4d065-->
### Stochastische Konvergenzraten und Varianzreduktion

<!--en:13a971b4-->
Für allgemeine konvexe Zielfunktionen erreicht der stochastische Gradientenabstieg mit Minibatches (Minibatch-SGD) die minimax-optimale Rate $O(1/\sqrt{T})$ :cite:`Nemirovski.Yudin.1983` (:numref:`subsec_mdl-fixed-budget-rates`). Für nichtkonvexe glatte Zielfunktionen garantiert er $\min_{t \le T} \mathbb{E}[\|\nabla f(\mathbf{x}_t)\|^2] = O(1/\sqrt{T})$ :cite:`Ghadimi.Lan.2013` (:numref:`subsec_mdl-nonconvex-sgd`). Die Mittelung der Iterierten nach Polyak und Ruppert :cite:`Polyak.Juditsky.1992` erreicht die optimale asymptotische Kovarianz (:numref:`subsec_practice-weight-averaging`). Bei endlichen Datensätzen beseitigt **SVRG** :cite:`Johnson.Zhang.2013` (:numref:`sec_mdl-variance-reduction`) die stationäre Kugel um das Minimum. Dazu berechnet das Verfahren in regelmäßigen Abständen Kontrollvariaten auf dem gesamten Datensatz. Datenaugmentierung und Merkmale, die sich rasch ändern, begrenzen jedoch seine Wirksamkeit im Deep Learning.

<!--en:50312ace-->
## Matrixoptimierer und Optimierer für strukturierte Parameter

<!--en:97ac58c1-->
* **Natürlicher Gradient, K-FAC und SOAP:** Der **natürliche Gradient** :cite:`Amari.1998` nutzt die inverse Fisher-Informationsmatrix zur Vorkonditionierung der Aktualisierungen. Damit misst er die Schrittlänge im Raum der Ausgabeverteilungen. **K-FAC** :cite:`Martens.Grosse.2015` approximiert für jede Schicht den Block der Fisher-Informationsmatrix durch das Kronecker-Produkt der Kovarianzmatrizen der Eingaben und der Ausgabegradienten (:numref:`subsec_mdl-preconditioning-ladder`). **SOAP** :cite:`Vyas.Morwani.Zhao.ea.2024` führt Adam in der Eigenbasis der linken und rechten Vorkonditionierer von Shampoo aus :cite:`Gupta.Koren.Singer.2018`.

<!--en:5efd2c20-->
* **Funktionsweise und Skalierung von Muon:** :citet:`Wang.Zhang.Li.ea.2025` führen den Geschwindigkeitsvorteil von Muon gegenüber AdamW darauf zurück, wie die Projektionen im assoziativen Speicher mit seltenen Merkmalen umgehen. Dabei handelt es sich um die Wert- und Ausgabematrizen der Attention sowie die Matrizen des mehrschichtigen Perzeptrons (MLP). Das entspricht dem Vorteil von Adam gegenüber SGD bei seltenen Token :cite:`Kunstner.Chen.Lavington.ea.2023,Kunstner.Yadav.Milligan.ea.2024`. :citet:`Shah.Polloreno.Stratos.ea.2025` zeigen, dass Muon bis zu größeren Batchgrößen dateneffizient bleibt als AdamW. **Scion** :cite:`Pethick.Xie.Antonakopoulos.ea.2025` erweitert den steilsten Abstieg in der Operatornorm auf alle Schichten, einschließlich der Einbettungen und Ausgabeköpfe, ohne auf AdamW zurückzugreifen.

<!--en:88a4ea64-->
* **Effiziente und verteilte Orthogonalisierung:** **Polar Express** :cite:`Amsel.Persson.Musco.ea.2026` und :citet:`Cesista.You.Jordan.2025` optimieren die Koeffizienten des Newton-Schulz-Polynoms für jeden Schritt (:numref:`fig_opt_ns_maps`). :citet:`Kim.Oh.2026` untersuchen die Konvergenz bei näherungsweiser Orthogonalisierung in $q$ Schritten. **Dion** :cite:`Ahn.Xu.Abreu.ea.2025` orthogonalisiert Approximationen des Momentums von niedrigem Rang, um beim auf mehrere Geräte aufgeteilten Training den Datenaustausch zwischen den Geräten zu verringern.

<!--en:aa166138-->
## Lernratenpläne, Mittelung und groß angelegte Benchmarks

<!--en:1080d580-->
* **Warmstarts, Optimierung ohne Lernratenplan und Model Soups:** Ursprünglich kombinierte man die kosinusförmige Absenkung der Lernrate mit periodischen Warmstarts (**SGDR**, :citet:`Loshchilov.Hutter.2016`). Das ermöglicht **Snapshot-Ensembles** :cite:`Huang.Li.Pleiss.ea.2017`, die Vorhersagen aus verschiedenen Zyklen mitteln. **Model Soups** :cite:`Wortsman.Ilharco.Gadre.ea.2022` hingegen mitteln die Gewichte von Modellen, die von derselben Initialisierung aus feinabgestimmt wurden. **Schedule-Free SGD und AdamW** :cite:`Defazio.Yang.Mehta.ea.2024` kommen ohne Lernratenplan mit vorab festgelegtem Horizont $T$ aus. Sie berechnen die Gradienten an einem Punkt, den sie durch Interpolation zwischen der aktuellen Iterierten und dem laufenden Mittelwert der Iterierten bestimmen.

<!--en:13afdcce-->
* **Anstieg der Batchgröße und Skalierungsgesetze:** :citet:`Malladi.Lyu.Panigrahi.ea.2022` leiten aus stochastischen Differentialgleichungen die vollständigen Regeln dafür her, wie sich die Parameter der Momentenschätzung von Adam $(\beta_1, \beta_2, \epsilon)$ bei wachsender Batchgröße skalieren lassen. :citet:`Zhang.Morwani.Vyas.ea.2024` und :citet:`Merrill.Arora.Groeneveld.ea.2025` zeigen, dass die kritische Batchgröße vor allem mit der Zahl der verarbeiteten Token wächst und weniger mit der Modellgröße. :citet:`Dey.Gosal.Chen.ea.2023`, :citet:`Bi.Chen.Chen.ea.2024` und :citet:`Meta.2025` dokumentieren, wie sich Hyperparameter zwischen unterschiedlich großen Sprachmodellen im Produktiveinsatz übertragen lassen.

<!--en:44a96548-->
* **Benchmarks und Vergleichsprotokolle für Optimierer:** Neue Optimierer wie LARS/LAMB :cite:`You.Gitman.Ginsburg.2017` oder NorMuon :cite:`Li.Liu.Liang.ea.2026` scheinen oft Vorteile zu bringen. Stimmt man die Störhyperparameter der Referenzverfahren mit gleichem Aufwand ab, schrumpfen diese Vorteile oft oder verschwinden ganz :cite:`Nado.Gilmer.Shallue.ea.2021,Schmidt.Schneider.Hennig.2021,Wen.Hall.Ma.ea.2025`. Standardisierte Benchmarks wie **AlgoPerf** :cite:`Dahl.Schneider.Nado.ea.2023,Kasimbeg.Schneider.Eschenhagen.ea.2025`, Sammlungen von Benchmarks für Sprachmodelle :cite:`Semenov.Pagliardini.Jaggi.2025` und der Optimierungswettbewerb **NanoGPT speedrun** :cite:`Jordan.2024,Jordan.2026` schreiben ausdrücklich Budgets für die Hyperparameteroptimierung und Schwellen für die statistische Signifikanz vor. Das öffentliche **Trainingslogbuch zu OPT-175B** :cite:`zhang2022opt` dokumentiert Tag für Tag und ungefiltert, wie beim Training in großem Maßstab Verlustspitzen auftraten, wie das Team auf frühere Checkpoints zurückging und wie es die Hardware nach Ausfällen wieder in Betrieb nahm.

<!--en:744960d2-->
[Diskussion](https://d2l.discourse.group/t/36851)
