<!--en:16ebb42e-->
# Optimierung

<!--en:38ad263e-->
Wir trainieren jedes neuronale Netz in diesem Buch nach demselben Schema. Wir berechnen den Gradienten des Verlusts, gehen einen Schritt in die entgegengesetzte Richtung und wiederholen den Vorgang. In :numref:`chap_optimization` haben wir an realen Netzen untersucht, wie sich Momentum, adaptive Schrittweiten (engl. adaptive step sizes) und Lernratenpläne (engl. learning rate schedules) auf diesen Ablauf auswirken. Drei Fragen blieben offen. Warum hat die Lernrate eine feste Obergrenze? Warum kommen wir selbst mit einer Schrittweite deutlich unter dieser Grenze manchmal nur mühsam voran? Und warum verhindert das Gradientenrauschen (engl. gradient noise), dass das Training zur Ruhe kommt?

<!--en:8b34c7c6-->
Die Optimierungstheorie beantwortet alle drei Fragen anhand der Geometrie der Zielfunktion (engl. objective). Die größte Krümmung bestimmt die größte sichere Schrittweite. Das Verhältnis der größten zur kleinsten Krümmung bestimmt die Zahl der Iterationen. Von der Konvexität hängt ab, ob ein stationärer Punkt global optimal ist. In diesem Kapitel betrachten wir immer wieder die schmale Mulde $\tfrac12(x_1^2 + 1000\,x_2^2)$. Sie ist in einer Richtung tausendmal steiler als in der anderen. Mit einer Schrittweite, die an der steilen Flanke sicher ist, kommen wir in der flachen Talrichtung kaum voran. Während in :numref:`chap_optimization` die Praxis und die Messungen im Vordergrund stehen, erläutern wir hier die zugrunde liegende Geometrie und geben in sich geschlossene Beweise.

<!--en:209bc91b-->
Das Kapitel besteht aus drei Teilen. Im ersten Teil arbeiten wir mit exakten Gradienten. In :numref:`sec_mdl-gradient-based-optimization` leiten wir die Obergrenze für die Schrittweite und die Anzahl der Iterationen her. Wir zeigen, wie Momentum das Krümmungsverhältnis durch seine Quadratwurzel ersetzt. In :numref:`sec_mdl-convexity` klären wir, wann stationäre Punkte globale Minima sind. Dazwischen behandeln wir in :numref:`sec_mdl-curvature-methods` Newton-Verfahren, Quasi-Newton-Verfahren und Trust-Region-Verfahren, die die Krümmung explizit modellieren. Im zweiten Teil geht es um verrauschte Gradienten. In :numref:`sec_mdl-stochastic-gradient-methods` zeigen wir, warum der stochastische Gradientenabstieg (engl. stochastic gradient descent, SGD) bei fester Schrittweite eine rauschbedingte Untergrenze (engl. noise floor) erreicht, die proportional zur Schrittweite ist. In :numref:`sec_mdl-adaptive-stochastic-methods` vergleichen wir die koordinatenweise Skalierung und AdamW mit Heavy-Ball-Momentum. Im dritten Teil entwickeln wir in :numref:`sec_mdl-constrained-optimization-duality` und :numref:`sec_mdl-lagrangian-duality` die KKT-Bedingungen, Projektionen, proximalen Schritte und die Dualität, die Normschranken zugrunde liegen (:numref:`sec_weight_decay`). In :numref:`sec_mdl-numerical-stability-conditioning` untersuchen wir schließlich Gleitkommagenauigkeit, Auslöschung und Konditionierung (:numref:`sec_numerics`).

<!--en:a64aa2fc-->
Wir setzen Kenntnisse über Gradienten und Hesse-Matrizen (:numref:`sec_mdl-multivariable_calculus`) sowie über Eigenwertzerlegungen (:numref:`sec_mdl-eigendecompositions`) voraus. Die Verlustfunktionen tiefer Netze sind weder global glatt noch konvex. Daher dienen unsere Sätze als lokale Modelle, deren Grenzen wir in jedem Abschnitt ausdrücklich angeben. Einen Überblick über Innere-Punkte-Verfahren und Verfahren zur Operatorzerlegung gibt :numref:`sec_mdl-optimization-further-reading`.

<!--en:5f1d3268-->
## Notation {.unnumbered}

<!--en:b82469ac-->
Im gesamten Kapitel verwenden wir die Konventionen aus :numref:`tab_mdl-opt-notation`. In jedem Abschnitt erinnern wir an die dort am häufigsten verwendeten Symbole. Bezeichnungen, die nur in einem Abschnitt vorkommen, führen wir bei der ersten Verwendung ein.

<!--en:a81b9720-->
:Symbole im Kapitel zur Optimierung. Ein Stern $\star$ bezeichnet ein Optimum, ein Sternchen $*$ eine konvex Konjugierte. Matrizen sind fett gesetzt.

<!--en:204b3fa2-->
| Symbol | Bedeutung |

<!--en:49e186c6-->
| $f$, $\ell(\mathbf{w})$ | Zielfunktion bzw. Trainingsverlust als Funktion der Gewichte |

<!--en:df138c29-->
| $\mathbf{x}^\star$, $f^\star$ | ein Minimierer und der minimale Wert |

<!--en:24179639-->
| $\nabla f$, $\nabla^2 f$, $\mathbf{g}_k$ | Gradient, Hesse-Matrix und der in Schritt $k$ verwendete Gradient (oder stochastische Gradient) |

<!--en:c5aa368b-->
| $L$, $\mu$ | Konstanten der Glattheit und der starken Konvexität mit $\mu\mathbf{I} \preceq \nabla^2 f \preceq L\mathbf{I}$ |

<!--en:95d715eb-->
| $\kappa = L/\mu$; $\kappa(\mathbf{A})$ | Konditionszahl des Problems bzw. einer Matrix, im zweiten Fall $\sigma_{\max}(\mathbf{A})/\sigma_{\min}(\mathbf{A})$ |

<!--en:ed4d4734-->
| $\lambda_i$, $\lambda_{\max}(\mathbf{A})$ | Eigenwerte einer Hesse-Matrix oder der Matrix $\mathbf{A}$ einer quadratischen Funktion |

<!--en:e8b62e6e-->
| $\eta$, $\beta$ | Schrittweite (Lernrate) und Momentum-Koeffizient |

<!--en:fea0fdff-->
| $k$, $t$ | Iterationsindex bzw. Schritt des Optimierers bei adaptiven Verfahren |

<!--en:87b65958-->
| $\epsilon$ | angestrebte Genauigkeit, bei Adam die Konstante, die eine Division durch null verhindert |

<!--en:d63c6b79-->
| $n$, $d$, $\mathcal{B}$, $b$ | Anzahl der Beispiele, Anzahl der Variablen, ein Minibatch und seine Größe |

<!--en:5aaef151-->
| $\boldsymbol{\Sigma}$ | Kovarianz des Gradienten eines einzelnen Beispiels |

<!--en:78ad6c38-->
| $\mathbf{m}_t$, $\mathbf{v}_t$; $\beta_1$, $\beta_2$ | Schätzungen des ersten und zweiten Moments bei Adam sowie ihre Abklingraten |

<!--en:d73d4dd2-->
| $\mathbf{P}$ | Vorkonditionierer für den Schritt $-\eta\,\mathbf{P}^{-1}\mathbf{g}$ |

<!--en:4f3048e3-->
| $g_i$, $h_j$ | Nebenbedingungen in Form von Ungleichungen $g_i(\mathbf{x}) \le 0$ bzw. Gleichungen $h_j(\mathbf{x}) = 0$ |

<!--en:ca9eefa8-->
| $\lambda$, $\nu$ | Multiplikatoren der Ungleichungs- und Gleichungsnebenbedingungen. Zugleich gewichtet $\lambda$ den Term $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ für den Gewichtszerfall. |

<!--en:ff8754d1-->
| $\mathcal{L}$, $q$ | Lagrange-Funktion und Dualfunktion |

<!--en:4cce05c1-->
| $p^\star$, $d^\star$ | Optimalwerte des primalen und des dualen Problems |

<!--en:0578cae4-->
| $f^*$ | konvex Konjugierte von $f$ |

<!--en:a986ab6b-->
| $\Pi_C$, $\mathrm{prox}_\psi$ | euklidische Projektion auf eine Menge $C$ und Proximaloperator von $\psi$ |

<!--en:d6b729a4-->
| $u$, $\varepsilon_{\mathrm{mach}}$ | Rundungseinheit und Maschinenepsilon eines Gleitkommaformats mit $u = \varepsilon_{\mathrm{mach}}/2$ |

<!--en:4901a14e-->
Mathematik für Deep Learning: Optimierung<br>**Glattheit · Konditionszahl · Momentum · Krümmung · Konvexität · stochastisches Rauschen · Vorkonditionierung · Dualität · Numerik**

<!--en:6c116032-->
::: {.slide title="Drei Fragen, die jede Trainingsschleife offenlässt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:5e9d67b7-->
- **Warum hat die Schrittweite eine feste Obergrenze?** Ist die Schrittweite in der steilsten Richtung größer als $2/L$, landen wir an der gegenüberliegenden Talflanke höher als zuvor. Der Verlust steigt.

<!--en:127c121a-->
- **Warum kommen wir mit sicheren Schritten kaum voran?** Bei einem Krümmungsverhältnis von $\kappa = L/\mu$ begrenzt die steilste Flanke die Schrittweite $\eta$. In der flachsten Richtung stockt der Abstieg.

<!--en:35f258cb-->
- **Warum bleibt der Verlust bei Minibatch-Rauschen über $f^\star$?** Bei konstanter Schrittweite pendelt sich SGD in einer Kugel um das Minimum ein. Der stationäre Verlust ist dort $\propto \eta\sigma^2/b$.

<!--en:ee854171-->
::: {.slide title="Ein schlecht konditioniertes Tal als durchgängiger Benchmark"}

<!--en:311edefe-->
[Roter Faden]{.kicker}

<!--en:1bc16447-->
Wir prüfen jeden Satz am Beispiel $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ ($\mu = 1, L = 1000, \kappa = 1000$):

<!--en:278d6061-->
- **Schrittzahl bei Verfahren erster Ordnung:** Der Gradientenabstieg (GD) mit fester Schrittweite braucht $6{,}908$ Schritte ($O(\kappa)$), Heavy-Ball-Momentum und Nesterov-Momentum brauchen $315$–$508$ Schritte ($O(\sqrt{\kappa})$).

<!--en:c7a78c17-->
- **Vorkonditionierung und Drehung:** Das exakte Newton-Verfahren erreicht das Minimum in $1$ Schritt. Bei achsenparallelem Tal erreicht Adam $f < 10^{-8}$ in $22$–$48$ Schritten. Dafür braucht GD $6{,}160$ Schritte. Eine Drehung um $45^\circ$ macht die diagonale Skalierung unwirksam.

<!--en:09955639-->
- **Rauschbedingte Untergrenze:** Bei konstanter Schrittweite stagniert SGD ($\eta = 10^{-3}, \sigma^2 = 1$) bei einem Verlust von $7.5 \times 10^{-4}$.

<!--en:df538ea7-->
::: {.slide title="Die drei Teile von Kapitel §{num:chap_mdl-optimization}"}

<!--en:99e0030a-->
[Kapitelübersicht]{.kicker}

<!--en:e26d048f-->
- **Teil I: Exakte Gradienten und Geometrie (§§{sec_mdl-gradient-based-optimization}–§{num:sec_mdl-convexity}):** Abstiegslemma für $L$-glatte Funktionen, Abhängigkeit der Schrittzahl vom Krümmungsverhältnis ($\kappa$ ohne Momentum, $\sqrt{\kappa}$ mit Momentum), Newton-Verfahren, BFGS, Trust-Region-Verfahren und Garantien bei Konvexität.

<!--en:f19071c4-->
- **Teil II: Stochastische und adaptive Verfahren (§§{sec_mdl-stochastic-gradient-methods}–§{num:sec_mdl-adaptive-stochastic-methods}):** Varianz proportional zu $1/b$, stationäre rauschbedingte Untergrenzen, Lernratenpläne nach Robbins und Monro, AdamW und Vorkonditionierung mit Matrizen (K-FAC, Muon).

<!--en:d86b44fa-->
- **Teil III: Nebenbedingungen, Dualität und Numerik (§§{sec_mdl-constrained-optimization-duality}–§{num:sec_mdl-numerical-stability-conditioning}):** KKT-Bedingungen, Projektionen, das proximale Verfahren ISTA, Fenchel-Dualität und Konditionierung in der Gleitkommaarithmetik.

<!--en:8a49cc84-->
::: {.slide title="Mathematische Größen und ihre Notation"}

<!--en:edb10563-->
[Symbolübersicht]{.kicker}

<!--en:77ffc341-->
- **Krümmung und Konditionierung:**
  - $L$, $\mu$: Obere Krümmungsschranke aus der Glattheit und untere Krümmungsschranke aus der starken Konvexität, $\mu\mathbf{I} \preceq \nabla^2 f(\mathbf{x}) \preceq L\mathbf{I}$.
  - $\kappa = L/\mu$: Konditionszahl des Problems. Für eine Matrix $\mathbf{A}$ gilt $\kappa(\mathbf{A}) = \sigma_{\max}(\mathbf{A})/\sigma_{\min}(\mathbf{A})$.
  - $\mathbf{P}$: Vorkonditionierer in Matrixform, der den Schritt zu $-\eta \mathbf{P}^{-1}\mathbf{g}_k$ umskaliert.

<!--en:931f118c-->
- **Zustandsgrößen stochastischer und adaptiver Verfahren:**
  - $\boldsymbol{\Sigma}$, $b$: Kovarianz des Gradienten eines einzelnen Beispiels und Minibatchgröße (Rauschleistung $\operatorname{tr}\boldsymbol{\Sigma}/b$).
  - $\mathbf{m}_t, \mathbf{v}_t$: Gleitende Mittelwerte zur Schätzung des ersten und zweiten Moments mit den Abklingfaktoren $\beta_1, \beta_2$.

<!--en:982b93c9-->
- **Nebenbedingungen, Dualität und Operatoren:**
  - $g_i(\mathbf{x}) \le 0$, $h_j(\mathbf{x}) = 0$: Ungleichungs- und Gleichungsnebenbedingungen mit den Multiplikatoren $\lambda_i \ge 0$ und $\nu_j$.
  - $\mathcal{L}$, $q$, $p^\star$, $d^\star$: Lagrange-Funktion, Dualfunktion, primales Optimum und duales Optimum ($d^\star \le p^\star$).
  - $f^*$, $\Pi_C$, $\mathrm{prox}_\psi$: Konvex Konjugierte, euklidische Projektion auf $C$ und Proximaloperator.

<!--en:bbf53d98-->
- **Gleitkommagenauigkeit:**
  - $\varepsilon_{\mathrm{mach}} = 2^{-p}$, $u = 2^{-p-1}$: Maschinenepsilon und Rundungseinheit bei $p$ Mantissenbits.

<!--en:15d021a2-->
::: {.slide title="Lesewege durch das Kapitel"}

<!--en:123f19cf-->
[Lesehilfe]{.kicker}

<!--en:b0bf7ff5-->
- **Grundlagen (§{sec_mdl-gradient-based-optimization} und §{sec_mdl-convexity}):** Beginnen Sie hier mit der Schrittweitenobergrenze $2/L$, der Beschleunigung um den Faktor $\sqrt{\kappa}$ durch Momentum, unteren Schranken durch Tangenten, der Jensen-Ungleichung und globalen Konvergenzraten.

<!--en:cd160025-->
- **Häufige Verfahren im Deep Learning (§{sec_mdl-stochastic-gradient-methods} und §{sec_mdl-adaptive-stochastic-methods}):** Lesen Sie anschließend über rauschbedingte Untergrenzen bei SGD, Kosinus- und WSD-Lernratenpläne, die Skaleninvarianz von Adam, den entkoppelten Gewichtszerfall bei AdamW und Muon.

<!--en:a87c8e25-->
- **Vertiefung nach Bedarf (§{sec_mdl-curvature-methods}, §§{sec_mdl-constrained-optimization-duality}–§{num:sec_mdl-numerical-stability-conditioning}):** Hier finden Sie L-BFGS und Trust-Region-Verfahren, KKT-Bedingungen, Projektionen und ISTA, Fenchel-Dualität sowie numerische Stabilität bei `bf16`.
