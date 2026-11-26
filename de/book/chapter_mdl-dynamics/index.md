<!--en:25a68c1e-->
# Dynamik: Differentialgleichungen und generative Flüsse

<!--en:5b5ca6dc-->
Ein Residualnetz addiert in jeder Schicht ein kleines Update zur Eingabe. Beim Gradientenabstieg (engl. gradient descent) bewegen sich die Gewichte Schritt für Schritt bergab. Ein Diffusionsmodell verwandelt Rauschen entlang eines erlernten Vektorfelds in ein Bild. Alle drei Verfahren diskretisieren Differentialgleichungen. Diese geben an, wie schnell sich ein Zustand ändert. Daraus ergeben sich unmittelbar zwei praktische Fragen: Mit wie wenigen Schritten kommt ein Sampler aus? Wie können wir die Geschwindigkeit oder die Richtung des Scores allein aus Beispielbildern lernen?

<!--en:557aa4ba-->
Die zweite Frage lässt sich vom anderen Ende her beantworten: Wir fügen den Trainingsbildern nach und nach Rauschen hinzu. Jedes einzelne Teilchen bewegt sich unter dem Einfluss des Rauschens unvorhersehbar, doch die Dichte der gesamten Teilchenwolke entwickelt sich deterministisch. Kehren wir diese Entwicklung um, gewinnen wir aus gaußschem Rauschen wieder Daten. Die Methoden der bisherigen Kapitel reichen dafür nicht aus. Die Backpropagation differenziert einen festen Berechnungsgraphen statt der Lösung einer gewöhnlichen Differentialgleichung (engl. ordinary differential equation, ODE). Die bisherige statische Betrachtung von Wahrscheinlichkeiten erfasst keine zeitliche Entwicklung von Dichten. Für brownsche Inkremente der Größenordnung $\sqrt{\Delta t}$ versagt zudem die gewöhnliche Kettenregel.

<!--en:67dc6f1b-->
Das Kapitel gliedert sich in drei Teile. Zunächst bestimmen wir exakt und numerisch, wie sich einzelne Zustände unter deterministischen ODEs entwickeln. Die am schnellsten abklingende Eigenkomponente begrenzt die Schrittweite beim Gradientenabstieg und bei jedem expliziten Runge-Kutta-Verfahren (:numref:`sec_mdl-euler-runge-kutta`). Geradere Transportpfade ermöglichen die Generierung in wenigen Schritten (:numref:`sec_mdl-rectified-flow`). Aus linearen ODEs mit externer Anregung ergibt sich dabei die Rekursionsgleichung von Zustandsraummodellen (:numref:`subsec_mdl-ode-driven-linear`, :numref:`sec_ssm`). Von tiefen Residualnetzen (:numref:`sec_resnet`) gelangen wir zu Neural ODEs und kontinuierlichen normalisierenden Flüssen (engl. continuous normalizing flows, :numref:`sec_mdl-neural-odes-cnf`). Im zweiten Teil erhalten wir mit der stochastischen Analysis (engl. stochastic calculus) eine angepasste Kettenregel (:numref:`sec_mdl-sdes`). Mit der Fokker-Planck-Gleichung verfolgen wir die Dichte der Teilchenwolke und sehen, dass wir zur Zeitumkehr nur den *Score* $\nabla \log p_t$ benötigen (:numref:`sec_mdl-fokker-planck-probability-flow`). Im dritten Teil lernen wir den Score oder die marginale Geschwindigkeit direkt durch Kleinste-Quadrate-Regression an verrauschten oder interpolierten Paaren. Dazu benötigen wir keine Feldwerte als Ground Truth (:numref:`sec_mdl-score-matching-diffusion-flow`, :numref:`sec_mdl-flow-matching-transport`).

<!--en:518ab661-->
Wenn Sie direkt zu :numref:`chap_diffusion` gelangen möchten, können Sie die kontinuierliche adjungierte Methode (:numref:`sec_mdl-adjoint-method`) und den optimalen Transport (:numref:`sec_mdl-ot-connection`) beim ersten Lesen überspringen. In den meisten Experimenten testen wir Sampler an Vektorfeldern, die sich in geschlossener Form angeben lassen. So lassen sich Diskretisierungsfehler isoliert untersuchen. Wir setzen Kenntnisse über Eigenwertzerlegungen (:numref:`sec_mdl-eigendecompositions`), Jacobi-Matrizen (:numref:`subsec_mdl-jacobian`) und bedingte gaußsche Verteilungen voraus. Maßtheorie benötigen wir nicht.

<!--en:5f1d3268-->
## Notation {.unnumbered}

<!--en:a2267486-->
:numref:`tab_mdl-dyn-notation` führt die Symbole auf, die in mehreren Abschnitten vorkommen. Jeder Abschnitt erläutert die dort benötigten Symbole noch einmal und definiert die Buchstaben, die nur dort vorkommen. Hat ein Buchstabe zwei Bedeutungen, gibt die entsprechende Tabellenzeile an, wo welche gilt.

<!--en:78248bb3-->
:Notation im Kapitel zur Dynamik.

<!--en:969fc472-->
| Symbol | Bedeutung | Erstmals verwendet in |

<!--en:812595bb-->
| $\mathbf x(t)\in\mathbb R^d$, $\dot{\mathbf x}$ | Zustand und seine zeitliche Ableitung | :numref:`sec_mdl-odes-solvers` |

<!--en:b5c29849-->
| $\mathbf x_0$ | Zustand bei $t=0$, also der Anfangspunkt einer Trajektorie. Beim Verrauschen ist dies ein Datenpunkt | :numref:`sec_mdl-odes-solvers` |

<!--en:ad7aa4f0-->
| $\mathbf f(\mathbf x,t)$ | Vektorfeld einer ODE bzw. Drift einer stochastischen Differentialgleichung (SDE) | :numref:`sec_mdl-odes-solvers`; :numref:`sec_mdl-sdes` |

<!--en:ffce713d-->
| $\Phi_t$; $\Phi_{s\to t}$ | Flussabbildung $\Phi_t(\mathbf x_0)=\mathbf x(t)$ bzw. Abbildung vom Zustand zur Zeit $s$ auf den Zustand zur Zeit $t$ | :numref:`sec_mdl-odes-solvers` |

<!--en:5ba1410a-->
| $L$ | Lipschitz-Konstante von $\mathbf f$ | :numref:`sec_mdl-odes-solvers` |

<!--en:eaf73320-->
| $T$ | Endzeitpunkt (Zeithorizont) | :numref:`sec_mdl-odes-solvers` |

<!--en:1d00be30-->
| $\mathbf A$; $\lambda_j=a_j+\mathrm i\,b_j$; $\mathbf V$, $\boldsymbol\Lambda$ | Matrix des linearen Feldes $\dot{\mathbf x}=\mathbf A\mathbf x$, ihre Eigenwerte, die Matrix der Eigenvektoren und die Diagonalmatrix der Eigenwerte | :numref:`sec_mdl-odes-solvers` |

<!--en:4fe1fce3-->
| $\lambda$ | Eigenwert von $\mathbf A$ oder einer Jacobi-Matrix. Bei zeitumgekehrten Samplern bestimmt der Faktor das Vielfache $\lambda g$ der ursprünglichen Rauschamplitude, das sie beim Hinzufügen von Rauschen verwenden | :numref:`sec_mdl-odes-solvers`; :numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:6d5a6d1a-->
| $\mathbf K$ | Matrix $\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$ der Rotation $\dot{\mathbf x}=\mathbf K\mathbf x$ | :numref:`sec_mdl-odes-solvers` |

<!--en:0435ba08-->
| $\mathbf J$ | Jacobi-Matrix von $\mathbf f$ | :numref:`sec_mdl-odes-solvers` |

<!--en:010cff50-->
| $\mathbf x^\star$ | Fixpunkt eines Feldes mit $\mathbf f(\mathbf x^\star)=\mathbf 0$, bei einem Gradientenfluss ein kritischer Punkt der Verlustfunktion | :numref:`sec_mdl-odes-solvers` |

<!--en:f0264de4-->
| $\mathbf u(t)$, $\mathbf B$ | Eingabe und zugehörige Matrix in $\dot{\mathbf x}=\mathbf A\mathbf x+\mathbf B\mathbf u(t)$ | :numref:`sec_mdl-odes-solvers` |

<!--en:39bf118f-->
| $h$, $t_n=nh$, $\mathbf x_n$ | Schrittweite ($\Delta t$ in :numref:`sec_mdl-sdes` und :numref:`sec_mdl-fokker-planck-probability-flow`), Zeitgitter und Zustand zur Zeit $t_n$ (exakt oder als Iterationswert eines numerischen Verfahrens) | :numref:`sec_mdl-odes-solvers`; :numref:`sec_mdl-euler-runge-kutta` |

<!--en:1ec22aa7-->
| $\bar{\mathbf A}$, $\bar{\mathbf B}$ | Exakter Schritt bei über die Zeitspanne $h$ konstanter Eingabe: $\mathbf x_{n+1}=\bar{\mathbf A}\mathbf x_n+\bar{\mathbf B}\mathbf u_n$ | :numref:`sec_mdl-odes-solvers` |

<!--en:13ffa0e2-->
| $R(z)$, $z=h\lambda$ | Stabilitätsfunktion eines numerischen Verfahrens | :numref:`sec_mdl-euler-runge-kutta` |

<!--en:faee3d21-->
| $\varepsilon_n$ | Globaler Fehler $\lVert\mathbf x_n-\mathbf x(t_n)\rVert$ | :numref:`sec_mdl-euler-runge-kutta` |

<!--en:abda2338-->
| $\mathcal L$, $\eta$, $\mathbf H$ | Verlust, Lernrate, Hesse-Matrix | :numref:`sec_mdl-odes-solvers`; :numref:`sec_mdl-euler-runge-kutta` |

<!--en:303f859c-->
| $N$ | Anzahl der Schritte bis zum Endzeitpunkt. Das numerische Verfahren durchläuft die Schritte $n=1,\ldots,N$ mit $Nh=T$ (die ersten beiden Abschnitte bezeichnen auch diese Anzahl mit $n$). In :numref:`sec_mdl-score-matching-diffusion-flow` zugleich die Anzahl der Rauschpegel $t=1,\ldots,N$ des DDPM | :numref:`sec_mdl-neural-odes-cnf` |

<!--en:cd91af6e-->
| $\boldsymbol\theta$ | Netzparameter | :numref:`sec_mdl-neural-odes-cnf` |

<!--en:80068c08-->
| $\mathbf a(t)$ | Adjungierter Zustand | :numref:`sec_mdl-neural-odes-cnf` |

<!--en:7240586f-->
| $p_t$ | Dichte des Zustands zur Zeit $t$ | :numref:`sec_mdl-neural-odes-cnf` |

<!--en:4bc87d8f-->
| $\boldsymbol\epsilon$ | In :numref:`sec_mdl-neural-odes-cnf` der zufällige Testvektor des Spurschätzers, ab :numref:`sec_mdl-sdes` standardnormalverteiltes Rauschen wie in $\mathbf x_t=\alpha_t\mathbf x_0+\sigma_t\boldsymbol\epsilon$. In :numref:`sec_mdl-flow-matching-transport` bezeichnet auch $\mathbf x_1$ dieses Rauschen als Endpunkt des Pfades | :numref:`sec_mdl-neural-odes-cnf`; :numref:`sec_mdl-sdes` |

<!--en:5d65b758-->
| $\mathbf X_t$ ($X_t$ in einer Dimension) | Zustand eines stochastischen Prozesses zur Zeit $t$, also eine Zufallsvariable. $\mathbf x$ ist ein Wert, den sie annimmt | :numref:`sec_mdl-sdes` |

<!--en:67e1a8a6-->
| $\mathbf W_t$ | Brownsche Bewegung | :numref:`sec_mdl-sdes` |

<!--en:38783c60-->
| $g$, $g(t)$; $\mathbf G$ | Konstante oder zeitabhängige Rauschamplitude einer SDE (ihr Diffusionskoeffizient) bzw. Rauschmatrix | :numref:`sec_mdl-sdes` |

<!--en:7745aba9-->
| $\theta$ | Rate des OU-Prozesses $dX=-\theta X\,dt+g\,dW$, im skalaren Beispiel aus :numref:`sec_mdl-neural-odes-cnf` der Parameter von $\dot x=\theta x$ | :numref:`sec_mdl-neural-odes-cnf`; :numref:`sec_mdl-sdes` |

<!--en:b708859e-->
| $\alpha_t$, $\sigma_t$ | Faktor für den bedingten Mittelwert und bedingte Standardabweichung von $\mathbf X_t$ bei gegebenem $\mathbf X_0$. Dabei ist $\sigma_t$ der Rauschpegel | :numref:`sec_mdl-sdes` |

<!--en:9116a1ed-->
| $\phi$ | Glatte Funktion des Zustands, auf die wir das Lemma von Itô anwenden und die bei der Herleitung der Fokker-Planck-Gleichung als Testfunktion dient (an anderer Stelle ein Winkel) | :numref:`sec_mdl-sdes` |

<!--en:fcaf06cd-->
| $\psi$ | Glattes Skalarfeld in den Grundlagen zur Vektoranalysis | :numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:d47a0041-->
| $\mathbf s_t=\nabla\log p_t$; $\mathbf s_{\boldsymbol\theta}$ | Score bzw. erlernter Score | :numref:`sec_mdl-fokker-planck-probability-flow`; :numref:`sec_mdl-score-matching-diffusion-flow` |

<!--en:0d1ec1c3-->
| $\mathbf u_t$; $\mathbf u_{\boldsymbol\theta}$ | Geschwindigkeit, mit der sich $p_t$ bewegt (Geschwindigkeit des Wahrscheinlichkeitsflusses), ohne Bezug zur Eingabe $\mathbf u(t)$. Das zweite Symbol bezeichnet die erlernte Geschwindigkeit | :numref:`sec_mdl-fokker-planck-probability-flow`; :numref:`sec_mdl-flow-matching-transport` |

<!--en:89139d0b-->
| $\mathbf j$ | Wahrscheinlichkeitsstrom | :numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:fff21276-->
| $\bar t=T-t$ | Rückwärts laufende Zeit | :numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:2518d8d3-->
| $w_k$, $r_k(\mathbf x)$, $v_k$ | Mischungsgewichte, Zugehörigkeitswahrscheinlichkeiten, Varianzen der Komponenten | :numref:`sec_mdl-fokker-planck-probability-flow` |

<!--en:d7526930-->
| $\beta(t)$; $\beta_t$, $\bar\alpha_t$ | Rate der varianzerhaltenden SDE, Rauschplan des DDPM (ungefähr $\beta(t)$ mal die Schrittweite, als $\beta_k$ geschrieben in :numref:`sec_mdl-sdes`) und sein fortlaufendes Produkt $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$, das unserem $\alpha_t^2$ entspricht | :numref:`sec_mdl-sdes`; :numref:`sec_mdl-score-matching-diffusion-flow` |

<!--en:c6687a99-->
| $\bar{\mathbf x}_t$, $\bar\sigma_t$ | DDIM-Koordinaten: der umskalierte Punkt $\mathbf x_t/\sqrt{\bar\alpha_t}$ und das zugehörige Verhältnis von Rauschen zu Signal $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ | :numref:`sec_mdl-score-matching-diffusion-flow` |

<!--en:0175a508-->
| $w(t)$; $\gamma$ | Gewicht des Rauschpegels $t$ im Verlust bzw. Guidance-Skala | :numref:`sec_mdl-score-matching-diffusion-flow` |

<!--en:32b140c1-->
| $\pi$ | Kopplung, also eine gemeinsame Verteilung von Daten-Rausch-Paaren, auch Paarung genannt | :numref:`sec_mdl-flow-matching-transport` |

<!--en:cc94fe75-->
| $\mathrm i$, $\mathbf I$ | Imaginäre Einheit (aufrecht gesetzt) und Einheitsmatrix | im gesamten Kapitel |

<!--en:109c4cb5-->
Dynamik: Differentialgleichungen, stochastische Flüsse und Transport<br>**ODEs und numerische Verfahren · Neural ODEs · Itô-SDEs · Fokker-Planck-Gleichung · Score-Matching · Flow-Matching**

<!--en:525812b2-->
::: {.slide title="Warum Deep Learning zeitkontinuierliche Dynamik braucht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:739b4546-->
- **Architekturen und Optimierer als Euler-Schritte:** Ein ResNet-Block $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathbf{f}(\mathbf{x}_l)$ ($h=1$) und der Gradientenabstieg $\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\nabla\mathcal{L}$ ($h=\eta$) entsprechen je einem expliziten Euler-Schritt für eine ODE. Für die Lernrate gilt die Schranke $\eta < 2/\lambda_{\max}(\mathbf{H})$.

<!--en:d38a9c2d-->
- **Generatives Sampling durch Umkehr des Verrauschens:** Brownsche Stöße ($\sim \sqrt{dt}$) glätten die Dichte $p_t(\mathbf{x})$ deterministisch gemäß der Fokker-Planck-Gleichung. Die Zeitumkehr führt von $\mathcal{N}(\mathbf{0}, \mathbf{I})$ zurück zu $p_{\text{data}}$.

<!--en:52e627fa-->
::: {.slide title="Die drei mathematischen Etappen in Kapitel §{num:chap_mdl-dynamics}"}

<!--en:c8dfc67f-->
[Aufbau des Kapitels]{.kicker}

<!--en:6cc56c3f-->
- **1. Deterministische ODEs und Flüsse (§§{sec_mdl-odes-solvers}–§{num:sec_mdl-neural-odes-cnf}):** Eindeutigkeit nach Picard-Lindelöf, Matrixexponential $e^{\mathbf{A}t}$, Stabilität von Runge-Kutta-Verfahren $R(h\lambda)$, kontinuierliche adjungierte ODE und die Spurformel von Liouville $\frac{d}{dt}\log p_t = -\operatorname{tr}\mathbf{J}$.

<!--en:fd6d1ff6-->
- **2. Stochastische Analysis und Dichteentwicklung (§§{sec_mdl-sdes}–§{num:sec_mdl-fokker-planck-probability-flow}):** Quadratische Variation $(dW)^2 = dt$, Lemma von Itô, gaußsche Übergangskerne des OU-Prozesses, Fokker-Planck-Gleichung, zeitumgekehrte SDE nach Anderson und Wahrscheinlichkeitsfluss-ODE.

<!--en:6aa66c40-->
- **3. Generative Felder lernen (§§{sec_mdl-score-matching-diffusion-flow}–§{num:sec_mdl-flow-matching-transport}):** Denoising-Score-Matching nach Vincent, Tweedies Formel, DDPM/DDIM, bedingtes Flow-Matching und Reflow.

<!--en:23ef19e0-->
::: {.slide title="Wiederkehrende Testfälle in diesem Kapitel"}

<!--en:05adf916-->
[Roter Faden]{.kicker}

<!--en:eee4fa07-->
- **Lineare und steife ODEs:** An einem stabilen Strudel in der Ebene ($\lambda = -0.5 \pm \mathrm{i}$) und einem steifen System mit zwei abklingenden Eigenkomponenten ($\mathrm{diag}(-50, -1)$) prüfen wir $e^{\mathbf{A}t}$, die Ordnung von RK4, adjungierte Variablen und Schrittweitengrenzen expliziter Verfahren.

<!--en:0f0792e5-->
- **Stochastische und multimodale Testfälle:** An der Ornstein-Uhlenbeck-SDE ($dX = -\theta X\,dt + g\,dW$) und einer bimodalen Gauß-Mischung prüfen wir exakte Übergangskerne, Scores, zeitumgekehrte SDEs und Schrittweiten beim DDIM-Sampling.

<!--en:83aba0c8-->
- **Transport in zwei Dimensionen:** Wir paaren die Two-Moons-Daten mit Stichproben aus $\mathcal{N}(\mathbf{0}, \mathbf{I})$. Daran untersuchen wir, wo sich Strecken kreuzen, welche Untergrenze $C$ beim bedingten Flow-Matching gilt und wie Reflow die Pfade begradigt.

<!--en:3cbdd1b1-->
::: {.slide title="Gemeinsame Gleichungen für Teilchen, Dichten und Zeitumkehr"}

<!--en:39418420-->
[Zentrale Zusammenhänge]{.kicker}

<!--en:9e3e68f4-->
- **Vorwärts- und zeitumgekehrte SDE:** Aus $d\mathbf{X} = \mathbf{f}\,dt + g\,d\mathbf{W}$ ergibt sich bei Zeitumkehr $d\mathbf{X} = [\mathbf{f} - g^2\nabla\log p_t]\,dt + g\,d\bar{\mathbf{W}}$.

<!--en:30d78ab6-->
- **Deterministische Wahrscheinlichkeitsfluss-ODE:** $\dot{\mathbf{x}} = \mathbf{u}_t(\mathbf{x}) = \mathbf{f}(\mathbf{x}, t) - \tfrac12 g(t)^2\nabla\log p_t(\mathbf{x})$ hat zu jeder Zeit dieselbe Randdichte $p_t(\mathbf{x})$ wie die SDE.

<!--en:ea5f1305-->
- **Zentrale Identität:** Die Umformung $\tfrac12 g^2\Delta p_t = \nabla\cdot(p_t \cdot \tfrac12 g^2\nabla\log p_t)$ beschreibt die Diffusion durch den Laplace-Operator als deterministischen Transport entlang des Score-Feldes.

<!--en:09587a8c-->
::: {.slide title="Voraussetzungen und Lesewege durch Kapitel §{num:chap_mdl-dynamics}"}

<!--en:123f19cf-->
[Lesehilfe]{.kicker}

<!--en:dcabe51c-->
- **Mathematische Voraussetzungen:** Eigenwertzerlegungen und Spektralnormen (§{chap_mdl-linear-algebra}), Jacobi-Matrizen von Funktionen mehrerer Variablen, Divergenz und partielle Integration (§{chap_mdl-calculus}) sowie bedingte Erwartungswerte bei gaußschen Verteilungen (§{chap_mdl-probability-statistics}). Maßtheorie ist nicht erforderlich.

<!--en:f4de0fea-->
- **Optimierung und Zustandsraummodelle (§{sec_mdl-odes-solvers}–§{num:sec_mdl-euler-runge-kutta}):** Exakte Diskretisierungen mit einem Halteglied nullter Ordnung für Zustandsraummodelle (Mamba/S4), transientes Wachstum durch Nichtnormalität beim Momentum-Verfahren und numerische Steifheit beim Gradientenabstieg.

<!--en:07b6eb34-->
- **Direkter Weg zu Diffusions- und Flussmodellen:** Lesen Sie **§{sec_mdl-odes-solvers}–§{num:sec_mdl-euler-runge-kutta}** zu Flüssen und Runge-Kutta-Verfahren, **§{sec_mdl-sdes}** zum Lemma von Itô und zum Übergangskern des Ornstein-Uhlenbeck-Prozesses, **§{sec_mdl-fokker-planck-probability-flow}** zu zeitumgekehrten SDEs und zur Wahrscheinlichkeitsfluss-ODE sowie **§{sec_mdl-score-matching-diffusion-flow}–§{num:sec_mdl-flow-matching-transport}** zu Score-Matching und Flow-Matching.

<!--en:6fc2ec71-->
- **Zur Vertiefung:** Wenn Ihr Ziel die Bilderzeugung ist, können Sie die kontinuierliche adjungierte Methode (**§{sec_mdl-neural-odes-cnf}**) und die Theorie des optimalen Transports nach Wasserstein (**§{sec_mdl-flow-matching-transport}**) beim ersten Lesen überspringen.
