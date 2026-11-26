<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=el rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Συνελίξεις για Εικόνες

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
Στην ενότητα :numref:`sec_why-conv`, εξαγάγαμε θεωρητικά το συνελικτικό επίπεδο από δύο γεωμετρικές αρχές: την ισομεταβλητότητα ως προς τη μετατόπιση και την τοπικότητα. Τώρα θα δούμε τι υπολογίζει στην πράξη αυτός ο τύπος πάνω σε μια εικόνα, καθώς και αν η κατάβαση κλίσης μπορεί να ανακαλύψει χρήσιμους πυρήνες αυτόματα από τα δεδομένα.

<!--en:55f0a41b kind=paragraph-->
Consider locating vertical edges where a bright region meets a dark one.
Subtracting each pixel's right-hand neighbor ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)
yields zero inside uniform regions, $+1$ where white transitions to black, and
$-1$ where black returns to white. Sweeping the two-weight detector $(1, -1)$
across every valid image location produces an *edge map*. In this section, we
implement two-dimensional cross-correlation, test our hand-crafted edge
detector, let gradient descent *learn* $(1, -1)$ directly from data, show how
a convolution becomes a single matrix multiplication (*im2col*), and
derive how stacking layers widens a unit's receptive field.
<!--tr:55f0a41b-->
Ας εξετάσουμε τον εντοπισμό κατακόρυφων ακμών όπου μια φωτεινή περιοχή συναντά μια σκοτεινή. Αφαιρώντας από κάθε εικονοστοιχείο τον δεξιό του γείτονα ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$), προκύπτει μηδέν μέσα στις ομοιόμορφες περιοχές, $+1$ όπου το λευκό μεταβαίνει στο μαύρο και $-1$ όπου το μαύρο επιστρέφει στο λευκό. Σαρώνοντας τον ανιχνευτή δύο βαρών $(1, -1)$ σε κάθε έγκυρη θέση της εικόνας, παράγεται ένας *χάρτης ακμών* (edge map). Σε αυτή την ενότητα, υλοποιούμε τη δισδιάστατη ετεροσυσχέτιση, δοκιμάζουμε τον χειροποίητο ανιχνευτή ακμών μας, αφήνουμε την κατάβαση κλίσης να *μάθει* το $(1, -1)$ απευθείας από τα δεδομένα, δείχνουμε πώς μια συνέλιξη μετατρέπεται σε έναν μόνο πολλαπλασιασμό πινάκων (*im2col*) και εξάγουμε πώς η στοίβαξη επιπέδων διευρύνει το υποδεκτικό πεδίο μιας μονάδας.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Ετεροσυσχέτιση

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### Η Πράξη

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Αφήνοντας προς το παρόν κατά μέρος τα κανάλια, έστω ότι η είσοδος $\mathbf{X}$ είναι ένας πίνακας $n_\textrm{h} \times n_\textrm{w}$ και ο πυρήνας $\mathbf{K}$ είναι ένας πίνακας $k_\textrm{h} \times k_\textrm{w}$. Ο πυρήνας ορίζει ένα ολισθαίνον *παράθυρο* σχήματος $k_\textrm{h} \times k_\textrm{w}$. Τοποθετώντας την πάνω αριστερή γωνία του παραθύρου στη θέση $(i, j)$, πολλαπλασιάζουμε το καλυπτόμενο τεμάχιο εισόδου στοιχείο προς στοιχείο με τον $\mathbf{K}$ και αθροίζουμε τα γινόμενα για να παράγουμε το στοιχείο εξόδου στη θέση $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Αυτή είναι η *δισδιάστατη ετεροσυσχέτιση* (two-dimensional cross-correlation) του $\mathbf{X}$ με τον $\mathbf{K}$, η οποία γράφεται $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ με δείκτες που ξεκινούν από το μηδέν ώστε να αντιστοιχούν στον κώδικα. Ενώ η ενότητα :numref:`sec_why-conv` κεντράριζε τις μετατοπίσεις συμμετρικά από $-\Delta$ έως $\Delta$, η αγκύρωση στην πάνω αριστερή γωνία διαφέρει μόνο κατά μια σταθερή μετατόπιση $\Delta$, ενώ παράλληλα χειρίζεται φυσικά πυρήνες άρτιου μεγέθους (όπως έναν πυρήνα $2 \times 2$ ή τον ανιχνευτή ακμών μας $1 \times 2$) που δεν διαθέτουν κεντρικό εικονοστοιχείο.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Δισδιάστατη ετεροσυσχέτιση. Τα σκιασμένα τμήματα είναι το πρώτο στοιχείο εξόδου καθώς και τα στοιχεία εισόδου και πυρήνα που χρησιμοποιούνται για τον υπολογισμό του: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
Στο :numref:`fig_correlation`, ολισθαίνουμε έναν πυρήνα $2 \times 2$ κατά μήκος μιας εισόδου $3 \times 3$ ένα εικονοστοιχείο τη φορά, από αριστερά προς τα δεξιά και από πάνω προς τα κάτω. Το παράθυρο χωρά ακριβώς σε τέσσερις θέσεις, δίνοντας την έξοδο $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Παρατηρήστε ότι η έξοδος συρρικνώνεται ελαφρώς κατά μήκος κάθε άξονα επειδή ο πυρήνας πρέπει να παραμένει εξ ολοκλήρου εντός της εισόδου. Η τοποθέτηση ενός παραθύρου πλάτους $k$ σε μήκος $n$ αφήνει $n - k + 1$ έγκυρες θέσεις, επομένως το σχήμα εξόδου είναι

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Έτσι, μια είσοδος $3 \times 3$ με έναν πυρήνα $2 \times 2$ δίνει $2 \times 2$, μια εικόνα $6 \times 8$ με έναν πυρήνα $1 \times 2$ δίνει $6 \times 7$ και μια εικόνα Fashion-MNIST $28 \times 28$ με έναν πυρήνα $5 \times 5$ δίνει $24 \times 24$. Όποτε θέλουμε να διατηρήσουμε τις χωρικές διαστάσεις, περιβάλλουμε την είσοδο με ένα πλαίσιο από μηδενικά (*συμπλήρωση*, padding), την οποία εξετάζουμε μαζί με τους διασκελισμούς και τη διαστολή στην ενότητα :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Υλοποίηση

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Υλοποιούμε τη σχέση :eqref:`eq_corr2d` απευθείας στη συνάρτηση `corr2d` χρησιμοποιώντας ένα ζεύγος βρόχων για κάθε συντεταγμένη εξόδου.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Για να επαληθεύσουμε τη `corr2d` σε σχέση με τον υπολογισμό με το χέρι στο :numref:`fig_correlation`, κατασκευάζουμε τον τανυστή εισόδου `X` διαστάσεων $3 \times 3$ και τον πυρήνα `K` διαστάσεων $2 \times 2$ και υπολογίζουμε το `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Η έξοδος αναπαράγει το :numref:`fig_correlation` στοιχείο προς στοιχείο (και όπως φάνηκε στην ενότητα :numref:`sec_why-conv`, η αυστηρή μαθηματική συνέλιξη με έναν αντεστραμμένο πυρήνα $\mathbf{K}'$ μαθαίνει την ακριβώς ίδια οικογένεια επιπέδων, οπότε ακολουθούμε την καθιερωμένη ορολογία της βαθιάς μάθησης και αποκαλούμε την ετεροσυσχέτιση «συνέλιξη»).

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Συνελικτικά Επίπεδα

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Το Επίπεδο

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Ένα δισδιάστατο συνελικτικό επίπεδο εκτελεί ετεροσυσχέτιση της εισόδου του με έναν εκπαιδεύσιμο πυρήνα και προσθέτει μια βαθμωτή πόλωση. Ακριβώς όπως τα βάρη του πολυεπίπεδου perceptron (MLP) στο κεφάλαιο :numref:`chap_perceptrons`, αρχικοποιούμε τον πυρήνα από $\mathcal{N}(0, 0.01^2)$ και την πόλωση στο μηδέν. Παρακάτω, η κλάση `Conv2D` καταχωρεί και τις δύο παραμέτρους και καλεί τη `corr2d` στο εμπρόσθιο πέρασμά της· ένα επίπεδο με πυρήνα ύψους $h$ και πλάτους $w$ ονομάζεται συνελικτικό επίπεδο $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Σε αντίθεση με ένα πυκνό επίπεδο, ένα συνελικτικό επίπεδο δεν δεσμεύεται σε μια σταθερή ανάλυση εισόδου επειδή οι παράμετροί του εξαρτώνται μόνο από το μέγεθος πυρήνα. Το ίδιο ακριβώς επίπεδο $2 \times 2$ (5 παράμετροι συνολικά) επεξεργάζεται τόσο τον τανυστή `X` διαστάσεων $3 \times 3$ όσο και μια εικόνα $6 \times 8$, επιστρέφοντας εξόδους με σχήματα $2 \times 2$ και $5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Ανίχνευση Ακμών

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Δοκιμάζουμε τώρα τον ανιχνευτή κατακόρυφων ακμών από την εισαγωγή σε μια συνθετική εικόνα $6 \times 8$, της οποίας οι τέσσερις μεσαίες στήλες είναι μαύρες (`0`) και οι εξωτερικές στήλες είναι λευκές (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
Ορίζουμε τον πυρήνα $1 \times 2$

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
Σε κάθε θέση $(i, j)$, αυτό υπολογίζει την οριζόντια *πεπερασμένη διαφορά* $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, μια διακριτή προσέγγιση (με ακρίβεια προσήμου) της οριζόντιας παραγώγου της φωτεινότητας, η οποία μηδενίζεται στις επίπεδες περιοχές και παρουσιάζει αιχμή όπου η ένταση μεταβάλλεται απότομα.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
Η ετεροσυσχέτιση της εικόνας λωρίδας `X` διαστάσεων $6 \times 8$ με τον πυρήνα οριζόντιας διαφοράς `K` διαστάσεων $1 \times 2$ μέσω `corr2d(X, K)` παράγει τον χάρτη ακμών `Y` διαστάσεων $6 \times 7$:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Πράγματι, ο `Y` ενεργοποιείται στο `+1` κατά μήκος του ορίου λευκού-προς-μαύρο, στο `-1` κατά μήκος του ορίου μαύρου-προς-λευκό και στο `0` οπουδήποτε αλλού. Εφαρμόζοντας τον ίδιο ακριβώς πυρήνα στη μετατεθειμένη εικόνα (μετατρέποντας την κατακόρυφη λωρίδα σε οριζόντια), προκύπτουν παντού μηδενικά:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Είναι προφανές ότι το `[1, -1]` αποκρίνεται αυστηρά σε κατακόρυφες ακμές. Ο εντοπισμός οριζόντιων ακμών, διαγωνίων, γωνιών ή υφών απαιτεί εντελώς διαφορετικούς πυρήνες.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Εκμάθηση Πυρήνα

<!--en:61cacfaf kind=paragraph-->
Hand-crafting `[1, -1]` is easy when we know the exact feature we want, yet no
engineer can specify thousands of intermediate filters across a deep network by
hand. Can gradient descent recover the kernel `[1, -1]` from the input–output
pair `(X, Y)` alone? We test this using each framework's built-in 2D
convolutional layer (with bias omitted and weights initialized from
$\mathcal{N}(0, 0.01^2)$), minimizing squared error summed across all 42 output
entries over 10 gradient steps at learning rate $0.03$. Because built-in layers
expect 4D tensors including batch and channel axes (:numref:`sec_channels`), we
reshape `X` and `Y` accordingly.
<!--tr:61cacfaf-->
Η χειροποίητη κατασκευή του `[1, -1]` είναι εύκολη όταν γνωρίζουμε το ακριβές χαρακτηριστικό που θέλουμε, ωστόσο κανένας μηχανικός δεν μπορεί να καθορίσει με το χέρι χιλιάδες ενδιάμεσα φίλτρα σε ένα βαθύ δίκτυο. Μπορεί η κατάβαση κλίσης να ανακτήσει τον πυρήνα `[1, -1]` μόνο από το ζεύγος εισόδου–εξόδου `(X, Y)`; Το δοκιμάζουμε χρησιμοποιώντας το ενσωματωμένο 2D συνελικτικό επίπεδο κάθε πλαισίου (με παράλειψη της πόλωσης και αρχικοποίηση βαρών από $\mathcal{N}(0, 0.01^2)$), ελαχιστοποιώντας το τετραγωνικό σφάλμα αθροισμένο σε όλα τα 42 στοιχεία εξόδου επί 10 βήματα κλίσης με ρυθμό μάθησης $0.03$. Επειδή τα ενσωματωμένα επίπεδα αναμένουν 4D τανυστές που περιλαμβάνουν άξονες παρτίδας και καναλιού (:numref:`sec_channels`), ανασχηματίζουμε τα `X` και `Y` αναλόγως.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Αρχικά η απώλεια βρίσκεται κοντά στο 12 (ο σχεδόν μηδενικός τυχαίος πυρήνας αστοχεί και στα 12 εικονοστοιχεία ακμών $\pm 1$ κατά περίπου 1), και μέσα σε μόλις εννέα βήματα πέφτει κάτω από το 0.01. Η εξέταση του εκπαιδευμένου πυρήνα επιβεβαιώνει τι βρήκε:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Και τα δύο στοιχεία καταλήγουν σε απόσταση περίπου 0.02 από το `[1, -1]`. Τίποτα σε αυτόν τον βρόχο δεν γνώριζε οτιδήποτε για τις ακμές: με μόνο ένα σήμα απώλειας, η κατάβαση κλίσης ανακάλυψε το φίλτρο από μόνη της. Εδώ επιβλέψαμε το μεμονωμένο επίπεδο απευθείας με το `Y`· σε ένα βαθύ CNN, η οπισθοδιάδοση μαθαίνει τους πυρήνες κάθε επιπέδου από άκρη σε άκρη (end-to-end) μόνο από την τελική απώλεια του έργου.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Συνελίξεις για Εικόνες<br>**2D ετεροσυσχέτιση · ανίχνευση ακμών · εκπαιδευμένοι πυρήνες · im2col · υποδεκτικά πεδία**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Ανίχνευση Τοπικών Ακμών σε μια Εικόνα"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Κίνητρο & Βασικό Νοητικό Μοντέλο]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Συνθετική εικόνα δοκιμής $6 \times 8$:** Λευκές εξωτερικές στήλες (`1`) περιβάλλουν μια μαύρη κατακόρυφη ζώνη 4 στηλών (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Οριζόντια πεπερασμένη διαφορά:** Η σάρωση του $\mathbf{K} = [1, -1]$ υπολογίζει $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, δίνοντας `+1` στις ακμές λευκού-προς-μαύρο και `-1` στις ακμές μαύρου-προς-λευκό.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Εκμάθηση έναντι χειροποίητης σχεδίασης:** Τα βαθιά CNN ανακαλύπτουν χιλιάδες τέτοιους πυρήνες αυτόματα μέσω κατάβασης κλίσης.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="Η Πράξη της 2D Ετεροσυσχέτισης"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Εσωτερικά Γινόμενα Ολισθαίνοντος Παραθύρου]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Δεδομένου ενός πίνακα εισόδου $\mathbf{X}$ διαστάσεων $n_\textrm{h} \times n_\textrm{w}$ και ενός πυρήνα $\mathbf{K}$ διαστάσεων $k_\textrm{h} \times k_\textrm{w}$, η 2D ετεροσυσχέτιση $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ τοποθετεί την πάνω αριστερή γωνία του παραθύρου στο $(i, j)$ και αθροίζει τα κατά στοιχείο γινόμενα:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Άνω αριστερή αγκύρωση:** Δείκτες $a, b \ge 0$ με αρχή το μηδέν ταιριάζουν με τη δεικτοδότηση πινάκων στον κώδικα και χειρίζονται άρτιους πυρήνες ($2 \times 2$, $1 \times 2$) καθαρά.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Σκιασμένη πάνω αριστερή έξοδος $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Έγκυρες Θέσεις Παραθύρου και Χωρικό Σχήμα Εξόδου"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Γιατί οι Συνελίξεις Χωρίς Συμπλήρωση Συρρικνώνουν το Πλέγμα]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Έγκυρες τοποθετήσεις παραθύρου:** Ένα παράθυρο $k_\textrm{h} \times k_\textrm{w}$ χωρά σε πλέγμα $n_\textrm{h} \times n_\textrm{w}$ σε $n - k + 1$ θέσεις ανά άξονα.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Παραδείγματα σχημάτων:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$)· $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Υλοποίηση 2D Ετεροσυσχέτισης σε Κώδικα"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Απευθείας Υλοποίηση με Εμφωλευμένους Βρόχους]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
Η συνάρτηση `corr2d` δεσμεύει τον τανυστή εξόδου $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ και υπολογίζει κάθε στοιχείο ως το άθροισμα του `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Κάθε επανάληψη απομονώνει ένα υποπαράθυρο $k_\textrm{h} \times k_\textrm{w}$ του `X`, πολλαπλασιάζει κατά στοιχείο με το `K` και ανάγει μέσω `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Επαλήθευση της corr2d και Ισοδυναμία Αντεστραμμένου Πυρήνα"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Ετεροσυσχέτιση έναντι Μαθηματικής Συνέλιξης]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Πυρήνας $\mathbf{K}'$ αντεστραμμένος κατά 180°:** Η ετεροσυσχέτιση με $\mathbf{K}$ ισούται με συνέλιξη με $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Ίδια εκπαιδευόμενη κλάση:** Τα πλαίσια υπολογίζουν ετεροσυσχέτιση και την αποκαλούν συνέλιξη.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Κατασκευή Μονάδας 2D Συνελικτικού Επιπέδου"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Εκπαιδεύσιμα Βάρη Πυρήνα και Βαθμωτή Πόλωση]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Ένα συνελικτικό επίπεδο περιβάλλει τη `corr2d` με μια εκπαιδεύσιμη παράμετρο βάρους $k_\textrm{h} \times k_\textrm{w}$ (αρχικοποιημένη από $\mathcal{N}(0, 0.01^2)$) και μια βαθμωτή πόλωση (αρχικοποιημένη στο $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Ανεξαρτησία Ανάλυσης των Συνελικτικών Επιπέδων"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Σταθερές Παράμετροι σε Αυθαίρετα Μεγέθη Εισόδου]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Αποσυνδεδεμένο από $H \times W$:** Οι ίδιες ακριβώς 5 παράμετροι (πυρήνας $2 \times 2$ + 1 πόλωση) επεξεργάζονται εισόδους $3 \times 3$, $6 \times 8$ ή megapixel χωρίς ανασχηματισμό βαρών.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Κατασκευή Συνθετικής Εικόνας Κατακόρυφων Ακμών"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Ένα Δυαδικό Πρότυπο Δοκιμής 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Κατασκευάζουμε τη δοκιμαστική εικόνα `X` διαστάσεων $6 \times 8$ με άσσους στα αριστερά και δεξιά όρια και μηδενικά στις τέσσερις μεσαίες στήλες (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Τμηματικά σταθερές στήλες:** Η ένταση είναι σταθερή (`1` ή `0`) εντός των στηλών $0\text{--}1$, $2\text{--}5$ και $6\text{--}7$, με άλμα μόνο μεταξύ $1 \to 2$ και $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Εφαρμογή του Πυρήνα Οριζόντιας Πεπερασμένης Διαφοράς"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Ανίχνευση Κατακόρυφων Ακμών με K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
Η ετεροσυσχέτιση του `X` με τον πυρήνα $1 \times 2$ $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ υπολογίζει $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ σε κάθε οριζόντιο ζεύγος:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Προσημασμένος χάρτης ακμών:** Η στήλη 1 δίνει `+1` (λευκό-προς-μαύρο), η στήλη 5 δίνει `-1` (μαύρο-προς-λευκό) και οι σταθερές εσωτερικές στήλες δίνουν `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Επιλεκτικότητα Προσανατολισμού των Πυρήνων Ακμών"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Γιατί Ένας Πυρήνας Δεν Ανιχνεύει Όλους τους Προσανατολισμούς]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Τυφλός σε οριζόντιες ακμές:** Η μετάθεση του `X` (`X.T`) καθιστά τους οριζόντιους γείτονες ίδιους, οπότε το $[1, -1]$ εξάγει μόνο μηδενικά.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Τράπεζες φίλτρων (§{sec_channels}):** Τα επίπεδα όρασης μαθαίνουν παράλληλα κανάλια για οριζόντιους, κατακόρυφους και διαγώνιους προσανατολισμούς.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Εκμάθηση Ανιχνευτή Ακμών μέσω Κατάβασης Κλίσης"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Επίβλεψη Ενσωματωμένου Συνελικτικού Επιπέδου στο (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **4D τανυστές πλαισίου:** Ανασχηματίστε τα `X` και `Y` σε 4D τανυστές (ένα παράδειγμα, ένα κανάλι) και ελαχιστοποιήστε το $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ με $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Εξέταση των Εκπαιδευμένων Βαρών Πυρήνα"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Γιατί η Αρχική Απώλεια Είναι 12 και Συγκλίνει στο [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Γιατί η απώλεια ξεκινά από $12.0$:** Σχεδόν μηδενικά βάρη προβλέπουν $\hat{Y} \approx 0$, χάνοντας τα δώδεκα εικονοστοιχεία ακμών $\pm 1$ στο `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Σύγκλιση σε 9 βήματα:** Η SGD ανακτά το $[0.99, -0.99]$ αυτόματα από τα δεδομένα.

=== DECISIONS ===
- Register: first-person plural ("εμείς") for prose; imperative plural for exercises.
- Question mark in Greek prose: standard Greek semicolon (`;`).
- "edge map" → «χάρτης ακμών» (established literal rendering in Greek image processing).
- "feature map" → «χάρτης χαρακτηριστικών» (consistent with glossary `feature` = χαρακτηριστικό).
- "patch" (input patch) → «τεμάχιο» (distinct from glossary `slice` = τμήμα).
- "im2col" → kept as *im2col* with Greek gloss on first mention.
- "end to end" → «από άκρη σε άκρη (end-to-end)» on first mention.
- "zero-based indices" → «δείκτες που ξεκινούν από το μηδέν».
- Decimal points in prose numbers/parameters (`0.01`, `0.03`, `0.02`, `12.0`) kept as in English (`TRANSLATING.md` §4).

=== NOTES ===
- 8d45aae8: math tokens (`$\mathbf{X}$`, `$n_\textrm{h} \times n_\textrm{w}$`, `$\mathbf{K}$`, `$k_\textrm{h} \times k_\textrm{w}$`) reordered to follow natural Greek word order (noun + symbol + dimensions).
