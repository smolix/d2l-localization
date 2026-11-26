<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=el rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Συνελίξεις για εικόνες

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
Στην :numref:`sec_why-conv` εξαγάγαμε στο χαρτί το συνελικτικό επίπεδο από δύο γεωμετρικές αρχές: την ισομεταβλητότητα ως προς τη μετατόπιση (translation equivariance) και την τοπικότητα. Τώρα θα δούμε τι υπολογίζει πράγματι αυτός ο τύπος πάνω σε μια εικόνα, και αν η κατάβαση κλίσης (gradient descent) μπορεί να ανακαλύψει αυτόματα, από τα δεδομένα, χρήσιμους πυρήνες.

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
Ας υποθέσουμε ότι θέλουμε να εντοπίσουμε κατακόρυφες ακμές, εκεί όπου μια φωτεινή περιοχή συναντά μια σκοτεινή. Αν αφαιρέσουμε από κάθε εικονοστοιχείο τον δεξιό γείτονά του ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$), προκύπτει μηδέν μέσα στις ομοιόμορφες περιοχές, $+1$ εκεί όπου το λευκό μεταβαίνει σε μαύρο και $-1$ εκεί όπου το μαύρο επιστρέφει σε λευκό. Αν σαρώσουμε τον ανιχνευτή των δύο βαρών $(1, -1)$ σε κάθε έγκυρη θέση της εικόνας, παίρνουμε έναν *χάρτη ακμών*. Σε αυτήν την ενότητα υλοποιούμε τη δισδιάστατη ετεροσυσχέτιση (cross-correlation), δοκιμάζουμε τον ανιχνευτή ακμών που φτιάξαμε με το χέρι, αφήνουμε την κατάβαση κλίσης να *μάθει* το $(1, -1)$ απευθείας από τα δεδομένα, δείχνουμε πώς μια συνέλιξη γίνεται ένας μοναδικός πολλαπλασιασμός πινάκων (*im2col*) και εξάγουμε πώς η στοίβαξη επιπέδων διευρύνει το υποδεκτικό πεδίο (receptive field) μιας μονάδας.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Ετεροσυσχέτιση

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### Η πράξη

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Αφήνοντας προς το παρόν στην άκρη τα κανάλια, έστω ότι η είσοδος $\mathbf{X}$ είναι ένας πίνακας $n_\textrm{h} \times n_\textrm{w}$ και ο πυρήνας $\mathbf{K}$ ένας πίνακας $k_\textrm{h} \times k_\textrm{w}$. Ο πυρήνας ορίζει ένα κινούμενο *παράθυρο* σχήματος $k_\textrm{h} \times k_\textrm{w}$. Τοποθετούμε την επάνω αριστερή γωνία του παραθύρου στη θέση $(i, j)$, πολλαπλασιάζουμε κατά στοιχείο το τμήμα της εισόδου που καλύπτεται με το $\mathbf{K}$ και αθροίζουμε τα γινόμενα, ώστε να προκύψει η καταχώριση της εξόδου στη θέση $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Αυτή είναι η *δισδιάστατη ετεροσυσχέτιση* του $\mathbf{X}$ με το $\mathbf{K}$, που γράφεται $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$, με δείκτες που ξεκινούν από το μηδέν, όπως στον κώδικα. Ενώ στην :numref:`sec_why-conv` τα σχετικά διαστήματα κεντράρονταν συμμετρικά από το $-\Delta$ ως το $\Delta$, η αγκύρωση στην επάνω αριστερή γωνία διαφέρει μόνο κατά μια σταθερή μετατόπιση $\Delta$, ενώ χειρίζεται φυσικά και πυρήνες άρτιου μεγέθους (όπως ένας πυρήνας $2 \times 2$ ή ο ανιχνευτής ακμών $1 \times 2$ που χρησιμοποιούμε), οι οποίοι δεν έχουν κεντρικό εικονοστοιχείο.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Δισδιάστατη ετεροσυσχέτιση. Τα σκιασμένα τμήματα είναι το πρώτο στοιχείο της εξόδου και τα στοιχεία της εισόδου και του πυρήνα που χρησιμοποιούνται για τον υπολογισμό του: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
Στην :numref:`fig_correlation` ολισθαίνουμε έναν πυρήνα $2 \times 2$ πάνω σε μια είσοδο $3 \times 3$, ένα εικονοστοιχείο τη φορά, από αριστερά προς τα δεξιά και από πάνω προς τα κάτω. Το παράθυρο χωράει ολόκληρο σε τέσσερις θέσεις και δίνει την έξοδο $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Παρατηρήστε ότι η έξοδος συρρικνώνεται ελαφρά κατά μήκος κάθε άξονα, επειδή ο πυρήνας πρέπει να παραμένει εξ ολοκλήρου μέσα στην είσοδο. Ένα παράθυρο πλάτους $k$ που τοποθετείται σε μήκος $n$ αφήνει $n - k + 1$ έγκυρες θέσεις, και έτσι το σχήμα της εξόδου είναι

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Έτσι, μια είσοδος $3 \times 3$ με πυρήνα $2 \times 2$ δίνει έξοδο $2 \times 2$, μια εικόνα $6 \times 8$ με πυρήνα $1 \times 2$ δίνει $6 \times 7$, και μια εικόνα Fashion-MNIST $28 \times 28$ με πυρήνα $5 \times 5$ δίνει $24 \times 24$. Όποτε θέλουμε να διατηρήσουμε τις χωρικές διαστάσεις, περιβάλλουμε την είσοδο με ένα περίγραμμα μηδενικών (*συμπλήρωση*, padding), το οποίο εξετάζουμε μαζί με τον διασκελισμό (stride) και τη διαστολή (dilation) στην :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Υλοποίηση

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Υλοποιούμε την :eqref:`eq_corr2d` απευθείας στη `corr2d`, με ένα ζεύγος βρόχων πάνω σε κάθε συντεταγμένη της εξόδου.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Για να επαληθεύσουμε τη `corr2d` με τον υπολογισμό που κάναμε με το χέρι στην :numref:`fig_correlation`, κατασκευάζουμε τον τανυστή εισόδου `X` διαστάσεων $3 \times 3$ και τον πυρήνα `K` διαστάσεων $2 \times 2$ και υπολογίζουμε το `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Η έξοδος αναπαράγει την :numref:`fig_correlation` στοιχείο προς στοιχείο (και, όπως δείξαμε στην :numref:`sec_why-conv`, η αυστηρή μαθηματική συνέλιξη με έναν ανεστραμμένο πυρήνα $\mathbf{K}'$ μαθαίνει ακριβώς την ίδια οικογένεια επιπέδων, γι' αυτό ακολουθούμε την καθιερωμένη ορολογία της βαθιάς μάθησης και αποκαλούμε «συνέλιξη» την ετεροσυσχέτιση).

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Συνελικτικά επίπεδα

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Το επίπεδο

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Ένα δισδιάστατο συνελικτικό επίπεδο (convolutional layer) υπολογίζει την ετεροσυσχέτιση της εισόδου του με έναν πυρήνα που μαθαίνεται και προσθέτει μια βαθμωτή πόλωση. Όπως και με τα βάρη του MLP μας στο :numref:`chap_perceptrons`, αρχικοποιούμε τον πυρήνα από την $\mathcal{N}(0, 0.01^2)$ και την πόλωση στο μηδέν. Παρακάτω, η `Conv2D` καταχωρίζει και τις δύο παραμέτρους και καλεί τη `corr2d` στο εμπρόσθιο πέρασμά της· ένα επίπεδο με πυρήνα ύψους $h$ και πλάτους $w$ ονομάζεται συνελικτικό επίπεδο $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Σε αντίθεση με ένα πυκνό επίπεδο, ένα συνελικτικό επίπεδο δεν είναι δεσμευμένο σε σταθερή ανάλυση εισόδου, επειδή οι παράμετροί του εξαρτώνται μόνο από το μέγεθος του πυρήνα. Το ίδιο ακριβώς επίπεδο $2 \times 2$ (5 παράμετροι συνολικά) επεξεργάζεται τόσο τον τανυστή $3 \times 3$ `X` όσο και μια εικόνα $6 \times 8$, και επιστρέφει εξόδους σχήματος $2 \times 2$ και $5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Ανίχνευση ακμών

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Δοκιμάζουμε τώρα τον ανιχνευτή κατακόρυφων ακμών της εισαγωγής σε μια συνθετική εικόνα $6 \times 8$, της οποίας οι τέσσερις μεσαίες στήλες είναι μαύρες (`0`) και οι εξωτερικές στήλες λευκές (`1`).

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
Σε κάθε θέση $(i, j)$, αυτός υπολογίζει την οριζόντια *πεπερασμένη διαφορά* $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, μια διακριτή προσέγγιση (ως προς το πρόσημο) της οριζόντιας παραγώγου της φωτεινότητας, η οποία μηδενίζεται στις ομοιόμορφες περιοχές και εμφανίζει απότομες τιμές όπου η ένταση αλλάζει απότομα.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
Η ετεροσυσχέτιση της εικόνας με τις ρίγες `X` διαστάσεων $6 \times 8$ με τον πυρήνα οριζόντιας διαφοράς `K` διαστάσεων $1 \times 2$ μέσω της `corr2d(X, K)` δίνει τον χάρτη ακμών `Y` διαστάσεων $6 \times 7$:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Πράγματι, το `Y` δίνει `+1` κατά μήκος του ορίου από λευκό σε μαύρο, `-1` κατά μήκος του ορίου από μαύρο σε λευκό και `0` παντού αλλού. Αν εφαρμόσουμε τον ίδιο ακριβώς πυρήνα στην ανάστροφη εικόνα (που μετατρέπει την κατακόρυφη ρίγα σε οριζόντια), παίρνουμε αποκλειστικά μηδενικά:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Είναι σαφές ότι το `[1, -1]` ανταποκρίνεται αποκλειστικά σε κατακόρυφες ακμές. Για να εντοπίσουμε οριζόντιες ακμές, διαγώνιες, γωνίες ή υφές χρειάζονται εντελώς διαφορετικοί πυρήνες.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Εκμάθηση ενός πυρήνα

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
Η κατασκευή του `[1, -1]` με το χέρι είναι εύκολη όταν γνωρίζουμε ακριβώς το χαρακτηριστικό που θέλουμε, όμως κανένας μηχανικός δεν μπορεί να ορίσει με το χέρι χιλιάδες ενδιάμεσα φίλτρα σε ένα βαθύ δίκτυο. Μπορεί η κατάβαση κλίσης να ανακτήσει τον πυρήνα `[1, -1]` μόνο από το ζεύγος εισόδου–εξόδου `(X, Y)`; Το δοκιμάζουμε χρησιμοποιώντας το ενσωματωμένο δισδιάστατο συνελικτικό επίπεδο κάθε πλαισίου (χωρίς πόλωση και με βάρη που αρχικοποιούνται από την $\mathcal{N}(0, 0.01^2)$), ελαχιστοποιώντας το τετραγωνικό σφάλμα (squared error), αθροισμένο σε όλες τις 42 καταχωρίσεις της εξόδου, σε 10 βήματα κατάβασης κλίσης με ρυθμό μάθησης $0.03$. Επειδή τα ενσωματωμένα επίπεδα αναμένουν τανυστές 4 διαστάσεων που περιλαμβάνουν άξονες παρτίδας και καναλιού (:numref:`sec_channels`), αλλάζουμε αντίστοιχα το σχήμα των `X` και `Y`.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Αρχικά η απώλεια βρίσκεται κοντά στο 12 (ο σχεδόν μηδενικός τυχαίος πυρήνας αστοχεί και στα 12 εικονοστοιχεία ακμής $\pm 1$ κατά περίπου 1) και μέσα σε μόλις εννέα βήματα πέφτει κάτω από 0.01. Η εξέταση του πυρήνα που μαθεύτηκε επιβεβαιώνει τι βρήκε:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Και οι δύο τιμές καταλήγουν σε απόσταση περίπου 0.02 από το `[1, -1]`. Τίποτα σε αυτόν τον βρόχο δεν γνώριζε οτιδήποτε για ακμές: με δεδομένο ένα σήμα απώλειας, η κατάβαση κλίσης ανακάλυψε το φίλτρο μόνη της. Εδώ επιβλέψαμε απευθείας το μεμονωμένο επίπεδο με το `Y`· σε ένα βαθύ CNN, η οπισθοδιάδοση (backpropagation) μαθαίνει τους πυρήνες κάθε επιπέδου από άκρο σε άκρο, μόνο από την απώλεια της τελικής εργασίας.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Συνελίξεις για εικόνες<br>**δισδιάστατη ετεροσυσχέτιση · ανίχνευση ακμών · πυρήνες που μαθεύονται · im2col · υποδεκτικά πεδία**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Ανίχνευση τοπικών ακμών σε μια εικόνα"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Κίνητρο και βασικό νοητικό μοντέλο]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Συνθετική εικόνα δοκιμής $6 \times 8$:** λευκές εξωτερικές στήλες (`1`) γύρω από μια μαύρη κατακόρυφη ζώνη 4 στηλών (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Οριζόντια πεπερασμένη διαφορά:** η σάρωση με $\mathbf{K} = [1, -1]$ υπολογίζει το $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ και δίνει `+1` στις ακμές από λευκό σε μαύρο και `-1` στις ακμές από μαύρο σε λευκό.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Εκμάθηση αντί για κατασκευή με το χέρι:** τα βαθιά CNN ανακαλύπτουν αυτόματα χιλιάδες τέτοιους πυρήνες μέσω κατάβασης κλίσης.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="Η δισδιάστατη ετεροσυσχέτιση"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Εσωτερικά γινόμενα με κινούμενο παράθυρο]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Για έναν πίνακα εισόδου $n_\textrm{h} \times n_\textrm{w}$ $\mathbf{X}$ και έναν πυρήνα $k_\textrm{h} \times k_\textrm{w}$ $\mathbf{K}$, η δισδιάστατη ετεροσυσχέτιση $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ τοποθετεί την επάνω αριστερή γωνία του παραθύρου στο $(i, j)$ και αθροίζει γινόμενα κατά στοιχείο:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Αγκύρωση επάνω αριστερά:** οι δείκτες από το μηδέν $a, b \ge 0$ ταιριάζουν με την ευρετηρίαση πινάκων στον κώδικα και χειρίζονται καθαρά πυρήνες άρτιου μεγέθους (όπως $2 \times 2$ ή $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Σκιασμένη έξοδος επάνω αριστερά $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Έγκυρες θέσεις παραθύρου και χωρικό σχήμα εξόδου"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Γιατί οι συνελίξεις χωρίς συμπλήρωση συρρικνώνουν το πλέγμα]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Έγκυρες θέσεις παραθύρου:** ένα παράθυρο $k_\textrm{h} \times k_\textrm{w}$ χωράει σε ένα πλέγμα $n_\textrm{h} \times n_\textrm{w}$ σε $n - k + 1$ θέσεις ανά άξονα.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Παραδείγματα σχήματος:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$)· $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Υλοποίηση της δισδιάστατης ετεροσυσχέτισης στον κώδικα"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Άμεση υλοποίηση με ένθετους βρόχους]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
Η συνάρτηση `corr2d` δεσμεύει τον τανυστή εξόδου $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ και υπολογίζει κάθε καταχώριση ως το άθροισμα του `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Κάθε επανάληψη παίρνει ένα τμήμα του `X` ως υποπαράθυρο $k_\textrm{h} \times k_\textrm{w}$, το πολλαπλασιάζει κατά στοιχείο με το `K` και το αθροίζει μέσω της `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Επαλήθευση της corr2d και η ισοδυναμία με ανεστραμμένο πυρήνα"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Ετεροσυσχέτιση και μαθηματική συνέλιξη]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Πυρήνας $\mathbf{K}'$ ανεστραμμένος κατά 180°:** η ετεροσυσχέτιση με το $\mathbf{K}$ ισούται με τη συνέλιξη με το $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Ίδια κλάση που μαθαίνεται:** τα πλαίσια υπολογίζουν ετεροσυσχέτιση και την αποκαλούν συνέλιξη.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Κατασκευή μιας δισδιάστατης συνελικτικής μονάδας"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Βάρη πυρήνα που μαθαίνονται και βαθμωτή πόλωση]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Ένα συνελικτικό επίπεδο τυλίγει τη `corr2d` με μια παράμετρο βαρών $k_\textrm{h} \times k_\textrm{w}$ που μαθαίνεται (αρχικοποιημένη από την $\mathcal{N}(0, 0.01^2)$) και μια βαθμωτή πόλωση (αρχικοποιημένη στο $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Ανεξαρτησία των συνελικτικών επιπέδων από την ανάλυση"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Σταθερές παράμετροι για αυθαίρετα μεγέθη εισόδου]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Αποσύνδεση από το $H \times W$:** οι ίδιες ακριβώς 5 παράμετροι (πυρήνας $2 \times 2$ + 1 πόλωση) επεξεργάζονται εισόδους $3 \times 3$, $6 \times 8$ ή μεγαπίξελ χωρίς αλλαγή των βαρών.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Κατασκευή συνθετικής εικόνας με κατακόρυφη ακμή"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Ένα δυαδικό πρότυπο δοκιμής 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Κατασκευάζουμε την εικόνα δοκιμής `X` διαστάσεων $6 \times 8$ με άσους στο αριστερό και στο δεξί περίγραμμα και μηδενικά στις τέσσερις μεσαίες στήλες (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Στήλες σταθερές κατά τμήματα:** η ένταση είναι σταθερή (`1` ή `0`) στις στήλες $0\text{--}1$, $2\text{--}5$ και $6\text{--}7$ και αλλάζει μόνο μεταξύ των στηλών $1 \to 2$ και $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Εφαρμογή του πυρήνα οριζόντιας πεπερασμένης διαφοράς"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Ανίχνευση κατακόρυφων ακμών με K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
Η ετεροσυσχέτιση του `X` με τον πυρήνα $1 \times 2$ $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ υπολογίζει το $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ σε κάθε οριζόντιο ζεύγος:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Χάρτης ακμών με πρόσημο:** η στήλη 1 δίνει `+1` (μετάβαση από λευκό σε μαύρο), η στήλη 5 δίνει `-1` (μετάβαση από μαύρο σε λευκό) και όλες οι σταθερές εσωτερικές στήλες δίνουν `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Επιλεκτικότητα των πυρήνων ακμών ως προς τον προσανατολισμό"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Γιατί ένας πυρήνας δεν μπορεί να ανιχνεύσει όλους τους προσανατολισμούς]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Τυφλός στις οριζόντιες ακμές:** η μεταφορά του `X` (`X.T`) κάνει τους οριζόντιους γείτονες ίδιους, και έτσι το $[1, -1]$ δίνει παντού μηδέν.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Τράπεζες φίλτρων (§{sec_channels}):** τα επίπεδα όρασης μαθαίνουν παράλληλα κανάλια για οριζόντιους, κατακόρυφους και διαγώνιους προσανατολισμούς.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Εκμάθηση ανιχνευτή ακμών με κατάβαση κλίσης"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Εποπτεία ενός ενσωματωμένου συνελικτικού επιπέδου στο (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Τανυστές πλαισίου 4 διαστάσεων:** αλλάζουμε το σχήμα των `X` και `Y` σε τανυστές 4 διαστάσεων (ένα παράδειγμα, ένα κανάλι) και ελαχιστοποιούμε το $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ με $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Εξέταση των βαρών του πυρήνα που μαθεύτηκε"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Γιατί η αρχική απώλεια είναι 12 και συγκλίνει στο [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Γιατί η απώλεια ξεκινά από $12.0$:** τα σχεδόν μηδενικά βάρη προβλέπουν $\hat{Y} \approx 0$ και αστοχούν στα δώδεκα εικονοστοιχεία ακμής $\pm 1$ του `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Σύγκλιση σε 9 βήματα:** η SGD ανακτά αυτόματα το $[0.99, -0.99]$ από τα δεδομένα.

=== DECISIONS ===
- Register: first person plural ("ας δούμε", "υλοποιούμε"), polite plural "παρατηρήστε" for imperatives to the reader; monotonic Greek; quotation marks « ».
- Numbers: decimals kept with a point as in the English (0.01, 0.03), no Greek comma, so they match code output.
- First use of a key term in a section gets the English in parentheses: ετεροσυσχέτιση (cross-correlation); later uses are Greek only.
- "edge" (image) → ακμή; "edge map" → χάρτης ακμών; "edge detector" → ανιχνευτής ακμών.
- "unit" (a neuron/activation of a layer) → μονάδα; "window" → παράθυρο; "patch" → ψηφίδα (but a plain input region covered by a window is τμήμα της εισόδου); "patch matrix" → πίνακας ψηφίδων.
- "framework" (PyTorch, JAX, ...) → πλαίσιο; "built-in layer" → ενσωματωμένο επίπεδο; "dense layer" → πυκνό επίπεδο.
- "feature map" → χάρτης χαρακτηριστικών; "end to end" → από άκρο σε άκρο; "workspace memory" → μνήμη εργασίας.
- "filter bank" → τράπεζα φίλτρων; "weight/bias parameter" bias of a layer → πόλωση (glossary), estimator bias is not used here.
- "mathematical convolution" → μαθηματική συνέλιξη; "flipped kernel" → ανεστραμμένος πυρήνας.
