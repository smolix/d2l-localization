<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=el rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Η Συνέλιξη ως Πολλαπλασιασμός Πινάκων

<!--en:5bdffcd7 kind=paragraph-->
While the nested loops in `corr2d` make the math transparent, visiting one
output pixel at a time in Python is far too slow for real workloads. A classic
remedy, and still one of the algorithms that libraries choose from, is to recast
cross-correlation as a single large matrix multiplication, the exact operation
GPUs execute at peak speed. Notice that every
output entry in :numref:`fig_correlation` is simply the dot product of the
flattened $k_\textrm{h} k_\textrm{w}$-element kernel with the corresponding
flattened input window. Extracting all
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ windows and
stacking them as rows yields a patch matrix; multiplying it by the flattened
kernel computes the entire cross-correlation in one shot. This standard layout
transformation is called *im2col* ("image to columns", or rows here).
:numref:`fig_conv_im2col` carries it out on the example of
:numref:`fig_correlation`.
<!--tr:5bdffcd7-->
Παρόλο που οι εμφωλευμένοι βρόχοι στη `corr2d` καθιστούν τα μαθηματικά διαφανή, η επίσκεψη ενός εικονοστοιχείου εξόδου τη φορά στην Python είναι πολύ αργή για πραγματικούς φόρτους εργασίας. Μια κλασική λύση, και ακόμη ένας από τους αλγορίθμους που επιλέγουν οι βιβλιοθήκες, είναι η αναδιατύπωση της ετεροσυσχέτισης ως ενός μεγάλου πολλαπλασιασμού πινάκων, της ακριβούς πράξης που οι GPU εκτελούν με μέγιστη ταχύτητα. Παρατηρήστε ότι κάθε στοιχείο εξόδου στο :numref:`fig_correlation` είναι απλώς το εσωτερικό γινόμενο του ισοπεδωμένου πυρήνα $k_\textrm{h} k_\textrm{w}$ στοιχείων με το αντίστοιχο ισοπεδωμένο παράθυρο εισόδου. Η εξαγωγή όλων των $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ παραθύρων και η στοίβαξή τους ως γραμμές δίνει έναν πίνακα τεμαχίων· ο πολλαπλασιασμός του με τον ισοπεδωμένο πυρήνα υπολογίζει ολόκληρη την ετεροσυσχέτιση μονομιάς. Αυτός ο καθιερωμένος μετασχηματισμός διάταξης ονομάζεται *im2col* («εικόνα σε στήλες», ή γραμμές εδώ). Το :numref:`fig_conv_im2col` τον εφαρμόζει στο παράδειγμα του :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![Η ετεροσυσχέτιση του :numref:`fig_correlation` ως ένα γινόμενο πινάκων. Κάθε ένα από τα τέσσερα παράθυρα $2 \times 2$ της εισόδου γίνεται γραμμή του πίνακα τεμαχίων, και ο πολλαπλασιασμός με τον ισοπεδωμένο πυρήνα δίνει την ισοπεδωμένη έξοδο. Τα σκιασμένα κελιά ακολουθούν το πρώτο παράθυρο στη γραμμή του και στην έξοδο 19. Το κεντρικό εικονοστοιχείο 4 βρίσκεται και στα τέσσερα παράθυρα, οπότε εμφανίζεται σε κάθε γραμμή.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
Ο συμβιβασμός είναι η επιπλέον μνήμη χώρου εργασίας, καθώς κάθε εσωτερικό εικονοστοιχείο αντιγράφεται σε έως και $k_\textrm{h} k_\textrm{w}$ τεμάχια. Παρακάτω επαληθεύουμε στο :numref:`fig_correlation` ότι αυτός ο μοναδικός πολλαπλασιασμός πινάκων ταυτίζεται απόλυτα με τη `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
Η απόκλιση είναι ταυτοτικά μηδέν: ένα μόνο γινόμενο πινάκων αναπαράγει τη `corr2d` τέλεια.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
Ισοδύναμα, μπορούμε να ξεδιπλώσουμε τον *πυρήνα* αντί για την είσοδο σε έναν αραιό ταινιωτό πίνακα $\mathbf{W}$ που πολλαπλασιάζει την ισοπεδωμένη εικόνα—το ακριβές 2D ανάλογο του πλαισίου (c) στο :numref:`fig_why_conv_weight_sharing`, όπου κάθε γραμμή επαναλαμβάνει τα ίδια λίγα βάρη πυρήνα. Η θεώρηση μιας συνέλιξης ως δομημένου πυκνού επιπέδου κλείνει τον κύκλο επιστρέφοντας στην εξαγωγή της ενότητας :numref:`sec_why-conv` (και οι ασκήσεις κατασκευάζουν ρητά τον $\mathbf{W}$).

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## Χάρτες Χαρακτηριστικών και Υποδεκτικά Πεδία

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
Ένα μεμονωμένο συνελικτικό επίπεδο βλέπει μόνο ένα τεμάχιο μεγέθους ίσου με τον πυρήνα του, ενώ η αναγνώριση ολόκληρων αντικειμένων απαιτεί ενσωμάτωση συμφραζομένων από πολύ ευρύτερες περιοχές. Θυμηθείτε από την ενότητα :numref:`subsec_why-conv-channels` ότι η έξοδος ενός συνελικτικού επιπέδου ονομάζεται *χάρτης χαρακτηριστικών* (feature map, όπως ο χάρτης κατακόρυφων ακμών `Y`). Για οποιοδήποτε στοιχείο $x$ σε έναν χάρτη χαρακτηριστικών, το *υποδεκτικό πεδίο* (receptive field) του είναι η περιοχή της εικόνας εισόδου (ή ενός προηγούμενου επιπέδου) της οποίας οι τιμές μπορούν να επηρεάσουν το $x$ κατά το εμπρόσθιο πέρασμα—δανειζόμενοι τον όρο νευροφυσιολογίας από το :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Στοιβαγμένα Επίπεδα

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
Στο :numref:`fig_correlation`, η σκιασμένη έξοδος `19` έχει υποδεκτικό πεδίο $2 \times 2$ που καλύπτει τα τέσσερα πάνω αριστερά εικονοστοιχεία του $\mathbf{X}$. Τώρα στοιβάξτε ένα *δεύτερο* συνελικτικό επίπεδο $2 \times 2$ πάνω από την έξοδο $\mathbf{Y}$ διαστάσεων $2 \times 2$ για να παραχθεί ένα μόνο βαθμωτό μέγεθος $z$: το $z$ εξαρτάται από και τα τέσσερα στοιχεία του $\mathbf{Y}$, τα οποία μαζί εξαρτώνται από και τα εννέα στοιχεία του $\mathbf{X}$. Η στοίβαξη επιπέδων διευρύνει ομαλά το υποδεκτικό πεδίο με το βάθος.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Πόσο γρήγορα μεγαλώνει; Κάθε συνελικτικό επίπεδο $k \times k$ φτάνει $k - 1$ εικονοστοιχεία πιο μακριά από ένα μεμονωμένο στοιχείο της εισόδου του, οπότε η στοίβαξη $L$ επιπέδων με μεγέθη πυρήνων $k_1, \ldots, k_L$ δίνει ένα υποδεκτικό πεδίο μήκους πλευράς

<!--en:b302e3ef kind=paragraph-->
For a stack of $L$ consecutive $3 \times 3$ layers, each layer adds 2 pixels,
covering $(2L + 1) \times (2L + 1)$ inputs: $5 \times 5$ after two layers and
$7 \times 7$ after three (:numref:`fig_conv_receptive_field`). This linear
growth is actually a blessing rather than a reason to reach for huge kernels:
three stacked $3 \times 3$ layers achieve the same $7 \times 7$ receptive field
as a single $7 \times 7$ layer using far fewer parameters ($3 \times 9 = 27$
weights vs. 49 per channel pair) while interleaving three nonlinearities instead
of one.
<!--tr:b302e3ef-->
Για μια στοίβα $L$ διαδοχικών επιπέδων $3 \times 3$, κάθε επίπεδο προσθέτει 2 εικονοστοιχεία, καλύπτοντας $(2L + 1) \times (2L + 1)$ εισόδους: $5 \times 5$ μετά από δύο επίπεδα και $7 \times 7$ μετά από τρία (:numref:`fig_conv_receptive_field`). Αυτή η γραμμική αύξηση είναι στην πραγματικότητα ευλογία και όχι λόγος να καταφεύγουμε σε τεράστιους πυρήνες: τρία στοιβαγμένα επίπεδα $3 \times 3$ επιτυγχάνουν το ίδιο υποδεκτικό πεδίο $7 \times 7$ με ένα μόνο επίπεδο $7 \times 7$, χρησιμοποιώντας πολύ λιγότερες παραμέτρους ($3 \times 9 = 27$ βάρη έναντι 49 ανά ζεύγος καναλιών) ενώ παρεμβάλλουν τρεις μη γραμμικότητες αντί για μία.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Υποδεκτικό πεδίο μιας μονάδας μετά από τρία μονοδιάστατα επίπεδα με μέγεθος πυρήνα 3. Τα σκιασμένα κελιά είναι οι μονάδες από τις οποίες εξαρτάται η κορυφαία μονάδα, και οι γραμμές συνδέουν καθεμία από αυτές με τις εισόδους της. Κάθε επίπεδο διευρύνει το υποδεκτικό πεδίο κατά 2, φτάνοντας τις 7 εισόδους.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Ενεργά Υποδεκτικά Πεδία

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
Λάβετε υπόψη ότι το $r$ μετρά το *θεωρητικό* υποδεκτικό πεδίο (κάθε εικονοστοιχείο εισόδου με *οποιοδήποτε* μονοπάτι προς μια μονάδα εξόδου) χωρίς να λέει πόσο ισχυρά συνεισφέρει στην πραγματικότητα κάθε εικονοστοιχείο. Μετρώντας απευθείας τις κλίσεις εισόδου, οι :citet:`Luo.Li.Urtasun.ea.2016` έδειξαν ότι το *ενεργό* (effective) υποδεκτικό πεδίο φθίνει απότομα μακριά από το κέντρο ακολουθώντας περίπου μια Γκαουσιανή καμπύλη καμπάνας, τόσο σε ανεκπαίδευτα όσο και σε εκπαιδευμένα δίκτυα. Οι βαθιές μονάδες μπορούν θεωρητικά να βλέπουν μακριά στην εικόνα, αλλά η ευαισθησία τους συγκεντρώνεται έντονα κοντά στο κέντρο.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Η μέτρηση των μονοπατιών εξηγεί το σχήμα καμπάνας. Μια είσοδος στο άκρο του υποδεκτικού πεδίου φτάνει στη μονάδα εξόδου κατά μήκος μιας μόνο αλυσίδας συνδέσεων, ενώ η κεντρική είσοδος μπορεί να ακολουθήσει πολλές διαδρομές μέσω των ενδιάμεσων επιπέδων. Το :numref:`fig_conv_rf_paths` τα μετρά για τον πύργο του :numref:`fig_conv_receptive_field`: η κεντρική είσοδος έχει επτά μονοπάτια προς την κορυφαία μονάδα και κάθε ακραία είσοδος έχει ένα. Εάν όλα τα βάρη έχουν παρόμοιο μέγεθος, η επιρροή μιας εισόδου αυξάνεται με τον αριθμό των μονοπατιών της.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Αριθμός μονοπατιών από κάθε μονάδα προς την κορυφαία μονάδα του :numref:`fig_conv_receptive_field` (τρία μονοδιάστατα επίπεδα με μέγεθος πυρήνα 3). Κάθε πλήθος είναι το άθροισμα των πληθών των μονάδων που τροφοδοτεί το κελί. Τα πιο σκούρα κελιά έχουν περισσότερα μονοπάτια από τα άλλα κελιά του επιπέδου τους, και οι ράβδοι επαναλαμβάνουν τα πλήθη της γραμμής εισόδου: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Δύο πρακτικά κενά παραμένουν στο βασικό μας επίπεδο: κάθε συνέλιξη χωρίς συμπλήρωση συρρικνώνει τις χωρικές διαστάσεις ενώ υποδειγματοληπτεί τα οριακά εικονοστοιχεία, και ένας μεμονωμένος πυρήνας ανιχνεύει μόνο ένα μοτίβο. Αντιμετωπίζουμε το χωρικό μέγεθος και τα όρια στη συνέχεια στην ενότητα :numref:`sec_padding`, και τους χάρτες χαρακτηριστικών πολλαπλών καναλιών στην ενότητα :numref:`sec_channels`.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## Σύνοψη

<!--en:92688a0c kind=paragraph-->
A convolutional layer applies two-dimensional cross-correlation with a learned
kernel plus a scalar bias, sliding the kernel across every valid window of the
input. A simple $1 \times 2$ finite-difference kernel `[1, -1]` detects vertical
edges anywhere in an image, and gradient descent recovers those exact weights
from input–output data alone, just as deep CNNs learn entire hierarchies of
filters end to end. Mathematical convolution flips the kernel, making no
difference once kernels are learned. A convolution can be evaluated as a single
matrix multiplication (*im2col*), and stacking $L$ convolutional
layers widens each unit's receptive field linearly by $k_i - 1$ per layer.
<!--tr:92688a0c-->
Ένα συνελικτικό επίπεδο εφαρμόζει δισδιάστατη ετεροσυσχέτιση με έναν εκπαιδεύσιμο πυρήνα συν μια βαθμωτή πόλωση, ολισθαίνοντας τον πυρήνα σε κάθε έγκυρο παράθυρο της εισόδου. Ένας απλός πυρήνας πεπερασμένης διαφοράς $1 \times 2$ `[1, -1]` ανιχνεύει κατακόρυφες ακμές οπουδήποτε σε μια εικόνα, και η κατάβαση κλίσης ανακτά αυτά ακριβώς τα βάρη μόνο από δεδομένα εισόδου–εξόδου, όπως ακριβώς τα βαθιά CNN μαθαίνουν ολόκληρες ιεραρχίες φίλτρων από άκρη σε άκρη. Η μαθηματική συνέλιξη αντιστρέφει τον πυρήνα, κάτι που δεν κάνει καμία διαφορά εφόσον οι πυρήνες εκπαιδεύονται. Μια συνέλιξη μπορεί να υπολογιστεί ως ένας μόνο πολλαπλασιασμός πινάκων (*im2col*), και η στοίβαξη $L$ συνελικτικών επιπέδων διευρύνει γραμμικά το υποδεκτικό πεδίο κάθε μονάδας κατά $k_i - 1$ ανά επίπεδο.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Γρήγορη Εκτέλεση: Συνέλιξη μέσω Πολλαπλασιασμού Πινάκων"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[Ο Μετασχηματισμός im2col (Εικόνα σε Στήλες)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Ένας πυκνός GEMM:** Ισοπεδώστε κάθε παράθυρο $k_\textrm{h} \times k_\textrm{w}$ σε γραμμή του `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ και πολλαπλασιάστε με $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Συμβιβασμός:** Μέγιστη ταχύτητα GPU Tensor Core με κόστος $k_\textrm{h} k_\textrm{w}\times$ αντιγραφή τεμαχίων στη μνήμη.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Δυϊκή Όψη: Ξεδίπλωμα του Πυρήνα σε Ταινιωτό Πίνακα"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Πολλαπλασιασμός με Πίνακα Βαρών Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Αντί να ξεδιπλώνουμε την εικόνα εισόδου σε αντιγραμμένα τεμάχια, μπορούμε να ξεδιπλώσουμε τον *πυρήνα* σε έναν αραιό ταινιωτό πίνακα $\mathbf{W}$ που πολλαπλασιάζει το ισοπεδωμένο διάνυσμα εισόδου $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Μορφή πίνακα–διανύσματος:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, όπου ο $\mathbf{W}$ έχει $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ γραμμές και $n_\textrm{h} n_\textrm{w}$ στήλες.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Σύνδεση με §{sec_why-conv} και οπισθοδιάδοση:** Κάθε γραμμή του $\mathbf{W}$ περιέχει τα ίδια $k_\textrm{h} k_\textrm{w}$ στοιχεία πυρήνα μετατοπισμένα (πλαίσιο c), καθιστώντας το οπίσθιο πέρασμα ως προς $\mathbf{X}$ απλό πολλαπλασιασμό με $\mathbf{W}^\top$ (ανάστροφη συνέλιξη).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Χάρτες Χαρακτηριστικών και Υποδεκτικά Πεδία κατά Βάθος"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Γραμμική Αύξηση Υποδεκτικού Πεδίου σε Στοιβαγμένα Επίπεδα]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
Το **υποδεκτικό πεδίο** μιας μονάδας είναι η περιοχή της αρχικής εικόνας εισόδου της οποίας τα εικονοστοιχεία μπορούν να επηρεάσουν την ενεργοποίησή της κατά το εμπρόσθιο πέρασμα:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Ένα επίπεδο ($L=1$):** Ένας πυρήνας $k_1 \times k_1$ βλέπει ένα τεμάχιο εισόδου $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Στοίβαξη $L$ επιπέδων διασκελισμού 1:** Το επίπεδο $i$ με μέγεθος πυρήνα $k_i$ φτάνει $k_i - 1$ εικονοστοιχεία πιο μακριά ανά άξονα από ένα στοιχείο του επιπέδου $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ στοιβαγμένα επίπεδα $3 \times 3$:** Κάθε επίπεδο προσθέτει $3 - 1 = 2$ εικονοστοιχεία, δίνοντας πλευρά $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Γιατί Βαθιές Στοίβες 3×3 Κερδίζουν Έναν Μεγάλο Πυρήνα"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Αποδοτικότητα Παραμέτρων & Μη Γραμμική Εκφραστικότητα]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Ίδιο υποδεκτικό πεδίο $7 \times 7$:** Τρία στοιβαγμένα επίπεδα $3 \times 3$ ($r = 1 + 2 + 2 + 2 = 7$) βλέπουν το ίδιο τεμάχιο $7 \times 7$ με ένα επίπεδο $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% λιγότερες παράμετροι:** $3 \times 9 c^2 = 27 c^2$ βάρη αντί για $49 c^2$ σε $c$ κανάλια.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Πλουσιότερη εκφραστικότητα:** Συνθέτει **3 μη γραμμικές ενεργοποιήσεις** αντί 1 (ο βασικός κανόνας VGG στην §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Θεωρητικό έναντι Ενεργού Υποδεκτικού Πεδίου"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Γκαουσιανή Φθίση Ευαισθησίας Μακριά από το Κέντρο]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Θεωρητικό ΥΠ ($r = 2L+1$):** Πλαίσιο οριοθέτησης όλων των εικονοστοιχείων με *τουλάχιστον ένα* μονοπάτι προς την έξοδο.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Κεντρικό οριακό θεώρημα:** Τα κεντρικά εικονοστοιχεία συνδέονται μέσω εκθετικά πολλών μονοπατιών, ενώ οι γωνίες μέσω **1 μονοπατιού**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Ενεργό ΥΠ (Luo et al., 2016):** Το μέτρο της κλίσης εισόδου $\partial y / \partial \mathbf{X}$ φθίνει σε Γκαουσιανή καμπύλη με $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Έλεγχος Κατανόησης: Ετεροσυσχέτιση & Υποδεκτικά Πεδία"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Έλεγχος Κατανόησης]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Ελέγξτε την κατανόησή σας στα σχήματα εξόδου, το `im2col` και την αριθμητική υποδεκτικών πεδίων:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Διαστάσεις πίνακα im2col:** Εφαρμόζετε έναν πυρήνα $3 \times 3$ ενός καναλιού χωρίς συμπλήρωση σε εικόνα $10 \times 10$ μέσω `im2col`. Ποιο είναι το χωρικό σχήμα εξόδου και ποιες οι ακριβείς διαστάσεις του πίνακα `patches` και του ισοπεδωμένου πυρήνα;

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Βάθος για κάλυψη του Fashion-MNIST:** Με διασκελισμό 1, πόσα στοιβαγμένα επίπεδα $3 \times 3$ απαιτούνται ώστε το θεωρητικό υποδεκτικό πεδίο μιας μονάδας να καλύπτει ολόκληρη εικόνα $28 \times 28$; Πόσα με επίπεδα $5 \times 5$;

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Μετατόπιση ορίου απώλειας ακμών:** Στο πείραμα εκμάθησης κατακόρυφων ακμών $6 \times 8$, γιατί η απώλεια ξεκίνησε από $\approx 12$, και ποια θα ήταν η αρχική απώλεια αν η μαύρη ζώνη εκτεινόταν έως το δεξί άκρο (`X[:, 2:8] = 0`);

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Απαντήσεις:**
(1) Το σχήμα εξόδου είναι $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ έξοδοι). Ο πίνακας `patches` έχει σχήμα $(64, 9)$ και πολλαπλασιάζει το διάνυσμα πυρήνα $(9, 1)$.
(2) Για $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ επίπεδα. Για $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ επίπεδα.
(3) Κάθε μία από τις 6 γραμμές είχε δύο ακμές $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Η επέκταση της μαύρης ζώνης στο δεξί όριο εξαλείφει τη δεξιά ακμή μαύρου-προς-λευκό, αφήνοντας μία ακμή ανά γραμμή και υποδιπλασιάζοντας την αρχική απώλεια σε $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Σύνοψη & Βασικά Συμπεράσματα"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Σύνθεση Ενότητας]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **2D Ετεροσυσχέτιση & Εκμάθηση Ακμών:** Ολισθαίνει τον $\mathbf{K}$ σε έγκυρα παράθυρα ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$)· η SGD μαθαίνει το $[1, -1]$ σε 9 βήματα.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Πολλαπλασιασμός im2col:** Ανταλλάσσει $k_\textrm{h} k_\textrm{w}\times$ αντιγραφή τεμαχίων με μέγιστη ρυθμαπόδοση GEMM στην GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Στοιβαγμένα Υποδεκτικά Πεδία:** $r = 1 + \sum (k_i - 1) = 2L+1$ για στοίβες $3 \times 3$, με Γκαουσιανή κεντρική ευαισθησία.

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
- "banded matrix" → «ταινιακός πίνακας».
- "effective receptive field" → «ενεργό (effective) υποδεκτικό πεδίο» on first mention, abbreviated «ΥΠ» on dense slides where English uses "RF".
- "Gaussian" → «Γκαουσιανή / Γκαουσιανό».

=== NOTES ===
- 24a25ec4: math tokens (`$\mathbf{Y}$`, `$2 \times 2$`) reordered to follow natural Greek word order (noun + symbol + dimensions).
