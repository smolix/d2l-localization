<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=el rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Συνέλιξη ως πολλαπλασιασμός πινάκων

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
Οι ένθετοι βρόχοι της `corr2d` κάνουν τα μαθηματικά διαφανή, όμως η επίσκεψη ενός εικονοστοιχείου εξόδου τη φορά σε Python είναι υπερβολικά αργή για πραγματικά φορτία εργασίας. Μια κλασική λύση, που εξακολουθεί να είναι ένας από τους αλγορίθμους μεταξύ των οποίων επιλέγουν οι βιβλιοθήκες, είναι να διατυπώσουμε την ετεροσυσχέτιση ως έναν μοναδικό μεγάλο πολλαπλασιασμό πινάκων (matrix multiplication), δηλαδή ακριβώς την πράξη που οι GPU εκτελούν με τη μέγιστη ταχύτητά τους. Παρατηρήστε ότι κάθε καταχώριση της εξόδου στην :numref:`fig_correlation` είναι απλώς το εσωτερικό γινόμενο του ισοπεδωμένου πυρήνα $k_\textrm{h} k_\textrm{w}$ στοιχείων με το αντίστοιχο ισοπεδωμένο παράθυρο της εισόδου. Αν εξαγάγουμε όλα τα $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ παράθυρα και τα στοιβάξουμε ως γραμμές, προκύπτει ένας πίνακας ψηφίδων· πολλαπλασιάζοντάς τον με τον ισοπεδωμένο πυρήνα υπολογίζουμε ολόκληρη την ετεροσυσχέτιση με μία κίνηση. Αυτός ο καθιερωμένος μετασχηματισμός διάταξης ονομάζεται *im2col* («εικόνα σε στήλες», ή εδώ σε γραμμές). Η :numref:`fig_conv_im2col` τον εκτελεί πάνω στο παράδειγμα της :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![Η ετεροσυσχέτιση της :numref:`fig_correlation` ως ένα μοναδικό γινόμενο πινάκων. Καθένα από τα τέσσερα παράθυρα $2 \times 2$ της εισόδου γίνεται γραμμή του πίνακα ψηφίδων, και ο πολλαπλασιασμός με τον ισοπεδωμένο πυρήνα δίνει την ισοπεδωμένη έξοδο. Τα σκιασμένα κελιά ακολουθούν το πρώτο παράθυρο ως τη γραμμή του και ως την έξοδο 19. Το κεντρικό εικονοστοιχείο 4 ανήκει και στα τέσσερα παράθυρα, γι' αυτό εμφανίζεται σε κάθε γραμμή.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
Το αντάλλαγμα είναι η επιπλέον μνήμη εργασίας, αφού κάθε εσωτερικό εικονοστοιχείο αντιγράφεται σε έως και $k_\textrm{h} k_\textrm{w}$ ψηφίδες. Παρακάτω επαληθεύουμε στην :numref:`fig_correlation` ότι αυτός ο μοναδικός πολλαπλασιασμός πινάκων συμφωνεί ακριβώς με τη `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
Η απόκλιση είναι ταυτοτικά μηδέν: ένα μοναδικό γινόμενο πινάκων αναπαράγει τέλεια τη `corr2d`.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
Ισοδύναμα, μπορούμε να ξεδιπλώσουμε τον *πυρήνα* αντί για την είσοδο, σε έναν αραιό ταινιωτό πίνακα $\mathbf{W}$ που πολλαπλασιάζει την ισοπεδωμένη εικόνα — το ακριβές δισδιάστατο ανάλογο του πλαισίου (c) της :numref:`fig_why_conv_weight_sharing`, όπου κάθε γραμμή επαναλαμβάνει τα ίδια λίγα βάρη του πυρήνα. Η θεώρηση της συνέλιξης ως ενός δομημένου πυκνού επιπέδου μας φέρνει πίσω στην εξαγωγή της :numref:`sec_why-conv` (και οι ασκήσεις κατασκευάζουν ρητά το $\mathbf{W}$).

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## Χάρτες χαρακτηριστικών και υποδεκτικά πεδία

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
Ένα μεμονωμένο συνελικτικό επίπεδο βλέπει μόνο μια ψηφίδα του μεγέθους του πυρήνα του, ενώ για την αναγνώριση ολόκληρων αντικειμένων απαιτείται η ενσωμάτωση πλαισίου σε πολύ ευρύτερες περιοχές. Θυμηθείτε από την :numref:`subsec_why-conv-channels` ότι η έξοδος ενός συνελικτικού επιπέδου ονομάζεται *χάρτης χαρακτηριστικών* (όπως ο χάρτης κατακόρυφων ακμών `Y`). Για οποιοδήποτε στοιχείο $x$ ενός χάρτη χαρακτηριστικών, το *υποδεκτικό πεδίο* του είναι η περιοχή της εικόνας εισόδου (ή ενός προηγούμενου επιπέδου) της οποίας οι τιμές μπορούν να επηρεάσουν το $x$ κατά το εμπρόσθιο πέρασμα — δανειζόμαστε τον όρο από τη νευροφυσιολογία, όπως στην :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Στοιβαγμένα επίπεδα

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
Στην :numref:`fig_correlation`, η σκιασμένη έξοδος `19` έχει υποδεκτικό πεδίο $2 \times 2$ που καλύπτει τα τέσσερα επάνω αριστερά εικονοστοιχεία του $\mathbf{X}$. Στοιβάξτε τώρα ένα *δεύτερο* συνελικτικό επίπεδο $2 \times 2$ πάνω στην έξοδο $2 \times 2$ $\mathbf{Y}$, ώστε να παραχθεί ένα μοναδικό βαθμωτό μέγεθος $z$: το $z$ εξαρτάται από και τα τέσσερα στοιχεία του $\mathbf{Y}$, τα οποία μαζί εξαρτώνται από και τα εννέα στοιχεία του $\mathbf{X}$. Η στοίβαξη επιπέδων διευρύνει ομαλά το υποδεκτικό πεδίο με το βάθος.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Πόσο γρήγορα αυξάνεται; Κάθε συνελικτικό επίπεδο $k \times k$ φτάνει $k - 1$ εικονοστοιχεία πιο μακριά από ένα μεμονωμένο στοιχείο της εισόδου του, και έτσι η στοίβαξη $L$ επιπέδων με μεγέθη πυρήνα $k_1, \ldots, k_L$ δίνει υποδεκτικό πεδίο με μήκος πλευράς

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
Για μια στοίβα $L$ διαδοχικών επιπέδων $3 \times 3$, κάθε επίπεδο προσθέτει 2 εικονοστοιχεία και καλύπτει $(2L + 1) \times (2L + 1)$ εισόδους: $5 \times 5$ μετά από δύο επίπεδα και $7 \times 7$ μετά από τρία (:numref:`fig_conv_receptive_field`). Αυτή η γραμμική αύξηση είναι στην πραγματικότητα ευλογία και όχι λόγος να καταφεύγουμε σε τεράστιους πυρήνες: τρία στοιβαγμένα επίπεδα $3 \times 3$ πετυχαίνουν το ίδιο υποδεκτικό πεδίο $7 \times 7$ με ένα μοναδικό επίπεδο $7 \times 7$ χρησιμοποιώντας πολύ λιγότερες παραμέτρους ($3 \times 9 = 27$ βάρη έναντι 49 ανά ζεύγος καναλιών), ενώ παρεμβάλλουν τρεις μη γραμμικότητες αντί για μία.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Υποδεκτικό πεδίο μιας μονάδας μετά από τρία μονοδιάστατα επίπεδα με μέγεθος πυρήνα 3. Τα σκιασμένα κελιά είναι οι μονάδες από τις οποίες εξαρτάται η επάνω μονάδα, και οι γραμμές συνδέουν καθεμία από αυτές με τις εισόδους της. Κάθε επίπεδο διευρύνει το υποδεκτικό πεδίο κατά 2, φτάνοντας τις 7 εισόδους.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Αποτελεσματικά υποδεκτικά πεδία

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
Έχετε κατά νου ότι το $r$ μετρά το *θεωρητικό* υποδεκτικό πεδίο (κάθε εικονοστοιχείο εισόδου που έχει *οποιαδήποτε* διαδρομή προς μια μονάδα εξόδου), χωρίς να λέει πόσο έντονα συνεισφέρει στην πράξη κάθε εικονοστοιχείο. Μετρώντας απευθείας τις κλίσεις ως προς την είσοδο, οι :citet:`Luo.Li.Urtasun.ea.2016` έδειξαν ότι το *αποτελεσματικό* υποδεκτικό πεδίο φθίνει απότομα μακριά από το κέντρο, με περίπου γκαουσιανή καμπύλη καμπάνας, τόσο σε μη εκπαιδευμένα όσο και σε εκπαιδευμένα δίκτυα. Οι βαθιές μονάδες μπορούν θεωρητικά να βλέπουν μακριά μέσα στην εικόνα, αλλά η ευαισθησία τους συγκεντρώνεται έντονα κοντά στο κέντρο.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Η καταμέτρηση των διαδρομών εξηγεί το σχήμα της καμπάνας. Μια είσοδος στην άκρη του υποδεκτικού πεδίου φτάνει στη μονάδα εξόδου μέσω μιας μοναδικής αλυσίδας συνδέσεων, ενώ η κεντρική είσοδος μπορεί να ακολουθήσει πολλές διαδρομές μέσα από τα ενδιάμεσα επίπεδα. Η :numref:`fig_conv_rf_paths` τις μετρά για τον πύργο της :numref:`fig_conv_receptive_field`: η κεντρική είσοδος έχει επτά διαδρομές προς την επάνω μονάδα και κάθε είσοδος στην άκρη έχει μία. Αν όλα τα βάρη έχουν παρόμοιο μέγεθος, η επιρροή μιας εισόδου αυξάνεται με τον αριθμό των διαδρομών της.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Αριθμός διαδρομών από κάθε μονάδα προς την επάνω μονάδα της :numref:`fig_conv_receptive_field` (τρία μονοδιάστατα επίπεδα με μέγεθος πυρήνα 3). Κάθε πλήθος είναι το άθροισμα των πλήθων των μονάδων που τροφοδοτεί το κελί. Τα πιο σκούρα κελιά έχουν περισσότερες διαδρομές από τα υπόλοιπα κελιά του επιπέδου τους, και οι ράβδοι επαναλαμβάνουν τα πλήθη της σειράς εισόδου: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Στο βασικό μας επίπεδο μένουν δύο πρακτικά κενά: κάθε συνέλιξη χωρίς συμπλήρωση συρρικνώνει τις χωρικές διαστάσεις και ταυτόχρονα υποδειγματοληπτεί τα εικονοστοιχεία του περιγράμματος, και ένας μεμονωμένος πυρήνας ανιχνεύει μόνο ένα μοτίβο. Το χωρικό μέγεθος και τα περιγράμματα τα αντιμετωπίζουμε στη συνέχεια στην :numref:`sec_padding`, και τους χάρτες χαρακτηριστικών πολλών καναλιών στην :numref:`sec_channels`.

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
Ένα συνελικτικό επίπεδο εφαρμόζει δισδιάστατη ετεροσυσχέτιση με έναν πυρήνα που μαθαίνεται συν μια βαθμωτή πόλωση, ολισθαίνοντας τον πυρήνα σε κάθε έγκυρο παράθυρο της εισόδου. Ένας απλός πυρήνας πεπερασμένης διαφοράς $1 \times 2$, ο `[1, -1]`, ανιχνεύει κατακόρυφες ακμές οπουδήποτε σε μια εικόνα, και η κατάβαση κλίσης ανακτά ακριβώς αυτά τα βάρη μόνο από δεδομένα εισόδου–εξόδου, όπως ακριβώς τα βαθιά CNN μαθαίνουν ολόκληρες ιεραρχίες φίλτρων από άκρο σε άκρο. Η μαθηματική συνέλιξη αναστρέφει τον πυρήνα, κάτι που δεν κάνει καμία διαφορά αφού οι πυρήνες μαθαίνονται. Μια συνέλιξη μπορεί να υπολογιστεί ως ένας μοναδικός πολλαπλασιασμός πινάκων (*im2col*), και η στοίβαξη $L$ συνελικτικών επιπέδων διευρύνει γραμμικά το υποδεκτικό πεδίο κάθε μονάδας κατά $k_i - 1$ ανά επίπεδο.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Γρήγορη εκτέλεση: συνέλιξη μέσω πολλαπλασιασμού πινάκων"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[Ο μετασχηματισμός im2col (εικόνα σε στήλες)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Ένας μοναδικός πυκνός πολλαπλασιασμός GEMM:** ισοπεδώνουμε κάθε παράθυρο $k_\textrm{h} \times k_\textrm{w}$ σε μια γραμμή των `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ και πολλαπλασιάζουμε με το $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Αντάλλαγμα:** μέγιστη ταχύτητα των Tensor Core της GPU, με κόστος $k_\textrm{h} k_\textrm{w}\times$ διπλασιασμό των ψηφίδων στη μνήμη εργασίας.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Διπλή όψη: ξεδίπλωμα του πυρήνα σε ταινιωτό πίνακα"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Πολλαπλασιασμός με πίνακα βαρών Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Αντί να ξεδιπλώσουμε την εικόνα εισόδου σε διπλασιασμένες ψηφίδες, μπορούμε να ξεδιπλώσουμε τον *πυρήνα* σε έναν αραιό ταινιωτό πίνακα $\mathbf{W}$ που πολλαπλασιάζει το ισοπεδωμένο διάνυσμα εισόδου $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Μορφή πίνακα–διανύσματος:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, όπου ο $\mathbf{W}$ έχει $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ γραμμές και $n_\textrm{h} n_\textrm{w}$ στήλες.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Σύνδεση με την §{sec_why-conv} και την οπισθοδιάδοση:** κάθε γραμμή του $\mathbf{W}$ περιέχει τα ίδια $k_\textrm{h} k_\textrm{w}$ στοιχεία του πυρήνα μετατοπισμένα κατά στήλες (πλαίσιο c), οπότε το οπίσθιο πέρασμα ως προς το $\mathbf{X}$ είναι απλώς πολλαπλασιασμός με το $\mathbf{W}^\top$ (ανάστροφη συνέλιξη).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Χάρτες χαρακτηριστικών και υποδεκτικά πεδία ανά βάθος"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Γραμμική αύξηση του υποδεκτικού πεδίου στα στοιβαγμένα επίπεδα]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
Το **υποδεκτικό πεδίο** μιας μονάδας είναι η περιοχή της αρχικής εικόνας εισόδου της οποίας τα εικονοστοιχεία μπορούν να επηρεάσουν την ενεργοποίηση της μονάδας κατά το εμπρόσθιο πέρασμα:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Ένα επίπεδο ($L=1$):** ένας πυρήνας $k_1 \times k_1$ βλέπει μια ψηφίδα εισόδου $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Στοίβαξη $L$ επιπέδων με διασκελισμό 1:** το επίπεδο $i$ με μέγεθος πυρήνα $k_i$ φτάνει $k_i - 1$ εικονοστοιχεία πιο μακριά κατά κάθε άξονα από ένα μεμονωμένο στοιχείο του επιπέδου $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ στοιβαγμένα επίπεδα $3 \times 3$:** κάθε επίπεδο προσθέτει $3 - 1 = 2$ εικονοστοιχεία, δίνοντας μήκος πλευράς $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Γιατί οι βαθιές στοίβες πυρήνων 3×3 υπερτερούν ενός μεγάλου πυρήνα"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Αποδοτικότητα παραμέτρων και μη γραμμική εκφραστικότητα]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Ίδιο υποδεκτικό πεδίο $7 \times 7$:** τρία στοιβαγμένα επίπεδα $3 \times 3$ ($r = 1 + 2 + 2 + 2 = 7$) βλέπουν ακριβώς την ίδια ψηφίδα $7 \times 7$ με ένα επίπεδο $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% λιγότερες παράμετροι:** $3 \times 9 c^2 = 27 c^2$ βάρη αντί για $49 c^2$ σε $c$ κανάλια.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Πλουσιότερη εκφραστικότητα:** συνδυάζει **3 μη γραμμικές ενεργοποιήσεις** αντί για 1 (ο βασικός κανόνας σχεδίασης του VGG στην §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Θεωρητικά και αποτελεσματικά υποδεκτικά πεδία"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Γκαουσιανή πτώση της ευαισθησίας μακριά από το κέντρο]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Θεωρητικό υποδεκτικό πεδίο ($r = 2L+1$):** το πλαίσιο οριοθέτησης όλων των εικονοστοιχείων που έχουν *τουλάχιστον μία* διαδρομή προς τη μονάδα εξόδου.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Κεντρικό οριακό θεώρημα:** τα κεντρικά εικονοστοιχεία συνδέονται μέσω εκθετικά πολλών διαδρομών, ενώ τα εικονοστοιχεία των γωνιών συνδέονται μέσω μόνο **1 διαδρομής**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Αποτελεσματικό υποδεκτικό πεδίο (Luo et al., 2016):** το μέγεθος της κλίσης ως προς την είσοδο $\partial y / \partial \mathbf{X}$ φθίνει σε γκαουσιανή καμπύλη καμπάνας με τυπική απόκλιση $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Έλεγχος κατανόησης: ετεροσυσχέτιση και υποδεκτικά πεδία"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Έλεγχος κατανόησης]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Ελέγξτε την κατανόησή σας σχετικά με τα σχήματα εξόδου, την `im2col` και την αριθμητική των υποδεκτικών πεδίων:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Διαστάσεις του πίνακα im2col:** εφαρμόζετε έναν πυρήνα $3 \times 3$ ενός καναλιού, χωρίς συμπλήρωση, σε μια εικόνα $10 \times 10$ μέσω της `im2col`. Ποιο είναι το χωρικό σχήμα της εξόδου και ποιες ακριβώς οι διαστάσεις του πίνακα `patches` και του ισοπεδωμένου διανύσματος πυρήνα;

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Βάθος για να καλυφθεί το Fashion-MNIST:** με διασκελισμό 1, πόσα στοιβαγμένα επίπεδα $3 \times 3$ απαιτούνται ώστε το θεωρητικό υποδεκτικό πεδίο μιας μονάδας να καλύπτει ολόκληρη μια εικόνα $28 \times 28$; Πόσα με επίπεδα $5 \times 5$;

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Μετατόπιση ορίου και απώλεια ακμών:** στο πείραμά μας εκμάθησης κατακόρυφης ακμής $6 \times 8$, γιατί η απώλεια ξεκίνησε στο $\approx 12$, και ποια θα ήταν η αρχική απώλεια αν η μαύρη ζώνη έφτανε μέχρι το δεξί άκρο (`X[:, 2:8] = 0`);

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Απαντήσεις:**
(1) Το σχήμα της εξόδου είναι $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ έξοδοι). Ο πίνακας `patches` έχει σχήμα $(64, 9)$ και πολλαπλασιάζεται με το διάνυσμα πυρήνα $(9, 1)$.
(2) Για $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ επίπεδα. Για $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ επίπεδα.
(3) Κάθε μία από τις 6 γραμμές είχε δύο ακμές $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Αν η μαύρη ζώνη φτάσει μέχρι το δεξί περίγραμμα, εξαφανίζεται η δεξιά ακμή από μαύρο σε λευκό, μένει μία ακμή ανά γραμμή και η αρχική απώλεια υποδιπλασιάζεται, περίπου στο $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Σύνοψη και βασικά συμπεράσματα"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Σύνθεση της ενότητας]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Δισδιάστατη ετεροσυσχέτιση και εκμάθηση ακμών:** ολισθαίνει το $\mathbf{K}$ σε κάθε έγκυρο παράθυρο ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$)· η SGD μαθαίνει το $[1, -1]$ σε 9 βήματα.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Πολλαπλασιασμός πινάκων im2col:** ανταλλάσσει τον διπλασιασμό ψηφίδων $k_\textrm{h} k_\textrm{w}\times$ με τη μέγιστη ρυθμαπόδοση GEMM της GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Στοιβαγμένα υποδεκτικά πεδία:** $r = 1 + \sum (k_i - 1) = 2L+1$ για στοίβες $3 \times 3$, με γκαουσιανή ευαισθησία στο κέντρο.

=== DECISIONS ===
- "patch" → ψηφίδα (consistent with u01); "patch matrix" → πίνακας ψηφίδων; "workspace" → μνήμη εργασίας.
- "banded matrix" → ταινιωτός πίνακας; "flatten" → ισοπεδώνω; "unroll" → ξεδιπλώνω.
- "theoretical / effective receptive field" → θεωρητικό / αποτελεσματικό υποδεκτικό πεδίο; "Gaussian" → γκαουσιανός.
- "nonlinearity" → μη γραμμικότητα (glossary); "activation" → ενεργοποίηση.
- GEMM, im2col, Toeplitz, Tensor Core, VGG stay in Latin script.
- "trade-off" → αντάλλαγμα; "throughput" → ρυθμαπόδοση (glossary).
