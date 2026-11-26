<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=el rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Η συνέλιξη ως πολλαπλασιασμός πινάκων

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
Με τους εμφωλευμένους βρόχους της `corr2d` παρακολουθούμε εύκολα τον
μαθηματικό υπολογισμό, αλλά η επεξεργασία ενός εικονοστοιχείου εξόδου τη
φορά στην Python είναι υπερβολικά αργή για πρακτική χρήση. Μια κλασική λύση,
που παραμένει μία από τις επιλογές των βιβλιοθηκών, είναι να μετατρέψουμε την
ετεροσυσχέτιση σε έναν μόνο μεγάλο πολλαπλασιασμό πινάκων, ακριβώς την πράξη
που οι GPU εκτελούν με τη μέγιστη ταχύτητά τους. Παρατηρούμε ότι κάθε στοιχείο
εξόδου στο σχήμα :numref:`fig_correlation` είναι απλώς το εσωτερικό γινόμενο
του διανύσματος που προκύπτει από την επιπεδοποίηση του πυρήνα των
$k_\textrm{h} k_\textrm{w}$ στοιχείων με το αντίστοιχο επιπεδοποιημένο παράθυρο
εισόδου. Εξάγουμε και τα
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ παράθυρα και τα
τοποθετούμε ως γραμμές σε έναν πίνακα τμημάτων. Πολλαπλασιάζοντας αυτόν τον
πίνακα με τον επιπεδοποιημένο πυρήνα, υπολογίζουμε όλη την ετεροσυσχέτιση με
μία πράξη. Αυτή η καθιερωμένη αναδιάταξη λέγεται *im2col* («εικόνα σε στήλες»,
ή εδώ σε γραμμές). Το σχήμα :numref:`fig_conv_im2col` δείχνει την αναδιάταξη
για το παράδειγμα του σχήματος :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![Η ετεροσυσχέτιση του σχήματος :numref:`fig_correlation` ως ένα γινόμενο πινάκων. Καθένα από τα τέσσερα παράθυρα $2 \times 2$ της εισόδου γίνεται γραμμή του πίνακα τμημάτων. Ο πολλαπλασιασμός με τον επιπεδοποιημένο πυρήνα δίνει την έξοδο σε μορφή διανύσματος. Τα σκιασμένα κελιά δείχνουν το πρώτο παράθυρο, την αντίστοιχη γραμμή και την τιμή εξόδου 19. Το κεντρικό εικονοστοιχείο 4 ανήκει και στα τέσσερα παράθυρα, άρα εμφανίζεται σε κάθε γραμμή.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
Χρειαζόμαστε όμως επιπλέον προσωρινό χώρο στη μνήμη, αφού κάθε εσωτερικό
εικονοστοιχείο αντιγράφεται σε έως και $k_\textrm{h} k_\textrm{w}$ τμήματα.
Παρακάτω επαληθεύουμε, στο παράδειγμα του σχήματος :numref:`fig_correlation`,
ότι αυτός ο ένας πολλαπλασιασμός πινάκων δίνει ακριβώς το ίδιο αποτέλεσμα με
την `corr2d`.

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
Η διαφορά είναι ακριβώς μηδέν. Ένα γινόμενο πινάκων αναπαράγει πλήρως το
αποτέλεσμα της `corr2d`.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
Ισοδύναμα, μπορούμε να αναπτύξουμε τον *πυρήνα* αντί για την είσοδο σε έναν
αραιό πίνακα με ζώνες $\mathbf{W}$, τον οποίο πολλαπλασιάζουμε με την
επιπεδοποιημένη εικόνα. Αυτό είναι ακριβώς το δισδιάστατο ανάλογο του μέρους
(c) του σχήματος :numref:`fig_why_conv_weight_sharing`, όπου κάθε γραμμή
επαναλαμβάνει τα ίδια λίγα βάρη του πυρήνα. Βλέποντας τη συνέλιξη ως πυκνό
επίπεδο με ειδική δομή, επιστρέφουμε στην παραγωγή του συνελικτικού επιπέδου
στην ενότητα :numref:`sec_why-conv` (στις ασκήσεις κατασκευάζουμε ρητά τον
$\mathbf{W}$).

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
Ένα μόνο συνελικτικό επίπεδο βλέπει ένα τμήμα ίσο σε μέγεθος με τον πυρήνα
του, ενώ για να αναγνωρίσουμε ολόκληρα αντικείμενα πρέπει να συνδυάσουμε
πληροφορίες από πολύ ευρύτερες περιοχές. Όπως είδαμε στην ενότητα
:numref:`subsec_why-conv-channels`, η έξοδος ενός συνελικτικού επιπέδου
λέγεται *χάρτης χαρακτηριστικών* (όπως ο χάρτης κατακόρυφων ακμών `Y`).
Για κάθε στοιχείο $x$ ενός χάρτη χαρακτηριστικών, το *υποδεκτικό πεδίο*
του είναι η περιοχή της εικόνας εισόδου (ή ενός προηγούμενου επιπέδου) της
οποίας οι τιμές μπορούν να επηρεάσουν το $x$ κατά το εμπρόσθιο πέρασμα.
Δανειζόμαστε τον όρο από τη νευροφυσιολογία, όπως στο
:numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Στοίβαξη επιπέδων

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
Στο σχήμα :numref:`fig_correlation`, το σκιασμένο στοιχείο εξόδου `19` έχει
υποδεκτικό πεδίο $2 \times 2$ που καλύπτει τα τέσσερα πάνω αριστερά
εικονοστοιχεία του $\mathbf{X}$. Προσθέτουμε τώρα ένα *δεύτερο* συνελικτικό
επίπεδο $2 \times 2$ πάνω στην έξοδο $2 \times 2$ $\mathbf{Y}$ για να
πάρουμε ένα μόνο βαθμωτό μέγεθος $z$. Το $z$ εξαρτάται και από τα τέσσερα
στοιχεία του $\mathbf{Y}$, που μαζί εξαρτώνται και από τα εννέα στοιχεία
του $\mathbf{X}$. Με τη στοίβαξη επιπέδων το υποδεκτικό πεδίο διευρύνεται
σταδιακά καθώς αυξάνεται το βάθος.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Πόσο γρήγορα μεγαλώνει; Κάθε συνελικτικό επίπεδο $k \times k$ καλύπτει
$k - 1$ περισσότερα εικονοστοιχεία ανά άξονα από όσα καλύπτει ένα μόνο
στοιχείο της εισόδου του. Επομένως, στοιβάζοντας $L$ επίπεδα με μεγέθη πυρήνα
$k_1, \ldots, k_L$, παίρνουμε ένα υποδεκτικό πεδίο με μήκος πλευράς ίσο με

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
Σε μια στοίβα $L$ διαδοχικών επιπέδων $3 \times 3$, κάθε επίπεδο προσθέτει 2
εικονοστοιχεία στο μήκος της πλευράς. Έτσι καλύπτονται
$(2L + 1) \times (2L + 1)$ στοιχεία εισόδου, $5 \times 5$ μετά από δύο
επίπεδα και $7 \times 7$ μετά από τρία (:numref:`fig_conv_receptive_field`).
Αυτή η γραμμική αύξηση είναι πλεονέκτημα, γιατί μας επιτρέπει να αποφύγουμε
τους πολύ μεγάλους πυρήνες. Τρία διαδοχικά επίπεδα $3 \times 3$ πετυχαίνουν το
ίδιο υποδεκτικό πεδίο $7 \times 7$ με ένα μόνο επίπεδο $7 \times 7$ και πολύ
λιγότερες παραμέτρους ($3 \times 9 = 27$ βάρη έναντι 49 ανά ζεύγος καναλιών),
ενώ παρεμβάλλουν τρεις μη γραμμικότητες αντί για μία.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Υποδεκτικό πεδίο μιας μονάδας μετά από τρία μονοδιάστατα επίπεδα με μέγεθος πυρήνα 3. Τα σκιασμένα κελιά είναι οι μονάδες από τις οποίες εξαρτάται η μονάδα της κορυφής. Οι γραμμές συνδέουν καθεμία από αυτές με τις εισόδους της. Κάθε επίπεδο αυξάνει το πλάτος του υποδεκτικού πεδίου κατά 2, ώσπου να καλύψει 7 εισόδους.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Ενεργά υποδεκτικά πεδία

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
Το $r$ μετρά το *θεωρητικό* υποδεκτικό πεδίο, που περιλαμβάνει κάθε
εικονοστοιχείο εισόδου με *οποιαδήποτε* διαδρομή προς μια μονάδα εξόδου,
χωρίς να δείχνει πόσο συμβάλλει στην πράξη το καθένα. Μετρώντας απευθείας
τις κλίσεις ως προς την είσοδο, οι :citet:`Luo.Li.Urtasun.ea.2016` έδειξαν
ότι η ευαισθησία στο *ενεργό* υποδεκτικό πεδίο φθίνει απότομα όσο
απομακρυνόμαστε από το κέντρο, σχηματίζοντας μια κατά προσέγγιση
κωδωνοειδή καμπύλη Gauss, τόσο σε ανεκπαίδευτα όσο και σε εκπαιδευμένα δίκτυα.
Θεωρητικά, οι μονάδες στα βαθύτερα επίπεδα μπορούν να δουν μακριά μέσα
στην εικόνα, αλλά η ευαισθησία τους συγκεντρώνεται έντονα γύρω από το κέντρο.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Η καταμέτρηση των διαδρομών εξηγεί το κωδωνοειδές σχήμα. Ένα στοιχείο εισόδου
στην άκρη του υποδεκτικού πεδίου φτάνει στη μονάδα εξόδου μέσω μιας μόνο
αλυσίδας συνδέσεων, ενώ το κεντρικό στοιχείο μπορεί να ακολουθήσει πολλές
διαδρομές μέσα από τα ενδιάμεσα επίπεδα. Το σχήμα :numref:`fig_conv_rf_paths`
τις μετρά για τη στοίβα του σχήματος :numref:`fig_conv_receptive_field`.
Το κεντρικό στοιχείο εισόδου έχει επτά διαδρομές προς τη μονάδα της κορυφής,
ενώ κάθε ακραίο στοιχείο έχει μία. Αν όλα τα βάρη έχουν παρόμοιο απόλυτο
μέγεθος, η επίδραση ενός στοιχείου εισόδου αυξάνεται με τον αριθμό των
διαδρομών του.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Πλήθος διαδρομών από κάθε μονάδα προς τη μονάδα της κορυφής στο σχήμα :numref:`fig_conv_receptive_field` (τρία μονοδιάστατα επίπεδα με μέγεθος πυρήνα 3). Το πλήθος για κάθε κελί είναι το άθροισμα των αντίστοιχων πληθών για τις μονάδες που δέχονται την έξοδό του. Τα πιο σκούρα κελιά έχουν περισσότερες διαδρομές από τα άλλα κελιά του επιπέδου τους. Οι ράβδοι δείχνουν ξανά τα πλήθη της γραμμής εισόδου (1, 3, 6, 7, 6, 3, 1).](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Το βασικό μας επίπεδο έχει ακόμη δύο πρακτικούς περιορισμούς. Κάθε συνέλιξη
χωρίς συμπλήρωση μειώνει τις χωρικές διαστάσεις και δειγματοληπτεί
λιγότερο συχνά τα εικονοστοιχεία στα όρια, ενώ ένας μόνο πυρήνας ανιχνεύει
ένα μόνο μοτίβο. Στη συνέχεια εξετάζουμε τις χωρικές διαστάσεις και τα όρια
στην ενότητα :numref:`sec_padding` και τους χάρτες χαρακτηριστικών με
πολλαπλά κανάλια στην ενότητα :numref:`sec_channels`.

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
Ένα συνελικτικό επίπεδο ολισθαίνει τον πυρήνα που έμαθε πάνω σε κάθε έγκυρο
παράθυρο της εισόδου, υπολογίζει τη δισδιάστατη ετεροσυσχέτιση και προσθέτει
έναν βαθμωτό όρο πόλωσης. Ένας απλός πυρήνας πεπερασμένης διαφοράς
$1 \times 2$, ο `[1, -1]`, ανιχνεύει κατακόρυφες ακμές οπουδήποτε στην εικόνα.
Η κατάβαση κλίσης ανακτά αυτά ακριβώς τα βάρη μόνο από δεδομένα εισόδου και
εξόδου, όπως τα βαθιά συνελικτικά νευρωνικά δίκτυα μαθαίνουν ολόκληρες
ιεραρχίες φίλτρων από άκρο σε άκρο. Η μαθηματική συνέλιξη αντιστρέφει τον
πυρήνα, κάτι που δεν αλλάζει τι μπορεί να μάθει το επίπεδο. Μπορούμε να
υπολογίσουμε μια συνέλιξη με έναν μόνο πολλαπλασιασμό πινάκων (*im2col*).
Με τη στοίβαξη $L$ συνελικτικών επιπέδων, το υποδεκτικό πεδίο κάθε μονάδας
διευρύνεται γραμμικά, κατά $k_i - 1$ ανά επίπεδο.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Γρήγορη συνέλιξη με πολλαπλασιασμό πινάκων"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[Μετασχηματισμός im2col (εικόνα σε στήλες)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Ένας πολλαπλασιασμός πυκνών πινάκων (GEMM)** — Μετατρέπουμε κάθε παράθυρο $k_\textrm{h} \times k_\textrm{w}$ σε γραμμή του πίνακα `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ και πολλαπλασιάζουμε με $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Ταχύτητα και κόστος μνήμης** — Μέγιστη ταχύτητα στους Tensor Cores της GPU, με προσωρινό χώρο για τα αντίγραφα των τμημάτων έως $k_\textrm{h} k_\textrm{w}\times$ τον χώρο της εισόδου.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Ισοδύναμη μορφή με ανάπτυξη του πυρήνα σε πίνακα με ζώνες"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Πολλαπλασιασμός με πίνακα βαρών Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Αντί να αναπτύξουμε την εικόνα εισόδου σε τμήματα με επαναλαμβανόμενα στοιχεία, μπορούμε να αναπτύξουμε τον *πυρήνα* σε έναν αραιό πίνακα με ζώνες $\mathbf{W}$, που πολλαπλασιάζει το διάνυσμα της επιπεδοποιημένης εισόδου $\textrm{vec}(\mathbf{X})$.

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Γινόμενο πίνακα και διανύσματος** — $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, όπου ο $\mathbf{W}$ έχει $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ γραμμές και $n_\textrm{h} n_\textrm{w}$ στήλες.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Σύνδεση με την §{sec_why-conv} και την οπισθοδιάδοση** — Κάθε γραμμή του $\mathbf{W}$ περιέχει τα ίδια $k_\textrm{h} k_\textrm{w}$ στοιχεία του πυρήνα, μετατοπισμένα κατά μήκος των στηλών (μέρος c). Έτσι, το οπίσθιο πέρασμα ως προς τον $\mathbf{X}$ είναι απλώς πολλαπλασιασμός με τον $\mathbf{W}^\top$ (ανάστροφη συνέλιξη).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Χάρτες χαρακτηριστικών και υποδεκτικά πεδία σε διαφορετικά βάθη"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Γραμμική αύξηση του υποδεκτικού πεδίου με τη στοίβαξη επιπέδων]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
Το **υποδεκτικό πεδίο** μιας μονάδας είναι η περιοχή της αρχικής εικόνας εισόδου της οποίας τα εικονοστοιχεία μπορούν να επηρεάσουν την ενεργοποίηση της μονάδας κατά το εμπρόσθιο πέρασμα.

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Ένα επίπεδο ($L=1$)** — Ένας πυρήνας $k_1 \times k_1$ βλέπει ένα τμήμα εισόδου $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Στοίβαξη $L$ επιπέδων με διασκελισμό 1** — Το επίπεδο $i$, με μέγεθος πυρήνα $k_i$, καλύπτει $k_i - 1$ περισσότερα εικονοστοιχεία σε κάθε άξονα από ένα μόνο στοιχείο του επιπέδου $i-1$.
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **Στοίβαξη $L$ επιπέδων $3 \times 3$** — Κάθε επίπεδο προσθέτει $3 - 1 = 2$ εικονοστοιχεία στο μήκος της πλευράς, που γίνεται $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Γιατί μια βαθιά στοίβα πυρήνων 3×3 υπερέχει ενός μεγάλου πυρήνα"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Λιγότερες παράμετροι και μη γραμμική εκφραστικότητα]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Ίδιο υποδεκτικό πεδίο $7 \times 7$** — Τρία διαδοχικά επίπεδα $3 \times 3$ ($r = 1 + 2 + 2 + 2 = 7$) βλέπουν ακριβώς το ίδιο τμήμα $7 \times 7$ με ένα επίπεδο $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% λιγότερες παράμετροι** — $3 \times 9 c^2 = 27 c^2$ βάρη αντί για $49 c^2$ στα $c$ κανάλια.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Μεγαλύτερη εκφραστικότητα** — Η στοίβα συνθέτει **3 μη γραμμικές ενεργοποιήσεις** αντί για 1 (ο βασικός κανόνας σχεδιασμού του VGG στην §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Θεωρητικά και ενεργά υποδεκτικά πεδία"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Φθίση της ευαισθησίας από το κέντρο κατά Gauss]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Θεωρητικό υποδεκτικό πεδίο ($r = 2L+1$)** — Πλαίσιο οριοθέτησης όλων των εικονοστοιχείων με *τουλάχιστον μία* διαδρομή προς τη μονάδα εξόδου.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Κεντρικό οριακό θεώρημα** — Το πλήθος διαδρομών από τα κεντρικά εικονοστοιχεία αυξάνεται εκθετικά, ενώ από τα γωνιακά υπάρχει μόνο **1 διαδρομή**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Ενεργό υποδεκτικό πεδίο (Luo et al., 2016)** — Το μέγεθος της κλίσης ως προς την είσοδο $\partial y / \partial \mathbf{X}$ φθίνει κατά μήκος μιας κωδωνοειδούς καμπύλης Gauss με τυπική απόκλιση $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Έλεγχος κατανόησης της ετεροσυσχέτισης και των υποδεκτικών πεδίων"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Έλεγχος κατανόησης]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Ελέγξτε αν μπορείτε να υπολογίσετε σχήματα εξόδου, διαστάσεις πινάκων της `im2col` και μεγέθη υποδεκτικών πεδίων.

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Διαστάσεις πινάκων στην im2col** — Εφαρμόζετε έναν μονοκάναλο πυρήνα $3 \times 3$ χωρίς συμπλήρωση σε εικόνα $10 \times 10$ μέσω της `im2col`. Ποιο είναι το χωρικό σχήμα της εξόδου και ποιες ακριβώς είναι οι διαστάσεις του πίνακα `patches` και του διανύσματος του επιπεδοποιημένου πυρήνα;

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Βάθος για την κάλυψη εικόνας Fashion-MNIST** — Με διασκελισμό 1, πόσα διαδοχικά επίπεδα $3 \times 3$ χρειάζονται ώστε το θεωρητικό υποδεκτικό πεδίο μιας μονάδας να καλύπτει ολόκληρη εικόνα $28 \times 28$; Πόσα χρειάζονται με επίπεδα $5 \times 5$;

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Απώλεια αν το όριο της λωρίδας μετακινηθεί** — Στο πείραμα εκμάθησης κατακόρυφων ακμών με την εικόνα $6 \times 8$, γιατί η απώλεια ξεκίνησε από $\approx 12$ και πόση θα ήταν αρχικά αν η μαύρη λωρίδα έφτανε μέχρι το δεξί άκρο (`X[:, 2:8] = 0`);

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Απαντήσεις**
(1) Το σχήμα εξόδου είναι $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ στοιχεία εξόδου). Ο πίνακας `patches` έχει σχήμα $(64, 9)$ και πολλαπλασιάζει το διάνυσμα του πυρήνα διαστάσεων $(9, 1)$.
(2) Με $3 \times 3$ έχουμε $2L + 1 \ge 28 \implies L = 14$ επίπεδα. Με $5 \times 5$ έχουμε $4L + 1 \ge 28 \implies L = 7$ επίπεδα.
(3) Καθεμία από τις 6 γραμμές είχε δύο ακμές με τιμές $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Αν η μαύρη λωρίδα φτάνει στο δεξί όριο, η δεξιά ακμή από μαύρο σε λευκό εξαφανίζεται. Μένει μία ακμή ανά γραμμή και η αρχική απώλεια μειώνεται στο μισό, σε $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Σύνοψη και βασικά συμπεράσματα"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Ανακεφαλαίωση της ενότητας]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Δισδιάστατη ετεροσυσχέτιση και εκμάθηση ακμών** — Ο πυρήνας $\mathbf{K}$ ολισθαίνει στα έγκυρα παράθυρα ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$). Η κατάβαση κλίσης μαθαίνει το $[1, -1]$ σε 9 βήματα.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Πολλαπλασιασμός πινάκων με im2col** — Η αντιγραφή στοιχείων έως και $k_\textrm{h} k_\textrm{w}\times$ στα τμήματα επιτρέπει μέγιστη ρυθμαπόδοση GEMM στη GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Υποδεκτικά πεδία σε στοίβες επιπέδων** — $r = 1 + \sum (k_i - 1) = 2L+1$ για στοίβες $3 \times 3$, με ευαισθησία που φθίνει από το κέντρο κατά Gauss.

=== DECISIONS ===
- “flatten” → «επιπεδοποιώ» (convert to a vector); “unfold / unroll” → «αναπτύσσω» for the matrix construction; “patch matrix” → «πίνακας τμημάτων»; “banded matrix” → «πίνακας με ζώνες».
- “theoretical / effective receptive field” → «θεωρητικό / ενεργό υποδεκτικό πεδίο»; “path” → «διαδρομή»; “Gaussian” → «Gauss» in «καμπύλη Gauss» or «κατά Gauss», as appropriate.
- “workspace memory” → «προσωρινός χώρος στη μνήμη»; “nonlinearity” → «μη γραμμικότητα», following the glossary for the next unit.

=== NOTES ===
- 77d044db: It is the sensitivity across the effective receptive field that decays, as the next sentence and the slide explain.
- 1fb18ec4, 09efe4f0: The duplication factor is an upper bound, as stated in 76713479.
- 00011fb7: As in 456670c5, the slide says SGD, but the displayed experiment uses a full gradient update.
