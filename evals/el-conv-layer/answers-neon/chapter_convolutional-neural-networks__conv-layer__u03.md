<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=el rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## Ασκήσεις

<!--en:6f4ddec6 kind=exercise-->
1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6f4ddec6-->
1. [code] **Χειροποίητοι πυρήνες.** Σχεδιάστε μερικούς πυρήνες με το χέρι.
    1. Δεδομένου ενός διανύσματος κατεύθυνσης $\mathbf{v} = (v_1, v_2)$, εξαγάγετε
       έναν πυρήνα ανίχνευσης ακμών που ανιχνεύει ακμές ορθογώνιες στο $\mathbf{v}$,
       δηλαδή ακμές στη διεύθυνση $(v_2, -v_1)$.
    1. Εξαγάγετε έναν τελεστή πεπερασμένων διαφορών για τη δεύτερη παράγωγο.
       Ποιες δομές στις εικόνες αποκρίνονται πιο έντονα σε αυτόν;
    1. Ποιο είναι το ελάχιστο μέγεθος ενός πυρήνα για να προκύψει παράγωγος
       τάξης $d$;
    1. Πώς θα σχεδιάζατε έναν πυρήνα θόλωσης; Γιατί μπορεί να θέλατε να
       χρησιμοποιήσετε έναν τέτοιο πυρήνα;
    1. Φορτώστε μια φωτογραφία, μετατρέψτε τη σε ένα κανάλι και υποδειγματοληπτήστε
       την σε περίπου $100 \times 100$ εικονοστοιχεία (η `corr2d` εκτελεί βρόχους
       στην Python). Εφαρμόστε τον ανιχνευτή ακμών, τον πυρήνα θόλωσης και τον
       πυρήνα όξυνσης «δύο φορές ο ταυτοτικός μείον ο πυρήνας θόλωσης» και
       εμφανίστε κάθε αποτέλεσμα δίπλα στην αρχική εικόνα. Τι δείχνει ο χάρτης
       ακμών μιας φωτογραφίας που δεν μπορούσε να δείξει ο χάρτης ακμών της
       μαύρης λωρίδας;

    *Προσαρμοσμένο από το Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    σημειωματάριο ConvolutionalNetworks.*

<!--en:ec8621f8 kind=exercise-->
2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*
<!--tr:ec8621f8-->
2. [code] **Η συνέλιξη ως ταινιωτός πίνακας.** Ο πίνακας τμημάτων της
   :numref:`subsec_conv_matmul` ξεδιπλώνει την είσοδο. Η αντίθετη κατασκευή
   ξεδιπλώνει τον πυρήνα. Έστω $\mathbf{X}$ μια είσοδος $4 \times 4$, $\mathbf{K}$
   ένας πυρήνας $3 \times 3$ και $\textrm{vec}(\cdot)$ η επιπέδωση γραμμή προς
   γραμμή.
    1. Κατασκευάστε τον πίνακα $\mathbf{W}$ διαστάσεων $4 \times 16$ με
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       όπου $\mathbf{Y}$ είναι η ετεροσυσχέτιση του $\mathbf{X}$ με τον
       $\mathbf{K}$. Πόσα μη μηδενικά στοιχεία έχει κάθε γραμμή και πόσες
       διακριτές τιμές εμφανίζονται στον $\mathbf{W}$;
    1. Κατασκευάστε τον $\mathbf{W}$ σε κώδικα από τον `K` και επιβεβαιώστε ότι
       το γινόμενο αναπαράγει τη `corr2d(X, K)`.
    1. Για μια είσοδο $n \times n$ και έναν πυρήνα $k \times k$, μετρήστε τα
       στοιχεία του $\mathbf{W}$ αποθηκευμένου ως πυκνού πίνακα, τα μη μηδενικά
       στοιχεία του $\mathbf{W}$ και τα στοιχεία του πίνακα τμημάτων. Πώς
       αυξάνεται κάθε πλήθος με το $n$;
    1. Διατηρήστε το μοτίβο αραιότητας του $\mathbf{W}$, αλλά δώστε σε κάθε
       γραμμή τις δικές της $k^2$ τιμές. Το αποτέλεσμα είναι ένα τοπικά
       συνδεδεμένο επίπεδο: ο τανυστής τέταρτης τάξης $\mathsf{V}$ της
       :numref:`sec_why-conv` στη σχέση
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       περιορισμένος σε ένα παράθυρο $k \times k$ μετατοπίσεων $(a, b)$ πριν
       η ισομεταβλητότητα ως προς τη μετατόπιση δεσμεύσει τα βάρη να είναι
       κοινά. Για μια είσοδο $n \times n$, μετρήστε τις παραμέτρους αυτού του
       επιπέδου, ενός πλήρως συνδεδεμένου επιπέδου με τα ίδια σχήματα εισόδου
       και εξόδου, και της συνέλιξης.

    *Προσαρμοσμένο από το CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

<!--en:54bcee26 kind=exercise-->
3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.
<!--tr:54bcee26-->
3. [code] **Ενεργό υποδεκτικό πεδίο.** ● Το θεωρητικό υποδεκτικό πεδίο $r$
   αυτής της ενότητας οριοθετεί ποιες είσοδοι *μπορούν* να επηρεάσουν μια
   έξοδο· το ενεργό υποδεκτικό πεδίο μετρά πόσο την επηρεάζουν στην πράξη
   :cite:`Luo.Li.Urtasun.ea.2016`. Κατασκευάστε μια στοίβα $L$ συνελικτικών
   επιπέδων $3 \times 3$ ενός καναλιού χωρίς συμπλήρωση, χωρίς πόλωση και χωρίς
   μη γραμμικότητα πάνω σε μια είσοδο $32 \times 32$, χρησιμοποιώντας το
   ενσωματωμένο επίπεδο της :numref:`subsec_learning_kernel`. Δειγματοληπτήστε
   κάθε στοιχείο πυρήνα ανεξάρτητα από μια τυπική κανονική κατανομή.
    1. Για $L = 3$ και ένα μόνο δείγμα πυρήνων, οπισθοδιαδώστε μια μοναδιαία
       κλίση από μια μονάδα εξόδου κοντά στο κέντρο του τελευταίου επιπέδου
       προς την είσοδο και σχεδιάστε το μέτρο της κλίσης εισόδου. Συγκρίνετε
       τον φορέα της με το θεωρητικό υποδεκτικό πεδίο $7 \times 7$ και αναφέρετε
       πού είναι μεγαλύτερο το μέτρο.
    1. Υπολογίστε τη μέση τιμή του τετραγώνου της κλίσης εισόδου σε 500 δείγματα
       πυρήνων, για $L = 3$ και για $L = 10$. Αθροίστε κάθε μέσο χάρτη κατά
       τις γραμμές του και κανονικοποιήστε το προκύπτον προφίλ στηλών ώστε να
       αθροίζει στη μονάδα. Πώς μεταβάλλεται με το βάθος ο λόγος της τυπικής
       απόκλισης του προφίλ προς το θεωρητικό μήκος πλευράς $2L + 1$;
    1. Το :numref:`fig_conv_rf_paths` καταμετρά τις διαδρομές για $L = 3$ σε μία
       διάσταση. Δείξτε ότι για γενικό $L$ η αναμενόμενη τετραγωνική κλίση
       εισόδου, αθροισμένη κατά τις γραμμές, είναι ανάλογη της συνέλιξης $L$
       αντιγράφων του διανύσματος $(1, 1, 1)$. Εξαγάγετε πώς αυξάνεται με το
       $L$ η τυπική απόκλιση του κανονικοποιημένου προφίλ.

<!--en:6ab5b67c kind=exercise-->
4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6ab5b67c-->
4. [code] **Οπίσθιο πέρασμα μιας συνέλιξης.** ● Έστω $\mathbf{Y}$ η
   ετεροσυσχέτιση ενός καναλιού μιας εισόδου $\mathbf{X}$ με έναν πυρήνα
   $\mathbf{K}$, όπως υπολογίζεται από τη `corr2d`, και έστω $\ell$ μια βαθμωτή
   απώλεια που εξαρτάται από τα $\mathbf{X}$ και $\mathbf{K}$ μόνο μέσω του
   $\mathbf{Y}$.
    1. Εξαγάγετε τις $\partial \ell/\partial \mathbf{K}$ και
       $\partial \ell/\partial \mathbf{X}$ συναρτήσει της
       $\partial \ell/\partial \mathbf{Y}$, στο ύφος των εξισώσεων
       οπισθοδιάδοσης της :numref:`sec_backprop`.
    1. Υλοποιήστε την έκφρασή σας για την $\partial \ell/\partial \mathbf{K}$
       χωρίς αυτόματη παραγώγιση για την απώλεια
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, όπου $\mathbf{G}$ είναι ένας σταθερός
       τυχαίος πίνακας με το σχήμα του $\mathbf{Y}$. Συγκρίνετέ την με μια
       κεντραρισμένη κλίση πεπερασμένων διαφορών με μέγεθος βήματος $10^{-2}$
       σε μικρά τυχαία `X` και `K`, και αναφέρετε το μεγαλύτερο σχετικό σφάλμα.
    1. Ελέγξτε την έκφρασή σας για την $\partial \ell/\partial \mathbf{K}$
       σε σύγκριση με την κλίση του πυρήνα που υπολογίζει η αυτόματη παραγώγιση
       στο πείραμα της :numref:`subsec_learning_kernel`.

    *Προσαρμοσμένο από το έργο του Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Κεφάλαιο 6, και το Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    σημειωματάριο ConvolutionalNetworks.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Συζητήσεις](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Συζητήσεις](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Συζητήσεις](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Συζητήσεις](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- "locally connected layer" → «τοπικά συνδεδεμένο επίπεδο»
- "support" (of a function/gradient) → «φορέας»
- "blur kernel" / "sharpening kernel" → «πυρήνας θόλωσης» / «πυρήνας όξυνσης»
- "Discussions" → «Συζητήσεις»
- Exercise `[code]` tag and `●` difficulty marker → preserved unchanged.

=== NOTES ===
- ec8621f8: math tokens reordered for natural Greek noun-phrase order («πίνακα $\mathbf{W}$ διαστάσεων $4 \times 16$»).
