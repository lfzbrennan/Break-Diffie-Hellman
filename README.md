# Breaking Diffie-Hellman With transformers

The Diffie-Hellman algorithm is a public encryption scheme designed for two parties
to agree on a shared secret key through public communication. The algorithm relies
on the security of the discrete log problem, ie, if an exponent g^x Mod N is known,
where g is a fixed constant (3 in many cases) and N is a large random prime,
it is infeasible to recover x. However, if there exists an algorithm to recover x,
then an eavesdropper can recover the shared secret key by observing the public
communication between these two parties.

The recent rise in an neural network architecture called transformers has revolutionized
the field of natural language processing, allowing researchers to create models that
can analyze language, and even create new text, with a similar level to humans.
Transformers have also been used in other fields, such as image processing, for various
tasks such as image generation.

This project attempts, and succeeds, as using transformer architectures, in particular Albert,
to break the security of the discrete log problem, and in turn, Diffie-Hellman. The model
is trained by taking the exponent g^x and modular N, and attempting to recover x, all
modeled as bit-strings. The model was first trained on a low number of bits, and
iteratively transferred to a larger number of bits, breaking Diffie-Hellman for
8, 16, 32, 64, 128, 256, 512, 1024, and 2048 length keys.

# Pretrained Models
Models pretrained using g=3, iteratively trained on larger and larger key lengths.
Scale used was 4.

Bits | Success Rate | URL
--- | --- | ---
32 | ~100% | https://drive.google.com/file/d/1zd-ipJsimsuRC0aeRUO8f-4sZlJV__43/view?usp=sharing
64 | ~100% | https://drive.google.com/file/d/1TYFmLkvhxj-Jykm9HB3E_JNQxOUEVlOL/view?usp=sharing
128 | ~100% | https://drive.google.com/file/d/1qvSvWhZgH-_UzvRbhXgOyqALfwqk2Sbg/view?usp=sharing
256 | ~70% | https://drive.google.com/file/d/1fmnXIZk_kpPjzjAeAaVNAXp9TNFnNpHH/view?usp=sharing


# Training
`python3 train.py --parallel --bits 512 output_dir --outputs/512_bits`

# Evaluation
`python3 train.py --eval --eval_path outputs/512_bits/final/model.pt`
