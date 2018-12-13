
12th December 2018
==================

Running some experiments here as part of a project involving low-rank
approximations in deep learning. Would like the experiments to run quickly.
Tested the code on a Titan X (Pascal) GPU. `dawn.py` ran in 212 seconds and
reached 94% accuracy.

Running the same script on our V100 GPU, it took 223 seconds to run the
same 24 epochs. Which is strange, because that GPU should be just as fast
as the GPU on the amazon instance used in the post.
