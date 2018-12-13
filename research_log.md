
12th December 2018
==================

Running some experiments here as part of a project involving low-rank
approximations in deep learning. Would like the experiments to run quickly.
Tested the code on a Titan X (Pascal) GPU. `dawn.py` ran in 212 seconds and
reached 94% accuracy.

Running the same script on our V100 GPU, it took 223 seconds to run the
same 24 epochs. Which is strange, because that GPU should be just as fast
as the GPU on the amazon instance used in the post.

Switched the full convolutions in `conv_bn` for separable convolutions and
the network now converges to 92% accuracy in 171 seconds.

Switched in a low-rank approximation using a generic bottleneck, and it
only converged to 36% in 163 seconds.

Using the same low-rank approximation, and changing the weight decay to be
scaled by the compression factor, got 21.7% accuracy in 209 seconds.

Oh no, I must have had the rank set to something different. Running it
again with the old weight decay settings and it reaches 21.3% in 209
seconds.

Converged to 81.2% with a compression scaling factor of 2, which sets the
rank to be integer division 2 of the number of input channels. This is
barely a compression of the separable network, so this might not be the
best way to do this.

At a compression ratio of around 20%, with the scaling factor set to 10, it
converges to about 78.3%. With just the original weight decay it converges
to 77.8%.

I suppose the problem is that this network doesn't really have time to
worry about underfitting in the short training schedule. In both
experiments, the train loss is *above* the test loss. What if we increase
the training schedule by 4 times?


