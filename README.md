Embedding Shepard’s interpolation into CNN models for unguided Depth Completion.

This is the official implementation of the paper Embedding Shepard’s interpolation into CNN models for unguided Depth Completion
<img width="825" height="488" alt="Screenshot 2025-08-02 at 00 49 56" src="https://github.com/user-attachments/assets/b5bf4b6b-3e0d-4ece-8cd8-1a8543b673a7" />
<img width="944" height="521" alt="Screenshot 2025-08-02 at 00 50 35" src="https://github.com/user-attachments/assets/4db31a4c-5c44-426b-840e-ad2ab35d783f" />





# Training
## Preprocessing
<pre>
# Split the dataset into training, validation and testing
# and store them in an HDF5 file format
</pre>
<pre>
# Check the hyperparameters in args.py and run
Python main.ply --mode train
</pre>

# Testing
<pre>
  python main.py --mode test
</pre>

<pre>
@inproceedings{10.1007/978-3-031-47546-7_23,
author = {Mengistu, Shambel Fente and Pistellato, Mara and Bergamasco, Filippo},
title = {Embedding Shepard’s Interpolation into&nbsp;CNN Models for&nbsp;Unguided Depth Completion},
year = {2023},
isbn = {978-3-031-47545-0},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-47546-7_23},
doi = {10.1007/978-3-031-47546-7_23},
pages = {336–350},
numpages = {15},
keywords = {Sparse convolution, CNN, Depth Completion, Inverse Distance Weighting, Shepard’s Interpolation},
location = {Rome, Italy}
}
</pre>
