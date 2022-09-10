# ATP: Dataset and Inputs Preparation

The atemporal probe (ATP) is a discriminative model for video-language analysis operating *on top of* (frozen) image-language encodings from the video-language dataset. In this document, we add some more details for the processing that comes pre-ATP.

## Datasets and Processing

We examine a number of different standard datasets in the paper for discriminative tasks (e.g. video QA). For consistency with prior work, we leverage standard pipelines from prior work and/or the dataset authors when applicable. Other pipelines may yield slightly differing results (e.g. differences in video decoding, etc.). Details can be found at the following links:

- **MSR-VTT\*, ActivityNet, DiDeMo**: see `scripts/` and `src/` [in this repo here](https://github.com/jayleicn/ClipBERT) for more.
- **NeXT-QA**: see [the official dataset repo](https://github.com/doc-doc/NExT-QA) for more.
- **VALUE-How2QA**: see [the official dataset repo](https://github.com/VALUE-Leaderboard/DataRelease) for more.

We thank the authors of the repos and the datasets above again. As mentioned in the main README, if you find any of the above excellent resources helpful in your work, please be sure to cite them / their corresponding papers as well.

## Feature Extraction

Our key results in the paper are mainly reported using [OpenAI's CLIP](https://github.com/openai/CLIP) as the frozen image-language encoder (ViT-B32). For feature processing, add the following dependencies:

```sh
# (inside your conda environment; see also https://github.com/openai/CLIP)
pip install ftfy regex tqdm

# to use CLIP as-is (no modifications)
pip install git+https://github.com/openai/CLIP.git

# to make modifications for feature extraction to fit your pipeline
git clone git@github.com:openai/CLIP.git
cd CLIP  # ... make your modifications
pip install -e .  # inside the CLIP/ folder; updates clip API
```

You can run the CLIP model as part of the feature processing pipelines mentioned in the above subsection, with some small modifications to dump out the feature embeddings for CLIP's vision and language encoders (i.e. the embeddings that are used to compute similarity in CLIP).

**Additional Notes:**
- For some datasets (e.g. VALUE-How2QA), we use the officially released features when applicable for baseline comparison (see links above).
- CLIP's encoder has a bound on the size of the input language encodings, a limitation which can impact performance of certain tasks/benchmarks (e.g. ActivityNet paragraph retrieval). In this work, we truncate the input size to fit, but other techniques may yield better results.

**Extensions:** We encourage the community to use whatever newer image-language models / processing pipelines emerge to analyze video-language datasets (including those beyond the benchmarks considered here). Ultimately, the goal of ATP is to be a tool to help better characterize the boundary of *capabilities* between what is addressable from a single (well-selected) frame and what really does require multi-frame (temporal/causal/etc) understanding in videos. Better underlying representations for image-language can only help to improve the characterization of this bound.


## Additional Notes

**Limitations and Broader Impact:** Please refer to the supplement in the project page for a discussion of limitations and broader impacts, as well as the datasheets and model cards in the linked repos above when applicable.

**NeXT-QA (ATP-Hard):** In the paper, we describe a subset of the NeXT-QA validation set identified using ATP for more challenging causal and temporal questions. If you are interested in running experiments on the ATP-Hard (Causal/Temporal) split, please reach out to `shyamal@cs.stanford.edu`.
