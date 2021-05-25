# Models

The folder contains models from the active learning simulations.  These models are fine-tuned by a subset of data that is sampled from an active learning strategy.  Each model is located under a subdirectory {seed}/{task}/{strategy}_{size} to indicate the initialization seed, the downstream task, the strategy used to sample data, and the amount of data sampled. 

Model serialization follows the practices of [Huggingface Transformers v2.8.0](https://huggingface.co/transformers/v2.8.0/serialization.html#serialization-best-practices). 
