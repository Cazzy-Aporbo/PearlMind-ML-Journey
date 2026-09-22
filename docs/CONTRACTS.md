# What goes in. What comes out.

| Path | Input | Output | Boundary / likely failure |
| :-- | :-- | :-- | :-- |
| `pearlmind train` | Default synthetic data or numeric CSV; target 0/1; optional group | Native JSON model, metadata, predictions CSV, metrics JSON, importance CSV | Rejects nonfinite features and invalid labels; random split unsuitable for time/group dependence |
| `pearlmind evaluate` | Saved model and labeled CSV with matching feature order | Observed accuracy, confusion matrix, optional group metrics | Feature order is part of the contract; never pass training data as an independent test |
| `pearlmind audit` | CSV columns actual, prediction, group | Descriptive binary group metrics | Missing denominators remain null; no legal or ethical certification |
| Local `/predict` | 1–1000 numeric rows with trained feature count | Predictions, probabilities, version; optional labeled audit | 422 for malformed input, 503 for no loaded model; no internet-facing auth/rate limit supplied |
| PyTorch CPU lab | 500 synthetic two-dimensional observations, seed, epochs | State dictionary, scaler statistics, loss plot, measured test score | One held-out split; not a production benchmark |
| Text lab | Local UTF-8 corpus, model dimensions, sequence length | Checkpoint, vocabulary, run manifest, validation loss | Unseen characters raise; small validation text cannot create a full window |
| Systems lab | Worker mean gradients + counts; bounded task name | Weighted update and finite state trace | Single process; no real distributed cluster or LLM agent |
| Dolly explorations | Licensed local Dolly JSONL for the chosen script | Dataset summaries and plots | Download/terms separate; historical checked-in plots are not current CI measurements |
| Delphi / health / quantum studies | Module-specific tensors, simulated values or external datasets | Experimental representations and comparisons | Not clinically validated; optional imports, scientific assumptions and real-data provenance need separate review |
| Original HTML guides | Browser | Visual reference material | Historical reference; does not establish executable coverage of every architecture |

File-level import and definition inventories are generated for the Pages source atlas. A syntax check is not a training run, and a shape test is not scientific validation. Check the displayed status before using a file as an example.
