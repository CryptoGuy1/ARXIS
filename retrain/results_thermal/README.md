# Thermal classifier, both protocols

YOLOv8n-cls retrained on the MultimodalGasData thermal frames under two protocols, with the settings of Supplementary Section S1 (Ultralytics 8.4.14, 50 epochs, image size 224, batch 32, Tesla T4). Each run reports its final-epoch weights on its own held-out frames.

runs/ holds one JSON per run: top-1, per-class accuracy, the confusion matrix, the held-out block positions, the optimizer as recorded at training start, and the hazard-miss count (Smoke or Mixture frames predicted as NoGas or Perfume). The block-wise seed-42 run was written before the hazard field existed, so its count is computed from its confusion matrix by summarize_thermal.py.

split_tables/ holds one row per frame with its Serial Number, image name, class and split (train, embargo or test). These are the auditable assignments behind each run.

curves/ holds the per-epoch training logs for all eight runs. The block-wise seed-1337 log was recovered later from the Kaggle output and checked against its JSON: 50 epochs, and a final-epoch top-1 of 0.79219, which matches the JSON.

checkpoints.json records, for each run, the SHA-256 hash and size of its final-epoch checkpoint, the top-1 the checkpoint stores from training, and the training arguments it recorded. The checkpoints themselves are not in the repository. Two checkpoint files that were uploaded twice were byte-identical.

scripts/ holds the two Kaggle drivers as they were run. thermal_split_script.py runs either protocol (set SPLIT to blockwise or randomized); it was used for the randomized runs and the block-wise runs with seeds 1337, 7, 2024 and 99. thermal_blockwise_seed42_script.py is the block-wise driver used for seed 42, which predates the SPLIT switch and the hazard field. Both read the dataset from /kaggle/input, and both need the thermal frames and Gas_Sensors_Measurements.csv.

summary.json is written by `make thermal` (retrain/summarize_thermal.py). That command checks every run against its confusion matrix, split table, and checkpoint record, then writes the means, standard deviations and ranges that the manuscript and Supplementary Section S1 quote.
