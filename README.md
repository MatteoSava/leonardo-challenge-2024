# Leonardo Challenge 2024

Multimodal learning challenge notebook from the Sapienza Training Camp organized with Leonardo Company S.p.A.

## Details

- **Event:** Sapienza Training Camp — Leonardo Company S.p.A.
- **Date:** September 4–6, 2024
- **Topic:** Multimodal Learning from Language and Vision Processing Neural Networks
- **Team:** CodeBusters
- **Result:** 2nd place

## Challenge

The task was to build a multimodal classifier that predicts a hidden context from multiple input modalities:

- an image of the object
- the object name
- a text description
- a 4-class target label

Given this multimodal dataset, the goal was to predict one of four classes using text, images and object metadata. The main risk was the model relying too heavily on the explicit object identity instead of capturing the actual hidden-context signal.

The notebook expects this local data layout:

    dataset/
      train.csv
      test.csv
      images/
        train/
        test/

The dataset is not included in this repository.

## Solution

The solution is a multimodal classifier built in PyTorch:

- `ViTModel` (`google/vit-base-patch16-224-in21k`) for visual feature extraction
- `GPT2Model` + `GPT2Tokenizer` to encode object name + description
- Normalization of visual and text representations
- Concatenation of the two feature vectors
- Small fully-connected classifier head for 4-class prediction

Training uses `AdamW` with a 90/10 train/validation split, saving the best model on validation accuracy. The notebook produces a `submission.csv` for the competition leaderboard.

An interesting aspect of the solution was reasoning about object redundancy: since the object was already explicitly represented in the input, it made sense to try compensating for that information in the text and visual embeddings, pushing the model toward the actual target — the hidden context.

## Usage

Install the environment with pinned dependencies:

    uv sync --dev

Then open `Leonardo Challenge 2024.ipynb` in your notebook environment using the project kernel.

## Notes

This repo is intentionally minimal: it contains the challenge artifact, pinned dependencies and the context needed to understand the approach.
