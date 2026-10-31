<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange?style=flat-square&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Dataset-AFHQ%20(Kaggle)-informational?style=flat-square&logo=kaggle" />
  <img src="https://img.shields.io/github/v/release/ATemova/face-recog-cat-dog?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Archived-lightgrey?style=flat-square" />
</p>

# 🐱🐶 Animal Face Classification: Cat / Dog / Wild

A convolutional neural network trained from scratch to classify animal faces from the [Animal Faces-HQ (AFHQ)](https://www.kaggle.com/datasets/andrewmvd/animal-faces) dataset into three classes: **cat**, **dog** and **wild**.

Built as the final project for a Data Mining & Machine Learning course.

## Dataset

| Split | Source folder | Used for |
|---|---|---|
| Train | `afhq/train` (80%, stratified) | fitting the model |
| Validation | `afhq/train` (20%, stratified) | early stopping and learning-rate schedule |
| Test | `afhq/val` | final evaluation only, never seen during training |

Roughly 5,000 training images per class at 512×512, resized to 150×150 for the model.

## Model

Three convolutional blocks (32 → 64 → 128 filters, each with batch normalization, max pooling and dropout), followed by a 512-unit dense layer and a softmax output.

- Optimizer: Adam, categorical cross-entropy
- Augmentation (training only): small rotations, shifts, zoom, horizontal flips
- Early stopping on validation loss with best-weight restoration
- Learning rate halved when validation loss plateaus
- Fixed random seed for reproducibility

## Results

On the held-out test set (`afhq/val`, 1,500 images):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Cat | 0.976 | 0.988 | 0.982 |
| Dog | 0.976 | 0.982 | 0.979 |
| Wild | 0.984 | 0.966 | 0.975 |
| **Overall accuracy** | | | **0.979** |

Test-set accuracy, the per-class classification report, the confusion matrix, training curves and sample (mis)classifications are all in the [notebook](data-mining-machine-learning-project.ipynb). The trained model (`afhq_cnn.keras`) is attached to the [latest release](https://github.com/ATemova/face-recog-cat-dog/releases/latest).

## Running it

### Locally (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
curl -L -o data/afhq.zip "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"
unzip -q data/afhq.zip -d data
rm data/afhq.zip
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 data-mining-machine-learning-project.ipynb
```

The dataset is read from `data/afhq` and the model is written to `outputs/afhq_cnn.keras`. Both paths can be overridden with the `AFHQ_DIR` and `OUT_DIR` environment variables.

### On Kaggle

Import the notebook, add the **Animal Faces** dataset (`andrewmvd/animal-faces`) as input, enable a GPU and run all cells. Paths are detected automatically.

Loading the released model:

```python
import tensorflow as tf
model = tf.keras.models.load_model("afhq_cnn.keras")
```

Inputs must be 150×150 RGB scaled to `[0, 1]`; output order is `["cat", "dog", "wild"]`.

## Project status

This project is complete and the repository is archived. See [CHANGELOG.md](CHANGELOG.md) for the final changes.

## License

[MIT](LICENSE)
