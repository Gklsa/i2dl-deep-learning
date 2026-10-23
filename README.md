# Introduction to Deep Learning — I2DL

Course exercises from **Introduction to Deep Learning (IN2346), Technical University of Munich (summer semester 2026)**, with later study extensions for exercises 11–12. The notebooks combine NumPy implementations, PyTorch vision models and Transformer components.

## Structure

| Folder | Topic and entry notebooks |
|---|---|
| `exercise_01/` | Introduction: `1_introduction.ipynb` |
| `exercise_03/` | Image datasets, transforms and mini-batch loading |
| `exercise_04/` | Logistic classifier, loss and gradient descent |
| `exercise_05/` | Neural-network layers, backpropagation and optimizers |
| `exercise_06/` | CIFAR-10 classification and hyperparameter search |
| `exercise_08/` | Autoencoders and transfer learning |
| `exercise_09/` | Facial-keypoint regression |
| `exercise_10/exercise_10/` | Semantic segmentation: `1_segmentation_nn.ipynb` |
| `exercise_11/` | Tokenization, embeddings, attention and positional encoding |
| `exercise_12/` | Dataset/collator and encoder–decoder Transformer notebooks |

Implementation exercises contain notebooks and an `exercise_code/` package with implementations and course tests. Exercise 07 is not included in this copy.

## Setup

Use **Python 3.10 or 3.11**. From the repository root, create an environment and install the packages used by the notebooks and helper modules. In Windows PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install "numpy==1.26.4" "torch==2.2.2" "torchvision==0.17.2" matplotlib pandas seaborn pillow tqdm torchinfo packaging tensorboard jupyter
python -m pip install "transformers==4.44.2" "tokenizers==0.19.1"
python -m jupyter notebook
```

Exercise 08 also contains a notebook cell that installs `pytorch-lightning`. Use the environment above as the notebook kernel. On other systems, activate the environment using the command for your shell.

## Run notebooks and tests

1. Open a notebook from its exercise folder, so imports of its own `exercise_code/` package and relative data paths resolve correctly.
2. Run the setup/import cells, then follow the exercise in order. Follow the notebook's dataset/download instructions before running data-dependent tests or training.
3. Run the provided test cells to check the corresponding implementation. These call the course test functions; the tests are not presented as a repository-wide `pytest` suite.
4. Restart the kernel when switching exercises: different folders reuse the name `exercise_code`.

For exercise 11, use `1_tokenizer.ipynb` through `4_positional_encoding.ipynb`. For exercise 12, start with `1_dataset_and_collator.ipynb`, then `2_transformer.ipynb`. Its dataset tests use the included course sample at `datasets/transformerDatasets/dummyDatasets/ds_dummy`.

Notebook outputs are cleared. The original course reference tensors for the exercise 11 and 12 tests are included under `exercise_code/tests/tensors/`, together with the small dummy dataset used by the dataset tests. Other datasets, your trained checkpoints and pretrained demonstration models are not bundled. Training results must be generated in your own environment; fitting the small Transformer training dataset alone does not establish performance on unseen sentences.

## Course attribution

The I2DL course team provides the starter notebooks, code, tests and datasets. This repository records coursework implementations and the later Transformer study extensions; the original course materials retain their attribution. No additional license is applied to the course materials.
