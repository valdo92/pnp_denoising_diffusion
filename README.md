# Multivariate Time Series Forecasting with Graph Neural Networks

**Authors:** [Valentin Exbrayat](https://github.com/valdo92), [Hugo Pavy](https://github.com/hpavy)  
**Date:** March 2026

This repository contains an implementation of the **Denoising Diffusion Models for Plug-and-Play Image Restoration** architecture, originally proposed by Zhu et al. (2023). We focused on reimplementing the paper for inpainting task. Once it was done, we produced experiments on some parameters and we tried to change the Plug and Play algorithm from HQS to PGD.

```bibtex
@misc{zhu2023denoisingdiffusionmodelsplugandplay,
      title={Denoising Diffusion Models for Plug-and-Play Image Restoration}, 
      author={Yuanzhi Zhu and Kai Zhang and Jingyun Liang and Jiezhang Cao and Bihan Wen and Radu Timofte and Luc Van Gool},
      year={2023},
      eprint={2305.08995},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2305.08995}, 
}
```



### 🔍 Overview

The goal of this work is to study the use of diffusion model for image inpainting in a Plug and Play (Pnp) framework. We reimplemented the architecture, following the code of the original repo: [link of the repo](https://github.com/yuanzhi-zhu/DiffPIR)

**Key Contributions:**
* **Easier implementation:** focusing only on inpainting
* **Stressing the model**: trying to increase the size of the mask to study how the model handles it.
* **Other PnP method**: Instead of HQS we tried to use PGD method with our model.
* **Study of the parameter $\sigma$**: Changing how the $\sigma$ parameter is computed.

You can find the report of our work in the file [report.pdf](report.pdf).

---

### 💻 Installation

This project manages dependencies using **[Poetry](https://python-poetry.org)**.

**Install dependencies and environment:**

```
pip install poetry
poetry install
```

---

### 🚀 Minimal run snippet

1) Configure `config.yaml` choosing the data and the hyperparameters.
* You need to put your data in the folder `data/`. 
* You need to write which data in this folder you want to use in a txt file. Then write the path of this file in config.

Example of the `config.yaml` file in order to write the path of the files: 

```
image_list_file: data/ffhq_100_val.txt
```

1) Run the main file (example):

```bash
poetry run python main.py
```


TO DO:

- [x] pouvoir save le config
- [ ] faire l'expérience sur la valeur de lambda
- [ ] faire l'expérience sur la diffusion
- [ ] faire varier omega avec les meilleurs valeurs
- [ ] mettre ça dans le rapport