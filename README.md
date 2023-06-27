<h1 align="center">
  <a href="https://github.com/hellocym/dcgan">
    <!-- Please provide path to your logo here -->
    <img src="docs/images/dcgan.gif" alt="Logo" width="100" height="100">
  </a>
</h1>

<div align="center">
  DCGAN
  <br />
  <a href="#about"><strong>Explore the docs »</strong></a>
  <br />
  <br />
  <a href="https://github.com/hellocym/dcgan/issues/new?assignees=&labels=bug&template=01_BUG_REPORT.md&title=bug%3A+">Report a Bug</a>
  ·
  <a href="https://github.com/hellocym/dcgan/issues/new?assignees=&labels=enhancement&template=02_FEATURE_REQUEST.md&title=feat%3A+">Request a Feature</a>
  .
  <a href="https://github.com/hellocym/dcgan/issues/new?assignees=&labels=question&template=04_SUPPORT_QUESTION.md&title=support%3A+">Ask a Question</a>
</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

</details>

---

## About

> This repo is the unofficial pytorch implementation of MNIST and CelebA image generating using [Deep Convolutional Generative Adversarial Network (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf)

## Prerequisites
> see [requirements.txt](./requirements.txt)

## Usage
1. Clone the repo
2. Install the requirements
3. If training on CelebA dataset, add kaggle.json to the root directory by following the instructions [here](https://github.com/Kaggle/kaggle-api#api-credentials)
3. Run the following command to train the model

```bash
python ./train.py --dataset [MNIST/CelebA]
```

## License

This project is licensed under the **MIT license**.

See [LICENSE](LICENSE) for more information.

## Acknowledgements
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/