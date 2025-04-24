<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/hiaac-finance/xai_aggregation">
    <img src="resources/logo-black.png" alt="Logo" height="150">
  </a>

  <p align="center">
    A python package for a rank based, multi-criteria aggregation method for explainable AI models.
    <div align="center">
      <img src="https://img.shields.io/github/repo-size/hiaac-finance/xai_aggregation" alt="Repo Size">
      <img src="https://img.shields.io/github/stars/hiaac-finance/xai_aggregation" alt="Stars">
      <img src="https://img.shields.io/github/license/hiaac-finance/xai_aggregation" alt="License">
      <img src="https://img.shields.io/readthedocs/xai-agg" alt="Documentation Status">
    </div>
    <br />
    <br />
    <a href="https://xai-agg.readthedocs.io/en/latest/">Read the Docs</a>
    ·
    <a href="">Paper</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About

<div align="center">
    <img src="resources/diagram-black.png" alt="Diagram" style="border-radius: 15px;">
</div>

Explainability is crucial for improving the transparency of black-box machine learning models. With the advancement of explanation methods such as LIME and SHAP, various XAI performance metrics have been developed to evaluate the quality of explanations. However, different explainers can provide contrasting explanations for the same prediction, introducing trade-offs across conflicting quality metrics. Although available aggregation approaches improve robustness,
reducing explanations’ variability, very limited research employed a multi-criteria decision-making approach. To address this gap, <a href="">this package's paper</a> introduces a multi-criteria rank-based weighted aggregation method that balances multiple quality metrics simultaneously to produce an ensemble of explanation models.<!--  -->


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation
Run `pip install xai-agg` to install the package.