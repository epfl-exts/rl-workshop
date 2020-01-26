
Climbing the Ladder: Reinforcement Learning in a Competitive Setting
---

Talk @ Applied Machine Learning days 2020

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pacm/rl-workshop) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pacm/rl-workshop/master)

Gitter channel: https://gitter.im/rl-workshop/community

**Part 1 – Introduction to RL / environment**

* 09:00 Intro to RL [slides](https://docs.google.com/presentation/d/10yaaF3BWMqUb-xMZZX3tJLASTG9Hxgdst0qKPVhd2DI/edit?usp=sharing)
* 10:00 Intro to the `DroneDelivery` environment - [notebook](https://colab.research.google.com/github/pacm/rl-workshop/blob/abdce93c5d9bc401b42251da36d3692fda00521d/01%20Intro%20to%20environment.ipynb) / [slides](https://docs.google.com/presentation/d/10uOOdxPDKaavjwHTqF1WhBjKfOeu6-cTnJ_Yrki9UvE/edit?usp=sharing)
* *10:30 Coffee break*
* 11:00 Intro to Q-learning and DQN - [notebook](https://colab.research.google.com/github/pacm/rl-workshop/blob/abdce93c5d9bc401b42251da36d3692fda00521d/02%20Intro%20to%20Q-learning%20and%20DQN.ipynb)
* *12:30 Lunch break*

**Part 2 – Modern RL methods**

* 13:30 Prioritized Experience Replay [notebook](https://colab.research.google.com/drive/1JR6Q3A9X4KUznZkgwQhLvX4J7sizzPes) [slides](https://docs.google.com/presentation/d/1DAxIhMu695aISunyIqut9lUuwJMvqmGpLPPqF0R5ieE/edit?usp=sharing)
* *15:00 Coffee break*
* 15:30 Curiosity [notebook](https://colab.research.google.com/drive/1WuLFg9jkQd1idmPZAsnDtp3c8Tibcmqx) / [slides](https://drive.google.com/file/d/1DWrQ4cyIGQztfK1QGNtLpjpzUfYq-xrA/view?usp=sharing)
* *17:00 End of workshop*


**Troubleshooting**

When running the notebook on your machine in Jupyter Lab, you will need to activate the `ipywidgets` plugin by running this command in the Conda environment

```bash
conda env create -f environment.yml
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

**Licence**

* [16ShipCollection](https://opengameart.org/content/1616-ship-collection) by master484 under Public Domain / CC0
* [Inconsolata-Bold](https://fonts.google.com/specimen/Inconsolata) by Raph Levien under [Open Font License](https://scripts.sil.org/cms/scripts/page.php?site_id=nrsi&id=OFL_web)
