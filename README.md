# RISnet: A Scalable Approach for Reconfigurable Intelligent Surface Optimization with Partial CSI

![GitHub](https://img.shields.io/github/license/bilepeng/risnet_partial_csi)
[![DOI](https://img.shields.io/badge/doi-10.1109/GLOBECOM54140.2023.10437049-informational)](https://doi.org/10.1109/GLOBECOM54140.2023.10437049)
[![arXiv](https://img.shields.io/badge/arXiv-2305.00667-informational)](https://arxiv.org/abs/2305.00667)

This repository is the source code and data for the paper

This repository contains the source code for the paper "RISnet: A Scalable
Approach for Reconfigurable Intelligent Surface Optimization with Partial CSI"
(B. Peng, K.-L. Besser, R. Raghunath, V. Jamali and E. A. Jorswieck, 2023 IEEE
Global Communications Conference, pp. 4810-4816, Dec. 2023.
[doi:10.1109/GLOBECOM54140.2023.10437049](https://doi.org/10.1109/GLOBECOM54140.2023.10437049),
[arXiv:2305.00667](https://arxiv.org/abs/2305.00667)).

The data is available under https://drive.google.com/file/d/1cXh4ME7bmY7a7llOj4Np2qBakrI2eHwH/view?usp=sharing.


## Usage
Put the unzipped files in `data/` and run the following commands to produce results in Figure 6.

```bash
python3 train.py --record True  --tsnr=1e12  --partialcsi True --trainingchannelpath data/channels_ris_rx_training.pt --testingchannelpath data/channels_ris_rx_testing.pt --name partial_0
python3 train.py --record True  --tsnr=1e12  --partialcsi False --trainingchannelpath data/channels_ris_rx_training.pt --testingchannelpath data/channels_ris_rx_testing.pt --name full_0
python3 train.py --record True  --tsnr=1e12  --partialcsi True --trainingchannelpath data/channels_ris_rx_training_5e-5.pt --testingchannelpath data/channels_ris_rx_testing_5e-5.pt --name partial_p
python3 train.py --record True  --tsnr=1e12  --partialcsi False --trainingchannelpath data/channels_ris_rx_training_5e-5.pt --testingchannelpath data/channels_ris_rx_testing_5e-5.pt --name full_p
python3 train.py --record True  --tsnr=1e12  --partialcsi True --trainingchannelpath data/channels_ris_rx_training_iid.pt --testingchannelpath data/channels_ris_rx_testing_iid.pt --name partial_iid
python3 train.py --record True  --tsnr=1e12  --partialcsi False --trainingchannelpath data/channels_ris_rx_training_iid.pt --testingchannelpath data/channels_ris_rx_testing_iid.pt --name full_iid
```


## Acknowledgements
This research was supported in part by the Federal Ministry of Education and
Research Germany (BMBF) as part of the 6G Research and Innovation Cluster
6G-RIC under Grant 16KISK031, by the German Research Foundation (DFG) under
grant BE 8098/1-1, in part by the DFG as part of project C8 within the
Collaborative Research Center (CRC) 1053-MAKI, and in part by the LOEWE
initiative (Hesse, Germany) within the emergenCITY center.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```bibtex
@inproceedings{Peng2023risnet,
  author = {Peng, Bile and Besser, Karl-Ludwig and Raghunath, Ramprasad and Jamali, Vahid and Jorswieck, Eduard A.},
  title = {{RISnet}: A Scalable Approach for Reconfigurable Intelligent Surface Optimization with Partial {CSI}},
  booktitle = {GLOBECOM 2023 -- 2023 IEEE Global Communications Conference},
  year = {2023},
  month = {12},
  pages = {4810--4816},
  publisher = {IEEE},
  venue = {Kuala Lumpur, Malaysia},
  archiveprefix = {arXiv},
  eprint = {2305.00667},
  primaryclass = {eess.SP},
  doi = {10.1109/GLOBECOM54140.2023.10437049},
}
```
