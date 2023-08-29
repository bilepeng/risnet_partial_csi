# RISnet: A Scalable Approach for Reconfigurable Intelligent Surface Optimization with Partial CSI

This repository is the source code and data for the paper

B. Peng, K.-L. Besser, R. Raghunath, V. Jamali and E. A. Jorswieck, "RISnet: A Scalable Approach for Reconfigurable Intelligent Surface Optimization with Partial CSI", accepted by Globecom 2023.

Data available under https://drive.google.com/file/d/1cXh4ME7bmY7a7llOj4Np2qBakrI2eHwH/view?usp=sharing.
Put the unzipped files in `data/` and run the following commands to produce results in Figure 6.
```
python3 train.py --record True  --tsnr=1e12  --partialcsi True --trainingchannelpath data/channels_ris_rx_training.pt --testingchannelpath data/channels_ris_rx_testing.pt --name partial_0
python3 train.py --record True  --tsnr=1e12  --partialcsi False --trainingchannelpath data/channels_ris_rx_training.pt --testingchannelpath data/channels_ris_rx_testing.pt --name full_0
python3 train.py --record True  --tsnr=1e12  --partialcsi True --trainingchannelpath data/channels_ris_rx_training_5e-5.pt --testingchannelpath data/channels_ris_rx_testing_5e-5.pt --name partial_p
python3 train.py --record True  --tsnr=1e12  --partialcsi False --trainingchannelpath data/channels_ris_rx_training_5e-5.pt --testingchannelpath data/channels_ris_rx_testing_5e-5.pt --name full_p
python3 train.py --record True  --tsnr=1e12  --partialcsi True --trainingchannelpath data/channels_ris_rx_training_iid.pt --testingchannelpath data/channels_ris_rx_testing_iid.pt --name partial_iid
python3 train.py --record True  --tsnr=1e12  --partialcsi False --trainingchannelpath data/channels_ris_rx_training_iid.pt --testingchannelpath data/channels_ris_rx_testing_iid.pt --name full_iid
```
