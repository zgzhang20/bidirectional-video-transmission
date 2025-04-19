Bi-Directional Motion Enhanced Semantic Communication for Wireless Video Transmission

## abstract
With the increasing proliferation of Ultra-High-Definition (UHD) videos, the demand for efficient video transmission schemes to alleviate network congestion is growing. In this paper, we proposed a bi-directional motion enhanced semantic communication system for efficient and robust video transmission. In particular, we introduce a bi-directional motion estimation module to capture inter-frame differences caused by camera movements, where the obtained forward and backward motion vectors are combined with the residual information to generate motion-compensated frames. We also introduce a predicted feature module to discard semantically redundant features, prioritizing crucial semantic-related content. Leveraging information from previously reconstructed frames, the frame prediction module refines predicted frames with the assistance of the motion compensation module. To enhance the system's robustness to channel noise, we propose a noise attention module that assigns varying importance weights to the extracted features under different channel conditions. Experimental results show that our proposed method outperforms existing deep learning (DL)-based approaches in terms of transmission efficiency, achieving about 33.3\% reduction in the number of transmitted symbols while improving the peak signal-to-noise ratio (PSNR) and multi-scale structural similarity index measure (MS-SSIM) performance by an average of 0.96dB and 0.0024 over an additive white Gaussian noise channel for different schemes. When employing the same compression ratio, our method achieves an average gain of 0.637dB in PSNR and 0.0038 in MS-SSIM over the slow Rayleigh fading channel.

## Requirements
- python 3.9
- Pytorch 1.9
- At least 1x24GB NVIDIA GPU

## Datasets
Vimeo-90k
BVI-DVC 

















