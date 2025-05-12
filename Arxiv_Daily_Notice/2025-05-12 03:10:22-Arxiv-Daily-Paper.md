# Showing new listings for Monday, 12 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 4papers 
#### Unsupervised Blind Speech Separation with a Diffusion Prior
 - **Authors:** Zhongweiyang Xu, Xulin Fan, Zhong-Qiu Wang, Xilin Jiang, Romit Roy Choudhury
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2505.05657

 - **Pdf link:** https://arxiv.org/pdf/2505.05657

 - **Abstract**
 Blind Speech Separation (BSS) aims to separate multiple speech sources from audio mixtures recorded by a microphone array. The problem is challenging because it is a blind inverse problem, i.e., the microphone array geometry, the room impulse response (RIR), and the speech sources, are all unknown. We propose ArrayDPS to solve the BSS problem in an unsupervised, array-agnostic, and generative manner. The core idea builds on diffusion posterior sampling (DPS), but unlike DPS where the likelihood is tractable, ArrayDPS must approximate the likelihood by formulating a separate optimization problem. The solution to the optimization approximates room acoustics and the relative transfer functions between microphones. These approximations, along with the diffusion priors, iterate through the ArrayDPS sampling process and ultimately yield separated voice sources. We only need a simple single-speaker speech diffusion model as a prior along with the mixtures recorded at the microphones; no microphone array information is necessary. Evaluation results show that ArrayDPS outperforms all baseline unsupervised methods while being comparable to supervised methods in terms of SDR. Audio demos are provided at: this https URL.
#### Toward a Sparse and Interpretable Audio Codec
 - **Authors:** John Vinyard
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.05654

 - **Pdf link:** https://arxiv.org/pdf/2505.05654

 - **Abstract**
 Most widely-used modern audio codecs, such as Ogg Vorbis and MP3, as well as more recent "neural" codecs like Meta's Encodec or the Descript Audio Codec are based on block-coding; audio is divided into overlapping, fixed-size "frames" which are then compressed. While they often yield excellent reproductions and can be used for downstream tasks such as text-to-audio, they do not produce an intuitive, directly-interpretable representation. In this work, we introduce a proof-of-concept audio encoder that represents audio as a sparse set of events and their times-of-occurrence. Rudimentary physics-based assumptions are used to model attack and the physical resonance of both the instrument being played and the room in which a performance occurs, hopefully encouraging a sparse, parsimonious, and easy-to-interpret representation.
#### Fast Differentiable Modal Simulation of Non-linear Strings, Membranes, and Plates
 - **Authors:** Rodrigo Diaz, Mark Sandler
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Computational Physics (physics.comp-ph)
 - **Arxiv link:** https://arxiv.org/abs/2505.05940

 - **Pdf link:** https://arxiv.org/pdf/2505.05940

 - **Abstract**
 Modal methods for simulating vibrations of strings, membranes, and plates are widely used in acoustics and physically informed audio synthesis. However, traditional implementations, particularly for non-linear models like the von Kármán plate, are computationally demanding and lack differentiability, limiting inverse modelling and real-time applications. We introduce a fast, differentiable, GPU-accelerated modal framework built with the JAX library, providing efficient simulations and enabling gradient-based inverse modelling. Benchmarks show that our approach significantly outperforms CPU and GPU-based implementations, particularly for simulations with many modes. Inverse modelling experiments demonstrate that our approach can recover physical parameters, including tension, stiffness, and geometry, from both synthetic and experimental data. Although fitting physical parameters is more sensitive to initialisation compared to other methods, it provides greater interpretability and more compact parameterisation. The code is released as open source to support future research and applications in differentiable physical modelling and sound synthesis.
#### Learning Music Audio Representations With Limited Data
 - **Authors:** Christos Plachouras, Emmanouil Benetos, Johan Pauwels
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.06042

 - **Pdf link:** https://arxiv.org/pdf/2505.06042

 - **Abstract**
 Large deep-learning models for music, including those focused on learning general-purpose music audio representations, are often assumed to require substantial training data to achieve high performance. If true, this would pose challenges in scenarios where audio data or annotations are scarce, such as for underrepresented music traditions, non-popular genres, and personalized music creation and listening. Understanding how these models behave in limited-data scenarios could be crucial for developing techniques to tackle them. In this work, we investigate the behavior of several music audio representation models under limited-data learning regimes. We consider music models with various architectures, training paradigms, and input durations, and train them on data collections ranging from 5 to 8,000 minutes long. We evaluate the learned representations on various music information retrieval tasks and analyze their robustness to noise. We show that, under certain conditions, representations from limited-data and even random models perform comparably to ones from large-dataset models, though handcrafted features outperform all learned representations in some tasks.


by Zyzzyva0381 (Windy). 


2025-05-12
