# Showing new listings for Monday, 5 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 5papers 
#### Physics-Informed Neural Network-Driven Sparse Field Discretization Method for Near-Field Acoustic Holography
 - **Authors:** Xinmeng Luan, Mirco Pezzoli, Fabio Antonacci, Augusto Sarti
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2505.00897

 - **Pdf link:** https://arxiv.org/pdf/2505.00897

 - **Abstract**
 We propose the Physics-Informed Neural Network-driven Sparse Field Discretization method (PINN-SFD), a novel self-supervised, physics-informed deep learning approach for addressing the Near-Field Acoustic Holography (NAH) problem. Unlike existing deep learning methods for NAH, which are predominantly supervised by large datasets, our approach does not require a training phase and it is physics-informed. The wave propagation field is discretized into sparse regions, a process referred to as field discretization, which includes a series of set of source planes, to address the inverse problem. Our method employs the discretized Kirchhoff-Helmholtz integral as the wave propagation model. By incorporating virtual planes, additional constraints are enforced near the actual sound source, improving the reconstruction process. Optimization is carried out using Physics-Informed Neural Networks (PINNs), where physics-based constraints are integrated into the loss functions to account for both direct (from equivalent source plane to hologram plane) and additional (from virtual planes to hologram plane) wave propagation paths. Additionally, sparsity is enforced on the velocity of the equivalent sources. Our comprehensive validation across various rectangular and violin top plates, covering a wide range of vibrational modes, demonstrates that PINN-SFD consistently outperforms the conventional Compressive-Equivalent Source Method (C-ESM), particularly in terms of reconstruction accuracy for complex vibrational patterns. Significantly, this method demonstrates reduced sensitivity to regularization parameters compared to C-ESM.
#### How much to Dereverberate? Low-Latency Single-Channel Speech Enhancement in Distant Microphone Scenarios
 - **Authors:** Satvik Venkatesh, Philip Coleman, Arthur Benilov, Simon Brown, Selim Sheta, Frederic Roskam
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.01338

 - **Pdf link:** https://arxiv.org/pdf/2505.01338

 - **Abstract**
 Dereverberation is an important sub-task of Speech Enhancement (SE) to improve the signal's intelligibility and quality. However, it remains challenging because the reverberation is highly correlated with the signal. Furthermore, the single-channel SE literature has predominantly focused on rooms with short reverb times (typically under 1 second), smaller rooms (under volumes of 1000 cubic meters) and relatively short distances (up to 2 meters). In this paper, we explore real-time low-latency single-channel SE under distant microphone scenarios, such as 5 to 10 meters, and focus on conference rooms and theatres, with larger room dimensions and reverberation times. Such a setup is useful for applications such as lecture demonstrations, drama, and to enhance stage acoustics. First, we show that single-channel SE in such challenging scenarios is feasible. Second, we investigate the relationship between room volume and reverberation time, and demonstrate its importance when randomly simulating room impulse responses. Lastly, we show that for dereverberation with short decay times, preserving early reflections before decaying the transfer function of the room improves overall signal quality.
#### SMSAT: A Multimodal Acoustic Dataset and Deep Contrastive Learning Framework for Affective and Physiological Modeling of Spiritual Meditation
 - **Authors:** Ahmad Suleman, Yazeed Alkhrijah, Misha Urooj Khan, Hareem Khan, Muhammad Abdullah Husnain Ali Faiz, Mohamad A. Alawad, Zeeshan Kaleem, Guan Gui
 - **Subjects:** Subjects:
Sound (cs.SD); Social and Information Networks (cs.SI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.00839

 - **Pdf link:** https://arxiv.org/pdf/2505.00839

 - **Abstract**
 Understanding how auditory stimuli influence emotional and physiological states is fundamental to advancing affective computing and mental health technologies. In this paper, we present a multimodal evaluation of the affective and physiological impacts of three auditory conditions, that is, spiritual meditation (SM), music (M), and natural silence (NS), using a comprehensive suite of biometric signal measures. To facilitate this analysis, we introduce the Spiritual, Music, Silence Acoustic Time Series (SMSAT) dataset, a novel benchmark comprising acoustic time series (ATS) signals recorded under controlled exposure protocols, with careful attention to demographic diversity and experimental consistency. To model the auditory induced states, we develop a contrastive learning based SMSAT audio encoder that extracts highly discriminative embeddings from ATS data, achieving 99.99% classification accuracy in interclass and intraclass evaluations. Furthermore, we propose the Calmness Analysis Model (CAM), a deep learning framework integrating 25 handcrafted and learned features for affective state classification across auditory conditions, attaining robust 99.99% classification accuracy. In contrast, pairwise t tests reveal significant deviations in cardiac response characteristics (CRC) between SM analysis via ANOVA inducing more significant physiological fluctuations. Compared to existing state of the art methods reporting accuracies up to 90%, the proposed model demonstrates substantial performance gains (up to 99%). This work contributes a validated multimodal dataset and a scalable deep learning framework for affective computing applications in stress monitoring, mental well-being, and therapeutic audio-based interventions.
#### CAV-MAE Sync: Improving Contrastive Audio-Visual Mask Autoencoders via Fine-Grained Alignment
 - **Authors:** Edson Araujo, Andrew Rouditchenko, Yuan Gong, Saurabhchand Bhati, Samuel Thomas, Brian Kingsbury, Leonid Karlinsky, Rogerio Feris, James R. Glass
 - **Subjects:** Subjects:
Multimedia (cs.MM); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.01237

 - **Pdf link:** https://arxiv.org/pdf/2505.01237

 - **Abstract**
 Recent advances in audio-visual learning have shown promising results in learning representations across modalities. However, most approaches rely on global audio representations that fail to capture fine-grained temporal correspondences with visual frames. Additionally, existing methods often struggle with conflicting optimization objectives when trying to jointly learn reconstruction and cross-modal alignment. In this work, we propose CAV-MAE Sync as a simple yet effective extension of the original CAV-MAE framework for self-supervised audio-visual learning. We address three key challenges: First, we tackle the granularity mismatch between modalities by treating audio as a temporal sequence aligned with video frames, rather than using global representations. Second, we resolve conflicting optimization goals by separating contrastive and reconstruction objectives through dedicated global tokens. Third, we improve spatial localization by introducing learnable register tokens that reduce semantic load on patch tokens. We evaluate the proposed approach on AudioSet, VGG Sound, and the ADE20K Sound dataset on zero-shot retrieval, classification and localization tasks demonstrating state-of-the-art performance and outperforming more complex architectures.
#### FlowDubber: Movie Dubbing with LLM-based Semantic-aware Learning and Flow Matching based Voice Enhancing
 - **Authors:** Gaoxiang Cong, Liang Li, Jiadong Pan, Zhedong Zhang, Amin Beheshti, Anton van den Hengel, Yuankai Qi, Qingming Huang
 - **Subjects:** Subjects:
Multimedia (cs.MM); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.01263

 - **Pdf link:** https://arxiv.org/pdf/2505.01263

 - **Abstract**
 Movie Dubbing aims to convert scripts into speeches that align with the given movie clip in both temporal and emotional aspects while preserving the vocal timbre of a given brief reference audio. Existing methods focus primarily on reducing the word error rate while ignoring the importance of lip-sync and acoustic quality. To address these issues, we propose a large language model (LLM) based flow matching architecture for dubbing, named FlowDubber, which achieves high-quality audio-visual sync and pronunciation by incorporating a large speech language model and dual contrastive aligning while achieving better acoustic quality via the proposed voice-enhanced flow matching than previous works. First, we introduce Qwen2.5 as the backbone of LLM to learn the in-context sequence from movie scripts and reference audio. Then, the proposed semantic-aware learning focuses on capturing LLM semantic knowledge at the phoneme level. Next, dual contrastive aligning (DCA) boosts mutual alignment with lip movement, reducing ambiguities where similar phonemes might be confused. Finally, the proposed Flow-based Voice Enhancing (FVE) improves acoustic quality in two aspects, which introduces an LLM-based acoustics flow matching guidance to strengthen clarity and uses affine style prior to enhance identity when recovering noise into mel-spectrograms via gradient vector field prediction. Extensive experiments demonstrate that our method outperforms several state-of-the-art methods on two primary benchmarks. The demos are available at {\href{this https URL}{\textcolor{red}{this https URL}}}.


by Zyzzyva0381 (Windy). 


2025-05-05
