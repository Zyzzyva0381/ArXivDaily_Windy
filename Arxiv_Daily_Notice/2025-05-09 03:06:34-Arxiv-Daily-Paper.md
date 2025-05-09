# Showing new listings for Friday, 9 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 10papers 
#### From Dialect Gaps to Identity Maps: Tackling Variability in Speaker Verification
 - **Authors:** Abdulhady Abas Abdullah, Soran Badawi, Dana A. Abdullah, Dana Rasul Hamad, Hanan Abdulrahman Taher, Sabat Salih Muhamad, Aram Mahmood Ahmed, Bryar A. Hassan, Sirwan Abdolwahed Aula, Tarik A. Rashid
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2505.04629

 - **Pdf link:** https://arxiv.org/pdf/2505.04629

 - **Abstract**
 The complexity and difficulties of Kurdish speaker detection among its several dialects are investigated in this work. Because of its great phonetic and lexical differences, Kurdish with several dialects including Kurmanji, Sorani, and Hawrami offers special challenges for speaker recognition systems. The main difficulties in building a strong speaker identification system capable of precisely identifying speakers across several dialects are investigated in this work. To raise the accuracy and dependability of these systems, it also suggests solutions like sophisticated machine learning approaches, data augmentation tactics, and the building of thorough dialect-specific corpus. The results show that customized strategies for every dialect together with cross-dialect training greatly enhance recognition performance.
#### Listen to Extract: Onset-Prompted Target Speaker Extraction
 - **Authors:** Pengjie Shen, Kangrui Chen, Shulin He, Pengru Chen, Shuqi Yuan, He Kong, Xueliang Zhang, Zhong-Qiu Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.05114

 - **Pdf link:** https://arxiv.org/pdf/2505.05114

 - **Abstract**
 We propose $\textit{listen to extract}$ (LExt), a highly-effective while extremely-simple algorithm for monaural target speaker extraction (TSE). Given an enrollment utterance of a target speaker, LExt aims at extracting the target speaker from the speaker's mixed speech with other speakers. For each mixture, LExt concatenates an enrollment utterance of the target speaker to the mixture signal at the waveform level, and trains deep neural networks (DNN) to extract the target speech based on the concatenated mixture signal. The rationale is that, this way, an artificial speech onset is created for the target speaker and it could prompt the DNN (a) which speaker is the target to extract; and (b) spectral-temporal patterns of the target speaker that could help extraction. This simple approach produces strong TSE performance on multiple public TSE datasets including WSJ0-2mix, WHAM! and WHAMR!.
#### Regression-based Melody Estimation with Uncertainty Quantification
 - **Authors:** Kavya Ranjan Saxena, Vipul Arora
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.05156

 - **Pdf link:** https://arxiv.org/pdf/2505.05156

 - **Abstract**
 Existing machine learning models approach the task of melody estimation from polyphonic audio as a classification problem by discretizing the pitch values, which results in the loss of finer frequency variations present in the melody. To better capture these variations, we propose to approach this task as a regression problem. Apart from predicting only the pitch for a particular region in the audio, we also predict its uncertainty to enhance the trustworthiness of the model. To perform regression-based melody estimation, we propose three different methods that use histogram representation to model the pitch values. Such a representation requires the support range of the histogram to be continuous. The first two methods address the abrupt discontinuity between unvoiced and voiced frequency ranges by mapping them to a continuous range. The third method reformulates melody estimation as a fully Bayesian task, modeling voicing detection as a classification problem, and voiced pitch estimation as a regression problem. Additionally, we introduce a novel method to estimate the uncertainty from the histogram representation that correlates well with the deviation of the mean of the predicted distribution from the ground truth. Experimental results demonstrate that reformulating melody estimation as a regression problem significantly improves the performance over classification-based approaches. Comparing the proposed methods with a state-of-the-art regression model, it is observed that the Bayesian method performs the best at estimating both the melody and its associated uncertainty.
#### FlexSpeech: Towards Stable, Controllable and Expressive Text-to-Speech
 - **Authors:** Linhan Ma, Dake Guo, He Wang, Jin Xu, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.05159

 - **Pdf link:** https://arxiv.org/pdf/2505.05159

 - **Abstract**
 Current speech generation research can be categorized into two primary classes: non-autoregressive and autoregressive. The fundamental distinction between these approaches lies in the duration prediction strategy employed for predictable-length sequences. The NAR methods ensure stability in speech generation by explicitly and independently modeling the duration of each phonetic unit. Conversely, AR methods employ an autoregressive paradigm to predict the compressed speech token by implicitly modeling duration with Markov properties. Although this approach improves prosody, it does not provide the structural guarantees necessary for stability. To simultaneously address the issues of stability and naturalness in speech generation, we propose FlexSpeech, a stable, controllable, and expressive TTS model. The motivation behind FlexSpeech is to incorporate Markov dependencies and preference optimization directly on the duration predictor to boost its naturalness while maintaining explicit modeling of the phonetic units to ensure stability. Specifically, we decompose the speech generation task into two components: an AR duration predictor and a NAR acoustic model. The acoustic model is trained on a substantial amount of data to learn to render audio more stably, given reference audio prosody and phone durations. The duration predictor is optimized in a lightweight manner for different stylistic variations, thereby enabling rapid style transfer while maintaining a decoupled relationship with the specified speaker timbre. Experimental results demonstrate that our approach achieves SOTA stability and naturalness in zero-shot TTS. More importantly, when transferring to a specific stylistic domain, we can accomplish lightweight optimization of the duration module solely with about 100 data samples, without the need to adjust the acoustic model, thereby enabling rapid and stable style transfer.
#### Normalize Everything: A Preconditioned Magnitude-Preserving Architecture for Diffusion-Based Speech Enhancement
 - **Authors:** Julius Richter, Danilo de Oliveira, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.05216

 - **Pdf link:** https://arxiv.org/pdf/2505.05216

 - **Abstract**
 This paper presents a new framework for diffusion-based speech enhancement. Our method employs a Schroedinger bridge to transform the noisy speech distribution into the clean speech distribution. To stabilize and improve training, we employ time-dependent scalings of the inputs and outputs of the network, known as preconditioning. We consider two skip connection configurations, which either include or omit the current process state in the denoiser's output, enabling the network to predict either environmental noise or clean speech. Each approach leads to improved performance on different speech enhancement metrics. To maintain stable magnitude levels and balance during training, we use a magnitude-preserving network architecture that normalizes all activations and network weights to unit length. Additionally, we propose learning the contribution of the noisy input within each network block for effective input conditioning. After training, we apply a method to approximate different exponential moving average (EMA) profiles and investigate their effects on the speech enhancement performance. In contrast to image generation tasks, where longer EMA lengths often enhance mode coverage, we observe that shorter EMA lengths consistently lead to better performance on standard speech enhancement metrics. Code, audio examples, and checkpoints are available online.
#### A Multi-Agent AI Framework for Immersive Audiobook Production through Spatial Audio and Neural Narration
 - **Authors:** Shaja Arul Selvamani, Nia D'Souza Ganapathy
 - **Subjects:** Subjects:
Sound (cs.SD); Human-Computer Interaction (cs.HC); Multiagent Systems (cs.MA); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.04885

 - **Pdf link:** https://arxiv.org/pdf/2505.04885

 - **Abstract**
 This research introduces an innovative AI-driven multi-agent framework specifically designed for creating immersive audiobooks. Leveraging neural text-to-speech synthesis with FastSpeech 2 and VALL-E for expressive narration and character-specific voices, the framework employs advanced language models to automatically interpret textual narratives and generate realistic spatial audio effects. These sound effects are dynamically synchronized with the storyline through sophisticated temporal integration methods, including Dynamic Time Warping (DTW) and recurrent neural networks (RNNs). Diffusion-based generative models combined with higher-order ambisonics (HOA) and scattering delay networks (SDN) enable highly realistic 3D soundscapes, substantially enhancing listener immersion and narrative realism. This technology significantly advances audiobook applications, providing richer experiences for educational content, storytelling platforms, and accessibility solutions for visually impaired audiences. Future work will address personalization, ethical management of synthesized voices, and integration with multi-sensory platforms.
#### Inter-Diffusion Generation Model of Speakers and Listeners for Effective Communication
 - **Authors:** Jinhe Huang, Yongkang Cheng, Yuming Hang, Gaoge Han, Jinewei Li, Jing Zhang, Xingjian Gu
 - **Subjects:** Subjects:
Graphics (cs.GR); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.04996

 - **Pdf link:** https://arxiv.org/pdf/2505.04996

 - **Abstract**
 Full-body gestures play a pivotal role in natural interactions and are crucial for achieving effective communication. Nevertheless, most existing studies primarily focus on the gesture generation of speakers, overlooking the vital role of listeners in the interaction process and failing to fully explore the dynamic interaction between them. This paper innovatively proposes an Inter-Diffusion Generation Model of Speakers and Listeners for Effective Communication. For the first time, we integrate the full-body gestures of listeners into the generation framework. By devising a novel inter-diffusion mechanism, this model can accurately capture the complex interaction patterns between speakers and listeners during communication. In the model construction process, based on the advanced diffusion model architecture, we innovatively introduce interaction conditions and the GAN model to increase the denoising step size. As a result, when generating gesture sequences, the model can not only dynamically generate based on the speaker's speech information but also respond in realtime to the listener's feedback, enabling synergistic interaction between the two. Abundant experimental results demonstrate that compared with the current state-of-the-art gesture generation methods, the model we proposed has achieved remarkable improvements in the naturalness, coherence, and speech-gesture synchronization of the generated gestures. In the subjective evaluation experiments, users highly praised the generated interaction scenarios, believing that they are closer to real life human communication situations. Objective index evaluations also show that our model outperforms the baseline methods in multiple key indicators, providing more powerful support for effective communication.
#### ReverbMiipher: Generative Speech Restoration meets Reverberation Characteristics Controllability
 - **Authors:** Wataru Nakata, Yuma Koizumi, Shigeki Karita, Robin Scheibler, Haruko Ishikawa, Adriana Guevara-Rukoz, Heiga Zen, Michiel Bacchiani
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.05077

 - **Pdf link:** https://arxiv.org/pdf/2505.05077

 - **Abstract**
 Reverberation encodes spatial information regarding the acoustic source environment, yet traditional Speech Restoration (SR) usually completely removes reverberation. We propose ReverbMiipher, an SR model extending parametric resynthesis framework, designed to denoise speech while preserving and enabling control over reverberation. ReverbMiipher incorporates a dedicated ReverbEncoder to extract a reverb feature vector from noisy input. This feature conditions a vocoder to reconstruct the speech signal, removing noise while retaining the original reverberation characteristics. A stochastic zero-vector replacement strategy during training ensures the feature specifically encodes reverberation, disentangling it from other speech attributes. This learned representation facilitates reverberation control via techniques such as interpolation between features, replacement with features from other utterances, or sampling from a latent space. Objective and subjective evaluations confirm ReverbMiipher effectively preserves reverberation, removes other artifacts, and outperforms the conventional two-stage SR and convolving simulated room impulse response approach. We further demonstrate its ability to generate novel reverberation effects through feature manipulation.
#### FLAM: Frame-Wise Language-Audio Modeling
 - **Authors:** Yusong Wu, Christos Tsirigotis, Ke Chen, Cheng-Zhi Anna Huang, Aaron Courville, Oriol Nieto, Prem Seetharaman, Justin Salamon
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.05335

 - **Pdf link:** https://arxiv.org/pdf/2505.05335

 - **Abstract**
 Recent multi-modal audio-language models (ALMs) excel at text-audio retrieval but struggle with frame-wise audio understanding. Prior works use temporal-aware labels or unsupervised training to improve frame-wise capabilities, but they still lack fine-grained labeling capability to pinpoint when an event occurs. While traditional sound event detection models can precisely localize events, they are limited to pre-defined categories, making them ineffective for real-world scenarios with out-of-distribution events. In this work, we introduce FLAM, an open-vocabulary contrastive audio-language model capable of localizing specific sound events. FLAM employs a memory-efficient and calibrated frame-wise objective with logit adjustment to address spurious correlations, such as event dependencies and label imbalances during training. To enable frame-wise supervision, we leverage a large-scale dataset with diverse audio events, LLM-generated captions and simulation. Experimental results and case studies demonstrate that FLAM significantly improves the open-vocabulary localization capability while maintaining strong performance in global retrieval and downstream tasks.
#### Hearing and Seeing Through CLIP: A Framework for Self-Supervised Sound Source Localization
 - **Authors:** Sooyoung Park, Arda Senocak, Joon Son Chung
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.05343

 - **Pdf link:** https://arxiv.org/pdf/2505.05343

 - **Abstract**
 Large-scale vision-language models demonstrate strong multimodal alignment and generalization across diverse tasks. Among them, CLIP stands out as one of the most successful approaches. In this work, we extend the application of CLIP to sound source localization, proposing a self-supervised method operates without explicit text input. We introduce a framework that maps audios into tokens compatible with CLIP's text encoder, producing audio-driven embeddings. These embeddings are used to generate sounding region masks, from which visual features are extracted and aligned with the audio embeddings through a contrastive audio-visual correspondence objective. Our findings show that alignment knowledge of pre-trained multimodal foundation model enables our method to generate more complete and compact localization for sounding objects. We further propose an LLM-guided extension that distills object-aware audio-visual scene understanding into the model during training to enhance alignment. Extensive experiments across five diverse tasks demonstrate that our method, in all variants, outperforms state-of-the-art approaches and achieves strong generalization in zero-shot settings.


by Zyzzyva0381 (Windy). 


2025-05-09
