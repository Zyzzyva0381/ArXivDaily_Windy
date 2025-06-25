# Showing new listings for Wednesday, 25 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 9papers 
#### Loss functions incorporating auditory spatial perception in deep learning -- a review
 - **Authors:** Boaz Rafaely, Stefan Weinzierl, Or Berebi, Fabian Brinkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.19404

 - **Pdf link:** https://arxiv.org/pdf/2506.19404

 - **Abstract**
 Binaural reproduction aims to deliver immersive spatial audio with high perceptual realism over headphones. Loss functions play a central role in optimizing and evaluating algorithms that generate binaural signals. However, traditional signal-related difference measures often fail to capture the perceptual properties that are essential to spatial audio quality. This review paper surveys recent loss functions that incorporate spatial perception cues relevant to binaural reproduction. It focuses on losses applied to binaural signals, which are often derived from microphone recordings or Ambisonics signals, while excluding those based on room impulse responses. Guided by the Spatial Audio Quality Inventory (SAQI), the review emphasizes perceptual dimensions related to source localization and room response, while excluding general spectral-temporal attributes. The literature survey reveals a strong focus on localization cues, such as interaural time and level differences (ITDs, ILDs), while reverberation and other room acoustic attributes remain less explored in loss function design. Recent works that estimate room acoustic parameters and develop embeddings that capture room characteristics indicate their potential for future integration into neural network training. The paper concludes by highlighting future research directions toward more perceptually grounded loss functions that better capture the listener's spatial experience.
#### Kling-Foley: Multimodal Diffusion Transformer for High-Quality Video-to-Audio Generation
 - **Authors:** Jun Wang, Xijuan Zeng, Chunyu Qiang, Ruilong Chen, Shiyao Wang, Le Wang, Wangjing Zhou, Pengfei Cai, Jiahui Zhao, Nan Li, Zihan Li, Yuzhe Liang, Xiaopeng Wang, Haorui Zheng, Ming Wen, Kang Yin, Yiran Wang, Nan Li, Feng Deng, Liang Dong, Chen Zhang, Di Zhang, Kun Gai
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.19774

 - **Pdf link:** https://arxiv.org/pdf/2506.19774

 - **Abstract**
 We propose Kling-Foley, a large-scale multimodal Video-to-Audio generation model that synthesizes high-quality audio synchronized with video content. In Kling-Foley, we introduce multimodal diffusion transformers to model the interactions between video, audio, and text modalities, and combine it with a visual semantic representation module and an audio-visual synchronization module to enhance alignment capabilities. Specifically, these modules align video conditions with latent audio elements at the frame level, thereby improving semantic alignment and audio-visual synchronization. Together with text conditions, this integrated approach enables precise generation of video-matching sound effects. In addition, we propose a universal latent audio codec that can achieve high-quality modeling in various scenarios such as sound effects, speech, singing, and music. We employ a stereo rendering method that imbues synthesized audio with a spatial presence. At the same time, in order to make up for the incomplete types and annotations of the open-source benchmark, we also open-source an industrial-level benchmark Kling-Audio-Eval. Our experiments show that Kling-Foley trained with the flow matching objective achieves new audio-visual SOTA performance among public models in terms of distribution matching, semantic alignment, temporal alignment and audio quality.
#### SHAMaNS: Sound Localization with Hybrid Alpha-Stable Spatial Measure and Neural Steerer
 - **Authors:** Diego Di Carlo (RIKEN AIP), Mathieu Fontaine (LTCI, IP Paris), Aditya Arie Nugraha (RIKEN AIP), Yoshiaki Bando (RIKEN AIP), Kazuyoshi Yoshii
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18954

 - **Pdf link:** https://arxiv.org/pdf/2506.18954

 - **Abstract**
 This paper describes a sound source localization (SSL) technique that combines an $\alpha$-stable model for the observed signal with a neural network-based approach for modeling steering vectors. Specifically, a physics-informed neural network, referred to as Neural Steerer, is used to interpolate measured steering vectors (SVs) on a fixed microphone array. This allows for a more robust estimation of the so-called $\alpha$-stable spatial measure, which represents the most plausible direction of arrival (DOA) of a target signal. As an $\alpha$-stable model for the non-Gaussian case ($\alpha$ $\in$ (0, 2)) theoretically defines a unique spatial measure, we choose to leverage it to account for residual reconstruction error of the Neural Steerer in the downstream tasks. The objective scores indicate that our proposed technique outperforms state-of-the-art methods in the case of multiple sound sources.
#### IndieFake Dataset: A Benchmark Dataset for Audio Deepfake Detection
 - **Authors:** Abhay Kumar, Kunal Verma, Omkar More
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.19014

 - **Pdf link:** https://arxiv.org/pdf/2506.19014

 - **Abstract**
 Advancements in audio deepfake technology offers benefits like AI assistants, better accessibility for speech impairments, and enhanced entertainment. However, it also poses significant risks to security, privacy, and trust in digital communications. Detecting and mitigating these threats requires comprehensive datasets. Existing datasets lack diverse ethnic accents, making them inadequate for many real-world scenarios. Consequently, models trained on these datasets struggle to detect audio deepfakes in diverse linguistic and cultural contexts such as in South-Asian countries. Ironically, there is a stark lack of South-Asian speaker samples in the existing datasets despite constituting a quarter of the worlds population. This work introduces the IndieFake Dataset (IFD), featuring 27.17 hours of bonafide and deepfake audio from 50 English speaking Indian speakers. IFD offers balanced data distribution and includes speaker-level characterization, absent in datasets like ASVspoof21 (DF). We evaluated various baselines on IFD against existing ASVspoof21 (DF) and In-The-Wild (ITW) datasets. IFD outperforms ASVspoof21 (DF) and proves to be more challenging compared to benchmark ITW dataset. The dataset will be publicly available upon acceptance.
#### Enhanced Hybrid Transducer and Attention Encoder Decoder with Text Data
 - **Authors:** Yun Tang, Eesung Kim, Vijendra Raj Apsingekar
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.19159

 - **Pdf link:** https://arxiv.org/pdf/2506.19159

 - **Abstract**
 A joint speech and text optimization method is proposed for hybrid transducer and attention-based encoder decoder (TAED) modeling to leverage large amounts of text corpus and enhance ASR accuracy. The joint TAED (J-TAED) is trained with both speech and text input modalities together, while it only takes speech data as input during inference. The trained model can unify the internal representations from different modalities, and be further extended to text-based domain adaptation. It can effectively alleviate data scarcity for mismatch domain tasks since no speech data is required. Our experiments show J-TAED successfully integrates speech and linguistic information into one model, and reduce the WER by 5.8 ~12.8% on the Librispeech dataset. The model is also evaluated on two out-of-domain datasets: one is finance and another is named entity focused. The text-based domain adaptation brings 15.3% and 17.8% WER reduction on those two datasets respectively.
#### A Robust Method for Pitch Tracking in the Frequency Following Response using Harmonic Amplitude Summation Filterbank
 - **Authors:** Sajad Sadeghkhani, Maryam Karimi Boroujeni, Hilmi R. Dajani, Saeid R. Seydnejad, Christian Giguère
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.19253

 - **Pdf link:** https://arxiv.org/pdf/2506.19253

 - **Abstract**
 The Frequency Following Response (FFR) reflects the brain's neural encoding of auditory stimuli including speech. Because the fundamental frequency (F0), a physical correlate of pitch, is one of the essential features of speech, there has been particular interest in characterizing the FFR at F0, especially when F0 varies over time. The standard method for extracting F0 in FFRs has been the Autocorrelation Function (ACF). This paper investigates harmonic-structure-based F0 estimation algorithms, originally developed for speech and music, and resolves their poor performance when applied to FFRs in two steps. Firstly, given that unlike in speech or music, stimulus F0 of FFRs is already known, we introduce a stimulus-aware filterbank that selectively aggregates amplitudes at F0 and its harmonics while suppressing noise at non-harmonic frequencies. This method, called Harmonic Amplitude Summation (HAS), evaluates F0 candidates only within a range centered around the stimulus F0. Secondly, unlike other pitch tracking methods that select the highest peak, our method chooses the most prominent one, as it better reflects the underlying periodicity of FFRs. To the best of our knowledge, this is the first study to propose an F0 estimation algorithm for FFRs that relies on harmonic structure. Analyzing recorded FFRs from 16 normal hearing subjects to 4 natural speech stimuli with a wide F0 variation from 89 Hz to 452 Hz showed that this method outperformed ACF by reducing the average Root-Mean-Square-Error (RMSE) within each response and stimulus F0 contour pair by 8.8% to 47.4%, depending on the stimulus.
#### JCAPT: A Joint Modeling Approach for CAPT
 - **Authors:** Tzu-Hsuan Yang, Yue-Yang He, Berlin Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.19315

 - **Pdf link:** https://arxiv.org/pdf/2506.19315

 - **Abstract**
 Effective pronunciation feedback is critical in second language (L2) learning, for which computer-assisted pronunciation training (CAPT) systems often encompass two key tasks: automatic pronunciation assessment (APA) and mispronunciation detection and diagnosis (MDD). Recent work has shown that joint modeling of these two tasks can yield mutual benefits. Our unified framework leverages Mamba, a selective state space model (SSM), while integrating phonological features and think token strategies to jointly enhance interpretability and fine-grained temporal reasoning in APA and MDD. To our knowledge, this is the first study to combine phonological attribution, SSM-based modeling, and prompting in CAPT. A series of experiments conducted on the speechocean762 benchmark demonstrate that our model consistently outperforms prior methods, particularly on the MDD task.
#### ClearerVoice-Studio: Bridging Advanced Speech Processing Research and Practical Deployment
 - **Authors:** Shengkui Zhao, Zexu Pan, Bin Ma
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.19398

 - **Pdf link:** https://arxiv.org/pdf/2506.19398

 - **Abstract**
 This paper introduces ClearerVoice-Studio, an open-source, AI-powered speech processing toolkit designed to bridge cutting-edge research and practical application. Unlike broad platforms like SpeechBrain and ESPnet, ClearerVoice-Studio focuses on interconnected speech tasks of speech enhancement, separation, super-resolution, and multimodal target speaker extraction. A key advantage is its state-of-the-art pretrained models, including FRCRN with 3 million uses and MossFormer with 2.5 million uses, optimized for real-world scenarios. It also offers model optimization tools, multi-format audio support, the SpeechScore evaluation toolkit, and user-friendly interfaces, catering to researchers, developers, and end-users. Its rapid adoption attracting 3000 GitHub stars and 239 forks highlights its academic and industrial impact. This paper details ClearerVoice-Studio's capabilities, architectures, training strategies, benchmarks, community impact, and future plan. Source code is available at this https URL.
#### TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems
 - **Authors:** Christoph Minixhofer, Ondrej Klejch, Peter Bell
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.19441

 - **Pdf link:** https://arxiv.org/pdf/2506.19441

 - **Abstract**
 Evaluation of Text to Speech (TTS) systems is challenging and resource-intensive. Subjective metrics such as Mean Opinion Score (MOS) are not easily comparable between works. Objective metrics are frequently used, but rarely validated against subjective ones. Both kinds of metrics are challenged by recent TTS systems capable of producing synthetic speech indistinguishable from real speech. In this work, we introduce Text to Speech Distribution Score 2 (TTSDS2), a more robust and improved version of TTSDS. Across a range of domains and languages, it is the only one out of 16 compared metrics to correlate with a Spearman correlation above 0.50 for every domain and subjective score evaluated. We also release a range of resources for evaluating synthetic speech close to real speech: A dataset with over 11,000 subjective opinion score ratings; a pipeline for continually recreating a multilingual test dataset to avoid data leakage; and a continually updated benchmark for TTS in 14 languages.


by Zyzzyva0381 (Windy). 


2025-06-25
