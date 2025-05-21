# Showing new listings for Wednesday, 21 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 37papers 
#### Exploring Emotional Synchrony in Dyadic Interactions: The Role of Speech Conditions in Facial and Vocal Affective Alignment
 - **Authors:** Von Ralph Dane Marquez Herbuela, Yukie Nagai
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2505.13455

 - **Pdf link:** https://arxiv.org/pdf/2505.13455

 - **Abstract**
 Understanding how humans express and synchronize emotions across multiple communication channels particularly facial expressions and speech has significant implications for emotion recognition systems and human computer interaction. Motivated by the notion that non-overlapping speech promotes clearer emotional coordination, while overlapping speech disrupts synchrony, this study examines how these conversational dynamics shape the spatial and temporal alignment of arousal and valence across facial and vocal modalities. Using dyadic interactions from the IEMOCAP dataset, we extracted continuous emotion estimates via EmoNet (facial video) and a Wav2Vec2-based model (speech audio). Segments were categorized based on speech overlap, and emotional alignment was assessed using Pearson correlation, lag adjusted analysis, and Dynamic Time Warping (DTW). Across analyses, non overlapping speech was associated with more stable and predictable emotional synchrony than overlapping speech. While zero-lag correlations were low and not statistically different, non overlapping speech showed reduced variability, especially for arousal. Lag adjusted correlations and best-lag distributions revealed clearer, more consistent temporal alignment in these segments. In contrast, overlapping speech exhibited higher variability and flatter lag profiles, though DTW indicated unexpectedly tighter alignment suggesting distinct coordination strategies. Notably, directionality patterns showed that facial expressions more often preceded speech during turn-taking, while speech led during simultaneous vocalizations. These findings underscore the importance of conversational structure in regulating emotional communication and provide new insight into the spatial and temporal dynamics of multimodal affective alignment in real world interaction.
#### SPIRIT: Patching Speech Language Models against Jailbreak Attacks
 - **Authors:** Amirbek Djanibekov, Nurdaulet Mukhituly, Kentaro Inui, Hanan Aldarmaki, Nils Lukas
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2505.13541

 - **Pdf link:** https://arxiv.org/pdf/2505.13541

 - **Abstract**
 Speech Language Models (SLMs) enable natural interactions via spoken instructions, which more effectively capture user intent by detecting nuances in speech. The richer speech signal introduces new security risks compared to text-based models, as adversaries can better bypass safety mechanisms by injecting imperceptible noise to speech. We analyze adversarial attacks and find that SLMs are substantially more vulnerable to jailbreak attacks, which can achieve a perfect 100% attack success rate in some instances. To improve security, we propose post-hoc patching defenses used to intervene during inference by modifying the SLM's activations that improve robustness up to 99% with (i) negligible impact on utility and (ii) without any re-training. We conduct ablation studies to maximize the efficacy of our defenses and improve the utility/security trade-off, validated with large-scale benchmarks unique to SLMs.
#### Direction-Aware Neural Acoustic Fields for Few-Shot Interpolation of Ambisonic Impulse Responses
 - **Authors:** Christopher Ick, Gordon Wichern, Yoshiki Masuyama, François Germain, Jonathan Le Roux
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.13617

 - **Pdf link:** https://arxiv.org/pdf/2505.13617

 - **Abstract**
 The characteristics of a sound field are intrinsically linked to the geometric and spatial properties of the environment surrounding a sound source and a listener. The physics of sound propagation is captured in a time-domain signal known as a room impulse response (RIR). Prior work using neural fields (NFs) has allowed learning spatially-continuous representations of RIRs from finite RIR measurements. However, previous NF-based methods have focused on monaural omnidirectional or at most binaural listeners, which does not precisely capture the directional characteristics of a real sound field at a single point. We propose a direction-aware neural field (DANF) that more explicitly incorporates the directional information by Ambisonic-format RIRs. While DANF inherently captures spatial relations between sources and listeners, we further propose a direction-aware loss. In addition, we investigate the ability of DANF to adapt to new rooms in various ways including low-rank adaptation.
#### Articulatory Feature Prediction from Surface EMG during Speech Production
 - **Authors:** Jihwan Lee, Kevin Huang, Kleanthis Avramidis, Simon Pistrosch, Monica Gonzalez-Machorro, Yoonjeong Lee, Björn Schuller, Louis Goldstein, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.13814

 - **Pdf link:** https://arxiv.org/pdf/2505.13814

 - **Abstract**
 We present a model for predicting articulatory features from surface electromyography (EMG) signals during speech production. The proposed model integrates convolutional layers and a Transformer block, followed by separate predictors for articulatory features. Our approach achieves a high prediction correlation of approximately 0.9 for most articulatory features. Furthermore, we demonstrate that these predicted articulatory features can be decoded into intelligible speech waveforms. To our knowledge, this is the first method to decode speech waveforms from surface EMG via articulatory features, offering a novel approach to EMG-based speech synthesis. Additionally, we analyze the relationship between EMG electrode placement and articulatory feature predictability, providing knowledge-driven insights for optimizing EMG electrode configurations. The source code and decoded speech samples are publicly available.
#### Pushing the Frontiers of Self-Distillation Prototypes Network with Dimension Regularization and Score Normalization
 - **Authors:** Yafeng Chen, Chong Deng, Hui Wang, Yiheng Jiang, Han Yin, Qian Chen, Wen Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.13826

 - **Pdf link:** https://arxiv.org/pdf/2505.13826

 - **Abstract**
 Developing robust speaker verification (SV) systems without speaker labels has been a longstanding challenge. Earlier research has highlighted a considerable performance gap between self-supervised and fully supervised approaches. In this paper, we enhance the non-contrastive self-supervised framework, Self-Distillation Prototypes Network (SDPN), by introducing dimension regularization that explicitly addresses the collapse problem through the application of regularization terms to speaker embeddings. Moreover, we integrate score normalization techniques from fully supervised SV to further bridge the gap toward supervised verification performance. SDPN with dimension regularization and score normalization sets a new state-of-the-art on the VoxCeleb1 speaker verification evaluation benchmark, achieving Equal Error Rate 1.29%, 1.60%, and 2.80% for trial VoxCeleb1-{O,E,H} respectively. These results demonstrate relative improvements of 28.3%, 19.6%, and 22.6% over the current best self-supervised methods, thereby advancing the frontiers of SV technology.
#### Improving Noise Robustness of LLM-based Zero-shot TTS via Discrete Acoustic Token Denoising
 - **Authors:** Ye-Xin Lu, Hui-Peng Du, Fei Liu, Yang Ai, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.13830

 - **Pdf link:** https://arxiv.org/pdf/2505.13830

 - **Abstract**
 Large language model (LLM) based zero-shot text-to-speech (TTS) methods tend to preserve the acoustic environment of the audio prompt, leading to degradation in synthesized speech quality when the audio prompt contains noise. In this paper, we propose a novel neural codec-based speech denoiser and integrate it with the advanced LLM-based TTS model, LauraTTS, to achieve noise-robust zero-shot TTS. The proposed codec denoiser consists of an audio codec, a token denoiser, and an embedding refiner. The token denoiser predicts the first two groups of clean acoustic tokens from the noisy ones, which can serve as the acoustic prompt for LauraTTS to synthesize high-quality personalized speech or be converted to clean speech waveforms through the embedding refiner and codec decoder. Experimental results show that our proposed codec denoiser outperforms state-of-the-art speech enhancement (SE) methods, and the proposed noise-robust LauraTTS surpasses the approach using additional SE models.
#### A Semantic Information-based Hierarchical Speech Enhancement Method Using Factorized Codec and Diffusion Model
 - **Authors:** Yang Xiang, Canan Huang, Desheng Hu, Jingguang Tian, Xinhui Hu, Chao Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.13843

 - **Pdf link:** https://arxiv.org/pdf/2505.13843

 - **Abstract**
 Most current speech enhancement (SE) methods recover clean speech from noisy inputs by directly estimating time-frequency masks or spectrums. However, these approaches often neglect the distinct attributes, such as semantic content and acoustic details, inherent in speech signals, which can hinder performance in downstream tasks. Moreover, their effectiveness tends to degrade in complex acoustic environments. To overcome these challenges, we propose a novel, semantic information-based, step-by-step factorized SE method using factorized codec and diffusion model. Unlike traditional SE methods, our hierarchical modeling of semantic and acoustic attributes enables more robust clean speech recovery, particularly in challenging acoustic scenarios. Moreover, this method offers further advantages for downstream TTS tasks. Experimental results demonstrate that our algorithm not only outperforms SOTA baselines in terms of speech quality but also enhances TTS performance in noisy environments.
#### U-SAM: An audio language Model for Unified Speech, Audio, and Music Understanding
 - **Authors:** Ziqian Wang, Xianjun Xia, Xinfa Zhu, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2505.13880

 - **Pdf link:** https://arxiv.org/pdf/2505.13880

 - **Abstract**
 The text generation paradigm for audio tasks has opened new possibilities for unified audio understanding. However, existing models face significant challenges in achieving a comprehensive understanding across diverse audio types, such as speech, general audio events, and music. Furthermore, their exclusive reliance on cross-entropy loss for alignment often falls short, as it treats all tokens equally and fails to account for redundant audio features, leading to weaker cross-modal alignment. To deal with the above challenges, this paper introduces U-SAM, an advanced audio language model that integrates specialized encoders for speech, audio, and music with a pre-trained large language model (LLM). U-SAM employs a Mixture of Experts (MoE) projector for task-aware feature fusion, dynamically routing and integrating the domain-specific encoder outputs. Additionally, U-SAM incorporates a Semantic-Aware Contrastive Loss Module, which explicitly identifies redundant audio features under language supervision and rectifies their semantic and spectral representations to enhance cross-modal alignment. Extensive experiments demonstrate that U-SAM consistently outperforms both specialized models and existing audio language models across multiple benchmarks. Moreover, it exhibits emergent capabilities on unseen tasks, showcasing its generalization potential. Code is available (this https URL).
#### Naturalness-Aware Curriculum Learning with Dynamic Temperature for Speech Deepfake Detection
 - **Authors:** Taewoo Kim, Guisik Kim, Choongsang Cho, Young Han Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.13976

 - **Pdf link:** https://arxiv.org/pdf/2505.13976

 - **Abstract**
 Recent advances in speech deepfake detection (SDD) have significantly improved artifacts-based detection in spoofed speech. However, most models overlook speech naturalness, a crucial cue for distinguishing bona fide speech from spoofed speech. This study proposes naturalness-aware curriculum learning, a novel training framework that leverages speech naturalness to enhance the robustness and generalization of SDD. This approach measures sample difficulty using both ground-truth labels and mean opinion scores, and adjusts the training schedule to progressively introduce more challenging samples. To further improve generalization, a dynamic temperature scaling method based on speech naturalness is incorporated into the training process. A 23% relative reduction in the EER was achieved in the experiments on the ASVspoof 2021 DF dataset, without modifying the model architecture. Ablation studies confirmed the effectiveness of naturalness-aware training strategies for SDD tasks.
#### SeamlessEdit: Background Noise Aware Zero-Shot Speech Editing with in-Context Enhancement
 - **Authors:** Kuan-Yu Chen, Jeng-Lin Li, Jian-Jiun Ding
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.14066

 - **Pdf link:** https://arxiv.org/pdf/2505.14066

 - **Abstract**
 With the fast development of zero-shot text-to-speech technologies, it is possible to generate high-quality speech signals that are indistinguishable from the real ones. Speech editing, including speech insertion and replacement, appeals to researchers due to its potential applications. However, existing studies only considered clean speech scenarios. In real-world applications, the existence of environmental noise could significantly degrade the quality of the generation. In this study, we propose a noise-resilient speech editing framework, SeamlessEdit, for noisy speech editing. SeamlessEdit adopts a frequency-band-aware noise suppression module and an in-content refinement strategy. It can well address the scenario where the frequency bands of voice and background noise are not separated. The proposed SeamlessEdit framework outperforms state-of-the-art approaches in multiple quantitative and qualitative evaluations.
#### Scaling and Enhancing LLM-based AVSR: A Sparse Mixture of Projectors Approach
 - **Authors:** Umberto Cappellazzo, Minsu Kim, Stavros Petridis, Daniele Falavigna, Alessio Brutti
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.14336

 - **Pdf link:** https://arxiv.org/pdf/2505.14336

 - **Abstract**
 Audio-Visual Speech Recognition (AVSR) enhances robustness in noisy environments by integrating visual cues. While recent advances integrate Large Language Models (LLMs) into AVSR, their high computational cost hinders deployment in resource-constrained settings. To address this, we propose Llama-SMoP, an efficient Multimodal LLM that employs a Sparse Mixture of Projectors (SMoP) module to scale model capacity without increasing inference costs. By incorporating sparsely-gated mixture-of-experts (MoE) projectors, Llama-SMoP enables the use of smaller LLMs while maintaining strong performance. We explore three SMoP configurations and show that Llama-SMoP DEDR (Disjoint-Experts, Disjoint-Routers), which uses modality-specific routers and experts, achieves superior performance on ASR, VSR, and AVSR tasks. Ablation studies confirm its effectiveness in expert activation, scalability, and noise robustness.
#### Pairwise Evaluation of Accent Similarity in Speech Synthesis
 - **Authors:** Jinzuomu Zhong, Suyuan Liu, Dan Wells, Korin Richmond
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.14410

 - **Pdf link:** https://arxiv.org/pdf/2505.14410

 - **Abstract**
 Despite growing interest in generating high-fidelity accents, evaluating accent similarity in speech synthesis has been underexplored. We aim to enhance both subjective and objective evaluation methods for accent similarity. Subjectively, we refine the XAB listening test by adding components that achieve higher statistical significance with fewer listeners and lower costs. Our method involves providing listeners with transcriptions, having them highlight perceived accent differences, and implementing meticulous screening for reliability. Objectively, we utilise pronunciation-related metrics, based on distances between vowel formants and phonetic posteriorgrams, to evaluate accent generation. Comparative experiments reveal that these metrics, alongside accent similarity, speaker similarity, and Mel Cepstral Distortion, can be used. Moreover, our findings underscore significant limitations of common metrics like Word Error Rate in assessing underrepresented accents.
#### Single-Channel Target Speech Extraction Utilizing Distance and Room Clues
 - **Authors:** Runwu Shi, Zirui Lin, Benjamin Yen, Jiang Wang, Ragib Amin Nihal, Kazuhiro Nakadai
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.14433

 - **Pdf link:** https://arxiv.org/pdf/2505.14433

 - **Abstract**
 This paper aims to achieve single-channel target speech extraction (TSE) in enclosures utilizing distance clues and room information. Recent works have verified the feasibility of distance clues for the TSE task, which can imply the sound source's direct-to-reverberation ratio (DRR) and thus can be utilized for speech separation and TSE systems. However, such distance clue is significantly influenced by the room's acoustic characteristics, such as dimension and reverberation time, making it challenging for TSE systems that rely solely on distance clues to generalize across a variety of different rooms. To solve this, we suggest providing room environmental information (room dimensions and reverberation time) for distance-based TSE for better generalization capabilities. Especially, we propose a distance and environment-based TSE model in the time-frequency (TF) domain with learnable distance and room embedding. Results on both simulated and real collected datasets demonstrate its feasibility. Demonstration materials are available at this https URL.
#### Mitigating Subgroup Disparities in Multi-Label Speech Emotion Recognition: A Pseudo-Labeling and Unsupervised Learning Approach
 - **Authors:** Yi-Cheng Lin, Huang-Cheng Chou, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2505.14449

 - **Pdf link:** https://arxiv.org/pdf/2505.14449

 - **Abstract**
 While subgroup disparities and performance bias are increasingly studied in computational research, fairness in categorical Speech Emotion Recognition (SER) remains underexplored. Existing methods often rely on explicit demographic labels, which are difficult to obtain due to privacy concerns. To address this limitation, we introduce an Implicit Demography Inference (IDI) module that leverages pseudo-labeling from a pre-trained model and unsupervised learning using k-means clustering to mitigate bias in SER. Our experiments show that pseudo-labeling IDI reduces subgroup disparities, improving fairness metrics by over 33% with less than a 3% decrease in SER accuracy. Also, the unsupervised IDI yields more than a 26% improvement in fairness metrics with a drop of less than 4% in SER performance. Further analyses reveal that the unsupervised IDI consistently mitigates race and age disparities, demonstrating its potential in scenarios where explicit demographic information is unavailable.
#### FlowTSE: Target Speaker Extraction with Flow Matching
 - **Authors:** Aviv Navon, Aviv Shamsian, Yael Segal-Feldman, Neta Glazer, Gil Hetz, Joseph Keshet
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2505.14465

 - **Pdf link:** https://arxiv.org/pdf/2505.14465

 - **Abstract**
 Target speaker extraction (TSE) aims to isolate a specific speaker's speech from a mixture using speaker enrollment as a reference. While most existing approaches are discriminative, recent generative methods for TSE achieve strong results. However, generative methods for TSE remain underexplored, with most existing approaches relying on complex pipelines and pretrained components, leading to computational overhead. In this work, we present FlowTSE, a simple yet effective TSE approach based on conditional flow matching. Our model receives an enrollment audio sample and a mixed speech signal, both represented as mel-spectrograms, with the objective of extracting the target speaker's clean speech. Furthermore, for tasks where phase reconstruction is crucial, we propose a novel vocoder conditioned on the complex STFT of the mixed signal, enabling improved phase estimation. Experimental results on standard TSE benchmarks show that FlowTSE matches or outperforms strong baselines.
#### Teaching Audio-Aware Large Language Models What Does Not Hear: Mitigating Hallucinations through Synthesized Negative Samples
 - **Authors:** Chun-Yi Kuan, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.14518

 - **Pdf link:** https://arxiv.org/pdf/2505.14518

 - **Abstract**
 Recent advancements in audio-aware large language models (ALLMs) enable them to process and understand audio inputs. However, these models often hallucinate non-existent sound events, reducing their reliability in real-world applications. To address this, we propose LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method that enhances ALLMs' ability to distinguish between present and absent sounds using synthesized data from the backbone LLM. Unlike prior approaches, our method requires no modification to LLM parameters and efficiently integrates audio representations via a lightweight adapter. Experiments show that LISTEN effectively mitigates hallucinations while maintaining impressive performance on existing audio question and reasoning benchmarks. At the same time, it is more efficient in both data and computation.
#### AdaKWS: Towards Robust Keyword Spotting with Test-Time Adaptation
 - **Authors:** Yang Xiao, Tianyi Peng, Yanghao Zhou, Rohan Kumar Das
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.14600

 - **Pdf link:** https://arxiv.org/pdf/2505.14600

 - **Abstract**
 Spoken keyword spotting (KWS) aims to identify keywords in audio for wide applications, especially on edge devices. Current small-footprint KWS systems focus on efficient model designs. However, their inference performance can decline in unseen environments or noisy backgrounds. Test-time adaptation (TTA) helps models adapt to test samples without needing the original training data. In this study, we present AdaKWS, the first TTA method for robust KWS to the best of our knowledge. Specifically, 1) We initially optimize the model's confidence by selecting reliable samples based on prediction entropy minimization and adjusting the normalization statistics in each batch. 2) We introduce pseudo-keyword consistency (PKC) to identify critical, reliable features without overfitting to noise. Our experiments show that AdaKWS outperforms other methods across various conditions, including Gaussian noise and real-scenario noises. The code will be released in due course.
#### Listen, Analyze, and Adapt to Learn New Attacks: An Exemplar-Free Class Incremental Learning Method for Audio Deepfake Source Tracing
 - **Authors:** Yang Xiao, Rohan Kumar Das
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.14601

 - **Pdf link:** https://arxiv.org/pdf/2505.14601

 - **Abstract**
 As deepfake speech becomes common and hard to detect, it is vital to trace its source. Recent work on audio deepfake source tracing (ST) aims to find the origins of synthetic or manipulated speech. However, ST models must adapt to learn new deepfake attacks while retaining knowledge of the previous ones. A major challenge is catastrophic forgetting, where models lose the ability to recognize previously learned attacks. Some continual learning methods help with deepfake detection, but multi-class tasks such as ST introduce additional challenges as the number of classes grows. To address this, we propose an analytic class incremental learning method called AnaST. When new attacks appear, the feature extractor remains fixed, and the classifier is updated with a closed-form analytical solution in one epoch. This approach ensures data privacy, optimizes memory usage, and is suitable for online training. The experiments carried out in this work show that our method outperforms the baselines.
#### Score-Based Training for Energy-Based TTS Models
 - **Authors:** Wanli Sun, Anton Ragni
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.13771

 - **Pdf link:** https://arxiv.org/pdf/2505.13771

 - **Abstract**
 Noise contrastive estimation (NCE) is a popular method for training energy-based models (EBM) with intractable normalisation terms. The key idea of NCE is to learn by comparing unnormalised log-likelihoods of the reference and noisy samples, thus avoiding explicitly computing normalisation terms. However, NCE critically relies on the quality of noisy samples. Recently, sliced score matching (SSM) has been popularised by closely related diffusion models (DM). Unlike NCE, SSM learns a gradient of log-likelihood, or score, by learning distribution of its projections on randomly chosen directions. However, both NCE and SSM disregard the form of log-likelihood function, which is problematic given that EBMs and DMs make use of first-order optimisation during inference. This paper proposes a new criterion that learns scores more suitable for first-order schemes. Experiments contrasts these approaches for training EBMs.
#### ClapFM-EVC: High-Fidelity and Flexible Emotional Voice Conversion with Dual Control from Natural Language and Speech
 - **Authors:** Yu Pan, Yanni Hu, Yuguang Yang, Jixun Yao, Jianhao Ye, Hongbin Zhou, Lei Ma, Jianjun Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.13805

 - **Pdf link:** https://arxiv.org/pdf/2505.13805

 - **Abstract**
 Despite great advances, achieving high-fidelity emotional voice conversion (EVC) with flexible and interpretable control remains challenging. This paper introduces ClapFM-EVC, a novel EVC framework capable of generating high-quality converted speech driven by natural language prompts or reference speech with adjustable emotion intensity. We first propose EVC-CLAP, an emotional contrastive language-audio pre-training model, guided by natural language prompts and categorical labels, to extract and align fine-grained emotional elements across speech and text modalities. Then, a FuEncoder with an adaptive intensity gate is presented to seamless fuse emotional features with Phonetic PosteriorGrams from a pre-trained ASR model. To further improve emotion expressiveness and speech naturalness, we propose a flow matching model conditioned on these captured features to reconstruct Mel-spectrogram of source speech. Subjective and objective evaluations validate the effectiveness of ClapFM-EVC.
#### Forensic deepfake audio detection using segmental speech features
 - **Authors:** Tianle Yang, Chengzhe Sun, Siwei Lyu, Phil Rose
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.13847

 - **Pdf link:** https://arxiv.org/pdf/2505.13847

 - **Abstract**
 This study explores the potential of using acoustic features of segmental speech sounds to detect deepfake audio. These features are highly interpretable because of their close relationship with human articulatory processes and are expected to be more difficult for deepfake models to replicate. The results demonstrate that certain segmental features commonly used in forensic voice comparison are effective in identifying deep-fakes, whereas some global features provide little value. These findings underscore the need to approach audio deepfake detection differently for forensic voice comparison and offer a new perspective on leveraging segmental features for this purpose.
#### BiCrossMamba-ST: Speech Deepfake Detection with Bidirectional Mamba Spectro-Temporal Cross-Attention
 - **Authors:** Yassine El Kheir, Tim Polzehl, Sebastian Möller
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.13930

 - **Pdf link:** https://arxiv.org/pdf/2505.13930

 - **Abstract**
 We propose BiCrossMamba-ST, a robust framework for speech deepfake detection that leverages a dual-branch spectro-temporal architecture powered by bidirectional Mamba blocks and mutual cross-attention. By processing spectral sub-bands and temporal intervals separately and then integrating their representations, BiCrossMamba-ST effectively captures the subtle cues of synthetic speech. In addition, our proposed framework leverages a convolution-based 2D attention map to focus on specific spectro-temporal regions, enabling robust deepfake detection. Operating directly on raw features, BiCrossMamba-ST achieves significant performance improvements, a 67.74% and 26.3% relative gain over state-of-the-art AASIST on ASVSpoof LA21 and ASVSpoof DF21 benchmarks, respectively, and a 6.80% improvement over RawBMamba on ASVSpoof DF21. Code and models will be made publicly available.
#### The Multimodal Information Based Speech Processing (MISP) 2025 Challenge: Audio-Visual Diarization and Recognition
 - **Authors:** Ming Gao, Shilong Wu, Hang Chen, Jun Du, Chin-Hui Lee, Shinji Watanabe, Jingdong Chen, Siniscalchi Sabato Marco, Odette Scharenborg
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.13971

 - **Pdf link:** https://arxiv.org/pdf/2505.13971

 - **Abstract**
 Meetings are a valuable yet challenging scenario for speech applications due to complex acoustic conditions. This paper summarizes the outcomes of the MISP 2025 Challenge, hosted at Interspeech 2025, which focuses on multi-modal, multi-device meeting transcription by incorporating video modality alongside audio. The tasks include Audio-Visual Speaker Diarization (AVSD), Audio-Visual Speech Recognition (AVSR), and Audio-Visual Diarization and Recognition (AVDR). We present the challenge's objectives, tasks, dataset, baseline systems, and solutions proposed by participants. The best-performing systems achieved significant improvements over the baseline: the top AVSD model achieved a Diarization Error Rate (DER) of 8.09%, improving by 7.43%; the top AVSR system achieved a Character Error Rate (CER) of 9.48%, improving by 10.62%; and the best AVDR system achieved a concatenated minimum-permutation Character Error Rate (cpCER) of 11.56%, improving by 72.49%.
#### Bridging Speech Emotion Recognition and Personality: Dataset and Temporal Interaction Condition Network
 - **Authors:** Yuan Gao, Hao Shi, Yahui Fu, Chenhui Chu, Tatsuya Kawahara
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.13978

 - **Pdf link:** https://arxiv.org/pdf/2505.13978

 - **Abstract**
 This study investigates the interaction between personality traits and emotional expression, exploring how personality information can improve speech emotion recognition (SER). We collected personality annotation for the IEMOCAP dataset, and the statistical analysis identified significant correlations between personality traits and emotional expressions. To extract finegrained personality features, we propose a temporal interaction condition network (TICN), in which personality features are integrated with Hubert-based acoustic features for SER. Experiments show that incorporating ground-truth personality traits significantly enhances valence recognition, improving the concordance correlation coefficient (CCC) from 0.698 to 0.785 compared to the baseline without personality information. For practical applications in dialogue systems where personality information about the user is unavailable, we develop a front-end module of automatic personality recognition. Using these automatically predicted traits as inputs to our proposed TICN model, we achieve a CCC of 0.776 for valence recognition, representing an 11.17% relative improvement over the baseline. These findings confirm the effectiveness of personality-aware SER and provide a solid foundation for further exploration in personality-aware speech processing applications.
#### Combining Deterministic Enhanced Conditions with Dual-Streaming Encoding for Diffusion-Based Speech Enhancement
 - **Authors:** Hao Shi, Xugang Lu, Kazuki Shimada, Tatsuya Kawahara
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.13983

 - **Pdf link:** https://arxiv.org/pdf/2505.13983

 - **Abstract**
 Diffusion-based speech enhancement (SE) models need to incorporate correct prior knowledge as reliable conditions to generate accurate predictions. However, providing reliable conditions using noisy features is challenging. One solution is to use features enhanced by deterministic methods as conditions. However, the information distortion and loss caused by deterministic methods might affect the diffusion process. In this paper, we first investigate the effects of using different deterministic SE models as conditions for diffusion. We validate two conditions depending on whether the noisy feature was used as part of the condition: one using only the deterministic feature (deterministic-only), and the other using both deterministic and noisy features (deterministic-noisy). Preliminary investigation found that using deterministic enhanced conditions improves hearing experiences on real data, while the choice between using deterministic-only or deterministic-noisy conditions depends on the deterministic models. Based on these findings, we propose a dual-streaming encoding Repair-Diffusion Model for SE (DERDM-SE) to more effectively utilize both conditions. Moreover, we found that fine-grained deterministic models have greater potential in objective evaluation metrics, while UNet-based deterministic models provide more stable diffusion performance. Therefore, in the DERDM-SE, we propose a deterministic model that combines coarse- and fine-grained processing. Experimental results on CHiME4 show that the proposed models effectively leverage deterministic models to achieve better SE evaluation scores, along with more stable performance compared to other diffusion-based SE models.
#### Recreating Neural Activity During Speech Production with Language and Speech Model Embeddings
 - **Authors:** Owais Mujtaba Khanday, Pablo Rodroguez San Esteban, Zubair Ahmad Lone, Marc Ouellet, Jose Andres Gonzalez Lopez
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14074

 - **Pdf link:** https://arxiv.org/pdf/2505.14074

 - **Abstract**
 Understanding how neural activity encodes speech and language production is a fundamental challenge in neuroscience and artificial intelligence. This study investigates whether embeddings from large-scale, self-supervised language and speech models can effectively reconstruct neural activity recordings captured during speech production. We leverage pre-trained embeddings from deep learning models trained on linguistic and acoustic data to represent high-level speech features and map them onto neural signals. We analyze the extent to which these embeddings preserve the spatio-temporal dynamics of brain activity. We evaluate reconstructed neural signals against ground truth recordings using correlation metrics and signal reconstruction quality assessments. The results indicate that neural activity can be effectively reconstructed using embeddings from large language and speech models across all study participants, yielding Pearson correlation coefficients ranging from 0.79 to 0.99.
#### AudioJailbreak: Jailbreak Attacks against End-to-End Large Audio-Language Models
 - **Authors:** Guangke Chen, Fu Song, Zhe Zhao, Xiaojun Jia, Yang Liu, Yanchen Qiao, Weizhe Zhang
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14103

 - **Pdf link:** https://arxiv.org/pdf/2505.14103

 - **Abstract**
 Jailbreak attacks to Large audio-language models (LALMs) are studied recently, but they achieve suboptimal effectiveness, applicability, and practicability, particularly, assuming that the adversary can fully manipulate user prompts. In this work, we first conduct an extensive experiment showing that advanced text jailbreak attacks cannot be easily ported to end-to-end LALMs via text-to speech (TTS) techniques. We then propose AudioJailbreak, a novel audio jailbreak attack, featuring (1) asynchrony: the jailbreak audio does not need to align with user prompts in the time axis by crafting suffixal jailbreak audios; (2) universality: a single jailbreak perturbation is effective for different prompts by incorporating multiple prompts into perturbation generation; (3) stealthiness: the malicious intent of jailbreak audios will not raise the awareness of victims by proposing various intent concealment strategies; and (4) over-the-air robustness: the jailbreak audios remain effective when being played over the air by incorporating the reverberation distortion effect with room impulse response into the generation of the perturbations. In contrast, all prior audio jailbreak attacks cannot offer asynchrony, universality, stealthiness, or over-the-air robustness. Moreover, AudioJailbreak is also applicable to the adversary who cannot fully manipulate user prompts, thus has a much broader attack scenario. Extensive experiments with thus far the most LALMs demonstrate the high effectiveness of AudioJailbreak. We highlight that our work peeks into the security implications of audio jailbreak attacks against LALMs, and realistically fosters improving their security robustness. The implementation and audio samples are available at our website this https URL.
#### Source Verification for Speech Deepfakes
 - **Authors:** Viola Negroni, Davide Salvi, Paolo Bestagini, Stefano Tubaro
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14188

 - **Pdf link:** https://arxiv.org/pdf/2505.14188

 - **Abstract**
 With the proliferation of speech deepfake generators, it becomes crucial not only to assess the authenticity of synthetic audio but also to trace its origin. While source attribution models attempt to address this challenge, they often struggle in open-set conditions against unseen generators. In this paper, we introduce the source verification task, which, inspired by speaker verification, determines whether a test track was produced using the same model as a set of reference signals. Our approach leverages embeddings from a classifier trained for source attribution, computing distance scores between tracks to assess whether they originate from the same source. We evaluate multiple models across diverse scenarios, analyzing the impact of speaker diversity, language mismatch, and post-processing operations. This work provides the first exploration of source verification, highlighting its potential and vulnerabilities, and offers insights for real-world forensic applications.
#### MatchDance: Collaborative Mamba-Transformer Architecture Matching for High-Quality 3D Dance Synthesis
 - **Authors:** Kaixing Yang, Xulong Tang, Yuxuan Hu, Jiahao Yang, Hongyan Liu, Qinnan Zhang, Jun He, Zhaoxin Fan
 - **Subjects:** Subjects:
Sound (cs.SD); Graphics (cs.GR); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14222

 - **Pdf link:** https://arxiv.org/pdf/2505.14222

 - **Abstract**
 Music-to-dance generation represents a challenging yet pivotal task at the intersection of choreography, virtual reality, and creative content generation. Despite its significance, existing methods face substantial limitation in achieving choreographic consistency. To address the challenge, we propose MatchDance, a novel framework for music-to-dance generation that constructs a latent representation to enhance choreographic consistency. MatchDance employs a two-stage design: (1) a Kinematic-Dynamic-based Quantization Stage (KDQS), which encodes dance motions into a latent representation by Finite Scalar Quantization (FSQ) with kinematic-dynamic constraints and reconstructs them with high fidelity, and (2) a Hybrid Music-to-Dance Generation Stage(HMDGS), which uses a Mamba-Transformer hybrid architecture to map music into the latent representation, followed by the KDQS decoder to generate 3D dance motions. Additionally, a music-dance retrieval framework and comprehensive metrics are introduced for evaluation. Extensive experiments on the FineDance dataset demonstrate state-of-the-art performance. Code will be released upon acceptance.
#### AquaSignal: An Integrated Framework for Robust Underwater Acoustic Analysis
 - **Authors:** Eirini Panteli, Paulo E. Santos, Nabil Humphrey
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14285

 - **Pdf link:** https://arxiv.org/pdf/2505.14285

 - **Abstract**
 This paper presents AquaSignal, a modular and scalable pipeline for preprocessing, denoising, classification, and novelty detection of underwater acoustic signals. Designed to operate effectively in noisy and dynamic marine environments, AquaSignal integrates state-of-the-art deep learning architectures to enhance the reliability and accuracy of acoustic signal analysis. The system is evaluated on a combined dataset from the Deepship and Ocean Networks Canada (ONC) benchmarks, providing a diverse set of real-world underwater scenarios. AquaSignal employs a U-Net architecture for denoising, a ResNet18 convolutional neural network for classifying known acoustic events, and an AutoEncoder-based model for unsupervised detection of novel or anomalous signals. To our knowledge, this is the first comprehensive study to apply and evaluate this combination of techniques on maritime vessel acoustic data. Experimental results show that AquaSignal improves signal clarity and task performance, achieving 71% classification accuracy and 91% accuracy in novelty detection. Despite slightly lower classification performance compared to some state-of-the-art models, differences in data partitioning strategies limit direct comparisons. Overall, AquaSignal demonstrates strong potential for real-time underwater acoustic monitoring in scientific, environmental, and maritime domains.
#### Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs
 - **Authors:** Rao Ma, Mengjie Qian, Vyas Raina, Mark Gales, Kate Knill
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14286

 - **Pdf link:** https://arxiv.org/pdf/2505.14286

 - **Abstract**
 The combination of pre-trained speech encoders with large language models has enabled the development of speech LLMs that can handle a wide range of spoken language processing tasks. While these models are powerful and flexible, this very flexibility may make them more vulnerable to adversarial attacks. To examine the extent of this problem, in this work we investigate universal acoustic adversarial attacks on speech LLMs. Here a fixed, universal, adversarial audio segment is prepended to the original input audio. We initially investigate attacks that cause the model to either produce no output or to perform a modified task overriding the original prompt. We then extend the nature of the attack to be selective so that it activates only when specific input attributes, such as a speaker gender or spoken language, are present. Inputs without the targeted attribute should be unaffected, allowing fine-grained control over the model outputs. Our findings reveal critical vulnerabilities in Qwen2-Audio and Granite-Speech and suggest that similar speech LLMs may be susceptible to universal adversarial attacks. This highlights the need for more robust training strategies and improved resistance to adversarial attacks.
#### FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation
 - **Authors:** Yutong Liu, Ziyue Zhang, Ban Ma-bao, Yuqing Cai, Yongbin Yu, Renzeng Duojie, Xiangxiang Wang, Fan Gao, Cheng Huang, Nyima Tashi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14351

 - **Pdf link:** https://arxiv.org/pdf/2505.14351

 - **Abstract**
 Tibetan is a low-resource language with minimal parallel speech corpora spanning its three major dialects-Ü-Tsang, Amdo, and Kham-limiting progress in speech modeling. To address this issue, we propose FMSD-TTS, a few-shot, multi-speaker, multi-dialect text-to-speech framework that synthesizes parallel dialectal speech from limited reference audio and explicit dialect labels. Our method features a novel speaker-dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects while preserving speaker identity. Extensive objective and subjective evaluations demonstrate that FMSD-TTS significantly outperforms baselines in both dialectal expressiveness and speaker similarity. We further validate the quality and utility of the synthesized speech through a challenging speech-to-speech dialect conversion task. Our contributions include: (1) a novel few-shot TTS system tailored for Tibetan multi-dialect speech synthesis, (2) the public release of a large-scale synthetic Tibetan speech corpus generated by FMSD-TTS, and (3) an open-source evaluation toolkit for standardized assessment of speaker similarity, dialect consistency, and audio quality.
#### PersonaTAB: Predicting Personality Traits using Textual, Acoustic, and Behavioral Cues in Fully-Duplex Speech Dialogs
 - **Authors:** Sho Inoue, Shai Wang, Haizhou Li
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14356

 - **Pdf link:** https://arxiv.org/pdf/2505.14356

 - **Abstract**
 Despite significant progress in neural spoken dialog systems, personality-aware conversation agents -- capable of adapting behavior based on personalities -- remain underexplored due to the absence of personality annotations in speech datasets. We propose a pipeline that preprocesses raw audio recordings to create a dialogue dataset annotated with timestamps, response types, and emotion/sentiment labels. We employ an automatic speech recognition (ASR) system to extract transcripts and timestamps, then generate conversation-level annotations. Leveraging these annotations, we design a system that employs large language models to predict conversational personality. Human evaluators were engaged to identify conversational characteristics and assign personality labels. Our analysis demonstrates that the proposed system achieves stronger alignment with human judgments compared to existing approaches.
#### S2SBench: A Benchmark for Quantifying Intelligence Degradation in Speech-to-Speech Large Language Models
 - **Authors:** Yuanbo Fang, Haoze Sun, Jun Liu, Tao Zhang, Zenan Zhou, Weipeng Chen, Xiaofen Xing, Xiangmin Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14438

 - **Pdf link:** https://arxiv.org/pdf/2505.14438

 - **Abstract**
 End-to-end speech large language models ((LLMs)) extend the capabilities of text-based models to directly process and generate audio tokens. However, this often leads to a decline in reasoning and generation performance compared to text input, a phenomenon referred to as intelligence degradation. To systematically evaluate this gap, we propose S2SBench, a benchmark designed to quantify performance degradation in Speech LLMs. It includes diagnostic datasets targeting sentence continuation and commonsense reasoning under audio input. We further introduce a pairwise evaluation protocol based on perplexity differences between plausible and implausible samples to measure degradation relative to text input. We apply S2SBench to analyze the training process of Baichuan-Audio, which further demonstrates the benchmark's effectiveness. All datasets and evaluation code are available at this https URL.
#### PAST: Phonetic-Acoustic Speech Tokenizer
 - **Authors:** Nadav Har-Tuv, Or Tal, Yossi Adi
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14470

 - **Pdf link:** https://arxiv.org/pdf/2505.14470

 - **Abstract**
 We present PAST, a novel end-to-end framework that jointly models phonetic information alongside signal reconstruction, eliminating the need for external pretrained models. Unlike previous approaches that rely on pretrained self-supervised models, PAST employs supervised phonetic data, directly integrating domain knowledge into the tokenization process via auxiliary tasks. Additionally, we introduce a streamable, causal variant of PAST, enabling real-time speech applications. Results demonstrate that PAST surpasses existing evaluated baseline tokenizers across common evaluation metrics, including phonetic representation and speech reconstruction. Notably, PAST also achieves superior performance when serving as a speech representation for speech language models, further highlighting its effectiveness as a foundation for spoken language generation. To foster further research, we release the full implementation. For code, model checkpoints, and samples see: this https URL
#### Representation Learning for Semantic Alignment of Language, Audio, and Visual Modalities
 - **Authors:** Parthasaarathy Sudarsanam, Irene Martín-Morató, Tuomas Virtanen
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14562

 - **Pdf link:** https://arxiv.org/pdf/2505.14562

 - **Abstract**
 This paper proposes a single-stage training approach that semantically aligns three modalities - audio, visual, and text using a contrastive learning framework. Contrastive training has gained prominence for multimodal alignment, utilizing large-scale unlabeled data to learn shared representations. Existing deep learning approach for trimodal alignment involves two-stages, that separately align visual-text and audio-text modalities. This approach suffers from mismatched data distributions, resulting in suboptimal alignment. Leveraging the AVCaps dataset, which provides audio, visual and audio-visual captions for video clips, our method jointly optimizes the representation of all the modalities using contrastive training. Our results demonstrate that the single-stage approach outperforms the two-stage method, achieving a two-fold improvement in audio based visual retrieval, highlighting the advantages of unified multimodal representation learning.
#### Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits
 - **Authors:** Tiantian Feng, Jihwan Lee, Anfeng Xu, Yoonjeong Lee, Thanathai Lertpetchpun, Xuan Shi, Helin Wang, Thomas Thebaud, Laureano Moro-Velazquez, Dani Byrd, Najim Dehak, Shrikanth Narayanan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14648

 - **Pdf link:** https://arxiv.org/pdf/2505.14648

 - **Abstract**
 We introduce Vox-Profile, a comprehensive benchmark to characterize rich speaker and speech traits using speech foundation models. Unlike existing works that focus on a single dimension of speaker traits, Vox-Profile provides holistic and multi-dimensional profiles that reflect both static speaker traits (e.g., age, sex, accent) and dynamic speech properties (e.g., emotion, speech flow). This benchmark is grounded in speech science and linguistics, developed with domain experts to accurately index speaker and speech characteristics. We report benchmark experiments using over 15 publicly available speech datasets and several widely used speech foundation models that target various static and dynamic speaker and speech properties. In addition to benchmark experiments, we showcase several downstream applications supported by Vox-Profile. First, we show that Vox-Profile can augment existing speech recognition datasets to analyze ASR performance variability. Vox-Profile is also used as a tool to evaluate the performance of speech generation systems. Finally, we assess the quality of our automated profiles through comparison with human evaluation and show convergent validity. Vox-Profile is publicly available at: this https URL.


by Zyzzyva0381 (Windy). 


2025-05-21
