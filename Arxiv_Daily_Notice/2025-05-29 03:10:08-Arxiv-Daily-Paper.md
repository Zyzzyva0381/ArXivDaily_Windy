# Showing new listings for Thursday, 29 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 28papers 
#### VietASR: Achieving Industry-level Vietnamese ASR with 50-hour labeled data and Large-Scale Speech Pretraining
 - **Authors:** Jianheng Zhuo, Yifan Yang, Yiwen Shao, Yong Xu, Dong Yu, Kai Yu, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.21527

 - **Pdf link:** https://arxiv.org/pdf/2505.21527

 - **Abstract**
 Automatic speech recognition (ASR) has made remarkable progress but heavily relies on large-scale labeled data, which is scarce for low-resource languages like Vietnamese. While existing systems such as Whisper, USM, and MMS achieve promising performance, their efficacy remains inadequate in terms of training costs, latency, and accessibility. To address these issues, we propose VietASR, a novel ASR training pipeline that leverages vast amounts of unlabeled data and a small set of labeled data. Through multi-iteration ASR-biased self-supervised learning on a large-scale unlabeled dataset, VietASR offers a cost-effective and practical solution for enhancing ASR performance. Experiments demonstrate that pre-training on 70,000-hour unlabeled data and fine-tuning on merely 50-hour labeled data yield a lightweight but powerful ASR model. It outperforms Whisper Large-v3 and commercial ASR systems on real-world data. Our code and models will be open-sourced to facilitate research in low-resource ASR.
#### WhisperD: Dementia Speech Recognition and Filler Word Detection with Whisper
 - **Authors:** Emmanuel Akinrintoyo, Nadine Abdelhalim, Nicole Salomons
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.21551

 - **Pdf link:** https://arxiv.org/pdf/2505.21551

 - **Abstract**
 Whisper fails to correctly transcribe dementia speech because persons with dementia (PwDs) often exhibit irregular speech patterns and disfluencies such as pauses, repetitions, and fragmented sentences. It was trained on standard speech and may have had little or no exposure to dementia-affected speech. However, correct transcription is vital for dementia speech for cost-effective diagnosis and the development of assistive technology. In this work, we fine-tune Whisper with the open-source dementia speech dataset (DementiaBank) and our in-house dataset to improve its word error rate (WER). The fine-tuning also includes filler words to ascertain the filler inclusion rate (FIR) and F1 score. The fine-tuned models significantly outperformed the off-the-shelf models. The medium-sized model achieved a WER of 0.24, outperforming previous work. Similarly, there was a notable generalisability to unseen data and speech patterns.
#### Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection
 - **Authors:** Jinming Zhang, Xuanru Zhou, Jiachen Lian, Shuhe Li, William Li, Zoe Ezzes, Rian Bogley, Lisa Wauters, Zachary Miller, Jet Vonk, Brittany Morin, Maria Gorno-Tempini, Gopala Anumanchipalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.22029

 - **Pdf link:** https://arxiv.org/pdf/2505.22029

 - **Abstract**
 Speech dysfluency detection is crucial for clinical diagnosis and language assessment, but existing methods are limited by the scarcity of high-quality annotated data. Although recent advances in TTS model have enabled synthetic dysfluency generation, existing synthetic datasets suffer from unnatural prosody and limited contextual diversity. To address these limitations, we propose LLM-Dys -- the most comprehensive dysfluent speech corpus with LLM-enhanced dysfluency simulation. This dataset captures 11 dysfluency categories spanning both word and phoneme levels. Building upon this resource, we improve an end-to-end dysfluency detection framework. Experimental validation demonstrates state-of-the-art performance. All data, models, and code are open-sourced at this https URL.
#### ARiSE: Auto-Regressive Multi-Channel Speech Enhancement
 - **Authors:** Pengjie Shen, Xueliang Zhang, Zhong-Qiu Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22051

 - **Pdf link:** https://arxiv.org/pdf/2505.22051

 - **Abstract**
 We propose ARiSE, an auto-regressive algorithm for multi-channel speech enhancement. ARiSE improves existing deep neural network (DNN) based frame-online multi-channel speech enhancement models by introducing auto-regressive connections, where the estimated target speech at previous frames is leveraged as extra input features to help the DNN estimate the target speech at the current frame. The extra input features can be derived from (a) the estimated target speech in previous frames; and (b) a beamformed mixture with the beamformer computed based on the previous estimated target speech. On the other hand, naively training the DNN in an auto-regressive manner is very slow. To deal with this, we propose a parallel training mechanism to speed up the training. Evaluation results in noisy-reverberant conditions show the effectiveness and potential of the proposed algorithms.
#### Evaluation of LLMs in Speech is Often Flawed: Test Set Contamination in Large Language Models for Speech Recognition
 - **Authors:** Yuan Tseng, Titouan Parcollet, Rogier van Dalen, Shucong Zhang, Sourav Bhattacharya
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2505.22251

 - **Pdf link:** https://arxiv.org/pdf/2505.22251

 - **Abstract**
 Recent work suggests that large language models (LLMs) can improve performance of speech tasks compared to existing systems. To support their claims, results on LibriSpeech and Common Voice are often quoted. However, this work finds that a substantial amount of the LibriSpeech and Common Voice evaluation sets appear in public LLM pretraining corpora. This calls into question the reliability of findings drawn from these two datasets. To measure the impact of contamination, LLMs trained with or without contamination are compared, showing that a contaminated LLM is more likely to generate test sentences it has seen during training. Speech recognisers using contaminated LLMs shows only subtle differences in error rates, but assigns significantly higher probabilities to transcriptions seen during training. Results show that LLM outputs can be biased by tiny amounts of data contamination, highlighting the importance of evaluating LLM-based speech systems with held-out data.
#### Articulatory modeling of the S-shaped F2 trajectories observed in Öhman's spectrographic analysis of VCV syllables
 - **Authors:** Frédéric Berthommier
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.22455

 - **Pdf link:** https://arxiv.org/pdf/2505.22455

 - **Abstract**
 The synthesis of Ohman's VCV sequences with intervocalic plosive consonants was first achieved 30 years ago using the DRM model. However, this approach remains primarily acoustic and lacks articulatory constraints. In this study, the same 75 VCVs are analyzed, but generated with the Maeda model, using trajectory planning that differentiates vowel-to-vowel transitions from consonantal influences. Synthetic data exhibit similar characteristics to Ohman's sequences, including the presence of S-shaped F2 trajectories. Furthermore, locus equations (LEs) for F2 and F3 are computed from synthetic CV data to investigate their underlying determinism, leading to a reassessment of conventional interpretations. The findings indicate that, although articulatory planning is structured separately for vowel and consonant groups, S-shaped F2 trajectories emerge from a composite mechanism governed by the coordinated synergy of all articulators.
#### VoiceMark: Zero-Shot Voice Cloning-Resistant Watermarking Approach Leveraging Speaker-Specific Latents
 - **Authors:** Haiyun Li, Zhiyong Wu, Xiaofeng Xie, Jingran Xie, Yaoxun Xu, Hanyang Peng
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.21568

 - **Pdf link:** https://arxiv.org/pdf/2505.21568

 - **Abstract**
 Voice cloning (VC)-resistant watermarking is an emerging technique for tracing and preventing unauthorized cloning. Existing methods effectively trace traditional VC models by training them on watermarked audio but fail in zero-shot VC scenarios, where models synthesize audio from an audio prompt without training. To address this, we propose VoiceMark, the first zero-shot VC-resistant watermarking method that leverages speaker-specific latents as the watermark carrier, allowing the watermark to transfer through the zero-shot VC process into the synthesized audio. Additionally, we introduce VC-simulated augmentations and VAD-based loss to enhance robustness against distortions. Experiments on multiple zero-shot VC models demonstrate that VoiceMark achieves over 95% accuracy in watermark detection after zero-shot VC synthesis, significantly outperforming existing methods, which only reach around 50%. See our code and demos at: this https URL
#### Loquacious Set: 25,000 Hours of Transcribed and Diverse English Speech Recognition Data for Research and Commercial Use
 - **Authors:** Titouan Parcollet, Yuan Tseng, Shucong Zhang, Rogier van Dalen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.21578

 - **Pdf link:** https://arxiv.org/pdf/2505.21578

 - **Abstract**
 Automatic speech recognition (ASR) research is driven by the availability of common datasets between industrial researchers and academics, encouraging comparisons and evaluations. LibriSpeech, despite its long success as an ASR benchmark, is now limited by its size and focus on clean, read speech, leading to near-zero word error rates. More recent datasets, including MOSEL, YODAS, Gigaspeech, OWSM, Libriheavy or People's Speech suffer from major limitations including licenses that researchers in the industry cannot use, unreliable transcriptions, incorrect audio data, or the lack of evaluation sets. This work presents the Loquacious Set, a 25,000-hour curated collection of commercially usable English speech. Featuring hundreds of thousands of speakers with diverse accents and a wide range of speech types (read, spontaneous, talks, clean, noisy), the Loquacious Set is designed to work for academics and researchers in the industry to build ASR systems in real-world scenarios.
#### An Investigation on Speaker Augmentation for End-to-End Speaker Extraction
 - **Authors:** Zhenghai You, Zhenyu Zhou, Lantian Li, Dong Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.21805

 - **Pdf link:** https://arxiv.org/pdf/2505.21805

 - **Abstract**
 Target confusion, defined as occasional switching to non-target speakers, poses a key challenge for end-to-end speaker extraction (E2E-SE) systems. We argue that this problem is largely caused by the lack of generalizability and discrimination of the speaker embeddings, and introduce a simple yet effective speaker augmentation strategy to tackle the problem. Specifically, we propose a time-domain resampling and rescaling pipeline that alters speaker traits while preserving other speech properties. This generates a variety of pseudo-speakers to help establish a generalizable speaker embedding space, while the speaker-trait-specific augmentation creates hard samples that force the model to focus on genuine speaker characteristics. Experiments on WSJ0-2Mix and LibriMix show that our method mitigates the target confusion and improves extraction performance. Moreover, it can be combined with metric learning, another effective approach to address target confusion, leading to further gains.
#### Voice Quality Dimensions as Interpretable Primitives for Speaking Style for Atypical Speech and Affect
 - **Authors:** Jaya Narain, Vasudha Kowtha, Colin Lea, Lauren Tooley, Dianna Yee, Vikramjit Mitra, Zifang Huang, Miquel Espi Marques, Jon Huang, Carlos Avendano, Shirley Ren
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.21809

 - **Pdf link:** https://arxiv.org/pdf/2505.21809

 - **Abstract**
 Perceptual voice quality dimensions describe key characteristics of atypical speech and other speech modulations. Here we develop and evaluate voice quality models for seven voice and speech dimensions (intelligibility, imprecise consonants, harsh voice, naturalness, monoloudness, monopitch, and breathiness). Probes were trained on the public Speech Accessibility (SAP) project dataset with 11,184 samples from 434 speakers, using embeddings from frozen pre-trained models as features. We found that our probes had both strong performance and strong generalization across speech elicitation categories in the SAP dataset. We further validated zero-shot performance on additional datasets, encompassing unseen languages and tasks: Italian atypical speech, English atypical speech, and affective speech. The strong zero-shot performance and the interpretability of results across an array of evaluations suggests the utility of using voice quality dimensions in speaking style-related tasks.
#### Music Source Restoration
 - **Authors:** Yongyi Zang, Zheqi Dai, Mark D. Plumbley, Qiuqiang Kong
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.21827

 - **Pdf link:** https://arxiv.org/pdf/2505.21827

 - **Abstract**
 We introduce Music Source Restoration (MSR), a novel task addressing the gap between idealized source separation and real-world music production. Current Music Source Separation (MSS) approaches assume mixtures are simple sums of sources, ignoring signal degradations employed during music production like equalization, compression, and reverb. MSR models mixtures as degraded sums of individually degraded sources, with the goal of recovering original, undegraded signals. Due to the lack of data for MSR, we present RawStems, a dataset annotation of 578 songs with unprocessed source signals organized into 8 primary and 17 secondary instrument groups, totaling 354.13 hours. To the best of our knowledge, RawStems is the first dataset that contains unprocessed music stems with hierarchical categories. We consider spectral filtering, dynamic range compression, harmonic distortion, reverb and lossy codec as possible degradations, and establish U-Former as a baseline method, demonstrating the feasibility of MSR on our dataset. We release the RawStems dataset annotations, degradation simulation pipeline, training code and pre-trained models to be publicly available.
#### Leveraging LLM for Stuttering Speech: A Unified Architecture Bridging Recognition and Event Detection
 - **Authors:** Shangkun Huang, Jing Deng, Jintao Kang, Rong Zheng
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22005

 - **Pdf link:** https://arxiv.org/pdf/2505.22005

 - **Abstract**
 The performance bottleneck of Automatic Speech Recognition (ASR) in stuttering speech scenarios has limited its applicability in domains such as speech rehabilitation. This paper proposed an LLM-driven ASR-SED multi-task learning framework that jointly optimized the ASR and Stuttering Event Detection (SED) tasks. We proposed a dynamic interaction mechanism where the ASR branch leveraged CTC-generated soft prompts to assist LLM context modeling, while the SED branch output stutter embeddings to enhance LLM comprehension of stuttered speech. We incorporated contrastive learning to strengthen the discriminative power of stuttering acoustic features and applied Focal Loss to mitigate the long-tailed distribution in stuttering event categories. Evaluations on the AS-70 Mandarin stuttering dataset demonstrated that our framework reduced the ASR character error rate (CER) to 5.45% (-37.71% relative reduction) and achieved an average SED F1-score of 73.63% (+46.58% relative improvement).
#### Overlap-Adaptive Hybrid Speaker Diarization and ASR-Aware Observation Addition for MISP 2025 Challenge
 - **Authors:** Shangkun Huang, Yuxuan Du, Jingwen Yang, Dejun Zhang, Xupeng Jia, Jing Deng, Jintao Kang, Rong Zheng
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22013

 - **Pdf link:** https://arxiv.org/pdf/2505.22013

 - **Abstract**
 This paper presents the system developed to address the MISP 2025 Challenge. For the diarization system, we proposed a hybrid approach combining a WavLM end-to-end segmentation method with a traditional multi-module clustering technique to adaptively select the appropriate model for handling varying degrees of overlapping speech. For the automatic speech recognition (ASR) system, we proposed an ASR-aware observation addition method that compensates for the performance limitations of Guided Source Separation (GSS) under low signal-to-noise ratio conditions. Finally, we integrated the speaker diarization and ASR systems in a cascaded architecture to address Track 3. Our system achieved character error rates (CER) of 9.48% on Track 2 and concatenated minimum permutation character error rate (cpCER) of 11.56% on Track 3, ultimately securing first place in both tracks and thereby demonstrating the effectiveness of the proposed methods in real-world meeting scenarios.
#### RESOUND: Speech Reconstruction from Silent Videos via Acoustic-Semantic Decomposed Modeling
 - **Authors:** Long-Khanh Pham, Thanh V. T. Tran, Minh-Tan Pham, Van Nguyen
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22024

 - **Pdf link:** https://arxiv.org/pdf/2505.22024

 - **Abstract**
 Lip-to-speech (L2S) synthesis, which reconstructs speech from visual cues, faces challenges in accuracy and naturalness due to limited supervision in capturing linguistic content, accents, and prosody. In this paper, we propose RESOUND, a novel L2S system that generates intelligible and expressive speech from silent talking face videos. Leveraging source-filter theory, our method involves two components: an acoustic path to predict prosody and a semantic path to extract linguistic features. This separation simplifies learning, allowing independent optimization of each representation. Additionally, we enhance performance by integrating speech units, a proven unsupervised speech representation technique, into waveform generation alongside mel-spectrograms. This allows RESOUND to synthesize prosodic speech while preserving content and speaker identity. Experiments conducted on two standard L2S benchmarks confirm the effectiveness of the proposed method across various metrics.
#### Mitigating Audiovisual Mismatch in Visual-Guide Audio Captioning
 - **Authors:** Le Xu, Chenxing Li, Yong Ren, Yujie Chen, Yu Gu, Ruibo Fu, Shan Yang, Dong Yu
 - **Subjects:** Subjects:
Multimedia (cs.MM); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22045

 - **Pdf link:** https://arxiv.org/pdf/2505.22045

 - **Abstract**
 Current vision-guided audio captioning systems frequently fail to address audiovisual misalignment in real-world scenarios, such as dubbed content or off-screen sounds. To bridge this critical gap, we present an entropy-aware gated fusion framework that dynamically modulates visual information flow through cross-modal uncertainty quantification. Our novel approach employs attention entropy analysis in cross-attention layers to automatically identify and suppress misleading visual cues during modal fusion. Complementing this architecture, we develop a batch-wise audiovisual shuffling technique that generates synthetic mismatched training pairs, greatly enhancing model resilience against alignment noise. Evaluations on the AudioCaps benchmark demonstrate our system's superior performance over existing baselines, especially in mismatched modality scenarios. Furthermore, our solution demonstrates an approximately 6x improvement in inference speed compared to the baseline.
#### AudioGenie: A Training-Free Multi-Agent Framework for Diverse Multimodality-to-Multiaudio Generation
 - **Authors:** Yan Rong, Jinting Wang, Shan Yang, Guangzhi Lei, Li Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Multiagent Systems (cs.MA); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22053

 - **Pdf link:** https://arxiv.org/pdf/2505.22053

 - **Abstract**
 Multimodality-to-Multiaudio (MM2MA) generation faces significant challenges in synthesizing diverse and contextually aligned audio types (e.g., sound effects, speech, music, and songs) from multimodal inputs (e.g., video, text, images), owing to the scarcity of high-quality paired datasets and the lack of robust multi-task learning frameworks. Recently, multi-agent system shows great potential in tackling the above issues. However, directly applying it to MM2MA task presents three critical challenges: (1) inadequate fine-grained understanding of multimodal inputs (especially for video), (2) the inability of single models to handle diverse audio events, and (3) the absence of self-correction mechanisms for reliable outputs. To this end, we propose AudioGenie, a novel training-free multi-agent system featuring a dual-layer architecture with a generation team and a supervisor team. For the generation team, a fine-grained task decomposition and an adaptive Mixture-of-Experts (MoE) collaborative entity are designed for dynamic model selection, and a trial-and-error iterative refinement module is designed for self-correction. The supervisor team ensures temporal-spatial consistency and verifies outputs through feedback loops. Moreover, we build MA-Bench, the first benchmark for MM2MA tasks, comprising 198 annotated videos with multi-type audios. Experiments demonstrate that our AudioGenie outperforms state-of-the-art (SOTA) methods across 9 metrics in 8 tasks. User study further validate the effectiveness of the proposed method in terms of quality, accuracy, alignment, and aesthetic. The anonymous project website with samples can be found at this https URL.
#### Voice Adaptation for Swiss German
 - **Authors:** Samuel Stucki, Jan Deriu, Mark Cieliebak
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22054

 - **Pdf link:** https://arxiv.org/pdf/2505.22054

 - **Abstract**
 This work investigates the performance of Voice Adaptation models for Swiss German dialects, i.e., translating Standard German text to Swiss German dialect speech. For this, we preprocess a large dataset of Swiss podcasts, which we automatically transcribe and annotate with dialect classes, yielding approximately 5000 hours of weakly labeled training material. We fine-tune the XTTSv2 model on this dataset and show that it achieves good scores in human and automated evaluations and can correctly render the desired dialect. Our work shows a step towards adapting Voice Cloning technology to underrepresented languages. The resulting model achieves CMOS scores of up to -0.28 and SMOS scores of 3.8.
#### Weakly Supervised Data Refinement and Flexible Sequence Compression for Efficient Thai LLM-based ASR
 - **Authors:** Mingchen Shao, Xinfa Zhu, Chengyou Wang, Bingshen Mu, Hai Li, Ying Yan, Junhui Liu, Danming Xie, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22063

 - **Pdf link:** https://arxiv.org/pdf/2505.22063

 - **Abstract**
 Despite remarkable achievements, automatic speech recognition (ASR) in low-resource scenarios still faces two challenges: high-quality data scarcity and high computational demands. This paper proposes EThai-ASR, the first to apply large language models (LLMs) to Thai ASR and create an efficient LLM-based ASR system. EThai-ASR comprises a speech encoder, a connection module and a Thai LLM decoder. To address the data scarcity and obtain a powerful speech encoder, EThai-ASR introduces a self-evolving data refinement strategy to refine weak labels, yielding an enhanced speech encoder. Moreover, we propose a pluggable sequence compression module used in the connection module with three modes designed to reduce the sequence length, thus decreasing computational demands while maintaining decent performance. Extensive experiments demonstrate that EThai-ASR has achieved state-of-the-art accuracy in multiple datasets. We release our refined text transcripts to promote further research.
#### Delayed-KD: Delayed Knowledge Distillation based CTC for Low-Latency Streaming ASR
 - **Authors:** Longhao Li, Yangze Li, Hongfei Xue, Jie Liu, Shuai Fang, Kai Wang, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22069

 - **Pdf link:** https://arxiv.org/pdf/2505.22069

 - **Abstract**
 CTC-based streaming ASR has gained significant attention in real-world applications but faces two main challenges: accuracy degradation in small chunks and token emission latency. To mitigate these challenges, we propose Delayed-KD, which applies delayed knowledge distillation on CTC posterior probabilities from a non-streaming to a streaming model. Specifically, with a tiny chunk size, we introduce a Temporal Alignment Buffer (TAB) that defines a relative delay range compared to the non-streaming teacher model to align CTC outputs and mitigate non-blank token mismatches. Additionally, TAB enables fine-grained control over token emission delay. Experiments on 178-hour AISHELL-1 and 10,000-hour WenetSpeech Mandarin datasets show consistent superiority of Delayed-KD. Impressively, Delayed-KD at 40 ms latency achieves a lower character error rate (CER) of 5.42% on AISHELL-1, comparable to the competitive U2++ model running at 320 ms latency.
#### On-the-fly Routing for Zero-shot MoE Speaker Adaptation of Speech Foundation Models for Dysarthric Speech Recognition
 - **Authors:** Shujie HU, Xurong Xie, Mengzhe Geng, Jiajun Deng, Huimeng Wang, Guinan Li, Chengxi Deng, Tianzi Wang, Mingyu Cui, Helen Meng, Xunying Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22072

 - **Pdf link:** https://arxiv.org/pdf/2505.22072

 - **Abstract**
 This paper proposes a novel MoE-based speaker adaptation framework for foundation models based dysarthric speech recognition. This approach enables zero-shot adaptation and real-time processing while incorporating domain knowledge. Speech impairment severity and gender conditioned adapter experts are dynamically combined using on-the-fly predicted speaker-dependent routing parameters. KL-divergence is used to further enforce diversity among experts and their generalization to unseen speakers. Experimental results on the UASpeech corpus suggest that on-the-fly MoE-based adaptation produces statistically significant WER reductions of up to 1.34% absolute (6.36% relative) over the unadapted baseline HuBERT/WavLM models. Consistent WER reductions of up to 2.55% absolute (11.44% relative) and RTF speedups of up to 7 times are obtained over batch-mode adaptation across varying speaker-level data quantities. The lowest published WER of 16.35% (46.77% on very low intelligibility) is obtained.
#### Visual Cues Support Robust Turn-taking Prediction in Noise
 - **Authors:** Sam O'Connor Russell, Naomi Harte
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22088

 - **Pdf link:** https://arxiv.org/pdf/2505.22088

 - **Abstract**
 Accurate predictive turn-taking models (PTTMs) are essential for naturalistic human-robot interaction. However, little is known about their performance in noise. This study therefore explores PTTM performance in types of noise likely to be encountered once deployed. Our analyses reveal PTTMs are highly sensitive to noise. Hold/shift accuracy drops from 84% in clean speech to just 52% in 10 dB music noise. Training with noisy data enables a multimodal PTTM, which includes visual features to better exploit visual cues, with 72% accuracy in 10 dB music noise. The multimodal PTTM outperforms the audio-only PTTM across all noise types and SNRs, highlighting its ability to exploit visual cues; however, this does not always generalise to new types of noise. Analysis also reveals that successful training relies on accurate transcription, limiting the use of ASR-derived transcriptions to clean conditions. We make code publicly available for future research.
#### AudioTurbo: Fast Text-to-Audio Generation with Rectified Diffusion
 - **Authors:** Junqi Zhao, Jinzheng Zhao, Haohe Liu, Yun Chen, Lu Han, Xubo Liu, Mark Plumbley, Wenwu Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22106

 - **Pdf link:** https://arxiv.org/pdf/2505.22106

 - **Abstract**
 Diffusion models have significantly improved the quality and diversity of audio generation but are hindered by slow inference speed. Rectified flow enhances inference speed by learning straight-line ordinary differential equation (ODE) paths. However, this approach requires training a flow-matching model from scratch and tends to perform suboptimally, or even poorly, at low step counts. To address the limitations of rectified flow while leveraging the advantages of advanced pre-trained diffusion models, this study integrates pre-trained models with the rectified diffusion method to improve the efficiency of text-to-audio (TTA) generation. Specifically, we propose AudioTurbo, which learns first-order ODE paths from deterministic noise sample pairs generated by a pre-trained TTA model. Experiments on the AudioCaps dataset demonstrate that our model, with only 10 sampling steps, outperforms prior models and reduces inference to 3 steps compared to a flow-matching-based acceleration model.
#### Developing a Top-tier Framework in Naturalistic Conditions Challenge for Categorized Emotion Prediction: From Speech Foundation Models and Learning Objective to Data Augmentation and Engineering Choices
 - **Authors:** Tiantian Feng, Thanathai Lertpetchpun, Dani Byrd, Shrikanth Narayanan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22133

 - **Pdf link:** https://arxiv.org/pdf/2505.22133

 - **Abstract**
 Speech emotion recognition (SER), particularly for naturally expressed emotions, remains a challenging computational task. Key challenges include the inherent subjectivity in emotion annotation and the imbalanced distribution of emotion labels in datasets. This paper introduces the \texttt{SAILER} system developed for participation in the INTERSPEECH 2025 Emotion Recognition Challenge (Task 1). The challenge dataset, which contains natural emotional speech from podcasts, serves as a valuable resource for studying imbalanced and subjective emotion annotations. Our system is designed to be simple, reproducible, and effective, highlighting critical choices in modeling, learning objectives, data augmentation, and engineering choices. Results show that even a single system (without ensembling) can outperform more than 95\% of the submissions, with a Macro-F1 score exceeding 0.4. Moreover, an ensemble of three systems further improves performance, achieving a competitively ranked score (top-3 performing team). Our model is at: this https URL.
#### Two-stage Audio-Visual Target Speaker Extraction System for Real-Time Processing On Edge Device
 - **Authors:** Zixuan Li, Xueliang Zhang, Lei Miao, Zhipeng Yan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22229

 - **Pdf link:** https://arxiv.org/pdf/2505.22229

 - **Abstract**
 Audio-Visual Target Speaker Extraction (AVTSE) aims to isolate a target speaker's voice in a multi-speaker environment with visual cues as auxiliary. Most of the existing AVTSE methods encode visual and audio features simultaneously, resulting in extremely high computational complexity and making it impractical for real-time processing on edge devices. To tackle this issue, we proposed a two-stage ultra-compact AVTSE system. Specifically, in the first stage, a compact network is employed for voice activity detection (VAD) using visual information. In the second stage, the VAD results are combined with audio inputs to isolate the target speaker's voice. Experiments show that the proposed system effectively suppresses background noise and interfering voices while spending little computational resources.
#### Advancing Hearing Assessment: An ASR-Based Frequency-Specific Speech Test for Diagnosing Presbycusis
 - **Authors:** Stefan Bleeck
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22231

 - **Pdf link:** https://arxiv.org/pdf/2505.22231

 - **Abstract**
 Traditional audiometry often fails to fully characterize the functional impact of hearing loss on speech understanding, particularly supra-threshold deficits and frequency-specific perception challenges in conditions like presbycusis. This paper presents the development and simulated evaluation of a novel Automatic Speech Recognition (ASR)-based frequency-specific speech test designed to provide granular diagnostic insights. Our approach leverages ASR to simulate the perceptual effects of moderate sloping hearing loss by processing speech stimuli under controlled acoustic degradation and subsequently analyzing phoneme-level confusion patterns. Key findings indicate that simulated hearing loss introduces specific phoneme confusions, predominantly affecting high-frequency consonants (e.g., alveolar/palatal to labiodental substitutions) and leading to significant phoneme deletions, consistent with the acoustic cues degraded in presbycusis. A test battery curated from these ASR-derived confusions demonstrated diagnostic value, effectively differentiating between simulated normal-hearing and hearing-impaired listeners in a comprehensive simulation. This ASR-driven methodology offers a promising avenue for developing objective, granular, and frequency-specific hearing assessment tools that complement traditional audiometry. Future work will focus on validating these findings with human participants and exploring the integration of advanced AI models for enhanced diagnostic precision.
#### Effective Context in Neural Speech Models
 - **Authors:** Yen Meng, Sharon Goldwater, Hao Tang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22487

 - **Pdf link:** https://arxiv.org/pdf/2505.22487

 - **Abstract**
 Modern neural speech models benefit from having longer context, and many approaches have been proposed to increase the maximum context a model can use. However, few have attempted to measure how much context these models actually use, i.e., the effective context. Here, we propose two approaches to measuring the effective context, and use them to analyze different speech Transformers. For supervised models, we find that the effective context correlates well with the nature of the task, with fundamental frequency tracking, phone classification, and word classification requiring increasing amounts of effective context. For self-supervised models, we find that effective context increases mainly in the early layers, and remains relatively short -- similar to the supervised phone model. Given that these models do not use a long context during prediction, we show that HuBERT can be run in streaming mode without modification to the architecture and without further fine-tuning.
#### Towards General Discrete Speech Codec for Complex Acoustic Environments: A Study of Reconstruction and Downstream Task Consistency
 - **Authors:** Haoran Wang, Guanyu Chen, Bohan Li, Hankun Wang, Yiwei Guo, Zhihan Li, Xie Chen, Kai Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22515

 - **Pdf link:** https://arxiv.org/pdf/2505.22515

 - **Abstract**
 Neural speech codecs excel in reconstructing clean speech signals; however, their efficacy in complex acoustic environments and downstream signal processing tasks remains underexplored. In this study, we introduce a novel benchmark named Environment-Resilient Speech Codec Benchmark (ERSB) to systematically evaluate whether neural speech codecs are environment-resilient. Specifically, we assess two key capabilities: (1) robust reconstruction, which measures the preservation of both speech and non-speech acoustic details, and (2) downstream task consistency, which ensures minimal deviation in downstream signal processing tasks when using reconstructed speech instead of the original. Our comprehensive experiments reveal that complex acoustic environments significantly degrade signal reconstruction and downstream task consistency. This work highlights the limitations of current speech codecs and raises a future direction that improves them for greater environmental resilience.
#### Effective and Efficient One-pass Compression of Speech Foundation Models Using Sparsity-aware Self-pinching Gates
 - **Authors:** Haoning Xu, Zhaoqing Li, Youjun Chen, Huimeng Wang, Guinan Li, Mengzhe Geng, Chengxi Deng, Xunying Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22608

 - **Pdf link:** https://arxiv.org/pdf/2505.22608

 - **Abstract**
 This paper presents a novel approach for speech foundation models compression that tightly integrates model pruning and parameter update into a single stage. Highly compact layer-level tied self-pinching gates each containing only a single learnable threshold are jointly trained with uncompressed models and used in fine-grained neuron level pruning. Experiments conducted on the LibriSpeech-100hr corpus suggest that our approach reduces the number of parameters of wav2vec2.0-base and HuBERT-large models by 65% and 60% respectively, while incurring no statistically significant word error rate (WER) increase on the test-clean dataset. Compared to previously published methods on the same task, our approach not only achieves the lowest WER of 7.05% on the test-clean dataset under a comparable model compression ratio of 4.26x, but also operates with at least 25% less model compression time.


by Zyzzyva0381 (Windy). 


2025-05-29
