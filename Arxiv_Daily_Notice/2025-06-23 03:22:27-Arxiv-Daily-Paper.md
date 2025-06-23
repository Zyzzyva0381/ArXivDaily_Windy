# Showing new listings for Monday, 23 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 22papers 
#### Spatio-spectral diarization of meetings by combining TDOA-based segmentation and speaker embedding-based clustering
 - **Authors:** Tobias Cord-Landwehr, Tobias Gburrek, Marc Deegen, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.16228

 - **Pdf link:** https://arxiv.org/pdf/2506.16228

 - **Abstract**
 We propose a spatio-spectral, combined model-based and data-driven diarization pipeline consisting of TDOA-based segmentation followed by embedding-based clustering. The proposed system requires neither access to multi-channel training data nor prior knowledge about the number or placement of microphones. It works for both a compact microphone array and distributed microphones, with minor adjustments. Due to its superior handling of overlapping speech during segmentation, the proposed pipeline significantly outperforms the single-channel pyannote approach, both in a scenario with a compact microphone array and in a setup with distributed microphones. Additionally, we show that, unlike fully spatial diarization pipelines, the proposed system can correctly track speakers when they change positions.
#### EDNet: A Distortion-Agnostic Speech Enhancement Framework with Gating Mamba Mechanism and Phase Shift-Invariant Training
 - **Authors:** Doyeop Kwak, Youngjoon Jang, Seongyu Kim, Joon Son Chung
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.16231

 - **Pdf link:** https://arxiv.org/pdf/2506.16231

 - **Abstract**
 Speech signals in real-world environments are frequently affected by various distortions such as additive noise, reverberation, and bandwidth limitation, which may appear individually or in combination. Traditional speech enhancement methods typically rely on either masking, which focuses on suppressing non-speech components while preserving observable structure, or mapping, which seeks to recover clean speech through direct transformation of the input. Each approach offers strengths in specific scenarios but may be less effective outside its target conditions. We propose the Erase and Draw Network (EDNet), a distortion-agnostic speech enhancement framework designed to handle a broad range of distortion types without prior assumptions about task or input characteristics. EDNet consists of two main components: (1) the Gating Mamba (GM) module, which adaptively combines masking and mapping through a learnable gating mechanism that selects between suppression (Erase) and reconstruction (Draw) based on local signal features, and (2) Phase Shift-Invariant Training (PSIT), a shift tolerant supervision strategy that improves phase estimation by enabling dynamic alignment during training while remaining compatible with standard loss functions. Experimental results on denoising, dereverberation, bandwidth extension, and multi distortion enhancement tasks show that EDNet consistently achieves strong performance across conditions, demonstrating its architectural flexibility and adaptability to diverse task settings.
#### RapFlow-TTS: Rapid and High-Fidelity Text-to-Speech with Improved Consistency Flow Matching
 - **Authors:** Hyun Joon Park, Jeongmin Liu, Jin Sob Kim, Jeong Yeol Yang, Sung Won Han, Eunwoo Song
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2506.16741

 - **Pdf link:** https://arxiv.org/pdf/2506.16741

 - **Abstract**
 We introduce RapFlow-TTS, a rapid and high-fidelity TTS acoustic model that leverages velocity consistency constraints in flow matching (FM) training. Although ordinary differential equation (ODE)-based TTS generation achieves natural-quality speech, it typically requires a large number of generation steps, resulting in a trade-off between quality and inference speed. To address this challenge, RapFlow-TTS enforces consistency in the velocity field along the FM-straightened ODE trajectory, enabling consistent synthetic quality with fewer generation steps. Additionally, we introduce techniques such as time interval scheduling and adversarial learning to further enhance the quality of the few-step synthesis. Experimental results show that RapFlow-TTS achieves high-fidelity speech synthesis with a 5- and 10-fold reduction in synthesis steps than the conventional FM- and score-based approaches, respectively.
#### State-Space Models in Efficient Whispered and Multi-dialect Speech Recognition
 - **Authors:** Aref Farhadipour, Homayoon Beigi, Volker Dellwo, Hadi Veisi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.16969

 - **Pdf link:** https://arxiv.org/pdf/2506.16969

 - **Abstract**
 Whispered speech recognition presents significant challenges for conventional automatic speech recognition systems, particularly when combined with dialect variation. However, utilizing an efficient method to solve this problem using a low-range dataset and processing load is beneficial. This paper proposes a solution using a Mamba-based state-space model and four fine-tuned self-supervised models consisting of Wav2Vec2, WavLM, HuBERT, and Whisper to address the dual challenges of whispered speech and dialect diversity. Based on our knowledge, this represents the best performance reported on the wTIMIT and CHAINS datasets for whispered speech recognition. We trained the models using whispered and normal speech data across Singaporean, US, and Irish dialects. The findings demonstrated that utilizing the proposed Mamba-based model could work as a highly efficient model trained with low amounts of whispered data to simultaneously work on whispered and normal speech recognition. The code for this work is freely available.
#### Explainable speech emotion recognition through attentive pooling: insights from attention-based temporal localization
 - **Authors:** Tahitoa Leygue (DIASI (CEA, LIST)), Astrid Sabourin (DIASI (CEA, LIST)), Christian Bolzmacher (DIASI (CEA, LIST)), Sylvain Bouchigny (DIASI (CEA, LIST)), Margarita Anastassova (DIASI (CEA, LIST)), Quoc-Cuong Pham (DIASI (CEA, LIST))
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.15754

 - **Pdf link:** https://arxiv.org/pdf/2506.15754

 - **Abstract**
 State-of-the-art transformer models for Speech Emotion Recognition (SER) rely on temporal feature aggregation, yet advanced pooling methods remain underexplored. We systematically benchmark pooling strategies, including Multi-Query Multi-Head Attentive Statistics Pooling, which achieves a 3.5 percentage point macro F1 gain over average pooling. Attention analysis shows 15 percent of frames capture 80 percent of emotion cues, revealing a localized pattern of emotional information. Analysis of high-attention frames reveals that non-linguistic vocalizations and hyperarticulated phonemes are disproportionately prioritized during pooling, mirroring human perceptual strategies. Our findings position attentive pooling as both a performant SER mechanism and a biologically plausible tool for explainable emotion localization. On Interspeech 2025 Speech Emotion Recognition in Naturalistic Conditions Challenge, our approach obtained a macro F1 score of 0.3649.
#### Early Attentive Sparsification Accelerates Neural Speech Transcription
 - **Authors:** Zifei Xu, Sayeh Sharify, Hesham Mostafa, Tristan Webb, Wanzin Yazar, Xin Wang
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.15912

 - **Pdf link:** https://arxiv.org/pdf/2506.15912

 - **Abstract**
 Transformer-based neural speech processing has achieved state-of-the-art performance. Since speech audio signals are known to be highly compressible, here we seek to accelerate neural speech transcription by time-domain signal sparsification early in the neural encoding stage, taking advantage of the interpretability of the self-attention mechanism in transformer audio encoders. With the Whisper family of models, we perform a systematic architecture search over the joint space of sparsification stage (a certain encoder layer) and compression ratio (sparsity). We found that the best resulting solutions under 1% accuracy degradation choose to sparsify the hidden state to 40-60% sparsity at an early encoding stage, and thereby achieve up to 1.6x runtime acceleration in English speech transcription tasks on Nvidia GPUs without any fine-tuning.
#### Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion
 - **Authors:** Markus Frohmann, Gabriel Meseguer-Brocal, Markus Schedl, Elena V. Epure
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.15981

 - **Pdf link:** https://arxiv.org/pdf/2506.15981

 - **Abstract**
 The rapid advancement of AI-based music generation tools is revolutionizing the music industry but also posing challenges to artists, copyright holders, and providers alike. This necessitates reliable methods for detecting such AI-generated content. However, existing detectors, relying on either audio or lyrics, face key practical limitations: audio-based detectors fail to generalize to new or unseen generators and are vulnerable to audio perturbations; lyrics-based methods require cleanly formatted and accurate lyrics, unavailable in practice. To overcome these limitations, we propose a novel, practically grounded approach: a multimodal, modular late-fusion pipeline that combines automatically transcribed sung lyrics and speech features capturing lyrics-related information within the audio. By relying on lyrical aspects directly from audio, our method enhances robustness, mitigates susceptibility to low-level artifacts, and enables practical applicability. Experiments show that our method, DE-detect, outperforms existing lyrics-based detectors while also being more robust to audio perturbations. Thus, it offers an effective, robust solution for detecting AI-generated music in real-world scenarios. Our code is available at this https URL.
#### VS-Singer: Vision-Guided Stereo Singing Voice Synthesis with Consistency Schrödinger Bridge
 - **Authors:** Zijing Zhao, Kai Wang, Hao Huang, Ying Hu, Liang He, Jichen Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16020

 - **Pdf link:** https://arxiv.org/pdf/2506.16020

 - **Abstract**
 To explore the potential advantages of utilizing spatial cues from images for generating stereo singing voices with room reverberation, we introduce VS-Singer, a vision-guided model designed to produce stereo singing voices with room reverberation from scene images. VS-Singer comprises three modules: firstly, a modal interaction network integrates spatial features into text encoding to create a linguistic representation enriched with spatial information. Secondly, the decoder employs a consistency Schrödinger bridge to facilitate one-step sample generation. Moreover, we utilize the SFE module to improve the consistency of audio-visual matching. To our knowledge, this study is the first to combine stereo singing voice synthesis with visual acoustic matching within a unified framework. Experimental results demonstrate that VS-Singer can effectively generate stereo singing voices that align with the scene perspective in a single step.
#### Improved Intelligibility of Dysarthric Speech using Conditional Flow Matching
 - **Authors:** Shoutrik Das, Nishant Singh, Arjun Gangwar, S Umesh
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16127

 - **Pdf link:** https://arxiv.org/pdf/2506.16127

 - **Abstract**
 Dysarthria is a neurological disorder that significantly impairs speech intelligibility, often rendering affected individuals unable to communicate effectively. This necessitates the development of robust dysarthric-to-regular speech conversion techniques. In this work, we investigate the utility and limitations of self-supervised learning (SSL) features and their quantized representations as an alternative to mel-spectrograms for speech generation. Additionally, we explore methods to mitigate speaker variability by generating clean speech in a single-speaker voice using features extracted from WavLM. To this end, we propose a fully non-autoregressive approach that leverages Conditional Flow Matching (CFM) with Diffusion Transformers to learn a direct mapping from dysarthric to clean speech. Our findings highlight the effectiveness of discrete acoustic units in improving intelligibility while achieving faster convergence compared to traditional mel-spectrogram-based approaches.
#### Single-Microphone-Based Sound Source Localization for Mobile Robots in Reverberant Environments
 - **Authors:** Jiang Wang, Runwu Shi, Benjamin Yen, He Kong, Kazuhiro Nakadai
 - **Subjects:** Subjects:
Robotics (cs.RO); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16173

 - **Pdf link:** https://arxiv.org/pdf/2506.16173

 - **Abstract**
 Accurately estimating sound source positions is crucial for robot audition. However, existing sound source localization methods typically rely on a microphone array with at least two spatially preconfigured microphones. This requirement hinders the applicability of microphone-based robot audition systems and technologies. To alleviate these challenges, we propose an online sound source localization method that uses a single microphone mounted on a mobile robot in reverberant environments. Specifically, we develop a lightweight neural network model with only 43k parameters to perform real-time distance estimation by extracting temporal information from reverberant signals. The estimated distances are then processed using an extended Kalman filter to achieve online sound source localization. To the best of our knowledge, this is the first work to achieve online sound source localization using a single microphone on a moving robot, a gap that we aim to fill in this work. Extensive experiments demonstrate the effectiveness and merits of our approach. To benefit the broader research community, we have open-sourced our code at this https URL.
#### End-to-End Speech Translation for Low-Resource Languages Using Weakly Labeled Data
 - **Authors:** Aishwarya Pothula, Bhavana Akkiraju, Srihari Bandarupalli, Charan D, Santosh Kesiraju, Anil Kumar Vuppala
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16251

 - **Pdf link:** https://arxiv.org/pdf/2506.16251

 - **Abstract**
 The scarcity of high-quality annotated data presents a significant challenge in developing effective end-to-end speech-to-text translation (ST) systems, particularly for low-resource languages. This paper explores the hypothesis that weakly labeled data can be used to build ST models for low-resource language pairs. We constructed speech-to-text translation datasets with the help of bitext mining using state-of-the-art sentence encoders. We mined the multilingual Shrutilipi corpus to build Shrutilipi-anuvaad, a dataset comprising ST data for language pairs Bengali-Hindi, Malayalam-Hindi, Odia-Hindi, and Telugu-Hindi. We created multiple versions of training data with varying degrees of quality and quantity to investigate the effect of quality versus quantity of weakly labeled data on ST model performance. Results demonstrate that ST systems can be built using weakly labeled data, with performance comparable to massive multi-modal multilingual baselines such as SONAR and SeamlessM4T.
#### Advancing Automated Speaking Assessment Leveraging Multifaceted Relevance and Grammar Information
 - **Authors:** Hao-Chien Lu, Jhen-Ke Lin, Hong-Yun Lin, Chung-Chun Wang, Berlin Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16285

 - **Pdf link:** https://arxiv.org/pdf/2506.16285

 - **Abstract**
 Current automated speaking assessment (ASA) systems for use in multi-aspect evaluations often fail to make full use of content relevance, overlooking image or exemplar cues, and employ superficial grammar analysis that lacks detailed error types. This paper ameliorates these deficiencies by introducing two novel enhancements to construct a hybrid scoring model. First, a multifaceted relevance module integrates question and the associated image content, exemplar, and spoken response of an L2 speaker for a comprehensive assessment of content relevance. Second, fine-grained grammar error features are derived using advanced grammar error correction (GEC) and detailed annotation to identify specific error categories. Experiments and ablation studies demonstrate that these components significantly improve the evaluation of content relevance, language use, and overall ASA performance, highlighting the benefits of using richer, more nuanced feature sets for holistic speaking assessment.
#### Optimizing Multilingual Text-To-Speech with Accents & Emotions
 - **Authors:** Pranav Pawar, Akshansh Dwivedi, Jenish Boricha, Himanshu Gohil, Aditya Dubey
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16310

 - **Pdf link:** https://arxiv.org/pdf/2506.16310

 - **Abstract**
 State-of-the-art text-to-speech (TTS) systems realize high naturalness in monolingual environments, synthesizing speech with correct multilingual accents (especially for Indic languages) and context-relevant emotions still poses difficulty owing to cultural nuance discrepancies in current frameworks. This paper introduces a new TTS architecture integrating accent along with preserving transliteration with multi-scale emotion modelling, in particularly tuned for Hindi and Indian English accent. Our approach extends the Parler-TTS model by integrating A language-specific phoneme alignment hybrid encoder-decoder architecture, and culture-sensitive emotion embedding layers trained on native speaker corpora, as well as incorporating a dynamic accent code switching with residual vector quantization. Quantitative tests demonstrate 23.7% improvement in accent accuracy (Word Error Rate reduction from 15.4% to 11.8%) and 85.3% emotion recognition accuracy from native listeners, surpassing METTS and VECL-TTS baselines. The novelty of the system is that it can mix code in real time - generating statements such as "Namaste, let's talk about <Hindi phrase>" with uninterrupted accent shifts while preserving emotional consistency. Subjective evaluation with 200 users reported a mean opinion score (MOS) of 4.2/5 for cultural correctness, much better than existing multilingual systems (p<0.01). This research makes cross-lingual synthesis more feasible by showcasing scalable accent-emotion disentanglement, with direct application in South Asian EdTech and accessibility software.
#### InstructTTSEval: Benchmarking Complex Natural-Language Instruction Following in Text-to-Speech Systems
 - **Authors:** Kexin Huang, Qian Tu, Liwei Fan, Chenchen Yang, Dong Zhang, Shimin Li, Zhaoye Fei, Qinyuan Cheng, Xipeng Qiu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16381

 - **Pdf link:** https://arxiv.org/pdf/2506.16381

 - **Abstract**
 In modern speech synthesis, paralinguistic information--such as a speaker's vocal timbre, emotional state, and dynamic prosody--plays a critical role in conveying nuance beyond mere semantics. Traditional Text-to-Speech (TTS) systems rely on fixed style labels or inserting a speech prompt to control these cues, which severely limits flexibility. Recent attempts seek to employ natural-language instructions to modulate paralinguistic features, substantially improving the generalization of instruction-driven TTS models. Although many TTS systems now support customized synthesis via textual description, their actual ability to interpret and execute complex instructions remains largely unexplored. In addition, there is still a shortage of high-quality benchmarks and automated evaluation metrics specifically designed for instruction-based TTS, which hinders accurate assessment and iterative optimization of these models. To address these limitations, we introduce InstructTTSEval, a benchmark for measuring the capability of complex natural-language style control. We introduce three tasks, namely Acoustic-Parameter Specification, Descriptive-Style Directive, and Role-Play, including English and Chinese subsets, each with 1k test cases (6k in total) paired with reference audio. We leverage Gemini as an automatic judge to assess their instruction-following abilities. Our evaluation of accessible instruction-following TTS systems highlights substantial room for further improvement. We anticipate that InstructTTSEval will drive progress toward more powerful, flexible, and accurate instruction-following TTS.
#### Towards Bitrate-Efficient and Noise-Robust Speech Coding with Variable Bitrate RVQ
 - **Authors:** Yunkee Chae, Kyogu Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16538

 - **Pdf link:** https://arxiv.org/pdf/2506.16538

 - **Abstract**
 Residual Vector Quantization (RVQ) has become a dominant approach in neural speech and audio coding, providing high-fidelity compression. However, speech coding presents additional challenges due to real-world noise, which degrades compression efficiency. Standard codecs allocate bits uniformly, wasting bitrate on noise components that do not contribute to intelligibility. This paper introduces a Variable Bitrate RVQ (VRVQ) framework for noise-robust speech coding, dynamically adjusting bitrate per frame to optimize rate-distortion trade-offs. Unlike constant bitrate (CBR) RVQ, our method prioritizes critical speech components while suppressing residual noise. Additionally, we integrate a feature denoiser to further improve noise robustness. Experimental results show that VRVQ improves rate-distortion trade-offs over conventional methods, achieving better compression efficiency and perceptual quality in noisy conditions. Samples are available at our project page: this https URL.
#### Automatic Speech Recognition Biases in Newcastle English: an Error Analysis
 - **Authors:** Dana Serditova, Kevin Tang, Jochen Steffens
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Computers and Society (cs.CY); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16558

 - **Pdf link:** https://arxiv.org/pdf/2506.16558

 - **Abstract**
 Automatic Speech Recognition (ASR) systems struggle with regional dialects due to biased training which favours mainstream varieties. While previous research has identified racial, age, and gender biases in ASR, regional bias remains underexamined. This study investigates ASR performance on Newcastle English, a well-documented regional dialect known to be challenging for ASR. A two-stage analysis was conducted: first, a manual error analysis on a subsample identified key phonological, lexical, and morphosyntactic errors behind ASR misrecognitions; second, a case study focused on the systematic analysis of ASR recognition of the regional pronouns ``yous'' and ``wor''. Results show that ASR errors directly correlate with regional dialectal features, while social factors play a lesser role in ASR mismatches. We advocate for greater dialectal diversity in ASR training data and highlight the value of sociolinguistic analysis in diagnosing and addressing regional biases.
#### Weight Factorization and Centralization for Continual Learning in Speech Recognition
 - **Authors:** Enes Yavuz Ugan, Ngoc-Quan Pham, Alexander Waibel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16574

 - **Pdf link:** https://arxiv.org/pdf/2506.16574

 - **Abstract**
 Modern neural network based speech recognition models are required to continually absorb new data without re-training the whole system, especially in downstream applications using foundation models, having no access to the original training data. Continually training the models in a rehearsal-free, multilingual, and language agnostic condition, likely leads to catastrophic forgetting, when a seemingly insignificant disruption to the weights can destructively harm the quality of the models. Inspired by the ability of human brains to learn and consolidate knowledge through the waking-sleeping cycle, we propose a continual learning approach with two distinct phases: factorization and centralization, learning and merging knowledge accordingly. Our experiments on a sequence of varied code-switching datasets showed that the centralization stage can effectively prevent catastrophic forgetting by accumulating the knowledge in multiple scattering low-rank adapters.
#### Streaming Non-Autoregressive Model for Accent Conversion and Pronunciation Improvement
 - **Authors:** Tuan-Nam Nguyen, Ngoc-Quan Pham, Seymanur Akti, Alexander Waibel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16580

 - **Pdf link:** https://arxiv.org/pdf/2506.16580

 - **Abstract**
 We propose a first streaming accent conversion (AC) model that transforms non-native speech into a native-like accent while preserving speaker identity, prosody and improving pronunciation. Our approach enables stream processing by modifying a previous AC architecture with an Emformer encoder and an optimized inference mechanism. Additionally, we integrate a native text-to-speech (TTS) model to generate ideal ground-truth data for efficient training. Our streaming AC model achieves comparable performance to the top AC models while maintaining stable latency, making it the first AC system capable of streaming.
#### Learning Magnitude Distribution of Sound Fields via Conditioned Autoencoder
 - **Authors:** Shoichi Koyama, Kenji Ishizuka
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16729

 - **Pdf link:** https://arxiv.org/pdf/2506.16729

 - **Abstract**
 A learning-based method for estimating the magnitude distribution of sound fields from spatially sparse measurements is proposed. Estimating the magnitude distribution of acoustic transfer function (ATF) is useful when phase measurements are unreliable or inaccessible and has a wide range of applications related to spatial audio. We propose a neural-network-based method for the ATF magnitude estimation. The key feature of our network architecture is the input and output layers conditioned on source and receiver positions and frequency and the aggregation module of latent variables, which can be interpreted as an autoencoder-based extension of the basis expansion of the sound field. Numerical simulation results indicated that the ATF magnitude is accurately estimated with a small number of receivers by our proposed method.
#### LM-SPT: LM-Aligned Semantic Distillation for Speech Tokenization
 - **Authors:** Daejin Jo, Jeeyoung Yun, Byungseok Roh, Sungwoong Kim
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16738

 - **Pdf link:** https://arxiv.org/pdf/2506.16738

 - **Abstract**
 With the rapid progress of speech language models (SLMs), discrete speech tokens have emerged as a core interface between speech and text, enabling unified modeling across modalities. Recent speech tokenization approaches aim to isolate semantic information from low-level acoustics to better align with language models. In particular, previous methods use SSL teachers such as HuBERT to extract semantic representations, which are then distilled into a semantic quantizer to suppress acoustic redundancy as well as capture content-related latent structures. However, they still produce speech token sequences significantly longer than their textual counterparts, creating challenges for efficient speech-language modeling. Reducing the frame rate is a natural solution, but standard techniques, such as rigid average pooling across frames, can distort or dilute the semantic structure required for effective LM alignment. To address this, we propose LM-SPT, a speech tokenization method that introduces a novel semantic distillation. Instead of directly matching teacher and student features via pooling, we reconstruct speech solely from semantic tokens and minimize the discrepancy between the encoded representations of the original and reconstructed waveforms, obtained from a frozen automatic speech recognition (ASR) encoder. This indirect yet data-driven supervision enables the tokenizer to learn discrete units that are more semantically aligned with language models. LM-SPT further incorporates architectural improvements to the encoder and decoder for speech tokenization, and supports multiple frame rates, including 25Hz, 12.5Hz, and 6.25Hz. Experimental results show that LM-SPT achieves superior reconstruction fidelity compared to baselines, and that SLMs trained with LM-SPT tokens achieve competitive performances on speech-to-text and consistently outperform baselines on text-to-speech tasks.
#### Hybrid-Sep: Language-queried audio source separation via pre-trained Model Fusion and Adversarial Diffusion Training
 - **Authors:** Jianyuan Feng, Guangzheng Li, Yangfei Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16833

 - **Pdf link:** https://arxiv.org/pdf/2506.16833

 - **Abstract**
 Language-queried Audio Separation (LASS) employs linguistic queries to isolate target sounds based on semantic descriptions. However, existing methods face challenges in aligning complex auditory features with linguistic context while preserving separation precision. Current research efforts focus primarily on text description augmentation and architectural innovations, yet the potential of integrating pre-trained self-supervised learning (SSL) audio models and Contrastive Language-Audio Pretraining (CLAP) frameworks, capable of extracting cross-modal audio-text relationships, remains underexplored. To address this, we present HybridSep, a two-stage LASS framework that synergizes SSL-based acoustic representations with CLAP-derived semantic embeddings. Our framework introduces Adversarial Consistent Training (ACT), a novel optimization strategy that treats diffusion as an auxiliary regularization loss while integrating adversarial training to enhance separation fidelity. Experiments demonstrate that HybridSep achieves significant performance improvements over state-of-the-art baselines (e.g., AudioSep, FlowSep) across multiple metrics, establishing new benchmarks for LASS tasks.
#### ITO-Master: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors
 - **Authors:** Junghyun Koo, Marco A. Martinez-Ramirez, Wei-Hsiang Liao, Giorgio Fabbro, Michele Mancusi, Yuki Mitsufuji
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.16889

 - **Pdf link:** https://arxiv.org/pdf/2506.16889

 - **Abstract**
 Music mastering style transfer aims to model and apply the mastering characteristics of a reference track to a target track, simulating the professional mastering process. However, existing methods apply fixed processing based on a reference track, limiting users' ability to fine-tune the results to match their artistic intent. In this paper, we introduce the ITO-Master framework, a reference-based mastering style transfer system that integrates Inference-Time Optimization (ITO) to enable finer user control over the mastering process. By optimizing the reference embedding during inference, our approach allows users to refine the output dynamically, making micro-level adjustments to achieve more precise mastering results. We explore both black-box and white-box methods for modeling mastering processors and demonstrate that ITO improves mastering performance across different styles. Through objective evaluation, subjective listening tests, and qualitative analysis using text-based conditioning with CLAP embeddings, we validate that ITO enhances mastering style similarity while offering increased adaptability. Our framework provides an effective and user-controllable solution for mastering style transfer, allowing users to refine their results beyond the initial style transfer.


by Zyzzyva0381 (Windy). 


2025-06-23
