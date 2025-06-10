# Showing new listings for Tuesday, 10 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 32papers 
#### AS-ASR: A Lightweight Framework for Aphasia-Specific Automatic Speech Recognition
 - **Authors:** Chen Bao, Chuanbing Huo, Qinyu Chen, Chang Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2506.06566

 - **Pdf link:** https://arxiv.org/pdf/2506.06566

 - **Abstract**
 This paper proposes AS-ASR, a lightweight aphasia-specific speech recognition framework based on Whisper-tiny, tailored for low-resource deployment on edge devices. Our approach introduces a hybrid training strategy that systematically combines standard and aphasic speech at varying ratios, enabling robust generalization, and a GPT-4-based reference enhancement method that refines noisy aphasic transcripts, improving supervision quality. We conduct extensive experiments across multiple data mixing configurations and evaluation settings. Results show that our fine-tuned model significantly outperforms the zero-shot baseline, reducing WER on aphasic speech by over 30% while preserving performance on standard speech. The proposed framework offers a scalable, efficient solution for real-world disordered speech recognition.
#### Accurate analysis of the pitch pulse-based magnitude/phase structure of natural vowels and assessment of three lightweight time/frequency voicing restoration methods
 - **Authors:** Aníbal J. S. Ferreira, Luis M. T. Jesus, Laurentino M. M. Leal, Jorge E. F. Spratley
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.06675

 - **Pdf link:** https://arxiv.org/pdf/2506.06675

 - **Abstract**
 Whispered speech is produced when the vocal folds are not used, either intentionally, or due to a temporary or permanent voice condition. The essential difference between natural speech and whispered speech is that periodic signal components that exist in certain regions of the former, called voiced regions, as a consequence of the vibration of the vocal folds, are missing in the latter. The restoration of natural speech from whispered speech requires delicate signal processing procedures that are especially useful if they can be implemented on low-resourced portable devices, in real-time, and on-the-fly, taking advantage of the established source-filter paradigm of voice production and related models. This paper addresses two challenges that are intertwined and are key in informing and making viable this envisioned technological realization. The first challenge involves characterizing and modeling the evolution of the harmonic phase/magnitude structure of a sequence of individual pitch periods in a voiced region of natural speech comprising sustained or co-articulated vowels. This paper proposes a novel algorithm segmenting individual pitch pulses, which is then used to obtain illustrative results highlighting important differences between sustained and co-articulated vowels, and suggesting practical synthetic voicing approaches. The second challenge involves model-based synthetic voicing. Three implementation alternatives are described that differ in their signal reconstruction approaches: frequency-domain, combined frequency and time-domain, and physiologically-inspired separate filtering of glottal excitation pulses individually generated. The three alternatives are compared objectively using illustrative examples, and subjectively using the results of listening tests involving synthetic voicing of sustained and co-articulated vowels in word context.
#### Exploring Length Generalization For Transformer-based Speech Enhancement
 - **Authors:** Qiquan Zhang, Hongxu Zhu, Xinyuan Qian, Eliathamby Ambikairajah, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06697

 - **Pdf link:** https://arxiv.org/pdf/2506.06697

 - **Abstract**
 Transformer network architecture has proven effective in speech enhancement. However, as its core module, self-attention suffers from quadratic complexity, making it infeasible for training on long speech utterances. In practical scenarios, speech enhancement models are often required to perform on noisy speech at run-time that is substantially longer than the training utterances. It remains a challenge how a Transformer-based speech enhancement model can generalize to long speech utterances. In this paper, extensive empirical studies are conducted to explore the model's length generalization ability. In particular, we conduct speech enhancement experiments on four training objectives and evaluate with five metrics. Our studies establish that positional encoding is an effective instrument to dampen the effect of utterance length on speech enhancement. We first explore several existing positional encoding methods, and the results show that relative positional encoding methods exhibit a better length generalization property than absolute positional encoding methods. Additionally, we also explore a simpler and more effective positional encoding scheme, i.e. LearnLin, that uses only one trainable parameter for each attention head to scale the real relative position between time frames, which learns the different preferences on short- or long-term dependencies of these heads. The results demonstrate that our proposal exhibits excellent length generalization ability with comparable or superior performance than other state-of-the-art positional encoding strategies.
#### Neural Spectral Band Generation for Audio Coding
 - **Authors:** Woongjib Choi, Byeong Hyeon Kim, Hyungseob Lim, Inseon Jang, Hong-Goo Kang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.06732

 - **Pdf link:** https://arxiv.org/pdf/2506.06732

 - **Abstract**
 Audio bandwidth extension is the task of reconstructing missing high frequency components of bandwidth-limited audio signals, where bandwidth limitation is a common issue for audio signals due to several reasons, including channel capacity and data constraints. While conventional spectral band replication is a well-established parametric approach to audio bandwidth extension, the SBR usually entails coarse feature extraction and reconstruction techniques, which leads to limitations when processing various types of audio signals. In parallel, numerous deep neural network-based audio bandwidth extension methods have been proposed. These DNN-based methods are usually referred to as blind BWE, as these methods do not rely on prior information extracted from original signals, and only utilize given low frequency band signals to estimate missing high frequency components. In order to replace conventional SBR with DNNs, simply adopting existing DNN-based methodologies results in suboptimal performance due to the blindness of these methods. My proposed research suggests a new approach to parametric non-blind bandwidth extension, as DNN-based side information extraction and DNN-based bandwidth extension are performed only at the front and end of the audio coding pipeline.
#### Rhythm Features for Speaker Identification
 - **Authors:** Nick Mehlman, Thomas Thebaud, Dani Byrd, Shri Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06834

 - **Pdf link:** https://arxiv.org/pdf/2506.06834

 - **Abstract**
 While deep learning models have demonstrated robust performance in speaker recognition tasks, they primarily rely on low-level audio features learned empirically from spectrograms or raw waveforms. However, prior work has indicated that idiosyncratic speaking styles heavily influence the temporal structure of linguistic units in speech signals (rhythm). This makes rhythm a strong yet largely overlooked candidate for a speech identity feature. In this paper, we test this hypothesis by applying deep learning methods to perform text-independent speaker identification from rhythm features. Our findings support the usefulness of rhythmic information for speaker recognition tasks but also suggest that high intra-subject variability in ad-hoc speech can degrade its effectiveness.
#### Multi-Distillation from Speech and Music Representation Models
 - **Authors:** Jui-Chiang Wei, Yi-Cheng Lin, Fabian Ritter-Gutierrez, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07237

 - **Pdf link:** https://arxiv.org/pdf/2506.07237

 - **Abstract**
 Real-world audio often mixes speech and music, yet models typically handle only one domain. This paper introduces a multi-teacher distillation framework that unifies speech and music models into a single one while significantly reducing model size. Our approach leverages the strengths of domain-specific teacher models, such as HuBERT for speech and MERT for music, and explores various strategies to balance both domains. Experiments across diverse tasks demonstrate that our model matches the performance of domain-specific models, showing the effectiveness of cross-domain distillation. Additionally, we conduct few-shot learning experiments, highlighting the need for general models in real-world scenarios where labeled data is limited. Our results show that our model not only performs on par with specialized models but also outperforms them in few-shot scenarios, proving that a cross-domain approach is essential and effective for diverse tasks with limited data.
#### Speaker-Distinguishable CTC: Learning Speaker Distinction Using CTC for Multi-Talker Speech Recognition
 - **Authors:** Asahi Sakuma, Hiroaki Sato, Ryuga Sugano, Tadashi Kumano, Yoshihiko Kawai, Tetsuji Ogawa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.07515

 - **Pdf link:** https://arxiv.org/pdf/2506.07515

 - **Abstract**
 This paper presents a novel framework for multi-talker automatic speech recognition without the need for auxiliary information. Serialized Output Training (SOT), a widely used approach, suffers from recognition errors due to speaker assignment failures. Although incorporating auxiliary information, such as token-level timestamps, can improve recognition accuracy, extracting such information from natural conversational speech remains challenging. To address this limitation, we propose Speaker-Distinguishable CTC (SD-CTC), an extension of CTC that jointly assigns a token and its corresponding speaker label to each frame. We further integrate SD-CTC into the SOT framework, enabling the SOT model to learn speaker distinction using only overlapping speech and transcriptions. Experimental comparisons show that multi-task learning with SD-CTC and SOT reduces the error rate of the SOT model by 26% and achieves performance comparable to state-of-the-art methods relying on auxiliary information.
#### SongBloom: Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement
 - **Authors:** Chenyu Yang, Shuai Wang, Hangting Chen, Wei Tan, Jianwei Yu, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Multimedia (cs.MM)
 - **Arxiv link:** https://arxiv.org/abs/2506.07634

 - **Pdf link:** https://arxiv.org/pdf/2506.07634

 - **Abstract**
 Generating music with coherent structure, harmonious instrumental and vocal elements remains a significant challenge in song generation. Existing language models and diffusion-based methods often struggle to balance global coherence with local fidelity, resulting in outputs that lack musicality or suffer from incoherent progression and mismatched lyrics. This paper introduces $\textbf{SongBloom}$, a novel framework for full-length song generation that leverages an interleaved paradigm of autoregressive sketching and diffusion-based refinement. SongBloom employs an autoregressive diffusion model that combines the high fidelity of diffusion models with the scalability of language models. Specifically, it gradually extends a musical sketch from short to long and refines the details from coarse to fine-grained. The interleaved generation paradigm effectively integrates prior semantic and acoustic context to guide the generation process. Experimental results demonstrate that SongBloom outperforms existing methods across both subjective and objective metrics and achieves performance comparable to the state-of-the-art commercial music generation platforms. Audio samples are available on our demo page: this https URL\_demo.
#### Unified Semi-Supervised Pipeline for Automatic Speech Recognition
 - **Authors:** Nune Tadevosyan, Nikolay Karpov, Andrei Andrusenko, Vitaly Lavrukhin, Ante Jukic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07659

 - **Pdf link:** https://arxiv.org/pdf/2506.07659

 - **Abstract**
 Automatic Speech Recognition has been a longstanding research area, with substantial efforts dedicated to integrating semi-supervised learning due to the scarcity of labeled datasets. However, most prior work has focused on improving learning algorithms using existing datasets, without providing a complete public framework for large-scale semi-supervised training across new datasets or languages. In this work, we introduce a fully open-source semi-supervised training framework encompassing the entire pipeline: from unlabeled data collection to pseudo-labeling and model training. Our approach enables scalable dataset creation for any language using publicly available speech data under Creative Commons licenses. We also propose a novel pseudo-labeling algorithm, TopIPL, and evaluate it in both low-resource (Portuguese, Armenian) and high-resource (Spanish) settings. Notably, TopIPL achieves relative WER improvements of 18-40% for Portuguese, 5-16% for Armenian, and 2-8% for Spanish.
#### TESU-LLM: Training Speech-LLMs Without Speech via Unified Encoder Alignment
 - **Authors:** Taesoo Kim, Jong Hwan Ko
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06343

 - **Pdf link:** https://arxiv.org/pdf/2506.06343

 - **Abstract**
 Recent advances in speech-enabled language models have shown promising results in building intelligent voice assistants. However, most existing approaches rely on large-scale paired speech-text data and extensive computational resources, which pose challenges in terms of scalability and accessibility. In this paper, we present \textbf{TESU-LLM}, a novel framework that enables training speech-capable language models using only text data. Our key insight is to leverage a unified encoder that maps semantically equivalent text and speech inputs to a shared latent space. By aligning the encoder output with the embedding space of a LLM via a lightweight projection network, we enable the model to generalize from text-only supervision to speech-based inference. Despite being trained exclusively on text, TESU-LLM achieves strong performance on various speech-related benchmarks, comparable to baseline methods trained with large-scale multimodal datasets and substantial computational resources. These results highlight the effectiveness and efficiency of our approach, offering a scalable path toward building speech LLMs without speech data.
#### CAtCh: Cognitive Assessment through Cookie Thief
 - **Authors:** Joseph T Colonel, Carolyn Hagler, Guiselle Wismer, Laura Curtis, Jacqueline Becker, Juan Wisnivesky, Alex Federman, Gaurav Pandey
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06603

 - **Pdf link:** https://arxiv.org/pdf/2506.06603

 - **Abstract**
 Several machine learning algorithms have been developed for the prediction of Alzheimer's disease and related dementia (ADRD) from spontaneous speech. However, none of these algorithms have been translated for the prediction of broader cognitive impairment (CI), which in some cases is a precursor and risk factor of ADRD. In this paper, we evaluated several speech-based open-source methods originally proposed for the prediction of ADRD, as well as methods from multimodal sentiment analysis for the task of predicting CI from patient audio recordings. Results demonstrated that multimodal methods outperformed unimodal ones for CI prediction, and that acoustics-based approaches performed better than linguistics-based ones. Specifically, interpretable acoustic features relating to affect and prosody were found to significantly outperform BERT-based linguistic features and interpretable linguistic features, respectively. All the code developed for this study is available at this https URL.
#### A Fast and Lightweight Model for Causal Audio-Visual Speech Separation
 - **Authors:** Wendi Sang, Kai Li, Runxuan Yang, Jianqiang Huang, Xiaolin Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06689

 - **Pdf link:** https://arxiv.org/pdf/2506.06689

 - **Abstract**
 Audio-visual speech separation (AVSS) aims to extract a target speech signal from a mixed signal by leveraging both auditory and visual (lip movement) cues. However, most existing AVSS methods exhibit complex architectures and rely on future context, operating offline, which renders them unsuitable for real-time applications. Inspired by the pipeline of RTFSNet, we propose a novel streaming AVSS model, named Swift-Net, which enhances the causal processing capabilities required for real-time applications. Swift-Net adopts a lightweight visual feature extraction module and an efficient fusion module for audio-visual integration. Additionally, Swift-Net employs Grouped SRUs to integrate historical information across different feature spaces, thereby improving the utilization efficiency of historical information. We further propose a causal transformation template to facilitate the conversion of non-causal AVSS models into causal counterparts. Experiments on three standard benchmark datasets (LRS2, LRS3, and VoxCeleb2) demonstrated that under causal conditions, our proposed Swift-Net exhibited outstanding performance, highlighting the potential of this method for processing speech in complex environments.
#### SynHate: Detecting Hate Speech in Synthetic Deepfake Audio
 - **Authors:** Rishabh Ranjan, Kishan Pipariya, Mayank Vatsa, Richa Singh
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06772

 - **Pdf link:** https://arxiv.org/pdf/2506.06772

 - **Abstract**
 The rise of deepfake audio and hate speech, powered by advanced text-to-speech, threatens online safety. We present SynHate, the first multilingual dataset for detecting hate speech in synthetic audio, spanning 37 languages. SynHate uses a novel four-class scheme: Real-normal, Real-hate, Fake-normal, and Fake-hate. Built from MuTox and ADIMA datasets, it captures diverse hate speech patterns globally and in India. We evaluate five leading self-supervised models (Whisper-small/medium, XLS-R, AST, mHuBERT), finding notable performance differences by language, with Whisper-small performing best overall. Cross-dataset generalization remains a challenge. By releasing SynHate and baseline code, we aim to advance robust, culturally sensitive, and multilingual solutions against synthetic hate speech. The dataset is available at this https URL.
#### Beyond Classification: Towards Speech Emotion Reasoning with Multitask AudioLLMs
 - **Authors:** Wenyu Zhang, Yingxu He, Geyu Lin, Zhuohan Liu, Shuo Sun, Bin Wang, Xunlong Zou, Jeremy H. M. Wong, Qiongqiong Wang, Hardik B. Sailor, Nancy F. Chen, Ai Ti Aw
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06820

 - **Pdf link:** https://arxiv.org/pdf/2506.06820

 - **Abstract**
 Audio Large Language Models (AudioLLMs) have achieved strong results in semantic tasks like speech recognition and translation, but remain limited in modeling paralinguistic cues such as emotion. Existing approaches often treat emotion understanding as a classification problem, offering little insight into the underlying rationale behind predictions. In this work, we explore emotion reasoning, a strategy that leverages the generative capabilities of AudioLLMs to enhance emotion recognition by producing semantically aligned, evidence-grounded explanations. To support this in multitask AudioLLMs, we introduce a unified framework combining reasoning-augmented data supervision, dual-encoder architecture, and task-alternating training. This approach enables AudioLLMs to effectively learn different tasks while incorporating emotional reasoning. Experiments on IEMOCAP and MELD show that our approach not only improves emotion prediction accuracy but also enhances the coherence and evidential grounding of the generated responses.
#### Multimodal Spatial Language Maps for Robot Navigation and Manipulation
 - **Authors:** Chenguang Huang, Oier Mees, Andy Zeng, Wolfram Burgard
 - **Subjects:** Subjects:
Robotics (cs.RO); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06862

 - **Pdf link:** https://arxiv.org/pdf/2506.06862

 - **Abstract**
 Grounding language to a navigating agent's observations can leverage pretrained multimodal foundation models to match perceptions to object or event descriptions. However, previous approaches remain disconnected from environment mapping, lack the spatial precision of geometric maps, or neglect additional modality information beyond vision. To address this, we propose multimodal spatial language maps as a spatial map representation that fuses pretrained multimodal features with a 3D reconstruction of the environment. We build these maps autonomously using standard exploration. We present two instances of our maps, which are visual-language maps (VLMaps) and their extension to audio-visual-language maps (AVLMaps) obtained by adding audio information. When combined with large language models (LLMs), VLMaps can (i) translate natural language commands into open-vocabulary spatial goals (e.g., "in between the sofa and TV") directly localized in the map, and (ii) be shared across different robot embodiments to generate tailored obstacle maps on demand. Building upon the capabilities above, AVLMaps extend VLMaps by introducing a unified 3D spatial representation integrating audio, visual, and language cues through the fusion of features from pretrained multimodal foundation models. This enables robots to ground multimodal goal queries (e.g., text, images, or audio snippets) to spatial locations for navigation. Additionally, the incorporation of diverse sensory inputs significantly enhances goal disambiguation in ambiguous environments. Experiments in simulation and real-world settings demonstrate that our multimodal spatial language maps enable zero-shot spatial and multimodal goal navigation and improve recall by 50% in ambiguous scenarios. These capabilities extend to mobile robots and tabletop manipulators, supporting navigation and interaction guided by visual, audio, and spatial cues.
#### Automatic Speech Recognition of African American English: Lexical and Contextual Effects
 - **Authors:** Hamid Mojarad, Kevin Tang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06888

 - **Pdf link:** https://arxiv.org/pdf/2506.06888

 - **Abstract**
 Automatic Speech Recognition (ASR) models often struggle with the phonetic, phonological, and morphosyntactic features found in African American English (AAE). This study focuses on two key AAE variables: Consonant Cluster Reduction (CCR) and ING-reduction. It examines whether the presence of CCR and ING-reduction increases ASR misrecognition. Subsequently, it investigates whether end-to-end ASR systems without an external Language Model (LM) are more influenced by lexical neighborhood effect and less by contextual predictability compared to systems with an LM. The Corpus of Regional African American Language (CORAAL) was transcribed using wav2vec 2.0 with and without an LM. CCR and ING-reduction were detected using the Montreal Forced Aligner (MFA) with pronunciation expansion. The analysis reveals a small but significant effect of CCR and ING on Word Error Rate (WER) and indicates a stronger presence of lexical neighborhood effect in ASR systems without LMs.
#### "In This Environment, As That Speaker": A Text-Driven Framework for Multi-Attribute Speech Conversion
 - **Authors:** Jiawei Jin, Zhuhan Yang, Yixuan Zhou, Zhiyong Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07036

 - **Pdf link:** https://arxiv.org/pdf/2506.07036

 - **Abstract**
 We propose TES-VC (Text-driven Environment and Speaker controllable Voice Conversion), a text-driven voice conversion framework with independent control of speaker timbre and environmental acoustics. TES-VC processes simultaneous text inputs for target voice and environment, accurately generating speech matching described timbre/environment while preserving source content. Trained on synthetic data with decoupled vocal/environment features via latent diffusion modeling, our method eliminates interference between attributes. The Retrieval-Based Timbre Control (RBTC) module enables precise manipulation using abstract descriptions without paired data. Experiments confirm TES-VC effectively generates contextually appropriate speech in both timbre and environment with high content retention and superior controllability which demonstrates its potential for widespread applications.
#### Insights on Harmonic Tones from a Generative Music Experiment
 - **Authors:** Emmanuel Deruty, Maarten Grachten
 - **Subjects:** Subjects:
Sound (cs.SD); Human-Computer Interaction (cs.HC); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07073

 - **Pdf link:** https://arxiv.org/pdf/2506.07073

 - **Abstract**
 The ultimate purpose of generative music AI is music production. The studio-lab, a social form within the art-science branch of cross-disciplinarity, is a way to advance music production with AI music models. During a studio-lab experiment involving researchers, music producers, and an AI model for music generating bass-like audio, it was observed that the producers used the model's output to convey two or more pitches with a single harmonic complex tone, which in turn revealed that the model had learned to generate structured and coherent simultaneous melodic lines using monophonic sequences of harmonic complex tones. These findings prompt a reconsideration of the long-standing debate on whether humans can perceive harmonics as distinct pitches and highlight how generative AI can not only enhance musical creativity but also contribute to a deeper understanding of music.
#### E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models
 - **Authors:** Jiaheng Dong, Hong Jia, Soumyajit Chatterjee, Abhirup Ghosh, James Bailey, Ting Dang
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07078

 - **Pdf link:** https://arxiv.org/pdf/2506.07078

 - **Abstract**
 Speech Foundation Models encounter significant performance degradation when deployed in real-world scenarios involving acoustic domain shifts, such as background noise and speaker accents. Test-time adaptation (TTA) has recently emerged as a viable strategy to address such domain shifts at inference time without requiring access to source data or labels. However, existing TTA approaches, particularly those relying on backpropagation, are memory-intensive, limiting their applicability in speech tasks and resource-constrained settings. Although backpropagation-free methods offer improved efficiency, existing ones exhibit poor accuracy. This is because they are predominantly developed for vision tasks, which fundamentally differ from speech task formulations, noise characteristics, and model architecture, posing unique transferability challenges. In this paper, we introduce E-BATS, the first Efficient BAckpropagation-free TTA framework designed explicitly for speech foundation models. E-BATS achieves a balance between adaptation effectiveness and memory efficiency through three key components: (i) lightweight prompt adaptation for a forward-pass-based feature alignment, (ii) a multi-scale loss to capture both global (utterance-level) and local distribution shifts (token-level) and (iii) a test-time exponential moving average mechanism for stable adaptation across utterances. Experiments conducted on four noisy speech datasets spanning sixteen acoustic conditions demonstrate consistent improvements, with 4.1%-13.5% accuracy gains over backpropagation-free baselines and 2.0-6.4 times GPU memory savings compared to backpropagation-based methods. By enabling scalable and robust adaptation under acoustic variability, this work paves the way for developing more efficient adaptation approaches for practical speech processing systems in real-world environments.
#### Streaming Endpointer for Spoken Dialogue using Neural Audio Codecs and Label-Delayed Training
 - **Authors:** Sathvik Udupa, Shinji Watanabe, Petr Schwarz, Jan Cernocky
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07081

 - **Pdf link:** https://arxiv.org/pdf/2506.07081

 - **Abstract**
 Accurate, low-latency endpointing is crucial for effective spoken dialogue systems. While traditional endpointers often rely on spectrum-based audio features, this work proposes real-time speech endpointing for multi-turn dialogues using streaming, low-bitrate Neural Audio Codec (NAC) features, building upon recent advancements in neural audio codecs. To further reduce cutoff errors, we introduce a novel label delay training scheme. At a fixed median latency of 160 ms, our combined NAC and label delay approach achieves significant relative cutoff error reductions: 42.7% for a single-stream endpointer and 37.5% for a two-stream configuration, compared to baseline methods. Finally, we demonstrate efficient integration with a codec-based pretrained speech large language model, improving its median response time by 1200 ms and reducing its cutoff error by 35%.
#### RBA-FE: A Robust Brain-Inspired Audio Feature Extractor for Depression Diagnosis
 - **Authors:** Yu-Xuan Wu, Ziyan Huang, Bin Hu, Zhi-Hong Guan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07118

 - **Pdf link:** https://arxiv.org/pdf/2506.07118

 - **Abstract**
 This article proposes a robust brain-inspired audio feature extractor (RBA-FE) model for depression diagnosis, using an improved hierarchical network architecture. Most deep learning models achieve state-of-the-art performance for image-based diagnostic tasks, ignoring the counterpart audio features. In order to tailor the noise challenge, RBA-FE leverages six acoustic features extracted from the raw audio, capturing both spatial characteristics and temporal dependencies. This hybrid attribute helps alleviate the precision limitation in audio feature extraction within other learning models like deep residual shrinkage networks. To deal with the noise issues, our model incorporates an improved spiking neuron model, called adaptive rate smooth leaky integrate-and-fire (ARSLIF). The ARSLIF model emulates the mechanism of ``retuning of cellular signal selectivity" in the brain attention systems, which enhances the model robustness against environmental noises in audio data. Experimental results demonstrate that RBA-FE achieves state-of-the-art accuracy on the MODMA dataset, respectively with 0.8750, 0.8974, 0.8750 and 0.8750 in precision, accuracy, recall and F1 score. Extensive experiments on the AVEC2014 and DAIC-WOZ datasets both show enhancements in noise robustness. It is further indicated by comparison that the ARSLIF neuron model suggest the abnormal firing pattern within the feature extraction on depressive audio data, offering brain-inspired interpretability.
#### Technical Report: A Practical Guide to Kaldi ASR Optimization
 - **Authors:** Mengze Hong, Di Jiang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07149

 - **Pdf link:** https://arxiv.org/pdf/2506.07149

 - **Abstract**
 This technical report introduces innovative optimizations for Kaldi-based Automatic Speech Recognition (ASR) systems, focusing on acoustic model enhancement, hyperparameter tuning, and language model efficiency. We developed a custom Conformer block integrated with a multistream TDNN-F structure, enabling superior feature extraction and temporal modeling. Our approach includes advanced data augmentation techniques and dynamic hyperparameter optimization to boost performance and reduce overfitting. Additionally, we propose robust strategies for language model management, employing Bayesian optimization and $n$-gram pruning to ensure relevance and computational efficiency. These systematic improvements significantly elevate ASR accuracy and robustness, outperforming existing methods and offering a scalable solution for diverse speech recognition scenarios. This report underscores the importance of strategic optimizations in maintaining Kaldi's adaptability and competitiveness in rapidly evolving technological landscapes.
#### Audio synthesizer inversion in symmetric parameter spaces with approximately equivariant flow matching
 - **Authors:** Ben Hayes, Charalampos Saitis, György Fazekas
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.07199

 - **Pdf link:** https://arxiv.org/pdf/2506.07199

 - **Abstract**
 Many audio synthesizers can produce the same signal given different parameter configurations, meaning the inversion from sound to parameters is an inherently ill-posed problem. We show that this is largely due to intrinsic symmetries of the synthesizer, and focus in particular on permutation invariance. First, we demonstrate on a synthetic task that regressing point estimates under permutation symmetry degrades performance, even when using a permutation-invariant loss function or symmetry-breaking heuristics. Then, viewing equivalent solutions as modes of a probability distribution, we show that a conditional generative model substantially improves performance. Further, acknowledging the invariance of the implicit parameter distribution, we find that performance is further improved by using a permutation equivariant continuous normalizing flow. To accommodate intricate symmetries in real synthesizers, we also propose a relaxed equivariance strategy that adaptively discovers relevant symmetries from data. Applying our method to Surge XT, a full-featured open source synthesizer used in real world audio production, we find our method outperforms regression and generative baselines across audio reconstruction metrics.
#### Methods for pitch analysis in contemporary popular music: Vitalic's use of tones that do not operate on the principle of acoustic resonance
 - **Authors:** Emmanuel Deruty, Pascal Arbez-Nicolas, David Meredith
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07207

 - **Pdf link:** https://arxiv.org/pdf/2506.07207

 - **Abstract**
 Vitalic is an electronic music producer who has been active since 2001. Vitalic's 2005 track "No Fun" features a main synthesiser part built from a sequence of single inharmonic tones that evoke two simultaneous melodies. This part serves as a starting point for examining Vitalic's use of tones that do not operate on the principle of acoustic resonance. The study considers tones that evoke two or more simultaneous pitches and examines various inharmonic partial layouts. Examples outside Vitalic's music are also provided to suggest that similar tone properties can be found elsewhere in contemporary popular music.
#### Towards Generalized Source Tracing for Codec-Based Deepfake Speech
 - **Authors:** Xuanjun Chen, I-Ming Lin, Lin Zhang, Haibin Wu, Hung-yi Lee, Jyh-Shing Roger Jang
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07294

 - **Pdf link:** https://arxiv.org/pdf/2506.07294

 - **Abstract**
 Recent attempts at source tracing for codec-based deepfake speech (CodecFake), generated by neural audio codec-based speech generation (CoSG) models, have exhibited suboptimal performance. However, how to train source tracing models using simulated CoSG data while maintaining strong performance on real CoSG-generated audio remains an open challenge. In this paper, we show that models trained solely on codec-resynthesized data tend to overfit to non-speech regions and struggle to generalize to unseen content. To mitigate these challenges, we introduce the Semantic-Acoustic Source Tracing Network (SASTNet), which jointly leverages Whisper for semantic feature encoding and Wav2vec2 with AudioMAE for acoustic feature encoding. Our proposed SASTNet achieves state-of-the-art performance on the CoSG test set of the CodecFake+ dataset, demonstrating its effectiveness for reliable source tracing.
#### Speech Recognition on TV Series with Video-guided Post-Correction
 - **Authors:** Haoyuan Yang, Yue Zhang, Liqiang Jing
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07323

 - **Pdf link:** https://arxiv.org/pdf/2506.07323

 - **Abstract**
 Automatic Speech Recognition (ASR) has achieved remarkable success with deep learning, driving advancements in conversational artificial intelligence, media transcription, and assistive technologies. However, ASR systems still struggle in complex environments such as TV series, where overlapping speech, domain-specific terminology, and long-range contextual dependencies pose significant challenges to transcription accuracy. Existing multimodal approaches fail to correct ASR outputs with the rich temporal and contextual information available in video. To address this limitation, we propose a novel multimodal post-correction framework that refines ASR transcriptions by leveraging contextual cues extracted from video. Our framework consists of two stages: ASR Generation and Video-based Post-Correction, where the first stage produces the initial transcript and the second stage corrects errors using Video-based Contextual Information Extraction and Context-aware ASR Correction. We employ the Video-Large Multimodal Model (VLMM) to extract key contextual information using tailored prompts, which is then integrated with a Large Language Model (LLM) to refine the ASR output. We evaluate our method on a multimodal benchmark for TV series ASR and demonstrate its effectiveness in improving ASR performance by leveraging video-based context to enhance transcription accuracy in complex multimedia environments.
#### Lightweight Joint Audio-Visual Deepfake Detection via Single-Stream Multi-Modal Learning Framework
 - **Authors:** Kuiyuan Zhang, Wenjie Pei, Rushi Lan, Yifang Guo, Zhongyun Hua
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07358

 - **Pdf link:** https://arxiv.org/pdf/2506.07358

 - **Abstract**
 Deepfakes are AI-synthesized multimedia data that may be abused for spreading misinformation. Deepfake generation involves both visual and audio manipulation. To detect audio-visual deepfakes, previous studies commonly employ two relatively independent sub-models to learn audio and visual features, respectively, and fuse them subsequently for deepfake detection. However, this may underutilize the inherent correlations between audio and visual features. Moreover, utilizing two isolated feature learning sub-models can result in redundant neural layers, making the overall model inefficient and impractical for resource-constrained environments. In this work, we design a lightweight network for audio-visual deepfake detection via a single-stream multi-modal learning framework. Specifically, we introduce a collaborative audio-visual learning block to efficiently integrate multi-modal information while learning the visual and audio features. By iteratively employing this block, our single-stream network achieves a continuous fusion of multi-modal features across its layers. Thus, our network efficiently captures visual and audio features without the need for excessive block stacking, resulting in a lightweight network design. Furthermore, we propose a multi-modal classification module that can boost the dependence of the visual and audio classifiers on modality content. It also enhances the whole resistance of the video classifier against the mismatches between audio and visual modalities. We conduct experiments on the DF-TIMIT, FakeAVCeleb, and DFDC benchmark datasets. Compared to state-of-the-art audio-visual joint detection methods, our method is significantly lightweight with only 0.48M parameters, yet it achieves superiority in both uni-modal and multi-modal deepfakes, as well as in unseen types of deepfakes.
#### Towards Energy-Efficient and Low-Latency Voice-Controlled Smart Homes: A Proposal for Offline Speech Recognition and IoT Integration
 - **Authors:** Peng Huang, Imdad Ullah, Xiaotong Wei, Tariq Ahamed Ahanger, Najm Hassan, Zawar Hussain Shah
 - **Subjects:** Subjects:
Sound (cs.SD); Computers and Society (cs.CY); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07494

 - **Pdf link:** https://arxiv.org/pdf/2506.07494

 - **Abstract**
 The smart home systems, based on AI speech recognition and IoT technology, enable people to control devices through verbal commands and make people's lives more efficient. However, existing AI speech recognition services are primarily deployed on cloud platforms on the Internet. When users issue a command, speech recognition devices like ``Amazon Echo'' will post a recording through numerous network nodes, reach multiple servers, and then receive responses through the Internet. This mechanism presents several issues, including unnecessary energy consumption, communication latency, and the risk of a single-point failure. In this position paper, we propose a smart home concept based on offline speech recognition and IoT technology: 1) integrating offline keyword spotting (KWS) technologies into household appliances with limited resource hardware to enable them to understand user voice commands; 2) designing a local IoT network with decentralized architecture to manage and connect various devices, enhancing the robustness and scalability of the system. This proposal of a smart home based on offline speech recognition and IoT technology will allow users to use low-latency voice control anywhere in the home without depending on the Internet and provide better scalability and energy sustainability.
#### LeVo: High-Quality Song Generation with Multi-Preference Alignment
 - **Authors:** Shun Lei, Yaoxun Xu, Zhiwei Lin, Huaicheng Zhang, Wei Tan, Hangting Chen, Jianwei Yu, Yixuan Zhang, Chenyu Yang, Haina Zhu, Shuai Wang, Zhiyong Wu, Dong Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07520

 - **Pdf link:** https://arxiv.org/pdf/2506.07520

 - **Abstract**
 Recent advances in large language models (LLMs) and audio language models have significantly improved music generation, particularly in lyrics-to-song generation. However, existing approaches still struggle with the complex composition of songs and the scarcity of high-quality data, leading to limitations in sound quality, musicality, instruction following, and vocal-instrument harmony. To address these challenges, we introduce LeVo, an LM-based framework consisting of LeLM and a music codec. LeLM is capable of parallelly modeling two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. It employs two decoder-only transformers and a modular extension training strategy to prevent interference between different token types. To further enhance musicality and instruction following, we introduce a multi-preference alignment method based on Direct Preference Optimization (DPO). This method handles diverse human preferences through a semi-automatic data construction process and DPO post-training. Experimental results demonstrate that LeVo consistently outperforms existing methods on both objective and subjective metrics. Ablation studies further justify the effectiveness of our designs. Audio examples are available at this https URL.
#### Generative Voice Bursts during Phone Call
 - **Authors:** Paritosh Ranjan, Surajit Majumder, Prodip Roy
 - **Subjects:** Subjects:
Sound (cs.SD); Neural and Evolutionary Computing (cs.NE); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07526

 - **Pdf link:** https://arxiv.org/pdf/2506.07526

 - **Abstract**
 In critical situations, conventional mobile telephony fails to convey emergency voice messages to a callee already engaged in another call. The standard call waiting alert does not provide the urgency or content of the waiting call. This paper proposes a novel method for transmitting Generative Voice Bursts short, context aware audio messages during ongoing calls, from either preauthorized or dynamically prioritized callers. By leveraging generative AI techniques, the system automatically generates spoken messages from contextual inputs example like location, health data, images, background noise when the caller is unable to speak due to incapacitation or environmental constraints. The solution incorporates voice, text, and priority inference mechanisms, allowing high priority emergency messages to bypass conventional call waiting barriers. The approach employs models such as GPT Neo for generative text, which is synthesized into audio and delivered in configurable intervals G seconds and counts N times, ensuring minimal disruption while preserving urgency. This method holds potential for significant impact across telecom, mobile device manufacturing, and emergency communication platforms.
#### Transcript-Prompted Whisper with Dictionary-Enhanced Decoding for Japanese Speech Annotation
 - **Authors:** Rui Hu, Xiaolong Lin, Jiawang Liu, Shixi Huang, Zhenpeng Zhan
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07646

 - **Pdf link:** https://arxiv.org/pdf/2506.07646

 - **Abstract**
 In this paper, we propose a method for annotating phonemic and prosodic labels on a given audio-transcript pair, aimed at constructing Japanese text-to-speech (TTS) datasets. Our approach involves fine-tuning a large-scale pre-trained automatic speech recognition (ASR) model, conditioned on ground truth transcripts, to simultaneously output phrase-level graphemes and annotation labels. To further correct errors in phonemic labeling, we employ a decoding strategy that utilizes dictionary prior knowledge. The objective evaluation results demonstrate that our proposed method outperforms previous approaches relying solely on text or audio. The subjective evaluation results indicate that the naturalness of speech synthesized by the TTS model, trained with labels annotated using our method, is comparable to that of a model trained with manual annotations.
#### W4S4: WaLRUS Meets S4 for Long-Range Sequence Modeling
 - **Authors:** Hossein Babaei, Mel White, Richard G. Baraniuk
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.07920

 - **Pdf link:** https://arxiv.org/pdf/2506.07920

 - **Abstract**
 State Space Models (SSMs) have emerged as powerful components for sequence modeling, enabling efficient handling of long-range dependencies via linear recurrence and convolutional computation. However, their effectiveness depends heavily on the choice and initialization of the state matrix. In this work, we build on the SaFARi framework and existing WaLRUS SSMs to introduce a new variant, W4S4 (WaLRUS for S4), a new class of SSMs constructed from redundant wavelet frames. WaLRUS admits a stable diagonalization and supports fast kernel computation without requiring low-rank approximations, making it both theoretically grounded and computationally efficient. We show that WaLRUS retains information over long horizons significantly better than HiPPO-based SSMs, both in isolation and when integrated into deep architectures such as S4. Our experiments demonstrate consistent improvements across delay reconstruction tasks, classification benchmarks, and long-range sequence modeling, confirming that high-quality, structured initialization enabled by wavelet-based state dynamic offers substantial advantages over existing alternatives. WaLRUS provides a scalable and versatile foundation for the next generation of deep SSM-based models.


by Zyzzyva0381 (Windy). 


2025-06-10
