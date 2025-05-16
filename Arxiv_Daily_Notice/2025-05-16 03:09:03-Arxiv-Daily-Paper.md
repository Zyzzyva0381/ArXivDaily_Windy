# Showing new listings for Friday, 16 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 8papers 
#### Who Said What WSW 2.0? Enhanced Automated Analysis of Preschool Classroom Speech
 - **Authors:** Anchen Sun, Tiantian Feng, Gabriela Gutierrez, Juan J Londono, Anfeng Xu, Batya Elbaum, Shrikanth Narayanan, Lynn K Perry, Daniel S Messinger
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2505.09972

 - **Pdf link:** https://arxiv.org/pdf/2505.09972

 - **Abstract**
 This paper introduces an automated framework WSW2.0 for analyzing vocal interactions in preschool classrooms, enhancing both accuracy and scalability through the integration of wav2vec2-based speaker classification and Whisper (large-v2 and large-v3) speech transcription. A total of 235 minutes of audio recordings (160 minutes from 12 children and 75 minutes from 5 teachers), were used to compare system outputs to expert human annotations. WSW2.0 achieves a weighted F1 score of .845, accuracy of .846, and an error-corrected kappa of .672 for speaker classification (child vs. teacher). Transcription quality is moderate to high with word error rates of .119 for teachers and .238 for children. WSW2.0 exhibits relatively high absolute agreement intraclass correlations (ICC) with expert transcriptions for a range of classroom language features. These include teacher and child mean utterance length, lexical diversity, question asking, and responses to questions and other utterances, which show absolute agreement intraclass correlations between .64 and .98. To establish scalability, we apply the framework to an extensive dataset spanning two years and over 1,592 hours of classroom audio recordings, demonstrating the framework's robustness for broad real-world applications. These findings highlight the potential of deep learning and natural language processing techniques to revolutionize educational research by providing accurate measures of key features of preschool classroom speech, ultimately guiding more effective intervention strategies and supporting early childhood language development.
#### Spatially Selective Active Noise Control for Open-fitting Hearables with Acausal Optimization
 - **Authors:** Tong Xiao, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP); Systems and Control (eess.SY)
 - **Arxiv link:** https://arxiv.org/abs/2505.10372

 - **Pdf link:** https://arxiv.org/pdf/2505.10372

 - **Abstract**
 Recent advances in active noise control have enabled the development of hearables with spatial selectivity, which actively suppress undesired noise while preserving desired sound from specific directions. In this work, we propose an improved approach to spatially selective active noise control that incorporates acausal relative impulse responses into the optimization process, resulting in significantly improved performance over the causal design. We evaluate the system through simulations using a pair of open-fitting hearables with spatially localized speech and noise sources in an anechoic environment. Performance is evaluated in terms of speech distortion, noise reduction, and signal-to-noise ratio improvement across different delays and degrees of acausality. Results show that the proposed acausal optimization consistently outperforms the causal approach across all metrics and scenarios, as acausal filters more effectively characterize the response of the desired source.
#### Quantized Approximate Signal Processing (QASP): Towards Homomorphic Encryption for audio
 - **Authors:** Tu Duyen Nguyen, Adrien Lesage, Clotilde Cantini, Rachid Riad
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Cryptography and Security (cs.CR)
 - **Arxiv link:** https://arxiv.org/abs/2505.10500

 - **Pdf link:** https://arxiv.org/pdf/2505.10500

 - **Abstract**
 Audio and speech data are increasingly used in machine learning applications such as speech recognition, speaker identification, and mental health monitoring. However, the passive collection of this data by audio listening devices raises significant privacy concerns. Fully homomorphic encryption (FHE) offers a promising solution by enabling computations on encrypted data and preserving user privacy. Despite its potential, prior attempts to apply FHE to audio processing have faced challenges, particularly in securely computing time frequency representations, a critical step in many audio tasks. Here, we addressed this gap by introducing a fully secure pipeline that computes, with FHE and quantized neural network operations, four fundamental time-frequency representations: Short-Time Fourier Transform (STFT), Mel filterbanks, Mel-frequency cepstral coefficients (MFCCs), and gammatone filters. Our methods also support the private computation of audio descriptors and convolutional neural network (CNN) classifiers. Besides, we proposed approximate STFT algorithms that lighten computation and bit use for statistical and machine learning analyses. We ran experiments on the VocalSet and OxVoc datasets demonstrating the fully private computation of our approach. We showed significant performance improvements with STFT approximation in private statistical analysis of audio markers, and for vocal exercise classification with CNNs. Our results reveal that our approximations substantially reduce error rates compared to conventional STFT implementations in FHE. We also demonstrated a fully private classification based on the raw audio for gender and vocal exercise classification. Finally, we provided a practical heuristic for parameter selection, making quantized approximate signal processing accessible to researchers and practitioners aiming to protect sensitive audio data.
#### SpecWav-Attack: Leveraging Spectrogram Resizing and Wav2Vec 2.0 for Attacking Anonymized Speech
 - **Authors:** Yuqi Li, Yuanzhong Zheng, Zhongtian Guo, Yaoxuan Wang, Jianjun Yin, Haojun Fei
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.09616

 - **Pdf link:** https://arxiv.org/pdf/2505.09616

 - **Abstract**
 This paper presents SpecWav-Attack, an adversarial model for detecting speakers in anonymized speech. It leverages Wav2Vec2 for feature extraction and incorporates spectrogram resizing and incremental training for improved performance. Evaluated on librispeech-dev and librispeech-test, SpecWav-Attack outperforms conventional attacks, revealing vulnerabilities in anonymized speech systems and emphasizing the need for stronger defenses, benchmarked against the ICASSP 2025 Attacker Challenge.
#### Introducing voice timbre attribute detection
 - **Authors:** Jinghao He, Zhengyan Sheng, Liping Chen, Kong Aik Lee, Zhen-Hua Ling
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.09661

 - **Pdf link:** https://arxiv.org/pdf/2505.09661

 - **Abstract**
 This paper focuses on explaining the timbre conveyed by speech signals and introduces a task termed voice timbre attribute detection (vTAD). In this task, voice timbre is explained with a set of sensory attributes describing its human perception. A pair of speech utterances is processed, and their intensity is compared in a designated timbre descriptor. Moreover, a framework is proposed, which is built upon the speaker embeddings extracted from the speech utterances. The investigation is conducted on the VCTK-RVA dataset. Experimental examinations on the ECAPA-TDNN and FACodec speaker encoders demonstrated that: 1) the ECAPA-TDNN speaker encoder was more capable in the seen scenario, where the testing speakers were included in the training set; 2) the FACodec speaker encoder was superior in the unseen scenario, where the testing speakers were not part of the training, indicating enhanced generalization capability. The VCTK-RVA dataset and open-source code are available on the website this https URL.
#### Theoretical Model of Acoustic Power Transfer Through Solids
 - **Authors:** Ippokratis Kochliaridis, Michail E. Kiziroglou
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Applied Physics (physics.app-ph)
 - **Arxiv link:** https://arxiv.org/abs/2505.09784

 - **Pdf link:** https://arxiv.org/pdf/2505.09784

 - **Abstract**
 Acoustic Power Transfer is a relatively new technology. It is a modern type of a wireless interface, where data signals and supply voltages are transmitted, with the use of mechanical waves, through a medium. The simplest application of such systems is the measurement of frequency response for audio speakers. It consists of a variable signal generator, a measuring amplifier which drives an acoustic source and the loudspeaker driver. The receiver contains a microphone circuit with a level recorder. Acoustic Power Transfer could have many applications, such as: Cochlear Implants, Sonar Systems and Wireless Charging. However, it is a new technology, thus it needs further investigation.
#### ListenNet: A Lightweight Spatio-Temporal Enhancement Nested Network for Auditory Attention Detection
 - **Authors:** Cunhang Fan, Xiaoke Yang, Hongyu Zhang, Ying Chen, Lu Li, Jian Zhou, Zhao Lv
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.10348

 - **Pdf link:** https://arxiv.org/pdf/2505.10348

 - **Abstract**
 Auditory attention detection (AAD) aims to identify the direction of the attended speaker in multi-speaker environments from brain signals, such as Electroencephalography (EEG) signals. However, existing EEG-based AAD methods overlook the spatio-temporal dependencies of EEG signals, limiting their decoding and generalization abilities. To address these issues, this paper proposes a Lightweight Spatio-Temporal Enhancement Nested Network (ListenNet) for AAD. The ListenNet has three key components: Spatio-temporal Dependency Encoder (STDE), Multi-scale Temporal Enhancement (MSTE), and Cross-Nested Attention (CNA). The STDE reconstructs dependencies between consecutive time windows across channels, improving the robustness of dynamic pattern extraction. The MSTE captures temporal features at multiple scales to represent both fine-grained and long-range temporal patterns. In addition, the CNA integrates hierarchical features more effectively through novel dynamic attention mechanisms to capture deep spatio-temporal correlations. Experimental results on three public datasets demonstrate the superiority of ListenNet over state-of-the-art methods in both subject-dependent and challenging subject-independent settings, while reducing the trainable parameter count by approximately 7 times. Code is available at:this https URL.
#### T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback
 - **Authors:** Zehan Wang, Ke Lei, Chen Zhu, Jiawei Huang, Sashuai Zhou, Luping Liu, Xize Cheng, Shengpeng Ji, Zhenhui Ye, Tao Jin, Zhou Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.10561

 - **Pdf link:** https://arxiv.org/pdf/2505.10561

 - **Abstract**
 Text-to-audio (T2A) generation has achieved remarkable progress in generating a variety of audio outputs from language prompts. However, current state-of-the-art T2A models still struggle to satisfy human preferences for prompt-following and acoustic quality when generating complex multi-event audio. To improve the performance of the model in these high-level applications, we propose to enhance the basic capabilities of the model with AI feedback learning. First, we introduce fine-grained AI audio scoring pipelines to: 1) verify whether each event in the text prompt is present in the audio (Event Occurrence Score), 2) detect deviations in event sequences from the language description (Event Sequence Score), and 3) assess the overall acoustic and harmonic quality of the generated audio (Acoustic&Harmonic Quality). We evaluate these three automatic scoring pipelines and find that they correlate significantly better with human preferences than other evaluation metrics. This highlights their value as both feedback signals and evaluation metrics. Utilizing our robust scoring pipelines, we construct a large audio preference dataset, T2A-FeedBack, which contains 41k prompts and 249k audios, each accompanied by detailed scores. Moreover, we introduce T2A-EpicBench, a benchmark that focuses on long captions, multi-events, and story-telling scenarios, aiming to evaluate the advanced capabilities of T2A models. Finally, we demonstrate how T2A-FeedBack can enhance current state-of-the-art audio model. With simple preference tuning, the audio generation model exhibits significant improvements in both simple (AudioCaps test set) and complex (T2A-EpicBench) scenarios.


by Zyzzyva0381 (Windy). 


2025-05-16
