# Showing new listings for Wednesday, 18 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 12papers 
#### Improving Practical Aspects of End-to-End Multi-Talker Speech Recognition for Online and Offline Scenarios
 - **Authors:** Aswin Shanmugam Subramanian, Amit Das, Naoyuki Kanda, Jinyu Li, Xiaofei Wang, Yifan Gong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.14204

 - **Pdf link:** https://arxiv.org/pdf/2506.14204

 - **Abstract**
 We extend the frameworks of Serialized Output Training (SOT) to address practical needs of both streaming and offline automatic speech recognition (ASR) applications. Our approach focuses on balancing latency and accuracy, catering to real-time captioning and summarization requirements. We propose several key improvements: (1) Leveraging Continuous Speech Separation (CSS) single-channel front-end with end-to-end (E2E) systems for highly overlapping scenarios, challenging the conventional wisdom of E2E versus cascaded setups. The CSS framework improves the accuracy of the ASR system by separating overlapped speech from multiple speakers. (2) Implementing dual models -- Conformer Transducer for streaming and Sequence-to-Sequence for offline -- or alternatively, a two-pass model based on cascaded encoders. (3) Exploring segment-based SOT (segSOT) which is better suited for offline scenarios while also enhancing readability of multi-talker transcriptions.
#### A Survey on World Models Grounded in Acoustic Physical Information
 - **Authors:** Xiaoliang Chen, Le Chang, Xin Yu, Yunhe Huang, Xianling Tu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Robotics (cs.RO); Audio and Speech Processing (eess.AS); Applied Physics (physics.app-ph)
 - **Arxiv link:** https://arxiv.org/abs/2506.13833

 - **Pdf link:** https://arxiv.org/pdf/2506.13833

 - **Abstract**
 This survey provides a comprehensive overview of the emerging field of world models grounded in the foundation of acoustic physical information. It examines the theoretical underpinnings, essential methodological frameworks, and recent technological advancements in leveraging acoustic signals for high-fidelity environmental perception, causal physical reasoning, and predictive simulation of dynamic events. The survey explains how acoustic signals, as direct carriers of mechanical wave energy from physical events, encode rich, latent information about material properties, internal geometric structures, and complex interaction dynamics. Specifically, this survey establishes the theoretical foundation by explaining how fundamental physical laws govern the encoding of physical information within acoustic signals. It then reviews the core methodological pillars, including Physics-Informed Neural Networks (PINNs), generative models, and self-supervised multimodal learning frameworks. Furthermore, the survey details the significant applications of acoustic world models in robotics, autonomous driving, healthcare, and finance. Finally, it systematically outlines the important technical and ethical challenges while proposing a concrete roadmap for future research directions toward robust, causal, uncertainty-aware, and responsible acoustic intelligence. These elements collectively point to a research pathway towards embodied active acoustic intelligence, empowering AI systems to construct an internal "intuitive physics" engine through sound.
#### Making deep neural networks work for medical audio: representation, compression and domain adaptation
 - **Authors:** Charles C Onu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.13970

 - **Pdf link:** https://arxiv.org/pdf/2506.13970

 - **Abstract**
 This thesis addresses the technical challenges of applying machine learning to understand and interpret medical audio signals. The sounds of our lungs, heart, and voice convey vital information about our health. Yet, in contemporary medicine, these sounds are primarily analyzed through auditory interpretation by experts using devices like stethoscopes. Automated analysis offers the potential to standardize the processing of medical sounds, enable screening in low-resource settings where physicians are scarce, and detect subtle patterns that may elude human perception, thereby facilitating early diagnosis and treatment. Focusing on the analysis of infant cry sounds to predict medical conditions, this thesis contributes on four key fronts. First, in low-data settings, we demonstrate that large databases of adult speech can be harnessed through neural transfer learning to develop more accurate and robust models for infant cry analysis. Second, in cost-effective modeling, we introduce an end-to-end model compression approach for recurrent networks using tensor decomposition. Our method requires no post-hoc processing, achieves compression rates of several hundred-fold, and delivers accurate, portable models suitable for resource-constrained devices. Third, we propose novel domain adaptation techniques tailored for audio models and adapt existing methods from computer vision. These approaches address dataset bias and enhance generalization across domains while maintaining strong performance on the original data. Finally, to advance research in this domain, we release a unique, open-source dataset of infant cry sounds, developed in collaboration with clinicians worldwide. This work lays the foundation for recognizing the infant cry as a vital sign and highlights the transformative potential of AI-driven audio monitoring in shaping the future of accessible and affordable healthcare.
#### Acoustic scattering AI for non-invasive object classifications: A case study on hair assessment
 - **Authors:** Long-Vu Hoang, Tuan Nguyen, Tran Huy Dat
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.14148

 - **Pdf link:** https://arxiv.org/pdf/2506.14148

 - **Abstract**
 This paper presents a novel non-invasive object classification approach using acoustic scattering, demonstrated through a case study on hair assessment. When an incident wave interacts with an object, it generates a scattered acoustic field encoding structural and material properties. By emitting acoustic stimuli and capturing the scattered signals from head-with-hair-sample objects, we classify hair type and moisture using AI-driven, deep-learning-based sound classification. We benchmark comprehensive methods, including (i) fully supervised deep learning, (ii) embedding-based classification, (iii) supervised foundation model fine-tuning, and (iv) self-supervised model fine-tuning. Our best strategy achieves nearly 90% classification accuracy by fine-tuning all parameters of a self-supervised model. These results highlight acoustic scattering as a privacy-preserving, non-contact alternative to visual classification, opening huge potential for applications in various industries.
#### Pushing the Performance of Synthetic Speech Detection with Kolmogorov-Arnold Networks and Self-Supervised Learning Models
 - **Authors:** Tuan Dat Phuong, Long-Vu Hoang, Huy Dat Tran
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.14153

 - **Pdf link:** https://arxiv.org/pdf/2506.14153

 - **Abstract**
 Recent advancements in speech synthesis technologies have led to increasingly advanced spoofing attacks, posing significant challenges for automatic speaker verification systems. While systems based on self-supervised learning (SSL) models, particularly the XLSR-Conformer model, have demonstrated remarkable performance in synthetic speech detection, there remains room for architectural improvements. In this paper, we propose a novel approach that replaces the traditional Multi-Layer Perceptron in the XLSR-Conformer model with a Kolmogorov-Arnold Network (KAN), a novel architecture based on the Kolmogorov-Arnold representation theorem. Our results on ASVspoof2021 demonstrate that integrating KAN into the SSL-based models can improve the performance by 60.55% relatively on LA and DF sets, further achieving 0.70% EER on the 21LA set. These findings suggest that incorporating KAN into SSL-based models is a promising direction for advances in synthetic speech detection.
#### Can we train ASR systems on Code-switch without real code-switch data? Case study for Singapore's languages
 - **Authors:** Tuan Nguyen, Huy-Dat Tran
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.14177

 - **Pdf link:** https://arxiv.org/pdf/2506.14177

 - **Abstract**
 Code-switching (CS), common in multilingual settings, presents challenges for ASR due to scarce and costly transcribed data caused by linguistic complexity. This study investigates building CS-ASR using synthetic CS data. We propose a phrase-level mixing method to generate synthetic CS data that mimics natural patterns. Utilizing monolingual augmented with synthetic phrase-mixed CS data to fine-tune large pretrained ASR models (Whisper, MMS, SeamlessM4T). This paper focuses on three under-resourced Southeast Asian language pairs: Malay-English (BM-EN), Mandarin-Malay (ZH-BM), and Tamil-English (TA-EN), establishing a new comprehensive benchmark for CS-ASR to evaluate the performance of leading ASR models. Experimental results show that the proposed training strategy enhances ASR performance on monolingual and CS tests, with BM-EN showing highest gains, then TA-EN and ZH-BM. This finding offers a cost-effective approach for CS-ASR development, benefiting research and industry.
#### AsyncSwitch: Asynchronous Text-Speech Adaptation for Code-Switched ASR
 - **Authors:** Tuan Nguyen, Huy-Dat Tran
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.14190

 - **Pdf link:** https://arxiv.org/pdf/2506.14190

 - **Abstract**
 Developing code-switched ASR systems is challenging due to language ambiguity and limited exposure to multilingual, code-switched data, while collecting such speech is costly. Prior work generates synthetic audio from text, but these methods are computationally intensive and hard to scale. We introduce AsyncSwitch, a novel asynchronous adaptation framework that leverages large-scale, text-rich web data to pre-expose ASR models to diverse code-switched domains before fine-tuning on paired speech-text corpora. Our three-stage process (1) trains decoder self-attention and feedforward layers on code-switched text, (2) aligns decoder and encoder via cross-attention using limited speech-text data, and (3) fully fine-tunes the entire model. Experiments with Whisper on Malay-English code-switching demonstrate a 9.02% relative WER reduction, while improving monolingual performance in Singlish, Malay, and other English variants.
#### Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription
 - **Authors:** Anna Hamberger, Sebastian Murgul, Jochen Schmidt, Michael Heizmann
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.14223

 - **Pdf link:** https://arxiv.org/pdf/2506.14223

 - **Abstract**
 Music transcription plays a pivotal role in Music Information Retrieval (MIR), particularly for stringed instruments like the guitar, where symbolic music notations such as MIDI lack crucial playability information. This contribution introduces the Fretting-Transformer, an encoderdecoder model that utilizes a T5 transformer architecture to automate the transcription of MIDI sequences into guitar tablature. By framing the task as a symbolic translation problem, the model addresses key challenges, including string-fret ambiguity and physical playability. The proposed system leverages diverse datasets, including DadaGP, GuitarToday, and Leduc, with novel data pre-processing and tokenization strategies. We have developed metrics for tablature accuracy and playability to quantitatively evaluate the performance. The experimental results demonstrate that the Fretting-Transformer surpasses baseline methods like A* and commercial applications like Guitar Pro. The integration of context-sensitive processing and tuning/capo conditioning further enhances the model's performance, laying a robust foundation for future developments in automated guitar transcription.
#### SLEEPING-DISCO 9M: A large-scale pre-training dataset for generative music modeling
 - **Authors:** Tawsif Ahmed, Andrej Radonjic, Gollam Rabby
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.14293

 - **Pdf link:** https://arxiv.org/pdf/2506.14293

 - **Abstract**
 We present Sleeping-DISCO 9M, a large-scale pre-training dataset for music and song. To the best of our knowledge, there are no open-source high-quality dataset representing popular and well-known songs for generative music modeling tasks such as text-music, music-captioning, singing-voice synthesis, melody reconstruction and cross-model retrieval. Past contributions focused on isolated and constrained factors whose core perspective was to create synthetic or re-recorded music corpus (e.g. GTSinger, M4Singer) and arbitrarily large-scale audio datasets (e.g. DISCO-10M and LAIONDISCO-12M) had been another focus for the community. Unfortunately, adoption of these datasets has been below substantial in the generative music community as these datasets fail to reflect real-world music and its flavour. Our dataset changes this narrative and provides a dataset that is constructed using actual popular music and world-renowned artists.
#### Unifying Streaming and Non-streaming Zipformer-based ASR
 - **Authors:** Bidisha Sharma, Karthik Pandia Durai, Shankar Venkatesan, Jeena J Prakash, Shashi Kumar, Malolan Chetlur, Andreas Stolcke
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.14434

 - **Pdf link:** https://arxiv.org/pdf/2506.14434

 - **Abstract**
 There has been increasing interest in unifying streaming and non-streaming automatic speech recognition (ASR) models to reduce development, training, and deployment costs. We present a unified framework that trains a single end-to-end ASR model for both streaming and non-streaming applications, leveraging future context information. We propose to use dynamic right-context through the chunked attention masking in the training of zipformer-based ASR models. We demonstrate that using right-context is more effective in zipformer models compared to other conformer models due to its multi-scale nature. We analyze the effect of varying the number of right-context frames on accuracy and latency of the streaming ASR models. We use Librispeech and large in-house conversational datasets to train different versions of streaming and non-streaming models and evaluate them in a production grade server-client setup across diverse testsets of different domains. The proposed strategy reduces word error by relative 7.9\% with a small degradation in user-perceived latency. By adding more right-context frames, we are able to achieve streaming performance close to that of non-streaming models. Our approach also allows flexible control of the latency-accuracy tradeoff according to customers requirements.
#### An Open Research Dataset of the 1932 Cairo Congress of Arab Music
 - **Authors:** Baris Bozkurt (College of Interdisciplinary Studies, Zayed University, Dubai, United Arab Emirates)
 - **Subjects:** Subjects:
Sound (cs.SD); Digital Libraries (cs.DL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.14503

 - **Pdf link:** https://arxiv.org/pdf/2506.14503

 - **Abstract**
 This paper introduces ORD-CC32 , an open research dataset derived from the 1932 Cairo Congress of Arab Music recordings, a historically significant collection representing diverse Arab musical traditions. The dataset includes structured metadata, melodic and rhythmic mode tags (maqam and iqa), manually labeled tonic information, and acoustic features extracted using state-of-the-art pitch detection methods. These resources support computational studies of tuning, temperament, and regional variations in Arab music. A case study using pitch histograms demonstrates the potential for data-driven analysis of microtonal differences across regions. By making this dataset openly available, we aim to enable interdisciplinary research in computational ethnomusicology, music information retrieval (MIR), cultural studies, and digital heritage preservation. ORD-CC32 is shared on Zenodo with tools for feature extraction and metadata retrieval.
#### A Variational Framework for Improving Naturalness in Generative Spoken Language Models
 - **Authors:** Li-Wei Chen, Takuya Higuchi, Zakaria Aldeneh, Ahmed Hussen Abdelaziz, Alexander Rudnicky
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.14767

 - **Pdf link:** https://arxiv.org/pdf/2506.14767

 - **Abstract**
 The success of large language models in text processing has inspired their adaptation to speech modeling. However, since speech is continuous and complex, it is often discretized for autoregressive modeling. Speech tokens derived from self-supervised models (known as semantic tokens) typically focus on the linguistic aspects of speech but neglect prosodic information. As a result, models trained on these tokens can generate speech with reduced naturalness. Existing approaches try to fix this by adding pitch features to the semantic tokens. However, pitch alone cannot fully represent the range of paralinguistic attributes, and selecting the right features requires careful hand-engineering. To overcome this, we propose an end-to-end variational approach that automatically learns to encode these continuous speech attributes to enhance the semantic tokens. Our approach eliminates the need for manual extraction and selection of paralinguistic features. Moreover, it produces preferred speech continuations according to human raters. Code, samples and models are available at this https URL.


by Zyzzyva0381 (Windy). 


2025-06-18
