# Showing new listings for Friday, 30 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 10papers 
#### NGPU-LM: GPU-Accelerated N-Gram Language Model for Context-Biasing in Greedy ASR Decoding
 - **Authors:** Vladimir Bataev, Andrei Andrusenko, Lilit Grigoryan, Aleksandr Laptev, Vitaly Lavrukhin, Boris Ginsburg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.22857

 - **Pdf link:** https://arxiv.org/pdf/2505.22857

 - **Abstract**
 Statistical n-gram language models are widely used for context-biasing tasks in Automatic Speech Recognition (ASR). However, existing implementations lack computational efficiency due to poor parallelization, making context-biasing less appealing for industrial use. This work rethinks data structures for statistical n-gram language models to enable fast and parallel operations for GPU-optimized inference. Our approach, named NGPU-LM, introduces customizable greedy decoding for all major ASR model types - including transducers, attention encoder-decoder models, and CTC - with less than 7% computational overhead. The proposed approach can eliminate more than 50% of the accuracy gap between greedy and beam search for out-of-domain scenarios while avoiding significant slowdown caused by beam search. The implementation of the proposed NGPU-LM is open-sourced.
#### LLM-Synth4KWS: Scalable Automatic Generation and Synthesis of Confusable Data for Custom Keyword Spotting
 - **Authors:** Pai Zhu, Quan Wang, Dhruuv Agarwal, Kurt Partridge
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.22995

 - **Pdf link:** https://arxiv.org/pdf/2505.22995

 - **Abstract**
 Custom keyword spotting (KWS) allows detecting user-defined spoken keywords from streaming audio. This is achieved by comparing the embeddings from voice enrollments and input audio. State-of-the-art custom KWS models are typically trained contrastively using utterances whose keywords are randomly sampled from training dataset. These KWS models often struggle with confusing keywords, such as "blue" versus "glue". This paper introduces an effective way to augment the training with confusable utterances where keywords are generated and grouped from large language models (LLMs), and speech signals are synthesized with diverse speaking styles from text-to-speech (TTS) engines. To better measure user experience on confusable KWS, we define a new northstar metric using the average area under DET curve from confusable groups (c-AUC). Featuring high scalability and zero labor cost, the proposed method improves AUC by 3.7% and c-AUC by 11.3% on the Speech Commands testing set.
#### Interspeech 2025 URGENT Speech Enhancement Challenge
 - **Authors:** Kohei Saijo, Wangyou Zhang, Samuele Cornell, Robin Scheibler, Chenda Li, Zhaoheng Ni, Anurag Kumar, Marvin Sach, Yihui Fu, Wei Wang, Tim Fingscheidt, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.23212

 - **Pdf link:** https://arxiv.org/pdf/2505.23212

 - **Abstract**
 There has been a growing effort to develop universal speech enhancement (SE) to handle inputs with various speech distortions and recording conditions. The URGENT Challenge series aims to foster such universal SE by embracing a broad range of distortion types, increasing data diversity, and incorporating extensive evaluation metrics. This work introduces the Interspeech 2025 URGENT Challenge, the second edition of the series, to explore several aspects that have received limited attention so far: language dependency, universality for more distortion types, data scalability, and the effectiveness of using noisy training data. We received 32 submissions, where the best system uses a discriminative model, while most other competitive ones are hybrid methods. Analysis reveals some key findings: (i) some generative or hybrid approaches are preferred in subjective evaluations over the top discriminative model, and (ii) purely generative SE models can exhibit language dependency.
#### Spoken question answering for visual queries
 - **Authors:** Nimrod Shabtay, Zvi Kons, Avihu Dekel, Hagai Aronowitz, Ron Hoory, Assaf Arbelle
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2505.23308

 - **Pdf link:** https://arxiv.org/pdf/2505.23308

 - **Abstract**
 Question answering (QA) systems are designed to answer natural language questions. Visual QA (VQA) and Spoken QA (SQA) systems extend the textual QA system to accept visual and spoken input respectively. This work aims to create a system that enables user interaction through both speech and images. That is achieved through the fusion of text, speech, and image modalities to tackle the task of spoken VQA (SVQA). The resulting multi-modal model has textual, visual, and spoken inputs and can answer spoken questions on images. Training and evaluating SVQA models requires a dataset for all three modalities, but no such dataset currently exists. We address this problem by synthesizing VQA datasets using two zero-shot TTS models. Our initial findings indicate that a model trained only with synthesized speech nearly reaches the performance of the upper-bounding model trained on textual QAs. In addition, we show that the choice of the TTS model has a minor impact on accuracy.
#### Vision-Integrated High-Quality Neural Speech Coding
 - **Authors:** Yao Guo, Yang Ai, Rui-Chen Zheng, Hui-Peng Du, Xiao-Hang Jiang, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.23379

 - **Pdf link:** https://arxiv.org/pdf/2505.23379

 - **Abstract**
 This paper proposes a novel vision-integrated neural speech codec (VNSC), which aims to enhance speech coding quality by leveraging visual modality information. In VNSC, the image analysis-synthesis module extracts visual features from lip images, while the feature fusion module facilitates interaction between the image analysis-synthesis module and the speech coding module, transmitting visual information to assist the speech coding process. Depending on whether visual information is available during the inference stage, the feature fusion module integrates visual features into the speech coding module using either explicit integration or implicit distillation strategies. Experimental results confirm that integrating visual information effectively improves the quality of the decoded speech and enhances the noise robustness of the neural speech codec, without increasing the bitrate.
#### DeepFilterGAN: A Full-band Real-time Speech Enhancement System with GAN-based Stochastic Regeneration
 - **Authors:** Sanberk Serbest, Tijana Stojkovic, Milos Cernak, Andrew Harper
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2505.23515

 - **Pdf link:** https://arxiv.org/pdf/2505.23515

 - **Abstract**
 In this work, we propose a full-band real-time speech enhancement system with GAN-based stochastic regeneration. Predictive models focus on estimating the mean of the target distribution, whereas generative models aim to learn the full distribution. This behavior of predictive models may lead to over-suppression, i.e. the removal of speech content. In the literature, it was shown that combining a predictive model with a generative one within the stochastic regeneration framework can reduce the distortion in the output. We use this framework to obtain a real-time speech enhancement system. With 3.58M parameters and a low latency, our system is designed for real-time streaming with a lightweight architecture. Experiments show that our system improves over the first stage in terms of NISQA-MOS metric. Finally, through an ablation study, we show the importance of noisy conditioning in our system. We participated in 2025 Urgent Challenge with our model and later made further improvements.
#### StressTest: Can YOUR Speech LM Handle the Stress?
 - **Authors:** Iddo Yosha, Gallil Maimon, Yossi Adi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.22765

 - **Pdf link:** https://arxiv.org/pdf/2505.22765

 - **Abstract**
 Sentence stress refers to emphasis, placed on specific words within a spoken utterance to highlight or contrast an idea, or to introduce new information. It is often used to imply an underlying intention that is not explicitly stated. Recent advances in speech-aware language models (SLMs) have enabled direct processing of audio, allowing models to bypass transcription and access the full richness of the speech signal and perform audio reasoning tasks such as spoken question answering. Despite the crucial role of sentence stress in shaping meaning and speaker intent, it remains largely overlooked in evaluation and development of such models. In this work, we address this gap by introducing StressTest, a benchmark specifically designed to evaluate a model's ability to distinguish between interpretations of spoken sentences based on the stress pattern. We assess the performance of several leading SLMs and find that, despite their overall capabilities, they perform poorly on such tasks. To overcome this limitation, we propose a novel synthetic data generation pipeline, and create Stress17k, a training set that simulates change of meaning implied by stress variation. Then, we empirically show that optimizing models with this synthetic dataset aligns well with real-world recordings and enables effective finetuning of SLMs. Results suggest, that our finetuned model, StresSLM, significantly outperforms existing models on both sentence stress reasoning and detection tasks. Code, models, data, and audio samples - this http URL.
#### EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic, Expressiveness, and Linguistic Challenges Using Model-as-a-Judge
 - **Authors:** Ruskin Raj Manku, Yuzhi Tang, Xingjian Shi, Mu Li, Alex Smola
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.23009

 - **Pdf link:** https://arxiv.org/pdf/2505.23009

 - **Abstract**
 Text-to-Speech (TTS) benchmarks often fail to capture how well models handle nuanced and semantically complex text. Building on $\textit{EmergentTTS}$, we introduce $\textit{EmergentTTS-Eval}$, a comprehensive benchmark covering six challenging TTS scenarios: emotions, paralinguistics, foreign words, syntactic complexity, complex pronunciation (e.g. URLs, formulas), and questions. Crucially, our framework automates both test-case generation and evaluation, making the benchmark easily extensible. Starting from a small set of human-written seed prompts, we iteratively extend them using LLMs to target specific structural, phonetic and prosodic challenges, resulting in 1,645 diverse test cases. Moreover, we employ a model-as-a-judge approach, using a Large Audio Language Model (LALM) to assess the speech across multiple dimensions such as expressed emotion, prosodic, intonational, and pronunciation accuracy. We evaluate state-of-the-art open-source and proprietary TTS systems, such as 11Labs, Deepgram, and OpenAI's 4o-mini-TTS, on EmergentTTS-Eval, demonstrating its ability to reveal fine-grained performance differences. Results show that the model-as-a-judge approach offers robust TTS assessment and a high correlation with human preferences. We open source the evaluation $\href{this https URL}{code}$ and the $\href{this https URL}{dataset}$.
#### Spoken Language Modeling with Duration-Penalized Self-Supervised Units
 - **Authors:** Nicol Visser, Herman Kamper
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.23494

 - **Pdf link:** https://arxiv.org/pdf/2505.23494

 - **Abstract**
 Spoken language models (SLMs) operate on acoustic units obtained by discretizing self-supervised speech representations. Although the characteristics of these units directly affect performance, the interaction between codebook size and unit coarseness (i.e., duration) remains unexplored. We investigate SLM performance as we vary codebook size and unit coarseness using the simple duration-penalized dynamic programming (DPDP) method. New analyses are performed across different linguistic levels. At the phone and word levels, coarseness provides little benefit, as long as the codebook size is chosen appropriately. However, when producing whole sentences in a resynthesis task, SLMs perform better with coarser units. In lexical and syntactic language modeling tasks, coarser units also give higher accuracies at lower bitrates. We therefore show that coarser units aren't always better, but that DPDP is a simple and efficient way to obtain coarser units for the tasks where they are beneficial.
#### Spectrotemporal Modulation: Efficient and Interpretable Feature Representation for Classifying Speech, Music, and Environmental Sounds
 - **Authors:** Andrew Chang, Yike Li, Iran R. Roman, David Poeppel
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.23509

 - **Pdf link:** https://arxiv.org/pdf/2505.23509

 - **Abstract**
 Audio DNNs have demonstrated impressive performance on various machine listening tasks; however, most of their representations are computationally costly and uninterpretable, leaving room for optimization. Here, we propose a novel approach centered on spectrotemporal modulation (STM) features, a signal processing method that mimics the neurophysiological representation in the human auditory cortex. The classification performance of our STM-based model, without any pretraining, is comparable to that of pretrained audio DNNs across diverse naturalistic speech, music, and environmental sounds, which are essential categories for both human cognition and machine perception. These results show that STM is an efficient and interpretable feature representation for audio classification, advancing the development of machine listening and unlocking exciting new possibilities for basic understanding of speech and auditory sciences, as well as developing audio BCI and cognitive computing.


by Zyzzyva0381 (Windy). 


2025-05-30
