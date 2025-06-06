# Showing new listings for Friday, 6 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 16papers 
#### Phi-Omni-ST: A multimodal language model for direct speech-to-speech translation
 - **Authors:** Yuxuan Hu, Haibin Wu, Ruchao Fan, Xiaofei Wang, Heng Lu, Yao Qian, Jinyu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04392

 - **Pdf link:** https://arxiv.org/pdf/2506.04392

 - **Abstract**
 Speech-aware language models (LMs) have demonstrated capabilities in understanding spoken language while generating text-based responses. However, enabling them to produce speech output efficiently and effectively remains a challenge. In this paper, we present Phi-Omni-ST, a multimodal LM for direct speech-to-speech translation (ST), built on the open-source Phi-4 MM model. Phi-Omni-ST extends its predecessor by generating translated speech using an audio transformer head that predicts audio tokens with a delay relative to text tokens, followed by a streaming vocoder for waveform synthesis. Our experimental results on the CVSS-C dataset demonstrate Phi-Omni-ST's superior performance, significantly surpassing existing baseline models trained on the same dataset. Furthermore, when we scale up the training data and the model size, Phi-Omni-ST reaches on-par performance with the current SOTA model.
#### Can we reconstruct a dysarthric voice with the large speech model Parler TTS?
 - **Authors:** Ariadna Sanchez, Simon King
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.04397

 - **Pdf link:** https://arxiv.org/pdf/2506.04397

 - **Abstract**
 Speech disorders can make communication hard or even impossible for those who develop them. Personalised Text-to-Speech is an attractive option as a communication aid. We attempt voice reconstruction using a large speech model, with which we generate an approximation of a dysarthric speaker's voice prior to the onset of their condition. In particular, we investigate whether a state-of-the-art large speech model, Parler TTS, can generate intelligible speech while maintaining speaker identity. We curate a dataset and annotate it with relevant speaker and intelligibility information, and use this to fine-tune the model. Our results show that the model can indeed learn to generate from the distribution of this challenging data, but struggles to control intelligibility and to maintain consistent speaker identity. We propose future directions to improve controllability of this class of model, for the voice reconstruction task.
#### Bringing Interpretability to Neural Audio Codecs
 - **Authors:** Samir Sadok, Julien Hauret, Éric Bavu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04492

 - **Pdf link:** https://arxiv.org/pdf/2506.04492

 - **Abstract**
 The advent of neural audio codecs has increased in popularity due to their potential for efficiently modeling audio with transformers. Such advanced codecs represent audio from a highly continuous waveform to low-sampled discrete units. In contrast to semantic units, acoustic units may lack interpretability because their training objectives primarily focus on reconstruction performance. This paper proposes a two-step approach to explore the encoding of speech information within the codec tokens. The primary goal of the analysis stage is to gain deeper insight into how speech attributes such as content, identity, and pitch are encoded. The synthesis stage then trains an AnCoGen network for post-hoc explanation of codecs to extract speech attributes from the respective tokens directly.
#### French Listening Tests for the Assessment of Intelligibility, Quality, and Identity of Body-Conducted Speech Enhancement
 - **Authors:** Thomas Joubaud, Julien Hauret, Véronique Zimpfer, Éric Bavu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04495

 - **Pdf link:** https://arxiv.org/pdf/2506.04495

 - **Abstract**
 This study evaluates the Extreme Bandwidth Extension Network (EBEN) model on body-conduction sensors through listening tests. Using the Vibravox dataset, we assess intelligibility with a French Modified Rhyme Test, speech quality with a MUSHRA (MUltiple Stimuli with Hidden Reference and Anchor) protocol and speaker identity preservation with an A/B identification task. The experiments involved male and female speakers recorded with a forehead accelerometer, rigid in-ear and throat microphones. The results confirm that EBEN enhances both speech quality and intelligibility. It slightly degrades speaker identification performance when applied to female speakers' throat microphone recordings. The findings also demonstrate a correlation between Short-Time Objective Intelligibility (STOI) and perceived quality in body-conducted speech, while speaker verification using ECAPA2-TDNN aligns well with identification performance. No tested metric reliably predicts EBEN's effect on intelligibility.
#### Towards Efficient Speech-Text Jointly Decoding within One Speech Language Model
 - **Authors:** Haibin Wu, Yuxuan Hu, Ruchao Fan, Xiaofei Wang, Kenichi Kumatani, Bo Ren, Jianwei Yu, Heng Lu, Lijuan Wang, Yao Qian, Jinyu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2506.04518

 - **Pdf link:** https://arxiv.org/pdf/2506.04518

 - **Abstract**
 Speech language models (Speech LMs) enable end-to-end speech-text modelling within a single model, offering a promising direction for spoken dialogue systems. The choice of speech-text jointly decoding paradigm plays a critical role in performance, efficiency, and alignment quality. In this work, we systematically compare representative joint speech-text decoding strategies-including the interleaved, and parallel generation paradigms-under a controlled experimental setup using the same base language model, speech tokenizer and training data. Our results show that the interleaved approach achieves the best alignment. However it suffers from slow inference due to long token sequence length. To address this, we propose a novel early-stop interleaved (ESI) pattern that not only significantly accelerates decoding but also yields slightly better performance. Additionally, we curate high-quality question answering (QA) datasets to further improve speech QA performance.
#### EMO-Debias: Benchmarking Gender Debiasing Techniques in Multi-Label Speech Emotion Recognition
 - **Authors:** Yi-Cheng Lin, Huang-Cheng Chou, Yu-Hsuan Li Liang, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2506.04652

 - **Pdf link:** https://arxiv.org/pdf/2506.04652

 - **Abstract**
 Speech emotion recognition (SER) systems often exhibit gender bias. However, the effectiveness and robustness of existing debiasing methods in such multi-label scenarios remain underexplored. To address this gap, we present EMO-Debias, a large-scale comparison of 13 debiasing methods applied to multi-label SER. Our study encompasses techniques from pre-processing, regularization, adversarial learning, biased learners, and distributionally robust optimization. Experiments conducted on acted and naturalistic emotion datasets, using WavLM and XLSR representations, evaluate each method under conditions of gender imbalance. Our analysis quantifies the trade-offs between fairness and accuracy, identifying which approaches consistently reduce gender performance gaps without compromising overall model performance. The findings provide actionable insights for selecting effective debiasing strategies and highlight the impact of dataset distributions.
#### Multivariate Probabilistic Assessment of Speech Quality
 - **Authors:** Fredrik Cumlin, Xinyu Liang, Victor Ungureanu, Chandan K. A. Reddy, Christian Schüldt, Saikat Chatterjee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04890

 - **Pdf link:** https://arxiv.org/pdf/2506.04890

 - **Abstract**
 The mean opinion score (MOS) is a standard metric for assessing speech quality, but its singular focus fails to identify specific distortions when low scores are observed. The NISQA dataset addresses this limitation by providing ratings across four additional dimensions: noisiness, coloration, discontinuity, and loudness, alongside MOS. In this paper, we extend the explored univariate MOS estimation to a multivariate framework by modeling these dimensions jointly using a multivariate Gaussian distribution. Our approach utilizes Cholesky decomposition to predict covariances without imposing restrictive assumptions and extends probabilistic affine transformations to a multivariate context. Experimental results show that our model performs on par with state-of-the-art methods in point estimation, while uniquely providing uncertainty and correlation estimates across speech quality dimensions. This enables better diagnosis of poor speech quality and informs targeted improvements.
#### Effects of Speaker Count, Duration, and Accent Diversity on Zero-Shot Accent Robustness in Low-Resource ASR
 - **Authors:** Zheng-Xin Yong, Vineel Pratap, Michael Auli, Jean Maillard
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04364

 - **Pdf link:** https://arxiv.org/pdf/2506.04364

 - **Abstract**
 To build an automatic speech recognition (ASR) system that can serve everyone in the world, the ASR needs to be robust to a wide range of accents including unseen accents. We systematically study how three different variables in training data -- the number of speakers, the audio duration per each individual speaker, and the diversity of accents -- affect ASR robustness towards unseen accents in a low-resource training regime. We observe that for a fixed number of ASR training hours, it is more beneficial to increase the number of speakers (which means each speaker contributes less) than the number of hours contributed per speaker. We also observe that more speakers enables ASR performance gains from scaling number of hours. Surprisingly, we observe minimal benefits to prioritizing speakers with different accents when the number of speakers is controlled. Our work suggests that practitioners should prioritize increasing the speaker count in ASR training data composition for new languages.
#### Grapheme-Coherent Phonemic and Prosodic Annotation of Speech by Implicit and Explicit Grapheme Conditioning
 - **Authors:** Hien Ohnaka, Yuma Shirahata, Byeongseon Park, Ryuichi Yamamoto
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04527

 - **Pdf link:** https://arxiv.org/pdf/2506.04527

 - **Abstract**
 We propose a model to obtain phonemic and prosodic labels of speech that are coherent with graphemes. Unlike previous methods that simply fine-tune a pre-trained ASR model with the labels, the proposed model conditions the label generation on corresponding graphemes by two methods: 1) Add implicit grapheme conditioning through prompt encoder using pre-trained BERT features. 2) Explicitly prune the label hypotheses inconsistent with the grapheme during inference. These methods enable obtaining parallel data of speech, the labels, and graphemes, which is applicable to various downstream tasks such as text-to-speech and accent estimation from text. Experiments showed that the proposed method significantly improved the consistency between graphemes and the predicted labels. Further, experiments on accent estimation task confirmed that the created parallel data by the proposed method effectively improve the estimation accuracy.
#### LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Models
 - **Authors:** Wen Ding, Fan Qian
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04586

 - **Pdf link:** https://arxiv.org/pdf/2506.04586

 - **Abstract**
 We introduce LESS (Large Language Model Enhanced Semi-supervised Learning), a versatile framework that leverages Large Language Models (LLMs) to correct pseudo labels generated from in-the-wild data. Within the LESS framework, pseudo-labeled text from Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST) of the unsupervised data is refined by an LLM, and augmented by a data filtering strategy to optimize LLM knowledge transfer efficiency. Experiments on both Mandarin ASR and Spanish-to-English AST tasks show that LESS achieves a notable absolute WER reduction of 3.77% on the Wenet Speech test set, as well as BLEU scores of 34.0 and 64.7 on Callhome and Fisher test sets respectively. These results validate the adaptability of LESS across different languages, tasks, and domains. Ablation studies conducted with various LLMs and prompt configurations provide novel insights into leveraging LLM-derived knowledge for speech processing applications.
#### LLM-based phoneme-to-grapheme for phoneme-based speech recognition
 - **Authors:** Te Ma, Min Bi, Saierdaer Yusuyin, Hao Huang, Zhijian Ou
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04711

 - **Pdf link:** https://arxiv.org/pdf/2506.04711

 - **Abstract**
 In automatic speech recognition (ASR), phoneme-based multilingual pre-training and crosslingual fine-tuning is attractive for its high data efficiency and competitive results compared to subword-based models. However, Weighted Finite State Transducer (WFST) based decoding is limited by its complex pipeline and inability to leverage large language models (LLMs). Therefore, we propose LLM-based phoneme-to-grapheme (LLM-P2G) decoding for phoneme-based ASR, consisting of speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G). A challenge is that there seems to have information loss in cascading S2P and P2G. To address this challenge, we propose two training strategies: data augmentation with noisy phonemes (DANP), and randomized top-$K$ marginalized (TKM) training and decoding. Our experimental results show that LLM-P2G outperforms WFST-based systems in crosslingual ASR for Polish and German, by relative WER reductions of 3.6% and 6.9% respectively.
#### IIITH-BUT system for IWSLT 2025 low-resource Bhojpuri to Hindi speech translation
 - **Authors:** Bhavana Akkiraju, Aishwarya Pothula, Santosh Kesiraju, Anil Kumar Vuppala
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04714

 - **Pdf link:** https://arxiv.org/pdf/2506.04714

 - **Abstract**
 This paper presents the submission of IIITH-BUT to the IWSLT 2025 shared task on speech translation for the low-resource Bhojpuri-Hindi language pair. We explored the impact of hyperparameter optimisation and data augmentation techniques on the performance of the SeamlessM4T model fine-tuned for this specific task. We systematically investigated a range of hyperparameters including learning rate schedules, number of update steps, warm-up steps, label smoothing, and batch sizes; and report their effect on translation quality. To address data scarcity, we applied speed perturbation and SpecAugment and studied their effect on translation quality. We also examined the use of cross-lingual signal through joint training with Marathi and Bhojpuri speech data. Our experiments reveal that careful selection of hyperparameters and the application of simple yet effective augmentation techniques significantly improve performance in low-resource settings. We also analysed the translation hypotheses to understand various kinds of errors that impacted the translation quality in terms of BLEU.
#### Improving AI-generated music with user-guided training
 - **Authors:** Vishwa Mohan Singh, Sai Anirudh Aryasomayajula, Ahan Chatterjee, Beste Aydemir, Rifat Mehreen Amin
 - **Subjects:** Subjects:
Sound (cs.SD); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04852

 - **Pdf link:** https://arxiv.org/pdf/2506.04852

 - **Abstract**
 AI music generation has advanced rapidly, with models like diffusion and autoregressive algorithms enabling high-fidelity outputs. These tools can alter styles, mix instruments, or isolate them. Since sound can be visualized as spectrograms, image-generation algorithms can be applied to generate novel music. However, these algorithms are typically trained on fixed datasets, which makes it challenging for them to interpret and respond to user input accurately. This is especially problematic because music is highly subjective and requires a level of personalization that image generation does not provide. In this work, we propose a human-computation approach to gradually improve the performance of these algorithms based on user interactions. The human-computation element involves aggregating and selecting user ratings to use as the loss function for fine-tuning the model. We employ a genetic algorithm that incorporates user feedback to enhance the baseline performance of a model initially trained on a fixed dataset. The effectiveness of this approach is measured by the average increase in user ratings with each iteration. In the pilot test, the first iteration showed an average rating increase of 0.2 compared to the baseline. The second iteration further improved upon this, achieving an additional increase of 0.39 over the first iteration.
#### A Practitioner's Guide to Building ASR Models for Low-Resource Languages: A Case Study on Scottish Gaelic
 - **Authors:** Ondřej Klejch, William Lamb, Peter Bell
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04915

 - **Pdf link:** https://arxiv.org/pdf/2506.04915

 - **Abstract**
 An effective approach to the development of ASR systems for low-resource languages is to fine-tune an existing multilingual end-to-end model. When the original model has been trained on large quantities of data from many languages, fine-tuning can be effective with limited training data, even when the language in question was not present in the original training data. The fine-tuning approach has been encouraged by the availability of public-domain E2E models and is widely believed to lead to state-of-the-art results. This paper, however, challenges that belief. We show that an approach combining hybrid HMMs with self-supervised models can yield substantially better performance with limited training data. This combination allows better utilisation of all available speech and text data through continued self-supervised pre-training and semi-supervised training. We benchmark our approach on Scottish Gaelic, achieving WER reductions of 32% relative over our best fine-tuned Whisper model.
#### The NTNU System at the S&I Challenge 2025 SLA Open Track
 - **Authors:** Hong-Yun Lin, Tien-Hong Lo, Yu-Hsuan Fang, Jhen-Ke Lin, Chung-Chun Wang, Hao-Chien Lu, Berlin Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.05121

 - **Pdf link:** https://arxiv.org/pdf/2506.05121

 - **Abstract**
 A recent line of research on spoken language assessment (SLA) employs neural models such as BERT and wav2vec 2.0 (W2V) to evaluate speaking proficiency across linguistic and acoustic modalities. Although both models effectively capture features relevant to oral competence, each exhibits modality-specific limitations. BERT-based methods rely on ASR transcripts, which often fail to capture prosodic and phonetic cues for SLA. In contrast, W2V-based methods excel at modeling acoustic features but lack semantic interpretability. To overcome these limitations, we propose a system that integrates W2V with Phi-4 multimodal large language model (MLLM) through a score fusion strategy. The proposed system achieves a root mean square error (RMSE) of 0.375 on the official test set of the Speak & Improve Challenge 2025, securing second place in the competition. For comparison, the RMSEs of the top-ranked, third-ranked, and official baseline systems are 0.364, 0.384, and 0.444, respectively.
#### AudioLens: A Closer Look at Auditory Attribute Perception of Large Audio-Language Models
 - **Authors:** Chih-Kai Yang, Neo Ho, Yi-Jyun Lee, Hung-yi Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.05140

 - **Pdf link:** https://arxiv.org/pdf/2506.05140

 - **Abstract**
 Understanding the internal mechanisms of large audio-language models (LALMs) is crucial for interpreting their behavior and improving performance. This work presents the first in-depth analysis of how LALMs internally perceive and recognize auditory attributes. By applying vocabulary projection on three state-of-the-art LALMs, we track how attribute information evolves across layers and token positions. We find that attribute information generally decreases with layer depth when recognition fails, and that resolving attributes at earlier layers correlates with better accuracy. Moreover, LALMs heavily rely on querying auditory inputs for predicting attributes instead of aggregating necessary information in hidden states at attribute-mentioning positions. Based on our findings, we demonstrate a method to enhance LALMs. Our results offer insights into auditory attribute processing, paving the way for future improvements.


by Zyzzyva0381 (Windy). 


2025-06-06
