# Showing new listings for Friday, 4 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 13papers 
#### Pronunciation Editing for Finnish Speech using Phonetic Posteriorgrams
 - **Authors:** Zirui Li, Lauri Juvela, Mikko Kurimo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02115

 - **Pdf link:** https://arxiv.org/pdf/2507.02115

 - **Abstract**
 Synthesizing second-language (L2) speech is potentially highly valued for L2 language learning experience and feedback. However, due to the lack of L2 speech synthesis datasets, it is difficult to synthesize L2 speech for low-resourced languages. In this paper, we provide a practical solution for editing native speech to approximate L2 speech and present PPG2Speech, a diffusion-based multispeaker Phonetic-Posteriorgrams-to-Speech model that is capable of editing a single phoneme without text alignment. We use Matcha-TTS's flow-matching decoder as the backbone, transforming Phonetic Posteriorgrams (PPGs) to mel-spectrograms conditioned on external speaker embeddings and pitch. PPG2Speech strengthens the Matcha-TTS's flow-matching decoder with Classifier-free Guidance (CFG) and Sway Sampling. We also propose a new task-specific objective evaluation metric, the Phonetic Aligned Consistency (PAC), between the edited PPGs and the PPGs extracted from the synthetic speech for editing effects. We validate the effectiveness of our method on Finnish, a low-resourced, nearly phonetic language, using approximately 60 hours of data. We conduct objective and subjective evaluations of our approach to compare its naturalness, speaker similarity, and editing effectiveness with TTS-based editing. Our source code is published at this https URL.
#### An Investigation on Combining Geometry and Consistency Constraints into Phase Estimation for Speech Enhancement
 - **Authors:** Chun-Wei Ho, Pin-Jui Ku, Hao Yen, Sabato Marco Siniscalchi, Yu Tsao, Chin-Hui Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02192

 - **Pdf link:** https://arxiv.org/pdf/2507.02192

 - **Abstract**
 We propose a novel iterative phase estimation framework, termed multi-source Griffin-Lim algorithm (MSGLA), for speech enhancement (SE) under additive noise conditions. The core idea is to leverage the ad-hoc consistency constraint of complex-valued short-time Fourier transform (STFT) spectrograms to address the sign ambiguity challenge commonly encountered in geometry-based phase estimation. Furthermore, we introduce a variant of the geometric constraint framework based on the law of sines and cosines, formulating a new phase reconstruction algorithm using noise phase estimates. We first validate the proposed technique through a series of oracle experiments, demonstrating its effectiveness under ideal conditions. We then evaluate its performance on the VB-DMD and WSJ0-CHiME3 data sets, and show that the proposed MSGLA variants match well or slightly outperform existing algorithms, including direct phase estimation and DNN-based sign prediction, especially in terms of background noise suppression.
#### Open-Source System for Multilingual Translation and Cloned Speech Synthesis
 - **Authors:** Mateo Cámara, Juan Gutiérrez, María Pilar Daza, José Luis Blanco
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02530

 - **Pdf link:** https://arxiv.org/pdf/2507.02530

 - **Abstract**
 We present an open-source system designed for multilingual translation and speech regeneration, addressing challenges in communication and accessibility across diverse linguistic contexts. The system integrates Whisper for speech recognition with Voice Activity Detection (VAD) to identify speaking intervals, followed by a pipeline of Large Language Models (LLMs). For multilingual applications, the first LLM segments speech into coherent, complete sentences, which a second LLM then translates. For speech regeneration, the system uses a text-to-speech (TTS) module with voice cloning capabilities to replicate the original speaker's voice, maintaining naturalness and speaker identity. The system's open-source components can operate locally or via APIs, offering cost-effective deployment across various use cases. These include real-time multilingual translation in Zoom sessions, speech regeneration for public broadcasts, and Bluetooth-enabled multilingual playback through personal devices. By preserving the speaker's voice, the system ensures a seamless and immersive experience, whether translating or regenerating speech. This open-source project is shared with the community to foster innovation and accessibility. We provide a detailed system performance analysis, including latency and word accuracy, demonstrating its potential to enable inclusive, adaptable communication solutions in real-world multilingual scenarios.
#### Multi-Utterance Speech Separation and Association Trained on Short Segments
 - **Authors:** Yuzhu Wang, Archontis Politis, Konstantinos Drossos, Tuomas Virtanen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.02562

 - **Pdf link:** https://arxiv.org/pdf/2507.02562

 - **Abstract**
 Current deep neural network (DNN) based speech separation faces a fundamental challenge -- while the models need to be trained on short segments due to computational constraints, real-world applications typically require processing significantly longer recordings with multiple utterances per speaker than seen during training. In this paper, we investigate how existing approaches perform in this challenging scenario and propose a frequency-temporal recurrent neural network (FTRNN) that effectively bridges this gap. Our FTRNN employs a full-band module to model frequency dependencies within each time frame and a sub-band module that models temporal patterns in each frequency band. Despite being trained on short fixed-length segments of 10 s, our model demonstrates robust separation when processing signals significantly longer than training segments (21-121 s) and preserves speaker association across utterance gaps exceeding those seen during training. Unlike the conventional segment-separation-stitch paradigm, our lightweight approach (0.9 M parameters) performs inference on long audio without segmentation, eliminating segment boundary distortions while simplifying deployment. Experimental results demonstrate the generalization ability of FTRNN for multi-utterance speech separation and speaker association.
#### Multi-agent Auditory Scene Analysis
 - **Authors:** Caleb Rascon, Luis Gato-Diaz, Eduardo García-Alarcón
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2507.02755

 - **Pdf link:** https://arxiv.org/pdf/2507.02755

 - **Abstract**
 Auditory scene analysis (ASA) aims to retrieve information from the acoustic environment, by carrying out three main tasks: sound source location, separation, and classification. These tasks are traditionally executed with a linear data flow, where the sound sources are first located; then, using their location, each source is separated into its own audio stream; from each of which, information is extracted that is relevant to the application scenario (audio event detection, speaker identification, emotion classification, etc.). However, running these tasks linearly increases the overall response time, while making the last tasks (separation and classification) highly sensitive to errors of the first task (location). A considerable amount of effort and computational complexity has been employed in the state-of-the-art to develop techniques that are the least error-prone possible. However, doing so gives rise to an ASA system that is non-viable in many applications that require a small computational footprint and a low response time, such as bioacoustics, hearing-aid design, search and rescue, human-robot interaction, etc. To this effect, in this work, a multi-agent approach is proposed to carry out ASA where the tasks are run in parallel, with feedback loops between them to compensate for local errors, such as: using the quality of the separation output to correct the location error; and using the classification result to reduce the localization's sensitivity towards interferences. The result is a multi-agent auditory scene analysis (MASA) system that is robust against local errors, without a considerable increase in complexity, and with a low response time. The complete proposed MASA system is provided as a framework that uses open-source tools for sound acquisition and reproduction (JACK) and inter-agent communication (ROS2), allowing users to add their own agents.
#### DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment
 - **Authors:** Ke-Han Lu, Zhehuai Chen, Szu-Wei Fu, Chao-Han Huck Yang, Sung-Feng Huang, Chih-Kai Yang, Chee-En Yu, Chun-Wei Chen, Wei-Chih Chen, Chien-yu Huang, Yi-Cheng Lin, Yu-Xiang Lin, Chi-An Fu, Chun-Yi Kuan, Wenze Ren, Xuanjun Chen, Wei-Ping Huang, En-Pei Hu, Tzu-Quan Lin, Yuan-Kuei Wu, Kuan-Po Huang, Hsiao-Ying Huang, Huang-Cheng Chou, Kai-Wei Chang, Cheng-Han Chiang, Boris Ginsburg, Yu-Chiang Frank Wang, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.02768

 - **Pdf link:** https://arxiv.org/pdf/2507.02768

 - **Abstract**
 We introduce DeSTA2.5-Audio, a general-purpose Large Audio Language Model (LALM) designed for robust auditory perception and instruction-following, without requiring task-specific audio instruction-tuning. Recent LALMs typically augment Large Language Models (LLMs) with auditory capabilities by training on large-scale, manually curated or LLM-synthesized audio-instruction datasets. However, these approaches have often suffered from the catastrophic forgetting of the LLM's original language abilities. To address this, we revisit the data construction pipeline and propose DeSTA, a self-generated cross-modal alignment strategy in which the backbone LLM generates its own training targets. This approach preserves the LLM's native language proficiency while establishing effective audio-text alignment, thereby enabling zero-shot generalization without task-specific tuning. Using DeSTA, we construct DeSTA-AQA5M, a large-scale, task-agnostic dataset containing 5 million training samples derived from 7,000 hours of audio spanning 50 diverse datasets, including speech, environmental sounds, and music. DeSTA2.5-Audio achieves state-of-the-art or competitive performance across a wide range of audio-language benchmarks, including Dynamic-SUPERB, MMAU, SAKURA, Speech-IFEval, and VoiceBench. Comprehensive comparative studies demonstrate that our self-generated strategy outperforms widely adopted data construction and training strategies in both auditory perception and instruction-following capabilities. Our findings underscore the importance of carefully designed data construction in LALM development and offer practical insights for building robust, general-purpose LALMs.
#### Self-Steering Deep Non-Linear Spatially Selective Filters for Efficient Extraction of Moving Speakers under Weak Guidance
 - **Authors:** Jakob Kienegger, Alina Mannanova, Huajian Fang, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.02791

 - **Pdf link:** https://arxiv.org/pdf/2507.02791

 - **Abstract**
 Recent works on deep non-linear spatially selective filters demonstrate exceptional enhancement performance with computationally lightweight architectures for stationary speakers of known directions. However, to maintain this performance in dynamic scenarios, resource-intensive data-driven tracking algorithms become necessary to provide precise spatial guidance conditioned on the initial direction of a target speaker. As this additional computational overhead hinders application in resource-constrained scenarios such as real-time speech enhancement, we present a novel strategy utilizing a low-complexity tracking algorithm in the form of a particle filter instead. Assuming a causal, sequential processing style, we introduce temporal feedback to leverage the enhanced speech signal of the spatially selective filter to compensate for the limited modeling capabilities of the particle filter. Evaluation on a synthetic dataset illustrates how the autoregressive interplay between both algorithms drastically improves tracking accuracy and leads to strong enhancement performance. A listening test with real-world recordings complements these findings by indicating a clear trend towards our proposed self-steering pipeline as preferred choice over comparable methods.
#### Towards Perception-Informed Latent HRTF Representations
 - **Authors:** You Zhang, Andrew Francl, Ruohan Gao, Paul Calamia, Zhiyao Duan, Ishwarya Ananthabhotla
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.02815

 - **Pdf link:** https://arxiv.org/pdf/2507.02815

 - **Abstract**
 Personalized head-related transfer functions (HRTFs) are essential for ensuring a realistic auditory experience over headphones, because they take into account individual anatomical differences that affect listening. Most machine learning approaches to HRTF personalization rely on a learned low-dimensional latent space to generate or select custom HRTFs for a listener. However, these latent representations are typically learned in a manner that optimizes for spectral reconstruction but not for perceptual compatibility, meaning they may not necessarily align with perceptual distance. In this work, we first study whether traditionally learned HRTF representations are well correlated with perceptual relations using auditory-based objective perceptual metrics; we then propose a method for explicitly embedding HRTFs into a perception-informed latent space, leveraging a metric-based loss function and supervision via Metric Multidimensional Scaling (MMDS). Finally, we demonstrate the applicability of these learned representations to the task of HRTF personalization. We suggest that our method has the potential to render personalized spatial audio, leading to an improved listening experience.
#### Analyzing and Improving Speaker Similarity Assessment for Speech Synthesis
 - **Authors:** Marc-André Carbonneau, Benjamin van Niekerk, Hugo Seuté, Jean-Philippe Letendre, Herman Kamper, Julian Zaïdi
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02176

 - **Pdf link:** https://arxiv.org/pdf/2507.02176

 - **Abstract**
 Modeling voice identity is challenging due to its multifaceted nature. In generative speech systems, identity is often assessed using automatic speaker verification (ASV) embeddings, designed for discrimination rather than characterizing identity. This paper investigates which aspects of a voice are captured in such representations. We find that widely used ASV embeddings focus mainly on static features like timbre and pitch range, while neglecting dynamic elements such as rhythm. We also identify confounding factors that compromise speaker similarity measurements and suggest mitigation strategies. To address these gaps, we propose U3D, a metric that evaluates speakers' dynamic rhythm patterns. This work contributes to the ongoing challenge of assessing speaker identity consistency in the context of ever-better voice cloning systems. We publicly release our code.
#### JoyTTS: LLM-based Spoken Chatbot With Voice Cloning
 - **Authors:** Fangru Zhou, Jun Zhao, Guoxin Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02380

 - **Pdf link:** https://arxiv.org/pdf/2507.02380

 - **Abstract**
 JoyTTS is an end-to-end spoken chatbot that combines large language models (LLM) with text-to-speech (TTS) technology, featuring voice cloning capabilities. This project is built upon the open-source MiniCPM-o and CosyVoice2 models and trained on 2000 hours of conversational data. We have also provided the complete training code to facilitate further development and optimization by the community. On the testing machine seed-tts-zh, it achieves a SS (speaker similarity) score of 0.73 and a WER (Word Error Rate) of 5.09. The code and models, along with training and inference scripts, are available at this https URL.
#### Posterior Transition Modeling for Unsupervised Diffusion-Based Speech Enhancement
 - **Authors:** Mostafa Sadeghi (MULTISPEECH), Jean-Eudes Ayilo (MULTISPEECH), Romain Serizel (MULTISPEECH), Xavier Alameda-Pineda (ROBOTLEARN)
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02391

 - **Pdf link:** https://arxiv.org/pdf/2507.02391

 - **Abstract**
 We explore unsupervised speech enhancement using diffusion models as expressive generative priors for clean speech. Existing approaches guide the reverse diffusion process using noisy speech through an approximate, noise-perturbed likelihood score, combined with the unconditional score via a trade-off hyperparameter. In this work, we propose two alternative algorithms that directly model the conditional reverse transition distribution of diffusion states. The first method integrates the diffusion prior with the observation model in a principled way, removing the need for hyperparameter tuning. The second defines a diffusion process over the noisy speech itself, yielding a fully tractable and exact likelihood score. Experiments on the WSJ0-QUT and VoiceBank-DEMAND datasets demonstrate improved enhancement metrics and greater robustness to domain shifts compared to both supervised and unsupervised baselines.
#### Benchmarking Akan ASR Models Across Domain-Specific Datasets: A Comparative Evaluation of Performance, Scalability, and Adaptability
 - **Authors:** Mark Atta Mensah, Isaac Wiafe, Akon Ekpezu, Justice Kwame Appati, Jamal-Deen Abdulai, Akosua Nyarkoa Wiafe-Akenten, Frank Ernest Yeboah, Gifty Odame
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02407

 - **Pdf link:** https://arxiv.org/pdf/2507.02407

 - **Abstract**
 Most existing automatic speech recognition (ASR) research evaluate models using in-domain datasets. However, they seldom evaluate how they generalize across diverse speech contexts. This study addresses this gap by benchmarking seven Akan ASR models built on transformer architectures, such as Whisper and Wav2Vec2, using four Akan speech corpora to determine their performance. These datasets encompass various domains, including culturally relevant image descriptions, informal conversations, biblical scripture readings, and spontaneous financial dialogues. A comparison of the word error rate and character error rate highlighted domain dependency, with models performing optimally only within their training domains while showing marked accuracy degradation in mismatched scenarios. This study also identified distinct error behaviors between the Whisper and Wav2Vec2 architectures. Whereas fine-tuned Whisper Akan models led to more fluent but potentially misleading transcription errors, Wav2Vec2 produced more obvious yet less interpretable outputs when encountering unfamiliar inputs. This trade-off between readability and transparency in ASR errors should be considered when selecting architectures for low-resource language (LRL) applications. These findings highlight the need for targeted domain adaptation techniques, adaptive routing strategies, and multilingual training frameworks for Akan and other LRLs.
#### De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks
 - **Authors:** Wei Fan, Kejiang Chen, Chang Liu, Weiming Zhang, Nenghai Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02606

 - **Pdf link:** https://arxiv.org/pdf/2507.02606

 - **Abstract**
 The rapid advancement of speech generation models has heightened privacy and security concerns related to voice cloning (VC). Recent studies have investigated disrupting unauthorized voice cloning by introducing adversarial perturbations. However, determined attackers can mitigate these protective perturbations and successfully execute VC. In this study, we conduct the first systematic evaluation of these protective perturbations against VC under realistic threat models that include perturbation purification. Our findings reveal that while existing purification methods can neutralize a considerable portion of the protective perturbations, they still lead to distortions in the feature space of VC models, which degrades the performance of VC. From this perspective, we propose a novel two-stage purification method: (1) Purify the perturbed speech; (2) Refine it using phoneme guidance to align it with the clean speech distribution. Experimental results demonstrate that our method outperforms state-of-the-art purification methods in disrupting VC defenses. Our study reveals the limitations of adversarial perturbation-based VC defenses and underscores the urgent need for more robust solutions to mitigate the security and privacy risks posed by VC. The code and audio samples are available at this https URL.


by Zyzzyva0381 (Windy). 


2025-07-04
