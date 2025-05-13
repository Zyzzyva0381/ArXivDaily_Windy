# Showing new listings for Tuesday, 13 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 10papers 
#### RADE: A Neural Codec for Transmitting Speech over HF Radio Channels
 - **Authors:** David Rowe, Jean-Marc Valin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.06671

 - **Pdf link:** https://arxiv.org/pdf/2505.06671

 - **Abstract**
 Speech compression is commonly used to send voice over radio channels in applications such as mobile telephony and two-way push-to-talk (PTT) radio. In classical systems, the speech codec is combined with forward error correction, modulation and radio hardware. In this paper we describe an autoencoder that replaces many of the traditional signal processing elements with a neural network. The encoder takes a vocoder feature set (short term spectrum, pitch, voicing), and produces discrete time, but continuously valued quadrature amplitude modulation (QAM) symbols. We use orthogonal frequency domain multiplexing (OFDM) to send and receive these symbols over high frequency (HF) radio channels. The decoder converts received QAM symbols to vocoder features suitable for synthesis. The autoencoder has been trained to be robust to additive Gaussian noise and multipath channel impairments while simultaneously maintaining a Peak To Average Power Ratio (PAPR) of less than 1~dB. Over simulated and real world HF radio channels we have achieved output speech intelligibility that clearly surpasses existing analog and digital radio systems over a range of SNRs.
#### TACOS: Temporally-aligned Audio CaptiOnS for Language-Audio Pretraining
 - **Authors:** Paul Primus, Florian Schmid, Gerhard Widmer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.07609

 - **Pdf link:** https://arxiv.org/pdf/2505.07609

 - **Abstract**
 Learning to associate audio with textual descriptions is valuable for a range of tasks, including pretraining, zero-shot classification, audio retrieval, audio captioning, and text-conditioned audio generation. Existing contrastive language-audio pretrained models are typically trained using global, clip-level descriptions, which provide only weak temporal supervision. We hypothesize that CLAP-like language-audio models - particularly, if they are expected to produce frame-level embeddings - can benefit from a stronger temporal supervision. To confirm our hypothesis, we curate a novel dataset of approximately 12,000 audio recordings from Freesound, each annotated with single-sentence free-text descriptions linked to a specific temporal segment in an audio recording. We use large language models to clean these annotations by removing references to non-audible events, transcribed speech, typos, and annotator language bias. We further propose a frame-wise contrastive training strategy that learns to align text descriptions with temporal regions in an audio recording and demonstrate that our model has better temporal text-audio alignment abilities compared to models trained only on global captions when evaluated on the AudioSet Strong benchmark. The dataset and our source code are available on Zenodo and GitHub, respectively.
#### Is MixIT Really Unsuitable for Correlated Sources? Exploring MixIT for Unsupervised Pre-training in Music Source Separation
 - **Authors:** Kohei Saijo, Yoshiaki Bando
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.07631

 - **Pdf link:** https://arxiv.org/pdf/2505.07631

 - **Abstract**
 In music source separation (MSS), obtaining isolated sources or stems is highly costly, making pre-training on unlabeled data a promising approach. Although source-agnostic unsupervised learning like mixture-invariant training (MixIT) has been explored in general sound separation, they have been largely overlooked in MSS due to its implicit assumption of source independence. We hypothesize, however, that the difficulty of applying MixIT to MSS arises from the ill-posed nature of MSS itself, where stem definitions are application-dependent and models lack explicit knowledge of what should or should not be separated, rather than from high inter-source correlation. While MixIT does not assume any source model and struggles with such ambiguities, our preliminary experiments show that it can still separate instruments to some extent, suggesting its potential for unsupervised pre-training. Motivated by these insights, this study investigates MixIT-based pre-training for MSS. We first pre-train a model on in-the-wild, unlabeled data from the Free Music Archive using MixIT, and then fine-tune it on MUSDB18 with supervision. Using the band-split TF-Locoformer, one of the state-of-the-art MSS models, we demonstrate that MixIT-based pre-training improves the performance over training from scratch.
#### TS-SUPERB: A Target Speech Processing Benchmark for Speech Self-Supervised Learning Models
 - **Authors:** Junyi Peng, Takanori Ashihara, Marc Delcroix, Tsubasa Ochiai, Oldrich Plchot, Shoko Araki, Jan Černocký
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.06660

 - **Pdf link:** https://arxiv.org/pdf/2505.06660

 - **Abstract**
 Self-supervised learning (SSL) models have significantly advanced speech processing tasks, and several benchmarks have been proposed to validate their effectiveness. However, previous benchmarks have primarily focused on single-speaker scenarios, with less exploration of target-speaker tasks in noisy, multi-talker conditions -- a more challenging yet practical case. In this paper, we introduce the Target-Speaker Speech Processing Universal Performance Benchmark (TS-SUPERB), which includes four widely recognized target-speaker processing tasks that require identifying the target speaker and extracting information from the speech mixture. In our benchmark, the speaker embedding extracted from enrollment speech is used as a clue to condition downstream models. The benchmark result reveals the importance of evaluating SSL models in target speaker scenarios, demonstrating that performance cannot be easily inferred from related single-speaker tasks. Moreover, by using a unified SSL-based target speech encoder, consisting of a speaker encoder and an extractor module, we also investigate joint optimization across TS tasks to leverage mutual information and demonstrate its effectiveness.
#### Beyond Identity: A Generalizable Approach for Deepfake Audio Detection
 - **Authors:** Yasaman Ahmadiadli, Xiao-Ping Zhang, Naimul Khan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2505.06766

 - **Pdf link:** https://arxiv.org/pdf/2505.06766

 - **Abstract**
 Deepfake audio presents a growing threat to digital security, due to its potential for social engineering, fraud, and identity misuse. However, existing detection models suffer from poor generalization across datasets, due to implicit identity leakage, where models inadvertently learn speaker-specific features instead of manipulation artifacts. To the best of our knowledge, this is the first study to explicitly analyze and address identity leakage in the audio deepfake detection domain. This work proposes an identity-independent audio deepfake detection framework that mitigates identity leakage by encouraging the model to focus on forgery-specific artifacts instead of overfitting to speaker traits. Our approach leverages Artifact Detection Modules (ADMs) to isolate synthetic artifacts in both time and frequency domains, enhancing cross-dataset generalization. We introduce novel dynamic artifact generation techniques, including frequency domain swaps, time domain manipulations, and background noise augmentation, to enforce learning of dataset-invariant features. Extensive experiments conducted on ASVspoof2019, ADD 2022, FoR, and In-The-Wild datasets demonstrate that the proposed ADM-enhanced models achieve F1 scores of 0.230 (ADD 2022), 0.604 (FoR), and 0.813 (In-The-Wild), consistently outperforming the baseline. Dynamic Frequency Swap proves to be the most effective strategy across diverse conditions. These findings emphasize the value of artifact-based learning in mitigating implicit identity leakage for more generalizable audio deepfake detection.
#### Collection: Datasets from AFAR Challenge
 - **Authors:** Saad Masrur, Ozgur Ozdemir, Anil Gurses, Ismail Guvenc, Mihail L.Sichitiu, Rudra Dutta, Magreth Mushi, homas Zajkowski, Cole Dickerson, Gautham Reddy, Sergio Vargas Villar, Chau-Wai Wong, Baisakhi Chatterjee, Sonali Chaudhari, Zhizhen Li, Yuchen Liu, Paul Kudyba, Haijian Sun, Jaya Sravani Mandapaka, Kamesh Namuduri, Weijie Wang, Fraida Fund
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.06823

 - **Pdf link:** https://arxiv.org/pdf/2505.06823

 - **Abstract**
 This paper presents a comprehensive real-world and Digital Twin (DT) dataset collected as part of the Find A Rover (AFAR) Challenge, organized by the NSF Aerial Experimentation and Research Platform for Advanced Wireless (AERPAW) testbed and hosted at the Lake Wheeler Field in Raleigh, North Carolina. The AFAR Challenge was a competition involving five finalist university teams, focused on promoting innovation in UAV-assisted radio frequency (RF) source localization. Participating teams were tasked with designing UAV flight trajectories and localization algorithms to detect the position of a hidden unmanned ground vehicle (UGV), also referred to as a rover, emitting wireless probe signals generated by GNU Radio. The competition was structured to evaluate solutions in a DT environment first, followed by deployment and testing in AERPAW's outdoor wireless testbed. For each team, the UGV was placed at three different positions, resulting in a total of 30 datasets, 15 collected in a DT simulation environment and 15 in a physical outdoor testbed. Each dataset contains time-synchronized measurements of received signal strength (RSS), received signal quality (RSQ), GPS coordinates, UAV velocity, and UAV orientation (roll, pitch, and yaw). Data is organized into structured folders by team, environment (DT and real-world), and UGV location. The dataset supports research in UAV-assisted RF source localization, air-to-ground (A2G) wireless propagation modeling, trajectory optimization, signal prediction, autonomous navigation, and DT validation. With approximately 300k time-synchronized samples collected from real-world experiments, the dataset provides a substantial foundation for training and evaluating deep learning (DL) models. Overall, the AFAR dataset serves as a valuable resource for advancing robust, real-world solutions in UAV-enabled wireless communications and sensing systems.
#### Multi-band Frequency Reconstruction for Neural Psychoacoustic Coding
 - **Authors:** Dianwen Ng, Kun Zhou, Yi-Wen Chao, Zhiwei Xiong, Bin Ma, Eng Siong Chng
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.07235

 - **Pdf link:** https://arxiv.org/pdf/2505.07235

 - **Abstract**
 Achieving high-fidelity audio compression while preserving perceptual quality across diverse content remains a key challenge in Neural Audio Coding (NAC). We introduce MUFFIN, a fully convolutional Neural Psychoacoustic Coding (NPC) framework that leverages psychoacoustically guided multi-band frequency reconstruction. At its core is a Multi-Band Spectral Residual Vector Quantization (MBS-RVQ) module that allocates bitrate across frequency bands based on perceptual salience. This design enables efficient compression while disentangling speaker identity from content using distinct codebooks. MUFFIN incorporates a transformer-inspired convolutional backbone and a modified snake activation to enhance resolution in fine-grained spectral regions. Experimental results on multiple benchmarks demonstrate that MUFFIN consistently outperforms existing approaches in reconstruction quality. A high-compression variant achieves a state-of-the-art 12.5 Hz rate with minimal loss. MUFFIN also proves effective in downstream generative tasks, highlighting its promise as a token representation for integration with language models. Audio samples and code are available.
#### Multi-Domain Audio Question Answering Toward Acoustic Content Reasoning in The DCASE 2025 Challenge
 - **Authors:** Chao-Han Huck Yang, Sreyan Ghosh, Qing Wang, Jaeyeon Kim, Hengyi Hong, Sonal Kumar, Guirui Zhong, Zhifeng Kong, S Sakshi, Vaibhavi Lokegaonkar, Oriol Nieto, Ramani Duraiswami, Dinesh Manocha, Gunhee Kim, Jun Du, Rafael Valle, Bryan Catanzaro
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.07365

 - **Pdf link:** https://arxiv.org/pdf/2505.07365

 - **Abstract**
 We present Task 5 of the DCASE 2025 Challenge: an Audio Question Answering (AQA) benchmark spanning multiple domains of sound understanding. This task defines three QA subsets (Bioacoustics, Temporal Soundscapes, and Complex QA) to test audio-language models on interactive question-answering over diverse acoustic scenes. We describe the dataset composition (from marine mammal calls to soundscapes and complex real-world clips), the evaluation protocol (top-1 accuracy with answer-shuffling robustness), and baseline systems (Qwen2-Audio-7B, AudioFlamingo 2, Gemini-2-Flash). Preliminary results on the development set are compared, showing strong variation across models and subsets. This challenge aims to advance the audio understanding and reasoning capabilities of audio-language models toward human-level acuity, which are crucial for enabling AI agents to perceive and interact about the world effectively.
#### Lightweight End-to-end Text-to-speech Synthesis for low resource on-device applications
 - **Authors:** Biel Tura Vecino, Adam Gabryś, Daniel Mątwicki, Andrzej Pomirski, Tom Iddon, Marius Cotescu, Jaime Lorenzo-Trueba
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.07701

 - **Pdf link:** https://arxiv.org/pdf/2505.07701

 - **Abstract**
 Recent works have shown that modelling raw waveform directly from text in an end-to-end (E2E) fashion produces more natural-sounding speech than traditional neural text-to-speech (TTS) systems based on a cascade or two-stage approach. However, current E2E state-of-the-art models are computationally complex and memory-consuming, making them unsuitable for real-time offline on-device applications in low-resource scenarios. To address this issue, we propose a Lightweight E2E-TTS (LE2E) model that generates high-quality speech requiring minimal computational resources. We evaluate the proposed model on the LJSpeech dataset and show that it achieves state-of-the-art performance while being up to $90\%$ smaller in terms of model parameters and $10\times$ faster in real-time-factor. Furthermore, we demonstrate that the proposed E2E training paradigm achieves better quality compared to an equivalent architecture trained in a two-stage approach. Our results suggest that LE2E is a promising approach for developing real-time, high quality, low-resource TTS applications for on-device applications.
#### Spoken Language Understanding on Unseen Tasks With In-Context Learning
 - **Authors:** Neeraj Agrawal, Sriram Ganapathy
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.07731

 - **Pdf link:** https://arxiv.org/pdf/2505.07731

 - **Abstract**
 Spoken language understanding (SLU) tasks involve diverse skills that probe the information extraction, classification and/or generation capabilities of models. In this setting, task-specific training data may not always be available. While traditional task-specific SLU models are unable to cater to such requirements, the speech-text large language models (LLMs) offer a promising alternative with emergent abilities. However, out of-the-box, our evaluations indicate that the zero/few-shot performance of prominent open-source speech-text LLMs on SLU tasks are not up to the mark. In this paper, we introduce a novel approach to robust task-agnostic fine-tuning using randomized class labels. With this proposed fine-tuning, we illustrate that the performance of the speech-text LLMs on an unseen task is significantly improved over standard approaches. Critically, the proposed approach avoids the requirement of task-specific data annotations for enabling new tasks in speech-text LLMs.


by Zyzzyva0381 (Windy). 


2025-05-13
