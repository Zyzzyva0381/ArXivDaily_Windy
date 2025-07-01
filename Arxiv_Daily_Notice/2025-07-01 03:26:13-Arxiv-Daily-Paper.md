# Showing new listings for Tuesday, 1 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 17papers 
#### Speaker Targeting via Self-Speaker Adaptation for Multi-talker ASR
 - **Authors:** Weiqing Wang, Taejin Park, Ivan Medennikov, Jinhan Wang, Kunal Dhawan, He Huang, Nithin Rao Koluguri, Jagadeesh Balam, Boris Ginsburg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.22646

 - **Pdf link:** https://arxiv.org/pdf/2506.22646

 - **Abstract**
 We propose a self-speaker adaptation method for streaming multi-talker automatic speech recognition (ASR) that eliminates the need for explicit speaker queries. Unlike conventional approaches requiring target speaker embeddings or enrollment audio, our technique dynamically adapts individual ASR instances through speaker-wise speech activity prediction. The key innovation involves injecting speaker-specific kernels generated via speaker supervision activations into selected ASR encoder layers. This enables instantaneous speaker adaptation to target speakers while handling fully overlapped speech even in a streaming scenario. Experiments show state-of-the-art performance in both offline and streaming scenarios, demonstrating that our self-adaptive method effectively addresses severe speech overlap through streamlined speaker-focused recognition. The results validate the proposed self-speaker adaptation approach as a robust solution for multi-talker ASR under severe overlapping speech conditions.
#### Adaptable Non-parametric Approach for Speech-based Symptom Assessment: Isolating Private Medical Data in a Retrieval Datastore
 - **Authors:** Yu-Wen Chen, Julia Hirschberg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22972

 - **Pdf link:** https://arxiv.org/pdf/2506.22972

 - **Abstract**
 The automatic assessment of health-related acoustic cues has the potential to improve healthcare accessibility and affordability. Although parametric models are promising, they face challenges in privacy and adaptability. To address these, we propose a NoN-Parametric framework for Speech-based symptom Assessment (NoNPSA). By isolating medical data in a retrieval datastore, NoNPSA avoids encoding private information in model parameters and enables efficient data updates. A self-supervised learning (SSL) model pre-trained on general-purpose datasets extracts features, which are used for similarity-based retrieval. Metadata-aware refinement filters the retrieved data, and associated labels are used to compute an assessment score. Experimental results show that NoNPSA achieves competitive performance compared to fine-tuning SSL-based methods, while enabling greater privacy, update efficiency, and adaptability--showcasing the potential of non-parametric approaches in healthcare.
#### Less is More: Data Curation Matters in Scaling Speech Enhancement
 - **Authors:** Chenda Li, Wangyou Zhang, Wei Wang, Robin Scheibler, Kohei Saijo, Samuele Cornell, Yihui Fu, Marvin Sach, Zhaoheng Ni, Anurag Kumar, Tim Fingscheidt, Shinji Watanabe, Yanmin Qian
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.23859

 - **Pdf link:** https://arxiv.org/pdf/2506.23859

 - **Abstract**
 The vast majority of modern speech enhancement systems rely on data-driven neural network models. Conventionally, larger datasets are presumed to yield superior model performance, an observation empirically validated across numerous tasks in other domains. However, recent studies reveal diminishing returns when scaling speech enhancement data. We focus on a critical factor: prevalent quality issues in ``clean'' training labels within large-scale datasets. This work re-examines this phenomenon and demonstrates that, within large-scale training sets, prioritizing high-quality training data is more important than merely expanding the data volume. Experimental findings suggest that models trained on a carefully curated subset of 700 hours can outperform models trained on the 2,500-hour full dataset. This outcome highlights the crucial role of data curation in scaling speech enhancement systems effectively.
#### URGENT-PK: Perceptually-Aligned Ranking Model Designed for Speech Enhancement Competition
 - **Authors:** Jiahe Wang, Chenda Li, Wei Wang, Wangyou Zhang, Samuele Cornell, Marvin Sach, Robin Scheibler, Kohei Saijo, Yihui Fu, Zhaoheng Ni, Anurag Kumar, Tim Fingscheidt, Shinji Watanabe, Yanmin Qian
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.23874

 - **Pdf link:** https://arxiv.org/pdf/2506.23874

 - **Abstract**
 The Mean Opinion Score (MOS) is fundamental to speech quality assessment. However, its acquisition requires significant human annotation. Although deep neural network approaches, such as DNSMOS and UTMOS, have been developed to predict MOS to avoid this issue, they often suffer from insufficient training data. Recognizing that the comparison of speech enhancement (SE) systems prioritizes a reliable system comparison over absolute scores, we propose URGENT-PK, a novel ranking approach leveraging pairwise comparisons. URGENT-PK takes homologous enhanced speech pairs as input to predict relative quality rankings. This pairwise paradigm efficiently utilizes limited training data, as all pairwise permutations of multiple systems constitute a training instance. Experiments across multiple open test sets demonstrate URGENT-PK's superior system-level ranking performance over state-of-the-art baselines, despite its simple network architecture and limited training data.
#### Enhancing Neural Audio Fingerprint Robustness to Audio Degradation for Music Identification
 - **Authors:** R. O. Araz, G. Cortes-Sebastia, E. Molina, J. Serra, X. Serra, Y. Mitsufuji, D. Bogdanov
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22661

 - **Pdf link:** https://arxiv.org/pdf/2506.22661

 - **Abstract**
 Audio fingerprinting (AFP) allows the identification of unknown audio content by extracting compact representations, termed audio fingerprints, that are designed to remain robust against common audio degradations. Neural AFP methods often employ metric learning, where representation quality is influenced by the nature of the supervision and the utilized loss function. However, recent work unrealistically simulates real-life audio degradation during training, resulting in sub-optimal supervision. Additionally, although several modern metric learning approaches have been proposed, current neural AFP methods continue to rely on the NT-Xent loss without exploring the recent advances or classical alternatives. In this work, we propose a series of best practices to enhance the self-supervision by leveraging musical signal properties and realistic room acoustics. We then present the first systematic evaluation of various metric learning approaches in the context of AFP, demonstrating that a self-supervised adaptation of the triplet loss yields superior performance. Our results also reveal that training with multiple positive samples per anchor has critically different effects across loss functions. Our approach is built upon these insights and achieves state-of-the-art performance on both a large, synthetically degraded dataset and a real-world dataset recorded using microphones in diverse music venues.
#### WavShape: Information-Theoretic Speech Representation Learning for Fair and Privacy-Aware Audio Processing
 - **Authors:** Oguzhan Baser, Ahmet Ege Tanriverdi, Kaan Kale, Sandeep P. Chinchali, Sriram Vishwanath
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22789

 - **Pdf link:** https://arxiv.org/pdf/2506.22789

 - **Abstract**
 Speech embeddings often retain sensitive attributes such as speaker identity, accent, or demographic information, posing risks in biased model training and privacy leakage. We propose WavShape, an information-theoretic speech representation learning framework that optimizes embeddings for fairness and privacy while preserving task-relevant information. We leverage mutual information (MI) estimation using the Donsker-Varadhan formulation to guide an MI-based encoder that systematically filters sensitive attributes while maintaining speech content essential for downstream tasks. Experimental results on three known datasets show that WavShape reduces MI between embeddings and sensitive attributes by up to 81% while retaining 97% of task-relevant information. By integrating information theory with self-supervised speech models, this work advances the development of fair, privacy-aware, and resource-efficient speech systems.
#### A Self-Training Approach for Whisper to Enhance Long Dysarthric Speech Recognition
 - **Authors:** Shiyao Wang, Jiaming Zhou, Shiwan Zhao, Yong Qin
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22810

 - **Pdf link:** https://arxiv.org/pdf/2506.22810

 - **Abstract**
 Dysarthric speech recognition (DSR) enhances the accessibility of smart devices for dysarthric speakers with limited mobility. Previously, DSR research was constrained by the fact that existing datasets typically consisted of isolated words, command phrases, and a limited number of sentences spoken by a few individuals. This constrained research to command-interaction systems and speaker adaptation. The Speech Accessibility Project (SAP) changed this by releasing a large and diverse English dysarthric dataset, leading to the SAP Challenge to build speaker- and text-independent DSR systems. We enhanced the Whisper model's performance on long dysarthric speech via a novel self-training method. This method increased training data and adapted the model to handle potentially incomplete speech segments encountered during inference. Our system achieved second place in both Word Error Rate and Semantic Score in the SAP Challenge.
#### Boosting CTC-Based ASR Using LLM-Based Intermediate Loss Regularization
 - **Authors:** Duygu Altinok
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22846

 - **Pdf link:** https://arxiv.org/pdf/2506.22846

 - **Abstract**
 End-to-end (E2E) automatic speech recognition (ASR) systems have revolutionized the field by integrating all components into a single neural network, with attention-based encoder-decoder models achieving state-of-the-art performance. However, their autoregressive decoding process limits inference speed, making them unsuitable for real-time applications. In contrast, CTC-based models offer faster, non-autoregressive decoding but struggle to model linguistic dependencies effectively. Addressing this challenge, we propose a novel auxiliary loss framework called Language-Aware Intermediate Loss (LAIL) to enhance CTC-based ASR using the linguistic knowledge of large language models (LLMs). By attaching connector layers to intermediate encoder layers, LAIL maps outputs to the embedding space of an LLM and computes a causal language modeling loss during training. This approach enhances linguistic modeling while preserving the computational efficiency of CTC decoding. Using the Conformer architecture and various LLaMA models, we demonstrate significant improvements in Word Error Rate (WER) on the LibriSpeech, TEDLIUM2, and WSJ corpora, achieving state-of-the-art performance for CTC-based ASR with minimal computational overhead.
#### Mind the Gap: Entity-Preserved Context-Aware ASR Structured Transcriptions
 - **Authors:** Duygu Altinok
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22858

 - **Pdf link:** https://arxiv.org/pdf/2506.22858

 - **Abstract**
 Automatic Speech Recognition (ASR) systems, such as Whisper, achieve high transcription accuracy but struggle with named entities and numerical data, especially when proper formatting is required. These issues increase word error rate (WER) and impair semantic understanding in critical domains like legal, financial, and medical applications. We propose a novel training approach that extends the semantic context of ASR models by adding overlapping context windows during training. By sliding 5-second overlaps on both sides of 30-second chunks, we create a 40-second "effective semantic window," improving entity recognition and formatting while focusing predictions on the central 30 seconds. To address entities spanning chunk boundaries, we reassign such entities entirely to the right-hand chunk, ensuring proper formatting. Additionally, enriched training data with embedded entity labels enables the model to learn both recognition and type-specific formatting. Evaluated on the Spoken Wikipedia dataset, our method improves performance across semantic tasks, including named entity recognition (NER) and entity formatting. These results highlight the effectiveness of context-aware training in addressing ASR limitations for long-form transcription and complex entity recognition tasks.
#### Feasibility of spectral-element modeling of wave propagation through the anatomy of marine mammals
 - **Authors:** Carlos García A., Vladimiro Boselli, Aida Hejazi Nooghabi, Andrea Colombi, Lapo Boschi
 - **Subjects:** Subjects:
Computational Engineering, Finance, and Science (cs.CE); Sound (cs.SD); Audio and Speech Processing (eess.AS); Tissues and Organs (q-bio.TO)
 - **Arxiv link:** https://arxiv.org/abs/2506.22944

 - **Pdf link:** https://arxiv.org/pdf/2506.22944

 - **Abstract**
 This study introduces the first 3D spectral-element method (SEM) simulation of ultrasonic wave propagation in a bottlenose dolphin (Tursiops truncatus) head. Unlike traditional finite-element methods (FEM), which struggle with high-frequency simulations due to costly linear-system inversions and slower convergence, SEM offers exponential convergence and efficient parallel computation. Using Computed Tomography (CT) scan data, we developed a detailed hexahedral mesh capturing complex anatomical features, such as acoustic fats and jaws. Our simulations of plane and spherical waves confirm SEM's effectiveness for ultrasonic time-domain modeling. This approach opens new avenues for marine biology, contributing to research in echolocation, the impacts of anthropogenic marine noise pollution and the biophysics of hearing and click generation in marine mammals. By overcoming FEM's limitations, SEM provides a powerful scalable tool to test hypotheses about dolphin bioacoustics, with significant implications for conservation and understanding marine mammal auditory systems under increasing environmental challenges.
#### VisionScores -- A system-segmented image score dataset for deep learning tasks
 - **Authors:** Alejandro Romero Amezcua, Mariano José Juan Rivera Meraz
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.23030

 - **Pdf link:** https://arxiv.org/pdf/2506.23030

 - **Abstract**
 VisionScores presents a novel proposal being the first system-segmented image score dataset, aiming to offer structure-rich, high information-density images for machine and deep learning tasks. Delimited to two-handed piano pieces, it was built to consider not only certain graphic similarity but also composition patterns, as this creative process is highly instrument-dependent. It provides two scenarios in relation to composer and composition type. The first, formed by 14k samples, considers works from different authors but the same composition type, specifically, Sonatinas. The latter, consisting of 10.8K samples, presents the opposite case, various composition types from the same author, being the one selected Franz Liszt. All of the 24.8k samples are formatted as grayscale jpg images of $128 \times 512$ pixels. VisionScores supplies the users not only the formatted samples but the systems' order and pieces' metadata. Moreover, unsegmented full-page scores and the pre-formatted images are included for further analysis.
#### AURA: Agent for Understanding, Reasoning, and Automated Tool Use in Voice-Driven Tasks
 - **Authors:** Leander Melroy Maben, Gayathri Ganesh Lakshmy, Srijith Radhakrishnan, Siddhant Arora, Shinji Watanabe
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.23049

 - **Pdf link:** https://arxiv.org/pdf/2506.23049

 - **Abstract**
 Despite advances in language and speech technologies, no open-source system enables full speech-to-speech, multi-turn dialogue with integrated tool use and agentic reasoning. We introduce AURA (Agent for Understanding, Reasoning, and Automated Tool Use), the first open-source, speech-native assistant capable of completing complex, goal-driven tasks through dynamic tool invocation and multi-turn conversation. AURA combines open-weight ASR, TTS, and LLMs in a cascaded pipeline and supports tools such as calendar booking, contact lookup, web search, and email. Its modular design allows easy integration of new tools using natural language prompts and action classes. On VoiceBench, AURA scores 92.75% on OpenBookQA-outperforming all open-weight systems and nearing GPT-4o-and 4.39 on AlpacaEval, competitive with other open-weight systems. Human evaluation shows 90% task success on complex, multi-turn speech tasks.
#### XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs
 - **Authors:** Yitian Gong, Luozhijie Jin, Ruifan Deng, Dong Zhang, Xin Zhang, Qinyuan Cheng, Zhaoye Fei, Shimin Li, Xipeng Qiu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.23325

 - **Pdf link:** https://arxiv.org/pdf/2506.23325

 - **Abstract**
 Speech codecs serve as bridges between speech signals and large language models. An ideal codec for speech language models should not only preserve acoustic information but also capture rich semantic information. However, existing speech codecs struggle to balance high-quality audio reconstruction with ease of modeling by language models. In this study, we analyze the limitations of previous codecs in balancing semantic richness and acoustic fidelity. We propose XY-Tokenizer, a novel codec that mitigates the conflict between semantic and acoustic capabilities through multi-stage, multi-task learning. Experimental results demonstrate that XY-Tokenizer achieves performance in both semantic and acoustic tasks comparable to that of state-of-the-art codecs operating at similar bitrates, even though those existing codecs typically excel in only one aspect. Specifically, XY-Tokenizer achieves strong text alignment, surpassing distillation-based semantic modeling methods such as SpeechTokenizer and Mimi, while maintaining a speaker similarity score of 0.83 between reconstructed and original audio. The reconstruction performance of XY-Tokenizer is comparable to that of BigCodec, the current state-of-the-art among acoustic-only codecs, which achieves a speaker similarity score of 0.84 at a similar bitrate. Code and models are available at this https URL.
#### You Sound a Little Tense: L2 Tailored Clear TTS Using Durational Vowel Properties
 - **Authors:** Paige Tuttösí, H. Henny Yeung, Yue Wang, Jean-Julien Aucouturier, Angelica Lim
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.23367

 - **Pdf link:** https://arxiv.org/pdf/2506.23367

 - **Abstract**
 We present the first text-to-speech (TTS) system tailored to second language (L2) speakers. We use duration differences between American English tense (longer) and lax (shorter) vowels to create a "clarity mode" for Matcha-TTS. Our perception studies showed that French-L1, English-L2 listeners had fewer (at least 9.15%) transcription errors when using our clarity mode, and found it more encouraging and respectful than overall slowed down speech. Remarkably, listeners were not aware of these effects: despite the decreased word error rate in clarity mode, listeners still believed that slowing all target words was the most intelligible, suggesting that actual intelligibility does not correlate with perceived intelligibility. Additionally, we found that Whisper-ASR did not use the same cues as L2 speakers to differentiate difficult vowels and is not sufficient to assess the intelligibility of TTS systems for these individuals.
#### JAM-Flow: Joint Audio-Motion Synthesis with Flow Matching
 - **Authors:** Mingi Kwon, Joonghyuk Shin, Jaeseok Jung, Jaesik Park, Youngjung Uh
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.23552

 - **Pdf link:** https://arxiv.org/pdf/2506.23552

 - **Abstract**
 The intrinsic link between facial motion and speech is often overlooked in generative modeling, where talking head synthesis and text-to-speech (TTS) are typically addressed as separate tasks. This paper introduces JAM-Flow, a unified framework to simultaneously synthesize and condition on both facial motion and speech. Our approach leverages flow matching and a novel Multi-Modal Diffusion Transformer (MM-DiT) architecture, integrating specialized Motion-DiT and Audio-DiT modules. These are coupled via selective joint attention layers and incorporate key architectural choices, such as temporally aligned positional embeddings and localized joint attention masking, to enable effective cross-modal interaction while preserving modality-specific strengths. Trained with an inpainting-style objective, JAM-Flow supports a wide array of conditioning inputs-including text, reference audio, and reference motion-facilitating tasks such as synchronized talking head generation from text, audio-driven animation, and much more, within a single, coherent model. JAM-Flow significantly advances multi-modal generative modeling by providing a practical solution for holistic audio-visual synthesis. project page: this https URL
#### Efficient Interleaved Speech Modeling through Knowledge Distillation
 - **Authors:** Mohammadmahdi Nouriborji, Morteza Rohanian
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.23670

 - **Pdf link:** https://arxiv.org/pdf/2506.23670

 - **Abstract**
 Current speech language models exceed the size and latency constraints of many deployment environments. We build compact, expressive speech generation models through layer-aligned distillation, matching hidden states, attention maps, and softened logits to compress large multimodal transformers by 3x with minimal loss in performance. We introduce TinyWave, a family of 2B-parameter models for speech-to-speech and interleaved speech-text generation, trained on 50,000 hours of public audio. TinyWave supports (i) speech-only generation using phonetic or expressive tokens and (ii) mixed speech-text continuations. Evaluation on Libri-Light shows TinyWave within 1.4 normalized perplexity points of its teacher. Accuracy on spoken StoryCloze and SALMon reaches 93-97% of the teacher's performance, outperforming size-matched baselines. These models are optimized for deployment on commodity hardware, enabling applications in real-time conversational agents, assistive technologies, and low-resource environments. We release models, training code, and evaluation scripts to support reproducible research on compact, expressive speech generation.
#### StreamFlow: Streaming Flow Matching with Block-wise Guided Attention Mask for Speech Token Decoding
 - **Authors:** Dake Guo, Jixun Yao, Linhan Ma, Wang He, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.23986

 - **Pdf link:** https://arxiv.org/pdf/2506.23986

 - **Abstract**
 Recent advancements in discrete token-based speech generation have highlighted the importance of token-to-waveform generation for audio quality, particularly in real-time interactions. Traditional frameworks integrating semantic tokens with flow matching (FM) struggle with streaming capabilities due to their reliance on a global receptive field. Additionally, directly implementing token-by-token streaming speech generation often results in degraded audio quality. To address these challenges, we propose StreamFlow, a novel neural architecture that facilitates streaming flow matching with diffusion transformers (DiT). To mitigate the long-sequence extrapolation issues arising from lengthy historical dependencies, we design a local block-wise receptive field strategy. Specifically, the sequence is first segmented into blocks, and we introduce block-wise attention masks that enable the current block to receive information from the previous or subsequent block. These attention masks are combined hierarchically across different DiT-blocks to regulate the receptive field of DiTs. Both subjective and objective experimental results demonstrate that our approach achieves performance comparable to non-streaming methods while surpassing other streaming methods in terms of speech quality, all the while effectively managing inference time during long-sequence generation. Furthermore, our method achieves a notable first-packet latency of only 180 ms.\footnote{Speech samples: this https URL}


by Zyzzyva0381 (Windy). 


2025-07-01
