# Showing new listings for Thursday, 5 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 15papers 
#### Towards Source Attribution of Singing Voice Deepfake with Multimodal Foundation Models
 - **Authors:** Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Swarup Ranjan Behera, Priyabrata Mallick, Pailla Balakrishna Reddy, Arun Balaji Buduru, Rajesh Sharma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.03364

 - **Pdf link:** https://arxiv.org/pdf/2506.03364

 - **Abstract**
 In this work, we introduce the task of singing voice deepfake source attribution (SVDSA). We hypothesize that multimodal foundation models (MMFMs) such as ImageBind, LanguageBind will be most effective for SVDSA as they are better equipped for capturing subtle source-specific characteristics-such as unique timbre, pitch manipulation, or synthesis artifacts of each singing voice deepfake source due to their cross-modality pre-training. Our experiments with MMFMs, speech foundation models and music foundation models verify the hypothesis that MMFMs are the most effective for SVDSA. Furthermore, inspired from related research, we also explore fusion of foundation models (FMs) for improved SVDSA. To this end, we propose a novel framework, COFFE which employs Chernoff Distance as novel loss function for effective fusion of FMs. Through COFFE with the symphony of MMFMs, we attain the topmost performance in comparison to all the individual FMs and baseline fusion methods.
#### HYFuse: Aligning Heterogeneous Speech Pre-Trained Representations in Hyperbolic Space for Speech Emotion Recognition
 - **Authors:** Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Swarup Ranjan Behera, Pailla Balakrishna Reddy, Arun Balaji Buduru, Rajesh Sharma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.03403

 - **Pdf link:** https://arxiv.org/pdf/2506.03403

 - **Abstract**
 Compression-based representations (CBRs) from neural audio codecs such as EnCodec capture intricate acoustic features like pitch and timbre, while representation-learning-based representations (RLRs) from pre-trained models trained for speech representation learning such as WavLM encode high-level semantic and prosodic information. Previous research on Speech Emotion Recognition (SER) has explored both, however, fusion of CBRs and RLRs haven't been explored yet. In this study, we solve this gap and investigate the fusion of RLRs and CBRs and hypothesize they will be more effective by providing complementary information. To this end, we propose, HYFuse, a novel framework that fuses the representations by transforming them to hyperbolic space. With HYFuse, through fusion of x-vector (RLR) and Soundstream (CBR), we achieve the top performance in comparison to individual representations as well as the homogeneous fusion of RLRs and CBRs and report SOTA.
#### BitTTS: Highly Compact Text-to-Speech Using 1.58-bit Quantization and Weight Indexing
 - **Authors:** Masaya Kawamura, Takuya Hasumi, Yuma Shirahata, Ryuichi Yamamoto
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.03515

 - **Pdf link:** https://arxiv.org/pdf/2506.03515

 - **Abstract**
 This paper proposes a highly compact, lightweight text-to-speech (TTS) model for on-device applications. To reduce the model size, the proposed model introduces two techniques. First, we introduce quantization-aware training (QAT), which quantizes model parameters during training to as low as 1.58-bit. In this case, most of 32-bit model parameters are quantized to ternary values {-1, 0, 1}. Second, we propose a method named weight indexing. In this method, we save a group of 1.58-bit weights as a single int8 index. This allows for efficient storage of model parameters, even on hardware that treats values in units of 8-bit. Experimental results demonstrate that the proposed method achieved 83 % reduction in model size, while outperforming the baseline of similar model size without quantization in synthesis quality.
#### Tone recognition in low-resource languages of North-East India: peeling the layers of SSL-based speech models
 - **Authors:** Parismita Gogoi, Sishir Kalita, Wendy Lalhminghlui, Viyazonuo Terhiija, Moakala Tzudir, Priyankoo Sarmah, S. R. M. Prasanna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.03606

 - **Pdf link:** https://arxiv.org/pdf/2506.03606

 - **Abstract**
 This study explores the use of self-supervised learning (SSL) models for tone recognition in three low-resource languages from North Eastern India: Angami, Ao, and Mizo. We evaluate four Wav2vec2.0 base models that were pre-trained on both tonal and non-tonal languages. We analyze tone-wise performance across the layers for all three languages and compare the different models. Our results show that tone recognition works best for Mizo and worst for Angami. The middle layers of the SSL models are the most important for tone recognition, regardless of the pre-training language, i.e. tonal or non-tonal. We have also found that the tone inventory, tone types, and dialectal variations affect tone recognition. These findings provide useful insights into the strengths and weaknesses of SSL-based embeddings for tonal languages and highlight the potential for improving tone recognition in low-resource settings. The source code is available at GitHub 1 .
#### Sound Field Reconstruction Using Physics-Informed Boundary Integral Networks
 - **Authors:** Stefano Damiano, Toon van Waterschoot
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.03917

 - **Pdf link:** https://arxiv.org/pdf/2506.03917

 - **Abstract**
 Sound field reconstruction refers to the problem of estimating the acoustic pressure field over an arbitrary region of space, using only a limited set of measurements. Physics-informed neural networks have been adopted to solve the problem by incorporating in the training loss function the governing partial differential equation, either the Helmholtz or the wave equation. In this work, we introduce a boundary integral network for sound field reconstruction. Relying on the Kirchhoff-Helmholtz boundary integral equation to model the sound field in a given region of space, we employ a shallow neural network to retrieve the pressure distribution on the boundary of the considered domain, enabling to accurately retrieve the acoustic pressure inside of it. Assuming the positions of measurement microphones are known, we train the model by minimizing the mean squared error between the estimated and measured pressure at those locations. Experimental results indicate that the proposed model outperforms existing physics-informed data-driven techniques.
#### HiFiTTS-2: A Large-Scale High Bandwidth Speech Dataset
 - **Authors:** Ryan Langman, Xuesong Yang, Paarth Neekhara, Shehzeen Hussain, Edresson Casanova, Evelina Bakhturina, Jason Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04152

 - **Pdf link:** https://arxiv.org/pdf/2506.04152

 - **Abstract**
 This paper introduces HiFiTTS-2, a large-scale speech dataset designed for high-bandwidth speech synthesis. The dataset is derived from LibriVox audiobooks, and contains approximately 36.7k hours of English speech for 22.05 kHz training, and 31.7k hours for 44.1 kHz training. We present our data processing pipeline, including bandwidth estimation, segmentation, text preprocessing, and multi-speaker detection. The dataset is accompanied by detailed utterance and audiobook metadata generated by our pipeline, enabling researchers to apply data quality filters to adapt the dataset to various use cases. Experimental results demonstrate that our data pipeline and resulting dataset can facilitate the training of high-quality, zero-shot text-to-speech (TTS) models at high bandwidths.
#### Comparative Analysis of Fast and High-Fidelity Neural Vocoders for Low-Latency Streaming Synthesis in Resource-Constrained Environments
 - **Authors:** Reo Yoneyama, Masaya Kawamura, Ryo Terashima, Ryuichi Yamamoto, Tomoki Toda
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.03554

 - **Pdf link:** https://arxiv.org/pdf/2506.03554

 - **Abstract**
 In real-time speech synthesis, neural vocoders often require low-latency synthesis through causal processing and streaming. However, streaming introduces inefficiencies absent in batch synthesis, such as limited parallelism, inter-frame dependency management, and parameter loading overhead. This paper proposes multi-stream Wavehax (MS-Wavehax), an efficient neural vocoder for low-latency streaming, by extending the aliasing-free neural vocoder Wavehax with multi-stream decomposition. We analyze the latency-throughput trade-off in a CPU-only environment and identify key bottlenecks in streaming neural vocoders. Our findings provide practical insights for optimizing chunk sizes and designing vocoders tailored to specific application demands and hardware constraints. Furthermore, our subjective evaluations show that MS-Wavehax delivers high speech quality under causal and non-causal conditions while being remarkably compact and easily deployable in resource-constrained environments.
#### MFLA: Monotonic Finite Look-ahead Attention for Streaming Speech Recognition
 - **Authors:** Yinfeng Xia, Huiyan Li, Chenyang Le, Manhong Wang, Yutao Sun, Xingyang Ma, Yanmin Qian
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.03722

 - **Pdf link:** https://arxiv.org/pdf/2506.03722

 - **Abstract**
 Applying large pre-trained speech models like Whisper has shown promise in reducing training costs for various speech tasks. However, integrating these models into streaming systems remains a challenge. This paper presents a novel prefix-to-prefix training framework for streaming recognition by fine-tuning the Whisper. We introduce the Continuous Integrate-and-Fire mechanism to establish a quasi-monotonic alignment between continuous speech sequences and discrete text tokens. Additionally, we design Monotonic Finite Look-ahead Attention, allowing each token to attend to infinite left-context and finite right-context from the speech sequences. We also employ the wait-k decoding strategy to simplify the decoding process while ensuring consistency between training and testing. Our theoretical analysis and experiments demonstrate that this approach achieves a controllable trade-off between latency and quality, making it suitable for various streaming applications.
#### Brain-tuned Speech Models Better Reflect Speech Processing Stages in the Brain
 - **Authors:** Omer Moussa, Mariya Toneva
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS); Neurons and Cognition (q-bio.NC)
 - **Arxiv link:** https://arxiv.org/abs/2506.03832

 - **Pdf link:** https://arxiv.org/pdf/2506.03832

 - **Abstract**
 Pretrained self-supervised speech models excel in speech tasks but do not reflect the hierarchy of human speech processing, as they encode rich semantics in middle layers and poor semantics in late layers. Recent work showed that brain-tuning (fine-tuning models using human brain recordings) improves speech models' semantic understanding. Here, we examine how well brain-tuned models further reflect the brain's intermediate stages of speech processing. We find that late layers of brain-tuned models substantially improve over pretrained models in their alignment with semantic language regions. Further layer-wise probing reveals that early layers remain dedicated to low-level acoustic features, while late layers become the best at complex high-level tasks. These findings show that brain-tuned models not only perform better but also exhibit a well-defined hierarchical processing going from acoustic to semantic representations, making them better model organisms for human speech processing.
#### Towards Better Disentanglement in Non-Autoregressive Zero-Shot Expressive Voice Conversion
 - **Authors:** Seymanur Akti, Tuan Nam Nguyen, Alexander Waibel
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04013

 - **Pdf link:** https://arxiv.org/pdf/2506.04013

 - **Abstract**
 Expressive voice conversion aims to transfer both speaker identity and expressive attributes from a target speech to a given source speech. In this work, we improve over a self-supervised, non-autoregressive framework with a conditional variational autoencoder, focusing on reducing source timbre leakage and improving linguistic-acoustic disentanglement for better style transfer. To minimize style leakage, we use multilingual discrete speech units for content representation and reinforce embeddings with augmentation-based similarity loss and mix-style layer normalization. To enhance expressivity transfer, we incorporate local F0 information via cross-attention and extract style embeddings enriched with global pitch and energy features. Experiments show our model outperforms baselines in emotion and speaker similarity, demonstrating superior style adaptation and reduced source style leakage.
#### The mutual exclusivity bias of bilingual visually grounded speech models
 - **Authors:** Dan Oneata, Leanne Nortje, Yevgen Matusevych, Herman Kamper
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04037

 - **Pdf link:** https://arxiv.org/pdf/2506.04037

 - **Abstract**
 Mutual exclusivity (ME) is a strategy where a novel word is associated with a novel object rather than a familiar one, facilitating language learning in children. Recent work has found an ME bias in a visually grounded speech (VGS) model trained on English speech with paired images. But ME has also been studied in bilingual children, who may employ it less due to cross-lingual ambiguity. We explore this pattern computationally using bilingual VGS models trained on combinations of English, French, and Dutch. We find that bilingual models generally exhibit a weaker ME bias than monolingual models, though exceptions exist. Analyses show that the combined visual embeddings of bilingual models have a smaller variance for familiar data, partly explaining the increase in confusion between novel and familiar concepts. We also provide new insights into why the ME bias exists in VGS models in the first place. Code and data: this https URL
#### A Statistics-Driven Differentiable Approach for Sound Texture Synthesis and Analysis
 - **Authors:** Esteban Gutiérrez, Frederic Font, Xavier Serra, Lonce Wyse
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04073

 - **Pdf link:** https://arxiv.org/pdf/2506.04073

 - **Abstract**
 In this work, we introduce TexStat, a novel loss function specifically designed for the analysis and synthesis of texture sounds characterized by stochastic structure and perceptual stationarity. Drawing inspiration from the statistical and perceptual framework of McDermott and Simoncelli, TexStat identifies similarities between signals belonging to the same texture category without relying on temporal structure. We also propose using TexStat as a validation metric alongside Frechet Audio Distances (FAD) to evaluate texture sound synthesis models. In addition to TexStat, we present TexEnv, an efficient, lightweight and differentiable texture sound synthesizer that generates audio by imposing amplitude envelopes on filtered noise. We further integrate these components into TexDSP, a DDSP-inspired generative model tailored for texture sounds. Through extensive experiments across various texture sound types, we demonstrate that TexStat is perceptually meaningful, time-invariant, and robust to noise, features that make it effective both as a loss function for generative tasks and as a validation metric. All tools and code are provided as open-source contributions and our PyTorch implementations are efficient, differentiable, and highly configurable, enabling its use in both generative tasks and as a perceptually grounded evaluation metric.
#### Acoustically Precise Hesitation Tagging Is Essential for End-to-End Verbatim Transcription Systems
 - **Authors:** Jhen-Ke Lin, Hao-Chien Lu, Chung-Chun Wang, Hong-Yun Lin, Berlin Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04076

 - **Pdf link:** https://arxiv.org/pdf/2506.04076

 - **Abstract**
 Verbatim transcription for automatic speaking assessment demands accurate capture of disfluencies, crucial for downstream tasks like error analysis and feedback. However, many ASR systems discard or generalize hesitations, losing important acoustic details. We fine-tune Whisper models on the Speak & Improve 2025 corpus using low-rank adaptation (LoRA), without recourse to external audio training data. We compare three annotation schemes: removing hesitations (Pure), generic tags (Rich), and acoustically precise fillers inferred by Gemini 2.0 Flash from existing audio-transcript pairs (Extra). Our challenge system achieved 6.47% WER (Pure) and 5.81% WER (Extra). Post-challenge experiments reveal that fine-tuning Whisper Large V3 Turbo with the "Extra" scheme yielded a 5.5% WER, an 11.3% relative improvement over the "Pure" scheme (6.2% WER). This demonstrates that explicit, realistic filled-pause labeling significantly enhances ASR accuracy for verbatim L2 speech transcription.
#### A Novel Data Augmentation Approach for Automatic Speaking Assessment on Opinion Expressions
 - **Authors:** Chung-Chun Wang, Jhen-Ke Lin, Hao-Chien Lu, Hong-Yun Lin, Berlin Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04077

 - **Pdf link:** https://arxiv.org/pdf/2506.04077

 - **Abstract**
 Automated speaking assessment (ASA) on opinion expressions is often hampered by the scarcity of labeled recordings, which restricts prompt diversity and undermines scoring reliability. To address this challenge, we propose a novel training paradigm that leverages a large language models (LLM) to generate diverse responses of a given proficiency level, converts responses into synthesized speech via speaker-aware text-to-speech synthesis, and employs a dynamic importance loss to adaptively reweight training instances based on feature distribution differences between synthesized and real speech. Subsequently, a multimodal large language model integrates aligned textual features with speech signals to predict proficiency scores directly. Experiments conducted on the LTTC dataset show that our approach outperforms methods relying on real data or conventional augmentation, effectively mitigating low-resource constraints and enabling ASA on opinion expressions with cross-modal information.
#### UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation
 - **Authors:** Jinting Wang, Shan Yang, Li Liu
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.04134

 - **Pdf link:** https://arxiv.org/pdf/2506.04134

 - **Abstract**
 Cued Speech (CS) enhances lipreading through hand coding, providing precise speech perception support for the hearing-impaired. CS Video-to-Speech generation (CSV2S) task aims to convert the CS visual expressions (CS videos) of hearing-impaired individuals into comprehensible speech signals. Direct generation of speech from CS video (called single CSV2S) yields poor performance due to insufficient CS data. Current research mostly focuses on CS Recognition (CSR), which convert video content into linguistic text. Based on this, one straightforward way of CSV2S is to combine CSR with a Text-to-Speech system. This combined architecture relies on text as an intermediate medium for stepwise cross-modal alignment, which may lead to error propagation and temporal misalignment between speech and video dynamics. To address these challenges, we propose a novel approach that directly generates speech from CS videos without relying on intermediate text. Building upon this, we propose UniCUE, the first unified framework for CSV2S, whose core innovation lies in the integration of the CSR task that provides fine-grained visual-semantic information to facilitate speech generation from CS videos. More precisely, (1) a novel fine-grained semantic alignment pool to ensure precise mapping between visual features and speech contents; (2) a VisioPhonetic adapter to bridge cross-task representations, ensuring seamless compatibility between two distinct tasks (i.e., CSV2S and CSR); (3) a pose-aware visual processor is introduced to enhance fine-grained spatiotemporal correlations between lip and hand movements in CS video. Experiments on our new established Chinese CS dataset (14 cuers1: 8 hearing-impaired and 6 normal-hearing) show that our UniCUE significantly reduces Word Error Rate by 78.3% and improves lip-speech synchronization by 32% compared to the single CSV2S.


by Zyzzyva0381 (Windy). 


2025-06-05
