# Showing new listings for Tuesday, 3 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 64papers 
#### Pushing the Limits of Beam Search Decoding for Transducer-based ASR models
 - **Authors:** Lilit Grigoryan, Vladimir Bataev, Andrei Andrusenko, Hainan Xu, Vitaly Lavrukhin, Boris Ginsburg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.00185

 - **Pdf link:** https://arxiv.org/pdf/2506.00185

 - **Abstract**
 Transducer models have emerged as a promising choice for end-to-end ASR systems, offering a balanced trade-off between recognition accuracy, streaming capabilities, and inference speed in greedy decoding. However, beam search significantly slows down Transducers due to repeated evaluations of key network components, limiting practical applications. This paper introduces a universal method to accelerate beam search for Transducers, enabling the implementation of two optimized algorithms: ALSD++ and AES++. The proposed method utilizes batch operations, a tree-based hypothesis structure, novel blank scoring for enhanced shallow fusion, and CUDA graph execution for efficient GPU inference. This narrows the speed gap between beam and greedy modes to only 10-20% for the whole system, achieves 14-30% relative improvement in WER compared to greedy decoding, and improves shallow fusion for low-resource up to 11% compared to existing implementations. All the algorithms are open sourced.
#### Towards Temporally Explainable Dysarthric Speech Clarity Assessment
 - **Authors:** Seohyun Park, Chitralekha Gupta, Michelle Kah Yian Kwan, Xinhui Fung, Alexander Wenjun Yip, Suranga Nanayakkara
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Human-Computer Interaction (cs.HC); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.00454

 - **Pdf link:** https://arxiv.org/pdf/2506.00454

 - **Abstract**
 Dysarthria, a motor speech disorder, affects intelligibility and requires targeted interventions for effective communication. In this work, we investigate automated mispronunciation feedback by collecting a dysarthric speech dataset from six speakers reading two passages, annotated by a speech therapist with temporal markers and mispronunciation descriptions. We design a three-stage framework for explainable mispronunciation evaluation: (1) overall clarity scoring, (2) mispronunciation localization, and (3) mispronunciation type classification. We systematically analyze pretrained Automatic Speech Recognition (ASR) models in each stage, assessing their effectiveness in dysarthric speech evaluation (Code available at: this https URL, Supplementary webpage: this https URL). Our findings offer clinically relevant insights for automating actionable feedback for pronunciation assessment, which could enable independent practice for patients and help therapists deliver more effective interventions.
#### M3ANet: Multi-scale and Multi-Modal Alignment Network for Brain-Assisted Target Speaker Extraction
 - **Authors:** Cunhang Fan, Ying Chen, Jian Zhou, Zexu Pan, Jingjing Zhang, Youdian Gao, Xiaoke Yang, Zhengqi Wen, Zhao Lv
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.00466

 - **Pdf link:** https://arxiv.org/pdf/2506.00466

 - **Abstract**
 The brain-assisted target speaker extraction (TSE) aims to extract the attended speech from mixed speech by utilizing the brain neural activities, for example Electroencephalography (EEG). However, existing models overlook the issue of temporal misalignment between speech and EEG modalities, which hampers TSE performance. In addition, the speech encoder in current models typically uses basic temporal operations (e.g., one-dimensional convolution), which are unable to effectively extract target speaker information. To address these issues, this paper proposes a multi-scale and multi-modal alignment network (M3ANet) for brain-assisted TSE. Specifically, to eliminate the temporal inconsistency between EEG and speech modalities, the modal alignment module that uses a contrastive learning strategy is applied to align the temporal features of both modalities. Additionally, to fully extract speech information, multi-scale convolutions with GroupMamba modules are used as the speech encoder, which scans speech features at each scale from different directions, enabling the model to capture deep sequence information. Experimental results on three publicly available datasets show that the proposed model outperforms current state-of-the-art methods across various evaluation metrics, highlighting the effectiveness of our proposed method. The source code is available at: this https URL.
#### Quality Assessment of Noisy and Enhanced Speech with Limited Data: UWB-NTIS System for VoiceMOS 2024 and Beyond
 - **Authors:** Marie Kunešová
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.00506

 - **Pdf link:** https://arxiv.org/pdf/2506.00506

 - **Abstract**
 In this preprint, we present the UWB-NTIS-TTS team's submission to Track 3 of the VoiceMOS 2024 Challenge, the goal of which was to automatically assess the speech quality of noisy and de-noised speech in terms of the ITU-T P.835 metrics of "SIG", "BAK", and "OVRL". Our proposed system, based on wav2vec 2.0, placed among the top systems in the challenge, achieving the best prediction of the BAK scores (background noise intrusiveness), the second-best prediction of the OVRL score (overall audio quality), and the third-best prediction of SIG (speech signal quality) out of the five participating systems. We describe our approach, such as the two-stage fine-tuning process we used to contend with the challenge's very limiting restrictions on allowable training data, and present the results achieved both on the VoiceMOS 2024 Challenge data and on the recently released CHiME 7 - UDASE dataset.
#### Quantifying and Reducing Speaker Heterogeneity within the Common Voice Corpus for Phonetic Analysis
 - **Authors:** Miao Zhang, Aref Farhadipour, Annie Baker, Jiachen Ma, Bogdan Pricop, Eleanor Chodroff
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.00733

 - **Pdf link:** https://arxiv.org/pdf/2506.00733

 - **Abstract**
 With its crosslinguistic and cross-speaker diversity, the Mozilla Common Voice Corpus (CV) has been a valuable resource for multilingual speech technology and holds tremendous potential for research in crosslinguistic phonetics and speech sciences. Properly accounting for speaker variation is, however, key to the theoretical and statistical bases of speech research. While CV provides a client ID as an approximation to a speaker ID, multiple speakers can contribute under the same ID. This study aims to quantify and reduce heterogeneity in the client ID for a better approximation of a true, though still anonymous speaker ID. Using ResNet-based voice embeddings, we obtained a similarity score among recordings with the same client ID, then implemented a speaker discrimination task to identify an optimal threshold for reducing perceived speaker heterogeneity. These results have major downstream applications for phonetic analysis and the development of speaker-based speech technology.
#### CLAP-ART: Automated Audio Captioning with Semantic-rich Audio Representation Tokenizer
 - **Authors:** Daiki Takeuchi, Binh Thien Nguyen, Masahiro Yasuda, Yasunori Ohishi, Daisuke Niizumi, Noboru Harada
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.00800

 - **Pdf link:** https://arxiv.org/pdf/2506.00800

 - **Abstract**
 Automated Audio Captioning (AAC) aims to describe the semantic contexts of general sounds, including acoustic events and scenes, by leveraging effective acoustic features. To enhance performance, an AAC method, EnCLAP, employed discrete tokens from EnCodec as an effective input for fine-tuning a language model BART. However, EnCodec is designed to reconstruct waveforms rather than capture the semantic contexts of general sounds, which AAC should describe. To address this issue, we propose CLAP-ART, an AAC method that utilizes ``semantic-rich and discrete'' tokens as input. CLAP-ART computes semantic-rich discrete tokens from pre-trained audio representations through vector quantization. We experimentally confirmed that CLAP-ART outperforms baseline EnCLAP on two AAC benchmarks, indicating that semantic-rich discrete tokens derived from semantically rich AR are beneficial for AAC.
#### HASRD: Hierarchical Acoustic and Semantic Representation Disentanglement
 - **Authors:** Amir Hussein, Sameer Khurana, Gordon Wichern, Francois G. Germain, Jonathan Le Roux
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.00843

 - **Pdf link:** https://arxiv.org/pdf/2506.00843

 - **Abstract**
 Effective speech representations for spoken language models must balance semantic relevance with acoustic fidelity for high-quality reconstruction. However, existing approaches struggle to achieve both simultaneously. To address this, we introduce Hierarchical Acoustic and Semantic Representation Disentanglement (HASRD, pronounced `hazard'), a framework that factorizes self-supervised learning representations into discrete semantic and acoustic tokens. HASRD assigns the semantic representation to the first codebook, while encoding acoustic residuals in subsequent codebooks. This preserves ASR performance while achieving high-quality reconstruction. Additionally, we enhance HASRD's encoder efficiency, improving ASR performance without compromising reconstruction quality. Compared to SpeechTokenizer, HASRD achieves a 44% relative WER improvement, superior reconstruction quality, and 2x lower bitrate, demonstrating its effectiveness in disentangling acoustic and semantic information.
#### Leveraging AM and FM Rhythm Spectrograms for Dementia Classification and Assessment
 - **Authors:** Parismita Gogoi, Vishwanath Pratap Singh, Seema Khadirnaikar, Soma Siddhartha, Sishir Kalita, Jagabandhu Mishra, Md Sahidullah, Priyankoo Sarmah, S. R. M. Prasanna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.00861

 - **Pdf link:** https://arxiv.org/pdf/2506.00861

 - **Abstract**
 This study explores the potential of Rhythm Formant Analysis (RFA) to capture long-term temporal modulations in dementia speech. Specifically, we introduce RFA-derived rhythm spectrograms as novel features for dementia classification and regression tasks. We propose two methodologies: (1) handcrafted features derived from rhythm spectrograms, and (2) a data-driven fusion approach, integrating proposed RFA-derived rhythm spectrograms with vision transformer (ViT) for acoustic representations along with BERT-based linguistic embeddings. We compare these with existing features. Notably, our handcrafted features outperform eGeMAPs with a relative improvement of $14.2\%$ in classification accuracy and comparable performance in the regression task. The fusion approach also shows improvement, with RFA spectrograms surpassing Mel spectrograms in classification by around a relative improvement of $13.1\%$ and a comparable regression score with the baselines.
#### Crowdsourcing MUSHRA Tests in the Age of Generative Speech Technologies: A Comparative Analysis of Subjective and Objective Testing Methods
 - **Authors:** Laura Lechler, Chamran Moradi, Ivana Balic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.00950

 - **Pdf link:** https://arxiv.org/pdf/2506.00950

 - **Abstract**
 The MUSHRA framework is widely used for detecting subtle audio quality differences but traditionally relies on expert listeners in controlled environments, making it costly and impractical for model development. As a result, objective metrics are often used during development, with expert evaluations conducted later. While effective for traditional DSP codecs, these metrics often fail to reliably evaluate generative models. This paper proposes adaptations for conducting MUSHRA tests with non-expert, crowdsourced listeners, focusing on generative speech codecs. We validate our approach by comparing results from MTurk and Prolific crowdsourcing platforms with expert listener data, assessing test-retest reliability and alignment. Additionally, we evaluate six objective metrics, showing that traditional metrics undervalue generative models. Our findings reveal platform-specific biases and emphasize codec-aware metrics, offering guidance for scalable perceptual testing of speech codecs.
#### Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching
 - **Authors:** Jialong Zuo, Shengpeng Ji, Minghui Fang, Mingze Li, Ziyue Jiang, Xize Cheng, Xiaoda Yang, Chen Feiyang, Xinyu Duan, Zhou Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01014

 - **Pdf link:** https://arxiv.org/pdf/2506.01014

 - **Abstract**
 Zero-Shot Voice Conversion (VC) aims to transform the source speaker's timbre into an arbitrary unseen one while retaining speech content. Most prior work focuses on preserving the source's prosody, while fine-grained timbre information may leak through prosody, and transferring target prosody to synthesized speech is rarely studied. In light of this, we propose R-VC, a rhythm-controllable and efficient zero-shot voice conversion model. R-VC employs data perturbation techniques and discretize source speech into Hubert content tokens, eliminating much content-irrelevant information. By leveraging a Mask Generative Transformer for in-context duration modeling, our model adapts the linguistic content duration to the desired target speaking style, facilitating the transfer of the target speaker's rhythm. Furthermore, R-VC introduces a powerful Diffusion Transformer (DiT) with shortcut flow matching during training, conditioning the network not only on the current noise level but also on the desired step size, enabling high timbre similarity and quality speech generation in fewer sampling steps, even in just two, thus minimizing latency. Experimental results show that R-VC achieves comparable speaker similarity to state-of-the-art VC methods with a smaller dataset, and surpasses them in terms of speech naturalness, intelligibility and style transfer performance.
#### PseudoVC: Improving One-shot Voice Conversion with Pseudo Paired Data
 - **Authors:** Songjun Cao, Qinghua Wu, Jie Chen, Jin Li, Long Ma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01039

 - **Pdf link:** https://arxiv.org/pdf/2506.01039

 - **Abstract**
 As parallel training data is scarce for one-shot voice conversion (VC) tasks, waveform reconstruction is typically performed by various VC systems. A typical one-shot VC system comprises a content encoder and a speaker encoder. However, two types of mismatches arise: one for the inputs to the content encoder during training and inference, and another for the inputs to the speaker encoder. To address these mismatches, we propose a novel VC training method called \textit{PseudoVC} in this paper. First, we introduce an innovative information perturbation approach named \textit{Pseudo Conversion} to tackle the first mismatch problem. This approach leverages pretrained VC models to convert the source utterance into a perturbed utterance, which is fed into the content encoder during training. Second, we propose an approach termed \textit{Speaker Sampling} to resolve the second mismatch problem, which will substitute the input to the speaker encoder by another utterance from the same speaker during training. Experimental results demonstrate that our proposed \textit{Pseudo Conversion} outperforms previous information perturbation methods, and the overall \textit{PseudoVC} method surpasses publicly available VC models. Audio examples are available.
#### PARROT: Synergizing Mamba and Attention-based SSL Pre-Trained Models via Parallel Branch Hadamard Optimal Transport for Speech Emotion Recognition
 - **Authors:** Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, Swarup Ranjan Behera, Jaya Sai Kiran Patibandla, Arun Balaji Buduru, Rajesh Sharma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01138

 - **Pdf link:** https://arxiv.org/pdf/2506.01138

 - **Abstract**
 The emergence of Mamba as an alternative to attention-based architectures has led to the development of Mamba-based self-supervised learning (SSL) pre-trained models (PTMs) for speech and audio processing. Recent studies suggest that these models achieve comparable or superior performance to state-of-the-art (SOTA) attention-based PTMs for speech emotion recognition (SER). Motivated by prior work demonstrating the benefits of PTM fusion across different speech processing tasks, we hypothesize that leveraging the complementary strengths of Mamba-based and attention-based PTMs will enhance SER performance beyond the fusion of homogenous attention-based PTMs. To this end, we introduce a novel framework, PARROT that integrates parallel branch fusion with Optimal Transport and Hadamard Product. Our approach achieves SOTA results against individual PTMs, homogeneous PTMs fusion, and baseline fusion techniques, thus, highlighting the potential of heterogeneous PTM fusion for SER.
#### Towards Fusion of Neural Audio Codec-based Representations with Spectral for Heart Murmur Classification via Bandit-based Cross-Attention Mechanism
 - **Authors:** Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Swarup Ranjan Behera, Priyabrata Mallick, Santanu Roy, Arun Balaji Buduru, Rajesh Sharma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01148

 - **Pdf link:** https://arxiv.org/pdf/2506.01148

 - **Abstract**
 In this study, we focus on heart murmur classification (HMC) and hypothesize that combining neural audio codec representations (NACRs) such as EnCodec with spectral features (SFs), such as MFCC, will yield superior performance. We believe such fusion will trigger their complementary behavior as NACRs excel at capturing fine-grained acoustic patterns such as rhythm changes, spectral features focus on frequency-domain properties such as harmonic structure, spectral energy distribution crucial for analyzing the complex of heart sounds. To this end, we propose, BAOMI, a novel framework banking on novel bandit-based cross-attention mechanism for effective fusion. Here, a agent provides more weightage to most important heads in multi-head cross-attention mechanism and helps in mitigating the noise. With BAOMI, we report the topmost performance in comparison to individual NACRs, SFs, and baseline fusion techniques and setting new state-of-the-art.
#### Source Tracing of Synthetic Speech Systems Through Paralinguistic Pre-Trained Representations
 - **Authors:** Girish, Mohd Mujtaba Akhtar, Orchid Chetia Phukan, Drishti Singh, Swarup Ranjan Behera, Pailla Balakrishna Reddy, Arun Balaji Buduru, Rajesh Sharma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01157

 - **Pdf link:** https://arxiv.org/pdf/2506.01157

 - **Abstract**
 In this work, we focus on source tracing of synthetic speech generation systems (STSGS). Each source embeds distinctive paralinguistic features--such as pitch, tone, rhythm, and intonation--into their synthesized speech, reflecting the underlying design of the generation model. While previous research has explored representations from speech pre-trained models (SPTMs), the use of representations from SPTM pre-trained for paralinguistic speech processing, which excel in paralinguistic tasks like synthetic speech detection, speech emotion recognition has not been investigated for STSGS. We hypothesize that representations from paralinguistic SPTM will be more effective due to its ability to capture source-specific paralinguistic cues attributing to its paralinguistic pre-training. Our comparative study of representations from various SOTA SPTMs, including paralinguistic, monolingual, multilingual, and speaker recognition, validates this hypothesis. Furthermore, we explore fusion of representations and propose TRIO, a novel framework that fuses SPTMs using a gated mechanism for adaptive weighting, followed by canonical correlation loss for inter-representation alignment and self-attention for feature refinement. By fusing TRILLsson (Paralinguistic SPTM) and x-vector (Speaker recognition SPTM), TRIO outperforms individual SPTMs, baseline fusion methods, and sets new SOTA for STSGS in comparison to previous works.
#### GigaAM: Efficient Self-Supervised Learner for Speech Recognition
 - **Authors:** Aleksandr Kutsakov, Alexandr Maximenko, Georgii Gospodinov, Pavel Bogomolov, Fyodor Minkin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01192

 - **Pdf link:** https://arxiv.org/pdf/2506.01192

 - **Abstract**
 Self-Supervised Learning (SSL) has demonstrated strong performance in speech processing, particularly in automatic speech recognition. In this paper, we explore an SSL pretraining framework that leverages masked language modeling with targets derived from a speech recognition model. We also present chunkwise attention with dynamic chunk size sampling during pretraining to enable both full-context and streaming fine-tuning. Our experiments examine scaling with respect to model size and the amount of data. Using our method, we train the GigaAM family of models, including a state-of-the-art model for Russian speech recognition that outperforms Whisper-large-v3 by 50%. We have released our foundation and ASR models, along with the inference code, under the MIT license as open-source resources to the research community. Available at this https URL.
#### Confidence intervals for forced alignment boundaries using model ensembles
 - **Authors:** Matthew C. Kelley
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01256

 - **Pdf link:** https://arxiv.org/pdf/2506.01256

 - **Abstract**
 Forced alignment is a common tool to align audio with orthographic and phonetic transcriptions. Most forced alignment tools provide only a single estimate of a boundary. The present project introduces a method of deriving confidence intervals for these boundaries using a neural network ensemble technique. Ten different segment classifier neural networks were previously trained, and the alignment process is repeated with each model. The alignment ensemble is then used to place the boundary at the median of the boundaries in the ensemble, and 97.85% confidence intervals are constructed using order statistics. On the Buckeye and TIMIT corpora, the ensemble boundaries show a slight improvement over using just a single model. The confidence intervals are incorporated into Praat TextGrids using a point tier, and they are also output as a table for researchers to analyze separately as diagnostics or to incorporate uncertainty into their analyses.
#### Online Audio-Visual Autoregressive Speaker Extraction
 - **Authors:** Zexu Pan, Wupeng Wang, Shengkui Zhao, Chong Zhang, Kun Zhou, Yukun Ma, Bin Ma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01270

 - **Pdf link:** https://arxiv.org/pdf/2506.01270

 - **Abstract**
 This paper proposes a novel online audio-visual speaker extraction model. In the streaming regime, most studies optimize the audio network only, leaving the visual frontend less explored. We first propose a lightweight visual frontend based on depth-wise separable convolution. Then, we propose a lightweight autoregressive acoustic encoder to serve as the second cue, to actively explore the information in the separated speech signal from past steps. Scenario-wise, for the first time, we study how the algorithm performs when there is a change in focus of attention, i.e., the target speaker. Experimental results on LRS3 datasets show that our visual frontend performs comparably to the previous state-of-the-art on both SkiM and ConvTasNet audio backbones with only 0.1 million network parameters and 2.1 MACs per second of processing. The autoregressive acoustic encoder provides an additional 0.9 dB gain in terms of SI-SNRi, and its momentum is robust against the change in attention.
#### Inter-Speaker Relative Cues for Text-Guided Target Speech Extraction
 - **Authors:** Wang Dai, Archontis Politis, Tuomas Virtanen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01483

 - **Pdf link:** https://arxiv.org/pdf/2506.01483

 - **Abstract**
 We propose a novel approach that utilize inter-speaker relative cues for distinguishing target speakers and extracting their voices from mixtures. Continuous cues (e.g., temporal order, age, pitch level) are grouped by relative differences, while discrete cues (e.g., language, gender, emotion) retain their categories. Relative cues offers greater flexibility than fixed speech attribute classification, facilitating much easier expansion of text-guided target speech extraction datasets. Our experiments show that combining all relative cues yields better performance than random subsets, with gender and temporal order being the most robust across languages and reverberant conditions. Additional cues like pitch level, loudness, distance, speaking duration, language, and pitch range also demonstrate notable benefit in complex scenarios. Fine-tuning pre-trained WavLM Base+ CNN encoders improves overall performance over the baseline of using only a Conv1d encoder.
#### LinearVC: Linear transformations of self-supervised features through the lens of voice conversion
 - **Authors:** Herman Kamper, Benjamin van Niekerk, Julian Zaïdi, Marc-André Carbonneau
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2506.01510

 - **Pdf link:** https://arxiv.org/pdf/2506.01510

 - **Abstract**
 We introduce LinearVC, a simple voice conversion method that sheds light on the structure of self-supervised representations. First, we show that simple linear transformations of self-supervised features effectively convert voices. Next, we probe the geometry of the feature space by constraining the set of allowed transformations. We find that just rotating the features is sufficient for high-quality voice conversion. This suggests that content information is embedded in a low-dimensional subspace which can be linearly transformed to produce a target voice. To validate this hypothesis, we finally propose a method that explicitly factorizes content and speaker information using singular value decomposition; the resulting linear projection with a rank of just 100 gives competitive conversion results. Our work has implications for both practical voice conversion and a broader understanding of self-supervised speech representations. Samples and code: this https URL.
#### Lessons Learned from the URGENT 2024 Speech Enhancement Challenge
 - **Authors:** Wangyou Zhang, Kohei Saijo, Samuele Cornell, Robin Scheibler, Chenda Li, Zhaoheng Ni, Anurag Kumar, Marvin Sach, Wei Wang, Yihui Fu, Shinji Watanabe, Tim Fingscheidt, Yanmin Qian
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.01611

 - **Pdf link:** https://arxiv.org/pdf/2506.01611

 - **Abstract**
 The URGENT 2024 Challenge aims to foster speech enhancement (SE) techniques with great universality, robustness, and generalizability, featuring a broader task definition, large-scale multi-domain data, and comprehensive evaluation metrics. Nourished by the challenge outcomes, this paper presents an in-depth analysis of two key, yet understudied, issues in SE system development: data cleaning and evaluation metrics. We highlight several overlooked problems in traditional SE pipelines: (1) mismatches between declared and effective audio bandwidths, along with label noise even in various "high-quality" speech corpora; (2) lack of both effective SE systems to conquer the hardest conditions (e.g., speech overlap, strong noise / reverberation) and reliable measure of speech sample difficulty; (3) importance of combining multifaceted metrics for a comprehensive evaluation correlating well with human judgment. We hope that this endeavor can inspire improved SE pipeline designs in the future.
#### Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech
 - **Authors:** Karl El Hajal, Enno Hermann, Sevada Hovsepyan, Mathew Magimai.-Doss
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01618

 - **Pdf link:** https://arxiv.org/pdf/2506.01618

 - **Abstract**
 Automatic speech recognition (ASR) systems struggle with dysarthric speech due to high inter-speaker variability and slow speaking rates. To address this, we explore dysarthric-to-healthy speech conversion for improved ASR performance. Our approach extends the Rhythm and Voice (RnV) conversion framework by introducing a syllable-based rhythm modeling method suited for dysarthric speech. We assess its impact on ASR by training LF-MMI models and fine-tuning Whisper on converted speech. Experiments on the Torgo corpus reveal that LF-MMI achieves significant word error rate reductions, especially for more severe cases of dysarthria, while fine-tuning Whisper on converted data has minimal effect on its performance. These results highlight the potential of unsupervised rhythm and voice conversion for dysarthric ASR. Code available at: this https URL
#### Self-Supervised Speech Quality Assessment (S3QA): Leveraging Speech Foundation Models for a Scalable Speech Quality Metric
 - **Authors:** Mattson Ogg, Caitlyn Bishop, Han Yi, Sarah Robinson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01655

 - **Pdf link:** https://arxiv.org/pdf/2506.01655

 - **Abstract**
 Methods for automatically assessing speech quality are critical for many human language technologies. Behavioral ratings provided by human raters (e.g., mean opinion scores; MOS) are considered the gold standard, but they are susceptible to variability between individual raters, cannot easily be generalized across corpora, and are labor-intensive to collect, thus limiting the acoustic challenges they can quantify. Here, we present a new, scalable method for automatically assessing speech quality: the self-supervised speech quality assessment (S3QA) model. First, we processed high quality utterances from multiple speech corpora, using a wide range of acoustic manipulations intended to emulate common sources of quality degradation in the real-world: frequency filtering, reverberation, background noise, and digital compression. Second, we leveraged an existing, pre-trained speech foundation model, WavLM, to computationally derive a self-supervised training target for the level of signal degradation by calculating the cosine distances between the clean and degraded versions of each utterance in the embedding space. Next, we trained a transformer-based model to predict the cosine distance, or degradation index, given only the degraded versions of these utterances. Finally, the trained model was evaluated on unseen test corpora of synthetic mixtures, NISQA, and VOiCES. We show that the S3QA model trained on this task performs well and is aligned with both behavioral ratings (MOS), speech technology performance (automatic speech recognition) and other important features of the held-out data (e.g., microphone distances). This approach provides an automated, scalable method for assessing speech quality across a wide range of acoustic challenges, and could easily be adapted to other use cases where acoustic simulations are available.
#### Benchmarking Neural Speech Codec Intelligibility with SITool
 - **Authors:** Anna Leschanowsky, Kishor Kayyar Lakshminarayana, Anjana Rajasekhar, Lyonel Behringer, Ibrahim Kilinc, Guillaume Fuchs, Emanuël A. P. Habets
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01731

 - **Pdf link:** https://arxiv.org/pdf/2506.01731

 - **Abstract**
 Speech intelligibility assessment is essential for evaluating neural speech codecs, yet most evaluation efforts focus on overall quality rather than intelligibility. Only a few publicly available tools exist for conducting standardized intelligibility tests, like the Diagnostic Rhyme Test (DRT) and Modified Rhyme Test (MRT). We introduce the Speech Intelligibility Toolkit for Subjective Evaluation (SITool), a Flask-based web application for conducting DRT and MRT in laboratory and crowdsourcing settings. We use SITool to benchmark 13 neural and traditional speech codecs, analyzing phoneme-level degradations and comparing subjective DRT results with objective intelligibility metrics. Our findings show that, while neural speech codecs can outperform traditional ones in subjective intelligibility, only STOI and ESTOI - not WER - significantly correlate with subjective results, although they struggle to capture gender and wordlist-specific variations observed in subjective evaluations.
#### On-device Streaming Discrete Speech Units
 - **Authors:** Kwanghee Choi, Masao Someki, Emma Strubell, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.01845

 - **Pdf link:** https://arxiv.org/pdf/2506.01845

 - **Abstract**
 Discrete speech units (DSUs) are derived from clustering the features of self-supervised speech models (S3Ms). DSUs offer significant advantages for on-device streaming speech applications due to their rich phonetic information, high transmission efficiency, and seamless integration with large language models. However, conventional DSU-based approaches are impractical as they require full-length speech input and computationally expensive S3Ms. In this work, we reduce both the attention window and the model size while preserving the effectiveness of DSUs. Our results demonstrate that we can reduce floating-point operations (FLOPs) by 50% with only a relative increase of 6.5% in character error rate (CER) on the ML-SUPERB 1h dataset. These findings highlight the potential of DSUs for real-time speech processing in resource-constrained environments.
#### DNCASR: End-to-End Training for Speaker-Attributed ASR
 - **Authors:** Xianrui Zheng, Chao Zhang, Philip C. Woodland
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01916

 - **Pdf link:** https://arxiv.org/pdf/2506.01916

 - **Abstract**
 This paper introduces DNCASR, a novel end-to-end trainable system designed for joint neural speaker clustering and automatic speech recognition (ASR), enabling speaker-attributed transcription of long multi-party meetings. DNCASR uses two separate encoders to independently encode global speaker characteristics and local waveform information, along with two linked decoders to generate speaker-attributed transcriptions. The use of linked decoders allows the entire system to be jointly trained under a unified loss function. By employing a serialised training approach, DNCASR effectively addresses overlapping speech in real-world meetings, where the link improves the prediction of speaker indices in overlapping segments. Experiments on the AMI-MDM meeting corpus demonstrate that the jointly trained DNCASR outperforms a parallel system that does not have links between the speaker and ASR decoders. Using cpWER to measure the speaker-attributed word error rate, DNCASR achieves a 9.0% relative reduction on the AMI-MDM Eval set.
#### Probing Audio-Generation Capabilities of Text-Based Language Models
 - **Authors:** Arjun Prasaath Anbazhagan, Parteek Kumar, Ujjwal Kaur, Aslihan Akalin, Kevin Zhu, Sean O'Brien
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00003

 - **Pdf link:** https://arxiv.org/pdf/2506.00003

 - **Abstract**
 How does textual representation of audio relate to the Large Language Model's (LLMs) learning about the audio world? This research investigates the extent to which LLMs can be prompted to generate audio, despite their primary training in textual data. We employ a three-tier approach, progressively increasing the complexity of audio generation: 1) Musical Notes, 2) Environmental Sounds, and 3) Human Speech. To bridge the gap between text and audio, we leverage code as an intermediary, prompting LLMs to generate code that, when executed, produces the desired audio output. To evaluate the quality and accuracy of the generated audio, we employ FAD and CLAP scores. Our findings reveal that while LLMs can generate basic audio features, their performance deteriorates as the complexity of the audio increases. This suggests that while LLMs possess a latent understanding of the auditory world, their ability to translate this understanding into tangible audio output remains rudimentary. Further research into techniques that can enhance the quality and diversity of LLM-generated audio can lead to an improvement in the performance of text-based LLMs in generating audio.
#### ACE-Step: A Step Towards Music Generation Foundation Model
 - **Authors:** Junmin Gong, Sean Zhao, Sen Wang, Shengyuan Xu, Joe Guo
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00045

 - **Pdf link:** https://arxiv.org/pdf/2506.00045

 - **Abstract**
 We introduce ACE-Step, a novel open-source foundation model for music generation that overcomes key limitations of existing approaches and achieves state-of-the-art performance through a holistic architectural design. Current methods face inherent trade-offs between generation speed, musical coherence, and controllability. For example, LLM-based models (e.g. Yue, SongGen) excel at lyric alignment but suffer from slow inference and structural artifacts. Diffusion models (e.g. DiffRhythm), on the other hand, enable faster synthesis but often lack long-range structural coherence. ACE-Step bridges this gap by integrating diffusion-based generation with Sana's Deep Compression AutoEncoder (DCAE) and a lightweight linear transformer. It also leverages MERT and m-hubert to align semantic representations (REPA) during training, allowing rapid convergence. As a result, our model synthesizes up to 4 minutes of music in just 20 seconds on an A100 GPU-15x faster than LLM-based baselines-while achieving superior musical coherence and lyric alignment across melody, harmony, and rhythm metrics. Moreover, ACE-Step preserves fine-grained acoustic details, enabling advanced control mechanisms such as voice cloning, lyric editing, remixing, and track generation (e.g. lyric2vocal, singing2accompaniment). Rather than building yet another end-to-end text-to-music pipeline, our vision is to establish a foundation model for music AI: a fast, general-purpose, efficient yet flexible architecture that makes it easy to train subtasks on top of it. This paves the way for the development of powerful tools that seamlessly integrate into the creative workflows of music artists, producers, and content creators. In short, our goal is to build a stable diffusion moment for music. The code, the model weights and the demo are available at: this https URL.
#### Vedavani: A Benchmark Corpus for ASR on Vedic Sanskrit Poetry
 - **Authors:** Sujeet Kumar, Pretam Ray, Abhinay Beerukuri, Shrey Kamoji, Manoj Balaji Jagadeeshan, Pawan Goyal
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00145

 - **Pdf link:** https://arxiv.org/pdf/2506.00145

 - **Abstract**
 Sanskrit, an ancient language with a rich linguistic heritage, presents unique challenges for automatic speech recognition (ASR) due to its phonemic complexity and the phonetic transformations that occur at word junctures, similar to the connected speech found in natural conversations. Due to these complexities, there has been limited exploration of ASR in Sanskrit, particularly in the context of its poetic verses, which are characterized by intricate prosodic and rhythmic patterns. This gap in research raises the question: How can we develop an effective ASR system for Sanskrit, particularly one that captures the nuanced features of its poetic form? In this study, we introduce Vedavani, the first comprehensive ASR study focused on Sanskrit Vedic poetry. We present a 54-hour Sanskrit ASR dataset, consisting of 30,779 labelled audio samples from the Rig Veda and Atharva Veda. This dataset captures the precise prosodic and rhythmic features that define the language. We also benchmark the dataset on various state-of-the-art multilingual speech models.$^{1}$ Experimentation revealed that IndicWhisper performed the best among the SOTA models.
#### OWSM v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning
 - **Authors:** Yifan Peng, Shakeel Muhammad, Yui Sudo, William Chen, Jinchuan Tian, Chyi-Jiunn Lin, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00338

 - **Pdf link:** https://arxiv.org/pdf/2506.00338

 - **Abstract**
 The Open Whisper-style Speech Models (OWSM) project has developed a series of fully open speech foundation models using academic-scale resources, but their training data remains insufficient. This work enhances OWSM by integrating YODAS, a large-scale web-crawled dataset with a Creative Commons license. However, incorporating YODAS is nontrivial due to its wild nature, which introduces challenges such as incorrect language labels and audio-text misalignments. To address this, we develop a scalable data-cleaning pipeline using public toolkits, yielding a dataset with 166,000 hours of speech across 75 languages. Our new series of OWSM v4 models, trained on this curated dataset alongside existing OWSM data, significantly outperform previous versions on multilingual benchmarks. Our models even match or surpass frontier industrial models like Whisper and MMS in multiple scenarios. We will publicly release the cleaned YODAS data, pre-trained models, and all associated scripts via the ESPnet toolkit.
#### DiffDSR: Dysarthric Speech Reconstruction Using Latent Diffusion Model
 - **Authors:** Xueyuan Chen, Dongchao Yang, Wenxuan Wu, Minglin Wu, Jing Xu, Xixin Wu, Zhiyong Wu, Helen Meng
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00350

 - **Pdf link:** https://arxiv.org/pdf/2506.00350

 - **Abstract**
 Dysarthric speech reconstruction (DSR) aims to convert dysarthric speech into comprehensible speech while maintaining the speaker's identity. Despite significant advancements, existing methods often struggle with low speech intelligibility and poor speaker similarity. In this study, we introduce a novel diffusion-based DSR system that leverages a latent diffusion model to enhance the quality of speech reconstruction. Our model comprises: (i) a speech content encoder for phoneme embedding restoration via pre-trained self-supervised learning (SSL) speech foundation models; (ii) a speaker identity encoder for speaker-aware identity preservation by in-context learning mechanism; (iii) a diffusion-based speech generator to reconstruct the speech based on the restored phoneme embedding and preserved speaker identity. Through evaluations on the widely-used UASpeech corpus, our proposed model shows notable enhancements in speech intelligibility and speaker similarity.
#### RPRA-ADD: Forgery Trace Enhancement-Driven Audio Deepfake Detection
 - **Authors:** Ruibo Fu, Xiaopeng Wang, Zhengqi Wen, Jianhua Tao, Yuankun Xie, Zhiyong Wang, Chunyu Qiang, Xuefei Liu, Cunhang Fan, Chenxing Li, Guanjun Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00375

 - **Pdf link:** https://arxiv.org/pdf/2506.00375

 - **Abstract**
 Existing methods for deepfake audio detection have demonstrated some effectiveness. However, they still face challenges in generalizing to new forgery techniques and evolving attack patterns. This limitation mainly arises because the models rely heavily on the distribution of the training data and fail to learn a decision boundary that captures the essential characteristics of forgeries. Additionally, relying solely on a classification loss makes it difficult to capture the intrinsic differences between real and fake audio. In this paper, we propose the RPRA-ADD, an integrated Reconstruction-Perception-Reinforcement-Attention networks based forgery trace enhancement-driven robust audio deepfake detection framework. First, we propose a Global-Local Forgery Perception (GLFP) module for enhancing the acoustic perception capacity of forgery traces. To significantly reinforce the feature space distribution differences between real and fake audio, the Multi-stage Dispersed Enhancement Loss (MDEL) is designed, which implements a dispersal strategy in multi-stage feature spaces. Furthermore, in order to enhance feature awareness towards forgery traces, the Fake Trace Focused Attention (FTFA) mechanism is introduced to adjust attention weights dynamically according to the reconstruction discrepancy matrix. Visualization experiments not only demonstrate that FTFA improves attention to voice segments, but also enhance the generalization capability. Experimental results demonstrate that the proposed method achieves state-of-the-art performance on 4 benchmark datasets, including ASVspoof2019, ASVspoof2021, CodecFake, and FakeSound, achieving over 20% performance improvement. In addition, it outperforms existing methods in rigorous 3*3 cross-domain evaluations across Speech, Sound, and Singing, demonstrating strong generalization capability across diverse audio domains.
#### Neuro2Semantic: A Transfer Learning Framework for Semantic Reconstruction of Continuous Language from Human Intracranial EEG
 - **Authors:** Siavash Shams, Richard Antonello, Gavin Mischler, Stephan Bickel, Ashesh Mehta, Nima Mesgarani
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.00381

 - **Pdf link:** https://arxiv.org/pdf/2506.00381

 - **Abstract**
 Decoding continuous language from neural signals remains a significant challenge in the intersection of neuroscience and artificial intelligence. We introduce Neuro2Semantic, a novel framework that reconstructs the semantic content of perceived speech from intracranial EEG (iEEG) recordings. Our approach consists of two phases: first, an LSTM-based adapter aligns neural signals with pre-trained text embeddings; second, a corrector module generates continuous, natural text directly from these aligned embeddings. This flexible method overcomes the limitations of previous decoding approaches and enables unconstrained text generation. Neuro2Semantic achieves strong performance with as little as 30 minutes of neural data, outperforming a recent state-of-the-art method in low-data settings. These results highlight the potential for practical applications in brain-computer interfaces and neural decoding technologies.
#### MagiCodec: Simple Masked Gaussian-Injected Codec for High-Fidelity Reconstruction and Generation
 - **Authors:** Yakun Song, Jiawei Chen, Xiaobin Zhuang, Chenpeng Du, Ziyang Ma, Jian Wu, Jian Cong, Dongya Jia, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xie Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00385

 - **Pdf link:** https://arxiv.org/pdf/2506.00385

 - **Abstract**
 Neural audio codecs have made significant strides in efficiently mapping raw audio waveforms into discrete token representations, which are foundational for contemporary audio generative models. However, most existing codecs are optimized primarily for reconstruction quality, often at the expense of the downstream modelability of the encoded tokens. Motivated by the need to overcome this bottleneck, we introduce $\textbf{MagiCodec}$, a novel single-layer, streaming Transformer-based audio codec. MagiCodec is designed with a multistage training pipeline that incorporates Gaussian noise injection and latent regularization, explicitly targeting the enhancement of semantic expressiveness in the generated codes while preserving high reconstruction fidelity. We analytically derive the effect of noise injection in the frequency domain, demonstrating its efficacy in attenuating high-frequency components and fostering robust tokenization. Extensive experimental evaluations show that MagiCodec surpasses state-of-the-art codecs in both reconstruction quality and downstream tasks. Notably, the tokens produced by MagiCodec exhibit Zipf-like distributions, as observed in natural languages, thereby improving compatibility with language-model-based generative architectures. The code and pre-trained models are available at this https URL.
#### Causal Structure Discovery for Error Diagnostics of Children's ASR
 - **Authors:** Vishwanath Pratap Singh, Md. Sahidullah, Tomi Kinnunen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00402

 - **Pdf link:** https://arxiv.org/pdf/2506.00402

 - **Abstract**
 Children's automatic speech recognition (ASR) often underperforms compared to that of adults due to a confluence of interdependent factors: physiological (e.g., smaller vocal tracts), cognitive (e.g., underdeveloped pronunciation), and extrinsic (e.g., vocabulary limitations, background noise). Existing analysis methods examine the impact of these factors in isolation, neglecting interdependencies-such as age affecting ASR accuracy both directly and indirectly via pronunciation skills. In this paper, we introduce a causal structure discovery to unravel these interdependent relationships among physiology, cognition, extrinsic factors, and ASR errors. Then, we employ causal quantification to measure each factor's impact on children's ASR. We extend the analysis to fine-tuned models to identify which factors are mitigated by fine-tuning and which remain largely unaffected. Experiments on Whisper and Wav2Vec2.0 demonstrate the generalizability of our findings across different ASR systems.
#### XMAD-Bench: Cross-Domain Multilingual Audio Deepfake Benchmark
 - **Authors:** Ioan-Paul Ciobanu, Andrei-Iulian Hiji, Nicolae-Catalin Ristea, Paul Irofti, Cristian Rusu, Radu Tudor Ionescu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00462

 - **Pdf link:** https://arxiv.org/pdf/2506.00462

 - **Abstract**
 Recent advances in audio generation led to an increasing number of deepfakes, making the general public more vulnerable to financial scams, identity theft, and misinformation. Audio deepfake detectors promise to alleviate this issue, with many recent studies reporting accuracy rates close to 99%. However, these methods are typically tested in an in-domain setup, where the deepfake samples from the training and test sets are produced by the same generative models. To this end, we introduce XMAD-Bench, a large-scale cross-domain multilingual audio deepfake benchmark comprising 668.8 hours of real and deepfake speech. In our novel dataset, the speakers, the generative methods, and the real audio sources are distinct across training and test splits. This leads to a challenging cross-domain evaluation setup, where audio deepfake detectors can be tested ``in the wild''. Our in-domain and cross-domain experiments indicate a clear disparity between the in-domain performance of deepfake detectors, which is usually as high as 100%, and the cross-domain performance of the same models, which is sometimes similar to random chance. Our benchmark highlights the need for the development of robust audio deepfake detectors, which maintain their generalization capacity across different languages, speakers, generative methods, and data sources. Our benchmark is publicly released at this https URL.
#### Learning to Upsample and Upmix Audio in the Latent Domain
 - **Authors:** Dimitrios Bralios, Paris Smaragdis, Jonah Casebeer
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00681

 - **Pdf link:** https://arxiv.org/pdf/2506.00681

 - **Abstract**
 Neural audio autoencoders create compact latent representations that preserve perceptually important information, serving as the foundation for both modern audio compression systems and generation approaches like next-token prediction and latent diffusion. Despite their prevalence, most audio processing operations, such as spatial and spectral up-sampling, still inefficiently operate on raw waveforms or spectral representations rather than directly on these compressed representations. We propose a framework that performs audio processing operations entirely within an autoencoder's latent space, eliminating the need to decode to raw audio formats. Our approach dramatically simplifies training by operating solely in the latent domain, with a latent L1 reconstruction term, augmented by a single latent adversarial discriminator. This contrasts sharply with raw-audio methods that typically require complex combinations of multi-scale losses and discriminators. Through experiments in bandwidth extension and mono-to-stereo up-mixing, we demonstrate computational efficiency gains of up to 100x while maintaining quality comparable to post-processing on raw audio. This work establishes a more efficient paradigm for audio processing pipelines that already incorporate autoencoders, enabling significantly faster and more resource-efficient workflows across various audio tasks.
#### Chain-of-Thought Training for Open E2E Spoken Dialogue Systems
 - **Authors:** Siddhant Arora, Jinchuan Tian, Hayato Futami, Jee-weon Jung, Jiatong Shi, Yosuke Kashiwagi, Emiru Tsunoo, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00722

 - **Pdf link:** https://arxiv.org/pdf/2506.00722

 - **Abstract**
 Unlike traditional cascaded pipelines, end-to-end (E2E) spoken dialogue systems preserve full differentiability and capture non-phonemic information, making them well-suited for modeling spoken interactions. However, existing E2E approaches often require large-scale training data and generates responses lacking semantic coherence. We propose a simple yet effective strategy leveraging a chain-of-thought (CoT) formulation, ensuring that training on conversational data remains closely aligned with the multimodal language model (LM)'s pre-training on speech recognition~(ASR), text-to-speech synthesis (TTS), and text LM tasks. Our method achieves over 1.5 ROUGE-1 improvement over the baseline, successfully training spoken dialogue systems on publicly available human-human conversation datasets, while being compute-efficient enough to train on just 300 hours of public human-human conversation data, such as the Switchboard. We will publicly release our models and training code.
#### Length Aware Speech Translation for Video Dubbing
 - **Authors:** Harveen Singh Chadha, Aswin Shanmugam Subramanian, Vikas Joshi, Shubham Bansal, Jian Xue, Rupeshkumar Mehta, Jinyu Li
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00740

 - **Pdf link:** https://arxiv.org/pdf/2506.00740

 - **Abstract**
 In video dubbing, aligning translated audio with the source audio is a significant challenge. Our focus is on achieving this efficiently, tailored for real-time, on-device video dubbing scenarios. We developed a phoneme-based end-to-end length-sensitive speech translation (LSST) model, which generates translations of varying lengths short, normal, and long using predefined tags. Additionally, we introduced length-aware beam search (LABS), an efficient approach to generate translations of different lengths in a single decoding pass. This approach maintained comparable BLEU scores compared to a baseline without length awareness while significantly enhancing synchronization quality between source and target audio, achieving a mean opinion score (MOS) gain of 0.34 for Spanish and 0.65 for Korean, respectively.
#### FUSE: Universal Speech Enhancement using Multi-Stage Fusion of Sparse Compression and Token Generation Models for the URGENT 2025 Challenge
 - **Authors:** Nabarun Goswami, Tatsuya Harada
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00809

 - **Pdf link:** https://arxiv.org/pdf/2506.00809

 - **Abstract**
 We propose a multi-stage framework for universal speech enhancement, designed for the Interspeech 2025 URGENT Challenge. Our system first employs a Sparse Compression Network to robustly separate sources and extract an initial clean speech estimate from noisy inputs. This is followed by an efficient generative model that refines speech quality by leveraging self-supervised features and optimizing a masked language modeling objective on acoustic tokens derived from a neural audio codec. In the final stage, a fusion network integrates the outputs of the first two stages with the original noisy signal, achieving a balanced improvement in both signal fidelity and perceptual quality. Additionally, a shift trick that aggregates multiple time-shifted predictions, along with output blending, further boosts performance. Experimental results on challenging multilingual datasets with variable sampling rates and diverse distortion types validate the effectiveness of our approach.
#### Counterfactual Activation Editing for Post-hoc Prosody and Mispronunciation Correction in TTS Models
 - **Authors:** Kyowoon Lee, Artyom Stitsyuk, Gunu Jho, Inchul Hwang, Jaesik Choi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00832

 - **Pdf link:** https://arxiv.org/pdf/2506.00832

 - **Abstract**
 Recent advances in Text-to-Speech (TTS) have significantly improved speech naturalness, increasing the demand for precise prosody control and mispronunciation correction. Existing approaches for prosody manipulation often depend on specialized modules or additional training, limiting their capacity for post-hoc adjustments. Similarly, traditional mispronunciation correction relies on grapheme-to-phoneme dictionaries, making it less practical in low-resource settings. We introduce Counterfactual Activation Editing, a model-agnostic method that manipulates internal representations in a pre-trained TTS model to achieve post-hoc control of prosody and pronunciation. Experimental results show that our method effectively adjusts prosodic features and corrects mispronunciations while preserving synthesis quality. This opens the door to inference-time refinement of TTS outputs without retraining, bridging the gap between pre-trained TTS models and editable speech synthesis.
#### Speech Unlearning
 - **Authors:** Jiali Cheng, Hadi Amiri
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00848

 - **Pdf link:** https://arxiv.org/pdf/2506.00848

 - **Abstract**
 We introduce machine unlearning for speech tasks, a novel and underexplored research problem that aims to efficiently and effectively remove the influence of specific data from trained speech models without full retraining. This has important applications in privacy preservation, removal of outdated or noisy data, and bias mitigation. While machine unlearning has been studied in computer vision and natural language processing, its application to speech is largely unexplored due to the high-dimensional, sequential, and speaker-dependent nature of speech data. We define two fundamental speech unlearning tasks: sample unlearning, which removes individual data points (e.g., a voice recording), and class unlearning, which removes an entire category (e.g., all data from a speaker), while preserving performance on the remaining data. Experiments on keyword spotting and speaker identification demonstrate that unlearning speech data is significantly more challenging than unlearning image or text data. We conclude with key future directions in this area, including structured training, robust evaluation, feature-level unlearning, broader applications, scalable methods, and adversarial robustness.
#### Fine-Tuning ASR for Stuttered Speech: Personalized vs. Generalized Approaches
 - **Authors:** Dena Mujtaba, Nihar Mahapatra
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00853

 - **Pdf link:** https://arxiv.org/pdf/2506.00853

 - **Abstract**
 Stuttering -- characterized by involuntary disfluencies such as blocks, prolongations, and repetitions -- is often misinterpreted by automatic speech recognition (ASR) systems, resulting in elevated word error rates and making voice-driven technologies inaccessible to people who stutter. The variability of disfluencies across speakers and contexts further complicates ASR training, compounded by limited annotated stuttered speech data. In this paper, we investigate fine-tuning ASRs for stuttered speech, comparing generalized models (trained across multiple speakers) to personalized models tailored to individual speech characteristics. Using a diverse range of voice-AI scenarios, including virtual assistants and video interviews, we evaluate how personalization affects transcription accuracy. Our findings show that personalized ASRs significantly reduce word error rates, especially in spontaneous speech, highlighting the potential of tailored models for more inclusive voice technologies.
#### CoVoMix2: Advancing Zero-Shot Dialogue Generation with Fully Non-Autoregressive Flow Matching
 - **Authors:** Leying Zhang, Yao Qian, Xiaofei Wang, Manthan Thakker, Dongmei Wang, Jianwei Yu, Haibin Wu, Yuxuan Hu, Jinyu Li, Yanmin Qian, Sheng Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00885

 - **Pdf link:** https://arxiv.org/pdf/2506.00885

 - **Abstract**
 Generating natural-sounding, multi-speaker dialogue is crucial for applications such as podcast creation, virtual agents, and multimedia content generation. However, existing systems struggle to maintain speaker consistency, model overlapping speech, and synthesize coherent conversations efficiently. In this paper, we introduce CoVoMix2, a fully non-autoregressive framework for zero-shot multi-talker dialogue generation. CoVoMix2 directly predicts mel-spectrograms from multi-stream transcriptions using a flow-matching-based generative model, eliminating the reliance on intermediate token representations. To better capture realistic conversational dynamics, we propose transcription-level speaker disentanglement, sentence-level alignment, and prompt-level random masking strategies. Our approach achieves state-of-the-art performance, outperforming strong baselines like MoonCast and Sesame in speech quality, speaker consistency, and inference speed. Notably, CoVoMix2 operates without requiring transcriptions for the prompt and supports controllable dialogue generation, including overlapping speech and precise timing control, demonstrating strong generalizability to real-world speech generation scenarios.
#### In-the-wild Audio Spatialization with Flexible Text-guided Localization
 - **Authors:** Tianrui Pan, Jie Liu, Zewen Huang, Jie Tang, Gangshan Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00927

 - **Pdf link:** https://arxiv.org/pdf/2506.00927

 - **Abstract**
 To enhance immersive experiences, binaural audio offers spatial awareness of sounding objects in AR, VR, and embodied AI applications. While existing audio spatialization methods can generally map any available monaural audio to binaural audio signals, they often lack the flexible and interactive control needed in complex multi-object user-interactive environments. To address this, we propose a Text-guided Audio Spatialization (TAS) framework that utilizes flexible text prompts and evaluates our model from unified generation and comprehension perspectives. Due to the limited availability of premium and large-scale stereo data, we construct the SpatialTAS dataset, which encompasses 376,000 simulated binaural audio samples to facilitate the training of our model. Our model learns binaural differences guided by 3D spatial location and relative position prompts, augmented by flipped-channel audio. It outperforms existing methods on both simulated and real-recorded datasets, demonstrating superior generalization and accuracy. Besides, we develop an assessment model based on Llama-3.1-8B, which evaluates the spatial semantic coherence between our generated binaural audio and text prompts through a spatial reasoning task. Results demonstrate that text prompts provide flexible and interactive control to generate binaural audio with excellent quality and semantic consistency in spatial locations. Dataset is available at \href{this https URL}
#### General-purpose audio representation learning for real-world sound scenes
 - **Authors:** Goksenin Yuksel, Marcel van Gerven, Kiki van der Heijden
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00934

 - **Pdf link:** https://arxiv.org/pdf/2506.00934

 - **Abstract**
 While audio foundation models perform well on myriad of tasks from sound classification to speech analysis, these models are trained and tested on dry, non-spatial, single-source audio clips. This limits their success in real-world situations and results in spatially unaware audio embeddings. To address these limitations, we propose a novel self-supervised training approach for General-Purpose, Real-world Audio Models (GRAMs). The GRAM training approach enables robust spatial audio representation learning for naturalistic, noisy sound scenes and can be applied to any masking-based deep learning model. We demonstrate the success of our approach by training two state-of-the-art models, one with a transformer and one with a mamba backbone. We assess the quality of the extracted audio representations from GRAMs using the original version of the HEAR benchmark, a newly synthesized, naturalistic version of the HEAR benchmark, and novel sound localization tasks based on HEAR benchmark datasets. The results show that our approach minimizes the performance gap between dry, non-spatial, single-source sound scenes and naturalistic sound scenes for crucial tasks such as auditory scene analysis, outperforming existing state-of-the-art audio foundation models at a fraction of the training steps. Moreover, GRAMs show state-of-the-art performance on sound localization tasks, exceeding even supervised sound localization models. In sum, the proposed approach represents a significant advancement towards robust audio foundation models for real-world applications with state-of-the-art performance on naturalistic sound scenes as well as spatial audio representation learning.
#### NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
 - **Authors:** Qichao Wang, Ziqiao Meng, Wenqian Cui, Yifei Zhang, Pengcheng Wu, Bingzhe Wu, Irwin King, Liang Chen, Peilin Zhao
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00975

 - **Pdf link:** https://arxiv.org/pdf/2506.00975

 - **Abstract**
 Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications.
#### What do self-supervised speech models know about Dutch? Analyzing advantages of language-specific pre-training
 - **Authors:** Marianne de Heer Kloots, Hosein Mohebbi, Charlotte Pouw, Gaofei Shen, Willem Zuidema, Martijn Bentum
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.00981

 - **Pdf link:** https://arxiv.org/pdf/2506.00981

 - **Abstract**
 How language-specific are speech representations learned by self-supervised models? Existing work has shown that a range of linguistic features can be successfully decoded from end-to-end models trained only on speech recordings. However, it's less clear to what extent pre-training on specific languages improves language-specific linguistic information. Here we test the encoding of Dutch phonetic and lexical information in internal representations of self-supervised Wav2Vec2 models. Pre-training exclusively on Dutch improves the representation of Dutch linguistic features as compared to pre-training on similar amounts of English or larger amounts of multilingual data. This language-specific advantage is well-detected by trained clustering or classification probes, and partially observable using zero-shot metrics. Furthermore, the language-specific benefit on linguistic feature encoding aligns with downstream performance on Automatic Speech Recognition.
#### DS-TTS: Zero-Shot Speaker Style Adaptation from Voice Clips via Dynamic Dual-Style Feature Modulation
 - **Authors:** Ming Meng, Ziyi Yang, Jian Yang, Zhenjie Su, Yonggui Zhu, Zhaoxin Fan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01020

 - **Pdf link:** https://arxiv.org/pdf/2506.01020

 - **Abstract**
 Recent advancements in text-to-speech (TTS) technology have increased demand for personalized audio synthesis. Zero-shot voice cloning, a specialized TTS task, aims to synthesize a target speaker's voice using only a single audio sample and arbitrary text, without prior exposure to the speaker during training. This process employs pattern recognition techniques to analyze and replicate the speaker's unique vocal features. Despite progress, challenges remain in adapting to the vocal style of unseen speakers, highlighting difficulties in generalizing TTS systems to handle diverse voices while maintaining naturalness, expressiveness, and speaker fidelity. To address the challenges of unseen speaker style adaptation, we propose DS-TTS, a novel approach aimed at enhancing the synthesis of diverse, previously unheard voices. Central to our method is a Dual-Style Encoding Network (DuSEN), where two distinct style encoders capture complementary aspects of a speaker's vocal identity. These speaker-specific style vectors are seamlessly integrated into the Dynamic Generator Network (DyGN) via a Style Gating-Film (SGF) mechanism, enabling more accurate and expressive reproduction of unseen speakers' unique vocal characteristics. In addition, we introduce a Dynamic Generator Network to tackle synthesis issues that arise with varying sentence lengths. By dynamically adapting to the length of the input, this component ensures robust performance across diverse text inputs and speaker styles, significantly improving the model's ability to generalize to unseen speakers in a more natural and expressive manner. Experimental evaluations on the VCTK dataset suggest that DS-TTS demonstrates superior overall performance in voice cloning tasks compared to existing state-of-the-art models, showing notable improvements in both word error rate and speaker similarity.
#### A Two-Stage Hierarchical Deep Filtering Framework for Real-Time Speech Enhancement
 - **Authors:** Shenghui Lu, Hukai Huang, Jinanglong Yao, Kaidi Wang, Qingyang Hong, Lin Li
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01023

 - **Pdf link:** https://arxiv.org/pdf/2506.01023

 - **Abstract**
 This paper proposes a model that integrates sub-band processing and deep filtering to fully exploit information from the target time-frequency (TF) bin and its surrounding TF bins for single-channel speech enhancement. The sub-band module captures surrounding frequency bin information at the input, while the deep filtering module applies filtering at the output to both the target TF bin and its surrounding TF bins. To further improve the model performance, we decouple deep filtering into temporal and frequency components and introduce a two-stage framework, reducing the complexity of filter coefficient prediction at each stage. Additionally, we propose the TAConv module to strengthen convolutional feature extraction. Experimental results demonstrate that the proposed hierarchical deep filtering network (HDF-Net) effectively utilizes surrounding TF bin information and outperforms other advanced systems while using fewer resources.
#### ReFlow-VC: Zero-shot Voice Conversion Based on Rectified Flow and Speaker Feature Optimization
 - **Authors:** Pengyu Ren, Wenhao Guan, Kaidi Wang, Peijie Chen, Qingyang Hong, Lin Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01032

 - **Pdf link:** https://arxiv.org/pdf/2506.01032

 - **Abstract**
 In recent years, diffusion-based generative models have demonstrated remarkable performance in speech conversion, including Denoising Diffusion Probabilistic Models (DDPM) and others. However, the advantages of these models come at the cost of requiring a large number of sampling steps. This limitation hinders their practical application in real-world scenarios. In this paper, we introduce ReFlow-VC, a novel high-fidelity speech conversion method based on rectified flow. Specifically, ReFlow-VC is an Ordinary Differential Equation (ODE) model that transforms a Gaussian distribution to the true Mel-spectrogram distribution along the most direct path. Furthermore, we propose a modeling approach that optimizes speaker features by utilizing both content and pitch information, allowing speaker features to reflect the properties of the current speech more accurately. Experimental results show that ReFlow-VC performs exceptionally well in small datasets and zero-shot scenarios.
#### FusionAudio-1.2M: Towards Fine-grained Audio Captioning with Multimodal Contextual Fusion
 - **Authors:** Shunian Chen, Xinyuan Xie, Zheshu Chen, Liyan Zhao, Owen Lee, Zhan Su, Qilin Sun, Benyou Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01111

 - **Pdf link:** https://arxiv.org/pdf/2506.01111

 - **Abstract**
 High-quality, large-scale audio captioning is crucial for advancing audio understanding, yet current automated methods often generate captions that lack fine-grained detail and contextual accuracy, primarily due to their reliance on limited unimodal or superficial multimodal information. Drawing inspiration from human auditory perception, which adeptly integrates cross-modal cues and performs sophisticated auditory scene analysis, we introduce a novel two-stage automated pipeline. This pipeline first employs specialized pretrained models to extract diverse contextual cues (e.g., speech, music, general sounds, and visual information from associated video). A large language model (LLM) then synthesizes these rich, multimodal inputs to generate detailed and context-aware audio captions. Key contributions of this work include: (1) the proposed scalable method for fine-grained audio caption generation; (2) FusionAudio, a new large-scale dataset comprising 1.2 million such detailed captions, combined with 6 million QA pairs; and (3) enhanced audio models developed using FusionAudio, specifically a CLAP-based audio encoder with superior audio-text alignment and instruction following. This paper paves the way for more nuanced and accurate automated understanding of complex audio environments. Code and data can be found in this https URL.
#### Comparative Evaluation of Acoustic Feature Extraction Tools for Clinical Speech Analysis
 - **Authors:** Anna Seo Gyeong Choi, Alexander Richardson, Ryan Partlan, Sunny Tang, Sunghye Cho
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01129

 - **Pdf link:** https://arxiv.org/pdf/2506.01129

 - **Abstract**
 This study compares three acoustic feature extraction toolkits (OpenSMILE, Praat, and Librosa) applied to clinical speech data from individuals with schizophrenia spectrum disorders (SSD) and healthy controls (HC). By standardizing extraction parameters across the toolkits, we analyzed speech samples from 77 SSD and 87 HC participants and found significant toolkit-dependent variations. While F0 percentiles showed high cross-toolkit correlation (r=0.962 to 0.999), measures like F0 standard deviation and formant values often had poor, even negative, agreement. Additionally, correlation patterns differed between SSD and HC groups. Classification analysis identified F0 mean, HNR, and MFCC1 (AUC greater than 0.70) as promising discriminators. These findings underscore reproducibility concerns and advocate for standardized protocols, multi-toolkit cross-validation, and transparent reporting.
#### From Words to Waves: Analyzing Concept Formation in Speech and Text-Based Foundation Models
 - **Authors:** Asım Ersoy, Basel Mousi, Shammur Chowdhury, Firoj Alam, Fahim Dalvi, Nadir Durrani
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01133

 - **Pdf link:** https://arxiv.org/pdf/2506.01133

 - **Abstract**
 The emergence of large language models (LLMs) has demonstrated that systems trained solely on text can acquire extensive world knowledge, develop reasoning capabilities, and internalize abstract semantic concepts--showcasing properties that can be associated with general intelligence. This raises an intriguing question: Do such concepts emerge in models trained on other modalities, such as speech? Furthermore, when models are trained jointly on multiple modalities: Do they develop a richer, more structured semantic understanding? To explore this, we analyze the conceptual structures learned by speech and textual models both individually and jointly. We employ Latent Concept Analysis, an unsupervised method for uncovering and interpreting latent representations in neural networks, to examine how semantic abstractions form across modalities. For reproducibility we made scripts and other resources available to the community.
#### Mispronunciation Detection Without L2 Pronunciation Dataset in Low-Resource Setting: A Case Study in Finland Swedish
 - **Authors:** Nhan Phan, Mikko Kuronen, Maria Kautonen, Riikka Ullakonoja, Anna von Zansen, Yaroslav Getman, Ekaterina Voskoboinik, Tamás Grósz, Mikko Kurimo
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01156

 - **Pdf link:** https://arxiv.org/pdf/2506.01156

 - **Abstract**
 Mispronunciation detection (MD) models are the cornerstones of many language learning applications. Unfortunately, most systems are built for English and other major languages, while low-resourced language varieties, such as Finland Swedish (FS), lack such tools. In this paper, we introduce our MD model for FS, trained on 89 hours of first language (L1) speakers' spontaneous speech and tested on 33 minutes of L2 transcribed read-aloud speech. We trained a multilingual wav2vec 2.0 model with entropy regularization, followed by temperature scaling and top-k normalization after the inference to better adapt it for MD. The main novelty of our method lies in its simplicity, requiring minimal L2 data. The process is also language-independent, making it suitable for other low-resource languages. Our proposed algorithm allows us to balance Recall (43.2%) and Precision (29.8%), compared with the baseline model's Recall (77.5%) and Precision (17.6%).
#### WCTC-Biasing: Retraining-free Contextual Biasing ASR with Wildcard CTC-based Keyword Spotting and Inter-layer Biasing
 - **Authors:** Yu Nakagome, Michael Hentschel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01263

 - **Pdf link:** https://arxiv.org/pdf/2506.01263

 - **Abstract**
 Despite recent advances in end-to-end speech recognition methods, the output tends to be biased to the training data's vocabulary, resulting in inaccurate recognition of proper nouns and other unknown terms. To address this issue, we propose a method to improve recognition accuracy of such rare words in CTC-based models without additional training or text-to-speech systems. Specifically, keyword spotting is performed using acoustic features of intermediate layers during inference, and a bias is applied to the subsequent layers of the acoustic model for detected keywords. For keyword detection, we adopt a wildcard CTC that is both fast and tolerant of ambiguous matches, allowing flexible handling of words that are difficult to match strictly. Since this method does not require retraining of existing models, it can be easily applied to even large-scale models. In experiments on Japanese speech recognition, the proposed method achieved a 29% improvement in the F1 score for unknown words.
#### Zero-Shot Text-to-Speech for Vietnamese
 - **Authors:** Thi Vu, Linh The Nguyen, Dat Quoc Nguyen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01322

 - **Pdf link:** https://arxiv.org/pdf/2506.01322

 - **Abstract**
 This paper introduces PhoAudiobook, a newly curated dataset comprising 941 hours of high-quality audio for Vietnamese text-to-speech. Using PhoAudiobook, we conduct experiments on three leading zero-shot TTS models: VALL-E, VoiceCraft, and XTTS-V2. Our findings demonstrate that PhoAudiobook consistently enhances model performance across various metrics. Moreover, VALL-E and VoiceCraft exhibit superior performance in synthesizing short sentences, highlighting their robustness in handling diverse linguistic contexts. We publicly release PhoAudiobook to facilitate further research and development in Vietnamese text-to-speech.
#### Whale: Large-Scale multilingual ASR model with w2v-BERT and E-Branchformer with large speech data
 - **Authors:** Yosuke Kashiwagi, Hayato Futami, Emiru Tsunoo, Satoshi Asakawa
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01439

 - **Pdf link:** https://arxiv.org/pdf/2506.01439

 - **Abstract**
 This paper reports on the development of a large-scale speech recognition model, Whale. Similar to models such as Whisper and OWSM, Whale leverages both a large model size and a diverse, extensive dataset. Whale's architecture integrates w2v-BERT self-supervised model, an encoder-decoder backbone built on E-Branchformer, and a joint CTC-attention decoding strategy. The training corpus comprises varied speech data, of not only public corpora but also in-house data, thereby enhancing the model's robustness to different speaking styles and acoustic conditions. Through evaluations on multiple benchmarks, Whale achieved comparable performance to existing models. In particular, it achieves a word error rate of 2.4% on the Librispeech test-clean set and a character error rate of 3.4% on the CSJ eval3 set, outperforming Whisper large-v3 and OWSM v3.1.
#### Universal Preference-Score-based Pairwise Speech Quality Assessment
 - **Authors:** Yu-Fei Shi, Yang Ai, Zhen-Hua Ling
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01455

 - **Pdf link:** https://arxiv.org/pdf/2506.01455

 - **Abstract**
 To compare the performance of two speech generation sys- tems, one of the most effective approaches is estimating the preference score between their generated speech. This pa- per proposes a novel universal preference-score-based pairwise speech quality assessment (UPPSQA) model, aimed at predict- ing the preference score between paired speech samples to de- termine which one has better quality. The model first predicts the absolute mean opinion score (MOS) for the two speech sam- ples separately, and then aggregates them into a relative prefer- ence score using a preference function. To address the scarcity of preference data, we also construct a new pairwise speech dataset based on a MOS dataset for experiments. Experimental results confirm that, whether in training scenarios with differ- ent data types and label conditions, or in both in-domain and out-of-domain test scenarios, the prediction accuracy of UPP- SQA outperforms that of the baseline models, demonstrating its universality.
#### TalTech Systems for the Interspeech 2025 ML-SUPERB 2.0 Challenge
 - **Authors:** Tanel Alumäe, Artem Fedorchenko
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01458

 - **Pdf link:** https://arxiv.org/pdf/2506.01458

 - **Abstract**
 This paper describes the language identification and multilingual speech recognition system developed at Tallinn University of Technology for the Interspeech 2025 ML-SUPERB 2.0 Challenge. A hybrid language identification system is used, consisting of a pretrained language embedding model and a light-weight speech recognition model with a shared encoder across languages and language-specific bigram language models. For speech recognition, three models are used, where only a single model is applied for each language, depending on the training data availability and performance on held-out data. The model set consists of a finetuned version of SeamlessM4T, MMS-1B-all with custom language adapters and MMS-zeroshot. The system obtained the top overall score in the challenge.
#### Few-step Adversarial Schrödinger Bridge for Generative Speech Enhancement
 - **Authors:** Seungu Han, Sungho Lee, Juheon Lee, Kyogu Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01460

 - **Pdf link:** https://arxiv.org/pdf/2506.01460

 - **Abstract**
 Deep generative models have recently been employed for speech enhancement to generate perceptually valid clean speech on large-scale datasets. Several diffusion models have been proposed, and more recently, a tractable Schrödinger Bridge has been introduced to transport between the clean and noisy speech distributions. However, these models often suffer from an iterative reverse process and require a large number of sampling steps -- more than 50. Our investigation reveals that the performance of baseline models significantly degrades when the number of sampling steps is reduced, particularly under low-SNR conditions. We propose integrating Schrödinger Bridge with GANs to effectively mitigate this issue, achieving high-quality outputs on full-band datasets while substantially reducing the required sampling steps. Experimental results demonstrate that our proposed model outperforms existing baselines, even with a single inference step, in both denoising and dereverberation tasks.
#### Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task?
 - **Authors:** Zijian Zhao, Dian Jin, Zijing Zhou, Xiaoyu Zhang
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01482

 - **Pdf link:** https://arxiv.org/pdf/2506.01482

 - **Abstract**
 Stage lighting plays an essential role in live music performances, influencing the engaging experience of both musicians and audiences. Given the high costs associated with hiring or training professional lighting engineers, Automatic Stage Lighting Control (ASLC) has gained increasing attention. However, most existing approaches only classify music into limited categories and map them to predefined light patterns, resulting in formulaic and monotonous outcomes that lack rationality. To address this issue, this paper presents an end-to-end solution that directly learns from experienced lighting engineers -- Skip-BART. To the best of our knowledge, this is the first work to conceptualize ASLC as a generative task rather than merely a classification problem. Our method modifies the BART model to take audio music as input and produce light hue and value (intensity) as output, incorporating a novel skip connection mechanism to enhance the relationship between music and light within the frame this http URL validate our method through both quantitative analysis and an human evaluation, demonstrating that Skip-BART outperforms conventional rule-based methods across all evaluation metrics and shows only a limited gap compared to real lighting this http URL, our method yields a p-value of 0.72 in a statistical comparison based on human evaluations with human lighting engineers, suggesting that the proposed approach closely matches human lighting engineering performance. To support further research, we have made our self-collected dataset, code, and trained model parameters available at this https URL .
#### Continual Speech Learning with Fused Speech Features
 - **Authors:** Guitao Wang, Jinming Zhao, Hao Yang, Guilin Qi, Tongtong Wu, Gholamreza Haffari
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01496

 - **Pdf link:** https://arxiv.org/pdf/2506.01496

 - **Abstract**
 Rapid growth in speech data demands adaptive models, as traditional static methods fail to keep pace with dynamic and diverse speech information. We introduce continuous speech learning, a new set-up targeting at bridging the adaptation gap in current speech models. We use the encoder-decoder Whisper model to standardize speech tasks into a generative format. We integrate a learnable gated-fusion layer on the top of the encoder to dynamically select task-specific features for downstream tasks. Our approach improves accuracy significantly over traditional methods in six speech processing tasks, demonstrating gains in adapting to new speech tasks without full retraining.
#### Learning Perceptually Relevant Temporal Envelope Morphing
 - **Authors:** Satvik Dixit, Sungjoon Park, Chris Donahue, Laurie M. Heller
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.01588

 - **Pdf link:** https://arxiv.org/pdf/2506.01588

 - **Abstract**
 Temporal envelope morphing, the process of interpolating between the amplitude dynamics of two audio signals, is an emerging problem in generative audio systems that lacks sufficient perceptual grounding. Morphing of temporal envelopes in a perceptually intuitive manner should enable new methods for sound blending in creative media and for probing perceptual organization in psychoacoustics. However, existing audio morphing techniques often fail to produce intermediate temporal envelopes when input sounds have distinct temporal structures; many morphers effectively overlay both temporal structures, leading to perceptually unnatural results. In this paper, we introduce a novel workflow for learning envelope morphing with perceptual guidance: we first derive perceptually grounded morphing principles through human listening studies, then synthesize large-scale datasets encoding these principles, and finally train machine learning models to create perceptually intermediate morphs. Specifically, we present: (1) perceptual principles that guide envelope morphing, derived from our listening studies, (2) a supervised framework to learn these principles, (3) an autoencoder that learns to compress temporal envelope structures into latent representations, and (4) benchmarks for evaluating audio envelope morphs, using both synthetic and naturalistic data, and show that our approach outperforms existing methods in producing temporally intermediate morphs. All code, models, and datasets will be made publicly available upon publication.
#### Datasheets Aren't Enough: DataRubrics for Automated Quality Metrics and Accountability
 - **Authors:** Genta Indra Winata, David Anugraha, Emmy Liu, Alham Fikri Aji, Shou-Yi Hung, Aditya Parashar, Patrick Amadeus Irawan, Ruochen Zhang, Zheng-Xin Yong, Jan Christian Blaise Cruz, Niklas Muennighoff, Seungone Kim, Hanyang Zhao, Sudipta Kar, Kezia Erina Suryoraharjo, M. Farid Adilazuarda, En-Shiun Annie Lee, Ayu Purwarianti, Derry Tanti Wijaya, Monojit Choudhury
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.01789

 - **Pdf link:** https://arxiv.org/pdf/2506.01789

 - **Abstract**
 High-quality datasets are fundamental to training and evaluating machine learning models, yet their creation-especially with accurate human annotations-remains a significant challenge. Many dataset paper submissions lack originality, diversity, or rigorous quality control, and these shortcomings are often overlooked during peer review. Submissions also frequently omit essential details about dataset construction and properties. While existing tools such as datasheets aim to promote transparency, they are largely descriptive and do not provide standardized, measurable methods for evaluating data quality. Similarly, metadata requirements at conferences promote accountability but are inconsistently enforced. To address these limitations, this position paper advocates for the integration of systematic, rubric-based evaluation metrics into the dataset review process-particularly as submission volumes continue to grow. We also explore scalable, cost-effective methods for synthetic data generation, including dedicated tools and LLM-as-a-judge approaches, to support more efficient evaluation. As a call to action, we introduce DataRubrics, a structured framework for assessing the quality of both human- and model-generated datasets. Leveraging recent advances in LLM-based evaluation, DataRubrics offers a reproducible, scalable, and actionable solution for dataset quality assessment, enabling both authors and reviewers to uphold higher standards in data-centric research. We also release code to support reproducibility of LLM-based evaluations at this https URL.


by Zyzzyva0381 (Windy). 


2025-06-03
