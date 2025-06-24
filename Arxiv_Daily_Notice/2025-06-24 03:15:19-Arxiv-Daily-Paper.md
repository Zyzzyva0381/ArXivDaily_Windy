# Showing new listings for Tuesday, 24 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 20papers 
#### Enhancing Few-shot Keyword Spotting Performance through Pre-Trained Self-supervised Speech Models
 - **Authors:** Alican Gok, Oguzhan Buyuksolak, Osman Erman Okman, Murat Saraclar
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.17686

 - **Pdf link:** https://arxiv.org/pdf/2506.17686

 - **Abstract**
 Keyword Spotting plays a critical role in enabling hands-free interaction for battery-powered edge devices. Few-Shot Keyword Spotting (FS-KWS) addresses the scalability and adaptability challenges of traditional systems by enabling recognition of custom keywords with only a few examples. However, existing FS-KWS systems achieve subpar accuracy at desirable false acceptance rates, particularly in resource-constrained edge environments. To address these issues, we propose a training scheme that leverages self-supervised learning models for robust feature extraction, dimensionality reduction, and knowledge distillation. The teacher model, based on Wav2Vec 2.0 is trained using Sub-center ArcFace loss, which enhances inter-class separability and intra-class compactness. To enable efficient deployment on edge devices, we introduce attention-based dimensionality reduction and train a standard lightweight ResNet15 student model. We evaluate the proposed approach on the English portion of the Multilingual Spoken Words Corpus (MSWC) and the Google Speech Commands (GSC) datasets. Notably, the proposed training method improves the 10-shot classification accuracy from 33.4% to 74.1% on 11 classes at 1% false alarm accuracy on the GSC dataset, thus making it significantly better-suited for a real use case scenario.
#### Low-resource keyword spotting using contrastively trained transformer acoustic word embeddings
 - **Authors:** Julian Herreilers, Christiaan Jacobs, Thomas Niesler
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.17690

 - **Pdf link:** https://arxiv.org/pdf/2506.17690

 - **Abstract**
 We introduce a new approach, the ContrastiveTransformer, that produces acoustic word embeddings (AWEs) for the purpose of very low-resource keyword spotting. The ContrastiveTransformer, an encoder-only model, directly optimises the embedding space using normalised temperature-scaled cross entropy (NT-Xent) loss. We use this model to perform keyword spotting for radio broadcasts in Luganda and Bambara, the latter a severely under-resourced language. We compare our model to various existing AWE approaches, including those constructed from large pre-trained self-supervised models, a recurrent encoder which previously used the NT-Xent loss, and a DTW baseline. We demonstrate that the proposed contrastive transformer approach offers performance improvements over all considered existing approaches to very low-resource keyword spotting in both languages.
#### Blind Source Separation in Biomedical Signals Using Variational Methods
 - **Authors:** Yasaman Torabi, Shahram Shirani, James P. Reilly
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.18281

 - **Pdf link:** https://arxiv.org/pdf/2506.18281

 - **Abstract**
 This study introduces a novel unsupervised approach for separating overlapping heart and lung sounds using variational autoencoders (VAEs). In clinical settings, these sounds often interfere with each other, making manual separation difficult and error-prone. The proposed model learns to encode mixed signals into a structured latent space and reconstructs the individual components using a probabilistic decoder, all without requiring labeled data or prior knowledge of source characteristics. We apply this method to real recordings obtained from a clinical manikin using a digital stethoscope. Results demonstrate distinct latent clusters corresponding to heart and lung sources, as well as accurate reconstructions that preserve key spectral features of the original signals. The approach offers a robust and interpretable solution for blind source separation and has potential applications in portable diagnostic tools and intelligent stethoscope systems.
#### Infant Cry Emotion Recognition Using Improved ECAPA-TDNN with Multiscale Feature Fusion and Attention Enhancement
 - **Authors:** Junyu Zhou, Yanxiong Li, Haolin Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18402

 - **Pdf link:** https://arxiv.org/pdf/2506.18402

 - **Abstract**
 Infant cry emotion recognition is crucial for parenting and medical applications. It faces many challenges, such as subtle emotional variations, noise interference, and limited data. The existing methods lack the ability to effectively integrate multi-scale features and temporal-frequency relationships. In this study, we propose a method for infant cry emotion recognition using an improved Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network (ECAPA-TDNN) with both multi-scale feature fusion and attention enhancement. Experiments on a public dataset show that the proposed method achieves accuracy of 82.20%, number of parameters of 1.43 MB and FLOPs of 0.32 Giga. Moreover, our method has advantage over the baseline methods in terms of accuracy. The code is at this https URL.
#### Zero-Shot Cognitive Impairment Detection from Speech Using AudioLLM
 - **Authors:** Mostafa Shahin, Beena Ahmed, Julien Epps
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.17351

 - **Pdf link:** https://arxiv.org/pdf/2506.17351

 - **Abstract**
 Cognitive impairment (CI) is of growing public health concern, and early detection is vital for effective intervention. Speech has gained attention as a non-invasive and easily collectible biomarker for assessing cognitive decline. Traditional CI detection methods typically rely on supervised models trained on acoustic and linguistic features extracted from speech, which often require manual annotation and may not generalise well across datasets and languages. In this work, we propose the first zero-shot speech-based CI detection method using the Qwen2- Audio AudioLLM, a model capable of processing both audio and text inputs. By designing prompt-based instructions, we guide the model in classifying speech samples as indicative of normal cognition or cognitive impairment. We evaluate our approach on two datasets: one in English and another multilingual, spanning different cognitive assessment tasks. Our results show that the zero-shot AudioLLM approach achieves performance comparable to supervised methods and exhibits promising generalizability and consistency across languages, tasks, and datasets.
#### Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation
 - **Authors:** Quoc Thinh Vo, Joe Woods, Priontu Chowdhury, David K. Han
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.17409

 - **Pdf link:** https://arxiv.org/pdf/2506.17409

 - **Abstract**
 Localizing acoustic sound sources in the ocean is a challenging task due to the complex and dynamic nature of the environment. Factors such as high background noise, irregular underwater geometries, and varying acoustic properties make accurate localization difficult. To address these obstacles, we propose a multi-branch network architecture designed to accurately predict the distance between a moving acoustic source and a receiver, tested on real-world underwater signal arrays. The network leverages Convolutional Neural Networks (CNNs) for robust spatial feature extraction and integrates Conformers with self-attention mechanism to effectively capture temporal dependencies. Log-mel spectrogram and generalized cross-correlation with phase transform (GCC-PHAT) features are employed as input representations. To further enhance the model performance, we introduce an Adaptive Gain Control (AGC) layer, that adaptively adjusts the amplitude of input features, ensuring consistent energy levels across varying ranges, signal strengths, and noise conditions. We assess the model's generalization capability by training it in one domain and testing it in a different domain, using only a limited amount of data from the test domain for fine-tuning. Our proposed method outperforms state-of-the-art (SOTA) approaches in similar settings, establishing new benchmarks for underwater sound localization.
#### From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training
 - **Authors:** Mingyang Yao, Ke Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.17497

 - **Pdf link:** https://arxiv.org/pdf/2506.17497

 - **Abstract**
 Despite progress in controllable symbolic music generation, data scarcity remains a challenge for certain control modalities. Composer-style music generation is a prime example, as only a few pieces per composer are available, limiting the modeling of both styles and fundamental music elements (e.g., melody, chord, rhythm). In this paper, we investigate how general music knowledge learned from a broad corpus can enhance the mastery of specific composer styles, with a focus on piano piece generation. Our approach follows a two-stage training paradigm. First, we pre-train a REMI-based music generation model on a large corpus of pop, folk, and classical music. Then, we fine-tune it on a small, human-verified dataset from four renowned composers, namely Bach, Mozart, Beethoven, and Chopin, using a lightweight adapter module to condition the model on style indicators. To evaluate the effectiveness of our approach, we conduct both objective and subjective evaluations on style accuracy and musicality. Experimental results demonstrate that our method outperforms ablations and baselines, achieving more precise composer-style modeling and better musical aesthetics. Additionally, we provide observations on how the model builds music concepts from the generality pre-training and refines its stylistic understanding through the mastery fine-tuning.
#### CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning
 - **Authors:** Angelos-Nikolaos Kanatas, Charilaos Papaioannou, Alexandros Potamianos
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.17818

 - **Pdf link:** https://arxiv.org/pdf/2506.17818

 - **Abstract**
 Recent advances in music foundation models have improved audio representation learning, yet their effectiveness across diverse musical traditions remains limited. We introduce CultureMERT-95M, a multi-culturally adapted foundation model developed to enhance cross-cultural music representation learning and understanding. To achieve this, we propose a two-stage continual pre-training strategy that integrates learning rate re-warming and re-decaying, enabling stable adaptation even with limited computational resources. Training on a 650-hour multi-cultural data mix, comprising Greek, Turkish, and Indian music traditions, results in an average improvement of 4.9% in ROC-AUC and AP across diverse non-Western music auto-tagging tasks, surpassing prior state-of-the-art, with minimal forgetting on Western-centric benchmarks. We further investigate task arithmetic, an alternative approach to multi-cultural adaptation that merges single-culture adapted models in the weight space. Task arithmetic performs on par with our multi-culturally trained model on non-Western auto-tagging tasks and shows no regression on Western datasets. Cross-cultural evaluation reveals that single-culture models transfer with varying effectiveness across musical traditions, whereas the multi-culturally adapted model achieves the best overall performance. To support research on world music representation learning, we publicly release CultureMERT-95M and CultureMERT-TA-95M, fostering the development of more culturally aware music foundation models.
#### Splitformer: An improved early-exit architecture for automatic speech recognition on edge devices
 - **Authors:** Maxence Lasbordes, Daniele Falavigna, Alessio Brutti
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18035

 - **Pdf link:** https://arxiv.org/pdf/2506.18035

 - **Abstract**
 The ability to dynamically adjust the computational load of neural models during inference in a resource aware manner is crucial for on-device processing scenarios, characterised by limited and time-varying computational resources. Early-exit architectures represent an elegant and effective solution, since they can process the input with a subset of their layers, exiting at intermediate branches (the upmost layers are hence removed from the model). From a different perspective, for automatic speech recognition applications there are memory-efficient neural architectures that apply variable frame rate analysis, through downsampling/upsampling operations in the middle layers, reducing the overall number of operations and improving significantly the performance on well established benchmarks. One example is the Zipformer. However, these architectures lack the modularity necessary to inject early-exit branches. With the aim of improving the performance in early-exit models, we propose introducing parallel layers in the architecture that process downsampled versions of their inputs. % in conjunction with standard processing layers. We show that in this way the speech recognition performance on standard benchmarks significantly improve, at the cost of a small increase in the overall number of model parameters but without affecting the inference time.
#### Face-Voice Association for Audiovisual Active Speaker Detection in Egocentric Recordings
 - **Authors:** Jason Clarke, Yoshihiko Gotoh, Stefan Goetze
 - **Subjects:** Subjects:
Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18055

 - **Pdf link:** https://arxiv.org/pdf/2506.18055

 - **Abstract**
 Audiovisual active speaker detection (ASD) is conventionally performed by modelling the temporal synchronisation of acoustic and visual speech cues. In egocentric recordings, however, the efficacy of synchronisation-based methods is compromised by occlusions, motion blur, and adverse acoustic conditions. In this work, a novel framework is proposed that exclusively leverages cross-modal face-voice associations to determine speaker activity. An existing face-voice association model is integrated with a transformer-based encoder that aggregates facial identity information by dynamically weighting each frame based on its visual quality. This system is then coupled with a front-end utterance segmentation method, producing a complete ASD system. This work demonstrates that the proposed system, Self-Lifting for audiovisual active speaker detection(SL-ASD), achieves performance comparable to, and in certain cases exceeding, that of parameter-intensive synchronisation-based approaches with significantly fewer learnable parameters, thereby validating the feasibility of substituting strict audiovisual synchronisation modelling with flexible biometric associations in challenging egocentric scenarios.
#### JIS: A Speech Corpus of Japanese Idol Speakers with Various Speaking Styles
 - **Authors:** Yuto Kondo, Hirokazu Kameoka, Kou Tanaka, Takuhiro Kaneko
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18296

 - **Pdf link:** https://arxiv.org/pdf/2506.18296

 - **Abstract**
 We construct Japanese Idol Speech Corpus (JIS) to advance research in speech generation AI, including text-to-speech synthesis (TTS) and voice conversion (VC). JIS will facilitate more rigorous evaluations of speaker similarity in TTS and VC systems since all speakers in JIS belong to a highly specific category: "young female live idols" in Japan, and each speaker is identified by a stage name, enabling researchers to recruit listeners familiar with these idols for listening experiments. With its unique speaker attributes, JIS will foster compelling research, including generating voices tailored to listener preferences-an area not yet widely studied. JIS will be distributed free of charge to promote research in speech generation AI, with usage restricted to non-commercial, basic research. We describe the construction of JIS, provide an overview of Japanese live idol culture to support effective and ethical use of JIS, and offer a basic analysis to guide application of JIS.
#### Rethinking Mean Opinion Scores in Speech Quality Assessment: Aggregation through Quantized Distribution Fitting
 - **Authors:** Yuto Kondo, Hirokazu Kameoka, Kou Tanaka, Takuhiro Kaneko
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18307

 - **Pdf link:** https://arxiv.org/pdf/2506.18307

 - **Abstract**
 Speech quality assessment (SQA) aims to evaluate the quality of speech samples without relying on time-consuming listener questionnaires. Recent efforts have focused on training neural-based SQA models to predict the mean opinion score (MOS) of speech samples produced by text-to-speech or voice conversion systems. This paper targets the enhancement of MOS prediction models' performance. We propose a novel score aggregation method to address the limitations of conventional annotations for MOS, which typically involve ratings on a scale from 1 to 5. Our method is based on the hypothesis that annotators internally consider continuous scores and then choose the nearest discrete rating. By modeling this process, we approximate the generative distribution of ratings by quantizing the latent continuous distribution. We then use the peak of this latent distribution, estimated through the loss between the quantized distribution and annotated ratings, as a new representative value instead of MOS. Experimental results demonstrate that substituting MOSNet's predicted target with this proposed value improves prediction performance.
#### Selecting N-lowest scores for training MOS prediction models
 - **Authors:** Yuto Kondo, Hirokazu Kameoka, Kou Tanaka, Takuhiro Kaneko
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18326

 - **Pdf link:** https://arxiv.org/pdf/2506.18326

 - **Abstract**
 The automatic speech quality assessment (SQA) has been extensively studied to predict the speech quality without time-consuming questionnaires. Recently, neural-based SQA models have been actively developed for speech samples produced by text-to-speech or voice conversion, with a primary focus on training mean opinion score (MOS) prediction models. The quality of each speech sample may not be consistent across the entire duration, and it remains unclear which segments of the speech receive the primary focus from humans when assigning subjective evaluation for MOS calculation. We hypothesize that when humans rate speech, they tend to assign more weight to low-quality speech segments, and the variance in ratings for each sample is mainly due to accidental assignment of higher scores when overlooking the poor quality speech segments. Motivated by the hypothesis, we analyze the VCC2018 and BVCC datasets. Based on the hypothesis, we propose the more reliable representative value N_low-MOS, the mean of the $N$-lowest opinion scores. Our experiments show that LCC and SRCC improve compared to regular MOS when employing N_low-MOS to MOSNet training. This result suggests that N_low-MOS is a more intrinsic representative value of subjective speech quality and makes MOSNet a better comparator of VC models.
#### Smooth Operators: LLMs Translating Imperfect Hints into Disfluency-Rich Transcripts
 - **Authors:** Duygu Altinok
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18510

 - **Pdf link:** https://arxiv.org/pdf/2506.18510

 - **Abstract**
 Accurate detection of disfluencies in spoken language is crucial for enhancing the performance of automatic speech and language processing systems, as well as fostering the development of more inclusive speech and language technologies. Leveraging the growing trend of large language models (LLMs) as versatile learners capable of processing both lexical and non-lexical inputs (e.g., audio and video), we propose a novel approach to transcribing disfluencies as explicit tokens with timestamps, enabling the generation of fully annotated disfluency-rich transcripts. Our method integrates acoustic representations extracted from an audio encoder with textual inputs of varying quality: clean transcriptions without disfluencies, time-aligned transcriptions from aligners, or outputs from phoneme-based ASR models -- all of which may contain imperfections. Importantly, our experiments demonstrate that textual inputs do not need to be flawless. As long as they include timestamp-related cues, LLMs can effectively smooth the input and produce fully disfluency-annotated transcripts, underscoring their robustness in handling imperfect hints.
#### TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography
 - **Authors:** Yuqin Dai, Wanlu Zhu, Ronghui Li, Xiu Li, Zhenyu Zhang, Jun Li, Jian Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Graphics (cs.GR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18671

 - **Pdf link:** https://arxiv.org/pdf/2506.18671

 - **Abstract**
 Music-driven dance generation has garnered significant attention due to its wide range of industrial applications, particularly in the creation of group choreography. During the group dance generation process, however, most existing methods still face three primary issues: multi-dancer collisions, single-dancer foot sliding and abrupt swapping in the generation of long group dance. In this paper, we propose TCDiff++, a music-driven end-to-end framework designed to generate harmonious group dance. Specifically, to mitigate multi-dancer collisions, we utilize a dancer positioning embedding to better maintain the relative positioning among dancers. Additionally, we incorporate a distance-consistency loss to ensure that inter-dancer distances remain within plausible ranges. To address the issue of single-dancer foot sliding, we introduce a swap mode embedding to indicate dancer swapping patterns and design a Footwork Adaptor to refine raw motion, thereby minimizing foot sliding. For long group dance generation, we present a long group diffusion sampling strategy that reduces abrupt position shifts by injecting positional information into the noisy input. Furthermore, we integrate a Sequence Decoder layer to enhance the model's ability to selectively process long sequences. Extensive experiments demonstrate that our TCDiff++ achieves state-of-the-art performance, particularly in long-duration scenarios, ensuring high-quality and coherent group dance generation.
#### DuetGen: Music Driven Two-Person Dance Generation via Hierarchical Masked Modeling
 - **Authors:** Anindita Ghosh, Bing Zhou, Rishabh Dabral, Jian Wang, Vladislav Golyanik, Christian Theobalt, Philipp Slusallek, Chuan Guo
 - **Subjects:** Subjects:
Graphics (cs.GR); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18680

 - **Pdf link:** https://arxiv.org/pdf/2506.18680

 - **Abstract**
 We present DuetGen, a novel framework for generating interactive two-person dances from music. The key challenge of this task lies in the inherent complexities of two-person dance interactions, where the partners need to synchronize both with each other and with the music. Inspired by the recent advances in motion synthesis, we propose a two-stage solution: encoding two-person motions into discrete tokens and then generating these tokens from music. To effectively capture intricate interactions, we represent both dancers' motions as a unified whole to learn the necessary motion tokens, and adopt a coarse-to-fine learning strategy in both the stages. Our first stage utilizes a VQ-VAE that hierarchically separates high-level semantic features at a coarse temporal resolution from low-level details at a finer resolution, producing two discrete token sequences at different abstraction levels. Subsequently, in the second stage, two generative masked transformers learn to map music signals to these dance tokens: the first producing high-level semantic tokens, and the second, conditioned on music and these semantic tokens, producing the low-level tokens. We train both transformers to learn to predict randomly masked tokens within the sequence, enabling them to iteratively generate motion tokens by filling an empty token sequence during inference. Through the hierarchical masked modeling and dedicated interaction representation, DuetGen achieves the generation of synchronized and interactive two-person dances across various genres. Extensive experiments and user studies on a benchmark duet dance dataset demonstrate state-of-the-art performance of DuetGen in motion realism, music-dance alignment, and partner coordination.
#### Evaluating Multichannel Speech Enhancement Algorithms at the Phoneme Scale Across Genders
 - **Authors:** Nasser-Eddine Monir, Paul Magron, Romain Serizel
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18691

 - **Pdf link:** https://arxiv.org/pdf/2506.18691

 - **Abstract**
 Multichannel speech enhancement algorithms are essential for improving the intelligibility of speech signals in noisy environments. These algorithms are usually evaluated at the utterance level, but this approach overlooks the disparities in acoustic characteristics that are observed in different phoneme categories and between male and female speakers. In this paper, we investigate the impact of gender and phonetic content on speech enhancement algorithms. We motivate this approach by outlining phoneme- and gender-specific spectral features. Our experiments reveal that while utterance-level differences between genders are minimal, significant variations emerge at the phoneme level. Results show that the tested algorithms better reduce interference with fewer artifacts on female speech, particularly in plosives, fricatives, and vowels. Additionally, they demonstrate greater performance for female speech in terms of perceptual and speech recognition metrics.
#### Frequency-Weighted Training Losses for Phoneme-Level DNN-based Speech Enhancement
 - **Authors:** Nasser-Eddine Monir, Paul Magron, Romain Serizel
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18714

 - **Pdf link:** https://arxiv.org/pdf/2506.18714

 - **Abstract**
 Recent advances in deep learning have significantly improved multichannel speech enhancement algorithms, yet conventional training loss functions such as the scale-invariant signal-to-distortion ratio (SDR) may fail to preserve fine-grained spectral cues essential for phoneme intelligibility. In this work, we propose perceptually-informed variants of the SDR loss, formulated in the time-frequency domain and modulated by frequency-dependent weighting schemes. These weights are designed to emphasize time-frequency regions where speech is prominent or where the interfering noise is particularly strong. We investigate both fixed and adaptive strategies, including ANSI band-importance weights, spectral magnitude-based weighting, and dynamic weighting based on the relative amount of speech and noise. We train the FaSNet multichannel speech enhancement model using these various losses. Experimental results show that while standard metrics such as the SDR are only marginally improved, their perceptual frequency-weighted counterparts exhibit a more substantial improvement. Besides, spectral and phoneme-level analysis indicates better consonant reconstruction, which points to a better preservation of certain acoustic cues.
#### An Audio-centric Multi-task Learning Framework for Streaming Ads Targeting on Spotify
 - **Authors:** Shivam Verma, Vivian Chen, Darren Mei
 - **Subjects:** Subjects:
Information Retrieval (cs.IR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18735

 - **Pdf link:** https://arxiv.org/pdf/2506.18735

 - **Abstract**
 Spotify, a large-scale multimedia platform, attracts over 675 million monthly active users who collectively consume millions of hours of music, podcasts, audiobooks, and video content. This diverse content consumption pattern introduces unique challenges for computational advertising, which must effectively integrate a variety of ad modalities, including audio, video, and display, within a single user experience. Traditional ad recommendation models, primarily designed for foregrounded experiences, often struggle to reconcile the platform's inherent audio-centrality with the demands of optimizing ad performance across multiple formats and modalities. To overcome these challenges, we introduce Cross-modal Adaptive Mixture-of-Experts (CAMoE), a novel framework for optimizing click-through rate (CTR) prediction in both audio-centric and multi-modal settings. CAMoE enhances traditional mixture-of-experts models by incorporating modality-aware task grouping, adaptive loss masking, and deep-cross networks (DCN) to capture complex feature interactions within a multi-modal ad ecosystem. Through extensive ablation studies, we demonstrate that this approach achieves near Pareto-optimal performance across audio, video, and display ad formats, significantly improving AUC-PR compared to conventional single-task and content-based multi-task learning baselines. When deployed at scale on Spotify's ad serving platform, CAMoE delivered substantial gains, yielding a 14.5% increase in CTR for audio ads, a 1.3% increase for video ads, and a 4.8% reduction in expected cost-per-click (eCPC) for audio slots.
#### USAD: Universal Speech and Audio Representation via Distillation
 - **Authors:** Heng-Jui Chang, Saurabhchand Bhati, James Glass, Alexander H. Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18843

 - **Pdf link:** https://arxiv.org/pdf/2506.18843

 - **Abstract**
 Self-supervised learning (SSL) has revolutionized audio representations, yet models often remain domain-specific, focusing on either speech or non-speech tasks. In this work, we present Universal Speech and Audio Distillation (USAD), a unified approach to audio representation learning that integrates diverse audio types - speech, sound, and music - into a single model. USAD employs efficient layer-to-layer distillation from domain-specific SSL models to train a student on a comprehensive audio dataset. USAD offers competitive performance across various benchmarks and datasets, including frame and instance-level speech processing tasks, audio tagging, and sound classification, achieving near state-of-the-art results with a single encoder on SUPERB and HEAR benchmarks.


by Zyzzyva0381 (Windy). 


2025-06-24
