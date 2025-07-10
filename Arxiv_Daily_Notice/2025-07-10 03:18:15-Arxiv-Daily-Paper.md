# Showing new listings for Thursday, 10 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 17papers 
#### Pronunciation-Lexicon Free Training for Phoneme-based Crosslingual ASR via Joint Stochastic Approximation
 - **Authors:** Saierdaer Yusuyin, Te Ma, Hao Huang, Zhijian Ou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.06249

 - **Pdf link:** https://arxiv.org/pdf/2507.06249

 - **Abstract**
 Recently, pre-trained models with phonetic supervision have demonstrated their advantages for crosslingual speech recognition in data efficiency and information sharing across languages. However, a limitation is that a pronunciation lexicon is needed for such phoneme-based crosslingual speech recognition. In this study, we aim to eliminate the need for pronunciation lexicons and propose a latent variable model based method, with phonemes being treated as discrete latent variables. The new method consists of a speech-to-phoneme (S2P) model and a phoneme-to-grapheme (P2G) model, and a grapheme-to-phoneme (G2P) model is introduced as an auxiliary inference model. To jointly train the three models, we utilize the joint stochastic approximation (JSA) algorithm, which is a stochastic extension of the EM (expectation-maximization) algorithm and has demonstrated superior performance particularly in estimating discrete latent variable models. Based on the Whistle multilingual pre-trained S2P model, crosslingual experiments are conducted in Polish (130 h) and Indonesian (20 h). With only 10 minutes of phoneme supervision, the new method, JSA-SPG, achieves 5\% error rate reductions compared to the best crosslingual fine-tuning approach using subword or full phoneme supervision. Furthermore, it is found that in language domain adaptation (i.e., utilizing cross-domain text-only data), JSA-SPG outperforms the standard practice of language model fusion via the auxiliary support of the G2P model by 9% error rate reductions. To facilitate reproducibility and encourage further exploration in this field, we open-source the JSA-SPG training code and complete pipeline.
#### Open-Set Source Tracing of Audio Deepfake Systems
 - **Authors:** Nicholas Klein, Hemlata Tak, Elie Khoury
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.06470

 - **Pdf link:** https://arxiv.org/pdf/2507.06470

 - **Abstract**
 Existing research on source tracing of audio deepfake systems has focused primarily on the closed-set scenario, while studies that evaluate open-set performance are limited to a small number of unseen systems. Due to the large number of emerging audio deepfake systems, robust open-set source tracing is critical. We leverage the protocol of the Interspeech 2025 special session on source tracing to evaluate methods for improving open-set source tracing performance. We introduce a novel adaptation to the energy score for out-of-distribution (OOD) detection, softmax energy (SME). We find that replacing the typical temperature-scaled energy score with SME provides a relative average improvement of 31% in the standard FPR95 (false positive rate at true positive rate of 95%) measure. We further explore SME-guided training as well as copy synthesis, codec, and reverberation augmentations, yielding an FPR95 of 8.3%.
#### Training Strategies for Modality Dropout Resilient Multi-Modal Target Speaker Extraction
 - **Authors:** Srikanth Korse, Mohamed Elminshawi, Emanuel A. P. Habets, Srikanth Raj Chetupalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06566

 - **Pdf link:** https://arxiv.org/pdf/2507.06566

 - **Abstract**
 The primary goal of multi-modal TSE (MTSE) is to extract a target speaker from a speech mixture using complementary information from different modalities, such as audio enrolment and visual feeds corresponding to the target speaker. MTSE systems are expected to perform well even when one of the modalities is unavailable. In practice, the systems often suffer from modality dominance, where one of the modalities outweighs the others, thereby limiting robustness. Our study investigates training strategies and the effect of architectural choices, particularly the normalization layers, in yielding a robust MTSE system in both non-causal and causal configurations. In particular, we propose the use of modality dropout training (MDT) as a superior strategy to standard and multi-task training (MTT) strategies. Experiments conducted on two-speaker mixtures from the LRS3 dataset show the MDT strategy to be effective irrespective of the employed normalization layer. In contrast, the models trained with the standard and MTT strategies are susceptible to modality dominance, and their performance depends on the chosen normalization layer. Additionally, we demonstrate that the system trained with MDT strategy is robust to using extracted speech as the enrollment signal, highlighting its potential applicability in scenarios where the target speaker is not enrolled.
#### Musical Source Separation Bake-Off: Comparing Objective Metrics with Human Perception
 - **Authors:** Noah Jaffe, John Ashley Burgoyne
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06917

 - **Pdf link:** https://arxiv.org/pdf/2507.06917

 - **Abstract**
 Music source separation aims to extract individual sound sources (e.g., vocals, drums, guitar) from a mixed music recording. However, evaluating the quality of separated audio remains challenging, as commonly used metrics like the source-to-distortion ratio (SDR) do not always align with human perception. In this study, we conducted a large-scale listener evaluation on the MUSDB18 test set, collecting approximately 30 ratings per track from seven distinct listener groups. We compared several objective energy-ratio metrics, including legacy measures (BSSEval v4, SI-SDR variants), and embedding-based alternatives (Frechet Audio Distance using CLAP-LAION-music, EnCodec, VGGish, Wave2Vec2, and HuBERT). While SDR remains the best-performing metric for vocal estimates, our results show that the scale-invariant signal-to-artifacts ratio (SI-SAR) better predicts listener ratings for drums and bass stems. Frechet Audio Distance (FAD) computed with the CLAP-LAION-music embedding also performs competitively--achieving Kendall's tau values of 0.25 for drums and 0.19 for bass--matching or surpassing energy-based metrics for those stems. However, none of the embedding-based metrics, including CLAP, correlate positively with human perception for vocal estimates. These findings highlight the need for stem-specific evaluation strategies and suggest that no single metric reliably reflects perceptual quality across all source types. We release our raw listener ratings to support reproducibility and further research.
#### Deep Feed-Forward Neural Network for Bangla Isolated Speech Recognition
 - **Authors:** Dipayan Bhadra, Mehrab Hosain, Fatema Alam
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.07068

 - **Pdf link:** https://arxiv.org/pdf/2507.07068

 - **Abstract**
 As the most important human-machine interfacing tool, an insignificant amount of work has been carried out on Bangla Speech Recognition compared to the English language. Motivated by this, in this work, the performance of speaker-independent isolated speech recognition systems has been implemented and analyzed using a dataset that is created containing both isolated Bangla and English spoken words. An approach using the Mel Frequency Cepstral Coefficient (MFCC) and Deep Feed-Forward Fully Connected Neural Network (DFFNN) of 7 layers as a classifier is proposed in this work to recognize isolated spoken words. This work shows 93.42% recognition accuracy which is better compared to most of the works done previously on Bangla speech recognition considering the number of classes and dataset size.
#### Incremental Averaging Method to Improve Graph-Based Time-Difference-of-Arrival Estimation
 - **Authors:** Klaus Brümann, Kouei Yamaoka, Nobutaka Ono, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.07087

 - **Pdf link:** https://arxiv.org/pdf/2507.07087

 - **Abstract**
 Estimating the position of a speech source based on time-differences-of-arrival (TDOAs) is often adversely affected by background noise and reverberation. A popular method to estimate the TDOA between a microphone pair involves maximizing a generalized cross-correlation with phase transform (GCC-PHAT) function. Since the TDOAs across different microphone pairs satisfy consistency relations, generally only a small subset of microphone pairs are used for source position estimation. Although the set of microphone pairs is often determined based on a reference microphone, recently a more robust method has been proposed to determine the set of microphone pairs by computing the minimum spanning tree (MST) of a signal graph of GCC-PHAT function reliabilities. To reduce the influence of noise and reverberation on the TDOA estimation accuracy, in this paper we propose to compute the GCC-PHAT functions of the MST based on an average of multiple cross-power spectral densities (CPSDs) using an incremental method. In each step of the method, we increase the number of CPSDs over which we average by considering CPSDs computed indirectly via other microphones from previous steps. Using signals recorded in a noisy and reverberant laboratory with an array of spatially distributed microphones, the performance of the proposed method is evaluated in terms of TDOA estimation error and 2D source position estimation error. Experimental results for different source and microphone configurations and three reverberation conditions show that the proposed method considering multiple CPSDs improves the TDOA estimation and source position estimation accuracy compared to the reference microphone- and MST-based methods that rely on a single CPSD as well as steered-response power-based source position estimation.
#### Super Kawaii Vocalics: Amplifying the "Cute" Factor in Computer Voice
 - **Authors:** Yuto Mandai, Katie Seaborn, Tomoyasu Nakano, Xin Sun, Yijia Wang, Jun Kato
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computers and Society (cs.CY); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06235

 - **Pdf link:** https://arxiv.org/pdf/2507.06235

 - **Abstract**
 "Kawaii" is the Japanese concept of cute, which carries sociocultural connotations related to social identities and emotional responses. Yet, virtually all work to date has focused on the visual side of kawaii, including in studies of computer agents and social robots. In pursuit of formalizing the new science of kawaii vocalics, we explored what elements of voice relate to kawaii and how they might be manipulated, manually and automatically. We conducted a four-phase study (grand N = 512) with two varieties of computer voices: text-to-speech (TTS) and game character voices. We found kawaii "sweet spots" through manipulation of fundamental and formant frequencies, but only for certain voices and to a certain extent. Findings also suggest a ceiling effect for the kawaii vocalics of certain voices. We offer empirical validation of the preliminary kawaii vocalics model and an elementary method for manipulating kawaii perceptions of computer voice.
#### Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World
 - **Authors:** Vinu Sankar Sadasivan, Soheil Feizi, Rajiv Mathews, Lun Wang
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06256

 - **Pdf link:** https://arxiv.org/pdf/2507.06256

 - **Abstract**
 This paper investigates the real-world vulnerabilities of audio-based large language models (ALLMs), such as Qwen2-Audio. We first demonstrate that an adversary can craft stealthy audio perturbations to manipulate ALLMs into exhibiting specific targeted behaviors, such as eliciting responses to wake-keywords (e.g., "Hey Qwen"), or triggering harmful behaviors (e.g. "Change my calendar event"). Subsequently, we show that playing adversarial background noise during user interaction with the ALLMs can significantly degrade the response quality. Crucially, our research illustrates the scalability of these attacks to real-world scenarios, impacting other innocent users when these adversarial noises are played through the air. Further, we discuss the transferrability of the attack, and potential defensive measures.
#### IMPACT: Industrial Machine Perception via Acoustic Cognitive Transformer
 - **Authors:** Changheon Han, Yuseop Sim, Hoin Jung, Jiho Lee, Hojun Lee, Yun Seok Kang, Sucheol Woo, Garam Kim, Hyung Wook Park, Martin Byung-Guk Jun
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06481

 - **Pdf link:** https://arxiv.org/pdf/2507.06481

 - **Abstract**
 Acoustic signals from industrial machines offer valuable insights for anomaly detection, predictive maintenance, and operational efficiency enhancement. However, existing task-specific, supervised learning methods often scale poorly and fail to generalize across diverse industrial scenarios, whose acoustic characteristics are distinct from general audio. Furthermore, the scarcity of accessible, large-scale datasets and pretrained models tailored for industrial audio impedes community-driven research and benchmarking. To address these challenges, we introduce DINOS (Diverse INdustrial Operation Sounds), a large-scale open-access dataset. DINOS comprises over 74,149 audio samples (exceeding 1,093 hours) collected from various industrial acoustic scenarios. We also present IMPACT (Industrial Machine Perception via Acoustic Cognitive Transformer), a novel foundation model for industrial machine sound analysis. IMPACT is pretrained on DINOS in a self-supervised manner. By jointly optimizing utterance and frame-level losses, it captures both global semantics and fine-grained temporal structures. This makes its representations suitable for efficient fine-tuning on various industrial downstream tasks with minimal labeled data. Comprehensive benchmarking across 30 distinct downstream tasks (spanning four machine types) demonstrates that IMPACT outperforms existing models on 24 tasks, establishing its superior effectiveness and robustness, while providing a new performance benchmark for future research.
#### STARS: A Unified Framework for Singing Transcription, Alignment, and Refined Style Annotation
 - **Authors:** Wenxiang Guo, Yu Zhang, Changhao Pan, Zhiyuan Zhu, Ruiqi Li, Zhetao Chen, Wenhao Xu, Fei Wu, Zhou Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06670

 - **Pdf link:** https://arxiv.org/pdf/2507.06670

 - **Abstract**
 Recent breakthroughs in singing voice synthesis (SVS) have heightened the demand for high-quality annotated datasets, yet manual annotation remains prohibitively labor-intensive and resource-intensive. Existing automatic singing annotation (ASA) methods, however, primarily tackle isolated aspects of the annotation pipeline. To address this fundamental challenge, we present STARS, which is, to our knowledge, the first unified framework that simultaneously addresses singing transcription, alignment, and refined style annotation. Our framework delivers comprehensive multi-level annotations encompassing: (1) precise phoneme-audio alignment, (2) robust note transcription and temporal localization, (3) expressive vocal technique identification, and (4) global stylistic characterization including emotion and pace. The proposed architecture employs hierarchical acoustic feature processing across frame, word, phoneme, note, and sentence levels. The novel non-autoregressive local acoustic encoders enable structured hierarchical representation learning. Experimental validation confirms the framework's superior performance across multiple evaluation dimensions compared to existing annotation approaches. Furthermore, applications in SVS training demonstrate that models utilizing STARS-annotated data achieve significantly enhanced perceptual naturalness and precise style control. This work not only overcomes critical scalability challenges in the creation of singing datasets but also pioneers new methodologies for controllable singing voice synthesis. Audio samples are available at this https URL.
#### Revealing the Hidden Temporal Structure of HubertSoft Embeddings based on the Russian Phonetic Corpus
 - **Authors:** Anastasia Ananeva, Anton Tomilov, Marina Volkova
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06794

 - **Pdf link:** https://arxiv.org/pdf/2507.06794

 - **Abstract**
 Self-supervised learning (SSL) models such as Wav2Vec 2.0 and HuBERT have shown remarkable success in extracting phonetic information from raw audio without labelled data. While prior work has demonstrated that SSL embeddings encode phonetic features at the frame level, it remains unclear whether these models preserve temporal structure, specifically, whether embeddings at phoneme boundaries reflect the identity and order of adjacent phonemes. This study investigates the extent to which boundary-sensitive embeddings from HubertSoft, a soft-clustering variant of HuBERT, encode phoneme transitions. Using the CORPRES Russian speech corpus, we labelled 20 ms embedding windows with triplets of phonemes corresponding to their start, centre, and end segments. A neural network was trained to predict these positions separately, and multiple evaluation metrics, such as ordered, unordered accuracy and a flexible centre accuracy, were used to assess temporal sensitivity. Results show that embeddings extracted at phoneme boundaries capture both phoneme identity and temporal order, with especially high accuracy at segment boundaries. Confusion patterns further suggest that the model encodes articulatory detail and coarticulatory effects. These findings contribute to our understanding of the internal structure of SSL speech representations and their potential for phonological analysis and fine-grained transcription tasks.
#### Data-Balanced Curriculum Learning for Audio Question Answering
 - **Authors:** Gijs Wijngaard, Elia Formisano, Michele Esposito, Michel Dumontier
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06815

 - **Pdf link:** https://arxiv.org/pdf/2507.06815

 - **Abstract**
 Audio question answering (AQA) requires models to understand acoustic content and perform complex reasoning. Current models struggle with dataset imbalances and unstable training dynamics. This work combines curriculum learning with statistical data balancing to address these challenges. The method labels question difficulty using language models, then trains progressively from easy to hard examples. Statistical filtering removes overrepresented audio categories, and guided decoding constrains outputs to valid multiple-choice formats. Experiments on the DCASE 2025 training set and five additional public datasets show that data curation improves accuracy by 11.7% over baseline models, achieving 64.2% on the DCASE 2025 benchmark.
#### Physics-Informed Direction-Aware Neural Acoustic Fields
 - **Authors:** Yoshiki Masuyama, François G. Germain, Gordon Wichern, Christopher Ick, Jonathan Le Roux
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.06826

 - **Pdf link:** https://arxiv.org/pdf/2507.06826

 - **Abstract**
 This paper presents a physics-informed neural network (PINN) for modeling first-order Ambisonic (FOA) room impulse responses (RIRs). PINNs have demonstrated promising performance in sound field interpolation by combining the powerful modeling capability of neural networks and the physical principles of sound propagation. In room acoustics, PINNs have typically been trained to represent the sound pressure measured by omnidirectional microphones where the wave equation or its frequency-domain counterpart, i.e., the Helmholtz equation, is leveraged. Meanwhile, FOA RIRs additionally provide spatial characteristics and are useful for immersive audio generation with a wide range of applications. In this paper, we extend the PINN framework to model FOA RIRs. We derive two physics-informed priors for FOA RIRs based on the correspondence between the particle velocity and the (X, Y, Z)-channels of FOA. These priors associate the predicted W-channel and other channels through their partial derivatives and impose the physically feasible relationship on the four channels. Our experiments confirm the effectiveness of the proposed method compared with a neural network without the physics-informed prior.
#### Advances in Intelligent Hearing Aids: Deep Learning Approaches to Selective Noise Cancellation
 - **Authors:** Haris Khan, Shumaila Asif, Hassan Nasir
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.07043

 - **Pdf link:** https://arxiv.org/pdf/2507.07043

 - **Abstract**
 The integration of artificial intelligence into hearing assistance marks a paradigm shift from traditional amplification-based systems to intelligent, context-aware audio processing. This systematic literature review evaluates advances in AI-driven selective noise cancellation (SNC) for hearing aids, highlighting technological evolution, implementation challenges, and future research directions. We synthesize findings across deep learning architectures, hardware deployment strategies, clinical validation studies, and user-centric design. The review traces progress from early machine learning models to state-of-the-art deep networks, including Convolutional Recurrent Networks for real-time inference and Transformer-based architectures for high-accuracy separation. Key findings include significant gains over traditional methods, with recent models achieving up to 18.3 dB SI-SDR improvement on noisy-reverberant benchmarks, alongside sub-10 ms real-time implementations and promising clinical outcomes. Yet, challenges remain in bridging lab-grade models with real-world deployment - particularly around power constraints, environmental variability, and personalization. Identified research gaps include hardware-software co-design, standardized evaluation protocols, and regulatory considerations for AI-enhanced hearing devices. Future work must prioritize lightweight models, continual learning, contextual-based classification and clinical translation to realize transformative hearing solutions for millions globally.
#### A Novel Hybrid Deep Learning Technique for Speech Emotion Detection using Feature Engineering
 - **Authors:** Shahana Yasmin Chowdhury, Bithi Banik, Md Tamjidul Hoque, Shreya Banerjee
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07046

 - **Pdf link:** https://arxiv.org/pdf/2507.07046

 - **Abstract**
 Nowadays, speech emotion recognition (SER) plays a vital role in the field of human-computer interaction (HCI) and the evolution of artificial intelligence (AI). Our proposed DCRF-BiLSTM model is used to recognize seven emotions: neutral, happy, sad, angry, fear, disgust, and surprise, which are trained on five datasets: RAVDESS (R), TESS (T), SAVEE (S), EmoDB (E), and Crema-D (C). The model achieves high accuracy on individual datasets, including 97.83% on RAVDESS, 97.02% on SAVEE, 95.10% for CREMA-D, and a perfect 100% on both TESS and EMO-DB. For the combined (R+T+S) datasets, it achieves 98.82% accuracy, outperforming previously reported results. To our knowledge, no existing study has evaluated a single SER model across all five benchmark datasets (i.e., R+T+S+C+E) simultaneously. In our work, we introduce this comprehensive combination and achieve a remarkable overall accuracy of 93.76%. These results confirm the robustness and generalizability of our DCRF-BiLSTM framework across diverse datasets.
#### Comparative Analysis of CNN and Transformer Architectures with Heart Cycle Normalization for Automated Phonocardiogram Classification
 - **Authors:** Martin Sondermann, Pinar Bisgin, Niklas Tschorn, Anja Burmann, Christoph M. Friedrich
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07058

 - **Pdf link:** https://arxiv.org/pdf/2507.07058

 - **Abstract**
 The automated classification of phonocardiogram (PCG) recordings represents a substantial advancement in cardiovascular diagnostics. This paper presents a systematic comparison of four distinct models for heart murmur detection: two specialized convolutional neural networks (CNNs) and two zero-shot universal audio transformers (BEATs), evaluated using fixed-length and heart cycle normalization approaches. Utilizing the PhysioNet2022 dataset, a custom heart cycle normalization method tailored to individual cardiac rhythms is introduced. The findings indicate the following AUROC values: the CNN model with fixed-length windowing achieves 79.5%, the CNN model with heart cycle normalization scores 75.4%, the BEATs transformer with fixed-length windowing achieves 65.7%, and the BEATs transformer with heart cycle normalization results in 70.1%. The findings indicate that physiological signal constraints, especially those introduced by different normalization strategies, have a substantial impact on model performance. The research provides evidence-based guidelines for architecture selection in clinical settings, emphasizing the need for a balance between accuracy and computational efficiency. Although specialized CNNs demonstrate superior performance overall, the zero-shot transformer models may offer promising efficiency advantages during development, such as faster training and evaluation cycles, despite their lower classification accuracy. These findings highlight the potential of automated classification systems to enhance cardiac diagnostics and improve patient care.
#### Latent Acoustic Mapping for Direction of Arrival Estimation: A Self-Supervised Approach
 - **Authors:** Adrian S. Roman, Iran R. Roman, Juan P. Bello
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07066

 - **Pdf link:** https://arxiv.org/pdf/2507.07066

 - **Abstract**
 Acoustic mapping techniques have long been used in spatial audio processing for direction of arrival estimation (DoAE). Traditional beamforming methods for acoustic mapping, while interpretable, often rely on iterative solvers that can be computationally intensive and sensitive to acoustic variability. On the other hand, recent supervised deep learning approaches offer feedforward speed and robustness but require large labeled datasets and lack interpretability. Despite their strengths, both methods struggle to consistently generalize across diverse acoustic setups and array configurations, limiting their broader applicability. We introduce the Latent Acoustic Mapping (LAM) model, a self-supervised framework that bridges the interpretability of traditional methods with the adaptability and efficiency of deep learning methods. LAM generates high-resolution acoustic maps, adapts to varying acoustic conditions, and operates efficiently across different microphone arrays. We assess its robustness on DoAE using the LOCATA and STARSS benchmarks. LAM achieves comparable or superior localization performance to existing supervised methods. Additionally, we show that LAM's acoustic maps can serve as effective features for supervised models, further enhancing DoAE accuracy and underscoring its potential to advance adaptive, high-performance sound localization systems.


by Zyzzyva0381 (Windy). 


2025-07-10
