# Showing new listings for Thursday, 22 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 24papers 
#### QUADS: QUAntized Distillation Framework for Efficient Speech Language Understanding
 - **Authors:** Subrata Biswas, Mohammad Nur Hossain Khan, Bashima Islam
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.14723

 - **Pdf link:** https://arxiv.org/pdf/2505.14723

 - **Abstract**
 Spoken Language Understanding (SLU) systems must balance performance and efficiency, particularly in resource-constrained environments. Existing methods apply distillation and quantization separately, leading to suboptimal compression as distillation ignores quantization constraints. We propose QUADS, a unified framework that optimizes both through multi-stage training with a pre-tuned model, enhancing adaptability to low-bit regimes while maintaining accuracy. QUADS achieves 71.13\% accuracy on SLURP and 99.20\% on FSC, with only minor degradations of up to 5.56\% compared to state-of-the-art models. Additionally, it reduces computational complexity by 60--73$\times$ (GMACs) and model size by 83--700$\times$, demonstrating strong robustness under extreme quantization. These results establish QUADS as a highly efficient solution for real-world, resource-constrained SLU applications.
#### TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis
 - **Authors:** Yu Zhang, Wenxiang Guo, Changhao Pan, Dongyu Yao, Zhiyuan Zhu, Ziyue Jiang, Yuhan Wang, Tao Jin, Zhou Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.14910

 - **Pdf link:** https://arxiv.org/pdf/2505.14910

 - **Abstract**
 Customizable multilingual zero-shot singing voice synthesis (SVS) has various potential applications in music composition and short video dubbing. However, existing SVS models overly depend on phoneme and note boundary annotations, limiting their robustness in zero-shot scenarios and producing poor transitions between phonemes and notes. Moreover, they also lack effective multi-level style control via diverse prompts. To overcome these challenges, we introduce TCSinger 2, a multi-task multilingual zero-shot SVS model with style transfer and style control based on various prompts. TCSinger 2 mainly includes three key modules: 1) Blurred Boundary Content (BBC) Encoder, predicts duration, extends content embedding, and applies masking to the boundaries to enable smooth transitions. 2) Custom Audio Encoder, uses contrastive learning to extract aligned representations from singing, speech, and textual prompts. 3) Flow-based Custom Transformer, leverages Cus-MOE, with F0 supervision, enhancing both the synthesis quality and style modeling of the generated singing voice. Experimental results show that TCSinger 2 outperforms baseline models in both subjective and objective metrics across multiple related tasks.
#### EASY: Emotion-aware Speaker Anonymization via Factorized Distillation
 - **Authors:** Jixun Yao, Hexin Liu, Eng Siong Chng, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.15004

 - **Pdf link:** https://arxiv.org/pdf/2505.15004

 - **Abstract**
 Emotion plays a significant role in speech interaction, conveyed through tone, pitch, and rhythm, enabling the expression of feelings and intentions beyond words to create a more personalized experience. However, most existing speaker anonymization systems employ parallel disentanglement methods, which only separate speech into linguistic content and speaker identity, often neglecting the preservation of the original emotional state. In this study, we introduce EASY, an emotion-aware speaker anonymization framework. EASY employs a novel sequential disentanglement process to disentangle speaker identity, linguistic content, and emotional representation, modeling each speech attribute in distinct subspaces through a factorized distillation approach. By independently constraining speaker identity and emotional representation, EASY minimizes information leakage, enhancing privacy protection while preserving original linguistic content and emotional state. Experimental results on the VoicePrivacy Challenge official datasets demonstrate that our proposed approach outperforms all baseline systems, effectively protecting speaker privacy while maintaining linguistic content and emotional state.
#### Towards Pre-training an Effective Respiratory Audio Foundation Model
 - **Authors:** Daisuke Niizumi, Daiki Takeuchi, Masahiro Yasuda, Binh Thien Nguyen, Yasunori Ohishi, Noboru Harada
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.15307

 - **Pdf link:** https://arxiv.org/pdf/2505.15307

 - **Abstract**
 Recent advancements in foundation models have sparked interest in respiratory audio foundation models. However, the effectiveness of applying conventional pre-training schemes to datasets that are small-sized and lack diversity has not been sufficiently verified. This study aims to explore better pre-training practices for respiratory sounds by comparing numerous pre-trained audio models. Our investigation reveals that models pre-trained on AudioSet, a general audio dataset, are more effective than the models specifically pre-trained on respiratory sounds. Moreover, combining AudioSet and respiratory sound datasets for further pre-training enhances performance, and preserving the frequency-wise information when aggregating features is vital. Along with more insights found in the experiments, we establish a new state-of-the-art for the OPERA benchmark, contributing to advancing respiratory audio foundation models. Our code is available online at this https URL.
#### Analysis of ABC Frontend Audio Systems for the NIST-SRE24
 - **Authors:** Sara Barahona, Anna Silnova, Ladislav Mošner, Junyi Peng, Oldřich Plchot, Johan Rohdin, Lin Zhang, Jiangyu Han, Petr Palka, Federico Landini, Lukáš Burget, Themos Stafylakis, Sandro Cumani, Dominik Boboš, Miroslav Hlavaček, Martin Kodovsky, Tomáš Pavlíček
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15320

 - **Pdf link:** https://arxiv.org/pdf/2505.15320

 - **Abstract**
 We present a comprehensive analysis of the embedding extractors (frontends) developed by the ABC team for the audio track of NIST SRE 2024. We follow the two scenarios imposed by NIST: using only a provided set of telephone recordings for training (fixed) or adding publicly available data (open condition). Under these constraints, we develop the best possible speaker embedding extractors for the pre-dominant conversational telephone speech (CTS) domain. We explored architectures based on ResNet with different pooling mechanisms, recently introduced ReDimNet architecture, as well as a system based on the XLS-R model, which represents the family of large pre-trained self-supervised models. In open condition, we train on VoxBlink2 dataset, containing 110 thousand speakers across multiple languages. We observed a good performance and robustness of VoxBlink-trained models, and our experiments show practical recipes for developing state-of-the-art frontends for speaker recognition.
#### On the Relevance of Clinical Assessment Tasks for the Automatic Detection of Parkinson's Disease Medication State from Speech
 - **Authors:** David Gimeno-Gómez, Rubén Solera-Ureña, Anna Pompili, Carlos-D. Martínez-Hinarejos, Rita Cardoso, Isabel Guimarães, Joaquim Ferreira, Alberto Abad
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15378

 - **Pdf link:** https://arxiv.org/pdf/2505.15378

 - **Abstract**
 The automatic identification of medication states of Parkinson's disease (PD) patients can assist clinicians in monitoring and scheduling personalized treatments, as well as studying the effects of medication in alleviating the motor symptoms that characterize the disease. This paper explores speech as a non-invasive and accessible biomarker for identifying PD medication states, introducing a novel approach that addresses this task from a speaker-independent perspective. While traditional machine learning models achieve competitive results, self-supervised speech representations prove essential for optimal performance, significantly surpassing knowledge-based acoustic descriptors. Experiments across diverse speech assessment tasks highlight the relevance of prosody and continuous speech in distinguishing medication states, reaching an F1-score of 88.2%. These findings may streamline clinicians' work and reduce patient effort in voice recordings.
#### Segmentation-Variant Codebooks for Preservation of Paralinguistic and Prosodic Information
 - **Authors:** Nicholas Sanders, Yuanchao Li, Korin Richmond, Simon King
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.15667

 - **Pdf link:** https://arxiv.org/pdf/2505.15667

 - **Abstract**
 Quantization in SSL speech models (e.g., HuBERT) improves compression and performance in tasks like language modeling, resynthesis, and text-to-speech but often discards prosodic and paralinguistic information (e.g., emotion, prominence). While increasing codebook size mitigates some loss, it inefficiently raises bitrates. We propose Segmentation-Variant Codebooks (SVCs), which quantize speech at distinct linguistic units (frame, phone, word, utterance), factorizing it into multiple streams of segment-specific discrete features. Our results show that SVCs are significantly more effective at preserving prosodic and paralinguistic information across probing tasks. Additionally, we find that pooling before rather than after discretization better retains segment-level information. Resynthesis experiments further confirm improved style realization and slightly improved quality while preserving intelligibility.
#### ToxicTone: A Mandarin Audio Dataset Annotated for Toxicity and Toxic Utterance Tonality
 - **Authors:** Yu-Xiang Luo, Yi-Cheng Lin, Ming-To Chuang, Jia-Hung Chen, I-Ning Tsai, Pei Xing Kiew, Yueh-Hsuan Huang, Chien-Feng Liu, Yu-Chen Chen, Bo-Han Feng, Wenze Ren, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2505.15773

 - **Pdf link:** https://arxiv.org/pdf/2505.15773

 - **Abstract**
 Despite extensive research on toxic speech detection in text, a critical gap remains in handling spoken Mandarin audio. The lack of annotated datasets that capture the unique prosodic cues and culturally specific expressions in Mandarin leaves spoken toxicity underexplored. To address this, we introduce ToxicTone -- the largest public dataset of its kind -- featuring detailed annotations that distinguish both forms of toxicity (e.g., profanity, bullying) and sources of toxicity (e.g., anger, sarcasm, dismissiveness). Our data, sourced from diverse real-world audio and organized into 13 topical categories, mirrors authentic communication scenarios. We also propose a multimodal detection framework that integrates acoustic, linguistic, and emotional features using state-of-the-art speech and emotion encoders. Extensive experiments show our approach outperforms text-only and baseline models, underscoring the essential role of speech-specific cues in revealing hidden toxic expressions.
#### Replay Attacks Against Audio Deepfake Detection
 - **Authors:** Nicolas Müller, Piotr Kawa, Wei-Herng Choong, Adriana Stan, Aditya Tirumala Bukkapatnam, Karla Pizzi, Alexander Wagner, Philip Sperl
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14862

 - **Pdf link:** https://arxiv.org/pdf/2505.14862

 - **Abstract**
 We show how replay attacks undermine audio deepfake detection: By playing and re-recording deepfake audio through various speakers and microphones, we make spoofed samples appear authentic to the detection model. To study this phenomenon in more detail, we introduce ReplayDF, a dataset of recordings derived from M-AILABS and MLAAD, featuring 109 speaker-microphone combinations across six languages and four TTS models. It includes diverse acoustic conditions, some highly challenging for detection. Our analysis of six open-source detection models across five datasets reveals significant vulnerability, with the top-performing W2V2-AASIST model's Equal Error Rate (EER) surging from 4.7% to 18.2%. Even with adaptive Room Impulse Response (RIR) retraining, performance remains compromised with an 11.0% EER. We release ReplayDF for non-commercial research use.
#### Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
 - **Authors:** Chin-Jou Li, Eunjung Yeo, Kwanghee Choi, Paula Andrea Pérez-Toro, Masao Someki, Rohan Kumar Das, Zhengjun Yue, Juan Rafael Orozco-Arroyave, Elmar Nöth, David R. Mortensen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14874

 - **Pdf link:** https://arxiv.org/pdf/2505.14874

 - **Abstract**
 Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
#### In-Context Learning Boosts Speech Recognition via Human-like Adaptation to Speakers and Language Varieties
 - **Authors:** Nathan Roll, Calbert Graham, Yuka Tatsumi, Kim Tien Nguyen, Meghan Sumner, Dan Jurafsky
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14887

 - **Pdf link:** https://arxiv.org/pdf/2505.14887

 - **Abstract**
 Human listeners readily adjust to unfamiliar speakers and language varieties through exposure, but do these adaptation benefits extend to state-of-the-art spoken language models? We introduce a scalable framework that allows for in-context learning (ICL) in Phi-4 Multimodal using interleaved task prompts and audio-text pairs, and find that as few as 12 example utterances (~50 seconds) at inference time reduce word error rates by a relative 19.7% (1.2 pp.) on average across diverse English corpora. These improvements are most pronounced in low-resource varieties, when the context and target speaker match, and when more examples are provided--though scaling our procedure yields diminishing marginal returns to context length. Overall, we find that our novel ICL adaptation scheme (1) reveals a similar performance profile to human listeners, and (2) demonstrates consistent improvements to automatic speech recognition (ASR) robustness across diverse speakers and language backgrounds. While adaptation succeeds broadly, significant gaps remain for certain varieties, revealing where current models still fall short of human flexibility. We release our prompts and code on GitHub.
#### Discrete Audio Representations for Automated Audio Captioning
 - **Authors:** Jingguang Tian, Haoqin Sun, Xinhui Hu, Xinkang Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.14989

 - **Pdf link:** https://arxiv.org/pdf/2505.14989

 - **Abstract**
 Discrete audio representations, termed audio tokens, are broadly categorized into semantic and acoustic tokens, typically generated through unsupervised tokenization of continuous audio representations. However, their applicability to automated audio captioning (AAC) remains underexplored. This paper systematically investigates the viability of audio token-driven models for AAC through comparative analyses of various tokenization methods. Our findings reveal that audio tokenization leads to performance degradation in AAC models compared to those that directly utilize continuous audio representations. To address this issue, we introduce a supervised audio tokenizer trained with an audio tagging objective. Unlike unsupervised tokenizers, which lack explicit semantic understanding, the proposed tokenizer effectively captures audio event information. Experiments conducted on the Clotho dataset demonstrate that the proposed audio tokens outperform conventional audio tokens in the AAC task.
#### SHEET: A Multi-purpose Open-source Speech Human Evaluation Estimation Toolkit
 - **Authors:** Wen-Chin Huang, Erica Cooper, Tomoki Toda
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15061

 - **Pdf link:** https://arxiv.org/pdf/2505.15061

 - **Abstract**
 We introduce SHEET, a multi-purpose open-source toolkit designed to accelerate subjective speech quality assessment (SSQA) research. SHEET stands for the Speech Human Evaluation Estimation Toolkit, which focuses on data-driven deep neural network-based models trained to predict human-labeled quality scores of speech samples. SHEET provides comprehensive training and evaluation scripts, multi-dataset and multi-model support, as well as pre-trained models accessible via Torch Hub and HuggingFace Spaces. To demonstrate its capabilities, we re-evaluated SSL-MOS, a speech self-supervised learning (SSL)-based SSQA model widely used in recent scientific papers, on an extensive list of speech SSL models. Experiments were conducted on two representative SSQA datasets named BVCC and NISQA, and we identified the optimal speech SSL model, whose performance surpassed the original SSL-MOS implementation and was comparable to state-of-the-art methods.
#### Hybrid Audio Detection Using Fine-Tuned Audio Spectrogram Transformers: A Dataset-Driven Evaluation of Mixed AI-Human Speech
 - **Authors:** Kunyang Huang, Bin Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15136

 - **Pdf link:** https://arxiv.org/pdf/2505.15136

 - **Abstract**
 The rapid advancement of artificial intelligence (AI) has enabled sophisticated audio generation and voice cloning technologies, posing significant security risks for applications reliant on voice authentication. While existing datasets and models primarily focus on distinguishing between human and fully synthetic speech, real-world attacks often involve audio that combines both genuine and cloned segments. To address this gap, we construct a novel hybrid audio dataset incorporating human, AI-generated, cloned, and mixed audio samples. We further propose fine-tuned Audio Spectrogram Transformer (AST)-based models tailored for detecting these complex acoustic patterns. Extensive experiments demonstrate that our approach significantly outperforms existing baselines in mixed-audio detection, achieving 97\% classification accuracy. Our findings highlight the importance of hybrid datasets and tailored models in advancing the robustness of speech-based authentication systems.
#### Voice-ENHANCE: Speech Restoration using a Diffusion-based Voice Conversion Framework
 - **Authors:** Kyungguen Byun, Jason Filos, Erik Visser, Sunkuk Moon
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15254

 - **Pdf link:** https://arxiv.org/pdf/2505.15254

 - **Abstract**
 We propose a speech enhancement system that combines speaker-agnostic speech restoration with voice conversion (VC) to obtain a studio-level quality speech signal. While voice conversion models are typically used to change speaker characteristics, they can also serve as a means of speech restoration when the target speaker is the same as the source speaker. However, since VC models are vulnerable to noisy conditions, we have included a generative speech restoration (GSR) model at the front end of our proposed system. The GSR model performs noise suppression and restores speech damage incurred during that process without knowledge about the target speaker. The VC stage then uses guidance from clean speaker embeddings to further restore the output speech. By employing this two-stage approach, we have achieved speech quality objective metric scores comparable to state-of-the-art (SOTA) methods across multiple datasets.
#### Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation
 - **Authors:** Yuhao Zhang, Xiangnan Ma, Kaiqi Kou, Peizhuo Liu, Weiqiao Shan, Benyou Wang, Tong Xiao, Yuxin Huang, Zhengtao Yu, Jingbo Zhu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15333

 - **Pdf link:** https://arxiv.org/pdf/2505.15333

 - **Abstract**
 The success of building textless speech-to-speech translation (S2ST) models has attracted much attention. However, S2ST still faces two main challenges: 1) extracting linguistic features for various speech signals, called cross-modal (CM), and 2) learning alignment of difference languages in long sequences, called cross-lingual (CL). We propose the unit language to overcome the two modeling challenges. The unit language can be considered a text-like representation format, constructed using $n$-gram language modeling. We implement multi-task learning to utilize the unit language in guiding the speech modeling process. Our initial results reveal a conflict when applying source and target unit languages simultaneously. We propose task prompt modeling to mitigate this conflict. We conduct experiments on four languages of the Voxpupil dataset. Our method demonstrates significant improvements over a strong baseline and achieves performance comparable to models trained with text.
#### Decoding Phone Pairs from MEG Signals Across Speech Modalities
 - **Authors:** Xabier de Zuazo, Eva Navas, Ibon Saratxaga, Mathieu Bourguignon, Nicola Molinaro
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15355

 - **Pdf link:** https://arxiv.org/pdf/2505.15355

 - **Abstract**
 Understanding the neural mechanisms underlying speech production is essential for both advancing cognitive neuroscience theory and developing practical communication technologies. In this study, we investigated magnetoencephalography signals to decode phones from brain activity during speech production and perception (passive listening and voice playback) tasks. Using a dataset comprising 17 participants, we performed pairwise phone classification, extending our analysis to 15 phonetic pairs. Multiple machine learning approaches, including regularized linear models and neural network architectures, were compared to determine their effectiveness in decoding phonetic information. Our results demonstrate significantly higher decoding accuracy during speech production (76.6%) compared to passive listening and playback modalities (~51%), emphasizing the richer neural information available during overt speech. Among the models, the Elastic Net classifier consistently outperformed more complex neural networks, highlighting the effectiveness of traditional regularization techniques when applied to limited and high-dimensional MEG datasets. Besides, analysis of specific brain frequency bands revealed that low-frequency oscillations, particularly Delta (0.2-3 Hz) and Theta (4-7 Hz), contributed the most substantially to decoding accuracy, suggesting that these bands encode critical speech production-related neural processes. Despite using advanced denoising methods, it remains unclear whether decoding solely reflects neural activity or if residual muscular or movement artifacts also contributed, indicating the need for further methodological refinement. Overall, our findings underline the critical importance of examining overt speech production paradigms, which, despite their complexity, offer opportunities to improve brain-computer interfaces to help individuals with severe speech impairments.
#### Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding
 - **Authors:** Zijian Lin, Yang Zhang, Yougen Yuan, Yuming Yan, Jinjiang Liu, Zhiyong Wu, Pengfei Hu, Qun Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15380

 - **Pdf link:** https://arxiv.org/pdf/2505.15380

 - **Abstract**
 Modern autoregressive speech synthesis models leveraging language models have demonstrated remarkable performance. However, the sequential nature of next token prediction in these models leads to significant latency, hindering their deployment in scenarios where inference speed is critical. In this work, we propose Speech Speculative Decoding (SSD), a novel framework for autoregressive speech synthesis acceleration. Specifically, our method employs a lightweight draft model to generate candidate token sequences, which are subsequently verified in parallel by the target model using the proposed SSD framework. Experimental results demonstrate that SSD achieves a significant speedup of 1.4x compared with conventional autoregressive decoding, while maintaining high fidelity and naturalness. Subjective evaluations further validate the effectiveness of SSD in preserving the perceptual quality of the target model while accelerating inference.
#### Prosody-Adaptable Audio Codecs for Zero-Shot Voice Conversion via In-Context Learning
 - **Authors:** Junchuan Zhao, Xintong Wang, Ye Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15402

 - **Pdf link:** https://arxiv.org/pdf/2505.15402

 - **Abstract**
 Recent advances in discrete audio codecs have significantly improved speech representation modeling, while codec language models have enabled in-context learning for zero-shot speech synthesis. Inspired by this, we propose a voice conversion (VC) model within the VALLE-X framework, leveraging its strong in-context learning capabilities for speaker adaptation. To enhance prosody control, we introduce a prosody-aware audio codec encoder (PACE) module, which isolates and refines prosody from other sources, improving expressiveness and control. By integrating PACE into our VC model, we achieve greater flexibility in prosody manipulation while preserving speaker timbre. Experimental evaluation results demonstrate that our approach outperforms baseline VC systems in prosody preservation, timbre consistency, and overall naturalness, surpassing baseline VC systems.
#### Audio Jailbreak: An Open Comprehensive Benchmark for Jailbreaking Large Audio-Language Models
 - **Authors:** Zirui Song, Qian Jiang, Mingxuan Cui, Mingzhe Li, Lang Gao, Zeyu Zhang, Zixiang Xu, Yanbo Wang, Chenxi Wang, Guangxian Ouyang, Zhenhao Chen, Xiuying Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15406

 - **Pdf link:** https://arxiv.org/pdf/2505.15406

 - **Abstract**
 The rise of Large Audio Language Models (LAMs) brings both potential and risks, as their audio outputs may contain harmful or unethical content. However, current research lacks a systematic, quantitative evaluation of LAM safety especially against jailbreak attacks, which are challenging due to the temporal and semantic nature of speech. To bridge this gap, we introduce AJailBench, the first benchmark specifically designed to evaluate jailbreak vulnerabilities in LAMs. We begin by constructing AJailBench-Base, a dataset of 1,495 adversarial audio prompts spanning 10 policy-violating categories, converted from textual jailbreak attacks using realistic text to speech synthesis. Using this dataset, we evaluate several state-of-the-art LAMs and reveal that none exhibit consistent robustness across attacks. To further strengthen jailbreak testing and simulate more realistic attack conditions, we propose a method to generate dynamic adversarial variants. Our Audio Perturbation Toolkit (APT) applies targeted distortions across time, frequency, and amplitude domains. To preserve the original jailbreak intent, we enforce a semantic consistency constraint and employ Bayesian optimization to efficiently search for perturbations that are both subtle and highly effective. This results in AJailBench-APT, an extended dataset of optimized adversarial audio samples. Our findings demonstrate that even small, semantically preserved perturbations can significantly reduce the safety performance of leading LAMs, underscoring the need for more robust and semantically aware defense mechanisms.
#### Word Level Timestamp Generation for Automatic Speech Recognition and Translation
 - **Authors:** Ke Hu, Krishna Puvvada, Elena Rastorgueva, Zhehuai Chen, He Huang, Shuoyang Ding, Kunal Dhawan, Hainan Xu, Jagadeesh Balam, Boris Ginsburg
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15646

 - **Pdf link:** https://arxiv.org/pdf/2505.15646

 - **Abstract**
 We introduce a data-driven approach for enabling word-level timestamp prediction in the Canary model. Accurate timestamp information is crucial for a variety of downstream tasks such as speech content retrieval and timed subtitles. While traditional hybrid systems and end-to-end (E2E) models may employ external modules for timestamp prediction, our approach eliminates the need for separate alignment mechanisms. By leveraging the NeMo Forced Aligner (NFA) as a teacher model, we generate word-level timestamps and train the Canary model to predict timestamps directly. We introduce a new <|timestamp|> token, enabling the Canary model to predict start and end timestamps for each word. Our method demonstrates precision and recall rates between 80% and 90%, with timestamp prediction errors ranging from 20 to 120 ms across four languages, with minimal WER degradation. Additionally, we extend our system to automatic speech translation (AST) tasks, achieving timestamp prediction errors around 200 milliseconds.
#### Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model
 - **Authors:** Ke Hu, Ehsan Hosseini-Asl, Chen Chen, Edresson Casanova, Subhankar Ghosh, Piotr Żelasko, Zhehuai Chen, Jason Li, Jagadeesh Balam, Boris Ginsburg
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15670

 - **Pdf link:** https://arxiv.org/pdf/2505.15670

 - **Abstract**
 Spoken dialogue is an intuitive form of human-computer interaction, yet current speech language models often remain constrained to turn-based exchanges, lacking real-time adaptability such as user barge-in. We propose a novel duplex speech to speech (S2S) architecture featuring continuous user inputs and codec agent outputs with channel fusion that directly models simultaneous user and agent streams. Using a pretrained streaming encoder for user input enables the first duplex S2S model without requiring speech pretrain. Separate architectures for agent and user modeling facilitate codec fine-tuning for better agent voices and halve the bitrate (0.6 kbps) compared to previous works. Experimental results show that the proposed model outperforms previous duplex models in reasoning, turn-taking, and barge-in abilities. The model requires significantly less speech data, as speech pretrain is skipped, which markedly simplifies the process of building a duplex S2S model from any LLMs. Finally, it is the first openly available duplex S2S model with training and inference code to foster reproducibility.
#### "Alexa, can you forget me?" Machine Unlearning Benchmark in Spoken Language Understanding
 - **Authors:** Alkis Koudounas, Claudio Savelli, Flavio Giobergia, Elena Baralis
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15700

 - **Pdf link:** https://arxiv.org/pdf/2505.15700

 - **Abstract**
 Machine unlearning, the process of efficiently removing specific information from machine learning models, is a growing area of interest for responsible AI. However, few studies have explored the effectiveness of unlearning methods on complex tasks, particularly speech-related ones. This paper introduces UnSLU-BENCH, the first benchmark for machine unlearning in spoken language understanding (SLU), focusing on four datasets spanning four languages. We address the unlearning of data from specific speakers as a way to evaluate the quality of potential "right to be forgotten" requests. We assess eight unlearning techniques and propose a novel metric to simultaneously better capture their efficacy, utility, and efficiency. UnSLU-BENCH sets a foundation for unlearning in SLU and reveals significant differences in the effectiveness and computational feasibility of various techniques.
#### MIKU-PAL: An Automated and Standardized Multi-Modal Method for Speech Paralinguistic and Affect Labeling
 - **Authors:** Cheng Yifan, Zhang Ruoyi, Shi Jiatong
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.15772

 - **Pdf link:** https://arxiv.org/pdf/2505.15772

 - **Abstract**
 Acquiring large-scale emotional speech data with strong consistency remains a challenge for speech synthesis. This paper presents MIKU-PAL, a fully automated multimodal pipeline for extracting high-consistency emotional speech from unlabeled video data. Leveraging face detection and tracking algorithms, we developed an automatic emotion analysis system using a multimodal large language model (MLLM). Our results demonstrate that MIKU-PAL can achieve human-level accuracy (68.5% on MELD) and superior consistency (0.93 Fleiss kappa score) while being much cheaper and faster than human annotation. With the high-quality, flexible, and consistent annotation from MIKU-PAL, we can annotate fine-grained speech emotion categories of up to 26 types, validated by human annotators with 83% rationality ratings. Based on our proposed system, we further released a fine-grained emotional speech dataset MIKU-EmoBench(131.2 hours) as a new benchmark for emotional text-to-speech and visual voice cloning.


by Zyzzyva0381 (Windy). 


2025-05-22
