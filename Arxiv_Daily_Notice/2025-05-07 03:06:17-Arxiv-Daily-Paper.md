# Showing new listings for Wednesday, 7 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 9papers 
#### The Search for Squawk: Agile Modeling in Bioacoustics
 - **Authors:** Vincent Dumoulin, Otilia Stretcu, Jenny Hamer, Lauren Harrell, Rob Laber, Hugo Larochelle, Bart van Merriënboer, Amanda Navine, Patrick Hart, Ben Williams, Timothy A.C. Lamont, Tries B. Rasak, Mars Coral Restoration Team, Sheryn Brodie, Brendan Doohan, Phil Eichinski, Paul Roe, Lin Schwarzkopf, Tom Denton
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.03071

 - **Pdf link:** https://arxiv.org/pdf/2505.03071

 - **Abstract**
 Passive acoustic monitoring (PAM) has shown great promise in helping ecologists understand the health of animal populations and ecosystems. However, extracting insights from millions of hours of audio recordings requires the development of specialized recognizers. This is typically a challenging task, necessitating large amounts of training data and machine learning expertise. In this work, we introduce a general, scalable and data-efficient system for developing recognizers for novel bioacoustic problems in under an hour. Our system consists of several key components that tackle problems in previous bioacoustic workflows: 1) highly generalizable acoustic embeddings pre-trained for birdsong classification minimize data hunger; 2) indexed audio search allows the efficient creation of classifier training datasets, and 3) precomputation of embeddings enables an efficient active learning loop, improving classifier quality iteratively with minimal wait time. Ecologists employed our system in three novel case studies: analyzing coral reef health through unidentified sounds; identifying juvenile Hawaiian bird calls to quantify breeding success and improve endangered species monitoring; and Christmas Island bird occupancy modeling. We augment the case studies with simulated experiments which explore the range of design decisions in a structured way and help establish best practices. Altogether these experiments showcase our system's scalability, efficiency, and generalizability, enabling scientists to quickly address new bioacoustic challenges.
#### Fairness of Automatic Speech Recognition in Cleft Lip and Palate Speech
 - **Authors:** Susmita Bhattacharjee, Jagabandhu Mishra, H.S. Shekhawat, S. R. Mahadeva Prasanna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.03697

 - **Pdf link:** https://arxiv.org/pdf/2505.03697

 - **Abstract**
 Speech produced by individuals with cleft lip and palate (CLP) is often highly nasalized and breathy due to structural anomalies, causing shifts in formant structure that affect automatic speech recognition (ASR) performance and fairness. This study hypothesizes that publicly available ASR systems exhibit reduced fairness for CLP speech and confirms this through experiments. Despite formant disruptions, mild and moderate CLP speech retains some spectro-temporal alignment with normal speech, motivating augmentation strategies to enhance fairness. The study systematically explores augmenting CLP speech with normal speech across severity levels and evaluates its impact on ASR fairness. Three ASR models-GMM-HMM, Whisper, and XLS-R-were tested on AIISH and NMCPC datasets. Results indicate that training with normal speech and testing on mixed data improves word error rate (WER). Notably, WER decreased from $22.64\%$ to $18.76\%$ (GMM-HMM, AIISH) and $28.45\%$ to $18.89\%$ (Whisper, NMCPC). The superior performance of GMM-HMM on AIISH may be due to its suitability for Kannada children's speech, a challenge for foundation models like XLS-R and Whisper. To assess fairness, a fairness score was introduced, revealing improvements of $17.89\%$ (AIISH) and $47.50\%$ (NMCPC) with augmentation.
#### BLAB: Brutally Long Audio Bench
 - **Authors:** Orevaoghene Ahia, Martijn Bartelds, Kabir Ahuja, Hila Gonen, Valentin Hofmann, Siddhant Arora, Shuyue Stella Li, Vishal Puttagunta, Mofetoluwa Adeyemi, Charishma Buchireddy, Ben Walls, Noah Bennett, Shinji Watanabe, Noah A. Smith, Yulia Tsvetkov, Sachin Kumar
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.03054

 - **Pdf link:** https://arxiv.org/pdf/2505.03054

 - **Abstract**
 Developing large audio language models (LMs) capable of understanding diverse spoken interactions is essential for accommodating the multimodal nature of human communication and can increase the accessibility of language technologies across different user populations. Recent work on audio LMs has primarily evaluated their performance on short audio segments, typically under 30 seconds, with limited exploration of long-form conversational speech segments that more closely reflect natural user interactions with these models. We introduce Brutally Long Audio Bench (BLAB), a challenging long-form audio benchmark that evaluates audio LMs on localization, duration estimation, emotion, and counting tasks using audio segments averaging 51 minutes in length. BLAB consists of 833+ hours of diverse, full-length audio clips, each paired with human-annotated, text-based natural language questions and answers. Our audio data were collected from permissively licensed sources and underwent a human-assisted filtering process to ensure task compliance. We evaluate six open-source and proprietary audio LMs on BLAB and find that all of them, including advanced models such as Gemini 2.0 Pro and GPT-4o, struggle with the tasks in BLAB. Our comprehensive analysis reveals key insights into the trade-offs between task difficulty and audio duration. In general, we find that audio LMs struggle with long-form speech, with performance declining as duration increases. They perform poorly on localization, temporal reasoning, counting, and struggle to understand non-phonemic information, relying more on prompts than audio content. BLAB serves as a challenging evaluation framework to develop audio LMs with robust long-form audio understanding capabilities.
#### CoGenAV: Versatile Audio-Visual Representation Learning via Contrastive-Generative Synchronization
 - **Authors:** Detao Bai, Zhiheng Ma, Xihan Wei, Liefeng Bo
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.03186

 - **Pdf link:** https://arxiv.org/pdf/2505.03186

 - **Abstract**
 The inherent synchronization between a speaker's lip movements, voice, and the underlying linguistic content offers a rich source of information for improving speech processing tasks, especially in challenging conditions where traditional audio-only systems falter. We introduce CoGenAV, a powerful and data-efficient model designed to learn versatile audio-visual representations applicable across a wide range of speech and audio-visual tasks. CoGenAV is trained by optimizing a dual objective derived from natural audio-visual synchrony, contrastive feature alignment and generative text prediction, using only 223 hours of labeled data from the LRS2 dataset. This contrastive-generative synchronization strategy effectively captures fundamental cross-modal correlations. We showcase the effectiveness and versatility of the learned CoGenAV representations on multiple benchmarks. When utilized for Audio-Visual Speech Recognition (AVSR) on LRS2, these representations contribute to achieving a state-of-the-art Word Error Rate (WER) of 1.27. They also enable strong performance in Visual Speech Recognition (VSR) with a WER of 22.0 on LRS2, and significantly improve performance in noisy environments by over 70%. Furthermore, CoGenAV representations benefit speech reconstruction tasks, boosting performance in Speech Enhancement and Separation, and achieve competitive results in audio-visual synchronization tasks like Active Speaker Detection (ASD). Our model will be open-sourced to facilitate further development and collaboration within both academia and industry.
#### A study on audio synchronous steganography detection and distributed guide inference model based on sliding spectral features and intelligent inference drive
 - **Authors:** Wei Meng
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.03193

 - **Pdf link:** https://arxiv.org/pdf/2505.03193

 - **Abstract**
 With the rise of short video platforms in global communication, embedding steganographic data in audio synchronization streams has emerged as a new covert communication method. To address the limitations of traditional techniques in detecting synchronized steganography, this paper proposes a detection and distributed guidance reconstruction model based on short video "Yupan" samples released by China's South Sea Fleet on TikTok. The method integrates sliding spectrum feature extraction and intelligent inference mechanisms. A 25 ms sliding window with short-time Fourier transform (STFT) is used to extract the main frequency trajectory and construct the synchronization frame detection model (M1), identifying a frame flag "FFFFFFFFFFFFFFFFFF80". The subsequent 32-byte payload is decoded by a structured model (M2) to infer distributed guidance commands. Analysis reveals a low-entropy, repetitive byte sequence in the 36 to 45 second audio segment with highly concentrated spectral energy, confirming the presence of synchronization frames. Although plaintext semantics are not restored, the consistency in command field layout suggests features of military communication protocols. The multi-segment splicing model further shows cross-video embedding and centralized decoding capabilities. The proposed framework validates the effectiveness of sliding spectral features for synchronized steganography detection and builds an extensible inference model for covert communication analysis and tactical guidance simulation on open platforms.
#### MGFF-TDNN: A Multi-Granularity Feature Fusion TDNN Model with Depth-Wise Separable Module for Speaker Verification
 - **Authors:** Ya Li, Bin Zhou, Bo Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.03228

 - **Pdf link:** https://arxiv.org/pdf/2505.03228

 - **Abstract**
 In speaker verification, traditional models often emphasize modeling long-term contextual features to capture global speaker characteristics. However, this approach can neglect fine-grained voiceprint information, which contains highly discriminative features essential for robust speaker embeddings. This paper introduces a novel model architecture, termed MGFF-TDNN, based on multi-granularity feature fusion. The MGFF-TDNN leverages a two-dimensional depth-wise separable convolution module, enhanced with local feature modeling, as a front-end feature extractor to effectively capture time-frequency domain features. To achieve comprehensive multi-granularity feature fusion, we propose the M-TDNN structure, which integrates global contextual modeling with fine-grained feature extraction by combining time-delay neural networks and phoneme-level feature pooling. Experiments on the VoxCeleb dataset demonstrate that the MGFF-TDNN achieves outstanding performance in speaker verification while remaining efficient in terms of parameters and computational resources.
#### SonicRAG : High Fidelity Sound Effects Synthesis Based on Retrival Augmented Generation
 - **Authors:** Yu-Ren Guo, Wen-Kai Tai
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.03244

 - **Pdf link:** https://arxiv.org/pdf/2505.03244

 - **Abstract**
 Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing (NLP) and multimodal learning, with successful applications in text generation and speech synthesis, enabling a deeper understanding and generation of multimodal content. In the field of sound effects (SFX) generation, LLMs have been leveraged to orchestrate multiple models for audio synthesis. However, due to the scarcity of annotated datasets, and the complexity of temproal modeling. current SFX generation techniques still fall short in achieving high-fidelity audio. To address these limitations, this paper introduces a novel framework that integrates LLMs with existing sound effect databases, allowing for the retrieval, recombination, and synthesis of audio based on user requirements. By leveraging this approach, we enhance the diversity and quality of generated sound effects while eliminating the need for additional recording costs, offering a flexible and efficient solution for sound design and application.
#### SepALM: Audio Language Models Are Error Correctors for Robust Speech Separation
 - **Authors:** Zhaoxi Mu, Xinyu Yang, Gang Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.03273

 - **Pdf link:** https://arxiv.org/pdf/2505.03273

 - **Abstract**
 While contemporary speech separation technologies adeptly process lengthy mixed audio waveforms, they are frequently challenged by the intricacies of real-world environments, including noisy and reverberant settings, which can result in artifacts or distortions in the separated speech. To overcome these limitations, we introduce SepALM, a pioneering approach that employs audio language models (ALMs) to rectify and re-synthesize speech within the text domain following preliminary separation. SepALM comprises four core components: a separator, a corrector, a synthesizer, and an aligner. By integrating an ALM-based end-to-end error correction mechanism, we mitigate the risk of error accumulation and circumvent the optimization hurdles typically encountered in conventional methods that amalgamate automatic speech recognition (ASR) with large language models (LLMs). Additionally, we have developed Chain-of-Thought (CoT) prompting and knowledge distillation techniques to facilitate the reasoning and training processes of the ALM. Our experiments substantiate that SepALM not only elevates the precision of speech separation but also markedly bolsters adaptability in novel acoustic environments.
#### Knowledge Distillation for Speech Denoising by Latent Representation Alignment with Cosine Distance
 - **Authors:** Diep Luong, Mikko Heikkinen, Konstantinos Drossos, Tuomas Virtanen
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.03442

 - **Pdf link:** https://arxiv.org/pdf/2505.03442

 - **Abstract**
 Speech denoising is a generally adopted and impactful task, appearing in many common and everyday-life use cases. Although there are very powerful methods published, most of those are too complex for deployment in everyday and low-resources computational environments, like hand-held devices, intelligent glasses, hearing aids, etc. Knowledge distillation (KD) is a prominent way for alleviating this complexity mismatch and is based on the transferring/distilling of knowledge from a pre-trained complex model, the teacher, to another less complex one, the student. Existing KD methods for speech denoising are based on processes that potentially hamper the KD by bounding the learning of the student to the distribution, information ordering, and feature dimensionality learned by the teacher. In this paper, we present and assess a method that tries to treat this issue, by exploiting the well-known denoising-autoencoder framework, the linear inverted bottlenecks, and the properties of the cosine similarity. We use a public dataset and conduct repeated experiments with different mismatching scenarios between the teacher and the student, reporting the mean and standard deviation of the metrics of our method and another, state-of-the-art method that is used as a baseline. Our results show that with the proposed method, the student can perform better and can also retain greater mismatching conditions compared to the teacher.


by Zyzzyva0381 (Windy). 


2025-05-07
