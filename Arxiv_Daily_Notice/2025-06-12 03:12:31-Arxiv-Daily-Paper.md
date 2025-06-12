# Showing new listings for Thursday, 12 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 14papers 
#### Enhancing Acoustic-to-Articulatory Speech Inversion by Incorporating Nasality
 - **Authors:** Saba Tabatabaee, Suzanne Boyce, Liran Oren, Mark Tiede, Carol Espy-Wilson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09231

 - **Pdf link:** https://arxiv.org/pdf/2506.09231

 - **Abstract**
 Speech is produced through the coordination of vocal tract constricting organs: lips, tongue, velum, and glottis. Previous works developed Speech Inversion (SI) systems to recover acoustic-to-articulatory mappings for lip and tongue constrictions, called oral tract variables (TVs), which were later enhanced by including source information (periodic and aperiodic energies, and F0 frequency) as proxies for glottal control. Comparison of the nasometric measures with high-speed nasopharyngoscopy showed that nasalance can serve as ground truth, and that an SI system trained with it reliably recovers velum movement patterns for American English speakers. Here, two SI training approaches are compared: baseline models that estimate oral TVs and nasalance independently, and a synergistic model that combines oral TVs and source features with nasalance. The synergistic model shows relative improvements of 5% in oral TVs estimation and 9% in nasalance estimation compared to the baseline models.
#### You Are What You Say: Exploiting Linguistic Content for VoicePrivacy Attacks
 - **Authors:** Ünal Ege Gaznepoglu, Anna Leschanowsky, Ahmad Aloradi, Prachi Singh, Daniel Tenbrinck, Emanuël A. P. Habets, Nils Peters
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2506.09521

 - **Pdf link:** https://arxiv.org/pdf/2506.09521

 - **Abstract**
 Speaker anonymization systems hide the identity of speakers while preserving other information such as linguistic content and emotions. To evaluate their privacy benefits, attacks in the form of automatic speaker verification (ASV) systems are employed. In this study, we assess the impact of intra-speaker linguistic content similarity in the attacker training and evaluation datasets, by adapting BERT, a language model, as an ASV system. On the VoicePrivacy Attacker Challenge datasets, our method achieves a mean equal error rate (EER) of 35%, with certain speakers attaining EERs as low as 2%, based solely on the textual content of their utterances. Our explainability study reveals that the system decisions are linked to semantically similar keywords within utterances, stemming from how LibriSpeech is curated. Our study suggests reworking the VoicePrivacy datasets to ensure a fair and unbiased evaluation and challenge the reliance on global EER for privacy evaluations.
#### A Study on Speech Assessment with Visual Cues
 - **Authors:** Shafique Ahmed, Ryandhimas E. Zezario, Nasir Saleem, Amir Hussain, Hsin-Min Wang, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.09549

 - **Pdf link:** https://arxiv.org/pdf/2506.09549

 - **Abstract**
 Non-intrusive assessment of speech quality and intelligibility is essential when clean reference signals are unavailable. In this work, we propose a multimodal framework that integrates audio features and visual cues to predict PESQ and STOI scores. It employs a dual-branch architecture, where spectral features are extracted using STFT, and visual embeddings are obtained via a visual encoder. These features are then fused and processed by a CNN-BLSTM with attention, followed by multi-task learning to simultaneously predict PESQ and STOI. Evaluations on the LRS3-TED dataset, augmented with noise from the DEMAND corpus, show that our model outperforms the audio-only baseline. Under seen noise conditions, it improves LCC by 9.61% (0.8397->0.9205) for PESQ and 11.47% (0.7403->0.8253) for STOI. These results highlight the effectiveness of incorporating visual cues in enhancing the accuracy of non-intrusive speech assessment.
#### Recognizing Every Voice: Towards Inclusive ASR for Rural Bhojpuri Women
 - **Authors:** Sakshi Joshi, Eldho Ittan George, Tahir Javed, Kaushal Bhogale, Nikhil Narasimhan, Mitesh M. Khapra
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09653

 - **Pdf link:** https://arxiv.org/pdf/2506.09653

 - **Abstract**
 Digital inclusion remains a challenge for marginalized communities, especially rural women in low-resource language regions like Bhojpuri. Voice-based access to agricultural services, financial transactions, government schemes, and healthcare is vital for their empowerment, yet existing ASR systems for this group remain largely untested. To address this gap, we create SRUTI ,a benchmark consisting of rural Bhojpuri women speakers. Evaluation of current ASR models on SRUTI shows poor performance due to data scarcity, which is difficult to overcome due to social and cultural barriers that hinder large-scale data collection. To overcome this, we propose generating synthetic speech using just 25-30 seconds of audio per speaker from approximately 100 rural women. Augmenting existing datasets with this synthetic data achieves an improvement of 4.7 WER, providing a scalable, minimally intrusive solution to enhance ASR and promote digital inclusion in low-resource language.
#### Fine-Tuning Large Audio-Language Models with LoRA for Precise Temporal Localization of Prolonged Exposure Therapy Elements
 - **Authors:** Suhas BN, Andrew M. Sherrill, Jyoti Alaparthi, Dominik Mattioli, Rosa I. Arriaga, Chris W. Wiese, Saeed Abdullah
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC)
 - **Arxiv link:** https://arxiv.org/abs/2506.09707

 - **Pdf link:** https://arxiv.org/pdf/2506.09707

 - **Abstract**
 Prolonged Exposure (PE) therapy is an effective treatment for post-traumatic stress disorder (PTSD), but evaluating therapist fidelity remains labor-intensive due to the need for manual review of session recordings. We present a method for the automatic temporal localization of key PE fidelity elements -- identifying their start and stop times -- directly from session audio and transcripts. Our approach fine-tunes a large pre-trained audio-language model, Qwen2-Audio, using Low-Rank Adaptation (LoRA) to process focused 30-second windows of audio-transcript input. Fidelity labels for three core protocol phases -- therapist orientation (P1), imaginal exposure (P2), and post-imaginal processing (P3) -- are generated via LLM-based prompting and verified by trained raters. The model is trained to predict normalized boundary offsets using soft supervision guided by task-specific prompts. On a dataset of 313 real PE sessions, our best configuration (LoRA rank 8, 30s windows) achieves a mean absolute error (MAE) of 5.3 seconds across tasks. We further analyze the effects of window size and LoRA rank, highlighting the importance of context granularity and model adaptation. This work introduces a scalable framework for fidelity tracking in PE therapy, with potential to support clinician training, supervision, and quality assurance.
#### Regularizing Learnable Feature Extraction for Automatic Speech Recognition
 - **Authors:** Peter Vieting, Maximilian Kannen, Benedikt Hilmes, Ralf Schlüter, Hermann Ney
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.09804

 - **Pdf link:** https://arxiv.org/pdf/2506.09804

 - **Abstract**
 Neural front-ends are an appealing alternative to traditional, fixed feature extraction pipelines for automatic speech recognition (ASR) systems since they can be directly trained to fit the acoustic model. However, their performance often falls short compared to classical methods, which we show is largely due to their increased susceptibility to overfitting. This work therefore investigates regularization methods for training ASR models with learnable feature extraction front-ends. First, we examine audio perturbation methods and show that larger relative improvements can be obtained for learnable features. Additionally, we identify two limitations in the standard use of SpecAugment for these front-ends and propose masking in the short time Fourier transform (STFT)-domain as a simple but effective modification to address these challenges. Finally, integrating both regularization approaches effectively closes the performance gap between traditional and learnable features.
#### PHRASED: Phrase Dictionary Biasing for Speech Translation
 - **Authors:** Peidong Wang, Jian Xue, Rui Zhao, Junkun Chen, Aswin Shanmugam Subramanian, Jinyu Li
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09175

 - **Pdf link:** https://arxiv.org/pdf/2506.09175

 - **Abstract**
 Phrases are essential to understand the core concepts in conversations. However, due to their rare occurrence in training data, correct translation of phrases is challenging in speech translation tasks. In this paper, we propose a phrase dictionary biasing method to leverage pairs of phrases mapping from the source language to the target language. We apply the phrase dictionary biasing method to two types of widely adopted models, a transducer-based streaming speech translation model and a multimodal large language model. Experimental results show that the phrase dictionary biasing method outperforms phrase list biasing by 21% relatively for the streaming speech translation model. In addition, phrase dictionary biasing enables multimodal large language models to use external phrase information, achieving 85% relative improvement in phrase recall.
#### SimClass: A Classroom Speech Dataset Generated via Game Engine Simulation For Automatic Speech Recognition Research
 - **Authors:** Ahmed Adel Attia, Jing Liu, Carl Espy-Wilson
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09206

 - **Pdf link:** https://arxiv.org/pdf/2506.09206

 - **Abstract**
 The scarcity of large-scale classroom speech data has hindered the development of AI-driven speech models for education. Public classroom datasets remain limited, and the lack of a dedicated classroom noise corpus prevents the use of standard data augmentation techniques. In this paper, we introduce a scalable methodology for synthesizing classroom noise using game engines, a framework that extends to other domains. Using this methodology, we present SimClass, a dataset that includes both a synthesized classroom noise corpus and a simulated classroom speech dataset. The speech data is generated by pairing a public children's speech corpus with YouTube lecture videos to approximate real classroom interactions in clean conditions. Our experiments on clean and noisy speech demonstrate that SimClass closely approximates real classroom speech, making it a valuable resource for developing robust speech recognition and enhancement models.
#### Ming-Omni: A Unified Multimodal Model for Perception and Generation
 - **Authors:** Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, Furong Xu, GuangMing Yao, Jun Zhou, Jingdong Chen, Jianxin Sun, Jiajia Liu, Jianjiang Zhu, Jun Peng, Kaixiang Ji, Kaiyou Song, Kaimeng Ren, Libin Wang, Lixiang Ru, Lele Xie, Longhua Tan, Lyuxin Xue, Lan Wang, Mochen Bai, Ning Gao, Pei Chen, Qingpei Guo, Qinglong Zhang, Qiang Xu, Rui Liu, Ruijie Xiong, Sirui Gao, Tinghao Liu, Taisong Li, Weilong Chai, Xinyu Xiao, Xiaomei Wang, Xiaoxue Chen, Xiao Lu, Xiaoyu Li, Xingning Dong, Xuzheng Yu, Yi Yuan, Yuting Gao, Yunxiao Sun, Yipeng Chen, Yifei Wu, Yongjie Lyu, Ziping Ma, Zipeng Feng, Zhijiang Fang, Zhihao Qiu, Ziyuan Huang, Zhengyu He
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09344

 - **Pdf link:** https://arxiv.org/pdf/2506.09344

 - **Abstract**
 We propose Ming-Omni, a unified multimodal model capable of processing images, text, audio, and video, while demonstrating strong proficiency in both speech and image generation. Ming-Omni employs dedicated encoders to extract tokens from different modalities, which are then processed by Ling, an MoE architecture equipped with newly proposed modality-specific routers. This design enables a single model to efficiently process and fuse multimodal inputs within a unified framework, thereby facilitating diverse tasks without requiring separate models, task-specific fine-tuning, or structural redesign. Importantly, Ming-Omni extends beyond conventional multimodal models by supporting audio and image generation. This is achieved through the integration of an advanced audio decoder for natural-sounding speech and Ming-Lite-Uni for high-quality image generation, which also allow the model to engage in context-aware chatting, perform text-to-speech conversion, and conduct versatile image editing. Our experimental results showcase Ming-Omni offers a powerful solution for unified perception and generation across all modalities. Notably, our proposed Ming-Omni is the first open-source model we are aware of to match GPT-4o in modality support, and we release all code and model weights to encourage further research and development in the community.
#### CoLMbo: Speaker Language Model for Descriptive Profiling
 - **Authors:** Massa Baali, Shuo Han, Syed Abdul Hannan, Purusottam Samal, Karanveer Singh, Soham Deshmukh, Rita Singh, Bhiksha Raj
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09375

 - **Pdf link:** https://arxiv.org/pdf/2506.09375

 - **Abstract**
 Speaker recognition systems are often limited to classification tasks and struggle to generate detailed speaker characteristics or provide context-rich descriptions. These models primarily extract embeddings for speaker identification but fail to capture demographic attributes such as dialect, gender, and age in a structured manner. This paper introduces CoLMbo, a Speaker Language Model (SLM) that addresses these limitations by integrating a speaker encoder with prompt-based conditioning. This allows for the creation of detailed captions based on speaker embeddings. CoLMbo utilizes user-defined prompts to adapt dynamically to new speaker characteristics and provides customized descriptions, including regional dialect variations and age-related traits. This innovative approach not only enhances traditional speaker profiling but also excels in zero-shot scenarios across diverse datasets, marking a significant advancement in the field of speaker recognition.
#### OWSM-Biasing: Contextualizing Open Whisper-Style Speech Models for Automatic Speech Recognition with Dynamic Vocabulary
 - **Authors:** Yui Sudo, Yusuke Fujita, Atsushi Kojima, Tomoya Mizumoto, Lianbo Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09448

 - **Pdf link:** https://arxiv.org/pdf/2506.09448

 - **Abstract**
 Speech foundation models (SFMs), such as Open Whisper-Style Speech Models (OWSM), are trained on massive datasets to achieve accurate automatic speech recognition. However, even SFMs struggle to accurately recognize rare and unseen words. While contextual biasing (CB) is a promising approach to improve recognition of such words, most CB methods are trained from scratch, resulting in lower performance than SFMs due to the lack of pre-trained knowledge. This paper integrates an existing CB method with OWSM v3.1 while freezing its pre-trained parameters. By leveraging the knowledge embedded in SFMs, the proposed method enables effective CB while preserving the advantages of SFMs, even with a small dataset. Experimental results show that the proposed method improves the biasing word error rate (B-WER) by 11.6 points, resulting in a 0.9 point improvement in the overall WER while reducing the real-time factor by 7.5% compared to the non-biasing baseline on the LibriSpeech 100 test-clean set.
#### Training-Free Voice Conversion with Factorized Optimal Transport
 - **Authors:** Alexander Lobashev, Assel Yermekova, Maria Larchenko
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09709

 - **Pdf link:** https://arxiv.org/pdf/2506.09709

 - **Abstract**
 This paper introduces Factorized MKL-VC, a training-free modification for kNN-VC pipeline. In contrast with original pipeline, our algorithm performs high quality any-to-any cross-lingual voice conversion with only 5 second of reference audio. MKL-VC replaces kNN regression with a factorized optimal transport map in WavLM embedding subspaces, derived from Monge-Kantorovich Linear solution. Factorization addresses non-uniform variance across dimensions, ensuring effective feature transformation. Experiments on LibriSpeech and FLEURS datasets show MKL-VC significantly improves content preservation and robustness with short reference audio, outperforming kNN-VC. MKL-VC achieves performance comparable to FACodec, especially in cross-lingual voice conversion domain.
#### Incorporating Linguistic Constraints from External Knowledge Source for Audio-Visual Target Speech Extraction
 - **Authors:** Wenxuan Wu, Shuai Wang, Xixin Wu, Helen Meng, Haizhou Li
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09792

 - **Pdf link:** https://arxiv.org/pdf/2506.09792

 - **Abstract**
 Audio-visual target speaker extraction (AV-TSE) models primarily rely on target visual cues to isolate the target speaker's voice from others. We know that humans leverage linguistic knowledge, such as syntax and semantics, to support speech perception. Inspired by this, we explore the potential of pre-trained speech-language models (PSLMs) and pre-trained language models (PLMs) as auxiliary knowledge sources for AV-TSE. In this study, we propose incorporating the linguistic constraints from PSLMs or PLMs for the AV-TSE model as additional supervision signals. Without introducing any extra computational cost during inference, the proposed approach consistently improves speech quality and intelligibility. Furthermore, we evaluate our method in multi-language settings and visual cue-impaired scenarios and show robust performance gains.
#### UmbraTTS: Adapting Text-to-Speech to Environmental Contexts with Flow Matching
 - **Authors:** Neta Glazer, Aviv Navon, Yael Segal, Aviv Shamsian, Hilit Segev, Asaf Buchnick, Menachem Pirchi, Gil Hetz, Joseph Keshet
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.09874

 - **Pdf link:** https://arxiv.org/pdf/2506.09874

 - **Abstract**
 Recent advances in Text-to-Speech (TTS) have enabled highly natural speech synthesis, yet integrating speech with complex background environments remains challenging. We introduce UmbraTTS, a flow-matching based TTS model that jointly generates both speech and environmental audio, conditioned on text and acoustic context. Our model allows fine-grained control over background volume and produces diverse, coherent, and context-aware audio scenes. A key challenge is the lack of data with speech and background audio aligned in natural context. To overcome the lack of paired training data, we propose a self-supervised framework that extracts speech, background audio, and transcripts from unannotated recordings. Extensive evaluations demonstrate that UmbraTTS significantly outperformed existing baselines, producing natural, high-quality, environmentally aware audios.


by Zyzzyva0381 (Windy). 


2025-06-12
