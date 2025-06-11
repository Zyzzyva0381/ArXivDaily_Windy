# Showing new listings for Wednesday, 11 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 6papers 
#### Approaching Dialogue State Tracking via Aligning Speech Encoders and LLMs
 - **Authors:** Šimon Sedláček, Bolaji Yusuf, Ján Švec, Pradyoth Hegde, Santosh Kesiraju, Oldřich Plchot, Jan Černocký
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2506.08633

 - **Pdf link:** https://arxiv.org/pdf/2506.08633

 - **Abstract**
 In this work, we approach spoken Dialogue State Tracking (DST) by bridging the representation spaces of speech encoders and LLMs via a small connector module, with a focus on fully open-sourced and open-data components (WavLM-large, OLMo). We focus on ablating different aspects of such systems including full/LoRA adapter fine-tuning, the effect of agent turns in the dialogue history, as well as fuzzy matching-based output post-processing, which greatly improves performance of our systems on named entities in the dialogue slot values. We conduct our experiments on the SpokenWOZ dataset, and additionally utilize the Speech-Aware MultiWOZ dataset to augment our training data. Ultimately, our best-performing WavLM + connector + OLMo-1B aligned models achieve state of the art on the SpokenWOZ test set (34.66% JGA), and our system with Gemma-2-9B-instruct further surpasses this result, reaching 42.17% JGA on SpokenWOZ test.
#### A Review on Score-based Generative Models for Audio Applications
 - **Authors:** Ge Zhu, Yutong Wen, Zhiyao Duan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.08457

 - **Pdf link:** https://arxiv.org/pdf/2506.08457

 - **Abstract**
 Diffusion models have emerged as powerful deep generative techniques, producing high-quality and diverse samples in applications in various domains including audio. These models have many different design choices suitable for different applications, however, existing reviews lack in-depth discussions of these design choices. The audio diffusion model literature also lacks principled guidance for the implementation of these design choices and their comparisons for different applications. This survey provides a comprehensive review of diffusion model design with an emphasis on design principles for quality improvement and conditioning for audio applications. We adopt the score modeling perspective as a unifying framework that accommodates various interpretations, including recent approaches like flow matching. We systematically examine the training and sampling procedures of diffusion models, and audio applications through different conditioning mechanisms. To address the lack of audio diffusion model codebases and to promote reproducible research and rapid prototyping, we introduce an open-source codebase at this https URL that implements our reviewed framework for various audio applications. We demonstrate its capabilities through three case studies: audio generation, speech enhancement, and text-to-speech synthesis, with benchmark evaluations on standard datasets.
#### Passive acoustic non-line-of-sight localization without a relay surface
 - **Authors:** Tal I. Sommer, Ori Katz
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.08471

 - **Pdf link:** https://arxiv.org/pdf/2506.08471

 - **Abstract**
 The detection and localization of a source hidden outside the Line-of-Sight (LOS) traditionally rely on the acquisition of indirect signals, such as those reflected from visible relay surfaces such as floors or walls. These reflected signals are then utilized to reconstruct the obscured scene. In this study, we present an approach that utilize signals diffracted from an edge of an obstacle to achieve three-dimensional (3D) localization of an acoustic point source situated outside the LOS. We address two scenarios - a doorway and a convex corner - and propose a localization method for each of them. For the first scenario, we utilize the two edges of the door as virtual detector arrays. For the second scenario, we exploit the spectral signature of a knife-edge diffraction, inspired by the human perception of sound location by the head-related transfer function (HRTF). In both methods, knife-edge diffraction is utilized to extend the capabilities of non-line-of-sight (NLOS) acoustic sensing, enabling localization in environments where conventional relay-surface based approaches may be limited.
#### Multi-Teacher Language-Aware Knowledge Distillation for Multilingual Speech Emotion Recognition
 - **Authors:** Mehedi Hasan Bijoy, Dejan Porjazovski, Tamás Grósz, Mikko Kurimo
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.08717

 - **Pdf link:** https://arxiv.org/pdf/2506.08717

 - **Abstract**
 Speech Emotion Recognition (SER) is crucial for improving human-computer interaction. Despite strides in monolingual SER, extending them to build a multilingual system remains challenging. Our goal is to train a single model capable of multilingual SER by distilling knowledge from multiple teacher models. To address this, we introduce a novel language-aware multi-teacher knowledge distillation method to advance SER in English, Finnish, and French. It leverages Wav2Vec2.0 as the foundation of monolingual teacher models and then distills their knowledge into a single multilingual student model. The student model demonstrates state-of-the-art performance, with a weighted recall of 72.9 on the English dataset and an unweighted recall of 63.4 on the Finnish dataset, surpassing fine-tuning and knowledge distillation baselines. Our method excels in improving recall for sad and neutral emotions, although it still faces challenges in recognizing anger and happiness.
#### Addressing Pitfalls in Auditing Practices of Automatic Speech Recognition Technologies: A Case Study of People with Aphasia
 - **Authors:** Katelyn Xiaoying Mei, Anna Seo Gyeong Choi, Hilke Schellmann, Mona Sloane, Allison Koenecke
 - **Subjects:** Subjects:
Computers and Society (cs.CY); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.08846

 - **Pdf link:** https://arxiv.org/pdf/2506.08846

 - **Abstract**
 Automatic Speech Recognition (ASR) has transformed daily tasks from video transcription to workplace hiring. ASR systems' growing use warrants robust and standardized auditing approaches to ensure automated transcriptions of high and equitable quality. This is especially critical for people with speech and language disorders (such as aphasia) who may disproportionately depend on ASR systems to navigate everyday life. In this work, we identify three pitfalls in existing standard ASR auditing procedures, and demonstrate how addressing them impacts audit results via a case study of six popular ASR systems' performance for aphasia speakers. First, audits often adhere to a single method of text standardization during data pre-processing, which (a) masks variability in ASR performance from applying different standardization methods, and (b) may not be consistent with how users - especially those from marginalized speech communities - would want their transcriptions to be standardized. Second, audits often display high-level demographic findings without further considering performance disparities among (a) more nuanced demographic subgroups, and (b) relevant covariates capturing acoustic information from the input audio. Third, audits often rely on a single gold-standard metric -- the Word Error Rate -- which does not fully capture the extent of errors arising from generative AI models, such as transcription hallucinations. We propose a more holistic auditing framework that accounts for these three pitfalls, and exemplify its results in our case study, finding consistently worse ASR performance for aphasia speakers relative to a control group. We call on practitioners to implement these robust ASR auditing practices that remain flexible to the rapidly changing ASR landscape.
#### Step-Audio-AQAA: a Fully End-to-End Expressive Large Audio Language Model
 - **Authors:** Ailin Huang, Bingxin Li, Bruce Wang, Boyong Wu, Chao Yan, Chengli Feng, Heng Wang, Hongyu Zhou, Hongyuan Wang, Jingbei Li, Jianjian Sun, Joanna Wang, Mingrui Chen, Peng Liu, Ruihang Miao, Shilei Jiang, Tian Fei, Wang You, Xi Chen, Xuerui Yang, Yechang Huang, Yuxiang Zhang, Zheng Ge, Zheng Gong, Zhewei Huang, Zixin Zhang, Bin Wang, Bo Li, Buyun Ma, Changxin Miao, Changyi Wan, Chen Xu, Dapeng Shi, Dingyuan Hu, Enle Liu, Guanzhe Huang, Gulin Yan, Hanpeng Hu, Haonan Jia, Jiahao Gong, Jiaoren Wu, Jie Wu, Jie Yang, Junzhe Lin, Kaixiang Li, Lei Xia, Longlong Gu, Ming Li, Nie Hao, Ranchen Ming, Shaoliang Pang, Siqi Liu, Song Yuan, Tiancheng Cao, Wen Li, Wenqing He, Xu Zhao, Xuelin Zhang, Yanbo Yu, Yinmin Zhong, Yu Zhou, Yuanwei Liang, Yuanwei Lu, Yuxiang Yang, Zidong Yang, Zili Zhang, Binxing Jiao, Heung-Yeung Shum, Jiansheng Chen, Jing Li, Xiangyu Zhang, Xinhao Zhang, Yibo Zhu, Daxin Jiang, Shuchang Zhou, Chen Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.08967

 - **Pdf link:** https://arxiv.org/pdf/2506.08967

 - **Abstract**
 Large Audio-Language Models (LALMs) have significantly advanced intelligent human-computer interaction, yet their reliance on text-based outputs limits their ability to generate natural speech responses directly, hindering seamless audio interactions. To address this, we introduce Step-Audio-AQAA, a fully end-to-end LALM designed for Audio Query-Audio Answer (AQAA) tasks. The model integrates a dual-codebook audio tokenizer for linguistic and semantic feature extraction, a 130-billion-parameter backbone LLM and a neural vocoder for high-fidelity speech synthesis. Our post-training approach employs interleaved token-output of text and audio to enhance semantic coherence and combines Direct Preference Optimization (DPO) with model merge to improve performance. Evaluations on the StepEval-Audio-360 benchmark demonstrate that Step-Audio-AQAA excels especially in speech control, outperforming the state-of-art LALMs in key areas. This work contributes a promising solution for end-to-end LALMs and highlights the critical role of token-based vocoder in enhancing overall performance for AQAA tasks.


by Zyzzyva0381 (Windy). 


2025-06-11
