# Showing new listings for Thursday, 8 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 9papers 
#### Robust Speech Recognition with Schrödinger Bridge-Based Speech Enhancement
 - **Authors:** Rauf Nasretdinov, Roman Korostik, Ante Jukić
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.04237

 - **Pdf link:** https://arxiv.org/pdf/2505.04237

 - **Abstract**
 In this work, we investigate application of generative speech enhancement to improve the robustness of ASR models in noisy and reverberant conditions. We employ a recently-proposed speech enhancement model based on Schrödinger bridge, which has been shown to perform well compared to diffusion-based approaches. We analyze the impact of model scaling and different sampling methods on the ASR performance. Furthermore, we compare the considered model with predictive and diffusion-based baselines and analyze the speech recognition performance when using different pre-trained ASR models. The proposed approach significantly reduces the word error rate, reducing it by approximately 40% relative to the unprocessed speech signals and by approximately 8% relative to a similarly sized predictive approach.
#### Recognizing Ornaments in Vocal Indian Art Music with Active Annotation
 - **Authors:** Sumit Kumar, Parampreet Singh, Vipul Arora
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2505.04419

 - **Pdf link:** https://arxiv.org/pdf/2505.04419

 - **Abstract**
 Ornamentations, embellishments, or microtonal inflections are essential to melodic expression across many musical traditions, adding depth, nuance, and emotional impact to performances. Recognizing ornamentations in singing voices is key to MIR, with potential applications in music pedagogy, singer identification, genre classification, and controlled singing voice generation. However, the lack of annotated datasets and specialized modeling approaches remains a major obstacle for progress in this research area. In this work, we introduce Rāga Ornamentation Detection (ROD), a novel dataset comprising Indian classical music recordings curated by expert musicians. The dataset is annotated using a custom Human-in-the-Loop tool for six vocal ornaments marked as event-based labels. Using this dataset, we develop an ornamentation detection model based on deep time-series analysis, preserving ornament boundaries during the chunking of long audio recordings. We conduct experiments using different train-test configurations within the ROD dataset and also evaluate our approach on a separate, manually annotated dataset of Indian classical concert recordings. Our experimental results support the superior performance of our proposed approach over the baseline CRNN.
#### Accelerating Audio Research with Robotic Dummy Heads
 - **Authors:** Austin Lu, Kanad Sarkar, Yongjie Zhuang, Leo Lin, Ryan M Corey, Andrew C Singer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Human-Computer Interaction (cs.HC); Robotics (cs.RO); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.04548

 - **Pdf link:** https://arxiv.org/pdf/2505.04548

 - **Abstract**
 This work introduces a robotic dummy head that fuses the acoustic realism of conventional audiological mannequins with the mobility of robots. The proposed device is capable of moving, talking, and listening as people do, and can be used to automate spatially-stationary audio experiments, thus accelerating the pace of audio research. Critically, the device may also be used as a moving sound source in dynamic experiments, due to its quiet motor. This feature differentiates our work from previous robotic acoustic research platforms. Validation that the robot enables high quality audio data collection is provided through various experiments and acoustic measurements. These experiments also demonstrate how the robot might be used to study adaptive binaural beamforming. Design files are provided as open-source to stimulate novel audio research.
#### EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning
 - **Authors:** Zhenghao Xing, Xiaowei Hu, Chi-Wing Fu, Wenhai Wang, Jifeng Dai, Pheng-Ann Heng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.04623

 - **Pdf link:** https://arxiv.org/pdf/2505.04623

 - **Abstract**
 Multimodal large language models (MLLMs) have advanced perception across text, vision, and audio, yet they often struggle with structured cross-modal reasoning, particularly when integrating audio and visual signals. We introduce EchoInk-R1, a reinforcement learning framework that enhances such reasoning in MLLMs. Built upon the Qwen2.5-Omni-7B foundation and optimized with Group Relative Policy Optimization (GRPO), EchoInk-R1 tackles multiple-choice question answering over synchronized audio-image pairs. To enable this, we curate AVQA-R1-6K, a dataset pairing such audio-image inputs with multiple-choice questions derived from OmniInstruct-v1. EchoInk-R1-7B achieves 85.77% accuracy on the validation set, outperforming the base model, which scores 80.53%, using only 562 reinforcement learning steps. Beyond accuracy, EchoInk-R1 demonstrates reflective reasoning by revisiting initial interpretations and refining responses when facing ambiguous multimodal inputs. These results suggest that lightweight reinforcement learning fine-tuning enhances cross-modal reasoning in MLLMs. EchoInk-R1 is the first framework to unify audio, visual, and textual modalities for general open-world reasoning via reinforcement learning. Code and data are publicly released to facilitate further research.
#### LLAMAPIE: Proactive In-Ear Conversation Assistants
 - **Authors:** Tuochao Chen, Nicholas Batchelder, Alisa Liu, Noah Smith, Shyamnath Gollakota
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.04066

 - **Pdf link:** https://arxiv.org/pdf/2505.04066

 - **Abstract**
 We introduce LlamaPIE, the first real-time proactive assistant designed to enhance human conversations through discreet, concise guidance delivered via hearable devices. Unlike traditional language models that require explicit user invocation, this assistant operates in the background, anticipating user needs without interrupting conversations. We address several challenges, including determining when to respond, crafting concise responses that enhance conversations, leveraging knowledge of the user for context-aware assistance, and real-time, on-device processing. To achieve this, we construct a semi-synthetic dialogue dataset and propose a two-model pipeline: a small model that decides when to respond and a larger model that generates the response. We evaluate our approach on real-world datasets, demonstrating its effectiveness in providing helpful, unobtrusive assistance. User studies with our assistant, implemented on Apple Silicon M2 hardware, show a strong preference for the proactive assistant over both a baseline with no assistance and a reactive model, highlighting the potential of LlamaPie to enhance live conversations.
#### Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment
 - **Authors:** Xueyao Zhang, Yuancheng Wang, Chaoren Wang, Ziniu Li, Zhuo Chen, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.04113

 - **Pdf link:** https://arxiv.org/pdf/2505.04113

 - **Abstract**
 Modern zero-shot text-to-speech (TTS) systems, despite using extensive pre-training, often struggle in challenging scenarios such as tongue twisters, repeated words, code-switching, and cross-lingual synthesis, leading to intelligibility issues. To address these limitations, this paper leverages preference alignment techniques, which enable targeted construction of out-of-pretraining-distribution data to enhance performance. We introduce a new dataset, named the Intelligibility Preference Speech Dataset (INTP), and extend the Direct Preference Optimization (DPO) framework to accommodate diverse TTS architectures. After INTP alignment, in addition to intelligibility, we observe overall improvements including naturalness, similarity, and audio quality for multiple TTS models across diverse domains. Based on that, we also verify the weak-to-strong generalization ability of INTP for more intelligible models such as CosyVoice 2 and Ints. Moreover, we showcase the potential for further improvements through iterative alignment based on Ints. Audio samples are available at this https URL.
#### SwinLip: An Efficient Visual Speech Encoder for Lip Reading Using Swin Transformer
 - **Authors:** Young-Hu Park, Rae-Hong Park, Hyung-Min Park
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.04394

 - **Pdf link:** https://arxiv.org/pdf/2505.04394

 - **Abstract**
 This paper presents an efficient visual speech encoder for lip reading. While most recent lip reading studies have been based on the ResNet architecture and have achieved significant success, they are not sufficiently suitable for efficiently capturing lip reading features due to high computational complexity in modeling spatio-temporal information. Additionally, using a complex visual model not only increases the complexity of lip reading models but also induces delays in the overall network for multi-modal studies (e.g., audio-visual speech recognition, speech enhancement, and speech separation). To overcome the limitations of Convolutional Neural Network (CNN)-based models, we apply the hierarchical structure and window self-attention of the Swin Transformer to lip reading. We configure a new lightweight scale of the Swin Transformer suitable for processing lip reading data and present the SwinLip visual speech encoder, which efficiently reduces computational load by integrating modified Convolution-augmented Transformer (Conformer) temporal embeddings with conventional spatial embeddings in the hierarchical structure. Through extensive experiments, we have validated that our SwinLip successfully improves the performance and inference speed of the lip reading network when applied to various backbones for word and sentence recognition, reducing computational load. In particular, our SwinLip demonstrated robust performance in both English LRW and Mandarin LRW-1000 datasets and achieved state-of-the-art performance on the Mandarin LRW-1000 dataset with less computation compared to the existing state-of-the-art model.
#### Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration
 - **Authors:** Shigeki Karita, Yuma Koizumi, Heiga Zen, Haruko Ishikawa, Robin Scheibler, Michiel Bacchiani
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.04457

 - **Pdf link:** https://arxiv.org/pdf/2505.04457

 - **Abstract**
 Training data cleaning is a new application for generative model-based speech restoration (SR). This paper introduces Miipher-2, an SR model designed for million-hour scale data, for training data cleaning for large-scale generative models like large language models. Key challenges addressed include generalization to unseen languages, operation without explicit conditioning (e.g., text, speaker ID), and computational efficiency. Miipher-2 utilizes a frozen, pre-trained Universal Speech Model (USM), supporting over 300 languages, as a robust, conditioning-free feature extractor. To optimize efficiency and minimize memory, Miipher-2 incorporates parallel adapters for predicting clean USM features from noisy inputs and employs the WaneFit neural vocoder for waveform synthesis. These components were trained on 3,000 hours of multi-lingual, studio-quality recordings with augmented degradations, while USM parameters remained fixed. Experimental results demonstrate Miipher-2's superior or comparable performance to conventional SR models in word-error-rate, speaker similarity, and both objective and subjective sound quality scores across all tested languages. Miipher-2 operates efficiently on consumer-grade accelerators, achieving a real-time factor of 0.0078, enabling the processing of a million-hour speech dataset in approximately three days using only 100 such accelerators.
#### Score Distillation Sampling for Audio: Source Separation, Synthesis, and Beyond
 - **Authors:** Jessie Richter-Powell, Antonio Torralba, Jonathan Lorraine
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.04621

 - **Pdf link:** https://arxiv.org/pdf/2505.04621

 - **Abstract**
 We introduce Audio-SDS, a generalization of Score Distillation Sampling (SDS) to text-conditioned audio diffusion models. While SDS was initially designed for text-to-3D generation using image diffusion, its core idea of distilling a powerful generative prior into a separate parametric representation extends to the audio domain. Leveraging a single pretrained model, Audio-SDS enables a broad range of tasks without requiring specialized datasets. In particular, we demonstrate how Audio-SDS can guide physically informed impact sound simulations, calibrate FM-synthesis parameters, and perform prompt-specified source separation. Our findings illustrate the versatility of distillation-based methods across modalities and establish a robust foundation for future work using generative priors in audio tasks.


by Zyzzyva0381 (Windy). 


2025-05-08
