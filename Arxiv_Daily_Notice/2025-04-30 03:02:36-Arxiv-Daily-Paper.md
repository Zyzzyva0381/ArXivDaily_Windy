# Showing new listings for Wednesday, 30 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 9papers 
#### Towards Flow-Matching-based TTS without Classifier-Free Guidance
 - **Authors:** Yuzhe Liang, Wenzhe Liu, Chunyu Qiang, Zhikang Niu, Yushen Chen, Ziyang Ma, Wenxi Chen, Nan Li, Chen Zhang, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.20334

 - **Pdf link:** https://arxiv.org/pdf/2504.20334

 - **Abstract**
 Flow matching has demonstrated strong generative capabilities and has become a core component in modern Text-to-Speech (TTS) systems. To ensure high-quality speech synthesis, Classifier-Free Guidance (CFG) is widely used during the inference of flow-matching-based TTS models. However, CFG incurs substantial computational cost as it requires two forward passes, which hinders its applicability in real-time scenarios. In this paper, we explore removing CFG from flow-matching-based TTS models to improve inference efficiency, while maintaining performance. Specifically, we reformulated the flow matching training target to directly approximate the CFG optimization trajectory. This training method eliminates the need for unconditional model evaluation and guided tuning during inference, effectively cutting the computational overhead in half. Furthermore, It can be seamlessly integrated with existing optimized sampling strategies. We validate our approach using the F5-TTS model on the LibriTTS dataset. Experimental results show that our method achieves a 9$\times$ inference speed-up compared to the baseline F5-TTS, while preserving comparable speech quality. We will release the code and models to support reproducibility and foster further research in this area.
#### ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting
 - **Authors:** Yu Zhang, Wenxiang Guo, Changhao Pan, Zhiyuan Zhu, Tao Jin, Zhou Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2504.20630

 - **Pdf link:** https://arxiv.org/pdf/2504.20630

 - **Abstract**
 Multimodal immersive spatial drama generation focuses on creating continuous multi-speaker binaural speech with dramatic prosody based on multimodal prompts, with potential applications in AR, VR, and others. This task requires simultaneous modeling of spatial information and dramatic prosody based on multimodal inputs, with high data collection costs. To the best of our knowledge, our work is the first attempt to address these challenges. We construct MRSDrama, the first multimodal recorded spatial drama dataset, containing binaural drama audios, scripts, videos, geometric poses, and textual prompts. Then, we propose ISDrama, the first immersive spatial drama generation model through multimodal prompting. ISDrama comprises these primary components: 1) Multimodal Pose Encoder, based on contrastive learning, considering the Doppler effect caused by moving speakers to extract unified pose information from multimodal prompts. 2) Immersive Drama Transformer, a flow-based mamba-transformer model that generates high-quality drama, incorporating Drama-MOE to select proper experts for enhanced prosody and pose control. We also design a context-consistent classifier-free guidance strategy to coherently generate complete drama. Experimental results show that ISDrama outperforms baseline models on objective and subjective metrics. The demos and dataset are available at this https URL.
#### APG-MOS: Auditory Perception Guided-MOS Predictor for Synthetic Speech
 - **Authors:** Zhicheng Lian, Lizhi Wang, Hua Huang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.20447

 - **Pdf link:** https://arxiv.org/pdf/2504.20447

 - **Abstract**
 Automatic speech quality assessment aims to quantify subjective human perception of speech through computational models to reduce the need for labor-consuming manual evaluations. While models based on deep learning have achieved progress in predicting mean opinion scores (MOS) to assess synthetic speech, the neglect of fundamental auditory perception mechanisms limits consistency with human judgments. To address this issue, we propose an auditory perception guided-MOS prediction model (APG-MOS) that synergistically integrates auditory modeling with semantic analysis to enhance consistency with human judgments. Specifically, we first design a perceptual module, grounded in biological auditory mechanisms, to simulate cochlear functions, which encodes acoustic signals into biologically aligned electrochemical representations. Secondly, we propose a residual vector quantization (RVQ)-based semantic distortion modeling method to quantify the degradation of speech quality at the semantic level. Finally, we design a residual cross-attention architecture, coupled with a progressive learning strategy, to enable multimodal fusion of encoded electrochemical signals and semantic representations. Experiments demonstrate that APG-MOS achieves superior performance on two primary benchmarks. Our code and checkpoint will be available on a public repository upon publication.
#### TriniMark: A Robust Generative Speech Watermarking Method for Trinity-Level Attribution
 - **Authors:** Yue Li, Weizhi Liu, Dongdong Lin
 - **Subjects:** Subjects:
Multimedia (cs.MM); Cryptography and Security (cs.CR); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.20532

 - **Pdf link:** https://arxiv.org/pdf/2504.20532

 - **Abstract**
 The emergence of diffusion models has facilitated the generation of speech with reinforced fidelity and naturalness. While deepfake detection technologies have manifested the ability to identify AI-generated content, their efficacy decreases as generative models become increasingly sophisticated. Furthermore, current research in the field has not adequately addressed the necessity for robust watermarking to safeguard the intellectual property rights associated with synthetic speech and generative models. To remedy this deficiency, we propose a \textbf{ro}bust generative \textbf{s}peech wat\textbf{e}rmarking method (TriniMark) for authenticating the generated content and safeguarding the copyrights by enabling the traceability of the diffusion model. We first design a structure-lightweight watermark encoder that embeds watermarks into the time-domain features of speech and reconstructs the waveform directly. A temporal-aware gated convolutional network is meticulously designed in the watermark decoder for bit-wise watermark recovery. Subsequently, the waveform-guided fine-tuning strategy is proposed for fine-tuning the diffusion model, which leverages the transferability of watermarks and enables the diffusion model to incorporate watermark knowledge effectively. When an attacker trains a surrogate model using the outputs of the target model, the embedded watermark can still be learned by the surrogate model and correctly extracted. Comparative experiments with state-of-the-art methods demonstrate the superior robustness of our method, particularly in countering compound attacks.
#### DiffusionRIR: Room Impulse Response Interpolation using Diffusion Models
 - **Authors:** Sagi Della Torre, Mirco Pezzoli, Fabio Antonacci, Sharon Gannot
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.20625

 - **Pdf link:** https://arxiv.org/pdf/2504.20625

 - **Abstract**
 Room Impulse Responses (RIRs) characterize acoustic environments and are crucial in multiple audio signal processing tasks. High-quality RIR estimates drive applications such as virtual microphones, sound source localization, augmented reality, and data augmentation. However, obtaining RIR measurements with high spatial resolution is resource-intensive, making it impractical for large spaces or when dense sampling is required. This research addresses the challenge of estimating RIRs at unmeasured locations within a room using Denoising Diffusion Probabilistic Models (DDPM). Our method leverages the analogy between RIR matrices and image inpainting, transforming RIR data into a format suitable for diffusion-based reconstruction. Using simulated RIR data based on the image method, we demonstrate our approach's effectiveness on microphone arrays of different curvatures, from linear to semi-circular. Our method successfully reconstructs missing RIRs, even in large gaps between microphones. Under these conditions, it achieves accurate reconstruction, significantly outperforming baseline Spline Cubic Interpolation in terms of Normalized Mean Square Error and Cosine Distance between actual and interpolated RIRs. This research highlights the potential of using generative models for effective RIR interpolation, paving the way for generating additional data from limited real-world measurements.
#### Non-native Children's Automatic Speech Assessment Challenge (NOCASA)
 - **Authors:** Yaroslav Getman, Tamás Grósz, Mikko Kurimo, Giampiero Salvi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.20678

 - **Pdf link:** https://arxiv.org/pdf/2504.20678

 - **Abstract**
 This paper presents the "Non-native Children's Automatic Speech Assessment" (NOCASA) - a data competition part of the IEEE MLSP 2025 conference. NOCASA challenges participants to develop new systems that can assess single-word pronunciations of young second language (L2) learners as part of a gamified pronunciation training app. To achieve this, several issues must be addressed, most notably the limited nature of available training data and the highly unbalanced distribution among the pronunciation level categories. To expedite the development, we provide a pseudo-anonymized training data (TeflonNorL2), containing 10,334 recordings from 44 speakers attempting to pronounce 205 distinct Norwegian words, human-rated on a 1 to 5 scale (number of stars that should be given in the game). In addition to the data, two already trained systems are released as official baselines: an SVM classifier trained on the ComParE_16 acoustic feature set and a multi-task wav2vec 2.0 model. The latter achieves the best performance on the challenge test set, with an unweighted average recall (UAR) of 36.37%.
#### ECOSoundSet: a finely annotated dataset for the automated acoustic identification of Orthoptera and Cicadidae in North, Central and temperate Western Europe
 - **Authors:** David Funosas, Elodie Massol, Yves Bas, Svenja Schmidt, Dominik Arend, Alexander Gebhard, Luc Barbaro, Sebastian König, Rafael Carbonell Font, David Sannier, Fernand Deroussen, Jérôme Sueur, Christian Roesti, Tomi Trilar, Wolfgang Forstmeier, Lucas Roger, Eloïsa Matheu, Piotr Guzik, Julien Barataud, Laurent Pelozuelo, Stéphane Puissant, Sandra Mueller, Björn Schuller, Jose M. Montoya, Andreas Triantafyllopoulos, Maxime Cauchoix
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.20776

 - **Pdf link:** https://arxiv.org/pdf/2504.20776

 - **Abstract**
 Currently available tools for the automated acoustic recognition of European insects in natural soundscapes are limited in scope. Large and ecologically heterogeneous acoustic datasets are currently needed for these algorithms to cross-contextually recognize the subtle and complex acoustic signatures produced by each species, thus making the availability of such datasets a key requisite for their development. Here we present ECOSoundSet (European Cicadidae and Orthoptera Sound dataSet), a dataset containing 10,653 recordings of 200 orthopteran and 24 cicada species (217 and 26 respective taxa when including subspecies) present in North, Central, and temperate Western Europe (Andorra, Belgium, Denmark, mainland France and Corsica, Germany, Ireland, Luxembourg, Monaco, Netherlands, United Kingdom, Switzerland), collected partly through targeted fieldwork in South France and Catalonia and partly through contributions from various European entomologists. The dataset is composed of a combination of coarsely labeled recordings, for which we can only infer the presence, at some point, of their target species (weak labeling), and finely annotated recordings, for which we know the specific time and frequency range of each insect sound present in the recording (strong labeling). We also provide a train/validation/test split of the strongly labeled recordings, with respective approximate proportions of 0.8, 0.1 and 0.1, in order to facilitate their incorporation in the training and evaluation of deep learning algorithms. This dataset could serve as a meaningful complement to recordings already available online for the training of deep learning algorithms for the acoustic classification of orthopterans and cicadas in North, Central, and temperate Western Europe.
#### Enhancing Non-Core Language Instruction-Following in Speech LLMs via Semi-Implicit Cross-Lingual CoT Reasoning
 - **Authors:** Hongfei Xue, Yufeng Tang, Hexin Liu, Jun Zhang, Xuelong Geng, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.20835

 - **Pdf link:** https://arxiv.org/pdf/2504.20835

 - **Abstract**
 Large language models have been extended to the speech domain, leading to the development of speech large language models (SLLMs). While existing SLLMs demonstrate strong performance in speech instruction-following for core languages (e.g., English), they often struggle with non-core languages due to the scarcity of paired speech-text data and limited multilingual semantic reasoning capabilities. To address this, we propose the semi-implicit Cross-lingual Speech Chain-of-Thought (XS-CoT) framework, which integrates speech-to-text translation into the reasoning process of SLLMs. The XS-CoT generates four types of tokens: instruction and response tokens in both core and non-core languages, enabling cross-lingual transfer of reasoning capabilities. To mitigate inference latency in generating target non-core response tokens, we incorporate a semi-implicit CoT scheme into XS-CoT, which progressively compresses the first three types of intermediate reasoning tokens while retaining global reasoning logic during training. By leveraging the robust reasoning capabilities of the core language, XS-CoT improves responses for non-core languages by up to 45\% in GPT-4 score when compared to direct supervised fine-tuning on two representative SLLMs, Qwen2-Audio and SALMONN. Moreover, the semi-implicit XS-CoT reduces token delay by more than 50\% with a slight drop in GPT-4 scores. Importantly, XS-CoT requires only a small amount of high-quality training data for non-core languages by leveraging the reasoning capabilities of core languages. To support training, we also develop a data pipeline and open-source speech instruction-following datasets in Japanese, German, and French.
#### End-to-end Audio Deepfake Detection from RAW Waveforms: a RawNet-Based Approach with Cross-Dataset Evaluation
 - **Authors:** Andrea Di Pierno (1 and 2), Luca Guarnera (2), Dario Allegra (2), Sebastiano Battiato (2) ((1) IMT School of Advanced Studies, Lucca, Italy, (2) Department of Mathematics and Computer Science, University of Catania, Italy)
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.20923

 - **Pdf link:** https://arxiv.org/pdf/2504.20923

 - **Abstract**
 Audio deepfakes represent a growing threat to digital security and trust, leveraging advanced generative models to produce synthetic speech that closely mimics real human voices. Detecting such manipulations is especially challenging under open-world conditions, where spoofing methods encountered during testing may differ from those seen during training. In this work, we propose an end-to-end deep learning framework for audio deepfake detection that operates directly on raw waveforms. Our model, RawNetLite, is a lightweight convolutional-recurrent architecture designed to capture both spectral and temporal features without handcrafted preprocessing. To enhance robustness, we introduce a training strategy that combines data from multiple domains and adopts Focal Loss to emphasize difficult or ambiguous samples. We further demonstrate that incorporating codec-based manipulations and applying waveform-level audio augmentations (e.g., pitch shifting, noise, and time stretching) leads to significant generalization improvements under realistic acoustic conditions. The proposed model achieves over 99.7% F1 and 0.25% EER on in-domain data (FakeOrReal), and up to 83.4% F1 with 16.4% EER on a challenging out-of-distribution test set (AVSpoof2021 + CodecFake). These findings highlight the importance of diverse training data, tailored objective functions and audio augmentations in building resilient and generalizable audio forgery detectors. Code and pretrained models are available at this https URL.


by Zyzzyva0381 (Windy). 


2025-04-30
