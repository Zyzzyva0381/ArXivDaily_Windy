# Showing new listings for Friday, 27 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 9papers 
#### CodecSlime: Temporal Redundancy Compression of Neural Speech Codec via Dynamic Frame Rate
 - **Authors:** Hankun Wang, Yiwei Guo, Chongtian Shao, Bohan Li, Xie Chen, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.21074

 - **Pdf link:** https://arxiv.org/pdf/2506.21074

 - **Abstract**
 Neural speech codecs have been widely used in audio compression and various downstream tasks. Current mainstream codecs are fixed-frame-rate (FFR), which allocate the same number of tokens to every equal-duration slice. However, speech is inherently non-uniform in temporal information density. As a result, many tokens are wasted on steady-state segments like long vowels and silences. To address this mismatch, we present CodecSlime, a plugin-style method for compressing temporal redundancy through supporting dynamic frame rate (DFR) on neural speech codecs for the first time. Our method is unsupervised and architecture-agnostic, combining two key innovations, ScheDFR and Melt-and-Cool, for adapting inference and training, respectively. When integrated into a typical VQ-GAN codec backbone and operating at 40 Hz DFR ($\approx$ 600 bps), the reconstruction WER of CodecSlime is reduced by up to 46% relative to conventional FFR baselines with the same model architecture and similar bitrates, while other metrics are also competitive. CodecSlime also enables flexible trade-offs between reconstruction quality and bitrate: a single model supports inference at multiple frame rates and consistently outperforms FFR models at the corresponding frame rates. Audio samples are available at this https URL.
#### Post-training for Deepfake Speech Detection
 - **Authors:** Wanying Ge, Xin Wang, Xuechen Liu, Junichi Yamagishi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21090

 - **Pdf link:** https://arxiv.org/pdf/2506.21090

 - **Abstract**
 We introduce a post-training approach that adapts self-supervised learning (SSL) models for deepfake speech detection by bridging the gap between general pre-training and domain-specific fine-tuning. We present AntiDeepfake models, a series of post-trained models developed using a large-scale multilingual speech dataset containing over 56,000 hours of genuine speech and 18,000 hours of speech with various artifacts in over one hundred languages. Experimental results show that the post-trained models already exhibit strong robustness and generalization to unseen deepfake speech. When they are further fine-tuned on the Deepfake-Eval-2024 dataset, these models consistently surpass existing state-of-the-art detectors that do not leverage post-training. Model checkpoints and source code are available online.
#### Performance improvement of spatial semantic segmentation with enriched audio features and agent-based error correction for DCASE 2025 Challenge Task 4
 - **Authors:** Jongyeon Park, Joonhee Lee, Do-Hyeon Lim, Hong Kook Kim, Hyeongcheol Geum, Jeong Eun Lim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2506.21174

 - **Pdf link:** https://arxiv.org/pdf/2506.21174

 - **Abstract**
 This technical report presents submission systems for Task 4 of the DCASE 2025 Challenge. This model incorporates additional audio features (spectral roll-off and chroma features) into the embedding feature extracted from the mel-spectral feature to im-prove the classification capabilities of an audio-tagging model in the spatial semantic segmentation of sound scenes (S5) system. This approach is motivated by the fact that mixed audio often contains subtle cues that are difficult to capture with mel-spectrograms alone. Thus, these additional features offer alterna-tive perspectives for the model. Second, an agent-based label correction system is applied to the outputs processed by the S5 system. This system reduces false positives, improving the final class-aware signal-to-distortion ratio improvement (CA-SDRi) metric. Finally, we refine the training dataset to enhance the classi-fication accuracy of low-performing classes by removing irrele-vant samples and incorporating external data. That is, audio mix-tures are generated from a limited number of data points; thus, even a small number of out-of-class data points could degrade model performance. The experiments demonstrate that the submit-ted systems employing these approaches relatively improve CA-SDRi by up to 14.7% compared to the baseline of DCASE 2025 Challenge Task 4.
#### Hybrid Deep Learning and Signal Processing for Arabic Dialect Recognition in Low-Resource Settings
 - **Authors:** Ghazal Al-Shwayyat, Omer Nezih Gerek
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.21386

 - **Pdf link:** https://arxiv.org/pdf/2506.21386

 - **Abstract**
 Arabic dialect recognition presents a significant challenge in speech technology due to the linguistic diversity of Arabic and the scarcity of large annotated datasets, particularly for underrepresented dialects. This research investigates hybrid modeling strategies that integrate classical signal processing techniques with deep learning architectures to address this problem in low-resource scenarios. Two hybrid models were developed and evaluated: (1) Mel-Frequency Cepstral Coefficients (MFCC) combined with a Convolutional Neural Network (CNN), and (2) Discrete Wavelet Transform (DWT) features combined with a Recurrent Neural Network (RNN). The models were trained on a dialect-filtered subset of the Common Voice Arabic dataset, with dialect labels assigned based on speaker metadata. Experimental results demonstrate that the MFCC + CNN architecture achieved superior performance, with an accuracy of 91.2% and strong precision, recall, and F1-scores, significantly outperforming the Wavelet + RNN configuration, which achieved an accuracy of 66.5%. These findings highlight the effectiveness of leveraging spectral features with convolutional models for Arabic dialect recognition, especially when working with limited labeled data. The study also identifies limitations related to dataset size, potential regional overlaps in labeling, and model optimization, providing a roadmap for future research. Recommendations for further improvement include the adoption of larger annotated corpora, integration of self-supervised learning techniques, and exploration of advanced neural architectures such as Transformers. Overall, this research establishes a strong baseline for future developments in Arabic dialect recognition within resource-constrained environments.
#### ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing
 - **Authors:** Huadai Liu, Jialei Wang, Kaicheng Luo, Wen Wang, Qian Chen, Zhou Zhao, Wei Xue
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.21448

 - **Pdf link:** https://arxiv.org/pdf/2506.21448

 - **Abstract**
 While end-to-end video-to-audio generation has greatly improved, producing high-fidelity audio that authentically captures the nuances of visual content remains challenging. Like professionals in the creative industries, such generation requires sophisticated reasoning about items such as visual dynamics, acoustic environments, and temporal relationships. We present \textbf{ThinkSound}, a novel framework that leverages Chain-of-Thought (CoT) reasoning to enable stepwise, interactive audio generation and editing for videos. Our approach decomposes the process into three complementary stages: foundational foley generation that creates semantically coherent soundscapes, interactive object-centric refinement through precise user interactions, and targeted editing guided by natural language instructions. At each stage, a multimodal large language model generates contextually aligned CoT reasoning that guides a unified audio foundation model. Furthermore, we introduce \textbf{AudioCoT}, a comprehensive dataset with structured reasoning annotations that establishes connections between visual content, textual descriptions, and sound synthesis. Experiments demonstrate that ThinkSound achieves state-of-the-art performance in video-to-audio generation across both audio metrics and CoT metrics and excels in out-of-distribution Movie Gen Audio benchmark. The demo page is available at this https URL.
#### A Multi-Stage Framework for Multimodal Controllable Speech Synthesis
 - **Authors:** Rui Niu, Weihao Wu, Jie Chen, Long Ma, Zhiyong Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.20945

 - **Pdf link:** https://arxiv.org/pdf/2506.20945

 - **Abstract**
 Controllable speech synthesis aims to control the style of generated speech using reference input, which can be of various modalities. Existing face-based methods struggle with robustness and generalization due to data quality constraints, while text prompt methods offer limited diversity and fine-grained control. Although multimodal approaches aim to integrate various modalities, their reliance on fully matched training data significantly constrains their performance and applicability. This paper proposes a 3-stage multimodal controllable speech synthesis framework to address these challenges. For face encoder, we use supervised learning and knowledge distillation to tackle generalization issues. Furthermore, the text encoder is trained on both text-face and text-speech data to enhance the diversity of the generated speech. Experimental results demonstrate that this method outperforms single-modal baseline methods in both face based and text prompt based speech synthesis, highlighting its effectiveness in generating high-quality speech.
#### Prompt-Guided Turn-Taking Prediction
 - **Authors:** Koji Inoue, Mikey Elmers, Yahui Fu, Zi Haur Pang, Divesh Lala, Keiko Ochi, Tatsuya Kawahara
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21191

 - **Pdf link:** https://arxiv.org/pdf/2506.21191

 - **Abstract**
 Turn-taking prediction models are essential components in spoken dialogue systems and conversational robots. Recent approaches leverage transformer-based architectures to predict speech activity continuously and in real-time. In this study, we propose a novel model that enables turn-taking prediction to be dynamically controlled via textual prompts. This approach allows intuitive and explicit control through instructions such as "faster" or "calmer" adapting dynamically to conversational partners and contexts. The proposed model builds upon a transformer-based voice activity projection (VAP) model, incorporating textual prompt embeddings into both channel-wise transformers and a cross-channel transformer. We evaluated the feasibility of our approach using over 950 hours of human-human spoken dialogue data. Since textual prompt data for the proposed approach was not available in existing datasets, we utilized a large language model (LLM) to generate synthetic prompt sentences. Experimental results demonstrated that the proposed model improved prediction accuracy and effectively varied turn-taking timing behaviors according to the textual prompts.
#### Integrating Vehicle Acoustic Data for Enhanced Urban Traffic Management: A Study on Speed Classification in Suzhou
 - **Authors:** Pengfei Fan, Yuli Zhang, Xinheng Wang, Ruiyuan Jiang, Hankang Gu, Dongyao Jia, Shangbo Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21269

 - **Pdf link:** https://arxiv.org/pdf/2506.21269

 - **Abstract**
 This study presents and publicly releases the Suzhou Urban Road Acoustic Dataset (SZUR-Acoustic Dataset), which is accompanied by comprehensive data-acquisition protocols and annotation guidelines to ensure transparency and reproducibility of the experimental workflow. To model the coupling between vehicular noise and driving speed, we propose a bimodal-feature-fusion deep convolutional neural network (BMCNN). During preprocessing, an adaptive denoising and normalization strategy is applied to suppress environmental background interference; in the network architecture, parallel branches extract Mel-frequency cepstral coefficients (MFCCs) and wavelet-packet energy features, which are subsequently fused via a cross-modal attention mechanism in the intermediate feature space to fully exploit time-frequency information. Experimental results demonstrate that BMCNN achieves a classification accuracy of 87.56% on the SZUR-Acoustic Dataset and 96.28% on the public IDMT-Traffic dataset. Ablation studies and robustness tests on the Suzhou dataset further validate the contributions of each module to performance improvement and overfitting mitigation. The proposed acoustics-based speed classification method can be integrated into smart-city traffic management systems for real-time noise monitoring and speed estimation, thereby optimizing traffic flow control, reducing roadside noise pollution, and supporting sustainable urban planning.
#### Aligning Spoken Dialogue Models from User Interactions
 - **Authors:** Anne Wu, Laurent Mazaré, Neil Zeghidour, Alexandre Défossez
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21463

 - **Pdf link:** https://arxiv.org/pdf/2506.21463

 - **Abstract**
 We propose a novel preference alignment framework for improving spoken dialogue models on real-time conversations from user interactions. Current preference learning methods primarily focus on text-based language models, and are not directly suited to the complexities of real-time speech interactions, with richer dynamics (e.g. interruption, interjection) and no explicit segmentation between speaker this http URL create a large-scale dataset of more than 150,000 preference pairs from raw multi-turn speech conversations, annotated with AI feedback, to cover preferences over both linguistic content and temporal context variations. We leverage offline alignment methods to finetune a full-duplex autoregressive speech-to-speech model. Extensive experiments demonstrate that feedback on generic conversations can be consistently effective in improving spoken dialogue models to produce more factual, safer and more contextually aligned interactions. We deploy the finetuned model and conduct holistic human evaluations to assess the impact beyond single-turn conversations. Our findings shed light on the importance of a well-calibrated balance among various dynamics, crucial for natural real-time speech dialogue systems.


by Zyzzyva0381 (Windy). 


2025-06-27
