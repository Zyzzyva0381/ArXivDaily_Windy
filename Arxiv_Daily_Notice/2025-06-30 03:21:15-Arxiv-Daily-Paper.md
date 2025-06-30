# Showing new listings for Monday, 30 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 18papers 
#### HighRateMOS: Sampling-Rate Aware Modeling for Speech Quality Assessment
 - **Authors:** Wenze Ren, Yi-Cheng Lin, Wen-Chin Huang, Ryandhimas E. Zezario, Szu-Wei Fu, Sung-Feng Huang, Erica Cooper, Haibin Wu, Hung-Yu Wei, Hsin-Min Wang, Hung-yi Lee, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21951

 - **Pdf link:** https://arxiv.org/pdf/2506.21951

 - **Abstract**
 Modern speech quality prediction models are trained on audio data resampled to a specific sampling rate. When faced with higher-rate audio at test time, these models can produce biased scores. We introduce HighRateMOS, the first non-intrusive mean opinion score (MOS) model that explicitly considers sampling rate. HighRateMOS ensembles three model variants that exploit the following information: (i) a learnable embedding of speech sampling rate, (ii) Wav2vec 2.0 self-supervised embeddings, (iii) multi-scale CNN spectral features, and (iv) MFCC features. In AudioMOS 2025 Track3, HighRateMOS ranked first in five out of eight metrics. Our experiments confirm that modeling the sampling rate directly leads to more robust and sampling-rate-agnostic speech quality predictions.
#### WTFormer: A Wavelet Conformer Network for MIMO Speech Enhancement with Spatial Cues Peservation
 - **Authors:** Lu Han, Junqi Zhao, Renhua Peng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.22001

 - **Pdf link:** https://arxiv.org/pdf/2506.22001

 - **Abstract**
 Current multi-channel speech enhancement systems mainly adopt single-output architecture, which face significant challenges in preserving spatio-temporal signal integrity during multiple-input multiple-output (MIMO) processing. To address this limitation, we propose a novel neural network, termed WTFormer, for MIMO speech enhancement that leverages the multi-resolution characteristics of wavelet transform and multi-dimensional collaborative attention to effectively capture globally distributed spatial features, while using Conformer for time-frequency modeling. A multi task loss strategy accompanying MUSIC algorithm is further proposed for optimization training to protect spatial information to the greatest extent. Experimental results on the LibriSpeech dataset show that WTFormer can achieve comparable denoising performance to advanced systems while preserving more spatial information with only 0.98M parameters.
#### Cross-lingual Data Selection Using Clip-level Acoustic Similarity for Enhancing Low-resource Automatic Speech Recognition
 - **Authors:** Shunsuke Mitsumori, Sara Kashiwagi, Keitaro Tanaka, Shigeo Morishima
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22194

 - **Pdf link:** https://arxiv.org/pdf/2506.22194

 - **Abstract**
 This paper presents a novel donor data selection method to enhance low-resource automatic speech recognition (ASR). While ASR performs well in high-resource languages, its accuracy declines in low-resource settings due to limited training data. A common solution is to leverage multilingual self-supervised learning (SSL) models with donor languages. However, existing methods rely on language-level similarity, overlooking clip-level variations. To address this limitation, we propose clip-wise acoustic token distribution similarity (CATDS), a fine-grained selection method that identifies acoustically relevant donor clips for better alignment with the target language. Unlike existing clip-level selection methods, our method aligns with the representation of SSL models and offers more challenging yet valuable samples. Experimental results show that CATDS outperforms traditional selection methods and can even utilize donor languages previously considered detrimental.
#### DiffSoundStream: Efficient Speech Tokenization via Diffusion Decoding
 - **Authors:** Yang Yang, Yunpeng Li, George Sung, Shao-Fu Shih, Craig Dooley, Alessio Centazzo, Ramanan Rajeswaran
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2506.22362

 - **Pdf link:** https://arxiv.org/pdf/2506.22362

 - **Abstract**
 Token-based language modeling is a prominent approach for speech generation, where tokens are obtained by quantizing features from self-supervised learning (SSL) models and extracting codes from neural speech codecs, generally referred to as semantic tokens and acoustic tokens. These tokens are often modeled autoregressively, with the inference speed being constrained by the token rate. In this work, we propose DiffSoundStream, a solution that improves the efficiency of speech tokenization in non-streaming scenarios through two techniques: (1) conditioning the neural codec on semantic tokens to minimize redundancy between semantic and acoustic tokens, and (2) leveraging latent diffusion models to synthesize high-quality waveforms from semantic and coarse-level acoustic tokens. Experiments show that at 50 tokens per second, DiffSoundStream achieves speech quality on par with a standard SoundStream model operating at twice the token rate. Additionally, we achieve step-size distillation using just four diffusion sampling steps with only a minor quality loss.
#### Efficient Multilingual ASR Finetuning via LoRA Language Experts
 - **Authors:** Jiahong Li, Yiwen Shao, Jianheng Zhuo, Chenda Li, Liliang Tang, Dong Yu, Yanmin Qian
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21555

 - **Pdf link:** https://arxiv.org/pdf/2506.21555

 - **Abstract**
 Recent advancements in deep learning have significantly enhanced multilingual automatic speech recognition (ASR) due to the development of advanced model architectures and available large-scale multilingual datasets. Despite that, multilingual ASR still suffers from the curse of multilinguality in that different languages tend to interfere with each other, making it difficult for the ASR model to identify multiple languages effectively while sharing model capacity across them. This paper proposes an efficient finetuning framework for customized multilingual ASR via prepared LoRA language experts based on Whisper. Through LoRA expert fusion or knowledge distillation, our approach achieves better recognition performance on target languages than standard fine-tuning methods. Experimental results demonstrate that the proposed models yield approximately 10\% and 15\% relative performance gains in language-aware and language-agnostic scenarios, respectively.
#### Adapting Whisper for Parameter-efficient Code-Switching Speech Recognition via Soft Prompt Tuning
 - **Authors:** Hongli Yang, Yizhou Peng, Hao Huang, Sheng Li
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21576

 - **Pdf link:** https://arxiv.org/pdf/2506.21576

 - **Abstract**
 Large-scale multilingual ASR models like Whisper excel in high-resource settings but face challenges in low-resource scenarios, such as rare languages and code-switching (CS), due to computational costs and catastrophic forgetting. We explore Soft Prompt Tuning (SPT), a parameter-efficient method to enhance CS ASR while preserving prior knowledge. We evaluate two strategies: (1) full fine-tuning (FFT) of both soft prompts and the entire Whisper model, demonstrating improved cross-lingual capabilities compared to traditional methods, and (2) adhering to SPT's original design by freezing model parameters and only training soft prompts. Additionally, we introduce SPT4ASR, a combination of different SPT variants. Experiments on the SEAME and ASRU2019 datasets show that deep prompt tuning is the most effective SPT approach, and our SPT4ASR methods achieve further error reductions in CS ASR, maintaining parameter efficiency similar to LoRA, without degrading performance on existing languages.
#### Language-Aware Prompt Tuning for Parameter-Efficient Seamless Language Expansion in Multilingual ASR
 - **Authors:** Hongli Yang, Sheng Li, Hao Huang, Ayiduosi Tuohan, Yizhou Peng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21577

 - **Pdf link:** https://arxiv.org/pdf/2506.21577

 - **Abstract**
 Recent advancements in multilingual automatic speech recognition (ASR) have been driven by large-scale end-to-end models like Whisper. However, challenges such as language interference and expanding to unseen languages (language expansion) without degrading performance persist. This paper addresses these with three contributions: 1) Entire Soft Prompt Tuning (Entire SPT), which applies soft prompts to both the encoder and decoder, enhancing feature extraction and decoding; 2) Language-Aware Prompt Tuning (LAPT), which leverages cross-lingual similarities to encode shared and language-specific features using lightweight prompt matrices; 3) SPT-Whisper, a toolkit that integrates SPT into Whisper and enables efficient continual learning. Experiments across three languages from FLEURS demonstrate that Entire SPT and LAPT outperform Decoder SPT by 5.0% and 16.0% in language expansion tasks, respectively, providing an efficient solution for dynamic, multilingual ASR models with minimal computational overhead.
#### ChildGuard: A Specialized Dataset for Combatting Child-Targeted Hate Speech
 - **Authors:** Gautam Siddharth Kashyap, Mohammad Anas Azeez, Rafiq Ali, Zohaib Hasan Siddiqui, Jiechao Gao, Usman Naseem
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21613

 - **Pdf link:** https://arxiv.org/pdf/2506.21613

 - **Abstract**
 The increasing prevalence of child-targeted hate speech online underscores the urgent need for specialized datasets to address this critical issue. Existing hate speech datasets lack agespecific annotations, fail to capture nuanced contexts, and overlook the unique emotional impact on children. To bridge this gap, we introduce ChildGuard1, a curated dataset derived from existing corpora and enriched with child-specific annotations. ChildGuard captures diverse contexts of child-targeted hate speech, spanning age groups. We benchmark existing state-of-the-art hate speech detection methods, including Large Language Models (LLMs), and assess their effectiveness in detecting and contextualizing child-targeted hate speech. To foster further research in this area, we publicly release ChildGuard, providing a robust foundation for developing improved methods to detect and mitigate such harm.
#### IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech
 - **Authors:** Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21619

 - **Pdf link:** https://arxiv.org/pdf/2506.21619

 - **Abstract**
 Large-scale text-to-speech (TTS) models are typically categorized into autoregressive and non-autoregressive systems. Although autoregressive systems exhibit certain advantages in speech naturalness, their token-by-token generation mechanism makes it difficult to precisely control the duration of synthesized speech. This is a key limitation in applications such as video dubbing that require strict audio-visual synchronization. This paper introduces IndexTTS2, which proposes a novel and autoregressive-model-friendly method for speech duration control. The method supports two generation modes: one allows explicit specification of the number of generated tokens for precise duration control; the other does not require manual input and lets the model freely generate speech while preserving prosodic characteristics from the input prompt. Furthermore, IndexTTS2 achieves disentanglement between emotional expression and speaker identity, enabling independent control of timbre and emotion. In the zero-shot setting, the model can perfectly reproduce the emotional characteristics of the input prompt. Users may also provide a separate emotion prompt, even from a different speaker, allowing the model to reconstruct the target timbre while conveying the desired emotion. To enhance clarity during strong emotional expressions, we incorporate GPT latent representations to improve speech stability. Meanwhile, to lower the barrier for emotion control, we design a soft instruction mechanism based on textual descriptions by fine-tuning Qwen3. This enables effective guidance of speech generation with desired emotional tendencies using natural language input. Experimental results demonstrate that IndexTTS2 outperforms existing state-of-the-art zero-shot TTS models in word error rate, speaker similarity, and emotional fidelity.
#### Adapting Foundation Speech Recognition Models to Impaired Speech: A Semantic Re-chaining Approach for Personalization of German Speech
 - **Authors:** Niclas Pokel, PehuÃ©n Moure, Roman Boehringer, Yingqiang Gao
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21622

 - **Pdf link:** https://arxiv.org/pdf/2506.21622

 - **Abstract**
 Speech impairments caused by conditions such as cerebral palsy or genetic disorders pose significant challenges for automatic speech recognition (ASR) systems. Despite recent advances, ASR models like Whisper struggle with non-normative speech due to limited training data and the difficulty of collecting and annotating non-normative speech samples. In this work, we propose a practical and lightweight pipeline to personalize ASR models, formalizing the selection of words and enriching a small, speech-impaired dataset with semantic coherence. Applied to data from a child with a structural speech impairment, our approach shows promising improvements in transcription quality, demonstrating the potential to reduce communication barriers for individuals with atypical speech patterns.
#### Identifying Speaker Information in Feed-Forward Layers of Self-Supervised Speech Transformers
 - **Authors:** Tzu-Quan Lin, Hsi-Chun Cheng, Hung-yi Lee, Hao Tang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21712

 - **Pdf link:** https://arxiv.org/pdf/2506.21712

 - **Abstract**
 In recent years, the impact of self-supervised speech Transformers has extended to speaker-related applications. However, little research has explored how these models encode speaker information. In this work, we address this gap by identifying neurons in the feed-forward layers that are correlated with speaker information. Specifically, we analyze neurons associated with k-means clusters of self-supervised features and i-vectors. Our analysis reveals that these clusters correspond to broad phonetic and gender classes, making them suitable for identifying neurons that represent speakers. By protecting these neurons during pruning, we can significantly preserve performance on speaker-related task, demonstrating their crucial role in encoding speaker information.
#### Explainable anomaly detection for sound spectrograms using pooling statistics with quantile differences
 - **Authors:** Nicolas Thewes, Philipp Steinhauer, Patrick Trampert, Markus Pauly, Georg Schneider
 - **Subjects:** Subjects:
Applications (stat.AP); Sound (cs.SD); Audio and Speech Processing (eess.AS); Computation (stat.CO)
 - **Arxiv link:** https://arxiv.org/abs/2506.21921

 - **Pdf link:** https://arxiv.org/pdf/2506.21921

 - **Abstract**
 Anomaly detection is the task of identifying rarely occurring (i.e. anormal or anomalous) samples that differ from almost all other samples in a dataset. As the patterns of anormal samples are usually not known a priori, this task is highly challenging. Consequently, anomaly detection lies between semi- and unsupervised learning. The detection of anomalies in sound data, often called 'ASD' (Anomalous Sound Detection), is a sub-field that deals with the identification of new and yet unknown effects in acoustic recordings. It is of great importance for various applications in Industry 4.0. Here, vibrational or acoustic data are typically obtained from standard sensor signals used for predictive maintenance. Examples cover machine condition monitoring or quality assurance to track the state of components or products. However, the use of intelligent algorithms remains a controversial topic. Management generally aims for cost-reduction and automation, while quality and maintenance experts emphasize the need for human expertise and comprehensible solutions. In this work, we present an anomaly detection approach specifically designed for spectrograms. The approach is based on statistical evaluations and is theoretically motivated. In addition, it features intrinsic explainability, making it particularly suitable for applications in industrial settings. Thus, this algorithm is of relevance for applications in which black-box algorithms are unwanted or unsuitable.
#### Analyzing and Fine-Tuning Whisper Models for Multilingual Pilot Speech Transcription in the Cockpit
 - **Authors:** Kartheek Kumar Reddy Nareddy, Sarah Ternus, Julia Niebling
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21990

 - **Pdf link:** https://arxiv.org/pdf/2506.21990

 - **Abstract**
 The developments in transformer encoder-decoder architectures have led to significant breakthroughs in machine translation, Automatic Speech Recognition (ASR), and instruction-based chat machines, among other applications. The pre-trained models were trained on vast amounts of generic data over a few epochs (fewer than five in most cases), resulting in their strong generalization capabilities. Nevertheless, the performance of these models does suffer when applied to niche domains like transcribing pilot speech in the cockpit, which involves a lot of specific vocabulary and multilingual conversations. This paper investigates and improves the transcription accuracy of cockpit conversations with Whisper models. We have collected around 85 minutes of cockpit simulator recordings and 130 minutes of interview recordings with pilots and manually labeled them. The speakers are middle aged men speaking both German and English. To improve the accuracy of transcriptions, we propose multiple normalization schemes to refine the transcripts and improve Word Error Rate (WER). We then employ fine-tuning to enhance ASR performance, utilizing performance-efficient fine-tuning with Low-Rank Adaptation (LoRA). Hereby, WER decreased from 68.49 \% (pretrained whisper Large model without normalization baseline) to 26.26\% (finetuned whisper Large model with the proposed normalization scheme).
#### Robust and Efficient Autoregressive Speech Synthesis with Dynamic Chunk-wise Prediction Policy
 - **Authors:** Bohan Li, Zhihan Li, Haoran Wang, Hanglei Zhang, Yiwei Guo, Hankun Wang, Xie Chen, Kai Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22023

 - **Pdf link:** https://arxiv.org/pdf/2506.22023

 - **Abstract**
 Recently, autoregressive (AR) language models have emerged as a dominant approach in speech synthesis, offering expressive generation and scalable training. However, conventional AR speech synthesis models relying on the next-token prediction paradigm often encounter significant challenges when handling long speech sequences. These models often struggle to construct stable frame-to-frame attention, leading to increased latency and degraded synthesis quality, thereby limiting their feasibility for real-time applications. To address these limitations, we introduce a novel dynamic chunk-wise autoregressive synthesis framework, termed DCAR, designed to enhance both efficiency and intelligibility robustness in AR speech generation. DCAR introduces a chunk-to-frame attention mechanism through training with multi-token prediction, enabling dynamic chunk prediction in variable speech contexts using a lightweight module trained on-policy. DCAR dynamically adjusts the token prediction span, significantly reducing the sequence length dependency while obtaining high synthesis quality. Comprehensive empirical evaluations demonstrate that DCAR substantially outperforms traditional next-token prediction models, achieving up to 72.27% intelligibility improvement and 2.61x inference speedup simultaneously on the test set. Furthermore, we conduct comprehensive analysis to support it as a versatile foundation for next-generation speech synthesis systems.
#### SAGE: Spliced-Audio Generated Data for Enhancing Foundational Models in Low-Resource Arabic-English Code-Switched Speech Recognition
 - **Authors:** Muhammad Umar Farooq, Oscar Saz
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22143

 - **Pdf link:** https://arxiv.org/pdf/2506.22143

 - **Abstract**
 This paper investigates the performance of various speech SSL models on dialectal Arabic (DA) and Arabic-English code-switched (CS) speech. To address data scarcity, a modified audio-splicing approach is introduced to generate artificial CS speech data. Fine-tuning an already fine-tuned SSL model with the proposed Spliced-Audio Generated (SAGE) data results in an absolute improvement on Word Error Rate (WER) of 7.8% on Arabic and English CS benchmarks. Additionally, an Experience Replay (ER) inspired approach is proposed to enhance generalisation across DA and CS speech while mitigating catastrophic forgetting. Integrating an out-of-domain 3-gram language model reduces the overall mean WER from 31.7% to 26.6%. Few-shot fine-tuning for code-switching benchmarks further improves WER by 4.9%. A WER of 31.1% on Arabic-English CS benchmarks surpasses large-scale multilingual models, including USM and Whisper-large-v2 (both over ten times larger) by an absolute margin of 5.5% and 8.4%, respectively.
#### Fine-Tuning MIDI-to-Audio Alignment using a Neural Network on Piano Roll and CQT Representations
 - **Authors:** Sebastian Murgul, Moritz Reiser, Michael Heizmann, Christoph Seibert
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22237

 - **Pdf link:** https://arxiv.org/pdf/2506.22237

 - **Abstract**
 In this paper, we present a neural network approach for synchronizing audio recordings of human piano performances with their corresponding loosely aligned MIDI files. The task is addressed using a Convolutional Recurrent Neural Network (CRNN) architecture, which effectively captures spectral and temporal features by processing an unaligned piano roll and a spectrogram as inputs to estimate the aligned piano roll. To train the network, we create a dataset of piano pieces with augmented MIDI files that simulate common human timing errors. The proposed model achieves up to 20% higher alignment accuracy than the industry-standard Dynamic Time Warping (DTW) method across various tolerance windows. Furthermore, integrating DTW with the CRNN yields additional improvements, offering enhanced robustness and consistency. These findings demonstrate the potential of neural networks in advancing state-of-the-art MIDI-to-audio alignment.
#### Reconstructing Intelligible Speech from the Pressure Sensor Data in HVACs
 - **Authors:** Tarikul Islam Tamiti, Biraj Joshi, Rida Hasan, Anomadarshi Barua
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22311

 - **Pdf link:** https://arxiv.org/pdf/2506.22311

 - **Abstract**
 Pressure sensors are an integrated component of modern Heating, Ventilation, and Air Conditioning (HVAC) systems. As these pressure sensors operate within the 0-10 Pa range, support high sampling frequencies of 0.5-2 kHz, and are often placed close to human proximity, they can be used to eavesdrop on confidential conversation, since human speech has a similar audible range of 0-10 Pa and a bandwidth of 4 kHz for intelligible quality. This paper presents WaLi, which reconstructs intelligible speech from the low-resolution and noisy pressure sensor data by providing the following technical contributions: (i) WaLi reconstructs intelligible speech from a minimum of 0.5 kHz sampling frequency of pressure sensors, whereas previous work can only detect hot words/phrases. WaLi uses complex-valued conformer and Complex Global Attention Block (CGAB) to capture inter-phoneme and intra-phoneme dependencies that exist in the low-resolution pressure sensor data. (ii) WaLi handles the transient noise injected from HVAC fans and duct vibrations, by reconstructing both the clean magnitude and phase of the missing frequencies of the low-frequency aliased components. Extensive measurement studies on real-world pressure sensors show an LSD of 1.24 and NISQA-MOS of 1.78 for 0.5 kHz to 8 kHz upsampling. We believe that such levels of accuracy pose a significant threat when viewed from a privacy perspective that has not been addressed before for pressure sensors.
#### A Practical Approach to Power Saving in Hearables Using Sub-Nyquist Sampling with Bandwidth Extension
 - **Authors:** Tarikul Islam Tamiti, Anomadarshi Barua
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.22321

 - **Pdf link:** https://arxiv.org/pdf/2506.22321

 - **Abstract**
 Hearables are wearable computers that are worn on the ear. Bone conduction microphones (BCMs) are used with air conduction microphones (ACMs) in hearables as a supporting modality for multimodal speech enhancement (SE) in noisy conditions. However, existing works don't consider the following practical aspects for low-power implementations on hearables: (i) They do not explore how lowering the sampling frequencies and bit resolutions in analog-to-digital converters (ADCs) of hearables jointly impact low-power processing and multimodal SE in terms of speech quality and intelligibility. (ii) They don't discuss how GAN-like audio quality can be achieved without using actual GAN discriminators. And (iii) They don't process signals from ACMs/BCMs at sub-Nyquist sampling rate because, in their frameworks, they lack a wideband reconstruction methodology from their narrowband parts. We propose SUBARU (\textbf{Sub}-Nyquist \textbf{A}udio \textbf{R}esolution \textbf{U}psampling), which achieves the following: SUBARU (i) intentionally uses sub-Nyquist sampling and low bit resolution in ADCs, achieving a 3.31x reduction in power consumption; (ii) introduces novel multi-scale and multi-period virtual discriminators, which achieve GAN-like audio quality without using GANs' adversarial training; and (iii) achieves streaming operations on mobile platforms and SE in in-the-wild noisy conditions with an inference time of 1.74ms and a memory footprint of less than 13.77MB.


by Zyzzyva0381 (Windy). 


2025-06-30
