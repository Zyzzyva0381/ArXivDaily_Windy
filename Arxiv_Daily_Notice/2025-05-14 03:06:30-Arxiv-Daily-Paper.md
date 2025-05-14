# Showing new listings for Wednesday, 14 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 5papers 
#### MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder
 - **Authors:** Bowen Zhang, Congchao Guo, Geng Yang, Hang Yu, Haozhe Zhang, Heidi Lei, Jialong Mai, Junjie Yan, Kaiyue Yang, Mingqi Yang, Peikai Huang, Ruiyang Jin, Sitan Jiang, Weihua Cheng, Yawei Li, Yichen Xiao, Yiying Zhou, Yongmao Zhang, Yuan Lu, Yucen He
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.07916

 - **Pdf link:** https://arxiv.org/pdf/2505.07916

 - **Abstract**
 We introduce MiniMax-Speech, an autoregressive Transformer-based Text-to-Speech (TTS) model that generates high-quality speech. A key innovation is our learnable speaker encoder, which extracts timbre features from a reference audio without requiring its transcription. This enables MiniMax-Speech to produce highly expressive speech with timbre consistent with the reference in a zero-shot manner, while also supporting one-shot voice cloning with exceptionally high similarity to the reference voice. In addition, the overall quality of the synthesized audio is enhanced through the proposed Flow-VAE. Our model supports 32 languages and demonstrates excellent performance across multiple objective and subjective evaluations metrics. Notably, it achieves state-of-the-art (SOTA) results on objective voice cloning metrics (Word Error Rate and Speaker Similarity) and has secured the top position on the public TTS Arena leaderboard. Another key strength of MiniMax-Speech, granted by the robust and disentangled representations from the speaker encoder, is its extensibility without modifying the base model, enabling various applications such as: arbitrary voice emotion control via LoRA; text to voice (T2V) by synthesizing timbre features directly from text description; and professional voice cloning (PVC) by fine-tuning timbre features with additional data. We encourage readers to visit this https URL for more examples.
#### Investigating self-supervised features for expressive, multilingual voice conversion
 - **Authors:** Álvaro Martín-Cortinas, Daniel Sáez-Trigueros, Grzegorz Beringer, Iván Vallés-Pérez, Roberto Barra-Chicote, Biel Tura-Vecino, Adam Gabryś, Piotr Bilinski, Thomas Merritt, Jaime Lorenzo-Trueba
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.08278

 - **Pdf link:** https://arxiv.org/pdf/2505.08278

 - **Abstract**
 Voice conversion (VC) systems are widely used for several applications, from speaker anonymisation to personalised speech synthesis. Supervised approaches learn a mapping between different speakers using parallel data, which is expensive to produce. Unsupervised approaches are typically trained to reconstruct the input signal, which is composed of the content and the speaker information. Disentangling these components is a challenge and often leads to speaker leakage or prosodic information removal. In this paper, we explore voice conversion by leveraging the potential of self-supervised learning (SSL). A combination of the latent representations of SSL models, concatenated with speaker embeddings, is fed to a vocoder which is trained to reconstruct the input. Zero-shot voice conversion results show that this approach allows to keep the prosody and content of the source speaker while matching the speaker similarity of a VC system based on phonetic posteriorgrams (PPGs).
#### A Survey of Deep Learning for Complex Speech Spectrograms
 - **Authors:** Yuying Xie, Zheng-Hua Tan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2505.08694

 - **Pdf link:** https://arxiv.org/pdf/2505.08694

 - **Abstract**
 Recent advancements in deep learning have significantly impacted the field of speech signal processing, particularly in the analysis and manipulation of complex spectrograms. This survey provides a comprehensive overview of the state-of-the-art techniques leveraging deep neural networks for processing complex spectrograms, which encapsulate both magnitude and phase information. We begin by introducing complex spectrograms and their associated features for various speech processing tasks. Next, we explore the key components and architectures of complex-valued neural networks, which are specifically designed to handle complex-valued data and have been applied for complex spectrogram processing. We then discuss various training strategies and loss functions tailored for training neural networks to process and model complex spectrograms. The survey further examines key applications, including phase retrieval, speech enhancement, and speech separation, where deep learning has achieved significant progress by leveraging complex spectrograms or their derived feature representations. Additionally, we examine the intersection of complex spectrograms with generative models. This survey aims to serve as a valuable resource for researchers and practitioners in the field of speech signal processing and complex-valued neural networks.
#### Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities
 - **Authors:** George Saon, Avihu Dekel, Alexander Brooks, Tohru Nagano, Abraham Daniels, Aharon Satt, Ashish Mittal, Brian Kingsbury, David Haws, Edmilson Morais, Gakuto Kurata, Hagai Aronowitz, Ibrahim Ibrahim, Jeff Kuo, Kate Soule, Luis Lastras, Masayuki Suzuki, Ron Hoory, Samuel Thomas, Sashi Novitasari, Takashi Fukuda, Vishal Sunder, Xiaodong Cui, Zvi Kons
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.08699

 - **Pdf link:** https://arxiv.org/pdf/2505.08699

 - **Abstract**
 Granite-speech LLMs are compact and efficient speech language models specifically designed for English ASR and automatic speech translation (AST). The models were trained by modality aligning the 2B and 8B parameter variants of granite-3.3-instruct to speech on publicly available open-source corpora containing audio inputs and text targets consisting of either human transcripts for ASR or automatically generated translations for AST. Comprehensive benchmarking shows that on English ASR, which was our primary focus, they outperform several competitors' models that were trained on orders of magnitude more proprietary data, and they keep pace on English-to-X AST for major European languages, Japanese, and Chinese. The speech-specific components are: a conformer acoustic encoder using block attention and self-conditioning trained with connectionist temporal classification, a windowed query-transformer speech modality adapter used to do temporal downsampling of the acoustic embeddings and map them to the LLM text embedding space, and LoRA adapters to further fine-tune the text LLM. Granite-speech-3.3 operates in two modes: in speech mode, it performs ASR and AST by activating the encoder, projector, and LoRA adapters; in text mode, it calls the underlying granite-3.3-instruct model directly (without LoRA), essentially preserving all the text LLM capabilities and safety. Both models are freely available on HuggingFace (this https URL and this https URL) and can be used for both research and commercial purposes under a permissive Apache 2.0 license.
#### Unveiling the Best Practices for Applying Speech Foundation Models to Speech Intelligibility Prediction for Hearing-Impaired People
 - **Authors:** Haoshuai Zhou, Boxuan Cao, Changgeng Mo, Linkai Li, Shan Xiang Wang
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.08215

 - **Pdf link:** https://arxiv.org/pdf/2505.08215

 - **Abstract**
 Speech foundation models (SFMs) have demonstrated strong performance across a variety of downstream tasks, including speech intelligibility prediction for hearing-impaired people (SIP-HI). However, optimizing SFMs for SIP-HI has been insufficiently explored. In this paper, we conduct a comprehensive study to identify key design factors affecting SIP-HI performance with 5 SFMs, focusing on encoder layer selection, prediction head architecture, and ensemble configurations. Our findings show that, contrary to traditional use-all-layers methods, selecting a single encoder layer yields better results. Additionally, temporal modeling is crucial for effective prediction heads. We also demonstrate that ensembling multiple SFMs improves performance, with stronger individual models providing greater benefit. Finally, we explore the relationship between key SFM attributes and their impact on SIP-HI performance. Our study offers practical insights into effectively adapting SFMs for speech intelligibility prediction for hearing-impaired populations.


by Zyzzyva0381 (Windy). 


2025-05-14
