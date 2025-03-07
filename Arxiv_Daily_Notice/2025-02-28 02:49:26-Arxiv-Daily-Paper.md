# Showing new listings for Thursday, 27 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 4papers 
#### Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis
 - **Authors:** Ziyue Jiang, Yi Ren, Ruiqi Li, Shengpeng Ji, Zhenhui Ye, Chen Zhang, Bai Jionghao, Xiaoda Yang, Jialong Zuo, Yu Zhang, Rui Liu, Xiang Yin, Zhou Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.18924

 - **Pdf link:** https://arxiv.org/pdf/2502.18924

 - **Abstract**
 While recent zero-shot text-to-speech (TTS) models have significantly improved speech quality and expressiveness, mainstream systems still suffer from issues related to speech-text alignment modeling: 1) models without explicit speech-text alignment modeling exhibit less robustness, especially for hard sentences in practical applications; 2) predefined alignment-based models suffer from naturalness constraints of forced alignments. This paper introduces \textit{S-DiT}, a TTS system featuring an innovative sparse alignment algorithm that guides the latent diffusion transformer (DiT). Specifically, we provide sparse alignment boundaries to S-DiT to reduce the difficulty of alignment learning without limiting the search space, thereby achieving high naturalness. Moreover, we employ a multi-condition classifier-free guidance strategy for accent intensity adjustment and adopt the piecewise rectified flow technique to accelerate the generation process. Experiments demonstrate that S-DiT achieves state-of-the-art zero-shot TTS speech quality and supports highly flexible control over accent intensity. Notably, our system can generate high-quality one-minute speech with only 8 sampling steps. Audio samples are available at this https URL.
#### Clip-TTS: Contrastive Text-content and Mel-spectrogram, A High-Huality Text-to-Speech Method based on Contextual Semantic Understanding
 - **Authors:** Tianyun Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.18889

 - **Pdf link:** https://arxiv.org/pdf/2502.18889

 - **Abstract**
 Traditional text-to-speech (TTS) methods primarily focus on establishing a mapping between phonemes and mel-spectrograms. However, during the phoneme encoding stage, there is often a lack of real mel-spectrogram auxiliary information, which results in the encoding process lacking true semantic understanding. At the same time, traditional TTS systems often struggle to balance the inference speed of the model with the quality of the synthesized speech. Methods that generate high-quality synthesized speech tend to have slower inference speeds, while faster inference methods often sacrifice speech quality. In this paper, I propose Clip-TTS, a TTS method based on the Clip architecture. This method uses the Clip framework to establish a connection between text content and real mel-spectrograms during the text encoding stage, enabling the text encoder to directly learn the true semantics of the global context, thereby ensuring the quality of the synthesized speech. In terms of model architecture, I adopt the basic structure of Transformer, which allows Clip-TTS to achieve fast inference speeds. Experimental results show that on the LJSpeech and Baker datasets, the speech generated by Clip-TTS achieves state-of-the-art MOS scores, and it also performs excellently on multi-emotion this http URL samples are available at: this https URL.
#### CS-Dialogue: A 104-Hour Dataset of Spontaneous Mandarin-English Code-Switching Dialogues for Speech Recognition
 - **Authors:** Jiaming Zhou, Yujie Guo, Shiwan Zhao, Haoqin Sun, Hui Wang, Jiabei He, Aobo Kong, Shiyao Wang, Xi Yang, Yequan Wang, Yonghua Lin, Yong Qin
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.18913

 - **Pdf link:** https://arxiv.org/pdf/2502.18913

 - **Abstract**
 Code-switching (CS), the alternation between two or more languages within a single conversation, presents significant challenges for automatic speech recognition (ASR) systems. Existing Mandarin-English code-switching datasets often suffer from limitations in size, spontaneity, and the lack of full-length dialogue recordings with transcriptions, hindering the development of robust ASR models for real-world conversational scenarios. This paper introduces CS-Dialogue, a novel large-scale Mandarin-English code-switching speech dataset comprising 104 hours of spontaneous conversations from 200 speakers. Unlike previous datasets, CS-Dialogue provides full-length dialogue recordings with complete transcriptions, capturing naturalistic code-switching patterns in continuous speech. We describe the data collection and annotation processes, present detailed statistics of the dataset, and establish benchmark ASR performance using state-of-the-art models. Our experiments, using Transformer, Conformer, and Branchformer, demonstrate the challenges of code-switching ASR, and show that existing pre-trained models such as Whisper still have the space to improve. The CS-Dialogue dataset will be made freely available for all academic purposes.
#### DualSpec: Text-to-spatial-audio Generation via Dual-Spectrogram Guided Diffusion Model
 - **Authors:** Lei Zhao, Sizhou Chen, Linfeng Feng, Xiao-Lei Zhang, Xuelong Li
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.18952

 - **Pdf link:** https://arxiv.org/pdf/2502.18952

 - **Abstract**
 Text-to-audio (TTA), which generates audio signals from textual descriptions, has received huge attention in recent years. However, recent works focused on text to monaural audio only. As we know, spatial audio provides more immersive auditory experience than monaural audio, e.g. in virtual reality. To address this issue, we propose a text-to-spatial-audio (TTSA) generation framework named this http URL, it first trains variational autoencoders (VAEs) for extracting the latent acoustic representations from sound event audio. Then, given text that describes sound events and event directions, the proposed method uses the encoder of a pretrained large language model to transform the text into text features. Finally, it trains a diffusion model from the latent acoustic representations and text features for the spatial audio generation. In the inference stage, only the text description is needed to generate spatial audio. Particularly, to improve the synthesis quality and azimuth accuracy of the spatial sound events simultaneously, we propose to use two kinds of acoustic features. One is the Mel spectrograms which is good for improving the synthesis quality, and the other is the short-time Fourier transform spectrograms which is good at improving the azimuth accuracy. We provide a pipeline of constructing spatial audio dataset with text prompts, for the training of the VAEs and diffusion model. We also introduce new spatial-aware evaluation metrics to quantify the azimuth errors of the generated spatial audio recordings. Experimental results demonstrate that the proposed method can generate spatial audio with high directional and event consistency.


by Zyzzyva0381 (Windy). 


2025-02-28
