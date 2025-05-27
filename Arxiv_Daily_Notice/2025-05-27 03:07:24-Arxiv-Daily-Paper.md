# Showing new listings for Monday, 26 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 23papers 
#### From Weak Labels to Strong Results: Utilizing 5,000 Hours of Noisy Classroom Transcripts with Minimal Accurate Data
 - **Authors:** Ahmed Adel Attia, Dorottya Demszky, Jing Liu, Carol Espy-Wilson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.17088

 - **Pdf link:** https://arxiv.org/pdf/2505.17088

 - **Abstract**
 Recent progress in speech recognition has relied on models trained on vast amounts of labeled data. However, classroom Automatic Speech Recognition (ASR) faces the real-world challenge of abundant weak transcripts paired with only a small amount of accurate, gold-standard data. In such low-resource settings, high transcription costs make re-transcription impractical. To address this, we ask: what is the best approach when abundant inexpensive weak transcripts coexist with limited gold-standard data, as is the case for classroom speech data? We propose Weakly Supervised Pretraining (WSP), a two-step process where models are first pretrained on weak transcripts in a supervised manner, and then fine-tuned on accurate data. Our results, based on both synthetic and real weak transcripts, show that WSP outperforms alternative methods, establishing it as an effective training methodology for low-resource ASR in real-world scenarios.
#### Voicing Personas: Rewriting Persona Descriptions into Style Prompts for Controllable Text-to-Speech
 - **Authors:** Yejin Lee, Jaehoon Kang, Kyuhong Shim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2505.17093

 - **Pdf link:** https://arxiv.org/pdf/2505.17093

 - **Abstract**
 In this paper, we propose a novel framework to control voice style in prompt-based, controllable text-to-speech systems by leveraging textual personas as voice style prompts. We present two persona rewriting strategies to transform generic persona descriptions into speech-oriented prompts, enabling fine-grained manipulation of prosodic attributes such as pitch, emotion, and speaking rate. Experimental results demonstrate that our methods enhance the naturalness, clarity, and consistency of synthesized speech. Finally, we analyze implicit social biases introduced by LLM-based rewriting, with a focus on gender. We underscore voice style as a crucial factor for persona-driven AI dialogue systems.
#### Speechless: Speech Instruction Training Without Speech for Low Resource Languages
 - **Authors:** Alan Dao (Gia Tuan Dao), Dinh Bach Vu, Huy Hoang Ha, Tuan Le Duc Anh, Shreyas Gopal, Yue Heng Yeo, Warren Keng Hoong Low, Eng Siong Chng, Jia Qi Yip
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.17417

 - **Pdf link:** https://arxiv.org/pdf/2505.17417

 - **Abstract**
 The rapid growth of voice assistants powered by large language models (LLM) has highlighted a need for speech instruction data to train these systems. Despite the abundance of speech recognition data, there is a notable scarcity of speech instruction data, which is essential for fine-tuning models to understand and execute spoken commands. Generating high-quality synthetic speech requires a good text-to-speech (TTS) model, which may not be available to low resource languages. Our novel approach addresses this challenge by halting synthesis at the semantic representation level, bypassing the need for TTS. We achieve this by aligning synthetic semantic representations with the pre-trained Whisper encoder, enabling an LLM to be fine-tuned on text instructions while maintaining the ability to understand spoken instructions during inference. This simplified training process is a promising approach to building voice assistant for low-resource languages.
#### Private kNN-VC: Interpretable Anonymization of Converted Speech
 - **Authors:** Carlos Franzreb, Arnab Das, Tim Polzehl, Sebastian Möller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.17584

 - **Pdf link:** https://arxiv.org/pdf/2505.17584

 - **Abstract**
 Speaker anonymization seeks to conceal a speaker's identity while preserving the utility of their speech. The achieved privacy is commonly evaluated with a speaker recognition model trained on anonymized speech. Although this represents a strong attack, it is unclear which aspects of speech are exploited to identify the speakers. Our research sets out to unveil these aspects. It starts with kNN-VC, a powerful voice conversion model that performs poorly as an anonymization system, presumably because of prosody leakage. To test this hypothesis, we extend kNN-VC with two interpretable components that anonymize the duration and variation of phones. These components increase privacy significantly, proving that the studied prosodic factors encode speaker identity and are exploited by the privacy attack. Additionally, we show that changes in the target selection algorithm considerably influence the outcome of the privacy attack.
#### Audio-to-Audio Emotion Conversion With Pitch And Duration Style Transfer
 - **Authors:** Soumya Dutta, Avni Jain, Sriram Ganapathy
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.17655

 - **Pdf link:** https://arxiv.org/pdf/2505.17655

 - **Abstract**
 Given a pair of source and reference speech recordings, audio-to-audio (A2A) style transfer involves the generation of an output speech that mimics the style characteristics of the reference while preserving the content and speaker attributes of the source. In this paper, we propose a novel framework, termed as A2A Zero-shot Emotion Style Transfer (A2A-ZEST), that enables the transfer of reference emotional attributes to the source while retaining its speaker and speech contents. The A2A-ZEST framework consists of an analysis-synthesis pipeline, where the analysis module decomposes speech into semantic tokens, speaker representations, and emotion embeddings. Using these representations, a pitch contour estimator and a duration predictor are learned. Further, a synthesis module is designed to generate speech based on the input representations and the derived factors. This entire paradigm of analysis-synthesis is trained purely in a self-supervised manner with an auto-encoding loss. For A2A emotion style transfer, the emotion embedding extracted from the reference speech along with the rest of the representations from the source speech are used in the synthesis module to generate the style translated speech. In our experiments, we evaluate the converted speech on content/speaker preservation (w.r.t. source) as well as on the effectiveness of the emotion style transfer (w.r.t. reference). The proposal, A2A-ZEST, is shown to improve over other prior works on these evaluations, thereby enabling style transfer without any parallel training data. We also illustrate the application of the proposed work for data augmentation in emotion recognition tasks.
#### Source Separation of Small Classical Ensembles: Challenges and Opportunities
 - **Authors:** Gerardo Roa-Dabike, Trevor J. Cox, Jon P. Barker, Michael A. Akeroyd, Scott Bannister, Bruno Fazenda, Jennifer Firth, Simone Graetzer, Alinka Greasley, Rebecca R. Vos, William M. Whitmer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.17823

 - **Pdf link:** https://arxiv.org/pdf/2505.17823

 - **Abstract**
 Musical (MSS) source separation of western popular music using non-causal deep learning can be very effective. In contrast, MSS for classical music is an unsolved problem. Classical ensembles are harder to separate than popular music because of issues such as the inherent greater variation in the music; the sparsity of recordings with ground truth for supervised training; and greater ambiguity between instruments. The Cadenza project has been exploring MSS for classical music. This is being done so music can be remixed to improve listening experiences for people with hearing loss. To enable the work, a new database of synthesized woodwind ensembles was created to overcome instrumental imbalances in the EnsembleSet. For the MSS, a set of ConvTasNet models was used with each model being trained to extract a string or woodwind instrument. ConvTasNet was chosen because it enabled both causal and non-causal approaches to be tested. Non-causal approaches have dominated MSS work and are useful for recorded music, but for live music or processing on hearing aids, causal signal processing is needed. The MSS performance was evaluated on the two small datasets (Bach10 and URMP) of real instrument recordings where the ground-truth is available. The performances of the causal and non-causal systems were similar. Comparing the average Signal-to-Distortion (SDR) of the synthesized validation set (6.2 dB causal; 6.9 non-causal), to the real recorded evaluation set (0.3 dB causal, 0.4 dB non-causal), shows that mismatch between synthesized and recorded data is a problem. Future work needs to either gather more real recordings that can be used for training, or to improve the realism and diversity of the synthesized recordings to reduce the mismatch...
#### Effects of auditory distance cues and reverberation on spatial perception and listening strategies
 - **Authors:** Fulvio Missoni, Katarina Poole, Lorenzo Picinali, Andrea Canessa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.18020

 - **Pdf link:** https://arxiv.org/pdf/2505.18020

 - **Abstract**
 Spatial hearing, the brain's ability to use auditory cues to identify the origin of sounds, is crucial for everyday listening. While simplified paradigms have advanced the understanding of spatial hearing, their lack of ecological validity limits their applicability to real-life conditions. This study aims to address this gap by investigating the effects of listener movement, reverberation, and distance on localisation accuracy in a more ecologically valid context. Participants performed active localisation tasks with no specific instructions on listening strategy, in either anechoic or reverberant conditions. The results indicate that the head movements were more frequent in reverberant environments, suggesting an adaptive strategy to mitigate uncertainty in binaural cues due to reverberation. While distance did not affect the listening strategy, it influenced the localisation performance. Our outcomes suggest that listening behaviour is adapted depending on the current acoustic conditions to support an effective perception of the space.
#### Improving endpoint detection in end-to-end streaming ASR for conversational speech
 - **Authors:** Anandh C, Karthik Pandia Durai, Jeena Prakash, Manickavela Arumugam, Kadri Hacioglu, S.Pavankumar Dubagunta, Andreas Stolcke, Shankar Venkatesan, Aravind Ganapathiraju
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17070

 - **Pdf link:** https://arxiv.org/pdf/2505.17070

 - **Abstract**
 ASR endpointing (EP) plays a major role in delivering a good user experience in products supporting human or artificial agents in human-human/machine conversations. Transducer-based ASR (T-ASR) is an end-to-end (E2E) ASR modelling technique preferred for streaming. A major limitation of T-ASR is delayed emission of ASR outputs, which could lead to errors or delays in EP. Inaccurate EP will cut the user off while speaking, returning incomplete transcript while delays in EP will increase the perceived latency, degrading the user experience. We propose methods to improve EP by addressing delayed emission along with EP mistakes. To address the delayed emission problem, we introduce an end-of-word token at the end of each word, along with a delay penalty. The EP delay is addressed by obtaining a reliable frame-level speech activity detection using an auxiliary network. We apply the proposed methods on Switchboard conversational speech corpus and evaluate it against a delay penalty method.
#### Impact of Frame Rates on Speech Tokenizer: A Case Study on Mandarin and English
 - **Authors:** Haoyang Zhang, Hexin Liu, Xiangyu Zhang, Qiquan Zhang, Yuchen Hu, Junqi Zhao, Fei Tian, Xuerui Yang, Eng Siong Chng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17076

 - **Pdf link:** https://arxiv.org/pdf/2505.17076

 - **Abstract**
 The speech tokenizer plays a crucial role in recent speech tasks, generally serving as a bridge between speech signals and language models. While low-frame-rate codecs are widely employed as speech tokenizers, the impact of frame rates on speech tokens remains underexplored. In this study, we investigate how varying frame rates affect speech tokenization by examining Mandarin and English, two typologically distinct languages. We encode speech at different frame rates and evaluate the resulting semantic tokens in the speech recognition task. Our findings reveal that frame rate variations influence speech tokenization differently for each language, highlighting the interplay between frame rates, phonetic density, and language-specific acoustic features. The results provide insights into optimizing frame rate selection for speech tokenizers, with implications for automatic speech recognition, text-to-speech, and other speech-related applications.
#### Semantic-Aware Interpretable Multimodal Music Auto-Tagging
 - **Authors:** Andreas Patakis, Vassilis Lyberatos, Spyridon Kantarelis, Edmund Dervakos, Giorgos Stamou
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17233

 - **Pdf link:** https://arxiv.org/pdf/2505.17233

 - **Abstract**
 Music auto-tagging is essential for organizing and discovering music in extensive digital libraries. While foundation models achieve exceptional performance in this domain, their outputs often lack interpretability, limiting trust and usability for researchers and end-users alike. In this work, we present an interpretable framework for music auto-tagging that leverages groups of musically meaningful multimodal features, derived from signal processing, deep learning, ontology engineering, and natural language processing. To enhance interpretability, we cluster features semantically and employ an expectation maximization algorithm, assigning distinct weights to each group based on its contribution to the tagging process. Our method achieves competitive tagging performance while offering a deeper understanding of the decision-making process, paving the way for more transparent and user-centric music tagging systems.
#### Benchmarking Expressive Japanese Character Text-to-Speech with VITS and Style-BERT-VITS2
 - **Authors:** Zackary Rackauckas, Julia Hirschberg
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17320

 - **Pdf link:** https://arxiv.org/pdf/2505.17320

 - **Abstract**
 Synthesizing expressive Japanese character speech poses unique challenges due to pitch-accent sensitivity and stylistic variability. This paper benchmarks two open-source text-to-speech models--VITS and Style-BERT-VITS2 JP Extra (SBV2JE)--on in-domain, character-driven Japanese speech. Using three character-specific datasets, we evaluate models across naturalness (mean opinion and comparative mean opinion score), intelligibility (word error rate), and speaker consistency. SBV2JE matches human ground truth in naturalness (MOS 4.37 vs. 4.38), achieves lower WER, and shows slight preference in CMOS. Enhanced by pitch-accent controls and a WavLM-based discriminator, SBV2JE proves effective for applications like language learning and character dialogue generation, despite higher computational demands.
#### VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering
 - **Authors:** Zackary Rackauckas, Julia Hirschberg
 - **Subjects:** Subjects:
Information Retrieval (cs.IR); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17326

 - **Pdf link:** https://arxiv.org/pdf/2505.17326

 - **Abstract**
 We introduce VoxRAG, a modular speech-to-speech retrieval-augmented generation system that bypasses transcription to retrieve semantically relevant audio segments directly from spoken queries. VoxRAG employs silence-aware segmentation, speaker diarization, CLAP audio embeddings, and FAISS retrieval using L2-normalized cosine similarity. We construct a 50-query test set recorded as spoken input by a native English speaker. Retrieval quality was evaluated using LLM-as-a-judge annotations. For very relevant segments, cosine similarity achieved a Recall@10 of 0.34. For somewhat relevant segments, Recall@10 rose to 0.60 and nDCG@10 to 0.27, highlighting strong topical alignment. Answer quality was judged on a 0--2 scale across relevance, accuracy, completeness, and precision, with mean scores of 0.84, 0.58, 0.56, and 0.46 respectively. While precision and retrieval quality remain key limitations, VoxRAG shows that transcription-free speech-to-speech retrieval is feasible in RAG systems.
#### LLM-based Generative Error Correction for Rare Words with Synthetic Data and Phonetic Context
 - **Authors:** Natsuo Yamashita, Masaaki Yamamoto, Hiroaki Kokubo, Yohei Kawaguchi
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17410

 - **Pdf link:** https://arxiv.org/pdf/2505.17410

 - **Abstract**
 Generative error correction (GER) with large language models (LLMs) has emerged as an effective post-processing approach to improve automatic speech recognition (ASR) performance. However, it often struggles with rare or domain-specific words due to limited training data. Furthermore, existing LLM-based GER approaches primarily rely on textual information, neglecting phonetic cues, which leads to over-correction. To address these issues, we propose a novel LLM-based GER approach that targets rare words and incorporates phonetic information. First, we generate synthetic data to contain rare words for fine-tuning the GER model. Second, we integrate ASR's N-best hypotheses along with phonetic context to mitigate over-correction. Experimental results show that our method not only improves the correction of rare words but also reduces the WER and CER across both English and Japanese datasets.
#### UniTTS: An end-to-end TTS system without decoupling of acoustic and semantic information
 - **Authors:** Rui Wang, Qianguo Sun, Tianrong Chen, Zhiyun Zeng, Junlong Wu, Jiaxing Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17426

 - **Pdf link:** https://arxiv.org/pdf/2505.17426

 - **Abstract**
 The emergence of multi-codebook neutral audio codecs such as Residual Vector Quantization (RVQ) and Group Vector Quantization (GVQ) has significantly advanced Large-Language-Model (LLM) based Text-to-Speech (TTS) systems. These codecs are crucial in separating semantic and acoustic information while efficiently harnessing semantic priors. However, since semantic and acoustic information cannot be fully aligned, a significant drawback of these methods when applied to LLM-based TTS is that large language models may have limited access to comprehensive audio information. To address this limitation, we propose DistilCodec and UniTTS, which collectively offer the following advantages: 1) This method can distill a multi-codebook audio codec into a single-codebook audio codec with 32,768 codes while achieving a near 100\% utilization. 2) As DistilCodec does not employ a semantic alignment scheme, a large amount of high-quality unlabeled audio (such as audiobooks with sound effects, songs, etc.) can be incorporated during training, further expanding data diversity and broadening its applicability. 3) Leveraging the comprehensive audio information modeling of DistilCodec, we integrated three key tasks into UniTTS's pre-training framework: audio modality autoregression, text modality autoregression, and speech-text cross-modal autoregression. This allows UniTTS to accept interleaved text and speech/audio prompts while substantially preserving LLM's text capabilities. 4) UniTTS employs a three-stage training process: Pre-Training, Supervised Fine-Tuning (SFT), and Alignment. Source code and model checkpoints are publicly available at this https URL and this https URL.
#### Exploring the Effect of Segmentation and Vocabulary Size on Speech Tokenization for Speech Language Models
 - **Authors:** Shunsuke Kando, Yusuke Miyao, Shinnosuke Takamichi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17446

 - **Pdf link:** https://arxiv.org/pdf/2505.17446

 - **Abstract**
 The purpose of speech tokenization is to transform a speech signal into a sequence of discrete representations, serving as the foundation for speech language models (SLMs). While speech tokenization has many options, their effect on the performance of SLMs remains unclear. This paper investigates two key aspects of speech tokenization: the segmentation width and the cluster size of discrete units. First, we segment speech signals into fixed/variable widths and pooled representations. We then train K-means models in multiple cluster sizes. Through the evaluation on zero-shot spoken language understanding benchmarks, we find the positive effect of moderately coarse segmentation and bigger cluster size. Notably, among the best-performing models, the most efficient one achieves a 50% reduction in training data and a 70% decrease in training runtime. Our analysis highlights the importance of combining multiple tokens to enhance fine-grained spoken language understanding.
#### Reverse-Speech-Finder: A Neural Network Backtracking Architecture for Generating Alzheimer's Disease Speech Samples and Improving Diagnosis Performance
 - **Authors:** Victor OK Li, Yang Han, Jacqueline CK Lam, Lawrence YL Cheung
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17477

 - **Pdf link:** https://arxiv.org/pdf/2505.17477

 - **Abstract**
 This study introduces Reverse-Speech-Finder (RSF), a groundbreaking neural network backtracking architecture designed to enhance Alzheimer's Disease (AD) diagnosis through speech analysis. Leveraging the power of pre-trained large language models, RSF identifies and utilizes the most probable AD-specific speech markers, addressing both the scarcity of real AD speech samples and the challenge of limited interpretability in existing models. RSF's unique approach consists of three core innovations: Firstly, it exploits the observation that speech markers most probable of predicting AD, defined as the most probable speech-markers (MPMs), must have the highest probability of activating those neurons (in the neural network) with the highest probability of predicting AD, defined as the most probable neurons (MPNs). Secondly, it utilizes a speech token representation at the input layer, allowing backtracking from MPNs to identify the most probable speech-tokens (MPTs) of AD. Lastly, it develops an innovative backtracking method to track backwards from the MPNs to the input layer, identifying the MPTs and the corresponding MPMs, and ingeniously uncovering novel speech markers for AD detection. Experimental results demonstrate RSF's superiority over traditional methods such as SHAP and Integrated Gradients, achieving a 3.5% improvement in accuracy and a 3.2% boost in F1-score. By generating speech data that encapsulates novel markers, RSF not only mitigates the limitations of real data scarcity but also significantly enhances the robustness and accuracy of AD diagnostic models. These findings underscore RSF's potential as a transformative tool in speech-based AD detection, offering new insights into AD-related linguistic deficits and paving the way for more effective non-invasive early intervention strategies.
#### Analyzing Mitigation Strategies for Catastrophic Forgetting in End-to-End Training of Spoken Language Models
 - **Authors:** Chi-Yuan Hsiao, Ke-Han Lu, Kai-Wei Chang, Chih-Kai Yang, Wei-Chih Chen, Hung-yi Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17496

 - **Pdf link:** https://arxiv.org/pdf/2505.17496

 - **Abstract**
 End-to-end training of Spoken Language Models (SLMs) commonly involves adapting pre-trained text-based Large Language Models (LLMs) to the speech modality through multi-stage training on diverse tasks such as ASR, TTS and spoken question answering (SQA). Although this multi-stage continual learning equips LLMs with both speech understanding and generation capabilities, the substantial differences in task and data distributions across stages can lead to catastrophic forgetting, where previously acquired knowledge is lost. This paper investigates catastrophic forgetting and evaluates three mitigation strategies-model merging, discounting the LoRA scaling factor, and experience replay to balance knowledge retention with new learning. Results show that experience replay is the most effective, with further gains achieved by combining it with other methods. These findings provide insights for developing more robust and efficient SLM training pipelines.
#### What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection
 - **Authors:** Binh Nguyen, Shuji Shi, Ryan Ofman, Thai Le
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17513

 - **Pdf link:** https://arxiv.org/pdf/2505.17513

 - **Abstract**
 Recent advances in text-to-speech technologies have enabled realistic voice generation, fueling audio-based deepfake attacks such as fraud and impersonation. While audio anti-spoofing systems are critical for detecting such threats, prior work has predominantly focused on acoustic-level perturbations, leaving the impact of linguistic variation largely unexplored. In this paper, we investigate the linguistic sensitivity of both open-source and commercial anti-spoofing detectors by introducing transcript-level adversarial attacks. Our extensive evaluation reveals that even minor linguistic perturbations can significantly degrade detection accuracy: attack success rates surpass 60% on several open-source detector-voice pairs, and notably one commercial detection accuracy drops from 100% on synthetic audio to just 32%. Through a comprehensive feature attribution analysis, we identify that both linguistic complexity and model-level audio embedding similarity contribute strongly to detector vulnerability. We further demonstrate the real-world risk via a case study replicating the Brad Pitt audio deepfake scam, using transcript adversarial attacks to completely bypass commercial detectors. These results highlight the need to move beyond purely acoustic defenses and account for linguistic variation in the design of robust anti-spoofing systems. All source code will be publicly available.
#### Swedish Whispers; Leveraging a Massive Speech Corpus for Swedish Speech Recognition
 - **Authors:** Leonora Vesterbacka, Faton Rekathati, Robin Kurtz, Justyna Sikora, Agnes Toftgård
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17538

 - **Pdf link:** https://arxiv.org/pdf/2505.17538

 - **Abstract**
 This work presents a suite of fine-tuned Whisper models for Swedish, trained on a dataset of unprecedented size and variability for this mid-resourced language. As languages of smaller sizes are often underrepresented in multilingual training datasets, substantial improvements in performance can be achieved by fine-tuning existing multilingual models, as shown in this work. This work reports an overall improvement across model sizes compared to OpenAI's Whisper evaluated on Swedish. Most notably, we report an average 47% reduction in WER comparing our best performing model to OpenAI's whisper-large-v3, in evaluations across FLEURS, Common Voice, and NST.
#### JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models
 - **Authors:** Zifan Peng, Yule Liu, Zhen Sun, Mingchen Li, Zeren Luo, Jingyi Zheng, Wenhan Dong, Xinlei He, Xuechao Wang, Yingjie Xue, Shengmin Xu, Xinyi Huang
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17568

 - **Pdf link:** https://arxiv.org/pdf/2505.17568

 - **Abstract**
 Audio Language Models (ALMs) have made significant progress recently. These models integrate the audio modality directly into the model, rather than converting speech into text and inputting text to Large Language Models (LLMs). While jailbreak attacks on LLMs have been extensively studied, the security of ALMs with audio modalities remains largely unexplored. Currently, there is a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare attacks and ALMs. In this paper, we present JALMBench, the \textit{first} comprehensive benchmark to assess the safety of ALMs against jailbreak attacks. JALMBench includes a dataset containing 2,200 text samples and 51,381 audio samples with over 268 hours. It supports 12 mainstream ALMs, 4 text-transferred and 4 audio-originated attack methods, and 5 defense methods. Using JALMBench, we provide an in-depth analysis of attack efficiency, topic sensitivity, voice diversity, and attack representations. Additionally, we explore mitigation strategies for the attacks at both the prompt level and the response level.
#### CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training
 - **Authors:** Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, Jieping Ye
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17589

 - **Pdf link:** https://arxiv.org/pdf/2505.17589

 - **Abstract**
 In our prior works, we introduced a scalable streaming speech synthesis model, CosyVoice 2, which integrates a large language model (LLM) and a chunk-aware flow matching (FM) model, and achieves low-latency bi-streaming speech synthesis and human-parity quality. Despite these advancements, CosyVoice 2 exhibits limitations in language coverage, domain diversity, data volume, text formats, and post-training techniques. In this paper, we present CosyVoice 3, an improved model designed for zero-shot multilingual speech synthesis in the wild, surpassing its predecessor in content consistency, speaker similarity, and prosody naturalness. Key features of CosyVoice 3 include: 1) A novel speech tokenizer to improve prosody naturalness, developed via supervised multi-task training, including automatic speech recognition, speech emotion recognition, language identification, audio event detection, and speaker analysis. 2) A new differentiable reward model for post-training applicable not only to CosyVoice 3 but also to other LLM-based speech synthesis models. 3) Dataset Size Scaling: Training data is expanded from ten thousand hours to one million hours, encompassing 9 languages and 18 Chinese dialects across various domains and text formats. 4) Model Size Scaling: Model parameters are increased from 0.5 billion to 1.5 billion, resulting in enhanced performance on our multilingual benchmark due to the larger model capacity. These advancements contribute significantly to the progress of speech synthesis in the wild. We encourage readers to listen to the demo at this https URL.
#### Toward Optimal ANC: Establishing Mutual Information Lower Bound
 - **Authors:** François Derrida, Shahar Lutati, Eliya Nachmani
 - **Subjects:** Subjects:
Information Theory (cs.IT); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.17877

 - **Pdf link:** https://arxiv.org/pdf/2505.17877

 - **Abstract**
 Active Noise Cancellation (ANC) algorithms aim to suppress unwanted acoustic disturbances by generating anti-noise signals that destructively interfere with the original noise in real time. Although recent deep learning-based ANC algorithms have set new performance benchmarks, there remains a shortage of theoretical limits to rigorously assess their improvements. To address this, we derive a unified lower bound on cancellation performance composed of two components. The first component is information-theoretic: it links residual error power to the fraction of disturbance entropy captured by the anti-noise signal, thereby quantifying limits imposed by information-processing capacity. The second component is support-based: it measures the irreducible error arising in frequency bands that the cancellation path cannot address, reflecting fundamental physical constraints. By taking the maximum of these two terms, our bound establishes a theoretical ceiling on the Normalized Mean Squared Error (NMSE) attainable by any ANC algorithm. We validate its tightness empirically on the NOISEX dataset under varying reverberation times, demonstrating robustness across diverse acoustic conditions.
#### DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations
 - **Authors:** Ziqiao Peng, Yanbo Fan, Haoyu Wu, Xuan Wang, Hongyan Liu, Jun He, Zhaoxin Fan
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.18096

 - **Pdf link:** https://arxiv.org/pdf/2505.18096

 - **Abstract**
 In face-to-face conversations, individuals need to switch between speaking and listening roles seamlessly. Existing 3D talking head generation models focus solely on speaking or listening, neglecting the natural dynamics of interactive conversation, which leads to unnatural interactions and awkward transitions. To address this issue, we propose a new task -- multi-round dual-speaker interaction for 3D talking head generation -- which requires models to handle and generate both speaking and listening behaviors in continuous conversation. To solve this task, we introduce DualTalk, a novel unified framework that integrates the dynamic behaviors of speakers and listeners to simulate realistic and coherent dialogue interactions. This framework not only synthesizes lifelike talking heads when speaking but also generates continuous and vivid non-verbal feedback when listening, effectively capturing the interplay between the roles. We also create a new dataset featuring 50 hours of multi-round conversations with over 1,000 characters, where participants continuously switch between speaking and listening roles. Extensive experiments demonstrate that our method significantly enhances the naturalness and expressiveness of 3D talking heads in dual-speaker conversations. We recommend watching the supplementary video: this https URL.


by Zyzzyva0381 (Windy). 


2025-05-27
