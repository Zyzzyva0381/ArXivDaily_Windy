# Showing new listings for Monday, 9 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 12papers 
#### Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning
 - **Authors:** Yangui Fang, Jing Peng, Xu Li, Yu Xi, Chengwei Zhang, Guohui Zhong, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2506.05671

 - **Pdf link:** https://arxiv.org/pdf/2506.05671

 - **Abstract**
 Recent advances in automatic speech recognition (ASR) have combined speech encoders with large language models (LLMs) through projection, forming Speech LLMs with strong performance. However, adapting them to new domains remains challenging, especially in low-resource settings where paired speech-text data is scarce. We propose a text-only fine-tuning strategy for Speech LLMs using unpaired target-domain text without requiring additional audio. To preserve speech-text alignment, we introduce a real-time evaluation mechanism during fine-tuning. This enables effective domain adaptation while maintaining source-domain performance. Experiments on LibriSpeech, SlideSpeech, and Medical datasets show that our method achieves competitive recognition performance, with minimal degradation compared to full audio-text fine-tuning. It also improves generalization to new domains without catastrophic forgetting, highlighting the potential of text-only fine-tuning for low-resource domain adaptation of ASR.
#### Bridging the Modality Gap: Softly Discretizing Audio Representation for LLM-based Automatic Speech Recognition
 - **Authors:** Mu Yang, Szu-Jui Chen, Jiamin Xie, John Hansen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.05706

 - **Pdf link:** https://arxiv.org/pdf/2506.05706

 - **Abstract**
 One challenge of integrating speech input with large language models (LLMs) stems from the discrepancy between the continuous nature of audio data and the discrete token-based paradigm of LLMs. To mitigate this gap, we propose a method for integrating vector quantization (VQ) into LLM-based automatic speech recognition (ASR). Using the LLM embedding table as the VQ codebook, the VQ module aligns the continuous representations from the audio encoder with the discrete LLM inputs, enabling the LLM to operate on a discretized audio representation that better reflects the linguistic structure. We further create a soft "discretization" of the audio representation by updating the codebook and performing a weighted sum over the codebook embeddings. Empirical results demonstrate that our proposed method significantly improves upon the LLM-based ASR baseline, particularly in out-of-domain conditions. This work highlights the potential of soft discretization as a modality bridge in LLM-based ASR.
#### Diarization-Aware Multi-Speaker Automatic Speech Recognition via Large Language Models
 - **Authors:** Yuke Lin, Ming Cheng, Ze Li, Beilong Tang, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.05796

 - **Pdf link:** https://arxiv.org/pdf/2506.05796

 - **Abstract**
 Multi-speaker automatic speech recognition (MS-ASR) faces significant challenges in transcribing overlapped speech, a task critical for applications like meeting transcription and conversational analysis. While serialized output training (SOT)-style methods serve as common solutions, they often discard absolute timing information, limiting their utility in time-sensitive scenarios. Leveraging recent advances in large language models (LLMs) for conversational audio processing, we propose a novel diarization-aware multi-speaker ASR system that integrates speaker diarization with LLM-based transcription. Our framework processes structured diarization inputs alongside frame-level speaker and semantic embeddings, enabling the LLM to generate segment-level transcriptions. Experiments demonstrate that the system achieves robust performance in multilingual dyadic conversations and excels in complex, high-overlap multi-speaker meeting scenarios. This work highlights the potential of LLMs as unified back-ends for joint speaker-aware segmentation and transcription.
#### Audio-Aware Large Language Models as Judges for Speaking Styles
 - **Authors:** Cheng-Han Chiang, Xiaofei Wang, Chung-Ching Lin, Kevin Lin, Linjie Li, Radu Kopetz, Yao Qian, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2506.05984

 - **Pdf link:** https://arxiv.org/pdf/2506.05984

 - **Abstract**
 Audio-aware large language models (ALLMs) can understand the textual and non-textual information in the audio input. In this paper, we explore using ALLMs as an automatic judge to assess the speaking styles of speeches. We use ALLM judges to evaluate the speeches generated by SLMs on two tasks: voice style instruction following and role-playing. The speaking style we consider includes emotion, volume, speaking pace, word emphasis, pitch control, and non-verbal elements. We use four spoken language models (SLMs) to complete the two tasks and use humans and ALLMs to judge the SLMs' responses. We compare two ALLM judges, GPT-4o-audio and Gemini-2.5-pro, with human evaluation results and show that the agreement between Gemini and human judges is comparable to the agreement between human evaluators. These promising results show that ALLMs can be used as a judge to evaluate SLMs. Our results also reveal that current SLMs, even GPT-4o-audio, still have room for improvement in controlling the speaking style and generating natural dialogues.
#### CO-VADA: A Confidence-Oriented Voice Augmentation Debiasing Approach for Fair Speech Emotion Recognition
 - **Authors:** Yun-Shao Tsai, Yi-Cheng Lin, Huang-Cheng Chou, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2506.06071

 - **Pdf link:** https://arxiv.org/pdf/2506.06071

 - **Abstract**
 Bias in speech emotion recognition (SER) systems often stems from spurious correlations between speaker characteristics and emotional labels, leading to unfair predictions across demographic groups. Many existing debiasing methods require model-specific changes or demographic annotations, limiting their practical use. We present CO-VADA, a Confidence-Oriented Voice Augmentation Debiasing Approach that mitigates bias without modifying model architecture or relying on demographic information. CO-VADA identifies training samples that reflect bias patterns present in the training data and then applies voice conversion to alter irrelevant attributes and generate samples. These augmented samples introduce speaker variations that differ from dominant patterns in the data, guiding the model to focus more on emotion-relevant features. Our framework is compatible with various SER models and voice conversion tools, making it a scalable and practical solution for improving fairness in SER systems.
#### Lightweight Prompt Biasing for Contextualized End-to-End ASR Systems
 - **Authors:** Bo Ren, Yu Shi, Jinyu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06252

 - **Pdf link:** https://arxiv.org/pdf/2506.06252

 - **Abstract**
 End-to-End Automatic Speech Recognition (ASR) has advanced significantly yet still struggles with rare and domain-specific entities. This paper introduces a simple yet efficient prompt-based biasing technique for contextualized ASR, enhancing recognition accuracy by leverage a unified multitask learning framework. The approach comprises two key components: a prompt biasing model which is trained to determine when to focus on entities in prompt, and a entity filtering mechanism which efficiently filters out irrelevant entities. Our method significantly enhances ASR accuracy on entities, achieving a relative 30.7% and 18.0% reduction in Entity Word Error Rate compared to the baseline model with shallow fusion on in-house domain dataset with small and large entity lists, respectively. The primary advantage of this method lies in its efficiency and simplicity without any structure change, making it lightweight and highly efficient.
#### SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing
 - **Authors:** Mingfei Chen, Zijun Cui, Xiulong Liu, Jinlin Xiang, Caleb Zheng, Jingyuan Li, Eli Shlizerman
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.05414

 - **Pdf link:** https://arxiv.org/pdf/2506.05414

 - **Abstract**
 3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs.
#### Improving Neural Diarization through Speaker Attribute Attractors and Local Dependency Modeling
 - **Authors:** David Palzer, Matthew Maciejewski, Eric Fosler-Lussier
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.05593

 - **Pdf link:** https://arxiv.org/pdf/2506.05593

 - **Abstract**
 In recent years, end-to-end approaches have made notable progress in addressing the challenge of speaker diarization, which involves segmenting and identifying speakers in multi-talker recordings. One such approach, Encoder-Decoder Attractors (EDA), has been proposed to handle variable speaker counts as well as better guide the network during training. In this study, we extend the attractor paradigm by moving beyond direct speaker modeling and instead focus on representing more detailed `speaker attributes' through a multi-stage process of intermediate representations. Additionally, we enhance the architecture by replacing transformers with conformers, a convolution-augmented transformer, to model local dependencies. Experiments demonstrate improved diarization performance on the CALLHOME dataset.
#### Voice Impression Control in Zero-Shot TTS
 - **Authors:** Keinichi Fujita, Shota Horiguchi, Yusuke Ijima
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.05688

 - **Pdf link:** https://arxiv.org/pdf/2506.05688

 - **Abstract**
 Para-/non-linguistic information in speech is pivotal in shaping the listeners' impression. Although zero-shot text-to-speech (TTS) has achieved high speaker fidelity, modulating subtle para-/non-linguistic information to control perceived voice characteristics, i.e., impressions, remains challenging. We have therefore developed a voice impression control method in zero-shot TTS that utilizes a low-dimensional vector to represent the intensities of various voice impression pairs (e.g., dark-bright). The results of both objective and subjective evaluations have demonstrated our method's effectiveness in impression control. Furthermore, generating this vector via a large language model enables target-impression generation from a natural language description of the desired impression, thus eliminating the need for manual optimization.
#### DeepFake Doctor: Diagnosing and Treating Audio-Video Fake Detection
 - **Authors:** Marcel Klemt, Carlotta Segna, Anna Rohrbach
 - **Subjects:** Subjects:
Multimedia (cs.MM); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.05851

 - **Pdf link:** https://arxiv.org/pdf/2506.05851

 - **Abstract**
 Generative AI advances rapidly, allowing the creation of very realistic manipulated video and audio. This progress presents a significant security and ethical threat, as malicious users can exploit DeepFake techniques to spread misinformation. Recent DeepFake detection approaches explore the multimodal (audio-video) threat scenario. In particular, there is a lack of reproducibility and critical issues with existing datasets - such as the recently uncovered silence shortcut in the widely used FakeAVCeleb dataset. Considering the importance of this topic, we aim to gain a deeper understanding of the key issues affecting benchmarking in audio-video DeepFake detection. We examine these challenges through the lens of the three core benchmarking pillars: datasets, detection methods, and evaluation protocols. To address these issues, we spotlight the recent DeepSpeak v1 dataset and are the first to propose an evaluation protocol and benchmark it using SOTA models. We introduce SImple Multimodal BAseline (SIMBA), a competitive yet minimalistic approach that enables the exploration of diverse design choices. We also deepen insights into the issue of audio shortcuts and present a promising mitigation strategy. Finally, we analyze and enhance the evaluation scheme on the widely used FakeAVCeleb dataset. Our findings offer a way forward in the complex area of audio-video DeepFake detection.
#### Label-Context-Dependent Internal Language Model Estimation for CTC
 - **Authors:** Zijian Yang, Minh-Nghia Phan, Ralf Schlüter, Hermann Ney
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06096

 - **Pdf link:** https://arxiv.org/pdf/2506.06096

 - **Abstract**
 Although connectionist temporal classification (CTC) has the label context independence assumption, it can still implicitly learn a context-dependent internal language model (ILM) due to modern powerful encoders. In this work, we investigate the implicit context dependency modeled in the ILM of CTC. To this end, we propose novel context-dependent ILM estimation methods for CTC based on knowledge distillation (KD) with theoretical justifications. Furthermore, we introduce two regularization methods for KD. We conduct experiments on Librispeech and TED-LIUM Release 2 datasets for in-domain and cross-domain evaluation, respectively. Experimental results show that context-dependent ILMs outperform the context-independent priors in cross-domain evaluation, indicating that CTC learns a context-dependent ILM. The proposed label-level KD with smoothing method surpasses other ILM estimation approaches, with more than 13% relative improvement in word error rate compared to shallow fusion.
#### NAT: Neural Acoustic Transfer for Interactive Scenes in Real Time
 - **Authors:** Xutong Jin, Bo Pang, Chenxi Xu, Xinyun Hou, Guoping Wang, Sheng Li
 - **Subjects:** Subjects:
Sound (cs.SD); Graphics (cs.GR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06190

 - **Pdf link:** https://arxiv.org/pdf/2506.06190

 - **Abstract**
 Previous acoustic transfer methods rely on extensive precomputation and storage of data to enable real-time interaction and auditory feedback. However, these methods struggle with complex scenes, especially when dynamic changes in object position, material, and size significantly alter sound effects. These continuous variations lead to fluctuating acoustic transfer distributions, making it challenging to represent with basic data structures and render efficiently in real time. To address this challenge, we present Neural Acoustic Transfer, a novel approach that utilizes an implicit neural representation to encode precomputed acoustic transfer and its variations, allowing for real-time prediction of sound fields under varying conditions. To efficiently generate the training data required for the neural acoustic field, we developed a fast Monte-Carlo-based boundary element method (BEM) approximation for general scenarios with smooth Neumann conditions. Additionally, we implemented a GPU-accelerated version of standard BEM for scenarios requiring higher precision. These methods provide the necessary training data, enabling our neural network to accurately model the sound radiation space. We demonstrate our method's numerical accuracy and runtime efficiency (within several milliseconds for 30s audio) through comprehensive validation and comparisons in diverse acoustic transfer scenarios. Our approach allows for efficient and accurate modeling of sound behavior in dynamically changing environments, which can benefit a wide range of interactive applications such as virtual reality, augmented reality, and advanced audio production.


by Zyzzyva0381 (Windy). 


2025-06-09
