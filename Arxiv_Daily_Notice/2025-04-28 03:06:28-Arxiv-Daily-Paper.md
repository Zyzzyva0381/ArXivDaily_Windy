# Showing new listings for Monday, 28 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 6papers 
#### DOSE : Drum One-Shot Extraction from Music Mixture
 - **Authors:** Suntae Hwang, Seonghyeon Kang, Kyungsu Kim, Semin Ahn, Kyogu Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2504.18157

 - **Pdf link:** https://arxiv.org/pdf/2504.18157

 - **Abstract**
 Drum one-shot samples are crucial for music production, particularly in sound design and electronic music. This paper introduces Drum One-Shot Extraction, a task in which the goal is to extract drum one-shots that are present in the music mixture. To facilitate this, we propose the Random Mixture One-shot Dataset (RMOD), comprising large-scale, randomly arranged music mixtures paired with corresponding drum one-shot samples. Our proposed model, Drum One- Shot Extractor (DOSE), leverages neural audio codec language models for end-to-end extraction, bypassing traditional source separation steps. Additionally, we introduce a novel onset loss, designed to encourage accurate prediction of the initial transient of drum one-shots, which is essential for capturing timbral characteristics. We compare this approach against a source separation-based extraction method as a baseline. The results, evaluated using Frechet Audio Distance (FAD) and Multi-Scale Spectral loss (MSS), demonstrate that DOSE, enhanced with onset loss, outperforms the baseline, providing more accurate and higher-quality drum one-shots from music mixtures. The code, model checkpoint, and audio examples are available at this https URL
#### Kimi-Audio Technical Report
 - **Authors:** KimiTeam, Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, Zhengtao Wang, Chu Wei, Yifei Xin, Xinran Xu, Jianwei Yu, Yutao Zhang, Xinyu Zhou, Y. Charles, Jun Chen, Yanru Chen, Yulun Du, Weiran He, Zhenxing Hu, Guokun Lai, Qingcheng Li, Yangyang Liu, Weidong Sun, Jianzhou Wang, Yuzhi Wang, Yuefeng Wu, Yuxin Wu, Dongchao Yang, Hao Yang, Ying Yang, Zhilin Yang, Aoxiong Yin, Ruibin Yuan, Yutong Zhang, Zaida Zhou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2504.18425

 - **Pdf link:** https://arxiv.org/pdf/2504.18425

 - **Abstract**
 We present Kimi-Audio, an open-source audio foundation model that excels in audio understanding, generation, and conversation. We detail the practices in building Kimi-Audio, including model architecture, data curation, training recipe, inference deployment, and evaluation. Specifically, we leverage a 12.5Hz audio tokenizer, design a novel LLM-based architecture with continuous features as input and discrete tokens as output, and develop a chunk-wise streaming detokenizer based on flow matching. We curate a pre-training dataset that consists of more than 13 million hours of audio data covering a wide range of modalities including speech, sound, and music, and build a pipeline to construct high-quality and diverse post-training data. Initialized from a pre-trained LLM, Kimi-Audio is continual pre-trained on both audio and text data with several carefully designed tasks, and then fine-tuned to support a diverse of audio-related tasks. Extensive evaluation shows that Kimi-Audio achieves state-of-the-art performance on a range of audio benchmarks including speech recognition, audio understanding, audio question answering, and speech conversation. We release the codes, model checkpoints, as well as the evaluation toolkits in this https URL.
#### Music Tempo Estimation on Solo Instrumental Performance
 - **Authors:** Zhanhong He, Roberto Togneri, Xiangyu Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Information Retrieval (cs.IR)
 - **Arxiv link:** https://arxiv.org/abs/2504.18502

 - **Pdf link:** https://arxiv.org/pdf/2504.18502

 - **Abstract**
 Recently, automatic music transcription has made it possible to convert musical audio into accurate MIDI. However, the resulting MIDI lacks music notations such as tempo, which hinders its conversion into sheet music. In this paper, we investigate state-of-the-art tempo estimation techniques and evaluate their performance on solo instrumental music. These include temporal convolutional network (TCN) and recurrent neural network (RNN) models that are pretrained on massive of mixed vocals and instrumental music, as well as TCN models trained specifically with solo instrumental performances. Through evaluations on drum, guitar, and classical piano datasets, our TCN models with the new training scheme achieved the best performance. Our newly trained TCN model increases the Acc1 metric by 38.6% for guitar tempo estimation, compared to the pretrained TCN model with an Acc1 of 61.1%. Although our trained TCN model is twice as accurate as the pretrained TCN model in estimating classical piano tempo, its Acc1 is only 50.9%. To improve the performance of deep learning models, we investigate their combinations with various post-processing methods. These post-processing techniques effectively enhance the performance of deep learning models when they struggle to estimate the tempo of specific instruments.
#### STNet: Prediction of Underwater Sound Speed Profiles with An Advanced Semi-Transformer Neural Network
 - **Authors:** Wei Huang, Jiajun Lu, Hao Zhang, Tianhe Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2504.17912

 - **Pdf link:** https://arxiv.org/pdf/2504.17912

 - **Abstract**
 Real time acquisition of accurate underwater sound velocity profile (SSP) is crucial for tracking the propagation trajectory of underwater acoustic signals, making it play a key role in ocean communication positioning. SSPs can be directly measured by instruments or inverted leveraging sound field data. Although measurement techniques provide a good accuracy, they are constrained by limited spatial coverage and require substantial time investment. The inversion method based on real-time measurement of acoustic field data improves operational efficiency, but loses the accuracy of SSP estimation and suffers from limited spatial applicability due to its stringent requirements for ocean observation infrastructure. To achieve accurate long-term ocean SSP estimation independent of real-time underwater data measurements, we propose a Semi-Transformer neural network (STNet) specifically designed for simulating sound velocity distribution patterns from the perspective of time series prediction. The proposed network architecture incorporates an optimized self-attention mechanism to effectively capture long-range temporal dependencies within historical sound velocity time-series data, facilitating accurate estimation of current SSPs or prediction of future SSPs. Through architectural optimization of the Transformer framework and integration of a time encoding mechanism, STNet could effectively improve computational efficiency. Comparative experimental results reveal that STNet outperforms state-of-the-art models in predictive accuracy and maintain good computational efficiency, demonstrating its potential for enabling accurate long-term full-depth ocean SSP forecasting.
#### Tracking Articulatory Dynamics in Speech with a Fixed-Weight BiLSTM-CNN Architecture
 - **Authors:** Leena G Pillai, D. Muhammad Noorul Mubarak, Elizabeth Sherly
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.18099

 - **Pdf link:** https://arxiv.org/pdf/2504.18099

 - **Abstract**
 Speech production is a complex sequential process which involve the coordination of various articulatory features. Among them tongue being a highly versatile active articulator responsible for shaping airflow to produce targeted speech sounds that are intellectual, clear, and distinct. This paper presents a novel approach for predicting tongue and lip articulatory features involved in a given speech acoustics using a stacked Bidirectional Long Short-Term Memory (BiLSTM) architecture, combined with a one-dimensional Convolutional Neural Network (CNN) for post-processing with fixed weights initialization. The proposed network is trained with two datasets consisting of simultaneously recorded speech and Electromagnetic Articulography (EMA) datasets, each introducing variations in terms of geographical origin, linguistic characteristics, phonetic diversity, and recording equipment. The performance of the model is assessed in Speaker Dependent (SD), Speaker Independent (SI), corpus dependent (CD) and cross corpus (CC) modes. Experimental results indicate that the proposed model with fixed weights approach outperformed the adaptive weights initialization with in relatively minimal number of training epochs. These findings contribute to the development of robust and efficient models for articulatory feature prediction, paving the way for advancements in speech production research and applications.
#### Seeing Soundscapes: Audio-Visual Generation and Separation from Soundscapes Using Audio-Visual Separator
 - **Authors:** Minjae Kang, Martim Brandão
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.18283

 - **Pdf link:** https://arxiv.org/pdf/2504.18283

 - **Abstract**
 Recent audio-visual generative models have made substantial progress in generating images from audio. However, existing approaches focus on generating images from single-class audio and fail to generate images from mixed audio. To address this, we propose an Audio-Visual Generation and Separation model (AV-GAS) for generating images from soundscapes (mixed audio containing multiple classes). Our contribution is threefold: First, we propose a new challenge in the audio-visual generation task, which is to generate an image given a multi-class audio input, and we propose a method that solves this task using an audio-visual separator. Second, we introduce a new audio-visual separation task, which involves generating separate images for each class present in a mixed audio input. Lastly, we propose new evaluation metrics for the audio-visual generation task: Class Representation Score (CRS) and a modified R@K. Our model is trained and evaluated on the VGGSound dataset. We show that our method outperforms the state-of-the-art, achieving 7% higher CRS and 4% higher R@2* in generating plausible images with mixed audio.


by Zyzzyva0381 (Windy). 


2025-04-28
