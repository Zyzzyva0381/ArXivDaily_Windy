# Showing new listings for Friday, 2 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 3papers 
#### Discovering phoneme-specific critical articulators through a data-driven approach
 - **Authors:** Jesuraj Bandekar, Sathvik Udupa, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.00007

 - **Pdf link:** https://arxiv.org/pdf/2505.00007

 - **Abstract**
 We propose an approach for learning critical articulators for phonemes through a machine learning approach. We formulate the learning with three models trained end to end. First, we use Acoustic to Articulatory Inversion (AAI) to predict time-varying speech articulators EMA. We also predict the phoneme-specific weights across articulators for each frame. To avoid overfitting, we also add a dropout layer before the weights prediction layer. Next, we normalize the predicted weights across articulators using min-max normalization for each frame. The normalized weights are multiplied by the ground truth $EMA$ and then we try to predict the phones at each frame. We train this whole setup end to end and use two losses. One loss is for the phone prediction which is the cross entropy loss and the other is for the AAI prediction which is the mean squared error loss. To maintain gradient flow between the phone prediction block and the $EMA$ prediction block, we use straight-through estimation. The goal here is to predict the weights of the articulator at each frame while training the model end to end.
#### Perceptual Implications of Automatic Anonymization in Pathological Speech
 - **Authors:** Soroosh Tayebi Arasteh, Saba Afza, Tri-Thien Nguyen, Lukas Buess, Maryam Parvin, Tomas Arias-Vergara, Paula Andrea Perez-Toro, Hiu Ching Hung, Mahshad Lotfinia, Thomas Gorges, Elmar Noeth, Maria Schuster, Seung Hee Yang, Andreas Maier
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2505.00409

 - **Pdf link:** https://arxiv.org/pdf/2505.00409

 - **Abstract**
 Automatic anonymization techniques are essential for ethical sharing of pathological speech data, yet their perceptual consequences remain understudied. This study presents the first comprehensive human-centered analysis of anonymized pathological speech, using a structured perceptual protocol involving ten native and non-native German listeners with diverse linguistic, clinical, and technical backgrounds. Listeners evaluated anonymized-original utterance pairs from 180 speakers spanning Cleft Lip and Palate, Dysarthria, Dysglossia, Dysphonia, and age-matched healthy controls. Speech was anonymized using state-of-the-art automatic methods (equal error rates in the range of 30-40%). Listeners completed Turing-style discrimination and quality rating tasks under zero-shot (single-exposure) and few-shot (repeated-exposure) conditions. Discrimination accuracy was high overall (91% zero-shot; 93% few-shot), but varied by disorder (repeated-measures ANOVA: p=0.007), ranging from 96% (Dysarthria) to 86% (Dysphonia). Anonymization consistently reduced perceived quality (from 83% to 59%, p<0.001), with pathology-specific degradation patterns (one-way ANOVA: p=0.005). Native listeners rated original speech slightly higher than non-native listeners (Delta=4%, p=0.199), but this difference nearly disappeared after anonymization (Delta=1%, p=0.724). No significant gender-based bias was observed. Critically, human perceptual outcomes did not correlate with automatic privacy or clinical utility metrics. These results underscore the need for listener-informed, disorder- and context-specific anonymization strategies that preserve privacy while maintaining interpretability, communicative functions, and diagnostic utility, especially for vulnerable populations such as children.
#### BERSting at the Screams: A Benchmark for Distanced, Emotional and Shouted Speech Recognition
 - **Authors:** Paige Tuttösí, Mantaj Dhillon, Luna Sang, Shane Eastwood, Poorvi Bhatia, Quang Minh Dinh, Avni Kapoor, Yewon Jin, Angelica Lim
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.00059

 - **Pdf link:** https://arxiv.org/pdf/2505.00059

 - **Abstract**
 Some speech recognition tasks, such as automatic speech recognition (ASR), are approaching or have reached human performance in many reported metrics. Yet, they continue to struggle in complex, real-world, situations, such as with distanced speech. Previous challenges have released datasets to address the issue of distanced ASR, however, the focus remains primarily on distance, specifically relying on multi-microphone array systems. Here we present the B(asic) E(motion) R(andom phrase) S(hou)t(s) (BERSt) dataset. The dataset contains almost 4 hours of English speech from 98 actors with varying regional and non-native accents. The data was collected on smartphones in the actors homes and therefore includes at least 98 different acoustic environments. The data also includes 7 different emotion prompts and both shouted and spoken utterances. The smartphones were places in 19 different positions, including obstructions and being in a different room than the actor. This data is publicly available for use and can be used to evaluate a variety of speech recognition tasks, including: ASR, shout detection, and speech emotion recognition (SER). We provide initial benchmarks for ASR and SER tasks, and find that ASR degrades both with an increase in distance and shout level and shows varied performance depending on the intended emotion. Our results show that the BERSt dataset is challenging for both ASR and SER tasks and continued work is needed to improve the robustness of such systems for more accurate real-world use.


by Zyzzyva0381 (Windy). 


2025-05-02
