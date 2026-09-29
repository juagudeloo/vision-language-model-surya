# Research Roadmap for Solar Physics Vision-Language Models: Integrating Surya and Earth Observation Advances

## Executive Summary

The emergence of multimodal foundation models has initiated a paradigm shift in heliophysics and Earth observation (EO). This report outlines a research roadmap for developing a Vision-Language Model (VLM) for solar physics, specifically focusing on the analysis of Solar Dynamics Observatory (SDO) and Solar Ultraviolet Imager (SUVI) data. Central to this effort is the **Surya** foundation model [26], which employs a specialized spectroformer architecture to process solar imagery and spectroscopy. While solar physics VLM research is in its nascent stages—with initial work focusing on image captioning for solar events [21]—the field of Earth remote sensing provides a mature blueprint for advancement.

Remote sensing (RS) has successfully evolved from basic image captioning to complex Visual Question Answering (VQA) and scientific reasoning [1], [4]. Advances in RS foundation models (e.g., GeoChat, RS-LLaVA, and Spectral-LLaVA) demonstrate the efficacy of instruction tuning and multispectral alignment in interpreting high-resolution, multi-instrument data [3], [9], [20]. By leveraging these RS methodologies—such as hierarchical spatial feature integration [7] and retrieval-augmented generation (RAG) [10]—solar physics can bridge the gap from descriptive analysis to predictive space weather forecasting and automated event detection across SDO/AIA, SDO/HMI, and SUVI imagery.

## Table of Contents
1. [Introduction](#introduction)
2. [The Surya Foundation Model: A Heliophysics Core](#the-surya-foundation-model)
3. [Learning from Earth Observation: Transferable Advances](#learning-from-earth-observation)
4. [Methodologies for Solar Physics VLMs](#methodologies)
5. [Task Evolution: From Captioning to VQA and Reasoning](#task-evolution)
6. [Research Gaps and Challenges](#research-gaps)
7. [Research Roadmap and Recommendations](#research-roadmap)
8. [Conclusion](#conclusion)

<a name="introduction"></a>
## 1. Introduction

Solar physics relies on high-cadence, multi-wavelength observations from instruments like the Atmospheric Imaging Assembly (AIA) and the Helioseismic and Magnetic Imager (HMI) on the Solar Dynamics Observatory (SDO), as well as the Solar Ultraviolet Imager (SUVI). These instruments generate massive datasets comprising extreme ultraviolet (EUV) imagery, magnetograms, and spectroscopic data, which are critical for monitoring solar activity such as flares, coronal mass ejections (CMEs), and sunspot evolution [21], [27], [30].

Despite the volume of data, the interpretation of solar features has traditionally required intensive human effort or specialized, single-task deep learning models [31]. The development of Vision-Language Models (VLMs) offers a transformative approach to unify these tasks through a natural language interface, enabling automated event description, interactive querying, and scientific reasoning. While initial solar VLM efforts have achieved success in image captioning [21], the field currently lacks the sophisticated VQA and reasoning capabilities seen in Earth remote sensing [1], [3].

<a name="the-surya-foundation-model"></a>
## 2. The Surya Foundation Model: A Heliophysics Core

The **Surya** model represents the first dedicated foundation model for heliophysics [26]. It is designed to overcome the limitations of general-domain vision models when applied to specialized solar data.

### 2.1 Spectroformer Architecture
Surya utilizes a **spectroformer** architecture, which is tailored to handle the unique spectral characteristics of solar imagery and spectroscopy [26]. This architecture allows the model to learn representations across multiple wavelengths simultaneously, capturing the physical relationships between different layers of the solar atmosphere (from the photosphere to the corona).

### 2.2 SuryaBench and Model Training
The training of Surya is supported by **SuryaBench**, a comprehensive benchmark dataset designed to advance machine learning in heliophysics and space weather prediction [25], [32]. SuryaBench integrates data from multiple solar missions, providing a standardized platform for evaluating model performance on tasks such as flare forecasting and solar feature identification. The development of Surya and its benchmark emphasizes the need for domain-specific foundation models that can represent the complex physics of the Sun more accurately than general-purpose vision encoders [26].

<a name="learning-from-earth-observation"></a>
## 3. Learning from Earth Observation: Transferable Advances

Earth remote sensing (RS) serves as a critical transfer field for solar physics VLMs due to shared challenges in high-resolution imagery, multispectral data, and temporal analysis.

### 3.1 Multispectral Representation
Similar to SDO/AIA's multiple EUV channels, RS utilizes multispectral data from satellites like Sentinel-2 and Landsat 8/9 [19]. Models such as **Spectral-LLaVA** integrate specialized multispectral vision encoders (e.g., SpectralGPT) with Large Language Models (LLMs) like LLaMA-3 to bridge vision and text [20]. This alignment allows models to perform semantic reasoning on spectral features that are not visible in standard RGB images—a direct parallel to analyzing solar magnetograms (HMI) and EUV imagery (AIA/SUVI).

### 3.2 Foundation Models and Instruction Tuning
The RS community has moved from domain-specific models to Large Vision-Language Models (LVLMs) such as **GeoChat** and **RS-LLaVA** [3], [9]. These models employ instruction tuning on large-scale datasets (e.g., RS-GPT4V, ChatEarthNet) to follow complex user queries [2], [19]. For solar physics, this suggests that the next step beyond the Surya foundation model should be the creation of a solar-specific instruction-following dataset that pairs SDO/SUVI imagery with scientific questions and expert descriptions.

### 3.3 Temporal Modeling and Change Detection
Solar activity is inherently dynamic, requiring temporal reasoning for flare forecasting and CME tracking. In RS, **Temporal Vision-Language Models** have evolved to handle bi-temporal and multi-temporal sequences for change detection and captioning [18]. Techniques such as bi-temporal Siamese encoders and temporal fusion modules [18] can be adapted to solar physics to analyze the evolution of active regions and sunspot numbers over time [30].

<a name="methodologies"></a>
## 4. Methodologies for Solar Physics VLMs

Building a robust VLM for solar physics requires integrating state-of-the-art architectures with solar-specific data modalities.

### 4.1 Architecture and Fusion Strategies
*   **Encoders:** Leveraging a pretrained spectroformer (Surya) or specialized RS encoders like SigLIP and RemoteCLIP can provide a strong visual backbone [6], [23], [26].
*   **Connectors:** Multi-layer Perceptrons (MLP) or Q-Formers are commonly used to project visual features into the language space [5], [16].
*   **Fusion:** Hierarchical Spatial Feature Integration (SFI) and Cross-Attention mechanisms are recommended to fuse multi-scale features from different SDO/AIA channels without losing fine-grained details [7], [13].
*   **Mixture of Experts (MoE):** Using an MoE architecture (like RS-MoE) could allow the VLM to route different solar tasks—such as flare forecasting vs. magnetogram analysis—to specialized expert LLMs [8], [11].

### 4.2 Data Modalities and Knowledge Integration
The VLM should ingest:
*   **SDO/AIA Imagery:** Multi-wavelength EUV images for coronal activity.
*   **SDO/HMI Magnetograms:** For magnetic field analysis and sunspot counting [30].
*   **SUVI Imagery:** For continuous monitoring of the solar disk.
*   **Spectroscopy:** Integrated via the spectroformer architecture [26].
*   **External Knowledge:** Implementing **Retrieval-Augmented Generation (RAG)**, as seen in RS-RAG, can ground model responses in solar physics textbooks and mission documentation, reducing hallucinations [10], [15].

<a name="task-evolution"></a>
## 5. Task Evolution: From Captioning to VQA and Reasoning

The development of solar VLMs follows a distinct evolutionary path, as mirrored in the RS literature [1], [4].

### 5.1 Image Captioning
Initial efforts, such as the transformer-based solar image captioning model, focus on generating descriptions for sunspots, flares, and prominences using the DeepSDO dataset [21]. While effective, these models are typically limited to descriptive tasks.

### 5.2 Visual Question Answering (VQA)
The RS field has successfully transitioned to VQA, where models can answer complex questions about spatial relationships and object attributes [5], [17]. Implementing VQA for SDO/SUVI data would enable researchers to ask questions like "Is there a significant flare risk in active region 13088?" or "How has the magnetic flux changed in the last 6 hours?"

### 5.3 Scientific Reasoning and Intelligent Agents
The final stage of evolution involves scientific reasoning and the creation of **intelligent agents** [22], [34]. Models like **Earth-Agent** demonstrate the ability to automate complex EO workflows. A "Solar-Agent" could autonomously monitor SDO feeds, detect anomalies, retrieve historical data for comparison, and issue space weather alerts based on its reasoning [29].

<a name="research-gaps"></a>
## 6. Research Gaps and Challenges

Despite the potential, several gaps remain in the development of solar VLMs:
*   **Lack of Domain-Specific Instruction Data:** Unlike RS, which has datasets like RS-GPT4V, solar physics lacks large-scale, high-quality image-text instruction sets [2], [12].
*   **Temporal Reasoning Limits:** Most current VLMs focus on single-image understanding and struggle with long-term temporal dependencies required for flare forecasting [18], [29].
*   **Hallucinations in Scientific Contexts:** General VLMs may generate physically incorrect descriptions of solar phenomena, necessitating strict grounding and RAG [10], [33].
*   **Multispectral Fusion Gaps:** While Surya addresses spectral representation, the seamless fusion of magnetograms and EUV imagery for joint reasoning is still under-explored compared to Earth multispectral/SAR fusion [14], [20].

<a name="research-roadmap"></a>
## 7. Research Roadmap and Recommendations

To build a high-performing solar VLM using Surya and SDO/SUVI data, the following roadmap is proposed:

### Phase 1: Foundation and Data Enrichment (Short-term)
1.  **Enhance Surya with Multispectral Pretraining:** Use contrastive learning (similar to CLIP/RemoteCLIP) on SDO/AIA and HMI pairs to align visual features across all 10 AIA channels [23], [27].
2.  **Dataset Construction:** Develop a solar version of ChatEarthNet, using SDO metadata and event logs to generate thousands of image-text pairs for captioning and basic VQA [19].

### Phase 2: Instruction Tuning and Fusion (Medium-term)
1.  **Instruction Tuning:** Apply Low-Rank Adaptation (LoRA) to a large LLM (e.g., LLaMA-3) using the new solar instruction dataset to enable complex VQA [9].
2.  **Hierarchical Fusion:** Implement a Hierarchical Spatial Feature Integration module to combine global solar disk context from SUVI with local active region details from AIA [7].

### Phase 3: Temporal Reasoning and RAG (Long-term)
1.  **Temporal VLM Development:** Incorporate temporal fusion modules and single-pixel time-series analysis to model the evolution of solar features for predictive tasks [24], [28].
2.  **Solar-RAG Integration:** Connect the VLM to a knowledge base of solar physics literature (e.g., NASA ADS) to provide expert-level scientific reasoning [10].

<a name="conclusion"></a>
## 8. Conclusion

Developing a Vision-Language Model for solar physics is a high-potential endeavor that can significantly accelerate the analysis of big data from SDO and SUVI. By using the **Surya spectroformer** as a visual foundation and adopting the **instruction tuning, multispectral fusion, and temporal modeling** techniques pioneered in Earth remote sensing, the heliophysics community can move from simple image captioning toward automated scientific reasoning and space weather forecasting. The convergence of these two fields offers a clear path toward building intelligent agents capable of interpreting the complex and dynamic nature of the Sun.

## References

[1]L. Bashmal, Y. Bazi, F. Melgani, M. M. A. Rahhal, and M. Zuair, “Language Integration in Remote Sensing: Tasks, datasets, and future directions,” IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/mgrs.2023.3316438.

[2]L. Xu et al., “RS-GPT4V: A Unified Multimodal Instruction-Following Dataset for Remote   Sensing Image Understanding,” June 2024, doi: 10.48550/arxiv.2406.12479.

[3]K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. Khan, “GeoChat: Grounded Large Vision-Language Model for Remote Sensing,” arXiv.org, vol. abs/2311.15826, Nov. 2023, doi: 10.48550/arxiv.2311.15826.

[4]J. A. Eyinade and A. J. Ademusire, “GeoLLMs in action: A systematic review of multimodal models for satellite image captioning and geospatial understanding,” Open access research journal of science and technology, vol. 14, no. 2, pp. 049–064, July 2025, doi: 10.53022/oarjst.2025.14.2.0093.

[5]X. Li, J. Ding, and M. Elhoseiny, “VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote   Sensing Image Understanding,” June 2024, doi: 10.48550/arxiv.2406.12384.

[6]Z. Li et al., “LHRS-Bot-Nova: Improved Multimodal Large Language Model for Remote   Sensing Vision-Language Interpretation,” Nov. 2024, doi: 10.48550/arxiv.2411.09301.

[7]K. Lu, R. Zhang, X. Huang, and Y. Xie, “Aquila: A Hierarchically Aligned Visual-Language Model for Enhanced   Remote Sensing Image Comprehension,” Nov. 2024, doi: 10.48550/arxiv.2411.06074.

[8]L. Hui et al., “RS-MoE: Mixture of Experts for Remote Sensing Image Captioning and   Visual Question Answering,” Nov. 2024, doi: 10.48550/arxiv.2411.01595.

[9]Y. Bazi, L. Bashmal, M. M. A. Rahhal, R. Ricci, and F. Melgani, “RS-LLaVA: A Large Vision-Language Model for Joint Captioning and Question Answering in Remote Sensing Imagery,” Remote sensing, Apr. 2024, doi: 10.3390/rs16091477.

[10]C. Wen et al., “RS-RAG: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model,” arXiv.org, vol. abs/2504.04988, Apr. 2025, doi: 10.48550/arxiv.2504.04988.

[11]H. Lin et al., “RS-MoE: A Vision-Language Model with Mixture of Experts for Remote Sensing Image Captioning and Visual Question Answering,” Nov. 03, 2024. [Online]. Available: https://arxiv.org/abs/2411.01595v2

[12]A. Xiao, J. Sun, Z. Gui, and W. He, “COREval: A Comprehensive and Objective Benchmark for Evaluating the   Remote Sensing Capabilities of Large Vision-Language Models,” Nov. 2024, doi: 10.48550/arxiv.2411.18145.

[13]Y. Dang et al., “FUSE-RSVLM: Feature Fusion Vision-Language Model for Remote Sensing,” Dec. 2025, doi: 10.48550/arxiv.2512.24022.

[14]X. Sun et al., “From Pixels to Prose: Advancing Multi-Modal Language Models for Remote   Sensing,” Nov. 2024, doi: 10.48550/arxiv.2411.05826.

[15]C. Wen et al., “Remote Sensing Retrieval-Augmented Generation: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model,” Apr. 07, 2025. doi: 10.1109/MGRS.2025.3645852.

[16]L. Kaiyu et al., “DescribeEarth: Describe Anything for Remote Sensing Images,” Oct. 2025, doi: 10.48550/arxiv.2509.25654.

[17]X. Zi et al., “RSVLM-QA: A Benchmark Dataset for Remote Sensing Vision Language Model-based Question Answering,” arXiv.org, vol. abs/2508.07918, Aug. 2025, doi: 10.48550/arxiv.2508.07918.

[18]C. Liu, J. Zhang, K. Chen, M. Wang, Z. Zou, and Z. Shi, “Remote Sensing Temporal Vision-Language Models: A Comprehensive Survey,” Dec. 2024, doi: 10.48550/arxiv.2412.02573.

[19]Z. Yuan, Z. Xiong, L. Mou, and X. X. Zhu, “ChatEarthNet: a global-scale image–text dataset empowering vision–language geo-foundation models,” Earth System Science Data, vol. 17, no. 3, pp. 1245–1263, Mar. 2025, doi: 10.5194/essd-17-1245-2025.

[20]E. Karanfil, N. İmamoğlu, E. Erdem, and A. Erdem, “A Vision-Language Framework for Multispectral Scene Representation Using   Language-Grounded Features,” Jan. 2025, doi: 10.48550/arxiv.2501.10144.

[21]J.-H. Baek, S. Kim, S. Choi, J. Park, and D. Kim, “Deep Learning-based Solar Image Captioning,” Advances in Space Research, Jan. 2024, doi: 10.1016/j.asr.2023.12.066.

[22]F. Peilin et al., “Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents,” Sept. 2025, doi: 10.48550/arxiv.2509.23141.

[23]“RemoteCLIP: A Vision Language Foundation Model for Remote Sensing,” June 2023, doi: 10.48550/arxiv.2306.11029.

[24]P. Jain, D. Marcos, D. Ienco, R. Interdonato, and T. Berchoux, “TimeSenCLIP: A Vision-Language Model for Remote Sensing Using Single-Pixel Time Series,” Aug. 2025, doi: 10.48550/arxiv.2508.11919.

[25]R. Sujit et al., “SuryaBench: Benchmark Dataset for Advancing Machine Learning in Heliophysics and Space Weather Prediction,” Oct. 2025, doi: 10.13016/m2nilh-d0bx.

[26]R. Sujit et al., “Surya: Foundation Model for Heliophysics,” Oct. 2025, doi: 10.13016/m2esx3-oqfx.

[27]S. Shen, Z. Gao, T. Chai, Y. Huang, and B. Pan, “Contrastive Heliophysical Image Pretraining for Solar Dynamics Observatory Records,” arXiv.org, vol. abs/2511.22958, Nov. 2025, doi: 10.48550/arxiv.2511.22958.

[28]P. Jain, D. Marcos, D. Ienco, R. Interdonato, and T. Berchoux, “TimeSenCLIP: A Time Series Vision-Language Model for Remote Sensing,” Aug. 16, 2025. doi: 10.1016/j.isprsjprs.2026.03.043.

[29]R. S., R. P., and B. U., “Solar flare forecasting with foundational transformer models across image, video, and time-series modalities,” Oct. 2025, doi: 10.48550/arxiv.2510.23400.

[30]S.-P. D., “Vision-Based CNN Prediction of Sunspot Numbers from SDO/HMI Images,” Oct. 2025, doi: 10.48550/arxiv.2510.03473.

[31]B. Shen et al., “Deep Computer Vision for Solar Physics Big Data: Opportunities and   Challenges,” Sept. 2024, doi: 10.48550/arxiv.2409.04850.

[32]“SuryaBench: Benchmark Dataset for Advancing Machine Learning Applications in Heliophysics and Space Weather,” Oct. 2025, doi: 10.5281/zenodo.17407005.

[33]M. B. Anderson, M. Cha, W. T. Freeman, J. T. Perron, N. Maidel, and K. Cahoy, “Measuring and Mitigating Hallucinations in Vision-Language Dataset   Generation for Remote Sensing,” Jan. 2025, doi: 10.48550/arxiv.2501.14905.

[34]W. Xu, Z. Yu, Y. Wang, J. Wang, and M. Peng, “RS-Agent: Automating Remote Sensing Tasks through Intelligent Agents,” June 2024, doi: 10.48550/arxiv.2406.07089.