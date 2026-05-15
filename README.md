# ThermoGen: AI-Driven Design of Hyperthermostable Enzymes
ThermoGen is an integrated AI-driven design framework for the development of hyperthermostable enzymes. By combining generative sequence modeling with Protein Language Model (PLM)-based thermostability prediction, ThermoGen enables the exploration of the protein sequence landscape to overcome thermodynamic constraints in biocatalysis.

As demonstrated in our recent work, ThermoGen was used to engineer NAD⁺-dependent methanol dehydrogenases (Mdhs) that retain activity at extreme temperatures (up to 105 °C), enabling efficient CO₂-to-methanol valorization and high-temperature enzymatic cascades.

## Overview
* **LLM-Assisted Extraction**:
Traditional enzymatic methanol oxidation is often limited by unfavorable thermodynamics. While elevating temperature can shift the reaction equilibrium toward product formation, it requires enzymes with extreme stability. ThermoGen addresses this "stability-activity" trade-off through a three-pillar computational pipeline:
* **Generative Modeling**: Diversifying the Mdh sequence landscape using deep generative architectures.
* **Thermostability Prediction**: Leveraging PLMs (e.g., ESM-based architectures) to perform zero-shot or supervised screening for melting temperature ($T_m$) and structural rigidity.
* **Structure-Guided Refinement**: Final optimization of candidates to reinforce structural integrity for operation in high-temperature regimes.

## 📌 Key Features
* **Thermodynamic Lever Integration**: Designed specifically to optimize enzymes for high-temperature biocatalysis.
* **PLM-Powered Screening**: High-throughput prediction of enzyme stability without the need for exhaustive MD simulations.
* **Closed-loop Validation**: Compatible with automated enzymatic DNA synthesis platforms for rapid "Design-Build-Test" cycles.
* **Proven Results**: Successfully generated Mdh variants with a 200-fold increase in productivity and $T_m > 80^\circ C$.
