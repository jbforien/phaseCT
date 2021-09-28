# PhaseCT
PhaseCT provides linear approximated phase-retrieval algorithms for X-ray imaging. Several algorithms presented in scientific literature are implemented such as contrast transfer function (CTF), transport of intensity equation (TIE), and Paganin. Pre-processing tools to perform image resizing and registration are also available.

## Installation
Download a_resizeMultiProj, b_alignMultiProj, c_phaseRetrieval, and findAlphaFit files in a directory.

## Usage
Download and adapt resizeAlignDemo and phaseDemo scripts.

Demo scripts will run image registration and phase-retrieval algorithms. Parameters selection and path definition need to be done at the beginning of the script, after libraries importation. No changes should be done after these lines. 

### Image registration algorithms:
- **crossCorr_imreg_dft** -  cross-correlation image registration in Fourier space using [imreg_dft](https://pythonhosted.org/imreg_dft/) library 
- **crossCorr_skimage_fourier** -  cross-correlation image registration in Fourier space using [scikit-image](https://scikit-image.org/) library 
- **crossCorr_skimage_real** -  cross-correlation image registration in real space using [scikit-image](https://scikit-image.org/) library 
- **mutualInfo_dipy** -  mutual information image registration using [dipy](https://dipy.org/) library 

### Phase-retrieval algorithms:
- **Paganin** - single-distance algorithm using known alpha-beta refractive indexes. [Paganin et al., 2002](https://doi.org/10.1046/j.1365-2818.2002.01010.x)
- **multiPaganin** - multi-distance algorithm using known alpha-beta refractive indexes. [Yu et al., 2018](https://doi.org/10.1364/OE.26.011110)
- **sglDstCTF** - single-distance algorithm using known alpha-beta refractive indexes. [Yu et al., 2018](https://doi.org/10.1364/OE.26.011110)
- **homoCTF** - multi-distance algorithm using known alpha-beta refractive indexes. [Yu et al., 2018](https://doi.org/10.1364/OE.26.011110)
- **CTF** - multi-distance algorithm. No assumptions on alpha-beta required. [Langer et al., 2008](https://doi.org/10.1118/1.2975224)
- **CTFPurePhase** - weak phase approximation, assumes know delta/beta. [Cloetens et al., 2002](https://doi.org/10.1117/12.452867)
- **CTFPurePhaseWithAbs** - weak phase approximation with absorption, assumes know delta/beta. [Cloetens et al., 2002](https://doi.org/10.1117/12.452867)
- **TIE** - Transport of Intensity Equation. Dual-distance algorithm using known alpha/beta. [Langer et al., 2008](https://doi.org/10.1118/1.2975224)
- **WTIE** - Weak Transport of Intensity Equation. Dual-distance algorithm using known alpha/beta. [Langer et al., 2008](https://doi.org/10.1118/1.2975224)

## Contributors
- [Jean-Baptiste Forien](https://github.com/jbforien)
- [K. Aditya Mohan](https://github.com/adityamnk)

## License
This project is licensed under the GPL, v2.0 License. LLNL-CODE-802730.