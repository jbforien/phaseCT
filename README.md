# phaseCT

Holotomography is a non-destructive x-ray imaging technique which allows to visualize the interior morphology of object without damaging them. It is similar to x-ray computed tomography (CT), but unlike conventional CT, which is based on the attenuation of x-rays passing through the object, holotomography exploits the phase shift caused by the sample. This phase-contrast imaging allows to increase image contrast by recovering refractive index of materials and is proven to be more sensitive to density variations than conventional absorption-based X-ray imaging when applied to low-Z materials. In holotomography, the phase is retrieved by measuring the variations in intensity of the object at different sample to detector distances. Several algorithms exist to perform this task such as, contrast transfer function (CTF), transport of intensity equation (TIE), and a mixed approach which combines both TIE and CTF. 

### Prerequisites
numpy  
pyffw  
scipy  
skimage  
sklearn

## License
phaseCT is distributed under the terms of the GNU Public License v2.0 - see the LICENSE file for details.  
LLNL-CODE-802730
