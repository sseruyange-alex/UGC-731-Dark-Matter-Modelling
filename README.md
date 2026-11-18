
# Modelling Dark Matter Distribution in Galaxy UGC 731

### Using Hybrid Optical and HI 21-cm Rotation Curves

## Overview

This repository contains the computational implementation and research materials for an astrophysics study investigating the dark matter distribution in the dwarf irregular galaxy UGC 731 (DDO 9).

The study combines optical H-alpha rotation curve measurements from the GHASP survey with HI 21-cm kinematic data from the WHISP survey to construct a hybrid rotation curve.

The observed galactic dynamics are analysed through baryonic decomposition and dark matter halo modelling.

## Research Objectives

1. Construct a hybrid optical and HI 21-cm rotation curve.
2. Determine the stellar and gas contributions to the gravitational potential.
3. Model the dark matter halo using Burkert and NFW density profiles.
4. Compare the halo models using chi-square, AIC and BIC.
5. Investigate the sensitivity of the results to uncertainties in the baryonic components and observational data.

## Methodology

- Optical rotation curve analysis.
- HI 21-cm data processing and surface density extraction.
- WISE W1 stellar photometry.
- Baryonic mass decomposition.
- Dark matter halo parameter estimation.
- Statistical model comparison and sensitivity analysis.

## Computational Tools

- Python
- NumPy
- SciPy
- Matplotlib
- Astropy
- Jupyter / scientific Python environment

## Main Models

- Burkert dark matter halo (cored profile).
- Navarro-Frenk-White (NFW) dark matter halo (cuspy profile).
- Generalised NFW halo.

## Results

The analysis compares the ability of cored and cuspy halo models to reproduce the observed rotation curve of UGC 731.

The findings indicate that the available observational data do not decisively distinguish between the two halo structures.

## Repository Contents

- `thesis/`: Research thesis and documentation.
- `code/`: Python modelling pipeline.
- `data/`: Observational and processed datasets.
- `results/`: Model outputs, figures and sensitivity analysis.
- `references/`: Scientific references and data provenance.

## Data Sources

- GHASP survey: Optical H-alpha rotation curve.
- WHISP survey: HI 21-cm observations.
- NASA/IPAC Infrared Science Archive: WISE W1 imaging.

## Author

Research conducted at Makerere University,
Department of Physics.

## Project Status

Completed undergraduate research project.
