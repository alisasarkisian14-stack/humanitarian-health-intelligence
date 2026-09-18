# Humanitarian Health Intelligence

## Overview

Humanitarian Health Intelligence is a data science project exploring how public health, environmental, demographic and humanitarian data can be integrated to support predictive health-risk assessment.

Cholera is used as the initial use case because transmission and outbreak severity are influenced by multiple interacting factors, including water and sanitation conditions, population vulnerability, environmental conditions, health-system access and humanitarian disruption.

The project investigates whether these signals can be combined to identify geographic areas at elevated risk and provide a foundation for future predictive modelling.

## Analytical Framework

The analysis combines multiple categories of risk information:

- **Epidemiological** — reported cholera cases and deaths
- **WASH** — access to safe drinking water, sanitation and open defecation
- **Health-system** — immunisation coverage and continuity
- **Environmental** — rainfall and climatic conditions
- **Humanitarian** — conflict- and disaster-related displacement
- **Demographic** — population and population distribution

Exploratory analysis is used to assess data quality, geographic and temporal coverage, potential relationships between indicators and cholera burden, and the suitability of variables for later modelling.

## Data Sources

| Source | Primary Role |
|---|---|
| WHO | Cholera surveillance and WASH indicators |
| UNICEF / WHO WUENIC | Immunisation coverage |
| CHIRPS | Gridded precipitation data |
| IDMC | Conflict and disaster displacement |
| WorldPop | Gridded population estimates |

## Modelling Concept

The intended modelling framework is a binary risk-classification problem in which a geographic area is classified as either **high risk** or **not high risk** for a future prediction period.

Candidate predictors include population-normalised disease burden, rainfall, WASH conditions, immunisation coverage, population characteristics and humanitarian displacement.

The final target definition and feature set will depend on appropriate geographic and temporal alignment of the underlying datasets.

## Data Architecture

The project follows a medallion-style data architecture:

`Raw Source Data → Silver Standardised Tables → Gold Modelling Dataset → Machine Learning / MLflow`

The **Raw** layer preserves original source data. The **Silver** layer will contain cleaned and standardised datasets with consistent identifiers, data types and temporal fields. The **Gold** layer will contain integrated, feature-engineered datasets prepared for modelling and reporting.

This separation supports reproducibility, data lineage and controlled transformation from source data to analytical outputs.

## Repository Structure

```text
humanitarian-health-intelligence/
├── README.md
├── notebooks/
├── data/
└── docs/
