# Multi-Run Statistical Analysis Report

**Institution:** University of Lincoln  
**Department:** School of Computer Science  
**Analysis Date:** 2025-08-19T22:27:30.046033  
**Total Experimental Runs:** 18

## Executive Summary

This report presents a comprehensive statistical analysis of federated learning algorithms across multiple experimental runs, providing reliability assessments and deployment confidence metrics for zero-day botnet detection in IoT-edge environments.

### Key Statistical Findings

| Algorithm | Runs | Mean Accuracy | Std Dev | CV | Reliability |
|-----------|------|---------------|---------|----|-----------|
| FedAvg | 6 | 19.8% | 0.0% | 0.000 | High |
| FedProx | 6 | 20.7% | 0.0% | 0.000 | High |
| AsyncFL | 6 | 20.0% | 0.0% | 0.000 | High |


## Reliability Assessment

### Algorithm Reliability Classification

**FedAvg:** Highly Reliable  
*Deployment Recommendation:* Recommended for production deployment

**Key Risk Factors:**
- No significant risk factors identified

**Mitigation Strategies:**
- Consider implementing learning rate decay
- Add momentum to gradient updates
- Validate client selection strategies


**FedProx:** Highly Reliable  
*Deployment Recommendation:* Recommended for production deployment

**Key Risk Factors:**
- No significant risk factors identified

**Mitigation Strategies:**
- Fine-tune proximal term coefficient (μ)
- Optimize client sampling strategy
- Implement adaptive μ selection


**AsyncFL:** Highly Reliable  
*Deployment Recommendation:* Recommended for production deployment

**Key Risk Factors:**
- No significant risk factors identified

**Mitigation Strategies:**
- Optimize staleness handling mechanisms
- Implement bounded staleness controls
- Validate asynchronous aggregation timing

