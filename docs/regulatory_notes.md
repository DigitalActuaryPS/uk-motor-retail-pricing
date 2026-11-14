# Regulatory notes (illustrative project, Simulated data)

This project is a portfolio exercise on simulated data. It is not legal advice and it does not demonstrate compliance with any rule. Section 1 lists genuine UK requirements that I checked against FCA pages or secondary summaries on 14 November 2026. Section 2 lists controls that exist only inside this project. Section 3 lists items I did not verify.

Verification key: **FCA primary** means an FCA Handbook or FCA publication page returned by search. **Secondary** means a law firm, consultancy or training site summarising the rule. Secondary-only items should be read against the Handbook before anyone relies on them.

## 1. Genuine UK requirements that bear on retail motor pricing

| Requirement | What it says (as found) | Verification | Where it touches this project |
|---|---|---|---|
| ICOBS 6B, home and motor insurance pricing | Rules in force from 1 January 2022 (PS21/5, amended by PS21/11) covering motor and home renewal pricing, add-ons sold with them, and fees at renewal. ICOBS 6B states it is not intended to affect how risk is priced. | FCA primary (Handbook ICOBS 6B page, FCA attestation review) | Renewal price cap and renewal versus new business comparison in M6 and M5 |
| ICOBS 6B.2.1R, equivalent new business price (ENBP) | Renewal price must not exceed the equivalent new business price, tested when the renewal notice is prepared. | Secondary (Learnsignal summary). The FCA-hosted summaries found describe the ban on price walking and the ENBP concept but I did not retrieve the rule text. | See section 2: the project does not implement an ENBP test |
| ICOBS 6B.2.51R, records | Firms must keep written records of how they continue to satisfy the pricing rules. | FCA primary (PS21/5 Q&A) | Approval template and risk register in M8 are examples of record shapes only |
| ICOBS 6B.2.60R, annual attestation | Firms subject to the pricing rules attest annually that they comply and that pricing does not discriminate against customers of longer tenure. Senior manager (or director for non-SM&CR firms) accountability was an FCA finding. | FCA primary (FCA attestation multi-firm review) | None. The project produces no attestation |
| PRIN 2A, Consumer Duty, price and value outcome (PRIN 2A.4) | Fair value means the amount paid is reasonable relative to the benefits of the product. Applies to retail customers. For non-investment insurance scope follows ICOBS. | FCA primary (Handbook PRIN 2A.4 page) | Context for the illustrative fair-value style checks in section 2 |
| PROD 4, product governance, fair value assessment (PROD 4.2.14AR onwards) | Enhanced product governance rules in force from 1 October 2021 for manufacturers and distributors of general insurance and pure protection products. They include fair value assessments. TR24/2 (August 2024) found many firms were not fully meeting PROD 4. | FCA primary (FCA PS21/11 page, TR24/2, Handbook PROD 4.2 page) | Governance outputs in M8 are examples of monitoring, not a fair value assessment |
| Relationship between PROD 4 and the Consumer Duty | An FCA multi-firm review states that complying with PROD 4 fair value rules would satisfy the Duty's price and value outcome for those products. A law firm summary says the Duty's products and services and price and value outcomes are not applied to products already covered by those rules. The two sources are consistent in direction but phrased differently. | FCA primary and secondary | Not modelled |

## 2. Illustrative controls in this project (not requirements, not compliance)

| Control in the repo | What it does | What it is not |
|---|---|---|
| Renewal price change cap (config `optimisation.renewal_price_change_cap`) | Limits the average renewal price change per segment in the optimiser | Not an ENBP test. ENBP compares each renewing customer's price with the price the same customer would get as a new business quote through the same channel. This project works at segment average level on simulated data |
| Renewal versus new business comparison table (`m6_illustrative_renewal_vs_nb_control`) | Compares average renewal and new business premium by segment | Not evidence of ICOBS 6B compliance |
| Rank deterioration cap | Limits worsening of market position by segment | A commercial control only |
| Margin floor, volume floors | Business constraints in the optimiser | Commercial controls only |
| Price change approval template | Example sign-off structure | Not an FCA-approved format and not an attestation |
| Model risk register (R01 to R14) | Lists limitations of this project's models | Not a regulatory register |
| PSI, out-of-time AUC and stress tests | Monitoring examples | Not a regulatory requirement as implemented here |

## 3. Unverified, check before relying

These are topics a pricing team would normally consider. I did not verify them in this session and the repo does not implement them.

- Equality Act 2010 and the treatment of protected characteristics in pricing, including gender: unverified, check before relying.
- UK GDPR and Data Protection Act 2018 requirements on automated decision making and profiling, and on the data used for price optimisation: unverified, check before relying.
- Use of price optimisation and elasticity models inside the ICOBS 6B ENBP framework, including how ENBP is calculated per channel and how optimisation interacts with it: unverified, check before relying. The ENBP rule text itself was not retrieved from the Handbook.
- Senior Managers and Certification Regime responsibilities for pricing decisions: unverified, check before relying.
- Prudential model risk expectations for insurers (for example PRA material): unverified, check before relying. The model risk register here is a project artefact.
- FCA pricing data submission and value measures reporting obligations: unverified, check before relying.
- Any change to these rules after the dates on the pages I saw (ICOBS 6B page last updated 1 August 2025, PRIN 2A.4 page last updated 9 December 2025, PROD 4.2 page last updated 26 June 2026): unverified, check before relying.

## Sources

- FCA Handbook, ICOBS 6B: https://handbook.fca.org.uk/handbook/icobs6b
- FCA, General insurance pricing attestation multi-firm review: https://www.fca.org.uk/publications/multi-firm-reviews/general-insurance-pricing-attestation
- FCA, PS21/5 Q&As on the published rules: https://www.fca.org.uk/publication/policy/ps21-5-questions-answers.pdf
- FCA, PS21/11 General insurance pricing practices, amendments: https://www.fca.org.uk/publications/policy-statements/ps21-11-general-insurance-pricing-practices-amendments
- FCA Handbook, PRIN 2A.4: https://handbook.fca.org.uk/handbook/prin2a/prin2as4
- FCA Handbook, PROD 4.2: https://handbook.fca.org.uk/handbook/prod4/prod4s2
- FCA, TR24/2 product governance thematic review: https://www.fca.org.uk/publication/thematic-reviews/tr24-2.pdf
- FCA, Price and value outcome good and poor practice: https://www.fca.org.uk/publications/good-and-poor-practice/price-value-outcome-good-poor-practice-update
- Learnsignal, ICOBS 6B guide (secondary, source for the ENBP wording): https://www.learnsignal.com/blog/general-insurance-pricing-icobs-6b-equivalent-new-business-price-guide/
- Norton Rose Fulbright, FCA Consumer Duty for insurance (secondary): https://nortonrosefulbright.com/en/knowledge/publications/07649343/fca-consumer-duty-for-insurance
