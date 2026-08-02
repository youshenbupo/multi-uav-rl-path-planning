# Related-work evidence matrix

This is an evidence ledger, not a claim of exhaustive literature coverage.
Statements below are limited to the named primary source and source tier. A
missing feature in one reviewed paper does not establish a field-wide novelty
claim.

| Work | Primary evidence reviewed | What the source establishes | Evidence-bounded relation to the mainline | Remaining verification gate |
| --- | --- | --- | --- | --- |
| Yuan et al., *DACOM: Learning Delay-Aware Communication for Multi-Agent Reinforcement Learning*, AAAI 2023, [DOI](https://doi.org/10.1609/aaai.v37i10.26389) | [AAAI proceedings page](https://ojs.aaai.org/index.php/AAAI/article/view/26389) and official nine-page PDF, fully reviewed 2026-08-02 | Sections *Communication Channels*, *Delay-Aware Modeling*, *Architecture Design*, and *Training* model end-to-end delay or bitrate as network state, allow recent network observations such as a weighted moving average to estimate current delay, and use TimeNet to choose how long an agent waits for messages. Messages arriving within the threshold are aggregated; otherwise a latest buffered message can be used. The delay-aware critic conditions on observations, network state, actions, and delays. Experiments cover particle games, traffic control, and SMAC under mean delay ratios; the paper reports collision and arrival rates in traffic control and explicitly notes weak gains in delay-insensitive, very-low-delay, and very-high-delay regimes. | Direct delayed-communication MARL prior art. DACOM adapts message waiting to channel state. The reviewed method does not introduce a predictive neighbour-motion interaction graph, calibrated prediction uncertainty propagated into graph features, packet-loss experiments, or an execution-side CBF shield. Its traffic experiment evaluates collisions through task reward/metrics rather than a CBF-enforced safety margin. | This row now supports a paper-specific method distinction, not a claim that no other work combines these elements. Broader systematic coverage remains required before any “first” or exhaustive novelty statement. |
| Liu et al., *Deep Hierarchical Communication Graph in Multi-Agent Reinforcement Learning*, IJCAI 2023, [DOI](https://doi.org/10.24963/ijcai.2023/24) | [Official IJCAI proceedings page](https://www.ijcai.org/proceedings/2023/24) and official nine-page PDF, fully reviewed 2026-08-02 | Sections 3--4 learn message-dependent agent dependencies as a directed acyclic graph. Graph selection is treated as an action; an acyclicity penalty and critic train it, followed by projection into an admissible DAG set. DHCG-P performs autoregressive intention sharing and DHCG-Q factorizes joint value into individual utilities and pairwise payoffs. Section 6 reports five independent seeds on Predator-Prey, MACO, and SMAC, including topology and communication-method comparisons. | Direct learned communication-graph prior art. Its graph input is observation-action history plus an initial decision/intention, and its contribution is dependency/order and acyclicity. The reviewed method does not model communication age, delay, packet loss, delivered-packet neighbour-state prediction uncertainty, UAV dynamic obstacles, or execution-side CBF adaptation. | This row supports a paper-specific distinction only. It does not establish field-wide novelty, safety superiority, or applicability under intermittent communication. |
| Deng et al., *Safe Multi-Agent Reinforcement Learning Through Neural Graph Control Barrier Functions*, AAMAS 2026, [DOI](https://doi.org/10.65109/krtj4225) | [Official AAMAS proceedings entry](https://ifaamas.org/Proceedings/aamas2026/forms/contents.htm) and official three-page extended-abstract PDF, reviewed 2026-07-29 and hash-reverified 2026-08-02 | The extended abstract describes a Dec-POMDP architecture with a frozen MAPPO reference policy and a GNN-parameterized graph-CBF additive correction. It trains the correction from pointwise CBF-QP expert actions plus barrier losses, then evaluates boundary/collision constraints in modified MPE Simple Spread against MAPPO, MACPO, and MAPPO-Lagrangian. | Direct neural graph-CBF prior art. The reviewed short paper distills QP supervision into a neural correction, whereas this project retains an online CPU OSQP execution shield whose margin is adapted from communication uncertainty. The short paper does not document communication staleness/delay/loss, delivered-packet prediction uncertainty, or dynamic-obstacle experiments. | Only the official extended abstract is available in the reviewed indexes; OpenAlex reports no repository full text. Do not infer omitted details or use the short version for a universal safety/novelty claim. Inspect a longer archival version if one appears. |
| Thumiger and Deghat, *A Multi-Agent Deep Reinforcement Learning Approach for Practical Decentralized UAV Collision Avoidance*, IEEE Control Systems Letters 2022, [DOI](https://doi.org/10.1109/lcsys.2021.3138941) | [Crossref record](https://api.crossref.org/works/10.1109/LCSYS.2021.3138941), ORCID affiliation record, and [OpenAlex record](https://api.openalex.org/works/https://doi.org/10.1109/LCSYS.2021.3138941), reviewed 2026-08-02; full text unavailable | Publisher metadata verifies the venue, pages 2174--2179, authors, and UNSW affiliation. The indexed abstract says the decentralized collision-avoidance controller uses an LSTM-based deep-RL architecture, is designed for variable agent counts, and is tested in simulation and a real three-dimensional drone environment. | High-priority decentralized multi-UAV collision-avoidance comparator. Because the full text could not be retrieved, no paper-specific distinction is recorded for communication assumptions, obstacle dynamics, uncertainty, or safety mechanism. | IEEE Xplore returned HTTP 418; the Crossref publisher PDF URL and IEEE stamp PDF also yielded no file. OpenAlex marks the work closed access with no repository full text. Obtain an authorized publisher or author copy before making any method-difference claim. |

## Preserved primary-source artifacts

- `outputs/literature_primary_sources_20260802/Yuan_et_al_2023_DACOM_AAAI_official.pdf`
  - SHA-256: `E186B67D9558F117F6C83BBC75A38D6616825BA257DE6739A685E962DAA11EC9`
  - nine pages; PDF magic and all page renders verified.
- `outputs/literature_primary_sources_20260802/Liu_et_al_2023_DHCG_IJCAI_official.pdf`
  - SHA-256: `310BAA7F8446EEF14A69D9516046627DBE8ED2EF3A564A5E1CE70BDFF6FCD652`
  - nine pages; PDF magic and all page renders verified.
- `outputs/literature_primary_sources_20260802/Deng_et_al_2026_neural_graph_CBF_AAMAS_extended_abstract_official.pdf`
  - SHA-256: `0AE5CA03242EDE2EB5E60822729EE6B6279F4FA4C14B836E411658260E09D29A`
  - official three-page extended abstract; hash matches the prior review record.

The failed IEEE retrieval paths produced no local PDF and therefore no source
artifact to hash. Temporary text and page renders used for PDF inspection were
created under `tmp/pdfs/` and removed after review; they are not research
outputs.

## Current defensible positioning

Primary-source review now supports two narrow paper-specific comparisons:

1. DACOM adapts message waiting to observed/estimated channel delay, whereas
   this project predicts delivered neighbour packets forward and propagates
   age-derived uncertainty into both graph risk features and an online CBF
   margin.
2. DHCG learns an acyclic intention/dependency topology, whereas this project's
   predictive interaction graph is driven by stale delivered motion state and
   communication uncertainty under delay/loss.

The AAMAS extended abstract establishes graph-CBF prior art, and the IEEE
abstract establishes practical decentralized multi-UAV DRL prior art. Neither
source tier is sufficient for a strong exclusionary comparison. The manuscript
must therefore present the contribution as a tested, auditable integration and
must not claim to be first, exhaustive, formally safe, statistically superior,
or robustly generalizing.

The fixed working contribution remains:

1. predict delivered neighbour packets forward using timestamp and velocity;
2. propagate packet-age uncertainty into predictive interaction-graph risk;
3. use the same uncertainty to adapt the execution-side CBF margin; and
4. evaluate the complete chain under dynamic obstacles and six communication/
   obstacle conditions with independently trained seeds and retained raw
   telemetry.
