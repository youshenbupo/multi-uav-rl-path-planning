# Related-work evidence matrix

This is an evidence ledger, not a claim of exhaustive literature coverage.  A
cell marked **not established** means that the currently reviewed primary
landing-page abstract or bibliographic record does not support the claim.  It
does not mean that the paper lacks the feature.  Full-paper inspection is
required before the manuscript makes a stronger distinction.

| Work | Primary evidence reviewed | What the reviewed source establishes | Relation to the mainline | Remaining verification gate |
| --- | --- | --- | --- | --- |
| Yuan et al., *DACOM: Learning Delay-Aware Communication for Multi-Agent Reinforcement Learning*, AAAI 2023, [DOI](https://doi.org/10.1609/aaai.v37i10.26389) | [AAAI proceedings landing page](https://ojs.aaai.org/index.php/AAAI/article/view/26389), abstract and metadata reviewed 2026-07-20; official PDF link rechecked 2026-07-29 but browser download timed out/no local PDF resulted | It treats communication delay as harmful to cooperative MARL and learns a waiting-time adjustment (TimeNet) to address delay-associated uncertainty. | Directly relevant delayed-communication MARL prior art.  The reviewed abstract does not establish predicted neighbour-state uncertainty in graph edge features, dynamic-obstacle multi-UAV experiments, or CBF safety-margin adaptation. | Full text remains unreviewed. Obtain/read the official PDF or an author-provided primary copy for observation model, experiments, and any graph/safety component before asserting a formal novelty gap. |
| Liu et al., *Deep Hierarchical Communication Graph in Multi-Agent Reinforcement Learning*, IJCAI 2023, [DOI](https://doi.org/10.24963/ijcai.2023/24) | [Crossref record](https://api.crossref.org/works/10.24963/ijcai.2023/24), deposited abstract and metadata reviewed 2026-07-20 | It learns directed acyclic communication topologies from messages, with topology selection trained end-to-end and tested on cooperative MARL benchmarks. | Directly relevant learned communication-graph prior art.  The reviewed abstract frames topology/dependency and message cost, not intermittent packet delivery or execution-side safety. | Inspect the IJCAI PDF for timing assumptions and graph inputs before claiming that its graph is not uncertainty-aware. |
| Deng et al., *Safe Multi-Agent Reinforcement Learning Through Neural Graph Control Barrier Functions*, AAMAS 2026, [DOI](https://doi.org/10.65109/krtj4225) | [Crossref record](https://api.crossref.org/works/10.65109/krtj4225), metadata only, reviewed 2026-07-20 | The record verifies title, authors, AAMAS 2026 venue, and the neural graph-CBF topic. | A high-priority safety/graph comparator for the paper's CBF discussion. | Obtain and read the primary paper before recording method, assumptions, guarantees, or claimed differences. |
| Thumiger and Deghat, *A Multi-Agent Deep Reinforcement Learning Approach for Practical Decentralized UAV Collision Avoidance*, IEEE Control Systems Letters 2022, [DOI](https://doi.org/10.1109/lcsys.2021.3138941) | [Crossref record](https://api.crossref.org/works/10.1109/LCSYS.2021.3138941), metadata only, reviewed 2026-07-21 | The record verifies a decentralized multi-UAV collision-avoidance MARL setting. | A high-priority domain comparator for UAV coordination. | Inspect the IEEE primary paper for communication assumptions, dynamic-obstacle model, and safety mechanism before recording differences. |

## Current defensible positioning

The manuscript may currently state only the following evidence-bounded
motivation: delayed communication and learned communication graphs are each
established MARL topics, while graph-CBF safety is an active AAMAS-area topic.
It must **not yet** state that this project is the first to combine any pair of
these ideas.  The working contribution is instead framed as a testable design:

1. predict delivered neighbour packets forward using timestamp and velocity;
2. propagate packet-age uncertainty into predictive interaction-graph risk; and
3. use the same uncertainty to adapt the execution-side CBF margin in dynamic
   obstacle evaluations.

The novelty claim remains conditional on full-paper comparison and independent
five-seed evidence.  The next literature gate is to inspect primary PDFs for
the three entries above, then add a multi-UAV communication/safety section with
the same evidence discipline.
