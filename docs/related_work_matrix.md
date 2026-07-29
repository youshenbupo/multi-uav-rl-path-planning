# Related-work evidence matrix

This is an evidence ledger, not a claim of exhaustive literature coverage.  A
cell marked **not established** means that the currently reviewed primary
landing-page abstract or bibliographic record does not support the claim.  It
does not mean that the paper lacks the feature.  Full-paper inspection is
required before the manuscript makes a stronger distinction.

| Work | Primary evidence reviewed | What the reviewed source establishes | Relation to the mainline | Remaining verification gate |
| --- | --- | --- | --- | --- |
| Yuan et al., *DACOM: Learning Delay-Aware Communication for Multi-Agent Reinforcement Learning*, AAAI 2023, [DOI](https://doi.org/10.1609/aaai.v37i10.26389) | [AAAI proceedings landing page](https://ojs.aaai.org/index.php/AAAI/article/view/26389), abstract and metadata reviewed 2026-07-20; official PDF link rechecked 2026-07-29 but browser download timed out/no local PDF resulted | It treats communication delay as harmful to cooperative MARL and learns a waiting-time adjustment (TimeNet) to address delay-associated uncertainty. | Directly relevant delayed-communication MARL prior art.  The reviewed abstract does not establish predicted neighbour-state uncertainty in graph edge features, dynamic-obstacle multi-UAV experiments, or CBF safety-margin adaptation. | Full text remains unreviewed. Obtain/read the official PDF or an author-provided primary copy for observation model, experiments, and any graph/safety component before asserting a formal novelty gap. |
| Liu et al., *Deep Hierarchical Communication Graph in Multi-Agent Reinforcement Learning*, IJCAI 2023, [DOI](https://doi.org/10.24963/ijcai.2023/24) | [Official IJCAI proceedings page](https://www.ijcai.org/proceedings/2023/24), abstract and metadata reviewed 2026-07-29; its official PDF link was identified, but two browser download attempts timed out and produced no local PDF | Its abstract describes end-to-end learned message-based directed acyclic communication topologies, an acyclicity constraint/intrinsic reward plus projection, and policy/value variants evaluated on cooperative MARL benchmarks. | Directly relevant learned communication-graph prior art. The reviewed abstract does not establish intermittent packet delivery, predicted neighbour-state uncertainty in graph inputs, dynamic-obstacle UAV experiments, or execution-side CBF adaptation. | Full text remains unreviewed. Obtain/read the official IJCAI PDF or an author-provided primary copy for timing assumptions, graph inputs, and any safety component before asserting a formal novelty gap. |
| Deng et al., *Safe Multi-Agent Reinforcement Learning Through Neural Graph Control Barrier Functions*, AAMAS 2026, [DOI](https://doi.org/10.65109/krtj4225) | [Official AAMAS 2026 proceedings entry](https://ifaamas.org/Proceedings/aamas2026/forms/contents.htm) and [official 3-page extended-abstract PDF](https://ifaamas.org/Proceedings/aamas2026/pdfs/KRTJ4225.pdf) reviewed 2026-07-29 (SHA-256 `0AE5CA03242EDE2EB5E60822729EE6B6279F4FA4C14B836E411658260E09D29A`) | The extended abstract describes a Dec-POMDP architecture with a frozen MAPPO reference policy and a GNN-parameterized graph-CBF additive correction. It trains the correction by matching pointwise CBF-QP expert actions plus barrier losses, and evaluates boundary/collision constraints in modified MPE Simple Spread against MAPPO, MACPO, and MAPPO-Lagrangian. | Directly relevant neural graph-CBF prior art. The reviewed extended abstract describes neural correction distilled from QP supervision, whereas this project retains an online CPU OSQP shield with a communication-uncertainty-adaptive margin. It does not document communication staleness/delay/loss, delivered-packet prediction uncertainty, or dynamic-obstacle experiments. | This is an AAMAS extended abstract, not evidence for a universal safety claim or a formal novelty gap. If a longer archival version is located, inspect it before any stronger comparison; do not infer omitted communication/dynamic-obstacle details from the short text. |
| Thumiger and Deghat, *A Multi-Agent Deep Reinforcement Learning Approach for Practical Decentralized UAV Collision Avoidance*, IEEE Control Systems Letters 2022, [DOI](https://doi.org/10.1109/lcsys.2021.3138941) | [Crossref record](https://api.crossref.org/works/10.1109/LCSYS.2021.3138941), metadata only, reviewed 2026-07-21; DOI resolved to official IEEE Xplore on 2026-07-29 but the page returned `Unusual Traffic Detected` (HTTP 418) before abstract/PDF access | The record verifies a decentralized multi-UAV collision-avoidance MARL setting. | A high-priority domain comparator for UAV coordination. | Full text remains unreviewed. Obtain/read an authorized IEEE or author-provided primary copy for communication assumptions, dynamic-obstacle model, and safety mechanism before recording differences. |

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
