Title: CPextractor: evidence-grounded extraction and curation of crystal plasticity parameters from the literature
Subtitle: Submission-ready manuscript draft grounded in the current project snapshot
Authors: [Author Name 1]^1, [Author Name 2]^1, [Author Name 3]^2,*
Affiliations: ^1[Department / Institute, University, City, Country]; ^2[Department / Institute, University, City, Country]
Correspondence: *Correspondence to: [corresponding.author@email.edu]

# Abstract

Crystal plasticity models are widely used to connect microstructure, crystallography and deformation, yet the parameter sets required to instantiate these models remain dispersed across prose, tables, captions, equations and supplementary files. This fragmentation slows model reuse, obscures provenance and makes cross-paper comparison difficult. The central challenge is not simply to recover reported values, but to recover them with the constitutive context that gives them scientific meaning, including the material state, phase, loading condition, model family, scope of applicability and supporting evidence. Here we present CPextractor, an evidence-grounded literature-to-database pipeline for crystal plasticity parameter curation. CPextractor integrates corpus acquisition, XML-first full-text parsing with PDF fallback, two-stage large language model extraction, deterministic normalization and binding, claim-level evidence grounding, multi-agent review, confidence fusion and gated database ingestion. In the current repository snapshot, the system processed 288 full-text papers and produced 6,093 normalized parameter claims, all linked to recoverable text or table evidence. Of these, 4,335 claims were retained as high-confidence after review and 2,864 claims from 185 papers passed the ingestion gate into the database-ready layer. These corpus-scale results show that CPextractor can convert heterogeneous crystal plasticity papers into auditable scientific records while preserving explicit trust signals for downstream reuse. By treating extraction, provenance, uncertainty and database integration as one workflow, CPextractor provides a practical foundation for reusable parameter infrastructure, evidence-grounded retrieval and more reproducible integrated computational materials engineering.

Keywords: crystal plasticity; scientific information extraction; evidence grounding; materials informatics; parameter database; reproducibility

# Introduction

Crystal plasticity has become one of the most important constitutive frameworks for interpreting anisotropic deformation across structurally complex materials. It is routinely used to link crystallography, phase constitution, texture and microstructural state to slip activity, hardening, strain localization and macroscopic response. In integrated computational materials engineering workflows, its value is not only conceptual but operational: crystal plasticity models are often the practical bridge between experimental characterization, mesoscale constitutive reasoning and predictive simulation.

That usefulness, however, depends on access to interpretable parameter sets. A model cannot be reused from the literature unless the relevant elastic constants, rate sensitivities, hardening terms, interaction coefficients and calibration conditions can be recovered with sufficient context to determine what the reported values actually mean. In practice, this remains a major bottleneck. Researchers still assemble these inputs manually by reading papers, extracting values from tables, reconciling notation across studies and inferring constitutive meaning from surrounding text. The process is slow, difficult to audit and poorly suited to scale.

The difficulty is deeper than simple value extraction. In crystal plasticity papers, a reported quantity is rarely a standalone fact. The same symbol may map to different constitutive roles across papers, whereas different symbols may refer to the same physical quantity when conditioned on a particular model form. Critical qualifiers such as material state, phase, crystal structure, strain rate, temperature, hardening law, solver context and provenance are often distributed across paragraphs, captions, equations and table headers rather than stated in one place. A parameter record that omits these links may be numerically correct but scientifically unusable.

Existing resources do not fully address this gap. Large materials databases have transformed access to compositions, structures and computed or measured properties, but they are not designed to store grounded crystal plasticity parameter bundles with explicit document provenance. Likewise, scientific text-mining systems in materials science have demonstrated that domain-aware natural language processing can recover entities such as compositions, synthesis conditions and measured properties. However, constitutive simulation parameters impose stricter requirements because they must remain attached to model form, scope and supporting evidence to be reusable. Generic entity extraction is therefore insufficient.

We developed CPextractor to address this specific problem. Rather than treating literature mining as value harvesting, CPextractor treats the minimal reusable scientific unit as a parameter claim: a value attached to a material context, constitutive role, provenance trace and evidence source. The pipeline integrates structured full-text parsing, staged large language model extraction, deterministic normalization, evidence grounding, multi-agent review and database gating so that low-trust outputs can be retained for audit while higher-trust records are selectively admitted for reuse. This design positions the project at the intersection of scientific information extraction and research data infrastructure.

The manuscript is organized around that systems contribution. We first define the claim-centric representation and explain why it is necessary for crystal plasticity literature. We then show how CPextractor operationalizes this representation through a layered extraction and trust workflow. Next, we report corpus-scale outputs from the current repository snapshot, demonstrating that the system can transform hundreds of full-text papers into evidence-linked claims and selectively admit only a subset into the final database-ready layer. Finally, we discuss what this architecture enables for scientific reuse, where its current boundaries lie and what is required for a fully benchmark-locked submission.

# Results

## CPextractor is designed around reusable parameter claims rather than isolated extracted values

The core design decision in CPextractor is to make the parameter claim, not the raw extracted mention, the primary scientific unit. This distinction matters because crystal plasticity parameters are only useful when their constitutive meaning is preserved. A number reported in a table becomes reusable only when its identity, unit, scope, material binding, model context and evidence source are all recoverable.

This requirement shaped the project architecture. CPextractor first extracts candidate parameter records into an intermediate `parameters.registry` layer, but these records are not treated as final outputs. They are subsequently normalized, linked to materials and phases, bound to condition and mechanism scope, grounded to evidence and converted into atomic `parameter_claims`. The final document representation therefore includes not only claims, but also the support graph needed to interpret them, including materials, phases, process states, conditions, models, mechanisms and reusable evidence objects.

This claim-centric design also addresses one of the main structural difficulties of crystal plasticity literature: constitutive meaning is often distributed across document elements. A parameter may be named in an equation, quantified in a table, qualified in a caption and contextualized in method text. CPextractor is designed to preserve these links rather than flatten them into decontextualized fields. In that sense, the system is not merely an extractor; it is a curation pipeline that formalizes how simulation-ready literature knowledge should be represented.

## The extraction workflow combines high-recall candidate recovery with explicit trust filtering

CPextractor uses a staged architecture to separate broad candidate recovery from final trust assignment. The first stage of the workflow focuses on document acquisition and structure preservation. Papers are gathered from DOI lists, Scopus-assisted discovery or local full-text folders, then decomposed into sections, tables, references and related evidence-bearing assets. The project is XML-first, reflecting the importance of preserving native structure in parameter-rich papers, but includes a PDF fallback route when structured publisher XML is unavailable.

The second stage uses a two-step large language model extraction workflow. A selector first identifies the sections and tables most likely to contain crystal plasticity parameters, thereby constraining the extraction context to evidence-bearing regions. An extractor then returns schema-constrained records from those selected fragments. This staged design is important for long, heterogeneous papers, because it reduces the risk that distant discussion text or unrelated numerical content contaminates parameter interpretation.

Raw predictions are subsequently refined by deterministic post-processing. These steps include parameter identity normalization, unit normalization, provenance normalization, condition binding, model-equation linkage and evidence grounding. Together, they convert extraction candidates into interpretable scientific objects. Finally, a layered quality-control stack applies rule validation, evidence grounding checks, multi-agent review and confidence fusion. This enables the system to preserve high recall at the extraction stage while still being conservative at the ingestion stage.

## Corpus-scale processing yields thousands of evidence-linked claims, but the system remains selective about what enters the final database layer

To evaluate whether this architecture can support literature-scale curation, we examined the current repository snapshot across the local full-text corpus. CPextractor processed 288 full-text papers, with raw extraction artifacts available for 287 papers and review artifacts for 287 papers. Across this corpus, the pipeline produced 6,062 raw parameter records and 6,093 normalized parameter claims.

The difference between raw record count and final claim count reflects an important feature of the project: post-processing is not merely a cleaning step, but also a scientific restructuring step. The system is designed to transform extraction output into a more reusable representation, which can sometimes refine how parameter-level units are counted and grouped.

Every normalized claim in the current snapshot remained linked to recoverable text or table evidence, yielding 6,093 evidence-linked claims. This is one of the most consequential outcomes in the present repository state. Provenance is not an optional accessory for constitutive parameters. A value that cannot be traced back to a source sentence, table cell or supporting snippet is difficult to audit and risky to reuse in simulation.

The project then applies a deliberately conservative review filter. Of the 6,093 normalized claims, 4,335 were retained as high-confidence after committee review, corresponding to 71.1% of normalized claims. After document-level gating, 2,864 claims from 185 papers remained database-ready, corresponding to 47.0% of normalized claims and 64.2% of processed papers. More than one third of processed papers were therefore prevented from entering the final ingestion layer.

This selectivity is a feature, not a defect. A scientific curation pipeline should not maximize admitted output at any cost. It should make a clear distinction between high-recall candidate extraction and higher-trust reusable records. The current corpus run shows that CPextractor behaves in exactly this way: it extracts broadly, retains audit information for ambiguous cases and admits only a stricter subset into the curated database layer.

## Document-level verdicts reveal the practical value of the layered quality-control stack

The project’s document-level verdicts further clarify how trust is enforced in practice. Among the 287 papers with review outputs, 113 received accepted document verdicts, 167 were flagged and 7 were rejected. At the ingestion layer, 102 of 288 processed papers were blocked, corresponding to a blocked rate of 35.4%.

These outcomes show why CPextractor should not be interpreted as a single-pass extraction model. The system includes four distinct quality layers: deterministic rule validation, evidence grounding, multi-agent large language model review and confidence fusion. Each layer serves a different role. Rule checks capture issues such as unit inconsistency, empty extraction states or local structural failures. Grounding checks test whether the reported claims remain tied to real evidence. Review agents probe support, normalization and internal consistency. Confidence fusion turns these heterogeneous signals into operational decisions about trust.

For end users, this separation is valuable because it supports two different scientific workflows at once. First, uncertain or low-quality outputs remain inspectable, which is useful for curator review and error analysis. Second, database-ready records are filtered more aggressively, which is necessary for downstream retrieval, comparison and model initialization. The ability to preserve auditability without collapsing it into automatic admission is one of the project’s main practical contributions.

## The data model and retrieval stack translate curated claims into scientific infrastructure

CPextractor is designed not only to extract and review records, but to serve them. Accepted claims are persisted in a PostgreSQL-backed schema that stores papers, structured extractions, text chunks, references, evaluation artifacts and vectorized scientific content. This enables both exact structured retrieval and evidence-aware semantic retrieval.

The structured layer supports targeted scientific queries, such as filtering for parameters under a specific material class, phase condition or model family. The vector and chunk layers support retrieval-augmented question answering over both local text and parameter-level representations. In the current implementation, the chatbot retrieves from text chunks, parameter vectors and structured claim content, then synthesizes answers under evidence constraints. This makes the database useful not only for storage, but also for exploration and scientific interrogation.

Viewed at the systems level, the contribution is therefore broader than parameter extraction alone. CPextractor links literature acquisition, structured parsing, scientific normalization, uncertainty-aware review, curation decisions and retrieval into one connected workflow. That integration is what makes the project suitable as both a database systems paper and a scientific information extraction paper.

## The annotation and benchmarking framework is in place, but the final claim-level benchmark requires lock-in before journal submission

The repository already contains a substantial benchmark and annotation framework. It includes draft-export tools for claim-level annotation, compact spreadsheet-based review workflows, bundle-level gold definitions, difficulty-aware pilot packet generation and scripts for claim benchmarking, gate benchmarking and slice-specific evaluation. The current snapshot also contains a pilot gold set comprising 1,709 annotated claim rows.

This infrastructure is strategically important because it defines the path from an operational pipeline to a reviewer-ready validation section. The project is not limited to reporting throughput or corpus counts. It is already structured to support claim precision, recall, F1, grounding accuracy, unit accuracy, mapping accuracy, binding accuracy, provenance accuracy and bundle completeness.

However, the current benchmark outputs in the repository are not yet in a form that can responsibly support final journal claims about extraction accuracy. For that reason, the present manuscript does not invent or overstate unsupported precision and recall numbers. Instead, it focuses on the aspects that are already directly defensible from the repository state: the claim-centric system design, corpus-scale evidence-linked outputs, layered trust filtering and curated database admission. Once benchmark alignment is locked, the formal claim-level validation, baseline comparison and ablation package should be inserted into this section.

# Discussion

CPextractor reframes literature mining for constitutive modeling as a curation problem rather than a pure extraction problem. This is the central conceptual advance of the project. The scientific bottleneck is not simply that values are difficult to find. It is that values are difficult to reuse unless their constitutive role, scope, provenance and evidence remain intact. The project addresses this by turning extraction into one stage within a broader workflow for claim construction and trust assignment.

This perspective clarifies why the present corpus-scale results matter. A system that reports many candidate values but cannot justify them is of limited value for scientific reuse. By contrast, a system that maintains evidence links, attaches explicit review signals and selectively admits only a more trustworthy subset of records can support tasks that actually matter to modelers: parameter search, cross-paper comparison, retrieval with source inspection and uncertainty-aware reuse. The current snapshot already shows that such a pipeline can operate across hundreds of full-text papers.

The project also suggests a more general design pattern for scientific data infrastructure. Crystal plasticity is a demanding test case because its literature is notation-heavy, model-conditioned and structurally heterogeneous. But the same combination of structured parsing, staged extraction, deterministic binding, evidence grounding and confidence-gated curation is likely to be valuable in adjacent domains such as phase-field modeling, creep and fatigue constitutive law extraction, fracture parameter databases and multiscale process-model curation.

Several boundaries remain important to state explicitly. First, the strongest present evidence supports the systems architecture and corpus-scale curation behavior more strongly than final extraction-accuracy claims. Second, full-text quality remains a constraint; XML-native papers preserve structure much better than fallback PDF reconstructions. Third, confidence should not be conflated with physical truth. A grounded claim can still be model-specific or condition-limited. Fourth, literature-derived databases must continue to navigate publisher licensing constraints, which makes evidence-pointer releases and corpus reconstruction workflows especially important.

These boundaries also define the next steps. The most immediate requirement for a fully submitted version is a benchmark-locked validation package with claim-level accuracy, slice analysis, judge calibration and baseline comparisons. Beyond that, equation-aware parsing, richer supplement integration, solver-aware replay studies and expert-in-the-loop refinement could further strengthen both the scientific and practical value of the system. Even in its current form, however, CPextractor establishes a credible route from unstructured constitutive literature to evidence-grounded, queryable and reviewable scientific data assets.

# Methods

## Overview

CPextractor is a literature-to-database workflow for extracting, normalizing, auditing and serving crystal plasticity parameters from full-text papers. The pipeline is modular by design so that document acquisition, parsing, extraction, post-processing, review and downstream serving can evolve without changing the central scientific unit of the system, namely the evidence-grounded parameter claim.

The system is organized around three linked ideas. First, document structure should be preserved as early as possible because constitutive meaning is often distributed across sections, tables and captions. Second, extraction should be staged so that evidence localization and schema-constrained interpretation are separated. Third, curation should be trust-aware, meaning that not every extracted value is automatically eligible for database admission.

## Corpus acquisition and full-text organization

CPextractor accepts three entry routes into the corpus: DOI lists, Scopus-assisted collection and local full-text folders. For each target paper, the pipeline constructs a stable local paper directory so that full text, sections, tables, references and downstream artifacts can be associated under one DOI-linked workspace.

When publisher XML is available, it is treated as the preferred source representation. XML articles are decomposed into local assets including paper metadata, section text, structured tables, equations and references. This representation preserves table hierarchy, caption context and local document structure that would otherwise be difficult to reconstruct from flattened text.

When XML is unavailable, the project falls back to PDF-oriented reconstruction. This route is inherently more error-prone because tables, equations and local structural cues may be partially degraded during document recovery. Nevertheless, the pipeline maintains a common downstream interface so that extraction and post-processing modules can operate on both XML-first and PDF-derived paper assets.

## Two-stage evidence selection and schema-constrained extraction

The extraction engine uses a staged large language model strategy. In the first stage, a selector identifies the sections and tables most likely to contain crystal plasticity parameters. This step reduces context length and helps prevent the detailed extractor from drawing unsupported links from distant discussion text or unrelated numerical content.

In the second stage, the extractor operates on those selected fragments and returns schema-constrained outputs. The extraction schema is designed to answer five scientific questions for each record: what material is studied, under which process state or loading condition, under which model or constitutive framework, over what phase or mechanism scope, and with what direct supporting evidence.

The schema has evolved substantially across project versions, but its guiding principle has remained stable: the system should expose explicit scientific bindings rather than hide them in free text. The active extractor line therefore includes structured objects for materials, process states, conditions, models, mechanisms, microstructure features and parameter claims, with lightweight evidence fields that can later be upgraded into reusable evidence objects during post-processing.

## Deterministic post-processing and scientific binding

Raw extraction output is not sufficiently stable for reuse without deterministic refinement. CPextractor therefore applies a sequence of post-processing modules that normalize parameter identity, normalize units, backfill metadata, resolve condition binding, link model equations and construct final claims.

The scientific purpose of these modules is to convert raw mentions into interpretable parameter records. Material state is kept separate from loading condition. Physical material constitution is kept separate from modeling representation. Mechanism and system scope are made explicit rather than implied. These distinctions are important because two numerically similar values may not be comparable if they operate at different scopes or under different constitutive assumptions.

The output of this stage is a final hierarchy centered on `parameter_claims`, supported by materials, phases, process states, conditions, models and evidence objects. This structure is intentionally aligned with the requirements of human reuse rather than with the simpler requirement of producing a superficially correct extraction.

## Evidence grounding, review agents and confidence fusion

Each claim is grounded back to local evidence, including section spans, line or character ranges and table-specific context such as row and column labels. The project deliberately separates lightweight extraction-time evidence from the later grounding stage so that reusable evidence objects can be produced after the claim structure has stabilized.

Grounded claims are then evaluated by a multi-agent review stack that includes evidence, normalization, consistency and meta-level judgments. These agents operate alongside deterministic quality checks. The result is not a single opaque model score, but a layered audit trail that records where support is strong, where ambiguity remains and where review escalation may be necessary.

Confidence fusion combines these heterogeneous signals into parameter-level and document-level trust outcomes. Low-trust claims can remain visible in review artifacts while still being excluded from database admission. This distinction is crucial for a scientific curation workflow because it separates the goals of sensitivity, auditability and conservative downstream reuse.

## Annotation and evaluation framework

The evaluation design of CPextractor is built around claim-level annotation rather than document-level impressionistic review. Model outputs can be exported as claim drafts, corrected by human annotators and converted into gold-standard files for downstream benchmarking. The schema supports evaluation of claim detection, grounding accuracy, bundle completeness, value and unit correctness, mapping correctness, binding correctness and provenance correctness.

The project also includes a usability-oriented annotation mode in which a parameter record is judged not only for correctness but for whether it contains the minimum scientific context needed for reuse by another researcher. This is an important conceptual choice because a merely correct value is not always a scientifically useful record.

## Database persistence and retrieval

Accepted outputs are stored in PostgreSQL tables for papers, structured extractions, chunks, references, evaluation artifacts and vectorized parameter content. Structured querying supports exact filtering by material, parameter type, scope or related metadata. Vector-based retrieval supports semantic search across both document text and parameter-centric content.

The chatbot layer combines these routes into a hybrid evidence-constrained retrieval workflow. This allows natural-language questions to be answered from locally retrieved supporting evidence rather than from unsupported synthesis alone. The resulting system is therefore not just a paper parser, but a reusable scientific data service built on top of evidence-grounded literature curation.

# Data and Code Availability

The codebase, schema definitions, evaluation scripts and manuscript source are available in the CPextractor project workspace. Before external submission, this section should be updated with the public repository URL, archive DOI, software license and release date. Where publisher licensing prevents redistribution of raw full text, the public release should provide DOI lists, reconstruction workflows, schema snapshots, benchmark annotations and evidence-pointer metadata sufficient for authorized users to recreate the corpus locally.

# Competing Interests

The authors declare no competing interests.

# Acknowledgements

This section should be updated with project funding, institutional support and any computational or annotation assistance before submission.

# Conclusion

CPextractor establishes a practical route from crystal plasticity literature to reusable scientific infrastructure. By integrating structured parsing, staged extraction, deterministic binding, evidence grounding, multi-agent review, confidence fusion and gated database admission, the project moves beyond value harvesting toward auditable constitutive parameter curation.

The current repository snapshot already demonstrates that this architecture can operate at corpus scale, producing thousands of evidence-linked claims while selectively admitting only a higher-trust subset into the final database-ready layer. With a benchmark-locked validation package added before journal submission, CPextractor is well positioned as a strong manuscript at the intersection of scientific information extraction, materials data infrastructure and evidence-grounded computational modeling.

# Figure Captions

Figure 1 | End-to-end architecture of CPextractor. The figure should summarize the workflow from DOI- or search-driven corpus acquisition to XML-first parsing, evidence selection, schema-constrained extraction, deterministic normalization, evidence grounding, multi-agent review, confidence fusion and gated database ingestion. The visual emphasis should be on the transition from heterogeneous full-text documents to atomic evidence-grounded parameter claims.

Figure 2 | Corpus-scale outputs and trust filtering. The main panel should report the progression from processed papers to raw parameter records, normalized claims, evidence-linked claims, high-confidence claims and final database-ready entries. Supporting panels should summarize document-level verdicts and the proportion of papers blocked by the ingestion gate.

Figure 3 | Scientific data model and retrieval interface. The figure should show the transition from `parameters.registry` to `parameter_claims`, evidence objects and the final hierarchy of materials, phases, process states, conditions and models. A companion panel should illustrate structured database queries, vector retrieval and evidence-constrained chatbot answering.

# Figure Roadmap

Figure contract for the current manuscript draft:

Core conclusion: Crystal plasticity literature can be converted into reusable parameter claims only when extraction, evidence grounding, review and database gating are integrated into a single trust-aware workflow.

Figure archetype: asymmetric mixed-modality figure set across the paper, with one schematic-led systems figure and two quantitative summary figures.

Target journal/output: high-impact computational materials or materials informatics journal; editable vector export for final submission.

Backend: [Choose Python or R before figure generation]

Final size: Fig. 1 full width workflow figure; Figs. 2-3 full width quantitative composites.

Panel map:
a: corpus acquisition and parsing assets
b: evidence selection and extraction
c: normalization, grounding and review stack
d: claim construction and database ingestion

Evidence hierarchy:
hero evidence: staged workflow plus trust-aware claim construction
validation evidence: corpus-scale counts and retention through review/gating
controls/robustness: document-level verdict distribution and blocked-paper fraction

Statistics needed: corpus counts; admitted versus blocked fractions; any final benchmark metrics once locked

Source data needed: per-paper processing summary; claim counts; gate outcomes; verdict distributions

Image-integrity notes: all quantitative panels should be generated from exported summary tables or JSON metrics; no decorative values should be introduced manually

Reviewer risk: avoid presenting the current pilot benchmark as final claim-level accuracy; keep figures aligned to defended repository outputs unless benchmark lock is completed
