## CPextractor Submission Checklist

### Fill before submission

- Replace author names, affiliations, and corresponding author email.
- Add funding details in `Acknowledgements`.
- Add public repository URL, archive DOI, and software license in `Data and Code Availability`.
- Confirm target journal and adapt title/abstract length if the venue has strict word limits.

### Scientific validation decision

- If the benchmark lock is completed, insert final claim-level precision, recall, F1, grounding accuracy, bundle completeness, and any baseline comparisons into the Results and Abstract.
- If the benchmark lock is not completed, submit only to a venue that tolerates a systems/data-infrastructure emphasis, or reduce submission claims so they do not imply finalized benchmark validation.

### Figures

- Generate final figures from the manuscript figure roadmap.
- Keep figure claims aligned to repository-supported outputs unless new benchmark numbers are formally locked.
- Export editable vector files plus high-resolution TIFF versions for submission.

### Final QA

- Re-render the DOCX or convert to the journal-preferred format once `soffice` or an equivalent renderer is available.
- Run one final consistency pass for terminology: `parameter claim`, `evidence grounding`, `confidence fusion`, `database-ready layer`.
- Check that every quantitative statement in the Abstract and Results matches the exported corpus summary.
