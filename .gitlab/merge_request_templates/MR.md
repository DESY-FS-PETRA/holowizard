## MR Creator Checklist
### Summary

Please provide a brief description of the changes and the motivation behind them.
- [ ] I looked through the Review checklist and expect my MR to fulfill all requirements

---

### Related Issues

- Closes #ISSUE_ID
- Related to #ISSUE_ID (if applicable)

---

### Follow-up Issues created

- (if applicable)





## Review checklist

This checklist is meant to assist creators of MRs (to let them know what reviewers will typically look for) and reviewers (to guide them in a structured review process). Items do not need to be checked explicitly for a MR to be eligible for merging.

#### Purpose and scope
- [ ] The MR has a single goal that is clear from the MR title and/or description.
- [ ] All code changes represent a single set of modifications that logically belong together.
- [ ] No more than 500 lines of code are changed or there is no obvious way to split the MR into multiple MRs.

#### Code quality
- [ ] The code can be understood easily.
- [ ] Newly introduced names for variables etc. are self-descriptive and consistent with existing naming conventions.
- [ ] There are no redundancies that can be removed by simple modularization/refactoring.
- [ ] There are no leftover debug statements or commented code sections.
- [ ] The code adheres to our conventions (link to be created)

#### Documentation
- [ ] New functions and types are documented with a docstring or top-level comment.
- [ ] Inline comments are used to document longer or unusual code sections.
- [ ] Comments describe intent ("why?") and not just functionality ("what?").
- [ ] If the MR introduces a significant change or new feature, it is documented in `NEWS.md` with its MR number.

#### Testing
- [ ] The MR passes all tests.
- [ ] New or modified lines of code are covered by tests.
- [ ] New or modified tests run in less then 10 seconds.

#### Performance
- [ ] If the MR intent is to improve performance, before/after are posted in the MR.

#### Verification
- [ ] The correctness of the code was verified using appropriate tests.

*Review checklist created with :heart: by the Trixi.jl community, modified by @eberlese*