---
name: release-notes
description: Turns a list of changes, commits or a changelog into user-facing release notes. Use when the user asks for release notes, a changelog entry, or a "what's new" summary for a version.
license: MIT
metadata:
  author: homework3
  version: "1.0"
---

# release-notes

1. Read the template at `/skills/release-notes/assets/template.md` with
   `read_file`. Always use it; don't invent your own layout.
2. Sort every change into one of the template's sections:
   - **New** - features users can now do.
   - **Improved** - existing features that got better or faster.
   - **Fixed** - bugs that no longer happen.
   Leave out internal-only changes (refactors, CI, tests, dependency bumps)
   unless they change what users see.
3. Write each item in plain language from the user's point of view, starting
   with a verb ("Export reports as PDF", not "Added PDFExporter class").
4. Delete any section that ends up empty.
5. If the user didn't give a version number, use `vNEXT`.
6. Save the result to `/release_notes.md` with `write_file`, then show it to
   the user.
