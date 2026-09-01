# Accessibility of `figure_reward_vs_f1.png` — measured, not asserted

Every ratio in this file was computed by `make_figure.py` at render time and written to
`accessibility_audit.json`. Nothing here is estimated or eyeballed.

---

## 1. The requirements, and where they come from

### Primary source: Cornell IT, "Web Accessibility" (<https://it.cornell.edu/accessibility>)

Read in full. It adopts a standard rather than listing criteria itself:

> "all web content should conform to the latest version of W3C's Web Content Accessibility
> Guidelines (WCAG) using the AA standard. See University Policy 5.12, Web Accessibility
> Standards, for specifics."

It links WCAG 2.2 (<https://www.w3.org/TR/WCAG22/>) as "the latest version", and requires
that "meeting the WCAG standard requires a combination of automated and manual testing",
naming the manual basics as "keyboard testing, color contrast checks, and reviews of media
to ensure that all images, video, and audio content include the necessary supports."

### Cornell University Policy 5.12, Web Accessibility Standards

> "all new, newly added or redesigned university web content, web pages, web functionality,
> websites, and web applications must be made accessible to people with disabilities to the
> standard prescribed by the most recently published Web Content Accessibility Guidelines
> (WCAG)… When fundamental alteration or undue burden applies, equally effective alternative
> means of access must be provided."

### Cornell IT, "Alternate Text for Images" (<https://it.cornell.edu/accessibility/alternate-text-images>)

> "In order to be WCAG AA compliant, all images must have accompanying text that conveys the
> same information. Images that add no meaningful information should be marked as decorative."

It gives the WebAIM rules it endorses for writing that text: "Be accurate and equivalent /
Be succinct / Avoid redundancy / Avoid 'image of…' 'graphic of….' 'photo of…' 'link to…'",
and step-by-step instructions for applying alt text in **Word**, **InDesign** and **Adobe
Acrobat** — i.e. Cornell expects this to be applied in the document, not in the image file.

### Cornell IT, "WCAG AA Checklist" (<https://it.cornell.edu/accessibility/wcag-aa-checklist>)

The page links Cornell's own checklist spreadsheet, "developed and maintained by the Custom
Web Development team… their interpretation of the WCAG guidelines and success criteria",
organised by P.O.U.R. The rows that bind this figure, quoted verbatim from that
spreadsheet:

| Criterion | Cornell's wording |
| --- | --- |
| 1.1.1 Non-text Content (A) | "Complex images (graphs, maps, charts) must have a text description of all relevant information." |
| 1.1.1 Non-text Content (A) | "Informative and grouped images must contain alternative text describing the purpose or meaning of the image(s)." |
| 1.4.1 Use of Color (A) | "Color may not exclusively distinguish between plain text and interactive text or distinguish one type of content from another **without a 3:1 color contrast difference**." |
| 1.4.1 Use of Color (A) | "Color may not exclusively identify content or distinguish differences in any content." |
| 1.4.3 Contrast (Minimum) (AA) | "Non Large-scale text must have a color contrast ratio of 4.5:1." / "Large-scale (24px or 19px bold) text must have a color contrast ratio of 3:1." |
| 1.4.11 Non-text Contrast (AA) | "Graphical objects which describe important content must meet a 3:1 color contrast ratio; except flags, real life imagery, branding, reference screencaps, and heatmaps." |
| 1.4.4 Resize Text (AA) | "Text can be resized up to 200% without page content disappearing or losing functionality." |
| 1.4.5 Images of Text (AA) | "Images of text are not used when the same presentation can be made with native HTML/CSS." |

For reference, the W3C wording of the same criteria (from the WCAG 2.2 Recommendation
that Cornell adopts): 1.4.3 requires "a contrast ratio of at least 4.5:1" for normal text
and 3:1 for large-scale text, where "large scale" is "at least 18 point or 14 point bold";
1.4.11 requires "at least 3:1 against adjacent color(s)" for "[p]arts of graphics required
to understand the content"; 1.4.1 requires that "[c]olor is not used as the only visual
means of conveying information… or distinguishing a visual element."

---

## 2. Point-by-point conformance, with the measured numbers

Measurement method: WCAG 2.x relative luminance and the (L1 + 0.05)/(L2 + 0.05) contrast
formula, implemented in `make_figure.py` (`relative_luminance`, `contrast_ratio`) and run
on the exact hex colours the figure draws with. Background is `#FFFFFF`.

### SC 1.4.3 Contrast (Minimum) — text — **PASS**

All text in the figure is below the 18 pt / 24 px "large scale" threshold, so the stricter
4.5:1 applies to all of it.

| Text role | Colour | Measured contrast vs background | Required | Result |
| --- | --- | --- | --- | --- |
| All primary text: titles, axis labels, tick labels, legend text, bar value labels | `#1B1F24` | **16.56:1** | 4.5:1 | PASS |
| Secondary in-axes notes | `#4C5661` | **7.47:1** | 4.5:1 | PASS |

### SC 1.4.11 Non-text Contrast — data marks — **PASS**

| Series | Colour | Measured contrast vs background | Required | Result |
| --- | --- | --- | --- | --- |
| onset + pitch + duration (panels b, c) and GRPO (panel a) | `#1B1F24` | **16.56:1** | 3:1 | PASS |
| onset + pitch (panels b, c) | `#0072B2` | **5.19:1** | 3:1 | PASS |
| onset ±1 bin + pitch (panels b, c) | `#D55E00` | **3.87:1** | 3:1 | PASS |
| PPO, token-level (panel a) | `#AA4499` | **5.26:1** | 3:1 | PASS |

Lowest measured value across all data marks: **3.87:1**.

### SC 1.4.1 Use of Color — **PASS, but only because of the redundant encodings**

Cornell's checklist permits colour to distinguish content *only* where the colours differ
by at least 3:1. Measured pairwise contrast between series that share a panel:

| Panel | Pair | Measured | ≥ 3:1 by colour alone? |
| --- | --- | --- | --- |
| (a) | GRPO vs PPO | 3.15:1 | yes |
| (b), (c) | onset+pitch+duration vs onset+pitch | 3.19:1 | yes |
| (b), (c) | onset+pitch+duration vs onset ±1 bin+pitch | 4.28:1 | yes |
| (b), (c) | **onset+pitch vs onset ±1 bin+pitch** | **1.34:1** | **no** |

So one pair — the blue and the orange — would **fail** if colour were the only cue.
It is not. Every series carries at least two non-colour cues:

- **Panel (a):** GRPO = solid line + open circles; PPO = dashed line + open squares.
- **Panel (b):** solid fill / diagonal hatch / dotted hatch, a fixed left-to-right bar
  order that repeats in every group, a white gap between bars so no two fills ever touch,
  a `#1B1F24` outline on every bar (16.56:1 against the background), and **the numeric
  value printed above every single bar** — the strongest mitigation, since the reader
  never has to distinguish two bars to read the data.
- **Panel (c):** solid + circle / dashed + square / dash-dot + triangle.

### Colour-vision-deficiency simulation — **measured**

Simulated with `colorspacious` 1.1.2 (`sRGB1+CVD`, severity 100), then re-measured with
the same WCAG formula.

| CVD type | Lowest data-mark contrast vs background | onset+pitch vs onset ±1 bin+pitch |
| --- | --- | --- |
| deuteranomaly (severity 100) | 3.38:1 | 1.65:1 |
| protanomaly (severity 100) | 4.59:1 | 1.07:1 |
| tritanomaly (severity 100) | 3.90:1 | 1.18:1 |

Every mark still clears 3:1 against the background under all three simulations. The blue /
orange pair remains indistinguishable by colour under all three — which is exactly the
case the hatches, line styles, markers and printed values are there to cover. Full
simulated hex values are in `accessibility_audit.json`.

### Grayscale — **measured and inspected**

`figure_reward_vs_f1_gray.png` is produced by converting the finished 300 dpi raster to
WCAG relative luminance and re-encoding it — not by re-plotting in grey, so it is a true
test of the delivered artwork. It was then opened and inspected.

| Series | Colour | Relative luminance | Equivalent grey |
| --- | --- | --- | --- |
| onset + pitch + duration / GRPO | `#1B1F24` | 0.0134 | `#1F1F1F` |
| onset + pitch | `#0072B2` | 0.1525 | `#6D6D6D` |
| onset ±1 bin + pitch | `#D55E00` | 0.2215 | `#828282` |
| PPO (token-level) | `#AA4499` | 0.1498 | `#6C6C6C` |

Result of the inspection: all three panels remain fully readable. In panel (b) the
`#6D6D6D` and `#828282` bars are close in grey value (21 levels apart), and they are told
apart by their hatch and by their printed values, not by tone. In panels (a) and (c) the
line styles and markers separate every series unambiguously.

### SC 1.1.1 Non-text Content — **PASS via the text alternative below plus `SOURCES.md`**

Cornell's checklist: "Complex images (graphs, maps, charts) must have a text description
of all relevant information." Alt text alone cannot carry 500-plus plotted points, so this
is satisfied in two layers: the short alt text in §3, and `SOURCES.md`, which tabulates
every landmark value with its source file and includes the protocol, the criterion
definitions and the limitations. `plotted_values.json` carries the complete series.

### Text size

WCAG sets no absolute minimum font size for print, so there is no threshold to pass or
fail here; stating the sizes is the honest alternative to claiming compliance. The figure
is 6.8 × 6.1 in at 300 dpi, sized to sit at roughly 1:1 in a letter-page text column.

| Element | Size (pt) |
| --- | --- |
| Axis labels | 9.0 |
| Panel titles (bold) | 8.8 |
| Tick labels | 8.5 |
| Panel (b) category labels | 7.8 |
| Bar value labels | 7.4 |
| Legends | 7.2 |
| In-axes explanatory notes (smallest text in the figure) | 7.0 |

### Other criteria

- **1.4.4 Resize Text / 1.4.10 Reflow.** A raster chart cannot reflow. The mitigation is
  the vector `figure_reward_vs_f1.pdf`, which scales losslessly to any magnification, plus
  the text data in `SOURCES.md`. Cornell's own checklist excludes "content where a
  two-dimensional layout is necessary (video, data tables, maps, diagrams)" from 1.4.10.
- **1.4.5 Images of Text.** The only text in the image is chart labelling, which is
  essential to the graphic and cannot be replaced by native markup. Not a violation.
- **Keyboard access, media captions, forms.** Not applicable to a static figure.

---

## 3. Alt text for the figure

**Short alternative text** (apply in Word / InDesign / Acrobat per Cornell's instructions;
follows WebAIM's "accurate and equivalent / succinct / avoid 'image of'"):

> Three-panel chart. A reinforcement-learning reward rises well above its supervised
> starting value for two post-training arms, but the note-level F1 of those same
> checkpoints does not improve; a third arm rewarded on note-level F1 directly does raise
> it. Values are tabulated in the accompanying data appendix.

**Long description** (Cornell: complex images "must have a text description of all relevant
information"):

> Panel (a) plots validation reward, the sum of three per-token accuracies on a 0–3 scale,
> against post-training optimiser step for two arms, GRPO and token-level PPO. Both start
> at the supervised value 1.327 and rise within a few hundred steps to roughly 1.50–1.55,
> peaking at 1.584 for GRPO at step 250 and 1.620 for PPO at step 775. GRPO's curve stops
> at step 2400 and PPO's at step 4975 because both jobs were killed by the scheduler.
> Panel (b) is a grouped bar chart of macro note-level F1, in percent, for three
> checkpoints under three note-matching criteria, ordered strictest first. The supervised
> initialisation scores 11.1, 18.9 and 22.7; GRPO scores 10.4, 17.2 and 21.0; PPO scores
> 11.0, 18.6 and 21.8. Every GRPO and PPO bar is level with or below the corresponding
> supervised bar. Panel (c) plots the same three criteria against optimiser step for a
> third arm, PPO-F1, whose reward is the loosest criterion. That criterion rises from 39.9
> percent at step 0 to a best of 45.6 percent at step 4900; the middle criterion rises
> from 25.5 to about 29 percent and the strictest from 18.9 to about 21 percent.
> Full values, sources and limitations are in `SOURCES.md`.

---

## 4. Draft for Section IIC — "Address how the work will be accessible per ADA standards"

*(180 words — the form asks for 100–200.)*

> This work will meet WCAG 2.2 Level AA, the standard Cornell adopts in University Policy
> 5.12 and on it.cornell.edu/accessibility. For the figure in this essay I measured
> conformance rather than assuming it. All text sits at 16.56:1 or 7.47:1 contrast against
> its background, above the 4.5:1 minimum in Success Criterion 1.4.3, and every plotted
> series clears the 3:1 minimum in 1.4.11, the lowest measuring 3.87:1. Cornell's WCAG AA
> checklist allows colour alone to distinguish content only where the colours differ by at
> least 3:1; two of my series differ by only 1.34:1, so no series depends on colour. Each
> carries a distinct line style, marker shape or hatch pattern, and every bar is labelled
> with its numeric value. I rendered a greyscale copy of the finished artwork and confirmed
> that all series remain separable. Following Cornell's requirement that complex images
> such as graphs and charts carry "a text description of all relevant information," the
> figure is delivered with alt text and a data appendix listing every plotted value.
> Documents will be exported as tagged PDFs with that alt text applied.

---

## 5. What does **not** conform, stated plainly

1. **The primary Cornell source governs *web* content, not print documents.** Policy 5.12
   and it.cornell.edu/accessibility are scoped to "web content, web pages, web
   functionality, websites, and web applications." A figure in a printed or PDF essay is
   outside that scope. WCAG 2.2 AA is applied here as the substantive standard because it
   is the standard Cornell names, and the alt-text page does explicitly cover Word,
   InDesign and Acrobat. **Cornell's PDF Remediation Checklist
   (<https://it.cornell.edu/landing-page-kba/5816/5135>) was not read** — it sits behind a
   knowledge-base landing page — so no claim is made about document-level PDF tagging
   conformance beyond what is stated above.
2. **The PNG and PDF files carry no embedded alt text.** Matplotlib does not write one, and
   Cornell's instructions are to apply alt text in the authoring application. Until the
   figure is placed in Word/InDesign/Acrobat and the text from §3 is entered there, SC
   1.1.1 is satisfied only by the companion text files, not by the image itself. **This is
   an action the user must take.**
3. **Two series are not distinguishable by colour.** `#0072B2` and `#D55E00` measure
   1.34:1 (1.07–1.65:1 under simulated CVD). This is deliberate — luminance was traded for
   grayscale spread — and is covered by hatch, line style, marker and printed values, but
   it means the figure would fail 1.4.1 if those redundant cues were stripped.
4. **Gridlines measure 1.49:1 against the background** (`#CFD4D9`), below 3:1. They are
   argued to be non-essential under 1.4.11's "parts of graphics *required* to understand
   the content" wording, because every axis is ticked and labelled and every bar carries
   its printed value. That is a judgement call, not a measured pass. Darkening them is a
   one-line change if a reviewer disagrees.
5. **No assistive-technology testing was performed.** No screen reader, no magnifier, no
   user testing. Cornell states plainly that "meeting the WCAG standard requires a
   combination of automated and manual testing"; only the automated half was done here,
   plus a manual grayscale inspection. Siteimprove, Cornell's scanner, applies to public
   websites and was not applicable.
6. **The CVD simulation is a model, not a person.** `colorspacious`'s severity-100
   simulation is an approximation of dichromatic vision, and clearing a numeric threshold
   under it is not the same as being legible to a colour-blind reader.
7. **No minimum print font size was verified against a standard**, because WCAG does not
   set one. The smallest text in the figure is 7.0 pt at the intended print size.

---

### Sources consulted for this file

- Cornell IT, "Web Accessibility." <https://it.cornell.edu/accessibility> (accessed 24 Aug 2026)
- Cornell University Policy 5.12, "Web Accessibility Standards." <https://policy.cornell.edu/policy-library/web-accessibility-standards>
- Cornell IT, "Alternate Text for Images." <https://it.cornell.edu/accessibility/alternate-text-images>
- Cornell IT, "WCAG AA Checklist," and the linked checklist spreadsheet maintained by CIT Custom Web Development. <https://it.cornell.edu/accessibility/wcag-aa-checklist>
- W3C, *Web Content Accessibility Guidelines (WCAG) 2.2*. <https://www.w3.org/TR/WCAG22/>
