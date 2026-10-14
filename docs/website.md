# Project website

Project page: https://little-fridge.github.io/ViewCraft-page/

Website-only repository: https://github.com/Little-Fridge/ViewCraft-page

The ViewCraft code repository is public. The project page is hosted from a separate public repository containing only the website and publication assets.

## Source and publishing

The source of the project page is `website/` in this repository. To update it:

1. Edit and locally preview the files in `website/`.
2. Copy `index.html`, `style.css`, `script.js`, `.nojekyll` and the referenced `assets/` into the website-only repository.
3. Commit and push the website repository's `main` branch. GitHub Pages serves its root directory.

Do not copy the code checkout, its Git history, internal galleries, evaluation records, model configuration or credentials into the public website repository. The website repository has its own independent history.

All asset links are relative to support the `/ViewCraft-page/` project path. Update the canonical/Open Graph URLs in `website/index.html` and the README links if the Pages address changes.

The site links to the paper at https://arxiv.org/abs/2610.15915 and publishes the paper figures. Its code and skill links are publicly accessible.
