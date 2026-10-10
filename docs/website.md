# Project website

Project page: https://little-fridge.github.io/ViewCraft-page/

Website-only repository: https://github.com/Little-Fridge/ViewCraft-page

The ViewCraft code repository remains private. The account's current plan does not support Pages for that private repository, so the project page is hosted from a separate public repository containing only the website and publication assets.

## Source and publishing

The source of the project page is `website/` in this repository. To update it:

1. Edit and locally preview the files in `website/`.
2. Copy `index.html`, `style.css`, `script.js`, `.nojekyll` and the referenced `assets/` into the website-only repository.
3. Commit and push the website repository's `main` branch. GitHub Pages serves its root directory.

Do not copy the code checkout, its Git history, internal galleries, evaluation records, model configuration or credentials into the public website repository. The website repository has its own independent history.

All asset links are relative to support the `/ViewCraft-page/` project path. Update the canonical/Open Graph URLs in `website/index.html` and the README links if the Pages address changes.

The site publishes the preprint PDF and paper figures. Its code links currently require access to the private code repository. Remove the private-access note on the site when the code repository is intentionally made public.
