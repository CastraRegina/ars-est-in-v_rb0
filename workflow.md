
# Workflow

## Text

### Prepare Input Text

- Create text file or use AI to generate text

### Check Input Text

- Check the text to ensure that its content is correct. --> TODO
- Check text for copyright infringements and other legal issues --> TODO
  - TODO: which tools are available to check for copyright infringements?
  - Document text license

### Hyphenation of Input Text

- Check/correct orthography --> TODO
- Hyphenate text --> TODO
- Check hyphenation, modify/correct by hand --> TODO

## Font

### Check Font

- Check font for copyright infringements and other legal issues --> TODO
- Document font license

## Image

### Prepare Input Image

- Create image file or download public domain image

#### Check Input Image

- Check image for copyright infringements and other legal issues --> TODO
  - In US, images are copyrighted for life + 70 years
  - In EU, images are copyrighted for life + 70 years

#### Vectorize Image to SVG in black and white

- Convert image to PBM and vectorize PBM to SVG

```bash
convert input.png -threshold 50% temp.pbm
potrace temp.pbm -s -o output.svg
```

- Check SVG in `inkscape`
- See also:  
  [https://vectorise.me](https://vectorise.me)  
  [https://picshifter.com](https://picshifter.com/tools/svg-converter)

## Artwork

- Define specific number of the artwork (format: YYxxx)
- Define name of the artwork
- Check name for copyright infringements and other legal issues --> TODO
- Check if it belongs to a series / (collection)
  - [Diptych](https://en.wikipedia.org/wiki/Diptych)
  - [Triptych](https://en.wikipedia.org/wiki/Triptych)
  - [Polyptych](https://en.wikipedia.org/wiki/Polyptych)
- Check if it should be part of the [Artist's portfolio](https://en.wikipedia.org/wiki/Artist%27s_portfolio)

## Deployment

The "Catalogue raisonné in preparation" (Werkverzeichnis in Vorbereitung) webpage is based on AHA-stack

### Steps

- Save settings (and maybe the artwork itself) to private repository
- Update Webpage (GitHub Pages) / catalogue raisonné  
  [Artist website best practices](https://www.meinekunstseite.com/blog/was-gehoert-zu-einer-beeindruckenden-kuenstler-website)
- Update online shops
- Update social media / newsletter / blog

## Additional Information

### Register to services

- Check availability
  - domain: `whois google.com`
  - [https://namechk.com/](https://namechk.com/)
  - [https://namecheckly.com/](https://namecheckly.com/)
  - [https://namechecker.org](https://namechecker.org)
  - [https://dnschecker.org/social-media-name-checker.php](https://dnschecker.org/social-media-name-checker.php)
  - [https://qezir.com/check](https://qezir.com/check)
  - [https://tools.namerobot.de/socialcheck](https://tools.namerobot.de/socialcheck)

### Webpage

#### Webpage Requirements

- Use AHA-stack (Astro, htmx, Alpine.js, Tailwind CSS)
- Multi language (i18n): /en/, /de/
- Multi target devices with multi resolution (XS, S, M, L, XL)
- No SmartTV (10-Foot UI) support initially
- Simple and robust
- State-of-the-Art implementation
- Fast preview and with reloading details if requested
- No reloading of parts from external sources, e.g. Google fonts or similar
- Optimization for SEO (JSON-LD, OpenGraph, Twitter Cards)
- Webpage should be also full functional without JavaScript (progressive enhancement)
- Initial start page with sub pages and files
  All pages in footer with links to "About", "Contact", "Legal notice" and "Data privacy statement"
  - About (Über)
  - Contact (Kontakt)
  - Legal notice (Impressum)
  - Data privacy statement (Datenschutzerklärung / DSGVO)
  - 404-Page (multi language)
  - robots.txt and sitemap.xml
- CSS details: Use Tailwind CSS

#### Webpage Content

- Digital Fine Art Print --- Giclée / Giclee Print
- "A limited edition Giclée print of my digital painting."
- Showcase...
  - final prints.
  - the digital creation process, which adds value to limited edition Giclée prints.
- Notes to [Edition](https://en.wikipedia.org/wiki/Edition_%28printmaking%29)

#### Future Functionalities of Webpage

- Support searching the list of artworks by date of creation, category, technique, dimensions, price range
- Support subpages for Series (Serie), Portfolio (Mappe), exhibitions, and artist statement
- Implement "Image Protection Measures" / "Digital Rights Management (DRM)" for featured high-res images
  - Digital watermarking (pre-applied to images)
  - Canvas-based rendering (prevents direct download)
  - Terms of service prohibiting unauthorized use
  - Techniques not applicable to static GitHub Pages:
    - Backend watermarking on-demand (requires backend server)
    - Hotlinking protection via referrer checking (limited control on GitHub Pages)
  - Ineffective/outdated techniques (considered but not recommended):
    - Right-click/disable save functionality (easily bypassed)
    - Overlay protection (disabled in browser dev tools)
    - EXIF copyright metadata (stripped on upload, irrelevant for web)
    - Image slicing (outdated, easily reconstructed by browsers)
  - Note: General images displayed as low-resolution; only selected featured images shown in high-res

#### Recommendations by AI

**Essential Pages (Minimum):**

1. **Home/Portfolio**
   - Title section with 1-3 featured artworks
   - Brief artist statement (2-3 sentences)
   - Clear navigation to gallery and shop
   - Call-to-action: "View Collection" or "Shop Prints"

2. **Gallery/Collection**
   - Grid layout of all available works
   - Filter by series/year/availability
   - Each artwork shows:
     - High-quality image
     - Title, year, dimensions
     - Edition info (e.g., "Limited edition of 25")
     - Price or "Inquire" button
   - Click for detail view with larger image

3. **About**
   - Professional photo
   - Artist bio (150-200 words)
   - Focus on digital art technique and Giclée process
   - CV/exhibitions (optional, can be simplified)

4. **Contact**
   - Contact form
   - Email address
   - Social media links
   - For inquiries about commissions or purchases

5. **Shop/Store**
   - Direct purchase integration
   - Clear pricing
   - Print specifications (paper quality, dimensions, edition)
   - Shipping information
   - Return policy

**Minimal Content Strategy Examples:**

**Homepage:**

```text
[Title Image: Featured Artwork]
John Doe - Digital Artist
Creating limited edition Giclée prints from original digital paintings

[View Gallery] [Shop Now]

Recent Works:
[3-4 thumbnail images]
```

**Gallery Page:**

```text
Digital Art Collection 2024
[Filter: All | Available | Sold]

[Grid of 12-20 images with hover effect]
Each: Title | Year | Edition Info
```

**About Page:**

```text
[Photo]
About the Artist
[2-paragraph bio focusing on digital art process]

My Process
[Short section explaining digital painting → Giclée print]

Selected Exhibitions
[3-5 most important shows]
```

**Key Content Elements:**

- High-quality images with zoom/detail view
- Clear pricing and edition information
- Process showcase adds value to limited editions
- Multi-language support (/en/, /de/)
- SEO: JSON-LD, OpenGraph, Twitter Cards

#### Install AHA-stack (Astro, htmx, Alpine.js, Tailwind CSS)

1. Prepare the system

    ``` bash
    sudo apt update && sudo apt upgrade -y
    ```

2. Install Node.js (via nvm – recommended)

    ``` bash
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    source ~/.bashrc
    nvm install --lts
    nvm use --lts
    node -v   # verify
    npm -v
    ```

3. Create an Astro project

    ``` bash
    npm create astro@latest catalogue-aha
    ```

    When prompted, choose a template (e.g., empty or minimal).  
    Select Yes for TypeScript and Git if desired.  
    Then:

    ``` bash
    cd catalogue-aha
    npm install
    ```

4. Add Tailwind CSS

    ``` bash
    npx astro add tailwind
    ```

    This automatically installs Tailwind, creates a `tailwind.config.js`, and sets up the CSS.

5. Add htmx and Alpine.js locally

    ``` bash
    mkdir -p public/js
    curl -o public/js/htmx.min.js https://unpkg.com/htmx.org@2.0.9/dist/htmx.min.js
    curl -o public/js/alpine.min.js https://unpkg.com/alpinejs@3.15.11/dist/cdn.min.js
    ```

    Create a layout file (e.g., `src/layouts/BaseLayout.astro`):

    ``` astro
    ---
    // frontmatter (optional)
    ---
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <script src="/js/htmx.min.js" defer></script>
        <script src="/js/alpine.min.js" defer></script>
      </head>
      <body>
        <slot />
      </body>
    </html>
    ```

6. Create additional directories (optional)

    ``` bash
    mkdir -p src/components src/layouts src/data public/images
    ```

7. Add example data (optional)

    ``` bash
    nano src/data/artworks.json   # or use your editor
    ```

    Paste sample content:

    ``` bash
    [
      {
        "id": "example-work",
        "title": "Example Work",
        "year": 2024,
        "description": "Example description",
        "image_preview": "/images/example.jpg",
        "image_highres": "/images/example-large.jpg",
        "edition": "Giclée print"
      }
    ]
    ```  

8. Run the development server in folder `catalogue-aha`

    ``` bash
    npm run dev
    ```

    Visit `http://localhost:4321` to see your project.

9. Build for production

    ``` bash
    npm run build
    ```

    The static output will be in the `dist/` folder.

10. Test the AHA Stack

    To see a working page, you need to create an Astro file inside src/pages/. For example:

    - Create a homepage

      ```bash
      nano src/pages/index.astro
      ```

    - Add simple content

      ```astro
      ---
      // Component script (optional)
      ---

      <html lang="en">
        <head>
          <title>My AHA App</title>
        </head>
        <body>
          <h1>Hello, AHA stack!</h1>
          <div x-data="{ count: 0 }">
            <button @click="count++" x-text="count"></button>
          </div>
          <div hx-get="/api/hello" hx-trigger="load">Loading...</div>
        </body>
      </html>
      ```

    - Save and refresh <http://localhost:4321>

    You should now see a page with a counter (Alpine.js) and a loading placeholder (htmx – you'd need a backend endpoint for the actual request, but it demonstrates the libraries are working).

### Privacy

- [www.privacytools.io](https://www.privacytools.io)

#### Domain

- [secure.orangewebsite.com](https://secure.orangewebsite.com/index.php)
- ( [njal.la](https://njal.la/domains/) )

#### Email

- [Tutao](https://tuta.com/de) use with FIDO U2F
- [Proton](https://proton.me/)

### Artwork Classification Terminology

#### Individual Works

**One-Off / Standalone Work (Einzelwerk)**  
Individual artwork not part of a series, cycle, or body of work.

#### Panel Formats

- Diptych (Diptychon): Two-panel work
- Triptych (Triptychon): Three-panel work
- Tetraptych/Quadriptych (Tetrapttychon/Quadriptychon): Four-panel work
- Pentaptych (Pentaptychon): Five-panel work
- Polyptych (Polyptychon): Multi-panel work (more than three sections)

#### Groupings by Artists

**Suite (Suite)**  
Cohesive group of works created together, common in printmaking.  
Similar to series but typically smaller and more tightly unified.

**Series (Serie)**  
Related works created in succession, exploring variations of a subject, technique, or visual idea.  
A series may constitute a body of work or be part of a larger body of work.

**Cycle (Zyklus)**  
Works designed to be viewed together as a unified whole,
often following a narrative progression or specific chronological order.

**Body of Work / Work Group (Werkgruppe)**  
A cohesive set of artworks unified by common theme, technique, style, or conceptual approach. Typically comprises 10-20+ substantial pieces demonstrating depth and consistency. Distinct from an artist's complete oeuvre.

**Portfolio (Mappe)**  
Curated selection of works intended to remain together for complete understanding, focused on narrative rather than artist career. Unlike series, portfolios represent thematic completeness.

#### Multiples

**Edition (Auflage)**  
Multiple copies produced from a single master (prints, sculptures, photographs). Each piece is individually created and considered unique. Limited editions have predetermined quantity; open editions are unlimited. Distinct from reproductions.

#### Collections and Holdings

**Collection (Kollektion / Sammlung)**  
Grouped works held by a collector or institution, or a thematic assembly by an artist.

#### Complete Output

**Oeuvre (Gesamtwerk)**  
The complete body of work produced by an artist throughout their lifetime.

#### Documentation

**Catalogue Raisonné (Werkverzeichnis)**  
Comprehensive annotated listing of all works by an artist, organized by medium, period, or other parameters. Used for authentication and scholarship.
