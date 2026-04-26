
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
  - Instagram
  - Pinterest
  - TikTok
  - Youtube
  - LinkedIn
  - (Xing)

  ...
  - Cara
  - Facebook ([real-name policy](https://en.wikipedia.org/wiki/Facebook_real-name_policy_controversy))
  - Behance
  - FinerWorks
  - ArtStation
  - Society6 - Print-on-demand
  - Saatchi Art - Fine art
  - KubaParis
  
  ...
  - Twitter / X - AI-learning
  - Squarespace
  - Shopify
  - Etsy
  - Patreon
  - Reddit

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

### List of services

Create an account for each:

- Googlemail (pwd+2factor)
- GitHub (pwd+2factor+2ndSMS)
- Instagram (pwd+2factor)
- Pinterest (login using google account)
- TikTok (login using google account)
- Youtube (login using google account)
- LinkedIn (pwd+2factor)

- Xing (pwd+2factor)
- X / Twitter (login using google account)

- printful
- redbubble (pwd w/o 2factor authentication)
- printify
- inprint
- Gelato

- Etsy (login using google account)
- reddit (login using google account)
- patreon (login using google account)

- Squarespace
- Shopify
- Cara
- Behance
- FinerWorks
- Society6 - Print-on-demand
- Saatchi Art - Fine art
- ArtStation
- KubaParis

### Print on Demand Sites for Artists

- Printful - Best for branded stores
- Redbubble - Best for starting
- Printify - Low price focused
- INPRNT - Best for fine art and collectors
- Gelato - Best for global delivery

### Web Presentation

- Where can I publish my art for free?
  - Facebook Groups
  - Instagram
  - Wetcanvas Forums
  - Reddit
  - Your Website & Blog
  - Steemit
  - Pinterest
  - Google My Business
  - Flickr
  - Jose Art Gallery

### Webpage

#### Webpage Requirements

- Use AHA-stack (Astro, htmx, Alpine.js, Tailwind CSS):
  - A - Astro - server-side HTML, static pages
  - H - htmx - HTML-over-wire dynamics without SPA
  - A - Alpine.js - small client-side interactivity
- Accessibility: comply with WCAG 2.2 AA and EN 301 549 or newer
- Multi language (i18n)
  - Use example.com/en/ , example.com/de/ ...
  - Reload whole page when switching between languages
- Multi target devices with multi resolution:  
  support viewport-classes (XS, S, M, L, XL): use modern CSS with Container Queries,  
  but for SmartTVs (10-Foot UI) special implementation is needed,
  so do not consider/support it at the beginning.
- Simple and robust
- State-of-the-Art implementation
- Fast preview and with reloading details if requested
- No reloading of parts from external sources, e.g. Google fonts or similar
- Optimization for SEO, provide
  - JSON-LD
  - OpenGraph
  - Twitter Cards
- Webpage should be also full functional without JavaScript
- Intial start page with sub pages and files  
  All pages in footer with links to
  "About", "Contact", "Legal notice" and "Data privacy statement" on the left side of the page  
  (and also links to social media services
  (Instagram, Youtube, TikTok, Pinterest, LinkedIn, Xing)
  on the left side of the page)
  - About (Über)
  - Contact (Kontakt)
  - Legal notice (Impressum)
  - Data privacy statement (Datenschutzerklärung / DSGVO)
  - 404-Page (multi language)
  - robots.txt and sitemap.xml
- htmx usage  
  Constraint: Achieve dynamic "Single Page Application" behavior using only static HTML fragments (no backend).
  - Infinite Scroll: Implement seamless loading of additional content/images as the user scrolls down.
  - High-Res Detail View: Enable loading and displaying high-resolution images (or detail fragments)
    into a specific container when a thumbnail / preview image is clicked.
- CSS details
  - Use Tailwind CSS

#### Webpage Content

- Digital Fine Art Print --- Giclée / Giclee Print
- "A limited edition Giclée print of my digital painting."
- Showcase...
  - final prints.
  - the digital creation process, which adds value to limited edition Giclée prints.
- Notes to [Edition](https://en.wikipedia.org/wiki/Edition_%28printmaking%29)

#### Title Image Sizing (Recommended)

| Width        | Device        | Recommended Height | Aspect Ratio   |
|--------------|---------------|--------------------|----------------|
| 640px (XS)   | Smartphone    | 360-480px          | 16:9 to 4:3    |
| 1024px (S/M) | Tablet        | 480-600px          | 16:9 to 16:10  |
| 1920px (L)   | Desktop       | 540-720px          | 16:9 to 16:10  |
| 2560px (XL)  | Large Desktop | 600-800px          | 16:9 to 16:10  |

#### Artwork Preview Image Sizing (Recommended)

##### Preview Images (Grid Thumbnails)

| Device        | Width (px) | Height (px) | Aspect Ratio | Max File Size |
|---------------|------------|-------------|--------------|---------------|
| Smartphone    | 300-400    | 300-400     | 1:1          | 50-100 KB     |
| Tablet        | 400-600    | 400-600     | 1:1          | 100-150 KB    |
| Desktop       | 500-700    | 500-700     | 1:1          | 150-200 KB    |
| Large Desktop | 600-800    | 600-800     | 1:1          | 200-250 KB    |

##### Detail View Images (High-Res)

| Device        | Width (px) | Height (px) | Aspect Ratio | Max File Size |
|---------------|------------|-------------|--------------|---------------|
| Smartphone    | 800-1200   | 800-1200    | 1:1 or 4:3   | 300-500 KB    |
| Tablet        | 1200-1600  | 1200-1600   | 1:1 or 4:3   | 500-800 KB    |
| Desktop       | 1600-2400  | 1600-2400   | 1:1 or 4:3   | 800KB-1.5 MB  |
| Large Desktop | 2400-3200  | 2400-3200   | 1:1 or 4:3   | 1.5-2.5 MB    |

##### Implementation Approach

- Use `srcset` with 4 sizes per image (300w, 600w, 1200w, 2400w)
- Browser selects optimal size based on device
- Preview images: smaller, lighter, faster loading
- Detail images: larger, higher quality for zoom/inspection
- Fixed aspect ratio (CSS `aspect-ratio` or intrinsic dimensions)
- Lazy loading (`loading="lazy"`) for grid images
- Reserve space to avoid layout shift
- Alt text required for accessibility

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

**Body of Work / Work Group (Werkgruppe)**  
A distinct set of artworks linked by a common theme, technique, or conceptual approach, regardless of the timeframe.

**Series (Serie)**  
A sequence of related works created in succession, often exploring variations of the same subject or visual idea.

**Cycle (Zyklus)**  
A cohesive group of works designed to be viewed together as a whole, often following a narrative or a specific chronological order.

- Diptych: A work consisting of two associated artistic panels.
- Triptych: A work divided into three sections or panels.
- Polyptych: A work composed of more than three connected panels.

**Portfolio (Portfolio)**  
A curated selection of an artist's best or most representative works, often used for presentation or professional showcase.

**Collection (Kollektion / Sammlung)**  
An assembled group of works, usually referring to the holdings of a collector or institution, or a thematic grouping by the artist.

**Edition (Edition)**  
A series of identical or similar copies (e.g., prints or sculptures) produced from a single master or mold, usually in a limited number.
