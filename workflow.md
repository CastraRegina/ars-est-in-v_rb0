
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
- Check if it belongs to a series / collection
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
  - Youtube
  - TikTok
  - Pinterest
  - LinkedIn
  - Xing

  ...
  - Twitter / X
  - Shopify
  - Etsy

## Additional Information

### Web Presentation

Where can I publish my art for free?

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

- Use AHA-stack:
  - A - Astro - server-side HTML, static pages
  - H - htmx - HTML-over-wire dynamics without SPA
  - A - Alpine.js - small client-side interactivity
- Accessibility: comply with WCAG 2.2 AA and EN 301 549 or newer
- Multi language (i18n)
  - Use example.com/en/ , example.com/de/ ...
  - Reload whole page when switching between languages
- Multi target devices with multi resolution:  
  support viewport-classes (XS, S, M, L, XL): use modern CSS with Container Queries,  
  but for SmartTVs (10-Foot UI) special implementation is needed.
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
  (and also links to social media services on the left side of the page)
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

#### Recommendations by AI

**Essential Pages (Minimum):**

1. **Home/Portfolio**
   - Hero section with 1-3 featured artworks
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
[Hero Image: Featured Artwork]
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

### Privacy

- [www.privacytools.io](https://www.privacytools.io)

#### Domain

- [secure.orangewebsite.com](https://secure.orangewebsite.com/index.php)
- ( [njal.la](https://njal.la/domains/) )

#### Email

- [Tutao](https://tuta.com/de) use with FIDO U2F
