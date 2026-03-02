
# Workflow

## Text

### Prepare Input Text

- Create text file or use AI to generate text

### Check Input Text

- Check text for copyright infringements and other legal issues --> TODO
  - TODO: which tools are available to check for copyright infringements?
  - Document text license

### Hyphenation of Input Text

- Check orthography --> TODO
- Hyphenate text --> TODO
- Check hyphenation --> TODO

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

- Define specific number of the artwork
- Define name of the artwork
- Check name for copyright infringements and other legal issues --> TODO
- Check if it belongs to an edition

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
  - Twitter / X
  - LinkedIn
  - Xing

## Additional Information

### Webpage requirements

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

### Privacy

- [www.privacytools.io](https://www.privacytools.io)

#### Domain

- [secure.orangewebsite.com](https://secure.orangewebsite.com/index.php)
- ( [njal.la](https://njal.la/domains/) )

#### Email

- [Tutao](https://tuta.com/de) use with FIDO U2F
