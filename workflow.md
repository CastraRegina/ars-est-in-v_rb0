
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

The "Catalogue raisonné in preparation" webpage is based on

- A - Astro Server-side HTML, static pages
- H - htmx HTML-over-wire dynamics without SPA
- A - Alpine.js Small client-side interactivity

### Steps

- Save settings (and maybe the artwork itself) to private repository
- Update Webpage (GitHub Pages) / catalogue raisonné  
  [Artist website best practices](https://www.meinekunstseite.com/blog/was-gehoert-zu-einer-beeindruckenden-kuenstler-website)
- Update online shops
- Update social media / newsletter / blog

## Additional Information

### Privacy

- [www.privacytools.io](https://www.privacytools.io)

#### Domain

- [secure.orangewebsite.com](https://secure.orangewebsite.com/index.php)
- ( [njal.la](https://njal.la/domains/) )

#### Email

- [Tutao](https://tuta.com/de) use with FIDO U2F
