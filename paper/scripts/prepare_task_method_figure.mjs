// Prepare the approved, editable schematic for the 5.5-inch manuscript column.
// Usage: RUNTIME_NODE_MODULES=... node prepare_task_method_figure.mjs in.pptx out.pptx
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [source, output] = process.argv.slice(2);
const modules = process.env.RUNTIME_NODE_MODULES;
if (!source || !output || !modules) throw new Error('Specify source, output, and RUNTIME_NODE_MODULES');
const { FileBlob, PresentationFile } = await import(pathToFileURL(path.join(modules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const p = await PresentationFile.importPptx(await FileBlob.load(source));
if (p.slides.items.length !== 1) throw new Error('Expected the approved one-slide schematic');
const slide = p.slides.items[0];
const snapshot = await p.inspect({ kind: 'slide,textbox,shape,image', maxChars: 50000 });
await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output + '.before.ndjson', snapshot.ndjson);

const edits = [
  ['train-domain-title', 'Alternating training steps', 29],
  ['models-title', 'Separate LoRA arms', 24],
  ['pair-link-label', 'paired', 24],
  ['source-only', 'Only the source identity changes', 24],
  ['target-proxy', 'Independent, screened proxy', 24],
  ['specificity-heading', 'Specificity', 24],
  ['same-noun-task', 'Noncausal ball: retain', 24],
  ['latent-input', 'Same noised latent\n+ timestep', 24],
  ['frozen-teacher', 'Frozen teacher\nSource-free text', 24],
  ['m-label', 'Matched', 24],
  ['s-label', 'SRCD', 24],
  ['erase-loss', 'Flow fitting\nDistillation', 24],
  ['per-arm-update', 'Separate per arm', 24],
  ['preserve-title', 'Preservation', 26],
  ['preserve-data', 'Generic video', 24],
  ['preserve-loss', 'Student / base\nmatching', 24],
  ['factual-example-label', 'factual', 24],
  ['event-fields', 'Source: ball\nReceiver: water\nFootprint: splash', 24],
];
for (const [name, text, size] of edits) {
  const shape = slide.shapes.getItem(name);
  if (!shape) throw new Error('Missing approved shape: ' + name);
  shape.text = text;
  shape.text.style = { typeface: 'Arial', fontSize: size, autoFit: 'none', wrap: 'none' };
}
// These small changes fit enlarged labels into the approved layout.
slide.shapes.getItem('frozen-teacher').position = { left: 251, top: 486, width: 177, height: 56 };
slide.shapes.getItem('alternation-label').delete();
slide.shapes.getItem('per-arm-update').position = { left: 692, top: 514, width: 225, height: 28 };
slide.shapes.getItem('target-proxy').position = { left: 933, top: 199, width: 316, height: 32 };
slide.shapes.getItem('same-noun-task').position = { left: 1005, top: 273, width: 247, height: 34 };

// Restore the original vector Lucide paths in place of their raster previews.
// No icon paths are invented or edited. PowerPoint keeps these as SVG assets.
const iconPositions = new Map([
  ['49,123', 'file-text'], ['390,111', 'text-cursor-input'],
  ['390,230', 'replace'], ['938,123', 'clapperboard'], ['63,412', 'layers'],
  ['271,449', 'bot'], ['309,449', 'lock-keyhole'], ['698,459', 'target'],
  ['947,452', 'repeat-2'], ['1020,452', 'video'],
]);
const imageRows = snapshot.ndjson.split('\n').filter(Boolean).map(x => JSON.parse(x)).filter(x => x.kind === 'image');
if (imageRows.length !== iconPositions.size) throw new Error('Approved icon inventory changed');
for (const row of imageRows) {
  const item = p.resolve(row.id);
  const name = iconPositions.get(row.bbox.slice(0, 2).join(','));
  if (!name) throw new Error('Unexpected image at ' + row.bbox);
  const iconPath = path.join(modules, 'lucide/dist/esm/icons', name + '.js');
  const { default: nodes } = await import(pathToFileURL(iconPath).href);
  const color = name === 'lock-keyhole' ? '#227BCB' : '#242526';
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="192" height="192" viewBox="0 0 24 24" fill="none" stroke="' + color + '" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round">' + nodes.map(([tag, attrs]) => '<' + tag + ' ' + Object.entries(attrs).map(([k,v]) => k+'="'+v+'"').join(' ') + '/>').join('') + '</svg>';
  const frame = item.frame;
  item.replace({ blob: Buffer.from(svg), contentType: 'image/svg+xml', alt: name, fit: 'contain' });
  item.frame = frame;
}
const originalNotes = slide.speakerNotes.text;
slide.speakerNotes.text = originalNotes.replace('Icons are embedded as replaceable PNG assets.', 'Icons use their original SVG paths.');
await (await PresentationFile.exportPptx(p)).save(output);
const preview = await slide.export({format:'png',scale:1.5});
await fs.writeFile(output + '.png', new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(output + '.layout.json', await (await slide.export({format:'layout'})).text());
console.log(JSON.stringify({output,slides:1,minimum_label_px:24,svg_icons:slide.images.items.length}));
