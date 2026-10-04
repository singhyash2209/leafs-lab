// Reuse the packaged portable reader and its semantic fallback, with no CDN.
import {readFileSync,writeFileSync} from 'node:fs';
import {buildPortableArtifact} from './portable/build_portable_artifact.mjs';
const artifact=JSON.parse(readFileSync('outputs/artifact.json','utf8'));
const datasets=artifact.snapshot?.datasets;
if(artifact.surface!=='dashboard'||artifact.snapshot?.status!=='ready'||!datasets) throw Error('Invalid dashboard');
for(const spec of [...artifact.manifest.cards,...artifact.manifest.charts,...artifact.manifest.tables]) {
 if(!Array.isArray(datasets[spec.dataset])) throw Error('Missing dataset: '+spec.dataset);
}
const runtimeEncoded=readFileSync(new URL('./portable/reader.gz.base64',import.meta.url),'utf8').trim();
writeFileSync('outputs/dashboard.html',buildPortableArtifact(artifact,{runtimeEncoded}));
console.log('Built self-contained dashboard with packaged portable reader');
