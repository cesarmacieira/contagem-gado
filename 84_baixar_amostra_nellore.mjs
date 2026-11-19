// Reproduz a amostra de inspecao pelo indice ZIP64 local, sem baixar o ZIP todo.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {inflateRawSync} from 'node:zlib';
import {createHash} from 'node:crypto';
const root=path.dirname(fileURLToPath(import.meta.url));
const dir=path.join(root,'data/fontes_pesquisa/nellore_inspecao');
const url='https://evertontetila.ws.ufgd.edu.br/Datasets/NelloreBeefCattleDataset.zip';
const entries=JSON.parse(await fs.readFile(path.join(dir,'indice.json'),'utf8'));
const samples=JSON.parse(await fs.readFile(path.join(dir,'amostras.json'),'utf8'));
export async function extract(e){
  let offset=e.offset;
  if(offset===0xffffffff){
    const extra=Buffer.from(e.extra,'hex');
    for(let q=0;q+4<=extra.length;){
      const id=extra.readUInt16LE(q),len=extra.readUInt16LE(q+2);
      if(id===1){
        let z=q+4;
        if(e.size===0xffffffff)z+=8;
        if(e.compressed===0xffffffff)z+=8;
        offset=Number(extra.readBigUInt64LE(z));break;
      }
      q+=4+len;
    }
    if(offset===0xffffffff)throw Error('Offset ZIP64 ausente');
  }
  const r=await fetch(url,{headers:{Range:`bytes=${offset}-${offset+e.compressed+1023}`},signal:AbortSignal.timeout(30000)});
  if(r.status!==206||!r.headers.get('content-range')?.startsWith(`bytes ${offset}-`))throw Error('Range invalido');
  const b=Buffer.from(await r.arrayBuffer());
  if(b.readUInt32LE(0)!==0x04034b50)throw Error('Cabecalho ZIP invalido');
  const start=30+b.readUInt16LE(26)+b.readUInt16LE(28);
  if(start+e.compressed>b.length)throw Error('Download incompleto');
  const body=b.subarray(start,start+e.compressed);
  const result=e.method===8?inflateRawSync(body):e.method===0?body:null;
  if(!result||result.length!==e.size)throw Error('Tamanho/metodo invalido');
  return result;
}
if(typeof process!=='undefined' && process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
for(const s of samples){
  const dest=path.join(dir,path.basename(s.arquivo));
  let current;try{current=await fs.readFile(dest);}catch{}
  if(!current||createHash('sha256').update(current).digest('hex')!==s.sha256){
    const e=entries.find(e=>e.name===s.origem);if(!e)throw Error('Entrada ausente');
    const b=await extract(e);
    if(createHash('sha256').update(b).digest('hex')!==s.sha256)throw Error('SHA divergente');
    await fs.writeFile(dest,b);
  }
  const label=await extract(entries.find(e=>e.name===s.rotulo));
  await fs.writeFile(dest.replace(/\.[^.]+$/,'.txt'),label);
}
console.log('Amostra de inspecao conferida; nenhuma incorporacao ao treino/produto.');
}
